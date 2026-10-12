"""Bounded, synthetic full-context native PPO regression; never saves model weights.

Run on node3/node4 via ``python -m sao_bridge.joint_memory_check --output runs/NAME``.
Uses original W0, four DP ranks, two singleton microbatches/rank, and a separate
colocated reference process. The critic shares the actor workers, as in MA-SAO.
Synthetic costs and rewards are deliberately not research-training results.
Original H0/K2 can be selected with ``--model-role h0
--critic-updates-per-round 2 --critic-warmup-rounds 1`` without reducing context.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback

os.environ.setdefault(
    "AREAL_CACHE_DIR",
    str(Path(__file__).resolve().parents[2] / "runs/joint_memory_cache_node3/areal"),
)

import torch
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.infra.platforms import current_platform

from .joint_critic import JointEpisodeCritic
from .joint_context import CONTEXT_TOKEN_BUDGET
from .terminal_actor import TerminalFSDPPPOActor

LENGTH = 32768
ACTIONS = 4096
BATCH = 8
DP = 4


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def memory() -> dict:
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free, "device_total_bytes": total,
    }


def tensor_summary(value: torch.Tensor) -> dict:
    cpu = value.detach().float().cpu().contiguous()
    return {
        "shape": list(cpu.shape), "finite": bool(torch.isfinite(cpu).all()),
        "min": float(cpu.min()), "max": float(cpu.max()),
        "mean": float(cpu.mean()),
        "sha256_float32": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        "first_16": cpu.reshape(-1)[:16].tolist(),
    }


def synthetic_batch(eos_id: int) -> list[dict]:
    generator = torch.Generator().manual_seed(20261003)
    rows = []
    for index in range(BATCH):
        ids = torch.randint(100, 30000, (1, LENGTH), generator=generator)
        ids[0, -1] = eos_id
        mask = torch.zeros_like(ids, dtype=torch.bool)
        # Thirty-two turns, each with 128 action tokens and 896 context tokens.
        mask.reshape(32, 1024)[:, -128:] = True
        context = torch.zeros_like(ids)
        context[0, :CONTEXT_TOKEN_BUDGET] = torch.arange(CONTEXT_TOKEN_BUDGET) % 1000 + 101
        rows.append({
            "input_ids": ids,
            "attention_mask": torch.ones_like(mask),
            "loss_mask": mask,
            "logprobs": torch.zeros_like(ids, dtype=torch.float32),
            "versions": torch.where(mask, 0, -1),
            "rewards": torch.tensor([float(index % 2)]),
            "joint_context_ids": context,
        })
    return rows


class AuditMixin:
    def _save_initial_outputs(self, name: str, values: list[torch.Tensor]) -> None:
        path = Path(os.environ["JOINT_MEMORY_OUTPUT"]) / f"{name}-rank{self.rank}.pt"
        if not path.exists():
            torch.save([value.detach().cpu() for value in values], path)

    def _record(self, event: str, **fields) -> None:
        role = "critic" if self.config.is_critic else (
            "ref" if self.config.optimizer is None else "actor"
        )
        path = Path(os.environ["JOINT_MEMORY_OUTPUT"]) / f"{role}-rank{self.rank}.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({
                "event": event, "time": time.time(), "pid": os.getpid(),
                "rank": self.rank, "role": role, **memory(), **fields,
            }, allow_nan=False) + "\n")

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._record("initialized", model=self.config.path, config=asdict(self.config))

    def optimizer_step(self):
        snapshots = []
        for name, parameter in self.model.named_parameters():
            if parameter.grad is not None:
                local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
                snapshots.append((name, local, local.detach().reshape(-1)[:16].clone()))
        stats = super().optimizer_step()
        changed = sum(
            not torch.equal(before, parameter.detach().reshape(-1)[:16])
            for _, parameter, before in snapshots
        )
        self._record("optimizer_step", **stats, sampled_changed_tensors=changed)
        if not (stats["update_successful"] and math.isfinite(stats["grad_norm"])
                and stats["grad_norm"] > 0 and changed > 0):
            raise AssertionError(f"No verified finite nonzero optimizer update: {stats}")
        return stats


class MemoryActor(AuditMixin, TerminalFSDPPPOActor):
    def clear_batches(self, *args, **kwargs):
        self._record("before_clear_batches")
        result = super().clear_batches(*args, **kwargs)
        self._record("after_clear_batches")
        return result

    @torch.no_grad()
    def compute_logp(self, data):
        torch.cuda.reset_peak_memory_stats()
        if self.config.optimizer is None and not getattr(self, "_compared", False):
            baseline = FSDPPPOActor.compute_logp(self, data)
            expected = [value.detach().cpu().clone() for value in baseline]
            self._record("reference_native_before_release",
                         outputs=[tensor_summary(value) for value in baseline])
            del baseline
            current_platform.empty_cache()
            self._record("reference_explicit_release")
            result = super().compute_logp(data)
            equal = all(torch.equal(a, b.detach().cpu()) for a, b in zip(expected, result))
            self._record("reference_output_invariance", bitwise_equal=equal,
                         outputs=[tensor_summary(value) for value in result])
            if not equal:
                raise AssertionError("Reference outputs changed across allocator release")
            self._compared = True
        else:
            result = super().compute_logp(data)
        self._record("compute_logp", lengths=[int(row["attention_mask"].sum()) for row in data],
                     outputs=[tensor_summary(value) for value in result])
        role = "reference" if self.config.optimizer is None else "proximal"
        self._save_initial_outputs(role, result)
        return result

    @torch.no_grad()
    def compute_advantages(self, data):
        # Synthetic behavior is precisely the live actor, not invented log-probabilities.
        for row in data:
            row["logprobs"] = torch.roll(row["prox_logp"], shifts=1, dims=-1)
        return super().compute_advantages(data)

    def ppo_update(self, data):
        torch.cuda.reset_peak_memory_stats()
        self._record("ppo_begin", lengths=[int(row["attention_mask"].sum()) for row in data],
                     action_tokens=[int(row["loss_mask"].sum()) for row in data])
        super().ppo_update(data)
        self._record("ppo_end")


class MemoryCritic(AuditMixin, JointEpisodeCritic):
    @torch.no_grad()
    def compute_values(self, data):
        result = super().compute_values(data)
        self._save_initial_outputs("values", result)
        self._record(
            "critic_input_contract", settings=self.settings,
            prefix_tokens=[int(row["joint_context_ids"].gt(0).sum())
                           if self.cooperative_context else 0 for row in data],
            output_lengths=[value.shape[-1] for value in result],
        )
        return result

    def ppo_update(self, data):
        torch.cuda.reset_peak_memory_stats()
        self._record("ppo_begin")
        super().ppo_update(data)
        self._record("ppo_end", value_norm_updates=self.value_norm.updates
                     if self.value_norm is not None else None)


def make_config(repo: Path, output: Path, *, model_role: str = "w0", rounds: int = 2,
                reference_storage_dtype: str | None = None,
                critic_cooperative_context: bool = True, value_normalization: bool = True):
    import yaml
    from omegaconf import OmegaConf
    from areal.api.cli_args import PPOConfig, to_structured_cfg

    if type(critic_cooperative_context) is not bool or type(value_normalization) is not bool:
        raise ValueError("Native probe critic switches must be booleans")
    source = repo / "runs/alfworld_joint_async_6h_v5/solver/optimizer/trainer.yaml"
    raw = yaml.safe_load(source.read_text())
    raw.update(experiment_name=output.name, trial_name="memory",
               total_train_epochs=rounds, total_train_steps=rounds)
    raw["cluster"].update(fileroot=str(output / "engine"), n_gpus_per_node=4)
    raw["cluster"]["name_resolve"]["nfs_record_root"] = str(output / "name_resolve")
    environment = {
        "NCCL_IB_DISABLE": "1", "OMP_NUM_THREADS": "1",
        "GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo",
        "TMPDIR": "/dev/shm", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "AREAL_HOST_IP": "127.0.0.1", "MA_SAO_JOINT_EPISODES": "1",
        "MA_SAO_VALUE_NORM": str(int(value_normalization)), "JOINT_MEMORY_OUTPUT": str(output),
        "MA_SAO_COOPERATIVE_CONTEXT": str(int(critic_cooperative_context)),
    }
    cache = output / "runtime-cache"
    environment.update({
        "AREAL_CACHE_DIR": str(cache / "areal"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torch"),
        "PYTORCH_KERNEL_CACHE_PATH": str(cache / "kernels"),
        "VLLM_CACHE_ROOT": str(cache / "vllm"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    model_name = {"w0": "target-agent-sft", "h0": "harness-r1"}[model_role]
    for name in ("actor", "critic", "ref"):
        raw[name]["path"] = str(repo / "assets/Harness-R1" / model_name)
        raw[name]["experiment_name"] = output.name
        raw[name]["trial_name"] = "memory"
    raw["actor"]["scheduling_spec"][0]["env_vars"] = environment
    raw["ref"]["scheduling_spec"][0]["env_vars"] = environment
    if reference_storage_dtype is not None:
        if reference_storage_dtype not in ("bfloat16", "float32"):
            raise ValueError("Invalid reference storage dtype")
        raw["ref"]["optimizer_dtype"] = reference_storage_dtype
    # Keep native colocation/fork semantics, full context, clipping and PPO loss.
    config = OmegaConf.to_object(to_structured_cfg(OmegaConf.create(raw), PPOConfig))
    write_json(output / "config.json", asdict(config))
    return config


def summarize_audit(output: Path) -> dict:
    scope = json.loads((output / "scope.json").read_text())
    actor_steps = scope["actor_steps"]
    warmup_rounds = scope.get("critic_only_warmup_rounds", 0)
    critic_steps = (actor_steps + warmup_rounds) * scope["critic_steps_per_actor_step"]
    audit = {"status": "passed", "ranks": [], "synthetic_not_research_training": True}
    for rank in range(DP):
        records = {
            role: [json.loads(line) for line in
                   (output / f"{role}-rank{rank}.jsonl").read_text().splitlines()]
            for role in ("actor", "critic", "ref")
        }
        assert records["actor"][0]["pid"] == records["critic"][0]["pid"]
        assert records["actor"][0]["pid"] != records["ref"][0]["pid"]
        details = {"rank": rank}
        for role in ("actor", "critic"):
            updates = [row for row in records[role] if row["event"] == "optimizer_step"]
            expected = actor_steps if role == "actor" else critic_steps
            assert len(updates) == expected, f"{role} rank {rank} did not finish {expected} updates"
            assert all(row["update_successful"] and math.isfinite(row["grad_norm"])
                       and row["grad_norm"] > 0 and row["sampled_changed_tensors"] > 0
                       for row in updates)
            details[role] = {
                "grad_norms": [row["grad_norm"] for row in updates],
                "changed_tensors": [row["sampled_changed_tensors"] for row in updates],
                "step_peak_allocated_gib": [row["peak_allocated_bytes"] / 2**30 for row in updates],
                "step_peak_reserved_gib": [row["peak_reserved_bytes"] / 2**30 for row in updates],
            }
        starts = [row for row in records["actor"] if row["event"] == "ppo_begin"]
        assert len(starts) == actor_steps
        assert all(row["lengths"] == [LENGTH, LENGTH]
                   and row["action_tokens"] == [ACTIONS, ACTIONS] for row in starts)
        ref = {row["event"]: row for row in records["ref"]}
        assert ref["reference_output_invariance"]["bitwise_equal"]
        details["reference"] = {
            "bitwise_equal": True,
            "native_reserved_gib": ref["reference_native_before_release"]["reserved_bytes"] / 2**30,
            "fixed_reserved_gib": ref["reference_output_invariance"]["reserved_bytes"] / 2**30,
            "outputs": ref["reference_output_invariance"]["outputs"],
        }
        audit["ranks"].append(details)
        if "component_ablation" in scope:
            expected_settings = {
                "cooperative_context": scope["component_ablation"] != "without_context",
                "value_normalization": scope["component_ablation"] != "without_value_norm",
            }
            contracts = [row for row in records["critic"] if row["event"] == "critic_input_contract"]
            assert len(contracts) == actor_steps + warmup_rounds
            assert all(row["settings"] == expected_settings for row in contracts)
            assert all(row["prefix_tokens"] == [scope["critic_context_prefix"]] * 2
                       and row["output_lengths"] == [LENGTH] * 2 for row in contracts)
            updates = [row for row in records["critic"] if row["event"] == "ppo_end"]
            assert [row["value_norm_updates"] for row in updates] == (
                list(range(1, critic_steps + 1)) if expected_settings["value_normalization"]
                else [None] * critic_steps)
    after = json.loads((output / "gpu-after.json").read_text())
    from .resources import research_resources_released
    assert research_resources_released(after)
    audit["gpu_after"] = after
    return audit


def run(
    output: Path, *, model_role: str = "w0", actor_rounds: int = 2,
    critic_updates_per_round: int = 1, critic_warmup_rounds: int = 0,
    reference_storage_dtype: str | None = None,
    verify_optimizer_handoff: bool = False,
    component_ablation: str = "full",
) -> None:
    from areal.api import FinetuneSpec
    from areal.infra.scheduler.local import LocalScheduler
    from .resources import gpu_release_audit, research_resources_released

    if socket.gethostname() not in ("dev-NDH200-025", "dev-NDH200-026"):
        raise RuntimeError("This bounded regression is authorized on node3/node4 only")
    if (model_role not in {"h0", "w0"} or actor_rounds < 2
            or critic_updates_per_round < 1 or critic_warmup_rounds < 0):
        raise ValueError("Require original H0/W0, at least two actor rounds, K>=1 and warmup>=0")
    if component_ablation not in ("full", "without_context", "without_value_norm"):
        raise ValueError("Native critic probe requires one isolated context or ValueNorm removal")
    context_enabled = component_ablation != "without_context"
    normalized = component_ablation != "without_value_norm"
    rounds = actor_rounds + critic_warmup_rounds
    output.mkdir(parents=True, exist_ok=False)
    before = gpu_release_audit()
    write_json(output / "gpu-before.json", before)
    if not research_resources_released(before):
        raise RuntimeError(f"Research node ownership check failed: {before}")
    os.environ.update(JOINT_MEMORY_OUTPUT=str(output), MA_SAO_JOINT_EPISODES="1",
                      MA_SAO_VALUE_NORM=str(int(normalized)),
                      MA_SAO_COOPERATIVE_CONTEXT=str(int(context_enabled)),
                      AREAL_HOST_IP="127.0.0.1")
    os.environ.pop("MA_SAO_ASYNC_JOINT_ROOT", None)
    os.environ.setdefault("SAO_LOCAL_ADMIN_TOKEN", "synthetic-memory-check-not-a-server")
    repo = Path(__file__).resolve().parents[2]
    config = make_config(repo, output, model_role=model_role, rounds=rounds,
                         reference_storage_dtype=reference_storage_dtype,
                         critic_cooperative_context=context_enabled, value_normalization=normalized)
    write_json(output / "runtime_interventions.json", {
        "reference_storage_dtype": config.ref.optimizer_dtype,
        "actor_storage_dtype": config.actor.optimizer_dtype,
        "critic_storage_dtype": config.critic.optimizer_dtype,
        "compute_dtype": config.actor.dtype,
        "reference_has_optimizer": config.ref.optimizer is not None,
    })
    write_json(output / "scope.json", {
        "kind": "synthetic_not_research_training",
        "input_source": "synthetic; no saved rollout tensors found in failed optimizer run",
        "model": config.actor.path, "length": LENGTH, "action_tokens": ACTIONS,
        "batch": BATCH, "data_parallel_ranks": DP, "per_rank_trajectories": 2,
        "microbatches_per_rank": 2, "actor_steps": actor_rounds,
        "critic_steps_per_actor_step": critic_updates_per_round,
        "critic_only_warmup_rounds": critic_warmup_rounds, "model_role": model_role,
        "component_ablation": component_ablation,
        "critic_context_prefix": CONTEXT_TOKEN_BUDGET if context_enabled else 0,
        "critic_full_length": LENGTH + (CONTEXT_TOKEN_BUDGET if context_enabled else 0),
        "reference_offload": False, "checkpoint_writes": verify_optimizer_handoff,
    })
    for directory in (output / "engine", output / "name_resolve"):
        directory.mkdir()
    scheduler = None
    try:
        scheduler = LocalScheduler(exp_config=config, gpu_devices=list(range(DP)))
        actor = MemoryActor.as_controller(config.actor, scheduler)
        critic = MemoryCritic.as_controller(config.critic, scheduler)
        ref = MemoryActor.as_controller(config.ref, scheduler)
        spec = FinetuneSpec(total_train_epochs=rounds, dataset_size=BATCH, train_batch_size=BATCH)
        for role, engine in (("actor", actor), ("critic", critic), ("ref", ref)):
            engine.create_process_group()
            engine.initialize(role=role, addr=None, ft_spec=spec)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.actor.path, trust_remote_code=True)
        for step in range(rounds):
            rows = synthetic_batch(tokenizer.eos_token_id)
            values = critic.compute_values(rows)
            reference = ref.compute_logp(rows)
            proximal = actor.compute_logp(rows)
            for row, value, ref_logp, prox_logp in zip(rows, values, reference, proximal):
                row.update(values=value, ref_logp=ref_logp, prox_logp=prox_logp)
            advantages = actor.compute_advantages(rows)
            if step >= critic_warmup_rounds:
                actor.ppo_update(advantages)
            for _ in range(critic_updates_per_round):
                critic.ppo_update(advantages)
            if step >= critic_warmup_rounds:
                actor.step_lr_scheduler()
            critic.step_lr_scheduler()
            # Release only RPC tensors owned by this test iteration.
            actor.clear_batches(*rows, *advantages)
            critic.clear_batches(*rows, *advantages)
            ref.clear_batches(*rows, *advantages)
            write_json(output / f"step-{step}.json", {
                "step": step, "complete": True,
                "critic_only_warmup": step < critic_warmup_rounds,
                "critic_optimizer_steps": critic_updates_per_round,
            })
        if verify_optimizer_handoff:
            from areal.api import SaveLoadMeta
            for name, engine in (("actor", actor), ("critic", critic)):
                meta = SaveLoadMeta(path=str(output / "handoff" / name),
                                    weight_format="dcp", with_optim=True)
                engine.save(meta=meta)
                engine.load(meta=meta)
            write_json(output / "HANDOFF_PASSED.json", {
                "status": "passed", "rank_fingerprints_checked": 8,
                "native_dcp_save_and_load": True,
            })
    except BaseException:
        write_json(output / "result.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        if scheduler is not None:
            scheduler.delete_workers(reverse_order=True)
        for _ in range(15):
            after = gpu_release_audit()
            if not after["foreign_gpu_pids"]:
                break
            time.sleep(1)
        write_json(output / "gpu-after.json", after)
        if not research_resources_released(after):
            write_json(output / "result.json", {
                "status": "failed", "reason": "GPU release/filler audit failed", "audit": after,
            })
            raise RuntimeError(f"GPU release/filler check failed: {after}")
    write_json(output / "audit-summary.json", summarize_audit(output))
    write_json(output / "result.json", {
        "status": "passed", "actor_steps": actor_rounds,
        "critic_steps": rounds * critic_updates_per_round,
        "critic_updates_per_round": critic_updates_per_round,
        "critic_only_warmup_rounds": critic_warmup_rounds,
        "node": socket.gethostname(), "gpu_release_verified": True,
        "node3_released": socket.gethostname() == "dev-NDH200-025", "filler_preserved": True,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-role", choices=("w0", "h0"), default="w0")
    parser.add_argument("--actor-rounds", type=int, default=2)
    parser.add_argument("--critic-updates-per-round", type=int, default=1)
    parser.add_argument("--critic-warmup-rounds", type=int, default=0)
    parser.add_argument("--reference-storage-dtype", choices=("bfloat16", "float32"))
    parser.add_argument("--verify-optimizer-handoff", action="store_true")
    parser.add_argument("--component-ablation", choices=("full", "without_context", "without_value_norm"),
                        default="full")
    args = parser.parse_args()
    run(
        args.output.resolve(), model_role=args.model_role, actor_rounds=args.actor_rounds,
        critic_updates_per_round=args.critic_updates_per_round,
        critic_warmup_rounds=args.critic_warmup_rounds,
        reference_storage_dtype=args.reference_storage_dtype,
        verify_optimizer_handoff=args.verify_optimizer_handoff,
        component_ablation=args.component_ablation,
    )


if __name__ == "__main__":
    # RPC workers must import engine classes by module name, never ``__main__``.
    from sao_bridge.joint_memory_check import main as module_main
    module_main()
