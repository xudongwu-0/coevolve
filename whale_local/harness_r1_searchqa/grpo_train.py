"""Distributed full-parameter online Harness-R1 GRPO (FSDP+vLLM adaptation).

Run with torchrun on >=4 training GPUs and separate GPUs for vLLM. The actor
and fixed reference originate from a fresh local Engineer SFT checkpoint.
No real training is started by --help or the CPU objective tests.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import functools
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

from .workflow import (BACKEND, UPSTREAM_COMMIT, atomic_json, checkpoint_identity,
                       sha256, validate_rollouts)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", type=Path, required=True, help="fresh SFT HF directory")
    result.add_argument("--data", type=Path, required=True, help="train failure-packet JSONL")
    result.add_argument("--reward-config", type=Path, required=True)
    result.add_argument("--config", type=Path, default=Path("configs/searchqa_engineer_grpo.json"))
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--updates", type=int, required=True, help="online rollout/update count")
    result.add_argument("--rollout-gpus", required=True, help="physical GPU ids, disjoint from actor")
    result.add_argument("--resume", type=Path, help="committed checkpoint-NNNNNN directory")
    result.add_argument("--seed", type=int, default=42)
    return result


def validate_config(config):
    expected = {
        "backend": BACKEND, "upstream_commit": UPSTREAM_COMMIT,
        "responses_per_prompt": 8, "packets_per_rollout": 4,
        "samples_per_rollout": 32, "optimizer_iterations": 4,
        "temperature": 0.7, "top_p": 0.95, "learning_rate": 1e-6,
        "weight_decay": 0.1, "adam_betas": [0.9, 0.98],
        "clip_low": 0.2, "clip_high": 0.28, "tis_clip": 2.0,
        "rollout_gdn_prefill_backend": "triton",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"{key} must match pinned Harness-R1: {value!r}")
    if config["kl_coef"] < 0 or config["kl_type"] not in {"k1", "low_var_kl"}:
        raise ValueError("invalid optional KL configuration")
    if (config["max_prompt_tokens"] + config["max_response_tokens"]
            > config["max_context_tokens"]):
        raise ValueError("prompt+response budget exceeds context")
    for key in ("rollout_tp", "max_response_tokens", "max_prompt_tokens", "rollout_timeout"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    return config


def packet_batch(rows, update, seed):
    if len(rows) < 4:
        raise ValueError("at least four train packets required")
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    return [rows[order[(4 * update + i) % len(rows)]] for i in range(4)]


def _logprobs(model, record, device, temperature):
    import torch
    from .grpo_objective import response_logprobs
    generated = record["response_ids"]
    # Last prompt token predicts the first response token; no prefill loss.
    inputs = torch.tensor([record["prompt_ids"] + generated[:-1]], device=device)
    targets = torch.tensor(generated, device=device)
    logits = model(input_ids=inputs, use_cache=False,
                   logits_to_keep=len(generated)).logits[0]
    if logits.shape[0] != len(generated):
        raise RuntimeError("model does not support response-only logits_to_keep")
    return response_logprobs(logits, targets, temperature)


def train_update(model, optimizer, records, config, device, *, rank=0, world=1,
                 reference=None):
    """Four minibatches of eight, not four epochs over the 32 responses."""
    import torch
    from .grpo_objective import group_advantages, grpo_loss
    if len(records) != 32 or 8 % world:
        raise ValueError("32 samples and a world size dividing eight required")
    rewards = torch.tensor([r["reward"] for r in records], device=device)
    advantages, constant = group_advantages(rewards)
    local_indices = [i for i in range(32) if i % world == rank]
    model.eval()  # Dropout must not change old/current probability definitions.
    with torch.no_grad():
        old = {i: _logprobs(model, records[i], device, config["temperature"]).cpu()
               for i in local_indices}
        ref = ({i: _logprobs(reference, records[i], device, config["temperature"]).cpu()
                for i in local_indices} if reference is not None else {})
    metrics = []
    for iteration in range(4):
        optimizer.zero_grad(set_to_none=True)
        selected = [i for i in local_indices if i // 8 == iteration]
        total_loss = 0.0
        for i in selected:
            new = _logprobs(model, records[i], device, config["temperature"])
            rollout = torch.tensor(records[i]["rollout_logprobs"], device=device)
            loss, _ = grpo_loss(
                new, old[i].to(device), rollout, advantages[i],
                reference_logprobs=ref[i].to(device) if ref else None,
                clip_low=config["clip_low"], clip_high=config["clip_high"],
                tis_clip=config["tis_clip"], kl_coef=config["kl_coef"],
                kl_type=config["kl_type"])
            (loss / len(selected)).backward()
            total_loss += float(loss.detach()) / len(selected)
        if hasattr(model, "clip_grad_norm_"):
            norm = model.clip_grad_norm_(config["max_grad_norm"])
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite global gradient norm")
        optimizer.step()
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("nonfinite actor parameter after Adam update")
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError("nonfinite Adam optimizer state")
        metrics.append({"iteration": iteration, "loss": total_loss, "grad_norm": float(norm)})
    return {"optimizer_steps": len(metrics), "constant_reward_groups": constant.cpu().tolist(),
            "all_groups_constant": bool(constant.all()), "iterations": metrics,
            "zero_advantage_policy": "retain samples and execute four AdamW steps; "
                                     "zero policy gradient, optimizer momentum and weight decay remain active"}


def _rank_zero(action, rank):
    """Broadcast infrastructure failures before other ranks enter a collective."""
    import torch.distributed as dist
    message = [None]
    if rank == 0:
        try:
            message[0] = {"ok": True, "value": action()}
        except Exception as exc:
            message[0] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(message, src=0)
    if not message[0]["ok"]:
        raise RuntimeError(message[0]["error"])
    return message[0]["value"]


def validate_engineer_scale(model_type: str, parameters: int, *, allow_4b: bool = False) -> None:
    if model_type == "qwen3_5_text" and parameters >= 8_000_000_000:
        return
    if allow_4b and model_type == "qwen3" and 3_500_000_000 <= parameters <= 4_500_000_000:
        return
    raise ValueError("Expected full 9B text Engineer or explicitly enabled native 4B pilot")


def _load_fsdp(path, device, *, frozen=False, allow_4b=False):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    config = AutoConfig.from_pretrained(str(path), local_files_only=True)
    if config.model_type != "qwen3_5_text" and not (allow_4b and config.model_type == "qwen3"):
        raise ValueError("expected fresh text-only Qwen3.5 Engineer SFT, not the untouched VLM base")
    # Load on CPU; auto-wrap moves/shards one decoder at a time. Do not put the
    # entire FP32 9B model and Adam on a single GPU before sharding.
    model, loading = AutoModelForCausalLM.from_pretrained(
        str(path), dtype=torch.float32, local_files_only=True, attn_implementation="sdpa",
        output_loading_info=True)
    errors = {key: loading.get(key) for key in
              ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
              if loading.get(key)}
    if errors:
        raise ValueError(f"Engineer checkpoint must load strictly without remapping/random weights: {errors}")
    model.config.use_cache = False
    validate_engineer_scale(config.model_type, sum(p.numel() for p in model.parameters()),
                            allow_4b=allow_4b)
    model.requires_grad_(not frozen)
    names = set(model._no_split_modules or ())
    classes = {type(module) for module in model.modules()
               if type(module).__name__ in names}
    if not classes:
        raise ValueError("no decoder auto-wrap classes; refusing root-only FSDP")
    wrapped = FSDP(
        model, device_id=device, use_orig_params=True,
        auto_wrap_policy=functools.partial(transformer_auto_wrap_policy,
                                          transformer_layer_cls=classes),
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.float32,
                                       buffer_dtype=torch.float32),
        limit_all_gathers=True)
    if not frozen:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper,
        )
        # Unlike HF's training-only switch, this wrapper also checkpoints the
        # deterministic eval-mode GRPO forwards that require gradients.
        apply_activation_checkpointing(
            wrapped, checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: type(module) in classes)
    wrapped.eval()
    return wrapped


def _export_actor(model, tokenizer, path, rank):
    import torch
    from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                       FullStateDictConfig, StateDictType)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                             FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        state = model.state_dict()

    def save():
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
        path.mkdir(parents=True)
        state_bf16 = {key: value.to(torch.bfloat16) if value.is_floating_point() else value
                      for key, value in state.items()}
        model.module.save_pretrained(str(path), state_dict=state_bf16,
                                     safe_serialization=True, max_shard_size="4GB")
        tokenizer.save_pretrained(str(path))
        identity = checkpoint_identity(path)
        atomic_json(path / "actor_identity.json", identity)
        return identity

    return _rank_zero(save, rank)


def rollout_environment(gpus):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus,
                       HARNESS_R1_REGISTER_VLLM_TEXT="1", VLLM_PLUGINS="")
    distributed_keys = {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK", "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT",
    }
    for key in tuple(environment):
        # An independent vLLM job must create its own TCPStore, not join torchrun's.
        if key in distributed_keys or key.startswith("TORCHELASTIC_"):
            environment.pop(key)
    return environment


def _generate(args, config, reward_config, rows, actor, actor_sha, update, *,
              verify_only=False):
    job_path = args.output / f"rollout-job-{update:06d}.json"
    result_path = args.output / f"rollout-{update:06d}.json"
    atomic_json(job_path, {
        "config": config, "reward_config": reward_config, "rows": rows,
        "actor": str(actor.resolve()), "actor_sha256": actor_sha,
        "result": str(result_path.resolve()), "seed": args.seed + update,
        "verify_only": verify_only,
    })
    environment = rollout_environment(args.rollout_gpus)
    process = subprocess.Popen(
        [sys.executable, "-m", "harness_r1_searchqa.workflow", "--rollout-job", str(job_path)],
        env=environment, start_new_session=True)
    try:
        returncode = process.wait(timeout=config["rollout_timeout"])
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise RuntimeError("vLLM/reward worker timed out; terminated its process group") from None
    if returncode:
        raise subprocess.CalledProcessError(returncode, process.args)
    result = json.loads(result_path.read_text())
    if result["actor_sha256"] != actor_sha or not result["engine_reloaded_from_disk"]:
        raise RuntimeError("rollout engine failed disk actor synchronization")
    if verify_only:
        return None
    validate_rollouts(result["records"], rows, actor_sha)
    return result["records"]


def main():
    args = parser().parse_args()
    import torch
    import torch.distributed as dist
    import yaml
    from transformers import AutoTokenizer

    config = validate_config(json.loads(args.config.read_text()))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in {4, 8} or not torch.cuda.is_available():
        raise RuntimeError("full 9B Adam requires torchrun on four or eight training GPUs")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise ValueError("set explicit CUDA_VISIBLE_DEVICES for disjoint training/rollout GPUs")
    training_gpus = visible.split(",")
    rollout_gpus = args.rollout_gpus.split(",")
    if len(training_gpus) < world or len(rollout_gpus) != config["rollout_tp"]:
        raise ValueError("GPU topology does not match training world and rollout TP")
    if set(training_gpus) & set(rollout_gpus) or len(set(rollout_gpus)) != len(rollout_gpus):
        raise ValueError("actor and rollout GPUs must be distinct")
    if args.updates <= 0:
        raise ValueError("--updates must be positive")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=60))
    torch.manual_seed(args.seed)

    def preflight():
        if args.output.exists() and any(args.output.iterdir()) and args.resume is None:
            raise FileExistsError("output is nonempty; use --resume or a fresh output")
        args.output.mkdir(parents=True, exist_ok=True)
        rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
        packet_batch(rows, 0, args.seed)
        reward_config = yaml.safe_load(args.reward_config.read_text())
        if reward_config.get("harness_r1_response_protocol") != "prefill_think_patch":
            raise ValueError("reward config must use prefill_think_patch")
        from .reward import _validate_metadata, _rows_by_id
        from types import SimpleNamespace
        reward_args = SimpleNamespace(**reward_config)
        train = _rows_by_id(reward_args.harness_r1_searchqa_data)
        for row in rows:
            ids, *_ = _validate_metadata(reward_args, row.get("metadata"))
            if any(task not in train for task in ids):
                raise ValueError("failure packet contains non-train task identities")
        source_root = Path(__file__).resolve().parent
        provenance = {
            "backend": BACKEND, "upstream_commit": UPSTREAM_COMMIT,
            "sft_checkpoint": checkpoint_identity(args.model),
            "data_sha256": sha256(args.data), "reward_config_sha256": sha256(args.reward_config),
            "searchqa_manifest_sha256": sha256(Path(reward_args.harness_r1_searchqa_data) / "manifest.json"),
            "config": config, "world_size": world, "seed": args.seed,
            "source_sha256": {name: sha256(source_root / name)
                              for name in ("grpo_train.py", "grpo_objective.py", "upstream_ppo.py",
                                           "workflow.py", "reward.py")},
            "upstream_source_sha256": {
                name: sha256(Path(__file__).resolve().parents[2] /
                             "third_party/harness-r1-upstream" / name)
                for name in ("scripts/train_engineer_rl.sh",
                             "code/Relax/relax/utils/utils.py",
                             "code/Relax/relax/utils/training/ppo_utils.py",
                             "code/Relax/relax/backends/megatron/loss.py")},
        }
        start = 0
        if args.resume:
            saved = json.loads((args.resume / "COMMITTED.json").read_text())
            if saved["provenance"] != provenance:
                raise ValueError("resume source/data/SFT/config/world identity mismatch")
            if checkpoint_identity(args.resume / "actor")["sha256"] != saved["actor_sha256"]:
                raise ValueError("resume actor identity mismatch")
            for i in range(world):
                if sha256(args.resume / f"rank-{i:04d}.pt") != saved["rank_state_sha256"][str(i)]:
                    raise ValueError("resume FP32 actor/optimizer shard identity mismatch")
            start = saved["updates_completed"]
        atomic_json(args.output / "provenance.json", provenance)
        return rows, reward_config, provenance, start

    rows, reward_config, provenance, start = _rank_zero(preflight, rank)
    if start >= args.updates:
        raise ValueError("--updates must exceed updates already completed")
    load_path = args.resume / "actor" if args.resume else args.model
    tokenizer = AutoTokenizer.from_pretrained(str(load_path), local_files_only=True)
    allow_4b = config.get("engineer_scale") == "4b_pilot"
    model = _load_fsdp(load_path, device, allow_4b=allow_4b)
    if (tokenizer.eos_token_id is None
            or model.module.config.eos_token_id != tokenizer.eos_token_id):
        raise ValueError("fresh SFT must export tokenizer im_end EOS into the text model config")
    reference = _load_fsdp(args.model, device, frozen=True, allow_4b=allow_4b) if config["kl_coef"] else None
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["learning_rate"], betas=tuple(config["adam_betas"]),
        eps=config["adam_eps"], weight_decay=config["weight_decay"], foreach=False)
    if args.resume:
        saved = torch.load(args.resume / f"rank-{rank:04d}.pt", map_location="cpu",
                           weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        # Actor disk export is BF16 for rollout. Restore exact FP32 shards for
        # optimizer continuation, instead of silently rounding master weights.
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            model.load_state_dict(saved["actor_local_state"])
        torch.set_rng_state(saved["rng_cpu"])
        torch.cuda.set_rng_state(saved["rng_cuda"], device)
    actor_identity = _rank_zero(lambda: checkpoint_identity(load_path), rank)
    records = _rank_zero(
        lambda: _generate(args, config, reward_config, packet_batch(rows, start, args.seed),
                          load_path, actor_identity["sha256"], start), rank)
    for update in range(start, args.updates):
        metrics = train_update(model, optimizer, records, config, device,
                               rank=rank, world=world, reference=reference)
        checkpoint_dir = args.output / f"checkpoint-{update + 1:06d}"
        actor_path = checkpoint_dir / "actor"
        identity = _export_actor(model, tokenizer, actor_path, rank)
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            local_state = model.state_dict()
        torch.save({"optimizer": optimizer.state_dict(), "actor_local_state": local_state,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(device)},
                   checkpoint_dir / f"rank-{rank:04d}.pt")
        all_metrics = [None] * world
        dist.all_gather_object(all_metrics, metrics)
        # Reload *after every update*, including the final update. A failed
        # reload leaves an uncommitted checkpoint and cannot advance progress.
        next_records = _rank_zero(
            lambda: _generate(args, config, reward_config,
                              packet_batch(rows, update + 1, args.seed),
                              actor_path, identity["sha256"], update + 1,
                              verify_only=update + 1 == args.updates), rank)

        def commit():
            source_root = Path(__file__).resolve().parent
            if (sha256(args.data) != provenance["data_sha256"]
                    or sha256(args.reward_config) != provenance["reward_config_sha256"]
                    or any(sha256(source_root / name) != digest
                           for name, digest in provenance["source_sha256"].items())):
                raise RuntimeError("training source/data changed; checkpoint cannot be committed")
            state = {
                "provenance": provenance, "updates_completed": update + 1,
                "optimizer_steps": (update + 1) * 4, "actor_sha256": identity["sha256"],
                "engine_reloaded_from_disk": True, "metrics_by_rank": all_metrics,
                "rank_state_sha256": {
                    str(i): sha256(checkpoint_dir / f"rank-{i:04d}.pt") for i in range(world)},
            }
            atomic_json(checkpoint_dir / "COMMITTED.json", state)
            atomic_json(args.output / "progress.json", state)
            print(json.dumps({"update": update + 1, "optimizer_steps": state["optimizer_steps"],
                              "actor": str(actor_path), "metrics": metrics}), flush=True)

        _rank_zero(commit, rank)
        records = next_records
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
