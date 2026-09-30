"""Configure and call the upstream SAO-patched AReaL PPOTrainer, without new losses."""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack, contextmanager
import getpass
import os
from pathlib import Path
import secrets

from .common import (
    EDITOR_SYSTEM, INITIAL_SOURCE, SAO, read_json, verify_upstreams, write_json,
)
from .checkpoint import StepFrequency


def make_config(model: Path, run: Path, role: str, steps: int, batch_size: int) -> dict:
    import yaml
    if role not in {"editor", "solver"} or steps < 1 or batch_size < 1:
        raise ValueError("Invalid role, step count, or batch size")
    config = yaml.safe_load((SAO / "configs/phase2_hard_sao.yaml").read_text())
    config.update(
        experiment_name="coevolve-independent-sao", trial_name=role,
        total_train_steps=steps, total_train_epochs=steps, seed=42,
    )
    config["cluster"].update(n_nodes=1, n_gpus_per_node=2, fileroot=str(run / "engine"))
    config["cluster"]["name_resolve"]["nfs_record_root"] = str(run / "name_resolve")
    config["scheduler"]["type"] = "local"
    config["rollout"].update(
        backend="vllm:d1p1t1", max_concurrent_rollouts=batch_size,
        max_head_offpolicyness=1, dump_to_file=True,
        admin_api_key="${oc.env:SAO_LOCAL_ADMIN_TOKEN}",
    )
    config["rollout"]["agent"]["admin_api_key"] = "${oc.env:SAO_LOCAL_ADMIN_TOKEN}"
    config["gconfig"].update(max_new_tokens=512, max_tokens=2048, n_samples=1)
    config["actor"].update(
        backend="fsdp:d1p1t1", path=str(model), weight_update_mode="disk",
    )
    config["actor"]["mb_spec"]["max_tokens_per_mb"] = 2048
    config["actor"]["optimizer"]["lr"] = 5e-6
    config["actor"]["scheduling_spec"][0]["env_vars"] = {
        "NCCL_IB_DISABLE": "1", "OMP_NUM_THREADS": "4",
    }
    config["critic"]["optimizer"]["lr"] = 1e-5
    config["critic"]["mb_spec"]["max_tokens_per_mb"] = 2048
    config["critic"]["scheduling_strategy"]["fork"] = False
    config["ref"] = None
    config["vllm"].update(max_model_len=2048, max_num_seqs=batch_size,
                          gpu_memory_utilization=0.30, enforce_eager=True)
    config["train_dataset"].update(
        path=str(run / "data"), batch_size=batch_size, num_workers=0,
        drop_last=True, shuffle=False,
    )
    config["saver"].update(freq_steps=1, freq_secs=None, freq_epochs=None, mode="sync")
    config["recover"].update(mode="disabled", freq_steps=None, freq_secs=None)
    config["evaluator"].update(freq_steps=None, freq_secs=None, freq_epochs=None)
    config["stats_logger"]["wandb"]["mode"] = "disabled"
    config["reward"] = {"backend": "local", "on_error": "raise", "max_workers": 1}
    return copy.deepcopy(config)


@contextmanager
def upstream_trainer(config, dataset):
    from areal import PPOTrainer

    with ExitStack() as cleanup:
        class ResourceManagedTrainer(PPOTrainer):
            def _save_hf(self, epoch, epoch_step, global_step):
                self.saver.freq_ctl.begin(
                    epoch, epoch_step, global_step,
                    final=global_step + 1 == self.config.total_train_steps,
                )
                return super()._save_hf(epoch, epoch_step, global_step)

            def _init_scheduler(self):
                scheduler = super()._init_scheduler()
                # Upstream __exit__ cannot run if its constructor fails.
                cleanup.callback(scheduler.delete_workers, reverse_order=True)
                return scheduler

            def _create_critic(self, critic_config, allocation):
                if not os.environ.get("MA_SAO_CONTEXT_FILE"):
                    return super()._create_critic(critic_config, allocation)
                from .central_critic import JointStateCritic
                if allocation.backend != "fsdp":
                    raise ValueError("Joint-state critic currently requires FSDP")
                critic = JointStateCritic.as_controller(critic_config, self.scheduler)
                critic.create_process_group(parallel_strategy=allocation.parallel)
                return critic

        with ResourceManagedTrainer(
            config, train_dataset=dataset, valid_dataset=None,
        ) as trainer:
            trainer.saver.freq_ctl = StepFrequency(trainer.saver.freq_ctl)
            warmup = int(os.environ.get("MA_SAO_CRITIC_WARMUP_STEPS", "0"))
            if warmup:
                skip_actor_updates(trainer.actor, warmup)
            yield trainer


def skip_actor_updates(actor, steps: int) -> dict:
    """Critic-only warm-up: the first `steps` native actor updates are skipped.

    The upstream loop still computes values, advantages and SAO's K critic updates on
    the warm-up batches (recycled partner-evaluation episodes); the policy is untouched.
    """
    state = {"calls": 0, "skipped": 0}
    update, schedule = actor.ppo_update, actor.step_lr_scheduler

    def ppo_update(batch):
        state["calls"] += 1
        if state["calls"] <= steps:
            state["skipped"] += 1
            return None
        return update(batch)

    def step_lr_scheduler():
        if state["calls"] > steps:
            return schedule()

    actor.ppo_update, actor.step_lr_scheduler = ppo_update, step_lr_scheduler
    return state


def editor_data(batch_size: int, feedback: str) -> list[dict]:
    focuses = [
        "Prefer exact arithmetic and check the requested quantity.",
        "Avoid repetition and reserve tokens for a final answer.",
        "Use a concise plan, then check boundary cases.",
        "Continue incomplete derivations when a final boxed answer is missing.",
    ]
    return [{
        "id": f"editor-{i}",
        "messages": [
            {"role": "system", "content": EDITOR_SYSTEM},
            {"role": "user", "content": (
                f"Current harness:\n{INITIAL_SOURCE}\n"
                f"Development feedback:\n{feedback}\n"
                f"Consider this general direction: {focuses[i % len(focuses)]}"
            )},
        ],
    } for i in range(batch_size * 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", required=True, choices=["editor", "solver"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluator-url")
    parser.add_argument("--feedback", default="Some solutions are incorrect or unfinished.")
    args = parser.parse_args()
    provenance = verify_upstreams()
    if len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) != 2:
        parser.error("Select exactly two assigned GPUs with CUDA_VISIBLE_DEVICES")
    if args.role == "editor" and not args.evaluator_url:
        parser.error("--evaluator-url is required for editor training")
    os.environ["SAO_CRITIC_K"] = "2"
    os.environ.setdefault("SAO_LOCAL_ADMIN_TOKEN", secrets.token_urlsafe(32))
    from areal.api.cli_args import PPOConfig, load_expr_config
    from datasets import Dataset
    import yaml

    raw = make_config(args.model.resolve(), args.run.resolve(), args.role,
                      args.steps, args.batch_size)
    config_path = args.run / f"{args.role}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    config, _ = load_expr_config(["--config", str(config_path)], PPOConfig)
    if config.critic is None or config.gconfig.n_samples != 1:
        raise RuntimeError("Independent SAO requires a critic and one rollout per prompt")
    rs = config.actor.rejection_sampling
    if rs is None or (rs.level, rs.action, rs.lower, rs.upper) != ("token", "mask", 0.3, 5.0):
        raise RuntimeError("Published SAO DIS configuration was not preserved")
    rows = (
        editor_data(args.batch_size, args.feedback) if args.role == "editor"
        else read_json(args.run / "data/train.json")
    )
    if len(rows) < args.batch_size:
        raise ValueError("Dataset is smaller than the configured batch")
    with upstream_trainer(config, Dataset.from_list(rows)) as trainer:
        trainer.train(
            workflow="sao_bridge.workflow.CoevolveWorkflow",
            workflow_kwargs={
                "role": args.role, "tokenizer": str(args.model.resolve()),
                "gconfig": config.gconfig, "harness_path": str(args.harness.resolve()),
                "receipts_dir": str(args.run / "receipts" / args.role),
                "evaluator_url": args.evaluator_url,
                "total_tokens": 512,
            },
        )
    checkpoint_root = (
        args.run / "engine/checkpoints" / getpass.getuser()
        / config.experiment_name / args.role
    )
    paths = {}
    for name in ("actor", "critic"):
        found = list((checkpoint_root / name).glob(
            f"epoch*epochstep*globalstep{args.steps - 1}"
        ))
        if len(found) != 1 or not list(found[0].glob("*.safetensors")):
            raise RuntimeError(f"Missing final {name} HF checkpoint under {checkpoint_root}")
        paths[name] = str(found[0].resolve())
    write_json(args.run / f"{args.role}_trained.json", {
        "role": args.role, "checkpoints": paths, "upstream": provenance,
        "trainer": "SAO-patched areal.PPOTrainer", "actor_steps": args.steps,
        "critic_steps": 2 * args.steps, "model_initial": str(args.model.resolve()),
        "algorithm_reimplemented": False, "centralized_critic": False,
    })


if __name__ == "__main__":
    main()
