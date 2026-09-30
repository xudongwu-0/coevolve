"""One H or W phase using the unchanged upstream SAO optimization loop."""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import secrets
import time

from .common import read_json, verify_upstreams, write_json
from .train import make_config, upstream_trainer


class MetricsRecorder:
    def __init__(self, delegate, path: Path):
        self.delegate = delegate
        self.path = path

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def commit(self, epoch, step, global_step, data):
        self.delegate.commit(epoch, step, global_step, data)
        with self.path.open("a") as stream:
            stream.write(json.dumps({
                "epoch": epoch, "step": step, "global_step": global_step,
                "metrics": data,
            }, default=float) + "\n")


def configure(spec: dict, run: Path) -> dict:
    role = spec["role"]
    context_tokens = spec.get("context_tokens", 8192)
    if type(context_tokens) is not int or context_tokens <= spec["tokens"]:
        raise ValueError("Training context must leave room for prompts and output tokens")
    cfg = make_config(Path(spec["model"]), run, role, spec["steps"], spec["batch_size"])
    cfg["trial_name"] = spec["phase"]
    cfg["experiment_name"] = spec["experiment"]
    cfg["seed"] = spec["seed"]
    actor_gpus, rollout_gpus = spec.get("actor_gpus", 2), spec.get("rollout_gpus", 2)
    cfg["cluster"]["n_gpus_per_node"] = actor_gpus + rollout_gpus
    cfg["actor"]["backend"] = f"fsdp:d{actor_gpus}p1t1"
    cfg["actor"]["weight_update_mode"] = spec.get("weight_update_mode", "disk")
    cfg["rollout"]["backend"] = f"vllm:d{rollout_gpus}p1t1"
    cfg["actor"]["optimizer"]["lr"] = spec["lr"]
    cfg["critic"]["optimizer"]["lr"] = spec.get("critic_lr", 5 * spec["lr"])
    cfg["critic"]["path"] = spec.get("critic_model", spec["model"])
    cfg["critic"]["gradient_checkpointing"] = True
    cfg["gconfig"].update(
        max_new_tokens=spec["tokens"], max_tokens=context_tokens,
        temperature=spec.get("temperature", 0.7 if role == "editor" else 1.0),
    )
    cfg["actor"]["mb_spec"]["max_tokens_per_mb"] = context_tokens
    cfg["critic"]["mb_spec"]["max_tokens_per_mb"] = context_tokens + 2048
    cfg["rollout"].update(
        max_head_offpolicyness=spec.get("staleness", 4),
        # One extra batch may be generated while the previous batch trains.
        max_concurrent_rollouts=spec["batch_size"] * (spec.get("staleness", 1) + 1),
        consumer_batch_size=spec["batch_size"], dump_to_file=True,
    )
    cfg["vllm"].update(
        # Rollout GPUs are dedicated (separation strategy), so use them for KV cache.
        max_model_len=context_tokens, max_num_seqs=spec["batch_size"] * 2, gpu_memory_utilization=0.7,
        enforce_eager=False,
    )
    # A single final save per phase; periodic 8 GB saves dominated short phases.
    cfg["saver"].update(freq_steps=spec["steps"])
    env = cfg["actor"]["scheduling_spec"][0]["env_vars"]
    env.update(VLLM_PLUGINS="", TOKENIZERS_PARALLELISM="false",
               LITELLM_LOCAL_MODEL_COST_MAP="True",
               GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo")
    for name in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "PYTHONDONTWRITEBYTECODE",
                 "AREAL_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "HOME", "AREAL_HOST_IP",
                 "SAO_SEARCHQA_RETRIEVAL_URL"):
        if name in os.environ:
            env[name] = os.environ[name]
    context = spec.get("central_context")
    if context:
        env["MA_SAO_CONTEXT_FILE"] = str(Path(context).resolve())
    return cfg


def main():
    import yaml
    from areal.api.cli_args import PPOConfig, load_expr_config
    from datasets import Dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = read_json(args.spec)
    run = Path(spec["run"]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    provenance = verify_upstreams()
    os.environ["SAO_CRITIC_K"] = "2"
    os.environ["SAO_LOCAL_ADMIN_TOKEN"] = secrets.token_urlsafe(32)
    warmup = int(spec.get("critic_warmup_steps", 0))
    if not 0 <= warmup < spec["steps"]:
        raise ValueError("Critic warm-up must leave at least one actor update")
    os.environ["MA_SAO_CRITIC_WARMUP_STEPS"] = str(warmup)
    if spec.get("central_context"):
        os.environ["MA_SAO_CONTEXT_FILE"] = str(Path(spec["central_context"]).resolve())
    else:
        os.environ.pop("MA_SAO_CONTEXT_FILE", None)
    cfg_path = run / "trainer.yaml"
    cfg_path.write_text(yaml.safe_dump(configure(spec, run), sort_keys=False))
    config, _ = load_expr_config(["--config", str(cfg_path)], PPOConfig)
    rows = read_json(Path(spec["data"]))
    started = time.monotonic()
    with upstream_trainer(config, Dataset.from_list(rows)) as trainer:
        trainer.stats_logger = MetricsRecorder(trainer.stats_logger, run / "metrics.jsonl")
        trainer.train(
            workflow="sao_bridge.large_workflow.MathWorkflow",
            workflow_kwargs={
                "role": spec["role"], "tokenizer": spec["model"],
                "gconfig": config.gconfig, "harness_path": spec["harness"],
                "receipts_dir": str(run / "receipts"),
                "evaluator_url": spec.get("evaluator_url"),
                "total_tokens": spec["tokens"],
                "interaction_credit": spec.get("interaction_credit", 0.0),
                "recycle_pool": spec.get("recycle_pool"),
                "domain": spec.get("domain", "math"),
                "context_tokens": spec.get("context_tokens", 8192),
            },
        )
    root = run / "engine/checkpoints" / getpass.getuser() / spec["experiment"] / spec["phase"]
    checkpoints = {}
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if [row["global_step"] for row in metrics] != list(range(spec["steps"])):
        raise RuntimeError("Native optimizer metrics do not match the declared step budget")
    for role in ("actor", "critic"):
        found = sorted((root / role).glob("epoch*globalstep*"),
                       key=lambda p: int(p.name.split("globalstep")[-1]))
        if not found or not list(found[-1].glob("*.safetensors")):
            raise RuntimeError(f"Missing {role} checkpoint")
        if int(found[-1].name.split("globalstep")[-1]) != spec["steps"] - 1:
            raise RuntimeError(f"{role} checkpoint omits final optimizer updates")
        checkpoints[role] = str(found[-1])
    write_json(run / "trained.json", {
        "checkpoints": checkpoints, "actor_steps": spec["steps"] - warmup,
        "critic_only_warmup_steps": warmup,
        "critic_steps": 2 * spec["steps"], "elapsed_seconds": time.monotonic() - started,
        "upstream": provenance, "role": spec["role"], "phase": spec["phase"],
        "central_critic": bool(spec.get("central_context")),
        "stage_optimizer_reset": True,
    })


if __name__ == "__main__":
    main()
