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


def validate_capacity_audit(spec: dict, audit: dict) -> None:
    capacity = spec.get("microbatch_tokens", spec.get("context_tokens", 8192))
    if audit.get("role") != spec["role"]:
        raise ValueError("Trajectory capacity audit role does not match the training role")
    if audit.get("microbatch_tokens") != capacity:
        raise ValueError("Trajectory capacity audit does not match the microbatch limit")
    required = audit.get("required_microbatch_tokens")
    if type(required) is not int or required <= 0:
        raise ValueError("Trajectory capacity audit omitted a positive token requirement")
    if required > capacity:
        raise ValueError(
            f"Microbatch token limit ({capacity}) cannot fit the longest possible "
            f"trajectory ({required})"
        )
    if required > spec.get("context_tokens", 8192):
        raise ValueError("Longest possible trajectory exceeds the training context"        )


def validate_training_rows(spec: dict, rows: list[dict]) -> None:
    required = spec["steps"] * spec["batch_size"]
    if len(rows) < required:
        raise ValueError(
            f"Training data has {len(rows)} rows but {required} distinct rows "
            "are required for the declared step budget"
        )
    sample_ids = [row.get("id") for row in rows[:required]]
    if any(not isinstance(sample_id, str) or not sample_id for sample_id in sample_ids):
        raise ValueError("Every consumed training row must have a nonempty string ID")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("The declared step budget would consume duplicate training IDs")


def consumed_row_ledger(spec: dict, rows: list[dict]) -> dict:
    validate_training_rows(spec, rows)
    count = spec["steps"] * spec["batch_size"]
    consumed = rows[:count]
    return {
        "rows": count,
        "sample_indices": list(range(count)),
        "ids": [row["id"] for row in consumed],
        "fresh_ids": [row["id"] for row in consumed if not row.get("replay")],
        "replay_ids": [row["id"] for row in consumed if row.get("replay")],
    }


def training_capacity_audit(spec: dict, rows: list[dict]) -> dict:
    capacity = spec.get("microbatch_tokens", spec.get("context_tokens", 8192))
    if spec["role"] != "editor":
        return {
            "role": spec["role"], "applicable": False,
            "microbatch_tokens": capacity,
            "required_microbatch_tokens": 1,
        }
    from transformers import AutoTokenizer
    from .domain_api import get_domain

    tokenizer = AutoTokenizer.from_pretrained(
        spec["model"], trust_remote_code=True, local_files_only=True,
    )
    domain = get_domain(spec.get("domain", "math"))
    prefill = getattr(domain, "EDITOR_ASSISTANT_PREFILL", None)
    extra_body = getattr(domain, "EDITOR_EXTRA_BODY", None) or {}
    template_kwargs = dict(extra_body.get("chat_template_kwargs", {}))
    prompt_lengths = []
    replay_lengths = []
    for row in rows:
        if row.get("replay"):
            call = row["replay"]["call"]
            replay_lengths.append(
                len(call["prompt_ids"]) + len(call["response_ids"])
            )
            continue
        messages = list(row["messages"])
        if prefill is not None:
            messages.append({"role": "assistant", "content": prefill})
        encoded = tokenizer.apply_chat_template(
            messages, tokenize=True, **template_kwargs,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        prompt_lengths.append(len(input_ids))
    requirements = list(replay_lengths)
    if prompt_lengths:
        requirements.append(max(prompt_lengths) + spec["tokens"])
    if not requirements:
        raise ValueError("Editor trajectory capacity audit found no training rows")
    audit = {
        "role": "editor", "applicable": True,
        "fresh_rows": len(prompt_lengths), "replay_rows": len(replay_lengths),
        "minimum_fresh_prompt_tokens": min(prompt_lengths) if prompt_lengths else None,
        "maximum_fresh_prompt_tokens": max(prompt_lengths) if prompt_lengths else None,
        "maximum_new_tokens": spec["tokens"],
        "maximum_replay_trajectory_tokens": (
            max(replay_lengths) if replay_lengths else None
        ),
        "required_microbatch_tokens": max(requirements),
        "microbatch_tokens": capacity,
        "context_tokens": spec.get("context_tokens", 8192),
        "chat_template_kwargs": template_kwargs,
        "assistant_prefill": prefill,
    }
    validate_capacity_audit(spec, audit)
    return audit


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
    if spec["batch_size"] < actor_gpus:
        raise ValueError(
            f"Training batch size ({spec['batch_size']}) must be at least the "
            f"actor data-parallel size ({actor_gpus})"
        )
    if spec["batch_size"] % actor_gpus:
        raise ValueError(
            f"Training batch size ({spec['batch_size']}) must be divisible by the "
            f"actor data-parallel size ({actor_gpus})"
        )
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
    microbatch_tokens = spec.get("microbatch_tokens", context_tokens)
    if type(microbatch_tokens) is not int or microbatch_tokens <= 0:
        raise ValueError("Microbatch token limit must be a positive integer")
    cfg["actor"]["mb_spec"]["max_tokens_per_mb"] = microbatch_tokens
    cfg["critic"]["mb_spec"]["max_tokens_per_mb"] = microbatch_tokens + 2048
    cfg["rollout"].update(
        max_head_offpolicyness=spec.get(
            "max_head_offpolicyness", spec.get("staleness", 4)
        ),
        # One extra batch may be generated while the previous batch trains.
        max_concurrent_rollouts=spec["batch_size"] * (spec.get("staleness", 1) + 1),
        consumer_batch_size=spec["batch_size"], dump_to_file=True,
    )
    cfg["vllm"].update(
        # Rollout GPUs are dedicated (separation strategy), so use them for KV cache.
        max_model_len=context_tokens, max_num_seqs=spec["batch_size"] * 2, gpu_memory_utilization=0.7,
        enforce_eager=False, gdn_prefill_backend=spec.get("gdn_prefill_backend"),
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
                 "PYTORCH_ALLOC_CONF",
                 "SAO_SEARCHQA_RETRIEVAL_URL", "SAO_ALFWORLD_CONTROLLER_URL",
                 "SAO_ALFWORLD_TASK_NAME", "SAO_ALFWORLD_HTTP_TIMEOUT"):
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
    rows = read_json(Path(spec["data"]))
    validate_training_rows(spec, rows)
    write_json(run / "consumed_rows.json", consumed_row_ledger(spec, rows))
    capacity_audit = spec.get("capacity_audit")
    if capacity_audit is None:
        capacity_audit = training_capacity_audit(spec, rows)
    validate_capacity_audit(spec, capacity_audit)
    write_json(run / "capacity_audit.json", capacity_audit)
    cfg_path = run / "trainer.yaml"
    cfg_path.write_text(yaml.safe_dump(configure(spec, run), sort_keys=False))
    config, _ = load_expr_config(["--config", str(cfg_path)], PPOConfig)
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
