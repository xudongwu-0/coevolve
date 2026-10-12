"""One H or W phase using native SAO with explicit trajectory boundary controls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import getpass
import json
import os
from pathlib import Path
import secrets
import time

from .common import read_json, verify_upstreams, write_json
from .joint_schedule import ALTERNATING, accounting, validate_schedule
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
    if spec.get("terminal_episodes") and any(
        row.get("replay") or row.get("recycled_ids") or row.get("warmup")
        for row in rows[:required]
    ):
        raise ValueError("Terminal episode mode requires fresh complete trajectories")


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
                "recorded_at_unix": time.time(),
                "metrics": data,
            }, default=float) + "\n")


def configure(spec: dict, run: Path) -> dict:
    role = spec["role"]
    critic_mode = spec.get("critic_mode", "learned")
    if critic_mode not in {"learned", "none"}:
        raise ValueError("Critic mode must be learned or none")
    if critic_mode == "none" and spec.get("central_context"):
        raise ValueError("A centralized critic context requires a learned critic")
    trajectory_style = spec.get("solver_trajectory_style", "individual")
    if trajectory_style not in {"individual", "concat"}:
        raise ValueError("Solver trajectory style must be individual or concat")
    for key in (
        "actor_gradient_checkpointing", "critic_gradient_checkpointing",
        "terminal_episodes",
        "independent_trajectories",
        "processed_behavior_logprobs",
    ):
        if key in spec and type(spec[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if spec.get("terminal_episodes") and role == "solver" and trajectory_style != "concat":
        raise ValueError("Terminal Solver episodes require concat trajectory style")
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
    cfg["actor"]["gradient_checkpointing"] = spec.get(
        "actor_gradient_checkpointing", True
    )
    cfg["rollout"]["backend"] = f"vllm:d{rollout_gpus}p1t1"
    if "rollout_setup_timeout" in spec:
        timeout = spec["rollout_setup_timeout"]
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("Rollout setup timeout must be a positive integer")
        cfg["rollout"]["setup_timeout"] = timeout
    cfg["actor"]["optimizer"]["lr"] = spec["lr"]
    kl_ctl = spec.get("kl_ctl", 0.0)
    if type(kl_ctl) not in (int, float) or not 0 <= kl_ctl <= 1:
        raise ValueError("Fixed-reference KL coefficient must be finite and in [0, 1]")
    cfg["actor"]["kl_ctl"] = kl_ctl
    reference_storage = spec.get("reference_storage_dtype")
    if reference_storage not in (None, "bfloat16", "float32"):
        raise ValueError("Reference storage dtype must be bfloat16 or float32")
    if reference_storage is not None and not kl_ctl:
        raise ValueError("Reference storage override requires a fixed KL reference")
    if kl_ctl:
        cfg["ref"] = {
            "backend": cfg["actor"]["backend"],
            "experiment_name": cfg["experiment_name"], "trial_name": cfg["trial_name"],
            "path": spec.get("reference_model", spec["model"]),
            "init_from_scratch": False, "disable_dropout": True,
            "dtype": cfg["actor"]["dtype"], "attn_impl": cfg["actor"]["attn_impl"],
            "temperature": "${gconfig.temperature}",
            "optimizer": None, "mb_spec": {"max_tokens_per_mb": spec.get("microbatch_tokens", context_tokens)},
            "scheduling_strategy": {"type": "colocation", "target": "actor"},
            "scheduling_spec": cfg["actor"]["scheduling_spec"],
        }
        if reference_storage is not None:
            cfg["ref"]["optimizer_dtype"] = reference_storage
    if spec.get("joint_root"):
        cfg["actor"].update(eps_clip=0.2, eps_clip_higher=0.2)
    if critic_mode == "learned":
        cfg["critic"]["optimizer"]["lr"] = spec.get("critic_lr", 5 * spec["lr"])
        cfg["critic"]["path"] = spec.get("critic_model", spec["model"])
        cfg["critic"]["gradient_checkpointing"] = spec.get(
            "critic_gradient_checkpointing", True
        )
    else:
        cfg["critic"] = None
    cfg["gconfig"].update(
        max_new_tokens=spec["tokens"], max_tokens=context_tokens,
        temperature=spec.get("temperature", 0.7 if role == "editor" else 1.0),
    )
    microbatch_tokens = spec.get("microbatch_tokens", context_tokens)
    if type(microbatch_tokens) is not int or microbatch_tokens <= 0:
        raise ValueError("Microbatch token limit must be a positive integer")
    cfg["actor"]["mb_spec"]["max_tokens_per_mb"] = microbatch_tokens
    if cfg["critic"] is not None:
        cfg["critic"]["mb_spec"]["max_tokens_per_mb"] = microbatch_tokens + 2048
    if spec.get("joint_root") or spec.get("independent_trajectories", False):
        # Qwen3.5's HF recurrent layers do not reset at packed sequence boundaries.
        for name in ("actor", "critic", "ref"):
            if cfg.get(name) is not None:
                cfg[name]["mb_spec"]["n_mbs"] = spec["batch_size"] // actor_gpus
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
    # Most phases need only the final checkpoint. Solver-validation campaigns can
    # predeclare sparse checkpoints for held-out checkpoint selection.
    save_interval = int(spec.get("save_interval_steps", spec["steps"]))
    if save_interval < 1 or spec["steps"] % save_interval:
        raise ValueError("Checkpoint interval must divide the declared step budget")
    cfg["saver"].update(freq_steps=save_interval)
    env = cfg["actor"]["scheduling_spec"][0]["env_vars"]
    env.update(VLLM_PLUGINS="", TOKENIZERS_PARALLELISM="false",
               LITELLM_LOCAL_MODEL_COST_MAP="True",
               GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo")
    for name in ("TMPDIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
                 "VLLM_CACHE_ROOT", "CUDA_CACHE_PATH", "PYTHONDONTWRITEBYTECODE",
                 "AREAL_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE", "HOME", "AREAL_HOST_IP",
                 "PYTORCH_ALLOC_CONF",
                 "SAO_SEARCHQA_RETRIEVAL_URL", "SAO_ALFWORLD_CONTROLLER_URL",
                 "SAO_ALFWORLD_TASK_NAME", "SAO_ALFWORLD_HTTP_TIMEOUT",
                 "MA_SAO_JOINT_EPISODES", "MA_SAO_VALUE_NORM", "MA_SAO_COOPERATIVE_CONTEXT",
                 "MA_SAO_ASYNC_JOINT_ROOT", "MA_SAO_JOINT_ROLE", "MA_SAO_PEER_WEIGHT_MODE",
                 "MA_SAO_ANCHOR_SCHEDULE"):
        if name in os.environ:
            env[name] = os.environ[name]
    if spec.get("joint_root"):
        context_enabled = spec.get("critic_cooperative_context", True)
        if type(context_enabled) is not bool:
            raise ValueError("Critic cooperative context must be a boolean")
        env["MA_SAO_COOPERATIVE_CONTEXT"] = str(int(context_enabled))
        env["MA_SAO_VALUE_NORM"] = str(int(spec.get("value_normalization", False)))
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
    validate_schedule(spec)
    scheduled = spec.get("schedule", "simultaneous") != "simultaneous"
    from .joint_anchor import SCHEDULE as ANCHOR_SCHEDULE
    if spec.get("schedule") == ANCHOR_SCHEDULE:
        os.environ["MA_SAO_ANCHOR_SCHEDULE"] = "1"
    else:
        os.environ.pop("MA_SAO_ANCHOR_SCHEDULE", None)
    run = Path(spec["run"]).resolve()
    run.mkdir(parents=True, exist_ok=True)
    provenance = verify_upstreams()
    critic_mode = spec.get("critic_mode", "learned")
    critic_updates = spec.get("critic_updates_per_step", 2)
    if type(critic_updates) is not int or critic_updates < 1:
        raise ValueError("Critic updates per step must be a positive integer")
    os.environ["SAO_CRITIC_K"] = str(critic_updates)
    os.environ["SAO_LOCAL_ADMIN_TOKEN"] = secrets.token_urlsafe(32)
    warmup = int(spec.get("critic_warmup_steps", 0))
    if not 0 <= warmup < spec["steps"]:
        raise ValueError("Critic warm-up must leave at least one actor update")
    if critic_mode == "none" and warmup:
        raise ValueError("Critic-free training cannot use critic warm-up")
    os.environ["MA_SAO_CRITIC_WARMUP_STEPS"] = str(warmup)
    if spec.get("central_context"):
        os.environ["MA_SAO_CONTEXT_FILE"] = str(Path(spec["central_context"]).resolve())
    else:
        os.environ.pop("MA_SAO_CONTEXT_FILE", None)
    if spec.get("joint_root"):
        if spec.get("central_context") or not spec.get("terminal_episodes"):
            raise ValueError("Joint episodes require dynamic context and complete terminal trajectories")
        os.environ["MA_SAO_JOINT_EPISODES"] = "1"
        context_enabled = spec.get("critic_cooperative_context", True)
        if type(context_enabled) is not bool:
            raise ValueError("Critic cooperative context must be a boolean")
        os.environ["MA_SAO_COOPERATIVE_CONTEXT"] = str(int(context_enabled))
        if spec.get("joint_max_lag", 0) == 1:
            os.environ["MA_SAO_ASYNC_JOINT_ROOT"] = str(Path(spec["joint_root"]).resolve())
            os.environ["MA_SAO_JOINT_ROLE"] = spec["role"]
            weight_mode = spec.get("peer_weight_mode", "bounded")
            if weight_mode not in ("bounded", "unit"):
                raise ValueError("Unknown teammate importance weight mode")
            os.environ["MA_SAO_PEER_WEIGHT_MODE"] = weight_mode
        else:
            os.environ.pop("MA_SAO_ASYNC_JOINT_ROOT", None)
            os.environ.pop("MA_SAO_JOINT_ROLE", None)
            os.environ.pop("MA_SAO_PEER_WEIGHT_MODE", None)
        if spec.get("value_normalization", False):
            os.environ["MA_SAO_VALUE_NORM"] = "1"
        else:
            os.environ.pop("MA_SAO_VALUE_NORM", None)
    else:
        os.environ.pop("MA_SAO_JOINT_EPISODES", None)
        os.environ.pop("MA_SAO_COOPERATIVE_CONTEXT", None)
        os.environ.pop("MA_SAO_VALUE_NORM", None)
        os.environ.pop("MA_SAO_ASYNC_JOINT_ROOT", None)
        os.environ.pop("MA_SAO_JOINT_ROLE", None)
        os.environ.pop("MA_SAO_PEER_WEIGHT_MODE", None)
    rows = read_json(Path(spec["data"]))
    validate_training_rows(spec, rows)
    ledger_name = "planned_rows.json" if spec.get("joint_root") else "consumed_rows.json"
    write_json(run / ledger_name, consumed_row_ledger(spec, rows))
    capacity_audit = spec.get("capacity_audit")
    if capacity_audit is None:
        capacity_audit = training_capacity_audit(spec, rows)
    validate_capacity_audit(spec, capacity_audit)
    write_json(run / "capacity_audit.json", capacity_audit)
    cfg_path = run / "trainer.yaml"
    cfg_path.write_text(yaml.safe_dump(configure(spec, run), sort_keys=False))
    config, _ = load_expr_config(["--config", str(cfg_path)], PPOConfig)
    if spec.get("joint_root") or spec.get("processed_behavior_logprobs", False):
        from .joint_workflow import JointVLLMConfig
        config.vllm = JointVLLMConfig(**asdict(config.vllm))
        write_json(run / "rollout_probability_contract.json", {
            "logprobs_mode": config.vllm.logprobs_mode,
            "temperature": config.gconfig.temperature,
            "reason": "Behavior logprobs must include the actual sampling transforms.",
        })
    if (config.critic is None) != (critic_mode == "none"):
        raise RuntimeError("Resolved critic configuration does not match critic_mode")
    started = time.monotonic()
    active_started_at = None
    active_finished_at = None
    with upstream_trainer(
        config, Dataset.from_list(rows),
        terminal_episodes=spec.get("terminal_episodes", False),
        schedule_spec=spec if scheduled else None,
    ) as trainer:
        trainer.stats_logger = MetricsRecorder(trainer.stats_logger, run / "metrics.jsonl")
        from .joint_continuation import handoff
        handoff(trainer, spec, run, loading=True)
        active_started_at = time.time()
        write_json(run / "training_active.json", {
            "status": "running",
            "started_at_unix": active_started_at,
        })
        try:
            workflow = "sao_bridge.large_workflow.MathWorkflow"
            workflow_kwargs = {
                "role": spec["role"], "tokenizer": spec["model"],
                "gconfig": config.gconfig, "harness_path": spec["harness"],
                "receipts_dir": str(run / "receipts"),
                "evaluator_url": spec.get("evaluator_url"),
                "total_tokens": spec["tokens"],
                "interaction_credit": spec.get("interaction_credit", 0.0),
                "recycle_pool": spec.get("recycle_pool"),
                "domain": spec.get("domain", "math"),
                "context_tokens": spec.get("context_tokens", 8192),
                "solver_trajectory_style": spec.get("solver_trajectory_style", "individual"),
            }
            if spec.get("joint_root"):
                from .joint_round import install_version_publication
                from .joint_rollout_budget import install_joint_rollout_budget
                install_joint_rollout_budget(trainer.rollout, spec["steps"] * spec["batch_size"])
                install_version_publication(trainer, Path(spec["joint_root"]), spec["role"])
                workflow = "sao_bridge.joint_workflow.JointWorkflow"
                workflow_kwargs = {
                    "role": spec["role"], "tokenizer": spec["model"],
                    "gconfig": config.gconfig, "joint_root": spec["joint_root"],
                    "total_tokens": spec["tokens"],
                    "context_tokens": spec["context_tokens"],
                    "max_lag": spec.get("joint_max_lag", 0),
                    "editor_reward_mode": spec.get("editor_reward_mode", "team"),
                    "peer_weight_mode": spec.get("peer_weight_mode", "bounded"),
                }
                if spec.get("schedule") == ANCHOR_SCHEDULE:
                    workflow_kwargs.update({key: spec[key] for key in
                                            ("anchor_mode", "anchor_harness", "anchor_harness_sha256")})
                if spec.get("editor_update_every_round", False):
                    workflow_kwargs["editor_update_every_round"] = True
            trainer.train(
                workflow=workflow, workflow_kwargs=workflow_kwargs,
            )
            handoff(trainer, spec, run, loading=False)
        except BaseException as exc:
            active_finished_at = time.time()
            write_json(run / "training_active.json", {
                "status": "failed",
                "started_at_unix": active_started_at,
                "finished_at_unix": active_finished_at,
                "error": repr(exc),
            })
            raise
        active_finished_at = time.time()
        write_json(run / "training_active.json", {
            "status": "completed",
            "started_at_unix": active_started_at,
            "finished_at_unix": active_finished_at,
        })
    root = run / "engine/checkpoints" / getpass.getuser() / spec["experiment"] / spec["phase"]
    checkpoints = {}
    checkpoint_history = {}
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    if [row["global_step"] for row in metrics] != list(range(spec["steps"])):
        raise RuntimeError("Native optimizer metrics do not match the declared step budget")
    checkpoint_roles = ("actor",) if critic_mode == "none" else ("actor", "critic")
    for role in checkpoint_roles:
        found = sorted((root / role).glob("epoch*globalstep*"),
                       key=lambda p: int(p.name.split("globalstep")[-1]))
        if not found or not list(found[-1].glob("*.safetensors")):
            raise RuntimeError(f"Missing {role} checkpoint")
        if int(found[-1].name.split("globalstep")[-1]) != spec["steps"] - 1:
            raise RuntimeError(f"{role} checkpoint omits final optimizer updates")
        checkpoints[role] = str(found[-1])
        checkpoint_history[role] = [
            {"global_step": int(path.name.split("globalstep")[-1]), "path": str(path)}
            for path in found
            if list(path.glob("*.safetensors"))
        ]
    write_json(run / "trained.json", {
        "checkpoints": checkpoints, "actor_steps": spec["steps"] - warmup,
        "critic_only_warmup_steps": warmup,
        "critic_steps": (
            critic_updates * spec["steps"] if critic_mode == "learned" else 0
        ),
        "critic_mode": critic_mode,
        "terminal_episodes": spec.get("terminal_episodes", False),
        "critic_updates_per_step": critic_updates,
        "actor_gradient_checkpointing": spec.get(
            "actor_gradient_checkpointing", True
        ),
        "critic_gradient_checkpointing": (
            spec.get("critic_gradient_checkpointing", True)
            if critic_mode == "learned"
            else None
        ),
        "solver_trajectory_style": spec.get(
            "solver_trajectory_style", "individual"
        ),
        "training_active_started_at_unix": active_started_at,
        "training_active_finished_at_unix": active_finished_at,
        "elapsed_seconds": time.monotonic() - started,
        "upstream": provenance, "role": spec["role"], "phase": spec["phase"],
        "central_critic": bool(spec.get("central_context") or (
            spec.get("joint_root") and spec.get("critic_cooperative_context", True))),
        "component_ablation": spec.get("component_ablation"),
        "editor_update_every_round": spec.get("editor_update_every_round", False),
        "value_normalization": bool(spec.get("joint_root") and spec.get("value_normalization", False)),
        "joint_root": spec.get("joint_root"),
        "stage_optimizer_reset": True, "checkpoint_history": checkpoint_history,
        **(accounting(spec, spec["steps"]) if scheduled else {}),
    })


if __name__ == "__main__":
    main()
