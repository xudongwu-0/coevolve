"""Fresh E0/A1, concurrent independent-SAO versus MA-SAO comparison.

Preparation is CPU-only. Execution requires a separate, exact-configuration
native validation receipt; this module never allocates nodes or retries work.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time


SCHEMA = "e0-a1-concurrent-comparison-v1"
BASELINE = "independent-sao-shared-episode-v1"
STANDALONE_MA = "ma-sao-every-round-joint-v1"
ARMS = ("ma_sao", "sao")
ROLES = ("editor", "solver")
MILESTONES = [36, 72, 108, 144]
REMOTE_ROOT = Path("/mnt/vast/chengboliu/xudong")
REVISION = "f6a915953f8206f3f5c7ef0b40240e1f4d2b6676"
STANDALONE_ABLATION = "ma-sao-single-addback-v1"
ABLATION_MECHANISMS = {
    "joint_only": ("unit", False, False),
    "add_context": ("unit", True, False),
    "add_peer": ("bounded", False, False),
    "add_valuenorm": ("unit", False, True),
}
MECHANISMS = {
    **ABLATION_MECHANISMS,
    "ma_sao": ("bounded", True, True),
    "sao": ("unit", False, False),
}
EXECUTION_BLOCKER = (
    "Concurrent ordinary SAO requires the original ten-task differential Engineer reward. "
    "The shared-episode independent-SAO prototype is not authorized as its replacement. "
    "Version-pinned ten-task reward serving and separate cost ledgers are not implemented; "
    "do not prepare or launch either arm from this prototype."
)


def require_objective_implementation() -> None:
    raise NotImplementedError(EXECUTION_BLOCKER)


def require_executable_spec(spec: dict) -> None:
    if spec.get("comparison_schema") == SCHEMA:
        if spec.get("execution_mode") in ("standalone-ma-v1", "standalone-ma-ablation-v1"):
            validate_method(spec)
            from .slurm_runtime import binding
            root, plan = binding()
            arm = spec["comparison_arm"]
            kind = "standalone-ma-ablation-training" if arm in ABLATION_MECHANISMS else "standalone-ma-training"
            path = root / arm / f"{spec['role']}_spec.json"
            if (
                plan.get("kind") != kind
                or plan["config"].get("active_arms") != [arm]
                or plan["specs"].get(str(path)) != sha(path)
                or load(path) != spec
            ):
                raise ValueError("Standalone MA spec is not bound to its authorized allocation plan")
            return
        require_objective_implementation()


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def immutable(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def contained(path: Path, root: Path = REMOTE_ROOT) -> Path:
    resolved, boundary = path.resolve(), root.resolve()
    if not resolved.is_relative_to(boundary) or resolved == boundary:
        raise ValueError(f"Path is outside the experiment boundary: {path}")
    return resolved


def budget() -> dict:
    return {
        "paired_rounds": 144, "episodes": 1152, "generated_role_trajectories": 2304,
        "per_role_consumed": 1152, "per_role_actor_updates": 141,
        "per_role_critic_updates": 288, "critic_only_rounds": 3,
        "effective_batch_per_role": 8, "per_device_microbatch_sequences": 1,
        "learner_ranks_per_role": 4, "accumulation_microbatches_per_rank": 2,
        "checkpoints": MILESTONES, "evaluated_checkpoints": [144],
    }


def validate_method(spec: dict) -> bool:
    """Validate the complete intervention, never an alias for without_peer."""
    if spec.get("comparison_schema") != SCHEMA:
        return False
    arm = spec.get("comparison_arm")
    definition = {
        "standalone-ma-v1": STANDALONE_MA,
        "standalone-ma-ablation-v1": STANDALONE_ABLATION,
    }.get(spec.get("execution_mode"), BASELINE)
    if (arm not in MECHANISMS or spec.get("baseline_definition") != definition
            or definition == STANDALONE_MA and arm != "ma_sao"
            or (definition == STANDALONE_ABLATION) != (arm in ABLATION_MECHANISMS)):
        raise ValueError("Unknown concurrent comparison or baseline definition")
    if spec.get("component_ablation") is not None:
        raise ValueError("A method-level comparison is not an isolated historical ablation")
    actual = (
        spec.get("peer_weight_mode"), spec.get("critic_cooperative_context"),
        spec.get("value_normalization"),
    )
    if actual != MECHANISMS[arm] or any(type(v) is not bool for v in actual[1:]):
        raise ValueError("Concurrent method mechanisms do not match the declared arm")
    expected = {
        "schedule": "simultaneous", "editor_reward_mode": "team",
        "steps": 144, "batch_size": 8, "seed": 1000043,
        "critic_warmup_steps": 3, "critic_updates_per_step": 2,
        "actor_gpus": 4, "rollout_gpus": 4, "context_tokens": 32768,
        "microbatch_tokens": 32768, "tokens": 4096, "save_interval_steps": 36,
        "lr": 5e-7, "critic_lr": 2.5e-6, "kl_ctl": .02,
        "staleness": 1, "max_head_offpolicyness": 1, "joint_max_lag": 1,
        "max_concurrent_rollouts": 8, "vllm_max_num_seqs": 2,
        "critic_mode": "learned", "terminal_episodes": True,
        "solver_trajectory_style": "concat", "fresh_critic_seed": 1000043,
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError("Concurrent comparison changed its paired schedule or matched budget")
    if (
        spec.get("anchor_mode") is not None or spec.get("editor_update_every_round", False)
        or spec.get("reference_model") != spec.get("model")
        or spec.get("critic_model") != spec.get("model")
        or spec.get("preserve_optimizer_state", False)
        or spec.get("central_context")
    ):
        raise ValueError("Fresh concurrent runs cannot inherit old critics, references or schedules")
    recovery = spec.get("rolling_recovery", {})
    if recovery.get("interval_rounds") != 36 or recovery.get("hf_rounds") != MILESTONES:
        raise ValueError("Both HF and paired native states must save at36/72/108/144")
    return True


def paired_rows(source: list[dict]) -> list[dict]:
    if len(source) != 1152:
        raise ValueError("Supply exactly1152 pinned input rows, not cached rollout trajectories")
    rows = []
    for index, original in enumerate(source):
        if (
            type(original.get("task_id")) is not int or not 0 <= original["task_id"] < 240
            or not original.get("messages") or not original.get("task_context")
            or original.get("replay") or original.get("recycled_ids")
        ):
            raise ValueError("Comparison inputs must be fresh train0-239 prompt/task records")
        row = {key: copy.deepcopy(original[key]) for key in (
            "task_id", "messages", "task_context", "critic_evidence",
        ) if key in original}
        row.update(id=f"paired-{index:05d}-task{row['task_id']}", task_split="test_set",
                   joint_version=index // 8, joint_slot=index % 8)
        rows.append(row)
    for offset in range(0, len(rows), 8):
        if len({row["task_id"] for row in rows[offset:offset + 8]}) != 8:
            raise ValueError("Each paired batch must contain eight different task assignments")
    return rows


def role_spec(config: dict, arm: str, role: str, data: Path, pins: dict) -> dict:
    if arm not in MECHANISMS or role not in ROLES:
        raise ValueError("Unknown comparison arm/role")
    root = Path(config["run"])
    model = config["engineer_model" if role == "editor" else "solver_model"]
    peer, context, norm = MECHANISMS[arm]
    spec = {
        "comparison_schema": SCHEMA, "comparison_arm": arm, "baseline_definition": BASELINE,
        "protocol": "joint-ma-sao-async-v1", "role": role, "schedule": "simultaneous",
        "run": str(root / arm / role / "optimizer"), "joint_root": str(root / arm / "joint"),
        "model": model, "critic_model": model, "reference_model": model,
        "data": str(data), "harness": config["no_patch"], "phase": role,
        "experiment": f"{root.name}-{arm}", "steps": 144, "batch_size": 8, "seed": 1000043,
        "starting_point": "official-e0-a1-fresh-critics", "fresh_critic_seed": 1000043,
        "critic_warmup_steps": 3, "critic_updates_per_step": 2, "critic_mode": "learned",
        "actor_gpus": 4, "rollout_gpus": 4, "context_tokens": 32768,
        "microbatch_tokens": 32768, "tokens": 4096, "save_interval_steps": 36,
        "lr": 5e-7, "critic_lr": 2.5e-6, "kl_ctl": .02,
        "temperature": .7 if role == "editor" else 1.,
        "staleness": 1, "max_head_offpolicyness": 1, "joint_max_lag": 1,
        "max_concurrent_rollouts": 8, "vllm_max_num_seqs": 2,
        "vllm_gpu_memory_utilization": .7,
        "terminal_episodes": True, "solver_trajectory_style": "concat",
        "actor_gradient_checkpointing": True, "critic_gradient_checkpointing": True,
        "reference_storage_dtype": "bfloat16", "rollout_setup_timeout": 1800,
        "gdn_prefill_backend": "triton",
        "peer_weight_mode": peer, "critic_cooperative_context": context,
        "value_normalization": norm, "editor_reward_mode": "team",
        "stop_on_diagnostic_thresholds": False,
        "training_deadline_unix": config["training_deadline_unix"],
        "phase_timeout_seconds": config.get("phase_timeout_seconds", 48 * 3600),
        "pinned_files": pins, "fsdp_offload_params": config.get("fsdp_offload_params", True),
        "rolling_recovery": {
            "root": str(root / arm / "native"), "interval_rounds": 36,
            "hf_rounds": MILESTONES, "attempt": "attempt-0001",
            "retain_generations": True,
        },
    }
    if arm in ABLATION_MECHANISMS:
        spec.update(execution_mode="standalone-ma-ablation-v1",
                    baseline_definition=STANDALONE_ABLATION)
    if config.get("training_kernel_overlay"):
        for key in ("training_kernel_overlay", "training_kernel_versions", "training_kernel_validation"):
            spec[key] = config[key]
    validate_method(spec)
    return spec


def prepare(config_path: Path) -> dict:
    require_objective_implementation()
    config = load(config_path)
    if config.get("baseline_definition") != BASELINE:
        raise ValueError("Explicitly declare the concurrent independent-SAO shared-episode adaptation")
    root = contained(Path(config["run"]))
    if root.exists():
        raise FileExistsError("A fresh comparison cannot reuse or overwrite an existing run")
    repo = Path(__file__).resolve().parents[2]
    expected = repo / "assets/Harness-R1"
    for key, name in (("engineer_model", "harness-r1"), ("solver_model", "target-agent-sft")):
        path = contained(Path(config[key]))
        if path != (expected / name).resolve() or not (path / "config.json").is_file():
            raise ValueError("Initialization must be the verified official E0/A1 directories")
        if any((path / filename).exists() for filename in (
            "joint_value_norm.json", "joint_schedule.json", "joint_critic_settings.json",
        )):
            raise ValueError("Official initialization unexpectedly contains learned critic state")
    if config.get("model_revision") != REVISION:
        raise ValueError("Starting model publication revision changed")
    nodes = config.get("nodes", {})
    hosts = [nodes.get(arm, {}).get(role) for arm in ARMS for role in ROLES]
    if len(set(hosts)) != 4 or any(not isinstance(host, str) or not host for host in hosts):
        raise ValueError("Exactly four distinct allocated nodes, one per role, are required")
    if config["training_deadline_unix"] <= time.time():
        raise ValueError("Explicit future training deadline required")
    inputs = {}
    for key in ("source_data", "no_patch", "evaluation_protocol", "initial_state"):
        path = contained(Path(config[key]))
        inputs[str(path)] = sha(path)
    contained(Path(config["worker_python"]))
    if config.get("training_kernel_overlay"):
        validation = contained(Path(config["training_kernel_validation"]))
        inputs[str(validation)] = sha(validation)
    rows = paired_rows(load(Path(config["source_data"])))
    evaluation = load(Path(config["evaluation_protocol"]))
    state = load(Path(config["initial_state"]))
    # Evaluation inputs are supplied explicitly, not resolved from historical model paths.
    for path in (evaluation["core200_protocol"], evaluation["evaluation"]["task_manifest"],
                 evaluation["h2"]["template"], state["packets"]):
        inputs[str(contained(Path(path)))] = sha(Path(path))
    state.update(engineer=config["engineer_model"], solver=config["solver_model"],
                 engineer_critic=config["engineer_model"], solver_critic=config["solver_model"],
                 solver_epoch_version=0, packets_sha256=sha(Path(state["packets"])))
    evaluation.update(no_patch=config["no_patch"], incumbent_min_gain=2 / 48)
    selection = load(Path(evaluation["evaluation"]["task_manifest"]))["selection_task_ids"]
    core = load(Path(evaluation["core200_protocol"]))
    if (
        len(selection) != 48 or len(set(selection)) != 48
        or len(core["all_task_ids"]) != 200 or len(set(core["all_task_ids"])) != 200
        or set(selection) - set(core["all_task_ids"])
        or set(range(240)) & set(core["all_task_ids"])
        or evaluation["evaluation"]["samples"] != 1
        or evaluation["evaluation"]["solver_tokens"] != 4096
        or core["solver_tokens"] != 4096
    ):
        raise ValueError("Expected pinned selection48/main200 development cohorts and4096 action budget")
    template = load(Path(evaluation["h2"]["template"]))
    if template["engineer_tokens"] != 4096:
        raise ValueError("Endpoint Engineer proposal budget must remain4096")
    if (evaluation["evaluation"]["seed"] != core["seed"]
            or evaluation["evaluation"]["temperature"] != core["temperature"]):
        raise ValueError("Selection and main inference contracts must use the same declared seed/temperature")
    # Freeze the live implementation separately; never modify historical snapshots.
    for path in sorted((repo / "whale_local/sao_bridge").glob("*.py")):
        inputs[str(path)] = sha(path)
    immutable(root / "joint_data.json", rows)
    immutable(root / "PROTOCOL.json", evaluation)
    immutable(root / "INITIAL_STATE.json", state)
    for path in (root / "joint_data.json", root / "PROTOCOL.json", root / "INITIAL_STATE.json"):
        inputs[str(path)] = sha(path)
    specs = {}
    for arm in ARMS:
        for role in ROLES:
            path = root / arm / f"{role}_spec.json"
            immutable(path, role_spec(config, arm, role, root / "joint_data.json", inputs))
            specs[str(path)] = sha(path)
    plan = {
        "schema": SCHEMA, "config": config, "budget_per_arm": budget(),
        "inputs": inputs, "specs": specs, "created_at_unix": time.time(),
        "semantics": BASELINE,
        "critic_initialization": "Corresponding Actor backbone; fresh deterministic token-value head, "
                                 "fresh Adam, empty MA ValueNorm; no old trained Critic loaded.",
        "limitations": [
            "Concurrent independent SAO on shared team episodes, not historical staged10-task delta reward.",
            "Method bundle comparison: peer weighting, cooperative Critic context and ValueNorm differ.",
            "Same sample/update budget is not equal token/FLOP/time budget.",
            "One seed and reused development selection48/main200, not an untouched test.",
            "No native9B/H100/FLA feasibility claim until separate validation passes.",
        ],
    }
    immutable(root / "COMPARISON.json", plan)
    return plan


def verify(root: Path) -> dict:
    plan = load(root / "COMPARISON.json")
    if plan.get("schema") != SCHEMA or Path(plan["config"]["run"]).resolve() != root.resolve():
        raise ValueError("Comparison root/identity mismatch")
    if (root / "CANCELLED.json").exists() or (root / "FAILED.json").exists():
        raise RuntimeError("Comparison is cancelled or failed; no automatic restart")
    for path, expected in {**plan["inputs"], **plan["specs"]}.items():
        if sha(Path(path)) != expected:
            raise ValueError(f"Pinned comparison input changed: {path}")
    for path in plan["specs"]:
        validate_method(load(Path(path)))
    return plan


def final_pairs(root: Path, *, arms: tuple[str, ...] = ARMS) -> dict:
    """Verify the requested committed endpoints and their training-process release."""
    plan = verify(root)
    if not arms or len(set(arms)) != len(arms) or any(arm not in MECHANISMS for arm in arms):
        raise ValueError("Unknown endpoint arm selection")
    from .joint_recovery import science_digest, validate_commit
    pairs = {}
    for arm in arms:
        native = root / arm / "native"
        latest = load(native / "LATEST.json")
        commit = native / latest["generation"] / "COMMITTED.json"
        if latest.get("completed_rounds") != 144 or sha(commit) != latest["sha256"]:
            raise ValueError("Each requested arm must finish paired native144 before its evaluation")
        validate_commit(commit, root=native)
        pairs[arm] = {}
        for role in ROLES:
            spec = load(root / arm / f"{role}_spec.json")
            native_role = load(commit.parent / f"{role}.json")
            if (native_role["science_sha256"] != science_digest(spec)
                    or native_role["source_run"] != spec["run"]
                    or native_role["data_sha256"] != sha(Path(spec["data"]))):
                raise ValueError("Native144 state belongs to a different experiment/specification")
            receipt = load(root / arm / role / "COMPLETE.json")
            released = load(root / arm / role / "RELEASED.json")
            if (
                released.get("remaining_gpu_pids") or released.get("vanished_gpu_pids")
                or released.get("slurm_allocation", {}).get("host")
                != plan["config"]["nodes"][arm][role]
            ):
                raise ValueError("The requested arm must release both training roles before evaluation")
            metrics = root / arm / role / "optimizer/metrics.jsonl"
            if (
                receipt.get("status") != "completed" or receipt.get("steps") != 144
                or receipt.get("actor_updates") != 141 or receipt.get("critic_updates") != 288
                or receipt.get("metrics_sha256") != sha(metrics)
            ):
                raise ValueError("Final role completion/budget does not match committed144")
            pairs[arm][role] = receipt["checkpoint_history"]["143"]
            for model in pairs[arm][role].values():
                model_path = contained(Path(model), Path(spec["run"]))
                if (
                    not model_path.is_dir() or not model_path.name.endswith("globalstep143")
                    or str(model_path.relative_to(Path(spec["run"])))
                    not in native_role["protected_hf_relative_paths"]
                ):
                    raise ValueError("Evaluation may use only final144 model paths")
    return pairs


def native_gate(root: Path, receipt_path: Path) -> dict:
    """Require real native evidence; a tiny DDP/import smoke is insufficient."""
    plan = verify(root)
    receipt = load(contained(receipt_path))
    if (
        receipt.get("schema") != "concurrent-h100-native-gate-v1"
        or receipt.get("status") != "passed"
        or receipt.get("comparison_sha256") != sha(root / "COMPARISON.json")
        or receipt.get("specs") != plan["specs"]
        or receipt.get("outside_root_writes") != []
        or receipt.get("write_boundary_audited") is not True
        or receipt.get("checkpoint_roundtrip_verified") is not True
    ):
        raise ValueError("Missing exact H100/9B/native-checkpoint/write-boundary validation")
    profiles = receipt.get("profiles", {})
    if set(profiles) != {f"{arm}/{role}" for arm in ARMS for role in ROLES}:
        raise ValueError("Native validation must cover both roles and both method profiles")
    for name, profile in profiles.items():
        if (
            profile.get("context_tokens") != 32768
            or profile.get("action_tokens") != 4096
            or profile.get("microbatch_sequences") != 1
            or profile.get("actor_optimizer_steps", 0) < 2
            or profile.get("critic_optimizer_steps", 0) < 4
            or profile.get("finite_nonzero_gradients") is not True
            or profile.get("gpu_model") != "H10080GB"
            or not isinstance(profile.get("fresh_critic_head_sha256"), str)
            or len(profile["fresh_critic_head_sha256"]) != 64
            or not profile.get("evidence")
        ):
            raise ValueError(f"Native profile is incomplete: {name}")
        for path, expected in profile["evidence"].items():
            if sha(contained(Path(path))) != expected:
                raise ValueError("Native evidence changed")
    for role in ROLES:
        if (profiles[f"ma_sao/{role}"]["fresh_critic_head_sha256"]
                != profiles[f"sao/{role}"]["fresh_critic_head_sha256"]):
            raise ValueError("The two arms did not initialize the same role's fresh Critic head")
    return receipt


def run_role(root: Path, arm: str, role: str, validation: Path) -> dict:
    require_objective_implementation()
    if arm not in MECHANISMS or role not in ROLES:
        raise ValueError("Unknown arm/role")
    native_gate(root, validation)
    from .slurm_runtime import execution_environment, require_node
    os.environ.update(execution_environment(root))
    allocation = require_node()
    plan = verify(root)
    if allocation["host"] != plan["config"]["nodes"][arm][role]:
        raise ValueError("This role was launched on the wrong bound node")
    from .joint_experiment import run_role as native_role
    return native_role(root / arm / f"{role}_spec.json")


def evaluate_final(root: Path, arm: str, validation: Path | None) -> dict:
    if arm not in ("ma_sao", *ABLATION_MECHANISMS) or validation is not None:
        require_objective_implementation()
    plan = verify(root)
    standalone = (
        plan.get("kind") == (
            "standalone-ma-ablation-training" if arm in ABLATION_MECHANISMS
            else "standalone-ma-training"
        )
        and plan["config"].get("active_arms") == [arm]
    )
    if not standalone:
        require_objective_implementation()
    if arm not in MECHANISMS:
        raise ValueError("Unknown final comparison arm")
    if not standalone:
        if validation is None:
            raise ValueError("The combined prototype requires native validation")
        native_gate(root, validation)
    from .slurm_runtime import execution_environment, require_node
    os.environ.update(execution_environment(root))
    require_node()
    pairs = final_pairs(root, arms=(arm,)) if standalone else final_pairs(root)
    from .alfworld_ma_mechanism_study import select_at_pair
    from .alfworld_strong_select import matrix_config, run_matrix, checked_native_report
    from .launch import Launcher
    from .resources import gpu_release_audit, research_resources_released

    output = contained(root / arm / "final144", root)
    if output.exists():
        raise FileExistsError("Never repeat or overwrite an endpoint; diagnose partial evaluation explicitly")
    if not research_resources_released(gpu_release_audit()):
        raise RuntimeError("Evaluation requires released GPUs after this arm's two training roles finish")
    protocol = load(root / "PROTOCOL.json")
    launch = Launcher(output / "supervision")
    try:
        selection = select_at_pair(
            launch, root, pairs[arm], output, protocol["no_patch"],
        )
        core = load(Path(protocol["core200_protocol"]))
        config = matrix_config(protocol, pairs[arm]["solver"]["actor"], f"{arm}-144",
                               {"selected": selection["harness"]}, output / "evaluation200")
        config.update(
            task_ids=core["all_task_ids"], samples=1, episode_checkpointing=True,
            **{key: core[key] for key in ("seed", "temperature", "solver_tokens")},
        )
        matrix, path = run_matrix(launch, config, output / "evaluation200", 1)
        scores = {}
        for name, value in matrix["variants"].items():
            report_path = Path(value["report"])
            report = checked_native_report(load(report_path), core["all_task_ids"], config)
            scores[name] = {
                "successes": sum(row["reward"] for row in report["results"]),
                "questions": 200, "report": str(report_path), "sha256": sha(report_path),
            }
        result = {"status": "completed", "arm": arm, "rounds": 144, "selection": selection,
                  "scores": scores, "matrix": str(path),
                  "both_arms_complete_before_evaluation": not standalone,
                  "own_arm_complete_before_evaluation": True}
    finally:
        launch.close()
        immutable(output / "RELEASED.json", gpu_release_audit())
    if not research_resources_released(load(output / "RELEASED.json")):
        raise RuntimeError("Final evaluation did not release its owned GPU processes")
    immutable(output / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "final-paths", "role", "final-eval"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--arm", choices=tuple(MECHANISMS))
    parser.add_argument("--role", choices=ROLES)
    parser.add_argument("--native-validation", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.config is None:
            parser.error("prepare requires --config")
        result = prepare(args.config)
    else:
        if args.root is None:
            parser.error("this command requires --root")
        if args.command in ("role", "final-eval"):
            if not args.arm or not args.native_validation or (args.command == "role" and not args.role):
                parser.error("execution requires --arm, --native-validation, and --role for training")
            result = (
                run_role(args.root, args.arm, args.role, args.native_validation)
                if args.command == "role" else evaluate_final(args.root, args.arm, args.native_validation)
            )
        else:
            result = verify(args.root) if args.command == "verify" else final_pairs(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
