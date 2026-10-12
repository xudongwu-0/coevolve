"""Single-seed MA component controls with an isolated Engineer-frequency ablation."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import time

from .alfworld_core_extend import validate_cohort
from .alfworld_reward_service import validate_metadata
from .alfworld_strong_phase import checked_native_report, checked_release, checkpoint_pair
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import ROOT, file_sha, read_json, write_json
from .joint_anchor import build_rows
from .joint_campaign import initial_inputs, share_checkpoints
from .joint_experiment import run_role, validate_spec
from .joint_health import metric_rows, metric_value
from .joint_schedule import role_active, update_counts
from .launch import Launcher
from .resources import gpu_release_audit, research_resources_released

ARMS = ("full", "without_peer", "without_context", "without_value_norm", "always_joint")
ROUNDS = (96, 144)
CORE = ROOT / "runs/alfworld_core_200_oneshot_v2/protocol.json"
OLD = ROOT / "runs/alfworld_ma_balanced_fast_v1r2/training"


def file_manifest(paths) -> dict:
    paths = list(dict.fromkeys(Path(path) for path in paths))
    with ThreadPoolExecutor(max_workers=4) as pool:
        return dict(zip(map(str, paths), pool.map(file_sha, paths), strict=True))


def arm_budgets(arm: str, *, rounds: int = 144, batch_size: int = 8) -> dict:
    if arm not in ARMS:
        raise ValueError("Unknown mechanism-study arm")
    spec = {
        "schedule": "joint_joint_anchor", "anchor_mode": "sampled_control",
        "steps": rounds, "batch_size": batch_size, "critic_warmup_steps": 3,
        "critic_updates_per_step": 2, "component_ablation": arm,
        "editor_update_every_round": arm == "always_joint",
    }
    roles = {
        role: update_counts(spec, rounds, role) for role in ("editor", "solver")
    }
    return {
        "roles": {
            role: {**counts, "consumed_trajectories": counts["active_rounds"] * batch_size}
            for role, counts in roles.items()
        },
        "shared_environment_episodes": rounds * batch_size,
        "generated_role_trajectories": 2 * rounds * batch_size,
    }


def staged_solver_rows(joint_rows: list[dict]) -> list[dict]:
    queues = {}
    for row in joint_rows:
        task = row["task_id"]
        if type(task) is not int or not 0 <= task < 240:
            raise ValueError("Staged control must keep the original training task population")
        queues.setdefault(task // 10, deque()).append(row)
    recent, output = deque(maxlen=15), []
    for index in range(len(joint_rows)):
        available = [group for group, queue in queues.items() if queue and group not in recent]
        if not available:
            raise ValueError("Exact task exposure cannot satisfy the ordinary Solver active-window contract")
        group = min(available, key=lambda value: (-len(queues[value]), value))
        source = queues[group].popleft()
        output.append({
            "id": f"staged-solver-{index:05d}-task{source['task_id']}",
            "task_id": source["task_id"], "task_name": f"staged-packet-{group:02d}",
            "partner_version": 1, "recycled_ids": [], "warmup": False,
            "source_joint_sample_id": source["id"],
        })
        recent.append(group)
    if Counter(row["task_id"] for row in output) != Counter(row["task_id"] for row in joint_rows):
        raise ValueError("Staged Solver changed per-task exposure counts")
    return output


def staged_editor_rows(source_rows: list[dict], metadata: list[dict], solver_version: str) -> list[dict]:
    if sorted(row["start"] for row in metadata) != list(range(0, 240, 10)):
        raise ValueError("The frozen W1 reward baseline must cover all24 complete training cohorts")
    by_start = {}
    for row in metadata:
        validate_metadata(row, solver_version)
        by_start[row["start"]] = row
    rows = []
    for index, source in enumerate(source_rows):
        baseline = copy.deepcopy(by_start[source["task_id"] // 10 * 10])
        baseline["prompt_evidence_scope"] = "Unchanged historical W0 messages; reward baseline is authentic W1"
        rows.append({
            "id": f"staged-editor-{index:05d}", "messages": copy.deepcopy(source["messages"]),
            "metadata_json": json.dumps(baseline, sort_keys=True, separators=(",", ":")),
            "partner_version": 1, "replay": None, "source_joint_sample_id": source["id"],
        })
    return rows


def staged_assignments(rows: list[dict], harness: str) -> list[dict]:
    groups = {}
    for row in rows:
        groups.setdefault(row["task_name"], set()).add(row["task_id"])
    assignments = [{"name": name, "task_ids": sorted(tasks), "harness": harness}
                   for name, tasks in groups.items()]
    all_ids = [task for row in assignments for task in row["task_ids"]]
    if len(all_ids) != len(set(all_ids)) or set(all_ids) != set(range(240)):
        raise ValueError("Staged environment workers must partition all240 original training tasks")
    return assignments


def prepare(root: Path) -> dict:
    os.umask(0o002)
    if ROOT.parent.name != "source-snapshots":
        raise ValueError("Prepare the mechanism study from an independent immutable source snapshot")
    if (root / "PROTOCOL.json").exists():
        raise ValueError("Never refreeze a prepared single-seed mechanism study")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o2775)
    source_rows, original_w0 = initial_inputs(root / "historical-w0-inputs", 96, 8)
    old_state = read_json(OLD / "INITIAL_STATE.json")
    early = read_json(ROOT / "runs/alfworld_strong_sao_ma_v1/common/INITIAL_STATE.json")
    h1_record_path = ROOT / "runs/alfworld_engineer_fast_v1_s42/training/optimizer/trained.json"
    w1_record_path = ROOT / "runs/alfworld_solver_fast_v4_s42/training/optimizer/trained.json"
    h1_record, w1_record = read_json(h1_record_path), read_json(w1_record_path)
    w1 = w1_record["checkpoints"]
    if (any(record.get("central_critic") or record.get("joint_root") for record in (h1_record, w1_record))
            or checkpoint_pair(h1_record, 15)["critic"] != early["engineer_critic"]):
        raise ValueError("Initial Critics must come from the explicitly identified pre-MA ordinary SAO lineage")
    baseline = ROOT / "runs/alfworld_alternating_e15_w79_8h_v1/common/partner_w1"
    baseline_receipt = read_json(baseline / "COMPLETE.json")
    baseline_identity = read_json(baseline / "identity.json")
    if baseline_receipt["status"] != "completed" or Path(baseline_receipt["solver_model"]) != Path(w1["actor"]):
        raise ValueError("The ordinary-SAO reward baseline belongs to a different W1")
    _, weights = model_digests(Path(w1["actor"]))
    if weights != baseline_receipt["solver_weights_sha256"]:
        raise ValueError("W1 weights changed since the authentic reward-baseline collection")
    state = {
        **original_w0, "engineer": old_state["engineer"], "engineer_critic": early["engineer_critic"],
        "solver": w1["actor"], "solver_critic": w1["critic"],
        "solver_version": baseline_identity["solver_version"], "solver_epoch_version": 1,
        "harness": old_state["harness"], "solver_weights_sha256": weights,
        "historical_packet_solver": original_w0["solver"],
        "initialization": (
            "Common official H0 Actor, pre-MA W1 Actor, raw ordinary-H1/W1 critics and fresh Adam. "
            "No trained-MA critic, ValueNorm state, or selected97% checkpoint is inherited. "
            "Historical W0 prompt messages remain identical across methods."
        ),
    }
    for key in ("engineer_critic", "solver_critic"):
        path = Path(state[key])
        if not (path / "model.safetensors").is_file() or (path / "joint_value_norm.json").exists():
            raise ValueError("Every arm requires the same already-saved raw-unit pre-MA critic")
    _immutable_json(root / "INITIAL_STATE.json", state)
    joint_rows = build_rows(source_rows, cycles=48, batch_size=8)
    _immutable_json(root / "schedule_source_data.json", source_rows)
    _immutable_json(root / "joint_data.json", joint_rows)
    core = read_json(CORE)
    validate_cohort(core)
    base = read_json(OLD / "protocol.json")
    acceleration = {key: read_json(OLD / "solver_spec.json")[key] for key in (
        "training_kernel_overlay", "training_kernel_versions", "training_kernel_validation",
    )}
    protocol = {
        "status": "frozen_before_new_training", "source": str(ROOT), "root": str(root),
        "arms": list(ARMS), "training_seed": 1000043, "training_seeds": [1000043],
        "study_revision": 2, "unit_suffix": "single-seed-v2",
        "signal_tracing": True,
        "joint_rounds": 144, "batch_size": 8, "curve_rounds": list(ROUNDS),
        "primary_endpoint": 144, "schedule": "joint_joint_anchor", "anchor_mode": "sampled_control",
        "initial_state": str(root / "INITIAL_STATE.json"), "core200_protocol": str(CORE),
        "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
        "h2": base["h2"],
        "evaluation": {**base["evaluation"], "eval_concurrency": 128, "environment_concurrency": 32,
                       "matrix_concurrency": 3},
        "incumbent_harness": state["harness"], "incumbent_min_gain": 2 / 48,
        "per_role_consumed": {"editor": 768, "solver": 1152},
        "per_role_actor_updates": {"editor": 93, "solver": 141},
        "per_role_critic_updates": {"editor": 192, "solver": 288},
        "per_arm_budgets": {arm: arm_budgets(arm) for arm in ARMS},
        "critic_warmup_active_rounds": 3, "learning_rate": 5e-7, "critic_learning_rate": 2.5e-6,
        "kl_ctl": .02, "reference_storage_dtype": "bfloat16",
        "prompt_invariance": "All arms collect identical1152 ordered inputs with originalW0 Engineer messages.",
        "solver_exposure": "Identical1152 ordered Solver task exposures in every arm.",
        "schedule_control": (
            "Only activate Engineer optimization on the original third rounds. Keep sampled-control routing, "
            "source repetition, third-round unit teammate correction, reward and critic inputs unchanged. "
            "Legacy routing metadata describes baseline eligibility; native_engineer_round_active and "
            "optimizer metrics describe executed updates. Consumed role trajectories increase1920 to2304 "
            "(20%); generated trajectories and environment episodes remain equal."
        ),
        "staged_sao_comparator_excluded_by_user": True,
        "new_training_uses_no_fresh100": True,
        "resource_gate": [
            str(ROOT / "runs/alfworld_w3h3_from_w2_early79_v1/FINAL_AUDIT.json"),
            str(ROOT / "runs/alfworld_ma_engineer_weights_w0_v1/FINAL_AUDIT.json"),
        ],
        "native_gate": (
            "Verified full32K full/no-context/no-ValueNorm evidence plus real6-round full/always-joint "
            "paired canaries, including third-round native updates and unchanged peer correction."
        ),
        "limitations": [
            "Single training seed at user request; question bootstrap is not training-seed robustness.",
            "Original200 and selection48 are reused development questions.",
            "New full control does not reproduce the historical97% initialization with already-trained MA critics.",
            "Always-joint matches generated/environment budgets, not consumed updates or total compute.",
            "Historical97% also differs in the round48 Adam reset and24 versus16 proposal-selection opportunities.",
            "No main200 checkpoint selection or performance-based retries.",
        ],
        "at_unix": time.time(),
    }
    _immutable_json(root / "PROTOCOL.json", protocol)
    common_files = [
        root / "PROTOCOL.json", root / "INITIAL_STATE.json", root / "schedule_source_data.json",
        root / "joint_data.json",
        baseline / "identity.json", baseline / "COMPLETE.json", baseline / "baseline_metadata.json",
        Path(state["packets"]), Path(state["harness"]), CORE,
        Path(protocol["evaluation"]["task_manifest"]), Path(protocol["no_patch"]),
        Path(protocol["h2"]["template"]), Path(acceleration["training_kernel_validation"]),
        h1_record_path, w1_record_path, OLD / "INITIAL_STATE.json", OLD / "protocol.json",
        OLD / "editor_spec.json", OLD / "solver_spec.json",
        ROOT / "runs/alfworld_strong_sao_ma_v1/common/INITIAL_STATE.json",
    ]
    small_pins = file_manifest(common_files)
    for arm in ARMS:
        directory = root / arm
        for role in ("editor", "solver"):
            spec = read_json(OLD / f"{role}_spec.json")
            model = state["engineer" if role == "editor" else "solver"]
            spec.update(
                run=str(directory / role / "optimizer"), joint_root=str(directory / "joint"),
                model=model, critic_model=state["engineer_critic" if role == "editor" else "solver_critic"],
                reference_model=model, experiment=f"{root.name}-{arm}", steps=144, batch_size=8,
                harness=protocol["no_patch"],
                data=str(root / "joint_data.json"), schedule_source_data=str(root / "schedule_source_data.json"),
                anchor_harness=state["harness"], anchor_harness_sha256=file_sha(Path(state["harness"])),
                seed=protocol["training_seed"], critic_warmup_steps=3, save_interval_steps=48,
                peer_weight_mode="unit" if arm == "without_peer" else "bounded",
                critic_cooperative_context=arm != "without_context",
                value_normalization=arm != "without_value_norm", component_ablation=arm,
                editor_update_every_round=arm == "always_joint",
                stop_on_diagnostic_thresholds=False, training_deadline_unix=time.time() + 72 * 3600,
                phase_timeout_seconds=8 * 3600, pinned_files=small_pins, **acceleration,
            )
            validate_spec(spec)
            _immutable_json(directory / f"{role}_spec.json", spec)
            common_files.append(directory / f"{role}_spec.json")
    models = {state[key] for key in ("engineer", "engineer_critic", "solver", "solver_critic")}
    models.add(original_w0["solver"])
    for model in models:
        common_files.extend(path for path in Path(model).iterdir() if path.is_file())
    common_files.extend((ROOT / "whale_local").rglob("*.py"))
    _immutable_json(root / "PINS.json", {"pins": file_manifest(common_files), "at_unix": time.time()})
    return protocol


def verify_inputs(root: Path) -> None:
    expected = read_json(root / "PINS.json")["pins"]
    actual = file_manifest(expected)
    for path, digest in expected.items():
        if actual[path] != digest:
            raise ValueError(f"Frozen mechanism study input changed: {path}")


def train_joint_role(root: Path, arm: str, role: str) -> dict:
    if arm not in ARMS or role not in ("editor", "solver"):
        raise ValueError("Unknown native component arm or role")
    result = run_role(root / arm / f"{role}_spec.json")
    share_checkpoints(root / arm / role / "optimizer", result)
    _immutable_json(root / arm / f"{role}_SHARED.json", {"result": result, "ready": True})
    return result


def choose_harness(proposals: list[dict], reports: dict, *, label: str, margin: float) -> str:
    if len(proposals) != 8 or [row["sample_index"] for row in proposals] != list(range(8)):
        raise ValueError("Every endpoint must retain exactly eight fixed proposal slots")
    incumbent = reports["incumbent_harness"]["accuracy"]
    qualified = [
        f"{label}-proposal{row['sample_index']:02d}" for row in proposals
        if row["valid"] and reports[f"{label}-proposal{row['sample_index']:02d}"]["accuracy"]
        - incumbent >= margin - 1e-12
    ]
    return min(qualified, key=lambda name: (-reports[name]["accuracy"], name)) if qualified else "incumbent_harness"


def select_at_pair(launch: Launcher, root: Path, pair: dict, output: Path, incumbent: str) -> dict:
    protocol = read_json(root / "PROTOCOL.json")
    state = read_json(root / "INITIAL_STATE.json")
    values = generate_proposals(launch, protocol, state, {"candidate": pair["editor"]},
                                output / "generation")["candidate"]
    harnesses = {"incumbent_harness": incumbent}
    for value in values:
        name = f"candidate-proposal{value['sample_index']:02d}"
        path = output / "patches" / f"{name}.json"
        _immutable_json(path, value["patch"] if value["valid"] else read_json(Path(protocol["no_patch"])))
        harnesses[name] = str(path)
    config = matrix_config(protocol, pair["solver"]["actor"], output.name, harnesses, output / "selection")
    config["episode_checkpointing"] = True
    matrix, path = run_matrix(launch, config, output / "selection", 1)
    reports = matrix_reports(matrix, protocol)
    chosen = choose_harness(values, reports, label="candidate", margin=protocol["incumbent_min_gain"])
    selection = {
        "harness": harnesses[chosen], "label": chosen, "new_harness_selected": chosen != "incumbent_harness",
        "valid_proposals": sum(value["valid"] for value in values),
        "selection_scores": {name: report["accuracy"] for name, report in reports.items()},
        "selection_matrix": str(path), "pair": pair, "at_unix": time.time(),
        "main200_used_to_select": False,
    }
    _immutable_json(output / "SELECTED.json", selection)
    return selection


def evaluate_pair(root: Path, arm: str, rounds: int, pair: dict, *, incumbent: str,
                  fixed_selection: dict | None = None) -> dict:
    if arm not in ARMS or rounds not in ROUNDS:
        raise ValueError("Only the two frozen endpoints of declared arms may be evaluated")
    release = gpu_release_audit()
    if not research_resources_released(release):
        raise RuntimeError(f"Endpoint evaluation requires a released node: {release}")
    protocol = read_json(root / "PROTOCOL.json")
    output = root / arm / f"round{rounds:04d}"
    if (output / "RESULT.json").exists():
        raise ValueError("Do not repeat an endpoint after observing its200-task outcomes")
    launch = Launcher(output / "supervision")
    try:
        if fixed_selection is None:
            selection = select_at_pair(launch, root, pair, output, incumbent)
        else:
            if arm != "without_team_learning":
                raise ValueError("Only staged SAO fixes its previously selected Harness during Solver learning")
            selection = {**fixed_selection, "pair": pair, "reused_staged_harness": True}
            _immutable_json(output / "SELECTED.json", selection)
        cohort = read_json(Path(protocol["core200_protocol"]))
        config = {
            **matrix_config(protocol, pair["solver"]["actor"], f"{arm}-{rounds}",
                            {"selected": selection["harness"]}, output / "evaluation200"),
            "task_ids": cohort["all_task_ids"], "samples": 1,
            **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
            "matrix_concurrency": 2,
            "episode_checkpointing": True,
        }
        opened = time.time()
        matrix, matrix_path = run_matrix(launch, config, output / "evaluation200", 1)
        scores = {}
        for name, value in matrix["variants"].items():
            path = Path(value["report"])
            report = checked_native_report(read_json(path), cohort["all_task_ids"], config)
            scores[name] = {
                "successes": int(sum(row["reward"] for row in report["results"])),
                "accuracy": report["accuracy"], "questions": 200,
                "report": str(path), "sha256": file_sha(path),
            }
        result = {
            "status": "completed", "arm": arm, "rounds": rounds, "selection": selection,
            "scores": scores, "main_matrix": str(matrix_path), "main_started_at_unix": opened,
            "physical_main_episodes": matrix["unique_harnesses"] * 200,
            "primary_endpoint": rounds == 144, "at_unix": time.time(),
        }
        if selection["at_unix"] >= opened:
            raise ValueError("Harness choice was not fixed before opening the200-task matrix")
    finally:
        launch.close()
        write_json(output / "RELEASED.json", gpu_release_audit())
    checked_release(output)
    _immutable_json(output / "RESULT.json", result)
    return result


def audit_joint_arm(root: Path, arm: str, *, rounds: int = 144) -> dict:
    protocol = read_json(root / "PROTOCOL.json")
    summaries = {}
    directory = root / arm
    for role in ("editor", "solver"):
        spec = read_json(directory / f"{role}_spec.json")
        validate_spec(spec)
        result = read_json(directory / role / "COMPLETE.json")
        checked_release(directory / role)
        metrics = metric_rows(Path(spec["run"]) / "metrics.jsonl")
        counts = update_counts(spec, rounds)
        expected = arm_budgets(arm, rounds=rounds)["roles"][role]
        if (spec["steps"] != rounds or result["status"] != "completed" or len(metrics) != rounds
                or [row["global_step"] for row in metrics] != list(range(rounds))
                or result["actor_updates"] != expected["actor_steps"]
                or result["critic_updates"] != expected["critic_steps"]
                or result["consumed_policy_trajectories"] != expected["consumed_trajectories"]):
            raise ValueError("A native ablation did not complete its matched update and consumed-trajectory budgets")
        tokens, invalid = 0, 0
        for step in range(rounds):
            if role_active(spec, step):
                expected_unit = float(arm == "without_peer" or step % 3 == 2)
                if metric_value(metrics[step]["metrics"], "joint_policy/peer_unit_weight_mode") != expected_unit:
                    raise ValueError("Executed teammate correction differs from the declared component arm")
            for slot in range(8):
                current = directory / "joint" / f"round-{step:04d}"
                receipt = read_json(current / f"{role}-receipt-{slot:03d}.json")
                outcome = read_json(current / f"outcome-{slot:03d}.json")
                expected_unit = arm == "without_peer" or step % 3 == 2
                if (
                    receipt["reward"] != outcome["reward"] or receipt["team_reward"] != outcome["reward"]
                    or receipt["effective_peer_weight_mode"] != ("unit" if expected_unit else "bounded")
                    or receipt["editor_reward_mode"] != "team"
                    or receipt["native_engineer_round_active"] != role_active(spec, step, "editor")
                ):
                    raise ValueError("A joint episode lost shared terminal reward or peer-correction identity")
                generated = sum(len(call["response_ids"]) for call in receipt["calls"])
                if generated != receipt["generated_tokens"] or not 0 < generated <= 4096:
                    raise ValueError("A component arm changed its real-action token accounting")
                tokens += generated
                invalid += receipt["model_error"] is not None
        summaries[role] = {
            **counts, "consumed_trajectories": result["consumed_policy_trajectories"],
            "generated_trajectories": result["generated_policy_trajectories"],
            "recorded_action_tokens": tokens, "invalid_harness_episode_count": invalid,
            "elapsed_seconds": result["elapsed_seconds"],
        }
    episodes = rounds * protocol["batch_size"]
    if sum(1 for _ in (directory / "joint").glob("round-*/outcome-*.json")) != episodes:
        raise ValueError("The joint ablation did not generate its exact shared environment budget")
    audit = {"status": "passed", "arm": arm, "roles": summaries,
             "shared_environment_episodes": episodes, "at_unix": time.time()}
    _immutable_json(directory / "TRAINING_AUDIT.json", audit)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train-role", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--role", choices=("editor", "solver"))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare(root)
    elif args.action == "train-role":
        result = train_joint_role(root, args.arm, args.role)
    else:
        result = audit_joint_arm(root, args.arm)
    print({"status": result["status"], "action": args.action}, flush=True)


if __name__ == "__main__":
    main()
