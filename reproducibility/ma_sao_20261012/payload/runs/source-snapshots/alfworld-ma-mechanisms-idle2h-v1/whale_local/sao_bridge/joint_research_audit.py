"""Recheck the frozen convergence curve before unlocking downstream research."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import checked_matrix, matrix_reports
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .anchor_retention_study import audit_training
from .common import ROOT, file_sha, read_json
from .joint_campaign import select_pair
from .joint_continuation import convergence_status
from .joint_schedule import update_counts


def research_gate(points: list[dict], comparison: dict) -> dict:
    state = convergence_status(points)
    if comparison["control_accuracy"] != 0.92 or comparison["candidate_accuracy"] != points[-1]["accuracy"]:
        raise ValueError("Research gate comparison does not describe the final endpoint and historical92%")
    low, high = comparison["paired_question_bootstrap_95pct"]
    if not -1 <= low <= high <= 1:
        raise ValueError("Invalid paired confidence interval")
    criteria = {
        "operational_plateau": state["operational_plateau"],
        "last_four_strictly_above_historical": len(points) >= 4 and all(
            p["accuracy"] > 0.92 for p in points[-4:]
        ),
        "final_paired_lower_bound_positive": low > 0,
    }
    passed = all(criteria.values())
    return {
        "criteria": criteria, "working_advantage_gate_passed": passed,
        "next_phase": "freeze_candidate_and_design_matched_efficiency" if passed
        else "diagnose_then_one_frozen_candidate",
        "convergence": state,
        "interpretation": "Exploratory single-seed reused-development gate, not confirmatory superiority",
    }


def audit_optimizer(block: Path, block_index: int, *, inherited_steps: dict | None = None) -> dict:
    if type(block_index) is not int or block_index < 1:
        raise ValueError("Invalid optimizer block index")
    if inherited_steps is not None and (
        set(inherited_steps) != {"editor", "solver"}
        or any(
            set(steps) != {"actor", "critic"}
            or any(type(value) is not int or value < 1 for value in steps.values())
            for steps in inherited_steps.values()
        )
    ):
        raise ValueError("Inherited optimizer steps must describe both trained Actor/Critic pairs")
    details = {}
    for role in ("editor", "solver"):
        initial = inherited_steps[role] if inherited_steps is not None else {"actor": 0, "critic": 0}
        spec = read_json(block / f"{role}_spec.json")
        if inherited_steps is not None and spec.get("inherited_optimizer_steps") != initial:
            raise ValueError("Optimizer offset differs from the frozen initial checkpoint")
        optimizer = block / role / "optimizer"
        loaded = read_json(optimizer / "OPTIMIZER_LOAD.json")
        native = block_index > 1 or inherited_steps is not None
        mode = "native_dcp_with_adam" if native else "initial_fresh_adam"
        if loaded["mode"] != mode:
            raise ValueError("Unexpected Adam reset")
        saved = read_json(optimizer / "OPTIMIZER_SAVED.json")
        if set(saved["paths"]) != {"actor", "critic"}:
            raise ValueError("Optimizer handoff must contain both actor and critic")
        per_block = update_counts(spec, spec["steps"])
        checks = [(saved["paths"], block_index)]
        if native:
            inputs = spec.get("optimizer_state_input")
            if (not isinstance(inputs, dict) or set(inputs) != {"actor", "critic"}
                    or loaded.get("paths") != inputs):
                raise ValueError("Loaded native optimizer source changed")
            checks.append((inputs, block_index - 1))
        elif spec.get("optimizer_state_input") is not None:
            raise ValueError("Fresh Adam unexpectedly declares native input")
        for paths, completed_blocks in checks:
            for component, path in paths.items():
                fingerprints = {p.name: p for p in Path(path).glob("adam-rank*.json")}
                expected = initial[component] + completed_blocks * per_block[f"{component}_steps"]
                if set(fingerprints) != {f"adam-rank{rank}.json" for rank in range(4)}:
                    raise ValueError("Missing optimizer rank fingerprint")
                for fingerprint in fingerprints.values():
                    value = read_json(fingerprint)
                    if value["min_adam_step"] != expected or value["max_adam_step"] != expected:
                        raise ValueError("Native optimizer step continuity failed")
        details[role] = {"loaded": loaded, "saved": saved,
                         "cumulative_actor_updates": initial["actor"] + block_index * per_block["actor_steps"]}
    return details


def endpoint_matrix_config(stage: Path) -> dict:
    original = read_json(stage / "config-01.json")
    recovery = stage / "INFRASTRUCTURE_RECOVERY.json"
    if not recovery.exists():
        return original
    receipt = read_json(recovery)
    failed = stage / "attempt-01"
    evidence = failed / "supervision/foreign_workload.json"
    if (receipt["reason"] != "cpu_test_gpu_context_before_rollout"
            or receipt["original_config_sha256"] != file_sha(stage / "config-01.json")
            or receipt["failure_sha256"] != file_sha(failed / "FAILED.json")
            or receipt["foreign_workload_sha256"] != file_sha(evidence)
            or receipt["new_config_sha256"] != file_sha(stage / "config-02.json")
            or "watchdog stopped experiment" not in read_json(failed / "FAILED.json")["error"]
            or (failed / "PROGRESS.json").exists() or (failed / "COMPLETE.json").exists()
            or any((failed / "reports").glob("*.json"))):
        raise ValueError("Invalid zero-rollout matrix infrastructure recovery")
    foreign = [row for row in read_json(evidence)["processes"] if row["pid"] == receipt["foreign_pid"]]
    if len(foreign) != 1 or "pytest" not in foreign[0]["cmd"]:
        raise ValueError("Recovery evidence does not identify the temporary CPU test process")
    current = read_json(stage / "config-02.json")
    if (current != {**original, "run": str(stage / "attempt-02")}
            or original["run"] != str(failed)):
        raise ValueError("Matrix recovery changed an input other than the attempt output directory")
    return current


def audit(root: Path) -> dict:
    result = read_json(root / "RESULT.json")
    if result["status"] != "completed" or (root / "FAILED.json").exists():
        raise ValueError("A failed or incomplete run cannot unlock the research gate")
    design = read_json(root / "DESIGN.json")
    points = result["points"]
    state = convergence_status(points)
    if state != result["convergence"] or read_json(root / "CURVE.json")["points"] != points:
        raise ValueError("Persisted curve and convergence result disagree")
    expected_stop = "operational_development_plateau" if state["operational_plateau"] else "budget_exhausted_not_converged"
    requested_stop = result["stop_reason"] == "user_requested_schedule_change_not_converged"
    if requested_stop:
        request = read_json(root / "REQUESTED_BOUNDARY.json")
        if (state["operational_plateau"] or request != result.get("user_requested_boundary")
                or request["last_block"] * 2 != len(points) - 1):
            raise ValueError("User-requested boundary does not match completed blocks")
    if (not requested_stop and result["stop_reason"] != expected_stop) or (len(points) - 1) % 2:
        raise ValueError("Result is not a complete frozen block sequence")
    parent = Path(design["parent"])
    parent_endpoint = design.get("parent_endpoint", "matched")
    initial_rounds = design.get("initial_rounds", 48)
    inherited_steps = design.get("inherited_optimizer_steps")
    historical = read_json(ROOT / "runs/alfworld_core_200_oneshot_v2/merged/w1_h2.json")
    parent_report = read_json(parent / f"{parent_endpoint}_joint_harness_report.json")
    if points[0]["rounds"] != initial_rounds or points[0]["accuracy"] != parent_report["accuracy"]:
        raise ValueError("Convergence parent identity changed")
    cohort = read_json(EVAL_PROTOCOL)
    pins = {str(root / "RESULT.json"): file_sha(root / "RESULT.json")}
    costs = {"consumed_role_trajectories": 0, "generated_role_trajectories": 0,
             "training_environment_episodes": 0, "evaluation_environment_episodes": 0}
    incumbent = read_json(parent / f"{parent_endpoint}_COMPLETE.json")["selection"]["harness"]
    if inherited_steps is not None:
        for role in ("editor", "solver"):
            inputs = read_json(root / "block-01" / f"{role}_spec.json")["optimizer_state_input"]
            if inputs != read_json(parent / role / "optimizer/OPTIMIZER_SAVED.json")["paths"]:
                raise ValueError("Initial native state is not the selected parent checkpoint")
    endpoints = []
    for index, point in enumerate(points[1:]):
        block_index, local_rounds = index // 2 + 1, 48 * (index % 2 + 1)
        block = root / f"block-{block_index:02d}"
        label = f"round{local_rounds:04d}"
        endpoint_path = block / f"{label}_COMPLETE.json"
        if point["result"] != str(endpoint_path) or point["rounds"] != initial_rounds + 48 * (index + 1):
            raise ValueError("Endpoint is not the predeclared checkpoint")
        endpoint = read_json(endpoint_path)
        selected = endpoint["selection"]
        protocol = read_json(block / "protocol.json")
        if selected["rounds"] != local_rounds or selected["global_step"] != local_rounds - 1:
            raise ValueError("Wrong local checkpoint selected")
        if index % 2 == 0:
            roles = audit_training(block)
            if roles != read_json(block / "TRAINING_AUDIT.json")["roles"]:
                raise ValueError("Training ledger changed after the driver audit")
            audit_optimizer(block, block_index, inherited_steps=inherited_steps)
            for role, counts in roles.items():
                costs["consumed_role_trajectories"] += counts["consumed_policy_trajectories"]
                costs["generated_role_trajectories"] += counts["generated_policy_trajectories"]
            costs["training_environment_episodes"] += 768
        for stage in (f"selection-{local_rounds - 1}", "evaluation200"):
            stage_root = block / label / stage
            config = endpoint_matrix_config(stage_root)
            recovery = stage_root / "INFRASTRUCTURE_RECOVERY.json"
            if recovery.exists():
                pins[str(recovery)] = file_sha(recovery)
            matrix = checked_matrix(Path(config["run"]), config)
            if (config["solver_model"] != selected["pair"]["solver"]["actor"]
                    or config["seed"] != 6000045 or config["temperature"] != 1
                    or config["solver_tokens"] != 4096 or config["samples"] != 1):
                raise ValueError("Endpoint decoding contract changed")
            costs["evaluation_environment_episodes"] += matrix["unique_harnesses"] * matrix["question_count"]
            if stage != "evaluation200":
                if config["harnesses"]["incumbent_harness"] != incumbent:
                    raise ValueError("Harness incumbent was not carried from selection alone")
                reports = matrix_reports(matrix, protocol)
                step, name = select_pair(
                    {str(local_rounds - 1): reports}, incumbent_min_gain=protocol["incumbent_min_gain"],
                )
                if (int(step), name) != (selected["global_step"], selected["patch_label"]):
                    raise ValueError("Harness selection rule changed")
                continue
            if config["task_ids"] != cohort["all_task_ids"] or config["harnesses"] != {"joint_harness": selected["harness"]}:
                raise ValueError("Main evaluation task or harness changed")
            for variant, metadata in matrix["variants"].items():
                native = checked_native_report(read_json(Path(metadata["report"])), cohort["all_task_ids"], config)
                compact_path = block / f"{label}_{variant}_report.json"
                compact = read_json(compact_path)
                pins[str(compact_path)] = file_sha(compact_path)
                if endpoint["reports"][variant]["sha256"] != pins[str(compact_path)]:
                    raise ValueError("Compact report hash changed")
                if compact["results"] != [{k: row[k] for k in compact["results"][0]} for row in native["results"]]:
                    raise ValueError("Compact report lost native evidence")
                score = point["accuracy" if variant == "joint_harness" else "no_patch_accuracy"]
                if score != native["accuracy"] or score != endpoint["scores"][variant]["all200"]["accuracy"]:
                    raise ValueError("Curve score disagrees with native results")
        candidate = read_json(block / f"{label}_joint_harness_report.json")
        comparison = paired_difference(historical, candidate, seed=9000045)
        parent_comparison = paired_difference(parent_report, candidate, seed=20261007)
        if comparison != point["versus_historical_sao"] or comparison != endpoint["versus_w1_h2"]:
            raise ValueError("Historical paired comparison changed")
        if parent_comparison != point["versus_parent"]:
            raise ValueError("Parent paired comparison changed")
        incumbent = selected["harness"]
        if incumbent != point["harness"]:
            raise ValueError("Curve harness identity changed")
        endpoints.append({"rounds": point["rounds"], "comparison": comparison,
                          "selected_harness": incumbent, "patch_label": selected["patch_label"]})
    if not endpoints:
        raise ValueError("No completed continuation endpoints")
    for key in ("consumed_role_trajectories", "generated_role_trajectories", "training_environment_episodes"):
        if costs[key] != result[f"new_{key}"]:
            raise ValueError(f"Run cost disagrees with receipts: {key}")
    if "transport_recovery" in design:
        failed = read_json(Path(design["transport_recovery"]["failed_cost"]))
    else:
        if design.get("research_schedule") != "simultaneous":
            raise ValueError("Run omitted its explicit predecessor cost contract")
        failed = {"consumed_total": 0, "scope": "New candidate; prior mixed study is separate research cost",
                  "prior_study": design["comparison_study"]}
    for record in result["release"].values():
        if record["foreign_gpu_pids"] or not record["filler_unit_active"]:
            raise ValueError("Run did not release research resources")
    return {
        "status": "passed", "at_unix": time.time(), "evidence_pins": pins,
        "endpoints": endpoints, "successful_continuation_cost": costs,
        "failed_predecessor_cost": failed,
        "continuation_attempts_consumed_including_failure": costs["consumed_role_trajectories"] + failed["consumed_total"],
        "cost_scope": "This continuation and its transport predecessor; inherited parent and native engineering validation are separate",
        "research_gate": research_gate(points, endpoints[-1]["comparison"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    _immutable_json(args.root / "FINAL_AUDIT.json", result)
    print(result["research_gate"])


if __name__ == "__main__":
    main()
