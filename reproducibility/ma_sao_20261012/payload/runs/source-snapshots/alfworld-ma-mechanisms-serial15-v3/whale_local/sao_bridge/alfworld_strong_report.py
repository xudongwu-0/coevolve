"""Open matched development reports only after both strong arms are frozen."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

from .analysis import difference_in_differences, paired_difference
from .alfworld_strong_phase import checked_native_report, checked_release
from .alfworld_strong_select import (
    SelectionAttemptsExhausted, matrix_config, matrix_reports, run_matrix,
)
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node, stop_known


def frozen_states(protocol_path: Path) -> tuple[dict, dict, dict]:
    protocol = read_json(protocol_path)
    root = Path(protocol["run"])
    states, receipts = {}, {}
    for arm in ("sequential", "ma"):
        path = root / arm / "ARM_FROZEN.json"
        if not path.is_file():
            raise FileNotFoundError(f"Cannot open report before both arms freeze: {path}")
        receipt = read_json(path)
        state_path = root / arm / "FINAL_STATE.json"
        if (
            receipt["status"] != "frozen" or receipt["report_opened"]
            or receipt["protocol_sha256"] != file_sha(protocol_path)
            or receipt["final_state_sha256"] != file_sha(state_path)
            or receipt["phase_order"] != protocol["phase_orders"][arm]
        ):
            raise ValueError("Final arm freeze identity changed")
        state = read_json(state_path)
        if (
            state["protocol_sha256"] != receipt["protocol_sha256"]
            or state["solver"] != receipt["solver_model"]
            or state["harness_sha256"] != receipt["harness_sha256"]
            or contract_digest(read_json(Path(state["harness"]))) != receipt["harness_sha256"]
        ):
            raise ValueError("Frozen model/harness identity changed")
        states[arm], receipts[arm] = state, receipt
    return protocol, states, receipts


def wait_frozen(protocol_path: Path, timeout: float = 86400) -> None:
    protocol = read_json(protocol_path)
    root = Path(protocol["run"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all((root / arm / "ARM_FROZEN.json").is_file() for arm in ("sequential", "ma")):
            frozen_states(protocol_path)
            return
        time.sleep(30)
    raise TimeoutError("Both arms have not frozen; report remains closed")


def wait_reports(protocol_path: Path, timeout: float = 86400) -> None:
    root = Path(read_json(protocol_path)["run"])
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(
            (root / arm / "final_report" / name).is_file()
            for arm in ("sequential", "ma") for name in ("COMPLETE.json", "RELEASED.json")
        ):
            return
        time.sleep(30)
    raise TimeoutError("Both frozen report jobs have not completed and released GPUs")


def report_config(protocol: dict, states: dict, root: Path, solver: str, label: str) -> dict:
    arm = label.removeprefix("frozen-")
    if protocol["evaluation"].get("report_mode") == "final_systems" and arm in states:
        return matrix_config(
            protocol, solver, label, {f"{arm}_final": states[arm]["harness"]},
            root, split="report",
        )
    return matrix_config(
        protocol, solver, label, {
            "common_h1": str(Path(protocol["run"]) / "common/selected_harness.json"),
            "sequential_final": states["sequential"]["harness"],
            "ma_final": states["ma"]["harness"],
        }, root, split="report",
    )


def publish_reports(root: Path, reports: dict[str, dict]) -> dict:
    paths = {}
    for label, report in reports.items():
        path = root / "reports" / f"{label}.json"
        _immutable_json(path, report)
        paths[label] = {"path": str(path), "sha256": file_sha(path), "accuracy": report["accuracy"]}
    return paths


def run(protocol_path: Path, arm: str, *, wait: bool = False) -> dict:
    require_research_node()
    if arm not in {"sequential", "ma"}:
        raise ValueError("Unknown report arm")
    if wait:
        wait_frozen(protocol_path)
    protocol, states, freezes = frozen_states(protocol_path)
    initial_group = protocol.get("initial_solver_group", "initial_w0")
    if initial_group not in {"initial_w0", "initial_w1"}:
        raise ValueError("Unknown initial Solver report group")
    root = Path(protocol["run"]) / arm / "final_report"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {
            "protocol_sha256": file_sha(protocol_path), "arm": arm,
            "arm_freezes": {
                name: file_sha(Path(protocol["run"]) / name / "ARM_FROZEN.json") for name in states
            },
            "metric": "mean@2", "reused_development_report": True,
        }
        _immutable_json(root / "identity.json", identity)
        if (root / "COMPLETE.json").exists() and (root / "RELEASED.json").exists():
            checked_release(root)
            complete = read_json(root / "COMPLETE.json")
            for group in complete["report_groups"].values():
                for value in group["reports"].values():
                    if file_sha(Path(value["path"])) != value["sha256"]:
                        raise ValueError("Completed report evidence changed")
            return complete
        for name in states:
            if file_sha(Path(protocol["run"]) / name / "ARM_FROZEN.json") != identity["arm_freezes"][name]:
                raise ValueError("Freeze changed before report opening")
        _, weights = model_digests(Path(states[arm]["solver"]))
        if weights != freezes[arm]["solver_weights_sha256"]:
            raise ValueError("Frozen report Solver weights changed")
        _immutable_json(root / "REPORT_OPENING.json", {
            **identity, "solver_weights_sha256": weights,
            "report_task_ids": read_json(Path(protocol["evaluation"]["task_manifest"]))["report_task_ids"],
        })
        if (root / "FAILED.json").exists():
            archive = root / "failed_controller_attempts"
            archive.mkdir(exist_ok=True)
            (root / "FAILED.json").rename(archive / f"{len(list(archive.glob('*.json'))) + 1:03d}.json")
        write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
        launch = Launcher(root / "supervision")
        started = time.monotonic()
        try:
            config = report_config(protocol, states, root / "matrix", states[arm]["solver"], f"frozen-{arm}")
            matrix, output = run_matrix(launch, config, root / "matrix", protocol["max_phase_attempts"])
            reports = matrix_reports(matrix, protocol, split="report")
            groups = {
                arm: {
                    "solver": states[arm]["solver"], "solver_weights_sha256": weights,
                    "matrix": str(output), "reports": publish_reports(root, reports),
                },
            }
            if arm == "ma":
                _, initial_weights = model_digests(Path(protocol["initial_solver"]))
                if (
                    protocol["evaluation"].get("report_mode") != "final_systems"
                    and weights == initial_weights
                    and Path(states[arm]["solver"]).resolve() == Path(protocol["initial_solver"]).resolve()
                ):
                    groups[initial_group] = {**groups[arm], "reuse_reason": "identical frozen Solver path/weights and all report settings"}
                else:
                    initial_config = report_config(
                        protocol, states, root / f"{initial_group}_matrix",
                        protocol["initial_solver"],
                        "frozen-initial-W1" if initial_group == "initial_w1" else "frozen-initial-W0",
                    )
                    initial, initial_output = run_matrix(
                        launch, initial_config, root / f"{initial_group}_matrix", protocol["max_phase_attempts"],
                    )
                    initial_reports = matrix_reports(initial, protocol, split="report")
                    groups[initial_group] = {
                        "solver": protocol["initial_solver"], "solver_weights_sha256": initial_weights,
                        "matrix": str(initial_output),
                        "reports": publish_reports(root / initial_group, initial_reports),
                    }
            complete = {
                "status": "completed", "identity": identity, "report_groups": groups,
                "elapsed_seconds": time.monotonic() - started,
                "interpretation": "Predeclared single-seed comparison on previously used development report tasks, not an untouched or seed-replicated test.",
            }
            write_json(root / "COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
            raise
        finally:
            launch.close()
            write_json(root / "RELEASED.json", {"released_at": time.time(), **gpu_release_audit()})


def compare(protocol_path: Path) -> dict:
    protocol, states, freezes = frozen_states(protocol_path)
    initial_group = protocol.get("initial_solver_group", "initial_w0")
    minimal = protocol["evaluation"].get("report_mode") == "final_systems"
    root = Path(protocol["run"])
    freeze_hashes = {arm: file_sha(root / arm / "ARM_FROZEN.json") for arm in states}
    groups = {}
    for arm in ("sequential", "ma"):
        report_root = root / arm / "final_report"
        checked_release(report_root)
        complete = read_json(report_root / "COMPLETE.json")
        if (
            complete["status"] != "completed"
            or complete["identity"]["protocol_sha256"] != file_sha(protocol_path)
            or complete["identity"]["arm"] != arm
            or complete["identity"]["arm_freezes"] != freeze_hashes
        ):
            raise ValueError("Final report protocol or completion changed")
        for label, group in complete["report_groups"].items():
            if label in groups:
                raise ValueError("Duplicate final report Solver group")
            groups[label] = group
    if initial_group not in {"initial_w0", "initial_w1"} or set(groups) != {"sequential", "ma", initial_group}:
        raise ValueError("Final report omitted a declared Solver group")
    for arm in states:
        if (
            groups[arm]["solver"] != states[arm]["solver"]
            or groups[arm]["solver_weights_sha256"] != freezes[arm]["solver_weights_sha256"]
        ):
            raise ValueError("Final report evaluated a different frozen Solver")
    if Path(groups[initial_group]["solver"]).resolve() != Path(protocol["initial_solver"]).resolve():
        raise ValueError("Initial-model report evaluated a different Solver")
    task_ids = read_json(Path(protocol["evaluation"]["task_manifest"]))["report_task_ids"]
    reports = {}
    for label, group in groups.items():
        values = {}
        for harness, value in group["reports"].items():
            path = Path(value["path"])
            if file_sha(path) != value["sha256"]:
                raise ValueError("Final report changed after publication")
            values[harness] = checked_native_report(read_json(path), task_ids, protocol["evaluation"])
        expected = (
            {"no_patch", f"{label}_final"} if minimal and label in states
            else {"no_patch", "common_h1", "sequential_final", "ma_final"}
        )
        if set(values) != expected:
            raise ValueError("Cross-harness report matrix is incomplete")
        reports[label] = values
    evaluation = protocol["evaluation"]
    difference = paired_difference(
        reports["sequential"]["sequential_final"], reports["ma"]["ma_final"],
        seed=evaluation["bootstrap_seed"], draws=evaluation["bootstrap_draws"],
    )
    interaction = None if minimal else difference_in_differences(
        reports["sequential"]["sequential_final"], reports["ma"]["sequential_final"],
        reports["sequential"]["ma_final"], reports["ma"]["ma_final"],
        seed=evaluation["bootstrap_seed"] + 1, draws=evaluation["bootstrap_draws"],
    )
    matched_budget = all(
        receipt.get("all_phases_completed", True) for receipt in freezes.values()
    )
    identical_final_policy = (
        freezes["sequential"]["solver_weights_sha256"] == freezes["ma"]["solver_weights_sha256"]
        and freezes["sequential"]["harness_sha256"] == freezes["ma"]["harness_sha256"]
    )
    result = {
        "status": "completed",
        "gate_passed": (
            matched_budget and not identical_final_policy
            and difference["candidate_minus_control"] >= evaluation["minimum_ma_gain"]
            and difference["paired_question_bootstrap_95pct"][0] > 0
        ),
        "identical_final_policy": identical_final_policy,
        "matched_completed_phase_budgets": matched_budget,
        "identity_interpretation": (
            "Both arms selected identical Solver weights and harness. Any measured score "
            "difference is evaluation variation, not an algorithmic policy improvement."
            if identical_final_policy else
            "At least one final policy component differs; paired score uncertainty still applies."
        ),
        "paired_final_system": difference,
        "report_matrix": {
            solver: {harness: report["accuracy"] for harness, report in values.items()}
            for solver, values in reports.items()
        },
        "cross_harness_interaction": {
            "status": "not_run", "reason": "Cross-swap ablation excluded from the time-bounded campaign.",
        } if interaction is None else {
            "estimand": "[J(W_ma,H_ma)-J(W_ma,H_seq)]-[J(W_seq,H_ma)-J(W_seq,H_seq)]",
            "difference": interaction["ma_change_minus_sao_change"],
            "paired_question_bootstrap_95pct": interaction["paired_question_bootstrap_95pct"],
            "interpretation": "Matched final-policy interaction contrast, not an isolated causal attribution to cadence or centralized critic.",
        },
        "minimum_ma_gain": evaluation["minimum_ma_gain"],
        "protocol_sha256": file_sha(protocol_path),
        "final_states": {name: state["solver"] for name, state in states.items()},
        "report_completion_hashes": {
            arm: file_sha(root / arm / "final_report/COMPLETE.json") for arm in ("sequential", "ma")
        },
        "limitations": [
            "Single training seed and reused development report; no training-seed uncertainty.",
            "Shared selected prefix; full prior search/training cost is charged to both arms.",
            "Bounded continuation budget, not proof of convergence.",
            (
                "Order and centralized critic change together; no component attribution without follow-up."
                if protocol["phase_orders"]["sequential"] != protocol["phase_orders"]["ma"] else
                "Same H-to-W order; MA adds pre-action centralized critic context. No multi-seed claim."
            ),
            (
                "Both arms completed their assigned phase budgets."
                if matched_budget else
                "At least one phase budget was incomplete; no MA advantage claim is licensed."
            ),
            "No cadence search or WebShop claim is licensed merely by finishing this comparison.",
        ],
    }
    _immutable_json(root / "PAIRED_REPORT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("sequential", "ma"))
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        if args.arm:
            parser.error("--compare cannot be combined with --arm")
        if args.wait:
            wait_reports(args.protocol)
        result = compare(args.protocol)
    else:
        if args.arm is None:
            parser.error("--arm is required to evaluate frozen policies")
        try:
            result = run(args.protocol, args.arm, wait=args.wait)
        except (ValueError, SelectionAttemptsExhausted) as exc:
            raise SystemExit(78) from exc
    print(result, flush=True)


if __name__ == "__main__":
    main()
