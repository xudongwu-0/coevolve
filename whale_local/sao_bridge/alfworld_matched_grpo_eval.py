"""Select a final-budget GRPO harness on W1, then deploy it unchanged on W0/W1."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL, SAO_CONFIG
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .budget import contract_digest
from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit

BASE_PROTOCOL = ROOT / "runs/alfworld_core_quick_20261002_v1/protocol.json"
INITIAL_STATE = ROOT / "runs/alfworld_alternating_e15_w79_8h_v1/common/INITIAL_STATE.json"


def wait_artifact(path: Path, training: Path, root: Path, timeout: float = 21600) -> dict:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        for failed in (training / "FAILED.json", root / "w0_FAILED.json", root / "w1_FAILED.json"):
            if failed.exists():
                raise RuntimeError(f"Matched experiment failed: {failed}: {read_json(failed)}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(5)
    return read_json(path)


def choose_harness(reports: dict, proposals: list[dict]) -> str:
    if [p["sample_index"] for p in proposals] != list(range(8)):
        raise ValueError("Selection must retain all eight proposal slots")
    names = [f"grpo-{i:02d}" for i in range(8)]
    if any(name not in reports for name in names):
        raise ValueError("Selection is missing proposal outcomes")
    return min(names, key=lambda name: (-reports[name]["accuracy"], name))


def evaluate(training: Path, root: Path, role: str) -> dict:
    if role not in {"w0", "w1"}:
        raise ValueError("Unknown matched evaluation role")
    root.mkdir(parents=True, exist_ok=True)
    launch = None
    result = None
    try:
        receipt = wait_artifact(training / "TRAINING_COMPLETE.json", training, root)
        if (receipt["updates_completed"] != 6 or receipt["optimizer_steps"] != 24
                or receipt["consumed_policy_trajectories"] != 192):
            raise ValueError("Evaluation requires the complete matched GRPO budget")
        wait_artifact(training / ("target/gpu_release.json" if role == "w0"
                                 else "training_gpu_release.json"), training, root)
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Evaluation requires free GPUs and an active filler")
        protocol, state, cohort = read_json(BASE_PROTOCOL), read_json(INITIAL_STATE), read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        _immutable_json(root / f"{role}_IDENTITY.json", {
            "training_receipt_sha256": file_sha(training / "TRAINING_COMPLETE.json"),
            "proposal_protocol_sha256": file_sha(BASE_PROTOCOL),
            "state_sha256": file_sha(INITIAL_STATE),
            "cohort_sha256": file_sha(EVAL_PROTOCOL),
            "engineer_checkpoint": "checkpoint-000006",
            "checkpoint_selection": "final 192-consumed checkpoint fixed before all evaluation",
            "harness_selection": "eight proposals, W1 selection24 mean@2, ties by slot",
            "deployment": "same selected harness on both W0 and W1; W0 is transfer",
        })
        launch = Launcher(root / role / "supervision")
        if role == "w0":
            actor = training / "engine/checkpoint-000006/actor"
            from harness_r1_searchqa.workflow import checkpoint_identity
            if checkpoint_identity(actor)["sha256"] != receipt["actor_sha256"]:
                raise ValueError("Final GRPO actor differs from its committed identity")
            proposals = generate_proposals(
                launch, protocol, state, {"grpo": {"actor": str(actor)}}, root / "generation",
            )["grpo"]
            harnesses = {}
            for proposal in proposals:
                label = f"grpo-{proposal['sample_index']:02d}"
                path = Path(protocol["no_patch"])
                if proposal["valid"]:
                    path = root / "patches" / f"{label}.json"
                    _immutable_json(path, proposal["patch"])
                harnesses[label] = str(path)
            _immutable_json(root / "PROPOSALS.json", {
                "proposals": proposals, "harnesses": harnesses,
                "actor_sha256": receipt["actor_sha256"],
            })
        shared = wait_artifact(root / "PROPOSALS.json", training, root)
        solver = read_json(SAO_CONFIG)["solver_model"] if role == "w0" else state["solver"]
        quick = matrix_config(protocol, solver, f"{role}-matched-grpo",
                              shared["harnesses"], root / role / "selection")
        if role == "w1":
            matrix, output = run_matrix(launch, quick, root / role / "selection", 1)
            reports = matrix_reports(matrix, protocol)
            choice = choose_harness(reports, shared["proposals"])
            _immutable_json(root / f"{role}_QUICK.json", {
                "scores": {label: report["accuracy"] for label, report in reports.items()},
                "proposal_mean": sum(reports[f"grpo-{i:02d}"]["accuracy"] for i in range(8)) / 8,
                "best_slot_diagnostic": choice, "matrix": str(output),
                "valid_proposals": sum(p["valid"] for p in shared["proposals"]),
            })
            _immutable_json(root / "SELECTED.json", {
                "label": choice, "harness": shared["harnesses"][choice],
                "harness_sha256": contract_digest(read_json(Path(shared["harnesses"][choice]))),
                "selection_accuracy": reports[choice]["accuracy"],
                "selection_questions": 24, "selection_samples": 2,
            })
        selected = wait_artifact(root / "SELECTED.json", training, root)
        if contract_digest(read_json(Path(selected["harness"]))) != selected["harness_sha256"]:
            raise ValueError("Selected harness changed before deployment")
        config = {
            **quick, "run": str(root / role / "evaluation200"),
            "harnesses": {"grpo_h1": selected["harness"]},
            "task_ids": cohort["all_task_ids"], "samples": 1,
            "seed": cohort["seed"], "temperature": cohort["temperature"],
            "solver_tokens": cohort["solver_tokens"],
            "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 2,
        }
        matrix, output = run_matrix(launch, config, root / role / "evaluation200", 1)
        checked = {}
        sources = {}
        for label, variant in matrix["variants"].items():
            path = Path(variant["report"])
            checked[label] = checked_native_report(read_json(path), cohort["all_task_ids"], {
                "samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"],
            })
            shared_path = root / f"{role}_{label}_report.json"
            compact = {
                **checked[label],
                "results": [{key: row[key] for key in (
                    "id", "task_id", "sample_index", "rollout_seed", "reward", "tokens",
                )} for row in checked[label]["results"]],
            }
            _immutable_json(shared_path, compact)
            sources[label] = {
                "path": str(shared_path), "sha256": file_sha(shared_path),
                "native_path": str(path), "native_sha256": file_sha(path),
            }
        result = {
            "status": "completed", "role": role, "sources": sources,
            "selected": selected, "matrix": str(output),
            "cohorts": {
                name: {
                    label: {"accuracy": subset_report(report, ids)["accuracy"],
                            "successes": sum(r["reward"] for r in subset_report(report, ids)["results"])}
                    for label, report in checked.items()
                }
                for name, ids in (("all200", cohort["all_task_ids"]),
                                  ("strict197", cohort["strict_goal_disjoint_task_ids"]))
            },
            "finished_at_unix": time.time(),
        }
    except BaseException as exc:
        write_json(root / f"{role}_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
        write_json(root / f"{role}_RELEASED.json", gpu_release_audit())
    _immutable_json(root / f"{role}_COMPLETE.json", result)
    return result


def summarize(training: Path, root: Path) -> dict:
    roles = {role: wait_artifact(root / f"{role}_COMPLETE.json", training, root)
             for role in ("w0", "w1")}
    if roles["w0"]["selected"] != roles["w1"]["selected"]:
        raise ValueError("W0 and W1 did not deploy the same frozen harness")
    reports = {}
    for role, value in roles.items():
        path = Path(value["sources"]["grpo_h1"]["path"])
        if file_sha(path) != value["sources"]["grpo_h1"]["sha256"]:
            raise ValueError("Native report changed after evaluation")
        reports[role] = read_json(path)
    old = ROOT / "runs/alfworld_core_200_oneshot_v2/merged"
    comparisons = {
        f"{role}_grpo_minus_sao_h1": paired_difference(
            read_json(old / f"{role}_h1.json"), reports[role], seed=9000045)
        for role in ("w0", "w1")
    }
    result = {
        "status": "completed", "roles": roles, "comparisons": comparisons,
        "w1_grpo_minus_w1_h2": paired_difference(
            read_json(old / "w1_h2.json"), reports["w1"], seed=9000045),
        "limitations": [
            "Same H0 and 192 sampled training trajectories; not matched tokens, reward episodes or FLOPs.",
            "SAO retained its selected step15; GRPO uses the predeclared final 192-consumed checkpoint.",
            "One W1-selected patch is deployed unchanged on W0; this is a transfer evaluation.",
            "Selection24 is included in development200; strict197 removes goal-family overlap, not tuning exposure.",
            "Question-level bootstrap is not training-seed robustness or a selection-adjusted confidence interval.",
        ],
        "finished_at_unix": time.time(),
    }
    _immutable_json(root / "COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--role", choices=("w0", "w1", "summarize"), required=True)
    args = parser.parse_args()
    if args.role == "summarize":
        summarize(args.training.resolve(), args.root.resolve())
    else:
        evaluate(args.training.resolve(), args.root.resolve(), args.role)


if __name__ == "__main__":
    main()
