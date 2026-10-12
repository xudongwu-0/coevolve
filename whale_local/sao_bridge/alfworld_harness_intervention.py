"""Single-hook ablation of the historical late-MA harness, with fixed weights."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import time

from .alfworld_ma_improvement_study import wait_campaign
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_core_extend import validate_cohort
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import run_matrix, supplement_runtime
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, read_json, write_json, file_sha
from .launch import Launcher
from .resources import gpu_release_audit


def remove_action_guard(patch: dict) -> dict:
    result = copy.deepcopy(patch)
    removed = [a for a in result["actions"]
               if a.get("type") == "add_code_hook" and a.get("hook") == "on_before_action"]
    if len(removed) != 1:
        raise ValueError("Intervention requires exactly one action guard")
    result["actions"] = [a for a in result["actions"] if a is not removed[0]]
    return result


def disable_wrong_object_block(patch: dict) -> dict:
    result = copy.deepcopy(patch)
    hooks = [a for a in result["actions"] if a.get("type") == "add_code_hook"
             and a.get("hook") == "on_before_action"]
    needle = "    if target not in a:\n"
    if len(hooks) != 1 or hooks[0]["code"].count(needle) != 1:
        raise ValueError("Expected exactly one wrong-object branch")
    hooks[0]["code"] = hooks[0]["code"].replace(needle, "    if False and target not in a:\n")
    return result


def prepare(root: Path, campaign: Path) -> dict:
    historical = ROOT / "runs/alfworld_ma_crossplay_v1/protocol.json"
    old = read_json(historical)
    source = Path(old["patches"]["h_late"])
    intervention = root / "h112_without_action_guard.json"
    _immutable_json(intervention, remove_action_guard(read_json(source)))
    model = old["models"]["matched"]
    pins = [historical, source, intervention, EVAL_PROTOCOL, campaign / "protocol.json",
            Path(model) / "config.json", *sorted(Path(model).glob("*.safetensors"))]
    protocol = {
        "protocol": "late-harness-action-guard-ablation-v1", "campaign": str(campaign),
        "queue_timeout_seconds": 8 * 3600, "created_at_unix": time.time(),
        "question": "Does the late action guard contribute to failure by preventing recovery from wrong-object pickup?",
        "evidence": "Synthetic state reproduces contradictory advice/blocking;14/25 old regressions contain the blocker message. Association alone is not causality.",
        "solver": model, "original": str(source), "intervention": str(intervention),
        "intervention_rule": "Remove exactly the single on_before_action hook; retain all other fields/actions byte-equivalent after JSON serialization.",
        "planned_environment_episodes": 600, "new_training_trajectories": 0,
        "primary_contrast": "without_guard minus original, all200 question-paired95% interval",
        "interpretation": "Post-hoc single-hook mechanistic ablation; not a learned MA method, not a new superior-model claim. Improvement attributes effect to the whole removed hook, not uniquely to one branch inside it.",
        "pinned_files": {str(p): file_sha(p) for p in pins},
    }
    _immutable_json(root / "protocol.json", protocol)
    return protocol


def run(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    launch = None
    try:
        for path, digest in protocol["pinned_files"].items():
            if file_sha(Path(path)) != digest:
                raise ValueError(f"Frozen intervention input changed: {path}")
        transform = disable_wrong_object_block if protocol.get("branch_ablation") else remove_action_guard
        if read_json(Path(protocol["intervention"])) != transform(read_json(Path(protocol["original"]))):
            raise ValueError("Intervention changed more than the action guard")
        wait_campaign(protocol)
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Intervention requires released GPUs and active filler")
        cohort = read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        supplement_runtime(Path(protocol["solver"]), Path(protocol["solver"]))
        config = {
            "solver_model": protocol["solver"], "solver_version": "ma64-guard-ablation",
            "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
            "harnesses": {"original": protocol["original"], "without_guard": protocol["intervention"]},
            "task_ids": cohort["all_task_ids"], "samples": 1,
            **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
            "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 3,
        }
        launch = Launcher(root / "supervision")
        if protocol.get("branch_ablation"):
            config["harnesses"] = {"original": protocol["original"],
                                   "without_wrong_object_block": protocol["intervention"],
                                   "without_guard": protocol["without_guard"]}
        matrix, _ = run_matrix(launch, config, root / "matrix", 1)
        reports, refs = {}, {}
        for name, entry in matrix["variants"].items():
            report = checked_native_report(read_json(Path(entry["report"])), cohort["all_task_ids"],
                {"samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"]})
            report = {**report, "results": [{k: r[k] for k in
                ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")} for r in report["results"]]}
            path = root / "reports" / f"{name}.json"
            _immutable_json(path, report)
            reports[name] = report
            refs[name] = {"accuracy": report["accuracy"], "path": str(path), "sha256": file_sha(path)}
        result = {"status": "completed", "scores": refs,
                  "contrast": paired_difference(reports["original"], reports["without_wrong_object_block"] if protocol.get("branch_ablation") else reports["without_guard"], seed=20261024),
                  "protocol_sha256": file_sha(root / "protocol.json"),
                  "environment_episodes": 200 * len(reports), "new_training_trajectories": 0,
                  "predecessor_recovery": protocol.get("recovered_training"),
                  "interpretation": protocol["interpretation"], "finished_at_unix": time.time()}
        if protocol.get("branch_ablation"):
            result["remaining_guard_effect"] = paired_difference(
                reports["without_wrong_object_block"], reports["without_guard"], seed=20261025)
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
    _immutable_json(root / "COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.campaign is None:
            parser.error("prepare needs --campaign")
        result = prepare(args.root.resolve(), args.campaign.resolve())
    else:
        result = run(args.root.resolve())
    print(result, flush=True)


if __name__ == "__main__":
    main()
