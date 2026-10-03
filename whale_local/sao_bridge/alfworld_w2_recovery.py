"""Continue the actual rejected W2 with fixed h2 and report a prespecified curve."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_solver_validation import no_patch_assignments
from .alfworld_strong_phase import checked_native_report, checkpoint_pair
from .alfworld_strong_select import run_matrix
from .alfworld_train_data import _immutable_json, solver_rows, training_catalogue
from .alfworld_w_train import assignment_schedule_audit
from .analysis import paired_difference
from .common import ROOT, ensure_model_runtime_metadata, file_sha, read_json, write_json
from .joint_health import metric_rows
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


OLD_PHASE = ROOT / "runs/alfworld_alternating_e15_w79_8h_v1/sequential/w2/attempt-01"
OLD_EVALUATION = ROOT / "runs/alfworld_w2_h2_200_v1/COMPLETE.json"
TEMPLATE = ROOT / "examples/alfworld_solver_fast_v4_s42.json"
CHECKPOINT_STEPS = (19, 39, 59, 79)


def continuation_config(old: dict, actor: str, critic: str, root: Path) -> dict:
    if actor == old["solver_model"] or "globalstep79" not in actor:
        raise ValueError("Recovery must start at rejected W2-step79, never the W1 input")
    config = copy.deepcopy(old)
    config.update(
        run=str(root / "training"), phase="w2_recovery", experiment=root.name,
        solver_model=actor, solver_critic=critic, data=str(root / "data.json"),
        assignments=str(root / "assignments.json"), harness=str(root / "fixed_h2.json"),
        steps=80, batch_size=8, save_interval_steps=20, seed=2000042,
        independent_trajectories=True, phase_timeout_seconds=6 * 3600,
    )
    return config


def prepare(root: Path, prerequisite: Path) -> dict:
    old = read_json(OLD_PHASE / "config.json")
    pair = checkpoint_pair(read_json(OLD_PHASE / "optimizer/trained.json"), 79)
    previous = read_json(OLD_EVALUATION)
    if previous["candidate_solver"] != pair["actor"] or previous["candidate_promoted"]:
        raise ValueError("W2 recovery identity differs from the actually rejected/evaluated model")
    template = read_json(TEMPLATE)
    catalogue = Path(template["training_catalogue"])
    games = training_catalogue(catalogue, template["training_catalogue_sha256"])
    full = read_json(Path(template["training_data"]))
    if full != solver_rows(games, template["training_catalogue_sha256"], catalogue_path=catalogue):
        raise ValueError("Recovery training pool differs from the frozen train-only catalogue")
    used = {r["task_id"] for path in (
        ROOT / "runs/alfworld_solver_fast_v4_s42/solver_train.json",
        Path(old["data"]),
    ) for r in read_json(path)}
    fresh = [row for row in full if row["task_id"] not in used][:640]
    if len(fresh) != 640:
        raise ValueError("Not enough unused train-only instances for the declared recovery budget")
    rows = [{**row, "id": f"{row['id']}-recovery-{i}", "partner_version": old["harness_version"]}
            for i, row in enumerate(fresh)]
    _immutable_json(root / "data.json", rows)
    _immutable_json(root / "fixed_h2.json", read_json(Path(old["harness"])))
    _immutable_json(root / "assignments.json", no_patch_assignments(root / "fixed_h2.json", training_rows=rows))
    config = continuation_config(old, pair["actor"], pair["critic"], root)
    assignment_schedule_audit(config)
    _immutable_json(root / "training_config.json", config)
    protocol = {
        "protocol": "alfworld-rejected-w2-fixed-h2-recovery-v1", "root": str(root),
        "prerequisite": str(prerequisite), "prerequisite_label": "matched",
        "queue_timeout_seconds": 24 * 3600,
        "initial_pair": pair, "fixed_h2": str(root / "fixed_h2.json"),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "initial_w2_steps": 80, "incremental_steps": 80, "batch_size": 8,
        "incremental_consumed_trajectories": 640,
        "prior_h1_w1_h2_w2_consumed_trajectories": 1664,
        "original_w2_h2_all200": previous["cohorts"]["all200"]["scores"]["h2"]["accuracy"],
        "optimizer_state": "Fresh Adam; existing W2 actor and critic weights, not exact optimizer-state resume.",
        "backend_change": "One trajectory per microbatch repairs Qwen3.5 recurrent cross-sequence leakage.",
        "held_fixed": "h2, original LR2e-6/critic1e-5, K1, 5 critic warmup steps, terminal SAO, no KL.",
        "hypothesis": "Rejected W2 may recover under continued fixed-h2 SAO; finite budget cannot prove irrecoverability.",
        "primary_endpoint": "Final extra80 steps versus freshly remeasured W2+h2, paired all200 mean@1.",
        "curve": "Report all prespecified extra20/40/60/80-step checkpoints; no all200 checkpoint selection.",
        "pinned_files": {str(path): file_sha(path) for path in (
            root / "data.json", root / "fixed_h2.json", root / "assignments.json",
            root / "training_config.json", EVAL_PROTOCOL, OLD_EVALUATION,
        )},
    }
    _immutable_json(root / "protocol.json", protocol)
    return protocol


def wait_predecessor(protocol: dict) -> None:
    root, label = Path(protocol["prerequisite"]), protocol["prerequisite_label"]
    deadline = time.monotonic() + protocol["queue_timeout_seconds"]
    while not all((root / f"{label}_{suffix}.json").exists() for suffix in ("COMPLETE", "RELEASED")):
        failed = root / f"{label}_FAILED.json"
        if failed.exists():
            raise RuntimeError(f"Predecessor evaluation failed: {read_json(failed)}")
        if time.monotonic() >= deadline:
            raise TimeoutError("W2 recovery queue timed out without predecessor release")
        time.sleep(15)
    if read_json(root / f"{label}_COMPLETE.json")["status"] not in ("completed", "unavailable"):
        raise ValueError("Unexpected predecessor completion status")
    audit = gpu_release_audit()
    if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
        raise RuntimeError("W2 recovery requires released local GPUs and active filler")


def evaluate_point(launch: Launcher, root: Path, label: str, model: str, protocol: dict) -> dict:
    ensure_model_runtime_metadata(Path(model), Path(protocol["initial_pair"]["actor"]))
    cohort = read_json(EVAL_PROTOCOL)
    validate_cohort(cohort)
    config = {
        "solver_model": model, "solver_version": f"w2-recovery-{label}",
        "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
        "harnesses": {"h2": protocol["fixed_h2"]},
        "task_ids": cohort["all_task_ids"], "samples": 1,
        **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
        "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 2,
    }
    matrix, _ = run_matrix(launch, config, root / "evaluation" / label, 1)
    result = {}
    for variant, item in matrix["variants"].items():
        report = checked_native_report(
            read_json(Path(item["report"])), cohort["all_task_ids"],
            {"samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"]},
        )
        compact = {**report, "results": [
            {key: row[key] for key in ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
            for row in report["results"]
        ]}
        path = root / "reports" / f"{label}-{variant}.json"
        _immutable_json(path, compact)
        result[variant] = {
            "all200": report["accuracy"],
            "strict197": subset_report(report, cohort["strict_goal_disjoint_task_ids"])["accuracy"],
            "report": str(path), "sha256": file_sha(path),
        }
    _immutable_json(root / f"{label}_COMPLETE.json", result)
    return result


def run(root: Path) -> dict:
    require_research_node()
    protocol = read_json(root / "protocol.json")
    launch = None
    try:
        for name, digest in protocol["pinned_files"].items():
            if file_sha(Path(name)) != digest:
                raise ValueError(f"Frozen W2 recovery input changed: {name}")
        wait_predecessor(protocol)
        launch = Launcher(root / "supervision")
        initial = evaluate_point(launch, root, "initial", protocol["initial_pair"]["actor"], protocol)
        process = launch.module(
            "sao_bridge.alfworld_w_train", ["--config", root / "training_config.json"],
            gpus="0,1,2,3,4,5,6,7", name="w2-recovery-train",
        )
        deadline = time.monotonic() + 6 * 3600
        metrics_path = root / "training/optimizer/metrics.jsonl"
        while process.poll() is None:
            launch.check()
            rows = metric_rows(metrics_path)
            if rows and any(isinstance(v, (int, float)) and not math.isfinite(v)
                            for v in rows[-1]["metrics"].values()):
                raise FloatingPointError("W2 recovery optimizer emitted a nonfinite metric")
            if time.monotonic() >= deadline:
                raise TimeoutError("W2 recovery training exceeded six hours")
            time.sleep(5)
        if process.returncode:
            raise RuntimeError(f"W2 continuation failed ({process.returncode}); no fabricated recovery score")
        trained = read_json(root / "training/optimizer/trained.json")
        control = read_json(Path(initial["h2"]["report"]))
        reference = read_json(ROOT / "runs/alfworld_core_200_oneshot_v2/merged/w1_h2.json")
        curve = []
        for step in protocol["checkpoint_steps"]:
            pair = checkpoint_pair(trained, step)
            scores = evaluate_point(launch, root, f"extra{step + 1}", pair["actor"], protocol)
            candidate = read_json(Path(scores["h2"]["report"]))
            curve.append({
                "extra_steps": step + 1, "total_w2_steps": 80 + step + 1,
                "extra_consumed_trajectories": (step + 1) * protocol["batch_size"],
                "scores": scores,
                "versus_remeasured_w2_h2": paired_difference(control, candidate, seed=9000048),
                "versus_w1_h2": paired_difference(reference, candidate, seed=9000049),
            })
        final = curve[-1]["versus_remeasured_w2_h2"]
        result = {
            "status": "completed", "initial": initial, "curve": curve,
            "final_improved_point_estimate": final["candidate_minus_control"] > 0,
            "final_positive_paired_lower_bound": final["paired_question_bootstrap_95pct"][0] > 0,
            "conclusion_scope": "Recovery within an extra80-step budget with repaired backend/fresh Adam; not proof of irreversible collapse.",
            "finished_at_unix": time.time(), "protocol_sha256": file_sha(root / "protocol.json"),
            "incumbent_replaced": False,
        }
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
    _immutable_json(root / "COMPLETE.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--prerequisite", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.prerequisite is None:
            parser.error("prepare requires --prerequisite")
        result = prepare(args.root.resolve(), args.prerequisite.resolve())
    else:
        result = run(args.root.resolve())
    print(result, flush=True)


if __name__ == "__main__":
    main()
