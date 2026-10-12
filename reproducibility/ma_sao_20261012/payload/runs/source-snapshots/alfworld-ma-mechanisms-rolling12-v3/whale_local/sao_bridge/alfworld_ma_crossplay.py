"""Post-hoc fixed-patch cross-play diagnosing the observed late MA system decline."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import run_matrix, supplement_runtime
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit


def combinations(initial: dict, matched: dict, late: dict) -> tuple[dict, dict]:
    if matched["selection"]["rounds"] != 64 or late["selection"]["rounds"] <= 64:
        raise ValueError("Cross-play requires the matched point and a later selected point")
    models = {
        "matched": matched["selection"]["pair"]["solver"]["actor"],
        "late": late["selection"]["pair"]["solver"]["actor"],
    }
    if models["matched"] == models["late"]:
        raise ValueError("Cross-play cannot compare identical Solver checkpoints")
    patches = {
        "h0": initial["selection"]["harness"],
        "h_matched": matched["selection"]["harness"],
        "h_late": late["selection"]["harness"],
    }
    return models, patches


def prepare(root: Path, ma: Path, predecessor: Path) -> dict:
    files = [ma / f"{label}_COMPLETE.json" for label in ("initial", "matched", "best")]
    records = [read_json(path) for path in files]
    if any(record["status"] != "completed" for record in records):
        raise ValueError("Cross-play requires completed MA endpoints")
    models, patches = combinations(*records)
    inputs = [*files, EVAL_PROTOCOL, *(Path(path) for path in patches.values())]
    for model in models.values():
        weights = sorted(Path(model).glob("*.safetensors"))
        if not weights:
            raise ValueError(f"Cross-play Solver checkpoint has no weights: {model}")
        inputs.extend([Path(model) / "config.json", *weights])
    protocol = {
        "protocol": "ma-fixed-patch-crossplay-v1", "root": str(root),
        "ma_root": str(ma), "predecessor": str(predecessor),
        "models": models, "patches": patches,
        "late_round": records[2]["selection"]["rounds"],
        "pinned_files": {str(path): file_sha(path) for path in inputs},
        "queue_timeout_seconds": 12 * 3600,
        "logical_systems": 8, "planned_environment_episodes": 1600,
        "new_training_trajectories": 0,
        "interpretation": "Post-hoc reused-development diagnosis, not a confirmatory test or checkpoint search.",
        "hypotheses": [
            "Compare weights while holding each saved patch fixed.",
            "Compare patches while holding Solver weights fixed.",
            "Remeasure diagonals rather than treating old service outputs as bitwise deterministic.",
        ],
        "incumbent_replaced": False,
    }
    _immutable_json(root / "protocol.json", protocol)
    return protocol


def wait_predecessor(protocol: dict) -> None:
    root = Path(protocol["predecessor"])
    deadline = time.monotonic() + protocol["queue_timeout_seconds"]
    while True:
        if (root / "FAILED.json").exists():
            raise RuntimeError(f"Cross-play predecessor failed: {read_json(root / 'FAILED.json')}")
        if all((root / name).exists() for name in ("COMPLETE.json", "RELEASED.json")):
            if read_json(root / "COMPLETE.json")["status"] != "completed":
                raise ValueError("Cross-play predecessor has no successful completion")
            audit = gpu_release_audit()
            if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
                raise RuntimeError("Cross-play requires released local GPUs and active filler")
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Cross-play predecessor did not finish within its queue budget")
        time.sleep(15)


def run(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    launch = None
    try:
        for path, digest in protocol["pinned_files"].items():
            if file_sha(Path(path)) != digest:
                raise ValueError(f"Cross-play frozen input changed: {path}")
        cohort = read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        wait_predecessor(protocol)
        launch = Launcher(root / "supervision")
        reports, scores, environment_episodes = {}, {}, 0
        for name, model in protocol["models"].items():
            supplement_runtime(Path(model), Path(model))
            config = {
                "solver_model": model, "solver_version": f"ma-crossplay-{name}",
                "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
                "harnesses": protocol["patches"], "task_ids": cohort["all_task_ids"],
                "samples": 1,
                **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
                "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 4,
            }
            matrix, _ = run_matrix(launch, config, root / name, 1)
            environment_episodes += 200 * len({
                entry["harness_sha256"] for entry in matrix["variants"].values()
            })
            reports[name], scores[name] = {}, {}
            for patch, entry in matrix["variants"].items():
                report = checked_native_report(
                    read_json(Path(entry["report"])), cohort["all_task_ids"],
                    {"samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"]},
                )
                compact = {**report, "results": [
                    {key: row[key] for key in ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
                    for row in report["results"]
                ]}
                path = root / "reports" / f"{name}-{patch}.json"
                _immutable_json(path, compact)
                reports[name][patch] = compact
                scores[name][patch] = {
                    "all200": report["accuracy"],
                    "strict197": subset_report(report, cohort["strict_goal_disjoint_task_ids"])["accuracy"],
                    "report": str(path), "sha256": file_sha(path),
                }
        contrasts = {
            "late_minus_matched_weights_with_fixed_" + patch: paired_difference(
                reports["matched"][patch], reports["late"][patch], seed=9000060 + i,
            ) for i, patch in enumerate(("no_patch", "h0", "h_matched", "h_late"))
        }
        for i, model in enumerate(("matched", "late")):
            contrasts[f"late_minus_matched_patch_with_fixed_{model}_weights"] = paired_difference(
                reports[model]["h_matched"], reports[model]["h_late"], seed=9000064 + i,
            )
        result = {
            "status": "completed", "scores": scores, "contrasts": contrasts,
            "new_training_trajectories": 0, "environment_episodes": environment_episodes,
            "logical_scored_episodes": 1600,
            "protocol_sha256": file_sha(root / "protocol.json"),
            "interpretation": protocol["interpretation"], "incumbent_replaced": False,
            "finished_at_unix": time.time(),
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
    parser.add_argument("--ma-root", type=Path)
    parser.add_argument("--predecessor", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.ma_root is None or args.predecessor is None:
            parser.error("prepare requires --ma-root and --predecessor")
        result = prepare(args.root.resolve(), args.ma_root.resolve(), args.predecessor.resolve())
    else:
        result = run(args.root.resolve())
    print(result, flush=True)


if __name__ == "__main__":
    main()
