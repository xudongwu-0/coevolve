"""Merge fixed sample zero with a frozen, training-excluded one-shot extension."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_strong_phase import checked_native_report, checked_release
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import file_sha, read_json, write_json


def validate_cohort(protocol: dict) -> None:
    old, new, all_ids = (protocol[key] for key in ("old_task_ids", "new_task_ids", "all_task_ids"))
    strict = protocol["strict_goal_disjoint_task_ids"]
    related = protocol["untrained_instance_but_train_goal_related_task_ids"]
    if (
        protocol["protocol"] != "alfworld-core200-untrained-oneshot-v1"
        or protocol["old_sample_index_to_reuse"] != 0
        or protocol["samples_per_question"] != 1
        or len(old) != 24 or len(new) != 176
        or all_ids != old + new or len(set(all_ids)) != 200
        or set(all_ids) & set(protocol["training_task_ids_excluded"])
        or len(strict) != len(set(strict)) or len(strict) != 197
        or len(related) != 3 or len(set(related)) != 3
        or set(strict) & set(related) or set(strict) | set(related) != set(all_ids)
    ):
        raise ValueError("Extension must contain exactly 200 untrained instances, with fixed sample zero")


def merge_reports(old: dict, new: dict, protocol: dict) -> dict:
    validate_cohort(protocol)
    evaluation = {
        "seed": protocol["seed"], "solver_tokens": protocol["solver_tokens"], "samples": 1,
    }
    checked_native_report(old, protocol["old_task_ids"], {**evaluation, "samples": 2})
    checked_native_report(new, protocol["new_task_ids"], evaluation)
    rows = [row for row in old["results"] if row["sample_index"] == 0] + new["results"]
    report = {
        "metric": "mean@1", "question_count": 200, "samples_per_question": 1,
        "accuracy": sum(row["reward"] for row in rows) / 200,
        "results": [
            {key: row[key] for key in ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
            for row in rows
        ],
    }
    return checked_native_report(report, protocol["all_task_ids"], evaluation)


def subset_report(report: dict, task_ids: list[int]) -> dict:
    selected = set(task_ids)
    rows = [row for row in report["results"] if row["task_id"] in selected]
    if not rows or len(rows) != len(selected):
        raise ValueError("Subset lost or duplicated task outcomes")
    return {
        **report, "question_count": len(rows), "results": rows,
        "accuracy": sum(row["reward"] for row in rows) / len(rows),
    }


def run(protocol_path: Path, timeout: float) -> dict:
    protocol = read_json(protocol_path)
    validate_cohort(protocol)
    root = Path(protocol["run"])
    for name, digest in protocol["pinned_files"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f"Frozen evaluation input changed: {name}")
    old_core = read_json(Path(protocol["old_core_table"]))
    deadline = time.monotonic() + timeout
    reports, sources = {}, {}
    for role, config_path in protocol["configs"].items():
        requested = read_json(Path(config_path))
        launch_path = root / ("W1_LAUNCH.json" if role == "w1" else "w0_owner/LAUNCH.json")
        launch = read_json(launch_path)
        if (
            launch["config_sha256"] != file_sha(Path(config_path))
            or launch["protocol_sha256"] != file_sha(protocol_path)
        ):
            raise ValueError("Launched evaluation configuration no longer matches its freeze")
        output = Path(requested["run"])
        while not all((output / name).exists() for name in ("COMPLETE.json", "RELEASED.json")):
            if (output / "FAILED.json").exists():
                raise RuntimeError(f"Extension evaluation failed: {read_json(output / 'FAILED.json')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Timed out waiting for complete extension matrices")
            time.sleep(5)
        checked_release(output)
        recorded = read_json(output / "config.json")
        if any(recorded.get(key) != value for key, value in requested.items()):
            raise ValueError("Evaluated matrix differs from the frozen request")
        if read_json(output / "COMPLETE.json")["status"] != "completed":
            raise ValueError("Native evaluation did not complete")
        matrix = read_json(output / "report.json")
        if set(matrix["variants"]) != {"no_patch", *requested["harnesses"]}:
            raise ValueError("Extension matrix changed its frozen systems")
        for label, variant in matrix["variants"].items():
            name = role if label == "no_patch" else f"{role}_{label}"
            old_source = old_core["sources"][name]
            old_path, new_path = Path(old_source["path"]), Path(variant["report"])
            if file_sha(old_path) != old_source["sha256"]:
                raise ValueError("Original quick report changed before sample-zero reuse")
            expected_harness = old_source.get("harness_sha256", old_path.stem)
            if variant["harness_sha256"] != expected_harness:
                raise ValueError("Extension used a different selected harness")
            if recorded["solver_model"] != old_core["sources"][role]["solver_model"]:
                raise ValueError("Extension used a different Solver checkpoint")
            reports[name] = merge_reports(read_json(old_path), read_json(new_path), protocol)
            _immutable_json(root / "merged" / f"{name}.json", reports[name])
            sources[name] = {
                "old_report": str(old_path), "old_sha256": file_sha(old_path),
                "new_report": str(new_path), "new_sha256": file_sha(new_path),
            }
    if set(reports) != {"w0", "w0_h1", "w0_official_r1", "w1", "w1_h1", "w1_h2"}:
        raise ValueError("Core extension requires all six frozen systems")
    cohorts = {}
    for label, ids in (
        ("all200", protocol["all_task_ids"]),
        ("strict197", protocol["strict_goal_disjoint_task_ids"]),
        ("new176", protocol["new_task_ids"]),
    ):
        selected = {name: subset_report(report, ids) for name, report in reports.items()}
        cohorts[label] = {
            "question_count": len(ids),
            "scores": {name: {
                "accuracy": report["accuracy"],
                "successes": sum(row["reward"] for row in report["results"]),
            } for name, report in selected.items()},
            "paired": {
                f"{right}_minus_{left}": paired_difference(selected[left], selected[right], seed=9000045)
                for left, right in (
                    ("w0", "w0_h1"), ("w0", "w1"), ("w1", "w1_h1"),
                    ("w1_h1", "w1_h2"), ("w0_official_r1", "w1_h2"),
                )
            },
        }
    result = {
        "status": "completed", "samples_per_question": 1, "cohorts": cohorts,
        "sources": sources, "protocol_sha256": file_sha(protocol_path),
        "limitations": protocol["limitations"], "finished_at_unix": time.time(),
    }
    _immutable_json(root / "COMPLETE.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    try:
        result = run(args.protocol, args.timeout)
    except Exception as exc:
        root = Path(read_json(args.protocol)["run"])
        write_json(root / "AGGREGATION_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    print({name: cohort["scores"] for name, cohort in result["cohorts"].items()}, flush=True)
