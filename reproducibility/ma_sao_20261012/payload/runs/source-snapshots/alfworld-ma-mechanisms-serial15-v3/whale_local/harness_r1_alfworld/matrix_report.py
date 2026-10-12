"""Aggregate immutable 2x2 ALFWorld start-matrix waves without cherry-picking."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from sao_bridge.common import read_json, write_json

PAIRINGS = {
    "rl_agent_sft": ("agent_sft", "rl"),
    "cold_sft_agent_sft": ("agent_sft", "cold_sft"),
    "rl_original": ("original", "rl"),
    "cold_sft_original": ("original", "cold_sft"),
}


def aggregate(roots: list[Path]) -> dict:
    totals = {name: Counter() for name in PAIRINGS}
    baseline_rows = {name: {} for name in PAIRINGS}
    attempts = Counter()
    elapsed = 0.0
    seen_cohorts = set()
    for root in roots:
        complete = read_json(root / "COMPLETE.json")
        if complete["status"] != "completed" or complete["completed"] != 16:
            raise ValueError(f"incomplete matrix wave: {root}")
        if len(complete["results"]) != 16:
            raise ValueError(f"matrix wave does not contain 16 jobs: {root}")
        elapsed += complete["elapsed_seconds"]
        wave_cohorts = set()
        for result in complete["results"]:
            name = result["pairing"]
            if name not in PAIRINGS:
                raise ValueError(f"unknown matrix pairing: {name}")
            cohort = (result["start"], result["start"] + result["size"])
            wave_cohorts.add(cohort)
            attempt = result["attempts"][result["selected_attempt"] - 1]
            attempts[len(result["attempts"])] += 1
            summary = attempt["summary"]
            packet_path = Path(attempt["output"]) / "failure_packets.jsonl"
            if packet_path.is_file():
                packets = [
                    json.loads(line) for line in packet_path.read_text().splitlines()
                    if line.strip()
                ]
                baseline = {
                    int(task_id): float(reward)
                    for packet in packets
                    for task_id, reward in packet["metadata"]["baseline_rewards"].items()
                }
            else:
                baseline = {
                    task_id: 1.0
                    for task_id in range(result["start"], result["start"] + result["size"])
                }
            expected_ids = set(range(result["start"], result["start"] + result["size"]))
            if set(baseline) != expected_ids:
                raise ValueError(f"baseline packet identity mismatch: {attempt['output']}")
            for task_id, reward in baseline.items():
                if task_id in baseline_rows[name]:
                    raise ValueError(f"duplicate task {task_id} in pairing {name}")
                baseline_rows[name][task_id] = reward
            if summary.get("status") == "baseline_all_pass_no_failures":
                official = {
                    "total_tasks": result["size"], "baseline_pass": result["size"],
                    "patched_pass": result["size"], "delta_pass": 0, "valid_patches": 0,
                }
                categories = {
                    "infra_failure": 0, "invalid_patch": 0, "no_op": 0,
                }
                rescued = regressed = 0
            else:
                official = summary["official_summary"]
                categories = summary["categories"]
                rescued, regressed = summary["rescued_tasks"], summary["regressed_tasks"]
            totals[name].update(
                jobs=1, tasks=official["total_tasks"],
                baseline=official["baseline_pass"], patched=official["patched_pass"],
                delta=official["delta_pass"], valid=official["valid_patches"],
                infra=categories["infra_failure"],
                invalid=categories["invalid_patch"],
                noop=categories["no_op"],
                rescued=rescued, regressed=regressed,
            )
        if len(wave_cohorts) != 4 or seen_cohorts & wave_cohorts:
            raise ValueError("matrix waves must contain four new matched cohorts")
        seen_cohorts.update(wave_cohorts)
    baseline_consistency = {}
    for target, left, right in (
        ("agent_sft", "rl_agent_sft", "cold_sft_agent_sft"),
        ("original", "rl_original", "cold_sft_original"),
    ):
        if set(baseline_rows[left]) != set(baseline_rows[right]):
            raise ValueError(f"target baseline cohorts differ for {target}")
        mismatches = [
            task_id for task_id in baseline_rows[left]
            if baseline_rows[left][task_id] != baseline_rows[right][task_id]
        ]
        baseline_consistency[target] = {
            "tasks": len(baseline_rows[left]), "mismatch_count": len(mismatches),
            "mismatch_task_ids": mismatches,
        }
    summaries = {}
    for name, values in totals.items():
        summaries[name] = {
            **dict(values),
            "baseline_rate": values["baseline"] / values["tasks"],
            "patched_rate": values["patched"] / values["tasks"],
            "delta_rate": values["delta"] / values["tasks"],
        }
    return {
        "status": "completed",
        "wave_count": len(roots),
        "cohort_count": len(seen_cohorts),
        "matrix_job_count": sum(values["jobs"] for values in totals.values()),
        "wall_seconds_sum": elapsed,
        "attempt_count_distribution": dict(attempts),
        "pairings": summaries,
        "baseline_consistency": baseline_consistency,
        "interpretation": (
            "Training/development diagnostic only. Each patch saw failures from the same "
            "10-task cohort; do not interpret as sealed-test generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, aggregate(args.root))


if __name__ == "__main__":
    main()
