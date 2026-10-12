"""Aggregate task-level rewards from packet probes and compare paired checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

from .matrix_probe import save


def _baseline(attempt: dict, start: int, size: int) -> dict[int, float]:
    packets = Path(attempt["output"]) / "failure_packets.jsonl"
    if not packets.is_file():
        return {task_id: 1.0 for task_id in range(start, start + size)}
    rows = [json.loads(line) for line in packets.read_text().splitlines() if line.strip()]
    if not rows:
        return {task_id: 1.0 for task_id in range(start, start + size)}
    result = {
        int(task_id): float(reward)
        for row in rows
        for task_id, reward in row["metadata"]["baseline_rewards"].items()
    }
    if set(result) != set(range(start, start + size)):
        raise ValueError(f"incomplete baseline reward vector: {attempt['output']}")
    return result


def aggregate(roots: list[Path], pairing: str = "rl_agent_sft") -> dict:
    task_rows = {}
    packets = []
    for root in roots:
        complete = json.loads((root / "COMPLETE.json").read_text())
        if complete.get("status") != "completed":
            raise ValueError(f"incomplete packet probe: {root}")
        for result in complete["results"]:
            if result["pairing"] != pairing:
                continue
            attempt = result["attempts"][result["selected_attempt"] - 1]
            summary = attempt["summary"]
            baseline = _baseline(attempt, result["start"], result["size"])
            patched = dict(baseline)
            valid = summary.get("status") != "baseline_all_pass_no_failures"
            if valid:
                categories = summary["categories"]
                if categories["infra_failure"]:
                    raise ValueError(f"infrastructure failure is not a score: {attempt['output']}")
                valid = categories["valid_evaluated"] == 1
                if valid:
                    paired = {
                        int(row["task_id"]): float(row["patched_reward"])
                        for row in summary["paired_tasks"]
                    }
                    if set(paired) != set(baseline):
                        raise ValueError(f"incomplete patched reward vector: {attempt['output']}")
                    patched = paired
            packets.append({
                "start": result["start"], "end": result["start"] + result["size"],
                "valid_patch": valid, "output": attempt["output"],
            })
            for task_id in sorted(baseline):
                if task_id in task_rows:
                    raise ValueError(f"duplicate packet reward for task {task_id}")
                task_rows[task_id] = {
                    "task_id": task_id, "baseline_reward": baseline[task_id],
                    "patched_reward": patched[task_id],
                    "delta_reward": patched[task_id] - baseline[task_id],
                    "valid_patch": valid,
                }
    rows = [task_rows[task_id] for task_id in sorted(task_rows)]
    if not rows:
        raise ValueError(f"no packet results for pairing {pairing}")
    return {
        "status": "completed", "pairing": pairing, "tasks": len(rows),
        "baseline_pass": sum(row["baseline_reward"] for row in rows),
        "patched_pass": sum(row["patched_reward"] for row in rows),
        "delta_pass": sum(row["delta_reward"] for row in rows),
        "baseline_rate": sum(row["baseline_reward"] for row in rows) / len(rows),
        "patched_rate": sum(row["patched_reward"] for row in rows) / len(rows),
        "delta_rate": sum(row["delta_reward"] for row in rows) / len(rows),
        "valid_packets": sum(row["valid_patch"] for row in packets),
        "invalid_packets": sum(not row["valid_patch"] for row in packets),
        "packets": packets, "task_rows": rows,
        "interpretation": (
            "Packet-transductive metric: each Engineer proposal saw failures from the same "
            "cohort. Invalid proposals use the frozen baseline as paired fallback."
        ),
    }


def paired_bootstrap(reference: dict, candidate: dict, field: str,
                     samples: int = 10000, seed: int = 20260930) -> dict:
    left = {row["task_id"]: float(row[field]) for row in reference["task_rows"]}
    right = {row["task_id"]: float(row[field]) for row in candidate["task_rows"]}
    if set(left) != set(right) or not left:
        raise ValueError("paired reports must cover the same nonempty task IDs")
    task_ids = sorted(left)
    deltas = [right[task_id] - left[task_id] for task_id in task_ids]
    rng = random.Random(seed)
    estimates = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    lower = estimates[int(0.05 * samples)]
    gain = sum(deltas) / len(deltas)
    return {
        "field": field, "tasks": len(deltas), "gain": gain,
        "one_sided_95pct_lower_bound": lower,
        "bootstrap_samples": samples, "seed": seed,
        "accepted_1pp_rule": gain >= 0.01 and lower > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append")
    parser.add_argument("--pairing", default="rl_agent_sft")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--field", choices=["baseline_reward", "patched_reward"],
                        default="patched_reward")
    args = parser.parse_args()
    if args.reference or args.candidate:
        if not args.reference or not args.candidate or args.root:
            raise ValueError("comparison requires exactly --reference and --candidate")
        report = paired_bootstrap(
            json.loads(args.reference.read_text()),
            json.loads(args.candidate.read_text()), args.field,
        )
    else:
        if not args.root:
            raise ValueError("aggregation requires at least one --root")
        report = aggregate(args.root, args.pairing)
    save(args.output, report)


if __name__ == "__main__":
    main()
