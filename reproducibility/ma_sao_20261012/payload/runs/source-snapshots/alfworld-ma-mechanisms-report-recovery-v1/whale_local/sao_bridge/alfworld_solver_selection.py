"""Select an ALFWorld Solver checkpoint using paired no-patch development rewards."""
from __future__ import annotations

import argparse
from pathlib import Path

from .alfworld_global_harness import (
    accepted_candidate, lower_bound, packet_metadata,
)
from .common import read_json, write_json


def baseline_rewards(refresh_root: Path, solver_version: str) -> dict[int, float]:
    rewards = {}
    for packet in packet_metadata(refresh_root, solver_version):
        for task_id, reward in packet["baseline_rewards"].items():
            task_id = int(task_id)
            if task_id in rewards:
                raise ValueError(f"duplicate baseline reward for task {task_id}")
            rewards[task_id] = float(reward)
    if not rewards:
        raise ValueError("solver selection has no development rewards")
    return rewards


def paired_solver_report(incumbent: dict[int, float], candidate: dict[int, float],
                         bootstrap_samples: int, bootstrap_seed: int,
                         minimum_absolute_gain: float) -> dict:
    if set(incumbent) != set(candidate):
        raise ValueError("solver selection changed development task identity")
    if bootstrap_samples < 100:
        raise ValueError("solver selection requires at least 100 bootstrap samples")
    rows = [{
        "task_id": task_id,
        "incumbent_reward": incumbent[task_id],
        "candidate_reward": candidate[task_id],
        "delta_reward": candidate[task_id] - incumbent[task_id],
    } for task_id in sorted(incumbent)]
    deltas = [row["delta_reward"] for row in rows]
    report = {
        "tasks": len(rows),
        "incumbent_pass": sum(incumbent.values()),
        "candidate_pass": sum(candidate.values()),
        "gain": sum(deltas) / len(deltas),
        "one_sided_95pct_lower_bound": lower_bound(
            deltas, bootstrap_samples, bootstrap_seed,
        ),
        "minimum_absolute_gain": minimum_absolute_gain,
        "task_rows": rows,
    }
    report["accepted"] = accepted_candidate(report, minimum_absolute_gain)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    run = Path(config["run"]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    incumbent = baseline_rewards(
        Path(config["incumbent_refresh"]),
        config["incumbent_solver_version"],
    )
    candidate = baseline_rewards(
        Path(config["candidate_refresh"]),
        config["candidate_solver_version"],
    )
    report = paired_solver_report(
        incumbent,
        candidate,
        int(config.get("bootstrap_samples", 10000)),
        int(config.get("bootstrap_seed", 20260930)),
        float(config["minimum_absolute_gain"]),
    )
    write_json(run / "paired_rows.json", report["task_rows"])
    write_json(run / "COMPLETE.json", {
        key: value for key, value in {
            **report,
            "status": "completed",
            "phase": config["phase"],
            "selection_split": config["selection_split"],
            "incumbent_model": config["incumbent_model"],
            "candidate_model": config["candidate_model"],
            "incumbent_solver_version": config["incumbent_solver_version"],
            "candidate_solver_version": config["candidate_solver_version"],
        }.items() if key != "task_rows"
    })


if __name__ == "__main__":
    main()
