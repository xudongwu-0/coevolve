"""Predeclared two-seed gate for H0-fixed 4B Solver SAO validation."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import read_json, write_json


def evaluate_pair(run_paths: list[Path], protocol: dict) -> dict:
    if len(run_paths) != len(protocol["required_seeds"]):
        raise ValueError("Pair gate requires exactly the predeclared run count")
    runs = []
    specs = []
    for path in run_paths:
        complete_path = path / "COMPLETE.json"
        if not complete_path.is_file():
            raise FileNotFoundError(f"Run is not complete: {path}")
        run = read_json(complete_path)
        spec = read_json(path / "spec.json")
        if run.get("status") != "complete" or run.get("harness") != "fixed_h0":
            raise ValueError(f"Invalid solver-validation receipt: {path}")
        if run.get("test_split_opened"):
            raise ValueError("The 4B development gate must not open the test split")
        runs.append(run)
        specs.append(spec)
    seeds = [run["seed"] for run in runs]
    if sorted(seeds) != sorted(protocol["required_seeds"]) or len(set(seeds)) != len(seeds):
        raise ValueError("Run seeds differ from the predeclared pair")
    comparable_keys = set(specs[0]) - {"seed", "run"}
    if any(
        {key: spec[key] for key in comparable_keys}
        != {key: specs[0][key] for key in comparable_keys}
        for spec in specs[1:]
    ):
        raise ValueError("Paired runs do not share the same training/evaluation protocol")
    deltas = [run["paired_report"]["candidate_minus_control"] for run in runs]
    lower_bounds = [
        run["paired_report"]["paired_question_bootstrap_95pct"][0] for run in runs
    ]
    checks = {
        "all_selected_trained_checkpoint": all(
            run["selected_trained_checkpoint"] for run in runs
        ),
        "all_noncollapsing": min(deltas) >= protocol["minimum_seed_delta"],
        "mean_delta": sum(deltas) / len(deltas) >= protocol["minimum_mean_delta"],
        "significant_seed_count": sum(
            bound > 0 for bound in lower_bounds
        ) >= protocol["minimum_significant_seeds"],
    }
    return {
        "status": "complete",
        "gate_passed": all(checks.values()),
        "checks": checks,
        "required_seeds": protocol["required_seeds"],
        "seed_results": [
            {
                "seed": run["seed"],
                "selected_trained_checkpoint": run["selected_trained_checkpoint"],
                "candidate_minus_control": delta,
                "paired_question_bootstrap_95pct": run["paired_report"][
                    "paired_question_bootstrap_95pct"
                ],
            }
            for run, delta in zip(runs, deltas)
        ],
        "mean_candidate_minus_control": sum(deltas) / len(deltas),
        "gate_protocol": protocol,
        "interpretation": (
            "This is a two-training-seed development gate. Question-level bootstrap "
            "intervals do not establish population-level training-seed significance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_pair(args.run, read_json(args.protocol))
    write_json(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
