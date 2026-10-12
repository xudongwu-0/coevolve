"""Paired, example-clustered comparisons for completed rewrite initialization pilots."""
from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import paired_difference
from .common import read_json, write_json


def paired_edits(control: dict, candidate: dict, field: str) -> dict:
    if field not in {"legal", "instruction_following"}:
        raise ValueError("Unsupported edit metric")
    for key in ("examples", "samples_per_example", "temperature", "max_tokens"):
        if control[key] != candidate[key]:
            raise ValueError(f"Edit evaluation protocol differs: {key}")

    def converted(report):
        return {
            "metric": field, "question_count": report["examples"],
            "samples_per_question": report["samples_per_example"],
            "results": [
                {"id": row["id"], "sample_index": row["sample"],
                 "rollout_seed": row["seed"], "reward": float(row[field])}
                for row in report["results"]
            ],
        }

    return paired_difference(converted(control), converted(candidate))


def compare(control: Path, candidate: Path) -> dict:
    for root in (control, candidate):
        if not (root / "COMPLETE.json").exists() or (root / "FAILED.json").exists():
            raise ValueError(f"Comparison requires a successfully completed trial: {root}")
    left, right = (read_json(root / "spec.json") for root in (control, candidate))
    for key in ("initial_model", "data_sha256", "math_questions"):
        if left[key] != right[key]:
            raise ValueError(f"Initialization/evaluation protocol differs: {key}")
    for key, default in {"samples": 2, "math_samples": 2, "edit_tokens": 1024,
                         "math_tokens": 2048, "server_context": 8192}.items():
        if left.get(key, default) != right.get(key, default):
            raise ValueError(f"Evaluation sampling/token protocol differs: {key}")
    if ("math_data_sha256" in left and "math_data_sha256" in right
            and left["math_data_sha256"] != right["math_data_sha256"]):
        raise ValueError("Math development datasets differ")
    splits = right.get("evaluation_splits", ["dev", "test"])
    edits = {}
    for split in splits:
        a, b = (read_json(root / f"{split}_edits.json") for root in (control, candidate))
        edits[split] = {field: paired_edits(a, b, field)
                       for field in ("legal", "instruction_following")}
    a, b = (read_json(root / "math_dev.json") for root in (control, candidate))

    def prompts(report):
        return {(row["id"], row["sample_index"], row["rollout_seed"]):
                row["messages"][:2] for row in report["results"]}

    if prompts(a) != prompts(b):
        raise ValueError("Math evaluation prompts or sample identities differ")
    return {
        "control": str(control.resolve()), "candidate": str(candidate.resolve()),
        "teacher_data_sha256": left["data_sha256"], "edits": edits,
        "math_dev": paired_difference(a, b),
        "harness_generation": {
            label: {key: value for key, value in
                    read_json(root / "harness_generation.json").items() if key != "proposals"}
            for label, root in (("control", control), ("candidate", candidate))
        },
        "limitations": [
            "Small rewrite pilot; no claim of broad teacher-capability transfer.",
            "Question/example-cluster uncertainty, not training-seed robustness.",
            "Math development and repeatedly used development edits are exploratory.",
            "Harness legality/diversity is not measured harness quality.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a fresh comparison output")
    result = compare(args.control, args.candidate)
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
