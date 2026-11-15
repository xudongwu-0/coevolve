"""Recompute saved non-distilled SAO-GRPO paired evidence without models or GPUs."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from whale_local.webshop.core import compare_reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, default=PROJECT / "reproducibility/webshop_sao_grpo_20261115")
    args = parser.parse_args()
    root = args.evidence.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    for filename, expected in manifest["files"].items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts or (root / relative).is_symlink():
            raise ValueError("Unsafe evidence path")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Evidence hash mismatch: " + filename)
    initial, control, reference = [
        json.loads((root / (name + ".json")).read_text()) for name in ("initial", "control", "reference")
    ]
    paired = compare_reports(control, reference, comparison="sao_grpo")
    before_after = compare_reports(initial, reference, comparison="sao_grpo")
    if paired != json.loads((root / "paired.json").read_text()):
        raise ValueError("Recomputed paired result differs")
    if before_after != json.loads((root / "initial_comparison.json").read_text()):
        raise ValueError("Recomputed initial comparison differs")
    for report in (initial, control, reference):
        count = sum(episode["success"] for episode in report["episodes"])
        score = sum(episode["task_score"] for episode in report["episodes"]) / len(report["episodes"])
        if count != report["metrics"]["successes"] or abs(score - report["metrics"]["mean_task_score"]) > 1e-12:
            raise ValueError("Stored headline metrics disagree with episode outcomes")
    print(json.dumps({
        "state": "saved_pair_recomputed_and_hash_verified",
        "gpu_or_model_used": False,
        "successes": {n: r["metrics"]["successes"] for n, r in zip(
            ("initial", "control", "reference"), (initial, control, reference))},
        "reference_minus_control": paired,
        "reference_minus_initial": before_after,
        "scope": "Outcome/provenance projection only; original tokens, probabilities and environment traces stay in the experiment directory.",
    }, indent=2))


if __name__ == "__main__":
    main()
