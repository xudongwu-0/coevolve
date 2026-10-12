"""Measure full uncached versus verified-baseline-reuse dev evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from sao_bridge.common import file_sha, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-urls", required=True)
    parser.add_argument("--target-model-sha256", required=True)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shared-baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    profiles = {}
    for mode in ("uncached", "reused"):
        output = args.output / mode
        command = [
            sys.executable, "-B", "-m", "harness_r1_searchqa.evaluate",
            "--target-urls", args.target_urls,
            "--target-model-sha256", args.target_model_sha256,
            "--retrieval-url", args.retrieval_url, "--data-root", str(args.data_root),
            "--noop-control", "--samples", "8", "--concurrency", "64",
            "--output", str(output),
        ]
        if mode == "reused":
            command.extend(["--shared-baseline", str(args.shared_baseline)])
        started = time.monotonic()
        subprocess.run(command, check=True)
        complete = json.loads((output / "COMPLETE.json").read_text())
        elapsed = time.monotonic() - started
        profiles[mode] = {
            "wall_seconds": elapsed,
            "evaluation_seconds": complete["elapsed_seconds"],
            "episodes_executed": complete["episodes_executed"],
            "baseline_episodes_reused": complete["baseline_episodes_reused"],
            "generated_tokens": complete["generated_tokens"],
            "requests": complete["requests"],
            "completion_sha256": file_sha(output / "COMPLETE.json"),
            "baseline_mean": complete["baseline_mean"],
            "candidate_mean": complete["candidate_mean"],
            "accepted": complete["accepted"],
        }
        write_json(args.output / "STATUS.json", {"completed_modes": profiles})
    before, after = profiles["uncached"], profiles["reused"]
    write_json(args.output / "COMPLETE.json", {
        "purpose": "infra_only_noop_control_not_learned_harness_gain",
        "profiles": profiles,
        "wall_speedup": before["wall_seconds"] / after["wall_seconds"],
        "episodes_reduction_fraction": 1 - after["episodes_executed"] / before["episodes_executed"],
        "shared_baseline_setup_seconds": json.loads(
            (args.shared_baseline / "COMPLETE.json").read_text()
        )["elapsed_seconds"],
        "fresh_baseline_required_once": True,
        "no_op_acceptance_must_be_false": not before["accepted"] and not after["accepted"],
    })


if __name__ == "__main__":
    main()
