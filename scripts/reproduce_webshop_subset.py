#!/usr/bin/env python3
"""Recompute saved WebShop subset evidence without a model, GPU or network."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from whale_local.webshop.core import (  # noqa: E402
    Episode, Protocol, compare_reports, evaluation_report,
)


def compact_report(report):
    result = {key: value for key, value in report.items() if key != "episodes"}
    result["episodes"] = []
    for episode in report["episodes"]:
        compact = {key: value for key, value in episode.items() if key != "turns"}
        compact["action_count"] = len(episode["turns"])
        compact["invalid_actions"] = sum(turn["invalid"] for turn in episode["turns"])
        compact["generated_tokens"] = sum(len(turn["sample"]["token_ids"]) for turn in episode["turns"])
        result["episodes"].append(compact)
    return result


def recompute_metrics(report):
    episodes = []
    for row in report["episodes"]:
        episode = Episode(
            row["task_id"], row["split"], row["seed"], row["task_manifest"],
            row.get("policy_version", 0), task_score=row["task_score"],
            terminated=row["terminated"], stop_reason=row["stop_reason"],
        )
        if episode.success != row["success"]:
            raise ValueError("Stored success disagrees with terminal task score")
        episodes.append(episode)
    protocol = Protocol.from_dict(report["protocol"])
    recomputed = evaluation_report(
        episodes, protocol, report["model"], report["assets"], report.get("training"),
    )["metrics"]
    recomputed["invalid_actions"] = sum(row["invalid_actions"] for row in report["episodes"])
    recomputed["generated_tokens"] = sum(row["generated_tokens"] for row in report["episodes"])
    if recomputed != report["metrics"]:
        raise ValueError("Stored aggregate metrics disagree with per-task evidence")
    return recomputed


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(run_root, destination):
    if destination.exists():
        raise ValueError("Never overwrite an evidence bundle")
    reports = {}
    for role, name in (
        ("initial", "baseline32_node25_v3"),
        ("grpo", "grpo_eval32_node25_v3"),
        ("success_sd", "sd_eval32_node26_v3"),
    ):
        reports[role] = compact_report(json.loads((run_root / name / "REPORT.json").read_text()))
        recompute_metrics(reports[role])
    paired = compare_reports(reports["grpo"], reports["success_sd"])
    compare_reports(reports["initial"], reports["grpo"])
    destination.mkdir(parents=True, exist_ok=False)
    for role, report in reports.items():
        (destination / f"{role}.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    (destination / "SOURCE_MANIFEST.json").write_bytes(
        (run_root / "frozen/subset-v3/MANIFEST.json").read_bytes(),
    )
    files = sorted(destination.glob("*.json"))
    manifest = {
        "schema": "coevolve_webshop_subset_bundle_v1",
        "interpretation": "Saved single-seed subset evidence, not a new model evaluation.",
        "files": {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in files},
        "historical_user_baseline": {
            "success_reported": 45, "score_reported": 74,
            "units_and_manifest_verified": False, "comparable": False,
        },
        "gpu_or_api_used_by_reproduction": False,
    }
    (destination / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


def reproduce(bundle):
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    if manifest["schema"] != "coevolve_webshop_subset_bundle_v1":
        raise ValueError("Unsupported evidence bundle")
    for name, metadata in manifest["files"].items():
        if Path(name).name != name:
            raise ValueError("Unsafe bundle path")
        path = bundle / name
        if path.stat().st_size != metadata["bytes"] or sha256(path) != metadata["sha256"]:
            raise ValueError(f"Evidence payload changed: {name}")
    reports = {role: json.loads((bundle / f"{role}.json").read_text())
               for role in ("initial", "grpo", "success_sd")}
    metrics = {role: recompute_metrics(report) for role, report in reports.items()}
    paired = compare_reports(reports["grpo"], reports["success_sd"])
    if paired != json.loads((bundle / "paired.json").read_text()):
        raise ValueError("Stored paired comparison disagrees with episode evidence")
    return {
        "metrics": metrics, "paired": paired, "gpu_or_api_used": False,
        "interpretation": "Recomputed stored subset evidence; not a new model evaluation.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "reproducibility/webshop_subset_20261114")
    parser.add_argument("--emit-from", type=Path)
    args = parser.parse_args(argv)
    if args.emit_from is not None:
        write_bundle(args.emit_from, args.bundle)
    print(json.dumps(reproduce(args.bundle), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
