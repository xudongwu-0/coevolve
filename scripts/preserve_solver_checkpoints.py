#!/usr/bin/env python3
"""Generate a hash manifest without deleting any prior solver checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sao_bridge.common import file_sha, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = args.project / "runs"
    checkpoints: dict[Path, list[dict]] = {}
    for receipt in sorted(runs.glob("**/trained.json")):
        data = json.loads(receipt.read_text())
        if data.get("role") != "solver":
            continue
        for role, raw_path in data.get("checkpoints", {}).items():
            if role not in {"actor", "critic"}:
                continue
            path = Path(raw_path)
            if not path.is_absolute() or not path.resolve().is_relative_to(runs.resolve()):
                raise ValueError(f"Solver receipt refers outside the experiment runs: {receipt}")
            checkpoints.setdefault(path, []).append({
                "receipt": str(receipt), "receipt_sha256": file_sha(receipt),
                "role": role, "phase": data.get("phase"),
                "actor_steps": data.get("actor_steps"),
            })
    sft = runs / "rewrite_timebox_20260929T0745Z/round_01/training/checkpoint"
    checkpoints.setdefault(sft, []).append({"role": "sft1152", "required_initializer": True})
    entries = []
    for checkpoint, references in checkpoints.items():
        if not checkpoint.is_dir() or not list(checkpoint.glob("*.safetensors")):
            raise FileNotFoundError(f"Preserved solver checkpoint is missing: {checkpoint}")
        files = []
        for path in sorted(checkpoint.iterdir()):
            if path.is_file():
                files.append({
                    "name": path.name, "bytes": path.stat().st_size,
                    "sha256": file_sha(path),
                })
        entries.append({
            "path": str(checkpoint), "preserve": True,
            "selection": "all completed solver W weights, not only dev-best",
            "references": references, "files": files,
        })
        print(json.dumps({"hashed_checkpoint": str(checkpoint), "files": len(files)}), flush=True)
    sources = []
    source_paths = [
        *sorted((args.project / "whale_local").glob("*.py")),
        *sorted((args.project / "whale_local/sao_bridge").glob("*.py")),
        *sorted((args.project / "whale_local/sao_bridge").glob("*.patch")),
        *sorted((args.project / "examples").glob("*.json")),
        *sorted((args.project / "docs").glob("*.md")),
    ]
    for path in source_paths:
        sources.append({"path": str(path), "sha256": file_sha(path)})
    write_json(args.output, {
        "status": "retained_weights_hashed_no_deletion",
        "scope": "solver checkpoints referenced by trained.json plus SFT1152; originals stay in assets",
        "checkpoint_count": len(entries), "checkpoints": entries,
        "algorithms_protocols_analysis": sources,
        "deleted_paths": [],
        "cleanup_requires_explicit_exact_allowlist_and_reverification": True,
    })


if __name__ == "__main__":
    main()
