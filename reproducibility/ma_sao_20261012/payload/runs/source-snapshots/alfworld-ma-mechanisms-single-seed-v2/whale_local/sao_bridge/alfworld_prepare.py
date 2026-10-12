"""Freeze the local ALFWorld SAO/MA-SAO partitions and checkpoint receipts."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from .common import ROOT, file_sha, read_json, write_json

UPSTREAM = ROOT / "third_party/harness-r1-upstream"
PIN = "94f2e087f573e1b82fc9568bb634d4ae9a887e28"


def partition_rows(paths: list[str], data: dict) -> dict[str, list[dict]]:
    if len(paths) != 500:
        raise ValueError("expected the official 500-task ALFWorld mixed split")
    partitions = {}
    seen = set()
    for name in ("train", "reward", "dev", "test"):
        entry = data[name]
        start, end, count = entry["start"], entry["end"], entry["count"]
        if (type(start) is not int or type(end) is not int or type(count) is not int
                or not 0 <= start < end <= len(paths) or end - start != count):
            raise ValueError(f"invalid ALFWorld partition: {name}")
        rows = []
        for task_id in range(start, end):
            if task_id in seen:
                raise ValueError("ALFWorld campaign partitions overlap")
            seen.add(task_id)
            path = paths[task_id]
            rows.append({
                "id": f"alfworld-{task_id:03d}", "task_id": task_id,
                "task_path": path, "task_type": path.split("/")[2].split("-")[0],
            })
        partitions[name] = rows
    if seen != set(range(len(paths))):
        raise ValueError("ALFWorld campaign partitions must cover all 500 tasks")
    return partitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    if config["upstream"]["harness_r1_commit"] != PIN:
        raise ValueError("campaign Harness-R1 pin changed")
    head = __import__("subprocess").check_output(
        ["git", "-c", f"safe.directory={UPSTREAM}", "-C", str(UPSTREAM),
         "rev-parse", "HEAD"], text=True,
    ).strip()
    if head != PIN:
        raise ValueError("checked-out Harness-R1 revision differs from campaign")
    split = UPSTREAM / "data/alfworld/test_set.json"
    paths = json.loads(split.read_text())["mixed"]
    partitions = partition_rows(paths, config["data"])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "campaign": config["campaign"],
        "campaign_config": str(args.config.resolve()),
        "campaign_config_sha256": file_sha(args.config),
        "upstream_commit": PIN,
        "source_split": str(split),
        "source_split_sha256": file_sha(split),
        "checkpoint_receipts": {},
        "splits": {},
    }
    for role in ("engineer", "solver"):
        receipt = Path(config["initialization"][f"{role}_receipt"])
        manifest["checkpoint_receipts"][role] = {
            "path": str(receipt), "sha256": file_sha(receipt),
        }
    for name, rows in partitions.items():
        path = output / f"{name}.json"
        write_json(path, rows)
        manifest["splits"][name] = {
            "file": path.name, "count": len(rows), "sha256": file_sha(path),
            "task_ids": [rows[0]["task_id"], rows[-1]["task_id"]],
            "task_types": dict(sorted(Counter(row["task_type"] for row in rows).items())),
            "decision_use": config["data"][name]["decision_use"],
        }
    write_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
