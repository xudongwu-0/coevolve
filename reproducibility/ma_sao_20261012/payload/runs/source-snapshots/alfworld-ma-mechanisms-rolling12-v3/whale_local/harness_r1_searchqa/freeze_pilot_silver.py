"""Freeze a small clean positive distillation dataset without changing source receipts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from sao_bridge.common import file_sha, write_json
from .bootstrap import digest
from .patch import extract_and_normalize_patch, patch_digest


def freeze(source: Path, packets: Path, output: Path, minimum: int = 32) -> dict:
    identity = json.loads((source / "INPUT_IDENTITY.json").read_text())
    if file_sha(packets) != identity["data_sha256"]:
        raise ValueError("Pilot packets differ from teacher collection")
    if output.exists() or minimum < 1:
        raise ValueError("Require a fresh output and positive pilot size")
    data = [json.loads(line) for line in packets.read_text().splitlines() if line.strip()]
    rows, audit, seen = [], [], set()
    for path in sorted((source / "samples").glob("*.json")):
        sample = json.loads(path.read_text())
        reward = sample.get("reward", {})
        score = reward.get("score")
        if (
            sample.get("status") != "scored" or sample.get("clean_execution") is not True
            or type(reward.get("hook_errors")) is not int or reward["hook_errors"] != 0
            or not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0
            or sample.get("finish_reason") != "stop"
        ):
            continue
        index = identity["packet_start"] + sample["packet_index"]
        row = data[index]
        if sample["request_digest"] != digest(row):
            raise ValueError("Positive sample references a different failure packet")
        patch = extract_and_normalize_patch(sample["response"])
        if patch_digest(patch) != sample["patch_sha256"]:
            raise ValueError("Positive patch bytes differ from their execution receipt")
        messages = [*row["prompt"], {"role": "assistant", "content": sample["response"]}]
        key = digest(messages)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "messages": messages, "provenance": {
                "generation_id": sample["id"],
                "engineer_manifest_sha256": identity["engineer_manifest_sha256"],
                "task_ids": row["metadata"]["task_ids"], "reward_delta": score,
                "patch_sha256": sample["patch_sha256"], "split": "train",
            },
        })
        audit.append({"sample": str(path), "sha256": file_sha(path), "hook_errors": 0})
        if len(rows) == minimum:
            break
    if len(rows) < minimum:
        raise RuntimeError(f"Only {len(rows)} unique clean positives; need {minimum}")
    output.mkdir(parents=True)
    dataset = output / "sft.jsonl"
    dataset.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    write_json(output / "CLEAN_EXECUTION_AUDIT.json", audit)
    receipt = {
        "stage": "fresh_base_silver_collection", "pilot": True,
        "examples": len(rows), "minimum_examples": minimum, "usable_for_sft": True,
        "dataset_sha256": file_sha(dataset), "input_identity": identity,
        "teacher_source": str(source), "standard_256_example_gate_met": minimum >= 256,
        "frozen_at_unix": time.time(), "development_or_test_gain_claimed": False,
    }
    write_json(output / "COMPLETE.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum", type=int, default=32)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source, args.packets, args.output, args.minimum)))


if __name__ == "__main__":
    main()
