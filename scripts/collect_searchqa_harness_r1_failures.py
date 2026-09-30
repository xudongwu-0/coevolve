#!/usr/bin/env python3
"""Collect frozen-target SearchQA failures and build grouped RL JSONL."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from harness_r1_searchqa.data import build_rl_rows
from harness_r1_searchqa.episode import evaluate_rows
from harness_r1_searchqa.patch import base_patch, patch_digest
from harness_r1_searchqa.identity import baseline_config
from sao_bridge.evaluation import Sampler
from sao_bridge.searchqa_prepare import sha256


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-url", required=True)
    parser.add_argument("--target-model-sha256", required=True)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/3fs/xudong/coevolve/assets/searchqa_sao_20260929"),
    )
    parser.add_argument("--split", choices=("train",), default="train")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=3072)
    parser.add_argument("--rollout-seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--packet-size", type=int, default=8)
    parser.add_argument("--maximum-packets", type=int, default=256)
    parser.add_argument("--incumbent-patch", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-rl-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-only", action="store_true")
    return parser.parse_args()


def load_patch(path: Path | None) -> dict[str, Any]:
    if path is None:
        return base_patch()
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("incumbent patch file must contain a JSON object")
    patch_digest(value)
    return value


def selected_rows(rows: list[dict[str, Any]], *, seed: int, start: int, count: int) -> list[dict[str, Any]]:
    if start < 0 or count <= 0:
        raise ValueError("start must be nonnegative and count must be positive")
    ranked = sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).digest(),
    )
    selected = ranked[start : start + count]
    if len(selected) != count:
        raise ValueError("requested row selection exceeds the frozen train split")
    return selected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


async def run(args: argparse.Namespace) -> None:
    os.environ["SAO_SEARCHQA_RETRIEVAL_URL"] = args.retrieval_url
    manifest = json.loads((args.data_root / "manifest.json").read_text())
    entry = manifest["splits"]["train"]
    train_path = args.data_root / entry["file"]
    if train_path.resolve().parent != args.data_root.resolve():
        raise ValueError("train split path escapes the frozen data directory")
    if sha256(train_path) != entry["sha256"]:
        raise ValueError("Frozen train split SHA256 mismatch")
    training_rows = json.loads(train_path.read_text())
    ids = [row["id"] for row in training_rows]
    if len(ids) != entry["count"] or len(set(ids)) != len(ids):
        raise ValueError("Frozen train split count or unique identity mismatch")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != entry["ids_sha256"]:
        raise ValueError("Frozen train split identity digest mismatch")
    rows = selected_rows(
        training_rows,
        seed=args.selection_seed,
        start=args.start,
        count=args.count,
    )
    seeds = [
        args.rollout_seed + int(str(row["id"])[:8], 16) % 1_000_000
        for row in rows
    ]
    patch = load_patch(args.incumbent_patch)
    configuration = baseline_config(
        target_model_sha256=args.target_model_sha256,
        target_temperature=args.temperature,
        total_tokens=args.tokens,
        retrieval_manifest=args.data_root / "retrieval_manifest.json",
        data_manifest=args.data_root / "manifest.json",
        incumbent_patch=patch,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    receipts_path = args.output_report.with_suffix(".episodes.jsonl")
    if args.output_report.exists() or receipts_path.exists():
        raise FileExistsError("Collector output already exists; use a fresh audited run directory")
    write_json(args.output_report.with_suffix(".input.json"), {
        "baseline_config": configuration,
        "target_urls": args.target_url,
        "retrieval_url": args.retrieval_url,
        "collector_source_sha256": sha256(Path(__file__)),
        "split": args.split, "start": args.start, "count": args.count,
    })
    completed = 0
    started = asyncio.get_running_loop().time()
    receipt_stream = receipts_path.open("x")

    def on_result(index: int, result: dict[str, Any]) -> None:
        nonlocal completed
        receipt_stream.write(json.dumps({"index": index, "result": result}, ensure_ascii=False) + "\n")
        receipt_stream.flush()
        completed += 1
        if completed % 32 == 0 or completed == len(rows):
            elapsed = asyncio.get_running_loop().time() - started
            write_json(args.output_report.with_suffix(".status.json"), {
                "stage": "frozen_target_train_collection", "completed": completed,
                "planned": len(rows), "elapsed_seconds": elapsed,
                "episodes_per_second": completed / max(elapsed, 1e-6),
                "remaining_seconds_estimate": elapsed / completed * (len(rows) - completed),
            })
    sampler = Sampler(
        args.target_url,
        concurrency=args.concurrency,
        temperature=args.temperature,
        chat_template_kwargs={"enable_thinking": False},
    )
    try:
        evaluation = await evaluate_rows(
            rows,
            sampler=sampler,
            patch=patch,
            total_tokens=args.tokens,
            rollout_seeds=seeds,
            concurrency=args.concurrency,
            on_result=on_result,
        )
    finally:
        receipt_stream.close()
        await sampler.close()
    after_configuration = baseline_config(
        target_model_sha256=args.target_model_sha256,
        target_temperature=args.temperature,
        total_tokens=args.tokens,
        retrieval_manifest=args.data_root / "retrieval_manifest.json",
        data_manifest=args.data_root / "manifest.json",
        incumbent_patch=patch,
    )
    if after_configuration != configuration:
        raise RuntimeError("Baseline runtime or immutable manifest changed during collection")
    report = {
        "protocol": "searchqa-harness-r1-baseline-v1",
        "split": args.split,
        "selection_seed": args.selection_seed,
        "start": args.start,
        "count": args.count,
        "target_url": args.target_url,
        "target_model_sha256": args.target_model_sha256,
        "data_manifest_sha256": sha256(args.data_root / "manifest.json"),
        "incumbent_patch": patch,
        "incumbent_patch_sha256": patch_digest(patch),
        "temperature": args.temperature,
        "total_tokens": args.tokens,
        "baseline_config": configuration,
        **evaluation,
    }
    write_json(args.output_report, report)
    if args.baseline_only:
        print(json.dumps({"stage": "baseline_collected", "mean": report["mean"],
                          "count": report["count"], "report": str(args.output_report)}))
        return
    rl_rows = build_rl_rows(
        report,
        incumbent_patch=patch,
        target_model_sha256=args.target_model_sha256,
        data_manifest=args.data_root / "manifest.json",
        packet_size=args.packet_size,
        maximum_packets=args.maximum_packets,
    )
    write_jsonl(args.output_rl_jsonl, rl_rows)
    print(
        json.dumps(
            {
                "baseline_mean": report["mean"],
                "episodes": len(report["results"]),
                "failures": sum(result["reward"] == 0 for result in report["results"]),
                "rl_packets": len(rl_rows),
                "report": str(args.output_report),
                "rl_jsonl": str(args.output_rl_jsonl),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(run(arguments()))
