"""Same-task dev comparison and gated held-out evaluation of harness patches."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.evaluation import Sampler
from .episode import evaluate_rows
from .patch import base_patch, merge_patches, patch_digest
from .selection import paired_gain
from .identity import baseline_config


def load_shared_baseline(directory: Path, signature: dict, rows: list[dict], seeds: list[int]) -> dict:
    complete = json.loads((directory / "COMPLETE.json").read_text())
    if complete.get("status") != "baseline_only" or complete.get("baseline_signature") != signature:
        raise ValueError("Shared baseline experiment identity mismatch")
    path = directory / "baseline.json"
    if complete.get("baseline_report_sha256") != file_sha(path):
        raise ValueError("Shared baseline report bytes changed")
    report = json.loads(path.read_text())
    results = report.get("results", [])
    actual = [(item.get("task_id"), item.get("rollout_seed")) for item in results]
    expected = [(row["id"], seed) for row, seed in zip(rows, seeds, strict=True)]
    if actual != expected or len(set(actual)) != len(expected):
        raise ValueError("Shared baseline task/seed pairing mismatch")
    return report


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-urls", required=True)
    parser.add_argument("--target-model-sha256", required=True)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--noop-control", action="store_true")
    parser.add_argument("--shared-baseline", type=Path)
    parser.add_argument("--incumbent", type=Path)
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()
    if args.samples <= 0 or args.output.exists():
        raise ValueError("Require a positive sample count and a fresh evaluation output path")
    if args.shared_baseline and (args.baseline_only or args.split != "dev"):
        raise ValueError("Shared baseline reuse is limited to paired development evaluation")
    incumbent = json.loads(args.incumbent.read_text()) if args.incumbent else base_patch()
    if args.baseline_only:
        if args.split != "dev":
            raise ValueError("Standalone baseline collection is development-only")
        candidate = incumbent
    elif args.noop_control:
        if args.split != "dev" or args.candidate is not None:
            raise ValueError("No-op infrastructure control is development-only without a candidate")
        candidate = incumbent
    elif args.candidate is None:
        raise ValueError("A candidate patch is required for paired evaluation")
    else:
        candidate = merge_patches(incumbent, json.loads(args.candidate.read_text()))
    if args.split == "test":
        if args.freeze is None:
            raise ValueError("Held-out test cannot be loaded without an irreversible dev freeze")
        frozen = json.loads(args.freeze.read_text())
        if (
            frozen.get("status") != "development_frozen"
            or frozen.get("patch_sha256") != patch_digest(candidate)
            or frozen.get("target_model_sha256") != args.target_model_sha256
        ):
            raise ValueError("Held-out freeze does not match target and candidate identities")
    manifest_path = args.data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["splits"][args.split]
    path = args.data_root / entry["file"]
    if path.resolve().parent != args.data_root.resolve() or file_sha(path) != entry["sha256"]:
        raise ValueError("Frozen evaluation split path or file digest mismatch")
    questions = json.loads(path.read_text())
    ids = [row["id"] for row in questions]
    if len(ids) != entry["count"] or len(set(ids)) != len(ids):
        raise ValueError("Evaluation question count or uniqueness changed")
    if hashlib.sha256("\n".join(ids).encode()).hexdigest() != entry["ids_sha256"]:
        raise ValueError("Evaluation question identity digest changed")
    rows, seeds = [], []
    for question in questions:
        for sample in range(args.samples):
            rows.append(question)
            seeds.append(args.seed + int(question["id"][:8], 16) % 1_000_000 + 1009 * sample)
    args.output.mkdir(parents=True)
    started = time.time()
    identity = {
        "split": args.split, "data_manifest_sha256": file_sha(manifest_path),
        "target_model_sha256": args.target_model_sha256,
        "incumbent_sha256": patch_digest(incumbent), "candidate_sha256": patch_digest(candidate),
        "temperature": args.temperature, "tokens": args.tokens,
        "seed": args.seed, "samples": args.samples,
    }
    signature = {
        **baseline_config(
            target_model_sha256=args.target_model_sha256,
            target_temperature=args.temperature, total_tokens=args.tokens,
            retrieval_manifest=args.data_root / "retrieval_manifest.json",
            data_manifest=manifest_path, incumbent_patch=incumbent,
        ),
        "split": args.split, "seed": args.seed, "samples": args.samples,
        "target_urls": args.target_urls, "retrieval_url": args.retrieval_url,
        "concurrency": args.concurrency,
    }
    identity["baseline_signature"] = signature
    write_json(args.output / "IDENTITY.json", identity)
    os.environ["SAO_SEARCHQA_RETRIEVAL_URL"] = args.retrieval_url
    sampler = Sampler(args.target_urls, concurrency=args.concurrency,
                      temperature=args.temperature, chat_template_kwargs={"enable_thinking": False})
    try:
        reports = []
        evaluations = []
        if args.shared_baseline:
            shared = load_shared_baseline(args.shared_baseline, signature, rows, seeds)
            write_json(args.output / "baseline.json", shared)
            reports.append(shared)
            identity.update(
                shared_baseline=str(args.shared_baseline),
                shared_baseline_receipt_sha256=file_sha(args.shared_baseline / "COMPLETE.json"),
            )
        else:
            evaluations.append(("baseline", incumbent))
        if not args.baseline_only:
            evaluations.append(("candidate", candidate))
        for name, patch in evaluations:
            with (args.output / f"{name}.episodes.jsonl").open("x") as stream:
                completed = 0

                def record(index: int, result: dict) -> None:
                    nonlocal completed
                    stream.write(json.dumps({"index": index, "result": result}) + "\n")
                    stream.flush()
                    completed += 1
                    if completed % 32 == 0:
                        write_json(args.output / "STATUS.json", {
                            "stage": name, "completed": completed, "planned": len(rows),
                            "elapsed_seconds": time.time() - started,
                        })

                report = await evaluate_rows(
                    rows, sampler=sampler, patch=patch, total_tokens=args.tokens,
                    rollout_seeds=seeds, concurrency=args.concurrency, on_result=record,
                )
            write_json(args.output / f"{name}.json", report)
            reports.append(report)
        comparison = (
            {"baseline_mean": reports[0]["mean"], "accepted": False,
             "status": "baseline_only", "gain_claimed": False}
            if args.baseline_only
            else paired_gain(reports[0]["results"], reports[1]["results"])
        )
        hook_errors = sum(
            bool(event.get("error")) or str(event.get("status", "")).startswith("candidate_")
            for result in reports[-1]["results"]
            for event in result.get("harness_trace", [])
        )
        comparison["candidate_hook_errors"] = hook_errors
        if hook_errors:
            comparison["accepted"] = False
            comparison["rejection_reason"] = "candidate_hook_runtime_errors"
        if not args.baseline_only and candidate["actions"] == incumbent["actions"]:
            comparison["accepted"] = False
            comparison["rejection_reason"] = "unmodified_harness"
        current = baseline_config(
            target_model_sha256=args.target_model_sha256,
            target_temperature=args.temperature, total_tokens=args.tokens,
            retrieval_manifest=args.data_root / "retrieval_manifest.json",
            data_manifest=manifest_path, incumbent_patch=incumbent,
        )
        if any(signature[key] != value for key, value in current.items()):
            raise RuntimeError("Frozen evaluation runtime/data configuration changed during requests")
        write_json(args.output / "COMPLETE.json", {
            **identity, **comparison, "merged_candidate": candidate,
            "baseline_report_sha256": file_sha(args.output / "baseline.json"),
            "episodes_executed": len(rows) * len(evaluations),
            "baseline_episodes_reused": len(rows) if args.shared_baseline else 0,
            "generated_tokens": sampler.generated_tokens,
            "requests": sampler.requests,
            "elapsed_seconds": time.time() - started,
            "decision_scope": "development_acceptance" if args.split == "dev" else "held_out_reporting_only",
        })
    finally:
        await sampler.close()


if __name__ == "__main__":
    asyncio.run(main())
