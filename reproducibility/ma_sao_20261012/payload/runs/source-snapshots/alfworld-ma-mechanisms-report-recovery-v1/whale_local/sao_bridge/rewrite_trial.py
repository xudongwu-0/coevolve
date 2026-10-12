"""Native SFT initialization with held-out edits and mathematical retention checks."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import time

from .common import ROOT, file_sha, read_json, write_json
from .evaluation import Sampler, evaluate
from .experiment import propose
from .launch import Launcher
from .resources import stop_known
from .retool import INITIAL_SOURCE, harness_spec
from .rewrite_data import EditExample, grade_output, validate_example


async def evaluate_edits(url: str, rows: list[dict], *, samples=2, temperature=0.7,
                         max_tokens=1024) -> dict:
    from .rewrite_sft_train import format_messages

    if not rows or samples < 1 or max_tokens < 1:
        raise ValueError("Edit evaluation needs examples and positive sample/token counts")
    sampler = Sampler(url, concurrency=128, temperature=temperature)
    started = time.monotonic()
    try:
        async def one(row, sample):
            seed = 42 + int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) % 1000000000
            reply = await sampler(format_messages(row), None, max_tokens, seed + sample)
            output = reply["choices"][0]["message"].get("content") or ""
            return {
                "id": row["id"], "family": row["family"], "split": row["split"],
                "format": row["format"], "sample": sample, "seed": seed + sample,
                "output": output, **grade_output(row, output),
                "truncated": reply["choices"][0].get("finish_reason") == "length",
            }
        results = await asyncio.gather(*(one(row, s) for row in rows for s in range(samples)))
        return {
            "examples": len(rows), "samples_per_example": samples,
            "temperature": temperature, "max_tokens": max_tokens,
            "legal_fraction": sum(r["legal"] for r in results) / len(results),
            "instruction_following_fraction": sum(r["instruction_following"] for r in results) / len(results),
            "tokens": sampler.generated_tokens, "elapsed_seconds": time.monotonic() - started,
            "results": results,
        }
    finally:
        await sampler.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("initial", "sft", "evaluate"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--origin-model", type=Path,
                        help="Required for evaluation-only checkpoints; declares the original base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--math-questions", type=int, default=64)
    parser.add_argument("--math-data", type=Path, default=ROOT / "assets/dapo_aime_protocol/dev.json")
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--math-samples", type=int, default=2)
    parser.add_argument("--edit-tokens", type=int, default=1024)
    parser.add_argument("--math-tokens", type=int, default=2048)
    parser.add_argument("--server-context", type=int, default=8192)
    parser.add_argument("--evaluation-gpus", default="0,1,2,3")
    parser.add_argument("--dev-only", action="store_true",
                        help="Tune on development checks without evaluating held-out edits")
    args = parser.parse_args()
    if (args.mode == "evaluate") != (args.origin_model is not None):
        parser.error("--origin-model is required only with --mode evaluate")
    if min(args.samples, args.math_samples, args.edit_tokens, args.math_tokens) < 1:
        parser.error("Sample and generation-token limits must be positive")
    if args.server_context <= max(args.edit_tokens, args.math_tokens, 768):
        parser.error("Server context must exceed generation limits and leave space for prompts")
    devices = args.evaluation_gpus.split(",")
    if (not devices or any(device not in {str(i) for i in range(8)} for device in devices)
            or len(set(devices)) != len(devices)):
        parser.error("--evaluation-gpus must list distinct GPU IDs from 0 through 7")
    root = args.output.resolve()
    if root.exists():
        raise FileExistsError("Use a fresh SFT trial directory")
    rows = read_json(args.data)
    for row in rows:
        validate_example(EditExample.model_validate(row))
    math_path = args.math_data.resolve()
    math_rows = read_json(math_path)[:args.math_questions]
    if len(math_rows) != args.math_questions or args.math_questions < 1:
        raise ValueError("Requested math regression set is unavailable")
    splits = ["dev"] if args.dev_only else ["dev", "test"]
    if any(not any(row["split"] == split for row in rows) for split in splits):
        raise ValueError("Every requested edit evaluation split must contain examples")
    for path in (args.model, args.origin_model):
        if path is not None and (not path.is_dir() or not list(path.glob("*.safetensors"))):
            raise FileNotFoundError(f"Expected a local model checkpoint: {path}")
    stop_known(require_exclusive=True)
    # These pilot artifacts contain only synthetic data and public-model reports.
    # The same login has different numeric UIDs on node3 and node4.
    root.mkdir(parents=True, mode=0o755)
    launch = Launcher(root)
    start = time.monotonic()
    model = args.model.resolve()
    spec = {
        "mode": args.mode, "initial_model": str((args.origin_model or model).resolve()),
        "input_model": str(model), "data": str(args.data.resolve()),
        "data_sha256": file_sha(args.data), "epochs": args.epochs, "lr": args.lr,
        "math_questions": args.math_questions, "host": os.uname().nodename,
        "math_data_sha256": file_sha(math_path),
        "evaluation_splits": splits, "samples": args.samples, "math_samples": args.math_samples,
        "edit_tokens": args.edit_tokens, "math_tokens": args.math_tokens,
        "server_context": args.server_context, "evaluation_gpus": devices,
        "counts": {split: sum(r["split"] == split for r in rows) for split in ("train", "dev", "test")},
        "source_sha256": {p.name: file_sha(p) for p in Path(__file__).parent.glob("*.py")},
        "claim": "SFT initialization experiment, not automatic teacher-capability transfer",
    }
    write_json(root / "spec.json", spec)
    try:
        if args.mode == "sft":
            train = launch.module(
                "sao_bridge.rewrite_sft_train",
                ["--model", str(model), "--data", str(args.data.resolve()),
                 "--output", str(root / "training"), "--lr", str(args.lr),
                 "--epochs", str(args.epochs), "--batch-size", "16", "--seed", "42"],
                gpus="0", name="sft_train",
            )
            launch.wait(train, "sft_train")
            model = root / "training" / "checkpoint"
            if not list(model.glob("*.safetensors")):
                raise RuntimeError("SFT did not produce an HF checkpoint")
        reports = {}
        with launch.vllm(str(model), port=18601, gpus=args.evaluation_gpus,
                         data_parallel=len(devices), max_model_len=args.server_context,
                         name="evaluation") as url:
            for split in spec["evaluation_splits"]:
                report = asyncio.run(evaluate_edits(
                    url, [r for r in rows if r["split"] == split],
                    samples=args.samples, max_tokens=args.edit_tokens,
                ))
                write_json(root / f"{split}_edits.json", report)
                reports[split] = {k: v for k, v in report.items() if k != "results"}
                launch.check()
            proposals = asyncio.run(propose(url, INITIAL_SOURCE, 32, 768, 54142))
            valid = [harness_spec(p["source"]) for p in proposals if p["valid"]]
            initial_values = {k: v for k, v in harness_spec(INITIAL_SOURCE).items() if k != "source"}
            values = [{k: v for k, v in item.items() if k != "source"} for item in valid]
            distinct = {str(sorted(item.items())) for item in values}
            harness_report = {
                "count": len(proposals), "temperature": 0.5,
                "legal_fraction": len(valid) / len(proposals),
                "changed_fraction": sum(item != initial_values for item in values) / len(proposals),
                "distinct_valid_configurations": len(distinct),
                "scope": "actual SAO editor prompt; legality/diversity, not task-score improvement",
            }
            write_json(root / "harness_generation.json", {**harness_report, "proposals": proposals})
            reports["harness_generation"] = harness_report
            async def math_check():
                sampler = Sampler(url, concurrency=128)
                try:
                    return await evaluate(sampler, math_rows, INITIAL_SOURCE,
                                          tokens=args.math_tokens, samples=args.math_samples, seed=1442)
                finally:
                    await sampler.close()
            math_report = asyncio.run(math_check())
            write_json(root / "math_dev.json", math_report)
            reports["math_dev"] = {k: v for k, v in math_report.items() if k != "results"}
        completion = {
            "model": str(model), "reports": reports,
            "elapsed_seconds": time.monotonic() - start,
            "AIME_opened": False, "teacher_dataset_sha256": file_sha(args.data),
        }
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "elapsed_seconds": time.monotonic() - start})
        raise
    finally:
        launch.close()
    write_json(root / "COMPLETE.json", completion)
    print("Completed", root, reports, flush=True)


if __name__ == "__main__":
    main()
