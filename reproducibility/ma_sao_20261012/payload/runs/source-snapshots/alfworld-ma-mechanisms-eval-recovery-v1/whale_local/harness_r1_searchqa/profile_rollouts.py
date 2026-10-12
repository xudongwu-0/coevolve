"""Measure real train-episode throughput, not synthetic GPU activity."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.evaluation import Sampler
from .episode import evaluate_rows
from .patch import base_patch
from .reward import _rows_by_id


class MeasuredSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latencies: list[float] = []

    async def __call__(self, *args, **kwargs):
        started = time.monotonic()
        result = await super().__call__(*args, **kwargs)
        self.latencies.append(time.monotonic() - started)
        return result


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-urls", required=True)
    parser.add_argument("--retrieval-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=128)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--concurrencies", default="64,128,256")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    levels = [int(value) for value in args.concurrencies.split(",")]
    if args.output.exists() or min(levels + [args.questions, args.samples]) < 1:
        raise ValueError("Require fresh output and positive workload sizes")
    train = _rows_by_id(str(args.data_root))
    questions = [train[key] for key in sorted(train)[:args.questions]]
    if len(questions) != args.questions:
        raise ValueError("Insufficient canonical train questions")
    rows, seeds = [], []
    for row in questions:
        for sample in range(args.samples):
            rows.append(row)
            seeds.append(args.seed + int(row["id"][:8], 16) % 1_000_000 + 1009 * sample)
    args.output.mkdir(parents=True)
    os.environ["SAO_SEARCHQA_RETRIEVAL_URL"] = args.retrieval_url
    write_json(args.output / "IDENTITY.json", {
        "purpose": "performance_only_no_optimizer_or_acceptance",
        "split": "train", "manifest_sha256": file_sha(args.data_root / "manifest.json"),
        "task_ids": [row["id"] for row in questions], "samples": args.samples,
        "target_urls": args.target_urls, "solver_tokens": 4096, "temperature": 1.0,
    })
    profiles = []
    for concurrency in levels:
        sampler = MeasuredSampler(args.target_urls, concurrency=concurrency, temperature=1.0,
                          chat_template_kwargs={"enable_thinking": False})
        telemetry, latencies = [], []
        started = time.monotonic()

        async def monitor() -> None:
            while True:
                command = await asyncio.create_subprocess_exec(
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits", stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = await command.communicate()
                if command.returncode:
                    raise RuntimeError(f"GPU telemetry failed: {stderr.decode()}")
                telemetry.append({
                    "seconds": time.monotonic() - started,
                    "gpus": [[float(value) for value in line.split(",")]
                             for line in stdout.decode().splitlines()],
                })
                await asyncio.sleep(1)

        task = asyncio.create_task(monitor())

        def record(index: int, result: dict) -> None:
            latencies.append(time.monotonic() - started)

        try:
            report = await evaluate_rows(
                rows, sampler=sampler, patch=base_patch(), total_tokens=4096,
                rollout_seeds=seeds, concurrency=concurrency, on_result=record,
            )
            elapsed = time.monotonic() - started
            profile = {
                "concurrency": concurrency, "episodes": len(rows), "seconds": elapsed,
                "episodes_per_second": len(rows) / elapsed,
                "generated_tokens": sampler.generated_tokens,
                "generated_tokens_per_second": sampler.generated_tokens / elapsed,
                "requests": sampler.requests, "mean_reward": report["mean"],
                "completion_elapsed_p50": statistics.median(latencies),
                "completion_elapsed_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
                "request_latency_p50": statistics.median(sampler.latencies),
                "request_latency_p95": sorted(sampler.latencies)[int(0.95 * (len(sampler.latencies) - 1))],
                "gpu_telemetry": telemetry, "source": "real_train_h0_episodes",
            }
            write_json(args.output / f"concurrency-{concurrency}.json", profile)
            profiles.append(profile)
            print(json.dumps({key: value for key, value in profile.items()
                              if key != "gpu_telemetry"}), flush=True)
        finally:
            await sampler.close()
            if task.done():
                task.result()
            else:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    write_json(args.output / "COMPLETE.json", {
        "profiles": profiles, "optimizer_steps": 0, "benchmark_gain_claimed": False,
        "best_concurrency": max(profiles, key=lambda row: row["generated_tokens_per_second"])["concurrency"],
    })


if __name__ == "__main__":
    asyncio.run(main())
