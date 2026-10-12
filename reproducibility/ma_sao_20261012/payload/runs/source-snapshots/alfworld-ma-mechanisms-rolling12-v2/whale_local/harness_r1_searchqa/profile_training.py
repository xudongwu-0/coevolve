"""Qualify genuine full-parameter training throughput under audited GPU ownership."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.launch import Launcher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--data-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-sizes", default="128,1024")
    args = parser.parse_args()
    chunks = [int(value) for value in args.chunk_sizes.split(",")]
    if args.output.exists() or any(value < 1 for value in chunks):
        raise ValueError("Require fresh profiling output and positive loss chunks")
    if file_sha(args.data) != args.data_sha256:
        raise ValueError("Profiling dataset identity mismatch")
    args.output.mkdir(parents=True)
    write_json(args.output / "STATUS.json", {
        "stage": "profiling_native_4b_full_parameter_steps",
        "formal_rl_started": False, "benchmark_gain_claimed": False,
    })
    launch = Launcher(args.output / "processes")
    profiles = []
    try:
        for chunk in chunks:
            checkpoint = args.output / f"chunk-{chunk}"
            started = time.time()
            process = launch.start([
                sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                "--local-addr=127.0.0.1", "--nproc-per-node=8", "--max-restarts=0",
                "-m", "harness_r1_searchqa.sft_train",
                "--model", str(args.model), "--base-receipt-sha256", args.receipt_sha256,
                "--data", str(args.data), "--data-sha256", args.data_sha256,
                "--output", str(checkpoint), "--world-size", "8",
                "--global-batch-size", "8", "--epochs", "1",
                "--pilot-minimum-examples", "32", "--engineer-think-prefill",
                "--loss-chunk-size", str(chunk), "--purpose", "performance-profile",
            ], gpus="0,1,2,3,4,5,6,7", name=f"chunk_{chunk}")
            telemetry = []
            while process.poll() is None:
                launch.check()
                sample = subprocess.run([
                    "nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, check=True, timeout=20)
                telemetry.append({
                    "unix": time.time(),
                    "gpus": [[float(value) for value in line.split(",")]
                             for line in sample.stdout.splitlines()],
                })
                time.sleep(1)
            launch.wait(process, f"Full 4B profiling chunk {chunk}")
            manifest_path = checkpoint / "manifest.json"
            marker = (checkpoint / "COMPLETE").read_text().splitlines()
            if marker != ["VERIFIED COMPLETE", file_sha(manifest_path)]:
                raise RuntimeError("Profiling checkpoint was not atomically verified")
            manifest = json.loads(manifest_path.read_text())
            steps = manifest["performance"]["steps"]
            if len(steps) != 4 or manifest["training"]["updates"] != 4:
                raise RuntimeError("Profiling did not execute all real optimizer steps")
            seconds = sum(row["step_seconds"] for row in steps)
            profiles.append({
                "loss_chunk": chunk, "optimizer_steps": 4,
                "input_tokens_per_second": sum(row["input_tokens"] for row in steps) / seconds,
                "supervised_tokens_per_second": sum(row["supervised_tokens"] for row in steps) / seconds,
                "compute_seconds": seconds, "end_to_end_seconds": time.time() - started,
                "steps": steps, "gpu_telemetry": telemetry,
                "checkpoint_manifest_sha256": file_sha(manifest_path),
                "purpose": "performance_profile_not_rl_result",
            })
            write_json(args.output / f"profile-{chunk}.json", profiles[-1])
        write_json(args.output / "COMPLETE.json", {
            "profiles": profiles, "formal_rl_started": False,
            "benchmark_gain_claimed": False,
        })
        write_json(args.output / "STATUS.json", {"stage": "performance_profile_verified"})
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        write_json(args.output / "STATUS.json", {"stage": "blocked", "error": str(exc)})
        raise
    finally:
        launch.close()


if __name__ == "__main__":
    main()
