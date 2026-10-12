"""Supervise loopback-only frozen model replicas with audited GPU ownership."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import signal
import threading
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.launch import Launcher
from sao_bridge.resources import require_research_node


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--model-file-sha256")
    identity.add_argument("--model-manifest", type=Path)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--context", type=int, default=16384)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--gdn-prefill-backend", choices=["triton", "flashinfer"])
    args = parser.parse_args()
    require_research_node()
    if args.model_manifest:
        expected = json.loads(args.model_manifest.read_text())
        actual_files = {path.name for path in args.model.glob("*.safetensors")}
        if actual_files != set(expected["weights"]):
            raise ValueError("Model shard filenames differ from preserved manifest")
        for name, digest in expected["weights"].items():
            if file_sha(args.model / name) != digest:
                raise ValueError(f"Frozen model shard identity changed: {name}")
    elif file_sha(args.model / "model.safetensors") != args.model_file_sha256:
        raise ValueError("Frozen model file hash differs from declared identity")
    devices = args.gpus.split(",")
    if len(devices) % args.tensor_parallel:
        raise ValueError("GPU count must be divisible by tensor parallel size")
    args.run.mkdir(parents=True, exist_ok=True)
    done = threading.Event()

    def stop(signum: int, frame: object) -> None:
        done.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    launcher = Launcher(args.run)
    try:
        with ExitStack() as stack:
            urls = stack.enter_context(launcher.vllm(
                str(args.model),
                port=args.port,
                gpus=args.gpus,
                name="frozen_model",
                tensor_parallel=args.tensor_parallel,
                data_parallel=len(devices) // args.tensor_parallel,
                max_model_len=args.context,
                server_args=(
                    ["--gdn-prefill-backend", args.gdn_prefill_backend]
                    if args.gdn_prefill_backend else None
                ),
            ))
            write_json(args.run / "READY.json", {
                "status": "ready", "pid": os.getpid(), "host": os.uname().nodename,
                "model": str(args.model), "model_file_sha256": args.model_file_sha256,
                "model_manifest": str(args.model_manifest) if args.model_manifest else None,
                "urls": urls, "gpus": devices, "context_tokens": args.context,
                "started_at_unix": time.time(), "binding": "loopback_only",
            })
            while not done.wait(5):
                launcher.check()
                if any(process.poll() is not None for process in launcher.children):
                    raise RuntimeError("A frozen model replica exited unexpectedly")
    finally:
        launcher.close()
        (args.run / "READY.json").unlink(missing_ok=True)
        write_json(args.run / "STOPPED.json", {
            "pid": os.getpid(), "stopped_at_unix": time.time(),
        })


if __name__ == "__main__":
    main()
