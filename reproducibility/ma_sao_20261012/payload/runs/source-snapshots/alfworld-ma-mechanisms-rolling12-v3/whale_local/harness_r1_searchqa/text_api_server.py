"""Standard vLLM OpenAI server for verified text-only Qwen3.5 checkpoints.

The Launcher must audit and reserve CUDA_VISIBLE_DEVICES before invocation.
This wrapper retains that selection and process/cgroup ownership; it starts no
independent launcher, monitor, or background process. Raw VLM serving must keep
using its existing entrypoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import sys


API_MODULE = "vllm.entrypoints.openai.api_server"
WORKER_EXTENSION = "harness_r1_searchqa.workflow.Qwen35TextWorkerExtension"


def _register_native_model():
    from .workflow import register_text_engine
    register_text_engine()


def validate_gpu_selection(visible: str | None, tensor_parallel_size: int):
    if not visible or tensor_parallel_size < 1:
        raise ValueError("Launcher must reserve explicit CUDA_VISIBLE_DEVICES and positive TP")
    devices = [value.strip() for value in visible.split(",")]
    if (any(not value or value == "-1" for value in devices)
            or len(set(devices)) != len(devices)
            or len(devices) < tensor_parallel_size):
        raise ValueError("CUDA selection must be unique and contain at least TP reserved GPUs")
    if any(not value.isdigit() and not value.startswith(("GPU-", "MIG-")) for value in devices):
        raise ValueError("CUDA selection must contain GPU indices or GPU/MIG UUIDs")


def prepare_argv(argv):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1",
                        choices=("127.0.0.1", "localhost", "::1"))
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gdn-prefill-backend", default="triton", choices=("triton",))
    parser.add_argument("--worker-extension-cls", default=WORKER_EXTENSION,
                        choices=(WORKER_EXTENSION,))
    parser.add_argument("--model-impl", default="vllm", choices=("vllm",))
    args, forwarded = parser.parse_known_args(argv)
    config = json.loads((args.model / "config.json").read_text())
    if (config.get("model_type") != "qwen3_5_text"
            or config.get("architectures") != ["Qwen3_5ForCausalLM"]):
        raise ValueError("this entrypoint requires a text-only Qwen3_5ForCausalLM checkpoint")
    if args.max_model_len < 1 or args.tensor_parallel_size < 1:
        raise ValueError("context length and tensor parallel size must be positive")
    normalized = [
        "--model", str(args.model), "--host", args.host, "--dtype", args.dtype,
        "--max-model-len", str(args.max_model_len),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gdn-prefill-backend", args.gdn_prefill_backend,
        "--worker-extension-cls", WORKER_EXTENSION, "--model-impl", "vllm",
    ]
    return args, normalized + forwarded


def main(argv=None):
    args, forwarded = prepare_argv(sys.argv[1:] if argv is None else argv)
    validate_gpu_selection(os.environ.get("CUDA_VISIBLE_DEVICES"), args.tensor_parallel_size)
    os.environ["HARNESS_R1_REGISTER_VLLM_TEXT"] = "1"
    os.environ["VLLM_PLUGINS"] = ""
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    _register_native_model()
    previous_argv = sys.argv
    try:
        sys.argv = [API_MODULE, *forwarded]
        # Keep this wrapper as multiprocessing's main module. Spawned processes
        # execute the guarded registration below, not a second HTTP server.
        runpy.run_module(API_MODULE, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = previous_argv


if os.environ.get("HARNESS_R1_REGISTER_VLLM_TEXT") == "1":
    _register_native_model()


if __name__ == "__main__":
    main()
