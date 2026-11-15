"""Run one explicitly bounded GRPO-only subset job on an idle node25/26 GPU."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT.parent
DEPENDENCIES = RUN.parent / "webshop_self_distillation_20261114"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", choices=("dev-NDH200-025", "dev-NDH200-026"), required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--arm", choices=("initial", "control", "reference"), required=True)
    parser.add_argument("--mode", choices=("train", "evaluate"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.name) or not 1 <= args.minutes <= 120:
        raise ValueError("Each named pilot must be bounded to1–120 minutes")
    if args.arm == "initial" and (args.mode != "evaluate" or args.adapter):
        raise ValueError("Initial arm evaluates only the untouched starting policy")
    if args.mode == "evaluate" and args.arm != "initial" and args.adapter is None:
        raise ValueError("Trained evaluation requires the exact pilot adapter")
    source = args.source.resolve()
    if not source.is_relative_to(RUN / "frozen"):
        raise ValueError("Execute only an immutable experiment-local source snapshot")
    sys.path.insert(0, str(source))
    from whale_local.webshop.core import Protocol
    from whale_local.webshop.runtime import file_sha256
    manifest = json.loads((source / "SOURCE_MANIFEST.json").read_text())
    for relative, expected in manifest["files"].items():
        if file_sha256(source / relative) != expected:
            raise ValueError("Frozen native source changed: " + relative)
    config = source / "configs" / (
        "webshop_sao_grpo_subset.json" if args.arm == "reference" else "webshop_sao_grpo_control.json")
    protocol = Protocol.from_dict(json.loads(config.read_text()))
    episodes = protocol.group_size * protocol.updates if args.mode == "train" else len(protocol.eval_ids) * len(protocol.eval_seeds)
    if protocol.distill_coefficient or protocol.updates > 16 or episodes > 128:
        raise ValueError("No distillation or full-data run is permitted")
    output = RUN / args.name
    if output.exists():
        raise ValueError("Do not overwrite or silently restart a pilot")
    cache = RUN / "runtime" / args.host / ("gpu" + str(args.gpu))
    for name in ("home", "tmp", "xdg", "triton", "torchinductor", "cuda"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    lease = {
        "purpose": "webshop-side-project", "hostname": args.host, "cuda_visible_devices": str(args.gpu),
        "expires_at": (now + timedelta(minutes=args.minutes, seconds=60)).isoformat(),
        "primary_sft_released": True, "modes": [args.mode], "max_episodes": episodes,
        "authorization": "Current user requested GRPO-only SAO improvements and bounded subset verification; no full run",
    }
    lease_path = RUN / (args.name + ".LEASE.json")
    lease_path.write_text(json.dumps(lease, indent=2) + "\n")
    java = DEPENDENCIES / "java/jdk-17.0.21+9"
    environment = {
        "CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONHASHSEED": "0",
        "JAVA_HOME": str(java), "PATH": str(java / "bin") + ":/usr/local/bin:/usr/bin:/bin",
        "JAVA_TOOL_OPTIONS": f"-Xmx2g -XX:ActiveProcessorCount=2 -XX:-UsePerfData -Djava.io.tmpdir={cache / 'tmp'}",
        "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false", "TMPDIR": str(cache / "tmp"), "HOME": str(cache / "home"),
        "XDG_CACHE_HOME": str(cache / "xdg"), "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"), "CUDA_CACHE_PATH": str(cache / "cuda"),
        "PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    }
    module = "whale_local.sao_bridge.webshop_grpo" if args.arm == "reference" else "whale_local.webshop"
    argv = [
        "sudo", "-n", "-u", "#1008", "-g", "#1008", "env",
        *[f"{key}={value}" for key, value in environment.items()],
        "timeout", "--signal=TERM", "--kill-after=30s", str(args.minutes * 60) + "s",
        str(DEPENDENCIES / "env/bin/python"), "-m", module, args.mode,
        "--config", str(config), "--env-root",
        str(DEPENDENCIES / "reference/source/agent_system/environments/env_package/webshop/webshop"),
        "--model", "/mnt/3fs/xudong/coevolve/assets/Qwen3.5-9B",
        "--lease", str(lease_path), "--output", str(output),
    ]
    if args.mode == "train" and args.arm == "control":
        argv += ["--method", "grpo"]
    if args.adapter is not None:
        argv += ["--adapter", str(args.adapter.resolve())]
    script = (
        f'test -z "$(nvidia-smi -i {args.gpu} --query-compute-apps=pid --format=csv,noheader)" || exit 73; '
        f"cd {shlex.quote(str(source))} && {shlex.join(argv)} > {shlex.quote(str(RUN / (args.name + '.log')))} 2>&1"
    )
    sys.path.insert(0, "/mnt/3fs/xudong/cowork-sft/base1019_ab40_followups_20261107_v1")
    import ops_1019
    status = {
        "host": args.host, "gpu": args.gpu, "arm": args.arm, "mode": args.mode,
        "source_commit": manifest["commit"], "source": str(source),
        "config_sha256": file_sha256(config), "started_at": now.isoformat(), "lease": lease,
        "status": "running", "output": str(output),
    }
    status_path = RUN / (args.name + ".STATUS.json")
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    started = time.monotonic()
    error = None
    try:
        ops_1019.remote(args.host, ["bash", "-lc", script], timeout=args.minutes * 60 + 90)
    except (RuntimeError, subprocess.TimeoutExpired) as failure:
        error = failure
        status.update(status="failed", error=str(failure))
    receipt = output / ("TRAINING.json" if args.mode == "train" else "REPORT.json")
    remaining = ops_1019.remote(args.host, ["nvidia-smi", "-i", str(args.gpu),
                                           "--query-compute-apps=pid", "--format=csv,noheader"], 30).strip()
    status.update(elapsed_seconds=time.monotonic() - started, complete_receipt=receipt.exists(),
                  gpu_empty_after_exit=not remaining)
    if receipt.exists():
        recorded = json.loads(receipt.read_text())
        status["metrics"] = recorded["metrics"]
    if error is None and receipt.exists() and not remaining:
        status["status"] = "complete"
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    if status["status"] != "complete":
        raise RuntimeError("Pilot incomplete or GPU still occupied: " + str(status_path)) from error


if __name__ == "__main__":
    main()
