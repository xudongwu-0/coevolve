"""Session-independent, read-only five-minute GPU history for node3 and node4."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta
import fcntl
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from .common import ROOT, file_sha, read_json, write_json
from .gpu_telemetry import parse_pmon
from .resources import audit, process_identity

BEIJING = ZoneInfo("Asia/Shanghai")
INTERVAL = 300
MAX_GAP = 360
HOSTS = {"node3": "dev-NDH200-025", "node4": "dev-NDH200-026"}
SSH = ["ssh", "-F", "/mnt/3fs/xudong/cluster-ssh/config",
       "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "node3-internal"]
SCHEMA = "coevolve-readonly-gpu-history-v1"
ELIGIBLE = {"idle", "verified_filler_only"}
SECRET = re.compile(r"(?:hf_[A-Za-z0-9]{15,}|sk-(?:proj-)?[A-Za-z0-9_-]{15,})")
LAUNCHERS = {
    "torch.distributed.run", "torch.distributed.launch",
    "vllm.entrypoints.openai.api_server", "sao_bridge.large_train",
}


def beijing(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, BEIJING).isoformat(timespec="seconds")


def safe_text(value: str) -> str:
    return SECRET.sub("[REDACTED]", value)[:2000]


def process_summary(record: dict) -> dict:
    args = record["cmd"]
    scripts = [Path(arg).name for arg in args[1:] if arg.endswith((".py", ".sh"))
               and not any(char in arg for char in ("\n", " ", ";", "://"))]
    modules = [args[index + 1] for index, arg in enumerate(args[:-1]) if arg == "-m"]
    try:
        username = pwd.getpwuid(record["uid"]).pw_name
    except KeyError:
        username = str(record["uid"])
    return {
        "pid": record["pid"], "uid": record["uid"], "user": username,
        "executable": safe_text(Path(args[0]).name) if args else None,
        "scripts": [safe_text(name) for name in scripts],
        "modules": [safe_text(name) for name in modules],
        "status": "identified", "full_arguments_or_environment_recorded": False,
    }


def launcher_inventory() -> list[dict]:
    import psutil

    result = []
    for proc in psutil.process_iter():
        try:
            args = proc.cmdline()
            module = args[args.index("-m") + 1] if "-m" in args and args.index("-m") + 1 < len(args) else None
            if args and (Path(args[0]).name in {"torchrun", "deepspeed", "mpirun"} or module in LAUNCHERS):
                result.append(process_summary({"pid": proc.pid, "uid": proc.uids().real, "cmd": args}))
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise PermissionError(f"Cannot identify possible GPU launcher PID{proc.pid}") from exc
    return result


def query(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=30).stdout


def gpu_rows(text: str) -> list[dict]:
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if len(row) != 8:
            raise ValueError("GPU history received an incomplete GPU metric row")
        index, gpu_uuid, name, utilization, used, total, power, temperature = (value.strip() for value in row)
        value = {
            "index": int(index), "uuid": gpu_uuid, "name": name, "utilization_percent": int(utilization),
            "memory_used_mib": int(used), "memory_total_mib": int(total),
            "power_watts": float(power), "temperature_c": int(temperature),
        }
        if (not 0 <= value["utilization_percent"] <= 100
                or not 0 <= value["memory_used_mib"] <= value["memory_total_mib"]
                or not math.isfinite(value["power_watts"])):
            raise ValueError("GPU history received invalid GPU metrics")
        rows.append(value)
    if len(rows) != 8 or {row["index"] for row in rows} != set(range(8)) or len({row["uuid"] for row in rows}) != 8:
        raise ValueError("GPU history requires eight uniquely identified GPUs")
    return sorted(rows, key=lambda row: row["index"])


def classification(node: dict) -> str:
    if node["unresolved_gpu_pids"]:
        return "unverifiable"
    if node["non_filler_gpu_pids"] or node["distributed_launchers"]:
        return "other_workload"
    if node["gpu_pids"]:
        return "verified_filler_only" if (
            node["filler_unit_verified"] is True and node["filler_kind"] in ("legacy", "adaptive")
            and set(node["gpu_pids"]) == set(node["verified_filler_pids"])
        ) else "unverifiable"
    return "idle" if all(
        gpu["utilization_percent"] == gpu["memory_used_mib"] == 0 for gpu in node["gpus"]
    ) else "unattributed_gpu_activity"


def probe_node() -> dict:
    host = os.uname().nodename
    if host not in HOSTS.values():
        raise RuntimeError("GPU history may only observe the two declared research nodes")
    identity = audit()
    gpus = gpu_rows(query([
        "nvidia-smi", "--query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]))
    indices = {gpu["uuid"]: gpu["index"] for gpu in gpus}
    applications = []
    for row in csv.reader(io.StringIO(query([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]))):
        if len(row) != 3:
            raise ValueError("GPU history received an incomplete process row")
        gpu_uuid, pid, memory = (part.strip() for part in row)
        applications.append({"gpu": indices[gpu_uuid], "pid": int(pid), "memory_mib": int(memory)})
    pmon = parse_pmon(query(["nvidia-smi", "pmon", "-s", "um", "-c", "1"]))
    pids = sorted({item["pid"] for item in applications + pmon + identity["processes"]})
    known = {item["pid"]: item for item in identity["processes"]}
    filler = {
        item["pid"] for item in identity["processes"]
        if identity["verified_unit"] and item["verified_filler"]
    }
    processes, unresolved = [], set(identity["vanished_gpu_pids"])
    for pid in pids:
        try:
            current = process_identity(pid)
            stable = pid in known and current == {
                key: known[pid][key] for key in ("pid", "uid", "cmd", "cgroup")
            }
            record = process_summary(current)
            record["kind"] = "verified_filler" if stable and pid in filler else "other_workload"
            if pid in known and not stable:
                unresolved.add(pid)
                record["kind"] = "identity_changed"
        except FileNotFoundError:
            unresolved.add(pid)
            record = {"pid": pid, "status": "vanished_during_probe", "kind": "unverifiable"}
        except PermissionError:
            unresolved.add(pid)
            record = {"pid": pid, "status": "permission_denied", "kind": "unverifiable"}
        record["gpu_allocations"] = [item for item in applications if item["pid"] == pid]
        record["utilization_samples"] = [
            {key: item[key] for key in ("gpu", "type", "sm_percent", "memory_percent")}
            for item in pmon if item["pid"] == pid
        ]
        processes.append(record)
    now = time.time()
    result = {
        "status": "ok", "host": host, "observed_at_unix": now, "observed_at_beijing": beijing(now),
        "gpus": gpus, "processes": processes, "gpu_pids": pids,
        "verified_filler_pids": [item["pid"] for item in processes if item["kind"] == "verified_filler"],
        "non_filler_gpu_pids": [item["pid"] for item in processes if item["kind"] == "other_workload"],
        "unresolved_gpu_pids": sorted(unresolved), "distributed_launchers": launcher_inventory(),
        "filler_unit_verified": identity["verified_unit"], "filler_kind": identity["filler_kind"],
    }
    result["classification"] = classification(result)
    return result


def collect_nodes() -> dict:
    command = [
        "env", "CUDA_VISIBLE_DEVICES=", "HF_HUB_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={ROOT}/whale_local:{ROOT}/third_party/sao/vendor/AReaL",
        sys.executable, "-B", "-m", "sao_bridge.gpu_history", "probe",
    ]

    def collect(name: str) -> dict:
        try:
            response = subprocess.run(
                command if name == "node4" else [*SSH, shlex.join(command)],
                check=True, capture_output=True, text=True, timeout=90,
            )
            node = json.loads(response.stdout)
            if node["host"] != HOSTS[name] or abs(time.time() - node["observed_at_unix"]) > 90:
                raise ValueError("Wrong node identity or stale GPU history response")
            if node["classification"] != classification(node):
                raise ValueError("GPU history classification does not match its evidence")
            return node
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.SubprocessError) as exc:
            now = time.time()
            detail = safe_text(str(exc))
            if isinstance(exc, subprocess.CalledProcessError):
                detail += "\n" + safe_text(exc.stderr or "")
            print(f"{beijing(now)} PROBE ERROR {name}: {detail}", file=sys.stderr, flush=True)
            return {
                "status": "error", "host": HOSTS[name], "classification": "probe_error",
                "observed_at_unix": now, "observed_at_beijing": beijing(now),
                "error_type": type(exc).__name__, "error": detail,
            }

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {name: pool.submit(collect, name) for name in HOSTS}
        return {name: future.result() for name, future in futures.items()}


def append_sample(root: Path, sample: dict) -> Path:
    date = datetime.fromtimestamp(sample["recorded_at_unix"], BEIJING).date().isoformat()
    path = root / "samples" / f"{date}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def history_rows(root: Path, since: float, now: float) -> list[dict]:
    first = datetime.fromtimestamp(since - MAX_GAP, BEIJING).date()
    last = datetime.fromtimestamp(now, BEIJING).date()
    result = []
    date = first
    while date <= last:
        path = root / "samples" / f"{date.isoformat()}.jsonl"
        if path.exists():
            with path.open() as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Malformed GPU history: {path}:{line_number}") from exc
                    if row["schema"] != SCHEMA:
                        raise ValueError(f"Unexpected GPU history schema: {path}:{line_number}")
                    if row["recorded_at_unix"] >= since - MAX_GAP:
                        result.append(row)
        date += timedelta(days=1)
    return result


def summarize(rows: list[dict], *, now: float, hours: float) -> dict:
    if not math.isfinite(hours) or hours <= 0:
        raise ValueError("History hours must be a finite positive number")
    required = hours * 3600
    latest = rows[-1] if rows else None
    fresh = latest is not None and 0 <= now - latest["recorded_at_unix"] <= MAX_GAP
    suffix = []
    if fresh:
        for row in reversed(rows):
            if set(row["nodes"]) != set(HOSTS) or not all(
                node["status"] == "ok" and node["classification"] in ELIGIBLE for node in row["nodes"].values()
            ):
                break
            if suffix:
                newer = suffix[-1]
                delta = newer["recorded_at_unix"] - row["recorded_at_unix"]
                monotonic_delta = newer["recorded_at_monotonic"] - row["recorded_at_monotonic"]
                if (
                    row["recorder_id"] != newer["recorder_id"]
                    or newer["sequence"] != row["sequence"] + 1 or not 0 < delta <= MAX_GAP
                    or abs(delta - monotonic_delta) > 60
                ):
                    break
            suffix.append(row)
    span = min(
        suffix[0]["recorded_at_unix"] - suffix[-1]["recorded_at_unix"],
        suffix[0]["recorded_at_monotonic"] - suffix[-1]["recorded_at_monotonic"],
    ) if len(suffix) > 1 else 0
    required_samples = math.ceil(required / INTERVAL) + 1
    return {
        "queried_at_beijing": beijing(now), "requested_hours": hours,
        "available_samples": len(rows), "latest_sample_beijing": latest["recorded_at_beijing"] if latest else None,
        "latest_sample_fresh": fresh, "trailing_idle_or_filler_samples": len(suffix),
        "trailing_idle_or_filler_seconds": span,
        "trailing_start_beijing": suffix[-1]["recorded_at_beijing"] if suffix else None,
        "required_samples": required_samples, "required_span_seconds": required,
        "meets_requested_window": fresh and len(suffix) >= required_samples and span >= required,
        "latest_node_states": {
            name: node["classification"] for name, node in latest["nodes"].items()
        } if latest else {},
        "read_only_history_not_launch_authority": True,
        "sampling_limit": "Five-minute observations cannot rule out work between samples; recheck live resources before launch.",
    }


def next_sample_time(previous: float, now: float) -> float:
    scheduled = previous + INTERVAL
    return scheduled if scheduled > now else now + INTERVAL


def record(root: Path) -> None:
    if os.uname().nodename != HOSTS["node4"] or os.getuid() != 1008:
        raise RuntimeError("The detached GPU-history recorder belongs to the node4 research owner")
    os.umask(0o077)
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or (root / "STOP.json").exists():
        raise ValueError("History directory must be an ordinary, explicitly unpaused directory")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop.set())
    with (root / "RECORDER.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        recorder_id = uuid.uuid4().hex
        started = {
            "schema": SCHEMA, "pid": os.getpid(), "recorder_id": recorder_id,
            "started_at_beijing": beijing(time.time()), "interval_seconds": INTERVAL,
            "source": str(Path(__file__).resolve()), "source_sha256": file_sha(Path(__file__)),
            "session_independent": True, "training_submission_allowed": False,
            "time_zone": "Asia/Shanghai", "nodes": HOSTS,
            "pid_start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
        }
        write_json(root / "RECORDER.json", started)
        sequence = 0
        next_sample = time.monotonic()
        try:
            while not stop.is_set() and not (root / "STOP.json").exists():
                if time.monotonic() < next_sample:
                    stop.wait(max(0, min(5, next_sample - time.monotonic())))
                    continue
                sequence += 1
                nodes = collect_nodes()
                now, monotonic = time.time(), time.monotonic()
                sample = {
                    "schema": SCHEMA, "recorder_id": recorder_id, "sequence": sequence,
                    "interval_seconds": INTERVAL, "recorded_at_unix": now,
                    "recorded_at_monotonic": monotonic, "recorded_at_beijing": beijing(now),
                    "nodes": nodes,
                    "both_idle_or_verified_filler": all(
                        node["status"] == "ok" and node["classification"] in ELIGIBLE for node in nodes.values()
                    ),
                }
                path = append_sample(root, sample)
                next_sample = next_sample_time(next_sample, monotonic)
                write_json(root / "LATEST.json", sample)
                write_json(root / "STATUS.json", {
                    **started, "status": "recording", "sequence": sequence,
                    "last_sample_beijing": beijing(now),
                    "next_sample_beijing": beijing(now + next_sample - monotonic),
                    "last_sample_file": str(path),
                    "node_states": {name: node["classification"] for name, node in nodes.items()},
                })
                print(f"{beijing(now)} sample={sequence} "
                      + " ".join(f"{name}={node['classification']}" for name, node in nodes.items()), flush=True)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            write_json(root / "STATUS.json", {
                **started, "status": "failed", "error": safe_text(str(exc)),
                "stopped_at_beijing": beijing(time.time()),
            })
            raise
        write_json(root / "STATUS.json", {
            **started, "status": "stopped", "sequence": sequence,
            "stopped_at_beijing": beijing(time.time()),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("probe", "record", "summary"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--hours", type=float, default=2)
    args = parser.parse_args()
    if args.action == "probe":
        print(json.dumps(probe_node(), ensure_ascii=False))
        return
    if args.root is None or not args.root.is_absolute():
        parser.error("--root must be an absolute history directory")
    if args.action == "record":
        record(args.root)
        return
    if not math.isfinite(args.hours) or args.hours <= 0:
        parser.error("--hours must be finite and positive")
    now = time.time()
    result = summarize(history_rows(args.root, now - args.hours * 3600, now), now=now, hours=args.hours)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
