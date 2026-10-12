"""Per-process GPU telemetry that separates training from the adaptive filler."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import threading
import time

from .resources import audit


def parse_pmon(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"Unexpected nvidia-smi pmon row: {line!r}")
        gpu, pid, process_type, sm, memory = fields[:5]
        if gpu.isdigit() and pid == "-":
            continue
        if not gpu.isdigit() or not pid.isdigit():
            raise ValueError(f"Invalid nvidia-smi pmon identity: {line!r}")

        def percent(value: str) -> int | None:
            if value == "-":
                return None
            parsed = int(value)
            if not 0 <= parsed <= 100:
                raise ValueError(f"Invalid pmon percentage: {value}")
            return parsed

        rows.append({
            "gpu": int(gpu),
            "pid": int(pid),
            "type": process_type,
            "sm_percent": percent(sm),
            "memory_percent": percent(memory),
            "command": fields[-1] if len(fields) >= 10 else None,
        })
    return rows


def _gpu_totals() -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",")]
        if len(values) != 5:
            raise ValueError(f"Unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, utilization, memory, power = values
        rows.append({
            "gpu": int(index),
            "uuid": uuid,
            "total_sm_percent": int(utilization),
            "memory_used_mib": int(memory),
            "power_watts": float(power),
        })
    if [row["gpu"] for row in rows] != list(range(8)):
        raise RuntimeError("Expected exactly eight ordered GPUs")
    return rows


def sample() -> dict:
    identity = audit()
    filler_pids = {
        row["pid"] for row in identity["processes"] if row["verified_filler"]
    }
    result = subprocess.run(
        ["nvidia-smi", "pmon", "-s", "um", "-c", "1"],
        capture_output=True, text=True, check=True,
    )
    processes = parse_pmon(result.stdout)
    for row in processes:
        row["kind"] = "filler" if row["pid"] in filler_pids else "experiment"
    gpus = _gpu_totals()
    for gpu in gpus:
        matching = [row for row in processes if row["gpu"] == gpu["gpu"]]
        gpu["experiment_sm_percent"] = min(100, sum(
            row["sm_percent"] or 0 for row in matching if row["kind"] == "experiment"
        ))
        gpu["filler_sm_percent"] = min(100, sum(
            row["sm_percent"] or 0 for row in matching if row["kind"] == "filler"
        ))
        gpu["experiment_processes"] = sum(
            row["kind"] == "experiment" for row in matching
        )
        gpu["filler_processes"] = sum(row["kind"] == "filler" for row in matching)
    return {
        "sampled_at": time.time(),
        "filler_unit_active": identity["unit"]["ActiveState"] == "active",
        "filler_kind": identity["filler_kind"],
        "verified_filler_pids": sorted(filler_pids),
        "processes": processes,
        "gpus": gpus,
        "mean_total_sm_percent": sum(
            row["total_sm_percent"] for row in gpus
        ) / len(gpus),
        "mean_experiment_sm_percent": sum(
            row["experiment_sm_percent"] for row in gpus
        ) / len(gpus),
        "mean_filler_sm_percent": sum(
            row["filler_sm_percent"] for row in gpus
        ) / len(gpus),
    }


def monitor(path: Path, stop: threading.Event, interval: float = 5.0) -> None:
    if interval <= 0:
        raise ValueError("GPU telemetry interval must be positive")
    while not stop.is_set():
        try:
            record = sample()
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            record = {
                "sampled_at": time.time(),
                "error": repr(exc),
            }
        with path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        stop.wait(interval)


def summarize(
    path: Path,
    *,
    start_time: float | None = None,
    end_time: float | None = None,
) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if start_time is not None:
        rows = [
            row for row in rows
            if float(row.get("sampled_at", float("-inf"))) >= start_time
        ]
    if end_time is not None:
        rows = [
            row for row in rows
            if float(row.get("sampled_at", float("inf"))) <= end_time
        ]
    valid = [row for row in rows if "error" not in row]
    if not valid:
        raise ValueError("GPU telemetry has no valid samples")

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in valid) / len(valid)

    summary = {
        "samples": len(rows),
        "valid_samples": len(valid),
        "error_samples": len(rows) - len(valid),
        "mean_total_sm_percent": mean("mean_total_sm_percent"),
        "mean_experiment_sm_percent": mean("mean_experiment_sm_percent"),
        "mean_filler_sm_percent": mean("mean_filler_sm_percent"),
        "filler_remained_active": all(row["filler_unit_active"] for row in valid),
        "interpretation": (
            "experiment SM is the non-filler per-process pmon signal; total SM may "
            "include the adaptive filler and must not be reported as algorithm utilization"
        ),
    }
    active = [
        row for row in valid
        if float(row["mean_experiment_sm_percent"]) >= 5.0
    ]
    summary["experiment_active_samples"] = len(active)
    summary["experiment_active_sample_fraction"] = len(active) / len(valid)
    if active:
        summary["mean_experiment_sm_percent_when_active"] = sum(
            float(row["mean_experiment_sm_percent"]) for row in active
        ) / len(active)
        summary["mean_filler_sm_percent_when_experiment_active"] = sum(
            float(row["mean_filler_sm_percent"]) for row in active
        ) / len(active)
    if all(len(row.get("gpus", [])) == 8 for row in valid):
        summary["peak_memory_used_mib_by_gpu"] = [
            max(row["gpus"][gpu]["memory_used_mib"] for row in valid)
            for gpu in range(8)
        ]
        summary["peak_memory_used_mib"] = max(
            summary["peak_memory_used_mib_by_gpu"]
        )
        summary["mean_power_watts_by_gpu"] = [
            sum(row["gpus"][gpu]["power_watts"] for row in valid) / len(valid)
            for gpu in range(8)
        ]
    return summary


def summarize_active_training(
    telemetry_path: Path,
    marker_path: Path,
) -> dict | None:
    if not marker_path.is_file():
        return None
    marker = json.loads(marker_path.read_text())
    if marker.get("status") != "completed":
        return None
    try:
        return summarize(
            telemetry_path,
            start_time=marker["started_at_unix"],
            end_time=marker["finished_at_unix"],
        )
    except ValueError as exc:
        return {"status": "invalid", "error": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--interval", type=float, default=5)
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be positive")
    stop = threading.Event()
    thread = threading.Thread(
        target=monitor, args=(args.output, stop, args.interval), daemon=True,
    )
    thread.start()
    time.sleep(args.duration)
    stop.set()
    thread.join(timeout=args.interval + 10)
    result = summarize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
