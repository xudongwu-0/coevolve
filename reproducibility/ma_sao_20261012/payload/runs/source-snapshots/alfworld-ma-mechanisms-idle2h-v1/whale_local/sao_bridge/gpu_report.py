"""Summarize eight-GPU telemetry without counting filler utilization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from .common import write_json

LAYOUTS = {
    "h": {"actor": [0, 1, 2], "rollout": [3], "frozen_solver": [4, 5, 6, 7]},
    "w": {"actor": [0, 1, 2, 3], "rollout": [4, 5, 6, 7]},
    "all": {"all": list(range(8))},
}


def summarize(path: Path, groups: dict[str, list[int]] | None = None) -> dict:
    samples = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows = {}
        for raw in record["rows"]:
            fields = [value.strip() for value in raw.split(",")]
            if len(fields) == 5:
                _, index, utilization, memory, power = fields
            elif len(fields) == 4:
                index, utilization, memory, power = fields
            else:
                raise ValueError(f"unexpected nvidia-smi telemetry row: {raw}")
            rows[int(index)] = {
                "utilization": float(utilization), "memory_mib": float(memory),
                "power_watts": float(power),
            }
        if set(rows) != set(range(8)):
            raise ValueError("GPU telemetry sample does not cover exactly devices 0..7")
        samples.append({"sampled_at": record["sampled_at"], "gpus": rows})
    if not samples:
        raise ValueError("GPU telemetry is empty")
    resident = [
        sample for sample in samples
        if all(sample["gpus"][index]["memory_mib"] >= 1024 for index in range(8))
    ]
    window = resident or samples

    def gpu_stats(source: list[dict], index: int) -> dict:
        util = [sample["gpus"][index]["utilization"] for sample in source]
        memory = [sample["gpus"][index]["memory_mib"] for sample in source]
        power = [sample["gpus"][index]["power_watts"] for sample in source]
        return {
            "mean_utilization": statistics.fmean(util),
            "median_utilization": statistics.median(util),
            "peak_utilization": max(util),
            "mean_memory_mib": statistics.fmean(memory),
            "peak_memory_mib": max(memory),
            "mean_power_watts": statistics.fmean(power),
        }

    groups = groups or {"all": list(range(8))}
    result = {
        "samples": len(samples), "resident_samples": len(resident),
        "elapsed_seconds": samples[-1]["sampled_at"] - samples[0]["sampled_at"],
        "all_samples": {str(index): gpu_stats(samples, index) for index in range(8)},
        "resident_window": {
            "per_gpu": {str(index): gpu_stats(window, index) for index in range(8)},
            "fraction_any_gpu_above_10pct": sum(
                any(sample["gpus"][index]["utilization"] > 10 for index in range(8))
                for sample in window
            ) / len(window),
            "fraction_all_gpus_above_10pct": sum(
                all(sample["gpus"][index]["utilization"] > 10 for index in range(8))
                for sample in window
            ) / len(window),
        },
        "groups": {},
        "warning": (
            "Includes model startup and environment/reward latency. Utilization is reported "
            "as infrastructure telemetry, never as algorithm quality."
        ),
    }
    for name, indices in groups.items():
        values = [
            sample["gpus"][index]["utilization"]
            for sample in window for index in indices
        ]
        result["groups"][name] = {
            "gpus": indices, "mean_utilization": statistics.fmean(values),
            "median_utilization": statistics.median(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layout", choices=["h", "w", "all"], default="all")
    args = parser.parse_args()
    write_json(args.output, summarize(args.input, LAYOUTS[args.layout]))


if __name__ == "__main__":
    main()
