"""Collect one Engineer/Solver pairing over arbitrary contiguous ALFWorld packets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time

from .matrix_probe import monitor_gpus, run_job, save


def packet_jobs(prefix: str, starts: list[int], size: int, concurrency: int,
                target_ports: list[int], engineer_ports: list[int],
                target_record: Path, target_label: str, engineer_record: Path,
                run_root: Path) -> list[dict]:
    if not starts or len(starts) != len(set(starts)):
        raise ValueError("packet starts must be nonempty and distinct")
    if not target_ports or len(target_ports) != len(engineer_ports):
        raise ValueError("target and Engineer replica lists must have equal nonzero length")
    jobs = []
    for index, start in enumerate(starts):
        if start < 0 or start + size > 500:
            raise ValueError("packet does not fit the official 500-task split")
        replica = index % len(target_ports)
        base_port = 36000 + index * 80
        jobs.append({
            "index": index, "pairing": "rl_agent_sft",
            "target": "current_solver", "engineer": "current_engineer",
            "start": start, "size": size, "concurrency": concurrency,
            "target_port": target_ports[replica],
            "engineer_port": engineer_ports[replica],
            "target_record": str(target_record.resolve()),
            "target_label": target_label,
            "engineer_record": str(engineer_record.resolve()),
            "controller_port": base_port, "worker_port": base_port + 20,
            "run_id": f"{prefix}_start{start}",
            "run_root": str(run_root.resolve()),
        })
    return jobs


def comma_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--starts", required=True)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--target-ports", required=True)
    parser.add_argument("--engineer-ports", required=True)
    parser.add_argument("--target-record", type=Path, required=True)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--engineer-record", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--monitor-seconds", type=float, default=5)
    args = parser.parse_args()
    planned = packet_jobs(
        args.prefix, comma_ints(args.starts), args.size, args.concurrency,
        comma_ints(args.target_ports), comma_ints(args.engineer_ports),
        args.target_record, args.target_label, args.engineer_record, args.run_root,
    )
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.run_root.resolve().mkdir(parents=True, exist_ok=True)
    save(root / "plan.json", planned)
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpus, args=(root / "gpu_samples.jsonl", stop, args.monitor_seconds),
        daemon=True,
    )
    monitor.start()
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers or len(planned)) as pool:
            futures = {
                pool.submit(
                    run_job, job, root, args.attempts, args.timeout_seconds,
                ): job
                for job in planned
            }
            results = [future.result() for future in as_completed(futures)]
    finally:
        stop.set()
        monitor.join(timeout=args.monitor_seconds + 5)
    results.sort(key=lambda row: row["index"])
    receipt = {
        "status": "completed" if all(row["status"] == "completed" for row in results)
        else "failed",
        "started_at": started, "elapsed_seconds": time.time() - started,
        "completed": sum(row["status"] == "completed" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    save(root / "COMPLETE.json", receipt)
    if receipt["status"] != "completed":
        raise RuntimeError("one or more ALFWorld packet jobs failed after bounded retries")


if __name__ == "__main__":
    main()
