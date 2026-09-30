"""Run the four released Engineer/Solver pairings on matched ALFWorld packets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[2]
ASSETS = PROJECT / "assets/alfworld_harness_r1"
RUNS = ASSETS / "runs"

TARGETS = {
    "original": {
        "ports": (19101, 19111, 19121, 19131),
        "record": PROJECT / "assets/Qwen3.5-9B/download_receipt.json",
        "label": "original-qwen3.5-9b",
    },
    "agent_sft": {
        "ports": (19301, 19311, 19321, 19331),
        "record": PROJECT / "assets/Harness-R1/target-agent-sft/provenance.json",
        "label": "target-agent-sft",
    },
}
ENGINEERS = {
    "rl": {
        "ports": (19201, 19211, 19221, 19231),
        "record": PROJECT / "assets/Harness-R1/harness-r1/provenance.json",
    },
    "cold_sft": {
        "ports": (19401, 19411, 19421, 19431),
        "record": PROJECT / "assets/Harness-R1/cold-start-sft/provenance.json",
    },
}
PAIRINGS = (
    ("rl_agent_sft", "agent_sft", "rl"),
    ("cold_sft_agent_sft", "agent_sft", "cold_sft"),
    ("rl_original", "original", "rl"),
    ("cold_sft_original", "original", "cold_sft"),
)


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def infrastructure_failures(summary: dict) -> int:
    return int(summary.get("categories", {}).get("infra_failure", 0))


def jobs(prefix: str, starts: list[int], size: int, concurrency: int) -> list[dict]:
    if len(starts) != 4 or len(set(starts)) != 4:
        raise ValueError("exactly four distinct packet starts are required")
    result = []
    for pairing_index, (name, target_name, engineer_name) in enumerate(PAIRINGS):
        target, engineer = TARGETS[target_name], ENGINEERS[engineer_name]
        for replica, start in enumerate(starts):
            if start < 0 or start + size > 500:
                raise ValueError("packet does not fit the official 500-task split")
            index = pairing_index * len(starts) + replica
            base_port = 22000 + index * 40
            result.append({
                "index": index,
                "pairing": name,
                "target": target_name,
                "engineer": engineer_name,
                "start": start,
                "size": size,
                "concurrency": concurrency,
                "target_port": target["ports"][replica],
                "engineer_port": engineer["ports"][replica],
                "target_record": str(target["record"]),
                "target_label": target["label"],
                "engineer_record": str(engineer["record"]),
                "controller_port": base_port,
                "worker_port": base_port + 10,
                "run_id": f"{prefix}_{name}_start{start}",
            })
    return result


def command(job: dict, attempt: int) -> list[str]:
    port_offset = 1000 * (attempt - 1)
    run_id = job["run_id"] + f"_attempt{attempt}"
    return [
        sys.executable, "-B", "-m", "whale_local.harness_r1_alfworld.driver", "paired",
        "--python", "/dev/shm/harness-r1-alfworld-venv/bin/python",
        "--size", str(job["size"]), "--task-start", str(job["start"]),
        "--run-id", run_id,
        "--run-root", job.get("run_root", str(RUNS)),
        "--target-url", f"http://127.0.0.1:{job['target_port']}/v1",
        "--target-model", "policy",
        "--target-agent-name", "qwen35-9b-nothink",
        "--target-checkpoint-label", job["target_label"],
        "--target-checkpoint-record", job["target_record"],
        "--engineer-url", f"http://127.0.0.1:{job['engineer_port']}/v1",
        "--engineer-model", "policy",
        "--engineer-checkpoint-record", job["engineer_record"],
        "--concurrency", str(job["concurrency"]),
        "--controller-port", str(job["controller_port"] + port_offset),
        "--worker-port", str(job["worker_port"] + port_offset),
    ]


def terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)


def run_job(job: dict, root: Path, attempts: int, timeout: int) -> dict:
    record = {**job, "status": "running", "attempts": []}
    save(root / "jobs" / f"{job['index']:02d}.json", record)
    for attempt in range(1, attempts + 1):
        cmd = command(job, attempt)
        run_id = cmd[cmd.index("--run-id") + 1]
        log_path = root / "logs" / f"{run_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                cmd, cwd=PROJECT, env=dict(os.environ, CUDA_VISIBLE_DEVICES="",
                                           PYTHONDONTWRITEBYTECODE="1"),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate(process)
                returncode = process.returncode
        output = Path(job.get("run_root", RUNS)) / run_id
        attempt_record = {
            "attempt": attempt, "pid": process.pid, "returncode": returncode,
            "timed_out": timed_out, "started_at": started,
            "elapsed_seconds": time.time() - started, "command": cmd,
            "log": str(log_path), "output": str(output),
        }
        summary = output / "adapter_summary.json"
        if returncode == 0 and summary.is_file():
            attempt_record["summary"] = json.loads(summary.read_text())
            infra_failures = infrastructure_failures(attempt_record["summary"])
            if infra_failures == 0:
                record.update(status="completed", selected_attempt=attempt)
                record["attempts"].append(attempt_record)
                save(root / "jobs" / f"{job['index']:02d}.json", record)
                return record
            attempt_record["failure"] = {
                "error": "adapter summary contains infrastructure failures",
                "infra_failure": infra_failures,
            }
        if "failure" not in attempt_record:
            attempt_record["failure"] = (
                json.loads((output / "adapter_failure.json").read_text())
                if (output / "adapter_failure.json").is_file() else
                {"error": "missing successful adapter summary"}
            )
        record["attempts"].append(attempt_record)
        save(root / "jobs" / f"{job['index']:02d}.json", record)
    record["status"] = "failed"
    save(root / "jobs" / f"{job['index']:02d}.json", record)
    return record


def monitor_gpus(path: Path, stop: threading.Event, interval: float) -> None:
    fields = "timestamp,index,utilization.gpu,memory.used,power.draw"
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            text=True, capture_output=True,
        )
        with path.open("a") as stream:
            stream.write(json.dumps({
                "sampled_at": time.time(), "returncode": result.returncode,
                "rows": result.stdout.splitlines(), "error": result.stderr.strip(),
            }) + "\n")
        stop.wait(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--starts", default="0,10,20,30")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--monitor-seconds", type=float, default=5)
    args = parser.parse_args()
    starts = [int(value) for value in args.starts.split(",")]
    planned = jobs(args.prefix, starts, args.size, args.concurrency)
    root = RUNS / args.prefix
    root.mkdir(parents=True, exist_ok=False)
    (root / "claim.json").write_text(json.dumps({
        "pid": os.getpid(), "host": os.uname().nodename, "started_at": time.time(),
    }) + "\n")
    save(root / "plan.json", planned)
    stop = threading.Event()
    monitor = threading.Thread(
        target=monitor_gpus, args=(root / "gpu_samples.jsonl", stop, args.monitor_seconds),
        daemon=True,
    )
    monitor.start()
    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=len(planned)) as pool:
            futures = {
                pool.submit(run_job, job, root, args.attempts, args.timeout_seconds): job
                for job in planned
            }
            results = [future.result() for future in as_completed(futures)]
    finally:
        stop.set()
        monitor.join(timeout=args.monitor_seconds + 5)
    results.sort(key=lambda row: row["index"])
    summary = {
        "status": "completed" if all(row["status"] == "completed" for row in results)
        else "failed",
        "started_at": started, "elapsed_seconds": time.time() - started,
        "completed": sum(row["status"] == "completed" for row in results),
        "failed": sum(row["status"] == "failed" for row in results),
        "results": results,
    }
    save(root / "COMPLETE.json", summary)
    if summary["status"] != "completed":
        raise RuntimeError("one or more matrix jobs failed after bounded retries")


if __name__ == "__main__":
    main()
