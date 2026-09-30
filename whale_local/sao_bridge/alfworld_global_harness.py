"""Select or evaluate one global ALFWorld harness without target-packet editing."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import subprocess
import threading
import time

import httpx

from .alfworld_h_train import monitor_gpus
from .common import read_json, write_json
from .launch import Launcher
from .resources import stop_known


def packet_metadata(refresh_root: Path, solver_version: str) -> list[dict]:
    complete = read_json(refresh_root / "probe/COMPLETE.json")
    metadata = []
    template = None
    for result in complete["results"]:
        attempt = result["attempts"][result["selected_attempt"] - 1]
        path = Path(attempt["output"]) / "failure_packets.jsonl"
        if path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != 1:
                raise ValueError(f"expected one failure packet: {path}")
            row = dict(rows[0]["metadata"], solver_version=solver_version)
            template = template or row
        else:
            if template is None:
                metadata.append((result, None))
                continue
            row = None
        metadata.append((result, row))
    if template is None:
        raise ValueError("all-pass probe has no metadata template")
    normalized = []
    for result, row in metadata:
        if row is None:
            start, end = result["start"], result["start"] + result["size"]
            row = {
                **template, "batch_tag": "baseline", "start": start, "end": end,
                "batch_size": result["size"], "baseline_pass": result["size"],
                "baseline_rewards": {str(task_id): 1.0 for task_id in range(start, end)},
                "solver_version": solver_version,
            }
        normalized.append(row)
    normalized.sort(key=lambda row: row["start"])
    return normalized


def candidate_patches(refresh_roots: list[Path]) -> list[dict]:
    candidates = {}
    for root in refresh_roots:
        complete = read_json(root / "probe/COMPLETE.json")
        for result in complete["results"]:
            attempt = result["attempts"][result["selected_attempt"] - 1]
            summary = attempt["summary"]
            if summary.get("status") == "baseline_all_pass_no_failures":
                continue
            if summary["categories"]["valid_evaluated"] != 1:
                continue
            path = Path(attempt["output"]) / "patched/baseline/patch.json"
            source = path.read_text()
            identity = hashlib.sha256(source.encode()).hexdigest()
            candidates.setdefault(identity, {
                "sha256": identity, "path": str(path.resolve()),
                "patch": json.loads(source),
            })
    return [candidates[key] for key in sorted(candidates)]


def lower_bound(deltas: list[float], samples: int, seed: int) -> float:
    rng = random.Random(seed)
    estimates = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(samples)
    )
    return estimates[int(0.05 * samples)]


def accepted_candidate(report: dict, minimum_absolute_gain: float) -> bool:
    if not 0 < minimum_absolute_gain <= 1:
        raise ValueError("minimum_absolute_gain must be in (0, 1]")
    return (
        report["gain"] >= minimum_absolute_gain
        and report["one_sided_95pct_lower_bound"] > 0
    )


async def evaluate_candidates(url: str, candidates: list[dict], packets: list[dict],
                              concurrency: int) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)

    async def score(client: httpx.AsyncClient, candidate: dict, metadata: dict) -> dict:
        source = "</think><patch>" + json.dumps(
            candidate["patch"], separators=(",", ":")) + "</patch>"
        payload = {
            "source": source,
            "sample_id": f"global-{candidate['sha256']}-{metadata['start']}",
            "metadata": metadata,
        }
        async with semaphore:
            response = await client.post(url + "/score", json=payload)
        if not response.is_success:
            raise RuntimeError(
                f"global harness evaluator HTTP {response.status_code}: {response.text[:2000]}")
        return response.json()

    async with httpx.AsyncClient(timeout=4800, trust_env=False) as client:
        pending = [
            (candidate, metadata, asyncio.create_task(score(client, candidate, metadata)))
            for candidate in candidates for metadata in packets
        ]
        results = {(candidate["sha256"], metadata["start"]): await task
                   for candidate, metadata, task in pending}
    reports = []
    for candidate in candidates:
        rows = []
        for metadata in packets:
            evaluation = results[candidate["sha256"], metadata["start"]]["evaluation"]
            baseline = {int(key): float(value)
                        for key, value in metadata["baseline_rewards"].items()}
            patched = {int(key): float(value)
                       for key, value in evaluation["patched_rewards"].items()}
            if set(baseline) != set(patched):
                raise ValueError("global harness evaluation changed task identity")
            rows.extend({
                "task_id": task_id, "baseline_reward": baseline[task_id],
                "patched_reward": patched[task_id],
                "delta_reward": patched[task_id] - baseline[task_id],
            } for task_id in sorted(baseline))
        reports.append({**candidate, "task_rows": rows})
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    run = Path(config["run"]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
    packets = packet_metadata(Path(config["packet_refresh"]), config["solver_version"])
    candidates = (
        [{"sha256": hashlib.sha256(Path(config["patch"]).read_bytes()).hexdigest(),
          "path": str(Path(config["patch"]).resolve()),
          "patch": read_json(Path(config["patch"]))}]
        if config.get("patch") else
        candidate_patches([Path(path) for path in config["candidate_refreshes"]])
    )
    if not candidates:
        raise ValueError("global harness selection has no valid candidates")
    launch = Launcher(run / "supervision")
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=monitor_gpus,
        args=(run / "gpu_samples.jsonl", telemetry_stop,
              float(config.get("monitor_interval_seconds", 5))),
        daemon=True,
    )
    telemetry.start()
    started = time.time()
    try:
        with launch.vllm(
            config["solver_model"], port=config.get("solver_port", 19801),
            gpus="0,1,2,3,4,5,6,7", name=f"{config['phase']}_solver",
            data_parallel=8, max_model_len=32768, tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton", language_model_only=True, max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as urls:
            reward = {
                "cache_root": str(run / "reward_cache"),
                "receipts": str(run / "reward_receipts"),
                "port_locks": str(run / "reward_port_locks"),
                "agentbench_dir": str(run / "runtime/AgentBench"),
                "tmpdir": config["tmpdir"], "target_urls": urls.split(","),
                "target_model": "policy", "target_agent_name": "qwen35-9b-nothink",
                "solver_version": config["solver_version"],
                "rollout_concurrency": config.get("rollout_concurrency", 8),
                "max_concurrency": config.get("reward_concurrency", 8),
                "controller_port_base": config.get("controller_port_base", 30000),
                "worker_port_base": config.get("worker_port_base", 32000),
                "port_probe_span": config.get("port_probe_span", 7000),
            }
            write_json(run / "reward_config.json", reward)
            port = config.get("reward_service_port", 28600)
            process = launch.module(
                "sao_bridge.alfworld_reward_service",
                ["--config", run / "reward_config.json", "--port", port],
                gpus="", name=f"{config['phase']}_reward",
            )
            launch.ready(process, f"http://127.0.0.1:{port}/health")
            reports = asyncio.run(evaluate_candidates(
                f"http://127.0.0.1:{port}", candidates, packets,
                config.get("candidate_concurrency", 8),
            ))
            samples = int(config.get("bootstrap_samples", 10000))
            seed = int(config.get("bootstrap_seed", 20260930))
            for report in reports:
                deltas = [row["delta_reward"] for row in report["task_rows"]]
                report.update(
                    tasks=len(deltas), baseline_pass=sum(
                        row["baseline_reward"] for row in report["task_rows"]),
                    patched_pass=sum(
                        row["patched_reward"] for row in report["task_rows"]),
                    gain=sum(deltas) / len(deltas),
                    one_sided_95pct_lower_bound=lower_bound(deltas, samples, seed),
                )
            reports.sort(key=lambda row: (-row["patched_pass"], row["sha256"]))
            best = reports[0]
            minimum_absolute_gain = float(config["minimum_absolute_gain"])
            accepted = accepted_candidate(best, minimum_absolute_gain)
            selected = best if accepted else {
                "sha256": hashlib.sha256(Path(config["no_patch"]).read_bytes()).hexdigest(),
                "path": str(Path(config["no_patch"]).resolve()),
                "patch": read_json(Path(config["no_patch"])),
                "gain": 0.0, "one_sided_95pct_lower_bound": 0.0,
            }
            write_json(run / "candidates.json", reports)
            write_json(run / "selected_patch.json", selected["patch"])
            write_json(run / "COMPLETE.json", {
                "status": "completed", "phase": config["phase"],
                "elapsed_seconds": time.time() - started,
                "tasks": sum(packet["batch_size"] for packet in packets),
                "candidate_count": len(reports), "accepted": accepted,
                "selected_sha256": selected["sha256"], "selected_gain": selected["gain"],
                "selected_lcb": selected["one_sided_95pct_lower_bound"],
                "minimum_absolute_gain": minimum_absolute_gain,
                "selection_split": config["selection_split"],
            })
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "elapsed_seconds": time.time() - started, "error": repr(exc),
        })
        raise
    finally:
        launch.close()
        telemetry_stop.set()
        telemetry.join(timeout=10)
        write_json(run / "RELEASED.json", {
            "released_at": time.time(),
            "remaining_gpu_pids": subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader,nounits"], text=True,
            ).split(),
        })


if __name__ == "__main__":
    main()
