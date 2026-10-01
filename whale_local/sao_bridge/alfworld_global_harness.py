"""Select or evaluate one global ALFWorld harness without target-packet editing."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import time

import httpx

from .alfworld_h_train import monitor_gpus
from .common import read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


def patch_identity(patch: dict) -> str:
    canonical = json.dumps(patch, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def probe_receipt(refresh_root: Path) -> dict:
    combined = refresh_root / "combined_probe/COMPLETE.json"
    path = combined if combined.is_file() else refresh_root / "probe/COMPLETE.json"
    return read_json(path)


def packet_metadata(refresh_root: Path, solver_version: str) -> list[dict]:
    complete = probe_receipt(refresh_root)
    metadata = []
    template = None
    for result in complete["results"]:
        attempt = result["attempts"][result["selected_attempt"] - 1]
        path = Path(attempt["output"]) / "failure_packets.jsonl"
        summary = attempt["summary"]
        if summary.get("status") == "baseline_all_pass_no_failures":
            row = None
        elif path.is_file():
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            if len(rows) != 1:
                raise ValueError(f"expected one failure packet: {path}")
            row = dict(rows[0]["metadata"], solver_version=solver_version)
            template = template or row
        else:
            raise FileNotFoundError(f"failure packet is missing: {path}")
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
        complete = probe_receipt(root)
        for result in complete["results"]:
            attempt = result["attempts"][result["selected_attempt"] - 1]
            summary = attempt["summary"]
            if summary.get("status") == "baseline_all_pass_no_failures":
                continue
            if summary["categories"]["valid_evaluated"] != 1:
                continue
            path = Path(attempt["output"]) / "patched/baseline/patch.json"
            source = path.read_text()
            patch = json.loads(source)
            identity = patch_identity(patch)
            candidates.setdefault(identity, {
                "sha256": identity, "path": str(path.resolve()),
                "patch": patch,
            })
    return [candidates[key] for key in sorted(candidates)]


def receipt_candidates(receipt_roots: list[Path],
                       maximum_behavior_version: int,
                       allowed_ids: set[str] | None = None) -> list[dict]:
    if maximum_behavior_version < 0:
        raise ValueError("maximum_behavior_version must be nonnegative")
    candidates = {}
    found_ids = set()
    for root in receipt_roots:
        for path in sorted(root.glob("*.json")):
            receipt = read_json(path)
            receipt_id = receipt.get("id")
            if allowed_ids is not None and receipt_id not in allowed_ids:
                continue
            if allowed_ids is not None:
                if receipt_id in found_ids:
                    raise ValueError(f"duplicate consumed receipt ID: {receipt_id}")
                found_ids.add(receipt_id)
            if receipt.get("replayed"):
                if allowed_ids is not None:
                    raise ValueError(f"consumed fresh receipt is marked replayed: {path}")
                continue
            if not (receipt.get("evaluation") or {}).get("valid"):
                continue
            versions = [
                int(version)
                for call in receipt.get("calls", [])
                for version in call.get("versions", [])
            ]
            if not versions or max(versions) > maximum_behavior_version:
                if allowed_ids is not None:
                    raise ValueError(
                        f"consumed receipt exceeds the behavior-version gate: {path}"
                    )
                continue
            source = receipt.get("action_text")
            if not isinstance(source, str) or source.count("<patch>") != 1:
                raise ValueError(f"valid receipt has no unique patch block: {path}")
            if source.count("</patch>") != 1:
                raise ValueError(f"valid receipt has no unique patch close tag: {path}")
            raw = source.split("<patch>", 1)[1].split("</patch>", 1)[0]
            patch = json.loads(raw)
            if (not isinstance(patch, dict)
                    or patch.get("benchmark") != "alfworld"
                    or not isinstance(patch.get("actions"), list)):
                raise ValueError(f"valid receipt has an invalid ALFWorld patch: {path}")
            identity = patch_identity(patch)
            candidates.setdefault(identity, {
                "sha256": identity,
                "path": str(path.resolve()),
                "patch": patch,
                "candidate_source": "optimizer_receipt",
                "maximum_behavior_version": max(versions),
            })
    if allowed_ids is not None and found_ids != allowed_ids:
        missing = sorted(allowed_ids - found_ids)
        raise ValueError(f"consumed receipt IDs are missing: {missing}")
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


def no_patch_report(no_patch: Path, packets: list[dict]) -> dict:
    patch = read_json(no_patch)
    rows = []
    for metadata in packets:
        baseline = {
            int(task_id): float(reward)
            for task_id, reward in metadata["baseline_rewards"].items()
        }
        rows.extend({
            "task_id": task_id,
            "baseline_reward": baseline[task_id],
            "patched_reward": baseline[task_id],
            "delta_reward": 0.0,
        } for task_id in sorted(baseline))
    baseline_pass = sum(row["baseline_reward"] for row in rows)
    return {
        "sha256": patch_identity(patch),
        "path": str(no_patch.resolve()),
        "patch": patch,
        "candidate_source": "no_patch_incumbent",
        "task_rows": rows,
        "tasks": len(rows),
        "baseline_pass": baseline_pass,
        "patched_pass": baseline_pass,
        "gain": 0.0,
        "one_sided_95pct_lower_bound": 0.0,
        "is_incumbent": True,
    }


def paired_patch_improvement(candidate: dict, incumbent: dict,
                             bootstrap_samples: int, bootstrap_seed: int) -> dict:
    candidate_rows = {
        row["task_id"]: row["patched_reward"] for row in candidate["task_rows"]
    }
    incumbent_rows = {
        row["task_id"]: row["patched_reward"] for row in incumbent["task_rows"]
    }
    if set(candidate_rows) != set(incumbent_rows):
        raise ValueError("candidate and incumbent changed development task identity")
    deltas = [
        candidate_rows[task_id] - incumbent_rows[task_id]
        for task_id in sorted(incumbent_rows)
    ]
    return {
        "gain": sum(deltas) / len(deltas),
        "one_sided_95pct_lower_bound": lower_bound(
            deltas, bootstrap_samples, bootstrap_seed,
        ),
    }


def complete_without_candidates(run: Path, config: dict, packets: list[dict],
                                minimum_absolute_gain: float) -> None:
    incumbent = Path(config.get("incumbent_patch", config["no_patch"]))
    patch = read_json(incumbent)
    selected_sha256 = patch_identity(patch)
    write_json(run / "candidates.json", [])
    write_json(run / "selected_patch.json", patch)
    write_json(run / "COMPLETE.json", {
        "status": "completed", "phase": config["phase"],
        "elapsed_seconds": 0.0,
        "tasks": sum(packet["batch_size"] for packet in packets),
        "candidate_count": 0, "accepted": False,
        "selected_sha256": selected_sha256, "selected_gain": 0.0,
        "selected_lcb": 0.0,
        "selected_absolute_gain_vs_no_patch": None,
        "incumbent_sha256": selected_sha256,
        "minimum_absolute_gain": minimum_absolute_gain,
        "selection_split": config["selection_split"],
        "reason": "no_valid_harness_candidates",
    })
    write_json(run / "RELEASED.json", {
        "released_at": time.time(),
        **gpu_release_audit(),
        "gpu_resources_claimed": False,
    })


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
            patched = normalize_patched_rewards(
                evaluation["patched_rewards"], metadata,
            )
            if set(baseline) != set(patched):
                raise ValueError("global harness evaluation changed task identity")
            rows.extend({
                "task_id": task_id, "baseline_reward": baseline[task_id],
                "patched_reward": patched[task_id],
                "delta_reward": patched[task_id] - baseline[task_id],
            } for task_id in sorted(baseline))
        reports.append({**candidate, "task_rows": rows})
    return reports


def normalize_patched_rewards(values: dict, metadata: dict) -> dict[int, float]:
    patched = {int(key): float(value) for key, value in values.items()}
    start, end = metadata["start"], metadata["end"]
    expected = set(range(start, end))
    if set(patched) == expected:
        return patched
    local = set(range(metadata["batch_size"]))
    if set(patched) == local and end - start == metadata["batch_size"]:
        return {start + task_id: reward for task_id, reward in patched.items()}
    raise ValueError(
        "official patched rewards are neither global task IDs nor an exact "
        "packet-local 0..batch_size-1 identity"
    )


def seed_reward_cache(run: Path, config: dict) -> None:
    source_value = config.get("seed_reward_cache")
    if not source_value:
        return
    source = Path(source_value).resolve()
    previous_config_path = source.parent / "config.json"
    if not source.is_dir() or not previous_config_path.is_file():
        raise ValueError("seed_reward_cache must be a prior global run reward_cache")
    previous = read_json(previous_config_path)
    identity_keys = (
        "packet_refresh", "candidate_refreshes", "candidate_receipt_roots",
        "candidate_id_ledger", "maximum_behavior_version", "patch",
        "incumbent_patch", "no_patch",
        "solver_model", "solver_version",
    )
    mismatched = [
        key for key in identity_keys
        if previous.get(key) != config.get(key)
    ]
    if mismatched:
        raise ValueError(
            f"seed reward cache evaluation identity changed: {mismatched}"
        )
    destination = run / "reward_cache"
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns(".lock"))
    from .alfworld_reward_service import infrastructure_failure
    discarded = []
    for reward_path in sorted(destination.rglob("reward.json")):
        if infrastructure_failure(read_json(reward_path)):
            discarded.append(str(reward_path.parent.relative_to(destination)))
            shutil.rmtree(reward_path.parent)
    files = [path for path in destination.rglob("*") if path.is_file()]
    write_json(run / "reward_cache_seed.json", {
        "source": str(source),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "discarded_infrastructure_entries": discarded,
        "identity_keys": list(identity_keys),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    run = Path(config["run"]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    started = time.time()
    try:
        packets = packet_metadata(
            Path(config["packet_refresh"]), config["solver_version"])
        no_patch = Path(config["no_patch"])
        no_patch_incumbent = no_patch_report(no_patch, packets)
        incumbent_path = Path(config.get("incumbent_patch", no_patch))
        incumbent_patch = read_json(incumbent_path)
        incumbent_sha256 = patch_identity(incumbent_patch)
        if config.get("patch"):
            patch = read_json(Path(config["patch"]))
            candidates = [{
                "sha256": patch_identity(patch),
                "path": str(Path(config["patch"]).resolve()),
                "patch": patch,
                "candidate_source": "declared_patch",
            }]
        else:
            candidate_sets = []
            if config.get("candidate_refreshes"):
                candidate_sets.extend(candidate_patches(
                    [Path(path) for path in config["candidate_refreshes"]]
                ))
            if config.get("candidate_receipt_roots"):
                allowed_ids = None
                if config.get("candidate_id_ledger"):
                    ledger = read_json(Path(config["candidate_id_ledger"]))
                    ids = ledger.get("fresh_ids")
                    if (not isinstance(ids, list) or not ids
                            or any(not isinstance(value, str) or not value for value in ids)
                            or len(ids) != len(set(ids))):
                        raise ValueError("candidate ID ledger has invalid fresh IDs")
                    allowed_ids = set(ids)
                candidate_sets.extend(receipt_candidates(
                    [Path(path) for path in config["candidate_receipt_roots"]],
                    int(config["maximum_behavior_version"]),
                    allowed_ids,
                ))
            candidates = {
                candidate["sha256"]: candidate for candidate in candidate_sets
            }
            candidates = [candidates[key] for key in sorted(candidates)]
        minimum_absolute_gain = float(config["minimum_absolute_gain"])
        if not 0 < minimum_absolute_gain <= 1:
            raise ValueError("minimum_absolute_gain must be in (0, 1]")
        seed_reward_cache(run, config)
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "stage": "pre_gpu_selection_audit",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
        })
        raise
    if not candidates:
        complete_without_candidates(run, config, packets, minimum_absolute_gain)
        return
    evaluation_candidates = list(candidates)
    if (incumbent_sha256 != no_patch_incumbent["sha256"]
            and incumbent_sha256 not in {
                candidate["sha256"] for candidate in evaluation_candidates
            }):
        evaluation_candidates.append({
            "sha256": incumbent_sha256,
            "path": str(incumbent_path.resolve()),
            "patch": incumbent_patch,
            "candidate_source": "declared_incumbent",
            "is_incumbent": True,
        })
    try:
        write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "stage": "resource_audit",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
        })
        raise
    launch = Launcher(run / "supervision")
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=monitor_gpus,
        args=(run / "gpu_samples.jsonl", telemetry_stop,
              float(config.get("monitor_interval_seconds", 5))),
        daemon=True,
    )
    telemetry.start()
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
                f"http://127.0.0.1:{port}", evaluation_candidates, packets,
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
            if incumbent_sha256 == no_patch_incumbent["sha256"]:
                incumbent = no_patch_incumbent
            else:
                incumbent = next(
                    report for report in reports
                    if report["sha256"] == incumbent_sha256
                )
                incumbent["is_incumbent"] = True
            fresh_reports = [
                report for report in reports
                if report["sha256"] != incumbent_sha256
            ]
            ranked = sorted(
                [*fresh_reports, incumbent],
                key=lambda row: (-row["patched_pass"], row["sha256"]),
            )
            best = ranked[0]
            improvement = paired_patch_improvement(best, incumbent, samples, seed)
            accepted = (
                best["sha256"] != incumbent_sha256
                and accepted_candidate(improvement, minimum_absolute_gain)
            )
            selected = best if accepted else incumbent
            selected_improvement = improvement if accepted else {
                "gain": 0.0,
                "one_sided_95pct_lower_bound": 0.0,
            }
            reports.sort(key=lambda row: (-row["patched_pass"], row["sha256"]))
            write_json(run / "candidates.json", reports)
            write_json(run / "selected_patch.json", selected["patch"])
            write_json(run / "COMPLETE.json", {
                "status": "completed", "phase": config["phase"],
                "elapsed_seconds": time.time() - started,
                "tasks": sum(packet["batch_size"] for packet in packets),
                "candidate_count": len(fresh_reports), "accepted": accepted,
                "selected_sha256": selected["sha256"],
                "selected_gain": selected_improvement["gain"],
                "selected_lcb": selected_improvement[
                    "one_sided_95pct_lower_bound"],
                "selected_absolute_gain_vs_no_patch": selected["gain"],
                "selected_absolute_lcb_vs_no_patch": selected[
                    "one_sided_95pct_lower_bound"],
                "incumbent_sha256": incumbent_sha256,
                "incumbent_absolute_gain_vs_no_patch": incumbent["gain"],
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
            **gpu_release_audit(),
        })


if __name__ == "__main__":
    main()
