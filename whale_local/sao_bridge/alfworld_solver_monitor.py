"""Read-only native-update audit and monitoring of predeclared 9B Solver runs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

from .alfworld_solver_pair_gate import evaluate_pair
from .common import file_sha, read_json, write_json


def native_metrics(path: Path, spec: dict) -> dict:
    lines = path.read_bytes().splitlines(keepends=True) if path.exists() else []
    rows = [json.loads(line) for line in lines if line.endswith(b"\n")]
    warmup = spec["critic_warmup_steps"]
    if len(rows) > spec["total_steps"]:
        raise ValueError("Optimizer exceeded the frozen step budget")
    targets = []
    actor_updates = 0
    nonzero_actor_updates = 0
    for step, row in enumerate(rows):
        if row["global_step"] != step:
            raise ValueError("Optimizer global steps are not contiguous")
        metrics = row["metrics"]
        low = metrics["ppo_critic/update/target_value/min"]
        high = metrics["ppo_critic/update/target_value/max"]
        if not 0 <= low <= high <= 1:
            raise ValueError("Complete Solver episode targets escaped [0, 1]")
        targets.extend((low, high))
        gradient = metrics.get("ppo_actor/update/grad_norm")
        if step < warmup:
            if gradient is not None:
                raise ValueError("Actor updated during critic-only warm-up")
        else:
            if gradient is None or not math.isfinite(gradient) or gradient < 0:
                raise ValueError("Missing or invalid Actor gradient after warm-up")
            if metrics["ppo_actor/n_seqs"] != spec["batch_size"]:
                raise ValueError("Concat trajectory count differs from the frozen batch")
            actor_updates += 1
            nonzero_actor_updates += gradient > 0
    return {
        "optimizer_steps": len(rows),
        "actor_updates": actor_updates,
        "nonzero_actor_updates": nonzero_actor_updates,
        "critic_target_range": [min(targets), max(targets)] if targets else None,
        "last_metric_at": rows[-1]["recorded_at_unix"] if rows else None,
        "metrics_write_in_progress": bool(lines and not lines[-1].endswith(b"\n")),
    }


def snapshot(root: Path, now: float) -> dict:
    spec = read_json(root / "spec.json")
    if (
        not spec["terminal_episodes"]
        or spec["critic_mode"] != "learned"
        or spec["solver_trajectory_style"] != "concat"
    ):
        raise ValueError("This monitor requires complete concat episodes and a learned critic")
    manifest = read_json(root / "source_manifest.json")
    if not manifest:
        raise ValueError("Empty frozen source manifest")
    for name, digest in manifest.items():
        if Path(name).name != name or file_sha(Path(__file__).with_name(name)) != digest:
            raise ValueError(f"Frozen Solver source changed: {name}")
    for path_key, hash_key in (
        ("training_data", "training_data_sha256"),
        ("training_catalogue", "training_catalogue_sha256"),
    ):
        if hash_key in spec and file_sha(Path(spec[path_key])) != spec[hash_key]:
            raise ValueError(f"Frozen Solver input changed: {path_key}")
    progress = native_metrics(root / "training/optimizer/metrics.jsonl", spec)
    status = read_json(root / "STATUS.json")
    failure_path = root / "FAILED.json"
    failure = read_json(failure_path) if failure_path.exists() else None
    completion_path = root / "COMPLETE.json"
    release_path = root / "gpu_release_audit.json"
    finished = completion_path.is_file()
    released = (
        finished and release_path.is_file()
        and release_path.stat().st_mtime_ns >= completion_path.stat().st_mtime_ns
    )
    if finished:
        if (
            progress["optimizer_steps"] != spec["total_steps"]
            or progress["actor_updates"] != spec["actor_steps"]
            or not progress["nonzero_actor_updates"]
            or progress["metrics_write_in_progress"]
        ):
            raise ValueError("Completion receipt does not match native Actor updates")
        trained = read_json(root / "training/optimizer/trained.json")
        if (
            trained["actor_steps"] != spec["actor_steps"]
            or trained["critic_only_warmup_steps"] != spec["critic_warmup_steps"]
            or trained["critic_steps"]
            != spec["total_steps"] * spec["critic_updates_per_step"]
        ):
            raise ValueError("Completed optimizer budget differs from the frozen protocol")
        expected = list(range(
            spec["save_interval_steps"] - 1,
            spec["total_steps"],
            spec["save_interval_steps"],
        ))
        for role in ("actor", "critic"):
            history = trained["checkpoint_history"][role]
            if [row["global_step"] for row in history] != expected:
                raise ValueError(f"Incomplete {role} checkpoint history")
            for checkpoint in history:
                if not list(Path(checkpoint["path"]).glob("*.safetensors")):
                    raise ValueError(f"Missing {role} checkpoint weights")
        if failure:
            raise ValueError("Run has both completion and failure receipts")
        if released:
            release = read_json(release_path)
            if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
                raise ValueError("Completed run did not release GPUs to the active filler")
    events = [root / "STATUS.json", root / "training/optimizer/metrics.jsonl"]
    last_progress = max(path.stat().st_mtime for path in events if path.exists())
    attempts = root / "training_attempts.json"
    return {
        "run": str(root), "seed": spec["seed"], "stage": status["stage"],
        "checkpoint_step": status.get("global_step"),
        "completed_and_released": finished and released,
        "failure": failure, "source_manifest_matches": True,
        "progress_age_seconds": max(0.0, now - last_progress),
        "training_attempts": len(read_json(attempts)) if attempts.exists() else 0,
        **progress,
    }


def watch(
    roots: list[Path], protocol_path: Path, output: Path, *,
    interval: float = 60, timeout: float = 86400, failure_grace: float = 600,
) -> dict:
    if interval <= 0 or timeout <= 0 or failure_grace < 0:
        raise ValueError("Invalid monitor polling, timeout or failure grace")
    protocol = read_json(protocol_path)
    if len(roots) != len(protocol["required_seeds"]) or len(set(roots)) != len(roots):
        raise ValueError("Monitor requires exactly the distinct predeclared runs")
    protocol_digest = file_sha(protocol_path)
    status_path = output.with_name(output.stem + "_monitor.json")
    identity_path = output.with_name(output.stem + "_monitor_protocol.json")
    failure_since = {}
    started = time.monotonic()
    previous = None
    try:
        identity = {
            "gate_protocol_sha256": protocol_digest,
            "runs": [
                {
                    "run": str(root),
                    "spec_sha256": file_sha(root / "spec.json"),
                    "source_manifest_sha256": file_sha(root / "source_manifest.json"),
                }
                for root in roots
            ],
        }
        if identity_path.exists() and read_json(identity_path) != identity:
            raise ValueError("Cannot resume monitoring after protocol or run identity changes")
        write_json(identity_path, identity)
        while time.monotonic() - started < timeout:
            now = time.time()
            if file_sha(protocol_path) != protocol_digest:
                raise ValueError("Pair gate changed while monitoring")
            for root, frozen in zip(roots, identity["runs"]):
                if (
                    file_sha(root / "spec.json") != frozen["spec_sha256"]
                    or file_sha(root / "source_manifest.json") != frozen["source_manifest_sha256"]
                ):
                    raise ValueError(f"Frozen run identity changed: {root}")
            reports = [snapshot(root, now) for root in roots]
            if sorted(row["seed"] for row in reports) != sorted(protocol["required_seeds"]):
                raise ValueError("Run seeds differ from the predeclared pair")
            report = {
                "updated_at_unix": now, "health": "running", "runs": reports,
                "gate_protocol_sha256": protocol_digest,
                "read_only": True, "controller_restarts_managed_by": "systemd",
            }
            for state in reports:
                key = state["run"]
                if state["failure"]:
                    failure_since.setdefault(key, now)
                    report["health"] = "waiting_for_controller_retry"
                    if now - failure_since[key] >= failure_grace:
                        raise RuntimeError(f"Controller failure did not recover: {key}")
                else:
                    failure_since.pop(key, None)
            write_json(status_path, report)
            marker = [
                (row["stage"], row["checkpoint_step"], row["optimizer_steps"],
                 row["training_attempts"], row["failure"])
                for row in reports
            ]
            if marker != previous:
                print(json.dumps(report), flush=True)
                previous = marker
            if all(row["completed_and_released"] for row in reports):
                result = evaluate_pair(roots, protocol)
                if len(roots) == 1:
                    result["interpretation"] = (
                        "This is a single-training-seed exploratory development gate, "
                        "not a replicated or confirmatory result. Question-level "
                        "bootstrap does not measure training-seed uncertainty or "
                        "correct adaptive searches across methods."
                    )
                if output.exists() and read_json(output) != result:
                    raise ValueError("Refusing to replace a different persisted pair result")
                write_json(output, result)
                write_json(status_path, {
                    **report, "health": "completed", "gate_passed": result["gate_passed"],
                })
                print(json.dumps(result), flush=True)
                return result
            time.sleep(interval)
        raise TimeoutError("Solver pair exceeded the monitor deadline")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
        write_json(status_path, {
            "updated_at_unix": time.time(), "health": "attention_required",
            "error": f"{type(exc).__name__}: {exc}",
            "read_only": True, "controller_restarts_managed_by": "systemd",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--timeout", type=float, default=86400)
    args = parser.parse_args()
    watch(
        [path.resolve() for path in args.run], args.protocol, args.output,
        interval=args.interval, timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
