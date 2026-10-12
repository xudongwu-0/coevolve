"""Read-only, persistent phase/optimizer monitoring for a local campaign worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import read_json, write_json


def snapshot(root: Path, service: dict, now: float) -> dict:
    worker = read_json(root / "WORKER_STATUS.json")
    run = Path(worker["run"])
    state_path = run / "STATUS.json"
    state = read_json(state_path) if state_path.exists() else {}
    events = [root / "WORKER_STATUS.json"]
    if state_path.exists():
        events.append(state_path)
    phases = {}
    for path in run.glob("[hw][0-9]*/metrics.jsonl"):
        content = path.read_text()
        lines = content.splitlines(keepends=True)
        rows = [json.loads(line) for line in lines if line.endswith("\n")]
        phase_spec = read_json(run / f"{path.parent.name}_spec.json")
        warmup = phase_spec.get("critic_warmup_steps", 0)
        phases[path.parent.name] = {
            "native_updates": len(rows),
            "actor_updates": max(0, len(rows) - warmup),
            "actor_target": phase_spec["steps"] - warmup,
            "critic_warmup_updates": min(len(rows), warmup),
            "checkpoint_saved": (path.parent / "trained.json").is_file(),
            "metrics_write_in_progress": bool(lines and not lines[-1].endswith("\n")),
        }
        events.append(path)
    events.extend(run.glob("h*_report.json"))
    events.extend(run.glob("[hw][0-9]*/trained.json"))
    last_progress = max(path.stat().st_mtime for path in events)
    phase = worker["phase"]
    failed = (phase in {"failed", "peer_failed"} or state.get("status") == "failed"
              or service["ActiveState"] == "failed")
    if phase == "completed" and not failed:
        health = "completed"
    elif failed:
        health = "failed"
    elif service["ActiveState"] not in {"active", "activating"}:
        health = "worker_not_running"
    elif phase != "waiting_for_peer_results" and now - last_progress > 1800:
        health = "stalled"
    else:
        health = "running"
    return {
        "updated_at": now, "host": os.uname().nodename, "health": health,
        "worker_phase": phase, "experiment_phase": state.get("status"),
        "service": service, "optimizer_progress": phases,
        "stages": state.get("stages", []),
        "last_progress_at": last_progress, "progress_age_seconds": max(0, now - last_progress),
        "error": state.get("error") or worker.get("error"),
        "failure_log": str(root / "supervision/hwh.log"),
        "read_only": True, "automatic_restart": False,
    }


def pair_snapshot(protocol: dict, now: float) -> dict:
    nodes = {}
    for host, node in protocol["nodes"].items():
        report = read_json(Path(node["root"]) / "MONITOR_STATUS.json")
        if report.get("host") != host:
            raise ValueError(f"Monitor host identity mismatch: {host}")
        health = report["health"]
        if health != "completed" and now - report["updated_at"] > 120:
            health = "monitor_stale"
        nodes[host] = {**report, "health": health}
    failed = any(row["health"] not in {"running", "completed"} for row in nodes.values())
    return {"updated_at": now, "nodes": nodes,
            "health": "attention_required" if failed else
                      "completed" if all(r["health"] == "completed" for r in nodes.values())
                      else "running"}


def watch_pair(path: Path):
    protocol = read_json(path)
    own = Path(protocol["nodes"][os.uname().nodename]["root"])
    previous = None
    while True:
        try:
            report = pair_snapshot(protocol, time.time())
        except (OSError, ValueError, KeyError) as exc:
            report = {"updated_at": time.time(), "health": "attention_required",
                      "error": f"{type(exc).__name__}: {exc}"}
        write_json(own / "PAIR_MONITOR.json", report)
        marker = json.dumps({
            host: (row["health"], row.get("experiment_phase"), row.get("optimizer_progress"))
            for host, row in report.get("nodes", {}).items()
        }, sort_keys=True)
        if marker != previous or report["health"] != "running":
            print(json.dumps(report), flush=True)
            previous = marker
        if report["health"] != "running":
            return 0 if report["health"] == "completed" else 2
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--root", type=Path)
    target.add_argument("--pair-protocol", type=Path)
    parser.add_argument("--unit")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.pair_protocol:
        sys.exit(watch_pair(args.pair_protocol))
    if not args.unit:
        parser.error("--unit is required with --root")
    previous = None
    while True:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "show", args.unit,
                 "--property=ActiveState,SubState,MainPID,Result,ExecMainStatus"],
                check=True, capture_output=True, text=True, timeout=15,
            )
            service = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
            report = snapshot(args.root, service, time.time())
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
            report = {"updated_at": time.time(), "health": "monitor_error",
                      "host": os.uname().nodename,
                      "error": f"{type(exc).__name__}: {exc}", "unit": args.unit}
            print(json.dumps(report), flush=True)
        write_json(args.root / "MONITOR_STATUS.json", report)
        (args.root / "MONITOR_STATUS.json").chmod(0o644)
        marker = (report["health"], report.get("experiment_phase"), report.get("error"))
        if marker != previous:
            print(json.dumps(report), flush=True)
            with (args.root / "monitor_events.jsonl").open("a") as stream:
                stream.write(json.dumps(report) + "\n")
            previous = marker
        if args.once or report["health"] == "completed":
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
