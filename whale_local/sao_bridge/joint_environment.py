"""Serve each joint round's sampled harnesses without changing official workers."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .common import read_json, write_json
from .joint_round import check_failures, record_failure, round_path, wait_record
from .launch import Launcher


def run(spec_path: Path, launch: Launcher) -> None:
    spec = read_json(spec_path)
    root = Path(spec["joint_root"])
    rows = read_json(Path(spec["data"]))
    try:
        for version in range(spec["steps"]):
            current = round_path(root, version)
            batch = rows[version * spec["batch_size"]:(version + 1) * spec["batch_size"]]
            assignments = []
            for row in batch:
                slot = row["joint_slot"]
                proposal = asyncio.run(wait_record(current / f"proposal-{slot:03d}.json", root))
                path = current / f"patch-{slot:03d}.json"
                write_json(path, proposal["patch"])
                assignments.append({
                    "name": f"joint-{version}-{slot}", "task_ids": [row["task_id"]],
                    "harness": str(path),
                })
            write_json(current / "assignments.json", assignments)
            process = launch.module(
                "sao_bridge.alfworld_env_service",
                ["--run", current / "environment", "--assignments", current / "assignments.json",
                 "--controller-port", "28000", "--worker-port", "28100",
                 "--concurrency", "1", "--async-controller"],
                gpus="", name=f"joint-env-{version:04d}",
            )
            deadline = time.monotonic() + 600
            while not (current / "environment/READY.json").exists():
                check_failures(root)
                launch.check()
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("Joint environment failed to become ready")
                time.sleep(1)
            write_json(current / "ENVIRONMENT_READY.json", {
                "version": version, "assignments": assignments,
                "controller": "http://127.0.0.1:28000/api",
            })
            for row in batch:
                asyncio.run(wait_record(current / f"outcome-{row['joint_slot']:03d}.json", root))
            launch.stop(process)
            write_json(current / "ENVIRONMENT_RELEASED.json", {"version": version})
    except BaseException as exc:
        record_failure(root, "solver", {"error": repr(exc), "where": "joint_environment"})
        raise
