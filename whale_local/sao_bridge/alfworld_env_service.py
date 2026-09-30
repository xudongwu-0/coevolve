"""Supervise an external-policy AgentBench ALFWorld controller and worker."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import httpx

from .alfworld_runtime import (
    ALFWORLD_ASSETS, ALFWORLD_DATA, prepare_agentbench_runtime,
)
from .common import ROOT, read_json, write_json

UPSTREAM = ROOT / "third_party/harness-r1-upstream"
AGENTBENCH = ALFWORLD_ASSETS / "AgentBench"
PYTHON = Path("/dev/shm/harness-r1-alfworld-venv/bin/python")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "take_action",
        "description": "Take an action.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "The action you would like to take"},
            },
            "required": ["action"],
            "additionalProperties": False,
        },
    },
}]


def task_definition(name: str, task_ids: list[int], concurrency: int, harness: dict,
                    agentbench: Path = AGENTBENCH) -> dict:
    if (not task_ids or len(set(task_ids)) != len(task_ids)
            or any(type(value) is not int or not 0 <= value < 500 for value in task_ids)):
        raise ValueError("task_ids must be distinct official ALFWorld indices")
    task = {name: {
        "module": "src.server.tasks.alfworld.ALFWorld",
        "parameters": {
            "name": name,
            "concurrency": concurrency,
            "data_path": str(agentbench / "data/alfworld"),
            "config_path": str(
                agentbench / "src/server/tasks/alfworld/configs/base_config.yaml"
            ),
            "prompts_path": str(
                agentbench
                / "src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json"
            ),
            "split": "test_set",
            "max_step": 50,
            "task_ids": task_ids,
            "start": min(task_ids),
            "end": max(task_ids) + 1,
            "sample_size": len(task_ids),
            "enabled": False,
            "h2": False, "h3": False, "h4": False, "h5": False, "h5_top_k": 1,
            "tools": TOOLS,
        },
    }}
    if harness["actions"]:
        scripts = UPSTREAM / "code/life-harness/AgentBench/scripts"
        sys.path.insert(0, str(scripts))
        from harness_r1_patch import compile_patch_to_task_definition
        task = compile_patch_to_task_definition(task, harness, "alfworld")
    return task


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def wait_ready(controller: subprocess.Popen, workers: list[subprocess.Popen], url: str,
               task_names: list[str], timeout: int) -> dict:
    started = time.monotonic()
    with httpx.Client(timeout=5, trust_env=False) as client:
        while time.monotonic() - started < timeout:
            processes = [("controller", controller), *(
                (f"worker-{index}", process) for index, process in enumerate(workers)
            )]
            for name, process in processes:
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited before ALFWorld readiness")
            try:
                response = client.get(url + "/list_workers")
                if (response.status_code == 200
                        and all(task_name in response.text for task_name in task_names)):
                    indices = {}
                    for task_name in task_names:
                        result = client.get(url + f"/get_indices?name={task_name}")
                        result.raise_for_status()
                        indices[task_name] = result.json()
                    return {"workers": response.json(), "indices": indices}
            except httpx.HTTPError:
                pass
            time.sleep(1)
    raise TimeoutError("ALFWorld controller/worker readiness timed out")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--task-name")
    parser.add_argument("--task-ids", type=Path)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--assignments", type=Path,
                        help="JSON list of {name, task_ids, harness} records")
    parser.add_argument("--controller-port", type=int, required=True)
    parser.add_argument("--worker-port", type=int, required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--startup-timeout", type=int, default=300)
    args = parser.parse_args()
    if not PYTHON.is_file() or PYTHON.is_symlink():
        raise RuntimeError("ALFWorld requires the copied isolated worker interpreter")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    agentbench = prepare_agentbench_runtime(run / "runtime/AgentBench")
    scratch = run / "scratch"
    scratch.mkdir()
    if args.assignments:
        if args.task_name or args.task_ids or args.harness:
            parser.error("--assignments cannot be combined with the single-task arguments")
        assignments = read_json(args.assignments)
    else:
        if not args.task_name or not args.task_ids or not args.harness:
            parser.error("single-task mode requires --task-name, --task-ids and --harness")
        assignments = [{
            "name": args.task_name, "task_ids": read_json(args.task_ids),
            "harness": str(args.harness),
        }]
    config = {}
    all_ids = []
    for assignment in assignments:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", assignment["name"]):
            raise ValueError("ALFWorld task assignment name contains unsafe characters")
        task_ids = assignment["task_ids"]
        harness = json.loads(Path(assignment["harness"]).read_text())
        definition = task_definition(
            assignment["name"], task_ids, args.concurrency, harness, agentbench,
        )
        if set(config) & set(definition):
            raise ValueError("duplicate ALFWorld task assignment name")
        config.update(definition)
        all_ids.extend(task_ids)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("ALFWorld task assignments must partition task IDs")
    task_names = list(config)
    config_path = run / "tasks.json"
    write_json(config_path, config)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        ALFWORLD_DATA=str(ALFWORLD_DATA),
        TMPDIR=str(scratch),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=":".join((str(agentbench / "src"), str(agentbench))),
    )
    controller_url = f"http://127.0.0.1:{args.controller_port}/api"
    with ExitStack() as logs:
        controller_log = logs.enter_context((run / "controller.log").open("wb"))
        controller = subprocess.Popen([
            str(PYTHON), str(agentbench / "scripts/local_agentrl_controller.py"),
            "--host", "127.0.0.1", "--port", str(args.controller_port),
        ], cwd=agentbench, env=env, stdout=controller_log, stderr=subprocess.STDOUT)
        workers = []
        for index, task_name in enumerate(task_names):
            port = args.worker_port + index
            worker_log = logs.enter_context((run / f"worker-{index:03d}.log").open("wb"))
            workers.append(subprocess.Popen([
                str(PYTHON), "-m", "agentrl.worker", "-c", str(config_path),
                "--controller", controller_url,
                "--self", f"http://127.0.0.1:{port}/api",
                "--host", "127.0.0.1", "--port", str(port), task_name,
            ], cwd=agentbench, env=env, stdout=worker_log, stderr=subprocess.STDOUT))
        stopping = False

        def interrupted(_signum, _frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGTERM, interrupted)
        signal.signal(signal.SIGINT, interrupted)
        try:
            ready = wait_ready(
                controller, workers, controller_url, task_names, args.startup_timeout,
            )
            write_json(run / "READY.json", {
                "controller_url": controller_url, "task_names": task_names,
                "assignments": assignments,
                "controller_pid": controller.pid,
                "worker_pids": [worker.pid for worker in workers],
                "health": ready,
                "agentbench_dir": str(agentbench),
                "scratch": str(scratch),
            })
            while not stopping:
                if controller.poll() is not None or any(
                        worker.poll() is not None for worker in workers):
                    raise RuntimeError("ALFWorld controller or worker exited unexpectedly")
                time.sleep(1)
        finally:
            for worker in reversed(workers):
                stop(worker)
            stop(controller)
            write_json(run / "STOPPED.json", {
                "controller_returncode": controller.returncode,
                "worker_returncodes": [worker.returncode for worker in workers],
            })


if __name__ == "__main__":
    main()
