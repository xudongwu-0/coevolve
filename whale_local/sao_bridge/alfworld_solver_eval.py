"""Paired, fixed-no-patch ALFWorld evaluation for one Solver checkpoint."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
from pathlib import Path
import threading
import time

from .alfworld_domain import INITIAL_SOURCE
from .common import read_json, write_json
from .evaluation import Sampler, evaluate
from .gpu_telemetry import monitor, summarize
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


def validate_config(config: dict) -> dict:
    required = {
        "run", "solver_model", "solver_version", "task_ids", "no_patch",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing ALFWorld Solver evaluation keys: {sorted(missing)}")
    normalized = dict(config)
    model = Path(normalized["solver_model"]).resolve()
    no_patch = Path(normalized["no_patch"]).resolve()
    if not model.is_dir() or not list(model.glob("*.safetensors")):
        raise FileNotFoundError("ALFWorld Solver checkpoint has no safetensors")
    if not no_patch.is_file():
        raise FileNotFoundError("ALFWorld no-patch artifact is unavailable")
    patch = read_json(no_patch)
    if (
        not isinstance(patch, dict)
        or patch.get("benchmark") != "alfworld"
        or patch.get("actions") != []
    ):
        raise ValueError("ALFWorld Solver evaluation requires the empty no-patch harness")
    task_ids = normalized["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
        or any(type(task_id) is not int or not 0 <= task_id < 500 for task_id in task_ids)
    ):
        raise ValueError("task_ids must be distinct official ALFWorld IDs")
    normalized.setdefault("samples", 1)
    normalized.setdefault("solver_tokens", 4096)
    normalized.setdefault("temperature", 1.0)
    normalized.setdefault("eval_concurrency", 32)
    normalized.setdefault("environment_concurrency", 32)
    normalized.setdefault("monitor_interval_seconds", 5)
    for key in (
        "seed", "samples", "solver_tokens", "eval_concurrency",
        "environment_concurrency",
    ):
        if type(normalized[key]) is not int or normalized[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    normalized.update(
        run=str(Path(normalized["run"]).resolve()),
        solver_model=str(model),
        no_patch=str(no_patch),
    )
    return normalized


def run(config_path: Path) -> dict:
    config = validate_config(read_json(config_path))
    root = Path(config["run"])
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_json(root / "config.json", config)
    write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(root / "supervision")
    telemetry_stop = threading.Event()
    telemetry_path = root / "gpu_process_samples.jsonl"
    telemetry = threading.Thread(
        target=monitor,
        args=(telemetry_path, telemetry_stop, config["monitor_interval_seconds"]),
        daemon=True,
    )
    telemetry.start()
    started = time.monotonic()
    try:
        version_id = hashlib.sha256(
            config["solver_version"].encode()
        ).hexdigest()[:12]
        task_name = f"solver-eval-{version_id}"
        assignments = [{
            "name": task_name,
            "task_ids": config["task_ids"],
            "harness": config["no_patch"],
        }]
        assignments_path = root / "assignments.json"
        write_json(assignments_path, assignments)
        env_process = launch.module(
            "sao_bridge.alfworld_env_service",
            [
                "--run", root / "environment",
                "--assignments", assignments_path,
                "--controller-port", config.get("controller_port", 28000),
                "--worker-port", config.get("worker_port", 28100),
                "--concurrency", config["environment_concurrency"],
            ],
            gpus="", name="environment",
        )
        controller_url = (
            f"http://127.0.0.1:{config.get('controller_port', 28000)}/api"
        )
        launch.ready(env_process, controller_url + "/list_workers")
        ready = root / "environment/READY.json"
        deadline = time.monotonic() + float(config.get("startup_timeout_seconds", 900))
        while not ready.is_file():
            launch.check()
            if env_process.poll() is not None:
                raise RuntimeError("ALFWorld environment exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("ALFWorld environment readiness timed out")
            time.sleep(1)
        launch.env["SAO_ALFWORLD_CONTROLLER_URL"] = controller_url
        launch.env["SAO_ALFWORLD_HTTP_TIMEOUT"] = str(
            config.get("environment_http_timeout", 900)
        )
        with launch.vllm(
            config["solver_model"],
            port=config.get("solver_port", 19801),
            gpus="0,1,2,3,4,5,6,7",
            name="solver",
            data_parallel=8,
            max_model_len=32768,
            tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton",
            language_model_only=True,
            max_num_seqs=max(8, config["eval_concurrency"] // 8),
            server_args=["--enable-prefix-caching"],
        ) as urls:
            rows = [
                {
                    "id": f"alfworld-{task_id}",
                    "task_id": task_id,
                    "task_name": task_name,
                }
                for task_id in config["task_ids"]
            ]

            async def evaluate_checkpoint():
                sampler = Sampler(
                    urls,
                    concurrency=config["eval_concurrency"],
                    temperature=float(config["temperature"]),
                    chat_template_kwargs={"enable_thinking": False},
                    request_timeout=float(config.get("request_timeout_seconds", 900)),
                )
                try:
                    return await evaluate(
                        sampler,
                        rows,
                        INITIAL_SOURCE,
                        tokens=config["solver_tokens"],
                        samples=config["samples"],
                        seed=config["seed"],
                        domain="alfworld",
                    )
                finally:
                    await sampler.close()

            report = asyncio.run(evaluate_checkpoint())
        write_json(root / "report.json", report)
        complete = {
            "status": "completed",
            "solver_model": config["solver_model"],
            "solver_version": config["solver_version"],
            "task_ids": config["task_ids"],
            "seed": config["seed"],
            "samples": config["samples"],
            "accuracy": report["accuracy"],
            "elapsed_seconds": time.monotonic() - started,
            "test_split_opened": any(task_id >= 400 for task_id in config["task_ids"]),
        }
        write_json(root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise
    finally:
        launch.close()
        telemetry_stop.set()
        telemetry.join(timeout=config["monitor_interval_seconds"] + 10)
        if telemetry_path.is_file():
            try:
                telemetry_summary = summarize(telemetry_path)
            except ValueError as exc:
                telemetry_summary = {
                    "status": "invalid",
                    "error": repr(exc),
                }
            write_json(root / "gpu_utilization_summary.json", telemetry_summary)
        write_json(root / "RELEASED.json", {
            "released_at": time.time(),
            **gpu_release_audit(),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config), flush=True)


if __name__ == "__main__":
    main()
