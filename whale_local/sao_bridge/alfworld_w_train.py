"""Run one native SAO Solver phase in official AgentBench ALFWorld."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

from .alfworld_h_train import monitor_gpus
from .budget import model_digests
from .common import ensure_model_runtime_metadata, read_json, write_json
from .joint_context import joint_context_audit
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


def assignment_schedule_audit(config: dict) -> dict:
    rows = read_json(Path(config["data"]))
    required = config["steps"] * config["batch_size"]
    if len(rows) < required:
        raise ValueError(
            f"Solver data has {len(rows)} rows but the phase requires {required}"
        )
    active_window = config["batch_size"] * (config.get("staleness", 1) + 1)
    batches = []
    for offset in range(0, required, config["batch_size"]):
        batch_rows = rows[offset:offset + config["batch_size"]]
        names = [row["task_name"] for row in batch_rows]
        if len(names) != len(set(names)):
            raise ValueError(
                f"Solver batch {offset // config['batch_size']} reuses an "
                "AgentBench assignment; interleave tasks across packet workers"
            )
        batches.append({
            "batch": offset // config["batch_size"],
            "task_ids": [row["task_id"] for row in batch_rows],
            "task_names": names,
        })
        window_names = [
            row["task_name"] for row in rows[offset:min(required, offset + active_window)]
        ]
        if len(window_names) != len(set(window_names)):
            raise ValueError(
                f"Solver active window at row {offset} reuses an AgentBench "
                "assignment; interleave across all concurrent rollouts"
            )
    return {
        "required_rows": required,
        "available_rows": len(rows),
        "active_window": active_window,
        "batches": batches,
    }


def training_spec(config: dict, run: Path) -> dict:
    return {
        "run": str(run / "optimizer"),
        "role": "solver",
        "domain": "alfworld",
        "context_tokens": config.get("context_tokens", 32768),
        "model": config["solver_model"],
        "critic_model": config.get("solver_critic", config["solver_model"]),
        "phase": config["phase"],
        "experiment": config["experiment"],
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "seed": config.get("seed", 42),
        "lr": config.get("learning_rate", 1e-6),
        "critic_lr": config.get("critic_learning_rate", 5e-6),
        "tokens": config.get("solver_tokens", 4096),
        "temperature": config.get("temperature", 1.0),
        "staleness": config.get("staleness", 1),
        "harness": config["harness"],
        "data": config["data"],
        "evaluator_url": None,
        "central_context": config.get("central_context"),
        "interaction_credit": 0.0,
        "weight_update_mode": config.get("weight_update_mode", "xccl"),
        "actor_gpus": config.get("actor_gpus", 4),
        "rollout_gpus": config.get("rollout_gpus", 4),
        "recycle_pool": config.get("recycle_pool"),
        "critic_warmup_steps": config.get("critic_warmup_steps", 0),
        "gdn_prefill_backend": "triton",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    schedule_audit = assignment_schedule_audit(config)
    context_audit = None
    if config.get("central_context"):
        context_audit = joint_context_audit(
            Path(config["central_context"]),
            Path(config.get("solver_critic", config["solver_model"])),
        )
    run = Path(config["run"]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(run / "schedule_audit.json", schedule_audit)
    if context_audit is not None:
        write_json(run / "joint_context_audit.json", context_audit)
    write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
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
        controller_port = config.get("controller_port", 28000)
        worker_port = config.get("worker_port", 28100)
        env_process = launch.module(
            "sao_bridge.alfworld_env_service",
            [
                "--run", run / "environment",
                "--assignments", config["assignments"],
                "--controller-port", controller_port,
                "--worker-port", worker_port,
                "--concurrency", config.get("environment_concurrency", 8),
            ],
            gpus="", name=f"{config['phase']}_environment",
        )
        controller_url = f"http://127.0.0.1:{controller_port}/api"
        launch.ready(env_process, controller_url + "/list_workers")
        ready_path = run / "environment/READY.json"
        deadline = time.monotonic() + float(config.get("environment_startup_timeout", 900))
        while not ready_path.is_file():
            launch.check()
            if env_process.poll() is not None:
                raise RuntimeError("ALFWorld environment service exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("ALFWorld worker registration timed out")
            time.sleep(1)
        launch.env["SAO_ALFWORLD_CONTROLLER_URL"] = controller_url
        launch.env["SAO_ALFWORLD_HTTP_TIMEOUT"] = str(
            config.get("environment_http_timeout", 900)
        )
        spec = training_spec(config, run)
        write_json(run / "train_spec.json", spec)
        trainer = launch.module(
            "sao_bridge.large_train", ["--spec", run / "train_spec.json"],
            gpus="0,1,2,3,4,5,6,7", name=f"{config['phase']}_train",
        )
        launch.wait(
            trainer, f"{config['phase']}_train",
            timeout=float(config.get("phase_timeout_seconds", 21600)),
        )
        trained = read_json(run / "optimizer/trained.json")
        expected_actor_steps = config["steps"] - config.get("critic_warmup_steps", 0)
        if trained["actor_steps"] != expected_actor_steps:
            raise RuntimeError("Solver optimizer did not complete its declared updates")
        supplements = ensure_model_runtime_metadata(
            Path(trained["checkpoints"]["actor"]), Path(config["solver_model"])
        )
        initial, initial_weights = model_digests(Path(config["solver_model"]))
        final, final_weights = model_digests(Path(trained["checkpoints"]["actor"]))
        if initial_weights == final_weights:
            raise RuntimeError("Solver actor checkpoint did not change")
        write_json(run / "COMPLETE.json", {
            "status": "completed", "phase": config["phase"],
            "elapsed_seconds": time.time() - started,
            "initial_solver_sha256": initial,
            "final_solver_sha256": final,
            "initial_solver_weights_sha256": initial_weights,
            "final_solver_weights_sha256": final_weights,
            "harness_version": config["harness_version"],
            "runtime_supplements": supplements,
            "trained": trained,
        })
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
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
