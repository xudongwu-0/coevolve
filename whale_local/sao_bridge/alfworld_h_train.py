"""Run one native SAO Engineer phase against a frozen ALFWorld Solver."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
import time

from .budget import model_digests
from .alfworld_reward_service import reward_data_identity, validate_metadata
from .common import ensure_model_runtime_metadata, read_json, write_json
from .gpu_telemetry import (
    monitor as monitor_gpu_processes,
    summarize as summarize_gpu_processes,
    summarize_active_training,
)
from .joint_context import joint_context_audit
from .large_train import training_capacity_audit, validate_training_rows
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


def training_spec(config: dict, run: Path, evaluator_url: str) -> dict:
    return {
        "run": str(run / "optimizer"),
        "role": "editor",
        "domain": "alfworld",
        "context_tokens": config.get("context_tokens", 40960),
        "microbatch_tokens": config.get("microbatch_tokens", 20000),
        "model": config["engineer_model"],
        "critic_model": config.get("engineer_critic", config["engineer_model"]),
        "phase": config["phase"],
        "experiment": config["experiment"],
        "steps": config["steps"],
        "batch_size": config["batch_size"],
        "seed": config.get("seed", 42),
        "lr": config.get("learning_rate", 1e-6),
        "critic_lr": config.get("critic_learning_rate", 5e-6),
        "tokens": config.get("engineer_tokens", 12288),
        "temperature": config.get("temperature", 0.7),
        "staleness": config.get("staleness", 1),
        "max_head_offpolicyness": config.get(
            "max_head_offpolicyness", config.get("staleness", 1)
        ),
        "harness": config["harness"],
        "data": config["data"],
        "evaluator_url": evaluator_url,
        "central_context": config.get("central_context"),
        "interaction_credit": config.get("interaction_credit", 0.0),
        "weight_update_mode": config.get("weight_update_mode", "xccl"),
        "actor_gpus": config.get("actor_gpus", 2),
        "rollout_gpus": config.get("rollout_gpus", 2),
        "recycle_pool": config.get("recycle_pool"),
        "critic_warmup_steps": config.get("critic_warmup_steps", 0),
        "save_interval_steps": config.get("save_interval_steps", config["steps"]),
        "critic_updates_per_step": config.get("critic_updates_per_step", 2),
        "critic_mode": config.get("critic_mode", "learned"),
        "terminal_episodes": config.get("terminal_episodes", False),
        "independent_trajectories": config.get("independent_trajectories", False),
        "processed_behavior_logprobs": config.get("processed_behavior_logprobs", False),
        "actor_gradient_checkpointing": config.get(
            "actor_gradient_checkpointing", True
        ),
        "critic_gradient_checkpointing": config.get(
            "critic_gradient_checkpointing", True
        ),
        "gdn_prefill_backend": "triton",
    }


def monitor_gpus(path: Path, stop: threading.Event, interval: float) -> None:
    monitor_gpu_processes(path, stop, interval)


def reward_config(config: dict, run: Path, target_urls: list[str]) -> dict:
    reward = {
        "cache_root": str(run / "reward_cache"),
        "receipts": str(run / "reward_receipts"),
        "port_locks": str(run / "reward_port_locks"),
        "agentbench_dir": str(run / "runtime/AgentBench"),
        "tmpdir": config["tmpdir"],
        "target_urls": target_urls,
        "target_model": "policy",
        "target_agent_name": "qwen35-9b-nothink",
        "solver_version": config["solver_version"],
        "rollout_concurrency": config.get("reward_rollout_concurrency", 8),
        "max_concurrency": config.get("reward_concurrency", config["batch_size"]),
        "controller_port_base": config.get("reward_controller_port_base", 30000),
        "worker_port_base": config.get("reward_worker_port_base", 32000),
        "port_probe_span": config.get("reward_port_probe_span", 7000),
        "split": config.get("training_split", "test_set"),
        "catalogue": config.get("training_catalogue"),
        "catalogue_sha256": config.get("training_catalogue_sha256"),
        "solver_weights_sha256": config.get("solver_weights_sha256"),
    }
    reward_data_identity(reward)
    return reward


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
        preflight_spec = training_spec(
            config, run, f"http://127.0.0.1:{config.get('reward_service_port', 28600)}"
        )
        rows = read_json(Path(config["data"]))
        validate_training_rows(preflight_spec, rows)
        capacity_audit = training_capacity_audit(preflight_spec, rows)
        write_json(run / "capacity_audit.json", capacity_audit)
        reward = reward_config(config, run, [])
        identity = reward_data_identity(reward)
        if identity is not None:
            if model_digests(Path(config["solver_model"]))[1] != identity["solver_weights_sha256"]:
                raise ValueError("Engineer reward Solver weights differ from the frozen identity")
            for row in rows:
                validate_metadata(
                    json.loads(row["metadata_json"]), config["solver_version"],
                    split=reward["split"], catalogue=reward["catalogue"],
                    catalogue_sha256=reward["catalogue_sha256"],
                    solver_weights_sha256=reward["solver_weights_sha256"],
                )
        if config.get("central_context"):
            write_json(
                run / "joint_context_audit.json",
                joint_context_audit(
                    Path(config["central_context"]),
                    Path(config.get("engineer_critic", config["engineer_model"])),
                ),
            )
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "stage": "pre_gpu_training_audit",
            "elapsed_seconds": time.time() - started,
            "error": repr(exc),
        })
        raise
    write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(run / "supervision")
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=monitor_gpus,
        args=(run / "gpu_process_samples.jsonl", telemetry_stop,
              float(config.get("monitor_interval_seconds", 5))),
        daemon=True,
    )
    telemetry.start()
    try:
        with launch.vllm(
            config["solver_model"], port=config.get("solver_port", 19501),
            gpus="4,5,6,7", name=f"{config['phase']}_solver",
            data_parallel=4, max_model_len=32768, tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton", language_model_only=True,
            max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as urls:
            reward = reward_config(config, run, urls.split(","))
            write_json(run / "reward_config.json", reward)
            reward_port = config.get("reward_service_port", 28600)
            reward_process = launch.module(
                "sao_bridge.alfworld_reward_service",
                ["--config", run / "reward_config.json", "--port", reward_port],
                gpus="", name=f"{config['phase']}_reward",
            )
            launch.ready(reward_process, f"http://127.0.0.1:{reward_port}/health")
            spec = training_spec(config, run, f"http://127.0.0.1:{reward_port}")
            spec["capacity_audit"] = capacity_audit
            write_json(run / "train_spec.json", spec)
            trainer = launch.module(
                "sao_bridge.large_train", ["--spec", run / "train_spec.json"],
                gpus="0,1,2,3", name=f"{config['phase']}_train",
            )
            launch.wait(
                trainer, f"{config['phase']}_train",
                timeout=float(config.get("phase_timeout_seconds", 21600)),
            )
            trained = read_json(run / "optimizer/trained.json")
            expected_actor_steps = config["steps"] - config.get("critic_warmup_steps", 0)
            if trained["actor_steps"] != expected_actor_steps:
                raise RuntimeError("Engineer optimizer did not complete its declared updates")
            supplements = ensure_model_runtime_metadata(
                Path(trained["checkpoints"]["actor"]), Path(config["engineer_model"])
            )
            initial, initial_weights = model_digests(Path(config["engineer_model"]))
            final, final_weights = model_digests(Path(trained["checkpoints"]["actor"]))
            if initial_weights == final_weights:
                raise RuntimeError("Engineer actor checkpoint did not change")
            write_json(run / "COMPLETE.json", {
                "status": "completed", "phase": config["phase"],
                "elapsed_seconds": time.time() - started,
                "initial_engineer_sha256": initial,
                "final_engineer_sha256": final,
                "initial_engineer_weights_sha256": initial_weights,
                "final_engineer_weights_sha256": final_weights,
                "solver_version": config["solver_version"],
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
        telemetry_path = run / "gpu_process_samples.jsonl"
        if telemetry_path.is_file():
            try:
                utilization = summarize_gpu_processes(telemetry_path)
            except ValueError as exc:
                utilization = {"status": "invalid", "error": repr(exc)}
            write_json(run / "gpu_utilization_summary.json", utilization)
            active_utilization = summarize_active_training(
                telemetry_path,
                run / "optimizer/training_active.json",
            )
            if active_utilization is not None:
                write_json(
                    run / "gpu_optimizer_utilization_summary.json",
                    active_utilization,
                )
        write_json(run / "RELEASED.json", {
            "released_at": time.time(),
            **gpu_release_audit(),
        })


if __name__ == "__main__":
    main()
