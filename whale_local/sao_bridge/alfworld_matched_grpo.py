"""Matched 192-trajectory ALFWorld continuation using the official GRPO objective."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import httpx

from .alfworld_h_train import reward_config
from .alfworld_reward_service import infrastructure_failure, validate_metadata
from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit, stop_known

H0 = ROOT / "assets/Harness-R1/harness-r1"
H0_WEIGHT_SHA = "7d16d5ad5734af16b1fd0619caeb04270a024df86ca6e6c6e9bdb60e45f39af7"
SAO_CONFIG = ROOT / "runs/alfworld_engineer_fast_v1_s42/training/config.json"
EVAL_PROTOCOL = ROOT / "runs/alfworld_core_200_oneshot_v2/protocol.json"
GRPO_CONFIG = ROOT / "configs/alfworld_engineer_matched_grpo.json"


def matched_rows(records: list[dict]) -> list[dict]:
    groups = {}
    counts = Counter()
    for record in records:
        key = json.dumps(record["messages"], sort_keys=True)
        metadata = json.loads(record["metadata_json"])
        row = {"prompt": record["messages"], "metadata": metadata}
        if key in groups and groups[key] != row:
            raise ValueError("Identical prompts have inconsistent reward metadata")
        groups[key] = row
        counts[key] += 1
    if len(records) != 192 or len(groups) != 24 or set(counts.values()) != {8}:
        raise ValueError("Expected exactly the matched 24 prompts repeated eight times")
    return list(groups.values())


def validate_reward_rows(rows: list[dict], config: dict) -> None:
    heldout = set(config["excluded_eval_task_ids"])
    for row in rows:
        metadata = row["metadata"]
        validate_metadata(metadata, config["solver_version"])
        tasks = set(range(metadata["start"], metadata["end"]))
        if tasks & heldout or not tasks <= set(range(240)):
            raise ValueError("GRPO reward data contains held-out or nonmatched tasks")


def preflight(args, config: dict, reward: dict) -> tuple[list[dict], dict]:
    original = read_json(SAO_CONFIG)
    if (args.model.resolve() != Path(original["engineer_model"]).resolve()
            or args.model.resolve() != H0.resolve()
            or file_sha(H0 / "model.safetensors") != H0_WEIGHT_SHA):
        raise ValueError("Matched GRPO must start from exactly the original SAO H0 weights")
    if args.data.resolve() != Path(original["data"]).resolve() or args.updates != 6:
        raise ValueError("Matched GRPO requires the original data and six 32-sample rollouts")
    if (config["domain"] != "alfworld" or config["max_response_tokens"] != 4096
            or config["max_context_tokens"] != 32768 or config["kl_coef"] != 0):
        raise ValueError("Matched GRPO generation budget or official no-KL objective changed")
    if reward["solver_version"] != original["solver_version"]:
        raise ValueError("Matched GRPO reward Solver must be the original W0")
    rows = matched_rows(read_json(args.data))
    validate_reward_rows(rows, reward)
    return rows, {
        "domain": "alfworld", "initialization": "official RL H0, not fresh SFT",
        "matched_sao_config_sha256": file_sha(SAO_CONFIG),
        "eval_exclusion_protocol_sha256": file_sha(EVAL_PROTOCOL),
        "policy_trajectories": 192, "distinct_prompts": 24,
        "samples_per_prompt": 8, "budget_checkpoints": [96, 192],
        "sao_actor_trajectories": 162, "sao_critic_only_trajectories": 30,
        "budget_scope": "policy trajectories and prompt multiset, not equal total tokens or FLOPs",
    }


async def score_records(records: list[dict], config: dict, seed: int) -> list[dict]:
    validate_reward_rows([{"metadata": r["metadata"]} for r in records], config)
    async with httpx.AsyncClient(timeout=7200, trust_env=False) as client:
        async def score(index: int, record: dict) -> dict:
            response = await client.post(config["reward_url"] + "/score", json={
                "source": record["response"], "metadata": record["metadata"],
                "sample_id": f"matched-grpo-{seed}-{index}",
            })
            response.raise_for_status()
            result = response.json()
            evaluation = result["evaluation"]
            value = float(result["reward"])
            if (infrastructure_failure(evaluation) or not math.isfinite(value)
                    or value != float(evaluation["score"])
                    or value != float(evaluation["delta_score"])):
                raise RuntimeError(f"Invalid official delta reward: {evaluation}")
            return evaluation
        return await asyncio.gather(*(score(i, r) for i, r in enumerate(records)))


def target(root: Path) -> None:
    config = read_json(SAO_CONFIG)
    run = root / "target"
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(run / "supervision")
    try:
        with launch.vllm(
            config["solver_model"], port=19601, gpus="0,1,2,3,4,5,6,7",
            name="fixed_w0", data_parallel=8, max_model_len=32768,
            tool_call_parser="qwen3_xml", gdn_prefill_backend="triton",
            language_model_only=True, max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as urls:
            config = dict(config, reward_concurrency=8, reward_rollout_concurrency=10,
                          tmpdir="/dev/shm/alfworld-matched-grpo-target")
            reward = reward_config(config, run, urls.split(","))
            write_json(run / "reward_config.json", reward)
            process = launch.module("sao_bridge.alfworld_reward_service", [
                "--config", run / "reward_config.json", "--port", 28760,
            ], gpus="", name="reward")
            launch.ready(process, "http://127.0.0.1:28760/health")
            write_json(run / "READY.json", {"solver_version": config["solver_version"],
                                          "ready_at_unix": time.time()})
            deadline = time.monotonic() + 6 * 3600
            while not (root / "STOP_TARGET").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Fixed W0 target lifetime exceeded six hours")
                if process.poll() is not None:
                    raise RuntimeError("Reward service exited")
                launch.check()
                time.sleep(2)
    except BaseException as exc:
        write_json(run / "FAILED.json", {"error": repr(exc)})
        raise
    finally:
        launch.close()
        write_json(run / "gpu_release.json", gpu_release_audit())


def train(root: Path) -> None:
    config = read_json(SAO_CONFIG)
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(root / "supervision")
    try:
        deadline = time.monotonic() + 1800
        while not (root / "target/READY.json").is_file():
            if (root / "target/FAILED.json").exists():
                raise RuntimeError(f"Fixed W0 target failed: {read_json(root / 'target/FAILED.json')}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Fixed W0 target did not become ready within 30 minutes")
            launch.check()
            time.sleep(2)
        tunnel = launch.start([
            "ssh", "-F", "/mnt/3fs/xudong/cluster-ssh/config", "-N",
            "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3", "-L", "28761:127.0.0.1:28760",
            "node3-internal",
        ], gpus="", name="reward_tunnel")
        launch.ready(tunnel, "http://127.0.0.1:28761/health", timeout=1800)
        protocol = read_json(EVAL_PROTOCOL)
        reward = {
            "reward_url": "http://127.0.0.1:28761",
            "solver_version": config["solver_version"],
            "excluded_eval_task_ids": protocol["all_task_ids"],
        }
        write_json(root / "reward_client.json", reward)
        grpo_config = read_json(GRPO_CONFIG)
        rollout_gpus = ",".join(str(i) for i in range(4, 4 + grpo_config["rollout_tp"]))
        process = launch.start([
            sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
            "--local-addr=127.0.0.1", "--nproc-per-node=4", "--max-restarts=0",
            "-m", "harness_r1_searchqa.grpo_train", "--domain", "alfworld",
            "--model", config["engineer_model"], "--data", config["data"],
            "--reward-config", str(root / "reward_client.json"),
            "--config", str(GRPO_CONFIG), "--output", str(root / "engine"),
            "--updates", "6", "--rollout-gpus", rollout_gpus, "--seed", "42",
        ], gpus="0,1,2,3", name="grpo")
        launch.wait(process, "Matched Harness-R1 GRPO", timeout=5 * 3600)
        progress = read_json(root / "engine/progress.json")
        if progress["updates_completed"] != 6 or progress["optimizer_steps"] != 24:
            raise RuntimeError("GRPO stopped before the matched 192-sample budget")
        write_json(root / "TRAINING_COMPLETE.json", progress)
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc)})
        raise
    finally:
        launch.close()
        write_json(root / "STOP_TARGET", {"stopped_at_unix": time.time()})
        write_json(root / "training_gpu_release.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("target", "train"), required=True)
    args = parser.parse_args()
    (target if args.mode == "target" else train)(args.root.resolve())


if __name__ == "__main__":
    main()
