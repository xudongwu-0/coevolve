"""Expose the official Harness-R1 ALFWorld patch reward to SAO rollouts."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .common import ROOT, read_json, write_json

UPSTREAM = ROOT / "third_party/harness-r1-upstream"
RELAX = UPSTREAM / "code/Relax"
AGENTBENCH = ROOT / "assets/alfworld_harness_r1/AgentBench"
ALFWORLD_PYTHON = "/dev/shm/harness-r1-alfworld-venv/bin/python"
DEFAULT_TMPDIR = ROOT / "assets/alfworld_harness_r1/scratch"


class RewardRequest(BaseModel):
    source: str
    sample_id: str
    metadata: dict
    reference_source: str | None = None


def official_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        harness_r1_eval_root=str(Path(config["cache_root"]).resolve()),
        harness_r1_agentbench_dir=str(AGENTBENCH),
        harness_r1_agentbench_python=ALFWORLD_PYTHON,
        harness_r1_alfworld_worker_python=ALFWORLD_PYTHON,
        harness_r1_target_urls=list(config["target_urls"]),
        harness_r1_target_model=config.get("target_model", "policy"),
        harness_r1_agent_name=config.get("target_agent_name", "qwen35-9b-nothink"),
        harness_r1_response_protocol="prefill_think_patch",
        harness_r1_require_think_patch=False,
        harness_r1_require_code_hook_only=True,
        harness_r1_reject_runtime_noop_patch=True,
        harness_r1_alfworld_split="test_set",
        harness_r1_alfworld_max_step=50,
        harness_r1_rollout_concurrency=config.get("rollout_concurrency", 8),
        harness_r1_rollout_max_tokens=4096,
        harness_r1_rollout_http_timeout=900,
        harness_r1_rollout_tool_choice="auto",
        harness_r1_rollout_chat_template_kwargs={"enable_thinking": False},
        harness_r1_reward_metric="delta_average_reward",
        harness_r1_reward_threshold=1.0,
        harness_r1_valid_bonus=0.0,
        harness_r1_allow_partial_eval_reward=False,
        harness_r1_retry_incomplete_eval=True,
        harness_r1_auto_retry=True,
        harness_r1_startup_timeout=900,
        harness_r1_rollout_timeout=2400,
        harness_r1_reward_timeout=4200,
        harness_r1_cache_lock_timeout=7200,
        harness_r1_controller_port_base=config.get("controller_port_base", 14000),
        harness_r1_worker_port_base=config.get("worker_port_base", 24000),
        harness_r1_port_probe_span=config.get("port_probe_span", 7000),
        harness_r1_eval_max_attempts=3,
        harness_r1_eval_retry_on_infra_failure=2,
        harness_r1_reward_max_concurrency=config.get("max_concurrency", 8),
        reward_max_concurrency=config.get("max_concurrency", 8),
    )


def validate_metadata(metadata: dict, solver_version: str) -> None:
    required = {
        "batch_tag", "split", "start", "end", "batch_size", "baseline_pass",
        "baseline_rewards", "reward_threshold", "target_agent_name", "target_model",
        "response_protocol", "solver_version",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"ALFWorld reward metadata missing: {sorted(missing)}")
    if metadata["solver_version"] != solver_version:
        raise ValueError("ALFWorld packet was collected from a different Solver version")
    start, end = metadata["start"], metadata["end"]
    rewards = {int(key): value for key, value in metadata["baseline_rewards"].items()}
    if (type(start) is not int or type(end) is not int or end <= start
            or set(rewards) != set(range(start, end))
            or metadata["batch_size"] != end - start):
        raise ValueError("ALFWorld reward packet must use one complete contiguous cohort")
    if metadata["response_protocol"] != "prefill_think_patch":
        raise ValueError("ALFWorld Engineer packet protocol changed")


def infrastructure_failure(result: dict) -> bool:
    status = str(result.get("eval_status", "")).lower()
    return any(token in status for token in (
        "eval_failed", "eval_timeout", "reward_error", "request_failed", "missing_source",
    ))


def create_app(config: dict) -> FastAPI:
    tmpdir = Path(config.get("tmpdir", DEFAULT_TMPDIR)).resolve()
    tmpdir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmpdir)
    sys.path.insert(0, str(RELAX))
    from examples.harness_r1.reward_alfworld_patch import reward_func
    from relax.utils.types import Sample

    args = official_args(config)
    Path(args.harness_r1_eval_root).mkdir(parents=True, exist_ok=True)
    receipts = Path(config["receipts"]).resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    os.environ["HARNESS_R1_PORT_LOCK_DIR"] = str(Path(config["port_locks"]).resolve())
    Path(os.environ["HARNESS_R1_PORT_LOCK_DIR"]).mkdir(parents=True, exist_ok=True)
    solver_version = config["solver_version"]
    semaphore = asyncio.Semaphore(config.get("max_concurrency", 8))
    app = FastAPI()

    @app.get("/health")
    def health():
        return {
            "ready": True, "solver_version": solver_version,
            "target_urls": len(args.harness_r1_target_urls),
            "reward": "official_harness_r1_delta_average_reward",
        }

    @app.post("/score")
    async def score(request: RewardRequest):
        try:
            validate_metadata(request.metadata, solver_version)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        samples = [Sample(response=request.source, metadata=dict(request.metadata))]
        if request.reference_source is not None:
            samples.append(Sample(
                response=request.reference_source, metadata=dict(request.metadata),
            ))
        started = time.monotonic()
        async with semaphore:
            results = await reward_func(args, samples)
        if isinstance(results, dict):
            results = [results]
        if len(results) != len(samples):
            raise HTTPException(status_code=502, detail="official reward result count mismatch")
        failed = [result for result in results if infrastructure_failure(result)]
        if failed:
            raise HTTPException(
                status_code=502,
                detail={"error": "official ALFWorld reward infrastructure failure",
                        "statuses": [row.get("eval_status") for row in failed]},
            )
        candidate = results[0]
        reward = float(candidate["score"])
        if len(results) == 2:
            reward -= float(results[1]["score"])
        identity = hashlib.sha256(
            (request.sample_id + "\0" + request.source).encode()
        ).hexdigest()
        receipt = {
            "sample_id": request.sample_id, "solver_version": solver_version,
            "reward": reward, "candidate": candidate,
            "reference": results[1] if len(results) == 2 else None,
            "elapsed_seconds": time.monotonic() - started,
        }
        await asyncio.to_thread(write_json, receipts / f"{identity}.json", receipt)
        return {
            "valid": bool(candidate.get("valid_patch")),
            "reward": reward, "evaluation": candidate,
            "reference_evaluation": results[1] if len(results) == 2 else None,
            "source": request.source, "reference_source": request.reference_source,
            "sample_id": request.sample_id,
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    import uvicorn
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
