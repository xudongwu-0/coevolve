"""Generate fresh-base Engineer candidates and select train-rerun SFT examples."""
from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TypeVar
import time

import httpx
import yaml

from sao_bridge.common import file_sha, write_json
from .patch import PatchValidationError, extract_and_normalize_patch, patch_digest
from .reward import reward_func, _validate_metadata

T = TypeVar("T")


async def transport_retry(
    request: Callable[[], Awaitable[T]], *, output: Path, identifier: str, operation: str,
) -> T:
    for attempt in range(1, 4):
        try:
            return await request()
        except (
            httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError,
            httpx.WriteError, httpx.ConnectTimeout, httpx.ReadTimeout,
        ) as exc:
            write_json(output / "infrastructure_errors"
                       / f"{identifier}-{operation}-{attempt:02d}-{time.time_ns()}.json", {
                "generation_id": identifier, "operation": operation, "attempt": attempt,
                "error_type": type(exc).__name__, "error": str(exc),
                "retrying": attempt < 3, "reward_assigned": False,
                "same_response_and_task_seeds": True,
            })
            if attempt == 3:
                raise
            await asyncio.sleep(attempt)
    raise AssertionError("Unreachable retry state")


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--engineer-urls", required=True)
    parser.add_argument("--engineer-manifest", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packet-count", type=int, default=32)
    parser.add_argument("--packet-start", type=int, default=0)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--reward-concurrency", type=int, default=4)
    parser.add_argument("--minimum-examples", type=int, default=256)
    parser.add_argument("--max-response-tokens", type=int, default=12288)
    parser.add_argument("--max-prompt-tokens", type=int, default=20479)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--wait-for-data-seconds", type=int, default=0)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    for name in ("packet_count", "samples", "concurrency", "reward_concurrency"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    deadline = time.monotonic() + args.wait_for_data_seconds
    while not args.data.is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(f"Verified failure packet data is not available: {args.data}")
        await asyncio.sleep(3)
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if args.packet_start < 0 or len(rows) < args.packet_start + args.packet_count:
        raise ValueError("Not enough verified failure packets for requested bootstrap")
    rows = rows[args.packet_start : args.packet_start + args.packet_count]
    urls = [url.rstrip("/") for url in args.engineer_urls.split(",")]
    reward_args = SimpleNamespace(**yaml.safe_load(args.reward_config.read_text()))
    for row in rows:
        _validate_metadata(reward_args, row["metadata"])
    args.output.mkdir(parents=True, exist_ok=True)
    request_identity = {
        "data_sha256": file_sha(args.data),
        "engineer_manifest_sha256": file_sha(args.engineer_manifest),
        "reward_config_sha256": file_sha(args.reward_config),
        "packet_start": args.packet_start,
        "packets": args.packet_count, "samples_per_prompt": args.samples,
        "response_protocol": "prefill_think_patch",
        "temperature": 0.7, "top_p": 0.95, "seed": args.seed,
        "max_response_tokens": args.max_response_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
    }
    identity_path = args.output / "INPUT_IDENTITY.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != request_identity:
            raise ValueError("Bootstrap resume identity differs from recorded inputs")
    else:
        write_json(identity_path, request_identity)
    generation_sem = asyncio.Semaphore(args.concurrency)
    scoring_sem = asyncio.Semaphore(args.reward_concurrency)
    completed: list[dict] = []
    started = time.monotonic()

    async with httpx.AsyncClient(timeout=1800, trust_env=False) as client:
        deadline = time.monotonic() + 900
        pending = set(urls)
        while pending:
            for url in list(pending):
                try:
                    response = await client.get(
                        url.removesuffix("/v1") + "/health", timeout=3,
                    )
                except (httpx.ConnectError, httpx.TimeoutException):
                    continue
                if response.status_code == 200:
                    pending.remove(url)
            if pending:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Engineer replicas not ready: {sorted(pending)}")
                await asyncio.sleep(3)

        scoring_tasks: dict[tuple[str, str], asyncio.Task[dict]] = {}

        async def score(result: dict, row: dict) -> dict:
            async def request() -> dict:
                async with scoring_sem:
                    return await reward_func(reward_args, SimpleNamespace(
                        response=result["response"], metadata=row["metadata"],
                    ))
            key = result["patch_sha256"], digest(row["metadata"])
            if key not in scoring_tasks:
                scoring_tasks[key] = asyncio.create_task(transport_retry(
                    request, output=args.output, identifier=result["id"], operation="target_rerun",
                ))
            return await scoring_tasks[key]

        prompt_lengths: dict[int, asyncio.Task[int]] = {}

        async def count_prompt(packet_index: int, identifier: str, row: dict) -> int:
            tokenized = await transport_retry(lambda: client.post(
                urls[packet_index % len(urls)].removesuffix("/v1") + "/tokenize", json={
                    "model": "policy", "messages": row["prompt"],
                    "add_generation_prompt": True,
                    "chat_template_kwargs": {"enable_thinking": True},
                }), output=args.output, identifier=identifier, operation="engineer_tokenize",
            )
            tokenized.raise_for_status()
            return tokenized.json()["count"]

        async def generate(packet_index: int, sample_index: int) -> dict:
            absolute_index = args.packet_start + packet_index
            identifier = f"packet-{absolute_index:06d}-sample-{sample_index:02d}"
            result_path = args.output / "samples" / f"{identifier}.json"
            row = rows[packet_index]
            if result_path.exists():
                retained = json.loads(result_path.read_text())
                if retained.get("request_digest") != digest(row):
                    raise ValueError(f"Bootstrap retained prompt identity mismatch: {identifier}")
                if retained.get("status") != "generated" or args.generate_only:
                    completed.append(retained)
                    return retained
                reward = await score(retained, row)
                if not isinstance(reward, dict) or reward.get("eval_status") != "ok":
                    raise RuntimeError(f"Retained candidate rerun failed: {reward}")
                if type(reward.get("hook_errors")) is not int:
                    raise RuntimeError("Reward rerun did not report hook execution errors")
                retained.update(status="scored", reward=reward,
                                clean_execution=reward["hook_errors"] == 0)
                write_json(result_path, retained)
                completed.append(retained)
                return retained
            seed = args.seed + absolute_index * 1009 + sample_index
            url = urls[seed % len(urls)]
            async with generation_sem:
                if packet_index not in prompt_lengths:
                    prompt_lengths[packet_index] = asyncio.create_task(
                        count_prompt(packet_index, identifier, row),
                    )
                length = await prompt_lengths[packet_index]
                result = {
                    "id": identifier, "request_digest": digest(row),
                    "packet_index": packet_index, "sample_index": sample_index,
                    "seed": seed, "prompt_tokens": length,
                }
                if length > args.max_prompt_tokens:
                    result.update(status="prompt_over_budget", valid_patch=False)
                else:
                    response = await transport_retry(lambda: client.post(url + "/chat/completions", json={
                        "model": "policy", "messages": row["prompt"],
                        "chat_template_kwargs": {"enable_thinking": True},
                        "max_tokens": args.max_response_tokens,
                        "temperature": 0.7, "top_p": 0.95, "top_k": -1,
                        "seed": seed,
                    }), output=args.output, identifier=identifier, operation="engineer_generate")
                    response.raise_for_status()
                    body = response.json()
                    choice = body["choices"][0]
                    text = choice["message"].get("content")
                    if not isinstance(text, str):
                        raise RuntimeError("Engineer response content is not text")
                    result.update(
                        response=text, finish_reason=choice.get("finish_reason"),
                        usage=body["usage"],
                    )
                    if choice.get("finish_reason") != "stop":
                        result.update(status="response_truncated", valid_patch=False)
                    else:
                        try:
                            patch = extract_and_normalize_patch(text)
                        except PatchValidationError as exc:
                            result.update(
                                status="invalid_patch", valid_patch=False,
                                validation_error=str(exc),
                            )
                        else:
                            result.update(
                                status="generated", valid_patch=True,
                                patch_sha256=patch_digest(patch),
                            )
            if result.get("valid_patch") and not args.generate_only:
                write_json(result_path, result)
                reward = await score(result, row)
                if not isinstance(reward, dict) or reward.get("eval_status") != "ok":
                    raise RuntimeError(f"Validated patch did not receive a complete rerun: {reward}")
                if type(reward.get("hook_errors")) is not int:
                    raise RuntimeError("Reward rerun did not report hook execution errors")
                result.update(status="scored", reward=reward,
                              clean_execution=reward["hook_errors"] == 0)
            write_json(result_path, result)
            completed.append(result)
            positive = sum(
                float(item.get("reward", {}).get("score", 0)) > 0
                and item.get("clean_execution") is True
                for item in completed
            )
            write_json(args.output / "STATUS.json", {
                "stage": "fresh_base_silver_collection",
                "completed": len(completed), "planned": len(rows) * args.samples,
                "valid": sum(bool(item.get("valid_patch")) for item in completed),
                "clean_execution": sum(
                    item.get("clean_execution") is True for item in completed
                ),
                "positive": positive, "elapsed_seconds": time.monotonic() - started,
            })
            return result

        jobs = [
            asyncio.create_task(generate(packet_index, sample_index))
            for packet_index in range(len(rows))
            for sample_index in range(args.samples)
        ]
        try:
            results = await asyncio.gather(*jobs)
        finally:
            for job in jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*jobs, return_exceptions=True)
    if args.generate_only:
        write_json(args.output / "GENERATIONS_COMPLETE.json", {
            "stage": "fresh_base_generation_only", "generations": len(results),
            "input_identity": request_identity,
            "reruns_completed": False, "sft_dataset_ready": False,
            "elapsed_seconds": time.monotonic() - started,
        })
        return
    for row in rows:
        _validate_metadata(reward_args, row["metadata"])
    sft_rows = []
    for result in results:
        if (float(result.get("reward", {}).get("score", 0)) <= 0
                or result.get("clean_execution") is not True):
            continue
        row = rows[result["packet_index"]]
        sft_rows.append({
            "messages": [
                *row["prompt"],
                {"role": "assistant", "content": result["response"]},
            ],
            "provenance": {
                "generation_id": result["id"],
                "engineer_manifest_sha256": request_identity["engineer_manifest_sha256"],
                "task_ids": row["metadata"]["task_ids"],
                "reward_delta": result["reward"]["score"],
                "patch_sha256": result["patch_sha256"],
                "split": "train",
            },
        })
    dataset_path = args.output / "sft.jsonl"
    temporary = dataset_path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in sft_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(dataset_path)
    write_json(args.output / "COMPLETE.json", {
        "stage": "fresh_base_silver_collection",
        "examples": len(sft_rows), "planned_generations": len(results),
        "usable_for_sft": len(sft_rows) >= args.minimum_examples,
        "minimum_examples": args.minimum_examples,
        "reason": (
            "minimum_positive_examples_met"
            if len(sft_rows) >= args.minimum_examples
            else "additional_train_packets_needed"
        ),
        "dataset_sha256": file_sha(dataset_path),
        "input_identity": request_identity,
        "elapsed_seconds": time.monotonic() - started,
        "development_or_test_gain_claimed": False,
        "current_invocation_tokenizations": len(prompt_lengths),
        "current_invocation_unique_target_scores": len(scoring_tasks),
    })


if __name__ == "__main__":
    asyncio.run(run(arguments()))
