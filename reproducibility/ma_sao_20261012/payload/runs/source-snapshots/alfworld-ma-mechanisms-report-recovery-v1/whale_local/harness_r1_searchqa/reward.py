"""Relax custom reward for cumulative SearchQA Harness-R1 patches."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any
from urllib.parse import urlsplit

from sao_bridge.evaluation import Sampler
from sao_bridge.searchqa_prepare import ASSETS
from sao_bridge.searchqa_prepare import sha256

from .episode import evaluate_rows
from .manifest import load_manifest_splits
from .identity import baseline_config
from .patch import (
    PatchValidationError,
    extract_and_normalize_patch,
    merge_patches,
    patch_digest,
)


REWARD_PROTOCOL = "searchqa-cumulative-same-task-delta-v1"
PROMPT_VERSION = "searchqa-harness-r1-failure-packet-v1"
def _arg(args: Any, name: str, default: Any) -> Any:
    value = getattr(args, name, default)
    return default if value is None else value


def _rows_by_id(data_root: str) -> dict[str, dict[str, Any]]:
    root = str(Path(data_root).resolve())
    splits = load_manifest_splits(Path(root), splits=("train",))
    rows = {str(row["id"]): row for row in splits["train"]}
    return rows


def _no_patch(metadata: dict[str, Any], reason: str, detail: str) -> dict[str, Any]:
    baseline = _baseline_rewards(metadata)
    baseline_mean = sum(baseline.values()) / len(baseline) if baseline else 0.0
    return {
        "score": 0.0,
        "delta_score": 0.0,
        "baseline_average_reward": baseline_mean,
        "patched_average_reward": baseline_mean,
        "valid_patch": False,
        "eval_status": reason,
        "detail": detail[:1000],
        "reward_protocol": REWARD_PROTOCOL,
        **_hook_stats([]),
    }


def _hook_stats(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Expose candidate failures without changing the same-task delta reward."""
    counts = {"ok": 0, "candidate_error": 0, "infrastructure_error": 0}
    errors = []
    for result in results:
        trace = result.get("harness_trace", [])
        if not isinstance(trace, list):
            raise RuntimeError("patched evaluation contains malformed harness trace")
        for event in trace:
            if (not isinstance(event, dict) or type(event.get("status")) is not str
                    or event["status"] not in counts):
                raise RuntimeError("patched evaluation contains unknown hook execution status")
            status = event["status"]
            if status == "infrastructure_error":
                raise RuntimeError("patched evaluation contains hook infrastructure failure")
            counts[status] += 1
            if status == "candidate_error":
                errors.append({"task_id": result["task_id"], **event})
    return {
        "hook_calls": sum(counts.values()),
        "hook_errors": counts["candidate_error"],
        "hook_status_counts": counts,
        "hook_runtime_errors": errors,
    }


def _baseline_rewards(metadata: dict[str, Any]) -> dict[str, float]:
    raw = metadata.get("baseline_rewards")
    if not isinstance(raw, dict):
        raise ValueError("metadata.baseline_rewards must be an object")
    rewards: dict[str, float] = {}
    for key, value in raw.items():
        if type(key) is not str or type(value) not in (int, float) or value not in {0.0, 1.0}:
            raise ValueError("SearchQA baseline rewards must be binary")
        rewards[key] = float(value)
    return rewards


def _validate_metadata(args: Any, metadata: Any) -> tuple[
    list[str],
    list[int],
    dict[str, float],
    dict[str, Any],
]:
    if not isinstance(metadata, dict):
        raise ValueError("sample metadata is missing")
    if metadata.get("benchmark") != "searchqa":
        raise ValueError("reward sample benchmark must be searchqa")
    if metadata.get("reward_protocol") != REWARD_PROTOCOL:
        raise ValueError("reward protocol mismatch")
    if metadata.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("prompt version mismatch")
    task_ids = metadata.get("task_ids")
    seeds = metadata.get("rollout_seeds")
    if (
        not isinstance(task_ids, list)
        or not isinstance(seeds, list)
        or not 1 <= len(task_ids) <= 16
        or len(task_ids) != len(seeds)
    ):
        raise ValueError("metadata task_ids and rollout_seeds must be aligned lists")
    if any(type(value) is not str or not value for value in task_ids):
        raise ValueError("metadata task ids must be nonempty strings")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("metadata task_ids must be unique")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("metadata rollout seeds must be nonnegative integers")
    baseline = _baseline_rewards(metadata)
    if set(baseline) != set(task_ids):
        raise ValueError("baseline reward identities do not match task_ids")
    incumbent = metadata.get("incumbent_patch")
    if not isinstance(incumbent, dict):
        raise ValueError("metadata incumbent_patch is missing")
    if patch_digest(incumbent) != metadata.get("incumbent_patch_sha256"):
        raise ValueError("incumbent patch digest mismatch")
    expected_target = str(_arg(args, "harness_r1_target_model_sha256", ""))
    if not expected_target or metadata.get("target_model_sha256") != expected_target:
        raise ValueError("target model identity mismatch")
    data_root = str(_arg(args, "harness_r1_searchqa_data", ""))
    manifest = Path(data_root) / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"SearchQA manifest is missing: {manifest}")
    if metadata.get("data_manifest_sha256") != sha256(manifest):
        raise ValueError("SearchQA data manifest identity mismatch")
    if metadata.get("split") != "train":
        raise ValueError("reward provenance must declare split=train")
    if (type(metadata.get("batch_tag")) is not str
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", metadata["batch_tag"])):
        raise ValueError("metadata batch_tag must be a safe single path component")
    if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("source_baseline_sha256", ""))):
        raise ValueError("source baseline digest is missing/invalid")
    row_map = _rows_by_id(data_root)
    if any(task_id not in row_map for task_id in task_ids):
        raise ValueError("reward metadata contains non-train task ids")
    retrieval_manifest = Path(data_root) / "retrieval_manifest.json"
    if sha256(retrieval_manifest) != sha256(ASSETS / "retrieval_manifest.json"):
        raise ValueError("configured retriever differs from episode's frozen retriever")
    expected_config = baseline_config(
        target_model_sha256=expected_target,
        target_temperature=_arg(args, "harness_r1_target_temperature", 0.3),
        total_tokens=_arg(args, "harness_r1_rollout_max_tokens", 4096),
        retrieval_manifest=retrieval_manifest, data_manifest=manifest, incumbent_patch=incumbent)
    if metadata.get("baseline_config") != expected_config:
        raise ValueError("baseline configuration/provenance mismatch; regenerate baseline")
    _serving_config(args)
    return task_ids, list(seeds), baseline, incumbent


def _serving_config(args: Any) -> dict[str, Any]:
    retrieval_url = str(_arg(args, "harness_r1_retrieval_url", "")).strip()
    targets = _arg(args, "harness_r1_target_urls", None)
    target_url = (",".join(str(value) for value in targets) if isinstance(targets, (list, tuple))
                  else str(targets or _arg(args, "harness_r1_target_url", "")))
    if not retrieval_url or not target_url:
        raise ValueError("Frozen target and retrieval serving URLs are required")
    for url in [retrieval_url, *target_url.split(",")]:
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Invalid frozen serving URL")
    retrieval_path = urlsplit(retrieval_url)
    if retrieval_path.scheme != "http" or retrieval_path.path not in ("", "/", "/retrieve"):
        raise ValueError("Invalid frozen retrieval service URL")
    eval_root = str(_arg(args, "harness_r1_eval_root", ""))
    if not eval_root or Path(eval_root).resolve() == Path("/"):
        raise ValueError("harness_r1_eval_root must be an explicit non-root path")
    configured = os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL")
    if configured and configured != retrieval_url:
        raise ValueError("conflicting SearchQA retrieval URL in reward worker")
    concurrency = _arg(args, "harness_r1_rollout_concurrency", 8)
    timeout = _arg(args, "harness_r1_rollout_http_timeout", 900)
    if (type(concurrency) is not int or concurrency < 1
            or type(timeout) not in (int, float) or not 0 < timeout < float("inf")):
        raise ValueError("Invalid rollout concurrency/HTTP timeout")
    return {
        "target_url": target_url, "retrieval_url": retrieval_url,
        "temperature": _arg(args, "harness_r1_target_temperature", 0.3),
        "max_tokens": _arg(args, "harness_r1_rollout_max_tokens", 4096),
        "concurrency": _arg(args, "harness_r1_rollout_concurrency", 8),
        "http_timeout": _arg(args, "harness_r1_rollout_http_timeout", 900),
        "response_protocol": _arg(args, "harness_r1_response_protocol", "prefill_think_patch"),
        "worker_settings": {key: value for key, value in vars(args).items()
                            if key.startswith("harness_r1_") or key == "reward_max_concurrency"},
    }


def _cache_key(
    metadata: dict[str, Any],
    merged_patch: dict[str, Any],
    task_ids: list[str],
    seeds: list[int],
    serving: dict[str, Any] | None = None,
) -> str:
    payload = {
        "protocol": REWARD_PROTOCOL,
        "batch_tag": metadata.get("batch_tag"),
        "tasks": task_ids,
        "seeds": seeds,
        "target": metadata.get("target_model_sha256"),
        "incumbent": metadata.get("incumbent_patch_sha256"),
        "candidate": patch_digest(merged_patch),
        "metadata": metadata,
        "serving": serving,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _read_cache(path: Path, identity: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("reward_protocol") != REWARD_PROTOCOL
            or result.get("cache_identity") != identity
            or result.get("eval_status") != "ok" or result.get("valid_patch") is not True):
        raise ValueError(f"stale or malformed reward cache: {path}")
    return result


async def _acquire_lock(lock: Path, cache: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if cache.is_file():
            return False
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"reward cache lock timed out: {lock}")
            await asyncio.sleep(0.5)
            continue
        os.close(descriptor)
        return True


def _write_cache(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


async def _score_one(args: Any, sample: Any) -> dict[str, Any]:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_ids, seeds, baseline, incumbent = _validate_metadata(args, metadata)
    try:
        candidate = extract_and_normalize_patch(
            sample.response,
            str(_arg(args, "harness_r1_response_protocol", "prefill_think_patch")),
        )
        merged = merge_patches(incumbent, candidate)
    except PatchValidationError as exc:
        return _no_patch(
            metadata,
            "invalid_patch_treated_as_no_patch",
            str(exc),
        )

    cache_root = Path(str(_arg(args, "harness_r1_eval_root", ""))).resolve()
    if str(cache_root) == "/" or not str(_arg(args, "harness_r1_eval_root", "")):
        raise ValueError("harness_r1_eval_root must be an explicit non-root path")
    cache_dir = (cache_root / "searchqa" / metadata["batch_tag"]).resolve()
    if not cache_dir.is_relative_to(cache_root):
        raise ValueError("Reward cache directory escapes cache root")
    cache_dir.mkdir(parents=True, exist_ok=True)
    serving = _serving_config(args)
    key = _cache_key(metadata, merged, task_ids, seeds,
                     {**serving, "candidate_patch_sha256": patch_digest(candidate)})
    cache_path = cache_dir / f"{key}.json"
    cached = _read_cache(cache_path, key)
    if cached is not None:
        return cached
    lock_path = cache_dir / f"{key}.lock"
    acquired = await _acquire_lock(
        lock_path,
        cache_path,
        float(_arg(args, "harness_r1_cache_lock_timeout", 7200)),
    )
    if not acquired:
        cached = _read_cache(cache_path, key)
        if cached is None:
            raise RuntimeError("reward cache appeared without a readable result")
        return cached

    sampler: Sampler | None = None
    try:
        data_root = str(_arg(args, "harness_r1_searchqa_data", ""))
        row_map = _rows_by_id(data_root)
        missing = [task_id for task_id in task_ids if task_id not in row_map]
        if missing:
            raise ValueError(f"reward metadata contains non-train task ids: {missing[:3]}")
        retrieval_url = str(_arg(args, "harness_r1_retrieval_url", "")).strip()
        if not retrieval_url:
            raise ValueError("harness_r1_retrieval_url is required")
        configured_retrieval = os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL")
        if configured_retrieval and configured_retrieval != retrieval_url:
            raise ValueError("conflicting SearchQA retrieval URL in reward worker")
        os.environ["SAO_SEARCHQA_RETRIEVAL_URL"] = retrieval_url
        target_urls = _arg(args, "harness_r1_target_urls", None)
        if isinstance(target_urls, (list, tuple)):
            target_url = ",".join(str(value) for value in target_urls)
        else:
            target_url = str(
                target_urls or _arg(args, "harness_r1_target_url", "")
            )
        if not target_url:
            raise ValueError("Harness-R1 target URL is required")
        sampler = Sampler(
            target_url,
            concurrency=int(_arg(args, "harness_r1_rollout_concurrency", 8)),
            temperature=float(_arg(args, "harness_r1_target_temperature", 0.3)),
            chat_template_kwargs={"enable_thinking": False},
            request_timeout=float(_arg(args, "harness_r1_rollout_http_timeout", 900)),
        )
        report = await evaluate_rows(
            [row_map[task_id] for task_id in task_ids],
            sampler=sampler,
            patch=merged,
            total_tokens=int(_arg(args, "harness_r1_rollout_max_tokens", 4096)),
            rollout_seeds=seeds,
            concurrency=int(_arg(args, "harness_r1_rollout_concurrency", 8)),
        )
        results = report.get("results")
        if not isinstance(results, list) or len(results) != len(task_ids):
            raise RuntimeError("patched result count does not match requested tasks")
        requested_seeds = dict(zip(task_ids, seeds))
        for item in results:
            if (not isinstance(item, dict) or type(item.get("reward")) not in (int, float)
                    or item["reward"] not in (0, 1)
                    or type(item.get("task_id")) is not str
                    or item["task_id"] not in requested_seeds
                    or type(item.get("rollout_seed")) is not int
                    or item["rollout_seed"] != requested_seeds[item["task_id"]]):
                raise RuntimeError("patched evaluation contains invalid rewards/infrastructure failures")
        hook_stats = _hook_stats(results)
        patched = {
            str(result["task_id"]): float(result["reward"])
            for result in results
        }
        if set(patched) != set(task_ids):
            raise RuntimeError("patched reward identities do not match requested tasks")
        baseline_mean = sum(baseline.values()) / len(baseline)
        patched_mean = sum(patched.values()) / len(patched)
        delta = patched_mean - baseline_mean
        result = {
            "score": delta,
            "delta_score": delta,
            "baseline_average_reward": baseline_mean,
            "patched_average_reward": patched_mean,
            "baseline_rewards": baseline,
            "patched_rewards": patched,
            "batch_size": len(task_ids),
            "valid_patch": True,
            "eval_status": "ok",
            "reward_protocol": REWARD_PROTOCOL,
            "candidate_patch_sha256": patch_digest(candidate),
            "merged_patch_sha256": patch_digest(merged),
            "merged_patch": merged,
            "target_model_sha256": metadata["target_model_sha256"],
            "cache_identity": key,
            "baseline_config": metadata["baseline_config"],
            "serving_config": serving,
            "source_baseline_sha256": metadata["source_baseline_sha256"],
            **hook_stats,
        }
        _write_cache(cache_path, result)
        return result
    finally:
        if sampler is not None:
            await sampler.close()
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


async def reward_func(
    args: Any,
    samples: Any | list[Any],
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    del kwargs
    if not isinstance(samples, list):
        return await _score_one(args, samples)
    limit = max(1, int(_arg(args, "reward_max_concurrency", 4)))
    semaphore = asyncio.Semaphore(limit)

    async def run(sample: Any) -> dict[str, Any]:
        async with semaphore:
            return await _score_one(args, sample)

    return await asyncio.gather(*(run(sample) for sample in samples))
