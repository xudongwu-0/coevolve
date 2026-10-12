"""Failure-packet construction for SearchQA Harness-R1 training."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sao_bridge.searchqa_prepare import sha256

from .patch import patch_digest
from .manifest import load_manifest_splits
from .identity import baseline_config
from .prompt import build_messages
from .reward import PROMPT_VERSION, REWARD_PROTOCOL


def load_train_rows(data_root: Path | str) -> list[dict[str, Any]]:
    """Verify/read only the frozen train split, never reward/dev/test files."""
    return load_manifest_splits(Path(data_root), splits=("train",))["train"]


def _bounded_text(value: Any, limit: int) -> str:
    if not isinstance(value, str):
        raise ValueError("trajectory text must be a string")
    return value[:limit]


def _public_message(message: Any) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ValueError("trajectory message must be an object")
    role = str(message.get("role") or "")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"unsupported trajectory role: {role!r}")
    limit = 2500 if role in {"assistant", "tool"} else 1800
    content = message.get("content")
    if content is None and role == "assistant" and message.get("tool_calls"):
        content = ""
    public: dict[str, Any] = {
        "role": role,
        "content": _bounded_text(content, limit),
    }
    if len(content) > limit:
        public["content_truncation"] = {"original_chars": len(content),
                                        "retained_chars": limit}
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        if any(not isinstance(call, dict) or not isinstance(call.get("function"), dict)
               for call in calls):
            raise ValueError("malformed trajectory tool call")
        public["tool_calls"] = [
            {
                "function": {
                    "name": _bounded_text((call.get("function") or {}).get("name"), 64),
                    "arguments": _bounded_text(
                        (call.get("function") or {}).get("arguments"), 700
                    ),
                }
            }
            for call in calls[:4]
        ]
        if len(calls) > 4:
            public["tool_calls_truncation"] = {"original_count": len(calls), "retained_count": 4}
        for original, retained in zip(calls, public["tool_calls"]):
            truncation = {key: {"original_chars": len(original["function"][key]),
                                "retained_chars": len(retained["function"][key])}
                          for key in ("name", "arguments")
                          if len(original["function"][key]) > len(retained["function"][key])}
            if truncation:
                retained["truncation"] = truncation
    elif calls is not None:
        raise ValueError("trajectory tool_calls must be a list")
    return public


def public_failure(result: dict[str, Any]) -> dict[str, Any]:
    if float(result.get("reward", 0.0)) != 0.0:
        raise ValueError("failure packets may contain only failed trajectories")
    question = result.get("question")
    messages = result.get("messages")
    if not isinstance(question, str) or not question:
        raise ValueError("failed trajectory is missing its public question")
    if not isinstance(messages, list) or not messages:
        raise ValueError("failed trajectory is missing public messages")
    return {
        "question": question[:4000],
        "question_truncated": len(question) > 4000,
        "trajectory_messages": [_public_message(message) for message in messages[:64]],
        "trajectory_bounds": {
            "original_messages": len(messages), "retained_messages": min(len(messages), 64),
            "messages_truncated": len(messages) > 64,
            "source_episode_truncated": bool(result.get("truncated", False)),
        },
        "tool_errors": int(result.get("tool_errors", 0)),
        "formatted_final_answer": bool(result.get("formatted", False)),
        "outcome": "failed",
    }


def build_rl_rows(
    report: dict[str, Any],
    *,
    incumbent_patch: dict[str, Any],
    target_model_sha256: str,
    data_manifest: Path,
    packet_size: int = 8,
    maximum_packets: int | None = None,
    prompt_token_limit: int | None = None,
    prompt_token_counter: Callable[[list[dict[str, str]]], int] | None = None,
) -> list[dict[str, Any]]:
    """Build explicit bounded packets; optional exact tokenizer budget rejects overflow.

    The counter must include the actual Engineer chat template and ``<think>``
    prefill. Reserve generation tokens when selecting the prompt limit. No token
    clipping is performed, and unconfigured token checking is recorded explicitly.
    """
    if type(packet_size) is not int or packet_size < 1 or packet_size > 16:
        raise ValueError("packet_size must be in 1..16")
    if maximum_packets is not None and (type(maximum_packets) is not int or maximum_packets < 1):
        raise ValueError("maximum_packets must be a positive integer")
    if ((prompt_token_limit is None) != (prompt_token_counter is None)
            or (prompt_token_limit is not None
                and (type(prompt_token_limit) is not int or prompt_token_limit < 1
                     or not callable(prompt_token_counter)))):
        raise ValueError("prompt token limit and exact callable counter must be supplied together")
    if not isinstance(report, dict) or report.get("split") != "train":
        raise ValueError("baseline report must declare split=train")
    results = report.get("results")
    if not isinstance(results, list):
        raise ValueError("baseline report results are missing")
    config = report.get("baseline_config")
    if not isinstance(config, dict):
        raise ValueError("baseline report is missing baseline_config provenance")
    expected_config = baseline_config(
        target_model_sha256=target_model_sha256,
        target_temperature=config.get("target_temperature"),
        total_tokens=config.get("total_tokens"),
        retrieval_manifest=data_manifest.parent / "retrieval_manifest.json",
        data_manifest=data_manifest, incumbent_patch=incumbent_patch)
    if config != expected_config:
        raise ValueError("baseline report configuration/provenance mismatch")
    train = {row["id"]: row for row in load_train_rows(data_manifest.parent)}
    seen = set()
    for result in results:
        if not isinstance(result, dict) or result.get("task_id") not in train:
            raise ValueError("baseline contains non-train task identity")
        identity = result["task_id"]
        if identity in seen or result.get("question") != train[identity]["question"]:
            raise ValueError("baseline duplicate task or question/source mismatch")
        if type(result.get("rollout_seed")) is not int or result["rollout_seed"] < 0:
            raise ValueError("baseline rollout seed must be a nonnegative integer")
        if type(result.get("reward")) not in (int, float) or result["reward"] not in (0, 1):
            raise ValueError("baseline reward must be binary")
        if any(event.get("status") == "infrastructure_error"
               for event in result.get("harness_trace", [])):
            raise ValueError("baseline contains infrastructure failure")
        seen.add(identity)
    if "task_ids" in report and report["task_ids"] != [result["task_id"] for result in results]:
        raise ValueError("baseline report task identities disagree")
    if "rollout_seeds" in report and report["rollout_seeds"] != [result["rollout_seed"] for result in results]:
        raise ValueError("baseline report seed identities disagree")
    failed = [result for result in results if float(result.get("reward", 0.0)) == 0.0]
    if len(failed) < packet_size:
        raise ValueError("not enough failures to form one complete packet")
    source_digest = hashlib.sha256(
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_digest = sha256(data_manifest)
    incumbent_digest = patch_digest(incumbent_patch)
    rows: list[dict[str, Any]] = []
    packet_count = len(failed) // packet_size
    if maximum_packets is not None:
        packet_count = min(packet_count, maximum_packets)
    for packet_index in range(packet_count):
        packet = failed[packet_index * packet_size : (packet_index + 1) * packet_size]
        task_ids = [str(item["task_id"]) for item in packet]
        seeds = [int(item["rollout_seed"]) for item in packet]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("failure packet contains duplicate task identities")
        failures = [public_failure(item) for item in packet]
        messages = build_messages(failures, incumbent_patch)
        prompt_tokens = prompt_token_counter(messages) if prompt_token_counter is not None else None
        if prompt_token_counter is not None:
            if type(prompt_tokens) is not int or prompt_tokens < 1:
                raise ValueError("prompt token counter must return a positive integer")
            if prompt_tokens > prompt_token_limit:
                raise ValueError(
                    f"packet {packet_index} prompt uses {prompt_tokens} tokens, exceeds "
                    f"budget {prompt_token_limit}; reduce packet size, never silently clip")
        batch_tag = f"packet-{packet_index:06d}-{source_digest[:12]}"
        rows.append(
            {
                "prompt": messages,
                "label": "",
                "metadata": {
                    "benchmark": "searchqa",
                    "split": "train",
                    "reward_protocol": REWARD_PROTOCOL,
                    "prompt_version": PROMPT_VERSION,
                    "batch_tag": batch_tag,
                    "task_ids": task_ids,
                    "rollout_seeds": seeds,
                    "baseline_rewards": {
                        task_id: float(item["reward"])
                        for task_id, item in zip(task_ids, packet)
                    },
                    "incumbent_patch": incumbent_patch,
                    "incumbent_patch_sha256": incumbent_digest,
                    "target_model_sha256": target_model_sha256,
                    "data_manifest_sha256": manifest_digest,
                    "source_baseline_sha256": source_digest,
                    "baseline_config": config,
                    "packet_selection": {
                        "source_results": len(results), "source_failures": len(failed),
                        "selected_failures": packet_count * packet_size,
                        "omitted_failures": len(failed) - packet_count * packet_size,
                    },
                    "prompt_bounds": {
                        "token_budget_checked": prompt_token_counter is not None,
                        "token_count": prompt_tokens,
                        "token_limit": prompt_token_limit,
                    },
                },
            }
        )
    return rows
