"""Counterpart-refresh replay with authentic editor behavior probabilities."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from .common import read_json


def interaction_reward(evaluation: dict, old_reward: float, coefficient: float) -> tuple[float, float]:
    refreshed = float(evaluation["reward"])
    if (
        not math.isfinite(coefficient) or coefficient < 0
        or not math.isfinite(old_reward) or not -1 <= old_reward <= 1
        or not math.isfinite(refreshed) or not -1 <= refreshed <= 1
    ):
        raise ValueError("Invalid paired interaction reward or coefficient")
    gain = refreshed - old_reward
    return (refreshed + coefficient * gain if evaluation["valid"] else -1.0), gain


def replay_rows(directory: Path, count: int, editor_step_offset: int) -> list[dict]:
    if count < 0 or editor_step_offset < 1:
        raise ValueError("Invalid cross-phase replay budget")
    if count == 0:
        return []
    records = [(path, read_json(path)) for path in directory.glob("*.json")]
    # Receipt filenames are random UUIDs. Selection order must be stable and independent
    # of reward; use declared task identity plus the sampled action as a deterministic tie.
    records.sort(key=lambda item: hashlib.sha256(
        (str(item[1].get("id", "")) + "\0" + str(item[1].get("action_text", ""))).encode()
    ).hexdigest())
    rows = []
    for path, record in records:
        if record.get("role") != "editor" or record.get("replayed"):
            continue
        calls = record.get("calls")
        if not calls or len(calls) != 1:
            raise ValueError(f"Editor replay is missing an exact policy call: {path}")
        call = calls[0]
        validate_call(call)
        row = {
            "id": "replay-" + record["id"], "messages": record["input_messages"],
            "replay": {
                "call": call, "source": record["action_text"],
                "origin": str(path.resolve()), "old_reward": record["reward"],
                "editor_step_offset": editor_step_offset,
                "old_partner_version": record["partner_version"],
                "reference_source": record["harness_source"],
            },
        }
        if record.get("metadata") is not None:
            row["metadata"] = record["metadata"]
        rows.append(row)
        if len(rows) == count:
            break
    if len(rows) < count:
        raise ValueError(f"Need {count} authentic replay records, found {len(rows)}")
    return rows


def validate_call(call: dict) -> None:
    responses = call["response_ids"]
    logps = call["response_logprobs"]
    versions = call["versions"]
    if not call["prompt_ids"] or not responses or len(responses) != len(logps):
        raise ValueError("Replay token/logprob alignment mismatch")
    if len(versions) != len(responses) or any(not math.isfinite(p) for p in logps):
        raise ValueError("Invalid behavior versions or log probabilities")
    if any(type(token) is not int or token < 0 for token in call["prompt_ids"] + responses):
        raise ValueError("Invalid replay token IDs")
    if call["temperature"] <= 0:
        raise ValueError("Policy-gradient replay requires stochastic behavior")


def replay_interaction(row: dict, tokenizer, reward: float):
    from areal.api.io_struct import ModelResponse
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    replay = row["replay"]
    call = replay["call"]
    validate_call(call)
    offset = replay["editor_step_offset"]
    response = ModelResponse(
        input_tokens=call["prompt_ids"], output_tokens=call["response_ids"],
        output_logprobs=call["response_logprobs"],
        output_versions=[version - offset for version in call["versions"]],
        stop_reason=call["stop_reason"], tokenizer=tokenizer,
    )
    return {row["id"]: InteractionWithTokenLogpReward(
        model_response=response, reward=reward,
        messages=row["messages"], chat_template_type="concat",
    )}


def recycle_pool(directory: Path, context_tokens: int = 8192,
                 request_indices: set[int] | None = None,
                 namespace: str | None = None) -> list[dict]:
    """Solver episodes produced while scoring the partner's harnesses.

    They are authentic samples of the frozen phase-initial solver policy (version 0 of
    the next solver phase). Order is by content hash, never by reward.
    """
    episodes = []
    for path in sorted(directory.glob("*.json")):
        result = read_json(path)
        if request_indices is not None and result["request_index"] not in request_indices:
            continue
        source = result["harness_source"]
        for row in result["results"]:
            calls = row.get("calls") or []
            if not calls:
                raise ValueError(f"Partner evaluation lacks exact token receipts: {path}")
            for call in calls:
                validate_call({**call, "versions": [0] * len(call["response_ids"])})
                if len(call["prompt_ids"]) + len(call["response_ids"]) > context_tokens:
                    raise ValueError("Recycled turn exceeds the training context")
            reward = float(row["reward"])
            if reward not in {0.0, 1.0}:
                raise ValueError("Recycled solver reward must be the binary verifier outcome")
            identity = path.name + "\0" + source + "\0" + row["id"]
            if namespace is not None:
                identity = namespace + "\0" + identity
            key = hashlib.sha256(identity.encode()).hexdigest()
            episodes.append({"id": key, "question_id": row["id"], "reward": reward,
                             "calls": calls, "harness_sha256": path.stem})
    episodes.sort(key=lambda episode: episode["id"])
    return episodes


def recycled_interactions(episode: dict, tokenizer) -> dict:
    """One native SAO sequence per recorded turn, all credited with the episode outcome."""
    from areal.api.io_struct import ModelResponse
    from areal.experimental.openai.types import InteractionWithTokenLogpReward

    result = {}
    for index, call in enumerate(episode["calls"]):
        versions = [0] * len(call["response_ids"])
        validate_call({**call, "versions": versions})
        response = ModelResponse(
            input_tokens=call["prompt_ids"], output_tokens=call["response_ids"],
            output_logprobs=call["response_logprobs"], output_versions=versions,
            stop_reason=call["stop_reason"], tokenizer=tokenizer,
        )
        result[f"recycled-{episode['id']}-{index}"] = InteractionWithTokenLogpReward(
            model_response=response, reward=episode["reward"],
        )
    return result
