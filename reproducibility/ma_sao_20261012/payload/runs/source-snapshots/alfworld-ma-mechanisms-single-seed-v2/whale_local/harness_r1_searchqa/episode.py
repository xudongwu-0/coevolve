"""Frozen SearchQA episode with Harness-R1 lifecycle hooks."""
from __future__ import annotations

import asyncio
import copy
import json
from statistics import mean
from typing import Any, Callable

from sao_bridge.evaluation import Sampler
from sao_bridge.retool import ContextExhausted
from sao_bridge.searchqa_domain import (
    MAX_TOOL_OUTPUT_CHARS,
    TOOLS,
    alias_exact_match,
    extract_answer,
    retrieve_documents,
)
from sao_bridge.searchqa_retrieval import checked_query

from .patch import base_patch
from .runtime import SearchQAHarnessRuntime


H0_CONFIG = {
    "system_prompt": "Answer factual questions using Wikipedia search when needed.",
    "max_steps": 2,
    "max_tool_calls": 1,
    "top_k": 1,
    "max_query_chars": 512,
    "document_char_limit": 800,
}


def _tool_message(call_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(payload, ensure_ascii=False),
    }


async def evaluate_one(
    row: dict[str, Any],
    *,
    sampler: Sampler,
    patch: dict[str, Any] | None,
    total_tokens: int,
    rollout_seed: int,
) -> dict[str, Any]:
    if type(total_tokens) is not int or total_tokens <= 0:
        raise ValueError("total_tokens must be a positive integer")
    question = str(row["question"])
    answers = [str(value) for value in row["answers"]]
    config = dict(H0_CONFIG)
    runtime = SearchQAHarnessRuntime(
        patch or base_patch(),
        question=question,
        max_steps=config["max_steps"],
        max_tool_calls=config["max_tool_calls"],
        top_k=config["top_k"],
        document_char_limit=config["document_char_limit"],
    )
    skills, tool_hint = runtime.initialize()
    system_prompt = config["system_prompt"]
    if skills:
        system_prompt += "\n\nSome tips that may help for this task:\n" + "\n".join(
            f"- {text}" for text in skills
        )
    tools = copy.deepcopy(TOOLS)
    if tool_hint:
        tools[0]["function"]["description"] += "\nHarness guidance: " + tool_hint
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Question: "
                + question
                + "\nUse the search function when needed. "
                + "Put your final answer only inside <answer>...</answer>."
            ),
        },
    ]
    records: list[dict[str, Any]] = []
    queries: list[str] = []
    calls: list[dict[str, Any]] = []
    last_text = ""
    last_observation = ""
    remaining = total_tokens
    tool_count = 0
    tool_errors = 0
    output_chars = 0
    exhausted = False
    has_hooks = bool(runtime.compiled["hooks"])

    for turn in range(config["max_steps"]):
        pre_hint = runtime.pre_hint(
            step=turn,
            remaining_steps=config["max_steps"] - turn,
            previous_queries=queries,
            last_observation=last_observation,
        )
        if pre_hint:
            messages.append({"role": "user", "content": pre_hint})
        cap = (
            remaining
            if turn == config["max_steps"] - 1
            else min(remaining, max(128, total_tokens // 2))
        )
        try:
            completion = await sampler(messages, tools, cap, rollout_seed + turn)
        except ContextExhausted:
            exhausted = True
            break
        used = completion["usage"]["completion_tokens"]
        if type(used) is not int or not 0 < used <= cap:
            raise RuntimeError("Invalid generation token accounting")
        remaining -= used
        if "record" in completion:
            records.append(completion["record"])
        message = dict(completion["choices"][0]["message"])
        last_text = message.get("content") or ""
        if not isinstance(last_text, str):
            raise RuntimeError("Assistant response content must be text")
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if not isinstance(tool_calls, list) or len(tool_calls) > 32:
            raise RuntimeError("Invalid assistant tool-call list")
        post_hints: list[str] = []

        if tool_calls:
            for tool in tool_calls:
                tool_count += 1
                call_id = tool.get("id") if isinstance(tool, dict) else None
                if not isinstance(call_id, str) or not call_id:
                    raise RuntimeError("Assistant tool call missing id")
                try:
                    if tool_count > config["max_tool_calls"]:
                        raise ValueError("Episode search-call limit reached")
                    if output_chars >= MAX_TOOL_OUTPUT_CHARS:
                        raise ValueError("Episode retrieval-output budget reached")
                    function = tool["function"]
                    arguments = json.loads(function["arguments"])
                    if (
                        function["name"] != "search"
                        or not isinstance(arguments, dict)
                        or set(arguments) != {"query"}
                    ):
                        raise ValueError("Use search with exactly one query string")
                    raw_query = checked_query(arguments["query"])
                    if len(raw_query) > config["max_query_chars"]:
                        raise ValueError("Query exceeds harness query-character limit")
                    before = runtime.before_query(
                        raw_query,
                        step=turn,
                        remaining_steps=config["max_steps"] - turn,
                        previous_queries=queries,
                        last_observation=last_observation,
                    )
                    if before.blocked:
                        raise ValueError(
                            before.message or "Harness blocked this search query"
                        )
                    query = checked_query(before.query)
                    if len(query) > config["max_query_chars"]:
                        raise ValueError("Harness query exceeds query-character limit")
                    documents = await retrieve_documents(query, config["top_k"])
                    documents = [
                        {
                            **document,
                            "text": document["text"][: config["document_char_limit"]],
                        }
                        for document in documents
                    ]
                    queries.append(query)
                    available = MAX_TOOL_OUTPUT_CHARS - output_chars
                    bounded = []
                    for document in documents:
                        if available <= 0:
                            break
                        text = document["text"][:available]
                        bounded.append({**document, "text": text})
                        available -= len(text)
                        output_chars += len(text)
                    result = {"query": query, "documents": bounded, "error": False}
                    observation = json.dumps(result, ensure_ascii=False)
                    last_observation = observation
                    calls.append(
                        {
                            "raw_query": raw_query,
                            "query": query,
                            "documents": bounded,
                        }
                    )
                    hint = runtime.post_step(
                        query=query,
                        observation=observation,
                        step=turn,
                        remaining_steps=config["max_steps"] - turn - 1,
                        previous_queries=queries,
                    )
                    if hint:
                        post_hints.append(hint)
                except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    tool_errors += 1
                    result = {
                        "error": True,
                        "stage": "tool_or_harness",
                        "message": f"{type(exc).__name__}: {str(exc)[:500]}",
                    }
                messages.append(_tool_message(call_id, result))
            for hint in post_hints:
                messages.append({"role": "user", "content": hint})
        elif extract_answer(last_text) is not None:
            break
        elif not has_hooks:
            break

        if remaining <= 0:
            exhausted = True
            break

    valid_format = extract_answer(last_text) is not None
    reward = alias_exact_match(last_text, answers)
    return {
        "task_id": str(row["id"]),
        "question": question,
        "reward": reward,
        "messages": messages,
        "calls": calls,
        "records": records,
        "tokens": total_tokens - remaining,
        "truncated": exhausted or remaining == 0 or not valid_format,
        "boxed": valid_format,
        "formatted": valid_format,
        "tool_calls": tool_count,
        "tool_errors": tool_errors,
        "rollout_seed": rollout_seed,
        "harness_trace": runtime.trace,
        "notebook": runtime.notebook,
    }


async def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    sampler: Sampler,
    patch: dict[str, Any] | None,
    total_tokens: int,
    rollout_seeds: list[int],
    concurrency: int,
    on_result: Callable[[int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if not rows or type(concurrency) is not int or concurrency <= 0:
        raise ValueError("evaluation requires nonempty rows and positive concurrency")
    if len(rows) != len(rollout_seeds):
        raise ValueError("rollout_seeds must align exactly with rows")
    semaphore = asyncio.Semaphore(concurrency)

    async def run(index: int) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            result = await evaluate_one(
                rows[index],
                sampler=sampler,
                patch=patch,
                total_tokens=total_tokens,
                rollout_seed=rollout_seeds[index],
            )
            if on_result is not None:
                on_result(index, result)
            return index, result

    indexed = await asyncio.gather(*(run(index) for index in range(len(rows))))
    results = [item for _, item in sorted(indexed)]
    return {
        "mean": mean(float(item["reward"]) for item in results),
        "count": len(results),
        "task_ids": [str(row["id"]) for row in rows],
        "rollout_seeds": list(rollout_seeds),
        "results": results,
    }
