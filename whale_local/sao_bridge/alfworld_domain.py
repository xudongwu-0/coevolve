"""Official AgentBench ALFWorld environment adapter for native SAO rollouts."""
from __future__ import annotations

import json
import logging
import os
from urllib.parse import urlsplit

import httpx

from .alfworld_train_data import task_count

logger = logging.getLogger(__name__)

INITIAL_SOURCE = '{"benchmark":"alfworld","actions":[]}\n'
EDITOR_SYSTEM = "Use the official Harness-R1 ALFWorld failure-packet prompt supplied in each row."
TOOL_CALL_PARSER = "qwen"
REASONING_PARSER = ""
SOLVER_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False}}
EDITOR_ASSISTANT_PREFILL = "<think>\n"
EDITOR_EXTRA_BODY = {
    "chat_template_kwargs": {
        "enable_thinking": True,
        "add_generation_prompt": False,
        "continue_final_message": True,
    }
}
EVALUATOR_TIMEOUT = 4800


def harness_spec(source: str) -> dict:
    try:
        patch = json.loads(source)
    except json.JSONDecodeError as exc:
        raise ValueError("ALFWorld harness must be a JSON patch") from exc
    if not isinstance(patch, dict) or patch.get("benchmark") != "alfworld":
        raise ValueError("ALFWorld harness patch must declare benchmark=alfworld")
    actions = patch.get("actions")
    if not isinstance(actions, list):
        raise ValueError("ALFWorld harness patch actions must be a list")
    canonical = json.dumps(patch, sort_keys=True, separators=(",", ":")) + "\n"
    return {"source": canonical, "patch": patch}


def _controller_url() -> str:
    value = os.environ.get("SAO_ALFWORLD_CONTROLLER_URL", "").rstrip("/")
    parsed = urlsplit(value)
    if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in ("", "/api")):
        raise RuntimeError("SAO_ALFWORLD_CONTROLLER_URL must be a loopback HTTP /api endpoint")
    return value if parsed.path == "/api" else value + "/api"


def _running(status: object) -> bool:
    return str(status).rsplit(".", 1)[-1].upper() == "RUNNING"


def _terminal_reward(payload: dict) -> float:
    reward = payload.get("reward")
    if type(reward) in (int, float):
        return float(reward)
    score = (payload.get("metrics") or {}).get("score")
    if type(score) in (int, float):
        return float(score)
    raise RuntimeError("terminal ALFWorld response omitted numeric reward")


def _tool_argument_errors(calls: list[dict]) -> list[dict]:
    errors = []
    for index, call in enumerate(calls):
        arguments = call["function"]["arguments"]
        try:
            decoded = json.loads(arguments)
        except (json.JSONDecodeError, TypeError) as exc:
            errors.append({"tool_call_index": index, "error": str(exc)})
        else:
            if not isinstance(decoded, dict):
                errors.append({
                    "tool_call_index": index,
                    "error": "Tool arguments must encode a JSON object",
                })
    return errors


async def run_episode(row: dict, spec: dict, generate, total_tokens=4096, seed=42) -> dict:
    del spec
    task_id = row.get("task_id", row.get("index"))
    task_split = row.get("task_split", "test_set")
    count = task_count(
        task_split, row.get("task_catalogue_sha256"),
        catalogue_path=row.get("task_catalogue"),
    )
    if type(task_id) is not int or not 0 <= task_id < count:
        raise ValueError(
            f"ALFWorld row requires an integer task_id in 0..{count - 1} for {task_split}"
        )
    task_name = row.get("task_name") or os.environ.get("SAO_ALFWORLD_TASK_NAME", "")
    if not task_name:
        raise RuntimeError("SAO_ALFWORLD_TASK_NAME is required")
    base = _controller_url()
    timeout = float(os.environ.get("SAO_ALFWORLD_HTTP_TIMEOUT", "900"))
    session_id = None
    terminal = False
    messages: list[dict] = []
    tokens = 0
    tool_calls = 0
    tool_errors = 0
    turns = 0
    status = None

    def failed_episode(reason: str, *, truncated: bool) -> dict:
        return {
            "id": row.get("id", f"alfworld-{task_id}"),
            "task_id": task_id, "task_split": task_split,
            "reward": 0.0, "tokens": tokens,
            "tool_calls": tool_calls, "tool_errors": tool_errors,
            "truncated": truncated, "boxed": bool(tool_calls),
            "status": reason, "turns": turns, "messages": messages,
            "metrics": {
                "score": 0.0, "truncated": truncated, "stop_reason": reason,
            },
            "stop_reason": reason,
        }

    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        try:
            response = await client.post(
                base + "/start_sample", json={"name": task_name, "index": task_id},
            )
            response.raise_for_status()
            session_id = response.headers.get("session_id")
            if not session_id:
                raise RuntimeError("ALFWorld controller omitted session_id")
            payload = response.json()
            messages.extend(payload.get("messages") or [])
            tools = payload.get("tools") or []
            status = payload.get("status")
            while _running(status):
                remaining = total_tokens - tokens
                if remaining <= 0:
                    return failed_episode("token_limit", truncated=True)
                completion = await generate(messages, tools, remaining, seed + turns)
                choice = completion["choices"][0]
                assistant = choice["message"]
                used = int((completion.get("usage") or {}).get("completion_tokens") or 0)
                if used <= 0:
                    raise RuntimeError("ALFWorld solver response omitted token accounting")
                tokens += used
                turns += 1
                calls = assistant.get("tool_calls") or []
                tool_calls += len(calls)
                tool_errors += int(not calls)
                messages.append(assistant)
                argument_errors = _tool_argument_errors(calls)
                if argument_errors:
                    # Invalid JSON cannot round-trip through the serving chat template.
                    tool_errors += len(argument_errors)
                    logger.warning(
                        "ALFWorld task %s generated invalid tool arguments: %s",
                        task_id, argument_errors,
                    )
                    result = failed_episode(
                        "invalid_tool_arguments",
                        truncated=choice.get("finish_reason") == "length",
                    )
                    result["invalid_tool_arguments"] = argument_errors
                    return result
                response = await client.post(
                    base + "/interact",
                    headers={"session_id": session_id},
                    json={"messages": [assistant]},
                )
                response.raise_for_status()
                payload = response.json()
                messages.extend(payload.get("messages") or [])
                tools = payload.get("tools") or tools
                status = payload.get("status")
            terminal = True
            reward = _terminal_reward(payload)
            return {
                "id": row.get("id", f"alfworld-{task_id}"),
                "task_id": task_id, "task_split": task_split,
                "reward": reward, "tokens": tokens,
                "tool_calls": tool_calls, "tool_errors": tool_errors,
                "truncated": False, "boxed": bool(tool_calls),
                "status": status, "turns": turns, "messages": messages,
                "metrics": payload.get("metrics") or {},
            }
        finally:
            if session_id is not None and not terminal:
                try:
                    await client.post(
                        base + "/cancel", headers={"session_id": session_id}, json={},
                    )
                except httpx.HTTPError:
                    logger.warning(
                        "Could not cancel ALFWorld session %s", session_id,
                        exc_info=True,
                    )
