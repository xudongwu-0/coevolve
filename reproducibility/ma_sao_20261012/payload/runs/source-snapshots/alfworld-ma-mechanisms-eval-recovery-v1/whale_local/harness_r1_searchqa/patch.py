"""SearchQA patch protocol backed by the pinned Harness-R1 hook sandbox."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
from types import ModuleType
from typing import Any

from .sandbox import invoke, SandboxInfrastructureError


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REVISION = "94f2e087f573e1b82fc9568bb634d4ae9a887e28"
UPSTREAM_CODE_RUNNER = (
    ROOT
    / "third_party/harness-r1-upstream/code/life-harness/AgentBench"
    / "src/server/harness/code_runner.py"
)
HOOK_NAMES = frozenset({"on_init", "make_pre_hint", "on_before_action", "on_post_step"})
MAX_ACTIONS = 4
MAX_DESCRIPTION_CHARS = 1000


class PatchValidationError(ValueError):
    """The Engineer response is not a valid SearchQA Harness-R1 patch."""


def _load_code_runner() -> ModuleType:
    name = "_coevolve_harness_r1_code_runner"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    if not UPSTREAM_CODE_RUNNER.is_file():
        raise RuntimeError(f"Pinned Harness-R1 code runner is missing: {UPSTREAM_CODE_RUNNER}")
    spec = importlib.util.spec_from_file_location(name, UPSTREAM_CODE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the pinned Harness-R1 code runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CODE_RUNNER = _load_code_runner()
HookCompileError = CODE_RUNNER.HookCompileError


def _json_object(text: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise PatchValidationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def invalid_constant(value):
        raise PatchValidationError(f"nonfinite JSON constant: {value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            invalid_constant(value)
        return number
    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant,
                           parse_float=finite_float)
    except (ValueError, RecursionError) as exc:
        raise PatchValidationError(f"patch JSON is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise PatchValidationError("patch must be a JSON object")
    return value


def extract_patch(response: str, response_protocol: str = "prefill_think_patch") -> dict[str, Any]:
    if not isinstance(response, str) or not response.strip():
        raise PatchValidationError("Engineer response is empty")
    if len(response) > 100000 or len(response.encode("utf-8")) > 100000:
        raise PatchValidationError("Engineer response exceeds 100000 bytes")
    text = response.strip()
    if response_protocol == "json_patch":
        return _json_object(text)
    if response_protocol == "prefill_think_patch":
        if re.match(r"(?is)^<think\b", text):
            raise PatchValidationError("prefilled-think response must not open a second <think> block")
        text = "<think>\n" + text
    elif response_protocol != "full_think_patch":
        raise PatchValidationError(f"unsupported response protocol: {response_protocol!r}")
    envelope = re.fullmatch(r"(?is)<think>((?:(?!</?(?:think|patch)\b).)*)</think>\s*<patch>(.*?)</patch>", text)
    if envelope is None:
        raise PatchValidationError("expected exactly one complete think/patch envelope")
    return _json_object(envelope.group(2).strip())


def _clean_description(value: Any) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise PatchValidationError("patch.description must be a string")
    text = " ".join(value.strip().split())
    if len(text) > MAX_DESCRIPTION_CHARS:
        raise PatchValidationError(
            f"patch.description exceeds {MAX_DESCRIPTION_CHARS} characters"
        )
    return text


def normalize_patch(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PatchValidationError("patch must be a JSON object")
    if set(raw) - {"schema_version", "benchmark", "description", "actions"}:
        raise PatchValidationError("patch contains unknown fields")
    if "schema_version" in raw and raw["schema_version"] != "harness-r1-patch-v1":
        raise PatchValidationError("unsupported patch schema_version")
    if "description" in raw and not isinstance(raw["description"], str):
        raise PatchValidationError("patch.description must be a string")
    benchmark = raw.get("benchmark")
    if benchmark != "searchqa":
        raise PatchValidationError("patch benchmark must be 'searchqa'")
    actions = raw.get("actions")
    if not isinstance(actions, list) or not 1 <= len(actions) <= MAX_ACTIONS:
        raise PatchValidationError(f"patch.actions must contain 1..{MAX_ACTIONS} actions")
    normalized_actions = []
    seen_hooks: set[str] = set()
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            raise PatchValidationError(f"action {index} must be an object")
        if set(action) != {"type", "hook", "code"}:
            raise PatchValidationError(
                f"action {index} must contain exactly type, hook, and code"
            )
        if action["type"] != "add_code_hook":
            raise PatchValidationError("only add_code_hook actions are allowed")
        hook = action["hook"]
        if type(hook) is not str or hook not in HOOK_NAMES:
            raise PatchValidationError(f"unsupported code hook: {hook!r}")
        if hook in seen_hooks:
            raise PatchValidationError(f"duplicate code hook: {hook}")
        code = action["code"]
        if not isinstance(code, str):
            raise PatchValidationError(f"{hook} code must be a nonempty string")
        if len(code) > CODE_RUNNER.MAX_HOOK_SOURCE_CHARS:
            raise PatchValidationError(f"{hook} code exceeds official source limit")
        if not code.strip():
            raise PatchValidationError(f"{hook} code must be a nonempty string")
        result = invoke(code, runner_source=UPSTREAM_CODE_RUNNER.read_text(), mode="validate")
        if result["status"] != "ok":
            raise PatchValidationError(f"{hook} failed sandbox validation: {result['message']}")
        seen_hooks.add(hook)
        normalized_actions.append(
            {"type": "add_code_hook", "hook": hook, "code": code.strip()}
        )
    return {
        "schema_version": "harness-r1-patch-v1",
        "benchmark": "searchqa",
        "description": _clean_description(raw.get("description")),
        "actions": sorted(normalized_actions, key=lambda item: item["hook"]),
    }


def extract_and_normalize_patch(
    response: str, response_protocol: str = "prefill_think_patch"
) -> dict[str, Any]:
    return normalize_patch(extract_patch(response, response_protocol))


def base_patch() -> dict[str, Any]:
    return {
        "schema_version": "harness-r1-patch-v1",
        "benchmark": "searchqa",
        "description": "Initial harness with no lifecycle code hooks.",
        "actions": [],
    }


def _normalize_incumbent(patch: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(patch, dict):
        raise PatchValidationError("incumbent must be a JSON object")
    actions = patch.get("actions")
    if actions == []:
        if (set(patch) - {"schema_version", "benchmark", "description", "actions"}
                or patch.get("benchmark") != "searchqa"
                or ("description" in patch and not isinstance(patch["description"], str))
                or patch.get("schema_version", "harness-r1-patch-v1") != "harness-r1-patch-v1"):
            raise PatchValidationError("invalid empty incumbent schema/benchmark")
        return {**base_patch(), "description": _clean_description(patch.get("description"))}
    return normalize_patch(patch)


def merge_patches(
    incumbent: dict[str, Any] | None, candidate: dict[str, Any]
) -> dict[str, Any]:
    normalized_candidate = normalize_patch(candidate)
    if incumbent is None:
        incumbent = base_patch()
    normalized_incumbent = _normalize_incumbent(incumbent)
    by_hook = {
        action["hook"]: action for action in normalized_incumbent["actions"]
    }
    for action in normalized_candidate["actions"]:
        by_hook[action["hook"]] = action
    return {
        "schema_version": "harness-r1-patch-v1",
        "benchmark": "searchqa",
        "description": normalized_candidate["description"],
        "actions": [by_hook[name] for name in sorted(by_hook)],
    }


def compile_patch(patch: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_incumbent(patch)
    hooks = {
        action["hook"]: action["code"]
        for action in normalized["actions"]
    }
    return {"patch": normalized, "hooks": hooks, "notebook": {}, "trace": []}


def run_hook(compiled: dict[str, Any], hook: str, ctx: dict[str, Any]) -> dict[str, Any] | None:
    source = compiled["hooks"].get(hook)
    if source is None:
        return None
    event = {"hook": hook, "step": ctx.get("step"), "status": "pending", "effect": None}
    compiled["trace"].append(event)
    try:
        result = invoke(source, runner_source=UPSTREAM_CODE_RUNNER.read_text(),
                        hook=hook, ctx=ctx, notebook=compiled["notebook"])
    except SandboxInfrastructureError as exc:
        event.update(status="infrastructure_error", error=str(exc)[:1000])
        raise
    event["status"] = result["status"]
    if result["status"] == "candidate_error":
        event["error"] = result["message"]
        return None
    compiled["notebook"] = result["notebook"]
    event["effect"] = result["effect"]
    return result["effect"]


def patch_digest(patch: dict[str, Any]) -> str:
    normalized = _normalize_incumbent(patch)
    payload = json.dumps(
        normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()
