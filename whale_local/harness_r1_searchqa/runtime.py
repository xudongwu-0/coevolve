"""Per-episode lifecycle for SearchQA Harness-R1 code hooks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .patch import compile_patch, run_hook


MAX_OBSERVATION_CHARS = 6000


@dataclass(frozen=True)
class BeforeQuery:
    query: str
    blocked: bool
    message: str
    effect: dict[str, Any] | None


class SearchQAHarnessRuntime:
    def __init__(
        self,
        patch: dict[str, Any],
        *,
        question: str,
        max_steps: int,
        max_tool_calls: int,
        top_k: int,
        document_char_limit: int,
    ):
        self.compiled = compile_patch(patch)
        self.question = question
        self.settings = {
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
            "top_k": top_k,
            "document_char_limit": document_char_limit,
        }
        self.trace: list[dict[str, Any]] = self.compiled["trace"]
        self.forced_next_query = ""

    @property
    def notebook(self) -> dict[str, Any]:
        return self.compiled["notebook"]

    def _context(
        self,
        *,
        step: int,
        remaining_steps: int,
        previous_queries: list[str],
        last_observation: str,
    ) -> dict[str, Any]:
        return {
            "benchmark": "searchqa",
            "question": self.question,
            "step": step,
            "remaining_steps": remaining_steps,
            "previous_queries": list(previous_queries),
            "last_observation": last_observation[:MAX_OBSERVATION_CHARS],
            "task": dict(self.settings),
            "admissible": [],
        }

    def initialize(self) -> tuple[list[str], str]:
        ctx = self._context(
            step=0,
            remaining_steps=self.settings["max_steps"],
            previous_queries=[],
            last_observation="",
        )
        effect = run_hook(self.compiled, "on_init", ctx)
        if effect is None:
            return [], ""
        skills = [
            item["text"]
            for item in effect.get("skills", [])
            if isinstance(item, dict) and item.get("text")
        ]
        tool_hint = str(effect.get("tool_hint") or "")
        return skills, tool_hint

    def pre_hint(
        self,
        *,
        step: int,
        remaining_steps: int,
        previous_queries: list[str],
        last_observation: str,
    ) -> str:
        ctx = self._context(
            step=step,
            remaining_steps=remaining_steps,
            previous_queries=previous_queries,
            last_observation=last_observation,
        )
        effect = run_hook(self.compiled, "make_pre_hint", ctx)
        if effect is None:
            return ""
        return str(effect.get("message") or "")

    def before_query(
        self,
        query: str,
        *,
        step: int,
        remaining_steps: int,
        previous_queries: list[str],
        last_observation: str,
    ) -> BeforeQuery:
        current = self.forced_next_query or query
        self.forced_next_query = ""
        ctx = self._context(
            step=step,
            remaining_steps=remaining_steps,
            previous_queries=previous_queries,
            last_observation=last_observation,
        )
        ctx["action"] = {"tool": "search", "value": current}
        ctx["raw_action"] = query
        effect = run_hook(self.compiled, "on_before_action", ctx)
        if "on_before_action" in self.compiled["hooks"]:
            self.trace[-1].update(raw_query=query, final_query=current)
        if effect is None:
            return BeforeQuery(current, False, "", None)
        kind = effect.get("kind")
        if kind in {"rewrite_action", "force_action"}:
            current = str(effect.get("action") or current).strip()
        blocked = kind == "block_and_prompt"
        message = str(effect.get("message") or "")
        self.trace[-1].update(raw_query=query, final_query=current)
        return BeforeQuery(current, blocked, message, effect)

    def post_step(
        self,
        *,
        query: str,
        observation: str,
        step: int,
        remaining_steps: int,
        previous_queries: list[str],
    ) -> str:
        ctx = self._context(
            step=step,
            remaining_steps=remaining_steps,
            previous_queries=previous_queries,
            last_observation=observation,
        )
        ctx["action"] = {"tool": "search", "value": query}
        ctx["observation"] = observation[:MAX_OBSERVATION_CHARS]
        effect = run_hook(self.compiled, "on_post_step", ctx)
        if effect is None:
            return ""
        if effect.get("kind") == "force_action":
            self.forced_next_query = str(effect.get("action") or "").strip()
        return str(effect.get("message") or "")
