"""Static Engineer prompt and leakage-safe SearchQA failure packets."""
from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """You are a Harness Engineer. Diagnose recurring failures in a
batch of trajectories from one frozen SearchQA agent, then write one reusable
runtime patch. Return only a private reasoning continuation followed by one
closing </think> tag and one complete <patch> JSON block ending with </patch>.
The opening <think> is already prefilled: do not emit another opening <think>.
Never return bare JSON after </think>. Never encode task answers, task ids, or
one-off question rules. The target model and retriever remain frozen."""

OUTPUT_SHAPE_EXAMPLE = r"""A reusable hint should respect the fixed search budget and use available evidence.
</think><patch>{"schema_version":"harness-r1-patch-v1","benchmark":"searchqa","description":"Respect the fixed search budget and use available evidence.","actions":[{"type":"add_code_hook","hook":"make_pre_hint","code":"def hook(ctx, nb):\n    return {\"message\": \"Use available evidence; do not request another search after the search budget is exhausted.\"}"}]}</patch>"""

PATCH_SCHEMA = """REQUIRED OUTPUT SHAPE
Continue the already-open private reasoning, then emit exactly </think><patch>,
one valid JSON object, and </patch>. Both closing tags are mandatory. Do not
return bare JSON, reopen <think>, add a second patch, use Markdown/code fences,
or write anything after </patch>.

Literal valid output example (general guidance, not a task-specific answer):
""" + OUTPUT_SHAPE_EXAMPLE + """

The patch JSON structure is:
{
  "schema_version": "harness-r1-patch-v1",
  "benchmark": "searchqa",
  "description": "short reusable diagnosis",
  "actions": [
    {
      "type": "add_code_hook",
      "hook": "make_pre_hint",
      "code": "def hook(ctx, nb):\\n    ...\\n    return {...}"
    }
  ]
}

Use 1 to 4 actions with no duplicate hook. Each code string defines exactly one
top-level function named exactly def hook(ctx, nb):. The action's hook value
selects its lifecycle: on_init, make_pre_hint, on_before_action, or on_post_step.
It is NOT the Python function name: never write def on_post_step(...) or
def make_pre_hint(...). Each action contains ONLY the three keys type, hook,
code; do not add name, description, or other action keys. Use valid JSON with
escaped newlines/quotes in the code string. Do not write Markdown, nested
function definitions, or methods/classes. At most five small top-level helper
functions are allowed. Imports,
files, network, eval/exec, global state, while loops, dunder/private access, and
hard-coded answers are forbidden. nb is a persistent per-episode JSON dictionary
limited to 16384 serialized bytes, depth 32 and 10000 values. Hooks execute in
isolated, CPU/memory/wall-bounded processes. Runtime failures are traced and
roll back notebook changes; infrastructure failures abort evaluation.
Trajectory bounds/truncation fields explicitly identify omitted text or steps;
never treat a bounded packet as a complete trajectory.

SearchQA lifecycle:
- All hooks receive ctx.benchmark, ctx.question, ctx.step, ctx.remaining_steps,
  ctx.previous_queries (list of strings), ctx.last_observation (string),
  ctx.task (settings dictionary), and ctx.admissible (empty list).
  Fixed settings are max_steps=2, max_tool_calls=1, top_k=1,
  document_char_limit=800. remaining_steps is NOT remaining tool calls.
  Read the total search-call budget as ctx.task["max_tool_calls"] (always 1);
  ctx.task has no remaining_tool_calls field.
- on_init: step=0, previous_queries=[], last_observation="". Return optional
  {"skills":[{"text": TEXT}], "tool_hint": TEXT}.
- make_pre_hint: ctx has question, step, remaining_steps, previous_queries,
  last_observation, and task settings. Return optional {"message": TEXT}.
- on_before_action: ctx additionally has action.tool="search" and
  action.value=the proposed query, plus raw_action (original query string).
  Return optional
  {"kind":"rewrite_action","action":QUERY},
  {"kind":"force_action","action":QUERY}, or
  {"kind":"block_and_prompt","message":TEXT}.
- on_post_step: ctx.action.value is the executed query. ctx.observation and
  ctx.last_observation are strings containing the JSON-encoded public search
  observation, each bounded to 6000 characters; inspect them as text, not a
  documents dictionary. There is no ctx.documents or ctx.retrieved_documents
  field, and no reference answer. Never call .get(...) on ctx.last_observation
  or ctx.observation: they are strings, including "" before the first search,
  not dictionaries. Return optional
  {"kind":"inject_hint","message":TEXT} or
  {"kind":"force_action","action":QUERY}, effective only within the fixed budget.

Prefer general strategies: concise entity/relation queries, avoid repeating
failed queries, inspect retrieved evidence before answering, and explicitly
recover from empty or irrelevant results using already available evidence.
The episode allows only ONE search call in total. Once it is exhausted, do NOT
demand another search, retry, or broaden the query for an extra call. Hints and
force_action cannot increase this budget. Do not invent documents or bypass the
fixed BM25 retriever, tool budget, token budget, or answer verifier."""

FORBIDDEN_PACKET_KEYS = frozenset(
    {
        "answer",
        "answers",
        "ground_truth",
        "label",
        "reference_answer",
        "reward",
    }
)
FORBIDDEN_FAILURE_ID_KEYS = frozenset({"task_id", "question_id", "example_id"})


def _check_public(value: Any, path: str = "failure_packet") -> None:
    if isinstance(value, dict):
        forbidden = FORBIDDEN_PACKET_KEYS.intersection(value)
        if forbidden:
            raise ValueError(f"{path} exposes forbidden keys: {sorted(forbidden)}")
        identities = FORBIDDEN_FAILURE_ID_KEYS.intersection(value)
        if identities:
            raise ValueError(f"{path} exposes task identity keys: {sorted(identities)}")
        for key, item in value.items():
            _check_public(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_public(item, f"{path}[{index}]")
    elif not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError(f"{path} contains unsupported value {type(value).__name__}")


def build_messages(
    failures: list[dict[str, Any]],
    incumbent_patch: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if not isinstance(failures, list) or not 1 <= len(failures) <= 16:
        raise ValueError("failure packet must contain 1..16 trajectories")
    for index, failure in enumerate(failures):
        if not isinstance(failure, dict):
            raise TypeError(f"failure_packet[{index}] must be an object")
        forbidden = FORBIDDEN_FAILURE_ID_KEYS.intersection(failure)
        if forbidden:
            raise ValueError(
                f"failure_packet[{index}] exposes task identity keys: {sorted(forbidden)}"
            )
    _check_public(failures)
    packet = json.dumps(failures, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(packet.encode("utf-8")) > 100_000:
        raise ValueError("failure packet exceeds 100000 UTF-8 bytes")
    incumbent_text = json.dumps(
        incumbent_patch
        or {
            "schema_version": "harness-r1-patch-v1",
            "benchmark": "searchqa",
            "description": "Initial harness with no lifecycle code hooks.",
            "actions": [],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    user = (
        PATCH_SCHEMA
        + "\n\nCURRENT ACCEPTED HARNESS\n"
        + incumbent_text
        + "\nEND CURRENT ACCEPTED HARNESS\n"
        + "Your patch may replace a subset of its hooks. Unnamed hooks remain active."
        + "\n\nBEGIN FAILURE PACKET\n"
        + packet
        + "\nEND FAILURE PACKET\n"
        + "Write one benchmark-level patch that addresses repeated failure modes."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def sft_assistant_content(reasoning: str, patch: dict[str, Any]) -> str:
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("SFT reasoning must be nonempty")
    patch_text = json.dumps(patch, ensure_ascii=False, separators=(",", ":"))
    return reasoning.strip() + "\n</think>\n<patch>\n" + patch_text + "\n</patch>"
