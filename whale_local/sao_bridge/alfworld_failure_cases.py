"""Auditable paired failure cases from saved native ALFWorld trajectories."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

from .alfworld_strong_phase import checked_native_report
from .alfworld_train_data import _immutable_json
from .common import file_sha, read_json


def action_trace(row: dict) -> list[dict]:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Failure-case analysis needs the unabridged native messages")
    turns = []
    for message in messages:
        if message.get("role") == "assistant":
            actions = []
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    raise ValueError("Native tool arguments must retain their raw JSON string")
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    actions.append({"raw_arguments": arguments, "invalid_json": True})
                else:
                    action = parsed.get("action") if isinstance(parsed, dict) else None
                    actions.append({
                        "action": action, "raw_arguments": arguments,
                        "function": function.get("name"), "invalid_json": False,
                    })
            turns.append({
                "turn": len(turns) + 1, "actions": actions,
                "assistant_text": message.get("content"), "observations": [],
            })
        elif turns and message.get("role") in ("tool", "user"):
            turns[-1]["observations"].append(message.get("content"))
    return turns


def failure_categories(row: dict) -> list[str]:
    if row["reward"] != 0:
        raise ValueError("Only failed environment episodes have failure categories")
    trace = action_trace(row)
    counts = Counter(action["action"] for turn in trace for action in turn["actions"]
                     if isinstance(action.get("action"), str))
    nothing = sum("nothing happens" in str(observation).lower()
                  for turn in trace for observation in turn["observations"])
    categories = []
    if row.get("stop_reason") == "invalid_tool_arguments" or any(
        action["invalid_json"] for turn in trace for action in turn["actions"]
    ):
        categories.append("invalid_tool_arguments")
    if row.get("truncated") or row.get("stop_reason") == "token_limit":
        categories.append("token_budget_exhaustion")
    if counts and max(counts.values()) >= 4:
        categories.append("repeated_action_at_least_four_times")
    if nothing >= 3:
        categories.append("at_least_three_nothing_happens_observations")
    if len(trace) >= 50:
        categories.append("at_least_fifty_model_turns")
    if any(not turn["actions"] for turn in trace):
        categories.append("text_response_without_tool_action")
    return categories or ["terminal_failure_without_the_above_signatures"]


def paired_cases(control: dict, full: dict, *, initial: dict | None = None, limit: int = 4) -> dict:
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Case count must be a bounded positive integer")
    tables = []
    for report in (control, full, *([initial] if initial is not None else [])):
        rows = report["results"]
        table = {row["task_id"]: row for row in rows}
        if len(table) != len(rows) or not table:
            raise ValueError("Case analysis requires one unambiguous sample per question")
        tables.append(table)
    if any(set(table) != set(tables[0]) for table in tables[1:]):
        raise ValueError("Paired case reports cover different question populations")
    for task in tables[0]:
        identities = {(table[task]["id"], table[task]["sample_index"], table[task]["rollout_seed"])
                      for table in tables}
        if len(identities) != 1:
            raise ValueError("A case changed task, sample, or inference seed")
    control_rows, full_rows = tables[:2]
    improved = sorted(task for task in control_rows
                      if control_rows[task]["reward"] == 0 and full_rows[task]["reward"] == 1)
    reversed_ids = sorted(task for task in control_rows
                         if control_rows[task]["reward"] == 1 and full_rows[task]["reward"] == 0)
    categories = {task: failure_categories(control_rows[task]) for task in improved}
    control_failure_counts = Counter(
        label for row in control_rows.values() if row["reward"] == 0 for label in failure_categories(row))
    full_failure_counts = Counter(
        label for row in full_rows.values() if row["reward"] == 0 for label in failure_categories(row))
    regressed = sorted(task for task in improved if initial is not None and tables[2][task]["reward"] == 1)
    ordered = [*regressed, *[task for task in improved if task not in regressed]]
    selected, covered = [], set()
    for task in ordered:
        if len(selected) == limit:
            break
        if not selected or set(categories[task]) - covered:
            selected.append(task)
            covered.update(categories[task])
    for task in ordered:
        if len(selected) == limit:
            break
        if task not in selected:
            selected.append(task)
    examples = []
    for task in selected:
        left, right = control_rows[task], full_rows[task]
        examples.append({
            "task_id": task, "sample_index": left["sample_index"], "rollout_seed": left["rollout_seed"],
            "observable_categories": categories[task],
            "initial_control_reward": tables[2][task]["reward"] if initial is not None else None,
            "control": left, "full_ma": right,
            "control_action_trace": action_trace(left), "full_ma_action_trace": action_trace(right),
        })
    return {
        "questions": len(control_rows),
        "control_failure_full_success": len(improved), "control_failure_full_success_ids": improved,
        "control_success_full_failure": len(reversed_ids), "control_success_full_failure_ids": reversed_ids,
        "both_fail": sum(control_rows[task]["reward"] == full_rows[task]["reward"] == 0
                         for task in control_rows),
        "both_succeed": sum(control_rows[task]["reward"] == full_rows[task]["reward"] == 1
                            for task in control_rows),
        "initial_success_lost_by_control_but_solved_by_full_ids": regressed,
        "category_population_counts": dict(Counter(label for values in categories.values() for label in values)),
        "all_control_failure_category_counts": dict(control_failure_counts),
        "all_full_ma_failure_category_counts": dict(full_failure_counts),
        "selection_rule": (
            "Prioritize initial-control successes lost by the final control and solved by full MA; "
            "then increasing task ID, cover new observable categories, fill to the fixed case limit."
        ),
        "examples": examples,
        "limitations": [
            "Development-set illustrative cases, not causal proof of a particular mechanism.",
            "All discordant IDs and reverse-direction counts are disclosed, not only chosen examples.",
            "If no matched full-MA success exists, report zero cases without inventing one.",
        ],
    }


def _cell(text, limit=420):
    value = str(text).replace("|", "\\|").replace("\n", "<br>")
    return value if len(value) <= limit else value[:limit] + " [excerpt truncated; see raw JSON]"


def case_markdown(result: dict, *, title: str) -> str:
    lines = [
        f"# {title}", "",
        "| Paired population | Count |", "| --- | ---: |",
        f"| Control fails, full MA succeeds | {result['control_failure_full_success']} |",
        f"| Control succeeds, full MA fails | {result['control_success_full_failure']} |",
        f"| Both fail | {result['both_fail']} |", f"| Both succeed | {result['both_succeed']} |", "",
        "Selection rule: " + result["selection_rule"], "",
        "These are observed behaviors, not proof of the causal learning mechanism.",
        "Full paired IDs, unabridged messages and source hashes are in [the case record](./CASES.json).", "",
        "| Observable signature among all failed questions | Control | Full MA |",
        "| --- | ---: | ---: |",
    ]
    categories = sorted(set(result["all_control_failure_category_counts"])
                        | set(result["all_full_ma_failure_category_counts"]))
    lines.extend(f"| {label} | {result['all_control_failure_category_counts'].get(label, 0)} | "
                 f"{result['all_full_ma_failure_category_counts'].get(label, 0)} |" for label in categories)
    lines.extend(["", "Signatures overlap. A zero count is limited to this evaluated population, not a universal guarantee.", ""])
    if not result["examples"]:
        lines.append("No control-failure/full-MA-success case exists in this matched population.")
    for case in result["examples"]:
        lines.extend([
            f"## Task {case['task_id']}, seed {case['rollout_seed']}", "",
            "Observable signatures: " + ", ".join(case["observable_categories"]), "",
        ])
        for label, key, trace_key in (
            ("Control", "control", "control_action_trace"), ("Full MA", "full_ma", "full_ma_action_trace"),
        ):
            row = case[key]
            user = next((message for message in row["messages"] if message.get("role") == "user"), None)
            if user is None:
                raise ValueError("A native failure case omitted its initial task observation")
            text = str(user.get("content"))
            goal = re.search(r"(?:your task is to|your task is|your goal is)[^\n]*", text, re.IGNORECASE)
            lines.extend([
                f"### {label}", "",
                f"Reward={row['reward']}; generated tokens={row['tokens']}; status={row.get('status')}.",
                "Initial task: " + _cell(goal.group(0) if goal else text, 700), "",
                "| Turn | Literal action or response | Literal environment observation |",
                "| ---: | --- | --- |",
            ])
            trace = case[trace_key]
            shown = trace if len(trace) <= 14 else trace[:6] + trace[-8:]
            for turn in shown:
                actions = [action.get("action", action["raw_arguments"]) for action in turn["actions"]]
                lines.append(
                    f"| {turn['turn']} | {_cell(actions or turn['assistant_text'])} | "
                    f"{_cell(turn['observations'])} |")
            if len(trace) > len(shown):
                lines.append(f"\n{len(trace) - len(shown)} middle turns omitted here; retained in the JSON record.")
            lines.append("")
    return "\n".join(lines) + "\n"


def write_case_study(
    output: Path, control_path: Path, full_path: Path, cohort: dict, *, title: str,
    initial_path: Path | None = None, limit: int = 4,
) -> dict:
    paths = [control_path, full_path, *([initial_path] if initial_path is not None else [])]
    reports = [checked_native_report(read_json(path), cohort["all_task_ids"], {**cohort, "samples": 1})
               for path in paths]
    result = paired_cases(reports[0], reports[1], initial=reports[2] if initial_path is not None else None,
                          limit=limit)
    result["sources"] = {str(path): file_sha(path) for path in paths}
    _immutable_json(output / "CASES.json", result)
    text = case_markdown(result, title=title)
    target = output / "CASES.md"
    if target.exists() and target.read_text() != text:
        raise ValueError("Refusing to change an already recorded failure-case selection")
    target.write_text(text)
    return result
