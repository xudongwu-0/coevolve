"""Predeclared editor tasks and uniform, outcome-independent replay mixing."""
from __future__ import annotations

from pathlib import Path

from .common import read_json, write_json
from .domain_api import get_domain
from .replay import replay_rows


def editor_dataset(harness: str, feedback: float, count: int, partner_version: int,
                   *, replay_dir: Path | None = None, replay_fraction: float = 0.0,
                   editor_step_offset: int = 0, domain="math",
                   failure_feedback: str = "") -> list[dict]:
    task = get_domain(domain)

    if count < 1 or not 0 <= replay_fraction <= 0.5:
        raise ValueError("Invalid editor task/replay budget")
    replay_count = int(count * replay_fraction)
    if replay_count and replay_dir is None:
        raise ValueError("Replay requires authentic previous-phase records")
    repeats = replay_rows(replay_dir, replay_count, editor_step_offset) if replay_count else []
    replay_positions = {i * count // replay_count for i in range(replay_count)}
    instructions = (
        "Improve mathematical correctness without changing the verifier.",
        "Avoid unfinished reasoning; use Python to check exact arithmetic when useful.",
        "Use tool feedback to recover from mistakes without repeating failed computations.",
        "Budget reasoning and tool turns so a boxed answer is produced before truncation.",
    )
    if domain != "math":
        instructions = (
            "Improve task correctness while preserving the fixed verifier.",
            "Use the observed failures to improve observation formatting and action parsing.",
            "Recover from tool or format errors without repeating failed actions.",
            "Reserve enough output budget to finish the task successfully.",
        )
    trace_feedback = (
        f"\nTraining-only failure traces:\n{failure_feedback}\n" if failure_feedback else ""
    )
    response_instruction = (
        "Return only the complete allowed CandidateEnv class module."
        if domain == "math" else
        "Return only the complete allowed CandidateHarness class module."
    )
    calibration = f"Current development accuracy: {feedback:.4f}.\n" if domain == "math" else ""
    rows = []
    for index in range(count):
        if index in replay_positions:
            row = repeats.pop(0)
        else:
            row = {
                "id": f"editor-{partner_version}-{index}",
                "messages": [
                    {"role": "system", "content": task.EDITOR_SYSTEM},
                    {"role": "user", "content": (
                        f"Current harness:\n{harness}\n"
                        f"{calibration}"
                        f"{trace_feedback}"
                        f"{instructions[index % len(instructions)]}\n"
                        f"{response_instruction}"
                    )},
                ],
                "replay": None,
            }
        rows.append({**row, "partner_version": partner_version})
    return rows


def solver_dataset(source: Path, output: Path, count: int, offset: int = 0,
                   recycled_ids: list[str] | None = None,
                   warmup_ids: list[str] | None = None, warmup_rows: int = 0):
    rows = read_json(source)
    if offset < 0 or count < 1 or len(rows) < offset + count:
        raise ValueError("Insufficient distinct solver training questions")
    selected = [dict(row) for row in rows[offset:offset + count]]
    if recycled_ids or warmup_rows:
        # Round-robin attachment keeps the recycled share identical in every optimizer batch.
        for row in selected:
            row["recycled_ids"] = []
            row["warmup"] = False
        for index, episode_id in enumerate(recycled_ids or []):
            selected[index % count]["recycled_ids"].append(episode_id)
    if warmup_rows:
        if not warmup_ids or len(warmup_ids) < warmup_rows:
            raise ValueError("Critic warm-up needs at least one partner episode per row")
        # Same keys as fresh rows (datasets infers the schema from rows); no question is
        # shown to the policy and no sample is generated for these rows.
        blank = {key: "" if isinstance(value, str) else value for key, value in selected[0].items()}
        warm = [dict(blank, id=f"warmup-{i}", warmup=True, recycled_ids=[])
                for i in range(warmup_rows)]
        for index, episode_id in enumerate(warmup_ids):
            warm[index % warmup_rows]["recycled_ids"].append(episode_id)
        selected = warm + selected
    write_json(output, selected)
    return selected
