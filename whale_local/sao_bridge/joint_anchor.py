"""Deterministic inputs and patch routing for a paired anchor-retention study.

Routing records retain baseline Engineer eligibility. The isolated every-round
ablation overrides native update gates, without changing routing or critic input.
"""
from __future__ import annotations

import copy


SCHEDULE = "joint_joint_anchor"
MODES = ("sampled_control", "fixed_anchor")


def is_anchor_round(round_index: int) -> bool:
    if type(round_index) is not int or round_index < 0:
        raise ValueError("Anchor schedule requires a nonnegative integer round")
    return round_index % 3 == 2


def source_round(round_index: int) -> int:
    """Two fresh joint inputs followed by one repeated input, never a trajectory."""
    is_anchor_round(round_index)
    cycle, phase = divmod(round_index, 3)
    return 2 * cycle + (cycle % 2 if phase == 2 else phase)


def build_rows(rows: list[dict], *, cycles: int, batch_size: int) -> list[dict]:
    if any(type(x) is not int or x < 1 for x in (cycles, batch_size)):
        raise ValueError("Anchor cycles and batch size must be positive integers")
    if len(rows) != 2 * cycles * batch_size:
        raise ValueError("Anchor schedule needs exactly two source rounds per cycle")
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("Source episode IDs must be unique")
    output = []
    for step in range(3 * cycles):
        for slot in range(batch_size):
            index = source_round(step) * batch_size + slot
            row = copy.deepcopy(rows[index])
            row.update(
                id=f"anchor-{step:04d}-{slot:03d}-{rows[index]['id']}",
                joint_version=step, joint_slot=slot,
                schedule_source_index=index, schedule_source_id=rows[index]['id'],
                anchor_round=is_anchor_round(step),
            )
            output.append(row)
    return output


def route_patch(proposal: dict, anchor_patch: dict, *, round_index: int,
                mode: str) -> tuple[dict, dict]:
    if mode not in MODES:
        raise ValueError("Unknown anchor study arm")
    anchor = is_anchor_round(round_index)
    fixed = anchor and mode == "fixed_anchor"
    return copy.deepcopy(anchor_patch if fixed else proposal), {
        "anchor_round": anchor,
        "applied_patch_source": "fixed_anchor" if fixed else "engineer_proposal",
        "engineer_update_eligible": not anchor,
        "solver_peer_weight_rule": "unit" if anchor else "bounded",
    }
