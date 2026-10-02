"""Per-episode privileged context with no changes to decentralized Actor inputs."""
from __future__ import annotations

import torch
from areal.engine.fsdp_engine import FSDPPPOCritic

from .central_critic import prepend_context
from .joint_context import CONTEXT_TOKEN_BUDGET


def attach_joint_context(trajectory: dict, context_ids: list[int]) -> dict:
    if not context_ids or len(context_ids) > CONTEXT_TOKEN_BUDGET:
        raise ValueError("Joint critic context exceeds its explicit token budget")
    ids = trajectory["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or len(context_ids) > ids.shape[1]:
        raise ValueError("Expected one complete trajectory with room for joint context metadata")
    context = torch.zeros_like(ids)
    context[0, :len(context_ids)] = torch.tensor(context_ids, dtype=ids.dtype, device=ids.device) + 1
    return {**trajectory, "joint_context_ids": context}


def expand_joint_context(trajectory: dict) -> tuple[dict, int]:
    metadata = trajectory["joint_context_ids"]
    if metadata.shape != trajectory["input_ids"].shape or metadata.shape[0] != 1:
        raise ValueError("Joint critic expects one unbatched trajectory")
    encoded = metadata[0][metadata[0] > 0]
    prefix = (encoded - 1).tolist()
    if not prefix or len(prefix) > CONTEXT_TOKEN_BUDGET:
        raise ValueError("Joint critic context is missing or oversized")
    if not torch.equal(metadata[0, :len(prefix)], encoded):
        raise ValueError("Joint context metadata must have contiguous valid tokens")
    plain = {key: value for key, value in trajectory.items() if key != "joint_context_ids"}
    return prepend_context(plain, prefix), len(prefix)


class JointEpisodeCritic(FSDPPPOCritic):
    @torch.no_grad()
    def compute_values(self, data):
        expanded = [expand_joint_context(row) for row in data]
        values = super().compute_values([row for row, _ in expanded])
        return [value[..., size:] for value, (_, size) in zip(values, expanded)]

    def ppo_update(self, data):
        return super().ppo_update([expand_joint_context(row)[0] for row in data])
