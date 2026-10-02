"""Complete-episode boundaries applied after worker RPC tensors materialize."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.infra.platforms import current_platform
from areal.utils.data import batched_call


def zero_terminal_value(trajectory: dict) -> dict:
    if "values" not in trajectory:
        return trajectory
    values = trajectory["values"]
    attention = trajectory["attention_mask"]
    if values.ndim != 2 or values.shape != attention.shape:
        raise ValueError("Terminal critic values must match the 2-D attention mask")
    lengths = attention.sum(-1, keepdim=True).long()
    positions = torch.arange(attention.shape[-1], device=attention.device)
    if (
        torch.any(lengths < 2)
        or not torch.equal(attention, positions.unsqueeze(0) < lengths)
    ):
        raise ValueError("Terminal trajectories require nonempty right-padded sequences")
    values = values.clone()
    # The last token's value is the state AFTER the final action, not an
    # action-state training target. It must be zero even without batch padding.
    values.scatter_(-1, lengths - 1, 0.0)
    return {**trajectory, "values": values}


class TerminalFSDPPPOActor(FSDPPPOActor):
    @torch.no_grad()
    def compute_logp(self, *args, **kwargs):
        result = super().compute_logp(*args, **kwargs)
        if self.config.optimizer is None and os.environ.get("MA_SAO_JOINT_EPISODES") == "1":
            # The colocated reference has a separate allocator from the learner.
            current_platform.empty_cache()
        return result

    @torch.no_grad()
    def compute_advantages(self, data):
        return batched_call(self._terminal_advantages, data, pass_meta=True)

    def _terminal_advantages(self, data, meta):
        joint_root = os.environ.get("MA_SAO_ASYNC_JOINT_ROOT")
        weights = None
        if joint_root:
            from .joint_updates import prepare_joint_update
            from areal.utils import stats_tracker
            weights, stats = prepare_joint_update(
                data, Path(joint_root), os.environ["MA_SAO_JOINT_ROLE"], self.get_version(),
            )
            with stats_tracker.scope("joint_policy"):
                stats_tracker.scalar(**stats)
        result = self.actor._compute_advantages(zero_terminal_value(data), meta)
        if weights is not None:
            peer_weights = torch.tensor(
                weights, device=result["advantages"].device, dtype=result["advantages"].dtype,
            ).unsqueeze(-1)
            result["advantages"] = result["advantages"] * peer_weights
        return result
