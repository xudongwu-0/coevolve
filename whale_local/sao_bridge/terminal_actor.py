"""Complete-episode boundaries applied after worker RPC tensors materialize."""
from __future__ import annotations

import torch
from areal.engine.fsdp_engine import FSDPPPOActor
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
    def compute_advantages(self, data):
        return batched_call(self._terminal_advantages, data, pass_meta=True)

    def _terminal_advantages(self, data, meta):
        return self.actor._compute_advantages(zero_terminal_value(data), meta)
