"""Privileged joint-state value inputs; native SAO value objectives stay unchanged."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from areal.engine.fsdp_engine import FSDPPPOCritic
from transformers import AutoTokenizer

from .joint_context import CONTEXT_TOKEN_BUDGET, context_text


def prepend_context(sample: dict, prefix: list[int]) -> dict:
    ids = sample["input_ids"]
    if ids.ndim != 2 or ids.shape[0] != 1 or not prefix:
        raise ValueError("Expected one unpadded/padded trajectory and a nonempty context")
    n = len(prefix)
    result = {}
    for key, value in sample.items():
        if torch.is_tensor(value) and value.ndim == 2 and value.shape == ids.shape:
            if key == "position_ids":
                continue
            lead = torch.zeros((1, n), dtype=value.dtype, device=value.device)
            if key == "input_ids":
                lead[0] = torch.tensor(prefix, dtype=value.dtype, device=value.device)
            elif key == "attention_mask":
                lead.fill_(1)
            elif key == "versions":
                lead.fill_(-1)
            result[key] = torch.cat([lead, value], dim=-1)
        else:
            result[key] = value
    result["position_ids"] = (result["attention_mask"].long().cumsum(-1) - 1).clamp(min=0)
    return result


class JointStateCritic(FSDPPPOCritic):
    def __init__(self, config):
        super().__init__(config)
        context = json.loads(Path(os.environ["MA_SAO_CONTEXT_FILE"]).read_text())
        tokenizer = AutoTokenizer.from_pretrained(config.path, local_files_only=True)
        self.context_ids = tokenizer.encode(context_text(context), add_special_tokens=False)
        if len(self.context_ids) > CONTEXT_TOKEN_BUDGET:
            raise ValueError(
                f"Joint context exceeds the reserved {CONTEXT_TOKEN_BUDGET}-token budget"
            )

    @torch.no_grad()
    def compute_values(self, data):
        expanded = [prepend_context(row, self.context_ids) for row in data]
        values = super().compute_values(expanded)
        return [value[..., len(self.context_ids):] for value in values]

    def ppo_update(self, data):
        expanded = [prepend_context(row, self.context_ids) for row in data]
        return super().ppo_update(expanded)
