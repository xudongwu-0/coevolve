"""Per-episode privileged context with no changes to decentralized Actor inputs."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path

import torch
from areal.engine.fsdp_engine import FSDPPPOCritic

from .central_critic import prepend_context
from .common import ROOT, file_sha, read_json, write_json
from .joint_context import CONTEXT_TOKEN_BUDGET


@lru_cache(maxsize=1)
def official_value_norm():
    path = ROOT / "third_party/mappo-reference/onpolicy/utils/valuenorm.py"
    expected = "df19f747f5845c337d1933ee1a092fa3c758f9ada38bfa3e32aa7d277ac845a2"
    if file_sha(path) != expected:
        raise RuntimeError("Pinned MAPPO ValueNorm source changed")
    spec = importlib.util.spec_from_file_location("coevolve_mappo_value_norm", path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned MAPPO ValueNorm")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ValueNorm


class JointValueNorm:
    def __init__(self):
        self.normalizer = official_value_norm()(1)
        self.updates = 0
        self.behavior_mean, self.behavior_std = 0.0, 1.0

    def moments(self) -> tuple[float, float]:
        # Preserve the raw units of the inherited SAO critic before calibration.
        if self.updates == 0:
            return 0.0, 1.0
        mean, variance = self.normalizer.running_mean_var()
        return float(mean.item()), float(variance.sqrt().item())

    def denormalize(self, values: list[torch.Tensor]) -> list[torch.Tensor]:
        self.behavior_mean, self.behavior_std = self.moments()
        return [value * self.behavior_std + self.behavior_mean for value in values]

    def prepare_update(self, rows: list[dict]) -> tuple[list[dict], dict]:
        targets = torch.cat([
            row["returns"][row["loss_mask"].bool()].detach().float().cpu() for row in rows
        ])
        old = torch.cat([
            row["values"][row["loss_mask"].bool()].detach().float().cpu() for row in rows
        ])
        if not targets.numel() or not torch.isfinite(targets).all() or not torch.isfinite(old).all():
            raise FloatingPointError("Joint critic has empty or nonfinite active value targets")
        self.normalizer.update(targets.unsqueeze(-1))
        self.updates += 1
        mean, std = self.moments()
        if not std > 0 or not torch.isfinite(torch.tensor([mean, std])).all():
            raise FloatingPointError("MAPPO ValueNorm moments are invalid")
        variance = targets.var(correction=0)
        residual = targets - old
        stats = {
            "return_mean": float(targets.mean()), "return_std": float(variance.sqrt()),
            "normalizer_mean": mean, "normalizer_std": std,
            "raw_value_mse": float(residual.square().mean()),
            "explained_variance_defined": float(variance > 1e-8),
            "explained_variance": float(1 - residual.var(correction=0) / variance)
            if variance > 1e-8 else 0.0,
            "active_value_tokens": int(targets.numel()),
        }
        transformed = [{
            **row,
            "returns": (row["returns"] - mean) / std,
            "values": (row["values"] - self.behavior_mean) / self.behavior_std,
        } for row in rows]
        return transformed, stats

    def state(self) -> dict:
        return {
            "updates": self.updates,
            "tensors": {key: value.tolist() for key, value in self.normalizer.state_dict().items()},
            "behavior_mean": self.behavior_mean, "behavior_std": self.behavior_std,
        }

    def restore(self, state: dict) -> None:
        if type(state["updates"]) is not int or state["updates"] < 0:
            raise ValueError("Invalid ValueNorm update count")
        tensors = {key: torch.tensor(value, dtype=torch.float32)
                   for key, value in state["tensors"].items()}
        if (not all(torch.isfinite(value).all() for value in tensors.values())
                or not state["behavior_std"] > 0
                or not torch.isfinite(torch.tensor([
                    state["behavior_mean"], state["behavior_std"],
                ])).all()):
            raise ValueError("Invalid ValueNorm checkpoint moments")
        self.normalizer.load_state_dict(tensors, strict=True)
        self.updates = state["updates"]
        self.behavior_mean, self.behavior_std = state["behavior_mean"], state["behavior_std"]


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
    def __init__(self, config):
        super().__init__(config)
        self.value_norm = JointValueNorm() if os.environ.get("MA_SAO_VALUE_NORM") == "1" else None
        path = Path(config.path) / "joint_value_norm.json"
        if path.exists():
            if self.value_norm is None:
                raise ValueError("A normalized joint critic cannot resume without ValueNorm")
            self.value_norm.restore(read_json(path))

    @torch.no_grad()
    def compute_values(self, data):
        expanded = [expand_joint_context(row) for row in data]
        values = super().compute_values([row for row, _ in expanded])
        values = [value[..., size:] for value, (_, size) in zip(values, expanded)]
        return self.value_norm.denormalize(values) if self.value_norm is not None else values

    def ppo_update(self, data):
        expanded = [expand_joint_context(row)[0] for row in data]
        if self.value_norm is not None:
            from areal.utils import stats_tracker
            expanded, stats = self.value_norm.prepare_update(expanded)
            with stats_tracker.scope("joint_value_norm"):
                stats_tracker.scalar(**stats)
        return super().ppo_update(expanded)

    def save(self, meta):
        super().save(meta)
        if self.value_norm is not None:
            import torch.distributed as dist
            if not dist.is_initialized() or dist.get_rank() == 0:
                write_json(Path(meta.path) / "joint_value_norm.json", self.value_norm.state())

    def load(self, meta):
        super().load(meta)
        if self.value_norm is not None:
            self.value_norm.restore(read_json(Path(meta.path) / "joint_value_norm.json"))
