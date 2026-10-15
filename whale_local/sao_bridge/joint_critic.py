"""Per-episode privileged context with no changes to decentralized Actor inputs."""
from __future__ import annotations

from functools import lru_cache
import importlib.util
import os
from pathlib import Path

import torch
import torch.distributed as dist
from areal.engine.fsdp_engine import FSDPPPOCritic
from areal.utils.data import all_gather_tensor_container

from .central_critic import prepend_context
from .common import ROOT, file_sha, read_json, write_json
from .joint_context import CONTEXT_TOKEN_BUDGET
from .joint_continuation import ContinuationStateMixin


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


def active_value_targets(rows: list[dict], *, group=None, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    targets = torch.cat([
        row["returns"][row["loss_mask"].bool()].detach().float() for row in rows
    ])
    old = torch.cat([
        row["values"][row["loss_mask"].bool()].detach().float() for row in rows
    ])
    active = torch.stack((targets, old))
    if dist.is_initialized() and dist.get_world_size(group) > 1:
        if device is not None:
            active = active.to(device)
        active = torch.cat(all_gather_tensor_container(active, group=group), dim=1)
    targets, old = active.cpu()
    if not targets.numel() or not torch.isfinite(targets).all() or not torch.isfinite(old).all():
        raise FloatingPointError("Joint critic has empty or nonfinite active value targets")
    return targets, old


def raw_value_statistics(targets: torch.Tensor, old: torch.Tensor) -> dict:
    variance = targets.var(correction=0)
    residual = targets - old
    return {
        "return_mean": float(targets.mean()), "return_std": float(variance.sqrt()),
        "raw_value_mse": float(residual.square().mean()),
        "explained_variance_defined": float(variance > 1e-8),
        "explained_variance": float(1 - residual.var(correction=0) / variance)
        if variance > 1e-8 else 0.0,
        "active_value_tokens": int(targets.numel()),
    }


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

    def prepare_update(
        self, rows: list[dict], *, group=None, device: torch.device | None = None,
    ) -> tuple[list[dict], dict]:
        targets, old = active_value_targets(rows, group=group, device=device)
        self.normalizer.update(targets.unsqueeze(-1))
        self.updates += 1
        mean, std = self.moments()
        if not std > 0 or not torch.isfinite(torch.tensor([mean, std])).all():
            raise FloatingPointError("MAPPO ValueNorm moments are invalid")
        stats = {
            **raw_value_statistics(targets, old),
            "normalizer_mean": mean, "normalizer_std": std,
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


def expand_joint_context(trajectory: dict, *, enabled: bool = True) -> tuple[dict, int]:
    if type(enabled) is not bool:
        raise ValueError("Cooperative context switch must be a boolean")
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
    return (prepend_context(plain, prefix), len(prefix)) if enabled else (plain, 0)


def validate_critic_settings(path: Path, settings: dict) -> None:
    contract = path / "joint_critic_settings.json"
    if contract.exists() and read_json(contract) != settings:
        raise ValueError("Critic checkpoint context/normalization settings changed across continuation")
    if not settings["value_normalization"] and (path / "joint_value_norm.json").exists():
        raise ValueError("A normalized joint critic cannot resume without ValueNorm")


class JointEpisodeCritic(ContinuationStateMixin, FSDPPPOCritic):
    def _create_device_model(self):
        seed = os.environ.get("MA_SAO_FRESH_CRITIC_SEED")
        if seed is None:
            return super()._create_device_model()
        # The pinned Qwen3.5 loader initializes its missing score head on CPU.
        # Isolate this RNG from Actor construction and the MA-only ValueNorm.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            return super()._create_device_model()

    def __init__(self, config):
        super().__init__(config)
        context = os.environ.get("MA_SAO_COOPERATIVE_CONTEXT", "1")
        if context not in ("0", "1"):
            raise ValueError("Critic cooperative context environment must be0 or1")
        self.cooperative_context = context == "1"
        self.value_norm = JointValueNorm() if os.environ.get("MA_SAO_VALUE_NORM") == "1" else None
        self.settings = {
            "cooperative_context": self.cooperative_context,
            "value_normalization": self.value_norm is not None,
        }
        validate_critic_settings(Path(config.path), self.settings)
        path = Path(config.path) / "joint_value_norm.json"
        if path.exists():
            if self.value_norm is None:
                raise ValueError("A normalized joint critic cannot resume without ValueNorm")
            self.value_norm.restore(read_json(path))

    @torch.no_grad()
    def compute_values(self, data):
        expanded = [expand_joint_context(row, enabled=self.cooperative_context) for row in data]
        values = super().compute_values([row for row, _ in expanded])
        values = [value[..., size:] for value, (_, size) in zip(values, expanded)]
        return self.value_norm.denormalize(values) if self.value_norm is not None else values

    def ppo_update(self, data):
        from areal.utils import stats_tracker

        expanded = [expand_joint_context(row, enabled=self.cooperative_context)[0] for row in data]
        if self.value_norm is not None:
            expanded, stats = self.value_norm.prepare_update(
                expanded, group=self.data_parallel_group, device=self.device,
            )
            with stats_tracker.scope("joint_value_norm"):
                stats_tracker.scalar(**stats)
        else:
            targets, old = active_value_targets(expanded, group=self.data_parallel_group, device=self.device)
            with stats_tracker.scope("joint_critic"):
                stats_tracker.scalar(**raw_value_statistics(targets, old))
        return super().ppo_update(expanded)

    def save(self, meta):
        super().save(meta)
        if not dist.is_initialized() or dist.get_rank() == 0:
            write_json(Path(meta.path) / "joint_critic_settings.json", self.settings)
            if self.value_norm is not None:
                write_json(Path(meta.path) / "joint_value_norm.json", self.value_norm.state())

    def load(self, meta):
        validate_critic_settings(Path(meta.path), self.settings)
        super().load(meta)
        if self.value_norm is not None:
            self.value_norm.restore(read_json(Path(meta.path) / "joint_value_norm.json"))
