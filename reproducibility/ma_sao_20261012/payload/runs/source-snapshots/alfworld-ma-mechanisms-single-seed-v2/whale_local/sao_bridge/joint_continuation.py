"""Native Adam-state handoff for bounded, evaluation-separated joint training."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from .common import read_json, write_json


def optimizer_fingerprint(engine) -> dict:
    if engine.optimizer is None:
        raise ValueError("Continuation requires an optimizer")
    digest = hashlib.sha256()
    steps = []
    tensors = 0
    for name, parameter in engine.model.named_parameters():
        state = engine.optimizer.state.get(parameter)
        if not state:
            continue
        if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
            raise ValueError(f"Missing Adam state for {name}")
        steps.append(float(state["step"].item()))
        for key, value in [("parameter", parameter), *sorted(state.items())]:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Unexpected Adam state type: {key}")
            local = value.to_local() if hasattr(value, "to_local") else value
            flat = local.detach().reshape(-1)
            stride = max(1, flat.numel() // 64)
            sample = flat[::stride][:64].float().cpu()
            if not torch.isfinite(sample).all():
                raise FloatingPointError(f"Nonfinite continuation state: {name}/{key}")
            digest.update(f"{name}/{key}/{tuple(local.shape)}/{local.dtype}".encode())
            digest.update(sample.numpy().tobytes())
            tensors += 1
    if not steps or min(steps) <= 0:
        raise ValueError("Continuation omitted nonempty, updated Adam state")
    return {
        "sampled_sha256": digest.hexdigest(), "state_tensors": tensors,
        "parameters_with_state": len(steps), "min_adam_step": min(steps),
        "max_adam_step": max(steps),
        "scope": "All Adam step counters and up to64 elements of each parameter/moment tensor",
    }


class ContinuationStateMixin:
    def save(self, meta):
        if meta.weight_format == "dcp" and meta.with_optim:
            # DCP's combined loader rejects untouched lazy Adam parameters.
            # Keep FP32 model storage in DCP; native rank shards preserve sparse Adam state.
            result = super().save(replace(meta, with_optim=False))
            self._save_optimizer_state(meta.path)
            write_json(Path(meta.path) / f"adam-rank{self.rank}.json", optimizer_fingerprint(self))
            return result
        return super().save(meta)

    def load(self, meta):
        if meta.weight_format == "dcp" and meta.with_optim:
            result = super().load(replace(meta, with_optim=False))
            self._load_optimizer_state(meta.path)
            if self._per_layer_optim_wrapper is not None:
                self._per_layer_optim_wrapper.refresh_states()
            expected = read_json(Path(meta.path) / f"adam-rank{self.rank}.json")
            if optimizer_fingerprint(self) != expected:
                raise ValueError("Loaded parameter/Adam state differs from saved fingerprint")
            return result
        return super().load(meta)


def handoff(trainer, spec: dict, run: Path, *, loading: bool) -> None:
    enabled = spec.get("preserve_optimizer_state", False)
    if type(enabled) is not bool:
        raise ValueError("preserve_optimizer_state must be boolean")
    inputs = spec.get("optimizer_state_input")
    if not enabled:
        if inputs is not None:
            raise ValueError("Optimizer-state input requires explicit preservation")
        return
    if trainer.critic is None or not spec.get("joint_root"):
        raise ValueError("Optimizer handoff requires a joint Actor/Critic pair")
    for engine in (trainer.actor, trainer.critic):
        config = engine.config.optimizer
        if config.lr_scheduler_type != "constant" or config.warmup_steps_proportion != 0:
            raise ValueError("Continuation supports constant, zero-warmup LR only")
    if loading and inputs is None:
        write_json(run / "OPTIMIZER_LOAD.json", {"mode": "initial_fresh_adam"})
        return
    if loading and (not isinstance(inputs, dict) or set(inputs) != {"actor", "critic"}):
        raise ValueError("Continuation requires both Actor and Critic state paths")
    from areal.api import SaveLoadMeta
    paths = {}
    for name in ("actor", "critic"):
        path = Path(inputs[name]) if loading else run / "optimizer_state" / name
        meta = SaveLoadMeta(path=str(path), weight_format="dcp", with_optim=True)
        getattr(getattr(trainer, name), "load" if loading else "save")(meta=meta)
        paths[name] = str(path)
    write_json(run / ("OPTIMIZER_LOAD.json" if loading else "OPTIMIZER_SAVED.json"), {
        "mode": "native_dcp_with_adam", "paths": paths,
        "storage": "FP32 DCP model plus native per-rank Adam shards; untouched parameters retain no Adam state",
        "lr_schedule": "constant; step counter does not alter learning rate",
        "sampling": "Fresh serving process at each block; not bitwise RNG/rollout resume",
    })


def convergence_status(points: list[dict], *, minimum_extra_rounds: int = 192) -> dict:
    if not points or any(
        type(p.get("rounds")) is not int or p["rounds"] < 48
        or not isinstance(p.get("accuracy"), (int, float)) or not 0 <= p["accuracy"] <= 1
        for p in points
    ):
        raise ValueError("Invalid convergence curve")
    if any(b["rounds"] - a["rounds"] != 48 for a, b in zip(points, points[1:])):
        raise ValueError("Convergence observations must be spaced48 rounds apart")
    best = points[0]["accuracy"]
    last_improvement = 0
    for i, point in enumerate(points[1:], 1):
        if point["accuracy"] >= best + 0.005 - 1e-12:
            last_improvement = i
            best = point["accuracy"]
    tail = [p["accuracy"] for p in points[-4:]]
    span = max(tail) - min(tail)
    stale = len(points) - 1 - last_improvement
    eligible = points[-1]["rounds"] - points[0]["rounds"] >= minimum_extra_rounds and len(tail) == 4
    plateau = eligible and stale >= 3 and span <= 0.02 + 1e-12
    return {
        "operational_plateau": plateau, "best_observed_accuracy": max(p["accuracy"] for p in points),
        "last_accuracy": points[-1]["accuracy"], "last_four_range": span,
        "evaluations_without_half_point_record": stale,
        "interpretation": "Stable observed development plateau, not mathematical convergence"
        if plateau else "No operational convergence established",
    }
