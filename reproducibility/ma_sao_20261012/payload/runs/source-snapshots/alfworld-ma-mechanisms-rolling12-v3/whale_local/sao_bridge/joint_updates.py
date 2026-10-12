"""Coordinate proximal policies and bound asynchronous teammate importance weights."""
from __future__ import annotations

import fcntl
import math
from pathlib import Path
import time

import torch

from .common import read_json, write_json
from .joint_round import check_failures, round_path


def likelihood_summary(data: dict, index: int, version: int) -> dict:
    mask = torch.roll(data["loss_mask"][index].bool(), -1)
    behavior = torch.roll(data["logprobs"][index].float(), -1)[mask]
    proximal = data["prox_logp"][index].float()[mask]
    versions = torch.roll(data["versions"][index], -1)[mask]
    if (
        not behavior.numel() or behavior.shape != proximal.shape
        or not torch.isfinite(behavior).all() or not torch.isfinite(proximal).all()
        or torch.any(versions < max(0, version - 1)) or torch.any(versions > version)
    ):
        raise ValueError("Joint proximal statistics lost valid action probabilities or policy versions")
    ratio = proximal.double() - behavior.double()
    result = {
        "version": version, "slot": int(data["joint_slot"][index]),
        "action_tokens": int(mask.sum()), "log_importance_sum": float(ratio.sum()),
        "log_importance_mean": float(ratio.mean()),
        "behavior_version_min": int(versions.min()), "behavior_version_max": int(versions.max()),
        "sample_cross_entropy": float(-proximal.mean()),
    }
    if "ref_logp" in data:
        reference = data["ref_logp"][index].float()[mask]
        if not torch.isfinite(reference).all():
            raise FloatingPointError("Joint fixed-reference log probabilities are nonfinite")
        log_ratio = reference.double() - proximal.double()
        result["reference_kl_overflow"] = bool(torch.any(log_ratio.abs() > 20))
        clipped = log_ratio.clamp(-20, 20)
        result["fixed_reference_kl"] = float((clipped.exp() - clipped - 1).mean())
    return result


def publish_proximal(root: Path, role: str, summary: dict) -> dict:
    current = round_path(root, summary["version"])
    current.mkdir(parents=True, exist_ok=True)
    path = current / f"{role}-proximal-{summary['slot']:03d}.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            recorded = read_json(path)
            if set(recorded) != set(summary):
                raise ValueError("Distributed joint proximal metadata fields differ")
            for key, value in summary.items():
                if isinstance(value, float):
                    if not math.isclose(recorded[key], value, rel_tol=1e-5, abs_tol=1e-4):
                        raise ValueError(f"Distributed proximal policies disagree: {key}")
                elif recorded[key] != value:
                    raise ValueError(f"Distributed proximal identity differs: {key}")
            return recorded
        write_json(path, summary)
        return summary


def teammate_weight(summary: dict, version: int, slot: int, bound: float = 2.0) -> tuple[float, bool]:
    if (
        summary["version"] != version or summary["slot"] != slot
        or not math.isfinite(summary["log_importance_sum"]) or not bound > 1
    ):
        raise ValueError("Teammate likelihood belongs to a different joint cohort")
    value = summary["log_importance_sum"]
    limit = math.log(bound)
    return math.exp(min(limit, max(-limit, value))), abs(value) > limit


def prepare_joint_update(data: dict, root: Path, role: str, version: int,
                         weight_mode: str = "bounded") -> tuple[list[float], dict]:
    if weight_mode not in ("bounded", "unit"):
        raise ValueError("Unknown teammate importance weight mode")
    if role not in ("editor", "solver") or torch.any(data["joint_round"] != version):
        raise ValueError("Both joint learners must consume the same ordered round")
    own = [publish_proximal(root, role, likelihood_summary(data, i, version))
           for i in range(data["input_ids"].shape[0])]
    partner = "solver" if role == "editor" else "editor"
    deadline = time.monotonic() + 1800
    weights, clipped = [], []
    for summary in own:
        path = round_path(root, version) / f"{partner}-proximal-{summary['slot']:03d}.json"
        while not path.exists():
            check_failures(root)
            if time.monotonic() >= deadline:
                raise TimeoutError("Teammate proximal likelihood did not arrive")
            time.sleep(0.1)
        weight, limited = teammate_weight(read_json(path), version, summary["slot"])
        weights.append(weight)
        clipped.append(limited)
    raw_weights, raw_clipped = weights, clipped
    if weight_mode == "unit":
        weights, clipped = [1.] * len(weights), [False] * len(weights)
    stats = {
        "peer_importance_mean": sum(weights) / len(weights),
        "peer_importance_clipped_fraction": sum(clipped) / len(clipped),
        "peer_effective_sample_fraction": sum(weights) ** 2 / (len(weights) * sum(w * w for w in weights)),
        "local_log_importance_mean": sum(s["log_importance_mean"] for s in own) / len(own),
        "fixed_reference_kl": sum(s.get("fixed_reference_kl", 0.) for s in own) / len(own),
        "fixed_reference_available": float(all("fixed_reference_kl" in s for s in own)),
        "reference_kl_overflow": float(any(s.get("reference_kl_overflow", False) for s in own)),
        "peer_raw_importance_mean": sum(raw_weights) / len(raw_weights),
        "peer_raw_clipped_fraction": sum(raw_clipped) / len(raw_clipped),
        "peer_unit_weight_mode": float(weight_mode == "unit"),
    }
    return weights, stats
