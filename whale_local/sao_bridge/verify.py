"""Verify persisted actor changes and independent critic checkpoints."""
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
import torch

from .common import read_json, write_json


def tensor_files(directory: Path) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in result:
                    raise ValueError(f"Duplicate tensor {name}")
                result[name] = path
    if not result:
        raise ValueError(f"No model tensors in {directory}")
    return result


def compare(before: Path, after: Path, require_score: bool = False) -> dict:
    left, right = tensor_files(before), tensor_files(after)
    common = sorted(set(left) & set(right))
    selected = [
        name for name in common if name.endswith("score.weight")
        or name.endswith("layers.0.self_attn.q_proj.weight")
        or name.endswith("layers.0.mlp.down_proj.weight")
    ]
    if require_score and not any(name.endswith("score.weight") for name in selected):
        raise RuntimeError("Critic checkpoint has no value/score head")
    if not selected:
        raise RuntimeError("No expected tensors available for checkpoint comparison")
    checks = {}
    for name in selected:
        with safe_open(left[name], framework="pt", device="cpu") as reader:
            original = reader.get_tensor(name)
        with safe_open(right[name], framework="pt", device="cpu") as reader:
            saved = reader.get_tensor(name)
        if not torch.isfinite(original).all() or not torch.isfinite(saved).all():
            raise RuntimeError(f"Nonfinite checkpoint tensor: {name}")
        previous = original.to(dtype=saved.dtype).float()
        current = saved.float()
        if not torch.isfinite(previous).all():
            raise RuntimeError(f"Initial tensor overflows checkpoint precision: {name}")
        checks[name] = {
            "changed": not torch.equal(previous, current),
            "max_abs_delta": float((previous - current).abs().max()),
            "initial_dtype": str(original.dtype), "saved_dtype": str(saved.dtype),
            "comparison": "initial_tensor_cast_to_saved_precision",
            "raw_max_abs_delta": float((original.float() - current).abs().max()),
        }
    if not any(item["changed"] for item in checks.values()):
        raise RuntimeError("No persisted parameter change beyond checkpoint precision conversion")
    return checks


def verify(run: Path) -> dict:
    result = {}
    for role in ("editor", "solver"):
        receipt = read_json(run / f"{role}_trained.json")
        actor = Path(receipt["checkpoints"]["actor"])
        critic = Path(receipt["checkpoints"]["critic"])
        actor_checks = compare(Path(receipt["model_initial"]), actor)
        critic_steps = sorted(critic.parent.glob("epoch*epochstep*globalstep*"))
        if len(critic_steps) < 2:
            raise RuntimeError("Need at least two saved critic steps to verify changes")
        critic_checks = compare(critic_steps[0], critic, require_score=True)
        result[role] = {"actor": actor_checks, "critic": critic_checks}
    write_json(run / "checkpoint_verification.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    verify(parser.parse_args().run)
