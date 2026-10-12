"""Auditable early warnings and recovery of fully committed joint checkpoints."""
from __future__ import annotations

import fcntl
import json
import math
from pathlib import Path
import time

from .common import ensure_model_runtime_metadata, read_json, write_json
from .joint_round import round_path
from .joint_schedule import (
    ALTERNATING, actor_updates, checkpoint_eligible, role_active, round_metadata,
    schedule_name, update_counts, validate_checkpoint_metadata, validate_metric_row,
)


def metric_value(metrics: dict, suffix: str) -> float:
    values = [value for key, value in metrics.items() if key.endswith(suffix)]
    if len(values) != 1 or not isinstance(values[0], (int, float)):
        raise ValueError(f"Missing or ambiguous joint training metric: {suffix}")
    return float(values[0])


def assess(rows: list[dict], spec: dict) -> dict:
    if not rows:
        return {"status": "starting", "completed_steps": 0, "warnings": []}
    if [row["global_step"] for row in rows] != list(range(len(rows))):
        raise ValueError("Joint metrics are not a complete ordered prefix")
    latest = rows[-1]
    metrics = latest["metrics"]
    nonfinite = [key for key, value in metrics.items()
                 if isinstance(value, (int, float)) and not math.isfinite(value)]
    hard = [f"nonfinite metric: {key}" for key in nonfinite]
    if schedule_name(spec) != "simultaneous":
        try:
            validate_metric_row(spec, latest)
        except ValueError as exc:
            hard.append(str(exc))
    elif actor_updates(spec, latest["global_step"]):
        grad = metric_value(metrics, "ppo_actor/update/grad_norm")
        if grad <= 0:
            hard.append("actor has no nonzero gradient")
    warnings, stop_reasons = [], []
    value_scope = "joint_value_norm" if spec.get("value_normalization", True) else "joint_critic"
    for suffix, threshold in (
        ("joint_policy/fixed_reference_kl", 0.10),
        (f"{value_scope}/raw_value_mse", 4.0),
    ):
        eligible_rows = [row for row in rows if role_active(spec, row["global_step"])] \
            if suffix.startswith(value_scope + "/") else rows
        if len(eligible_rows) >= 3 and all(
            metric_value(row["metrics"], suffix) > threshold for row in eligible_rows[-3:]
        ):
            target = stop_reasons if spec.get("stop_on_diagnostic_thresholds", True) else warnings
            target.append(f"three consecutive {suffix} > {threshold}")
    if len(rows) >= 3 and all(
        metric_value(row["metrics"], "joint_policy/peer_importance_clipped_fraction") > 0.75
        for row in rows[-3:]
    ):
        # Long autoregressive trajectory ratios can clip without policy collapse.
        warnings.append("three consecutive peer importance clipping fractions > 0.75")
    if metric_value(metrics, "joint_policy/fixed_reference_available") != 1:
        hard.append("fixed-reference diagnostic is unavailable")
    if metric_value(metrics, "joint_policy/reference_kl_overflow") > 0:
        hard.append("fixed-reference log ratio overflow")
    current = round_path(Path(spec["joint_root"]), latest["global_step"])
    proposals = [read_json(current / f"proposal-{slot:03d}.json")
                 for slot in range(spec["batch_size"])]
    receipts = [read_json(current / f"{spec['role']}-receipt-{slot:03d}.json")
                for slot in range(spec["batch_size"])]
    success = sum(row["reward"] for row in receipts) / len(receipts)
    return {
        "status": "stop" if hard or stop_reasons else "warning" if warnings else "healthy",
        "completed_steps": len(rows), "hard_errors": hard,
        "stop_reasons": stop_reasons, "warnings": warnings,
        "latest_team_success": success,
        "invalid_patch_fraction": sum(row["model_error"] is not None for row in proposals) / len(proposals),
        "completion_cap_fraction": sum(
            sum(len(call["response_ids"]) for call in row["calls"]) >= spec["tokens"]
            for row in receipts
        ) / len(receipts),
        "fixed_reference_kl": metric_value(metrics, "joint_policy/fixed_reference_kl"),
        "value_mse": metric_value(metrics, f"{value_scope}/raw_value_mse")
        if role_active(spec, latest["global_step"]) else None,
        "schedule": round_metadata(spec, latest["global_step"]),
        "interpretation": "Training diagnostics only; loss is not a held-out performance guarantee.",
    }


def request_stop(root: Path, reason: dict) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with (root / "STOP.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root / "STOP.json"
        if not path.exists():
            write_json(path, {**reason, "at_unix": time.time()})
        return read_json(path)


def metric_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = path.read_text()
    # An in-flight append is not a committed optimizer receipt.
    return [json.loads(line) for line in content.splitlines(keepends=True)
            if line.endswith("\n") and line.strip()]


def recover_checkpoints(spec: dict, rows: list[dict]) -> dict:
    root = Path(spec["run"]) / "engine/checkpoints"
    normalized = spec.get("value_normalization", True)
    initial_norm_updates = 0
    if schedule_name(spec) != "simultaneous":
        initial_norm = Path(spec.get("critic_model", spec["model"])) / "joint_value_norm.json"
        if initial_norm.exists():
            initial_norm_updates = read_json(initial_norm).get("updates")
            if type(initial_norm_updates) is not int or initial_norm_updates < 0:
                raise ValueError("Invalid initial critic ValueNorm update count")
    by_role = {}
    for role in ("actor", "critic"):
        found = {}
        for path in root.glob(f"*/{spec['experiment']}/{spec['phase']}/{role}/epoch*globalstep*"):
            step = int(path.name.split("globalstep")[-1])
            if step >= len(rows) or not checkpoint_eligible(spec, step + 1):
                continue
            if not (path / "config.json").is_file() or not list(path.glob("*.safetensors")):
                continue
            if role == "critic" and normalized and not (path / "joint_value_norm.json").is_file():
                raise ValueError("Committed normalized critic omitted its running statistics")
            if role == "critic" and spec.get("component_ablation") is not None:
                from .joint_critic import validate_critic_settings
                if not (path / "joint_critic_settings.json").is_file():
                    raise ValueError("Ablation checkpoint omitted its executed critic settings")
                validate_critic_settings(path, {
                    "cooperative_context": spec.get("critic_cooperative_context", True),
                    "value_normalization": normalized,
                })
            validate_checkpoint_metadata(path, spec, step + 1)
            if role == "critic" and normalized and schedule_name(spec) != "simultaneous":
                if read_json(path / "joint_value_norm.json").get("updates") != (
                    initial_norm_updates + update_counts(spec, step + 1)["critic_steps"]
                ):
                    raise ValueError("Scheduled critic ValueNorm count differs from native updates")
            if step in found:
                raise ValueError("Ambiguous native checkpoint owner")
            found[step] = path
        by_role[role] = found
    steps = sorted(set(by_role["actor"]) & set(by_role["critic"]))
    if not steps:
        raise RuntimeError("No complete post-warmup actor/critic checkpoint can be recovered")
    result = {}
    for step in steps:
        actor = by_role["actor"][step]
        ensure_model_runtime_metadata(actor, Path(spec["model"]))
        result[str(step)] = {role: str(by_role[role][step]) for role in by_role}
    return result
