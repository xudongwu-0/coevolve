"""Scheduling-only MA-SAO control; collection versions are not optimizer counts."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .common import read_json, write_json


ROLES = ("editor", "solver")
ALTERNATING = "alternating_blocks"


def schedule_name(spec: dict) -> str:
    name = spec.get("schedule", "simultaneous")
    if name not in ("simultaneous", ALTERNATING):
        raise ValueError("Unknown joint update schedule")
    block = spec.get("schedule_block_size", 16)
    if type(block) is not int or block < 1:
        raise ValueError("Schedule block size must be a positive integer")
    return name


def role_active(spec: dict, round_index: int, role: str | None = None) -> bool:
    role = spec["role"] if role is None else role
    if role not in ROLES or type(round_index) is not int or round_index < 0:
        raise ValueError("Invalid role or collection round")
    return schedule_name(spec) == "simultaneous" or (
        round_index // spec.get("schedule_block_size", 16)
    ) % 2 == ROLES.index(role)


def source_round(spec: dict, round_index: int) -> int:
    if type(round_index) is not int or round_index < 0:
        raise ValueError("Invalid collection round")
    if schedule_name(spec) == "simultaneous":
        return round_index
    block = spec.get("schedule_block_size", 16)
    return (round_index // (2 * block)) * block + round_index % block


def update_counts(spec: dict, completed_rounds: int, role: str | None = None) -> dict:
    if type(completed_rounds) is not int or not 0 <= completed_rounds <= spec["steps"]:
        raise ValueError("Completed collection rounds exceed the schedule")
    role = spec["role"] if role is None else role
    active = sum(role_active(spec, step, role) for step in range(completed_rounds))
    warmup = min(active, spec.get("critic_warmup_steps", 0))
    return {
        "active_rounds": active,
        "inactive_rounds": completed_rounds - active,
        "critic_only_warmup_steps": warmup,
        "actor_steps": active - warmup,
        "critic_steps": active * spec.get("critic_updates_per_step", 2),
    }


def actor_updates(spec: dict, round_index: int, role: str | None = None) -> bool:
    return role_active(spec, round_index, role) and (
        update_counts(spec, round_index, role)["active_rounds"]
        >= spec.get("critic_warmup_steps", 0)
    )


def round_metadata(spec: dict, round_index: int) -> dict:
    return {
        "collection_round": round_index,
        "active_role": next((role for role in ROLES if role_active(spec, round_index, role)), None)
        if schedule_name(spec) == ALTERNATING else "both",
        "source_round": source_round(spec, round_index),
        "role_active": role_active(spec, round_index),
        "actor_update": actor_updates(spec, round_index),
        "critic_updates": spec.get("critic_updates_per_step", 2) if role_active(spec, round_index) else 0,
        "collection_policy_version": round_index,
        "actor_updates_before_round": update_counts(spec, round_index)["actor_steps"],
        "max_policy_version_lag": spec.get("joint_max_lag", 0),
    }


def accounting(spec: dict, completed_rounds: int, *, generated: int | None = None) -> dict:
    counts = update_counts(spec, completed_rounds)
    batch = spec["batch_size"]
    consumed = counts["active_rounds"] * batch
    processed = completed_rounds * batch
    result = {
        **counts,
        "consumed_policy_trajectories": consumed,
        "committed_generated_policy_trajectories": processed,
        "inactive_counterpart_trajectories": processed - consumed,
        "actor_updated_policy_trajectories": counts["actor_steps"] * batch,
        "critic_warmup_policy_trajectories": counts["critic_only_warmup_steps"] * batch,
        "collection_policy_version": completed_rounds,
        "policy_version_semantics": "Native collection/lag version, not the number of weight-changing updates.",
    }
    if generated is not None:
        if type(generated) is not int or generated < processed:
            raise ValueError("Generated receipts cannot undercount committed paired rounds")
        result.update(
            generated_policy_trajectories=generated,
            uncommitted_generated_policy_trajectories=generated - processed,
        )
    return result


def total_consumed(spec: dict, completed_rounds: int) -> int:
    return sum(update_counts(spec, completed_rounds, role)["active_rounds"]
               for role in ROLES) * spec["batch_size"]


def scheduled_rows(rows: list[dict], spec: dict) -> list[dict]:
    """Repeat inputs, never trajectories; each paired collection has a fresh ID."""
    if schedule_name(spec) != ALTERNATING:
        return copy.deepcopy(rows)
    output = []
    for step in range(spec["steps"]):
        source = source_round(spec, step)
        for slot in range(spec["batch_size"]):
            index = source * spec["batch_size"] + slot
            if index >= len(rows):
                raise ValueError("Insufficient source rows for the active-role sequence")
            row = copy.deepcopy(rows[index])
            row.update(
                id=f"alternating-{step:04d}-{slot:03d}-{rows[index]['id']}",
                joint_version=step, joint_slot=slot,
                schedule_source_index=index, schedule_source_id=rows[index]["id"],
                schedule_active_role=ROLES[(step // spec.get("schedule_block_size", 16)) % 2],
            )
            output.append(row)
    return output


def validate_schedule(spec: dict, rows: list[dict] | None = None) -> None:
    if schedule_name(spec) != ALTERNATING:
        return
    if (
        spec["protocol"] != "joint-ma-sao-async-v1"
        or spec.get("starting_point") != "h0_w0"
        or spec["steps"] != 128 or spec["batch_size"] != 8
        or spec.get("schedule_block_size", 16) != 16
        or spec.get("critic_warmup_steps") != 5
        or spec.get("critic_updates_per_step") != 2
        or spec.get("save_interval_steps") != 4
        or spec.get("critic_mode") != "learned"
        or not spec.get("joint_root") or not spec.get("terminal_episodes")
        or spec.get("central_context") or not spec.get("value_normalization")
        or any(spec.get(key) != 1 for key in ("staleness", "max_head_offpolicyness", "joint_max_lag"))
        or spec.get("kl_ctl") != 0.005 or spec.get("lr") != 5e-7
        or spec.get("critic_lr") != 2.5e-6
        or spec.get("reference_model") != spec.get("model")
        or spec.get("critic_model") != spec.get("model")
    ):
        raise ValueError("Scheduling-only control must preserve the frozen H0/W0 MA settings")
    if rows is None:
        return
    if spec.get("schedule_source_data"):
        if scheduled_rows(read_json(Path(spec["schedule_source_data"])), spec) != rows:
            raise ValueError("Scheduled data differ from the pinned simultaneous source sequence")
    if len(rows) != spec["steps"] * spec["batch_size"]:
        raise ValueError("Scheduled data must cover all paired collection rounds")
    copies = {}
    ids = set()
    for index, row in enumerate(rows):
        step, slot = divmod(index, spec["batch_size"])
        source = source_round(spec, step) * spec["batch_size"] + slot
        if (
            row.get("schedule_source_index") != source
            or row.get("schedule_active_role") != round_metadata(spec, step)["active_role"]
            or not isinstance(row.get("schedule_source_id"), str) or not row["schedule_source_id"]
            or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids
        ):
            raise ValueError("Scheduled rows lost the active input sequence or fresh episode identity")
        ids.add(row.get("id"))
        payload = {key: value for key, value in row.items()
                   if key not in ("id", "joint_version", "joint_slot", "schedule_active_role")}
        if source in copies and copies[source] != payload:
            raise ValueError("Both active-role sequences must use identical copied inputs")
        copies[source] = payload


def consumed_row_ledger(spec: dict, rows: list[dict], completed_rounds: int) -> dict:
    counts = accounting(spec, completed_rounds)
    indices = [index for index in range(completed_rounds * spec["batch_size"])
               if role_active(spec, index // spec["batch_size"])]
    consumed = [rows[index] for index in indices]
    return {
        "rows": len(consumed), "ids": [row["id"] for row in consumed],
        "fresh_ids": [row["id"] for row in consumed], "replay_ids": [],
        "sample_indices": indices, "metrics_committed_prefix_only": True,
        "source_sample_indices": [row.get("schedule_source_index", index)
                                  for index, row in zip(indices, consumed)],
        **counts,
    }


def validate_metric_row(spec: dict, row: dict) -> None:
    step, metrics = row["global_step"], row["metrics"]
    for scope, expected in (
        ("ppo_actor", actor_updates(spec, step)),
        ("ppo_critic", role_active(spec, step)),
    ):
        if scope == "ppo_critic" and schedule_name(spec) != ALTERNATING:
            continue
        gradients = [v for key, v in metrics.items() if key.endswith(f"{scope}/update/grad_norm")]
        update_metrics = [key for key in metrics if f"{scope}/update/" in key]
        if expected:
            if len(gradients) != 1 or not isinstance(gradients[0], (int, float)) or (
                not math.isfinite(gradients[0]) or gradients[0] <= 0
            ):
                raise ValueError(f"{scope} omitted a finite nonzero scheduled gradient")
        elif update_metrics:
            raise ValueError(f"{scope} reported an update during an inactive/warmup round")
    if schedule_name(spec) == ALTERNATING and not role_active(spec, step):
        if any("joint_value_norm/" in key for key in metrics):
            raise ValueError("Inactive critic reported a ValueNorm update")


def checkpoint_eligible(spec: dict, completed_rounds: int) -> bool:
    roles = ROLES if schedule_name(spec) == ALTERNATING else (spec["role"],)
    return all(update_counts(spec, completed_rounds, role)["actor_steps"] > 0 for role in roles)


def checkpoint_metadata(spec: dict, completed_rounds: int) -> dict:
    return {
        "schema": "joint-scheduling-only-v1", "schedule": schedule_name(spec),
        "schedule_block_size": spec.get("schedule_block_size", 16),
        "role": spec["role"], "completed_rounds": completed_rounds,
        "global_step": completed_rounds - 1,
        "max_policy_version_lag": spec.get("joint_max_lag", 0),
        "critic_warmup_active_rounds": spec.get("critic_warmup_steps", 0),
        "critic_updates_per_active_round": spec.get("critic_updates_per_step", 2),
        **accounting(spec, completed_rounds),
        "optimizer_resets_between_blocks": False,
    }


def validate_checkpoint_metadata(path: Path, spec: dict, completed_rounds: int) -> None:
    if schedule_name(spec) == ALTERNATING:
        receipt = path / "joint_schedule.json"
        if not receipt.is_file() or read_json(receipt) != checkpoint_metadata(spec, completed_rounds):
            raise ValueError("Scheduled checkpoint omitted or changed its update-count metadata")


class ScheduleMetrics:
    def __init__(self, delegate, spec: dict):
        self.delegate, self.spec = delegate, spec

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def commit(self, epoch, step, global_step, data):
        validate_metric_row(self.spec, {"global_step": global_step, "metrics": data})
        # MetricsRecorder wraps this object; mutate before it records the same data.
        metadata = round_metadata(self.spec, global_step)
        counts = update_counts(self.spec, global_step + 1)
        data.update({
            f"joint_schedule/{key}": int(value) if isinstance(value, bool) else value
            for key, value in {**metadata, **counts}.items()
            if isinstance(value, (bool, int))
        })
        return self.delegate.commit(epoch, step, global_step, data)


def install_schedule(trainer, spec: dict) -> dict:
    """Gate native controller RPCs, keeping optimizers resident across blocks.

    Inactive value forwards are omitted because JointValueNorm.denormalize changes
    behavior moments. Native advantages accept no values (zero fallback), and still
    perform the paired proximal handshake; those advantages are never optimized.
    """
    validate_schedule(spec)
    if schedule_name(spec) != ALTERNATING or trainer.critic is None:
        raise ValueError("Schedule gates require alternating joint actor/critic engines")
    for name in ("adv_norm", "reward_norm"):
        norm = getattr(trainer.config.actor, name, None)
        if norm is not None and (
            getattr(norm, "mean_level", None) == "running"
            or getattr(norm, "std_level", None) == "running"
        ):
            raise ValueError("Inactive advantage computation cannot mutate running normalization")
    if trainer.config.saver.mode != "sync":
        raise ValueError("Schedule checkpoint receipts require synchronous saves")
    state = {"round": 0, "actor_steps": 0, "critic_steps": 0}
    calls = {"actor": 0, "critic": 0, "values": 0, "actor_lr": 0, "critic_lr": 0}
    actor, critic = trainer.actor, trainer.critic
    actor_update, critic_update = actor.ppo_update, critic.ppo_update
    actor_lr, critic_lr = actor.step_lr_scheduler, critic.step_lr_scheduler
    values, set_version = critic.compute_values, actor.set_version

    def compute_values(batch):
        calls["values"] += 1
        return values(batch) if role_active(spec, state["round"]) else []

    def ppo_actor(batch):
        calls["actor"] += 1
        if actor_updates(spec, state["round"]):
            result = actor_update(batch)
            state["actor_steps"] += 1
            return result

    def ppo_critic(batch):
        calls["critic"] += 1
        if role_active(spec, state["round"]):
            result = critic_update(batch)
            state["critic_steps"] += 1
            return result

    def actor_scheduler():
        calls["actor_lr"] += 1
        if actor_updates(spec, state["round"]):
            return actor_lr()

    def critic_scheduler():
        calls["critic_lr"] += 1
        if role_active(spec, state["round"]):
            return critic_lr()

    def advance(version):
        expected = update_counts(spec, state["round"] + 1)
        if version != state["round"] + 1 or calls != {
            "actor": 1, "critic": spec["critic_updates_per_step"],
            "values": 1, "actor_lr": 1, "critic_lr": 1,
        } or any(state[key] != expected[key] for key in ("actor_steps", "critic_steps")):
            raise RuntimeError("Native loop changed the scheduled update/version ordering")
        result = set_version(version)
        state["round"] = version
        calls.update(dict.fromkeys(calls, 0))
        return result

    def saving(engine):
        original = engine.save

        def save(meta):
            result = original(meta=meta)
            write_json(Path(meta.path) / "joint_schedule.json",
                       checkpoint_metadata(spec, state["round"]))
            return result

        return save

    actor.ppo_update, actor.step_lr_scheduler, actor.set_version = ppo_actor, actor_scheduler, advance
    critic.ppo_update, critic.step_lr_scheduler, critic.compute_values = ppo_critic, critic_scheduler, compute_values
    actor.save, critic.save = saving(actor), saving(critic)
    trainer.stats_logger = ScheduleMetrics(trainer.stats_logger, spec)
    return state
