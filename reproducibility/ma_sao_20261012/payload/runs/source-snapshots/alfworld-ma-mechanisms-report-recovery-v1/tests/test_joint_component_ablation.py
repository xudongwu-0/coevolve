from contextlib import nullcontext
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from areal.engine.fsdp_engine import FSDPPPOCritic
from sao_bridge.common import write_json
from sao_bridge.joint_critic import (
    JointEpisodeCritic, JointValueNorm, attach_joint_context, expand_joint_context,
    validate_critic_settings,
)
from sao_bridge.joint_health import assess, recover_checkpoints
from sao_bridge.joint_schedule import (
    checkpoint_metadata, install_schedule, role_active, round_metadata, scheduled_rows,
    update_counts, validate_checkpoint_metadata, validate_component_ablation, validate_schedule,
)
from sao_bridge.large_train import configure
from test_joint_campaign import metrics, publish_episodes, training_spec
from test_joint_schedule import fake_trainer, metric_row, native_round


@pytest.mark.parametrize("name,peer,context,norm", [
    ("full", "bounded", True, True),
    ("without_peer", "unit", True, True),
    ("without_context", "bounded", False, True),
    ("without_value_norm", "bounded", True, False),
])
def test_each_declared_ablation_changes_only_its_mechanism(name, peer, context, norm):
    spec = {
        "component_ablation": name, "peer_weight_mode": peer,
        "critic_cooperative_context": context, "value_normalization": norm,
    }
    assert validate_component_ablation(spec) == name
    for key in ("critic_cooperative_context", "value_normalization"):
        with pytest.raises(ValueError, match="declared mechanism"):
            validate_component_ablation({**spec, key: not spec[key]})


@pytest.mark.parametrize("name", ["combined_critic", "unknown", [], True])
def test_unknown_or_combined_ablation_is_rejected(name):
    with pytest.raises(ValueError, match="Unknown isolated"):
        validate_component_ablation({"component_ablation": name})


@pytest.mark.parametrize("name", ["without_peer", "without_context", "without_value_norm"])
def test_old_anchor_defaults_remain_strict_and_opt_in_variants_work(name):
    spec = {
        "schedule": "joint_joint_anchor", "steps": 144, "anchor_mode": "sampled_control",
        "critic_mode": "learned", "terminal_episodes": True, "joint_root": "joint",
        "value_normalization": name != "without_value_norm",
        "peer_weight_mode": "unit" if name == "without_peer" else "bounded",
        "critic_cooperative_context": name != "without_context", "schedule_source_data": "rows",
        "component_ablation": name,
    }
    validate_schedule(spec)
    if name != "without_context":
        spec.pop("component_ablation")
        with pytest.raises(ValueError, match="anchor-retention"):
            validate_schedule(spec)


def schedule_ablation_spec(tmp_path, *, every_round=False, role="editor"):
    return {
        "schedule": "joint_joint_anchor", "steps": 144, "batch_size": 8,
        "anchor_mode": "sampled_control", "critic_mode": "learned",
        "terminal_episodes": True, "joint_root": str(tmp_path / "joint"),
        "schedule_source_data": "rows", "role": role,
        "value_normalization": True, "peer_weight_mode": "bounded",
        "critic_cooperative_context": True, "critic_warmup_steps": 3,
        "critic_updates_per_step": 2, "joint_max_lag": 1,
        "component_ablation": "always_joint" if every_round else "full",
        "editor_update_every_round": every_round,
    }


def test_always_joint_changes_updates_not_collection_sequence_or_solver_budget(tmp_path):
    full = schedule_ablation_spec(tmp_path)
    joint = schedule_ablation_spec(tmp_path, every_round=True)
    validate_schedule(full)
    validate_schedule(joint)
    rows = [{"id": str(i), "task_id": i % 240} for i in range(768)]
    assert scheduled_rows(rows, full) == scheduled_rows(rows, joint)
    assert update_counts(full, 144)["actor_steps"] == 93
    assert update_counts(joint, 144) == {
        "active_rounds": 144, "inactive_rounds": 0, "critic_only_warmup_steps": 3,
        "actor_steps": 141, "critic_steps": 288,
    }
    assert update_counts(full, 144, "solver") == update_counts(joint, 144, "solver")
    assert round_metadata(full, 5)["active_role"] == "solver"
    assert round_metadata(joint, 5)["active_role"] == "both"
    assert not role_active(full, 5) and role_active(joint, 5)


@pytest.mark.parametrize("role", ["editor", "solver"])
@pytest.mark.parametrize("every_round", [False, True])
def test_native_update_calls_match_declared_frequency(tmp_path, role, every_round):
    spec = schedule_ablation_spec(tmp_path, every_round=every_round, role=role)
    trainer = fake_trainer()
    state = install_schedule(trainer, spec)
    for step in range(144):
        native_round(trainer, step)
        trainer.stats_logger.commit(0, step, step, metric_row(spec, step)["metrics"])
    expected = update_counts(spec, 144)
    assert state["actor_steps"] == trainer.actor.calls["update"] == expected["actor_steps"]
    assert state["critic_steps"] == trainer.critic.normalizer.updates == expected["critic_steps"]
    assert trainer.critic.calls["values"] == expected["active_rounds"]


def test_schedule_override_cannot_combine_interventions_or_load_old_schedule(tmp_path):
    spec = schedule_ablation_spec(tmp_path, every_round=True)
    for mutation in (
        {"component_ablation": "full"}, {"peer_weight_mode": "unit"},
        {"critic_cooperative_context": False}, {"value_normalization": False},
        {"editor_update_every_round": False}, {"editor_update_every_round": 1},
        {"anchor_mode": "fixed_anchor"}, {"schedule": "simultaneous"},
    ):
        with pytest.raises(ValueError):
            validate_schedule({**spec, **mutation})
    full = schedule_ablation_spec(tmp_path)
    write_json(tmp_path / "joint_schedule.json", checkpoint_metadata(full, 2))
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint_metadata(tmp_path, spec, 2)
    write_json(tmp_path / "joint_schedule.json", checkpoint_metadata(spec, 2))
    validate_checkpoint_metadata(tmp_path, spec, 2)


def trajectory():
    row = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.ones(1, 4, dtype=torch.bool),
        "loss_mask": torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
        "values": torch.tensor([[0., .25, .75, 0.]]),
        "returns": torch.tensor([[0., 0., 1., 0.]]),
    }
    return attach_joint_context(row, [90, 91])


@pytest.mark.parametrize("context,norm", [(True, True), (False, True), (True, False)])
def test_real_critic_paths_keep_actor_inputs_and_raw_value_coordinates(monkeypatch, context, norm):
    engine = object.__new__(JointEpisodeCritic)
    engine.cooperative_context = context
    engine.value_norm = JointValueNorm() if norm else None
    monkeypatch.setattr(JointEpisodeCritic, "data_parallel_group", property(lambda self: None), raising=False)
    monkeypatch.setattr(JointEpisodeCritic, "device", property(lambda self: torch.device("cpu")), raising=False)
    row = trajectory()
    original = {key: value.clone() for key, value in row.items()}
    with patch.object(FSDPPPOCritic, "compute_values", side_effect=lambda data: [
        value["input_ids"].float() for value in data
    ]) as compute:
        values = engine.compute_values([row])
    torch.testing.assert_close(values[0], row["input_ids"].float())
    assert compute.call_args.args[0][0]["input_ids"].shape[-1] == 4 + 2 * context
    with (
        patch.object(FSDPPPOCritic, "ppo_update", side_effect=lambda data: data),
        patch("areal.utils.stats_tracker.scope", return_value=nullcontext()),
        patch("areal.utils.stats_tracker.scalar") as scalar,
    ):
        updated = engine.ppo_update([row])[0]
    active = updated["returns"][updated["loss_mask"]]
    torch.testing.assert_close(active, torch.tensor([-1., 1.] if norm else [0., 1.]))
    assert scalar.call_args.kwargs["raw_value_mse"] == .0625
    assert "joint_context_ids" not in updated
    for key in original:
        torch.testing.assert_close(row[key], original[key])
    if norm:
        assert engine.value_norm.updates == 1


def test_context_disable_keeps_complete_loss_tokens_and_rejects_corrupt_metadata():
    row = trajectory()
    plain, shift = expand_joint_context(row, enabled=False)
    assert shift == 0
    assert torch.equal(plain["input_ids"], row["input_ids"])
    assert torch.equal(plain["loss_mask"], row["loss_mask"])
    row["joint_context_ids"].zero_()
    with pytest.raises(ValueError, match="missing"):
        expand_joint_context(row, enabled=False)


def test_critic_checkpoint_cannot_change_value_coordinate_system(tmp_path):
    settings = {"cooperative_context": False, "value_normalization": True}
    write_json(tmp_path / "joint_critic_settings.json", settings)
    validate_critic_settings(tmp_path, settings)
    with pytest.raises(ValueError, match="settings changed"):
        validate_critic_settings(tmp_path, {**settings, "cooperative_context": True})
    raw = tmp_path / "legacy_normalized"
    write_json(raw / "joint_value_norm.json", {"updates": 10})
    with pytest.raises(ValueError, match="cannot resume"):
        validate_critic_settings(raw, {"cooperative_context": True, "value_normalization": False})


def test_unnormalized_committed_checkpoint_needs_settings_not_fabricated_norm(tmp_path):
    spec = {
        **training_spec(tmp_path), "value_normalization": False,
        "critic_cooperative_context": True, "component_ablation": "without_value_norm",
    }
    root = Path(spec["run"]) / "engine/checkpoints/user/joint/editor"
    for role in ("actor", "critic"):
        path = root / role / "epoch0epochstep3globalstep3"
        write_json(path / "config.json", {})
        (path / "model.safetensors").write_bytes(b"test")
        if role == "critic":
            write_json(path / "joint_critic_settings.json", {
                "cooperative_context": True, "value_normalization": False,
            })
    with patch("sao_bridge.joint_health.ensure_model_runtime_metadata"):
        assert list(recover_checkpoints(spec, [metrics(index) for index in range(4)])) == ["3"]
        write_json(path / "joint_value_norm.json", {"updates": 1})
        with pytest.raises(ValueError, match="cannot resume"):
            recover_checkpoints(spec, [metrics(index) for index in range(4)])


def test_config_sends_independent_flags_even_if_parent_environment_disagrees(tmp_path, monkeypatch):
    monkeypatch.setenv("SAO_LOCAL_ADMIN_TOKEN", "test-only")
    monkeypatch.setenv("MA_SAO_COOPERATIVE_CONTEXT", "1")
    monkeypatch.setenv("MA_SAO_VALUE_NORM", "0")
    spec = {**training_spec(tmp_path), "critic_cooperative_context": False, "value_normalization": True}
    env = configure(spec, tmp_path)["actor"]["scheduling_spec"][0]["env_vars"]
    assert env["MA_SAO_COOPERATIVE_CONTEXT"] == "0"
    assert env["MA_SAO_VALUE_NORM"] == "1"


def test_raw_critic_health_reports_bad_finite_learning_without_stopping_ablation(tmp_path):
    spec = {**training_spec(tmp_path), "value_normalization": False, "stop_on_diagnostic_thresholds": False}
    publish_episodes(spec, 2)
    rows = [metrics(index, kl=.2) for index in range(3)]
    for row in rows:
        row["metrics"].pop("joint_value_norm/raw_value_mse")
        row["metrics"]["joint_critic/raw_value_mse"] = 5.
    health = assess(rows, spec)
    assert health["status"] == "warning" and health["value_mse"] == 5.
    assert len(health["warnings"]) == 2 and not health["stop_reasons"]
    rows[-1]["metrics"]["ppo_actor/update/grad_norm"] = float("nan")
    assert assess(rows, spec)["status"] == "stop"
