import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
sys.path.insert(0, str(ROOT / "third_party/sao/vendor/AReaL"))

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.trainer.ppo.actor import PPOActor
from sao_bridge.common import read_json, write_json
from sao_bridge.joint_campaign import prepare, recover_failed_role
from sao_bridge.joint_critic import JointValueNorm
from sao_bridge.joint_experiment import validate_spec
from sao_bridge.joint_health import assess, recover_checkpoints
from sao_bridge.joint_round import checked_behavior, install_version_publication
from sao_bridge.joint_updates import prepare_joint_update
from sao_bridge.joint_schedule import (
    ALTERNATING, accounting, actor_updates, checkpoint_eligible, checkpoint_metadata,
    consumed_row_ledger, install_schedule, role_active, round_metadata, scheduled_rows,
    source_round, total_consumed, update_counts, validate_metric_row, validate_schedule,
)
from sao_bridge.large_train import configure
from sao_bridge.train import upstream_trainer


def spec_for(root, role="editor"):
    return {
        "schedule": ALTERNATING, "schedule_block_size": 16, "starting_point": "h0_w0",
        "protocol": "joint-ma-sao-async-v1", "role": role,
        "model": str(root / role), "critic_model": str(root / role),
        "reference_model": str(root / role), "phase": role, "experiment": "control",
        "steps": 128, "batch_size": 8, "critic_warmup_steps": 5,
        "critic_updates_per_step": 2, "save_interval_steps": 4,
        "critic_mode": "learned", "terminal_episodes": True,
        "solver_trajectory_style": "concat", "value_normalization": True,
        "staleness": 1, "max_head_offpolicyness": 1, "joint_max_lag": 1,
        "lr": 5e-7, "critic_lr": 2.5e-6, "kl_ctl": 0.005,
        "seed": 42, "tokens": 4096, "context_tokens": 32768,
        "actor_gpus": 4, "rollout_gpus": 4, "joint_root": str(root / "joint"),
        "run": str(root / role / "optimizer"), "data": str(root / "data.json"),
        "training_deadline_unix": time.time() + 3600, "pinned_files": {},
    }


def original_rows():
    return [{
        "id": f"h0w0-{i:05d}-task{i % 240}", "joint_version": i // 8,
        "joint_slot": i % 8, "task_id": i % 240, "task_context": f"task{i % 240}",
        "messages": [{"role": "user", "content": f"Frozen W0 evidence row{i}"}],
        "critic_evidence": f"Historical W0 context row{i}",
    } for i in range(512)]


def test_full_schedule_counts_and_active_sequences_match_joint_first512(tmp_path):
    spec = spec_for(tmp_path)
    rows = scheduled_rows(original_rows(), spec)
    write_json(Path(spec["data"]), rows)
    validate_spec(spec)
    assert len(rows) == 1024
    assert len({row["id"] for row in rows}) == 1024
    for role in ("editor", "solver"):
        role_spec = {**spec, "role": role}
        assert update_counts(role_spec, 128) == {
            "active_rounds": 64, "inactive_rounds": 64,
            "critic_only_warmup_steps": 5, "actor_steps": 59, "critic_steps": 128,
        }
        active = [row for row in rows if role_active(role_spec, row["joint_version"])]
        for key in ("task_id", "messages", "task_context", "critic_evidence"):
            assert [row[key] for row in active] == [row[key] for row in original_rows()]
        ledger = consumed_row_ledger(role_spec, rows, 128)
        assert ledger["source_sample_indices"] == list(range(512))
        assert ledger["rows"] == ledger["inactive_counterpart_trajectories"] == 512
        assert ledger["actor_updated_policy_trajectories"] == 472
        assert ledger["critic_warmup_policy_trajectories"] == 40
    assert total_consumed(spec, 128) == 1024
    assert total_consumed({**spec, "schedule": "simultaneous"}, 64) == 1024
    assert sum(accounting({**spec, "role": role}, 128)["committed_generated_policy_trajectories"]
               for role in ("editor", "solver")) == 2048


def test_round_versions_and_policy_updates_are_distinct(tmp_path):
    spec = spec_for(tmp_path, "solver")
    assert not any(actor_updates(spec, step) for step in range(21))
    assert actor_updates(spec, 21)
    assert source_round(spec, 16) == 0
    assert source_round(spec, 32) == 16
    assert round_metadata(spec, 32) == {
        "collection_round": 32, "active_role": "editor", "source_round": 16,
        "role_active": False, "actor_update": False, "critic_updates": 0,
        "collection_policy_version": 32, "actor_updates_before_round": 11,
        "max_policy_version_lag": 1,
    }
    checked_behavior([{"response_ids": [4, 5], "response_logprobs": [-0.2, -0.3],
                       "versions": [31, 32]}], 32, max_lag=1)
    counts = accounting(spec, 32, generated=272)
    assert counts["collection_policy_version"] == 32 and counts["actor_steps"] == 11
    assert counts["uncommitted_generated_policy_trajectories"] == 16
    assert counts["inactive_counterpart_trajectories"] == 128
    with pytest.raises(ValueError, match="undercount"):
        accounting(spec, 32, generated=255)


def freeze(value):
    if isinstance(value, torch.Tensor):
        return value.detach().tolist()
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [freeze(item) for item in value]
    return value


class CPUEngine:
    def __init__(self, critic=False):
        self.parameter = torch.nn.Parameter(torch.tensor([1.]))
        self.optimizer = torch.optim.Adam([self.parameter], lr=5e-7)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.)
        self.normalizer = JointValueNorm() if critic else None
        self.version = 0
        self.calls = dict.fromkeys(("update", "scheduler", "values", "proximal", "advantages"), 0)

    def snapshot(self):
        return freeze({
            "parameter": self.parameter, "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "normalizer": self.normalizer.state() if self.normalizer else None,
        })

    def compute_values(self, batch):
        self.calls["values"] += 1
        return self.normalizer.denormalize([torch.ones(1, 3)])

    def compute_logp(self, batch):
        self.calls["proximal"] += 1
        return [torch.zeros(1, 3)]

    def compute_advantages(self, batch):
        self.calls["advantages"] += 1
        return batch

    def ppo_update(self, batch):
        self.calls["update"] += 1
        if self.normalizer:
            self.normalizer.prepare_update([{
                "returns": torch.tensor([[0., 1., 2.]]),
                "values": torch.ones(1, 3),
                "loss_mask": torch.tensor([[0, 1, 1]]),
            }])
        self.parameter.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def step_lr_scheduler(self):
        self.calls["scheduler"] += 1
        self.scheduler.step()

    def set_version(self, version):
        self.version = version

    def save(self, meta):
        write_json(Path(meta.path) / "config.json", {})
        (Path(meta.path) / "model.safetensors").write_bytes(b"cpu-fixture-not-model-weights")
        if self.normalizer:
            write_json(Path(meta.path) / "joint_value_norm.json", self.normalizer.state())


def fake_trainer():
    return SimpleNamespace(
        actor=CPUEngine(), critic=CPUEngine(critic=True),
        rollout=SimpleNamespace(set_version=Mock()), stats_logger=SimpleNamespace(commit=Mock()),
        config=SimpleNamespace(actor=SimpleNamespace(adv_norm=None, reward_norm=None),
                               saver=SimpleNamespace(mode="sync")),
    )


def native_round(trainer, step):
    batch = [{"input_ids": torch.ones(1, 3), "loss_mask": torch.tensor([[0, 1, 1]])}]
    for row, values in zip(batch, trainer.critic.compute_values(batch)):
        row["values"] = values
    trainer.actor.compute_logp(batch)
    batch = trainer.actor.compute_advantages(batch)
    trainer.actor.ppo_update(batch)
    trainer.actor.step_lr_scheduler()
    for _ in range(2):
        trainer.critic.ppo_update(batch)
    trainer.critic.step_lr_scheduler()
    trainer.actor.set_version(step + 1)
    trainer.critic.set_version(step + 1)
    trainer.rollout.set_version(step + 1)


@pytest.mark.parametrize("role", ["editor", "solver"])
def test_native_order_gates_preserve_inactive_optimizer_and_normalizer_states(tmp_path, role):
    spec = spec_for(tmp_path, role)
    trainer = fake_trainer()
    actor_optimizer, critic_optimizer = trainer.actor.optimizer, trainer.critic.optimizer
    state = install_schedule(trainer, spec)
    install_version_publication(trainer, Path(spec["joint_root"]), role)
    for step in range(128):
        actor_before, critic_before = trainer.actor.snapshot(), trainer.critic.snapshot()
        native_round(trainer, step)
        if not actor_updates(spec, step):
            assert trainer.actor.snapshot() == actor_before
        if not role_active(spec, step):
            assert trainer.critic.snapshot() == critic_before
        assert trainer.actor.optimizer is actor_optimizer
        assert trainer.critic.optimizer is critic_optimizer
        assert trainer.actor.version == trainer.critic.version == step + 1
        trainer.stats_logger.commit(0, step, step, metric_row(spec, step)["metrics"])
    assert state == {"round": 128, "actor_steps": 59, "critic_steps": 128}
    assert trainer.actor.calls == {
        "update": 59, "scheduler": 59, "values": 0, "proximal": 128, "advantages": 128,
    }
    assert trainer.critic.calls["update"] == trainer.critic.normalizer.updates == 128
    assert trainer.critic.calls["values"] == trainer.critic.calls["scheduler"] == 64
    assert read_json(Path(spec["joint_root"]) / f"round-0128/{role}_ready.json") == {
        "role": role, "version": 128,
    }
    for name in ("actor", "critic"):
        path = tmp_path / name / "saved"
        getattr(trainer, name).save(SimpleNamespace(path=path))
        assert read_json(path / "joint_schedule.json") == checkpoint_metadata(spec, 128)


def test_gate_rejects_native_loop_drift_and_stateful_normalization(tmp_path):
    spec = spec_for(tmp_path)
    trainer = fake_trainer()
    trainer.config.actor.adv_norm = SimpleNamespace(mean_level="running")
    with pytest.raises(ValueError, match="running normalization"):
        install_schedule(trainer, spec)
    trainer.config.actor.adv_norm = None
    install_schedule(trainer, spec)
    with pytest.raises(RuntimeError, match="ordering"):
        trainer.actor.set_version(1)


def test_trainer_context_installs_schedule_instead_of_global_warmup(tmp_path, monkeypatch):
    class NativeFixture:
        def __init__(self, config, **kwargs):
            self.__dict__.update(fake_trainer().__dict__)
            self.saver = SimpleNamespace(freq_ctl=Mock())

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    monkeypatch.setenv("MA_SAO_CRITIC_WARMUP_STEPS", "5")
    spec = spec_for(tmp_path, "solver")
    with patch("areal.PPOTrainer", NativeFixture):
        with upstream_trainer(None, [], schedule_spec=spec) as trainer:
            for step in range(22):
                native_round(trainer, step)
            assert trainer.actor.calls["update"] == 1
            assert trainer.critic.calls["update"] == 12
        with upstream_trainer(None, []) as trainer:
            for _ in range(6):
                trainer.actor.ppo_update([])
                trainer.actor.step_lr_scheduler()
            assert trainer.actor.calls["update"] == 1


def test_inactive_counterpart_still_completes_real_proximal_handshake(tmp_path):
    trainers = {}
    for role in ("editor", "solver"):
        trainer = fake_trainer()
        trainer.actor.compute_advantages = (
            lambda data, role=role: prepare_joint_update(
                data, tmp_path / "joint", role, 0,
            )
        )
        install_schedule(trainer, spec_for(tmp_path, role))
        trainers[role] = trainer
    trajectory = {
        "input_ids": torch.tensor([[1, 2, 3]]), "loss_mask": torch.tensor([[0, 1, 1]]),
        "logprobs": torch.tensor([[0., -0.1, -0.2]]),
        "prox_logp": torch.tensor([[-0.1, -0.2, 0.]]),
        "ref_logp": torch.tensor([[-0.1, -0.2, 0.]]),
        "versions": torch.zeros(1, 3, dtype=torch.long),
        "joint_round": torch.tensor([0]), "joint_slot": torch.tensor([0]),
    }
    before = trainers["solver"].critic.snapshot()
    assert trainers["solver"].critic.compute_values([trajectory]) == []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(trainer.actor.compute_advantages, copy.deepcopy(trajectory))
                   for trainer in trainers.values()]
        for future in futures:
            weights, stats = future.result(timeout=10)
            assert weights == [1.] and stats["fixed_reference_available"] == 1.
    assert trainers["solver"].critic.snapshot() == before
    assert len(list((tmp_path / "joint/round-0000").glob("*-proximal-000.json"))) == 2


def metric_row(spec, step):
    result = {
        "global_step": step, "metrics": {
            "joint_policy/fixed_reference_kl": 0.01,
            "joint_policy/fixed_reference_available": 1.,
            "joint_policy/peer_importance_clipped_fraction": 0.,
            "joint_policy/reference_kl_overflow": 0.,
        },
    }
    if role_active(spec, step):
        result["metrics"].update({
            "ppo_critic/update/grad_norm": 1., "joint_value_norm/raw_value_mse": 0.2,
        })
    if actor_updates(spec, step):
        result["metrics"]["ppo_actor/update/grad_norm"] = 1.
    return result


def test_absent_inactive_and_active_warmup_gradients_are_not_collapse(tmp_path):
    spec = spec_for(tmp_path, "solver")
    for step in range(128):
        row = metric_row(spec, step)
        validate_metric_row(spec, row)
        current = Path(spec["joint_root"]) / f"round-{step:04d}"
        for slot in range(8):
            write_json(current / f"proposal-{slot:03d}.json", {"model_error": None})
            write_json(current / f"solver-receipt-{slot:03d}.json", {
                "reward": 1., "calls": [{"response_ids": [4]}],
            })
        report = assess([metric_row(spec, i) for i in range(step + 1)], spec)
        assert report["status"] == "healthy"
        if not role_active(spec, step):
            assert report["value_mse"] is None
    leaked = metric_row(spec, 0)
    leaked["metrics"]["ppo_actor/update/grad_norm"] = 1.
    with pytest.raises(ValueError, match="inactive/warmup"):
        validate_metric_row(spec, leaked)
    leaked = metric_row(spec, 0)
    leaked["metrics"]["joint_value_norm/raw_value_mse"] = 0.2
    with pytest.raises(ValueError, match="ValueNorm"):
        validate_metric_row(spec, leaked)
    missing = metric_row(spec, 21)
    del missing["metrics"]["ppo_actor/update/grad_norm"]
    with pytest.raises(ValueError, match="nonzero"):
        validate_metric_row(spec, missing)


@pytest.mark.parametrize("initial_updates", [0, 128])
def test_checkpoint_pair_requires_both_roles_actor_updates_and_honest_counts(tmp_path, initial_updates):
    spec = spec_for(tmp_path)
    if initial_updates:
        write_json(Path(spec["critic_model"]) / "joint_value_norm.json", {"updates": initial_updates})
    assert not checkpoint_eligible(spec, 20)
    assert checkpoint_eligible(spec, 22)
    for role in ("actor", "critic"):
        for step in (3, 19, 23, 27):
            path = Path(spec["run"]) / f"engine/checkpoints/user/control/editor/{role}/epoch0globalstep{step}"
            write_json(path / "config.json", {})
            (path / "model.safetensors").write_bytes(b"fixture")
            write_json(path / "joint_schedule.json", checkpoint_metadata(spec, step + 1))
            if role == "critic":
                write_json(path / "joint_value_norm.json", {
                    "updates": initial_updates + update_counts(spec, step + 1)["critic_steps"],
                })
    with patch("sao_bridge.joint_health.ensure_model_runtime_metadata"):
        rows = [metric_row(spec, i) for i in range(26)]
        history = recover_checkpoints(spec, rows)
        assert list(history) == ["23"]
        norm = Path(history["23"]["critic"]) / "joint_value_norm.json"
        expected = read_json(norm)
        write_json(norm, {"updates": expected["updates"] + 2})
        with pytest.raises(ValueError, match="ValueNorm count"):
            recover_checkpoints(spec, rows)
        write_json(norm, expected)
        (Path(history["23"]["actor"]) / "joint_schedule.json").unlink()
        with pytest.raises(ValueError, match="update-count metadata"):
            recover_checkpoints(spec, rows)


@pytest.mark.parametrize("key,value", [
    ("schedule", "alternating_sao"), ("schedule_block_size", 8), ("steps", 64),
    ("critic_warmup_steps", 2), ("critic_updates_per_step", 1), ("kl_ctl", 0.),
    ("staleness", 0), ("joint_max_lag", 0), ("lr", 5e-6), ("value_normalization", False),
])
def test_control_rejects_non_scheduling_changes(tmp_path, key, value):
    with pytest.raises(ValueError):
        validate_schedule({**spec_for(tmp_path), key: value})


def test_paired_inputs_cannot_silently_drift(tmp_path):
    spec = spec_for(tmp_path)
    rows = scheduled_rows(original_rows(), spec)
    rows[16 * 8]["messages"][0]["content"] = "different evidence"
    with pytest.raises(ValueError, match="identical copied inputs"):
        validate_schedule(spec, rows)


def test_scheduled_config_parses_natively_without_changing_ppo_objective(tmp_path, monkeypatch):
    monkeypatch.setenv("SAO_LOCAL_ADMIN_TOKEN", "cpu-fixture-only")
    spec = spec_for(tmp_path)
    config = configure(spec, tmp_path)
    baseline = configure({key: value for key, value in spec.items()
                          if key not in ("schedule", "schedule_block_size")}, tmp_path)
    assert config == baseline
    path = tmp_path / "trainer.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved, _ = load_expr_config(["--config", str(path)], PPOConfig)
    assert resolved.actor.should_compute_prox_logp()
    assert resolved.actor.optimizer.lr == 5e-7
    assert resolved.critic.optimizer.lr == 2.5e-6
    assert resolved.actor.kl_ctl == 0.005
    assert resolved.rollout.max_head_offpolicyness == 1
    assert resolved.saver.freq_steps == 4
    assert resolved.actor.adv_norm.mean_level == resolved.actor.adv_norm.std_level == "batch"
    actor = PPOActor(resolved.actor, SimpleNamespace())
    trajectory = {
        "input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3),
        "loss_mask": torch.tensor([[0, 1, 1]]), "rewards": torch.tensor([1.]),
        "logprobs": torch.tensor([[0., -0.1, -0.2]]),
        "prox_logp": torch.tensor([[-0.1, -0.2, 0.]]),
        "ref_logp": torch.tensor([[-0.1, -0.2, 0.]]),
    }
    before = copy.deepcopy(actor.adv_norm.__dict__)
    result = actor._compute_advantages(trajectory)
    assert "values" not in result
    assert torch.isfinite(result["advantages"]).all()
    assert actor.adv_norm.__dict__ == before


def test_prepare_control_keeps_h0w0_prefix_zero_and_matches_round128(tmp_path):
    from sao_bridge import joint_campaign

    state = {"engineer": "h0", "solver": "w0", "packets": str(tmp_path / "packets.json"),
             "packets_sha256": "frozen"}
    write_json(tmp_path / "INITIAL_STATE.json", state)
    read = read_json

    def inputs(path):
        path = Path(path)
        if path == joint_campaign.BASE_PROTOCOL:
            return {"no_patch": "empty"}
        if path == joint_campaign.EVAL_PROTOCOL:
            return {"all_task_ids": [240]}
        if path.name in ("editor_spec.json", "solver_spec.json"):
            return spec_for(tmp_path, path.name.split("_")[0])
        if path.name == "data.json" and "alfworld_joint_ma_pilot_v1" in path.parts:
            return original_rows()[:8]
        return read(path)

    with (
        patch("sao_bridge.joint_campaign.read_json", side_effect=inputs),
        patch("sao_bridge.joint_campaign.initial_inputs", return_value=(original_rows(), state)),
        patch("sao_bridge.joint_campaign.validate_cohort"),
        patch("sao_bridge.joint_campaign.file_sha", return_value="frozen"),
        patch("sao_bridge.joint_experiment.file_sha", return_value="frozen"),
    ):
        protocol = prepare(
            tmp_path, tmp_path / "simultaneous", time.time() + 8 * 3600,
            from_initial=True, memory_check=tmp_path / "MEMORY_COMPLETE.json", schedule=ALTERNATING,
        )
    assert protocol["budget"]["common_executed_trajectories"] == 0
    assert protocol["budget"]["matched_rounds"] == 128
    assert protocol["budget"]["simultaneous_comparison_rounds"] == 64
    assert protocol["budget"]["matched_policy_trajectories"] == 1024
    assert protocol["budget"]["maximum_generated_h_plus_w"] == 2048
    assert "not independent SAO" in protocol["control"]
    for role in ("editor", "solver"):
        prepared = read_json(tmp_path / f"{role}_spec.json")
        validate_schedule(prepared, read_json(Path(prepared["data"])))
        assert prepared["schedule_source_data"] in prepared["pinned_files"]


def test_failure_recovery_counts_only_active_committed_consumption(tmp_path):
    spec = spec_for(tmp_path, "solver")
    write_json(Path(spec["data"]), scheduled_rows(original_rows(), spec))
    for step in range(25):
        for slot in range(8):
            write_json(Path(spec["joint_root"]) / f"round-{step:04d}/solver-receipt-{slot:03d}.json", {})
    optimizer = Path(spec["run"])
    optimizer.mkdir(parents=True)
    with (
        patch("sao_bridge.joint_campaign.metric_rows",
              return_value=[metric_row(spec, i) for i in range(24)]),
        patch("sao_bridge.joint_campaign.recover_checkpoints",
              return_value={"23": {"actor": "actor", "critic": "critic"}}),
        patch("sao_bridge.joint_campaign.file_sha", return_value="metrics"),
    ):
        result = recover_failed_role(spec)
    assert result["actor_updates"] == 3 and result["critic_updates"] == 16
    assert result["consumed_policy_trajectories"] == 64
    assert result["inactive_counterpart_trajectories"] == 128
    assert result["generated_policy_trajectories"] == 200
    assert result["uncommitted_generated_policy_trajectories"] == 8
    assert result["failed_training_not_relabelled_completed"]
    assert read_json(optimizer / "consumed_rows.json")["source_sample_indices"] == list(range(64))


@pytest.mark.parametrize("role", ["editor", "solver"])
def test_anchor_schedule_preserves_inactive_state_and_exact_counts(tmp_path, role):
    from sao_bridge.joint_anchor import SCHEDULE
    spec = {**spec_for(tmp_path, role), "schedule": SCHEDULE, "steps": 48,
            "critic_warmup_steps": 3, "anchor_mode": "fixed_anchor",
            "schedule_source_data": str(tmp_path / "source.json")}
    write_json(Path(spec["schedule_source_data"]), original_rows()[:256])
    rows = scheduled_rows(original_rows()[:256], spec)
    validate_schedule(spec, rows)
    from sao_bridge.common import file_sha
    anchor = tmp_path / 'anchor.json'
    write_json(anchor, {'benchmark': 'alfworld', 'actions': []})
    spec.update(anchor_harness=str(anchor), anchor_harness_sha256=file_sha(anchor))
    spec['pinned_files'] = {str(p): file_sha(p) for p in
                            (anchor, Path(spec['schedule_source_data']))}
    write_json(Path(spec['data']), rows)
    validate_spec(spec)
    unpinned = copy.deepcopy(spec)
    unpinned['pinned_files'].pop(str(anchor))
    with pytest.raises(ValueError, match='not pinned'):
        validate_spec(unpinned)
    with pytest.raises(ValueError, match='matching update schedule'):
        validate_spec({**spec, 'schedule': 'simultaneous'})
    trainer = fake_trainer()
    state = install_schedule(trainer, spec)
    for step in range(48):
        actor_before, critic_before = trainer.actor.snapshot(), trainer.critic.snapshot()
        native_round(trainer, step)
        if not actor_updates(spec, step):
            assert trainer.actor.snapshot() == actor_before
        if not role_active(spec, step):
            assert trainer.critic.snapshot() == critic_before
        trainer.stats_logger.commit(0, step, step, metric_row(spec, step)["metrics"])
    active = 32 if role == "editor" else 48
    assert state == {"round": 48, "actor_steps": active - 3, "critic_steps": 2 * active}
    assert trainer.critic.normalizer.updates == 2 * active
    assert trainer.critic.calls["values"] == trainer.critic.calls["scheduler"] == active
    assert trainer.actor.calls["scheduler"] == active - 3
    assert accounting(spec, 48)["consumed_policy_trajectories"] == active * 8
    assert total_consumed(spec, 48) == 640
    tampered = copy.deepcopy(rows)
    tampered[16]["anchor_round"] = False
    with pytest.raises(ValueError, match="pinned source"):
        validate_schedule(spec, tampered)
