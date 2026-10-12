import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import shutil
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
sys.path.insert(0, str(ROOT / "third_party/sao/vendor/AReaL"))

from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.joint_continuation import ContinuationStateMixin, optimizer_fingerprint
from sao_bridge.joint_recovery import (
    PairedRecovery, SCHEMA, commit_pair, file_inventory, generation_path,
    finalize_committed_role, initialize_generation, latest_checkpoint, materialize_prefix, restore_trainer, science_digest,
    validate_commit, validate_recovery_spec,
)
from sao_bridge.joint_round import install_version_publication
from sao_bridge.joint_schedule import checkpoint_metadata, install_schedule, update_counts
from sao_bridge.large_train import MetricsRecorder


def make_spec(tmp_path, role="editor", *, normalized=True):
    data = tmp_path / "data.json"
    if not data.exists():
        write_json(data, [{"id": str(i)} for i in range(24 * 8)])
    return {
        "role": role, "steps": 24, "batch_size": 8, "schedule": "joint_joint_anchor",
        "anchor_mode": "sampled_control", "critic_warmup_steps": 3,
        "critic_updates_per_step": 2, "joint_max_lag": 1, "actor_gpus": 1,
        "run": str(tmp_path / "first" / role / "optimizer"),
        "joint_root": str(tmp_path / "first/joint"), "data": str(data),
        "critic_cooperative_context": True, "value_normalization": normalized,
        "experiment": "rolling-test", "phase": role, "training_deadline_unix": 1e12,
        "model": str(tmp_path / "initial-model"),
        "recovery_start_round": 0,
        "rolling_recovery": {
            "root": str(tmp_path / "recovery"), "interval_rounds": 12,
            "attempt": "attempt-0001", "resume": None, "hf_rounds": [24],
        },
    }


def role_payload(spec, completed, generation):
    role = spec["role"]
    payload = generation / role
    payload.mkdir(parents=True)
    write_json(generation / "OWNER.json", {
        "schema": SCHEMA, "root": spec["rolling_recovery"]["root"], "generation": generation.name,
        "completed_rounds": completed,
    })
    counts = update_counts(spec, completed)
    write_json(payload / "spec.json", spec)
    (payload / "metrics.jsonl").write_text("".join(
        json.dumps({"global_step": step, "metrics": {}}) + "\n" for step in range(completed)
    ))
    write_json(payload / "controller.json", {"saver": {}, "evaluator": {}, "stats_logger": {}})
    for name in ("actor", "critic"):
        state = payload / name
        state.mkdir()
        (state / ".metadata").write_bytes(b"native metadata")
        (state / "__0_0.distcp").write_bytes(b"model shard")
        (state / "optim_world_size_1_rank_0.pt").write_bytes(b"adam shard")
        (state / "training-rank0.pt").write_bytes(b"rng shard")
        write_json(state / "adam-rank0.json", {
            "min_adam_step": counts[f"{name}_steps"],
            "max_adam_step": counts[f"{name}_steps"], "sampled_sha256": "sample",
        })
        write_json(state / "joint_schedule.json", checkpoint_metadata(spec, completed))
        if name == "critic":
            write_json(state / "joint_critic_settings.json", {
                "cooperative_context": True, "value_normalization": spec["value_normalization"],
            })
            if spec["value_normalization"]:
                write_json(state / "joint_value_norm.json", {"updates": counts["critic_steps"]})
    write_json(generation / f"{role}.json", {
        "schema": SCHEMA, "role": role, "completed_rounds": completed,
        "batch_size": 8, "committed_sample_cursor": completed * 8,
        "science_sha256": science_digest(spec), "data_sha256": file_sha(Path(spec["data"])),
        "source_run": spec["run"], "source_joint": spec["joint_root"],
        "protected_hf_relative_paths": [], "files": file_inventory(payload),
    })


def paired_payload(tmp_path, completed=12, attempt=1, *, normalized=True):
    root = tmp_path / "recovery"
    generation = generation_path(root, f"round-{completed:04d}-attempt-{attempt:04d}")
    for role in ("editor", "solver"):
        spec = make_spec(tmp_path, role, normalized=normalized)
        spec["rolling_recovery"]["attempt"] = f"attempt-{attempt:04d}"
        role_payload(spec, completed, generation)
    return root, generation


def test_partial_pair_cannot_replace_the_previous_recovery_point(tmp_path):
    root, first = paired_payload(tmp_path)
    assert commit_pair(root, first, 12)
    latest = (root / "LATEST.json").read_bytes()
    next_point = generation_path(root, "round-0024-attempt-0001")
    role_payload(make_spec(tmp_path), 24, next_point)
    assert not commit_pair(root, next_point, 24)
    assert (root / "LATEST.json").read_bytes() == latest
    assert first.is_dir()
    assert latest_checkpoint(root) == first / "COMMITTED.json"


def test_new_verified_pair_is_committed_before_previous_payload_is_deleted(tmp_path):
    root, first = paired_payload(tmp_path)
    commit_pair(root, first, 12)
    _, second = paired_payload(tmp_path, 24)
    commit_pair(root, second, 24)
    assert latest_checkpoint(root) == second / "COMMITTED.json"
    assert not first.exists()
    assert read_json(root / "retention" / f"{first.name}.json")["status"] == "deleted"
    assert (root / "history" / f"{first.name}.json").is_file()
    assert commit_pair(root, second, 24)


def test_only_solver_initializes_shared_owner_and_editor_checks_it(tmp_path):
    root = tmp_path / "recovery"
    generation = generation_path(root, "round-0012-attempt-0001")
    assert not initialize_generation(root, generation, 12, role="editor")
    assert not root.exists()
    assert initialize_generation(root, generation, 12, role="solver")
    before = file_sha(generation / "OWNER.json")
    assert initialize_generation(root, generation, 12, role="editor")
    assert file_sha(generation / "OWNER.json") == before
    with pytest.raises(ValueError, match="requested generation"):
        initialize_generation(root, generation, 24, role="solver")
    owner = read_json(generation / "OWNER.json")
    write_json(generation / "OWNER.json", {**owner, "root": "different-owner"})
    with pytest.raises(ValueError, match="different owner"):
        initialize_generation(root, generation, 12, role="editor")


@pytest.mark.parametrize("role", [None, "unknown"])
def test_pair_publication_rejects_unknown_roles_before_writing(tmp_path, role):
    root = tmp_path / "recovery"
    generation = generation_path(root, "round-0012-attempt-0001")
    with pytest.raises(ValueError, match="Engineer or Solver"):
        initialize_generation(root, generation, 12, role=role)
    with pytest.raises(ValueError, match="Engineer or Solver"):
        commit_pair(root, generation, 12, role=role)
    assert not root.exists()


def test_editor_never_publishes_and_rejects_a_changed_published_digest(tmp_path, monkeypatch):
    import sao_bridge.joint_recovery as recovery

    root, generation = paired_payload(tmp_path)
    assert not commit_pair(root, generation, 12, role="editor")
    assert not (generation / "COMMITTED.json").exists()
    assert commit_pair(root, generation, 12, role="solver")
    monkeypatch.setattr(recovery.fcntl, "flock", Mock(side_effect=AssertionError("Editor tried to lock")))
    assert commit_pair(root, generation, 12, role="editor")
    pointer = read_json(root / "LATEST.json")
    write_json(root / "LATEST.json", {**pointer, "sha256": "changed"})
    with pytest.raises(ValueError, match="changed before Engineer"):
        commit_pair(root, generation, 12, role="editor")


def test_concurrent_native_callbacks_have_one_writer_without_cross_host_flock(tmp_path, monkeypatch):
    import sao_bridge.joint_recovery as recovery

    _, template = paired_payload(tmp_path / "template")
    execution = tmp_path / "execution"
    execution.mkdir()
    specs = {role: make_spec(execution, role) for role in ("editor", "solver")}
    trainers = {}
    for role, spec in specs.items():
        spec["training_deadline_unix"] = time.time() + 20
        run = Path(spec["run"])
        run.mkdir(parents=True)
        (run / "metrics.jsonl").write_text(
            "".join(json.dumps({"global_step": step, "metrics": {}}) + "\n" for step in range(12))
        )
        counts = update_counts(spec, 12)
        trainer = SimpleNamespace(joint_schedule_state={
            "round": 12, "actor_steps": counts["actor_steps"], "critic_steps": counts["critic_steps"],
        })
        for name in ("actor", "critic"):
            source = template / role / name
            engine = SimpleNamespace(
                config=SimpleNamespace(optimizer=SimpleNamespace(
                    lr_scheduler_type="constant", warmup_steps_proportion=0,
                )),
                save=lambda *, meta, source=source: shutil.copytree(source, meta.path),
            )
            setattr(trainer, name, engine)
        for name in ("saver", "evaluator", "stats_logger"):
            setattr(trainer, name, SimpleNamespace(state_dict=lambda: {}))
        trainers[role] = trainer
    committed, continue_publication = threading.Event(), threading.Event()
    identity = threading.local()
    shared_writers = []
    original_write = recovery.write_json
    original_immutable = recovery._immutable_json

    def immutable(path, value):
        if path.name in ("OWNER.json", "COMMITTED.json"):
            shared_writers.append((path.name, identity.role))
        return original_immutable(path, value)

    def write(path, value):
        if path.parent.name == "history":
            committed.set()
            if not continue_publication.wait(10):
                raise TimeoutError("Test publisher was not released")
        if path.name == "LATEST.json" or path.parent.name == "history":
            shared_writers.append((path.name, identity.role))
        return original_write(path, value)

    def callback(role):
        identity.role = role
        PairedRecovery(trainers[role], specs[role]).after_commit(12)

    monkeypatch.setattr(recovery.fcntl, "flock", lambda *args: None)
    monkeypatch.setattr(recovery, "_immutable_json", immutable)
    monkeypatch.setattr(recovery, "write_json", write)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(callback, role) for role in ("editor", "solver")]
        try:
            assert committed.wait(10)
            assert not (Path(specs["editor"]["run"]) / "RECOVERY_SAVED.json").exists()
            assert not (execution / "recovery/LATEST.json").exists()
        finally:
            continue_publication.set()
        for future in futures:
            future.result(timeout=15)
    assert {role for _, role in shared_writers} == {"solver"}
    assert sum(name == "OWNER.json" for name, _ in shared_writers) == 1
    checkpoint = latest_checkpoint(execution / "recovery")
    for spec in specs.values():
        saved = read_json(Path(spec["run"]) / "RECOVERY_SAVED.json")
        assert saved["completed_rounds"] == 12 and Path(saved["checkpoint"]) == checkpoint


def test_replacement_removes_owned_partial_generations_but_preserves_their_audit(tmp_path):
    root, first = paired_payload(tmp_path)
    commit_pair(root, first, 12)
    partial = generation_path(root, "round-0024-attempt-0001")
    partial.mkdir()
    write_json(partial / "OWNER.json", {
        "schema": SCHEMA, "root": str(root), "generation": partial.name, "completed_rounds": 24,
    })
    (partial / "unfinished.distcp").touch()
    _, complete = paired_payload(tmp_path, 24, attempt=2)
    commit_pair(root, complete, 24)
    assert not partial.exists() and not first.exists()
    audit = read_json(root / "retention" / f"{partial.name}.json")
    assert audit["status"] == "deleted"
    assert audit["files"]["unfinished.distcp"]["bytes"] == 0
    assert latest_checkpoint(root) == complete / "COMMITTED.json"


def test_interrupted_deletion_uses_its_exact_prior_authorization(tmp_path):
    root, generation = paired_payload(tmp_path, 24, attempt=2)
    partial = generation_path(root, "round-0012-attempt-0001")
    partial.mkdir()
    (partial / "leftover.distcp").write_bytes(b"leftover")
    write_json(root / "retention" / f"{partial.name}.json", {
        "status": "deleting", "deletion_target": str(partial.resolve()),
        "owner": {"schema": SCHEMA, "root": str(root), "generation": partial.name, "completed_rounds": 12},
    })
    commit_pair(root, generation, 24)
    assert not partial.exists()
    assert read_json(root / "retention" / f"{partial.name}.json")["status"] == "deleted"


@pytest.mark.parametrize("damage", ["adam", "model", "norm", "metrics", "role-round"])
def test_damaged_pair_never_deletes_a_good_recovery_point(tmp_path, damage):
    root, first = paired_payload(tmp_path)
    commit_pair(root, first, 12)
    _, second = paired_payload(tmp_path, 24)
    paths = {
        "adam": second / "solver/actor/optim_world_size_1_rank_0.pt",
        "model": second / "solver/critic/__0_0.distcp",
        "norm": second / "solver/critic/joint_value_norm.json",
        "metrics": second / "solver/metrics.jsonl",
    }
    if damage == "role-round":
        record = read_json(second / "solver.json")
        record["completed_rounds"] = 12
        write_json(second / "solver.json", record)
    else:
        paths[damage].write_bytes(b"truncated")
    with pytest.raises(ValueError):
        commit_pair(root, second, 24)
    assert first.is_dir()
    assert latest_checkpoint(root) == first / "COMMITTED.json"


def test_value_norm_ablation_is_a_valid_recovery_state(tmp_path):
    root, generation = paired_payload(tmp_path, normalized=False)
    assert commit_pair(root, generation, 12)
    assert validate_commit(generation / "COMMITTED.json", root=root)["completed_rounds"] == 12


@pytest.mark.parametrize("name", ["..", "/tmp", "round-0012-attempt-0001/../foreign", "round-12"])
def test_retention_rejects_unowned_or_ambiguous_paths(tmp_path, name):
    with pytest.raises(ValueError):
        generation_path(tmp_path, name)


def test_retention_rejects_a_symlinked_generation(tmp_path):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (tmp_path / "round-0012-attempt-0001").symlink_to(foreign, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        generation_path(tmp_path, "round-0012-attempt-0001")
    assert foreign.exists()


def test_scientific_identity_allows_only_execution_relocation(tmp_path):
    spec = make_spec(tmp_path)
    moved = copy.deepcopy(spec)
    moved.update(run="/new/optimizer", joint_root="/new/joint", recovery_start_round=12)
    moved["rolling_recovery"].update(attempt="attempt-0002", resume="/new/COMMITTED.json")
    assert science_digest(spec) == science_digest(moved)
    for key, value in (("critic_warmup_steps", 0), ("batch_size", 16),
                       ("value_normalization", False), ("data", "/wrong.json")):
        changed = {**moved, key: value}
        assert science_digest(changed) != science_digest(spec)


@pytest.mark.parametrize("start", [-12, 1, 24, True])
def test_invalid_cursor_fails_explicitly(tmp_path, start):
    spec = make_spec(tmp_path)
    spec["recovery_start_round"] = start
    spec["rolling_recovery"]["resume"] = "/checkpoint"
    with pytest.raises(ValueError):
        validate_recovery_spec(spec)


def test_metrics_are_persisted_before_the_recovery_callback(tmp_path):
    path = tmp_path / "metrics.jsonl"
    calls = []

    def after_commit(completed):
        calls.append((completed, read_json(path)["global_step"]))

    MetricsRecorder(Mock(), path, after_commit).commit(0, 0, 0, {"loss": 1})
    assert calls == [(1, 0)]


def test_restores_committed_not_prefetched_data_cursor_and_versions(tmp_path):
    root, generation = paired_payload(tmp_path)
    commit_pair(root, generation, 12)
    spec = make_spec(tmp_path)
    spec.update(run=str(tmp_path / "second/editor/optimizer"),
                joint_root=str(tmp_path / "second/joint"), recovery_start_round=12)
    spec["rolling_recovery"].update(attempt="attempt-0002", resume=str(generation / "COMMITTED.json"))
    trainer = Mock()
    trainer.train_dataloader = StatefulDataLoader(list(range(192)), batch_size=8, num_workers=0)
    trainer.config.train_dataset = SimpleNamespace(shuffle=False, num_workers=0)
    restore_trainer(trainer, spec)
    assert next(iter(trainer.train_dataloader)).tolist() == list(range(96, 104))
    assert len(trainer.train_dataloader) == 24
    assert trainer.recover_info.last_step_info.next().global_step == 12
    for engine in (trainer.actor, trainer.critic):
        assert engine.load.call_args.kwargs["meta"].with_optim
        engine.set_version.assert_called_once_with(12)
    trainer.rollout.workflow_executor.staleness_manager.on_version_recovered.assert_called_once_with(12)
    receipt = read_json(Path(spec["run"]) / "RECOVERY_LOADED.json")
    assert not receipt["adam_reset"] and not receipt["warmup_repeated"]


def test_resumed_publication_does_not_reset_to_version_zero(tmp_path):
    trainer = SimpleNamespace(rollout=Mock())
    install_version_publication(trainer, tmp_path, "editor", initial_version=12)
    assert not (tmp_path / "round-0000").exists()
    assert read_json(tmp_path / "round-0012/editor_ready.json")["version"] == 12
    trainer.rollout.set_version(13)
    assert read_json(tmp_path / "round-0013/editor_ready.json")["version"] == 13


@pytest.mark.parametrize("arm", ["full", "always_joint"])
@pytest.mark.parametrize("role", ["editor", "solver"])
def test_resumed_schedule_keeps_phase_warmup_and_optimizer_counts(tmp_path, arm, role):
    spec = read_json(ROOT / "runs/alfworld_ma_mechanisms_single_seed_v2" / arm / f"{role}_spec.json")
    spec["recovery_start_round"] = 12
    trainer = SimpleNamespace(actor=Mock(), critic=Mock(), stats_logger=Mock(),
                              config=SimpleNamespace(saver=SimpleNamespace(mode="sync")))
    trainer.config.actor = SimpleNamespace(adv_norm=None, reward_norm=None)
    state = install_schedule(trainer, spec)
    before = update_counts(spec, 12)
    assert state == {"round": 12, "actor_steps": before["actor_steps"],
                     "critic_steps": before["critic_steps"]}
    for completed in range(13, 16):
        trainer.critic.compute_values([])
        trainer.actor.ppo_update([])
        trainer.actor.step_lr_scheduler()
        trainer.critic.ppo_update([])
        trainer.critic.ppo_update([])
        trainer.critic.step_lr_scheduler()
        trainer.actor.set_version(completed)
        expected = update_counts(spec, completed)
        assert state == {"round": completed, "actor_steps": expected["actor_steps"],
                         "critic_steps": expected["critic_steps"]}


def test_materialized_prefix_retains_uncommitted_old_attempt_evidence(tmp_path):
    root, generation = paired_payload(tmp_path)
    spec = make_spec(tmp_path)
    joint = Path(spec["joint_root"])
    for step in range(14):
        write_json(joint / f"round-{step:04d}/outcome-000.json", {"reward": 1})
    commit_pair(root, generation, 12)
    spec.update(run=str(tmp_path / "second/editor/optimizer"),
                joint_root=str(tmp_path / "second/joint"), recovery_start_round=12)
    spec["rolling_recovery"].update(attempt="attempt-0002", resume=str(generation / "COMMITTED.json"))
    materialize_prefix(spec)
    restored = Path(spec["joint_root"])
    assert len(list(restored.glob("round-*"))) == 12
    assert not (restored / "round-0012").exists()
    assert (joint / "round-0013/outcome-000.json").is_file()
    assert len((Path(spec["run"]) / "metrics.jsonl").read_text().splitlines()) == 12


def test_checkpoint_cadence_is_twelve_not_every_optimization_call(tmp_path):
    spec = make_spec(tmp_path)
    recovery = PairedRecovery(Mock(), spec)
    for completed in range(1, 12):
        recovery.after_commit(completed)
    assert not Path(spec["rolling_recovery"]["root"]).exists()


def test_final_paired_save_can_be_published_without_loading_or_training(tmp_path, monkeypatch):
    import sao_bridge.common as common
    import sao_bridge.joint_health as health
    from sao_bridge.joint_schedule import actor_updates, role_active

    root, generation = paired_payload(tmp_path, 24)
    for role in ("editor", "solver"):
        spec = make_spec(tmp_path, role)
        metrics = []
        for step in range(24):
            values = {}
            if actor_updates(spec, step):
                values["ppo_actor/update/grad_norm"] = 1.0
            if role_active(spec, step):
                values["ppo_critic/update/grad_norm"] = 1.0
            metrics.append({"global_step": step, "metrics": values})
            for slot in range(8):
                write_json(Path(spec["joint_root"]) / f"round-{step:04d}" / f"{role}-receipt-{slot:03d}.json",
                           {"reward": 1})
        payload = generation / role
        (payload / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in metrics))
        record = read_json(generation / f"{role}.json")
        record["files"] = file_inventory(payload)
        write_json(generation / f"{role}.json", record)
    commit_pair(root, generation, 24)
    spec = make_spec(tmp_path)
    spec.update(run=str(tmp_path / "finalized/editor/optimizer"),
                joint_root=str(tmp_path / "finalized/joint"), recovery_start_round=24)
    spec["rolling_recovery"].update(attempt="attempt-0002", resume=str(generation / "COMMITTED.json"),
                                    finalize_only=True)
    monkeypatch.setattr(health, "recover_checkpoints", lambda *_: {
        "23": {"actor": "/retained/actor", "critic": "/retained/critic"},
    })
    monkeypatch.setattr(common, "ensure_model_runtime_metadata", lambda *_: None)
    release = {"remaining_gpu_pids": [], "verified_filler_pids": [],
               "foreign_gpu_pids": [], "filler_unit_active": True}
    result = finalize_committed_role(spec, release, elapsed_seconds=123)
    assert result["status"] == "completed"
    assert result["actor_updates"] == 13 and result["critic_updates"] == 32
    assert result["generated_policy_trajectories"] == 192
    assert result["consumed_policy_trajectories"] == 128
    assert not (Path(spec["run"]) / "RECOVERY_LOADED.json").exists()
    assert read_json(Path(spec["run"]).parent / "COMPLETE.json") == result


def test_native_dcp_adam_rng_and_next_update_survive_round_trip(tmp_path, monkeypatch):
    from areal.api import SaveLoadMeta
    from areal.engine.fsdp_engine import FSDPEngine
    import torch.distributed as dist

    class Storage:
        _save_optimizer_state = FSDPEngine._save_optimizer_state
        _load_optimizer_state = FSDPEngine._load_optimizer_state

        @property
        def cpu_group(self):
            return dist.group.WORLD

        def save(self, meta):
            FSDPEngine._save_to_dcp(self, meta.path, meta.with_optim)

        def load(self, meta):
            FSDPEngine._load_from_dcp(self, meta.path, meta.with_optim)

    class Engine(ContinuationStateMixin, Storage):
        pass

    monkeypatch.setenv("MA_SAO_ROLLING_RECOVERY", "1")
    assert not dist.is_initialized()
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        engine = Engine()
        engine.rank = 0
        engine.world_size = 1
        engine.model = torch.nn.ModuleDict({"text": torch.nn.Linear(3, 2), "unused": torch.nn.Linear(3, 2)})
        engine.optimizer = torch.optim.Adam(engine.model.parameters(), lr=0.01)
        engine.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(engine.optimizer, lambda _: 1)
        engine._per_layer_optim_wrapper = None

        def update():
            engine.optimizer.zero_grad()
            engine.model["text"](torch.rand(1, 3)).square().sum().backward()
            engine.optimizer.step()
            engine.lr_scheduler.step()

        for _ in range(12):
            update()
        before = optimizer_fingerprint(engine)
        meta = SaveLoadMeta(path=str(tmp_path / "state"), weight_format="dcp", with_optim=True)
        engine.save(meta)
        expected_random = (random.random(), np.random.rand())
        update()
        expected = optimizer_fingerprint(engine)
        expected_lr = engine.lr_scheduler.state_dict()
        engine.load(meta)
        assert optimizer_fingerprint(engine) == before
        assert (random.random(), np.random.rand()) == expected_random
        update()
        assert optimizer_fingerprint(engine) == expected
        assert engine.lr_scheduler.state_dict() == expected_lr
        assert all(parameter not in engine.optimizer.state for parameter in engine.model["unused"].parameters())
    finally:
        dist.destroy_process_group()
