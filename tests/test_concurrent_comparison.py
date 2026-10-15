import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import concurrent_comparison as comparison
from sao_bridge import slurm_runtime


def config(root):
    return {
        "run": str(root), "engineer_model": str(root / "models/harness-r1"),
        "solver_model": str(root / "models/target-agent-sft"),
        "no_patch": str(root / "empty.json"), "training_deadline_unix": 1e12,
        "worker_python": str(root / "worker/bin/python"),
        "nodes": {"ma_sao": {"editor": "n0", "solver": "n1"},
                  "sao": {"editor": "n2", "solver": "n3"}},
    }


def specs(root):
    return {
        (arm, role): comparison.role_spec(config(root), arm, role, root / "data.json", {})
        for arm in comparison.ARMS for role in comparison.ROLES
    }


def plan(root):
    root.mkdir(exist_ok=True)
    pinned = {}
    for (arm, role), spec in specs(root).items():
        path = root / arm / f"{role}_spec.json"
        comparison.immutable(path, spec)
        pinned[str(path)] = comparison.sha(path)
    value = {"schema": comparison.SCHEMA, "config": config(root), "inputs": {}, "specs": pinned}
    comparison.immutable(root / "COMPARISON.json", value)
    return value


def test_four_roles_share_every_budget_and_fresh_model_backbone(tmp_path):
    values = specs(tmp_path)
    for (arm, role), spec in values.items():
        assert comparison.validate_method(spec)
        assert spec["model"] == spec["critic_model"] == spec["reference_model"]
        assert Path(spec["model"]).name == ("harness-r1" if role == "editor" else "target-agent-sft")
        assert spec["rolling_recovery"]["hf_rounds"] == [36, 72, 108, 144]
        assert spec["rolling_recovery"]["retain_generations"] is True
        assert spec["schedule"] == "simultaneous"
        assert spec["fresh_critic_seed"] == 1000043
        assert spec["fsdp_offload_params"] is True
    allowed = {"comparison_arm", "run", "joint_root", "experiment", "rolling_recovery",
               "peer_weight_mode", "critic_cooperative_context", "value_normalization"}
    for role in comparison.ROLES:
        a, b = values["ma_sao", role], values["sao", role]
        assert {k: v for k, v in a.items() if k not in allowed} == {
            k: v for k, v in b.items() if k not in allowed}
    assert comparison.budget()["per_role_actor_updates"] == 141
    assert comparison.budget()["per_role_critic_updates"] == 288
    assert comparison.budget()["accumulation_microbatches_per_rank"] == 2


def test_native_training_entry_cannot_bypass_objective_blocker(tmp_path):
    for value in specs(tmp_path).values():
        with pytest.raises(NotImplementedError, match="ten-task"):
            comparison.require_executable_spec(value)


def test_objective_blocker_does_not_disable_legacy_specs():
    comparison.require_executable_spec({"component_ablation": "without_peer"})


@pytest.mark.parametrize("key,value", [
    ("schedule", "alternating_blocks"), ("steps", 72), ("batch_size", 4),
    ("context_tokens", 16384), ("tokens", 2048), ("seed", 42),
    ("critic_warmup_steps", 5), ("critic_updates_per_step", 1),
    ("component_ablation", "without_peer"), ("critic_model", "old-critic"),
    ("reference_model", "old-actor"), ("editor_reward_mode", "validity_gated"),
    ("actor_gpus", 2), ("rollout_gpus", 6), ("max_concurrent_rollouts", 16),
    ("vllm_max_num_seqs", 16), ("fresh_critic_seed", 42),
    ("critic_cooperative_context", 0),
])
def test_scientific_or_budget_drift_rejected(tmp_path, key, value):
    spec = specs(tmp_path)["sao", "editor"]
    spec[key] = value
    with pytest.raises(ValueError):
        comparison.validate_method(spec)


def test_without_peer_is_not_sao(tmp_path):
    spec = specs(tmp_path)["sao", "editor"]
    spec.update(critic_cooperative_context=True, value_normalization=True)
    with pytest.raises(ValueError, match="mechanisms"):
        comparison.validate_method(spec)
    assert comparison.validate_method({"component_ablation": "without_peer"}) is False


def test_data_relabels_inputs_not_old_trajectories_or_anchor_control():
    source = [
        {"task_id": i % 240, "messages": [{"role": "user", "content": str(i)}],
         "task_context": f"task{i % 240}", "critic_evidence": "frozen evidence",
         "anchor_round": True, "schedule_active_role": "solver", "joint_version": -1}
        for i in range(1152)
    ]
    before = copy.deepcopy(source)
    rows = comparison.paired_rows(source)
    assert source == before
    assert len({r["id"] for r in rows}) == 1152
    assert rows[8]["joint_version"] == 1 and rows[8]["joint_slot"] == 0
    assert not any("anchor_round" in r or "schedule_active_role" in r for r in rows)
    assert [r["messages"] for r in rows] == [r["messages"] for r in source]
    source[0]["replay"] = True
    with pytest.raises(ValueError, match="fresh"):
        comparison.paired_rows(source)


def test_verify_is_fail_closed_on_changes_and_cancellation(tmp_path):
    plan(tmp_path)
    comparison.verify(tmp_path)
    comparison.immutable(tmp_path / "CANCELLED.json", {"reason": "user"})
    with pytest.raises(RuntimeError, match="cancelled"):
        comparison.verify(tmp_path)


def test_spec_mutation_invalidates_manifest(tmp_path):
    plan(tmp_path)
    path = tmp_path / "sao/editor_spec.json"
    with path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="changed"):
        comparison.verify(tmp_path)


def test_path_boundary_rejects_parent_and_symlink_escape(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    assert comparison.contained(root / "ipc", root) == root / "ipc"
    with pytest.raises(ValueError, match="outside"):
        comparison.contained(root / "../outside", root)
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        comparison.contained(root / "escape/file", root)


def test_partial_slurm_identity_and_missing_allocation_fail(monkeypatch):
    monkeypatch.setenv(slurm_runtime.ENV[0], "x")
    monkeypatch.delenv(slurm_runtime.ENV[1], raising=False)
    with pytest.raises(ValueError, match="Incomplete"):
        slurm_runtime.active()
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (Path("x"), {}))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(RuntimeError, match="allocation"):
        slurm_runtime.require_node()


@pytest.mark.parametrize("state,owner,nodes,host", [
    ("PENDING", None, "n0 n1 n2 n3", "n0"),
    ("RUNNING", -1, "n0 n1 n2 n3", "n0"),
    ("RUNNING", None, "n0 n1 n2 n9", "n0"),
    ("RUNNING", None, "n0 n1 n2 n3", "login"),
])
def test_wrong_allocation_never_grants_node(monkeypatch, tmp_path, state, owner, nodes, host):
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, {"config": config(tmp_path)}))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename=host))
    owner = os.getuid() if owner is None else owner
    def command(args, **kwargs):
        return (f"JobState={state} NumNodes=4 UserId=test({owner}) NodeList=n[0-3]"
                if "job" in args else nodes)
    monkeypatch.setattr(slurm_runtime.subprocess, "check_output", command)
    with pytest.raises(RuntimeError):
        slurm_runtime.require_node()


def test_correct_allocation_is_read_only_and_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, {"config": config(tmp_path)}))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv(slurm_runtime.ENV[1], "f" * 64)
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="n2"))
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        return (f"JobState=RUNNING NumNodes=4 UserId=test({os.getuid()}) NodeList=n[0-3]"
                if "job" in args else "n0\nn1\nn2\nn3\n")
    monkeypatch.setattr(slurm_runtime.subprocess, "check_output", command)
    assert slurm_runtime.require_node()["host"] == "n2"
    assert all(args[:2] == ["scontrol", "show"] for args in calls)


def test_ddp_smoke_cannot_unlock_native_training(monkeypatch, tmp_path):
    plan(tmp_path)
    receipt = tmp_path / "tiny_ddp.json"
    comparison.immutable(receipt, {"status": "passed", "world_size": 8})
    monkeypatch.setattr(comparison, "contained", lambda path, root=None: path)
    with pytest.raises(ValueError, match="H100"):
        comparison.native_gate(tmp_path, receipt)


def test_native_gate_requires_equal_heads_and_all_four_profiles(monkeypatch, tmp_path):
    value = plan(tmp_path)
    evidence = tmp_path / "evidence.json"
    comparison.immutable(evidence, {"status": "passed"})
    profile = {
        "context_tokens": 32768, "action_tokens": 4096, "microbatch_sequences": 1,
        "actor_optimizer_steps": 2, "critic_optimizer_steps": 4,
        "finite_nonzero_gradients": True, "gpu_model": "H10080GB",
        "fresh_critic_head_sha256": "a" * 64,
        "evidence": {str(evidence): comparison.sha(evidence)},
    }
    receipt = {
        "schema": "concurrent-h100-native-gate-v1", "status": "passed",
        "comparison_sha256": comparison.sha(tmp_path / "COMPARISON.json"),
        "specs": value["specs"], "outside_root_writes": [],
        "write_boundary_audited": True, "checkpoint_roundtrip_verified": True,
        "profiles": {f"{a}/{r}": copy.deepcopy(profile) for a in comparison.ARMS for r in comparison.ROLES},
    }
    path = tmp_path / "native.json"
    comparison.immutable(path, receipt)
    monkeypatch.setattr(comparison, "contained", lambda path, root=None: path)
    comparison.native_gate(tmp_path, path)
    receipt["profiles"]["sao/editor"]["fresh_critic_head_sha256"] = "b" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="same role"):
        comparison.native_gate(tmp_path, path)


def test_final_eval_does_not_import_gpu_stack_before_both_completion(monkeypatch, tmp_path):
    monkeypatch.setattr(comparison, "require_objective_implementation", lambda: None)
    plan(tmp_path)
    monkeypatch.setattr(comparison, "native_gate", lambda *args: {})
    monkeypatch.setattr(slurm_runtime, "execution_environment", lambda *args: {})
    monkeypatch.setattr(slurm_runtime, "require_node", lambda: {})
    def incomplete(_):
        raise ValueError("both arms incomplete")
    monkeypatch.setattr(comparison, "final_pairs", incomplete)
    with pytest.raises(ValueError, match="incomplete"):
        comparison.evaluate_final(tmp_path, "sao", tmp_path / "native.json")


@pytest.mark.parametrize("entry", ["prepare", "role", "final-eval"])
def test_objective_mismatch_blocks_before_any_read_or_execution(tmp_path, entry):
    with pytest.raises(NotImplementedError, match="ten-task differential"):
        if entry == "prepare":
            comparison.prepare(tmp_path / "does-not-exist.json")
        elif entry == "role":
            comparison.run_role(tmp_path, "sao", "editor", tmp_path / "native.json")
        else:
            comparison.evaluate_final(tmp_path, "ma_sao", tmp_path / "native.json")


@pytest.mark.parametrize("arm", comparison.ARMS)
@pytest.mark.parametrize("role", comparison.ROLES)
def test_real_schedule_hooks_support_simultaneous_paired_recovery(tmp_path, monkeypatch, arm, role):
    # The sparse Windows checkout lacks unrelated WHALE math reward code.
    math_source = Path(__file__).resolve().parents[1] / "WHALE/domains/math_reasoning/verl/utils/reward_score/math_dapo.py"
    if not math_source.exists():
        monkeypatch.setitem(sys.modules, "math_reward", SimpleNamespace(binary_reward=None, native=None))
    from sao_bridge.joint_schedule import install_schedule, validate_checkpoint_metadata
    from unittest.mock import Mock
    actor = SimpleNamespace(
        ppo_update=Mock(), step_lr_scheduler=Mock(), set_version=Mock(), save=Mock())
    critic = SimpleNamespace(
        ppo_update=Mock(), step_lr_scheduler=Mock(), compute_values=Mock(), save=Mock())
    trainer = SimpleNamespace(
        actor=actor, critic=critic, stats_logger=Mock(),
        config=SimpleNamespace(actor=SimpleNamespace(), saver=SimpleNamespace(mode="sync")))
    value = specs(tmp_path)[arm, role]
    original_actor, original_critic = actor.ppo_update, critic.ppo_update
    state = install_schedule(trainer, value)
    for step in range(36):
        critic.compute_values(None)
        actor.ppo_update(None)
        critic.ppo_update(None)
        critic.ppo_update(None)
        actor.step_lr_scheduler()
        critic.step_lr_scheduler()
        actor.set_version(step + 1)
    assert state == {"round": 36, "actor_steps": 33, "critic_steps": 72}
    assert original_actor.call_count == 33 and original_critic.call_count == 72
    saved = tmp_path / "checkpoint"
    saved.mkdir()
    actor.save(SimpleNamespace(path=str(saved)))
    validate_checkpoint_metadata(saved, value, 36)
    with pytest.raises(ValueError, match="metadata"):
        validate_checkpoint_metadata(saved, value, 72)
