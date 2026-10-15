import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sao_bridge import concurrent_comparison as comparison
from sao_bridge import ma_slurm, slurm_runtime


EXPECTED = {
    "joint_only": ("unit", False, False),
    "add_context": ("unit", True, False),
    "add_peer": ("bounded", False, False),
    "add_valuenorm": ("unit", False, True),
}
KEYS = ("peer_weight_mode", "critic_cooperative_context", "value_normalization")


def config(root, arm):
    return {
        "run": str(root), "active_arms": [arm],
        "nodes": {arm: {"editor": "n0", "solver": "n1"}},
        "engineer_model": str(root / "models/harness-r1"),
        "solver_model": str(root / "models/target-agent-sft"),
        "no_patch": str(root / "empty.json"), "training_deadline_unix": 1e12,
    }


def bound(root, arm):
    settings = config(root, arm)
    spec = comparison.role_spec(settings, arm, "editor", root / "data.json", {})
    path = root / arm / "editor_spec.json"
    comparison.immutable(path, spec)
    plan = {"schema": comparison.SCHEMA, "kind": "standalone-ma-ablation-training",
            "config": settings, "inputs": {}, "specs": {str(path): comparison.sha(path)}}
    comparison.immutable(root / "COMPARISON.json", plan)
    return plan, spec


@pytest.mark.parametrize("arm", EXPECTED)
@pytest.mark.parametrize("role", comparison.ROLES)
def test_exact_mechanism_interventions_and_matching_training_budget(tmp_path, arm, role):
    baseline = comparison.role_spec(config(tmp_path, "joint_only"), "joint_only", role,
                                    tmp_path / "data.json", {})
    spec = comparison.role_spec(config(tmp_path, arm), arm, role, tmp_path / "data.json", {})
    assert tuple(spec[key] for key in KEYS) == EXPECTED[arm]
    assert sum(a != b for a, b in zip(EXPECTED[arm], EXPECTED["joint_only"])) == (
        0 if arm == "joint_only" else 1)
    allowed = {*KEYS, "comparison_arm", "run", "joint_root", "experiment", "rolling_recovery"}
    assert {key: value for key, value in spec.items() if key not in allowed} == {
        key: value for key, value in baseline.items() if key not in allowed}
    assert spec["editor_reward_mode"] == "team"
    assert spec["steps"] == 144 and spec["batch_size"] == 8
    assert spec["schedule"] == "simultaneous"
    assert spec["rolling_recovery"]["hf_rounds"] == [36, 72, 108, 144]
    assert spec["rolling_recovery"]["interval_rounds"] == 36
    assert spec["rolling_recovery"]["retain_generations"]
    assert spec["model"] == spec["critic_model"] == spec["reference_model"]
    assert spec["critic_warmup_steps"] == 3 and spec["critic_updates_per_step"] == 2
    assert comparison.validate_method(spec)


@pytest.mark.parametrize("arm", EXPECTED)
def test_exact_allocation_manifest_required_for_each_addback(monkeypatch, tmp_path, arm):
    plan, spec = bound(tmp_path, arm)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    comparison.require_executable_spec(spec)
    assert ma_slurm.single_arm(plan) == arm
    altered = copy.deepcopy(spec)
    altered["run"] += "-different"
    with pytest.raises(ValueError, match="bound"):
        comparison.require_executable_spec(altered)
    plan["kind"] = "standalone-ma-training"
    with pytest.raises(ValueError, match="bound"):
        comparison.require_executable_spec(spec)


@pytest.mark.parametrize("arm", EXPECTED)
def test_addbacks_use_two_allocated_nodes_and_no_other_nodes(monkeypatch, tmp_path, arm):
    plan, _ = bound(tmp_path, arm)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv(slurm_runtime.ENV[1], comparison.sha(tmp_path / "COMPARISON.json"))
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="n0"))
    def command(args, **kwargs):
        return (f"JobState=RUNNING NumNodes=2 UserId=user({os.getuid()}) NodeList=n[0-1]"
                if "job" in args else "n0\nn1\n")
    monkeypatch.setattr(slurm_runtime.subprocess, "check_output", command)
    assert slurm_runtime.require_node()["nodes"] == ["n0", "n1"]
    plan["config"]["nodes"][arm]["solver"] = "outside"
    with pytest.raises(RuntimeError, match="bound"):
        slurm_runtime.require_node()


@pytest.mark.parametrize("arm", EXPECTED)
def test_each_addback_evaluates_only_its_own_final_checkpoint(monkeypatch, tmp_path, arm):
    bound(tmp_path, arm)
    monkeypatch.setattr(slurm_runtime, "execution_environment", lambda _: {})
    monkeypatch.setattr(slurm_runtime, "require_node", lambda: {})
    def incomplete(root, *, arms):
        assert root == tmp_path and arms == (arm,)
        raise ValueError("Own 144 checkpoint incomplete")
    monkeypatch.setattr(comparison, "final_pairs", incomplete)
    with pytest.raises(ValueError, match="Own 144"):
        comparison.evaluate_final(tmp_path, arm, None)


@pytest.mark.parametrize("arm", EXPECTED)
@pytest.mark.parametrize("key", KEYS)
def test_hidden_extra_mechanisms_are_rejected(tmp_path, arm, key):
    spec = comparison.role_spec(config(tmp_path, arm), arm, "solver", tmp_path / "data.json", {})
    spec[key] = ("bounded" if spec[key] == "unit" else "unit") if key == KEYS[0] else not spec[key]
    with pytest.raises(ValueError, match="mechanisms"):
        comparison.validate_method(spec)


def test_original_sao_stays_blocked_and_cannot_be_relabelled(tmp_path):
    spec = comparison.role_spec(config(tmp_path, "sao"), "sao", "solver", tmp_path / "data.json", {})
    with pytest.raises(NotImplementedError, match="ten-task"):
        comparison.require_executable_spec(spec)
    spec.update(execution_mode="standalone-ma-ablation-v1",
                baseline_definition=comparison.STANDALONE_ABLATION)
    with pytest.raises(ValueError, match="baseline"):
        comparison.validate_method(spec)
