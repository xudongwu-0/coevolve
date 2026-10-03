from pathlib import Path
from unittest.mock import patch
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from areal.utils.data import allocate_balanced_mbs
from areal.api.cli_args import MicroBatchSpec
from sao_bridge.alfworld_w2_recovery import continuation_config, wait_predecessor
from sao_bridge.alfworld_w_train import training_spec
from sao_bridge.common import write_json
from sao_bridge.large_train import configure


def config(tmp_path):
    old = {
        "solver_model": "w1", "solver_critic": "c1", "harness": "h2",
        "learning_rate": 2e-6, "critic_learning_rate": 1e-5,
        "critic_updates_per_step": 1, "critic_warmup_steps": 5,
        "terminal_episodes": True, "solver_trajectory_style": "concat",
    }
    return continuation_config(old, "w2/globalstep79", "c2/globalstep79", tmp_path)


def test_recovery_uses_actual_rejected_weights_and_preserves_learning_parameters(tmp_path):
    cfg = config(tmp_path)
    assert cfg["solver_model"] == "w2/globalstep79"
    assert cfg["solver_critic"] == "c2/globalstep79"
    assert cfg["learning_rate"] == 2e-6 and cfg["critic_learning_rate"] == 1e-5
    assert cfg["critic_updates_per_step"] == 1 and cfg["critic_warmup_steps"] == 5
    assert cfg["steps"] * cfg["batch_size"] == 640
    assert cfg["save_interval_steps"] == 20
    assert cfg["independent_trajectories"]
    with pytest.raises(ValueError, match="never the W1"):
        continuation_config({"solver_model": "w1"}, "w1", "critic", tmp_path)
    with pytest.raises(ValueError, match="step79"):
        continuation_config({"solver_model": "w1"}, "w2/globalstep39", "critic", tmp_path)


def test_single_role_recovery_repairs_sequence_isolation_without_enabling_ma(tmp_path, monkeypatch):
    monkeypatch.setenv("SAO_LOCAL_ADMIN_TOKEN", "unit-test-only")
    cfg = config(tmp_path)
    spec = training_spec(cfg, tmp_path)
    native = configure(spec, tmp_path)
    assert spec["independent_trajectories"]
    assert not spec.get("joint_root")
    assert native["actor"]["kl_ctl"] == 0
    for name in ("actor", "critic"):
        assert native[name]["mb_spec"]["n_mbs"] == 2
        mb = MicroBatchSpec(**native[name]["mb_spec"])
        for lengths in ([32768, 32000], [100, 24075]):
            assert sorted(allocate_balanced_mbs(mb, lengths)) == [[0], [1]]
    legacy = training_spec({**cfg, "independent_trajectories": False}, tmp_path)
    assert configure(legacy, tmp_path)["actor"]["mb_spec"].get("n_mbs") != 2
    bad = {**spec, "independent_trajectories": "yes"}
    with pytest.raises(ValueError, match="boolean"):
        configure(bad, tmp_path)


def test_queue_requires_released_gpu_not_predecessor_success_score(tmp_path):
    protocol = {"prerequisite": str(tmp_path), "prerequisite_label": "matched",
                "queue_timeout_seconds": 60}
    write_json(tmp_path / "matched_COMPLETE.json", {"status": "unavailable"})
    write_json(tmp_path / "matched_RELEASED.json", {})
    with patch("sao_bridge.alfworld_w2_recovery.gpu_release_audit", return_value={
        "foreign_gpu_pids": [], "filler_unit_active": True,
    }):
        wait_predecessor(protocol)
    with patch("sao_bridge.alfworld_w2_recovery.gpu_release_audit", return_value={
        "foreign_gpu_pids": [123], "filler_unit_active": True,
    }):
        with pytest.raises(RuntimeError, match="released local GPUs"):
            wait_predecessor(protocol)


def test_queue_surfaces_failed_evaluation(tmp_path):
    write_json(tmp_path / "matched_FAILED.json", {"error": "environment failed"})
    with pytest.raises(RuntimeError, match="Predecessor evaluation failed"):
        wait_predecessor({"prerequisite": str(tmp_path), "prerequisite_label": "matched",
                         "queue_timeout_seconds": 60})
