from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.workflow_executor import check_trajectory_format
from areal.utils.data import batched_call
from sao_bridge.common import read_json, write_json
from sao_bridge.joint_campaign import select_pair, wait_comparison
from sao_bridge.joint_health import assess, metric_rows, recover_checkpoints, request_stop
from sao_bridge.large_train import configure


def training_spec(tmp_path):
    return {
        "role": "editor", "model": str(tmp_path), "steps": 128, "batch_size": 8,
        "lr": 5e-7, "kl_ctl": 0.005, "phase": "editor", "experiment": "joint",
        "seed": 42, "tokens": 4096, "context_tokens": 32768,
        "actor_gpus": 4, "rollout_gpus": 4, "joint_root": str(tmp_path / "joint"),
        "critic_warmup_steps": 2, "terminal_episodes": True, "save_interval_steps": 4,
        "run": str(tmp_path / "optimizer"),
    }


def metrics(step, kl=0.01):
    return {
        "global_step": step,
        "metrics": {
            "ppo_actor/update/grad_norm": 1.,
            "joint_policy/fixed_reference_kl": kl,
            "joint_policy/fixed_reference_available": 1.,
            "joint_policy/peer_importance_clipped_fraction": 0.,
            "joint_policy/reference_kl_overflow": 0.,
            "joint_value_norm/raw_value_mse": 0.2,
        },
    }


def publish_episodes(spec, step):
    current = Path(spec["joint_root"]) / f"round-{step:04d}"
    for slot in range(8):
        write_json(current / f"proposal-{slot:03d}.json", {"model_error": None})
        write_json(current / f"editor-receipt-{slot:03d}.json", {
            "reward": 1., "calls": [{"response_ids": [4]}],
        })


def test_reference_and_proximal_probabilities_have_same_resolved_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("SAO_LOCAL_ADMIN_TOKEN", "test-only")
    spec = training_spec(tmp_path)
    config = configure(spec, tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved, _ = load_expr_config(["--config", str(path)], PPOConfig)
    assert resolved.gconfig.temperature == resolved.actor.temperature == resolved.ref.temperature == 0.7
    assert resolved.actor.eps_clip == resolved.actor.eps_clip_higher == 0.2
    assert resolved.ref.optimizer is None
    assert resolved.actor.should_compute_prox_logp()


def test_joint_round_and_slot_survive_native_batching(tmp_path):
    def row(slot, length):
        return {
            "input_ids": torch.ones(1, length, dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "loss_mask": torch.ones(1, length, dtype=torch.bool),
            "logprobs": torch.zeros(1, length),
            "versions": torch.zeros(1, length, dtype=torch.long),
            "rewards": torch.tensor([1.]),
            "joint_round": torch.tensor([2]), "joint_slot": torch.tensor([slot]),
        }
    rows = [row(3, 5), row(4, 8)]
    assert all(check_trajectory_format(r) for r in rows)
    changed = batched_call(lambda data: data, rows)
    assert [int(r["joint_round"].item()) for r in changed] == [2, 2]
    assert [int(r["joint_slot"].item()) for r in changed] == [3, 4]


def test_health_requires_consecutive_soft_warnings_and_reports_nonfinite(tmp_path):
    spec = training_spec(tmp_path)
    publish_episodes(spec, 2)
    report = assess([metrics(0), metrics(1), metrics(2, 0.2)], spec)
    assert report["status"] == "healthy"
    report = assess([metrics(i, 0.2) for i in range(3)], spec)
    assert report["status"] == "stop"
    assert report["latest_team_success"] == 1
    clipped = [metrics(i) for i in range(3)]
    for row in clipped:
        row["metrics"]["joint_policy/peer_importance_clipped_fraction"] = 1.
    report = assess(clipped, spec)
    assert report["status"] == "warning"
    assert report["warnings"] and not report["stop_reasons"]
    clipped[-1]["metrics"]["joint_policy/fixed_reference_available"] = 0.
    assert assess(clipped, spec)["status"] == "stop"
    bad = metrics(2)
    bad["metrics"]["ppo_actor/update/grad_norm"] = float("nan")
    assert assess([metrics(0), metrics(1), bad], spec)["hard_errors"]
    missing = metrics(2)
    del missing["metrics"]["joint_policy/fixed_reference_kl"]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        assess([metrics(0), metrics(1), missing], spec)


def test_stop_preserves_first_reason_and_partial_metrics_are_not_committed(tmp_path):
    first = request_stop(tmp_path, {"reason": "deadline"})
    assert request_stop(tmp_path, {"reason": "later"}) == first
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"global_step":0}\n{"global_step":')
    assert metric_rows(path) == [{"global_step": 0}]


def test_recovery_only_uses_fully_committed_pairs_and_normalizer(tmp_path):
    spec = training_spec(tmp_path)
    root = Path(spec["run"]) / "engine/checkpoints/user/joint/editor"
    for role in ("actor", "critic"):
        for step in (3, 7):
            path = root / role / f"epoch0epochstep{step}globalstep{step}"
            write_json(path / "config.json", {})
            (path / "model.safetensors").write_bytes(b"test")
            if role == "critic":
                write_json(path / "joint_value_norm.json", {"updates": 4})
    with patch("sao_bridge.joint_health.ensure_model_runtime_metadata"):
        assert list(recover_checkpoints(spec, [metrics(i) for i in range(4)])) == ["3"]
        (root / "critic/epoch0epochstep3globalstep3/joint_value_norm.json").unlink()
        with pytest.raises(ValueError, match="running statistics"):
            recover_checkpoints(spec, [metrics(i) for i in range(4)])


def test_selection_never_promotes_no_patch_or_uses_200_scores():
    reports = {
        "11": {"no_patch": {"accuracy": 1.}, "proposal00": {"accuracy": 0.8}},
        "15": {"no_patch": {"accuracy": 0.9}, "proposal01": {"accuracy": 0.8}},
    }
    assert select_pair(reports) == ("11", "proposal00")
    reports["15"]["proposal01"]["accuracy"] = 0.9
    assert select_pair(reports) == ("15", "proposal01")


def test_required_grpo_failure_prevents_silent_ma_start(tmp_path):
    write_json(tmp_path / "w1_FAILED.json", {"error": "reward infrastructure"})
    with pytest.raises(RuntimeError, match="Required GRPO"):
        wait_comparison({"comparison": str(tmp_path), "training_deadline_unix": time.time() + 3600})
