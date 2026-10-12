from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.joint_round import checked_behavior
from sao_bridge.joint_updates import likelihood_summary, prepare_joint_update, teammate_weight


def trajectory(log_shift=0.):
    return {
        "input_ids": torch.tensor([[8, 9, 10, 11, 12]]),
        "loss_mask": torch.tensor([[0, 0, 0, 1, 1]]),
        "logprobs": torch.tensor([[0., 0., 0., -0.1, -0.2]]),
        "prox_logp": torch.tensor([[0., 0., -0.1 + log_shift, -0.2 + log_shift, 0.]]),
        "ref_logp": torch.tensor([[0., 0., -0.1, -0.2, 0.]]),
        "versions": torch.tensor([[-1, -1, -1, 0, 1]]),
        "joint_round": torch.tensor([1]), "joint_slot": torch.tensor([0]),
    }


def test_summary_matches_action_alignment_not_prompt_tokens():
    result = likelihood_summary(trajectory(0.1), 0, 1)
    assert result["action_tokens"] == 2
    assert result["log_importance_sum"] == pytest.approx(0.2)
    assert result["log_importance_mean"] == pytest.approx(0.1)
    assert result["fixed_reference_kl"] == pytest.approx(math.exp(-0.1) + 0.1 - 1)
    assert result["behavior_version_min"] == 0
    assert result["behavior_version_max"] == 1


def test_peer_weights_are_bounded_and_identity_checked():
    data = {"version": 1, "slot": 0, "log_importance_sum": 1000.}
    assert teammate_weight(data, 1, 0) == (2., True)
    assert teammate_weight({**data, "log_importance_sum": -1000.}, 1, 0) == (0.5, True)
    with pytest.raises(ValueError):
        teammate_weight(data, 2, 0)
    with pytest.raises(ValueError):
        teammate_weight({**data, "log_importance_sum": float("nan")}, 1, 0)


def test_two_learners_exchange_proximal_policies_before_updates(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as workers:
        editor = workers.submit(prepare_joint_update, trajectory(0.), tmp_path, "editor", 1)
        solver = workers.submit(prepare_joint_update, trajectory(0.1), tmp_path, "solver", 1)
        eweights, estats = editor.result(timeout=5)
        sweights, sstats = solver.result(timeout=5)
    assert eweights == pytest.approx([math.exp(0.2)])
    assert sweights == [1.]
    assert estats["peer_effective_sample_fraction"] == 1
    assert sstats["fixed_reference_available"] == 1
    assert len(list((tmp_path / "round-0001").glob("*-proximal-000.json"))) == 2
    with pytest.raises(ValueError, match="ordered round"):
        prepare_joint_update(trajectory(), tmp_path, "editor", 2)


def test_async_behavior_allows_only_true_one_round_lag():
    calls = [{"response_ids": [10, 11], "response_logprobs": [-0.1, -0.2],
              "versions": [0, 1]}]
    checked_behavior(calls, 1, max_lag=1)
    with pytest.raises(ValueError):
        checked_behavior(calls, 1)
    with pytest.raises(ValueError):
        checked_behavior(calls, 2, max_lag=1)


def test_unit_ablation_preserves_exchange_and_raw_diagnostics(tmp_path):
    with ThreadPoolExecutor(max_workers=2) as workers:
        editor = workers.submit(prepare_joint_update, trajectory(0.), tmp_path, "editor", 1, "unit")
        solver = workers.submit(prepare_joint_update, trajectory(10.), tmp_path, "solver", 1, "unit")
        eweights, estats = editor.result(timeout=5)
        sweights, sstats = solver.result(timeout=5)
    assert eweights == sweights == [1.]
    assert estats['peer_raw_importance_mean'] == 2.
    assert estats['peer_raw_clipped_fraction'] == 1.
    assert estats['peer_importance_clipped_fraction'] == 0.
    assert estats['peer_unit_weight_mode'] == sstats['peer_unit_weight_mode'] == 1.
    assert len(list((tmp_path / 'round-0001').glob('*-proximal-000.json'))) == 2
    with pytest.raises(ValueError, match='Unknown teammate'):
        prepare_joint_update(trajectory(), tmp_path, 'editor', 1, 'typo')


@pytest.mark.parametrize("enabled,version,expected", [(False, 2, "bounded"),
                                                     (True, 1, "bounded"),
                                                     (True, 2, "unit"),
                                                     (True, 3, "bounded")])
def test_materialized_actor_worker_selects_anchor_weight_mode(monkeypatch, enabled, version, expected):
    from types import SimpleNamespace
    from unittest.mock import Mock
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))
    from sao_bridge.terminal_actor import TerminalFSDPPPOActor
    monkeypatch.setenv("MA_SAO_ASYNC_JOINT_ROOT", "/unused-test-root")
    monkeypatch.setenv("MA_SAO_JOINT_ROLE", "solver")
    monkeypatch.setenv("MA_SAO_PEER_WEIGHT_MODE", "bounded")
    if enabled:
        monkeypatch.setenv("MA_SAO_ANCHOR_SCHEDULE", "1")
    else:
        monkeypatch.delenv("MA_SAO_ANCHOR_SCHEDULE", raising=False)
    exchange = Mock(return_value=([1.], {}))
    monkeypatch.setattr("sao_bridge.joint_updates.prepare_joint_update", exchange)
    actor = SimpleNamespace(get_version=lambda: version,
        actor=SimpleNamespace(_compute_advantages=lambda data, meta: {"advantages": torch.ones(1, 2)}))
    result = TerminalFSDPPPOActor._terminal_advantages(actor, {}, None)
    assert exchange.call_args.kwargs["weight_mode"] == expected
    assert torch.equal(result["advantages"], torch.ones(1, 2))
