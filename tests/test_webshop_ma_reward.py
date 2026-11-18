import copy

import pytest

from whale_local.sao_bridge.joint_reward import role_reward
from whale_local.sao_bridge.webshop_reward import WebShopRewardConfig, reward_receipt, verified_reward
from whale_local.sao_bridge.webshop_ma_environment import MASAOWebShopEnvironment


def receipt(**changes):
    arguments = dict(task_score=1., terminated=True, invalid_actions=0, environment_actions=5,
                     stop_reason="terminal")
    arguments.update(changes)
    return reward_receipt(**arguments)


def test_existing_binary_ma_sao_role_rewards_are_unchanged():
    for role in ("editor", "solver"):
        for value in (0., 1.):
            assert role_reward(role, value, None) == value
    with pytest.raises(ValueError):
        role_reward("solver", .5, None)


def test_default_webshop_control_is_terminal_only():
    result = receipt(task_score=.8)
    assert result["task_score"] == .8
    assert result["team_training_reward"] == 0
    assert not result["success"]


def test_invalid_cost_does_not_relabel_benchmark_success():
    result = receipt(invalid_actions=2,
                     config=WebShopRewardConfig(mode="validity_cost", invalid_action_cost=.1))
    assert result["success"] and result["task_score"] == 1.
    assert result["team_training_reward"] == pytest.approx(.96)
    for role in ("editor", "solver"):
        assert role_reward(role, .96, None, webshop_receipt=result) == pytest.approx(.96)


def test_partial_matching_credit_is_explicit_and_never_success():
    result = receipt(task_score=.8, config=WebShopRewardConfig(mode="partial_credit", partial_credit_weight=.1))
    assert not result["success"]
    assert result["team_training_reward"] == pytest.approx(.08)
    with pytest.raises(ValueError, match="explicit"):
        WebShopRewardConfig(partial_credit_weight=.1)


def test_timeout_does_not_get_final_purchase_partial_credit():
    result = receipt(task_score=.8, terminated=False, stop_reason="max_steps",
                     config=WebShopRewardConfig(mode="partial_credit", partial_credit_weight=.1))
    assert result["team_training_reward"] == 0


def test_engineer_validity_gate_keeps_its_original_meaning():
    result = receipt()
    assert role_reward("editor", 1., "invalid harness", "validity_gated", webshop_receipt=result) == 0
    assert role_reward("solver", 1., "invalid harness", "validity_gated", webshop_receipt=result) == 1


def test_modified_reward_receipt_is_rejected():
    result = receipt()
    result["team_training_reward"] = .5
    with pytest.raises(ValueError, match="modified"):
        verified_reward(result)
    with pytest.raises(ValueError):
        role_reward("solver", .5, None, webshop_receipt=result)


@pytest.mark.parametrize("changes", [
    {"task_score": float("nan")}, {"invalid_actions": -1}, {"invalid_actions": 6},
    {"environment_actions": True}, {"stop_reason": "http_503"},
    {"terminated": False}, {"task_score": 1.1},
])
def test_invalid_or_infrastructure_outcomes_cannot_be_scored(changes):
    with pytest.raises(ValueError):
        receipt(**changes)


class Environment:
    def reset(self, task):
        self.n = 0
        return "start", "instruction", {"task": task}

    def available_actions(self):
        return {"clickables": ["Buy Now"]}

    def step(self, action):
        self.n += 1
        return "following", .8, self.n == 2, {"invalid": action == "invalid"}

    def close(self):
        pass


def test_ma_environment_counts_invalid_actions_and_prevents_post_terminal_step():
    env = MASAOWebShopEnvironment(Environment(), max_steps=3)
    env.reset(12)
    env.step("invalid")
    env.step("click[Buy Now]")
    outcome = env.outcome()
    assert outcome["reward"] == 0 and outcome["task_score"] == .8
    assert outcome["webshop_reward_receipt"]["invalid_actions"] == 1
    with pytest.raises(ValueError):
        env.step("click[Buy Now]")
    env.reset(12)
    assert env.invalid_actions == 0 and env.actions == 0


def test_no_free_retry_or_early_budget_exhaustion():
    env = MASAOWebShopEnvironment(Environment(), max_steps=3)
    env.reset(12)
    env.step("invalid")
    with pytest.raises(ValueError, match="early"):
        env.outcome()
    assert env.outcome(stop_reason="token_budget")["webshop_reward_receipt"]["environment_actions"] == 1


def test_infrastructure_exception_propagates_without_fake_reward():
    class Broken(Environment):
        def step(self, action):
            raise ConnectionError("HTTP503")
    env = MASAOWebShopEnvironment(Broken())
    env.reset(12)
    with pytest.raises(ConnectionError):
        env.step("search[x]")
    with pytest.raises(ValueError, match="No real"):
        env.outcome()
