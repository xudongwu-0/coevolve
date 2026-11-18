"""WebShop reward variants for unchanged MA-SAO learners and benchmark scoring."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class WebShopRewardConfig:
    mode: str = "terminal"
    invalid_action_cost: float = 0.0
    partial_credit_weight: float = 0.0

    def __post_init__(self):
        if self.mode not in {"terminal", "validity_cost", "partial_credit"}:
            raise ValueError("Unknown MA-SAO WebShop reward variant")
        for name in ("invalid_action_cost", "partial_credit_weight"):
            value = getattr(self, name)
            if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= .2:
                raise ValueError(name + " must be finite and in[0,.2]")
        if self.mode != "partial_credit" and self.partial_credit_weight:
            raise ValueError("Partial matching credit requires an explicit partial_credit experiment")
        if self.mode == "terminal" and self.invalid_action_cost:
            raise ValueError("Terminal control cannot silently add action penalties")


def reward_receipt(*, task_score, terminated, invalid_actions, environment_actions,
                   stop_reason, config=WebShopRewardConfig()):
    if type(task_score) not in (float, int) or not math.isfinite(task_score) or not 0 <= task_score <= 1:
        raise ValueError("Original WebShop task score must be in[0,1]")
    if type(terminated) is not bool or type(invalid_actions) is not int or type(environment_actions) is not int:
        raise ValueError("Invalid outcome or action accounting types")
    if not 0 <= invalid_actions <= environment_actions or environment_actions < 1:
        raise ValueError("Invalid action counters")
    if stop_reason not in {"terminal", "max_steps", "context_exhausted", "token_budget"}:
        raise ValueError("Infrastructure errors are not valid model-failure reward outcomes")
    if (stop_reason == "terminal") != terminated:
        raise ValueError("Terminal status and stop reason disagree")
    success = terminated and task_score == 1.0
    partial = config.partial_credit_weight * task_score if terminated and not success else 0.0
    cost = config.invalid_action_cost * invalid_actions / environment_actions
    return {
        "schema": "ma_sao_webshop_reward_v1", "benchmark": "webshop",
        "task_score": float(task_score), "terminated": terminated, "success": success,
        "environment_actions": environment_actions, "invalid_actions": invalid_actions,
        "stop_reason": stop_reason, "config": asdict(config),
        "terminal_reward": float(success), "partial_credit": partial, "invalid_cost": cost,
        "team_training_reward": float(success) + partial - cost,
        "benchmark_metrics_unchanged": True,
    }


def verified_reward(receipt):
    if receipt.get("schema") != "ma_sao_webshop_reward_v1":
        raise ValueError("Unsupported WebShop reward receipt")
    rebuilt = reward_receipt(
        task_score=receipt["task_score"], terminated=receipt["terminated"],
        invalid_actions=receipt["invalid_actions"], environment_actions=receipt["environment_actions"],
        stop_reason=receipt["stop_reason"], config=WebShopRewardConfig(**receipt["config"]),
    )
    if rebuilt != receipt:
        raise ValueError("WebShop reward receipt was modified or inconsistently computed")
    return rebuilt["team_training_reward"]
