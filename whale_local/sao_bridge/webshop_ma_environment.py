"""Offline/native WebShop environment seam, independent of the MA-SAO optimizer."""

from .webshop_reward import WebShopRewardConfig, reward_receipt


class MASAOWebShopEnvironment:
    """Wrap an injected real WebShop environment; no model or service starts here."""

    def __init__(self, environment, *, reward_config=WebShopRewardConfig(), max_steps=15):
        if type(max_steps) is not int or not 1 <= max_steps <= 30:
            raise ValueError("WebShop action budget must be explicitly bounded")
        self.environment = environment
        self.reward_config = reward_config
        self.max_steps = max_steps
        self.active = False

    def reset(self, task_id):
        observation, instruction, manifest = self.environment.reset(task_id)
        self.task_id, self.manifest = task_id, manifest
        self.actions, self.invalid_actions = 0, 0
        self.score, self.done, self.active = 0.0, False, True
        return observation, instruction, manifest

    def available_actions(self):
        if not self.active:
            raise ValueError("Reset the task before reading available actions")
        return self.environment.available_actions()

    def step(self, action):
        if not self.active or self.done or self.actions >= self.max_steps:
            raise ValueError("Cannot execute outside the active bounded episode")
        # Exceptions propagate: do not convert infra failure to a scored failure.
        observation, score, done, info = self.environment.step(action)
        if type(done) is not bool or type(score) not in (int, float) or not 0 <= score <= 1:
            raise ValueError("Unexpected real WebShop transition")
        if not isinstance(info, dict) or type(info.get("invalid")) is not bool:
            raise ValueError("The environment must explicitly identify invalid actions")
        self.actions += 1
        self.invalid_actions += int(info["invalid"])
        self.score, self.done = float(score), done
        return observation, score, done, {
            **info, "environment_actions": self.actions, "invalid_actions": self.invalid_actions,
        }

    def outcome(self, stop_reason=None):
        if not self.active or not self.actions:
            raise ValueError("No real environment episode to score")
        reason = stop_reason or ("terminal" if self.done else "max_steps")
        if reason == "max_steps" and self.actions != self.max_steps:
            raise ValueError("Do not label an early stop as exhausted environment budget")
        receipt = reward_receipt(
            task_score=self.score, terminated=self.done, invalid_actions=self.invalid_actions,
            environment_actions=self.actions, stop_reason=reason, config=self.reward_config,
        )
        return {
            "benchmark": "webshop", "task_id": self.task_id, "task_manifest": self.manifest,
            "reward": receipt["team_training_reward"], "task_score": self.score,
            "success": receipt["success"], "webshop_reward_receipt": receipt,
        }

    def close(self):
        return self.environment.close()
