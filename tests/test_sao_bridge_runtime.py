"""CPU integration contracts requiring the isolated SAO runtime."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from fastapi.testclient import TestClient
from safetensors.torch import save_file
import torch
import yaml

from sao_bridge.common import INITIAL_SOURCE
from sao_bridge.service import create_app
from sao_bridge.train import make_config, upstream_trainer, zero_terminal_value
from sao_bridge.verify import compare


class FakeSolver:
    def __init__(self):
        self.calls = []

    def evaluate(self, source, rows):
        self.calls.append(source)
        return {"accuracy": 0.5 if source == INITIAL_SOURCE else 0.75}


class EvaluatorTests(unittest.TestCase):
    def test_relative_reward_cache_and_invalid_action(self):
        solver = FakeSolver()
        client = TestClient(create_app(solver, [{"id": "test"}], INITIAL_SOURCE))
        self.assertTrue(client.get("/health").json()["ready"])
        candidate = INITIAL_SOURCE.replace("MAX_TURNS = 2", "MAX_TURNS = 3")
        for _ in range(2):
            response = client.post("/score", json={"source": candidate})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["reward"], 0.25)
        self.assertEqual(solver.calls, [INITIAL_SOURCE, candidate])
        invalid = client.post("/score", json={"source": "print('invalid')"}).json()
        self.assertFalse(invalid["valid"])
        self.assertEqual(invalid["reward"], -1)
        self.assertIn("error", invalid)
        self.assertEqual(len(solver.calls), 2)

    def test_model_failure_is_not_a_zero_reward(self):
        solver = FakeSolver()
        client = TestClient(create_app(solver, [{"id": "test"}], INITIAL_SOURCE))
        with patch.object(solver, "evaluate", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                client.post("/score", json={"source": INITIAL_SOURCE})


class UpstreamConfigTests(unittest.TestCase):
    def test_both_roles_load_with_native_sao_settings(self):
        from areal.api.cli_args import PPOConfig, load_expr_config

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SAO_LOCAL_ADMIN_TOKEN": "test-only-not-a-live-service"}
        ):
            run = Path(directory)
            for role in ("editor", "solver"):
                config_path = run / f"{role}.yaml"
                config_path.write_text(yaml.safe_dump(make_config(
                    ROOT / "assets/Qwen2.5-1.5B-Instruct", run, role, 2, 4
                )))
                config, _ = load_expr_config(["--config", str(config_path)], PPOConfig)
                self.assertEqual(config.scheduler.type, "local")
                self.assertEqual(config.gconfig.n_samples, 1)
                self.assertIsNotNone(config.critic)
                self.assertFalse(config.critic.scheduling_strategy.fork)
                self.assertTrue(config.actor.use_decoupled_loss)
                rejection = config.actor.rejection_sampling
                self.assertEqual(
                    (rejection.level, rejection.action, rejection.lower, rejection.upper),
                    ("token", "mask", 0.3, 5.0),
                )

    def test_constructor_failure_cleans_owned_workers(self):
        calls = []

        class Scheduler:
            def delete_workers(self, reverse_order):
                calls.append(reverse_order)

        class FailingTrainer:
            def __init__(self, *args, **kwargs):
                self._init_scheduler()
                raise RuntimeError("startup failed")

            def _init_scheduler(self):
                return Scheduler()

        with patch("areal.PPOTrainer", FailingTrainer):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                with upstream_trainer(None, None):
                    self.fail("Constructor should not succeed")
        self.assertEqual(calls, [True])


class CheckpointTests(unittest.TestCase):
    def test_requires_finite_actual_changes_and_value_head(self):
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory) / "before", Path(directory) / "after"
            before.mkdir()
            after.mkdir()
            key = "model.layers.0.self_attn.q_proj.weight"
            values = {key: torch.zeros(2, 2), "score.weight": torch.zeros(1, 2)}
            save_file(values, before / "model.safetensors")
            save_file(values, after / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "No persisted parameter change"):
                compare(before, after)
            values["score.weight"] = torch.ones(1, 2)
            save_file(values, after / "model.safetensors")
            self.assertTrue(compare(before, after, require_score=True)["score.weight"]["changed"])
            values[key][0, 0] = float("nan")
            save_file(values, after / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                compare(before, after)
            save_file({key: torch.ones(2, 2)}, after / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "no value/score head"):
                compare(before, after, require_score=True)


class TerminalBootstrapTests(unittest.TestCase):
    def actor(self):
        from areal.api.cli_args import PPOActorConfig
        from areal.trainer.ppo.actor import PPOActor

        return PPOActor(PPOActorConfig(
            adv_norm=None, kl_ctl=0.0, discount=1.0, gae_lambda=1.0,
        ), None)

    def trajectory(self, length, width, reward=1.0):
        attention = torch.arange(width).unsqueeze(0) < length
        loss_mask = attention.long()
        loss_mask[:, :2] = 0
        return {
            "input_ids": torch.ones((1, width), dtype=torch.long),
            "attention_mask": attention,
            "loss_mask": loss_mask,
            "logprobs": torch.zeros((1, width)),
            "prox_logp": torch.zeros((1, width)),
            "rewards": torch.tensor([reward]),
            "values": torch.full((1, width), 3.0),
        }

    def valid_returns(self, trajectory, terminal):
        if terminal:
            trajectory = zero_terminal_value(trajectory)
        result = self.actor()._compute_advantages(trajectory)
        return result["returns"][result["loss_mask"].bool()]

    def test_legacy_bootstrap_depends_on_batch_padding(self):
        torch.testing.assert_close(
            self.valid_returns(self.trajectory(5, 5), False),
            torch.full((3,), 4.0),
        )
        torch.testing.assert_close(
            self.valid_returns(self.trajectory(5, 6), False),
            torch.ones(3),
        )

    def test_terminal_returns_match_reward_for_padded_and_unpadded_episodes(self):
        for reward in (0.0, 1.0, -1.0):
            for width in (5, 6, 9):
                with self.subTest(reward=reward, width=width):
                    result = self.valid_returns(
                        self.trajectory(5, width, reward), True,
                    )
                    torch.testing.assert_close(result, torch.full((3,), reward))

    def test_terminal_values_preserve_action_states_and_input(self):
        trajectory = self.trajectory(5, 7)
        corrected = zero_terminal_value(trajectory)
        torch.testing.assert_close(trajectory["values"], torch.full((1, 7), 3.0))
        torch.testing.assert_close(
            corrected["values"], torch.tensor([[3., 3., 3., 3., 0., 3., 3.]]),
        )
        self.assertIs(corrected["input_ids"], trajectory["input_ids"])

    def test_terminal_returns_cross_masked_tool_observations(self):
        trajectory = self.trajectory(9, 9)
        trajectory["loss_mask"][:, 4:7] = 0
        trajectory["values"] = torch.tensor([[8., 3., -2., 5., 9., -7., 4., 6., 13.]])
        torch.testing.assert_close(
            self.valid_returns(trajectory, True), torch.ones(4),
        )

    def test_critic_free_and_invalid_attention(self):
        trajectory = self.trajectory(5, 6)
        del trajectory["values"]
        self.assertIs(zero_terminal_value(trajectory), trajectory)
        trajectory = self.trajectory(5, 6)
        trajectory["attention_mask"][:, 1] = False
        with self.assertRaisesRegex(ValueError, "right-padded"):
            zero_terminal_value(trajectory)


if __name__ == "__main__":
    unittest.main()
