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
from sao_bridge.train import make_config, upstream_trainer
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


if __name__ == "__main__":
    unittest.main()
