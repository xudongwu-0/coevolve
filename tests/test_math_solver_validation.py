from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.math_solver_validation import (
    select_checkpoint,
    validate_spec,
    validation_rows,
)
from sao_bridge.math_solver_pair_gate import evaluate_pair
from sao_bridge.common import write_json
from sao_bridge.retool import harness_spec


def evaluation(values):
    return {
        "accuracy": sum(values) / len(values),
        "question_count": len(values),
        "samples_per_question": 1,
        "metric": "mean@1",
        "results": [
            {
                "id": f"q{index}",
                "reward": value,
                "sample_index": 0,
                "rollout_seed": index,
            }
            for index, value in enumerate(values)
        ],
    }


class SolverValidationTests(unittest.TestCase):
    def base_spec(self, root):
        model, data = root / "model", root / "data"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        data.mkdir()
        rows = [{"id": f"id-{index}"} for index in range(200)]
        write_json(data / "train.json", rows)
        write_json(data / "reward.json", [{"id": f"reward-{i}"} for i in range(8)])
        write_json(data / "dev.json", [{"id": f"dev-{i}"} for i in range(8)])
        return {
            "run": str(root / "run"),
            "model": str(model),
            "data": str(data),
            "seed": 42,
            "actor_steps": 4,
            "critic_warmup_steps": 2,
            "batch_size": 4,
            "lr": 2e-6,
            "tokens": 128,
            "save_interval_steps": 3,
            "selection_questions": 4,
            "selection_samples": 1,
            "report_dev_offset": 0,
            "report_dev_questions": 4,
            "report_train_tail": 4,
            "report_samples": 1,
        }

    def test_spec_and_split_are_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = validate_spec(self.base_spec(Path(temporary)))
            selection, report = validation_rows(spec)
        self.assertEqual(len(selection), 4)
        self.assertEqual(len(report), 8)
        self.assertFalse(spec["terminal_episodes"])
        self.assertEqual(spec["critic_updates_per_step"], 2)

    def test_terminal_mode_and_critic_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.base_spec(Path(temporary))
            with self.assertRaisesRegex(ValueError, "require concat"):
                validate_spec({**spec, "terminal_episodes": True})
            normalized = validate_spec({
                **spec, "terminal_episodes": True, "solver_trajectory_style": "concat",
            })
            with self.assertRaisesRegex(ValueError, "Critic-free"):
                validate_spec({**spec, "critic_mode": "none"})
            with self.assertRaisesRegex(ValueError, "must be boolean"):
                validate_spec({**spec, "terminal_episodes": 1})
        self.assertTrue(normalized["terminal_episodes"])

    def test_scaling_cannot_consume_held_out_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.base_spec(Path(temporary))
            spec.update(actor_steps=49, critic_warmup_steps=1, save_interval_steps=25)
            with self.assertRaisesRegex(ValueError, "held-out train tail"):
                validation_rows(validate_spec(spec))

    def test_validation_uses_current_six_setting_h0(self):
        from sao_bridge.math_solver_validation import INITIAL_SOURCE

        parsed = harness_spec(INITIAL_SOURCE)
        self.assertEqual(parsed["MAX_TURNS"], 4)
        self.assertEqual(parsed["TOOL_OUTPUT_LIMIT"], 2048)

    def test_checkpoint_interval_must_divide_total_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.base_spec(Path(temporary))
            spec["save_interval_steps"] = 4
            with self.assertRaisesRegex(ValueError, "must divide"):
                validate_spec(spec)

    def test_selection_retains_control_without_positive_bound(self):
        control = evaluation([0, 1, 0, 1])
        candidate = evaluation([1, 0, 0, 1])
        result = select_checkpoint(
            control,
            [{"global_step": 3, "path": "/checkpoint", "evaluation": candidate}],
            seed=42,
        )
        self.assertFalse(result["selected_trained_checkpoint"])
        self.assertIsNone(result["selected"]["path"])

    def test_selection_uses_qualified_best_checkpoint(self):
        control = evaluation([0] * 20)
        weaker = evaluation([1] * 12 + [0] * 8)
        stronger = evaluation([1] * 18 + [0] * 2)
        candidates = [
            {"global_step": 3, "path": "/weak", "evaluation": weaker},
            {"global_step": 7, "path": "/strong", "evaluation": stronger},
        ]
        with patch(
            "sao_bridge.math_solver_validation.paired_lower_bound",
            side_effect=[0.1, 0.2],
        ):
            result = select_checkpoint(control, candidates, seed=42)
        self.assertTrue(result["selected_trained_checkpoint"])
        self.assertEqual(result["selected"]["path"], "/strong")

    def test_pair_gate_requires_robust_gain(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = []
            for seed, delta, lower in ((42, 0.02, -0.01), (43, 0.06, 0.02)):
                path = root / str(seed)
                paths.append(path)
                write_json(path / "spec.json", {
                    "seed": seed, "run": str(path), "actor_steps": 64,
                })
                write_json(path / "COMPLETE.json", {
                    "status": "complete",
                    "harness": "fixed_h0",
                    "seed": seed,
                    "test_split_opened": False,
                    "selected_trained_checkpoint": True,
                    "paired_report": {
                        "candidate_minus_control": delta,
                        "paired_question_bootstrap_95pct": [lower, delta + 0.03],
                    },
                })
            result = evaluate_pair(paths, {
                "required_seeds": [42, 43],
                "minimum_seed_delta": 0.0,
                "minimum_mean_delta": 0.03,
                "minimum_significant_seeds": 1,
            })
        self.assertTrue(result["gate_passed"])
        self.assertAlmostEqual(result["mean_candidate_minus_control"], 0.04)


if __name__ == "__main__":
    unittest.main()
