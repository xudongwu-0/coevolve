from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_pair_gate import evaluate_pair
from sao_bridge.common import write_json


class AlfworldSolverPairGateTests(unittest.TestCase):
    def make_run(
        self,
        root: Path,
        seed: int,
        delta: float,
        lower_bound: float,
    ) -> Path:
        run = root / f"seed-{seed}"
        run.mkdir()
        write_json(run / "spec.json", {
            "run": str(run),
            "seed": seed,
            "actor_steps": 8,
            "batch_size": 7,
        })
        write_json(run / "COMPLETE.json", {
            "status": "complete",
            "seed": seed,
            "harness": "fixed_no_patch",
            "selected_trained_checkpoint": True,
            "test_split_opened": False,
            "paired_report": {
                "candidate_minus_control": delta,
                "paired_question_bootstrap_95pct": [lower_bound, delta + 0.1],
            },
        })
        return run

    def protocol(self) -> dict:
        return {
            "required_seeds": [42, 43],
            "minimum_seed_delta": 0.0,
            "minimum_mean_delta": 0.03,
            "minimum_significant_seeds": 1,
        }

    def test_pair_passes_only_with_training_seed_robustness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.make_run(root, 42, 0.04, 0.01)
            right = self.make_run(root, 43, 0.03, -0.02)
            result = evaluate_pair([left, right], self.protocol())
        self.assertTrue(result["gate_passed"])
        self.assertAlmostEqual(result["mean_candidate_minus_control"], 0.035)

    def test_pair_rejects_a_regressing_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = self.make_run(root, 42, 0.08, 0.01)
            right = self.make_run(root, 43, -0.01, -0.04)
            result = evaluate_pair([left, right], self.protocol())
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["checks"]["all_noncollapsing"])


if __name__ == "__main__":
    unittest.main()
