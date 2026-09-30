from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.rewrite_compare import compare, paired_edits


class RewriteComparisonTests(unittest.TestCase):
    def test_math_token_budget_mismatch_is_rejected_before_comparing_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            control, candidate = Path(directory) / "control", Path(directory) / "candidate"
            common = {"initial_model": "same-base", "data_sha256": "same-data", "math_questions": 64}
            for root, spec in ((control, common), (candidate, {**common, "math_tokens": 8192})):
                root.mkdir()
                (root / "COMPLETE.json").write_text("{}")
                (root / "spec.json").write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "math_tokens"):
                compare(control, candidate)

    def report(self):
        return {
            "examples": 2, "samples_per_example": 2, "temperature": 0.7, "max_tokens": 1024,
            "results": [{"id": str(i), "sample": j, "seed": i * 10 + j,
                         "legal": True, "instruction_following": i == 1}
                        for i in range(2) for j in range(2)],
        }

    def test_groups_repeated_samples_by_example(self):
        control = self.report()
        candidate = deepcopy(control)
        candidate["results"][0]["instruction_following"] = True
        result = paired_edits(control, candidate, "instruction_following")
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["candidate_minus_control"], 0.25)
        self.assertEqual(result["improved_questions"], 1)

    def test_rejects_seed_or_protocol_mismatch(self):
        control = self.report()
        candidate = deepcopy(control)
        candidate["temperature"] = 0.8
        with self.assertRaisesRegex(ValueError, "protocol"):
            paired_edits(control, candidate, "legal")
        candidate = deepcopy(control)
        candidate["results"][0]["seed"] += 1234
        with self.assertRaisesRegex(ValueError, "identities"):
            paired_edits(control, candidate, "legal")


if __name__ == "__main__":
    unittest.main()
