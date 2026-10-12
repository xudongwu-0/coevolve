from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_core_quick import summarize_proposals


class CoreQuickTests(unittest.TestCase):
    def test_invalid_slots_count_and_best_is_separate_from_mean(self):
        rows = [{"sample_index": i, "valid": i < 2} for i in range(8)]
        scores = [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        reports = {f"official-{i:02d}": {"accuracy": score} for i, score in enumerate(scores)}
        result = summarize_proposals(reports, {"official": rows})["official"]
        self.assertEqual(result["proposal_mean_accuracy"], 0.625)
        self.assertEqual(result["best_selection_accuracy"], 1.0)
        self.assertEqual(result["valid_proposals"], 2)
        self.assertEqual(result["selected_label"], "official-00")

    def test_missing_or_reordered_slots_are_rejected(self):
        for indices in (list(range(7)), [1, 0, 2, 3, 4, 5, 6, 7]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                summarize_proposals({}, {"official": [
                    {"sample_index": i, "valid": True} for i in indices
                ]})
