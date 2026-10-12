from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.analysis import difference_in_differences, expected_sample_ledger, paired_difference, paired_lower_bound, stage_reports
from sao_bridge.common import write_json


class PairedAnalysisTests(unittest.TestCase):
    def result(self, rewards):
        return {
            "metric": "mean@2", "samples_per_question": 2, "question_count": 2,
            "results": [{"id": str(i // 2), "reward": reward}
                        for i, reward in enumerate(rewards)],
        }

    def test_samples_are_clustered_within_questions(self):
        result = paired_difference(self.result([0, 1, 0, 0]), self.result([1, 1, 0, 1]))
        self.assertEqual(result["candidate_minus_control"], 0.5)
        self.assertEqual(result["paired_question_bootstrap_95pct"], [0.5, 0.5])
        self.assertEqual(result["question_count"], 2)
        self.assertEqual(result["improved_questions"], 2)

    def test_reject_mismatched_or_incomplete_evaluation(self):
        control = self.result([0, 1, 0, 0])
        candidate = self.result([1, 1, 0, 1])
        candidate["results"][0]["id"] = "different"
        with self.assertRaises(ValueError):
            paired_difference(control, candidate)
        with self.assertRaises(ValueError):
            paired_difference(control, self.result([0, 1, 0, float("nan")]))

    def test_conservative_selection_bound_uses_paired_questions(self):
        control = {"results": [{"id": str(i), "reward": 0.0} for i in range(32)]}
        better = {"results": [{"id": str(i), "reward": 1.0} for i in range(32)]}
        self.assertGreater(paired_lower_bound(control, better, alpha=0.05 / 8), 0)
        same = {"results": [{"id": str(i), "reward": float(i % 2)} for i in range(32)]}
        self.assertLessEqual(paired_lower_bound(same, same, alpha=0.05 / 8), 0)

    def test_difference_in_differences_removes_initial_offset(self):
        base0 = self.result([0, 0, 0, 0])
        base1 = self.result([0, 1, 0, 0])
        ma0 = self.result([1, 0, 0, 0])
        ma1 = self.result([1, 1, 1, 0])
        result = difference_in_differences(base0, base1, ma0, ma1)
        self.assertEqual(result["ma_change_minus_sao_change"], 0.25)
        self.assertEqual(result["estimand"], "(MA_final - MA_initial) - (SAO_final - SAO_initial)")

    def test_exact_physical_sample_ledger_for_control_and_ma(self):
        base = dict(method="independent", cycles=1, batch_size=16,
                    editor_batch_size=16, editor_steps=3, proposals=8,
                    solver_batch_size=64, solver_steps=32, reward_questions=16,
                    select_questions=128, select_samples=2, report_questions=128,
                    report_train_tail=384, report_samples=2)
        control = expected_sample_ledger(base)
        self.assertEqual(control["generated_total"], 15056)
        self.assertEqual(control["harness_scoring_rollouts"], 4096)
        ma = expected_sample_ledger(dict(base, method="ma_sao", replay_fraction=0.25,
            critic_warmup_steps=8, warmup_recycled_per_row=2,
            solver_recycle_fraction=0.5))
        self.assertEqual(ma["generated_total"], 14976)
        self.assertEqual(ma["recycled_partner_episodes_reused"], 2048)

    def test_retained_incumbent_uses_its_completed_final_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / "spec.json", {"cycles": 1, "proposal_source": "final_policy"})
            write_json(run / "h1_before_report.json", self.result([0, 0, 0, 0]))
            write_json(run / "h1_selection.json", {"selected": {"incumbent": True}})
            self.assertEqual(list(stage_reports(run)), ["h0w0"])
            final = run / "h1_selected_report.json"
            write_json(final, self.result([1, 0, 0, 0]))
            self.assertEqual(stage_reports(run)["h1w0"], final)

    def test_legacy_incumbent_may_reuse_before_report(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            write_json(run / "spec.json", {"cycles": 1})
            before = run / "h1_before_report.json"
            write_json(before, self.result([0, 0, 0, 0]))
            write_json(run / "h1_selection.json", {"selected": {"incumbent": True}})
            self.assertEqual(stage_reports(run)["h1w0"], before)


if __name__ == "__main__":
    unittest.main()
