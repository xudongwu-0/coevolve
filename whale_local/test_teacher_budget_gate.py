"""Verify equal-budget accounting, routing, and reconstructed training sources."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest

from teacher_gate import make_request
from teacher_budget_gate import plan_routes, route_and_select, verify_artifacts
from test_teacher_gate import example, annotations


class EqualBudgetGate(unittest.TestCase):
    def test_budget_includes_audits_and_excludes_confident_harness_from_followups(self):
        rows = [example(0), example(1), example(2), example(3, reward=1)]
        request = make_request(rows)
        response = annotations(request, ["harness", "weight", "scoring_ambiguity"])
        plan = plan_routes(rows, request, response, seed=42, rollouts_per_puzzle=4, audit_pairs=2)
        self.assertEqual(4 + 2 * len(plan["audit_indices"]) + len(plan["followup_schedule"]), 16)
        self.assertEqual({x["sample_index"] for x in plan["followup_schedule"]}, {1})
        self.assertFalse(plan["stop_after_success"])

    def test_minimum_budget_and_no_eligible_targets(self):
        rows = [example(0), example(1)]
        request = make_request(rows)
        response = annotations(request, ["environment", "scoring_ambiguity"])
        for g in [1, 4]:
            plan = plan_routes(rows, request, response, seed=42, rollouts_per_puzzle=g, audit_pairs=8)
            self.assertEqual(2 + 2 * len(plan["audit_indices"]) + len(plan["followup_schedule"]), 2*g)
            self.assertTrue(all(x["reason"] == "diagnostic_only" for x in plan["followup_schedule"]))

    def test_full_source_reconstruction_rejects_missing_budgeted_sample(self):
        rows = [example(0), example(1), example(2), example(3, reward=1)]
        request = make_request(rows)
        response = annotations(request, ["harness", "weight", "scoring_ambiguity"])
        with TemporaryDirectory() as tmp:
            out = Path(tmp); probe = out / "probe.py"; probe.write_text("# visible-only probe")
            def teacher(actual_rows, directory):
                self.assertEqual(actual_rows, rows)
                (out / "teacher_request.json").write_text(json.dumps(request))
                (out / "teacher_response.json").write_text(json.dumps(response))
                return request, response, {}
            def replay(key, harness, seed):
                row = example(int(key), reward=int(key == "1"))
                row.update(rollout_seed=seed, harness_sha256="new" if harness == probe else "old")
                return row
            with patch("teacher_budget_gate.request_annotations", teacher):
                selected, report = route_and_select(rows, replay=replay, original_harness="old", probe_harness=probe,
                                                    out=out, rollouts_per_puzzle=4, audit_pairs=2, seed=42)
            self.assertEqual(report["actual_small_model_rollouts"], 16)
            self.assertEqual(len(selected), 9)
            self.assertEqual(verify_artifacts(out, rows)[0], selected)
            path = out / "gate_resamples.jsonl"
            path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
            with self.assertRaises(AssertionError):
                verify_artifacts(out, rows)


if __name__ == "__main__":
    unittest.main()
