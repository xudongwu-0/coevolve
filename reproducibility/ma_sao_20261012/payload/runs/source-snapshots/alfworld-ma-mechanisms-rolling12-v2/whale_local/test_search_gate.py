"""Budget and provenance invariants for the prospective SearchQA gate."""
import unittest

from search_gate import plan_routes
from teacher_gate import sha_json


def fixture(labels):
    rows = [{"example_index": i, "task_id": str(i), "reward": float(label == "success")}
            for i, label in enumerate(labels)]
    cases = [{"sample_index": i, "visible": "observed tool output"}
             for i, label in enumerate(labels) if label != "success"]
    payload = {"cases": cases}
    request = {"payload": payload, "request_digest": sha_json(payload)}
    response = {"request_digest": request["request_digest"], "annotations": [
        {"sample_index": i, "label": label, "confidence": 0.9,
         "rationale": "Predicted route from visible evidence", "evidence_quote": "observed tool output"}
        for i, label in enumerate(labels) if label != "success"]}
    return rows, request, response


class SearchGateBudgetTests(unittest.TestCase):
    def test_uniform_uncertainty_recovers_uniform_budget(self):
        plan = plan_routes(*fixture(["uncertain"] * 64), seed=43, total_rollouts=256)
        self.assertEqual(plan["followup_counts"], [3] * 64)
        self.assertTrue(plan["retain_all_verified_successes"])

    def test_priority_moves_budget_without_excluding_other_tasks(self):
        plan = plan_routes(*fixture(["weight"] * 16 + ["harness"] * 16 + ["success"] * 32),
                           seed=43, total_rollouts=256)
        self.assertEqual(len(plan["followup_schedule"]), 192)
        self.assertEqual(len(set(map(tuple, plan["followup_schedule"]))), 192)
        self.assertGreater(min(plan["followup_counts"][:16]), max(plan["followup_counts"][16:32]))
        self.assertLessEqual(max(plan["followup_counts"]), 7)

    def test_extreme_allocations_never_overrun_budget_or_cap(self):
        for total in [8, 32, 64]:
            plan = plan_routes(*fixture(["weight"] + ["harness"] * 7), seed=43, total_rollouts=total)
            self.assertEqual(8 + len(plan["followup_schedule"]), total)
            self.assertTrue(all(0 <= n <= 7 for n in plan["followup_counts"]))

    def test_stale_or_ungrounded_annotations_are_rejected(self):
        rows, request, response = fixture(["weight", "harness"])
        response["request_digest"] = "0" * 64
        with self.assertRaises(AssertionError):
            plan_routes(rows, request, response, seed=43, total_rollouts=8)
        response["request_digest"] = request["request_digest"]
        response["annotations"][0]["evidence_quote"] = "not in visible evidence"
        with self.assertRaises(AssertionError):
            plan_routes(rows, request, response, seed=43, total_rollouts=8)


if __name__ == "__main__":
    unittest.main()
