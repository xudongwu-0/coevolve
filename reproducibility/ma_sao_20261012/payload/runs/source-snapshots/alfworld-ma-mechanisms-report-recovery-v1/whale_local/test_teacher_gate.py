"""Meaningful gate checks: stale labels, invented evidence and budget bypasses."""
from copy import deepcopy
import unittest
from teacher_gate import make_request, validate_response, plan_routes
from test_failure_gate import trajectory


def example(i, reward=0):
    r = trajectory(i, key=str(i), reward=reward)
    r.update(answer_redacted=True, answer="")
    r["policy_calls"][0].update(messages=[{"role": "user", "content": "visible board"}], content="model move")
    return r


def annotations(request, labels):
    return {"request_digest": request["request_digest"], "annotations": [
        {"sample_index": c["sample_index"], "label": label, "confidence": 0.8,
         "rationale": "Routing hypothesis from the visible failure.", "evidence_quote": "model move"}
        for c, label in zip(request["payload"]["cases"], labels)]}


class TeacherGateSafety(unittest.TestCase):
    def setUp(self):
        self.rows = [example(0), example(1), example(2), example(3, reward=1)]
        self.request = make_request(self.rows)
        self.response = annotations(self.request, ["harness", "weight", "scoring_ambiguity"])

    def test_current_evidence_binding_and_no_solutions(self):
        self.assertNotIn("prompt_ids", str(self.request))
        stale = deepcopy(self.response); stale["request_digest"] = "0" * 64
        with self.assertRaises(AssertionError): validate_response(self.request, stale)
        changed = deepcopy(self.rows); changed[0]["harness_sha256"] = "new"
        with self.assertRaises(AssertionError):
            plan_routes(changed, self.request, self.response, seed=42, rollouts_per_puzzle=4, audit_pairs=2)

    def test_annotations_cannot_invent_evidence_or_targets(self):
        bad = deepcopy(self.response); bad["annotations"][0]["evidence_quote"] = "not present"
        with self.assertRaises(AssertionError): validate_response(self.request, bad)
        bad = deepcopy(self.response); bad["annotations"][0]["teacher_target"] = "correct answer"
        with self.assertRaises(AssertionError): validate_response(self.request, bad)
        bad = deepcopy(self.response); bad["annotations"][1]["sample_index"] = 0
        with self.assertRaises(AssertionError): validate_response(self.request, bad)

    def test_teacher_routes_actually_limit_followup_sampling(self):
        plan = plan_routes(self.rows, self.request, self.response, seed=42, rollouts_per_puzzle=4, audit_pairs=2)
        self.assertEqual([x["max_followups"] for x in plan["sampling"]], [0, 3, 0, 0])
        self.assertEqual(plan["max_small_model_rollouts"], 11)
        self.assertTrue(plan["sampling"][3]["retain_initial_success"])
        low = deepcopy(self.response); low["annotations"][1]["confidence"] = 0.1
        self.assertEqual(validate_response(self.request, low)[1]["route"], "uncertain")


if __name__ == "__main__":
    unittest.main()
