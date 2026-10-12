import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sao_bridge.analysis import async_overhead, sample_ledger
from sao_bridge.common import read_json, write_json
from sao_bridge.phase_recovery import cached_evaluation, editor_pair_audit, validate_resume
from sao_bridge.replay import recycle_pool


class RecoveryTests(unittest.TestCase):
    def fixture(self, root, *, extra=True):
        spec = dict(editor_steps=1, batch_size=1, reward_questions=1)
        write_json(root / "spec.json", spec)
        write_json(root / "h1_editor_data.json", [{"id": "a"}, {"id": "b"}])
        for index, name in enumerate(("a", "b")):
            write_json(root / f"h1/receipts/{name}.json", {
                "id": name, "role": "editor", "replayed": False, "evaluation": {
                    "request_index": index, "source": "source", "reference_source": "reference",
                },
            })
        call = dict(prompt_ids=[1], response_ids=[2], response_logprobs=[-0.5],
                    temperature=1.0)
        for index, name in enumerate(("a", "b", "a") if extra else ("a", "b")):
            for role in ("candidate", "reference"):
                write_json(root / f"h1_reward_evals/{index}-{role}.json", {
                    "request_index": index, "sample_id": name, "pair_role": role,
                    "harness_source": "source" if role == "candidate" else "reference",
                    "question_count": 1, "samples_per_question": 1,
                    "results": [{"id": "q", "reward": 1.0, "calls": [call]}],
                })
        return spec

    def test_extra_pairs_are_metered_but_never_recycled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root)
            audit = editor_pair_audit(root, "h1", spec)
            self.assertEqual(audit["unused_request_indices"], [2])
            self.assertEqual(audit["editor_rollouts"], 1)
            self.assertEqual(audit["harness_scoring_rollouts"], 2)
            self.assertFalse(audit["strict_sample_budget_matched"])
            self.assertEqual(len(audit["raw_files_sha256"]), 6)
            pool = recycle_pool(root / "h1_reward_evals",
                                request_indices=set(audit["committed_request_indices"]))
            self.assertEqual(len(pool), 4)
            write_json(root / "h1_pair_audit.json", audit)
            self.assertEqual(async_overhead(root)["generated_total"], 3)
            ledger = sample_ledger(root)
            self.assertEqual(ledger["editor_rollouts"], 3)
            self.assertEqual(ledger["harness_scoring_rollouts"], 6)
            self.assertEqual(ledger["generated_total"], 9)
            self.assertEqual(len(list((root / "h1_reward_evals").glob("*.json"))), 6)

    def test_missing_pairs_duplicates_and_formal_overhead_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "Formal"):
                editor_pair_audit(root, "h1", {**spec, "formal_baseline": True})
            (root / "h1_reward_evals/0-reference.json").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete"):
                editor_pair_audit(root, "h1", spec)
            self.fixture(root)
            write_json(root / "h1/receipts/extra.json",
                       read_json(root / "h1/receipts/a.json"))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                editor_pair_audit(root, "h1", spec)

    def test_exact_run_has_no_overhead(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root, extra=False)
            audit = editor_pair_audit(root, "h1", spec)
            self.assertTrue(audit["strict_sample_budget_matched"])

    def test_cached_report_requires_all_question_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            report = dict(question_count=1, samples_per_question=2,
                          results=[{"id": "a"}, {"id": "a"}])
            write_json(path, report)
            self.assertEqual(cached_evaluation(path, [{"id": "a"}], 2), report)
            with self.assertRaisesRegex(ValueError, "mismatched"):
                cached_evaluation(path, [{"id": "b"}], 2)

    def test_resume_only_completed_same_spec_phases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = self.fixture(root, extra=False)
            write_json(root / "STATUS.json", {"status": "failed", "elapsed_seconds": 123.0})
            with self.assertRaisesRegex(ValueError, "partial optimizer"):
                validate_resume(root, spec)
            write_json(root / "h1/trained.json", {"actor_steps": 1, "critic_only_warmup_steps": 0})
            write_json(root / "h1_spec.json", {"steps": 1})
            (root / "h1/metrics.jsonl").write_text(json.dumps({"global_step": 0}) + "\n")
            with patch("sao_bridge.analysis.verify_phases") as verify:
                self.assertEqual(validate_resume(root, spec), 123.0)
                verify.assert_called_once_with(root)
            with self.assertRaisesRegex(ValueError, "specification"):
                validate_resume(root, {**spec, "editor_steps": 2})


if __name__ == "__main__":
    unittest.main()
