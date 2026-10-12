import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_engineer_eval import validate_job_task_ids
from sao_bridge.alfworld_strong_refresh import packet_from_baseline, successful_job_output
from sao_bridge.budget import contract_digest
from sao_bridge.common import file_sha, read_json, write_json


class StrongRefreshTests(unittest.TestCase):
    def test_training_baselines_require_explicit_purpose_and_never_open_dev(self):
        request = {"kind": "baseline", "task_ids": list(range(10))}
        with self.assertRaises(ValueError):
            validate_job_task_ids(request)
        self.assertEqual(
            validate_job_task_ids({**request, "purpose": "training_evidence"}), list(range(10)),
        )
        for changed in (
            {"task_ids": [240]}, {"task_ids": [True]}, {"task_ids": [0, 0]},
            {"kind": "score"}, {"purpose": "report"},
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_job_task_ids({**request, "purpose": "training_evidence", **changed})
        self.assertEqual(
            validate_job_task_ids({"kind": "score", "task_ids": [240, 300]}), [240, 300],
        )
        with self.assertRaises(ValueError):
            validate_job_task_ids({"kind": "score", "task_ids": [400]})

    @staticmethod
    def fixture(root, *, all_pass=False):
        baseline = root / "baseline"
        baseline.mkdir(parents=True)
        write_json(baseline / "run_config.json", {"official": "baseline"})
        (baseline / "runs.jsonl").write_text('{"index":0,"output":{"fixture":true}}\n')
        source = SimpleNamespace(run_root=baseline, batch_tag="baseline")
        scores = {index: float(all_pass or index % 2) for index in range(10)}
        metadata = {
            "benchmark": "alfworld", "batch_tag": "baseline", "source_run_root": str(baseline),
            "split": "test_set", "start": 0, "end": 10, "batch_size": 10,
            "baseline_pass": int(sum(scores.values())), "baseline_rewards": scores,
            "reward_threshold": 1.0, "target_agent_name": "qwen35-9b-nothink",
            "target_model": "policy", "response_protocol": "prefill_think_patch",
            "schema_style": "alfworld_life_multihook_v1",
        }
        record = {
            "prompt": [{"role": "system", "content": "fixture"},
                       {"role": "user", "content": "authentic failure trace fixture"}],
            "metadata": metadata,
        }
        builder = Mock()
        builder.iter_batch_sources.return_value = [source]
        builder.build_record.return_value = record
        state = {"solver": "/models/w1", "solver_version": "w1"}
        return scores, record, builder, state

    def test_compaction_binds_solver_scores_and_frozen_raw_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, record, builder, state = self.fixture(root)
            before = (root / "baseline/runs.jsonl").read_bytes()
            with patch.dict("sys.modules", {"build_alfworld_dataset": builder}):
                packet, metadata = packet_from_baseline(
                    root, list(range(10)), scores, state, "weights-w1", "request-sha",
                )
            self.assertEqual(packet["messages"], record["prompt"])
            self.assertEqual(metadata["solver_version"], "w1")
            self.assertEqual(metadata["solver_weights_sha256"], "weights-w1")
            self.assertEqual(metadata["baseline_rewards"], scores)
            self.assertEqual(metadata["task_ids"], list(range(10)))
            self.assertEqual(
                metadata["adapter_baseline_identity_sha256"],
                file_sha(root / "baseline_identity.json"),
            )
            self.assertEqual((root / "baseline/runs.jsonl").read_bytes(), before)
            args = builder.build_record.call_args.args[0]
            self.assertEqual((args.max_traces, args.max_obs_chars, args.max_assistant_chars), (4, 220, 180))
            self.assertEqual(args.max_approx_tokens, 3000)
            self.assertFalse(args.include_incomplete)
            self.assertEqual(read_json(root / "baseline_identity.json")["solver_model"], "/models/w1")

    def test_all_pass_keeps_calibration_without_fabricating_failure_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, _, builder, state = self.fixture(root, all_pass=True)
            builder.build_record.return_value = None
            with patch.dict("sys.modules", {"build_alfworld_dataset": builder}):
                packet, metadata = packet_from_baseline(
                    root, list(range(10)), scores, state, "weights-w1", "request-sha",
                )
            self.assertIsNone(packet)
            self.assertEqual(metadata["baseline_pass"], 10)
            self.assertEqual(metadata["baseline_rewards"], scores)

    def test_real_failures_or_changed_rewards_cannot_be_silently_discarded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scores, record, builder, state = self.fixture(root)
            original = copy.deepcopy(record)
            builder.build_record.return_value = None
            with (
                patch.dict("sys.modules", {"build_alfworld_dataset": builder}),
                self.assertRaisesRegex(ValueError, "genuine training failures"),
            ):
                packet_from_baseline(root, list(range(10)), scores, state, "weights-w1", "request-sha")
            record = copy.deepcopy(original)
            record["metadata"]["baseline_rewards"][0] = 1.0
            builder.build_record.return_value = record
            with (
                patch.dict("sys.modules", {"build_alfworld_dataset": builder}),
                self.assertRaisesRegex(ValueError, "baseline rewards"),
            ):
                packet_from_baseline(root, list(range(10)), scores, state, "weights-w1", "request-sha")

    def test_successful_baseline_attempt_requires_exact_request_and_raw_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            request = {"kind": "baseline", "purpose": "training_evidence", "task_ids": list(range(10))}
            result = {"status": "complete", "request_sha256": contract_digest(request), "scores": {}}
            write_json(root / "cohort/request.json", request)
            write_json(root / "cohort/attempt-01/FAILED.json", {"error": "infra"})
            write_json(root / "cohort/attempt-02/result.json", result)
            self.assertEqual(successful_job_output(root, "cohort", result), root / "cohort/attempt-02")
            with self.assertRaisesRegex(ValueError, "identity"):
                successful_job_output(root, "cohort", {**result, "request_sha256": "wrong"})
            write_json(root / "cohort/attempt-01/result.json", result)
            with self.assertRaisesRegex(ValueError, "unique"):
                successful_job_output(root, "cohort", result)


if __name__ == "__main__":
    unittest.main()
