import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_strong_cost import (
    allocation_window, logged_updates, matrix_work, native_generation, proposal_work, reward_work,
)
from sao_bridge.common import file_sha, write_json
from sao_bridge.evaluation import stable_id_seed_offset


class StrongCostTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def native_fixture(self):
        optimizer = self.root / "training/optimizer"
        write_json(optimizer / "consumed_rows.json", {
            "rows": 2, "ids": ["a", "b"], "fresh_ids": ["a", "b"], "replay_ids": [],
        })
        receipt = {
            "id": "a", "role": "editor", "replayed": False, "trajectory_style": "concat",
            "exported_fresh_trajectories": 1, "recycled_partner_episodes": [],
            "generated_tokens": 2,
            "evaluation": {"evaluation": {"eval_status": "invalid_patch_treated_as_no_patch"}},
            "calls": [{"response_ids": [1, 2], "response_logprobs": [-1, -1],
                       "versions": [0, 0], "stop_reason": "length"}],
        }
        for index, identity in enumerate(("a", "b", "a")):
            write_json(optimizer / f"receipts/{index}.json", {**receipt, "id": identity})
        write_json(optimizer.parent / "COMPLETE.json", {"status": "completed"})
        return optimizer, receipt

    def test_prefetch_and_invalid_generations_count_without_picking_a_duplicate(self):
        optimizer, _ = self.native_fixture()
        before = file_sha(optimizer / "receipts/0.json")
        result = native_generation(optimizer)
        self.assertEqual(result["declared_input_rows"], 2)
        self.assertEqual(result["recorded_generated_episodes"], 3)
        self.assertEqual(result["recorded_generated_tokens"], 6)
        self.assertEqual(result["repeated_input_ids"], {"a": 2})
        self.assertEqual(result["extra_receipts_over_declared_rows"], 1)
        self.assertEqual(result["episode_statuses"]["invalid_patch_treated_as_no_patch"], 3)
        self.assertIn("cannot identify", result["consumed_behavior_mapping"])
        self.assertEqual(file_sha(optimizer / "receipts/0.json"), before)

    def test_rejects_unallocated_replay_and_inconsistent_native_token_accounting(self):
        optimizer, receipt = self.native_fixture()
        for changes in (
            {"id": "unallocated"}, {"replayed": True}, {"generated_tokens": 3},
            {"exported_fresh_trajectories": 2}, {"recycled_partner_episodes": ["borrowed"]},
        ):
            with self.subTest(changes=changes):
                write_json(optimizer / "receipts/0.json", {**receipt, **changes})
                with self.assertRaises(ValueError):
                    native_generation(optimizer)
        changed = copy.deepcopy(receipt)
        changed["calls"][0]["response_logprobs"] = [-1]
        write_json(optimizer / "receipts/0.json", changed)
        with self.assertRaisesRegex(ValueError, "logprob"):
            native_generation(optimizer)

    def test_closed_training_cannot_omit_a_declared_input_receipt(self):
        optimizer, _ = self.native_fixture()
        (optimizer / "receipts/1.json").unlink()
        with self.assertRaisesRegex(ValueError, "omitted"):
            native_generation(optimizer)
        (optimizer.parent / "COMPLETE.json").unlink()
        self.assertEqual(native_generation(optimizer)["missing_declared_input_ids"], ["b"])

    def matrix_fixture(self, root, *, failed=False):
        config = {"task_ids": [240, 241], "samples": 2, "seed": 42, "solver_tokens": 8}
        write_json(root / "config.json", config)
        rows = [{
            "id": f"alfworld-{task}", "task_id": task, "sample_index": sample,
            "rollout_seed": 42 + stable_id_seed_offset(f"alfworld-{task}") + 1009 * sample,
            "reward": float(sample), "tokens": 3,
        } for task in config["task_ids"] for sample in range(2)]
        write_json(root / "reports/physical-harness.json", {
            "metric": "mean@2", "samples_per_question": 2, "question_count": 2,
            "accuracy": 0.5, "results": rows,
        })
        if failed:
            write_json(root / "FAILED.json", {"error": "later harness failed"})
        else:
            write_json(root / "COMPLETE.json", {"status": "completed"})
            write_json(root / "report.json", {
                "unique_harnesses": 1, "logical_harnesses": 3,
                "variants": {name: {"report": str(root / "reports/physical-harness.json")}
                             for name in ("no_patch", "alias1", "alias2")},
            })

    def test_counts_physical_matrix_reports_once_but_counts_failed_attempt_work(self):
        self.matrix_fixture(self.root / "matrix/attempt-01", failed=True)
        self.matrix_fixture(self.root / "matrix/attempt-02")
        result = matrix_work(self.root)
        self.assertEqual(result["recorded_episodes"], 8)
        self.assertEqual(result["recorded_generated_tokens"], 24)
        self.assertEqual(result["failed_matrix_attempts"], 1)
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIn("lower-bound gap", result["scope"])

    def test_closed_matrix_requires_complete_physical_reports(self):
        self.matrix_fixture(self.root / "matrix/attempt-01")
        (self.root / "matrix/attempt-01/reports/physical-harness.json").unlink()
        with self.assertRaisesRegex(ValueError, "physical reports"):
            matrix_work(self.root)

    def test_proposals_include_retries_and_do_not_filter_invalid_model_outputs(self):
        for attempt in (1, 2):
            write_json(self.root / f"jobs/proposal00/attempt-{attempt}/response.json",
                       {"usage": {"completion_tokens": 4096}})
        result = proposal_work(self.root)
        self.assertEqual(result["recorded_generation_responses"], 2)
        self.assertEqual(result["recorded_generated_tokens"], 8192)

    def test_reward_cache_deduplication_retains_recorded_failed_eval_episodes(self):
        write_json(self.root / "reward_cache/baseline/key/reward.json", {
            "patched_n": 2, "timing": {"eval_attempts": [
                {"patched_n": 1, "returncode": 1}, {"patched_n": 2, "returncode": 0},
            ]},
        })
        for name in ("request1", "request2"):
            write_json(self.root / f"reward_receipts/{name}.json", {"cache": "key"})
        result = reward_work(self.root)
        self.assertEqual(result["unique_reward_request_receipts"], 2)
        self.assertEqual(result["unique_completed_reward_cache_entries"], 1)
        self.assertEqual(result["recorded_official_episodes"], 3)
        self.assertEqual(result["failed_attempts_in_completed_reward_entries"], 1)

    def test_discarded_attempt_updates_and_incomplete_log_tail_are_explicit(self):
        path = self.root / "metrics.jsonl"
        rows = [
            {"global_step": 0, "metrics": {}},
            {"global_step": 1, "metrics": {"ppo_actor/update/grad_norm": 0.5}},
        ]
        path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows) + b'{"global_step":')
        result = logged_updates(self.root)
        self.assertEqual(result["optimizer_steps"], 2)
        self.assertEqual(result["actor_steps"], 1)
        self.assertTrue(result["incomplete_final_line"])

    def test_allocation_includes_recovery_and_peer_report_waiting(self):
        result = allocation_window(10, 3610, 5410)
        self.assertEqual(result["allocated_gpu_hours"], 12)
        self.assertEqual(result["through_arm_freeze_gpu_hours"], 8)
        self.assertEqual(result["freeze_to_report_release_gpu_hours"], 4)
        for times in ((10, 9, 20), (10, 20, 19), (10, 20, float("inf"))):
            with self.subTest(times=times), self.assertRaises(ValueError):
                allocation_window(*times)


if __name__ == "__main__":
    unittest.main()
