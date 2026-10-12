from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_core_extend import merge_reports, subset_report, validate_cohort
from sao_bridge.alfworld_strong_phase import stable_id_seed_offset


class CoreExtensionTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "protocol": "alfworld-core200-untrained-oneshot-v1",
            "old_task_ids": list(range(240, 264)), "new_task_ids": list(range(264, 440)),
            "all_task_ids": list(range(240, 440)),
            "strict_goal_disjoint_task_ids": list(range(240, 437)),
            "untrained_instance_but_train_goal_related_task_ids": [437, 438, 439],
            "training_task_ids_excluded": list(range(240)),
            "old_sample_index_to_reuse": 0, "samples_per_question": 1,
            "seed": 6000045, "solver_tokens": 4096,
        }

    def report(self, ids, samples, old=False):
        rows = [
            {
                "id": f"alfworld-{task}", "task_id": task, "sample_index": sample,
                "rollout_seed": 6000045 + stable_id_seed_offset(f"alfworld-{task}") + 1009 * sample,
                "reward": sample if old else 1, "tokens": 12,
            }
            for task in ids for sample in range(samples)
        ]
        return {
            "metric": f"mean@{samples}", "samples_per_question": samples,
            "question_count": len(ids), "results": rows,
            "accuracy": sum(row["reward"] for row in rows) / len(rows),
        }

    def test_exact_200_reuses_first_sample_not_best_or_mean(self):
        old = self.report(self.protocol["old_task_ids"], 2, old=True)
        new = self.report(self.protocol["new_task_ids"], 1)
        result = merge_reports(old, new, self.protocol)
        self.assertEqual(result["question_count"], 200)
        self.assertEqual(result["accuracy"], 176 / 200)
        self.assertEqual(len(result["results"]), 200)
        self.assertTrue(all(row["sample_index"] == 0 for row in result["results"]))
        strict = subset_report(result, self.protocol["strict_goal_disjoint_task_ids"])
        self.assertEqual(strict["accuracy"], 173 / 197)
        self.assertEqual(strict["question_count"], 197)

    def test_training_overlap_wrong_count_and_wrong_sample_are_rejected(self):
        for change in (
            {"training_task_ids_excluded": [240]},
            {"new_task_ids": list(range(264, 439))},
            {"old_sample_index_to_reuse": 1},
            {"strict_goal_disjoint_task_ids": list(range(240, 438))},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_cohort({**self.protocol, **change})

    def test_duplicate_or_wrong_seed_native_samples_are_rejected(self):
        old = self.report(self.protocol["old_task_ids"], 2, old=True)
        new = self.report(self.protocol["new_task_ids"], 1)
        bad = deepcopy(new)
        bad["results"][-1] = deepcopy(bad["results"][0])
        with self.assertRaises(ValueError):
            merge_reports(old, bad, self.protocol)
        bad = deepcopy(new)
        bad["results"][0]["rollout_seed"] += 1
        with self.assertRaises(ValueError):
            merge_reports(old, bad, self.protocol)
