from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_engineer_data import evaluation_ids
from sao_bridge.alfworld_train_data import goal_group
from harness_r1_alfworld.driver import TYPES


class AlfworldEngineerDataTests(unittest.TestCase):
    def fixture(self):
        paths = [
            f"json_2.1.1/train/{TYPES[index % 6]}-goal-{index}/trial-0/game.tw-pddl"
            for index in range(500)
        ]
        return paths, {
            "selection_task_ids": list(range(240, 300)),
            "report_task_ids": list(range(300, 400)),
        }

    def test_evaluation_excludes_all_training_goal_siblings(self):
        paths, partitions = self.fixture()
        paths[240] = paths[0].replace("trial-0", "trial-1")
        paths[300] = paths[1].replace("trial-0", "trial-2")
        result = evaluation_ids(paths, partitions, list(range(240)))
        self.assertEqual(result, evaluation_ids(paths, partitions, list(range(240))))
        self.assertEqual(len(result["selection_task_ids"]), 24)
        self.assertEqual(len(result["report_task_ids"]), 99)
        self.assertNotIn(240, result["selection_task_ids"])
        self.assertNotIn(300, result["report_task_ids"])
        self.assertEqual({
            goal_group(paths[index]).split("-")[0] for index in result["selection_task_ids"]
        }, set(TYPES))

    def test_small_or_duplicate_training_and_evaluation_sets_are_rejected(self):
        paths, partitions = self.fixture()
        with self.assertRaisesRegex(ValueError, "distinct training"):
            evaluation_ids(paths, partitions, [0, 0])
        with self.assertRaisesRegex(ValueError, "Not enough"):
            evaluation_ids(paths, {**partitions, "selection_task_ids": [240]}, [0])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            evaluation_ids(
                paths, {**partitions, "report_task_ids": list(range(300, 401))},
                list(range(240)),
            )


if __name__ == "__main__":
    unittest.main()
