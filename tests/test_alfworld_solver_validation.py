from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_validation import (
    no_patch_assignments,
    select_checkpoint,
    subset_report,
    validate_spec,
)
from sao_bridge.alfworld_w_train import training_spec
from sao_bridge.common import write_json


def report(task_ids, rewards):
    return {
        "accuracy": sum(rewards) / len(rewards),
        "question_count": len(task_ids),
        "samples_per_question": 1,
        "metric": "mean@1",
        "tokens": 10 * len(task_ids),
        "tool_calls": len(task_ids),
        "tool_errors": 0,
        "truncated_fraction": 0.0,
        "boxed_fraction": 1.0,
        "results": [
            {
                "id": f"alfworld-{task_id}",
                "task_id": task_id,
                "reward": reward,
                "tokens": 10,
                "tool_calls": 1,
                "tool_errors": 0,
                "truncated": False,
                "boxed": True,
                "sample_index": 0,
                "rollout_seed": task_id,
            }
            for task_id, reward in zip(task_ids, rewards)
        ],
    }


class AlfworldSolverValidationTests(unittest.TestCase):
    def base_spec(self, root):
        model = root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"weights")
        no_patch = root / "no_patch.json"
        write_json(no_patch, {"benchmark": "alfworld", "actions": []})
        data = root / "train.json"
        write_json(data, [
            {
                "id": f"row-{index}",
                "task_id": index,
                "task_name": f"alfworld-h0-b{index // 10 * 10:03d}",
                "warmup": False,
            }
            for index in range(16)
        ])
        return {
            "run": str(root / "run"),
            "initial_solver": str(model),
            "no_patch": str(no_patch),
            "training_data": str(data),
            "seed": 42,
            "actor_steps": 2,
            "critic_warmup_steps": 0,
            "batch_size": 4,
            "learning_rate": 1e-6,
            "critic_learning_rate": 5e-6,
            "actor_gpus": 4,
            "rollout_gpus": 4,
            "save_interval_steps": 2,
            "selection_task_ids": [240, 241],
            "report_task_ids": [300, 301],
        }

    def test_spec_preserves_split_isolation(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = validate_spec(self.base_spec(Path(temporary)))
        self.assertEqual(spec["total_steps"], 2)
        self.assertEqual(spec["critic_updates_per_step"], 1)
        self.assertFalse(spec["terminal_episodes"])

    def test_terminal_mode_requires_complete_concat_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = self.base_spec(Path(temporary))
            with self.assertRaisesRegex(ValueError, "require concat"):
                validate_spec({**spec, "terminal_episodes": True})
            normalized = validate_spec({
                **spec, "terminal_episodes": True, "solver_trajectory_style": "concat",
            })
        self.assertTrue(normalized["terminal_episodes"])

    def test_no_patch_assignments_partition_train_tasks(self):
        assignments = no_patch_assignments(Path("/tmp/no-patch.json"))
        self.assertEqual(len(assignments), 24)
        self.assertEqual(
            [task_id for row in assignments for task_id in row["task_ids"]],
            list(range(240)),
        )

    def test_subset_report_recomputes_metrics(self):
        full = report([240, 241, 300], [1, 0, 1])
        selected = subset_report(full, [240, 241])
        self.assertEqual(selected["question_count"], 2)
        self.assertEqual(selected["accuracy"], 0.5)
        self.assertEqual(selected["tokens"], 20)

    def test_checkpoint_selection_can_retain_w0(self):
        control = report([240, 241], [1, 0])
        candidate = report([240, 241], [0, 1])
        result = select_checkpoint(
            control,
            [{"global_step": 1, "path": "/w1", "evaluation": candidate}],
            seed=42,
        )
        self.assertFalse(result["selected_trained_checkpoint"])

    def test_checkpoint_selection_uses_best_qualified(self):
        control = report([240, 241], [0, 0])
        first = report([240, 241], [1, 0])
        second = report([240, 241], [1, 1])
        with patch(
            "sao_bridge.alfworld_solver_validation.paired_lower_bound",
            side_effect=[0.1, 0.2],
        ):
            result = select_checkpoint(control, [
                {"global_step": 3, "path": "/w1", "evaluation": first},
                {"global_step": 7, "path": "/w2", "evaluation": second},
            ], seed=42)
        self.assertEqual(result["selected"]["path"], "/w2")

    def test_w_train_forwards_speed_and_checkpoint_controls(self):
        config = {
            "solver_model": "/models/w0",
            "phase": "w1",
            "experiment": "validation",
            "steps": 8,
            "batch_size": 6,
            "harness": "/data/no-patch.json",
            "data": "/data/train.json",
            "actor_gpus": 6,
            "rollout_gpus": 2,
            "save_interval_steps": 4,
            "critic_updates_per_step": 1,
            "critic_mode": "none",
            "terminal_episodes": True,
            "solver_trajectory_style": "concat",
            "actor_gradient_checkpointing": False,
            "critic_gradient_checkpointing": False,
        }
        result = training_spec(config, Path("/runs/w1"))
        self.assertEqual(result["save_interval_steps"], 4)
        self.assertEqual(result["critic_updates_per_step"], 1)
        self.assertEqual(result["critic_mode"], "none")
        self.assertTrue(result["terminal_episodes"])
        self.assertEqual(result["solver_trajectory_style"], "concat")
        self.assertFalse(result["actor_gradient_checkpointing"])
        self.assertFalse(result["critic_gradient_checkpointing"])


if __name__ == "__main__":
    unittest.main()
