from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_engineer_training import run
from sao_bridge.common import read_json, write_json


class AlfworldEngineerTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.phase = self.root / "runs/engineer/training"
        self.model = self.root / "model"
        write_json(self.model / "provenance.json", {"model": "fixture"})
        self.data = self.root / "data.json"
        write_json(self.data, [{"id": "packet"}])
        self.config = self.root / "config.json"
        write_json(self.config, {
            "run": str(self.phase), "data": str(self.data),
            "solver_model": str(self.model), "engineer_model": str(self.model),
        })
        for name, value in (
            ("ROOT", self.root),
            ("gpu_release_audit", lambda: {
                "foreign_gpu_pids": [], "filler_kind": "adaptive", "filler_unit_active": True,
            }),
        ):
            override = patch(f"sao_bridge.alfworld_engineer_training.{name}", value)
            override.start()
            self.addCleanup(override.stop)
        launcher = patch("sao_bridge.alfworld_engineer_training.Launcher")
        self.launch = launcher.start().return_value
        self.addCleanup(launcher.stop)
        self.launch.wait.side_effect = self.complete

    def complete(self, *args, **kwargs):
        write_json(self.phase / "COMPLETE.json", {"status": "completed", "fixture": True})

    def test_completion_is_reconciled_without_repeating_training(self):
        result = run(self.config)
        self.assertEqual(result["status"], "completed")
        root = self.phase.parent
        (root / "TRAINING_COMPLETE.json").unlink()
        write_json(root / "STATUS.json", {"stage": "failed"})
        write_json(root / "FAILED.json", {"error": "interrupted publication"})
        self.assertEqual(run(self.config), result)
        self.assertEqual(self.launch.module.call_count, 1)
        self.assertEqual(
            read_json(root / "STATUS.json")["stage"], "training_complete_pending_evaluation",
        )
        self.assertFalse((root / "FAILED.json").exists())

    def test_failure_is_archived_and_retried_under_the_same_identity(self):
        def failed(*args, **kwargs):
            write_json(self.phase / "FAILED.json", {"error": "worker aborted"})
            raise RuntimeError("worker aborted")

        self.launch.wait.side_effect = failed
        with self.assertRaisesRegex(RuntimeError, "worker aborted"):
            run(self.config)
        self.launch.wait.side_effect = self.complete
        run(self.config)
        root = self.phase.parent
        self.assertTrue((root / "failed_training_attempts/attempt-01/FAILED.json").is_file())
        attempts = read_json(root / "training_attempts.json")
        self.assertEqual(len(attempts), 2)
        self.assertIn("error", attempts[0])
        self.assertTrue(attempts[1]["complete"])

    def test_frozen_input_drift_and_foreign_gpu_work_are_not_silently_retried(self):
        run(self.config)
        write_json(self.data, [{"id": "changed"}])
        with self.assertRaisesRegex(ValueError, "frozen"):
            run(self.config)
        write_json(self.data, [{"id": "packet"}])
        with patch("sao_bridge.alfworld_engineer_training.gpu_release_audit", return_value={
            "foreign_gpu_pids": [123], "filler_kind": "adaptive", "filler_unit_active": True,
        }):
            with self.assertRaisesRegex(RuntimeError, "free GPUs"):
                run(self.config)


if __name__ == "__main__":
    unittest.main()
