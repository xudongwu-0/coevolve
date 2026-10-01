from pathlib import Path
import sys
import tempfile
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_eval import validate_config
from sao_bridge.common import write_json


class AlfworldSolverEvaluationTests(unittest.TestCase):
    def test_config_requires_distinct_official_task_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            no_patch = root / "no_patch.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            config = {
                "run": str(root / "run"),
                "solver_model": str(model),
                "solver_version": "w0",
                "task_ids": [300, 301],
                "no_patch": str(no_patch),
                "seed": 42,
            }
            normalized = validate_config(config)
            self.assertEqual(normalized["samples"], 1)
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_config({**config, "task_ids": [300, 300]})


if __name__ == "__main__":
    unittest.main()
