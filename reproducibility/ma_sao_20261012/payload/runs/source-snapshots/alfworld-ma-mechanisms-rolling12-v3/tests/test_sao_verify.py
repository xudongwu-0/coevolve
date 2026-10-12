from pathlib import Path
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.verify import compare


class CheckpointVerificationTests(unittest.TestCase):
    key = "model.layers.0.self_attn.q_proj.weight"

    def checkpoints(self, root, before, after):
        left, right = root / "before", root / "after"
        left.mkdir()
        right.mkdir()
        save_file({self.key: before}, left / "model.safetensors")
        save_file({self.key: after}, right / "model.safetensors")
        return left, right

    def test_precision_conversion_alone_is_not_an_optimizer_update(self):
        with tempfile.TemporaryDirectory() as directory:
            before = torch.full((2, 2), 1.0001, dtype=torch.float32)
            paths = self.checkpoints(Path(directory), before, before.to(torch.bfloat16))
            with self.assertRaisesRegex(RuntimeError, "precision conversion"):
                compare(*paths)

    def test_real_change_survives_precision_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            before = torch.full((2, 2), 1.0001, dtype=torch.float32)
            after = before.to(torch.bfloat16)
            after[0, 0] += 0.125
            result = compare(*self.checkpoints(Path(directory), before, after))[self.key]
            self.assertTrue(result["changed"])
            self.assertEqual(result["max_abs_delta"], 0.125)
            self.assertEqual(result["saved_dtype"], "torch.bfloat16")

    def test_same_precision_and_nonfinite_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            before = torch.ones(2, 2)
            after = before + 0.25
            paths = self.checkpoints(Path(directory), before, after)
            self.assertEqual(compare(*paths)[self.key]["max_abs_delta"], 0.25)
            save_file({self.key: torch.full((2, 2), float("nan"))},
                      paths[1] / "model.safetensors")
            with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
                compare(*paths)


if __name__ == "__main__":
    unittest.main()
