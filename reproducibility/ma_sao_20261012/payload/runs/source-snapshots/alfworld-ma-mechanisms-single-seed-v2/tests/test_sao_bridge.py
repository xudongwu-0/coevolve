"""CPU contracts for the task adapter, without model downloads or GPU imports."""
import ast
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge.common import (
    INITIAL_SOURCE, binary_reward, harness_spec, solver_messages, verify_upstreams,
)


class HarnessTests(unittest.TestCase):
    def test_original_template_and_fixed_answer_format(self):
        spec = harness_spec(INITIAL_SOURCE)
        messages = solver_messages(spec, "What is 2+2?")
        self.assertIn("What is 2+2?", messages[1]["content"])
        self.assertIn("\\boxed", messages[1]["content"])
        self.assertEqual(spec["MAX_TURNS"], 2)

    def test_reject_extra_format_fields(self):
        source = INITIAL_SOURCE.replace('"{question}"', '"{question} {answer}"')
        with self.assertRaises(ValueError):
            harness_spec(source)

    def test_reject_attribute_access_in_format(self):
        source = INITIAL_SOURCE.replace('"{question}"', '"{question} {question.__class__}"')
        with self.assertRaises(ValueError):
            harness_spec(source)

    def test_reject_python_methods(self):
        source = INITIAL_SOURCE + "\n    async def get_prompt_messages(self, state):\n        return []\n"
        with self.assertRaises(ValueError):
            harness_spec(source)

    def test_reject_duplicate_attribute(self):
        with self.assertRaises(ValueError):
            harness_spec(INITIAL_SOURCE + '\n    SYSTEM_PROMPT = "duplicate"\n')

    def test_reject_code_execution(self):
        source = INITIAL_SOURCE + '\nprint("unsafe")\n'
        with self.assertRaises((ValueError, AssertionError)):
            harness_spec(source)

    def test_reward_is_native_strict_boxed(self):
        self.assertEqual(binary_reward(r"\boxed{4}", "4"), 1)
        self.assertEqual(binary_reward(r"\boxed{5}", "4"), 0)
        self.assertEqual(binary_reward("4", "4"), 0)

    def test_upstream_pin_and_patch(self):
        provenance = verify_upstreams()
        self.assertEqual(len(provenance["sao"]), 40)
        self.assertEqual(len(provenance["sao_patch_sha256"]), 64)

    def test_adapter_has_no_handwritten_rl_objective(self):
        path = ROOT / "whale_local/sao_bridge/train.py"
        tree = ast.parse(path.read_text())
        self.assertTrue(any(
            isinstance(node, ast.ImportFrom) and node.module == "areal"
            and any(alias.name == "PPOTrainer" for alias in node.names)
            for node in ast.walk(tree)
        ))
        self.assertNotIn(".backward(", path.read_text())
        self.assertNotIn("optimizer.step(", path.read_text())


if __name__ == "__main__":
    unittest.main()
