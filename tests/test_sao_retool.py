import asyncio
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.retool import ContextExhausted, INITIAL_SOURCE, harness_spec, prepare_python, run_episode
from sao_bridge.benchmark import DAPO_PREFIX, DAPO_SUFFIX, normalize


class ReToolTests(unittest.TestCase):
    def test_fenced_code_and_last_expression(self):
        for language in ("python", "py", ""):
            self.assertEqual(prepare_python(f"```{language}\nx = 2\nx + 2\n```"),
                             "x = 2\nprint(x + 2)")
        self.assertEqual(prepare_python("print(4)"), "print(4)")

    def test_schema_rejects_execution(self):
        self.assertEqual(harness_spec(INITIAL_SOURCE)["MAX_TURNS"], 4)
        with self.assertRaises(ValueError):
            harness_spec(INITIAL_SOURCE + "\nprint('not allowed')\n")
        with self.assertRaises(ValueError):
            harness_spec(INITIAL_SOURCE.replace('"{question}"', '"{question.__class__}"'))

    def test_tool_observation_and_final_reward(self):
        observed = []
        async def generate(messages, tools, max_tokens, seed):
            observed.append(list(messages))
            if len(observed) == 1:
                msg = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "c", "type": "function", "function": {
                        "name": "code_interpreter", "arguments": '{"code":"2+2"}',
                    }}]}
            else:
                msg = {"role": "assistant", "content": r"\boxed{4}"}
            return {"choices": [{"message": msg}], "usage": {"completion_tokens": 10}}
        with patch("sao_bridge.retool.execute_python", return_value={"output": "4", "error": False}):
            row = asyncio.run(run_episode(
                {"id": "0", "question": "two plus two", "answer": "4"},
                harness_spec(INITIAL_SOURCE), generate, 512,
            ))
        self.assertEqual(row["reward"], 1)
        self.assertEqual(row["tokens"], 20)
        self.assertEqual(row["tool_calls"], 1)
        self.assertEqual(observed[1][-1]["role"], "tool")

    def test_instruction_wrapper_removed_for_decontamination(self):
        row = normalize({"prompt": [{"role": "user", "content": DAPO_PREFIX + "x?" + DAPO_SUFFIX}],
                         "reward_model": {"ground_truth": "-3"}}, "train")
        self.assertEqual(row["question"], "x?")
        self.assertEqual(row["answer"], "-3")

    def test_context_exhaustion_ends_episode_as_truncated(self):
        calls = []
        async def generate(messages, tools, max_tokens, seed):
            calls.append(max_tokens)
            if len(calls) == 1:
                return {"choices": [{"message": {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "c", "type": "function", "function": {
                        "name": "code_interpreter", "arguments": '{"code":"1"}'}}]}}],
                        "usage": {"completion_tokens": 10}}
            raise ContextExhausted("full")
        with patch("sao_bridge.retool.execute_python", return_value={"output": "1", "error": False}):
            row = asyncio.run(run_episode({"id": "0", "question": "q", "answer": "4"},
                                          harness_spec(INITIAL_SOURCE), generate, 512))
        self.assertTrue(row["truncated"])
        self.assertEqual(row["reward"], 0)
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
