import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import chess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge import chess_domain as domain
from sao_bridge.retool import ContextExhausted

loader = importlib.util.spec_from_file_location("prepare_sao_chess", ROOT / "scripts/prepare_sao_chess.py")
prep = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prep)


def row(line=None, fen=chess.STARTING_FEN):
    return {"id": "private-puzzle-id", "start_fen": fen,
            "solution_moves": line or ["e2e4", "e7e5", "g1f3"]}


class ChessDomainTests(unittest.TestCase):
    def setUp(self):
        self.spec = domain.harness_spec(domain.INITIAL_SOURCE)
        self.observations = []
        self.records = []

    def generator(self, actions, used=8):
        actions = iter(actions)

        async def generate(messages, tools, max_tokens, seed):
            self.assertEqual(tools, [])
            self.assertGreaterEqual(max_tokens, used)
            self.observations.append(copy.deepcopy(messages))
            record = {"prompt_token_ids": [10, 20], "response_token_ids": [30],
                      "response_logprobs": [-0.7], "seed": seed, "max_tokens": max_tokens}
            self.records.append(record)
            action = next(actions)
            message = action if isinstance(action, dict) else {"role": "assistant", "content": action}
            return {"choices": [{"message": message, "finish_reason": "stop"}],
                    "usage": {"completion_tokens": used}, "record": record}
        return generate

    def episode(self, actions, value=None, spec=None, total_tokens=4096, used=8):
        return asyncio.run(domain.run_episode(
            value or row(), spec or self.spec, self.generator(actions, used),
            total_tokens=total_tokens,
        ))

    def test_harness_schema_and_bounded_literals(self):
        self.assertEqual(self.spec["_hooks"], [])
        for before, after in [
            ("MAX_TURNS = 9", "MAX_TURNS = 0"),
            ("MAX_TURNS = 9", "MAX_TURNS = True"),
            ("FORMAT_RETRY_BUDGET = 1", "FORMAT_RETRY_BUDGET = 4"),
            ("ILLEGAL_MOVE_RETRY_BUDGET = 1", "ILLEGAL_MOVE_RETRY_BUDGET = -1"),
            ("MAX_TURNS = 9", "MAX_TURNS = 100"),
        ]:
            with self.subTest(after=after), self.assertRaises(ValueError):
                domain.harness_spec(domain.INITIAL_SOURCE.replace(before, after))
        for source in [
            "import chess\n" + domain.INITIAL_SOURCE,
            domain.INITIAL_SOURCE + "\nprint('no')\n",
            domain.INITIAL_SOURCE.replace("CandidateHarness:", "CandidateHarness(object):"),
            domain.INITIAL_SOURCE + "    REFERENCE = ['e2e4']\n",
        ]:
            with self.assertRaises(ValueError):
                domain.harness_spec(source)

    def test_complete_line_native_opponent_and_all_exact_records(self):
        with patch.object(domain, "_execute_hook", side_effect=AssertionError("No sandbox needed")):
            result = self.episode(["<move>e2e4</move>", "<move>g1f3</move>"])
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["stop_reason"], "solved")
        self.assertEqual(result["accepted_solver_moves"], ["e2e4", "g1f3"])
        self.assertEqual(result["opponent_moves"], ["e7e5"])
        self.assertEqual(result["plies_played"], 3)
        self.assertEqual(result["tokens"], 16)
        self.assertEqual(result["tool_calls"], 0)
        self.assertTrue(result["boxed"])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["harness_error"])
        self.assertIs(result["harness_errors"], result["hook_errors"])
        self.assertIsNone(result["failure_reason"])
        self.assertEqual(len(result["calls"]), 2)
        for actual, original in zip(result["calls"], self.records):
            self.assertIs(actual, original)
        second = self.observations[1][-1]["content"]
        self.assertIn("Last opponent reply: e7e5", second)
        self.assertIn("FEN:", second)
        self.assertIn("8 r n b q k b n r", second)
        self.assertIn("[g1f3] SAN=Nf3", second)
        self.assertNotIn("private-puzzle-id", json.dumps(self.observations))
        self.assertNotIn("solution_moves", json.dumps(self.observations))
        self.assertTrue({"id", "reward", "messages", "calls", "tokens", "truncated",
                         "boxed", "tool_calls", "tool_errors", "elapsed_seconds"} <= result.keys())

    def test_legal_but_wrong_fails_immediately_without_reference_leak(self):
        result = self.episode(["<move>d2d4</move>"])
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["wrong_moves"], 1)
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(result["stop_reason"], "wrong_move")
        self.assertEqual(result["opponent_moves"], [])
        self.assertEqual(result["failure_trace"], [
            {"turn": 1, "error": "wrong_move", "action": "d2d4"}])

    def test_partial_line_does_not_receive_reward(self):
        spec = dict(self.spec, MAX_TURNS=1)
        result = self.episode(["<move>e2e4</move>"], spec=spec)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["plies_played"], 2)
        self.assertEqual(result["stop_reason"], "turn_limit")
        self.assertTrue(result["truncated"])

    def test_malformed_retry_and_illegal_retry_explicitly_count_failures(self):
        result = self.episode(["nonsense", "<move>e2e5</move>",
                               "<move>e2e4</move>", "<move>g1f3</move>"])
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["format_errors"], 1)
        self.assertEqual(result["illegal_moves"], 1)
        self.assertEqual(result["failed_actions"], 2)
        self.assertEqual((result["format_retries"], result["illegal_retries"]), (1, 1))
        self.assertEqual(len(result["calls"]), 4)
        self.assertIn("response must specify exactly one move", self.observations[1][-1]["content"])
        self.assertIn("That move is illegal", self.observations[2][-1]["content"])

    def test_invalid_same_square_move_never_raises_infrastructure_error(self):
        result = self.episode(["<move>e2e2</move>", "<move>e2e2</move>"])
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["stop_reason"], "illegal_move")
        self.assertEqual(result["illegal_moves"], 2)
        self.assertEqual(len(result["calls"]), 2)

    def test_ambiguous_and_oversized_outputs_are_failed_actions(self):
        result = self.episode(["<move>e2e4</move><move>d2d4</move>", "e2e4"])
        self.assertEqual(result["format_errors"], 2)
        self.assertFalse(result["boxed"])
        result = self.episode(["x" * (domain.MAX_RESPONSE_CHARS + 1)])
        self.assertEqual(result["stop_reason"], "response_too_long")
        self.assertEqual(result["failed_actions"], 1)
        self.assertEqual(len(result["calls"]), 1)

    def test_thinking_and_promotion(self):
        self.assertEqual(domain.parse_action({"response": "<think>d2d4 e2e4</think>\n<move>E2E4</move>"}), "e2e4")
        result = self.episode(["<move>a7a8q</move>"], row(
            ["a7a8q"], "7k/P7/8/8/8/8/8/7K w - - 0 1"))
        self.assertEqual(result["reward"], 1)

    def test_even_length_reference_finishes_after_native_opponent(self):
        result = self.episode(["<move>e2e4</move>"], row(["e2e4", "e7e5"]))
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["plies_played"], 2)

    def test_corrupt_reference_rejected_before_generation(self):
        for value in [row(["e2e5"]), row(["e2e4"] * 10),
                      row(["a1a2"], "8/8/8/8/8/8/8/K6k w - - 0 1")]:
            if value["solution_moves"] == ["a1a2"]:
                value["start_fen"] = "8/8/8/8/8/8/8/8 w - - 0 1"
            with self.assertRaises(ValueError):
                self.episode([], value)
        self.assertEqual(self.records, [])

    def test_budget_exhaustion_keeps_actual_token_usage(self):
        result = self.episode(["<move>e2e4</move>"], total_tokens=8, used=8)
        self.assertEqual(result["tokens"], 8)
        self.assertEqual(result["stop_reason"], "token_limit")
        self.assertTrue(result["truncated"])
        self.assertEqual(result["reward"], 0)

        async def full(*args):
            raise ContextExhausted("full")
        result = asyncio.run(domain.run_episode(row(), self.spec, full))
        self.assertEqual(result["tokens"], 0)
        self.assertEqual(result["stop_reason"], "context_limit")
        self.assertTrue(result["truncated"])

    def test_unadvertised_tools_are_failed_not_executed(self):
        result = self.episode([{"role": "assistant", "content": "", "tool_calls": [
            {"id": "bad", "type": "function", "function": {"name": "engine", "arguments": "{}"}}]}])
        self.assertEqual(result["stop_reason"], "unsupported_tool")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(result["tool_errors"], 1)
        self.assertEqual(result["failed_actions"], 1)

    def test_falsy_record_is_preserved(self):
        async def generate(*args):
            return {"choices": [{"message": {"content": "<move>e2e4</move>"}}],
                    "usage": {"completion_tokens": 1}, "record": {}}
        result = asyncio.run(domain.run_episode(row(["e2e4"]), self.spec, generate))
        self.assertEqual(result["calls"], [{}])

    def test_custom_hooks_have_only_public_input_and_cannot_change_verifier(self):
        spec = dict(self.spec, _hooks=["format_observation", "parse_action", "recover_action"])
        hook_inputs = []

        def execute(spec, hook, observation):
            hook_inputs.append((hook, copy.deepcopy(observation)))
            self.assertEqual(set(observation), {
                "fen", "board_ascii", "side_to_move", "legal_moves", "last_opponent_move",
                "accepted_solver_moves", "solver_move_number", "turn", "turns_remaining",
                "remaining_tokens", "response", "error", "parsed_action", "retries_remaining"})
            if hook == "format_observation":
                return json.dumps({"board": observation["fen"], "legal": observation["legal_moves"]})
            if hook == "recover_action":
                return "Retry with one SAN move."
            matches = [move["uci"] for move in observation["legal_moves"]
                       if move["san"] == observation["response"]]
            return matches[0] if len(matches) == 1 else None

        with patch.object(domain, "_execute_hook", side_effect=execute):
            result = self.episode(["not SAN", "e4", "Nf3"], spec=spec)
            self.assertEqual(result["reward"], 1)
            result = self.episode(["d4"], spec=spec)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["stop_reason"], "wrong_move")
        self.assertIn("recover_action", [name for name, _ in hook_inputs])

    def test_hook_bad_output_or_timeout_fails_without_fallback(self):
        for bad in [domain.HookExecutionError("bad candidate"), TimeoutError("deadline"), {"wrong": "type"}]:
            spec = dict(self.spec, _hooks=["parse_action"])
            kwargs = {"side_effect": bad} if isinstance(bad, Exception) else {"return_value": bad}
            with patch.object(domain, "_execute_hook", **kwargs):
                result = self.episode(["<move>e2e4</move>"], row(["e2e4"]), spec=spec)
            self.assertEqual(result["reward"], 0)
            self.assertEqual(result["stop_reason"], "candidate_execution_error")
            self.assertEqual(len(result["hook_errors"]), 1)
            self.assertEqual(len(result["calls"]), 1)
            self.assertEqual(result["accepted_solver_moves"], [])
            self.assertFalse(result["truncated"])
            self.assertTrue(result["harness_error"])
            self.assertIs(result["harness_errors"], result["hook_errors"])
            self.assertTrue(result["failure_reason"])

    def test_format_and_recovery_hook_failure_return_failed_trajectories(self):
        for hook, actions in [("format_observation", ["<move>e2e4</move>"]),
                              ("recover_action", ["malformed"])]:
            spec = dict(self.spec, _hooks=[hook])
            with patch.object(domain, "_execute_hook", side_effect=domain.HookExecutionError("failed")):
                result = self.episode(actions, spec=spec)
            self.assertEqual(result["reward"], 0)
            self.assertEqual(result["stop_reason"], "candidate_execution_error")
            self.assertEqual(len(result["calls"]), len(actions))
            self.assertEqual(result["hook_errors"][0]["hook"], hook)

    def test_first_formatter_failure_records_error_only_turn_never_baseline_or_reward(self):
        spec = dict(self.spec, _hooks=["format_observation"])
        with patch.object(domain, "_execute_hook", side_effect=domain.HookExecutionError("failed details")), \
                patch.object(domain, "format_observation", side_effect=AssertionError("Forbidden fallback")):
            result = self.episode(["<move>e2e4</move>"], row(["e2e4"]), spec=spec)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["stop_reason"], "candidate_execution_error")
        self.assertTrue(result["harness_error"])
        self.assertEqual(result["failure_reason"], "HookExecutionError: failed details")
        self.assertTrue(result["diagnostic_error_turn"])
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(result["tokens"], 8)
        self.assertEqual(len(result["calls"]), 1)
        self.assertIs(result["calls"][0], self.records[0])
        self.assertEqual(result["calls"][0]["max_tokens"], 128)
        self.assertEqual(result["accepted_solver_moves"], [])
        self.assertFalse(result["boxed"])
        diagnostic_input = json.dumps(self.observations)
        self.assertNotIn("FEN", diagnostic_input)
        self.assertNotIn("Legal moves", diagnostic_input)
        self.assertNotIn("e2e4", diagnostic_input)
        self.assertNotIn("failed details", diagnostic_input)

    def test_candidate_error_reason_is_bounded(self):
        spec = dict(self.spec, _hooks=["parse_action"])
        with patch.object(domain, "_execute_hook", side_effect=domain.HookExecutionError("x" * 10000)):
            result = self.episode(["<move>e2e4</move>"], spec=spec)
        self.assertEqual(len(result["failure_reason"]), domain.MAX_HARNESS_ERROR_CHARS)
        self.assertEqual(result["harness_errors"][0]["reason"], result["failure_reason"])

    def test_formatter_failure_after_first_generation_keeps_records_without_diagnostic_call(self):
        spec = dict(self.spec, _hooks=["format_observation"])
        with patch.object(domain, "_execute_hook", side_effect=[
            "Public initial chess observation.", domain.HookExecutionError("second turn failed")
        ]):
            result = self.episode(["<move>e2e4</move>"], spec=spec)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(len(result["calls"]), 1)
        self.assertFalse(result["diagnostic_error_turn"])
        self.assertEqual(result["stop_reason"], "candidate_execution_error")

    def test_unicode_response_payload_limit_is_domain_failure_not_infrastructure(self):
        result = self.episode(["\U0001f600" * 10000])
        self.assertEqual(result["stop_reason"], "response_too_long")
        self.assertEqual(result["reward"], 0)

    def test_recovery_can_stop_but_not_supply_moves(self):
        spec = dict(self.spec, _hooks=["recover_action"])
        with patch.object(domain, "_execute_hook", return_value=None):
            result = self.episode(["bad"], spec=spec)
        self.assertEqual(result["stop_reason"], "invalid_format")
        self.assertEqual(result["generation_calls"], 1)
        self.assertEqual(result["reward"], 0)

    def test_isolated_san_parser_is_executable(self):
        source = domain.INITIAL_SOURCE + '''
    def parse_action(self, observation):
        matches = [move["uci"] for move in observation["legal_moves"]
                   if move["san"] == observation["response"]]
        return matches[0] if len(matches) == 1 else None
'''
        spec = domain.harness_spec(source)
        self.assertEqual(spec["_hooks"], ["parse_action"])
        result = self.episode(["e4", "Nf3"], spec=spec)
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["hook_errors"], [])

    def test_real_isolated_malformed_formatter_cannot_earn_winning_move_reward(self):
        source = domain.INITIAL_SOURCE + '''
    def format_observation(self, observation):
        return {"not": "text"}
'''
        result = self.episode(["<move>e2e4</move>"], row(["e2e4"]),
                              spec=domain.harness_spec(source))
        self.assertEqual(result["reward"], 0)
        self.assertTrue(result["diagnostic_error_turn"])
        self.assertTrue(result["harness_error"])
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["failure_reason"], "CandidateExecutionError: Invalid hook text output")

    def test_sandbox_infrastructure_errors_propagate(self):
        from sao_bridge.program_harness import SandboxInfrastructureError
        spec = dict(self.spec, _hooks=["parse_action"])
        with patch.object(domain, "_execute_hook", side_effect=SandboxInfrastructureError("setup")):
            with self.assertRaises(SandboxInfrastructureError):
                self.episode(["<move>e2e4</move>"], spec=spec)

    def test_formatter_infrastructure_failure_does_not_generate_diagnostic_trajectory(self):
        from sao_bridge.program_harness import SandboxInfrastructureError
        spec = dict(self.spec, _hooks=["format_observation"])
        with patch.object(domain, "_execute_hook", side_effect=SandboxInfrastructureError("setup")):
            with self.assertRaises(SandboxInfrastructureError):
                self.episode([], spec=spec)
        self.assertEqual(self.records, [])


class ChessDataTests(unittest.TestCase):
    def test_trigger_and_conservative_position_dedup(self):
        raw = {"PuzzleId": "example", "FEN": chess.STARTING_FEN, "Moves": "e2e4 e7e5",
               "Rating": 1000, "Themes": ["opening"]}
        converted = prep.convert_row(raw, 123)
        self.assertEqual(converted["puzzle_id"], "example")
        self.assertEqual(converted["id"], prep.digest(b"lichess:example"))
        self.assertIsInstance(int(converted["id"][:8], 16), int)
        board = chess.Board()
        board.push_uci("e2e4")
        self.assertEqual(converted["start_fen"], board.fen())
        self.assertEqual(converted["solution_moves"], ["e7e5"])
        self.assertEqual(converted["source_row"], 123)
        fen = board.fen()
        pieces, side = fen.split()[:2]
        self.assertEqual(domain.position_key(fen), domain.position_key(f"{pieces} {side} - - 70 99"))
        self.assertNotEqual(domain.position_key(fen), domain.position_key(f"{pieces} w - - 0 1"))
        with self.assertRaises(ValueError):
            prep.convert_row({**raw, "Moves": "e2e5 e7e5"}, 0)

    def test_partition_independent_of_input_order_and_globally_unique(self):
        rows = [{"id": str(i), "position_key": str(i)} for i in range(6)]
        sizes = {"train": 3, "reward": 1, "dev": 1, "test": 1}
        result = prep.partition(rows, sizes)
        self.assertEqual(result, prep.partition(rows[::-1], sizes))
        self.assertEqual({key: len(value) for key, value in result.items()}, sizes)
        with self.assertRaises(ValueError):
            prep.partition(rows[:-1] + [rows[0]], sizes)

    def test_persistent_manifest_all_rows_legality_dedup_hashes_and_permissions(self):
        assets = ROOT / "assets/chess_sao_20260929"
        result = prep.verify(assets)
        self.assertTrue(result["verified"])
        self.assertEqual(result["unique_positions"], 17024)
        self.assertEqual(result["splits"], {"train": 16384, "reward": 128, "dev": 256, "test": 256})
        self.assertLess(result["retained_source_bytes"], 80 * 1024 * 1024)
        manifest = json.loads((assets / "manifest.json").read_text())
        for name in result["splits"]:
            self.assertEqual(manifest["splits"][name]["path"], f"{name}.json")
            self.assertEqual(manifest["splits"][name]["count"], result["splits"][name])
            self.assertIsInstance(json.loads((assets / f"{name}.json").read_text()), list)


if __name__ == "__main__":
    unittest.main()
