import asyncio
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_eval import (
    bind_environment, evaluate_harnesses, matrix_harnesses, run, validate_config,
)
from sao_bridge.alfworld_env_service import validate_assignment_cohorts
from sao_bridge.alfworld_domain import logger as alfworld_logger, run_episode
from sao_bridge.common import write_json
from sao_bridge.evaluation import Sampler


class AlfworldSolverEvaluationTests(unittest.TestCase):
    def test_shared_cohort_is_explicit_and_training_assignments_remain_disjoint(self):
        first = {"name": "a", "task_ids": [240, 241]}
        repeated = [first, {**first, "name": "b"}]
        with self.assertRaisesRegex(ValueError, "partition"):
            validate_assignment_cohorts(repeated)
        validate_assignment_cohorts(repeated, harness_matrix=True)
        disjoint = [first, {"name": "b", "task_ids": [242, 243]}]
        validate_assignment_cohorts(disjoint)
        with self.assertRaisesRegex(ValueError, "same exact"):
            validate_assignment_cohorts(disjoint, harness_matrix=True)
        for change in ({"task_ids": [241, 242]}, {"split": "sao_train_v1"},
                       {"catalogue_sha256": "different"}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "same exact"):
                validate_assignment_cohorts([first, {**first, **change}], harness_matrix=True)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            validate_assignment_cohorts([], harness_matrix=True)

    def episode(self, completion, *, total_tokens=4096, environment_finishes=True):
        requests = []

        def handle(request):
            requests.append((request.url.path, json.loads(request.content)))
            payload = {"status": "RUNNING", "messages": [], "tools": []}
            if request.url.path == "/api/interact" and environment_finishes:
                payload.update(status="COMPLETED", reward=1.0)
            return httpx.Response(
                200, json=payload, headers={"session_id": "fixture-session"},
            )

        async def run():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
            generate = AsyncMock(
                side_effect=completion if isinstance(completion, Exception) else None,
                return_value=completion,
            )
            with (
                patch("sao_bridge.alfworld_domain.task_count", return_value=500),
                patch(
                    "sao_bridge.alfworld_domain._controller_url",
                    return_value="http://localhost/api",
                ),
                patch("sao_bridge.alfworld_domain.httpx.AsyncClient", return_value=client),
            ):
                result = await run_episode(
                    {"id": "alfworld-240", "task_id": 240, "task_name": "fixture"},
                    {}, generate, total_tokens=total_tokens,
                )
            self.assertEqual(generate.await_count, 1)
            return result, requests

        return asyncio.run(run())

    @staticmethod
    def completion(arguments, *, finish_reason="tool_calls", tokens=8):
        return {
            "choices": [{
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "call-0", "type": "function",
                        "function": {"name": "action", "arguments": arguments},
                    }],
                },
                "finish_reason": finish_reason,
            }],
            "usage": {"completion_tokens": tokens},
        }

    def test_malformed_generated_arguments_fail_only_the_model_episode(self):
        for arguments in ('{"action": "look', '["look"]', None):
            with self.subTest(arguments=arguments):
                completion = self.completion(arguments)
                with self.assertLogs(alfworld_logger, level="WARNING"):
                    result, requests = self.episode(completion)
                self.assertEqual(result["reward"], 0.0)
                self.assertEqual(result["stop_reason"], "invalid_tool_arguments")
                self.assertEqual(result["tokens"], 8)
                self.assertEqual(result["turns"], 1)
                self.assertEqual(result["tool_errors"], 1)
                self.assertFalse(result["truncated"])
                self.assertEqual(result["messages"][-1], completion["choices"][0]["message"])
                self.assertEqual(result["invalid_tool_arguments"][0]["tool_call_index"], 0)
                self.assertEqual(
                    [path for path, _ in requests], ["/api/start_sample", "/api/cancel"],
                )

    def test_all_calls_are_checked_before_the_next_history_round_trip(self):
        completion = self.completion('{"action": "look"}', finish_reason="length")
        completion["choices"][0]["message"]["tool_calls"].append({
            "id": "call-1", "type": "function",
            "function": {"name": "action", "arguments": '{"action": "take'},
        })
        with self.assertLogs(alfworld_logger, level="WARNING"):
            result, _ = self.episode(completion)
        self.assertEqual(result["invalid_tool_arguments"][0]["tool_call_index"], 1)
        self.assertEqual(result["tool_calls"], 2)
        self.assertTrue(result["truncated"])

    def test_valid_tool_calls_and_terminal_reward_are_unchanged(self):
        completion = self.completion('{"action": "look"}')
        result, requests = self.episode(completion)
        self.assertEqual(result["reward"], 1.0)
        self.assertEqual(result["tool_errors"], 0)
        self.assertNotIn("invalid_tool_arguments", result)
        self.assertEqual(
            requests[-1],
            ("/api/interact", {"messages": [completion["choices"][0]["message"]]}),
        )
        self.assertEqual(len(requests), 2)

    def test_token_budget_failure_still_cancels_without_another_generation(self):
        result, requests = self.episode(
            self.completion('{"action": "look"}', tokens=8),
            total_tokens=8, environment_finishes=False,
        )
        self.assertEqual(result["stop_reason"], "token_limit")
        self.assertEqual(result["reward"], 0.0)
        self.assertTrue(result["truncated"])
        self.assertEqual(requests[-1][0], "/api/cancel")

    def test_http_failures_remain_infrastructure_errors(self):
        request = httpx.Request("POST", "http://localhost/v1/chat/completions")
        response = httpx.Response(400, request=request, json={"error": "invalid model"})
        error = httpx.HTTPStatusError("invalid model", request=request, response=response)
        with self.assertRaises(httpx.HTTPStatusError):
            self.episode(error)

    def test_sampler_preserves_http_rejection_details(self):
        async def run():
            sampler = Sampler("http://localhost/v1")
            await sampler.client.aclose()
            sampler.client = httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(400, json={"error": "invalid model"}),
            ))
            try:
                with self.assertRaisesRegex(httpx.HTTPStatusError, "invalid model"):
                    await sampler([{"role": "user", "content": "fixture"}], [], 8, 42)
                self.assertEqual(sampler.requests, 0)
            finally:
                await sampler.close()

        asyncio.run(run())

    def test_environment_binding_reaches_parent_and_children(self):
        launch = type("Launch", (), {"env": {}})()
        with patch.dict("os.environ", {}, clear=True):
            bind_environment(launch, "http://127.0.0.1:28000/api", 123)
            self.assertEqual(
                launch.env["SAO_ALFWORLD_CONTROLLER_URL"],
                "http://127.0.0.1:28000/api",
            )
            self.assertEqual(
                __import__("os").environ["SAO_ALFWORLD_HTTP_TIMEOUT"],
                "123",
            )

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
            with self.assertRaisesRegex(ValueError, "explicit"):
                validate_config({**config, "harnesses": {"candidate": str(no_patch)}})
            self.assertEqual(
                validate_config(
                    {**config, "harnesses": {"candidate": str(no_patch)}},
                    harness_matrix=True,
                )["solver_tokens"],
                normalized["solver_tokens"],
            )
            with self.assertRaisesRegex(ValueError, "matrix_concurrency"):
                validate_config({**config, "matrix_concurrency": 0}, harness_matrix=True)

    def test_matrix_preserves_duplicate_labels_and_rejects_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_patch, duplicate = root / "no_patch.json", root / "duplicate.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            duplicate.write_text('{"actions": [], "benchmark": "alfworld"}\n')
            config = {
                "no_patch": str(no_patch), "harnesses": {
                    "proposal0": str(no_patch), "proposal1": str(duplicate),
                },
            }
            unique, aliases = matrix_harnesses(config)
            self.assertEqual(len(unique), 1)
            self.assertEqual(set(aliases), {"no_patch", "proposal0", "proposal1"})
            self.assertEqual(len({value["sha256"] for value in aliases.values()}), 1)
            for requested in ({}, [], {"no_patch": str(no_patch)}, {"../escape": str(no_patch)}):
                with self.subTest(requested=requested), self.assertRaises(ValueError):
                    matrix_harnesses({**config, "harnesses": requested})
            write_json(duplicate, {"benchmark": "math", "actions": []})
            with self.assertRaisesRegex(ValueError, "benchmark"):
                matrix_harnesses(config)

    def test_matrix_uses_one_sampler_and_common_task_sample_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_patch, candidate = root / "no_patch.json", root / "candidate.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            write_json(candidate, {"benchmark": "alfworld", "actions": [{"fixture": 1}]})
            config = {
                "no_patch": str(no_patch),
                "harnesses": {"candidate": str(candidate), "duplicate": str(candidate)},
                "task_ids": [240, 300], "seed": 42, "samples": 2,
                "solver_tokens": 4096, "temperature": 1.0,
                "eval_concurrency": 32, "environment_concurrency": 32,
            }
            unique, aliases = matrix_harnesses(config)
            assignments = []
            for index, (identity, value) in enumerate(unique.items()):
                snapshot = root / f"{identity}.json"
                write_json(snapshot, value["patch"])
                assignments.append({
                    "name": f"fixture-{index}", "harness": str(snapshot),
                    "harness_sha256": identity,
                })
            sampler = Mock(close=AsyncMock())
            evaluation = AsyncMock(side_effect=[{"accuracy": 0.5}, {"accuracy": 0.75}])
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler) as factory,
                patch("sao_bridge.alfworld_solver_eval.evaluate", evaluation),
            ):
                report = asyncio.run(evaluate_harnesses(
                    "http://localhost/v1", config, assignments, root, aliases,
                ))
            factory.assert_called_once()
            sampler.close.assert_awaited_once()
            self.assertEqual(evaluation.await_count, 2)
            first, second = evaluation.await_args_list
            self.assertEqual(first.kwargs, second.kwargs)
            self.assertEqual(first.kwargs["samples"], 2)
            self.assertEqual(first.kwargs["seed"], 42)
            self.assertEqual(
                [(row["id"], row["task_id"]) for row in first.args[1]],
                [(row["id"], row["task_id"]) for row in second.args[1]],
            )
            self.assertNotEqual(first.args[1][0]["task_name"], second.args[1][0]["task_name"])
            self.assertNotEqual(first.args[2], second.args[2])
            self.assertEqual(report["logical_harnesses"], 3)
            self.assertEqual(report["unique_harnesses"], 2)
            self.assertEqual(report["variants"]["candidate"], report["variants"]["duplicate"])
            self.assertEqual(report["variants"]["candidate"]["accuracy"], 0.75)
            self.assertEqual(len(list((root / "reports").glob("*.json"))), 2)

    def test_single_harness_output_is_unchanged_and_failures_close_sampler(self):
        config = {
            "task_ids": [240], "seed": 42, "samples": 2, "solver_tokens": 4096,
            "temperature": 1.0, "eval_concurrency": 32, "environment_concurrency": 32,
        }
        assignment = [{"name": "fixture", "harness": "unused-in-no-patch-mode"}]
        for value in ({"accuracy": 0.5}, RuntimeError("environment failed")):
            sampler = Mock(close=AsyncMock())
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler),
                patch(
                    "sao_bridge.alfworld_solver_eval.evaluate",
                    AsyncMock(side_effect=value if isinstance(value, Exception) else None,
                              return_value=value),
                ),
            ):
                if isinstance(value, Exception):
                    with self.assertRaisesRegex(RuntimeError, "environment failed"):
                        asyncio.run(evaluate_harnesses("http://localhost/v1", config, assignment, Path(".")))
                else:
                    report = asyncio.run(evaluate_harnesses(
                        "http://localhost/v1", config, assignment, Path("."),
                    ))
                    self.assertIs(report, value)
            sampler.close.assert_awaited_once()

    def test_matrix_real_evaluator_pairs_seeds_and_keeps_completed_reports_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            no_patch = root / "no_patch.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            config = {
                "task_ids": [240, 300], "seed": 42, "samples": 2,
                "solver_tokens": 4096, "temperature": 1.0,
                "eval_concurrency": 32, "environment_concurrency": 32,
            }
            assignments = [
                {"name": name, "harness": str(no_patch), "harness_sha256": name}
                for name in ("first", "second")
            ]
            aliases = {name: {"sha256": name, "source_path": str(no_patch)}
                       for name in ("first", "second")}
            seeds = {}

            async def episode(row, spec, sampler, *, total_tokens, seed):
                seeds[row["task_name"], row["task_id"], seed] = total_tokens
                return {
                    "id": row["id"], "task_id": row["task_id"], "reward": 1.0,
                    "tokens": 8, "tool_calls": 1, "tool_errors": 0,
                    "truncated": False, "boxed": True,
                }

            domain = Mock(harness_spec=Mock(return_value={}),
                          run_episode=AsyncMock(side_effect=episode))
            sampler = Mock(close=AsyncMock())
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler),
                patch("sao_bridge.evaluation.get_domain", return_value=domain),
            ):
                result = asyncio.run(evaluate_harnesses(
                    "http://localhost/v1", config, assignments, root, aliases,
                ))
            self.assertEqual(len(seeds), 8)
            self.assertEqual(
                {(task, seed) for name, task, seed in seeds if name == "first"},
                {(task, seed) for name, task, seed in seeds if name == "second"},
            )
            for value in result["variants"].values():
                report = json.loads(Path(value["report"]).read_text())
                self.assertEqual(report["metric"], "mean@2")
                self.assertEqual(len(report["results"]), 4)
                self.assertEqual({row["sample_index"] for row in report["results"]}, {0, 1})
            sampler.close.assert_awaited_once()

            failed_root = root / "failure"
            sampler = Mock(close=AsyncMock())
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler),
                patch(
                    "sao_bridge.alfworld_solver_eval.evaluate",
                    AsyncMock(side_effect=[{"accuracy": 0.5}, RuntimeError("real HTTP failure")]),
                ),
                self.assertRaisesRegex(RuntimeError, "real HTTP failure"),
            ):
                asyncio.run(evaluate_harnesses(
                    "http://localhost/v1", config, assignments, failed_root, aliases,
                ))
            sampler.close.assert_awaited_once()
            self.assertTrue((failed_root / "reports/first.json").is_file())
            self.assertFalse((failed_root / "reports/second.json").exists())
            self.assertEqual(
                json.loads((failed_root / "PROGRESS.json").read_text())["completed_unique_harnesses"],
                1,
            )

    def test_matrix_parallelism_is_bounded_and_cancels_peers_before_closing_sampler(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "no_patch.json"
            write_json(source, {"benchmark": "alfworld", "actions": []})
            config = {
                "task_ids": [240], "seed": 42, "samples": 2, "solver_tokens": 4096,
                "temperature": 1.0, "eval_concurrency": 64, "environment_concurrency": 32,
                "matrix_concurrency": 2,
            }
            assignments = [{
                "name": f"fixture-{index}", "harness": str(source),
                "harness_sha256": str(index),
            } for index in range(5)]
            aliases = {str(index): {"sha256": str(index), "source_path": str(source)}
                       for index in range(5)}
            active = maximum = 0

            async def score(*args, **kwargs):
                nonlocal active, maximum
                active += 1
                maximum = max(maximum, active)
                try:
                    await asyncio.sleep(0.01)
                    return {"accuracy": 0.5}
                finally:
                    active -= 1

            async def close():
                self.assertEqual(active, 0)

            sampler = Mock(close=AsyncMock(side_effect=close))
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler),
                patch("sao_bridge.alfworld_solver_eval.evaluate", side_effect=score),
            ):
                result = asyncio.run(evaluate_harnesses(
                    "http://localhost/v1", config, assignments, root, aliases,
                ))
            self.assertEqual(maximum, 2)
            self.assertEqual(result["unique_harnesses"], 5)
            sampler.close.assert_awaited_once()

            cancelled = False

            async def failing_score(sampler, rows, *args, **kwargs):
                nonlocal active, cancelled
                if rows[0]["task_name"] == "fixture-1":
                    raise RuntimeError("real HTTP failure")
                active += 1
                try:
                    await asyncio.sleep(60)
                finally:
                    active -= 1
                    cancelled = True

            sampler = Mock(close=AsyncMock(side_effect=close))
            with (
                patch("sao_bridge.alfworld_solver_eval.Sampler", return_value=sampler),
                patch("sao_bridge.alfworld_solver_eval.evaluate", side_effect=failing_score),
                self.assertRaisesRegex(RuntimeError, "real HTTP failure"),
            ):
                asyncio.run(evaluate_harnesses(
                    "http://localhost/v1", config, assignments, root / "failed", aliases,
                ))
            self.assertTrue(cancelled)
            sampler.close.assert_awaited_once()

    def test_runner_wires_matrix_environment_and_freezes_patches_before_workers_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"fixture")
            no_patch, candidate = root / "empty.json", root / "candidate.json"
            write_json(no_patch, {"benchmark": "alfworld", "actions": []})
            original = {"benchmark": "alfworld", "actions": [{"fixture": 1}]}
            write_json(candidate, original)
            config = {
                "run": str(root / "run"), "solver_model": str(model),
                "solver_version": "fixture", "task_ids": [240, 300],
                "no_patch": str(no_patch), "harnesses": {"candidate": str(candidate)},
                "seed": 42, "samples": 2,
            }
            config_path = root / "config.json"
            write_json(config_path, config)
            launch = Mock(env={})
            launch.vllm.return_value = nullcontext("http://localhost/v1")

            def environment(module, arguments, **kwargs):
                self.assertEqual(module, "sao_bridge.alfworld_env_service")
                self.assertIn("--harness-matrix", arguments)
                self.assertIn("--async-controller", arguments)
                run_root = Path(arguments[arguments.index("--run") + 1])
                write_json(run_root / "READY.json", {"ready": True})
                assignment_path = Path(arguments[arguments.index("--assignments") + 1])
                assignments = json.loads(assignment_path.read_text())
                validate_assignment_cohorts(assignments, harness_matrix=True)
                snapshots = [json.loads(Path(value["harness"]).read_text()) for value in assignments]
                self.assertIn(original, snapshots)
                self.assertNotIn(str(candidate), [value["harness"] for value in assignments])
                write_json(candidate, {"benchmark": "alfworld", "actions": []})
                return Mock(poll=Mock(return_value=None))

            launch.module.side_effect = environment

            async def evaluate_matrix(urls, normalized, assignments, run_root, aliases):
                snapshots = [json.loads(Path(value["harness"]).read_text()) for value in assignments]
                self.assertIn(original, snapshots)
                self.assertEqual(set(aliases), {"no_patch", "candidate"})
                return {"variants": {"candidate": {"accuracy": 0.75}},
                        "unique_harnesses": 2, "logical_harnesses": 2}

            with (
                patch("sao_bridge.alfworld_solver_eval.Launcher", return_value=launch),
                patch("sao_bridge.alfworld_solver_eval.stop_known", return_value={}),
                patch("sao_bridge.alfworld_solver_eval.gpu_release_audit", return_value={}),
                patch("sao_bridge.alfworld_solver_eval.threading.Thread"),
                patch("sao_bridge.alfworld_solver_eval.evaluate_harnesses", side_effect=evaluate_matrix),
                patch.dict("os.environ"),
            ):
                complete = run(config_path, harness_matrix=True)
            self.assertEqual(complete["evaluation_mode"], "frozen_harness_matrix")
            self.assertEqual(complete["unique_harnesses"], 2)
            self.assertNotIn("accuracy", complete)
            launch.vllm.assert_called_once()
            launch.close.assert_called_once()
            self.assertTrue((root / "run/RELEASED.json").is_file())


if __name__ == "__main__":
    unittest.main()
