import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

import httpx


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_solver_eval import bind_environment, validate_config
from sao_bridge.alfworld_domain import run_episode
from sao_bridge.common import write_json
from sao_bridge.evaluation import Sampler


class AlfworldSolverEvaluationTests(unittest.TestCase):
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
                with self.assertLogs("sao_bridge.alfworld_domain", level="WARNING"):
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
        with self.assertLogs("sao_bridge.alfworld_domain", level="WARNING"):
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


if __name__ == "__main__":
    unittest.main()
