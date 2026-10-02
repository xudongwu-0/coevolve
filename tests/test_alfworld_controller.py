import asyncio
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_controller import create_app, logger
from sao_bridge.common import ROOT


class AlfworldControllerTests(unittest.TestCase):
    def test_original_blocking_controller_expires_workers_during_start_burst(self):
        path = ROOT / "third_party/harness-r1-upstream/code/life-harness/AgentBench/scripts/local_agentrl_controller.py"
        spec = importlib.util.spec_from_file_location("original_controller_fixture", path)
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        clock = [1000.0]
        original.time = SimpleNamespace(time=lambda: clock[0])
        original.workers["fixture"] = {"http://127.0.0.1:28100/api": {
            "address": "http://127.0.0.1:28100/api", "last_seen": clock[0],
        }}

        def blocking_worker(*args, **kwargs):
            clock[0] += 2
            return Mock(status_code=200, json=Mock(return_value={"status": "RUNNING"}))

        async def run():
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=original.app), base_url="http://test",
            ) as client:
                return await asyncio.gather(*[
                    client.post("/api/start_sample", json={"name": "fixture", "index": index})
                    for index in range(128)
                ])

        with patch.object(original.requests, "post", side_effect=blocking_worker):
            responses = asyncio.run(run())
        self.assertEqual(sum(response.status_code == 200 for response in responses), 30)
        self.assertEqual(sum(response.status_code == 404 for response in responses), 98)
        self.assertIn("no alive worker", responses[-1].json()["detail"])

    def test_128_starts_and_heartbeat_are_concurrent_not_serial(self):
        async def run():
            entered, release = asyncio.Event(), asyncio.Event()
            active = 0
            seen = []

            async def worker(request):
                nonlocal active
                if request.url.path.endswith("worker_status"):
                    return httpx.Response(200, json={"current": 0})
                self.assertTrue(request.url.path.endswith("start_sample"))
                payload = json.loads(request.content)
                self.assertEqual(set(payload), {"index", "custom_task", "session_id"})
                seen.append(payload["session_id"])
                active += 1
                if active == 128:
                    entered.set()
                await release.wait()
                return httpx.Response(200, json={"status": "RUNNING"})

            app = create_app(transport=httpx.MockTransport(worker))
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                for index in range(4):
                    response = await client.post("/api/receive_heartbeat", json={
                        "name": f"worker-{index}", "address": f"http://127.0.0.1:{28100 + index}/api",
                        "concurrency": 32, "indices": list(range(32)),
                    })
                    self.assertEqual(response.status_code, 200)
                tasks = [asyncio.create_task(client.post("/api/start_sample", json={
                    "name": f"worker-{index // 32}", "index": index % 32,
                })) for index in range(128)]
                try:
                    await asyncio.wait_for(entered.wait(), timeout=3)
                    heartbeat = await asyncio.wait_for(client.post("/api/receive_heartbeat", json={
                        "name": "worker-0", "address": "http://127.0.0.1:28100/api",
                        "concurrency": 32, "indices": list(range(32)),
                    }), timeout=1)
                    self.assertEqual(heartbeat.status_code, 200)
                finally:
                    release.set()
                responses = await asyncio.gather(*tasks)
                self.assertTrue(all(response.status_code == 200 for response in responses))
                self.assertEqual(len(set(seen)), 128)
                self.assertEqual({response.headers["session_id"] for response in responses},
                                 {str(index) for index in range(1, 129)})
        asyncio.run(run())

    def test_official_session_routing_unwrap_cancel_and_failures(self):
        async def run():
            calls = []
            fail_cancel = False

            async def worker(request):
                endpoint = request.url.path.rsplit("/", 1)[-1]
                payload = json.loads(request.content) if request.content else None
                calls.append((request.method, endpoint, payload))
                if endpoint == "worker_status":
                    return httpx.Response(200, json={"current": 0})
                if endpoint == "get_indices":
                    return httpx.Response(200, json=[7])
                if endpoint == "start_sample":
                    return httpx.Response(200, json={"status": "RUNNING", "messages": [], "tools": []})
                if endpoint == "interact":
                    return httpx.Response(200, json={"env_out": {
                        "status": "COMPLETED", "reward": 1.0,
                    }})
                if endpoint == "cancel_with_notice" and fail_cancel:
                    raise httpx.ReadError("fixture transport failure", request=request)
                if endpoint == "cancel_with_notice":
                    await client.post("/api/cancel_notice", json={"session_id": payload["session_id"]})
                    return httpx.Response(200, json={"ok": True})
                if endpoint == "calculate_overall":
                    return httpx.Response(503, text="fixture upstream failure")
                raise AssertionError(endpoint)

            app = create_app(transport=httpx.MockTransport(worker))
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                self.assertEqual((await client.post("/api/receive_heartbeat", json={
                    "name": "task", "address": "http://127.0.0.1:28100",
                    "concurrency": 32, "indices": [7],
                })).status_code, 200)
                self.assertEqual((await client.get("/api/get_indices?name=task")).json(), [7])
                started = await client.post("/api/start_sample", json={"name": "task", "index": 7})
                sid = started.headers["session_id"]
                interacted = await client.post(
                    "/api/interact", headers={"session_id": sid},
                    json={"messages": [{"role": "assistant", "content": "look"}]},
                )
                self.assertEqual(interacted.json(), {"status": "COMPLETED", "reward": 1.0})
                self.assertEqual(interacted.headers["session_id"], sid)
                fail_cancel = True
                with self.assertLogs(logger, level="ERROR"):
                    cancelled = await client.post("/api/cancel", json={"session_id": sid})
                self.assertEqual(cancelled.status_code, 502)
                self.assertIn("fixture transport failure", cancelled.text)
                fail_cancel = False
                self.assertEqual((await client.post("/api/cancel", json={"session_id": sid})).status_code, 200)
                self.assertEqual((await client.post("/api/interact", json={"session_id": sid})).status_code, 404)
                overall = await client.post("/api/calculate_overall", json={"name": "task"})
                self.assertEqual(overall.status_code, 503)
                self.assertIn("fixture upstream failure", overall.text)
                self.assertIn(("POST", "start_sample", {
                    "index": 7, "custom_task": None, "session_id": 1,
                }), calls)
        asyncio.run(run())

    def test_cancel_ack_does_not_release_client_before_worker_completion_notice(self):
        async def run():
            accepted = asyncio.Event()

            async def worker(request):
                endpoint = request.url.path.rsplit("/", 1)[-1]
                if endpoint == "worker_status":
                    return httpx.Response(200, json={"current": 1})
                if endpoint == "start_sample":
                    return httpx.Response(200, json={"status": "RUNNING"})
                if endpoint == "cancel_with_notice":
                    accepted.set()
                    return httpx.Response(200, json={"session_id": 1})
                raise AssertionError(endpoint)

            app = create_app(transport=httpx.MockTransport(worker))
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                heartbeat = {
                    "name": "task", "address": "http://127.0.0.1:28100/api",
                    "concurrency": 1, "indices": [7],
                }
                self.assertEqual((await client.post("/api/receive_heartbeat", json=heartbeat)).status_code, 200)
                started = await client.post("/api/start_sample", json={"name": "task", "index": 7})
                sid = started.headers["session_id"]
                cancellation = asyncio.create_task(client.post("/api/cancel", json={"session_id": sid}))
                await asyncio.wait_for(accepted.wait(), timeout=1)
                self.assertFalse(cancellation.done())
                self.assertEqual((await client.post("/api/receive_heartbeat", json=heartbeat)).status_code, 200)
                await client.post("/api/cancel_notice", json={"session_id": sid})
                self.assertEqual((await asyncio.wait_for(cancellation, timeout=1)).status_code, 200)
                self.assertEqual((await client.post("/api/interact", json={"session_id": sid})).status_code, 404)
        asyncio.run(run())

    def test_unknown_or_expired_worker_and_invalid_address_are_errors(self):
        async def run():
            app = create_app(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"current": 0}),
            ))
            async with app.router.lifespan_context(app), httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test",
            ) as client:
                self.assertEqual((await client.post(
                    "/api/start_sample", json={"name": "missing", "index": 0},
                )).status_code, 404)
                invalid = await client.post("/api/receive_heartbeat", json={
                    "name": "task", "address": "http://example.com/api", "concurrency": 1, "indices": [0],
                })
                self.assertEqual(invalid.status_code, 400)
                with patch("sao_bridge.alfworld_controller.time.time", return_value=1000):
                    await client.post("/api/receive_heartbeat", json={
                        "name": "task", "address": "http://127.0.0.1:28100/api",
                        "concurrency": 1, "indices": [0],
                    })
                with patch("sao_bridge.alfworld_controller.time.time", return_value=1061):
                    response = await client.post("/api/start_sample", json={"name": "task", "index": 0})
                self.assertEqual(response.status_code, 404)
        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
