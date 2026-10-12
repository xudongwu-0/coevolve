from pathlib import Path
import os
import tempfile
import time
import unittest

from sao_bridge.campaign_monitor import pair_snapshot, snapshot, watch_pair
from sao_bridge.common import write_json


class MonitorTests(unittest.TestCase):
    def test_watcher_exits_nonzero_on_failure_and_zero_only_on_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            host = os.uname().nodename
            path = root / "protocol.json"
            write_json(path, {"nodes": {host: {"root": str(root)}}})
            for health, expected_exit in (("failed", 2), ("completed", 0)):
                write_json(root / "MONITOR_STATUS.json",
                           {"host": host, "health": health, "updated_at": time.time()})
                self.assertEqual(watch_pair(path), expected_exit)

    def test_pair_watch_detects_failures_and_stale_heartbeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = {"nodes": {name: {"root": str(root / name)} for name in ("a", "b")}}
            for name in protocol["nodes"]:
                write_json(root / name / "MONITOR_STATUS.json",
                           {"host": name, "health": "running", "updated_at": 10})
            self.assertEqual(pair_snapshot(protocol, 20)["health"], "running")
            self.assertEqual(pair_snapshot(protocol, 131)["health"], "attention_required")
            for name in protocol["nodes"]:
                write_json(root / name / "MONITOR_STATUS.json",
                           {"host": name, "health": "completed", "updated_at": 10})
            self.assertEqual(pair_snapshot(protocol, 200)["health"], "completed")
            write_json(root / "b/MONITOR_STATUS.json",
                       {"host": "b", "health": "failed", "updated_at": 200})
            self.assertEqual(pair_snapshot(protocol, 200)["health"], "attention_required")

    def test_reports_real_steps_and_partial_line_without_false_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "hwh"
            write_json(root / "WORKER_STATUS.json", {"run": str(run), "phase": "running"})
            write_json(run / "STATUS.json", {"status": "w1_start"})
            write_json(run / "w1_spec.json", {"steps": 40, "critic_warmup_steps": 8})
            path = run / "w1/metrics.jsonl"
            path.parent.mkdir()
            path.write_text('{"global_step":0}\n{"global_step":')
            result = snapshot(root, {"ActiveState": "active"}, time.time())
            self.assertEqual(result["health"], "running")
            self.assertEqual(result["optimizer_progress"]["w1"]["actor_updates"], 0)
            self.assertTrue(result["optimizer_progress"]["w1"]["metrics_write_in_progress"])
            result = snapshot(root, {"ActiveState": "active"}, time.time() + 1900)
            self.assertEqual(result["health"], "stalled")
            write_json(run / "STATUS.json", {"status": "failed", "error": "bad physical count"})
            result = snapshot(root, {"ActiveState": "failed"}, time.time())
            self.assertEqual(result["health"], "failed")
            self.assertEqual(result["error"], "bad physical count")

    def test_peer_wait_is_not_stalled_and_inactive_worker_is_not_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "WORKER_STATUS.json",
                       {"run": str(root / "hwh"), "phase": "waiting_for_peer_results"})
            result = snapshot(root, {"ActiveState": "active"}, time.time() + 1900)
            self.assertEqual(result["health"], "running")
            result = snapshot(root, {"ActiveState": "inactive"}, time.time())
            self.assertEqual(result["health"], "worker_not_running")


if __name__ == "__main__":
    unittest.main()
