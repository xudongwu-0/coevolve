import os
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import sys
import unittest
from unittest.mock import Mock, patch

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.launch import Launcher


class OwnershipTests(unittest.TestCase):
    def test_retry_preserves_previous_process_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            launcher = object.__new__(Launcher)
            launcher.root = Path(temporary)
            launcher.env = {}
            launcher.children = []
            launcher.owned_identities = {}
            launcher.owned_by_root = {}
            launcher._audit_lock = threading.Lock()
            launcher.check = Mock()
            process = Mock(pid=os.getpid(), args=["fixture"])
            process.poll.return_value = None
            log = launcher.root / "evaluation.log"
            log.write_bytes(b"original startup error\n")
            with patch("sao_bridge.launch.subprocess.Popen", return_value=process):
                launcher.start(["fixture"], gpus="", name="evaluation")
                log.write_bytes(b"second attempt\n")
                launcher.start(["fixture"], gpus="", name="evaluation")
            archived = sorted(
                path.read_bytes() for path in (launcher.root / "previous_logs").glob("*.log")
            )
            self.assertEqual(archived, [b"original startup error\n", b"second attempt\n"])
            self.assertEqual(log.read_bytes(), b"")

    def test_default_cuda_allocator_uses_expandable_segments(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ, {}, clear=False
        ):
            os.environ.pop("PYTORCH_ALLOC_CONF", None)
            launcher = Launcher(Path(temporary))
            self.assertEqual(
                launcher.env["PYTORCH_ALLOC_CONF"], "expandable_segments:True"
            )

    def test_reparented_worker_remains_owned_but_reused_pid_does_not(self):
        launcher = object.__new__(Launcher)
        launcher.owned_identities = {42: 100.0}
        process = Mock()
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING
        process.uids.return_value.real = os.getuid()
        process.create_time.return_value = 100.0
        with patch("sao_bridge.launch.psutil.Process", return_value=process):
            self.assertFalse(launcher.is_foreign(42))
            process.create_time.return_value = 200.0
            self.assertTrue(launcher.is_foreign(42))
            process.status.return_value = psutil.STATUS_ZOMBIE
            self.assertFalse(launcher.is_foreign(42))

    def test_unknown_live_gpu_work_is_foreign(self):
        launcher = object.__new__(Launcher)
        launcher.owned_identities = {}
        process = Mock()
        process.is_running.return_value = True
        process.status.return_value = psutil.STATUS_RUNNING
        process.uids.return_value.real = os.getuid()
        with patch("sao_bridge.launch.psutil.Process", return_value=process):
            self.assertTrue(launcher.is_foreign(42))
        with patch("sao_bridge.launch.psutil.Process", side_effect=psutil.NoSuchProcess(42)):
            self.assertFalse(launcher.is_foreign(42))

    def test_stop_cleans_a_tracked_child_after_direct_parent_exits(self):
        launcher = object.__new__(Launcher)
        launcher.children = []
        launcher.owned_identities = {}
        launcher.owned_by_root = {}
        parent = subprocess.Popen([sys.executable, "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); time.sleep(1)"])
        launcher.children.append(parent)
        time.sleep(0.3)
        launcher.remember_owned()
        tracked = set(launcher.owned_by_root[parent.pid]) - {parent.pid}
        self.assertTrue(tracked)
        parent.wait(timeout=5)
        launcher.stop(parent)
        self.assertTrue(all(not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
                            for pid in tracked))


if __name__ == "__main__":
    unittest.main()
