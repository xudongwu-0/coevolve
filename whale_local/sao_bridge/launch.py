"""Attached node-local process ownership and bounded resource supervision."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import httpx
import psutil

from .common import ROOT, write_json
from .resources import require_research_node, stop_known


class Launcher:
    def __init__(self, root: Path, budget=None):
        require_research_node()
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.budget = budget
        self.children = []
        self.owned_identities = {}
        self.owned_by_root = {}
        self.events = []
        self.last_audit = 0.0
        self._stop_watchdog = threading.Event()
        self._watchdog_error = None
        self._audit_lock = threading.Lock()
        cache = ROOT / "assets/runtime-cache" / os.uname().nodename
        for name in ("triton", "torch", "vllm", "cuda", "areal", "flashinfer", "home"):
            (cache / name).mkdir(parents=True, exist_ok=True)
        ipc = Path("/dev/shm") / f"sao-{os.getuid()}-{self.root.name}"
        ipc.mkdir(mode=0o700, exist_ok=True)
        self.env = dict(
            # Children use the running interpreter's venv (a node-local tmpfs copy imports
            # vLLM ~6x faster than the shared 3FS copy; code, models and data stay shared).
            os.environ, PATH=str(Path(sys.prefix) / "bin") + ":" + os.environ["PATH"],
            PYTHONPATH=":".join(map(str, (
                ROOT / "whale_local", ROOT / "third_party/sao/vendor/AReaL",
                ROOT / "third_party/sao",
            ))),
            VLLM_PLUGINS="", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
            TOKENIZERS_PARALLELISM="false", WANDB_MODE="disabled",
            LITELLM_LOCAL_MODEL_COST_MAP="True",
            NCCL_IB_DISABLE="1", NCCL_NET="Socket",
            TMPDIR=str(ipc), TRITON_CACHE_DIR=str(cache / "triton"),
            TORCHINDUCTOR_CACHE_DIR=str(cache / "torch"),
            VLLM_CACHE_ROOT=str(cache / "vllm"), CUDA_CACHE_PATH=str(cache / "cuda"),
            AREAL_CACHE_DIR=str(cache / "areal"),
            FLASHINFER_WORKSPACE_BASE=str(cache / "flashinfer"),
            HOME=str(cache / "home"),
            PYTHONDONTWRITEBYTECODE="1",
            PYTORCH_ALLOC_CONF=os.environ.get(
                "PYTORCH_ALLOC_CONF", "expandable_segments:True"
            ),
            GLOO_SOCKET_IFNAME="lo", NCCL_SOCKET_IFNAME="lo",
            AREAL_HOST_IP="127.0.0.1",
        )
        self._watchdog = threading.Thread(target=self._watch, name="sao-resource-watchdog", daemon=True)
        self._watchdog.start()

    def _watch(self):
        while not self._stop_watchdog.wait(20):
            try:
                self.check()
            except (OSError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
                self._watchdog_error = exc
                write_json(self.root / "watchdog_error.json", {
                    "error": repr(exc), "traceback": traceback.format_exc(),
                })
                for child in reversed(self.children):
                    self.stop(child)
                return

    def check(self):
        if self._watchdog_error is not None:
            raise RuntimeError("Resource/budget watchdog stopped experiment") from self._watchdog_error
        if self.budget:
            self.budget.check()
        self.remember_owned()
        if time.monotonic() - self.last_audit > 60:
            with self._audit_lock:
                if time.monotonic() - self.last_audit <= 60:
                    return
                event = stop_known(require_exclusive=False)
                self.last_audit = time.monotonic()
            # Workers may fork while nvidia-smi is being sampled; refresh ancestry
            # after the snapshot before classifying an unknown same-user PID.
            self.remember_owned()
            foreign = [p["pid"] for p in event["processes"]
                       if not p["verified_filler"] and self.is_foreign(p["pid"])]
            if foreign:
                write_json(self.root / "foreign_workload.json", event)
                raise RuntimeError(f"Non-experiment GPU workload appeared: {foreign}")
            if event["stopped"]:
                self.events.append(event)
                write_json(self.root / "resource_events.json", self.events)

    def remember_owned(self):
        for child in self.children:
            if child.poll() is not None:
                continue
            try:
                root = psutil.Process(child.pid)
                tracked = self.owned_by_root.setdefault(child.pid, {})
                for process in [root, *root.children(recursive=True)]:
                    try:
                        identity = (process.create_time(), process.uids().real)
                        tracked[process.pid] = identity
                        if identity[1] == os.getuid():
                            self.owned_identities[process.pid] = identity[0]
                    except psutil.NoSuchProcess:
                        continue
            except psutil.NoSuchProcess:
                continue

    def is_foreign(self, pid):
        try:
            process = psutil.Process(pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return False
            return (process.uids().real != os.getuid()
                    or self.owned_identities.get(pid) != process.create_time())
        except psutil.NoSuchProcess:
            return False

    def start(self, command: list[str], *, gpus: str, name: str):
        self.check()
        # Serialize Popen + ownership registration with GPU audits. Otherwise a new
        # worker can appear in nvidia-smi before its root is registered.
        with self._audit_lock:
            log_path = self.root / f"{name}.log"
            if log_path.exists() or log_path.is_symlink():
                archive_root = self.root / "previous_logs"
                archive_root.mkdir(exist_ok=True)
                previous = archive_root / f"{log_path.stem}-{time.time_ns()}.log"
                if previous.exists() or previous.is_symlink():
                    raise FileExistsError(previous)
                log_path.rename(previous)
            with log_path.open("wb") as log:
                process = subprocess.Popen(
                    command, cwd=ROOT, env=dict(self.env, CUDA_VISIBLE_DEVICES=gpus),
                    stdout=log, stderr=subprocess.STDOUT,
                )
            self.children.append(process)
            self.owned_identities[process.pid] = psutil.Process(process.pid).create_time()
            self.owned_by_root[process.pid] = {
                process.pid: (self.owned_identities[process.pid], os.getuid())}
            write_json(self.root / "processes.json", [
                {"pid": p.pid, "command": p.args, "returncode": p.poll()} for p in self.children
            ])
        print(f"{name}: PID {process.pid}, GPUs {gpus}", flush=True)
        return process

    def module(self, module: str, args: list, *, gpus: str, name: str):
        return self.start(
            [sys.executable, "-B", "-m", module, *map(str, args)], gpus=gpus, name=name,
        )

    def wait(self, process, name: str, timeout: float = 43200):
        start = time.monotonic()
        while process.poll() is None:
            self.check()
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"{name} timed out")
            time.sleep(2)
        if process.returncode != 0:
            raise RuntimeError(f"{name} failed ({process.returncode}); inspect its log")

    def ready(self, process, url: str, timeout=900):
        start = time.monotonic()
        with httpx.Client(trust_env=False, timeout=2) as client:
            while time.monotonic() - start < timeout:
                self.check()
                if process.poll() is not None:
                    raise RuntimeError(f"Service {url} exited before readiness")
                try:
                    response = client.get(url)
                    if response.status_code == 200:
                        return
                except (httpx.ConnectError, httpx.TimeoutException):
                    pass
                time.sleep(2)
        raise TimeoutError(f"Service {url} readiness exceeded {timeout}s")

    def stop(self, process):
        tracked = dict(self.owned_by_root.get(process.pid, {}))
        try:
            root = psutil.Process(process.pid)
            for child in [root, *root.children(recursive=True)]:
                try:
                    tracked[child.pid] = (child.create_time(), child.uids().real)
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            root = None

        owned = []
        privileged = []
        for pid, (created, uid) in tracked.items():
            try:
                candidate = psutil.Process(pid)
                if candidate.create_time() != created:
                    continue
                (owned if uid == os.getuid() else privileged).append(candidate)
            except psutil.NoSuchProcess:
                continue
        for child in reversed(owned):
            try:
                child.terminate()
            except psutil.NoSuchProcess:
                pass
        _, alive = psutil.wait_procs(owned, timeout=20)
        for child in alive:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        psutil.wait_procs(alive, timeout=20)

        # sudo/bwrap are root-owned and cannot be signalled by this user. Their code
        # execution is bounded to 15 seconds; require them to exit rather than leak.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            remaining = []
            for child in privileged:
                try:
                    if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                        remaining.append(child)
                except psutil.NoSuchProcess:
                    pass
            if not remaining:
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(
                "Privileged sandbox descendants survived their bounded parent shutdown: "
                + str([child.pid for child in remaining]))
        if process.poll() is None:
            process.wait(timeout=20)
        self.owned_by_root.pop(process.pid, None)

    def close(self):
        self._stop_watchdog.set()
        self._watchdog.join(timeout=60)
        for child in reversed(self.children):
            self.stop(child)

    @contextmanager
    def vllm(self, model: str, *, port: int, gpus: str, name: str,
             tensor_parallel: int = 1, data_parallel: int = 1,
             max_model_len: int = 8192, tool_call_parser: str = "hermes",
             gdn_prefill_backend: str | None = None,
             language_model_only: bool = False,
             max_num_seqs: int = 256,
             server_args: list[str] | None = None):
        devices = gpus.split(",")
        if len(devices) != tensor_parallel * data_parallel:
            raise ValueError("Server allocation does not match parallel replica count")
        if max_model_len < 1:
            raise ValueError("Server context length must be positive")
        processes, urls = [], []
        try:
            for index in range(data_parallel):
                replica_port = port + index * 10
                assigned = ",".join(devices[index*tensor_parallel:(index+1)*tensor_parallel])
                process = self.module(
                    "vllm.entrypoints.openai.api_server",
                    ["--model", model, "--host", "127.0.0.1", "--port", replica_port,
                     "--served-model-name", "policy", "--dtype", "bfloat16",
                     "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.85",
                     "--tensor-parallel-size", tensor_parallel,
                     "--max-num-seqs", max_num_seqs,
                     "--distributed-executor-backend", "mp",
                     "--enable-auto-tool-choice", "--tool-call-parser", tool_call_parser,
                     "--generation-config", "vllm"]
                    + (["--gdn-prefill-backend", gdn_prefill_backend]
                       if gdn_prefill_backend else [])
                    + (["--language-model-only"] if language_model_only else [])
                    + list(server_args or []),
                    gpus=assigned, name=name + f"_replica{index}",
                )
                processes.append(process)
                urls.append(f"http://127.0.0.1:{replica_port}/v1")
            for process, url in zip(processes, urls):
                self.ready(process, url.removesuffix("/v1") + "/health")
            yield ",".join(urls)
        finally:
            for process in reversed(processes):
                self.stop(process)
