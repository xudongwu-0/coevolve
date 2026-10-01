"""Training-aware, supervised GPU utilization filler.

The supervisor owns one small cuBLAS worker per eligible GPU. It distinguishes
its workers from external CUDA processes, yields immediately to busy workloads,
and only fills sustained idle capacity. The worker uses a small allocation so a
new training process is not blocked by the filler's memory footprint.
"""
from __future__ import annotations

import argparse
import csv
import ctypes as C
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import Any


MODULE = "sao_bridge.adaptive_gpu_filler"
DEFAULT_BASE = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925")
DEFAULT_VENDOR = DEFAULT_BASE / "vendor"
IDLE_MATRIX_SIZE = 4096
SHARED_MATRIX_SIZE = 2048
GPU_COUNT = 8
ACTIVE_WINDOW_SECONDS = 0.01
STOP = False


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def event(kind: str, **values: Any) -> None:
    print(json.dumps({"utc": utc(), "event": kind, **values}, sort_keys=True), flush=True)


def stop_signal(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def state_dir() -> Path:
    configured = os.environ.get("COEVOLVE_GPU_FILLER_STATE_DIR")
    return Path(configured) if configured else DEFAULT_BASE / "nodes" / socket.gethostname()


def configure_process() -> Path:
    node = state_dir()
    for child in ("home", "cache", "cache/cuda", "tmp"):
        (node / child).mkdir(parents=True, exist_ok=True)
    os.environ.update(
        HOME=str(node / "home"),
        XDG_CACHE_HOME=str(node / "cache"),
        TMPDIR=str(node / "tmp"),
        CUDA_CACHE_PATH=str(node / "cache/cuda"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
    )
    os.chdir(node)
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    return node


def function(lib: Any, name: str, args: list[Any], result: Any = C.c_int) -> Any:
    value = getattr(lib, name)
    value.argtypes = args
    value.restype = result
    return value


def check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"{operation} returned error {status}")


@dataclass(frozen=True)
class ControllerConfig:
    idle_target_pct: float = 86.0
    shared_target_pct: float = 75.0
    external_busy_pct: float = 40.0
    external_low_pct: float = 5.0
    external_idle_grace_s: float = 6.0
    shared_low_cap: float = 0.15
    shared_medium_cap: float = 0.30
    shared_long_cap: float = 0.50
    shared_medium_after_s: float = 30.0
    shared_long_after_s: float = 120.0
    idle_cap: float = 0.92
    scale_up_step: float = 0.05
    scale_down_step: float = 0.15
    min_shared_free_mib: int = 8192
    thermal_yield_c: int = 82
    sample_interval_s: float = 2.0


@dataclass(frozen=True)
class ExternalLoad:
    pids: tuple[int, ...] = ()
    memory_mib: int = 0
    sm_pct: float = 0.0

    @property
    def present(self) -> bool:
        return bool(self.pids)


@dataclass(frozen=True)
class Decision:
    mode: str
    duty: float
    reason: str
    target_total_pct: float
    cap: float


def _ramp(current: float, wanted: float, config: ControllerConfig) -> float:
    if wanted < current:
        return wanted
    return min(wanted, current + config.scale_up_step)


def decide_policy(
    sample: dict[str, Any],
    external: ExternalLoad,
    current_duty: float,
    external_idle_s: float,
    *,
    reserved: bool = False,
    config: ControllerConfig = ControllerConfig(),
) -> Decision:
    free_mib = int(sample["memory_total_mib"]) - int(sample["memory_used_mib"])
    total_util = float(sample["utilization_pct"])
    if reserved:
        return Decision("yielded", 0.0, "reserved", 0.0, 0.0)
    if int(sample["temperature_c"]) >= config.thermal_yield_c:
        return Decision("yielded", 0.0, "thermal_yield", 0.0, 0.0)
    if external.present and free_mib < config.min_shared_free_mib:
        return Decision("yielded", 0.0, "memory_yield", 0.0, 0.0)

    if external.present:
        if external.sm_pct >= config.external_busy_pct:
            return Decision("yielded", 0.0, "external_busy", config.shared_target_pct, 0.0)
        if external_idle_s < config.external_idle_grace_s:
            return Decision("yielded", 0.0, "external_grace", config.shared_target_pct, 0.0)
        if external_idle_s >= config.shared_long_after_s:
            cap = config.shared_long_cap
        elif external_idle_s >= config.shared_medium_after_s:
            cap = config.shared_medium_cap
        else:
            cap = config.shared_low_cap
        wanted = min(cap, max(0.0, (config.shared_target_pct - external.sm_pct) / 100.0))
        if total_util > config.shared_target_pct + 5:
            wanted = min(
                wanted,
                max(0.0, current_duty - max(
                    config.scale_down_step,
                    (total_util - config.shared_target_pct) / 100.0,
                )),
            )
        duty = _ramp(current_duty, wanted, config)
        return Decision(
            "shared",
            round(duty, 4),
            "external_idle" if external.sm_pct <= config.external_low_pct else "external_partial",
            config.shared_target_pct,
            cap,
        )

    wanted = config.idle_cap
    if total_util > config.idle_target_pct + 4:
        wanted = max(
            0.0,
            current_duty - max(
                config.scale_down_step,
                (total_util - config.idle_target_pct) / 100.0,
            ),
        )
    elif total_util >= config.idle_target_pct - 2:
        wanted = current_duty
    duty = _ramp(current_duty, wanted, config)
    return Decision("idle", round(duty, 4), "no_external_workload",
                    config.idle_target_pct, config.idle_cap)


def parse_pmon(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 12 or not fields[0].isdigit() or not fields[1].isdigit():
            continue
        rows.append({
            "gpu": int(fields[0]),
            "pid": int(fields[1]),
            "sm_pct": 0 if fields[3] == "-" else int(fields[3]),
            "memory_pct": 0 if fields[4] == "-" else int(fields[4]),
            "framebuffer_mib": 0 if fields[9] == "-" else int(fields[9]),
            "command": " ".join(fields[11:]),
        })
    return rows


def gpu_sample() -> list[dict[str, Any]]:
    fields = (
        "index", "uuid", "utilization.gpu", "memory.used", "memory.total",
        "temperature.gpu", "power.draw",
    )
    output = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-gpu=" + ",".join(fields),
         "--format=csv,noheader,nounits"],
        text=True,
        timeout=10,
    )
    result = []
    for row in csv.reader(output.splitlines()):
        result.append({
            "index": int(row[0]),
            "uuid": row[1].strip(),
            "utilization_pct": int(row[2]),
            "memory_used_mib": int(row[3]),
            "memory_total_mib": int(row[4]),
            "temperature_c": int(row[5]),
            "power_w": float(row[6]),
        })
    return result


def compute_apps_sample() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=10,
    )
    result = []
    for row in csv.reader(output.splitlines()):
        if len(row) != 3 or not row[0].strip().startswith("GPU-"):
            continue
        try:
            memory_mib = int(row[2])
        except ValueError:
            memory_mib = 0
        result.append({
            "uuid": row[0].strip(),
            "pid": int(row[1]),
            "memory_mib": memory_mib,
        })
    return result


def pmon_sample() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        ["/usr/bin/nvidia-smi", "pmon", "-c", "1", "-s", "um"],
        text=True,
        timeout=10,
    )
    return parse_pmon(output)


def external_loads(
    samples: list[dict[str, Any]],
    apps: list[dict[str, Any]],
    pmon: list[dict[str, Any]],
    owned_pids: set[int],
) -> dict[int, ExternalLoad]:
    index_for_uuid = {sample["uuid"]: sample["index"] for sample in samples}
    pids: dict[int, set[int]] = {sample["index"]: set() for sample in samples}
    memory: dict[int, int] = {sample["index"]: 0 for sample in samples}
    sm: dict[int, int] = {sample["index"]: 0 for sample in samples}
    for app in apps:
        if app["pid"] in owned_pids or app["uuid"] not in index_for_uuid:
            continue
        gpu = index_for_uuid[app["uuid"]]
        pids[gpu].add(app["pid"])
        memory[gpu] += app["memory_mib"]
    for row in pmon:
        if row["pid"] in owned_pids or row["gpu"] not in pids:
            continue
        pids[row["gpu"]].add(row["pid"])
        sm[row["gpu"]] += row["sm_pct"]
        if row["framebuffer_mib"] and not any(
            app["pid"] == row["pid"] and index_for_uuid.get(app["uuid"]) == row["gpu"]
            for app in apps
        ):
            memory[row["gpu"]] += row["framebuffer_mib"]
    return {
        gpu: ExternalLoad(
            pids=tuple(sorted(pids[gpu])),
            memory_mib=memory[gpu],
            sm_pct=float(min(100, sm[gpu])),
        )
        for gpu in pids
    }


def read_reservations(path: Path | None, gpu_count: int) -> tuple[set[int], str | None]:
    if path is None or not path.exists():
        return set(), None
    try:
        data = json.loads(path.read_text())
        if data.get("expires_epoch") is not None and float(data["expires_epoch"]) <= time.time():
            return set(), None
        values = data.get("reserved_gpus", [])
        if "gpu" in data:
            values = [*values, data["gpu"]]
        reserved = {int(value) for value in values}
        if not reserved.issubset(set(range(gpu_count))):
            raise ValueError(f"invalid GPU indices: {sorted(reserved)}")
        return reserved, None
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return set(range(gpu_count)), str(exc)


def _read_commands(fd: int, pending: bytes, timeout: float) -> tuple[bytes, list[dict[str, Any]], bool]:
    ready, _, _ = select.select([fd], [], [], max(0.0, timeout))
    if not ready:
        return pending, [], True
    chunk = os.read(fd, 4096)
    if not chunk:
        return pending, [], False
    pending += chunk
    commands = []
    while b"\n" in pending:
        line, pending = pending.split(b"\n", 1)
        if line:
            commands.append(json.loads(line))
    return pending, commands, True


def worker(
    gpu: int,
    matrix_size: int,
    control_fd: int | None,
    initial_duty: float,
    run_seconds: float | None,
) -> None:
    node = configure_process()
    vendor = Path(os.environ.get("COEVOLVE_GPU_FILLER_VENDOR", DEFAULT_VENDOR))
    C.CDLL(str(vendor / "libcublasLt.so.13"), mode=C.RTLD_GLOBAL)
    runtime = C.CDLL(str(vendor / "libcudart.so.13"), mode=C.RTLD_GLOBAL)
    blas = C.CDLL(str(vendor / "libcublas.so.13"), mode=C.RTLD_GLOBAL)
    pointer = C.c_void_p
    set_device = function(runtime, "cudaSetDevice", [C.c_int])
    malloc = function(runtime, "cudaMalloc", [C.POINTER(pointer), C.c_size_t])
    memset = function(runtime, "cudaMemset", [pointer, C.c_int, C.c_size_t])
    memcpy = function(runtime, "cudaMemcpy", [pointer, pointer, C.c_size_t, C.c_int])
    free = function(runtime, "cudaFree", [pointer])
    priority_range = function(
        runtime, "cudaDeviceGetStreamPriorityRange", [C.POINTER(C.c_int), C.POINTER(C.c_int)],
    )
    stream_create = function(
        runtime, "cudaStreamCreateWithPriority", [C.POINTER(pointer), C.c_uint, C.c_int],
    )
    stream_sync = function(runtime, "cudaStreamSynchronize", [pointer])
    stream_destroy = function(runtime, "cudaStreamDestroy", [pointer])
    create = function(blas, "cublasCreate_v2", [C.POINTER(pointer)])
    destroy = function(blas, "cublasDestroy_v2", [pointer])
    math_mode = function(blas, "cublasSetMathMode", [pointer, C.c_int])
    set_stream = function(blas, "cublasSetStream_v2", [pointer, pointer])
    gemm = function(
        blas,
        "cublasSgemm_v2",
        [
            pointer, C.c_int, C.c_int, C.c_int, C.c_int, C.c_int,
            C.POINTER(C.c_float), pointer, C.c_int, pointer, C.c_int,
            C.POINTER(C.c_float), pointer, C.c_int,
        ],
    )

    check(set_device(gpu), "cudaSetDevice")
    least_priority, greatest_priority = C.c_int(), C.c_int()
    check(priority_range(C.byref(least_priority), C.byref(greatest_priority)),
          "cudaDeviceGetStreamPriorityRange")
    stream = pointer()
    check(stream_create(C.byref(stream), 1, least_priority.value),
          "cudaStreamCreateWithPriority")
    handle = pointer()
    check(create(C.byref(handle)), "cublasCreate")
    check(set_stream(handle, stream), "cublasSetStream")
    check(math_mode(handle, 3), "cublasSetMathMode(TF32)")

    matrix_bytes = matrix_size * matrix_size * 4
    buffers: list[Any] = []
    started = time.monotonic()
    duty = max(0.0, min(0.95, initial_duty))
    pending = b""
    batches = 0
    last_heartbeat = 0.0
    alpha, beta = C.c_float(1.0 / matrix_size), C.c_float(0)
    scalar = struct.unpack("f", b"\x3f" * 4)[0]
    expected = scalar * scalar
    try:
        for index in range(3):
            value = pointer()
            check(malloc(C.byref(value), matrix_bytes), "cudaMalloc")
            buffers.append(value)
            check(memset(value, 0x3F if index < 2 else 0, matrix_bytes), "initialize matrix")
        event(
            "worker_started",
            gpu=gpu,
            pid=os.getpid(),
            matrix_size=matrix_size,
            allocated_mib=3 * matrix_bytes // 2**20,
            stream_priority=least_priority.value,
            duty=duty,
        )
        while not STOP and (run_seconds is None or time.monotonic() - started < run_seconds):
            if control_fd is not None:
                pending, commands, parent_alive = _read_commands(control_fd, pending, 0)
                if not parent_alive:
                    break
                for command in commands:
                    if command.get("action") == "stop":
                        return
                    if command.get("action") == "duty":
                        duty = max(0.0, min(0.95, float(command["value"])))
            if duty <= 0:
                if control_fd is not None:
                    pending, commands, parent_alive = _read_commands(control_fd, pending, 0.2)
                    if not parent_alive:
                        break
                    for command in commands:
                        if command.get("action") == "stop":
                            return
                        if command.get("action") == "duty":
                            duty = max(0.0, min(0.95, float(command["value"])))
                else:
                    time.sleep(0.2)
            else:
                cycle_started = time.perf_counter()
                while time.perf_counter() - cycle_started < ACTIVE_WINDOW_SECONDS:
                    check(
                        gemm(
                            handle, 0, 0, matrix_size, matrix_size, matrix_size,
                            C.byref(alpha), buffers[0], matrix_size, buffers[1], matrix_size,
                            C.byref(beta), buffers[2], matrix_size,
                        ),
                        "cublasSgemm",
                    )
                    check(stream_sync(stream), "cudaStreamSynchronize")
                    batches += 1
                active_seconds = time.perf_counter() - cycle_started
                idle_seconds = max(0.0, active_seconds / duty - active_seconds)
                idle_deadline = time.monotonic() + idle_seconds
                while not STOP and time.monotonic() < idle_deadline:
                    wait = min(0.1, idle_deadline - time.monotonic())
                    if control_fd is None:
                        time.sleep(wait)
                        continue
                    pending, commands, parent_alive = _read_commands(control_fd, pending, wait)
                    if not parent_alive:
                        return
                    for command in commands:
                        if command.get("action") == "stop":
                            return
                        if command.get("action") == "duty":
                            duty = max(0.0, min(0.95, float(command["value"])))
                    if duty <= 0:
                        break
            if time.monotonic() - last_heartbeat >= 10:
                value = C.c_float()
                if batches:
                    check(memcpy(C.byref(value), buffers[2], 4, 2), "copy validation sample")
                    if not math.isfinite(value.value) or abs(value.value - expected) > expected * 0.01:
                        raise RuntimeError(f"GEMM validation failed: {value.value}")
                atomic_json(node / f"adaptive-worker-{gpu}.json", {
                    "utc": utc(),
                    "epoch": time.time(),
                    "gpu": gpu,
                    "pid": os.getpid(),
                    "matrix_size": matrix_size,
                    "allocated_mib": 3 * matrix_bytes // 2**20,
                    "duty": duty,
                    "batches": batches,
                    "validated_value": value.value if batches else None,
                    "expected_value": expected,
                })
                last_heartbeat = time.monotonic()
    finally:
        for value in buffers:
            free(value)
        destroy(handle)
        stream_destroy(stream)
        if control_fd is not None:
            os.close(control_fd)
    event("worker_stopped", gpu=gpu, pid=os.getpid(), batches=batches)


@dataclass
class WorkerSlot:
    process: subprocess.Popen[bytes]
    control_fd: int
    mode: str
    matrix_size: int
    duty: float
    launched_monotonic: float
    restarts: int


class Supervisor:
    def __init__(self, config: ControllerConfig, reservation: Path | None):
        self.config = config
        self.reservation = reservation
        self.node = configure_process()
        self.workers: dict[int, WorkerSlot] = {}
        self.external_idle_since: dict[int, float | None] = {
            gpu: None for gpu in range(GPU_COUNT)
        }
        self.restart_counts = {gpu: 0 for gpu in range(GPU_COUNT)}
        self.last_status_write = 0.0

    def launch(self, gpu: int, mode: str, duty: float) -> None:
        matrix_size = IDLE_MATRIX_SIZE if mode == "idle" else SHARED_MATRIX_SIZE
        read_fd, write_fd = os.pipe()
        os.set_inheritable(read_fd, True)
        command = [
            sys.executable, "-B", "-m", MODULE, "worker",
            "--gpu", str(gpu),
            "--matrix-size", str(matrix_size),
            "--control-fd", str(read_fd),
        ]
        log_path = self.node / f"adaptive-gpu-{gpu}.log"
        try:
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command,
                    cwd=self.node,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    pass_fds=(read_fd,),
                    start_new_session=True,
                )
        finally:
            os.close(read_fd)
        os.set_blocking(write_fd, False)
        slot = WorkerSlot(
            process=process,
            control_fd=write_fd,
            mode=mode,
            matrix_size=matrix_size,
            duty=0.0,
            launched_monotonic=time.monotonic(),
            restarts=self.restart_counts[gpu],
        )
        self.workers[gpu] = slot
        self.set_duty(gpu, duty)
        event(
            "worker_launched",
            gpu=gpu,
            pid=process.pid,
            mode=mode,
            matrix_size=matrix_size,
            restarts=slot.restarts,
        )

    def set_duty(self, gpu: int, duty: float) -> None:
        slot = self.workers[gpu]
        duty = round(max(0.0, min(0.95, duty)), 4)
        if abs(slot.duty - duty) < 0.005:
            return
        payload = (json.dumps({"action": "duty", "value": duty}) + "\n").encode()
        try:
            os.write(slot.control_fd, payload)
        except (BrokenPipeError, BlockingIOError, OSError) as exc:
            event("worker_control_error", gpu=gpu, pid=slot.process.pid, error=str(exc))
            self.halt(gpu)
            return
        slot.duty = duty

    def halt(self, gpu: int) -> None:
        slot = self.workers.pop(gpu, None)
        if slot is None:
            return
        try:
            os.write(slot.control_fd, b'{"action":"stop"}\n')
        except OSError:
            pass
        try:
            os.close(slot.control_fd)
        except OSError:
            pass
        if slot.process.poll() is None:
            slot.process.terminate()
            try:
                slot.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                slot.process.kill()
                slot.process.wait(timeout=5)
        event("worker_halted", gpu=gpu, pid=slot.process.pid, code=slot.process.returncode)

    def reconcile(self, gpu: int, decision: Decision) -> None:
        slot = self.workers.get(gpu)
        if slot is not None and slot.process.poll() is not None:
            event("worker_exited", gpu=gpu, pid=slot.process.pid, code=slot.process.returncode)
            try:
                os.close(slot.control_fd)
            except OSError:
                pass
            self.workers.pop(gpu)
            self.restart_counts[gpu] += 1
            slot = None
        if decision.mode == "yielded":
            self.halt(gpu)
            return
        if slot is None or slot.mode != decision.mode:
            self.halt(gpu)
            self.launch(gpu, decision.mode, decision.duty)
        else:
            self.set_duty(gpu, decision.duty)

    def heartbeat_age(self, gpu: int) -> float | None:
        slot = self.workers.get(gpu)
        if slot is None:
            return None
        try:
            data = json.loads((self.node / f"adaptive-worker-{gpu}.json").read_text())
            if data["pid"] != slot.process.pid:
                return None
            return max(0.0, time.time() - float(data["epoch"]))
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def run(self) -> None:
        initial = gpu_sample()
        if len(initial) != GPU_COUNT or {row["index"] for row in initial} != set(range(GPU_COUNT)):
            raise RuntimeError(f"Expected exactly {GPU_COUNT} GPUs: {initial!r}")
        consecutive_sample_errors = 0
        try:
            while not STOP:
                loop_started = time.monotonic()
                try:
                    samples = gpu_sample()
                    apps = compute_apps_sample()
                    pmon = pmon_sample()
                    if len(samples) != GPU_COUNT:
                        raise RuntimeError(f"Expected {GPU_COUNT} GPU samples, got {len(samples)}")
                    owned = {
                        slot.process.pid for slot in self.workers.values()
                        if slot.process.poll() is None
                    }
                    loads = external_loads(samples, apps, pmon, owned)
                    reserved, reservation_error = read_reservations(
                        self.reservation, GPU_COUNT,
                    )
                    if reservation_error:
                        event("reservation_error", path=str(self.reservation),
                              error=reservation_error, action="yield_all")
                    now = time.monotonic()
                    status_rows = []
                    for sample in samples:
                        gpu = sample["index"]
                        external = loads[gpu]
                        if external.present and external.sm_pct <= self.config.external_low_pct:
                            if self.external_idle_since[gpu] is None:
                                self.external_idle_since[gpu] = now
                        else:
                            self.external_idle_since[gpu] = None
                        idle_seconds = (
                            now - self.external_idle_since[gpu]
                            if self.external_idle_since[gpu] is not None else 0.0
                        )
                        current_duty = self.workers[gpu].duty if gpu in self.workers else 0.0
                        decision = decide_policy(
                            sample,
                            external,
                            current_duty,
                            idle_seconds,
                            reserved=gpu in reserved,
                            config=self.config,
                        )
                        self.reconcile(gpu, decision)
                        heartbeat_age = self.heartbeat_age(gpu)
                        slot = self.workers.get(gpu)
                        if (
                            slot is not None
                            and now - slot.launched_monotonic > 45
                            and (heartbeat_age is None or heartbeat_age > 35)
                        ):
                            event("worker_heartbeat_timeout", gpu=gpu, pid=slot.process.pid,
                                  heartbeat_age_s=heartbeat_age)
                            self.halt(gpu)
                            self.restart_counts[gpu] += 1
                            self.launch(gpu, decision.mode, decision.duty)
                            slot = self.workers.get(gpu)
                            heartbeat_age = self.heartbeat_age(gpu)
                        status_rows.append({
                            **sample,
                            "external": asdict(external),
                            "external_idle_s": round(idle_seconds, 3),
                            "decision": asdict(decision),
                            "worker_pid": slot.process.pid if slot else None,
                            "worker_mode": slot.mode if slot else None,
                            "worker_duty": slot.duty if slot else 0.0,
                            "worker_matrix_size": slot.matrix_size if slot else None,
                            "worker_allocated_mib": (
                                3 * slot.matrix_size * slot.matrix_size * 4 // 2**20
                                if slot else 0
                            ),
                            "worker_alive": bool(slot and slot.process.poll() is None),
                            "worker_heartbeat_age_s": heartbeat_age,
                            "worker_restarts": self.restart_counts[gpu],
                        })
                    if now - self.last_status_write >= 5:
                        atomic_json(self.node / "adaptive-status.json", {
                            "utc": utc(),
                            "epoch": time.time(),
                            "hostname": socket.gethostname(),
                            "supervisor_pid": os.getpid(),
                            "controller": asdict(self.config),
                            "reservation_path": (
                                str(self.reservation) if self.reservation else None
                            ),
                            "reservation_error": reservation_error,
                            "reserved_gpus": sorted(reserved),
                            "gpus": status_rows,
                            "external_sm_mean_pct": (
                                sum(row["external"]["sm_pct"] for row in status_rows) / GPU_COUNT
                            ),
                            "filler_duty_mean": (
                                sum(row["worker_duty"] for row in status_rows) / GPU_COUNT
                            ),
                            "total_utilization_mean_pct": (
                                sum(row["utilization_pct"] for row in status_rows) / GPU_COUNT
                            ),
                        })
                        self.last_status_write = now
                    consecutive_sample_errors = 0
                except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
                    consecutive_sample_errors += 1
                    event("telemetry_error", error=str(exc),
                          consecutive=consecutive_sample_errors, action="yield")
                    for gpu in list(self.workers):
                        self.set_duty(gpu, 0.0)
                    if consecutive_sample_errors >= 3:
                        raise RuntimeError("GPU telemetry failed three consecutive times") from exc
                elapsed = time.monotonic() - loop_started
                if elapsed < self.config.sample_interval_s:
                    time.sleep(self.config.sample_interval_s - elapsed)
        finally:
            for gpu in list(self.workers):
                self.halt(gpu)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    supervisor_parser = subparsers.add_parser("supervisor")
    supervisor_parser.add_argument(
        "--reservation",
        type=Path,
        default=(
            Path(os.environ["COEVOLVE_GPU_FILLER_RESERVATION"])
            if os.environ.get("COEVOLVE_GPU_FILLER_RESERVATION") else None
        ),
    )
    worker_parser = subparsers.add_parser("worker")
    worker_parser.add_argument("--gpu", type=int, required=True)
    worker_parser.add_argument("--matrix-size", type=int, required=True)
    worker_parser.add_argument("--control-fd", type=int)
    worker_parser.add_argument("--duty", type=float, default=0.0)
    worker_parser.add_argument("--run-seconds", type=float)
    args = parser.parse_args()
    if args.mode == "supervisor":
        Supervisor(ControllerConfig(), args.reservation).run()
    else:
        if args.gpu not in range(GPU_COUNT):
            parser.error(f"--gpu must be in [0, {GPU_COUNT - 1}]")
        if args.matrix_size not in {SHARED_MATRIX_SIZE, IDLE_MATRIX_SIZE}:
            parser.error(
                f"--matrix-size must be {SHARED_MATRIX_SIZE} or {IDLE_MATRIX_SIZE}"
            )
        worker(
            args.gpu,
            args.matrix_size,
            args.control_fd,
            args.duty,
            args.run_seconds,
        )


if __name__ == "__main__":
    main()
