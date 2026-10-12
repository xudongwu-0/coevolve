"""Sample both research nodes before an explicitly reauthorized native continuation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback

from .alfworld_train_data import _immutable_json
from .common import ROOT, file_sha, read_json, verify_upstreams, write_json

INTERVAL = 600
IDLE_SECONDS = 7200
MIN_SAMPLES = 13
MAX_GAP = 660
HOSTS = {"node3": "dev-NDH200-025", "node4": "dev-NDH200-026"}
ARMS = ("without_peer", "without_context", "without_value_norm", "always_joint")
SCHEMA = "ma-sao-idle2h-resume-v1"
SERIAL_SCHEMA = "ma-sao-serial15-resume-v1"
SERIAL_GPU_INDICES = {"node3": list(range(8)), "node4": list(range(1, 8))}
PAUSE_RECORDS = (
    "USER_NO_H200_AND_HANDOFF.json",
    "TWO_HOUR_RESUME_CANCELLED.json",
    "without_peer/commit_recovery01/CANCELLED.json",
)
SERIAL_PAUSE_RECORDS = (*PAUSE_RECORDS, "idle2h_20261012/CANCELLED.json")


@dataclass
class IdleWindow:
    first_monotonic: float | None = None
    first_unix: float | None = None
    last_monotonic: float | None = None
    last_unix: float | None = None
    samples: int = 0
    elapsed: float = 0
    reset_reason: str | None = None

    def reset(self, reason: str) -> None:
        self.first_monotonic = self.first_unix = None
        self.last_monotonic = self.last_unix = None
        self.samples = 0
        self.elapsed = 0
        self.reset_reason = reason

    def observe(self, eligible: bool, monotonic: float, unix: float) -> bool:
        if not eligible:
            self.reset("busy_or_unverifiable_sample")
            return False
        if self.last_monotonic is not None:
            gap = monotonic - self.last_monotonic
            wall_gap = unix - self.last_unix
            if gap < 0 or gap > MAX_GAP or abs(gap - wall_gap) > 60:
                self.reset("sampling_gap_or_clock_change")
            elif gap < INTERVAL - 60:
                return False
        if self.first_monotonic is None:
            self.first_monotonic, self.first_unix = monotonic, unix
        self.last_monotonic, self.last_unix = monotonic, unix
        self.samples += 1
        self.elapsed = min(monotonic - self.first_monotonic, unix - self.first_unix)
        return self.samples >= MIN_SAMPLES and self.elapsed >= IDLE_SECONDS


def verify_authority(path: Path, digest: str) -> dict:
    if not path.is_absolute() or path.is_symlink() or file_sha(path) != digest:
        raise ValueError("The new idle-resume authority changed")
    policy = read_json(path)
    root = Path(policy["run_root"])
    serial = policy["schema"] == SERIAL_SCHEMA
    pause_records = SERIAL_PAUSE_RECORDS if serial else PAUSE_RECORDS
    if (
        policy["schema"] not in (SCHEMA, SERIAL_SCHEMA) or policy["h200_training_allowed"] is not True
        or policy["automatic_submission_allowed"] is not True
        or policy["supersedes_prior_pause"] is not True
        or policy["poll_interval_seconds"] != INTERVAL
        or policy["minimum_idle_seconds"] != (0 if serial else IDLE_SECONDS)
        or policy["minimum_samples"] != (1 if serial else MIN_SAMPLES)
        or policy["nodes"] != HOSTS or policy["arms"] != list(ARMS)
        or policy["expires_at_unix"] is not None or not root.is_absolute()
        or set(policy["superseded_cancellations"]) != set(pause_records)
    ):
        raise ValueError("The idle-resume authority has an unexpected scope")
    if serial and (
        policy.get("gpu_indices") != SERIAL_GPU_INDICES
        or any(type(index) is not int for indices in policy["gpu_indices"].values() for index in indices)
        or policy.get("launch_mode") != "immediate_serial"
        or policy.get("learner_ranks_per_role") != 4
        or policy.get("retry_after_interruption") is not False
        or policy.get("retention_policy") != "original_latest_native_and_hf96_hf144"
        or policy.get("self_failure_recovery") != "diagnose_fix_then_native_resume"
        or policy.get("external_kill_recovery") != "stop_no_restart"
        or policy.get("evaluation_host") != HOSTS["node3"]
        or policy.get("evaluation_replicas") != 8
    ):
        raise ValueError("The serial-training authority changed its fixed GPU or recovery scope")
    if (path.parent / "CANCELLED.json").exists() or (root / "CANCELLED_BEFORE_TRAINING.json").exists():
        raise InterruptedError("The new idle-resume controller was cancelled")
    for relative, expected in policy["superseded_cancellations"].items():
        if file_sha(root / relative) != expected:
            raise InterruptedError("A prior pause directive changed after the new authorization")
    if file_sha(root / "PROTOCOL.json") != policy["original_protocol_sha256"]:
        raise ValueError("The frozen scientific protocol changed")
    if read_json(root / "PROTOCOL.json")["original_absolute_deadline_unix"] != policy["original_deadline_unix"]:
        raise ValueError("The authorization changed the original deadline's provenance")
    return policy


def execution_deadline(original: float, binding: dict | None, *, root: Path, arm: str) -> float:
    if binding is None:
        return original
    policy = verify_authority(Path(binding["authority"]), binding["authority_sha256"])
    admission_path = Path(binding["admission"])
    if (
        root.resolve() != Path(policy["run_root"]).resolve() or arm not in policy["arms"]
        or original != policy["original_deadline_unix"]
        or admission_path.parent != Path(binding["authority"]).parent / "admissions"
        or file_sha(admission_path) != binding["admission_sha256"]
    ):
        raise ValueError("Execution is outside the admitted continuation scope")
    admission = read_json(admission_path)
    serial = policy["schema"] == SERIAL_SCHEMA
    if (
        admission["authority_sha256"] != binding["authority_sha256"]
        or admission["eligible"] is not True
        or admission["samples"] < (1 if serial else MIN_SAMPLES)
        or admission["elapsed_seconds"] < (0 if serial else IDLE_SECONDS)
        or admission["first_sample_unix"] < policy["authorized_at_unix"]
        or admission["last_sample_unix"] - admission["first_sample_unix"] < (0 if serial else IDLE_SECONDS)
        or admission["at_unix"] > time.time()
        or admission["final_recheck"]["eligible"] is not True
    ):
        raise ValueError("Native continuation lacks a verified two-hour admission")
    if serial and (
        admission.get("mode") != "immediate_serial"
        or admission.get("gpu_indices") != SERIAL_GPU_INDICES
        or not 0 <= admission["at_unix"] - admission["last_sample_unix"] <= 60
        or type(admission["samples"]) is not int
        or admission["samples"] != 1 or admission["elapsed_seconds"] != 0
        or admission["first_sample_unix"] != admission["last_sample_unix"]
        or any(
            admission["final_recheck"]["nodes"][node].get("allocated_gpu_indices") != indices
            or classify_node(admission["final_recheck"]["nodes"][node]) != "idle"
            for node, indices in SERIAL_GPU_INDICES.items()
        )
    ):
        raise ValueError("Native continuation lacks its immediate partition admission")
    # Keep the old deadline inside the unchanged scientific identity. Only this
    # separately pinned authorization replaces its execution-time limit.
    return math.inf


def spec_deadline(spec: dict) -> float:
    options = spec.get("rolling_recovery", {})
    binding = options.get("execution_authority")
    if binding is None:
        return spec["training_deadline_unix"]
    return execution_deadline(
        spec["training_deadline_unix"], binding,
        root=Path(options["root"]).parent.parent, arm=spec["component_ablation"],
    )


def classify_node(value: dict) -> str:
    cards = value["gpus"]
    if (
        len(cards) != 8 or {card["index"] for card in cards} != set(range(8))
        or any(card["name"] != "NVIDIA H200" for card in cards)
        or any(type(card["utilization"]) is not int or not 0 <= card["utilization"] <= 100
               or type(card["memory_mib"]) is not int or card["memory_mib"] < 0 for card in cards)
    ):
        raise ValueError("Expected eight fully observable H200 GPUs")
    allocated = value.get("allocated_gpu_indices", list(range(8)))
    if allocated not in (list(range(8)), list(range(1, 8))):
        raise ValueError("Unexpected physical GPU partition")
    cards = [card for card in cards if card["index"] in allocated]
    if value["foreign_gpu_pids"] or value["vanished_gpu_pids"] or value["distributed_launchers"]:
        return "busy_or_unverifiable"
    if not value["gpu_pids"]:
        return "idle" if all(card["utilization"] == card["memory_mib"] == 0 for card in cards) else "unattributed_gpu_activity"
    if (
        value["verified_filler_unit"] is True and value["filler_kind"] in ("adaptive", "legacy")
        and set(value["gpu_pids"]) == set(value["verified_filler_pids"])
    ):
        return "verified_filler_only"
    return "unverified_gpu_process"


def probe_local() -> dict:
    from .alfworld_ma_rolling_recovery import distributed_launchers
    from .resources import scoped_audit, require_research_node

    require_research_node()
    state = scoped_audit()
    response = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=30,
    )
    cards = []
    for line in response.stdout.splitlines():
        index, name, utilization, memory = (part.strip() for part in line.split(","))
        cards.append({"index": int(index), "name": name,
                      "utilization": int(utilization), "memory_mib": int(memory)})
    value = {
        "host": state["host"], "gpus": cards,
        "gpu_pids": sorted({item["pid"] for item in state["processes"]}),
        "verified_filler_pids": sorted({item["pid"] for item in state["processes"] if item["verified_filler"]}),
        "foreign_gpu_pids": sorted({item["pid"] for item in state["processes"] if not item["verified_filler"]}),
        "vanished_gpu_pids": state["vanished_gpu_pids"],
        "verified_filler_unit": state["verified_unit"], "filler_kind": state["filler_kind"],
        "distributed_launchers": distributed_launchers(strict=True), "at_unix": time.time(),
    }
    if "resource_scope" in state:
        value["allocated_gpu_indices"] = state["resource_scope"]["gpu_indices"]
        value["resource_scope"] = state["resource_scope"]
    value["classification"] = classify_node(value)
    return value


def resource_snapshot() -> dict:
    from .training_speed_campaign import PYTHON, SSH
    from .resources import scope_environment

    command = [
        "env", "CUDA_VISIBLE_DEVICES=", "HF_HUB_OFFLINE=1", "PYTHONDONTWRITEBYTECODE=1",
        *(f"{name}={value}" for name, value in scope_environment().items()),
        f"PYTHONPATH={ROOT}/whale_local:{ROOT}/third_party/sao/vendor/AReaL",
        PYTHON, "-B", "-m", "sao_bridge.alfworld_ma_idle_resume", "--probe-local",
    ]

    def node(name: str) -> dict:
        args = command if name == "node4" else [
            *SSH[:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH[-1], shlex.join(command),
        ]
        response = subprocess.run(
            args, capture_output=True, text=True, check=True, timeout=90,
        )
        return json.loads(response.stdout)

    with ThreadPoolExecutor(max_workers=2) as pool:
        node4, node3 = pool.submit(node, "node4"), pool.submit(node, "node3")
        nodes = {"node4": node4.result(), "node3": node3.result()}
    for name, node in nodes.items():
        if node["host"] != HOSTS[name] or abs(time.time() - node["at_unix"]) > 60:
            raise ValueError("A resource sample came from the wrong host or is stale")
        node["classification"] = classify_node(node)
    return {
        "nodes": nodes, "at_unix": time.time(),
        "eligible": all(node["classification"] in ("idle", "verified_filler_only") for node in nodes.values()),
    }


def verify_prepared(control: Path) -> dict:
    from . import alfworld_ma_mechanism_campaign as campaign
    from . import alfworld_ma_mechanism_study as study
    from .joint_recovery import latest_checkpoint, science_digest

    prepared = read_json(control / "PREPARED.json")
    policy = verify_authority(control / "AUTHORIZATION.json", prepared["authority_sha256"])
    root = Path(policy["run_root"])
    if (
        prepared["status"] != "prepared"
        or Path(prepared["execution_source"]).resolve() != ROOT.resolve()
        or prepared["execution_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
    ):
        raise ValueError("The new monitor is outside its validated frozen source")
    for name, digest in prepared["pins"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f"Continuation validation evidence changed: {name}")
    source = read_json(ROOT / "SOURCE_MANIFEST.json")
    for name, digest in source["runtime_files"].items():
        if file_sha(ROOT / name) != digest:
            raise ValueError(f"Frozen runtime changed: {name}")
    verify_upstreams()
    study.verify_inputs(root)
    campaign.require_full_control_review(root)
    for arm in ARMS:
        checkpoint = latest_checkpoint(root / arm / "recovery")
        if checkpoint is not None:
            for role in ("editor", "solver"):
                spec = read_json(root / arm / f"{role}_spec.json")
                if read_json(checkpoint.parent / f"{role}.json")["science_sha256"] != science_digest(spec):
                    raise ValueError("Saved checkpoint no longer matches its original scientific spec")
    return policy


def record_status(control: Path, value: dict, *, waiting: bool = False) -> None:
    record = {**value, "pid": os.getpid(), "at_unix": time.time()}
    write_json(control / "STATUS.json", record)
    if waiting:
        root = Path(read_json(control / "AUTHORIZATION.json")["run_root"])
        write_json(root / "PROGRAM_MONITOR.json", {**record, "controller": str(control)})


def sample_resources() -> dict:
    try:
        return resource_snapshot()
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"RESOURCE PROBE ERROR: {exc!r}", flush=True)
        return {"eligible": False, "probe_error": repr(exc), "at_unix": time.time()}


def wait_for_window(control: Path, authority_sha256: str) -> dict:
    policy = verify_authority(control / "AUTHORIZATION.json", authority_sha256)
    if policy["schema"] == SERIAL_SCHEMA:
        verify_prepared(control)
        final = resource_snapshot()
        verify_authority(control / "AUTHORIZATION.json", authority_sha256)
        if not final["eligible"] or any(
            final["nodes"][node].get("allocated_gpu_indices") != indices
            or classify_node(final["nodes"][node]) != "idle"
            for node, indices in SERIAL_GPU_INDICES.items()
        ):
            raise ResourcesChanged("The allocated15 GPUs are not idle before immediate admission")
        observed = final["at_unix"]
        path = control / "admissions" / f"{time.time_ns()}.json"
        _immutable_json(path, {
            "eligible": True, "authority_sha256": authority_sha256, "mode": "immediate_serial",
            "gpu_indices": SERIAL_GPU_INDICES, "samples": 1, "elapsed_seconds": 0,
            "first_sample_unix": observed, "last_sample_unix": observed,
            "final_recheck": final, "at_unix": time.time(),
        })
        return {
            "authority": str(control / "AUTHORIZATION.json"), "authority_sha256": authority_sha256,
            "admission": str(path), "admission_sha256": file_sha(path),
        }
    window = IdleWindow()
    while True:
        verify_authority(control / "AUTHORIZATION.json", authority_sha256)
        snapshot = sample_resources()
        observed, sampled_at = time.monotonic(), time.time()
        ready = window.observe(snapshot["eligible"], observed, sampled_at)
        next_check = observed + INTERVAL
        event = {
            "status": "verifying_before_launch" if ready else "waiting_for_two_hour_window",
            "resources": snapshot, "consecutive_samples": window.samples,
            "observed_idle_seconds": window.elapsed, "first_sample_unix": window.first_unix,
            "earliest_start_unix": window.first_unix + IDLE_SECONDS if window.first_unix is not None else None,
            "next_check_unix": sampled_at + INTERVAL, "reset_reason": window.reset_reason,
            "gpu_work_started": False,
        }
        with (control / "SAMPLES.jsonl").open("a") as stream:
            stream.write(json.dumps({**event, "at_unix": sampled_at}) + "\n")
        record_status(control, event, waiting=True)
        print(f"RESOURCE samples={window.samples}/{MIN_SAMPLES} elapsed={window.elapsed:.1f}/{IDLE_SECONDS} "
              f"eligible={snapshot['eligible']}", flush=True)
        if ready:
            verify_prepared(control)
            final = sample_resources()
            verify_authority(control / "AUTHORIZATION.json", authority_sha256)
            if final["eligible"] and time.monotonic() - observed <= MAX_GAP:
                admission = {
                    "eligible": True, "authority_sha256": authority_sha256,
                    "samples": window.samples, "elapsed_seconds": window.elapsed,
                    "first_sample_unix": window.first_unix, "last_sample_unix": sampled_at,
                    "sampled_not_continuous_telemetry": True,
                    "final_recheck": final, "at_unix": time.time(),
                }
                path = control / "admissions" / f"{time.time_ns()}.json"
                _immutable_json(path, admission)
                return {
                    "authority": str(control / "AUTHORIZATION.json"),
                    "authority_sha256": authority_sha256,
                    "admission": str(path), "admission_sha256": file_sha(path),
                }
            window.reset("prelaunch_recheck_failed_or_late")
            record_status(control, {
                "status": "waiting_for_two_hour_window", "resources": final,
                "consecutive_samples": 0, "observed_idle_seconds": 0,
                "reset_reason": window.reset_reason, "gpu_work_started": False,
            }, waiting=True)
        while time.monotonic() < next_check:
            verify_authority(control / "AUTHORIZATION.json", authority_sha256)
            time.sleep(max(0, min(5, next_check - time.monotonic())))


def pending_arms(root: Path) -> list[str]:
    from .alfworld_ma_mechanism_program import checked_result

    pending = []
    for arm in ARMS:
        if (root / arm / "FINAL_AUDIT.json").exists():
            checked_result(root, arm)
        elif any((root / arm / name).exists() for name in ("RESULT.json", "TRAINING_AUDIT.json")):
            raise ValueError(f"{arm} needs evaluation/report recovery, not automatic retraining")
        else:
            pending.append(arm)
    return pending


class ResourcesChanged(RuntimeError):
    pass


def execute_worker(control: Path, session: Path) -> None:
    from . import alfworld_ma_mechanism_program as program
    from . import alfworld_ma_rolling_recovery as rolling

    policy = verify_prepared(control)
    root = Path(policy["run_root"])
    binding = read_json(session / "BINDING.json")
    for arm in pending_arms(root):
        execution_deadline(policy["original_deadline_unix"], binding, root=root, arm=arm)
        if not rolling.admission_snapshot()["released"]:
            raise ResourcesChanged("Resources changed before starting the next declared arm")
        if not (root / arm / "STARTED.json").exists():
            _immutable_json(root / arm / "STARTED.json", {
                "pid": os.getpid(), "execution_authority": binding, "at_unix": time.time(),
            })
        write_json(session / "WORKER_STATUS.json", {
            "phase": "training", "arm": arm, "pid": os.getpid(), "at_unix": time.time(),
        })
        training = rolling.run_attempt(root, arm, execution_authority=binding)
        write_json(session / "WORKER_STATUS.json", {
            "phase": "evaluating", "arm": arm, "pid": os.getpid(), "at_unix": time.time(),
        })
        if policy["schema"] == SERIAL_SCHEMA:
            finish_on_node3(control, session, arm, training)
        else:
            rolling.finish_arm(root, arm, training)
        program.write_master_report(root)
    _immutable_json(session / "COMPLETE.json", {"status": "completed", "at_unix": time.time()})


def finish_on_node3(control: Path, session: Path, arm: str, training: dict) -> None:
    from .alfworld_ma_mechanism_campaign import start_remote, stop_remote
    from .launch import Launcher

    receipt = session / f"{arm}_TRAINING.json"
    _immutable_json(receipt, training)
    launch = Launcher(session / f"{arm}-evaluation-supervision")
    remote = None
    try:
        remote = start_remote(
            launch, Path(training["attempt"]), f"{arm}-evaluation",
            "sao_bridge.alfworld_ma_idle_resume",
            ["--control", str(control), "--evaluate-training", str(receipt)],
        )
        launch.wait(remote["process"], "unchanged eight-replica endpoint evaluation", timeout=8 * 3600)
    finally:
        try:
            if remote is not None:
                stop_remote(remote)
        finally:
            launch.close()


def evaluate_training(control: Path, receipt: Path) -> None:
    from . import alfworld_ma_rolling_recovery as rolling
    from .resources import SCOPE_ENV

    if os.uname().nodename != HOSTS["node3"] or (os.getuid(), os.getgid()) != (1009, 1008):
        raise RuntimeError("Serial endpoint evaluation belongs to its declared node3 UID/GID")
    policy = verify_prepared(control)
    root = Path(policy["run_root"])
    if (
        policy["schema"] != SERIAL_SCHEMA or receipt.is_symlink()
        or receipt.parent.parent != control / "sessions"
        or os.environ.get(SCOPE_ENV[0]) != str(control / "AUTHORIZATION.json")
        or os.environ.get(SCOPE_ENV[1]) != file_sha(control / "AUTHORIZATION.json")
    ):
        raise ValueError("Endpoint evaluation is outside its admitted serial session")
    training = read_json(receipt)
    attempt = Path(training["attempt"])
    prepared = read_json(attempt / "PREPARED.json")
    complete = read_json(attempt / "COMPLETE.json")
    arm = prepared["arm"]
    binding = read_json(receipt.parent / "BINDING.json")
    if (
        arm not in ARMS or attempt.parent != root / arm / "attempts"
        or prepared.get("execution_authority") != binding
        or training["status"] != "completed" or complete["status"] != "completed"
        or training["audit"] != complete["audit"]
    ):
        raise ValueError("Endpoint evaluation lacks the exact completed native training audit")
    execution_deadline(policy["original_deadline_unix"], binding, root=root, arm=arm)
    rolling.finish_arm(root, arm, training)


def session_attempt(root: Path, previous: str | None, session: Path) -> Path | None:
    active = root / "ACTIVE_ATTEMPT.json"
    if not active.exists() or read_json(active)["path"] == previous:
        return None
    current = read_json(active)
    path = Path(current["path"])
    if (
        current["arm"] not in ARMS or path.is_symlink()
        or path.parent != root / current["arm"] / "attempts"
        or read_json(path / "PREPARED.json").get("execution_authority") != read_json(session / "BINDING.json")
    ):
        raise ValueError("Active attempt does not belong to this admitted continuation")
    return path


def retry_evidence(root: Path, previous: str | None, returncode: int, session: Path) -> dict | None:
    path = session_attempt(root, previous, session)
    if path is not None:
        arm = path.parent.parent.name
        if (root / arm / "TRAINING_AUDIT.json").exists():
            return None
        failure = path / "FAILED.json"
        if failure.exists():
            value = read_json(failure)
            if value["retry_after_idle_allowed"] and value["evidence"]:
                return {"failure": str(failure), "sha256": file_sha(failure), "evidence": value["evidence"]}
            return None
        if returncode in (-signal.SIGTERM, -signal.SIGKILL, 137, 143):
            return {"kind": "owned_worker_terminated", "returncode": returncode, "attempt": str(path)}
    failure = session / "FAILED.json"
    if failure.exists() and read_json(failure)["error_type"] == "ResourcesChanged":
        return {"kind": "fresh_resource_preflight_changed", "failure": str(failure)}
    return None


def run_session(control: Path, binding: dict) -> bool:
    from .alfworld_ma_mechanism_campaign import stop_remote
    from .launch import Launcher

    policy = verify_authority(Path(binding["authority"]), binding["authority_sha256"])
    root = Path(policy["run_root"])
    session = control / "sessions" / Path(binding["admission"]).stem
    _immutable_json(session / "BINDING.json", binding)
    active = root / "ACTIVE_ATTEMPT.json"
    previous = read_json(active)["path"] if active.exists() else None
    launch = Launcher(session / "supervision")
    process = None
    contention = None
    try:
        process = launch.module(
            "sao_bridge.alfworld_ma_idle_resume", ["--control", control, "--worker", session],
            gpus="", name="continuation",
        )
        while process.poll() is None:
            verify_authority(Path(binding["authority"]), binding["authority_sha256"])
            launch.check()
            worker = session / "WORKER_STATUS.json"
            record_status(control, {
                "status": "training_or_evaluating", "worker_pid": process.pid, "session": str(session),
                "worker": read_json(worker) if worker.exists() else {"phase": "starting"},
                "resource_reservation_claimed": False,
            })
            time.sleep(5)
    except RuntimeError:
        evidence = session / "supervision/foreign_workload.json"
        if not evidence.exists():
            raise
        contention = {"kind": "supervisor_yielded_to_foreign_work", "path": str(evidence), "sha256": file_sha(evidence)}
    finally:
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    print("Owned continuation did not stop within120s; cleaning its tracked descendants", flush=True)
            current = session_attempt(root, previous, session)
            if current is not None:
                for receipt in (current / "launches").glob("*-editor.json"):
                    stop_remote(read_json(receipt))
        finally:
            launch.close()
    if process is None:
        raise RuntimeError("Continuation process was not started")
    if process.returncode == 0 and (session / "COMPLETE.json").exists():
        return True
    if policy["schema"] == SERIAL_SCHEMA:
        raise RuntimeError(f"Serial continuation stopped; no automatic retry is authorized: {session}")
    evidence = retry_evidence(root, previous, process.returncode, session)
    if evidence is None and contention is not None:
        current = session_attempt(root, previous, session)
        if current is None or (
            not (root / current.parent.parent.name / "TRAINING_AUDIT.json").exists()
            and not (current / "FAILED.json").exists()
        ):
            evidence = contention
    if evidence is None:
        raise RuntimeError(f"Continuation failed without automatic-retry authority; inspect {session}")
    _immutable_json(session / "RETRY_AFTER_NEW_IDLE_WINDOW.json", {
        "evidence": evidence, "minimum_new_idle_seconds": IDLE_SECONDS, "at_unix": time.time(),
    })
    print(f"INTERRUPTED: retained {session}; starting a new two-hour observation window", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path)
    parser.add_argument("--probe-local", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--evaluate-training", type=Path)
    args = parser.parse_args()
    if args.probe_local:
        print(json.dumps(probe_local()))
        return
    if args.control is None or not args.control.is_absolute():
        parser.error("--control must name the absolute, prepared controller directory")
    if args.evaluate_training is not None:
        if args.worker is not None or args.check_only:
            parser.error("Endpoint evaluation cannot also be a worker or validation command")
        os.umask(0o002)
        evaluate_training(args.control, args.evaluate_training)
        return
    if os.uname().nodename != HOSTS["node4"] or os.getuid() != 1008:
        raise RuntimeError("Only the node4 research owner may coordinate automatic continuation")
    control = args.control
    os.umask(0o002)
    from .resources import SCOPE_ENV, scope_environment

    digest = file_sha(control / "AUTHORIZATION.json")
    authority = verify_authority(control / "AUTHORIZATION.json", digest)
    if authority["schema"] == SERIAL_SCHEMA:
        os.environ.update({
            SCOPE_ENV[0]: str(control / "AUTHORIZATION.json"), SCOPE_ENV[1]: digest,
        })
    elif scope_environment():
        raise ValueError("A historical idle controller cannot inherit a different GPU partition")

    def interrupted(signum, frame):
        raise InterruptedError(f"Owned controller received signal{signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if args.worker is not None:
        if args.worker.parent != control / "sessions":
            raise ValueError("Worker session escaped its controller directory")
        try:
            execute_worker(control, args.worker)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError, ExceptionGroup) as exc:
            write_json(args.worker / "FAILED.json", {
                "error_type": type(exc).__name__, "error": repr(exc),
                "traceback": traceback.format_exc(), "at_unix": time.time(),
            })
            raise
        return
    policy = verify_prepared(control)
    pending = pending_arms(Path(policy["run_root"]))
    if args.check_only:
        print(json.dumps({"status": "passed", "pending_arms": pending, "gpu_work_started": False}))
        return
    digest = file_sha(control / "AUTHORIZATION.json")
    with (Path(policy["run_root"]) / "TWO_HOUR_CONTROL.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_json(control / "STARTED.json", {
            "pid": os.getpid(), "attached": True,
            "fresh_observation_window": policy["schema"] != SERIAL_SCHEMA, "at_unix": time.time(),
        })
        try:
            while pending_arms(Path(policy["run_root"])):
                binding = wait_for_window(control, digest)
                if run_session(control, binding):
                    if pending_arms(Path(policy["run_root"])):
                        raise ValueError("Worker exited successfully without all four audited ablations")
                    break
        except (InterruptedError, KeyboardInterrupt) as exc:
            record_status(control, {"status": "stopped", "error": repr(exc)}, waiting=True)
            raise
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError, ExceptionGroup) as exc:
            write_json(control / "FAILED.json", {
                "error": repr(exc), "traceback": traceback.format_exc(), "at_unix": time.time(),
            })
            record_status(control, {"status": "blocked_error", "error": repr(exc)}, waiting=True)
            raise
        record_status(control, {"status": "completed", "remaining_arms": []}, waiting=True)


if __name__ == "__main__":
    main()
