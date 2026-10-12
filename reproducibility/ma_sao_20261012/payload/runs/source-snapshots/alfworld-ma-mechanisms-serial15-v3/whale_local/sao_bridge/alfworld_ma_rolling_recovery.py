"""Twelve-round recovery, transparent interrupted attempts, and bounded resubmission."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from . import alfworld_ma_mechanism_campaign as campaign
from . import alfworld_ma_mechanism_program as program
from . import alfworld_ma_mechanism_study as study
from .alfworld_failure_cases import write_case_study
from .alfworld_train_data import _immutable_json
from .alfworld_ma_idle_resume import execution_deadline
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .joint_anchor import build_rows
from .joint_health import metric_rows
from .joint_recovery import finalize_committed_role, latest_checkpoint
from .launch import Launcher
from .resources import gpu_release_audit
from .training_speed_campaign import PYTHON, SSH


def relocate(value, old: Path, new: Path):
    if isinstance(value, str):
        return str(new) + value[len(str(old)):] if value == str(old) or value.startswith(str(old) + "/") else value
    if isinstance(value, list):
        return [relocate(child, old, new) for child in value]
    if isinstance(value, dict):
        return {relocate(key, old, new): relocate(child, old, new) for key, child in value.items()}
    return value


def prepare(root: Path, original: Path, *, validation: Path) -> dict:
    if ROOT.parent.name != "source-snapshots" or (root / "PROTOCOL.json").exists():
        raise ValueError("Prepare once from a new immutable operational source snapshot")
    study.verify_inputs(original)
    tests = read_json(validation)
    if (
        tests["status"] != "passed"
        or tests["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or tests["test_log_sha256"] != file_sha(Path(tests["test_log"]))
    ):
        raise ValueError("The exact operational snapshot has not passed its CPU validation")
    old_protocol = read_json(original / "PROTOCOL.json")
    if old_protocol["study_revision"] != 2 or old_protocol["training_seed"] != 1000043:
        raise ValueError("Rolling recovery requires the frozen v2 single-seed study")
    authority_path = original / "resume_12round/AUTHORIZATION.json"
    authority = read_json(authority_path)
    if authority["checkpoint_interval_rounds"] != 12 or not authority.get("immediate_initial_launch_authorized_at_utc"):
        raise ValueError("Missing explicit rolling-recovery and initial-launch authority")
    deadline = read_json(original / "PROGRAM_STARTED.json")["deadline_unix"]
    if time.time() >= deadline:
        raise TimeoutError("The original study deadline has already expired")
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o2775)
    protocol = relocate(old_protocol, original, root)
    protocol.update(
        source=str(ROOT), unit_suffix="single-seed-v2-rolling12",
        execution_revision="paired-native-recovery-v1", original_absolute_deadline_unix=deadline,
        recovery_interval_rounds=12, recovery_idle_seconds=1800,
        recovery_sampling="Native model/Adam/ValueNorm/RNG/cursor continuation, but fresh serving after "
                          "interruption; uncommitted speculative work is retained as wasted cost, not replayed",
    )
    _immutable_json(root / "PROTOCOL.json", protocol)
    for name in ("INITIAL_STATE.json", "joint_data.json", "schedule_source_data.json"):
        _immutable_json(root / name, relocate(read_json(original / name), original, root))
    spec_paths = []
    for arm in study.ARMS:
        for role in ("editor", "solver"):
            spec = relocate(read_json(original / arm / f"{role}_spec.json"), original, root)
            spec["experiment"] = f"{root.name}-{arm}"
            spec["training_deadline_unix"] = deadline
            spec["recovery_start_round"] = 0
            spec["rolling_recovery"] = {
                "root": str(root / arm / "recovery"), "interval_rounds": 12,
                "attempt": "attempt-0001", "resume": None, "hf_rounds": [96, 144],
            }
            spec["pinned_files"] = {
                path: file_sha(Path(path)) for path in spec["pinned_files"]
            }
            old_spec = read_json(original / arm / f"{role}_spec.json")
            ignored = {
                "run", "joint_root", "experiment", "data", "schedule_source_data", "pinned_files",
                "training_deadline_unix", "rolling_recovery", "recovery_start_round",
            }
            if {key: value for key, value in spec.items() if key not in ignored} != {
                key: value for key, value in old_spec.items() if key not in ignored
            }:
                raise ValueError("Operational recovery changed a scientific setting")
            study.validate_spec(spec)
            path = root / arm / f"{role}_spec.json"
            _immutable_json(path, spec)
            spec_paths.append(path)
    old_pins = read_json(original / "PINS.json")["pins"]
    old_source = Path(old_protocol["source"])
    pins = {}
    for name in old_pins:
        path = Path(name)
        if path.is_relative_to(original):
            path = root / path.relative_to(original)
        elif path.is_relative_to(old_source / "whale_local"):
            path = ROOT / path.relative_to(old_source)
        pins[str(path)] = file_sha(path)
    for path in (ROOT / "whale_local").rglob("*.py"):
        pins[str(path)] = file_sha(path)
    pins[str(authority_path)] = file_sha(authority_path)
    _immutable_json(root / "PINS.json", {"pins": pins, "at_unix": time.time()})
    for name in ("MEMORY_GATE_INHERITANCE.json", "PRIOR_RESULTS_REPORTED.json", "SPEED_POLICY.json"):
        _immutable_json(root / name, read_json(original / name))
    gate = read_json(original / "NATIVE_GATE.json")
    gate.update(
        pins_sha256=file_sha(root / "PINS.json"),
        memory_gate_inheritance_sha256=file_sha(root / "MEMORY_GATE_INHERITANCE.json"),
        inherited_from=str(original / "NATIVE_GATE.json"),
        inherited_sha256=file_sha(original / "NATIVE_GATE.json"),
        additional_rolling_restore_gate_required=True,
    )
    _immutable_json(root / "NATIVE_GATE.json", gate)
    _immutable_json(root / "CODE_VALIDATION.json", {
        **tests, "pins_sha256": file_sha(root / "PINS.json"),
        "original_scientific_validation": str(original / "CODE_VALIDATION.json"),
        "original_scientific_validation_sha256": file_sha(original / "CODE_VALIDATION.json"),
    })
    prior = {
        "original_v2": str(original), "v2_attempt02": str(original.parent / (original.name + "_attempt02")),
        "new_native_calibration_weights_are_not_formal_initialization": True,
        "interrupted_partial_work_is_not_free_or_continued_without_state": True,
        "evidence": {},
    }
    for directory in (original, original.parent / (original.name + "_attempt02")):
        for name in ("full/INTERRUPTED.json", "full/INTERRUPTED.md"):
            path = directory / name
            prior["evidence"][str(path)] = file_sha(path)
    _immutable_json(root / "PRIOR_WORK.json", prior)
    program.audit_historical_path(root)
    _immutable_json(root / "PROGRAM_PREPARED.json", {
        "status": "prepared", "original_deadline_unix": deadline,
        "authority": str(authority_path), "authority_sha256": file_sha(authority_path),
        "pins_sha256": file_sha(root / "PINS.json"), "at_unix": time.time(),
    })
    prepare_canary(root)
    (root / "PROTOCOL.md").write_text(
        "# MA-SAO: twelve-round paired recovery\n\n"
        "Scientific v2, seed1000043, H0/W1 with raw pre-MA Critics, fixed144 rounds, original200.\n"
        "Full control first; four isolated ablations only after its audited, reported release.\n\n"
        "Every12 committed rounds save both Actors, both Critics, native Adam, ValueNorm,\n"
        "worker RNG/LR state and committed input/metric/schedule cursors. Healthy learners stay resident.\n"
        "Only a validated same-round pair advances LATEST; then the preceding owned native payload\n"
        "is deleted. HF evaluation milestones96/144 and each arm's final native state remain.\n"
        "HF Critic exports are retained alongside Actors for existing audit compatibility.\n\n"
        "An interrupted attempt is immutable. Resume in a fresh owned attempt directory;\n"
        "reuse only the committed prefix and regenerate discarded in-flight work. Fresh inference\n"
        "serving means this is not bitwise rollout replay. All discarded work and failed attempts\n"
        "remain in the execution ledger. SIGTERM/SIGKILL or recorded resource contention may requeue\n"
        "only after30 continuous empty minutes on both nodes. Unexpected code/data errors stop explicitly.\n"
        "The watcher cannot guarantee recovery if the watcher itself is uncatchably killed.\n\n"
        "Native validation: full schedule, a planned stop after the first12-round pair, then resume\n"
        "to15 with the same initialization/seed and unchanged accumulated budgets. No canary weights\n"
        "enter formal training. Original wall deadline is not extended.\n"
    )
    return {"status": "prepared", "root": str(root), "deadline_unix": deadline}


def prepare_canary(root: Path) -> None:
    directory = root / "rolling-canary"
    source = read_json(root / "schedule_source_data.json")[:80]
    rows = build_rows(source, cycles=5, batch_size=8)
    if rows != read_json(root / "joint_data.json")[:120]:
        raise ValueError("Recovery canary changed the original ordered collection inputs")
    _immutable_json(directory / "source_data.json", source)
    _immutable_json(directory / "data.json", rows)
    protocol = read_json(root / "PROTOCOL.json")
    _immutable_json(directory / "PROTOCOL.json", {
        **protocol, "root": str(directory), "joint_rounds": 15, "arms": ["full"],
        "unit_suffix": "rolling-canary-v1", "curve_rounds": [],
        "purpose": "native_restore_validation_not_formal_training",
    })
    for role in ("editor", "solver"):
        spec = read_json(root / "full" / f"{role}_spec.json")
        spec.update(
            run=str(directory / "full" / role / "optimizer"),
            joint_root=str(directory / "full/joint"), steps=15, save_interval_steps=15,
            data=str(directory / "data.json"), schedule_source_data=str(directory / "source_data.json"),
            experiment=f"{root.name}-recovery-canary",
            rolling_recovery={
                "root": str(directory / "full/recovery"), "interval_rounds": 12,
                "attempt": "attempt-0001", "resume": None, "hf_rounds": [15],
            },
        )
        spec["pinned_files"].update(study.file_manifest([
            directory / "PROTOCOL.json", directory / "source_data.json", directory / "data.json",
        ]))
        study.validate_spec(spec)
        _immutable_json(directory / "full" / f"{role}_spec.json", spec)


def prepare_attempt(root: Path, arm: str, *, execution_authority: dict | None = None) -> tuple[Path, int]:
    protocol = read_json(root / "PROTOCOL.json")
    if execution_authority is not None:
        execution_deadline(protocol["original_absolute_deadline_unix"], execution_authority, root=root, arm=arm)
        protocol["execution_authority"] = execution_authority
    directory = root / arm
    recovery = latest_checkpoint(directory / "recovery")
    start = read_json(recovery)["completed_rounds"] if recovery else 0
    finalizing = start == read_json(directory / "solver_spec.json")["steps"]
    attempts = directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    indices = [int(path.name.split("-")[-1]) for path in attempts.glob("attempt-[0-9][0-9][0-9][0-9]")]
    number = max(indices, default=0) + 1
    path = attempts / f"attempt-{number:04d}"
    path.mkdir()
    path.chmod(0o2775)
    _immutable_json(path / "PROTOCOL.json", {
        **protocol, "root": str(path), "unit_suffix": f"{protocol['unit_suffix']}-{arm}-{number:04d}",
    })
    for role in ("editor", "solver"):
        spec = read_json(directory / f"{role}_spec.json")
        spec.update(run=str(path / arm / role / "optimizer"), joint_root=str(path / arm / "joint"),
                    recovery_start_round=start)
        spec["rolling_recovery"].update(
            attempt=f"attempt-{number:04d}", resume=str(recovery) if recovery else None,
            finalize_only=finalizing,
        )
        if execution_authority is not None:
            spec["rolling_recovery"]["execution_authority"] = execution_authority
        study.validate_spec(spec)
        _immutable_json(path / arm / f"{role}_spec.json", spec)
    _immutable_json(path / "PREPARED.json", {
        "status": "prepared", "arm": arm, "resume_round": start,
        "recovery": str(recovery) if recovery else None,
        "recovery_sha256": file_sha(recovery) if recovery else None, "at_unix": time.time(),
        **({"execution_authority": execution_authority} if execution_authority is not None else {}),
    })
    return path, start


def attempt_cost(path: Path, arm: str, start: int) -> dict:
    joint = path / arm / "joint"
    return {
        "resume_round": start,
        "new_persisted_outcomes": sum(
            1 for directory in joint.glob("round-*") if not directory.is_symlink()
            for _ in directory.glob("outcome-*.json")
        ),
        "new_generated_role_receipts": {
            role: sum(1 for directory in joint.glob("round-*") if not directory.is_symlink()
                      for _ in directory.glob(f"{role}-receipt-*.json"))
            for role in ("editor", "solver")
        },
        "new_committed_role_rounds": {
            role: max(0, len(metric_rows(path / arm / role / "optimizer/metrics.jsonl")) - start)
            for role in ("editor", "solver")
        },
        "unpersisted_inflight_work_unknown": True,
    }


def termination_evidence(path: Path, arm: str) -> list[dict]:
    evidence = []
    signal_path = path / "SIGNAL.json"
    if signal_path.exists() and read_json(signal_path)["signal"] == signal.SIGTERM:
        evidence.append({"path": str(signal_path), "kind": "controller_sigterm"})
    exits = path / arm / "PROCESS_EXIT_FAILURE.json"
    if exits.exists() and any(
        item["returncode"] in (-signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM, 128 + signal.SIGKILL)
        for item in read_json(exits)["processes"]
    ):
        evidence.append({"path": str(exits), "kind": "role_termination"})
    for role in ("editor", "solver"):
        native = path / arm / role / "NATIVE_EXIT.json"
        if native.exists() and read_json(native)["returncode"] in (-signal.SIGTERM, -signal.SIGKILL):
            evidence.append({"path": str(native), "kind": "native_termination"})
        failure = path / arm / role / "FAILED.json"
        if failure.exists():
            value = read_json(failure)
            if value.get("foreign_workload") is not None:
                evidence.append({"path": str(failure), "kind": "role_yielded_to_foreign_workload"})
            elif value.get("native_returncode") in (-signal.SIGTERM, -signal.SIGKILL):
                evidence.append({"path": str(failure), "kind": "nested_native_termination"})
    foreign = path / arm / "coordinator-supervision/foreign_workload.json"
    if foreign.exists():
        evidence.append({"path": str(foreign), "kind": "yielded_to_foreign_workload"})
    return evidence


def run_attempt(root: Path, arm: str, *, planned_stop: int | None = None,
                execution_authority: dict | None = None) -> dict:
    os.umask(0o002)
    path, start = prepare_attempt(root, arm, execution_authority=execution_authority)
    output = path / arm
    _immutable_json(path / "STARTED.json", {"pid": os.getpid(), "at_unix": time.time()})
    write_json(root / "ACTIVE_ATTEMPT.json", {"arm": arm, "path": str(path), "resume_round": start})
    if read_json(output / "solver_spec.json")["rolling_recovery"]["finalize_only"]:
        resources = admission_snapshot()
        if not resources["released"]:
            raise RuntimeError("Resources changed before publishing a final saved state")
        started = min(read_json(record)["at_unix"] for record in
                      (root / arm / "attempts").glob("attempt-*/STARTED.json"))
        for role, release in (("editor", resources["node3"]["release"]), ("solver", resources["node4"])):
            finalize_committed_role(read_json(output / f"{role}_spec.json"), release,
                                    elapsed_seconds=time.time() - started)
        audit = study.audit_joint_arm(path, arm, rounds=start)
        write_json(path / "COST.json", attempt_cost(path, arm, start))
        _immutable_json(path / "COMPLETE.json", {
            "status": "completed", "audit": audit, "native_training_launched": False, "at_unix": time.time(),
        })
        return {"status": "completed", "attempt": str(path), "audit": audit}
    launch = Launcher(output / "coordinator-supervision")
    remote = None
    prior_handler = signal.getsignal(signal.SIGTERM)

    def terminated(signum, frame):
        write_json(path / "SIGNAL.json", {"signal": signum, "at_unix": time.time()})
        raise InterruptedError("Training controller received SIGTERM; retain attempt and wait before recovery")

    signal.signal(signal.SIGTERM, terminated)
    planned = False
    try:
        remote = campaign.start_remote(
            launch, path, f"{arm}-editor", "sao_bridge.alfworld_ma_mechanism_study",
            ["train-role", "--root", str(path), "--arm", arm, "--role", "editor"],
        )
        local = launch.module(
            "sao_bridge.alfworld_ma_mechanism_study",
            ["train-role", "--root", path, "--arm", arm, "--role", "solver"],
            gpus="", name=f"{arm}-solver",
        )
        print(f"START {arm} {path.name}: same seed; committed cursor={start}; checkpoint every12",
              flush=True)
        while True:
            launch.check()
            states = [{"pid": proc.pid, "returncode": proc.poll()} for proc in (local, remote["process"])]
            if any(item["returncode"] not in (None, 0) for item in states):
                write_json(output / "PROCESS_EXIT_FAILURE.json", {"processes": states, "at_unix": time.time()})
                raise RuntimeError(f"Native role process exited unsuccessfully: {states}")
            progress = {
                role: len(metric_rows(output / role / "optimizer/metrics.jsonl"))
                for role in ("editor", "solver")
            }
            write_json(root / "PROGRAM_MONITOR.json", {
                "status": "training", "arm": arm, "attempt": str(path), "progress": progress,
                "resume_round": start, "at_unix": time.time(),
            })
            if planned_stop is not None and start < planned_stop and all(
                (output / role / "optimizer/RECOVERY_SAVED.json").exists()
                and read_json(output / role / "optimizer/RECOVERY_SAVED.json")["completed_rounds"] >= planned_stop
                for role in ("editor", "solver")
            ):
                checkpoint = latest_checkpoint(root / arm / "recovery")
                if checkpoint is None or read_json(checkpoint)["completed_rounds"] != planned_stop:
                    raise ValueError("Planned interruption has no exact paired recovery point")
                _immutable_json(path / "PLANNED_INTERRUPT.json", {
                    "completed_rounds": planned_stop, "owned_processes": states,
                    "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha(checkpoint),
                    "at_unix": time.time(),
                })
                planned = True
                break
            if all(item["returncode"] == 0 for item in states):
                break
            if time.time() >= execution_deadline(
                read_json(root / "PROTOCOL.json")["original_absolute_deadline_unix"],
                execution_authority, root=root, arm=arm,
            ):
                raise TimeoutError("The unchanged study deadline expired")
            time.sleep(2)
    except (OSError, ValueError, RuntimeError) as exc:
        evidence = termination_evidence(path, arm)
        failure = {
            "status": "interrupted" if evidence else "blocked_error", "error": repr(exc),
            "retry_after_idle_allowed": bool(evidence), "evidence": evidence, "at_unix": time.time(),
        }
        _immutable_json(path / "FAILED.json", failure)
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_handler)
        try:
            if remote is not None:
                campaign.stop_remote(remote)
        finally:
            launch.close()
            write_json(path / "COST.json", attempt_cost(path, arm, start))
            write_json(path / "RELEASED.json", gpu_release_audit())
    if planned:
        return {"status": "planned_interruption", "attempt": str(path), "resume_round": planned_stop}
    rounds = read_json(output / "solver_spec.json")["steps"]
    audit = study.audit_joint_arm(path, arm, rounds=rounds)
    _immutable_json(path / "COMPLETE.json", {"status": "completed", "audit": audit, "at_unix": time.time()})
    return {"status": "completed", "attempt": str(path), "audit": audit}


def distributed_launchers(*, strict: bool = False) -> list[dict]:
    import psutil

    modules = {"torch.distributed.run", "torch.distributed.launch",
               "vllm.entrypoints.openai.api_server", "sao_bridge.large_train"}
    result = []
    for process in psutil.process_iter(["pid", "cmdline", "uids"]):
        try:
            args = (process.cmdline() if strict else process.info["cmdline"]) or []
            uid = process.uids().real if strict else None
            if args and Path(args[0]).name in {"torchrun", "deepspeed", "mpirun"}:
                result.append({"pid": process.pid, "uid": uid if strict else process.info["uids"].real,
                               "module": Path(args[0]).name})
            elif "-m" in args and args.index("-m") + 1 < len(args):
                module = args[args.index("-m") + 1]
                if module in modules:
                    result.append({"pid": process.pid, "uid": uid if strict else process.info["uids"].real,
                                   "module": module})
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            if strict:
                raise
    return result


def admission_snapshot() -> dict:
    result = program.resource_snapshot()
    code = (
        "import json; from sao_bridge.alfworld_ma_rolling_recovery import distributed_launchers; "
        "print(json.dumps(distributed_launchers()))"
    )
    command = [
        "env", "CUDA_VISIBLE_DEVICES=", "TMPDIR=/dev/shm", "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={ROOT}/whale_local:{ROOT}/third_party/sao/vendor/AReaL",
        PYTHON, "-B", "-c", code,
    ]
    remote = subprocess.run(
        [*SSH[:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH[-1], shlex.join(command)],
        check=True, capture_output=True, text=True, timeout=60,
    )
    launchers = {"node4": distributed_launchers(), "node3": json.loads(remote.stdout)}
    result.update(distributed_launchers=launchers,
                  released=result["released"] and not any(launchers.values()))
    return result


def wait_idle(root: Path, *, retry: bool) -> None:
    deadline = read_json(root / "PROTOCOL.json")["original_absolute_deadline_unix"]
    program.wait_for_resources(
        root, "retry-idle30" if retry else "initial-admission",
        deadline=time.monotonic() + deadline - time.time(),
        interval=60 if retry else 10, stable_checks=31 if retry else 1,
        probe=admission_snapshot,
    )


def resilient_training(root: Path, arm: str, *, planned_stop: int | None = None) -> dict:
    retry = False
    while True:
        wait_idle(root, retry=retry)
        active_path = root / "ACTIVE_ATTEMPT.json"
        previous = read_json(active_path)["path"] if active_path.exists() else None
        try:
            return run_attempt(root, arm, planned_stop=planned_stop)
        except (OSError, ValueError, RuntimeError):
            if not active_path.exists():
                raise
            active = read_json(active_path)
            if active["path"] == previous:
                raise
            failure_path = Path(active["path"]) / "FAILED.json"
            if not failure_path.exists() or not read_json(failure_path)["retry_after_idle_allowed"]:
                raise
            failure = read_json(failure_path)
            write_json(root / "PROGRAM_MONITOR.json", {
                **failure, "status": "waiting_after_termination", "arm": arm,
                "minimum_continuous_idle_seconds": 1800,
            })
            print(f"INTERRUPTED {arm}: {failure['error']}; waiting30 continuous idle minutes", flush=True)
            retry = True


def run_canary(root: Path) -> dict:
    wait_idle(root, retry=False)
    campaign.require_prerequisites(root)
    directory = root / "rolling-canary"
    if (root / "ROLLING_NATIVE_GATE.json").exists():
        raise ValueError("The already-recorded native recovery gate cannot be replayed")
    stopped = resilient_training(directory, "full", planned_stop=12)
    if stopped["status"] != "planned_interruption":
        raise ValueError("Native validation did not exercise a real interrupted checkpoint")
    resumed = resilient_training(directory, "full")
    final_attempt = Path(resumed["attempt"])
    checks = {}
    for role in ("editor", "solver"):
        run = final_attempt / "full" / role / "optimizer"
        loaded = read_json(run / "RECOVERY_LOADED.json")
        if loaded["completed_rounds"] != 12 or loaded["adam_reset"] or loaded["warmup_repeated"]:
            raise ValueError("Native recovery reset Adam, warmup or its cursor")
        metrics = metric_rows(run / "metrics.jsonl")
        if [row["global_step"] for row in metrics] != list(range(15)):
            raise ValueError("Restored native metrics repeat or skip a collection round")
        checks[role] = {
            "loaded": loaded, "loaded_sha256": file_sha(run / "RECOVERY_LOADED.json"),
            "metrics_sha256": file_sha(run / "metrics.jsonl"),
            "complete": str(run.parent / "COMPLETE.json"),
            "complete_sha256": file_sha(run.parent / "COMPLETE.json"),
        }
    result = {
        "status": "passed", "stopped": stopped, "resumed": resumed, "roles": checks,
        "source_manifest_sha256": file_sha(ROOT / "SOURCE_MANIFEST.json"),
        "pins_sha256": file_sha(root / "PINS.json"), "formal_weights_reused": False,
        "logical_shared_episodes": 120, "extra_speculative_work": "See every attempt COST.json",
        "at_unix": time.time(),
    }
    _immutable_json(root / "ROLLING_NATIVE_GATE.json", result)
    (directory / "RESULT.md").write_text(
        "# Native twelve-round recovery validation\n\n"
        "Passed: native paired save at12, explicit owned-process interruption, full Adam/ValueNorm/"
        "RNG/cursor restore, then the original schedule through15.\n\n"
        "120 logical shared episodes; speculative/interrupted work is additional and retained in "
        "attempt COST.json records. These are validation weights, not formal initial weights.\n"
        f"\nFinal attempt: `{final_attempt}`.\n"
    )
    return result


def finish_arm(root: Path, arm: str, training: dict) -> dict:
    directory = root / arm
    attempt = Path(training["attempt"])
    for name in ("editor", "solver", "joint"):
        destination = directory / name
        if destination.exists() or destination.is_symlink():
            raise ValueError("Formal arm finalization cannot replace existing output")
        destination.symlink_to(attempt / arm / name, target_is_directory=True)
    _immutable_json(directory / "TRAINING_AUDIT.json", training["audit"])
    records = {role: read_json(directory / role / "COMPLETE.json") for role in ("editor", "solver")}
    incumbent = read_json(root / "PROTOCOL.json")["incumbent_harness"]
    endpoints = {}
    for rounds in study.ROUNDS:
        pair = {role: records[role]["checkpoint_history"][str(rounds - 1)] for role in records}
        endpoints[str(rounds)] = study.evaluate_pair(root, arm, rounds, pair, incumbent=incumbent)
        incumbent = endpoints[str(rounds)]["selection"]["harness"]
    return publish_arm_result(root, arm, training, endpoints)


def publish_arm_result(root: Path, arm: str, training: dict, endpoints: dict, *,
                       evaluation_recovery: dict | None = None) -> dict:
    if set(endpoints) != {str(rounds) for rounds in study.ROUNDS} or any(
        value["status"] != "completed" or value["arm"] != arm or value["rounds"] != int(rounds)
        for rounds, value in endpoints.items()
    ):
        raise ValueError("Publish only the two complete declared endpoint evaluations")
    directory = root / arm
    protocol = read_json(root / "PROTOCOL.json")
    result = {
        "status": "verified", "arm": arm, "training": training["audit"], "endpoints": endpoints,
        "training_seeds": protocol["training_seeds"], "primary_endpoint": 144,
        "at_unix": time.time(), "elapsed_seconds": time.time() - read_json(directory / "STARTED.json")["at_unix"],
        "interruption_costs": {
            str(path): read_json(path) for path in sorted((directory / "attempts").glob("attempt-*/COST.json"))
        },
        "recovery_semantics": protocol["recovery_sampling"],
    }
    if evaluation_recovery is not None:
        result["evaluation_recovery"] = evaluation_recovery
    if arm != "full":
        full = read_json(root / "full/RESULT.json")["endpoints"]["144"]["scores"]["selected"]
        control = endpoints["144"]["scores"]["selected"]
        result["paired_against_full"] = paired_difference(
            read_json(Path(full["report"])), read_json(Path(control["report"])), seed=20261008,
        )
        result["failure_case_population"] = write_case_study(
            directory / "failure-cases", Path(control["report"]), Path(full["report"]),
            read_json(Path(protocol["core200_protocol"])), title=f"{arm} failures and matched full-MA outcomes",
        )
    _immutable_json(directory / "RESULT.json", result)
    campaign.write_arm_report(root, arm, result)
    with (directory / "RESULT.md").open("a") as stream:
        stream.write("\n\nTwelve-round paired native recovery was enabled. All interrupted attempts, "
                     "speculative work and retained/deleted-generation receipts are preserved under "
                     "`attempts/` and `recovery/`; inference restarts are not bitwise rollout replay.\n")
        if evaluation_recovery is not None:
            stream.write(
                "\nEvaluation was interrupted and resumed without repeating completed episodes or "
                "generating replacement Harness candidates. Input-bound recovery evidence and "
                "additional incomplete episode costs are retained in RESULT.json/evaluation_recovery.\n"
            )
    _immutable_json(directory / "FINAL_AUDIT.json", {
        "status": "passed", "result_sha256": file_sha(directory / "RESULT.json"),
        "report_sha256": file_sha(directory / "RESULT.md"), "at_unix": time.time(),
    })
    return result


def run_formal(root: Path, arm: str) -> dict:
    wait_idle(root, retry=False)
    campaign.require_prerequisites(root)
    gate = read_json(root / "ROLLING_NATIVE_GATE.json")
    if (
        gate["status"] != "passed" or gate["formal_weights_reused"]
        or gate["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or gate["pins_sha256"] != file_sha(root / "PINS.json")
    ):
        raise ValueError("Formal training requires its exact source-bound native recovery validation")
    if arm != "full":
        campaign.require_full_control_review(root)
    _immutable_json(root / arm / "STARTED.json", {"pid": os.getpid(), "at_unix": time.time()})
    training = resilient_training(root, arm)
    result = finish_arm(root, arm, training)
    if arm == "full":
        program.audit_full_result(root)
    program.write_master_report(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "canary", "run-arm"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--arm", choices=study.ARMS, default="full")
    args = parser.parse_args()
    os.umask(0o002)
    root = args.root.resolve()
    if args.action == "prepare":
        if args.original is None or args.validation is None:
            parser.error("prepare requires --original and --validation")
        result = prepare(root, args.original.resolve(), validation=args.validation.resolve())
    else:
        stage = f"{args.action}-{args.arm}"
        _immutable_json(root / f"{stage}_CONTROL_STARTED.json", {
            "pid": os.getpid(), "attached": True, "at_unix": time.time(),
            "native_validation": args.action == "canary",
        })
        remaining = read_json(root / "PROTOCOL.json")["original_absolute_deadline_unix"] - time.time()
        if remaining <= 0:
            raise TimeoutError("The original study deadline expired before execution")

        def expired(signum, frame):
            raise TimeoutError("The unchanged absolute study deadline expired")

        prior_alarm = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            result = run_canary(root) if args.action == "canary" else run_formal(root, args.arm)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ExceptionGroup) as exc:
            write_json(root / f"{stage}_CONTROL_FAILED.json", {
                "status": "incomplete", "error": repr(exc), "at_unix": time.time(),
            })
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior_alarm)
        _immutable_json(root / f"{stage}_CONTROL_COMPLETE.json", {
            "status": result["status"], "at_unix": time.time(),
        })
    print(json.dumps({"status": result["status"], "action": args.action, "root": str(root)}), flush=True)


if __name__ == "__main__":
    main()
