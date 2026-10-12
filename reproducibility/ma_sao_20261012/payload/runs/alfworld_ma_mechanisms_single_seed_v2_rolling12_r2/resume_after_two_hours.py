"""User-authorized two-hour cooldown around the unchanged native recovery runtime."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import signal
import time

from sao_bridge import alfworld_ma_mechanism_campaign as campaign
from sao_bridge import alfworld_ma_mechanism_program as program
from sao_bridge import alfworld_ma_mechanism_study as study
from sao_bridge import alfworld_ma_rolling_recovery as rolling
from sao_bridge.alfworld_train_data import _immutable_json
from sao_bridge.common import ROOT, file_sha, read_json, write_json
from sao_bridge.joint_health import metric_rows
from sao_bridge.joint_recovery import latest_checkpoint


SCRIPT = Path(__file__).resolve()
RUN = SCRIPT.parent
COOLDOWN = 7200


def verify_authority(root: Path, digest: str) -> dict:
    path = root / "USER_PAUSE_AND_DELAY.json"
    if file_sha(path) != digest:
        raise ValueError("The delayed controller's user authority changed; do not resume")
    if (root / "TWO_HOUR_RESUME_CANCELLED.json").exists():
        raise InterruptedError("The scheduled two-hour resumption was cancelled")
    policy = read_json(path)
    if policy["minimum_resubmission_interval_seconds"] != COOLDOWN:
        raise ValueError("Two-hour resubmission policy changed")
    return policy


def earliest_start(policy: dict, last_interruption: float | None = None) -> float:
    from datetime import datetime

    initial = datetime.fromisoformat(policy["resume_not_before_utc"]).timestamp()
    return max(initial, last_interruption + COOLDOWN) if last_interruption is not None else initial


def wait_until(root: Path, earliest: float, *, deadline: float, authority_sha256: str) -> None:
    while True:
        verify_authority(root, authority_sha256)
        now = time.time()
        if now >= deadline:
            raise TimeoutError("The original study deadline expired during the two-hour pause")
        if now >= earliest:
            return
        write_json(root / "TWO_HOUR_MONITOR.json", {
            "status": "paused_until_not_before", "pid": os.getpid(), "not_before_unix": earliest,
            "remaining_seconds": earliest - now, "at_unix": now, "gpu_work_started": False,
        })
        time.sleep(min(60, earliest - now, deadline - now))


def admit(root: Path, study_root: Path, *, authority_sha256: str, deadline: float) -> None:
    policy = verify_authority(root, authority_sha256)
    retry = root / "TWO_HOUR_RETRY_STATE.json"
    last = read_json(retry)["interrupted_at_unix"] if retry.exists() else None
    earliest = earliest_start(policy, last)
    wait_until(root, earliest, deadline=deadline, authority_sha256=authority_sha256)
    program.wait_for_resources(
        root, "two-hour-resume-admission",
        deadline=time.monotonic() + deadline - time.time(), interval=60, stable_checks=1,
        probe=rolling.admission_snapshot,
    )
    verify_authority(root, authority_sha256)
    if time.time() < earliest or time.time() >= deadline:
        raise TimeoutError("Attempt fell outside its authorized two-hour/deadline window")
    write_json(root / "TWO_HOUR_MONITOR.json", {
        "status": "admitted_after_two_hour_delay", "study_root": str(study_root),
        "not_before_unix": earliest, "at_unix": time.time(), "resource_reservation_claimed": False,
    })


def train(root: Path, study_root: Path, arm: str, *, authority_sha256: str,
          deadline: float, planned_stop: int | None = None) -> dict:
    while True:
        admit(root, study_root, authority_sha256=authority_sha256, deadline=deadline)
        active_path = study_root / "ACTIVE_ATTEMPT.json"
        previous = read_json(active_path)["path"] if active_path.exists() else None
        try:
            return rolling.run_attempt(study_root, arm, planned_stop=planned_stop)
        except (OSError, ValueError, RuntimeError) as exc:
            if not active_path.exists():
                raise
            active = read_json(active_path)
            if active["path"] == previous:
                raise
            failed_path = Path(active["path"]) / "FAILED.json"
            if not failed_path.exists() or not read_json(failed_path)["retry_after_idle_allowed"]:
                raise
            failure = read_json(failed_path)
            interrupted = time.time()
            retry = {
                "interrupted_at_unix": interrupted,
                "next_attempt_not_before_unix": interrupted + COOLDOWN,
                "failed_attempt": active["path"], "failure_sha256": file_sha(failed_path),
                "evidence": failure["evidence"], "cooldown_seconds": COOLDOWN,
            }
            write_json(root / "TWO_HOUR_RETRY_STATE.json", retry)
            write_json(root / "TWO_HOUR_MONITOR.json", {
                **retry, "status": "paused_after_termination", "error": repr(exc),
            })
            (Path(active["path"]) / "INTERRUPTED.md").write_text(
                "# Interrupted native attempt\n\n"
                f"Error: `{exc!r}`.\n\n"
                "Preserved failure evidence and exact persisted costs are in FAILED.json and COST.json.\n"
                "No retry may start before the recorded two-hour cooldown and a fresh empty-node check.\n"
                "This is not a completed training or evaluation result.\n"
            )
            print(f"INTERRUPTED {arm}: next attempt not before {interrupted + COOLDOWN}", flush=True)


def record_canary_gate(root: Path, result: dict) -> dict:
    directory = root / "rolling-canary"
    checkpoint = latest_checkpoint(directory / "full/recovery")
    if checkpoint is None or read_json(checkpoint)["completed_rounds"] != 15:
        raise ValueError("Native validation has no complete fifteen-round paired state")
    if result["audit"]["status"] != "passed" or result["audit"]["shared_environment_episodes"] != 120:
        raise ValueError("Native recovery validation did not pass its exact trajectory/update audit")
    roles = {}
    for role in ("editor", "solver"):
        source = Path(read_json(checkpoint.parent / f"{role}.json")["source_run"])
        loaded_path = source / "RECOVERY_LOADED.json"
        loaded = read_json(loaded_path)
        if loaded["completed_rounds"] != 12 or loaded["adam_reset"] or loaded["warmup_repeated"]:
            raise ValueError("A real native twelve-round restore was not exercised")
        metrics = metric_rows(source / "metrics.jsonl")
        if [row["global_step"] for row in metrics] != list(range(15)):
            raise ValueError("Restored native metrics repeat or omit a collection round")
        complete_path = Path(result["attempt"]) / "full" / role / "COMPLETE.json"
        roles[role] = {
            "loaded": loaded, "loaded_sha256": file_sha(loaded_path),
            "metrics_sha256": file_sha(source / "metrics.jsonl"),
            "complete": str(complete_path), "complete_sha256": file_sha(complete_path),
        }
    if roles["editor"]["loaded"]["checkpoint_sha256"] != roles["solver"]["loaded"]["checkpoint_sha256"]:
        raise ValueError("Native recovery loaded different paired checkpoints")
    gate = {
        "status": "passed", "resumed": result, "roles": roles,
        "source_manifest_sha256": file_sha(ROOT / "SOURCE_MANIFEST.json"),
        "pins_sha256": file_sha(root / "PINS.json"), "formal_weights_reused": False,
        "logical_shared_episodes": 120, "extra_speculative_work": "See every attempt COST.json",
        "operational_driver_sha256": file_sha(SCRIPT),
        "user_pause_authority_sha256": file_sha(root / "USER_PAUSE_AND_DELAY.json"),
        "all_attempt_costs": {
            str(path): read_json(path)
            for path in sorted((directory / "full/attempts").glob("attempt-*/COST.json"))
        },
        "at_unix": time.time(),
    }
    _immutable_json(root / "ROLLING_NATIVE_GATE.json", gate)
    (directory / "RESULT.md").write_text(
        "# Native twelve-round interrupted recovery validation\n\n"
        "Passed: paired native save at12, actual process restart and native Adam/ValueNorm/RNG/cursor "
        "restore, then the unchanged schedule through15. No validation weights enter formal training.\n\n"
        "The first user-paused twelve-round attempt and every subsequent failed/speculative episode "
        "are additional costs, retained in attempt COST.json and the native gate's all_attempt_costs.\n"
        "120 episodes is the completed logical path, not the total physical cost across attempts.\n"
    )
    return gate


def execute(root: Path, action: str, arm: str, *, authority_sha256: str, deadline: float) -> dict:
    admit(root, root, authority_sha256=authority_sha256, deadline=deadline)
    campaign.require_prerequisites(root)
    if action == "canary":
        if (root / "ROLLING_NATIVE_GATE.json").exists():
            raise ValueError("Do not rerun an already-passed recovery gate")
        directory = root / "rolling-canary"
        checkpoint = latest_checkpoint(directory / "full/recovery")
        completed = read_json(checkpoint)["completed_rounds"] if checkpoint else 0
        if completed < 12:
            train(root, directory, "full", authority_sha256=authority_sha256,
                  deadline=deadline, planned_stop=12)
        result = train(root, directory, "full", authority_sha256=authority_sha256, deadline=deadline)
        return record_canary_gate(root, result)
    gate = read_json(root / "ROLLING_NATIVE_GATE.json")
    if (
        gate["status"] != "passed" or gate["formal_weights_reused"]
        or gate["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or gate["pins_sha256"] != file_sha(root / "PINS.json")
    ):
        raise ValueError("Formal training requires the exact passed native recovery gate")
    if arm != "full":
        campaign.require_full_control_review(root)
    if (root / arm / "RESULT.json").exists() or (root / arm / "TRAINING_AUDIT.json").exists():
        raise ValueError("Do not replay completed or partially evaluated formal work")
    started = root / arm / "STARTED.json"
    if not started.exists():
        _immutable_json(started, {"pid": os.getpid(), "at_unix": time.time(), "cooldown_seconds": COOLDOWN})
    result = train(root, root, arm, authority_sha256=authority_sha256, deadline=deadline)
    result = rolling.finish_arm(root, arm, result)
    if arm == "full":
        program.audit_full_result(root)
    program.write_master_report(root)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("canary", "run-arm"))
    parser.add_argument("--arm", choices=study.ARMS, default="full")
    args = parser.parse_args()
    os.umask(0o002)
    validation = read_json(RUN / "TWO_HOUR_CODE_VALIDATION.json")
    if (
        validation["status"] != "passed" or validation["driver_sha256"] != file_sha(SCRIPT)
        or validation["test_sha256"] != file_sha(Path(validation["test_file"]))
        or validation["test_log_sha256"] != file_sha(Path(validation["test_log"]))
        or validation["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
    ):
        raise ValueError("Delayed resumption driver is not bound to passing validation")
    authority_sha256 = file_sha(RUN / "USER_PAUSE_AND_DELAY.json")
    policy = verify_authority(RUN, authority_sha256)
    protocol = read_json(RUN / "PROTOCOL.json")
    if Path(protocol["source"]).resolve() != ROOT.resolve():
        raise ValueError("Delayed resumption cannot change the frozen scientific/runtime source")
    deadline = protocol["original_absolute_deadline_unix"]
    remaining = deadline - time.time()
    if remaining <= 0:
        raise TimeoutError("The original study deadline has expired")
    with (RUN / "TWO_HOUR_CONTROL.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stage = f"two-hour-{args.action}-{args.arm}"
        _immutable_json(RUN / f"{stage}_STARTED.json", {
            "pid": os.getpid(), "attached": True, "at_unix": time.time(),
            "not_before_unix": earliest_start(policy), "deadline_unix": deadline,
            "authority_sha256": authority_sha256, "driver_sha256": file_sha(SCRIPT),
        })

        def expired(signum, frame):
            raise TimeoutError("The original absolute deadline expired")

        prior = signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            result = execute(RUN, args.action, args.arm, authority_sha256=authority_sha256, deadline=deadline)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ExceptionGroup) as exc:
            write_json(RUN / f"{stage}_FAILED.json", {
                "status": "incomplete", "error": repr(exc), "at_unix": time.time(),
            })
            raise
        except KeyboardInterrupt:
            write_json(RUN / f"{stage}_STOPPED.json", {"status": "interrupted", "at_unix": time.time()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior)
        _immutable_json(RUN / f"{stage}_COMPLETE.json", {"status": result["status"], "at_unix": time.time()})


if __name__ == "__main__":
    main()
