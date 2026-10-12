"""Finish the interrupted full144 evaluation without replaying completed work."""
from __future__ import annotations

import fcntl
from datetime import datetime
import os
from pathlib import Path
import signal
import time

from sao_bridge import alfworld_ma_mechanism_program as program
from sao_bridge import alfworld_ma_mechanism_study as study
from sao_bridge import alfworld_ma_rolling_recovery as rolling
from sao_bridge.alfworld_strong_phase import checked_native_report, checked_release
from sao_bridge.alfworld_strong_select import checked_matrix, generate_proposals, matrix_config, matrix_reports, run_matrix
from sao_bridge.alfworld_train_data import _immutable_json
from sao_bridge.common import ROOT, file_sha, read_json, write_json
from sao_bridge.launch import Launcher
from sao_bridge.resources import gpu_release_audit


SCRIPT = Path(__file__).resolve()
RECOVERY = SCRIPT.parent
RUN = RECOVERY.parent


def verify_policy(prepared: dict) -> None:
    if file_sha(RUN / "USER_PAUSE_AND_DELAY.json") != prepared["user_authority_sha256"]:
        raise ValueError("User authority changed after preparing evaluation recovery")
    if (RUN / "TWO_HOUR_RESUME_CANCELLED.json").exists() or (RECOVERY / "CANCELLED.json").exists():
        raise InterruptedError("User cancelled delayed evaluation recovery")


def verify_bindings(prepared: dict) -> None:
    verify_policy(prepared)
    authority = read_json(RUN / "USER_PAUSE_AND_DELAY.json")
    interruption = read_json(RECOVERY / "INTERRUPTION_AUDIT.json")
    earliest = max(
        datetime.fromisoformat(authority["resume_not_before_utc"]).timestamp(),
        interruption["first_verified_exited_at_unix"] + authority["minimum_resubmission_interval_seconds"],
    )
    if (
        prepared["status"] != "prepared" or prepared["driver_sha256"] != file_sha(SCRIPT)
        or prepared["evaluation_source"] != str(ROOT)
        or prepared["evaluation_source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or prepared["not_before_unix"] != earliest
        or prepared["deadline_unix"] != read_json(RUN / "PROTOCOL.json")["original_absolute_deadline_unix"]
    ):
        raise ValueError("Evaluation recovery changed its validated operational source")
    for path, digest in prepared["pins"].items():
        if file_sha(Path(path)) != digest:
            raise ValueError(f"Preserved evaluation or training evidence changed: {path}")
    for role in ("editor", "solver"):
        program.verify_executed_role_identity(RUN, "full", role)
    if read_json(RUN / "full/TRAINING_AUDIT.json")["shared_environment_episodes"] != 1152:
        raise ValueError("Evaluation recovery cannot replace incomplete native training")


def wait_for_window(prepared: dict) -> None:
    earliest, deadline = prepared["not_before_unix"], prepared["deadline_unix"]
    while True:
        verify_policy(prepared)
        now = time.time()
        if now >= deadline:
            raise TimeoutError("Original study deadline expired before evaluation recovery")
        if now >= earliest:
            break
        write_json(RECOVERY / "MONITOR.json", {
            "status": "waiting_two_hour_cooldown", "pid": os.getpid(), "at_unix": now,
            "not_before_unix": earliest, "remaining_seconds": earliest - now,
            "gpu_work_started": False,
        })
        time.sleep(min(60, earliest - now, deadline - now))
    program.wait_for_resources(
        RECOVERY, "evaluation-only-resume", interval=60, stable_checks=1,
        deadline=time.monotonic() + deadline - time.time(), probe=rolling.admission_snapshot,
    )
    verify_policy(prepared)
    if not earliest <= time.time() < deadline:
        raise TimeoutError("Evaluation resume is outside its authorized time window")


def selection_config(prepared: dict) -> dict:
    previous = Path(prepared["previous_selection_attempt"])
    current = read_json(previous / "config.json")
    current.update(
        run=str(RUN / "full/round0144/selection/attempt-02"),
        resume_matrix=str(RECOVERY / "RESUME_MATRIX.json"),
    )
    return current


def recovery_costs(prepared: dict, current: dict) -> dict:
    previous, resumed = Path(prepared["previous_selection_attempt"]), Path(current["run"])
    prior_journals = previous / "episode_journals"
    completed = [
        path for path in prior_journals.glob("*/episode-*.json")
        if not path.name.endswith(".attempts.json")
    ]
    for path in completed:
        copied = resumed / "episode_journals" / path.relative_to(prior_journals)
        if file_sha(path) != file_sha(copied):
            raise ValueError("Recovery replaced a completed episode's outcome or trajectory")
    started = sum(len(read_json(path)) for path in prior_journals.glob("*/episode-*.attempts.json"))
    total = sum(
        len(read_json(path)) for path in (resumed / "episode_journals").glob("*/episode-*.attempts.json")
    )
    matrix = read_json(resumed / "report.json")
    logical = matrix["unique_harnesses"] * len(current["task_ids"]) * current["samples"]
    if total != started + logical - len(completed):
        raise ValueError("Evaluation recovery repeated or omitted an episode attempt")
    return {
        "interruption_audit": str(RECOVERY / "INTERRUPTION_AUDIT.json"),
        "interruption_audit_sha256": file_sha(RECOVERY / "INTERRUPTION_AUDIT.json"),
        "resume_manifest": str(RECOVERY / "RESUME_MATRIX.json"),
        "resume_manifest_sha256": file_sha(RECOVERY / "RESUME_MATRIX.json"),
        "evaluation_source_manifest_sha256": file_sha(ROOT / "SOURCE_MANIFEST.json"),
        "driver_sha256": file_sha(SCRIPT), "reused_completed_episodes": len(completed),
        "previous_started_episodes": started, "previous_unfinished_episodes": started - len(completed),
        "selection_logical_episodes": logical, "selection_physical_episode_starts": total,
        "repeated_completed_episodes": 0, "regenerated_harness_candidates": 0,
        "unpersisted_inflight_tokens_unknown": True, "signal_and_sender": "unknown",
    }


def finish_evaluation(prepared: dict) -> dict:
    verify_bindings(prepared)
    directory = RUN / "full"
    output = directory / "round0144"
    if any((output / name).exists() for name in ("SELECTED.json", "RESULT.json")):
        raise ValueError("This one-shot recovery cannot replace an existing selection or final endpoint")
    if (output / "evaluation200").exists() or (directory / "RESULT.json").exists():
        raise ValueError("Final scored work already started; do not replay it")
    records = {role: read_json(directory / role / "COMPLETE.json") for role in ("editor", "solver")}
    pair = {role: records[role]["checkpoint_history"]["143"] for role in records}
    diagnostic = read_json(directory / "round0096/RESULT.json")
    incumbent = diagnostic["selection"]["harness"]
    protocol = read_json(RUN / "PROTOCOL.json")
    current = selection_config(prepared)
    configuration = output / "selection/config-02.json"
    if Path(current["run"]).exists():
        raise ValueError("The bounded second selection attempt already exists")
    _immutable_json(configuration, current)
    launch = Launcher(RECOVERY / "supervision")
    try:
        values = generate_proposals(
            launch, protocol, read_json(RUN / "INITIAL_STATE.json"),
            {"candidate": pair["editor"]}, output / "generation",
        )["candidate"]
        expected_harnesses = {"incumbent_harness": incumbent}
        for value in values:
            label = f"candidate-proposal{value['sample_index']:02d}"
            path = output / "patches" / f"{label}.json"
            patch = value["patch"] if value["valid"] else read_json(Path(protocol["no_patch"]))
            if read_json(path) != patch:
                raise ValueError("A preserved proposal artifact changed after interruption")
            expected_harnesses[label] = str(path)
        if current["harnesses"] != expected_harnesses or current["solver_model"] != pair["solver"]["actor"]:
            raise ValueError("Recovered selection changed the trained Solver or candidate set")
        write_json(RECOVERY / "MONITOR.json", {
            "status": "resuming_selection48", "pid": os.getpid(), "at_unix": time.time(),
            "completed_episodes_to_reuse": prepared["completed_selection_episodes"],
            "training_restarted": False,
        })
        process = launch.module(
            "sao_bridge.alfworld_solver_eval", ["--config", configuration, "--harness-matrix"],
            gpus="0,1,2,3,4,5,6,7", name="selection-recovered-attempt-2",
        )
        launch.wait(process, "remaining frozen selection48 episodes", timeout=14400)
        matrix = checked_matrix(Path(current["run"]), current)
        reports = matrix_reports(matrix, protocol)
        chosen = study.choose_harness(values, reports, label="candidate", margin=protocol["incumbent_min_gain"])
        selection = {
            "harness": expected_harnesses[chosen], "label": chosen,
            "new_harness_selected": chosen != "incumbent_harness",
            "valid_proposals": sum(value["valid"] for value in values),
            "selection_scores": {name: report["accuracy"] for name, report in reports.items()},
            "selection_matrix": current["run"], "pair": pair, "at_unix": time.time(),
            "main200_used_to_select": False,
        }
        _immutable_json(output / "SELECTED.json", selection)
        costs = recovery_costs(prepared, current)
        _immutable_json(RECOVERY / "SELECTION_RECOVERY_AUDIT.json", costs)
        cohort = read_json(Path(protocol["core200_protocol"]))
        config = {
            **matrix_config(protocol, pair["solver"]["actor"], "full-144",
                            {"selected": selection["harness"]}, output / "evaluation200"),
            "task_ids": cohort["all_task_ids"], "samples": 1,
            **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
            "matrix_concurrency": 2, "episode_checkpointing": True,
        }
        opened = time.time()
        write_json(RECOVERY / "MONITOR.json", {
            "status": "evaluating_final200", "pid": os.getpid(), "at_unix": opened,
            "selection_fixed_before_main": True, "training_restarted": False,
        })
        matrix, matrix_path = run_matrix(launch, config, output / "evaluation200", 1)
        scores = {}
        for name, value in matrix["variants"].items():
            path = Path(value["report"])
            report = checked_native_report(read_json(path), cohort["all_task_ids"], config)
            scores[name] = {
                "successes": int(sum(row["reward"] for row in report["results"])),
                "accuracy": report["accuracy"], "questions": 200,
                "report": str(path), "sha256": file_sha(path),
            }
        endpoint = {
            "status": "completed", "arm": "full", "rounds": 144, "selection": selection,
            "scores": scores, "main_matrix": str(matrix_path), "main_started_at_unix": opened,
            "physical_main_episodes": matrix["unique_harnesses"] * 200,
            "primary_endpoint": True, "at_unix": time.time(),
        }
        if selection["at_unix"] >= opened:
            raise ValueError("The Harness selection was not fixed before main200 evaluation")
    finally:
        launch.close()
        write_json(output / "RELEASED.json", gpu_release_audit())
    checked_release(output)
    _immutable_json(output / "RESULT.json", endpoint)
    verify_bindings(prepared)
    training = {
        "status": "completed", "attempt": prepared["training_attempt"],
        "audit": read_json(directory / "TRAINING_AUDIT.json"),
    }
    result = rolling.publish_arm_result(
        RUN, "full", training, {"96": diagnostic, "144": endpoint}, evaluation_recovery=costs,
    )
    program.audit_full_result(RUN)
    program.write_master_report(RUN)
    return result


def main() -> None:
    os.umask(0o002)
    prepared = read_json(RECOVERY / "PREPARED.json")
    validation = read_json(RECOVERY / "CODE_VALIDATION.json")
    if (
        validation["status"] != "passed" or validation["driver_sha256"] != file_sha(SCRIPT)
        or validation["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or validation["test_log_sha256"] != file_sha(Path(validation["test_log"]))
    ):
        raise ValueError("Evaluation-only recovery lacks exact-source passing tests")
    with (RUN / "TWO_HOUR_CONTROL.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(RECOVERY / "STARTED.json", {
            "pid": os.getpid(), "at_unix": time.time(), "attached": True,
            "not_before_unix": prepared["not_before_unix"], "prepared_sha256": file_sha(RECOVERY / "PREPARED.json"),
        })

        def interrupted(signum, frame):
            write_json(RECOVERY / "SIGNAL.json", {"signal": signum, "at_unix": time.time()})
            raise InterruptedError(f"Evaluation-only controller received signal{signum}")

        def expired(signum, frame):
            raise TimeoutError("The original absolute study deadline expired")

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(signum, interrupted)
        signal.signal(signal.SIGALRM, expired)
        remaining = prepared["deadline_unix"] - time.time()
        if remaining <= 0:
            raise TimeoutError("No time remains before the original study deadline")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            verify_bindings(prepared)
            wait_for_window(prepared)
            result = finish_evaluation(prepared)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ExceptionGroup) as exc:
            write_json(RECOVERY / "FAILED.json", {
                "status": "incomplete", "error": repr(exc), "at_unix": time.time(),
                "completed_training_must_not_be_replayed": True,
            })
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        _immutable_json(RECOVERY / "COMPLETE.json", {
            "status": result["status"], "at_unix": time.time(),
            "full_result_sha256": file_sha(RUN / "full/RESULT.json"),
        })
        write_json(RECOVERY / "MONITOR.json", {
            "status": "full_result_audited_pending_report_before_ablations", "at_unix": time.time(),
        })


if __name__ == "__main__":
    main()
