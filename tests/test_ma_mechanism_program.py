from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from sao_bridge import alfworld_ma_mechanism_program as program
from sao_bridge.common import file_sha, read_json, write_json


def save_arm(root, arm):
    directory = root / arm
    result = {"endpoints": {"144": {"scores": {
        "selected": {"successes": 190, "accuracy": .95},
        "no_patch": {"successes": 180, "accuracy": .9},
    }}}}
    if arm != "full":
        result["paired_against_full"] = {
            "candidate_minus_control": -.025, "paired_question_bootstrap_95pct": [-.05, .005],
            "improved_questions": 1, "regressed_questions": 6,
        }
    write_json(directory / "RESULT.json", result)
    (directory / "RESULT.md").write_text("verified fixture\n")
    write_json(directory / "FINAL_AUDIT.json", {
        "status": "passed", "result_sha256": file_sha(directory / "RESULT.json"),
        "report_sha256": file_sha(directory / "RESULT.md"),
    })


@pytest.mark.parametrize(
    "fault", [None, "reward", "seed", "tokens", "messages", "empty-messages", "model", "harness", "matrix"],
)
def test_historical_case_analysis_resolves_only_the_exact_unabridged_source(tmp_path, monkeypatch, fault):
    endpoint = tmp_path / "round0096_COMPLETE.json"
    source = tmp_path / "round0096/evaluation200/attempt-01"
    native_path = source / "reports/native.json"
    compact_path = tmp_path / "round0096_joint_harness_report.json"
    peak = {"selection": {"pair": {"solver": {"actor": "historical-solver"}}, "harness": "historical-harness"}}
    write_json(endpoint, peak)
    config = {"solver_model": "historical-solver", "harnesses": {"joint_harness": "historical-harness"}}
    if fault == "model":
        config["solver_model"] = "different-solver"
    elif fault == "harness":
        config["harnesses"]["joint_harness"] = "different-harness"
    write_json(source.parent / "config-01.json", config)
    compact = {
        "accuracy": 1., "results": [{
            "id": "alfworld-240", "task_id": 240, "sample_index": 0,
            "rollout_seed": 6919639, "reward": 1., "tokens": 100,
        }],
    }
    write_json(compact_path, compact)
    native = {**compact, "results": [{
        **compact["results"][0],
        "messages": [{"role": "user", "content": "Your task is to: put the apple on the table."}],
    }]}
    if fault in ("reward", "tokens"):
        native["results"][0][fault] = 0
    elif fault == "seed":
        native["results"][0]["rollout_seed"] += 1
    elif fault == "messages":
        native["results"][0].pop("messages")
    elif fault == "empty-messages":
        native["results"][0]["messages"] = []
    write_json(native_path, native)
    calls = []

    def checked_matrix(path, expected):
        assert path == source and expected == config
        if fault == "matrix":
            raise ValueError("Historical matrix did not pass its audit")
        calls.append(path)
        return {"variants": {"joint_harness": {"report": str(native_path)}}}

    monkeypatch.setattr(program, "checked_matrix", checked_matrix)
    lineage = {"historical_endpoint": str(endpoint), "historical_report": str(compact_path)}
    if fault is None:
        before = file_sha(compact_path)
        assert program.historical_trace_report(lineage) == native_path
        assert file_sha(compact_path) == before and len(calls) == 1
    else:
        with pytest.raises(ValueError):
            program.historical_trace_report(lineage)


def test_full_path_recovery_preserves_old_audit_and_uses_native_case_messages(tmp_path, monkeypatch):
    save_arm(tmp_path, "full")
    compact_path = tmp_path / "historical-summary.json"
    native_path = tmp_path / "historical-native.json"
    selected_path = tmp_path / "selected-native.json"
    for path in (compact_path, native_path, selected_path):
        write_json(path, {"results": []})
    result_path = tmp_path / "full/RESULT.json"
    result = read_json(result_path)
    result["endpoints"]["144"]["scores"]["selected"]["report"] = str(selected_path)
    write_json(result_path, result)
    final_path = tmp_path / "full/FINAL_AUDIT.json"
    final = read_json(final_path)
    final["result_sha256"] = file_sha(result_path)
    write_json(final_path, final)
    old = tmp_path / "full/HISTORICAL_PATH_RECHECK.json"
    write_json(old, {"at_unix": 123, "status": "passed"})
    old_sha = file_sha(old)
    destination = "full/postprocess_resume01/HISTORICAL_PATH_RECHECK.json"
    lineage = {"historical_report": str(compact_path), "known_differences": ["historical initialization"]}

    def audit_historical_path(root, *, destination):
        write_json(root / destination, lineage)
        return lineage

    calls = []

    def cases(output, control, full, cohort, *, title):
        assert control == selected_path and full == native_path
        calls.append(output)
        return {"examples": [], "control_success_full_failure": 2}

    monkeypatch.setattr(program, "audit_historical_path", audit_historical_path)
    monkeypatch.setattr(program, "historical_trace_report", lambda value: native_path)
    monkeypatch.setattr(program, "verify_executed_role_identity", lambda *args: None)
    monkeypatch.setattr(program, "paired_difference", lambda *args, **kwargs: {"candidate_minus_control": -.02})
    monkeypatch.setattr(program, "write_case_study", cases)
    cohort = tmp_path / "cohort.json"
    write_json(cohort, {"all_task_ids": []})
    write_json(tmp_path / "PROTOCOL.json", {"core200_protocol": str(cohort)})
    audit = program.audit_full_result(tmp_path, lineage_destination=destination)
    assert file_sha(old) == old_sha
    assert audit["lineage_audit"] == str(tmp_path / destination)
    assert audit["historical_unabridged_report_sha256"] == file_sha(native_path)
    assert audit["historical_native_matches_compact_report"] is True
    assert audit["historical_failure_case_population"]["control_success_full_failure"] == 2
    assert calls == [tmp_path / "full/failure-cases-vs97"]


@pytest.mark.parametrize("improved,regressed,expected", [
    (0, 0, 1.), (0, 6, .03125), (6, 0, .03125), (1, 5, .21875), (3, 3, 1.),
])
def test_exact_paired_tests_include_ties_and_reverse_direction(improved, regressed, expected):
    assert program.paired_exact_p(improved, regressed) == pytest.approx(expected)


def test_four_comparison_holm_adjustment_is_monotone_and_rejects_invalid_values():
    result = program.holm_adjust({"d": .2, "c": .04, "b": .02, "a": .01})
    assert result == pytest.approx({"a": .04, "b": .06, "c": .08, "d": .2})
    with pytest.raises(ValueError):
        program.holm_adjust({"bad": float("nan")})
    with pytest.raises(ValueError):
        program.paired_exact_p(-1, 0)


def fake_clock(monkeypatch):
    clock = [0.]
    monkeypatch.setattr(program.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(program.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    return clock


def test_resource_gate_needs_consecutive_free_checks_without_touching_jobs(tmp_path, monkeypatch):
    clock = fake_clock(monkeypatch)
    values = iter([False, True, False, True, True, True])
    monkeypatch.setattr(program, "resource_snapshot", lambda: {"released": next(values)})
    result = program.wait_for_resources(tmp_path, "full", deadline=100, interval=1, stable_checks=3)
    assert result["released"] is True and clock[0] == 5
    monitor = read_json(tmp_path / "PROGRAM_MONITOR.json")
    assert monitor["status"] == "resources_ready" and monitor["consecutive_free_checks"] == 3
    assert not (tmp_path / "full/STARTED.json").exists()


def test_resource_wait_timeout_and_probe_error_are_explicit(tmp_path, monkeypatch, capsys):
    fake_clock(monkeypatch)

    def fail():
        raise subprocess.TimeoutExpired("read-only ssh", 60)

    monkeypatch.setattr(program, "resource_snapshot", fail)
    with pytest.raises(TimeoutError, match="resource window"):
        program.wait_for_resources(tmp_path, "full", deadline=2, interval=1)
    assert "RESOURCE PROBE ERROR" in capsys.readouterr().out
    assert "TimeoutExpired" in read_json(tmp_path / "PROGRAM_MONITOR.json")["probe_error"]


def test_partial_table_does_not_claim_complete_or_adjust_a_partial_family(tmp_path):
    write_json(tmp_path / "PRIOR_WORK.json", {"interrupted_rounds": 22})
    write_json(tmp_path / "NATIVE_GATE.json", {"calibration_environment_episodes": 96})
    save_arm(tmp_path, "full")
    save_arm(tmp_path, "without_peer")
    partial = program.write_master_report(tmp_path)
    assert partial["status"] == "in_progress" and not partial["holm_adjusted_p_four_comparisons"]
    assert partial["prior_work"]["interrupted_rounds"] == 22
    assert "+2.5 [-0.5,+5.0]" in (tmp_path / "RESULT.md").read_text()
    for arm in program.study.ARMS[2:]:
        save_arm(tmp_path, arm)
    complete = program.write_master_report(tmp_path)
    assert complete["status"] == "completed"
    assert set(complete["holm_adjusted_p_four_comparisons"]) == set(program.study.ARMS[1:])


def test_baseline_report_hash_and_explicit_release_gate(tmp_path):
    save_arm(tmp_path, "full")
    write_json(tmp_path / "PROGRAM_PREPARED.json", {"status": "prepared", "pins": {}})
    write_json(tmp_path / "full/BASELINE_PATH_AUDIT.json", {
        "status": "passed", "full_result_sha256": file_sha(tmp_path / "full/RESULT.json"),
    })
    with pytest.raises(FileNotFoundError):
        program.run(tmp_path, ablations=True)
    assert not (tmp_path / "ABLATIONS_PROGRAM_STARTED.json").exists()
    write_json(tmp_path / "BASELINE_RELEASE.json", {
        "status": "audited_and_reported", "full_result_sha256": "wrong", "baseline_audit_sha256": "wrong",
    })
    with pytest.raises(ValueError, match="review"):
        program.run(tmp_path, ablations=True)
    (tmp_path / "full/RESULT.md").write_text("changed\n")
    with pytest.raises(ValueError, match="final audit"):
        program.checked_result(tmp_path, "full")


def test_removed_staged_arm_cannot_start(tmp_path, monkeypatch):
    monkeypatch.setattr(program.campaign, "require_prerequisites", lambda root: {"arms": list(program.study.ARMS)})
    with pytest.raises(ValueError, match="Unknown"):
        program.campaign.run_arm(tmp_path, "without_team_learning")
    assert not (tmp_path / "without_team_learning/STARTED.json").exists()


def test_remote_launch_is_attached_traced_and_uses_shared_execution_group(tmp_path, monkeypatch):
    write_json(tmp_path / "PROTOCOL.json", {"unit_suffix": "single-seed-v2", "signal_tracing": True})
    monkeypatch.setattr(program.campaign, "unit_state", lambda *args, **kwargs: {"LoadState": "not-found"})
    commands = []

    def start(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=123)

    record = program.campaign.start_remote(
        SimpleNamespace(start=start), tmp_path, "full-editor", "sao_bridge.example", ["--root", str(tmp_path)],
    )
    command = shlex.split(commands[0][-1])
    assert record["unit"].endswith("single-seed-v2")
    assert "--wait" in command and "--pipe" in command
    assert "--uid=1009" in command and "--gid=1008" in command
    assert command.index("/usr/bin/strace") < command.index(program.PYTHON)


def configure_fake_program(root, monkeypatch):
    write_json(root / "PROGRAM_PREPARED.json", {
        "status": "prepared", "pins": {}, "overall_wall_limit_hours": 40, "resource_wait_limit_hours": 12,
    })
    monkeypatch.setattr(program, "wait_for_resources", lambda *args, **kwargs: {"released": True})
    monkeypatch.setattr(program.campaign, "require_prerequisites", lambda *args, **kwargs: {})
    monkeypatch.setattr(program.signal, "signal", lambda *args: None)
    monkeypatch.setattr(program.signal, "setitimer", lambda *args: None)


def test_control_failure_persists_and_cannot_automatically_retry(tmp_path, monkeypatch):
    configure_fake_program(tmp_path, monkeypatch)
    called = []

    def fail(root, arm, **kwargs):
        called.append(arm)
        raise RuntimeError("injected native failure")

    monkeypatch.setattr(program.campaign, "train_joint", fail)
    with pytest.raises(RuntimeError, match="injected"):
        program.run(tmp_path)
    assert called == ["full"]
    failure = read_json(tmp_path / "PROGRAM_FAILED.json")
    assert failure["phase"] == "native-full" and failure["automatic_retry"] is False
    assert not (tmp_path / "CONTROL_PROGRAM_COMPLETE.json").exists()
    with pytest.raises(ValueError, match="explicit recovery"):
        program.run(tmp_path)
    assert called == ["full"]
    assert program.watch(tmp_path, 1)["status"] == "program_failed"


def test_successful_control_stops_before_ablations_and_requires_review(tmp_path, monkeypatch):
    configure_fake_program(tmp_path, monkeypatch)
    write_json(tmp_path / "PRIOR_WORK.json", {"interrupted_rounds": 22})
    write_json(tmp_path / "PINS.json", {"pins": {}})
    write_json(tmp_path / "MEMORY_GATE_INHERITANCE.json", {"status": "passed"})
    calls = []

    def calibration(root, arm, **kwargs):
        audit = {"status": "passed", "shared_environment_episodes": 48}
        write_json(root / arm / "TRAINING_AUDIT.json", audit)
        return audit

    def run_arm(root, arm):
        calls.append(arm)
        save_arm(root, arm)

    monkeypatch.setattr(program.campaign, "train_joint", calibration)
    monkeypatch.setattr(program.campaign, "run_arm", run_arm)
    monkeypatch.setattr(program, "audit_full_result", lambda root: {
        "full_accuracy": .95, "below_reference_by_at_least_3pp": False,
    })
    result = program.run(tmp_path)
    assert result["status"] == "control_ready_for_review" and calls == ["full"]
    assert read_json(tmp_path / "RESULT.json")["status"] == "in_progress"
    assert read_json(tmp_path / "PROGRAM_MONITOR.json")["status"] == "control_ready_for_review"
    assert program.watch(tmp_path, 1)["completed_arms"] == ["full"]
    assert not (tmp_path / "PROGRAM_COMPLETE.json").exists()
    with pytest.raises(FileNotFoundError):
        program.run(tmp_path, ablations=True)
    assert calls == ["full"]
