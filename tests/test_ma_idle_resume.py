import copy
import math
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from sao_bridge import alfworld_ma_idle_resume as monitor
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.joint_recovery import science_digest


def authority(tmp_path):
    root = tmp_path / "study"
    control = root / "idle-resume"
    write_json(root / "PROTOCOL.json", {"original_absolute_deadline_unix": 100.0})
    for relative in monitor.PAUSE_RECORDS:
        write_json(root / relative, {"h200_training_allowed": False})
    policy = {
        "schema": monitor.SCHEMA, "run_root": str(root), "authorized_at_unix": 1000.0,
        "h200_training_allowed": True, "automatic_submission_allowed": True,
        "supersedes_prior_pause": True, "poll_interval_seconds": 600,
        "minimum_idle_seconds": 7200, "minimum_samples": 13,
        "nodes": monitor.HOSTS, "arms": list(monitor.ARMS), "expires_at_unix": None,
        "superseded_cancellations": {relative: file_sha(root / relative) for relative in monitor.PAUSE_RECORDS},
        "original_protocol_sha256": file_sha(root / "PROTOCOL.json"), "original_deadline_unix": 100.0,
    }
    path = control / "AUTHORIZATION.json"
    write_json(path, policy)
    return root, control, file_sha(path)


def admission(tmp_path):
    root, control, digest = authority(tmp_path)
    path = control / "admissions/one.json"
    write_json(path, {
        "eligible": True, "authority_sha256": digest, "samples": 13,
        "elapsed_seconds": 7200, "first_sample_unix": 1000, "last_sample_unix": 8200,
        "final_recheck": {"eligible": True}, "at_unix": 8201,
    })
    return root, control, {
        "authority": str(control / "AUTHORIZATION.json"), "authority_sha256": digest,
        "admission": str(path), "admission_sha256": file_sha(path),
    }


def idle_node():
    return {
        "gpus": [{"index": index, "name": "NVIDIA H200", "utilization": 0, "memory_mib": 0}
                 for index in range(8)],
        "gpu_pids": [], "verified_filler_pids": [], "foreign_gpu_pids": [],
        "vanished_gpu_pids": [], "distributed_launchers": [],
        "verified_filler_unit": False, "filler_kind": None,
    }


def test_thirteen_samples_span_at_least_two_real_hours():
    window = monitor.IdleWindow()
    for index in range(12):
        assert not window.observe(True, index * 600, 1000 + index * 600)
    assert not window.observe(True, 7199, 8199)
    assert window.samples == 13
    assert not window.observe(True, 7200, 8200)
    assert window.observe(True, 7800, 8800)
    exact = monitor.IdleWindow()
    assert [exact.observe(True, i * 600, 1000 + i * 600) for i in range(13)] == [False] * 12 + [True]


def test_fast_repeated_samples_do_not_satisfy_the_gate():
    window = monitor.IdleWindow()
    for value in range(300):
        assert not window.observe(True, value, 1000 + value)
    assert window.samples == 1


@pytest.mark.parametrize("kind", ["busy", "gap", "forward_clock", "backward_clock"])
def test_busy_errors_gaps_and_clock_changes_reset_the_whole_window(kind):
    window = monitor.IdleWindow()
    for index in range(12):
        window.observe(True, index * 600, 1000 + index * 600)
    if kind == "busy":
        assert not window.observe(False, 7200, 8200)
        assert window.samples == 0
    else:
        monotonic, unix = {
            "gap": (7300, 8300), "forward_clock": (7200, 9200), "backward_clock": (7200, 7199),
        }[kind]
        assert not window.observe(True, monotonic, unix)
        assert window.samples == 1 and window.elapsed == 0


def test_process_reentry_never_reuses_previous_idle_credit():
    original = monitor.IdleWindow()
    for index in range(13):
        original.observe(True, index * 600, 1000 + index * 600)
    restarted = monitor.IdleWindow()
    assert not restarted.observe(True, 7300, 8300)
    assert restarted.samples == 1 and restarted.elapsed == 0


def test_idle_and_verified_filler_are_eligible_even_when_mixed_over_time():
    idle = idle_node()
    filler = copy.deepcopy(idle)
    filler.update(gpu_pids=[42], verified_filler_pids=[42], verified_filler_unit=True, filler_kind="adaptive")
    filler["gpus"][0].update(utilization=100, memory_mib=1000)
    assert monitor.classify_node(idle) == "idle"
    assert monitor.classify_node(filler) == "verified_filler_only"
    window = monitor.IdleWindow()
    for index in range(13):
        value = idle if index % 2 else filler
        ready = window.observe(monitor.classify_node(value) in ("idle", "verified_filler_only"),
                               index * 600, index * 600 + 1000)
    assert ready


@pytest.mark.parametrize("change", [
    {"foreign_gpu_pids": [7]}, {"vanished_gpu_pids": [7]},
    {"distributed_launchers": [{"pid": 7}]},
    {"gpu_pids": [7], "verified_filler_pids": [7], "verified_filler_unit": False},
])
def test_foreign_unknown_disappearing_or_preallocation_processes_are_not_idle(change):
    value = idle_node()
    value.update(change)
    assert monitor.classify_node(value) not in ("idle", "verified_filler_only")


def test_high_utilization_without_an_attributable_process_is_not_idle():
    value = idle_node()
    value["gpus"][0]["utilization"] = 40
    assert monitor.classify_node(value) == "unattributed_gpu_activity"
    value["gpus"][0].update(utilization=0, memory_mib=100)
    assert monitor.classify_node(value) == "unattributed_gpu_activity"


def test_missing_card_or_unknown_metrics_fail_closed():
    value = idle_node()
    value["gpus"].pop()
    with pytest.raises(ValueError, match="eight"):
        monitor.classify_node(value)
    value = idle_node()
    value["gpus"][0]["utilization"] = "N/A"
    with pytest.raises(ValueError, match="eight"):
        monitor.classify_node(value)


def test_probe_failure_is_visible_and_never_success_shaped(monkeypatch, capsys):
    def failed():
        raise subprocess.TimeoutExpired("read-only-ssh", 90)
    monkeypatch.setattr(monitor, "resource_snapshot", failed)
    value = monitor.sample_resources()
    assert not value["eligible"] and "TimeoutExpired" in value["probe_error"]
    assert "RESOURCE PROBE ERROR" in capsys.readouterr().out


def test_old_deadline_remains_default_and_new_authorization_is_operational_only(tmp_path):
    root, control, binding = admission(tmp_path)
    assert monitor.execution_deadline(100, None, root=root, arm="without_peer") == 100
    assert math.isinf(monitor.execution_deadline(100, binding, root=root, arm="without_peer"))
    spec = {
        "training_deadline_unix": 100.0, "component_ablation": "without_peer",
        "rolling_recovery": {"root": str(root / "without_peer/recovery")}, "learning_rate": 5e-7,
    }
    before = science_digest(spec)
    spec["rolling_recovery"]["execution_authority"] = binding
    assert science_digest(spec) == before and math.isinf(monitor.spec_deadline(spec))
    spec["training_deadline_unix"] = 101
    assert science_digest(spec) != before
    with pytest.raises(ValueError, match="scope"):
        monitor.spec_deadline(spec)


@pytest.mark.parametrize("change", ["new_cancel", "old_cancel", "authority_change", "wrong_arm", "wrong_root"])
def test_cancellation_and_scope_are_enforced_after_admission(tmp_path, change):
    root, control, binding = admission(tmp_path)
    arm, target = "without_peer", root
    if change == "new_cancel":
        write_json(control / "CANCELLED.json", {"cancelled": True})
    elif change == "old_cancel":
        write_json(root / monitor.PAUSE_RECORDS[0], {"updated_pause": True})
    elif change == "authority_change":
        write_json(control / "AUTHORIZATION.json", {"changed": True})
    elif change == "wrong_arm":
        arm = "full"
    else:
        target = tmp_path / "another_study"
    with pytest.raises((ValueError, InterruptedError)):
        monitor.execution_deadline(100, binding, root=target, arm=arm)


@pytest.mark.parametrize("field,value", [
    ("samples", 12), ("elapsed_seconds", 7199),
    ("first_sample_unix", 999), ("last_sample_unix", 8199), ("eligible", False),
])
def test_incomplete_admission_never_relaxes_the_legacy_deadline(tmp_path, field, value):
    root, _, binding = admission(tmp_path)
    path = Path(binding["admission"])
    write_json(path, {**read_json(path), field: value})
    binding["admission_sha256"] = file_sha(path)
    with pytest.raises(ValueError, match="two-hour"):
        monitor.execution_deadline(100, binding, root=root, arm="without_peer")


def test_wait_loop_uses_exact_ten_minute_cadence_and_fresh_prelaunch_probe(tmp_path, monkeypatch):
    _, control, digest = authority(tmp_path)
    clock = [1000.0]
    calls = []
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock[0] - 1000)
    monkeypatch.setattr(monitor.time, "time", lambda: clock[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: {})
    def probe():
        calls.append(clock[0])
        return {"eligible": True}
    monkeypatch.setattr(monitor, "sample_resources", probe)
    binding = monitor.wait_for_window(control, digest)
    assert calls == [1000 + i * 600 for i in range(13)] + [8200]
    receipt = read_json(Path(binding["admission"]))
    assert receipt["samples"] == 13 and receipt["elapsed_seconds"] == 7200
    assert len((control / "SAMPLES.jsonl").read_text().splitlines()) == 13


def test_busy_final_recheck_requires_a_whole_new_window(tmp_path, monkeypatch):
    _, control, digest = authority(tmp_path)
    clock = [1000.0]
    calls = []
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock[0] - 1000)
    monkeypatch.setattr(monitor.time, "time", lambda: clock[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: {})
    def probe():
        calls.append(clock[0])
        return {"eligible": len(calls) != 14}
    monkeypatch.setattr(monitor, "sample_resources", probe)
    binding = monitor.wait_for_window(control, digest)
    receipt = read_json(Path(binding["admission"]))
    assert receipt["first_sample_unix"] == 8800
    assert receipt["last_sample_unix"] == 16000 and clock[0] == 16000


def test_time_spent_before_the_first_completed_probe_is_not_idle_credit(tmp_path, monkeypatch):
    _, control, digest = authority(tmp_path)
    clock = [1000.0]
    first = [True]
    monkeypatch.setattr(monitor.time, "monotonic", lambda: clock[0] - 1000)
    monkeypatch.setattr(monitor.time, "time", lambda: clock[0])
    monkeypatch.setattr(monitor.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: {})
    def probe():
        if first[0]:
            clock[0] += 50
            first[0] = False
        return {"eligible": True}
    monkeypatch.setattr(monitor, "sample_resources", probe)
    binding = monitor.wait_for_window(control, digest)
    receipt = read_json(Path(binding["admission"]))
    assert receipt["first_sample_unix"] == 1050
    assert receipt["last_sample_unix"] == 8250
    assert receipt["elapsed_seconds"] == 7200


def session_fixture(tmp_path):
    root, control, binding = admission(tmp_path)
    session = control / "sessions/one"
    write_json(session / "BINDING.json", binding)
    path = root / "without_peer/attempts/attempt-0002"
    write_json(path / "PREPARED.json", {"execution_authority": binding})
    write_json(root / "ACTIVE_ATTEMPT.json", {"arm": "without_peer", "path": str(path)})
    return root, session, path


def test_only_verified_new_attempt_interruptions_can_retry(tmp_path):
    root, session, path = session_fixture(tmp_path)
    assert monitor.retry_evidence(root, "old-attempt", -signal.SIGKILL, session)["kind"] == "owned_worker_terminated"
    assert monitor.retry_evidence(root, str(path), -signal.SIGKILL, session) is None
    write_json(path / "FAILED.json", {"retry_after_idle_allowed": False, "evidence": []})
    assert monitor.retry_evidence(root, "old-attempt", -signal.SIGKILL, session) is None
    write_json(path / "FAILED.json", {"retry_after_idle_allowed": True, "evidence": [{"kind": "native_termination"}]})
    assert monitor.retry_evidence(root, "old-attempt", 1, session)["failure"] == str(path / "FAILED.json")
    write_json(root / "without_peer/TRAINING_AUDIT.json", {"status": "passed"})
    assert monitor.retry_evidence(root, "old-attempt", -signal.SIGKILL, session) is None


def test_an_unowned_active_attempt_cannot_be_cleaned_up_or_retried(tmp_path):
    root, session, path = session_fixture(tmp_path)
    write_json(path / "PREPARED.json", {"execution_authority": {"unrelated": True}})
    with pytest.raises(ValueError, match="does not belong"):
        monitor.session_attempt(root, "old-attempt", session)


def test_completed_training_is_not_replayed_after_evaluation_failure(tmp_path):
    root, _, _ = authority(tmp_path)
    write_json(root / "without_peer/TRAINING_AUDIT.json", {"status": "passed"})
    with pytest.raises(ValueError, match="not automatic retraining"):
        monitor.pending_arms(root)


def test_attempt_preparation_preserves_the_original_scientific_identity(tmp_path, monkeypatch):
    from sao_bridge import alfworld_ma_rolling_recovery as rolling

    root, _, binding = admission(tmp_path)
    write_json(root / "PROTOCOL.json", {
        "original_absolute_deadline_unix": 100.0, "unit_suffix": "test",
    })
    policy_path = Path(binding["authority"])
    policy = read_json(policy_path)
    policy["original_protocol_sha256"] = file_sha(root / "PROTOCOL.json")
    write_json(policy_path, policy)
    binding["authority_sha256"] = file_sha(policy_path)
    admission_path = Path(binding["admission"])
    write_json(admission_path, {**read_json(admission_path), "authority_sha256": binding["authority_sha256"]})
    binding["admission_sha256"] = file_sha(admission_path)
    old = {}
    for role in ("editor", "solver"):
        spec = {
            "steps": 144, "role": role, "component_ablation": "without_peer",
            "training_deadline_unix": 100.0,
            "rolling_recovery": {"root": str(root / "without_peer/recovery")},
        }
        old[role] = spec
        write_json(root / "without_peer" / f"{role}_spec.json", spec)
    monkeypatch.setattr(rolling, "latest_checkpoint", lambda _: None)
    monkeypatch.setattr(rolling.study, "validate_spec", lambda _: None)
    path, start = rolling.prepare_attempt(root, "without_peer", execution_authority=binding)
    assert start == 0
    for role in old:
        spec = read_json(path / "without_peer" / f"{role}_spec.json")
        assert science_digest(spec) == science_digest(old[role])
        assert math.isinf(monitor.spec_deadline(spec))


def test_invalid_authorization_creates_no_new_attempt(tmp_path):
    from sao_bridge.alfworld_ma_rolling_recovery import prepare_attempt

    root, control, binding = admission(tmp_path)
    write_json(control / "CANCELLED.json", {"cancelled": True})
    with pytest.raises(InterruptedError):
        prepare_attempt(root, "without_peer", execution_authority=binding)
    assert not (root / "without_peer/attempts").exists()


def test_worker_runs_only_remaining_arms_with_the_same_admission(tmp_path, monkeypatch):
    from sao_bridge import alfworld_ma_rolling_recovery as rolling
    from sao_bridge import alfworld_ma_mechanism_program as program

    root, control, binding = admission(tmp_path)
    session = control / "sessions/one"
    write_json(session / "BINDING.json", binding)
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: read_json(control / "AUTHORIZATION.json"))
    monkeypatch.setattr(rolling, "admission_snapshot", lambda: {"released": True})
    events = []
    def train(directory, arm, **kwargs):
        assert directory == root and kwargs["execution_authority"] == binding
        events.append(("train", arm))
        return {"attempt": str(root / arm / "attempts/attempt-0001")}
    monkeypatch.setattr(rolling, "run_attempt", train)
    monkeypatch.setattr(rolling, "finish_arm", lambda directory, arm, _: events.append(("evaluate", arm)))
    monkeypatch.setattr(program, "write_master_report", lambda _: events.append(("report", None)))
    monitor.execute_worker(control, session)
    assert [arm for kind, arm in events if kind == "train"] == list(monitor.ARMS)
    assert [arm for kind, arm in events if kind == "evaluate"] == list(monitor.ARMS)
    assert not (root / "full/STARTED.json").exists()
    assert read_json(session / "COMPLETE.json")["status"] == "completed"


def test_worker_rechecks_resources_before_creating_any_attempt(tmp_path, monkeypatch):
    from sao_bridge import alfworld_ma_rolling_recovery as rolling

    root, control, binding = admission(tmp_path)
    session = control / "sessions/one"
    write_json(session / "BINDING.json", binding)
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: read_json(control / "AUTHORIZATION.json"))
    monkeypatch.setattr(rolling, "admission_snapshot", lambda: {"released": False})
    with pytest.raises(monitor.ResourcesChanged):
        monitor.execute_worker(control, session)
    assert not (root / "without_peer/STARTED.json").exists()


def test_strict_launcher_probe_does_not_ignore_access_denied(tmp_path, monkeypatch):
    import psutil
    from sao_bridge.alfworld_ma_rolling_recovery import distributed_launchers

    def denied():
        raise psutil.AccessDenied(pid=123)
    process = SimpleNamespace(pid=123, info={"cmdline": None, "uids": None}, cmdline=denied)
    monkeypatch.setattr(psutil, "process_iter", lambda _: iter([process]))
    with pytest.raises(psutil.AccessDenied):
        distributed_launchers(strict=True)
    assert distributed_launchers() == []


@pytest.mark.parametrize("executable", ["torchrun", "deepspeed", "mpirun"])
def test_launcher_entrypoints_are_busy_before_any_gpu_allocation(monkeypatch, executable):
    import psutil
    from sao_bridge.alfworld_ma_rolling_recovery import distributed_launchers

    process = SimpleNamespace(
        pid=123, info={}, cmdline=lambda: ["/usr/bin/" + executable, "train.py"],
        uids=lambda: SimpleNamespace(real=1008),
    )
    monkeypatch.setattr(psutil, "process_iter", lambda _: iter([process]))
    assert distributed_launchers(strict=True) == [{"pid": 123, "uid": 1008, "module": executable}]
