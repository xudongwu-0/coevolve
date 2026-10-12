import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
DRIVER = ROOT / "runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/resume_after_two_hours.py"
loader = importlib.util.spec_from_file_location("ma_two_hour_resume", DRIVER)
assert loader is not None and loader.loader is not None
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)

from sao_bridge.common import file_sha, read_json, write_json


def authorize(tmp_path):
    policy = {
        "resume_not_before_utc": "2026-10-09T10:34:57.344+00:00",
        "minimum_resubmission_interval_seconds": 7200,
    }
    path = tmp_path / "USER_PAUSE_AND_DELAY.json"
    write_json(path, policy)
    return policy, file_sha(path)


def test_exact_requested_first_resume_time_and_two_hour_retry_interval(tmp_path):
    policy, _ = authorize(tmp_path)
    assert driver.earliest_start(policy) == pytest.approx(1791542097.344, rel=0, abs=1e-6)
    interruption = 1791543000.0
    assert driver.earliest_start(policy, interruption) == interruption + 7200
    assert driver.earliest_start(policy, 0) == driver.earliest_start(policy)


def test_wait_never_returns_before_exact_not_before_timestamp(tmp_path, monkeypatch):
    _, digest = authorize(tmp_path)
    clock = [1000.0]
    waits = []
    monkeypatch.setattr(driver.time, "time", lambda: clock[0])

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(driver.time, "sleep", sleep)
    driver.wait_until(tmp_path, 8200.125, deadline=9000, authority_sha256=digest)
    assert clock[0] == 8200.125
    assert sum(waits) == 7200.125
    assert max(waits) == 60
    assert not read_json(tmp_path / "TWO_HOUR_MONITOR.json")["gpu_work_started"]


def test_deadline_can_stop_wait_without_any_early_training(tmp_path, monkeypatch):
    _, digest = authorize(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(driver.time, "time", lambda: clock[0])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(TimeoutError, match="deadline"):
        driver.wait_until(tmp_path, 7200, deadline=30, authority_sha256=digest)
    assert clock[0] == 30


@pytest.mark.parametrize("change", ["cancel", "changed_authority"])
def test_new_cancellation_or_authority_change_prevents_automatic_restart(tmp_path, change):
    _, digest = authorize(tmp_path)
    if change == "cancel":
        write_json(tmp_path / "TWO_HOUR_RESUME_CANCELLED.json", {"paused": True})
    else:
        write_json(tmp_path / "USER_PAUSE_AND_DELAY.json", {"changed": True})
    with pytest.raises((InterruptedError, ValueError)):
        driver.verify_authority(tmp_path, digest)


def test_admission_is_after_cooldown_not_after_thirty_minute_gate(tmp_path, monkeypatch):
    policy, digest = authorize(tmp_path)
    earliest = driver.earliest_start(policy)
    clock = [earliest - 7200]
    calls = []
    monkeypatch.setattr(driver.time, "time", lambda: clock[0])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def probe(root, phase, **kwargs):
        assert clock[0] >= earliest
        assert kwargs["stable_checks"] == 1 and kwargs["interval"] == 60
        assert kwargs["probe"] is driver.rolling.admission_snapshot
        calls.append(clock[0])

    monkeypatch.setattr(driver.program, "wait_for_resources", probe)
    driver.admit(tmp_path, tmp_path / "canary", authority_sha256=digest, deadline=earliest + 9000)
    assert calls == [earliest]


def test_stale_failure_record_cannot_authorize_a_new_preparation_error(tmp_path, monkeypatch):
    _, digest = authorize(tmp_path)
    previous = tmp_path / "old-attempt"
    write_json(previous / "FAILED.json", {"retry_after_idle_allowed": True})
    write_json(tmp_path / "ACTIVE_ATTEMPT.json", {"path": str(previous)})
    monkeypatch.setattr(driver, "admit", lambda *_, **__: None)

    def fail(*_, **__):
        raise ValueError("new metadata corruption")

    monkeypatch.setattr(driver.rolling, "run_attempt", fail)
    with pytest.raises(ValueError, match="metadata corruption"):
        driver.train(tmp_path, tmp_path, "full", authority_sha256=digest, deadline=1e12)


def test_verified_termination_persists_a_full_two_hour_cooldown(tmp_path, monkeypatch):
    _, digest = authorize(tmp_path)
    attempts = [0]
    admission = []
    monkeypatch.setattr(driver.time, "time", lambda: 10000.0)
    monkeypatch.setattr(driver, "admit", lambda *_, **__: admission.append(len(admission)))

    def run(*_, **__):
        attempts[0] += 1
        if attempts[0] == 1:
            path = tmp_path / "attempt-0001"
            write_json(tmp_path / "ACTIVE_ATTEMPT.json", {"path": str(path)})
            write_json(path / "FAILED.json", {
                "retry_after_idle_allowed": True, "evidence": [{"kind": "native_termination"}],
            })
            raise RuntimeError("verified native termination")
        assert read_json(tmp_path / "TWO_HOUR_RETRY_STATE.json")["next_attempt_not_before_unix"] == 17200.0
        return {"status": "completed"}

    monkeypatch.setattr(driver.rolling, "run_attempt", run)
    result = driver.train(tmp_path, tmp_path, "full", authority_sha256=digest, deadline=1e12)
    assert result["status"] == "completed" and len(admission) == 2
    assert (tmp_path / "attempt-0001/INTERRUPTED.md").is_file()


def test_persisted_retry_cooldown_survives_controller_reentry(tmp_path, monkeypatch):
    policy, digest = authorize(tmp_path)
    initial = driver.earliest_start(policy)
    interrupted = initial + 100
    write_json(tmp_path / "TWO_HOUR_RETRY_STATE.json", {"interrupted_at_unix": interrupted})
    deadlines = []
    monkeypatch.setattr(driver.time, "time", lambda: interrupted + 7200)
    monkeypatch.setattr(driver, "wait_until", lambda root, earliest, **kw: deadlines.append(earliest))
    monkeypatch.setattr(driver.program, "wait_for_resources", lambda *_, **__: None)
    driver.admit(tmp_path, tmp_path, authority_sha256=digest, deadline=initial + 20000)
    assert deadlines == [interrupted + 7200]


def test_partial_checkpoint_restarts_validation_not_formal_training(tmp_path, monkeypatch):
    _, digest = authorize(tmp_path)
    monkeypatch.setattr(driver, "admit", lambda *_, **__: None)
    monkeypatch.setattr(driver.campaign, "require_prerequisites", lambda *_, **__: None)
    monkeypatch.setattr(driver, "latest_checkpoint", lambda _: None)
    calls = []

    def train(root, directory, arm, **kwargs):
        assert directory == root / "rolling-canary" and arm == "full"
        calls.append(kwargs.get("planned_stop"))
        return {"status": "completed"}

    monkeypatch.setattr(driver, "train", train)
    monkeypatch.setattr(driver, "record_canary_gate", lambda *_, **__: {"status": "passed"})
    result = driver.execute(tmp_path, "canary", "full", authority_sha256=digest, deadline=1e12)
    assert result["status"] == "passed"
    assert calls == [12, None]
    assert not (tmp_path / "full/STARTED.json").exists()


def canary_fixture(tmp_path, monkeypatch):
    source_root = tmp_path / "frozen"
    write_json(source_root / "SOURCE_MANIFEST.json", {})
    monkeypatch.setattr(driver, "ROOT", source_root)
    write_json(tmp_path / "PINS.json", {})
    authorize(tmp_path)
    checkpoint = tmp_path / "rolling-canary/full/recovery/round-0015-attempt-0003/COMMITTED.json"
    write_json(checkpoint, {"completed_rounds": 15})
    monkeypatch.setattr(driver, "latest_checkpoint", lambda _: checkpoint)
    original = tmp_path / "source-after-real-restore"
    finalized = tmp_path / "finalization-only-attempt"
    for role in ("editor", "solver"):
        source = original / role / "optimizer"
        write_json(source / "RECOVERY_LOADED.json", {
            "completed_rounds": 12, "adam_reset": False, "warmup_repeated": False,
            "checkpoint_sha256": "same-committed-round12",
        })
        (source / "metrics.jsonl").write_text(
            "".join(json.dumps({"global_step": i}) + "\n" for i in range(15))
        )
        write_json(checkpoint.parent / f"{role}.json", {"source_run": str(source)})
        write_json(finalized / "full" / role / "COMPLETE.json", {"status": "completed"})
    costs = tmp_path / "rolling-canary/full/attempts/attempt-0001/COST.json"
    write_json(costs, {"new_persisted_outcomes": 96, "unpersisted_inflight_work_unknown": True})
    return {
        "status": "completed", "attempt": str(finalized),
        "audit": {"status": "passed", "shared_environment_episodes": 120},
    }, original, costs


def test_final_checkpoint_publication_still_requires_real_prior_native_restore(tmp_path, monkeypatch):
    result, _, costs = canary_fixture(tmp_path, monkeypatch)
    gate = driver.record_canary_gate(tmp_path, result)
    assert gate["status"] == "passed" and gate["logical_shared_episodes"] == 120
    assert gate["all_attempt_costs"][str(costs)]["new_persisted_outcomes"] == 96
    assert not gate["formal_weights_reused"]
    assert read_json(tmp_path / "ROLLING_NATIVE_GATE.json")["status"] == "passed"


@pytest.mark.parametrize("corruption", ["different_checkpoint", "adam_reset", "repeated_warmup", "wrong_cursor"])
def test_canary_gate_rejects_unverified_restore(tmp_path, monkeypatch, corruption):
    result, original, _ = canary_fixture(tmp_path, monkeypatch)
    loaded_path = original / "solver/optimizer/RECOVERY_LOADED.json"
    loaded = read_json(loaded_path)
    key, value = {
        "different_checkpoint": ("checkpoint_sha256", "a-different-round12"),
        "adam_reset": ("adam_reset", True),
        "repeated_warmup": ("warmup_repeated", True),
        "wrong_cursor": ("completed_rounds", 0),
    }[corruption]
    loaded[key] = value
    write_json(loaded_path, loaded)
    with pytest.raises(ValueError):
        driver.record_canary_gate(tmp_path, result)
    assert not (tmp_path / "ROLLING_NATIVE_GATE.json").exists()


@pytest.mark.parametrize("corruption", ["wrong_budget", "failed_audit", "repeated_round", "incomplete_pair"])
def test_canary_gate_rejects_incomplete_trajectory_or_checkpoint_evidence(tmp_path, monkeypatch, corruption):
    result, original, _ = canary_fixture(tmp_path, monkeypatch)
    if corruption == "wrong_budget":
        result["audit"]["shared_environment_episodes"] = 119
    elif corruption == "failed_audit":
        result["audit"]["status"] = "failed"
    elif corruption == "repeated_round":
        (original / "solver/optimizer/metrics.jsonl").write_text(
            "".join(json.dumps({"global_step": i}) + "\n" for i in [*range(14), 13])
        )
    else:
        write_json(driver.latest_checkpoint(tmp_path), {"completed_rounds": 12})
    with pytest.raises(ValueError):
        driver.record_canary_gate(tmp_path, result)
    assert not (tmp_path / "ROLLING_NATIVE_GATE.json").exists()
