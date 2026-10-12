import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
SCRIPT = ROOT / "runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/resume_after_commit_fix.py"
loader = importlib.util.spec_from_file_location("checkpoint_fix_resume", SCRIPT)
assert loader is not None and loader.loader is not None
driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(driver)

from sao_bridge.common import file_sha, read_json, write_json


def test_only_an_exact_recorded_nonzero_exit_anchors_the_new_cooldown():
    value = 1791615153.772817
    assert driver.verified_exit_time(f"{value} +++ exited with 1 +++\n") == value
    policy = {"resume_not_before_utc": "2026-10-09T10:34:57.344+00:00"}
    assert driver.policy_driver.earliest_start(policy, value) == value + 7200
    for text in ("", "1791615153.77 +++ exited with 0 +++", "killed", f"{value} +++ exited with 1 +++\n" * 2):
        with pytest.raises(ValueError, match="one recorded"):
            driver.verified_exit_time(text)


@pytest.mark.parametrize("change", ["checkpoint-only", "learner", "file-after-freeze"])
def test_execution_amendment_accepts_only_the_frozen_checkpoint_module(tmp_path, monkeypatch, change):
    old, new = tmp_path / "old", tmp_path / "new"
    checkpoint = "whale_local/sao_bridge/joint_recovery.py"
    learner = "whale_local/sao_bridge/large_train.py"
    for source in (old, new):
        (source / checkpoint).parent.mkdir(parents=True)
        (source / checkpoint).write_text("old" if source == old else "fixed")
        (source / learner).write_text("learner")
    if change == "learner":
        (new / learner).write_text("changed learner")
    old_files = {path: file_sha(old / path) for path in (checkpoint, learner)}
    write_json(old / "SOURCE_MANIFEST.json", {"runtime_files": old_files})
    write_json(new / "SOURCE_MANIFEST.json", {
        "base_snapshot": str(old), "base_manifest_sha256": file_sha(old / "SOURCE_MANIFEST.json"),
        "runtime_files": {path: file_sha(new / path) for path in (checkpoint, learner)},
    })
    prepared = {
        "original_source": str(old), "execution_source": str(new),
        "original_manifest_sha256": file_sha(old / "SOURCE_MANIFEST.json"),
        "execution_manifest_sha256": file_sha(new / "SOURCE_MANIFEST.json"),
    }
    monkeypatch.setattr(driver, "ROOT", new)
    if change == "file-after-freeze":
        (new / checkpoint).write_text("modified after validation")
    if change == "checkpoint-only":
        driver.verify_source(prepared)
    else:
        with pytest.raises(ValueError):
            driver.verify_source(prepared)


@pytest.mark.parametrize("failure", ["cooldown", "resources", "prerequisites", "expired"])
def test_no_native_work_precedes_cooldown_admission_and_original_prerequisites(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(driver, "RECOVERY", tmp_path)
    monkeypatch.setattr(driver, "RUN", tmp_path)
    monkeypatch.setattr(driver, "verify_bindings", lambda prepared: None)
    monkeypatch.setattr(driver.time, "time", lambda: 500 if failure == "expired" else 100)
    calls = []

    def stage(name):
        def execute(*args, **kwargs):
            calls.append(name)
            if failure == name:
                raise RuntimeError(name)
            return {"status": "passed"}
        return execute

    monkeypatch.setattr(driver.policy_driver, "wait_until", stage("cooldown"))
    monkeypatch.setattr(driver.program, "wait_for_resources", stage("resources"))
    monkeypatch.setattr(driver, "original_prerequisites", stage("prerequisites"))
    monkeypatch.setattr(driver.rolling, "run_attempt", stage("native"))
    with pytest.raises((RuntimeError, TimeoutError)):
        driver.execute({"not_before_unix": 100, "deadline_unix": 200, "user_authority_sha256": "authority"})
    assert "native" not in calls


def test_prerequisite_failure_keeps_the_error_log_and_does_not_change_source(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "RECOVERY", tmp_path)
    monkeypatch.setattr(driver, "RUN", tmp_path)
    seen = {}

    def failed(command, **kwargs):
        seen.update(kwargs)
        return SimpleNamespace(returncode=1, stdout="Frozen input changed\n")

    monkeypatch.setattr(driver.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="Original native prerequisites failed"):
        driver.original_prerequisites({"original_source": str(tmp_path / "original")})
    assert (tmp_path / "ORIGINAL_PREREQUISITES.log").read_text() == "Frozen input changed\n"
    assert seen["env"]["PYTHONPATH"].startswith(str(tmp_path / "original/whale_local"))
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
