import importlib.util
from pathlib import Path
import signal
import time

import pytest


SCRIPT = (Path(__file__).resolve().parents[1]
          / "runs/alfworld_ma_mechanisms_single_seed_v2/recover_v2_attempt.py")
spec = importlib.util.spec_from_file_location("ma_recovery_v2", SCRIPT)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def original(tmp_path, monkeypatch):
    old = tmp_path / "v2"
    old.mkdir()
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    recovery.write_json(frozen / "SOURCE_MANIFEST.json", {"source": "frozen"})
    monkeypatch.setattr(recovery, "ROOT", frozen)
    source = frozen / "runtime.py"
    source.write_text("unchanged = True\n")
    common = {
        old / "PROTOCOL.json": {
            "root": str(old), "source": str(frozen), "study_revision": 2,
            "arms": list(recovery.study.ARMS), "training_seed": 1000043,
            "unit_suffix": "single-seed-v2",
        },
        old / "INITIAL_STATE.json": {"actor": "/models/initial", "critic": "/models/raw-critic"},
        old / "schedule_source_data.json": [{"task_id": 7}],
        old / "joint_data.json": [{"task_id": 7, "message": "unchanged"}],
    }
    for path, value in common.items():
        recovery.write_json(path, value)
    common_pins = recovery.study.file_manifest([*common, source])
    paths = list(common)
    for arm in recovery.study.ARMS:
        for role in ("editor", "solver"):
            path = old / arm / f"{role}_spec.json"
            recovery.write_json(path, {
                "run": str(old / arm / role / "optimizer"),
                "joint_root": str(old / arm / "joint"), "data": str(old / "joint_data.json"),
                "model": "/models/initial", "critic_model": "/models/raw-critic",
                "steps": 144, "seed": 1000043, "lr": 5e-7,
                "peer_weight_mode": "unit" if arm == "without_peer" else "bounded",
                "critic_cooperative_context": arm != "without_context",
                "value_normalization": arm != "without_value_norm",
                "editor_update_every_round": arm == "always_joint", "pinned_files": common_pins,
            })
            paths.append(path)
    recovery.write_json(old / "PINS.json", {"pins": recovery.study.file_manifest([*paths, source])})
    recovery.write_json(old / "CODE_VALIDATION.json", {
        "status": "passed", "pins_sha256": recovery.file_sha(old / "PINS.json"),
        "source_manifest_sha256": recovery.file_sha(frozen / "SOURCE_MANIFEST.json"),
    })
    memory_cases = {}
    for i in range(6):
        path = old / "memory" / str(i) / "result.json"
        recovery.write_json(path, {"status": "passed"})
        audit = path.parent / "audit-summary.json"
        recovery.write_json(audit, {"status": "passed"})
        memory_cases[str(i)] = {
            "result": str(path), "sha256": recovery.file_sha(path),
            "audit_sha256": recovery.file_sha(audit),
        }
    recovery.write_json(old / "MEMORY_GATE_INHERITANCE.json", {
        "status": "passed", "cases": memory_cases,
    })
    cases = {}
    for arm in ("full", "always_joint"):
        path = old / "schedule-calibration" / arm / "TRAINING_AUDIT.json"
        recovery.write_json(path, {
            "status": "passed", **recovery.study.arm_budgets(arm, rounds=6),
        })
        cases[arm] = {"audit": str(path), "sha256": recovery.file_sha(path)}
    recovery.write_json(old / "NATIVE_GATE.json", {
        "status": "passed", "pins_sha256": recovery.file_sha(old / "PINS.json"),
        "memory_gate_inheritance_sha256": recovery.file_sha(old / "MEMORY_GATE_INHERITANCE.json"),
        "real_schedule_cases": cases, "calibration_environment_episodes": 96,
    })
    recovery.write_json(old / "PROGRAM_STARTED.json", {"deadline_unix": time.time() + 3600})
    recovery.write_json(old / "PROGRAM_FAILED.json", {
        "status": "incomplete", "phase": "full", "automatic_retry": False,
    })
    recovery.write_json(old / "full/INTERRUPTED.json", {
        "fixed_endpoint_completed": False, "main200_scores_available": False,
        "formal_seed_unchanged": 1000043,
        "failure_sha256": recovery.file_sha(old / "PROGRAM_FAILED.json"),
        "recorded_formal_receipts": {"editor": 0, "solver": 0, "outcomes": 0},
        "roles": {
            role: {"committed_rounds": 0, "metrics_present": False,
                   "complete_receipt_present": False, "saved_hf_checkpoints": []}
            for role in ("editor", "solver")
        },
    })
    recovery.write_json(old / "full/SIGNAL_PROVENANCE.json", {"unresolved_local_sender": True})
    (old / "full/INTERRUPTED.md").write_text("Startup interrupted, not a performance result.\n")
    recovery.write_json(old / "full/INTERRUPTION_REPORT_AUDIT.json", {
        "status": "passed",
        **{key: recovery.file_sha(old / relative) for key, relative in (
            ("interruption_sha256", "full/INTERRUPTED.json"),
            ("provenance_sha256", "full/SIGNAL_PROVENANCE.json"),
            ("report_sha256", "full/INTERRUPTED.md"),
            ("native_gate_sha256", "NATIVE_GATE.json"),
        )},
    })
    (old / "launches").mkdir()
    (old / "launches/full-editor-signals.log").write_text("SIGTERM from external sender\n")
    for name in ("SPEED_POLICY", "PRIOR_RESULTS_REPORTED", "PRIOR_WORK"):
        recovery.write_json(old / f"{name}.json", {"retained": name})
    recovery.write_json(old / "post-interruption-watch/RESOURCE_WINDOW.json", {
        "resources": {"released": True}, "automatic_training_restart": False,
    })
    recovery.write_json(old / "PROGRAM_PREPARED.json", {
        "status": "prepared", "pins": recovery.study.file_manifest([old / "MEMORY_GATE_INHERITANCE.json"]),
        "resource_wait_limit_hours": 12, "overall_wall_limit_hours": 40,
    })
    log = old / "recovery-tests.log"
    log.write_text("fixture passing tests\n")
    validation = old / "RECOVERY_CODE_VALIDATION.json"
    monkeypatch.setattr(recovery, "VALIDATION", validation)
    recovery.write_json(validation, {
        "status": "passed", "driver_sha256": recovery.file_sha(recovery.SCRIPT),
        "location_helper_sha256": recovery.file_sha(recovery.HELPER),
        "source_manifest_sha256": recovery.file_sha(frozen / "SOURCE_MANIFEST.json"),
        "tests": str(Path(__file__)), "tests_sha256": recovery.file_sha(Path(__file__)),
        "test_log": str(log), "test_log_sha256": recovery.file_sha(log),
    })

    def historical(root):
        value = {"status": "passed", "pins": recovery.study.file_manifest([root / "PROTOCOL.json"])}
        recovery.write_json(root / "HISTORICAL_PATH_AUDIT.json", value)
        return value

    monkeypatch.setattr(recovery.program, "audit_historical_path", historical)
    return old


def prepared_root(original):
    new = original.parent / (original.name + "_attempt02")
    recovery.prepare(original, new)
    return new


def test_preparation_preserves_science_gates_deadline_and_failed_attempt(original):
    before = recovery.study.file_manifest([
        original / "PROGRAM_STARTED.json", original / "PROGRAM_FAILED.json",
        original / "full/STARTED.json" if (original / "full/STARTED.json").exists()
        else original / "full/INTERRUPTED.json", original / "NATIVE_GATE.json",
    ])
    new = prepared_root(original)
    record = recovery.read_json(new / "RECOVERY.json")
    assert record["original_absolute_deadline_unix"] == recovery.read_json(
        original / "PROGRAM_STARTED.json")["deadline_unix"]
    assert record["new_native_episodes"] == 0
    assert record["calibration_episodes_retained"] == 96
    assert record["score_driven_retry"] is False
    assert record["lifecycle_cause_claimed_fixed"] is False
    assert len(record["scientific_inputs"]) == 14
    assert recovery.read_json(new / "PROTOCOL.json")["unit_suffix"] == "single-seed-v2-attempt02"
    assert not (new / "PROGRAM_STARTED.json").exists()
    assert not (new / "schedule-calibration").exists()
    for arm in recovery.study.ARMS:
        for role in ("editor", "solver"):
            old_spec = recovery.read_json(original / arm / f"{role}_spec.json")
            new_spec = recovery.read_json(new / arm / f"{role}_spec.json")
            assert recovery.helpers.canonical(old_spec, original) == recovery.helpers.canonical(new_spec, new)
    assert recovery.study.file_manifest(before) == before
    assert recovery.read_json(new / "NATIVE_GATE.json")["real_schedule_cases"] == recovery.read_json(
        original / "NATIVE_GATE.json")["real_schedule_cases"]
    assert recovery.read_json(new / "PRIOR_WORK.json")["additional_interrupted_attempts"]
    recovery.verify_recovery(new)


@pytest.mark.parametrize("kind", ["progress", "metrics", "native", "provenance", "source", "deadline"])
def test_invalid_inheritance_is_rejected_before_creating_attempt(original, kind):
    if kind == "progress":
        path = original / "full/INTERRUPTED.json"
        value = recovery.read_json(path)
        value["roles"]["solver"]["committed_rounds"] = 1
        recovery.write_json(path, value)
    elif kind == "metrics":
        path = original / "full/solver/optimizer/metrics.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("{}\n")
    elif kind == "native":
        path = original / "schedule-calibration/full/TRAINING_AUDIT.json"
        recovery.write_json(path, {"status": "failed"})
    elif kind == "provenance":
        recovery.write_json(original / "full/SIGNAL_PROVENANCE.json", {"modified": True})
    elif kind == "source":
        (original.parent / "frozen/runtime.py").write_text("changed = True\n")
    else:
        recovery.write_json(original / "PROGRAM_STARTED.json", {"deadline_unix": time.time() - 1})
    new = original.parent / (original.name + "_attempt02")
    with pytest.raises((ValueError, TimeoutError)):
        recovery.prepare(original, new)
    assert not new.exists()


def test_cannot_reprepare_or_clear_original_failure(original):
    new = prepared_root(original)
    with pytest.raises(ValueError, match="unused independent"):
        recovery.prepare(original, new)
    with pytest.raises(ValueError, match="unused independent"):
        recovery.prepare(original, original)
    assert (original / "PROGRAM_FAILED.json").exists()


def test_changed_external_pin_cannot_be_hidden_by_canonicalization(original):
    (original.parent / "frozen/runtime.py").write_text("changed = True\n")
    new = original.parent / "v2_attempt02"
    with pytest.raises(ValueError, match="external scientific pin changed"):
        recovery.relocate_inputs(original, new)


def test_recovery_rejects_changed_inherited_evidence(original):
    new = prepared_root(original)
    recovery.write_json(original / "full/SIGNAL_PROVENANCE.json", {"changed_after_preparation": True})
    with pytest.raises(ValueError, match="inherited evidence changed"):
        recovery.verify_recovery(new)


def test_control_runs_only_full_without_repeating_native_or_releasing_ablations(original, monkeypatch):
    new = prepared_root(original)
    events = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    monkeypatch.setattr(recovery.program, "wait_for_resources",
                        lambda *args, **kwargs: events.append("wait") or {"released": True})
    monkeypatch.setattr(recovery.campaign, "run_arm", lambda root, arm: events.append(arm))
    monkeypatch.setattr(recovery.program, "audit_full_result", lambda root: {"full_accuracy": .9})
    monkeypatch.setattr(recovery.program, "write_master_report", lambda root: events.append("report"))
    monkeypatch.setattr(recovery.campaign, "train_joint",
                        lambda *args, **kwargs: pytest.fail("must not repeat native calibration"))
    result = recovery.run_control(new)
    assert events == ["wait", "full", "report"]
    assert result["full_accuracy"] == .9
    assert result["ablations_automatically_released"] is False
    assert recovery.read_json(new / "PROGRAM_STARTED.json")["deadline_unix"] == recovery.read_json(
        original / "PROGRAM_STARTED.json")["deadline_unix"]
    assert {sig: signal.getsignal(sig) for sig in handlers} == handlers
    with pytest.raises(ValueError):
        recovery.run_control(new)
    assert events == ["wait", "full", "report"]


@pytest.mark.parametrize("during", ["wait", "full"])
def test_failure_is_persistent_and_never_retried(original, monkeypatch, during):
    new = prepared_root(original)
    events = []

    def step(name):
        events.append(name)
        if name == during:
            raise InterruptedError("external signal15")
        return {"released": True}

    monkeypatch.setattr(recovery.program, "wait_for_resources", lambda *a, **k: step("wait"))
    monkeypatch.setattr(recovery.campaign, "run_arm", lambda *a, **k: step("full"))
    with pytest.raises(InterruptedError):
        recovery.run_control(new)
    failure = recovery.read_json(new / "PROGRAM_FAILED.json")
    assert failure["automatic_retry"] is False
    assert failure["phase"] == ("full" if during == "full" else "recovery-resource-admission")
    before = list(events)
    with pytest.raises(ValueError, match="no automatic retry"):
        recovery.run_control(new)
    assert events == before


def test_ablations_still_require_reported_baseline(original):
    new = prepared_root(original)
    with pytest.raises(FileNotFoundError):
        recovery.campaign.require_full_control_review(new)
    assert not (new / "ABLATIONS_PROGRAM_STARTED.json").exists()
