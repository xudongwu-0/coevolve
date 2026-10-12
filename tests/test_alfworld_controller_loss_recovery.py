import asyncio
import copy
import importlib.util
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import evaluation
from sao_bridge.alfworld_ma_mechanism_program import verify_executed_role_identity
from sao_bridge.alfworld_solver_eval import reuse_completed_reports
from sao_bridge.common import file_sha, read_json, write_json


def recovery_fixture(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    config = {
        "run": str(old), "solver_model": "frozen-model", "solver_version": "fixed144",
        "no_patch": "unchanged-empty-patch", "harnesses": {"candidate": "fixed-patch"},
        "task_ids": [240, 241], "samples": 1, "seed": 17, "solver_tokens": 4096,
        "temperature": 1.0, "episode_checkpointing": True,
    }
    write_json(old / "config.json", config)
    write_json(old / "RELEASED.json", {
        "remaining_gpu_pids": [], "foreign_gpu_pids": [], "verified_filler_pids": [],
        "vanished_gpu_pids": [], "filler_kind": None, "filler_unit_active": True,
    })
    audit = {
        "status": "observed_controller_loss", "previous_attempt": str(old),
        "previous_config_sha256": file_sha(old / "config.json"),
        "controller_alive": False, "evaluator_alive": False, "controller_exit_code": 1,
        "release_sha256": file_sha(old / "RELEASED.json"),
        "repeat_completed_episodes": False, "signal_and_sender": "unknown",
    }
    audit_path = tmp_path / "INTERRUPTION.json"
    write_json(audit_path, audit)
    manifest = {
        "previous_attempt": str(old), "previous_config_sha256": file_sha(old / "config.json"),
        "reason": "observed-controller-loss", "interruption_audit": str(audit_path),
        "interruption_audit_sha256": file_sha(audit_path), "complete_reports": {},
        "episode_journal_files": {},
    }
    manifest_path = tmp_path / "RESUME.json"
    write_json(manifest_path, manifest)
    return old, new, {**config, "run": str(new), "resume_matrix": str(manifest_path)}, manifest


def test_observed_process_loss_does_not_fabricate_a_transport_failure(tmp_path):
    old, new, config, _ = recovery_fixture(tmp_path)
    assert reuse_completed_reports(config, new, [{"harness_sha256": "same"}]) == {}
    assert not (old / "FAILED.json").exists()
    assert (new / "REUSED_REPORTS.json").exists()


@pytest.mark.parametrize("field,value", [
    ("controller_alive", True), ("evaluator_alive", True), ("controller_exit_code", 0),
    ("controller_exit_code", None), ("repeat_completed_episodes", True),
    ("previous_attempt", "/wrong/attempt"), ("release_sha256", "wrong"),
])
def test_process_loss_recovery_rejects_unverified_exit_audits(tmp_path, field, value):
    _, new, config, manifest = recovery_fixture(tmp_path)
    audit_path = Path(manifest["interruption_audit"])
    audit = read_json(audit_path)
    audit[field] = value
    write_json(audit_path, audit)
    manifest["interruption_audit_sha256"] = file_sha(audit_path)
    write_json(Path(config["resume_matrix"]), manifest)
    with pytest.raises(ValueError, match="post-exit"):
        reuse_completed_reports(config, new, [{"harness_sha256": "same"}])


@pytest.mark.parametrize("terminal", ["FAILED.json", "COMPLETE.json", "report.json"])
def test_process_loss_authority_cannot_replay_completed_or_known_failed_work(tmp_path, terminal):
    old, new, config, _ = recovery_fixture(tmp_path)
    write_json(old / terminal, {"error": "not an unrecorded process loss"})
    with pytest.raises(ValueError, match="post-exit"):
        reuse_completed_reports(config, new, [{"harness_sha256": "same"}])


def test_process_loss_recovery_preserves_scientific_inputs(tmp_path):
    _, new, config, _ = recovery_fixture(tmp_path)
    config["seed"] += 1
    with pytest.raises(ValueError, match="inputs"):
        reuse_completed_reports(config, new, [{"harness_sha256": "same"}])


def test_existing_read_error_recovery_remains_supported(tmp_path):
    old, new, config, manifest = recovery_fixture(tmp_path)
    write_json(old / "FAILED.json", {"error": "ReadError('connection lost')"})
    manifest.update(
        reason="inference-http-read-error", previous_failure_sha256=file_sha(old / "FAILED.json"),
    )
    write_json(Path(config["resume_matrix"]), manifest)
    assert reuse_completed_reports(config, new, [{"harness_sha256": "same"}]) == {}


def test_only_unfinished_episode_is_executed_after_verified_controller_loss(tmp_path, monkeypatch):
    old, new, config, manifest = recovery_fixture(tmp_path)
    rows = [{"id": f"alfworld-{task}", "task_id": task, "task_name": "fixed-name"} for task in (240, 241)]

    class Domain:
        def __init__(self, interrupt):
            self.interrupt = interrupt
            self.calls = []

        def harness_spec(self, source):
            return {}

        async def run_episode(self, row, spec, sampler, **kwargs):
            self.calls.append(row["id"])
            if self.interrupt and row["task_id"] == 241:
                raise RuntimeError("interrupted")
            return {
                "id": row["id"], "task_id": row["task_id"], "reward": int(not self.interrupt),
                "tokens": 1, "tool_calls": 0, "tool_errors": 0, "truncated": False, "boxed": True,
            }

    first = Domain(True)
    monkeypatch.setattr(evaluation, "get_domain", lambda _: first)
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(evaluation.evaluate(
            None, rows, "fixed-source", seed=17, domain="alfworld", episode_concurrency=1,
            episode_journal=old / "episode_journals/same",
        ))
    cached = old / "episode_journals/same/episode-000000.json"
    digest = file_sha(cached)
    manifest["episode_journal_files"] = {
        str(path.relative_to(old / "episode_journals")): file_sha(path)
        for path in (old / "episode_journals").glob("*/*.json")
    }
    write_json(Path(config["resume_matrix"]), manifest)
    reuse_completed_reports(config, new, [{"harness_sha256": "same"}])
    resumed = Domain(False)
    monkeypatch.setattr(evaluation, "get_domain", lambda _: resumed)
    report = asyncio.run(evaluation.evaluate(
        None, rows, "fixed-source", seed=17, domain="alfworld", episode_concurrency=1,
        episode_journal=new / "episode_journals/same",
    ))
    assert resumed.calls == ["alfworld-241"]
    assert report["accuracy"] == 0.5
    assert file_sha(new / "episode_journals/same/episode-000000.json") == digest
    assert len(read_json(new / "episode_journals/same/episode-000001.attempts.json")) == 2


def role_fixture(tmp_path):
    role = tmp_path / "full/attempts/attempt-0001/full/editor"
    spec = {
        "run": str(tmp_path / "full/editor/optimizer"), "joint_root": str(tmp_path / "full/joint"),
        "steps": 144, "seed": 1000043, "lr": 5e-7, "preserve_optimizer_state": False,
        "rolling_recovery": {
            "root": str(tmp_path / "full/recovery"), "interval_rounds": 12,
            "attempt": "attempt-0001", "resume": None, "hf_rounds": [96, 144],
        },
    }
    write_json(tmp_path / "full/editor_spec.json", spec)
    actual = copy.deepcopy(spec)
    actual.update(run=str(role / "optimizer"), joint_root=str(role.parent / "joint"), recovery_start_round=0)
    actual["rolling_recovery"]["finalize_only"] = False
    write_json(role.parent / "editor_spec.json", actual)
    write_json(role / "IDENTITY.json", actual)
    write_json(role.parent.parent / "PREPARED.json", {
        "status": "prepared", "arm": "full", "resume_round": 0, "recovery": None,
    })
    (tmp_path / "full/editor").symlink_to(role, target_is_directory=True)
    return role, actual


def test_rolling_identity_audit_allows_only_the_declared_attempt_relocation(tmp_path):
    role_fixture(tmp_path)
    verify_executed_role_identity(tmp_path, "full", "editor")


@pytest.mark.parametrize("field,value", [("lr", 1e-4), ("seed", 9), ("steps", 96)])
def test_rolling_identity_audit_rejects_real_scientific_changes(tmp_path, field, value):
    role, actual = role_fixture(tmp_path)
    actual[field] = value
    write_json(role / "IDENTITY.json", actual)
    write_json(role.parent / "editor_spec.json", actual)
    with pytest.raises(ValueError, match="frozen"):
        verify_executed_role_identity(tmp_path, "full", "editor")


def test_legacy_identity_audit_still_requires_exact_specification(tmp_path):
    spec = {"seed": 1000043, "lr": 5e-7}
    write_json(tmp_path / "full/editor_spec.json", spec)
    write_json(tmp_path / "full/editor/IDENTITY.json", spec)
    verify_executed_role_identity(tmp_path, "full", "editor")
    write_json(tmp_path / "full/editor/IDENTITY.json", {**spec, "seed": 8})
    with pytest.raises(ValueError, match="frozen"):
        verify_executed_role_identity(tmp_path, "full", "editor")


@pytest.fixture
def driver():
    path = (
        Path(__file__).resolve().parents[1]
        / "runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/resume_full_evaluation.py"
    )
    loader = importlib.util.spec_from_file_location("ma_full_evaluation_resume", path)
    assert loader is not None and loader.loader is not None
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


def test_evaluation_only_resumption_obeys_exact_two_hour_not_before(tmp_path, monkeypatch, driver):
    clock = [1000.0]
    probes = []
    monkeypatch.setattr(driver, "RECOVERY", tmp_path)
    monkeypatch.setattr(driver, "verify_policy", lambda _: None)
    monkeypatch.setattr(driver.time, "time", lambda: clock[0])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))

    def admit(*_, **kwargs):
        assert clock[0] == 8200.125
        assert kwargs["probe"] is driver.rolling.admission_snapshot
        probes.append(clock[0])

    monkeypatch.setattr(driver.program, "wait_for_resources", admit)
    driver.wait_for_window({"not_before_unix": 8200.125, "deadline_unix": 9000})
    assert probes == [8200.125]
    assert not read_json(tmp_path / "MONITOR.json")["gpu_work_started"]


def test_evaluation_wait_does_not_extend_original_deadline(tmp_path, monkeypatch, driver):
    clock = [0.0]
    monkeypatch.setattr(driver, "RECOVERY", tmp_path)
    monkeypatch.setattr(driver, "verify_policy", lambda _: None)
    monkeypatch.setattr(driver.time, "time", lambda: clock[0])
    monkeypatch.setattr(driver.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    with pytest.raises(TimeoutError, match="deadline"):
        driver.wait_for_window({"not_before_unix": 7200, "deadline_unix": 30})
    assert clock[0] == 30


def test_recovery_configuration_changes_only_run_and_resume_manifest(tmp_path, monkeypatch, driver):
    old, _, config, _ = recovery_fixture(tmp_path)
    monkeypatch.setattr(driver, "RUN", tmp_path)
    monkeypatch.setattr(driver, "RECOVERY", tmp_path / "recovery")
    current = driver.selection_config({"previous_selection_attempt": str(old)})
    assert {k: v for k, v in current.items() if k not in {"run", "resume_matrix"}} == {
        k: v for k, v in config.items() if k not in {"run", "resume_matrix"}
    }
    assert current["run"] == str(tmp_path / "full/round0144/selection/attempt-02")


@pytest.mark.parametrize("corruption", ["none", "changed_completed_result", "repeated_attempt"])
def test_physical_evaluation_cost_accounts_for_partial_work(tmp_path, monkeypatch, driver, corruption):
    previous, resumed = tmp_path / "old", tmp_path / "new"
    monkeypatch.setattr(driver, "RECOVERY", tmp_path / "recovery")
    monkeypatch.setattr(driver, "ROOT", tmp_path / "source")
    for name in ("INTERRUPTION_AUDIT.json", "RESUME_MATRIX.json"):
        write_json(driver.RECOVERY / name, {})
    write_json(driver.ROOT / "SOURCE_MANIFEST.json", {})
    for root in (previous, resumed):
        write_json(root / "episode_journals/a/episode-000000.json", {"fixed_result": 0})
        write_json(root / "episode_journals/a/episode-000000.attempts.json", [{"started": 1}])
        write_json(root / "episode_journals/a/episode-000001.attempts.json", [{"started": 1}])
    write_json(resumed / "episode_journals/a/episode-000001.attempts.json", [{"started": 1}, {"started": 2}])
    write_json(resumed / "report.json", {"unique_harnesses": 1})
    if corruption == "changed_completed_result":
        write_json(resumed / "episode_journals/a/episode-000000.json", {"fixed_result": 1})
    elif corruption == "repeated_attempt":
        write_json(resumed / "episode_journals/a/episode-000000.attempts.json", [{"started": 1}, {"started": 2}])
    arguments = (
        {"previous_selection_attempt": str(previous)}, {"run": str(resumed), "task_ids": [240, 241], "samples": 1},
    )
    if corruption != "none":
        with pytest.raises(ValueError):
            driver.recovery_costs(*arguments)
    else:
        costs = driver.recovery_costs(*arguments)
        assert costs["selection_logical_episodes"] == 2
        assert costs["selection_physical_episode_starts"] == 3
        assert costs["previous_unfinished_episodes"] == 1
        assert costs["reused_completed_episodes"] == 1
        assert costs["repeated_completed_episodes"] == 0
