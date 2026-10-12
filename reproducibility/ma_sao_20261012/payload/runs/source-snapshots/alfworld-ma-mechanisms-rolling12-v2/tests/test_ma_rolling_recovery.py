from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from sao_bridge.alfworld_ma_rolling_recovery import (
    admission_snapshot, attempt_cost, relocate, termination_evidence, wait_idle,
)
from sao_bridge.common import write_json


def test_only_owned_root_paths_are_relocated():
    old, new = Path("/study/old"), Path("/study/new")
    value = {
        "/study/old/data.json": ["/study/old", "/study/old/full/run", "/study/older/reference"],
        "model": "/models/initial", "seed": 1000043,
    }
    assert relocate(value, old, new) == {
        "/study/new/data.json": ["/study/new", "/study/new/full/run", "/study/older/reference"],
        "model": "/models/initial", "seed": 1000043,
    }


@pytest.mark.parametrize("code", [-signal.SIGTERM, -signal.SIGKILL, 137, 143])
def test_verified_role_termination_can_requeue(tmp_path, code):
    write_json(tmp_path / "full/PROCESS_EXIT_FAILURE.json", {
        "processes": [{"pid": 123, "returncode": code}],
    })
    assert termination_evidence(tmp_path, "full")[0]["kind"] == "role_termination"


def test_unexplained_python_failure_is_not_a_success_shaped_retry(tmp_path):
    write_json(tmp_path / "full/PROCESS_EXIT_FAILURE.json", {
        "processes": [{"pid": 123, "returncode": 1}],
    })
    write_json(tmp_path / "full/solver/NATIVE_EXIT.json", {"returncode": 1})
    assert termination_evidence(tmp_path, "full") == []
    write_json(tmp_path / "full/solver/NATIVE_EXIT.json", {"returncode": -signal.SIGTERM})
    assert termination_evidence(tmp_path, "full")[0]["kind"] == "native_termination"


def test_remote_role_contention_has_explicit_retry_evidence(tmp_path):
    write_json(tmp_path / "full/editor/FAILED.json", {
        "native_returncode": None, "foreign_workload": {"processes": [{"pid": 321}]},
    })
    assert termination_evidence(tmp_path, "full")[0]["kind"] == "role_yielded_to_foreign_workload"


def test_zero_gpu_memory_does_not_admit_an_active_distributed_launcher(monkeypatch):
    import sao_bridge.alfworld_ma_rolling_recovery as rolling

    monkeypatch.setattr(rolling.program, "resource_snapshot", lambda: {"released": True})
    monkeypatch.setattr(rolling, "distributed_launchers", lambda: [{"pid": 321, "module": "torch.distributed.run"}])
    monkeypatch.setattr(rolling.subprocess, "run", lambda *_, **__: SimpleNamespace(stdout="[]"))
    result = admission_snapshot()
    assert not result["released"]
    assert result["distributed_launchers"]["node4"][0]["pid"] == 321


def test_thirty_minute_gate_only_applies_to_retries(tmp_path, monkeypatch):
    import sao_bridge.alfworld_ma_rolling_recovery as rolling

    write_json(tmp_path / "PROTOCOL.json", {"original_absolute_deadline_unix": 1e12})
    calls = []
    monkeypatch.setattr(rolling.program, "wait_for_resources", lambda *args, **kwargs: calls.append(kwargs))
    wait_idle(tmp_path, retry=False)
    wait_idle(tmp_path, retry=True)
    assert calls[0]["stable_checks"] == 1
    assert calls[1]["stable_checks"] == 31 and calls[1]["interval"] == 60
    assert (calls[1]["stable_checks"] - 1) * calls[1]["interval"] == 1800
    assert calls[1]["probe"] is rolling.admission_snapshot


def test_a_new_preparation_error_cannot_reuse_old_termination_authority(tmp_path, monkeypatch):
    import sao_bridge.alfworld_ma_rolling_recovery as rolling

    previous = tmp_path / "old-attempt"
    write_json(previous / "FAILED.json", {"retry_after_idle_allowed": True})
    write_json(tmp_path / "ACTIVE_ATTEMPT.json", {"path": str(previous)})
    monkeypatch.setattr(rolling, "wait_idle", lambda *_, **__: None)

    def fail(*_, **__):
        raise ValueError("New recovery metadata is corrupt")

    monkeypatch.setattr(rolling, "run_attempt", fail)
    with pytest.raises(ValueError, match="corrupt"):
        rolling.resilient_training(tmp_path, "full")


def test_accounting_does_not_rebill_the_restored_prefix_as_new_work(tmp_path):
    prior = tmp_path / "old-round"
    write_json(prior / "outcome-000.json", {"reward": 1})
    joint = tmp_path / "full/joint"
    joint.mkdir(parents=True)
    (joint / "round-0000").symlink_to(prior, target_is_directory=True)
    write_json(joint / "round-0012/outcome-000.json", {"reward": 0})
    write_json(joint / "round-0012/editor-receipt-000.json", {"reward": 0})
    write_json(joint / "round-0012/solver-receipt-000.json", {"reward": 0})
    result = attempt_cost(tmp_path, "full", 12)
    assert result["new_persisted_outcomes"] == 1
    assert result["new_generated_role_receipts"] == {"editor": 1, "solver": 1}
    assert result["new_committed_role_rounds"] == {"editor": 0, "solver": 0}
    assert result["unpersisted_inflight_work_unknown"]
