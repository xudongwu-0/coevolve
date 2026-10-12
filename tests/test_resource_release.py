import copy

import pytest

from sao_bridge.resources import ADAPTIVE_SCRIPT, PAUSE_ROOT, research_resources_released


def paused_release():
    host = "dev-NDH200-026"
    return {
        "foreign_gpu_pids": [], "remaining_gpu_pids": [], "verified_filler_pids": [],
        "filler_unit_active": False,
        "operator_pause": {
            "verified": True, "host": host, "path": str(PAUSE_ROOT / host / "PAUSED"),
            "sha256": "a" * 64,
            "content": {"paused": True, "reason": "User requested stopping GPU stress on s23-s26"},
            "service": {
                "ActiveState": "inactive", "ConditionResult": "no", "User": "bruli",
                "ExecStart": f"argv[]=/usr/bin/python3 -B {ADAPTIVE_SCRIPT} supervisor ;",
            },
        },
    }


def test_explicit_operator_pause_is_not_reported_as_active_filler():
    release = paused_release()
    assert research_resources_released(release)
    assert release["filler_unit_active"] is False


def test_existing_active_filler_receipts_remain_valid():
    release = {
        "foreign_gpu_pids": [], "remaining_gpu_pids": [12], "verified_filler_pids": [12],
        "filler_unit_active": True,
    }
    assert research_resources_released(release)


@pytest.mark.parametrize("active", [False, True])
def test_foreign_workload_is_never_admitted(active):
    release = paused_release()
    release.update(filler_unit_active=active, foreign_gpu_pids=[99], remaining_gpu_pids=[99])
    assert not research_resources_released(release)


def test_unexplained_inactive_filler_still_rejected():
    release = paused_release()
    release.pop("operator_pause")
    assert not research_resources_released(release)


@pytest.mark.parametrize("fault", [
    "host", "path", "sha256", "verified", "paused", "reason", "service", "remaining",
])
def test_incomplete_or_incompatible_pause_evidence_rejected(fault):
    release = copy.deepcopy(paused_release())
    pause = release["operator_pause"]
    if fault in {"host", "path", "sha256"}:
        pause[fault] = "wrong"
    elif fault == "verified":
        pause[fault] = False
    elif fault in {"paused", "reason"}:
        pause["content"][fault] = False
    elif fault == "service":
        pause["service"]["User"] = "unrelated"
    else:
        release["remaining_gpu_pids"] = [99]
    assert not research_resources_released(release)


def test_explicit_serial_training_pause_is_a_valid_release():
    from sao_bridge.resources import SERIAL_PAUSE_REASON
    release = paused_release()
    release["operator_pause"]["content"]["reason"] = SERIAL_PAUSE_REASON
    release["operator_pause"]["service"].update(ConditionResult="yes", MainPID="0")
    assert research_resources_released(release)
    release["operator_pause"]["service"]["MainPID"] = "17"
    assert not research_resources_released(release)


def test_explicit_pause_targets_only_verified_supervisor(monkeypatch):
    from pathlib import Path
    from sao_bridge import resources

    supervisor = {"pid": 100, "uid": 200, "cmd": ["filler"], "cgroup": "verified"}
    foreign = {"pid": 999, "verified_filler": False}
    before = {
        "host": "dev-NDH200-026", "unit": {"ActiveState": "active"},
        "verified_unit": True, "supervisor": supervisor,
        "processes": [{"pid": 101, "verified_filler": True}, foreign],
    }
    after = {**before, "unit": {"ActiveState": "inactive"}, "processes": [foreign]}
    snapshots = iter([before, after])
    commands = []
    monkeypatch.setattr(resources, "require_research_node", lambda: None)
    monkeypatch.setattr(resources, "audit", lambda: next(snapshots))
    monkeypatch.setattr(resources, "process_identity", lambda pid: supervisor)
    monkeypatch.setattr(Path, "read_text", lambda self: "100 (python3) " + " ".join(["1"] * 20))
    monkeypatch.setattr(resources.subprocess, "run", lambda command, **kwargs: commands.append(command))
    monkeypatch.setattr(resources, "operator_pause_audit", lambda: {"verified": True})
    result = resources.pause_verified_fillers()
    assert commands[0][-1] == "write-pause-marker"
    assert commands[1] == ["sudo", "-n", "kill", "-TERM", "100"]
    assert commands[2] == ["sudo", "-n", "systemctl", "stop", resources.UNIT]
    assert result["preserved_foreign_gpu_pids"] == [999]
    assert result["stopped_filler_pids"] == [101]


@pytest.mark.parametrize("verified,active", [(False, "active"), (True, "inactive")])
def test_explicit_pause_rejects_unverified_or_inactive_unit(monkeypatch, verified, active):
    from sao_bridge import resources
    monkeypatch.setattr(resources, "require_research_node", lambda: None)
    monkeypatch.setattr(resources, "audit", lambda: {
        "verified_unit": verified, "unit": {"ActiveState": active},
    })
    with pytest.raises(RuntimeError, match="currently verified active"):
        resources.pause_verified_fillers()
