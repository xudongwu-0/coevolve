import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

from sao_bridge.common import read_json, write_json


@pytest.fixture
def insertion(tmp_path, monkeypatch):
    driver = Path(__file__).resolve().parents[1] / "runs/alfworld_fixed_w1_h3_v1/coordinator.py"
    spec = importlib.util.spec_from_file_location("fixed_solver_insertion", driver)
    module = importlib.util.module_from_spec(spec)
    root, ma = tmp_path / "insert", tmp_path / "ma"
    root.mkdir(); ma.mkdir()
    design = {
        "ma_root": str(ma), "ma_unit": "ma-unit", "ma_block": str(ma / "block-01"),
        "ma_stages": [{"unit": "editor"}, {"unit": "solver"}],
    }
    from sao_bridge import common
    original_read = common.read_json
    monkeypatch.setattr(common, "read_json", lambda path: design if Path(path) == driver.parent / "DESIGN.json"
                        else original_read(path))
    spec.loader.exec_module(module)
    module.ROOT = root
    write_json(ma / "MONITOR.json", {
        "phase": "training", "block": design["ma_block"], "progress": {"editor": 40, "solver": 40},
    })
    (ma / "driver.py").write_text("frozen")
    monkeypatch.setattr(module, "coordinator_pid", lambda: 12345)
    monkeypatch.setattr(module, "check_stage", lambda stage: False)
    monkeypatch.setattr(module, "stopped", lambda pid: True)
    run = Mock()
    monkeypatch.setattr(module.subprocess, "run", run)
    return module, root, ma, run


def test_pause_signals_only_coordinator_not_native_learners(insertion):
    module, root, ma, run = insertion
    assert module.pause_coordinator() == 12345
    run.assert_called_once_with(["sudo", "-n", "kill", "-STOP", "12345"], check=True, timeout=30)
    request = read_json(root / "PAUSE_REQUEST.json")
    assert request["native_learners_are_not_signalled"]
    assert request == read_json(ma / "FIXED_W1_H3_INSERTION.json")


@pytest.mark.parametrize("fault", ["evaluation", "late_training", "finished_learner"])
def test_pause_refuses_to_race_an_evaluation_or_block_boundary(insertion, monkeypatch, fault):
    module, root, ma, run = insertion
    monitor = read_json(ma / "MONITOR.json")
    if fault == "evaluation":
        monitor["phase"] = "evaluation"
    elif fault == "late_training":
        monitor["progress"]["solver"] = 85
    else:
        monkeypatch.setattr(module, "check_stage", lambda stage: True)
    write_json(ma / "MONITOR.json", monitor)
    with pytest.raises(RuntimeError):
        module.pause_coordinator()
    run.assert_not_called()
    assert not (root / "PAUSE_REQUEST.json").exists()


def test_resume_after_h3_requires_released_resources(insertion, monkeypatch):
    module, root, _, run = insertion
    monkeypatch.setattr(module, "stopped", lambda pid: False)
    release = Mock(return_value={"node3": "released"})
    monkeypatch.setattr(module, "released", release)
    module.resume_coordinator(12345, resources_are_free=True)
    release.assert_called_once()
    run.assert_called_once_with(["sudo", "-n", "kill", "-CONT", "12345"], check=True, timeout=30)
    assert read_json(root / "MA_RESUMED.json")["release"] == {"node3": "released"}


def test_resume_refuses_reused_pid_or_occupied_resources(insertion, monkeypatch):
    module, _, _, run = insertion
    with pytest.raises(RuntimeError, match="PID changed"):
        module.resume_coordinator(999, resources_are_free=True)
    monkeypatch.setattr(module, "released", Mock(side_effect=RuntimeError("occupied")))
    with pytest.raises(RuntimeError, match="occupied"):
        module.resume_coordinator(12345, resources_are_free=True)
    run.assert_not_called()
