import importlib.util
import os
from pathlib import Path

import pytest

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sao_bridge.common import file_sha, read_json, write_json


MODULE = Path(__file__).resolve().parents[1] / "runs/alfworld_ma_simultaneous_from97_v1/resume_after_whale.py"
spec = importlib.util.spec_from_file_location("peak_ma_deadline_resume", MODULE)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)


def first_boundary(tmp_path):
    endpoint = tmp_path / "block-01/round0096_COMPLETE.json"
    write_json(endpoint, {"selection": {"harness": "saved-native-harness"}})
    write_json(tmp_path / "ENDPOINT_RECOVERY_COMPLETE.json", {
        "status": "passed", "recovered_endpoint_sha256": file_sha(endpoint)})
    write_json(tmp_path / "CURVE.json", {"points": [
        {"rounds": 144}, {"rounds": 192},
        {"rounds": 240, "harness": "saved-native-harness", "result": str(endpoint)}]})
    return endpoint


def test_resume_binds_completed_native_checkpoint_and_selection(tmp_path):
    first_boundary(tmp_path)
    points, harness = resume.checked_boundary(tmp_path)
    assert [point["rounds"] for point in points] == [144, 192, 240]
    assert harness == "saved-native-harness"


@pytest.mark.parametrize("change", ["endpoint", "curve", "failed", "finalized", "harness"])
def test_resume_rejects_changed_or_incomplete_boundary(tmp_path, change):
    endpoint = first_boundary(tmp_path)
    if change == "endpoint":
        write_json(endpoint, {"selection": {"harness": "other"}})
    elif change in ("failed", "finalized"):
        write_json(tmp_path / ("FAILED.json" if change == "failed" else "RESULT.json"), {})
    else:
        curve = read_json(tmp_path / "CURVE.json")
        if change == "curve":
            curve["points"].pop()
        else:
            curve["points"][-1]["harness"] = "other"
        write_json(tmp_path / "CURVE.json", curve)
    with pytest.raises(ValueError):
        resume.checked_boundary(tmp_path)


@pytest.mark.parametrize("changed", [None, "foreign", "filler", "missing_node", "missing_pins", "evidence"])
def test_resource_release_requires_both_nodes_and_unchanged_evidence(tmp_path, changed):
    evidence = tmp_path / "COMPLETE.json"
    write_json(evidence, {"status": "completed"})
    boundary = {
        "status": "released", "pins": {str(evidence): file_sha(evidence)},
        "release": {node: {"foreign_gpu_pids": [], "filler_unit_active": True}
                    for node in ("node3", "node4")},
    }
    if changed == "foreign":
        boundary["release"]["node3"]["foreign_gpu_pids"] = [123]
    elif changed == "filler":
        boundary["release"]["node4"]["filler_unit_active"] = False
    elif changed == "missing_node":
        del boundary["release"]["node4"]
    elif changed == "missing_pins":
        boundary["pins"] = {}
    elif changed == "evidence":
        write_json(evidence, {"status": "changed"})
    write_json(tmp_path / "WHALE_RESOURCE_BOUNDARY.json", boundary)
    if changed:
        with pytest.raises(ValueError):
            resume.checked_whale_release(tmp_path)
    else:
        assert resume.checked_whale_release(tmp_path) == boundary
