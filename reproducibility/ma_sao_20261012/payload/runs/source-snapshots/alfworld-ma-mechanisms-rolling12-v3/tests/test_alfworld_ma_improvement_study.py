from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_ma_improvement_study import contrasts, prepare, wait_campaign
from sao_bridge.common import write_json, file_sha


def report(values):
    return {"metric": "mean@1", "samples_per_question": 1, "question_count": len(values),
            "results": [{"id": f"alfworld-{i}", "task_id": i, "sample_index": 0,
                         "rollout_seed": 42 + i, "reward": value}
                        for i, value in enumerate(values)]}


def test_factorial_separates_weight_gain_from_patch_compatibility_and_subset():
    reports = {
        "start": {"h0": report([1, 0, 0, 0]), "selected_patch": report([0, 0, 0, 0]),
                  "no_patch": report([0, 0, 0, 0])},
        "selected": {"h0": report([1, 1, 0, 0]), "selected_patch": report([1, 1, 1, 0]),
                     "no_patch": report([0, 1, 0, 0])},
        "baseline": {"h2": report([1, 1, 1, 1])},
    }
    c = contrasts(reports, [0, 1, 2, 3])
    assert c["weight_gain_fixed_h0"]["candidate_minus_control"] == .25
    assert c["patch_gain_fixed_selected_weights"]["candidate_minus_control"] == .25
    assert c["system_gain"]["candidate_minus_control"] == .5
    assert c["compatibility_interaction"]["difference"] == .5
    assert c["versus_fresh_baseline"]["candidate_minus_control"] == -.25
    assert contrasts(reports, [3])["versus_fresh_baseline"]["candidate_minus_control"] == -1
    with pytest.raises(ValueError, match="Subset lost"):
        contrasts(reports, [0, 99])
    reports["selected"]["h0"]["results"][0]["rollout_seed"] = 999
    with pytest.raises(ValueError, match="identities differ"):
        contrasts(reports, [0, 1, 2, 3])


def test_no_prospective_label_after_seeing_endpoint(tmp_path):
    write_json(tmp_path / "best_COMPLETE.json", {"status": "completed"})
    with pytest.raises(ValueError, match="before campaign results"):
        prepare(tmp_path / "study", tmp_path)


def test_failed_or_unreleased_predecessor_cannot_be_used(tmp_path):
    protocol = {"campaign": str(tmp_path), "queue_timeout_seconds": 1}
    for label in ("matched", "best"):
        write_json(tmp_path / f"{label}_COMPLETE.json", {"status": "completed"})
        write_json(tmp_path / f"{label}_RELEASED.json", {
            "foreign_gpu_pids": [], "filler_unit_active": True})
    assert wait_campaign(protocol) == ({"status": "completed"}, {"status": "completed"})
    write_json(tmp_path / "matched_RELEASED.json", {"foreign_gpu_pids": [42], "filler_unit_active": True})
    with pytest.raises(RuntimeError, match="release GPUs"):
        wait_campaign(protocol)
    write_json(tmp_path / "editor_FAILED.json", {"error": "failed"})
    with pytest.raises(RuntimeError, match="predecessor failed"):
        wait_campaign(protocol)


def test_recovery_allows_only_pinned_training_failure(tmp_path):
    protocol = {"campaign": str(tmp_path), "queue_timeout_seconds": 1}
    for label in ("matched", "best"):
        write_json(tmp_path / f"{label}_COMPLETE.json", {"status": "completed"})
        write_json(tmp_path / f"{label}_RELEASED.json", {
            "foreign_gpu_pids": [], "filler_unit_active": True})
    failure = tmp_path / "editor_FAILED.json"
    write_json(failure, {"error": "known failure"})
    protocol["recovered_training"] = {"pinned_files": {str(failure): file_sha(failure)}}
    assert wait_campaign(protocol)[0]["status"] == "completed"
    write_json(tmp_path / "best_FAILED.json", {"error": "new evaluation failure"})
    with pytest.raises(RuntimeError, match="predecessor failed: best"):
        wait_campaign(protocol)
    write_json(failure, {"error": "different failure"})
    with pytest.raises(RuntimeError, match="receipt changed"):
        wait_campaign(protocol)
