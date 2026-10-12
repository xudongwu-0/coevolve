from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.validity_credit_diagnostics import contrasts


def report(values):
    return {"metric": "mean@1", "samples_per_question": 1, "question_count": len(values),
            "results": [{"id": f"task{i}", "task_id": i, "sample_index": 0,
                         "rollout_seed": 42 + i, "reward": v} for i, v in enumerate(values)]}


def test_fixed_h0_gain_does_not_imply_selected_system_gain():
    reports = {"control": {"h0": report([1, 0]), "selected": report([1, 0]), "no_patch": report([0, 0])},
               "treatment": {"h0": report([1, 1]), "selected": report([0, 1]), "no_patch": report([0, 1])},
               "baseline": {"h2": report([1, 1])}}
    c = contrasts(reports, [0, 1])
    assert c["fixed_h0_weight_gain"]["candidate_minus_control"] == .5
    assert c["remeasured_system_gain"]["candidate_minus_control"] == 0
    assert c["treatment_patch_gain"]["candidate_minus_control"] == -.5
    assert c["versus_fresh_baseline"]["candidate_minus_control"] == -.5
    assert contrasts(reports, [0])["remeasured_system_gain"]["candidate_minus_control"] == -1
    reports['treatment']['selected']['results'][0]['rollout_seed'] = 900
    with pytest.raises(ValueError, match="identities differ"):
        contrasts(reports, [0, 1])
