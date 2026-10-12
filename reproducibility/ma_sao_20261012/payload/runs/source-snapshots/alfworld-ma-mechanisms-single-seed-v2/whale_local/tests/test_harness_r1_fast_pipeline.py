import json

from harness_r1_searchqa.fast_pipeline import select_candidate, record_results, remote_raw_urls
from sao_bridge.common import file_sha
from harness_r1_searchqa.serve_roles import validate_roles
import pytest


def test_selection_never_promotes_hook_errors(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    for index, (score, clean, errors) in enumerate([(1.0, False, 1), (0.125, True, 0)]):
        (samples / f"{index}.json").write_text(json.dumps({
            "id": str(index), "patch_sha256": str(index),
            "clean_execution": clean,
            "reward": {"hook_errors": errors, "score": score, "merged_patch": {"index": index}},
        }))
    assert select_candidate(tmp_path)["patch"] == {"index": 1}


def test_no_positive_candidate_is_explicitly_absent(tmp_path):
    (tmp_path / "samples").mkdir()
    assert select_candidate(tmp_path) is None


def test_disjoint_roles_accept_two_real_workloads():
    validate_roles([
        {"name": "target", "gpus": "0,1,2,3", "port": 18901},
        {"name": "engineer", "gpus": "4,5,6,7", "port": 19001},
    ])


def test_overlapping_gpu_roles_rejected():
    with pytest.raises(ValueError, match="disjoint"):
        validate_roles([
            {"name": "target", "gpus": "0,1,2,3", "port": 18901},
            {"name": "engineer", "gpus": "3,4,5,6", "port": 19001},
        ])


def test_readme_distinguishes_missing_candidate_from_rl_success(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Pilot\n")
    record_results(readme, {
        "controls": {"grpo4b": {"proposals": 32, "valid": 3, "clean": 2,
                                "dev": {"accepted": False}}},
        "rl_vs_sft": None, "rl_improvement_demonstrated": False,
    })
    text = readme.read_text()
    assert "not evaluated: no clean positive candidate" in text
    assert "RL improvement demonstrated: **False**" in text


def test_remote_raw_control_checks_model_template_and_roles(tmp_path):
    roles = tmp_path / "roles.json"
    model = tmp_path / "model"
    roles.write_text(json.dumps({"roles": [{
        "name": "engineer", "model": str(model), "gpus": "4,5,6,7",
        "engineer_think_prefill": True,
    }]}))
    ready = tmp_path / "READY.json"
    urls = ",".join(f"http://127.0.0.1:{port}/v1" for port in (19001, 19011, 19021, 19031))
    ready.write_text(json.dumps({
        "roles_sha256": file_sha(roles), "endpoints": {"engineer": urls},
    }))
    assert remote_raw_urls(ready, roles, model) == urls
    with pytest.raises(ValueError, match="model/template"):
        remote_raw_urls(ready, roles, tmp_path / "different_model")
    roles.write_text(roles.read_text() + "\n")
    with pytest.raises(ValueError, match="configuration changed"):
        remote_raw_urls(ready, roles, model)
