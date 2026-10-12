from pathlib import Path
import os
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_h_train import external_solver_binding, training_spec as engineer_spec
from sao_bridge.alfworld_w_train import training_spec as solver_spec
from sao_bridge.common import file_sha, write_json


def binding_fixture(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    path = tmp_path / "binding.json"
    record = {
        "status": "ready", "consumer_host": os.uname().nodename, "provider_host": "dev-NDH200-026",
        "model": str(model), "solver_version": "fixed-w1",
        "urls": ["http://127.0.0.1:19501/v1", "http://127.0.0.1:19511/v1"],
    }
    write_json(path, record)
    config = {
        "solver_model": str(model), "solver_version": "fixed-w1",
        "actor_gpus": 4, "rollout_gpus": 4, "batch_size": 8,
        "external_solver": {"binding": str(path), "binding_sha256": file_sha(path)},
    }
    return config, record, path


def test_existing_local_solver_path_is_unchanged():
    assert external_solver_binding({}) is None


def test_separate_node_allows_full_eight_gpu_learner_without_changing_counterpart(tmp_path):
    config, record, _ = binding_fixture(tmp_path)
    assert external_solver_binding(config) == record["urls"]


@pytest.mark.parametrize("fault", ["hash", "model", "version", "host", "remote_url", "missing_path", "duplicates", "allocation"])
def test_external_binding_rejects_identity_or_allocation_changes(tmp_path, fault):
    config, record, path = binding_fixture(tmp_path)
    if fault == "hash":
        config["external_solver"]["binding_sha256"] = "changed"
    elif fault == "model":
        record["model"] = str(tmp_path / "different_model")
    elif fault == "version":
        record["solver_version"] = "different"
    elif fault == "host":
        record["consumer_host"] = "different"
    elif fault == "remote_url":
        record["urls"] = ["http://example.com:19501/v1"]
    elif fault == "missing_path":
        record["urls"] = ["http://127.0.0.1:19501"]
    elif fault == "duplicates":
        record["urls"] *= 2
    else:
        config["actor_gpus"] = 2
    if fault not in ("hash", "allocation"):
        write_json(path, record)
        config["external_solver"]["binding_sha256"] = file_sha(path)
    with pytest.raises(ValueError):
        external_solver_binding(config)


def test_phase_helpers_forward_opt_in_matched_reference_settings(tmp_path):
    common = {
        "phase": "stage", "experiment": "components", "steps": 96, "batch_size": 8,
        "harness": "empty", "data": "rows", "solver_model": "w1", "engineer_model": "h0",
        "kl_ctl": .02, "reference_model": "reference", "reference_storage_dtype": "bfloat16",
    }
    for spec in (engineer_spec(common, tmp_path, "http://127.0.0.1:28600"), solver_spec(common, tmp_path)):
        assert spec["kl_ctl"] == .02 and spec["reference_model"] == "reference"
        assert spec["reference_storage_dtype"] == "bfloat16"
        assert not spec.get("joint_root") and not spec.get("central_context")
