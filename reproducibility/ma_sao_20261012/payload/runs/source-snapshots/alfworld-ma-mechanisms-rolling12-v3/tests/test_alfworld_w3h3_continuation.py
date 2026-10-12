import copy
from pathlib import Path

import pytest

from sao_bridge import alfworld_w3h3_continuation as study
from sao_bridge.alfworld_h_train import training_spec as h_spec
from sao_bridge.alfworld_sao_control import backend_wiring_audit
from sao_bridge.alfworld_w_train import training_spec as w_spec
from sao_bridge.common import read_json, write_json


def test_early_stop_means_79_percent_not_globalstep79():
    trained = read_json(study.RECOVERY / "training/optimizer/trained.json")
    evaluation = read_json(study.RECOVERY / "evaluation/extra40/request.json")
    scores = read_json(study.RECOVERY / "extra40_COMPLETE.json")
    pair = study.early_pair(trained, evaluation, scores)
    assert pair["actor"].endswith("epoch0epochstep39globalstep39")
    evaluation["solver_model"] = trained["checkpoints"]["actor"]
    with pytest.raises(ValueError, match="approximately79"):
        study.early_pair(trained, evaluation, scores)


def test_solver_population_and_order_unchanged_only_new_trajectory_ids():
    previous = read_json(study.RECOVERY / "data.json")
    rows = study.solver_rows_for_cycle(previous)
    assert len(rows) == 640
    for old, new in zip(previous, rows, strict=True):
        assert new["id"] != old["id"]
        assert new["partner_version"] == 3
        assert {k: v for k, v in old.items() if k not in ("id", "partner_version")} == {
            k: v for k, v in new.items() if k not in ("id", "partner_version")}
    assert not {row["id"] for row in rows} & {row["id"] for row in previous}
    with pytest.raises(ValueError):
        study.solver_rows_for_cycle(previous[:-1])
    invalid = copy.deepcopy(previous)
    invalid[-1]["id"] = invalid[0]["id"]
    with pytest.raises(ValueError):
        study.solver_rows_for_cycle(invalid)


def test_both_real_role_specs_have_corrected_sao_and_acceleration(tmp_path):
    state = read_json(study.HISTORY / "sequential/STATE_AFTER_h2.json")
    state.update(solver="/saved/w2-early", solver_critic="/saved/w2-early-critic",
                 solver_version="early-w2")
    acceleration = {key: read_json(study.ACCELERATION)[key] for key in study.ACCELERATION_KEYS}
    h, w = study.configs(tmp_path, state, acceleration)
    assert (h["steps"], h["batch_size"], h["critic_warmup_steps"]) == (32, 6, 5)
    assert (w["steps"], w["batch_size"], w["critic_warmup_steps"]) == (80, 8, 5)
    assert h["engineer_model"] == state["engineer"]
    assert h["solver_model"] == w["solver_model"] == state["solver"]
    assert w["solver_critic"] == state["solver_critic"]
    assert h["phase"] == "h3" and w["phase"] == "w3"
    for config, role, spec in (
        (h, "editor", h_spec(h, Path(h["run"]), "unused")),
        (w, "solver", w_spec(w, Path(w["run"]))),
    ):
        assert backend_wiring_audit(config, role=role)["no_ma_no_kl"]
        assert spec["processed_behavior_logprobs"] and spec["independent_trajectories"]
        assert config["learning_rate"] == 2e-6 and config["critic_learning_rate"] == 1e-5
        for key in study.ACCELERATION_KEYS:
            assert spec[key] == acceleration[key]
        assert spec["rollout_setup_timeout"] == 1800


def test_solver_acceleration_is_opt_in_for_existing_configs(tmp_path):
    old = read_json(study.RECOVERY / "training_config.json")
    spec = w_spec(old, tmp_path)
    assert all(key not in spec for key in study.ACCELERATION_KEYS)
    assert "rollout_setup_timeout" not in spec
    with pytest.raises(ValueError, match="backend role"):
        backend_wiring_audit(old, role="unknown")


@pytest.mark.parametrize("valid", [True, False])
def test_experimental_w3_never_mislabels_h2_fallback_as_h3(tmp_path, monkeypatch, valid):
    no_patch = tmp_path / "no_patch.json"
    write_json(no_patch, {"benchmark": "alfworld", "actions": []})
    protocol = {
        "initial_engineer_pair": {"actor": "/old/h2", "critic": "/old/h2-critic"},
        "historical_h2": "/saved/h2.json", "no_patch": str(no_patch),
        "evaluation": {"promotion_min_delta": 0.02},
    }
    values = {label: [{"sample_index": slot, "valid": valid, "patch": read_json(no_patch)}
                      for slot in range(8)] for label in ("before", "after")}
    reports = {"incumbent_harness": {"accuracy": 1.0}}
    for label in values:
        for slot in range(8):
            reports[f"{label}-proposal{slot:02d}"] = {
                "accuracy": 0.0, "metric": "mean@2", "question_count": 1, "samples_per_question": 2,
                "results": [
                    {"id": "alfworld-240", "task_id": 240, "sample_index": sample,
                     "rollout_seed": 100 + sample * 1009, "reward": 0.0}
                    for sample in range(2)
                ],
            }
    monkeypatch.setattr(study, "checkpoint_pair", lambda *_: {"actor": "/new/h3", "critic": "/new/critic"})
    monkeypatch.setattr(study, "generate_proposals", lambda *_: values)
    monkeypatch.setattr(study, "matrix_config", lambda *_: {})
    monkeypatch.setattr(study, "run_matrix", lambda *_: ({}, tmp_path / "selection/matrix"))
    monkeypatch.setattr(study, "matrix_reports", lambda *_: reports)
    monkeypatch.setattr(study, "proposal_work", lambda *_: {"proposals": 16})
    if valid:
        selected = study.select_h3(None, tmp_path, protocol, {"solver": "/w2", "solver_version": "v2"}, {})
        result = read_json(tmp_path / "H3_SELECTION.json")
        assert selected == result["actual_h3_candidate"]
        assert selected != protocol["historical_h2"]
        assert result["deployment_harness"] == protocol["historical_h2"]
        assert result["sides"]["after"]["incumbent_retained"]
    else:
        with pytest.raises(RuntimeError, match="No valid actual H3"):
            study.select_h3(None, tmp_path, protocol, {"solver": "/w2", "solver_version": "v2"}, {})
        assert read_json(tmp_path / "MODEL_FAILURE.json")["new_solver_updates"] == 0
