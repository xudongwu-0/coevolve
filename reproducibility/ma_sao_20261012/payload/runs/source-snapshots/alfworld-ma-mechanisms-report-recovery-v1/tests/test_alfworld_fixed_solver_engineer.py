import copy
from pathlib import Path

import pytest

from sao_bridge import alfworld_fixed_solver_engineer as study
from sao_bridge.alfworld_h_train import training_spec
from sao_bridge.alfworld_sao_control import backend_wiring_audit
from sao_bridge.analysis import paired_difference
from sao_bridge.common import read_json


@pytest.fixture
def historical():
    old = read_json(study.HISTORY / "sequential/h2/phase_config.json")
    state = read_json(study.HISTORY / "sequential/STATE_AFTER_h2.json")
    return old, state


def test_one_full_engineer_phase_with_solver_unchanged(tmp_path, historical):
    old, state = historical
    config = study.continuation_config(old, state, tmp_path, {})
    assert config["engineer_model"] == state["engineer"]
    assert config["engineer_critic"] == state["engineer_critic"]
    assert config["solver_model"] == old["solver_model"]
    assert config["solver_version"] == old["solver_version"]
    assert config["steps"] == 32 and config["batch_size"] == 6
    assert config["critic_warmup_steps"] == 5
    assert config["seed"] == old["seed"] and config["save_interval_steps"] == 32
    assert config["learning_rate"] == 2e-6 and config["critic_learning_rate"] == 1e-5
    assert backend_wiring_audit(config)["no_ma_no_kl"]
    assert config["run"] == str(tmp_path / "training")


@pytest.mark.parametrize("fault", ["solver", "solver_version", "engineer", "batch"])
def test_continuation_refuses_changed_counterpart_or_phase(tmp_path, historical, fault):
    old, state = historical
    if fault == "batch":
        old["batch_size"] = 8
    elif fault == "engineer":
        state["engineer"] = old["engineer_model"]
    else:
        state[fault] = "different"
    with pytest.raises(ValueError):
        study.continuation_config(old, state, tmp_path, {})


def test_opt_in_acceleration_reaches_actual_engineer_spec(tmp_path, historical):
    old, state = historical
    acceleration = {
        "training_kernel_overlay": "/kernels", "training_kernel_versions": {"triton": "3.7.1"},
        "training_kernel_validation": "/passed.json",
    }
    config = study.continuation_config(old, state, tmp_path, acceleration)
    config["pinned_files"] = {"/passed.json": "digest"}
    spec = training_spec(config, tmp_path, "unused")
    for key, value in acceleration.items():
        assert spec[key] == value
    assert spec["pinned_files"] == config["pinned_files"] and spec["rollout_setup_timeout"] == 1800
    original = training_spec(old, tmp_path, "unused")
    assert "training_kernel_overlay" not in original and "rollout_setup_timeout" not in original


def fixture_reports(label, rewards):
    values = [{"sample_index": slot, "valid": True} for slot in range(8)]
    reports = {}
    for slot in range(8):
        reports[f"{label}-proposal{slot:02d}"] = {
            "metric": "mean@2", "question_count": 2, "samples_per_question": 2,
            "accuracy": sum(rewards[slot]) / 2,
            "results": [
                {"id": f"alfworld-{task}", "task_id": task, "sample_index": sample,
                 "rollout_seed": 100 + task * 17 + sample * 1009, "reward": rewards[slot][task]}
                for task in range(2) for sample in range(2)
            ],
        }
    reports["incumbent_harness"] = {"accuracy": 0.5}
    return values, reports


def test_population_preserves_slots_native_seeds_and_question_pairing():
    before, left = fixture_reports("before", [[0, 0]] * 8)
    after, right = fixture_reports("after", [[0, 1]] * 8)
    a = study.proposal_population_report(before, left, "before")
    b = study.proposal_population_report(after, right, "after")
    assert a["question_count"] == 2 and a["samples_per_question"] == 16
    assert len(a["results"]) == 32
    assert {row["sample_index"] for row in a["results"]} == set(range(16))
    assert {row["rollout_seed"] for row in a["results"]} == {100, 1109, 117, 1126}
    result = paired_difference(a, b, draws=100)
    assert result["candidate_minus_control"] == 0.5 and result["improved_questions"] == 1


def test_selection_retains_common_incumbent_without_hiding_raw_candidate():
    values, reports = fixture_reports("after", [[0, 0]] * 7 + [[0, 1]])
    selected = study.select_side(values, reports, "after", 0.02)
    assert selected["selected"] == "incumbent_harness"
    assert selected["best_valid_proposal"] == "after-proposal07"
    reports["after-proposal07"]["accuracy"] = 0.6
    assert study.select_side(values, reports, "after", 0.02)["selected"] == "after-proposal07"


def test_all_invalid_is_explicit_model_failure_not_learned_success():
    values, reports = fixture_reports("after", [[0, 1]] * 8)
    for value in values:
        value["valid"] = False
    selected = study.select_side(values, reports, "after", 0.02)
    assert selected["status"] == "all_invalid_model_outputs"
    assert selected["best_valid_proposal"] is None
    assert selected["selected"] == "incumbent_harness" and selected["valid_proposals"] == 0


@pytest.mark.parametrize("fault", ["missing_slot", "sample_protocol", "duplicate_identity"])
def test_population_rejects_incomplete_or_incompatible_scores(fault):
    values, reports = fixture_reports("before", [[0, 1]] * 8)
    if fault == "missing_slot":
        values.pop()
    elif fault == "sample_protocol":
        reports["before-proposal01"]["samples_per_question"] = 1
    else:
        reports["before-proposal01"]["results"][-1] = copy.deepcopy(reports["before-proposal01"]["results"][0])
    with pytest.raises(ValueError):
        result = study.proposal_population_report(values, reports, "before")
        paired_difference(result, result)


def test_uncertain_gain_cannot_be_reported_as_established():
    assert study.outcome({
        "candidate_minus_control": .015, "paired_question_bootstrap_95pct": [-.01, .04],
    }) == "positive_point_estimate_not_established"
    assert study.outcome({
        "candidate_minus_control": -.02, "paired_question_bootstrap_95pct": [-.04, -.005],
    }) == "exploratory_degradation"
