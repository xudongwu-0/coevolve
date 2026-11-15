from dataclasses import replace
import math

import pytest
import torch

from whale_local.webshop.core import Episode, Protocol, Sample, Turn, training_rows, evaluation_report, compare_reports
from whale_local.webshop.__main__ import optimize_group
from whale_local.sao_bridge.webshop_grpo import validate_solver_protocol, main as sao_main


class Codec:
    def privileged(self, *args):
        raise AssertionError("No privileged teacher context is permitted")


def protocol(**changes):
    base = Protocol(train_ids=(600,), eval_ids=(256,), eval_seeds=(0,), group_size=2,
                    updates=1, grpo_profile="reference_step", distill_coefficient=0,
                    reference_kl_coefficient=0)
    return replace(base, **changes)


def episode(success, seed, invalid=(), lengths=(2,), score=None):
    result = Episode(600, "train", seed, {"task_id": 600, "goal": "same"}, policy_version=0)
    result.terminated = True
    result.task_score = float(success) if score is None else score
    result.turns = [
        Turn("actual prompt", [1, 2], Sample("click[item]", [4] * length, [-1.] * length),
             "observation", "click[item]", "next", i in invalid)
        for i, length in enumerate(lengths)
    ]
    return result


def test_reference_advantages_use_all_action_turns_and_sample_std():
    rows, stats = training_rows([episode(True, 1, lengths=(2, 2)), episode(False, 2)],
                                Codec(), protocol(), use_distillation=False)
    scores = torch.tensor([10., 10., 0.])
    expected = (scores - scores.mean()) / (scores.std() + 1e-6)
    torch.testing.assert_close(torch.tensor([r.advantage for r in rows]), expected)
    assert stats["reward_statistics_scope"] == "all_action_turns_in_exact_task_group"
    assert stats["teacher_turns"] == stats["eligible_success_targets"] == 0


def test_invalid_action_is_penalized_without_changing_environment_score():
    a, b = episode(False, 1, invalid=(0,)), episode(False, 2)
    rows, stats = training_rows([a, b], Codec(), protocol(), use_distillation=False)
    assert rows[0].advantage < 0 < rows[1].advantage
    assert stats["invalid_action_turns"] == 1
    assert stats["valid_action_ratio"] == .5
    assert a.task_score == b.task_score == 0
    assert not a.success and not b.success


def test_partial_environment_score_is_not_falsely_a_training_success():
    a, b = episode(False, 1, score=.9), episode(False, 2, score=.2)
    rows, stats = training_rows([a, b], Codec(), protocol(), use_distillation=False)
    assert all(r.advantage == 0 for r in rows)
    assert stats["successes"] == 0
    assert (a.task_score, b.task_score) == (.9, .2)


def test_all_valid_failed_groups_remain_zero_advantage():
    rows, stats = training_rows([episode(False, 1), episode(False, 2)], Codec(),
                                protocol(), use_distillation=False)
    assert stats["zero_reward_variance"]
    assert all(r.advantage == 0 for r in rows)


def test_reference_profile_rejects_distillation():
    with pytest.raises(ValueError, match="does not permit distillation"):
        training_rows([episode(True, 1), episode(False, 2)], Codec(), protocol())


def test_reference_group_rejects_mixed_task_or_policy_identity():
    a, b = episode(True, 1), episode(False, 2)
    b.policy_version = 1
    with pytest.raises(ValueError, match="task identity"):
        training_rows([a, b], Codec(), protocol(), use_distillation=False)


def test_invalid_turn_tokens_and_behavior_probabilities_are_retained():
    a, b = episode(False, 1, invalid=(0,), lengths=(3,)), episode(True, 2)
    rows, _ = training_rows([a, b], Codec(), protocol(), use_distillation=False)
    assert rows[0].response_ids == a.turns[0].sample.token_ids
    assert rows[0].old_log_probs == a.turns[0].sample.log_probs
    assert rows[0].student_prompt_ids == a.turns[0].prompt_ids


class TinyPolicy:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.model.weight)

    def log_probs(self, prompt, response, grad=False):
        return self.model.weight.reshape(1).expand(len(response)) - 1


def test_native_optimizer_uses_token_mean_instead_of_episode_mean():
    data = [episode(True, 1, lengths=(2,)), episode(False, 2, lengths=(6,))]
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.model.parameters(), lr=0)
    stats = optimize_group(policy, data, protocol(), "grpo", optimizer)
    a = 5 / (math.sqrt(50) + 1e-6)
    assert stats["loss"] == pytest.approx(-(2 * a - 6 * a) / 8, rel=1e-5)
    assert stats["gradient_norm"] == pytest.approx(a / 2, rel=1e-5)
    assert stats["loss_aggregation"] == "token_mean"
    assert stats["teacher_tokens"] == 0


def test_legacy_objective_is_preserved_as_explicit_control():
    data = [episode(True, 1, lengths=(2,)), episode(False, 2, lengths=(6,))]
    policy = TinyPolicy()
    stats = optimize_group(policy, data, protocol(grpo_profile="legacy_episode"), "grpo",
                           torch.optim.SGD(policy.model.parameters(), lr=0))
    assert stats["loss"] == pytest.approx(0, abs=1e-6)
    assert stats["loss_aggregation"] == "episode_mean"


def test_solver_bridge_requires_no_distillation_and_eight_rollouts():
    cfg = protocol(group_size=8).as_dict()
    validate_solver_protocol(cfg)
    with pytest.raises(ValueError, match="distill_coefficient=0"):
        validate_solver_protocol({**cfg, "distill_coefficient": .5})
    with pytest.raises(ValueError, match="eight rollouts"):
        validate_solver_protocol({**cfg, "group_size": 3})


def test_solver_bridge_does_not_authorize_full_training():
    cfg = protocol(group_size=8, updates=100).as_dict()
    with pytest.raises(ValueError, match="subset pilots"):
        validate_solver_protocol(cfg)


def test_new_protocol_validation_fails_instead_of_silent_default():
    with pytest.raises(ValueError, match="Unknown GRPO"):
        protocol(grpo_profile="unknown")
    with pytest.raises(ValueError, match="invalid_action_penalty"):
        protocol(invalid_action_penalty=float("nan"))


def paired_reports():
    reference = protocol(group_size=8, eval_ids=tuple(range(256, 288)))
    initial = replace(reference, grpo_profile="legacy_episode")
    reports = []
    for cfg, success in [(initial, False), (reference, True)]:
        data = []
        for task in cfg.eval_ids:
            sample = Episode(task, "eval", 0, {"task_id": task})
            sample.terminated, sample.task_score = True, float(success)
            data.append(sample)
        training = {
            "method": "grpo", "protocol": cfg.as_dict(), "base_model": "/same/model",
            "base_model_identity": {"sha": "same"}, "implementation_identity": {"sha": "same"},
            "assets": {"sha": "same"}, "trainable_parameters": 10,
            "metrics": [{"gradient_norm": .1, "reward_std": 1, "teacher_tokens": 0}],
        }
        report = evaluation_report(data, cfg, "/model", {"sha": "same"}, training)
        report.update(base_model_identity={"sha": "same"}, implementation_identity={"sha": "same"})
        reports.append(report)
    return reports


def test_grpo_promotion_gate_does_not_require_a_distillation_teacher():
    left, right = paired_reports()
    result = compare_reports(left, right, comparison="sao_grpo")
    assert result["learning_signal_engaged"]
    assert not result["subset_evidence_gate"]["teacher_signal_engaged"]
    assert result["subset_evidence_gate"]["passed"]


def test_grpo_comparison_does_not_hide_changed_harness_or_budget():
    left, right = paired_reports()
    right["protocol"]["solver_harness"] = "Changed inference guidance"
    with pytest.raises(ValueError, match="Protocol"):
        compare_reports(left, right, comparison="sao_grpo")


def test_grpo_comparison_cannot_rename_distillation_as_grpo():
    left, right = paired_reports()
    right["training"]["method"] = "success_sd"
    with pytest.raises(ValueError, match="distilled"):
        compare_reports(left, right, comparison="sao_grpo")
