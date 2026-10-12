from types import SimpleNamespace

import pytest
import torch

from harness_r1_searchqa.grpo_objective import group_advantages, grpo_loss
from harness_r1_searchqa.grpo_train import validate_engineer_scale
from harness_r1_searchqa.upstream_ppo import ROOT, official_modules


def test_normalization_is_directly_official_even_for_float_roundoff():
    _, official = official_modules()
    args = SimpleNamespace(
        custom_reward_post_process_path=None, reward_key=None,
        advantage_estimator="grpo", rewards_normalization=True,
        grpo_std_normalization=True, n_samples_per_prompt=8, rollout_batch_size=2,
    )
    values = [0.0] * 7 + [1.0] + [0.4] * 8
    _, expected = official.post_process_rewards(
        args, [official.Sample(reward=value) for value in values],
    )
    actual, _ = group_advantages(torch.tensor(values))
    torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
    assert official.__file__.startswith(str(ROOT))


@pytest.mark.parametrize("advantage", [-1.25, 0.0, 1.25])
def test_loss_and_gradient_match_official_clipping_and_tis(advantage):
    official, _ = official_modules()
    new = torch.tensor([-0.1, -1.8, -0.5, -3.0], requires_grad=True)
    old = torch.tensor([-0.5, -1.0, -0.6, -2.0])
    rollout = torch.tensor([-3.0, -0.4, -0.8, -2.5])
    adv = torch.tensor(advantage)
    actual, _ = grpo_loss(new, old, rollout, adv)
    reference = new.detach().clone().requires_grad_(True)
    token_loss, _ = official.compute_policy_loss(
        old - reference, adv.expand_as(reference), 0.2, 0.28,
    )
    expected = (token_loss * (old - rollout).exp().clamp(0, 2.0)).mean()
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(new.grad, reference.grad)


def test_4b_requires_explicit_pilot_and_full_parameters():
    validate_engineer_scale("qwen3", 4_000_000_000, allow_4b=True)
    validate_engineer_scale("qwen3_5_text", 8_953_803_264)
    with pytest.raises(ValueError):
        validate_engineer_scale("qwen3", 4_000_000_000)
    with pytest.raises(ValueError):
        validate_engineer_scale("qwen3", 40_000, allow_4b=True)
