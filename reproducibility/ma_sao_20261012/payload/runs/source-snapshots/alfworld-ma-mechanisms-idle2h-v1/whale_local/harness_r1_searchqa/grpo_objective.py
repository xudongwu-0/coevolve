"""Harness-R1 GRPO using unchanged pinned Relax objective primitives.

Backend adaptation, not an exact Relax reproduction. No SAO/proxy objective.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from types import SimpleNamespace

from .upstream_ppo import official_modules


def group_advantages(rewards: torch.Tensor, group_size: int = 8):
    rewards = rewards.float()
    if rewards.ndim != 1 or group_size < 2 or rewards.numel() % group_size:
        raise ValueError("rewards must contain complete response groups of size >=2")
    if not torch.isfinite(rewards).all():
        raise ValueError("nonfinite reward")
    groups = rewards.reshape(-1, group_size)
    std = groups.std(-1, keepdim=True, correction=1)
    constant = std.squeeze(-1) == 0
    _, upstream = official_modules()
    args = SimpleNamespace(
        custom_reward_post_process_path=None, reward_key=None,
        advantage_estimator="grpo", rewards_normalization=True,
        grpo_std_normalization=True, n_samples_per_prompt=group_size,
        rollout_batch_size=groups.shape[0],
    )
    samples = [upstream.Sample(reward=value) for value in rewards.detach().cpu().tolist()]
    _, normalized = upstream.post_process_rewards(args, samples)
    advantages = torch.tensor(normalized, dtype=torch.float32, device=rewards.device)
    if not torch.isfinite(advantages).all():
        raise FloatingPointError("Official Harness-R1 produced non-finite advantages")
    return advantages, constant


def response_logprobs(logits: torch.Tensor, tokens: torch.Tensor,
                      temperature: float = 0.7, chunk_size: int = 128):
    """Compute response-only probabilities without a full FP32 vocabulary copy."""
    if logits.ndim != 2 or tokens.shape != logits.shape[:1] or temperature <= 0:
        raise ValueError("response logits/tokens or temperature mismatch")

    def chunk(x, y):
        return -F.cross_entropy(x.float() / temperature, y, reduction="none")

    pieces = []
    for start in range(0, tokens.numel(), chunk_size):
        x, y = logits[start:start + chunk_size], tokens[start:start + chunk_size]
        pieces.append(checkpoint(chunk, x, y, use_reentrant=False)
                      if x.requires_grad else chunk(x, y))
    if not pieces:
        raise ValueError("empty generated response")
    return torch.cat(pieces)


def grpo_loss(new_logprobs: torch.Tensor, old_logprobs: torch.Tensor,
              rollout_logprobs: torch.Tensor, advantage: torch.Tensor,
              *, reference_logprobs: torch.Tensor | None = None,
              clip_low: float = 0.2, clip_high: float = 0.28,
              tis_clip: float = 2.0, kl_coef: float = 0.0,
              kl_type: str = "low_var_kl"):
    """One sample mean; caller averages samples, not all tokens in the batch."""
    if new_logprobs.ndim != 1 or not new_logprobs.numel():
        raise ValueError("response logprobs must be a nonempty vector")
    for value in (new_logprobs, old_logprobs, rollout_logprobs):
        if value.shape != new_logprobs.shape or not torch.isfinite(value).all():
            raise ValueError("nonfinite or misaligned token logprobs")
    if advantage.numel() != 1 or not torch.isfinite(advantage).all():
        raise ValueError("one finite advantage required per response")
    if kl_coef < 0 or not 0 <= clip_low < 1 or clip_high < 0 or tis_clip <= 0:
        raise ValueError("invalid objective coefficients")
    old = old_logprobs.detach()
    upstream, _ = official_modules()
    policy, clip_fraction = upstream.compute_policy_loss(
        old - new_logprobs, advantage.expand_as(new_logprobs), clip_low, clip_high,
    )
    tis = (old - rollout_logprobs.detach()).exp().clamp(0, tis_clip)
    policy = policy * tis
    kl = torch.zeros_like(policy)
    if kl_coef:
        if reference_logprobs is None or reference_logprobs.shape != policy.shape:
            raise ValueError("fixed SFT reference required for KL")
        if not torch.isfinite(reference_logprobs).all():
            raise ValueError("nonfinite reference logprobs")
        if kl_type not in {"k1", "low_var_kl"}:
            raise ValueError("unsupported KL estimator")
        kl = upstream.compute_approx_kl(
            new_logprobs, reference_logprobs.detach(), kl_type,
        )
    loss = (policy + kl_coef * kl).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite GRPO loss")
    return loss, {
        "policy_loss": float(policy.detach().mean()),
        "kl": float(kl.detach().mean()),
        "clip_fraction": float(clip_fraction.detach().mean()),
        "tis": float(tis.mean()),
    }
