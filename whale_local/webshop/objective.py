"""Response-only on-policy GRPO with optional successful-context distillation."""
from __future__ import annotations

import math
import torch


def policy_loss(student_log_probs, old_log_probs, response_mask, advantages, *,
                teacher_log_probs=None, teacher_mask=None, coefficient=0.5,
                gap_clip=5.0, ppo_clip=0.2, reference_log_probs=None,
                reference_kl_coefficient=0.0):
    """Return one loss per trajectory, not per action turn.

    Inputs have shape [episode, concatenated generated action tokens]. Prompts,
    observations and padding must have mask zero. Teacher scores must describe
    these *same student-sampled tokens*, never a separately sampled teacher action.
    """
    if student_log_probs.ndim != 2:
        raise ValueError("Expected [episode, generated token] log probabilities")
    shape = student_log_probs.shape
    if old_log_probs.shape != shape or response_mask.shape != shape:
        raise ValueError("Behavior log probabilities/masks must align with student tokens")
    if advantages.shape != (shape[0],):
        raise ValueError("One GRPO advantage is required per episode")
    if (not all(math.isfinite(x) for x in (ppo_clip, gap_clip, coefficient, reference_kl_coefficient))
            or not 0 < ppo_clip < 1 or gap_clip <= 0 or coefficient < 0
            or reference_kl_coefficient < 0):
        raise ValueError("Invalid loss coefficients")
    mask = response_mask.bool()
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Each optimized trajectory must contain action tokens")
    for value in (student_log_probs, old_log_probs):
        if not torch.isfinite(value[mask]).all():
            raise ValueError("Nonfinite valid-token log probabilities")
    if not torch.isfinite(advantages).all():
        raise ValueError("Nonfinite episode advantages")
    # Select before arithmetic: zero times NaN/Inf is still NaN.
    student = torch.where(mask, student_log_probs.float(), 0.0)
    old = torch.where(mask, old_log_probs.detach().float(), 0.0)
    advantage = advantages.detach().float()[:, None].expand(shape).clone()
    teacher_tokens = torch.zeros_like(mask)
    gap = torch.zeros_like(student)
    if teacher_log_probs is not None:
        if teacher_mask is None or teacher_mask.shape != shape or teacher_log_probs.shape != shape:
            raise ValueError("Teacher scores and eligibility mask must align")
        teacher_tokens = teacher_mask.bool() & mask
        if not torch.isfinite(teacher_log_probs[teacher_tokens]).all():
            raise ValueError("Nonfinite eligible teacher log probabilities")
        teacher = torch.where(teacher_tokens, teacher_log_probs.detach().float(), 0.0)
        gap = torch.where(teacher_tokens, teacher - old, 0.0).clamp(-gap_clip, gap_clip)
        advantage = advantage + coefficient * gap.detach()
    elif teacher_mask is not None and teacher_mask.bool().any():
        raise ValueError("Teacher mask without teacher probabilities")
    ratio = (student - old).exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1.0 - ppo_clip, 1.0 + ppo_clip) * advantage
    token_loss = -torch.minimum(unclipped, clipped)
    reference_kl = torch.zeros_like(student)
    if reference_log_probs is not None:
        if reference_log_probs.shape != shape or not torch.isfinite(reference_log_probs[mask]).all():
            raise ValueError("Reference probabilities must align and be finite on action tokens")
        reference = torch.where(mask, reference_log_probs.detach().float(), 0.0)
        reference_gap = reference - student
        reference_kl = reference_gap.exp() - reference_gap - 1
        token_loss = token_loss + reference_kl_coefficient * reference_kl
    elif reference_kl_coefficient:
        raise ValueError("Reference KL coefficient requires fixed-reference log probabilities")
    counts = mask.sum(dim=1)
    per_episode = torch.where(mask, token_loss, 0.0).sum(dim=1) / counts
    loss = per_episode.mean()
    with torch.no_grad():
        metrics = {
            "loss": float(loss.detach()),
            "teacher_tokens": int(teacher_tokens.sum()),
            "action_tokens": int(mask.sum()),
            "teacher_gap_mean": float(gap[teacher_tokens].mean()) if teacher_tokens.any() else 0.0,
            "clip_fraction": float(((ratio - 1.0).abs()[mask] > ppo_clip).float().mean()),
            "reference_kl": float(reference_kl[mask].mean()),
        }
    return loss, metrics
