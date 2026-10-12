"""CPU-importable original Coevolve outcome-GRPO objective."""
import numpy as np
import torch


def advantages(rewards, groups):
    result = np.zeros(len(rewards), dtype=np.float64)
    summary = []
    for group in dict.fromkeys(groups):
        indices = [i for i, g in enumerate(groups) if g == group]
        values = np.asarray([rewards[i] for i in indices], dtype=np.float64)
        mean = float(values.mean())
        std = float(values.std(ddof=0))
        if std > 1e-8:
            result[indices] = (values - mean) / std
        summary.append(dict(group=group, indices=indices, rewards=values.tolist(),
                            mean=mean, population_std=std, zero_variance=std <= 1e-8))
    return result, summary


def clipped_loss(logp, old, rollout, advantage):
    ratio = torch.exp(logp - old)
    correction = torch.exp(old - rollout).clamp(max=2).detach()
    raw = ratio * advantage
    clipped = ratio.clamp(0.8, 1.28) * advantage
    return -(correction * torch.minimum(raw, clipped)).mean(), ratio, correction
