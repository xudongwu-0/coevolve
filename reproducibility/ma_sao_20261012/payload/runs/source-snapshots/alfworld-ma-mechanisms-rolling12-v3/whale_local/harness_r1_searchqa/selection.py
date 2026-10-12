"""Paired question-level evidence for accepting cumulative harness patches."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np


def paired_gain(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    seed: int = 42,
    resamples: int = 5000,
    minimum_gain: float = 0.005,
) -> dict[str, Any]:
    if not baseline or resamples < 100 or minimum_gain < 0:
        raise ValueError("Paired comparison requires nonempty results and a valid resampling budget")

    def index(results: list[dict[str, Any]]) -> dict[tuple[str, int], float]:
        indexed: dict[tuple[str, int], float] = {}
        for result in results:
            key = (str(result["task_id"]), int(result["rollout_seed"]))
            if key in indexed:
                raise ValueError("Duplicate task/seed pair in evaluation")
            value = float(result["reward"])
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError("SearchQA paired reward must be binary")
            indexed[key] = value
        return indexed

    before, after = index(baseline), index(candidate)
    if before.keys() != after.keys():
        raise ValueError("Baseline and candidate task/seed identities differ")
    groups: dict[str, list[float]] = defaultdict(list)
    for key in before:
        groups[key[0]].append(after[key] - before[key])
    if len(groups) < 2:
        raise ValueError("Question-level uncertainty requires at least two questions")
    sample_counts = {len(values) for values in groups.values()}
    if len(sample_counts) != 1:
        raise ValueError("Each paired question must have the same number of samples")
    differences = np.array([np.mean(groups[key]) for key in sorted(groups)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    bootstraps = rng.choice(differences, size=(resamples, len(differences))).mean(axis=1)
    lower, upper = np.quantile(bootstraps, [0.025, 0.975])
    gain = float(differences.mean())
    return {
        "baseline_mean": sum(before.values()) / len(before),
        "candidate_mean": sum(after.values()) / len(after),
        "absolute_gain": gain,
        "paired_bootstrap_ci95": [float(lower), float(upper)],
        "question_count": len(groups),
        "samples_per_question": next(iter(sample_counts)),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "minimum_absolute_gain": minimum_gain,
        "accepted": bool(gain >= minimum_gain and lower > 0),
        "uncertainty_unit": "paired_question_mean",
    }
