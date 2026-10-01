"""Validation for pre-action centralized-critic context."""
from __future__ import annotations

import json
from pathlib import Path

CONTEXT_KEYS = {
    "phase", "actor_role", "partner_checkpoint", "harness_source",
    "calibration_accuracy", "calibration_split", "partner_version",
}
CONTEXT_TOKEN_BUDGET = 1024


def context_text(context: dict) -> str:
    if set(context) != CONTEXT_KEYS:
        raise ValueError("Joint context must contain exactly the declared pre-action fields")
    if context["calibration_split"] not in {"train", "dev"}:
        raise ValueError("Only pre-action train or legacy development calibration is permitted")
    if context["phase"] not in {"h1", "w", "h2"}:
        raise ValueError("Invalid joint phase")
    if context["actor_role"] not in {"editor", "solver"}:
        raise ValueError("Invalid actor role")
    if not 0 <= float(context["calibration_accuracy"]) <= 1:
        raise ValueError("Invalid calibration accuracy")
    return "Joint pre-action training state:\n" + json.dumps(context, sort_keys=True) + "\n"


def joint_context_audit(context_path: Path, model_path: Path) -> dict:
    from transformers import AutoTokenizer

    context = json.loads(context_path.read_text())
    text = context_text(context)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )
    tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if tokens > CONTEXT_TOKEN_BUDGET:
        raise ValueError(
            f"Joint context uses {tokens} tokens but the reserved budget is "
            f"{CONTEXT_TOKEN_BUDGET}; store a stable harness identity, not full source"
        )
    return {
        "context_path": str(context_path.resolve()),
        "model_path": str(model_path.resolve()),
        "tokens": tokens,
        "token_budget": CONTEXT_TOKEN_BUDGET,
        "phase": context["phase"],
        "actor_role": context["actor_role"],
        "harness_identity": context["harness_source"],
        "calibration_split": context["calibration_split"],
    }
