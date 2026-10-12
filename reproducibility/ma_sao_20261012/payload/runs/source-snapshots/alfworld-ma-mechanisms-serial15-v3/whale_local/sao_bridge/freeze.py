"""Freeze the tuned independent-SAO protocol before its single formal run."""
from __future__ import annotations

import argparse
from pathlib import Path

from .budget import freeze_baseline, initial_model_digest, initial_role_models
from .common import file_sha, read_json

# Every quantity that determines how many samples a trial may generate or evaluate.
SAMPLE_BUDGET_KEYS = (
    "cycles", "editor_steps", "editor_batch_size", "solver_steps", "solver_batch_size",
    "reward_questions", "proposals", "proposal_source", "select_questions", "select_samples",
    "report_questions", "report_train_tail", "report_samples", "solver_tokens",
    "editor_tokens", "staleness", "batch_size",
)
BASELINE_KEYS = ("method", "solver_lr", "editor_lr", "editor_temperature", "weight_update_mode")


def contract_from_spec(spec: dict, budget_hours: float, seeds: list[int] | None = None) -> dict:
    protocol = {"split": "test", "samples": 8}
    protocol.update({key: spec[key] for key in SAMPLE_BUDGET_KEYS})
    models = initial_role_models(spec)
    contract = {
        "hardware": "NVIDIA H200", "gpu_count": 8,
        "budget_gpu_seconds": int(budget_hours * 3600 * 8),
        "initial_actor_sha256": initial_model_digest(Path(models["solver"])),
        "data_manifest_sha256": file_sha(Path(spec["data"]) / "manifest.json"),
        "evaluation_protocol": protocol,
        "baseline_hyperparameters": {key: spec[key] for key in BASELINE_KEYS},
        "training_seed": seeds if seeds else spec["seed"],
        "selection_metric": "dev_report_accuracy_at_fixed_sample_budget",
        "formal_baseline_runs": len(seeds) if seeds else 1,
    }
    if "editor_model" in spec:
        contract["initial_editor_sha256"] = (
            contract["initial_actor_sha256"] if models["editor"] == models["solver"]
            else initial_model_digest(Path(models["editor"]))
        )
    return contract


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned-spec", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--budget-hours", type=float, default=4.0,
                        help="generous GPU-time ceiling; the matched quantity is the sample budget")
    args = parser.parse_args()
    digest = freeze_baseline(args.contract, contract_from_spec(read_json(args.tuned_spec), args.budget_hours))
    print("frozen", args.contract, digest)
