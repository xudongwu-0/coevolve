"""Frozen post-training factorial diagnosis of the guarded MA continuation."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import run_matrix, supplement_runtime
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference, difference_in_differences
from .common import ROOT, read_json, write_json, file_sha
from .launch import Launcher
from .resources import gpu_release_audit


def prepare(root: Path, campaign: Path) -> dict:
    # Commit the decision rule before any final report exists, not after seeing scores.
    if any((campaign / f"{label}_COMPLETE.json").exists() for label in ("matched", "best")):
        raise ValueError("Prospective study must be prepared before campaign results")
    protocol = read_json(campaign / "protocol.json")
    state = read_json(Path(protocol["initial_state"]))
    cohort = read_json(EVAL_PROTOCOL)
    validate_cohort(cohort)
    baseline_config = Path(cohort["configs"]["w1"])
    baseline = read_json(baseline_config)
    pins = [campaign / "protocol.json", Path(protocol["initial_state"]),
            Path(protocol["evaluation"]["task_manifest"]), EVAL_PROTOCOL,
            baseline_config, Path(baseline["harnesses"]["h2"]), Path(state["harness"]),
            campaign / "RESTART_PROVENANCE.json"]
    result = {
        "protocol": "guarded-ma-factorial-study-v1", "campaign": str(campaign),
        "created_at_unix": time.time(), "queue_timeout_seconds": 8 * 3600,
        "research_question": "Does guarded joint continuation improve the deployed system, and is the change due to weights, harness, or compatibility?",
        "method": "H0/MA64 initialization, low H/W learning rates, KL to branch starts, incumbent patch replacement only above 2/48 development gain.",
        "primary_endpoint": "Existing best-arm selection48-chosen pair; never choose between matched/best using all200.",
        "secondary_endpoint": "Existing round32 probe, reported regardless of its score.",
        "factorial": "Remeasure branch-start and selected Solver with h0 and selected patch; also no-patch. Remeasure W1+h2 under the same runtime.",
        "hypotheses": [
            "Selected system improves on the branch start and historical/fresh W1+h2.",
            "At fixed h0, Solver continuation improves performance.",
            "At fixed selected Solver, the new patch improves over h0.",
            "Patch gain changes between start and selected Solver (compatibility interaction).",
        ],
        "point_success": "Primary all200 accuracy strictly exceeds historical 0.92; also report paired difference versus freshly remeasured W1+h2.",
        "uncertainty": "Question-paired bootstrap95%; an interval including zero is inconclusive, not proven superiority.",
        "splits": "All200 primary, selection48 and its152-question complement separately; all are previously exposed development data, not a sealed test.",
        "limitations": ["Single training seed", "Multiple method components changed together; no single-component causal attribution", "Additional training and selection costs", "No test-result-driven checkpoint or harness substitution"],
        "maximum_environment_episodes": 1600, "new_training_trajectories": 0,
        "initial_solver": state["solver"], "initial_harness": state["harness"],
        "baseline_solver": baseline["solver_model"], "baseline_harness": baseline["harnesses"]["h2"],
        "pinned_files": {str(p): file_sha(p) for p in pins},
    }
    _immutable_json(root / "protocol.json", result)
    return result


def amend_for_recovered_training(root: Path, previous: Path) -> dict:
    """Preserve the original design and explicitly bind a partial-run recovery."""
    protocol = read_json(previous / "protocol.json")
    campaign = Path(protocol["campaign"])
    if any((campaign / f"{label}_COMPLETE.json").exists() for label in ("matched", "best")):
        raise ValueError("Recovery amendment must precede endpoint results")
    pins, counts = {}, {}
    for role in ("editor", "solver"):
        shared = campaign / f"{role}_SHARED.json"
        result = read_json(shared)["result"]
        metrics = campaign / role / "optimizer/metrics.jsonl"
        if (result["status"] != "recovered_committed_checkpoints_after_training_failure"
                or result.get("failed_training_not_relabelled_completed") is not True
                or file_sha(metrics) != result["metrics_sha256"]):
            raise ValueError("Unverified committed checkpoint recovery")
        for path in (shared, metrics, campaign / f"{role}_FAILED.json"):
            pins[str(path)] = file_sha(path)
        counts[role] = {key: result[key] for key in
                       ("steps", "consumed_policy_trajectories", "generated_policy_trajectories")}
    protocol["recovered_training"] = {
        "at_unix": time.time(), "original_protocol": str(previous / "protocol.json"),
        "original_protocol_sha256": file_sha(previous / "protocol.json"),
        "pinned_files": pins, "counts": counts,
        "interpretation": "Amended partial-training analysis; planned 64 rounds were not completed. Original endpoint selection rule retained over committed checkpoints.",
    }
    protocol["pinned_files"].update(pins)
    _immutable_json(root / "protocol.json", protocol)
    return protocol


def wait_campaign(protocol: dict) -> tuple[dict, dict]:
    campaign = Path(protocol["campaign"])
    deadline = time.monotonic() + protocol["queue_timeout_seconds"]
    while True:
        recovery = protocol.get("recovered_training", {})
        for path, digest in recovery.get("pinned_files", {}).items():
            if file_sha(Path(path)) != digest:
                raise RuntimeError("Recovered training receipt changed")
        for role in ("editor", "solver", "matched", "best"):
            if (campaign / f"{role}_FAILED.json").exists():
                if role in ("editor", "solver") and str(campaign / f"{role}_FAILED.json") in recovery.get("pinned_files", {}):
                    continue
                raise RuntimeError(f"Study predecessor failed: {role}")
        if all((campaign / f"{label}_{suffix}.json").exists()
               for label in ("matched", "best") for suffix in ("COMPLETE", "RELEASED")):
            records = {label: read_json(campaign / f"{label}_COMPLETE.json")
                       for label in ("matched", "best")}
            if any(record["status"] != "completed" for record in records.values()):
                raise ValueError("Study requires both prespecified endpoints")
            for label in records:
                release = read_json(campaign / f"{label}_RELEASED.json")
                if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
                    raise RuntimeError("Predecessor did not release GPUs")
            return records["matched"], records["best"]
        if time.monotonic() >= deadline:
            raise TimeoutError("Study predecessor exceeded queue budget")
        time.sleep(10)


def contrasts(reports: dict, task_ids: list[int]) -> dict:
    r = {model: {patch: subset_report(report, task_ids) for patch, report in variants.items()}
         for model, variants in reports.items()}
    pairs = {
        "system_gain": (r["start"]["h0"], r["selected"]["selected_patch"]),
        "weight_gain_fixed_h0": (r["start"]["h0"], r["selected"]["h0"]),
        "weight_gain_no_patch": (r["start"]["no_patch"], r["selected"]["no_patch"]),
        "patch_gain_fixed_selected_weights": (r["selected"]["h0"], r["selected"]["selected_patch"]),
        "patch_gain_fixed_start_weights": (r["start"]["h0"], r["start"]["selected_patch"]),
        "versus_fresh_baseline": (r["baseline"]["h2"], r["selected"]["selected_patch"]),
    }
    result = {name: paired_difference(*pair, seed=20261004 + i)
              for i, (name, pair) in enumerate(pairs.items())}
    interaction = difference_in_differences(
        r["start"]["h0"], r["start"]["selected_patch"],
        r["selected"]["h0"], r["selected"]["selected_patch"], seed=20261014,
    )
    result["compatibility_interaction"] = {
        "difference": interaction["ma_change_minus_sao_change"],
        "paired_question_bootstrap_95pct": interaction["paired_question_bootstrap_95pct"],
        "estimand": "(selected-patch minus h0 at selected weights) minus (selected-patch minus h0 at starting weights)",
    }
    return result


def run(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    launch = None
    try:
        for path, digest in protocol["pinned_files"].items():
            if file_sha(Path(path)) != digest:
                raise ValueError(f"Frozen study input changed: {path}")
        matched, best = wait_campaign(protocol)
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Study requires released GPUs and active filler")
        campaign = Path(protocol["campaign"])
        chosen = best["selection"]
        if chosen != read_json(campaign / "best/SELECTED.json"):
            raise ValueError("Primary model differs from frozen development selection")
        models = {"start": protocol["initial_solver"],
                  "selected": chosen["pair"]["solver"]["actor"],
                  "baseline": protocol["baseline_solver"]}
        patches = {"h0": protocol["initial_harness"], "selected_patch": chosen["harness"]}
        inputs = [campaign / "best/SELECTED.json", campaign / "best_COMPLETE.json",
                  *(Path(p) for p in patches.values())]
        for model in models.values():
            weights = sorted(Path(model).glob("*.safetensors"))
            if not weights:
                raise ValueError(f"No model weights: {model}")
            inputs.extend([Path(model) / "config.json", *weights])
        _immutable_json(root / "BOUND_INPUTS.json", {
            "models": models, "patches": patches,
            "files": {str(p): file_sha(p) for p in inputs},
            "rule": "Existing best SELECTED.json; no all200 outcome selection",
        })
        cohort = read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        launch = Launcher(root / "supervision")
        reports, refs, episodes = {}, {}, 0
        for name, model in models.items():
            supplement_runtime(Path(model), Path(model))
            config = {
                "solver_model": model, "solver_version": f"guarded-ma-study-{name}",
                "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
                "harnesses": {"h2": protocol["baseline_harness"]} if name == "baseline" else patches,
                "task_ids": cohort["all_task_ids"], "samples": 1,
                **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
                "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 3,
            }
            matrix, _ = run_matrix(launch, config, root / name, 1)
            episodes += 200 * len({entry["harness_sha256"] for entry in matrix["variants"].values()})
            reports[name], refs[name] = {}, {}
            for patch, entry in matrix["variants"].items():
                report = checked_native_report(read_json(Path(entry["report"])), cohort["all_task_ids"],
                    {"samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"]})
                compact = {**report, "results": [{k: row[k] for k in
                    ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
                    for row in report["results"]]}
                path = root / "reports" / f"{name}-{patch}.json"
                _immutable_json(path, compact)
                reports[name][patch] = compact
                refs[name][patch] = {"accuracy": report["accuracy"], "path": str(path), "sha256": file_sha(path)}
        manifest = read_json(Path(read_json(campaign / "protocol.json")["evaluation"]["task_manifest"]))
        splits = {"all200": cohort["all_task_ids"], "selection48": manifest["selection_task_ids"],
                  "report152": manifest["report_task_ids"]}
        if set(splits["selection48"]) & set(splits["report152"]) or set(splits["all200"]) != set(splits["selection48"] + splits["report152"]):
            raise ValueError("Development partition is invalid")
        restart = read_json(campaign / "RESTART_PROVENANCE.json")
        result = {
            "status": "completed", "protocol_sha256": file_sha(root / "protocol.json"),
            "primary_campaign_scores": best["scores"], "secondary_round32_scores": matched["scores"],
            "primary_campaign_versus_historical": best["versus_w1_h2"],
            "primary_point_exceeds_92": best["scores"]["joint_harness"]["all200"]["accuracy"] > .92,
            "scores": refs, "contrasts": {name: contrasts(reports, ids) for name, ids in splits.items()},
            "environment_episodes": episodes, "new_training_trajectories": 0,
            "research_training_consumed": restart["parent_actual_consumed"] + restart["failed_attempt_consumed_h_plus_w"] + chosen["actual_campaign_consumed"],
            "incumbent_replaced": False, "finished_at_unix": time.time(),
            "recovered_training": protocol.get("recovered_training"),
            "limitations": protocol["limitations"] + [protocol["splits"]],
        }
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
    _immutable_json(root / "COMPLETE.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--campaign", type=Path)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.campaign is None:
            parser.error("prepare needs --campaign")
        result = prepare(args.root.resolve(), args.campaign.resolve())
    else:
        result = run(args.root.resolve())
    print(result, flush=True)


if __name__ == "__main__":
    main()
