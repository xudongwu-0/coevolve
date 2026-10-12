"""Native target audits and paired question-level, not training-seed, uncertainty."""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .common import read_json, write_json


NATIVE_TARGET_ROUNDOFF_ATOL = 1e-6


def validate_terminal_target_range(low: float, high: float, *, minimum_reward: float = 0.0) -> None:
    # FP32 GAE subtraction/addition can leave tiny residuals at a reward boundary.
    if not (
        math.isfinite(low) and math.isfinite(high)
        and minimum_reward - NATIVE_TARGET_ROUNDOFF_ATOL <= low <= high <= 1 + NATIVE_TARGET_ROUNDOFF_ATOL
    ):
        raise ValueError("Native terminal critic targets left the role reward range")


def paired_lower_bound(control: dict, candidate: dict, *, alpha: float, seed=42,
                       draws=4000) -> float:
    """One-sided paired question bootstrap bound for conservative model selection."""
    if not 0 < alpha < 0.5:
        raise ValueError("Invalid bootstrap alpha")
    grouped = []
    control_rows, candidate_rows = defaultdict(list), defaultdict(list)
    for row in control["results"]:
        control_rows[row["id"]].append(float(row["reward"]))
    for row in candidate["results"]:
        candidate_rows[row["id"]].append(float(row["reward"]))
    if control_rows.keys() != candidate_rows.keys():
        raise ValueError("Selection evaluations do not contain identical questions")
    for key in sorted(control_rows):
        if len(control_rows[key]) != len(candidate_rows[key]):
            raise ValueError("Selection sample counts differ")
        grouped.append(float(np.mean(candidate_rows[key]) - np.mean(control_rows[key])))
    delta = np.asarray(grouped)
    boot = np.random.default_rng(seed).choice(delta, size=(draws, len(delta)), replace=True).mean(1)
    return float(np.quantile(boot, alpha))


def paired_difference(control: dict, candidate: dict, *, seed=42, draws=10000) -> dict:
    if draws < 100:
        raise ValueError("At least 100 bootstrap draws are required")
    if (control["metric"], control["samples_per_question"]) != (
        candidate["metric"], candidate["samples_per_question"]
    ):
        raise ValueError("Evaluation sample protocols differ")

    def question_means(result):
        groups = defaultdict(list)
        seen = set()
        for row in result["results"]:
            if "sample_index" in row and "rollout_seed" in row:
                identity = (row["id"], row["sample_index"], row["rollout_seed"])
                if identity in seen:
                    raise ValueError("Duplicate evaluation sample identity")
                seen.add(identity)
            reward = float(row["reward"])
            if not math.isfinite(reward) or reward not in {0.0, 1.0}:
                raise ValueError("Expected binary verifier rewards")
            groups[row["id"]].append(reward)
        if not groups or len(groups) != result["question_count"]:
            raise ValueError("Question count does not match evaluation receipts")
        if any(len(values) != result["samples_per_question"] for values in groups.values()):
            raise ValueError("Incomplete samples for an evaluated question")
        return {key: float(np.mean(values)) for key, values in groups.items()}

    def identities(result):
        if not result["results"] or "sample_index" not in result["results"][0]:
            return None
        return {(row["id"], row["sample_index"], row["rollout_seed"])
                for row in result["results"]}
    if identities(control) is not None and identities(control) != identities(candidate):
        raise ValueError("Paired evaluation sample identities differ")
    left, right = question_means(control), question_means(candidate)
    if left.keys() != right.keys():
        raise ValueError("Paired evaluation requires the same questions")
    delta = np.array([right[key] - left[key] for key in sorted(left)])
    rng = np.random.default_rng(seed)
    boot = rng.choice(delta, size=(draws, len(delta)), replace=True).mean(axis=1)
    return {
        "question_count": len(delta), "metric": control["metric"],
        "control_accuracy": float(np.mean(list(left.values()))),
        "candidate_accuracy": float(np.mean(list(right.values()))),
        "candidate_minus_control": float(delta.mean()),
        "paired_question_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        "improved_questions": int((delta > 0).sum()),
        "regressed_questions": int((delta < 0).sum()),
        "bootstrap_draws": draws, "bootstrap_seed": seed,
        "limitation": "Question-level uncertainty only; not training-seed robustness. "
                      "Development selection scores are exploratory, not confirmatory.",
    }


def difference_in_differences(control_before: dict, control_after: dict,
                              candidate_before: dict, candidate_after: dict,
                              *, seed=42, draws=10000) -> dict:
    """Question-cluster bootstrap of (candidate change) - (control change)."""
    def means(result):
        groups = defaultdict(list)
        for row in result["results"]:
            groups[row["id"]].append(float(row["reward"]))
        return {key: float(np.mean(value)) for key, value in groups.items()}
    cb, ca, mb, ma = map(means, (control_before, control_after,
                                 candidate_before, candidate_after))
    if not cb.keys() == ca.keys() == mb.keys() == ma.keys():
        raise ValueError("Difference-in-differences requires identical questions")
    keys = sorted(cb)
    delta = np.asarray([(ma[k] - mb[k]) - (ca[k] - cb[k]) for k in keys])
    boot = np.random.default_rng(seed).choice(delta, (draws, len(delta)), replace=True).mean(1)
    return {
        "question_count": len(delta),
        "ma_change_minus_sao_change": float(delta.mean()),
        "paired_question_bootstrap_95pct": np.quantile(boot, [0.025, 0.975]).tolist(),
        "ma_improved_more_questions": int((delta > 0).sum()),
        "sao_improved_more_questions": int((delta < 0).sum()),
        "estimand": "(MA_final - MA_initial) - (SAO_final - SAO_initial)",
    }


def summarize_run(run: Path) -> dict:
    spec = read_json(run / "spec.json")
    phases = {}
    for path in sorted(run.glob("[hw]*/metrics.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if not rows:
            raise ValueError(f"Empty optimizer metric log: {path}")
        metrics = [row["metrics"] for row in rows]
        receipts = [read_json(p) for p in (path.parent / "receipts").glob("*.json")]
        timings = {
            key: sum(row.get(key, 0) for row in metrics)
            for key in metrics[0] if key.startswith("timeperf/") and not key.endswith("__count")
        }
        consumed = sum(row["ppo_actor/n_seqs"] for row in metrics)
        phase = {
            "actor_updates": len(rows), "optimizer_sequences": consumed,
            "completed_episode_receipts": len(receipts),
            "completed_replay_receipts": sum(bool(r.get("replayed")) for r in receipts),
            "consumed_task_reward_mean": float(np.mean([
                m["ppo_actor/task_reward/avg"] for m in metrics])),
            "generated_tokens": sum(r.get("generated_tokens", 0) for r in receipts),
            "optimizer_timing_seconds": timings,
            "dis_rejected_token_fraction_step_mean": float(np.mean([
                m["ppo_actor/update/dis/mask_ratio"] for m in metrics])),
            "note": "Completed receipts include asynchronous prefetch; they are not all consumed updates.",
        }
        receipt = path.parent / "trained.json"
        if receipt.exists():
            phase["training_receipt"] = read_json(receipt)
            seconds = phase["training_receipt"]["elapsed_seconds"]
            phase["optimizer_sequences_per_training_second"] = consumed / seconds
        phases[path.parent.name] = phase
    selections = {}
    for path in sorted(run.glob("h*_selection.json")):
        result = read_json(path)
        selected = result["selected"]
        selections[path.stem] = {
            "accuracy": selected["accuracy"], "incumbent_retained": selected["incumbent"],
            "scored_candidates_including_incumbent": len(result["candidates"]),
        }
    return {
        "run": str(run.resolve()), "method": spec["method"],
        "status": read_json(run / "STATUS.json"),
        "complete": (run / "COMPLETE.json").exists(),
        "formal": bool(spec.get("contract")), "phases": phases, "dev_selections": selections,
        "test_scores_opened_by_report": False,
        "sample_ledger": sample_ledger(run),
    }


def verify_phases(run: Path) -> dict:
    from .verify import compare, tensor_files
    checks = {}
    for path in sorted(run.glob("[hw]*/trained.json")):
        phase = path.parent.name
        spec = read_json(run / f"{phase}_spec.json")
        receipt = read_json(path)
        actor, critic = (Path(receipt["checkpoints"][role]) for role in ("actor", "critic"))
        initial_critic = Path(spec["critic_model"])
        # A fresh critic's value head has no pre-training counterpart; verify the shared
        # backbone changed and the saved head exists (compare() rejects nonfinite tensors).
        fresh = not any(name.endswith("score.weight") for name in tensor_files(initial_critic))
        if fresh and not any(name.endswith("score.weight") for name in tensor_files(critic)):
            raise ValueError("Saved critic has no value head")
        checks[phase] = {
            "actor": compare(Path(spec["model"]), actor),
            "critic": compare(initial_critic, critic, require_score=not fresh),
            "fresh_value_head": fresh,
        }
    if not checks:
        raise ValueError("No completed training phases to verify")
    return checks



def expected_sample_ledger(spec: dict) -> dict:
    phases = spec.get("cycles", 1) + 1
    editor_batch = spec.get("editor_batch_size", spec["batch_size"])
    editor_requests = spec["editor_steps"] * editor_batch + editor_batch
    replayed = (int(editor_requests * spec.get("replay_fraction", 0.0))
                * (phases - 1) if spec["method"] == "ma_sao" else 0)
    warmup = spec.get("critic_warmup_steps", 0) if spec["method"] == "ma_sao" else 0
    solver_batch = spec.get("solver_batch_size", spec["batch_size"])
    ledger = {
        "editor_rollouts": phases * editor_requests - replayed,
        "editor_replayed_actions": replayed,
        "editor_proposal_rollouts": phases * spec["proposals"],
        "solver_training_rollouts": spec["solver_steps"] * solver_batch
                                      + (0 if warmup else solver_batch),
        "harness_scoring_rollouts": phases * editor_requests * 2 * spec["reward_questions"],
        "dev_selection_rollouts": phases * (1 + spec["proposals"])
                                  * spec["select_questions"] * spec.get("select_samples", 1),
        "dev_report_rollouts": phases * 2 * (spec["report_questions"]
                                + spec.get("report_train_tail", 0)) * spec["report_samples"],
        "recycled_partner_episodes_reused": (
            warmup * solver_batch * spec.get("warmup_recycled_per_row", 4)
            + int(spec["solver_steps"] * solver_batch
                  * spec.get("solver_recycle_fraction", 0.0))
            if spec["method"] == "ma_sao" else 0),
    }
    ledger["generated_total"] = sum(value for key, value in ledger.items()
                                    if key.endswith("rollouts"))
    return ledger


def verify_sample_ledger(run: Path) -> dict:
    observed, expected = sample_ledger(run), expected_sample_ledger(read_json(run / "spec.json"))
    adjusted = dict(expected)
    overhead = async_overhead(run)
    for key, value in overhead.items():
        adjusted[key] += value
    if observed != adjusted:
        raise ValueError(f"Physical sample ledger mismatch: observed={observed}, expected={expected}")
    write_json(run / "sample_budget_verification.json", {
        "planned": expected, "observed": observed, "unused_async_overhead": overhead,
        "strict_sample_budget_matched": not any(overhead.values()),
        "note": "Unused async work counts toward physical cost, never partner training. "
                "Cancelled in-flight tokens without receipts are not fully metered.",
    })
    return observed


def async_overhead(run: Path) -> dict:
    from .phase_recovery import editor_pair_audit
    spec = read_json(run / "spec.json")
    result = dict(editor_rollouts=0, editor_replayed_actions=0, harness_scoring_rollouts=0)
    for path in run.glob("h[0-9]*_pair_audit.json"):
        audit = editor_pair_audit(run, path.name.removesuffix("_pair_audit.json"), spec)
        for key in result:
            result[key] += audit[key]
    result["generated_total"] = result["editor_rollouts"] + result["harness_scoring_rollouts"]
    return result


def sample_ledger(run: Path) -> dict:
    """Generated samples by purpose. Recycled episodes are reused, never regenerated."""
    def receipts(pattern, role):
        rows = [read_json(p) for p in run.glob(pattern)]
        return [r for r in rows if r.get("role") == role and not r.get("replayed")]
    ledger = {
        "editor_rollouts": len(receipts("h*/receipts/*.json", "editor")),
        "editor_replayed_actions": sum(bool(read_json(p).get("replayed"))
                                       for p in run.glob("h*/receipts/*.json")),
        "editor_proposal_rollouts": sum(len(read_json(p)) for p in run.glob("h*_proposals.json")),
        "solver_training_rollouts": len(receipts("w*/receipts/*.json", "solver")),
        "harness_scoring_rollouts": sum(len(read_json(p)["results"])
                                        for p in run.glob("h*_reward_evals/*.json")),
        "dev_selection_rollouts": sum(len(read_json(p)["results"])
                                      for p in run.glob("h*_select.json")),
        "dev_report_rollouts": sum(len(read_json(p)["results"])
                                   for p in run.glob("h*_report.json")),
        "recycled_partner_episodes_reused": sum(len(read_json(p))
                                                for p in run.glob("w*_recycle.json")),
    }
    overhead = async_overhead(run)
    ledger["editor_rollouts"] += overhead["editor_rollouts"]
    ledger["editor_replayed_actions"] += overhead["editor_replayed_actions"]
    ledger["generated_total"] = sum(v for k, v in ledger.items()
                                    if k.endswith("rollouts"))
    return ledger

def stage_reports(run: Path) -> dict[str, Path]:
    """Map each completed stage to its disjoint-dev report receipt."""
    spec = read_json(run / "spec.json")
    files = {}
    for cycle in range(spec.get("cycles", 1) + 1):
        phase = f"h{cycle + 1}"
        before = run / f"{phase}_before_report.json"
        if not before.exists():
            break
        files["h0w0" if cycle == 0 else f"h{cycle}w{cycle}"] = before
        selection = run / f"{phase}_selection.json"
        if not selection.exists():
            break
        chosen = read_json(selection)["selected"]
        selected = run / f"{phase}_selected_report.json"
        if selected.exists():
            files[f"{phase}w{cycle}"] = selected
        elif chosen["incumbent"] and spec.get("proposal_source") != "final_policy":
            # Legacy runs reused the before-report receipt when retaining the incumbent.
            files[f"{phase}w{cycle}"] = before
        else:
            break
    return files


def compare_runs(control: Path, candidate: Path) -> dict:
    left, right = stage_reports(control), stage_reports(candidate)
    result = {"control": str(control), "candidate": str(candidate), "stages": {}}
    for stage in [s for s in left if s in right]:
        a, b = read_json(left[stage]), read_json(right[stage])
        result["stages"][stage] = paired_difference(a, b)
    return result


def _print_comparison(control: Path, candidate: Path) -> None:
    result = compare_runs(control, candidate)
    for stage, row in result["stages"].items():
        low, high = row["paired_question_bootstrap_95pct"]
        print(f"{stage}: control {row['control_accuracy']:.3f}  candidate {row['candidate_accuracy']:.3f}"
              f"  diff {row['candidate_minus_control']:+.3f}  95% CI [{low:+.3f}, {high:+.3f}]"
              f"  (+{row['improved_questions']}/-{row['regressed_questions']} questions, {row['metric']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--compare-to", type=Path, help="control run for a paired report comparison")
    parser.add_argument("--verify-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.compare_to:
        _print_comparison(args.compare_to, args.run)
        raise SystemExit
    result = summarize_run(args.run)
    if args.verify_checkpoints:
        result["parameter_checks"] = verify_phases(args.run)
    write_json(args.run / "development_report.json", result)
    print(json.dumps({
        "run": result["run"], "complete": result["complete"],
        "phases": {k: v["actor_updates"] for k, v in result["phases"].items()},
        "dev": result["dev_selections"],
        "checkpoints_verified": bool(result.get("parameter_checks")),
    }, indent=2))
