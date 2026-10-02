"""Deadline-bound joint MA-SAO training followed by two-node development reporting."""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL
from .alfworld_matched_grpo_eval import BASE_PROTOCOL, INITIAL_STATE
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .joint_experiment import run_role, validate_spec
from .joint_health import request_stop
from .launch import Launcher
from .resources import gpu_release_audit


PROTOCOL = "joint-ma-sao-async-v1"


def prepare(root: Path, comparison: Path, deadline: float) -> dict:
    if deadline - time.time() < 3600:
        raise ValueError("Joint campaign needs an explicit final-hour evaluation reserve")
    base = ROOT / "runs/alfworld_joint_ma_pilot_v1_r2"
    source = ROOT / "runs/alfworld_joint_ma_pilot_v1/data.json"
    old_rows = read_json(source)
    cohort = read_json(EVAL_PROTOCOL)
    validate_cohort(cohort)
    steps, batch = 128, 8
    rows = []
    for i in range(steps * batch):
        row = copy.deepcopy(old_rows[i % len(old_rows)])
        row.update(id=f"async-{i:05d}-task{row['task_id']}",
                   joint_version=i // batch, joint_slot=i % batch)
        if row["task_id"] in cohort["all_task_ids"]:
            raise ValueError("Joint training intersects the frozen 200-question cohort")
        rows.append(row)
    data = root / "data.json"
    _immutable_json(data, rows)
    protocol = read_json(BASE_PROTOCOL)
    protocol.update(
        protocol=PROTOCOL, run=str(root), comparison=str(comparison),
        report_deadline_unix=deadline, training_deadline_unix=deadline - 3600,
        initial_state=str(INITIAL_STATE),
        source_data_sha256=file_sha(source), data_sha256=file_sha(data),
        budget={
            "common_executed_trajectories": 832,
            "batch_per_role": batch, "maximum_rounds": steps,
            "matched_successful_h2_rounds": 12,
            "matched_h2_plus_rejected_w2_rounds": 52,
            "accounting": "Consumed H+W trajectories, including critic warmup; not matched FLOPs/tokens.",
            "generation": "One-round prefetch and incomplete episodes are separately recorded, not free.",
        },
        selection={
            "matched": "Exactly round12, no checkpoint choice; best of eight new harness proposals.",
            "best": "Up to last three common committed checkpoint pairs; selection24 mean@2 only.",
            "harness": "New Engineer proposals only; no substitution by incumbent h2.",
            "report": "Freeze selection before a fresh single sample on the same all200/strict197.",
        },
    )
    _immutable_json(root / "protocol.json", protocol)
    for role in ("editor", "solver"):
        spec = read_json(base / f"{role}_spec.json")
        spec.update(
            protocol=PROTOCOL, run=str(root / role / "optimizer"),
            joint_root=str(root / "joint"), experiment=root.name, steps=steps,
            data=str(data), staleness=1, max_head_offpolicyness=1, joint_max_lag=1,
            value_normalization=True, kl_ctl=0.005, reference_model=spec["model"],
            critic_warmup_steps=2, critic_updates_per_step=2, save_interval_steps=4,
            training_deadline_unix=deadline - 3600,
            lr=5e-7, critic_lr=2.5e-6,
        )
        spec["pinned_files"] = {
            str(path): file_sha(path) for path in (data, INITIAL_STATE, EVAL_PROTOCOL)
        }
        validate_spec(spec)
        _immutable_json(root / f"{role}_spec.json", spec)
    return protocol


def wait_comparison(protocol: dict) -> None:
    comparison = Path(protocol["comparison"])
    while not all((comparison / name).exists()
                  for name in ("COMPLETE.json", "w0_RELEASED.json", "w1_RELEASED.json")):
        for failed in comparison.glob("*FAILED.json"):
            raise RuntimeError(f"Required GRPO comparison failed: {read_json(failed)}")
        if time.time() >= protocol["training_deadline_unix"] - 1800:
            raise TimeoutError("GRPO comparison left less than 30 minutes for joint training")
        time.sleep(10)
    if read_json(comparison / "COMPLETE.json")["status"] != "completed":
        raise ValueError("Required GRPO comparison is incomplete")


def train(root: Path, role: str) -> dict:
    protocol = read_json(root / "protocol.json")
    try:
        wait_comparison(protocol)
        result = run_role(root / f"{role}_spec.json")
        # Only this campaign's new checkpoint tree is shared across node UIDs.
        optimizer = root / role / "optimizer"
        for parent, dirs, files in os.walk(optimizer):
            Path(parent).chmod(Path(parent).stat().st_mode | 0o055)
            for name in files:
                path = Path(parent) / name
                if not path.is_symlink():
                    path.chmod(path.stat().st_mode | 0o044)
        _immutable_json(root / f"{role}_SHARED.json", {"ready": True, "result": result})
        return result
    except BaseException as exc:
        write_json(root / f"{role}_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise


def available_pairs(root: Path) -> tuple[dict, dict]:
    protocol = read_json(root / "protocol.json")
    while not all((root / f"{role}_SHARED.json").is_file() for role in ("editor", "solver")):
        for failed in root.glob("*FAILED.json"):
            raise RuntimeError(f"Joint campaign failed before model publication: {read_json(failed)}")
        if time.time() >= protocol["report_deadline_unix"]:
            raise TimeoutError("Joint models were not published before the reporting deadline")
        time.sleep(5)
    results = {role: read_json(root / f"{role}_SHARED.json")["result"]
               for role in ("editor", "solver")}
    steps = sorted(set(results["editor"]["checkpoint_history"])
                   & set(results["solver"]["checkpoint_history"]), key=int)
    if not steps:
        raise RuntimeError("No same-round committed pair exists")
    return {step: {role: results[role]["checkpoint_history"][step]
                   for role in results} for step in steps}, results


def select_pair(reports: dict) -> tuple[str, str]:
    choices = [(step, label, report["accuracy"])
               for step, variants in reports.items() for label, report in variants.items()
               if label.startswith("proposal")]
    if not choices:
        raise ValueError("No new-harness candidate was evaluated")
    step, label, _ = min(choices, key=lambda row: (-row[2], int(row[0]), row[1]))
    return step, label


def evaluate(root: Path, label: str) -> dict:
    if label not in ("matched", "best"):
        raise ValueError("Unknown joint reporting arm")
    output = root / label
    launch = None
    try:
        pairs, training = available_pairs(root)
        steps = ["11"] if label == "matched" else list(pairs)[-3:]
        if any(step not in pairs for step in steps):
            result = {"status": "unavailable", "reason": "Stopped before the matched round12 checkpoint",
                      "training": training}
            _immutable_json(root / f"{label}_COMPLETE.json", result)
            return result
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Joint evaluation requires released GPUs and active filler")
        protocol, state = read_json(root / "protocol.json"), read_json(INITIAL_STATE)
        cohort = read_json(EVAL_PROTOCOL)
        validate_cohort(cohort)
        launch = Launcher(output / "supervision")
        choices = {step: pairs[step]["editor"] for step in steps}
        proposals = generate_proposals(launch, protocol, state, choices, output / "generation")
        reports, harnesses = {}, {}
        for step in steps:
            paths = {}
            for proposal in proposals[step]:
                name = f"proposal{proposal['sample_index']:02d}"
                path = Path(protocol["no_patch"])
                if proposal["valid"]:
                    path = output / "patches" / f"{step}-{name}.json"
                    _immutable_json(path, proposal["patch"])
                paths[name] = str(path)
            harnesses[step] = paths
            config = matrix_config(protocol, pairs[step]["solver"]["actor"], f"ma-step{step}",
                                   paths, output / f"selection-{step}")
            matrix, _ = run_matrix(launch, config, output / f"selection-{step}", 1)
            reports[step] = matrix_reports(matrix, protocol)
        step, patch = select_pair(reports)
        selection = {
            "global_step": int(step), "rounds": int(step) + 1, "patch_label": patch,
            "harness": harnesses[step][patch], "pair": pairs[step],
            "scores": {s: {name: r["accuracy"] for name, r in variants.items()}
                       for s, variants in reports.items()},
            "valid_proposals": {s: sum(p["valid"] for p in proposals[s]) for s in steps},
            "incremental_consumed_h_plus_w": (int(step) + 1) * 16,
            "cumulative_consumed_h_plus_w": 832 + (int(step) + 1) * 16,
            "actual_campaign_consumed": sum(r["consumed_policy_trajectories"] for r in training.values()),
            "actual_campaign_generated_complete": sum(r["generated_policy_trajectories"] for r in training.values()),
            "sampled_engineer_actions_including_unrewarded": training["editor"]["sampled_engineer_actions"],
            "frozen_at_unix": time.time(),
        }
        _immutable_json(output / "SELECTED.json", selection)
        config = {
            **matrix_config(protocol, pairs[step]["solver"]["actor"], f"ma-step{step}",
                            {"joint_harness": selection["harness"]}, output / "evaluation200"),
            "task_ids": cohort["all_task_ids"], "samples": 1, "seed": cohort["seed"],
            "temperature": cohort["temperature"], "solver_tokens": cohort["solver_tokens"],
            "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 2,
        }
        matrix, _ = run_matrix(launch, config, output / "evaluation200", 1)
        summaries, shared = {}, {}
        for variant, row in matrix["variants"].items():
            report = checked_native_report(read_json(Path(row["report"])), cohort["all_task_ids"], {
                "samples": 1, "seed": cohort["seed"], "solver_tokens": cohort["solver_tokens"],
            })
            compact = {**report, "results": [
                {key: row[key] for key in ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
                for row in report["results"]
            ]}
            path = root / f"{label}_{variant}_report.json"
            _immutable_json(path, compact)
            shared[variant] = {"path": str(path), "sha256": file_sha(path)}
            summaries[variant] = {
                name: {"accuracy": subset_report(report, ids)["accuracy"],
                       "successes": sum(r["reward"] for r in subset_report(report, ids)["results"])}
                for name, ids in (("all200", cohort["all_task_ids"]),
                                  ("strict197", cohort["strict_goal_disjoint_task_ids"]))
            }
        candidate = read_json(Path(shared["joint_harness"]["path"]))
        baseline = read_json(ROOT / "runs/alfworld_core_200_oneshot_v2/merged/w1_h2.json")
        result = {
            "status": "completed", "selection": selection, "scores": summaries, "reports": shared,
            "versus_w1_h2": paired_difference(baseline, candidate, seed=9000045),
            "training": training, "finished_at_unix": time.time(),
            "met_report_deadline": time.time() <= protocol["report_deadline_unix"],
            "limitations": [
                "Exploratory single seed; all200 is a reused development cohort, not a sealed test.",
                "strict197 excludes goal-family overlap, not development selection exposure.",
                "Best-arm checkpoint and harness selection costs are not free.",
                "Matching refers to consumed H+W trajectories, not tokens/FLOPs or prefetched generations.",
                "MAPPO-inspired joint CTDE/ValueNorm/PPO; native token GAE and SAO TIS/DIS retained.",
                "Clipped teammate trajectory correction is an experimental biased extension, not standard MAPPO.",
            ],
        }
    except BaseException as exc:
        write_json(root / f"{label}_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
        write_json(root / f"{label}_RELEASED.json", gpu_release_audit())
    _immutable_json(root / f"{label}_COMPLETE.json", result)
    return result


def guard(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    while time.time() < protocol["training_deadline_unix"]:
        if all((root / f"{label}_COMPLETE.json").exists() for label in ("matched", "best")):
            break
        time.sleep(5)
    request_stop(root / "joint", {"reason": "persistent_final_hour_reserve"})
    while time.time() < protocol["report_deadline_unix"]:
        if all((root / f"{label}_COMPLETE.json").exists() for label in ("matched", "best")):
            break
        time.sleep(5)
    results = {}
    for label in ("matched", "best"):
        path = root / f"{label}_COMPLETE.json"
        failed = root / f"{label}_FAILED.json"
        results[label] = read_json(path) if path.exists() else {
            "status": "failed" if failed.exists() else "incomplete_at_deadline",
            "detail": read_json(failed) if failed.exists() else "No complete 200-question result.",
        }
    result = {"status": "completed" if all(r["status"] == "completed" for r in results.values())
              else "incomplete", "results": results, "reported_at_unix": time.time()}
    _immutable_json(root / "DEADLINE_REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate", "guard"), required=True)
    parser.add_argument("--role", choices=("editor", "solver", "matched", "best"))
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if args.comparison is None or args.deadline is None:
            parser.error("prepare needs --comparison and --deadline")
        result = prepare(root, args.comparison.resolve(), args.deadline)
    elif args.mode == "train":
        if args.role not in ("editor", "solver"):
            parser.error("train needs an editor/solver role")
        result = train(root, args.role)
    elif args.mode == "evaluate":
        result = evaluate(root, args.role)
    else:
        result = guard(root)
    print(result, flush=True)


if __name__ == "__main__":
    main()
