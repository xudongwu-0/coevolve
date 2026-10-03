"""Deadline-bound joint MA-SAO training followed by two-node development reporting."""
from __future__ import annotations

import argparse
import copy
import fcntl
from pathlib import Path
import random
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_matched_grpo import EVAL_PROTOCOL, SAO_CONFIG
from .alfworld_reward_service import validate_metadata
from .alfworld_matched_grpo_eval import BASE_PROTOCOL, INITIAL_STATE
from .alfworld_strong_phase import checked_native_report
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .joint_experiment import run_role, validate_spec
from .joint_health import metric_rows, recover_checkpoints, request_stop
from .launch import Launcher
from .resources import gpu_release_audit


PROTOCOL = "joint-ma-sao-async-v1"


def initial_inputs(root: Path, steps: int, batch: int) -> tuple[list[dict], dict]:
    from .alfworld_train_data import evaluation_paths

    original = read_json(SAO_CONFIG)
    packets_path = Path(original["data"]).parent / "packets.json"
    packets = read_json(packets_path)
    by_task = {}
    for packet in packets:
        metadata = packet["metadata"]
        validate_metadata(metadata, original["solver_version"])
        for task in metadata["task_ids"]:
            if task in by_task or not 0 <= task < 240:
                raise ValueError("Initial W0 evidence has duplicated or nontraining tasks")
            by_task[task] = packet
    if set(by_task) != set(range(240)):
        raise ValueError("Initial W0 evidence must cover exactly the original train0-239")
    paths = evaluation_paths()
    order = list(range(240))
    random.Random(1000042).shuffle(order)
    rows = []
    for i in range(steps * batch):
        task = order[i % len(order)]
        messages = copy.deepcopy(by_task[task]["messages"])
        evidence = "\n".join(m["content"] for m in messages if m["role"] == "user")
        rows.append({
            "id": f"h0w0-{i:05d}-task{task}", "joint_version": i // batch,
            "joint_slot": i % batch, "task_id": task, "task_split": "test_set",
            "messages": messages, "task_context": paths[task],
            "critic_evidence": "Historical W0 training evidence, not current outcomes:\n" + evidence[-1200:],
        })
    state = {
        "engineer": original["engineer_model"], "engineer_critic": original["engineer_model"],
        "solver": original["solver_model"], "solver_critic": original["solver_model"],
        "solver_version": original["solver_version"], "solver_epoch_version": 0,
        "engineer_version": 0, "harness": original["harness"],
        "packets": str(packets_path), "packets_sha256": file_sha(packets_path),
        "initialization": "Original official H0 and W0; no H1/W1 weights or evidence.",
        "original_sao_config_sha256": file_sha(SAO_CONFIG),
    }
    _immutable_json(root / "INITIAL_STATE.json", state)
    return rows, state


def prepare(root: Path, comparison: Path, deadline: float, *, from_initial: bool = False,
            memory_check: Path | None = None) -> dict:
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
    if from_initial:
        if memory_check is None:
            raise ValueError("H0/W0 training requires an explicit full-context memory-check receipt")
        rows, state = initial_inputs(root, steps, batch)
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
    if from_initial:
        protocol.update(
            starting_point="h0_w0", initial_state=str(root / "INITIAL_STATE.json"),
            source_data_sha256=state["packets_sha256"], memory_check=str(memory_check),
            initialization_scope="Official H0 is already RL-trained; neither model is a bare base model.",
        )
        protocol["budget"].update(
            common_executed_trajectories=0, matched_rounds=64,
            matched_policy_trajectories=1024,
            checkpoint_trajectories=[192, 832, 1024],
        )
        del protocol["budget"]["matched_successful_h2_rounds"]
        del protocol["budget"]["matched_h2_plus_rejected_w2_rounds"]
        protocol["selection"]["matched"] = "Exactly round64: 1024 total H+W trajectories from H0/W0."
        protocol["selection"]["initial"] = "Unchanged H0/W0: eight W0-evidence proposals, selection24 then all200."
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
        state_path = Path(protocol["initial_state"])
        if from_initial:
            model = state["engineer" if role == "editor" else "solver"]
            spec.update(
                model=model, critic_model=model, reference_model=model,
                critic_warmup_steps=5, phase_timeout_seconds=8 * 3600,
            )
        spec["pinned_files"] = {
            str(path): file_sha(path) for path in (data, state_path, EVAL_PROTOCOL)
        }
        if from_initial:
            spec["pinned_files"][state["packets"]] = state["packets_sha256"]
        validate_spec(spec)
        _immutable_json(root / f"{role}_spec.json", spec)
    return protocol


def wait_comparison(protocol: dict) -> None:
    if protocol.get("starting_point") == "h0_w0":
        root = Path(protocol["run"])
        memory = Path(protocol["memory_check"])
        while not (root / "initial_COMPLETE.json").exists() or not memory.exists():
            for path in (root / "initial_FAILED.json", memory.parent / "MEMORY_FAILED.json"):
                if path.exists():
                    raise RuntimeError(f"H0/W0 prerequisite failed: {read_json(path)}")
            if time.time() >= protocol["training_deadline_unix"]:
                raise TimeoutError("Initial evaluation or full-context memory validation missed training deadline")
            time.sleep(10)
        check = read_json(memory)
        if (check["status"] != "passed" or check["max_sequence_tokens"] < 32768
                or check["actor_optimizer_steps"] < 1 or not check["finite_nonzero_actor_gradient"]):
            raise ValueError("H0/W0 requires a passed full-context native Actor optimizer step")
        if read_json(root / "initial_COMPLETE.json")["status"] != "completed":
            raise ValueError("H0/W0 initial evaluation is incomplete")
        with (root / "TRAINING_PREREQUISITES.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            _immutable_json(root / "TRAINING_PREREQUISITES.json", {
                "status": "passed", "initial_evaluation_sha256": file_sha(root / "initial_COMPLETE.json"),
                "memory_check": str(memory), "memory_check_sha256": file_sha(memory),
            })
        return
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


def share_checkpoints(optimizer: Path, result: dict) -> None:
    from .domain_campaign import public_tree

    optimizer = optimizer.resolve()
    for pair in result["checkpoint_history"].values():
        for value in pair.values():
            checkpoint = Path(value).resolve()
            if not checkpoint.is_relative_to(optimizer):
                raise ValueError("Cannot change permissions outside this role's optimizer run")
            public_tree(checkpoint)
            for parent in checkpoint.parents:
                if not parent.is_relative_to(optimizer):
                    break
                parent.chmod(parent.stat().st_mode | 0o001)


def train(root: Path, role: str) -> dict:
    protocol = read_json(root / "protocol.json")
    try:
        wait_comparison(protocol)
        result = run_role(root / f"{role}_spec.json")
        share_checkpoints(root / role / "optimizer", result)
        _immutable_json(root / f"{role}_SHARED.json", {"ready": True, "result": result})
        return result
    except BaseException as exc:
        write_json(root / f"{role}_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        if protocol.get("starting_point") == "h0_w0":
            try:
                spec = read_json(root / f"{role}_spec.json")
                recovered = recover_failed_role(spec)
                share_checkpoints(Path(spec["run"]), recovered)
                _immutable_json(root / f"{role}_SHARED.json", {"ready": True, "result": recovered})
            except (OSError, ValueError, RuntimeError) as recovery_error:
                write_json(root / f"{role}_RECOVERY_FAILED.json", {
                    "error": repr(recovery_error), "training_error": repr(exc),
                })
        raise


def recover_failed_role(spec: dict) -> dict:
    optimizer, joint = Path(spec["run"]), Path(spec["joint_root"])
    rows = metric_rows(optimizer / "metrics.jsonl")
    history = recover_checkpoints(spec, rows)
    return {
        "status": "recovered_committed_checkpoints_after_training_failure",
        "role": spec["role"], "steps": len(rows),
        "actor_updates": max(0, len(rows) - spec["critic_warmup_steps"]),
        "critic_updates": len(rows) * spec["critic_updates_per_step"],
        "checkpoint_history": history,
        "checkpoints": history[max(history, key=int)],
        "metrics_sha256": file_sha(optimizer / "metrics.jsonl"),
        "consumed_policy_trajectories": len(rows) * spec["batch_size"],
        "generated_policy_trajectories": sum(1 for _ in joint.glob(f"round-*/{spec['role']}-receipt-*.json")),
        "sampled_engineer_actions": sum(1 for _ in joint.glob("round-*/proposal-*.json")),
        "optimizer_resets_between_rounds": False,
        "failed_training_not_relabelled_completed": True,
    }


def available_pairs(root: Path) -> tuple[dict, dict]:
    protocol = read_json(root / "protocol.json")
    while not all((root / f"{role}_SHARED.json").is_file() for role in ("editor", "solver")):
        for failed in root.glob("*FAILED.json"):
            if (protocol.get("starting_point") == "h0_w0"
                    and failed.name in ("editor_FAILED.json", "solver_FAILED.json")):
                continue
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
    if label not in ("initial", "matched", "best"):
        raise ValueError("Unknown joint reporting arm")
    output = root / label
    launch = None
    try:
        protocol = read_json(root / "protocol.json")
        state = read_json(Path(protocol["initial_state"]))
        if label == "initial":
            if protocol.get("starting_point") != "h0_w0":
                raise ValueError("Initial evaluation is only defined for the H0/W0 campaign")
            pairs = {"-1": {
                "editor": {"actor": state["engineer"], "critic": state["engineer_critic"]},
                "solver": {"actor": state["solver"], "critic": state["solver_critic"]},
            }}
            training = {role: {
                "status": "untrained_initialization", "consumed_policy_trajectories": 0,
                "generated_policy_trajectories": 0, "sampled_engineer_actions": 0,
            } for role in ("editor", "solver")}
        else:
            pairs, training = available_pairs(root)
        matched_rounds = protocol["budget"].get("matched_rounds", 12)
        steps = [str(matched_rounds - 1)] if label == "matched" else list(pairs)[-3:]
        if any(step not in pairs for step in steps):
            result = {"status": "unavailable", "reason": f"Stopped before the matched round{matched_rounds} checkpoint",
                      "training": training}
            _immutable_json(root / f"{label}_COMPLETE.json", result)
            return result
        audit = gpu_release_audit()
        if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
            raise RuntimeError("Joint evaluation requires released GPUs and active filler")
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
            "cumulative_consumed_h_plus_w": protocol["budget"]["common_executed_trajectories"] + (int(step) + 1) * 16,
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
        if protocol.get("starting_point") == "h0_w0" and label != "initial":
            initial = read_json(root / "initial_joint_harness_report.json")
            result["versus_initial_h0_w0"] = paired_difference(initial, candidate, seed=9000046)
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
    if any((root / f"{role}_FAILED.json").exists() for role in ("editor", "solver")):
        result["status"] = "incomplete"
    _immutable_json(root / "DEADLINE_REPORT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("prepare", "train", "evaluate", "guard"), required=True)
    parser.add_argument("--role", choices=("editor", "solver", "initial", "matched", "best"))
    parser.add_argument("--comparison", type=Path)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--from-initial", action="store_true")
    parser.add_argument("--memory-check", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        if args.comparison is None or args.deadline is None:
            parser.error("prepare needs --comparison and --deadline")
        result = prepare(root, args.comparison.resolve(), args.deadline,
                         from_initial=args.from_initial, memory_check=args.memory_check)
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
