"""Audit recorded generation and allocated node time after both strong arms freeze."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from .alfworld_strong_phase import checked_native_report, checked_release, training_evidence
from .alfworld_strong_report import frozen_states
from .alfworld_train_data import _immutable_json
from .budget import contract_digest
from .common import file_sha, read_json
from .gpu_telemetry import summarize


def manifest_digest(paths: list[Path]) -> str:
    return contract_digest({str(path): file_sha(path) for path in sorted(set(paths))})


def native_generation(optimizer: Path) -> dict:
    ledger_path = optimizer / "consumed_rows.json"
    ledger = read_json(ledger_path)
    declared = ledger["ids"]
    if (
        ledger["rows"] != len(declared) or len(set(declared)) != len(declared)
        or ledger["fresh_ids"] != declared or ledger["replay_ids"]
    ):
        raise ValueError("Strong cost audit requires the exact fresh-only input ledger")
    files = sorted((optimizer / "receipts").glob("*.json"))
    counts, stops, statuses = Counter(), Counter(), Counter()
    tokens = calls = 0
    for path in files:
        receipt = read_json(path)
        if (
            receipt["id"] not in declared or receipt["replayed"]
            or receipt["trajectory_style"] != "concat"
            or receipt["exported_fresh_trajectories"] != 1
            or receipt["recycled_partner_episodes"]
        ):
            raise ValueError(f"Receipt is not an allocated fresh concat episode: {path}")
        if not receipt["calls"]:
            raise ValueError(f"Generation receipt has no native calls: {path}")
        generated = 0
        for call in receipt["calls"]:
            length = len(call["response_ids"])
            if length != len(call["response_logprobs"]) or length != len(call["versions"]):
                raise ValueError(f"Native response/logprob/version accounting differs: {path}")
            generated += length
            stops[str(call["stop_reason"])] += 1
        if generated != receipt["generated_tokens"]:
            raise ValueError(f"Native generated token total differs from recorded calls: {path}")
        tokens += generated
        calls += len(receipt["calls"])
        counts[receipt["id"]] += 1
        if receipt["role"] == "editor":
            statuses[receipt["evaluation"]["evaluation"]["eval_status"]] += 1
        elif receipt["role"] == "solver":
            statuses[receipt["status"]] += 1
        else:
            raise ValueError(f"Unknown native role in generation receipt: {path}")
    missing = sorted(set(declared) - counts.keys())
    if (optimizer.parent / "COMPLETE.json").is_file() and missing:
        raise ValueError("Completed native training omitted receipts for declared inputs")
    return {
        "declared_input_rows": ledger["rows"],
        "recorded_generated_episodes": len(files),
        "unique_generated_input_ids": len(counts),
        "repeated_input_ids": {key: count for key, count in sorted(counts.items()) if count > 1},
        "missing_declared_input_ids": missing,
        "extra_receipts_over_declared_rows": len(files) - ledger["rows"],
        "recorded_generation_calls": calls,
        "recorded_generated_tokens": tokens,
        "stop_reasons": dict(stops), "episode_statuses": dict(statuses),
        "evidence_sha256": manifest_digest([ledger_path, *files]),
        "consumed_behavior_mapping": (
            "The input ledger is declared before optimization. Repeated IDs from prefetch "
            "cannot identify which receipt was consumed; no arbitrary duplicate is selected."
        ),
        "scope": "Completed recorded generations only; canceled/unpersisted work is not recoverable here.",
    }


def reward_work(root: Path) -> dict:
    files = sorted((root / "reward_cache").glob("*/*/reward.json"))
    attempts = episodes = failed_attempts = 0
    for path in files:
        reward = read_json(path)
        history = reward["timing"]["eval_attempts"]
        if not history or history[-1]["patched_n"] != reward["patched_n"]:
            raise ValueError(f"Official reward attempt accounting changed: {path}")
        for attempt in history:
            if type(attempt["patched_n"]) is not int or attempt["patched_n"] < 0:
                raise ValueError(f"Invalid official reward episode count: {path}")
            attempts += 1
            episodes += attempt["patched_n"]
            failed_attempts += attempt["returncode"] != 0
    return {
        "unique_reward_request_receipts": len(list((root / "reward_receipts").glob("*.json"))),
        "unique_completed_reward_cache_entries": len(files),
        "recorded_official_eval_attempts": attempts,
        "recorded_official_episodes": episodes,
        "failed_attempts_in_completed_reward_entries": failed_attempts,
        "evidence_sha256": manifest_digest(files),
        "scope": (
            "Unique completed cache entries, including recorded retries. Invalid edits do not "
            "run the Solver. Failed/in-flight evaluations without reward.json remain uncounted."
        ),
    }


def matrix_work(root: Path) -> dict:
    attempts = []
    for attempt in sorted(root.glob("*/attempt-*")):
        if not attempt.is_dir():
            continue
        config_path = attempt / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Matrix attempt has no frozen configuration: {attempt}")
        config = read_json(config_path)
        paths = sorted((attempt / "reports").glob("*.json"))
        episodes = tokens = 0
        for path in paths:
            report = checked_native_report(read_json(path), config["task_ids"], config)
            episodes += len(report["results"])
            tokens += sum(row["tokens"] for row in report["results"])
        complete = attempt / "COMPLETE.json"
        failed = attempt / "FAILED.json"
        if complete.is_file():
            result = read_json(attempt / "report.json")
            if result["unique_harnesses"] != len(paths):
                raise ValueError("Completed matrix has missing or additional physical reports")
            status = "completed"
        elif failed.is_file():
            status = "failed"
        else:
            raise ValueError(f"Closed arm contains an unfinished matrix attempt: {attempt}")
        attempts.append({
            "root": str(attempt), "status": status,
            "status_receipt_sha256": file_sha(complete if status == "completed" else failed),
            "recorded_unique_harness_reports": len(paths),
            "recorded_episodes": episodes, "recorded_generated_tokens": tokens,
            "evidence_sha256": manifest_digest([config_path, *paths]),
        })
    return {
        "attempts": attempts,
        "recorded_episodes": sum(row["recorded_episodes"] for row in attempts),
        "recorded_generated_tokens": sum(row["recorded_generated_tokens"] for row in attempts),
        "failed_matrix_attempts": sum(row["status"] == "failed" for row in attempts),
        "scope": (
            "Physical per-harness reports, not logical alias labels or published copies. "
            "All attempts count; canceled variants lacking a completed report are a lower-bound gap."
        ),
    }


def proposal_work(root: Path) -> dict:
    files = sorted((root / "jobs").glob("*/attempt-*/response.json"))
    tokens = 0
    for path in files:
        usage = read_json(path)["usage"]
        value = usage["completion_tokens"]
        if type(value) is not int or value < 0:
            raise ValueError(f"Proposal response omitted valid completion token usage: {path}")
        tokens += value
    return {
        "recorded_generation_responses": len(files), "recorded_generated_tokens": tokens,
        "evidence_sha256": manifest_digest(files),
        "scope": "Every saved response, including invalid proposals and retried generation jobs.",
    }


def logged_updates(optimizer: Path) -> dict:
    path = optimizer / "metrics.jsonl"
    if not path.is_file():
        return {"status": "no_metric_file", "scope": "No recorded updates; setup/in-flight work is unknown."}
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b"\n")]
    if [row["global_step"] for row in rows] != list(range(len(rows))):
        raise ValueError("Recorded optimizer steps are not contiguous")
    gradients = [
        row["metrics"]["ppo_actor/update/grad_norm"] for row in rows
        if "ppo_actor/update/grad_norm" in row["metrics"]
    ]
    return {
        "optimizer_steps": len(rows), "actor_steps": len(gradients),
        "nonzero_finite_actor_gradients": sum(math.isfinite(value) and value > 0 for value in gradients),
        "invalid_actor_gradients": sum(not math.isfinite(value) or value < 0 for value in gradients),
        "incomplete_final_line": bool(lines and not lines[-1].endswith(b"\n")),
        "metrics_sha256": file_sha(path),
    }


def allocation_window(start: float, frozen: float, released: float) -> dict:
    if not all(math.isfinite(value) for value in (start, frozen, released)) or not 0 < start <= frozen <= released:
        raise ValueError("Invalid original launch, freeze or final release timestamps")
    return {
        "original_first_phase_launch": start, "arm_frozen_at": frozen,
        "final_report_released_at": released,
        "elapsed_seconds": released - start,
        "allocated_gpu_hours": 8 * (released - start) / 3600,
        "through_arm_freeze_gpu_hours": 8 * (frozen - start) / 3600,
        "freeze_to_report_release_gpu_hours": 8 * (released - frozen) / 3600,
        "scope": (
            "Eight-GPU node allocation, not active compute. Includes startup, recovery, "
            "saving, gaps, peer-arm/report waiting and filler-only intervals."
        ),
    }


def audit_arm(protocol_path: Path, arm: str) -> dict:
    protocol, _, freezes = frozen_states(protocol_path)
    if arm not in freezes:
        raise ValueError("Unknown strong cost-audit arm")
    root = Path(protocol["run"]) / arm
    report_root = root / "final_report"
    if read_json(report_root / "COMPLETE.json")["status"] != "completed":
        raise ValueError("Final report must complete before cost audit")
    release = checked_release(report_root)
    phases, telemetry_paths = {}, []
    for phase in protocol["phase_orders"][arm]:
        phase_root = root / phase
        completion = read_json(phase_root / "PHASE_COMPLETE.json")
        config = read_json(phase_root / "phase_config.json")
        identity = completion["input_identity"]
        phase_freeze = next(row for row in freezes[arm]["phases"] if row["phase"] == phase)
        if (
            file_sha(phase_root / "PHASE_COMPLETE.json") != phase_freeze["native_completion_sha256"]
            or file_sha(root / f"{phase}_selection/COMPLETE.json") != phase_freeze["selection_completion_sha256"]
            or identity["protocol_sha256"] != file_sha(protocol_path)
        ):
            raise ValueError("Phase accounting receipts differ from the final freeze")
        for key, name in (
            ("config_sha256", "phase_config.json"), ("data_sha256", "data.json"),
            ("state_sha256", "input_state.json"),
        ):
            if file_sha(phase_root / name) != identity[key]:
                raise ValueError("Native phase accounting input changed")
        trained, audit = training_evidence(Path(completion["training_root"]), config)
        budget = protocol["h2"] if phase == "h2" else protocol["w"]
        expected_steps = budget["steps"] if phase == "h2" else budget["steps_per_phase"]
        if audit["optimizer_steps"] != expected_steps or config["batch_size"] != budget["batch_size"]:
            raise ValueError("Completed phase cost differs from the protocol budget")
        native = {}
        for attempt in sorted(phase_root.glob("attempt-*")):
            if not attempt.is_dir():
                continue
            ledger = attempt / "optimizer/consumed_rows.json"
            native[attempt.name] = {
                "logged_updates": logged_updates(attempt / "optimizer"),
                "generation": native_generation(attempt / "optimizer") if ledger.is_file() else {
                    "status": "no_input_ledger", "scope": "Startup failure; generated work is unknown.",
                },
                "official_reward_work": reward_work(attempt),
            }
            telemetry = attempt / "gpu_process_samples.jsonl"
            if telemetry.is_file():
                telemetry_paths.append(telemetry)
        selected = root / f"{phase}_selection"
        telemetry_paths.append(selected / "gpu_process_samples.jsonl")
        phases[phase] = {
            "native_audit": audit, "attempts": native,
            "optimizer_consumed_episodes": audit["optimizer_steps"] * config["batch_size"],
            "actor_consumed_episodes": audit["actor_steps"] * config["batch_size"],
            "optimizer_active_seconds": (
                trained["training_active_finished_at_unix"] - trained["training_active_started_at_unix"]
            ),
            "selection_matrices": matrix_work(selected),
            "proposal_generation": proposal_work(selected),
            "selection_decision": read_json(selected / "COMPLETE.json")["decision"],
        }
    report_matrices = matrix_work(report_root)
    telemetry_paths.extend(
        Path(row["root"]) / "gpu_process_samples.jsonl" for row in report_matrices["attempts"]
    )
    telemetry = {str(path): {"file_sha256": file_sha(path), **summarize(path)} for path in telemetry_paths}
    samples = sum(row["valid_samples"] for row in telemetry.values())
    means = {
        key: sum(row[key] * row["valid_samples"] for row in telemetry.values()) / samples
        for key in ("mean_experiment_sm_percent", "mean_filler_sm_percent", "mean_total_sm_percent")
    }
    first_phase = protocol["phase_orders"][arm][0]
    launched = read_json(root / f"launch_receipt_{first_phase}.json")["started_at_unix"]
    failures = sorted((root / "arm_failures").glob("*.json"))
    return {
        "status": "completed", "arm": arm, "protocol_sha256": file_sha(protocol_path),
        "audit_source_sha256": manifest_digest([
            Path(__file__).with_name(name) for name in (
                "alfworld_strong_cost.py", "alfworld_strong_phase.py", "alfworld_strong_report.py",
                "analysis.py", "common.py", "budget.py", "gpu_telemetry.py",
            )
        ]),
        "arm_freeze_sha256": file_sha(root / "ARM_FROZEN.json"),
        "final_report_completion_sha256": file_sha(report_root / "COMPLETE.json"),
        "allocation": allocation_window(launched, freezes[arm]["frozen_at_unix"], release["released_at"]),
        "phases": phases, "final_report_matrices": report_matrices,
        "controller_failure_receipts": {str(path): read_json(path) for path in failures},
        "telemetry": {
            "nonoverlapping_sources": telemetry, "valid_samples": samples, **means,
            "filler_active_in_all_valid_samples": all(row["filler_remained_active"] for row in telemetry.values()),
            "scope": (
                "Sample-weighted pmon, not time-integrated FLOPs or full allocated-window utilization. "
                "Selection-parent telemetry includes its child matrices; those are not counted twice."
            ),
        },
        "common_prefix": protocol["common_prefix"],
        "limitations": [
            "Generation totals are recorded lower bounds, not an assertion that canceled work cost zero.",
            "Full original common-prefix training/search and common H1 selection are additional, charged equally.",
            "Separate Solver-v5 and strong-Solver train probe are R&D, not free common-prefix compute.",
            "The phase stop is a fixed budget, not a convergence proof.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=["sequential", "ma"], required=True)
    args = parser.parse_args()
    result = audit_arm(args.protocol, args.arm)
    path = Path(read_json(args.protocol)["run"]) / args.arm / "COST_AUDIT.json"
    _immutable_json(path, result)
    print(json.dumps({"output": str(path), "allocated_gpu_hours": result["allocation"]["allocated_gpu_hours"]}))


if __name__ == "__main__":
    main()
