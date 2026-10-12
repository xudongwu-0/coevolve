"""Bounded resource waiting, native gates, baseline audit and four MA ablations."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

from . import alfworld_ma_mechanism_campaign as campaign
from . import alfworld_ma_mechanism_study as study
from .alfworld_failure_cases import write_case_study
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .joint_anchor import build_rows
from .resources import gpu_release_audit, research_resources_released
from .training_speed_campaign import PYTHON, SSH


def resource_snapshot() -> dict:
    local = gpu_release_audit()
    code = (
        "import json,time; "
        "from sao_bridge.resources import gpu_release_audit,research_resources_released; "
        "r=gpu_release_audit(); "
        "print(json.dumps({'release':r,'released':research_resources_released(r),'at_unix':time.time()}))"
    )
    command = [
        "env", "CUDA_VISIBLE_DEVICES=", "TMPDIR=/dev/shm", "PYTHONDONTWRITEBYTECODE=1",
        f"PYTHONPATH={ROOT}/whale_local:{ROOT}/third_party/sao/vendor/AReaL",
        PYTHON, "-B", "-c", code,
    ]
    remote = subprocess.run(
        [*SSH[:-1], "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", SSH[-1], shlex.join(command)],
        text=True, capture_output=True, timeout=60, check=True,
    )
    remote = json.loads(remote.stdout)
    return {
        "node4": local, "node3": remote,
        "released": research_resources_released(local) and remote["released"],
        "at_unix": time.time(),
    }


def wait_for_resources(root: Path, phase: str, *, deadline: float,
                       interval: float = 60, stable_checks: int = 3, probe=None) -> dict:
    if interval <= 0 or stable_checks < 1:
        raise ValueError("Resource waiting needs a positive interval and stability count")
    consecutive = 0
    while time.monotonic() < deadline:
        try:
            snapshot = (probe or resource_snapshot)()
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
            consecutive = 0
            write_json(root / "PROGRAM_MONITOR.json", {
                "status": "waiting_for_resources", "phase": phase,
                "probe_error": repr(exc), "at_unix": time.time(),
            })
            print(f"RESOURCE PROBE ERROR before {phase}: {exc!r}", flush=True)
        else:
            consecutive = consecutive + 1 if snapshot["released"] else 0
            write_json(root / "PROGRAM_MONITOR.json", {
                "status": "resources_ready" if consecutive >= stable_checks else "waiting_for_resources",
                "phase": phase, "consecutive_free_checks": consecutive,
                "resources": snapshot, "at_unix": time.time(),
            })
            print(f"RESOURCE {phase}: free={snapshot['released']}; stable={consecutive}/{stable_checks}",
                  flush=True)
            if consecutive >= stable_checks:
                return snapshot
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(interval, remaining))
    raise TimeoutError(f"No stable two-node resource window before the declared limit: {phase}")


def audit_historical_path(root: Path, *, destination: str = "HISTORICAL_PATH_AUDIT.json") -> dict:
    state = read_json(root / "INITIAL_STATE.json")
    protocol = read_json(root / "PROTOCOL.json")
    old = ROOT / "runs/alfworld_ma_balanced_fast_v1r2/training"
    convergence = ROOT / "runs/alfworld_ma_convergence_v1r4"
    first = read_json(old / "INITIAL_STATE.json")
    w1_path = ROOT / "runs/alfworld_solver_fast_v4_s42/training/optimizer/trained.json"
    w1 = read_json(w1_path)
    peak_path = convergence / "block-01/round0096_COMPLETE.json"
    peak = read_json(peak_path)
    first_audit = read_json(old.parent / "TRAINING_AUDIT.json")
    if (state["engineer"] != first["engineer"] or state["solver"] != first["solver"]
            or state["solver"] != w1["checkpoints"]["actor"]
            or w1["actor_steps"] != 75 or w1["critic_steps"] != 80
            or peak["scores"]["joint_harness"]["all200"]["successes"] != 194
            or first_audit["status"] != "passed"):
        raise ValueError("The claimed H0/W1 to historical97% lineage does not match its records")
    paths = [
        old / "INITIAL_STATE.json", old / "data.json", old.parent / "TRAINING_AUDIT.json",
        old / "matched_COMPLETE.json", w1_path, convergence / "DESIGN.json",
        convergence / "block-01/data.json", peak_path,
    ]
    critic_sources = {}
    for role, key in (("editor", "engineer_critic"), ("solver", "solver_critic")):
        record_path = ROOT / f"runs/alfworld_joint_h0w0_8h_v1/{role}/COMPLETE.json"
        original = read_json(record_path)
        selected = original["checkpoint_history"]["63"]["critic"]
        norm = Path(selected) / "joint_value_norm.json"
        if first[key] != selected or not norm.is_file() or state[key] == selected:
            raise ValueError("Historical MA64 and current pre-MA Critic initializations are conflated")
        if (Path(state[key]) / "joint_value_norm.json").exists():
            raise ValueError("Current baseline unexpectedly inherits MA ValueNorm state")
        current_spec = read_json(root / "full" / f"{role}_spec.json")
        original_spec_path = old / f"{role}_spec.json"
        original_spec = read_json(original_spec_path)
        for field in ("model", "reference_model", "lr", "critic_lr", "kl_ctl", "batch_size",
                      "tokens", "context_tokens", "temperature", "critic_updates_per_step", "seed",
                      "schedule", "anchor_mode", "peer_weight_mode"):
            if current_spec[field] != original_spec[field]:
                raise ValueError(f"The new full control changed a claimed common setting: {role}/{field}")
        actual = {
            "actor_steps": first_audit["roles"][role]["actor_steps"] + peak["training"][role]["actor_steps"],
            "critic_steps": first_audit["roles"][role]["critic_steps"] + peak["training"][role]["critic_steps"],
            "consumed_trajectories": first_audit["roles"][role]["consumed_policy_trajectories"]
            + peak["training"][role]["consumed_policy_trajectories"],
        }
        expected = study.arm_budgets("full")["roles"][role]
        if any(actual[key] != expected[key] for key in actual):
            raise ValueError("Historical48+96 and new144 do not match their stated MA update budgets")
        critic_sources[role] = {
            "historical_critic": selected, "historical_norm": str(norm),
            "selected_upstream_checkpoint_round": 64,
            "upstream_run_actual_committed_rounds": original["steps"],
            "current_critic": state[key], "ma_segment_budget": actual,
        }
        paths += [record_path, norm, original_spec_path, root / "full" / f"{role}_spec.json"]
    old_rows = read_json(old / "data.json") + read_json(convergence / "block-01/data.json")
    new_rows = read_json(root / "joint_data.json")
    fields = ("task_id", "messages", "task_context", "critic_evidence")
    if len(old_rows) != 1152 or len(new_rows) != 1152:
        raise ValueError("Historical or new full lost its fixed training population")
    for field in fields:
        if [row[field] for row in old_rows] != [row[field] for row in new_rows]:
            raise ValueError(f"Historical and new ordered training inputs differ: {field}")
    for role in ("editor", "solver"):
        spec_path = convergence / "block-01" / f"{role}_spec.json"
        spec = read_json(spec_path)
        if (spec["steps"] != 96 or spec["critic_warmup_steps"] != 0
                or spec["model"] != read_json(old / "matched_COMPLETE.json")["selection"]["pair"][role]["actor"]):
            raise ValueError("The round48 to144 continuation does not load the declared MA48 Actor")
        paths.append(spec_path)
    cohort = read_json(Path(protocol["core200_protocol"]))
    report_path = Path(peak["reports"]["joint_harness"]["path"])
    if file_sha(report_path) != peak["reports"]["joint_harness"]["sha256"]:
        raise ValueError("The historical97% raw report changed")
    report = study.checked_native_report(read_json(report_path), cohort["all_task_ids"], {
        **protocol["evaluation"], **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
    })
    if sum(row["reward"] for row in report["results"]) != 194:
        raise ValueError("Historical97% is not194 successes on the exact original200")
    paths += [report_path, Path(peak["selection"]["harness"]), root / "joint_data.json",
              root / "INITIAL_STATE.json", root / "PROTOCOL.json"]
    result = {
        "status": "passed", "at_unix": time.time(), "historical_accuracy": .97,
        "historical_report": str(report_path), "historical_endpoint": str(peak_path),
        "actors_match_historical_start": True, "ordered_training_rows_match": 1152,
        "matched_row_fields": list(fields), "critic_sources": critic_sources,
        "known_differences": [
            "Historical MA64 Critic/ValueNorm versus common pre-MA ordinary-SAO Critics.",
            "Historical fresh-Adam and serving/collection restart at round48 versus one144-round optimizer run.",
            "Historical48/96/144 candidate searches (24 proposals) versus96/144 (16 proposals).",
            "Historical97% was a development-selected peak; new144 is a fixed endpoint.",
        ],
        "not_a_bitwise_historical_reproduction": True,
        "provenance_scope": "Checkpoint paths, native receipts, ordered inputs, hyperparameters and raw evaluation digests; not a re-execution of the historical trajectory.",
        "pins": study.file_manifest(paths),
    }
    _immutable_json(root / destination, result)
    return result


def prepare_program(root: Path) -> dict:
    protocol = read_json(root / "PROTOCOL.json")
    if protocol["arms"] != list(study.ARMS) or protocol.get("study_revision") != 2:
        raise ValueError("The revised program must contain exactly the four requested ablations")
    historical = audit_historical_path(root)
    prior = ROOT / "runs/alfworld_ma_mechanisms_single_seed_v1"
    gate = read_json(prior / "NATIVE_GATE.json")
    if gate["status"] != "passed" or gate["pins_sha256"] != file_sha(prior / "PINS.json"):
        raise ValueError("Original full-context native evidence is not valid")
    prior_source = Path(read_json(prior / "PROTOCOL.json")["source"])
    unchanged = [
        "joint_memory_check.py", "joint_critic.py", "joint_context.py",
        "terminal_actor.py", "training_acceleration.py", "joint_updates.py",
    ]
    for name in unchanged:
        if file_sha(ROOT / "whale_local/sao_bridge" / name) != file_sha(prior_source / "whale_local/sao_bridge" / name):
            raise ValueError(f"Inherited native kernel evidence no longer applies: {name}")
    for case in gate["cases"].values():
        path = Path(case["result"])
        if (read_json(path)["status"] != "passed" or file_sha(path) != case["sha256"]
                or file_sha(path.parent / "audit-summary.json") != case["audit_sha256"]):
            raise ValueError("A reused native result or audit changed")
    _immutable_json(root / "MEMORY_GATE_INHERITANCE.json", {
        "status": "passed", "original_gate": str(prior / "NATIVE_GATE.json"),
        "original_gate_sha256": file_sha(prior / "NATIVE_GATE.json"), "cases": gate["cases"],
        "unchanged_runtime_modules": unchanged, "native_cases_reexecuted": 0,
        "schedule_native_validation_still_required": True,
    })
    calibration = root / "schedule-calibration"
    source_rows = read_json(root / "schedule_source_data.json")[:32]
    _immutable_json(calibration / "source_data.json", source_rows)
    _immutable_json(calibration / "data.json", build_rows(source_rows, cycles=2, batch_size=8))
    _immutable_json(calibration / "PROTOCOL.json", {
        **protocol, "root": str(calibration), "joint_rounds": 6,
        "arms": ["full", "always_joint"], "unit_suffix": "schedule-calibration-v2",
        "purpose": "native_execution_validation_not_an_additional_research_seed_or_selected_model",
    })
    additions = study.file_manifest([
        calibration / "source_data.json", calibration / "data.json", calibration / "PROTOCOL.json",
    ])
    for arm in ("full", "always_joint"):
        for role in ("editor", "solver"):
            spec = read_json(root / arm / f"{role}_spec.json")
            spec.update(
                run=str(calibration / arm / role / "optimizer"), joint_root=str(calibration / arm / "joint"),
                data=str(calibration / "data.json"), schedule_source_data=str(calibration / "source_data.json"),
                steps=6, save_interval_steps=6, experiment=f"{root.name}-canary-{arm}",
                phase_timeout_seconds=7200, pinned_files={**spec["pinned_files"], **additions},
            )
            study.validate_spec(spec)
            _immutable_json(calibration / arm / f"{role}_spec.json", spec)
    speed_path = ROOT / "runs/alfworld_ma_balanced_fast_v1r2/INDEPENDENT_AUDIT.json"
    speed = read_json(speed_path)
    speed_probe = ROOT / "runs/alfworld_ma_engineer_weights_w0_v1/throughput-probe/RESULT.md"
    _immutable_json(root / "SPEED_POLICY.json", {
        "status": "reuse_measured_validated_acceleration",
        "timing_evidence": str(speed_path), "timing_evidence_sha256": file_sha(speed_path),
        "warm_round_seconds_before": speed["timings"]["parent"]["warm_round_mean_seconds"],
        "warm_round_seconds_after": speed["timings"]["candidate"]["warm_round_mean_seconds"],
        "measured_training_speedup": speed["timings"]["warm_round_speedup"],
        "runtime": "FLA0.5.2/Triton3.7.1; BF16 fixed references, unchanged FP32 learner masters; four-worker full-file SHA.",
        "rejected_generation_layout_probe": str(speed_probe),
        "rejected_probe_sha256": file_sha(speed_probe),
        "new_training_speedup_claimed": False,
        "tokens_context_batch_rollout_layout_and_learning_budgets_reduced": False,
    })
    _immutable_json(root / "PRIOR_RESULTS_REPORTED.json", read_json(prior / "PRIOR_RESULTS_REPORTED.json"))
    interruption = read_json(prior / "full/INTERRUPTED.json")
    for role in ("editor", "solver"):
        if file_sha(prior / "full" / role / "optimizer/metrics.jsonl") != interruption["roles"][role]["metrics_sha256"]:
            raise ValueError("The previously interrupted experiment's costs changed")
    superseded = ROOT / "runs/alfworld_ma_mechanisms_single_seed_v1_attempt02"
    cancellation = superseded / "CANCELLED_BEFORE_TRAINING.json"
    if (read_json(cancellation)["status"] != "cancelled_before_training"
            or any(superseded.rglob("*STARTED*.json"))):
        raise ValueError("The superseded preparation must be cancelled with zero training starts")
    _immutable_json(root / "PRIOR_WORK.json", {
        "interrupted_attempt": str(prior), "interruption": interruption,
        "interruption_sha256": file_sha(prior / "full/INTERRUPTED.json"),
        "not_same_optimizer_continuation": True,
        "prior_native_evidence": str(prior / "NATIVE_GATE.json"),
        "scope_replaced_zero_training_attempt": str(superseded),
        "cancellation_sha256": file_sha(cancellation),
        "unpersisted_partial_token_cost_unknown": True,
    })
    files = [root / "MEMORY_GATE_INHERITANCE.json", root / "HISTORICAL_PATH_AUDIT.json",
             root / "SPEED_POLICY.json", root / "PRIOR_RESULTS_REPORTED.json",
             root / "PRIOR_WORK.json", cancellation,
             *calibration.glob("*.json"), *calibration.glob("*/*_spec.json")]
    files += [Path(path) for path in historical["pins"]]
    result = {
        "status": "prepared", "pins": study.file_manifest(files),
        "full_first_then_four_ablations": True, "automatic_training_retries": 0,
        "resource_wait_limit_hours": 12, "overall_wall_limit_hours": 40,
        "baseline_review": "Always audit the full path; a score<=94% is flagged for detailed explanation, never grounds for score-driven retries.",
        "native_calibration_environment_episodes": 96, "at_unix": time.time(),
    }
    _immutable_json(root / "PROGRAM_PREPARED.json", result)
    return result


def checked_result(root: Path, arm: str) -> dict:
    audit = read_json(root / arm / "FINAL_AUDIT.json")
    if (audit["status"] != "passed" or audit["result_sha256"] != file_sha(root / arm / "RESULT.json")
            or audit["report_sha256"] != file_sha(root / arm / "RESULT.md")):
        raise ValueError("A completed arm no longer matches its final audit")
    return read_json(root / arm / "RESULT.json")


def paired_exact_p(improved: int, regressed: int) -> float:
    if any(type(value) is not int or value < 0 for value in (improved, regressed)):
        raise ValueError("Paired discordant counts must be nonnegative integers")
    n = improved + regressed
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1)) / 2**n)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in pvalues.values()):
        raise ValueError("Invalid exact paired p-value")
    result, previous = {}, 0.
    for index, (name, value) in enumerate(sorted(pvalues.items(), key=lambda item: (item[1], item[0]))):
        previous = max(previous, min(1., (len(pvalues) - index) * value))
        result[name] = previous
    return result


def write_master_report(root: Path) -> dict:
    arms = {arm: checked_result(root, arm) for arm in study.ARMS
            if (root / arm / "FINAL_AUDIT.json").exists()}
    complete = len(arms) == 5
    pvalues = {
        arm: paired_exact_p(value["paired_against_full"]["improved_questions"],
                           value["paired_against_full"]["regressed_questions"])
        for arm, value in arms.items() if arm != "full"
    }
    adjusted = holm_adjust(pvalues) if complete else {}
    result = {
        "status": "completed" if complete else "in_progress", "arms": arms,
        "historical_reference_accuracy": .97, "primary_control": "new_full",
        "paired_exact_p": pvalues, "holm_adjusted_p_four_comparisons": adjusted,
        "single_training_seed": 1000043, "fixed_primary_rounds": 144,
        "prior_work": read_json(root / "PRIOR_WORK.json"),
        "native_calibration": read_json(root / "NATIVE_GATE.json"),
        "training_seed_robustness_claimed": False, "at_unix": time.time(),
    }
    write_json(root / "RESULT.json", result)
    lines = [
        "# MA-SAO: full control and four single-seed ablations", "",
        f"Status: **{result['status']}**. Primary endpoint144; original200 development questions.", "",
        "| Configuration | Saved system | Empty Harness | Full minus ablation, pp (paired95%) | Holm p |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| Historical97% (not a causal control) |194/200 (97.0%)|185/200 (92.5%)|not applicable|not applicable|",
    ]
    for arm in study.ARMS:
        if arm not in arms:
            lines.append(f"|{arm}|pending|pending|pending|pending|")
            continue
        endpoint = arms[arm]["endpoints"]["144"]["scores"]
        difference = "control"
        if arm != "full":
            comparison = arms[arm]["paired_against_full"]
            low, high = comparison["paired_question_bootstrap_95pct"]
            difference = f"{-100*comparison['candidate_minus_control']:+.1f} [{-100*high:+.1f},{-100*low:+.1f}]"
        lines.append(
            f"|[{arm}](./{arm}/RESULT.md)|{endpoint['selected']['successes']}/200 "
            f"({100*endpoint['selected']['accuracy']:.1f}%)|"
            f"{endpoint['no_patch']['successes']}/200 ({100*endpoint['no_patch']['accuracy']:.1f}%)|"
            f"{difference}|{adjusted.get(arm, 'pending' if arm != 'full' else 'control')}|"
        )
    lines += [
        "", "Always-joint preserves third-round peer correction and the environment/generated budget,",
        "but consumes20% more role trajectories. It is not an equal-update or equal-compute comparison.",
        "Intervals and exact paired tests describe these questions, not independent training seeds.",
        "Four-comparison Holm adjustment is reported only after all four ablations finish.",
        "Reused development data and historical selection limit confirmatory claims; null/reverse results stay.",
        "Costs and paired failure/reverse cases are retained in each linked arm report.",
        "The previous interrupted22 rounds and the96 calibration episodes are additional costs,",
        "not omitted or represented as same-optimizer continuation.",
        "", "[Historical path audit](./HISTORICAL_PATH_AUDIT.json)",
        "[Speed policy](./SPEED_POLICY.json)",
    ]
    (root / "RESULT.md").write_text("\n".join(lines) + "\n")
    return result


def verify_executed_role_identity(root: Path, arm: str, role: str) -> None:
    expected = read_json(root / arm / f"{role}_spec.json")
    directory = root / arm / role
    if expected.get("rolling_recovery") is not None:
        from .joint_recovery import validate_recovery_spec

        actual_directory = directory.resolve()
        attempt = actual_directory.parent.parent
        if (
            not actual_directory.is_relative_to((root / arm / "attempts").resolve())
            or actual_directory.name != role or actual_directory.parent.name != arm
            or attempt.parent != (root / arm / "attempts").resolve()
        ):
            raise ValueError("The published role escaped its owned rolling attempt")
        prepared = read_json(attempt / "PREPARED.json")
        if prepared["status"] != "prepared" or prepared["arm"] != arm:
            raise ValueError("The executed role has no matching prepared attempt")
        start = prepared["resume_round"]
        expected.update(
            run=str(actual_directory / "optimizer"), joint_root=str(actual_directory.parent / "joint"),
            recovery_start_round=start,
        )
        expected["rolling_recovery"].update(
            attempt=attempt.name, resume=prepared["recovery"], finalize_only=start == expected["steps"],
        )
        validate_recovery_spec(expected)
        if read_json(actual_directory.parent / f"{role}_spec.json") != expected:
            raise ValueError("The executed rolling attempt changed its frozen role specification")
    if read_json(directory / "IDENTITY.json") != expected:
        raise ValueError("The actual learner did not use its frozen executed specification")


def audit_full_result(root: Path) -> dict:
    result = checked_result(root, "full")
    lineage = audit_historical_path(root, destination="full/HISTORICAL_PATH_RECHECK.json")
    scores = result["endpoints"]["144"]["scores"]["selected"]
    report = read_json(Path(scores["report"]))
    for role in ("editor", "solver"):
        verify_executed_role_identity(root, "full", role)
    audit = {
        "status": "passed", "full_result_sha256": file_sha(root / "full/RESULT.json"),
        "lineage_audit_sha256": file_sha(root / "full/HISTORICAL_PATH_RECHECK.json"),
        "full_accuracy": scores["accuracy"], "below_reference_by_at_least_3pp": scores["accuracy"] <= .94,
        "known_historical_path_differences": lineage["known_differences"],
        "historical_comparison_not_component_causal": paired_difference(
            read_json(Path(lineage["historical_report"])), report, seed=20261008),
        "decision": "Retain the fixed result; require a reported baseline/path review before releasing ablations.",
        "score_driven_retry_or_initialization_change": False, "at_unix": time.time(),
    }
    audit["historical_failure_case_population"] = write_case_study(
        root / "full/failure-cases-vs97", Path(scores["report"]), Path(lineage["historical_report"]),
        read_json(Path(read_json(root / "PROTOCOL.json")["core200_protocol"])),
        title="New full-control failures and matched historical97% outcomes; not component-causal",
    )
    _immutable_json(root / "full/BASELINE_PATH_AUDIT.json", audit)
    (root / "full/BASELINE_PATH_AUDIT.md").write_text(
        "# Full-MA path review\n\n"
        f"Fixed144 score: {scores['successes']}/200 ({100*scores['accuracy']:.1f}%).\n\n"
        "The executed identity, ordered inputs, role budgets and historical chain passed their checks.\n"
        "A lower score alone is not evidence of a broken implementation. Known differences are:\n\n"
        + "\n".join(f"- {item}" for item in lineage["known_differences"])
        + "\n\nAll results are retained. No score-based retry or silent initialization change is authorized.\n"
    )
    return audit


def run(root: Path, *, ablations: bool = False) -> dict:
    prepared = read_json(root / "PROGRAM_PREPARED.json")
    if prepared["status"] != "prepared" or study.file_manifest(prepared["pins"]) != prepared["pins"]:
        raise ValueError("Prepared scientific/runtime evidence changed before monitoring")
    if (root / "CANCELLED_BEFORE_TRAINING.json").exists():
        raise ValueError("The requested study was cancelled")
    if (root / "PROGRAM_FAILED.json").exists():
        raise ValueError("An interrupted program requires explicit recovery, not an automatic retry")
    if ablations:
        campaign.require_full_control_review(root)
    else:
        _immutable_json(root / "PROGRAM_STARTED.json", {
            "pid": os.getpid(), "at_unix": time.time(), "attached": True,
            "deadline_unix": time.time() + prepared["overall_wall_limit_hours"] * 3600,
        })
    stage = "ABLATIONS" if ablations else "CONTROL"
    _immutable_json(root / f"{stage}_PROGRAM_STARTED.json", {
        "pid": os.getpid(), "at_unix": time.time(), "attached": True,
        "source": str(ROOT), "automatic_training_retries": 0,
    })
    remaining = read_json(root / "PROGRAM_STARTED.json")["deadline_unix"] - time.time()
    deadline = time.monotonic() + remaining
    phase = "without_peer" if ablations else "native-gates"

    def interrupted(signum, frame):
        write_json(root / f"SIGNAL_{time.time_ns()}.json", {
            "signal": signum, "pid": os.getpid(), "phase": phase, "at_unix": time.time(),
        })
        raise InterruptedError(f"Received signal{signum}; preserving this attempt without automatic retry")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)

    def expired(signum, frame):
        raise TimeoutError("The shared control-plus-ablation40h wall limit expired")

    signal.signal(signal.SIGALRM, expired)

    def wait(phase):
        return wait_for_resources(
            root, phase, deadline=min(deadline, time.monotonic() + prepared["resource_wait_limit_hours"] * 3600),
        )

    try:
        if remaining <= 0:
            raise TimeoutError("The shared control-plus-ablation wall limit has already expired")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        if not ablations:
            wait(phase)
            campaign.require_prerequisites(root, native=False)
            calibration = root / "schedule-calibration"
            cases = {}
            for arm in ("full", "always_joint"):
                phase = f"native-{arm}"
                if (calibration / arm / "STARTED.json").exists():
                    raise ValueError("A partial native canary needs explicit recovery, not repetition")
                print(f"START {phase}: six rounds; calibration only", flush=True)
                _immutable_json(calibration / arm / "STARTED.json", {"at_unix": time.time()})
                write_json(root / "PROGRAM_MONITOR.json", {
                    "status": "running", "phase": phase,
                    "arm_monitor": str(calibration / arm / "MONITOR.json"), "at_unix": time.time(),
                })
                audit = campaign.train_joint(calibration, arm, rounds=6)
                cases[arm] = {
                    "audit": str(calibration / arm / "TRAINING_AUDIT.json"),
                    "sha256": file_sha(calibration / arm / "TRAINING_AUDIT.json"),
                    "shared_environment_episodes": audit["shared_environment_episodes"],
                }
                if arm == "full":
                    wait("before-native-always_joint")
            _immutable_json(root / "NATIVE_GATE.json", {
                "status": "passed", "pins_sha256": file_sha(root / "PINS.json"),
                "memory_gate_inheritance_sha256": file_sha(root / "MEMORY_GATE_INHERITANCE.json"),
                "real_schedule_cases": cases, "calibration_environment_episodes": 96,
                "new_independent_training_seeds": 0, "at_unix": time.time(),
            })
        for arm in study.ARMS[1:] if ablations else ("full",):
            phase = arm
            wait(phase)
            write_json(root / "PROGRAM_MONITOR.json", {
                "status": "running", "phase": arm, "arm_monitor": str(root / arm / "MONITOR.json"),
                "at_unix": time.time(),
            })
            print(f"START {arm}: H0/W1; seed1000043; fixed144; original200", flush=True)
            campaign.run_arm(root, arm)
            if arm == "full":
                review = audit_full_result(root)
                print(f"BASELINE PATH REVIEW: score={review['full_accuracy']:.3f}; "
                      f"large_reference_gap={review['below_reference_by_at_least_3pp']}; "
                      "identity/data/budget checks passed; no score-based retry", flush=True)
            write_master_report(root)
            value = checked_result(root, arm)["endpoints"]["144"]["scores"]["selected"]
            print(f"COMPLETE {arm}: {value['successes']}/200; {root / arm / 'RESULT.md'}", flush=True)
        if not ablations:
            result = {"status": "control_ready_for_review", "at_unix": time.time()}
            _immutable_json(root / "CONTROL_PROGRAM_COMPLETE.json", result)
            write_json(root / "PROGRAM_MONITOR.json", result)
            return result
        result = write_master_report(root)
        if result["status"] != "completed":
            raise RuntimeError("The program cannot complete with an unfinished ablation")
        _immutable_json(root / "PROGRAM_COMPLETE.json", {
            "status": "completed", "result_sha256": file_sha(root / "RESULT.json"),
            "report_sha256": file_sha(root / "RESULT.md"), "at_unix": time.time(),
        })
        write_json(root / "PROGRAM_MONITOR.json", {"status": "completed", "at_unix": time.time()})
        return result
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.SubprocessError, ExceptionGroup) as exc:
        failure = {"status": "incomplete", "phase": phase, "error": repr(exc),
                   "automatic_retry": False, "at_unix": time.time()}
        write_json(root / "PROGRAM_FAILED.json", failure)
        write_json(root / "PROGRAM_MONITOR.json", failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def watch(root: Path, completed_arms: int) -> dict:
    if not 1 <= completed_arms <= 5:
        raise ValueError("A milestone must be one through five completed configurations")
    deadline = time.monotonic() + read_json(root / "PROGRAM_STARTED.json")["deadline_unix"] - time.time()
    while time.monotonic() < deadline:
        failure = root / "PROGRAM_FAILED.json"
        if failure.exists():
            return {**read_json(failure), "status": "program_failed"}
        result_path = root / "RESULT.json"
        if result_path.exists():
            result = read_json(result_path)
            if len(result["arms"]) >= completed_arms:
                return {
                    "status": "milestone_reached", "completed_arms": list(result["arms"]),
                    "scores": {arm: value["endpoints"]["144"]["scores"]["selected"]["accuracy"]
                               for arm, value in result["arms"].items()},
                    "at_unix": time.time(),
                }
        time.sleep(30)
    raise TimeoutError("No completion/failure milestone within the declared monitoring limit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run-control", "run-ablations", "watch"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--completed-arms", type=int, default=1)
    args = parser.parse_args()
    os.umask(0o002)
    root = args.root.resolve()
    if args.action == "prepare":
        result = prepare_program(root)
    elif args.action == "watch":
        print(json.dumps(watch(root, args.completed_arms)), flush=True)
        return
    else:
        result = run(root, ablations=args.action == "run-ablations")
    print(json_summary(result), flush=True)


def json_summary(result: dict) -> str:
    return json.dumps({"status": result["status"], "at_unix": result.get("at_unix")})


if __name__ == "__main__":
    main()
