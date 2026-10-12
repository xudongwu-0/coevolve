"""Persistent two-node pilot and matched evaluation for joint-round MA-SAO."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading
import time

from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .budget import contract_digest
from .common import ensure_model_runtime_metadata, file_sha, read_json, write_json
from .gpu_telemetry import monitor, summarize
from .joint_round import check_failures, record_failure
from .joint_health import assess, metric_rows, recover_checkpoints, request_stop
from .joint_schedule import (
    accounting, consumed_row_ledger, update_counts, validate_component_ablation,
    validate_metric_row, validate_schedule,
)
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node, research_resources_released


def validate_spec(spec: dict) -> None:
    from .training_acceleration import overlay_path
    overlay_path(spec)
    ablation = validate_component_ablation(spec)
    if type(spec.get("critic_cooperative_context", True)) is not bool:
        raise ValueError("Critic cooperative context must be a boolean")
    if not spec.get("critic_cooperative_context", True) and ablation != "without_context":
        raise ValueError("Removing cooperative context requires its isolated ablation declaration")
    if type(spec.get("stop_on_diagnostic_thresholds", True)) is not bool:
        raise ValueError("Diagnostic stopping policy must be a boolean")
    from .joint_anchor import SCHEDULE as ANCHOR
    if spec.get('schedule') == ANCHOR:
        required = ('anchor_harness', 'anchor_harness_sha256', 'schedule_source_data')
        if any(not spec.get(key) for key in required) or spec.get('joint_max_lag') != 1:
            raise ValueError('Anchor training requires pinned routing inputs and asynchronous exchange')
        for key in ('anchor_harness', 'schedule_source_data'):
            path = Path(spec[key])
            digest = file_sha(path)
            if spec.get('pinned_files', {}).get(str(path)) != digest:
                raise ValueError('Anchor routing/source file is not pinned')
            if key == 'anchor_harness' and digest != spec['anchor_harness_sha256']:
                raise ValueError('Anchor harness digest changed')
    elif spec.get('anchor_mode') is not None:
        raise ValueError('Anchor routing cannot run without the matching update schedule')
    if spec.get("peer_weight_mode", "bounded") not in ("bounded", "unit"):
        raise ValueError("Unknown teammate importance weight mode")
    rows = read_json(Path(spec["data"]))
    validate_schedule(spec, rows)
    asynchronous = spec["protocol"] == "joint-ma-sao-async-v1"
    lag = 1 if asynchronous else 0
    if (
        spec["protocol"] not in ("joint-ma-sao-pilot-v1", "joint-ma-sao-async-v1")
        or spec["role"] not in ("editor", "solver")
        or spec["steps"] * spec["batch_size"] != len(rows)
        or spec["staleness"] != lag or spec["max_head_offpolicyness"] != lag
        or spec.get("joint_max_lag", 0) != lag
        or spec.get("central_context") or not spec["terminal_episodes"]
        or spec["critic_mode"] != "learned"
    ):
        raise ValueError("Invalid joint pilot budget or policy-version protocol")
    if asynchronous and (
        (not spec.get("value_normalization") and ablation != "without_value_norm")
        or not 0 < spec.get("kl_ctl", 0) <= 1
        or not isinstance(spec.get("training_deadline_unix"), (int, float))
    ):
        raise ValueError("Joint training requires declared normalization, reference KL and deadline")
    for index, row in enumerate(rows):
        if (
            row["joint_version"] != index // spec["batch_size"]
            or row["joint_slot"] != index % spec["batch_size"]
            or not 0 <= row["task_id"] < 240
            or row.get("replay") or row.get("recycled_ids")
        ):
            raise ValueError("Joint rows must be ordered, fresh train-only episodes")
    for start in range(0, len(rows), spec["batch_size"]):
        if len({row["task_id"] for row in rows[start:start + spec["batch_size"]]}) != spec["batch_size"]:
            raise ValueError("Joint environment batch repeats task assignments")
    for name, digest in spec["pinned_files"].items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f"Joint pilot input changed: {name}")


def run_role(spec_path: Path) -> dict:
    require_research_node()
    spec = read_json(spec_path)
    validate_spec(spec)
    root, joint = Path(spec["run"]), Path(spec["joint_root"])
    _immutable_json(root.parent / "IDENTITY.json", spec)
    previous = root.parent / "COMPLETE.json"
    if previous.exists():
        return read_json(previous)
    if root.exists():
        raise ValueError("Joint optimizers do not silently restart or reuse partial rounds")
    audit = gpu_release_audit()
    if not research_resources_released(audit):
        raise RuntimeError("Joint pilot requires released research GPUs")
    launch = Launcher(root.parent / "supervision")
    if spec.get("training_kernel_overlay"):
        from .training_acceleration import overlay_path
        launch.env["PYTHONPATH"] = f"{overlay_path(spec)}:{launch.env['PYTHONPATH']}"
    stop = threading.Event()
    samples = root.parent / "gpu_process_samples.jsonl"
    telemetry = threading.Thread(target=monitor, args=(samples, stop, 5), daemon=True)
    telemetry.start()
    started = time.time()
    try:
        if spec["role"] == "solver":
            from .joint_environment import run as run_environment
            environment = threading.Thread(
                target=run_environment, args=(spec_path, launch),
                name="joint-environment", daemon=True,
            )
            environment.start()
            launch.env["SAO_ALFWORLD_CONTROLLER_URL"] = "http://127.0.0.1:28000/api"
            launch.env["SAO_ALFWORLD_HTTP_TIMEOUT"] = "900"
        process = launch.module(
            "sao_bridge.large_train", ["--spec", spec_path],
            gpus="0,1,2,3,4,5,6,7", name="joint_native",
        )
        deadline = time.monotonic() + spec.get("phase_timeout_seconds", 10800)
        last_count, last_progress = 0, time.monotonic()
        asynchronous = spec["protocol"] == "joint-ma-sao-async-v1"
        while process.poll() is None:
            launch.check()
            check_failures(joint)
            if asynchronous:
                rows = metric_rows(root / "metrics.jsonl")
                if len(rows) != last_count:
                    health = assess(rows, spec)
                    write_json(root.parent / "HEALTH.json", health)
                    last_count, last_progress = len(rows), time.monotonic()
                    if health["status"] == "stop":
                        request_stop(joint, {"reason": "training_health", "role": spec["role"], "health": health})
                if time.time() >= spec["training_deadline_unix"]:
                    request_stop(joint, {"reason": "evaluation_reserve_deadline", "role": spec["role"]})
                if time.monotonic() - last_progress > 1800:
                    request_stop(joint, {"reason": "no_optimizer_progress_for_1800s", "role": spec["role"]})
                if (joint / "STOP.json").exists():
                    launch.stop(process)
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint native learner timed out")
            time.sleep(2)
        stopped = asynchronous and (joint / "STOP.json").exists()
        if process.returncode and not stopped:
            raise RuntimeError(f"Joint native learner failed ({process.returncode}); inspect its log")
        if spec["role"] == "solver" and not stopped:
            environment.join(timeout=30)
            if environment.is_alive():
                raise RuntimeError("Joint environment did not finish its final round")
        check_failures(joint)
        rows = metric_rows(root / "metrics.jsonl")
        if asynchronous:
            history = recover_checkpoints(spec, rows)
            last = max(map(int, history))
            trained = {
                **update_counts(spec, len(rows)),
                "checkpoints": history[str(last)],
            }
        else:
            trained = read_json(root / "trained.json")
        if not stopped and (
            len(rows) != spec["steps"]
            or any(trained[key] != update_counts(spec, spec["steps"])[key]
                   for key in ("actor_steps", "critic_steps"))
        ):
            raise ValueError("Joint native learner did not complete its declared update budget")
        for index, row in enumerate(rows):
            if row["global_step"] != index:
                raise ValueError("Joint actor warmup or update ordering changed")
            validate_metric_row(spec, row)
        ensure_model_runtime_metadata(Path(trained["checkpoints"]["actor"]), Path(spec["model"]))
        result = {
            "status": "stopped_with_checkpoints" if stopped else "completed",
            "role": spec["role"], "steps": len(rows),
            "actor_updates": trained["actor_steps"], "critic_updates": trained["critic_steps"],
            "metrics_sha256": file_sha(root / "metrics.jsonl"),
            "checkpoints": trained["checkpoints"], "elapsed_seconds": time.time() - started,
            "joint_rounds": len(rows), "optimizer_resets_between_rounds": False,
            "finished_at_unix": time.time(),
        }
        if asynchronous:
            write_json(root / "consumed_rows.json", consumed_row_ledger(
                spec, read_json(Path(spec["data"])), len(rows),
            ))
            result.update(
                checkpoint_history=history,
                stop=read_json(joint / "STOP.json") if stopped else None,
                **accounting(spec, len(rows)),
                generated_policy_trajectories=sum(
                    1 for _ in joint.glob(f"round-*/{spec['role']}-receipt-*.json")
                ),
                sampled_engineer_actions=sum(1 for _ in joint.glob("round-*/proposal-*.json")),
            )
            if spec.get("schedule"):
                result.update(
                    schedule=spec["schedule"],
                    checkpoint_update_counts={
                        step: update_counts(spec, int(step) + 1) for step in history
                    },
                    **accounting(spec, len(rows), generated=result["generated_policy_trajectories"]),
                )
    except BaseException as exc:
        failure = {"error": repr(exc), "role": spec["role"], "failed_at_unix": time.time()}
        write_json(root.parent / "FAILED.json", failure)
        record_failure(joint, spec["role"], failure)
        raise
    finally:
        launch.close()
        stop.set()
        telemetry.join(timeout=10)
        try:
            if samples.exists():
                write_json(root.parent / "gpu_utilization_summary.json", summarize(samples, start_time=started))
        except Exception as exc:
            failure = {"error": repr(exc), "where": "telemetry", "role": spec["role"]}
            write_json(root.parent / "FAILED.json", failure)
            record_failure(joint, spec["role"], failure)
            raise
        finally:
            write_json(root.parent / "RELEASED.json", gpu_release_audit())
    _immutable_json(previous, result)
    return result


def evaluate(protocol_path: Path) -> dict:
    protocol = read_json(protocol_path)
    root = Path(protocol["run"])
    deadline = time.monotonic() + 14400
    for role in ("editor", "solver"):
        while not all((root / role / name).is_file() for name in ("COMPLETE.json", "RELEASED.json")):
            check_failures(root / "joint")
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint pilot did not complete and release both learners")
            time.sleep(10)
    output = root / "evaluation"
    identity = {
        "protocol_sha256": file_sha(protocol_path),
        "training_receipts": {role: file_sha(root / role / "COMPLETE.json") for role in ("editor", "solver")},
    }
    _immutable_json(output / "identity.json", identity)
    if (output / "COMPLETE.json").exists():
        return read_json(output / "COMPLETE.json")
    state = read_json(Path(protocol["initial_state"]))
    trained = {role: read_json(root / role / "COMPLETE.json") for role in ("editor", "solver")}
    launch = Launcher(output / "supervision")
    try:
        candidate = trained["editor"]["checkpoints"]
        proposals = generate_proposals(launch, protocol, state, {"joint_final": candidate}, output)["joint_final"]
        harnesses = {"initial_h1": state["harness"]}
        for proposal in proposals:
            label = f"proposal{proposal['sample_index']:02d}"
            path = output / "patches" / f"{label}.json" if proposal["valid"] else Path(protocol["no_patch"])
            if proposal["valid"]:
                _immutable_json(path, proposal["patch"])
            harnesses[label] = str(path)
        model = trained["solver"]["checkpoints"]["actor"]
        config = matrix_config(protocol, model, "joint-final", harnesses, output / "selection")
        matrix, _ = run_matrix(launch, config, output / "selection", 2)
        reports = matrix_reports(matrix, protocol)
        incumbent = reports["initial_h1"]["accuracy"]
        choices = [
            label for label in harnesses if label != "initial_h1"
            and reports[label]["accuracy"] >= incumbent + protocol["evaluation"]["promotion_min_delta"]
        ]
        selected = min(choices, key=lambda label: (-reports[label]["accuracy"], label)) if choices else "initial_h1"
        _immutable_json(output / "SELECTION.json", {
            "selected_harness": selected,
            "scores": {key: value["accuracy"] for key, value in reports.items()},
            "valid_proposals": sum(row["valid"] for row in proposals),
            "initial_system_not_replaced_until_reported": True,
        })
        pair_reports = {}
        for label, solver, harness in (
            ("initial_h1_w1", state["solver"], state["harness"]),
            ("joint_ma_candidate", model, harnesses[selected]),
        ):
            config = matrix_config(
                protocol, solver, label, {"deployed": harness},
                output / label, split="report",
            )
            matrix, _ = run_matrix(launch, config, output / label, 2)
            pair_reports[label] = matrix_reports(matrix, protocol, split="report")["deployed"]
            _immutable_json(output / f"{label}.json", pair_reports[label])
        difference = paired_difference(
            pair_reports["initial_h1_w1"], pair_reports["joint_ma_candidate"],
            seed=protocol["evaluation"]["bootstrap_seed"], draws=10000,
        )
        result = {
            "status": "completed", "paired": difference,
            "system_scores": {key: value["accuracy"] for key, value in pair_reports.items()},
            "selected_harness": selected, "trained_roles": trained,
            "improved_point_estimate": difference["candidate_minus_control"] > 0,
            "positive_question_lcb": difference["paired_question_bootstrap_95pct"][0] > 0,
            "protocol_sha256": file_sha(protocol_path),
            "limitations": [
                "First short-budget single-seed pilot; no convergence or MAPPO-parity claim.",
                "Joint rounds use the native SAO terminal GAE/value update, not MAPPO ValueNorm.",
                "Historical W79 failure evidence is fixed task input; current outcomes come from paired live policies.",
                "Invalid Engineer actions explicitly execute no-patch, with shared Solver success as team reward.",
                "Reused development cohorts; trained candidate is reported even if worse than the initial system.",
            ],
        }
        _immutable_json(output / "COMPLETE.json", result)
        return result
    except BaseException as exc:
        write_json(output / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        launch.close()
        write_json(output / "RELEASED.json", gpu_release_audit())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--spec", type=Path)
    mode.add_argument("--evaluate", type=Path)
    args = parser.parse_args()
    print(run_role(args.spec) if args.spec else evaluate(args.evaluate), flush=True)
