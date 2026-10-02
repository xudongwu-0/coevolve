"""Persistent two-node pilot and matched evaluation for joint-round MA-SAO."""
from __future__ import annotations

import argparse
import json
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
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


def validate_spec(spec: dict) -> None:
    rows = read_json(Path(spec["data"]))
    if (
        spec["protocol"] != "joint-ma-sao-pilot-v1"
        or spec["role"] not in ("editor", "solver")
        or spec["steps"] * spec["batch_size"] != len(rows)
        or spec["staleness"] != 0 or spec["max_head_offpolicyness"] != 0
        or spec.get("central_context") or not spec["terminal_episodes"]
        or spec["critic_mode"] != "learned"
    ):
        raise ValueError("Invalid joint pilot budget or policy-version protocol")
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
    if audit["foreign_gpu_pids"] or not audit["filler_unit_active"]:
        raise RuntimeError("Joint pilot requires released research GPUs with filler active")
    launch = Launcher(root.parent / "supervision")
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
        deadline = time.monotonic() + 10800
        while process.poll() is None:
            launch.check()
            check_failures(joint)
            if time.monotonic() >= deadline:
                raise TimeoutError("Joint native learner timed out")
            time.sleep(2)
        if process.returncode:
            raise RuntimeError(f"Joint native learner failed ({process.returncode}); inspect its log")
        if spec["role"] == "solver":
            environment.join(timeout=30)
            if environment.is_alive():
                raise RuntimeError("Joint environment did not finish its final round")
        check_failures(joint)
        trained = read_json(root / "trained.json")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
        warmup = spec["critic_warmup_steps"]
        if len(rows) != spec["steps"] or trained["actor_steps"] != spec["steps"] - warmup:
            raise ValueError("Joint native learner did not complete its declared update budget")
        for index, row in enumerate(rows):
            gradient = row["metrics"].get("ppo_actor/update/grad_norm")
            if row["global_step"] != index or (index < warmup and gradient is not None):
                raise ValueError("Joint actor warmup or update ordering changed")
            if index >= warmup and (gradient is None or not gradient > 0):
                raise ValueError("Joint Actor omitted a nonzero update")
        ensure_model_runtime_metadata(Path(trained["checkpoints"]["actor"]), Path(spec["model"]))
        result = {
            "status": "completed", "role": spec["role"], "steps": len(rows),
            "actor_updates": trained["actor_steps"], "critic_updates": trained["critic_steps"],
            "metrics_sha256": file_sha(root / "metrics.jsonl"),
            "checkpoints": trained["checkpoints"], "elapsed_seconds": time.time() - started,
            "joint_rounds": spec["steps"], "optimizer_resets_between_rounds": False,
            "finished_at_unix": time.time(),
        }
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
