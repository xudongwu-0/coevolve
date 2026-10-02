"""Persistent, bounded native phases for the frozen strong ALFWorld continuation."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
from pathlib import Path
import socket
import time

from .alfworld_phase_data import editor_dataset
from .alfworld_reward_service import validate_metadata
from .alfworld_solver_validation import no_patch_assignments
from .alfworld_train_data import (
    _immutable_json, evaluation_paths, goal_group, solver_rows, training_catalogue,
)
from .alfworld_w_train import assignment_schedule_audit
from .analysis import NATIVE_TARGET_ROUNDOFF_ATOL, validate_terminal_target_range
from .common import file_sha, read_json, write_json
from .budget import contract_digest
from .evaluation import stable_id_seed_offset
from .joint_context import joint_context_audit
from .large_train import training_capacity_audit, validate_training_rows
from .launch import Launcher
from .resources import gpu_release_audit


class PhaseAttemptsExhausted(RuntimeError):
    pass


def legacy_native_audit(audit: dict) -> dict:
    return {
        key: value for key, value in audit.items()
        if key not in {"critic_target_range", "critic_target_roundoff_atol"}
    }


def publish_phase_completion(root: Path, complete: dict) -> dict:
    path = root / "PHASE_COMPLETE.json"
    if path.is_file():
        previous = read_json(path)
        legacy = {**complete, "native_audit": legacy_native_audit(complete["native_audit"])}
        if previous == legacy:
            return previous
    _immutable_json(path, complete)
    return complete


def checked_native_report(report: dict, task_ids: list[int], evaluation: dict) -> dict:
    expected = {(task_id, sample) for task_id in task_ids
                for sample in range(evaluation["samples"])}
    results = report["results"]
    actual = [(row["task_id"], row["sample_index"]) for row in results]
    if (
        len(actual) != len(expected) or set(actual) != expected
        or report["question_count"] != len(task_ids)
        or report["samples_per_question"] != evaluation["samples"]
        or report["metric"] != f"mean@{evaluation['samples']}"
    ):
        raise ValueError("Native report changed question/sample coverage or metric")
    for row in results:
        expected_id = f"alfworld-{row['task_id']}"
        expected_seed = (
            evaluation["seed"] + stable_id_seed_offset(expected_id)
            + 1009 * row["sample_index"]
        )
        if (
            row["id"] != expected_id or row["rollout_seed"] != expected_seed
            or type(row["reward"]) not in (int, float)
            or row["reward"] not in (0.0, 1.0)
            or type(row["tokens"]) is not int
            or not 0 <= row["tokens"] <= evaluation["solver_tokens"]
        ):
            raise ValueError("Native report changed seeds, rewards or episode token budget")
    mean = sum(row["reward"] for row in results) / len(results)
    if not math.isclose(report["accuracy"], mean, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Native report accuracy differs from its complete trajectories")
    return report


def checked_release(root: Path) -> dict:
    released = read_json(root / "RELEASED.json")
    if (
        released["foreign_gpu_pids"]
        or not set(released["remaining_gpu_pids"]) <= set(released["verified_filler_pids"])
        or not released["filler_unit_active"]
    ):
        raise RuntimeError("A prerequisite did not release research GPUs with filler active")
    return released


def wait_completed(root: Path, timeout: float = 10800) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (root / "FAILED.json").exists():
            raise RuntimeError(f"Prerequisite failed: {root / 'FAILED.json'}")
        complete, released = root / "COMPLETE.json", root / "RELEASED.json"
        if complete.exists() and released.exists():
            if read_json(complete)["status"] not in {"complete", "completed"}:
                raise ValueError("Prerequisite completion status is not successful")
            checked_release(root)
            return
        time.sleep(10)
    raise TimeoutError(f"Prerequisite did not complete and release GPUs: {root}")


def checkpoint_pair(trained: dict, step: int) -> dict:
    pair = {}
    for role in ("actor", "critic"):
        matching = [
            entry["path"] for entry in trained["checkpoint_history"][role]
            if entry["global_step"] == step
        ]
        if len(matching) != 1:
            raise ValueError(f"Missing unambiguous {role} checkpoint at step {step}")
        pair[role] = matching[0]
    return pair


def train_calibration(packets: list[dict], solver_version: str) -> dict:
    rewards = {}
    for packet in packets:
        metadata = packet["metadata"]
        validate_metadata(metadata, solver_version)
        if metadata["end"] > 240:
            raise ValueError("Engineer calibration must use only legacy training tasks")
        scores = {int(key): value for key, value in metadata["baseline_rewards"].items()}
        if rewards.keys() & scores.keys():
            raise ValueError("Training calibration repeats baseline task IDs")
        rewards.update(scores)
    if not rewards:
        raise ValueError("Training calibration requires authentic nonempty baseline evidence")
    return {
        "accuracy": sum(rewards.values()) / len(rewards),
        "task_ids": sorted(rewards),
        "split": "train",
        "harness": "no_patch",
        "solver_version": solver_version,
    }


def finalize_common(protocol: dict, matrix_root: Path, *, initial_state: dict | None = None) -> dict:
    checked_release(matrix_root)
    config = read_json(matrix_root / "config.json")
    complete = read_json(matrix_root / "COMPLETE.json")
    manifest = read_json(Path(protocol["evaluation"]["task_manifest"]))
    task_ids = manifest["selection_task_ids"]
    if (
        complete["status"] != "completed"
        or complete.get("evaluation_mode") != "frozen_harness_matrix"
        or Path(config["solver_model"]).resolve() != Path(protocol["initial_solver"]).resolve()
        or config["task_ids"] != task_ids
        or any(config[key] != protocol["evaluation"][key] for key in (
            "samples", "seed", "temperature", "solver_tokens", "eval_concurrency",
            "environment_concurrency", "matrix_concurrency",
        ))
    ):
        raise ValueError("Common selection differs from the frozen native protocol")
    variants = read_json(matrix_root / "report.json")["variants"]
    proposals = read_json(Path(protocol["common_prefix"]["proposal_file"]))
    if [row["sample_index"] for row in proposals] != list(range(8)):
        raise ValueError("Common prefix must retain its eight original proposals")
    expected = {"no_patch": read_json(Path(protocol["no_patch"]))}
    for proposal in proposals:
        expected[f"proposal{proposal['sample_index']:02d}"] = (
            proposal["patch"] if proposal["valid"] else expected["no_patch"]
        )
    if set(variants) != set(expected):
        raise ValueError("Common matrix lost or added a logical proposal")
    reports = {}
    for label, variant in variants.items():
        if variant["harness_sha256"] != contract_digest(expected[label]):
            raise ValueError("Common matrix patch identity changed")
        reports[label] = checked_native_report(
            read_json(Path(variant["report"])), task_ids, protocol["evaluation"],
        )
        if variant["accuracy"] != reports[label]["accuracy"]:
            raise ValueError("Common matrix summary disagrees with its trajectories")
    floor = protocol["evaluation"]["promotion_min_delta"]
    baseline = reports["no_patch"]["accuracy"]
    qualified = [
        label for label in expected if label != "no_patch"
        and reports[label]["accuracy"] - baseline >= floor
    ]
    selected = min(qualified, key=lambda label: (-reports[label]["accuracy"], label)) if qualified else "no_patch"
    common = Path(protocol["run"]) / "common"
    harness = common / "selected_harness.json"
    _immutable_json(harness, expected[selected])
    _immutable_json(common / "selected_report.json", reports[selected])
    _immutable_json(common / "no_patch_report.json", reports["no_patch"])
    trained = read_json(
        Path(protocol["common_prefix"]["engineer_training"]) / "optimizer/trained.json"
    )
    pair = checkpoint_pair(trained, protocol["common_prefix"]["selected_checkpoint_step"])
    if initial_state is None:
        template = read_json(Path(protocol["h2"]["template"]))
        packets_path = Path(template["data"]).parent / "packets.json"
        packets = read_json(packets_path)
        state = {
            "engineer": pair["actor"], "engineer_critic": pair["critic"],
            "solver": protocol["initial_solver"], "solver_critic": protocol["initial_solver"],
            "solver_version": template["solver_version"], "solver_epoch_version": 0,
            "engineer_version": 1,
            "packets": str(packets_path), "packets_sha256": file_sha(packets_path),
            "packet_solver": protocol["initial_solver"],
            "calibration": train_calibration(packets, template["solver_version"]),
        }
    else:
        state = dict(initial_state)
        if (
            state["engineer"] != pair["actor"] or state["engineer_critic"] != pair["critic"]
            or Path(state["solver"]).resolve() != Path(protocol["initial_solver"]).resolve()
            or Path(state["packet_solver"]).resolve() != Path(state["solver"]).resolve()
            or config["solver_version"] != state["solver_version"]
            or state["packets_sha256"] != file_sha(Path(state["packets"]))
            or state["protocol_sha256"] != file_sha(Path(protocol["run"]) / "frozen_protocol.json")
            or state["calibration"]["split"] != "train"
            or state["calibration"]["solver_version"] != state["solver_version"]
        ):
            raise ValueError("Warm-start common state changed its models or current-partner evidence")
        for packet in read_json(Path(state["packets"])):
            validate_metadata(packet["metadata"], state["solver_version"])
    state.update({
        "harness_version": int(selected != "no_patch"),
        "harness": str(harness), "harness_sha256": contract_digest(expected[selected]),
        "selection_report": str(common / "selected_report.json"),
        "selection_label": selected, "selection_accuracy": reports[selected]["accuracy"],
        "selection_no_patch_accuracy": baseline,
        "protocol_sha256": file_sha(Path(protocol["run"]) / "frozen_protocol.json"),
        "source_matrix": str(matrix_root.resolve()),
        "source_matrix_completion_sha256": file_sha(matrix_root / "COMPLETE.json"),
    })
    _immutable_json(common / "INITIAL_STATE.json", state)
    return state


def phase_configuration(protocol: dict, arm: str, phase: str, state: dict, root: Path) -> dict:
    if arm not in protocol["phase_orders"] or phase not in protocol["phase_orders"][arm]:
        raise ValueError("Unknown frozen arm or phase")
    if state["protocol_sha256"] != file_sha(Path(protocol["run"]) / "frozen_protocol.json"):
        raise ValueError("Phase state differs from the frozen protocol")
    if contract_digest(read_json(Path(state["harness"]))) != state["harness_sha256"]:
        raise ValueError("Selected deployed harness changed")
    root.mkdir(parents=True, exist_ok=True)
    evaluation = read_json(Path(protocol["evaluation"]["task_manifest"]))
    eval_paths = evaluation_paths()
    heldout = {
        goal_group(eval_paths[index]) for index in
        evaluation["selection_task_ids"] + evaluation["report_task_ids"]
    }
    experiment = f"{Path(protocol['run']).name}-{arm}"
    if phase == "h2":
        config = read_json(Path(protocol["h2"]["template"]))
        packets_path = Path(state["packets"])
        if (
            file_sha(packets_path) != state["packets_sha256"]
            or Path(state["packet_solver"]).resolve() != Path(state["solver"]).resolve()
        ):
            raise ValueError("Engineer evidence belongs to a different frozen counterpart")
        packets = read_json(packets_path)
        for packet in packets:
            metadata = packet["metadata"]
            validate_metadata(metadata, state["solver_version"])
            if metadata["end"] > 240 or any(
                goal_group(eval_paths[index]) in heldout
                for index in range(metadata["start"], metadata["end"])
            ):
                raise ValueError("Engineer training overlaps held-out goals")
        rows = editor_dataset(
            packets, protocol["h2"]["steps"] * protocol["h2"]["batch_size"],
            state["solver_epoch_version"], sample_offset=protocol["h2"]["sample_offset"],
        )
        config.update(
            steps=protocol["h2"]["steps"], batch_size=protocol["h2"]["batch_size"],
            critic_warmup_steps=protocol["h2"]["critic_warmup_steps"],
            seed=protocol["h2"]["seed"], engineer_model=state["engineer"],
            engineer_critic=state["engineer_critic"], solver_model=state["solver"],
            solver_version=state["solver_version"], harness=protocol["no_patch"],
            tmpdir=f"/dev/shm/{experiment}-{phase}", interaction_credit=0.0,
        )
        module = "sao_bridge.alfworld_h_train"
        role, critic, partner = "editor", state["engineer_critic"], state["solver"]
    else:
        template = read_json(Path(protocol["w"]["template"]))
        keys = (
            "batch_size", "learning_rate", "critic_learning_rate", "solver_tokens",
            "temperature", "staleness", "actor_gpus", "rollout_gpus",
            "critic_warmup_steps", "save_interval_steps", "critic_updates_per_step",
            "critic_mode", "terminal_episodes", "solver_trajectory_style",
            "actor_gradient_checkpointing", "critic_gradient_checkpointing",
        )
        config = {key: template[key] for key in keys}
        offset = protocol["w"]["sample_offsets"][phase]
        count = protocol["w"]["steps_per_phase"] * protocol["w"]["batch_size"]
        catalogue_path = Path(template["training_catalogue"])
        games = training_catalogue(catalogue_path, template["training_catalogue_sha256"])
        full_rows = read_json(Path(template["training_data"]))
        if full_rows != solver_rows(
            games, template["training_catalogue_sha256"], catalogue_path=catalogue_path,
        ):
            raise ValueError("W data differs from its checksum-bound training catalogue")
        raw = full_rows[offset:offset + count]
        if len(raw) != count:
            raise ValueError("Declared W block exceeds its immutable training pool")
        if any(goal_group(games[row["task_id"]]) in heldout for row in raw):
            raise ValueError("W training overlaps held-out goals")
        rows = [dict(
            row, id=f"{row['id']}-{phase}-sample{offset + index}",
            partner_version=state["harness_version"],
        ) for index, row in enumerate(raw)]
        data = root / "data.json"
        assignments = root / "assignments.json"
        _immutable_json(assignments, no_patch_assignments(Path(state["harness"]), training_rows=rows))
        config.update(
            steps=protocol["w"]["steps_per_phase"], seed=protocol["w"]["seeds"][phase],
            solver_model=state["solver"], solver_critic=state["solver_critic"],
            harness=state["harness"], harness_version=state["harness_version"],
            data=str(data), assignments=str(assignments),
            environment_concurrency=template["training_environment_concurrency"],
            phase_timeout_seconds=template["training_timeout_seconds"],
            context_tokens=32768,
        )
        module = "sao_bridge.alfworld_w_train"
        role, critic, partner = "solver", state["solver_critic"], state["engineer"]
    data = root / "data.json"
    _immutable_json(data, rows)
    config.update(
        run=str(root / "attempt-01"), phase=phase, experiment=experiment, data=str(data),
        protocol=protocol["protocol"], exploratory_single_seed=True,
    )
    if not config["terminal_episodes"] or config["critic_mode"] != "learned":
        raise ValueError("Strong phases must retain corrected learned terminal training")
    if arm == "ma":
        if (
            state["calibration"]["split"] != "train"
            or state["calibration"]["solver_version"] != state["solver_version"]
            or not state["calibration"]["task_ids"]
            or any(type(index) is not int or not 0 <= index < 240
                   for index in state["calibration"]["task_ids"])
        ):
            raise ValueError("MA critic cannot consume development/report calibration")
        context = {
            "phase": "h2" if role == "editor" else "w", "actor_role": role,
            "partner_checkpoint": partner, "harness_source": "sha256:" + state["harness_sha256"],
            "calibration_accuracy": state["calibration"]["accuracy"],
            "calibration_split": "train",
            "partner_version": state["solver_epoch_version"] if role == "editor" else state["engineer_version"],
        }
        context_path = root / "joint_context.json"
        _immutable_json(context_path, context)
        config["central_context"] = str(context_path)
        _immutable_json(root / "joint_context_audit.json", joint_context_audit(context_path, Path(critic)))
    if role == "editor":
        from .alfworld_h_train import training_spec
        spec = training_spec(config, root / "attempt-01", "http://127.0.0.1:28600")
    else:
        from .alfworld_w_train import training_spec
        spec = training_spec(config, root / "attempt-01")
        _immutable_json(root / "schedule_audit.json", assignment_schedule_audit(config))
    validate_training_rows(spec, rows)
    _immutable_json(root / "capacity_audit.json", training_capacity_audit(spec, rows))
    _immutable_json(root / "input_state.json", state)
    _immutable_json(root / "phase_config.json", config)
    _immutable_json(root / "phase_identity.json", {
        "module": module, "config_sha256": file_sha(root / "phase_config.json"),
        "data_sha256": file_sha(data), "state_sha256": file_sha(root / "input_state.json"),
        "protocol_sha256": file_sha(Path(protocol["run"]) / "frozen_protocol.json"),
    })
    return config


def training_evidence(root: Path, config: dict) -> tuple[dict, dict]:
    if read_json(root / "COMPLETE.json")["status"] != "completed":
        raise ValueError("Native phase omitted a successful completion")
    if read_json(root / "config.json") != {**config, "run": str(root)}:
        raise ValueError("Native phase execution config changed")
    checked_release(root)
    trained = read_json(root / "optimizer/trained.json")
    warmup = config["critic_warmup_steps"]
    if (
        trained["actor_steps"] != config["steps"] - warmup
        or trained["critic_only_warmup_steps"] != warmup
        or trained["critic_steps"] != config["steps"] * config["critic_updates_per_step"]
        or not trained["terminal_episodes"] or trained["critic_mode"] != "learned"
        or trained["role"] not in {"editor", "solver"}
    ):
        raise ValueError("Native phase did not complete its Actor budget")
    lines = (root / "optimizer/metrics.jsonl").read_text().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    if [row["global_step"] for row in rows] != list(range(config["steps"])):
        raise ValueError("Native metric steps do not cover the frozen phase budget")
    nonzero = 0
    targets = []
    low_bound = -1.0 if trained["role"] == "editor" else 0.0
    for index, row in enumerate(rows):
        metrics = row["metrics"]
        low, high = (metrics["ppo_critic/update/target_value/min"],
                     metrics["ppo_critic/update/target_value/max"])
        validate_terminal_target_range(low, high, minimum_reward=low_bound)
        targets.extend((low, high))
        gradient = metrics.get("ppo_actor/update/grad_norm")
        if index < warmup:
            if gradient is not None:
                raise ValueError("Actor changed during critic-only warmup")
        else:
            if gradient is None or not math.isfinite(gradient) or gradient < 0:
                raise ValueError("Native Actor update omitted a finite gradient")
            if metrics["ppo_actor/n_seqs"] != config["batch_size"]:
                raise ValueError("Native Actor consumed a different batch shape")
            nonzero += gradient > 0
    audit = {
        "optimizer_steps": len(rows), "actor_steps": trained["actor_steps"],
        "nonzero_actor_gradients": nonzero, "warmup_isolated": True,
        "critic_targets_within_role_reward_range": True,
        "critic_target_range": [min(targets), max(targets)],
        "critic_target_roundoff_atol": NATIVE_TARGET_ROUNDOFF_ATOL,
        "metrics_sha256": file_sha(root / "optimizer/metrics.jsonl"),
        "native_completion_sha256": file_sha(root / "COMPLETE.json"),
    }
    return trained, audit


def execute_phase(root: Path, maximum_attempts: int = 2) -> dict:
    if type(maximum_attempts) is not int or maximum_attempts < 1:
        raise ValueError("Native phase retry budget must be a positive integer")
    identity = read_json(root / "phase_identity.json")
    for key, name in (("config_sha256", "phase_config.json"), ("data_sha256", "data.json"),
                      ("state_sha256", "input_state.json")):
        if file_sha(root / name) != identity[key]:
            raise ValueError(f"Frozen phase input changed: {name}")
    config = read_json(root / "phase_config.json")
    attempts_path = root / "attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else []
    launch = Launcher(root / "supervision")
    try:
        for attempt in range(1, maximum_attempts + 1):
            output = root / f"attempt-{attempt:02d}"
            if (output / "COMPLETE.json").exists() and (output / "RELEASED.json").exists():
                trained, audit = training_evidence(output, config)
                complete = {
                    "status": "completed", "training_root": str(output),
                    "trained": trained, "attempts": attempts,
                    "input_identity": identity, "native_audit": audit,
                }
                return publish_phase_completion(root, complete)
            if output.exists() or attempt <= len(attempts):
                continue
            current = copy.deepcopy(config)
            current["run"] = str(output)
            config_path = root / f"config-{attempt:02d}.json"
            _immutable_json(config_path, current)
            attempts.append({"attempt": attempt, "started_at_unix": time.time(),
                             "host": socket.gethostname(), "config_sha256": file_sha(config_path)})
            write_json(attempts_path, attempts)
            process = launch.module(
                identity["module"], ["--config", config_path],
                gpus="0,1,2,3,4,5,6,7", name=f"native-attempt-{attempt}",
            )
            try:
                launch.wait(process, "strong native phase",
                            timeout=float(current["phase_timeout_seconds"]) + 600)
            except (RuntimeError, TimeoutError) as exc:
                attempts[-1].update(finished_at_unix=time.time(), error=repr(exc))
                write_json(attempts_path, attempts)
                raise
            trained, audit = training_evidence(output, config)
            attempts[-1].update(finished_at_unix=time.time(), complete=True)
            write_json(attempts_path, attempts)
            complete = {"status": "completed", "training_root": str(output),
                        "trained": trained, "attempts": attempts, "input_identity": identity,
                        "native_audit": audit}
            return publish_phase_completion(root, complete)
        raise PhaseAttemptsExhausted(f"Strong native phase exhausted {maximum_attempts} attempts")
    finally:
        launch.close()
        write_json(root / "RELEASED.json", {"released_at": time.time(), **gpu_release_audit()})


def run_phase(
    protocol_path: Path, arm: str, phase: str, root: Path, *,
    matrix_root: Path | None = None, state_path: Path | None = None,
) -> dict:
    protocol = read_json(protocol_path)
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        failed = root / "FAILED.json"
        if failed.exists():
            archive = root / "controller_failures"
            archive.mkdir(exist_ok=True)
            destination = archive / f"{len(list(archive.glob('*.json'))) + 1:03d}.json"
            failed.rename(destination)
        try:
            if state_path:
                state = read_json(state_path)
            elif phase == protocol["phase_orders"][arm][0]:
                if arm == "sequential":
                    if matrix_root is None:
                        raise ValueError("Initial sequential phase requires --matrix-root")
                    write_json(root / "STATUS.json", {"stage": "waiting_common_matrix"})
                    wait_completed(matrix_root)
                    state = finalize_common(protocol, matrix_root)
                else:
                    shared = Path(protocol["run"]) / "common/INITIAL_STATE.json"
                    deadline = time.monotonic() + 10800
                    while not shared.exists():
                        if time.monotonic() >= deadline:
                            raise TimeoutError("Common selected state is not available")
                        time.sleep(10)
                    state = read_json(shared)
            else:
                raise ValueError("Continuation phases require an explicit selected --state")
            if state["protocol_sha256"] != file_sha(protocol_path):
                raise ValueError("Selected state belongs to a different frozen protocol")
            phase_configuration(protocol, arm, phase, state, root)
            write_json(root / "STATUS.json", {"stage": "native_training", "phase": phase})
            result = execute_phase(root, protocol["max_phase_attempts"])
            write_json(root / "STATUS.json", {
                "stage": "completed", "phase": phase, "training_root": result["training_root"],
            })
            return result
        except BaseException as exc:
            write_json(failed, {"error": repr(exc), "phase": phase, "failed_at_unix": time.time()})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("sequential", "ma"), required=True)
    parser.add_argument("--phase", choices=("h2", "w1", "w2"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    try:
        result = run_phase(
            args.protocol, args.arm, args.phase, args.root,
            matrix_root=args.matrix_root, state_path=args.state,
        )
    except (ValueError, PhaseAttemptsExhausted) as exc:
        raise SystemExit(78) from exc
    print(result, flush=True)


if __name__ == "__main__":
    main()
