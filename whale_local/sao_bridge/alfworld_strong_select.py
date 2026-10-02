"""Matched proposal/checkpoint selection after a complete strong native phase."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import threading
import time

from .alfworld_engineer_eval import Jobs
from .alfworld_phase_data import editor_dataset
from .alfworld_solver_validation import select_checkpoint
from .alfworld_strong_phase import (
    checked_native_report, checked_release, checkpoint_pair, training_evidence,
)
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import ensure_model_runtime_metadata, file_sha, read_json, write_json
from .gpu_telemetry import monitor, summarize
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


class SelectionAttemptsExhausted(RuntimeError):
    pass


def selection_binding(protocol: dict, state: dict) -> dict:
    evaluation = protocol["evaluation"]
    manifest = read_json(Path(evaluation["task_manifest"]))
    return {
        "solver_model": str(Path(state["solver"]).resolve()),
        "harness_sha256": state["harness_sha256"],
        "report_sha256": file_sha(Path(state["selection_report"])),
        "task_ids": manifest["selection_task_ids"],
        "evaluation": {key: evaluation[key] for key in (
            "samples", "seed", "temperature", "solver_tokens", "eval_concurrency",
            "environment_concurrency", "matrix_concurrency",
        )},
        "protocol_sha256": state["protocol_sha256"],
    }


def publish_selection_binding(protocol: dict, state: dict) -> None:
    path = Path(state["selection_report"]).with_suffix(".binding.json")
    _immutable_json(path, selection_binding(protocol, state))


def freeze_selection_identity(root: Path, identity: dict) -> None:
    path = root / "identity.json"
    if path.is_file():
        legacy = {
            **identity, "native_audit": {
                key: value for key, value in identity["native_audit"].items()
                if key not in {"critic_target_range", "critic_target_roundoff_atol"}
            },
        }
        if legacy != identity and read_json(path) == legacy:
            _immutable_json(root / "target_range_audit.json", {
                "selection_identity_sha256": file_sha(path),
                "native_audit": identity["native_audit"],
                "migration": "Add raw target extrema/tolerance; retain every original binding.",
            })
            return
    _immutable_json(path, identity)


def supplement_runtime(checkpoint: Path, reference: Path) -> None:
    if any(not (checkpoint / name).is_file() for name in (
        "preprocessor_config.json", "processor_config.json",
    )):
        ensure_model_runtime_metadata(checkpoint, reference)


def matrix_config(
    protocol: dict, solver: str, version: str, harnesses: dict[str, str], root: Path,
    *, split: str = "selection",
) -> dict:
    if split not in {"selection", "report"}:
        raise ValueError("Unknown paired evaluation split")
    evaluation = protocol["evaluation"]
    manifest = read_json(Path(evaluation["task_manifest"]))
    return {
        "run": str(root), "solver_model": solver, "solver_version": version,
        "no_patch": protocol["no_patch"], "harnesses": harnesses,
        "task_ids": manifest[f"{split}_task_ids"],
        **{key: evaluation[key] for key in (
            "samples", "seed", "temperature", "solver_tokens", "eval_concurrency",
            "environment_concurrency", "matrix_concurrency",
        )},
    }


def checked_matrix(output: Path, config: dict) -> dict:
    recorded = read_json(output / "config.json")
    if any(recorded.get(key) != value for key, value in config.items()):
        raise ValueError("Cached matrix inputs changed")
    checked_release(output)
    if read_json(output / "COMPLETE.json")["status"] != "completed":
        raise ValueError("Matrix did not complete successfully")
    report = read_json(output / "report.json")
    expected = {"no_patch": config["no_patch"], **config["harnesses"]}
    if set(report["variants"]) != set(expected):
        raise ValueError("Matrix lost or added logical harnesses")
    for label, path in expected.items():
        if report["variants"][label]["harness_sha256"] != contract_digest(read_json(Path(path))):
            raise ValueError("Matrix evaluated a different harness artifact")
    return report


def run_matrix(launch: Launcher, config: dict, root: Path, attempts: int) -> tuple[dict, Path]:
    _immutable_json(root / "request.json", config)
    for attempt in range(1, attempts + 1):
        output = root / f"attempt-{attempt:02d}"
        current = {**config, "run": str(output)}
        config_path = root / f"config-{attempt:02d}.json"
        _immutable_json(config_path, current)
        if (output / "COMPLETE.json").is_file() and (output / "RELEASED.json").is_file():
            return checked_matrix(output, current), output
        if output.exists():
            continue
        process = launch.module(
            "sao_bridge.alfworld_solver_eval",
            ["--config", config_path, "--harness-matrix"],
            gpus="0,1,2,3,4,5,6,7", name=f"{root.name}-attempt-{attempt}",
        )
        launch.wait(process, "strong selection matrix", timeout=14400)
        return checked_matrix(output, current), output
    raise SelectionAttemptsExhausted(f"Selection matrix exhausted {attempts} attempts: {root}")


def matrix_reports(matrix: dict, protocol: dict, *, split: str = "selection") -> dict[str, dict]:
    if split not in {"selection", "report"}:
        raise ValueError("Unknown paired evaluation split")
    task_ids = read_json(Path(protocol["evaluation"]["task_manifest"]))[f"{split}_task_ids"]
    reports = {}
    for label, value in matrix["variants"].items():
        report = checked_native_report(
            read_json(Path(value["report"])), task_ids, protocol["evaluation"],
        )
        if report["accuracy"] != value["accuracy"]:
            raise ValueError("Matrix summary and complete trajectories disagree")
        reports[label] = report
    return reports


def select_engineer(
    reports: dict[str, dict], proposals: dict[str, list[dict]], minimum_delta: float,
) -> dict:
    if set(proposals) != {"incumbent", "step15", "step31"}:
        raise ValueError("Strong H2 selection requires its incumbent and both trained checkpoints")
    accuracies = {}
    for label, values in proposals.items():
        if [value["sample_index"] for value in values] != list(range(8)):
            raise ValueError("Engineer selection lost proposal slots")
        accuracies[label] = sum(
            reports[f"{label}-proposal{index:02d}"]["accuracy"] for index in range(8)
        ) / 8
    qualified = [
        label for label in ("step15", "step31")
        if accuracies[label] - accuracies["incumbent"] >= minimum_delta
    ]
    selected = max(qualified, key=lambda label: (accuracies[label], int(label[4:]))) if qualified else "incumbent"
    allowed = ["incumbent_harness", "no_patch", *[
        f"{selected}-proposal{index:02d}" for index in range(8)
    ]]
    incumbent_accuracy = reports["incumbent_harness"]["accuracy"]
    qualified_harnesses = [
        label for label in allowed
        if label != "incumbent_harness"
        and reports[label]["accuracy"] - incumbent_accuracy >= minimum_delta
    ]
    harness = min(
        qualified_harnesses, key=lambda label: (-reports[label]["accuracy"], label),
    ) if qualified_harnesses else "incumbent_harness"
    return {
        "selected_engineer": selected, "selected_harness": harness,
        "engineer_accuracies": accuracies,
        "engineer_metric": "mean over eight proposals of native Solver mean@2",
        "valid_proposals": {label: sum(row["valid"] for row in values) for label, values in proposals.items()},
        "useful_editing_evidence": (
            sum(row["valid"] for row in proposals[selected]) > 0
            and accuracies[selected] > reports["no_patch"]["accuracy"]
        ),
        "harness_accuracy": reports[harness]["accuracy"],
        "incumbent_harness_accuracy": incumbent_accuracy,
        "minimum_delta": minimum_delta,
    }


def generate_proposals(
    launch: Launcher, protocol: dict, state: dict, pairs: dict, root: Path,
) -> dict:
    packets = read_json(Path(state["packets"]))
    if not packets or file_sha(Path(state["packets"])) != state["packets_sha256"]:
        raise ValueError("Frozen Engineer prompt packets are unavailable or changed")
    chosen = [packets[index * len(packets) // 8] for index in range(8)]
    prompts = editor_dataset(chosen, 8, state["solver_epoch_version"])
    _immutable_json(root / "generation_prompts.json", prompts)
    template = read_json(Path(protocol["h2"]["template"]))
    proposals = {}
    for label, pair in pairs.items():
        output = root / "proposals" / f"{label}.json"
        if output.exists():
            proposals[label] = read_json(output)
            if len(proposals[label]) != 8:
                raise ValueError("Cached Engineer checkpoint lost proposal slots")
            for index, value in enumerate(proposals[label]):
                name = f"{label}-proposal{index:02d}"
                request = read_json(root / "jobs" / name / "request.json")
                expected = {
                    "kind": "generate", "name": name, "sample_index": index,
                    "prompt_id": prompts[index]["id"], "model_label": label,
                    "engineer_checkpoint": pair["actor"],
                    "decode_seed": protocol["h2"]["proposal_seed"] + 1009 * index,
                    "temperature": template["temperature"], "tokens": template["engineer_tokens"],
                    "messages": prompts[index]["messages"],
                }
                if (
                    any(request.get(key) != item for key, item in expected.items())
                    or value != read_json(root / "jobs" / name / "result.json")
                    or value["request_sha256"] != contract_digest(request)
                ):
                    raise ValueError("Cached matched proposal identity changed")
            continue
        supplement_runtime(Path(pair["actor"]), Path(state["engineer"]))
        with launch.vllm(
            pair["actor"], port=19601, gpus="0,1,2,3,4,5,6,7",
            name=f"engineer-{label}", data_parallel=8, max_model_len=32768,
            tool_call_parser="qwen3_xml", gdn_prefill_backend="triton",
            language_model_only=True, max_num_seqs=8,
            server_args=["--enable-prefix-caching"],
        ) as urls:
            endpoints = urls.split(",")
            jobs = Jobs(root / "jobs", launch, concurrency=8)
            names = []
            for index, row in enumerate(prompts):
                name = f"{label}-proposal{index:02d}"
                names.append(name)
                jobs.add({
                    "kind": "generate", "name": name, "sample_index": index,
                    "prompt_id": row["id"], "model_label": label,
                    "engineer_checkpoint": pair["actor"],
                    "decode_seed": protocol["h2"]["proposal_seed"] + 1009 * index,
                    "temperature": template["temperature"], "tokens": template["engineer_tokens"],
                    "messages": row["messages"], "url": endpoints[index % len(endpoints)],
                })
            results = jobs.wait()
            proposals[label] = [results[name] for name in names]
            write_json(output, proposals[label])
    return proposals


def select_harness_phase(
    launch: Launcher, protocol: dict, state: dict, trained: dict, root: Path,
) -> tuple[dict, dict]:
    pairs = {
        "incumbent": {"actor": state["engineer"], "critic": state["engineer_critic"]},
        **{f"step{step}": checkpoint_pair(trained, step) for step in protocol["h2"]["checkpoint_steps"]},
    }
    proposals = generate_proposals(launch, protocol, state, pairs, root)
    harnesses = {"incumbent_harness": state["harness"]}
    for label, values in proposals.items():
        for value in values:
            name = f"{label}-proposal{value['sample_index']:02d}"
            if value["valid"]:
                path = root / "patches" / f"{name}.json"
                _immutable_json(path, value["patch"])
            else:
                path = Path(protocol["no_patch"])
            harnesses[name] = str(path)
    config = matrix_config(protocol, state["solver"], state["solver_version"], harnesses, root / "matrix")
    matrix, output = run_matrix(launch, config, root / "matrix", protocol["max_phase_attempts"])
    reports = matrix_reports(matrix, protocol)
    decision = select_engineer(reports, proposals, protocol["evaluation"]["promotion_min_delta"])
    selected = decision["selected_engineer"]
    state = dict(state)
    if selected != "incumbent":
        state.update(
            engineer=pairs[selected]["actor"], engineer_critic=pairs[selected]["critic"],
            engineer_version=state["engineer_version"] + 1,
        )
    harness_label = decision["selected_harness"]
    source = protocol["no_patch"] if harness_label == "no_patch" else harnesses[harness_label]
    patch = read_json(Path(source))
    identity = contract_digest(patch)
    chosen_path = root / "selected_harness.json"
    _immutable_json(chosen_path, patch)
    state.update(
        harness=str(chosen_path), harness_sha256=identity,
        harness_version=state["harness_version"] + int(identity != state["harness_sha256"]),
    )
    _immutable_json(root / "selected_report.json", reports[harness_label])
    state.update(
        selection_report=str(root / "selected_report.json"),
        selection_accuracy=reports[harness_label]["accuracy"],
        selection_no_patch_accuracy=reports["no_patch"]["accuracy"],
        selection_label=harness_label, source_matrix=str(output),
    )
    return state, decision


def select_weight_phase(
    launch: Launcher, protocol: dict, state: dict, trained: dict, root: Path,
) -> tuple[dict, dict]:
    binding_path = Path(state["selection_report"]).with_suffix(".binding.json")
    if read_json(binding_path) != selection_binding(protocol, state):
        raise ValueError("Incumbent selection report belongs to a different model/harness or decode protocol")
    control = checked_native_report(
        read_json(Path(state["selection_report"])),
        read_json(Path(protocol["evaluation"]["task_manifest"]))["selection_task_ids"],
        protocol["evaluation"],
    )
    candidates, pairs, candidate_reports, matrices = [], {}, {}, {}
    harnesses = {
        "deployed": state["harness"],
        "common_h1": str(Path(protocol["run"]) / "common/selected_harness.json"),
    }
    for step in protocol["w"]["checkpoint_steps"]:
        pair = checkpoint_pair(trained, step)
        pairs[step] = pair
        supplement_runtime(Path(pair["actor"]), Path(state["solver"]))
        config = matrix_config(
            protocol, pair["actor"], f"strong-native-step{step}-{Path(root).parent.name}",
            harnesses, root / f"step{step}",
        )
        matrix, output = run_matrix(launch, config, root / f"step{step}", protocol["max_phase_attempts"])
        reports = matrix_reports(matrix, protocol)
        candidate_reports[step] = reports
        matrices[step] = output
        candidates.append({"global_step": step, "path": pair["actor"], "evaluation": reports["deployed"]})
    decision = select_checkpoint(
        control, candidates, protocol["evaluation"]["seed"],
        minimum_delta=protocol["evaluation"]["promotion_min_delta"],
    )
    state = dict(state)
    chosen = decision["selected"]["global_step"]
    if chosen is not None:
        pair = pairs[chosen]
        _, weights = model_digests(Path(pair["actor"]))
        state.update(
            solver=pair["actor"], solver_critic=pair["critic"],
            solver_epoch_version=state["solver_epoch_version"] + 1,
            solver_version=f"strong-native-{weights[:16]}", solver_weights_sha256=weights,
            source_matrix=str(matrices[chosen]),
            selection_no_patch_accuracy=candidate_reports[chosen]["no_patch"]["accuracy"],
        )
        report = candidate_reports[chosen]["deployed"]
    else:
        decision["reason"] = "No checkpoint cleared the frozen floor; retained the current accepted Solver"
        report = control
    _immutable_json(root / "selected_report.json", report)
    state.update(
        selection_report=str(root / "selected_report.json"),
        selection_accuracy=report["accuracy"], selection_label=f"step{chosen}" if chosen is not None else "incumbent",
    )
    return state, decision


def run(protocol_path: Path, arm: str, phase: str, training_root: Path, root: Path) -> dict:
    protocol = read_json(protocol_path)
    if phase not in protocol["phase_orders"][arm]:
        raise ValueError("Unknown strong phase")
    state = read_json(training_root / "input_state.json")
    config = read_json(training_root / "phase_config.json")
    completion = read_json(training_root / "PHASE_COMPLETE.json")
    trained, native_audit = training_evidence(Path(completion["training_root"]), config)
    checked_release(training_root)
    if state["protocol_sha256"] != file_sha(protocol_path):
        raise ValueError("Selection state belongs to a different protocol")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze_selection_identity(root, {
            "protocol_sha256": file_sha(protocol_path), "arm": arm, "phase": phase,
            "input_state_sha256": file_sha(training_root / "input_state.json"),
            "phase_completion_sha256": file_sha(training_root / "PHASE_COMPLETE.json"),
            "native_audit": native_audit,
        })
        if (root / "COMPLETE.json").exists() and (root / "RELEASED.json").exists():
            checked_release(root)
            complete = read_json(root / "COMPLETE.json")
            if complete["selected_state_sha256"] != file_sha(root / "SELECTED_STATE.json"):
                raise ValueError("Completed selected state changed")
            return complete
        failed = root / "FAILED.json"
        if failed.exists():
            archive = root / "controller_failures"
            archive.mkdir(exist_ok=True)
            failed.rename(archive / f"{len(list(archive.glob('*.json'))) + 1:03d}.json")
        write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
        launch = Launcher(root / "supervision")
        stop = threading.Event()
        telemetry = threading.Thread(target=monitor, args=(root / "gpu_process_samples.jsonl", stop, 5), daemon=True)
        telemetry.start()
        started = time.monotonic()
        try:
            if phase == "h2":
                state, decision = select_harness_phase(launch, protocol, state, trained, root)
            else:
                state, decision = select_weight_phase(launch, protocol, state, trained, root)
            _immutable_json(root / "decision.json", decision)
            publish_selection_binding(protocol, state)
            _immutable_json(root / "SELECTED_STATE.json", state)
            complete = {
                "status": "completed", "arm": arm, "phase": phase, "elapsed_seconds": time.monotonic() - started,
                "selected_state": str(root / "SELECTED_STATE.json"),
                "selected_state_sha256": file_sha(root / "SELECTED_STATE.json"),
                "decision": decision, "report_opened": False,
                "counterpart_packets_current": Path(state["packet_solver"]).resolve() == Path(state["solver"]).resolve(),
            }
            write_json(root / "COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
            raise
        finally:
            launch.close()
            stop.set()
            telemetry.join(timeout=15)
            path = root / "gpu_process_samples.jsonl"
            if path.is_file():
                try:
                    summary = summarize(path)
                except ValueError as exc:
                    summary = {"status": "invalid", "error": repr(exc)}
                write_json(root / "gpu_utilization_summary.json", summary)
            write_json(root / "RELEASED.json", {"released_at": time.time(), **gpu_release_audit()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("sequential", "ma"), required=True)
    parser.add_argument("--phase", choices=("h2", "w1", "w2"), required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.protocol, args.arm, args.phase, args.training_root.resolve(), args.root), flush=True)


if __name__ == "__main__":
    main()
