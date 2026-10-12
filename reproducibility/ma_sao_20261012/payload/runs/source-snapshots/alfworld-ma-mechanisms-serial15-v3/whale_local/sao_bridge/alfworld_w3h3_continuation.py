"""One ordinary SAO H3-then-W3 cycle from the user-selected W2 recovery peak."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
from pathlib import Path
import subprocess
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_fixed_solver_engineer import select_side
from .alfworld_h_train import training_spec as h_spec
from .alfworld_phase_data import editor_dataset
from .alfworld_reward_service import validate_metadata
from .alfworld_sao_control import backend_wiring_audit, role_config
from .alfworld_solver_validation import no_patch_assignments
from .alfworld_strong_cost import native_generation, proposal_work, reward_work
from .alfworld_strong_phase import (
    checked_native_report, checked_release, checkpoint_pair, training_evidence,
)
from .alfworld_strong_refresh import run as refresh_partner
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json, evaluation_paths, goal_group, training_catalogue
from .alfworld_w_train import assignment_schedule_audit, training_spec as w_spec
from .analysis import paired_difference
from .budget import contract_digest, model_digests
from .common import ROOT, file_sha, read_json, write_json
from .large_train import training_capacity_audit, validate_training_rows
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node, research_resources_released

HISTORY = ROOT / "runs/alfworld_alternating_e15_w79_8h_v1"
RECOVERY = ROOT / "runs/alfworld_w2_recovery_v2"
CORE = ROOT / "runs/alfworld_core_200_oneshot_v2/protocol.json"
ACCELERATION = ROOT / "runs/alfworld_fixed_w1_h3_v1/training_config.json"
ACCELERATION_KEYS = (
    "training_kernel_overlay", "training_kernel_versions", "training_kernel_validation",
)


def early_pair(trained: dict, evaluation: dict, scores: dict) -> dict:
    pair = checkpoint_pair(trained, 39)
    if (evaluation["solver_model"] != pair["actor"] or scores["h2"]["all200"] != 0.795
            or "w2_recovery" not in pair["actor"]):
        raise ValueError("The requested approximately79% W2 is recovery-extra40, not step79")
    return pair


def solver_rows_for_cycle(previous: list[dict]) -> list[dict]:
    if len(previous) != 640 or len({row["id"] for row in previous}) != 640:
        raise ValueError("W3 must reuse the exact640-question recovery training sequence")
    return [
        {**row, "id": f"{row['id']}-w3-{index}", "partner_version": 3}
        for index, row in enumerate(previous)
    ]


def configs(root: Path, state: dict, acceleration: dict) -> tuple[dict, dict]:
    h = role_config("h2", root, state)
    h.update(
        run=str(root / "h3/training"), phase="h3", experiment=root.name,
        data=str(root / "h3/data.json"), harness=str(ROOT / "examples/alfworld_no_patch.json"),
        tmpdir=str(root / "h3/runtime_cache"), save_interval_steps=32,
        rollout_setup_timeout=1800, **acceleration,
    )
    w = copy.deepcopy(read_json(RECOVERY / "training_config.json"))
    w.update(
        run=str(root / "w3/training"), phase="w3", experiment=root.name,
        solver_model=state["solver"], solver_critic=state["solver_critic"],
        harness=state["harness"], harness_version=3, data=str(root / "w3/data.json"),
        assignments=str(root / "w3/assignments.json"),
        steps=80, batch_size=8, save_interval_steps=20,
        independent_trajectories=True, processed_behavior_logprobs=True,
        rollout_setup_timeout=1800, **acceleration,
    )
    return h, w


def prepare(root: Path) -> dict:
    if (root / "protocol.json").exists():
        raise RuntimeError("Do not refreeze an existing W3/H3 experiment")
    trained = read_json(RECOVERY / "training/optimizer/trained.json")
    old_evaluation = read_json(RECOVERY / "evaluation/extra40/request.json")
    old_scores = read_json(RECOVERY / "extra40_COMPLETE.json")
    pair = early_pair(trained, old_evaluation, old_scores)
    for item in old_scores.values():
        if file_sha(Path(item["report"])) != item["sha256"]:
            raise ValueError("The recorded early-W2 evaluation changed")
    state = read_json(HISTORY / "sequential/STATE_AFTER_h2.json")
    h_pair = checkpoint_pair(read_json(HISTORY / "sequential/h2/attempt-01/optimizer/trained.json"), 31)
    if (state["engineer"], state["engineer_critic"]) != (h_pair["actor"], h_pair["critic"]):
        raise ValueError("H3 must start at the original selected H2 Actor and Critic")
    if contract_digest(read_json(Path(state["harness"]))) != contract_digest(read_json(RECOVERY / "fixed_h2.json")):
        raise ValueError("The recovered W2 did not use the original H2 strategy")
    model_files = []
    for checkpoint in (*pair.values(), *h_pair.values()):
        directory = Path(checkpoint)
        if not directory.is_dir() or not (directory / "model.safetensors").is_file():
            raise ValueError(f"Missing saved continuation checkpoint: {directory}")
        if list(directory.glob("*.distcp")) or list(directory.glob("adam*.pt")):
            raise ValueError("Native optimizer state exists; review the declared fresh-Adam boundary")
        model_files.extend(path for path in directory.iterdir() if path.is_file())
    _, weights = model_digests(Path(pair["actor"]))
    state.update(
        solver=pair["actor"], solver_critic=pair["critic"], solver_epoch_version=2,
        solver_version=f"w2-recovery-extra40-{weights[:12]}",
        solver_weights_sha256=weights, harness=str(RECOVERY / "fixed_h2.json"),
        harness_sha256=contract_digest(read_json(RECOVERY / "fixed_h2.json")),
    )
    base = read_json(HISTORY / "frozen_protocol.json")
    core = read_json(CORE)
    validate_cohort(core)
    previous_rows = read_json(RECOVERY / "data.json")
    rows = solver_rows_for_cycle(previous_rows)
    catalogue = Path(rows[0]["task_catalogue"])
    catalogue_sha = rows[0]["task_catalogue_sha256"]
    games = training_catalogue(catalogue, catalogue_sha)
    reserved = {goal_group(game) for game in evaluation_paths()}
    if any(goal_group(games[row["task_id"]]) in reserved for row in rows):
        raise ValueError("The unchanged W3 train-only population overlaps the original500")
    _immutable_json(root / "w3/data.json", rows)
    acceleration = {key: read_json(ACCELERATION)[key] for key in ACCELERATION_KEYS}
    h, w = configs(root, state, acceleration)
    _immutable_json(root / "h3_template.json", h)
    _immutable_json(root / "w3_template.json", w)
    _immutable_json(root / "BACKEND_VALIDATION.json", {
        "h3": backend_wiring_audit(h, role="editor"),
        "w3": backend_wiring_audit(w, role="solver"),
    })
    protocol = {
        "protocol": "early-w2-extra40-h3-w3-ordinary-sao-v1", "root": str(root),
        "source": str(ROOT), "initial_pair": pair, "initial_engineer_pair": h_pair,
        "historical_h2": state["harness"], "initial_h2_accuracy_reference": 0.795,
        "initial_choice": "User-selected recovery-extra40 peak from an already inspected200-task curve; retrospective early stopping, not prospective test selection.",
        "phase_order": ["h3", "w3"],
        "h3": {"steps": 32, "batch_size": 6, "actor_updates": 27, "critic_updates": 32},
        "w3": {"steps": 80, "batch_size": 8, "actor_updates": 75, "critic_updates": 80},
        "consumed_role_trajectories": {"engineer": 192, "solver": 640},
        "engineer_training_population": list(range(240)),
        "engineer_sample_offset": base["h2"]["sample_offset"] + 192,
        "solver_training_population": "Exactly the same640 instances and order as W2 recovery; only trajectory IDs and partner version change.",
        "h2": {**base["h2"], "checkpoint_steps": [31]},
        "evaluation": {**base["evaluation"], "eval_concurrency": 128, "environment_concurrency": 32},
        "core200_protocol": str(CORE), "no_patch": str(ROOT / "examples/alfworld_no_patch.json"),
        "selection_rule": "Eight matched before/after proposals on original selection24 mean@2. Train the experimental W3 branch against the best valid actual H3 candidate, even when the old0.02 deployment gate retains h2. Never rename an h2 fallback H3.",
        "failure_rule": "If all eight H3 candidates are invalid, explicitly stop without pretending an incumbent fallback is learned H3.",
        "checkpoint_rule": "Start from saved W2-extra40; H3 endpoint is extra32 and W3 endpoint is extra80. Save W3 at20-step intervals but do not select checkpoints on main200.",
        "optimizer_state": "The selected W2-extra40 and H2-step31 exports contain HF Actor/Critic weights only; one fresh-Adam boundary per role and the original five critic-warmup steps.",
        "backend": "Corrected ordinary SAO, independent sequences and processed behavior logprobs, K1, original learning rates, no MA/CTDE, ValueNorm or KL.",
        "backend_change_from_recovery": "The earlier recovery had independent sequences but not processed-behavior-logprob wiring; use the already validated repaired ordinary-SAO path for this new cycle.",
        "fresh100_used": False, "maximum_attempts_per_stage": 1,
        "limitations": [
            "The original200 is repeatedly used development data, not a sealed test",
            "The starting79.5% checkpoint was retrospectively requested after its curve was observed",
            "One continuation seed and no equal-total-cost algorithm attribution",
            "Matched proposal selection overlaps some of the original200; report the outside-selection subset too",
        ],
    }
    _immutable_json(root / "protocol.json", protocol)
    state["protocol_sha256"] = file_sha(root / "protocol.json")
    _immutable_json(root / "input_state.json", state)
    files = model_files + list((ROOT / "whale_local").rglob("*.py"))
    files += [
        CORE, HISTORY / "frozen_protocol.json", HISTORY / "sequential/STATE_AFTER_h2.json",
        RECOVERY / "protocol.json", RECOVERY / "data.json", RECOVERY / "fixed_h2.json",
        RECOVERY / "training/optimizer/trained.json", RECOVERY / "extra40_COMPLETE.json",
        RECOVERY / "evaluation/extra40/request.json", RECOVERY / "training_config.json",
        Path(protocol["no_patch"]), Path(protocol["h2"]["template"]),
        Path(protocol["evaluation"]["task_manifest"]), Path(acceleration["training_kernel_validation"]),
        root / "protocol.json", root / "input_state.json", root / "h3_template.json",
        root / "w3_template.json", root / "w3/data.json", root / "BACKEND_VALIDATION.json",
        catalogue,
    ]
    files += [Path(item["report"]) for item in old_scores.values()]
    _immutable_json(root / "EXECUTION_PINS.json", {
        "pins": {str(path): file_sha(path) for path in files}, "at_unix": time.time(),
    })
    return protocol


def freeze_config(root: Path, config: dict, role: str) -> dict:
    phase = root / ("h3" if role == "editor" else "w3")
    paths = [root / "protocol.json", phase / "data.json", Path(config["training_kernel_validation"])]
    if role == "solver":
        paths.append(Path(config["assignments"]))
    else:
        paths += [root / "partner_w2/REFRESHED_STATE.json", root / "partner_w2/packets.json"]
    config = {**config, "pinned_files": {str(path): file_sha(path) for path in paths}}
    spec = h_spec(config, Path(config["run"]), "http://127.0.0.1:28600") if role == "editor" else w_spec(
        config, Path(config["run"]))
    rows = read_json(Path(config["data"]))
    validate_training_rows(spec, rows)
    _immutable_json(phase / "capacity_audit.json", training_capacity_audit(spec, rows))
    _immutable_json(phase / "backend_audit.json", backend_wiring_audit(config, role=role))
    if role == "solver":
        _immutable_json(phase / "schedule_audit.json", assignment_schedule_audit(config))
    _immutable_json(phase / "config.json", config)
    return config


def require_idle() -> dict:
    release = gpu_release_audit()
    if not research_resources_released(release):
        raise RuntimeError(f"The ordinary-SAO cycle cannot use occupied research GPUs: {release}")
    return release


def train_phase(launch: Launcher, root: Path, phase: str, config: dict) -> dict:
    require_idle()
    write_json(root / "STATUS.json", {"phase": f"training_{phase}", "at_unix": time.time()})
    module = "sao_bridge.alfworld_h_train" if phase == "h3" else "sao_bridge.alfworld_w_train"
    process = launch.module(module, ["--config", root / phase / "config.json"],
                            gpus="0,1,2,3,4,5,6,7", name=f"{phase}-native")
    launch.wait(process, f"{phase} ordinary SAO training", timeout=config["phase_timeout_seconds"] + 600)
    trained, audit = training_evidence(Path(config["run"]), config)
    result = {
        "status": "passed", "native": audit,
        "generation": native_generation(Path(config["run"]) / "optimizer"),
        "config_sha256": file_sha(root / phase / "config.json"),
    }
    if phase == "h3":
        result["reward_work"] = reward_work(Path(config["run"]))
    _immutable_json(root / phase / "TRAINING_AUDIT.json", result)
    return trained


def select_h3(launch: Launcher, root: Path, protocol: dict, state: dict, trained: dict) -> str:
    selection = root / "selection"
    proposals = generate_proposals(launch, protocol, state, {
        "before": protocol["initial_engineer_pair"], "after": checkpoint_pair(trained, 31),
    }, selection)
    harnesses = {"incumbent_harness": protocol["historical_h2"]}
    for label, values in proposals.items():
        for value in values:
            name = f"{label}-proposal{value['sample_index']:02d}"
            path = selection / "patches" / f"{name}.json"
            _immutable_json(path, value["patch"] if value["valid"] else read_json(Path(protocol["no_patch"])))
            harnesses[name] = str(path)
    config = matrix_config(protocol, state["solver"], state["solver_version"], harnesses, selection / "matrix")
    matrix, output = run_matrix(launch, config, selection / "matrix", 1)
    reports = matrix_reports(matrix, protocol)
    sides = {label: select_side(values, reports, label, protocol["evaluation"]["promotion_min_delta"])
             for label, values in proposals.items()}
    comparison = paired_difference(sides["before"]["population"], sides["after"]["population"], seed=9000081)
    for label, side in sides.items():
        _immutable_json(root / f"{label}_proposal_population.json", side.pop("population"))
    candidate = sides["after"]["best_valid_proposal"]
    _immutable_json(root / "H3_SELECTION.json", {
        "sides": sides, "population_comparison": comparison,
        "actual_h3_candidate": harnesses[candidate] if candidate is not None else None,
        "deployment_harness": harnesses[sides["after"]["selected"]],
        "experimental_w3_uses_actual_h3_not_deployment_fallback": True,
        "main200_not_opened_for_new_cycle": True, "matrix": str(output),
        "proposal_work": proposal_work(selection),
    })
    if candidate is None:
        _immutable_json(root / "MODEL_FAILURE.json", {
            "status": "all_h3_candidates_invalid", "new_solver_updates": 0,
            "incumbent_fallback_not_misreported_as_h3": True,
        })
        raise RuntimeError("No valid actual H3 candidate; cannot train the requested W3+H3 branch")
    return harnesses[candidate]


def evaluate(launch: Launcher, root: Path, label: str, model: str, h3: str, protocol: dict) -> dict:
    require_idle()
    cohort = read_json(Path(protocol["core200_protocol"]))
    validate_cohort(cohort)
    config = {
        "solver_model": model, "solver_version": f"{root.name}-{label}",
        "no_patch": protocol["no_patch"], "harnesses": {"h2": protocol["historical_h2"], "h3": h3},
        "task_ids": cohort["all_task_ids"], "samples": 1,
        **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
        "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 3,
    }
    write_json(root / "STATUS.json", {"phase": f"evaluating_{label}_original200", "at_unix": time.time()})
    matrix, output = run_matrix(launch, config, root / "evaluation" / label, 1)
    scores = {}
    for name, metadata in matrix["variants"].items():
        path = Path(metadata["report"])
        report = checked_native_report(read_json(path), cohort["all_task_ids"], config)
        _immutable_json(root / "reports" / f"{label}_{name}.json", report)
        scores[name] = {
            "successes": int(sum(row["reward"] for row in report["results"])),
            "questions": 200, "accuracy": report["accuracy"],
            "strict197": subset_report(report, cohort["strict_goal_disjoint_task_ids"])["accuracy"],
            "report": str(path), "report_sha256": file_sha(path),
        }
    _immutable_json(root / f"{label}_EVALUATED.json", {
        "status": "completed", "scores": scores, "matrix": str(output),
        "unique_harnesses": matrix["unique_harnesses"],
    })
    return scores


def audit_result(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    state = read_json(root / "input_state.json")
    h3 = read_json(root / "H3_SELECTION.json")["actual_h3_candidate"]
    if h3 is None:
        raise ValueError("An actual H3 policy is required for this final result")
    for phase, role in (("h3", "editor"), ("w3", "solver")):
        config = read_json(root / phase / "config.json")
        backend_wiring_audit(config, role=role)
        trained, _ = training_evidence(Path(config["run"]), config)
        if phase == "h3" and config["solver_model"] != state["solver"]:
            raise ValueError("H3 did not train against the frozen early-W2 checkpoint")
        if phase == "w3" and (config["solver_model"] != state["solver"] or config["harness"] != h3):
            raise ValueError("W3 did not start from early-W2 with the actual H3 candidate")
        if trained["role"] != role:
            raise ValueError("A native phase trained the wrong role")
    rows = read_json(root / "w3/data.json")
    if rows != solver_rows_for_cycle(read_json(RECOVERY / "data.json")):
        raise ValueError("W3 changed the requested training instances or order")
    scores, reports, pins = {}, {}, {}
    cohort = read_json(Path(protocol["core200_protocol"]))
    for label in ("w2_early", "w3"):
        evaluated = read_json(root / f"{label}_EVALUATED.json")
        checked_release(Path(evaluated["matrix"]))
        scores[label] = evaluated["scores"]
        for harness, metadata in evaluated["scores"].items():
            path = Path(metadata["report"])
            if file_sha(path) != metadata["report_sha256"]:
                raise ValueError("A primary200 raw report changed")
            report = checked_native_report(read_json(path), cohort["all_task_ids"], {
                **cohort, "samples": 1,
            })
            if report["accuracy"] != metadata["accuracy"]:
                raise ValueError("A primary200 success total differs from its raw episodes")
            reports[f"{label}_{harness}"] = report
            pins[str(path)] = file_sha(path)
    comparisons = {
        "h3_minus_h2_at_fixed_w2": paired_difference(reports["w2_early_h2"], reports["w2_early_h3"], seed=9000082),
        "w3_minus_w2_at_fixed_h3": paired_difference(reports["w2_early_h3"], reports["w3_h3"], seed=9000083),
        "final_minus_start": paired_difference(reports["w2_early_h2"], reports["w3_h3"], seed=9000084),
        "h3_minus_h2_at_fixed_w3": paired_difference(reports["w3_h2"], reports["w3_h3"], seed=9000085),
    }
    selection_ids = set(read_json(Path(protocol["evaluation"]["task_manifest"]))["selection_task_ids"])
    outside = [task for task in cohort["all_task_ids"] if task not in selection_ids]
    comparisons["final_minus_start_outside_selection"] = paired_difference(
        subset_report(reports["w2_early_h2"], outside),
        subset_report(reports["w3_h3"], outside), seed=9000086)
    for name, expected in read_json(root / "EXECUTION_PINS.json")["pins"].items():
        if file_sha(Path(name)) != expected:
            raise ValueError(f"A frozen original continuation input changed: {name}")
    return {
        "status": "verified", "scores": scores, "comparisons": comparisons,
        "h3_selection": read_json(root / "H3_SELECTION.json"),
        "native_training": {phase: read_json(root / phase / "TRAINING_AUDIT.json") for phase in ("h3", "w3")},
        "consumed_role_trajectories": protocol["consumed_role_trajectories"],
        "partner_refresh_environment_episodes": 240,
        "main200_environment_episodes": sum(
            read_json(root / f"{label}_EVALUATED.json")["unique_harnesses"] * 200
            for label in ("w2_early", "w3")),
        "fresh100_used": False, "release": require_idle(), "pins": pins,
        "limitations": protocol["limitations"] + [protocol["optimizer_state"], protocol["backend_change_from_recovery"]],
        "at_unix": time.time(),
    }


def run(root: Path) -> dict:
    require_research_node()
    os.umask(0o002)
    with (root / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "STARTED.json").exists():
            raise RuntimeError("Do not repeat a started ordinary-SAO cycle")
        validation = read_json(root / "EXECUTION_VALIDATION.json")
        if validation["status"] != "passed":
            raise ValueError("CPU execution validation is required before any GPU allocation")
        for name, digest in read_json(root / "EXECUTION_PINS.json")["pins"].items():
            if file_sha(Path(name)) != digest:
                raise ValueError(f"Frozen continuation input changed: {name}")
        protocol = read_json(root / "protocol.json")
        _immutable_json(root / "STARTED.json", {
            "pid": os.getpid(), "at_unix": time.time(), "release": require_idle(),
            "protocol_sha256": file_sha(root / "protocol.json"), "fresh100_used": False,
        })
        write_json(root / "STATUS.json", {"phase": "refreshing_w2_train240_evidence", "at_unix": time.time()})
        refreshed = refresh_partner(root / "protocol.json", root / "input_state.json", root / "partner_w2")
        if not refreshed["engineer_training_ready"]:
            raise RuntimeError("No training failure packets; do not invent Engineer training evidence")
        checked_release(root / "partner_w2")
        state = read_json(root / "partner_w2/REFRESHED_STATE.json")
        packets = read_json(Path(state["packets"]))
        for packet in packets:
            validate_metadata(packet["metadata"], state["solver_version"])
            if not 0 <= packet["metadata"]["start"] < packet["metadata"]["end"] <= 240:
                raise ValueError("H3 evidence left the original training population")
        rows = editor_dataset(packets, 192, state["solver_epoch_version"],
                              sample_offset=protocol["engineer_sample_offset"])
        _immutable_json(root / "h3/data.json", rows)
        h_config = freeze_config(root, read_json(root / "h3_template.json"), "editor")
        launch = Launcher(root / "supervision")
        try:
            h_trained = train_phase(launch, root, "h3", h_config)
            write_json(root / "STATUS.json", {"phase": "selecting_actual_h3", "at_unix": time.time()})
            h3 = select_h3(launch, root, protocol, state, h_trained)
            evaluate(launch, root, "w2_early", state["solver"], h3, protocol)
            w_config = {**read_json(root / "w3_template.json"), "harness": h3}
            _immutable_json(root / "w3/assignments.json", no_patch_assignments(
                Path(h3), training_rows=read_json(root / "w3/data.json")))
            w_config = freeze_config(root, w_config, "solver")
            w_trained = train_phase(launch, root, "w3", w_config)
            w_pair = checkpoint_pair(w_trained, 79)
            _immutable_json(root / "FINAL_PAIR.json", {
                "solver": w_pair, "engineer": checkpoint_pair(h_trained, 31), "harness": h3,
                "harness_sha256": file_sha(Path(h3)), "not_an_incumbent_fallback": True,
            })
            evaluate(launch, root, "w3", w_pair["actor"], h3, protocol)
        finally:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
        result = audit_result(root)
        _immutable_json(root / "RESULT.json", result)
        _immutable_json(root / "FINAL_AUDIT.json", {
            "status": "passed", "result_sha256": file_sha(root / "RESULT.json"), "at_unix": time.time(),
        })
        write_json(root / "STATUS.json", {"phase": "verified", "at_unix": time.time()})
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = {"prepare": prepare, "run": run, "audit": audit_result}[args.action](args.root.resolve())
        print({"status": result.get("status", "prepared"), "root": str(args.root)}, flush=True)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, ExceptionGroup) as exc:
        write_json(args.root / f"{args.action.upper()}_FAILED.json", {
            "error": repr(exc), "at_unix": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
