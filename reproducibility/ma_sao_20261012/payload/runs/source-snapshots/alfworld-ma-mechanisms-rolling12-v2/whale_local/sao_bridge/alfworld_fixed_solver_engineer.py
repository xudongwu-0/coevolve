"""One more ordinary SAO Engineer phase from historical H2 with W1 frozen."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_h_train import training_spec
from .alfworld_phase_data import editor_dataset
from .alfworld_reward_service import validate_metadata
from .alfworld_sao_control import backend_wiring_audit, choose_valid_proposal, role_config
from .alfworld_strong_cost import native_generation, proposal_work, reward_work
from .alfworld_strong_phase import checked_native_report, checked_release, checkpoint_pair, training_evidence
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .budget import contract_digest, model_digests
from .common import ROOT, file_sha, read_json, write_json
from .large_train import training_capacity_audit, validate_training_rows
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node

HISTORY = ROOT / "runs/alfworld_alternating_e15_w79_8h_v1"
EVAL_PROTOCOL = ROOT / "runs/alfworld_core_200_oneshot_v2/protocol.json"


def continuation_config(old: dict, state: dict, root: Path, acceleration: dict) -> dict:
    if (state["solver"] != old["solver_model"] or state["solver_version"] != old["solver_version"]
            or state["engineer"] == old["engineer_model"]):
        raise ValueError("Continuation must use selected H2 and the unchanged historical W1")
    config = role_config("h2", root, state)
    for key in (
        "steps", "batch_size", "critic_warmup_steps", "critic_updates_per_step",
        "learning_rate", "critic_learning_rate", "temperature", "actor_gpus", "rollout_gpus",
    ):
        if config[key] != old[key]:
            raise ValueError(f"Historical Engineer phase contract changed: {key}")
    config.update(
        run=str(root / "training"), phase="h3_fixed_w1", experiment=root.name,
        data=str(root / "data.json"), harness=old["harness"],
        tmpdir=str(root / "runtime_cache"), save_interval_steps=32,
        seed=old["seed"], phase_timeout_seconds=4 * 3600,
        rollout_setup_timeout=1800, **acceleration,
    )
    return config


def prepare(root: Path, acceleration: dict) -> dict:
    old_path = HISTORY / "sequential/h2/phase_config.json"
    state_path = HISTORY / "sequential/STATE_AFTER_h2.json"
    old, state = read_json(old_path), read_json(state_path)
    trained_path = HISTORY / "sequential/h2/attempt-01/optimizer/trained.json"
    pair = checkpoint_pair(read_json(trained_path), 31)
    if (state["engineer"], state["engineer_critic"]) != (pair["actor"], pair["critic"]):
        raise ValueError("Historical selected Engineer is not H2-step31")
    packets_path = Path(state["packets"])
    if (file_sha(packets_path) != state["packets_sha256"]
            or Path(state["packet_solver"]).resolve() != Path(state["solver"]).resolve()):
        raise ValueError("Fixed-Solver training packets changed counterpart")
    packets = read_json(packets_path)
    for packet in packets:
        validate_metadata(packet["metadata"], state["solver_version"])
        if packet["metadata"]["end"] > 240:
            raise ValueError("Engineer continuation must use only the historical training cohort")
    base = read_json(HISTORY / "frozen_protocol.json")
    count = old["steps"] * old["batch_size"]
    offset = base["h2"]["sample_offset"] + count
    rows = editor_dataset(packets, count, state["solver_epoch_version"], sample_offset=offset)
    old_ids = {row["id"] for row in read_json(Path(old["data"]))}
    if len({row["id"] for row in rows}) != count or any(row["id"] in old_ids for row in rows):
        raise ValueError("Continuation reused old trajectory identities")
    _immutable_json(root / "data.json", rows)
    _immutable_json(root / "input_state.json", state)
    config = continuation_config(old, state, root, acceleration)
    validation = Path(config["training_kernel_validation"])
    config["pinned_files"] = {
        str(path): file_sha(path) for path in (validation, root / "data.json", root / "input_state.json")
    }
    spec = training_spec(config, Path(config["run"]), "http://127.0.0.1:28600")
    validate_training_rows(spec, rows)
    _immutable_json(root / "capacity_audit.json", training_capacity_audit(spec, rows))
    _immutable_json(root / "backend_audit.json", backend_wiring_audit(config))
    _immutable_json(root / "training_config.json", config)
    solver_weights = model_digests(Path(state["solver"]))[1]
    if solver_weights != state["solver_weights_sha256"]:
        raise ValueError("Fixed W1 weights differ from the historical selected state")
    cohort = read_json(EVAL_PROTOCOL)
    validate_cohort(cohort)
    evaluation = {**base["evaluation"], "eval_concurrency": 128, "environment_concurrency": 32}
    protocol = {
        "protocol": "fixed-w1-h2-to-h3-single-phase-v1", "root": str(root),
        "question": "With the stronger W1 frozen, does one additional Engineer phase improve its proposal distribution or selected system?",
        "initial_pair": pair, "fixed_solver": state["solver"], "solver_version": state["solver_version"],
        "solver_weights_sha256": solver_weights, "historical_h2": state["harness"],
        "native_rounds": 32, "actor_updates": 27, "critic_updates": 32,
        "consumed_engineer_trajectories": count, "new_solver_updates": 0,
        "sample_offset": offset, "training_seed": config["seed"], "independent_seed_sweep": False,
        "optimizer_state": "H2 exports Actor/Critic HF weights but no Adam; one explicit fresh-Adam boundary and the original five critic warmup rounds.",
        "backend": "Use the existing corrected ordinary-SAO backend: independent trajectories and processed sampling logprobs; no MA/CTDE, ValueNorm, KL, or interaction-credit changes.",
        "backend_limitation": "This extends historical H2 using the repaired backend, not a bitwise reproduction of its old packing/probability defects.",
        "h2": {**base["h2"], "checkpoint_steps": [31]},
        "evaluation": evaluation, "no_patch": old["harness"],
        "primary_endpoint": "Paired all200 comparison of before/after selected systems; both get eight matched fresh proposals and the same saved h2 incumbent.",
        "generator_endpoint": "Mean over all eight proposal slots on original selection24 mean@2, with question-cluster uncertainty; invalid slots remain explicit no-patch evaluations.",
        "selection_rule": "Best valid fresh proposal must beat the shared saved h2 by the original0.02 selection margin; otherwise retain the incumbent. Also report the raw final-H3 candidate even if rejected.",
        "checkpoint_rule": "Only the final extra32-round checkpoint is evaluated; no checkpoint search.",
        "evaluation_cost_upper_bound": 18 * 24 * 2 + 5 * 200,
        "limitations": [
            "One training continuation, not training-seed robustness or proof of a universal Engineer ceiling",
            "Repeated development tasks, not a new sealed test; selection/report overlap is explicitly reported",
            "Only the saved selected strategy and eight matched proposal samples are measured, not the entire generator distribution",
        ],
        "pinned_files": {str(path): file_sha(path) for path in (
            old_path, state_path, trained_path, packets_path, HISTORY / "frozen_protocol.json",
            root / "data.json", root / "input_state.json", root / "training_config.json",
            root / "capacity_audit.json", root / "backend_audit.json",
            Path(state["harness"]), Path(old["harness"]), Path(base["h2"]["template"]),
            Path(evaluation["task_manifest"]), EVAL_PROTOCOL, validation,
        )},
    }
    _immutable_json(root / "protocol.json", protocol)
    return protocol


def proposal_population_report(values: list[dict], reports: dict, label: str) -> dict:
    if [value["sample_index"] for value in values] != list(range(8)):
        raise ValueError("Proposal population lost its eight matched slots")
    first = reports[f"{label}-proposal00"]
    samples = first["samples_per_question"]
    rows = []
    for slot in range(8):
        report = reports[f"{label}-proposal{slot:02d}"]
        if (report["question_count"], report["samples_per_question"], report["metric"]) != (
            first["question_count"], samples, first["metric"],
        ):
            raise ValueError("Proposal scoring protocols differ")
        for row in report["results"]:
            rows.append({
                **{key: row[key] for key in ("id", "task_id", "rollout_seed", "reward")},
                "proposal_slot": slot, "solver_sample_index": row["sample_index"],
                "sample_index": slot * samples + row["sample_index"],
            })
    return {
        "metric": f"mean_over8_proposals_of_{first['metric']}",
        "question_count": first["question_count"], "samples_per_question": 8 * samples,
        "accuracy": sum(row["reward"] for row in rows) / len(rows), "results": rows,
        "interpretation": "Eight different policies, not16 independent samples from one policy; preserve native Solver seeds and cluster inference by question.",
    }


def select_side(values: list[dict], reports: dict, label: str, margin: float) -> dict:
    population = proposal_population_report(values, reports, label)
    valid = sum(value["valid"] for value in values)
    best = choose_valid_proposal(values, reports, label) if valid else None
    incumbent = reports["incumbent_harness"]["accuracy"]
    promoted = best is not None and reports[best]["accuracy"] - incumbent >= margin - 1e-12
    return {
        "status": "valid_proposals_available" if valid else "all_invalid_model_outputs",
        "valid_proposals": valid, "proposal_count": 8, "mean_slot_accuracy": population["accuracy"],
        "best_valid_proposal": best, "selected": best if promoted else "incumbent_harness",
        "incumbent_retained": not promoted, "population": population,
    }


def outcome(comparison: dict) -> str:
    delta = comparison["candidate_minus_control"]
    low, high = comparison["paired_question_bootstrap_95pct"]
    if delta > 0:
        return "exploratory_improvement" if low > 0 else "positive_point_estimate_not_established"
    if delta < 0 and high < 0:
        return "exploratory_degradation"
    return "no_established_improvement"


def run(root: Path) -> dict:
    require_research_node()
    protocol = read_json(root / "protocol.json")
    if (root / "STARTED.json").exists() or (root / "COMPLETE.json").exists():
        raise RuntimeError("The fixed-Solver phase cannot be silently rerun or partially resumed")
    launch = None
    try:
        for name, digest in read_json(root / "EXECUTION_PINS.json")["pins"].items():
            if file_sha(Path(name)) != digest:
                raise ValueError(f"Frozen Engineer continuation input changed: {name}")
        release = gpu_release_audit()
        if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
            raise RuntimeError("Fixed-Solver phase requires released local research GPUs")
        if model_digests(Path(protocol["fixed_solver"]))[1] != protocol["solver_weights_sha256"]:
            raise ValueError("Frozen Solver changed before the Engineer phase")
        _immutable_json(root / "STARTED.json", {"at_unix": time.time(), "release": release})
        config = read_json(root / "training_config.json")
        backend_wiring_audit(config)
        launch = Launcher(root / "supervision")
        write_json(root / "MONITOR.json", {"phase": "engineer_training", "at_unix": time.time()})
        process = launch.module(
            "sao_bridge.alfworld_h_train", ["--config", root / "training_config.json"],
            gpus="0,1,2,3,4,5,6,7", name="h3-native",
        )
        launch.wait(process, "fixed-W1 H3 native training", timeout=config["phase_timeout_seconds"] + 600)
        trained, native_audit = training_evidence(root / "training", config)
        pair = checkpoint_pair(trained, 31)
        generation_cost = native_generation(root / "training/optimizer")
        reward_cost = reward_work(root / "training")
        _immutable_json(root / "TRAINING_AUDIT.json", {
            "status": "passed", "native": native_audit,
            "generation_cost": generation_cost, "reward_cost": reward_cost,
        })
        state = read_json(root / "input_state.json")
        selection = root / "selection"
        write_json(root / "MONITOR.json", {"phase": "matched_proposal_generation", "at_unix": time.time()})
        proposals = generate_proposals(
            launch, protocol, state, {"before": protocol["initial_pair"], "after": pair}, selection,
        )
        harnesses = {"incumbent_harness": protocol["historical_h2"]}
        for label, values in proposals.items():
            for value in values:
                name = f"{label}-proposal{value['sample_index']:02d}"
                path = selection / "patches" / f"{name}.json"
                _immutable_json(path, value["patch"] if value["valid"] else read_json(Path(protocol["no_patch"])))
                harnesses[name] = str(path)
        config_selection = matrix_config(
            protocol, state["solver"], state["solver_version"], harnesses, selection / "matrix",
        )
        write_json(root / "MONITOR.json", {"phase": "matched_proposal_selection", "at_unix": time.time()})
        matrix, selection_output = run_matrix(launch, config_selection, selection / "matrix", 1)
        reports = matrix_reports(matrix, protocol)
        sides = {
            label: select_side(values, reports, label, protocol["evaluation"]["promotion_min_delta"])
            for label, values in proposals.items()
        }
        population_comparison = paired_difference(
            sides["before"]["population"], sides["after"]["population"], seed=9000070,
        )
        for label, side in sides.items():
            _immutable_json(root / f"{label}_proposal_population.json", side.pop("population"))
        failures = {label: side["status"] for label, side in sides.items() if not side["valid_proposals"]}
        if failures:
            _immutable_json(root / "MODEL_FAILURE.json", {
                "kind": "all_eight_proposals_invalid", "sides": failures,
                "incumbent_is_retained_deployment_not_a_learned_replacement": True,
            })
        selected = {
            label: harnesses[side["selected"]] for label, side in sides.items()
        }
        main_harnesses = {**selected, "historical_h2": protocol["historical_h2"]}
        raw_after = sides["after"]["best_valid_proposal"]
        if raw_after is not None:
            main_harnesses["after_candidate"] = harnesses[raw_after]
        _immutable_json(root / "SELECTION.json", {
            "sides": sides, "population_comparison": population_comparison,
            "selected_harnesses": main_harnesses, "main_report_opened": False,
            "selection_matrix": str(selection_output),
        })
        cohort = read_json(EVAL_PROTOCOL)
        main_config = {
            "solver_model": protocol["fixed_solver"], "solver_version": protocol["solver_version"],
            "no_patch": protocol["no_patch"], "harnesses": main_harnesses,
            "task_ids": cohort["all_task_ids"], "samples": 1,
            **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
            "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 4,
        }
        write_json(root / "MONITOR.json", {"phase": "fixed_w1_main200", "at_unix": time.time()})
        main_matrix, main_output = run_matrix(launch, main_config, root / "evaluation200", 1)
        scored, scores, pins = {}, {}, {}
        for label, metadata in main_matrix["variants"].items():
            native_path = Path(metadata["report"])
            native = checked_native_report(read_json(native_path), cohort["all_task_ids"], main_config)
            compact = {**native, "results": [
                {key: row[key] for key in ("id", "task_id", "sample_index", "rollout_seed", "reward", "tokens")}
                for row in native["results"]
            ]}
            path = root / "reports" / f"{label}.json"
            _immutable_json(path, compact)
            pins[str(path)] = file_sha(path)
            pins[str(native_path)] = file_sha(native_path)
            scored[label] = compact
            scores[label] = {
                "all200": native["accuracy"],
                "strict197": subset_report(native, cohort["strict_goal_disjoint_task_ids"])["accuracy"],
            }
        comparison = paired_difference(scored["before"], scored["after"], seed=9000071)
        selection_ids = set(config_selection["task_ids"])
        outside = [task for task in cohort["all_task_ids"] if task not in selection_ids]
        outside_comparison = paired_difference(
            subset_report(scored["before"], outside), subset_report(scored["after"], outside), seed=9000072,
        )
        if model_digests(Path(protocol["fixed_solver"]))[1] != protocol["solver_weights_sha256"]:
            raise ValueError("Fixed W1 weights changed during the experiment")
        result = {
            "status": "completed", "scientific_outcome": outcome(comparison),
            "scores": scores, "selected_system_comparison": comparison,
            "proposal_population_comparison": population_comparison,
            "outside_selection_comparison": outside_comparison,
            "raw_after_candidate_comparison": paired_difference(
                scored["before"], scored["after_candidate"], seed=9000073,
            ) if raw_after is not None else None,
            "selection": sides, "model_failures": failures,
            "solver_unchanged_weights_sha256": protocol["solver_weights_sha256"],
            "training": native_audit, "new_consumed_engineer_trajectories": 192,
            "training_generation_cost": generation_cost, "training_reward_cost": reward_cost,
            "proposal_generation_cost": proposal_work(selection),
            "new_solver_updates": 0,
            "evaluation_environment_episodes": (
                matrix["unique_harnesses"] * matrix["question_count"] * protocol["evaluation"]["samples"]
                + main_matrix["unique_harnesses"] * main_matrix["question_count"]
            ),
            "main_matrix": str(main_output), "evidence_pins": pins,
            "selection_report_overlap": sorted(selection_ids & set(cohort["all_task_ids"])),
            "limitations": protocol["limitations"] + [protocol["optimizer_state"], protocol["backend_limitation"]],
            "at_unix": time.time(),
        }
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    finally:
        if launch is not None:
            launch.close()
            write_json(root / "RELEASED.json", gpu_release_audit())
    checked_release(root)
    _immutable_json(root / "COMPLETE.json", result)
    write_json(root / "MONITOR.json", {"phase": "completed", "at_unix": time.time()})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.root.resolve())["scientific_outcome"])


if __name__ == "__main__":
    main()
