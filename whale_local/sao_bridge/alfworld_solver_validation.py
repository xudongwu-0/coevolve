"""Persistent two-split validation of sequential SAO on the 9B ALFWorld Solver."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import socket
import time

from .analysis import paired_difference, paired_lower_bound
from .common import (
    ensure_model_runtime_metadata,
    file_sha,
    read_json,
    write_json,
)
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


REQUIRED_KEYS = {
    "run", "initial_solver", "no_patch", "training_data", "seed",
    "actor_steps", "critic_warmup_steps", "batch_size", "learning_rate",
    "critic_learning_rate", "actor_gpus", "rollout_gpus",
    "save_interval_steps",
}


def validate_spec(spec: dict) -> dict:
    missing = REQUIRED_KEYS - spec.keys()
    if missing:
        raise ValueError(f"Missing 9B Solver validation keys: {sorted(missing)}")
    normalized = dict(spec)
    normalized.setdefault("selection_task_ids", list(range(240, 300)))
    normalized.setdefault("report_task_ids", list(range(300, 400)))
    normalized.setdefault("samples", 1)
    normalized.setdefault("solver_tokens", 4096)
    normalized.setdefault("temperature", 1.0)
    normalized.setdefault("staleness", 1)
    normalized.setdefault("training_environment_concurrency", 1)
    normalized.setdefault("eval_environment_concurrency", 32)
    normalized.setdefault("eval_concurrency", 32)
    normalized.setdefault("max_training_attempts", 2)
    normalized.setdefault("critic_updates_per_step", 1)
    normalized.setdefault("critic_mode", "learned")
    normalized.setdefault("solver_trajectory_style", "individual")
    normalized.setdefault("actor_gradient_checkpointing", True)
    normalized.setdefault("critic_gradient_checkpointing", True)
    normalized.setdefault("selection_min_delta", 0.02)
    integer_keys = (
        "seed", "actor_steps", "critic_warmup_steps", "batch_size",
        "actor_gpus", "rollout_gpus", "save_interval_steps", "samples",
        "solver_tokens", "staleness", "training_environment_concurrency",
        "eval_environment_concurrency", "eval_concurrency",
        "max_training_attempts", "critic_updates_per_step",
    )
    for key in integer_keys:
        if type(normalized[key]) is not int or normalized[key] < 1:
            if key == "critic_warmup_steps" and normalized[key] == 0:
                continue
            raise ValueError(f"{key} must be a positive integer")
    for key in ("actor_gradient_checkpointing", "critic_gradient_checkpointing"):
        if type(normalized[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if normalized["critic_mode"] not in {"learned", "none"}:
        raise ValueError("critic_mode must be learned or none")
    if normalized["solver_trajectory_style"] not in {"individual", "concat"}:
        raise ValueError("solver_trajectory_style must be individual or concat")
    if normalized["critic_mode"] == "none" and normalized["critic_warmup_steps"]:
        raise ValueError("Critic-free training cannot use critic warm-up")
    if not 0 <= float(normalized["selection_min_delta"]) <= 1:
        raise ValueError("selection_min_delta must be in [0, 1]")
    total_steps = normalized["actor_steps"] + normalized["critic_warmup_steps"]
    if total_steps % normalized["save_interval_steps"]:
        raise ValueError("save_interval_steps must divide total optimizer steps")
    if normalized["actor_gpus"] + normalized["rollout_gpus"] != 8:
        raise ValueError("9B Solver validation requires all eight GPUs")
    if normalized["batch_size"] % normalized["actor_gpus"]:
        raise ValueError("batch_size must be divisible by actor_gpus")
    if not 0 < float(normalized["learning_rate"]) < 1:
        raise ValueError("learning_rate must be in (0, 1)")
    if not 0 < float(normalized["critic_learning_rate"]) < 1:
        raise ValueError("critic_learning_rate must be in (0, 1)")
    model = Path(normalized["initial_solver"]).resolve()
    no_patch = Path(normalized["no_patch"]).resolve()
    data = Path(normalized["training_data"]).resolve()
    if not model.is_dir() or not list(model.glob("*.safetensors")):
        raise FileNotFoundError("Initial 9B Solver has no safetensors")
    if read_json(no_patch) != {"benchmark": "alfworld", "actions": []}:
        raise ValueError("Solver validation requires the exact empty no-patch harness")
    rows = read_json(data)
    required_rows = total_steps * normalized["batch_size"]
    if len(rows) < required_rows:
        raise ValueError("Solver training data is smaller than the optimizer budget")
    consumed = rows[:required_rows]
    if any(
        type(row.get("task_id")) is not int
        or not 0 <= row["task_id"] < 240
        or row.get("warmup")
        for row in consumed
    ):
        raise ValueError("Solver training rows must be fresh train-split tasks")
    selection = normalized["selection_task_ids"]
    report = normalized["report_task_ids"]
    split_bounds = {
        "selection": (240, 300),
        "report": (300, 400),
    }
    for name, task_ids in (("selection", selection), ("report", report)):
        lower, upper = split_bounds[name]
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or len(task_ids) != len(set(task_ids))
            or any(
                type(task_id) is not int or not lower <= task_id < upper
                for task_id in task_ids
            )
        ):
            raise ValueError(f"{name} task IDs are invalid")
    train_ids = {row["task_id"] for row in consumed}
    if len(train_ids) != len(consumed):
        raise ValueError("Each consumed Solver trajectory must use a distinct train task")
    if train_ids & set(selection) or train_ids & set(report) or set(selection) & set(report):
        raise ValueError("ALFWorld train, selection and report task IDs must be disjoint")
    normalized.update(
        run=str(Path(normalized["run"]).resolve()),
        initial_solver=str(model),
        no_patch=str(no_patch),
        training_data=str(data),
        total_steps=total_steps,
    )
    return normalized


def no_patch_assignments(no_patch: Path) -> list[dict]:
    return [
        {
            "name": f"alfworld-h0-b{start:03d}",
            "task_ids": list(range(start, start + 10)),
            "harness": str(no_patch.resolve()),
        }
        for start in range(0, 240, 10)
    ]


def subset_report(report: dict, task_ids: list[int]) -> dict:
    wanted = set(task_ids)
    rows = [row for row in report["results"] if row["task_id"] in wanted]
    samples = report["samples_per_question"]
    if len(rows) != len(wanted) * samples:
        raise ValueError("ALFWorld evaluation report is missing requested tasks")
    if {row["task_id"] for row in rows} != wanted:
        raise ValueError("ALFWorld evaluation report changed task identity")
    result = dict(report)
    result.update(
        accuracy=sum(float(row["reward"]) for row in rows) / len(rows),
        question_count=len(wanted),
        tokens=sum(int(row["tokens"]) for row in rows),
        tool_calls=sum(int(row["tool_calls"]) for row in rows),
        tool_errors=sum(int(row["tool_errors"]) for row in rows),
        truncated_fraction=sum(bool(row["truncated"]) for row in rows) / len(rows),
        boxed_fraction=sum(bool(row["boxed"]) for row in rows) / len(rows),
        results=rows,
    )
    return result


def select_checkpoint(
    control: dict,
    candidates: list[dict],
    seed: int,
    *,
    minimum_delta: float = 0.02,
) -> dict:
    if not candidates:
        raise ValueError("No 9B Solver checkpoints were evaluated")
    alpha = 0.05 / len(candidates)
    scored = []
    for index, candidate in enumerate(candidates):
        evaluation = candidate["evaluation"]
        scored.append({
            "global_step": candidate["global_step"],
            "path": candidate["path"],
            "accuracy": evaluation["accuracy"],
            "candidate_minus_control": evaluation["accuracy"] - control["accuracy"],
            "selection_lcb": paired_lower_bound(
                control,
                evaluation,
                alpha=alpha,
                seed=seed + 130_363 * (index + 1),
            ),
        })
    qualified = [
        row for row in scored
        if row["candidate_minus_control"] >= minimum_delta
    ]
    if qualified:
        selected = max(qualified, key=lambda row: (row["accuracy"], row["global_step"]))
        reason = "largest held-out accuracy among checkpoints clearing the delta floor"
    else:
        selected = {
            "global_step": None,
            "path": None,
            "accuracy": control["accuracy"],
            "candidate_minus_control": 0.0,
            "selection_lcb": 0.0,
        }
        reason = "no checkpoint cleared the held-out selection gate; retained W0"
    return {
        "control_accuracy": control["accuracy"],
        "alpha_per_checkpoint": alpha,
        "minimum_delta": minimum_delta,
        "candidates": scored,
        "selected": selected,
        "selected_trained_checkpoint": selected["path"] is not None,
        "reason": reason,
    }


def _status(root: Path, stage: str, started: float, **extra) -> None:
    write_json(root / "STATUS.json", {
        "stage": stage,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
        **extra,
    })


def _fresh_retry_root(path: Path, archive: Path) -> None:
    if not path.exists():
        return
    if (path / "COMPLETE.json").is_file():
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    if archive.exists():
        raise FileExistsError(archive)
    shutil.move(str(path), archive)


def _run_evaluation(
    root: Path,
    launch: Launcher,
    spec: dict,
    *,
    name: str,
    model: str,
    version: str,
    task_ids: list[int],
) -> dict:
    run = root / "evaluations" / name
    complete_path = run / "COMPLETE.json"
    if complete_path.is_file():
        return read_json(run / "report.json")
    _fresh_retry_root(
        run,
        root / "failed_evaluations" / f"{name}-{time.time_ns()}",
    )
    config = {
        "run": str(run),
        "solver_model": model,
        "solver_version": version,
        "task_ids": task_ids,
        "no_patch": spec["no_patch"],
        "seed": spec["seed"] + 5_000_003,
        "samples": spec["samples"],
        "solver_tokens": spec["solver_tokens"],
        "temperature": spec["temperature"],
        "eval_concurrency": spec["eval_concurrency"],
        "environment_concurrency": spec["eval_environment_concurrency"],
        "controller_port": spec.get("eval_controller_port", 28000),
        "worker_port": spec.get("eval_worker_port", 28100),
        "solver_port": spec.get("eval_solver_port", 19801),
    }
    config_path = root / "evaluation_configs" / f"{name}.json"
    write_json(config_path, config)
    process = launch.module(
        "sao_bridge.alfworld_solver_eval",
        ["--config", config_path],
        gpus="0,1,2,3,4,5,6,7",
        name=name,
    )
    launch.wait(
        process,
        name,
        timeout=float(spec.get("evaluation_timeout_seconds", 14400)),
    )
    return read_json(run / "report.json")


def _run_training(root: Path, launch: Launcher, spec: dict) -> dict:
    run = root / "training"
    trained_path = run / "optimizer/trained.json"
    if trained_path.is_file() and (run / "COMPLETE.json").is_file():
        return read_json(trained_path)
    attempts_path = root / "training_attempts.json"
    attempts = read_json(attempts_path) if attempts_path.is_file() else []
    _fresh_retry_root(
        run,
        root / "failed_training_attempts" / f"attempt-{len(attempts):02d}",
    )
    if len(attempts) >= spec["max_training_attempts"]:
        write_json(root / "FAILED.json", {
            "stage": "training",
            "error": "training attempts exhausted",
            "attempts": attempts,
        })
        raise SystemExit(78)
    attempts.append({
        "attempt": len(attempts) + 1,
        "started_at_unix": time.time(),
        "host": socket.gethostname(),
    })
    write_json(attempts_path, attempts)
    assignments = root / "no_patch_train_assignments.json"
    write_json(assignments, no_patch_assignments(Path(spec["no_patch"])))
    config = {
        "run": str(run),
        "phase": "w1",
        "experiment": root.name,
        "solver_model": spec["initial_solver"],
        "solver_critic": spec["initial_solver"],
        "harness": spec["no_patch"],
        "harness_version": 0,
        "data": spec["training_data"],
        "assignments": str(assignments),
        "steps": spec["total_steps"],
        "batch_size": spec["batch_size"],
        "seed": spec["seed"],
        "learning_rate": spec["learning_rate"],
        "critic_learning_rate": spec["critic_learning_rate"],
        "solver_tokens": spec["solver_tokens"],
        "temperature": spec["temperature"],
        "staleness": spec["staleness"],
        "actor_gpus": spec["actor_gpus"],
        "rollout_gpus": spec["rollout_gpus"],
        "critic_warmup_steps": spec["critic_warmup_steps"],
        "save_interval_steps": spec["save_interval_steps"],
        "critic_updates_per_step": spec["critic_updates_per_step"],
        "critic_mode": spec["critic_mode"],
        "solver_trajectory_style": spec["solver_trajectory_style"],
        "actor_gradient_checkpointing": spec["actor_gradient_checkpointing"],
        "critic_gradient_checkpointing": spec["critic_gradient_checkpointing"],
        "environment_concurrency": spec["training_environment_concurrency"],
        "monitor_interval_seconds": 5,
        "phase_timeout_seconds": spec.get("training_timeout_seconds", 43200),
    }
    config_path = root / "training_config.json"
    write_json(config_path, config)
    process = launch.module(
        "sao_bridge.alfworld_w_train",
        ["--config", config_path],
        gpus="0,1,2,3,4,5,6,7",
        name=f"training_attempt_{len(attempts)}",
    )
    try:
        launch.wait(
            process,
            "9B ALFWorld Solver training",
            timeout=float(spec.get("training_timeout_seconds", 43200)) + 600,
        )
    except BaseException as exc:
        attempts[-1].update(finished_at_unix=time.time(), error=repr(exc))
        write_json(attempts_path, attempts)
        raise
    attempts[-1].update(finished_at_unix=time.time(), complete=True)
    write_json(attempts_path, attempts)
    return read_json(trained_path)


def run(spec_path: Path) -> dict:
    spec = validate_spec(read_json(spec_path))
    root = Path(spec["run"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    spec_path_frozen = root / "spec.json"
    if spec_path_frozen.is_file() and read_json(spec_path_frozen) != spec:
        raise ValueError("Cannot change a persisted 9B Solver validation spec")
    write_json(spec_path_frozen, spec)
    sources = {
        name: file_sha(Path(__file__).with_name(name))
        for name in (
            "alfworld_solver_validation.py",
            "alfworld_solver_eval.py",
            "alfworld_w_train.py",
            "large_train.py",
            "large_workflow.py",
            "gpu_telemetry.py",
            "analysis.py",
            "evaluation.py",
            "alfworld_domain.py",
            "launch.py",
            "resources.py",
            "train.py",
        )
    }
    manifest = root / "source_manifest.json"
    if manifest.is_file() and read_json(manifest) != sources:
        raise ValueError("Cannot resume 9B Solver validation after source changes")
    write_json(manifest, sources)
    (root / "FAILED.json").unlink(missing_ok=True)
    stop_known(require_exclusive=True)
    launch = Launcher(root / "controller_supervision")
    try:
        combined_ids = spec["selection_task_ids"] + spec["report_task_ids"]
        _status(root, "baseline_evaluation", started)
        baseline_all = _run_evaluation(
            root,
            launch,
            spec,
            name="baseline_w0",
            model=spec["initial_solver"],
            version=f"w0-seed{spec['seed']}",
            task_ids=combined_ids,
        )
        baseline_selection = subset_report(
            baseline_all, spec["selection_task_ids"],
        )
        baseline_report = subset_report(baseline_all, spec["report_task_ids"])
        write_json(root / "baseline_selection.json", baseline_selection)
        write_json(root / "baseline_report.json", baseline_report)
        _status(root, "training", started)
        trained = _run_training(root, launch, spec)
        history = trained.get("checkpoint_history", {}).get("actor", [])
        if not history:
            history = [{
                "global_step": spec["total_steps"] - 1,
                "path": trained["checkpoints"]["actor"],
            }]
        candidates = []
        for checkpoint in history:
            ensure_model_runtime_metadata(
                Path(checkpoint["path"]), Path(spec["initial_solver"]),
            )
            step = checkpoint["global_step"]
            _status(root, "checkpoint_selection", started, global_step=step)
            evaluation = _run_evaluation(
                root,
                launch,
                spec,
                name=f"selection_globalstep{step}",
                model=checkpoint["path"],
                version=f"w1-step{step}-seed{spec['seed']}",
                task_ids=spec["selection_task_ids"],
            )
            candidates.append({**checkpoint, "evaluation": evaluation})
        selection = select_checkpoint(
            baseline_selection,
            candidates,
            spec["seed"],
            minimum_delta=spec["selection_min_delta"],
        )
        write_json(root / "checkpoint_selection.json", selection)
        selected = selection["selected"]["path"]
        if selected is None:
            selected_report = baseline_report
        else:
            step = selection["selected"]["global_step"]
            _status(root, "selected_report", started, global_step=step)
            selected_report = _run_evaluation(
                root,
                launch,
                spec,
                name=f"report_globalstep{step}",
                model=selected,
                version=f"w1-report-step{step}-seed{spec['seed']}",
                task_ids=spec["report_task_ids"],
            )
        difference = paired_difference(
            baseline_report,
            selected_report,
            seed=spec["seed"] + 7_000_003,
        )
        write_json(root / "paired_report.json", difference)
        complete = {
            "status": "complete",
            "seed": spec["seed"],
            "harness": "fixed_no_patch",
            "initial_solver": spec["initial_solver"],
            "selected_solver": selected or spec["initial_solver"],
            "selected_trained_checkpoint": selected is not None,
            "selection": selection,
            "paired_report": difference,
            "training": {
                "actor_steps": spec["actor_steps"],
                "critic_warmup_steps": spec["critic_warmup_steps"],
                "batch_size": spec["batch_size"],
                "actor_gpus": spec["actor_gpus"],
                "rollout_gpus": spec["rollout_gpus"],
                "critic_updates_per_step": spec["critic_updates_per_step"],
                "critic_mode": spec["critic_mode"],
                "solver_trajectory_style": spec["solver_trajectory_style"],
                "checkpoint_history": history,
            },
            "test_split_opened": False,
            "elapsed_seconds": time.monotonic() - started,
        }
        write_json(root / "COMPLETE.json", complete)
        _status(root, "complete", started, paired_report=difference)
        return complete
    except SystemExit:
        raise
    except BaseException as exc:
        write_json(root / "FAILED.json", {
            "status": "failed",
            "stage": (
                read_json(root / "STATUS.json").get("stage")
                if (root / "STATUS.json").is_file()
                else "startup"
            ),
            "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        _status(root, "failed", started, error=repr(exc))
        raise
    finally:
        launch.close()
        write_json(root / "gpu_release_audit.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.spec), flush=True)


if __name__ == "__main__":
    main()
