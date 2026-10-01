"""Persistent, H0-fixed sequential SAO validation for the 4B math Solver."""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
import shutil
import socket
import threading
import time

from .analysis import paired_difference, paired_lower_bound
from .common import file_sha, read_json, write_json
from .evaluation import Sampler, evaluate
from .gpu_telemetry import monitor as monitor_gpus, summarize_active_training
from .launch import Launcher
from .phase_data import solver_dataset
from .resources import gpu_release_audit, stop_known
from .retool import INITIAL_SOURCE


REQUIRED_KEYS = {
    "run", "model", "data", "seed", "actor_steps", "critic_warmup_steps",
    "batch_size", "lr", "tokens", "save_interval_steps",
}


def validate_spec(spec: dict) -> dict:
    missing = REQUIRED_KEYS - spec.keys()
    if missing:
        raise ValueError(f"Missing solver-validation keys: {sorted(missing)}")
    normalized = dict(spec)
    integers = (
        "seed", "actor_steps", "critic_warmup_steps", "batch_size", "tokens",
        "save_interval_steps", "selection_questions", "selection_samples",
        "report_dev_offset", "report_dev_questions", "report_train_tail",
        "report_samples", "max_training_attempts", "actor_gpus", "rollout_gpus",
        "critic_updates_per_step",
    )
    defaults = {
        "selection_questions": 256,
        "selection_samples": 2,
        "report_dev_offset": 128,
        "report_dev_questions": 128,
        "report_train_tail": 384,
        "report_samples": 2,
        "max_training_attempts": 2,
        "actor_gpus": 4,
        "rollout_gpus": 4,
        "critic_updates_per_step": 2,
        "critic_mode": "learned",
        "solver_trajectory_style": "individual",
        "terminal_episodes": False,
        "actor_gradient_checkpointing": True,
        "critic_gradient_checkpointing": True,
    }
    for key, value in defaults.items():
        normalized.setdefault(key, value)
    for key in integers:
        if type(normalized[key]) is not int or normalized[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in (
        "actor_steps", "batch_size", "tokens", "save_interval_steps",
        "selection_questions", "selection_samples", "report_dev_questions",
        "report_samples", "max_training_attempts",
        "actor_gpus", "rollout_gpus",
        "critic_updates_per_step",
    ):
        if normalized[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in (
        "terminal_episodes", "actor_gradient_checkpointing",
        "critic_gradient_checkpointing",
    ):
        if type(normalized[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if normalized["critic_mode"] not in {"learned", "none"}:
        raise ValueError("critic_mode must be learned or none")
    if normalized["critic_mode"] == "none" and normalized["critic_warmup_steps"]:
        raise ValueError("Critic-free training cannot use critic warm-up")
    if normalized["solver_trajectory_style"] not in {"individual", "concat"}:
        raise ValueError("solver_trajectory_style must be individual or concat")
    if normalized["terminal_episodes"] and normalized["solver_trajectory_style"] != "concat":
        raise ValueError("Terminal Solver episodes require concat trajectory style")
    total_steps = normalized["actor_steps"] + normalized["critic_warmup_steps"]
    if total_steps % normalized["save_interval_steps"]:
        raise ValueError("save_interval_steps must divide warm-up plus actor steps")
    if normalized["batch_size"] % normalized["actor_gpus"]:
        raise ValueError("batch_size must be divisible by actor_gpus")
    if not 0 < float(normalized["lr"]) < 1:
        raise ValueError("lr must be in (0, 1)")
    model, data = Path(normalized["model"]).resolve(), Path(normalized["data"]).resolve()
    if not model.is_dir() or not list(model.glob("*.safetensors")):
        raise FileNotFoundError("Solver model has no local safetensors")
    for split in ("train", "reward", "dev"):
        if not (data / f"{split}.json").is_file():
            raise FileNotFoundError(f"Missing {split} split")
    normalized.update(
        run=str(Path(normalized["run"]).resolve()),
        model=str(model),
        data=str(data),
        total_steps=total_steps,
        actor_gpus=normalized["actor_gpus"],
        rollout_gpus=normalized["rollout_gpus"],
    )
    if normalized["actor_gpus"] + normalized["rollout_gpus"] != 8:
        raise ValueError("Solver validation requires all eight node GPUs")
    return normalized


def validation_rows(spec: dict) -> tuple[list[dict], list[dict]]:
    data = Path(spec["data"])
    train, reward, dev = (
        read_json(data / "train.json"),
        read_json(data / "reward.json"),
        read_json(data / "dev.json"),
    )
    selection = reward[:spec["selection_questions"]]
    start = spec["report_dev_offset"]
    report = dev[start:start + spec["report_dev_questions"]]
    tail = spec["report_train_tail"]
    if tail:
        report += train[-tail:]
    train_count = spec["total_steps"] * spec["batch_size"]
    if len(selection) != spec["selection_questions"]:
        raise ValueError("Selection split is smaller than requested")
    if len(report) != spec["report_dev_questions"] + tail:
        raise ValueError("Report split is smaller than requested")
    if train_count > len(train) - tail:
        raise ValueError("Training rows overlap the held-out train tail")
    ids = {
        "training": {row["id"] for row in train[:train_count]},
        "selection": {row["id"] for row in selection},
        "report": {row["id"] for row in report},
    }
    if ids["training"] & ids["selection"] or ids["training"] & ids["report"]:
        raise ValueError("Training and evaluation questions overlap")
    if ids["selection"] & ids["report"]:
        raise ValueError("Selection and report questions overlap")
    return selection, report


def select_checkpoint(control: dict, candidates: list[dict], seed: int) -> dict:
    if not candidates:
        raise ValueError("No trained checkpoints were evaluated")
    alpha = 0.05 / len(candidates)
    scored = []
    for index, candidate in enumerate(candidates):
        scored.append({
            "global_step": candidate["global_step"],
            "path": candidate["path"],
            "accuracy": candidate["evaluation"]["accuracy"],
            "candidate_minus_control": (
                candidate["evaluation"]["accuracy"] - control["accuracy"]
            ),
            "selection_lcb": paired_lower_bound(
                control, candidate["evaluation"], alpha=alpha,
                seed=seed + 104_729 * (index + 1),
            ),
        })
    qualified = [row for row in scored if row["selection_lcb"] > 0]
    if qualified:
        selected = max(qualified, key=lambda row: (row["accuracy"], row["global_step"]))
        reason = "positive Bonferroni-adjusted paired lower bound"
    else:
        selected = {
            "global_step": None,
            "path": None,
            "accuracy": control["accuracy"],
            "candidate_minus_control": 0.0,
            "selection_lcb": 0.0,
        }
        reason = "no checkpoint cleared the conservative held-out gate; retained W0"
    return {
        "control_accuracy": control["accuracy"],
        "alpha_per_checkpoint": alpha,
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


def _evaluate_model(
    launch: Launcher,
    model: str,
    rows: list[dict],
    output: Path,
    *,
    samples: int,
    seed: int,
    tokens: int,
    name: str,
) -> dict:
    if output.is_file():
        result = read_json(output)
        if (
            result.get("question_count") != len(rows)
            or result.get("samples_per_question") != samples
        ):
            raise ValueError(f"Cached evaluation protocol mismatch: {output}")
        return result
    gpus = ",".join(str(index) for index in range(8))
    with launch.vllm(
        model,
        port=18801,
        gpus=gpus,
        name=name,
        data_parallel=8,
        max_model_len=8192,
        max_num_seqs=64,
    ) as url:
        async def run():
            sampler = Sampler(url, concurrency=256)
            try:
                return await evaluate(
                    sampler, rows, INITIAL_SOURCE, tokens=tokens,
                    samples=samples, seed=seed,
                )
            finally:
                await sampler.close()

        result = asyncio.run(run())
    write_json(output, result)
    return result


def _training_attempt(root: Path, spec: dict, launch: Launcher) -> dict:
    attempts_path = root / "training_attempts.json"
    attempts = read_json(attempts_path) if attempts_path.is_file() else []
    train_dir = root / "train"
    trained = train_dir / "trained.json"
    if trained.is_file():
        return read_json(trained)
    if train_dir.exists():
        failed = root / "failed_training_attempts" / f"attempt-{len(attempts):02d}"
        failed.parent.mkdir(parents=True, exist_ok=True)
        if failed.exists():
            raise FileExistsError(failed)
        shutil.move(str(train_dir), failed)
    if len(attempts) >= spec["max_training_attempts"]:
        write_json(root / "FAILED.json", {
            "stage": "training",
            "error": "training attempts exhausted",
            "attempts": attempts,
        })
        raise SystemExit(78)
    attempt = {
        "attempt": len(attempts) + 1,
        "started_at_unix": time.time(),
        "host": socket.gethostname(),
    }
    attempts.append(attempt)
    write_json(attempts_path, attempts)
    train_count = spec["total_steps"] * spec["batch_size"]
    data_path = root / "solver_train.json"
    solver_dataset(Path(spec["data"]) / "train.json", data_path, train_count)
    harness = root / "h0.py"
    harness.write_text(INITIAL_SOURCE)
    phase_spec = {
        "run": str(train_dir),
        "role": "solver",
        "model": spec["model"],
        "critic_model": spec["model"],
        "phase": "w1",
        "experiment": root.name,
        "steps": spec["total_steps"],
        "batch_size": spec["batch_size"],
        "seed": spec["seed"],
        "lr": spec["lr"],
        "critic_lr": spec.get("critic_lr", 5 * spec["lr"]),
        "tokens": spec["tokens"],
        "temperature": spec.get("temperature", 1.0),
        "staleness": spec.get("staleness", 1),
        "harness": str(harness),
        "data": str(data_path),
        "weight_update_mode": spec.get("weight_update_mode", "xccl"),
        "actor_gpus": spec["actor_gpus"],
        "rollout_gpus": spec["rollout_gpus"],
        "critic_warmup_steps": spec["critic_warmup_steps"],
        "save_interval_steps": spec["save_interval_steps"],
        **{
            key: spec[key]
            for key in (
                "critic_updates_per_step", "critic_mode", "solver_trajectory_style",
                "terminal_episodes", "actor_gradient_checkpointing",
                "critic_gradient_checkpointing",
            )
        },
    }
    write_json(root / "train_spec.json", phase_spec)
    process = launch.module(
        "sao_bridge.large_train", ["--spec", root / "train_spec.json"],
        gpus=",".join(str(index) for index in range(8)),
        name=f"train_attempt_{attempt['attempt']}",
    )
    try:
        launch.wait(process, "4B Solver SAO training")
    except BaseException as exc:
        attempt.update(finished_at_unix=time.time(), error=repr(exc))
        write_json(attempts_path, attempts)
        raise
    attempt.update(finished_at_unix=time.time(), complete=True)
    write_json(attempts_path, attempts)
    return read_json(trained)


def run(spec_path: Path) -> dict:
    spec = validate_spec(read_json(spec_path))
    root = Path(spec["run"])
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    started = time.monotonic()
    frozen_spec = root / "spec.json"
    if frozen_spec.is_file() and read_json(frozen_spec) != spec:
        raise ValueError("Cannot change a persisted solver-validation specification")
    write_json(frozen_spec, spec)
    manifest = {
        name: file_sha(Path(__file__).with_name(name))
        for name in (
            "math_solver_validation.py", "large_train.py", "large_workflow.py",
            "train.py", "checkpoint.py", "common.py", "phase_data.py",
            "evaluation.py", "retool.py", "analysis.py", "launch.py",
            "resources.py", "gpu_telemetry.py",
        )
    }
    manifest_path = root / "source_manifest.json"
    if manifest_path.is_file() and read_json(manifest_path) != manifest:
        raise ValueError("Cannot resume solver validation after source changes")
    write_json(manifest_path, manifest)
    selection_rows, report_rows = validation_rows(spec)
    stop_known(require_exclusive=True)
    launch = Launcher(root)
    telemetry_stop = threading.Event()
    telemetry_path = root / "gpu_process_samples.jsonl"
    telemetry = threading.Thread(
        target=monitor_gpus,
        args=(telemetry_path, telemetry_stop, 5.0),
        daemon=True,
    )
    telemetry.start()
    try:
        _status(root, "baseline_selection", started)
        baseline_selection = _evaluate_model(
            launch, spec["model"], selection_rows, root / "baseline_selection.json",
            samples=spec["selection_samples"], seed=spec["seed"] + 2_000_003,
            tokens=spec["tokens"], name="baseline_selection",
        )
        _status(root, "baseline_report", started)
        baseline_report = _evaluate_model(
            launch, spec["model"], report_rows, root / "baseline_report.json",
            samples=spec["report_samples"], seed=spec["seed"] + 5_000_003,
            tokens=spec["tokens"], name="baseline_report",
        )
        _status(root, "training", started)
        trained = _training_attempt(root, spec, launch)
        history = trained.get("checkpoint_history", {}).get("actor", [])
        if not history:
            history = [{
                "global_step": spec["total_steps"] - 1,
                "path": trained["checkpoints"]["actor"],
            }]
        evaluated = []
        for checkpoint in history:
            step = checkpoint["global_step"]
            _status(root, "checkpoint_selection", started, global_step=step)
            result = _evaluate_model(
                launch, checkpoint["path"], selection_rows,
                root / f"selection_globalstep{step}.json",
                samples=spec["selection_samples"], seed=spec["seed"] + 2_000_003,
                tokens=spec["tokens"], name=f"selection_step_{step}",
            )
            evaluated.append({**checkpoint, "evaluation": result})
        selection = select_checkpoint(baseline_selection, evaluated, spec["seed"])
        write_json(root / "checkpoint_selection.json", selection)
        selected = selection["selected"]["path"]
        _status(root, "selected_report", started, selected_checkpoint=selected)
        if selected is None:
            selected_report = baseline_report
        else:
            selected_report = _evaluate_model(
                launch, selected, report_rows, root / "selected_report.json",
                samples=spec["report_samples"], seed=spec["seed"] + 5_000_003,
                tokens=spec["tokens"], name="selected_report",
            )
        difference = paired_difference(
            baseline_report, selected_report, seed=spec["seed"] + 7_000_003,
        )
        write_json(root / "paired_report.json", difference)
        complete = {
            "status": "complete",
            "seed": spec["seed"],
            "harness": "fixed_h0",
            "initial_solver": spec["model"],
            "selected_solver": selected or spec["model"],
            "selected_trained_checkpoint": selected is not None,
            "selection": selection,
            "paired_report": difference,
            "training": {
                "actor_steps": spec["actor_steps"],
                "critic_warmup_steps": spec["critic_warmup_steps"],
                "batch_size": spec["batch_size"],
                "generated_training_episodes": spec["total_steps"] * spec["batch_size"],
                "critic_updates_per_step": spec["critic_updates_per_step"],
                "critic_mode": spec["critic_mode"],
                "solver_trajectory_style": spec["solver_trajectory_style"],
                "terminal_episodes": spec["terminal_episodes"],
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
        _status(root, "failed", started, error=repr(exc))
        raise
    finally:
        launch.close()
        telemetry_stop.set()
        telemetry.join(timeout=10)
        if telemetry_path.is_file():
            utilization = summarize_active_training(
                telemetry_path, root / "train/training_active.json",
            )
            if utilization is not None:
                write_json(root / "gpu_optimizer_utilization_summary.json", utilization)
        write_json(root / "gpu_release_audit.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.spec)
    print({
        "seed": result["seed"],
        "selected_trained_checkpoint": result["selected_trained_checkpoint"],
        "candidate_minus_control": result["paired_report"]["candidate_minus_control"],
        "interval": result["paired_report"]["paired_question_bootstrap_95pct"],
    }, flush=True)


if __name__ == "__main__":
    main()
