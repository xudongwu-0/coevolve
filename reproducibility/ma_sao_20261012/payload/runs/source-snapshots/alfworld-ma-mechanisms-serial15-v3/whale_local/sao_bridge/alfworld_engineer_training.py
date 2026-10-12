"""Persist and retry one finite Engineer training phase; evaluation is separate."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from .alfworld_solver_validation import _fresh_retry_root
from .alfworld_train_data import _immutable_json
from .common import ROOT, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit


def publish_completion(root: Path, result: dict) -> None:
    if result.get("status") != "completed":
        raise ValueError("Engineer training completion receipt is invalid")
    write_json(root / "TRAINING_COMPLETE.json", result)
    (root / "FAILED.json").unlink(missing_ok=True)
    write_json(root / "STATUS.json", {
        "stage": "training_complete_pending_evaluation", "updated_at_unix": time.time(),
    })


def run(config_path: Path) -> dict:
    config = read_json(config_path)
    training = Path(config["run"]).resolve()
    root = training.parent
    if training.name != "training" or root.parent != ROOT / "runs":
        raise ValueError("Engineer training must use a dedicated runs/<name>/training directory")
    root.mkdir(parents=True, exist_ok=True)
    source_names = (
        "alfworld_engineer_training.py", "alfworld_solver_validation.py",
        "alfworld_h_train.py", "alfworld_reward_service.py",
        "alfworld_runtime.py", "alfworld_train_data.py", "alfworld_domain.py",
        "large_train.py", "large_workflow.py", "train.py", "terminal_actor.py",
        "launch.py", "resources.py", "gpu_telemetry.py", "budget.py", "common.py",
    )
    identity = {
        "config": config,
        "data_sha256": file_sha(Path(config["data"])),
        "engineer_provenance_sha256": file_sha(Path(config["engineer_model"]) / "provenance.json"),
        "solver_provenance_sha256": file_sha(Path(config["solver_model"]) / "provenance.json"),
        "sources": {name: file_sha(Path(__file__).with_name(name)) for name in source_names},
    }
    _immutable_json(root / "training_identity.json", identity)
    release = gpu_release_audit()
    if (
        release["foreign_gpu_pids"] or release["filler_kind"] != "adaptive"
        or not release["filler_unit_active"]
    ):
        raise RuntimeError("Engineer phase requires free GPUs and the active adaptive filler")
    completed = training / "COMPLETE.json"
    if completed.is_file():
        result = read_json(completed)
        publish_completion(root, result)
        write_json(root / "training_gpu_release.json", release)
        return result
    attempts_path = root / "training_attempts.json"
    attempts = read_json(attempts_path) if attempts_path.exists() else []
    if len(attempts) >= 2:
        write_json(root / "FAILED.json", {"stage": "training", "error": "two attempts exhausted"})
        raise SystemExit(78)
    _fresh_retry_root(training, root / "failed_training_attempts" / f"attempt-{len(attempts):02d}")
    attempts.append({"attempt": len(attempts) + 1, "started_at_unix": time.time()})
    write_json(attempts_path, attempts)
    (root / "FAILED.json").unlink(missing_ok=True)
    write_json(root / "STATUS.json", {
        "stage": "training", "updated_at_unix": time.time(), "attempt": len(attempts),
    })
    launch = Launcher(root / "training_supervision")
    try:
        process = launch.module(
            "sao_bridge.alfworld_h_train", ["--config", config_path],
            gpus="0,1,2,3,4,5,6,7", name=f"training_attempt_{len(attempts)}",
        )
        launch.wait(
            process, "fast Engineer training",
            timeout=float(config.get("phase_timeout_seconds", 14400)) + 600,
        )
        result = read_json(completed)
        attempts[-1].update(finished_at_unix=time.time(), complete=True)
        publish_completion(root, result)
        return result
    except BaseException as exc:
        attempts[-1].update(finished_at_unix=time.time(), error=repr(exc))
        write_json(root / "FAILED.json", {"stage": "training", "error": repr(exc)})
        write_json(root / "STATUS.json", {
            "stage": "failed", "updated_at_unix": time.time(), "error": repr(exc),
        })
        raise
    finally:
        launch.close()
        write_json(attempts_path, attempts)
        write_json(root / "training_gpu_release.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config.resolve()), flush=True)


if __name__ == "__main__":
    main()
