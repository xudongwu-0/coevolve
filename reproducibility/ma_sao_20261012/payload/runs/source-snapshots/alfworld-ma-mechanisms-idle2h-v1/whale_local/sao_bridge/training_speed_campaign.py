"""Monitor the existing training, validate acceleration, and evaluate its endpoint."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import time

from .alfworld_train_data import _immutable_json
from .common import ROOT, file_sha, read_json, write_json
from .joint_health import metric_rows


PYTHON = "/dev/shm/sao-venv-ppo/bin/python"
SSH = ["ssh", "-F", "/mnt/3fs/xudong/cluster-ssh/config", "node3-internal"]


def unit_state(unit: str, remote: bool = False) -> dict:
    command = ["systemctl", "show", unit, "--property=LoadState,ActiveState,SubState,MainPID,Result"]
    result = subprocess.check_output(SSH + [shlex.join(command)] if remote else command,
                                     text=True, timeout=30)
    return dict(line.split("=", 1) for line in result.splitlines() if "=" in line)


def launch_stage(root: Path, name: str, module: str, args: list[str],
                 *, remote: bool, accelerated: bool = False, source: Path = ROOT) -> dict:
    config = read_json(root / "DESIGN.json")
    unit = f"alfworld-speed-{name}-{config.get('unit_suffix', 'v1')}"
    if unit_state(unit, remote)["LoadState"] != "not-found":
        raise RuntimeError(f"Refusing duplicate stage: {unit}")
    paths = [str(source / "whale_local"), str(source / "third_party/sao/vendor/AReaL")]
    if accelerated:
        paths.insert(0, config["overlay"])
    cache = root / f"cache-{name}"
    cache.mkdir(exist_ok=False)
    command = [
        "sudo", "-n", "systemd-run", f"--unit={unit}", f"--uid={1009 if remote else 1008}",
        "--gid=1008", "--property=UMask=0002", "--property=KillMode=control-group",
        "--property=TimeoutStopSec=90", "--property=RuntimeMaxSec=14400",
        f"--property=WorkingDirectory={source}",
        f"--property=StandardOutput=append:{root}/{name}.log", "--property=StandardError=inherit",
        "--setenv=TMPDIR=/dev/shm", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=HF_HUB_OFFLINE=1", "--setenv=WANDB_MODE=disabled",
        "--setenv=OMP_NUM_THREADS=4", f"--setenv=PYTHONPATH={':'.join(paths)}",
        f"--setenv=TRITON_CACHE_DIR={cache}/triton",
        f"--setenv=TORCHINDUCTOR_CACHE_DIR={cache}/torch",
        f"--setenv=HOME={cache}", "--setenv=CUDA_VISIBLE_DEVICES=0,1,2,3",
        PYTHON, "-B", "-m", module, *args,
    ]
    subprocess.run(SSH + [shlex.join(command)] if remote else command, check=True, timeout=60)
    receipt = {"unit": unit, "remote": remote, "at_unix": time.time(), "command": command}
    _immutable_json(root / f"{name}_LAUNCHED.json", receipt)
    return receipt


def check_stage(stage: dict) -> bool:
    state = unit_state(stage["unit"], stage["remote"])
    if state["ActiveState"] == "failed" or state["Result"] != "success":
        raise RuntimeError(f"Stage failed: {stage['unit']}: {state}")
    if state["ActiveState"] == "inactive":
        return True
    if state["ActiveState"] not in ("active", "activating", "deactivating"):
        raise RuntimeError(f"Unexpected stage state: {stage}: {state}")
    return False


def monitor(root: Path, stages: list[dict], *, phase: str, watch: list[dict] | None = None) -> None:
    deadline = time.monotonic() + 4 * 3600
    while True:
        for stage in watch or []:
            check_stage(stage)
        states = {stage["unit"]: unit_state(stage["unit"], stage["remote"]) for stage in stages}
        done = [check_stage(stage) for stage in stages]
        write_json(root / "MONITOR.json", {"phase": phase, "at_unix": time.time(), "states": states})
        if all(done):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Stage deadline exceeded: {phase}")
        time.sleep(30)


def run(root: Path) -> None:
    design = read_json(root / "DESIGN.json")
    for path, digest in read_json(root / "EXECUTION_PINS.json")["pins"].items():
        if file_sha(Path(path)) != digest:
            raise ValueError(f"Speed-campaign input changed: {path}")
    training_root = Path(design["training_root"])
    predecessor = [
        {"unit": "alfworld-explicit-hint-repeat-driver-v1", "remote": False},
        {"unit": "alfworld-explicit-hint-repeat-solver-v1", "remote": False},
        {"unit": "alfworld-explicit-hint-repeat-editor-v1", "remote": True},
    ]
    deadline = time.monotonic() + 4 * 3600
    while True:
        done = [check_stage(stage) for stage in predecessor]
        progress = {}
        for role in ("editor", "solver"):
            rows = metric_rows(training_root / role / "optimizer/metrics.jsonl")
            progress[role] = {"committed_rounds": len(rows),
                              "last_recorded_at": rows[-1]["recorded_at_unix"] if rows else None}
            if rows and time.time() - rows[-1]["recorded_at_unix"] > 1800 and not all(done):
                raise TimeoutError(f"Existing {role} training has no committed progress for 30 minutes")
        write_json(root / "MONITOR.json", {
            "phase": "waiting_for_existing_training", "at_unix": time.time(), "training": progress,
            "states": {s["unit"]: unit_state(s["unit"], s["remote"]) for s in predecessor},
        })
        if all(done):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Existing training observation deadline exceeded")
        time.sleep(30)
    audit = read_json(training_root.parent / "TRAINING_AUDIT.json")
    if audit["status"] != "passed" or audit["new_consumed_role_trajectories"] != 640:
        raise ValueError("Existing replication did not complete its fixed budget")

    if design.get("existing_evaluation"):
        evaluation = design["existing_evaluation"]
        check_stage(evaluation)
        _immutable_json(root / "EVALUATION_REUSE.json", {
            "stage": evaluation, "reason": "Existing primary continues unchanged after kernel-only failure",
            "at_unix": time.time(),
        })
    else:
        evaluation = launch_stage(root, "existing-primary", "sao_bridge.joint_campaign",
                                  ["--root", str(training_root), "--mode", "evaluate", "--role", "matched"],
                                  remote=False, source=Path(design["existing_source"]))
    kernel_root = Path(design.get("existing_kernel", str(root / "kernel")))
    if not design.get("existing_kernel"):
        kernel = launch_stage(root, "kernel", "sao_bridge.training_acceleration",
                              ["--output", str(kernel_root)], remote=True, accelerated=True)
        monitor(root, [kernel], phase="kernel_validation", watch=[evaluation])
    if read_json(kernel_root / "KERNEL_COMPLETE.json")["status"] != "passed":
        raise ValueError("Kernel check did not pass")
    h0 = None
    h0_args = [
        "--output", str(root / "native-h0-fla"), "--model-role", "h0",
        "--actor-rounds", "2", "--critic-updates-per-round", "2", "--critic-warmup-rounds", "1",
    ]
    if design.get("reference_storage_dtype"):
        h0_args += ["--reference-storage-dtype", design["reference_storage_dtype"]]
    if design.get("parallel_h0_on_node4"):
        if not check_stage(evaluation):
            raise RuntimeError("Parallel H0 validation requires the primary evaluation to release node4")
        h0 = launch_stage(root, "native-h0-fla", "sao_bridge.joint_memory_check",
                          h0_args, remote=False, accelerated=True)
    native = {}
    for name, accelerated in (("native-baseline", False), ("native-fla", True)):
        if name == "native-baseline" and design.get("existing_native_baseline"):
            native[name] = Path(design["existing_native_baseline"])
            continue
        args = [
            "--output", str(root / name), "--model-role", "w0",
            "--actor-rounds", "2", "--critic-updates-per-round", "2",
            "--critic-warmup-rounds", "1",
        ]
        if accelerated and design.get("reference_storage_dtype"):
            args += ["--reference-storage-dtype", design["reference_storage_dtype"]]
        stage = launch_stage(root, name, "sao_bridge.joint_memory_check", args,
                             remote=True, accelerated=accelerated)
        monitor(root, [stage], phase=name, watch=[evaluation, *([h0] if h0 else [])])
        native[name] = root / name
    from .training_acceleration import compare_native
    numerical = compare_native(native["native-baseline"], native["native-fla"], root / "NATIVE_COMPARISON.json")
    if h0 is None:
        h0 = launch_stage(root, "native-h0-fla", "sao_bridge.joint_memory_check", h0_args,
                          remote=True, accelerated=True)
    monitor(root, [h0, evaluation], phase="h0_validation_and_primary_evaluation")
    if read_json(root / "native-h0-fla/result.json")["status"] != "passed":
        raise ValueError("Accelerated H0 full-context validation failed")
    primary = read_json(training_root / "matched_COMPLETE.json")
    if primary["status"] != "completed":
        raise ValueError("Primary evaluation did not complete")
    _immutable_json(root / "COMPLETE.json", {
        "status": "passed", "at_unix": time.time(), "native_comparison": numerical,
        "primary_endpoint": primary["scores"], "primary_comparison": primary["versus_w1_h2"],
        "primary_path": str(training_root / "matched_COMPLETE.json"),
        "new_sao_training": 0, "new_training_seeds": 0,
        "research_training_scope": "Only the already-running MA-SAO arm; native acceleration tests are synthetic",
        "manual_extension": "Deferred by user reprioritization; not silently removed from the original design",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o002)
    try:
        run(args.root.resolve())
    except BaseException as exc:
        write_json(args.root / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise


if __name__ == "__main__":
    main()
