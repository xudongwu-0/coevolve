"""Prepare and audit the prespecified 32-round role-credit experiment."""
from __future__ import annotations

import copy
from pathlib import Path
import time
import subprocess
import shlex

from .alfworld_train_data import _immutable_json
from .common import read_json, file_sha, write_json
from .joint_experiment import validate_spec
from .joint_health import metric_rows
from .joint_reward import role_reward


def audit_training(root: Path, steps: int, warmup: int) -> dict:
    """Require actual updates and a unique, complete paired rollout ledger."""
    result = {}
    expected = {(v, s) for v in range(steps) for s in range(8)}
    if list((root / "joint").glob("*_FAILED.json")):
        raise ValueError("Shared environment failure cannot pass the study gate")
    for role in ("editor", "solver"):
        if (root / role / "FAILED.json").exists() or (root / f"{role}_FAILED.json").exists():
            raise ValueError("Training failure cannot pass the study gate")
        record = read_json(root / role / "COMPLETE.json")
        metrics = metric_rows(root / role / "optimizer/metrics.jsonl")
        if (record["status"] != "completed" or record["steps"] != steps
                or record["actor_updates"] != steps - warmup
                or record["critic_updates"] != steps * 2
                or len(metrics) != steps
                or [row["global_step"] for row in metrics] != list(range(steps))
                or file_sha(root / role / "optimizer/metrics.jsonl") != record["metrics_sha256"]):
            raise ValueError("Optimizer budget or metrics mismatch")
        receipts = [read_json(p) for p in (root / "joint").glob(f"round-*/{role}-receipt-*.json")]
        if (len(receipts) != len(expected)
                or {(x["version"], x["slot"]) for x in receipts} != expected
                or len({x["sample_id"] for x in receipts}) != len(expected)):
            raise ValueError("Generated ledger is missing, duplicated or over budget")
        spec = read_json(root / f"{role}_spec.json")
        mode = spec.get("editor_reward_mode", "team")
        invalid = invalid_success = 0
        for receipt in receipts:
            if receipt["editor_reward_mode"] != mode or receipt["reward"] != role_reward(
                    role, receipt["team_reward"], receipt["model_error"], mode):
                raise ValueError("Role credit differs from frozen objective")
            invalid += receipt["model_error"] is not None
            invalid_success += receipt["model_error"] is not None and receipt["team_reward"] == 1
        release = read_json(root / role / "RELEASED.json")
        if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
            raise ValueError("Training GPUs not released")
        result[role] = {"steps": steps, "consumed": steps * 8, "generated": len(receipts),
                        "invalid": invalid, "invalid_team_success": invalid_success,
                        "mode": mode, "metrics_sha256": record["metrics_sha256"]}
    outcomes = [read_json(p) for p in (root / "joint").glob("round-*/outcome-*.json")]
    if len(outcomes) != len(expected) or {(x["version"], x["slot"]) for x in outcomes} != expected:
        raise ValueError("Paired environment outcome ledger differs from training budget")
    return result


def prepare_arms(study: Path, parent: Path) -> dict:
    design = read_json(study / "protocol.json")
    for path, digest in design["pinned_files"].items():
        if file_sha(Path(path)) != digest:
            raise ValueError("Frozen design input changed")
    prepared = {}
    for arm, mode in design["arms"].items():
        root = study / arm
        root.mkdir(parents=True, exist_ok=True)
        root.chmod(0o2775)
        steps = design["rounds_per_arm"]
        rows = read_json(parent / "data.json")[:steps * 8]
        _immutable_json(root / "data.json", rows)
        protocol = copy.deepcopy(read_json(parent / "protocol.json"))
        protocol.update(run=str(root), report_deadline_unix=time.time() + 10 * 3600,
                        training_deadline_unix=time.time() + 9 * 3600,
                        credit_study=str(study / "protocol.json"), editor_reward_mode=mode)
        protocol["budget"].update(maximum_rounds=steps, matched_rounds=steps,
                                  checkpoint_trajectories=list(range(128, steps * 16 + 1, 128)))
        protocol["selection"]["best"] = "Not an endpoint of this study; do not run."
        protocol["selection"]["matched"] = "Prespecified round32 endpoint; selection48 only."
        protocol["data_sha256"] = file_sha(root / "data.json")
        _immutable_json(root / "protocol.json", protocol)
        # Reuse the exact same initialization measurement solely as a prerequisite.
        _immutable_json(root / "initial_COMPLETE.json", read_json(parent / "initial_COMPLETE.json"))
        copy_initial_reports(parent, root)
        for role in ("editor", "solver"):
            spec = read_json(parent / f"{role}_spec.json")
            spec.update(run=str(root / role / "optimizer"), joint_root=str(root / "joint"),
                        experiment=f"{study.name}_{arm}", steps=steps, data=str(root / "data.json"),
                        editor_reward_mode=mode,
                        training_deadline_unix=protocol["training_deadline_unix"])
            spec["pinned_files"].pop(str(parent / "data.json"))
            spec["pinned_files"].update({str(root / "data.json"): file_sha(root / "data.json"),
                                        str(study / "protocol.json"): file_sha(study / "protocol.json")})
            validate_spec(spec)
            _immutable_json(root / f"{role}_spec.json", spec)
        prepared[arm] = str(root)
    _immutable_json(study / "PREPARED.json", prepared)
    return prepared


def copy_initial_reports(parent: Path, root: Path) -> dict:
    """Carry the reused measurement itself, not just its completion summary."""
    copied = {}
    for variant in ("joint_harness", "no_patch"):
        source = parent / f"initial_{variant}_report.json"
        target = root / source.name
        _immutable_json(target, read_json(source))
        copied[str(source)] = file_sha(source)
    return copied


def launch_unit(study: Path, source: Path, root: Path, mode: str, role: str | None = None):
    name = f"validity-credit-{root.name}-{role or mode}-v1"
    remote = role == "editor"
    marker = study / "units" / f"{name}.json"
    if marker.exists():
        return read_json(marker)
    main = study.parents[1]
    args = ["sudo", "-n", "systemd-run", f"--unit={name}",
            f"--uid={1009 if remote else 1008}", "--gid=1008",
            "--property=UMask=0002", "--property=KillMode=control-group",
            "--property=TimeoutStopSec=90", "--property=RuntimeMaxSec=14400",
            f"--property=WorkingDirectory={source}",
            f"--property=StandardOutput=append:{root}/{role or mode}.service.log",
            "--property=StandardError=inherit", "--setenv=TMPDIR=/dev/shm",
            f"--setenv=PYTHONPATH={source}/whale_local:{main}/third_party/sao/vendor/AReaL",
            "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=HF_HUB_OFFLINE=1",
            "--setenv=WANDB_MODE=disabled", "/dev/shm/sao-venv-ppo/bin/python", "-B",
            "-m", "sao_bridge.joint_campaign", "--root", str(root), "--mode", mode]
    args += ["--role", role or "matched"]
    if remote:
        args = ["ssh", "-F", "/mnt/3fs/xudong/cluster-ssh/config", "node3-internal", shlex.join(args)]
    subprocess.run(args, check=True, timeout=60)
    result = {"unit": name, "remote": remote, "at_unix": time.time()}
    _immutable_json(marker, result)
    return result


def wait_files(root: Path, required: list[str], timeout: float = 14400):
    deadline = time.monotonic() + timeout
    while not all((root / name).exists() for name in required):
        failed = list(root.glob("*FAILED.json")) + list(root.glob("*/FAILED.json"))
        recovery = root / "RESOLVED_FAILURES.json"
        if recovery.exists():
            resolved = read_json(recovery)
            # A newly overwritten failure cannot be hidden by an old recovery.
            if any(file_sha(Path(p)) != digest for p, digest in resolved["evidence"].items()):
                raise ValueError("Recovery evidence changed")
            failed = [p for p in failed if resolved["failures"].get(str(p.resolve())) != file_sha(p)]
        if failed:
            raise RuntimeError(f"Study dependency failed: {failed}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Study dependency timeout: {root}")
        time.sleep(10)


def run(study: Path, source: Path, smoke: Path) -> dict:
    from .analysis import paired_difference

    try:
        wait_files(smoke, [f"{role}/{name}.json" for role in ("editor", "solver")
                           for name in ("COMPLETE", "RELEASED")])
        _immutable_json(study / "GPU_SMOKE_AUDIT.json", audit_training(smoke, 2, 1))
        design = read_json(study / "protocol.json")
        endpoints = {}
        for arm in ("control", "treatment"):
            root = study / arm
            for role in ("editor", "solver"):
                launch_unit(study, source, root, "train", role)
            wait_files(root, [f"{role}_SHARED.json" for role in ("editor", "solver")])
            _immutable_json(root / "TRAINING_AUDIT.json", audit_training(root, 32, 3))
            launch_unit(study, source, root, "evaluate")
            wait_files(root, ["matched_COMPLETE.json", "matched_RELEASED.json"])
            endpoint = read_json(root / "matched_COMPLETE.json")
            release = read_json(root / "matched_RELEASED.json")
            if (endpoint["status"] != "completed" or endpoint["selection"]["rounds"] != 32
                    or release["foreign_gpu_pids"] or not release["filler_unit_active"]):
                raise ValueError("Fixed round32 endpoint or GPU release failed")
            entry = endpoint["reports"]["joint_harness"]
            if file_sha(Path(entry["path"])) != entry["sha256"]:
                raise ValueError("Endpoint report changed")
            endpoints[arm] = read_json(Path(entry["path"]))
        result = {"status": "primary_complete_diagnostics_pending",
                  "paired_treatment_minus_control": paired_difference(
                      endpoints["control"], endpoints["treatment"], seed=20261040),
                  "treatment_point_exceeds_historical92": endpoints["treatment"]["accuracy"] > .92,
                  "training_consumed_including_prior_branch_and_smoke":
                      design["training_cost_before_this_study"] + 32 + 1024,
                  "pending": ["Fixed-h0 diagnostic", "Contemporaneous W1+h2 baseline",
                              "Selection48/report152 contrasts", "Robustness if superiority candidate"],
                  "at_unix": time.time()}
        _immutable_json(study / "PRIMARY_RESULTS.json", result)
        return result
    except BaseException as exc:
        write_json(study / "FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.study.resolve(), args.source.resolve(), args.smoke.resolve()), flush=True)
