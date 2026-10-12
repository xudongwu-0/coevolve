"""Deadline-bounded, agent-independent H2/W2 continuation of validated E15/W79."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from .alfworld_engineer_eval import EvaluationAttemptsExhausted
from .alfworld_strong_arm import checked_selected_state
from .alfworld_strong_phase import (
    PhaseAttemptsExhausted, checked_release, checkpoint_pair, finalize_common,
)
from .alfworld_strong_refresh import run as refresh_partner
from .alfworld_strong_report import compare
from .alfworld_strong_select import (
    SelectionAttemptsExhausted, generate_proposals, matrix_config,
    publish_selection_binding, run_matrix,
)
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import ROOT, ensure_model_runtime_metadata, file_sha, read_json, write_json
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


def validate_protocol(path: Path) -> dict:
    protocol = read_json(path)
    settings = protocol["autonomous"]
    if (
        protocol["phase_orders"] != {"sequential": ["h2", "w2"], "ma": ["h2", "w2"]}
        or protocol["initial_solver_group"] != "initial_w1"
        or protocol["evaluation"]["report_mode"] != "final_systems"
        or protocol["max_phase_attempts"] != 2
        or Path(settings["source_worktree"]).resolve() != ROOT.resolve()
        or not re.fullmatch(r"coevolve-alt8h-[a-z0-9-]+", settings["unit_prefix"])
    ):
        raise ValueError("Campaign changed its matched order, source, initial group or retry budget")
    start = settings["started_at_unix"]
    if not (
        start < settings["handoff_deadline_unix"] < settings["common_deadline_unix"]
        < settings["training_deadline_unix"] < settings["deadline_unix"]
        and settings["deadline_unix"] - start == 8 * 3600
        and settings["deadline_unix"] - settings["training_deadline_unix"] >= 3600
    ):
        raise ValueError("Campaign must retain its original eight-hour deadline and report reserve")
    if not settings["pinned_files"] or any(
        file_sha(Path(name)) != digest for name, digest in settings["pinned_files"].items()
    ):
        raise ValueError("Campaign source evidence or templates changed")
    return protocol


def stage_unit(protocol: dict, arm: str, stage: str) -> str:
    if arm not in ("sequential", "ma") or stage not in ("common", "arm", "report"):
        raise ValueError("Unknown campaign unit role")
    return f"{protocol['autonomous']['unit_prefix']}-{arm}-{stage}.service"


def unit_status(unit: str) -> dict:
    text = subprocess.check_output([
        "systemctl", "show", unit,
        "--property=LoadState,ActiveState,MainPID,Result,ExecMainStatus,NRestarts",
    ], text=True)
    result = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if result.get("LoadState") not in {"loaded", "not-found"}:
        raise RuntimeError(f"Cannot inspect campaign unit {unit}: {result}")
    return result


def stop_unit(protocol: dict, arm: str, unit: str, *, prior: bool = False) -> None:
    allowed = (
        unit in protocol["autonomous"]["prior_units"][arm] and unit.startswith("coevolve-strong-")
        if prior else unit in {stage_unit(protocol, arm, stage) for stage in ("common", "arm", "report")}
    )
    if not allowed or not unit.endswith(".service"):
        raise ValueError("Refusing to stop a unit outside this campaign's explicit ownership")
    subprocess.run(["sudo", "-n", "systemctl", "stop", unit], check=True)


def install_expiry(protocol: dict, arm: str, stage: str, deadline: float) -> None:
    target = stage_unit(protocol, arm, stage)
    name = target.removesuffix(".service") + "-expiry"
    if unit_status(name + ".timer")["LoadState"] != "not-found":
        return
    when = datetime.fromtimestamp(deadline, timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    command = [
        "sudo", "-n", "systemd-run", "--unit=" + name, "--on-calendar=" + when,
        "--timer-property=AccuracySec=1s", "--timer-property=Persistent=yes",
        "/usr/bin/systemctl", "stop", target,
    ]
    _immutable_json(Path(protocol["run"]) / arm / f"{stage}_expiry.json", {
        "target_unit": target, "deadline_unix": deadline, "command": command,
        "filler_is_not_a_target": True,
    })
    subprocess.run(command, check=True)


def launch_stage(protocol: dict, arm: str, stage: str, args: list[str], deadline: float) -> None:
    unit = stage_unit(protocol, arm, stage)
    remaining = int(deadline - time.time())
    if remaining <= 0:
        raise TimeoutError("Stage cannot start after its original deadline")
    command = [
        "sudo", "-n", "systemd-run", "--unit=" + unit,
        "--uid=" + str(os.getuid()), "--gid=" + str(os.getgid()),
        "--property=WorkingDirectory=" + str(ROOT), "--property=UMask=0022",
        "--property=KillMode=control-group", "--property=TimeoutStopSec=60",
        "--property=Restart=on-failure", "--property=RestartSec=20",
        "--property=RestartPreventExitStatus=78", "--property=StartLimitBurst=3",
        "--property=StartLimitIntervalSec=3600", f"--property=RuntimeMaxSec={remaining}",
        "--property=StandardOutput=append:" + str(Path(protocol["run"]) / arm / f"{stage}.service.log"),
        "--property=StandardError=inherit", "--setenv=TMPDIR=/dev/shm",
        "--setenv=PYTHONPATH=" + str(ROOT / "whale_local"),
        "--setenv=PATH=" + str(Path(sys.executable).parent) + ":" + os.environ.get("PATH", os.defpath),
        "--setenv=CUDA_VISIBLE_DEVICES=", "--setenv=PYTHONDONTWRITEBYTECODE=1",
        "--setenv=PYTHONUNBUFFERED=1", "--setenv=TOKENIZERS_PARALLELISM=false",
        "--setenv=HF_HUB_DISABLE_IMPLICIT_TOKEN=1",
        sys.executable, "-B", *args,
    ]
    root = Path(protocol["run"]) / arm
    _immutable_json(root / f"{stage}_command.json", {"args": args, "source": str(ROOT), "unit": unit})
    launches = root / f"{stage}_launches.json"
    previous = read_json(launches) if launches.exists() else []
    write_json(launches, previous + [{
        "launched_at_unix": time.time(), "deadline_unix": deadline, "command": command,
    }])
    subprocess.run(command, check=True)


def execute_stage(
    protocol: dict, arm: str, stage: str, args: list[str], deadline: float,
    expected: list[Path],
) -> dict:
    unit = stage_unit(protocol, arm, stage)
    while True:
        status = unit_status(unit)
        present = all(path.is_file() for path in expected)
        if status["LoadState"] == "not-found":
            if present:
                return {"status": "complete", "unit": unit, "completion_receipts": list(map(str, expected))}
            if time.time() >= deadline:
                return {"status": "deadline", "unit": unit, "completion_receipts": []}
            launch_stage(protocol, arm, stage, args, deadline)
        elif status["ActiveState"] in {"inactive", "failed"}:
            complete = present and status["Result"] == "success" and status["ExecMainStatus"] == "0"
            return {"status": "complete" if complete else "failed", "unit": unit, "unit_status": status}
        elif time.time() >= deadline:
            stop_unit(protocol, arm, unit)
            return {"status": "deadline", "unit": unit, "unit_status": unit_status(unit)}
        elif status["ActiveState"] not in {"active", "activating", "deactivating", "reloading"}:
            raise RuntimeError(f"Unknown campaign stage state: {status}")
        write_json(Path(protocol["run"]) / arm / "COORDINATOR_STATUS.json", {
            "stage": stage, "unit": unit, "updated_at_unix": time.time(),
            "stage_deadline_unix": deadline, "campaign_deadline_unix": protocol["autonomous"]["deadline_unix"],
        })
        time.sleep(10)


def handoff(protocol: dict, arm: str) -> None:
    root = Path(protocol["run"]) / arm
    receipt = root / "PRIOR_HANDOFF.json"
    if receipt.exists():
        return
    settings = protocol["autonomous"]
    stopped = []
    while True:
        active = [
            unit for unit in settings["prior_units"][arm]
            if unit_status(unit).get("ActiveState") in {"active", "activating", "deactivating", "reloading"}
        ]
        if not active:
            break
        if time.time() >= settings["handoff_deadline_unix"]:
            for unit in active:
                stop_unit(protocol, arm, unit, prior=True)
                stopped.append(unit)
            break
        write_json(root / "COORDINATOR_STATUS.json", {
            "stage": "waiting_for_prior_work", "active_prior_units": active,
            "handoff_deadline_unix": settings["handoff_deadline_unix"], "updated_at_unix": time.time(),
        })
        time.sleep(10)
    release = gpu_release_audit()
    if release["foreign_gpu_pids"] or not release["filler_unit_active"] or release["filler_kind"] != "adaptive":
        raise RuntimeError("Prior work has not released this node with the adaptive filler active")
    _immutable_json(receipt, {
        "completed_at_unix": time.time(), "explicitly_stopped_prior_units": stopped,
        "gpu_release_audit": release, "old_evidence_retained": True,
    })


def bootstrap(path: Path) -> dict:
    require_research_node()
    protocol = validate_protocol(path)
    root = Path(protocol["run"]) / "common"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "COMPLETE.json").is_file():
        complete = read_json(root / "COMPLETE.json")
        if complete["initial_state_sha256"] != file_sha(root / "INITIAL_STATE.json"):
            raise ValueError("Shared warm-start state changed")
        checked_release(root)
        return complete
    for name in ("FAILED.json", "RELEASED.json"):
        previous = root / name
        if previous.exists():
            archive = root / "controller_history" / f"{time.time_ns()}-{name}"
            archive.parent.mkdir(exist_ok=True)
            previous.rename(archive)
    settings = protocol["autonomous"]
    engineer = checkpoint_pair(
        read_json(Path(protocol["common_prefix"]["engineer_training"]) / "optimizer/trained.json"),
        protocol["common_prefix"]["selected_checkpoint_step"],
    )
    solver = checkpoint_pair(
        read_json(Path(settings["solver_training"]) / "optimizer/trained.json"),
        settings["solver_checkpoint_step"],
    )
    if solver["actor"] != protocol["initial_solver"]:
        raise ValueError("Warm start is not the selected, validated Solver checkpoint")
    ensure_model_runtime_metadata(Path(solver["actor"]), Path(settings["solver_runtime_reference"]))
    _, weights = model_digests(Path(solver["actor"]))
    seed = {
        "engineer": engineer["actor"], "engineer_critic": engineer["critic"],
        "engineer_version": 1, "solver": solver["actor"], "solver_critic": solver["critic"],
        "solver_version": f"sao-v4-step79-{weights[:12]}", "solver_weights_sha256": weights,
        "solver_epoch_version": 1, "harness": protocol["no_patch"], "harness_version": 0,
        "harness_sha256": contract_digest(read_json(Path(protocol["no_patch"]))),
        "protocol_sha256": file_sha(path),
    }
    _immutable_json(root / "WARM_START_STATE.json", seed)
    launch = Launcher(root / "supervision")
    try:
        refreshed = refresh_partner(path, root / "WARM_START_STATE.json", root / "partner_w1")
        if not refreshed["engineer_training_ready"]:
            raise ValueError("W1 has no authentic training failures; do not fabricate H2 evidence")
        state = read_json(Path(refreshed["state"]))
        generation = root / "h1_generation"
        proposals = generate_proposals(launch, protocol, state, {"incumbent": engineer}, generation)["incumbent"]
        if Path(protocol["common_prefix"]["proposal_file"]) != generation / "proposals/incumbent.json":
            raise ValueError("Shared H1 proposal path differs from the frozen protocol")
        harnesses = {}
        for row in proposals:
            label = f"proposal{row['sample_index']:02d}"
            patch = root / "patches" / f"{label}.json" if row["valid"] else Path(protocol["no_patch"])
            if row["valid"]:
                _immutable_json(patch, row["patch"])
            harnesses[label] = str(patch)
        config = matrix_config(protocol, state["solver"], state["solver_version"], harnesses, root / "matrix")
        _, output = run_matrix(launch, config, root / "matrix", protocol["max_phase_attempts"])
        initial = finalize_common(protocol, output, initial_state=state)
        publish_selection_binding(protocol, initial)
        complete = {
            "status": "completed", "protocol_sha256": file_sha(path),
            "initial_state_sha256": file_sha(root / "INITIAL_STATE.json"),
            "solver": initial["solver"], "engineer": initial["engineer"],
            "solver_critic": initial["solver_critic"], "engineer_critic": initial["engineer_critic"],
            "selection_accuracy": initial["selection_accuracy"],
            "selection_no_patch_accuracy": initial["selection_no_patch_accuracy"],
            "failure_packets": refreshed["failure_packets"], "report_opened": False,
            "completed_at_unix": time.time(),
        }
        _immutable_json(root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
        raise
    finally:
        launch.close()
        write_json(root / "RELEASED.json", {"released_at": time.time(), **gpu_release_audit()})


def freeze_available(path: Path, arm: str, reason: dict) -> dict:
    protocol = validate_protocol(path)
    root = Path(protocol["run"]) / arm
    frozen = root / "ARM_FROZEN.json"
    if frozen.exists():
        receipt = read_json(frozen)
        if receipt["final_state_sha256"] != file_sha(root / "FINAL_STATE.json"):
            raise ValueError("Completed arm freeze changed")
        checked_selected_state(root / "FINAL_STATE.json", protocol, file_sha(path))
        return receipt
    release = gpu_release_audit()
    if release["foreign_gpu_pids"] or not release["filler_unit_active"]:
        raise RuntimeError("Cannot freeze a budget-stopped arm before its GPU processes exit")
    with (root / "arm_controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = checked_selected_state(
            Path(protocol["run"]) / "common/INITIAL_STATE.json", protocol, file_sha(path),
        )
        phases = []
        for phase in protocol["phase_orders"][arm]:
            native_path = root / phase / "PHASE_COMPLETE.json"
            selection_path = root / f"{phase}_selection/COMPLETE.json"
            state_path = root / f"STATE_AFTER_{phase}.json"
            if not all(item.exists() for item in (native_path, selection_path, state_path)):
                break
            native, selected = read_json(native_path), read_json(selection_path)
            if selected["selected_state_sha256"] != file_sha(state_path):
                raise ValueError("Last validated phase state changed")
            if read_json(root / phase / "input_state.json") != state:
                raise ValueError("Completed phase no longer follows the accepted predecessor")
            state = checked_selected_state(state_path, protocol, file_sha(path))
            phases.append({
                "phase": phase, "native_audit": native["native_audit"],
                "native_completion_sha256": file_sha(native_path),
                "selection_completion_sha256": file_sha(selection_path),
                "selected_state_sha256": selected["selected_state_sha256"],
            })
        _immutable_json(root / "FINAL_STATE.json", state)
        _, weights = model_digests(Path(state["solver"]))
        if state.get("solver_weights_sha256", weights) != weights:
            raise ValueError("Last accepted Solver weights changed")
        result = {
            "status": "frozen", "arm": arm, "protocol_sha256": file_sha(path),
            "final_state": str(root / "FINAL_STATE.json"),
            "final_state_sha256": file_sha(root / "FINAL_STATE.json"),
            "solver_model": state["solver"], "solver_weights_sha256": weights,
            "harness": state["harness"], "harness_sha256": state["harness_sha256"],
            "phase_order": protocol["phase_orders"][arm], "phases": phases,
            "all_phases_completed": len(phases) == len(protocol["phase_orders"][arm]),
            "report_opened": False, "frozen_at_unix": time.time(),
            "stop_reason": reason, "partial_training_checkpoints_not_promoted": True,
        }
        _immutable_json(frozen, result)
        return result


def coordinate(path: Path, arm: str) -> dict:
    require_research_node()
    protocol = validate_protocol(path)
    settings = protocol["autonomous"]
    if socket.gethostname().split(".")[0] != settings["nodes"][arm]:
        raise ValueError("This arm is assigned to the other research node")
    root = Path(protocol["run"])
    arm_root = root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    with (arm_root / "coordinator.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(arm_root / "COORDINATOR_IDENTITY.json", {
            "protocol_sha256": file_sha(path), "source": str(ROOT), "arm": arm,
            "deadline_unix": settings["deadline_unix"],
        })
        completed_path = arm_root / "COORDINATOR_COMPLETE.json"
        if completed_path.exists():
            return read_json(completed_path)
        try:
            for stage, key in (("common", "common_deadline_unix"), ("arm", "training_deadline_unix"),
                               ("report", "deadline_unix")):
                if stage != "common" or arm == "sequential":
                    install_expiry(protocol, arm, stage, settings[key])
            handoff(protocol, arm)
            outcomes = {}
            if arm == "sequential":
                outcomes["common"] = execute_stage(
                    protocol, arm, "common",
                    ["-m", "sao_bridge.alfworld_alternating_campaign", "--protocol", str(path), "--bootstrap"],
                    settings["common_deadline_unix"], [root / "common/COMPLETE.json", root / "common/RELEASED.json"],
                )
                if outcomes["common"]["status"] != "complete":
                    raise RuntimeError(f"Common warm start failed: {outcomes['common']}")
            else:
                while not all((root / "common" / name).is_file() for name in ("COMPLETE.json", "RELEASED.json")):
                    if time.time() >= settings["common_deadline_unix"] + 90:
                        raise TimeoutError("Shared W1/H1 initialization did not complete within its budget")
                    write_json(arm_root / "COORDINATOR_STATUS.json", {
                        "stage": "waiting_for_common_w1_h1", "updated_at_unix": time.time(),
                        "common_deadline_unix": settings["common_deadline_unix"],
                    })
                    time.sleep(10)
            common = read_json(root / "common/COMPLETE.json")
            if common["protocol_sha256"] != file_sha(path) or common["initial_state_sha256"] != file_sha(root / "common/INITIAL_STATE.json"):
                raise ValueError("Shared initialized models or evidence changed")
            checked_release(root / "common")
            outcomes["arm"] = execute_stage(
                protocol, arm, "arm",
                ["-m", "sao_bridge.alfworld_strong_arm", "--protocol", str(path), "--arm", arm],
                settings["training_deadline_unix"], [arm_root / "ARM_FROZEN.json"],
            )
            freeze_available(path, arm, outcomes["arm"])
            outcomes["report"] = execute_stage(
                protocol, arm, "report",
                ["-m", "sao_bridge.alfworld_strong_report", "--protocol", str(path), "--arm", arm, "--wait"],
                settings["deadline_unix"], [arm_root / "final_report/COMPLETE.json", arm_root / "final_report/RELEASED.json"],
            )
            if arm == "sequential":
                reports = [
                    root / name / "final_report" / receipt
                    for name in ("sequential", "ma") for receipt in ("COMPLETE.json", "RELEASED.json")
                ]
                while not all(item.is_file() for item in reports) and time.time() < settings["deadline_unix"] + 90:
                    time.sleep(10)
                if all(item.is_file() for item in reports):
                    compare(path)
                else:
                    write_json(root / "COMPARISON_UNAVAILABLE.json", {
                        "status": "incomplete", "missing_receipts": [str(item) for item in reports if not item.is_file()],
                        "at_unix": time.time(), "reason": "Final report budget expired; no fabricated scores.",
                    })
            result = {
                "status": "completed" if all(item["status"] == "complete" for item in outcomes.values()) else "incomplete",
                "outcomes": outcomes, "finished_at_unix": time.time(),
                "deadline_unix": settings["deadline_unix"], "release": gpu_release_audit(),
            }
            _immutable_json(completed_path, result)
            write_json(arm_root / "COORDINATOR_STATUS.json", result)
            return result
        except BaseException as exc:
            write_json(arm_root / "COORDINATOR_FAILED.json", {
                "error": repr(exc), "failed_at_unix": time.time(), "deadline_unix": settings["deadline_unix"],
            })
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--arm", choices=("sequential", "ma"))
    mode.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()
    try:
        result = bootstrap(args.protocol) if args.bootstrap else coordinate(args.protocol, args.arm)
        print(result, flush=True)
    except (ValueError, PhaseAttemptsExhausted, SelectionAttemptsExhausted, EvaluationAttemptsExhausted) as exc:
        raise SystemExit(78) from exc


if __name__ == "__main__":
    main()
