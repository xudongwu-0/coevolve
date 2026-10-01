"""Resume one frozen strong SAO/MA arm through training, selection and refresh."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import subprocess
import time

from .alfworld_engineer_eval import EvaluationAttemptsExhausted
from .alfworld_strong_phase import PhaseAttemptsExhausted, checked_release, run_phase
from .alfworld_strong_refresh import run as refresh_partner
from .alfworld_strong_select import (
    SelectionAttemptsExhausted, run as select_phase, selection_binding,
)
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import file_sha, read_json, write_json
from .resources import require_research_node


def wait_existing_phase(root: Path, receipt: Path, timeout: float) -> dict:
    unit = read_json(receipt)["unit"]
    if not unit.startswith("coevolve-strong-") or not unit.endswith(".service"):
        raise ValueError("Unexpected external strong phase unit")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        complete = root / "PHASE_COMPLETE.json"
        if complete.exists() and (root / "RELEASED.json").exists():
            checked_release(root)
            return read_json(complete)
        state = subprocess.check_output(
            ["systemctl", "show", unit, "--property=ActiveState", "--value"], text=True,
        ).strip()
        if state not in {"active", "activating", "reloading", "deactivating"}:
            raise RuntimeError(f"External phase {unit} is {state} without complete evidence")
        time.sleep(10)
    raise TimeoutError(f"External phase timed out: {unit}")


def checked_selected_state(path: Path, protocol: dict, expected_protocol_sha: str) -> dict:
    state = read_json(path)
    if state["protocol_sha256"] != expected_protocol_sha:
        raise ValueError("Selected state belongs to a different experiment")
    if contract_digest(read_json(Path(state["harness"]))) != state["harness_sha256"]:
        raise ValueError("Selected harness changed")
    binding_path = Path(state["selection_report"]).with_suffix(".binding.json")
    if read_json(binding_path) != selection_binding(protocol, state):
        raise ValueError("Selected score no longer matches its model/harness/seed identity")
    return state


def run(protocol_path: Path, arm: str) -> dict:
    require_research_node()
    protocol = read_json(protocol_path)
    if arm not in protocol["phase_orders"]:
        raise ValueError("Unknown strong comparison arm")
    root = Path(protocol["run"]) / arm
    root.mkdir(parents=True, exist_ok=True)
    protocol_sha = file_sha(protocol_path)
    initial_path = Path(protocol["run"]) / "common/INITIAL_STATE.json"
    initial = checked_selected_state(initial_path, protocol, protocol_sha)
    with (root / "arm_controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(root / "ARM_IDENTITY.json", {
            "protocol_sha256": protocol_sha, "initial_state_sha256": file_sha(initial_path),
            "arm": arm, "phase_order": protocol["phase_orders"][arm],
            "report_must_remain_closed": True,
        })
        frozen = root / "ARM_FROZEN.json"
        if frozen.exists():
            result = read_json(frozen)
            if result["final_state_sha256"] != file_sha(root / "FINAL_STATE.json"):
                raise ValueError("Frozen final arm state changed")
            checked_selected_state(root / "FINAL_STATE.json", protocol, protocol_sha)
            return result
        if (root / "ARM_FAILED.json").exists():
            archive = root / "arm_failures"
            archive.mkdir(exist_ok=True)
            (root / "ARM_FAILED.json").rename(archive / f"{len(list(archive.glob('*.json'))) + 1:03d}.json")
        started = time.monotonic()
        phases = []
        state = initial
        try:
            for index, phase in enumerate(protocol["phase_orders"][arm]):
                phase_root = root / phase
                state_path = root / f"STATE_BEFORE_{phase}.json"
                if phase == "h2" and Path(state["packet_solver"]).resolve() != Path(state["solver"]).resolve():
                    if arm != "ma":
                        raise ValueError("Sequential arm unexpectedly changed Solver before H2")
                    refresh_input = root / "STATE_BEFORE_PARTNER_REFRESH.json"
                    _immutable_json(refresh_input, state)
                    write_json(root / "ARM_STATUS.json", {"stage": "partner_refresh", "phase": phase})
                    refreshed = refresh_partner(protocol_path, refresh_input, root / "partner_after_w1")
                    if not refreshed["engineer_training_ready"]:
                        raise ValueError(
                            "Current partner has no training failures; cannot fabricate an equal-budget H2 phase"
                        )
                    state = checked_selected_state(Path(refreshed["state"]), protocol, protocol_sha)
                _immutable_json(state_path, state)
                write_json(root / "ARM_STATUS.json", {"stage": "native_phase", "phase": phase})
                external = root / f"launch_receipt_{phase}.json"
                if index == 0 and external.exists():
                    completed = wait_existing_phase(phase_root, external, timeout=32400)
                    if read_json(phase_root / "input_state.json") != state:
                        raise ValueError("Externally started native phase used a different initial state")
                else:
                    completed = run_phase(
                        protocol_path, arm, phase, phase_root, state_path=state_path,
                    )
                write_json(root / "ARM_STATUS.json", {"stage": "phase_selection", "phase": phase})
                selected = select_phase(
                    protocol_path, arm, phase, phase_root, root / f"{phase}_selection",
                )
                state = checked_selected_state(Path(selected["selected_state"]), protocol, protocol_sha)
                _immutable_json(root / f"STATE_AFTER_{phase}.json", state)
                phases.append({
                    "phase": phase, "native_audit": completed["native_audit"],
                    "native_completion_sha256": file_sha(phase_root / "PHASE_COMPLETE.json"),
                    "selection_completion_sha256": file_sha(root / f"{phase}_selection/COMPLETE.json"),
                    "selected_state_sha256": selected["selected_state_sha256"],
                })
            _immutable_json(root / "FINAL_STATE.json", state)
            _, weights = model_digests(Path(state["solver"]))
            if state.get("solver_weights_sha256", weights) != weights:
                raise ValueError("Final selected Solver weights changed before freezing")
            result = {
                "status": "frozen", "arm": arm, "protocol_sha256": protocol_sha,
                "final_state": str(root / "FINAL_STATE.json"),
                "final_state_sha256": file_sha(root / "FINAL_STATE.json"),
                "solver_model": state["solver"], "solver_weights_sha256": weights,
                "harness": state["harness"], "harness_sha256": state["harness_sha256"],
                "phase_order": protocol["phase_orders"][arm], "phases": phases,
                "report_opened": False, "frozen_at_unix": time.time(),
                "controller_segment_elapsed_seconds": time.monotonic() - started,
                "stop_reason": "predeclared strong continuation budget, not proof of convergence",
            }
            _immutable_json(frozen, result)
            write_json(root / "ARM_STATUS.json", {"stage": "frozen", "report_opened": False})
            return result
        except BaseException as exc:
            write_json(root / "ARM_FAILED.json", {
                "error": repr(exc), "failed_at_unix": time.time(), "arm": arm,
            })
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", choices=("sequential", "ma"), required=True)
    args = parser.parse_args()
    try:
        result = run(args.protocol, args.arm)
    except (ValueError, PhaseAttemptsExhausted, SelectionAttemptsExhausted, EvaluationAttemptsExhausted) as exc:
        raise SystemExit(78) from exc
    print(result, flush=True)


if __name__ == "__main__":
    main()
