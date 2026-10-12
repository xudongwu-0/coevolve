"""Attached, bounded execution and evidence-backed reporting for one mechanism-study arm."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from . import alfworld_ma_mechanism_study as study
from .alfworld_failure_cases import write_case_study
from .alfworld_strong_cost import native_generation, reward_work
from .alfworld_strong_phase import checked_release, checkpoint_pair
from .alfworld_train_data import _immutable_json
from .analysis import paired_difference
from .common import ROOT, file_sha, read_json, write_json
from .joint_campaign import share_checkpoints
from .joint_health import metric_rows, metric_value
from .launch import Launcher
from .resources import gpu_release_audit, research_resources_released, scope_environment
from .training_speed_campaign import PYTHON, SSH, unit_state


def require_prerequisites(root: Path, *, native: bool = True) -> dict:
    if os.uname().nodename != "dev-NDH200-026" or os.getuid() != 1008:
        raise RuntimeError("The mechanism coordinator belongs to node4")
    protocol = read_json(root / "PROTOCOL.json")
    if (root / "CANCELLED_BEFORE_TRAINING.json").exists():
        raise ValueError("This experiment was cancelled before training")
    if Path(protocol["source"]).resolve() != ROOT.resolve():
        raise ValueError("The executing mechanism code is outside its frozen source snapshot")
    validation = read_json(root / "CODE_VALIDATION.json")
    if (validation["status"] != "passed" or validation["pins_sha256"] != file_sha(root / "PINS.json")
            or validation["source_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")):
        raise ValueError("Exact prepared study inputs have not passed source-bound validation")
    study.verify_inputs(root)
    for name in protocol["resource_gate"]:
        path = Path(name)
        value = read_json(path)
        if value["status"] != "passed" or value["result_sha256"] != file_sha(path.parent / "RESULT.json"):
            raise ValueError("The earlier requested experiments have not completed and passed their audits")
    if not (root / "PRIOR_RESULTS_REPORTED.json").is_file():
        raise ValueError("Report both earlier experiments and SAO failure cases before ablation execution")
    if native:
        gate = read_json(root / "NATIVE_GATE.json")
        if gate["status"] != "passed" or gate["pins_sha256"] != file_sha(root / "PINS.json"):
            raise ValueError("Separate context and ValueNorm controls need their native full-context gate")
        if protocol.get("study_revision", 1) >= 2:
            cases = gate["real_schedule_cases"]
            if (set(cases) != {"full", "always_joint"}
                    or gate["calibration_environment_episodes"] != 96
                    or gate["memory_gate_inheritance_sha256"] != file_sha(root / "MEMORY_GATE_INHERITANCE.json")):
                raise ValueError("The revised native gate lost its declared calibration evidence")
            for arm, case in cases.items():
                path = Path(case["audit"])
                audit = read_json(path)
                expected = study.arm_budgets(arm, rounds=6)
                if (case["sha256"] != file_sha(path) or audit["status"] != "passed"
                        or audit["shared_environment_episodes"] != 48
                        or any(audit["roles"][role][key] != expected["roles"][role][key]
                               for role in ("editor", "solver")
                               for key in ("actor_steps", "critic_steps", "consumed_trajectories"))):
                    raise ValueError("Executed calibration budgets no longer match the native gate")
    release = gpu_release_audit()
    if not research_resources_released(release):
        raise RuntimeError(f"Node4 is not released for this experiment: {release}")
    return protocol


def require_full_control_review(root: Path) -> None:
    directory = root / "full"
    final = read_json(directory / "FINAL_AUDIT.json")
    audit = read_json(directory / "BASELINE_PATH_AUDIT.json")
    release = read_json(root / "BASELINE_RELEASE.json")
    result_sha = file_sha(directory / "RESULT.json")
    if (final["status"] != "passed" or final["result_sha256"] != result_sha
            or final["report_sha256"] != file_sha(directory / "RESULT.md")
            or audit["status"] != "passed" or audit["full_result_sha256"] != result_sha
            or release["status"] != "audited_and_reported"
            or release["full_result_sha256"] != result_sha
            or release["baseline_audit_sha256"] != file_sha(directory / "BASELINE_PATH_AUDIT.json")):
        raise ValueError("Report and review the exact completed baseline before releasing ablations")


def start_remote(launch: Launcher, root: Path, name: str, module: str, args: list[str], *,
                 native_probe: bool = False) -> dict:
    protocol = read_json(root / "PROTOCOL.json")
    unit = f"alfworld-mechanism-{name}-{protocol.get('unit_suffix', 'single-seed-v1')}"
    if unit_state(unit, remote=True)["LoadState"] != "not-found":
        raise RuntimeError(f"Refusing to replace an existing owned experiment unit: {unit}")
    paths = [str(ROOT / "whale_local"), str(ROOT / "third_party/sao/vendor/AReaL")]
    if native_probe:
        spec = read_json(root / "full/editor_spec.json")
        paths.insert(0, spec["training_kernel_overlay"])
    traced = []
    if protocol.get("signal_tracing"):
        traced = [
            "/usr/bin/strace", "-ttt", "-e", "trace=none",
            "-e", "signal=SIGTERM,SIGINT,SIGHUP,SIGQUIT",
            "-o", str(root / "launches" / f"{name}-signals.log"),
        ]
    command = [
        "sudo", "-n", "systemd-run", "--wait", "--pipe", "--collect", f"--unit={unit}",
        "--uid=1009", "--gid=1008", "--property=UMask=0002", "--property=KillMode=control-group",
        "--property=TimeoutStopSec=90", "--property=RuntimeMaxSec=43200",
        f"--property=WorkingDirectory={ROOT}", "--setenv=TMPDIR=/dev/shm",
        "--setenv=PYTHONDONTWRITEBYTECODE=1", "--setenv=HF_HUB_OFFLINE=1",
        "--setenv=WANDB_MODE=disabled", f"--setenv=PYTHONPATH={':'.join(paths)}",
        *(f"--setenv={key}={value}" for key, value in scope_environment().items()),
        "--setenv=CUDA_VISIBLE_DEVICES=" + ("0,1,2,3" if native_probe else ""),
        *traced, PYTHON, "-u", "-B", "-m", module, *args,
    ]
    receipt = {"unit": unit, "command": command, "remote": True, "at_unix": time.time()}
    _immutable_json(root / "launches" / f"{name}.json", receipt)
    process = launch.start(SSH + [shlex.join(command)], gpus="", name=f"{name}-remote")
    return {**receipt, "process": process}


def stop_remote(record: dict) -> None:
    state = unit_state(record["unit"], remote=True)
    if state["LoadState"] != "not-found" and state["ActiveState"] in ("active", "activating", "deactivating"):
        subprocess.run(SSH + [shlex.join(["sudo", "-n", "systemctl", "stop", record["unit"]])],
                       check=True, timeout=120)


def wait_processes(launch: Launcher, processes: list, output: Path, *, phase: str, timeout: float) -> None:
    started = time.monotonic()
    while True:
        launch.check()
        states = [{"pid": process.pid, "returncode": process.poll()} for process in processes]
        if any(record["returncode"] not in (None, 0) for record in states):
            write_json(output / "PROCESS_EXIT_FAILURE.json", {
                "phase": phase, "processes": states, "at_unix": time.time(),
            })
            raise RuntimeError(f"{phase} exited unsuccessfully: {states}")
        progress = {}
        for role in ("editor", "solver"):
            for path in (output / role / "optimizer/metrics.jsonl",
                         output / role / "training/optimizer/metrics.jsonl"):
                if path.exists():
                    rows = metric_rows(path)
                    progress[role] = {
                        "committed_steps": len(rows),
                        "last_committed_unix": rows[-1]["recorded_at_unix"] if rows else None,
                    }
        write_json(output / "MONITOR.json", {
            "phase": phase, "processes": states, "progress": progress, "at_unix": time.time(),
        })
        if all(record["returncode"] == 0 for record in states):
            return
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"{phase} exceeded its explicit wall safety limit")
        time.sleep(15)


def native_gates(root: Path) -> dict:
    require_prerequisites(root, native=False)
    if (root / "NATIVE_GATE_STARTED.json").exists():
        raise ValueError("Native gates cannot be silently rerun after partial execution")
    _immutable_json(root / "NATIVE_GATE_STARTED.json", {"pid": os.getpid(), "at_unix": time.time()})
    launch = Launcher(root / "native-gate-supervision")
    remote = None
    receipts = {}
    started = time.monotonic()
    try:
        for arm in ("full", "without_context", "without_value_norm"):
            base = root / "native-gates" / arm
            common = [
                "--actor-rounds", "2", "--critic-updates-per-round", "2", "--critic-warmup-rounds", "1",
                "--reference-storage-dtype", "bfloat16", "--component-ablation", arm,
            ]
            remote = start_remote(
                launch, root, f"gate-{arm}-h0", "sao_bridge.joint_memory_check",
                ["--output", str(base / "h0"), "--model-role", "h0", *common], native_probe=True,
            )
            overlay = read_json(root / "full/solver_spec.json")["training_kernel_overlay"]
            local = launch.module(
                "sao_bridge.joint_memory_check",
                ["--output", str(base / "w0"), "--model-role", "w0", *common],
                gpus="0,1,2,3", name=f"gate-{arm}-w0",
                env_overrides={"PYTHONPATH": overlay + ":" + launch.env["PYTHONPATH"]},
            )
            wait_processes(launch, [local, remote["process"]], base,
                           phase=f"native-{arm}", timeout=5400)
            for role in ("h0", "w0"):
                path = base / role / "result.json"
                if read_json(path)["status"] != "passed":
                    raise ValueError("A full-context native component gate did not pass")
                receipts[f"{arm}/{role}"] = {
                    "result": str(path), "sha256": file_sha(path),
                    "audit_sha256": file_sha(base / role / "audit-summary.json"),
                }
    finally:
        if remote is not None:
            stop_remote(remote)
        launch.close()
        write_json(root / "NATIVE_GATE_RELEASED.json", gpu_release_audit())
    if not research_resources_released(read_json(root / "NATIVE_GATE_RELEASED.json")):
        raise RuntimeError("Native validation did not release node4")
    result = {
        "status": "passed", "pins_sha256": file_sha(root / "PINS.json"), "cases": receipts,
        "research_training_trajectories": 0, "synthetic_actor_updates_per_role_case": 2,
        "synthetic_critic_updates_per_role_case": 6,
        "elapsed_seconds": time.monotonic() - started, "at_unix": time.time(),
    }
    _immutable_json(root / "NATIVE_GATE.json", result)
    return result


def train_joint(root: Path, arm: str, *, rounds: int = 144) -> dict:
    directory = root / arm
    launch = Launcher(directory / "coordinator-supervision")
    remote = None
    try:
        remote = start_remote(launch, root, f"{arm}-editor", "sao_bridge.alfworld_ma_mechanism_study", [
            "train-role", "--root", str(root), "--arm", arm, "--role", "editor",
        ])
        local = launch.module("sao_bridge.alfworld_ma_mechanism_study", [
            "train-role", "--root", root, "--arm", arm, "--role", "solver",
        ], gpus="", name=f"{arm}-solver")
        wait_processes(launch, [local, remote["process"]], directory,
                       phase="native_training", timeout=9 * 3600)
    finally:
        if remote is not None:
            stop_remote(remote)
        launch.close()
        write_json(directory / "COORDINATOR_RELEASED.json", gpu_release_audit())
    return study.audit_joint_arm(root, arm, rounds=rounds)


def staged_native_audit(root: Path) -> dict:
    directory = root / "without_team_learning"
    protocol = read_json(root / "PROTOCOL.json")
    roles = {}
    for role, expected_steps in (("editor", 96), ("solver", 144)):
        config = read_json(directory / f"{role}_config.json")
        output = Path(config["run"])
        checked_release(output)
        receipt = read_json(output / "COMPLETE.json")
        trained = read_json(output / "optimizer/trained.json")
        rows = metric_rows(output / "optimizer/metrics.jsonl")
        if (receipt["status"] != "completed" or len(rows) != expected_steps
                or [row["global_step"] for row in rows] != list(range(expected_steps))
                or trained["actor_steps"] != protocol["per_role_actor_updates"][role]
                or trained["critic_steps"] != protocol["per_role_critic_updates"][role]
                or trained["central_critic"] or trained.get("value_normalization")):
            raise ValueError("Staged SAO did not complete its declared separate-role update budgets")
        gradients = []
        for index, row in enumerate(rows):
            if any(isinstance(value, (int, float)) and not math.isfinite(value)
                   for value in row["metrics"].values()):
                raise FloatingPointError("Staged SAO emitted a nonfinite optimizer metric")
            if index >= 3:
                gradients.append(metric_value(row["metrics"], "ppo_actor/update/grad_norm"))
        if any(value <= 0 for value in gradients):
            raise ValueError("Staged SAO omitted a finite nonzero scheduled Actor update")
        spec = read_json(output / "train_spec.json")
        if spec["kl_ctl"] != .02 or spec["critic_updates_per_step"] != 2 or spec.get("joint_root"):
            raise ValueError("Staged method changed matched regularization or reused joint updates")
        roles[role] = {
            "native_steps": expected_steps, "actor_updates": trained["actor_steps"],
            "critic_updates": trained["critic_steps"], "consumed_trajectories": expected_steps * 8,
            "generation": native_generation(output / "optimizer"), "elapsed_seconds": receipt["elapsed_seconds"],
        }
    result = {
        "status": "passed", "arm": "without_team_learning", "roles": roles,
        "engineer_reward_environment_work": reward_work(directory / "editor/training"),
        "solver_consumed_environment_episodes": 1152,
        "equal_environment_budget_claimed": False,
        "same_per_task_solver_exposures": True, "same_active_engineer_messages": True,
        "method_level_not_one_flag_causal_ablation": True, "at_unix": time.time(),
    }
    _immutable_json(directory / "TRAINING_AUDIT.json", result)
    return result


def staged_editor_role(root: Path) -> dict:
    if os.uname().nodename != "dev-NDH200-025" or (os.getuid(), os.getgid()) != (1009, 1008):
        raise RuntimeError("The separate-node staged Engineer belongs to its declared node3 UID/GID")
    directory = root / "without_team_learning"
    config = read_json(directory / "editor_config.json")
    template = read_json(directory / "editor_template.json")
    if {key: value for key, value in config.items() if key != "external_solver"} != template:
        raise ValueError("Staged Engineer settings changed after the common study freeze")
    subprocess.run(
        [sys.executable, "-B", "-m", "sao_bridge.alfworld_h_train",
         "--config", str(directory / "editor_config.json")], check=True,
    )
    output = Path(config["run"]) / "optimizer"
    trained = read_json(output / "trained.json")
    history = {str(step): checkpoint_pair(trained, step) for step in (47, 95)}
    share_checkpoints(output, {"checkpoint_history": history})
    receipt = {"status": "passed", "checkpoint_history": history, "at_unix": time.time()}
    _immutable_json(directory / "EDITOR_SHARED.json", receipt)
    return receipt


def train_staged(root: Path) -> tuple[dict, dict]:
    directory = root / "without_team_learning"
    state = read_json(root / "INITIAL_STATE.json")
    launch = Launcher(directory / "coordinator-supervision")
    remote = None
    try:
        with launch.vllm(
            state["solver"], port=19501, gpus="0,1,2,3,4,5,6,7", name="frozen-w1",
            data_parallel=8, max_model_len=32768, tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton", language_model_only=True, max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as urls:
            tunnel_command = [*SSH, "-N", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
                              "-o", "ServerAliveCountMax=3"]
            for index in range(8):
                port = 19501 + index * 10
                tunnel_command += ["-R", f"127.0.0.1:{port}:127.0.0.1:{port}"]
            tunnel = launch.start(tunnel_command, gpus="", name="frozen-w1-reverse-tunnel")
            for _ in range(3):
                if tunnel.poll() is not None:
                    raise RuntimeError("Frozen Solver SSH tunnel exited during startup")
                time.sleep(1)
            binding_path = directory / "FROZEN_SOLVER_READY.json"
            _immutable_json(binding_path, {
                "status": "ready", "consumer_host": "dev-NDH200-025", "provider_host": "dev-NDH200-026",
                "model": state["solver"], "solver_version": state["solver_version"],
                "solver_weights_sha256": state["solver_weights_sha256"],
                "urls": urls.split(","), "tunnel_pid": tunnel.pid, "at_unix": time.time(),
                "provider_lifetime": "Owned by the attached campaign through the entire Engineer phase",
            })
            config = {
                **read_json(directory / "editor_template.json"),
                "external_solver": {"binding": str(binding_path), "binding_sha256": file_sha(binding_path)},
            }
            _immutable_json(directory / "editor_config.json", config)
            remote = start_remote(launch, root, "staged-editor", "sao_bridge.alfworld_ma_mechanism_campaign",
                                  ["staged-editor-role", "--root", str(root)])
            wait_processes(launch, [remote["process"]], directory, phase="staged_engineer_training",
                           timeout=12 * 3600)
            launch.stop(tunnel)
        h_trained = read_json(directory / "editor/training/optimizer/trained.json")
        h_pair = checkpoint_pair(h_trained, 95)
        selection = study.select_at_pair(launch, root, {
            "editor": h_pair, "solver": {"actor": state["solver"], "critic": state["solver_critic"]},
        }, directory / "fixed_harness_selection", state["harness"])
        rows = read_json(root / "staged_solver_data.json")
        assignments_path = directory / "solver/assignments.json"
        _immutable_json(assignments_path, study.staged_assignments(rows, selection["harness"]))
        solver_config = {
            **read_json(directory / "solver_template.json"), "harness": selection["harness"],
            "assignments": str(assignments_path),
        }
        _immutable_json(directory / "solver_config.json", solver_config)
        local = launch.module("sao_bridge.alfworld_w_train",
                              ["--config", directory / "solver_config.json"],
                              gpus="", name="staged-solver")
        wait_processes(launch, [local], directory, phase="staged_solver_training", timeout=12 * 3600)
    finally:
        if remote is not None:
            stop_remote(remote)
        launch.close()
        write_json(directory / "COORDINATOR_RELEASED.json", gpu_release_audit())
    return staged_native_audit(root), selection


def write_arm_report(root: Path, arm: str, result: dict) -> None:
    directory = root / arm
    lines = [
        f"# Single-seed mechanism study: {arm}", "",
        "This is a new fixed-endpoint development experiment, not a rerun selected to recover97%.",
        "One training seed; no training-seed robustness claim.", "",
        "| Saved endpoint | Selected Harness + Solver | Solver without Harness | New valid proposals |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for rounds, endpoint in result["endpoints"].items():
        score = endpoint["scores"]
        lines.append(
            f"| {rounds} | {score['selected']['successes']}/200 ({100 * score['selected']['accuracy']:.1f}%) | "
            f"{score['no_patch']['successes']}/200 ({100 * score['no_patch']['accuracy']:.1f}%) | "
            f"{endpoint['selection']['valid_proposals']}/8 |")
    lines += [
        "", "**Primary endpoint:144, fixed before training.**",
        "The selected Harness is frozen on selection48 before its main200 evaluation.",
        "A carried incumbent is not described as a new learned Harness.", "",
        "Native training and actual work: [training audit](./TRAINING_AUDIT.json).",
        "Machine-readable endpoint reports and bindings: [result](./RESULT.json).", "",
    ]
    if arm == "always_joint":
        lines += [
            "Only the Engineer update mask changes: both roles learn on every round.",
            "Third-round peer weights remain unit; routing, input order and critic context stay unchanged.",
            "Generated trajectories/environment episodes match full MA; consumed role trajectories",
            "increase1920 to2304 (+20%). This is not equal-update or equal-compute evidence.", "",
        ]
    if "paired_against_full" in result:
        comparison = result["paired_against_full"]
        lines += [
            "The paired statistics compare full MA (control) with this ablation (candidate):",
            f"`{json_summary(comparison)}`", "",
            "Question-level uncertainty is not independent training-seed uncertainty.",
            "Representative same-question cases, including reverse counts:",
            "[case study](./failure-cases/CASES.md).", "",
        ]
    lines.extend(read_json(root / "PROTOCOL.json")["limitations"])
    (directory / "RESULT.md").write_text("\n".join(lines) + "\n")


def json_summary(value: dict) -> str:
    import json
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def run_arm(root: Path, arm: str) -> dict:
    protocol = require_prerequisites(root)
    if arm not in study.ARMS or arm not in protocol["arms"]:
        raise ValueError("Unknown mechanism-study arm")
    directory = root / arm
    if (directory / "STARTED.json").exists():
        raise ValueError("A started arm needs explicit failure recovery, not a duplicate restart")
    if arm != "full" and not (root / "full/FINAL_AUDIT.json").exists():
        raise ValueError("Finish and report the shared full-MA control before an ablation")
    if arm != "full" and protocol.get("study_revision", 1) >= 2:
        require_full_control_review(root)
    _immutable_json(directory / "STARTED.json", {"pid": os.getpid(), "at_unix": time.time()})
    started = time.monotonic()
    training = train_joint(root, arm)
    records = {role: read_json(directory / role / "COMPLETE.json") for role in ("editor", "solver")}
    pairs = {str(rounds): {role: records[role]["checkpoint_history"][str(rounds - 1)] for role in records}
             for rounds in study.ROUNDS}
    incumbent = protocol["incumbent_harness"]
    endpoints = {}
    for rounds in study.ROUNDS:
        endpoints[str(rounds)] = study.evaluate_pair(
            root, arm, rounds, pairs[str(rounds)], incumbent=incumbent,
        )
        incumbent = endpoints[str(rounds)]["selection"]["harness"]
    result = {
        "status": "verified", "arm": arm, "training": training, "endpoints": endpoints,
        "elapsed_seconds": time.monotonic() - started, "training_seeds": protocol["training_seeds"],
        "primary_endpoint": 144, "at_unix": time.time(),
    }
    if arm != "full":
        full = read_json(root / "full/RESULT.json")["endpoints"]["144"]["scores"]["selected"]
        control = endpoints["144"]["scores"]["selected"]
        full_report, ablation_report = read_json(Path(full["report"])), read_json(Path(control["report"]))
        result["paired_against_full"] = paired_difference(full_report, ablation_report, seed=20261008)
        result["failure_case_population"] = write_case_study(
            directory / "failure-cases", Path(control["report"]), Path(full["report"]),
            read_json(Path(protocol["core200_protocol"])), title=f"{arm} failures and matched full-MA outcomes",
        )
    _immutable_json(directory / "RESULT.json", result)
    write_arm_report(root, arm, result)
    _immutable_json(directory / "FINAL_AUDIT.json", {
        "status": "passed", "result_sha256": file_sha(directory / "RESULT.json"),
        "report_sha256": file_sha(directory / "RESULT.md"), "at_unix": time.time(),
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("native-gates", "run-arm", "staged-editor-role"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--arm", choices=study.ARMS)
    args = parser.parse_args()
    root = args.root.resolve()
    os.umask(0o002)
    try:
        if args.action == "native-gates":
            result = native_gates(root)
        elif args.action == "staged-editor-role":
            result = staged_editor_role(root)
        else:
            result = run_arm(root, args.arm)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError, ExceptionGroup) as exc:
        output = root / args.arm if args.arm else root
        write_json(output / "CAMPAIGN_FAILED.json", {"error": repr(exc), "at_unix": time.time()})
        raise
    print({"status": result["status"], "arm": args.arm, "action": args.action}, flush=True)


if __name__ == "__main__":
    main()
