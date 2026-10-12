"""One bounded continuation after the verified cross-node checkpoint fix."""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from sao_bridge import alfworld_ma_mechanism_campaign as campaign
from sao_bridge import alfworld_ma_mechanism_program as program
from sao_bridge import alfworld_ma_rolling_recovery as rolling
from sao_bridge.alfworld_train_data import _immutable_json
from sao_bridge.common import ROOT, file_sha, read_json, write_json
from sao_bridge.joint_recovery import latest_checkpoint


SCRIPT = Path(__file__).resolve()
RECOVERY = SCRIPT.parent
RUN = SCRIPT.parents[2]
ARM = "without_peer"
BASE_DRIVER = RUN / "resume_after_two_hours.py"
loader = importlib.util.spec_from_file_location("original_two_hour_policy", BASE_DRIVER)
if loader is None or loader.loader is None:
    raise RuntimeError("The original two-hour policy driver cannot be loaded")
policy_driver = importlib.util.module_from_spec(loader)
loader.loader.exec_module(policy_driver)


def verified_exit_time(trace: str) -> float:
    matches = re.findall(r"^(\d+\.\d+) \+\+\+ exited with 1 \+\+\+\s*$", trace, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError("Recovery requires one recorded nonzero controller exit")
    return float(matches[0])


def verify_source(prepared: dict) -> None:
    source = read_json(ROOT / "SOURCE_MANIFEST.json")
    base = Path(prepared["original_source"])
    old = read_json(base / "SOURCE_MANIFEST.json")
    if (
        prepared["execution_source"] != str(ROOT)
        or prepared["execution_manifest_sha256"] != file_sha(ROOT / "SOURCE_MANIFEST.json")
        or prepared["original_manifest_sha256"] != file_sha(base / "SOURCE_MANIFEST.json")
        or source["base_snapshot"] != str(base)
        or source["base_manifest_sha256"] != prepared["original_manifest_sha256"]
        or set(source["runtime_files"]) != set(old["runtime_files"])
    ):
        raise ValueError("The checkpoint-only execution amendment changed its source binding")
    changed = []
    for name, digest in source["runtime_files"].items():
        if file_sha(ROOT / name) != digest or file_sha(base / name) != old["runtime_files"][name]:
            raise ValueError(f"Frozen native runtime changed: {name}")
        if digest != old["runtime_files"][name]:
            changed.append(name)
    if changed != ["whale_local/sao_bridge/joint_recovery.py"]:
        raise ValueError("Checkpoint recovery cannot change the learner or other runtime files")


def verify_bindings(prepared: dict) -> None:
    policy = policy_driver.verify_authority(RUN, prepared["user_authority_sha256"])
    if (RECOVERY / "CANCELLED.json").exists():
        raise InterruptedError("Checkpoint-error recovery was cancelled")
    protocol = read_json(RUN / "PROTOCOL.json")
    if (
        prepared["status"] != "prepared" or prepared["driver_sha256"] != file_sha(SCRIPT)
        or prepared["base_driver_sha256"] != file_sha(BASE_DRIVER)
        or prepared["original_source"] != protocol["source"]
        or prepared["deadline_unix"] != protocol["original_absolute_deadline_unix"]
        or prepared["arm"] != ARM or prepared["resume_round"] != 24
        or prepared["not_before_unix"] != policy_driver.earliest_start(
            policy, verified_exit_time((RUN / "WITHOUT_PEER_SIGNALS.log").read_text()),
        )
    ):
        raise ValueError("The bounded checkpoint recovery changed its authority or time window")
    verify_source(prepared)
    for path, digest in prepared["pins"].items():
        if file_sha(Path(path)) != digest:
            raise ValueError(f"Checkpoint recovery evidence changed: {path}")
    for name in ("CPU_TESTS.log", "DRIVER_TESTS.log"):
        if str(RECOVERY / name) not in prepared["pins"]:
            raise ValueError("Recovery must retain both checkpoint and driver validation")
    probe = read_json(RECOVERY / "two-node-cpu/RESULT.json")
    if (
        probe["status"] != "passed" or not probe["actual_two_host_publication_and_retention_verified"]
        or probe["source_manifest_sha256"] != prepared["execution_manifest_sha256"]
    ):
        raise ValueError("The exact checkpoint publisher lacks a real two-host CPU check")
    checkpoint = latest_checkpoint(RUN / ARM / "recovery")
    if (
        checkpoint is None or str(checkpoint) != prepared["checkpoint"]
        or file_sha(checkpoint) != prepared["checkpoint_sha256"]
        or read_json(checkpoint)["completed_rounds"] != 24
    ):
        raise ValueError("The reviewed round24 checkpoint changed before continuation")
    failure = read_json(RUN / ARM / "attempts/attempt-0001/FAILED.json")
    if failure["status"] != "blocked_error" or failure["retry_after_idle_allowed"]:
        raise ValueError("Do not relabel the original code error as authorized termination recovery")
    expected = RUN / ARM / "attempts/attempt-0002"
    if prepared["next_attempt"] != str(expected) or expected.exists():
        raise ValueError("This one-shot amendment cannot replay an existing second attempt")
    if any((RUN / ARM / name).exists() for name in ("RESULT.json", "TRAINING_AUDIT.json", "round0096", "round0144")):
        raise ValueError("Do not replace completed or scored ablation work")
    campaign.require_full_control_review(RUN)


def original_prerequisites(prepared: dict) -> dict:
    base = Path(prepared["original_source"])
    code = """
import json,sys
from pathlib import Path
from sao_bridge.alfworld_ma_mechanism_campaign import require_prerequisites,require_full_control_review
from sao_bridge.common import ROOT,read_json,file_sha
root=Path(sys.argv[1])
require_prerequisites(root)
require_full_control_review(root)
gate=read_json(root/'ROLLING_NATIVE_GATE.json')
assert gate['status']=='passed' and not gate['formal_weights_reused']
assert gate['source_manifest_sha256']==file_sha(ROOT/'SOURCE_MANIFEST.json')
assert gate['pins_sha256']==file_sha(root/'PINS.json')
print(json.dumps({'status':'passed','original_source_manifest_sha256':file_sha(ROOT/'SOURCE_MANIFEST.json'),
                  'native_gate_sha256':file_sha(root/'ROLLING_NATIVE_GATE.json'),
                  'original_pins_sha256':file_sha(root/'PINS.json')}))
"""
    environment = {
        **os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": f"{base / 'whale_local'}:{base / 'third_party/sao/vendor/AReaL'}",
    }
    process = subprocess.run(
        ["/dev/shm/sao-venv-ppo/bin/python", "-B", "-c", code, str(RUN)],
        env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900,
    )
    (RECOVERY / "ORIGINAL_PREREQUISITES.log").write_text(process.stdout)
    if process.returncode:
        raise RuntimeError(f"Original native prerequisites failed ({process.returncode}); see ORIGINAL_PREREQUISITES.log")
    result = json.loads(process.stdout)
    if result["status"] != "passed" or result["original_source_manifest_sha256"] != prepared["original_manifest_sha256"]:
        raise ValueError("Original prerequisite verification used a different native source")
    return result


def execute(prepared: dict) -> dict:
    verify_bindings(prepared)
    write_json(RECOVERY / "MONITOR.json", {
        "status": "waiting_two_hour_cooldown_after_fixed_code_error", "pid": os.getpid(),
        "not_before_unix": prepared["not_before_unix"], "deadline_unix": prepared["deadline_unix"],
        "gpu_work_started": False, "at_unix": time.time(),
    })
    policy_driver.wait_until(
        RUN, prepared["not_before_unix"], deadline=prepared["deadline_unix"],
        authority_sha256=prepared["user_authority_sha256"],
    )
    program.wait_for_resources(
        RUN, "checkpoint-fix-resume", interval=60, stable_checks=1,
        deadline=time.monotonic() + prepared["deadline_unix"] - time.time(),
        probe=rolling.admission_snapshot,
    )
    verify_bindings(prepared)
    prerequisites = original_prerequisites(prepared)
    verify_bindings(prepared)
    if not prepared["not_before_unix"] <= time.time() < prepared["deadline_unix"]:
        raise TimeoutError("Checkpoint-error continuation fell outside its authorized window")
    write_json(RECOVERY / "ORIGINAL_PREREQUISITES.json", prerequisites)
    _immutable_json(RECOVERY / "EXECUTION_AMENDMENT.json", {
        "status": "validated_checkpoint_only_amendment", "prepared_sha256": file_sha(RECOVERY / "PREPARED.json"),
        "base_scientific_protocol_sha256": file_sha(RUN / "PROTOCOL.json"),
        "execution_source": str(ROOT), "execution_manifest_sha256": prepared["execution_manifest_sha256"],
        "original_native_gate": prerequisites,
        "inherited_native_scope": "Unchanged native payload, Adam/RNG/ValueNorm/cursor restore and learner; "
                                  "single-publisher coordination separately CPU-tested on both actual nodes.",
        "not_a_new_gpu_canary_result": True, "attempt": prepared["next_attempt"],
        "original_error_and_failure_latches_preserved": True, "at_unix": time.time(),
    })
    write_json(RECOVERY / "MONITOR.json", {
        "status": "starting_native_continuation", "resume_round": 24, "pid": os.getpid(),
        "execution_source": str(ROOT), "at_unix": time.time(),
    })
    training = rolling.run_attempt(RUN, ARM)
    if training["attempt"] != prepared["next_attempt"]:
        raise ValueError("Native continuation did not use its declared next attempt")
    result = rolling.finish_arm(RUN, ARM, training)
    program.write_master_report(RUN)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    os.umask(0o002)
    prepared = read_json(RECOVERY / "PREPARED.json")
    verify_bindings(prepared)
    if args.check_only:
        print(json.dumps({"status": "passed", "gpu_work_started": False}))
        return
    remaining = prepared["deadline_unix"] - time.time()
    if remaining <= 0:
        raise TimeoutError("The original study deadline has expired")
    with (RUN / "TWO_HOUR_CONTROL.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(RECOVERY / "STARTED.json", {
            "pid": os.getpid(), "attached": True, "prepared_sha256": file_sha(RECOVERY / "PREPARED.json"),
            "not_before_unix": prepared["not_before_unix"], "deadline_unix": prepared["deadline_unix"],
            "at_unix": time.time(),
        })

        def expired(signum, frame):
            raise TimeoutError("The unchanged original study deadline expired")

        signal.signal(signal.SIGALRM, expired)
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            result = execute(prepared)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, ExceptionGroup) as exc:
            write_json(RECOVERY / "FAILED.json", {
                "status": "incomplete", "error": repr(exc), "at_unix": time.time(),
                "original_failure_preserved": True, "automatic_retry": False,
            })
            raise
        except KeyboardInterrupt:
            write_json(RECOVERY / "STOPPED.json", {"status": "interrupted", "at_unix": time.time()})
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        _immutable_json(RECOVERY / "COMPLETE.json", {"status": result["status"], "at_unix": time.time()})


if __name__ == "__main__":
    main()
