"""CPU-only two-host checkpoint publication with synthetic native payloads."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
from types import SimpleNamespace


SOURCE = Path("/mnt/3fs/xudong/coevolve/runs/source-snapshots/alfworld-ma-mechanisms-rolling12-v3")
OUTPUT = Path(__file__).resolve().parent / "two-node-cpu"
sys.path.insert(0, str(SOURCE / "whale_local"))
sys.path.insert(0, str(SOURCE / "third_party/sao/vendor/AReaL"))

from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.joint_recovery import PairedRecovery, latest_checkpoint
from sao_bridge.joint_schedule import update_counts


def prepare() -> None:
    OUTPUT.mkdir()
    OUTPUT.chmod(0o2775)
    loader = importlib.util.spec_from_file_location("checkpoint_fixtures", SOURCE / "tests/test_joint_recovery.py")
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    for completed in (12, 24):
        module.paired_payload(OUTPUT / f"template-{completed}", completed)
    execution = OUTPUT / "execution"
    execution.mkdir()
    execution.chmod(0o2775)
    for role in ("editor", "solver"):
        spec = module.make_spec(execution, role)
        spec["training_deadline_unix"] = time.time() + 300
        write_json(OUTPUT / f"{role}-spec.json", spec)


def run_role(role: str) -> None:
    spec = read_json(OUTPUT / f"{role}-spec.json")
    run = Path(spec["run"])
    run.mkdir(parents=True)
    for completed in (12, 24):
        counts = update_counts(spec, completed)
        trainer = SimpleNamespace(joint_schedule_state={
            "round": completed, "actor_steps": counts["actor_steps"], "critic_steps": counts["critic_steps"],
        })
        template = OUTPUT / f"template-{completed}/recovery/round-{completed:04d}-attempt-0001" / role
        for name in ("actor", "critic"):
            source = template / name
            setattr(trainer, name, SimpleNamespace(
                config=SimpleNamespace(optimizer=SimpleNamespace(
                    lr_scheduler_type="constant", warmup_steps_proportion=0,
                )),
                save=lambda *, meta, source=source: shutil.copytree(source, meta.path),
            ))
        for name in ("saver", "evaluator", "stats_logger"):
            setattr(trainer, name, SimpleNamespace(state_dict=lambda: {}))
        (run / "metrics.jsonl").write_text("".join(
            json.dumps({"global_step": step, "metrics": {}}) + "\n" for step in range(completed)
        ))
        PairedRecovery(trainer, spec).after_commit(completed)
        checkpoint = latest_checkpoint(Path(spec["rolling_recovery"]["root"]))
        if checkpoint is None or read_json(checkpoint)["completed_rounds"] != completed:
            raise ValueError("A role acknowledged an unpublished checkpoint")
        write_json(OUTPUT / f"{role}-ack-{completed:04d}.json", {
            "role": role, "completed_rounds": completed, "checkpoint_sha256": file_sha(checkpoint),
            "hostname": socket.gethostname(), "pid": os.getpid(), "uid": os.getuid(), "gid": os.getgid(),
            "source_manifest_sha256": file_sha(SOURCE / "SOURCE_MANIFEST.json"), "at_unix": time.time(),
        })
        while not all((OUTPUT / f"{peer}-ack-{completed:04d}.json").is_file()
                      for peer in ("editor", "solver")):
            if time.time() >= spec["training_deadline_unix"]:
                raise TimeoutError("The other CPU probe role did not acknowledge its published pair")
            time.sleep(.1)
        print(f"{role} acknowledged published round {completed}", flush=True)


def verify() -> None:
    acknowledgements = {
        str(completed): {
            role: read_json(OUTPUT / f"{role}-ack-{completed:04d}.json")
            for role in ("editor", "solver")
        } for completed in (12, 24)
    }
    for records in acknowledgements.values():
        if (
            records["editor"]["hostname"] != "dev-NDH200-025"
            or records["solver"]["hostname"] != "dev-NDH200-026"
            or records["editor"]["checkpoint_sha256"] != records["solver"]["checkpoint_sha256"]
        ):
            raise ValueError("The two real nodes did not acknowledge the same checkpoint")
    recovery = OUTPUT / "execution/recovery"
    checkpoint = latest_checkpoint(recovery)
    if checkpoint is None or read_json(checkpoint)["completed_rounds"] != 24:
        raise ValueError("The final CPU probe checkpoint was not committed")
    old = recovery / "round-0012-attempt-0001"
    retirement = read_json(recovery / "retention/round-0012-attempt-0001.json")
    if old.exists() or retirement["status"] != "deleted":
        raise ValueError("The replaced probe checkpoint was not retired")
    result = {
        "status": "passed", "cpu_only": True, "synthetic_checkpoint_payloads": True,
        "not_a_gpu_training_or_performance_result": True,
        "actual_two_host_publication_and_retention_verified": True,
        "acknowledgements": acknowledgements, "retirement": retirement,
        "checkpoint_sha256": file_sha(checkpoint),
        "source_manifest_sha256": file_sha(SOURCE / "SOURCE_MANIFEST.json"),
        "probe_script_sha256": file_sha(Path(__file__)), "at_unix": time.time(),
    }
    write_json(OUTPUT / "RESULT.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "editor", "solver", "verify"))
    args = parser.parse_args()
    os.umask(0o002)
    if args.action == "prepare":
        prepare()
    elif args.action == "verify":
        verify()
    else:
        run_role(args.action)
