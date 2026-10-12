"""Persistent, fail-closed cold-start-SFT then initial online-GRPO pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from sao_bridge.common import file_sha, write_json
from sao_bridge.launch import Launcher


def wait_for_silver(directory: Path, unit: str, timeout: float, *, require_usable: bool = True) -> dict:
    started = time.monotonic()
    receipt = directory / "COMPLETE.json"
    while not receipt.is_file():
        state = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "--property=ActiveState", "--value"],
            text=True,
        ).strip()
        if state not in {"active", "activating"}:
            raise RuntimeError(f"Silver collection ended without a completion receipt: {unit} ({state})")
        if time.monotonic() - started > timeout:
            raise TimeoutError("Silver collection exceeded the configured waiting budget")
        time.sleep(10)
    complete = json.loads(receipt.read_text())
    if complete.get("dataset_sha256") != file_sha(directory / "sft.jsonl"):
        raise ValueError("Silver dataset bytes do not match their completion receipt")
    if require_usable and (not complete.get("usable_for_sft") or complete.get("examples", 0) < 256):
        raise RuntimeError(
            f"Not enough clean positive examples for cold-start SFT: {complete.get('examples')}; "
            "collect additional disjoint training packets, do not relax the outcome gate"
        )
    return complete


def merge_clean_silver(sources: list[Path], output: Path) -> dict:
    identities = []
    rows = []
    audit = []
    seen = set()
    shared = None
    for source in sources:
        receipt = json.loads((source / "COMPLETE.json").read_text())
        data = source / "sft.jsonl"
        if receipt["dataset_sha256"] != file_sha(data):
            raise ValueError("Cannot merge a modified silver dataset")
        identity = receipt["input_identity"]
        current = {key: identity[key] for key in (
            "data_sha256", "engineer_manifest_sha256", "reward_config_sha256",
            "response_protocol", "temperature", "top_p", "max_response_tokens",
            "max_prompt_tokens",
        )}
        if shared is not None and current != shared:
            raise ValueError("Silver sources have different frozen experiment identities")
        shared = current
        identities.append(identity)
        source_rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
        if len(source_rows) != receipt["examples"]:
            raise ValueError("Silver receipt example count differs from its dataset")
        for row in source_rows:
            provenance = row["provenance"]
            identifier = provenance["generation_id"]
            if not isinstance(identifier, str) or not re.fullmatch(
                r"packet-[0-9]{6}-sample-[0-9]{2}", identifier
            ):
                raise ValueError("Unsafe silver generation identifier")
            sample = json.loads((source / "samples" / f"{identifier}.json").read_text())
            if (
                sample.get("clean_execution") is not True
                or sample.get("valid_patch") is not True
                or sample.get("finish_reason") != "stop"
                or sample["reward"].get("hook_errors") != 0
                or sample["reward"]["score"] != provenance["reward_delta"]
                or provenance["reward_delta"] <= 0
                or not math.isfinite(provenance["reward_delta"])
                or provenance["split"] != "train"
                or sample["response"] != row["messages"][-1]["content"]
            ):
                raise ValueError("Silver row lacks its clean positive execution evidence")
            key = hashlib.sha256(json.dumps(row["messages"], sort_keys=True).encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            provenance["generation_id"] = f"{source.name}/{identifier}"
            audit.append({
                "generation_id": provenance["generation_id"],
                "generation_seed": sample["seed"], "hook_errors": 0,
                "sample_sha256": file_sha(source / "samples" / f"{identifier}.json"),
                "source": str(source),
            })
            rows.append(row)
    if len(rows) < 256:
        raise RuntimeError(f"Bounded cold-start collection yielded only {len(rows)} unique clean positives")
    output.mkdir(parents=True, exist_ok=False)
    data = output / "sft.jsonl"
    with data.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(output / "CLEAN_EXECUTION_AUDIT.json", audit)
    receipt = {
        "stage": "fresh_base_silver_collection", "examples": len(rows),
        "usable_for_sft": True, "minimum_examples": 256,
        "dataset_sha256": file_sha(data), "input_identities": identities,
        "input_identity": {"engineer_manifest_sha256": shared["engineer_manifest_sha256"]},
        "clean_execution_audit_sha256": file_sha(output / "CLEAN_EXECUTION_AUDIT.json"),
        "sources": [str(source) for source in sources],
        "aggregation": "deduplicated_clean_positive_teacher_samples",
        "development_or_test_gain_claimed": False,
    }
    write_json(output / "COMPLETE.json", receipt)
    return receipt


def verify_grpo_commit(directory: Path, updates: int) -> dict:
    checkpoint = directory / f"checkpoint-{updates:06d}"
    receipt = json.loads((checkpoint / "COMMITTED.json").read_text())
    if receipt.get("updates_completed") != updates or receipt.get("optimizer_steps") != updates * 4:
        raise RuntimeError("GRPO did not commit the requested optimizer updates")
    if receipt.get("engine_reloaded_from_disk") is not True:
        raise RuntimeError("GRPO actor was not verified by rollout-engine reload")
    actor_sha = receipt.get("actor_sha256")
    if not isinstance(actor_sha, str) or not re.fullmatch("[0-9a-f]{64}", actor_sha):
        raise RuntimeError("GRPO commit lacks an actor identity")
    actor = checkpoint / "actor"
    identity = json.loads((actor / "actor_identity.json").read_text())
    files = identity.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("GRPO actor file inventory is absent")
    if not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise RuntimeError("GRPO actor inventory contains no model weights")
    inventory_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    if identity.get("sha256") != inventory_sha or actor_sha != inventory_sha:
        raise RuntimeError("GRPO actor identity does not bind its file inventory")
    for name, expected in files.items():
        path = actor / name
        if not path.resolve().is_relative_to(actor.resolve()) or path.is_symlink():
            raise RuntimeError("GRPO actor inventory contains a path escape")
        if file_sha(path) != expected:
            raise RuntimeError(f"GRPO actor file identity changed: {name}")
    states = receipt.get("rank_state_sha256")
    if not isinstance(states, dict) or set(states) != {"0", "1", "2", "3"}:
        raise RuntimeError("GRPO must commit all four training-rank states")
    for rank, expected in states.items():
        name = f"rank-{int(rank):04d}.pt"
        if file_sha(checkpoint / name) != expected:
            raise RuntimeError(f"GRPO rank-state identity changed: {name}")

    def finite(value: object) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("Non-finite GRPO commit metrics")
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, list):
            for item in value:
                finite(item)

    metrics = receipt.get("metrics_by_rank")
    if not isinstance(metrics, list) or len(metrics) != 4:
        raise RuntimeError("GRPO commit lacks optimizer metrics")
    finite(metrics)
    for item in metrics:
        if item.get("optimizer_steps") != 4 or len(item.get("iterations", [])) != 4:
            raise RuntimeError("GRPO rank did not perform all four optimizer iterations")
        for iteration in item["iterations"]:
            if not isinstance(iteration.get("loss"), (float, int)) or not isinstance(
                iteration.get("grad_norm"), (float, int)
            ):
                raise RuntimeError("GRPO optimizer iteration lacks loss or gradient evidence")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver", type=Path, required=True)
    parser.add_argument("--silver-unit", required=True)
    parser.add_argument("--engineer-unit", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reward-config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=float, default=21600)
    parser.add_argument("--initial-grpo-updates", type=int, default=1)
    parser.add_argument("--extra-silver-samples", type=int, default=0)
    parser.add_argument("--engineer-urls")
    parser.add_argument("--engineer-manifest", type=Path)
    args = parser.parse_args()
    for unit in (args.silver_unit, args.engineer_unit):
        if not re.fullmatch(r"searchqa-harness-r1-[a-z0-9-]+(?:\.service)?", unit):
            raise ValueError("Pipeline may operate only exact Harness-R1 experiment units")
    if args.initial_grpo_updates < 1:
        raise ValueError("Initial GRPO smoke stage requires at least one update")
    if args.extra_silver_samples < 0 or (
        args.extra_silver_samples and (not args.engineer_urls or not args.engineer_manifest)
    ):
        raise ValueError("Additional silver sampling requires exact Engineer endpoints and manifest")
    args.run.mkdir(parents=True, exist_ok=False)
    status_path = args.run / "STATUS.json"
    write_json(status_path, {
        "stage": "waiting_for_verified_silver", "silver": str(args.silver),
        "sft_started": False, "grpo_started": False, "experiment_complete": False,
    })
    launch = None
    try:
        complete = wait_for_silver(
            args.silver, args.silver_unit, args.wait_seconds,
            require_usable=not bool(args.extra_silver_samples),
        )
        silver = args.silver
        if not complete.get("usable_for_sft") or complete.get("examples", 0) < 256:
            extra = args.run / "additional_silver"
            write_json(status_path, {
                "stage": "collecting_additional_teacher_samples",
                "initial_clean_positives": complete["examples"],
                "extra_samples_per_packet": args.extra_silver_samples,
                "same_immutable_train_packets": True,
                "experiment_complete": False,
            })
            packets = sum(bool(line.strip()) for line in args.data.read_text().splitlines())
            subprocess.run([
                sys.executable, "-B", "-m", "harness_r1_searchqa.bootstrap",
                "--data", str(args.data), "--engineer-urls", args.engineer_urls,
                "--engineer-manifest", str(args.engineer_manifest),
                "--reward-config", str(args.reward_config), "--output", str(extra),
                "--packet-count", str(packets), "--samples", str(args.extra_silver_samples),
                "--seed", "1000042", "--minimum-examples", "256",
            ], check=True, timeout=args.wait_seconds,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
            silver = args.run / "merged_silver"
            complete = merge_clean_silver([args.silver, extra], silver)
        subprocess.run(
            ["systemctl", "--user", "stop", args.engineer_unit],
            check=True, timeout=120,
        )
        launch = Launcher(args.run / "processes")
        dataset = silver / "sft.jsonl"
        checkpoint = args.run / "engineer_sft"
        write_json(status_path, {
            "stage": "cold_start_sft", "examples": complete["examples"],
            "dataset_sha256": complete["dataset_sha256"],
            "experiment_complete": False,
        })
        sft = launch.start([
            sys.executable, "-B", "-m", "torch.distributed.run",
            "--standalone", "--local-addr=127.0.0.1", "--nproc-per-node=8",
            "--max-restarts=0", "-m", "harness_r1_searchqa.sft_train",
            "--data", str(dataset), "--data-sha256", complete["dataset_sha256"],
            "--output", str(checkpoint),
        ], gpus="0,1,2,3,4,5,6,7", name="engineer_sft")
        launch.wait(sft, "Engineer full SFT")
        marker = checkpoint / "COMPLETE"
        manifest = checkpoint / "manifest.json"
        if not marker.is_file() or not manifest.is_file():
            raise RuntimeError("SFT exited without a verified atomic checkpoint")
        lines = marker.read_text().splitlines()
        if len(lines) != 2 or lines[0] != "VERIFIED COMPLETE" or lines[1] != file_sha(manifest):
            raise RuntimeError("SFT completion marker does not bind its manifest")
        files = json.loads(manifest.read_text())["checkpoint_files_sha256"]
        for name, expected in files.items():
            if file_sha(checkpoint / name) != expected:
                raise RuntimeError(f"SFT checkpoint file identity changed: {name}")
        write_json(status_path, {
            "stage": "initial_online_grpo", "sft_checkpoint": str(checkpoint),
            "sft_manifest_sha256": file_sha(manifest),
            "grpo_updates_requested": args.initial_grpo_updates,
            "experiment_complete": False,
        })
        grpo = launch.start([
            sys.executable, "-B", "-m", "torch.distributed.run",
            "--standalone", "--local-addr=127.0.0.1", "--nproc-per-node=4",
            "--max-restarts=0", "-m", "harness_r1_searchqa.grpo_train",
            "--model", str(checkpoint), "--data", str(args.data),
            "--reward-config", str(args.reward_config),
            "--output", str(args.run / "grpo_smoke"),
            "--updates", str(args.initial_grpo_updates),
            "--rollout-gpus", "4,5,6,7",
        ], gpus="0,1,2,3", name="engineer_grpo_smoke")
        launch.wait(grpo, "Engineer online GRPO smoke")
        commit = verify_grpo_commit(args.run / "grpo_smoke", args.initial_grpo_updates)
        write_json(status_path, {
            "stage": "initial_optimizer_stage_verified",
            "sft_checkpoint": str(checkpoint),
            "grpo_output": str(args.run / "grpo_smoke"),
            "optimizer_steps_verified": commit["optimizer_steps"],
            "actor_sha256": commit["actor_sha256"],
            "requires_matched_dev_controls": True,
            "experiment_complete": False, "gain_claimed": False,
        })
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError) as exc:
        write_json(status_path, {
            "stage": "blocked", "error_type": type(exc).__name__,
            "error": str(exc), "experiment_complete": False, "gain_claimed": False,
        })
        raise
    finally:
        if launch is not None:
            launch.close()


if __name__ == "__main__":
    main()
