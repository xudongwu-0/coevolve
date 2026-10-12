"""Collect current-partner train-only baselines and compact authentic failure packets."""
from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from .alfworld_engineer_eval import Jobs, normalize_rewards
from .alfworld_reward_service import validate_metadata
from .alfworld_strong_phase import checked_release, train_calibration
from .alfworld_train_data import _immutable_json
from .budget import contract_digest, model_digests
from .common import ROOT, file_sha, read_json, write_json
from .gpu_telemetry import monitor, summarize
from .launch import Launcher
from .resources import gpu_release_audit, stop_known
from harness_r1_alfworld.driver import freeze_baseline, preserve_paired_ids


def builder_args() -> SimpleNamespace:
    return SimpleNamespace(
        reward_threshold=1.0, max_traces=4, strategy="round_robin",
        max_obs_chars=220, max_assistant_chars=180, max_approx_tokens=3000,
        response_protocol="prefill_think_patch", schema_style="alfworld_life_multihook_v1",
        include_incomplete=False, min_failures=1, max_prompt_chars=0,
        alfworld_split="test_set", target_agent_name="qwen35-9b-nothink", target_model="policy",
        prompt_version="sao_compact_train_evidence_v1",
    )


def packet_from_baseline(
    output: Path, task_ids: list[int], scores: dict, state: dict,
    solver_weights: str, request_sha256: str,
) -> tuple[dict | None, dict]:
    scripts = ROOT / "third_party/harness-r1-upstream/code/Relax/examples/harness_r1"
    sys.path.insert(0, str(scripts))
    from build_alfworld_dataset import build_record, iter_batch_sources

    if task_ids != list(range(task_ids[0], task_ids[0] + 10)) or not 0 <= task_ids[0] < 240:
        raise ValueError("Partner refresh must use complete legacy training cohorts")
    scores = normalize_rewards(scores, task_ids)
    sources = iter_batch_sources(output / "baseline")
    if len(sources) != 1:
        raise ValueError("Expected exactly one official baseline batch per cohort")
    source = sources[0]
    frozen = freeze_baseline(source.run_root)
    if not frozen:
        raise ValueError("Baseline evidence did not contain frozen official rollout files")
    identity_path = source.run_root.parent / "baseline_identity.json"
    _immutable_json(identity_path, {
        "baseline_files": frozen, "solver_model": state["solver"],
        "solver_version": state["solver_version"], "solver_weights_sha256": solver_weights,
        "request_sha256": request_sha256,
    })
    record = build_record(builder_args(), source)
    if record is None:
        if any(value != 1.0 for value in scores.values()):
            raise ValueError("Official packet builder omitted genuine training failures")
        metadata = {
            "batch_tag": source.batch_tag, "split": "test_set",
            "start": task_ids[0], "end": task_ids[-1] + 1, "batch_size": len(task_ids),
            "baseline_pass": len(task_ids), "baseline_rewards": scores,
            "reward_threshold": 1.0, "target_agent_name": "qwen35-9b-nothink",
            "target_model": "policy", "response_protocol": "prefill_think_patch",
            "solver_version": state["solver_version"], "solver_weights_sha256": solver_weights,
            "adapter_baseline_identity_sha256": file_sha(identity_path),
            "source_run_root": str(source.run_root), "task_ids": task_ids,
        }
        packet = None
    else:
        metadata = record["metadata"]
        if normalize_rewards(metadata["baseline_rewards"], task_ids) != scores:
            raise ValueError("Compacted prompt changed its checked baseline rewards")
        preserve_paired_ids([record], task_ids, file_sha(identity_path))
        metadata.update(solver_version=state["solver_version"], solver_weights_sha256=solver_weights)
        packet = {
            "id": f"alfworld-packet-{task_ids[0]}-{task_ids[-1] + 1}",
            "messages": record["prompt"], "metadata": metadata, "replay": None,
        }
    validate_metadata(metadata, state["solver_version"])
    if freeze_baseline(source.run_root) != frozen:
        raise ValueError("Packet construction modified baseline evidence")
    return packet, metadata


def successful_job_output(root: Path, name: str, result: dict) -> Path:
    request = read_json(root / name / "request.json")
    if result["request_sha256"] != contract_digest(request) or result["status"] != "complete":
        raise ValueError("Partner baseline completion changed identity")
    matches = [
        path.parent for path in (root / name).glob("attempt-*/result.json")
        if read_json(path) == result
    ]
    if len(matches) != 1:
        raise ValueError("Partner baseline has no unique successful raw-output attempt")
    return matches[0]


def run(protocol_path: Path, state_path: Path, root: Path) -> dict:
    protocol, state = read_json(protocol_path), read_json(state_path)
    if state["protocol_sha256"] != file_sha(protocol_path):
        raise ValueError("Counterpart state belongs to a different frozen protocol")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _immutable_json(root / "identity.json", {
            "protocol_sha256": file_sha(protocol_path), "state_sha256": file_sha(state_path),
            "solver_model": state["solver"], "solver_version": state["solver_version"],
            "cohorts": list(range(0, 240, 10)), "temperature": 0.0,
            "tokens_per_turn": 4096, "max_steps": 50,
        })
        if (root / "COMPLETE.json").exists() and (root / "RELEASED.json").exists():
            checked_release(root)
            complete = read_json(root / "COMPLETE.json")
            if complete["state_sha256"] != file_sha(root / "REFRESHED_STATE.json"):
                raise ValueError("Completed refreshed state changed")
            return complete
        if (root / "FAILED.json").exists():
            archive = root / "controller_failures"
            archive.mkdir(exist_ok=True)
            (root / "FAILED.json").rename(archive / f"{len(list(archive.glob('*.json'))) + 1:03d}.json")
        write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
        launch = Launcher(root / "supervision")
        stop = threading.Event()
        telemetry = threading.Thread(target=monitor, args=(root / "gpu_process_samples.jsonl", stop, 5), daemon=True)
        telemetry.start()
        started = time.monotonic()
        try:
            _, weights = model_digests(Path(state["solver"]))
            if weights != state["solver_weights_sha256"]:
                raise ValueError("Frozen counterpart weights changed before refresh")
            with launch.vllm(
                state["solver"], port=19501, gpus="0,1,2,3,4,5,6,7", name="current-partner",
                data_parallel=8, max_model_len=32768, tool_call_parser="qwen3_xml",
                gdn_prefill_backend="triton", language_model_only=True,
                max_num_seqs=32, server_args=["--enable-prefix-caching"],
            ) as urls:
                endpoints = urls.split(",")
                jobs = Jobs(root / "jobs", launch, concurrency=8)
                for index, start in enumerate(range(0, 240, 10)):
                    jobs.add({
                        "kind": "baseline", "purpose": "training_evidence",
                        "name": f"train-baseline-{start:03d}",
                        "task_ids": list(range(start, start + 10)), "concurrency": 10,
                        "url": endpoints[index % len(endpoints)], "port_index": index,
                        "solver_model": state["solver"], "solver_version": state["solver_version"],
                        "solver_weights_sha256": weights,
                    })
                results = jobs.wait()
            packets, baseline_metadata = [], []
            for start in range(0, 240, 10):
                name = f"train-baseline-{start:03d}"
                result = results[name]
                output = successful_job_output(root / "jobs", name, result)
                packet, metadata = packet_from_baseline(
                    output, list(range(start, start + 10)), result["scores"], state,
                    weights, result["request_sha256"],
                )
                baseline_metadata.append(metadata)
                if packet is not None:
                    packets.append(packet)
            calibration = train_calibration(
                [{"metadata": metadata} for metadata in baseline_metadata], state["solver_version"],
            )
            if calibration["task_ids"] != list(range(240)):
                raise ValueError("Counterpart refresh did not cover all declared train tasks")
            _immutable_json(root / "packets.json", packets)
            _immutable_json(root / "baseline_metadata.json", baseline_metadata)
            state = dict(
                state, packets=str(root / "packets.json"), packets_sha256=file_sha(root / "packets.json"),
                packet_solver=state["solver"], calibration=calibration,
                partner_refresh_receipt=str(root / "COMPLETE.json"),
            )
            _immutable_json(root / "REFRESHED_STATE.json", state)
            complete = {
                "status": "completed", "solver_model": state["solver"],
                "solver_weights_sha256": weights, "baseline_tasks": 240,
                "baseline_accuracy": calibration["accuracy"], "failure_packets": len(packets),
                "engineer_training_ready": bool(packets),
                "state": str(root / "REFRESHED_STATE.json"),
                "state_sha256": file_sha(root / "REFRESHED_STATE.json"),
                "elapsed_seconds": time.monotonic() - started, "report_opened": False,
                "interpretation": "Authentic current-partner greedy/per-turn-budget train evidence, not native total-token-budget evaluation.",
            }
            write_json(root / "COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
            raise
        finally:
            launch.close()
            stop.set()
            telemetry.join(timeout=15)
            path = root / "gpu_process_samples.jsonl"
            if path.is_file():
                try:
                    summary = summarize(path)
                except ValueError as exc:
                    summary = {"status": "invalid", "error": repr(exc)}
                write_json(root / "gpu_utilization_summary.json", summary)
            write_json(root / "RELEASED.json", {"released_at": time.time(), **gpu_release_audit()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.protocol, args.state, args.root), flush=True)


if __name__ == "__main__":
    main()
