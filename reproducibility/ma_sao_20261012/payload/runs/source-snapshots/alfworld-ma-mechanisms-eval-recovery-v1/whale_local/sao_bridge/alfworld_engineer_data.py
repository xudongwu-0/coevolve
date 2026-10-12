"""Rebuild compact Engineer prompts from frozen train evidence, without new rollouts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

from .alfworld_phase_data import editor_dataset
from .alfworld_reward_service import validate_metadata
from .alfworld_train_data import (
    _immutable_json, evaluation_partitions, evaluation_paths, goal_group,
)
from .common import ROOT, file_sha, read_json
from harness_r1_alfworld.driver import freeze_baseline, preserve_paired_ids, select_cohort


UPSTREAM = ROOT / "third_party/harness-r1-upstream"


def evaluation_ids(paths: list[str], partitions: dict, training_ids: list[int]) -> dict:
    if (
        not training_ids or len(training_ids) != len(set(training_ids))
        or any(type(index) is not int or not 0 <= index < 240 for index in training_ids)
    ):
        raise ValueError("Engineer evidence must declare distinct training task IDs")
    if any(
        type(index) is not int or not 240 <= index < 400
        for name in ("selection_task_ids", "report_task_ids")
        for index in partitions[name]
    ):
        raise ValueError("Engineer evaluation must use disjoint development tasks")
    train_goals = {goal_group(paths[index]) for index in training_ids}
    eligible = {
        name: [
            index for index in partitions[name]
            if goal_group(paths[index]) not in train_goals
        ]
        for name in ("selection_task_ids", "report_task_ids")
    }
    selection = eligible["selection_task_ids"]
    if len(selection) < 24 or len(eligible["report_task_ids"]) < 32:
        raise ValueError("Not enough goal-isolated Engineer evaluation tasks")
    chosen = select_cohort([paths[index] for index in selection], 24)
    result = {
        "selection_task_ids": [selection[position] for position in chosen],
        "report_task_ids": eligible["report_task_ids"],
    }
    selection_goals = {goal_group(paths[index]) for index in result["selection_task_ids"]}
    report_goals = {goal_group(paths[index]) for index in result["report_task_ids"]}
    if (
        len(selection_goals) != len(result["selection_task_ids"])
        or len(report_goals) != len(result["report_task_ids"])
        or selection_goals & report_goals
        or any(index >= 400 for ids in result.values() for index in ids)
    ):
        raise ValueError("Engineer selection/report goals are not disjoint development tasks")
    return result


def compact_packets(source: Path, *, approximate_tokens: int = 3000) -> list[dict]:
    if approximate_tokens <= 0:
        raise ValueError("Engineer evidence budget must be positive")
    scripts = UPSTREAM / "code/Relax/examples/harness_r1"
    sys.path.insert(0, str(scripts))
    from build_alfworld_dataset import build_record, iter_batch_sources

    packets = read_json(source)
    compact = []
    for packet in packets:
        metadata = packet["metadata"]
        validate_metadata(metadata, metadata["solver_version"])
        ids = metadata["task_ids"]
        if ids != list(range(metadata["start"], metadata["end"])) or max(ids) >= 240:
            raise ValueError("Compact Engineer prompts may use only complete legacy train cohorts")
        baseline = Path(metadata["source_run_root"])
        identity = baseline.parent / "baseline_identity.json"
        if file_sha(identity) != metadata["adapter_baseline_identity_sha256"]:
            raise ValueError("Original Engineer baseline identity changed")
        frozen = read_json(identity)["baseline_files"]
        if freeze_baseline(baseline) != frozen:
            raise ValueError("Original Engineer failure evidence changed")
        sources = iter_batch_sources(baseline)
        if len(sources) != 1:
            raise ValueError("Expected exactly one frozen baseline cohort per packet")
        args = SimpleNamespace(
            reward_threshold=metadata["reward_threshold"],
            max_traces=4, strategy="round_robin",
            max_obs_chars=220, max_assistant_chars=180,
            max_approx_tokens=approximate_tokens,
            response_protocol=metadata["response_protocol"],
            schema_style=metadata["schema_style"],
            include_incomplete=False, min_failures=1, max_prompt_chars=0,
            alfworld_split=metadata["split"],
            target_agent_name=metadata["target_agent_name"],
            target_model=metadata["target_model"],
            prompt_version="sao_compact_train_evidence_v1",
        )
        record = build_record(args, sources[0])
        if record is None:
            raise ValueError("Frozen failure evidence did not produce a compact packet")
        preserve_paired_ids(
            [record], ids, metadata["adapter_baseline_identity_sha256"],
        )
        for key in ("start", "end", "batch_size", "baseline_pass", "baseline_rewards"):
            previous, current = metadata[key], record["metadata"][key]
            if key == "baseline_rewards":
                previous = {int(k): v for k, v in previous.items()}
            if previous != current:
                raise ValueError(f"Compaction changed baseline reward identity: {key}")
        record["metadata"]["solver_version"] = metadata["solver_version"]
        compact.append({
            "id": packet["id"], "messages": record["prompt"],
            "metadata": record["metadata"], "replay": None,
        })
        if freeze_baseline(baseline) != frozen:
            raise ValueError("Packet builder modified the frozen baseline")
    if len({row["id"] for row in compact}) != len(compact):
        raise ValueError("Duplicate compact Engineer packet IDs")
    return compact


def prepare(source: Path, output: Path, *, rows: int = 192) -> dict:
    if rows < 1:
        raise ValueError("Engineer dataset row budget must be positive")
    packets = compact_packets(source)
    if len(packets) < 8:
        raise ValueError("The quick Engineer protocol needs eight matched proposal prompts")
    training_ids = [
        index for packet in packets for index in packet["metadata"]["task_ids"]
    ]
    versions = {packet["metadata"]["solver_version"] for packet in packets}
    if len(versions) != 1:
        raise ValueError("Engineer packets mix frozen Solver versions")
    paths = evaluation_paths()
    cohorts = evaluation_ids(paths, evaluation_partitions(), training_ids)
    chosen = [packets[index * len(packets) // 8] for index in range(8)]
    training = editor_dataset(packets, rows, 0)
    evaluation = editor_dataset(chosen, len(chosen), 0)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (
        ("packets.json", packets),
        ("train.json", training),
        ("evaluation_prompts.json", evaluation),
        ("evaluation_task_ids.json", cohorts),
    ):
        _immutable_json(output / name, value)
    manifest = {
        "source_packets": str(source.resolve()),
        "source_packets_sha256": file_sha(source),
        "solver_version": next(iter(versions)),
        "training_task_ids": training_ids,
        "training_goal_count": len({goal_group(paths[index]) for index in training_ids}),
        "packet_count": len(packets),
        "training_rows": len(training),
        "evaluation_packet_ids": [row["id"] for row in chosen],
        "evaluation_candidates_per_model": len(evaluation),
        **cohorts,
        "compaction": {
            "max_traces": 4, "max_approx_tokens": 3000,
            "max_obs_chars": 220, "max_assistant_chars": 180,
            "builder": "pinned official Harness-R1 build_record",
            "new_solver_rollouts": 0,
        },
        "files": {
            name: file_sha(output / name) for name in (
                "packets.json", "train.json", "evaluation_prompts.json",
                "evaluation_task_ids.json",
            )
        },
        "interpretation": (
            "Matched E0/E1 proposals use the same compact training evidence. "
            "Selection and report tasks are goal-disjoint from all training "
            "evidence and each other. This tests global harness proposal quality "
            "conditioned on these fixed packets, not transfer to unseen packets."
        ),
    }
    _immutable_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=192)
    args = parser.parse_args()
    result = prepare(args.source.resolve(), args.output.resolve(), rows=args.rows)
    print({
        "packets": result["packet_count"], "training_rows": result["training_rows"],
        "evaluation_candidates": result["evaluation_candidates_per_model"],
        "selection_tasks": len(result["selection_task_ids"]),
        "report_tasks": len(result["report_task_ids"]),
    }, flush=True)


if __name__ == "__main__":
    main()
