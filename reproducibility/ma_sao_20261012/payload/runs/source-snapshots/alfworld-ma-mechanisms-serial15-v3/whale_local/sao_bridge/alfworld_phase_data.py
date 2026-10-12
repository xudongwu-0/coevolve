"""Build immutable SAO datasets from audited Harness-R1 ALFWorld packets."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from .common import read_json, write_json

TRAINING_VALIDATOR_REMINDER = """

Training-time executable validator limits:
- Each add_code_hook `code` string must contain exactly one top-level
  `def hook(ctx, nb)` and at most 5 top-level helper functions.
- Nested functions and imports are forbidden.
- Each hook code string must stay below 8000 characters and 1200 AST nodes;
  keep it substantially smaller rather than approaching either limit.
- Each string literal must be at most 700 characters and returned hint/message
  text at most 900 characters.
Simplify the patch if needed; never trade executable validity for complexity.
""".rstrip()


def packet_rows(run_roots: list[Path], solver_version: str) -> list[dict]:
    rows = []
    seen = set()
    for root in run_roots:
        complete = read_json(root / "COMPLETE.json")
        if complete.get("status") != "completed":
            raise ValueError(f"incomplete matrix run: {root}")
        for result in complete["results"]:
            if result["pairing"] != "rl_agent_sft":
                continue
            attempt = result["attempts"][result["selected_attempt"] - 1]
            output = Path(attempt["output"])
            summary = attempt["summary"]
            if summary.get("status") == "baseline_all_pass_no_failures":
                continue
            if summary["categories"]["infra_failure"]:
                raise ValueError(f"infrastructure failure in packet source: {output}")
            packet_file = output / "failure_packets.jsonl"
            if not packet_file.is_file():
                continue
            for line in packet_file.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                metadata = dict(record["metadata"], solver_version=solver_version)
                task_ids = sorted(map(int, metadata["baseline_rewards"]))
                if task_ids != list(range(metadata["start"], metadata["end"])):
                    raise ValueError("training packet is not a complete contiguous cohort")
                key = (metadata["start"], metadata["end"])
                if key in seen:
                    raise ValueError(f"duplicate ALFWorld packet cohort: {key}")
                seen.add(key)
                rows.append({
                    "id": f"alfworld-packet-{metadata['start']}-{metadata['end']}",
                    "messages": record["prompt"],
                    "metadata": metadata,
                    "replay": None,
                })
    return sorted(rows, key=lambda row: row["metadata"]["start"])


def editor_dataset(packets: list[dict], count: int, partner_version: int, *,
                   replay_dir: Path | None = None, replay_fraction: float = 0.0,
                   editor_step_offset: int = 0, sample_offset: int = 0) -> list[dict]:
    if not packets or count < 1 or sample_offset < 0:
        raise ValueError("editor training requires nonempty packets and a positive count")
    if not 0 <= replay_fraction <= 0.5:
        raise ValueError("invalid ALFWorld Engineer replay fraction")
    replay_count = int(count * replay_fraction)
    if replay_count and replay_dir is None:
        raise ValueError("ALFWorld Engineer replay requires authentic previous receipts")
    if replay_count:
        from .replay import replay_rows
        replays = replay_rows(replay_dir, replay_count, editor_step_offset)
        current = {
            (row["metadata"]["start"], row["metadata"]["end"]): row["metadata"]
            for row in packets
        }
        for row in replays:
            previous = row.get("metadata")
            if previous is None:
                raise ValueError("ALFWorld Engineer replay receipt omitted packet metadata")
            key = (previous["start"], previous["end"])
            if key not in current:
                raise ValueError(f"refreshed ALFWorld replay packet is missing: {key}")
            row["metadata"] = copy.deepcopy(current[key])
    else:
        replays = []
    replay_positions = (
        {index * count // replay_count for index in range(replay_count)}
        if replay_count else set()
    )
    rows = []
    for index in range(count):
        if index in replay_positions:
            row = replays.pop(0)
            row["id"] = (
                f"{row['id']}-w{partner_version}-sample{sample_offset + index}")
        else:
            row = copy.deepcopy(packets[(sample_offset + index) % len(packets)])
            row["id"] = (
                f"{row['id']}-w{partner_version}-sample{sample_offset + index}")
            system = [message for message in row["messages"] if message["role"] == "system"]
            if len(system) != 1:
                raise ValueError("ALFWorld Engineer row must contain one system message")
            system[0]["content"] += TRAINING_VALIDATOR_REMINDER
        row["partner_version"] = partner_version
        row["metadata_json"] = json.dumps(
            row.pop("metadata"), sort_keys=True, separators=(",", ":"))
        rows.append(row)
    return rows


def solver_dataset(task_ids: list[int], count: int, offset: int, partner_version: int,
                   task_names: dict[int, str] | None = None) -> list[dict]:
    if not task_ids or count < 1 or offset < 0:
        raise ValueError("invalid ALFWorld solver dataset request")
    scheduled_ids = list(task_ids)
    if task_names is not None:
        missing = set(task_ids) - set(task_names)
        if missing:
            raise ValueError(f"ALFWorld task assignments are missing IDs: {sorted(missing)}")
        groups: dict[str, list[int]] = {}
        for task_id in task_ids:
            groups.setdefault(task_names[task_id], []).append(task_id)
        scheduled_ids = [
            group[position]
            for position in range(max(map(len, groups.values())))
            for group in groups.values()
            if position < len(group)
        ]
    rows = []
    for index in range(count):
        task_id = scheduled_ids[(offset + index) % len(scheduled_ids)]
        row = {
            "id": (
                f"alfworld-task-{task_id}-h{partner_version}-"
                f"sample{offset + index}"
            ),
            "task_id": task_id, "partner_version": partner_version,
            "recycled_ids": [], "warmup": False,
        }
        if task_names is not None:
            row["task_name"] = task_names[task_id]
        rows.append(row)
    return rows


def assignments_from_matrix(run_roots: list[Path], no_patch: Path,
                            harness_version: int) -> tuple[list[dict], dict[int, str]]:
    assignments = []
    task_names = {}
    for root in run_roots:
        complete = read_json(root / "COMPLETE.json")
        if complete.get("status") != "completed":
            raise ValueError(f"incomplete matrix run: {root}")
        for result in complete["results"]:
            if result["pairing"] != "rl_agent_sft":
                continue
            attempt = result["attempts"][result["selected_attempt"] - 1]
            summary = attempt["summary"]
            start, end = result["start"], result["start"] + result["size"]
            if summary.get("status") == "baseline_all_pass_no_failures":
                harness = no_patch
            elif (summary["categories"]["valid_evaluated"] == 1
                  and summary["categories"]["infra_failure"] == 0):
                harness = Path(attempt["output"]) / "patched/baseline/patch.json"
            else:
                harness = no_patch
            if not harness.is_file():
                raise FileNotFoundError(f"ALFWorld harness artifact is missing: {harness}")
            name = f"alfworld-h{harness_version}-b{start:03d}"
            task_ids = list(range(start, end))
            assignments.append({
                "name": name, "task_ids": task_ids, "harness": str(harness.resolve()),
            })
            for task_id in task_ids:
                if task_id in task_names:
                    raise ValueError(f"duplicate ALFWorld assignment for task {task_id}")
                task_names[task_id] = name
    assignments.sort(key=lambda row: row["task_ids"][0])
    return assignments, task_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--solver-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int)
    parser.add_argument("--partner-version", type=int, default=0)
    parser.add_argument("--replay-dir", type=Path)
    parser.add_argument("--replay-fraction", type=float, default=0.0)
    parser.add_argument("--editor-step-offset", type=int, default=0)
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args()
    packets = packet_rows(args.matrix_root, args.solver_version)
    rows = editor_dataset(
        packets, args.count, args.partner_version,
        replay_dir=args.replay_dir, replay_fraction=args.replay_fraction,
        editor_step_offset=args.editor_step_offset, sample_offset=args.sample_offset,
    ) if args.count else packets
    write_json(args.output, rows)


if __name__ == "__main__":
    main()
