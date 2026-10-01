"""Train-only ALFWorld games isolated from existing evaluation goal groups."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from pathlib import Path
import re

from .alfworld_runtime import ALFWORLD_DATA
from .common import file_sha, read_json, write_json

TRAIN_SPLIT = "sao_train_v1"
TRAIN_WORKERS = 32
VALIDATION_SPLITS = ("valid_seen", "valid_unseen", "valid_train")


def goal_group(game: str) -> str:
    if not isinstance(game, str):
        raise ValueError("ALFWorld game paths must be strings")
    path = Path(game)
    if (
        path.is_absolute() or ".." in path.parts or path.as_posix() != game
        or len(path.parts) != 5
        or path.parts[0] != "json_2.1.1" or path.name != "game.tw-pddl"
    ):
        raise ValueError(f"Invalid relative ALFWorld game path: {game}")
    return path.parts[2]


def evaluation_paths(data_root: Path | None = None) -> list[str]:
    root = data_root if data_root is not None else ALFWORLD_DATA
    heldout = read_json(root / "test_set.json")
    if not isinstance(heldout, dict) or any(
        not isinstance(group, list) for group in heldout.values()
    ):
        raise ValueError("Official ALFWorld catalogue must contain lists of games")
    paths = [path for group in heldout.values() for path in group]
    if len(paths) != 500 or any(not isinstance(path, str) for path in paths):
        raise ValueError("Expected the full official 500-task ALFWorld catalogue")
    if len(set(paths)) != 500:
        raise ValueError("Official ALFWorld catalogue contains duplicate games")
    for path in paths:
        goal_group(path)
    return paths


def evaluation_partitions(data_root: Path | None = None) -> dict:
    root = data_root if data_root is not None else ALFWORLD_DATA
    paths = evaluation_paths(root)
    seen = set()
    result = {}
    for name, start, end in (("selection", 240, 300), ("report", 300, 400)):
        indices = []
        for index in range(start, end):
            group = goal_group(paths[index])
            if group not in seen:
                seen.add(group)
                indices.append(index)
        if not indices:
            raise ValueError(f"No goal-disjoint ALFWorld {name} tasks remain")
        result[f"{name}_task_ids"] = indices
    return {
        **result,
        "existing_catalogue_sha256": file_sha(root / "test_set.json"),
        "selection_report_goal_group_overlap": 0,
        "selection_and_report_each_have_unique_goal_groups": True,
        "reserved_400_499_not_used": True,
    }


def reserved_goals(data_root: Path) -> set[str]:
    paths = evaluation_paths(data_root)
    reserved = {goal_group(path) for path in paths}
    for split in VALIDATION_SPLITS:
        games = list((data_root / "json_2.1.1" / split).glob("*/*/game.tw-pddl"))
        if not games:
            raise ValueError(f"Missing reserved ALFWorld validation games: {split}")
        reserved.update(game.parents[1].name for game in games)
    return reserved


def build_catalogue(data_root: Path) -> dict:
    reserved = reserved_goals(data_root)
    groups = defaultdict(list)
    for game in sorted((data_root / "json_2.1.1/train").glob("*/*/game.tw-pddl")):
        relative = game.relative_to(data_root).as_posix()
        group = goal_group(relative)
        if group not in reserved:
            groups[group].append(relative)
    if not groups:
        raise ValueError("No train-only ALFWorld goal groups remain")

    def order(value: str) -> str:
        return hashlib.sha256(value.encode()).hexdigest()

    groups = {
        group: sorted(groups[group], key=order)
        for group in sorted(groups, key=order)
    }
    # Cover every goal group before adding a second instance of any goal.
    games = [
        group[position]
        for position in range(max(map(len, groups.values())))
        for group in groups.values()
        if position < len(group)
    ]
    return {"train": games}


@lru_cache(maxsize=8)
def _load_catalogue(
    path: Path, expected_sha256: str, data_root: Path,
) -> tuple[str, ...]:
    if file_sha(path) != expected_sha256:
        raise ValueError("ALFWorld training catalogue checksum changed")
    catalogue = read_json(path)
    if not isinstance(catalogue, dict) or set(catalogue) != {"train"}:
        raise ValueError("Training catalogue must contain only the train game list")
    games = catalogue["train"]
    if not isinstance(games, list) or not games:
        raise ValueError("Training catalogue is empty or malformed")
    reserved = reserved_goals(data_root)
    for game in games:
        group = goal_group(game)
        if Path(game).parts[1] != "train" or group in reserved:
            raise ValueError("Training catalogue overlaps a reserved evaluation goal group")
    if len(games) != len(set(games)):
        raise ValueError("Training catalogue contains duplicate game instances")
    return tuple(games)


def training_catalogue(
    catalogue_path: str | Path | None, expected_sha256: str | None, *,
    data_root: Path | None = None,
) -> tuple[str, ...]:
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256,
    ):
        raise ValueError("Explicit training catalogue SHA-256 is required")
    if not isinstance(catalogue_path, (str, Path)):
        raise ValueError("Explicit training catalogue path is required")
    root = (data_root if data_root is not None else ALFWORLD_DATA).resolve()
    path = Path(catalogue_path).resolve()
    if file_sha(path) != expected_sha256:
        raise ValueError("ALFWorld training catalogue checksum changed")
    return _load_catalogue(path, expected_sha256, root)


def task_count(
    split: str, catalogue_sha256: str | None = None, *,
    catalogue_path: str | Path | None = None,
) -> int:
    if split == "test_set":
        if catalogue_path is not None or catalogue_sha256 is not None:
            raise ValueError("The official test_set catalogue cannot be overridden")
        return 500
    if split != TRAIN_SPLIT:
        raise ValueError(f"Unsupported ALFWorld task split: {split}")
    return len(training_catalogue(catalogue_path, catalogue_sha256))


def solver_rows(
    games: tuple[str, ...], catalogue_sha256: str, *, catalogue_path: Path,
) -> list[dict]:
    if not games:
        raise ValueError("Solver rows require a nonempty training catalogue")
    return [
        {
            "id": f"alfworld-{TRAIN_SPLIT}-{index}",
            "task_id": index,
            "task_name": f"alfworld-train-{index % TRAIN_WORKERS:02d}",
            "task_split": TRAIN_SPLIT,
            "task_catalogue": str(catalogue_path.resolve()),
            "task_catalogue_sha256": catalogue_sha256,
            "partner_version": 0,
            "recycled_ids": [],
            "warmup": False,
        }
        for index in range(len(games))
    ]


def _immutable_json(path: Path, content: dict | list) -> None:
    if path.exists():
        if read_json(path) != content:
            raise ValueError(f"Refusing to change frozen ALFWorld data: {path}")
    else:
        write_json(path, content)


def prepare(output: Path, data_root: Path = ALFWORLD_DATA) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    catalogue_path = output / f"{TRAIN_SPLIT}.json"
    _immutable_json(catalogue_path, build_catalogue(data_root))
    digest = file_sha(catalogue_path)
    games = training_catalogue(catalogue_path, digest, data_root=data_root)
    data_path = output / "solver_train.json"
    _immutable_json(data_path, solver_rows(games, digest, catalogue_path=catalogue_path))
    manifest = {
        "split": TRAIN_SPLIT,
        "catalogue": str(catalogue_path.resolve()),
        "catalogue_sha256": digest,
        "training_data": str(data_path.resolve()),
        "training_data_sha256": file_sha(data_path),
        "game_instances": len(games),
        "distinct_goal_groups": len({goal_group(game) for game in games}),
        "training_workers": TRAIN_WORKERS,
        "data_source_root": str(data_root.resolve()),
        "task_families": dict(sorted(Counter(
            goal_group(game).split("-")[0] for game in games
        ).items())),
        "excluded_existing_catalogue_sha256": file_sha(data_root / "test_set.json"),
        "excluded_validation_splits": list(VALIDATION_SPLITS),
        "reserved_goal_group_overlap": 0,
        "interpretation": (
            "Different instances may share a training goal group. All existing "
            "500-task and validation-split goal groups are excluded; no model "
            "evaluation or validation reward is used to build this catalogue."
        ),
    }
    _immutable_json(output / "manifest.json", manifest)
    _immutable_json(output / "evaluation_task_ids.json", evaluation_partitions(data_root))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.output), flush=True)


if __name__ == "__main__":
    main()
