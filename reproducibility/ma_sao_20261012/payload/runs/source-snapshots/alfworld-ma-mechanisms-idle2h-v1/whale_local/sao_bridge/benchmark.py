"""Deterministic DAPO/ReTool splits with sealed AIME evaluation rows."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

from .common import file_sha, write_json

DAPO_PREFIX = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $Answer (without quotes) where $Answer is the answer "
    "to the problem.\n\n"
)
DAPO_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'


def task_id(question: str) -> str:
    canonical = re.sub(r"\s+", "", question).casefold()
    return hashlib.sha256(canonical.encode()).hexdigest()


def normalize(row: dict, source: str) -> dict:
    if "prompt" in row and "reward_model" in row:
        messages = row["prompt"]
        users = [m["content"] for m in messages if m["role"] == "user"]
        if len(users) != 1:
            raise ValueError("Expected one DAPO problem prompt")
        question = users[0].strip()
        if question.startswith(DAPO_PREFIX) and question.endswith(DAPO_SUFFIX):
            question = question[len(DAPO_PREFIX):-len(DAPO_SUFFIX)].strip()
        answer = row["reward_model"]["ground_truth"]
        if isinstance(answer, dict):
            target = answer.get("target")
            if target is None or len(target) != 1:
                raise ValueError("Unsupported DAPO target shape")
            answer = target[0]
    else:
        qkey = next((key for key in ("problem", "Problem", "question", "Question")
                     if key in row), None)
        akey = next((key for key in ("answer", "Answer") if key in row), None)
        if qkey is None or akey is None:
            raise ValueError(f"Unsupported benchmark schema: {sorted(row)}")
        question, answer = str(row[qkey]).strip(), row[akey]
    if not question or answer is None:
        raise ValueError("Empty benchmark problem or answer")
    answer = str(answer).strip()
    if not re.fullmatch(r"-?\d+", answer):
        raise ValueError("DAPO/AIME target must be an integer string")
    return {"id": task_id(question), "question": question,
            "answer": str(int(answer)), "source": source}


def prepare(train_paths: list[Path], test_paths: list[Path], out: Path, seed: int = 42):
    import pandas as pd

    if out.exists():
        raise FileExistsError(out)
    train, test = {}, {}
    conflicts = set()
    source_counts = {}
    for paths, destination in ((train_paths, train), (test_paths, test)):
        for path in paths:
            frame = pd.read_parquet(path)
            source_counts[str(path)] = len(frame)
            for raw in frame.to_dict("records"):
                row = normalize(raw, path.parent.name)
                if row["id"] in destination and destination[row["id"]]["answer"] != row["answer"]:
                    if destination is test:
                        raise ValueError("Conflicting held-out labels")
                    conflicts.add(row["id"])
                destination[row["id"]] = row
    for key in conflicts:
        del train[key]
    overlap = set(train) & set(test)
    for key in overlap:
        del train[key]
    ordered = sorted(train.values(), key=lambda r: hashlib.sha256(
        f"{seed}:{r['id']}".encode()).hexdigest())
    if len(ordered) < 4096 + 512 or len(test) < 60:
        raise ValueError("Insufficient DAPO training or full two-year AIME examples")
    splits = {
        "reward": ordered[:256], "dev": ordered[256:512],
        "train": ordered[512:], "test": sorted(test.values(), key=lambda row: row["id"]),
    }
    out.mkdir(parents=True, mode=0o700)
    manifest = {
        "benchmark": "DAPO-Math-17k / AIME2024+2025 with Python tools",
        "paper": "https://arxiv.org/abs/2609.00196",
        "split_seed": seed, "exact_normalized_train_test_duplicates_removed": len(overlap),
        "conflicting_train_questions_excluded": len(conflicts),
        "source_row_counts": source_counts,
        "source_sha256": {str(p.resolve()): file_sha(p) for p in train_paths + test_paths},
        "selection_uses_test": False, "splits": {},
    }
    for name, rows in splits.items():
        write_json(out / f"{name}.json", rows)
        manifest["splits"][name] = {
            "count": len(rows), "sha256": file_sha(out / f"{name}.json"),
            "ids_sha256": hashlib.sha256("\n".join(r["id"] for r in rows).encode()).hexdigest(),
        }
    write_json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = prepare(args.train, args.test, args.output)
    print({key: value["count"] for key, value in manifest["splits"].items()})
