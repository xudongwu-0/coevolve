"""Validate completed phase boundaries and account for unused asynchronous work."""
from collections import Counter
from pathlib import Path

from .common import file_sha, read_json


def editor_pair_audit(run: Path, phase: str, spec: dict) -> dict:
    tasks = read_json(run / f"{phase}_editor_data.json")
    by_id = {row["id"]: row for row in tasks}
    expected = (spec["editor_steps"] + 1) * spec.get("editor_batch_size", spec["batch_size"])
    receipts = [read_json(p) for p in (run / phase / "receipts").glob("*.json")]
    if (len(by_id) != expected or len(receipts) != expected
            or Counter(row["id"] for row in receipts) != Counter(by_id.keys())):
        raise ValueError(f"{phase}: missing or duplicate authentic editor receipts")
    committed = {}
    for row in receipts:
        index = row["evaluation"]["request_index"]
        if index in committed or row["role"] != "editor":
            raise ValueError(f"{phase}: ambiguous committed reward request")
        committed[index] = row
    pairs = {}
    files = {}
    for path in (run / f"{phase}_reward_evals").glob("*.json"):
        record = read_json(path)
        index, role = record["request_index"], record["pair_role"]
        key = (index, role)
        if (role not in {"candidate", "reference"} or key in pairs
                or record["sample_id"] not in by_id
                or record["question_count"] != spec["reward_questions"]
                or record["samples_per_question"] != 1
                or len(record["results"]) != spec["reward_questions"]):
            raise ValueError(f"{phase}: invalid physical reward receipt: {path.name}")
        pairs[key] = record
        files[path.name] = file_sha(path)
    indices = {index for index, _ in pairs}
    if not set(committed).issubset(indices):
        raise ValueError(f"{phase}: missing committed physical reward pairs")
    for index in indices:
        if any((index, role) not in pairs for role in ("candidate", "reference")):
            raise ValueError(f"{phase}: incomplete physical reward pair {index}")
        candidate, reference = (pairs[index, role] for role in ("candidate", "reference"))
        if candidate["sample_id"] != reference["sample_id"]:
            raise ValueError(f"{phase}: mismatched paired task identity")
        if index in committed:
            row = committed[index]
            evaluation = row["evaluation"]
            if (candidate["sample_id"] != row["id"]
                    or candidate["harness_source"] != evaluation["source"]
                    or reference["harness_source"] != evaluation["reference_source"]):
                raise ValueError(f"{phase}: reward pair differs from consumed receipt")
    unused = sorted(indices - committed.keys())
    if unused and (spec.get("formal_baseline") or spec.get("contract")):
        raise ValueError("Formal fixed-budget run contains unused asynchronous overhead")
    fresh = sum(not by_id[pairs[i, "candidate"]["sample_id"]].get("replay") for i in unused)
    return {
        "committed_request_indices": sorted(committed),
        "unused_request_indices": unused,
        "editor_rollouts": fresh,
        "editor_replayed_actions": len(unused) - fresh,
        "harness_scoring_rollouts": len(unused) * 2 * spec["reward_questions"],
        "raw_files_sha256": files,
        "strict_sample_budget_matched": not unused,
        "note": "All completed physical pairs retained; unused pairs excluded from partner recycling. "
                "Cancelled in-flight tokens without receipts are not fully metered.",
    }


def validate_resume(run: Path, spec: dict) -> float:
    if read_json(run / "spec.json") != spec:
        raise ValueError("Resume cannot change the experiment specification")
    status = read_json(run / "STATUS.json")
    if status["status"] != "failed":
        raise ValueError("Only a stopped failed experiment may resume")
    for path in run.glob("[hw][0-9]*"):
        if not path.is_dir() or not path.name[1:].isdigit():
            continue
        if not (path / "trained.json").is_file():
            raise ValueError(f"Cannot repeat a partial optimizer phase: {path.name}")
        phase_spec = read_json(run / f"{path.name}_spec.json")
        receipt = read_json(path / "trained.json")
        import json
        metrics = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if ([row["global_step"] for row in metrics] != list(range(phase_spec["steps"]))
                or receipt["actor_steps"] + receipt["critic_only_warmup_steps"] != phase_spec["steps"]):
            raise ValueError(f"Incomplete saved optimizer phase: {path.name}")
    from .analysis import verify_phases
    verify_phases(run)
    return float(status["elapsed_seconds"])


def cached_evaluation(path: Path, rows: list, samples: int) -> dict:
    report = read_json(path)
    if (report["samples_per_question"] != samples
            or report["question_count"] != len(rows)
            or Counter(row["id"] for row in report["results"])
            != Counter({row["id"]: samples for row in rows})):
        raise ValueError(f"Incomplete or mismatched cached evaluation: {path}")
    return report
