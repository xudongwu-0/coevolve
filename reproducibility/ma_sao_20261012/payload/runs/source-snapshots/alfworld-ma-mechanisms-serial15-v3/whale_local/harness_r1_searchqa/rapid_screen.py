"""Train-only paired triage, never final acceptance.

Default stages are nested 32 then 128 held-out train questions, two seeds each.
The second stage adds only the remaining 96 questions. Question-bootstrap CIs
come from selection.paired_gain; promising candidates still require the
existing full 256-question x 8-sample dev confirmation. No test split is read.

CLI: python -m harness_r1_searchqa.rapid_screen --help
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

from sao_bridge.common import write_json
from sao_bridge.evaluation import Sampler
from sao_bridge.searchqa_prepare import ASSETS, sha256

from .data import load_train_rows
from .episode import H0_CONFIG, evaluate_one
from .identity import runtime_source_identity
from .patch import PatchValidationError, extract_patch, merge_patches, patch_digest, _normalize_incumbent
from .reward import _hook_stats
from .selection import paired_gain
from . import selection


SAMPLES = 2
STAGES = (32, 128)
CONFIRMATION = {"split": "dev", "questions": 256, "samples_per_question": 8}


class CandidateExecutionRejected(ValueError):
    """An observed candidate/ incumbent hook error stops all further episodes."""


def load_exclusions(paths: list[Path]) -> tuple[set[str], list[dict]]:
    """Read supplied RL JSON/JSONL packets, excluding every metadata.task_id."""
    excluded, identities = set(), []
    if not paths:
        raise ValueError("Supply excluded RL packet files explicitly")
    for path in paths:
        content = Path(path).read_bytes()
        if Path(path).suffix == ".jsonl":
            records = [json.loads(line) for line in content.splitlines() if line.strip()]
        else:
            records = json.loads(content)
            if isinstance(records, dict):
                records = [records]
        if not isinstance(records, list):
            raise ValueError("Excluded packet files must contain RL records")
        for record in records:
            metadata = record.get("metadata") if isinstance(record, dict) else None
            if (not isinstance(metadata, dict) or metadata.get("split") != "train"
                    or metadata.get("benchmark") != "searchqa"):
                raise ValueError("Excluded packets must declare SearchQA train metadata")
            task_ids = metadata.get("task_ids")
            if (not isinstance(task_ids, list) or not task_ids
                    or any(type(identity) is not str or not identity for identity in task_ids)):
                raise ValueError("Excluded packets require explicit metadata.task_ids")
            excluded.update(task_ids)
        identities.append({"path": str(Path(path).resolve()),
                           "sha256": hashlib.sha256(content).hexdigest(),
                           "packet_count": len(records)})
    return excluded, identities


def select_heldout_train(
    data_root: Path, excluded_task_ids: set[str], *, count: int, seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Stable proportional-stratum ordering; any shorter prefix is nested."""
    if type(count) is not int or count < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Selection count and seed must be positive/nonnegative integers")
    train = load_train_rows(data_root)
    train_ids = {row["id"] for row in train}
    if not excluded_task_ids <= train_ids:
        raise ValueError("Excluded packets contain non-train task identities")
    eligible = [row for row in train if row["id"] not in excluded_task_ids]
    if len(eligible) < count:
        raise ValueError(f"Need {count} held-out train questions; only {len(eligible)} remain")
    field = next((name for name in ("data_source", "source_dataset", "dataset", "stratum")
                  if all(type(row.get(name)) is str and row[name] for row in eligible)), None)
    groups = defaultdict(list)
    for row in eligible:
        groups[row[field] if field else "all"].append(row)
    def order(row):
        return hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(), row["id"]
    for rows in groups.values():
        rows.sort(key=order)
    counts = {name: len(rows) for name, rows in groups.items()}
    chosen = Counter()
    selected = []
    for position in range(count):
        available = [name for name in groups if chosen[name] < counts[name]]
        name = min(available, key=lambda name: (
            chosen[name] - (position + 1) * counts[name] / len(eligible), name))
        selected.append(groups[name][chosen[name]])
        chosen[name] += 1
    return selected, {
        "split": "train", "selection_seed": seed, "train_questions": len(train),
        "eligible_questions": len(eligible), "excluded_questions": len(excluded_task_ids),
        "excluded_task_ids_sha256": hashlib.sha256(
            "\n".join(sorted(excluded_task_ids)).encode()).hexdigest(),
        "stratification_field": field, "eligible_stratum_counts": counts,
        "selected_stratum_counts": dict(chosen),
        "selected_task_ids": [row["id"] for row in selected],
    }


def _effect_identity(patch: dict) -> dict:
    return {action["hook"]: ast.dump(ast.parse(action["code"]), include_attributes=False)
            for action in patch["actions"]}


def _validate_result(result: dict, row: dict, seed: int) -> None:
    if (not isinstance(result, dict) or result.get("task_id") != row["id"]
            or type(result.get("rollout_seed")) is not int or result["rollout_seed"] != seed
            or type(result.get("reward")) not in (int, float) or result["reward"] not in (0, 1)):
        raise ValueError("Screen episode returned invalid paired identity/seed/reward")
    stats = _hook_stats([result])  # Infrastructure/unknown statuses raise, never become zeros.
    if stats["hook_errors"]:
        raise CandidateExecutionRejected(f"Hook runtime error on {row['id']}")


def _partial_hook_stats(results):
    try:
        return _hook_stats(results)
    except RuntimeError as exc:
        return {"aggregation_error": str(exc), "clean_execution": False}


async def _collect_arm(rows, seeds, *, sampler, patch, concurrency, record, on_start) -> list[dict]:
    semaphore = asyncio.Semaphore(concurrency)
    stopped = False
    async def run(index):
        nonlocal stopped
        async with semaphore:
            if stopped:
                return None
            try:
                on_start()
                result = await evaluate_one(rows[index], sampler=sampler, patch=patch,
                                            total_tokens=4096, rollout_seed=seeds[index])
                record(index, result)
                _validate_result(result, rows[index], seeds[index])
                return index, result
            except BaseException:
                stopped = True
                raise
    tasks = [asyncio.create_task(run(index)) for index in range(len(rows))]
    try:
        completed = await asyncio.gather(*tasks)
        return [result for _, result in sorted(completed)]
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def run_screen(
    *, target_urls: str, data_root: Path, candidate: dict, incumbent: dict,
    excluded_packets: list[Path], output: Path, temperature: float = 1.0,
    tokens: int = 4096, seed: int = 42, concurrency: int = 32,
    max_questions: int = 128, retrieval_url: str | None = None,
    target_model_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist paired triage; transport/infrastructure failures persist ABORTED and raise."""
    if (type(temperature) not in (int, float) or temperature != 1.0
            or type(tokens) is not int or tokens != 4096
            or type(seed) is not int or seed < 0
            or type(concurrency) is not int or concurrency < 1
            or type(max_questions) is not int or max_questions not in STAGES):
        raise ValueError("Frozen screen requires temperature=1, tokens=4096, 32/128 questions")
    if (H0_CONFIG["max_tool_calls"] != 1 or H0_CONFIG["max_steps"] != 2
            or H0_CONFIG["top_k"] != 1 or H0_CONFIG["document_char_limit"] != 800):
        raise ValueError("Frozen screen tool/step/retrieval budget changed")
    for url in target_urls.split(","):
        parsed = urlsplit(url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Invalid target serving URL")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    completed_episodes = 0
    started_episodes = 0
    stages = []
    observed = {"incumbent": [], "candidate": []}
    sampler = None
    identity = {
        "decision_scope": "heldout_train_screen_only", "accepted": False,
        "final_accepted": False, "required_confirmation": CONFIRMATION,
        "target_urls": target_urls, "target_model_sha256": target_model_sha256,
        "temperature": temperature, "top_p": 1.0, "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": False},
        "total_tokens": tokens, "episode_config": dict(H0_CONFIG),
        "seed": seed, "samples_per_question": SAMPLES, "concurrency": concurrency,
        "max_questions": max_questions, "runtime_source_sha256": runtime_source_identity(),
        "screen_source_sha256": sha256(Path(__file__)),
        "selection_source_sha256": sha256(Path(selection.__file__)),
    }
    def finish(status, reason):
        result = {**identity, "status": status, "reason": reason, "stages": stages,
                  "episodes_completed": completed_episodes,
                  "episodes_started": started_episodes,
                  "elapsed_seconds": time.monotonic() - started,
                  "pilot_only": max_questions == 32,
                  "hook_stats_by_arm": {arm: _partial_hook_stats(results)
                                        for arm, results in observed.items()}}
        write_json(output / "COMPLETE.json", result)
        return result
    try:
        identity["incumbent_patch_sha256"] = patch_digest(incumbent)
        try:
            merged = merge_patches(incumbent, candidate)
        except PatchValidationError as exc:
            return finish("rejected", f"invalid_candidate_patch: {str(exc)[:600]}")
        identity.update(candidate_patch_sha256=patch_digest(merged), merged_candidate=merged)
        if _effect_identity(merged) == _effect_identity(incumbent):
            return finish("rejected", "unmodified_patch")
        excluded, packet_identities = load_exclusions(excluded_packets)
        questions, selected = select_heldout_train(
            Path(data_root), excluded, count=max_questions, seed=seed)
        retrieval_url = retrieval_url or os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL", "")
        configured = os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL")
        if configured and configured != retrieval_url:
            raise ValueError("Conflicting frozen retrieval service URL")
        parsed = urlsplit(retrieval_url)
        if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
                or parsed.query or parsed.fragment or parsed.path not in ("", "/", "/retrieve")):
            raise ValueError("Supply trusted retrieval URL explicitly or through environment")
        retrieval_manifest = Path(data_root) / "retrieval_manifest.json"
        if sha256(retrieval_manifest) != sha256(ASSETS / "retrieval_manifest.json"):
            raise ValueError("Screen retriever differs from frozen episode retriever")
        identity.update(
            data_manifest_sha256=sha256(Path(data_root) / "manifest.json"),
            retrieval_manifest_sha256=sha256(retrieval_manifest), retrieval_url=retrieval_url,
            excluded_packets=packet_identities, heldout_selection=selected)
        write_json(output / "IDENTITY.json", identity)
        os.environ["SAO_SEARCHQA_RETRIEVAL_URL"] = retrieval_url
        sampler = Sampler(target_urls, concurrency=concurrency, temperature=1.0,
                          chat_template_kwargs={"enable_thinking": False})
        cumulative = {"incumbent": [], "candidate": []}
        previous_count = 0
        for size in (value for value in STAGES if value <= max_questions):
            stage_started = time.monotonic()
            stage_episode_start = completed_episodes
            stage_attempt_start = started_episodes
            rows, seeds = [], []
            for question in questions[previous_count:size]:
                for sample in range(SAMPLES):
                    rows.append(question)
                    seeds.append(seed + int(question["id"][:8], 16) % 1_000_000 + 1009 * sample)
            stage = {"questions": size, "added_questions": size - previous_count,
                     "planned_incremental_episodes": len(rows) * 2}
            stages.append(stage)
            try:
                for arm, spec in (("incumbent", incumbent), ("candidate", merged)):
                    with (output / f"stage-{size:03d}.{arm}.episodes.jsonl").open("x") as stream:
                        def record(index, result):
                            nonlocal completed_episodes
                            completed_episodes += 1
                            observed[arm].append(result)
                            stream.write(json.dumps({"index": index, "result": result,
                                                     "elapsed_seconds": time.monotonic() - started},
                                                    allow_nan=False) + "\n")
                            stream.flush()
                        def on_start():
                            nonlocal started_episodes
                            started_episodes += 1
                        cumulative[arm].extend(await _collect_arm(
                            rows, seeds, sampler=sampler, patch=spec,
                            concurrency=concurrency, record=record, on_start=on_start))
                    write_json(output / f"stage-{size:03d}.{arm}.json",
                               {"results": cumulative[arm], **_hook_stats(cumulative[arm])})
                evidence = paired_gain(cumulative["incumbent"], cumulative["candidate"], seed=seed)
                gate = evidence.pop("accepted")
                stage.update(**evidence, ci_gain_gate_passed=gate,
                             status=("promising" if gate else "rejected"
                                     if evidence["paired_bootstrap_ci95"][1] < 0 else "inconclusive"))
            finally:
                stage.update(episodes_completed=completed_episodes - stage_episode_start,
                             episodes_started=started_episodes - stage_attempt_start,
                             cumulative_episodes_completed=completed_episodes,
                             elapsed_seconds=time.monotonic() - stage_started,
                             hook_stats_by_arm={arm: _partial_hook_stats(results)
                                                for arm, results in observed.items()})
                write_json(output / f"stage-{size:03d}.json", stage)
            if runtime_source_identity() != identity["runtime_source_sha256"]:
                raise RuntimeError("Frozen operational sources changed during screen")
            if stage["status"] == "rejected":
                return finish("rejected", "paired_question_ci_indicates_harm")
            previous_count = size
        return finish(stages[-1]["status"], "train_screen_requires_full_dev_confirmation")
    except CandidateExecutionRejected as exc:
        return finish("rejected", f"hook_runtime_error: {exc}")
    except Exception as exc:
        write_json(output / "ABORTED.json", {
            **identity, "status": "aborted", "error_type": type(exc).__name__,
            "error": str(exc)[:1000], "episodes_completed": completed_episodes,
            "episodes_started": started_episodes,
            "stages": stages, "elapsed_seconds": time.monotonic() - started,
            "hook_stats_by_arm": {arm: _partial_hook_stats(results)
                                 for arm, results in observed.items()},
        })
        raise
    finally:
        if sampler is not None:
            await sampler.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-urls", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--excluded-packets", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retrieval-url")
    parser.add_argument("--target-model-sha256")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--max-questions", type=int, choices=STAGES, default=128)
    args = vars(parser.parse_args())
    args["candidate"] = extract_patch(args["candidate"].read_text(), "json_patch")
    args["incumbent"] = _normalize_incumbent(json.loads(args["incumbent"].read_text()))
    asyncio.run(run_screen(**args))


if __name__ == "__main__":
    cli()
