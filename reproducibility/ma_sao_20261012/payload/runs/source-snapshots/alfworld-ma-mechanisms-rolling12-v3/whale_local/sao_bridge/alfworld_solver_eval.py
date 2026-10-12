"""Native ALFWorld evaluation, with an explicit fixed-harness matrix mode."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
from pathlib import Path
import re
import shutil
import threading
import time

from .alfworld_domain import INITIAL_SOURCE, harness_spec
from .common import file_sha, read_json, write_json
from .evaluation import Sampler, evaluate
from .gpu_telemetry import monitor, summarize
from .launch import Launcher
from .resources import gpu_release_audit, stop_known


def bind_environment(launch: Launcher, controller_url: str, timeout: float) -> None:
    values = {
        "SAO_ALFWORLD_CONTROLLER_URL": controller_url,
        "SAO_ALFWORLD_HTTP_TIMEOUT": str(timeout),
    }
    launch.env.update(values)
    os.environ.update(values)


def validate_config(config: dict, *, harness_matrix: bool = False) -> dict:
    required = {
        "run", "solver_model", "solver_version", "task_ids", "no_patch",
        "seed",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing ALFWorld Solver evaluation keys: {sorted(missing)}")
    if "harnesses" in config and not harness_matrix:
        raise ValueError("Non-default harnesses require explicit --harness-matrix mode")
    normalized = dict(config)
    model = Path(normalized["solver_model"]).resolve()
    no_patch = Path(normalized["no_patch"]).resolve()
    if not model.is_dir() or not list(model.glob("*.safetensors")):
        raise FileNotFoundError("ALFWorld Solver checkpoint has no safetensors")
    if not no_patch.is_file():
        raise FileNotFoundError("ALFWorld no-patch artifact is unavailable")
    patch = read_json(no_patch)
    if (
        not isinstance(patch, dict)
        or patch.get("benchmark") != "alfworld"
        or patch.get("actions") != []
    ):
        raise ValueError("ALFWorld Solver evaluation requires the empty no-patch harness")
    task_ids = normalized["task_ids"]
    if (
        not isinstance(task_ids, list)
        or not task_ids
        or len(task_ids) != len(set(task_ids))
        or any(type(task_id) is not int or not 0 <= task_id < 500 for task_id in task_ids)
    ):
        raise ValueError("task_ids must be distinct official ALFWorld IDs")
    normalized.setdefault("samples", 1)
    normalized.setdefault("solver_tokens", 4096)
    normalized.setdefault("temperature", 1.0)
    normalized.setdefault("eval_concurrency", 32)
    normalized.setdefault("environment_concurrency", 32)
    normalized.setdefault("monitor_interval_seconds", 5)
    if harness_matrix:
        normalized.setdefault("matrix_concurrency", 1)
        if (
            type(normalized["matrix_concurrency"]) is not int
            or normalized["matrix_concurrency"] < 1
        ):
            raise ValueError("matrix_concurrency must be a positive integer")
    if type(normalized.get("episode_checkpointing", False)) is not bool:
        raise ValueError("Episode checkpointing must be explicitly boolean")
    if normalized.get("resume_matrix") and (not harness_matrix or not normalized.get("episode_checkpointing")):
        raise ValueError("Partial matrix recovery requires explicit per-episode checkpointing")
    priority = normalized.get("priority_harness_labels", [])
    if (not isinstance(priority, list) or any(not isinstance(label, str) for label in priority)
            or len(priority) != len(set(priority))
            or set(priority) - {"no_patch", *normalized.get("harnesses", {})}):
        raise ValueError("Primary-first evaluation can only prioritize existing unique Harness labels")
    for key in (
        "seed", "samples", "solver_tokens", "eval_concurrency",
        "environment_concurrency",
    ):
        if type(normalized[key]) is not int or normalized[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    normalized.update(
        run=str(Path(normalized["run"]).resolve()),
        solver_model=str(model),
        no_patch=str(no_patch),
    )
    return normalized


def matrix_harnesses(config: dict) -> tuple[dict, dict]:
    requested = config.get("harnesses")
    if not isinstance(requested, dict) or not requested:
        raise ValueError("Harness matrix requires a nonempty label-to-path mapping")
    if "no_patch" in requested:
        raise ValueError("The no_patch matrix label is reserved for the common control")
    unique, aliases = {}, {}
    for label, path in {"no_patch": config["no_patch"], **requested}.items():
        if not isinstance(label, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", label,
        ):
            raise ValueError("Harness matrix labels must be safe, nonempty identifiers")
        if not isinstance(path, str):
            raise ValueError("Harness matrix paths must be strings")
        source = Path(path).resolve()
        spec = harness_spec(source.read_text())
        identity = hashlib.sha256(spec["source"].rstrip("\n").encode()).hexdigest()
        unique.setdefault(identity, {"patch": spec["patch"]})
        aliases[label] = {"sha256": identity, "source_path": str(source)}
    return unique, aliases


def reuse_completed_reports(config: dict, root: Path, assignments: list[dict]) -> dict:
    if not config.get("resume_matrix"):
        return {}
    from .alfworld_strong_phase import checked_native_report, checked_release

    manifest = read_json(Path(config["resume_matrix"]))
    previous = Path(manifest["previous_attempt"])
    if (file_sha(previous / "config.json") != manifest["previous_config_sha256"]
            or file_sha(previous / "FAILED.json") != manifest["previous_failure_sha256"]
            or "ReadError" not in read_json(previous / "FAILED.json")["error"]
            or manifest["reason"] != "inference-http-read-error"):
        raise ValueError("Matrix recovery needs the preserved input-bound transport failure")
    checked_release(previous)
    old = read_json(previous / "config.json")
    execution_only = {"run", "evaluation_mode", "episode_checkpointing", "resume_matrix"}
    if {key: value for key, value in old.items() if key not in execution_only} != {
        key: value for key, value in config.items() if key not in execution_only
    }:
        raise ValueError("Partial matrix recovery changed the original evaluation inputs")
    complete = manifest["complete_reports"]
    actual = {path.stem for path in (previous / "reports").glob("*.json")}
    known = {row["harness_sha256"] for row in assignments}
    if set(complete) != actual or set(complete) - known:
        raise ValueError("Recovery omitted a completed report or changed its Harness identity")
    reused = {}
    for identity, metadata in complete.items():
        path = Path(metadata["path"])
        if (path.resolve() != (previous / "reports" / f"{identity}.json").resolve()
                or file_sha(path) != metadata["sha256"]):
            raise ValueError("A previously completed policy report changed")
        report = checked_native_report(read_json(path), config["task_ids"], config)
        destination = root / "reports" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha(destination) != metadata["sha256"]:
            raise ValueError("Recovery tried to replace a different complete report")
        if not destination.exists():
            shutil.copyfile(path, destination)
        reused[identity] = report
    journals = manifest.get("episode_journal_files", {})
    old_journals = previous / "episode_journals"
    actual_journals = {str(path.relative_to(old_journals)) for path in old_journals.glob("*/*.json")}
    if set(journals) != actual_journals:
        raise ValueError("Recovery omitted committed episode journal evidence")
    for relative, expected in journals.items():
        origin = old_journals / relative
        if (not origin.resolve().is_relative_to(old_journals.resolve())
                or file_sha(origin) != expected):
            raise ValueError("A source episode journal changed or escaped its attempt")
        destination = root / "episode_journals" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and file_sha(destination) != expected:
            raise ValueError("Recovery tried to overwrite newer episode evidence")
        if not destination.exists():
            shutil.copyfile(origin, destination)
    write_json(root / "REUSED_REPORTS.json", {
        "source_manifest": config["resume_matrix"], "complete_reports": complete,
        "reused_complete_episodes": len(complete) * len(config["task_ids"]) * config["samples"],
        "episode_journal_files": journals,
        "rule": "Keep every complete report; no score-based retry or candidate replacement.",
    })
    return reused


async def evaluate_harnesses(
    urls: str, config: dict, assignments: list[dict], root: Path,
    aliases: dict | None = None,
) -> dict:
    if not assignments or (aliases is None and len(assignments) != 1):
        raise ValueError("Multiple harness evaluations require explicit matrix aliases")
    reused = reuse_completed_reports(config, root, assignments)
    sampler = Sampler(
        urls,
        concurrency=config["eval_concurrency"],
        temperature=float(config["temperature"]),
        chat_template_kwargs={"enable_thinking": False},
        request_timeout=float(config.get("request_timeout_seconds", 900)),
    )
    reports = {}
    semaphore = asyncio.Semaphore(config.get("matrix_concurrency", 1))

    async def evaluate_assignment(assignment):
        async with semaphore:
            rows = [{
                "id": f"alfworld-{task_id}",
                "task_id": task_id,
                "task_name": assignment["name"],
            } for task_id in config["task_ids"]]
            identity = assignment.get("harness_sha256", "no_patch")
            if identity in reused:
                report = reused[identity]
            else:
                journal = {}
                if config.get("episode_checkpointing"):
                    journal["episode_journal"] = root / "episode_journals" / identity
                report = await evaluate(
                    sampler, rows,
                    Path(assignment["harness"]).read_text() if aliases is not None else INITIAL_SOURCE,
                    tokens=config["solver_tokens"],
                    samples=config["samples"],
                    seed=config["seed"],
                    domain="alfworld",
                    episode_concurrency=config["environment_concurrency"],
                    **journal,
                )
            if aliases is not None:
                identity = assignment["harness_sha256"]
                path = root / "reports" / f"{identity}.json"
                if identity not in reused:
                    write_json(path, report)
                reports[identity] = {
                    "accuracy": report["accuracy"], "report": str(path),
                    "harness_sha256": identity,
                }
                write_json(root / "PROGRESS.json", {
                    "completed_unique_harnesses": len(reports),
                    "total_unique_harnesses": len(assignments),
                    "updated_at_unix": time.time(),
                })
            return report

    priority = {aliases[label]["sha256"] for label in config.get("priority_harness_labels", [])} if aliases else set()
    ordered = sorted(assignments, key=lambda assignment: assignment.get("harness_sha256") not in priority)
    tasks = [asyncio.create_task(evaluate_assignment(assignment)) for assignment in ordered]
    try:
        results = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await sampler.close()
    if aliases is None:
        return results[0]
    return {
        "metric": "paired fixed-harness matrix",
        "unique_harnesses": len(reports),
        "logical_harnesses": len(aliases),
        "samples_per_question": config["samples"],
        "question_count": len(config["task_ids"]),
        "variants": {
            label: {**reports[value["sha256"]], "source_path": value["source_path"]}
            for label, value in aliases.items()
        },
        "duplicate_policy": (
            "Identical canonical patches share a rollout report; all labels remain. "
            "Task/sample decode seeds are identical across distinct harnesses."
        ),
    }


def run(config_path: Path, *, harness_matrix: bool = False) -> dict:
    config = validate_config(read_json(config_path), harness_matrix=harness_matrix)
    unique, aliases = matrix_harnesses(config) if harness_matrix else ({}, None)
    root = Path(config["run"])
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    if harness_matrix:
        config["evaluation_mode"] = "frozen_harness_matrix"
    write_json(root / "config.json", config)
    write_json(root / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(root / "supervision")
    telemetry_stop = threading.Event()
    telemetry_path = root / "gpu_process_samples.jsonl"
    telemetry = threading.Thread(
        target=monitor,
        args=(telemetry_path, telemetry_stop, config["monitor_interval_seconds"]),
        daemon=True,
    )
    telemetry.start()
    started = time.monotonic()
    try:
        version_id = hashlib.sha256(
            config["solver_version"].encode()
        ).hexdigest()[:12]
        task_name = f"solver-eval-{version_id}"
        assignments = [{
            "name": task_name,
            "task_ids": config["task_ids"],
            "harness": config["no_patch"],
        }]
        if aliases is not None:
            assignments = []
            for index, (identity, value) in enumerate(unique.items()):
                snapshot = root / "harnesses" / f"{identity}.json"
                write_json(snapshot, value["patch"])
                assignments.append({
                    "name": f"{task_name}-h{index:02d}",
                    "task_ids": config["task_ids"],
                    "harness": str(snapshot),
                    "harness_sha256": identity,
                })
            write_json(root / "harness_matrix.json", {
                "aliases": aliases,
                "assignments": assignments,
                "solver_model": config["solver_model"],
                "solver_version": config["solver_version"],
            })
        assignments_path = root / "assignments.json"
        write_json(assignments_path, assignments)
        env_process = launch.module(
            "sao_bridge.alfworld_env_service",
            [
                "--run", root / "environment",
                "--assignments", assignments_path,
                "--controller-port", config.get("controller_port", 28000),
                "--worker-port", config.get("worker_port", 28100),
                "--concurrency", config["environment_concurrency"],
            ] + (["--harness-matrix", "--async-controller"] if harness_matrix else []),
            gpus="", name="environment",
        )
        controller_url = (
            f"http://127.0.0.1:{config.get('controller_port', 28000)}/api"
        )
        launch.ready(env_process, controller_url + "/list_workers")
        ready = root / "environment/READY.json"
        deadline = time.monotonic() + float(config.get("startup_timeout_seconds", 900))
        while not ready.is_file():
            launch.check()
            if env_process.poll() is not None:
                raise RuntimeError("ALFWorld environment exited before readiness")
            if time.monotonic() >= deadline:
                raise TimeoutError("ALFWorld environment readiness timed out")
            time.sleep(1)
        bind_environment(
            launch,
            controller_url,
            config.get("environment_http_timeout", 900),
        )
        with launch.vllm(
            config["solver_model"],
            port=config.get("solver_port", 19801),
            gpus="0,1,2,3,4,5,6,7",
            name="solver",
            data_parallel=8,
            max_model_len=32768,
            tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton",
            language_model_only=True,
            max_num_seqs=max(8, config["eval_concurrency"] // 8),
            server_args=["--enable-prefix-caching"],
        ) as urls:
            report = asyncio.run(evaluate_harnesses(
                urls, config, assignments, root, aliases,
            ))
        write_json(root / "report.json", report)
        complete = {
            "status": "completed",
            "solver_model": config["solver_model"],
            "solver_version": config["solver_version"],
            "task_ids": config["task_ids"],
            "seed": config["seed"],
            "samples": config["samples"],
            "elapsed_seconds": time.monotonic() - started,
            "test_split_opened": any(task_id >= 400 for task_id in config["task_ids"]),
        }
        if aliases is None:
            complete["accuracy"] = report["accuracy"]
        else:
            complete.update(
                evaluation_mode="frozen_harness_matrix",
                variants=report["variants"],
                unique_harnesses=report["unique_harnesses"],
                logical_harnesses=report["logical_harnesses"],
            )
        write_json(root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {
            "status": "failed",
            "error": repr(exc),
            "elapsed_seconds": time.monotonic() - started,
        })
        raise
    finally:
        launch.close()
        telemetry_stop.set()
        telemetry.join(timeout=config["monitor_interval_seconds"] + 10)
        if telemetry_path.is_file():
            try:
                telemetry_summary = summarize(telemetry_path)
            except ValueError as exc:
                telemetry_summary = {
                    "status": "invalid",
                    "error": repr(exc),
                }
            write_json(root / "gpu_utilization_summary.json", telemetry_summary)
        write_json(root / "RELEASED.json", {
            "released_at": time.time(),
            **gpu_release_audit(),
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--harness-matrix", action="store_true",
        help="Evaluate explicit harnesses with one shared Solver serving pool",
    )
    args = parser.parse_args()
    print(run(args.config, harness_matrix=args.harness_matrix), flush=True)


if __name__ == "__main__":
    main()
