"""Matched Engineer proposals scored by the official sparse-task ALFWorld evaluator."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

from .analysis import paired_difference
from .alfworld_runtime import ALFWORLD_DATA, prepare_agentbench_runtime
from .alfworld_train_data import _immutable_json, evaluation_paths, goal_group
from .budget import contract_digest
from .common import ROOT, ensure_model_runtime_metadata, file_sha, read_json, write_json
from .gpu_telemetry import monitor as monitor_gpu_processes, summarize as summarize_gpu_processes
from .launch import Launcher
from .resources import gpu_release_audit
from harness_r1_alfworld.driver import baseline_command, checked_rows, cpu_env, run as run_official


UPSTREAM = ROOT / "third_party/harness-r1-upstream"
ALFWORLD_PYTHON = Path("/dev/shm/harness-r1-alfworld-venv/bin/python")


class EvaluationAttemptsExhausted(RuntimeError):
    pass


def official_modules():
    sys.path.insert(0, str(UPSTREAM / "code/Relax"))
    from examples.harness_r1 import eval_alfworld_patches, reward_alfworld_patch

    return eval_alfworld_patches, reward_alfworld_patch


def reserve_ports(reward, job: dict):
    os.environ["HARNESS_R1_PORT_LOCK_DIR"] = job["port_locks"]
    os.environ["HARNESS_R1_AVOID_EPHEMERAL_PORTS"] = "1"
    return reward._select_free_port_pair(
        controller_base=12000, worker_base=22000, num_workers=1, span=4000,
        seed=int(contract_digest(job["request"])[:8], 16) + job["attempt"] * 104729,
    )


def normalize_rewards(values: dict, task_ids: list[int]) -> dict[int, float]:
    if not task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Expected a nonempty, distinct evaluation cohort")
    rewards = {int(key): float(value) for key, value in values.items()}
    if len(rewards) != len(values) or any(
        not math.isfinite(value) or value not in (0.0, 1.0) for value in rewards.values()
    ):
        raise ValueError("ALFWorld evaluation rewards must be complete and binary")
    if set(rewards) == set(task_ids):
        return rewards
    if set(rewards) == set(range(len(task_ids))):
        return {task_ids[index]: rewards[index] for index in range(len(task_ids))}
    raise ValueError("Sparse ALFWorld evaluation returned the wrong task identities")


def checked_scores(output: Path, task_ids: list[int]) -> dict[int, float]:
    raw_ids = set()
    for path in output.glob("**/runs.jsonl"):
        for line in path.read_text().splitlines():
            if line.strip():
                raw_ids.add(json.loads(line)["index"])
    if raw_ids == set(task_ids):
        expected = task_ids
    elif raw_ids == set(range(len(task_ids))):
        expected = list(range(len(task_ids)))
    else:
        raise ValueError("Official rollout did not cover the exact sparse cohort")
    rows = checked_rows(output, expected)
    scores = {}
    for row in rows:
        result = row["output"]["result"]
        if "reward" not in result:
            raise ValueError("Official rollout omitted its environment reward")
        scores[row["index"]] = result["reward"]
    return normalize_rewards(scores, task_ids)


def proposal_report(
    proposals: list[dict], scores: dict[str, dict], task_ids: list[int],
    baseline: dict[int, float],
) -> dict:
    if [row["sample_index"] for row in proposals] != list(range(8)):
        raise ValueError("Engineer evaluation must retain all eight matched proposals")
    rows = []
    accuracies = []
    for proposal in proposals:
        if proposal["valid"]:
            rewards = normalize_rewards(scores[proposal["patch_sha256"]], task_ids)
        else:
            rewards = {task_id: baseline[task_id] for task_id in task_ids}
        accuracies.append(sum(rewards.values()) / len(task_ids))
        rows.extend({
            "id": f"alfworld-{task_id:03d}",
            "task_id": task_id,
            "sample_index": proposal["sample_index"],
            "rollout_seed": proposal["decode_seed"],
            "reward": rewards[task_id],
            "valid_proposal": proposal["valid"],
        } for task_id in task_ids)
    accuracy = sum(row["reward"] for row in rows) / len(rows)
    no_patch = sum(baseline[index] for index in task_ids) / len(task_ids)
    return {
        "metric": "mean@8", "samples_per_question": 8, "question_count": len(task_ids),
        "accuracy": accuracy, "no_patch_accuracy": no_patch,
        "mean_gain_vs_no_patch": accuracy - no_patch,
        "proposal_accuracies": accuracies,
        "valid_proposals": sum(row["valid"] for row in proposals),
        "results": rows,
        "sample_axis": "Engineer proposal; rollout_seed denotes the Engineer decode seed",
        "invalid_proposal_policy": "no-patch baseline, counted in the eight-proposal denominator",
    }


def generation_args(request: dict) -> SimpleNamespace:
    return SimpleNamespace(
        engineer_model="policy", engineer_assistant_prefill="<think>\n",
        engineer_temperature=request["temperature"], engineer_top_p=1.0,
        engineer_max_tokens=request["tokens"],
        engineer_chat_template_kwargs={"enable_thinking": True},
        engineer_reasoning_effort="",
        engineer_extra_body={"add_generation_prompt": False, "seed": request["decode_seed"]},
        engineer_base_url=request["url"], engineer_api_key="EMPTY",
        engineer_use_env_proxy=False, engineer_timeout=900, engineer_retries=0,
        engineer_retry_sleep=5.0, engineer_retry_max_sleep=30.0,
        response_protocol="prefill_think_patch",
    )


def validate_job_task_ids(request: dict) -> list[int]:
    task_ids = request["task_ids"]
    purpose = request.get("purpose", "evaluation")
    if purpose == "training_evidence" and request["kind"] == "baseline":
        lower, upper = 0, 240
    elif purpose == "evaluation":
        lower, upper = 240, 400
    else:
        raise ValueError("Only explicit baseline jobs may collect training evidence")
    if (
        not isinstance(task_ids, list) or not task_ids
        or any(type(index) is not int or not lower <= index < upper for index in task_ids)
        or task_ids != sorted(set(task_ids))
    ):
        raise ValueError("Evaluation/training jobs must retain their sorted, unique task partition")
    return task_ids


def run_job(path: Path) -> dict:
    job = read_json(path)
    request = job["request"]
    output = Path(job["output"])
    output.mkdir(parents=True, exist_ok=True)
    evaluator, reward = official_modules()
    port_locks = []
    try:
        if request["kind"] == "generate":
            args = generation_args(request)
            response = evaluator.call_chat(args, request["messages"])
            write_json(output / "response.json", response)
            message = response["choices"][0]["message"]
            text = evaluator.message_text(message, "prefill_think_patch")
            (output / "response.txt").write_text(text)
            patch, error = reward._parse_patch(
                text, response_protocol="prefill_think_patch", require_code_hook_only=True,
            )
            if patch is not None:
                incompatible = reward._alfworld_runtime_incompatible_reason(patch)
                if incompatible:
                    patch, error = None, incompatible
            result = {
                "sample_index": request["sample_index"], "decode_seed": request["decode_seed"],
                "prompt_id": request["prompt_id"], "model_label": request["model_label"],
                "valid": patch is not None, "model_failure": error or None,
                "patch": patch, "patch_sha256": contract_digest(patch) if patch is not None else None,
            }
        else:
            task_ids = validate_job_task_ids(request)
            runtime = output / "runtime"
            agentbench = prepare_agentbench_runtime(runtime / "AgentBench")
            os.environ.update(cpu_env(runtime))
            controller, worker, port_locks = reserve_ports(reward, job)
            write_json(output / "ports.json", {
                "controller": controller, "worker": worker,
                "ephemeral_range_excluded": True, "file_locks_held": True,
            })
            if request["kind"] == "baseline":
                write_json(output / "task_ids.json", task_ids)
                args = SimpleNamespace(
                    python=ALFWORLD_PYTHON, run_id=request["name"],
                    size=len(task_ids), concurrency=request["concurrency"],
                    controller_port=controller, worker_port=worker,
                    target_url=request["url"], target_model="policy",
                    target_agent_name="qwen35-9b-nothink",
                )
                run_official(
                    baseline_command(args, output, agentbench), output / "baseline.log", runtime,
                )
                scores = checked_scores(output / "baseline", task_ids)
            elif request["kind"] == "score":
                baseline = normalize_rewards(request["baseline"], task_ids)
                metadata = {
                    "batch_tag": "cohort", "split": "test_set", "start": min(task_ids),
                    "end": max(task_ids) + 1, "task_ids": task_ids, "batch_size": len(task_ids),
                    "baseline_pass": int(sum(baseline.values())),
                    "baseline_rewards": baseline, "reward_threshold": 1.0,
                }
                write_json(output / "patch.json", request["patch"])
                args = SimpleNamespace(
                    run_id=request["name"], agentbench_dir=agentbench,
                    agentbench_python=ALFWORLD_PYTHON, alfworld_worker_python=ALFWORLD_PYTHON,
                    alfworld_split="test_set", alfworld_max_step=50,
                    controller_port_base=controller, worker_port_base=worker,
                    rollout_concurrency=request["concurrency"], rollout_auto_retry=True,
                    startup_timeout=300, rollout_timeout=1800, eval_timeout=2400,
                    target_base_url=request["url"], target_model="policy",
                    target_agent_name="qwen35-9b-nothink",
                    rollout_max_tokens=4096, rollout_tool_choice="auto",
                    rollout_chat_template_kwargs={"enable_thinking": False},
                    rollout_disable_parallel_tool_calls=False, rollout_single_tool_call_only=False,
                )
                evaluated = evaluator.evaluate_patch(
                    args, {"metadata": metadata}, output / "patch.json", output, 0,
                )
                write_json(output / "official_result.json", evaluated)
                if evaluated["eval_status"] != "ok":
                    raise RuntimeError(
                        f"Official patch evaluation failed, not a model score: {evaluated}"
                    )
                scores = normalize_rewards(evaluated["patched_rewards"], task_ids)
                if checked_scores(Path(evaluated["eval_root"]), task_ids) != scores:
                    raise ValueError("Official summary differs from its checked rollout evidence")
            else:
                raise ValueError(f"Unknown Engineer evaluation job: {request['kind']}")
            result = {"scores": scores}
        complete = {
            "status": "complete", "request_sha256": contract_digest(request), **result,
        }
        write_json(output / "result.json", complete)
        return complete
    except BaseException as exc:
        write_json(output / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
        raise
    finally:
        reward._release_port_locks(port_locks)


def retry_credits(root: Path) -> dict:
    recovery = root / "infrastructure_recovery.json"
    record = read_json(recovery) if recovery.exists() else None
    credits = record["additional_attempts"] if record is not None else {}
    if any(type(value) is not int or value != 2 for value in credits.values()):
        raise ValueError("Infrastructure recovery permits exactly two additional attempts per named job")
    identity = root / "identity.json"
    if identity.exists() and read_json(identity).get("infrastructure_recovery_sha256") != (
        file_sha(recovery) if recovery.exists() else None
    ):
        raise ValueError("Infrastructure recovery authorization changed")
    if record is not None:
        for relative, digest in record["preserved_file_sha256"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha(path) != digest:
                raise ValueError(f"Successful pre-recovery evaluation evidence changed: {relative}")
    return credits


class Jobs:
    def __init__(self, root: Path, launch: Launcher, concurrency: int):
        if type(concurrency) is not int or concurrency < 1:
            raise ValueError("Evaluation concurrency must be a positive integer")
        self.root = root
        self.launch = launch
        self.concurrency = concurrency
        self.pending = deque()
        self.active = []
        self.results = {}
        self.requests = {}
        self.retry_credits = retry_credits(root.parent)

    def add(self, request: dict) -> None:
        name = request["name"]
        if name in self.requests or not name.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Evaluation job names must be distinct safe path components")
        self.requests[name] = request
        root = self.root / name
        _immutable_json(root / "request.json", request)
        result = root / "result.json"
        if result.exists():
            self._accept(request, read_json(result))
            return
        attempts_path = root / "attempts.json"
        attempts = read_json(attempts_path) if attempts_path.exists() else []
        for index in range(len(attempts)):
            recovered = root / f"attempt-{index + 1}" / "result.json"
            if recovered.exists():
                value = read_json(recovered)
                self._accept(request, value)
                write_json(result, value)
                return
        self.pending.append(request)

    def _accept(self, request: dict, result: dict) -> None:
        if (
            result.get("status") != "complete"
            or result.get("request_sha256") != contract_digest(request)
        ):
            raise ValueError("Cached evaluation job identity or completion status changed")
        self.results[request["name"]] = result

    def tick(self) -> None:
        self.launch.check()
        for record in self.active[:]:
            process, request, attempt, started = record
            if process.poll() is None:
                if time.monotonic() - started > 3000:
                    raise TimeoutError(f"Engineer evaluation job timed out: {request['name']}")
                continue
            self.active.remove(record)
            root = self.root / request["name"]
            attempts = read_json(root / "attempts.json")
            result_path = root / f"attempt-{attempt}" / "result.json"
            if process.returncode == 0 and result_path.exists():
                result = read_json(result_path)
                self._accept(request, result)
                write_json(root / "result.json", result)
                attempts[-1].update(finished_at_unix=time.time(), complete=True)
            else:
                attempts[-1].update(
                    finished_at_unix=time.time(), returncode=process.returncode,
                    error="job failed or omitted its completion receipt",
                )
                self.pending.append(request)
                print(f"Retrying failed evaluation job {request['name']} after attempt {attempt}", flush=True)
            write_json(root / "attempts.json", attempts)
        while self.pending and len(self.active) < self.concurrency:
            request = self.pending.popleft()
            root = self.root / request["name"]
            attempts_path = root / "attempts.json"
            attempts = read_json(attempts_path) if attempts_path.exists() else []
            limit = 2 + self.retry_credits.get(request["name"], 0)
            if len(attempts) >= limit:
                raise EvaluationAttemptsExhausted(
                    f"Evaluation job exhausted {limit} attempts: {request['name']}"
                )
            attempt = len(attempts) + 1
            output = root / f"attempt-{attempt}"
            job_path = root / f"job-{attempt}.json"
            write_json(job_path, {
                "request": request, "output": str(output), "attempt": attempt,
                "port_locks": str(self.root.parent / "port_locks"),
            })
            attempts.append({"started_at_unix": time.time(), "attempt": attempt})
            write_json(attempts_path, attempts)
            process = self.launch.module(
                "sao_bridge.alfworld_engineer_eval", ["--job", job_path],
                gpus="", name=f"{request['name']}-attempt-{attempt}",
            )
            self.active.append((process, request, attempt, time.monotonic()))

    def wait(self) -> dict:
        while self.pending or self.active:
            self.tick()
            if self.pending or self.active:
                time.sleep(1)
        return self.results


def validate_spec(path: Path) -> tuple[dict, dict, list[dict]]:
    spec = read_json(path)
    root = Path(spec["run"]).resolve()
    training = Path(spec["training_root"]).resolve()
    if root.parent != training.parent or root.name != "evaluation" or training.name != "training":
        raise ValueError("Engineer evaluation must be a sibling of its frozen training phase")
    if read_json(root.parent / "evaluation_protocol.json") != spec:
        raise ValueError("Engineer evaluation differs from its pre-training protocol")
    launch_receipt = read_json(root.parent / "launch_receipt.json")
    training_identity = read_json(root.parent / "training_identity.json")
    training_config = training_identity["config"]
    if (
        spec["initial_engineer"] != training_config["engineer_model"]
        or spec["solver_model"] != training_config["solver_model"]
        or spec["solver_version"] != training_config["solver_version"]
        or spec["seed"] != training_config["seed"]
        or file_sha(Path(spec["initial_engineer"]) / "provenance.json")
        != training_identity["engineer_provenance_sha256"]
    ):
        raise ValueError("Engineer evaluation changed the training initialization")
    manifest_path = Path(spec["data_manifest"])
    if file_sha(manifest_path) != launch_receipt["data_manifest_sha256"]:
        raise ValueError("Engineer data manifest changed after training launch")
    manifest = read_json(manifest_path)
    for name, digest in manifest["files"].items():
        if file_sha(manifest_path.parent / name) != digest:
            raise ValueError(f"Frozen Engineer dataset changed: {name}")
    prompts = read_json(Path(spec["evaluation_prompts"]))
    if (
        len(prompts) != spec["candidates_per_model"] or len(prompts) != 8
        or not spec["keep_invalid_and_duplicate_candidates_in_mean"]
        or not spec["reuse_candidates_for_report"]
        or spec["target_temperature"] != 0
        or spec["target_tokens_per_turn"] != 4096 or spec["target_max_steps"] != 50
        or len(spec["checkpoint_steps"]) != 2 or len(set(spec["checkpoint_steps"])) != 2
    ):
        raise ValueError("Unsupported matched Engineer evaluation budget or semantics")
    source_packets_path = Path(manifest["source_packets"])
    if file_sha(source_packets_path) != manifest["source_packets_sha256"]:
        raise ValueError("Original Engineer packet source changed")
    source_packets = read_json(source_packets_path)
    original = Path(source_packets[0]["metadata"]["source_run_root"]).parent
    identity = read_json(original / "baseline_identity.json")
    if file_sha(original / "manifest.json") != identity["controls"]["manifest.json"]:
        raise ValueError("Original Engineer benchmark identity changed")
    catalogue_digest = read_json(original / "manifest.json")["split_sha256"]
    if file_sha(ALFWORLD_DATA / "test_set.json") != catalogue_digest:
        raise ValueError("Engineer evaluation catalogue differs from its frozen training evidence")
    paths = evaluation_paths()
    train_goals = {goal_group(paths[index]) for index in manifest["training_task_ids"]}
    groups = []
    for key in ("selection_task_ids", "report_task_ids"):
        ids = manifest[key]
        goals = {goal_group(paths[index]) for index in ids}
        if len(goals) != len(ids) or goals & train_goals or any(not 240 <= i < 400 for i in ids):
            raise ValueError("Engineer evaluation is not goal-disjoint from training")
        groups.append(goals)
    if groups[0] & groups[1]:
        raise ValueError("Engineer selection and report goals overlap")
    model_identity = read_json(Path(spec["solver_identity_receipt"]))
    provenance = Path(spec["solver_model"]) / "provenance.json"
    if (
        file_sha(provenance) != model_identity["provenance_sha256"]
        or not spec["solver_version"].endswith(model_identity["provenance_sha256"][:12])
        or not model_identity["matches_packet_solver_version"]
        or model_identity != launch_receipt["solver_identity_check"]
    ):
        raise ValueError("Engineer evaluation changed the fixed Solver identity")
    return spec, manifest, prompts


def _status(root: Path, stage: str, **extra) -> None:
    write_json(root / "STATUS.json", {
        "stage": stage, "updated_at_unix": time.time(), **extra,
    })


def _model_options(model: str, *, port: int, gpus: str, name: str, replicas: int, context: int):
    return {
        "model": model, "port": port, "gpus": gpus, "name": name,
        "data_parallel": replicas, "max_model_len": context,
        "tool_call_parser": "qwen3_xml", "gdn_prefill_backend": "triton",
        "language_model_only": True, "max_num_seqs": 32,
        "server_args": ["--enable-prefix-caching"],
    }


def _score_models(
    root: Path, launch: Launcher, spec: dict, proposals: dict[str, list[dict]],
    labels: list[str], task_ids: list[int], baseline: dict[int, float], urls: list[str],
    stage: str,
) -> dict[str, dict]:
    task_ids = sorted(task_ids)
    unique = {
        proposal["patch_sha256"]: proposal["patch"]
        for label in labels for proposal in proposals[label] if proposal["valid"]
    }
    jobs = Jobs(root / "jobs", launch, spec["candidate_evaluation_concurrency"])
    names = {}
    for ordinal, digest in enumerate(sorted(unique)):
        name = f"{stage}-{digest[:24]}"
        names[digest] = name
        jobs.add({
            "name": name, "kind": "score", "task_ids": task_ids,
            "baseline": {str(index): baseline[index] for index in task_ids},
            "patch": unique[digest], "url": urls[ordinal % len(urls)],
            "concurrency": spec["target_rollout_concurrency"],
            "port_index": 20 + ordinal + (40 if stage == "report" else 0),
        })
    results = jobs.wait()
    scores = {digest: results[name]["scores"] for digest, name in names.items()}
    reports = {
        label: proposal_report(proposals[label], scores, task_ids, baseline) for label in labels
    }
    for label, report in reports.items():
        write_json(root / stage / f"{label}.json", report)
    return reports


def run(spec_path: Path) -> dict:
    spec, manifest, prompts = validate_spec(spec_path)
    root = Path(spec["run"])
    root.mkdir(parents=True, exist_ok=True)
    source_files = [
        Path(__file__),
        *(Path(__file__).with_name(name) for name in (
            "analysis.py", "alfworld_runtime.py", "alfworld_train_data.py",
            "budget.py", "common.py", "launch.py", "resources.py", "gpu_telemetry.py",
        )),
        ROOT / "whale_local/harness_r1_alfworld/driver.py",
        UPSTREAM / "code/Relax/examples/harness_r1/eval_alfworld_patches.py",
        UPSTREAM / "code/Relax/examples/harness_r1/reward_alfworld_patch.py",
        UPSTREAM / "code/life-harness/AgentBench/scripts/harness_r1_patch.py",
        UPSTREAM / "code/life-harness/AgentBench/scripts/harness_r1_batch_debug.py",
    ]
    frozen = {
        "spec": spec, "data_manifest_sha256": file_sha(Path(spec["data_manifest"])),
        "solver_identity_sha256": file_sha(Path(spec["solver_identity_receipt"])),
        "evaluation_catalogue_sha256": file_sha(ALFWORLD_DATA / "test_set.json"),
        "sources": {str(path.relative_to(ROOT)): file_sha(path) for path in source_files},
        "infrastructure_recovery_sha256": (
            file_sha(root / "infrastructure_recovery.json")
            if (root / "infrastructure_recovery.json").exists() else None
        ),
    }
    _immutable_json(root / "identity.json", frozen)
    if (root / "COMPLETE.json").exists():
        return read_json(root / "COMPLETE.json")
    for name in ("FAILED.json", "RELEASED.json"):
        previous = root / name
        if previous.exists():
            archive = root / "controller_history" / f"{time.time_ns()}-{name}"
            archive.parent.mkdir(exist_ok=True)
            previous.rename(archive)
    started = time.time()
    launch = None
    telemetry = None
    telemetry_stop = threading.Event()
    compute_started = None
    try:
        _status(root, "waiting_for_training")
        deadline = time.monotonic() + 32400
        while not (root.parent / "TRAINING_NATIVE_AUDIT.json").is_file():
            if (root.parent / "NATIVE_MONITOR_FAILED.json").is_file():
                raise RuntimeError("Engineer training failed its native-update audit")
            attempts = root.parent / "training_attempts.json"
            if (
                (root.parent / "FAILED.json").is_file()
                and attempts.exists() and len(read_json(attempts)) >= 2
            ):
                raise RuntimeError("Engineer training exhausted both attempts")
            if time.monotonic() >= deadline:
                raise TimeoutError("Engineer evaluation waited too long for audited training")
            time.sleep(30)
        validate_spec(spec_path)
        if any(file_sha(ROOT / path) != digest for path, digest in frozen["sources"].items()):
            raise ValueError("Engineer evaluator source changed while waiting for training")
        jobs_root = root / "jobs"
        if jobs_root.exists():
            credits = retry_credits(root)
            for job_root in jobs_root.iterdir():
                if not job_root.is_dir() or (job_root / "result.json").exists():
                    continue
                attempts_path = job_root / "attempts.json"
                if not attempts_path.exists():
                    continue
                attempts = read_json(attempts_path)
                recovered = any(
                    (job_root / f"attempt-{index + 1}" / "result.json").exists()
                    for index in range(len(attempts))
                )
                if not recovered and len(attempts) >= 2 + credits.get(job_root.name, 0):
                    raise EvaluationAttemptsExhausted(
                        f"Evaluation job exhausted its attempts before GPU allocation: {job_root.name}"
                    )
        audit = read_json(root.parent / "TRAINING_NATIVE_AUDIT.json")
        if audit["status"] != "complete" or not audit["source_manifest_matches"]:
            raise ValueError("Engineer training audit is incomplete")
        trained = read_json(Path(spec["training_root"]) / "optimizer/trained.json")
        history = {row["global_step"]: row["path"] for row in trained["checkpoint_history"]["actor"]}
        if set(spec["checkpoint_steps"]) != set(history):
            raise ValueError("Engineer checkpoints differ from the two predeclared candidates")
        models = {"e0": spec["initial_engineer"]}
        for step in spec["checkpoint_steps"]:
            model = Path(history[step])
            ensure_model_runtime_metadata(model, Path(spec["initial_engineer"]))
            models[f"step{step}"] = str(model)
        _immutable_json(root / "models.json", models)
        release = gpu_release_audit()
        if (
            release["foreign_gpu_pids"] or release["filler_kind"] != "adaptive"
            or not release["filler_unit_active"]
        ):
            raise RuntimeError("Engineer evaluator cannot claim GPUs from another workload")
        launch = Launcher(root / "supervision")
        compute_started = time.time()
        telemetry = threading.Thread(
            target=monitor_gpu_processes,
            args=(root / "gpu_process_samples.jsonl", telemetry_stop, 5.0),
            daemon=True,
        )
        telemetry.start()
        _status(root, "baseline_and_matched_generation")
        task_ids = sorted(manifest["selection_task_ids"] + manifest["report_task_ids"])
        proposals = {}
        with launch.vllm(**_model_options(
            spec["solver_model"], port=19701, gpus="3,4,5,6,7",
            name="fixed_solver_5", replicas=5, context=32768,
        )) as solver_urls:
            urls = solver_urls.split(",")
            jobs = Jobs(root / "jobs", launch, 29)
            baseline_names = []
            for index, url in enumerate(urls):
                name = f"baseline-{index}"
                baseline_names.append(name)
                jobs.add({
                    "name": name, "kind": "baseline", "task_ids": task_ids[index::len(urls)],
                    "url": url, "concurrency": spec["target_rollout_concurrency"],
                    "port_index": index,
                })
            jobs.tick()
            generation_names = {}
            with ExitStack() as engineer_servers:
                for index, (label, model) in enumerate(models.items()):
                    cached = all(
                        (root / "jobs" / f"generate-{label}-{sample}" / "result.json").is_file()
                        for sample in range(len(prompts))
                    )
                    if cached:
                        url = f"http://127.0.0.1:{19601 + index * 20}/v1"
                    else:
                        url = engineer_servers.enter_context(launch.vllm(**_model_options(
                            model, port=19601 + index * 20, gpus=str(index),
                            name=label, replicas=1, context=16384,
                        )))
                    names = []
                    for sample, prompt in enumerate(prompts):
                        name = f"generate-{label}-{sample}"
                        names.append(name)
                        jobs.add({
                            "name": name, "kind": "generate", "model_label": label,
                            "model": model, "url": url, "messages": prompt["messages"],
                            "prompt_id": prompt["id"], "sample_index": sample,
                            "decode_seed": spec["seed"] + 5_000_003 + sample,
                            "temperature": spec["engineer_temperature"],
                            "tokens": spec["engineer_tokens"],
                        })
                    generation_names[label] = names
                    jobs.tick()
                results = jobs.wait()
                for label, names in generation_names.items():
                    proposals[label] = [results[name] for name in names]
                    write_json(root / "proposals" / f"{label}.json", proposals[label])
            baseline = {}
            for name in baseline_names:
                values = {int(key): value for key, value in results[name]["scores"].items()}
                if set(values) & set(baseline):
                    raise ValueError("Baseline cohorts overlap")
                baseline.update(values)
            baseline = normalize_rewards(baseline, task_ids)
            write_json(root / "baseline.json", baseline)
            with launch.vllm(**_model_options(
                spec["solver_model"], port=19801, gpus="0,1,2",
                name="fixed_solver_extra_3", replicas=3, context=32768,
            )) as extra_urls:
                urls.extend(extra_urls.split(","))
                _status(root, "checkpoint_selection")
                selection = _score_models(
                    root, launch, spec, proposals, list(models), manifest["selection_task_ids"],
                    baseline, urls, "selection",
                )
                candidates = []
                for step in spec["checkpoint_steps"]:
                    label = f"step{step}"
                    difference = paired_difference(
                        selection["e0"], selection[label],
                        seed=spec["seed"] + 7_000_003, draws=spec["bootstrap_draws"],
                    )
                    candidates.append({"label": label, "global_step": step, **difference})
                best = max(candidates, key=lambda row: (row["candidate_accuracy"], -row["global_step"]))
                selected = best if best["candidate_minus_control"] >= spec["selection_min_delta"] else None
                write_json(root / "selection.json", {
                    "candidates": candidates, "selected": selected,
                    "minimum_delta": spec["selection_min_delta"],
                })
                difference = None
                useful_editing = False
                report_no_patch_gains = None
                if selected is not None:
                    _status(root, "independent_report", selected=selected["label"])
                    reports = _score_models(
                        root, launch, spec, proposals, ["e0", selected["label"]],
                        manifest["report_task_ids"], baseline, urls, "report",
                    )
                    difference = paired_difference(
                        reports["e0"], reports[selected["label"]],
                        seed=spec["seed"] + 9_000_003, draws=spec["bootstrap_draws"],
                    )
                    report_no_patch_gains = {
                        label: report["mean_gain_vs_no_patch"] for label, report in reports.items()
                    }
                    useful_editing = (
                        reports[selected["label"]]["valid_proposals"] > 0
                        and reports[selected["label"]]["mean_gain_vs_no_patch"] > 0
                    )
                passed = difference is not None and (
                    difference["candidate_minus_control"] >= spec["report_min_delta"]
                    and (
                        not spec["require_positive_report_lcb"]
                        or difference["paired_question_bootstrap_95pct"][0] > 0
                    )
                )
        validate_spec(spec_path)
        retry_credits(root)
        if any(file_sha(ROOT / path) != digest for path, digest in frozen["sources"].items()):
            raise ValueError("Engineer evaluator source changed before result publication")
        complete = {
            "status": "complete", "gate_passed": passed,
            "useful_editing_evidence": passed and useful_editing,
            "report_mean_gains_vs_no_patch": report_no_patch_gains,
            "selected": selected, "paired_report": difference,
            "report_opened": difference is not None, "selection_candidates": candidates,
            "candidates_per_model": 8, "training_seed": spec["seed"],
            "selection_task_count": len(manifest["selection_task_ids"]),
            "report_task_count": len(manifest["report_task_ids"]),
            "invalid_candidates": {
                label: sum(not row["valid"] for row in values) for label, values in proposals.items()
            },
            "elapsed_seconds": time.time() - started,
            "compute_started_at_unix": compute_started,
            "elapsed_compute_seconds": time.time() - compute_started,
            "sealed_test_opened": False,
            "interpretation": (
                "Single-seed exploratory E1/E0 comparison with matched evidence, prompts, "
                "decode seeds and proposal counts. Question bootstrap conditions on these "
                "eight proposals and does not establish seed-level or unseen-packet transfer."
            ),
        }
        write_json(root / "COMPLETE.json", complete)
        write_json(root.parent / "ENGINEER_VALIDATION_COMPLETE.json", complete)
        _status(root, "complete", gate_passed=passed)
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "failed_at_unix": time.time()})
        _status(root, "failed", error=repr(exc))
        raise
    finally:
        if launch is not None:
            launch.close()
            telemetry_stop.set()
            if telemetry is not None:
                telemetry.join(timeout=10)
            telemetry_path = root / "gpu_process_samples.jsonl"
            if telemetry_path.exists():
                try:
                    utilization = summarize_gpu_processes(
                        telemetry_path, start_time=compute_started,
                    )
                except ValueError as exc:
                    utilization = {"status": "invalid", "error": repr(exc)}
                write_json(root / "gpu_utilization_summary.json", utilization)
            write_json(root / "RELEASED.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job", type=Path)
    mode.add_argument("--spec", type=Path)
    args = parser.parse_args()
    try:
        print(run_job(args.job) if args.job else run(args.spec), flush=True)
    except EvaluationAttemptsExhausted as exc:
        print(str(exc), file=sys.stderr, flush=True)
        raise SystemExit(78) from exc


if __name__ == "__main__":
    main()
