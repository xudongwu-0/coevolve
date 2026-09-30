"""Run the official baseline -> failure packets -> engineer -> paired evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

PROJECT = Path(__file__).resolve().parents[2]
UPSTREAM = PROJECT / "third_party/harness-r1-upstream"
ASSETS = PROJECT / "assets/alfworld_harness_r1"
PIN = "94f2e087f573e1b82fc9568bb634d4ae9a887e28"
PYTHON = Path("/dev/shm/harness-r1-alfworld-venv/bin/python")
TYPES = (
    "pick_and_place_simple", "look_at_obj_in_light",
    "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep", "pick_two_obj_and_place",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cohort(paths: list[str], size: int) -> list[int]:
    if not 6 <= size <= len(paths):
        raise ValueError("cohort must cover all six types and fit the official split")
    groups = {kind: [] for kind in TYPES}
    for index, path in enumerate(paths):
        kind = path.split("/")[2].split("-")[0]
        if kind not in groups:
            raise ValueError(f"unknown task type: {kind}")
        groups[kind].append(index)
    selected = []
    while len(selected) < size:
        progressed = False
        for kind in TYPES:
            if groups[kind] and len(selected) < size:
                selected.append(groups[kind].pop(0))
                progressed = True
        if not progressed:
            raise ValueError("not enough tasks")
    return selected


def prepare_runtime(size: int = 32, task_start: int | None = None) -> dict:
    head = subprocess.check_output(
        ["git", "-c", f"safe.directory={UPSTREAM}", "-C", str(UPSTREAM),
         "rev-parse", "HEAD"], text=True
    ).strip()
    if head != PIN:
        raise RuntimeError(f"upstream pin changed: {head}")
    split = UPSTREAM / "data/alfworld/test_set.json"
    paths = json.loads(split.read_text())["mixed"]
    if len(paths) != 500:
        raise ValueError("expected official 500-task mixed test split")
    runtime = ASSETS / "AgentBench"
    runtime.mkdir(parents=True, exist_ok=True)
    source = UPSTREAM / "code/life-harness/AgentBench"
    for name in ("src", "scripts"):
        link = runtime / name
        if not link.exists():
            link.symlink_to(source / name, target_is_directory=True)
    configs = runtime / "configs"
    configs.mkdir(exist_ok=True)
    for entry in (source / "configs").iterdir():
        if entry.name == "generated":
            continue
        link = configs / entry.name
        if not link.exists():
            link.symlink_to(entry, target_is_directory=entry.is_dir())
    data = ASSETS / "data"
    data.mkdir(exist_ok=True)
    (runtime / "data").mkdir(exist_ok=True)
    link = runtime / "data/alfworld"
    if not link.exists():
        link.symlink_to(data, target_is_directory=True)
    destination = data / "test_set.json"
    if destination.exists() and digest(destination) != digest(split):
        raise RuntimeError("runtime official split has changed")
    if not destination.exists():
        destination.write_bytes(split.read_bytes())
    if task_start is None:
        ids = select_cohort(paths, size)
        label = "transductive_diagnostic_not_untouched_test"
        selection = "six-type round-robin, first occurrence in official split"
    else:
        if not 0 <= task_start < len(paths) or task_start + size > len(paths):
            raise ValueError("contiguous cohort must fit the official split")
        ids = list(range(task_start, task_start + size))
        label = "protocol_partition_transductive_training_or_development"
        selection = f"contiguous official split indices [{task_start},{task_start + size})"
    cohort = {
        "label": label,
        "upstream_commit": PIN, "split": "test_set", "split_sha256": digest(split),
        "selection": selection,
        "task_ids": ids, "task_paths": [paths[i] for i in ids],
    }
    suffix = f"{size}" if task_start is None else f"{size}_start{task_start}"
    (ASSETS / f"cohort{suffix}.json").write_text(json.dumps(cohort, indent=2) + "\n")
    (ASSETS / f"task_ids{suffix}.json").write_text(json.dumps(ids) + "\n")
    return cohort


def cpu_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", ALFWORLD_DATA=str(ASSETS / "data"),
               TMPDIR=str(ASSETS / "scratch"), PYTHONDONTWRITEBYTECODE="1")
    (ASSETS / "scratch").mkdir(exist_ok=True)
    return env


def run(cmd: list[str], log: Path) -> None:
    with log.open("w") as stream:
        result = subprocess.run(cmd, env=cpu_env(), cwd=PROJECT,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"official command failed ({result.returncode}); see {log}")


def official_paths() -> tuple[Path, Path]:
    return (UPSTREAM / "code/life-harness/AgentBench/scripts",
            UPSTREAM / "code/Relax/examples/harness_r1")


def baseline_command(args, out: Path) -> list[str]:
    scripts, _ = official_paths()
    return [
        str(args.python), str(scripts / "harness_r1_batch_debug.py"),
        "--agentbench-dir", str(ASSETS / "AgentBench"),
        "--agentbench-python", str(args.python),
        "--alfworld-worker-python", str(args.python),
        "--bench", "alfworld", "--run-id", args.run_id + "_baseline",
        "--alfworld-split", "test_set", "--alfworld-max-step", "50",
        "--task-ids-file", str(out / "task_ids.json"),
        "--batch-size", str(args.size), "--num-batches", "1",
        "--rollout-concurrency", str(args.concurrency),
        "--controller-port", str(args.controller_port),
        "--worker-port-base", str(args.worker_port),
        "--skip-adb", "--auto-retry",
        "--rollout-base-url", args.target_url, "--rollout-model", args.target_model,
        "--agent-name", args.target_agent_name,
        "--rollout-temperature", "0.0", "--rollout-max-tokens", "4096",
        "--rollout-tool-choice", "auto",
        "--rollout-chat-template-kwargs", '{"enable_thinking":false}',
        "--output-root", str(out / "baseline"),
    ]


def checked_rows(root: Path, ids: list[int]) -> list[dict]:
    rows = {}
    for file in root.glob("**/runs.jsonl"):
        for line in file.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row.get("index")] = row
    if set(rows) != set(ids):
        raise RuntimeError(f"infrastructure: task coverage mismatch: {set(rows)} != {set(ids)}")
    allowed = {"completed", "task_limit_reached", "agent_invalid_action",
               "agent_context_limit"}
    for row in rows.values():
        output = row.get("output") or {}
        status = str(output.get("status", "")).lower().replace(" ", "_")
        if status not in allowed or not isinstance(output.get("result"), dict):
            raise RuntimeError(f"infrastructure: invalid rollout result status {status!r}")
    return [rows[i] for i in ids]


def freeze_baseline(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name in {"runs.jsonl", "run_config.json"}}


def patch_is_no_op(patch: dict, task: dict) -> bool:
    scripts, _ = official_paths()
    sys.path.insert(0, str(scripts))
    from harness_r1_patch import compile_patch_to_task_definition
    patched = compile_patch_to_task_definition(task, patch, "alfworld")
    return patched == task


def endpoint_identity(url: str, model: str) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url.rstrip("/") + "/models", timeout=20) as response:
        payload = json.load(response)
    if model not in {row.get("id") for row in payload.get("data", [])}:
        raise RuntimeError(f"endpoint does not advertise required model {model!r}")
    return payload


def preserve_paired_ids(records: list[dict], task_ids: list[int], identity: str) -> list[dict]:
    for record in records:
        metadata = record["metadata"]
        if set(map(int, metadata["baseline_rewards"])) != set(task_ids):
            raise RuntimeError("official packet changed paired task identity")
        metadata["task_ids"] = task_ids
        metadata["batch_size"] = len(task_ids)
        metadata["adapter_baseline_identity_sha256"] = identity
    return records


def classify_results(results: list[dict]) -> dict:
    counts = {"infra_failure": 0, "invalid_patch": 0, "no_op": 0, "valid_evaluated": 0}
    for row in results:
        status = str(row.get("eval_status", ""))
        patch_status = str(row.get("patch_status", ""))
        if ("failed" in status or "engineer_request_failed" in patch_status
                or "missing_source" in patch_status):
            category = "infra_failure"
        elif not row.get("valid_patch"):
            category = "invalid_patch"
        elif row.get("adapter_no_op"):
            category = "no_op"
        else:
            category = "valid_evaluated"
        counts[category] += 1
    return counts


def paired(args) -> None:
    if args.python.is_symlink():
        raise RuntimeError("official evaluator resolves Python symlinks; use an isolated venv "
                           "created with virtualenv --copies, not a symlink interpreter")
    cohort = prepare_runtime(args.size, args.task_start)
    out = args.run_root.resolve() / args.run_id
    out.mkdir(parents=True, exist_ok=False)
    (out / "task_ids.json").write_text(json.dumps(cohort["task_ids"]))
    source_identity = {}
    for folder in ("code/life-harness/AgentBench/src/server/tasks/alfworld",
                   "code/life-harness/AgentBench/src/server/harness",
                   "code/life-harness/AgentBench/scripts",
                   "code/Relax/examples/harness_r1"):
        for path in sorted((UPSTREAM / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yaml", ".json"}:
                source_identity[str(path.relative_to(UPSTREAM))] = digest(path)
    manifest = dict(cohort, target={"base_url": args.target_url, "model": args.target_model,
                    "checkpoint_model": args.target_checkpoint_label,
                    "checkpoint_record": args.target_checkpoint_record,
                    "temperature": 0, "enable_thinking": False, "tool_choice": "auto",
                    "max_tokens": 4096,
                    "endpoint_models": endpoint_identity(args.target_url, args.target_model)},
                    engineer={"base_url": args.engineer_url,
                    "model": args.engineer_model, "protocol": "prefill_think_patch",
                    "checkpoint_record": args.engineer_checkpoint_record,
                    "endpoint_models": endpoint_identity(args.engineer_url, args.engineer_model)},
                    upstream_source_identity=source_identity,
                    environment_freeze_sha256=digest(ASSETS / "environment_freeze.txt"),
                    asset_manifest_sha256=digest(ASSETS / "asset_manifest.json"),
                    baseline_command=baseline_command(args, out))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    control_identity = {name: digest(out / name) for name in ("manifest.json", "task_ids.json")}
    run(manifest["baseline_command"], out / "baseline.log")
    checked_rows(out / "baseline", cohort["task_ids"])
    frozen = freeze_baseline(out / "baseline")
    (out / "baseline_identity.json").write_text(json.dumps(
        {"baseline_files": frozen, "controls": control_identity}, indent=2))
    _, examples = official_paths()
    packets = out / "failure_packets.jsonl"
    run([str(args.python), str(examples / "build_alfworld_dataset.py"),
         "--source-root", str(out / "baseline"), "--output", str(packets),
         "--alfworld-split", "test_set", "--target-agent-name", args.target_agent_name,
         "--target-model", args.target_model, "--response-protocol", "prefill_think_patch",
         "--schema-style", "alfworld_life_multihook_v1", "--max-traces", str(args.size),
         "--include-incomplete"],
        out / "packets.log")
    records = [json.loads(line) for line in packets.read_text().splitlines() if line.strip()]
    if not records:
        (out / "adapter_summary.json").write_text(json.dumps({"status": "baseline_all_pass_no_failures"}))
        return
    # Pinned builder omits sparse task_ids; evaluator already natively supports them.
    preserve_paired_ids(records, cohort["task_ids"], digest(out / "baseline_identity.json"))
    packets.write_text("".join(json.dumps(row) + "\n" for row in records))
    cmd = [str(args.python), str(examples / "eval_alfworld_patches.py"),
           "--input", str(packets), "--output-root", str(out / "patched"),
           "--run-id", args.run_id + "_patched", "--engineer-base-url", args.engineer_url,
           "--engineer-model", args.engineer_model, "--engineer-concurrency", "1",
           "--engineer-api-key-env", "HARNESS_R1_ENGINEER_API_KEY",
           "--engineer-max-tokens", "12288", "--engineer-temperature", "0.0",
           "--engineer-chat-template-kwargs", '{"enable_thinking":true}',
           "--engineer-extra-body", '{"add_generation_prompt":false}',
           "--engineer-assistant-prefill", "<think>\n",
           "--response-protocol", "prefill_think_patch",
           "--target-base-url", args.target_url, "--target-model", args.target_model,
           "--target-agent-name", args.target_agent_name, "--alfworld-split", "test_set",
           "--alfworld-worker-python", str(args.python), "--alfworld-max-step", "50",
           "--rollout-concurrency", str(args.concurrency), "--rollout-max-tokens", "4096",
           "--rollout-tool-choice", "auto", "--rollout-auto-retry",
           "--rollout-chat-template-kwargs", '{"enable_thinking":false}',
           "--agentbench-dir", str(ASSETS / "AgentBench"),
           "--agentbench-python", str(args.python),
           "--controller-port-base", str(args.controller_port + 1),
           "--worker-port-base", str(args.worker_port + 20)]
    run(cmd, out / "engineer_eval.log")
    if freeze_baseline(out / "baseline") != frozen:
        raise RuntimeError("immutable baseline files changed")
    if any(digest(out / name) != value for name, value in control_identity.items()):
        raise RuntimeError("immutable baseline control identity changed")
    results = json.loads((out / "patched/results.json").read_text())
    for result in results:
        patch_path = out / "patched" / result["batch_tag"] / "patch.json"
        if patch_path.exists():
            patch = json.loads(patch_path.read_text())
            task_file = ASSETS / "AgentBench/configs/generated/harness_r1" / (
                args.run_id + "_baseline") / "alfworld-noharness-b000.task.json"
            result["adapter_no_op"] = patch_is_no_op(patch, json.loads(task_file.read_text()))
    summary = {"label": cohort["label"], "categories": classify_results(results),
               "baseline_identity_sha256": digest(out / "baseline_identity.json"),
               "official_summary": json.loads((out / "patched/summary.json").read_text())}
    # Official evaluator assigns baseline fallback to infra failures: never count that as evidence.
    summary["paired_score_valid"] = all(
        result.get("valid_patch") and result.get("eval_status") == "ok" for result in results)
    for result in results:
        if result.get("valid_patch") and result.get("eval_status") == "ok":
            checked_rows(out / "patched" / result["batch_tag"] / "eval", cohort["task_ids"])
    baseline_by_id = {int(k): float(v) for record in records
                      for k, v in record["metadata"]["baseline_rewards"].items()}
    evaluated = [result for result in results
                 if result.get("valid_patch") and result.get("eval_status") == "ok"]
    summary["paired_tasks"] = []
    for result in evaluated:
        patched = {int(k): float(v) for k, v in result["patched_rewards"].items()}
        for task_id, task_path in zip(cohort["task_ids"], cohort["task_paths"]):
            summary["paired_tasks"].append({"task_id": task_id, "task_path": task_path,
                "baseline_reward": baseline_by_id[task_id], "patched_reward": patched[task_id],
                "delta_reward": patched[task_id] - baseline_by_id[task_id]})
    summary["rescued_tasks"] = sum(row["baseline_reward"] < 1 <= row["patched_reward"]
                                   for row in summary["paired_tasks"])
    summary["regressed_tasks"] = sum(row["patched_reward"] < 1 <= row["baseline_reward"]
                                     for row in summary["paired_tasks"])
    (out / "adapter_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "paired"])
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--task-start", type=int,
                        help="Use a contiguous cohort beginning at this official split index")
    parser.add_argument("--python", type=Path, default=PYTHON)
    parser.add_argument("--run-id", default="diagnostic12")
    parser.add_argument("--run-root", type=Path, default=ASSETS / "runs")
    parser.add_argument("--target-url", default="http://127.0.0.1:8110/v1")
    parser.add_argument("--target-model", default="Qwen3.5-9B", choices=["Qwen3.5-9B", "policy"],
                        help="Wire alias only; checkpoint identity is recorded separately")
    parser.add_argument("--target-agent-name", default="qwen35-9b-nothink")
    parser.add_argument("--target-checkpoint-label", default="original-qwen3.5-9b")
    parser.add_argument("--target-checkpoint-record", default="assets/Qwen3.5-9B/download_receipt.json")
    parser.add_argument("--engineer-url", default="http://127.0.0.1:8142/v1")
    parser.add_argument("--engineer-model", default="harness-engineer")
    parser.add_argument("--engineer-checkpoint-record", default="externally_managed_unverified")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--controller-port", type=int, default=15240)
    parser.add_argument("--worker-port", type=int, default=15340)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare_runtime(args.size, args.task_start), indent=2))
    else:
        output = args.run_root.resolve() / args.run_id
        existed = output.exists()
        try:
            paired(args)
        except Exception as exc:
            if not existed and output.is_dir():
                (output / "adapter_failure.json").write_text(json.dumps({
                    "category": "infrastructure_or_identity_control_failure",
                    "error": str(exc), "paired_score_valid": False}, indent=2))
            raise


if __name__ == "__main__":
    main()
