"""Local ALFWorld adaptation of WHALE: online RSFT alternating with Meta-Harness.

``prepare`` is CPU-only. ``run`` requires an explicitly assigned, idle research
node. This is not an official ALFWorld result or a Harness-R1/GRPO training arm.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from urllib.parse import urlsplit


def isolate_local_paths(paths: list[str], source_root: Path) -> tuple[list[str], list[str]]:
    local = source_root.resolve() / "whale_local"
    removed = [entry for entry in paths if entry and "whale_local" in Path(entry).parts
               and not Path(entry).resolve().is_relative_to(local)]
    return [entry for entry in paths if entry not in removed], removed


# The installed venv's .pth appends main/whale_local even with a snapshot PYTHONPATH.
# Remove that fallback before importing any local dependency.
sys.path[:], PRUNED_LOCAL_PATHS = isolate_local_paths(
    sys.path, Path(__file__).resolve().parents[2])
from .common import ROOT, file_sha, read_json, write_json

UPSTREAM = ROOT / "WHALE"
NATIVE = UPSTREAM / "domains/search_qa"
MH = NATIVE / "meta_harness"
PATCH_SCRIPTS = ROOT / "third_party/harness-r1-upstream/code/life-harness/AgentBench/scripts"
DEFAULT_CONFIG = ROOT / "configs/alfworld_whale_baseline.json"
GPUS = "0,1,2,3,4,5,6,7"
NODES = {"dev-NDH200-025": (1009, 1008), "dev-NDH200-026": (1008, 1008)}


def frozen_config_path(frozen: dict) -> Path:
    relative = Path(frozen.get("config_relative_path", ""))
    if (not frozen.get("config_relative_path") or relative.is_absolute()
            or ".." in relative.parts):
        raise ValueError("A source-relative freeze is required; current source requires v6")
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError("Frozen config escapes the executed source tree")
    return path


def run_output_path(path: Path) -> Path:
    path = path.resolve()
    path.relative_to((ROOT / "runs").resolve())
    return path


def local_source_manifest() -> dict[str, str]:
    """Hash the local static import closure, not HEAD or unrelated work files."""
    pending = [
        "sao_bridge.alfworld_whale_baseline", "sao_bridge.alfworld_env_service",
        "sao_bridge.alfworld_controller", "harness_r1_searchqa.grpo_train",
    ]
    visited, files = set(), {}
    local = ROOT / "whale_local"
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        location = local.joinpath(*module.split("."))
        path = location.with_suffix(".py")
        package = module.rpartition(".")[0]
        if not path.is_file():
            path = location / "__init__.py"
            package = module
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(local.resolve()):
            raise ValueError(f"Local code symlink escapes the source snapshot: {path}")
        files[str(path.relative_to(ROOT))] = file_sha(path)
        parents = module.split(".")
        pending.extend(".".join(parents[:i]) for i in range(1, len(parents)))
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    name = importlib.util.resolve_name("." * node.level + name, package)
                pending.append(name)
                pending.extend(f"{name}.{alias.name}" for alias in node.names if alias.name != "*")
    return dict(sorted(files.items()))


def execution_source_audit(frozen: dict | None = None) -> dict:
    executed_root = Path(__file__).resolve().parents[2]
    if ROOT.resolve() != executed_root:
        raise ValueError("Mixed main-tree and snapshot sao_bridge imports")
    if Path.cwd().resolve() != executed_root:
        raise ValueError("Run from the executed source snapshot root, not a different working tree")
    local = executed_root / "whale_local"
    for entry in sys.path:
        if entry and "whale_local" in Path(entry).parts and not Path(entry).resolve().is_relative_to(local):
            raise ValueError(f"Foreign mutable local-code import path: {entry}")
    origins = {}
    for name, module in list(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin and "whale_local" in Path(origin).parts:
            path = Path(origin).resolve()
            if not path.is_relative_to(local):
                raise ValueError(f"Foreign mutable local-code module: {name}: {path}")
            origins[name] = str(path.relative_to(executed_root))
    if frozen is not None:
        for name, expected in frozen.get("source_dependency_sha256", {}).items():
            path = executed_root / name
            if (not path.resolve().is_relative_to(local) or file_sha(path) != expected):
                raise ValueError(f"Frozen snapshot source dependency changed: {name}")
    return {"execution_source_root": str(executed_root), "loaded_local_module_origins": origins,
            "pruned_foreign_local_paths": PRUNED_LOCAL_PATHS,
            "git_revision_is_not_a_validation_key": True}


def validate_frozen(frozen_path: Path, expected_sha: str | None = None, *,
                    verify_weights: bool = True) -> tuple[dict, dict]:
    if expected_sha is not None and file_sha(frozen_path) != expected_sha:
        raise ValueError("Explicit source freeze digest changed")
    frozen = read_json(frozen_path)
    source = execution_source_audit(frozen)
    current = preparation(frozen_config_path(frozen), verify_weights=verify_weights)
    if current != frozen:
        raise ValueError("Frozen source/input manifest changed")
    return frozen, {**source, "status": "validated_cpu_only",
                    "frozen_sha256": file_sha(frozen_path),
                    "source_files_verified": len(frozen.get("source_dependency_sha256", {})),
                    "model_weights_verified": frozen["model_weights_verified"],
                    "gpu_allocation_started": False}


class WallBudget:
    def __init__(self, seconds: int):
        self.deadline = time.monotonic() + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Frozen WHALE wall-clock budget exhausted")


def create_launcher(root: Path, budget):
    from .launch import Launcher
    launch = Launcher(root, budget=budget)

    def local_module(module, args, *, gpus, name, env_overrides=None):
        return launch.start(
            [sys.executable, "-B", "-m", "sao_bridge.alfworld_whale_baseline",
             "exec-local", module, *map(str, args)],
            gpus=gpus, name=name, env_overrides=env_overrides)

    launch.module = local_module
    return launch


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def immutable(path: Path, value) -> None:
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen artifact changed: {path}")
    else:
        write_json(path, value)


def official_function(path: Path, name: str):
    """Load one unmodified pure upstream function, never its remote transports."""
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise RuntimeError(f"Missing unique official function: {name}")
    module = ast.Module(body=nodes, type_ignores=[])
    namespace = {"Path": Path}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name]


def select_harness(points: list[tuple[str, float, float]], root: Path) -> tuple[str, dict]:
    if not points or len({p[0] for p in points}) != len(points):
        raise ValueError("Meta-Harness requires distinct, evaluated candidates")
    for _, score, turns in points:
        if not math.isfinite(score) or not 0 <= score <= 1 or not math.isfinite(turns) or turns < 0:
            raise ValueError("Invalid Meta-Harness measurement")
    pareto = official_function(MH / "benchmark.py", "compute_pareto_frontier")(points)
    frontier = {"_pareto": [
        {"harness": name, "avg_success_rate": round(score, 4),
         "avg_mean_turn_count": round(turns, 3)} for name, score, turns in pareto
    ]}
    name, available = official_function(
        MH / "meta_harness_search_r1.py", "pick_accepted_from_frontier",
    )(frontier, root)
    if name is None or available != name:
        raise RuntimeError("Official MH selected a missing artifact")
    return name, frontier


def patch_module():
    if str(PATCH_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PATCH_SCRIPTS))
    import harness_r1_patch
    return harness_r1_patch


def parse_candidate(completion: str) -> dict:
    module = patch_module()
    raw = module.extract_prefilled_think_patch_json_object(completion)
    patch = module.normalize_patch(raw, bench="alfworld")
    module.require_code_hook_only_patch(patch)
    return patch


def loopback_urls(value: str) -> str:
    for url in value.split(","):
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path != "/v1"):
            raise ValueError("Inference must use loopback HTTP /v1 endpoints")
    return value


def split_plan(rows: list[dict], games, config: dict, *, development_rows: list[dict],
               development_games: tuple[str, ...], protocol: dict) -> dict:
    from .alfworld_train_data import TRAIN_SPLIT, goal_group
    count = config["cycles"] * config["rsft_batches_per_cycle"] * config["episodes_per_batch"]
    if len(rows) < count or len(games) != 500:
        raise ValueError("Insufficient MA reference records or invalid legacy catalogue")
    selected = rows[:count]
    for index, row in enumerate(selected):
        task = row.get("task_id")
        if (type(task) is not int or not 0 <= task < 240 or row.get("task_split") != "test_set"
                or task in protocol["all_task_ids"]):
            raise ValueError("MA legacy training population/core200 exclusion changed")
        if (row.get("joint_version") != index // 8 or row.get("joint_slot") != index % 8
                or row.get("id") != f"h0w0-{index:05d}-task{task}"):
            raise ValueError("MA record order/identity changed")
        if row.get("task_context") != games[task]:
            raise ValueError("MA reference task context differs from the actual legacy game")
    train = [row["task_id"] for row in selected]
    used = {goal_group(games[index]) for index in train}
    related = [index for index in protocol["all_task_ids"] if goal_group(games[index]) in used]
    strict_overlap = set(related) & set(protocol["strict_goal_disjoint_task_ids"])
    if strict_overlap:
        raise ValueError("MA training population overlaps strict197 evaluation goals")
    if (len(development_rows) != len(development_games) or any(
            row["task_id"] != index or row["task_split"] != TRAIN_SPLIT
            for index, row in enumerate(development_rows))):
        raise ValueError("Development rows must index the verified train-only catalogue")
    forbidden = used | {goal_group(games[index]) for index in protocol["all_task_ids"]}
    order = list(range(len(development_rows)))
    random.Random(config["training_seed"]).shuffle(order)
    dev = []
    for index in order:
        group = goal_group(development_games[index])
        if group not in forbidden:
            dev.append(index)
            forbidden.add(group)
        if len(dev) == config["development_tasks"]:
            break
    if len(dev) != config["development_tasks"]:
        raise ValueError("Insufficient goal-disjoint training-domain development tasks")
    return {"train_ids": train, "train_record_indices": list(range(count)),
            "train_record_ids": [row["id"] for row in selected], "unique_training_tasks": len(set(train)),
            "development_ids": dev, "train_all200_task_overlap": 0,
            "train_all200_goal_related_task_ids": related, "train_strict197_goal_overlap": 0,
            "train_development_goal_overlap": 0, "final_eval_used_for_selection": False}


def training_episode_rows(rows: list[dict], indices: list[int]) -> list[dict]:
    return [{**{key: rows[index][key] for key in ("id", "task_id", "task_split")},
             "source_record_index": index} for index in indices]


def validate_config(config: dict) -> None:
    expected = {
        "protocol": "alfworld-whale-local-rsft-mh-v1", "score_threshold": 0.5,
        "sft_epochs": 1, "sft_micro_batch_size_per_gpu": 1, "temperature": 1.0,
        "solver_tokens": 4096, "max_context_tokens": 32768,
        "world_size": 8, "optimizer_continuation": True,
        "initial_solver": "assets/Harness-R1/target-agent-sft",
        "fixed_proposer": "assets/Harness-R1/harness-r1",
        "training_population": "ma_joint_legacy_test_set_first_records",
        "training_rows": "runs/alfworld_joint_h0w0_8h_v1/data.json",
        "joint_solver_spec": "runs/alfworld_joint_h0w0_8h_v1/solver_spec.json",
        "training_catalogue": "assets/alfworld_harness_r1/data/test_set.json",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Frozen WHALE contract changed: {key}")
    for key in ("cycles", "rsft_batches_per_cycle", "episodes_per_batch",
                "development_tasks", "mh_iterations", "proposals_per_iteration",
                "sft_mini_batch_size", "proposer_tokens", "training_seed",
                "development_seed", "proposer_seed", "episode_concurrency",
                "wall_budget_seconds", "phase_timeout_seconds"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["cycles"] < 2 or config["rsft_batches_per_cycle"] < 2:
        raise ValueError("This baseline must exercise online refresh and alternating cycles")
    if (not 0 < config["learning_rate"] < 1 or config["weight_decay"] < 0
            or config["proposer_tokens"] >= config["max_context_tokens"]):
        raise ValueError("Invalid optimization/token budget")


def inherited_cost(config: dict) -> dict:
    root = ROOT / config["initial_selection_reports"]
    reports = list(sorted(root.glob("*.json")))
    if not reports:
        raise ValueError("Inherited H0 selection reports missing")
    selections = [read_json(path) for path in reports]
    history = [ROOT / path for path in config["historical_initial_reports"]]
    historical = [read_json(path) for path in history]
    proposals = read_json(ROOT / config["initial_h0_proposals"])
    if len(proposals) != 8:
        raise ValueError("Inherited H0 proposal budget changed")
    return {
        "selection_solver_episodes": sum(len(value["results"]) for value in selections),
        "selection_solver_generated_tokens": sum(value["tokens"] for value in selections),
        "historical_initial_report_episodes": sum(len(value["results"]) for value in historical),
        "historical_initial_report_generated_tokens": sum(
            row["tokens"] for report in historical for row in report["results"]),
        "h0_proposer_completions": len(proposals),
        "h0_proposer_tokens": None,
        "h0_proposer_tokens_limitation": "Compact proposal receipt omits usage; unknown, not zero.",
        "source_sha256": {str(path.relative_to(ROOT)): file_sha(path) for path in [
            *reports, *history, ROOT / config["initial_h0_proposals"]]},
        "scope": "Previously paid shared selection and historical baseline reports; not additional queued work.",
    }


def preparation(config_path: Path, *, verify_weights: bool = True) -> dict:
    from .alfworld_core_extend import validate_cohort
    from .alfworld_train_data import training_catalogue, evaluation_paths
    from .alfworld_runtime import ALFWORLD_DATA
    execution_source_audit()
    config_path = config_path.resolve()
    relative_config = config_path.relative_to(ROOT.resolve()).as_posix()
    config = read_json(config_path)
    validate_config(config)
    for relative, expected in config["pinned_files"].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError(f"Frozen source/input changed: {relative}")
    protocol = read_json(ROOT / config["evaluation_protocol"])
    validate_cohort(protocol)
    if (protocol["seed"] != 6000045 or protocol["temperature"] != 1.0
            or protocol["solver_tokens"] != config["solver_tokens"]):
        raise ValueError("Final evaluation decoding no longer matches core200")
    rows = read_json(ROOT / config["training_rows"])
    joint_spec = read_json(ROOT / config["joint_solver_spec"])
    if (Path(joint_spec["data"]).resolve() != (ROOT / config["training_rows"]).resolve()
            or Path(joint_spec["model"]).resolve() != (ROOT / config["initial_solver"]).resolve()
            or joint_spec["batch_size"] != 8):
        raise ValueError("MA Solver data/model reference changed")
    games = evaluation_paths()
    if (ROOT / config["training_catalogue"]).resolve() != (ALFWORLD_DATA / "test_set.json").resolve():
        raise ValueError("Legacy MA training requires the actual worker's test_set catalogue")
    development_rows = read_json(ROOT / config["development_rows"])
    catalogue = ROOT / config["development_catalogue"]
    catalogue_sha = file_sha(catalogue)
    development_games = training_catalogue(catalogue, catalogue_sha)
    for row in development_rows:
        if (Path(row["task_catalogue"]).resolve() != catalogue.resolve()
                or row["task_catalogue_sha256"] != catalogue_sha):
            raise ValueError("Development row catalogue differs from the frozen input")
    plan = split_plan(rows, games, config, development_rows=development_rows,
                      development_games=development_games, protocol=protocol)
    selected_games = [games[index] for index in plan["train_ids"] + protocol["all_task_ids"]]
    selected_games.extend(development_games[index] for index in plan["development_ids"])
    game_files = {}
    for game in selected_games:
        for name in ("game.tw-pddl", "initial_state.pddl", "traj_data.json"):
            path = ALFWORLD_DATA / Path(game).parent / name
            game_files[str(path.relative_to(ROOT))] = file_sha(path)
    for relative, expected in config["model_weights"].items():
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(ROOT / relative)
        if verify_weights and file_sha(ROOT / relative) != expected:
            raise ValueError(f"Original H0/W0 weights changed: {relative}")
    runtime_files = {}
    for model in (config["initial_solver"], config["fixed_proposer"]):
        for name in ("config.json", "generation_config.json", "tokenizer.json",
                     "tokenizer_config.json", "chat_template.jinja", "processor_config.json",
                     "preprocessor_config.json"):
            runtime_files[f"{model}/{name}"] = file_sha(ROOT / model / name)
    native_file = NATIVE / "verl/workers/actor/dp_actor.py"
    # Source and ABI existence, without importing torch or initializing CUDA.
    actor = next(node for node in ast.parse(native_file.read_text()).body
                 if isinstance(node, ast.ClassDef) and node.name == "DataParallelPPOActor")
    if not any(isinstance(node, ast.FunctionDef) and node.name == "update_sft_policy"
               for node in actor.body):
        raise RuntimeError("Official native RSFT implementation unavailable")
    official_function(MH / "benchmark.py", "compute_pareto_frontier")
    official_function(MH / "meta_harness_search_r1.py", "pick_accepted_from_frontier")
    selected = read_json(ROOT / config["initial_harness"])
    patch_module().normalize_patch(selected, bench="alfworld")
    return {
        "status": "prepared_cpu_only_gpu_not_validated", "config": config,
        "freeze_format": "source-relative-v6-ma-population", "config_relative_path": relative_config,
        "joint_data_reference": {
            "path": config["training_rows"], "sha256": file_sha(ROOT / config["training_rows"]),
            "solver_spec_path": config["joint_solver_spec"],
            "solver_spec_sha256": file_sha(ROOT / config["joint_solver_spec"]),
            "total_reference_records": len(rows), "selected_prefix_records": len(plan["train_ids"]),
            "selected_episode_rows_sha256": digest(training_episode_rows(rows, plan["train_record_indices"])),
            "ma_1024_total_solver_records": 512, "ma_1024_total_editor_records": 512,
            "ma_solver_batch_size": joint_spec["batch_size"], "ma_solver_seed": joint_spec["seed"],
            "selection": "First consecutive Solver task records; preserve order and repeated tasks.",
            "historical_prompts_and_critic_evidence_used": False,
            "budget_caveat": "640 WHALE Solver episodes versus 512 Solver +512 editor records "
                             "at MA total1024; batch size, seed schedule and optimizer objectives differ.",
        },
        "gpu_smoke_contract": {
            "forward_input_tokens": 32768, "response_mask_tokens": 4096,
            "updates_before_export": 2, "updates_after_exact_restore": 1,
            "timeout_seconds": 7200, "short_smoke_can_authorize_training": False,
        },
        "source_dependency_sha256": local_source_manifest(),
        "config_sha256": file_sha(config_path), "plan": plan,
        "adapter_sha256": file_sha(Path(__file__)),
        "inherited_initialization_cost": inherited_cost(config),
        "game_files_sha256": game_files,
        "model_weights_verified": verify_weights,
        "model_runtime_sha256": runtime_files,
        "planned_training_episodes": len(plan["train_ids"]),
        "planned_proposals": config["cycles"] * config["mh_iterations"] * config["proposals_per_iteration"],
        "maximum_development_episodes": config["cycles"] * config["development_tasks"] * (
            1 + config["mh_iterations"] * config["proposals_per_iteration"]),
        "final_evaluation": {"all200": protocol["all_task_ids"],
                             "strict197": protocol["strict_goal_disjoint_task_ids"],
                             "seed": protocol["seed"], "samples": 1},
        "proposer": "fixed original RL H0; local replacement of official Claude Opus 4.7",
        "rsft_data_parallelism": "replicated singleton calls across FSDP ranks, exact token mean",
    }


def validate_episodes(records: list[dict], rows: list[dict], seed: int, tokens: int,
                      *, exact_tokens: bool) -> None:
    from .evaluation import stable_id_seed_offset
    if len(records) != len(rows):
        raise ValueError("Missing or duplicated ALFWorld episodes")
    for record, row in zip(records, rows):
        if (record.get("id") != row["id"] or record.get("task_id") != row["task_id"]
                or record.get("task_split") != row.get("task_split", "test_set")
                or record.get("sample_index") != 0
                or record.get("rollout_seed") != seed + stable_id_seed_offset(row["id"])
                or type(record.get("reward")) not in (int, float)
                or record["reward"] not in (0.0, 1.0)
                or type(record.get("tokens")) is not int or not 0 < record["tokens"] <= tokens):
            raise ValueError("ALFWorld episode identity/seed/reward/token mismatch")
        status = str(record.get("status", "")).rsplit(".", 1)[-1].upper().replace(" ", "_")
        if status not in {"COMPLETED", "TASK_LIMIT_REACHED", "AGENT_VALIDATION_FAILED",
                          "TOKEN_LIMIT", "INVALID_TOOL_ARGUMENTS"}:
            raise RuntimeError(f"ALFWorld infrastructure/unknown terminal status: {status}")
        if record["reward"] and (status != "COMPLETED" or record.get("truncated")):
            raise ValueError("Nonterminal/truncated success is not an RSFT target")
        if exact_tokens:
            calls = record.get("calls", [])
            if len(calls) != record["turns"] or not calls:
                raise ValueError("Exact sampled calls missing")
            for call in calls:
                if (not call.get("prompt_ids") or not call.get("response_ids")
                        or call.get("temperature") != 1.0
                        or len(call["prompt_ids"]) + len(call["response_ids"]) > 32768
                        or any(type(t) is not int or t < 0
                               for t in call["prompt_ids"] + call["response_ids"])):
                    raise ValueError("Malformed exact sampled token IDs")
            if sum(len(call["response_ids"]) for call in calls) != record["tokens"]:
                raise ValueError("Exact token accounting does not match verifier episode")


def accepted_calls(records: list[dict], threshold: float = 0.5) -> list[dict]:
    return [call for row in records if row["reward"] > threshold for call in row["calls"]]


async def collect(urls: str, rows: list[dict], harness: Path, config: dict,
                  seed: int, *, training: bool) -> dict:
    from . import alfworld_domain
    from .evaluation import Sampler, stable_id_seed_offset
    sampler = Sampler(loopback_urls(urls), concurrency=config["episode_concurrency"],
                      temperature=1.0, record_tokens=training,
                      chat_template_kwargs={"enable_thinking": False})
    semaphore = asyncio.Semaphore(config["episode_concurrency"])
    spec = alfworld_domain.harness_spec(harness.read_text())

    async def episode(row):
        calls = []

        async def generate(messages, tools, max_tokens, call_seed):
            response = await sampler(messages, tools, max_tokens, call_seed)
            if training:
                calls.append(response["record"])
            return response

        rollout_seed = seed + stable_id_seed_offset(row["id"])
        async with semaphore:
            record = await alfworld_domain.run_episode(
                row, spec, generate, total_tokens=config["solver_tokens"], seed=rollout_seed)
        record.update(sample_index=0, rollout_seed=rollout_seed)
        if training:
            record["calls"] = calls
        return record

    try:
        results = await asyncio.gather(*(episode(row) for row in rows))
    finally:
        await sampler.close()
    validate_episodes(results, rows, seed, config["solver_tokens"], exact_tokens=training)
    return {"accuracy": sum(r["reward"] for r in results) / len(results),
            "mean_turn_count": sum(r["turns"] for r in results) / len(results),
            "question_count": len(rows), "samples_per_question": 1, "metric": "mean@1",
            "tokens": sum(r["tokens"] for r in results), "results": results}


@contextmanager
def environment(launch, root: Path, harness: Path, rows: list[dict], config: dict):
    from .alfworld_solver_eval import bind_environment
    assignment = {"name": "whale-alfworld", "harness": str(harness),
                  "task_ids": sorted({row["task_id"] for row in rows})}
    if rows[0].get("task_split", "test_set") != "test_set":
        assignment.update(split=rows[0]["task_split"], catalogue=rows[0]["task_catalogue"],
                          catalogue_sha256=rows[0]["task_catalogue_sha256"])
    write_json(root / "assignments.json", [assignment])
    process = launch.module("sao_bridge.alfworld_env_service", [
        "--run", root / "environment", "--assignments", root / "assignments.json",
        "--controller-port", "28600", "--worker-port", "28610",
        "--concurrency", config["episode_concurrency"], "--async-controller",
    ], gpus="", name=f"environment-{root.name}")
    try:
        controller = "http://127.0.0.1:28600/api"
        launch.ready(process, controller + "/list_workers")
        deadline = time.monotonic() + 900
        while not (root / "environment/READY.json").is_file():
            launch.check()
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("ALFWorld worker readiness failed")
            time.sleep(1)
        ready = read_json(root / "environment/READY.json")
        if sorted(ready["health"]["indices"]["whale-alfworld"]) != sorted(assignment["task_ids"]):
            raise RuntimeError("Official worker exposed the wrong task cohort")
        bind_environment(launch, controller, 900)
        yield [dict(row, task_name="whale-alfworld") for row in rows]
    finally:
        launch.stop(process)


@contextmanager
def serving(launch, model: Path, name: str):
    with launch.vllm(
        str(model), port=19701, gpus=GPUS, name=name, data_parallel=8,
        max_model_len=32768, tool_call_parser="qwen3_xml", gdn_prefill_backend="triton",
        language_model_only=True, max_num_seqs=8, server_args=["--enable-prefix-caching"],
    ) as urls:
        yield loopback_urls(urls)


def score(launch, model: Path, harness: Path, rows: list[dict], root: Path,
          config: dict, seed: int, *, training: bool = False) -> dict:
    with environment(launch, root, harness, rows, config) as assigned:
        with serving(launch, model, f"solver-{root.name}") as urls:
            report = asyncio.run(collect(urls, assigned, harness, config, seed, training=training))
    write_json(root / "report.json", report)
    return report


def proposal_messages(config: dict, incumbent: Path, archive: list[dict],
                      evidence: dict, slot: int) -> list[dict]:
    packets = read_json(ROOT / config["prompt_contract"])
    text = packets[0]["messages"][1]["content"]
    marker = "Failure Trace Packet"
    if text.count(marker) != 1:
        raise ValueError("Frozen prompt contract delimiter changed")
    contract = text.split(marker)[0]
    traces = [{"task_id": r["task_id"], "reward": r["reward"], "turns": r["turns"],
               "messages": r["messages"]} for r in evidence["results"] if not r["reward"]]
    # Fixed byte budget; never draw on final-evaluation reports or old W0 traces.
    context = json.dumps(traces, ensure_ascii=False)[:config["evidence_character_budget"]]
    task = (
        "\nMeta-Harness search: improve the entire reusable harness under the current "
        "frozen Solver, using the current incumbent, archive and fresh DEVELOPMENT traces. "
        "Do not change the verifier, token/step budgets, model or task catalogue. "
        "Generalize beyond these tasks. Propose one complete replacement, not a delta. "
        f"This is candidate slot {slot}; make a distinct hypothesis.\n"
        f"Incumbent:\n{incumbent.read_text()}\n"
        f"Search archive:\n{json.dumps(archive)}\n"
        f"Fresh development failures (bounded character excerpt):\n{context}\n"
    )
    return [packets[0]["messages"][0], {"role": "user", "content": contract + task},
            {"role": "assistant", "content": "<think>\n"}]


async def propose(urls: str, messages: list[dict], config: dict, seed: int) -> dict:
    import httpx
    url = loopback_urls(urls).split(",")[seed % 8]
    async with httpx.AsyncClient(timeout=900, trust_env=False) as client:
        body = {"model": "policy", "messages": messages, "temperature": 0.7, "top_p": 1.0,
                "top_k": -1, "seed": seed, "max_tokens": config["proposer_tokens"],
                "chat_template_kwargs": {"enable_thinking": True, "add_generation_prompt": False,
                                         "continue_final_message": True}}
        response = await client.post(url + "/chat/completions", json=body)
        response.raise_for_status()
        result = response.json()
    choice = result["choices"][0]
    text = choice["message"].get("content")
    if not isinstance(text, str) or not result["usage"].get("completion_tokens"):
        raise RuntimeError("Local H0 response missing completion/token accounting")
    output = {"text": text, "seed": seed, "usage": result["usage"],
              "finish_reason": choice["finish_reason"]}
    try:
        if choice["finish_reason"] == "length":
            raise ValueError("Proposer response truncated")
        output.update(valid=True, patch=parse_candidate(text))
    except ValueError as exc:
        output.update(valid=False, error=str(exc))
    return output


def harness_search(launch, model: Path, incumbent: Path, rows: list[dict],
                   root: Path, config: dict, cycle: int) -> Path:
    h0 = root / "harnesses/h0/harness.py"
    immutable(h0, read_json(incumbent))
    incumbent = h0
    report = score(launch, model, h0, rows, root / "h0", config, config["development_seed"])
    points = [("h0", report["accuracy"], report["mean_turn_count"])]
    reports = {"h0": report}
    archive = [{"name": "h0", "success_rate": report["accuracy"], "turns": report["mean_turn_count"]}]
    next_name = 1
    for iteration in range(config["mh_iterations"]):
        outputs = []
        with serving(launch, ROOT / config["fixed_proposer"], f"h0-c{cycle}-i{iteration}") as urls:
            for slot in range(config["proposals_per_iteration"]):
                messages = proposal_messages(config, incumbent, archive, report, slot)
                output = asyncio.run(propose(
                    urls, messages, config,
                    config["proposer_seed"] + cycle * 10000 + iteration * 100 + slot))
                output["name"] = f"h{next_name}"
                next_name += 1
                write_json(root / "proposals" / f"{output['name']}.json",
                           dict(output, messages=messages))
                outputs.append(output)
        for output in outputs:
            name = output["name"]
            if not output["valid"]:
                archive.append({"name": name, "valid": False, "error": output["error"]})
                continue
            candidate = root / "harnesses" / name / "harness.py"
            immutable(candidate, output["patch"])
            result = score(launch, model, candidate, rows, root / name, config,
                           config["development_seed"])
            reports[name] = result
            points.append((name, result["accuracy"], result["mean_turn_count"]))
            archive.append({"name": name, "valid": True, "success_rate": result["accuracy"],
                            "turns": result["mean_turn_count"], "description": output["patch"].get("description", "")})
        accepted, frontier = select_harness(points, root)
        incumbent = root / "harnesses" / accepted / "harness.py"
        report = reports[accepted]
        write_json(root / f"iteration-{iteration:02d}.json",
                   {"accepted": accepted, "frontier": frontier, "archive": archive})
    write_json(root / "SELECTED.json", {"harness": str(incumbent), "sha256": file_sha(incumbent),
                                       "model": str(model), "development_only": True})
    return incumbent


def native_batch(calls: list[dict], pad: int, mini_batch: int):
    import torch
    from verl import DataProto
    pmax, rmax = max(len(c["prompt_ids"]) for c in calls), max(len(c["response_ids"]) for c in calls)
    ids = torch.full((len(calls), pmax + rmax), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    response_mask = torch.zeros((len(calls), rmax), dtype=torch.long)
    for index, call in enumerate(calls):
        prompt, response = call["prompt_ids"], call["response_ids"]
        ids[index, pmax-len(prompt):pmax] = torch.tensor(prompt)
        ids[index, pmax:pmax+len(response)] = torch.tensor(response)
        attention[index, pmax-len(prompt):pmax+len(response)] = 1
        response_mask[index, :len(response)] = 1
    return DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": attention,
        "position_ids": (attention.cumsum(-1) - 1).clamp(min=0),
        "responses": ids[:, pmax:].clone(), "response_mask": response_mask,
    }, meta_info={"temperature": 1.0, "pad_token_id": pad, "sft_epochs": 1,
                  "sft_mini_batch_size": mini_batch, "sft_micro_batch_size_per_gpu": 1})


def smoke_calls(tokenizer) -> list[dict]:
    def repeat(tokens, length):
        if not tokens or length < 0:
            raise ValueError("Cannot construct fixed-length synthetic smoke tokens")
        return (tokens * ((length + len(tokens) - 1) // len(tokens)))[:length]

    calls = []
    for index in range(2):
        base = tokenizer.apply_chat_template(
            [{"role": "system", "content": "Synthetic memory validation only; this is not a benchmark task."},
             {"role": "user", "content": f"Synthetic observation stream {index}. "
              "Inspect the room and finish with a take_action tool call for look."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
        if hasattr(base, "get"):
            base = base["input_ids"]
        if not base or len(base) > 28672:
            raise ValueError("Unexpected synthetic smoke chat template length")
        observations = tokenizer.encode(
            "Synthetic observation: visible shelves, open passage, empty inventory. ", add_special_tokens=False)
        reasoning = tokenizer.encode(
            "Inspect the visible room, retain the observation, and avoid repeating an action. ",
            add_special_tokens=False)
        ending = tokenizer.encode(
            '\n<tool_call>\n{"name":"take_action","arguments":{"action":"look"}}\n</tool_call>',
            add_special_tokens=False) + [tokenizer.eos_token_id]
        tail = min(16, len(base))
        prompt = base[:-tail] + repeat(observations, 28672 - len(base)) + base[-tail:]
        response = repeat(reasoning, 4096 - len(ending)) + ending
        if (len(prompt) != 28672 or len(response) != 4096
                or any(type(value) is not int or value < 0 for value in prompt + response)):
            raise ValueError("Synthetic smoke must use exactly 32768 context /4096 response tokens")
        calls.append({"prompt_ids": prompt, "response_ids": response, "temperature": 1.0})
    return calls


def full_context_logprobs(model, call: dict, device, temperature: float):
    import torch
    from harness_r1_searchqa.grpo_objective import response_logprobs
    if len(call["prompt_ids"]) != 28672 or len(call["response_ids"]) != 4096:
        raise ValueError("Full-context smoke dimensions changed")
    inputs = torch.tensor([call["prompt_ids"] + call["response_ids"]], device=device)
    targets = torch.tensor(call["response_ids"], device=device)
    # Include even the last target token so the actual model forward reaches
    # 32768 positions. The final logit is unused by the shifted response loss.
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=4097).logits[0]
    if logits.shape[0] != 4097:
        raise RuntimeError("Full-context smoke requires exact response-only logits")
    return response_logprobs(logits[:-1], targets, temperature)


def require_full_context_smoke(result: dict) -> None:
    if (result.get("status") != "passed"
            or result.get("full_context_memory_validated") is not True
            or result.get("full_context_tokens") != 32768
            or result.get("response_mask_tokens") != 4096
            or result.get("optimizer_steps_before_restore") != 2
            or result.get("logical_optimizer_steps") != 3
            or result.get("exact_fp32_and_adam_restore") is not True
            or result.get("vllm_export_load_verified") is not True):
        raise ValueError("A passed full-context 32768/4096 warm-optimizer smoke is required")


def state_fingerprint(value) -> str:
    """Hash all local parameter/Adam tensors, not a sampling probe."""
    import torch
    result = hashlib.sha256()

    def visit(item):
        if torch.is_tensor(item):
            result.update(str((str(item.dtype), tuple(item.shape))).encode())
            raw = item.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            for start in range(0, raw.size, 2**20):
                result.update(raw[start:start + 2**20].tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                result.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            result.update(str(len(item)).encode())
            for child in item:
                visit(child)
        else:
            result.update(repr(item).encode())

    visit(value)
    return result.hexdigest()


def optimizer_steps(optimizer) -> list[int]:
    return sorted({int(state["step"].item()) for state in optimizer.state.values() if "step" in state})


def cost_report(root: Path, config: dict) -> dict:
    reports = {str(path.relative_to(root)): read_json(path) for path in root.glob("**/report.json")}
    proposals = [read_json(path) for path in root.glob("mh-*/proposals/*.json")]
    updates = [read_json(path) for path in root.glob("cycle-*/optimizer/COMMITTED.json")]
    calls = sum(value["accepted_calls"] for value in updates)
    tokens = sum(value["response_tokens"] for value in updates)
    return {
        "actual_solver_episodes": sum(len(value["results"]) for value in reports.values()),
        "training_rollout_episodes": sum(len(v["results"]) for p, v in reports.items() if p.startswith("cycle-")),
        "development_reward_episodes": sum(len(v["results"]) for p, v in reports.items() if p.startswith("mh-")),
        "final_report_episodes": sum(len(v["results"]) for p, v in reports.items() if p.startswith("final200/")),
        "actual_solver_generated_tokens": sum(value["tokens"] for value in reports.values()),
        "proposer_completions": len(proposals),
        "proposer_prompt_tokens": sum(value["usage"]["prompt_tokens"] for value in proposals),
        "proposer_generated_tokens": sum(value["usage"]["completion_tokens"] for value in proposals),
        "invalid_proposals": sum(not value["valid"] for value in proposals),
        "logical_rsft_forward_calls": calls,
        "logical_rsft_response_tokens": tokens,
        "rank_replicated_rsft_forward_calls": calls * config["world_size"],
        "rank_replicated_rsft_response_token_positions": tokens * config["world_size"],
        "logical_optimizer_steps": sum(value["metrics"]["actor/sft_updates"] for value in updates),
        "replicated_rank_multiplier": config["world_size"],
        "frozen_preparation_h0_proposals_inherited_not_in_incremental_cost": 8,
        "inherited_initialization_cost": "Shared H0 selection/evidence and historical baseline reports "
                                       "are not free; their original receipts are separate, not reexecuted.",
        "smoke_cost": "Separate SMOKE_COMPLETE.json; include its elapsed time and synthetic steps in total preparation.",
        "compute_not_matched_to_sao_or_ma": True,
        "not_measured": ["FLOPs", "GPU-hours", "checkpointing recomputation token positions"],
    }


def train_native(request_path: Path) -> None:
    """Exact upstream token-mean RSFT; identical singleton inputs on each FSDP rank."""
    started = time.monotonic()
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from transformers import AutoTokenizer
    from harness_r1_searchqa.grpo_train import _load_fsdp, _logprobs, _export_actor, _rank_zero
    from .common import ensure_model_runtime_metadata
    from .resources import require_research_node

    execution_source_audit()
    require_research_node()
    request = read_json(request_path)
    config = request["config"]
    validate_config(config)
    if file_sha(Path(__file__)) != request["adapter_sha256"]:
        raise ValueError("Adapter source changed before native RSFT")
    for name, expected in config["pinned_files"].items():
        if file_sha(ROOT / name) != expected:
            raise ValueError(f"Frozen dependency changed before native RSFT: {name}")
    rank, world, local = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
    if world != config["world_size"]:
        raise ValueError("Frozen FSDP topology changed")
    sys.path.insert(0, str(NATIVE))
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config.actor import FSDPActorConfig
    import inspect
    if Path(inspect.getfile(DataParallelPPOActor)).resolve() != (NATIVE / "verl/workers/actor/dp_actor.py").resolve():
        raise RuntimeError("Wrong native WHALE actor imported")
    smoke = request.get("purpose") == "synthetic_gpu_smoke"
    calls, report = None, None
    if smoke:
        if any(key in request for key in ("report", "rows", "reward")):
            raise ValueError("GPU smoke must not contain research trajectories/rewards")
        if not request.get("resume") and Path(request["model"]).resolve() != (ROOT / config["initial_solver"]).resolve():
            raise ValueError("GPU smoke must start from original W0")
    else:
        report_path = Path(request["report"])
        if file_sha(report_path) != request["report_sha256"]:
            raise ValueError("Rollout report changed before RSFT")
        report = read_json(report_path)
        validate_episodes(report["results"], request["rows"], request["seed"], config["solver_tokens"],
                          exact_tokens=True)
        calls = accepted_calls(report["results"], config["score_threshold"])
        if not calls:
            raise ValueError("Empty RSFT batch must be skipped explicitly by the controller")
    output = Path(request["output"])
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    try:
        torch.manual_seed(config["training_seed"])
        model = _load_fsdp(Path(request["model"]), device, allow_vlm=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"], foreach=False)
        tokenizer = AutoTokenizer.from_pretrained(request["model"], local_files_only=True)
        if smoke:
            calls = smoke_calls(tokenizer)
        restored_exactly = False
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType
        if request.get("resume"):
            previous = Path(request["resume"])
            receipt = read_json(previous / "COMMITTED.json")
            saved_path = previous / f"rank-{rank:04d}.pt"
            if (receipt["config_sha256"] != digest(config)
                    or file_sha(saved_path) != receipt["rank_state_sha256"][str(rank)]
                    or Path(receipt["checkpoint"]).resolve() != Path(request["model"]).resolve()):
                raise ValueError("Native optimizer continuation identity mismatch")
            saved = torch.load(saved_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(saved["optimizer"])
            with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
                model.load_state_dict(saved["actor_local_state"])
            torch.set_rng_state(saved["rng_cpu"])
            torch.cuda.set_rng_state(saved["rng_cuda"], device)
            if smoke:
                if (state_fingerprint(dict(model.named_parameters())) != saved["proof"]["parameters"]
                        or state_fingerprint(optimizer.state_dict()) != saved["proof"]["optimizer"]
                        or optimizer_steps(optimizer) != [2]):
                    raise RuntimeError("GPU smoke failed exact FP32 parameter/Adam restoration")
                restored_exactly = True
            del saved

        forward_evidence = []

        class SingletonActor(DataParallelPPOActor):
            def _forward_micro_batch(self, batch, temperature, calculate_entropy=False):
                if batch["input_ids"].shape[0] != 1 or calculate_entropy:
                    raise ValueError("WHALE Qwen3.5 RSFT requires singleton microbatches")
                width = batch["responses"].shape[1]
                prompt_width = batch["input_ids"].shape[1] - width
                prompt = batch["input_ids"][0, :prompt_width][
                    batch["attention_mask"][0, :prompt_width].bool()].tolist()
                response = batch["responses"][0][batch["response_mask"][0].bool()].tolist()
                call = {"prompt_ids": prompt, "response_ids": response}
                if smoke:
                    lp = full_context_logprobs(self.actor_module, call, device, temperature)
                    forward_evidence.append({
                        "input_tokens": len(prompt) + len(response), "response_mask_tokens": len(response),
                        "optimizer_steps": optimizer_steps(optimizer),
                    })
                else:
                    lp = _logprobs(self.actor_module, call, device, temperature)
                return {"log_probs": torch.nn.functional.pad(lp, (0, width - len(response)))[None]}

            def _optimizer_step(self):
                norm = super()._optimizer_step()
                if not torch.isfinite(norm):
                    raise FloatingPointError("Native RSFT nonfinite gradient; checkpoint rejected")
                return norm

        actor = SingletonActor(FSDPActorConfig(
            rollout_n=1, ppo_mini_batch_size=config["sft_mini_batch_size"],
            ppo_micro_batch_size_per_gpu=1, use_remove_padding=False,
            use_torch_compile=False, use_dynamic_bsz=False, grad_clip=1.0,
        ), model, optimizer)
        before = [p.detach().clone() for p in model.parameters()]
        steps_before = optimizer_steps(optimizer)
        updates = 2 if smoke and not request.get("resume") else 1
        batch = native_batch(calls, tokenizer.pad_token_id, config["sft_mini_batch_size"])
        update_metrics = [actor.update_sft_policy(batch) for _ in range(updates)]
        metrics = update_metrics[0] if not smoke else {
            "actor/sft_updates": sum(value["actor/sft_updates"] for value in update_metrics),
            "actor/sft_token_count": sum(value["actor/sft_token_count"] for value in update_metrics),
            "updates": update_metrics,
        }
        if smoke and optimizer_steps(optimizer) != [3 if request.get("resume") else 2]:
            raise RuntimeError("GPU smoke warm-optimizer step count is incorrect")
        changed = any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
        del before
        change_tensor = torch.tensor(int(changed), device=device)
        dist.all_reduce(change_tensor)
        if not change_tensor.item():
            raise RuntimeError("Native RSFT produced no changed weight shards")
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise FloatingPointError("Native RSFT nonfinite parameter")
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise FloatingPointError("Native RSFT nonfinite optimizer state")
        checkpoint = output / "actor"
        identity = _export_actor(model, tokenizer, checkpoint, rank)
        _rank_zero(lambda: ensure_model_runtime_metadata(checkpoint, ROOT / config["initial_solver"]), rank)
        with FSDP.state_dict_type(model, StateDictType.LOCAL_STATE_DICT):
            local_state = model.state_dict()
        saved = {"optimizer": optimizer.state_dict(), "actor_local_state": local_state,
                 "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(device)}
        if smoke:
            saved["proof"] = {"parameters": state_fingerprint(dict(model.named_parameters())),
                              "optimizer": state_fingerprint(optimizer.state_dict())}
        torch.save(saved, output / f"rank-{rank:04d}.pt")
        smoke_proofs = [None] * world
        if smoke:
            dist.all_gather_object(smoke_proofs, {
                "rank": rank, "steps_before": steps_before, "steps_after": optimizer_steps(optimizer),
                "restored_exactly": restored_exactly, "proof": saved["proof"],
                "forward_evidence": forward_evidence,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device),
            })
        dist.barrier()
        _rank_zero(lambda: write_json(output / "COMMITTED.json", {
            "checkpoint": str(checkpoint), "config_sha256": digest(config),
            "identity": identity, "metrics": metrics, "accepted_calls": 0 if smoke else len(calls),
            "accepted_episodes": 0 if smoke else sum(r["reward"] > 0.5 for r in report["results"]),
            "purpose": "synthetic_gpu_smoke" if smoke else "research_rsft",
            "elapsed_seconds": time.monotonic() - started,
            "synthetic_calls": len(calls) * updates if smoke else 0,
            "smoke_proofs": smoke_proofs if smoke else None,
            "response_tokens": updates * sum(len(c["response_ids"]) for c in calls),
            "rank_state_sha256": {str(i): file_sha(output / f"rank-{i:04d}.pt") for i in range(world)},
            "native_rsft": True, "replicated_data_across_ranks": world, "weight_shards_changed": True,
            "rollout_reload_verified": False,
        }), rank)
    finally:
        dist.destroy_process_group()


def run_gpu_smoke(frozen_path: Path, assigned_node: str, output: Path) -> dict:
    from .resources import gpu_release_audit
    started = time.monotonic()
    frozen = read_json(frozen_path)
    if (assigned_node not in NODES or os.uname().nodename != assigned_node
            or (os.getuid(), os.getgid()) != NODES[assigned_node]):
        raise RuntimeError("GPU smoke requires explicitly assigned node and correct UID/GID")
    validate_frozen(frozen_path)
    if not frozen["model_weights_verified"]:
        raise ValueError("GPU smoke source/input freeze changed")
    output = run_output_path(output)
    execution = frozen_path.parent / "execution"
    if (output == execution or execution in output.parents or output in execution.parents
            or output.exists()):
        raise ValueError("GPU smoke needs a fresh separate runs/ output directory")
    release = gpu_release_audit()
    if (release["foreign_gpu_pids"] or release["filler_kind"] != "adaptive"
            or not release["filler_unit_active"]):
        raise RuntimeError("GPU smoke allocation is not exclusively free with adaptive filler")
    output.mkdir(parents=True)
    launch = create_launcher(output / "supervision", budget=WallBudget(7200))
    stages = []
    model, resume = ROOT / frozen["config"]["initial_solver"], None
    try:
        for stage in range(2):
            request = {"config": frozen["config"], "adapter_sha256": frozen["adapter_sha256"],
                       "purpose": "synthetic_gpu_smoke", "model": str(model),
                       "resume": str(resume) if resume else None,
                       "output": str(output / f"stage-{stage:02d}")}
            path = output / f"request-{stage:02d}.json"
            write_json(path, request)
            child = launch.start([
                sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                "--local-addr=127.0.0.1", "--nproc-per-node=8", "--max-restarts=0",
                "-m", "sao_bridge.alfworld_whale_baseline", "train-native", "--request", str(path),
            ], gpus=GPUS, name=f"smoke-{stage}")
            launch.wait(child, "real W0 RSFT GPU smoke", timeout=3000)
            receipt = read_json(Path(request["output"]) / "COMMITTED.json")
            expected_before = [] if stage == 0 else [2]
            expected_calls = 4 if stage == 0 else 2
            if (receipt["purpose"] != "synthetic_gpu_smoke" or receipt["accepted_episodes"] != 0
                    or receipt["synthetic_calls"] != expected_calls or len(receipt["smoke_proofs"]) != 8
                    or {proof.get("rank") for proof in receipt["smoke_proofs"]} != set(range(8))
                    or receipt["response_tokens"] != expected_calls * 4096
                    or any(proof["steps_before"] != expected_before or proof["steps_after"] != [2 if stage == 0 else 3]
                           or (stage == 1 and not proof["restored_exactly"])
                           for proof in receipt["smoke_proofs"])):
                raise RuntimeError("GPU smoke native proof incomplete")
            for proof in receipt["smoke_proofs"]:
                evidence = proof.get("forward_evidence", [])
                expected_steps = [[], [], [1], [1]] if stage == 0 else [[2], [2]]
                if (len(evidence) != expected_calls
                        or any(row.get("input_tokens") != 32768 or row.get("response_mask_tokens") != 4096
                               for row in evidence)
                        or [row.get("optimizer_steps") for row in evidence] != expected_steps):
                    raise RuntimeError("GPU smoke full-context/warm-optimizer forward proof incomplete")
            stages.append(receipt)
            model, resume = Path(receipt["checkpoint"]), Path(request["output"])
        with serving(launch, model, "smoke-vllm-reload"):
            pass
        result = {
            "status": "passed", "purpose": "synthetic_gpu_smoke",
            "frozen_sha256": file_sha(frozen_path), "adapter_sha256": frozen["adapter_sha256"],
            "initial_model": str(ROOT / frozen["config"]["initial_solver"]),
            "forward_backward_export": True, "exact_fp32_and_adam_restore": True,
            "vllm_export_load_verified": True, "research_episodes": 0, "synthetic_calls": 6,
            "logical_optimizer_steps": 3, "rank_replicated_forward_calls": 48,
            "full_context_tokens": 32768, "response_mask_tokens": 4096,
            "optimizer_steps_before_restore": 2, "full_context_memory_validated": True,
            "logical_synthetic_response_tokens": sum(r["response_tokens"] for r in stages),
            "rank_replicated_synthetic_response_positions": 8 * sum(r["response_tokens"] for r in stages),
            "elapsed_seconds": time.monotonic() - started,
            "limitation": "Synthetic full-context memory/restore validation only, not ALFWorld performance.",
            "stage_receipts": [str(output / f"stage-{i:02d}/COMMITTED.json") for i in range(2)],
        }
        require_full_context_smoke(result)
        write_json(output / "SMOKE_COMPLETE.json", result)
        return result
    except BaseException as exc:
        write_json(output / "FAILED.json", {"error": repr(exc)})
        raise
    finally:
        launch.close()
        write_json(output / "RELEASED.json", gpu_release_audit())


def queue_prerequisites(complete: Path, released: Path) -> bool:
    failed = complete.with_name(complete.name.removesuffix("_COMPLETE.json") + "_FAILED.json")
    for failure in (failed, complete.parent / "FAILED.json"):
        if failure.exists():
            raise RuntimeError(f"MA prerequisite failed: {failure}")
    if not complete.is_file() or not released.is_file():
        return False
    if read_json(complete).get("status") not in {"completed", "unavailable"}:
        raise ValueError("MA best evaluation must be completed or unavailable, never failed")
    release = read_json(released)
    if (release.get("foreign_gpu_pids") != [] or release.get("filler_kind") != "adaptive"
            or release.get("filler_unit_active") is not True):
        raise ValueError("MA best release receipt is not clean with active adaptive filler")
    return True


def queued_run(frozen_path: Path, expected_sha: str, assigned_node: str, root: Path,
               smoke_output: Path, complete: Path, released: Path, wait_seconds: int) -> dict:
    """CPU-only waiting; the first GPU action follows all immutable/live gates."""
    from .resources import gpu_release_audit
    if (assigned_node != "dev-NDH200-026" or os.uname().nodename != assigned_node
            or (os.getuid(), os.getgid()) != NODES[assigned_node] or not 1 <= wait_seconds <= 86400):
        raise ValueError("Persistent WHALE queue is explicitly node4-only")
    root = run_output_path(root)
    request = {"frozen": str(frozen_path.resolve()), "frozen_sha256": expected_sha,
               "node": assigned_node, "smoke_output": str(smoke_output.resolve()),
               "complete": str(complete.resolve()), "released": str(released.resolve()),
               "wait_seconds": wait_seconds}
    root.mkdir(parents=True, exist_ok=True)
    if (root / "QUEUE.json").exists():
        saved = read_json(root / "QUEUE.json")
        if saved["request"] != request:
            raise ValueError("Cannot replace a persistent queue's request")
    else:
        saved = {"request": request, "deadline_unix": time.time() + wait_seconds}
        immutable(root / "QUEUE.json", saved)
    if (root / "STARTED.json").exists():
        raise RuntimeError("GPU phase was already started; automatic research restart is forbidden")
    baseline_started = False
    stage = "waiting_for_joint"
    write_json(root / "STATUS.json", {
        "stage": stage, "baseline_started": False,
        "gpu_allocation_started": False, "deadline_unix": saved["deadline_unix"],
        "purpose": "queue_wait_not_training",
    })
    try:
        while True:
            if file_sha(frozen_path) != expected_sha:
                raise ValueError("Queued source freeze digest changed")
            frozen = read_json(frozen_path)
            execution_source_audit(frozen)
            if (file_sha(Path(__file__)) != frozen["adapter_sha256"]
                    or file_sha(frozen_config_path(frozen)) != frozen["config_sha256"]):
                raise ValueError("Queued adapter/config no longer matches source freeze")
            if time.time() >= saved["deadline_unix"]:
                raise TimeoutError("Persistent WHALE queue deadline exceeded")
            ready = queue_prerequisites(complete, released)
            live = gpu_release_audit() if ready else None
            if ready and live["foreign_gpu_pids"] == [] and live["filler_kind"] == "adaptive" and live["filler_unit_active"]:
                break
            write_json(root / "WAITING.json", {
                "stage": "waiting_for_joint", "purpose": "queue_wait_not_training",
                "prerequisite_receipts_ready": ready, "live_release": live,
                "checked_at_unix": time.time(), "deadline_unix": saved["deadline_unix"],
            })
            time.sleep(60)
        immutable(root / "STARTED.json", {
            "frozen_sha256": expected_sha, "ma_complete_sha256": file_sha(complete),
            "ma_release_sha256": file_sha(released), "live_release": live,
            "ma_status": read_json(complete)["status"],
            "started_at_unix": time.time(),
        })
        stage = "gpu_smoke"
        write_json(root / "STATUS.json", {
            "stage": stage, "baseline_started": False, "gpu_allocation_started": True,
        })
        smoke = run_gpu_smoke(frozen_path, assigned_node, smoke_output)
        if (smoke["status"] != "passed" or smoke.get("frozen_sha256") != expected_sha
                or file_sha(frozen_path) != expected_sha):
            raise RuntimeError("Queued GPU smoke did not pass")
        require_full_context_smoke(smoke)
        baseline_started = True
        stage = "baseline"
        write_json(root / "STATUS.json", {
            "stage": stage, "baseline_started": True, "gpu_allocation_started": True,
        })
        result = run(frozen_path, assigned_node)
        write_json(root / "COMPLETE.json", {"smoke": smoke, "baseline": result})
        write_json(root / "STATUS.json", {"stage": "completed", "baseline_started": True})
        return result
    except BaseException as exc:
        queue_failure = root / "FAILED.json"
        failure = {"status": "failed", "stage": stage,
                   "baseline_started": baseline_started, "error": repr(exc)}
        write_json(queue_failure, failure)
        execution_failure = frozen_path.parent / "execution/FAILED.json"
        if not execution_failure.exists():
            write_json(execution_failure, {
                **failure, "originating_queue_receipt": str(queue_failure),
            })
        write_json(root / "STATUS.json", {
            "stage": "failed", "baseline_started": baseline_started, "error": repr(exc),
        })
        raise


def run(frozen_path: Path, assigned_node: str, *, wait_for_joint: Path | None = None,
        wait_label: str = "best", frozen_sha256: str | None = None) -> dict:
    if wait_for_joint is not None:
        if wait_label != "best":
            raise ValueError("WHALE waits only for MA best on node4")
        return queued_run(
            frozen_path, frozen_sha256 or file_sha(frozen_path), assigned_node,
            frozen_path.parent / "queue", frozen_path.parent / "gpu-smoke-queued",
            wait_for_joint / f"{wait_label}_COMPLETE.json",
            wait_for_joint / f"{wait_label}_RELEASED.json", 86400)
    if frozen_sha256 is not None and file_sha(frozen_path) != frozen_sha256:
        raise ValueError("Explicit source freeze digest changed")
    from .resources import gpu_release_audit
    frozen = read_json(frozen_path)
    config = frozen["config"]
    if (assigned_node not in NODES or os.uname().nodename != assigned_node
            or (os.getuid(), os.getgid()) != NODES[assigned_node]):
        raise RuntimeError("Explicit assigned research node and correct UID/GID required")
    if not frozen["model_weights_verified"]:
        raise ValueError("Execution requires a full weight-verified CPU preparation")
    current = preparation(frozen_config_path(frozen))
    if current != frozen:
        raise ValueError("Preparation freeze changed; do not re-freeze an active experiment")
    release = gpu_release_audit()
    if (release["foreign_gpu_pids"] or release["filler_kind"] != "adaptive"
            or not release["filler_unit_active"]):
        raise RuntimeError("Assigned node not idle with verified active adaptive filler")
    root = frozen_path.parent / "execution"
    root.mkdir(exist_ok=False)
    started = time.monotonic()
    launch = create_launcher(root / "supervision", budget=WallBudget(config["wall_budget_seconds"]))
    write_json(root / "resource_audit.json", release)
    all_rows = read_json(ROOT / config["training_rows"])
    rows = training_episode_rows(all_rows, frozen["plan"]["train_record_indices"])
    all_dev = read_json(ROOT / config["development_rows"])
    dev = [all_dev[index] for index in frozen["plan"]["development_ids"]]
    model, harness = ROOT / config["initial_solver"], ROOT / config["initial_harness"]
    resume = None
    updates, consumed = [], 0

    def check_freeze():
        execution_source_audit(frozen)
        if (file_sha(frozen_config_path(frozen)) != frozen["config_sha256"]
                or file_sha(Path(__file__)) != frozen["adapter_sha256"]):
            raise RuntimeError("Frozen WHALE configuration/adapter changed during execution")
        for name, expected in config["pinned_files"].items():
            if file_sha(ROOT / name) != expected:
                raise RuntimeError(f"Frozen dependency changed during execution: {name}")

    try:
        for cycle in range(config["cycles"]):
            for step in range(config["rsft_batches_per_cycle"]):
                check_freeze()
                work = root / f"cycle-{cycle:02d}-rsft-{step:02d}"
                batch = rows[consumed:consumed + config["episodes_per_batch"]]
                seed = config["training_seed"] + consumed * 1009
                report = score(launch, model, harness, batch, work, config, seed, training=True)
                consumed += len(batch)
                if not accepted_calls(report["results"]):
                    updates.append({"consumed": consumed, "updated": False, "reason": "no accepted trajectories"})
                    write_json(work / "SKIPPED.json", updates[-1])
                    continue
                request = {"config": config, "adapter_sha256": frozen["adapter_sha256"],
                           "model": str(model), "resume": str(resume) if resume else None,
                           "rows": batch, "seed": seed, "report": str(work / "report.json"),
                           "report_sha256": file_sha(work / "report.json"), "output": str(work / "optimizer")}
                write_json(work / "request.json", request)
                child = launch.start([
                    sys.executable, "-B", "-m", "torch.distributed.run", "--standalone",
                    "--local-addr=127.0.0.1", "--nproc-per-node=8", "--max-restarts=0",
                    "-m", "sao_bridge.alfworld_whale_baseline", "train-native",
                    "--request", str(work / "request.json"),
                ], gpus=GPUS, name=f"rsft-{cycle}-{step}")
                launch.wait(child, "native WHALE RSFT", timeout=config["phase_timeout_seconds"])
                receipt = read_json(work / "optimizer/COMMITTED.json")
                model, resume = Path(receipt["checkpoint"]), work / "optimizer"
                # A persisted export is not enough: vLLM must load it before promotion.
                with serving(launch, model, f"verify-rsft-{cycle}-{step}"):
                    pass
                updates.append({"consumed": consumed, "updated": True, "optimizer": str(resume),
                                "rollout_reload_verified": True})
                write_json(work / "RELOAD_VERIFIED.json", updates[-1])
            harness = harness_search(launch, model, harness, dev, root / f"mh-{cycle:02d}", config, cycle)
            write_json(root / "PROGRESS.json", {"cycles_completed": cycle + 1, "updates": updates})
        if consumed != len(rows):
            raise RuntimeError("Incomplete frozen WHALE budget")
        check_freeze()
        immutable(root / "FINAL_SELECTION.json", {
            "model": str(model), "harness": str(harness), "harness_sha256": file_sha(harness),
            "training_episodes": consumed, "updates": updates, "selected_before_final_evaluation": True,
        })
        evaluation = frozen["final_evaluation"]
        final_rows = [{"id": f"alfworld-{index}", "task_id": index} for index in evaluation["all200"]]
        report = score(launch, model, harness, final_rows, root / "final200", config, evaluation["seed"])
        from .alfworld_core_extend import subset_report
        strict = subset_report(report, evaluation["strict197"])
        complete = {"status": "completed", "label": "local WHALE-style ALFWorld adaptation",
                    "all200": report["accuracy"], "strict197": strict["accuracy"],
                    "training_episodes": consumed, "weight_update_batches": sum(x["updated"] for x in updates),
                    "published_alfworld_baseline": False}
        complete["cost"] = cost_report(root, config)
        complete["cost"]["inherited_initialization"] = frozen.get("inherited_initialization_cost")
        complete["elapsed_seconds"] = time.monotonic() - started
        write_json(root / "COMPLETE.json", complete)
        return complete
    except BaseException as exc:
        write_json(root / "FAILED.json", {"error": repr(exc), "consumed_episodes": consumed})
        raise
    finally:
        launch.close()
        write_json(root / "COST.json", {
            **cost_report(root, config),
            "inherited_initialization": frozen.get("inherited_initialization_cost"),
            "elapsed_seconds": time.monotonic() - started,
        })
        write_json(root / "RELEASED.json", gpu_release_audit())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare", help="CPU-only frozen run specification")
    prepare_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run", help="requires assigned idle node; starts GPU processes")
    run_parser.add_argument("--frozen", type=Path, required=True)
    run_parser.add_argument("--execute-on", choices=sorted(NODES))
    run_parser.add_argument("--validate-only", action="store_true",
                            help="CPU-only full freeze/import-origin validation; no allocation or queue")
    run_parser.add_argument("--gpu-smoke", action="store_true",
                            help="real 32768/4096-context warm-Adam export/restore smoke, no research episodes")
    run_parser.add_argument("--output", type=Path, help="required fresh separate directory for --gpu-smoke")
    run_parser.add_argument("--wait-for-joint", type=Path,
                            help="persistent CPU-only wait, then GPU smoke followed by baseline")
    run_parser.add_argument("--wait-label", choices=["best"], default="best")
    run_parser.add_argument("--frozen-sha256", help="bind the externally approved source freeze")
    queue_parser = sub.add_parser("queue", help="CPU-only wait; smoke+baseline after MA best receipts and source freeze")
    queue_parser.add_argument("--frozen", type=Path, required=True)
    queue_parser.add_argument("--frozen-sha256", required=True)
    queue_parser.add_argument("--execute-on", choices=["dev-NDH200-026"], required=True)
    queue_parser.add_argument("--queue-root", type=Path, required=True)
    queue_parser.add_argument("--smoke-output", type=Path, required=True)
    queue_parser.add_argument("--after-complete", type=Path, required=True)
    queue_parser.add_argument("--after-released", type=Path, required=True)
    queue_parser.add_argument("--wait-seconds", type=int, default=86400)
    train_parser = sub.add_parser("train-native", help="internal torchrun worker")
    train_parser.add_argument("--request", type=Path, required=True)
    exec_parser = sub.add_parser("exec-local", help="internal snapshot-isolated module bootstrap")
    exec_parser.add_argument("module", choices=(
        "sao_bridge.alfworld_env_service", "sao_bridge.alfworld_controller",
        "vllm.entrypoints.openai.api_server"))
    exec_parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == "prepare":
        value = preparation(args.config.resolve())
        immutable(args.output.resolve(), value)
        print(json.dumps({key: value[key] for key in (
            "status", "config_sha256", "planned_training_episodes", "planned_proposals")}))
    elif args.command == "run":
        if args.validate_only:
            if args.wait_for_joint is not None or args.gpu_smoke or args.output is not None:
                parser.error("--validate-only cannot queue, smoke, or create an execution output")
            _, report = validate_frozen(args.frozen.resolve(), args.frozen_sha256)
            print(json.dumps(report))
        elif args.execute_on is None:
            parser.error("--execute-on is required unless --validate-only")
        elif args.wait_for_joint is not None:
            if args.gpu_smoke or args.output is not None:
                parser.error("--wait-for-joint already chains GPU smoke and baseline; omit --gpu-smoke/--output")
            print(json.dumps(run(
                args.frozen.resolve(), args.execute_on, wait_for_joint=args.wait_for_joint.resolve(),
                wait_label=args.wait_label, frozen_sha256=args.frozen_sha256)))
        elif args.gpu_smoke:
            if args.output is None:
                parser.error("--gpu-smoke requires --output")
            if args.frozen_sha256 and file_sha(args.frozen) != args.frozen_sha256:
                parser.error("Explicit source freeze digest changed")
            print(json.dumps(run_gpu_smoke(args.frozen.resolve(), args.execute_on, args.output)))
        else:
            if args.output is not None:
                parser.error("--output is only supported with --gpu-smoke")
            print(json.dumps(run(args.frozen.resolve(), args.execute_on, frozen_sha256=args.frozen_sha256)))
    elif args.command == "queue":
        print(json.dumps(queued_run(
            args.frozen.resolve(), args.frozen_sha256, args.execute_on, args.queue_root,
            args.smoke_output, args.after_complete, args.after_released, args.wait_seconds)))
    elif args.command == "exec-local":
        import runpy
        execution_source_audit()
        sys.argv = [args.module, *args.arguments]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)
    else:
        train_native(args.request.resolve())


if __name__ == "__main__":
    main()
