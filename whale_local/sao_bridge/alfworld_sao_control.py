"""Matched-training-population repaired SAO, not an exact historical reproduction.

CPU: prepare --root runs/alfworld_sao_corrected_v3 --inputs-root /path/to/shared/repository
CPU: validate --root runs/alfworld_sao_corrected_v3 --validate-only
Node3, after W2 recovery COMPLETE+RELEASED: train-role --role w1
Node4, after WHALE COMPLETE+RELEASED: run-h-pipeline (h1, h2, then finalize).
Every command requires --root. A partial stage is never resumed or retried.
Prepare and execute from the same immutable source snapshot. Inputs may reside
in the shared repository; source hashes never follow its changing Git HEAD.
The declared third_party/sao dependency may be linked outside the snapshot;
its SAO/AReaL repository commits, resolved roots and selected files stay pinned.
Entrypoints prune foreign local-source paths injected by .pth files; supervised
bridge/vLLM children use the same wrapper without modifying the shared venv.

The prespecified H1/H2 endpoint is step31 (not historical selected H1 step15).
Eight fresh proposals per final Engineer are selected on selection24 only.
W1 is always step79. The final three systems share one fresh all200 matrix;
strict197 is a subset of those same trajectories, not another evaluation.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def isolated_import_paths(paths: list[str], source_root: Path) -> tuple[list[str], list[Path]]:
    source_root = source_root.resolve()
    allowed = [
        source_root / "whale_local",
        (source_root / "third_party/sao/vendor/AReaL").resolve(),
        (source_root / "third_party/sao").resolve(),
    ]
    result, rejected = [str(path) for path in allowed], []
    for entry in paths:
        path = Path(entry or os.getcwd()).resolve()
        foreign = []
        if (path / "sao_bridge").is_dir() and path != allowed[0]:
            foreign.append(path)
        if (path / "areal/api/cli_args.py").is_file() and path != allowed[1]:
            foreign.append(path)
        if (path / "configs/phase2_hard_sao.yaml").is_file() and path != allowed[2]:
            foreign.append(path)
        if (path / "whale_local/sao_bridge").is_dir() and path != source_root:
            foreign.extend(path / name for name in (
                "whale_local", "third_party/sao/vendor/AReaL", "third_party/sao",
            ))
        if foreign:
            rejected.extend(path.resolve() for path in foreign if path.resolve() not in allowed[1:])
        elif str(path) not in result:
            result.append(str(path))
    return result, rejected


def check_import_origins(source_root: Path, rejected: list[Path]) -> None:
    source_root = source_root.resolve()
    for name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str):
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to(source_root) and any(path.is_relative_to(root) for root in rejected):
            raise RuntimeError(f"Mutable local source imported outside snapshot: {name}: {path}")


_REJECTED_IMPORT_ROOTS: list[Path] = []
if __name__ == "__main__":
    _EXECUTED_SOURCE_ROOT = Path(__file__).resolve().parents[2]
    sys.path[:], _REJECTED_IMPORT_ROOTS = isolated_import_paths(sys.path, _EXECUTED_SOURCE_ROOT)
    check_import_origins(_EXECUTED_SOURCE_ROOT, _REJECTED_IMPORT_ROOTS)

from .alfworld_core_extend import subset_report, validate_cohort
from .alfworld_h_train import training_spec as h_spec
from .alfworld_phase_data import editor_dataset
from .alfworld_reward_service import validate_metadata
from .alfworld_solver_validation import no_patch_assignments
from .alfworld_strong_phase import (
    checked_native_report, checked_release, checkpoint_pair, training_evidence,
)
from .alfworld_strong_select import generate_proposals, matrix_config, matrix_reports, run_matrix
from .alfworld_train_data import (
    _immutable_json, evaluation_paths, goal_group,
)
from .alfworld_w_train import assignment_schedule_audit, training_spec as w_spec
from .budget import contract_digest, model_digests
from .common import AREAL_REVISION, ROOT, SAO_REVISION, file_sha, read_json, write_json
from .large_train import configure, consumed_row_ledger, training_capacity_audit, validate_training_rows
from .launch import Launcher
from .resources import gpu_release_audit, require_research_node


INITIAL = ROOT / "runs/alfworld_joint_h0w0_8h_v1"
H_TEMPLATE = ROOT / "examples/alfworld_engineer_fast_v1_s42.json"
W_TEMPLATE = ROOT / "examples/alfworld_solver_fast_v4_s42.json"
EVAL_PROTOCOL = ROOT / "runs/alfworld_core_200_oneshot_v2/protocol.json"
DEFAULT_ROOT = ROOT / "runs/alfworld_sao_corrected_v3"
W_PREDECESSOR = ROOT / "runs/alfworld_w2_recovery_v1"
H_PREDECESSOR = ROOT / "runs/alfworld_whale_baseline_v4/execution"
NODES = {"w1": "dev-NDH200-025", "h1": "dev-NDH200-026", "h2": "dev-NDH200-026"}
COUNTS = {"h1": 192, "w1": 640, "h2": 192}
TIMEOUT = 36 * 3600


def verify_pins(pins: dict) -> None:
    for name, digest in pins.items():
        if file_sha(Path(name)) != digest:
            raise ValueError(f"Frozen SAO input changed: {name}")


def pins(paths) -> dict:
    return {str(path.resolve()): file_sha(path) for path in paths}


def declared_upstreams(source_root: Path) -> dict:
    result = {}
    for relative, revision in (
        ("third_party/sao", SAO_REVISION),
        ("third_party/sao/vendor/AReaL", AREAL_REVISION),
    ):
        root = (source_root / relative).resolve()
        actual_root, actual_revision = subprocess.check_output(
            ["git", "-c", f"safe.directory={root}", "-C", str(root),
             "rev-parse", "--show-toplevel", "HEAD"],
            text=True, timeout=30,
        ).strip().splitlines()
        if Path(actual_root).resolve() != root or actual_revision != revision:
            raise ValueError(f"Declared upstream repository/commit changed: {relative}")
        if result and not root.is_relative_to(Path(result["third_party/sao"]["root"])):
            raise ValueError("Declared AReaL dependency escaped its SAO vendor root")
        result[relative] = {"root": str(root), "revision": revision}
    return result


def source_provenance(paths: list[Path], source_root: Path = ROOT) -> dict:
    source_root = source_root.resolve()
    files, upstreams = {}, {}
    for path in paths:
        logical = Path(os.path.abspath(path))
        if not logical.is_relative_to(source_root):
            raise ValueError(f"Source pin escapes executed source snapshot: {path}")
        relative = logical.relative_to(source_root)
        resolved = path.resolve()
        if relative.is_relative_to("third_party/sao"):
            if not upstreams:
                upstreams = declared_upstreams(source_root)
            dependency = (
                "third_party/sao/vendor/AReaL"
                if relative.is_relative_to("third_party/sao/vendor/AReaL")
                else "third_party/sao"
            )
            expected = Path(upstreams[dependency]["root"]) / relative.relative_to(dependency)
            if resolved != expected:
                raise ValueError(f"Source pin escapes declared upstream layout: {path}")
        elif not resolved.is_relative_to(source_root):
            raise ValueError(f"Source pin escapes executed source snapshot: {path}")
        files[str(relative)] = file_sha(resolved)
    if not files:
        raise ValueError("Source provenance cannot be empty")
    return {"source_root": str(source_root), "source_files": files,
            "source_upstreams": upstreams,
            "source_sha256": contract_digest({"files": files, "upstreams": upstreams})}


def verify_source(protocol: dict) -> None:
    source_root = ROOT.resolve()
    if Path(protocol["source_root"]).resolve() != source_root:
        raise ValueError("Execute SAO from its frozen source snapshot, not mutable main")
    expected = source_provenance(
        [source_root / name for name in protocol["source_files"]], source_root,
    )
    if any(expected[key] != protocol[key] for key in expected):
        raise ValueError("Frozen source snapshot changed")
    from . import alfworld_h_train, alfworld_w_train, common, large_train
    for module in (alfworld_h_train, alfworld_w_train, common, large_train):
        if not Path(module.__file__).resolve().is_relative_to(source_root):
            raise ValueError("Imported training backend escaped the executed snapshot")
    check_import_origins(source_root, _REJECTED_IMPORT_ROOTS)


def receipt_provenance(root: Path, protocol: dict) -> dict:
    return {
        "protocol_sha256": file_sha(root / "protocol.json"),
        "source_root": protocol["source_root"], "source_sha256": protocol["source_sha256"],
    }


def checked_gpu_release(value: dict) -> None:
    if (
        value["foreign_gpu_pids"] or not value["filler_unit_active"]
        or not set(value["remaining_gpu_pids"]) <= set(value["verified_filler_pids"])
    ):
        raise RuntimeError("SAO requires released GPUs and active identity-checked filler")


def backend_wiring_audit(config: dict, *, role: str | None = None) -> dict:
    """Check the single-role path, not merely the joint trainer's flag."""
    if role is None:
        role = "solver" if config["phase"] == "w1" else "editor"
    if role not in {"solver", "editor"}:
        raise ValueError("Ordinary SAO backend role must be solver or editor")
    spec = w_spec(config, Path(config["run"])) if role == "solver" else h_spec(
        config, Path(config["run"]), "http://127.0.0.1:28600",
    )
    if not spec.get("independent_trajectories") or not spec.get("processed_behavior_logprobs"):
        raise ValueError("Both role trainers must forward independent_trajectories and processed_behavior_logprobs")
    if any(spec.get(key) for key in (
        "joint_root", "central_context", "value_normalization", "interaction_credit", "kl_ctl",
    )):
        raise ValueError("Ordinary SAO cannot enable MA, CTDE, ValueNorm or KL")
    native = configure(spec, Path(config["run"]))
    expected = config["batch_size"] // config["actor_gpus"]
    if (
        native["actor"]["kl_ctl"] != 0 or native["ref"] is not None
        or native["recover"]["mode"] != "disabled"
        or any(native[key]["mb_spec"].get("n_mbs") != expected for key in ("actor", "critic"))
    ):
        raise ValueError("Corrected SAO requires isolated sequences, fresh Adam and no reference KL")
    from . import large_train
    tree = ast.parse(Path(large_train.__file__).read_text())
    guards = [
        node for node in ast.walk(tree) if isinstance(node, ast.If)
        and "processed_behavior_logprobs" in ast.unparse(node.test)
        and "JointVLLMConfig" in "\n".join(ast.unparse(child) for child in node.body)
    ]
    if not guards:
        raise ValueError(
            "Backend blocker: large_train.main must use JointVLLMConfig when "
            "joint_root OR processed_behavior_logprobs is true"
        )
    return {"independent_trajectories": True, "microbatches_per_rank": expected,
            "processed_behavior_logprobs": True, "no_ma_no_kl": True}


def role_config(role: str, root: Path, state: dict) -> dict:
    if role not in COUNTS:
        raise ValueError("Unknown SAO role")
    template = read_json(W_TEMPLATE if role == "w1" else H_TEMPLATE)
    config = {
        **template, "run": str(root / role / "training"), "phase": role,
        "experiment": root.name, "steps": 80 if role == "w1" else 32,
        "batch_size": 8 if role == "w1" else 6,
        "data": str(root / role / "data.json"), "harness": str(root / "no_patch.json"),
        "seed": 1000042 if role == "h2" else 42,
        "learning_rate": 2e-6, "critic_learning_rate": 1e-5,
        "critic_updates_per_step": 1, "critic_warmup_steps": 5,
        "terminal_episodes": True, "critic_mode": "learned",
        "independent_trajectories": True, "processed_behavior_logprobs": True,
        "interaction_credit": 0.0, "central_context": None, "value_normalization": False,
        "staleness": 1, "save_interval_steps": 20 if role == "w1" else 16,
        "phase_timeout_seconds": 6 * 3600,
    }
    if role == "w1":
        for key in ("training_catalogue", "training_catalogue_sha256", "training_data"):
            config.pop(key, None)
        config.update(
            solver_model=state["solver"], solver_critic=state["solver_critic"],
            assignments=str(root / role / "assignments.json"),
            environment_concurrency=template["training_environment_concurrency"],
            context_tokens=32768, actor_gpus=4, rollout_gpus=4,
            training_split="test_set",
        )
    else:
        config.update(
            engineer_model=state["engineer"], engineer_critic=state["engineer_critic"],
            solver_model=state["solver"], solver_version=state["solver_version"],
            actor_gpus=3, rollout_gpus=1, temperature=0.7,
            tmpdir=str(root / role / "runtime_cache"),
        )
    return config


def freeze_role(root: Path, role: str, state: dict, rows: list[dict]) -> dict:
    if len(rows) != COUNTS[role]:
        raise ValueError("SAO role changed the original consumed-trajectory budget")
    config = role_config(role, root, state)
    _immutable_json(root / role / "data.json", rows)
    if role == "w1":
        _immutable_json(root / role / "assignments.json", matched_solver_assignments(
            root / "no_patch.json", rows,
        ))
        _immutable_json(root / role / "schedule_audit.json", assignment_schedule_audit(config))
    spec = w_spec(config, Path(config["run"])) if role == "w1" else h_spec(
        config, Path(config["run"]), "http://127.0.0.1:28600",
    )
    validate_training_rows(spec, rows)
    _immutable_json(root / role / "capacity_audit.json", training_capacity_audit(spec, rows))
    _immutable_json(root / role / "input_state.json", state)
    _immutable_json(root / role / "config.json", config)
    _immutable_json(root / role / "INPUTS.json", pins(
        [root / role / name for name in ("data.json", "input_state.json", "config.json")]
        + ([root / role / "assignments.json"] if role == "w1" else [])
    ))
    return config


def matched_solver_rows(joint_rows: list[dict], report_ids: list[int]) -> list[dict]:
    if len(joint_rows) < COUNTS["w1"]:
        raise ValueError("Current MA data needs at least 640 ordered rows for W1")
    rows = []
    heldout = set(report_ids)
    for index, source in enumerate(joint_rows[:COUNTS["w1"]]):
        task_id = source["task_id"]
        if (
            type(task_id) is not int or not 0 <= task_id < 240 or task_id in heldout
            or source.get("task_split") != "test_set"
            or source.get("task_catalogue") is not None
            or source.get("task_catalogue_sha256") is not None
            or not isinstance(source.get("task_context"), str) or not source["task_context"]
            or not isinstance(source.get("messages"), list) or not source["messages"]
        ):
            raise ValueError("Matched W1 rows must retain MA's official train0-239 population without catalogue overrides")
        if not isinstance(source.get("id"), str) or not source["id"]:
            raise ValueError("Current MA row lacks a provenance ID")
        rows.append({
            "id": f"sao-w1-{index:05d}-{source['id']}",
            "task_id": task_id, "task_split": "test_set",
            "task_name": f"alfworld-sao-w1-{index % 32:02d}",
            "partner_version": 0, "recycled_ids": [], "warmup": False,
            "source_joint_id": source["id"], "source_joint_index": index,
            "task_context": source["task_context"],
            "source_engineer_prompt_sha256": contract_digest(source["messages"]),
        })
    return rows


def matched_solver_assignments(harness: Path, rows: list[dict]) -> list[dict]:
    if any(
        row["task_split"] != "test_set" or not 0 <= row["task_id"] < 240
        or row.get("task_catalogue") is not None or row.get("task_catalogue_sha256") is not None
        for row in rows
    ):
        raise ValueError("Matched Solver assignments cannot change the official training population")
    assignments = no_patch_assignments(harness, training_rows=[
        {**row, "task_catalogue": None, "task_catalogue_sha256": None} for row in rows
    ])
    for assignment in assignments:
        assignment.pop("catalogue")
        assignment.pop("catalogue_sha256")
        assignment["task_ids"] = list(dict.fromkeys(assignment["task_ids"]))
    return assignments


def checked_packets(state: dict) -> list[dict]:
    path = Path(state["packets"])
    if file_sha(path) != state["packets_sha256"]:
        raise ValueError("Engineer packets changed")
    packets = read_json(path)
    if not packets:
        raise ValueError("No authentic failures available; do not substitute old W0 evidence")
    seen = set()
    for packet in packets:
        metadata = packet["metadata"]
        validate_metadata(metadata, state["solver_version"])
        ids = set(metadata["task_ids"])
        if (
            not ids or min(ids) < 0 or max(ids) >= 240 or seen & ids
            or metadata["split"] != "test_set"
        ):
            raise ValueError("Engineer packets are not unique legacy train0-239 cohorts")
        if state.get("solver_weights_sha256") and metadata.get("solver_weights_sha256") != state["solver_weights_sha256"]:
            raise ValueError("Refreshed packets do not bind current W1 weights")
        seen.update(ids)
    return packets


def prepare(root: Path, w_predecessor: Path, h_predecessor: Path, *,
            inputs_root: Path | None = None) -> dict:
    root = root.resolve()
    inputs_root = (inputs_root or ROOT).resolve()
    initial = inputs_root / "runs/alfworld_joint_h0w0_8h_v1"
    eval_protocol = inputs_root / "runs/alfworld_core_200_oneshot_v2/protocol.json"
    if (root / "protocol.json").exists():
        protocol = read_json(root / "protocol.json")
        requested = {"w1": str(w_predecessor.resolve()), "h1": str(h_predecessor.resolve())}
        if protocol["prerequisites"] != requested:
            raise ValueError("Frozen SAO predecessors differ; prepare a fresh root instead of replacing the queue")
        if protocol["inputs_root"] != str(inputs_root):
            raise ValueError("Frozen SAO input repository changed")
        verify_source(protocol)
        verify_pins(protocol["pinned_files"])
        return protocol
    state = read_json(initial / "INITIAL_STATE.json")
    known = read_json(initial / "INITIAL_MODEL_IDENTITIES.json")
    expected = {
        "engineer": known["models"]["editor"]["path"],
        "solver": known["models"]["solver"]["path"],
    }
    if any(state[key] != value or state[f"{key}_critic"] != value for key, value in expected.items()):
        raise ValueError("SAO must start from official original H0/W0, not historical trained weights")
    identities = {}
    for label, key in (("editor", "engineer"), ("solver", "solver")):
        model = Path(state[key])
        actual = {path.name: file_sha(path) for path in sorted(model.glob("*.safetensors"))}
        if (
            actual != known["models"][label]["weights"]
            or file_sha(model / "config.json") != known["models"][label]["config_sha256"]
        ):
            raise ValueError("Original H0/W0 weights differ from the initial comparison")
        identities[key] = {
            "model": str(model), "weights": actual,
            "weights_sha256": contract_digest(actual),
            "config_sha256": file_sha(model / "config.json"),
        }
    _immutable_json(root / "INITIAL_MODELS.json", identities)
    _immutable_json(root / "no_patch.json", read_json(Path(state["harness"])))
    h_template = read_json(H_TEMPLATE)
    manifest_path = Path(state["packets"]).parent / "manifest.json"
    manifest, cohort = read_json(manifest_path), read_json(eval_protocol)
    validate_cohort(cohort)
    if len(manifest["selection_task_ids"]) != 24 or set(manifest["selection_task_ids"]) & set(range(240)):
        raise ValueError("Selection must contain exactly the frozen train-disjoint 24 tasks")
    packets = checked_packets(state)
    h_rows = read_json(Path(h_template["data"]))
    if h_rows != editor_dataset(packets, 192, 0):
        raise ValueError("H1 data differs from the original 192 training prompts")
    joint_data = initial / "data.json"
    ma_protocol_path = initial / "protocol.json"
    ma_protocol = read_json(ma_protocol_path)
    if ma_protocol["data_sha256"] != file_sha(joint_data):
        raise ValueError("Current MA task/prompt order differs from its frozen joint data")
    joint_rows = read_json(joint_data)
    w_rows = matched_solver_rows(joint_rows, cohort["all_task_ids"])
    data_root = inputs_root / "assets/alfworld_harness_r1/data"
    eval_paths = evaluation_paths(data_root)
    heldout = {goal_group(eval_paths[i]) for i in cohort["strict_goal_disjoint_task_ids"] + manifest["selection_task_ids"]}
    if any(
        row["task_context"] != eval_paths[row["task_id"]]
        or goal_group(eval_paths[row["task_id"]]) in heldout for row in w_rows
    ):
        raise ValueError("W1 training overlaps selection/strict197 goals")
    freeze_role(root, "h1", state, h_rows)
    freeze_role(root, "w1", state, w_rows)
    _immutable_json(root / "w1/POPULATION_AUDIT.json", {
        "label": "matched-training-population repaired SAO",
        "source_joint_data": str(joint_data), "source_joint_data_sha256": file_sha(joint_data),
        "source_indices": list(range(640)), "task_ids": [row["task_id"] for row in w_rows],
        "unique_tasks": sorted({row["task_id"] for row in w_rows}),
        "task_context_sequence_sha256": contract_digest([row["task_context"] for row in w_rows]),
        "ma_engineer_prompt_sequence_sha256": contract_digest([row["messages"] for row in joint_rows[:640]]),
        "ma_solver_prefix_rows": 512, "sao_solver_prefix_rows": 640,
        "shared_first512_task_order": True, "all200_task_overlap": [],
        "prompt_interpretation": "Official Solver task contexts match MA's ordered prefix. MA Engineer messages are provenance-only, never injected into the ordinary Solver.",
        "historical_curriculum_reproduced": False,
    })
    sources = [ROOT / "whale_local/sao_bridge" / f"{name}.py" for name in (
        "alfworld_sao_control", "alfworld_h_train", "alfworld_w_train",
        "large_train", "train", "terminal_actor", "large_workflow", "alfworld_domain",
        "alfworld_phase_data", "alfworld_strong_phase", "alfworld_strong_refresh",
        "alfworld_strong_select", "alfworld_solver_eval", "alfworld_engineer_eval",
        "alfworld_reward_service", "alfworld_train_data", "joint_workflow",
        "launch", "resources", "common",
    )]
    vendor = ROOT / "third_party/sao/vendor/AReaL/areal"
    sources += [vendor / name for name in (
        "engine/fsdp_engine.py", "engine/vllm_remote.py", "api/cli_args.py",
        "utils/data.py", "infra/launcher/vllm_server.py",
        "trainer/ppo/actor.py", "trainer/ppo/critic.py",
    )]
    sources += [H_TEMPLATE, W_TEMPLATE, ROOT / "third_party/sao/configs/phase2_hard_sao.yaml"]
    provenance = source_provenance(sources, ROOT)
    inputs = [
        initial / "INITIAL_STATE.json",
        initial / "INITIAL_MODEL_IDENTITIES.json", manifest_path, eval_protocol,
        Path(state["packets"]), Path(h_template["data"]), joint_data, ma_protocol_path,
        root / "w1/POPULATION_AUDIT.json", root / "INITIAL_MODELS.json", root / "no_patch.json",
        root / "h1/INPUTS.json", root / "w1/INPUTS.json",
    ]
    protocol = {
        "protocol": "matched-population-repaired-sao-h0w0-v1", "run": str(root),
        "inputs_root": str(inputs_root), **provenance,
        "classification": "matched-training-population repaired SAO: method-level baseline, not an exact historical curriculum reproduction or scheduling-only causal ablation",
        "initialization": "H0 is the official already-RL-trained Harness-R1 checkpoint, not a bare base model; W0 is official target-agent-sft.",
        "initial_state": state, "no_patch": str(root / "no_patch.json"),
        "budget": {"consumed": COUNTS, "total": 1024, "includes_critic_warmup": True,
                   "role_allocation": {"sao": {"engineer": 384, "solver": 640},
                                       "current_ma": {"engineer": 512, "solver": 512}},
                   "not_matched": ["environment calls", "generated/prefetched trajectories", "tokens", "FLOPs"]},
        "training_data": "H1 original192 prompts from 24 W0 packets; W1 takes current MA joint-data first640 official test_set train0-239 rows in identical task/context order, not the historical distinct2296-task pool. MA matched W uses first512. H2 fresh W1 train0-239 evidence.",
        "schedule": "Independent H1 and empty-harness W1 may run in parallel; H2 starts only after both completed and released.",
        "checkpoint_policy": {"h1": 31, "w1": 79, "h2": 31},
        "historical_deviation": "W1 curriculum changed to current MA's ordered train0-239 population rather than historical distinct2296-task pool. Predeclared final H1 step31 replaces historical selected step15; all192 H1 trajectories charged. H2 also final31. Role allocations remain SAO H384/W640 versus MA H512/W512; not a scheduling-only comparison.",
        "curves_saved_steps": {"h1": [15, 31], "w1": [19, 39, 59, 79], "h2": [15, 31]},
        "harness_selection": "Best valid fresh proposal among eight per H1/H2 final model on selection24 mean@2; deterministic tie by slot. Never old harness or incumbent substitution; no positive-gain filter.",
        "optimizer": "Fresh Adam per stage; H2 carries H1 actor and same-role critic weights only.",
        "no_ma": ["no CTDE", "no ValueNorm", "no peer weights", "no replay", "no KL"],
        "h2": {"template": str(H_TEMPLATE), "proposal_seed": 5000045},
        "evaluation": {
            "task_manifest": str(manifest_path), "samples": 2, "seed": 6000045,
            "temperature": 1.0, "solver_tokens": 4096, "eval_concurrency": 64,
            "environment_concurrency": 16, "matrix_concurrency": 4,
        },
        "report_protocol": str(eval_protocol),
        "report_policy": "One fresh all200 mean@1 matrix W1+empty/H1/H2 after harness freeze; strict197 is derived, not rerun. Previously exposed selection tasks remain disclosed.",
        "nodes": NODES, "queue_timeout_seconds": TIMEOUT,
        "prerequisites": {"w1": str(w_predecessor.resolve()), "h1": str(h_predecessor.resolve())},
        "source_provenance": "Hashes of current corrected backend source; CPU checks are not GPU validation.",
        "pinned_files": pins(inputs + sources),
    }
    _immutable_json(root / "protocol.json", protocol)
    blockers = []
    for role in ("h1", "w1"):
        try:
            backend_wiring_audit(read_json(root / role / "config.json"))
        except ValueError as exc:
            blockers.append(f"{role}: {exc}")
    _immutable_json(root / "PREPARED.json", {
        "status": "cpu_prepared", **receipt_provenance(root, protocol),
        "gpu_started": False, "gpu_validated": False, "backend_blockers": blockers,
        "training_allowed": not blockers,
    })
    return protocol


def validate_only(root: Path) -> dict:
    protocol = read_json(root / "protocol.json")
    verify_source(protocol)
    verify_pins(protocol["pinned_files"])
    prepared = read_json(root / "PREPARED.json")
    if prepared["protocol_sha256"] != file_sha(root / "protocol.json"):
        raise ValueError("CPU preparation belongs to another protocol")
    audits = {}
    for role in ("h1", "w1", "h2"):
        if not (root / role / "config.json").exists():
            continue
        verify_pins(read_json(root / role / "INPUTS.json"))
        audits[role] = backend_wiring_audit(read_json(root / role / "config.json"))
    return {"status": "cpu_validated", "gpu_started": False, "gpu_validated": False,
            "roles": audits, "prerequisites": protocol["prerequisites"],
            **receipt_provenance(root, protocol)}


def wait_prerequisite(target: Path, queue: Path, timeout: float = TIMEOUT) -> None:
    if not 0 < timeout <= TIMEOUT:
        raise ValueError("SAO queue timeout must be in (0, 36h]")
    queue.mkdir(parents=True, exist_ok=True)
    marker = queue / "WAITING.json"
    if not marker.exists():
        _immutable_json(marker, {"target": str(target.resolve()), "deadline_unix": time.time() + timeout})
    record = read_json(marker)
    if record["target"] != str(target.resolve()):
        raise ValueError("Persistent queue predecessor changed")
    while True:
        if (target / "FAILED.json").exists():
            raise RuntimeError(f"Predecessor failed: {target}")
        if (target / "COMPLETE.json").exists() and (target / "RELEASED.json").exists():
            if read_json(target / "COMPLETE.json")["status"] not in ("complete", "completed"):
                raise ValueError("Predecessor did not actually complete")
            checked_release(target)
            _immutable_json(queue / "SATISFIED.json", pins([
                target / "COMPLETE.json", target / "RELEASED.json",
            ]))
            return
        if time.time() >= record["deadline_unix"]:
            raise TimeoutError(f"Persistent 36h queue expired: {target}")
        time.sleep(15)


def completed_role(root: Path, role: str) -> dict:
    stage = root / role
    if (stage / "FAILED.json").exists():
        raise RuntimeError(f"Failed SAO stage cannot be substituted or resumed: {role}")
    complete = read_json(stage / "COMPLETE.json")
    if complete["status"] != "completed" or complete["consumed_trajectories"] != COUNTS[role]:
        raise ValueError("SAO role did not finish its complete budget")
    checked_release(stage)
    verify_pins(read_json(stage / "INPUTS.json"))
    verify_pins(complete["artifacts"])
    return complete


def execute_training(root: Path, role: str, launch: Launcher) -> dict:
    stage = root / role
    config = read_json(stage / "config.json")
    verify_pins(read_json(stage / "INPUTS.json"))
    _immutable_json(stage / "backend_audit.json", backend_wiring_audit(config))
    output = Path(config["run"])
    if output.exists():
        raise RuntimeError("Partial optimizer state is not an exact resume; use a newly declared run")
    process = launch.module(
        f"sao_bridge.alfworld_{'w' if role == 'w1' else 'h'}_train",
        ["--config", stage / "config.json"], gpus="0,1,2,3,4,5,6,7", name=f"{role}-train",
    )
    launch.wait(process, f"ordinary SAO {role}", timeout=config["phase_timeout_seconds"])
    trained, audit = training_evidence(output, config)
    expected = consumed_row_ledger(
        w_spec(config, output) if role == "w1" else h_spec(config, output, "unused"),
        read_json(stage / "data.json"),
    )
    if read_json(output / "optimizer/consumed_rows.json") != expected:
        raise ValueError("Native consumed-row ledger differs from the frozen budget")
    probability = read_json(output / "optimizer/rollout_probability_contract.json")
    if probability["logprobs_mode"] != "processed_logprobs" or probability["temperature"] != config["temperature"]:
        raise ValueError("GPU training did not use the corrected behavior probability contract")
    pair = checkpoint_pair(trained, 79 if role == "w1" else 31)
    if any(not Path(path).resolve().is_relative_to(output.resolve()) for path in pair.values()):
        raise ValueError("Selected checkpoint escaped this fresh training run")
    history = {str(step): checkpoint_pair(trained, step) for step in (
        (19, 39, 59, 79) if role == "w1" else (15, 31)
    )}
    _immutable_json(stage / "native_audit.json", audit)
    return {
        "status": "completed", "role": role, "pair": pair, "checkpoint_history": history,
        "consumed_trajectories": COUNTS[role], "optimizer_state_resumed": False,
        "actor_weights_sha256": model_digests(Path(pair["actor"]))[1],
        "critic_weights_sha256": model_digests(Path(pair["critic"]))[1],
        "artifacts": pins([
            stage / "config.json", stage / "data.json", stage / "native_audit.json",
            output / "COMPLETE.json", output / "optimizer/trained.json",
            output / "optimizer/metrics.jsonl", output / "optimizer/consumed_rows.json",
            output / "optimizer/rollout_probability_contract.json",
        ]),
    }


def prepare_h2(root: Path, protocol: dict, launch: Launcher) -> dict:
    h1, w1 = completed_role(root, "h1"), completed_role(root, "w1")
    state = dict(
        protocol["initial_state"], engineer=h1["pair"]["actor"], engineer_critic=h1["pair"]["critic"],
        solver=w1["pair"]["actor"], solver_critic=w1["pair"]["critic"],
        solver_version=f"sao-corrected-w1-{w1['actor_weights_sha256'][:12]}",
        solver_weights_sha256=w1["actor_weights_sha256"], solver_epoch_version=1,
        protocol_sha256=file_sha(root / "protocol.json"),
    )
    for result in (h1, w1):
        for label in ("actor", "critic"):
            if model_digests(Path(result["pair"][label]))[1] != result[f"{label}_weights_sha256"]:
                raise ValueError("Fresh H1/W1 checkpoint weights changed before H2")
    _immutable_json(root / "h2/refresh_input.json", state)
    refresh_root = root / "h2/current_w1_refresh"
    if refresh_root.exists():
        raise RuntimeError("Partial W1 evidence refresh cannot be silently reused")
    process = launch.module(
        "sao_bridge.alfworld_strong_refresh",
        ["--protocol", root / "protocol.json", "--state", root / "h2/refresh_input.json",
         "--root", refresh_root], gpus="0,1,2,3,4,5,6,7", name="fresh-w1-training-evidence",
    )
    launch.wait(process, "fresh W1 train0-239 evidence", timeout=6 * 3600)
    checked_release(refresh_root)
    receipt = read_json(refresh_root / "COMPLETE.json")
    if (
        receipt["status"] != "completed" or not receipt["engineer_training_ready"]
        or receipt["baseline_tasks"] != 240 or receipt["report_opened"]
        or receipt["solver_weights_sha256"] != w1["actor_weights_sha256"]
        or receipt["state_sha256"] != file_sha(refresh_root / "REFRESHED_STATE.json")
    ):
        raise ValueError("Fresh W1 evidence is incomplete; no old-packet fallback permitted")
    refreshed = read_json(refresh_root / "REFRESHED_STATE.json")
    if refreshed["solver"] != state["solver"] or refreshed["packet_solver"] != state["solver"]:
        raise ValueError("H2 evidence belongs to another Solver")
    packets = checked_packets(refreshed)
    freeze_role(root, "h2", refreshed, editor_dataset(packets, 192, 1, sample_offset=192))
    _immutable_json(root / "h2/refresh_binding.json", pins([
        refresh_root / "COMPLETE.json", refresh_root / "REFRESHED_STATE.json",
        refresh_root / "packets.json", refresh_root / "baseline_metadata.json",
    ]))
    return refreshed


def choose_valid_proposal(proposals: list[dict], reports: dict, label: str) -> str:
    if [row["sample_index"] for row in proposals] != list(range(8)):
        raise ValueError("Fresh harness selection lost its eight proposal slots")
    choices = [f"{label}-proposal{row['sample_index']:02d}" for row in proposals if row["valid"]]
    if not choices:
        raise ValueError(f"{label} generated no valid harness; do not substitute an older policy")
    return min(choices, key=lambda name: (-reports[name]["accuracy"], name))


def select_harnesses(root: Path, protocol: dict, state: dict, h2: dict, launch: Launcher) -> dict:
    pairs = {"h1": completed_role(root, "h1")["pair"], "h2": h2["pair"]}
    selection = root / "h2/harness_selection"
    proposals = generate_proposals(launch, protocol, state, pairs, selection)
    harnesses = {}
    for label, values in proposals.items():
        for value in values:
            name = f"{label}-proposal{value['sample_index']:02d}"
            path = selection / "patches" / f"{name}.json"
            _immutable_json(path, value["patch"] if value["valid"] else read_json(root / "no_patch.json"))
            harnesses[name] = str(path)
    config = matrix_config(protocol, state["solver"], state["solver_version"], harnesses, selection / "matrix")
    matrix, _ = run_matrix(launch, config, selection / "matrix", 1)
    reports = matrix_reports(matrix, protocol)
    chosen = {}
    for label in ("h1", "h2"):
        name = choose_valid_proposal(proposals[label], reports, label)
        path = selection / f"selected_{label}.json"
        _immutable_json(path, read_json(Path(harnesses[name])))
        chosen[label] = {"harness": str(path), "sha256": file_sha(path),
                         "proposal": name, "selection_accuracy": reports[name]["accuracy"],
                         "engineer": pairs[label]["actor"]}
    _immutable_json(selection / "FROZEN.json", {
        "status": "frozen", "harnesses": chosen, "report_opened": False,
        "solver": state["solver"], "selection_task_ids": config["task_ids"],
        "proposal_artifacts": pins([selection / "proposals" / f"{label}.json" for label in pairs]),
    })
    return chosen


def train_role(root: Path, role: str) -> dict:
    require_research_node()
    protocol = read_json(root / "protocol.json")
    if os.uname().nodename != protocol["nodes"][role]:
        raise RuntimeError(f"{role} is assigned to {protocol['nodes'][role]}")
    stage = root / role
    stage.mkdir(parents=True, exist_ok=True)
    with (stage / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_source(protocol)
        verify_pins(protocol["pinned_files"])
        if (stage / "COMPLETE.json").exists():
            return completed_role(root, role)
        if (stage / "STARTED.json").exists() or (stage / "FAILED.json").exists():
            raise RuntimeError("Partial SAO stage cannot be implicitly resumed")
        launch = None
        try:
            deadline_path = stage / "queue/DEADLINE.json"
            if not deadline_path.exists():
                _immutable_json(deadline_path, {
                    "deadline_unix": time.time() + protocol["queue_timeout_seconds"],
                })
            deadline = read_json(deadline_path)["deadline_unix"]
            dependencies = ("h1", "w1") if role == "h2" else (role,)
            for dependency in dependencies:
                target = root / dependency if role == "h2" else Path(protocol["prerequisites"][dependency])
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError("Persistent SAO stage queue exceeded its total 36h limit")
                wait_prerequisite(target, stage / "queue" / dependency, min(TIMEOUT, remaining))
            verify_pins(protocol["pinned_files"])
            verify_source(protocol)
            backend_wiring_audit(role_config(role, root, protocol["initial_state"]))
            checked_gpu_release(gpu_release_audit())
            _immutable_json(stage / "STARTED.json", {
                "started_unix": time.time(), "node": os.uname().nodename,
                "protocol_sha256": file_sha(root / "protocol.json"),
            })
            launch = Launcher(stage / "supervision")
            if role == "h2":
                state = prepare_h2(root, protocol, launch)
            result = execute_training(root, role, launch)
            result.update(receipt_provenance(root, protocol))
            if role == "h2":
                result["harnesses"] = select_harnesses(root, protocol, state, result, launch)
                result["artifacts"].update(pins([
                    stage / "refresh_binding.json", stage / "harness_selection/FROZEN.json",
                    *[Path(item["harness"]) for item in result["harnesses"].values()],
                ]))
            launch.close()
            launch = None
            release = gpu_release_audit()
            checked_gpu_release(release)
            _immutable_json(stage / "RELEASED.json", {**release, **receipt_provenance(root, protocol)})
            _immutable_json(stage / "COMPLETE.json", result)
            return result
        except BaseException as exc:
            write_json(stage / "FAILED.json", {"status": "failed", "error": repr(exc), "time": time.time()})
            raise
        finally:
            if launch is not None:
                launch.close()
                write_json(stage / "RELEASED.json", {
                    **gpu_release_audit(), **receipt_provenance(root, protocol),
                })


def finalize(root: Path) -> dict:
    require_research_node()
    protocol = read_json(root / "protocol.json")
    if os.uname().nodename != protocol["nodes"]["h2"]:
        raise RuntimeError("Final SAO reporting belongs on node4; node3 remains free")
    with (root / "finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_source(protocol)
        verify_pins(protocol["pinned_files"])
        if (root / "COMPLETE.json").exists():
            checked_release(root)
            result = read_json(root / "COMPLETE.json")
            verify_pins(result["artifacts"])
            return result
        if (root / "REPORT_STARTED.json").exists() or (root / "FAILED.json").exists():
            raise RuntimeError("Final all200 report is once-only; no retry or substitute result")
        results = {role: completed_role(root, role) for role in COUNTS}
        verify_pins(read_json(root / "h2/refresh_binding.json"))
        chosen = results["h2"]["harnesses"]
        for item in chosen.values():
            if file_sha(Path(item["harness"])) != item["sha256"]:
                raise ValueError("Selected fresh harness changed before final report")
        w1 = results["w1"]
        if model_digests(Path(w1["pair"]["actor"]))[1] != w1["actor_weights_sha256"]:
            raise ValueError("W1 weights changed before final report")
        cohort = read_json(Path(protocol["report_protocol"]))
        validate_cohort(cohort)
        checked_gpu_release(gpu_release_audit())
        _immutable_json(root / "REPORT_STARTED.json", {
            "protocol_sha256": file_sha(root / "protocol.json"),
            "harnesses": chosen, "started_unix": time.time(),
        })
        launch = Launcher(root / "report_supervision")
        try:
            config = {
                "solver_model": w1["pair"]["actor"], "solver_version": "sao-corrected-w1-final79",
                "no_patch": protocol["no_patch"],
                "harnesses": {label: item["harness"] for label, item in chosen.items()},
                "task_ids": cohort["all_task_ids"], "samples": 1,
                **{key: cohort[key] for key in ("seed", "temperature", "solver_tokens")},
                "eval_concurrency": 128, "environment_concurrency": 32, "matrix_concurrency": 3,
            }
            matrix, _ = run_matrix(launch, config, root / "final_matrix", 1)
            scores, artifacts = {}, []
            for label, item in matrix["variants"].items():
                report = checked_native_report(read_json(Path(item["report"])), config["task_ids"], config)
                output = root / "reports" / f"w1-{label}.json"
                _immutable_json(output, report)
                artifacts.append(output)
                scores[label] = {
                    "all200": report["accuracy"],
                    "strict197": subset_report(report, cohort["strict_goal_disjoint_task_ids"])["accuracy"],
                    "report": str(output),
                }
            if set(scores) != {"no_patch", "h1", "h2"}:
                raise ValueError("Final comparator omitted one of W1 empty/H1/H2")
            launch.close()
            release = gpu_release_audit()
            checked_gpu_release(release)
            complete = {
                "status": "completed", "classification": protocol["classification"],
                **receipt_provenance(root, protocol),
                "consumed_trajectories": sum(row["consumed_trajectories"] for row in results.values()),
                "budget": protocol["budget"], "training_data": protocol["training_data"],
                "scores": scores, "historical_deviation": protocol["historical_deviation"],
                "gpu_validated": True, "artifacts": pins(artifacts),
                "interpretation": "Single-seed method-level SAO endpoint. No claim of MA superiority or scheduling causality; compare fresh paired MA reports separately.",
            }
            _immutable_json(root / "RELEASED.json", {**release, **receipt_provenance(root, protocol)})
            _immutable_json(root / "COMPLETE.json", complete)
            return complete
        except BaseException as exc:
            write_json(root / "FAILED.json", {"status": "failed", "error": repr(exc)})
            raise
        finally:
            launch.close()
            if not (root / "RELEASED.json").exists():
                write_json(root / "RELEASED.json", {
                    **gpu_release_audit(), **receipt_provenance(root, protocol),
                })


def run_h_pipeline(root: Path) -> dict:
    train_role(root, "h1")
    train_role(root, "h2")
    return finalize(root)


def launch_snapshot_module(self, module: str, args: list, *, gpus: str, name: str,
                           env_overrides: dict[str, str] | None = None):
    if not (module.startswith("sao_bridge.") or module == "vllm.entrypoints.openai.api_server"):
        raise ValueError("SAO snapshot wrapper only launches bridge modules and its vLLM server")
    return self.start(
        [sys.executable, "-B", "-m", "sao_bridge.alfworld_sao_control",
         "_snapshot-module", module, *map(str, args)],
        gpus=gpus, name=name, env_overrides=env_overrides,
    )


def main() -> None:
    check_import_origins(ROOT, _REJECTED_IMPORT_ROOTS)
    Launcher.module = launch_snapshot_module
    if len(sys.argv) >= 3 and sys.argv[1] == "_snapshot-module":
        module = sys.argv[2]
        if (
            not (module.startswith("sao_bridge.") or module == "vllm.entrypoints.openai.api_server")
            or module == "sao_bridge.alfworld_sao_control"
        ):
            raise ValueError("Invalid owned snapshot module")
        sys.argv = [module, *sys.argv[3:]]
        sys.modules.pop(module, None)
        runpy.run_module(module, run_name="__main__")
        check_import_origins(ROOT, _REJECTED_IMPORT_ROOTS)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "validate", "train-role", "run-h-pipeline", "finalize"))
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--inputs-root", type=Path, default=ROOT)
    parser.add_argument("--validate-only", action="store_true", help="CPU validation; never wait, audit GPUs or launch workers")
    parser.add_argument("--role", choices=tuple(COUNTS))
    parser.add_argument("--w-predecessor", type=Path, default=W_PREDECESSOR)
    parser.add_argument("--h-predecessor", type=Path, default=H_PREDECESSOR)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.validate_only or args.action == "validate":
        print(validate_only(root), flush=True)
    elif args.action == "prepare":
        prepare(root, args.w_predecessor, args.h_predecessor, inputs_root=args.inputs_root)
        print(read_json(root / "PREPARED.json"), flush=True)
    elif args.action == "train-role":
        if args.role is None:
            parser.error("train-role requires --role")
        print(train_role(root, args.role), flush=True)
    elif args.action == "run-h-pipeline":
        print(run_h_pipeline(root), flush=True)
    else:
        print(finalize(root), flush=True)


if __name__ == "__main__":
    main()
