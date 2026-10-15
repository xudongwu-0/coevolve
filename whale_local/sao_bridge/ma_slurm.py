"""Independent two-node MA144 execution; never submits or restarts Slurm jobs."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import socket
import time

from . import concurrent_comparison as comparison
from . import slurm_runtime


ROOT = comparison.REMOTE_ROOT
REPO = ROOT / "coevolve"
PAYLOAD = REPO / "reproducibility/ma_sao_20261012/payload"
STUDY = PAYLOAD / "runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2"
OVERLAY = ROOT / ".overlays/coevolve-training"
KERNEL_VERSIONS = {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "triton": "3.7.1"}


def preserved(old: str) -> Path:
    relative = Path(old).relative_to("/mnt/3fs/xudong/coevolve")
    candidates = [PAYLOAD / relative, REPO / relative]
    if relative.parts[:2] == ("runs", "source-snapshots"):
        reduced = Path(*relative.parts[3:])
        candidates.extend((PAYLOAD / reduced, REPO / reduced))
    for path in candidates:
        if path.is_file():
            return comparison.contained(path)
    raise FileNotFoundError(f"Preserved evaluation input is unavailable: {old}")


def inputs() -> tuple[dict, dict, list[dict], dict[str, str]]:
    original = comparison.load(STUDY / "PROTOCOL.json")
    core_path = preserved(original["core200_protocol"])
    selection_path = preserved(original["evaluation"]["task_manifest"])
    template_path = preserved(original["h2"]["template"])
    no_patch = REPO / "examples/alfworld_no_patch.json"
    packets = REPO / "assets/alfworld_engineer_fast_v1/packets.json"
    core, selection = comparison.load(core_path), comparison.load(selection_path)
    if (
        len(core["all_task_ids"]) != 200 or len(set(core["all_task_ids"])) != 200
        or len(selection["selection_task_ids"]) != 48
        or set(selection["selection_task_ids"]) - set(core["all_task_ids"])
        or set(range(240)) & set(core["all_task_ids"])
        or core["solver_tokens"] != 4096
        or comparison.load(template_path)["engineer_tokens"] != 4096
    ):
        raise ValueError("Endpoint cohorts or token budgets changed")
    protocol = {
        "core200_protocol": str(core_path), "no_patch": str(no_patch),
        "h2": {**original["h2"], "template": str(template_path)},
        "evaluation": {
            **original["evaluation"], "task_manifest": str(selection_path),
            "report_after_both_arms_freeze": False,
            "report_after_own_arm_completion": True,
        },
        "incumbent_min_gain": 2 / 48,
        "schedule": "every-round-joint", "primary_endpoint": 144,
    }
    if (protocol["evaluation"]["samples"] != 1
            or any(protocol["evaluation"][key] != core[key] for key in ("seed", "temperature", "solver_tokens"))):
        raise ValueError("Selection and endpoint decoding contracts differ")
    state = {
        "engineer": str(REPO / "assets/Harness-R1/harness-r1"),
        "solver": str(REPO / "assets/Harness-R1/target-agent-sft"),
        "packets": str(packets), "packets_sha256": comparison.sha(packets),
        "solver_epoch_version": 0, "harness": str(no_patch),
    }
    source_data = STUDY / "joint_data.json"
    rows = comparison.paired_rows(comparison.load(source_data))
    pinned = {str(path): comparison.sha(path) for path in (
        STUDY / "PROTOCOL.json", core_path, selection_path, template_path, packets, no_patch, source_data,
    )}
    for key in ("engineer", "solver"):
        model = comparison.contained(Path(state[key]))
        if not (model / "model.safetensors").is_file():
            raise FileNotFoundError("Verified published starting Actor is missing")
        for name in ("config.json", "preprocessor_config.json", "processor_config.json"):
            pinned[str(model / name)] = comparison.sha(model / name)
    return protocol, state, rows, pinned


def single_arm(plan: dict) -> str:
    arms = plan["config"].get("active_arms")
    if not isinstance(arms, list) or len(arms) != 1 or arms[0] not in ("ma_sao", *comparison.ABLATION_MECHANISMS):
        raise ValueError("Expected one full-MA or explicitly named add-back arm")
    return arms[0]


def bind(run: Path, nodes: list[str], deadline: float, *, arm: str = "ma_sao") -> dict:
    if arm not in ("ma_sao", *comparison.ABLATION_MECHANISMS):
        raise ValueError("Unknown standalone MA/add-back arm")
    run = comparison.contained(run)
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or len(nodes) != 2 or len(set(nodes)) != 2 or deadline <= time.time():
        raise ValueError("Binding requires the active two-node job and a future deadline")
    if run != ROOT / "runs/xd-ma144" / job or run.exists():
        raise ValueError("Use a new job-owned MA run directory; no automatic restart")
    inputs()
    config = {
        "run": str(run / "bootstrap"), "study_run": str(run),
        "runtime_cache_root": str(run / "cache"), "active_arms": [arm],
        "nodes": {arm: dict(zip(comparison.ROLES, nodes))},
        "worker_python": str(ROOT / ".envs/alfworld/bin/python"),
        "training_deadline_unix": deadline, "phase_timeout_seconds": 4 * 3600,
    }
    plan = {"schema": comparison.SCHEMA, "kind": "ma-kernel-preflight",
            "config": config, "inputs": {}, "specs": {}, "job_id": job}
    comparison.immutable(run / "bootstrap/COMPARISON.json", plan)
    enter(run / "bootstrap")
    return plan


def enter(root: Path, role: str | None = None) -> dict:
    os.environ.update(slurm_runtime.execution_environment(root))
    allocation = slurm_runtime.require_node()
    plan = comparison.load(root / "COMPARISON.json")
    if role is not None and allocation["host"] != plan["config"]["nodes"][single_arm(plan)][role]:
        raise ValueError("The requested MA role is on the wrong allocated host")
    return plan


def kernel(run: Path, role: str) -> dict:
    enter(run / "bootstrap", role)
    from .training_acceleration import kernel_check
    output = comparison.contained(run / "preflight" / role, run)
    output.mkdir(parents=True, exist_ok=True)
    result = kernel_check(output)
    if any(result["versions"].get(name) != version for name, version in KERNEL_VERSIONS.items()):
        raise ValueError("Kernel check used the wrong training overlay")
    return result


def collective(run: Path, role: str) -> dict:
    enter(run / "bootstrap", role)
    import ctypes
    import statistics
    import torch
    import torch.distributed as dist

    rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != 8:
        raise ValueError("Communication preflight requires eight local ranks")
    library = Path(os.environ["VLLM_NCCL_SO_PATH"]).resolve(strict=True)
    mapped = Path("/proc/self/maps").read_text()
    if str(library) not in mapped or torch.cuda.nccl.version() != (2, 27, 5):
        raise RuntimeError("The path-adapted NCCL library is not loaded")
    version = ctypes.c_int()
    if ctypes.CDLL(None).ncclGetVersion(ctypes.byref(version)) != 0 or version.value != 22705:
        raise RuntimeError("Wrong globally bound NCCL ABI")
    torch.cuda.set_device(rank)
    device = torch.cuda.get_device_properties(rank)
    if "H100" not in device.name or not 78 * 2**30 <= device.total_memory <= 82 * 2**30:
        raise RuntimeError("Communication preflight is not running on H10080GB")
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    try:
        tensor = torch.empty(2**21, device="cuda", dtype=torch.float32)
        durations = []
        for index in range(12):
            tensor.fill_(rank)
            torch.cuda.synchronize()
            started = time.perf_counter()
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if not torch.all(tensor == 28).item():
                raise AssertionError("NCCL all-reduce produced an incorrect result")
            if index >= 2:
                durations.append(elapsed)
        row = {"rank": rank, "hostname": socket.gethostname().split(".")[0],
               "gpu_uuid": str(device.uuid), "gpu_model": device.name,
               "all_reduce_bytes": tensor.numel() * tensor.element_size(),
               "median_seconds": statistics.median(durations), "nccl_library": str(library)}
        gathered = [None] * 8 if rank == 0 else None
        dist.gather_object(row, gathered, dst=0)
        if rank == 0:
            if len({item["gpu_uuid"] for item in gathered}) != 8:
                raise ValueError("Preflight ranks did not use distinct GPUs")
            result = {"status": "passed", "scope": "eight-GPU collective, not full-model training",
                      "job_id": os.environ["SLURM_JOB_ID"], "ranks": gathered}
            comparison.immutable(run / "preflight" / role / "COMM_COMPLETE.json", result)
        dist.barrier()
        return row
    finally:
        dist.destroy_process_group()


def prepare(run: Path) -> dict:
    bootstrap = enter(run / "bootstrap")
    arm = single_arm(bootstrap)
    protocol, state, rows, pinned = inputs()
    config = copy.deepcopy(bootstrap["config"])
    root = run / "training"
    if root.exists():
        raise FileExistsError("Training preparation already exists")
    config.update(
        run=str(root), engineer_model=state["engineer"], solver_model=state["solver"],
        no_patch=protocol["no_patch"], model_revision=comparison.REVISION,
        training_kernel_overlay=str(OVERLAY), training_kernel_versions=KERNEL_VERSIONS,
    )
    for role in comparison.ROLES:
        for name in ("KERNEL_COMPLETE.json", "COMM_COMPLETE.json"):
            path = run / "preflight" / role / name
            if comparison.load(path).get("status") != "passed":
                raise ValueError("Both nodes must pass kernel and communication preflight")
            pinned[str(path)] = comparison.sha(path)
    repo = Path(__file__).resolve().parents[2]
    for path in sorted((repo / "whale_local").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".patch", ".yaml", ".json"):
            pinned[str(path)] = comparison.sha(path)
    for name, value in (("joint_data.json", rows), ("PROTOCOL.json", protocol), ("INITIAL_STATE.json", state)):
        path = root / name
        comparison.immutable(path, value)
        pinned[str(path)] = comparison.sha(path)
    specs = {}
    for role in comparison.ROLES:
        role_config = {**config, "training_kernel_validation": str(run / "preflight" / role / "KERNEL_COMPLETE.json")}
        spec = comparison.role_spec(role_config, arm, role, root / "joint_data.json", pinned)
        spec["execution_mode"] = "standalone-ma-v1" if arm == "ma_sao" else "standalone-ma-ablation-v1"
        spec["baseline_definition"] = comparison.STANDALONE_MA if arm == "ma_sao" else comparison.STANDALONE_ABLATION
        comparison.validate_method(spec)
        path = root / arm / f"{role}_spec.json"
        comparison.immutable(path, spec)
        specs[str(path)] = comparison.sha(path)
    plan = {"schema": comparison.SCHEMA, "kind": (
                "standalone-ma-training" if arm == "ma_sao" else "standalone-ma-ablation-training"),
            "config": config, "inputs": pinned, "specs": specs,
            "budget_per_arm": comparison.budget(), "created_at_unix": time.time(),
            "evaluation_after": "this arm's complete144 checkpoint; no wait for other arms",
            "native_model_validation": "first real training run; not claimed by operator/collective preflight"}
    comparison.immutable(root / "COMPARISON.json", plan)
    return comparison.verify(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("check-inputs", "bind", "kernel", "collective", "prepare", "role", "finish", "evaluate"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--role", choices=comparison.ROLES)
    parser.add_argument("--nodes", nargs=2)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--arm", choices=("ma_sao", *comparison.ABLATION_MECHANISMS), default="ma_sao")
    args = parser.parse_args()
    if args.arm != "ma_sao" and args.phase != "bind":
        parser.error("--arm is selected only while binding the allocation")
    if args.phase == "check-inputs":
        _, _, rows, pinned = inputs()
        result = {"status": "input-contract-passed", "rows": len(rows), "pinned_inputs": pinned}
    else:
        if args.run is None:
            parser.error("--run is required")
        run = comparison.contained(args.run)
        if args.phase == "bind":
            if args.nodes is None or args.deadline is None:
                parser.error("bind requires --nodes and --deadline")
            result = bind(run, args.nodes, args.deadline, arm=args.arm)
        elif args.phase in ("kernel", "collective", "role"):
            if args.role is None:
                parser.error("this phase requires --role")
            if args.phase == "kernel":
                result = kernel(run, args.role)
            elif args.phase == "collective":
                result = collective(run, args.role)
            else:
                plan = enter(run / "training", args.role)
                comparison.verify(run / "training")
                from .joint_experiment import run_role
                result = run_role(run / "training" / single_arm(plan) / f"{args.role}_spec.json")
        elif args.phase == "prepare":
            result = prepare(run)
        elif args.phase == "finish":
            plan = enter(run / "training")
            result = comparison.final_pairs(run / "training", arms=(single_arm(plan),))
            comparison.immutable(run / "TRAINING_144_COMPLETE.json", result)
        else:
            plan = enter(run / "training", "editor")
            result = comparison.evaluate_final(run / "training", single_arm(plan), None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
