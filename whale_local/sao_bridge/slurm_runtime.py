"""Opt-in, allocation-bound adapter; never submits jobs or touches legacy services."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time

from .concurrent_comparison import ARMS, MECHANISMS, ROLES, SCHEMA, REMOTE_ROOT, contained, load, sha


ENV = ("COEVOLVE_COMPARISON_ROOT", "COEVOLVE_COMPARISON_SHA256")


def active() -> bool:
    values = [os.environ.get(key) for key in ENV]
    if any(values) and not all(values):
        raise ValueError("Incomplete allocation-bound comparison identity")
    return all(values)


def binding() -> tuple[Path, dict]:
    if not active():
        raise ValueError("No explicit concurrent-comparison execution identity")
    root = contained(Path(os.environ[ENV[0]]))
    path = root / "COMPARISON.json"
    if sha(path) != os.environ[ENV[1]]:
        raise ValueError("Allocation comparison identity changed")
    plan = load(path)
    if plan.get("schema") != SCHEMA or Path(plan["config"]["run"]).resolve() != root:
        raise ValueError("Wrong comparison schema/root")
    if (root / "CANCELLED.json").exists() or (root / "FAILED.json").exists():
        raise RuntimeError("Comparison cancelled/failed; no automatic restart")
    return root, plan


def require_node() -> dict:
    root, plan = binding()
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("GPU work requires an existing Slurm allocation")
    text = subprocess.check_output(
        ["scontrol", "show", "job", "-o", job], text=True, timeout=30,
    )
    fields = dict(re.findall(r"(\w+)=(\S+)", text))
    arms = plan["config"].get("active_arms", list(ARMS))
    if not arms or len(set(arms)) != len(arms) or any(arm not in MECHANISMS for arm in arms):
        raise ValueError("Invalid allocated comparison arms")
    expected = [
        plan["config"]["nodes"][arm][role] for arm in arms for role in ROLES
    ]
    if (
        fields.get("JobState") != "RUNNING"
        or fields.get("NumNodes") != str(len(expected))
        or not fields.get("UserId", "").endswith(f"({os.getuid()})")
    ):
        raise RuntimeError("Expected this user's running allocation with the declared node count")
    nodes = subprocess.check_output(
        ["scontrol", "show", "hostnames", fields["NodeList"]], text=True, timeout=30,
    ).split()
    host = os.uname().nodename.split(".")[0]
    if len(nodes) != len(expected) or len(set(expected)) != len(expected) or set(nodes) != set(expected) or host not in nodes:
        raise RuntimeError("Current host/allocation does not match the bound experiment nodes")
    return {"job_id": job, "host": host, "nodes": nodes,
            "comparison": str(root), "comparison_sha256": os.environ[ENV[1]]}


def audit() -> dict:
    from .resources import process_identity

    allocation = require_node()
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits",
    ], text=True, timeout=30).splitlines()
    if len(devices) != 8:
        raise RuntimeError("Each comparison role requires eight H10080GB GPUs")
    for index, line in enumerate(devices):
        number, name, memory = (part.strip() for part in line.split(","))
        if int(number) != index or "H100" not in name or not 78000 <= float(memory) <= 83000:
            raise RuntimeError("Allocation GPU hardware differs from the H10080GB contract")
    pids = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=30).split()
    records, vanished = [], []
    for pid in sorted(set(map(int, pids))):
        try:
            records.append({**process_identity(pid), "verified_filler": False, "filler_kind": None})
        except FileNotFoundError:
            vanished.append(pid)
    return {
        "host": allocation["host"], "time": time.time(), "unit": {"ActiveState": "inactive"},
        "supervisor": None, "processes": records, "vanished_gpu_pids": vanished,
        "filler_kind": None, "verified_unit": False, "slurm_allocation": allocation,
        "stopped": False, "cooperative": False,
    }


def runtime_directories(launch_root: Path) -> tuple[Path, Path]:
    root, plan = binding()
    launch_root = contained(launch_root, root)
    cache = cache_directory(root, plan)
    ipc = contained(launch_root / "ipc", root)
    return cache, ipc


def cache_directory(root: Path, plan: dict) -> Path:
    owner = Path(plan["config"].get("study_run", str(root)))
    if owner not in (root, root.parent):
        raise ValueError("Runtime cache owner must be the run or its job directory")
    default = root / "runtime"
    base = contained(Path(plan["config"].get("runtime_cache_root", str(default))), owner)
    return contained(base / os.uname().nodename / "cache", owner)


def worker_python() -> Path:
    _, plan = binding()
    path = Path(plan["config"]["worker_python"])
    expected = REMOTE_ROOT / ".envs/alfworld/bin/python"
    configuration = expected.parent.parent / "pyvenv.cfg"
    if (path != expected or path.parent.resolve() != path.parent
            or configuration.resolve() != configuration or not configuration.is_file()
            or not contained(path, REMOTE_ROOT).is_file()):
        raise ValueError("Worker must use the approved project-local ALFWorld virtualenv")
    # Resolving the executable before launching would discard pyvenv.cfg.
    return path


def rpc_directory() -> Path:
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch(r"[0-9]+", job):
        raise RuntimeError("Comparison IPC requires an existing Slurm allocation")
    requested = REMOTE_ROOT / "tmp" / f"vllm-{job}"
    path = contained(requested, REMOTE_ROOT)
    if path != requested:
        raise ValueError("Comparison RPC directory is redirected")
    # vLLM appends a UUID; Linux sockaddr_un includes the terminating NUL.
    if len(os.fsencode(str(path))) + 1 + 36 >= 108:
        raise ValueError("Comparison RPC path exceeds the Unix-socket path limit")
    return path


def execution_environment(root: Path) -> dict[str, str]:
    if os.environ.get("SAO_RESOURCE_AUTHORITY") or os.environ.get("SAO_RESOURCE_AUTHORITY_SHA256"):
        raise ValueError("Never combine new Slurm allocation authority with an old study gate")
    root = contained(root)
    plan = load(root / "COMPARISON.json")
    runtime = contained(root / "runtime" / os.uname().nodename / "controller", root)
    cache = cache_directory(root, plan)
    owner = Path(plan["config"].get("study_run", str(root)))
    directories = {
        "TMPDIR": runtime / "ipc", "TMP": runtime / "ipc", "TEMP": runtime / "ipc",
        "HOME": cache / "home", "XDG_CACHE_HOME": cache,
        "HF_HOME": cache / "huggingface", "MPLCONFIGDIR": cache / "matplotlib",
        "TRITON_CACHE_DIR": cache / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torch",
        "VLLM_CACHE_ROOT": cache / "vllm", "CUDA_CACHE_PATH": cache / "cuda",
        "AREAL_CACHE_DIR": cache / "areal",
        "FLASHINFER_WORKSPACE_BASE": cache / "flashinfer",
        "XDG_CONFIG_HOME": cache / "home/.config",
        "XDG_DATA_HOME": cache / "home/.local/share",
        "XDG_STATE_HOME": cache / "home/.local/state",
        "XDG_RUNTIME_DIR": runtime / "xdg",
        "TORCH_HOME": cache / "torch-home", "TORCH_EXTENSIONS_DIR": cache / "extensions",
    }
    for path in set(directories.values()):
        contained(path, owner).mkdir(parents=True, exist_ok=True)
    rpc = rpc_directory()
    rpc.mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {
        ENV[0]: str(root), ENV[1]: sha(root / "COMPARISON.json"),
        "NCCL_SHM_DISABLE": "1", "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "VLLM_RPC_BASE_PATH": str(rpc),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn", "PYTHONDONTWRITEBYTECODE": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        **{name: str(path) for name, path in directories.items()},
    }
    # Configuration alone does not prove third-party code avoids /dev/shm.
    # The separate native gate requires an audited IPC/write-boundary receipt.
    env["SAO_ALFWORLD_WORKER_PYTHON"] = plan["config"]["worker_python"]
    return env
