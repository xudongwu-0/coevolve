"""Create a run-local AgentBench layout for official ALFWorld workers."""
from __future__ import annotations

from pathlib import Path

from .common import ROOT

UPSTREAM_AGENTBENCH = (
    ROOT / "third_party/harness-r1-upstream/code/life-harness/AgentBench"
)
ALFWORLD_ASSETS = ROOT / "assets/alfworld_harness_r1"
ALFWORLD_DATA = ALFWORLD_ASSETS / "data"


def _link(link: Path, target: Path) -> None:
    target = target.resolve()
    if link.is_symlink():
        if link.resolve() != target:
            raise RuntimeError(f"AgentBench runtime link changed: {link}")
        return
    if link.exists():
        raise RuntimeError(f"AgentBench runtime path is not a symlink: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def prepare_agentbench_runtime(runtime: Path) -> Path:
    runtime = runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    for name in ("src", "scripts"):
        _link(runtime / name, UPSTREAM_AGENTBENCH / name)
    configs = runtime / "configs"
    configs.mkdir(exist_ok=True)
    for entry in (UPSTREAM_AGENTBENCH / "configs").iterdir():
        if entry.name != "generated":
            _link(configs / entry.name, entry)
    (configs / "generated").mkdir(exist_ok=True)
    data = runtime / "data"
    data.mkdir(exist_ok=True)
    _link(data / "alfworld", ALFWORLD_DATA)
    return runtime
