"""Import unchanged, hash-pinned Harness-R1/Relax training primitives."""
from __future__ import annotations

from functools import lru_cache
import importlib
from pathlib import Path
import sys
from types import ModuleType

from sao_bridge.common import file_sha

ROOT = Path(__file__).resolve().parents[2] / "third_party/harness-r1-upstream/code/Relax"
SOURCE_SHA256 = {
    "relax/utils/training/ppo_utils.py":
        "c3bc628e43b688379aca9a4149bcbd4cf877bea82ef86b713f9cb3394ebaedf3",
    "relax/utils/utils.py":
        "1073c99e9f258a5c08559cef0fd2eb6a986031ddc94dc88707c584f27138fc9e",
}


@lru_cache(maxsize=1)
def official_modules() -> tuple[ModuleType, ModuleType]:
    for name, expected in SOURCE_SHA256.items():
        if file_sha(ROOT / name) != expected:
            raise RuntimeError(f"Pinned official Harness-R1 source changed: {name}")
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    ppo = importlib.import_module("relax.utils.training.ppo_utils")
    rewards = importlib.import_module("relax.utils.utils")
    for module in (ppo, rewards):
        if module.__file__ is None or not Path(module.__file__).resolve().is_relative_to(ROOT.resolve()):
            raise RuntimeError("Imported Relax module is not from the pinned Harness-R1 checkout")
    return ppo, rewards
