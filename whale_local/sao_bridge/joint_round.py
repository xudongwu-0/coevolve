"""Shared, version-bound records for paired Engineer/Solver rollout rounds."""
from __future__ import annotations

import asyncio
from pathlib import Path
import time

from .alfworld_train_data import _immutable_json
from .common import read_json


ROLES = ("editor", "solver")


def check_role(role: str) -> None:
    if role not in ROLES:
        raise ValueError("Joint role must be editor or solver")


def round_path(root: Path, version: int) -> Path:
    if type(version) is not int or version < 0:
        raise ValueError("Joint version must be a nonnegative integer")
    return root / f"round-{version:04d}"


def publish_version(root: Path, role: str, version: int) -> None:
    check_role(role)
    _immutable_json(round_path(root, version) / f"{role}_ready.json", {
        "role": role, "version": version,
    })


def check_failures(root: Path) -> None:
    for role in ROLES:
        failed = root / f"{role}_FAILED.json"
        if failed.exists():
            raise RuntimeError(f"Joint partner failed: {read_json(failed)}")


async def wait_record(path: Path, root: Path, timeout: float = 1800) -> dict:
    deadline = time.monotonic() + timeout
    while not path.exists():
        check_failures(root)
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Joint record did not arrive: {path}")
        await asyncio.sleep(0.2)
    return read_json(path)


async def wait_versions(root: Path, version: int, timeout: float = 1800) -> None:
    for role in ROLES:
        value = await wait_record(round_path(root, version) / f"{role}_ready.json", root, timeout)
        if value != {"role": role, "version": version}:
            raise ValueError("Joint policy publication identity changed")


def checked_behavior(calls: list[dict], version: int) -> None:
    import math

    if not calls:
        raise ValueError("Joint episode has no authentic policy action")
    for call in calls:
        size = len(call["response_ids"])
        if (
            not size or len(call["response_logprobs"]) != size
            or call["versions"] != [version] * size
            or any(not math.isfinite(value) or value > 1e-6 for value in call["response_logprobs"])
        ):
            raise ValueError("Joint trajectory crossed policy versions or lost behavior probabilities")


def install_version_publication(trainer, root: Path, role: str) -> None:
    check_role(role)
    original = trainer.rollout.set_version

    def set_version(version):
        original(version)
        publish_version(root, role, version)

    trainer.rollout.set_version = set_version
    publish_version(root, role, 0)
