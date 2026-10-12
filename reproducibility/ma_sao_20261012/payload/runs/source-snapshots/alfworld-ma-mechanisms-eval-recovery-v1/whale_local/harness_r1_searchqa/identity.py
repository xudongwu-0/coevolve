"""Provenance shared by baseline generation, failure packets and reward workers.

Baseline report producers must store ``baseline_config`` from ``baseline_config``
and ``split="train"`` before collection. Never retrofit a changed configuration
onto an already collected report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sao_bridge.searchqa_prepare import sha256

from .patch import patch_digest, UPSTREAM_CODE_RUNNER


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime_source_identity() -> str:
    directory = Path(__file__).resolve().parent
    # Training/bootstrap utilities are not part of hook or reward execution.
    sources = [directory / name for name in (
        "__init__.py", "patch.py", "runtime.py", "sandbox.py", "episode.py",
        "reward.py", "data.py", "prompt.py", "manifest.py", "identity.py",
    )] + [UPSTREAM_CODE_RUNNER]
    sources += [directory.parent / "sao_bridge" / name for name in
                ("program_harness.py", "evaluation.py", "searchqa_domain.py",
                 "searchqa_retrieval.py", "searchqa_prepare.py", "retool.py")]
    return digest({str(path.relative_to(directory.parents[1])): sha256(path)
                   for path in sorted(sources)})


def baseline_config(*, target_model_sha256: str, target_temperature: float,
                    total_tokens: int, retrieval_manifest: Path,
                    data_manifest: Path, incumbent_patch: dict) -> dict:
    from .episode import H0_CONFIG
    if (type(target_temperature) not in (int, float) or not 0 <= target_temperature <= 2
            or type(total_tokens) is not int or total_tokens <= 0
            or not isinstance(target_model_sha256, str) or not target_model_sha256):
        raise ValueError("Invalid frozen target configuration")
    return {
        "target_model_sha256": target_model_sha256,
        "target_temperature": target_temperature,
        "total_tokens": total_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "retrieval_manifest_sha256": sha256(retrieval_manifest),
        "data_manifest_sha256": sha256(data_manifest),
        "incumbent_patch_sha256": patch_digest(incumbent_patch),
        "runtime_source_sha256": runtime_source_identity(),
        "episode_config": dict(H0_CONFIG),
    }
