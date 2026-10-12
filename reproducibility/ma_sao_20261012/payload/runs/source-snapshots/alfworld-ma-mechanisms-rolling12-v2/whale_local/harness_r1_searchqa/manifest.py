"""Frozen manifest loader with explicit split access (training never opens test).

Only requested split files are read; the preparation manifest supplies the
cross-split freeze. Within selected splits, identities/questions cannot overlap.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from sao_bridge.searchqa_domain import METRIC
from sao_bridge.searchqa_prepare import question_key, sha256


def load_manifest_splits(root: Path, *, splits: tuple[str, ...] = ("train",)) -> dict:
    root = Path(root).resolve()
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"train", "reward", "dev", "test"}:
        raise ValueError("Explicit valid split restriction is required")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("benchmark") != "searchqa" or manifest.get("metric") != METRIC:
        raise ValueError("Unexpected SearchQA manifest/verifier")
    loaded, seen_ids, seen_questions = {}, set(), set()
    for split in splits:
        entry = manifest["splits"][split]
        path = (root / entry["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Split file escapes manifest root")
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen SearchQA split changed: {split}")
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or len(rows) != entry["count"]:
            raise ValueError("Split count/schema mismatch")
        digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
        if digest != entry["ids_sha256"]:
            raise ValueError("Split ids changed")
        for row in rows:
            identity = row["id"]
            question = question_key(row["question"])
            if (not re.fullmatch(r"[0-9a-f]{64}", identity)
                    or identity != hashlib.sha256(row["source_row_key"].encode()).hexdigest()):
                raise ValueError("SearchQA row identity mismatch")
            if identity in seen_ids or question in seen_questions:
                raise ValueError("SearchQA split overlap")
            if not row["answers"] or row["answer"] != row["answers"][0]:
                raise ValueError("Invalid answer-alias compatibility")
            seen_ids.add(identity)
            seen_questions.add(question)
        loaded[split] = rows
    return loaded
