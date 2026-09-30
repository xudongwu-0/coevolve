#!/usr/bin/env python3
"""Reproducible public Lichess prefix selection, with bounded HTTP Range downloads."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys

import chess
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge.chess_domain import position_key, validate_reference

REVISION = "5dc959d3f4e3413d1fe5b19a09ba6e595c5ed5c1"
UPSTREAM_REVISION = "fbe125eb7abea7f760c99ab9acc1a6261e708fc6"
SHARDS = [
    ("data/train-00000-of-00003.parquet", 292245200,
     "a759851c458c5234948c529e7e6995f483a4fd5695c40313714444ac15cd44f4"),
    ("data/train-00001-of-00003.parquet", 292116997,
     "7e2e60149e62000c9d2ef6f9551c0d8b9d8f9227f634d5a180fa4642f9ef1806"),
    ("data/train-00002-of-00003.parquet", 292133272,
     "e1e717c6ade5796c32646690c09ddedecd303c6d3bd68068fef0d8d9695bc451"),
]
SPLITS = {"train": 16384, "reward": 128, "dev": 256, "test": 256}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()


def write_public(path: Path, data: bytes):
    path.write_bytes(data)
    path.chmod(0o644)


class RangeSource(io.RawIOBase):
    """Persist only requested source byte ranges, never silently download a shard."""

    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "source_ranges"
        self.directory.mkdir(exist_ok=True)
        self.directory.chmod(0o755)
        self.path, self.size, self.source_sha256 = SHARDS[0]
        self.url = f"https://huggingface.co/datasets/Lichess/chess-puzzles/resolve/{REVISION}/{self.path}"
        self.position = 0
        self.ranges = []
        self.downloaded = 0
        manifest_path = root / "manifest.json"
        self.expected_ranges = {}
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["revision"] != REVISION:
                raise ValueError("Existing source cache has a different pinned revision")
            self.expected_ranges = {entry["path"]: entry["sha256"]
                                    for entry in manifest["source_selection"]["source_ranges"]}
        self.session = requests.Session()
        self.session.trust_env = False

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else (
            self.position + offset if whence == 1 else self.size + offset
        )
        if not 0 <= position <= self.size:
            raise ValueError("Source seek is out of bounds")
        self.position = position
        return position

    def read(self, size=-1):
        start = self.position
        end = min(self.size, start + size) if size >= 0 else self.size
        if start == end:
            return b""
        if end - start > 64 * 1024 * 1024:
            raise ValueError("Refusing a source range larger than 64 MiB")
        path = self.directory / f"{start:012d}-{end:012d}.bin"
        if path.exists():
            data = path.read_bytes()
        else:
            with self.session.get(
                self.url + f"?download=true&sao_range={start}-{end}",
                headers={"Range": f"bytes={start}-{end - 1}", "Accept-Encoding": "identity"},
                stream=True, timeout=(20, 120),
            ) as response:
                response.raise_for_status()
                expected = f"bytes {start}-{end - 1}/{self.size}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError("Source does not honor exact HTTP ranges; refusing fallback")
                data = response.raw.read(end - start + 1)
            if len(data) != end - start:
                raise RuntimeError("Wrong source range length")
            write_public(path, data)
            self.downloaded += len(data)
        if len(data) != end - start:
            raise RuntimeError("Corrupt cached source range")
        entry = {"path": str(path.relative_to(self.root)), "start": start, "end_exclusive": end,
                 "bytes": len(data), "sha256": digest(data)}
        expected_sha256 = self.expected_ranges.get(entry["path"])
        if expected_sha256 is not None and entry["sha256"] != expected_sha256:
            raise RuntimeError("Cached source range checksum mismatch")
        if entry not in self.ranges:
            self.ranges.append(entry)
        self.position = end
        return data


def convert_row(raw: dict, source_row: int) -> dict:
    initial_fen = raw["FEN"]
    moves = raw["Moves"].lower().split()
    if not 2 <= len(moves) <= 10:
        raise ValueError("solution_length")
    board = chess.Board(initial_fen)
    if not board.is_valid():
        raise ValueError("invalid_position")
    trigger = chess.Move.from_uci(moves[0])
    if trigger not in board.legal_moves:
        raise ValueError("illegal_trigger")
    board.push(trigger)
    row = {
        "id": digest(f"lichess:{raw['PuzzleId']}".encode()), "puzzle_id": raw["PuzzleId"],
        "source": "Lichess/chess-puzzles",
        "source_revision": REVISION, "source_shard": SHARDS[0][0], "source_row": source_row,
        "initial_fen": initial_fen, "trigger_move": moves[0], "start_fen": board.fen(),
        "solution_moves": moves[1:], "rating": int(raw["Rating"]),
        "themes": raw["Themes"] if isinstance(raw["Themes"], list) else raw["Themes"].split(),
        "position_key": position_key(board.fen()),
    }
    validate_reference(row)
    return row


def partition(rows: list[dict], sizes: dict[str, int] = SPLITS, seed: int = 42) -> dict:
    if len(rows) != sum(sizes.values()):
        raise ValueError("Need exactly the requested number of globally unique rows")
    if len({row["position_key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate starting position/side")
    ordered = sorted(rows, key=lambda row: (
        digest(f"{seed}:{row['position_key']}".encode()), row["position_key"]
    ))
    result, cursor = {}, 0
    for split, size in sizes.items():
        result[split] = [{**row, "split": split} for row in ordered[cursor:cursor + size]]
        cursor += size
    return result


def prepare(output: Path, seed: int = 42):
    output.mkdir(parents=True, exist_ok=True)
    output.chmod(0o755)
    source = RangeSource(output)
    parquet = pq.ParquetFile(source)
    needed = sum(SPLITS.values())
    rows, seen, seen_ids, skipped, row_groups = [], set(), set(), Counter(), []
    source_rows = bytearray()
    scanned = 0
    for group in range(parquet.num_row_groups):
        examined = 0
        for batch in parquet.iter_batches(batch_size=2048, row_groups=[group]):
            for raw in batch.to_pylist():
                source_rows.extend(json_bytes(raw))
                source_row = scanned
                scanned += 1
                examined += 1
                try:
                    row = convert_row(raw, source_row)
                except (ValueError, KeyError, TypeError) as exc:
                    skipped[str(exc)[:120]] += 1
                    continue
                if row["position_key"] in seen:
                    skipped["duplicate_starting_position_and_side"] += 1
                    continue
                if row["id"] in seen_ids:
                    skipped["duplicate_puzzle_id"] += 1
                    continue
                seen.add(row["position_key"])
                seen_ids.add(row["id"])
                rows.append(row)
                if len(rows) == needed:
                    break
            if len(rows) == needed:
                break
        row_groups.append({"index": group, "rows_in_group": parquet.metadata.row_group(group).num_rows,
                           "rows_examined": examined})
        if len(rows) == needed:
            break
    if len(rows) != needed:
        raise RuntimeError("Pinned first shard did not fill requested splits; no fallback permitted")
    raw_path = output / "source_rows.jsonl"
    write_public(raw_path, bytes(source_rows))
    splits = partition(rows, seed=seed)
    artifacts, canonical_artifacts = {}, {}
    for name, values in splits.items():
        data = b"".join(json_bytes(row) for row in values)
        path = output / f"{name}.jsonl"
        write_public(path, data)
        artifacts[name] = {"path": path.name, "rows": len(values), "count": len(values), "bytes": len(data),
                           "sha256": digest(data),
                           "solution_plies": dict(sorted(Counter(len(r["solution_moves"]) for r in values).items()))}
        canonical_data = json_bytes(values)
        canonical_path = output / f"{name}.json"
        write_public(canonical_path, canonical_data)
        canonical_artifacts[name] = {
            **artifacts[name], "path": canonical_path.name, "bytes": len(canonical_data),
            "sha256": digest(canonical_data),
        }
    manifest = {
        "schema_version": 1, "domain": "chess", "created_date": "2026-09-29",
        "description": "Local native SAO/Qwen3-4B WHALE-style replication; not exact-paper scores.",
        "dataset": "Lichess/chess-puzzles", "revision": REVISION, "license": "CC0-1.0",
        "dataset_url": f"https://huggingface.co/datasets/Lichess/chess-puzzles/tree/{REVISION}",
        "full_dataset_download_bytes": sum(shard[1] for shard in SHARDS),
        "upstream": {"repository": "https://github.com/krafton-ai/WHALE",
                     "revision": UPSTREAM_REVISION, "path": "domains/chess_puzzles",
                     "adapted_semantics": "Apply trigger move, 1..9 legal reference plies, alternate native opponent replies, exact full-sequence success.",
                     "differences": "Qwen3-4B; 4096 default tokens; reward128/dev256/test256; isolated CandidateHarness hooks; conservative global dedup; deterministic prefix then hash-ranked splits."},
        "public_shards": [{"path": p, "bytes": n, "lfs_sha256": h} for p, n, h in SHARDS],
        "source_selection": {
            "shards": [SHARDS[0][0]], "order": "Physical Parquet row order, first shard only.",
            "rows_examined": scanned, "row_interval": [0, scanned],
            "row_groups": row_groups, "selected_unique_rows": len(rows), "skipped": dict(skipped),
            "raw_rows_path": raw_path.name, "raw_rows_sha256": digest(bytes(source_rows)),
            "raw_rows_bytes": len(source_rows),
            "range_bytes_retained": sum(entry["bytes"] for entry in source.ranges),
            "transfer_note": "Parquet column chunks span the first 489949-row group: 70.38 MB is retained to decode the 17157-row prefix; other row groups and shards are not downloaded.",
            "source_ranges": source.ranges,
            "full_shard_hash_verified": False,
            "checksum_note": "Full shard LFS SHA256 is pinned public metadata, not a claimed full download verification. Retained exact byte ranges and canonical raw rows have locally verified SHA256.",
            "bias": "Deterministic prefix subset, not uniform sampling from the full 6.1M-row source.",
        },
        "filters": {"solution_plies_min": 1, "solution_plies_max": 9,
                    "legal_initial_position": True, "legal_trigger": True,
                    "legal_entire_reference": True, "rating_filter": None},
        "partition": {"seed": seed, "global_dedup": "start_fen piece placement + side to move; ignores clocks/castling/en-passant",
                      "algorithm": "Select first 17024 valid unique rows, sort by SHA256(str(seed)+':'+position_key), assign contiguous train/reward/dev/test quotas.",
                      "sizes": SPLITS, "cross_split_position_overlap": 0, "cross_split_id_overlap": 0},
        "dependencies": {"chess": chess.__version__, "pyarrow": __import__("pyarrow").__version__,
                         "requests": requests.__version__},
        "row_schema": {
            "id": "SHA256('lichess:'+PuzzleId), lowercase hex; stable and safe for int(id[:8],16) seeds.",
            "puzzle_id": "Original public Lichess PuzzleId, retained for source provenance only.",
            "start_fen": "Public puzzle position after the source trigger.",
            "solution_moves": "Private-to-verifier reference; never pass rows to solver/hooks/editor.",
            "feedback": "Rows have no transcript. Use training run_episode messages/failure_trace, not source labels.",
        },
        "splits": canonical_artifacts,
        "files": artifacts,
        "file_formats": {
            "splits": "Canonical controller input: split.json UTF-8 JSON arrays.",
            "files": "Equivalent split.jsonl copies, one row per line, retained for streaming consumers.",
        },
        "reproduce": "/dev/shm/sao-venv-ppo/bin/python scripts/prepare_sao_chess.py --output assets/chess_sao_20260929",
        "verify": "/dev/shm/sao-venv-ppo/bin/python scripts/prepare_sao_chess.py --verify --output assets/chess_sao_20260929",
    }
    write_public(output / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n")
    source.session.close()
    return verify(output)


def verify(output: Path):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["revision"] != REVISION or manifest["partition"]["sizes"] != SPLITS:
        raise ValueError("Unexpected source revision or split sizes")
    positions, ids, rows = set(), set(), []
    for split, count in SPLITS.items():
        meta = manifest["splits"][split]
        if meta["rows"] != count or meta["count"] != count:
            raise ValueError("Wrong manifest split count")
        path = output / meta["path"]
        data = path.read_bytes()
        if digest(data) != meta["sha256"] or len(data) != meta["bytes"]:
            raise ValueError("Split checksum mismatch")
        values = json.loads(data)
        if len(values) != count:
            raise ValueError("Wrong split size")
        streaming_meta = manifest["files"][split]
        streaming_data = (output / streaming_meta["path"]).read_bytes()
        if (digest(streaming_data) != streaming_meta["sha256"]
                or len(streaming_data) != streaming_meta["bytes"]
                or [json.loads(line) for line in streaming_data.splitlines()] != values):
            raise ValueError("JSON and JSONL split representations do not match")
        for row in values:
            if row["id"] != digest(f"lichess:{row['puzzle_id']}".encode()):
                raise ValueError("Wrong stable hexadecimal puzzle ID")
            validate_reference(row)
            board = chess.Board(row["initial_fen"])
            trigger = chess.Move.from_uci(row["trigger_move"])
            if trigger not in board.legal_moves:
                raise ValueError("Illegal source trigger")
            board.push(trigger)
            key = position_key(row["start_fen"])
            if board.fen() != row["start_fen"] or row["position_key"] != key:
                raise ValueError("Wrong derived starting position")
            if key in positions or row["id"] in ids:
                raise ValueError("Global duplicate or split overlap")
            if row["split"] != split or row["source_revision"] != REVISION:
                raise ValueError("Wrong row provenance")
            positions.add(key)
            ids.add(row["id"])
            rows.append(row)
    selection = manifest["source_selection"]
    raw = (output / selection["raw_rows_path"]).read_bytes()
    if digest(raw) != selection["raw_rows_sha256"] or len(raw) != selection["raw_rows_bytes"]:
        raise ValueError("Raw source-row checksum mismatch")
    retained = [json.loads(line) for line in raw.splitlines()]
    if len(retained) != selection["rows_examined"]:
        raise ValueError("Wrong scanned-row count")
    for row in rows:
        expected = convert_row(retained[row["source_row"]], row["source_row"])
        if {k: v for k, v in row.items() if k != "split"} != expected:
            raise ValueError("Selected row differs from retained public source")
    expected_splits = partition(rows, seed=manifest["partition"]["seed"])
    for name in SPLITS:
        expected = b"".join(json_bytes(row) for row in expected_splits[name])
        if digest(expected) != manifest["files"][name]["sha256"]:
            raise ValueError("Partition is not reproducible")
        if digest(json_bytes(expected_splits[name])) != manifest["splits"][name]["sha256"]:
            raise ValueError("Canonical JSON partition is not reproducible")
    for entry in selection["source_ranges"]:
        data = (output / entry["path"]).read_bytes()
        if digest(data) != entry["sha256"] or len(data) != entry["end_exclusive"] - entry["start"]:
            raise ValueError("Source-range checksum mismatch")
    for path in [output, *output.rglob("*")]:
        expected_mode = 0o755 if path.is_dir() else 0o644
        if path.stat().st_mode & 0o777 != expected_mode:
            raise ValueError(f"Unexpected asset mode: {path}")
    return {"verified": True, "splits": SPLITS, "unique_positions": len(positions),
            "rows_examined": selection["rows_examined"],
            "retained_source_bytes": selection["range_bytes_retained"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "assets/chess_sao_20260929")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output) if args.verify else prepare(args.output, args.seed), sort_keys=True))


if __name__ == "__main__":
    main()
