"""Reconstruct immutable SearchQA splits from pinned public FlashRAG files."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/searchqa_sao_20260929"
FLASHRAG_REPO = "RUC-NLPIR/FlashRAG_datasets"
FLASHRAG_REVISION = "bcafb8dd07d453be3cbeeeb3f78be1841bddf92c"
WHALE_REVISION = "fbe125eb7abea7f760c99ab9acc1a6261e708fc6"
EVAL_SOURCES = {
    "2wikimultihopqa": "dev", "bamboogle": "test", "hotpotqa": "dev",
    "musique": "dev", "nq": "test", "popqa": "test", "triviaqa": "test",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def immutable_json(path: Path, value) -> None:
    text = stable_json(value)
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to change frozen identity: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    with path.open("x") as stream:
        stream.write(text)
    path.chmod(0o644)


def question_key(question: str) -> str:
    text = unicodedata.normalize("NFKC", question).casefold()
    return " ".join(re.findall(r"\w+", text))


def download_source(pair: tuple[str, str], assets: Path) -> dict:
    source, split = pair
    filename = f"{source}/{split}.jsonl"
    url = f"https://huggingface.co/datasets/{FLASHRAG_REPO}/resolve/{FLASHRAG_REVISION}/{filename}"
    path = assets / "sources" / f"{source}_{split}.jsonl"
    receipt = path.with_suffix(".receipt.json")
    if receipt.exists():
        metadata = json.loads(receipt.read_text())
        if metadata["url"] != url or sha256(path) != metadata["sha256"]:
            raise RuntimeError(f"Pinned source integrity failure: {path}")
        return metadata
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    downloading = path.with_suffix(".downloading")
    with urllib.request.urlopen(url, timeout=180) as response, downloading.open("wb") as stream:
        while block := response.read(4 * 1024 * 1024):
            stream.write(block)
    downloading.replace(path)
    path.chmod(0o644)
    metadata = {"repository": FLASHRAG_REPO, "revision": FLASHRAG_REVISION,
                "file": filename, "url": url, "path": str(path.relative_to(assets)),
                "sha256": sha256(path), "bytes": path.stat().st_size}
    immutable_json(receipt, metadata)
    return metadata


def rows_from_source(source: str, split: str, assets: Path) -> list[dict]:
    rows = []
    with (assets / "sources" / f"{source}_{split}.jsonl").open() as stream:
        for index, line in enumerate(stream):
            raw = json.loads(line)
            question = raw["question"].strip()
            if not question.endswith("?"):
                question += "?"
            answers = list(dict.fromkeys(raw["golden_answers"]))
            if not question_key(question) or not answers or any(
                not isinstance(answer, str) or not answer.strip() for answer in answers
            ):
                raise ValueError(f"Invalid public QA row {source}/{split}/{index}")
            source_id = str(raw.get("id", index))
            source_row_key = f"searchqa:{source}:{split}:{index}:{source_id}"
            rows.append({
                "id": hashlib.sha256(source_row_key.encode()).hexdigest(),
                "source_row_key": source_row_key,
                "question": question, "answers": answers, "answer": answers[0],
                "data_source": source, "source_split": split,
                "source_id": source_id, "source_index": index,
            })
    return rows


def deterministic_take(rows: list[dict], count: int, salt: str, excluded: set[str]) -> list[dict]:
    def rank(row):
        key = row["source_row_key"]
        return hashlib.sha256(f"42:{salt}:{key}".encode()).hexdigest(), key
    selected = []
    for row in sorted(rows, key=rank):
        key = question_key(row["question"])
        if key not in excluded:
            selected.append(row)
            excluded.add(key)
            if len(selected) == count:
                return selected
    raise ValueError(f"Insufficient disjoint rows for {salt}: {len(selected)} < {count}")


def prepare(assets: Path = ASSETS) -> dict:
    assets = assets.resolve()
    pairs = [("nq", "train"), ("hotpotqa", "train"), *EVAL_SOURCES.items()]
    with ThreadPoolExecutor(max_workers=4) as pool:
        provenance = list(pool.map(lambda pair: download_source(pair, assets), pairs))
    used: set[str] = set()
    test = []
    for source, split in EVAL_SOURCES.items():
        test.extend(deterministic_take(rows_from_source(source, split, assets), 100,
                                       f"report-test:{source}", used))
    # Protect *all* official evaluation questions, not just the 700 reported rows.
    official_keys = set(used)
    for source, split in EVAL_SOURCES.items():
        official_keys.update(question_key(row["question"]) for row in rows_from_source(source, split, assets))
    used.update(official_keys)
    pools = {source: rows_from_source(source, "train", assets) for source in ("nq", "hotpotqa")}
    splits = {"train": [], "reward": [], "dev": [], "test": test}
    for split, per_source in (("reward", 64), ("dev", 128), ("train", 9473)):
        for source, rows in pools.items():
            splits[split].extend(deterministic_take(rows, per_source, f"{split}:{source}", used))
    entries = {}
    for split, rows in splits.items():
        path = assets / f"{split}.json"
        immutable_json(path, rows)
        entries[split] = {
            "path": str(path), "file": path.name, "count": len(rows),
            "sha256": sha256(path),
            "ids_sha256": hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest(),
            "by_source": dict(sorted(Counter(row["data_source"] for row in rows).items())),
        }
    manifest = {
        "schema_version": 2, "benchmark": "searchqa", "seed": 42,
        "row_id": "SHA256(source_row_key), with source_row_key='searchqa:<source>:<source_split>:<source_index>:<source_id>'. All rows use identical field names/types.",
        "whale_revision": WHALE_REVISION, "sources": provenance, "splits": entries,
        "metric": "normalized-answer-alias-exact-match-adaptation",
        "metric_note": "Not WHALE paper accuracy: the paid gpt-5.4-mini judge is replaced by fixed normalized alias EM.",
        "reconstruction": {
            "paper_intermediates_available": False,
            "train": "18946 deterministically sampled public train rows: 9473 NQ + 9473 HotpotQA; not the unavailable WHALE filtered training identities.",
            "reward": "128 disjoint public train rows: 64 NQ + 64 HotpotQA.",
            "dev": "256 further disjoint public train rows: 128 NQ + 128 HotpotQA; shared immutable original/SFT selector, not report-test.",
            "test": "100 per benchmark using public test when present, otherwise public dev; 700 total.",
            "selection": "Ascending SHA256('42:<role>:<source>:<source_row_key>'), ties by source_row_key; test first, reward, dev, train. SHA256 row ids do not alter selected rows/order.",
            "deduplication": "Global NFKC/casefold/alphanumeric question key. All official eval question keys excluded from train/reward/dev. Selected splits have no repeated question keys or ids.",
            "aliases": "All FlashRAG golden_answers preserved without generated aliases; answer is first-alias compatibility field.",
        },
        "fixed_retrieval": {
            "corpus_repository": "PeterJinGo/wiki-18-corpus",
            "corpus_revision": "69c1c00ffe7c5554c68d8548355cb22e46aabc51",
            "corpus_sha256": "7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db",
            "engine": "tantivy==0.25.1 / full-corpus BM25",
            "adaptation": "Same public Search-R1 wiki18 corpus; CPU lexical BM25 instead of e5-base-v2 dense FAISS. No answer-conditioned documents, per-row corpora, random results, or network search fallback.",
            "top_k_ceiling": 5,
        },
    }
    immutable_json(assets / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=ASSETS)
    args = parser.parse_args()
    manifest = prepare(args.assets)
    print(json.dumps({"manifest": str(args.assets / "manifest.json"),
                      "splits": manifest["splits"]}, indent=2))


if __name__ == "__main__":
    main()
