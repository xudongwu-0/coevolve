"""Freeze an extractive HotpotQA bridge subset with official distractor contexts."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen
import argparse
import hashlib
import json
import time

import pandas as pd

from pipeline import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"train": 64, "mh_val": 32, "confirmation": 128, "test": 128}
    spec = {"source": "hotpotqa/hotpot_qa", "config": "distractor", "source_split": "train",
            "filter": "type=bridge and answer not yes/no; no policy reward filtering",
            "sizes": sizes, "retrieval": "BM25 over the ten official context paragraphs for each task",
            "verifier": "WHALE SearchQA default strict normalized exact match; not the paper's optional LLM judge"}
    write_json(out / "specification.json", spec)
    revision = json.loads((out / "source_revision.json").read_text())["revision"]
    def fetch(offset):
        path = out / f"source_{offset:05d}.json"
        if not path.exists():
            url = f"https://datasets-server.huggingface.co/rows?dataset=hotpotqa/hotpot_qa&config=distractor&split=train&offset={offset}&length=100"
            for attempt in range(5):
                try:
                    with urlopen(url, timeout=45) as response:
                        assert response.headers.get("x-revision") == revision
                        content = response.read()
                    break
                except OSError:
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            path.write_bytes(content)
        return path, json.loads(path.read_text())["rows"]
    splits = {name: [] for name in sizes}
    corpora, seen, pages = {}, set(), []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, 5000, 400):
            for path, entries in pool.map(fetch, range(start, start + 400, 100)):
                pages.append({"path": str(path), "sha256": sha(path)})
                for entry in entries:
                    raw = entry["row"]
                    key = hashlib.sha256(raw["question"].strip().encode()).hexdigest()
                    if raw["type"] != "bridge" or raw["answer"].strip().lower() in {"yes", "no"} or key in seen:
                        continue
                    seen.add(key)
                    bucket = int(key[:16], 16) % sum(sizes.values())
                    for split, size in sizes.items():
                        if bucket < size:
                            break
                        bucket -= size
                    if len(splits[split]) >= sizes[split]:
                        continue
                    index = len(splits[split])
                    splits[split].append({"data_source": "hotpotqa_distractor_bridge",
                        "extra_info": {"question": raw["question"], "index": index, "task_id": key},
                        "reward_model": {"ground_truth": {"target": [raw["answer"]]}}})
                    corpora[key] = [{"title": title, "text": "".join(sentences)}
                                   for title, sentences in zip(raw["context"]["title"], raw["context"]["sentences"])]
                print({name: len(rows) for name, rows in splits.items()}, flush=True)
                if all(len(splits[k]) == sizes[k] for k in sizes):
                    break
            if all(len(splits[k]) == sizes[k] for k in sizes):
                break
    assert all(len(splits[k]) == sizes[k] for k in sizes)
    manifest = spec | {"source_revision": revision, "source_pages": pages, "splits": {}}
    all_keys = []
    for name, rows in splits.items():
        path = out / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][name] = {"path": str(path), "sha256": sha(path), "count": len(rows)}
        all_keys.extend(r["extra_info"]["task_id"] for r in rows)
    assert len(all_keys) == len(set(all_keys))
    write_json(out / "corpora.json", corpora)
    manifest["corpora"] = {"path": str(out / "corpora.json"), "sha256": sha(out / "corpora.json")}
    manifest["question_disjoint"] = True
    write_json(out / "manifest.json", manifest)
    print(out / "manifest.json", flush=True)


if __name__ == "__main__":
    main()
