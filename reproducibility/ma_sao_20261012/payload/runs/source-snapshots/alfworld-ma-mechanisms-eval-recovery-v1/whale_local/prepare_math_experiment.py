"""Freeze MATH number-theory integer tasks without filtering on policy rewards."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from math_reward import native
from pipeline import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(exist_ok=False, parents=True)
    pools, all_train = {}, set()
    for split in ["train", "test"]:
        raw = pd.read_parquet(source / (split + ".parquet")).to_dict("records")
        pool = {}
        for row in raw:
            key = hashlib.sha256(row["problem"].strip().encode()).hexdigest()
            if split == "train":
                all_train.add(key)
            if split == "test" and key in all_train:
                continue
            boxed = native.last_boxed_only_string(row["solution"])
            answer = native.remove_boxed(boxed).strip() if boxed else ""
            if row["level"] not in {"Level 2", "Level 3", "Level 4"}:
                continue
            if not re.fullmatch(r"-?(0|[1-9]\d*)", answer) or str(int(answer)) != answer:
                continue
            pool.setdefault(key, (row, answer))
        pools[split] = pool
    sizes = {"train": 128, "mh_val": 48, "confirmation": 128, "test": 128}
    assert len(pools["train"]) >= 304 and len(pools["test"]) >= 128
    keys = sorted(pools["train"])
    selected = {"train": keys[:128], "mh_val": keys[128:176], "confirmation": keys[176:304],
                "test": sorted(pools["test"])[:128]}
    assert len(set.union(*(set(k) for k in selected.values()))) == sum(sizes.values())
    manifest = {"source": "EleutherAI/hendrycks_math", "config": "number_theory",
        "source_revision": json.loads((source / "source_revision.json").read_text())["revision"],
        "source_files": {s: {"path": str(source / (s + ".parquet")), "sha256": sha(source / (s + ".parquet"))}
                         for s in ["train", "test"]},
        "filter": "Levels 2, 3, 4; canonical integer boxed answer; no policy-reward filtering",
        "selection": "question SHA256 order; train/MH/confirmation from official train; final test from official test, excluding all original train question hashes",
        "eligible_counts": {s: len(p) for s, p in pools.items()},
        "verifier": "unchanged WHALE math_dapo strict boxed scorer, converted to binary by native ReTool rule",
        "sizes": sizes, "splits": {}, "question_disjoint": True}
    for name, chosen in selected.items():
        pool = pools["test" if name == "test" else "train"]
        rows = []
        for i, key in enumerate(chosen):
            row, answer = pool[key]
            rows.append({"data_source": "math_number_theory_integer",
                "extra_info": {"question": row["problem"], "task_id": key, "index": i, "level": row["level"]},
                "reward_model": {"ground_truth": {"target": [answer]}}})
        path = out / (name + ".parquet")
        pd.DataFrame(rows).to_parquet(path, index=False)
        manifest["splits"][name] = {"path": str(path), "sha256": sha(path), "count": len(rows)}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"manifest": str(out / "manifest.json"), "sizes": sizes,
                      "eligible_counts": manifest["eligible_counts"]}))


if __name__ == "__main__":
    main()
