"""Replace a diagnostic confirmation split using unused pinned-source questions."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline import sha, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--reason", default="FP32 master-parameter correction decided before any diagnostic confirmation scores were inspected")
    args = parser.parse_args()
    parent, out = args.parent.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((parent / "manifest.json").read_text())
    excluded = {r["task_id"] for e in manifest["splits"].values()
                for r in pd.read_parquet(e["path"])["extra_info"]}
    exclusion_records = []
    for file in args.exclude_manifest:
        extra = json.loads(file.read_text())
        for entry in extra["splits"].values():
            assert sha(entry["path"]) == entry["sha256"]
            excluded.update(r["task_id"] for r in pd.read_parquet(entry["path"])["extra_info"])
        exclusion_records.append({"path": str(file.resolve()), "sha256": sha(file)})
    candidates, pages = {}, []
    source_files = sorted(parent.glob("source_[0-9]*.json"))
    if not source_files:
        source_files = [Path(p["path"]) for p in manifest["source_pages"]]
        assert all(sha(p["path"]) == p["sha256"] for p in manifest["source_pages"])
    for file in source_files:
        pages.append({"path": str(file), "sha256": sha(file)})
        for entry in json.loads(file.read_text())["rows"]:
            raw = entry["row"]
            key = hashlib.sha256(raw["question"].strip().encode()).hexdigest()
            if raw["type"] == "bridge" and raw["answer"].strip().lower() not in {"yes", "no"} and key not in excluded:
                candidates.setdefault(key, raw)
    assert len(candidates) >= 128
    chosen = sorted(candidates)[:128]
    assert not set(chosen).intersection(excluded)
    rows = [{"data_source": "hotpotqa_distractor_bridge",
        "extra_info": {"question": candidates[key]["question"], "index": index, "task_id": key},
        "reward_model": {"ground_truth": {"target": [candidates[key]["answer"]]}}}
        for index, key in enumerate(chosen)]
    path = out / "confirmation.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    old_confirm = {r["task_id"] for r in pd.read_parquet(manifest["splits"]["confirmation"]["path"])["extra_info"]}
    corpora = {key: value for key, value in json.loads(Path(manifest["corpora"]["path"]).read_text()).items()
               if key not in old_confirm}
    for key in chosen:
        raw = candidates[key]
        corpora[key] = [{"title": title, "text": "".join(sentences)}
            for title, sentences in zip(raw["context"]["title"], raw["context"]["sentences"])]
    write_json(out / "corpora.json", corpora)
    manifest["splits"]["confirmation"] = {"path": str(path), "sha256": sha(path), "count": 128}
    manifest["corpora"] = {"path": str(out / "corpora.json"), "sha256": sha(out / "corpora.json")}
    manifest["source_pages"] = pages
    manifest["confirmation_refresh"] = {"parent_manifest": str(parent / "manifest.json"),
        "parent_sha256": sha(parent / "manifest.json"), "unused_eligible_pool": len(candidates),
        "additional_excluded_manifests": exclusion_records, "excluded_questions": len(excluded),
        "selection": "first 128 by question SHA256 among unused eligible pinned-source rows",
        "reason": args.reason,
        "all_previous_questions_excluded_from_new_confirmation": True,
        "train_mh_and_reserved_gate_test_unchanged": True}
    write_json(out / "manifest.json", manifest)
    print(out / "manifest.json")


if __name__ == "__main__":
    main()
