"""Freeze a fresh, position-disjoint pilot dataset without looking at policy rewards."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import argparse
import hashlib
import json
import time
from urllib.error import URLError

import pandas as pd
from autoharness_chess_puzzle.grpo_dataset import convert_lichess_row, _split_for_key

SOURCE = "Lichess/chess-puzzles"
REVISION = "479ea9bc9f681385f5adb23fa27a96c2dc8ae599"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--exclude-manifest", required=True, type=Path, action="append")
    parser.add_argument("--train", type=int, default=128)
    parser.add_argument("--mh-val", type=int, default=64)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--min-rating", type=int, default=0)
    parser.add_argument("--max-rating", type=int, default=1000)
    parser.add_argument("--min-plies", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.min_rating <= args.max_rating or not 1 <= args.min_plies <= args.max_plies <= 9:
        parser.error("Require 0 <= min-rating <= max-rating and 1 <= min-plies <= max-plies <= 9")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sizes = {"train": args.train, "mh_val": args.mh_val, "test": args.test}
    assert min(sizes.values()) > 0
    excluded = set()
    exclusions = []
    for manifest in args.exclude_manifest:
        prior = json.loads(manifest.read_text())
        for entry in prior["splits"].values():
            assert sha(entry["path"]) == entry["sha256"]
            excluded.update(json.loads(x)["position_key"] for x in pd.read_parquet(entry["path"])["extra_info"])
        exclusions.append({"path": str(manifest.resolve()), "sha256": sha(manifest)})
    specification = {"source": SOURCE, "source_revision": REVISION, "sizes": sizes,
                     "max_rating": args.max_rating, "solution_plies": [args.min_plies, args.max_plies], "excluded_manifests": exclusions}
    if args.min_rating:
        specification["min_rating"] = args.min_rating
    spec_path = out / "specification.json"
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == specification, "Dataset specification changed"
    else:
        spec_path.write_text(json.dumps(specification, indent=2) + "\n")
    if (out / "manifest.json").exists():
        ready = json.loads((out / "manifest.json").read_text())
        assert all(sha(s["path"]) == s["sha256"] for s in ready["splits"].values())
        print(json.dumps({"manifest": str(out / "manifest.json"), "counts": sizes}))
        return

    def fetch(offset):
        path = out / f"source_{offset:05d}.json"
        meta = path.with_suffix(".meta.json")
        if path.exists() and meta.exists():
            info = json.loads(meta.read_text())
            assert info["revision"] == REVISION and info["sha256"] == sha(path)
        else:
            url = "https://datasets-server.huggingface.co/rows?" + urlencode(
                dict(dataset=SOURCE, config="default", split="train", offset=offset, length=100))
            for attempt in range(5):
                try:
                    with urlopen(url, timeout=45) as response:
                        revision = response.headers.get("x-revision")
                        assert revision == REVISION, f"Source changed: {revision}; refuse mixed revisions"
                        content = response.read()
                    break
                except (URLError, TimeoutError, ConnectionError):
                    if attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            path.write_bytes(content)
            meta.write_text(json.dumps({"revision": revision, "sha256": sha(path), "url": url}) + "\n")
        return path, json.loads(path.read_text())["rows"]

    seen = set(excluded)
    splits = {k: [] for k in sizes}
    scanned, sources = 0, []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for batch_start in range(0, 20000, 400):
            for path, entries in pool.map(fetch, range(batch_start, batch_start + 400, 100)):
                sources.append({"path": str(path), "sha256": sha(path)})
                for entry in entries:
                    scanned += 1
                    raw = entry["row"]
                    if not (args.min_rating <= int(raw["Rating"]) <= args.max_rating
                            and args.min_plies <= len(raw["Moves"].split()) - 1 <= args.max_plies):
                        continue
                    row = convert_lichess_row(raw)
                    if row is None:
                        continue
                    key = row["extra_info"]["position_key"]
                    if key in seen:
                        continue
                    seen.add(key)
                    split = _split_for_key(key, sizes)
                    if len(splits[split]) < sizes[split]:
                        row["extra_info"] = json.dumps(row["extra_info"])
                        splits[split].append(row)
                print(f"scanned={scanned} counts={ {k: len(v) for k, v in splits.items()} }", flush=True)
                if all(len(splits[k]) == sizes[k] for k in sizes):
                    break
            if all(len(splits[k]) == sizes[k] for k in sizes):
                break
    assert all(len(splits[k]) == sizes[k] for k in sizes), "Insufficient source rows"
    manifest = specification | {"purpose": "small local efficacy pilot; not the paper benchmark",
        "source_interface": "HF dataset viewer; every page revision and digest verified",
        "scanned": scanned, "source_pages": sources, "excluded_positions": len(excluded),
        "partition": "upstream stable position hash; fixed sizes; no reward-based filtering",
        "splits": {}}
    keys = {}
    for name, rows in splits.items():
        path = out / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        keys[name] = {json.loads(r["extra_info"])["position_key"] for r in rows}
        assert len(keys[name]) == sizes[name] and not keys[name] & excluded
        manifest["splits"][name] = {"path": str(path), "count": len(rows), "sha256": sha(path),
            "puzzle_ids": [json.loads(r["extra_info"])["puzzle_id"] for r in rows]}
    assert not any(keys[a] & keys[b] for a, b in [("train", "mh_val"), ("train", "test"), ("mh_val", "test")])
    manifest["position_disjoint"] = True
    manifest["disjoint_from_prior_runs"] = True
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"manifest": str(out / "manifest.json"), "counts": sizes, "excluded": len(excluded)}))


if __name__ == "__main__":
    main()
