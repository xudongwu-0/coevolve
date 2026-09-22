"""Build a small, fixed Lichess integration split before inspecting rewards."""
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen
import hashlib
import json
import pandas as pd
from autoharness_chess_puzzle.grpo_dataset import convert_lichess_row, _split_for_key

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/whale_setup/data"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "manifest.json").exists():
        print((OUT / "manifest.json").read_text())
        return
    sizes = {"train": 16, "mh_val": 8, "test": 8}
    splits = {k: [] for k in sizes}
    seen = set()
    scanned = 0
    for offset in range(0, 10000, 100):
        path = OUT / f"source_{offset:05d}.json"
        if not path.exists():
            query = urlencode(dict(dataset="Lichess/chess-puzzles", config="default",
                                   split="train", offset=offset, length=100))
            with urlopen("https://datasets-server.huggingface.co/rows?" + query, timeout=90) as response:
                path.write_bytes(response.read())
        for entry in json.loads(path.read_text())["rows"]:
            raw = entry["row"]
            scanned += 1
            # Difficulty/length constraints are fixed, never based on policy success.
            if not (int(raw["Rating"]) <= 1000 and 1 <= len(raw["Moves"].split()) - 1 <= 3):
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
                # JSON avoids ndarray/string differences on parquet round trips.
                row["extra_info"] = json.dumps(row["extra_info"])
                splits[split].append(row)
        print(f"scanned={scanned} counts={ {k:len(v) for k,v in splits.items()} }", flush=True)
        if all(len(splits[k]) == sizes[k] for k in sizes):
            break
    assert all(len(splits[k]) == sizes[k] for k in sizes), "Insufficient source rows"
    manifest = {"source": "Lichess/chess-puzzles", "source_interface": "HF dataset viewer rows",
                "purpose": "integration, not paper benchmark", "max_rating": 1000,
                "solution_plies": [1, 3], "scanned": scanned,
                "partition": "upstream stable position hash; no reward-based filtering", "splits": {}}
    keys = {}
    for name, rows in splits.items():
        path = OUT / f"{name}.parquet"
        pd.DataFrame(rows).to_parquet(path, index=False)
        keys[name] = {json.loads(r["extra_info"])["position_key"] for r in rows}
        manifest["splits"][name] = {"path": str(path), "count": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "puzzle_ids": [json.loads(r["extra_info"])["puzzle_id"] for r in rows]}
    assert not (keys["train"] & keys["mh_val"] or keys["train"] & keys["test"] or keys["mh_val"] & keys["test"])
    manifest["position_disjoint"] = True
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
