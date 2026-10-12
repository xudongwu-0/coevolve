"""Download public immutable math assets, verifying published file hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

from .common import ROOT, write_json

ASSETS = [
    ("models", "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554",
     "Qwen3-4B-Instruct-2507"),
    ("datasets", "BytedTsinghua-SIA/DAPO-Math-17k", "65877096c24ffa7abc4e4fa5edb95cf3413a5674",
     "dapo_source"),
    ("datasets", "Maxwell-Jia/AIME_2024", "8d88b2876a82a080e2f172cc9b25d0d9d2cb4792",
     "aime2024_source"),
    ("datasets", "yentinglin/aime_2025", "6f71d77b0b89b9dabe07ab466c51df33f514df7f",
     "aime2025_source"),
]


def main():
    for kind, repo, revision, name in ASSETS:
        directory = ROOT / "assets" / name
        directory.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(
            f"https://huggingface.co/api/{kind}/{repo}/revision/{revision}?blobs=true",
            timeout=90,
        ) as response:
            metadata = json.load(response)
        if metadata["sha"] != revision:
            raise RuntimeError("Asset revision mismatch")
        files = []
        for entry in metadata["siblings"]:
            filename = entry["rfilename"]
            if kind == "models":
                keep = filename.endswith((".json", ".safetensors", ".txt", ".jinja")) or filename == "LICENSE"
            else:
                keep = filename.endswith(".parquet") and not filename.startswith(("part1/", "part2/"))
            if not keep:
                continue
            path = directory / Path(filename).name
            url = f"https://huggingface.co/{'datasets/' if kind == 'datasets' else ''}{repo}/resolve/{revision}/{filename}?download=true"
            expected = entry.get("lfs", {}).get("sha256")
            digest = hashlib.sha256()
            if path.exists():
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                        digest.update(block)
            else:
                temporary = path.with_suffix(path.suffix + ".partial")
                with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as stream:
                    for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                        digest.update(block)
                        stream.write(block)
                temporary.replace(path)
            if path.stat().st_size != entry["size"] or (expected and digest.hexdigest() != expected):
                raise RuntimeError(f"Asset digest mismatch: {path}")
            files.append({"path": path.name, "source_path": filename, "url": url,
                          "bytes": path.stat().st_size, "sha256": digest.hexdigest()})
            print(name, path.name, "verified", flush=True)
        write_json(directory / "provenance.json", {
            "repository": repo, "revision": revision, "files": files,
        })


if __name__ == "__main__":
    main()
