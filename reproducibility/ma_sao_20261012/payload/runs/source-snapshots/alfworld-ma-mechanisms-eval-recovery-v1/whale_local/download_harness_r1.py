"""Download and verify the released, revision-pinned Harness-R1 Engineer."""
from concurrent.futures import ThreadPoolExecutor
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download

from sao_bridge.common import file_sha, write_json

MODEL = "ShaoShuai0605/Harness-R1"
REVISION = "f6a915953f8206f3f5c7ef0b40240e1f4d2b6676"
SUBFOLDER = "harness-r1"
OUT = Path(__file__).resolve().parents[1] / "assets/Harness-R1"


def verify_download(path: Path, item) -> dict:
    if path.stat().st_size != item.size:
        raise ValueError(f"Downloaded size mismatch: {item.rfilename}")
    digest = file_sha(path)
    if item.lfs:
        expected = item.lfs.sha256
        actual = digest
        kind = "sha256"
    else:
        expected = item.blob_id
        content = path.read_bytes()
        actual = hashlib.sha1(
            f"blob {len(content)}\0".encode() + content
        ).hexdigest()
        kind = "git_blob_sha1"
    if not expected or actual != expected:
        raise ValueError(f"Downloaded digest mismatch: {item.rfilename}")
    return {
        "path": path.name, "size": item.size, "sha256": digest,
        "expected_hash": expected, "hash_kind": kind, "verified": True,
        "source": f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{item.rfilename}",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subfolder", choices=(
        "harness-r1", "cold-start-sft", "target-agent-sft", "agent-sft-harness-r1",
    ), default=SUBFOLDER)
    args = parser.parse_args()
    subfolder = args.subfolder
    status_path = OUT / (
        "DOWNLOAD_STATUS.json" if subfolder == SUBFOLDER
        else f"{subfolder}_DOWNLOAD_STATUS.json"
    )
    started = datetime.now(timezone.utc).isoformat()
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, token=False)
    if info.sha != REVISION:
        raise ValueError("Released Engineer revision changed")
    items = [item for item in info.siblings
             if item.rfilename.startswith(subfolder + "/")]
    if not items or not any(item.rfilename.endswith("/model.safetensors") for item in items):
        raise ValueError("Released Engineer checkpoint inventory is incomplete")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(status_path, {
        "status": "downloading", "model": MODEL, "revision": REVISION,
        "subfolder": subfolder, "started_at": started,
    })

    def download(item):
        print(f"Downloading {item.rfilename}: {item.size} bytes", flush=True)
        downloaded = Path(hf_hub_download(
            MODEL, item.rfilename, revision=REVISION, token=False,
            local_dir=OUT,
        ))
        receipt = verify_download(downloaded, item)
        print(f"Verified {item.rfilename}: {receipt['sha256']}", flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(download, items))
    model = OUT / subfolder
    write_json(model / "provenance.json", {
        "model": MODEL, "revision": REVISION, "subfolder": subfolder,
        "authentication": "explicit token=False; implicit token disabled",
        "started_at": started,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "files": sorted(files, key=lambda item: item["path"]),
    })
    write_json(status_path, {
        "status": "verified_complete", "model": MODEL, "revision": REVISION,
        "subfolder": subfolder,
        "model_path": str(model), "receipt_sha256": file_sha(model / "provenance.json"),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    })


if __name__ == "__main__":
    main()
