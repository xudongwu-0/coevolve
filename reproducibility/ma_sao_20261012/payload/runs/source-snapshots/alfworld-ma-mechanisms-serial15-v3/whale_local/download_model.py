"""Download the pinned public integration policy without modifying retained W0."""
from pathlib import Path
import json
import os

# HTTP download works with the local proxy; Xet stalled in this environment.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
from huggingface_hub import HfApi, snapshot_download

MODEL = "Qwen/Qwen3.5-2B"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
OUT = Path(__file__).resolve().parents[1] / "runs/whale_setup"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(MODEL, revision=REVISION, files_metadata=True, token=False)
    manifest = {"model": MODEL, "revision": info.sha,
                "files": [{"path": f.rfilename, "size": f.size} for f in info.siblings]}
    assert info.sha == REVISION
    (OUT / "base_model_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    path = snapshot_download(MODEL, revision=REVISION, token=False,
        allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "LICENSE"])
    (OUT / "base_model_path.txt").write_text(path + "\n")
    print(path)


if __name__ == "__main__":
    main()
