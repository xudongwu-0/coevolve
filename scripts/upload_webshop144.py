"""Publish the two final WebShop policy exports, not the native training states."""
import argparse
import getpass
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, get_token


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/webshop_ma_sao_speed_20261118/executions/validity_cost_144_final_only"
FILES = ("model.safetensors", "config.json", "generation_config.json", "tokenizer.json",
         "tokenizer_config.json", "preprocessor_config.json", "processor_config.json", "chat_template.jinja")


def release(role):
    complete = json.loads((RUN / role / "COMPLETE.json").read_text())
    actor = Path(complete["checkpoints"]["actor"])
    if (complete["role"], complete["steps"], complete["status"]) != (role, 144, "completed"):
        raise ValueError(f"{role}: expected completed step 144")
    if not actor.resolve().is_relative_to((RUN / role).resolve()) or actor.name != "epoch0epochstep143globalstep143":
        raise ValueError(f"{role}: unexpected actor export")
    paths = {name: actor / name for name in FILES}
    paths["LICENSE"] = ROOT / "assets/Qwen3.5-9B/LICENSE"
    for path in paths.values():
        if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
            raise ValueError(f"Missing or invalid file: {path.name}")
    model_role = "engineer" if role == "editor" else "solver"
    repo = f"wxd02/WebShop-MA-SAO-9B-{model_role}-step144"
    card = f"""---
base_model: Qwen/Qwen3.5-9B
license: apache-2.0
library_name: transformers
---
# WebShop MA-SAO {model_role} (step 144)

Inference Actor fine-tuned from Qwen3.5-9B. No optimizer or Critic is included.
The Solver with the **default** harness achieved 37/64 strict successes (57.81%)
on the local WebShop 1k preview holdout, not the original 500-task benchmark.
The 64 development tasks used for harness selection are not part of this score.
This is not a standalone Engineer score.
"""
    return repo, paths, card.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="offline preflight; never authenticates or uploads")
    args = parser.parse_args()
    exports = [release(role) for role in ("editor", "solver")]
    for repo, paths, _ in exports:
        print(f"{repo}: {paths['model.safetensors'].stat().st_size:,} bytes of weights", flush=True)
    if args.check:
        return
    api = HfApi(endpoint="https://huggingface.co",
                token=get_token() or getpass.getpass("Hugging Face write token (hidden): ").strip())
    if api.whoami()["name"] != "wxd02":
        raise SystemExit("Expected Hugging Face account wxd02; nothing uploaded.")
    for repo, paths, card in exports:
        with paths["model.safetensors"].open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
        info = api.model_info(repo, files_metadata=True)
        if info.private:
            raise ValueError(f"{repo} already exists as private; not changing its visibility")
        existing = {s.rfilename: s for s in info.siblings or []}
        unexpected = set(existing) - {".gitattributes", "README.md", *paths}
        weights = existing.get("model.safetensors")
        if unexpected or (weights and (not weights.lfs or weights.lfs.sha256 != digest)):
            raise ValueError(f"{repo} contains unrelated files or different weights; refusing to overwrite")
        ops = [CommitOperationAdd(name, path) for name, path in paths.items()]
        ops.append(CommitOperationAdd("README.md", card))
        api.create_commit(repo, operations=ops, commit_message="Publish step-144 inference actor",
                          parent_commit=info.sha)
        final = api.model_info(repo, files_metadata=True)
        files = {s.rfilename: s for s in final.siblings}
        if (final.private or not files["model.safetensors"].lfs
                or files["model.safetensors"].lfs.sha256 != digest
                or any(files[name].size != path.stat().st_size for name, path in paths.items())
                or files["README.md"].size != len(card)):
            raise RuntimeError(f"Remote verification failed for {repo}")
        print(f"Verified https://huggingface.co/{repo}/tree/{final.sha}", flush=True)


if __name__ == "__main__":
    main()
