#!/usr/bin/env python3
"""Prepare the paused 2026-10-12 evidence release; never submit GPU work or upload."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

from export_ma_handoff import (
    ROOT, export_git_payload, git_files, read_json, save_json, scan_and_hash, sha256, walk_files,
)

STUDY = ROOT / "runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2"
PRIOR = ROOT / "runs/ma_sao_handoff_20261010T081628Z"
CONTROL = STUDY / "serial15_20261012_self_fix01"
ATTEMPT = STUDY / "without_peer/attempts/attempt-0003"
TENSORS = {".pt", ".bin", ".safetensors", ".ckpt", ".distcp"}
HOST_AUDITS = {
    "gpu_process_samples.jsonl", "gpu_telemetry.json", "NO_H200_PAUSE_VERIFIED.json",
    "PAUSE_FINAL_CHECK.json", "FILLERS_PAUSED_VERIFIED.json", "PREREQUISITES_VERIFIED.json",
    "BASELINE_RELEASE.json", "NATIVE_GATE.json", "ROLLING_NATIVE_GATE.json",
    "RELEASED.json", "DELIVERY_PAUSE_VERIFIED.json",
}


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "--no-pager", "-C", str(repo), *args])


def areal_patch(output: Path) -> dict:
    repo = ROOT / "third_party/sao/vendor/AReaL"
    patch = bytearray(git(repo, "diff", "--binary", "HEAD", "--"))
    for raw in git(repo, "ls-files", "--others", "--exclude-standard", "-z").split(b"\0"):
        if not raw:
            continue
        name = raw.decode()
        scan_and_hash(repo / name)
        result = subprocess.run(
            ["git", "--no-pager", "-C", str(repo), "diff", "--no-index", "--binary", "--", "/dev/null", name],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode != 1:
            raise RuntimeError(f"Cannot capture untracked AReaL source: {name}; {result.stderr.decode()}")
        patch.extend(result.stdout)
    target = output / "upstream/AReaL-working-tree.patch"
    target.parent.mkdir(parents=True)
    with target.open("xb") as stream:
        stream.write(patch)
    scan_and_hash(target)
    files = []
    for path in git_files(repo):
        digest, size = scan_and_hash(path)
        files.append({"path": path.relative_to(repo).as_posix(), "bytes": size, "sha256": digest})
    provenance = {
        "base_revision": git(repo, "rev-parse", "HEAD").decode().strip(),
        "upstream": "https://github.com/inclusionAI/AReaL.git",
        "patch": target.relative_to(output).as_posix(), "patch_sha256": sha256(target),
        "apply_to": "Pristine pinned revision, instead of applying the component patches separately.",
        "files": files,
    }
    save_json(output / "upstream/AReaL.json", provenance)
    license_file = repo / "LICENSE"
    scan_and_hash(license_file)
    (output / "upstream/AReaL-LICENSE").write_bytes(license_file.read_bytes())
    return provenance


def model_catalogue() -> dict:
    old = read_json(PRIOR / "CHECKPOINTS.json")
    checkpoints = []
    for original in old["checkpoints"]:
        previous = {item["name"]: item for item in original["files"]}
        current = {key: value for key, value in original.items() if key not in ("files", "bytes")}
        if current["id"] == "without_peer-native-0024":
            pointer = read_json(STUDY / "without_peer/recovery/LATEST.json")
            if pointer["completed_rounds"] != 72:
                raise ValueError("The declared paused native72 state changed")
            current.update(
                id="without_peer-native-0072",
                source=str(STUDY / "without_peer/recovery" / pointer["generation"]),
                purpose="Resume the interrupted without_peer arm at72, not the76 recorded metric rows.",
            )
            previous = {}
        source = Path(current["source"])
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"Model catalogue requires a real existing directory: {source}")
        files = []
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Model catalogue cannot follow a symlink: {path}")
            if not path.is_file():
                continue
            name, size = path.relative_to(source).as_posix(), path.stat().st_size
            entry = {"path": name, "bytes": size}
            old_entry = previous.get(name, {})
            if size <= 2**20:
                entry.update(sha256=scan_and_hash(path)[0], hash_scope="rehashed_small_file")
            elif old_entry.get("bytes") == size and (old_entry.get("sha256") or old_entry.get("prior_sha256")):
                entry.update(
                    prior_sha256=old_entry.get("sha256") or old_entry["prior_sha256"],
                    hash_scope="prior_full_hash_not_recomputed_for_this_release",
                )
            else:
                entry["hash_scope"] = "size_only_requires_full_transfer_verification"
            files.append(entry)
        checkpoints.append({
            **current, "files": files, "bytes": sum(item["bytes"] for item in files),
            "proposed_remote_prefix": f"releases/20261012/checkpoints/{current['id']}",
        })
    return {
        "schema": "ma-sao-model-artifacts-v1", "remote_uploaded": False, "hf_repo_id": None,
        "hf_repo_type": "dataset", "hf_revision": None, "proposed_repo_name": "coevolve",
        "required_visibility": "private", "wandb_entity": None, "wandb_project": "coevolve",
        "priority": ["full-144-solver-actor", "full-144-editor-actor",
                     "without_peer-native-0072", "full-native-0144"],
        "resume_also_requires_initial_models": True,
        "checkpoints": checkpoints, "logical_bytes": sum(item["bytes"] for item in checkpoints),
        "scope": "Local inventory, not remote availability or full large-tensor hash verification.",
    }


def prepare(output: Path) -> dict:
    cancellation = read_json(CONTROL / "CANCELLED.json")
    if cancellation["h200_training_allowed"] is not False or cancellation["status"] != "cancelled":
        raise ValueError("Publication preparation requires the current experiment cancellation")
    prior = read_json(PRIOR / "PLAN.json")
    files, exclusions = set(), []

    def add(path: Path, *, required: bool = True) -> None:
        path.relative_to(ROOT)
        if (
            path.name in HOST_AUDITS or path.name == "tokenizer.json"
            or path.suffix.lower() in TENSORS or "admissions" in path.parts
        ):
            exclusions.append({"path": str(path), "reason": "host_process_privacy_or_separate_model_storage"})
        elif path.is_file() and not path.is_symlink():
            files.add(path)
        elif required:
            raise FileNotFoundError(f"Required evidence is unavailable: {path}")
        else:
            exclusions.append({"path": str(path), "reason": "historical_optional_path_unavailable"})

    for entry in read_json(PRIOR / "GIT_KEY_EVIDENCE.json")["files"]:
        add(ROOT / entry["path"])
    for name in prior["required_evidence_files"]:
        add(Path(name), required=False)
    for name in prior["source_extra_files"]:
        add(Path(name))
    for path in STUDY.glob("*.json"):
        add(path)
    for arm in ("full", "without_peer", "without_context", "without_value_norm", "always_joint"):
        for role in ("editor", "solver"):
            add(STUDY / arm / f"{role}_spec.json")
    for path in CONTROL.iterdir():
        if path.is_file() and path.suffix in {".json", ".log"}:
            add(path)
    for path in ATTEMPT.glob("*.json"):
        add(path)
    for role in ("editor", "solver"):
        directory = ATTEMPT / "without_peer" / role
        for path in directory.glob("*.json"):
            add(path)
        for name in ("metrics.jsonl", "RECOVERY_LOADED.json", "RECOVERY_SAVED.json",
                     "trainer.yaml", "capacity_audit.json", "rollout_probability_contract.json"):
            add(directory / "optimizer" / name)
    for arm in ("full", "without_peer"):
        recovery = STUDY / arm / "recovery"
        add(recovery / "LATEST.json")
        generation = recovery / read_json(recovery / "LATEST.json")["generation"]
        for path in generation.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                add(path)
        for folder in ("history", "retention"):
            for path in (recovery / folder).glob("*.json"):
                add(path)
    protocol = read_json(STUDY / "PROTOCOL.json")
    add(Path(protocol["core200_protocol"]))
    add(Path(protocol["no_patch"]))
    add(ROOT / "assets/alfworld_harness_r1/data/test_set.json")
    add(ROOT / "assets/alfworld_harness_r1/asset_manifest.json")
    add(ROOT / "assets/alfworld_harness_r1/environment_freeze.txt")
    for endpoint in read_json(STUDY / "full/RESULT.json")["endpoints"].values():
        for matrix in (Path(endpoint["main_matrix"]), Path(endpoint["selection"]["selection_matrix"])):
            for name in ("config.json", "report.json", "COMPLETE.json"):
                add(matrix / name)
            for path in (matrix / "reports").glob("*.json"):
                add(path)
        selection_config = read_json(Path(endpoint["selection"]["selection_matrix"]) / "config.json")
        for name in selection_config["harnesses"].values():
            add(Path(name))
    snapshots = [Path(name) for name in prior["source_snapshots"]]
    snapshots.extend(ROOT / "runs/source-snapshots" / name for name in (
        "alfworld-ma-mechanisms-idle2h-v1", "alfworld-ma-mechanisms-serial15-v3",
    ))
    for snapshot in snapshots:
        for path in walk_files(snapshot, exclusions):
            add(path)
    manifest = export_git_payload(output, sorted(files))
    save_json(output / "EXCLUSIONS.json", {"entries": exclusions})
    save_json(output / "STATE.json", {
        "schema": "ma-sao-progress-handoff-v1", "study": STUDY.relative_to(ROOT).as_posix(),
        "attempt": ATTEMPT.relative_to(ROOT).as_posix(),
        "pause": (CONTROL / "CANCELLED.json").relative_to(ROOT).as_posix(),
        "catalogue": "assets/alfworld_harness_r1/data/test_set.json",
        "source_snapshots": [path.relative_to(ROOT).as_posix() for path in snapshots],
        "expected": {
            "without_peer_metric_rounds": {"editor": 76, "solver": 76},
            "without_peer_native_rounds": 72,
            "full144_selected_successes": 186, "full144_weight_only_successes": 179,
        },
        "training_or_evaluation_submission_allowed": False,
        "full144_extension_requested_then_paused_without_implementation": True,
    })
    save_json(output / "MODEL_ARTIFACTS.json", model_catalogue())
    packages = sorted((distribution.metadata["Name"], distribution.version)
                      for distribution in importlib.metadata.distributions() if distribution.metadata["Name"])
    save_json(output / "ENVIRONMENT.json", {
        "python": sys.version, "platform": platform.platform(),
        "installed_packages": dict(packages),
        "training_overlay": {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "triton": "3.7.1"},
        "scope": "Observed environment, not a cross-platform solver lock or automatic install command.",
        "credential_environment_recorded": False,
    })
    repositories = []
    for name in ("WHALE", "third_party/sao", "third_party/harness-r1-upstream", "third_party/mappo-reference"):
        repo = ROOT / name
        if git(repo, "status", "--porcelain").strip():
            raise ValueError(f"An upstream submodule changed since the inspected provenance: {name}")
        repositories.append({"path": name, "revision": git(repo, "rev-parse", "HEAD").decode().strip()})
    save_json(output / "UPSTREAMS.json", {"submodules": repositories, "areal": areal_patch(output)})
    archives = []
    for name in ("assets", "evidence", "supplemental-logs"):
        metadata = read_json(PRIOR / f"{name}.manifest.json")
        archives.append({
            "archive": str(PRIOR / metadata["archive"]), "bytes": metadata["archive_bytes"],
            "sha256": metadata["archive_sha256"], "hash_scope": "previous_verified_archive_digest",
            "snapshot_date": "2026-10-10", "remote_uploaded": False,
            "remote_path": None, "remote_revision": None,
        })
    save_json(output / "REMOTE_ARTIFACTS.json", {
        "status": "not_uploaded", "archives": archives,
        "current_without_peer25_to76_full_trajectories_in_git": False,
        "current_trajectory_source": str(ATTEMPT / "without_peer/joint"),
        "current_state_and_metric_evidence_in_git": True,
        "github_repository": "https://github.com/xudongwu-0/coevolve",
        "hf_repo_id": None, "hf_revision": None,
        "do_not_assume_local_archive_paths_exist_on_a_new_machine": True,
    })
    return {"status": "prepared_local_only", "files": len(manifest["files"]),
            "bytes": manifest["uncompressed_bytes"], "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
