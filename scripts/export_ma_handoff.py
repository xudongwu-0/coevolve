#!/usr/bin/env python3
"""Create local-only source/evidence archives and non-copying checkpoint pins."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".py", ".yaml", ".yml", ".txt", ".sh", ".patch", ".toml", ".jinja"}
SKIP_DIRECTORIES = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
SECRET_NAME = re.compile(r"(^\.env(?:\.|$)|(?:^|[._-])(?:api[._-]?keys?|access[._-]?token|credentials|private[._-]?key)(?:[._-]|$)|\.(?:pem|key)$)", re.I)
SECRET_CONTENT = re.compile(rb"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,})")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def require_safe_name(path: Path) -> None:
    if any((part != ".env.example" and SECRET_NAME.search(part))
           or part in (".ssh", ".private_secrets") for part in path.parts):
        raise ValueError(f"Credential-like path is not an export artifact: {path}")


def scan_and_hash(path: Path) -> tuple[str, int]:
    require_safe_name(path)
    if path.name == ".env.example":
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            key, separator, value = line.partition("=")
            if separator and re.search(r"KEY|TOKEN|PASSWORD|CREDENTIAL|SECRET", key, flags=re.I):
                if value.strip().strip("\"'"):
                    raise ValueError(f"Credential field in an environment template must be empty: {path}")
    digest, count, overlap = hashlib.sha256(), 0, b""
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        for block in iter(lambda: stream.read(2**20), b""):
            if SECRET_CONTENT.search(overlap + block):
                raise ValueError(f"Credential-like content must be reviewed locally before export: {path}")
            overlap = block[-256:]
            digest.update(block)
            count += len(block)
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or count != before.st_size:
        raise ValueError(f"Source changed during archival: {path}")
    return digest.hexdigest(), count


def walk_files(root: Path, exclusions: list[dict], *, text_only: bool = False):
    def failed(error):
        exclusions.append({"path": str(error.filename), "reason": "unreadable_optional_subtree", "error": str(error)})

    for current, directories, files in os.walk(root, followlinks=False, onerror=failed):
        directory = Path(current)
        for name in list(directories):
            path = directory / name
            if name in SKIP_DIRECTORIES or path.is_symlink():
                directories.remove(name)
                exclusions.append({
                    "path": str(path), "reason": "symlink_not_followed" if path.is_symlink() else "cache_or_git_metadata",
                    **({"target": os.readlink(path)} if path.is_symlink() else {}),
                })
        for name in files:
            path = directory / name
            if path.is_symlink():
                exclusions.append({"path": str(path), "reason": "symlink_not_followed", "target": os.readlink(path)})
            elif not text_only or path.suffix in TEXT_SUFFIXES:
                yield path
            else:
                exclusions.append({"path": str(path), "reason": "non_text_payload_not_in_evidence_archive", "bytes": path.stat().st_size})


def git_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True, stdout=subprocess.PIPE,
    )
    files = []
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        path = root / os.fsdecode(raw)
        if path.is_file() and not path.is_symlink():
            files.append(path)
    return files


def create_archive(output: Path, name: str, files: list[Path], *, project: Path = ROOT) -> dict:
    target = output / f"{name}.tar.gz"
    if target.exists():
        raise FileExistsError(target)
    entries = []
    with tarfile.open(target, "w:gz", compresslevel=3) as archive:
        for index, path in enumerate(sorted(set(files))):
            relative = path.relative_to(project)
            if ".." in relative.parts or path.is_symlink():
                raise ValueError(f"Archive member is not a regular project file: {path}")
            digest, size = scan_and_hash(path)
            stat = path.stat()
            info = archive.gettarinfo(str(path), arcname=str(relative))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as stream:
                archive.addfile(info, stream)
            after = path.stat()
            if (stat.st_size, stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed while entering archive: {path}")
            entries.append({"path": str(relative), "bytes": size, "sha256": digest})
            if index and index % 1000 == 0:
                print(f"{name}: archived {index} files", flush=True)
    manifest = {
        "status": "created_local_only", "archive": target.name, "archive_sha256": sha256(target),
        "archive_bytes": target.stat().st_size, "uncompressed_bytes": sum(item["bytes"] for item in entries),
        "files": entries, "remote_uploaded": False, "at_unix": time.time(),
    }
    save_json(output / f"{name}.manifest.json", manifest)
    return manifest


def verify_archive(output: Path, name: str) -> dict:
    manifest = read_json(output / f"{name}.manifest.json")
    archive_path = output / manifest["archive"]
    if sha256(archive_path) != manifest["archive_sha256"]:
        raise ValueError("Compressed archive digest does not match its manifest")
    expected = {item["path"]: item for item in manifest["files"]}
    seen = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or member.name not in expected or member.name in seen:
                raise ValueError(f"Unexpected or nonregular archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("Archive omitted an expected regular file")
            digest, count = hashlib.sha256(), 0
            for block in iter(lambda: stream.read(2**20), b""):
                digest.update(block)
                count += len(block)
            if digest.hexdigest() != expected[member.name]["sha256"] or count != expected[member.name]["bytes"]:
                raise ValueError(f"Archived file content differs: {member.name}")
            seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive omitted manifest entries")
    return {"status": "passed", "archive": manifest["archive"], "files": len(seen), "archive_sha256": manifest["archive_sha256"]}


def payload_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        not relative or path.is_absolute() or ".." in path.parts
        or path.as_posix() != relative or relative == "."
    ):
        raise ValueError("Payload paths must be canonical project-relative paths")
    require_safe_name(path)
    target = root / path
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Payload path escaped its root")
    return target


def export_git_payload(output: Path, files: list[Path], *, project: Path = ROOT,
                       maximum_file_bytes: int = 50 * 2**20) -> dict:
    if output.exists():
        raise FileExistsError("Never replace a previously prepared reproducibility bundle")
    project = project.resolve()
    entries = []
    for source in sorted(set(files)):
        relative = source.relative_to(project).as_posix()
        destination = payload_path(output / "payload", relative)
        if source.is_symlink() or not source.resolve().is_relative_to(project):
            raise ValueError(f"Export requires an in-project regular file: {source}")
        if source.suffix.lower() in {".pt", ".bin", ".safetensors", ".ckpt", ".distcp"}:
            raise ValueError(f"Model tensors do not belong in the Git evidence payload: {source}")
        if source.stat().st_size > maximum_file_bytes:
            raise ValueError(f"Large payload belongs in model/data storage, not ordinary Git: {source}")
        digest, size = scan_and_hash(source)
        entries.append((source, destination, {"path": relative, "bytes": size, "sha256": digest}))
    output.mkdir(parents=True)
    for source, destination, entry in entries:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, destination.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=2**20)
        if destination.stat().st_size != entry["bytes"] or sha256(destination) != entry["sha256"]:
            raise ValueError(f"Source changed during reproducibility export: {source}")
    manifest = {
        "schema": "coevolve-git-evidence-v1", "source_project": str(project),
        "files": [entry for _, _, entry in entries],
        "uncompressed_bytes": sum(entry["bytes"] for _, _, entry in entries),
        "large_model_tensors_included": False, "training_submission_allowed": False,
        "remote_uploaded": False,
    }
    save_json(output / "MANIFEST.json", manifest)
    return manifest


def verify_git_payload(output: Path) -> dict:
    manifest = read_json(output / "MANIFEST.json")
    if manifest["schema"] != "coevolve-git-evidence-v1":
        raise ValueError("Unknown reproducibility bundle schema")
    root = output / "payload"
    expected = {}
    for entry in manifest["files"]:
        relative = entry["path"]
        path = payload_path(root, relative)
        if relative in expected or path.is_symlink() or not path.is_file():
            raise ValueError(f"Duplicate, nonregular, or missing payload: {relative}")
        if path.stat().st_size != entry["bytes"] or scan_and_hash(path)[0] != entry["sha256"]:
            raise ValueError(f"Payload size or digest changed: {relative}")
        expected[relative] = entry
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Reproducibility payload cannot contain symlinks: {path}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected) or not expected:
        raise ValueError("Reproducibility payload has undeclared or missing files")
    total = sum(entry["bytes"] for entry in expected.values())
    if total != manifest["uncompressed_bytes"]:
        raise ValueError("Reproducibility payload byte accounting changed")
    return {"status": "passed", "files": len(expected), "bytes": total}


def pin_checkpoints(output: Path, plan: dict) -> dict:
    entries = []
    for checkpoint in plan["checkpoints"]:
        source = Path(checkpoint["source"])
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"Checkpoint root is not an existing real directory: {source}")
        destination = output / "checkpoint-pins" / checkpoint["id"]
        destination.mkdir(parents=True, exist_ok=True)
        files = []
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Do not follow checkpoint symlinks: {path}")
            if not path.is_file():
                continue
            require_safe_name(path)
            relative = path.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = path.stat()
            copied_digest = None
            if not target.exists():
                try:
                    os.link(path, target)
                except OSError as exc:
                    if exc.errno != errno.EPERM:
                        raise
                    print(f"hardlink permission denied; copying read-only checkpoint file: {path}", flush=True)
                    digest = hashlib.sha256()
                    with path.open("rb") as stream, target.open("xb") as copied:
                        for block in iter(lambda: stream.read(8 * 2**20), b""):
                            copied.write(block)
                            digest.update(block)
                    copied_digest = digest.hexdigest()
            current, after = path.stat(), target.stat()
            if (before.st_size, before.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
                raise ValueError(f"Original checkpoint changed during preservation: {path}")
            linked = (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
            if before.st_size != after.st_size:
                raise ValueError(f"Preserved checkpoint has a different size: {target}")
            if not linked and copied_digest is None:
                copied_digest = sha256(target)
                if copied_digest != sha256(path):
                    raise ValueError(f"Existing checkpoint copy differs from the source: {target}")
            entry = {
                "name": str(relative), "bytes": before.st_size, "device": before.st_dev, "inode": before.st_ino,
                "local_pin": str(target.relative_to(output)),
                "storage": "hardlink" if linked else "read_only_source_copied_after_hardlink_permission_denial",
            }
            if copied_digest is not None:
                entry["sha256"] = copied_digest
                entry["hash_scope"] = "fully_hashed_during_local_copy"
            elif before.st_size <= 2**20:
                entry["sha256"] = sha256(path)
                entry["hash_scope"] = "fully_rehashed_now"
            elif str(path) in plan.get("prior_full_hashes", {}):
                entry["prior_sha256"] = plan["prior_full_hashes"][str(path)]
                entry["hash_scope"] = "recorded_prior_full_hash_not_rehashed_now"
            else:
                entry["hash_scope"] = "byte_size_and_original_inode_only_large_payload_not_rehashed"
            files.append(entry)
        entries.append({**checkpoint, "bytes": sum(item["bytes"] for item in files), "files": files})
        print(f"pinned {checkpoint['id']}: {entries[-1]['bytes'] / 2**30:.3f} GiB", flush=True)
    result = {
        "status": "locally_pinned_not_remote_backup", "same_filesystem_hardlinks": True,
        "independent_copy": False, "remote_uploaded": False, "large_payloads_fully_rehashed": False,
        "do_not_edit_pinned_files_in_place": True, "checkpoints": entries,
        "logical_bytes": sum(item["bytes"] for item in entries),
        "copied_bytes": sum(item["bytes"] for checkpoint in entries for item in checkpoint["files"]
                            if item["storage"] != "hardlink"),
        "at_unix": time.time(),
    }
    save_json(output / "CHECKPOINTS.json", result)
    return result


def verify_pins(output: Path) -> dict:
    manifest = read_json(output / "CHECKPOINTS.json")
    count = total = 0
    for checkpoint in manifest["checkpoints"]:
        root = Path(checkpoint["source"])
        for item in checkpoint["files"]:
            source, target = root / item["name"], output / item["local_pin"]
            old, pinned = source.stat(), target.stat()
            if old.st_size != item["bytes"] or pinned.st_size != item["bytes"]:
                raise ValueError(f"Checkpoint file size changed: {target}")
            if item["storage"] == "hardlink" and (old.st_dev, old.st_ino) != (pinned.st_dev, pinned.st_ino):
                raise ValueError(f"Checkpoint pin identity changed: {target}")
            if "sha256" in item and sha256(target) != item["sha256"]:
                raise ValueError(f"Small checkpoint metadata changed: {target}")
            count += 1
            total += pinned.st_size
    return {"status": "passed", "pinned_files": count, "logical_bytes": total, "independent_backup": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("action", choices=("source", "evidence", "assets", "pin", "verify"))
    args = parser.parse_args()
    plan = read_json(args.plan)
    output, run = Path(plan["output"]), Path(plan["run"])
    cancel = read_json(run / "USER_NO_H200_AND_HANDOFF.json")
    if cancel["h200_training_allowed"] or cancel["automatic_training_or_evaluation_submission_allowed"]:
        raise ValueError("Archive only while compute is explicitly paused")
    if not (run / "TWO_HOUR_RESUME_CANCELLED.json").is_file():
        raise ValueError("The automatic restart cancellation gate is missing")
    if args.action == "pin":
        result = pin_checkpoints(output, plan)
    elif args.action == "verify":
        result = {
            "status": "passed", "source": verify_archive(output, "source"),
            "evidence": verify_archive(output, "evidence"), "assets": verify_archive(output, "assets"),
            "checkpoint_pins": verify_pins(output),
            "remote_uploaded": False, "at_unix": time.time(),
        }
        save_json(output / "VERIFIED.json", result)
    else:
        exclusions = []
        if args.action == "source":
            files = [path for value in plan["git_roots"] for path in git_files(Path(value))]
            for value in plan["source_snapshots"]:
                files.extend(walk_files(Path(value), exclusions))
            files.extend(Path(value) for value in plan.get("source_extra_files", []))
        elif args.action == "evidence":
            files = list(walk_files(run, exclusions, text_only=True))
            files.extend(Path(value) for value in plan["required_evidence_files"])
            for value in plan.get("extra_evidence_roots", []):
                files.extend(walk_files(Path(value), exclusions, text_only=True))
        else:
            files = [path for value in plan["asset_roots"] for path in walk_files(Path(value), exclusions)]
        save_json(output / f"{args.action}.exclusions.json", exclusions)
        result = create_archive(output, args.action, files)
    print(json.dumps({key: value for key, value in result.items() if key not in ("files", "checkpoints")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
