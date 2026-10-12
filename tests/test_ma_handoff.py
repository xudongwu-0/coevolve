import importlib.util
import errno
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_ma_handoff.py"
loader = importlib.util.spec_from_file_location("ma_handoff", SCRIPT)
assert loader is not None and loader.loader is not None
handoff = importlib.util.module_from_spec(loader)
loader.loader.exec_module(handoff)


@pytest.mark.parametrize("name", [
    ".env", ".env.local", "access.token", "credentials.json", "id.key", "private-key.txt",
    "api_key.txt", "github_api_key.txt", "wandb_api_key.txt", "huggingface_api_key.txt",
    "api_keys/result.json",
])
def test_credential_paths_are_not_exported(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("private")
    with pytest.raises(ValueError, match="Credential-like path"):
        handoff.scan_and_hash(path)


def test_credential_content_is_detected_across_stream_boundaries(tmp_path):
    path = tmp_path / "result.json"
    path.write_bytes(b" " * (2**20 - 8) + b"-----BEGIN " + b"PRIVATE KEY-----")
    with pytest.raises(ValueError, match="Credential-like content"):
        handoff.scan_and_hash(path)


def test_empty_environment_template_is_included_but_populated_credentials_are_blocked(tmp_path):
    path = tmp_path / ".env.example"
    path.write_text("ADMIN_API_KEY=\nSAO_REWARD_WORKER_HOSTS=127.0.0.1\n")
    assert handoff.scan_and_hash(path)[1] == path.stat().st_size
    path.write_text("ADMIN_API_KEY=must-not-export\n")
    with pytest.raises(ValueError, match="must be empty"):
        handoff.scan_and_hash(path)


def test_archive_roundtrip_covers_every_byte_and_does_not_follow_links(tmp_path):
    source, output = tmp_path / "project", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "run.json").write_text(json.dumps({"successes": 186, "questions": 200}))
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ignored.json").write_text("do not include")
    (source / "linked").symlink_to(outside, target_is_directory=True)
    exclusions = []
    files = list(handoff.walk_files(source, exclusions, text_only=True))
    assert files == [source / "run.json"]
    assert exclusions[0]["reason"] == "symlink_not_followed"
    manifest = handoff.create_archive(output, "evidence", files, project=source)
    assert manifest["remote_uploaded"] is False
    assert handoff.verify_archive(output, "evidence")["files"] == 1
    archive = output / "evidence.tar.gz"
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Compressed archive digest"):
        handoff.verify_archive(output, "evidence")


def test_checkpoint_pins_keep_original_file_identity_without_claiming_backup(tmp_path):
    source, output = tmp_path / "state", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    original = source / "controller.json"
    original.write_text('{"round": 24}')
    manifest = handoff.pin_checkpoints(output, {
        "checkpoints": [{"id": "native24", "source": str(source), "purpose": "resume"}],
    })
    assert manifest["independent_copy"] is False and manifest["remote_uploaded"] is False
    assert handoff.verify_pins(output)["pinned_files"] == 1
    target = output / manifest["checkpoints"][0]["files"][0]["local_pin"]
    original.unlink()
    assert target.read_text() == '{"round": 24}'


def test_large_payload_hash_scope_is_explicit(tmp_path):
    source, output = tmp_path / "model", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "weights.bin").write_bytes(b"0" * (2**20 + 1))
    manifest = handoff.pin_checkpoints(output, {"checkpoints": [{"id": "model", "source": str(source)}]})
    entry = manifest["checkpoints"][0]["files"][0]
    assert "sha256" not in entry
    assert entry["hash_scope"] == "byte_size_and_original_inode_only_large_payload_not_rehashed"


def test_denied_hardlinks_are_copied_and_logged_without_changing_source_permissions(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "readonly", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    original = source / "state.json"
    original.write_text('{"round": 24}')
    mode = original.stat().st_mode

    def denied(*args):
        raise PermissionError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(handoff.os, "link", denied)
    manifest = handoff.pin_checkpoints(output, {"checkpoints": [{"id": "readonly", "source": str(source)}]})
    entry = manifest["checkpoints"][0]["files"][0]
    assert entry["storage"] == "read_only_source_copied_after_hardlink_permission_denial"
    assert entry["sha256"] == handoff.sha256(original)
    assert original.stat().st_mode == mode
    assert "hardlink permission denied" in capsys.readouterr().out
    assert handoff.verify_pins(output)["pinned_files"] == 1


def test_git_payload_roundtrip_rejects_overwrite_and_extra_files(tmp_path):
    project, output = tmp_path / "project", tmp_path / "bundle"
    source = project / "runs/study/RESULT.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"successes":186,"questions":200}')
    manifest = handoff.export_git_payload(output, [source], project=project)
    assert manifest["remote_uploaded"] is False
    assert handoff.verify_git_payload(output)["files"] == 1
    with pytest.raises(FileExistsError, match="Never replace"):
        handoff.export_git_payload(output, [source], project=project)
    (output / "payload/unlisted.txt").write_text("not in manifest")
    with pytest.raises(ValueError, match="undeclared"):
        handoff.verify_git_payload(output)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/../b", "./relative", ".", ""])
def test_git_payload_rejects_noncanonical_paths(tmp_path, name):
    with pytest.raises(ValueError, match="canonical"):
        handoff.payload_path(tmp_path, name)


def test_git_payload_refuses_symlink_escape_and_large_files(tmp_path):
    project, outside = tmp_path / "project", tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    (outside / "data.json").write_text("{}")
    (project / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="in-project"):
        handoff.export_git_payload(tmp_path / "escaped", [project / "escape/data.json"], project=project)
    source = project / "large.json"
    source.write_text("123456")
    with pytest.raises(ValueError, match="Large payload"):
        handoff.export_git_payload(tmp_path / "large", [source], project=project, maximum_file_bytes=5)
    assert not (tmp_path / "large").exists()


def test_git_payload_detects_tampering_and_duplicate_manifest_members(tmp_path):
    project, output = tmp_path / "project", tmp_path / "bundle"
    project.mkdir()
    source = project / "data.json"
    source.write_text("{}")
    manifest = handoff.export_git_payload(output, [source], project=project)
    manifest["files"].append(manifest["files"][0])
    (output / "MANIFEST.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Duplicate"):
        handoff.verify_git_payload(output)
    manifest["files"].pop()
    (output / "MANIFEST.json").write_text(json.dumps(manifest))
    (output / "payload/data.json").write_text("[]")
    with pytest.raises(ValueError, match="digest changed"):
        handoff.verify_git_payload(output)


def test_even_small_model_tensors_cannot_enter_git_payload(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "small.pt"
    source.write_bytes(b"model")
    with pytest.raises(ValueError, match="Model tensors"):
        handoff.export_git_payload(tmp_path / "bundle", [source], project=project)
