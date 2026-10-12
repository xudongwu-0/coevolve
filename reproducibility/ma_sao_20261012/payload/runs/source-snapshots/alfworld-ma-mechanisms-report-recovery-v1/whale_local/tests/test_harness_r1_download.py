import hashlib
from types import SimpleNamespace

import pytest

from download_harness_r1 import verify_download
from harness_r1_searchqa.serve_roles import role_server_args


def test_release_weight_must_match_remote_digest(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    item = SimpleNamespace(
        size=7, rfilename="harness-r1/model.safetensors",
        lfs=SimpleNamespace(sha256=hashlib.sha256(b"weights").hexdigest()),
    )
    receipt = verify_download(path, item)
    assert receipt["verified"] is True
    assert receipt["path"] == "model.safetensors"
    item.lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        verify_download(path, item)


def test_release_small_file_uses_git_blob_identity(tmp_path):
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    item = SimpleNamespace(
        size=2, rfilename="harness-r1/config.json", lfs=None,
        blob_id=hashlib.sha1(b"blob 2\0{}").hexdigest(),
    )
    assert verify_download(path, item)["hash_kind"] == "git_blob_sha1"
    item.size = 3
    with pytest.raises(ValueError, match="size mismatch"):
        verify_download(path, item)


def test_official_qwen_serving_is_explicit_and_preserves_default():
    assert role_server_args({}) == []
    assert role_server_args({
        "tool_call_parser": "qwen3_xml", "language_model_only": True,
        "gdn_prefill_backend": "triton",
        "prefix_caching": True,
    }) == [
        "--tool-call-parser", "qwen3_xml", "--language-model-only",
        "--gdn-prefill-backend", "triton",
        "--enable-prefix-caching",
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        role_server_args({"tool_call_parser": "not-a-parser"})
    with pytest.raises(ValueError, match="Unsupported"):
        role_server_args({"gdn_prefill_backend": "not-a-backend"})
