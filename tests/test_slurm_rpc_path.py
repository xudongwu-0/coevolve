import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import slurm_runtime


def test_rpc_path_is_job_bound_and_short_without_creating_files(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "7770407")
    path = slurm_runtime.rpc_directory()
    assert path == slurm_runtime.REMOTE_ROOT / "tmp" / "vllm-7770407"
    assert len(os.fsencode(str(path))) + 37 < 108


@pytest.mark.parametrize("job", ["", "../escape", "abc", "1/2"])
def test_rpc_path_rejects_missing_or_invalid_allocation(monkeypatch, job):
    monkeypatch.setenv("SLURM_JOB_ID", job)
    with pytest.raises(RuntimeError, match="allocation"):
        slurm_runtime.rpc_directory()


def test_rpc_path_rejects_overlong_root(monkeypatch):
    monkeypatch.setattr(slurm_runtime, "REMOTE_ROOT", Path("/" + "x" * 90))
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(ValueError, match="path limit"):
        slurm_runtime.rpc_directory()


def test_rpc_path_rejects_redirected_job_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(slurm_runtime, "REMOTE_ROOT", tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    (tmp_path / "tmp").mkdir()
    (tmp_path / "elsewhere").mkdir()
    (tmp_path / "tmp/vllm-123").symlink_to(tmp_path / "elsewhere", target_is_directory=True)
    with pytest.raises(ValueError, match="redirected"):
        slurm_runtime.rpc_directory()
