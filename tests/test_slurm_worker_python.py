from pathlib import Path

import pytest

from sao_bridge import slurm_runtime


def fixture(monkeypatch, tmp_path):
    interpreter = tmp_path / ".tools/python"
    interpreter.parent.mkdir()
    interpreter.write_text("interpreter fixture")
    worker = tmp_path / ".envs/alfworld/bin/python"
    worker.parent.mkdir(parents=True)
    worker.symlink_to(interpreter)
    (worker.parent.parent / "pyvenv.cfg").write_text("include-system-site-packages = false\n")
    plan = {"config": {"worker_python": str(worker)}}
    monkeypatch.setattr(slurm_runtime, "REMOTE_ROOT", tmp_path)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    return worker, interpreter, plan


def test_preserves_virtualenv_symlink_for_invocation(monkeypatch, tmp_path):
    worker, interpreter, _ = fixture(monkeypatch, tmp_path)
    assert worker.resolve() == interpreter
    assert slurm_runtime.worker_python() == worker
    assert slurm_runtime.worker_python() != interpreter


def test_rejects_base_interpreter_and_missing_venv_config(monkeypatch, tmp_path):
    worker, interpreter, plan = fixture(monkeypatch, tmp_path)
    plan["config"]["worker_python"] = str(interpreter)
    with pytest.raises(ValueError, match="virtualenv"):
        slurm_runtime.worker_python()
    plan["config"]["worker_python"] = str(worker)
    (worker.parent.parent / "pyvenv.cfg").unlink()
    with pytest.raises(ValueError, match="virtualenv"):
        slurm_runtime.worker_python()


def test_rejects_executable_target_outside_project(monkeypatch, tmp_path):
    worker, _, _ = fixture(monkeypatch, tmp_path)
    worker.unlink()
    worker.symlink_to(Path("/usr/bin/python3"))
    with pytest.raises(ValueError, match="outside"):
        slurm_runtime.worker_python()
