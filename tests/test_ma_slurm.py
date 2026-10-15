import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import concurrent_comparison as comparison
from sao_bridge import ma_slurm
from sao_bridge import slurm_runtime


def standalone(root):
    config = {
        "run": str(root), "active_arms": ["ma_sao"],
        "nodes": {"ma_sao": {"editor": "n0", "solver": "n1"}},
        "engineer_model": str(root / "models/harness-r1"),
        "solver_model": str(root / "models/target-agent-sft"),
        "no_patch": str(root / "empty.json"), "training_deadline_unix": 1e12,
    }
    spec = comparison.role_spec(config, "ma_sao", "editor", root / "data.json", {})
    spec.update(execution_mode="standalone-ma-v1", baseline_definition=comparison.STANDALONE_MA)
    path = root / "ma_sao/editor_spec.json"
    comparison.immutable(path, spec)
    plan = {"schema": comparison.SCHEMA, "kind": "standalone-ma-training",
            "config": config, "inputs": {}, "specs": {str(path): comparison.sha(path)}}
    comparison.immutable(root / "COMPARISON.json", plan)
    return plan, spec


def test_only_manifest_bound_ma_can_run(monkeypatch, tmp_path):
    plan, spec = standalone(tmp_path)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    comparison.require_executable_spec(spec)
    altered = copy.deepcopy(spec)
    altered["run"] += "-different"
    with pytest.raises(ValueError, match="bound"):
        comparison.require_executable_spec(altered)
    plan["specs"][str(tmp_path / "ma_sao/editor_spec.json")] = "0" * 64
    with pytest.raises(ValueError, match="bound"):
        comparison.require_executable_spec(spec)


def test_standalone_flag_cannot_relabel_sao(monkeypatch, tmp_path):
    plan, spec = standalone(tmp_path)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    spec["comparison_arm"] = "sao"
    with pytest.raises(ValueError, match="baseline"):
        comparison.require_executable_spec(spec)


@pytest.mark.parametrize("count,host,owner,valid", [
    (2, "n0", None, True), (4, "n0", None, False),
    (2, "login", None, False), (2, "n0", -1, False),
])
def test_two_node_binding(monkeypatch, tmp_path, count, host, owner, valid):
    plan, _ = standalone(tmp_path)
    monkeypatch.setattr(slurm_runtime, "binding", lambda: (tmp_path, plan))
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv(slurm_runtime.ENV[1], comparison.sha(tmp_path / "COMPARISON.json"))
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename=host))
    uid = os.getuid() if owner is None else owner
    def command(args, **kwargs):
        return (f"JobState=RUNNING NumNodes={count} UserId=user({uid}) NodeList=n[0-1]"
                if "job" in args else "n0\nn1\n")
    monkeypatch.setattr(slurm_runtime.subprocess, "check_output", command)
    if valid:
        assert slurm_runtime.require_node()["nodes"] == ["n0", "n1"]
    else:
        with pytest.raises(RuntimeError):
            slurm_runtime.require_node()


def test_bootstrap_and_training_share_only_job_owned_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(nodename="n0"))
    config = {"study_run": str(tmp_path), "runtime_cache_root": str(tmp_path / "cache")}
    for phase in ("bootstrap", "training"):
        assert slurm_runtime.cache_directory(tmp_path / phase, {"config": config}) == tmp_path / "cache/n0/cache"
    config["runtime_cache_root"] = str(tmp_path.parent / "escape")
    with pytest.raises(ValueError, match="outside"):
        slurm_runtime.cache_directory(tmp_path / "training", {"config": config})


def test_ma_evaluation_checks_only_its_own_completion_before_imports(monkeypatch, tmp_path):
    standalone(tmp_path)
    monkeypatch.setattr(slurm_runtime, "execution_environment", lambda _: {})
    monkeypatch.setattr(slurm_runtime, "require_node", lambda: {})
    def incomplete(root, *, arms):
        assert root == tmp_path and arms == ("ma_sao",)
        raise ValueError("MA not complete")
    monkeypatch.setattr(comparison, "final_pairs", incomplete)
    with pytest.raises(ValueError, match="MA not complete"):
        comparison.evaluate_final(tmp_path, "ma_sao", None)


def test_preserved_metadata_resolves_without_rewriting_old_files(monkeypatch, tmp_path):
    repo = tmp_path / "coevolve"
    payload = repo / "reproducibility/payload"
    target = payload / "runs/protocol/selection.json"
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"selection_task_ids": [240]}))
    monkeypatch.setattr(ma_slurm, "REPO", repo)
    monkeypatch.setattr(ma_slurm, "PAYLOAD", payload)
    monkeypatch.setattr(comparison, "contained", lambda p: p)
    old = "/mnt/3fs/xudong/coevolve/runs/source-snapshots/frozen/runs/protocol/selection.json"
    assert ma_slurm.preserved(old) == target
    with pytest.raises(FileNotFoundError):
        ma_slurm.preserved(old + ".missing")
