import sys
import copy
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from sao_bridge import alfworld_sao_control as control
from sao_bridge.alfworld_h_train import training_spec as h_spec
from sao_bridge.alfworld_w_train import training_spec as w_spec
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.large_train import configure


@pytest.fixture
def initial():
    return read_json(control.INITIAL / "INITIAL_STATE.json")


@pytest.fixture
def release():
    return {
        "foreign_gpu_pids": [], "remaining_gpu_pids": [42],
        "verified_filler_pids": [42], "filler_unit_active": True,
    }


def source_fields():
    return control.source_provenance([Path(control.__file__)])


@pytest.mark.parametrize("role,steps,batch", [("h1", 32, 6), ("w1", 80, 8), ("h2", 32, 6)])
def test_original_algorithm_and_budget(tmp_path, initial, role, steps, batch):
    cfg = control.role_config(role, tmp_path, initial)
    assert (cfg["steps"], cfg["batch_size"]) == (steps, batch)
    assert cfg["learning_rate"] == 2e-6
    assert cfg["critic_learning_rate"] == 1e-5
    assert cfg["critic_warmup_steps"] == 5
    assert cfg["critic_updates_per_step"] == 1
    assert cfg["terminal_episodes"] and cfg["critic_mode"] == "learned"
    assert cfg["central_context"] is None and not cfg["value_normalization"]
    assert cfg["interaction_credit"] == 0
    assert cfg["harness"] == str(tmp_path / "no_patch.json")
    assert cfg["independent_trajectories"] and cfg["processed_behavior_logprobs"]
    assert not cfg.get("joint_root")
    if role == "w1":
        assert (cfg["actor_gpus"], cfg["rollout_gpus"]) == (4, 4)
        assert cfg["solver_model"] == initial["solver"]
        assert cfg["training_split"] == "test_set"
        assert "training_catalogue" not in cfg and "training_catalogue_sha256" not in cfg
        assert "training_data" not in cfg
    else:
        assert (cfg["actor_gpus"], cfg["rollout_gpus"]) == (3, 1)
        assert cfg["temperature"] == 0.7
    assert sum(control.COUNTS.values()) == 1024


def test_h2_uses_new_pair_and_new_solver(tmp_path, initial):
    state = {**initial, "engineer": "fresh-h1", "engineer_critic": "fresh-c1",
             "solver": "fresh-w1", "solver_version": "fresh-w1-identity"}
    cfg = control.role_config("h2", tmp_path, state)
    assert (cfg["engineer_model"], cfg["engineer_critic"]) == ("fresh-h1", "fresh-c1")
    assert cfg["solver_model"] == "fresh-w1"
    assert cfg["solver_version"] == "fresh-w1-identity"
    assert cfg["seed"] == 1000042


@pytest.mark.parametrize("role", ["h1", "w1"])
def test_actual_single_role_wiring_is_corrected_without_ma(tmp_path, initial, role):
    from areal.api.cli_args import MicroBatchSpec
    from areal.utils.data import allocate_balanced_mbs

    cfg = control.role_config(role, tmp_path, initial)
    audit = control.backend_wiring_audit(cfg)
    assert audit["processed_behavior_logprobs"] and audit["no_ma_no_kl"]
    spec = w_spec(cfg, tmp_path) if role == "w1" else h_spec(cfg, tmp_path, "unused")
    native = configure(spec, tmp_path)
    for key in ("actor", "critic"):
        assert native[key]["mb_spec"]["n_mbs"] == 2
        microbatch = MicroBatchSpec(**native[key]["mb_spec"])
        assert sorted(allocate_balanced_mbs(
            microbatch, [100, native[key]["mb_spec"]["max_tokens_per_mb"]],
        )) == [[0], [1]]
    assert native["actor"]["kl_ctl"] == 0 and native["ref"] is None
    assert native["recover"]["mode"] == "disabled"


def test_legacy_h_flags_stay_disabled(tmp_path, initial):
    cfg = control.role_config("h1", tmp_path, initial)
    cfg.pop("independent_trajectories")
    cfg.pop("processed_behavior_logprobs")
    spec = h_spec(cfg, tmp_path, "unused")
    assert not spec["independent_trajectories"]
    assert not spec["processed_behavior_logprobs"]


def test_missing_probability_forwarding_is_a_blocker(tmp_path, initial, monkeypatch):
    cfg = control.role_config("w1", tmp_path, initial)
    original = control.w_spec
    monkeypatch.setattr(control, "w_spec", lambda *args: {
        **original(*args), "processed_behavior_logprobs": False,
    })
    with pytest.raises(ValueError, match="Both role trainers"):
        control.backend_wiring_audit(cfg)


def test_missing_typed_backend_guard_is_a_blocker(tmp_path, initial, monkeypatch):
    from sao_bridge import large_train

    source = tmp_path / "old_backend.py"
    source.write_text("if spec.get('joint_root'):\n    config.vllm = JointVLLMConfig()\n")
    monkeypatch.setattr(large_train, "__file__", str(source))
    with pytest.raises(ValueError, match="Backend blocker"):
        control.backend_wiring_audit(control.role_config("h1", tmp_path, initial))


@pytest.mark.parametrize("changes", [
    {"foreign_gpu_pids": [99]}, {"filler_unit_active": False}, {"remaining_gpu_pids": [99]},
])
def test_filler_and_gpu_audit_fail_closed(release, changes):
    with pytest.raises(RuntimeError, match="released GPUs"):
        control.checked_gpu_release({**release, **changes})


def test_queue_needs_both_success_and_release(tmp_path, release, monkeypatch):
    predecessor = tmp_path / "predecessor"
    queue = tmp_path / "queue"
    write_json(predecessor / "COMPLETE.json", {"status": "completed"})
    sleeps = []

    def release_later(seconds):
        sleeps.append(seconds)
        write_json(predecessor / "RELEASED.json", release)

    monkeypatch.setattr(control.time, "sleep", release_later)
    control.wait_prerequisite(predecessor, queue, timeout=60)
    assert sleeps == [15]
    assert set(read_json(queue / "SATISFIED.json")) == {
        str(predecessor / "COMPLETE.json"), str(predecessor / "RELEASED.json"),
    }


def test_failed_predecessor_never_counts_as_success(tmp_path, release):
    write_json(tmp_path / "predecessor/FAILED.json", {"error": "training failed"})
    write_json(tmp_path / "predecessor/COMPLETE.json", {"status": "completed"})
    write_json(tmp_path / "predecessor/RELEASED.json", release)
    with pytest.raises(RuntimeError, match="Predecessor failed"):
        control.wait_prerequisite(tmp_path / "predecessor", tmp_path / "queue", 60)


def test_unavailable_result_is_not_a_completed_comparator(tmp_path, release):
    write_json(tmp_path / "predecessor/COMPLETE.json", {"status": "unavailable"})
    write_json(tmp_path / "predecessor/RELEASED.json", release)
    with pytest.raises(ValueError, match="actually complete"):
        control.wait_prerequisite(tmp_path / "predecessor", tmp_path / "queue", 60)


def test_wait_deadline_is_persistent_and_bounded(tmp_path, monkeypatch):
    predecessor, queue = tmp_path / "predecessor", tmp_path / "queue"
    write_json(queue / "WAITING.json", {"target": str(predecessor), "deadline_unix": 100})
    monkeypatch.setattr(control.time, "time", lambda: 101)
    with pytest.raises(TimeoutError, match="queue expired"):
        control.wait_prerequisite(predecessor, queue, 60)
    assert read_json(queue / "WAITING.json")["deadline_unix"] == 100
    with pytest.raises(ValueError, match="36h"):
        control.wait_prerequisite(predecessor, queue, control.TIMEOUT + 1)


def test_whale_dependency_targets_full_v4_execution_not_wrapper_or_smoke(tmp_path, release, monkeypatch):
    assert control.H_PREDECESSOR == control.ROOT / "runs/alfworld_whale_baseline_v4/execution"
    assert control.W_PREDECESSOR == control.ROOT / "runs/alfworld_w2_recovery_v1"
    assert control.DEFAULT_ROOT == control.ROOT / "runs/alfworld_sao_corrected_v3"
    predecessor, queue = tmp_path / "execution", tmp_path / "queue"
    write_json(predecessor / "SMOKE_COMPLETE.json", {"status": "passed"})
    write_json(predecessor / "RELEASED.json", release)
    times = iter([0, 2])
    monkeypatch.setattr(control.time, "time", lambda: next(times))
    with pytest.raises(TimeoutError, match="queue expired"):
        control.wait_prerequisite(predecessor, queue, 1)
    assert not (queue / "SATISFIED.json").exists()


def test_existing_freeze_cannot_silently_ignore_new_predecessor(tmp_path):
    write_json(tmp_path / "protocol.json", {
        "prerequisites": {"w1": str(control.W_PREDECESSOR), "h1": "historical-whale-v1"},
        "pinned_files": {},
    })
    with pytest.raises(ValueError, match="prepare a fresh root"):
        control.prepare(tmp_path, control.W_PREDECESSOR, control.H_PREDECESSOR)


def test_no_partial_optimizer_resume(tmp_path, initial, monkeypatch):
    cfg = control.role_config("h1", tmp_path, initial)
    write_json(tmp_path / "h1/config.json", cfg)
    write_json(tmp_path / "h1/INPUTS.json", {})
    Path(cfg["run"]).mkdir()
    monkeypatch.setattr(control, "backend_wiring_audit", lambda config: {})
    launch = Mock()
    with pytest.raises(RuntimeError, match="not an exact resume"):
        control.execute_training(tmp_path, "h1", launch)
    launch.module.assert_not_called()


def test_packets_reject_old_solver_and_missing_fresh_failures(tmp_path, initial):
    packets = read_json(Path(initial["packets"]))
    path = tmp_path / "packets.json"
    write_json(path, packets)
    state = {**initial, "packets": str(path), "packets_sha256": file_sha(path),
             "solver_weights_sha256": "fresh-w1"}
    with pytest.raises(ValueError, match="current W1 weights"):
        control.checked_packets(state)
    write_json(path, [])
    state["packets_sha256"] = file_sha(path)
    with pytest.raises(ValueError, match="No authentic failures"):
        control.checked_packets(state)


def test_heldout_packets_are_rejected(tmp_path, initial):
    packets = read_json(Path(initial["packets"]))
    packets[0]["metadata"].update(
        start=240, end=250, task_ids=list(range(240, 250)),
        baseline_rewards={str(i): 0.0 for i in range(240, 250)}, baseline_pass=0,
    )
    path = tmp_path / "packets.json"
    write_json(path, packets)
    state = {**initial, "packets": str(path), "packets_sha256": file_sha(path)}
    with pytest.raises(ValueError):
        control.checked_packets(state)


def test_selection_uses_only_valid_fresh_proposals_not_incumbent():
    proposals = [{"sample_index": i, "valid": i in (1, 2)} for i in range(8)]
    reports = {f"h1-proposal{i:02d}": {"accuracy": 1.0 if i == 0 else 0.25} for i in range(8)}
    reports["old_harness"] = {"accuracy": 1.0}
    reports["no_patch"] = {"accuracy": 1.0}
    assert control.choose_valid_proposal(proposals, reports, "h1") == "h1-proposal01"
    for row in proposals:
        row["valid"] = False
    with pytest.raises(ValueError, match="no valid harness"):
        control.choose_valid_proposal(proposals, reports, "h1")
    with pytest.raises(ValueError, match="eight proposal"):
        control.choose_valid_proposal(proposals[:7], reports, "h1")


def test_source_and_artifact_pins_reject_mutation(tmp_path):
    source = tmp_path / "backend.py"
    source.write_text("corrected = True\n")
    binding = control.pins([source])
    control.verify_pins(binding)
    source.write_text("corrected = False\n")
    with pytest.raises(ValueError, match="Frozen SAO input"):
        control.verify_pins(binding)


def test_wrong_node_never_starts_gpu(tmp_path, monkeypatch):
    write_json(tmp_path / "protocol.json", {"nodes": control.NODES})
    monkeypatch.setattr(control, "require_research_node", lambda: None)
    monkeypatch.setattr(control.os, "uname", lambda: SimpleNamespace(nodename=control.NODES["h1"]))
    launcher = Mock()
    monkeypatch.setattr(control, "Launcher", launcher)
    with pytest.raises(RuntimeError, match="assigned to"):
        control.train_role(tmp_path, "w1")
    launcher.assert_not_called()


def test_predecessor_failure_prevents_any_training(tmp_path, release, monkeypatch):
    predecessor = tmp_path / "previous"
    write_json(predecessor / "FAILED.json", {"error": "not finished"})
    write_json(tmp_path / "protocol.json", {
        "nodes": control.NODES, "pinned_files": {}, "queue_timeout_seconds": 60,
        "prerequisites": {"w1": str(predecessor)}, **source_fields(),
    })
    monkeypatch.setattr(control, "require_research_node", lambda: None)
    monkeypatch.setattr(control.os, "uname", lambda: SimpleNamespace(nodename=control.NODES["w1"]))
    launcher, audit = Mock(), Mock()
    monkeypatch.setattr(control, "Launcher", launcher)
    monkeypatch.setattr(control, "gpu_release_audit", audit)
    with pytest.raises(RuntimeError, match="Predecessor failed"):
        control.train_role(tmp_path, "w1")
    launcher.assert_not_called()
    audit.assert_not_called()
    assert read_json(tmp_path / "w1/FAILED.json")["status"] == "failed"


def test_final_report_refuses_partial_retry(tmp_path, monkeypatch):
    write_json(tmp_path / "protocol.json", {
        "nodes": control.NODES, "pinned_files": {}, **source_fields(),
    })
    write_json(tmp_path / "REPORT_STARTED.json", {})
    monkeypatch.setattr(control, "require_research_node", lambda: None)
    monkeypatch.setattr(control.os, "uname", lambda: SimpleNamespace(nodename=control.NODES["h2"]))
    launcher = Mock()
    monkeypatch.setattr(control, "Launcher", launcher)
    with pytest.raises(RuntimeError, match="once-only"):
        control.finalize(tmp_path)
    launcher.assert_not_called()


def test_h2_never_reuses_partial_refresh(tmp_path, initial, monkeypatch):
    pair = {"actor": "new-actor", "critic": "new-critic"}
    monkeypatch.setattr(control, "completed_role", lambda *args: {
        "pair": pair, "actor_weights_sha256": "weights", "critic_weights_sha256": "weights",
    })
    monkeypatch.setattr(control, "model_digests", lambda path: ("full", "weights"))
    write_json(tmp_path / "protocol.json", {})
    (tmp_path / "h2/current_w1_refresh").mkdir(parents=True)
    launcher = Mock()
    with pytest.raises(RuntimeError, match="refresh cannot be silently reused"):
        control.prepare_h2(tmp_path, {"initial_state": initial}, launcher)
    launcher.module.assert_not_called()


def test_h2_refresh_builds_new_w1_prompts_only(tmp_path, initial, release, monkeypatch):
    pairs = {
        "h1": {"actor": "new-h1", "critic": "new-h1-critic"},
        "w1": {"actor": "new-w1", "critic": "new-w1-critic"},
    }
    monkeypatch.setattr(control, "completed_role", lambda root, role: {
        "pair": pairs[role], "actor_weights_sha256": "weights", "critic_weights_sha256": "weights",
    })
    monkeypatch.setattr(control, "model_digests", lambda path: ("full", "weights"))
    write_json(tmp_path / "protocol.json", {})
    launcher = Mock()
    refresh = tmp_path / "h2/current_w1_refresh"

    def finish_refresh(*args, **kwargs):
        state = read_json(tmp_path / "h2/refresh_input.json")
        packets = read_json(Path(initial["packets"]))
        for packet in packets:
            packet["metadata"].update(
                solver_version=state["solver_version"], solver_weights_sha256="weights",
            )
            packet["messages"][0]["content"] = "Fresh current-W1 training evidence."
        write_json(refresh / "packets.json", packets)
        write_json(refresh / "baseline_metadata.json", [p["metadata"] for p in packets])
        write_json(refresh / "REFRESHED_STATE.json", {
            **state, "packet_solver": "new-w1", "packets": str(refresh / "packets.json"),
            "packets_sha256": file_sha(refresh / "packets.json"),
        })
        write_json(refresh / "COMPLETE.json", {
            "status": "completed", "engineer_training_ready": True,
            "baseline_tasks": 240, "report_opened": False, "solver_weights_sha256": "weights",
            "state_sha256": file_sha(refresh / "REFRESHED_STATE.json"),
        })
        write_json(refresh / "RELEASED.json", release)

    launcher.wait.side_effect = finish_refresh
    frozen = Mock()
    monkeypatch.setattr(control, "freeze_role", frozen)
    state = control.prepare_h2(tmp_path, {"initial_state": initial}, launcher)
    assert state["solver"] == "new-w1" and state["engineer"] == "new-h1"
    args = frozen.call_args.args
    rows = args[3]
    assert args[1] == "h2" and len(rows) == 192
    assert all(row["partner_version"] == 1 for row in rows)
    assert all(row["messages"][0]["content"].startswith("Fresh current-W1") for row in rows)
    assert rows[0]["id"].endswith("-w1-sample192")
    assert rows[-1]["id"].endswith("-w1-sample383")
    assert launcher.module.call_args.args[0] == "sao_bridge.alfworld_strong_refresh"


def test_final_evaluates_one_three_way_matrix_and_derives_strict197(tmp_path, release, monkeypatch):
    from sao_bridge.evaluation import stable_id_seed_offset

    protocol = {
        "nodes": control.NODES, "pinned_files": {}, "report_protocol": str(control.EVAL_PROTOCOL),
        "no_patch": str(tmp_path / "no_patch.json"), "classification": "ordinary SAO",
        "historical_deviation": "final H1 step31", **source_fields(),
        "budget": {"total": 1024, "role_allocation": {
            "sao": {"engineer": 384, "solver": 640}, "current_ma": {"engineer": 512, "solver": 512},
        }},
        "training_data": "Current MA first640 test_set train0-239 rows.",
    }
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "h2/refresh_binding.json", {})
    harnesses = {}
    for label in ("h1", "h2"):
        path = tmp_path / f"{label}-fresh.json"
        write_json(path, {"benchmark": "alfworld", "actions": []})
        harnesses[label] = {"harness": str(path), "sha256": file_sha(path)}
    monkeypatch.setattr(control, "completed_role", lambda root, role: {
        "consumed_trajectories": control.COUNTS[role],
        "pair": {"actor": "new-w1"}, "actor_weights_sha256": "weights",
        "harnesses": harnesses,
    })
    monkeypatch.setattr(control, "require_research_node", lambda: None)
    monkeypatch.setattr(control.os, "uname", lambda: SimpleNamespace(nodename=control.NODES["h2"]))
    monkeypatch.setattr(control, "gpu_release_audit", lambda: release)
    monkeypatch.setattr(control, "model_digests", lambda path: ("full", "weights"))
    monkeypatch.setattr(control, "Launcher", Mock())
    calls = []

    def run_matrix(launch, config, root, attempts):
        calls.append(config)
        assert attempts == 1
        assert config["solver_model"] == "new-w1"
        assert set(config["harnesses"]) == {"h1", "h2"}
        variants = {}
        for label in ("no_patch", "h1", "h2"):
            results = [{
                "task_id": task_id, "id": f"alfworld-{task_id}", "sample_index": 0,
                "rollout_seed": config["seed"] + stable_id_seed_offset(f"alfworld-{task_id}"),
                "reward": 1.0, "tokens": 1,
            } for task_id in config["task_ids"]]
            path = root / f"{label}.json"
            write_json(path, {"accuracy": 1.0, "metric": "mean@1", "question_count": 200,
                             "samples_per_question": 1, "results": results})
            variants[label] = {"report": str(path)}
        return {"variants": variants}, root

    monkeypatch.setattr(control, "run_matrix", run_matrix)
    result = control.finalize(tmp_path)
    assert len(calls) == 1 and len(calls[0]["task_ids"]) == 200 and calls[0]["samples"] == 1
    assert result["consumed_trajectories"] == 1024
    assert result["budget"]["role_allocation"]["sao"] == {"engineer": 384, "solver": 640}
    assert result["budget"]["role_allocation"]["current_ma"] == {"engineer": 512, "solver": 512}
    assert set(result["scores"]) == {"no_patch", "h1", "h2"}
    assert all(item["strict197"] == item["all200"] == 1 for item in result["scores"].values())
    assert control.finalize(tmp_path) == result
    assert len(calls) == 1
    assert result["source_root"] == str(control.ROOT.resolve())
    assert result["source_sha256"] == protocol["source_sha256"]
    assert read_json(tmp_path / "RELEASED.json")["protocol_sha256"] == file_sha(tmp_path / "protocol.json")


def test_snapshot_source_hashes_ignore_mutable_main_and_git_head(tmp_path):
    main, snapshot = tmp_path / "main", tmp_path / "snapshot"
    for base in (main, snapshot):
        (base / ".git").mkdir(parents=True)
        (base / ".git/HEAD").write_text("commit-one\n")
        (base / "backend.py").write_text("corrected = True\n")
    before = control.source_provenance([snapshot / "backend.py"], snapshot)
    (main / "backend.py").write_text("other_future_code = True\n")
    (main / ".git/HEAD").write_text("docs-commit-two\n")
    (snapshot / ".git/HEAD").write_text("irrelevant-metadata\n")
    assert control.source_provenance([snapshot / "backend.py"], snapshot) == before
    assert before["source_root"] == str(snapshot)
    assert list(before["source_files"]) == ["backend.py"]
    (snapshot / "backend.py").write_text("changed = True\n")
    assert control.source_provenance([snapshot / "backend.py"], snapshot) != before


def test_snapshot_rejects_source_symlink_back_to_mutable_main(tmp_path):
    main, snapshot = tmp_path / "main", tmp_path / "snapshot"
    main.mkdir()
    snapshot.mkdir()
    (main / "backend.py").write_text("mutable = True\n")
    (snapshot / "backend.py").symlink_to(main / "backend.py")
    with pytest.raises(ValueError, match="escapes executed source snapshot"):
        control.source_provenance([snapshot / "backend.py"], snapshot)


@pytest.fixture
def linked_source_snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("whale_local", "examples"):
        shutil.copytree(control.ROOT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__"))
    (snapshot / "third_party").mkdir()
    for dependency in (control.ROOT / "third_party").iterdir():
        if dependency.is_dir():
            (snapshot / "third_party" / dependency.name).symlink_to(dependency.resolve(), target_is_directory=True)
    for name in ("assets", "runs", "WHALE"):
        (snapshot / name).symlink_to((control.ROOT / name).resolve(), target_is_directory=True)
    return snapshot


def test_linked_upstream_provenance_retains_logical_names_and_commit_bindings(linked_source_snapshot):
    snapshot = linked_source_snapshot
    names = [
        "whale_local/sao_bridge/alfworld_sao_control.py",
        "third_party/sao/configs/phase2_hard_sao.yaml",
        "third_party/sao/vendor/AReaL/areal/engine/fsdp_engine.py",
    ]
    proof = control.source_provenance([snapshot / name for name in names], snapshot)
    assert list(proof["source_files"]) == names
    assert proof["source_upstreams"] == {
        "third_party/sao": {
            "root": str((control.ROOT / "third_party/sao").resolve()),
            "revision": control.SAO_REVISION,
        },
        "third_party/sao/vendor/AReaL": {
            "root": str((control.ROOT / "third_party/sao/vendor/AReaL").resolve()),
            "revision": control.AREAL_REVISION,
        },
    }
    bridge = snapshot / names[0]
    bridge.unlink()
    bridge.symlink_to(control.ROOT / names[0])
    with pytest.raises(ValueError, match="escapes executed source snapshot"):
        control.source_provenance([bridge], snapshot)


def test_declared_upstream_rejects_wrong_commit_and_relocated_repository(linked_source_snapshot, monkeypatch):
    snapshot = linked_source_snapshot
    backend = snapshot / "third_party/sao/vendor/AReaL/areal/engine/fsdp_engine.py"
    monkeypatch.setattr(control, "SAO_REVISION", "0" * 40)
    with pytest.raises(ValueError, match="repository/commit changed"):
        control.source_provenance([backend], snapshot)
    (snapshot / "third_party/sao").unlink()
    (snapshot / "third_party/sao").symlink_to(control.ROOT)
    with pytest.raises(ValueError, match="repository/commit changed"):
        control.source_provenance([backend], snapshot)


def test_upstream_leaf_symlink_cannot_escape_declared_layout(tmp_path, monkeypatch):
    snapshot, shared, outside = tmp_path / "snapshot", tmp_path / "shared-sao", tmp_path / "outside.py"
    (snapshot / "third_party").mkdir(parents=True)
    (shared / "configs").mkdir(parents=True)
    outside.write_text("not_declared_upstream = True\n")
    (shared / "configs/phase2_hard_sao.yaml").symlink_to(outside)
    (snapshot / "third_party/sao").symlink_to(shared, target_is_directory=True)
    monkeypatch.setattr(control, "declared_upstreams", lambda root: {
        "third_party/sao": {"root": str(shared), "revision": control.SAO_REVISION},
    })
    with pytest.raises(ValueError, match="escapes declared upstream layout"):
        control.source_provenance([snapshot / "third_party/sao/configs/phase2_hard_sao.yaml"], snapshot)


def test_linked_upstream_imports_are_not_misclassified_as_mutable_bridge(linked_source_snapshot, monkeypatch):
    snapshot = linked_source_snapshot
    areal = (snapshot / "third_party/sao/vendor/AReaL").resolve()
    paths, rejected = control.isolated_import_paths([
        str(snapshot / "whale_local"), str(snapshot / "third_party/sao/vendor/AReaL"),
        str(control.ROOT / "whale_local"), str(control.ROOT),
    ], snapshot)
    assert str(areal) in paths
    assert areal not in rejected
    assert control.ROOT / "whale_local" in rejected
    monkeypatch.setitem(sys.modules, "declared_upstream_example", SimpleNamespace(
        __file__=str(areal / "areal/engine/fsdp_engine.py"),
    ))
    # The parent pytest process intentionally imports its own bridge from main.
    monkeypatch.setattr(control, "sys", SimpleNamespace(modules={
        "declared_upstream_example": sys.modules["declared_upstream_example"],
    }))
    control.check_import_origins(snapshot, rejected)


def test_actual_copied_snapshot_prepares_and_validates_with_shared_upstreams(linked_source_snapshot, tmp_path):
    snapshot = linked_source_snapshot
    output = tmp_path / "prepared"
    env = {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": ":".join(str(snapshot / name) for name in (
            "whale_local", "third_party/sao/vendor/AReaL", "third_party/sao",
        )),
    }
    base = [sys.executable, "-B", "-m", "sao_bridge.alfworld_sao_control"]
    prepared = subprocess.run(
        [*base, "prepare", "--root", str(output), "--inputs-root", str(control.ROOT),
         "--w-predecessor", str(control.ROOT / "runs/alfworld_w2_recovery_v1"),
         "--h-predecessor", str(control.ROOT / "runs/alfworld_whale_baseline_v6/execution")],
        cwd=snapshot, env=env, capture_output=True, text=True, timeout=600,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr
    protocol = read_json(output / "protocol.json")
    assert protocol["source_root"] == str(snapshot)
    assert protocol["source_upstreams"]["third_party/sao"]["revision"] == control.SAO_REVISION
    assert protocol["source_upstreams"]["third_party/sao/vendor/AReaL"]["revision"] == control.AREAL_REVISION
    assert not read_json(output / "PREPARED.json")["backend_blockers"]
    validated = subprocess.run(
        [*base, "validate", "--root", str(output), "--validate-only"],
        cwd=snapshot, env=env, capture_output=True, text=True, timeout=120,
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr
    assert "cpu_validated" in validated.stdout
    assert not list(output.glob("*/STARTED.json"))
    assert not (output / "COMPLETE.json").exists()
    assert not (output / "RELEASED.json").exists()
    common = snapshot / "whale_local/sao_bridge/common.py"
    common.write_text(common.read_text() + "\n")
    rejected = subprocess.run(
        [*base, "validate", "--root", str(output), "--validate-only"],
        cwd=snapshot, env=env, capture_output=True, text=True, timeout=120,
    )
    assert rejected.returncode != 0
    assert "Frozen source snapshot changed" in rejected.stderr


def test_validate_only_checks_frozen_source_without_gpu_or_waiting(tmp_path, initial, monkeypatch):
    protocol = {"pinned_files": {}, "prerequisites": {"h1": "whale", "w1": "recovery"},
                **source_fields()}
    write_json(tmp_path / "protocol.json", protocol)
    write_json(tmp_path / "PREPARED.json", {"protocol_sha256": file_sha(tmp_path / "protocol.json")})
    for role in ("h1", "w1"):
        write_json(tmp_path / role / "config.json", control.role_config(role, tmp_path, initial))
        write_json(tmp_path / role / "INPUTS.json", {})
    forbidden = Mock(side_effect=AssertionError("CPU validation must never start GPU work or queues"))
    for name in ("require_research_node", "gpu_release_audit", "Launcher", "wait_prerequisite"):
        monkeypatch.setattr(control, name, forbidden)
    result = control.validate_only(tmp_path)
    assert result["status"] == "cpu_validated" and not result["gpu_started"]
    assert set(result["roles"]) == {"h1", "w1"}
    assert result["source_sha256"] == protocol["source_sha256"]
    forbidden.assert_not_called()
    protocol["source_root"] = str(tmp_path / "different-execution-root")
    write_json(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="frozen source snapshot"):
        control.validate_only(tmp_path)


def test_h_pipeline_orders_all_stages_and_stops_on_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(control, "train_role", lambda root, role: calls.append(role))
    monkeypatch.setattr(control, "finalize", lambda root: calls.append("finalize") or "result")
    assert control.run_h_pipeline(tmp_path) == "result"
    assert calls == ["h1", "h2", "finalize"]
    calls.clear()

    def failed_h1(root, role):
        calls.append(role)
        raise RuntimeError("h1 failed")

    monkeypatch.setattr(control, "train_role", failed_h1)
    with pytest.raises(RuntimeError, match="h1 failed"):
        control.run_h_pipeline(tmp_path)
    assert calls == ["h1"]


def test_pth_mutable_local_source_paths_are_pruned_without_changing_venv(tmp_path):
    snapshot, main, site = tmp_path / "snapshot", tmp_path / "main", tmp_path / "site-packages"
    for root in (snapshot, main):
        (root / "whale_local/sao_bridge").mkdir(parents=True)
        (root / "third_party/sao/vendor/AReaL/areal/api").mkdir(parents=True)
        (root / "third_party/sao/vendor/AReaL/areal/api/cli_args.py").write_text("")
        (root / "third_party/sao/configs").mkdir()
        (root / "third_party/sao/configs/phase2_hard_sao.yaml").write_text("")
    paths, rejected = control.isolated_import_paths([
        str(main / "whale_local"), str(main / "third_party/sao/vendor/AReaL"),
        str(site), str(main), str(snapshot / "whale_local"),
    ], snapshot)
    assert paths[:3] == [str(snapshot / name) for name in (
        "whale_local", "third_party/sao/vendor/AReaL", "third_party/sao",
    )]
    assert str(site) in paths
    assert str(main / "whale_local") not in paths
    assert str(main / "third_party/sao/vendor/AReaL") not in paths
    assert main / "whale_local" in rejected
    assert not site.exists()


def test_already_loaded_mutable_module_is_rejected(tmp_path, monkeypatch):
    main, snapshot = tmp_path / "main/whale_local", tmp_path / "snapshot"
    monkeypatch.setitem(sys.modules, "stale_pth_local_helper", SimpleNamespace(
        __file__=str(main / "math_reward.py"),
    ))
    with pytest.raises(RuntimeError, match="Mutable local source imported"):
        control.check_import_origins(snapshot, [main])


@pytest.mark.parametrize("module", ["sao_bridge.alfworld_h_train", "vllm.entrypoints.openai.api_server"])
def test_all_owned_launches_wrap_child_startup_and_preserve_server_environment(module):
    launcher = Mock()
    control.launch_snapshot_module(
        launcher, module, ["--help"], gpus="0", name="child",
        env_overrides={"VLLM_PORT": "19502"},
    )
    call = launcher.start.call_args
    assert call.args[0] == [
        sys.executable, "-B", "-m", "sao_bridge.alfworld_sao_control",
        "_snapshot-module", module, "--help",
    ]
    assert call.kwargs["env_overrides"] == {"VLLM_PORT": "19502"}
    assert call.kwargs["gpus"] == "0"


@pytest.fixture
def joint_rows():
    return read_json(control.INITIAL / "data.json")


def test_matched_w1_task_and_prompt_provenance_follow_current_ma_prefix(joint_rows):
    cohort = read_json(control.EVAL_PROTOCOL)
    rows = control.matched_solver_rows(joint_rows, cohort["all_task_ids"])
    assert len(rows) == 640
    assert [r["task_id"] for r in rows] == [r["task_id"] for r in joint_rows[:640]]
    assert [r["task_context"] for r in rows] == [r["task_context"] for r in joint_rows[:640]]
    assert [r["source_joint_id"] for r in rows] == [r["id"] for r in joint_rows[:640]]
    assert [r["source_joint_index"] for r in rows] == list(range(640))
    assert [r["source_engineer_prompt_sha256"] for r in rows] == [
        control.contract_digest(r["messages"]) for r in joint_rows[:640]
    ]
    assert [r["task_id"] for r in rows[:512]] == [r["task_id"] for r in joint_rows[:512]]
    assert {r["task_id"] for r in rows} == set(range(240))
    assert not {r["task_id"] for r in rows} & set(cohort["all_task_ids"])
    assert all(r["task_split"] == "test_set" for r in rows)
    assert len({r["id"] for r in rows}) == 640
    assert all(not {"task_catalogue", "task_catalogue_sha256", "messages", "critic_evidence",
                    "joint_version", "joint_slot"} & r.keys() for r in rows)


def test_matched_w1_worker_assignments_cover_rows_without_active_window_reuse(tmp_path, initial, joint_rows):
    rows = control.matched_solver_rows(joint_rows, read_json(control.EVAL_PROTOCOL)["all_task_ids"])
    write_json(tmp_path / "w1/data.json", rows)
    config = control.role_config("w1", tmp_path, initial)
    schedule = control.assignment_schedule_audit(config)
    assert schedule["active_window"] == 16
    assert [r["task_name"] for r in rows] == [f"alfworld-sao-w1-{i % 32:02d}" for i in range(640)]
    for offset in range(640 - 15):
        assert len({r["task_name"] for r in rows[offset:offset + 16]}) == 16
    assignments = control.matched_solver_assignments(tmp_path / "empty.json", rows)
    assert len(assignments) == 32
    by_name = {item["name"]: item for item in assignments}
    for item in assignments:
        assert item["split"] == "test_set"
        assert "catalogue" not in item and "catalogue_sha256" not in item
        assert len(item["task_ids"]) == len(set(item["task_ids"]))
    for row in rows:
        assert row["task_id"] in by_name[row["task_name"]]["task_ids"]


@pytest.mark.parametrize("mutation", [
    {"task_id": 240}, {"task_split": "sao_train_v1"},
    {"task_catalogue": "old-pool.json"}, {"task_catalogue_sha256": "old-pool-sha"},
])
def test_matched_w1_rejects_heldout_tasks_and_old_training_catalogue(joint_rows, mutation):
    changed = list(joint_rows)
    changed[0] = {**changed[0], **mutation}
    with pytest.raises(ValueError, match="train0-239 population"):
        control.matched_solver_rows(changed, read_json(control.EVAL_PROTOCOL)["all_task_ids"])


def test_matched_w1_does_not_mutate_ma_data_or_repeat_short_pool(joint_rows):
    before = copy.deepcopy(joint_rows[:640])
    control.matched_solver_rows(joint_rows, read_json(control.EVAL_PROTOCOL)["all_task_ids"])
    assert joint_rows[:640] == before
    with pytest.raises(ValueError, match="at least 640"):
        control.matched_solver_rows(joint_rows[:512], [])
