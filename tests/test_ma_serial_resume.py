import copy
import json
import math
import os
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from sao_bridge import alfworld_ma_idle_resume as monitor
from sao_bridge import resources
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.joint_recovery import science_digest
from sao_bridge.launch import Launcher


@pytest.fixture
def serial(tmp_path, monkeypatch):
    for name in resources.SCOPE_ENV:
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "study"
    control = root / "serial15"
    write_json(root / "PROTOCOL.json", {"original_absolute_deadline_unix": 100.0})
    for relative in monitor.SERIAL_PAUSE_RECORDS:
        write_json(root / relative, {"h200_training_allowed": False})
    policy = {
        "schema": monitor.SERIAL_SCHEMA, "run_root": str(root), "authorized_at_unix": 1000.0,
        "h200_training_allowed": True, "automatic_submission_allowed": True,
        "supersedes_prior_pause": True, "poll_interval_seconds": 600,
        "minimum_idle_seconds": 0, "minimum_samples": 1,
        "nodes": monitor.HOSTS, "arms": list(monitor.ARMS), "expires_at_unix": None,
        "superseded_cancellations": {
            relative: file_sha(root / relative) for relative in monitor.SERIAL_PAUSE_RECORDS
        },
        "original_protocol_sha256": file_sha(root / "PROTOCOL.json"), "original_deadline_unix": 100.0,
        "gpu_indices": copy.deepcopy(monitor.SERIAL_GPU_INDICES), "launch_mode": "immediate_serial",
        "learner_ranks_per_role": 4, "retry_after_interruption": False,
        "retention_policy": "original_latest_native_and_hf96_hf144",
        "self_failure_recovery": "diagnose_fix_then_native_resume",
        "external_kill_recovery": "stop_no_restart",
        "evaluation_host": monitor.HOSTS["node3"], "evaluation_replicas": 8,
    }
    authority = control / "AUTHORIZATION.json"
    write_json(authority, policy)
    digest = file_sha(authority)
    nodes = {}
    for name, indices in monitor.SERIAL_GPU_INDICES.items():
        nodes[name] = {
            "gpus": [{"index": index, "name": "NVIDIA H200", "utilization": 0, "memory_mib": 0}
                     for index in range(8)],
            "allocated_gpu_indices": indices,
            "gpu_pids": [], "verified_filler_pids": [], "foreign_gpu_pids": [],
            "vanished_gpu_pids": [], "distributed_launchers": [],
            "verified_filler_unit": False, "filler_kind": None,
        }
    nodes["node4"]["gpus"][0].update(utilization=100, memory_mib=8000)
    admission = control / "admissions/one.json"
    write_json(admission, {
        "eligible": True, "authority_sha256": digest, "samples": 1, "elapsed_seconds": 0,
        "first_sample_unix": 1100, "last_sample_unix": 1100,
        "final_recheck": {"eligible": True, "nodes": nodes},
        "mode": "immediate_serial", "gpu_indices": monitor.SERIAL_GPU_INDICES, "at_unix": 1101,
    })
    binding = {
        "authority": str(authority), "authority_sha256": digest,
        "admission": str(admission), "admission_sha256": file_sha(admission),
    }
    return root, control, policy, binding


def enable(serial, monkeypatch, node="node4"):
    binding = serial[3]
    monkeypatch.setenv(resources.SCOPE_ENV[0], binding["authority"])
    monkeypatch.setenv(resources.SCOPE_ENV[1], binding["authority_sha256"])
    monkeypatch.setattr(resources.os, "uname", lambda: SimpleNamespace(nodename=monitor.HOSTS[node]))


def role_spec(serial, role="solver"):
    root, _, _, binding = serial
    return {
        "role": role, "actor_gpus": 4, "rollout_gpus": 4, "batch_size": 8,
        "training_deadline_unix": 100, "component_ablation": "without_peer",
        "rolling_recovery": {"root": str(root / "without_peer/recovery"), "execution_authority": binding},
    }


def test_immediate_authority_is_distinct_from_cancelled_two_hour_waiter(serial):
    root, _, _, binding = serial
    assert math.isinf(monitor.execution_deadline(100, binding, root=root, arm="without_peer"))
    assert monitor.execution_deadline(100, None, root=root, arm="without_peer") == 100


@pytest.mark.parametrize("change", ["gpu0", "actor_ranks", "retry", "retention", "old_cancel", "new_cancel"])
def test_new_authority_never_widens_scope_or_reuses_old_cancellation(serial, change):
    root, control, policy, binding = serial
    if change == "old_cancel":
        write_json(root / "idle2h_20261012/CANCELLED.json", {"changed": True})
    elif change == "new_cancel":
        write_json(control / "CANCELLED.json", {"cancelled": True})
    else:
        if change == "gpu0":
            policy["gpu_indices"]["node4"] = list(range(8))
        elif change == "actor_ranks":
            policy["learner_ranks_per_role"] = 3
        elif change == "retry":
            policy["retry_after_interruption"] = True
        else:
            policy["retention_policy"] = "delete_old_final_states"
        write_json(Path(binding["authority"]), policy)
        binding["authority_sha256"] = file_sha(Path(binding["authority"]))
    with pytest.raises((ValueError, InterruptedError)):
        monitor.verify_authority(Path(binding["authority"]), binding["authority_sha256"])


@pytest.mark.parametrize("change", ["occupied", "missing_scope", "stale", "wrong_mode", "boolean_samples"])
def test_manual_admission_requires_fresh_idle_partition_evidence(serial, change):
    root, _, _, binding = serial
    path = Path(binding["admission"])
    value = read_json(path)
    if change == "occupied":
        value["final_recheck"]["nodes"]["node4"]["gpus"][1]["memory_mib"] = 10
    elif change == "missing_scope":
        value["final_recheck"]["nodes"]["node4"].pop("allocated_gpu_indices")
    elif change == "stale":
        value["at_unix"] = 1161
    elif change == "wrong_mode":
        value["mode"] = "idle2h"
    else:
        value["samples"] = True
    write_json(path, value)
    binding["admission_sha256"] = file_sha(path)
    with pytest.raises(ValueError):
        monitor.execution_deadline(100, binding, root=root, arm="without_peer")


def test_native_science_and_four_learner_ranks_survive_rollout_resize(serial, monkeypatch):
    enable(serial, monkeypatch)
    spec = role_spec(serial)
    before = copy.deepcopy(spec)
    digest = science_digest(spec)
    assert resources.runtime_gpu_counts(spec) == (4, 3)
    assert spec == before and science_digest(spec) == digest
    assert resources.logical_gpu_string() == "0,1,2,3,4,5,6"
    assert resources.physical_gpu_string("0,1,2,3,4,5,6") == "1,2,3,4,5,6,7"
    enable(serial, monkeypatch, "node3")
    assert resources.runtime_gpu_counts(role_spec(serial, "editor")) == (4, 4)
    assert resources.physical_gpu_string("0,1,2,3,4,5,6,7") == "0,1,2,3,4,5,6,7"


@pytest.mark.parametrize("logical", ["7", "0,0", "1,01", "-1", "GPU-uuid", "0,8"])
def test_launch_cannot_escape_or_duplicate_allocated_devices(serial, monkeypatch, logical):
    enable(serial, monkeypatch)
    with pytest.raises(ValueError):
        resources.physical_gpu_string(logical)


def test_native_partition_needs_the_matching_admitted_role(serial, monkeypatch):
    enable(serial, monkeypatch)
    for role in ("solver", "editor"):
        spec = role_spec(serial, role)
        if role == "solver":
            spec["rolling_recovery"].pop("execution_authority")
        with pytest.raises(ValueError):
            resources.runtime_gpu_counts(spec)


@pytest.mark.parametrize("physical,excluded", [([0], True), ([1], False), ([0, 1], False), ([], False)])
def test_foreign_process_is_excluded_only_with_positive_gpu0_attribution(
    serial, monkeypatch, physical, excluded,
):
    enable(serial, monkeypatch)
    state = {
        "host": monitor.HOSTS["node4"], "unit": {"ActiveState": "inactive"},
        "processes": [{"pid": 999, "uid": 1002, "verified_filler": False}],
        "vanished_gpu_pids": [], "verified_unit": False, "filler_kind": None,
    }
    monkeypatch.setattr(resources, "audit", lambda: copy.deepcopy(state))
    outputs = iter([
        "\n".join(f"{index}, GPU-{index}" for index in range(8)),
        "\n".join(f"999, GPU-{index}" for index in physical),
    ])
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    result = resources.scoped_audit()
    assert (not result["processes"]) == excluded
    assert bool(result["resource_scope"]["excluded_gpu_processes"]) == excluded


def test_new_gpu_process_not_in_first_snapshot_is_still_detected(serial, monkeypatch):
    enable(serial, monkeypatch)
    monkeypatch.setattr(resources, "audit", lambda: {
        "host": monitor.HOSTS["node4"], "processes": [], "vanished_gpu_pids": [],
        "verified_unit": False, "filler_kind": None,
    })
    outputs = iter([
        "\n".join(f"{index}, GPU-{index}" for index in range(8)), "999, GPU-1",
    ])
    monkeypatch.setattr(resources.subprocess, "check_output", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(resources, "process_identity", lambda pid: {
        "pid": pid, "uid": 1002, "cmd": ["python", "training.py"], "cgroup": "foreign",
    })
    result = resources.scoped_audit()
    assert result["processes"][0]["pid"] == 999
    assert not result["processes"][0]["verified_filler"]


def test_launcher_passes_physical_subset_and_preserves_cpu_only_children(serial, monkeypatch, tmp_path):
    enable(serial, monkeypatch)
    launcher = object.__new__(Launcher)
    launcher.root = tmp_path / "supervision"
    launcher.root.mkdir()
    launcher.env = resources.scope_environment()
    launcher.children, launcher.owned_identities, launcher.owned_by_root = [], {}, {}
    launcher._audit_lock = threading.Lock()
    launcher.check = Mock()
    process = Mock(pid=os.getpid(), args=["fixture"])
    process.poll.return_value = None
    spawn = Mock(return_value=process)
    monkeypatch.setattr("sao_bridge.launch.subprocess.Popen", spawn)
    launcher.start(["fixture"], gpus="0,1,2,3,4,5,6", name="training")
    assert spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "1,2,3,4,5,6,7"
    launcher.start(["fixture"], gpus="", name="controller")
    assert spawn.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_unscoped_existing_behavior_is_unchanged(serial):
    assert resources.execution_gpu_indices() is None
    assert resources.runtime_gpu_counts({"actor_gpus": 4, "rollout_gpus": 4}) == (4, 4)
    assert resources.logical_gpu_string() == "0,1,2,3,4,5,6,7"
    assert resources.physical_gpu_string("4,5,6,7") == "4,5,6,7"


def test_partial_partition_environment_is_an_explicit_error(serial, monkeypatch):
    monkeypatch.setenv(resources.SCOPE_ENV[0], serial[3]["authority"])
    with pytest.raises(ValueError, match="Incomplete"):
        resources.logical_gpu_string()


def test_native_configuration_changes_only_rollout_capacity(serial, monkeypatch, tmp_path):
    from sao_bridge.large_train import configure

    spec = {
        **role_spec(serial), "model": str(tmp_path / "model"), "phase": "solver",
        "experiment": "partition", "steps": 144, "seed": 1000043,
        "lr": 5e-7, "critic_lr": 2.5e-6, "tokens": 4096, "context_tokens": 32768,
        "joint_root": str(tmp_path / "joint"), "kl_ctl": .02,
        "critic_mode": "learned", "reference_storage_dtype": "bfloat16",
    }
    original = configure(spec, tmp_path / "run")
    enable(serial, monkeypatch)
    partitioned = configure(spec, tmp_path / "run")
    assert partitioned["actor"]["backend"] == original["actor"]["backend"] == "fsdp:d4p1t1"
    assert partitioned["rollout"]["backend"] == "vllm:d3p1t1"
    assert partitioned["cluster"]["n_gpus_per_node"] == 7
    partitioned["rollout"]["backend"] = original["rollout"]["backend"]
    partitioned["cluster"]["n_gpus_per_node"] = original["cluster"]["n_gpus_per_node"]
    env = partitioned["actor"]["scheduling_spec"][0]["env_vars"]
    for name in resources.SCOPE_ENV:
        assert env.pop(name) == os.environ[name]
    assert partitioned == original


def test_serial_worker_evaluates_each_completed_arm_on_full_eight_card_node(serial, monkeypatch):
    from sao_bridge import alfworld_ma_mechanism_program as program
    from sao_bridge import alfworld_ma_rolling_recovery as rolling

    root, control, policy, binding = serial
    session = control / "sessions/one"
    write_json(session / "BINDING.json", binding)
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: policy)
    monkeypatch.setattr(rolling, "admission_snapshot", lambda: {"released": True})
    events = []
    def train(directory, arm, **kwargs):
        assert directory == root and kwargs["execution_authority"] == binding
        events.append(("train", arm))
        return {"attempt": str(root / arm / "attempts/attempt-0001")}
    monkeypatch.setattr(rolling, "run_attempt", train)
    monkeypatch.setattr(monitor, "finish_on_node3",
                        lambda controller, current, arm, training: events.append(("evaluate_node3", arm)))
    local_finish = Mock(side_effect=AssertionError("Must not change endpoint replica count"))
    monkeypatch.setattr(rolling, "finish_arm", local_finish)
    monkeypatch.setattr(program, "write_master_report", lambda _: None)
    monitor.execute_worker(control, session)
    assert events == [event for arm in monitor.ARMS for event in (("train", arm), ("evaluate_node3", arm))]
    local_finish.assert_not_called()
    assert read_json(session / "COMPLETE.json")["status"] == "completed"


@pytest.mark.parametrize("returncode", [-9, -15, 137, 143, 1])
def test_serial_session_never_blindly_retries_kill_or_self_failure(serial, monkeypatch, returncode):
    from sao_bridge import launch

    _, control, _, binding = serial
    process = Mock(returncode=returncode)
    process.poll.return_value = returncode
    owned = Mock()
    owned.module.return_value = process
    monkeypatch.setattr(launch, "Launcher", lambda _: owned)
    monkeypatch.setattr(monitor, "session_attempt", lambda *args: None)
    retry = Mock(side_effect=AssertionError("Killed or undiagnosed failures must not auto-retry"))
    monkeypatch.setattr(monitor, "retry_evidence", retry)
    with pytest.raises(RuntimeError, match="no automatic retry"):
        monitor.run_session(control, binding)
    retry.assert_not_called()
    owned.close.assert_called_once()


def test_immediate_window_records_real_snapshot_without_two_hour_credit(serial, monkeypatch):
    _, control, policy, binding = serial
    snapshot = read_json(Path(binding["admission"]))["final_recheck"]
    snapshot["at_unix"] = monitor.time.time()
    monkeypatch.setattr(monitor, "verify_prepared", lambda _: policy)
    monkeypatch.setattr(monitor, "resource_snapshot", lambda: snapshot)
    new_binding = monitor.wait_for_window(control, binding["authority_sha256"])
    admission = read_json(Path(new_binding["admission"]))
    assert admission["samples"] == 1 and admission["elapsed_seconds"] == 0
    assert admission["first_sample_unix"] == admission["last_sample_unix"] == snapshot["at_unix"]
    assert not (control / "SAMPLES.jsonl").exists()


def test_runtime_contains_and_verifies_all_native_upstream_patches():
    from sao_bridge.common import verify_upstreams

    result = verify_upstreams()
    for name in ("areal_local_network", "areal_packed_sdpa",
                 "areal_assistant_prefill", "areal_qwen35_critic"):
        assert len(result[name + "_sha256"]) == 64
