from pathlib import Path
import sys
import time
from unittest.mock import patch

import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from areal.api.cli_args import PPOConfig, load_expr_config
from areal.infra.workflow_executor import check_trajectory_format
from areal.utils.data import allocate_balanced_mbs, batched_call
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.joint_campaign import (
    available_pairs, initial_inputs, prepare, recover_failed_role,
    select_pair, share_checkpoints, wait_comparison,
)
from sao_bridge.joint_health import assess, metric_rows, recover_checkpoints, request_stop
from sao_bridge.large_train import configure


def training_spec(tmp_path):
    return {
        "role": "editor", "model": str(tmp_path), "steps": 128, "batch_size": 8,
        "lr": 5e-7, "kl_ctl": 0.005, "phase": "editor", "experiment": "joint",
        "seed": 42, "tokens": 4096, "context_tokens": 32768,
        "actor_gpus": 4, "rollout_gpus": 4, "joint_root": str(tmp_path / "joint"),
        "critic_warmup_steps": 2, "terminal_episodes": True, "save_interval_steps": 4,
        "run": str(tmp_path / "optimizer"),
    }


def metrics(step, kl=0.01):
    return {
        "global_step": step,
        "metrics": {
            "ppo_actor/update/grad_norm": 1.,
            "joint_policy/fixed_reference_kl": kl,
            "joint_policy/fixed_reference_available": 1.,
            "joint_policy/peer_importance_clipped_fraction": 0.,
            "joint_policy/reference_kl_overflow": 0.,
            "joint_value_norm/raw_value_mse": 0.2,
        },
    }


def publish_episodes(spec, step):
    current = Path(spec["joint_root"]) / f"round-{step:04d}"
    for slot in range(8):
        write_json(current / f"proposal-{slot:03d}.json", {"model_error": None})
        write_json(current / f"editor-receipt-{slot:03d}.json", {
            "reward": 1., "calls": [{"response_ids": [4]}],
        })


def test_reference_and_proximal_probabilities_have_same_resolved_temperature(tmp_path, monkeypatch):
    monkeypatch.setenv("SAO_LOCAL_ADMIN_TOKEN", "test-only")
    spec = training_spec(tmp_path)
    config = configure(spec, tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    resolved, _ = load_expr_config(["--config", str(path)], PPOConfig)
    assert resolved.gconfig.temperature == resolved.actor.temperature == resolved.ref.temperature == 0.7
    assert resolved.actor.eps_clip == resolved.actor.eps_clip_higher == 0.2
    assert resolved.ref.optimizer is None
    assert resolved.actor.should_compute_prox_logp()
    for component in (resolved.actor, resolved.critic, resolved.ref):
        assert component.mb_spec.n_mbs == 2
        for lengths in ([6000, 10000], [16, 18000], [31000, 32000]):
            groups = allocate_balanced_mbs(component.mb_spec, lengths)
            assert sorted(groups) == [[0], [1]]


def test_joint_round_and_slot_survive_native_batching(tmp_path):
    def row(slot, length):
        return {
            "input_ids": torch.ones(1, length, dtype=torch.long),
            "attention_mask": torch.ones(1, length, dtype=torch.bool),
            "loss_mask": torch.ones(1, length, dtype=torch.bool),
            "logprobs": torch.zeros(1, length),
            "versions": torch.zeros(1, length, dtype=torch.long),
            "rewards": torch.tensor([1.]),
            "joint_round": torch.tensor([2]), "joint_slot": torch.tensor([slot]),
        }
    rows = [row(3, 5), row(4, 8)]
    assert all(check_trajectory_format(r) for r in rows)
    changed = batched_call(lambda data: data, rows)
    assert [int(r["joint_round"].item()) for r in changed] == [2, 2]
    assert [int(r["joint_slot"].item()) for r in changed] == [3, 4]


def test_health_requires_consecutive_soft_warnings_and_reports_nonfinite(tmp_path):
    spec = training_spec(tmp_path)
    publish_episodes(spec, 2)
    report = assess([metrics(0), metrics(1), metrics(2, 0.2)], spec)
    assert report["status"] == "healthy"
    report = assess([metrics(i, 0.2) for i in range(3)], spec)
    assert report["status"] == "stop"
    assert report["latest_team_success"] == 1
    clipped = [metrics(i) for i in range(3)]
    for row in clipped:
        row["metrics"]["joint_policy/peer_importance_clipped_fraction"] = 1.
    report = assess(clipped, spec)
    assert report["status"] == "warning"
    assert report["warnings"] and not report["stop_reasons"]
    clipped[-1]["metrics"]["joint_policy/fixed_reference_available"] = 0.
    assert assess(clipped, spec)["status"] == "stop"
    bad = metrics(2)
    bad["metrics"]["ppo_actor/update/grad_norm"] = float("nan")
    assert assess([metrics(0), metrics(1), bad], spec)["hard_errors"]
    missing = metrics(2)
    del missing["metrics"]["joint_policy/fixed_reference_kl"]
    with pytest.raises(ValueError, match="Missing or ambiguous"):
        assess([metrics(0), metrics(1), missing], spec)


def test_stop_preserves_first_reason_and_partial_metrics_are_not_committed(tmp_path):
    first = request_stop(tmp_path, {"reason": "deadline"})
    assert request_stop(tmp_path, {"reason": "later"}) == first
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"global_step":0}\n{"global_step":')
    assert metric_rows(path) == [{"global_step": 0}]


def test_recovery_only_uses_fully_committed_pairs_and_normalizer(tmp_path):
    spec = training_spec(tmp_path)
    root = Path(spec["run"]) / "engine/checkpoints/user/joint/editor"
    for role in ("actor", "critic"):
        for step in (3, 7):
            path = root / role / f"epoch0epochstep{step}globalstep{step}"
            write_json(path / "config.json", {})
            (path / "model.safetensors").write_bytes(b"test")
            if role == "critic":
                write_json(path / "joint_value_norm.json", {"updates": 4})
    with patch("sao_bridge.joint_health.ensure_model_runtime_metadata"):
        assert list(recover_checkpoints(spec, [metrics(i) for i in range(4)])) == ["3"]
        (root / "critic/epoch0epochstep3globalstep3/joint_value_norm.json").unlink()
        with pytest.raises(ValueError, match="running statistics"):
            recover_checkpoints(spec, [metrics(i) for i in range(4)])


def test_selection_never_promotes_no_patch_or_uses_200_scores():
    reports = {
        "11": {"no_patch": {"accuracy": 1.}, "proposal00": {"accuracy": 0.8}},
        "15": {"no_patch": {"accuracy": 0.9}, "proposal01": {"accuracy": 0.8}},
    }
    assert select_pair(reports) == ("11", "proposal00")
    reports["15"]["proposal01"]["accuracy"] = 0.9
    assert select_pair(reports) == ("15", "proposal01")


def test_required_grpo_failure_prevents_silent_ma_start(tmp_path):
    write_json(tmp_path / "w1_FAILED.json", {"error": "reward infrastructure"})
    with pytest.raises(RuntimeError, match="Required GRPO"):
        wait_comparison({"comparison": str(tmp_path), "training_deadline_unix": time.time() + 3600})


def test_cross_uid_sharing_only_exposes_checkpoints_not_runtime_control_files(tmp_path):
    optimizer = tmp_path / "optimizer"
    checkpoint = optimizer / "engine/checkpoints/actor/step3"
    checkpoint.mkdir(parents=True, mode=0o700)
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights")
    weights.chmod(0o600)
    private = optimizer / "runtime-control"
    private.write_text("private runtime metadata")
    private.chmod(0o600)
    result = {"checkpoint_history": {"3": {"actor": str(checkpoint)}}}
    share_checkpoints(optimizer, result)
    assert weights.stat().st_mode & 0o004
    assert checkpoint.stat().st_mode & 0o005 == 0o005
    assert private.stat().st_mode & 0o077 == 0
    with pytest.raises(ValueError, match="outside"):
        share_checkpoints(optimizer, {"checkpoint_history": {"3": {"actor": str(tmp_path)}}})


def test_initial_inputs_use_only_original_models_and_w0_evidence(tmp_path):
    original = {
        "data": str(tmp_path / "old/train.json"), "engineer_model": "official-h0",
        "solver_model": "official-w0", "solver_version": "w0-version", "harness": "empty",
    }
    packets = [{
        "metadata": {"task_ids": list(range(start, start + 10))},
        "messages": [{"role": "user", "content": f"W0 evidence batch{start}"}],
    } for start in range(0, 240, 10)]
    with (
        patch("sao_bridge.joint_campaign.read_json", side_effect=[original, packets]),
        patch("sao_bridge.joint_campaign.file_sha", return_value="frozen-hash"),
        patch("sao_bridge.joint_campaign.validate_metadata") as validate,
        patch("sao_bridge.alfworld_train_data.evaluation_paths", return_value=[f"task{i}" for i in range(500)]),
    ):
        rows, state = initial_inputs(tmp_path, 128, 8)
    assert len(rows) == 1024 and len({r["task_id"] for r in rows}) == 240
    assert validate.call_count == 24
    assert all(call.args[1] == "w0-version" for call in validate.call_args_list)
    assert state["engineer"] == state["engineer_critic"] == "official-h0"
    assert state["solver"] == state["solver_critic"] == "official-w0"
    assert state["solver_epoch_version"] == 0
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        assert len({r["task_id"] for r in batch}) == 8
        assert [r["joint_slot"] for r in batch] == list(range(8))
        for row in batch:
            assert row["joint_version"] == start // 8
            assert row["task_context"] == f"task{row['task_id']}"
            assert row["messages"][0]["content"].startswith("W0 evidence")
            assert "W0" in row["critic_evidence"] and "W79" not in row["critic_evidence"]


def test_initial_prerequisites_reject_allocator_only_smoke_test(tmp_path):
    memory = tmp_path / "memory/MEMORY_COMPLETE.json"
    protocol = {
        "run": str(tmp_path), "starting_point": "h0_w0",
        "memory_check": str(memory), "training_deadline_unix": time.time() + 60,
    }
    write_json(tmp_path / "initial_COMPLETE.json", {"status": "completed"})
    check = {
        "status": "passed", "max_sequence_tokens": 32768,
        "actor_optimizer_steps": 0, "finite_nonzero_actor_gradient": False,
        "source_sha256": {
            "whale_local/sao_bridge/terminal_actor.py": file_sha(
                Path(__file__).resolve().parents[1] / "whale_local/sao_bridge/terminal_actor.py"
            ),
        },
    }
    write_json(memory, check)
    with pytest.raises(ValueError, match="native Actor"):
        wait_comparison(protocol)
    write_json(memory, {**check, "actor_optimizer_steps": 1, "finite_nonzero_actor_gradient": True})
    wait_comparison(protocol)
    assert read_json(tmp_path / "TRAINING_PREREQUISITES.json")["status"] == "passed"
    write_json(memory, {
        **check, "actor_optimizer_steps": 1, "finite_nonzero_actor_gradient": True,
        "source_sha256": {"whale_local/sao_bridge/terminal_actor.py": "different-runtime"},
    })
    with pytest.raises(ValueError, match="executed learner"):
        wait_comparison(protocol)
    write_json(memory, {**check, "max_sequence_tokens": 24075, "actor_optimizer_steps": 1,
                        "finite_nonzero_actor_gradient": True})
    with pytest.raises(ValueError, match="native Actor"):
        wait_comparison(protocol)


def test_initial_prerequisite_failure_is_not_a_zero_score(tmp_path):
    protocol = {
        "run": str(tmp_path), "starting_point": "h0_w0",
        "memory_check": str(tmp_path / "memory/MEMORY_COMPLETE.json"),
        "training_deadline_unix": time.time() + 60,
    }
    write_json(tmp_path / "initial_FAILED.json", {"error": "environment failure"})
    with pytest.raises(RuntimeError, match="prerequisite failed"):
        wait_comparison(protocol)


def test_fresh_campaign_resets_models_prefix_and_matched_budget(tmp_path):
    from sao_bridge import joint_campaign

    state = {
        "engineer": "h0", "solver": "w0", "packets": str(tmp_path / "packets.json"),
        "packets_sha256": "unused-by-mocked-validation",
    }
    write_json(tmp_path / "INITIAL_STATE.json", state)
    rows = [{"task_id": i % 240, "joint_version": i // 8, "joint_slot": i % 8}
            for i in range(1024)]
    original_read = read_json

    def read(path):
        path = Path(path)
        if path == joint_campaign.BASE_PROTOCOL:
            return {"no_patch": "empty"}
        if path == joint_campaign.EVAL_PROTOCOL:
            return {"all_task_ids": [240]}
        if path.name in ("editor_spec.json", "solver_spec.json"):
            return {**training_spec(tmp_path), "role": path.name.split("_")[0]}
        if path.name == "data.json" and "alfworld_joint_ma_pilot_v1" in path.parts:
            return rows[:8]
        return original_read(path)

    with (
        patch("sao_bridge.joint_campaign.read_json", side_effect=read),
        patch("sao_bridge.joint_campaign.initial_inputs", return_value=(rows, state)),
        patch("sao_bridge.joint_campaign.validate_cohort"),
        patch("sao_bridge.joint_campaign.validate_spec"),
        patch("sao_bridge.joint_campaign.file_sha", return_value="frozen"),
    ):
        protocol = prepare(tmp_path, tmp_path / "unused-grpo", time.time() + 8 * 3600,
                           from_initial=True, memory_check=tmp_path / "memory.json")
    assert protocol["starting_point"] == "h0_w0"
    assert protocol["budget"]["common_executed_trajectories"] == 0
    assert protocol["budget"]["matched_rounds"] == 64
    assert protocol["budget"]["matched_policy_trajectories"] == 1024
    assert protocol["budget"]["checkpoint_trajectories"] == [192, 832, 1024]
    assert "matched_successful_h2_rounds" not in protocol["budget"]
    for role, model in (("editor", "h0"), ("solver", "w0")):
        spec = read_json(tmp_path / f"{role}_spec.json")
        assert spec["model"] == spec["critic_model"] == spec["reference_model"] == model
        assert spec["critic_warmup_steps"] == 5
        assert str(tmp_path / "INITIAL_STATE.json") in spec["pinned_files"]
        assert str(joint_campaign.INITIAL_STATE) not in spec["pinned_files"]


def test_failed_training_only_publishes_committed_models_without_hiding_failure(tmp_path):
    spec = {
        **training_spec(tmp_path), "role": "solver", "critic_warmup_steps": 2,
        "critic_updates_per_step": 2,
    }
    checkpoint = {"3": {"actor": "actor3", "critic": "critic3"}}
    with (
        patch("sao_bridge.joint_campaign.metric_rows", return_value=[metrics(i) for i in range(6)]),
        patch("sao_bridge.joint_campaign.recover_checkpoints", return_value=checkpoint),
        patch("sao_bridge.joint_campaign.file_sha", return_value="metrics-hash"),
    ):
        result = recover_failed_role(spec)
    assert result["status"] == "recovered_committed_checkpoints_after_training_failure"
    assert result["consumed_policy_trajectories"] == 48
    assert result["checkpoint_history"] == checkpoint
    assert result["actor_updates"] == 4
    assert result["failed_training_not_relabelled_completed"]
    write_json(tmp_path / "protocol.json", {
        "starting_point": "h0_w0", "report_deadline_unix": time.time() + 30,
    })
    write_json(tmp_path / "solver_FAILED.json", {"error": "original OOM"})
    for role in ("editor", "solver"):
        write_json(tmp_path / f"{role}_SHARED.json", {"ready": True, "result": result})
    pairs, _ = available_pairs(tmp_path)
    assert list(pairs) == ["3"]
    assert read_json(tmp_path / "solver_FAILED.json")["error"] == "original OOM"


def test_failed_recovery_prevents_indefinite_publication_wait(tmp_path):
    write_json(tmp_path / "protocol.json", {
        "starting_point": "h0_w0", "report_deadline_unix": time.time() + 30,
    })
    write_json(tmp_path / "solver_FAILED.json", {"error": "original OOM"})
    write_json(tmp_path / "solver_RECOVERY_FAILED.json", {"error": "no committed model"})
    with pytest.raises(RuntimeError, match="no committed model"):
        available_pairs(tmp_path)
