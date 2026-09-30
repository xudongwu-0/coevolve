import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from harness_r1_searchqa import run_grpo_pilot
from harness_r1_searchqa.grpo_train import rollout_environment, validate_config


def test_independent_rollout_does_not_join_torchrun_agent_store(monkeypatch):
    inherited = {
        "TORCHELASTIC_USE_AGENT_STORE": "True", "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_RUN_ID": "actor-run", "TORCHELASTIC_ERROR_FILE": "actor-error",
        "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4",
        "GROUP_RANK": "0", "ROLE_RANK": "0", "ROLE_WORLD_SIZE": "4",
        "MASTER_ADDR": "localhost", "MASTER_PORT": "12345",
    }
    for key, value in inherited.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("NCCL_SOCKET_IFNAME", "lo")
    environment = rollout_environment("4,5,6,7")
    assert not inherited.keys() & environment.keys()
    assert environment["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert environment["NCCL_SOCKET_IFNAME"] == "lo"
    assert environment["HARNESS_R1_REGISTER_VLLM_TEXT"] == "1"


def test_tp1_recovery_preserves_learning_and_proposal_budgets():
    root = Path(__file__).resolve().parents[2] / "configs"
    original = json.loads((root / "searchqa_engineer_grpo_fast4b.json").read_text())
    recovery = validate_config(
        json.loads((root / "searchqa_engineer_grpo_fast4b_tp1.json").read_text())
    )
    differing = {key for key in original if original[key] != recovery[key]}
    assert differing == {"rollout_tp", "rollout_timeout"}
    assert recovery["rollout_tp"] == 1
    assert recovery["rollout_timeout"] == 900


@pytest.mark.parametrize("rollout_gpus", [None, "4"])
def test_runner_forwards_explicit_rollout_topology(monkeypatch, tmp_path, rollout_gpus):
    launch = Mock()
    monkeypatch.setattr(run_grpo_pilot, "Launcher", Mock(return_value=launch))
    monkeypatch.setattr(
        run_grpo_pilot, "verify_grpo_commit",
        Mock(return_value={"optimizer_steps": 4, "actor_sha256": "verified"}),
    )
    output = tmp_path / "run"
    argv = [
        "pilot", "--model", "model", "--data", "data", "--reward-config", "reward",
        "--config", "config", "--output", str(output),
    ]
    if rollout_gpus is not None:
        argv += ["--rollout-gpus", rollout_gpus]
    monkeypatch.setattr(run_grpo_pilot.sys, "argv", argv)
    run_grpo_pilot.main()
    command = launch.start.call_args.args[0]
    assert command[command.index("--rollout-gpus") + 1] == (rollout_gpus or "4,5,6,7")
    assert launch.start.call_args.kwargs["gpus"] == "0,1,2,3"
    assert json.loads((output / "STATUS.json").read_text())["optimizer_steps"] == 4
    launch.close.assert_called_once()
