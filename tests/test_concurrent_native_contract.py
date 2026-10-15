"""CPU integration tests for the staged checkout with its pinned AReaL tree."""
import copy
import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
sys.path.insert(0, str(ROOT / "third_party/sao/vendor/AReaL"))
pytest.importorskip("areal")

from sao_bridge.concurrent_comparison import ARMS, ROLES, role_spec
from sao_bridge.joint_critic import JointEpisodeCritic
from sao_bridge.joint_recovery import (
    SCHEMA, commit_pair, file_inventory, science_digest, validate_recovery_spec,
)
from sao_bridge.joint_schedule import checkpoint_metadata, role_active, update_counts
from sao_bridge.large_train import configure
from sao_bridge.common import file_sha, write_json
from areal.engine.fsdp_engine import FSDPPPOCritic


def spec(root, arm="ma_sao", role="editor"):
    return role_spec({
        "run": str(root), "engineer_model": str(root / "harness-r1"),
        "solver_model": str(root / "target-agent-sft"), "no_patch": str(root / "empty.json"),
        "training_deadline_unix": 1e12,
    }, arm, role, root / "data.json", {})


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("role", ROLES)
def test_native_config_keeps_one_sequence_microbatch_and_two_accumulations(tmp_path, arm, role):
    value = spec(tmp_path, arm, role)
    config = configure(value, Path(value["run"]))
    assert config["rollout"]["max_concurrent_rollouts"] == 8
    assert config["vllm"]["max_num_seqs"] == 2
    assert config["train_dataset"]["batch_size"] == 8
    for engine in ("actor", "critic", "ref"):
        assert config[engine]["mb_spec"]["n_mbs"] == 2
        assert config[engine]["fsdp"]["memory_efficient_load"] is True
        assert config[engine]["fsdp"]["offload_params"] is True
        assert config[engine]["fsdp"]["per_layer_optim_step"] is False
    assert config["actor"]["mb_spec"]["max_tokens_per_mb"] == 32768
    assert config["gconfig"]["max_tokens"] == 32768
    assert config["gconfig"]["max_new_tokens"] == 4096
    assert config["saver"]["freq_steps"] == 36
    assert config["actor"]["scheduling_spec"][0]["env_vars"]["MA_SAO_FRESH_CRITIC_SEED"] == "1000043"
    validate_recovery_spec(value)
    assert all(role_active(value, index) for index in range(144))
    assert update_counts(value, 144)["actor_steps"] == 141
    assert update_counts(value, 144)["critic_steps"] == 288


def test_fresh_head_cpu_rng_is_reproducible_and_does_not_advance_actor_rng(monkeypatch):
    monkeypatch.setenv("MA_SAO_FRESH_CRITIC_SEED", "1000043")
    monkeypatch.setattr(FSDPPPOCritic, "_create_device_model", lambda self: torch.randn(32))
    critic = JointEpisodeCritic.__new__(JointEpisodeCritic)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    first = critic._create_device_model()
    assert torch.equal(before, torch.random.get_rng_state())
    torch.rand(100)
    second = critic._create_device_model()
    assert torch.equal(first, second)


def test_joint_entry_blocks_comparison_before_writes_or_launch(tmp_path, monkeypatch):
    from sao_bridge import joint_experiment

    value = spec(tmp_path / "not-created")
    monkeypatch.setattr(joint_experiment, "require_research_node", lambda: None)
    monkeypatch.setattr(joint_experiment, "read_json", lambda path: value)
    with pytest.raises(NotImplementedError, match="ten-task"):
        joint_experiment.run_role(tmp_path / "spec.json")
    assert not (tmp_path / "not-created").exists()


def payload(value, completed, generation):
    role = value["role"]
    destination = generation / role
    destination.mkdir(parents=True)
    write_json(generation / "OWNER.json", {
        "schema": SCHEMA, "root": value["rolling_recovery"]["root"],
        "generation": generation.name, "completed_rounds": completed,
    })
    write_json(destination / "spec.json", value)
    (destination / "metrics.jsonl").write_text("".join(
        json.dumps({"global_step": i, "metrics": {}}) + "\n" for i in range(completed)))
    write_json(destination / "controller.json", {})
    counts = update_counts(value, completed)
    for engine in ("actor", "critic"):
        directory = destination / engine
        directory.mkdir()
        (directory / ".metadata").write_bytes(b"synthetic metadata")
        for rank in range(4):
            (directory / f"__{rank}_0.distcp").write_bytes(b"synthetic model")
            (directory / f"optim_world_size_4_rank_{rank}.pt").write_bytes(b"synthetic Adam")
            (directory / f"training-rank{rank}.pt").write_bytes(b"synthetic RNG")
            write_json(directory / f"adam-rank{rank}.json", {
                "min_adam_step": counts[f"{engine}_steps"],
                "max_adam_step": counts[f"{engine}_steps"],
            })
        write_json(directory / "joint_schedule.json", checkpoint_metadata(value, completed))
        if engine == "critic":
            write_json(directory / "joint_critic_settings.json", {
                "cooperative_context": value["critic_cooperative_context"],
                "value_normalization": value["value_normalization"],
            })
            if value["value_normalization"]:
                write_json(directory / "joint_value_norm.json", {"updates": counts["critic_steps"]})
    write_json(generation / f"{role}.json", {
        "schema": SCHEMA, "role": role, "completed_rounds": completed,
        "batch_size": 8, "committed_sample_cursor": completed * 8,
        "science_sha256": science_digest(value), "data_sha256": file_sha(Path(value["data"])),
        "source_run": value["run"], "source_joint": value["joint_root"],
        "protected_hf_relative_paths": [], "files": file_inventory(destination),
    })


@pytest.mark.parametrize("arm", ARMS)
def test_pair36_then72_retains_both_full_native_generations(tmp_path, arm):
    write_json(tmp_path / "data.json", [])
    value = spec(tmp_path, arm)
    root = Path(value["rolling_recovery"]["root"])
    generations = []
    for step in (36, 72):
        generation = root / f"round-{step:04d}-attempt-0001"
        for role in ROLES:
            payload(spec(tmp_path, arm, role), step, generation)
        assert commit_pair(root, generation, step)
        assert commit_pair(root, generation, step, role="editor")
        generations.append(generation)
    assert all((generation / "editor/actor/.metadata").exists() for generation in generations)
    assert json.loads((root / "LATEST.json").read_text())["completed_rounds"] == 72


def test_historical_recovery_still_rejects36_without_new_comparison(tmp_path):
    value = copy.deepcopy(spec(tmp_path))
    value.pop("comparison_schema")
    with pytest.raises(ValueError, match="cadence"):
        validate_recovery_spec(value)
