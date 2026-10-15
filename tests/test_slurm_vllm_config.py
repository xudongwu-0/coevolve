"""CPU checks against the installed, pinned AReaL/vLLM configuration interface."""
from dataclasses import asdict
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))
sys.path.insert(0, str(ROOT / "third_party/sao/vendor/AReaL"))
pytest.importorskip("torch")
pytest.importorskip("areal")

from areal.api.cli_args import vLLMConfig
from sao_bridge.joint_workflow import JointVLLMConfig
from sao_bridge.slurm_vllm import SlurmVLLMConfig


def test_cli_keeps_processed_logprobs_and_forbids_shared_memory():
    config = SlurmVLLMConfig(model="model", max_model_len=32768)
    args = vLLMConfig.build_args(config, tp_size=1, pp_size=1)
    assert args["distributed_executor_backend"] == "uni"
    assert args["mm_processor_cache_type"] == "lru"
    assert args["mm_tensor_ipc"] == "direct_rpc"
    assert args["logprobs_mode"] == "processed_logprobs"
    assert args["max_model_len"] == 32768
    assert args["tensor_parallel_size"] == args["pipeline_parallel_size"] == 1
    assert SlurmVLLMConfig(**asdict(config)) == config


@pytest.mark.parametrize("override", [
    {"distributed_executor_backend": "mp"},
    {"mm_processor_cache_type": "shm"},
    {"mm_tensor_ipc": "torch_shm"},
    {"logprobs_mode": "raw_logprobs"},
])
def test_rejects_unsafe_or_wrong_probability_settings(override):
    with pytest.raises(ValueError):
        SlurmVLLMConfig(**override)


def test_legacy_config_is_unchanged():
    legacy = JointVLLMConfig()
    assert legacy.distributed_executor_backend == "mp"
    assert "mm_tensor_ipc" not in asdict(legacy)


def test_torchrun_cannot_consume_application_run_and_role_options():
    from torch.distributed.run import parse_args

    args = parse_args([
        "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0",
        "-m", "--", "sao_bridge.ma_slurm", "collective", "--run", "/fixture",
        "--role", "editor",
    ])
    assert args.module is True
    assert args.training_script == "sao_bridge.ma_slurm"
    assert args.training_script_args == ["collective", "--run", "/fixture", "--role", "editor"]
