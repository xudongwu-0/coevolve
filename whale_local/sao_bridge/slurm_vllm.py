"""Single-GPU rollout settings for the root-confined Slurm comparison."""
from dataclasses import dataclass

from .joint_workflow import JointVLLMConfig


@dataclass
class SlurmVLLMConfig(JointVLLMConfig):
    distributed_executor_backend: str = "uni"
    mm_processor_cache_type: str = "lru"
    mm_tensor_ipc: str = "direct_rpc"

    def __post_init__(self):
        super().__post_init__()
        if self.distributed_executor_backend != "uni":
            raise ValueError("The Slurm comparison requires one GPU per uniprocess replica")
        if self.mm_processor_cache_type != "lru" or self.mm_tensor_ipc != "direct_rpc":
            raise ValueError("The Slurm comparison does not permit shared-memory multimodal IPC")
