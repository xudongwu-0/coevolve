from datetime import timedelta
import os
from pathlib import Path
import socket
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))


def collective_worker(rank, ports):
    import torch
    import torch.distributed as dist
    from areal.engine.core.distributed import init_custom_process_group
    from areal.utils.network import gethostip

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://{gethostip()}:{ports[0]}", rank=rank, world_size=2,
        timeout=timedelta(seconds=60),
    )
    group = init_custom_process_group(
        backend="nccl", init_method=f"tcp://{gethostip()}:{ports[1]}",
        rank=rank, world_size=2, group_name="sao_preflight",
        timeout=timedelta(seconds=60),
    )
    value = torch.tensor([rank + 1.], device=f"cuda:{rank}")
    dist.all_reduce(value, group=group)
    assert value.item() == 3.0
    dist.destroy_process_group(group)
    dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("SAO_GPU_INTEGRATION") == "1", "explicit two-GPU integration test")
class DistributedTests(unittest.TestCase):
    def test_native_custom_weight_transfer_group(self):
        import torch.multiprocessing as mp
        from sao_bridge.resources import stop_known

        stop_known(require_exclusive=True)
        ports = []
        for _ in range(2):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                ports.append(sock.getsockname()[1])
        mp.spawn(collective_worker, args=(ports,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
