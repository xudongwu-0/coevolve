import os
from pathlib import Path
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pytest

from sao_bridge.common import file_sha
from sao_bridge.whale_parallel_sft import parallel_sft_update, rank_state_hashes, shard_calls


def _parallel_worker(rank, root):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from sao_bridge import alfworld_whale_baseline as whale

    sys.path.insert(0, str(whale.NATIVE))
    torch.set_num_threads(1)
    dist.init_process_group("gloo", rank=rank, world_size=2,
                            init_method=(root / "rendezvous").as_uri())

    class Actor:
        def __init__(self):
            torch.manual_seed(932)
            self.actor_module = FSDP(torch.nn.Linear(3, 5), device_id=torch.device("cpu"),
                                     use_orig_params=True)
            self.actor_optimizer = torch.optim.AdamW(self.actor_module.parameters(),
                                                     lr=.003, weight_decay=.01, foreach=False)
            self.config = SimpleNamespace(use_dynamic_bsz=False)
            self.scaler = None

        def _forward_micro_batch(self, batch, temperature, calculate_entropy=False):
            width = batch["responses"].shape[1]
            prompt_width = batch["input_ids"].shape[1] - width
            prompt = batch["input_ids"][0, :prompt_width][batch["attention_mask"][0, :prompt_width].bool()]
            mask = batch["attention_mask"][0, prompt_width:].bool()
            response = batch["responses"][0][mask]
            features = torch.stack((response.float() / 17, torch.ones_like(response).float(),
                                    prompt.float().mean().expand_as(response) / 11), dim=-1)
            logits = self.actor_module(features)
            values = logits.log_softmax(-1).gather(-1, (response % 5).unsqueeze(-1)).squeeze(-1)
            return {"log_probs": torch.nn.functional.pad(values, (0, width - len(values)))[None]}

        def _optimizer_step(self):
            norm = self.actor_module.clip_grad_norm_(1.)
            self.actor_optimizer.step()
            return norm

    try:
        for count in (1, 3, 5, 7):
            calls = [{"prompt_ids": [1, index + 2], "response_ids": [3 + index] * (index % 4 + 1)}
                     for index in range(count)]
            reference, candidate = Actor(), Actor()
            full = whale.native_batch(calls, 0, 4)
            reference_loss = 0.
            for mini in full.split(min(4, count)):
                denominator = mini.batch["response_mask"].sum()
                reference.actor_optimizer.zero_grad()
                for micro in mini.split(1):
                    values = reference._forward_micro_batch(micro.batch, 1.)["log_probs"]
                    loss = -torch.where(micro.batch["response_mask"].bool(), values, 0).sum() / denominator
                    loss.backward()
                    reference_loss += float(loss.detach())
                reference._optimizer_step()
            local, active = shard_calls(calls, rank, 2)
            batch = whale.native_batch(local, 0, 2)
            batch.batch["response_mask"][torch.tensor([not value for value in active])] = 0
            result = parallel_sft_update(candidate, batch, torch.device("cpu"))
            assert abs(result["actor/sft_loss"] - reference_loss) < 2e-6
            assert result["actor/sft_token_count"] == sum(len(row["response_ids"]) for row in calls)
            assert result["actor/sft_updates"] == (count + 3) // 4
            for left, right in zip(reference.actor_module.parameters(), candidate.actor_module.parameters(), strict=True):
                torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)
                left_state = reference.actor_optimizer.state[left]
                right_state = candidate.actor_optimizer.state[right]
                assert left_state.keys() == right_state.keys()
                for key in left_state:
                    torch.testing.assert_close(left_state[key], right_state[key], rtol=2e-5, atol=2e-7)
            del reference, candidate
        (root / f"rank-{rank}-passed").write_text("parameters, Adam, token mean, and tail padding matched\n")
    finally:
        dist.destroy_process_group()


def test_true_sample_parallelism_preserves_parameters_adam_and_unequal_token_mean(tmp_path):
    import torch.multiprocessing as mp

    mp.spawn(_parallel_worker, args=(tmp_path,), nprocs=2, join=True)
    assert all((tmp_path / f"rank-{rank}-passed").exists() for rank in range(2))


def test_sharding_preserves_global_minibatches_and_only_pads_missing_tail_calls():
    calls = [{"prompt_ids": [1], "response_ids": [index + 2]} for index in range(10)]
    shards = [shard_calls(calls, rank, 4) for rank in range(4)]
    assert all(len(rows) == 3 for rows, _ in shards)
    actual = [rows[step] for step in range(3) for rows, active in shards if active[step]]
    assert actual == calls
    assert sum(not value for _, active in shards for value in active) == 2
    with pytest.raises(ValueError):
        shard_calls([], 0, 4)
    with pytest.raises(ValueError):
        shard_calls(calls, 4, 4)


def test_parallel_checkpoint_hashing_matches_every_original_file(tmp_path):
    for rank in range(4):
        (tmp_path / f"rank-{rank:04d}.pt").write_bytes(bytes([rank]) * (129 + rank))
    expected = {str(rank): file_sha(tmp_path / f"rank-{rank:04d}.pt") for rank in range(4)}
    assert rank_state_hashes(tmp_path, 4) == expected
    with pytest.raises(FileNotFoundError):
        rank_state_hashes(tmp_path, 5)
