from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from sao_bridge.joint_critic import JointValueNorm, official_value_norm


def sample():
    return {
        "returns": torch.tensor([[999., 0., 1., -999.]]),
        "values": torch.tensor([[100., 0.25, 0.75, -100.]]),
        "loss_mask": torch.tensor([[0, 1, 1, 0]]),
    }


def test_official_statistics_ignore_prompt_and_padding_and_keep_old_clip_units():
    norm = JointValueNorm()
    row = sample()
    raw = norm.denormalize([row["values"]])
    torch.testing.assert_close(raw[0], row["values"])
    changed, stats = norm.prepare_update([row])
    official = official_value_norm()(1)
    official.update(torch.tensor([[0.], [1.]]))
    mean, var = official.running_mean_var()
    assert norm.moments() == (mean.item(), var.sqrt().item())
    torch.testing.assert_close(changed[0]["returns"][0, 1:3], torch.tensor([-1., 1.]))
    torch.testing.assert_close(changed[0]["values"], row["values"])
    assert stats["raw_value_mse"] == 0.0625
    assert stats["active_value_tokens"] == 2
    assert stats["explained_variance_defined"] == 1
    assert stats["explained_variance"] == 0.75
    assert row["returns"][0, 0] == 999


def test_next_rollout_is_denormalized_for_gae_and_multiple_critic_updates_keep_clip_anchor():
    norm = JointValueNorm()
    norm.denormalize([sample()["values"]])
    norm.prepare_update([sample()])
    predicted = torch.tensor([[0., -1., 1., 0.]])
    raw = norm.denormalize([predicted])[0]
    torch.testing.assert_close(raw[0, 1:3], torch.tensor([0., 1.]))
    row = {**sample(), "values": raw}
    first, _ = norm.prepare_update([row])
    second, _ = norm.prepare_update([row])
    torch.testing.assert_close(first[0]["values"], predicted)
    torch.testing.assert_close(second[0]["values"], predicted)


def test_checkpoint_round_trip_preserves_exact_normalization():
    norm = JointValueNorm()
    norm.denormalize([sample()["values"]])
    norm.prepare_update([sample()])
    restored = JointValueNorm()
    restored.restore(norm.state())
    assert restored.state() == norm.state()
    torch.testing.assert_close(restored.denormalize([sample()["values"]])[0],
                               norm.denormalize([sample()["values"]])[0])


def test_invalid_and_constant_targets_are_reported_explicitly():
    norm = JointValueNorm()
    with pytest.raises(FloatingPointError):
        norm.prepare_update([{**sample(), "loss_mask": torch.zeros(1, 4)}])
    bad = sample()
    bad["returns"][0, 1] = float("nan")
    with pytest.raises(FloatingPointError):
        norm.prepare_update([bad])
    row = sample()
    row["returns"].fill_(1)
    _, stats = norm.prepare_update([row])
    assert stats["explained_variance_defined"] == 0
    for field in ("behavior_mean", "behavior_std"):
        state = {**norm.state(), field: float("inf")}
        with pytest.raises(ValueError, match="checkpoint moments"):
            norm.restore(state)


def _distributed_value_norm_worker(rank, rendezvous):
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        row = {
            "returns": torch.tensor([[float(rank)] * 3 + [999.]]),
            "values": torch.full((1, 4), 0.5),
            "loss_mask": torch.tensor([[1, rank, rank, 0]]),
        }
        norm = JointValueNorm()
        norm.denormalize([row["values"]])
        _, stats = norm.prepare_update([row], group=dist.group.WORLD, device=torch.device("cpu"))
        expected = official_value_norm()(1)
        expected.update(torch.tensor([[0.], [1.], [1.], [1.]]))
        mean, variance = expected.running_mean_var()
        assert norm.moments() == (mean.item(), variance.sqrt().item())
        assert stats["active_value_tokens"] == 4
        assert stats["raw_value_mse"] == 0.25
        states = [None, None]
        dist.all_gather_object(states, norm.state())
        assert states[0] == states[1]
        if rank == 1:
            row["returns"][0, 0] = float("nan")
        with pytest.raises(FloatingPointError):
            norm.prepare_update([row])
    finally:
        dist.destroy_process_group()


def test_distributed_statistics_match_full_batch_with_unequal_active_tokens(tmp_path):
    torch.multiprocessing.spawn(
        _distributed_value_norm_worker,
        args=(str(tmp_path / "value-norm-rendezvous"),),
        nprocs=2,
        join=True,
    )
