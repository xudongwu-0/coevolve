"""Equivalent global token-mean RSFT with distinct samples on FSDP ranks."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import file_sha


def rank_state_hashes(root: Path, world: int) -> dict[str, str]:
    if type(world) is not int or world < 1:
        raise ValueError("Native state hashing requires a positive world size")
    paths = [root / f"rank-{rank:04d}.pt" for rank in range(world)]
    with ThreadPoolExecutor(max_workers=world) as workers:
        hashes = list(workers.map(file_sha, paths))
    return {str(rank): value for rank, value in enumerate(hashes)}


def shard_calls(calls: list[dict], rank: int, world: int) -> tuple[list[dict], list[bool]]:
    if not calls or type(rank) is not int or type(world) is not int or not 0 <= rank < world:
        raise ValueError("Sample sharding requires nonempty calls and a valid rank")
    local = calls[rank::world]
    active = [True] * len(local)
    count = (len(calls) + world - 1) // world
    if len(local) < count:
        padding = min(calls, key=lambda call: len(call["prompt_ids"]) + len(call["response_ids"]))
        local.append(padding)
        active.append(False)
    if len(local) != count:
        raise ValueError("Every FSDP rank must execute the same number of singleton forwards")
    return local, active


def parallel_sft_update(actor, data, device) -> dict:
    import torch
    import torch.distributed as dist

    if (not dist.is_initialized() or actor.scaler is not None
            or actor.config.use_dynamic_bsz or data.meta_info["sft_epochs"] != 1
            or data.meta_info["sft_micro_batch_size_per_gpu"] != 1):
        raise ValueError("Parallel RSFT requires the declared BF16 singleton one-epoch FSDP contract")
    actor.actor_module.train()
    world = dist.get_world_size()
    mini_size = min(data.meta_info["sft_mini_batch_size"], len(data))
    losses, tokens, updates, norms = 0., 0., 0, []
    for mini in data.split(mini_size):
        total = mini.batch["response_mask"].sum().detach().to(device=device, dtype=torch.float32)
        dist.all_reduce(total)
        if not torch.isfinite(total) or total.item() <= 0:
            raise ValueError("Every global RSFT minibatch must contain actual response tokens")
        # FSDP averages gradients across ranks, so each local numerator uses T/world.
        denominator = total / world
        actor.actor_optimizer.zero_grad()
        for micro in mini.split(1):
            micro = micro.to(device)
            inputs = {**micro.batch, **micro.non_tensor_batch,
                      "pad_token_id": data.meta_info.get("pad_token_id", 0)}
            mask = inputs["response_mask"].bool()
            output = actor._forward_micro_batch(
                inputs, temperature=data.meta_info["temperature"], calculate_entropy=False)
            values = output["log_probs"]
            if values.shape != mask.shape or not torch.isfinite(values[mask]).all():
                raise FloatingPointError("Invalid response log probabilities in parallel RSFT")
            loss = -torch.where(mask, values, 0).sum() / denominator
            loss.backward()
            losses += float(loss.detach())
            tokens += int(mask.sum())
        norm = actor._optimizer_step()
        if not torch.isfinite(norm):
            raise FloatingPointError("Parallel RSFT gradient norm is nonfinite")
        norms.append(float(norm.detach()))
        updates += 1
    actor.actor_optimizer.zero_grad()
    totals = torch.tensor([losses, tokens], dtype=torch.float64, device=device)
    dist.all_reduce(totals)
    return {"actor/sft_loss": float(totals[0] / world),
            "actor/sft_token_count": float(totals[1]),
            "actor/sft_updates": float(updates), "actor/grad_norm": norms}
