"""Matched original Coevolve updates over exact sampled tokens."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import ROOT, read_json, write_json
from harness_grpo_objective import advantages, clipped_loss

sys.path.insert(0, str(ROOT / "WHALE/domains/math_reasoning"))
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig


def make_batch(calls: list[dict], pad: int):
    if not calls or any(not c["prompt_ids"] or not c["response_ids"] for c in calls):
        raise ValueError("Empty sampled policy call")
    pmax = max(len(c["prompt_ids"]) for c in calls)
    rmax = max(len(c["response_ids"]) for c in calls)
    ids = torch.full((len(calls), pmax + rmax), pad, dtype=torch.long)
    attention = torch.zeros_like(ids)
    mask = torch.zeros((len(calls), rmax), dtype=torch.long)
    for i, call in enumerate(calls):
        prompt, response = call["prompt_ids"], call["response_ids"]
        ids[i, pmax-len(prompt):pmax] = torch.tensor(prompt)
        ids[i, pmax:pmax+len(response)] = torch.tensor(response)
        attention[i, pmax-len(prompt):pmax+len(response)] = 1
        mask[i, :len(response)] = 1
    return DataProto.from_dict(tensors={
        "input_ids": ids, "attention_mask": attention,
        "position_ids": (attention.cumsum(-1) - 1).clamp(min=0),
        "responses": ids[:, pmax:].clone(), "response_mask": mask,
    }, meta_info={
        "temperature": calls[0]["temperature"], "pad_token_id": pad,
        "sft_epochs": 1, "sft_mini_batch_size": 4, "sft_micro_batch_size_per_gpu": 1,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=["editor", "solver"], required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--updates", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    rows = read_json(args.records)
    args.output.mkdir(parents=True)
    reward_values = [row["reward"] for row in rows]
    groups = [row.get("group", "") for row in rows]
    if args.role == "editor" and any(n != 8 for n in Counter(groups).values()):
        raise ValueError("Original editor requires complete groups of eight")
    adv, _ = advantages(reward_values, groups)
    accepted = [call for row in rows if row["reward"] > 0 for call in row["calls"]]
    if (args.role == "solver" and not accepted) or (args.role == "editor" and not adv.any()):
        write_json(args.output / "result.json", {
            "updated": False, "checkpoint": str(args.model.resolve()),
            "reason": "no accepted RSFT calls" if args.role == "solver" else "zero group variance",
        })
        return
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
    ).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.98) if args.role == "editor" else (0.9, 0.999),
        weight_decay=0.1 if args.role == "editor" else 0.01,
    )
    actor = DataParallelPPOActor(FSDPActorConfig(
        rollout_n=1, ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=1,
        use_remove_padding=False, use_torch_compile=False, use_dynamic_bsz=False,
        grad_clip=1.0,
    ), model, optimizer)
    probes = {name: p.detach().cpu().clone() for name, p in model.named_parameters()
              if name.endswith("layers.0.self_attn.q_proj.weight")}

    def forward(call):
        batch = make_batch([call], tokenizer.pad_token_id).to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return actor._forward_micro_batch(
                {**batch.batch, "pad_token_id": tokenizer.pad_token_id},
                temperature=call["temperature"], calculate_entropy=False,
            )["log_probs"][0, :len(call["response_ids"])]

    if args.role == "solver":
        model.train()
        metrics = actor.update_sft_policy(make_batch(accepted, tokenizer.pad_token_id).to("cuda"))
    else:
        if any(len(row["calls"]) != 1 for row in rows):
            raise ValueError("Editor must have exactly one sampled code completion")
        calls = [row["calls"][0] for row in rows]
        model.eval()
        with torch.no_grad():
            old = [forward(call).float().detach().cpu() for call in calls]
        metrics = []
        model.train()
        for _ in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for i, call in enumerate(calls):
                if adv[i] == 0:
                    continue
                rollout = torch.tensor(call["response_logprobs"], device="cuda")
                loss, _, _ = clipped_loss(forward(call).float(), old[i].to("cuda"), rollout, adv[i])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite original GRPO loss")
                (loss / len(calls)).backward()
                total += float(loss.detach()) / len(calls)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite original GRPO gradient")
            optimizer.step()
            metrics.append({"loss": total, "grad_norm": float(norm)})
    changed = False
    for name, value in model.named_parameters():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite parameter {name}")
        if name in probes:
            changed |= not torch.equal(value.detach().cpu(), probes[name])
    if not changed:
        raise RuntimeError("Original update produced no persisted probe change")
    checkpoint = args.output / "checkpoint"
    model.save_pretrained(checkpoint)
    tokenizer.save_pretrained(checkpoint)
    torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
    write_json(args.output / "result.json", {
        "updated": True, "checkpoint": str(checkpoint.resolve()), "metrics": metrics,
        "method": "original Coevolve outcome-GRPO" if args.role == "editor" else "native WHALE RSFT",
        "episodes": len(rows), "accepted_calls": len(accepted), "probe_changed": changed,
        "stage_optimizer_reset": True,
    })


if __name__ == "__main__":
    main()
