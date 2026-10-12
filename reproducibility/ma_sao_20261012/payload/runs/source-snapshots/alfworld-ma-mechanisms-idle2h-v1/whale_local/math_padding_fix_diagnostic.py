"""No-update follow-up: distinguish BF16 padding drift from label/position bugs."""
import gc
import json
from contextlib import nullcontext
import torch
import torch.distributed as dist
import math_worker as mw
from math_forward_diagnostic import stats

w = mw.w

from transformers.models.qwen3_5 import modeling_qwen3_5 as q35


def corrected_padding_mask(hidden_states, attention_mask):
    if attention_mask is not None and attention_mask.shape[1] > 1:
        return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)
    return hidden_states


q35.apply_mask_to_padding_states = corrected_padding_mask


def forward_logprobs(batch, nr, bf16):
    with (torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext()):
        logits = w.state["model"](input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            position_ids=batch["position_ids"], use_cache=False).logits[:, -batch["responses"].shape[-1]-1:-1][:, :nr]
    result = -torch.nn.functional.cross_entropy(logits[0].float(), batch["responses"][0, :nr], reduction="none")
    return result[None].cpu()


def main():
    w.RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    dist.init_process_group("nccl", init_method="file://" + str(w.RUN / "rdzv"), rank=0, world_size=1)
    try:
        identity = w.load_model(w.BASE)
        source = w.ROOT / "runs/whale_math_native_20260915/round_01"
        rows = [json.loads(x) for x in (source / "rollouts/trajectories.jsonl").read_text().splitlines()]
        accepted = [c for r in rows if r["reward"] for c in r["policy_calls"]]
        saved = torch.load(source / "accepted_batch.pt", map_location="cpu", weights_only=True)
        results = []
        with torch.no_grad():
            for i in [0, 414]:
                c = accepted[i]
                nr = len(c["response_ids"])
                single = {k: v.cuda() for k, v in w.make_batch([c]).batch.items()}
                padded = {k: v[i:i+1].cuda() for k, v in saved.items()}
                fp = forward_logprobs(single, nr, False)
                fp_pad = forward_logprobs(padded, nr, False)
                bf = forward_logprobs(single, nr, True)
                bf_pad = forward_logprobs(padded, nr, True)
                # Feed already-saved target tokens through autoregressive cache.
                # This is teacher forcing, not sampling new trajectories.
                prompt = torch.tensor([c["prompt_ids"]], device="cuda")
                ncache = min(64, nr)
                cached = []
                past = None
                current = prompt
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    for step in range(ncache):
                        mask = torch.ones((1, len(c["prompt_ids"])+step), dtype=torch.long, device="cuda")
                        out = w.state["model"](input_ids=current, attention_mask=mask, past_key_values=past,
                                               use_cache=True, logits_to_keep=1)
                        cached.append(torch.log_softmax(out.logits[0, -1].float(), -1)[c["response_ids"][step]].item())
                        past = out.past_key_values
                        current = torch.tensor([[c["response_ids"][step]]], device="cuda")
                cache = torch.tensor([cached])
                item = {"accepted_index": i, "call_id": c["call_id"], "fp32_padded_vs_unpadded": stats(fp_pad, fp),
                        "bf16_padded_vs_unpadded": stats(bf_pad, bf), "bf16_vs_fp32_unpadded": stats(bf, fp),
                        "bf16_cached_vs_dense_first_tokens": stats(cache, bf[:, :ncache]), "cached_teacher_forced_tokens": ncache}
                print(json.dumps(item), flush=True)
                results.append(item)
                del single, padded, fp, fp_pad, bf, bf_pad, past, out
                gc.collect();torch.cuda.empty_cache()
        assert w.digest(w.parameter_hashes(w.state["model"])) == identity["weight_digest"]
        w.atomic_json(w.RUN / "result.json", {"scope": "Two fixed TRAIN calls, diagnostic-only singleton padding mask correction; no generation/update/heldout",
            "identity": identity, "unchanged_weights": True, "results": results})
    finally:
        w.state.clear();gc.collect();torch.cuda.empty_cache();dist.destroy_process_group()


if __name__ == "__main__":main()
