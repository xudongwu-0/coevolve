"""Read-only, training-data-only comparison of native RSFT forward semantics."""
import gc
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
import math_worker as mw

w = mw.w


def stats(a, b):
    d = (a.float() - b.float()).abs()
    return {"mean_abs_logprob_difference": d.mean().item(),
            "max_abs_logprob_difference": d.max().item(),
            "a_mean_nll": -a.float().mean().item(),
            "b_mean_nll": -b.float().mean().item()}


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
        batch = torch.load(source / "accepted_batch.pt", map_location="cpu", weights_only=True)
        indices = sorted(set([0, min(range(len(accepted)), key=lambda i: len(accepted[i]["response_ids"])),
                              max(range(len(accepted)), key=lambda i: len(accepted[i]["response_ids"])),
                              max(range(len(accepted)), key=lambda i: len(accepted[i]["prompt_ids"]))]))
        results = []
        with torch.no_grad():
            for i in indices:
                c = accepted[i]
                nr = len(c["response_ids"])
                single = {k: v.cuda() for k, v in w.make_batch([c]).batch.items()}
                padded = {k: v[i:i+1].cuda() for k, v in batch.items()}
                w.state["model"].eval()
                plain = w.state["actor"]._forward_micro_batch(single, temperature=1.0)["log_probs"].cpu()
                pad = w.state["actor"]._forward_micro_batch(padded, temperature=1.0)["log_probs"][:, :nr].cpu()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = w.state["model"](input_ids=single["input_ids"], attention_mask=single["attention_mask"],
                                              use_cache=False).logits[:, -nr-1:-1]
                    # Independent float32 cross entropy, with HF automatic positions.
                    hf = -torch.nn.functional.cross_entropy(logits[0].float(), single["responses"][0], reduction="none")
                    hf = hf[None].cpu()
                del logits
                w.state["model"].train()
                train = w.state["actor"]._forward_micro_batch(single, temperature=1.0)["log_probs"].cpu()
                result = {"accepted_index": i, "call_id": c["call_id"], "prompt_tokens": len(c["prompt_ids"]),
                          "response_tokens": nr, "padded_vs_unpadded": stats(pad, plain),
                          "native_vs_independent_hf_cross_entropy": stats(plain, hf), "train_vs_eval": stats(train, plain)}
                print(json.dumps(result), flush=True)
                results.append(result)
                del single, padded, plain, pad, hf, train
                gc.collect()
                torch.cuda.empty_cache()
        final_digest = w.digest(w.parameter_hashes(w.state["model"]))
        assert final_digest == identity["weight_digest"]
        w.atomic_json(w.RUN / "result.json", {"scope": "Four prespecified training-call length extremes; no optimizer steps, generations, confirmation or test access",
                    "identity": identity, "final_digest_unchanged": True, "results": results})
    finally:
        w.state.clear()
        gc.collect()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
