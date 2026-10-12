# Fresh Harness Engineer SFT

Entrypoint: `whale_local/harness_r1_searchqa/sft_train.py`. The JSON configuration
documents its defaults; CLI flags, not this JSON file, configure execution.

Only the official, receipt-verified `assets/Qwen3.5-9B` may initialize training.
This is full-parameter **text** SFT, not LoRA or a previously trained editor:
all language decoder, embedding and output-head weights receive gradients.
The unused image tower and upstream MTP auxiliary head are excluded, not trained
or randomly initialized. The exported checkpoint has the HF `qwen3_5_text`
architecture and loads with `AutoModelForCausalLM.from_pretrained`.
Installed vLLM 0.19.1 does not register that text-only architecture by default,
although it ships `vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM`.
Future serving/GRPO launchers must register `Qwen3_5ForCausalLM` with
`ModelRegistry.register_model` in both the launcher and spawned workers before
loading this export. Do not rewrap the trained text checkpoint as a multimodal
model or introduce missing vision weights. HF loading needs no such registration.

The input is actual approved JSON-array or JSONL data. Normalized rows have exactly
`id` (unique string), `split` (`train`), and `messages`, containing exactly
`system`, `user`, `assistant` roles with string content. Assistant content is
literal reasoning continuation (no opening `<think>`), followed by `</think>`
and one `<patch>` JSON object `</patch>`. Prompt rendering uses the official
thinking-enabled generation template; prompt/prefill tokens are masked. The
continuation and `<|im_end|>` receive loss. No assistant-template normalization,
truncation, held-out rows, synthetic fallback, repeated supervision, or dropped tails.

The bootstrap's `messages + provenance` JSONL is also accepted directly when
its sibling `COMPLETE.json` certifies at least 256 positive train examples,
matching dataset SHA256, example count and Engineer manifest identity.
Every row must have a unique generation ID, positive finite reward delta,
train split, nonempty distinct task IDs, and patch/Engineer hashes.
Task-membership and actual-rerun verification belong to the producer and the
parent's dataset approval; the trainer never fabricates those evaluations.
Provenance never enters model input tokens.

Run CPU checks on node4 before allocating GPUs:

```bash
cd /mnt/3fs/xudong/coevolve
CUDA_VISIBLE_DEVICES='' /dev/shm/sao-venv-ppo/bin/python -m pytest \
  -q -p no:cacheprovider tests/test_harness_r1_sft.py
CUDA_VISIBLE_DEVICES='' /dev/shm/sao-venv-ppo/bin/python whale_local/harness_r1_searchqa/sft_train.py --help
CUDA_VISIBLE_DEVICES='' /dev/shm/sao-venv-ppo/bin/python whale_local/harness_r1_searchqa/sft_train.py --inspect-base-only
CUDA_VISIBLE_DEVICES='' /dev/shm/sao-venv-ppo/bin/python whale_local/harness_r1_searchqa/sft_train.py \
  --preflight-only --data "$VERIFIED_DATA" --data-sha256 "$APPROVED_SHA256"
```

Only after the parent supplies and approves the real frozen dataset, launch:

```bash
cd /mnt/3fs/xudong/coevolve
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/dev/shm/sao-venv-ppo/bin/python -m torch.distributed.run \
  --standalone --local-addr=127.0.0.1 --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  whale_local/harness_r1_searchqa/sft_train.py \
  --data "$VERIFIED_DATA" --data-sha256 "$APPROVED_SHA256" \
  --output "$FRESH_OUTPUT" --lr 1e-5 --epochs 2 \
  --global-batch-size 24 --max-length 32768
```

One unpadded example per rank and three accumulated microsteps give global
batch 24. Each backward reduces sharded FP32 gradients; deliberately no
`no_sync` full-gradient accumulation. BF16 compute, FP32 master weights and
Adam states, decoder activation checkpointing, SDPA, and chunk-checkpointed
vocabulary CE avoid the single-GPU Adam and full-context logits memory pitfalls.
Rank zero initially needs host RAM for the ~36 GB FP32 text model; checkpoint
gather/verification needs additional host RAM. No training-memory benchmark or
GPU-training success is claimed by the CPU checks.
The final optimizer group may contain fewer than 24 examples. If it leaves
ranks idle, those ranks forward a real row with exactly zero loss to match
FSDP collectives; no extra example, duplicated supervised target, or synthetic
formal training data is introduced.

The trainer starts no serving process. Node4 lacks the CUDA toolkit required
by vLLM's default FlashInfer GDN prefill. Future Qwen3.5 rollout servers must
use the supported `--gdn-prefill-backend triton` serving option.

Caches are isolated under `assets/runtime-cache/harness-engineer-sft` on 3FS.
The test module additionally hides GPUs internally before
importing Torch; always set `CUDA_VISIBLE_DEVICES=''` on the launching command
as well, including import probes, to protect shared serving allocation.
No dependency installation or LlamaFactory is required by this implementation.
Final weights, config, tokenizer and tensor coverage/finite checks are completed
in a sibling `.incomplete-*` directory before one atomic rename. `manifest.json`
records verified data/source/base/checkpoint SHA256s and completed updates;
parameter-change probes reject a no-op training run. Exported generation uses
the trained conversation EOS `<|im_end|>`.
`COMPLETE` authenticates the manifest hash and says `VERIFIED COMPLETE`. Failed
staging directories are not published. Outputs are inference HF checkpoints,
not resumable optimizer checkpoints. Completion does not replace patch legality
checks or held-out evaluation.
