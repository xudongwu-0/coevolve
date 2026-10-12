# SearchQA Harness-R1 adapter

Independent harness Engineer with frozen solver weights, official code-hook
contract, isolated execution, immutable failure packets and same-task delta reward.

## Method index

- [`upstream_ppo.py`](upstream_ppo.py): directly imports SHA-pinned official
  Harness-R1/Relax reward normalization and PPO/KL primitives.
- [`grpo_objective.py`](grpo_objective.py): checked response-only adapter;
  group normalization and clipping come from official code.
- [`grpo_train.py`](grpo_train.py), [`workflow.py`](workflow.py):
  FSDP/vLLM backend adaptation, not exact Megatron/SGLang reproduction.
- [`sft_train.py`](sft_train.py): fresh native Qwen3 4B or Qwen3.5 9B full-text
  cold-start SFT; explicit 32-label pilot option, no truncation.
- [`bootstrap.py`](bootstrap.py): real teacher proposals, strict clean-positive
  filtering, one tokenization/prompt and deduplicated in-flight target scoring.
- [`rapid_screen.py`](rapid_screen.py): nested held-out train triage, never final
  acceptance; excludes every supplied training-packet task identity.
- [`evaluate.py`](evaluate.py): paired full dev confirmation, checked common-baseline
  reuse, irreversible test-freeze gate and no-op rejection.
- [`fast_pipeline.py`](fast_pipeline.py): persistent raw/SFT/RL matched pilot;
  optional checked node3 raw-control collection overlaps node4 optimization
  and reuses the same proposal budget.
- [`run_grpo_pilot.py`](run_grpo_pilot.py): audited four-rank optimizer and explicit
  disjoint `--rollout-gpus`; the [TP1 recovery config](../../configs/searchqa_engineer_grpo_fast4b_tp1.json)
  preserves proposal/learning budgets when multi-GPU rollout initialization stalls.
- Independent rollout children remove torchrun's distributed and
  `TORCHELASTIC_*` environment, including its agent-store setting. Actor
  processes retain their own distributed configuration.
- [`serve_roles.py`](serve_roles.py): two disjoint model roles under one GPU auditor.
  Roles may explicitly select `qwen3_xml` tool parsing and language-only
  inference for original Qwen3.5 multimodal checkpoints; existing defaults
  and tokenizer templates remain unchanged.
  `gdn_prefill_backend: triton` avoids FlashInfer JIT's CUDA compiler dependency
  on nodes without `nvcc`. Health readiness alone does not validate generation;
  run the official structured tool-call probe before benchmark rollouts.
- [`profile_rollouts.py`](profile_rollouts.py), [`profile_training.py`](profile_training.py),
  [`profile_evaluation.py`](profile_evaluation.py): real-workload performance evidence.

See [the current fast-4B experiment README](../../runs/searchqa_harness_r1_20260930/fast4b/README.md)
for measured speed, utilization, immutable result receipts, protocol differences,
failure evidence and current conclusions. No claim of effective RL harness
rewriting is justified without independent paired RL-vs-SFT development evidence.

SearchQA uses WHALE's seven dataset families but local BM25/alias EM; paper
absolute scores are not directly comparable. Solver tool/token budgets remain
fixed, and accepted harnesses accumulate rather than resetting to h0.
