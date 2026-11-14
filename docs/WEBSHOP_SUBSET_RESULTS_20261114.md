# WebShop: completed real subset experiments, 2026-11-14

## Decision

**Do not launch full training/evaluation or deploy either self-distillation
checkpoint on this evidence. Both preregistered high-confidence gates failed.**

The real environment, rollout, training, checkpoint-load and evaluation paths
work. This is not a CPU-mock result. However, the strict successful-context
variant did not reliably beat matched GRPO, and the verified-no-op-filtered
variant regressed on its separate test subset. All16 leased GPUs on nodes25/26
were verified empty after the tests; no serving process was retained.
Primary SFT on nodes22/24 was not changed or accessed by these experiment commands.

## 1. Protocol and interpretation

- **Environment:** pinned public WebShop text environment,1,000 products,
  6,910 synthetic goals (`human_goals=False`), Lucene search, goal seed233.
- **Starting model:** existing local BF16 **Qwen3.5-9B**. It is not a primary-SFT
  QAT checkpoint or the historical ALFWorld97 Actor.
- **Training per arm:** IDs500–503;8student rollouts/task,4optimizer updates,
  **32real environment episodes**. Train seed1000043; temperature1, top-p1,
  no top-k truncation.
- **Evaluation:** one rollout/task, seed0, temperature0.4, top-p1, top-k0.
  Maximum15actions/episode,512generated tokens/action,4096total context tokens,
  history up to3turns; explicit non-thinking chat template.
- **Learning:** identical LoRA rank16/alpha32/dropout0/all-linear layouts,
  51,265,024 trainable parameters; AdamW LR1e-5, PPO clip0.2, reference KL0.01,
  gradient clip1. The SD coefficient is0.5 with teacher-gap clip5.
  Environment success reward is binary; partial score is reported separately.
- **Strict cohort:** test IDs32–63. Teacher context requires a clean successful
  *other* training trajectory from the same current-policy/task group.
- **Filtered cohort:** untouched test IDs128–159. An otherwise successful
  teacher path may omit invalid turns only if their before/after observations
  are exactly equal. Omission is explicitly labelled. **Student tokens,
  rewards, action counts and behavior probabilities are never filtered.**
- Protocol-debug tasks0/1 are excluded. Both candidates and the fresh second
  subset were declared before either trained checkpoint's heldout evaluation.
  The second candidate was motivated by a real train-task501 success that the
  strict filter discarded, not by heldout candidate outcomes.

The cohorts are different: **do not rank absolute scores across them**.
Only within-cohort paired comparisons have matched task identities and budgets.
These are task-ID-heldout diagnostics, not certified product-family-OOD tests.
Pretraining exposure is unverified.

## 2. Actual results

Success is `terminated AND raw environment score == 1.0`.
Score is the mean raw environment reward on `[0,1]`, multiplied by100 below
for a100-point display. Confidence intervals are95%.

### A. Strict successful-context teacher: IDs32–63

| Model | Success | Success Wilson CI | Mean score /100 | Score task-bootstrap CI /100 |
|---|---:|---:|---:|---:|
| Initial model | 15/32 =46.875% | [30.87,63.55]% | 55.7716 | [40.58,71.49] |
| Matched GRPO | 11/32 =34.375% | [20.41,51.69]% | 46.5590 | [31.38,62.43] |
| Strict SD | 12/32 =37.500% | [22.93,54.75]% | 50.7554 | [35.05,66.55] |

Paired **SD minus GRPO**:

- Success: **+3.125percentage points**, task-bootstrap CI **[-6.25,+12.50]pp**.
- Mean score: **+4.1964/100**, paired CI **[-5.8036,+15.0893]**.
- Rescued task IDs: **42,60**. Regressed ID: **55**.
- Exact two-sided McNemar **p=1.0**.
- **Promotion gate: FAIL.** Both trained checkpoints are also below the fresh
  initial model. SD versus initial is -9.375pp success and -5.0162/100 score;
  that is not an improvement over the starting checkpoint.

### B. Verified-no-op-filtered teacher: IDs128–159

| Model | Success | Success Wilson CI | Mean score /100 | Score task-bootstrap CI /100 |
|---|---:|---:|---:|---:|
| Initial model | 7/32 =21.875% | [11.02,38.75]% | 41.5079 | [25.87,56.54] |
| Matched GRPO | 9/32 =28.125% | [15.56,45.37]% | 45.1538 | [29.11,60.93] |
| Filtered SD | 7/32 =21.875% | [11.02,38.75]% | 39.3874 | [23.78,54.64] |

Paired **SD minus GRPO**:

- Success: **-6.250pp**, task-bootstrap CI **[-15.625,0]pp**.
- Mean score: **-5.7664/100**, paired CI **[-13.8095,+0.1563]**.
- Rescued task IDs: **none**. Regressed IDs: **139,156**.
- Exact two-sided McNemar **p=0.5**.
- **Promotion gate: FAIL.** SD ties initial success count but has lower mean
  score; it does not demonstrate a starting-model improvement either.

The complete64-row per-ID success/score/W-L-T table is
[per_task.csv](../reproducibility/webshop_review_20261114/per_task.csv).
All raw initial/GRPO/SD outcome vectors and paired statistics are preserved in
the [strict bundle](../reproducibility/webshop_subset_20261114/) and
[filtered bundle](../reproducibility/webshop_filtered_subset_20261114/).

## 3. Confidence gate and limits

The preregistered gate requires matched training budgets and identities,
at least32 fixed evaluation tasks, genuine teacher tokens/nonzero teacher gap
and gradient, positive lower success-difference CI, nonnegative lower
score-difference CI, and exact two-sided McNemar **p≤0.025**.
The0.025 threshold is Bonferroni0.05/2 for the two candidates and was declared
before trained heldout evaluation. Both candidates had real teacher signal,
but **neither passed the performance criteria**.

The5,000-draw bootstrap resamples paired tasks. These intervals are conditional
on one training run and one rollout seed per task; they do not capture training
seed variance or establish broad algorithmic superiority. The single-seed
standard-deviation field0 is not an uncertainty estimate. A small positive
point difference is not a reliable win.

## 4. Actual token/action and time costs

Every evaluation has32episodes and the same cap of480actions and245,760
generated tokens. Actual lengths differ with policy behavior.
Wall time includes environment/model hash/load setup; it is not isolated
kernel throughput or matched FLOPs.

| Cohort / model | Actual evaluation actions | Generated tokens | Wall seconds |
|---|---:|---:|---:|
| Strict / initial | 353 | 15,931 | 682.79 |
| Strict / GRPO | 346 | 39,745 | 2,121.35 |
| Strict / SD | 351 | 30,979 | 1,713.39 |
| Filtered / initial | 346 | 16,123 | 712.67 |
| Filtered / GRPO | 366 | 16,844 | 1,014.38 |
| Filtered / SD | 370 | 29,701 | 1,604.43 |

| Training arm | Episodes / updates | Actual actions | Generated tokens | Teacher-scored tokens | Wall seconds |
|---|---:|---:|---:|---:|---:|
| Strict / GRPO | 32 /4 | 397 | 37,417 | 0 | 2,583.12 |
| Strict / SD | 32 /4 | 379 | 35,091 | 26,588 | 2,541.60 |
| Filtered / GRPO | 32 /4 | 391 | 36,276 | 0 | 2,532.34 |
| Filtered / SD | 32 /4 | 384 | 36,182 | 22,697 | 2,574.80 |

There are128completed research-training episodes and192completed scored
evaluation episodes. Teacher scores are extra forward-pass compute, not
additional environment rollouts or generated teacher actions. The failed
protocol probes and zero-adapter diagnostic are separately retained and are
not included in these paired-research cost tables. No speedup claim is made.

The initial strict training groups were token-identical across the two
native learners:8episodes,6,418tokens,1clean success. Both had nonzero real
optimizer gradients; SD's first group used6,169teacher-scored tokens.
Native training reached approximately89.0GB allocated CUDA memory for strict
GRPO and97.1GB for strict SD on a single H200 each.

## 5. Source, model and checkpoint identity

- Public reference/environment:
  [`35a44f5c5647b1785ad23ddf0ffbd196b695bf86`](https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86).
  Apache-2.0 and nested WebShop MIT licenses retained locally.
- Strict native snapshot: experiment commit `db969bc`, `frozen/subset-v3`.
- Filtered native snapshot: commit `18ae3aa`, `frozen/subset-v4`.
  Actor function ASTs, runtime file and objective file were verified unchanged
  between these snapshots; the variant changes teacher-path eligibility/context.
- One identical base-model content map was verified across all six evaluations.
  Canonical content-map fingerprint:
  `7aa77ee543c8d8280dec60e9d0359ebea79abec5522055b2987ff5e49f0d66d9`.
- Common corpus/index/environment-source fingerprint:
  `d7b3302242f5fdf2066eeb92a1339a13a43b2daa7c777fc417bace54432c4956`.
- Runtime: torch2.10.0+cu129, transformers5.7.0, peft0.18.1, JDK17.0.21+9,
  Pyserini0.17.0, NumPy2.2.6. Native Torch fallback, not the optional fast
  FLA/causal-convolution path.

Final `adapter_model.safetensors` SHA-256:

| Checkpoint | SHA-256 |
|---|---|
| Strict GRPO | `b1d4212e89d31af82bf353057036dbf6f4c1368dc356934391526c297334439c` |
| Strict SD | `b2ccc7516ab034024c5d29ea63ebc62341ee35af11baeb9c1a576b2a91f9618e` |
| Filtered GRPO | `8cac6c76cf97e12493d9ae65653107bf7fef0449c0519736d0a44f078433f753` |
| Filtered SD | `085f1cae8f560908151f36d02f3046c0c1f291685ed1f5d36d5e01eb5530b5a7` |

Full shard/tokenizer hashes, adapter identities, task/price manifests,
hyperparameters and native runtime/cost records are in
[REVIEW.json](../reproducibility/webshop_review_20261114/REVIEW.json).
Adapters and full trajectories remain local under
`runs/webshop_self_distillation_20261114/`; no weights were uploaded.
These are inference adapters, not optimizer/RNG recovery checkpoints.

A separate native zero-LoRA sanity check verified all LoRA-B tensors were zero
and reproduced **all510 debug-gate tokens exactly** from the raw model across
eight actions. The wrapper itself therefore passed this functional check;
the diagnostic is not another performance candidate or a general proof over
all inputs.

## 6. Reconcile the user's historical45% /74

The historical **success45% / score74** has no recovered original run manifest,
checkpoint, task IDs, corpus size, action/context limits or sampling record in
the inspected checkout/history. It is **not a verified comparator**.

Correct scales are:

- Historical45% success means0.45.
- If historical74 is a conventional100-point score, it means0.74.
- Our first fresh score0.55771645 means**55.771645/100**, not55.77 raw units
  and not a54-point gain over0.74.
- Even the arithmetic55.77−74=-18.23points would not be a valid attributed
  performance comparison without matching the missing protocol/model evidence.

The identical initial checkpoint obtained15/32 on one registered subset and
7/32 on the other. That illustrates why absolute cross-subset point
comparisons are unsafe; it is not a model change or method gain.

Primary sources were checked rather than trusting a secondary search summary:

- The [pinned evaluation launcher](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/launchers/test_pretrained_1gpu.sh)
  sets **15WebShop actions**, not20.
- The [project page](https://aim-uofa.github.io/Self-Distillation-AgentRL/)
  and [paper](https://arxiv.org/abs/2610.19789) use128fixed WebShop goals and
  eight rollout seeds at temperature0.4. The paper's benchmark table uses
  history2 and a4096prompt-token limit; our pilot uses history3 and4096total
  context. The reference has an invalid-action penalty0.1 and100training
  updates, unlike this four-update LoRA pilot.
- Paper Table3's Qwen3.5-9B pretrained row reports score9.89±1.01 and
  success5.86±1.10%; its trained100-step GRPO row reports83.30±1.16 and
  70.02±1.38%. **Those rows do not authenticate the user's45/74 baseline,
  and no gain is claimed against them.**
- The public implementation also uses worker-specific environment seeds and
  a fixed evaluation selection seed1000; our task/price identity is held at
  goal seed233 with explicit manifests. Dataset/model revision parity with the
  historical run or paper cannot be assumed from a model-family name.

The secondary answer claiming pretrained45/74 with20steps was contradicted by
these primary sources and discarded.
Detailed facts and provenance are preserved in
[HISTORICAL_BASELINE_RECONCILIATION.json](../reproducibility/webshop_review_20261114/HISTORICAL_BASELINE_RECONCILIATION.json).

## 7. Recompute and cleanup

No model, GPU, network or optional training dependencies are needed:

```bash
python3 scripts/reproduce_webshop_subset.py \
  --bundle reproducibility/webshop_subset_20261114
python3 scripts/reproduce_webshop_subset.py \
  --bundle reproducibility/webshop_filtered_subset_20261114
```

These commands verify payload hashes and recompute saved per-ID metrics,
confidence intervals and paired gates. **They are not new model evaluations.**
The implementation/regression suite passed83CPU tests, including causal
response-mask, real tiny-transformer/LoRA, protocol and evidence-tamper tests.

The [final GPU receipt](../reproducibility/webshop_review_20261114/FINAL_GPU_RELEASE.json)
records no compute PIDs and0MiB on every GPU of nodes25/26. No inference server,
resource waiter or full-data campaign was retained. A failed early protocol
worker was stopped only after exact PID/UID/start-tick checks; no peer process
or shared service was terminated. Numeric UID/GID1008:1008 was used throughout.

The controller's root filesystem reported full during a tool-output operation;
3FS had about11TB free. Scientific outputs and private runtime caches stayed
on3FS. No prohibited temporary path was inspected or cleaned.

**Recommendation:** retain the starting checkpoint pending identification of
the historical45/74 run. Do not scale or deploy these SD variants. A next
research attempt needs a larger train-only development protocol, more training
seeds, verbosity/failure analysis, and a new untouched confirmatory set—not
blindly extending these runs or selecting the better-looking cohort.
