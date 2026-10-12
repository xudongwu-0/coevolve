# MA-SAO experiment log

Purpose: persistent record of every development experiment, its configuration,
result and conclusion, to avoid forgetting or repeating work. Newest last.
Report accuracy = a disjoint validation report set. Early runs used 128 dev
questions at mean@4; r4 onward uses 128 dev + 384 never-trained train-tail
questions at mean@2 (512 questions, guarded against solver training offsets).
It is exploratory validation, not held-out AIME. AIME test stays sealed until a final configuration is frozen.

For the current ALFWorld joint-round protocol and status, see
[project state](PROJECT_STATE.md) and [MA-SAO](MA_SAO.md). The protocol below
records the earlier math campaign, not the current joint implementation.

For the consolidated ALFWorld research questions, designs, result tables and
conclusions through2026-10-07 16:19 UTC, see the
[standalone experiment report](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md).

## Protocol (historical math campaign)

- Baseline ("SAO"): editor (harness model) and solver (weights) each updated
  with the pinned open-source SAO; order H→W→H. Success criterion: final
  `h2w1` report accuracy significantly above the un-updated `h0w0`.
- Only this baseline. Tune it on short runs, run the formal baseline once, then
  freeze it.
- MA-SAO may be tuned, but must use the **same sample budget**: identical locked
  editor/solver steps and batch sizes, harness-scoring questions, proposals,
  selection/report sizes and token limits. GPU time is reported, not matched.
- Short fast runs first; extend to H→W→H→W→H only after a clear HWH advantage.

## Noise floor

- Same initial model, same questions: report 0.379 vs 0.371, paired 95% CI
  about ±3.5 points (128 questions × 4 samples). Selection split with one
  sample differed by 7 points (0.266 vs 0.336) — selection noise alone can
  change which harness a run keeps. Selection now uses 2 samples per question.

## Runs

| Run | Arm | Status | h0w0 | h1w0 | h1w1 | h2w1 | Conclusion |
|---|---|---|---|---|---|---|---|
| sao_dapo_calibration_* | both | invalid | – | – | – | – | Wiring/debug only; tool fence, network, checkpoint and ownership bugs fixed. |
| dev_r1_independent | SAO | invalid | 0.379 | 0.393 | – | – | Packed-SDPA contamination (below). |
| dev_r1_ma | MA k=1 | invalid | 0.371 | 0.371 | – | – | Packed-SDPA contamination; W1 ran 16 steps at ~100 seqs/step. |
| dev_r2_* | both | lost | 0.381 / 0.375 | – | – | – | Editor DIS rejection fell from ~5% to 0.1% after the engine fix. Runs were killed when the attached shell ended on a user message; relaunched detached as r3. |
| dev_r3_independent | SAO lr_s 2e-6, lr_e 1e-6, W 16×32 | done | 0.389 | 0.389 | 0.404 | 0.402 | Solver DIS rejection 0.0% (was 33-60%). +1.3 pts overall, not significant: baseline does not yet beat no-update. W step ≈ 11 s. |
| dev_r3_ma | MA k=1, same budget | h1w1 read | 0.381 | 0.381 | 0.375 | – | Recycled data (staleness up to 15) still had ~0% DIS rejection: at lr 2e-6 the policy barely moves. W step ≈ 40 s (~100 seqs). No gain; W learning itself too weak to show any effect. |
| dev_r4_sao_lr5e6 | SAO lr_s 5e-6, lr_e 5e-6, W 16×32, report 512×2 | crashed at H2 | 0.410 | 0.412 | – | – | W1 stable (entropy 0.3-0.5, DIS 0%) but no upward train-reward trend in 16 steps. H2 crashed: vLLM 400 on context overflow (fixed, below). |
| dev_r4_sao_lr1e5 | SAO lr_s 1e-5 | stopped | 0.386 | 0.410 | 0.000 | – | **Collapse**: from W step 3 malformed tool calls (invented function names, non-JSON), reward 0, entropy 0.40→0.05. PPO clip ratio 0 throughout: one update per batch keeps ratio at 1, so lr alone sets the step. lr 1e-5 excluded. |
| dev_r5_sao | SAO lr 5e-6, W 32×32 | invalid (no tools in training) | 0.392 | 0.401 | 0.402 | 0.402 | +1.1 [−2.2, +4.5]. W1 did not change dev accuracy: training rollouts had **zero parsed tool calls** (finding 8). |
| dev_r5_ma_warm | MA: 4 critic-only warm-up steps, no mixing | invalid (no tools in training) | 0.376 | 0.409 | 0.402 | 0.402 | Warm-up executed (cccc then actor). Value loss after 4 warm-up steps still 2.4 (8 critic updates insufficient under grad-clip 1.0 with grad norms 50-150). Online training reward on the same questions: MA +8.2 pts over baseline (paired, 1024 q), largest where the baseline dipped during critic calibration (q256-512: +19.9 [+14.5,+25.8]); final dev unchanged. Suggestive only: fresh rollouts had no tools. |
| dev_r6_sao | SAO lr 5e-6, W 32×32, **tools fixed** | stopped | 0.387 | 0.387 | **0.313** | – | W1 degraded dev by −7.4. Truncation 0.17→0, boxed→1.0; train reward 0.43→0.19 on questions a no-tool policy solved at 46%. |
| dev_r6_ma_warm8 | MA, 8-step critic warm-up | stopped | 0.391 | 0.417 | – | – | Warm-up calibrated the critic (value loss 0.14-0.24 at first actor step) but the policy drifted the *opposite* way (truncation 0.17→0.65, tool calls 0.92→0.2-0.5), reward also down. |
| **dev_r7_sao_lr2e6** | **SAO lr_s 2e-6, lr_e 5e-6, W 32×32, H 3×16, tools** | **done — tuned baseline** | 0.387 | 0.416 | 0.454 | **0.454** | **Beats no-update: +6.7 [+3.4, +10.2]** (+128/−74 q). W1 alone +3.8; solver learned to use Python more (tool calls/episode 0.88→1.79) and truncate less (0.22→0.10). H2 kept h1. 8096 generated samples. |
| dev_r7_sao_lr1e6 | SAO lr 1e-6 | stopped | 0.384 | 0.407 | – | – | W1 drifted to fewer tools (1.0→0.22) and more truncation (0.22→0.42); dominated by 2e-6. |
| formal_hwh_sao | frozen contract hwh_v1 (same config as dev_r7) | **collapsed; contract void** | 0.384 | 0.390 | **0.214** | **0.209** | Training rollouts looked healthy (reward 0.34-0.42, stable entropy), but a gradient spike at step 29 (norm 6.0 vs ~0.9) left the saved W1 policy writing a degenerate `a\n` prefix before its code in 59% of tool calls (1506/2568 NameErrors). Same config: +6.7 in dev_r7, −17.5 here → not robust. Baseline tuning reopened (`hwh_v1.VOID.json`); MA is not compared against this control. |
| dev_r8_ma_v4 | MA: 8-step critic warm-up + 1 partner episode per fresh episode + 25% H2 replay; hwh_v1 budget | h1w1 read | 0.388 | 0.414 | 0.443 | – | Healthy W1 (tool error rate 0.25, truncation 0.13): +5.5 vs no-update after W1. Warm-up brought value loss 13.3→0.26 before the first actor update. Pool: 1280 partner episodes, 20 harnesses; DIS rejection <0.15% even at staleness 24. |
| dev_r8_ma_v4 (final) | see above | done | 0.388 | 0.414 | 0.443 | 0.430 | **+4.2 [+1.4, +7.2]** vs no-update. H2 picked a selection winner that lost 1.3 on report (selection noise). |
| dev_r9_sao_b64_s42 | SAO lr 2e-6, W 16×64 | done | 0.400 | 0.390 | 0.434 | 0.434 | +3.3 [−0.2, +6.7]; no collapse. |
| dev_r9_sao_b64_s43 | same, seed 43 | done | 0.397 | 0.413 | 0.404 | 0.404 | +0.7 [−2.2, +3.5]; no collapse, W1 −0.9. |
| **dev_r10_sao_b64x32_s42** | **FROZEN SAO baseline (hwh_v2_paired), seed 42** | done | 0.377 | 0.405 | 0.393 | 0.393 | +1.6 [−1.9, +5.1]; W1 −1.2, no collapse. 16000 generated samples. |
| **dev_r10_sao_b64x32_s43** | **FROZEN SAO baseline (hwh_v2_paired), seed 43** | done | 0.374 | 0.396 | 0.459 | 0.451 | **+7.7 [+4.8, +10.7]**; W1 +6.3. 10176 generated samples. |
| **dev_r11_ma_v5_s42** | **MA v5** = 8-step critic warm-up (2 partner episodes/row) + 1 partner episode per fresh episode + 25% H2 replay; hwh_v2 budget | done | 0.392 | 0.391 | 0.462 | **0.470** | +7.8 [+4.4, +11.3] vs no-update. **MA − SAO(s42) = +7.7 [+4.2, +11.3]** (+130/−75 q). |
| **dev_r11_ma_v5_s43** | same, seed 43 | done | 0.392 | 0.414 | 0.463 | **0.468** | +7.6 [+4.6, +10.6] vs no-update. **MA − SAO(s43) = +1.7 [−1.3, +4.5]**. |
| dev_r12_ma_warmonly_s42 / mixonly_s42 | ablations of MA v5 | running | | | | | Which component matters. |

## Findings

1. **Packed-SDPA contamination (fixed).** The pinned AReaL FSDP path packed
   several sequences per micro-batch under SDPA without a block-diagonal mask.
   Solver tokens rejected by DIS: 33-60% (editor ~5%). Reference HF forward vs
   vLLM behavior logprobs: |Δ| 0.01-0.04, ≤0.2% outside [0.3, 5]. Fix:
   `areal-packed-sdpa.patch`, packed logits match separate forwards to 4e-5.
   All earlier solver training was invalid for both arms.
2. **Throughput.** W phases use all 8 GPUs (actor d4 + rollout d4); harness
   candidates are scored concurrently; CUDA graphs enabled; one checkpoint save
   per phase. W1 step ≈ 8 s for 32 fresh episodes (baseline), ≈ 16 s with
   ~70 extra recycled sequences (MA k=1, pre-fix timing). Evaluation: 640
   tool-use episodes in ~50 s on 4 replicas.
3. **Experiment lifetime.** Attached shells are terminated when a new user
   message arrives; long experiments must run detached, logging to files.
4. **Baseline learning rate (r3).** With solver lr 2e-6 and 16×32 fresh episodes,
   H→W→H changes report accuracy by only +1.3 points (inside the ±3.5 noise).
   H1 kept the incumbent; H2 picked a candidate that won on selection (0.355 vs
   0.328) but not on report. Next: tune solver lr (5e-6, 1e-5) with editor lr
   5e-6; report set widened to 512 questions × 2 samples (128 dev + 384
   never-trained train-tail questions, guarded against solver offsets).
5. **Search-then-verify proposals (from r4).** Post-training proposal
   generation needed a separate 2-3 min vLLM start-up per H phase. Candidates
   now come from the editor's own training proposals (already scored on the
   phase's reward questions): the top 8 distinct valid harnesses are verified on
   the selection split. Same rule for both arms; no extra samples.
6. **Context overflow (fixed).** After W training, a solver can issue several
   long tool calls per turn; evaluation then asked vLLM for prompt+generation
   beyond 8192 tokens (HTTP 400), crashing the run. The AReaL training client
   instead clips generation to the remaining context and refuses a full prompt.
   Evaluation now applies the same rule (exact prompt count via vLLM
   `/tokenize`, one clipped retry); a full context ends the episode as
   truncated in both evaluation and training (`ContextExhausted`).
7. **SAO critic cold start (MA-SAO v3 hypothesis).** At the start of W1 the
   fresh value head has value loss ≈ 14 on 0/1 rewards and needs ~4-5 steps to
   calibrate; with G=1 the critic is SAO's only baseline, so ~25-30% of a
   16-step phase uses near-random advantages, and every alternation repeats
   this. The H-phase partner-evaluation episodes are exact samples of the
   W-phase initial policy, so MA-SAO runs critic-only warm-up steps on them
   before the first actor update (no new samples; native SAO critic updates).
8. **Training tool calls were never parsed (fixed).** The training client used
   AReaL `tool_call_parser="qwen"`, which maps to vLLM's Qwen3-Coder XML parser.
   Qwen3-4B-Instruct-2507 emits hermes JSON (`<tool_call>{"name":...}</tool_call>`):
   0/61 real training tool calls parsed with `qwen`, 56/61 with `hermes` (the rest
   are malformed JSON, also unparsed at evaluation). Training rollouts therefore
   never executed Python, while evaluation did — every W phase so far trained a
   different environment than the one measured, for both arms. Fixed to `hermes`.
   Solver turns are now exported one exact sequence per turn (`individual`):
   re-rendered tool-call JSON need not reproduce the sampled tokens, so concat
   multi-turn sequences could silently misalign behavior log-probabilities.
9. **Critic calibration speed.** SAO critic: lr 2.5e-5, value clip 0.5,
   gradient clipping 1.0; fresh value-head gradient norms are 50-150, so each
   update is scaled down ~100×. The baseline's value loss needs ~7 steps
   (14 updates) to fall below 0.2. MA warm-up therefore needs ≥8 steps.
10. **lr 5e-6 is a noise-driven random walk (both arms).** SAO uses Monte-Carlo
    advantages (discount 1, GAE λ 1), so the gradient is unbiased; yet with
    G=1 and 32 episodes per step the direction is dominated by noise, and Adam
    moves every step by ~lr. The two arms drifted in opposite directions
    (baseline → always-short answers; MA → long, truncated, tool-free answers),
    both losing reward. Calibrating the critic does not fix this. lr 1e-5
    collapses by step 3, 5e-6 drifts, 2e-6 (no-tool r3) was stable but flat.
    Implication for MA-SAO: variance reduction per step is the binding
    constraint — exactly what free partner-evaluation data can supply.
11. **Start-up overhead.** Each W/H trainer start took ~8 min, mostly Python
    imports: `import vllm` from the shared 3FS venv takes 45 s, and every vLLM
    server imports it in three sequential processes. From a node-local tmpfs
    copy it takes 6.8 s. `scripts/run_experiment.sh` rsyncs the venv to
    `/dev/shm/sao-venv-ppo` (32 s when unchanged) and runs from it; code,
    models, data and results stay on shared storage. Used from r8 onward.
12. **Baseline frozen.** The tuned configuration (dev_r7_sao_lr2e6) is frozen
    in `runs/contracts/hwh_v1.json`: initial model and data digests, every
    sample-budget quantity (steps, batch sizes, reward questions, proposals,
    selection/report sizes and samples, token limits, staleness, cycles) and
    the baseline hyper-parameters. `validate_trial` rejects any MA run that
    changes a sample-budget quantity; MA may change learning rates and its own
    mechanisms. The formal baseline is claimed exactly once. AIME stays sealed.
13. **SAO baseline instability (G=1).** Identical configuration: +6.7 (dev_r7)
    vs −17.5 (formal). The degenerate `a\n` code prefix appeared in 35% of
    tool calls in the formal run's very first batch (before any update, not
    seen in dev_r7), vanished for 25 steps, and returned in the last steps.
    Per-turn credit gives an erroneous tool call the episode's final reward
    when the solver recovers in a later turn, so such artifacts can be
    reinforced; with 32 episodes per step and no group baseline, variance is
    high enough that one run improves and another collapses. Remedy being
    tested for the baseline: 64 episodes per step at the same total samples,
    two seeds, before re-freezing.
14. **Batch vs steps (baseline).** With 1024 W episodes: 32×32 → +6.7 or
    −17.5 (can collapse); 16×64 → +3.3 / +0.7 (stable, not significant).
    Next: 32×64 (2048 episodes), two seeds; freeze as hwh_v2 only if both
    beat no-update significantly. Runs now take ~40 min (local venv).
15. **Baseline frozen (hwh_v2_paired, sha 670c8dac…).** SAO, solver lr 2e-6,
    W 32×64, editor lr 5e-6, H 3×16, 64 reward questions per harness, 8
    proposals, select 128×2, report 512×2, 2048 tokens. Seeds {42, 43}; both
    runs are the frozen reference (no selection among runs). Mean gain over
    no-update +4.6 (+1.6, +7.7). Tuning history: 32×32 could collapse
    (+6.7/−17.5), 16×64 was stable but weak (+3.3/+0.7). `hwh_v1` is void
    (collapsed formal run); `hwh_v2` (single seed) superseded before use.
    MA runs must pass `validate_trial` against this contract (identical
    sample-budget quantities and seed set).
16. **First MA-SAO result (hwh_v2_paired).** Final report accuracy, mean of
    seeds 42/43: SAO 0.422, MA v5 0.469 (+4.7). MA gains over no-update are
    consistent (+7.8, +7.6) while SAO's vary (+1.6, +7.7): partner-data critic
    warm-up plus extra partner episodes damp the seed-dependent W-phase drift
    along the tool-use/truncation axis (seed 42: SAO drifted to fewer tools
    0.86→0.57 and more truncation 0.23→0.39; MA held 1.1-1.2 tools and 0.11
    truncation). Same generated-sample caps; MA's W phase costs ~2× GPU time
    (≈285 vs ≈145 sequences per step, plus ~10 min warm-up).
17. **Physical sample-budget correction.** Reward-service caching by harness
    source made actual generated samples depend on how often the editor repeated
    an action: r10 SAO seeds generated 16,000 vs 10,176 samples; MA seeds 10,416
    vs 12,592. Thus hwh_v2 results establish a strong algorithmic signal but
    are not the final equal-physical-sample comparison. Fixed protocol (r13):
    every editor action performs an uncached candidate/reference pair with common
    random numbers (invalid actions sample incumbent/incumbent but retain -1);
    16 reward questions, exactly 64 requests/H; exactly 8 post-training proposal
    slots (duplicates sampled independently, grouped for selection); selected
    report always resampled. A runtime assertion requires exactly 64 action
    requests. This makes every physical generation category fixed. hwh_v3 will
    be frozen after paired baseline seeds finish.
18. **Strict physical-budget baseline calibration.** r13 is the first pair with
    exactly equal ledgers: 128 editor training actions, 16 editor proposals,
    2,112 solver-training episodes, 4,096 paired harness-scoring episodes,
    4,608 selection and 4,096 report episodes = **15,056 generated samples per
    seed**. Results: seed42 0.408→0.438 (+3.0); seed43 0.382→0.485 (+10.3).
    r13 started before the final conservative harness gate was installed; r14
    repeats it with common-random-number candidate scoring and Bonferroni
    one-sided paired-bootstrap acceptance. r14 will define hwh_v3.
19. **MA v6 design after review.** Under the strict protocol each H phase yields
    2,048 unique partner episodes (64 actions × candidate/reference × 16
    questions). Use the first 1,024 exactly once for 8 critic-only warm-up
    batches (2 episodes/row), and the remaining 1,024 exactly once across W actor
    batches (50% extra sequences). This removes v5 pseudoreplication (256
    episodes cycled ~8×). Reward seeds bind to deterministic action IDs, not
    concurrent arrival order. Primary estimand is question-level
    `(MA_final−MA_h0)−(SAO_final−SAO_h0)`.
20. **r14 seed 43 completed (2026-09-29 UTC).** Persisted stage report
    accuracies are 0.387695 -> 0.397461 -> 0.450195 -> 0.444336.
    The final-vs-initial gain is +5.664 percentage points; the exploratory
    question-cluster 95% interval is [+2.344, +8.984] over 512 questions,
    mean@2. H1 and H2 both retained the incumbent; repeated evaluation of the
    same checkpoint/harness is stochastic, so the H2 score change is not a
    measured harness improvement or regression. The complete physical sample
    ledger was independently verified at 15,056. AIME remains unopened.
    The run's original spec has `formal_baseline: false` and no `contract`;
    external claims do not constitute runtime enforcement. Preserve that
    provenance rather than retrospectively relabeling the run as formal.
    Seed 42 was still finishing its final evaluation when this was recorded.
21. **Reporting correction (no training change).** `stage_reports()` previously
    substituted the pre-selection report whenever the incumbent was retained,
    even when a fresh selected-report receipt existed. For r14 seed 43 that
    would have reported 45.020% instead of the actual final 44.434%. It now
    prefers the completed selected-report receipt and does not expose an
    incomplete final stage. Legacy runs without a resampled report retain their
    original behavior. Seven focused analysis tests pass; the checked result is
    persisted in `runs/dev_r14_sao_final_s43/verified_summary.json`.
22. **r14 pair finished; user pivoted to SFT initialization.** Seed 42:
    0.405273 -> 0.389648 -> 0.408203 -> 0.414063, final gain +0.879
    percentage points (question-cluster interval [-2.346, +4.102]).
    Seed 43 final gain is +5.664 points. Mean final score is 0.429199 versus
    initial 0.396484 (+3.271 points). Both complete ledgers equal 15,056;
    sampled actor/backbone tensors changed at H1/W1/H2 and were finite, and
    the carried H2 critic value head also changed. These are development
    results, not retrospectively formalized controls. The user requested a
    new SFT initialization before further SAO; preserve these checkpoints and
    do not pool their scores with the new starting-model protocol.
23. **2026-09-29 SFT pilot protocol (before outcomes).** Allowed compute hosts
    are now node3/node4 only. Node0's old utilization service was restored and
    all eight worker identities verified; no further node0 compute is allowed.
    An authenticated localhost FastAPI active-agent queue requests 192 original
    edits: 128 train, 32 development and 32 test across disjoint task families.
    Python configuration editing and strict JSON editing are both included;
    no mathematical test questions, answers or current harness optimization
    scores are used as teacher content. Teacher identity is recorded as an
    actual active Copilot SDK agent, not an invented unattended API model.
    The pilot uses native WHALE response-token SFT from the original Qwen3-4B
    initialization, lr 1e-5, two epochs, batch 16, FP32 optimizer/master weights.
    Compare original and SFT checkpoints on legality and exact edit fidelity
    at temperature 0.7, two draws per example, with unrequested fields protected.
    A separate 64-question DAPO development check (two draws, 2048-token cap)
    tests whether math/tool performance regresses. AIME stays unopened.
    In addition, generate 32 proposals with the actual SAO editor prompt at
    temperature 0.5 and a 768-token cap; report native harness legality and
    distinct configurations without claiming that diversity implies quality.
    This is format/instruction distillation, not evidence that 192 examples
    transfer broad teacher reasoning ability. Only measured improvements can
    justify adopting the SFT checkpoint; then rerun SAO from the declared new
    initial weights under a separately versioned protocol.
24. **Teacher data and first SFT update completed.** The authenticated queue
    accepted all 192 authored records; 48 task families are split-disjoint.
    Dataset SHA256 is
    `0e591f14b0d9f0e4183c14084e39ac3b4fc4a06e31a8493ec18545e87415ec8d`.
    Twenty-two focused CPU tests passed before GPU execution. Native WHALE SFT
    on node4 finished 16 updates over 128 train records / two epochs, consuming
    18,504 response tokens in 154.75 seconds (peak allocated GPU memory 73.29 GB).
    Parameters remained finite, sampled weights changed, and persisted HF
    checkpoint probes matched. Held-out acceptance is still pending; low
    training loss (token-weighted mean 0.03450) is not evidence of generalization.
    The original model is being evaluated on node3, the SFT model on node4.
    `chengboliu` resolves to UID 1009 on node3 but 1008 on node4; 3FS does not
    support POSIX ACLs. Only the nonsecret dataset was copied to the readable
    `assets/rewrite_sft_dataset_v1` directory; teacher access credentials remain
    private. Fresh node3-owned cache/output subdirectories were provisioned,
    without changing ownership or permissions of other project/user data.
25. **First paired SFT check: reject as a shared initialization.** The node3
    original-model run completed its edit/proposal checks but failed before math
    evaluation because the math data directory was owned by node4's numeric UID
    with mode 0700. Its children were cleaned up. The complete control was rerun
    on node4 (`runs/rewrite_initial_v1_node4`); both successful trials now use the
    same host, original model, dataset, prompts, sample identities and BF16
    inference. `rewrite_compare.py` validates these identities and persists the
    paired results in `runs/rewrite_sft_v1_node4/paired_comparison.json`.

    | Check | Original | SFT v1 |
    |---|---:|---:|
    | Dev edit legality / exact instruction following (32 x 2) | 96.875% | 100% |
    | Test edit legality (32 x 2) | 100% | 100% |
    | Test exact instruction following (32 x 2) | 96.875% | 98.438% |
    | Actual SAO prompt, valid proposals (32) | 100% | 100% |
    | Math development mean@2 (64 x 2) | 31.250% | 11.719% |
    | Math truncated responses | 25.000% | 36.719% |

    The test edit difference is only +1.563 points, with an example-cluster
    interval [-4.688, +9.375]; it is not evidence of a robust general improvement.
    The math difference is -19.531 points, with a question-cluster interval
    [-28.906, -10.156]. Three questions improved and 23 regressed. This pilot
    therefore fails the mathematical capability-retention check and must not
    replace both SAO actors. In particular, it does not establish that the
    original harness bottleneck was malformed output: actual-prompt legality
    was already 100% in this check. Distinct proposals were 3 versus 8, but
    diversity is not task-score improvement (the incomplete node3 control had
    9; sampling/server execution can vary).

    Next bounded calibration, declared before its outcome: SFT from the original
    model at lr 2e-6 for one epoch, with identical training data and native WHALE
    loss. Evaluate development edits, actual editor proposals, and the same math
    development check only; do not reuse the opened edit test for model selection.
    This is initialization safety calibration, not an MA-SAO algorithm result.
    A small/uncertain format gain cannot override a large math regression. If
    pure editing supervision cannot preserve solver ability, use role-specific
    initialization or verified math rehearsal rather than silently accepting a
    degraded solver. AIME remains unopened.
26. **Pilot preflight hardened.** Math inputs are now read and validated before
    releasing utilization workers or allocating GPUs; `--math-data` allows an
    explicitly readable shared copy without broad permission changes.
    `--dev-only` explicitly suppresses held-out edit evaluation during tuning.
    The new regression test verifies missing math inputs allocate no GPUs and
    create no trial directory. Existing failed-run records are preserved.
27. **Gentler SFT v2 retained the math point estimate, not a proven stronger base.**
    Native WHALE completed 8 updates / 9,252 response tokens in 130.52 seconds;
    checkpoint probes matched and all parameters were finite. Development edit
    legality and exact instruction following were 100% (original: 96.875%).
    Math mean@2 was 31.250%, equal to the original model on the same 64 questions.
    Its paired question interval for the difference is [-8.594, +7.813] points,
    so this is not a powered equivalence/noninferiority result. Math truncation
    fell from 25.000% to 14.844%. Held-out edits were not evaluated.
    Actual-prompt proposals were all legal but had only one distinct
    configuration out of 32 at temperature 0.5: proposal collapse, rather than
    richer harness improvement, is a remaining concern. Do not claim teacher
    reasoning transfer or improved harness quality.

    Proceed with a bounded SAO development probe, not a frozen control, to check
    whether this usable but unproven initialization improves after H-W-H.
    Both actors independently start from the exact v2 checkpoint; retain the
    native SAO loss and separate critics. H uses 2 x 16 updates per phase and W
    uses 16 x 64 fresh training episodes. The matched physical-generation cap is
    12,976 including prefetch, reward evaluations, eight proposals per H stage,
    selection and four reports. Reports use the same 512 untrained questions
    and mean@2 as r14, but old initialization runs are not controls for this
    probe. No AIME evaluation. Node4 is the only allocated compute host, with
    the existing exact filler-only audit. Estimated duration is 30-45 minutes;
    record actual runtime on completion. Spec:
    `runs/specs/sftv2_sao_probe_s42.json`.
28. **2026-09-29 04:22 UTC progress check.** The SFT v2 independent-SAO probe
    has completed H1's two actor updates and four critic updates, and saved both
    checkpoints. Its native training receipt reports 348.56 seconds. The
    unupdated H0/W0 report is 40.039% over 512 questions, mean@2.
    Eight post-training proposals were all legal and contained five distinct
    sources; proposal selection/reporting is still in progress, so this is not
    yet evidence of harness-score improvement. W1 and H2 are not completed.
    H1's two actor compute sections total 2.32 seconds and its two critic
    sections total 3.81 seconds; final checkpoint saving took 98.45 seconds.
    Thus short-run wall time is dominated by overhead, rollout/reward evaluation
    and persistence rather than optimizer arithmetic alone. Do not claim
    sustained full GPU utilization from memory allocation or a single snapshot.
    No completed/frozen baseline or new MA-SAO win is being claimed.
29. **User requested first-round-scale SFT and larger tests (04:29 UTC).**
    The pinned WHALE math launcher defaults to 14 weight-phase prompt batches
    (0.2 epoch), batch 256 and eight rollouts per prompt: 3,584 prompt exposures
    and 28,672 generated trajectories. Only verifier-accepted trajectories
    enter its response-token SFT; 28,672 is NOT a documented accepted-training
    count. The documented main-comparison 0.6-epoch schedule instead corresponds
    to 42 batches / 86,016 generations. We explicitly use the repository-default
    first-round reference, not conflate these two schedules.

    The new campaign targets 28,672 valid teacher rewrite records spanning at
    least 3,584 independent contexts, with 512 development and 1,024 fresh
    held-out test examples, eight samples per evaluated example. This is a
    conservative chosen SFT target, not a claim that WHALE trained on exactly
    that many accepted samples. Task families/source templates must be isolated
    across splits; exact harness0 sources are capped at 1% and the entire ReTool
    family at 10%. Meaningful prompt rewrites, tool-feedback repair, instruction
    changes and unrelated configuration domains replace expansion by merely
    changing values or names. Open-ended semantic rewrites require a separate
    transfer-quality check; exact wording comparison is only valid for the
    unambiguous edit suite. The opened 32-example pilot test is not a tuning set.
    AIME remains sealed. Mathematical retention and proposal diversity are
    acceptance gates, not assumed consequences of more SFT.

    Only node3/node4 may be used. Node3's eight GPU PIDs were checked against the
    exact bruli worker commands, UID 1012 and systemd cgroup; no foreign training
    was found in that snapshot. Its filler has NOT yet been stopped; re-audit
    immediately before allocation. Node4's existing small H-W-H run may finish
    as a diagnostic, but it is no longer a candidate for freezing the requested
    larger baseline. No more small-budget tuning runs will be launched.

    Generation is still through actual active-agent authors and an authenticated
    FastAPI queue, not an unattended model API on the H200s. The previous
    96-record shards took 14.2 and 44.4 minutes from request to acceptance
    (including scheduling/validation, not pure decoding). This makes teacher
    authoring and QC the dominant and uncertain cost. A planning range for the
    full SFT initialization plus enlarged evaluation is 12-36 hours with
    parallel authors, including an estimated 1-3 hours of multi-GPU training
    and 1-3 hours of evaluation/verification. These GPU estimates are not yet
    measured at scale. First calibrate a 512-training-record batch and native
    multi-GPU throughput, then narrow the forecast. Subsequent SAO/MA-SAO
    experiments are not included in this estimate.

    Machine-readable campaign settings: `assets/rewrite_round1_campaign.json`.
    Single-split teacher shards are now supported with explicit nonnegative
    quotas, allowing training data expansion without repeatedly regenerating
    or exposing the held-out sets. Empty/negative/missing split quotas are
    rejected.
30. **Small SFT v2 SAO probe finished; it fails the acceptance gate.**
    Runtime was 2,686.37 seconds (44.77 minutes). The 512-question mean@2 stage
    reports are 40.039% -> 38.574% -> 29.492% -> 28.027%.
    Final minus initial is -12.012 points, with a paired question interval
    [-15.234, -8.789]. Both harness selections retained their incumbent, so
    changes between repeated same-checkpoint/harness reports are sampling
    variation, not measured harness changes. The W-update configuration degraded
    the end-to-end result. This single diagnostic does not identify whether the
    underlying cause is initialization, update settings, critic behavior or an
    interaction; do not assert that simply adding more SFT will fix it.
    All 12,976 physical generations were independently reconciled. Raw stage
    outputs and `verified_summary.json` remain under
    `runs/sftv2_sao_probe_s42`; the run is not a successful or formal baseline.

    Checkpoint verification now compares initial tensors at the saved checkpoint
    precision, since this SFT source is FP32 and native SAO exports BF16.
    Conversion alone must not count as a learned update. Three regression tests
    passed, and actual H1/W1/H2 actor and critic probes changed beyond conversion.
    This remains sampled-tensor persistence evidence, not proof about every
    weight or about task quality. See `checkpoint_verification_precision_aware.json`.
31. **Large-campaign production and audit infrastructure.** The first JSON
    workflows shard was accepted: 128 train / 16 dev / 32 test, 28 isolated
    families, 80 training records with substantive prompt edits, in 9m44s from
    author start to acceptance. Semantic review so far is by its author only.
    The next 128-record training-only shard has been assigned; the other initial
    authors continue independently. Selected-shard assembly can freeze a
    calibration snapshot without waiting for later queued shards.

    `rewrite_audit.py` checks canonical duplicates, cross-split families and
    field schemas (ignoring class/module renaming), declared harness0/ReTool
    concentration caps, and full-budget counts when requested. Eleven focused
    data/audit tests pass. These checks explicitly do not prove semantic
    originality or useful rewriting; independent content review and transfer
    evaluation remain necessary. The verified primary-source method constraints
    are summarized in [MA_SAO_DESIGN_REFERENCES.md](MA_SAO_DESIGN_REFERENCES.md).
32. **Local-root storage incident resolved without touching others' data.**
    Node4's 29 GB root filesystem filled and the session task database rejected
    a write. Shared 3FS still had about 80 TB free. A 129 MB temporary search
    output was positively identified as this task's old r10-run inspection;
    it was moved intact into the shared runtime-cache archive with its original
    temporary path preserved as a symlink. This restored 124 MB on local root
    and the database write succeeded. No model, experiment receipt, other user's
    files or broad cache directory was deleted. Persistent large artifacts
    continue to live on shared storage; local-root headroom needs monitoring.
33. **More specific failure evidence and source clarification.** The failed
    SAO probe did not regress through increased token truncation: truncation
    fell from 14.941% to 6.543%, while tool calls rose from 1,082 to 2,414,
    tool errors from 228 to 757, and boxed outputs fell from 81.152% to 72.363%.
    NameError messages rose from 165 to 569. A conservative static inspection
    found that 67 before-update and 173 after-update missing names had been
    defined/imported in an apparently successful earlier tool call. Our executor
    actually starts a fresh Python process for each call. This supports testing
    tool-state/definition awareness, but does not prove that persistent execution
    or one prompt edit would recover the score. The remaining NameErrors have
    not been attributed. No generated code was executed for this inspection;
    evidence is in `tool_error_context_audit.json`.

    Primary-source review also found SAO's stated ten-step value warm-up and
    mathematical TIR fine-tuning initialization. The exact warm-up scheduling
    and source data are not specified sufficiently in that paragraph, so do not
    invent an actor-frozen ten-batch reproduction. Our pinned unofficial
    prototype has no corresponding warm-up in its independent run. This is a
    baseline-calibration issue to investigate, not our novel contribution or a
    proven explanation of the score drop.
34. **Recurring disk pressure and avoidable authoring latency.** Root pressure
    returned after the first cache move. Another four positively identified
    project file-list caches (about 107 MB total) were archived intact with
    original paths preserved as symlinks. This is only mitigation: node3 also
    had about 3.4 MB local-root headroom. Read-only inspection found a roughly
    100 MB deleted-open file held by a root-owned filesystem mount process;
    that process was not stopped or modified, and no system logs were changed.
    Large persistent artifacts remain on 3FS and transient tests/IPC use tmpfs.

    Two authors independently reported an exact 30-minute IDE-diagnostics
    timeout; actual authoring/validation was about 10-12 minutes per shard.
    All active implementation/authorship agents have been told to use direct
    Python validation/tests, not retry the unresponsive IDE adapter. This
    tooling delay must not be mistaken for teacher decoding time.
35. **512-record calibration snapshot and content-QA finding.**
    `assets/rewrite_round1_calibration_512_v1` contains 512 train / 48 dev / 96 test
    records, SHA256
    `8a3d71f3487617bfd2401946f4491e785acac7939e94275ab0b9ded8d58a89ff`.
    Combined structural checks found no duplicate IDs/canonical cases or
    cross-split field-schema/family collisions; harness0/ReTool fractions are
    zero. Node3 independently read the same hash. Only nonsecret data was made
    readable; the teacher token and raw queue remain private. This snapshot is
    for throughput calibration, not the requested full training run.

    Independent inspection of seven selected cases found internally consistent
    goals/edits, but exposed an important distinction: many long prompt edits
    supply the whole target wording in the input. Those teach fidelity/copying,
    not free rewriting. Among these 512 train records, 144 had potentially
    goal-synthesized long text by the lexical screen; one additional lexical
    positive was merely an escaped-newline replacement, confirmed by review.
    That screen is not a semantic-quality proof. Accepted examples remain
    immutable and can contribute to a bounded fidelity subset. Future shards
    are instructed to contain at least 104/128 genuinely goal-driven rewrites,
    with full target wording absent from the input and sidecar type annotations.
    The final corpus must predominantly teach synthesis rather than just grow
    a copying dataset. Exact held-out edits and free-rewrite transfer are
    separate evaluation objectives.
36. **Scaling interfaces and role safety, without a new performance claim.**
    Evaluation now supports eight samples, declared token/context budgets and
    up to eight independent replicas. Evaluation-only checkpoints carry both
    their actual path and declared original base, and never launch training;
    dev-only mode does not open edit tests. Ten focused evaluation/launcher tests
    passed. An optional `editor_model` also separates editor initialization from
    solver initialization, and contracts bind both digests; legacy shared-model
    behavior remains the default. Unvalidated shared-critic/distinct-init
    combinations fail explicitly. Twenty focused budget/phase/evaluation tests
    passed. These are verified CPU interfaces, not yet multi-GPU SFT validation
    or evidence of a successful larger model.
37. **2026-09-29 recovery: native eight-H200 SFT calibration completed.**
    The latest recovered conversation was checked before allocating compute.
    Only the previously authorized, identity-checked bruli filler occupied
    node4; node3 was inspected but not allocated. The existing opt-in two-GPU
    numerical-equivalence test passed in 11.24 seconds. Two attempts using the
    tool's detached launcher were terminated after its PID-acknowledgement
    timeout while the local root filesystem remained full. Their partial
    outputs are preserved and are NOT completed/resumed checkpoints.

    A fresh run from the original Qwen3-4B-Instruct-2507 then completed under
    the user systemd unit `sft-round1-calibration-20260929`. Its actual process
    end time was **2026-09-29 07:44:47 UTC (15:44:47 UTC+8)**, exit code zero.
    Artifacts are in
    `runs/rewrite_round1_calibration_node4_20260929T0740Z/training_verified`.
    This is the frozen **512-train-record throughput calibration**, not the
    28,672-record production run or an accepted stronger initialization.
    Global batch 128, one epoch, learning rate 1e-5, seed 42 and eight replicas
    produced exactly four updates with no dropped/repeated training examples.
    Dev/test targets did not enter the optimizer.

    Measured training time was **14.16 seconds**, or **36.16 examples/second**;
    steady update times were 3.29-3.39 seconds. Native loss decreased from
    0.4899 to 0.1407 (aggregate response-token-weighted loss 0.2968).
    All parameters were finite, replica probes agreed, and the saved FP32
    checkpoint and tokenizer passed full HF CPU reload/probe verification.
    The trainer reported 133.44 seconds including loading, saving and reload,
    excluding its initial imports/tokenization. Low training loss is not
    evidence of retained math ability or useful harness rewriting.

    At the same sequence-length distribution, 28,672 examples project to
    about **13.2 minutes of optimizer work**, not the previous unmeasured
    1-3 hour training estimate. This is a throughput extrapolation, not an
    end-to-end production ETA: longer synthesis examples and checkpoint I/O
    must be budgeted separately.

    The authenticated teacher queue was restored without changing its token
    or immutable responses, under `rewrite-teacher-queue-20260929`.
    At recovery it contained **640 train / 64 dev / 128 test** accepted
    examples, still far below the production target. The pending feedback_002,
    workflows_003 and pipelines_002 shards were assigned to three active
    authors, with the latest >=104/128 goal-synthesized training-edit
    requirement and explicit sidecar annotations. The API remains a queue,
    not an unattended teacher inference service; a server running alone does
    not imply continued data production.

    A matched, dev-only original/checkpoint evaluation is running under
    `sft-round1-dev-eval-20260929`: the frozen 48 edit-dev examples x8 and
    all 256 existing math-dev questions x8, plus the existing proposal
    legality/diversity check. This is a calibration diagnostic, not the full
    512-dev/1,024-test rewrite acceptance suite; edit tests and AIME remain
    excluded. No new SAO training or baseline-freezing claim has been made.
    A production SAO start still requires the complete audited teacher corpus,
    semantic/synthesis QA, larger rewrite/transfer evaluation, mathematical
    retention and the previously identified independent-SAO initialization/
    warm-up protocol checks. A fixed calendar completion promise is not yet
    supported by an uninterrupted, measured full-corpus production rate.
38. **Calibration quality evaluation finished: reject this checkpoint.**
    Both matched dev-only evaluations completed successfully at
    **2026-09-29 07:53:33 UTC (15:53:33 UTC+8)**. The pipeline exited zero
    and restored the authorized filler. The existing comparison utility
    verified identical base provenance, data hashes, prompts, sampling
    identities and evaluation budgets before question-cluster bootstrapping.
    Results are in the calibration run's `paired_dev_comparison.json`.

    | Development metric | Original | 512-record SFT | Difference, percentage points |
    |---|---:|---:|---:|
    | Edit legality, 48 examples x8 | 100.00% | 91.67% | -8.33 [-14.32, -3.91] |
    | Exact instruction fidelity, 48 examples x8 | 100.00% | 74.74% | -25.26 [-34.11, -17.19] |
    | Math mean@8, 256 questions x8 | 34.67% | 31.93% | -2.73 [-5.22, -0.39] |

    Brackets are paired example/question-cluster 95% intervals, not
    training-seed robustness. Valid proposal diversity increased from four
    to fifteen configurations, but that does not establish harness quality
    and cannot compensate for the measured regressions. The training loss
    reduction and successful distributed/checkpoint validation established
    engineering throughput only.

    This calibration checkpoint is **not accepted for either role's formal
    SAO initialization**. Retain the original checkpoint and do not blindly
    scale the same lr=1e-5 setup to 224 updates. Data authorship continues
    under the pre-existing synthesis/diversity requirements; do not rewrite
    training examples to match these held-out development outputs. There is
    still no evidence that simply increasing the corpus size resolves this
    regression. The tentative one-to-two-day production planning range is
    conditional on both sustained authoring and a subsequently validated
    initialization; it is not a confirmed SAO launch date.
39. **Recovered authors completed; handoff to the newer progressive-SFT task.**
    The three resumed shards, `r1_json_feedback_002`,
    `r1_json_workflows_003` and `r1_python_pipelines_002`, were accepted and
    verified through both API and persisted responses. Each adds 128 train,
    16 dev and 32 test examples, with all 128 training examples annotated by
    its author as goal-synthesized prompt edits. Elapsed authoring plus
    validation was 19m04s, 19m50s and 21m22s respectively. Independent parent
    inspection of four training examples from the first two shards found
    coherent goal-driven edits; this limited sample is not a full semantic
    quality audit.

    The eight recovered, accepted shards are frozen at
    `assets/rewrite_round1_recovered_1024_v1`: **1,024 train / 112 dev /
    224 test**, with 128/28/56 split-isolated families. Combined structural
    validation found no duplicate IDs/canonical examples, family leakage or
    cross-split field-schema collisions; harness0/ReTool fractions are zero.
    Examples SHA256:
    `07cec481d8cf824f1b70ad9ff7e229b215365cf0522e0dd4defbb443bc091c95`.
    Original immutable shards were preserved. Newer r2 jobs are not included
    in this snapshot, and no held-out records entered SFT.

    **Superseding user instruction:** the newer active task requests a
    **2-3 hour progressive-SFT budget**, concurrent teacher collection and
    training, and early stopping on diminishing gains/overfitting rather
    than a mandatory 28,672 examples. Therefore the older full-corpus gate
    and one-to-two-day forecast above are historical, not the current plan.
    The snapshot, throughput evidence and rejection report were communicated
    to that task's training coordinator. This recovery task launches no
    additional GPU work, avoiding competing node allocations. The teacher
    API remains available for continuing collection. Quality acceptance,
    rather than elapsed time or GPU occupancy, still determines whether an
    SFT checkpoint is usable for SAO.
40. **08:06 UTC direct request to start immediately: next candidate running.**
    The user again requested immediate execution. A fresh live check found
    only the authorized filler on node4 and no active SFT/evaluation job.
    The newer coordinator was notified before allocation. Its current
    progressive/resume/distributed CPU tests ran directly: 19 tests passed,
    including one opt-in GPU skip, in 23.37 seconds. The known unresponsive
    IDE diagnostics adapters were not used.

    At **2026-09-29 08:09:17 UTC (16:09:17 UTC+8)**, user systemd unit
    `rewrite-progressive-start-20260929` launched the next bounded candidate
    in `runs/rewrite_progressive_start_node4_20260929T0808Z`: original
    Qwen3-4B, the frozen 1,024 training examples, eight H200s, global batch
    128, one epoch, learning rate **2e-6**, seed 42. No earlier rejected
    checkpoint was resumed. The new native training-state export is enabled
    so a successful candidate can later continue without resetting AdamW.
    At 08:10:23 UTC, the first real optimizer update had completed over
    128 examples; all eight GPUs held the training processes.

    The persistent pipeline automatically follows training with the same
    dev-only 48x8 edit and 256x8 math evaluation and a paired comparison to
    the completed original-model control. It does not launch SAO or open
    tests/AIME. The coordinator was also warned that an improvement gate
    requiring higher exact-edit accuracy than the current 100% control has
    a ceiling and cannot select any candidate. Therefore this launch is a
    bounded train/evaluate job, not an endorsement of that controller's
    current selection rule. New data collection can continue independently;
    further node4 allocation must wait for this unit to release its GPUs.
41. **Timeboxed progressive continuation, 08:30 UTC.**
    The user's latest instruction removes the 28,672-record requirement:
    collect independent teacher shards concurrently, consume each new record
    once, and stop on diminishing development gains rather than repeat a small
    corpus to fill GPU time. The current target is to stop optimizer work by
    **09:45 UTC / 17:45 UTC+8**, with final verification budgeted through
    **10:45 UTC / 18:45 UTC+8 on September 29**. Earlier stopping is allowed;
    these are runtime limits, not a promise of a better model or SAO readiness.

    The 1,024-record stage completed successfully at lr 2e-6. Its exact-edit
    development legality and fidelity both remain 100%; math mean@8 is
    34.277% versus 34.668% for the original, a -0.391 point difference with
    paired interval [-2.393, +1.611]. Actual-prompt diversity fell from four
    configurations to one. It is not an accepted stronger initialization.
    Its verified HF checkpoint and AdamW moments are imported, not retrained
    from scratch, by `sao_bridge.rewrite_progressive`.

    Because the original's exact-edit score is already at its ceiling, the
    prospective selection score for further stages is the equal-weight mean
    of fixed-dev edit fidelity and math accuracy. A gain of at least 0.5
    percentage point over the incumbent is required, with edit legality and
    fidelity no worse than one point, math paired 95% lower bound no worse
    than -3 points, actual-harness legality at least 95%, and at least two
    distinct valid proposals. Two non-improving checks stop training;
    material edit/math regression stops immediately. These are exploratory
    development gates, not proof of open-ended rewriting quality. No gate
    is based on training loss or GPU occupancy.

    New batches contain only unconsumed training IDs. The trainer validates
    the saved optimizer state's digest, checkpoint identity, hyperparameters
    and prior IDs, and restores AdamW steps/moments without resetting them.
    Every stage uses one epoch; test records never enter optimization.
    The best checkpoint is frozen before the final paired held-out check.
    `sao_ready` remains false unless that check passes; SAO is not launched
    automatically and AIME remains sealed.

    Direct CPU validation passed 44 tests (one optional GPU skip), followed
    by a real one-GPU tiny-Qwen two-stage save/reload equivalence test.
    The IDE test/diagnostics adapters stalled for 30 minutes; no training
    timing is inferred from that delay and subsequent checks use the CLI.
    The persistent controller writes its protocol, lineage, selection
    receipts and status to `runs/rewrite_timebox_20260929T0745Z`.
42. **Real incremental continuation verified; concurrent collection reached 1,408.**
    The three new r2 shards added 384 independently authored training records
    through FastAPI. The combined 1,408 train / 112 dev / 224 test corpus
    passes structural duplication, family and schema-isolation checks;
    harness0/ReTool fractions remain zero. Semantic annotations are author
    judgments, not evidence of model improvement.

    While two authors were still working, the controller picked up the first
    128-record r2 shard and resumed the eight-GPU 1,024-record checkpoint.
    Its persisted completion receipt verifies AdamW **8 -> 9 updates**,
    **1,024 -> 1,152 cumulative training records**, `optimizer_reset: false`,
    no repeated IDs, finite parameters, replica agreement, and successful
    HF/tokenizer reload. The two other new shards remain available for the
    next stage if development gates justify continuing.

    The new update itself took 3.80 seconds, but shared-filesystem checkpoint
    loading, integrity checks and persistent FP32 model/optimizer export made
    the full stage take 948.97 seconds. Pure GPU throughput is therefore not
    an end-to-end stage ETA. The automatic dev-only check follows; neither
    the completed update nor its training loss is an acceptance claim.
    Final focused CPU checks passed 25 tests with one optional GPU skip;
    that GPU continuity test was separately executed successfully.
43. **09:05 UTC user-authorized SAO / latest-algorithm comparison.**
    The attached completed SFT status freezes the original
    `assets/Qwen3-4B-Instruct-2507`, not the 1,152-record SFT candidate.
    The new comparison explicitly follows that selected initialization for
    both independently updated roles. It must not be reported as an improved
    SFT start. The first H0/W0 report precedes any update in every run.

    The superseded protocol is preserved in
    `runs/origin_pair_v6_node4_20260929/protocol.json`.
    Seeds 42 and 43 cross hosts: node4 runs independent SAO seed 42 then
    MA-SAO v6 seed 43; node3 runs SAO seed 43 then MA-SAO v6 seed 42.
    Both initial controls may run concurrently. A treatment waits for its
    same-seed control's completed checkpoint verification and immutable
    result handoff, as required by the formal contract.

    Both methods use H(3x16)-W(32x64)-H(3x16), solver lr 2e-6,
    editor lr 5e-6, 32 reward questions per H phase, eight final-policy
    proposals, 128x2 disjoint selection samples, and 512x8 stage reports.
    The common generation cap is 31,440: SAO uses 31,440 and MA 31,360
    fresh generations because 16 editor actions are replayed and 64 solver
    prefetch generations are absent. Final AIME2024+2025 mean@8 (60x8)
    is outside that training ledger, opened only after each run's H-W-H
    state is frozen. No settings will be selected from these test outcomes.

    Latest v6 means unique cross-role recycling, not the old repeated-pool
    v5: from 4,096 H1 partner trajectories, 1,024 disjoint episodes provide
    eight critic-only warm-up steps (two episodes/row), and 2,048 different
    episodes are mixed once into the 2,048 fresh solver episodes. H2 has
    25% authentic editor replay with recomputed partner rewards. Central
    critic and interaction-credit beta remain off. Extra optimizer work and
    GPU time are reported, not disguised as matched compute.

    Only node3/node4 are allocated. Exact filler command/UID/cgroup identity
    is audited before stopping the authorized service; unknown workloads
    are never stopped. Per-host persistent workers publish only nonsecret
    evaluation/ledger reports for the differing numeric UIDs. CPU preflight
    covered 25 existing safety tests, then 23 focused campaign/replay/budget
    tests, all passing. The pinned upstream revisions and both runtime
    patch digests were verified before launch.
44. **User clarification: actual SFT1152 checkpoint, concurrent one-arm-per-node.**
    The user explicitly corrected item 43: evaluate the SFT-trained model
    itself, then run independent SAO on one node and improved MA-SAO on the
    other, comparing harness-weight-harness gains. The original-model worker
    was stopped at 09:11 UTC before any optimizer update or initial report.
    Its artifacts and `CANCELLED.json` remain intact.

    The replacement protocol is `examples/sft1152_hwh_pair_20260929.json`.
    Both roles in both arms start from the exact latest 1,152-record SFT
    checkpoint under `runs/rewrite_timebox_20260929T0745Z/round_01/training`.
    This user-authorized experiment does not retroactively declare that SFT
    passed its earlier superiority gate. Both arms use seed 42: independent
    SAO on node4 and unique-pool MA-SAO v6 on node3, concurrently. Each first
    measures H0/W0, then H1/W0, H1/W1 and H2/W1 on the same 512 questions
    with eight samples each. This reports task improvement relative to its
    own no-update start as well as paired MA-minus-SAO and
    difference-in-differences. It is a single-seed controlled diagnostic,
    not a completed formal seed-robust baseline.

    The H/W update schedule, native losses, budgets and v6 settings remain
    those declared in item 43; no role or model substitution is allowed.
    AIME now stays sealed because this request concerns the H-W-H task
    diagnostic, not final benchmark selection. The persistent workers
    verify model/data identities, actual parameter changes, physical
    generation counts and disjoint once-only recycling, then publish a
    paired comparison when both arms finish. Twenty-four focused CPU
    campaign/budget/replay/launcher tests passed before the corrected launch.
45. **SFT1152 task start measured before RL; node3 provisioning unblocked.**
    The node4 independent-SAO arm's H0/W0 report completed on 512 untouched
    reporting questions x8 samples: **36.6455% mean@8**, 10.864% truncation,
    83.179% boxed outputs, 4,967 tool calls and 1,262 tool errors. This report
    generated 4,368,664 tokens in 273.13 seconds and is the relevant no-update
    SFT starting score for this H-W-H comparison. It is not interchangeable
    with the earlier 256-question 33.54% SFT development check.

    Node3's Python 3.12 / torch 2.10 / vLLM 0.19.1 runtime was restored into
    tmpfs; real isolated arithmetic, SymPy and statelessness checks passed.
    UID1009 could already read the SFT checkpoint, but could not traverse the
    old UID1008-owned dataset directory. Only traversal of that directory and
    read access to the nonsecret manifest/train/reward/dev files were granted;
    data bytes and the sealed test-file permissions were not changed.
    Git safe-directory trust is process-local and restricted to the exact
    SAO and AReaL worktrees, not a global wildcard. Root-disk writes are avoided.

    Historical comparable H-W-H runs took 61-63 minutes for SAO and 90-101
    minutes for MA v5. This experiment has larger eight-sample reports,
    twice the recent reward-question count, and cold SFT checkpoint loading.
    Initial planning is therefore roughly 1.5-2 hours for SAO and 2-3 hours
    for MA v6 from their respective launches; measured stage times will
    replace this range. These are execution estimates, not predicted gains.
46. **Both SFT1152 arms are performing real optimization (2026-09-29
    09:34 UTC / 17:34 Beijing time).** Node4 runs persistent user service
    `sft1152-hwh-sao-node4-20260929.service` (worker PID 2316723); node3 runs
    `sft1152-hwh-ma-node3-20260929.service` (worker PID 1208238). Both use
    the same unchanged protocol, SFT1152 initialization and seed42. Only
    the audited utilization filler was stopped; node3's eight GPU processes
    were verified to belong to its worker service cgroup.

    Node4 completed all three H1 actor updates, six critic updates and saved
    both checkpoints (native phase: 510.23 seconds). Sampled actor and critic
    backbone tensors are finite and changed beyond FP32-to-BF16 conversion;
    the saved critic has a value head. H1 proposal selection/reporting is
    still in progress. Node3's four frozen vLLM replicas returned HTTP 200,
    its independent H0/W0 report completed, and its first H1 optimizer step
    committed with finite nonzero actor and critic gradients.

    The two independent no-update reports on the same 512 questions x8 are
    SAO-host **36.6455%** and MA-host **37.1582%**. This initial 0.5127-point
    difference is not an MA training gain: both reports precede optimization.
    The final comparison must retain each arm's own start and report the
    difference in improvements, alongside stagewise paired intervals.
    These scores also do not establish SFT improvement over the original
    model, which was not re-evaluated on this reporting set in this campaign.

    Current estimated full H-W-H completion windows, in Beijing time on
    September 29, are **18:45-19:15 for SAO** and **19:30-20:30 for MA-v6**;
    the complete paired report follows the slower arm. These remain
    conditional estimates, not completion claims. Persistent workers will
    verify generation ledgers/checkpoint changes and publish the paired
    results automatically; failures are recorded explicitly. This is a
    single-seed diagnostic with AIME unopened, not seed-robust superiority.
47. **MA worker stopped on an accounting assertion, not a failed optimizer
    (09:37 UTC); phase-boundary recovery and continuous monitoring.**
    The user correctly reported a failed node3 worker. All three H1 actor
    updates, six critic updates, 64 unique native workflow receipts and both
    checkpoints were complete. All 399 actor / 400 critic tensors were
    finite; sampled backbone tensors had genuinely changed. Node4 remained
    healthy and continued its W1 optimization.

    The MA evaluator had 66 complete candidate/reference pairs, not the
    assumed 64. Requests 64 and 65 repeated existing task IDs with different
    actions and finished after rollout-worker shutdown started; neither was
    attached to a committed native workflow receipt. Logs support an unused
    asynchronous tail but cannot conclusively distinguish cycling from
    retry. The old assertion reported this overcount as "incomplete".

    Recovery validates the unchanged specification, every saved optimizer
    step and saved checkpoint before reusing a completed phase. It refuses
    to restart a partially updated optimizer without its recovery state.
    Existing initial evaluations, H1 updates and checkpoints are reused,
    rather than spending another H1 training budget. Failure metadata and
    critical logs were preserved under the MA run's `recovery_01/`.

    Pair validation now follows the request indices in authentic workflow
    receipts. Missing pairs, duplicated committed tasks and mismatched
    sources still fail. The two unused requests remain in the raw records
    and are excluded from partner recycling: the canonical pool still has
    4,096 unique episodes, sufficient for disjoint 1,024-episode warm-up and
    2,048-episode mixing. The known extra cost is at least **two editor
    actions plus 128 solver rollouts**. This is explicitly reported as
    overhead and **not strictly compute-matched**; cancelled in-flight tokens
    without receipts are not fully metered. The frozen protocol is not
    silently rewritten, and formal fixed-budget trials still reject this
    deviation. No loss, learning-rate or optimizer-step budget was changed.

    A persistent read-only monitor now checks systemd state, actual native
    optimizer counts, checkpoint saves, stage reports and progress age every
    30 seconds. It writes `MONITOR_STATUS.json` per node; the paired watcher
    writes `PAIR_MONITOR.json` and exits with an alert on worker failure,
    missing/stale heartbeats, or 30 minutes without progress. It never kills
    or blindly restarts training. Recovery/accounting/replay/launcher and
    monitor regression tests passed, and node3 CPU validation passed against
    the actual preserved failed run before the guarded resume was launched.
48. **Recovery and persistent monitoring verified at 10:11 UTC / 18:11
    Beijing time.** MA worker 1260653 passed the former assertion, completed
    H1 selection, and committed two real W1 critic-only warm-up updates
    (first critic gradient norm 202.8114, value loss 15.5310; finite metrics).
    All 13 guarded H1 checkpoint/metadata files retained their exact
    pre-resume SHA256 values, and H1 metrics remain steps [0, 1, 2].
    The selected H1 harness remained the incumbent; its 37.6709% mean@8
    remeasurement is not evidence of a harness change.

    SAO completed and persisted W1's 32 actor / 64 critic updates in
    1,840.39 seconds, with finite, genuinely changed sampled actor/critic
    tensors, and entered H2. Neither arm is claimed complete.

    Both per-node monitor services and the detached paired watcher are
    responsive. The paired watcher refreshes the SAO root's
    `PAIR_MONITOR.json` and signals this session when it exits on failure,
    stalling, or joint completion. Thirty focused regression tests passed,
    including real watcher exit-code tests (2 for attention, 0 only for
    completion). Revised conditional completion planning is 18:45-19:15
    for SAO and 20:00-21:00 for MA / the full comparison, Beijing time.
49. **Both arms completed; watcher exited successfully (2026-09-29).**
    Persisted COMPLETE receipt timestamps are **18:27:31 Beijing for SAO**
    and **20:02:36 Beijing for MA-v6**. All four worker/monitor services
    exited successfully, and the paired watcher returned 0 after observing
    both completed states. Training processes released their GPUs and the
    known utilization services were restored.

    All results below are the same 512-question, eight-sample reporting
    protocol (mean@8), not AIME:

    | Stage | SAO | MA-v6 |
    |---|---:|---:|
    | SFT1152 H0/W0 | 36.6455% | 37.1582% |
    | H1/W0 | 36.7676% | 37.6709% |
    | H1/W1 | 36.4014% | 47.3633% |
    | H2/W1 | 36.6699% | 46.7773% |

    SAO's final-minus-initial change is **+0.0244 percentage points**,
    paired-question 95% interval **[-1.6602, +1.7334]**. MA's change is
    **+9.6191 points**, interval **[+7.2998, +11.8652]**.
    The difference in improvements is **+9.5947 points** in MA's favor,
    interval **[+6.8353, +12.3785]**. These intervals cover question-level
    uncertainty, not variation across independent training seeds.

    Both H1 and H2 retained the incumbent harness in both arms. Thus this
    run supports a solver-weight-stage gain for the MA configuration, not
    a demonstrated improvement from accepting a new harness. Changes
    between repeated reports with the same solver/harness must not be
    interpreted as H-stage gains or degradation.

    Final physical accounting is **31,440** completed generated samples for
    SAO and **31,746** for MA (+306, about 0.97%). MA's total known unused
    overhead was 386 generated samples above its 31,360 plan: two fresh
    editor actions plus 384 reward-scoring samples, with four additional
    replay requests that generated no fresh editor action. All orphan
    evaluations were preserved and excluded from recycling. Recorded
    experiment execution times were **73.61 minutes** and **143.69 minutes**
    respectively (MA excludes the interruption gap, but includes both
    executed segments). This is neither equal-time nor strictly equal-
    generation-budget superiority; unreceipted cancelled tokens remain
    unmetered.

    Published report hashes and identical initial model/data/protocol
    identities were reverified. Saved actor/critic changes passed for all
    six completed phases. MA's 1,024 warm-up and 2,048 mixed episodes passed
    uniqueness, disjointness and single-use checks. AIME remained unopened.
    The MA root's `PAIRED_COMPARISON.json` includes the full uncertainty and
    final cost disclosure; both final model states are preserved.
50. **No-training Qwen editor capability probe (2026-09-29).** The user
    requested a newer roughly 8B Qwen to test whether editor capability
    limits harness improvement. Official release records identify
    **Qwen3.5-9B** (released March 2, 2026) as the applicable model; newer
    Qwen3.6/3.8 open general models in the checked catalog are larger.
    Official revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` was downloaded
    and verified (19.33 GB, 15 files, 775 indexed tensors). No runtime
    dependency was changed. A search summary incorrectly denied that this
    model existed; direct official model-card/catalog verification overruled
    that summary.

    The solver remains the completed MA W1 checkpoint, and the incumbent
    harness, scoring code, tool permissions and 2,048-token solver limit
    remain fixed. Four frozen editor conditions use the same prompt,
    temperature 0.5, seed block and eight proposal slots: original Qwen3-4B,
    the current trained 4B editor, Qwen3.5-9B direct output (all 768 tokens),
    and Qwen3.5-9B thinking (separately labeled 4,096-token budget).
    No actor/critic optimizer is instantiated. All model digests are checked
    before and after inference.

    Diversity counts semantic values of the six allowed harness settings,
    not formatting variations. Each proposal slot is evaluated on the same
    128-question x2 selection split; invalid slots use incumbent evaluations
    but cannot win. Duplicate configurations use their first slot rather
    than cherry-picking noisy duplicate scores. The best changed candidate
    per condition is frozen by selection accuracy, then evaluated on the
    disjoint 512-question x8 report even if it fails the conservative
    selection gate. This separates inability to generate useful candidates
    from a strict selection gate. Both per-arm historical and study-wide
    selection bounds are recorded; no harness is automatically deployed.
    Identical reporting configurations share a report instead of presenting
    resampling noise as a change.

    This is an exploratory diagnostic on previously observed development
    reporting data, not a new untouched test or proof that parameter count
    alone causes an improvement. The larger thinking budget is not a
    same-budget comparison. AIME stays sealed. Twenty-one focused
    inference/control/selection/launcher regression tests passed before
    the GPU run.
51. **Qwen3.5 runtime recovery, without regenerating completed controls.**
    The first probe generated both 4B controls successfully (8/8 valid,
    two semantic configurations each), then the 9B server failed on its
    first inference request. Its H200 FlashInfer Gated DeltaNet prefill
    backend attempted a JIT build but this runtime has no `nvcc`/CUDA
    toolkit. This was an inference-backend failure, not a model output or
    harness-quality failure.

    The retry uses vLLM's documented, installed
    `--gdn-prefill-backend triton` option for the 9B editor only; weights,
    prompt, sampling budgets and the frozen solver are unchanged. No
    dependency or shared runtime was upgraded. The original failed run
    and logs remain intact. A guarded pre-selection recovery checks all
    model digests and protocol fields, copies the two complete control
    proposal sets with hashes, and generates only the unfinished 9B
    conditions. Cached conditions cannot change their serving or sampling
    configuration. Twenty-two focused regression tests passed before the
    retry; its run directory has the `_retry1` suffix.
52. **9B can generate valid harness edits without task training.** The
    corrected runtime completed all proposal conditions. Both 4B controls
    produced 8/8 valid changed modules with two distinct configurations.
    Qwen3.5-9B direct mode produced **8/8 valid changed modules and seven
    distinct configurations**, with no length truncation. At the separate
    4,096-token thinking budget, only 1/8 proposals produced a valid final
    module; seven exhausted the budget. Thus thinking-mode output-budget
    failure must not be confused with inability to edit the harness.

    The user received this quick generation result while task-quality
    evaluation continued. Validity is not a positive task reward. The
    original 128-question x2 selection baseline was 37.8906%; best changed
    candidates scored 42.5781% (original 4B), 38.2812% (current 4B), 39.0625%
    (9B direct) and 37.1094% (9B thinking). None passed either the historical
    per-arm or study-wide conservative selection gate. These selection
    scores alone are not the final quality conclusion.

    The selected 9B direct candidate changes only MAX_TURNS from four to six.
    Its exported candidate module was checked against the exact frozen
    selected source. All four selected candidates were frozen before the
    disjoint 512-question x8 reports began. No harness has been automatically
    adopted and no RL training has been started.
53. **Completed four-condition reports (2026-09-29 13:41 UTC).**
    The earlier MA-W1-solver probe finished successfully in 45.38 minutes.
    Every report contains exactly 512 unique questions with eight samples
    each, identical paired question/sample IDs and verified aggregate
    rewards. The shared incumbent scored 46.7285%.

    | Frozen editor candidate | Report accuracy | Delta (pp) | Paired question bootstrap 95% interval (pp) |
    | --- | ---: | ---: | --- |
    | Original 4B | 48.1689% | +1.4404 | [+0.1221, +2.8320] |
    | Trained H2 4B | 47.8271% | +1.0986 | [-0.1953, +2.3682] |
    | 9B direct | 47.6318% | +0.9033 | [-0.3174, +2.1729] |
    | 9B thinking, 4096-token cap | 42.7979% | -3.9307 | [-5.5176, -2.3682] |

    No arm has a positive family-adjusted report lower bound. In particular,
    the 9B direct-mode gain is not statistically established. The original
    4B interval above is unadjusted, not evidence of a family-corrected win.
    The thinking result describes its sole valid completed candidate, not
    an uncapped first-H1 experiment. Models were unchanged, optimizer updates
    were zero, the unit exited successfully and node4's known filler was
    restored. Physical completed samples were 32 editor proposals (including
    16 reused controls), 8,448 selection solves and 20,480 reporting solves.
    Results are preserved in
    [COMPLETE.json](../runs/qwen35_9b_editor_probe_20260929_node4_retry1/COMPLETE.json).
54. **Single full first-H1 rewrite, separately from the earlier probe.**
    The user's next request removes short editor truncation and tests the
    first zero-training harness change at the original SFT1152 starting
    point. The new
    [protocol](../examples/qwen35_9b_first_h1_full_20260929.json) freezes that
    solver and its original h0 harness, rather than the stronger MA W1
    solver. It requests exactly one Qwen3.5-9B thinking proposal, seed 1042,
    without best-of-N or regeneration until a favorable result appears.

    HTTP requests omit both output-token-limit fields. The server uses its
    native 262,144-token context with neutral `--generation-config vllm`,
    not a hidden short generation-config limit. The editor request timeout
    is 5,400 seconds; native context and runtime safety limits still exist.
    Only a natural `finish_reason=stop` is eligible for evaluation, even
    when a length-truncated fragment would parse as valid code. Raw outputs
    are saved before this check. These editor changes do not enlarge the
    fixed 2,048-token solver budget or its 8,192-token context.

    The single candidate is fixed before its 128x2 diagnostic and 512x8
    paired report, irrespective of the diagnostic gate. No candidate is
    chosen using report accuracy. The reporting questions were previously
    observed, AIME remains sealed and no training optimizer is created.
    Twenty-four focused regressions passed before the node3 launch request,
    including uncapped HTTP serialization, native-context forwarding,
    rejection of incomplete outputs, single-proposal enforcement and the
    unchanged solver budget. The original capped probe remains a separate
    completed experiment; its results cannot substitute for this test.

    The single-proposal experiment completed at **2026-09-29 14:02:14 UTC**
    (22:02:14 Beijing), after **17 minutes 40 seconds**. The first and only
    9B output ended naturally after **5,086 completion tokens**, including
    reasoning; it was valid and changed four settings: MAX_TURNS 4 to 6,
    TOOL_OUTPUT_LIMIT 2048 to 4096, SYSTEM_PROMPT and RETRY_PROMPT. There
    was no length truncation or candidate regeneration.

    On the paired 512x8 report, the original harness scored **37.4756%**
    and the first full 9B rewrite scored **38.0859%**, a **+0.6104 pp**
    difference with question-bootstrap 95% interval **[-0.8789, +2.0996] pp**.
    Thus this first complete zero-shot edit is executable and has a small
    positive point estimate, but does not establish a reliable improvement.
    The report records 141 questions improving and 140 regressing in
    per-question mean@8. Neither diagnostic selection gate passed, and
    nothing was automatically deployed. The solver and editor hashes were
    unchanged, with zero optimizer updates. AIME was not evaluated.
    The full run used one editor generation, 512 diagnostic solves and
    8,192 paired reporting solves. A subsequent context-fallback/long-timeout
    regression brought the focused passing test count to 25.
    [First-H1 result](../runs/qwen35_9b_first_h1_full_node3_20260929/probe/COMPLETE.json).

    Final audit verified all 4,096 paired rollout seeds and exact equality
    with the original HWH SFT1152 initial-actor digest:
    `c910aaabd251aa592a18d0fd4ee823d0faf163883130da0a63e1084b69224178`.
    Actual server health was HTTP 200 with native context 262144,
    generation config `vllm` and Triton GDN. The unit exited zero, experiment
    GPU processes were cleaned up and the authorized filler was restored.
    The unedited 402-byte first generated module is preserved as
    [candidate.py](../runs/qwen35_9b_first_h1_full_node3_20260929/probe/candidate.py);
    it has not replaced the original harness.
55. **SearchQA and chess campaign requested (2026-09-29 14:25 UTC).**
    The user requested original-4B versus SFT1152 baselines on both domains,
    followed by independent SAO versus MA-SAO alternating optimization and
    SAO weight-only / harness-only convergence controls. Only node3/node4
    may be allocated, with the exact audited utilization service temporarily
    stopped. All previous math results and checkpoints remain unchanged.

    Each domain chooses its common solver/editor initialization by the
    original-versus-SFT development mean@8, not by test performance. The
    selected checkpoint identity is shared by all four optimization
    conditions. Development data controls acceptance, continuation and
    early stopping; test data is reporting-only. Saved actor, critic and
    harness states, rather than the original actor, are carried into
    continuation blocks. Native SAO phase handoffs still reset Adam moments;
    this is disclosed rather than presented as uninterrupted optimizer state.

    SearchQA preparation uses real retrieval and a fixed normalized
    reference-alias exact-match verifier. This is a local Search-R1-style
    metric, not WHALE's paid LLM-judge metric, so published WHALE percentages
    are not directly comparable. Chess retains legal-move and complete
    reference-sequence verification outside the editable harness. Both
    domains add bounded executable Python hooks for public-observation,
    query/document or action-processing behavior, instead of the earlier
    math experiment's six-literal editing space. Generated hooks run in
    an isolated, resource-limited process without hidden labels or project
    access. Editor training/proposals receive actual training-side failure
    traces; they never receive test failures or test answers.

    The initial proposed operating budget is 4,096 solver output tokens
    across turns, 16,384 context, and 2,048 editor output tokens. A training
    block uses three 16-example H updates or 32 64-example W updates.
    MA retains the declared critic warm-up / disjoint single-use partner
    recycling scheme; extra optimizer work and physical rollout overhead
    must be reported. The operational plateau rule is a 0.5-point minimum
    development gain with patience three after at least three blocks.
    A safety block/time cap is a budget-exhausted stop, not convergence.
    These are controlled local experiments, not an exact WHALE replication.

    Domain-independent inference/reward/training dispatch, matched context
    limits, executable-editor prompting and phase-qualified recycling IDs
    passed 43 focused CPU regressions including existing math/native-SAO
    behavior. Data/retrieval/domain implementation and campaign orchestration
    are being prepared concurrently. No new GPU result is claimed at this
    preparation milestone.

    SearchQA now has a pinned full wiki18 corpus with **21,015,324 passages**
    and a real CPU Tantivy BM25 index. Training/reward/dev/test sizes are
    **18,946 / 128 / 256 / 700**; all seven official evaluation sources are
    represented by 100 reporting questions each. The reconstruction, public
    source revisions, SHA256 IDs, and dense-retrieval/LLM-judge deviations
    are recorded in the asset manifests. The initial harness forwards the
    query unchanged, returns top-1 evidence capped at 800 characters (an
    explicit proxy, not an exact 200-token limit), allows one search call
    and two assistant turns, and has no implicit no-tool retry.

    A real 64-request burst exposed the initial retrieval server's listen
    backlog of five: 32 requests failed, with 124 kernel listen drops.
    Before any GPU baseline, the owned server was replaced by a bounded
    16-worker / 1024-pending implementation with backlog 2048. No retries,
    deadline increases, corpus changes or ranking changes were used to mask
    the issue. Node4 passed 64/256/512 concurrent requests in 1.42/5.29/10.69
    seconds; node3's private forward passed 64/256 in 1.44/5.31 seconds.
    Exact ranking/content/score checks and transport evidence are retained.

    The first SearchQA development baselines completed on both nodes by
    **2026-09-29 15:39 UTC**, with zero optimizer updates and no test
    evaluation. Each contains exactly 256 questions x8 matched samples:

    | Initial actor | Correct / 2048 | Development mean@8 |
    | --- | ---: | ---: |
    | Original Qwen3-4B-Instruct-2507, node4 | 347 | 16.9434% |
    | Actual SFT1152 checkpoint, node3 | 362 | 17.6758% |

    The SFT-minus-original paired question interval is **[-1.2695,+2.7832] pp**
    around a **+0.7324 pp** point estimate. The predeclared higher-development
    rule selects SFT1152; this is not a statistically established SFT benefit.
    Both baseline units exited zero and restored the authorized fillers.
    They used four frozen-solver replicas per node; this layout is retained
    in the execution metadata rather than described as eight active inference
    GPUs. The baseline-completion watcher exited zero. Chess baseline and all
    optimizer conditions are still pending at this milestone.

    The subsequent Chess development pair also completed, with all eight
    solver replicas per node verified at context 16,384. Original 4B scored
    **34/2048 = 1.6602%**, versus SFT1152 **31/2048 = 1.5137%** on the same
    256 puzzles x8. These are complete-reference puzzle successes, not
    first-move accuracy. The SFT-minus-original paired interval is
    **[-0.7324,+0.4395] pp**, around **-0.1465 pp**. The predefined rule
    selects the original model for Chess, without claiming a significant
    difference. Both reports passed model/data/source/sample/seed audits,
    have zero harness runtime errors and zero optimizer updates; both
    baseline units exited zero and restored the authorized fillers.

    Controller validation now passes **44 tests**, including full-node
    standalone inference accounting and rejection of candidates whose
    full development report contains hook errors. Persistent full-campaign
    workers have been authorized: node4 runs SAO alternating plus SAO
    weight-only; node3 runs MA-SAO alternating plus SAO harness-only.
    Chess precedes SearchQA. Launch is not a completed optimization result;
    native updates, checkpoint handoffs, stopping and final reports remain
    under supervision.

    The first native Chess H1 completed on both nodes (**3 actor /6 critic
    updates** each; 499.96s SAO and 517.61s MA-SAO). A tensor audit confirmed
    real actor changes: 1,081,448 of 10,485,760 entries changed in the first
    query-projection matrix. However, supervision found **128/128 physical
    editor actions invalid**: 126 forbidden imports, typically `import re`
    inside `parse_action`, and two unsupported calls. This is an unusable
    editor-interface pilot, not evidence about algorithm effectiveness.

    Both owned workers were intentionally stopped after their durable H1
    optimizer receipts and before W1. All checkpoints, metrics, 8,192
    scoring rollouts and cost ledgers were retained; authorized fillers were
    restored. Exit1/SIGTERM statuses here record the intentional stop, not a
    hidden crash. The versioned `_v2` protocol will narrowly permit approved
    regex/JSON operations inside the existing isolated sandbox, validate
    real sampled programs, and restart every arm from a fresh common
    initializer. Dataset splits, h0, verifiers, budgets, seed and stopping
    rules stay fixed. Old pilot-trained weights will not be mixed into the
    amended comparison.

    The amended sandbox accepts **120/128** saved pilot programs (61/64
    node4, 59/64 node3); eight programs importing `random` remain correctly
    rejected. It permits only hook-local `re`/`json` imports through restricted
    facades, plus `isinstance`. Module-level imports, arbitrary modules,
    reflection/I/O and existing isolation/resource restrictions remain
    forbidden. **600 real isolated synthetic hook calls** passed, and the
    runtime suite passed **32 tests**. Both domain editor prompts now include
    the same exported API guide. This validates interface usability, not
    task accuracy or an algorithm gain.

    V2 integration passed **149 tests (one skipped)** on node4 and **81
    tests (one skipped), 785 subtests** on node3. Both initial harness source
    files remain byte-identical. The implementation and amended protocol
    are preserved in the verified 45-file `SOURCE_SNAPSHOT.zip` in v2/node4.
    All four v2 development baselines then completed with eight replicas
    per node and zero optimizer updates:

    | Domain | Original correct /2048 | SFT1152 correct /2048 | Common initializer |
    | --- | ---: | ---: | --- |
    | Chess | 34 (1.6602%) | 31 (1.5137%) | Original |
    | SearchQA | 346 (16.8945%) | 362 (17.6758%) | SFT1152 |

    Chess's interval is unchanged. SearchQA's SFT-minus-original difference
    is **+0.7813 pp**, paired interval **[-1.1719,+2.7344] pp**. These remain
    initializer choices, not significant SFT-effect claims. SearchQA's
    original count differs by one from the earlier four-replica run; no old
    score was substituted. Both complete-baseline units and their watcher
    exited zero. Full v2 workers have started from the unchanged initial
    checkpoints; usable H1 rewards and subsequent W1 remain to be verified.

    V2 Chess H1 subsequently completed on both nodes: **3 actor /6 critic
    updates** each, 563.68s SAO and 561.92s MA-SAO. Of 64 physical editor
    actions (including prefetch), **39 SAO /30 MA-SAO** were valid, with
    **5 /2 positive paired rewards** respectively. Invalid generated
    programs continue to be rejected rather than repaired or rewarded.
    None of the four post-policy proposals passed the development
    acceptance rule in either arm: both retained h0. The accepted
    **report**, not a new harness, scored **33/2048 = 1.6113%** in both arms,
    with zero runtime errors. The one-sample difference from the initial
    eight-replica result is not a causal H improvement or decline.

    W1 is now exercising real solver optimization. SAO has live actor and
    critic gradient updates; MA-SAO has verified critic-only warmup with
    no actor updates during that stage and disjoint **1,024 warmup /2,048
    actor-mixed** recycled episodes. Both continue automatically; no W1
    accuracy or full H-W-H improvement is claimed at this milestone.

    The first full Chess block later completed:

    | Development stage | SAO | MA-SAO |
    | --- | ---: | ---: |
    | Shared initial h0/W0 | 1.6602% | 1.6602% |
    | H1, h0 retained | 1.6113% | 1.6113% |
    | W1, h0 unchanged | 20.5078% (420/2048) | 10.5469% (216/2048) |
    | H2, h0 retained again | 20.6543% (423/2048) | 10.5957% (217/2048) |

    MA-minus-SAO after H2 is **-10.0586 pp**, paired question interval
    **[-14.0137,-6.3477] pp**. Both gates are `not_extended`, not convergence.
    The gains in this block come from weight updates, not accepted harness
    changes. W1 took **1641.54s SAO /2873.95s MA-SAO** with 32 actor updates
    each and **64 /80 critic updates**. MA's 6,528 native actor sequences
    comprise 3,723 fresh and 2,805 replayed turns; 1,405 other replayed turns
    were used for critic-only warmup. Unconsumed warmup prefetch was retained
    in the receipts and did not add optimizer updates. Both W1 actors also
    passed a numerical first-layer query-projection weight-change check.

    A substantial reproducibility caveat emerged in the independent
    weight-only baseline: its first W block scored **47/2048 = 2.2949%**,
    despite matching the alternating SAO W1's initial solver/critic source
    checkpoints, h0, training questions and order, hyperparameters, and
    global seed 42. Audited differences were run paths, episode namespace
    IDs and TMPDIR; no input-configuration mismatch was identified.
    Native sampled trajectories differ (first-step reward means 1/65
    versus zero). The independent run's format fraction was only 4.5898%,
    versus 97.3633% for alternating SAO W1 and 51.5625% for MA W1.
    Many malformed outputs use SAN rather than required UCI. This is not
    evidence that replay alone caused the format regression. Question-level
    confidence intervals do not capture the observed training variability;
    these single-run comparisons cannot establish a reproducible algorithm
    effect. The prescribed convergence runs continue without mid-run tuning.

    SearchQA's common SFT initializer and training-code identities were
    independently pre-frozen on both nodes using the existing CPU-only
    helpers. Main worker PIDs, source, protocol and active Chess GPU work
    were unchanged. This allows either node to enter SearchQA after its own
    Chess work finishes, without an avoidable initialization barrier behind
    the other node's longer single-component baseline.

    Chess harness-only completed the declared **three-block plateau** with
    **9 editor actor /18 critic updates, zero W updates** and unchanged
    original solver bytes. Its best development snapshot is still the
    initial **1.6602%**. Node3's Chess reporting completed before it entered
    SearchQA; no test outcome was used for subsequent decisions.
    Node4's independent Chess weight-only baseline continues improving:
    **2.2949%, 4.2480%, 15.7715%** after W1/W2/W3, so it has not plateaued.

    SearchQA MA-SAO H1 completed from SFT1152: **20/64 valid actions, one
    positive training reward**. All four final proposals were invalid
    (two unsupported lambdas, two out-of-range query limits), so h0 was
    retained; its full development report was **358/2048 = 17.4805%**.
    No mid-run prompt, validator or protocol changes were made. The 4,096
    scoring episodes contained **8,053 recorded model turns**, with no
    empty/misaligned token/log-probability records. Selected W1 replay
    retained byte-equivalent original calls in disjoint 1,024/2,048 pools.
    Eight critic-only warmup steps completed before the first actor update.
    A fresh native W1 trajectory, independently of frozen-solver/recycled
    evidence, was decoded and verified to contain a real search tool call
    and retrieval-response continuation with aligned 24+9 response tokens.

    SearchQA MA-SAO W1 completed **32 actor /80 critic updates**, including
    eight critic-only warmup steps. Its 7,988 native actor sequences comprise
    **3,952 fresh /4,036 mixed replay turns**; token/tool receipts remain
    nonempty and aligned. Development accuracy reached **434/2048 =
    21.1914%**, versus the SFT start's 17.6758%: **+3.5156 pp**, question
    interval **[0.0000,+7.0313] pp**. The 135 tool errors were all explicit
    attempts to exceed h0's one-search limit, not retrieval infrastructure
    failures. No harness runtime errors occurred. H2 is running with W1;
    the competing SearchQA SAO arm has not started yet, so no cross-algorithm
    SearchQA conclusion is available.

    Node3 then published SearchQA MA-SAO's first H-W-H block:
    **431/2048 = 21.0449%** after H2, with h0 retained and zero harness
    errors. H2 had 23/64 valid editor receipts, two positive rewards and
    48 fresh /16 refreshed replay actions; all four final proposals were
    invalid. W1 checkpoint bytes were unchanged by H2. The block contains
    **38 actor /92 critic updates**, including eight warmup steps, and
    **8,192 scoring rollouts** with no unused reward pairs. No extension
    decision is available until SAO's matching block exists. SearchQA
    harness-only has started opportunistically from fresh SFT1152/h0 while
    node4 continues Chess weight-only (W4 best **17.5293%**, W5 active).
    In addition to heartbeat/deadline monitoring, a read-only watchdog now
    alerts if native optimizer metrics cease advancing for 30 minutes.

    An operational interruption was caught at **22:00 UTC** during
    SearchQA harness-only H5's post-training proposal-server startup.
    Contemporaneous logs show the authorized utilization filler was
    restarted at 22:00:08 and allocated 79,872 MiB on GPU0 before vLLM
    checked memory. Only 60.71 GiB was free versus 118.83 GiB required.
    H5's three actor/six critic updates and checkpoints had already been
    saved; there were no proposals yet and no leftover experiment GPU
    processes. This was a resource reentry, not a training NaN or a corrupt
    checkpoint.

    After a fresh exclusive audit stopped only the authorized filler, node3
    resumed under the identical protocol/environment with memory fraction
    0.85 and context 16,384. The new unit's cutoff remains before the
    original **2026-10-01 16:46:30 UTC** safety deadline. The proposal server
    returned HTTP 200, H5 completed, and H6 started. **341 files**, including
    checkpoints, native metrics, pair audit and 192 receipts, remained
    byte-identical: no optimizer update was repeated. All 45 frozen source
    files were unchanged; failed-attempt logs/costs were retained. All
    campaign, native-progress and milestone watchers were restored.

    The current best SearchQA harness-only configuration changes only
    literal settings: **steps 2 -> 3, search calls 1 -> 2, top-k 1 -> 3,
    document characters 800 -> 1000**. It has no custom executable hooks;
    SFT1152 solver bytes remain fixed. H5 retained this H3-origin harness
    and reported 412/2048 = 20.1172%; later tiny score differences with
    unchanged code are not additional harness improvements.

    Independent Chess weight-only showed a genuine output-format collapse:
    W5 and W6 both scored zero, with repeated move tags in all 2,048
    evaluation episodes. Metrics remained finite and native optimizer
    updates succeeded; this is not the node3 resource failure. W4's best
    checkpoint (**17.5293%**) is preserved. W7 is the third patience check;
    final reporting will use the dev-selected best checkpoint, not silently
    substitute the degraded latest weights or alter the fixed protocol.

    SearchQA harness-only subsequently reached its declared **six-block,
    stale-three plateau**, not the safety cap. It performed **18 editor
    actor /36 critic updates, zero W updates**, retaining the H3-origin
    configuration above and fixed SFT1152 solver bytes. H6's four proposals
    were invalid; the frozen best development report remains **20.1172%**.
    Node3 is now alive at the peer barrier with no experiment GPU children
    and the authorized filler restored. Its SearchQA test reports remain
    pending the paired alternating-arm gate; no test outcome selected this
    harness or stopping point.

    SearchQA alternating blocks one and two passed the paired development
    gate. Block two reached **518/2048 = 25.2930%** for SAO and
    **581/2048 = 28.3691%** for MA-SAO. Neither arm accepted a harness
    change: both still used the byte-identical h0. MA-SAO's point estimate
    exceeded SAO by **3.0762 pp**, but its paired question-bootstrap interval
    **[-0.1465,+6.2500] pp** crossed zero. This is a single-seed exploratory
    development result, not evidence of statistically established or
    equal-compute superiority.

    During block three, node3's first MA-SAO H4 attempt completed three
    editor actor/six critic updates but failed the strict post-training
    receipt audit. The asynchronous evaluator produced request indices
    0--65 with two gaps; 64 consumed receipts contained two duplicate task
    identities and omitted two frozen tasks. All thirteen preceding
    SearchQA H phases, including node4's same-round H4, had exact unique
    indices 0--63. The invalid H4 checkpoint, 64 receipts, reward pairs and
    logs were therefore excluded from algorithm state rather than silently
    accepted.

    The complete invalid attempt was moved to
    `node3/RECOVERY_R2/INVALID_MA_H4_ATTEMPT1` with **282 files** hashed.
    Its physical three actor/six critic updates remain counted. W3 and the
    frozen H4 input/spec/data were byte-verified unchanged. Recovery reused
    the original protocol and safety deadline and reran only H4. The valid
    attempt then produced exactly **64 unique receipts**, request indices
    0--63, an identity-exact task multiset, and passed the physical pair and
    phase-evidence audits. `RECOVERY_R2/VERIFIED.json` records that W3 was
    not relaunched.

    Node4 completed its own H4 before observing node3's failure, then
    stopped at the peer barrier. Its H4 optimizer, 64 unique receipts, pair
    audit, candidate selection, final development report, phase evidence and
    published block were all verified before scoped SearchQA recovery; no
    node4 H4 update was repeated. The resumed worker preserved the original
    deadline and continued the independent weight-only arm while waiting.

    SearchQA block three subsequently published **493/2048 = 24.0723%**
    for SAO and **516/2048 = 25.1953%** for MA-SAO. Both declined from block
    two; MA-SAO remained **+1.1230 pp** above SAO, with paired interval
    **[-2.4414,+4.5898] pp**. Both still used h0. The preregistered paired
    rule therefore continued block four because MA-SAO remained positive
    relative to the common start and above SAO, not because block three
    improved over block two. SearchQA weight-only W1--W3 scored 22.4121%,
    28.4180% and 28.1250%; W2 remains best and W3 is the first stale block.

    Node4's first block-four SAO H5 attempt encountered the same strict
    asynchronous receipt-identity failure. Request indices 29--31 were
    absent and 64--66 were generated instead, leaving three repeated and
    three omitted frozen tasks. The completed three actor/six critic updates
    were archived with 280 hashed files in
    `node4/RECOVERY_H5_R1/INVALID_SAO_H5_ATTEMPT1`; they remain physical
    cost but are excluded from algorithm state. W4 and the exact frozen H5
    input/spec/data were unchanged. Scoped recovery retained the original
    safety deadline and reran only H5. The valid attempt and node3's
    independent MA-SAO H5 each passed with 64 unique task identities and
    exact request indices 0--63. `RECOVERY_H5_R1/VERIFIED.json` confirms
    that W4 was not relaunched.

    SearchQA block four then reached new single-run development highs:
    **549/2048 = 26.8066%** for SAO and **600/2048 = 29.2969%** for
    MA-SAO. Both still used byte-identical h0. MA-SAO exceeded SAO by
    **2.4902 pp**, with paired question-bootstrap interval
    **[-0.3906,+5.3223] pp**; the interval still crossed zero. The fixed
    gate continued block five because MA-SAO was positive relative to the
    common start and above SAO. This remains a single-seed, unequal-compute
    exploratory comparison.

    During this recovery node4's root filesystem reached 100% usage,
    primarily from other users' home directories. No other-user data was
    removed. Only current-user rebuildable package-cache entries and this
    session's known temporary tool outputs were deleted. Experiment
    checkpoints, receipts and logs remained on 3FS; disk pressure affected
    detached-monitor exit/output files, not native optimizer evidence.

67. **Harness-R1 Engineer experiment approved (2026-09-30).** The user
    approved freezing SearchQA's best completed MA-SAO W4 solver
    (historical dev mean@8 29.296875%) while independently initializing an
    Engineer from pretrained Qwen3.5-9B. Existing solver weights and SAO /
    MA-SAO algorithms remain preserved. Neither interrupted weight-only W4
    nor interrupted MA-SAO H6 is a valid initializer.

    The primary contrast is online-GRPO Engineer versus cold-start-SFT
    Engineer under matched proposal / target-rerun budgets, with the fresh
    pretrained Engineer as a third control. A positive reward on training
    failure packets is not a development or test improvement. The solver's
    temperature 1.0, top-p 1.0, top-k -1, no-thinking setting, 4096 total
    assistant tokens, 16384 context, h0 tool budget, full-wiki BM25 index,
    and answer-tag alias-EM verifier stay fixed.

    Four regression scenarios confirmed the new no-hook episode reproduces
    the old h0 messages, tool definitions, token caps, seeds, rewards and
    termination. Paired question-level bootstrap selection rejects unchanged
    harnesses and task/seed mismatches. Accepted patches become the next
    incumbent rather than resetting to h0; cumulative evolution is an
    explicit extension of the paper's fixed-baseline training stages.

    Both old campaign main workers have exited. Node3's exact old systemd
    unit has MainPID=0 and an empty cgroup. Only audited authorized filler
    processes were stopped. Eight frozen solver replicas run on node3
    localhost ports 18901--18971 (step 10); real structured tool calls were
    verified. Node4 reaches them through loopback-only SSH tunnels using the
    existing configured `node3-internal` identity. No unauthenticated
    network-facing target endpoint was opened. Train-only collection of
    3072 questions is managed by
    `searchqa-harness-r1-collect-node3-20260930.service`, with per-episode
    receipts and persistent progress in the new run directory.

    Initial fresh-9B serving reached health readiness but failed its first
    real request because FlashInfer GDN prefill required unavailable nvcc.
    That attempt is infrastructure failure, not an invalid patch or a zero
    reward. Only the exact Engineer unit was stopped and relaunched as r2
    with vLLM's supported Triton GDN prefill backend. No RL gain is claimed;
    serving, isolated hooks, cold-start data and distributed training are
    being validated before formal optimizer updates.

    First real data-plane results: all 3072 train episodes completed in
    98.99 seconds (31.03 episodes/second). Independently rerunning the
    frozen W4+h0 on 256 dev questions with eight samples gave
    **593/2048 = 28.955078125%** in 69.69 seconds. This newly measured
    baseline, not the old 600/2048 score, is the causal reference for the
    new experiment. No candidate harness gain has been measured yet.
    The first train collection predates final runtime provenance hardening;
    it will not be silently relabeled with a newer implementation hash.

    A second Engineer startup was rejected by the exclusive resource audit
    because an independently launched validation pytest process held a
    small GPU context. CPU validation is now explicitly requested to hide
    CUDA devices. Once that exact validation process exited, Engineer
    startup continued as r3. Existing audit protections were not weakened.

    A read-only preservation inventory found 98 completed solver actor /
    critic checkpoint directories (826.5 GB). Expensive full hashing was
    stopped rather than competing with model loads; no weights or rollout
    artifacts were deleted. Cleanup remains gated on a verified manifest.

    The first fresh-base Engineer pilot was stopped after 157 completions:
    only **4/157 (2.55%)** passed the strict patch parser and hook AST/schema
    checks. Most failures omitted the `<patch>` envelope; others used
    incorrect hook function names, nested definitions or unsupported action
    keys. Its only positive training delta (+0.125) was **not a clean
    example**: isolated execution raised `AttributeError` because the patch
    called `.get()` on the string `ctx.last_observation`. It is excluded
    from SFT and is not evidence of an effective harness.

    The static contract now includes a parser-valid complete output example,
    exact `def hook(ctx, nb)` and action keys, string observation types, and
    the unchanged one-search budget. Reward receipts expose actual hook
    calls, errors and status counts without changing the primary delta or
    treating infrastructure errors as zero. Silver requires positive delta
    **and zero hook execution errors**; development acceptance also rejects
    candidates with execution errors.

    After freezing this revised runtime, a fresh r3 collection completed
    all **3072 training episodes**, mean **29.166666667%**, and produced
    **256 token-bounded failure packets**. Runtime SHA256:
    `2eb35e7042f5eb812d88317de8d16211d3b5a15beeacdee08c5157a5fd8a772d`;
    packet file SHA256:
    `d76a5ea0720b06ee31d0e3e6a9a93976dd555fca564607d35a06ff1e801f5a01`.
    Old r2 receipts were not relabeled. SFT-bound prompts reserve 12288
    completion tokens plus EOS within context 32768, with a 20479-token
    prompt limit and no implicit truncation.

    Engineer r4 was again stopped by the resource audit after a separate
    GRPO pytest process opened a GPU context. The offending process exited;
    the mocked test was subsequently changed not to expose physical CUDA
    devices. CPU validation must hide CUDA throughout. R5 was started only
    after GPUs were released; all eight replicas passed health checks and
    a real generation request returned successfully. Audit protections
    remain unchanged.

    Corrected-contract silver collection runs persistently on node4, using
    node3's frozen target replicas for reruns. A fail-closed pipeline waits
    for at least 256 clean positive examples, then releases only the exact
    Engineer serving unit, runs full eight-rank SFT, verifies its atomic HF
    export, and runs an initial four-actor/four-rollout-GPU GRPO stage. Its
    postcondition checks committed optimizer counts, finite per-rank metrics,
    all rank-state hashes, actor-file identity, and actual rollout-engine
    reload. This initial stage is **not** the complete GRPO experiment or
    matched development comparison. No SFT/GRPO weight update or learned
    harness gain is claimed before those real executions are verified.

68. **Fast native-4B Harness-R1 pilot and infra qualification (2026-09-30).**
    The user requested small-scale, rapid, reliable feedback and official
    Harness-R1 code reuse before scaling. The original 4B is
    Qwen3-4B-Instruct-2507, not Qwen3.5. Thirty-two unique fresh-9B teacher
    programs with positive train delta and zero hook errors were frozen;
    their receipts contain 568 successful hook calls. The 32-label minimum
    is an explicit preliminary adaptation, not the standard 256-label gate.

    Eight solver replicas at concurrency64 produced 2022.98 tokens/s, versus
    2027.74 tokens/s with four replicas at concurrency32. Mean active-card
    utilization increased from 33.68% to 65.68%, median 0% to 93%.
    Concurrency256 was slower (1533.45 tokens/s). Node3 was rebalanced to
    four frozen targets plus four original-4B Engineers under one supervisor.
    Native 4B SFT profiles on node4 completed actual four-step full-parameter
    updates and atomic HF exports: loss chunks128/1024 gave 40106.94/40931.26
    input tokens/s and 261.46/255.99 seconds end-to-end. Compute alone took
    4.894/4.796 seconds; initialization and export dominate these small runs.
    Larger chunks did not increase measured overall GPU utilization.

    Full dev no-op timing measured verified baseline reuse: 140.04 seconds
    and4096 new episodes before, 74.32 seconds and2048 after, **1.884x**.
    The once-only baseline cost71.68 seconds. Both no-op arms were rejected.
    Empty candidate patches remain invalid: the first timing attempt was
    rejected and replaced with explicit dev-only no-op-control mode.
    Cached baseline hashes, exact task/seed pairing, runtime/model/retrieval
    identity, concurrency and endpoints must all match. Stochastic serving
    still varies across batches; reuse is not a remedy for uncertainty.

    Advantage normalization, PPO clipping and optional KL now directly call
    unchanged hash-pinned official Harness-R1/Relax primitives. FSDP/vLLM
    replaces Megatron/SGLang explicitly; no exact backend-reproduction claim.
    The first two four-actor/four-rollout-GPU GRPO attempts stalled during
    vLLM TP initialization, including the official NCCL-environment retry.
    Both were stopped by exact experiment unit, preserving logs and weights.
    The bounded r3 recovery started at 11:24:50 UTC with four actor GPUs and
    TP1 rollout on GPU4; GPUs5-7 temporarily remained unused. The separate
    configuration preserves all proposal/learning budgets and changes only
    rollout TP and timeout (900 seconds). Recovery is not optimizer success.
    Its matched raw/SFT/RL pipeline waits for real committed
    optimizer evidence and the infra receipt, then evaluates equal proposals,
    held-out train screening, full dev confirmation and direct RL-vs-SFT
    paired gains. No RL efficacy is claimed from initialization or launch.

    Persistent results and method/source indexes are maintained in
    `runs/searchqa_harness_r1_20260930/fast4b/README.md` and
    `whale_local/harness_r1_searchqa/README.md`. Existing solver weights,
    training algorithms and analysis remain preserved; no old artifacts
    were deleted.

    A stack inspection of the owned r3 engine identified the actual blocker:
    independent rollout inherited TORCHELASTIC_USE_AGENT_STORE=True and tried
    to connect to a nonexistent agent TCPStore. The rollout subprocess now
    strips torchrun rank/master and TORCHELASTIC_* variables while preserving
    actor configuration. The CPU regression and remote-control identity tests
    pass. Per the user's all-GPU requirement, r4 restores four actor plus four
    rollout GPUs on node4. Node3 retains four targets plus four raw Engineers;
    the pipeline concurrently collects and reuses its 32 raw proposals.
    Real TP4 initialization/warmup completed; all four rollout GPUs reported
    100% utilization at 11:40:23 UTC, not an average or optimizer result.
    Raw collection took8.212 seconds with zero clean positive examples.
    Algorithm efficacy still requires the committed update and matched controls.

    Final r4 receipts are complete. GRPO finished11:52:31 UTC after one online
    update/four optimizer steps and verified disk reload. All32 rollout
    proposals were invalid, all groups constant zero, every policy gradient
    zero. Matched controls finished12:00:12 UTC: raw/SFT/RL each32 proposals,
    each0 valid/clean, no train-qualified candidate. Candidate dev/test and
    RL-vs-SFT confidence intervals were therefore not run or claimed.
    This tiny cold-start protocol gate failed; no algorithm-scale conclusion
    follows. Additional zero-signal RL and SAO/MA-SAO scaling are deferred
    until executable proposal generation is reliable. Final focused tests:28 passed.

69. **Released Harness-R1 ALFWorld diagnostics and project handoff (2026-09-30).**
    Downloaded and verified `harness-r1`, `cold-start-sft`, and
    `target-agent-sft` at model revision
    `f6a915953f8206f3f5c7ef0b40240e1f4d2b6676`.
    These are two Engineer stages and one Solver, not one model per benchmark.
    Official runtime source remains unchanged at
    `94f2e087f573e1b82fc9568bb634d4ae9a887e28` and is registered as a
    submodule for reproducible source publication.

    All scientific arms use exact task IDs 0/1/2/3/8/17/27/29 from the official
    ALFWorld split, fixed original Qwen3.5-9B target for editing comparisons,
    temperature zero, 50 steps and 4096 target tokens. Original default harness
    scores4/8. The released RL Engineer emits a valid four-hook patch and
    scores4/8; generated guidance is observed and seven action sequences change.
    The parent-controlled cold-start proposal A scores4/8, rescuing task8 and
    regressing task27. Its corrected rerun reuses the exact original patch.
    A separately generated concurrent cold-start proposal B scores5/8,
    rescuing task8 without a regression. Both proposals must be reported;
    selecting only5/8 would overstate reliability and conceal an extra proposal.
    The standalone Agent-SFT Solver scores7/8 (+3 tasks), a Solver gain only.
    No full500 evaluation, independent held-out patch gain or new RL training
    is claimed.

    Initial direct SFT attempts exited zero while every task failed before
    model generation: FastDownward tried to copy its library into full-disk
    /tmp. They are excluded infrastructure receipts, not0/8 model scores.
    Dedicated TMPDIR scratch corrected both jobs. A new report audit rejects
    nonzero task-error, unknown or still-running statuses before model gains
    are reported. Earlier health-only model readiness also missed FlashInfer
    GDN's missing-nvcc generation failure; explicit Triton and a real structured
    tool-call probe fixed it without modifying weights or tokenizer templates.

    The canonical committed handoff is [PROJECT_STATE.md](PROJECT_STATE.md):
    objectives, historical Solver-versus-harness gains, exact algorithms,
    all current ALFWorld proposals, limitations, failure recovery and next gates.
    Runtime evidence remains in excluded assets/runs directories. Publication
    regression covers448 passing tests,5 skips and843 subtests; no new GPU
    optimizer run is attributed to that CPU validation.

70. **ALFWorld SAO/MA-SAO optimizer qualification and formal launch
    (2026-09-30).** The 0--239 2×2 starting matrix completed all96 jobs on
    first infrastructure attempt. The shared RL-Engineer/Agent-SFT start moved
    166/240 to184/240 (+18/+7.50pp), with23 valid and1 invalid patch,
    27 rescues and9 regressions. This remains same-packet transductive train
    evidence, not held-out gain.

    The first native H optimizer attempts found a real model-interface bug:
    AReaL's Qwen3.5 VLM path ignored critic mode and returned
    `[batch,seq,248320]` vocabulary logits. A dedicated token-value model now
    returns exactly `[batch,seq]`, persists its scalar head on save/reload,
    and has a fail-fast value/mask shape guard. The next 2-way FSDP run reached
    real critic backward and OOMed on an approximately18k-token trajectory
    with82 MiB free. It produced no metric, checkpoint or optimizer update.
    The archived process group required one exact-unit stop after distributed
    ranks failed to self-terminate.

    The corrected 3-way actor FSDP plus1 rollout layout completed one actor
    and two critic updates. It trained3 sequences/47,687 tokens; actor loss
    was0.00399, policy-gradient loss0.00423, actor grad norm9.366 and critic
    grad norm226.206. Both checkpoints exist and the actor safetensors SHA256
    changed from released `7d16d5...` to `cb06bf...`. End-to-end elapsed time
    was765.73 seconds; peak coordinator evidence was116.67/139.80 GiB,
    leaving about23 GiB. Its512-token outputs were intentionally only a shape
    smoke and all had zero task reward; they are not efficacy evidence.

    The real W smoke completed one actor and two critic updates over139 action
    trajectories,1,386,174 total tokens and82,518 valid action tokens.
    Mean task reward was0.6403, actor/critic grad norms were1.063/348.279,
    and the actor checkpoint identity changed. Actor update took10m32s and
    two critic passes21m47s; the full phase took2,713.00 seconds. Resident
    actor-GPU utilization averaged77.74% with100% median, while rollout GPUs
    averaged0.55% because a one-step run cannot overlap the long optimizer
    pass. Filler is not included.

    The W rollout log contained3,870 HTTP406 workflow failures. Consecutive
    IDs0/1/2/3 all targeted one single-session packet assignment, so only
    repeated tasks0/2 supplied useful episodes. Solver data now interleaves
    `0,10,20,30,...` across24 assignments, and a pre-GPU audit rejects repeats
    anywhere in the full asynchronous active window. The old smoke validates
    model/tool/value/update/checkpoint plumbing only and is not a formal
    initializer or throughput estimate for the corrected schedule.

    Protocol v6 was frozen before any formal optimizer update. Both roles use
    staged `2 steps × batch4` rounds, minimum two role rounds and the existing
    dev patience/gain/one-sided-bootstrap gates. H uses3+1+4 GPUs, a20k-token
    microbatch bound and4096 training output tokens; all23 valid released
    proposals were≤3014 tokens, while the sole12288-token response was an
    invalid runaway. Evaluation remains at12288. Missing Qwen processor files
    are restored after export with a receipt, and weight-change checks hash
    safetensors separately from metadata.

    Node3's 100-task E0/W0 dev calibration completed model work, but postprocessing
    rejected an empty all-pass failure-packet file and one patched-evaluation
    infrastructure failure at packet370. Empty all-pass files are now handled
    explicitly, infrastructure summaries trigger bounded retry, and only
    packet370 was rerun before MA H1; no optimizer update or model score is
    attributed to that failed evaluation. The recovered common dev reference
    is79/100 no-patch and76/100 packet-transductive (5 rescues,8 regressions,
    6 valid and4 invalid packet patches).

    The first formal H1 rollout attempt then exposed a prompt/validator mismatch.
    The multi-hook prompt omitted the existing per-code maximums of5 helper
    functions and1200 AST nodes. Node4's first4 candidates were all rejected
    for those two undeclared limits; node3 had not produced a candidate.
    Both exact units were stopped at zero metric rows, zero checkpoint files
    and zero optimizer updates. The task rows, reward, seed, checkpoints and
    optimizer are unchanged; regenerated fresh rows only append the literal
    existing validator limits. Both arms restarted from untouched E0/W0 as
    H1-r2. The four physical invalid rollouts remain counted and cannot become
    formal checkpoints. Sealed test IDs400--499 remain unread.

71. **H1-r2 zero-update diagnosis, protocol v7 and H1-r3 restart
    (2026-09-30).** The validator reminder worked: node3 produced four
    executable patches whose ten-task packet deltas were `+0.2`, `+0.1`, `0`
    and `-0.1`. Training then stopped before critic forward with
    `ValueError: The length of nums must be divisible by K`. AReaL's
    three-way actor dispatch requires consumer batch divisibility, while the
    v6 preflight checked only `batch>=DP`; therefore batch4 was never a valid
    formal 3-way configuration. Node3 retained four physical evaluations but
    has zero metric rows, checkpoints and optimizer updates.

    Node4 exposed a separate cross-node ownership failure. The shared
    AgentBench runtime had been initialized by node3 UID1009 with mode0775;
    node4 uses UID1008 and could not create official generated task configs.
    All affected evaluations returned `PermissionError` through the explicit
    HTTP502 infrastructure path rather than model reward. The exact node4 unit
    was stopped while metrics and checkpoints were still absent. Its two
    persisted receipts are syntax-invalid model patches scored zero normally;
    no infrastructure fallback was accepted.

    The root fix creates a writable AgentBench layout and scratch directory
    inside every H reward or W environment run. Pinned upstream source and
    fixed ALFWorld data are symlinked read-only; generated configs never cross
    UID ownership. `large_train.configure` now rejects any batch not divisible
    by actor DP before model allocation. Protocol v7 changes H only to
    `2 steps × batch6`; W remains `2 × 4`. All four partition SHA256 values,
    E0/W0, seed, task rows, reward and optimizer settings are unchanged.
    Manifest SHA256 is
    `fa6c2c98315fcfba2a65edd48dfb17ff0c7f4380fea1f35cfa35e57188680eaa`.

    Focused tests passed under both node identities (node3 uses
    `TMPDIR=/dev/shm` because `/tmp` is not reliable). Local discovery passed
    307 repository tests plus10 ALFWorld tests with5 skips; compile and all
    pinned upstream reverse checks passed. While repairing, the verified
    filler held all16 GPUs at roughly84--92% utilization and was excluded from
    algorithm metrics. It was identity-checked and stopped by each launcher.
    Sequential and MA H1-r3 then started simultaneously from untouched E0/W0.
    Sealed test400--499 remains unread.

72. **Exact trajectory-capacity audit and protocol v8 H1-r4
    (2026-09-30).** H1-r3 crossed both preceding failure points. Node4's
    run-local AgentBench tree was owned by UID1008 and completed official
    evaluations without `PermissionError` or HTTP502; node3 independently
    used UID1009. Both six-item batches completed reward collection and critic
    value forward, proving the AReaL three-way dispatch was now divisible.

    The actor then rejected `compute_logp` because each rank's sampled
    trajectories exceeded the20,000-token microbatch capacity. Exact physical
    ranges were20,456--21,117 tokens on node4 and20,309--21,139 on node3.
    This is a capacity/configuration failure, not OOM and not model reward.
    It occurred before actor or critic optimizer steps; both runs retain zero
    metric rows and zero checkpoint files. Node4/node3 performed9/6 physical
    reward evaluations because asynchronous work already in flight was allowed
    to finish before the wrappers released resources.

    A new pre-GPU audit applies the exact Qwen chat template, including the
    assistant `<think>` prefill, to every fresh Engineer row and reads exact
    token receipts for replay rows. Across all64 H1 rows, prompt length is
    9,830--18,606 tokens; with a4,096-token generation ceiling the declared
    worst case is22,702. Protocol v8 sets H microbatch capacity to23,040 and
    fails before stopping filler if any row no longer fits. This capacity can
    contain one worst-case trajectory but not two long trajectories.
    All split hashes, E0/W0, seed, batch6, reward and optimizer settings are
    unchanged. The v8 manifest SHA256 is
    `d952ef63f49f83dbb2c1bc730115cdcbfc870043e30253984e3541bed2d5a643`.

    Focused17 tests, compile checks and the capacity audit passed on both node
    identities. The verified filler covered the repair interval and was then
    released by the launchers. Both formal H1-r4 arms started from untouched
    E0/W0 with persisted preflight receipts showing
    `required=22702 < configured=23040`. Dev evaluation chains are queued only
    behind a valid H `COMPLETE.json`; failures restore filler instead of
    fabricating scores. Before those chains ran, the selector was corrected to
    enforce the protocol's declared `minimum_dev_gain=0.01` in addition to a
    positive one-sided95% LCB; a regression test prevents sub-1pp noise from
    being accepted. The batch6 change also made the earlier eight-row H2 file
    insufficient for two distinct batches. A generic pre-GPU row audit now
    requires `steps * batch_size` unique consumed IDs, and
    `h2_train_v3.json` contains64 rows at sample offsets64--127 (the first12
    are distinct); its SHA256 is
    `b6285ab0058aec4b3d756d3c60d5a7b211cecb06b995686c411d6670a132a7c2`.
    Sealed test400--499 remains unread.

73. **First complete formal H1 optimizer rounds (2026-09-30).** Protocol v8
    H1-r4 completed on node4 sequential SAO and node3 MA-SAO in1,197.7s and
    1,197.4s wall time. Each has exactly two metric rows, two successful actor
    updates, four critic updates, final actor/critic global-step1 checkpoints,
    runtime-metadata receipts, and distinct initial/final safetensors weight
    digests. Native trainer time was954.5s/964.7s. The maximum trajectory was
    21,918/21,958 tokens, below the23,040 capacity; there were no capacity,
    divisibility, permission, HTTP502, or OOM failures.

    Node4 step rewards averaged0.000 and-0.0333; node3 averaged0.0167 and
    0.1167. These are reward-split PPO observations, not acceptance claims.
    Actor updates reported finite gradients and `update_successful=1`; critic
    loss decreased from3.79 to1.24 on node4 and3.44 to1.78 on node3. The
    asynchronous rollout window fully hid the second reward batch behind the
    first update: measured rollout wait fell from162.2s/149.6s to roughly
    1ms. Final actor+critic serialization remained190--207s.

    Filler-excluded resident-window telemetry covered about1,200s per node.
    Actor-group mean utilization was41.4% on node4 and45.6% on node3;
    rollout was8.2%/6.8%, frozen Solver replicas8.9%/9.1%, and at least one
    GPU exceeded10% utilization in71.4%/70.9% of samples. All-GPU simultaneous
    utilization is intentionally0% because reward inference, actor/critic
    optimization, and synchronous checkpoint serialization are phased; all
    eight cards were nevertheless allocated to the declared3+1+4 layout.
    These figures exclude filler and do not represent algorithm quality.

    Both automatic dev chains then started on IDs300--399. Acceptance remains
    gated by at least+1pp paired gain and a positive one-sided95% LCB; H1
    training reward cannot select a checkpoint. Sealed test400--499 remains
    unread.

74. **Leakage-free H1 development gates and bounded recovery
    (2026-09-30).** Node4's first dev refresh attempt failed before model
    scoring because the legacy packet driver still wrote generated cohorts
    under a shared UID1009-owned asset directory. The driver now creates its
    AgentBench tree, generated config, cohort, task-ID file and scratch
    directory under each packet run. The bounded node4 retry completed all10
    packets; start350 returned exit0 with an infrastructure-failure summary on
    its first attempt and was correctly retried. Final no-patch and
    packet-transductive diagnostics were79/80 on node4 and78/81 on node3.
    Packet-transductive patches were excluded from formal selection.

    Formal pools were reconstructed only from H1 optimizer receipts actually
    consumed at behavior version0. Node4 had10 valid unique candidates and
    node3 had11; six asynchronously prefetched behavior-version1 receipts per
    arm were excluded. During the first global pass, official evaluation
    results exposed local task keys0--9 while baseline metadata used global
    IDs300--399. Both selectors were stopped before producing a decision.
    The selector now accepts either the exact declared global set or the exact
    local contiguous set and explicitly maps the latter by packet start.

    Node3 retry2 copied51 successful cache entries but also inherited seven
    `.lock` files from the interrupted run; upstream's `O_EXCL` cache lock
    would wait7200s for those files. Removing only the verified stale locks
    released the blocked evaluations. One remaining official evaluation then
    exhausted its internal retries with
    `eval_failed_treated_as_no_patch`; the run failed rather than turning that
    infrastructure error into model reward. Cache resume now excludes all
    `.lock` files and removes cached infrastructure-failure `reward.json`
    entries. Retry3 reused109 successful results, recomputed only that failed
    key and completed in253.2s. Commits `306c438`, `eac5e2b` and `7125452`
    implement these protections. The complete `whale_local/tests` suite passed
    135 tests; the final focused cache regression passed18.

    Node4 accepted candidate
    `60b43b8fd5f64852bd20234dbd4e50e71b46253a6f604adba86bc53f5f49a680`
    at87/100 versus79/100: paired gain+8pp and one-sided95% LCB+2pp.
    Node3 accepted
    `6e6750643c1b451c65c9be1db61f73db7e88d3ab9150ca005ec00e37d36a0234`
    at89/100 versus78/100: gain+11pp and LCB+5pp. These are clean
    train-candidate-to-dev global Harness results, not training rewards or
    same-packet rescue rates. Sealed test400--499 remains unread.

75. **Sequential H2 decision and next dual-node phases (2026-09-30).**
    Node4 continued from the accepted H1 actor/critic and completed H2's two
    actor/four critic updates in1,320.9s. Actor safetensors changed from
    `b0d656daddb0211678bb354ff234159e618f5de2244808cb15c55a92a30943e7`
    to
    `f203a239dde49069c69121cba1e3a4ce90170fd3da3f54c5385746179dca40f7`.
    Eleven consumed-version0 candidates were evaluated relative to the H1
    incumbent; none passed the incremental +1pp/positive-LCB gate. The
    incumbent re-evaluated at88/100 (+9pp versus no-patch, LCB+3pp), one pass
    above its first H1 evaluation. This run-level fluctuation is retained as
    measurement evidence and is not credited to H2. H2 was rejected.

    Sequential patience is therefore one of two non-improving H rounds after
    the accepted H1. H3 restarts from accepted H1, not rejected H2, using64
    unique rows at sample offsets128--191; the data SHA256 is
    `bb54c24f69d14ccccbbb26f51ac5ac05b0e648fd94cce7f885dbeb17e84349f9`.
    In parallel, node3 uses accepted E1/W0 to refresh all24 train packets and
    generate512 assignment-aligned W1 rows. W1's centralized critic context
    is frozen before action and records E1, selected harness identity and
    89/100 dev calibration. Solver cross-role replay remains disabled.

76. **Sequential Harness plateau and exact packet recovery
    (2026-09-30).** Node4 H3 restarted from accepted H1 rather than rejected
    H2, completed two actor/four critic updates in1,207.4s, and changed its
    final weights to
    `86b6eec4d052740e66731f0b69ae07cc1a0e23f4b5f2203db431cec8fd9e99d1`.
    Its 11-candidate dev gate had zero incremental gain over H1. The incumbent
    remained+9pp over no-patch with a+3pp one-sided95% LCB. H2 and H3 are two
    consecutive rejected Harness rounds, so the sequential arm freezes H1
    patch
    `60b43b8fd5f64852bd20234dbd4e50e71b46253a6f604adba86bc53f5f49a680`.

    MA W1 train refresh initially completed21/24 packets and sequential W1
    18/24. Neither failure set was converted to model score or recomputed in
    full. Serial partial recovery completed3 and6 missing starts, then a
    strict merger required exact, duplicate-free coverage of starts0--230.
    Final node3 baseline/packet-transductive rates were69.58%/71.25%;
    node4 was70.42%/71.25%. Both produced24 assignments,240 task identities
    and512 Solver rows. The transductive rates are training-data diagnostics,
    not held-out acceptance evidence.

77. **Formal dual-arm W1 and centralized-critic repairs
    (2026-09-30).** MA W1 had two zero-update failures. The first embedded a
    complete9.8KB patch in centralized context and exceeded the reserved
    1,024 tokens before model initialization. Context now records a stable
    patch SHA and H/W launchers tokenize-audit it before claiming GPUs. The
    second reached real rollout but failed before value forward because DP4
    gives each rank a1-D token trajectory; prefixing had assumed `[1,seq]`.
    The value path now prefixes1-D,2-D and general token-aligned tensors,
    updates masks/versions/values/returns/positions consistently, and strips
    the prefix from values returned to the actor.

    MA W1 retry2 then completed2 actor/4 critic updates in4,059.97s
    (optimizer3,919.08s) with a224-token pre-action centralized context.
    Weights changed from
    `435cb03ebd100f83817e129eb7fa11696123ec573923b102543a32f3e1fb559f`
    to
    `258e52983e8b59bb2b5c5d3c5d4bb141cf0adc1a324df4a23ff44d7052e43043`.
    Sequential W1 completed the same update budget in4,065.14s
    (optimizer3,920.90s), without centralized context, and ended at
    `81bc16629ec3fc66663c9ed3f08f65534b99ef565dea106b79dbf4343f0b6682`.
    Both used four actor/critic and four rollout GPUs. Long second-batch
    trajectories, not idle allocation, explain the roughly68-minute rounds.

78. **W1 gates, MA H2, and exact-consumption selection
    (2026-09-30).** MA W1 dev refresh recovered2 missing packets after an
    8/10 first pass. Its no-patch result tied W0 at78/100, so gain and LCB were
    zero and W1 was rejected. Sequential W1 recovered one missing packet
    after9/10; it scored81/100 versus W0's79/100, but its paired one-sided95%
    LCB was zero, so it too was rejected. Packet-transductive scores77/100
    and83/100 remain diagnostics.

    MA H2 used64 rows at offsets64--127:48 fresh plus16 authentic H1 replay
    rows in the dataset, interaction credit0.5, and maximum replay lag16
    decoupled from online rollout concurrency staleness1. It completed2
    actor/4 critic updates in1,217.98s; weights changed from
    `bf7017f860f0cfb69bdbb73a626c490e8091613add94500962fa61bc813dc1cd`
    to
    `a68594777e7db981a290e13d78898dac4735dcf6eb91344d61acf8ef4c973996`.
    The two actual optimizer batches consumed9 fresh and3 replay rows.

    The wider replay lag allowed20 unique fresh version-0 receipts to be
    prefetched, so the old version-only gate was no longer equivalent to
    optimizer consumption. The first H2 selector was stopped before producing
    a decision. Training now persists an exact first-`steps*batch` ID ledger;
    selection requires every declared fresh ID exactly once, rejects replay
    receipts, and ignores extra prefetched IDs. The corrected pool contained
    nine candidates. None beat H1; H2 was rejected with incremental gain0 and
    LCB0. H1 itself re-evaluated at90/100 versus no-patch78/100
    (+12pp, LCB+6pp); the one-task run fluctuation from89 is not H2 gain.
    Commits `e3ba7d6`, `5bbfb05`, and `2144c34` implement separated replay
    lag, replay exclusion, and exact consumed-ID gates. The latest focused
    suites passed32 tests.

    Both arms therefore start W2 from accepted W0, not rejected W1, using
    assignment-balanced rows at non-overlapping offsets512--1023. Node3 W2
    retains the pre-action centralized critic; node4 W2 is sequential. Both
    eight-GPU optimizer runs were active at23:02 UTC. Sealed test400--499
    remains unread.
79. **Both ALFWorld arms frozen; Solver SAO did not pass a dev gate
    (2026-10-01).** Sequential W2 completed2 actor/4 critic updates in
    4,728.70s from accepted W0 and changed its final Solver weight digest to
    `74eb852d15c21c867fa10b4eccf110b9873bcc7924a3bfad56cc4a263ad72e60`.
    Its paired no-patch dev comparison was79/100 for W0 versus77/100 for W2:
    −2pp with one-sided95% LCB−4pp. MA W2 completed the same update budget in
    4,162.78s with centralized critic enabled and digest
    `76f476f8cfb6f510bc63a0da03d1fb747faa4289b25112683386787cb06f800e`.
    It tied W0 at78/100, with gain and LCB both zero. Both candidates were
    rejected; W1/W2 satisfy Solver patience=2 in each arm, so both freeze the
    released `target-agent-sft` W0.

    MA H3 restarted from accepted E1, not rejected H2. It used offsets128--191,
    25% authentic replay, interaction credit0.5, and the same decentralized
    actor/pre-action centralized critic boundary. It completed2 actor/4 critic
    updates in1,163.34s and changed final Engineer weights to
    `7c5ddcccb21ba8298893f55dd63253c6bf02abdcaca8a14e1709c702f02025b3`.
    The optimizer consumed9 fresh and3 replay rows; the exact fresh gate found
    seven valid unique candidates. None improved on H1. H2/H3 therefore freeze
    MA patch
    `6e6750643c1b451c65c9be1db61f73db7e88d3ab9150ca005ec00e37d36a0234`.

    A final dev-only comparison re-evaluated both frozen H1+W0 arms:
    sequential88/100 and MA90/100. MA−sequential is+2pp with paired one-sided
    95% LCB0. It misses the preregistered+5pp/positive-LCB material-gain rule,
    so cadence search is explicitly disabled. The only accepted weight-state
    improvement in either arm is Harness H1; four completed Solver
    checkpoints changed real weights but none improved held-out no-patch task
    completion under the gate.
80. **Always-on adaptive GPU filler implemented and deployed
    (2026-10-01).** Auditing node0--2 found no reusable generic co-resident
    implementation: node0's unreadable guard honored a per-GPU reservation,
    node1 ran the fixed55%-memory filler, and node2 paused it under workload.
    The legacy worker held about80GiB per H200 and enforced an82% duty floor,
    so it could not safely stay active beside training.

    `sao_bridge.adaptive_gpu_filler` now supervises one small worker per
    eligible GPU and samples both compute-app identities and per-process SM
    utilization. It excludes only exact child PIDs. New or active external
    work yields immediately and releases the worker; sustained external
    idleness uses a48MiB matrix working set with tiered15%/30%/50% duty caps;
    an empty GPU uses192MiB and ramps toward86% total utilization. Less than
    8GiB free memory, temperature at least82C, or a reservation yields the
    card. Scale-down is immediate, scale-up is5 duty points per2s. Telemetry
    errors fail safe, worker heartbeats recover child failure, and systemd
    retains `Restart=always`.

    Node4 idle canary reached85.25% mean utilization across all8 GPUs at about
    791MiB total filler memory per GPU. A verified worker kill restarted with
    a new PID; a verified supervisor kill produced `NRestarts=1` and rebuilt
    all8 workers. A20s TF32 comparison completed55,584 batches with adaptive
    service active and56,414 with it off, a98.53% throughput ratio and1.47%
    observed overhead including handoff. A synthetic external process caused
    the service to downsize GPU0 and it resumed after process exit.

    The same versioned drop-in is installed on node3 and node4. Node3's root
    filesystem was initially full; only archived systemd journals were
    vacuumed, freeing enough space for the drop-in, and no user data was
    removed. Real sealed vLLM launch audits on both nodes record
    `filler_kind=adaptive`, `cooperative=true`, and `stopped=false`; all eight
    external model PIDs were detected and workers entered the small shared
    state during sustained model-load idle periods. The latest filler and
    ALFWorld focused regressions pass42 tests. After sealed jobs closed, the
    final full-yield/8GiB-margin code and `RequiresMountsFor` drop-in were
    reinstalled and restarted on both nodes. Both services are active/enabled,
    expose eight identity-verified workers and no foreign PID, and reached
    about86.4%/86.8% idle mean utilization. With a standalone workload at89%
    SM, node4 GPU0 reported `external_busy`, removed its worker and held zero
    filler buffers/context; it recreated the idle worker after natural exit.
    Node3 root still has only about553MiB free, so disk pressure remains an
    operational warning.
81. **Frozen-arm ALFWorld sealed result (2026-10-01).** Test IDs400--499
    remained unopened until both arms froze at H1+W0 and the final dev
    comparison disabled cadence search. Node4's sealed packet refresh
    completed10/10 cohorts. Node3 completed9/10; start470 had infrastructure
    failure on both bounded attempts, so only that start was rerun serially.
    The strict merger accepted exactly starts400--490 with no duplicate and no
    model-zero substitution.

    The two metadata sources have identical100 task IDs and paths, but
    independent no-patch rollouts differ on six task IDs:
    401,428,435,452,456,492. Their totals are67/100 and65/100. Because choosing
    either source after packet diagnostics would be asymmetric, before any
    frozen-patch result the rule was fixed to evaluate both patches on both
    sources and bootstrap the per-task average across the two replicates. The
    freeze receipt hashes the original final manifest, replication rule, both
    evaluation configs and metadata-comparison receipt while both final
    services were still active.

    | Source | No patch | Sequential H1 | MA H1 | MA−sequential | one-sided95% LCB |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | node4 metadata | 67 | 78 | 82 | +4pp | −1pp |
    | node3 metadata | 65 | 78 | 84 | +6pp | +1pp |
    | task-level two-replicate mean | 66 | 78 | 83 | **+5pp** | **0pp** |

    Sequential versus no-patch is+12pp with LCB+6pp. MA versus no-patch is
    +17pp with LCB+10.5pp. Both accepted H1 patches therefore generalize to
    sealed tasks under the frozen evaluation. MA reaches the preregistered
    +5pp point threshold over sequential exactly, but its LCB is zero; the
    conjunction fails, material MA superiority is not established, and
    cadence search remains disabled. No sealed result changes a checkpoint,
    patch, cadence, or hyperparameter.

    Each final fixed-patch evaluation completed without infrastructure
    failure in about352--365s while the adaptive filler service remained
    active and identity-verified. Legacy `RELEASED.json` files from these two
    runs list adaptive workers as remaining GPU PIDs; subsequent code now
    separates `verified_filler_pids` from `foreign_gpu_pids` so always-on
    utilization is not mistaken for leaked training work.
82. **Agent-independent adaptive filler lifecycle (2026-10-01).** Replaced
    the deployment dependency on a pre-existing legacy base unit plus manual
    drop-in with a repository-owned complete unit template and one lifecycle
    manager. `install` validates source, three CUDA runtime libraries, exactly
    eight GPUs and the rendered unit; installs the full
    `bruli-h200-load.service`; removes the compatibility drop-in; enables boot
    start; starts it; waits for a PID-matching status receipt; and runs the
    exact cgroup/user/command identity audit. The unit declares both required
    mounts, `Restart=always`, a5s restart delay and control-group cleanup.

    The manager also provides read-only `status`/`verify`, temporary
    `stop`/`start`, persistent `pause`/`resume`, `restart`, `render`, and
    state-preserving `uninstall`. Persistent pause is a node-local `PAUSED`
    condition marker and survives reboot and reinstall. A default
    `reservation.json` path is now wired into the unit for per-GPU yields.
    The complete development, deployment, operations, reservation, expected
    utilization and troubleshooting guide was initially
    `scripts/systemd/README.md`; entry83 supersedes that repository-local
    deployment with the standalone external folder.

    Node4 canary migrated to the full unit with no drop-in. Terminating the
    exact verified supervisor changed PID after the5s systemd delay and
    incremented `NRestarts` to1. `pause` stopped it, and a direct
    `systemctl start` remained inactive because the condition marker existed;
    `resume` cleared the marker and rebuilt eight verified workers. Node3 was
    then migrated and both services are active, enabled, identity-verified
    and drop-in-free. The latest empty-node snapshot after convergence was
    85.25% mean on node3 and85.5% on node4, with every GPU at85% or86% and
    about791MiB filler memory. Focused regressions now pass44 tests; shell
    syntax, rendered-unit verification and systemd verification also pass.
83. **Standalone filler relocation (2026-10-01).** The complete adaptive
    runtime, unit template, manager, tests and operator README were moved out
    of `coevolve` to `/mnt/3fs/xudong/gpu-utilization/`. The rendered service
    now executes `/usr/bin/python3 -B
    /mnt/3fs/xudong/gpu-utilization/adaptive_gpu_filler.py supervisor`,
    requires only the standalone code and state mounts, and has no coevolve,
    virtualenv or PYTHONPATH runtime dependency. The worker launcher uses its
    resolved standalone script path instead of a Python module import.

    Coevolve retains only the exact service identity matcher required by
    training launch/release audits. It now authorizes the standalone absolute
    command plus the existing unit cgroup and `bruli` UID; similarly named
    scripts or processes remain foreign. Standalone policy/unit tests moved
    with the runtime rather than introducing an external path dependency into
    repository tests. This relocation supersedes the repository-local paths
    recorded in entry82.

    After migration, node4 again passed two-level recovery: terminating one
    exact worker changed PID under the same supervisor, and terminating the
    supervisor changed PID with `NRestarts=1`. Both effective units contain no
    coevolve path, remain active/enabled, and coevolve's updated resource audit
    identifies eight standalone workers per node.

    A final-policy ABBA coexistence measurement used four identical30s,
    4,096-matrix TF32 trials at95% duty on node4 GPU0. Active trials completed
    83,103 and85,340 batches; service-off trials completed84,868 and84,932.
    Active/off means were84,221.5/84,900, ratio99.20%, or0.80% observed
    overhead. Both active handoff receipts require the external PID, yielded
    decision, no filler worker and zero filler buffers. This measures a
    compute-heavy proxy including initial handoff, not full PPO/vLLM
    end-to-end speed; low-SM workloads eligible for shared mode require their
    own A/B or an explicit reservation.
84. **Sequential SAO hardening protocol started (2026-10-01).** The next
    research order is frozen to avoid using MA mechanisms to rescue an
    unqualified sequential baseline: (A) fixed-H0 4B Solver, (B) fixed
    no-patch 9B ALFWorld Solver, (C) same-budget 9B Engineer E0/E1 capability,
    and only then (D) MA-SAO. If C fails, reproduce the official Harness-R1
    group-relative GRPO objective before attributing failure to the model or
    task.

    Stage A uses two predeclared seeds, 8 critic-only warm-up steps followed by
    64 actor steps at batch64, for4,608 authentic trajectories per seed.
    Checkpoints are saved after total steps24,48,72; selection uses the
    disjoint256-question reward split at mean@2 with a
    Bonferroni-adjusted paired lower bound; the untouched report is512
    questions at mean@2. The pair gate requires both seeds to select a trained
    checkpoint and not regress, mean gain at least+3pp, and at least one seed
    with a positive question-bootstrap95% lower bound. Test/AIME stays sealed.
    Runs are persistent systemd services on node3/node4 and retry only bounded
    infrastructure failures.

    A per-process GPU telemetry path now separates the exact adaptive-filler
    PIDs from all exclusive experiment PIDs. During an early node4 trainer
    startup sample, total SM was72.75% but only1.94% belonged to the
    experiment and71.88% to filler. During an active node3 optimizer window,
    total was50.0%, experiment48.22%, filler0%; the filler unit remained
    active in both samples. These are short diagnostic windows, not final
    throughput estimates. The filler is never stopped or killed by the
    adaptive-aware training allocator.
85. **9B promoted to the primary proof target; speed root cause isolated
    (2026-10-01).** Stage A remains a frozen two-seed calibration and will be
    reported as designed, but it is no longer a hard prerequisite that may
    indefinitely block 9B. The required sequential evidence is now a fixed
    no-patch 9B Solver gate followed by an equal-budget 9B Engineer E0/E1
    capability gate; MA-SAO stays locked until both pass.

    Historical 9B ALFWorld metrics show that environment concurrency is not
    the current wall-clock bottleneck. On a representative second optimizer
    step, asynchronous rollout wait was already0.0s, while actor update took
    885.2s and the two critic updates took1801.3s; value/log-prob recomputation
    added about137s. A deeper code/receipt audit then found the dominant token
    amplification: four episodes were exported as166 cumulative-prefix turn
    sequences and the actor reported about2.16M tokens. Per-episode receipts
    show individual-prefix/final-conversation token ratios from4.7x to31.2x;
    over12 persisted episodes the aggregate is1,736,474 versus90,153 tokens,
    or19.26x.
    AReaL already constructs an exact-token parent chain and preserves every
    assistant span's sampled log-probability and loss mask, including tool-call
    conversations; only the final local export still requested `individual`.
    The first speed qualification therefore keeps the proven4+4 asynchronous
    layout and tests exact-token `concat` plus K=1, initially retaining
    gradient checkpointing for a safe memory baseline. A learned-critic run
    remains the primary SAO baseline. Critic-free terminal-return PPO is a
    separately labelled ablation, not a relabelling of learned-critic SAO.

    A persistent 9B controller now enforces exact empty-patch assignments,
    train/selection/report separation (0–239/240–299/300–399), same-seed
    paired W0/W1 evaluation, sparse checkpoint selection, bounded
    infrastructure retry, source manifests and release audits. Tasks400–499
    remain sealed. The controller has passed pure configuration/selection
    tests but must not launch a formal run until the underlying speed controls
    are wired into the native trainer and a live evaluator/speed smoke passes.

    The temporary all-GPU filler reservation used for clean telemetry was
    removed after measurement. Both systemd units remain active in adaptive
    shared mode: they yield at high external SM and apply only low duty during
    eligible idle gaps. In matched five-minute steady 4B windows, node4
    experiment/filler SM averaged58.10%/3.29% and node3 averaged58.73%/3.55%;
    all49 samples per node were valid and both units stayed active. Future
    strict throughput A/B windows may use bounded reservations, but routine
    training does not stop or kill the filler.
86. **4B calibration failed its frozen gate; exact-concat 9B speed smoke
    passed (2026-10-01).** Both4B services completed72 optimizer steps with
    zero restarts and evaluated all three predeclared checkpoints. Neither
    seed cleared the Bonferroni-adjusted selection lower bound, so both
    correctly retained W0 and no trained-checkpoint report was opened. The
    final exploratory selection deltas were+1.37pp for seed42 and+5.08pp for
    seed43; seed43's lower bound was still-0.20pp. The frozen two-seed gate
    therefore failed. This does not block the already-promoted9B proof, and
    it is not valid to report the exploratory average as an independent
    improvement.

    The first learned-critic9B exact-concat smoke then completed one real
    batch of8 no-patch ALFWorld episodes. It exported exactly8 trajectories
    and42,996 actor tokens, versus166 trajectories and2,164,206 actor tokens
    in the representative historical batch4 individual-prefix step. Actor
    update fell from885.2s to12.84s; critic fell from1801.3s for K=2 to13.16s
    for K=1. Including the first50.18s rollout but excluding checkpoint save,
    the core step fell from about2826s to85s while batch doubled, or roughly
    66x episode throughput. Actor+critic checkpoint save still took191.3s,
    so sparse saves remain mandatory.

    The smoke used4 actor+4 rollout GPUs and kept actor/critic gradient
    checkpointing enabled. Peak node memory was102,540MiB; actor plus critic
    backward consumed only26.0s while rollout consumed50.2s. Disabling
    checkpointing can no longer improve steady throughput enough to justify
    its OOM risk, so the formal baseline keeps it enabled. The critic's first
    unclipped grad norm was324.2, confirming that critic-only warm-up remains
    necessary before actor updates.

    The new explicit-seed evaluator required two infrastructure fixes before
    its live smoke was valid. First, the shared evaluator assumed the first
    eight characters of every row ID were hexadecimal; ALFWorld semantic IDs
    now preserve the old hex behavior and otherwise use a stable SHA-256
    offset. Second, the ALFWorld controller URL had only been placed in child
    launch environments even though episodes execute in the evaluator parent;
    it is now bound to both. Both failed directories were archived and contain
    no scientific result. The third smoke completed tasks0 and1 with the
    official environment,8 vLLM replicas, explicit seed and exact no-patch;
    its1/2 score is only an infrastructure check. No task at or above400 was
    opened.

    The first dual-seed formal startup then exposed a scale-only concurrency
    bug before either baseline completed: the sampler limited concurrent LLM
    requests to32, but all160 episode coroutines called `start_sample` before
    acquiring that semaphore, overflowing the environment worker and returning
    HTTP406. Both services were stopped, their no-result roots were archived,
    and filler remained active. The shared evaluator now gates the entire
    episode lifetime with the declared environment concurrency; a regression
    test observes a hard peak of2 for6 concurrent test episodes. Formal runs
    must restart from fresh roots because the source manifest changed.
87. **Exact-concat 9B Solver finished; independent two-seed gate failed
    (2026-10-01).** Both seeds completed 30 optimizer steps: five critic-only
    warm-up steps and 25 actor steps, batch eight, learned critic K=1,
    gradient checkpointing enabled, fixed empty harness. The schedule covered
    240 distinct train tasks, of which the final 200 supplied actor updates.
    Every actor batch contained exactly eight concat trajectories, all three
    actor/critic checkpoints at global steps 9/19/29 were saved, and the two
    training wrappers completed in 52.36/52.50 minutes. Nonzero actor gradients
    and improving critic loss establish update health, not held-out benefit.

    | Seed | W0 selection | Step 9 / 19 / 29 selection | Selected step | Independent report W0 / candidate | Delta and paired question 95% CI |
    |---|---|---|---|---|---|
    | 42 | 61.67% | 65.00% / 56.67% / 56.67% | 9 | 74% / 72% | -2pp [-7,+3]pp |
    | 43 | 58.33% | 60.00% / 60.00% / 61.67% | 29 | 79% / 77% | -2pp [-11,+7]pp |

    Selection used tasks 240-299 and a frozen +2pp floor; the independent
    report used tasks 300-399. No task 400-499 was opened in this campaign.
    Both seeds selected a trained checkpoint, but the non-degradation,
    average +3pp, and significant-seed requirements all failed. The mean
    report delta was -2pp. These intervals cover question uncertainty, not
    training-seed population uncertainty. Seed 43 finished at 06:20:46 UTC
    and seed 42 at 06:40:35 UTC; the final pair receipt is
    `runs/alfworld_solver_v1_pair_gate_result.json`. Later selection decline
    in seed 42 motivates diagnosis, but does not by itself prove overfitting.
    Engineer capability and MA-SAO experiments remain gated.

    Seed 42's step-29 evaluator exposed a further semantic bug: exhausting
    the declared 4096-token episode budget raised an infrastructure exception
    and repeatedly restarted the whole evaluation. The controller was stopped
    without touching filler, and the domain now records that outcome as
    zero reward with `truncated=true`, preserving token/turn/message accounting
    and cancelling the live environment session. The targeted ALFWorld,
    evaluator and domain suite passed 34 tests. The resumed 60-task evaluation
    completed with 3/60 truncated failures and 56.67% accuracy. No training
    was repeated and no task, decoding seed, token budget, checkpoint or
    selection threshold changed. A named source-manifest migration receipt
    and the original manifest were retained; the run is not byte-identical
    throughout. Completed earlier evaluations had no exhausted episodes and
    were retained. The selected early checkpoint's independent report still
    failed, so this repair cannot be presented as a positive algorithm result.

    Formal optimizer-window telemetry averaged experiment/filler SM of
    40.21%/6.63% on node4 and 41.44%/6.60% on node3, with zero invalid samples.
    Peak memory was 134,988 and 129,332 MiB respectively; filler remained
    active throughout. The short smoke's approximately 66x episode throughput
    must not be presented as a full training-plus-evaluation speedup.
    A monitoring mistake queried system services with `systemctl --user`;
    the apparent absent controller/filler units were not service failures.
    The redundant user guardian was removed. Real controller restarts were
    identified through its system cgroup and evaluator failure receipts.

    At the 11:08 UTC status check, both finite experiment queues were complete
    and no new experiment was running. Fresh filler receipts showed node3
    87.13% and node4 86.50% total utilization, with external experiment SM zero.
    These numbers are filler load only. The attached worker status dated
    2026-09-29 belongs to an older MA-SAO run, not this Solver validation.
    Further work must diagnose sequential SAO sample efficiency/credit
    assignment and predeclare the next experiment rather than claim that
    successful execution has already proved 9B learning.
88. **Padding-dependent terminal bootstrap reproduced; 4B-first scale
    protocol declared (2026-10-01).** A CPU reproduction using the actual
    native PPOActor found that a completed episode with reward 1 and final
    state value 3 has return 4 when it fills its batch width, but return 1
    when the same episode has padding. Native GAE initializes its final
    bootstrap with `values[:, -1] * (seqlen == batch_width)`, which does not
    establish whether an application episode terminated. In the completed
    9B run, each rank held two episodes and the logged no-EOS ratio was
    consistently 0.5; cold-start value targets could exceed the binary reward.
    This is a reproducible implementation error, not yet a causal explanation
    for all observed held-out quality differences.

    The local trainer now has an explicit, default-off `terminal_episodes`
    control. For complete episodes it zeros only the post-final-action state
    value before native advantage computation, leaving every action-state
    value and native loss unchanged. Fresh Solver episodes must use concat;
    unverified replay/partial-turn trajectories are rejected. The control is
    wired through both Solver and Engineer wrappers and persisted receipts.
    Tests reproduce the old padding dependence and verify exact Monte-Carlo
    returns for rewards 0/1/-1, padded and unpadded episodes, masked tool
    observations, critic-free operation and unchanged input values.

    The user's updated order is 4B Solver algorithm/scale validation, then
    9B Solver, then 9B Engineer. Previously positive 4B Chess and SearchQA
    Solver results remain evidence of learnability; the recent stricter
    math calibration failure does not erase them. Prepared 9B terminal-fix
    configurations are deferred, not running.

    Predeclared 4B v3 uses the same initial 4B/h0, LR 2e-6, critic LR 1e-5,
    batch64, temperature1, staleness1 and eight critic-only warm-up steps.
    Exact concat plus correct terminal boundaries are enabled; learned
    critic K=2 is retained. Training covers 15,360 distinct questions in
    240 steps (232 actor), versus 4,608 questions in v2, preserving the
    384-question held-out train tail. Checkpoints every24 steps retain the
    old23/47/71 comparison points and add a longer scale curve, without
    optimizer resets. These checkpoint comparisons are exploratory and
    do not separately identify concat versus terminal-boundary effects.

    Ten checkpoints share a Bonferroni selection budget on256 questions
    at mean@2; only the selected model is reported on512 separate questions
    at mean@2. These report questions were previously used, so this is
    development replication rather than a pristine confirmatory test.
    The stronger promotion gate requires both trained models selected,
    each report gain at least2pp, mean gain at least5pp and both paired
    question95% lower bounds positive. The sealed AIME split stays closed.
    Runtime manifests now cover the boundary helper and all related math
    evaluation code, and optimizer-window telemetry excludes filler.

    The first live v3 startup failed before any optimizer update because
    controller-side data contains RTensor handles, not materialized torch
    tensors. CPU GAE tests alone did not cover that boundary. Both training
    units were stopped with filler untouched, and the no-update roots were
    archived. Boundary handling now runs inside an importable FSDP worker
    subclass, after RPC materialization and native batching, before native
    GAE. An actual worker batching/unpacking regression was added; the
    targeted suite passed65 tests and9 subtests. Fresh roots, unchanged
    scientific specs and new frozen source manifests are required for retry.

    Live verification under commit `7d765e5` succeeded on both nodes:
    seed42/43 reached15/16 optimizer records; every measured critic target
    was in[0,1]. The first8 steps had no actor-update metric, followed by7/8
    positive actor-gradient steps, exactly64 concatenated episodes per batch
    and2 critic updates per rank. The audit and metric-prefix hashes are in
    `runs/dev_solver_only_v3_live_terminal_validation.json`. Early optimizer
    windows measured62.399%/64.069% non-filler SM versus0.485%/0.558% filler
    SM (58/60 successful samples). Both `-r1` system controllers remained
    active with zero restarts; both filler supervisors stayed active.
    Added a controller-factory regression whose remote-handle stand-in has
    no tensor arithmetic, testing both default and terminal modes. Its
    isolated runtime file passed12 tests and11 subtests; the preceding
    six-file suite passed65 tests and9 subtests before this one extra test.
    This verifies execution and target correctness, NOT held-out improvement.
    The240-step training and selection/report protocol remain in progress.

89. **Train-only ALFWorld scaling prepared while frozen4B jobs run
    (2026-10-01; no9B model execution).** The legacy adapter had hard limits
    of500 catalogue entries and240 unique training tasks. Added an explicit
    checksum-bound training catalogue and run-local data overlays, retaining
    the default official test-set behavior and its evaluation ID bounds.
    The first catalogue-write attempt hit the shared dataset's read-only
    permissions; the implementation was corrected to leave that directory
    untouched rather than changing ownership or permissions.

    All500 existing evaluation games and every `valid_seen`, `valid_unseen`,
    and `valid_train` goal group are excluded. The eligible pool contains
    2,296 game instances across880 goal groups; these are not2,296 unique
    goals. Variant instances are ordered after initial goal coverage.
    Metadata-only auditing found17/24/5 shared goal groups between the old
    train-selection/train-report/selection-report partitions, respectively.
    Task IDs were disjoint, but the old results were not goal-unseen
    validation. Their outcomes and task-level intervals are not rewritten.

    The new scale specs reset toW0/H0, retain batch8, LR, K1, concat,
    terminal handling andGC, and declare5 critic-only plus235 actor steps:
    1,920 different game instances, covering880 training goals, with8 saved
    checkpoints. Evaluation uses59 selection and91 report goals, one game
    instance each and mean@2; selection/report goals are disjoint and no
    400--499 evaluation outcome is accessed. These reused tasks are still
    development data. This changes training distribution, amount of data,
    and evaluation protocol, not just optimizer steps. The existing9B pair
    promotion thresholds remain unchanged.

    The native trainer receives only the exact declared row prefix, with
    matching environment assignments; asynchronous prefetch cannot address
    an unregistered suffix. Catalogue/data hashes are frozen and cached
    reports cannot bypass the evaluation-catalogue hash check.
    Validation:51 tests plus7 subtests passed; both nodes passed CPU protocol
    checks. A real CPU-only AgentBench run on task512 performed one synthetic
    `look`, hit its one-token budget, cancelled its session, and terminated
    only its owned environment processes. No model or GPU was used and this
    is not an accuracy result. Receipts:
    `runs/alfworld_train_catalogue_cpu_smoke_v1/validation.json` and
    `assets/alfworld_solver_train_only_v1/protocol_preflight.json`.
    Both live4B source manifests were reverified unchanged.9B remains gated
    on stronger4B evidence; its v2 control is still valid and unstarted.

90. **Corrected/scaled4B Solver passes its stronger dual-seed gate
    (2026-10-01 14:15 UTC).** FixedH0,512 report questions, mean@2:

    | seed | selected step | W0 report | candidate report | delta | question-level95%CI |
    |---|---|---|---|---|---|
    |42|143|41.5039%|56.5430%|+15.0391pp|[+11.6211,+18.4570]pp|
    |43|191|38.1836%|51.5625%|+13.3789pp|[+9.8633,+16.9922]pp|

    Mean delta+14.2090pp. All predeclared checks passed: trained checkpoint
    selected in both seeds, each delta>=+2pp, mean>=+5pp, both question-level
    lower bounds>0. The report remains reused development data, not a new
    confirmatory test or population-level training-seed significance result.
    Terminal correction, exact concat and scaling were combined; do not claim
    a factor-separated causal attribution or compare gains across benchmarks.

    Both completed240 steps/15,360 unique training tasks and10 actor/critic
    checkpoint pairs. All measured critic targets stayed in[0,1]; warm-up
    actor isolation and real subsequent updates were verified. Optimizer
    windows took117.69/117.52 minutes, with58.40%/57.55% true experiment SM
    and1.82%/1.94% filler SM; filler remained active throughout.

    Seed42 had one evaluation-only restart at13:54:23 UTC, when step119
    replica6 exited before its health endpoint was ready. Existing training
    and completed evaluation caches were reused; no scientific settings
    changed. Original replica stderr was overwritten by the old launcher on
    retry, so the underlying startup cause is unknown. Later tool-parser
    warnings are not that startup failure. The recovery receipt preserves
    the observed facts; final controller elapsed_seconds is not full-trial
    duration after restart. Subsequent same-name process logs are now
    archived under previous_logs before opening the new log.

    The4B prerequisite is satisfied. Next execute the frozen9B v2 same-budget
    terminal-fix control, then the goal-isolated scale protocol; Engineer and
    MA-SAO remain gated.

    At the previous step71 budget, v3 selection gains were+6.4453/+6.6406pp,
    with ten-way-adjusted LCB0/+0.9766pp. Later selected steps143/191 reached
    +12.8906/+14.0625pp on selection. This is useful scaling-curve evidence,
    not separate report evidence for the early checkpoints.

    At14:41 UTC,9B v2 control services started on both nodes under2fed1fd;
    baseline APIs returned200 and both entered training at14:56 UTC.
    Original start timestamps, service names, source/spec identities and the
    passing4B prerequisite hash are stored in launch_receipt.json. No9B
    improvement is claimed at launch. Filler remains active.

91. **Terminal-corrected9B control completes; switch to fast single-seed
    exploration (2026-10-01 16:20 UTC).** Both seeds selected step29.
    Seed42 report74% ->81%, delta+7pp, question-level95%CI[0,+15]pp.
    Seed43 report81% ->74%, delta-7pp, CI[-17,+3]pp. Mean delta0pp;
    the original pair gate failed and is not relaxed retrospectively.
    Both selection scores rose61.67% ->70%. Seed43's baseline differs from
    v1's79% despite identical nominal decoding, so this is not a bitwise
    paired algorithm ablation. No stable9B learning claim follows.

    Both completed30 native steps/25 nonzero Actor updates, targets in[0,1],
    and all3 actor/critic checkpoints. No controller restart was observed.
    Optimizer windows43.36/46.65 minutes; true experiment SM40.25%/42.16%,
    filler6.55%/6.32%, filler active throughout. Persisted final native/source
    audit precedes any subsequent shared-source edit.

    User now prioritizes finding useful performance over seed replication
    or ablations. The unstarted240-step dual-seed v3 plan is suspended,
    not silently relabeled as executed. New Solver v4 uses one seed42,
    freshW0/H0,640 goal-isolated train games,5+75 steps, batch8, LR2e-6/1e-5,
    K1 and2 checkpoints.59/91 mean@2 selection/report remain independent
    of the new training goals. Any success is exploratory, not replicated.

    Prepare Engineer learning in parallel rather than waiting for a Solver
    pair gate. The official builder reconstructs24 compact packets from
    unchanged cached training evidence; no new GPU collection. Exact maximum
    prompt7,733 plus4,096 generation fits16,384 microbatch. The first single
    seed H budget is5+27 steps, batch6,3+1+4 GPU layout, terminal fix andK1.
    Matched E0/E1 generation uses8 identical evidence prompts and candidate
    counts; metadata-only filtering yields24 selection and70 report goals
    disjoint from all legacy training goals and each other. This evaluates
    global proposal quality conditioned on fixed evidence, not unseen-packet
    transfer.61 tests and22 subtests pass; GPU H evaluation is not yet done.

92. **Fast single-seed Solver/Engineer run in parallel; Engineer training
    completes (2026-10-01).** Node4 runs Solver fast-v4 under source5e599ba;
    node3 runs Engineer fast-v1 underbb1f214. No duplicate-seed v3 jobs were
    launched. Both retain active adaptive filler and per-process attribution.

    Engineer completed32 native steps,5 critic-only plus27 nonzero Actor
    updates, with two complete actor/critic checkpoints, one attempt, and
    matching frozen sources. Signed targets ranged approximately-0.70000005
    to+0.40000001. Optimizer-window true experiment SM33.16%, filler7.99%;
    peak127,844MiB. Training completion is not an Engineer improvement result.

    Matched E0/step15/step31 evaluation is now implemented: same8 compact
    prompts, per-prompt decode seeds, generation cap and fixed Solver.
    Official sparse-task evaluation scores only24 selection/70 report goals.
    Duplicate or invalid proposals remain in the eight-proposal mean.
    Successful deterministic target evaluations may be cached across identical
    patches; failed transport/environment jobs are never assigned a no-patch
    accuracy. Extra no-patch-relative diagnostics distinguish useful editing
    from merely abstaining instead of harmful edits, without changing the
    predeclared relative E1/E0 gate.

    A real CPU-only environment check used a local mock API on tasks240/247
    and verified exact sparse task mapping. It made zero real model/GPU calls
    and does not contribute to any accuracy result.

93. **Repair matched-evaluation port collisions without changing model
    evidence (2026-10-01).** All24 Engineer proposals completed: E0 andstep15
    each8/8 valid, step31 5/8 valid with3 missing complete patch blocks.
    Five baseline shards and15 selection target jobs completed. Five
    unfinished jobs have explicit worker `EADDRINUSE` logs (ports46172,
    46196,46244,46316,46340; two second attempts also collided); another job
    was interrupted by controller teardown. These fixed ports fall in the
    host ephemeral range32768--60999. The occupying processes at failure
    time were not captured, so do not claim a particular process caused it.

    Reuse the official locked/free-port allocator outside the ephemeral
    range. Authorize exactly two further attempts only for the six
    unfinished target jobs; preserve all original attempts and bind
    successful cache checksums in a recovery receipt. No Engineer
    regeneration, candidate filtering, task change or gate change.
    Cached proposals now skip Engineer model startup, and exhausted job
    budgets are detected before GPU allocation and exit78. Old identity
    and failure/release receipts are archived rather than silently replaced.
    This is an infrastructure recovery, not an algorithm outcome.

94. **Recover Engineer target evaluation; diagnose Solver malformed-call
    failures (2026-10-01).** The node3 r1 evaluation service now runs the
    locked-port fix under454a861. Its recovery receipt preserves91 file
    hashes, all24 proposals, and the original attempts; exactly the six
    incomplete target jobs received two extra attempts each. All six
    finished on locked ports outside the ephemeral range. Selection mean@8
    on24 tasks is E0 69.27%, step15 76.56%, step31 67.19%. Step15 is selected
    under the unchanged rule; the70-task independent report is pending.
    These selection scores are not a held-out improvement conclusion.

    Solver v4 finished80 native steps and75 nonzero Actor updates, with
    targets in[0,1] and one training attempt. Step79 evaluation repeatedly
    failed because a generated tool-call argument was unterminated JSON;
    the next vLLM history rendering raised JSONDecodeError/HTTP400. The
    controller retry loop was stopped, without stopping the independent
    filler. Native training audit, original sources and monitor identity
    were saved before editing.

    The adapter now records unroundtrippable generated argument JSON as
    a terminal zero-reward model-format failure, preserving its raw message
    and token accounting and cancelling the environment session. Ordinary
    HTTP/environment errors still fail the evaluation explicitly, now with
    the rejection body available. This is a declared conservative format
    policy correction, not an infrastructure-only change: the official
    worker could otherwise give feedback and continue after an invalid
    call, but the current serving template cannot render that history.
    Completed W0/step39 cached reports contain no malformed arguments in
    7,886/3,099 tool calls, respectively. Their outcomes are unaffected.
    Training is not repeated, and task sets, sampling seeds, token budgets,
    checkpoint candidates and promotion gates remain unchanged.
    Focused CPU validation:51 tests and25 subtests passed.

95. **First matched 9B SAO Engineer improvement passes the independent
    report gate (2026-10-01).** All target jobs completed after the bounded
    port repair, with no regenerated Engineer proposals. On70 goal-disjoint
    report tasks, the mean over the same8 proposals rises from E0 72.1429%
    to step15 80.0000%: +7.8571pp, question-cluster95%CI
    [+4.4643,+11.2500]pp. Both models have8/8 valid proposals. The common
    no-patch Solver score is75.7143%, so E15 also gives+4.2857pp over no-patch;
    this is useful editing, not only less harmful editing or abstention.
    There are44 improved and11 regressed question means.

    Step15 has11 Actor updates after5 critic-only warmup steps. Later
    step31 remains a negative selection result with3/8 invalid proposals.
    The original gate and useful-editing diagnostic both pass. Scope:
    one training seed, the same8 train-evidence prompts, and unseen target
    tasks; not unseen-packet transfer or seed-level robustness.
    The142-file final evidence audit independently recomputes the paired
    report and verifies66 completed jobs, frozen sources and recovery
    caches. Full evaluation wall time104.63 minutes includes failure and
    repair overhead; the18.59-minute recovery segment is not total runtime.
    Fresh GPU release audit confirms no non-filler processes remain.

    The conditional GRPO fallback is not needed for this positive SAO-H
    result; no new GRPO run occurred. MA-SAO still waits for a positive
    9B Solver baseline. A distinct node3 Solver v5 is prepared:6 learner+
    2 rollout GPUs, batch12,8 critic-only+40 Actor steps, LR1e-6/1e-5,
    checkpoints23/47. V4's measured serial Actor/critic times are28.37/
    27.92 seconds per Actor batch; rollout wait averages4.26 seconds with
    near-zero median. The proposed rebalance, lower LR and longer critic
    calibration are a joint exploratory recipe, not a speed claim or an
    ablation. It retains fresh W0/H0 and the original goal-isolated gate.

96. **9B Solver v4 also passes; both SAO learning prerequisites are now
    established at the requested exploratory standard (2026-10-01).**
    Selection59-task mean@2: W0 66.9492%, step39 66.1017%, step79 79.6610%.
    The selected step79 independently improves91-task mean@2 from78.0220%
    to89.5604%, +11.5385pp, paired question95%CI[+3.2967,+19.7802]pp.
    There are23 improved and8 regressed question means. All original
    single-seed gate checks pass. One selection and two report episodes
    contain malformed tool arguments; all are retained as explicit zero
    model-format outcomes. No infrastructure errors are assigned rewards.

    The final native evidence audit recomputes the comparison, verifies
    immutable pre-recovery training/cache hashes,80 optimizer steps/
    75 Actor updates, and a fresh GPU release receipt. Total wall time
    from the original launch is199.82 minutes, not the22.96-minute
    restarted controller segment. V1's paired-seed mean-2pp and v2's mean0pp
    remain negative evidence. V4 jointly changes training pool, scale and
    LR; its success is neither a causal ablation nor seed robustness.

    The Engineer prerequisite already passed in entry95. New MA-SAO work
    may now proceed, but requires genuinely matched starts and budgets;
    adding the separately measured gains does not prove co-adaptation.

    A separate v5 speed/stability direction uses a detached worktree.
    Its first two attempts made zero native updates: Hydra resolved a
    shared-upstream symlink differently from its relative config base.
    Canonicalize dependency PYTHONPATH entries before import, retaining
    the frozen worktree bridge. A real CPU Hydra regression covers this
    failure. The node3 monitor also needs an output inside its owned run,
    not the node4-owned parent. Preserve these failed startup attempts;
    recovery uses an explicitly linked new run, not erased retry history.

97. **2026-10-01 — verified v5 learning and bounded strong MA preparation.**
    The explicit v5-r1 retry on node3 now has real native updates, not merely
    loaded models: at20:32UTC,20 steps/12 nonzero Actor updates, targets[0,1],
    matching frozen source and no controller restarts. First12 Actor-step
    timing means are35.33s Actor,34.84s critic and11.72s rollout wait with
    batch12. Compare per-sample and whole-run costs after completion; this
    is not yet a quality result or a proven speedup over batch8.

    On node4, a frozen31c66f4 train-only W79 probe completed240/240 episodes
    in17.97 minutes:191 successes, one retained environment-invalid-action
    episode. No report/test was opened. Release audit shows only the active
    adaptive filler. This79.58% is training feedback, not a paired gain;
    the old packet baseline has a different token/serving protocol.

    Declare `examples/alfworld_strong_sao_ma_v1.json`: shared validated E15
    prefix/W0, sequential H2-W1-W2 versus MA W1-H2-W2. H2 is32 native
    steps(5 warmup+27 Actor); each W block is80(5+75). Both arms receive
    the same role budgets and Adam-reset counts. The prefix actually spent
    32 steps/27 Actor updates, although selected E15 has11 Actor updates;
    discarded prefix work is not free. W1/W2 use1280 different instances/
    880 train goals, isolated from the24/70 evaluation goals.

    An important compatibility gate remains enforced: corrected terminal
    training forbids legacy replay/recycled rows. Initial strong MA therefore
    uses train-only centralized context and counterpart-refresh alternation,
    with replay, interaction bonuses and Solver recycling OFF. Do not revert
    terminal semantics or manufacture behavior probabilities to add features.
    This shared-prefix, reused-development-cohort continuation is exploratory,
    not a fresh seed replication or proof of convergence.

    Native Solver evaluation gains an explicit `--harness-matrix` mode:
    one serving pool, frozen patches, all duplicate labels retained,
    common task/sample seeds, bounded parallel harnesses, incremental reports
    and peer cancellation before client cleanup. The default no-patch CLI
    remains strict.44 CPU tests plus12 subtests pass, including real evaluator
    seed pairing and error cleanup. Combined AReaL imports reset the stdlib
    logger manager; logging tests now capture the actual module logger rather
    than a newly looked-up, orphaned name. Upstream sources were not changed.
    Real GPU matrix validation and the paired continuation are still pending.

    The first common matrix unit failed during import, before creating an
    evaluation root or loading any GPU model: snapshot setup omitted the
    root-level WHALE submodule. Preserve the journal and zero-compute receipt
    in `common/INITIAL_STARTUP_FAILURE.json`; enumerate `.gitmodules` rather
    than only `third_party` on retry. Runtime-wiring inspection also found
    that the environment service deliberately required partitioned IDs.
    Add a separate explicit matrix flag permitting identical cohorts under
    distinct worker names, without weakening training partition checks.
    An end-to-end CPU runner test verifies that flag and patch snapshots;
    the expanded suite passes85 tests/32 subtests. Its older token-budget
    fixture now supplies structurally valid OpenAI tool arguments, preserving
    its intended budget test under the already-adopted format-error semantics.

98. **2026-10-01 — fix real high-concurrency controller starvation.**
    Matrix r1 registered all9 independent workers, applied8 distinct
    `harness_overlay` configurations and reached a healthy inference server.
    It then failed with start_sample404s after a burst of successful starts.
    The pinned minimal controller uses blocking requests in async handlers;
    queued heartbeats cannot update its60-second alive-worker clock.

    A deterministic CPU reproduction of128 two-second forwarded starts
    produces30 successes/98 false-dead404s under that controller. The new
    run-local async bridge handles128 concurrent forwards while admitting
    heartbeats, preserves session headers/response unwrapping and surfaces
    genuine HTTP/transport failures. It is explicitly enabled for matrix
    mode; no shared upstream or active v5 source was edited.
    Unit coverage includes concurrency, cancellation, expired workers and
    failure propagation (35 targeted tests/27 subtests passed).

    Real official-worker validation on legacy train0--31:128 starts and128
    cancellations succeeded, burst48.3235s, all workers alive, remaining
    sessions0. It generated no Solver completions and is not a score.
    `common/async_controller_preflight/COMPLETE.json` persists this evidence.
    Preserve r1 as infrastructure failure and retry the exact frozen matrix
    rather than assigning fallback rewards.

    Separately, v5 finished48 native steps/40 nonzero Actor updates. Its
    first step23 evaluation hit a vLLM transport ReadError, not this controller
    bug. Systemd recovered without retraining; step23 then completed at
    59.32% selection mean@2. Step47 selection is still pending, so no final
    v5 quality or speed conclusion is available yet.

99. **2026-10-01 — common native H1 selected; v5 exploration not promoted.**
    Async matrix r2 completes all9 unique harnesses,24 selection tasks and
    two Solver samples per task in806.25s. Independent auditing checks all
    432 task/sample identities, paired decode seeds, binary rewards and4096
    completion-token caps. The frozen rule selects proposal04:
    83.33% versus68.75% no-patch on selection. Report remains unopened.
    `common/INITIAL_STATE.json` and `FINAL_MATRIX_AUDIT.json` persist the
    selected E15/W0/H1 and full historical prefix accounting. This selected
    difference is not a held-out report improvement.

    V5 completes48 native steps/40 nonzero Actor updates and releases node3.
    Final selection: W0=66.95%, step23=59.32%, step47=67.80%.
    The+0.85pp final candidate gain misses the+2pp promotion floor; retain W0.
    No trained-model report is opened. Its report0pp is a retained-W0
    identity comparison, not a negative evaluation of the trained model.
    Frozen-worktree audit confirms source identity, budgets, target range
    and release. Active native training is4562.918s; training including
    startup is4787.047s. Per-active-Actor averages:35.196s Actor,
    34.461s critic,11.586s rollout wait. Batch size and update budget differ
    from v4, so neither raw step times nor this joint experiment isolate
    the effect of6/2 allocation. The strong paired study keeps v4's recipe.

    Add persistent native phase execution with input hashes, exclusive lease,
    preserved bounded attempts, exact same-role checkpoint handoff and real
    native metric audits.55 tests/20 subtests pass. The new audit also passes
    against the completed real positive Engineer and Solver runs. Real-model
    CPU preflight verifies192 H2 rows,640 W1 rows, E15's11829-token maximum
    versus16384 capacity, and198 train-only centralized-context tokens versus
    the1024 allowance. These are execution checks, not new learning outcomes.

100. **2026-10-01 — strong arms executing and durable continuation wired.**
    Node4 starts sequential H2; node3 starts MA W1 after the completed v5
    exploration releases its GPUs. Both use the same published E15/W0/H1.
    The first three Actor updates are observed on both nodes, not inferred
    from loaded models. Sequential H2 subsequently completes32 steps/
    27 nonzero Actor updates with an exact native audit and release.
    MA W1 is observed at41 steps/36 nonzero Actor updates and continues.

    Add whole-arm state handoff, checkpoint/proposal selection and
    current-partner baseline refresh. The controller adopts the independently
    running first phase, does not retrain it, preserves finite retry histories
    and stops at a final freeze without reading report scores. Cached control
    reports have explicit model/harness/question/decode bindings. H2 retains
    all8 slots per model, including invalid/no-patch and duplicates. Fresh
    current-W packets use checked official greedy train0--239 evidence,
    weight digests and immutable raw baseline identities; all-pass calibration
    is retained without invented traces.46 tests/23 subtests pass.
    This is not yet a completed SAO-versus-MA result.

101. **2026-10-02 — native CTDE budget complete; report opening remains gated.**
    MA W1 completes80 native steps/75 nonzero Actor updates with bounded
    terminal targets and critic-only warmup isolation. Quality selection is
    still running. Sequential H2's matched generator retains6/8 valid
    incumbent proposals,4/8 at continuation step15 and0/8 at step31.
    This is real late-phase regression; do not hide it or reinterpret
    invalid/no-patch fallbacks as useful editing. Selection still follows
    the frozen rules and does not read report scores.

    Implement a separate frozen-report runner and final comparison. Both arm
    freeze hashes are mandatory before any report model access/allocation;
    each final Solver is crossed with no-patch/common-H1/both final harnesses.
    The MA reporter also supplies W0. Persist raw trajectories and report
    hashes, recompute mean@2 and paired question bootstrap, require both
    +5pp and positive95% lower bound, and label the interaction contrast
    without causal overclaiming.50 tests/23 subtests pass, including exact
    +5pp acceptance, a smaller positive gain rejection and a positive mean
    with nonpositive lower-bound rejection. The real current experiment
    also refuses premature report opening. No final arm result exists yet.

    Subsequent selection is now complete: sequential H2 mean proposal score
    incumbent73.4375%, step15/step31 both68.75%; retain incumbent E15 and
    H1 at83.33% native selection. MA W1 step39=75%, step79=77.0833% versus
    83.3333% control; reject both and retain W0. MA H2 therefore reuses
    unchanged-W0 evidence rather than claiming a partner refresh occurred.
    The arm controllers proceed to sequential W1 and MA H2.

    Both report units and the CPU-only finalizer are deployed as independent
    systemd services. They wait for both immutable arm freezes and both
    complete/released reports respectively. Final output is
    `runs/alfworld_strong_sao_ma_v1/PAIRED_REPORT.json`; no report was opened
    while wiring or testing the pipeline.

    Whole-phase pmon means: sequential H2 experiment25.95%, filler15.28%;
    MA W1 experiment43.87%, filler7.79%. Filler remained active.
    A consistent speed denominator also qualifies the v5 result:
    v4 active5811.107s/600 Actor episodes=9.685s versus
    v5 active4562.918s/480=9.506s (only1.85% lower). Per all optimizer
    episodes is9.080 versus7.922s, with different warmup proportions.
    Neither comparison isolates a6/2 allocation effect, and v5 did not
    pass its quality gate.

102. **2026-10-02 — preserve numerical evidence and resume without retraining.**
    Sequential W1 completed80 native/75 nonzero Actor updates. Its original
    strict target-range audit rejected one FP32 boundary residual at step26,
    `-3.725290298461914e-9`; the native training and release were successful.
    Add a shared absolute`1e-6` audit tolerance and retain raw extrema.
    Larger/nonfinite/reversed ranges still fail; no target clipping, reward
    change, optimizer change, or extra training occurs.

    The first audit-only CPU controller migration exposed a coupled legacy
    cache issue: recomputed audit metadata added two fields to the frozen
    H2 selection identity. Preserve this failure too. The corrected migration
    verifies all old fields exactly and records the extra extrema/tolerance
    separately, leaving the original identity unchanged.34 tests/21 subtests
    and a disposable-copy check of the real historical H2 selection pass.
    The immutable`55a5b31`controller is now node4 unit
    `coevolve-strong-sao-v1-sequential-arm-r2`; verified W1 metrics SHA
    `86d2c2990c09d15854064fff935af9785c915a96fa0b6be710c4f3f6f1c766e4`
    is unchanged, native attempts remain1, and step39 selection has launched.
    Both recovery receipts and prior failures remain in the arm directory.
    Native budgets must not be inflated by falsely describing this as retraining,
    but the controller recovery delay must remain in reserved GPU-time accounting.

    MA H2 independently completed32 native/27 nonzero Actor updates and entered
    matched selection; it was not stalled. Its active controller was not stopped.
    Both arms still need W2 and final selection; no final report has been opened.

    Before report opening, commit`7067440`also added an identity-aware final gate:
    identical final Solver weight and harness hashes cannot be algorithmic
    improvement, regardless of independently served score variation.
    Raw paired scores remain available. The CPU-only finalizer migrated to
    `coevolve-strong-sao-ma-v1-finalizer-r1`, with its original launch retained.
    Filler remained active and unchanged throughout both numerical recoveries.

103. **2026-10-02 — actual W1 collapse, retained incumbent and honest cost bounds.**
    Sequential W1 step39/79 both score0% on selection24 mean@2, with and
    without H1. All48 deployed-policy samples at each checkpoint reach the
    4096-token limit. This is genuine policy degradation, not an HTTP failure
    converted to a score and not caused by the later numerical-audit repair.
    Native mean task reward is70.83% across steps5-19, then0 across20-59;
    finite nonzero gradients do not establish stable learning.
    Both checkpoints are rejected, retaining the original W0/H1. Full W2
    begins from that incumbent with its frozen disjoint training offset/seed.
    MA W1 had75%/77.08% and also retained the incumbent. Do not describe
    any future MA comparison as beating a deliberately deployed0% baseline.

    MA H2 generation preserves8/8 valid incumbent proposals,8/8 at local
    step15 and2/8 at step31. Its first selection matrix completes16/20
    unique harnesses before a worker28116 transport`ReadError`propagates
    as HTTP502. The worker has no recorded fatal exception and subsequently
    shuts down gracefully; a later cancel request logs`ClientDisconnect`.
    Do not infer a model-quality failure from that transport incident.
    All completed reports and the failed attempt remain. The bounded systemd
    controller launches matrix attempt2 without retraining or regenerating
    Engineer proposals; no final H2 decision exists at this observation.

    Cost audit tooling counts actual saved generations, not just the input
    ledger: sequential H2 has198 receipts for192 inputs; W1 has648 for640.
    W1's2,156,802 recorded completion tokens include495 token-limit episodes,
    142 completed,7 malformed-tool and4 environment-invalid-action episodes.
    H2 adds116 distinct successful official reward jobs/1,160 task episodes.
    Its selection records576 physical native episodes plus24 Engineer
    generation responses; logical duplicate harness labels are not charged
    as extra inference. Failed attempts' saved work still counts.
    15 cost/report tests and8 subtests pass, and these real completed artifacts
    validate the receipt schemas. Complete per-arm cost publication remains
    gated on final freeze/report release. Lost in-flight work is an explicit
    lower-bound gap, never a zero-cost assumption; reserved node time starts
    at the original phase launch and includes all recovery and waiting.

104. **2026-10-02 — rejected MA H2 and noninterrupting audit handoff.**
    MA H2's second matrix completes with incumbent/step15/step31 proposal
    means71.614583%/71.354167%/69.791667%. Retain E15 and the common H1;
    MA W2 starts from W0/E15/H1, as does sequential W2. Neither arm has
    accepted a counterpart change yet, so no effective co-adaptation has
    been demonstrated. `W2_MATCHED_INPUT_AUDIT.json` verifies the actual
    matched640 rows, model/critic and harness contents,80 steps/batch8,
    seed1000042, learning rates and4:4 layout; only MA has central context.
    W1/W2 together have1,280 distinct input IDs. Input parity is not an
    outcome or a convergence result.

    Commit`42a521f`preserves legacy completed phase bytes as well as
    selection identities after revalidating native evidence. This matters
    for cached non-first MA H2 during a later controller recovery.
    45 tests/29 subtests and real MA H2 phase+selection temporary-copy
    checks pass;9 training/trajectory implementation files match1a9af61.
    At02:48 UTC, node3 queues`coevolve-strong-ma-v1-arm-r1`with
    `--after-unit coevolve-strong-ma-v1-arm.service`. The new CPU-only
    controller waits without interrupting original PID3820062 or its W2;
    live W2 subsequently has14 steps/9 nonzero Actor updates. An already
    frozen arm is a no-op; recovery retains the same inputs/attempt limits.
    The migration is recorded in`ma/DEFERRED_AUDIT_HANDOFF.json`.
    No active training, reward, target, optimizer or filler setting changed.

    Published common cost fragments charge the complete original Engineer
    window23.124939 GPUh and common-H1 selection/repair window12.825886
    GPUh:35.950825 GPUh per logical arm, physically paid once. Separate
    Solver-v5 and the strong-Solver train probe remain additional R&D.
    Final per-arm costs and the MA advantage gate still await W2, immutable
    freezes and both gated report releases.

105. **2026-10-02 04:11 UTC — final W block and README handoff.**
    Sequential W2 has completed80 native steps/75 nonzero Actor updates and
    entered checkpoint selection. MA W2 has79 recorded steps/74 nonzero Actor
    updates; its original native phase remains active. Neither arm has frozen
    and no final report exists at this observation. Completed native training
    must not be confused with a completed comparison or convergence.

    CPU-only cost services are already waiting for the respective reporters:
    node4`coevolve-strong-sao-v1-sequential-cost`and node3
    `coevolve-strong-ma-v1-cost`, both frozen42a521f. Their launch receipts are
    stored per arm and their waiting cgroups have no GPU processes. Each cost
    publication remains conditional on genuine freeze/report/release evidence.

    The root README now presents the actual9B research task, positive
    single-role gates, real fixed-H1 instability, active services, remaining
    validation and resource/provenance boundaries. The older2B RSFT/GRPO
    description is explicitly historical instead of misleadingly presenting
    itself as the current method. W selection uses the original+2pp point
    floor; its recorded Bonferroni LCB is diagnostic, not a promotion condition.
    This clarification changes no running gate or training setting.

106. **2026-10-02 04:41 UTC — sequential freeze and explicit MA infrastructure recovery.**
    Sequential W2 step39/79 score62.50%/56.25% under H1, versus83.33% for
    W0/H1; both are rejected. The sequential arm freezes its original W0/E15/H1.
    MA W2 also completes80 native steps/75 nonzero Actor updates, but its
    selection has no complete score yet. Two initial matrix launches fail with
    vLLM distributed-port EADDRINUSE; c0aa909 assigns independent per-replica
    internal ports. All eight recovered services reach HTTP200 health.

    The next attempt encounters a different error: worker cancellation's
   30-second forwarding timeout returns502, the episode adapter fails to check
    that status, and the replacement episode receives406 from the occupied
    worker. a4eef56 uses the official asynchronous cancellation-notice protocol
    and waits for actual completion before returning the client's episode slot.
    Cleanup HTTP errors are explicit; an earlier generation error remains primary.
    51 tests/28 subtests pass. A live CPU-only test starts and cancels128 official
    worker sessions and verifies zero remaining sessions, with zero model calls.

    Separate immutable STARTUP_RECOVERY and CANCELLATION_RECOVERY authorizations
    bind the original requests, failure logs and releases, granting one extra
    infrastructure attempt each. The automatic max2 has therefore been extended
    by two explicit recoveries; do not claim unchanged retry counts or free work.
    Native training is still one completed attempt, and task/model/seed/gates are
    unchanged. Current MA unit is coevolve-strong-ma-v1-arm-r3, sourcea4eef56.
    Still-waiting reporters and cost jobs also migrate to that source under
    their respective -r2 units before any report opens; original launches remain.

    The README now reports complete system configurations in separate official
   500-task, local91-task and local70-task columns. Official ALFWorld results are
   40.6 vanilla,53.2 vanilla+Harness-R1,71.2 Agent-SFT,84.0 Agent-SFT+the matching
   Engineer. Local SAO-W is78.02->89.56; local no-patch->SAO-H is75.71->80.00.
    The72.14->80.00 matched-Engineer diagnostic is separate. E0 is the released,
   already-trained vanilla-target harness-r1, not SFT-only; the target-specific
   agent-sft-harness-r1 has not been locally evaluated. No direct superiority over
   the paper's GRPO/84.0% result is claimed.

## 2026-10-02: alternating E15/W79 campaign and joint-round correction

The autonomous8h campaign started05:22:03 UTC and completed its available
training/reporting at09:33:45. Ordinary SAO trained H2 for32 native/27 Actor
updates and selected step31, but W2's80/75 updates produced selection scores
81.25%/75.00% at step39/79 against incumbent W79+H2 at97.92%. Both new Solver
checkpoints were rejected. The declared final70x2 result is W79+H2 **96.43%**.
A separate matched W79 matrix gave no-patch91.43%,H1 95%,H2 97.86%;
H2-H1 +2.86pp, paired question95% interval[-1.43,+7.14]pp.
These serving runs differ and the larger number is not substituted for the
declared final result. H2 valid proposals were7/8 initially,1/8 at15 and3/8
at31; invalid slots count as no-patch. W2 had no completion-cap hits in selection.

MA never reached native training: both attempts failed Git ownership checks
after Launcher changed HOME. The reported95% is the untrained H1/W1 fallback,
not MA performance. Budgets are unmatched. Per-command trust of exact resolved
SAO/AReaL paths fixes the cause independently of HOME; the actual node3 child
environment was retested successfully.

At14:21 UTC the user prioritized a genuine MAPPO-style joint first result.
Source1b2fd3f,26 passing tests+3 subtests, was pushed before deployment.
At14:31 UTC persistent node3 Engineer/node4 Solver services and an automatic
waiting evaluator were launched. Pilot:16 rounds,batch8,4 warmup,12 Actor updates
per role;128 distinct training tasks,30 failures under historical W79 evidence.
Both native optimizers persist. Every sampled harness and current-Solver outcome
is paired by row/slot/version, with authentic per-role logprobs and one shared
terminal reward. Dynamic pre-action privileged context is critic-only.
Zero staleness deliberately avoids pretending unsupported stale joint correction.
MAPPO ValueNorm is not implemented; native SAO value machinery remains.
At this entry's write time GPU initialization was live, first updates/results
still required verification. No improvement or full MAPPO parity is claimed.

The first joint deployment subsequently failed before any update: a nested CPU
environment Launcher's resource watchdog misclassified its sibling native
learner's GPUs as foreign. dd543db shares parent process ownership and preserves
the original failure atomically during concurrent peer cancellation;28 tests
and3 subtests pass. A fresh immutable r2 kept all data/settings unchanged.
At14:42 UTC the user prioritized a matched core table; r2 was explicitly stopped
with0 native updates and0 paired outcomes, preserving both attempts and filler.

## 2026-10-02: highest-priority matched core quick table

Node4 sequentially evaluated W0 and W1; node3 generated8 valid target-specific
official Engineer proposals and evaluates them beside the8 frozen E15 proposals.
Output: `runs/alfworld_core_quick_20261002_v1`. Native24-task mean@2,
temperature1,total4096 completion tokens,seed6000045, same frozen H1 for both
Solver weights. No new training or multi-seed sweep.

Node4 finished at14:56 UTC: W0 **68.75%**, W0+H1 **75.00%**, W1 **87.50%**,
W1+H1 **91.67%**, W1+H2 **97.92%**. Pure-W gain+18.75pp has question95%
interval[+4.17,+35.42]pp; H1 overW0+6.25pp has[0,+12.50]pp; H2 overH1
onW1+6.25pp also has[0,+12.50]pp. These reused selection scores are exploratory,
not fresh report-set significance. All native sample identities match.

The official matrix remained active when this entry was written. d2e306a reuses
the native evaluator, verifies the released target-specific checkpoint, matches
proposal inputs/seeds and distinguishes mean over8 from best-of8.2 tests and
2 subtests pass. No official final score is inferred from partial candidates.

All8 required official candidates subsequently completed and were independently
validated before waiting for the extra E15 diagnostics. Scores by slot:
77.08%,83.33%,81.25%,70.83%,79.17%,85.42%,85.42%,79.17%.
The mean is**80.21%**, best**85.42%** (slot05, deterministic tie break).
Official-matrix W0 and frozen-H1 controls exactly repeat node4's68.75%/75%.
`OFFICIAL_CORE_READY.json` records the complete required8-candidate evidence,
not a false claim that remaining matrix diagnostics had already finished.
`CORE_TABLE.json` completes all six requested system rows, exact sample/seed/token
validation and source hashes. W1+H2 minus official best is**+12.50pp**,
question-bootstrap95% interval[+2.08,+25.00]pp; reused selection/search exposure
precludes treating this as independent significance or training-budget parity.

At15:08 UTC the remaining E15 diagnostics completed and GPU release was verified.
With the same8 historical W79 evidence prompts, proposal seeds/budgets and W0
native24-task evaluator, E15/H1 scores**69.53% proposal mean**,**81.25% best-of8**,
7/8 valid, versus target-specific official**80.21%/85.42%**,8/8 valid.
Thus even granting both Engineers8-candidate selection, H1 does not outperform
the official release in this quick comparison. All18 logical variants passed
complete task/sample/seed/token checks; `ENGINEER_MATCHED_DIAGNOSTIC.json` persists
the result. The previously reported six fixed-system rows and older70-task
greedy/per-turn protocol are unchanged.

## 2026-10-02 16:04 UTC: expanded untrained one-shot core table

The user requested175 additional tasks to reach200, but the existing cohort was
24, not25. Declared176 new tasks and reused sample_index0 from each old task,
never the better of two samples. All six models/harnesses were frozen before
new outcomes. Native temperature1,total4096 completion tokens,one sample.

The500-task catalogue contains197 game instances whose goal groups exclude all
legacy0-239 training tasks; these197 cover179 distinct goals. To reach200 without
repeating any locally trained game, three deterministic unseen instances
291/414/467 with related training goals were transparently included and a strict197
subset reported separately. The entire2296-game training-only catalogue is also
instance/goal-disjoint from the official catalogue.83 previously sealed IDs were
opened under this request. Existing development exposure and unknown official
pretraining exposure remain limitations.

First v1 W1 and W0 matrices failed after445.8s and570.9s with worker interact
ReadError->HTTP502, without worker exception logs or a complete variant report.
An identical W1 retry was stopped during startup after the second node exhibited
the same symptom. No failed/incomplete evaluation was converted to reward0.
57942d1 prevents worker TCP reuse and logs the session ID on transport errors.
A real socket fixture fails on a reused connection but passes four fresh requests;
no ambiguous state-changing request is retried.18 tests+6 subtests pass.
New immutable v2 retains all task/model/harness/seed/token/concurrency settings and
all prior failure hashes. Its full native evaluation completed, with both GPU
release audits verified. An early CPU aggregation launch/dependency and directory
readability issue was recorded and corrected without restarting GPU work.

| System | 200-task mean@1 | Strict197 | Newly evaluated176 |
|---|---:|---:|---:|
| W0 | 147/200 =73.50% | 144/197 =73.10% | 130/176 =73.86% |
| W0+official target-specific Harness-R1 | 166/200 =83.00% | 163/197 =82.74% | 146/176 =82.95% |
| W0+fixed H1 | 148/200 =74.00% | 146/197 =74.11% | 131/176 =74.43% |
| W1 | 174/200 =87.00% | 172/197 =87.31% | 154/176 =87.50% |
| W1+H1 | 181/200 =90.50% | 178/197 =90.36% | 158/176 =89.77% |
| W1+H2 | 184/200 =92.00% | 181/197 =91.88% | 161/176 =91.48% |

Pure-W gain+13.50pp, question95% interval[+6.50,+20.50]pp; fixedH1 onW0
only+0.50pp,[-4.50,+5.50]. H2 overH1 onW1+1.50pp,[-1.50,+5.00],
not established as a stable advantage. Best system minus official+9.00pp,
[+3.50,+15.00], but these question intervals do not additionally cluster repeated
goals or correct historical selection/search. The retained24 are not independent
report data. Final output `runs/alfworld_core_200_oneshot_v2/COMPLETE.json`;
all200 unique sample0 identities, seeds,tokens,input hashes and strict197 counts
verified. Completed16:04:13 UTC,30.5min from first launch including failures,
16.6min for the successful recovery round. The initial15min target was missed.

## 2026-10-02 16:48 UTC: actual final W2 under frozen h2

The user requested the rejected W2's200-task score. Freeze actual localstep79
from the sequentialW2 phase (80native,5warmup,75Actor), not the retained W1.
Its weightSHA256 is46fa776d8652188aba42f54212a9c7884db91ab42785fe1fbbc49bb11b018ea9,
distinct from W1. Keep h2 digest4b7f50de842f638d862e81deae70566c71deb2a5c1a090a54e22fe825c11e174.
Two100-task shards re-evaluate all200 at mean@1,temperature1,total4096tokens,
seed6000045. Source57942d1 uses fresh worker TCP connections. Bothshards finish
with no native failure; exact coverage/seeds/token budgets/source hashes and GPU
release verified. Runtime11.02min.

W2 no-patch100/200=50%; W2+h2 127/200=63.5%.
Strict197:100/197=50.76% and126/197=63.96%.
Against the frozen W1+h2 reference184/200=92%, W2 loses28.5pp, question95%
interval[-35.5,-21.5]pp;5 tasks improve,62 regress. Strict197 difference-27.92pp,
[-35.03,-20.81]pp. The gate's rejection was warranted, not a completed successful
alternating improvement. Root cause remains unresolved.
Output: `runs/alfworld_w2_h2_200_v1/COMPLETE.json`.

Native H1 accounting was also checked:32steps,192 consumedEngineer trajectories,
30critic-only and162Actor; selectedstep15 contains96/66 respectively.
The official README example NUM_ROLLOUT174,rolloutbatch4,samplesperprompt8 implies
5568nominal generated trajectories. This is not an audited release run budget or
a29x efficiency result: SAO continuation inherits the already-RL-trained official
Engineer, budgets/outcomes differ, and target reruns/critic/search/prefetch matter.
The pinned fooSynaptic SAO implementation explicitly declares itself unofficial;
its underlying AReaL revision and our integration changes are documented separately.

## 2026-10-02 21:31 UTC: matched192 GRPO completed; joint MA initialization

Frozen GRPO source3ebdcfa, run `runs/alfworld_matched_grpo_192_v4`,
completed six committed32-trajectory bundles and24 optimizer minibatches.
Generated=consumed=192; no final extra training generation. All four ranks'
gradients were finite and nonzero in all commits. Final checkpoint000006 actor
identity is8f2c62db0003175e275b7a957543478ee667fc7008322f4492af40e7a6e6f0d9.
The96-trajectory audit checkpoint was retained but not selected by evaluation.
Earlier import/TP2-allreduce/TP1-TCPStore failures are retained; none completed a
training rollout or optimizer commit. Final training verified20:57UTC, later than
the original estimate because full FP32 checkpoint export/verification dominated.

Evaluation `runs/alfworld_matched_grpo_200_v3` completed21:30:47UTC.
Eight proposals, seven valid. Slots00/07 both scored46/48 on W1 selection24;
the predeclared lowest-slot tie break selected00. Its patch digest is
fd04c0f9147d0e03364ffab2c5cf2f4d982e2d528fdd5717da7c6bc587f94dde.
The identical frozen patch was deployed to W0 and W1.

| System | all200 mean@1 | strict197 |
|---|---:|---:|
| W0 no-patch, fresh rerun | 150/200 =75.00% | 147/197 =74.62% |
| W0 + matched GRPO h1 | 167/200 =83.50% | 165/197 =83.76% |
| W1 no-patch, fresh rerun | 176/200 =88.00% | 174/197 =88.32% |
| W1 + matched GRPO h1 | 176/200 =88.00% | 174/197 =88.32% |

Against historical SAO h1, W0 gains9.5pp, question-level paired95% interval
[5,14]pp; W1 loses2.5pp,[-6.5,1.5]pp. Against W1+h2=92%, the new W1 system
loses4pp,[-8.5,0]pp. This does not establish general algorithm superiority.
The selection95.83% did not persist on200. Fresh no-patch scores differ from old
73.5%/87%; fixed cohort/seeds do not imply bitwise deterministic serving.
Reports have verified hashes,200 unique task/sample identities, declared seeds
and token budgets, and both GPU release audits. Same192 training trajectories is
not matched tokens, FLOPs, critic, reward-environment or selection cost.
Selection24 is included in development200; strict197 does not remove tuning
exposure. SAO selected step15 whereas GRPO used its predeclared final point.

Immediately after comparison completion, persistent MA services launched native
learners on node3/editor and node4/solver, PIDs294695/1939420 at21:31UTC.
Actual frozen source43c3de0 is bound by `LAUNCH_v4.json` under
`runs/alfworld_joint_async_6h_v1`; prior waiting queue versions consumed no MA
samples. Initialization is not a verified joint update or a performance result.
The22:41:31 training cutoff and23:41:31 reporting deadline remain unchanged.

## 2026-10-02 22:46 UTC: genuine joint updates, then OOM and checkpoint-only recovery

During initial native validation, three concrete correctness issues were found:
independent per-DP-shard ValueNorm moments; raw vLLM logprobs excluding the
Engineer's0.7 temperature; and HF Qwen3.5 recurrent-state leakage between
packed trajectories. Fixes4b850a7/b028cad/f3563e6 gather full-DP active targets,
request processed sampling logprobs, and allocate one trajectory per microbatch.
The effective batch remains8 per role. Real two-process tests reproduced and
fixed the normalization mismatch. These implementation differences prevent an
algorithm-only comparison against historical single-role runs without reruns.
A separate gitlink-bootstrap failure occurred before any model loading.
All attempts were retained, not relabelled as successful training.

Final corrected training `runs/alfworld_joint_async_6h_v5` started22:14:20UTC,
sourcef3563e6. Initial behavior/proximal consistency passed for all16 trajectories:
maximum absolute token-mean log-ratio H0.001488/W0.007527nats, explicit engineering
tolerance0.01. Same paired IDs/rewards/versions/slots verified. Across six committed
rounds,40/48 trajectories in each role actually exhibited the permitted lag1.
Both roles completed6 native steps, including2 critic-only warmups and4 nonzero
Actor updates. K=2 was retained. Healthy KL/value losses are not validation scores.

At22:33:19 Solver actor backward OOMed on a24075-token trajectory. The allocation
needed11.25GiB with6.04GiB free; the colocated reference had8.80GiB allocated but
40.40GiB reserved. The first joint failure receipt reflected peer cancellation;
the actual CUDA cause remains explicit in `solver/optimizer/training_active.json`
and the native actor traceback. No infrastructure exception became a reward0.

Only common globalstep3/round4 had complete actor/critic files and committed
metrics before failure. Its actors each contain2 updates. Real projection tensors
changed447542(H)/443087(W) of33554432 inspected parameters, and both critic
normalizers recorded8 updates. No partially saved/later in-memory model was used.
The separate `runs/alfworld_joint_async_6h_v5_recovered` publishes these models
with explicit recovered-after-failure status and evaluates eight new proposals,
selection24 mean@2, then frozen all200 mean@1/strict197. Round12 is unavailable.
Node4 best/guard and node3 unavailable-matched receipts are persistent.

Actual corrected-run consumption is96 H+W trajectories, not the selected point's64.
Previous failed attempts consumed another72 committed trajectories. Generated
policy actions across attempts are at least260, including unrewarded H actions;
incomplete W work and proposal-selection/evaluation costs are additional.
No claim of full-search matched budget is possible.

Main5d3c0b9 separately fixes typed RPC config roundtripping; running source was
not mutated. The reference-only cache-release correction passes41 tests+3 subtests.
A real node3 CUDA check freed8589934592bytes while preserving live tensors and
16384allocated bytes. This validates cache release, not a full long-context PPO
retry. The failed run remains failed, and the W1+h2 incumbent remains unchanged.

## 2026-10-02 23:07 UTC: final recovered MA report, target not achieved

Best-arm evaluation finished23:07:27UTC, about34minutes before the23:41:31 deadline.
The evaluated pair is committed globalstep3/round4, with2 Actor updates per role,
not the last in-memory round6 state. Eight fresh Engineer proposals had6 valid
patches. Slots01/02/03 tied45/48 on selection24 mean@2; the predefined smallest
slot rule froze01 before all200. No candidate was replaced using the200 scores.

| System | all200 mean@1 | strict197 |
|---|---:|---:|
| MA round4 Solver, no harness | 172/200 = 86.00% | 170/197 = 86.29% |
| MA round4 Solver + new Engineer harness | 180/200 = 90.00% | 177/197 = 89.85% |
| Retained historical W1+h2 | 184/200 = 92.00% | 181/197 = 91.88% |

MA minus W1+h2:-2.0pp, paired question-bootstrap95% interval[-5.5,+1.5]pp,
5 improved/9 regressed questions. Adding the new patch to the same new Solver:
+4.0pp[-0.5,+8.5],16 improved/8 regressed. Neither comparison establishes an
advantage. This short run does not establish that MA is intrinsically worse;
the primary performance goal was nevertheless not achieved, and matched round12
is unavailable. OOM is a training infrastructure failure, not proof of policy
collapse. Healthy losses did not guarantee either memory safety or higher scores.

The compact report hashes, native report equality,200 unique single samples,
exact seeds/token limit, training exclusion, strict197 subsets, checkpoint and
patch bindings, frozen selection and paired intervals were independently checked.
Evidence is `runs/alfworld_joint_async_6h_v5_recovered/FINAL_AUDIT.json`;
`best_COMPLETE.json` is completed while `DEADLINE_REPORT.json` remains incomplete
because the matched arm is unavailable. Neither the original FAILED records nor
the original release snapshot was rewritten. The latter briefly contained an
unverified PID2016329, which had exited by final audit; no manual kill was used.
Both nodes finally had only verified filler GPU processes, with filler active.

The selected point consumed64 new H+W trajectories; the corrected run consumed96.
Earlier failed attempts consumed72 more, for168 known committed MA trajectories
and at least260 generated policy actions across attempts. Adding the832 common
prefix yields1000 known committed trajectories, not total research cost:
uncommitted work, proposal selection and evaluation remain additional.

The prior matched GRPO conclusion also stands: W0+new h1=83.50% and
W1+new h1=88.00%, versus historical SAO h1 at74.00% and90.50%, respectively.
This is not a universally better Engineer. All200 is a reused development cohort;
strict197 excludes training goal-family overlap, not selection exposure or
unknown official-pretraining exposure. No training-seed robustness is claimed.

The live W1+h2 incumbent is unchanged. Code837a67e releases unused frozen-reference
cache and passed41 tests+3 subtests plus a real8GiB CUDA release check, but was not
hot-applied and has not passed a full long-context PPO rerun. Future work must
first validate that recovery and preserve honest full-budget accounting.

## 2026-10-03: independent H0/W0 MA preparation and actual initial scores

The user requested independent MA training from the original H0/W0 rather than
inheriting SAO H1/W1. Frozen preparation415fd64 resets Actor/critic/reference
paths to the official original models and uses only W0 historical evidence from
legacy training instances0-239. Existing trained weights, ValueNorm and optimizer
state are not inherited. H0 is already official RL, not a bare base model.
The new root is `runs/alfworld_joint_h0w0_8h_v1`; model hashes are frozen in
`INITIAL_MODEL_IDENTITIES.json`. No common832-sample prefix is charged.
Matched round64 means1024 H+W consumed trajectories, while round12 and52 retain
192/832 checkpoints. Five critic-only warmup rounds, K2, LR5e-7/2.5e-6 and KL0.005
are explicit. The report deadline is08:49:31UTC with the final hour reserved.

The actual pretraining evaluation generated8 H0 proposals using W0-only evidence.
Seven were valid. Selection24 chose slot01 at40/48; the patch was frozen before
all200 mean@1. W0 without harness scored148/200=74.00% (strict197146/197=74.11%).
W0 with that original-H0 patch scored170/200=85.00% (168/197=85.28%).
The paired gain is11pp[6,16],26 improved/4 regressed questions. This is not the
historical official counterpart Engineer score, and not a trained H1/W1 score.
The same reused-development and question-only uncertainty limitations apply.

`INITIAL_AUDIT.json` verifies original model bindings, zero native training
consumption, frozen proposal identity, native/compact report equality, hashes,
coverage/seeds/token caps and node4 release. Actual learning remains gated on
full-context native optimizer memory validation; synthetic engineering steps
must not be presented as MA research training.

## 2026-10-03 02:05 UTC: original H0/W0 first real joint updates verified

The native long-context memory checks exposed a second boundary: after Adam
state warmed, learner-held unused cache could OOM the colocated reference.
Joint PPO learners now release unused allocator cache after updates, complementing
the existing frozen-reference release. W0 passed two real32768-token Actor/ref
and33792-token critic updates; H0 passed K2 with one critic-only warmup, two Actor
and six critic updates. Reference outputs were bitwise unchanged. Neither test
saved model files. Five full-context warmup rounds were not claimed as tested.
Regression source, exact configuration and node3 release are recorded in
`runs/alfworld_joint_memory_20261003/`; integrated tests passed38+3subtests.

Production sourceb481fb0 was frozen separately from initial-evaluation415fd64.
Both real learners then started under persistent services. By02:05 both had
committed six native steps: five actual critic-only rounds, followed by one Actor
update. Each consumed48 real role trajectories, total96; K2 gave12 critic updates.
Actor gradients were H2.5064785/W0.5334495, finite and nonzero.
Each role's first48 trajectories included40 with actual behavior lag1.
Initial action token-mean log-ratio maxima were H0.0008099/W0.00022165, below the
explicit0.01 engineering threshold; IDs, slots, versions and shared rewards matched.

`FIRST_ACTOR_AUDIT.json`, `INITIAL_POLICY_CHECK.json` and `TRAINING_LAUNCH.json`
preserve this evidence. No post-training evaluation exists at this point;
healthy initial updates are not evidence that the85% initial system improved.
The original W1+h2 incumbent remains untouched. Independent matched/best services
and the deadline guard continue without the agent; filler remains enabled.

## 2026-10-03 02:14 UTC: research-driven follow-up requests

Three questions are now separated: recovery of rejected W2, a WHALE ALFWorld
baseline, and stability of joint versus alternating updates. Current independent
H0/W0 MA is not interrupted. Historical W2+h2=63.5% is a recovery starting point,
not a successful model to substitute for W1+h2.

The fixed-h2 recovery protocol starts at actual W2 globalstep79 and its critic.
It uses640 previously unused train-only instances1280-1919, preserving LR2e-6,
critic LR1e-5, K1, five critic warmup steps and no KL. Original optimizer moments
were not retained, so this is weight continuation with fresh Adam, not exact
optimizer-state resumption. A singleton-microbatch repair prevents the known
Qwen3.5 recurrent-state leakage; this backend change is an explicit confound.
Remeasure the original W2 with/without h2, then report every prespecified extra
20/40/60/80-step checkpoint on all200/strict197. The final80-step difference
versus the fresh initial score is the primary endpoint; intermediate comparisons
are exploratory, not a search for a favorable reporting checkpoint. No automatic
promotion occurs. No recovery in this finite budget cannot establish irreversibility.

Official WHALE documentation lists SearchQA, AIME and chess, not ALFWorld.
The official method is online RSFT alternated with Meta-Harness search, not
Harness-R1 or SAO under a different label. A local ALFWorld adaptation must disclose
proposer and harness action-space changes, all sampling/selection costs, and
separate CPU preparation from GPU validation and measured performance.

A scheduling-only control is being prepared within MA: H16-round blocks alternate
with W16-round blocks while holding critic/reward/normalization/KL/update settings
fixed. This is an alternating-MA ablation, not ordinary SAO. At128 collection rounds
it performs the same per-role64 active batches/512 training trajectories as
simultaneous MA at64 rounds, but inactive counterpart generation adds real cost.
Repeated sampling, different policy trajectories and additional environment work
must be reported rather than called perfectly matched compute. A full corrected-
backend SAO comparison is distinct from this ablation and from historical SAO scores.
No finite single-seed experiment can establish that MA never collapses.

## 2026-10-03 06:02 UTC: full matched point saved and research queues activated

Original H0/W0 MA reached91 native rounds per role without a new OOM. The complete
round64 Actor/critic pair is persisted and verified in `MATCHED_POINT_AUDIT.json`:
512 consumed trajectories,59 Actor and128 critic updates per role. This is the
1024-total matching point with no inherited SAO prefix; no post-training score
has yet been claimed.

W2 recovery is queued on node3 after current MA matched evaluation. WHALE v6 is
queued on node4 after current MA best evaluation, then requires a full32768-token,
4096-action warm-Adam/export/restore smoke before its research run. Its production
freeze is source-bound tobe92302 and actually validated from controls-v2 snapshot.
Smoke/queue failure explicitly publishes a pre-execution failure for downstream
consumers, never a fake baseline completion.

Matched-population ordinary SAO v4 is queued behind W2 on node3 and full WHALE
execution on node4. H1 and H2 use fixed finalstep31; W1 uses79. Its first640 W task
records exactly match current MA order, with0-239 legacy training population;
this is not exact historical-curriculum or E15-selection reproduction.
After full repaired SAO, the two-node scheduling-only MA v2 ablation is queued.
Both controls use frozen sourcea05800e and were validated on the real node UIDs.
The ablation reuses the exact physical H0/W0 initial evaluation with an explicit
reuse receipt and zero added training cost, not a fabricated fresh measurement.

Integrated controls passed172 tests+38subtests. Later queue-failure and upstream-
link fixes passed focused76 and82-test selections. A production SAO CPU preparation
initially rejected its intentional hash/commit-pinned AReaL symlink; no GPU or
sampling occurred. The failure is retained, not confused with policy collapse.
Strict local-source checks still reject mutable-main escapes. Presence of the
venv's main-path fallback alone did not demonstrate leakage in the active MA.

Because best-arm reporting may select among three checkpoint pairs, an additional
persistent07:10UTC stop request reserves approximately100minutes before08:49:31.
The original07:49:31 latest bound remains; round64 is already saved. This change
uses execution-time estimates, not post-training development scores. All original
models and incumbent W1+h2 remain unchanged; subsequent baseline/control queues
have their own bounded lifetimes and do not all fit the initial MA report window.

## 2026-10-03 08:12 UTC: independent MA completed, late system declined

The actual1024-trajectory round64 pair scored175/200=87.5% (strict197172/197),
against original H0/W0 at170/200=85%. The paired difference is+2.5pp[-2,+7].
Its Solver without a harness scored157/200=78.5%.
Later selection24 chose round112/proposal04, scoring165/200=82.5%
(strict197163/197), while the same Solver without a harness scored161/200=80.5%.
The late selected system is-5pp[-10,0] versus round64, and-9.5pp[-15.5,-3.5]
versus historical W1+h2. It did not beat the incumbent.

Importantly, the late empty-harness Solver is+6.5pp[1.5,11.5] versus original W0.
The harness addition is9pp[4,14.5] at round64 but only2pp[-3,7] late.
Thus system degradation should not automatically be attributed to collapsing
Solver weights or alternating updates: this run was joint, and its Solver
point estimates improved. Harness quality/selection/compatibility are concrete
next diagnostic questions, with no seed-robust causal claim yet.

The late candidate means on selection24 were75.78125%,73.69792%,60.15625% for
rounds104/108/112. The max-candidate rule nevertheless chose round112 because
one proposal had the highest score. This suggests a selection risk but is not
an Engineer-only comparison: Solver weights also differ. No200-score candidate
replacement occurred. The report preserves all selected system scores and
paired intervals; development selection and repeated-question limitations remain.

Training sourceb481fb0 completed115 committed rounds per role, each with110 Actor
and230 critic updates;1840 total consumed and1872 complete generated role
trajectories. The selected late path consumed1792, not the whole run cost.
No OOM occurred; training stopped by the planned reserve. Both reports and model/
patch/seed/token/sample contracts are verified in `FINAL_AUDIT.json`.
The deadline report is completed, but algorithmic superiority was not achieved.

W2 recoveryv1 then failed before any episode because the evaluator tried to
rewrite metadata in a shared read-only original checkpoint. Repair2ed6700
uses existing `supplement_runtime`, with a regression and real node3 UID check;
v2 preserves the original failed attempt and now proceeds from the true W2.
WHALEv6 failed during the second full-context synthetic GPU smoke process,
before baseline training. Its failure correctly blocked downstream SAO.
Dependent unstarted queues were paused for explicit recovery; no failed
infrastructure episode was scored as a model failure or silently retried.


## 2026-10-04: MA improvement retry after nullable tool-call failure

The latest Copilot objective remains MA-SAO > historical W1+h2=92%; WHALE is
paused. Cross-play is complete: MA64 no/h0/h64/h112 =77.5/90/89.5/78.5%,
MA112 =79.5/86.5/87.5/83.5%. These are reused-development diagnostics, not
replacement scores for the original87.5/82.5% MA reports or an untouched test.

Improvement v1 failed after eight committed rounds per role: task190, round9
reached an AgentRL Pydantic error because a text-only completion included
`tool_calls: null`. The worker traceback, not an inferred OOM, establishes the
cause. Omit that optional null field at the local adapter boundary; preserve
text, tokens, worker judgment and propagation of genuine HTTP failures.
Validation:70 tests+30 subtests; real CPU worker using the failing task and
patch accepted synthetic text-only output, then scored token exhaustion zero.
Receipt: runs/alfworld_ma_null_calls_smoke_20261004/RESULT.json. No model score
is claimed for this one environment episode.

Runs/alfworld_ma_improve_v2 restarts the same H0/MA64 initialization with fresh
Adam, retaining the64-round cap, round32 probe, guarded h0 candidate, low H/W
learning rates and KL. It is not exact optimizer continuation. Failed v1 cost:
128 consumed H+W trajectories,158 complete generated role trajectories
(overlapping counts), retained separately from parent's1840 actual/1024 path.
The old run and saved weights remain intact. The source snapshot is based on
0bbb1cf plus the tested boundary patch; SOURCE_MANIFEST.json pins file hashes.
Node3/editor+matched and node4/solver+best+guard started as persistent units
coevolve-ma-improve-v2-*. Both nodes passed release/filler audits beforehand.
Report deadline:2026-10-04 16:31:05 UTC. Launch is not a performance result;
no post-training score or superiority claim exists at this entry.


## 2026-10-04: research questions and prospective diagnosis of guarded MA

**RQ1 — What caused the late MA system decline?** Existing fixed-weight cross-play
finds MA64+h64=89.5% versus MA64+h112=78.5%, a -11pp paired difference
[-16,-6]pp. This supports harness degradation/selection/compatibility as a
concrete failure mode. It does not establish that Solver changes are harmless:
MA112+h0=86.5% versus MA64+h0=90%. A joint update alone does not prevent decline.

**RQ2 — Can constrained co-adaptation preserve a strong harness and improve the
Solver enough to beat W1+h2?** The running experiment is the guarded MA package:
H0 Actor + MA64 Solver/critics, fresh Adam, H/W LR1e-7/1e-6, KL0.02, K2,
three critic warmup rounds, simultaneous updates/lag1. Retain h0 unless a new
patch gains at least2/48 selection questions.64 new rounds, saved every8;
round32 is secondary, the selection48-chosen best-arm endpoint is primary.
Do not pick between these endpoints after reading all200 results. This tests
the package; LR, KL and selection changes are not individually identified.

**RQ3 — If the package succeeds or fails, which component explains the change?**
Before either campaign endpoint exists, freeze
`runs/alfworld_ma_improvement_study_v1/protocol.json`. After both training and
existing evaluations release GPUs, run a fixed2x2 matrix: branch-start/selected
Solver crossed with initial h0/selection-chosen patch, plus empty-harness
controls. Remeasure W1+h2 with the same runtime, seed, token and question
contract. Maximum1600 environment episodes, zero new training; identical
patches share physical evaluations. Node3 unit`coevolve-ma-improvement-study-v1`
is persistently queued; its source snapshot is separate from running training.
Runner and paired-analysis regression tests:12 passed.

Estimate weight gain at fixed h0, patch gain at fixed selected weights, and
interaction=(patch-h0 at selected weights)-(patch-h0 at starting weights).
Report question-paired95% intervals; empty intervals of evidence are not
claims of no effect. Publish all200, selection48 and the disjoint152 complement.
The complement was exposed in earlier work and is NOT an untouched test set.
Fresh baseline comparison prevents interpreting small service/retest shifts as
algorithmic superiority. The primary claim remains the original frozen endpoint,
with diagnostic remeasurement reported alongside, never silently substituted.

**Decision rules after results.** If the primary exceeds92% and the fresh paired
comparison supports improvement, the next question is robustness/cost under
independent seeds and truly unused tasks. If fixed-h0 weights regress, prioritize
a fixed-h0 Solver-only ablation with identical update budget before attributing
the failure to Engineer search. If weights improve but selected patches regress
outside selection48, prioritize selection generalization and paired marginal
harness benefit. If neither component shows a supported gain, do not claim that
joint MA has improved. A new training experiment needs a stated hypothesis and
frozen comparison, not only another learning-rate change.


### Additional mechanism probes before v2 endpoint results

First32-round training receipt audit: old MA31/256 invalid proposals,21 of those
received team success; guarded v2 24/256 invalid,17 received team success.
`joint_workflow` falls back to no patch and sends the same binary environmental
reward to both roles. This demonstrates that terminal reward alone does not
identify helpful Engineer contribution. It is not proof of positive PPO advantage
or a causal explanation of degradation. Different Solver initializations confound
between-run rate comparisons. A paired marginal-benefit objective remains a
future hypothesis, not an implemented/tested improvement.

Historical matched-weight cross-play:18/25 h64→h112 regressions are clean/heat/
cool tasks.14/25 regressed native traces contain the late wrong-object blocker.
A CPU hook probe shows h112 suggests setting aside a distractor but blocks its
placement even away from the goal; h0/h64 do not block this synthetic recovery.
Probe states are synthetic; no performance gain is inferred.

RQ4: Does the action guard itself contribute to h112's performance loss?
Freeze `runs/alfworld_h112_guard_ablation_v1`: fixed MA64 Solver, original h112
versus a copy with exactly the single on_before_action hook removed. All other
patch fields/actions unchanged, same200 tasks/seed/token contract, empty-harness
control,600 environment episodes and0 training trajectories. Run on node4 after
the current campaign releases GPUs. This is a post-hoc mechanistic intervention,
not a learned MA candidate or a baseline superiority claim. Pairwise95% interval
must support the effect; removing a whole hook cannot isolate one internal
condition. Independent source snapshot retains original frozen training files.
Artifacts: TRAINING_SIGNAL_DIAGNOSTIC.json, HISTORICAL_FAMILY_DIAGNOSTIC.json,
HOOK_RECOVERY_PROBE.json and HISTORICAL_REGRESSION_TRACES.json under the
improvement study root. These are supporting diagnostics, not new final scores.


### Partial-training recovery amendment, before endpoint scores

The guarded run generated512 trajectories per role but committed only H62/W63
updates (496+504=1000 consumed). A repeated version0/sample00003 request after
the final environment round failed with ConnectError. Native cyclic data loading
and overlapping dispatcher submission explain a plausible end-of-budget wrap;
no algorithmic collapse is inferred and no scheduler fix is claimed.
Common saved checkpoints end at round56. Existing recovery preserves failure
records and evaluates round32 plus the last three common checkpoints40/48/56.
The planned full64-round endpoint is unavailable; this is partial-run evidence.

The first diagnostic queues failed closed before GPU launch. Before endpoint
reports existed, v2 protocols explicitly inherited the original designs and pinned
both failures, recovered SHARED receipts, and metric hashes. Only these verified
training failures are eligible; new evaluation failures remain fatal. Original
selection rules and all contrasts unchanged. New recovery regression plus the
factorial/intervention tests:5 passed. Independent frozen source manifest:
`runs/source-snapshots/alfworld-ma-diagnostics-v2/SOURCE_MANIFEST.json`.
Node3 `alfworld-ma-improvement-study-v2`, node4
`alfworld-h112-guard-ablation-v2` await both original evaluations and GPU release.
Correct branch research consumption is1840+128+1000=2968, including the failed
first attempt; generated and consumed counts overlap and must not be summed.


### End-of-budget execution experiment and local repair

RQ: Can native overlapping submission wrap a finite joint ledger even when the
trainer requests only the planned batches? A CPU probe invokes the real upstream
BatchTaskDispatcher method with delayed fake results,24 cyclic inputs and3x8
requested outputs. It submits32 inputs, repeating the first8, while delivering24.
The local JointRolloutBudget adapter retains native queue/staleness capacity and
result waiting, but caps submissions at steps*batch_size; it submits/delivers24
without wrapping. Rejection, dynamic batches, insufficient inputs and excess
requests fail explicitly, rather than inventing/replacing paired trajectories.
Installed only on future joint training dispatchers, with no third-party/global
patch or modification of frozen running experiments.45 related tests passed;
real-native CPU comparison is FINITE_BUDGET_CPU_PROBE.json in study_v2.
This validates the scheduling mechanism on a controlled CPU test, not a completed
GPU training run or any task-accuracy improvement. GPU reproduction remains pending.

Diagnostic v2 first startup encountered empty gitlink dependency directories in
the new worktree, before runner entry or any evaluation. Dependency links were
repaired (DEPENDENCY_LINKS.json); frozen Python files unchanged. Original logs
retained. Both services restarted and verified active, waiting for the predecessor.


## 2026-10-04：改进分支与机制消融结果已完成

研究问题1：低学习率/KL/保留h0的方法包能否超过92%？主终点selection48选中
round40/proposal02，all200=88%，相对历史92%为-4pp，配对95%[-9,+1]pp；
round32辅助终点88.5%，不能因较高就替换主终点。训练仍是H62/W63部分完成。
同环境交叉重测：起点W+空/h0/新patch=78/89/85.5%；选中W=82/85/88%；
W1空/h2=87.5/89.5%。相对重测基线-1.5pp[-6.5,+3.5]，没有优越性证据。
固定h0权重变化-4pp[-8.5,+0.5]；空harness+4pp[-2,+10]；兼容性交互
+6.5pp[+0.5,+13]，提示协适应而非单一组件全面提升。单训练种子、复用开发题。

研究问题2：h112动作护栏是否损害表现？固定MA64，同次重测原h112=81%，
删除整钩子=87%，空=79%；+6pp[+2.5,+10]，14题改善/2题退步。
支持该钩子在当前条件下有害，但尚不能归因到其中某一个条件；不是学习方法成绩。
两组诊断分别1600/600环境episode，0新增训练；分支累计训练消费2968条。
完整结果：runs/alfworld_ma_improvement_study_v2/COMPLETE.json及
runs/alfworld_h112_guard_ablation_v2/COMPLETE.json。历史92%不被重测89.5%覆盖。

下一问题：仅禁用“错误物品阻拦”分支能否解释整钩子消融收益？已冻结并启动
runs/alfworld_h112_branch_ablation_v1，node4服务alfworld-h112-branch-ablation-v1。
固定MA64，原h112/仅禁用该分支/删除整钩子/空harness，共800episode，0训练。
主比较为单分支减原版；次比较为删整钩子减单分支。后者不显著不代表等价。
源码独立快照，相关6测试通过；其他动作及两物体任务防护代码保持原样。


## 持续总目标（2026-10-04用户明确授权）

持续提出研究问题、设计/运行实验并分析，直到MA-SAO超过历史W1+h2=92%，
且比较公平性/可信度经过核验。不能为达标选择性报告、改口径或把人工消融当学习成绩。
Codex active goal已创建；不要仅启动队列就把总目标标为完成，持续监控并推进。

当前单分支消融alfworld_h112_branch_ablation_v1在node4运行。下一研究问题是
非法patch回退空harness却获得成功奖励是否妨碍Engineer学习。
冻结协议runs/alfworld_validity_credit_study_v1/protocol.json：team对照与
validity_gated处理，两组各32轮、512条H+W，总1024条；同H0/MA64起点、
数据/种子/超参，仅非法patch的Editor奖励置零，Solver/环境不变。
固定round32为终点，8proposal+incumbent、selection48与2/48门槛不变，
不根据200题结果换模型。对照和处理都使用预算修复，旧v2仅作历史支持。
相关49测试通过。源码冻结runs/source-snapshots/alfworld-validity-credit-v1。

正式两臂尚未启动。先跑runs/alfworld_joint_budget_gpu_smoke_v1两轮原生GPU验证：
各16轨迹、1actor/4critic更新、总32条角色训练轨迹；必须无重复/故障且完整收尾。
队列alfworld-joint-budget-gpu-smoke-v1-queue等待单分支消融完成和GPU释放，
自动启动node3 editor/node4 solver。两个spec、协议、queue.py及日志可核验。
下一步：收单分支结果，核验GPU smoke的两role COMPLETE、metrics与完整唯一receipt；
通过后准备并依次运行同快照的32轮对照和处理，比较主终点及非法patch率。
若失败，修复并保留失败成本，不启动依赖训练。不得修改已冻结源码。


### Validity-credit execution queued after native GPU gate

Prepared both32-round arms with identical inherited H0/MA64 models, task order,
seeds, selection48 and optimizer settings. Only Editor invalid-patch reward mode
differs. Both use bounded submission. Frozen driver/learner snapshot:
runs/source-snapshots/alfworld-validity-credit-study-v1. Main service
alfworld-validity-credit-study-v1 is active, waiting for the two-round GPU smoke.
It requires exact update counts,16 unique receipts per role, paired outcomes,
correct role/team rewards and clean release before launching control training,
its fixed round32 evaluation, then treatment training/evaluation. Driver/audit
regressions4 passed. PRIMARY_RESULTS will remain explicitly diagnostics-pending;
it is not a full research completion flag. DIAGNOSTIC_PROTOCOL.json freezes
fixed-h0 and contemporaneous W1+h2 comparisons, max1600 environment episodes,
before primary scores exist. No new training beyond two512-role-trajectory arms.
Study helpers read committed artifacts and preserve all failed attempts; snapshots
must not be edited. Continue monitoring predecessor branch ablation, GPU smoke,
then study units. The overall>92% goal remains active and unachieved.


### Prespecified credit diagnostics now persistently queued

Implemented frozen diagnostics for the two credit arms: control/treatment Solver
with no patch, fixed h0, own selection48-chosen patch, plus contemporary W1
with no patch/h2. Same200 questions and inference contract, maximum1600 episodes.
Original primary results remain separate from remeasurement; all200/selection48/
report152 paired contrasts are reported. The analysis regression demonstrates
that fixed-h0 weight gains need not imply a selected-system gain and rejects
mismatched rollout identities. Five diagnostic/driver tests passed.
Node3 unit alfworld-validity-credit-diagnostics-v1 is verified active, waiting for
both primary endpoints. Source: runs/source-snapshots/alfworld-validity-credit-diagnostics-v1.
No new score yet; live branch-ablation inference continues normally as of16:41UTC.


## 2026-10-04 单分支主比较完成，辅助组恢复中

单分支v1在第四变体推理ReadError后整体FAILED；前三份200题native报告已完整落盘，
经checked_native_report校验后保留，未按分数选择：空78%，原h11278%，
仅禁用错误物品阻拦85.5%。主比较+7.5pp[+4,+11.5]，15改善/0退步，
PRIMARY_PARTIAL_RESULT.json明确primary_complete_auxiliary_failed，原FAILED不删除。
这支持该分支在当前固定MA64/复用开发题条件下有害；不是学习成绩或>92%证据。

只恢复缺失的without_guard辅助组，runs/alfworld_h112_branch_ablation_v2，
node4 unit alfworld-h112-branch-ablation-v2；沿用冻结推理源码及config，
resume.py和所有复用报告/输入固定SHA。额外跑without_guard+fresh_no_patch共400回合；
原3组完整600回合保留，fresh_no_patch只作跨次波动诊断，不覆盖原空对照。
辅助差异跨服务会话，必须披露。失败尝试环境成本上限800，另加恢复400。

原GPU smoke队列因前置失败而退出，0训练启动；新unit
alfworld-joint-budget-gpu-smoke-v1-queue-recovery等待branch v2 COMPLETE/RELEASED，
训练spec和源码不变，QUEUE_RECOVERY.json记录修订。后续credit study与diagnostics
服务仍active等待，不需重启或重复启动。两臂配置比对通过：数据完全相同，
除奖励模式外只有路径/实验名/截止时间差异；ARMS_COMPARABILITY_AUDIT.json。
总目标保持active。下一步完成辅助组、核验GPU smoke、继续32轮两臂与基线诊断。


### Mechanism localization from completed one-branch reports

MECHANISM_TRACE_AUDIT.json:15 rescued tasks,14 with the targeted blocker;
13 previously reached task step limit,1 completed unsuccessfully,1 invalid action.
Families: two-object3, simple4, cool5, heat1, clean2. This is post-hoc trace
interpretation, separate from the prespecified primary causal contrast.

A misleading surface trace was investigated: task324 requests `take pan 1 from
stoveburner 4` while holding pot1; that request is inadmissible. Calling the
unchanged upstream process_action on the actual preceding admissible list maps
it to `put pot 1 in/on stoveburner 4`. The learned guard examines final_action
after stock normalization, then rejects wrong-object placement. Thus raw tool
requests alone do not identify what the guard saw. NORMALIZATION_MECHANISM_PROBE.json
pins code/report hashes and records this CPU replay; no environment reward changed.
This supports a future recovery-preserving semantic patch contract, but that
method is not implemented or claimed effective. First finish the already frozen
validity-credit comparison; syntactic/runtime validity alone cannot detect this
kind of harmful but executable hook. No new GPU experiment was added from this probe.


### Prospective success verification, before any>92% candidate

Frozen runs/alfworld_ma_goal_validation_v1/protocol.json. Historical point target
requires at least185/200 at the predeclared primary endpoint. Bind the chosen
weights/harness before verification; never switch using diagnostic outcomes.
Verify with inference seeds20261005/20261006 against unchanged W1+h2 under the
same runtime/token/temperature contract. Publish both seed results and a question-
clustered paired interval; pooled candidate accuracy>92% and positive paired lower
bound are required for supported inference superiority. Repeat training with a
new seed before calling the gain robust. All200/report152 remain development data;
unseen-task claims require a separately audited unopened cohort. Costs and failed
attempts remain disclosed. No new compute was launched by this validation protocol.
It has not been triggered: current best new learned primary score remains88%.


## 2026-10-04 17:10 辅助对照完成，GPU预算修复验证已启动

branch_ablation_v2/COMPLETE与FINAL_AUDIT已核验协议/报告SHA及GPU释放：
原h11278%，仅禁用错误物品阻拦85.5%，删除整钩子86.5%。
主比较+7.5pp[+4,+11.5]；整钩子相对单分支+1pp[-1,+3]，不支持额外收益，
亦非等价证明。辅助差异跨服务会话；空harness重测78→77.5%，-0.5pp[-2,+1]。
原3组600完整episode复用；恢复新增400，失败尝试最多800不得漏记。
这是固定权重的人工机制实验，不是新学习模型达标成绩。

GPU smoke队列自动接续成功：17:08启动node3 editor/node4 solver，
units alfworld-joint-budget-gpu-smoke-v1-editor/solver。两节点MainPID均已核验active，
日志显示真实AReaL训练引擎初始化，尚无训练完成结论。
后续credit-study驱动仍等待精确两轮/各1actor与4critic更新、各16唯一轨迹核验。
通过后自动执行control32轮→固定终点评估→treatment32轮→固定终点评估，
再由node3诊断服务执行固定h0和W1+h2重测。继续监控，勿重复启动或改冻结源。


## 2026-10-04 17:25 GPU预算修复验证通过，正式对照训练已启动

runs/alfworld_joint_budget_gpu_smoke_v1/FINAL_AUDIT.json已核验：H/W均完整2步，
各1次Actor、4次Critic更新；各16条唯一生成/消费轨迹，16个配对环境结果；
无额外采样、无第0轮回绕故障，保存后正常退出并释放GPU。
最终Actor梯度范数H1.6631/W0.7823，有限非零；Critic梯度也有限非零。
这证明预算修复在真实原生训练中收尾成功，不是准确率提升证据。
累计本研究路径训练消费2968+32=3000条，未将生成数重复加上。

自动接续control32轮已启动：node3 validity-credit-control-editor-v1，
node4 validity-credit-control-solver-v1，MainPID与joint_native均核验在线。
root runs/alfworld_validity_credit_study_v1/control。仍用原H0/MA64起点，
不继承smoke权重；control奖励team，treatment奖励validity_gated只改非法H奖励。
主终点固定round32、selection48规则不变；当前尚无新准确率。
顺序：control训练/评估→treatment训练/评估→node3固定h0和W1+h2诊断。
驱动alfworld-validity-credit-study-v1与诊断队列保持active；继续监控失败/真实进度。


### CPU recovery-contract feasibility probe while control trains

runs/alfworld_recovery_contract_probe_v1 pins source/patch hashes and12 constructed
states:3 target types crossed with simple/heat/cool/clean task families, carrying
a distractor with an admissible put-away action away from the destination.
Native compile_hook/run_hook executes initialization, observation update, hint,
and before-action hook in sequence. h112 blocks12/12; h0,h64,h2,the guarded
round40 selected patch,and branch-disabled h112 block0/12. No rewrite/force effects.
Zero environment episodes and zero training trajectories. This is a post-hoc CPU
counterexample suite, not a test of accuracy or proof of semantic correctness;
the native runner suppresses runtime errors, which may also yield no effect.

Interpretation: a small behavioral check can detect the known h112 defect, but
the current88% selected patch already passes. Therefore adding only this check
cannot explain/solve the remaining gap to92%. It is a candidate regression guard,
not an asserted successful learning method. Frozen credit-control training is
unchanged; do not add semantic filtering to either existing arm mid-run.


### Fixed-h0 compatibility interaction in the completed guarded matrix

H0_COMPATIBILITY_INTERACTION.json under alfworld_ma_improvement_study_v2 pins
four existing reports. At initial weights h0-minus-empty=11pp; after continuation
it is3pp. The paired question difference-in-differences is-8pp,95%[-15,-1]pp.
This is a post-hoc exploratory contrast, not a new primary endpoint or an isolated
causal mechanism. It strengthens the question of retaining compatibility with a
strong fixed harness during Solver adaptation. It does not prove that adding
anchor rollouts will help. The currently running validity-credit arms remain
unchanged. As of17:49UTC control H/W both committed9 steps, healthy, and resumed
after the round8 checkpoint save; no new evaluation result exists.

### 2026-10-04 — Same-weight likelihood discrepancy in credit-control warmup

Research question: can high teammate importance clipping be explained solely by
actor updates? A post-hoc CPU audit of all 24 trajectories per role in the first
three critic-only warmup rounds is saved in
`runs/alfworld_validity_credit_study_v1/WARMUP_LIKELIHOOD_AUDIT.json`, with a
reproducible adjacent script and input SHA-256 hashes. Both roles have zero
fixed-reference KL on these records and no actor update during warmup.

Engineer behavior/proximal trajectory ratios already exceed the clipping bounds
in 18/24 records (14 below 0.5, 4 above 2); Solver ratios do so in 0/24.
Engineer median token count is 2760 and median per-token log-ratio is -0.000413;
median summed log-ratio is -1.011. These Engineer ratios weight Solver advantages,
not Engineer advantages. Warmup weights do not update either actor.

Conclusion: actor parameter changes alone cannot explain the discrepancy.
Backend numerical error, sampling transforms, token alignment, and implementation
issues remain unresolved alternatives. This is not evidence of an accuracy
effect. A candidate follow-up is a fixed-weight replay of identical tokens through
both probability paths, followed by a separately preregistered correction
experiment only if warranted. No GPU job or additional environment episode was
launched, and neither frozen reward-credit arm was changed. By 18:00 UTC both
control learners had committed 14/32 steps, with healthy latest diagnostics.

### 2026-10-04 18:44 UTC — Validity-credit control completed its frozen budget

Both control learners completed all 32 rounds: each consumed/generated exactly
256 unique role trajectories, performed 29 actor updates and 64 critic updates,
published final checkpoints, and released training GPUs. Re-running
`audit_training(control, 32, 3)` matched `control/TRAINING_AUDIT.json`, including
ordered metric hashes and complete paired outcome coverage. No training failure
or extra cyclic generation occurred. The control arm adds 512 consumed role
trajectories; the current research branch path total is 3512 including the
previous 2968 and the 32-trajectory budget smoke. Treatment is not yet trained.

There were 20 invalid patches among 256 paired training episodes, with 14 team
successes despite invalid patches. This verifies exposure to the hypothesized
credit-assignment issue in the control run, but does not establish its accuracy
effect. Fixed-round32 candidate generation/evaluation started automatically in
`validity-credit-control-evaluate-v1`; no endpoint performance result is available
yet. Both arms' frozen selection rules remain unchanged.

### 2026-10-04 19:00 UTC — Control selection48 finished; evaluation200 started

The fixed-round32 control generated eight proposals, seven valid. On selection48,
proposal03 scored44/48, incumbent h0 scored42/48, and empty harness40/48. The
automatic selector chose proposal03, meeting the predeclared two-success margin.
`control/matched/SELECTION_AUDIT.json` verifies every report's exact question and
sample coverage, decode seeds, binary rewards, token budget and accuracy. Nine
unique harnesses were evaluated (the invalid proposal shares empty-harness
results), costing432 completed environment episodes. Input/report hashes and the
selected artifact are bound in the audit. The original200 endpoint request uses
the selected patch and empty harness at the frozen Solver checkpoint; its process
is running. Selection performance is not endpoint performance, and no final
accuracy claim or treatment comparison is yet available.

### 2026-10-04 — Control200 result and summary-only recovery

Frozen round32 proposal03 scores175/200=87.5%; empty harness163/200=81.5%.
Versus historical W1+h2=92%, paired difference is-4.5pp,95%[-10,+0.5]pp
(10 improved,19 regressed). Versus reused branch start90%, difference is-2.5pp,
95%[-6.5,+1.5]pp. These are exploratory single-training-seed development results;
the experiment does not meet the goal. Treatment comparison is still pending.

Native evaluation completed all400 environment episodes, but the outer summarizer
failed because prepare_arms copied initial_COMPLETE without its full reports.
Recovery validated both endpoint reports' full200 coverage, seeds and token
budgets, copied the unchanged parent's initial reports into both arms, and executed
only the exact frozen summarization block. Input hashes, original failure hashes,
and the recovery script are retained. Zero additional model calls, environment
episodes or training trajectories were used. Original failure files remain.

prepare_arms now carries both initial reports. wait_files accepts only exact
failure hashes backed by immutable recovered-result hashes; new failures remain
fatal. Six driver/audit tests pass, including changed-failure and changed-evidence
rejection. A separate recovery snapshot changes only this orchestration module;
treatment still launches from the original learning/evaluation snapshot.

The first recovered diagnostic waiter hit a shared-directory traversal permission
error. Its log/failure record is preserved, and permissions for the specific pinned
report path were repaired and verified using the real uid1009/gid1008 service
identity before restarting it. Treatment H/W, recovery driver, and diagnostic
waiter are now all confirmed active. Fixed endpoints and comparison rules are
unchanged; no score-triggered retraining or reselection occurred.

### 2026-10-04 20:39 UTC — Treatment training and manipulation audit complete

Treatment completed32 rounds with256 generated/consumed trajectories per role,
29 actor and64 critic updates per role, no budget overflow or training failure.
An independent call to audit_training matched the automatic TRAINING_AUDIT.
Both arms now consumed1024 role trajectories; branch total including prior2968
and smoke32 is4024. Fixed-round32 treatment candidate evaluation is active.

BOTH_TRAINING_AUDIT.json pins both training audits and records that invalid patches
were20/256=7.81% for control and31/256=12.11% for treatment. Among invalid patches,
team successes were14 and19 respectively. Invalid-patch Engineer reward sum was14
for control and0 for treatment, while Solver retained environment success rewards.
Thus the reward manipulation executed correctly, but observed training invalidity
did not improve. These descriptive single-training-seed counts do not establish
an accuracy effect; final treatment evaluation and prespecified diagnostics remain
pending. No training conditions or endpoint-selection rules were changed.

### 2026-10-04 20:57 UTC — Treatment selection48 retained incumbent h0

All eight generated proposals were valid. Best new proposal03 scored40/48,
incumbent h0 scored39/48, and empty harness35/48. Because the new candidate gained
only one success, the fixed two-success replacement rule retained h0. Other new
candidate scores were36,36,37,25,35,34,39 successes for proposals00,01,02,04,05,06,07.
The chosen endpoint is therefore the round32 treatment Solver with original h0.

`treatment/matched/SELECTION_AUDIT.json` validates all report coverage, seeds,
binary rewards, token limits and scores. Ten unique harnesses cost480 completed
environment episodes versus control's432; this48-episode difference comes from
the control's invalid candidate sharing the empty report. Both arms still sample
eight new candidates and use identical selection rules; inference cost is not
claimed equal. Treatment200 evaluation has started using the locked selection.
Selection-set scores do not establish the final treatment effect.

### 2026-10-04 21:16 UTC — Validity-credit primary result is negative

Treatment selected h0 with its round32 Solver scores156/200=78%, compared with
control selected proposal03 at175/200=87.5%. Paired treatment-minus-control is
-9.5pp,95%[-15,-4]pp (8 improved questions,27 regressed). Empty-harness accuracy
also drops81.5% to73%: -8.5pp,95%[-14.5,-2.5]pp. On report152 the corresponding
differences are-10.53pp and-8.55pp; these remain reused development questions.

PRIMARY_AUDIT.json independently verifies both endpoint reports' full coverage,
seeds, binary rewards, token budgets, pinned hashes, round32 selection and release
records; the recomputed primary comparison matches PRIMARY_RESULTS.json. Both
arms used1024 total role trajectories. Full branch training cost is4024 including
previous failures/smoke. Selection plus primary evaluation cost1712 environment
episodes (control432+400, treatment480+400), with no recovery rerolls.

Conclusion: this single-seed experiment does not support zeroing invalid Engineer
credit as a useful improvement. Correctly applying the reward mask is insufficient
to improve either invalid rate or deployed performance. Empty-harness degradation
means a patch-only explanation is insufficient; the already-prespecified fixed-h0
and contemporary W1+h2 matrix is the next experiment to separate Solver change,
harness compatibility and inference variation. Question bootstrap is not
training-seed robustness. No superior candidate exists and the overall goal is
not achieved. Diagnostic service is active; no new training method is launched yet.

### 2026-10-04 21:28 UTC — Fixed-weight probability replay launched

Question: can the stored Engineer proximal/behavior mismatch be reproduced at
unchanged initial weights, and does batch ordering materially change it? The
training-dynamics audit finds finite inspected metrics but mean Solver peer-weight
clipping around68–70%. In critic-only warmup,18/24 Engineer trajectories already
exceed the factor2 cap, motivating a numerical/implementation diagnostic.

Protocol and input hashes are frozen in alfworld_fixed_weight_probability_replay_v1.
Eight original first-round single-call Engineer trajectories are scored twice by
the native FSDP backend, original order then cyclic shift. Initial model, temperature
and precision follow the original trainer; optimizer=None. This uses node4 GPUs0–3,
16 trajectory rescoring calls,0 new environment episodes/generation/updates.
Alignment and input-integrity tests:2 passed. Service launched; results pending.
This posthoc diagnostic cannot establish an accuracy effect. Node3 continues the
prespecified fixed-h0/contemporary-baseline matrix independently.

### 2026-10-04 21:31 UTC — Probability replay configuration recovery

v1 failed before workers/scoring: scheduling_strategy is nonoptional. Failure/log
hash retained,0 rescored trajectories/updates/environment episodes. v2 uses default
scheduling strategy and local unused rollout credential placeholders. Structured
configuration preflight passed. Original cohort/model hashes, estimator and order
are retained in a new frozen snapshot/protocol. Recovery service launched; pending.

### 2026-10-04 21:33 UTC — Probability replay directory recovery

v2 failed before worker creation because cluster.fileroot did not exist; FAILED
and RELEASED retained,0 scoring/updates/environment episodes. v3 precreates engine
and name_resolve directories, reuses frozen v2 code and unchanged model/cohort
hashes, and has launched as alfworld-fixed-weight-probability-replay-v3. Pending.

### 2026-10-04 21:38 UTC — Communication recovery and descriptive weight audit

v3 stalled during fourth-worker configuration: localhost health200, node-IP
ConnectionError on that port. Stopped owned unit after this diagnosis; GPU release
verified,0 model engines/scoring/updates/environment episodes. v4 uses the same
frozen v2 source/cohort/weights, with controller AREAL_HOST_IP=127.0.0.1 and required
directories precreated. Running; no estimator change.

IMPORTANCE_GROUP_AUDIT.json pins all256 proximal/receipt pairs per arm/role.
All232 actor-update-phase trajectories contain lagged versions. Engineer weights
(applied to Solver) average0.8204/0.8653 control/treatment; lower clipping133/232
and129/232. Success/failure weights average0.848/0.754 versus0.827/0.961. Weighted
team-success fractions70.69→73.07% and71.55→68.39% are descriptive reward reweighting,
not actual advantage/gradient effects or causal accuracy evidence. No new episodes.

### 2026-10-04 21:41 UTC — Control diagnostic and RTensor recovery

Control diagnostic600 episodes completed: selected89%, fixed h087.5%, empty81%,
each200 questions. These retests do not replace original primary87.5%/81.5%.
Treatment diagnostic is now running; complete contrasts await all cells.

Probability v4 performed the first8-trajectory native forward but returned
RTensors, causing CPU summary.detach failure. Preserve8 rescored trajectories
in COST.json,0 retrieved results/updates/environment episodes; GPUs released.
v5 materializes results using native RTensor.localize, with unchanged weights,
cohort and estimator. Frozen snapshot and service launched; results pending.

### 2026-10-04 21:45 UTC — Fixed-weight discrepancy reproduced; peer-weight ablation frozen

Replay v5 and independent AUDIT completed:7/8 initial Engineer trajectories clip
at factor2; max replay-minus-original logsum difference1.3494e-6; cyclic batch
permutation changes no token logp. Optimizer=None; release verified.16 completed
rescoring trajectories plus8 failed-v4 forward trajectories=24 total,0 generation,
updates or environment episodes. This does not isolate every backend/sampling
cause and does not prove an accuracy effect.

New question: does removing this trajectory-scale peer weighting improve final
MA-SAO performance? alfworld_peer_weight_ablation_v1/protocol.json freezes a
32-round512-role-trajectory unit-weight arm, team reward unchanged, same original
control initialization/seed/data/optimizer budget and round32 selection rule.
Existing control primary87.5% is reused transparently; this is one nonconcurrent
training realization, not seed robustness. Raw ratios/cohort validation/PPO own
ratios/KL/critic remain intact. Default bounded mode is unchanged.7 tests passed.
Native smoke and resource-gated launch remain pending. Historical92% threshold
and prospective superiority-validation requirements remain unchanged.

### 2026-10-04 21:51 UTC — Native gates and persistent ablation driver launched

Frozen peer-weight-ablation-v1 source prepared;11 relevant tests passed. Cache
isolation changed the expected path, so the old cache-root assertion first failed
and was updated to require the output-local runtime-cache. Validation addendum
records24 synthetic full-context trajectories (2 actor,6 critic steps;0 checkpoints
or environment episodes), then32 fresh role-trajectory unit smoke (16 joint episodes)
and512 main role trajectories. Smoke weights are not reused for main training.

Node4 memory service is live with H0,32768 context plus1024 critic-prefix tokens.
Persistent study driver independently audits the memory result and source hashes,
waits for node3 diagnostic completion/release, runs2-round smoke, verifies actual
unit weights/raw-ratio diagnostics and exact native update ledgers, then trains
32 rounds and evaluates the frozen endpoint. All specs and driver inputs prepared
and pinned. Main training has not started; no claimed new accuracy result.

### 2026-10-04 21:52 UTC — Control match audit and live native check

CONTROL_MATCH_AUDIT.json verifies byte-identical ordered data and identical
learning specs except peer_weight_mode; other differences are run paths, names,
deadlines and pinned provenance. LEARNING_SOURCE_DIFF.txt contains the complete
four-module learning-code diff. Native memory service remains live; first critic
updates have appeared, but required2 actor/6 critic steps and release are pending.
No main training or new accuracy result is claimed.

### 2026-10-04 21:54 UTC — Fixed-h0 diagnostic isolates a Solver-associated deficit

Control/treatment diagnostic1000 episodes completed. Fixed h087.5→78% gives
-9.5pp,paired95%[-15,-4] (7 improve,26 regress); empty81→73% gives-8pp
[-14,-2]; selected systems89→78% gives-11pp[-16.5,-6]. Partial report hashes
and comparisons are saved. Same fixed h0 does not remove degradation, so
selected-harness choice alone cannot explain it. Single training realization
and reused-development limitations remain. Contemporary baseline now running.

Unit-ablation full-context check completed2 warmup critic updates;2 actor and
remaining4 critic updates/release still pending. Both validation/driver services
remain live; no main unit training has started.

### 2026-10-04 21:59 UTC — Secondary diagnostic prepared before training

SECONDARY_PROTOCOL.json corrects the fixed-h0 diagnostic upper bound from200 to
400 episodes because native matrix evaluation includes an empty-harness control.
Primary selection/budget/endpoints remain unchanged. If selected harness equals
h0 by hash, reuse primary h0/empty reports and incur0 extra episodes; otherwise
run fixed h0 plus empty once. Compare against prior control fixed h0 and preceding
contemporary W1+h2, preserving historical92%. Secondary service is queued behind
PRIMARY_COMPLETE.json. Main training has not begun.

### 2026-10-04 22:06 UTC — Full-context modified Actor gate passed

Four ranks each completed2 actor/6 critic updates at32768 context plus1024 critic
prefix tokens. Actor gradients5.4123/2.9354 were finite and parameter samples
changed; fixed-reference outputs remained bitwise equal. Peak actor119.073GiB,
critic98.250GiB. All worker GPUs released and filler preserved. Driver independently
recomputed the native audit, published MEMORY_COMPLETE with source hashes, and
advanced to waiting for node3 diagnostics. Cost24 synthetic trajectories,0 new
environment episodes/checkpoints; unit smoke and main training still pending.

### 2026-10-04 22:10 UTC — Credit study final audit; unit smoke launched

All diagnostic report hashes, coverage, seeds, token budgets and18 paired
comparisons independently verified in FINAL_AUDIT.json. Fresh W1+h2 scores90%,
empty87.5%; treatment78% is-12pp versus fresh baseline,95%[-18,-6],9 improve/
33 regress. Historical92% remains the target. Original primary87.5% control and
78% treatment are retained. Fixed-h0 and empty deficits persist, so patch choice
alone is insufficient; no single training mechanism is causally established.

Cost:1024 role trajectories for both arms (branch cumulative4024 before peer
ablation);912 selection+800 primary+1400 diagnostic=3112 evaluation episodes,
separate from training episodes. Single-seed/reused-development limitations apply.

Diagnostic services ended normally and released node3. Driver automatically
launched peer-weight smoke Engineer/node3 and Solver/node4; both native processes
are live. Actual unit metrics/update/ledger gates precede the main32-round run.

### 2026-10-04 22:25 UTC — Native unit-weight smoke passed; main32 rounds launched

Independent SMOKE_FINAL_AUDIT confirms16 generated=16 consumed per role,1 actor/
4 critic updates each,2 rounds applied peer mean1/clip0/effective fraction1. Raw
ratios remain logged; Solver raw clipping75% in both rounds. Both roles ended
cleanly and released GPUs. Cost32 role trajectories/16 joint episodes; cumulative
branch consumption4056 before main training, with24 synthetic memory trajectories
separate. Smoke weights are not used to initialize the main arm.

Driver launched main Engineer/node3 and Solver/node4 services, both live. Frozen
32 rounds/512 role trajectories,3 critic warmup rounds,29 actor/64 critic updates
per role, unchanged team rewards and original control initialization. Only peer
weighting changes to unit. Automatic endpoint and secondary workflows remain
queued. Execution validation is not an accuracy improvement claim.

### 2026-10-04 22:45 UTC — Prospective verification analysis implemented

goal_validation_analysis.py validates both prespecified inference seeds, native
question coverage, rewards, per-question seeds and token budgets, then averages
within question before paired cluster bootstrap. Bootstrap seed20261080,10000
draws are recorded before any candidate verification.4 tests passed, including
anticorrelated seed outcomes that must remain in the same question cluster,
missing/mismatched seeds, duplicate questions and inference-pass versus overall
goal-completion separation. No candidate validation or new episodes launched.
Current unit-weight training continues with its first scheduled checkpoint save.


### Peer-weight ablation halfway audit — 2026-10-04T23:07:28.837463+00:00

Both roles completed 16/32 rounds and resumed after checkpoint saving.
HALFWAY_AUDIT.json binds the first 16 parsed metric rows per role and verifies
applied teammate weights=1, clipped fraction=0, effective fraction=1, and unit
mode=1 in every audited round. Both health assessments are healthy. Consumed
256 role trajectories so far (128 per role); endpoint accuracy is unavailable.
The fixed round-32 selection/evaluation protocol remains unchanged.


### Peer-weight ablation training complete — 2026-10-04T23:43:12.956267+00:00

Both roles finished exactly32 rounds, each generated=consumed256 trajectories.
Independent MAIN_FINAL_TRAINING_AUDIT.json reproduces the driver audit and
checks applied unit teammate weights in all32 metric rows per role. Native
endpoint evaluation started under alfworld-peer-weight-unit-evaluate-v1.
New research training cost:512 main +32 smoke role trajectories;24 synthetic
memory trajectories recorded separately. Branch cumulative consumed research
role trajectories=4568. Invalid patches25/256 (9.765625%);18 such episodes
had team success. These are training diagnostics, not endpoint accuracy.
The fixed round32 endpoint and original selection protocol remain unchanged.


### Unit-peer selection completed — 2026-10-05T00:01:05.590218+00:00

SELECTION_AUDIT.json validates all10 reports on48 tasks (480 episodes),
seeds/token limits, report hashes, release, and frozen incumbent guard.
Best new proposal03=39/48, incumbent h0=39/48, empty=32/48. Retain h0
because no new proposal exceeds it by at least2 successes. The final round32
all200 h0/empty evaluation started (native PID3804129). No all200 score yet.
Because selected=h0, the prespecified secondary diagnostic will reuse these
primary reports and add zero environment episodes.


### Unit-peer ablation completed, negative — 2026-10-05T00:19:25.683614+00:00

FINAL_AUDIT.json validates native all200 reports, summary bindings, primary
and fixed-h0 paired comparisons, normal release, and zero-reroll secondary reuse.
Selected h0=168/200(84%), empty=145/200(72.5%). Versus original control87.5%/81.5%,
deltas=-3.5pp[-8,+1] and-9pp[-14.5,-3.5]. Fixed-h0 diagnostic control87.5%
gives-3.5pp[-8.5,+1.5]. Historical92% gap=-8pp[-13.5,-2.5].
No support for unit teammate weights as a performance remedy. One nonconcurrent
training realization and reused development tasks; question bootstrap does not
establish seed robustness. Removing weights also changes average scaling, so
this cannot isolate the source of the earlier likelihood mismatch or its causal
impact on accuracy. Costs:512 main+32 smoke role trajectories,24 synthetic memory
trajectories separately;480 selection+400 primary+0 secondary environments.
Branch cumulative consumed research role trajectories4568. Goal remains unmet.


### Anchor retention next question — 2026-10-05T00:23:24.071998+00:00

Created runs/alfworld_anchor_retention_study_v1/DESIGN.json after the completed
negative unit-weight experiment. Planned fresh paired arms share joint/joint/
Solver-only schedule; only the applied harness on Solver-only rounds differs
(sampled proposal versus fixed h0). Engineer updates must be suppressed there
in both arms; shadow generations remain counted. At48 rounds, per-arm consumed
H+W=640, generated=768, environment episodes384. This is not budget-matched to
the earlier32-round control. Pure input/routing helpers in joint_anchor.py pass
7 tests (fresh IDs, exact retained input order, exposure mapping, intervention
separation, invalid input rejection). Native integration and gates remain pending;
no new GPU training or evaluation launched.


### Anchor schedule controller integration — 2026-10-05T00:25:59.560762+00:00

Extended joint_schedule, large_train, and joint_health for joint_joint_anchor.
Existing/helper tests30 passed; two new stateful CPU controller tests passed.
At48 rounds Engineer29 Actor/64 Critic updates, Solver45 Actor/96 Critic
updates; inactive Engineer parameter/optimizer/normalizer/scheduler snapshots
remain unchanged. Source rows are reconstructed and checked, consumed H+W640
per arm. These are CPU controller tests, not native distributed GPU evidence.
Workflow routing, worker peer-weight configuration, and native gates remain
pending; zero new GPU runs. One initial test-append shell invocation used
unavailable bare python and failed before writing; rerun with the established
venv interpreter succeeded.


### Anchor workflow integration — 2026-10-05T00:28:51.826217+00:00

JointWorkflow now pins h0 by hash, routes the actual environment harness, retains
the original proposal and applied patch, and cross-checks Editor outcome routing.
Both arms use unit peer weights on Solver-only rounds; bounded on joint rounds;
the paired proximal exchange remains. Worker environment propagation is wired.
Spec validation rejects unpinned anchor/source inputs and routing without the
matching schedule. Related tests50 passed plus3 subtests; latest source-pin and
stateful controller tests2 passed after validation changes. Native smoke/GPU
evidence is still pending; zero new GPU runs.


### Anchor memory validation launched — 2026-10-05T00:32:20.215209+00:00

Frozen source runs/source-snapshots/alfworld-anchor-retention-v1 with
ANCHOR_SOURCE_MANIFEST.json. Materialized worker boundary tests4 passed,
all joint-update tests9 passed. Added a separate scheduled-budget/routing audit;
its failure-case tests remain pending. Native full-context memory validation
alfworld-anchor-memory-v1 launched on node4 after verifying only filler owned
GPU processes. Predeclared cost24 synthetic trajectories,0 environment episodes,
0 formal training trajectories. Formal/smoke runs not launched yet.


### Anchor protocol frozen and runs prepared — 2026-10-05T00:35:49.378266+00:00

Audit failure-case tests7 passed: wrong applied patch/identity, extra outcomes,
failure marker, inactive Actor update, wrong peer mode rejected. Source v2
changes only offline audit and tests relative to memory-running v1; terminal
Actor and memory engine code identical. Frozen protocol.json specifies two
6-round smoke arms (160 consumed/192 generated total) and two48-round formal
arms (1280 consumed/1536 generated total), fixed endpoint/selection48/h0 guard,
primary bootstrap seeds20261081/82 and fixed-h0 diagnostic seed20261083.
prepare_runs.py created and validated immutable specs/data for all four runs.
Do not rerun preparation or rewrite frozen source/configs. Only native memory
validation is running; driver and smoke launch still pending.


### Anchor driver queued; memory network recovery — 2026-10-05T00:40:47.942738+00:00

Driver/diagnostic scripts and config hashes recorded in DRIVER_MANIFEST.json;
paired datasets byte-identical and all spec differences restricted to routing
arm/run identities/deadlines. Fixed group-write permissions on prepared roots.
Original memory validation stalled before model initialization: worker0 configured,
ports15206/15578 health200 on127.0.0.1 but unreachable via controller10.1.0.35.
Stopped only own waiting driver and memory service; gpu_release_audit confirms
no foreign research GPU process and filler intact. Zero original forward/update/
environment trajectories. Evidence and recovery driver hash recorded in
MEMORY_NETWORK_RECOVERY.json. New separate memory_recovery run uses unchanged
frozen source and controller AREAL_HOST_IP=127.0.0.1. Waiting driver recovered
under alfworld-anchor-study-recovery-v1; original evidence/configs preserved.
No smoke/formal trajectories yet.

## 2026-10-05T01:01:22.648337+00:00 anchor 内存验证通过，对照组原生联调已启动

32768 token 内存验证已通过：每 rank Actor 2 次、Critic 6 次更新，梯度有限且非零，
reference 输出一致性审计通过；memory_recovery/result.json 确认 GPU 释放与 filler 保留。
24 条合成验证轨迹单列，不计入环境研究训练。MEMORY_COMPLETE.json 已由驱动发布。
当前驱动 alfworld-anchor-study-recovery-v1 PID3832655 active；smoke_control 的
node3 Editor PID2135770、node4 Solver PID3845623 均经 systemctl 确认 active，
两端正在初始化 rollout 服务。尚无联调 TRAINING_AUDIT、正式终点或新准确率。
下一门禁为两组各 6 轮原生联调：实际补丁路由、inactive Engineer 更新隔离、
peer 权重规则、生成/消费成本、模型及 GPU 释放全部通过，才进入正式比较。
冻结协议、输入和源码不变；保留原始网络故障记录，不因观察超时重启任务。


## 2026-10-05T01:13:33.041949+00:00 anchor 对照联调首个 Solver-only 回合通过指标与路由审计

前3轮已提交：Editor累计1Actor/4Critic，Solver累计2Actor/6Critic。第3轮
Editor无Actor/Critic更新指标、无ValueNorm指标，Solver按计划1Actor/2Critic；
双方peer权重均为1。该轮8份实际环境补丁、原始提案、双方回执和奖励来源一致。
证据：runs/alfworld_anchor_retention_study_v1/SMOKE_CONTROL_FIRST_SOLVER_ONLY_AUDIT.json，
包含固定前三轮指标快照与回执哈希。前缀24环境、48生成角色轨迹、40消费，
其中8条Editor影子提案单列。此为原生指标证据，不宣称参数逐位审计或完整smoke通过。
两个训练角色与驱动仍运行；继续剩余联调及释放审计，未启动正式比较、无新准确率。

## 2026-10-05T01:22:20.023279+00:00 anchor smoke 因继承 ValueNorm 计数审计缺陷失败

Checkpoint audit omitted inherited ValueNorm update count 128. Editor native training completed but wrapper falsely failed; peer failure interrupted Solver final checkpoint save.

修正 joint_health.py：预期累计数=初始critic统计updates+本次排程critic_steps；
2个回归案例通过，并继续拒绝额外更新。原生学习算法未修改，冻结v2源保持原样。
原始smoke_control保留：48环境、96生成、72条已提交消费，保守按80条计成本。
两节点研究GPU已释放，filler保留。恢复需新根目录及新冻结源，重新运行完整配对smoke。
证据 runs/alfworld_anchor_retention_study_v1/VALUE_NORM_AUDIT_FAILURE.json；当前无活跃研究训练。

## 2026-10-05T01:24:55.619876+00:00 anchor 审计修复冻结为 v3，恢复研究 v2 已启动

当前运行根 runs/alfworld_anchor_retention_study_v2，源码 runs/source-snapshots/alfworld-anchor-retention-v3。
仅修正初始Critic继承ValueNorm累计数的检查，学习算法及输入逐字节不变；63项测试通过。
RECOVERY_MANIFEST.json 固定输入、参数等价性证据；原始 v1 失败记录不改写。
已复用32768 token验证：实际Actor/Critic引擎文件哈希不变，零新增memory计算。
驱动 alfworld-anchor-study-r2-v1 PID3868829；对照smoke Editor node3 PID2154960、
Solver node4 PID3868981，均经systemctl确认active。完成两组smoke并审计后自动正式配对比较。
此前成本按4568+失败smoke80=4648消费轨迹保守结转；全部新实验完成预计6088。
原失败smoke为72已提交消费、96生成、48环境，最后Solver未提交更新的成本差异单列。
无新准确率结论，历史92%目标及预先定义的验证协议不变。

## 2026-10-05T01:48:44.844271+00:00 anchor 对照smoke完整通过，固定H0实验组smoke已启动

当前 runs/alfworld_anchor_retention_study_v2；源码 anchor-retention-v3。
独立重算对照TRAINING_AUDIT通过：6轮，Editor 3Actor/8Critic、Solver 5Actor/12Critic；
最终ValueNorm为136/140（均含初始128），路由、团队奖励、影子提案及资源释放符合协议。
证据SMOKE_CONTROL_FINAL_AUDIT.json。新增80消费/96生成角色轨迹、48环境；
含此前失败保守计成本后累计4728消费，实验组新增另计。
固定H0实验组服务已启动并核实active：node3 Editor2173233，node4 Solver3890349，
驱动3868829继续运行。两组smoke通过后自动执行正式48轮配对实验；无新准确率。

## 2026-10-05T02:11:58.941758+00:00 两组anchor联调完整通过，正式对照48轮已启动

当前根 runs/alfworld_anchor_retention_study_v2，冻结源码 anchor-retention-v3。
PAIRED_SMOKE_FINAL_AUDIT.json独立重算两组完整审计：两组均Editor3Actor/8Critic、
Solver5Actor/12Critic；最终ValueNorm136/140（含初始128）；补丁路由、奖励、
影子提案及资源释放全部通过。两组共160消费/192生成角色轨迹、96环境；
含之前失败80保守收费后累计4808消费，正式新增单列。
正式control已从原H0/MA64及对应critic重新初始化，未使用smoke权重；
node3 Editor2191253、node4 Solver3911557均active，驱动3868829。
48轮、32joint+16Solver-only，每组640消费/768生成/384环境；固定final48评测。
control训练评测后自动anchor训练评测及固定h0诊断。尚无正式准确率；不挑中途检查点。

## 2026-10-05T02:40:45.341112+00:00 anchor正式control前12轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND12_AUDIT.json：
Editor5Actor/16Critic、Solver9Actor/24Critic，累计ValueNorm144/152（初始128）；
两端检查点、更新排程、96环境回执及奖励来源一致。前缀160消费/192生成角色轨迹，
其中32Editor影子轨迹，未来在途轨迹不混入。此前4808消费+此前缀160=4968，后续另计。
node3 Editor2191253、node4 Solver3911557、驱动3868829继续active。
未评测中途检查点；固定48轮终点、两组比较及92%目标不变。

## 2026-10-05T03:04:55.677281+00:00 anchor正式control前24轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND24_AUDIT.json。
Editor13Actor/32Critic、Solver21Actor/48Critic，ValueNorm160/176（初始128）；
24轮指标、192环境回执、实际补丁及奖励来源、检查点均通过核验。
此前缀320消费/384生成角色轨迹、64Editor影子轨迹，未来在途工作不混入。
包含此前失败保守计费的分支累计5128消费，后续另计。
node3 Editor2191253、node4 Solver3911557、驱动3868829持续运行。
仅执行训练审计，未评测中途检查点；固定第48轮主要终点不变。

## 2026-10-05T03:31:11.983892+00:00 anchor正式control前36轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND36_AUDIT.json。
Editor21Actor/48Critic、Solver33Actor/72Critic，ValueNorm176/200（初始128）；
36轮指标、288环境回执、补丁路由与奖励来源、检查点通过核验。
前缀480消费/576生成角色轨迹，其中96Editor影子轨迹；未来在途工作另计。
含此前失败保守计费累计5288消费。Editor2191253、Solver3911557、驱动3868829
经systemctl确认持续运行。未评测中间检查点，保持final48和后续配对比较。

## 2026-10-05T03:58:51.913134+00:00 anchor正式control完整48轮训练独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_FINAL_TRAINING_AUDIT.json。
独立重算与驱动TRAINING_AUDIT完全一致：Editor29Actor/64Critic，Solver45Actor/96Critic；
最终ValueNorm192/224（均含初始128）。完整384环境的补丁路由、奖励来源、
生成/消费账本及训练资源释放记录通过；640消费、768生成、128Editor影子轨迹。
含既有失败保守计费后累计5448消费角色轨迹，后续实验另计。
驱动3868829与control评测4005561均经systemctl确认active。
selection-47实际request/config均使用globalstep47的最终Solver及8份最终Engineer提案；
48题、seed6000045、temperature1、solver_tokens4096。当前尚无all200主终点结果。
固定h0实验组正式训练尚未开始；保持既定顺序与选择规则，不以中途结果调参。


## 2026-10-05T04:10:06.230642+00:00 anchor control选择完成，保留h0

CONTROL_SELECTION_AUDIT.json独立重算10份原生48题报告，覆盖及分数一致。
最佳新补丁45/48，h0为44/48，空补丁43/48；未达到至少多2题门槛，保留h0。
全部8候选有效，选择成本480环境；proposal05为0/48，仍保留其结果。
200题主评测request使用final47 Solver及h0，seed6000045、temperature1、4096tokens；
目前未完成主终点，不将选择分数当作准确率或超过92%的证据。

## 2026-10-05T04:26:42.257959+00:00 control原生200题完成，包装层GPU释放检查失败

原生h0=182/200(91%)、空补丁=174/200(87%)；逐题覆盖、seed、token预算及准确率核验通过。
selection480+主评测400环境均保留。内层RELEASED将PID4029906列为foreign，
此前gpu_process_samples明确将其识别为filler；外层及重新采样均foreign=[]且filler active。
驱动与control evaluate已经failed，anchor未启动；不得继续声称任务仍在运行。
证据CONTROL_NATIVE_RESULT_RELEASE_FAILURE_AUDIT.json。原失败文件和报告保持不变，
下一步恢复编排并复用原生结果，不重新抽样或改写失败历史。尚未完成两组比较。

## 2026-10-05T04:29:43.890026+00:00 control离线恢复完成，正式anchor已启动

原始200题h0=182/200(91%)、空补丁=174/200(87%)，完整覆盖、逐题seed/token、
哈希绑定及分数均核验。相对历史92%为-1pp，配对95%CI[-5.5,+3.5]，9改善/11退化；
未达到总目标，不将选择集93.75%当主成绩。
recover_control_endpoint.py离线发布control/matched_COMPLETE.json并绑定matched_RECOVERED.json；
原始根FAILED.json、control/matched_FAILED.json及内层RELEASED均保留，未重跑评测。
第一次离线断言发现原生config额外默认evaluation_mode/monitor_interval字段，确认原生代码后
显式纳入校验；无模型调用或部分终点发布。恢复新增训练0、评测0。
ORCHESTRATION_RECOVERY.json固定恢复脚本、原失败、结果与全部原输入哈希。
原driver.py及冻结v3源码不变，driver_recovery.py只执行未开始的anchor然后原定比较/诊断。
新驱动alfworld-anchor-study-r2-recovery-v1 PID4033201；anchor Editor node3 PID2283910，
Solver node4 PID4033353，经systemctl确认全部active。原驱动failed属历史记录；
监测新的RECOVERY_FAILED.json判断恢复分支异常。保持原48轮/640消费预算和最终选择规则。
当前已完成分支消费5448，anchor在途新增另计；两组全部完成预计6088。

## 2026-10-05T04:40:09.849809+00:00 anchor首个三轮排程核验通过

ANCHOR_FIRST_CYCLE_AUDIT.json核验两角色已提交前3轮指标；Editor0Actor/4Critic，
Solver0Actor/6Critic（仍处warmup）。第三轮Editor不更新Actor/Critic且无ValueNorm更新指标，
两端peer为unit。关联ANCHOR_FIRST_ROUTING_AUDIT的8/8实际h0和16奖励回执。
前缀40消费/48生成/24环境/8影子，累计已审计消费5488；后续在途工作另计。
尚未审计检查点或参数逐位不变，不主张效果改善。驱动4033201、Solver4033353、
Editor2283910经systemctl确认运行；完整48轮终点不变。

## 2026-10-05T05:00:57.873556+00:00 anchor正式实验组前12轮独立审计通过

ANCHOR_ROUND12_AUDIT.json核验Editor5Actor/16Critic、Solver9Actor/24Critic；
检查点及ValueNorm144/152（初始128）通过。96环境、192角色回执、实际路由及奖励一致。
前缀160消费/192生成/32Editor影子轨迹；累计已审计消费5608，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。

## 2026-10-05T05:26:40.684705+00:00 anchor正式实验组前24轮独立审计通过

ANCHOR_ROUND24_AUDIT.json核验Editor13Actor/32Critic、Solver21Actor/48Critic；
检查点及ValueNorm160/176（初始128）通过。192环境、384角色回执、实际路由及奖励一致。
前缀320消费/384生成/64Editor影子轨迹；累计已审计消费5768，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。

## 2026-10-05T05:51:28.000495+00:00 anchor正式实验组前36轮独立审计通过

ANCHOR_ROUND36_AUDIT.json核验Editor21Actor/48Critic、Solver33Actor/72Critic；
检查点及ValueNorm176/200（初始128）通过。288环境、576角色回执、实际路由及奖励一致。
前缀480消费/576生成/96Editor影子轨迹；累计已审计消费5928，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。

## 2026-10-05T06:15:35.904515+00:00 anchor正式48轮训练独立审计通过，终点评测启动

ANCHOR_FINAL_TRAINING_AUDIT.json独立重算与驱动一致，两组训练预算相同。
Editor29Actor/64Critic、Solver45Actor/96Critic；ValueNorm192/224（初始128）。
384环境路由和奖励、640消费/768生成/128影子轨迹及训练释放记录通过。
分支累计消费6088（包含失败保守计费），恢复无新增评测。
Engineer和Solver服务成功退出；驱动4033201持续运行，anchor评测PID4126635 active。
固定final48选补丁及200题主评测正在运行；目前仅control成绩91%/87%，anchor无准确率。

## 2026-10-05T06:30:13.238620+00:00 anchor选择完成，保留h0，配对主评测配置一致

ANCHOR_SELECTION_AUDIT.json重算全部9份原生48题报告，对应10逻辑变体。
最佳新补丁43/48，h0为42/48，空补丁40/48，未达到多2题门槛；7/8提案有效，
无效proposal06按既定规则复用空补丁报告，实际选择成本432环境。两组选择共912环境。
两组主评测使用相同h0及相同200题、seed、temperature、token/并发预算，final47 Solver。
实验组200题已提交，无准确率结果。两组均选h0，后续固定h0诊断可复用主结果。

## 2026-10-06：复用历史SAO，完成训练加速和单项MA-SAO调优

按用户新优先级，不启动新的SAO对照训练、不增加训练种子或铺开消融。
已在途的种子1000043先按原48轮方案完成：H0组合185/200（92.5%），
无提示181/200（90.5%）；相对历史92%差+0.5个百分点，配对95%区间[-3, 4]。
原人工提示扩展暂缓，未更改或抹除原冻结设计。

### 加速先行

原Qwen3.5训练使用PyTorch gated-delta-rule回退。隔离安装FLA0.5.2和Triton3.7.1，
不修改活动虚拟环境。FLA与旧Triton3.6组合被Hopper反向正确性guard拒绝；
升级后完整32K首次Actor反向OOM。保留两次失败，仅将固定reference存储
由FP32改为BF16后，W0/H0完整32K原生验证通过；Actor/Critic master weights仍为FP32。
前向、梯度、初始logprob/value误差、实际参数更新和GPU释放均通过预设检查。
最新定向CPU测试45项通过。

- 完整32K合成Actor+2Critic稳态更新：346.94→39.17秒，8.86倍。
- 真实联合训练相同稳态轮次均值：96.90→49.56秒，1.96倍。
- 成功尝试从两角色启动到全部完成：101.09→60.00分钟，1.68倍。
- 真实轨迹token长度不同，并减少中间checkpoint保存；不能把所有端到端收益
  归因于FLA，不能把8.86倍写成整轮或整次实验加速。
- 成功尝试墙钟不含加速验证、失败首尝试、驱动输入重哈希与评测。

### 唯一新训练候选及预先固定终点

同种子1000043、H0/W1和MA64 critics起点、数据、640消费/768生成/384环境预算，
Engineer LR 1e-7→5e-7，Solver LR 1e-6→5e-7；其余KL0.02、K2、3 active warmup、
joint_joint_anchor调度和上下文/token预算保持。两项LR和运行时作为一个整体方案，
不做组件归因。仅保存并评估final48，8提案、选择48题、至少比H0多2题才能替换。

首尝试因rollout冷编译超过300秒启动等待失败，零提案/环境轨迹/优化更新，
失败与释放记录保留。仅增加可选rollout_setup_timeout=1800，默认行为不变；
恢复分支从同一初始权重和fresh Adam开始，不从部分优化器状态恢复。
恢复训练Editor29 Actor/64 Critic、Solver45 Actor/96 Critic更新，全部48轮审计通过。

### 主结果与结论

| 系统 | all200 |
| --- | --- |
| 历史SAO W1+h2 | 184/200，92% |
| 同种子原MA-SAO + H0 | 185/200，92.5% |
| 新MA-SAO + 自动proposal06 | **187/200，93.5%** |
| 新Solver + 空策略 | 181/200，90.5% |

8个Engineer提案中5个有效。proposal06选择集48/48，H0为46/48，达到固定门槛；
主成绩不是选择集100%。无人工修改或提示拼接，主评测harness_error_episodes=0。
选择共7个独立策略×48题，主评测2×200题，总计736次新增环境任务。
相对历史SAO：+1.5个百分点，95%配对区间[-1.5, 5]，7题改善/4题退化。
相对同种子原MA-SAO：+1个百分点，区间[-2.5, 4.5]，8题改善/6题退化。
同一新Solver下，自动策略相对空策略为+3个百分点，区间[0, 6.5]；
没有新增同一新Solver配H0的200题消融，不能单独证明Engineer训练贡献。

**本轮获得高于历史92%的自动策略点估计，但预设优势验证未通过。
“MA-SAO优于独立交替SAO”仍未确立，也没有被这一配置逻辑证伪。**
单训练种子、历史对照、重复开发200题、整体方法包调整均限制解释。
不根据200题结果继续挑checkpoint或扫描种子，不用自适应筛选后的区间冒充确认性证据。
完整原生报告、选择规则、输入/提示哈希、题目/种子/token和配对统计已独立复核；
服务全部成功退出，两节点无剩余研究GPU进程，filler正常。

- [加速验证](../runs/alfworld_training_speed_v4/COMPLETE.json)
- [原配置主结果](../runs/alfworld_explicit_hint_training_replication_v1/repeat/matched_COMPLETE.json)
- [候选结果](../runs/alfworld_ma_balanced_fast_v1r2/RESULT.json)
- [独立复核与墙钟数据](../runs/alfworld_ma_balanced_fast_v1r2/INDEPENDENT_AUDIT.json)
- [完整实验结论与模型入口](../runs/alfworld_ma_balanced_fast_v1r2/CONCLUSION.md)

## 2026-10-07：MA长期曲线及传输故障恢复

研究顺序改为先观察并调优长期平台，再冻结方案比较相同样本量，最后归因。
历史固定h2的W2延长0/20/40/60/80步为63.5/71/79.5/38/49%，
重新核验全部10份报告；结论是先恢复再崩落，而非SAO已稳定收敛到92%。

MA从自动策略93.5%的round48出发，明确一次fresh Adam边界；
后续每96轮通过FP32 model-only DCP和原生逐rank Adam分片续训，
保留ValueNorm及原H0/W1 reference。CPU下一更新一致性及H0/W0完整32K
GPU往返验证通过。每48轮评测，至少192额外轮才判断观测平台；
最大768额外轮/24小时不是收敛保证。全部中间点保留，不挑最佳点冒充终点。

v1r3正式训练完成30轮后发生Solver到环境controller的HTTP ReadError，
两角色梯度有限，未到第48轮checkpoint。保留400已消费角色轨迹，
248提案、246完整环境结果、491完整角色回执（11条未提交更新）的失败成本。
不将其作为性能下降，也不假装从未保存的优化器继续。
禁用该层keepalive，沿用已有controller到worker的实现；
不重试非幂等动作POST，不将传输异常转换为模型失败奖励。
28项定向测试及13个subtest通过。

v1r4从同父权重、同种子、同数据重新启动，复用已通过的原生handoff验证。
02:30 UTC确认两节点原生进程进入初始化，尚无新增优化更新或评测成绩。
源码和设计独立冻结，v1r3全部失败记录保留。

当前为两个独立Actor和两个独立Critic，Critic有额外协作上下文但无参数共享；
不能把93.5%归因于共享团队Critic。用户同意先完成此冻结方案的长期曲线，
之后才根据价值估计质量等诊断选择一个有依据的改进。

- [冻结恢复设计](../runs/alfworld_ma_convergence_v1r4/DESIGN.json)
- [阶段门控](../runs/alfworld_ma_convergence_v1r4/RESEARCH_PROGRAM.json)
- [失败成本](../runs/alfworld_ma_convergence_v1r3/FAILED_COST.json)

### 04:41 UTC：首个长期指定终点持平

v1r4完成96额外轮训练，消费1280角色轨迹、生成1536角色轨迹、
768环境episode。Editor64/128及Solver96/192次Actor/Critic更新、
每rank原生Adam步骤通过审计。训练奖励按24轮窗口为
151/192、159/192、152/192、150/192，并非持续单调上升。

先评总round96（local48）：保留父自动proposal06，187/200=93.5%；
空策略183/200=91.5%。相对历史92%配对差+1.5pp[-2,5.5]，
相对父93.5%差0pp[-2.5,3]。只有一个新增点，不能宣称平台或整体继续提升。
总round144（local96）正在评测，保持冻结长训练及停止规则。

用户再次明确三个问题：同rollout的SAO/GRPO、SAO延长曲线、MA延长及更高平台。
前两项保留已测答案及局限，不为迎合方向改写为SAO普遍更好或已达绝对上限；
新增资源集中于第三项。新增最终结果核验器13项测试通过，
不修改运行中的冻结源码，真实曲线完成后再运行独立核验。

### 05:02 UTC：总round144达到97%，继续观察而非高点停止

第二个预定终点（local96、总round144）为194/200=97%，
同Solver空策略185/200=92.5%。8个提案全部有效，
proposal07选择集48/48，继承proposal06为46/48，达到固定多2题门槛。
提示来自本次Engineer自动输出，无人工改写，未根据all200选提示。
最终提示SHA256为`3e4ac25391bbb1e69c5831f76e8e7ade217ab3ac3766bd3cee2eda5723adc4fa`。

相对历史SAO92%：+5pp，配对95%区间[2,8.5]，11题改善/1题退化。
相对父93.5%：+3.5pp，区间[1,6.5]，7题改善/0题退化。
原生矩阵、指定检查点、选择规则、compact回执及配对统计已复核。
这支持当前候选继续训练能够提升，但只有一个97%点，
尚无稳定平台证据；重复开发评测和历史对照限制仍在。
第二个96轮块按冻结协议继续，不在高点停止或改写停止规则。

- [总round144结果](../runs/alfworld_ma_convergence_v1r4/block-01/round0096_COMPLETE.json)
- [完整曲线](../runs/alfworld_ma_convergence_v1r4/CURVE.json)

### SAO Weight退化机制与Harness-only问题：只读复核

用户要求继续监控MA，同时解释普通SAO的Weight collapse，以及固定Weight只进化
Harness的可能结果。没有为此启动额外GPU实验，也没有修改在途MA配置。

现有证据支持特定SAO续训配置损害Solver能力，但没有唯一根因：
固定同h2，W1为92%、W2为63.5%；空策略W1为87%、W2约50%，
所以不是只换了坏提示。W2恢复的initial/extra20/40/60/80固定h2报告，
截断率均0、Harness错误均0，工具错误总数分别1/0/1/1/2；
extra40到extra60的79.5%到38%退化不是大规模工具格式错误或token耗尽。

重新读取原W2和恢复W2各80轮metrics：原W2 Actor训练奖励日志按20轮窗口
约88.3/87.5/79.4/63.1%，恢复约71.7/83.1/71.3/40.6%
（首窗口Actor指标不含5轮Critic-only warmup，不等于全部episode均值）。
没有支持“熵单调降到零”的现象，恢复末段记录的熵反而上升；
梯度和value loss也未直接给出唯一崩塌原因。
普通配置Actor LR2e-6、Critic1e-5、batch8、K1、无固定reference KL，
累计策略漂移和终局奖励下的价值/信用分配误差是待验证解释，不是已证实归因。
旧多轨迹递归状态污染确实存在，但独立轨迹修复后仍再次退化，
不能把全部collapse归于该问题。h2在W2阶段固定，不能直接归因于队友持续变化。

固定W1的已有H1/H2为90.5/92%，差+1.5pp[-1.5,5]，是短程正向点估计，
尚无固定优质W1、仅训练Harness的长期平台结论。
Engineer自身也曾在更晚检查点出现候选质量/有效率下降；
保留旧Harness可以保护已部署组合，但不能等同于Engineer持续学习。
后续若做Harness-only，必须区分候选分布质量与incumbent选择后的系统曲线；
不抢占当前MA收敛实验资源。

当前MA调度明确为两轮H+W联合更新、一轮仅W更新，而非长块H/W交替。
同episode中H先提案、W再执行是因果执行顺序，不代表分阶段训练。
两个Critic参数独立，仅增加协作上下文。
第二块96轮已完成并通过审计：累计H Actor128/Critic256，
W Actor192/Critic384；4个rank计数一致，实际从前块恢复Adam而非重置。

### 07:13 UTC：总round192为96%，保留已有自动策略

第二块local48终点192/200=96%，同Solver空策略180/200=90%。
8个新提案全部有效，但未达到替换门槛，按selection规则保留总round144的proposal07。
相对历史SAO92%差+4pp，配对95%区间[1,7]。
当前93.5/93.5/97/96%的完整曲线不支持单调上升叙事，
但97%后的下一点仍高于历史92%；平台判据仍未满足。
空策略从92.5%回到90%，不能把组合提升全归因于Solver权重单独变强。
总round240正在评测，不依据此波动热改训练或抢先做组件消融。

### 07:41 UTC：总round240回落到94.5%，未达平台

第二块local96终点189/200=94.5%，空策略185/200=92.5%，继续保留proposal07。
相对历史SAO差+2.5pp[-1,6.5]，相对父93.5%点估计仅+1pp。
第二块两个终点均重新验证原生矩阵、选择及配对统计。
当前总round48/96/144/192/240为93.5/93.5/97/96/94.5%，
最后四点范围3.5pp；满足最低192额外轮但不满足平台规则，
末端区间也未通过优势门槛。第三块按冻结方案启动，
97%只能作为观测最高点，不能替代长期结论。

### 09:46 UTC：总round288为93%，继续记录回落

第三块local48终点186/200=93%，空策略184/200=92%，保留proposal07。
相对历史SAO为+1pp[-2.5,4.5]。完整曲线高点97%后为96/94.5/93%，
最后四点范围4pp，仍未满足平台规则；总round336正在评测。

仅分析已有日志，没有新增消融：三个训练块平均固定reference KL，
H约0.0027/0.0063/0.0103，W约0.0024/0.0039/0.0051；
Critic原始MSE没有越过健康阈值。round240和288选择集新提案最佳47/48，
incumbent46/48，未达到多2题门槛，不能根据200题结果反向替换。
Solver空策略并未同步大幅崩落，候选分布质量也非单调上升；
需要在完整曲线后区分Weight漂移、Harness配合和选择限制，
不能据健康loss断言泛化稳定，也不能把相关诊断冒充因果解释。

### SAO原始轨迹复核：环境步数耗尽与动作语义退化

在node3以原文件owner读取extra40/60两终点的4份原生报告，重新配对200题，
保存原报告SHA256及行为统计，未新增GPU轨迹或训练。
固定h2从159/200到76/200：88题由成功转失败，5题反向改善；
88个退化样本中69个运行到50次环境交互仍未成功。
空策略从166/200到92/200：78题退化、4题改善，退化中72个走满50步。
这与4096-token截断是不同限制；原统计truncated_fraction=0并不排除环境步数耗尽。

两例逐动作比对：
- 无策略task240原7步完成拿取并放置toiletpaper，extra60输出不存在的
  `toiletp_paper` / `toiletp_paperhanger`，随后反复导航到50步失败。
- 固定h2的task247原4步完成拿creditcard并使用desklamp，extra60拿卡后
  循环搜索shelf1..5，还输出`go to desk (if exists)`，50步失败。

extra60固定h2的124个失败episode终止状态包括98个completed、
25个agent invalid action、1个invalid_tool_arguments；
completed是流程结束而非任务成功。工具JSON错误少不能推出动作语义正确。
这些证据定位了已观察到的失败行为，不证明高学习率、KL缺失、
Critic误差或数据打包中的某一项是唯一参数更新根因。

- [只读行为审计及来源哈希](../runs/alfworld_ma_convergence_v1r4/SAO_FAILURE_MODE_AUDIT.json)

### 用户要求改为每轮H/W联合更新：独立冻结新候选

核验旧第四块70轮：H47/W70次Actor更新，H在联合轮次梯度非零，
不是后期完全冻结H；实际为旧锚定对照继承的2:3混合调度。
用户明确要求两者一起更新，因此不再默认该配比最佳。

在旧块73轮时仅停止CPU协调器，两个原生角色服务均保持active。
新的边界执行器完成旧第四块及总384/432评测，不启动第五块；
若原平台规则未满足，标记用户指定换方案而非收敛。
旧源码、数据、优化规则及已生成结果均不改写。

新simultaneous候选从同一93.5%父round48起步，不选97%高点初始化；
每轮H/W和两个Critic均更新，去掉anchor路由和Solver-only轮次，
bounded队友权重用于全部轮次。保持原学习率、KL、奖励、原始reference、
任务/prompt顺序、选择/评测及平台规则。
每96轮H/W各96Actor/192Critic，消费1536而非1280角色轨迹；
生成1536、环境768不变。多20%消费与队友权重规则变化均明确披露。

新增审计拒绝隐藏的H/Critic停更和anchor路由；结论审计支持明确用户边界，
不会把中止当作平台。包括真实输入准备、任务顺序、每轮双更新和Adam状态的
59项定向测试通过。新设计、源码快照、输入/原生验证哈希已冻结，
CPU协调器启动核验并等待旧实验FINAL_AUDIT和资源释放，尚无新训练成绩。

- [新候选设计](../runs/alfworld_ma_simultaneous_convergence_v1/DESIGN.json)
- [新候选执行器](../runs/alfworld_ma_simultaneous_convergence_v1/driver.py)
- [旧边界请求](../runs/alfworld_ma_convergence_v1r4/REQUESTED_BOUNDARY.json)

### 用户指定从97%原生检查点继续，并核查高分是否对应能力提升

用户在新93.5%起点候选尚未开训时改为保留总round144的97%点。
立即停止该CPU协调器，核查无STARTED、无block、零研究轨迹；
保留冻结文件并写明[取消原因](../runs/alfworld_ma_simultaneous_convergence_v1/CANCELLED_BEFORE_TRAINING.json)。
旧混合实验仍只完成第四块收尾，不再第五块，也不热改已运行的训练。

新[97%起点方案](../runs/alfworld_ma_simultaneous_from97_v1/DESIGN.json)从旧block01末的
完整原生状态恢复两个Actor/Critic、ValueNorm和Adam，H初始步数64/128、
W96/192。没有新增Adam重置，不使用HF舍入值代替FP32 learner恢复；
原H0/W1 reference保持不变。从source768续接，全部轮次H/W均更新。
首块期末应为H160/W192次累计Actor更新，Critic分别320/384；
每新块1536消费、1536生成、768环境episode，继承成功谱系1920消费另列。

平台门槛相对于新144起点，至少再192轮，最早总336；不能仍从48计数。
源代码、设计、HF权重、原生metadata和逐rank指纹已冻结，CPU协调器已核验为active，
仅等待输入核验、旧边界审计和node3固定交叉验证后才占GPU训练。
70项定向测试通过，且更严格的Adam审计已通过旧前三块真实状态。
这是用户选择观察高点初始化的探索性试验，不能掩盖选择偏差和此前研究成本。

原97%逐题核查：同W144空策略185/200、自动P144为194/200，改善9题、退化0题；
+4.5个百分点的题目配对95%区间[+2,+7.5]。7个改善目标涉及物体变换后放置，
2个涉及取两个物体；真实消息包含拿取提示、先清洗/加热/冷却及阻止过早放置等。
strict197仍为191/197对183/197。空策略W48到W144仅+2个百分点，
区间[-1.5,+5.5]，不能把全部组合收益归因于Weight学习。

选中策略与Engineer的原始生成receipt对象、规范化patch哈希及checkpoint一致，
不是事后手写提示。对两个Engineer检查点的8组生成消息逐一比对完全相同；
合法提案5/8变为8/8，但新候选中仍有35/48的低分提案。
因此“一个自动策略有效”与“Engineer整个提案分布因RL变好”必须分开。
核验请求摘要采用系统原有canonical contract digest，而不是JSON文件字节哈希；
审计程序初次用错摘要类型被断言拒绝，修正后重新执行，不修改原始证据。

在node3排队启动一次[固定交叉验证](../runs/alfworld_peak97_crossplay_v1/protocol.json)，
复用原推理seed，对W48/W144交叉空策略、初始H0、P48、P144并重测对角线；
8个逻辑系统、1600环境episode，零训练轨迹。诊断不重新选择续训起点，
不替代完整续训曲线，也不开展多训练种子扫描。

进一步逐项检查split的实际ID集合：train240与selection48/report200无任务ID交叉，
但selection48全部包含在report200，不能将二者表述为独立选择集和测试集。
保留首份行为计算，另生成增加该发现和敏感性统计的v2审计。
排除selection48后，新策略146/152=96.05%、空策略141/152=92.76%，5改善/0退化；
差+3.29个百分点，题目配对95%区间[+0.66,+6.58]。
再排除3个训练目标文本重合项，143/149对139/149，4改善/0退化，
差+2.68个百分点，区间[+0.67,+5.37]。这表明原始收益不只发生在被用来选策略的
48题中，但仍是事后开发集诊断；总144轮检查点由完整200题曲线挑出，
不构成新独立测试或多训练种子稳健性证据。

### 12:41 UTC进展：固定交叉验证完成，97%起点续训仍等待旧最终评测

交叉验证完成1600环境episode、零新增训练轨迹，并释放node3资源。
重新核验两个原生矩阵、八份原生报告、compact逐行/哈希和环境成本，全部一致。
W48在空/H0/P48/P144下分别90/91.5/95/95%，
W144分别91.5/92.5/95/96.5%。原97%组合复测96.5%，而原93.5%组合复测95%；
不再把原来单次+3.5个百分点差值当成已经复现的训练收益。
新旧完整组合差+1.5个百分点，4改善/1退化，配对95%区间[-0.5,+4]。
固定P48时两种W均95%，固定W144时P144比P48高1.5个百分点、区间[-1,+4]。

固定W144，新策略相对空策略+5个百分点，10改善/0退化、区间[+2,+8]；
排除选择48题后仍6改善/0退化，+3.95个百分点、区间[+1.32,+7.24]。
这支持自动策略的实际开发集效用，不足以证明Engineer生成分布整体改善、
独立Weight学习优势或未选择的确认性泛化优势。

旧混合方案已完成总384轮评测：94.5%，空策略同94.5%。
最近四点94.5/93/94.5/94.5%已触发原观测平台条件，但预定总432轮末评还在运行，
最终停止标签及全曲线审计需等该端点完成。97%原生续训协调器已完成输入核验，
等待旧FINAL_AUDIT后自动开始；此时无STARTED、无训练block，不能称已经续训。

本次状态询问期间，旧总432轮末评随后完成：带P144为95%，空策略95.5%。
最后四点93/94.5/94.5/95%，跨度2个百分点，按预定规则停止为
`operational_development_plateau`；原97%最高点不是稳定平台高度。
终点对历史92%差+3个百分点、9改善/3退化，题目配对95%区间[0,+6.5]；
优势门槛要求下界严格大于0，因此没有通过，不能提前进入确认性样本效率结论。
旧边界结束后FINAL_AUDIT尚未生成且无审计进程；直接执行完整审计并通过，
核对全部固定端点、选择、预算及四块原生Adam连续性，解除97%续训队列依赖。
新执行器自身包含最终审计，不依赖旧边界的附着监控脚本来完成该收尾。

## 按研究问题组织的实验报告（2026-10-07）

术语更正：原MA-SAO已经在联合学习，只是H:W更新比为2:3；
新候选提高到1:1，并同时改变原第三轮的队友校正。不能描述成
“原来不是联合训练，现在才联合训练”，也不能将其视为纯更新频率消融。
以下按科学问题归类；配置变体与失败尝试仍保留在各自原始记录中。
未完成实验只能报告设计和状态，不能把希望得到的效果写成结论。

### 实验1：相同Engineer rollout下，SAO是否优于GRPO？（已完成）

**研究问题。** SAO相对GRPO的优势是否在匹配策略训练样本量后仍然成立？

**实验设计。** 从相同H0出发，匹配192条Engineer训练轨迹；
分别在固定W1、W0下评测保存的策略。这里匹配的是Engineer消费预算，
不是全部环境交互、计算量与checkpoint选择过程完全相同。
当前复用已有实验，不重复训练。

**实验结论。** W1上SAO90.5%、GRPO88%，差异区间跨零；
W0迁移上SAO74%、GRPO83.5%。只支持W1设定下SAO点估计略高，
不支持SAO普遍优于或不差于GRPO，也不能将不显著差异当成统计等效。

### 实验2a：退化后的SAO能否通过延长Solver训练恢复并超过92%？（该分支已完成）

**研究问题。** 后期下降是暂时波动，还是在已测延长预算内仍无法恢复？

**实验设计。** 从真实退化W2出发，固定h2，只续训Solver；
测额外0/20/40/60/80步，同时评测带h2与空策略。
这次恢复有fresh Adam边界，并修正独立轨迹打包；
不是从92%点继续完整H/W交替周期，也不是精确优化器续跑。

**实验结论。** 固定h2为63.5/71/79.5/38/49%，空策略为49/74.5/83/46/44.5%：
先恢复、后再次崩落，没有回到92%。这否定了“在这段预算内持续缓慢恢复”的描述，
但不证明所有SAO配置都已收敛，也不证明92%是SAO的绝对上限。
来源：[延长训练结果](../runs/alfworld_w2_recovery_v2/COMPLETE.json)。

### 实验2b：固定较强Solver，Engineer再训练一阶段是否继续提升？（用户要求补做）

**研究问题。** 只观察固定h2后W2的退化，不能推出固定优质W1时Engineer也不能
继续学习。检验在W1不变时，H2再经一完整SAO阶段后的候选质量和部署效果。

**实验设计。** 从历史H2-step31 Actor/Critic出发，W1固定为原step79权重，
恢复一阶段32 native step、batch6、192消费轨迹，27 Actor/32 Critic更新，
原LR2e-6/critic1e-5、K1及5步Critic预热。没有保存的Adam，
明确fresh Adam；采用已有普通SAO后端修复和已验证内核加速，不加入MA/KL/ValueNorm。
训练使用同W1的原训练失败证据，新sample编号384起，不重放旧生成轨迹。
只评最终extra32点，续训前后分别生成8组完全匹配的prompt与推理seed，
比较selection24 mean@2的完整提案均值/合法率及相同门槛下的部署选择。
主200题评测before/after、历史h2、空策略及新最好合法候选；
保留非法槽、负结果和未被替换的候选，不能把回退h2当学习成功。
只有一个新训练阶段，没有独立多训练seed扫描。

**实验结论。** 当前尚无结果。前一条“实验2已完成”仅适用于固定h2续训W，
不涵盖本分支，也不等于SAO两角色都已达到不可提升的上限。
为优先得到结果，在当前MA96轮原生块完整结束后插入node3的单阶段实验，
仅暂停CPU调度器，不中断原生learner；之后恢复原MA流程。

### 实验3：MA-SAO长期训练能否形成高于历史SAO的稳定平台？（已完成旧调度）

**研究问题。** 不先限制相同rollout，观察MA-SAO能达到多高、是否能保持，
而不是只比较一个最好的短程点。

**实验设计。** 从93.5%的MA第48轮检查点续训，保留2:3更新配比、
原始reference和超参数；每48轮固定评测，完整报告至总432轮。
第48轮有一次已披露的fresh Adam边界，此后原生保持状态。
预定平台规则要求至少192额外轮、最近四点范围不超过2个百分点，
且三个间隔没有至少0.5个百分点的新纪录。
历史SAO92%仅作参考，不是同初始化、同预算的因果对照。

**实验结论。** 最高97%，最后四点93/94.5/94.5/95%，终点95%，
满足观测开发平台规则，但不是稳定97%或持续单调上升。
终点对历史92%为+3个百分点、95%配对区间[0,+6.5]，
未通过预先要求下界严格大于0的优势门槛。
末期空策略95.5%，不能简单归结为Solver一直变差。
来源：[完整曲线及审计](../runs/alfworld_ma_convergence_v1r4/FINAL_AUDIT.json)。

### 实验4：97%的高点来自真实策略效用，还是权重、搜索与测量波动？（已完成诊断）

**研究问题。** 区分一个保存的Harness有效、Solver能力提高、
Engineer整个提案分布改善，以及单次高分的波动。

**实验设计。** 固定W48/W144，与空策略、初始H0策略、P48/P144交叉，
共八个系统、1600环境episode，使用原推理seed重测对角组合。
不新增训练种子，不据此重新挑续训检查点。
检查真实动作与提示、生成receipt、训练/选择/报告ID，并报告排除选择48题后的敏感性。

**实验结论。** 原97%组合复测96.5%，原93.5%组合复测95%；
新旧完整组合差+1.5个百分点、区间[-0.5,+4]，训练增益尚不明确。
固定W144，新策略比空策略高5个百分点，10改善/0退化、区间[+2,+8]，
排除选择48题后仍6改善/0退化，支持该自动Harness的实际开发集效用。
固定旧P48时W48/W144都为95%；同组生成输入下合法提案5/8到8/8，
但仍有低质量候选，不能宣称Engineer整体生成能力已经稳定提高。
selection48全部在report200内，且97%检查点是看过完整开发曲线后选择的；
复测和事后子集分析都不是新的独立测试确认。
来源：[交叉验证](../runs/alfworld_peak97_crossplay_v1/COMPLETE.json)、
[行为及数据重合审计](../runs/alfworld_peak97_crossplay_v1/BEHAVIOR_AUDIT_v2.json)。

### 实验5：提高Engineer更新比重并统一协作校正，能否改善高点后的稳定性？（进行中）

**研究问题。** 在既有联合学习框架内，旧2:3配比是否限制了协同学习与高点保持？
检验的是一个调度候选，不是“联合训练是否优于非联合训练”。

**实验设计。** 按用户要求从总144轮的97%点恢复双Actor/Critic、ValueNorm及原生Adam，
保留原H0/W1 reference。每96轮由H64/W96次Actor更新变为H96/W96，
即H更新增加50%，W更新次数不变；Critic随各自角色更新。
消费轨迹由1280变1536（+20%），生成1536和环境768不变。
同时将原Solver-only第三轮的单位队友权重改为bounded校正，
因此结果不能单独归因于H更新频率。按冻结设计从source768续接，
平台最低192新增轮从144起算，最早总336才有资格判断。

**实验结论。** 已启动原生恢复与训练流程，尚无新主评测，结论未定。
既不能说1:1已优于2:3，也不能说已经把97%变成更高平台。
选择观察高点初始化的偏差和额外20%消费成本均须保留。
来源：[冻结设计](../runs/alfworld_ma_simultaneous_from97_v1/DESIGN.json)。

### 实验6：稳定候选在相同样本量下，是否仍优于交替SAO？（待开展）

**研究问题。** 更高的绝对高度是否也意味着更高样本效率，而不是仅靠更多训练或更强初始化？

**实验设计。** 在获得并冻结较好的长期候选后，对齐初始化或明确计入继承训练成本；
预先区分消费角色轨迹、生成轨迹、环境episode和候选搜索/评测预算，
在匹配预算点比较MA-SAO与交替SAO。已有SAO仅在合同匹配时复用，
当前不因该问题额外启动SAO或多训练种子扫描。

**实验结论。** 尚未开展，不能用当前长期高点或1536对1280的调度结果，
声称MA-SAO在相同rollout下已经更好。它与实验3的长期高度问题是两个不同问题。

### 工程实验：训练加速是否成立且保持既定更新流程？（已完成）

**研究问题。** 能否降低实验墙钟，而不是通过少训练、截短上下文或改变奖励来“加速”？

**实验设计。** 启用经过验证的FLA/Triton内核，仅降低固定reference存储精度，
保留Actor/Critic FP32 master、32K上下文与原更新流程；
进行完整原生梯度、内存和状态恢复验证，并分别测合成优化器与真实训练。

**实验结论。** 合成稳态更新8.86倍，真实稳态轮次1.96倍，成功训练总墙钟1.68倍。
真实轨迹长度及保存频率也影响墙钟，不能将8.86倍写成端到端RL加速，
也不能把速度收益当作MA-SAO效果优越的证据。

### 2026-10-07 15:04 UTC：明确后续三项优先级

用户规定先完成固定Solver续训Engineer的SAO瓶颈检验，再完成高点起步的
MA-SAO长期高度/稳定性检验，最后才开展WHALE的同样本量复现与评测。
较低优先级的MA同预算分析和机制消融暂不插队。

优先项1的H3已完成23/32 native step，仍无主评测成绩。
一次新增192条Engineer轨迹不足以证明绝对上限；正、负结果都需要区分
候选分布、保存策略效用和开发集不确定性。
优先项2仍使用用户之前指定的真实高点：原曲线与冻结设计是97.0%、总144轮，
而非本条用户提到的97.5%。不修改原始分数或检查点身份。
正在等待的MA首块96轮已完整结束，核验原生Adam累计H160/W192次Actor更新通过，
两个Critic状态也保留。H3实验完成并释放GPU后恢复同一MA协调器。
最终MA研究同时参考本次固定W1新结果，不能无视更强SAO而只对比历史92%。

优先项3尚未启动。WHALE预算需与明确的SAO阶段/初始化口径绑定，
不能只匹配“1024”这个数而省略候选评分环境成本。
如使用当前助手的可调用后端作为proposer，需授权入口、隔离会话、
实际模型身份和token成本记录；强proposer替换的系统收益不能全归因于算法。
已有WHALE适配尚无完整验证的基线结果，之前8-12小时顺利估计/12-24小时规划
仅是条件性工期，并非完成或优势承诺。

## 2026-10-07：固定W1继续Engineer完成，补齐SAO续训结论与完整实验报告

**研究问题。** 固定历史较强W1，从H2再训练一个历史大小的Engineer阶段，
能否改善候选质量和任务成功率，解决此前只检验Solver方向的证据缺口？

**实验设计。** 从真实H2-step31 Actor/Critic继续32 native step、batch6、
192条消费，前5步Critic预热；W1零更新。H2没有Adam，显式fresh Adam；
使用修正普通SAO后端，无MA/KL/ValueNorm。只评最终extra32，不搜索中间终点。
前后各8个匹配prompt/seed候选，selection24 mean@2、同一旧h2 incumbent
与2个百分点替换门槛。新最好合法候选即使未被选中也必须主评。

| 指标 | 续训前/对照 | 续训后/新候选 |
| --- | --- | --- |
| 合法候选 | 2/8 | 8/8 |
| 八槽平均成功率 | 88.8021% | 89.5833% |
| 最好新合法候选的选择分 | 93.75% | 93.75% |
| 部署选择 | 旧h2 | 同一旧h2 |
| 部署系统主200题 | 89% | 89%，同一物理报告别名 |
| 旧h2对新最好合法候选 | 89% | 86.5% |
| 空策略对新最好合法候选 | 86.5% | 86.5%，逐题行为不同 |

**实验结论。** 格式合法性改善，但没有可靠的任务效果提升。
候选均分差+0.78125pp，95%区间[-4.4271,+5.7292]；
新最好候选对旧h2为-2.5pp，区间[-7,+2]，8改善/13退化；
对空策略为0pp，区间[-4.5,+4.5]，10改善/10退化。
前后部署和旧h2共用策略与物理评测，其零差异不能解释为两个模型统计等效。
本次89%复测不覆盖历史92%，也不降低MA目标。
与固定h2续训Solver的恢复后再退化结合，只能说明当前两种续训方案和追加预算
没有可靠突破，不能声称Engineer/Solver完全不能学习或已证明SAO绝对上限。

训练与成本独立审计通过：32 native/27 Actor/32 Critic，27次Actor梯度均非零；
192条声明消费、198条记录生成、503089个训练生成token；
1120个训练奖励episode、1176个选择/主评episode。
六条额外生成来自异步预取，无法任意指定重复记录中哪一条实际被消费。
所有候选、原生主评、哈希、选择别名和配对统计复核后，FINAL_AUDIT已持久化为passed。

H3于15:58完成并释放GPU，原MA协调器恢复。16:19实查，首个96轮新MA原生块
已完整保存，Actor Adam step H160/W192、Critic320/384审计通过；
正在评新增48轮（总192轮）端点，尚无新完成成绩。原24小时墙钟不延长。
WHALE尚未启动，不在报告交付时把整个研究目标标记完成。

已整理独立的十二问题实验报告，保留SAO–GRPO限制、两个SAO续训方向、
原始MA和负面消融、人工提示验证、旧MA完整432轮曲线、97%行为与交叉归因、
新调度进行中状态、加速倍率边界及后续匹配预算计划。

- [完整实验报告](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md)
- [H3完整结果](../runs/alfworld_fixed_w1_h3_v1/COMPLETE.json)
- [H3最终审计](../runs/alfworld_fixed_w1_h3_v1/FINAL_AUDIT.json)
- [MA恢复收据](../runs/alfworld_fixed_w1_h3_v1/MA_RESUMED.json)

## 2026-10-07：七小时交付窗口、MA早期下降及WHALE正式运行

用户于16:37:12UTC要求：H3真正新候选200题及WHALE结果须在23:37:12UTC前完成，
MA提供早期结果，之后可以继续。不得再以MA完整收敛为WHALE启动前提。
已重新核验H3 `after_candidate` 原始200题、输入/策略/模型和SHA：173成功，
86.5%，格式合法率100%，harness错误episode为0；不是旧h2的178/200别名。

MA首个新96轮保存完整；每轮两侧都更新，Actor H160/W192、Critic320/384。
总144/192/240轮带策略为97/93/92%，空为92.5/90.5/89%。
总240相对父点-5pp，95%[-8.5,-2]，0改善/10退化；对历史92%为0pp，
[-3.5,+3.5]，6改善/6退化。两份恢复端点原始200题及compact/SHA独立核验。
新候选选择仍保持原门槛：总240最好新候选47/48、incumbent46/48，
未多两题，保留总192已选策略。不能称已稳定提升或最终收敛。

首次总240主评被本次pytest进程2379604创建的GPU上下文触发watchdog，
发生在任何episode前；是我们的CPU测试隔离失误，非模型collapse。
原协调器2059883退出，原失败及foreign_workload证据全部保留。
后续CPU命令/测试入口均屏蔽CUDA，恢复只重试原矩阵，不重训、不重选。
增强审计只接受相同配置、已绑定的特定故障及0旧rollout，相关23测试通过。
原生续训协调器另存并核验运行，等待WHALE完整资源释放后接block02；
其边界/恢复审计共35测试通过，保持原24h截止，不重置Adam。

WHALE v7曾完成224个Solver episode/7个RSFT批，随后将合法零奖励的
AGENT_INVALID_ACTION误判为基础设施错误。旧23727.45秒、3553逻辑成功调用、
112663个RSFT响应token、60个Adam小步保留；没有MH或最终200题，
不能当成完整基线。现在区分模型无效动作与真正基础设施错误，并在校验前保存报告。

v8启动在旧node4私有评测目录上遭遇node3读取权限错误，未进入smoke/研究训练。
v9将同字节输入复制给现有research group1008读取，没有改旧目录权限或证据。
它实际完成32K context/4096 response、warm Adam两步后恢复第三步、
完整FP32/Adam指纹、有限梯度及vLLM重载验证；1822.47秒，0研究episode。
FLA0.5.2/Triton3.7.1只注入learner；缺causal-conv1d的统一warning
不等于FLA chunk kernel未使用，实际模块逐层分别选择kernel。

为避免384候选重复启动环境且串行评分，v10只调整host评分路径：
同一迭代共享Solver、独立命名原生worker、最多8候选并发，所有slot/seed、
反馈边界和官方Pareto选择保持。56测试/48subtest通过；真实两个worker
同两题共4个显式无模型调用的CPU episode也通过，不能计作模型分数。
比较完整非评分AST、native配置、模型及依赖后显式复用v9原生组件证明，
不是伪造v10的新GPU smoke；组件门控收据与原GPU证据都绑定SHA。

node3正式v10服务已核验active，原W0+h0的初始开发10题结果6/10=60%。
该10题不是core200，也不是训练后效果；不得外推为WHALE最终性能。
计划两周期各320 RSFT+192 MH，合计640/384，固定本地H0 proposer、无外部API。
22:52:12UTC后不启动新批/新迭代；若不足预算，必须明确partial，
不能冒充“性能差提前停”。性能停止只在完整周期后检验预声明的20pp下降。
仍未取得WHALE最终200题结果。MA会在其原始200题核验及双节点释放后自动继续。

论文级报告同时补充真实本地SAO实现：native路径gamma=lambda=1；
不能把未调用的length-adaptive辅助函数写成已启用。名称不构成完整原论文复现。
全部当前结果仍属单训练分支、重复开发评测，不改写历史92%目标。

- [MA早期结果](../runs/alfworld_ma_simultaneous_from97_v1/EARLY_RESULT.json)
- [MA原生续训协调器](../runs/alfworld_ma_simultaneous_from97_v1/resume_after_whale.py)
- [WHALE v10冻结设计](../runs/alfworld_whale_sao_matched_v10/DESIGN.json)
- [原生GPU证明](../runs/alfworld_whale_sao_matched_v9/gpu-smoke/SMOKE_COMPLETE.json)
- [组件一致性与验证](../runs/alfworld_whale_sao_matched_v10/COMPONENT_VALIDATION.json)
- [论文级报告](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md)

### WHALE真实吞吐不足及原生64-episode边界

v10首批32episode为25成功、552个成功调用、9个Adam小步。
原生计时1250.31秒在最终串行8-rank文件SHA之前取值；
请求18:06:59到COMMITTED18:34:05约1626秒，不能把1250当完整墙钟。
第二批27/32成功、555个成功调用、9个Adam小步也已完整保存。
19:01左右在第二份native COMMITTED后暂停协调器342251，64条研究样本保留，
没有第三份报告；不把暂停记录成算法退化，也不抛弃这两批成本。

v11只做等价执行修正：将全局64个成功调用的小批分片至8个FSDP rank；
每rank以全局response-token数/8归一化，确保平均梯度仍是原全局token均值；
padding调用有真实forward但loss-mask为0，并独立计入物理计算量。
2-rank CPU FSDP在1/3/5/7条不等长调用、跨小批和尾部padding下，参数/Adam/
loss/token均通过数值对比。首次测试Gloo选择了不可连通的节点地址，改为
本机loopback后通过，未改变目标。46项测试与35subtest通过。

完整文件SHA改为并行，不跳过任何字节：真实8份旧rank-state在39.58秒复算，
与全部原摘要一致。另将重复的空载vLLM重载检查改为实际下一次使用后的验证；
使用前仍标pending，不制造成功式回退。计时移至最后哈希之后。
v11固定复用64条原始报告与最后FP32/Adam，剩576条和384个MH提案预算不变。
新source和config冻结，新的真实32768/4096 warm-Adam门控在node4运行，
还要求对v9合成loss与grad norm的relative1e-3/absolute1e-6检查；
在通过前不得把新的训练路径或完整预算标记成功。
原截止/性能停止规则不因吞吐不足而改写，存在无法在窗口内满额的风险，
必须据实报告实际样本量与主200，而不是把时间限制伪称性能早停。

### WHALE v11数值门控结果与v12批量修订

v11真实32K/4096、warm Adam、FP32恢复/vLLM重载通过，1210.03秒。
但预声明的跨多步relative1e-3/absolute1e-6门槛失败，原失败收据未删除或覆盖。
随后另行事前声明的固定权重全梯度probe不做optimizer step：原路径重复
gradient relative RMS0.0027486333，分片相对参考0.0032532655；
规则`min(.01,max(.001,2×repeat))`给出0.0054972666，三次loss逐值相同，
该78.19秒诊断通过。结论限定于同权重目标与梯度误差，不称多步轨迹等价。

v11恢复原64题和Adam18，连续保存三个32题批，最终160题/Adam43。
新批native elapsed686.26、474.26、447.63秒均含完整文件hash。
曾试图在128题边界暂停，但第5批报告已出现，守卫拒绝丢弃后来样本；
随后等待第5份native保存才暂停。该拒绝证据保留，没有回滚到128题。

因实际模型/状态I/O仍占主要时间，在未看到任何MH或最终200分数前冻结v12：
第一段32/32/32/32/32/160，第二段160/160，保留160条前缀和Adam43，
计划仍为640个Solver episode与384个MH提案、320/640题处交替。
段内online刷新频率改变，必须作为学习超参数修订披露，不能叫纯运行时等价。
环境并发32→64；H0和Solver各8个replica、每模型每GPU显存0.4，
分端口跨整个MH段常驻，迭代反馈/选择顺序和Pareto规则不变。

双模型真实服务门控通过：128并发请求、每个32767prompt+1生成token，
首token逐请求不同，耗时293.05秒。0研究环境episode、0optimizer update，
128次合成调用单列。六个native计算/恢复函数AST与分片helper SHA不变，
显式复用v11原生证明而不是伪造新的native smoke。48测试/38subtest通过。
v12续训/最终审计脚本还用真实160题/Adam43前缀验证，并确认错误batch布局被拒。
RSFT准入截止22:52:12不变；MH迭代准入改为23:12:12，主评保留25分钟。
用户23:37:12硬截止不延长，尚无WHALE最终200或完整交替结果。

- [v11门控修订](../runs/alfworld_whale_sao_matched_v11/VALIDATION_AMENDMENT.json)
- [160题真实原生边界](../runs/alfworld_whale_sao_matched_v11/THROUGHPUT_PAUSED.json)
- [v12冻结设计](../runs/alfworld_whale_sao_matched_v12/DESIGN.json)
- [128路双模型服务证明](../runs/alfworld_whale_sao_matched_v12/DUAL_SERVING_VALIDATION.json)

### WHALE首个完整320+192周期完成，第二周期继续

v12第一个160题后缀134成功，2495个成功调用、79092个响应token，
39次原生更新将Adam43连续推进到82，完整native751.58秒。
独立核验前后全部320题的ID/seed、真实成功调用与原生步数；
39个梯度norm均有限非零，范围[0.3391,1.8376]，没有优化器重置。

22:14左右首个MH段完成8次适应性迭代/192个提案，125合法并实际评分，
104组不同actions；55个输出截断、11个缺完整patch标记、1个语法错误。
全部slot的seed与上一迭代开发反馈逐一核验通过。开发从原6/10升至9/10，
但该9/10在本段搜索之前已有，8次选择均保留初始h0；不能把RSFT后的改善
归于MH。47个合法候选的逐题reward/turns与h0相同，仍保留实际评分成本。
未触发预声明20pp退化停止，第二个320题Solver阶段继续。
主200仍未产生，完整640/384预算尚未完成，硬截止和准入边界不变。

- [320题原生核验](../runs/alfworld_whale_sao_matched_v12/NATIVE320_AUDIT.json)
- [192提案独立核验](../runs/alfworld_whale_sao_matched_v12/MH0_AUDIT.json)

### WHALE实际640/216边界：同预算要求尚未完成

第二个Solver段两个160题批分别130和135成功，Adam82→121→159；
native完整耗时729.60和737.92秒。所有640条真实报告和159个连续原生步数核验，
v12新增116次更新的grad norm均有限非零，范围[0.3391,1.8544]。

第二MH段只在23:12:12准入截止前开始了一次迭代：24提案、6合法，
开发10/10，依旧保留h0。随后按冻结的报告预留规则停止，
23:14左右冻结最终640题Solver+h0并开始主200，没有看主评后再选模型。
实际640个Solver episode、216个提案、131个合法；一个完整交替周期，
加第二个完整Solver段和1/8个MH段。距384提案目标差168，
不能称完成用户要求的同样本复现，也不能说是性能差提前停止。
主200在进行中；该时间/预算失败必须和最终准确率一起交付。

- [640题与原生连续性核验](../runs/alfworld_whale_sao_matched_v12/NATIVE640_AUDIT.json)
- [实际搜索预算核验](../runs/alfworld_whale_sao_matched_v12/SEARCH_BUDGET_AUDIT.json)
- [主评前冻结的模型/策略](../runs/alfworld_whale_sao_matched_v12/execution/FINAL_SELECTION.json)

### WHALE主200完成：81.5%，但未完成要求的完整同预算复现

主200原始结果在23:31UTC前完成并独立核验：163/200=81.5%，
strict197为160/197=81.2183%。相对历史SAO W1+h2的92%，
差-10.5pp，配对95%[-16.5,-4.5]，9改善/30退化；
对W1+h1的90.5%差-9pp，[-15.5,-2.5]，14改善/32退化；
对原W0+h0的85%差-3.5pp，[-8.5,+1]，8改善/15退化。
开发10/10不能替代主200，未得到超越原系统或SAO的证据。

最终实际640个Solver episode、216个提案（131合法、85无效），
529个成功episode提供9972个SFT调用、316420个响应token、159个Adam小步。
实际物理forward17744、响应token位置562943，含23个masked padding调用。
环境成本640训练+1330MH评分+10初始开发+200主评=2180episode；
Solver生成1411074token，proposer生成693416token、prompt2681646token。
v12执行7989.83秒但继承160题旧前缀，不是完整训练从零的墙钟。

原始任务/seed/token、最终选择先于评测、全部640条原生连续性、
216个适应性提案及成本、原始200与配对统计均已核验。
最终审计明确sample_budget_fully_consumed=false、主评按时完成=true。
**未完成640/384同预算复现**，少168个提案；时间截断不是性能停止。
双节点资源真实释放并写入MA的SHA绑定边界，原生续训协调器接续，
不重训首块、不重置FP32/Adam/ValueNorm、不延长原24h截止。

- [WHALE最终核验结果](../runs/alfworld_whale_sao_matched_v12/RESULT.json)
- [原始200题](../runs/alfworld_whale_sao_matched_v12/execution/final200/report.json)
- [最终审计](../runs/alfworld_whale_sao_matched_v12/TERMINAL_AUDIT.json)
- [MA资源交接证据](../runs/alfworld_ma_simultaneous_from97_v1/WHALE_RESOURCE_BOUNDARY.json)

23:31 UTC，MA恢复边界通过，block02两个角色的续训已启动；
原首块不重训，原24小时截止保持。当前公开MA主评仍为97%→93%→92%，
不能在新端点评测前宣称下一块改善或平台更高。

## 2026-10-08：四项机制消融修订与冻结代码验证

用户要求不再追加分阶段SAO，而将第四项改为“每轮均同时更新Engineer和Solver”。
新研究为完整MA、去队友权重、去协作上下文、去ValueNorm、每轮共同更新，共五配置。
上下文和ValueNorm不合并；训练只用种子1000043，仍以原始200题和固定144轮为主终点。

新的调度消融只改变Engineer更新开关，不改采样顺序、第三轮unit队友矫正和Critic输入。
每臂均1152个共享episode、2304条生成轨迹；完整臂和三项删除消费H/W768/1152条，
Actor93/141次、Critic192/288次；每轮共同更新消费1152/1152条，
Actor141/141次、Critic288/288次。因此总消费增加20%，不冒充同更新量或同算力。
过去从97%出发的每轮更新还改了第三轮队友权重，不能代替本次单一机制对照。

新源快照从原v1的319个已核验运行文件复制，仅覆盖指定运行文件和测试，新增控制程序。
冻结源上115项CPU测试和3个子测试通过；尚不代表新的原生门控或正式训练已完成。
历史路线实际schema核查通过：H0/W1 Actor、1152条有序输入、48+96轮更新预算、
原始194/200报告均吻合。历史MA64 Critic/ValueNorm、round48的Adam重置、
24对16个候选槽和历史峰值选择仍是差异，不把旧97%当作新消融的因果对照。

完整MA必须先完成、汇报和审查，然后才放行四组消融；低于或等于94%时标记较大历史差距，
核查实际路径而非按分数择优重跑。新程序包含资源稳定性门控、错误停止、信号记录及共享40h上限。
旧attempt02已经在零开训状态取消；原22轮中断及全部成本仍保留。
最近的17:43 UTC只读资源检查仍显示双节点被占用，未停止其他人的任务。

继续使用已实测约1.955倍的训练加速，不声称新测得额外加速，不采用改变输出的更快生成布局。
资源可用后预计先3--4小时获得原生检查和完整MA结果，再11--16小时做完四组消融与评测；
总计约14--20小时，不包含排队和基础设施恢复时间。

- [修订实验协议与预算](../runs/alfworld_ma_mechanisms_single_seed_v2/PROTOCOL.md)
- [冻结代码测试记录](../runs/alfworld_ma_mechanisms_single_seed_v2/CODE_TESTS.log)
- [旧准备取消回执](../runs/alfworld_ma_mechanisms_single_seed_v1_attempt02/CANCELLED_BEFORE_TRAINING.json)

### 18:11 UTC：附着式控制程序与资源等待已实际运行

冻结输入、模型权重和每臂单一配置差异的完整复核通过，代码核验已落盘。
node3实际UID1009/GID1008的CUDA隐藏CPU服务核验通过，共享目录可写；
这项检查没有GPU计算、训练或环境episode。

18:10:44 UTC启动控制程序，PID3224405；18:11验证进程身份及有效心跳。
当前`waiting_for_resources`，两节点未同时释放，新的原生和正式训练开始回执均为0。
程序等待稳定窗口后先做两次原生检查，再只运行完整MA；完成后停在结果/路线复核门，
由助手汇报和核查，再放行四组消融。资源等待不等于已经开始训练。

- [代码及输入核验](../runs/alfworld_ma_mechanisms_single_seed_v2/CODE_VALIDATION.json)
- [远端执行身份核验](../runs/alfworld_ma_mechanisms_single_seed_v2/REMOTE_EXECUTION_PREFLIGHT.json)
- [监控启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2/MONITOR_START_AUDIT.json)
- [实时状态](../runs/alfworld_ma_mechanisms_single_seed_v2/PROGRAM_MONITOR.json)

### 2026-10-09 02:37 UTC：原生调度验证通过，正式MA遭外部SIGTERM

稳定空闲门控通过后，完整MA与每轮共同更新分别完成六轮原生检查。
完整臂H/W Actor1/3、Critic8/12、消费32/48；每轮共同更新3/3、12/12、48/48。
两项审计均通过，合计96个共享episode、192条生成/176条消费轨迹；
这不是组件性能比较，也没有用校准检查点初始化正式臂。

正式完整MA在02:20:30启动，02:22在模型加载阶段中断。
无已提交正式指标、轨迹/环境回执或检查点，不能报告训练分数或伪称Adam续训。
node3在02:22:19.718捕获UID0/PID1148490的SIGTERM；同一时刻sudo日志记录
`hongyuanzhu`账号以root执行指向MA训练模块的Python命令，早于我们自己的清理操作。
该账号随后在02:26启动12K上下文压力测试，02:31确认占用两节点全部16张GPU。
不能判断是人工、自动脚本还是另一会话发起；node4发送者及以往中断来源仍不确定。

原控制程序已退出，故障、成本、信号链和通过的原生结果全部保留，未自动重启。
02:37启动独立只读资源观察，不改动其他任务，也不在检测到空闲时直接绕过故障门。
正式恢复仍需稳定、无清理冲突的资源窗口和显式恢复记录。

- [中断结果及原生成本](../runs/alfworld_ma_mechanisms_single_seed_v2/full/INTERRUPTED.md)
- [来源证据与归因限制](../runs/alfworld_ma_mechanisms_single_seed_v2/full/SIGNAL_PROVENANCE.json)
- [中断报告核验](../runs/alfworld_ma_mechanisms_single_seed_v2/full/INTERRUPTION_REPORT_AUDIT.json)
- [只读资源观察](../runs/alfworld_ma_mechanisms_single_seed_v2/post-interruption-watch/PROGRAM_MONITOR.json)

### 2026-10-09 02:55 UTC：新窗口触发显式恢复准备，未重复原生校准

只读观察在02:45:42取得连续三次双节点空闲后退出，02:47和02:54复查仍无GPU计算进程。
既有用户授权是释放后继续，因此准备一次独立v2 attempt02，不清除原故障或启动回执。
并未取得排他预约，亦未声称此前外部终止原因已经修复；启动前仍需稳定资源门控。

恢复驱动23项针对性测试通过，覆盖14个输入的位置等价、原生证据继承、
原始绝对截止时间、已开始尝试拒绝重放、证据篡改拒绝和失败后不再重试。
只改变运行目录及服务唯一后缀，冻结训练源码不动，不改变种子、优化器初始化、
数据、角色机制、候选数量或测评协议。复用两项原生检查的96个episode证据，
但不使用其检查点；旧v1与v2失败尝试仍是额外成本，不能当作Adam续训。

本条记录的是准备阶段，尚无正式新训练成绩。完整MA完成后仍停在结果审阅门，
不能为了追求97%而按成绩换初始化或重跑。截止时间仍为2026-10-10 10:10:44 UTC。

- [恢复验证](../runs/alfworld_ma_mechanisms_single_seed_v2/RECOVERY_CODE_VALIDATION.json)
- [新空闲窗口证据](../runs/alfworld_ma_mechanisms_single_seed_v2/post-interruption-watch/RESOURCE_WINDOW.json)

### 2026-10-09 03:00 UTC：独立v2 attempt02启动输入复核和资源门控

实际准备通过：14个JSON输入、10份角色配置保持科学等价，继承原生检查和冻结源摘要，
历史1152条有序训练输入复核通过，原生重执行episode为0，原绝对截止时间未改变。
node3既有训练服务接口以UID1009/GID1008完成新目录读写检查；没有改动sudo授权或其他用户任务。

02:59:17启动附着式恢复程序，控制PID3401882，有界监督PID3401880。
03:00:23核验时仍在输入摘要复查，尚无正式full/STARTED或首轮提交证据；
首轮通知观察PID3402105已启动并有心跳。后续自动等待三次双节点稳定空闲再运行完整MA。
没有重新执行原生校准，没有把启动中的模型误报为训练结果，也没有放行四个消融。

- [恢复协议](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/PROTOCOL.md)
- [实际准备回执](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/PREPARATION_AUDIT.json)
- [启动存活核验](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/MONITOR_START_AUDIT.json)

### 2026-10-09 03:08 UTC：用户取消再次被kill后的恢复授权

用户在03:07:51 UTC明确表示，若对方再次kill我们的任务，可以不用重启。
当前正常运行的attempt02不停止；如果再次终止，则结束剩余队列，保留日志、
已有检查点和实际成本并汇报，不自动重启，也不另建显式恢复尝试。
正常完成完整MA并通过结果审阅后，仍可按既有方案运行四项消融。
现有零自动重试设置已核实，本次没有修改冻结训练代码或科学配置。

- [持久用户指令](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/NO_RESTART_AFTER_TERMINATION.json)

### 2026-10-09 03:18 UTC：attempt02在两轮Critic预热后被SIGTERM终止，不再恢复

正式开始回执为03:05:07 UTC。双方在03:12提交首轮，03:13:01与03:13:08分别提交第二轮；
本地协调器03:13:12.386记录signal15，03:13:23持久化失败并退出。
首轮通知虽然成功，其含义仅为中间里程碑，不能覆盖后来发生的中断。

最终核验H/W均为2个已提交轮次、0次Actor更新和4次Critic更新，仍处于Critic预热。
保留22个共享环境结果，按轮为8、8、6；44条生成角色轨迹中32条进入已提交更新。
已记录生成action tokens为Engineer57564、Solver17102；
已消费轨迹中的action tokens为42300、14313。未落盘的在途成本未知，不能记为0。
没有新HF检查点、没有原200题成绩、没有正式消融结果，不能作算法优劣结论。

03:17自有监督/控制/角色进程和观察进程均确认退出，远端自有服务关闭，两节点资源已释放。
本次信号发送者未识别，不复用前次账号归因；远端终止记录可能属于本地退出后的自有清理。
依照用户03:07:51指令，结束剩余队列，不再重启、另建恢复尝试或启动等待空闲后续训的观察。
所有日志、两项已通过原生校准及此前失败成本保留，没有修改其他用户任务。

- [中断结果报告](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/full/INTERRUPTED.md)
- [成本及信号核验](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/full/INTERRUPTED.json)
- [终止处置回执](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/STOPPED_BY_USER_POLICY.json)

### 2026-10-09 08:08 UTC：按新授权实现12轮配对断点和空闲后恢复，尚无正式新结果

用户07:30重新授权每12轮完整保存，成功提交下一份后删除上一份，保留评测节点；
训练被kill后等待两节点连续空闲30分钟再提交。07:40进一步明确首次空闲即可启动，
不必先等30分钟。两条新指令覆盖旧暂停/不再恢复限制，旧指令及所有中断成本保留。

周期保存包括双方同一已提交轮次的Actor/Critic、原生Adam、ValueNorm、
worker RNG/LR和已提交输入/指标/调度游标。新配对先完成文件大小、元数据摘要、
优化器采样指纹/步数及ValueNorm/调度检查，再原子提交LATEST，最后删除旧恢复代及废弃半成品。
这些检查不是所有大型tensor逐字节重读；载入时另执行原生优化器指纹比对。
保存时不重启健康的推理服务；真实中断后重新启动推理服务，废弃未提交异步工作并计成本，
不宣称逐位相同的采样续跑。96/144轮HF Actor和Critic都保留以兼容原审计。

第一版113项测试通过，但原生等待器仍未获得可用窗口。我们在零原生attempt状态下
主动结束等待，补齐最终断点已完成但完成回执缺失、半成品回收和分布式任务启动期门控。
第二版[120项针对性测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/CPU_VALIDATION.log)
通过，含真实CPU原生DCP/Adam/RNG恢复后下一步更新一致性，以及四种角色/调度的计数恢复。

新的[附着式控制程序](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/BOOTSTRAP_STARTED.json)
正在输入绑定，后续为实际12轮保存、主动停止自有校验进程、恢复至15轮的原生验收。
验收权重不得作为正式初始化。此时仍没有新144轮、200题或四项消融结果，
不能把系统测试写成算法性能证据。没有暂停、终止或修改其他用户进程。

### 2026-10-09 08:37 UTC：用户主动暂停原生恢复验收，后续改为至少两小时冷却

08:34:57.344 UTC用户要求暂停并改成两小时间隔。向身份核实后的自有主控制PID3514619
发出SIGINT，使其退出并清理自有角色，不走SIGTERM后的30分钟重试路径；
控制器以130退出。只读观察PID3518079亦停止。node3对应自有服务为inactive/not-found，
MainPID=0；本地与远端之后的GPU检查均无计算进程，没有修改其他用户工作。
这是主动暂停，不应统计成一次未知外部kill。

原生校验attempt-0001最终提交H/W各12轮，Actor更新5/9、Critic更新16/24，
96个共享环境结果、192条生成角色轨迹；未落盘在途工作仍未知。
当时正在完整原生保存，但LATEST配对提交不存在，因此**这些12轮不能作为已可恢复的进度**。
未删除部分保存、旧attempt或成本；下一次必须重新完成12轮保存/中断/恢复至15轮的原生验收，
不能把这次校验权重或其未完成保存用于正式初始化。

新时序为首次不早于10:34:57.344 UTC，且两节点新鲜资源检查通过；
以后非计划中断至少冷却7200秒再检查资源，而非旧31次检查/30分钟连续空闲规则。
用户指令按固定最低间隔解释，不要求两小时持续空闲，校验计划内重启也不额外等待两小时。
每12轮保存、同代双角色验证、滚动删除及96/144保留不变，原绝对截止时间不延长。
新控制器与最新指令独立持久化，原科学/执行快照、既有授权及输入摘要不覆盖。

- [用户最新指令](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/USER_PAUSE_AND_DELAY.json)
- [暂停实际状态及成本](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PAUSED_STATE.json)

08:50更新：两小时执行层控制器在冻结rolling12-v2运行时上通过30项针对性测试，
包括时间边界和异常恢复门控；不是新的大模型原生恢复成功证据。
已启动并核实附着式等待PID3543367，心跳正常、无原生子进程，
原attempt仍为0001，正式full/STARTED仍不存在。等待到10:34:57.344 UTC后才进行资源准入，
继而重新执行原生验收；校验通过后仍须报告、再提交正式MA。
该等待器不会每两小时打断健康训练，也不替代每12轮完整保存。

- [30项测试及摘要绑定](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_CODE_VALIDATION.json)
- [实际等待启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_STARTUP_VERIFIED.json)

### 2026-10-09 11:22 UTC：两小时等待后完成真实原生恢复验证，准备提交正式full

延迟窗口10:34:57.344 UTC到期后，两节点均通过资源/启动器准入。
attempt-0002从0训练至12，完成双角色完整配对保存后主动停止；
attempt-0003实际载入12轮Adam/ValueNorm/RNG及已提交输入游标，再训练至15。
原生门控11:16:20 UTC通过，控制进程正常退出，远端两次自有服务已为inactive/not-found。

独立检查加载/完成回执及指标摘要匹配，双方恢复next_sample_index=96，
未重置Adam、未重复预热，最终global_step精确为0至14。
H/W Actor步数7/12、Critic步数20/30、消费轨迹80/120，与预声明预算一致。
现仅有15轮最新大型原生状态；12轮旧代及首次暂停半成品已不在恢复目录，
但轻量历史、所有原始attempt、三份COST记录均保留。

三次attempt分别记录96、96、24个新共享结果，合计216；生成角色轨迹合计432。
120是后两次构成的成功逻辑路径，不是包括暂停在内的全部物理成本。
未落盘的在途工作仍未知。校验权重不进入正式MA，不能由此作算法优劣判断。

11:22新鲜检查双节点没有GPU计算进程或识别到的分布式启动器。
按既有授权进入完整MA控制实验，仍从共同H0/W1及普通SAO raw Critics出发，
fresh Adam/ValueNorm、seed1000043、固定144轮主终点与96轮诊断、12轮完整保存；
完整结果审阅后再启动四项消融。未改变绝对截止时间、两小时中断冷却或科学配置。

- [原生门控](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/ROLLING_NATIVE_GATE.json)
- [校验结果说明](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/rolling-canary/RESULT.md)
- [报告和新鲜资源检查](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/NATIVE_GATE_REPORT_AND_ADMISSION.json)

### 2026-10-09 11:28 UTC：正式full144原生训练启动，未产生最终评测

full/attempts/attempt-0001在11:26:14.504 UTC启动，控制PID3626493，
恢复位置0，使用共同H0/W1与普通SAO raw Critics，不载入原生校验或历史97%的权重状态。
11:28核实本地控制存活、远端自有服务active/running（MainPID1302169），
双方指标文件尚无已提交轮次，当前为模型加载阶段。

预算和机制开关不变：144轮主终点、96轮诊断、batch8、seed1000043、
逻辑生成1152个共享episode/2304条角色轨迹，H/W Actor步数93/141、Critic步数192/288。
这些是预声明最终预算，不是当前完成量。每12轮完整保存、两小时中断冷却，
完整控制先报告再启动四项消融；所有中断额外成本均保留。

此前实际原生保存测得12轮H/W约258.8/259.1秒，15轮约289.9/269.5秒，
只包含各角色保存阶段，不代表启动、HF导出、完整训练或评测总耗时。
独立检查滚动删除日志显示，首次暂停半成品和旧12轮代均在新完整配对提交后被删除，
最终15轮状态及轻量历史、全部原始attempt成本仍在。

- [正式原生启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/FULL_NATIVE_START_VERIFIED.json)
- [正式进度心跳](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PROGRAM_MONITOR.json)

### 2026-10-09 15:39 UTC：训练完成后144轮selection中断，按两小时策略恢复评测而非重训

144轮共同训练完整结束，14:22:34训练审计通过1152个共享episode；
最终原生配对断点和双方COMPLETE回执均已验证。96轮评测已有完整结果。
15:01用户询问时，控制PID3626493正在运行144轮的48题Harness筛选。
随后控制和评测PID3792500均退出，附着式shell返回1；推理日志约15:03:12同时结束，
原控制与评测均没有写出FAILED、RELEASED或Python traceback。
15:04:50首次核验两进程不存在、两节点释放。信号/发送者仍未知，不以时间相关性作外部归因。

144轮原8个提案和原incumbent/no_patch均保留，重复patch去重后9个独立Harness，
应完成432个selection episode。原始日志包含144次开始、82个完整结果，
62个在途episode没有完成结果；未生成任何完整Harness report或144轮SELECTED。
恢复绑定344个既有证据文件，复用全部82条完整结果、原候选和原96轮结果，
不重训、不增抽候选、不重跑完成题、不按成绩决定重试。

新的评测层仅扩展“控制进程已退出但未写终止回执”的可审计恢复入口，保留原ReadError入口，
并复用既有逐episode缓存和严格任务/seed/模型/Harness一致性检查。
只在新的空闲检查通过后写明确标记postmortem的资源释放观察，不将其冒充原进程finally回执。
同时解决了rolling目录转换与旧全路径身份比较不兼容的审计问题，
只允许由原spec及已记录PREPARED确定的路径/恢复元数据变化，学习率/seed等变化仍拒绝。
从原frozen源码复制的独立评测revision通过67项测试及13项子测试，原训练快照不改。

新等待器遵循保守的15:04:50+7200秒门槛，即最早17:04:50 UTC，
且两节点必须重新通过资源/启动器准入。15:36时两个节点已有其他分布式任务，
没有暂停或修改它们；到期仍占用会继续等待。
15:39已核验自有等待PID3819570及信号跟踪PID3819566存活，无GPU子进程，
尚未创建恢复评测attempt-02。此控制只恢复剩余selection并进行原main200，然后汇总/审计完整MA；
四项消融仍需实际完整结果报告后才能放行。原绝对截止时间不延长。

- [固定中断证据及恢复时序](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/SCHEDULE.json)
- [实际等待启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/STARTUP_VERIFIED.json)
- [67项冻结运行时测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/CPU_TESTS.log)

### 2026-10-10：完整MA144固定终点93.0%，历史路径与原始失败轨迹完成核验

05:15 UTC资源准入后，仅恢复原144轮selection的未完成部分。
82条已完成结果逐字节复用，原8个提案不变，7个有效；
05:40 UTC选定proposal01并冻结，随后main200于05:55 UTC完成。
96轮诊断为180/200（90.0%），同一Solver空Harness为178/200（89.0%）；
144轮预声明主终点为186/200（93.0%），空Harness为179/200（89.5%）。
不在96和144之间按main200成绩择优。

selection最终432个逻辑episode、494次物理开始，差额为首次中断的62个未完成尝试；
复用82条完整结果、重复完成题0、新增候选0。保留原不完整尝试及未知在途token成本。
正式训练仍是唯一一次144轮、1152个共享episode；没有因评测/报告故障重训。
原生恢复校验的216个物理共享episode及更早失败成本是额外开销，不计入正式逻辑预算。

原评测恢复控制返回1的准确原因是CPU失败样例报告读取了历史紧凑摘要；
摘要每题只含身份、reward、tokens，没有messages。历史原始matrix和200条完整轨迹尚在，
已验证原matrix、固定Solver/Harness，以及原始报告的紧凑投影与97%摘要完全相同。
新的报告专用快照仅修改历史轨迹解析与审计复用，不改既有训练/评测快照。
67项针对性测试在工作树和精确报告快照均通过，旧FAILED及所有已评分文件摘要保持不变；
旧历史路径审计保留，新的复核独立落盘。没有伪造轨迹、忽略失败题或放宽原始消息要求。

新93%对历史97%：新系统改善1题、退化9题，差−4个百分点，
配对问题bootstrap95%区间[−7,−1]个百分点。历史差异包括已训练MA64 Critics/ValueNorm、
48轮处Adam重启、24/16提案机会和开发集峰值/固定终点之别；
Actor起点、1152条有序训练数据及声明的MA段更新预算已核对。
这些差异不是已识别的单一原因，不能据此归因某一组件或声称跨种子显著性。
新系统14个失败中12个重复相同行动至少4次，13个达到50模型调用；
历史6个失败中对应为6和5。9个正向与1个反向差异题均披露，
示例270、329、342、356按固定规则选择；反向题为305。

四项消融仍未完成，不能写成“所有组件均必要”。原10:10:44 UTC截止时间不延长；
06:12时两节点空闲、剩约4小时，后续只能在原窗口内继续推进并保留断点。

- [完整MA结果](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/RESULT.md)
- [历史路径复核](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/BASELINE_PATH_AUDIT.json)
- [原始配对失败样例](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/failure-cases-vs97/CASES.md)
- [独立CPU报告恢复证据](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/postprocess_resume01/COMPLETE.json)

### 2026-10-10 06:18 UTC：报告完整基线后启动without_peer，仍服从原截止时间

06:15已完成并记录93%完整基线的结果/路径审查和报告放行；
原评测的1082个固定文件重新验hash、四份端点matrix及82条缓存复用均通过独立核验。
基线发布版本的状态/日志/算法文档另存摘要相同的只读证据副本，不因后续状态更新丢失原版本。

首项without_peer由原冻结rolling12-v2代码及已验证的两小时驱动启动，
06:18检查控制PID4062818、本地Solver4063504和node3自有服务MainPID1731201均存活。
两角色已提交轮次仍0；该记录仅证明原生启动/加载，不是性能结果。
共同初始化、seed、144轮预算、12轮保存及原10:10:44 UTC截止时间不变。
其余without_context、without_value_norm、always_joint均未启动；
没有把原先CPU报告错误作为理由重训full或改变初始化，也不声称余下四组能在当前窗口内完成。

- [真实原生启动回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/WITHOUT_PEER_NATIVE_START_VERIFIED.json)
- [基线放行记录](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/BASELINE_RELEASE.json)

### 2026-10-10 07:15 UTC：without_peer24轮保存并发故障、实际双节点复现及单发布者修复

06:52:12 Solver原生日志首先报COMMITTED.json.tmp→COMMITTED.json重命名FileNotFoundError；
随后原生退出清理产生CancelledError，最终外层控制于06:52:33.772817退出1。
控制信号追踪未记录外部kill，原FAILED正确保留为blocked_error、retry_after_idle_allowed=false。
两节点及自有远端服务已释放，未修改其他用户进程。

24轮配对断点已成功发布且通过原验证器：192个已提交共享episode，
H/W Actor13/21、Critic32/48；两角色指标覆盖0–23，原科学摘要不变。
实际记录199个共享结果和398条角色回执，多出的7个预取结果未提交；
未落盘在途工作仍未知。没有96/144评测，没有把24轮当成完成的消融分数。

首次直接SSH的锁探针因登录GID1009无写权限而失败，明确保留原因，没有放宽文件权限。
改用既有原生UID1009/GID1008接口，在node4持有排他flock期间，
node3确实同时取得同一共享文件的排他锁，证明当前fuse.hf3fs上的锁不能承担跨节点互斥。
修复不是仅随机化临时文件名：新快照只让Solver发布共享所有权/提交/历史/最新指针及执行保留策略，
Engineer只确认完整且摘要匹配的发布结果；各角色自己的完整原生payload仍分别保存。

工作树及精确rolling12-v3快照的66项断点/原生/时序测试通过；
校正驱动和两小时时序的28项测试通过，两套存在重叠，不按94项独立测试计数。
真实两节点CPU回调完成合成12、24轮配对、相同摘要确认和旧12轮删除。
合成payload不构成新GPU训练验收；Actor/Critic/目标/数据/seed/预算/原生恢复路径不变，
原科学pins和真实GPU恢复门控已在原冻结源码中重新通过。
执行修订单独记录，原协议、快照、失败回执与24轮证据均不覆盖。

新CPU等待控制4102113（追踪4102109）和只读观察器4102927在07:12–07:15验证存活，
尚未创建attempt-0002。最早恢复为08:52:33.772817 UTC，再检查资源和原门控，
从24轮Adam/ValueNorm/RNG/数据游标状态继续；只读观察器会核验首次新36轮配对。
原10:10:44.379806 UTC截止时间不变，余下执行窗口最多约78分钟。
完整MA186/200=93.0%继续有效，其余三项尚未启动；未完成四项组件必要性证据。

- [跨节点锁探针](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/CROSS_NODE_FLOCK_PROBE.json)
- [实际24轮状态审计](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/SAVED24_AUDIT.json)
- [实际双节点CPU发布验收](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/two-node-cpu/RESULT.json)
- [校正恢复绑定](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/PREPARED.json)

### 2026-10-10 08:17 UTC：用户取消自动重启和H200训练，保存交接材料

08:16 UTC用户要求暂停自动重启、不要使用H200，后续明确希望GitHub保存完整代码、
实验结果和数据，并考虑HF私有存储保存中间恢复状态和checkpoint。
原控制/追踪/只读观察进程初检时已不存在，attempt-0002未创建。
持久取消门控和禁止H200指令已记录；两个自动恢复入口均实际验证为抛出取消异常，
两节点GPU/分布式启动器为空、对应原生服务inactive/not-found。
未修改无关CPU服务或其他用户任务。

当前最终科学结果仍只有共同初始化full144的186/200（93.0%），
without_peer保留24轮、192个已提交共享episode和199个实际落盘结果，没有消融主成绩。
所有原始失败、科学配置及源码快照保留，原研究截止时间不延长。
归档工具只进行本地CPU/文件操作，排除凭据、验证压缩包成员、保存19组模型/恢复状态清单；
允许的硬链接用于防止原目录项被删除后丢失文件，不修改原文件内容/权限；
不同所有者的只读HF文件不能硬链接时明确记录并复制，而不是放宽源权限。
本地包、Git提交、同盘副本均不是远端备份，本次没有执行GitHub push或HF上传。

- [交接与恢复注意事项](./MA_SAO_HANDOFF_2026-10-10.md)
- [本地归档目录](../runs/ma_sao_handoff_20261010T081628Z/)

### 2026-10-12 02:56 UTC：新授权的10分钟采样、两小时条件恢复

用户重新授权：每10分钟读取node3/node4状态，两小时内均为空闲或已核验的利用率维持程序时，
自动继续原有后续训练。不是立即开训，不把更早的手工检查或历史状态当作空闲计时。
两节点各8张H200须完整可观测；其他GPU进程、分布式启动器、无法归属的GPU活动、
权限/SSH故障均重置窗口。至少13次有效采样、实际跨越7200秒，漏检或时钟跳变重新计时。
第一轮探测尚未完成的耗时不算空闲信用；达标后再次核验当前资源。

新授权单独保存，绑定原三个暂停文件的摘要；旧文件不删除、不改写，旧恢复入口仍拒绝执行。
原10月10日截止时间已过，因此只对这个新入口增加明确的运行期授权：
原spec内的截止时间、全部科学字段和原checkpoint的science_digest保持逐项一致，
新attempt仅在已有排除科学摘要的rolling_recovery字段内携带授权与准入回执。
仍保留每阶段8小时和无进展停止保护；范围仅为剩余四项固定144轮及原96/144评测，
不是新种子/新实验或任意延长已完成的full训练。

新的idle2h-v1源码快照基于已验证的rolling12-v3，变更限于监控/授权模块、
三个调用点文件和既有结果发布函数提取；Actor/Critic损失、优化器、数据、
队友校正、Critic输入、ValueNorm、schedule、原生payload/恢复计算均不变。
精确快照的179项CPU回归测试通过，覆盖7200秒/13样本边界、每600秒检查、
最后一刻变忙、取消、检测错误、未知进程、重启不继承信用和科学摘要不变。
11个关键恢复/提交/身份函数的AST与v3一致。
原v2运行时代码重新完整校验395个科学pins（约83.99GiB），24轮原生状态和full144结果不变。
本次没有新GPU canary，也没有把部署监控写成已完成24→36续训。

执行顺序：without_peer从24轮恢复，再运行without_context、without_value_norm、always_joint；
full144的186/200结果不重复。健康训练仍每12轮滚动保存、不主动重启。
有明确终止/争用证据时，清理本次自有进程后重新观察两小时；
未知代码错误或训练完成后的评测/报告中断保留证据并停止，不自动重训或重抽候选。
控制器为会话托管CPU进程，不配置开机自启或脱离会话服务；新控制目录中的CANCELLED.json可以暂停。
离散采样无法排除采样间短时作业，首次开训前后仍使用原资源安全检查。

- [新授权](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/AUTHORIZATION.json)
- [执行和验证绑定](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/PREPARED.json)
- [精确源码测试及恢复不变性](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/CODE_VALIDATION.json)
- [原始输入和24轮状态复核](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/ORIGINAL_PREREQUISITES_VERIFIED.json)
- [实时监控状态](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/STATUS.json)

03:22:54 UTC首次实时样本为两节点空闲，CPU控制PID1137442与排他协调锁均已验证，
计数1/13，尚无准入回执、worker或attempt-0002。下一次检查约03:32:54 UTC，
满足所有后续样本与最终核验时，不早于05:22:54 UTC准入。
这些是监控启动证据，不是已经完成两小时观察或已经恢复GPU训练。
详见[启动验证](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/STARTUP_VERIFIED.json)。

### 2026-10-12 11:32 北京时间：改为独立只读GPU历史记录，不启动新训练

用户先要求立即启动剩余消融，但实时检查发现node4有UID1002的真实GPU作业，
而node3空闲；没有抢占其他人的资源，也没有创建新的训练attempt。
用户随即改为要求不依赖对话的程序，每5分钟记录GPU利用率和占用程序、使用北京时间，
以便后续请求启动时掌握历史。旧CPU自动等待器已退出，另保存CANCELLED记录阻止其重启开训。

新增gpu_history模块复用原filler身份核验和pmon解析，记录双节点每卡利用率/显存、
PID/UID/程序名及各卡进程分配。日志不包含完整参数、环境变量或令牌。
每300秒采样，按Asia/Shanghai日期追加JSONL，最新状态原子写入，节点探测失败显式落盘。
查询最近两小时要求至少25个有效样本、实际7200秒跨度、最新样本未过期；
真实工作、错误、漏检、时钟跳变或进程重启都会中断连续空闲证据。
程序只有观测功能，历史合格不自动触发训练，不结束或修改任何GPU进程。

43项CPU测试通过；两台实际节点探测成功，node4的PID1138383被识别为其他真实工作，
非已核验维持程序。后台进程按用户明确要求独立于本次会话运行，
不安装开机自启，也不承诺主机重启或外部kill后自行恢复。
所有原实验结果、失败记录、native24及full144状态保持不变。

- [程序](../whale_local/sao_bridge/gpu_history.py)
- [日志目录](../runs/gpu_history_node3_node4/)
- [查询及停止方法](./PROJECT_STATE.md)

### 2026-10-12 12:45 北京时间：新授权串行15卡，外部kill后不恢复

用户授权停止node3/node4已核验维持程序，并采用单实验多卡、实验之间串行，
优先without_peer24→144，然后without_context、without_value_norm、always_joint。
后续明确要求外部kill后不重启，自身训练问题可以诊断修复后从原生断点继续。
完整计划和14–20小时的保守排期见[项目状态](./PROJECT_STATE.md)。

只对精确匹配UID、supervisor、systemd cgroup的filler执行操作。
初次TERM后Restart=always仍自动拉起，未把超时算作成功；
补充显式停止已核验service后，确认两节点MainPID=0、filler GPU PID清空。
node3初次共享回执写入因登录UID权限失败，节点动作已完成；
通过后续只读审计在协调节点持久化双节点证据。PID1138383始终保留。

运行配置为Engineer4训练+4rollout，Solver4训练+3rollout，只使用node4 GPU1–7。
四rank原生优化器、ValueNorm、RNG/已提交游标及原科学spec保持不变；
实际服务副本变化单独记录，不能承诺生成轨迹逐token等价。
为避免再次改变固定终点评测的服务副本数，训练释放资源后在node3用原8副本评测。
不重跑Full144，不改候选数、选择集、主评测集或种子，也不以分数决定重试。

工作树针对性248项测试通过，随后新增精确4+3配置、串行评测分派和不重启用例，
新的serial专项36项通过。首份冻结快照在CPU测试收集阶段缺少一个测试helper和
独立GPU记录模块，保留失败快照/日志，构建新快照而不原地改写；当时尚未启动GPU训练。
正式启动须以精确冻结快照验证、原输入/检查点核验和实时15卡资源复查为前提。

13:20北京时间补充：serial15-v2的attempt-0002启动失败，两个角色原生进程
均code1退出，native verify_upstreams找不到打包遗漏的areal-local-network.patch；
4份patch均未随只复制Python文件的脚本打包。COST确认无新增outcome、
无新生成receipt、无新提交轮次，不能把这次计作已恢复训练。
停止并持久取消原serial控制入口，保留所有失败证据；按用户允许自身故障修复续训，
新建self_fix01控制和完整serial15-v3运行目录，所有非缓存运行文件均复制并哈希，
在准入前实际执行原生upstream patch核验，添加回归测试。

serial15-v3精确快照348项测试和13项subtests通过；395份原科学pin约83.99GiB
再次完整哈希，24轮和Full144 native配对摘要保持不变。
node3真实原生UID经既有systemd接口验证控制目录写入及全部patch核验通过。
attempt-0003约13:14启动模型初始化，GPU0仍只有PID1138383，
node4其余7卡和node3八卡分配符合授权；此时双角色恢复回执尚未出现，
历史24行不能当作新增进度。新入口无自动失败重试循环，仍遵循外部kill不恢复。

13:37:52北京时间补充：attempt-0003的Engineer与Solver都已实际恢复24轮原生状态，
随后各完成12个新轮次，36轮新paired native已发布并通过完整元数据/分片核验。
COMMITTED摘要为`4bbe5102598bcc336d8a88f4a5fb2cd3e9589625721671061566b37038681fe1`；
此次首次真实GPU验证了单发布者修复后的跨节点续训和下一代checkpoint提交。
Engineer四rank Actor/Critic Adam从13/32恢复，Solver从21/48恢复，
没有重复预热、重置优化器或仅从HF权重恢复。
核验包括全部分片大小、小文件哈希、优化器指纹/步数、ValueNorm、RNG和游标；
未重新全量哈希大tensor。新配对发布后运行目录旧24轮payload按原策略退休，
历史归档/硬链接仍保留，不能据此宣称物理空间全部释放。

实际启动为13:14:30，保守完成窗口按原14–20小时排期换算至10月13日03:14–09:14。
首批正常轮约44–48秒不包含checkpoint及评测开销，不用于宣布加速比。
控制器及worker在13:38仍存活，PID1138383保留，独立GPU记录器最新采样13:38:17。
尚未生成任何新的消融终点评分，后续仍按既定顺序完成训练、96/144评测和审计。
证据见[首次新配对核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/FIRST_NEW_CHECKPOINT_VERIFIED.json)
及[实际启动排期](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/TIMING_PLAN.json)。

### 2026-10-12 14:32 北京时间：暂停实验，只读检查双节点

用户先提出在第二项消融前加入原Full144延长训练及三系统曲线，随后明确撤回
继续准备/启动的要求，要求暂停全部实验并查看node3/node4当前活动。
未创建或启动延长训练；后续不据这条已撤回请求继续实现或运行。

without_peer attempt-0003的SIGNAL记录为14:25:40、SIGTERM15；
工具会话报告已由stop_bash停止。双角色最终metrics为76轮，
完整paired native72核验通过，摘要
`4f56da0f6203ae6711072f0c000f63fc7c1b46ab6ce10c747330d4c5c5967b2a`。
按此前外部kill不恢复和此次明确暂停指令，没有自动重启。
新增当前self_fix01的持久CANCELLED，实际使用冻结运行代码核验拒绝启动，
旧两小时等待器和首个serial入口也保持取消。原Full144结果与native不变。

14:31:57只读探测node3八卡均0%/0MiB，无GPU计算进程；
node4 GPU1–7空闲，GPU0只有UID1002/hongyuanzhu的PID1138383，
可见入口为O-Voxel Stage3训练，8092MiB，pmon三次SM为21/18/19%。
node3原生训练service及两节点filler service均inactive/MainPID0，
本地原控制器、worker和阶段观察器均已退出。未操作其他用户进程；
独立5分钟GPU历史记录器仍存活，仅做只读记录。

### 2026-10-12：代码、切分和当前进展的可验证交接

用户要求将完整代码、算法、数据切分和实验结果提交GitHub，并在本地提供平台凭据文件；
随后提供凭据，明确GitHub可以push，HF/W&B只做认证验证并给出checkpoint上传计划。
全过程不启动训练、评测、HF上传或W&B run。

新增[离线复算入口](../scripts/reproduce_ma_progress.py)及
[交接说明](./REPRODUCE_MA_SAO.md)。Git证据包包含4225份文件、182643105字节，
保留七个冻结执行快照及原始路径，所有payload大小和SHA256通过核验。
系统Python无需额外包/GPU/API，实际重建了1152行训练ledger，
从逐题记录重算Full96的180/178与Full144的186/179，并确认without_peer
为76行metrics但native72配对元数据。37项打包/复算CPU测试通过。

数据切分明确披露：训练任务ID0–239与main200无实例重叠，
selection48则全部包含在main200；strict197额外排除291、414、467的goal关联。
事后非selection152子集不替换原主终点，不包装成新未触碰测试。
未改写原始协议、分数或选择记录。

AReaL完整工作树补丁包含tracked与untracked修改；从干净基准
3cf0dfbd2b0fbeabd6977184980e189d1567747a重建1180份文件并逐文件对照摘要，
避免只提交vendor路径却遗漏本地运行代码。大型模型不进Git，
19组模型库存约1051.84GiB；最近未完成状态更新为native72，而非旧交接的native24。

三个凭据位于仓库外私有目录，未复制到Git或打印内容。
GitHub文件实际为SSH私钥：成功认证xudongwu-0、读取coevolve/main，
receive-pack dry-run通过；HF账户wxd02、W&B账户2373025856w只读认证成功。
HF的wxd02/coevolve dataset查询404，没有创建仓库；W&B没有创建run。
正式GitHub推送结果以远端commit核验为准，不把dry-run记成已发布。

### 2026-10-12 15:14:47 北京时间：GitHub首次交接发布核验完成

主交接提交
[d3fb85964c55c705518118d098057e46f8d28ac2](https://github.com/xudongwu-0/coevolve/commit/d3fb85964c55c705518118d098057e46f8d28ac2)
已通过普通fast-forward推送到xudongwu-0/coevolve的main，没有force或amend。
随后SSH `ls-remote`确认远端SHA与本地提交相同，工作树干净。
从该提交导出的全新独立目录成功完成标准库离线复算，输出与包内回执一致。

最终Git索引包含4807项，其中4225份payload和13份包元数据完整纳入，
1249个不同blob核验没有凭据路径、已提供凭据值或已知密钥特征；
最大普通文件31298663字节，没有模型张量。冻结证据的原始空白逐字保留，
避免格式化破坏SHA256；当前代码的Git空白检查通过。
三份历史训练取消门控仍有效；没有创建HF仓库、W&B run或上传模型。

### 2026-10-12 15:44:40 北京时间：只上传历史97%的Actor checkpoint

用户15:21撤销此前多checkpoint上传计划，仅保留历史194/200的最高点；
15:34进一步要求只上传checkpoint、不上传Harness、不下载回验。
已按该范围完成私有HF dataset
[wxd02/coevolve](https://huggingface.co/datasets/wxd02/coevolve)发布，
不可变revision为`d0e84414e8ef9e2b6a84585e1365189faf501051`。

上传内容仅为历史`alfworld_ma_convergence_v1r4/block-01/round0096`的Engineer/Solver
Actor，两者均对应`epoch0epochstep95globalstep95`。这属于历史路线累计144轮，
不是共同初始化Full144的93%版本。原200题记录与冻结选择逐项核对，
带选定Harness为194/200、同一Solver无Harness为185/200。

两个Actor共20份checkpoint文件、37679454196字节，约35.09GiB；
包含原权重、配置和tokenizer，不包含Harness、Critic、Adam/native状态、
其他checkpoint或评测数据。所有文件均在上传前计算完整SHA256并核对已有摘要，
上传后仅用远端清单、大小及服务器LFS/Git对象摘要验收，**没有下载模型回验**。
具体身份与回执见[精确清单](../reproducibility/historical97_20261012/CHECKPOINT_UPLOAD_MANIFEST.json)
及[远端核验](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json)。

首次临时硬链接准备受Engineer源文件权限约束而停止，尚未产生远端模型写入；
随后改为SDK直接只读上传原文件，未修改其所有权或权限，临时链接已清理。
本地原checkpoint全部保留。W&B没有创建run或上传内容，训练与Full144延长继续暂停。
