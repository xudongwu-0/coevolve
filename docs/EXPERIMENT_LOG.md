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
