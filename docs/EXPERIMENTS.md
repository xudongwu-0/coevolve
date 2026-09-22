# Pilot results and interpretation

The exported records describe `whale_three_arm_hwh_20260918`, a completed local H→W→H pilot. See [aggregate JSON](results/pilot.json) and [figure](figures/three_arm_curve.png). Raw trajectories and checkpoint tensors are stored separately; this repository alone cannot independently recompute the original model outcomes.

## Statistical convention

Every point in the published figure uses the same128 MATH number-theory questions, draw0, seed42 and temperature1.0. Percentages are solved/128. Each task has a2048-token total assistant budget and1024-token per-call limit, thinking disabled. The initial harness allows two turns; candidates allow up to three while preserving the total output budget.

| Historical configuration | Single draw used in figure | Original mean over8 draws, for comparison only |
|---|---:|---:|
| Base solver, base harness | 54/128 =42.19% |41.21%|
| Trained solver, base harness |78/128 =60.94%|55.57%|
| Base solver, historical improved harness |73/128 =57.03%|56.45%|
| Trained solver, historical improved harness |89/128 =69.53%|69.53%|

The trained-solver/base-harness eight draw counts were78,73,63,69,80,72,63,71;569/1024=55.57%. The new pilot start repeats the same per-question success/failure outcomes as historical draw0. Mean over8 is not pass@8. Historical four-way configurations are independent combinations, not consecutive steps of the new experiment.

## Three-arm pilot

| Arm | Start | H1 | W update | H2 | Development rollouts | New W rollouts | Successful SFT trajectories | W steps | E steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| External proposer |78|90|92|92|1056|32|22|9|0|
| Fixed internal2B |78|69|71|84|432|32|21|9|0|
| Trained internal2B |78|69|71|93|480|32|21|9|8|

All success counts have denominator128. W uses RSFT at1e-7. Each E phase uses up to four outcome-GRPO steps at1e-6; groups with no reward variance skip optimization. All stages are frozen before evaluation begins. The run took approximately140 minutes on the original server, including startup/storage fixes, with up to13 sampling GPUs and one training GPU; this is not a hardware-independent runtime estimate.

The two internal arms selected identical H1 and produced identical W-training trajectories; the resulting W checkpoint was reused and charged logically to both arms. At H2 their solver, incumbent harness, editing inputs and matched proposal seeds agree, while editor weights differ.

## What improved

The trained editor's second phase changed matched development candidate gains from `[0,4,0,0]` to `[8,4,8,8]` solved tasks out of48. Three no-op proposals became useful changed code. This improved the chance of reproducing a good incomplete-answer recovery mechanism and resulted in93/128 vs84/128 for the fixed editor under the recorded selection protocol.

That strong42/48-development candidate already existed among the eight pre-training proposals. The selection pool contains only four post proposals and the incumbent. Consequently the result does not show that E training discovers a new best mechanism or beats retaining all pre-training candidates. The fixed editor repeats proposal seeds and is not an optimized equal-cost search baseline. E initialization and external model inference are additional costs.

## What failed

Both internal H1 candidates improved development28/48→38/48 but reduced evaluation78/128→69/128. There were17 gains and26 regressions; missing boxed final answers increased41→46. The selected global instruction emphasized fixing errors and comparing integers; overfitting a narrow failure description is a plausible explanation, not an isolated causal finding.

The first editor update also reduced matched mean development gain from6.75 to5 tasks; the second increased it from1 to7. These are current-context training rewards, not an independent test of editing generalization. The result supports continued experiments on trainable editors, not a guarantee of monotonic co-evolution or sample-efficiency superiority.

The external arm's final92 vs internal trained93 is a one-question difference in a small single-seed pilot. It does not establish that the small editor is generally better than an external model. The benchmark subset has historical use and is neither full MATH nor MATH-500.
