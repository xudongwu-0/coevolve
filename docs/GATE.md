# Earlier rollout-routing gate: method and negative result

This is a separate experiment from the trainable-editor pilot. It used a different128-question held-out partition and historical evaluation with eight samples per question. Do not mix its values with the draw0 editor curves.

## Implemented flow

1. Collect one common initial trajectory on each of128 training tasks:78 successes and50 failures.
2. Externally label failures as weight10, both20, harness19, scoring1. These labels are predictions, not causal proof.
3. Allocate the remaining384 samples using masses2 for weight/both,1 for success/uncertain,0.25 for harness/scoring/environment; deterministic weighted scheduling, maximum8 samples per task,512 total.
4. Keep every verifier-successful trajectory for the same native RSFT. Evaluate the same external harness proposal pool with240 additional development rollouts and apply WHALE selection.

The WHALE baseline uses four samples per task,512 total. It does not use the gate's labels: grouped WHALE counts are post-hoc analyses using our annotations. This implementation changes training-sample allocation; it does not implement a separate label-targeted harness repair branch.

## Equal-rollout comparison

| Metric | WHALE | Gate |
|---|---:|---:|
| Training rollouts |512|512|
| Harness-search rollouts |240|240|
| Total task rollouts |752|752|
| Successful training trajectories |312|308|
| Training questions with at least one success |114|103|
| Test successful samples /1024 |699|678|
| Test mean over8 |68.26%|66.21%|

Rollout savings were **0%**. Gate accuracy was2.05 percentage points lower; the historical paired interval was[-5.57,+1.46] points. Training+search generated tokens increased2.90%, while SFT response tokens decreased0.77%. The50 external annotations were not cost-metered. There is no equal-accuracy budget curve from which to infer fewer samples for the same performance.

## Failure mechanism and sample efficiency

At this finite budget, all19 harness-labeled tasks received no follow-up after their first failure. Yet baseline follow-ups yielded24 successful trajectories on13 of those tasks. A first failed attempt labeled “harness” can still produce useful weight-training data on another attempt.

Compared by exact sampled slots, the gate removed83 baseline slots yielding39 successes and added83 slots yielding35 successes. Training coverage fell by11 questions. “Which component seems responsible?” is therefore not equivalent to “where is the next rollout most useful for learning?” The labels are nonexclusive, and the gate did not measure the marginal return of alternative interventions.

A cached offline diagnostic enforcing at least two samples per task restored coverage103→110 but reduced successful trajectories308→299 at the same512 total. No new model was trained for that diagnostic, so it has no resulting test score and is not a validated fix.

The present evidence favors studying marginal training value, coverage and total intervention cost jointly. It does not support a claim that the current gate improves sample efficiency or can only help.
