# MA-SAO: full control and four single-seed ablations

Status: **in_progress**. Primary endpoint144; original200 development questions.

| Configuration | Saved system | Empty Harness | Full minus ablation, pp (paired95%) | Holm p |
| --- | ---: | ---: | ---: | ---: |
| Historical97% (not a causal control) |194/200 (97.0%)|185/200 (92.5%)|not applicable|not applicable|
|[full](./full/RESULT.md)|186/200 (93.0%)|179/200 (89.5%)|control|control|
|without_peer|pending|pending|pending|pending|
|without_context|pending|pending|pending|pending|
|without_value_norm|pending|pending|pending|pending|
|always_joint|pending|pending|pending|pending|

Always-joint preserves third-round peer correction and the environment/generated budget,
but consumes20% more role trajectories. It is not an equal-update or equal-compute comparison.
Intervals and exact paired tests describe these questions, not independent training seeds.
Four-comparison Holm adjustment is reported only after all four ablations finish.
Reused development data and historical selection limit confirmatory claims; null/reverse results stay.
Costs and paired failure/reverse cases are retained in each linked arm report.
The previous interrupted22 rounds and the96 calibration episodes are additional costs,
not omitted or represented as same-optimizer continuation.

[Historical path audit](./HISTORICAL_PATH_AUDIT.json)
[Speed policy](./SPEED_POLICY.json)
