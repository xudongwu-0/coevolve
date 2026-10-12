# WHALE local extensions

The supported publication entry points are documented in [the project README](../README.md),
[the method](../docs/METHOD.md), and [the run guide](../docs/RUNNING.md).

- `three_arm_hwh.py execute`: H -> W -> H comparison of trainable/fixed internal editors and an optional external proposer.
- `scripts/prepare_run.py` (at repository root): prepare explicit assets and GPU configuration.
- `harness_learner.py`: code-editor SFT; `harness_grpo.py`: outcome-GRPO updates.
- `alternating_solver_train.py`: native WHALE RSFT for the solver.
- `alternating_pool.py`, `math_runtime.py`, `math_worker.py`: independent sampling replicas and local transport.
- `harness_code_contract.py`, `harness_probe.py`: constrained editable code and runtime checks.
- `math_gate.py`: the earlier experimental rollout-routing gate; see [negative results](../docs/GATE.md).

Other preparation, audit, analysis and historical experiment scripts are retained as research source.
Their dated `runs/...` inputs are **not included**. In particular, the historical
`three_arm_hwh.py prepare` shortcut and `report_three_arm_hwh.py` require original experiment
artifacts; use the explicit preparation script for a new run. The published aggregate
results and plots are under `docs/`; checkpoints and raw trajectories are not distributed.
