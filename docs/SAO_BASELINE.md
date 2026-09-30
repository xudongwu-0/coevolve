# Independent editor/executor SAO baseline

This optional integration calls the existing open-source trainer. It does not
implement PPO, GAE, DIS, value losses, or an optimizer replacement in Coevolve.
It leaves the original RSFT solver / GRPO editor experiment unchanged.

## Provenance and terminology

- Paper: [Single-Rollout Asynchronous Optimization for Agentic Reinforcement
  Learning](https://arxiv.org/abs/2607.07508).
- The pinned [SAO implementation](../third_party/sao) is **unofficial**, Apache-2.0,
  revision `ac2728149041b5f15ceb75413f467ce0659179cf`.
- Its published integration patch is applied to AReaL revision
  `3cf0dfbd2b0fbeabd6977184980e189d1567747a`.
- Both roles invoke `areal.PPOTrainer` with the upstream SAO recipe: one rollout
  per prompt, a separate value model, two critic updates per actor update,
  decoupled policy loss, and token-level ratio rejection bounds `[0.3, 5.0]`.
  These settings are inherited from the pinned SAO config, not renamed GRPO.
- This is an independent-critic baseline, **not MAPPO**. A future centralized
  critic must explicitly condition on the joint editor/executor state, define
  cross-role credit assignment, and be compared under matched rollout budgets.
- SAO is related to PPO-family actor-critic training but is not identical to
  textbook PPO. Do not describe this nonofficial reproduction as the authors'
  official implementation.

## Deliberately bounded first experiment

The task is MATH number theory, Level 2-4 with integer boxed answers. The pilot
uses 16 official-train questions for executor updates, the next 4 for editor
rewards/selection, and 8 official-test questions for reporting. SHA256-based
selection is deterministic and excludes train/test duplicates. These tiny
splits test wiring and checkpoint updates; they are not statistically meaningful
performance evidence or reproduction of the published Coevolve numbers.

The editor emits a Python `CandidateEnv(ReToolEnv)` module containing exactly
three literal attributes: `SYSTEM_PROMPT`, `USER_PROMPT_TEMPLATE`, `MAX_TURNS`.
The existing Coevolve contract is checked first, then a narrower AST-only parser
extracts those attributes. **Generated Python is never executed.**

The solver runs text-only reasoning with a fixed "finish the solution" retry
message if no boxed answer appears, up to the harness turn limit. Total generated
tokens are capped at 512 per trajectory across turns. The initial parser/verifier
and answer-format requirement are fixed. Tool execution and arbitrary recovery
hooks are **not** implemented by this pilot; do not compare it directly to full
WHALE ReTool runs without aligning the action space and protocol.

Both actors start from independent copies of Qwen2.5-1.5B-Instruct, not the
historical Qwen3.5-2B warm starts. The editor learns from development-set accuracy
delta under a frozen executor. Invalid harness actions receive an explicit -1
environment reward. System/network/model failures raise errors instead of being
converted to incorrect answers. The executor learns from the original WHALE
strict boxed-answer reward and does not filter away failed trajectories.

The sequence is H -> W: train the editor, sample fresh candidate harnesses,
select against the incumbent on development data, then train the executor with
that selected harness. Each role has its own actor, critic, and output directory.
The first pilot is not a multi-cycle continual-training study.
Post-training selection samples 16 proposals at temperature 0.3 by default;
training rollouts retain temperature 1.0. Both proposal settings are explicit
CLI options and recorded in the run manifest. All proposed sources are validated,
and no valid proposals is an error, never a substitution of hand-written code.

## Environment and upstream setup

From the repository root:

```bash
git submodule update --init --recursive
bash third_party/sao/scripts/bootstrap_areal.sh
```

Use a separate Python 3.12 environment compatible with AReaL's
`vendor/AReaL/pyproject.vllm.toml`. The original Coevolve worker lock pins
Transformers 5.3.0, which this AReaL revision excludes; **do not replace that
environment or install these dependencies into it**. The initial environment
for this integration is `.venv-ppo` (the directory name predates choosing SAO).
No Ray cluster or other user's distributed service is required: the pilot uses
AReaL's `LocalScheduler` and manages only its own subprocesses.
The critic shares the actor RPC process (`scheduling_strategy.fork=false`),
but still has its own model and optimizer. This avoids the upstream fork
readiness timeout during cold imports. A scheduling-only trainer subclass
registers worker cleanup before model construction, including failed startup;
all rollout, loss, advantage and optimization methods remain upstream.

```bash
export PATH="$PWD/.venv-ppo/bin:$PATH"
export PYTHONPATH="$PWD/whale_local:$PWD/third_party/sao/vendor/AReaL:$PWD/third_party/sao"
```

The provisioned runtime uses Python 3.12.3, PyTorch 2.10.0+cu129, vLLM
0.19.1, Transformers 5.7.0 and TorchAO 0.16.0+cu129. It was installed from
the pinned AReaL `uv.vllm.lock`, not by resolving arbitrary latest packages.
To reproduce the dependency installation in an already-created isolated
environment with `uv` available:

```bash
mkdir -p .venv-ppo/provisioning/tmp
cp third_party/sao/vendor/AReaL/pyproject.vllm.toml .venv-ppo/provisioning/pyproject.toml
cp third_party/sao/vendor/AReaL/uv.vllm.lock .venv-ppo/provisioning/uv.lock
uv export --project .venv-ppo/provisioning --frozen --no-dev --extra vllm \
  --no-emit-project --no-hashes \
  --output-file .venv-ppo/provisioning/requirements-vllm.lock.txt
uv pip install --python .venv-ppo/bin/python \
  -r .venv-ppo/provisioning/requirements-vllm.lock.txt --torch-backend cu129
uv pip check --python .venv-ppo/bin/python
```

Using the conventional `pyproject.toml` filename inside the environment avoids
uv interpreting `pyproject.vllm.toml` as an unsupported lockfile. No Megatron,
flash-attn or flash-linear-attention build is required for this FSDP/SDPA pilot.
The pilot disables optional vLLM general plugins (including unused AWEX);
the upstream AReaL worker extension remains explicitly configured.

Place public model files under `assets/Qwen2.5-1.5B-Instruct` and the official
MATH number-theory parquet files under `assets/math_source/{train,test}.parquet`.
The run records file digests. Download provenance must be recorded separately
when obtaining these assets; a local filename alone does not prove its revision.
The provisioned model revision is `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`;
the dataset revision is `21a5633873b6a120296cce3e2df9d5550074f4a3`.
Each asset directory contains `provenance.json` with source URLs and verified
file digests. The source number-theory split contains 869 train and 540 test rows.

## Run

Only run after the GPUs are explicitly assigned and idle. This command does
not stop external processes, start a utilization daemon, or contact an external
LLM provider:

```bash
.venv-ppo/bin/python -m sao_bridge.pilot \
  --model assets/Qwen2.5-1.5B-Instruct \
  --source-data assets/math_source \
  --run runs/sao_math_smoke \
  --steps 2 \
  --train-gpus 0,1 --eval-gpu 2 --propose-gpu 3
```

The pilot records its child PIDs, node, command, assigned GPUs and logs in
`processes.json`. The evaluator FastAPI binds only to `127.0.0.1` and loads a
**local frozen model**. It is not an API to this interactive assistant, and
running FastAPI does not grant access to a provider model or a cheaper model.
No external API credentials are required.

Each stage calls the upstream trainer and writes both actor and critic HF
checkpoints. `checkpoint_verification.json` compares persisted tensor values:
actor against initialization, critic between saved update steps. Held-out
evaluation reloads the final executor checkpoint. `COMPLETE.json` is written
only after all stages, checkpoint-change verification and held-out evaluation
succeed. A checkpoint or a started process alone does not indicate completion.
Use a fresh run directory; automatic recovery/optimizer continuity across
separate pilot invocations is not claimed.

## Tests

```bash
python3 -m unittest discover -s tests -p test_sao_bridge.py -v
CUDA_VISIBLE_DEVICES='' .venv-ppo/bin/python -B -m unittest discover \
  -s tests -p 'test_sao_bridge*.py' -v
.venv-ppo/bin/python -m pytest third_party/sao/tests/test_dis.py \
  third_party/sao/tests/test_dis_loss.py third_party/sao/tests/test_core_algorithms.py \
  third_party/sao/tests/test_areal_integration.py
```

The pinned upstream suite has a known inconsistent assertion in
`test_areal_integration.py::test_dis_masks_out_of_band_tokens`: it expects
ratio `0.5` to be rejected with configured bounds `[0.3, 5.0]`. The actual mask
accepts it, consistently with those bounds. The upstream source and test are
left unchanged; report this failure rather than claiming the suite is all green.

## Verified node4 smoke run (2026-09-28)

The completed shared-storage run is
[`runs/sao_math_smoke_20260928_03`](../runs/sao_math_smoke_20260928_03).
Its [completion receipt](../runs/sao_math_smoke_20260928_03/COMPLETE.json)
and [tensor checks](../runs/sao_math_smoke_20260928_03/checkpoint_verification.json)
were produced only after both training stages and final checkpoint reload.
The manifest records the exact adapter source hashes used for this run.

| Check | Observed result |
| --- | --- |
| Editor actor / critic updates | 2 / 4 |
| Executor actor / critic updates | 2 / 4 |
| Persisted parameters | Both actors changed from initialization; both critics and value heads changed between saved steps; inspected tensors finite |
| Fresh low-temperature proposals | 16/16 passed the restricted contract |
| Development selection | All 16 proposals and incumbent scored 0/4; incumbent retained |
| Held-out test, before / after | 2/8 (25%) / 1/8 (12.5%) |
| Adapter CPU tests | 14 passed |
| Pinned upstream CPU tests | 9 passed, 1 unchanged assertion failure described above |

Each actor consumed two batches of four trajectories. The prepared executor
training split contains 16 questions, but this two-step smoke optimized only
eight trajectories, not a full pass over that split. Its batch mean task rewards
were 0 and 0.25. Both editor optimization batches received only invalid-action
penalties (mean -1). Asynchronous prefetch produced 11 recorded editor episodes
in total, including one valid action with reward +0.25; that late extra episode
must not be counted as a successful editor training update.

This is **successful integration verification, not successful performance
improvement**. The editor's training-time format validity is the immediate
research bottleneck. Before a meaningful MAPPO comparison, establish reliable
valid-action sampling or a declared format warm start, increase the data/update
budget, and use matched evaluation budgets and multiple seeds. Do not tune
against these eight held-out questions. No MAPPO/centralized critic is present.

Earlier attempts are retained for diagnosis: run `_01` exposed the upstream
critic-fork timeout; run `_02` completed editor optimization but had no valid
post-training proposal. Neither has a completion receipt. The final run uses
the documented scheduling cleanup, clearer format instructions and explicit
proposal budget. All owned GPU workers and the frozen evaluator were stopped
after verification; no utilization daemon was started by the pilot.

Resource-state caveat: `bruli-h200-load.service` was externally started again
at 08:00:08 UTC after the experiment, with `NRestarts=0`. At the final check,
its eight workers each occupied about 80,471 MiB. The trigger was not identified;
the pilot did not start it or change its persistent startup configuration.
Thus our experiment workers are gone, but node4 must not be assumed idle for
the next run. Recheck allocation and coordinate before stopping any workload.
