# WHALE-style ALFWorld baseline: local adaptation

**Current launch contract: v6, in the final section below.** The v1/v2/v3/v4/v5 source
freezes and their earlier command examples are historical and must not be
queued with the current source. Nothing in this document has activated a unit.

## Status and correct attribution

**Prepared and CPU-tested; no baseline GPU training/evaluation has been run by this
preparation. No ALFWorld score is claimed for WHALE.**

Official WHALE is **online rejection-sampling fine-tuning (RSFT), alternating with
Meta-Harness search**. It is neither a static Harness-R1 harness nor a renamed
GRPO/SAO objective. Primary sources:

- [Official repository](https://github.com/krafton-ai/WHALE), local gitlink
  `WHALE`, revision `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`.
- [Paper, arXiv:2609.00196](https://arxiv.org/html/2609.00196v1).
- `WHALE/README.md`, `WHALE/run/lib/alternate.sh`, and the SearchQA
  `verl/workers/actor/dp_actor.py` and `meta_harness/` implementations.

On 2026-10-03, the official README/reproduction release lists SearchQA, math and
chess, not ALFWorld. The fetched v1 paper HTML contained zero occurrences of
“ALFWorld”. **No published WHALE ALFWorld numerical result was found in these
sources.** Any measured result from this adapter must be labeled
**“Local WHALE-style ALFWorld adaptation (Qwen3.5-9B)”**, not the published WHALE
baseline.

## What is reused and what is adapted

`whale_local/sao_bridge/alfworld_whale_baseline.py` executes:

1. Sample fresh Solver episodes under the current harness, through the actual
   AgentBench ALFWorld controller/worker. The verifier's binary terminal reward
   decides acceptance; token caps, malformed calls and task-limit failures are
   failures, while infrastructure failures abort rather than become rewards.
2. Accept success trajectories (`reward > 0.5`); train on their **exact generated
   assistant token IDs**, masking prompts and tool observations. There is no
   value network, advantage, clipped PPO loss, reward model or GRPO update.
3. Invoke official **`DataParallelPPOActor.update_sft_policy` unmodified**:
   one epoch, token-weighted response NLL, minibatches of 64 assistant calls,
   singleton microbatches, AdamW LR `1e-6`, weight decay `.01`.
   Full FP32 master-weight shards and Adam/RNG state persist between batches
   and cycles; BF16 HF exports are for subsequent local vLLM sampling.
4. Stop training, freeze the updated Solver, evaluate the incoming harness on
   development tasks, then conduct iterative Meta-Harness search. Each
   iteration supplies the incumbent, scored/invalid candidate archive and
   fresh current-Solver development failures to the **fixed original H0**
   proposer. Candidate patches are compiled by the official Harness-R1 patch
   compiler and evaluated in actual ALFWorld episodes.
5. Use the official **`compute_pareto_frontier`** and
   **`pick_accepted_from_frontier`** functions without changes: maximize success,
   then minimize turns. They are loaded from their source ASTs to avoid
   importing the upstream Claude/API transports or the unavailable
   `verifiers` package. The archive remains under the same frozen Solver
   throughout a search phase; each new weight phase receives the selected
   harness, and the next search re-evaluates its incoming harness from scratch.
6. Repeat; select the final system before opening final evaluation.

ALFWorld patch files occupy the upstream selector's `harnesses/hN/harness.py`
layout for compatibility, but **contain JSON, not executable Python modules**.
They are passed to the existing ALFWorld patch compiler; the selector only
checks artifact existence. Generated code hooks are governed by that compiler's
existing validation/runtime restrictions.

**History/feedback is present, but full upstream MH parity is not claimed.**
These are not 30 independent H0 proposals: within each cycle, iteration 2
receives iteration 1's selected harness source, all prior candidate
scores/descriptions and invalid-candidate reasons, plus newly evaluated
failures of the current incumbent. The three proposals **within one iteration**
share that context and differ by slot/seed; they do not see each other's results
until the next iteration. A CPU test verifies the next prompt contains the
previous winner's source/score and fresh winner failures. The local code
implements this feedback loop; only the Pareto/acceptance functions and RSFT
update are source-native. It does **not** execute upstream `run_evolve`,
Claude's filesystem/tool session, or the full MH skill. Prior unsuccessful
candidate source files are stored, but only their compact descriptions/scores,
not their complete source/trajectories, enter the next prompt.

### Explicit deviations from the paper implementation

- The official default proposer is **Claude Opus 4.7**, an external coding-agent
  session. Here it is the local, frozen `assets/Harness-R1/harness-r1` (original
  **already-RL-trained H0, not fresh SFT**). It receives bounded prompts rather
  than arbitrary filesystem/code-agent tools. No external model calls occur.
- Search operates over the existing validated **ALFWorld code-hook patch
  interface**, not arbitrary full SearchQA `CandidateEnv` Python programs.
  The iterative archive/evaluation/selection logic is retained, but this is a
  domain/action-space/proposer adaptation, not a faithful paper reproduction.
- The original W0 is `assets/Harness-R1/target-agent-sft`. The incoming harness
  is the already-frozen original-H0 proposal01 used by the corrected MA
  initialization. Its eight-proposal initialization cost is inherited and
  explicitly **not counted as zero total preparation cost**.
- Assistant calls are separate SFT examples; minibatches count calls rather
  than entire multi-turn trajectories. Only successful episodes supply calls.
- The native VLM FSDP loader/export and response-only log-probability transport
  come from `harness_r1_searchqa.grpo_train`, **not its GRPO loss or update**.
  The singleton forward removes all batch padding and sets `use_cache=False`,
  avoiding recurrent-state mixing and full-prompt vocabulary-logit allocation.
- For an auditable, exact token-mean gradient, **each of eight FSDP ranks sees
  the same singleton call schedule**. Averaging identical rank gradients gives
  the native objective, including uneven/short accepted batches, without
  dropping successes or adding duplicate logical samples. This is deliberately
  compute-inefficient: approximately eightfold redundant forward/backward
  work relative to ideal data partitioning. Never call it compute-matched.
  This is not the shared SAO/AReaL joint actor/ref cache path.
- Zero-success batches consume their sampled-episode budget but make **no**
  optimizer step. Invalid/truncated proposals consume proposer slots/tokens
  and are archived, not silently replaced with successful baselines.

## Frozen specification

`configs/alfworld_whale_baseline.json` binds the source/input SHA-256s and:

| Item | Frozen setting |
|---|---|
| Alternation | 2 cycles, each 10 online RSFT batches followed by 5 MH iterations |
| RSFT samples | 32 fresh episodes/batch; **640 MA-prefix records over 240 distinct legacy games** |
| MH proposals | 3/iteration; **30 total** |
| Development | 24 separate train-only games, goal-disjoint from legacy training and core200 |
| Development evaluations | Incoming harness + every valid proposal; at most **768 episodes** |
| Solver sampling | Temperature 1, top-p 1, 4,096 total generated tokens/episode |
| Solver context / task steps | 32,768 context; existing official worker's 50-step limit |
| Proposer sampling | Temperature .7, top-p 1, 4,096 output tokens; fixed seeds |
| Proposer evidence | Contract only from frozen packet template, then fresh dev traces; 16,000-character trace excerpt |
| Final evaluation | Exact core200 protocol, one sample, seed 6,000,045 |
| Selection | Development success first, then turns; no final-task training or selection |

Solver training uses **the first 640 ordered records of the current MA
`runs/alfworld_joint_h0w0_8h_v1/data.json`**, with `task_split=test_set` and
legacy task IDs 0–239. Repeated tasks are retained; there is no new shuffle or
deduplication. Historical engineer messages and critic evidence are discarded.
The actual Solver receives only the live environment episode.

Only **development** uses the separate train-only catalogue `sao_train_v1`,
checked against all 500 legacy games and reserved validation goal groups.
Its numeric IDs belong to a different catalogue namespace; split and hash are
part of identity. A seeded shuffle selects 24 unique development goal groups.
These additional reward episodes are not part of the 640 training budget.
`prepare` records the exact
ordered task lists, model verification and adapter/config hashes in
`FROZEN.json`, including SHA-256s of the three game assets for every selected
train/development/final game. Execution rechecks the freeze and refuses an existing execution
directory: mid-run continuation is **not implemented**. Interrupted runs are
incomplete, never successful.

Principal hashes:

| Input | SHA-256 |
|---|---|
| W0 `model.safetensors` | `9f52da7a3e4166e3af6b6ba8c0722a8abb396a7ac876e1cb4f83d2453c7bc2fb` |
| H0 `model.safetensors` | `7d16d5ad5734af16b1fd0619caeb04270a024df86ca6e6c6e9bdb60e45f39af7` |
| Final protocol | `08fbac528fa502434438c4ea27725df437020e3859714e5895b7a671dfde7275` |
| Initial selected patch, file bytes | `efe32cf284f61473a9025f0f2c2b1dfc8708fb5970229bd1557db7cf75c498f6` |
| Current MA task records | `6e8a952f388e42d167e4388f638d5b5e94043f94bfaa158c4fbf420b11c6c12a` |
| Legacy training catalogue | `e6edcf2b2de1b32c8f011e52162012d7aa3356546cb4eb2f11ab0e7b941857ae` |
| Separate development catalogue | `118a6573cd7ddc809a70393767d5f16548cde8f248a9f42266198e16c8f28f81` |
| Native RSFT actor source | `10b698c8a358270a685df1504a30caef8f1c04d65b503409d6069cd0827f3d7f` |

File-byte patch hashes differ from the canonical patch hash in the MA audit;
neither is a different harness.

## Comparisons and limitations

The corrected initialization audit
`runs/alfworld_joint_h0w0_8h_v1/INITIAL_AUDIT.json` reports:

| Existing system (not a WHALE result) | all200 | strict197 |
|---|---:|---:|
| W0, empty harness | 148/200 = 74% | 146/197 = 74.1117% |
| Original H0 selected harness + W0 | 170/200 = 85% | 168/197 = 85.2792% |
| This WHALE-style adaptation | **not run** | **not run** |

The inherited baseline is not reselected using those outcomes. Final evaluation
uses the identical ordered `all_task_ids`, exact per-task seed offset, sample 0
and `strict_goal_disjoint_task_ids` from
`runs/alfworld_core_200_oneshot_v2/protocol.json`. Report both denominators;
three all200 games are train-goal-related in the legacy experiment. These
already-viewed results are **not a new sealed test**.

This is single-seed exploratory research. The **640 WHALE Solver episodes do
not equal MA's 1,024 total trajectory checkpoint, which contains 512 Solver
and 512 editor records**, including critic warmup. WHALE is aligned to the
first 640 MA Solver task records, not equal total allocation. Accepted tokens,
optimizer-step counts, proposer
training, critic work, search trials, context lengths, GPU-hours and wall time
all differ. Archive selection can overfit 24 dev games. A local 9B proposer
is not the paper's stronger coding agent. Search invalidity rates must be
reported. Compare quality at stated budgets, not as an algorithm-isolated
causal claim. Per-call tokens, raw proposal usage and episode reports are
persisted; `COMPLETE.json.cost` aggregates logical inference costs but **does
not measure GPU-hours/FLOPs**. Preserve systemd elapsed/resource logs separately.

Checkpoint storage is substantial (FP32 optimizer/master shards plus BF16
exports for up to 20 nonempty RSFT batches); allow **several TB of shared
storage**, not the full root partition. No checkpoint deletion is automatic.

## Commands and scheduling gate

From `/mnt/3fs/xudong/coevolve`, CPU-only validation/preparation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_alfworld_whale_baseline.py

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline prepare \
  --config configs/alfworld_whale_baseline.json \
  --output runs/alfworld_whale_baseline_v2/FROZEN.json
```

**Do not run the following until the coordinator explicitly assigns an idle
node after current MA/W2 work and reporting release.** The historical earliest
MA train/report deadlines are approximately 07:49/08:49 UTC on 2026-10-03;
these are not authorization or proof of idleness. Allocation checks reject
every foreign GPU process and require the verified active adaptive filler.
They never stop MA, W2 or the adaptive filler. Nothing is automatically queued.

Example future node3 foreground command, preferably inside the coordinator's
bounded systemd unit with `KillMode=control-group` and an approved wall budget:

```bash
# ONLY on assigned dev-NDH200-025, UID 1009/GID 1008:
cd /mnt/3fs/xudong/coevolve
TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen runs/alfworld_whale_baseline_v2/FROZEN.json \
  --execute-on dev-NDH200-025
```

For node4 use `dev-NDH200-026`, UID 1008/GID 1008. Both require eight assigned
GPUs, node-local `/dev/shm/sao-venv-ppo/bin/python` and the copied isolated
`/dev/shm/harness-r1-alfworld-venv/bin/python`. Launcher-managed processes are
loopback-only, local-model-only, sequentially owned and closed before the next
phase. Solver and proposer serving never overlap RSFT GPU ownership.

## Validation still required after allocation

CPU tests exercise the real official RSFT method on toy CPU tensors, including
token-mean gradients; exact official MH selection; actual source/data hashes
and cohort exclusion; patch validation; seed/token checks; fresh evidence
construction; and mocked HTTP/episode wiring. **Mocks are not ALFWorld scores.**

The first assigned run must still establish native 9B FSDP load/backward/save,
strict optimizer-shard restore, real ALFWorld controller interaction, local H0
format validity and successful vLLM reload. GPU memory, runtime and performance
remain unverified. Missing runtime dependencies, invalid hashes, controller
failures, nonfinite gradients or unsuccessful reloads abort with `FAILED.json`;
there is no alternate fake reward, foreign API fallback or silent trainer
substitution. Full `COMPLETE.json` exists only after the frozen schedule and
final native 200-task evaluation finish.

## GPU smoke and persistent node4 queue (prepared, not activated)

Revision `v2` adds the smoke/queue/cost contract; `v1/FROZEN.json` is retained as
historical preparation and intentionally fails the changed-source check. Do not
overwrite it or silently reuse its old digest.

After allocation, **standalone smoke only**, no ALFWorld tasks/reward episodes:

```bash
cd /mnt/3fs/xudong/coevolve
TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen runs/alfworld_whale_baseline_v2/FROZEN.json \
  --execute-on dev-NDH200-026 --gpu-smoke \
  --output runs/alfworld_whale_baseline_v2/gpu-smoke-standalone
```

**Historical v2 smoke, superseded by v5; it cannot authorize a current run.**
Its implementation specified two separate eight-rank processes, never executed
by this preparation. The first loads **original W0**,
executes one native forward/backward/AdamW step on two tiny synthetic
digit-response calls, and exports BF16 weights plus full FP32 local shards and
Adam/RNG states. The second loads that export, restores all FP32 parameter
shards and optimizer state, verifies full-tensor hashes on every rank, requires
Adam step 1 before and step 2 after a second real update, and exports again.
Finally all eight local vLLM replicas load the export. It records
`SMOKE_COMPLETE.json`, per-rank parameter/Adam proof hashes, step counts,
peak reserved GPU memory, elapsed time and synthetic costs. There are **zero
research trajectories**. Smoke weights can never become the baseline's W0:
the subsequent baseline starts again from the independently verified original
assets. Tiny synthetic sequences do **not** establish full-context memory,
ALFWorld correctness or any benchmark score.

Expected allocation and bounds:

- **One complete node at a time, all eight H200s.** RSFT uses 8-rank FSDP with
  the replicated singleton schedule; serving uses eight independent local
  replicas. Node3 UID/GID `1009:1008`, node4 `1008:1008`.
- Native loading initially creates a CPU FP32 model per rank; allow roughly
  320 GB CPU memory for these copies, plus checkpoint/Adam loading/export
  overhead. Host-memory peak is not yet measured; do not assume 320 GB is a
  sufficient hard cgroup cap.
- Smoke hard watchdog: 2 hours after setup, 50 minutes per native subprocess;
  several-hundred-GB shared storage for two retained optimizer checkpoints.
  Expected runtime has **not** been measured. No claim that the smoke or the
  full run fits a particular actual duration.
- Baseline hard watchdog: **12 hours** after setup (`wall_budget_seconds=43200`);
  individual training subprocesses cap at 2 hours. Exhausting a bound produces
  a failed/incomplete run, not a reduced-budget success. Up to 20 full
  optimizer checkpoints plus smoke need several TB of shared storage.
- Queue CPU waiting caps at 24 hours, with a persistent absolute deadline and
  60-second polls. An outer **39-hour** systemd limit covers waiting, smoke,
  baseline, CPU hashing/setup and shutdown. The outer process tree must use
  `KillMode=control-group`; `TMPDIR=/dev/shm`.

**Recommended coordinator-controlled persistent activation:** install the
following only after approving the frozen source and allocation policy. It is
a command recipe, not a service launched by this preparation. Set
`FROZEN_SHA` to the exact verified v2 digest in the preparation receipt; do not
compute an unchecked moving-source digest at workload start.

```bash
# On node4, after coordinator approval; this schedules future GPU use.
cd /mnt/3fs/xudong/coevolve
FROZEN_SHA='4f611d7249fc5959fcfbf8ee7e65f3af27b88417608105d8acea814eb47b6533'
sudo -n systemd-run --unit=coevolve-alfworld-whale-v2 \
  --property=Type=exec --property=User=1008 --property=Group=1008 \
  --property=WorkingDirectory=/mnt/3fs/xudong/coevolve \
  --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --property=RuntimeMaxSec=39h --property=Restart=no \
  --setenv=TMPDIR=/dev/shm --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=PYTHONPATH=/mnt/3fs/xudong/coevolve/whale_local \
  --setenv=HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline queue \
  --frozen runs/alfworld_whale_baseline_v2/FROZEN.json \
  --frozen-sha256 "$FROZEN_SHA" --execute-on dev-NDH200-026 \
  --queue-root runs/alfworld_whale_baseline_v2/queue \
  --smoke-output runs/alfworld_whale_baseline_v2/gpu-smoke-queued \
  --after-complete runs/alfworld_joint_h0w0_8h_v1/best_COMPLETE.json \
  --after-released runs/alfworld_joint_h0w0_8h_v1/best_RELEASED.json \
  --wait-seconds 86400
```

The wait phase does not construct `Launcher` or start CUDA. It requires both
MA best receipts, `status=completed`, clean released GPU identities and active
adaptive filler, then performs a **fresh live node audit**. Foreign workloads
make it keep waiting, never kill them. Before starting it checks the exact
freeze/config/adapter; smoke revalidates all pinned inputs, game assets and
original weights. It persists prerequisite receipt hashes in `STARTED.json`.
Smoke must pass before baseline execution. It will not automatically rerun a
GPU phase after `STARTED.json`, and never retries a failed/incomplete baseline
as if it were successful. The systemd unit is persistent across terminal
disconnects; this transient recipe does not install boot-time restart behavior.

## Full cost ledger, not just the 640 training episodes

`COST.json` is written on normal completion **and ordinary failure**; an outer
SIGKILL can leave only per-stage receipts. It separates:

- 640 logical train episodes;
- up to 768 additional development reward episodes, including the re-evaluated
  incoming harness in each cycle and every valid candidate;
- 200 final-report episodes; hence **up to 1,608 incremental Solver episodes**;
- 30 proposer completions, actual input/output usage and invalid counts;
- accepted assistant-call/response-token counts and actual optimizer steps;
- **eight times** those logical RSFT calls/response positions for rank-replicated
  compute. Activation-checkpoint recomputation is extra and not counted by
  that simple multiplier; this is not a FLOP estimate;
- separate GPU-smoke costs: four synthetic calls, two logical Adam steps,
  32 rank-replicated calls, actual response positions and elapsed time.

Inherited initial H0 selection physically used **384 Solver episodes**
(8 unique harness reports × 24 tasks × 2 samples), **277,838 generated Solver
tokens**, and **8 H0 proposals**. Two already-existing initial evaluation
reports add 400 historical Solver episodes. Their exact token totals and
source hashes are frozen in `inherited_initialization_cost`; these are shared
sunk costs, not reexecuted by the queue. The compact initial proposal receipts
omit proposer token usage: it is recorded as **unknown/null, not zero**.
Earlier training and failure-packet collection costs also remain inherited
and cannot be inferred from the compact selection receipts. Therefore no
fully measured end-to-end FLOP/GPU-hour total is claimed.

### Verified v2 source identity and direct baseline CLI

CPU validation: **38 tests passed**, including 19 adapter tests, and 35
subtests; three upstream torchao deprecation warnings. No GPU smoke was run
and no systemd queue was activated.

| Artifact | SHA-256 |
|---|---|
| `runs/alfworld_whale_baseline_v2/FROZEN.json` | `4f611d7249fc5959fcfbf8ee7e65f3af27b88417608105d8acea814eb47b6533` |
| Adapter `.py` | `c0b762c8d60b0bbe7082cae228e50b5266c38c7774782373134f28a30480ddc9` |
| Baseline config | `5cd5e90ebd5defbde6966961608db1d5235e555319d4070da70dd7ee92c76339` |
| Adapter CPU tests | `7d0c6cd26e7d3a5d3b0efde384f2f89f402bafc39c5baa8d36ff809f6dcad8a4` |

Historical initial reports contain **313,100** generated Solver tokens;
combined with inherited selection, the known historical total is **784
episodes / 590,938 Solver tokens**, plus the eight H0 proposals with unknown
token usage. This is separate from the up-to-1,608 incremental episodes.

Standalone full baseline, only after allocation and passing smoke:

```bash
cd /mnt/3fs/xudong/coevolve
TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen runs/alfworld_whale_baseline_v2/FROZEN.json \
  --execute-on dev-NDH200-026
```

Direct `run` deliberately does not wait for MA receipts and does not
automatically require a smoke receipt; it is the coordinator's manually
allocated escape hatch. Use **`queue` for the automated dependency chain**:
MA best complete → MA best released → exact source freeze/live idle audit →
real synthetic GPU smoke passes → full baseline. Do not invoke direct `run`
as a substitute for that queue while MA remains active.

## Historical v3: optional run-entry MA wait and persistent smoke chain

`run --wait-for-joint ROOT --wait-label best` now invokes the persistent queue
directly; no separate shell chaining or standalone smoke command is needed.
It writes a persistent absolute **24-hour wait deadline**, keeps
`STATUS.json.stage=waiting_for_joint` / `baseline_started=false`, and checks:

1. Neither `ROOT/best_FAILED.json` nor `ROOT/FAILED.json` exists. Either fails
   the queue immediately, even before completion/release receipts exist.
2. Both `ROOT/best_COMPLETE.json` and `ROOT/best_RELEASED.json` exist.
   Completion must be **`completed` or `unavailable`**, never `failed`.
   `unavailable` means only that the MA best arm has terminated without an
   available selection; it is not a successful MA score.
3. The release receipt shows no foreign GPU processes and active verified
   adaptive filler; a fresh **live node4 audit** must also be clean. A currently
   occupied live node keeps waiting without stopping anything.
4. The externally approved frozen digest, adapter and configuration still
   match. Smoke rechecks complete model/input/game-file hashes.
5. Run real original-W0 synthetic GPU smoke. The baseline is called **only**
   if it returns `passed` for that identical freeze. No smoke model becomes
   the baseline initialization. `STATUS.json` distinguishes `gpu_smoke`,
   `baseline`, `completed` and `failed`.

`--wait-for-joint` intentionally rejects combining `--gpu-smoke` or `--output`:
it already implements the complete smoke-first chain, with
`gpu-smoke-queued/` separate from `execution/`. This queue is node4-only;
node3's W2 recovery allocation is untouched. `--frozen-sha256` is optional in
the CLI but supplied explicitly in the approved persistent launch below;
without it, the digest is bound once at queue creation, never hot-reloaded.

**Validated CPU-only:** 23 adapter tests passed, including unavailable/failure
receipt handling, live foreign-GPU waiting without allocation, run-entry
routing and failed-smoke prevention. Native real-GPU smoke remains unexecuted.
Current source identities:

| Artifact | SHA-256 |
|---|---|
| v3 `FROZEN.json` | `733480acc8e5eebde6a300fd8027302663f80dec52a58000811240dccd41bad5` |
| Adapter | `0924d45ee7cacff325d720108fbb92e4ec658b0e3e50cf55b73c94c130c55648` |
| Config (unchanged from v2) | `5cd5e90ebd5defbde6966961608db1d5235e555319d4070da70dd7ee92c76339` |
| Adapter tests | `050aa93af52cbbb70941f6b556ee7c5dce283c7c3bf910deee39343a3c6395bf` |

Exact coordinator launch on node4, **not executed by this preparation**:

```bash
sudo -n systemd-run --unit=coevolve-alfworld-whale-v3 \
  --property=Type=exec --property=User=1008 --property=Group=1008 \
  --property=WorkingDirectory=/mnt/3fs/xudong/coevolve \
  --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --property=RuntimeMaxSec=39h --property=Restart=no \
  --setenv=TMPDIR=/dev/shm --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv=PYTHONPATH=/mnt/3fs/xudong/coevolve/whale_local \
  --setenv=HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v3/FROZEN.json \
  --frozen-sha256 733480acc8e5eebde6a300fd8027302663f80dec52a58000811240dccd41bad5 \
  --execute-on dev-NDH200-026 \
  --wait-for-joint /mnt/3fs/xudong/coevolve/runs/alfworld_joint_h0w0_8h_v1 \
  --wait-label best
```

The default outputs are
`runs/alfworld_whale_baseline_v3/{queue,gpu-smoke-queued,execution}`.
The queue survives terminal disconnection under systemd; it does not install
boot-time restart or automatically resume interrupted GPU training.

## Historical v4: detached-source execution and CPU-only validation

**v3 is not suitable for a detached-source launch:** its freeze recorded an
absolute `config_path` in mutable main. v4 removes that field and instead
records `config_relative_path: configs/alfworld_whale_baseline.json`, resolved
against the **executed adapter's source root**. The input/game/model artifacts
remain shared through the explicitly allowed `assets`/`runs` links.

The manifest binds **57 local Python files in the static runtime import
closure**, not `git HEAD`. A documentation commit or a detached checkout with
identical code/config therefore does **not** invalidate the freeze. All those
57 files must actually be present in the snapshot. In particular, the closure
includes `whale_local/sao_bridge/joint_schedule.py`; include the parent's
already-tested schedule implementation in the commit/snapshot, not just the
four WHALE-owned files. Missing or modified dependency bytes fail closed.

The installed node-local venv was found to append
`/mnt/3fs/xudong/coevolve/whale_local` via startup `.pth` files, even with a
snapshot-specific `PYTHONPATH`. The owned adapter now removes foreign
`whale_local` fallback paths **before importing local dependencies**, rejects
already-loaded foreign local-code modules, and requires the working directory
to equal the executed source root. Launcher-created environment/vLLM module
children enter through the same `exec-local` bootstrap; torchrun's native
trainer enters through the adapter. The async controller is launched by the
environment service using its explicit neighboring snapshot script path.
No shared venv or shared training file was changed.

The standard `WHALE` and `third_party` links remain allowed upstream code
dependencies; this is not a claim that their entire vendored trees are copied.
Their directly pinned interfaces retain checksum validation. Never link the
snapshot's `whale_local` or `configs` back to mutable main. `runs` output
containment resolves the shared symlink, so queue/smoke/execution artifacts land
in the intended shared v4 directory rather than failing a lexical-path check.

### Exact no-GPU validation from the final snapshot

After committing and creating the parent's actual detached source snapshot,
**cd into that snapshot**, then run:

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/whale_local" HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v4/FROZEN.json \
  --frozen-sha256 f9abb6c3af103e20fbf43dbecfe127cc7f94b05505c868b08ef6603c5298bca9 \
  --validate-only
```

This command needs no assigned node, performs no resource allocation, queues
nothing and does not construct `Launcher`. It verifies original H0/W0 weight
bytes, runtime metadata, game assets, frozen source dependencies, relative
configuration and import origins. Inspect the output:
`status=validated_cpu_only`, `source_files_verified=57`,
`model_weights_verified=true`, `gpu_allocation_started=false`, and
`execution_source_root` must identify the actual snapshot. It fails rather
than falling back to main if a snapshot dependency/config is missing.

Preparation was verified end-to-end on CPU from an independent copied source
tree with exactly the proposed `WHALE`/`third_party`/`assets`/`runs` links. That
tree had no `.git` directory, demonstrating that HEAD is not consulted.
`SNAPSHOT_VALIDATION.json` preserves this evidence; the CPU-only copy fixture
was removed after validation. The first validation exposed the venv fallback,
and a later containment audit fixed shared-run paths. Those unactivated
preparation candidates were **archived**, not silently overwritten; only the
current digest below is approved for activation.

| Historical v4 artifact | SHA-256 |
|---|---|
| `FROZEN.json` | `f9abb6c3af103e20fbf43dbecfe127cc7f94b05505c868b08ef6603c5298bca9` |
| Adapter | `cb2d19088b4d9c8a9f85d9f78064d5f9710472c5ad0d457ecb740caf033398f3` |
| Config | `5cd5e90ebd5defbde6966961608db1d5235e555319d4070da70dd7ee92c76339` |
| Adapter tests | `83443fdcc8680a916f4728d4407a3828fb583b84bf52973b2e025049b22ff03c` |

**28 CPU tests / 16 subtests passed. No GPU smoke, queue or baseline was
started.** The v4 full-baseline predecessor for SAO is now exactly:

```
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v4/execution/COMPLETE.json
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v4/execution/RELEASED.json
```

The SAO agent was notified before activation. Receipt fields and the
smoke-first/MA-best prerequisite chain are unchanged.

### Persistent activation from that same validated snapshot

After the no-GPU check passes, set `SNAPSHOT` to its printed
`execution_source_root`. The parent may activate:

```bash
sudo -n systemd-run --unit=coevolve-alfworld-whale-v4 \
  --property=Type=exec --property=User=1008 --property=Group=1008 \
  --property="WorkingDirectory=$SNAPSHOT" \
  --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --property=RuntimeMaxSec=39h --property=Restart=no \
  --setenv=TMPDIR=/dev/shm --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv="PYTHONPATH=$SNAPSHOT/whale_local" --setenv=HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v4/FROZEN.json \
  --frozen-sha256 f9abb6c3af103e20fbf43dbecfe127cc7f94b05505c868b08ef6603c5298bca9 \
  --execute-on dev-NDH200-026 \
  --wait-for-joint /mnt/3fs/xudong/coevolve/runs/alfworld_joint_h0w0_8h_v1 \
  --wait-label best
```

The queue waits at most 24 hours, rejects MA failure receipts, accepts only
`completed`/`unavailable` plus clean release, re-audits live GPUs, then runs
smoke and starts the original-W0 baseline only if smoke passes. Snapshot
source/config hashes are fixed throughout. Documentation/HEAD changes in
main have no bearing on this detached execution.

## Historical v5: mandatory full-context, warm-optimizer GPU gate

**v4 remains historical and unactivated. v5 is a new immutable freeze, not an
in-place change to v4.** The former tiny digit-response smoke did not validate
32K training memory. It can no longer authorize the persistent queue's baseline.
The current GPU gate has no shorter-context fallback:

1. Generate two deterministic **synthetic** observation/reasoning/tool-call
   examples with the original W0 tokenizer: **28,672 prompt + 4,096 response
   tokens each**, with a `take_action: look` suffix and EOS. These do not contain
   ALFWorld tasks, trajectories, rewards or research results.
2. On all eight FSDP ranks, run the unchanged native WHALE response-token SFT
   loss on singleton microbatches. The smoke-only forward explicitly consumes
   **all 32,768 input positions**, requests 4,097 response-boundary logits, and
   discards the final unused logit. The native response mask has 4,096 ones.
   This avoids a subtle off-by-one: the research `_logprobs` optimization would
   forward 32,767 positions for such a call.
3. Execute **two real AdamW updates before the first export**, including
   full-context backward while optimizer state is already populated. Save
   BF16 HF export, full FP32 local parameter shards, Adam and RNG states.
4. Start a separate eight-rank process, load the export, restore exact FP32
   shards and Adam state at **step 2**, and verify full-tensor fingerprints on
   every rank. Execute a third full-context update to step 3 and export again.
5. Load the final export in all eight local vLLM replicas. Only then publish
   `SMOKE_COMPLETE.json` with `full_context_memory_validated=true`,
   `full_context_tokens=32768`, `response_mask_tokens=4096`,
   `optimizer_steps_before_restore=2`, `logical_optimizer_steps=3`, and the
   exact-restore/export-load flags. The controller checks actual per-forward
   lengths and optimizer-step evidence on every rank.

The queue rejects a short/old `passed` receipt without those proofs. OOM,
timeout, missing dependencies, shape changes or failed restore/reload stop the
chain; no automatic research baseline is started. Its W0 is always reloaded
from the original asset, never from the smoke's synthetic-trained export.
**Actual GPU full-context validation remains pending:** only CPU tests and
source preparation have run. This synthetic memory check is not a real
ALFWorld integration test or a performance measurement.

Smoke cost is **6 logical synthetic forward/backward calls, 3 optimizer steps,
24,576 logical response-token positions**, or **48 forwards and 196,608
response-token positions across eight replicated ranks**. Actual forward
context positions total 196,608 logical / 1,572,864 rank-replicated, before
activation-checkpoint recomputation. Add smoke elapsed time, two retained
optimizer exports, their restore/load work, and vLLM reload to preparation
costs; none are included in the 640 research rollout episodes. Research cost
still includes up to 768 extra development reward episodes, 200 final episodes,
30 proposer slots, inherited selection/reporting, and replicated-rank RSFT
work as documented above. No FLOP/GPU-hour measurement is claimed.

Allocation remains **node4 `dev-NDH200-026`, UID/GID `1008:1008`, all eight H200s**.
No node3 allocation is requested. Host RAM and several-TB shared-storage
requirements above still apply. Bounds remain **24h waiting, 2h smoke (50min
per native subprocess), 12h baseline, 39h outer systemd limit**. Exhaustion
means incomplete/failed, not a shortened successful experiment.

### Tested v5 identities and readiness

| Current v5 artifact | SHA-256 |
|---|---|
| `runs/alfworld_whale_baseline_v5/FROZEN.json` | `4e6f97db3de3e404282bb4474e8d355d151ab5ce39b4ec22c1b176af08c67093` |
| Adapter | `6447ca907710d42a21c429dc399b8eddfe00e3e4f1cca7eea034af91475e9ab2` |
| Config | `5cd5e90ebd5defbde6966961608db1d5235e555319d4070da70dd7ee92c76339` |
| Adapter tests | `96a3c16864b6ffaa54fdda1b847eb5a521a9cebd5e5fb9945cbf1b950b2fa134` |

**30 CPU tests and 20 subtests passed**, with three upstream torchao
deprecation warnings. Actual original-W0 tokenizer shapes were checked on CPU.
A separately copied source fixture without `.git`, using only the standard
`WHALE/third_party/assets/runs` links, passed full weight/input/source validation
and environment-child bootstrap `--help`. It verified 57 local source files
and pruned the mutable-main venv `.pth` fallback. The fixture was removed;
`CPU_VALIDATION.json` and `SNAPSHOT_VALIDATION.json` retain the evidence.

The manifest binds file bytes, **not original preparation HEAD**. Documentation
commits do not invalidate it. The parent must include all 57 dependencies in
the detached snapshot, with copied `whale_local` and `configs`, not main-tree
links. External upstream interfaces remain shared and checksum-pinned as
described above. The parent must validate its actual final snapshot:

```bash
# First cd into the parent's detached source snapshot.
CUDA_VISIBLE_DEVICES='' TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/whale_local" HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/FROZEN.json \
  --frozen-sha256 4e6f97db3de3e404282bb4474e8d355d151ab5ce39b4ec22c1b176af08c67093 \
  --validate-only
```

This performs no queueing or GPU allocation. Require
`status=validated_cpu_only`, `source_files_verified=57`,
`model_weights_verified=true`, `gpu_allocation_started=false`, and the correct
snapshot `execution_source_root`.

### Exact persistent queue command — coordinator only, not launched

Set `SNAPSHOT` to the validated absolute `execution_source_root` above.
The following command is for the parent's future activation on node4:

```bash
sudo -n systemd-run --unit=coevolve-alfworld-whale-v5 \
  --property=Type=exec --property=User=1008 --property=Group=1008 \
  --property="WorkingDirectory=$SNAPSHOT" \
  --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --property=RuntimeMaxSec=39h --property=Restart=no \
  --setenv=TMPDIR=/dev/shm --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv="PYTHONPATH=$SNAPSHOT/whale_local" --setenv=HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/FROZEN.json \
  --frozen-sha256 4e6f97db3de3e404282bb4474e8d355d151ab5ce39b4ec22c1b176af08c67093 \
  --execute-on dev-NDH200-026 \
  --wait-for-joint /mnt/3fs/xudong/coevolve/runs/alfworld_joint_h0w0_8h_v1 \
  --wait-label best
```

The persistent chain is: MA `best_COMPLETE` + `best_RELEASED` → reject failed
receipts → accept only `completed`/`unavailable` → live GPU/filler audit →
unchanged freeze → full32K warm-optimizer smoke → original-W0 baseline only
on success. There are no hot configuration changes or automatic workload
restarts. Waiting is recorded separately from GPU training.

Standalone full-context smoke, **only after an explicit idle-node assignment**,
from that same snapshot:

```bash
TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/whale_local" \
HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/FROZEN.json \
  --frozen-sha256 4e6f97db3de3e404282bb4474e8d355d151ab5ce39b4ec22c1b176af08c67093 \
  --execute-on dev-NDH200-026 --gpu-smoke \
  --output /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/gpu-smoke-standalone
```

That output must not already exist and is separate from both queue smoke and
research execution. Prefer the persistent smoke-first chain; direct `run`
without wait flags remains a manually allocated coordinator escape hatch,
not a way to bypass a failed automated gate.

### Final receipt contract for the following SAO queue

The v5 full-baseline predecessor is exactly:

```
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/execution/COMPLETE.json
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v5/execution/RELEASED.json
```

Full `COMPLETE.json` has `status="completed"`,
`label="local WHALE-style ALFWorld adaptation"`, `training_episodes=640`,
finite `all200`/`strict197`, and `published_alfworld_baseline=false`.
`RELEASED.json` includes `foreign_gpu_pids`, `remaining_gpu_pids`,
`verified_filler_pids`, `vanished_gpu_pids`, `filler_kind`, and
`filler_unit_active`, from the common live audit after process cleanup.
The queue's own `queue/COMPLETE.json` is not this predecessor receipt.
Smoke writes only `gpu-smoke-queued/SMOKE_COMPLETE.json` with `status="passed"`
and `purpose="synthetic_gpu_smoke"`, never a full-baseline `execution/COMPLETE`.
No such GPU-success receipts exist from this CPU preparation.

## Current v6: current-MA Solver population alignment

**v5 and earlier freezes used a different Solver population and must not be
activated for this comparison.** The historical 2,296-game train-only pool
was appropriate to historical W1, but current MA uses the legacy `test_set`
catalogue's training task IDs 0–239. v6 corrects that confound while retaining
the full-context smoke, native RSFT objective and iterative local MH behavior.
It does not attempt to reproduce a historical W1 curriculum.

The new Solver sequence is the **first 640 records, in their original order**,
of `runs/alfworld_joint_h0w0_8h_v1/data.json`. The full reference has 1,024
records; all 640 selected record IDs are retained. Their 240 unique tasks occur
two or three times: **160 tasks occur three times, 80 twice**. Nothing is
reshuffled or deduplicated at episode level. The first task IDs are
`8,128,22,26,47,202,221,99,129,59,65,107,76,219,205,152`.
Worker task whitelists are deduplicated without removing repeated rollout
records. Native updates still use fresh success trajectories under the current
W/harness, not historical MA trajectories.

The manifest pins the exact MA data and original-W0 Solver specification.
Preparation checks record ID, joint version/slot, task split, legacy task
range, and recorded game path against the actual environment catalogue.
Historical `messages` and `critic_evidence` fields are **not copied into
Solver inputs**. The episode projection contains only ID, task ID/split, and
source record index.

### Exclusion and budget interpretation

- **No core200 task identity enters Solver training or new MH development.**
  Every training task is in legacy 0–239.
- As in current MA, three all200 games have goal groups related to legacy
  training: **291, 414, 467**. They remain in all200 for protocol consistency;
  **strict197 has zero training-goal overlap**. Do not claim all200 is entirely
  goal-disjoint or a newly sealed test.
- MH development remains **24 separate train-only games**, goal-disjoint
  from legacy training, all500 and reserved validation games. Reserving legacy
  games for development would prevent matching the requested full MA prefix.
  These development games are not Solver SFT inputs; their up-to-768 reward
  episodes are an explicitly additional MH search cost. This is a remaining
  selection-population difference, not equal MA/H reward data.
- **640 WHALE Solver episodes versus MA's 512 Solver +512 editor records at
  its total1,024 checkpoint** is an allocation difference, not budget equality.
  MA counts critic warmup; WHALE trains only accepted successes. WHALE has
  32-episode RSFT collection batches versus MA's 8 records/role, a different
  frozen seed schedule, and different optimizer steps and objectives.
  Task-record order is aligned; stochastic trajectories/optimizer work are not.
- Full cost still includes 30 H0 proposal slots, up to768 development episodes,
  200 final episodes, inherited initialization/reporting, smoke, eightfold
  replicated-rank RSFT work, and checkpoint/reload overhead. No FLOP match or
  source-native full-MH-session parity is claimed.

### Final v6 hashes and CPU evidence

| Artifact | SHA-256 |
|---|---|
| `runs/alfworld_whale_baseline_v6/FROZEN.json` | `c848628a8a03c4e340a106ae2e559cf95101ca028bd9dd37e2d61032f85134c7` |
| Adapter | `05b22dcba3e4724d250c9853726a058ceac0a35c6eb5b25298ceee4e2c19f5f8` |
| Config | `64c7089ee1a0562a5fe98fdcf2527bb4fd9b7df75b49eabdd0e8ebbdbe355fce` |
| CPU tests | `82714ee48c3bf0481aedb4587153de77f53fcef6f85900d50e806669568a3133` |
| Current MA `data.json` | `6e8a952f388e42d167e4388f638d5b5e94043f94bfaa158c4fbf420b11c6c12a` |
| Current MA `solver_spec.json` | `e69e04b45c9af41c20d867dc088e6ee63ebcc714b020a26fbfb2d313b43f62f9` |
| Ordered 640-record episode projection, canonical digest | `916dfd2e69566ab3819afe8d42ded9dedd298490809428fdac777c5373bf9bb9` |

**32 CPU tests +24 subtests passed**, including actual MA-prefix identity/order,
repeat retention, exclusion checks, rejection of reordered/misidentified
records, historical-prompt stripping, full-context smoke proofs and queue
failure gates. A fresh no-`.git` copied-source fixture with the standard four
links passed all57 source hashes, original model hashes and full input checks.
Mutable-main `.pth` fallback was pruned and the environment-child bootstrap
`--help` passed. The fixture was removed; v6 `CPU_VALIDATION.json` and
`SNAPSHOT_VALIDATION.json` persist. **No GPU/queue was launched.**

### Exact v6 commands and successor prerequisite

After the parent commits all57 recorded source dependencies and creates the
production snapshot, cd into it. Validate without GPU allocation or queueing:

```bash
CUDA_VISIBLE_DEVICES='' TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$PWD/whale_local" HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/FROZEN.json \
  --frozen-sha256 c848628a8a03c4e340a106ae2e559cf95101ca028bd9dd37e2d61032f85134c7 \
  --validate-only
```

Set `SNAPSHOT` to the validated `execution_source_root`. Parent-only future
activation on node4, **not executed by this preparation**:

```bash
sudo -n systemd-run --unit=coevolve-alfworld-whale-v6 \
  --property=Type=exec --property=User=1008 --property=Group=1008 \
  --property="WorkingDirectory=$SNAPSHOT" \
  --property=KillMode=control-group --property=TimeoutStopSec=120 \
  --property=RuntimeMaxSec=39h --property=Restart=no \
  --setenv=TMPDIR=/dev/shm --setenv=PYTHONDONTWRITEBYTECODE=1 \
  --setenv="PYTHONPATH=$SNAPSHOT/whale_local" --setenv=HF_HUB_OFFLINE=1 \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.alfworld_whale_baseline run \
  --frozen /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/FROZEN.json \
  --frozen-sha256 c848628a8a03c4e340a106ae2e559cf95101ca028bd9dd37e2d61032f85134c7 \
  --execute-on dev-NDH200-026 \
  --wait-for-joint /mnt/3fs/xudong/coevolve/runs/alfworld_joint_h0w0_8h_v1 \
  --wait-label best
```

All prior operational bounds remain: one node4 allocation/all8 H200s,
24h wait, 2h full32768/4096 warm-Adam smoke, 12h research budget, 39h outer
unit. The queue requires MA best COMPLETE+RELEASED, clean live GPU/filler audit,
unchanged v6 freeze and a passed full-context smoke; it never hot-updates
configuration or falls back to short sequences.

Standalone assigned-node smoke uses the same v6 `run --frozen ...` and digest,
with `--gpu-smoke --output
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/gpu-smoke-standalone`
instead of the two wait options. That directory must be fresh.

**Parent's new SAO freeze must explicitly use**
`--h-predecessor /mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/execution`.
The full-success and release contracts remain exactly those documented for v5,
but the final paths are now:

```
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/execution/COMPLETE.json
/mnt/3fs/xudong/coevolve/runs/alfworld_whale_baseline_v6/execution/RELEASED.json
```

These are not queue or smoke success receipts. Actual GPU full-context
validation and all ALFWorld performance results remain pending.
