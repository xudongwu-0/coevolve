# Fixed-sample SAO / MA-SAO research protocol

## Current ALFWorld continuation (2026-10-01)

The two independent 9B learning gates have passed; see
[project state](PROJECT_STATE.md). The next **single-seed exploratory continuation**
is declared in [the strong paired protocol](../examples/alfworld_strong_sao_ma_v1.json).
It is not a fresh, independently initialized reproduction of the older formal math
protocol below.

- Both arms inherit the same validated Engineer E15 prefix and official W0. E15
  contains 11 Actor updates, but the common history actually executed 32 native
  steps/27 Actor updates and searched E0/E15/E31. Charge that whole history,
  including discarded work, to both logical arms. Choose one common global H1
  from the eight frozen E15 proposals on selection data only.
- Strong sequential: common H1, then **H2-W1-W2**. MA: common H1, then
  **W1-H2-W2**. H2 receives 32 native steps (5 critic-only, 27 Actor); each W
  block receives the demonstrated 80-step recipe (5 critic-only, 75 Actor),
  not the historical two-update recipe. Do not reject/restart W after step39:
  the successful v4 run only improved at step79.
- Both arms use the same per-role learning settings, examples/offsets, proposal
  counts and phase-boundary Adam resets. Carry same-role critic weights, not an
  Engineer critic into the Solver. Actual generation, failed attempts, GPU time,
  refresh and selection costs must also be reported, not equated to update counts.
- MA uses a pre-action centralized critic and refreshes train-only failure
  packets after the Solver changes. Calibration is **train-only**, never report
  labels. Fresh-only terminal trajectories are mandatory. Replay, interaction
  bonuses and Solver recycling are **off**: the current terminal-mode validator
  explicitly rejects legacy replay, and official target receipts do not contain
  authentic Solver behavior probabilities. Do not bypass that guard to claim a
  richer algorithm.
- Selection/report use the goal-isolated 24/70 cohorts, mean@2, total 4096
  completion tokens/episode, temperature1 and matched per-task/sample seeds.
  These development cohorts have been used before; do not call them untouched
  or confirmatory. Freeze both final arms before reading this comparison's report.
  A large MA win requires at least +5pp and a positive paired 95% lower bound.
  Order and centralized context change together; a win alone would not identify
  which component is essential. A budget stop does not establish convergence.

`python -m sao_bridge.alfworld_solver_eval --config CONFIG --harness-matrix`
evaluates a JSON `harnesses` label-to-patch-path mapping using one Solver serving
pool. The usual no-patch entry point remains strict; matrix mode must be explicit.
The no-patch control is inserted automatically and its label cannot be overridden.
Canonical duplicates share a rollout file but retain every logical label, including
invalid proposals deliberately mapped to no-patch by the proposal selector.
Patches are snapshotted before environment startup. Bounded parallel harnesses
share the request limit; `environment_concurrency` is per harness worker, so the
declared configuration allows at most 4x32 active episodes. Partial completed
reports survive errors; peer evaluations are cancelled before the shared client
closes, and infrastructure errors never become zero-reward fallback reports.
The environment service also requires explicit matrix mode to repeat a cohort
across independently named workers. Every matrix worker must have identical
task IDs/split/catalogue; ordinary training assignments still must partition IDs.
When preparing detached source worktrees, link **all** paths in `.gitmodules`,
including root-level `WHALE`, as well as assets, then CPU-import the runner before
starting a GPU unit. Looking only inside `third_party` misses the math reward
dependency imported by the shared bridge.

Matrix mode opts into the bridge's nonblocking local controller. The pinned
minimal controller calls synchronous `requests` inside async FastAPI handlers:
a large start burst prevents heartbeats from being processed and trips the
60-second alive-worker deadline. Do not hide those404s as model failures or
simply increase that deadline. The bridge keeps the official worker/session
protocol and error responses, but awaits asynchronous worker forwarding.
The original single-harness controller remains the default unless the new flag
is explicit; running frozen experiments are not modified in place.

Concurrent vLLM replicas now receive distinct internal rendezvous port bases,
separate from their HTTP ports, rather than racing in the shared ephemeral pool.
MA W2 selection encountered two documented `EADDRINUSE` startup failures before
any evaluation started. An explicitly authorized `STARTUP_RECOVERY.json` may
permit exactly one additional matrix attempt after both original attempts:
it binds the unchanged request and original failure/log hashes, requires GPU
release and rejects any earlier completed/progress reports. This is an extra
**infrastructure attempt**, not an unchanged retry budget; its time and work
remain charged. It does not extend native training, change model/data/seeds,
relax the selection gate, or treat startup failure as a zero-reward sample.
The recovered MA matrix then exposed a separate lifecycle fault: the blocking
30-second worker cancellation timed out, its HTTP502 was not checked by the
episode adapter, and a replacement episode hit the still-full worker's406.
The controller now requests official `cancel_with_notice` and holds the client's
episode slot until the worker confirms completion, with a bounded300-second
deadline. Cleanup HTTP failures propagate instead of returning a zero-score
evaluation; an earlier generation exception remains the primary error.
One further explicit `CANCELLATION_RECOVERY.json` can bind this failed attempt
and its controller log for a single retry. The original attempts and any partial
generation remain charged; no model, task, seed or selection threshold changes.

The matrix runner and train-only context support have CPU regression coverage.
The full paired continuation is not yet a completed experiment; runtime receipts
and subsequent entries in [the experiment log](EXPERIMENT_LOG.md) are authoritative.

### Persistent native phase execution

`python -m sao_bridge.alfworld_strong_phase --protocol FROZEN_JSON --arm sequential
--phase h2 --root PHASE_ROOT --matrix-root COMMON_MATRIX_ROOT` waits for the
common matrix's completion and GPU-release receipts, independently checks all
task/sample seeds and scores, and publishes the shared initial state.
The initial MA W1 phase waits for that state. Later phases require an explicit
selected `--state`; they must not silently restart from the common prefix.

The phase runner freezes its data/config/state, preserves failed attempt
directories, takes an exclusive controller lease and permits at most the declared
two native attempts. Run it under systemd `Restart=on-failure` with
`RestartPreventExitStatus=78`; input-validation failures and exhausted attempts
must not restart forever. Successful native phases are audited and reused, never
silently retrained. `PHASE_COMPLETE.json` includes contiguous native steps,
warmup isolation, exact Actor batch size, role-appropriate terminal targets and
the count of nonzero gradients. The native phase entry point is not, by itself,
a completed arm: proposal selection, counterpart refresh and later phases still
need their declared receipts and final report gate.

`python -m sao_bridge.alfworld_strong_arm --protocol FROZEN_JSON --arm sequential`
(or `--arm ma`) is the persistent **whole-arm** controller. It adopts the already
launched first native phase without duplicating it, then performs the prescribed
training/selection order. Run it under the same bounded systemd restart policy.
It stops at `ARM_FROZEN.json`; it never opens report questions by itself.

Post-phase selection is implemented in `alfworld_strong_select`: H2 uses eight
matched prompts/seeds for the incumbent/E15/E31 continuation checkpoints,
retains invalid/no-patch and duplicate slots, selects the Engineer by mean
proposal performance, then chooses a deployed harness from that model,
the incumbent harness and no-patch. W checkpoints are only considered after the
full80-step phase. Shared/current control reports are bound to Solver path,
harness hash, question IDs, seeds and decode settings. Runtime supplementation
does not rewrite already-complete historical model metadata.
W promotion uses the frozen two-percentage-point accuracy floor. The selector
also records a Bonferroni-adjusted paired LCB for diagnosis; that LCB is not an
additional qualification condition. Do not confuse this selection rule with
the final MA advantage gate's positive paired95% lower-bound requirement.

After MA W1 actually changes W, `alfworld_strong_refresh` collects24 official
greedy baseline cohorts of10 **train-only** tasks. It freezes the raw records,
binds current Solver weight hashes and rebuilds compact prompts with the original
builder settings. All-pass cohorts remain in calibration without fabricated
failure prompts; if the entire training cohort has no failures, the controller
stops for a protocol decision rather than fabricating an equal-budget H2.
These per-turn-budget packets are not replaced by the earlier native
total-token-budget diagnostic. No Solver behavior log-probabilities are invented.

### Frozen report opening

`python -m sao_bridge.alfworld_strong_report --protocol FROZEN_JSON --arm ARM --wait`
waits for **both** immutable arm freezes before allocating report GPUs or
reading report outcomes. It verifies the frozen selected model/harness and
evaluates no-patch, common H1 and both final harnesses under the arm's final
Solver. The MA-side reporter also supplies the common W0 cross-harness row.
Identical patches retain all aliases without duplicate inference.

After both reporter units finish and release GPUs,
`python -m sao_bridge.alfworld_strong_report --protocol FROZEN_JSON --compare`
rechecks freeze/report hashes, exact task/sample seeds and binary trajectories,
then writes `PAIRED_REPORT.json`. Add `--wait` for a separate persistent CPU
finalizer on node4; it cannot finalize before both release receipts exist.
The +5pp floor is tested at its exact boundary;
a positive paired 95% lower bound is separately required. It also reports the
cross-harness final-policy interaction, without claiming a causal component
ablation. All outputs explicitly retain the reused-development/single-seed
limitation. Missing or inconsistent freezes cannot be bypassed by a report flag.
If both arms select identical Solver weights and harness, the method-improvement
gate cannot pass: any separately measured score difference is evaluation
variation, not a policy improvement. Raw paired scores are still preserved.

Native terminal-target audits allow absolute FP32 roundoff of at most `1e-6`
at the role reward boundaries, retaining the raw extrema and tolerance in each
receipt. Sequential W1 completed all 80 steps but the original exact-bound audit
stopped its controller on a single `-3.725290298461914e-9` residual at step26.
This is an audit-only correction, not target clipping or a reward/optimizer
change. Larger violations and non-finite/reversed ranges still fail. Preserve
the original failure and source; reuse the completed training rather than
spending another training attempt.
For selections frozen before these two audit fields existed, verify every old
binding exactly and publish the additive fields in `target_range_audit.json`.
Do not rewrite the old identity or regenerate its completed proposal matrix.
Completed native phase receipts are likewise reused byte-for-byte after a fresh
validation; extending an audit schema must not change the phase hash bound by
an already completed selection.

For a provenance-recorded controller-only upgrade while the old controller is
still running, queue the new `alfworld_strong_arm` with
`--after-unit coevolve-strong-ma-v1-arm.service`. It stays CPU-only until the
previous systemd unit is inactive/failed, including through its automatic
restarts. If the old arm already froze, the new controller exits without
training/evaluation. Otherwise it resumes under the same immutable inputs and
attempt limits. It does not interrupt the old job or bypass substantive failed
invariants. Use a separate systemd unit and preserve both launch receipts.

After both arms freeze and an arm's report completes/releases, run the cost audit
on that arm's owning node:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=whale_local python -B \
  -m sao_bridge.alfworld_strong_cost \
  --protocol runs/alfworld_strong_sao_ma_v1/frozen_protocol.json --arm sequential
```

For the current run, CPU-only systemd jobs are already queued:
`coevolve-strong-sao-v1-sequential-cost-r2` on node4 and
`coevolve-strong-ma-v1-cost-r2` on node3, using frozena4eef56.
Each waits for its existing report unit to stop, then invokes this CLI.
`COST_AUDIT_LAUNCH.json` records the command/source; a stopped or failed reporter
is not assumed successful, because the audit still validates actual completion,
freeze and release receipts before publishing. No GPU workload is allocated by
the waiting jobs, and they never stop the reporters.

Use `--arm ma` on node3. `COST_AUDIT.json` separates declared/optimizer-consumed
episodes from all saved fresh-generation receipts, including prefetch duplicates.
Never choose an arbitrary duplicate receipt as the consumed behavior. Count
physical matrix reports once rather than their logical aliases, retaining failed
attempts' completed reports, proposal responses and official reward retry counts.
Canceled work without a saved receipt remains an explicit lower-bound gap.
Eight-GPU allocation time starts at the original first-phase launch and ends at
final report release, including recovery and peer-arm waiting. Per-process
telemetry separates experiment/filler; nested matrix telemetry is not counted
again when its selection parent already covers it. Original shared-prefix
training/search and common H1 selection are additional equal charges; the separate
Solver-v5 and train-probe explorations remain separately reported R&D.

### E15/W79 eight-hour successor (2026-10-02)

The next user-requested experiment is a **different frozen run**, not a restart
of the W0 comparison above. Its [persistent controller](../whale_local/sao_bridge/alfworld_alternating_campaign.py)
starts both arms at validated E15/critic15 and v4-W79/critic79. Both orders are
H2 then W2; only the MA critic receives pre-action centralized context. Common
W79 train-only evidence is freshly collected with the official greedy/per-turn
protocol, then E15 proposes eight harnesses for this actual partner.

H2 uses32 native/27 Actor updates, W2 uses80/75, with matched role settings and
selection budgets. Native and scoring attempts remain bounded at2. The fixed
eight-hour wall deadline does not restart with a coordinator. Independent
systemd timers stop only the explicitly named campaign services, reserve the
last hour for reporting, and never stop the adaptive GPU filler. A timeout
freezes the last validated state; an incomplete phase budget disables a claim
of MA superiority. The `final_systems` report mode retains the necessary initial,
H2-intermediate and W2-final systems, but omits cross-swap interaction ablations.

The target-specific released `agent-sft-harness-r1` has also been downloaded and
hash-verified. Its [matched-baseline evaluator](../whale_local/sao_bridge/alfworld_matched_engineer_baseline.py)
reuses frozen SAO proposals and re-evaluates them beside eight matched official
proposals, preserving invalid slots. That GPU experiment is **deferred** while
the eight-hour alternating campaign has priority; no new official-baseline score
has been claimed.

## Objective and control

The primary endpoint is final held-out task accuracy under the same declared
**generated-sample budget**, not throughput alone. GPU time and optimizer-consumed
sequences are reported separately. Tune the independent SAO control on development
data, then freeze its settings, initialization, paired seed set and evaluation protocol.
Additional MA-SAO tuning is allowed and accounted for separately. Every formal
MA-SAO run starts from the same initial actor assets, not a tuning checkpoint.

`sao_bridge.budget` stores an immutable baseline contract, one exclusive formal
claim per predeclared seed, and a GPU-time ledger. The launcher must charge all reserved identical
H200 GPUs during H/W training, editor generation, reward evaluation, weight
transfer, checkpointing and waiting. Final held-out evaluation is outside that
training ledger but uses the identical fixed protocol for both methods.
Extra tuning compute is never presented as free or as equal total research cost.

As of 2026-09-29, only node3 and node4 are authorized for new experiments
(the older runs below used node0/node4). Node0's prior utilization service was
restored after its jobs completed. Both `sao_bridge.resources` and the launcher
reject new compute allocation on other hosts. `sao_bridge.resources`
checks command, owner and systemd cgroup and keeps the separately deployed
adaptive utilization supervisor active. Filler activity is excluded from
experiment utilization. It never disables/masks that service or kills an
unrecognized GPU job. A new run requires a fresh audit.

## Benchmark

The selected benchmark is the original paper's **tool-based mathematical
reasoning** setup: DAPO-Math-17k for training and AIME 2024/2025 for held-out
evaluation, with Python execution and WHALE's strict boxed-answer verifier.
This replaces the earlier tiny, text-only MATH number-theory smoke test.

The choice is based on WHALE Sections 6.1-6.2, not on favorable results from our
models. The paper identifies math as weight-dominant and documents both
truncation/format bottlenecks and harness search becoming more effective after
weight updates. Its stagewise baseline also exhibits conditional
over-optimization: a harness can improve its search-set score without improving
test performance. These are relevant failure modes for counterpart-aware credit
assignment. SearchQA is a useful future harness-dominant replication, but is
not silently substituted for this benchmark.

The planned model is Qwen3-4B-Instruct-2507 (non-thinking). Results therefore
compare our matched SAO control and method, **not an exact reproduction of the
paper's model, full executable-harness space or reported scores**.
Harness edits are restricted to validated literal settings and bounded recovery
controls; generated module code is not executed. Python *tool* code runs in
isolated namespaces as an unprivileged UID, without network, GPUs or project
mounts.

## Substantive method under investigation

1. **Experimental centralized value input (inactive in the leading method).** The critic receives a
   pre-action context containing role, phase, current harness, frozen counterpart
   version and development calibration. This is prepended only for critic
   forwards/updates; prompt/observation masks and actor token alignment are
   preserved. The actors never see privileged critic-only inputs.
2. **Experimental cross-role value transfer (inactive in the leading method).** A shared value checkpoint is transferred
   across H1 -> W -> H2. Role-conditioned inputs distinguish the different
   decision problems. This alone does not establish delayed credit assignment.
3. **Counterpart-refresh replay.** After W changes, a declared fraction of H2
   samples reuses authentic H1 editor actions and their original token
   probabilities, but recomputes rewards against the new frozen W. Selection
   into the replay buffer is independent of reward. The native SAO decoupled
   objective and ratio rejection address conditional action-policy mismatch,
   not every form of state-distribution shift; behavior probabilities must never
   be replaced with current-policy probabilities.
4. **Version accounting.** Actor-version lag and counterpart-version changes
   are distinct. Both must be recorded. A smaller asynchronous queue is an
   ablation/implementation setting, not sufficient by itself to call the
   algorithm multi-agent.
5. **Cross-role experience recycling (leading method).** Every editor action is
   scored by fresh candidate/reference solver trajectories. These are exact samples
   of the next W phase's initial policy. A disjoint, content-hash-ordered pool first
   calibrates the G=1 value model without actor updates, then unique remaining
   episodes are mixed once into W actor/critic updates. In the planned v6
   protocol no trajectory is repeated across these two pools;
   native behavior log-probabilities, versions and DIS correction are preserved.
   This reuses joint-system experience without increasing generated samples, but
   increases optimizer work and GPU time. Earlier v5 results reused a small
   remaining actor pool repeatedly and are not evidence for the v6 protocol.

6. **Paired cross-role interaction credit (experimental, beta currently 0).** Replay reevaluates the same H1
   candidate and its original incumbent under the new executor. Define
   `d_old = J(W0, candidate) - J(W0, reference)` and
   `d_new = J(W1, candidate) - J(W1, reference)`. The replay target reward is
   `d_new + beta * (d_new - d_old)`, with beta selected on development data.
   The second term is a measured interaction contrast, not a differentiated
   optimizer update. Both terms use the same fixed reward questions/seeds and
   no held-out answers. Invalid actions remain -1, without an interaction bonus.
   The extra reevaluation is charged to the trial's training-compute allowance.

Historical replay versions use a negative phase-relative offset. This does not
change the authentic behavior log probabilities or generated-token loss mask.
The native recomputed proximal-policy/DIS objective remains in use, but native
version metrics exclude negative versions. Historical actor/partner epochs must
therefore be reported separately from native within-phase staleness, not merged
into an apparently zero-lag statistic. Approximate version-based proximal
probabilities are not enabled.

The critic and replay adapters do not reimplement PPO, GAE or the value loss.
They call the pinned unofficial SAO/AReaL training loop. Stage-boundary HF
handoffs currently reset optimizer moments for both control and treatment;
cross-phase critic transfer preserves weights, not Adam state.

The centralized-critic variants are experimental CTDE-style adaptations for
heterogeneous, temporally separated roles. The leading experience-recycling
variant does not use that critic and should not be described as demonstrated
centralized value learning. Neither is a verbatim MAPPO implementation or a
gradient through a solver weight update, and neither guarantees improvement.
Separate warm-up, actor-recycling and reward-refresh ablations are needed before
attributing a gain to one mechanism.

## References

- [WHALE](https://arxiv.org/abs/2609.00196), especially Sections 6.1-6.2.
- [The Surprising Effectiveness of PPO in Cooperative, Multi-Agent
  Games](https://arxiv.org/abs/2103.01955).
- [Official MAPPO code](../third_party/mappo-reference), revision
  `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`: separate actor observations and
  centralized critic observations in `R_MAPPOPolicy`; distinct actor/value
  optimizers and clipped objectives in `R_MAPPO`.
- [Pinned SAO integration and smoke results](SAO_BASELINE.md).

## Reporting and extension gate

Record H0/W0, H1/W0, H1/W1 and H2/W1 to separate harness gain, weight gain and
the effect of re-adapting the harness. Report accuracy, output truncation,
valid-action/boxed-answer rates, tool use/errors, token consumption, policy lag
and actual GPU-time. Report physical sampled episodes separately from episodes
consumed by updates; asynchronous prefetch is not free.

Choose configurations using development results only. Open the held-out AIME
results after models/stages are frozen. The formal protocol predeclares paired training seeds; evaluation uncertainty is
not training-seed uncertainty. Two seeds remain exploratory, so confirmatory claims
require more seeds or sealed-test evidence. Extend to H-W-H-W-H only after establishing an advantage under the
shorter matched protocol, and compare the extension under its own matched
budget. More iterations and more compute alone are not evidence for MA-SAO.

The large experiment is **not yet verified**. A completion claim requires
actual GPU runs, persisted parameter changes, final checkpoint reloads and the
frozen comparison outputs. CPU adapter tests are not evidence of a task-score
gain.

## Reproducible development entry

The shared model is pinned to Qwen revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Public asset revisions and hashes live
in each asset directory's `provenance.json`. The DAPO source parquet physically
contains 1,791,700 rows, including repeated questions; these are not distinct
training tasks. After deterministic question normalization/deduplication and
excluding 15 conflicting-label question groups, this protocol has 16,391 solver
training questions, 256 reward questions, 256 development questions and 60
AIME evaluation questions. The DAPO `Answer:` wrapper is removed before both
exact-normalized overlap checks and the fixed boxed-answer task protocol.
This detects exact normalized duplicates, not semantic or pretraining leakage.

```bash
export PYTHONPATH="$PWD/whale_local"
.venv-ppo/bin/python -B -m sao_bridge.experiment \
  --spec examples/sao_calibration.json
```

The calibration specification is deliberately smaller than formal training and
does not open the AIME test scores. It is not a frozen formal baseline.
`examples/sao_calibration_central.json` is the critic-only ablation.
Each run needs a fresh output path. Progress, failure reasons, exact per-stage
specifications and process ownership are persisted. `FROZEN.json` precedes any
optional held-out evaluation; `COMPLETE.json` is absent on failed runs.

Runtime adaptations: independent local vLLM replicas avoid its fixed 30-second
DP-coordinator startup deadline on this shared filesystem. Compiler caches stay
in shared storage; Unix IPC sockets use node-local `/dev/shm` because 3FS does
not implement the needed Unix-socket operation. This is transient IPC only, not
the only copy of model/data/code. Node0's root disk is full and its package
manager has unrelated dependency conflicts; neither was globally repaired or
cleaned. The verified shared bubblewrap binary avoids changing those packages.

Development calibration found that Qwen3 emits `py` fenced tool arguments, not
only `python` fences. The tool now accepts `py`, `python`, and unlabelled fences
and retains upstream last-expression printing. The isolated interpreter mounts
only SymPy and mpmath package directories read-only, not the training virtualenv,
project, models or credentials. Real sandbox checks verify symbolic computation
and the absence of the shared SSH directory, with execution as UID 1008.

The central-critic `_03` diagnostic reached two real editor actor updates/four
critic updates, saved/reloaded its editor and started solver training with a
transferred critic. It was stopped when development trajectories revealed
26/26 tool calls failing on the unsupported `py` fence. Its 1/16 to 2/16
development selection change is **not valid evidence of method superiority**.
Both arms are recalibrated after the tool correction. Node-local loopback fixes
the two-rank Gloo initialization failure without changing the training loss.
Native XCCL weight transfer is being calibrated against the previously working
but expensive per-step shared-disk weight transfer. No formal baseline is frozen
on these interrupted diagnostic trials.

### Explicit single-node network override

Node4's native XCCL TCP rendezvous was reachable at `127.0.0.1` but refused
connections via its own `10.1.0.35` address at the same listening port. Rather
than change machine routing/firewall rules or the SAO update, a small separately
tracked AReaL runtime patch adds the optional `AREAL_HOST_IP` override:

```bash
git -C third_party/sao/vendor/AReaL apply \
  "$PWD/whale_local/sao_bridge/areal-local-network.patch"
```

Apply this once after upstream bootstrap on a fresh checkout. The launcher sets
it to `127.0.0.1` only for these node-local jobs; do not use loopback for a
multi-node training process group. With the variable unset, upstream automatic
address detection is unchanged. `verify_upstreams()` verifies both the original
SAO patch and this runtime patch, and records both digests. Actor/value losses
remain upstream implementations. Native two-GPU custom NCCL all-reduce
preflights passed on both node0 and node4 after this change.

Formal trials now validate the actual initial model/tokenizer file digests,
data-manifest digest, training seed and locked evaluation settings before a
baseline claim is made. MA learning-rate changes remain permitted; silently
changing initial weights, evaluation length or questions does not.
Formal MA runs must name the completed control with the same seed through
`reference_run`. The contract locks generation-defining settings. Replay can
reduce actual fresh generations below the same cap; category counts and actual
totals must still be reported rather than claimed identical. GPU time is a
reported efficiency outcome, not the matched quantity. MA may consume additional
optimizer steps on already-generated partner trajectories, which is reported as
additional compute/information reuse rather than hidden as equal compute.
Identical proposed harness sources are evaluated only once per selection stage
to avoid awarding duplicate candidates extra chances from numerical sampling
variation. Reported uncertainty resamples question-level paired score
differences, not individual samples as if they were independent training seeds.

### Checkpoint and process-lifecycle corrections

The pinned upstream trainer calls the same saver's frequency counter separately
for actor and critic. At an even save interval this can save only the critic.
The adapter caches the native frequency decision by optimizer-step identity, so
both roles are saved together, and explicitly includes the final training step.
It does not alter the optimizer or value objective. Three regression tests cover
paired saves, restored counters and an off-frequency final step. A training
receipt now requires every declared optimizer metric and final-step actor and
critic checkpoints, rather than accepting an earlier snapshot.

During native worker shutdown, a child can briefly be reparented before it
releases its GPU context. The resource watcher remembers owned PID/create-time
pairs, rather than treating a disappearing parent link as a new external job.
A reused PID or unknown live GPU process still causes our job to yield.

Development diagnostics can be generated without opening test scores:

```bash
PYTHONPATH="$PWD/whale_local" .venv-ppo/bin/python -B -m sao_bridge.analysis \
  --run runs/EXPERIMENT --verify-checkpoints
```

The report distinguishes optimizer-consumed sequence counts from completed
asynchronous episode receipts, records rejected-token fractions and phase
timings, and checks saved actor/value-head parameter changes. Inspecting a few
representative tensors is a persistence check, not a proof that every parameter
is numerically well-behaved or that the method improves evaluation accuracy.

## Correctness finding: packed-sequence contamination in the pinned trainer

Development trials showed 33-60% of solver tokens rejected by SAO's token-level
DIS bounds `[0.3, 5.0]`, even at zero staleness, versus ~5% for single-turn
editor sequences. A reference HuggingFace forward over the same recorded
multi-turn solver turns agreed with vLLM behavior log-probabilities
(mean |Δlogp| 0.01-0.04; ≤0.2% of tokens outside the bounds). The fault was in
the pinned AReaL FSDP path: with `attn_impl: sdpa`, several sequences are packed
into one micro-batch and `cu_seqlens` is passed, but SDPA ignores `cu_seqlens`
and the precomputed mask is `None`. Every packed sequence therefore attended to
all earlier sequences in the micro-batch. On a real Qwen3 forward, packing two
sequences changed the second sequence's token log-probabilities by 1.07 nats on
average (max 16); passing `attention_mask=None` does not trigger HF packed-mask
detection here.

`whale_local/sao_bridge/areal-packed-sdpa.patch` builds an explicit
block-diagonal causal mask from `cu_seqlens` for this path only. Packed logits
then match separate forwards to 4e-5 (float32). Actor, proximal recomputation and
critic forwards were all affected, for **both** methods, so every earlier solver
training result (including `dev_r1_*`) is invalid and marked `INVALIDATED.json`.
`verify_upstreams()` records this patch's digest. SAO's objective is unchanged.

## MA-SAO: cross-role experience recycling

In alternating co-evolution, scoring each editor action requires the partner
(solver) to solve reward questions under the proposed harness. Under the current
uncached paired protocol, one H phase with N editor actions and R questions
produces 2 x N x R complete candidate/reference trajectories of the frozen solver
W_t, even when multiple actions propose the same source. Independent SAO reduces
them to scalar editor rewards and discards them. W_t is exactly the next solver
phase's initial policy, so their actions are sampled from its initial policy
conditional on their recorded contexts. Their harness/context distribution need
not equal that of fresh training under the single selected harness.

MA-SAO keeps them. The reward service always records exact prompt/response
token IDs and behavior log-probabilities (both arms, identical cost). Each fresh
solver episode carries `k` recycled partner-evaluation episodes, assigned
round-robin in content-hash order (never by reward), so every optimizer batch has
the same recycled share. Each recorded turn becomes one native SAO sequence with
behavior version 0 and the binary verifier outcome; SAO's own decoupled
importance correction and DIS bounds handle the growing policy lag. Recycled
questions come from the reward split only; each H phase uses a disjoint reward
slice so later editor rewards are never measured on questions the solver trained
on. Development selection and reporting use disjoint dev halves; test is sealed.

Why this is a multi-agent change rather than extra tuning: the recycled data are
the joint system's experience, generated by one agent to evaluate the other.
They span the editor's whole proposal distribution, so the solver learns a
response to the partner's *policy*, not only to the single selected harness —
directly targeting WHALE's observation that weight updates change which harness
is best. In the reverse direction, counterpart-refresh replay re-credits earlier
editor actions under the updated solver. Costs: extra optimizer tokens per step
(recorded in the compute ledger) and stale data late in a phase (monitored via
native staleness and DIS rejection statistics).

### Objective and identifiability limits

Token likelihood ratios correct a changed action distribution conditional on a
recorded prefix; they do not transform a mixture of harnesses into the
state/transition distribution of the selected harness. Tool-turn limits,
truncation rules and tool-output limits can change the environment even when
the visible text prefix is identical. Actor recycling therefore implements a
multi-harness training objective, not a proven unbiased estimator of the
selected-harness-only objective. Critic warm-up on that pool also needs this
qualification: an unconditioned value model may average different hidden
harness contexts.

Likewise, refreshing an old editor action under the new solver corrects its
counterpart-dependent reward but does not remove differences between the old
incumbent context and the current one. Its stored original reference is
intentional and must not silently be replaced. A useful diagnostic is to group
partner episodes by the content hash of the actual harness source, not by the
evaluation receipt filename, and report how many match the chosen harness.

Minimal causal checks are warm-up-only, actor-recycling-only and their
combination, with and without editor reward refresh. Report unique episodes,
their reuse multiplicity, optimizer steps/tokens and GPU time. A repeated-fresh
data/extra-value-optimization control can test whether a gain is only due to
extra optimization, but it is an ablation rather than a replacement for the
user's independent-SAO baseline.

### SFT initialization status

The first teacher-authored rewrite SFT (lr 1e-5, two epochs) improved a small
format check but substantially reduced mathematical performance, so it is
rejected as a shared actor initialization. A gentler native-WHALE SFT (lr 2e-6,
one epoch) retained the original math point estimate on 64 development questions
and corrected one development edit example; uncertainty remains wide.
Actual-prompt proposal diversity fell to one configuration in 32 draws.
It is only a provisional starting point for the short independent-SAO H-W-H
probe, not an established stronger base model or a frozen baseline. That probe
has now failed its acceptance gate: 40.039% initially versus 28.027% finally.
Both actors in that diagnostic started independently from the same SFT v2
checkpoint; this does not justify using pure editor SFT as a solver initialization.
See the [experiment log](EXPERIMENT_LOG.md) for settings, paired intervals and
the 12,976-generation cap. Neither SFT pilot opened AIME.

The experiment specification now optionally accepts `editor_model`; `model`
continues to identify the solver's initial model. Omitting `editor_model`
preserves the shared-initialization behavior. Each independent actor and critic
starts from its own role's model, and the frozen contract binds both model
digests when they are distinct. A legacy contract cannot silently accept an
editor override. Cross-role shared-critic transfer with distinct initializations
is explicitly rejected until that combination is validated. This interface is
readiness for role-appropriate initialization, not a completed new experiment.

Larger rewrite evaluations accept `--samples`, `--math-samples`, `--edit-tokens`,
`--math-tokens`, `--evaluation-gpus` and `--server-context`. Eight-sample reporting
must be declared for both control and candidate; different token/context budgets
are rejected by the paired-comparison helper. For evaluation of an already saved
checkpoint, use `--mode evaluate --model CHECKPOINT --origin-model ORIGINAL_BASE`;
no optimizer is launched in that mode. Use `--dev-only` during selection and
evaluate the fresh test set only after choosing the checkpoint. The default
settings remain compatible with the old pilot. These options do not by
themselves constitute a completed large-scale evaluation.
