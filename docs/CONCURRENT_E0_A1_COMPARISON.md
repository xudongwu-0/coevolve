# Fresh E0/A1 concurrent comparison (2026-10-13)

## Current execution: full MA and four single-add-back ablations

The latest authorization replaces the ordinary-SAO comparison with the study
below. The ordinary ten-task differential objective remains blocked and is not
silently renamed or approximated by the joint-only baseline.

| Arm | Every-round joint updates | Cooperative Critic context | Bounded peer correction | Per-role ValueNorm |
| --- | --- | --- | --- | --- |
| `ma_sao` | yes | yes | yes | yes |
| `joint_only` | yes | no | no | no |
| `add_context` | yes | yes | no | no |
| `add_peer` | yes | no | yes | no |
| `add_valuenorm` | yes | no | no | yes |

The four ablations are the all-off baseline and three individual add-backs,
not leave-one-out variants. All five retain shared episodes/team terminal
rewards, independent role-local Actor/Critic parameters, native PPO self-policy
correction and Actor advantage standardization. Every arm starts from the
same official E0/A1 Actors with fresh Critics/optimizers, uses the same ordered
1152 input rows and seed1000043, runs144 joint rounds, and saves36/72/108/144.
Three Critic-only warmup rounds imply141 Actor and288 Critic updates per role.
Each arm evaluates its own final144 pair immediately after its roles release
their GPUs. Selection48 is contained in the reused main200 development set.

Add-backs use `standalone-ma-ablation-v1` and `ma-sao-single-addback-v1`;
the native entry requires exact mechanism flags, spec hashes and an
allocation-bound manifest naming only that arm. Full MA's existing entry is
preserved. No historical cancellation gate or frozen evidence is changed.

Each job requests two nodes /16 H100,64 CPUs and500G per node, four hours,
`root/hpc-mid/normal`, without requeue. At most four jobs can be requested
concurrently: pending requests also count toward the eight-node user limit.
The ValueNorm job remains unsubmitted until a successful final endpoint frees
capacity. The monitor rechecks live capacity and the submission wrapper
rechecks policy, hashes and quota. Failures halt automatic backfill and require
diagnosis; they do not trigger blind retries.

Slurm snapshots are authoritative. Project tests/builds execute remotely;
Windows holds checked mirrors and transport tools, not the training runtime.
The reviewed scripts are mirrored under
`slurm-comparison/remote-snapshots/{arm}` in the enclosing workspace.
The latest source passed113 remote tests plus3 subtests, and a real
comparison-bound ALFWorld worker startup/reset. These do not establish warmed
optimizer memory fit or completion within the four-hour wall limit.

The worker executable must retain its virtualenv invocation path. Resolve its
target to check project containment, but do not replace the invocation with
that target: doing so loses `pyvenv.cfg` and the worker's installed packages.

## Earlier execution plan: independent MA before ordinary SAO

The latest authorization permits MA-SAO to start without waiting for ordinary
SAO. Each arm evaluates its own committed144 endpoint immediately after its
two training roles finish and release their GPUs; neither waits for the other.
The resource budget is two nodes /16 H100 per arm,64 CPUs and500G per node,
four hours per job, `root/hpc-mid/normal`, and at most eight total user nodes
including pending requests. Four hours is a wall limit, not a measured ETA.

[ma_slurm.py](../whale_local/sao_bridge/ma_slurm.py) implements the independent
MA entry. Only `standalone-ma-v1` specs with the separate
`ma-sao-every-round-joint-v1` definition, exact manifest hashes, and an MA-only
allocation binding can pass the native entry guard. The ordinary-SAO prototype
remains blocked. The root-level `slurm-comparison` folder contains the matching
two-node batch, node environment, source snapshot, and guarded submission tools.
No historical cancellation marker or frozen evidence is changed.

The batch runs per-node eight-GPU communication checks and FLA forward/backward
checks before preparing144-round specs. It then starts both MA roles, verifies
paired native144 completion, and evaluates only that final checkpoint. The
communication/operator checks do **not** establish full-model memory fit,
checkpoint round-trip, or end-to-end speedup; these remain first-run validation.
There is no automatic retry/requeue. Source is copied to a separate execution
snapshot so later SAO development cannot alter MA's Python code mid-run.

## Historical ordinary-SAO proposal: still blocked, not part of the current study

The14:33 clarification rejects changing ordinary SAO's Engineer objective merely
to equalize environment counts. Accordingly `prepare`, `role` and `final-eval`
in the combined prototype **fail before process launch**. The
shared-episode independent-SAO code and configuration below are an unlaunched
prototype, NOT an implementation of the newly clarified comparison. Native
validation receipts cannot bypass this objective blocker. The native
`large_train` and `joint_experiment.run_role` entries also reject prototype specs
before creating run files or starting environment services.

### Recommended precise simultaneous comparison

Use a **new every-round-joint paired schedule for BOTH methods**, with144
collection/update rounds, not historical MA Full's joint/joint/Solver-only.
It is also not the old isolated `always_joint` ablation, which retained
third-round unit peer weighting. Declare the actual new schedule separately.

For each SAO round and each of eight Engineer slots:

1. Bind its original ten-task training packet, prompt, task IDs and a definite
   current Solver serving-version snapshot. Retain the original frozen prompt
   evidence, but never reuse its old Solver's numerical baseline reward.
2. Evaluate the generated candidate and the empty Harness on those same ten
   tasks, with the pinned official scoring/invalid-output rules. Compute the
   original mean patched-minus-empty reward. Both sides of each difference
   must use the same immutable Solver version and decoding contract.
3. Separately collect a fresh Solver training episode using that slot's
   generated Harness, with the ordinary Solver terminal reward. Do not
   silently recycle the ten scoring episodes into Solver optimization.
4. Update both role-local SAO learners in the same paired round, with separate
   Critics and no MA peer weighting/context/ValueNorm. The sampled current
   Harness replaces the long-stage fixed Harness; this is an explicit
   concurrent scheduling/routing adaptation, not staged SAO reproduction.

MA retains its shared-episode team rewards and MA learning components, but
updates both roles every round. Both arms can match1152 Engineer actions,
1152 Solver optimization trajectories, initial Actors, seed, native optimizer
settings,141 Actor/288 Critic updates per role, and fixed144 endpoint.
**Do not call this equal environment/token/FLOP or wall-clock budget.**

Without reuse or invalid-candidate shortcuts, SAO additionally runs
1152×10 patched and1152×10 baseline reward episodes, for24192 environment
episodes including its1152 fresh Solver-training episodes, versus MA's1152.
These are nominal counts, not measured costs. Baseline reuse is permitted
only for identical Solver version, task cohort, seed and decode identity.
Report actual baseline/patched/optimization counts separately, cache reuse,
all invalid candidates, failed/in-flight attempts and token costs. No reuse
across Solver updates and no substitution of old stored W0 rewards.

### Required runtime work before wiring Slurm

The proposed process layout is still four role processes, one/node,
4 learner+4 rollout GPUs each, two nodes/arm, loopback-local AReaL groups and
shared per-arm coordination files. Each arm has its own two-node allocation and
own-completion evaluation in that allocation. This GPU split is a
candidate requiring the CPU-offload and warm-Adam validation below, not a
claim that the historical GPU-resident configuration fits H10080GB.

However, existing `JointWorkflow` only implements a single shared terminal
outcome. Existing `alfworld_reward_service` assumes a fixed Solver version,
baseline metadata and external serving URLs. Joining those unchanged would
either use stale baselines or allow model updates between the ten patched
and empty evaluations. A new version-pinned reward request/response path,
serving-weight-update lease/barrier, and separate scoring-vs-training
trajectory/cost ledgers are needed. No extra inference GPUs should be silently
assumed. The protocol must preserve the declared lag policy in both arms,
not silently set lag0 and thereby neutralize peer correction.

The proposed CLI names below can remain the parent-facing contract once that
implementation exists, but **there is currently no valid ordinary-SAO training command**.
Initial model directories must remain read-only: validate existing processor
metadata, or use run-local overlays if files are missing; never unconditionally
write `runtime_supplements.json` into E0/A1.

### Reserved wrapper interface (not launch authorization)

Use `/mnt/vast/chengboliu/xudong/.envs/coevolve/bin/python -B` from the complete
checkout, with its `whale_local` and pinned `third_party/sao/vendor/AReaL`
directories on `PYTHONPATH`. Bind the four actual allocated hostnames once:
node0=`ma_sao/editor`, node1=`ma_sao/solver`, node2=`sao/editor`,
node3=`sao/solver`. Each node runs **one controller process**, which owns its
local learner/rollout children; do not start four independent controllers per
node or create a cross-node `torchrun` world. Expose the node's eight allocated
GPUs to its controller. Local AReaL masters/RPC use loopback; no global
`MASTER_ADDR`/`MASTER_PORT` is part of this interface. The missing SAO reward
version/lease protocol must be implemented before settling any additional
cross-role serving endpoints.

All commands use `-m sao_bridge.concurrent_comparison`:

| Phase | Arguments |
|---|---|
| Preparation, once | `prepare --config CONFIG.json` |
| CPU manifest inspection | `verify --root RUN` |
| Each of four concurrent roles | `role --root RUN --arm ARM --role ROLE --native-validation RECEIPT.json` |
| Endpoint gate after all four successful exits | `final-paths --root RUN` |
| Each arm's final evaluation | `final-eval --root RUN --arm ARM --native-validation RECEIPT.json` |

Parent owns Slurm submission, actual node binding, child-process supervision,
failure fan-out and audit logs. The combined interface above is historical
scaffolding, not the current launch contract. Each arm now requires **its own
successful completion, both roles' process/GPU release, and native144 gate**
before immediately evaluating on its own Engineer node. Never infer144
completion from an HF directory alone.

Keep inherited `SLURM_JOB_ID`; do not set legacy `SAO_RESOURCE_AUTHORITY*`.
The role/evaluation entry creates its hash-bound `COEVOLVE_COMPARISON_*`
environment and run-local cache/HOME/IPC directories. Parent must likewise
keep bootstrap paths and every controller/Slurm log under the approved tree
and set `PYTHONDONTWRITEBYTECODE=1` before Python starts. The config's
`worker_python` identifies the verified copied ALFWorld interpreter.

**The combined prototype deliberately exits with an objective blocker. Only
the separate MA entry described above is exempt; do not submit the prototype
as ordinary SAO.**

### H10080GB memory findings and candidate layout

Read-only safetensors-header inspection of both verified initial directories
found exactly9,409,813,744 Actor parameters, including1,017,118,720 LM-head and
456,010,480 vision parameters; tensor payload is17.527GiB in BF16. The pinned
Qwen3.5 Critic uses the same multimodal backbone, drops the LM head, and adds
4096 scalar-head weights:8,392,699,120 parameters.

With four FSDP ranks, FP32 Actor+Critic masters and two FP32 Adam moments plus
a BF16 reference budget approximately54.12GiB **per learner GPU**, before
gradients, all-gathers, logits, activations and allocator overhead. One Actor
gradient shard adds8.76GiB. This is an all-parameters optimizer-state estimate;
unused vision parameters may have no lazily created Adam moments. It is not a
measured peak or a rigorous lower bound. It does establish why microbatch1
alone is not an adequate memory argument.

The pinned AReaL implementation supports `fsdp.offload_params=True` through
FSDP2 `CPUOffloadPolicy`, with per-layer resharding after forward. PyTorch2.10's
policy moves parameter shards, gradient shards and optimizer states to CPU.
Its reduce-scatter implementation explicitly accumulates consecutive
microbatch gradients in CPU DTensor storage, synchronizing the second copy
before the CPU add. Its fused-optimizer supported-device list includes CPU,
so AReaL's existing `AdamW(fused=True)` constructor is not CUDA-only.

The candidate template therefore now defaults to CPU parameter offload for
Actor, Critic and BF16 reference, keeps FP32 optimizer storage, and explicitly
leaves `per_layer_optim_step=False`. This preserves batch8 on four DP ranks,
one complete trajectory/microbatch and two accumulated microbatches/rank.
It does not change sequence lengths, scoring, optimizer update counts or
numerical storage dtypes. Six-way DP is not a supported substitution: the
current adapter rejects batch8 not divisible by6.

CPU offload budgets approximately216.49GiB/node for the same persistent state
and up to66.32GiB/node for both engines' gradient shards. Provision substantial
headroom beyond their282.81GiB sum for independent CPU model loading on all
four ranks, pinned buffers, workers and native checkpoint serialization.
512GiB/node is a conservative candidate reservation, **not a measured minimum
or a verified available allocation**. Parent's14:53 report of live `hpc-mid`
metadata confirms128 CPUs and `RealMemory` approximately2,062,756MB per
eight-H100 node, so physical host capacity is not the obvious limiting factor.
Parent must still verify actual allocated/cgroup RAM and pinned-memory limits.
CPU Adam and repeated PCIe transfers may impose substantial throughput costs.

Do not substitute engine `enable_offload`/`torch_memory_saver`: that is a
different mechanism from FSDP CPU offload. Pinned AReaL also offers GPU-streamed
per-layer Adam with CPU moments, but it eagerly creates zero-step states for
all trainable parameters. Current continuation fingerprints reject any
zero-step state, including unused vision parameters. Enabling that optimization
would require a scoped, tested continuation-format adaptation first; this
prototype deliberately does not enable it.

A real H100 validation must still cover9B loading, FLA/vLLM, full32K/4096
examples, two accumulated microbatches, warmed Actor+Critic Adam, GPU peak and
host/cgroup peak, throughput,36-round paired native save/load, restored
optimizer/RNG/cursors, and write-boundary/IPC auditing. None of the header or
source inspection, CPU tests, or earlier tiny-DDP smoke establishes those facts.

---

## Superseded shared-episode prototype (not authorized for execution)

This is a **new live experiment**, not a restart of Full144, native72, or an old
waiter. Frozen evidence, source snapshots and cancellation records are unchanged.
Preparation and CPU tests do not authorize or establish successful GPU execution.

## Scientific contract

Both arms use official `harness-r1` (E0, trained against vanilla A0), and
`target-agent-sft` (A1), from
`ShaoShuai0605/Harness-R1@f6a915953f8206f3f5c7ef0b40240e1f4d2b6676`.
Neither E1 (`agent-sft-harness-r1`) nor local W1/pre-MA Critics is used.

Both actors participate and both learners are active in **every** collection
round. Each role has three Critic-only warmup rounds, followed by141 Actor
updates; both Critics perform288 updates. Each arm consumes1152 paired
episodes/2304 role trajectories, batch8, seed1000043, context32768 and4096
action tokens. Actor/Critic LR is5e-7/2.5e-6, fixed-reference KL0.02, and K=2.
The same1152 pinned prompt/task records and order are used in both arms.
Reusing the historical joint input ledger retains its input repetitions and
historical W0 evidence; it does **not** replay its generated trajectories.
Anchor metadata is removed: no third-round freezing or special peer weighting.

| Mechanism | MA-SAO | Concurrent independent SAO |
|---|---|---|
| Native single-rollout decoupled PPO/self-policy ratios | yes | yes |
| Actor advantage standardization, complete-episode GAE | yes | yes |
| Per-episode shared team terminal reward | yes | yes |
| Bounded peer trajectory weight | yes, every round | no (unit) |
| Extra pre-action cooperative Critic prefix | yes | no |
| Per-role Critic ValueNorm | yes | no |
| Separate role parameters and simultaneous schedule | yes | yes |

**Naming limitation:** the baseline is explicitly
`independent-sao-shared-episode-v1`, not historical staged SAO.
Historical Engineer SAO uses mean patched-minus-empty reward over ten tasks
against a frozen Solver. That reward/collection protocol is NOT mathematically
equivalent to the shared-episode baseline. Under the concurrent comparison,
each role instead optimizes its own native SAO objective on the common team
episodes; no peer factor modifies its normalized advantage, and its Critic
sees only its own trajectory in raw return units. Consequently this removes
the full MA learning bundle, not just `without_peer`. The configuration must
explicitly name this adaptation. If “SAO” must retain the old ten-task
differential objective, **do not launch this comparison**: matching both its
environment cost and the requested1152 paired-episode budget requires a
different explicitly approved sampling/reward protocol.

### Fresh Critics

Each Critic loads its corresponding E0/A1 backbone through the existing pinned
Qwen3.5 token-classification loader (`init_from_scratch=false`, `num_labels=1`).
Only the absent scalar `score` head is newly initialized, not the backbone.
The new live profile forces CPU memory-efficient loading and isolates head
initialization with seed1000043 without advancing the Actor CPU RNG. The
native gate additionally requires matching fresh-head fingerprints across
the two arms for each role. No old Critic, Adam, RNG or ValueNorm is loaded.
MA starts empty role ValueNorm; SAO uses raw values. References are each role's
initial Actor. Three warmup rounds are calibration, not imported Critic skill.

## CPU preparation and allocation contract

Module: `sao_bridge.concurrent_comparison`, with `PYTHONPATH=whale_local`.
`prepare --config CONFIG.json` creates a **new** directory, immutable
`COMPARISON.json`, common data/evaluation inputs, and four role specs.
`verify --root RUN` checks all pinned inputs and live bridge source hashes.
Preparation requires explicit paths; it does not discover or silently reuse
historical models.

Configuration fields:

The checked-in `examples/concurrent_e0_a1_144.json` is a **fail-closed template**:
replace allocated host placeholders, set a future deadline and provide the
explicit path-adapted evaluation inputs before preparation.

* `run`: new directory beneath `/mnt/vast/chengboliu/xudong`.
* `baseline_definition`: `independent-sao-shared-episode-v1`.
* `model_revision`: the revision above.
* `engineer_model`, `solver_model`: verified
  `coevolve/assets/Harness-R1/harness-r1` and `.../target-agent-sft`.
* `source_data`: JSON list of exactly1152 prompt/task records. The complete
  clone's bundled `.../alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/joint_data.json`
  is a possible explicitly disclosed source; do not modify that source.
* `no_patch`: `coevolve/examples/alfworld_no_patch.json`.
* `worker_python`: copied `/mnt/vast/chengboliu/xudong/.envs/alfworld/bin/python`.
* `nodes`: `{"ma_sao":{"editor":"HOST0","solver":"HOST1"},
  "sao":{"editor":"HOST2","solver":"HOST3"}}`, populated from the actual allocation.
* `training_deadline_unix`: explicit future deadline; optional
  `phase_timeout_seconds` (default48h).
* `fsdp_offload_params`: boolean, defaulttrue; same for both arms. Enabling
  this existing upstream option requires validation of that exact profile.
* `evaluation_protocol`: new, path-adapted JSON with `core200_protocol`,
  `evaluation` (selection manifest, samples1, seed, temperature, solver_tokens4096,
  and bounded concurrency fields), `h2.template`, and `h2.proposal_seed`.
  These are the existing `matrix_config`/`generate_proposals` fields.
* `initial_state`: new JSON with explicit `packets`; preparation pins its hash
  and overrides all Actor/Critic model identities with E0/A1.
* Optional existing FLA profile: `training_kernel_overlay`,
  `training_kernel_versions`, `training_kernel_validation` (passed numerical
  receipt, pinned). This does not replace the new H100 native gate.

The parent Slurm wrapper owns submission, priority verification and quota.
Use one four-node allocation, two nodes per arm; reserve the other four
allowed nodes for ablations. The adapter verifies a running four-node
allocation owned by the current UID via read-only `scontrol`, plus exact
node binding and eight H10080GB GPUs/node. It never requests nodes, changes
priority, operates systemd or touches another user's processes.

Each role uses its node's4 learner GPUs and4 dedicated single-GPU rollout
replicas. AReaL masters/RPC remain loopback-local; no cross-node `torchrun`
world is needed. H/W share the arm's `joint/` directory. Identical ports on
different role nodes do not conflict. Parent starts all four commands
concurrently and propagates any failure to all owned job steps:

```text
python -B -m sao_bridge.concurrent_comparison role --root RUN
    --arm {ma_sao,sao} --role {editor,solver} --native-validation RECEIPT.json
```

There is no auto-retry/resume CLI. An interrupted run leaves paired native
states and failure evidence for separately reviewed native recovery; never
restart it as fresh or delete a cancellation gate.

## Memory, paths and mandatory remaining gate

Learner microbatch is already **one complete trajectory**; batch8 on4 ranks
means two accumulation microbatches/rank. Reducing the sequence budget would
change the scientific contract. Rollout concurrency is reduced16→8 and
vLLM max sequences16→2 per replica, without changing context/action limits.
CPU model loading avoids an unsharded9B GPU loading spike. FP32 learner masters
and optimizer, BF16 fixed references, and checkpointing are retained.
This is not evidence that warm Adam plus32K activations fits80GB.

`Launcher` uses run-owned IPC and cache directories; TMPDIR/TMP/TEMP, HOME,
HF/XDG/compiler caches and worker paths stay under the project boundary.
The profile disables NCCL shared-memory transport and vLLM V1 multiprocessing.
**These environment switches alone do not prove all third-party IPC avoids
`/dev/shm`.** No automatic native-gate generator is supplied.

Execution fails closed without `concurrent-h100-native-gate-v1`: matching
COMPARISON/spec SHA values; all four arm/role profiles with32K/4096, microbatch1,
at least two Actor/four Critic warm-optimizer steps, finite nonzero gradients,
H10080GB, matching fresh-head SHA values and hash-bound actual evidence;
successful native checkpoint roundtrip; audited write boundary and no
outside-root writes. A tiny DDP smoke, import success, or old H200 receipt
cannot satisfy it. FLA and real vLLM/tool generation need actual validation.
Do not fabricate this receipt to unblock launch.

## Checkpoints and final evaluation

Both HF exports and complete paired native checkpoints are saved at
36/72/108/144. Only Solver publishes COMMITTED/LATEST after verifying both
roles' model/Adam/ValueNorm/RNG/schedule/cursors. All four native generations
are retained for this new experiment, unlike the old rolling12 retention.
Provision several TiB (historical paired native state was about384GiB,
roughly3TiB for eight retained pairs, plus HF exports and working space).
An HF Actor directory by itself is not resumable state.

After **all four roles finish and release resources**, on a node of the same
still-running allocation:

```text
python -B -m sao_bridge.concurrent_comparison final-paths --root RUN
python -B -m sao_bridge.concurrent_comparison final-eval --root RUN
    --arm ma_sao --native-validation RECEIPT.json
python -B -m sao_bridge.concurrent_comparison final-eval --root RUN
    --arm sao --native-validation RECEIPT.json
```

`final-paths` validates both native144 commits, role updates and metrics,
release receipts and the protected final HF paths (`globalstep143`).
No intermediate checkpoint is evaluated. For each arm, final Engineer
produces exactly eight candidates; selection48 chooses against the **empty
Harness**, with the existing2/48 gain rule. Then selected and empty Harness
are scored on main200, with no checkpoint search or additional allocation.
No old incumbent Harness is silently reused. Selection48 is contained in
main200: these remain reused development data, not an untouched test.
Partial evaluation stops for diagnosis instead of rerunning good results.

## Validation

New standard-library CPU tests:
`tests/test_concurrent_comparison.py`.
Staged full-environment CPU tests:
`tests/test_concurrent_native_contract.py`, plus existing
`test_joint_recovery.py`, `test_joint_component_ablation.py`,
`test_joint_schedule.py`, `test_joint_value_norm.py`, and resource/launch tests.
Run with CUDA hidden and a pytest base directory **inside the project**;
disable pytest's cache provider if the Windows-mounted checkout denies cache
directory operations. No GPU/resource requests are made by these CPU tests.
