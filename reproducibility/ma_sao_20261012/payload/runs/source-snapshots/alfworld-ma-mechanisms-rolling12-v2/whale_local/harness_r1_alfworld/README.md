# Official Harness-R1 ALFWorld CPU runner

External adapter for upstream commit
`94f2e087f573e1b82fc9568bb634d4ae9a887e28`. Upstream source is unchanged.
Run on **node3**, not node4. This module never starts model servers or touches
existing services. CPU controller/workers use an isolated environment.

## Commands

```bash
ssh -F /mnt/3fs/xudong/cluster-ssh/config -o BatchMode=yes node3-internal
cd /mnt/3fs/xudong/coevolve
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$PWD/assets/alfworld_harness_r1/scratch"
export ALFWORLD_DATA="$PWD/assets/alfworld_harness_r1/data"
export PATH="/dev/shm/harness-r1-alfworld-venv/bin:$PATH"
python -m whale_local.harness_r1_alfworld.driver prepare
python -m whale_local.harness_r1_alfworld.prepare_assets
python -m whale_local.harness_r1_alfworld.smoke
python -m whale_local.harness_r1_alfworld.protocol_smoke
# Full sparse-cohort plumbing check (~5 minutes CPU; synthetic endpoints only):
python -m whale_local.harness_r1_alfworld.protocol_smoke --size 32
python -m unittest whale_local.harness_r1_alfworld.test_driver -v

# URLs/model alias must match the separately managed released-9B Engineer server.
python -m whale_local.harness_r1_alfworld.driver paired \
  --size 12 --run-id diagnostic12_released9b_001 \
  --target-url http://TARGET_HOST:TARGET_PORT/v1 \
  --target-model policy \
  --engineer-url http://ENGINEER_HOST:ENGINEER_PORT/v1 \
  --engineer-model ENGINEER_SERVED_ALIAS \
  --engineer-checkpoint-record RELEASED_9B_CHECKPOINT_PATH_OR_REVISION \
  --concurrency 4
```

The first actual-checkpoint diagnostic defaults to **12 tasks**, covering all
six types, with one engineer patch and exact same-task rerun. Use `--size 8` for
an even smaller first answer. Run `--size 32` only after this first result.
No additional installation or RL backend is required. A single target/Engineer
replica pair is enough to start; do not wait for all eight GPU servers to become
ready if the first two healthy endpoints are already available.

The target checkpoint is original `Qwen3.5-9B`; its wire alias can be
`Qwen3.5-9B` or `policy`. Sampling is locked to temperature 0, nonthinking,
4096 tokens, official `take_action` tool schema with `tool_choice=auto`.
Engineer uses thinking enabled, temperature 0, 12288 tokens,
literal assistant prefill `<think>\n`, and **strict `prefill_think_patch`**,
not a relaxed parser. Official multihook schema permits real reusable Python
code-hook edits. Provide the actual released 9B checkpoint served alias; the
adapter cannot infer checkpoint provenance from an arbitrary alias.

### Serving and placement constraints

Official code accepts **one URL per role**, not comma-separated replica URLs.
Expose four target and four Engineer replicas through two transparent HTTP
load balancers. Forward JSON bodies and model alias unchanged; per-request
round-robin works because each request carries full chat history. One cohort
batch produces one engineer patch request, so four Engineer replicas do not
increase throughput for this single-batch diagnostic.

The Engineer request appends final assistant content exactly `<think>\n` and
sets `continue_final_message=true`. Respect continuation without another
assistant generation prefix; the adapter passes official
`--engineer-extra-body '{"add_generation_prompt":false}'` explicitly.
Strict responses must be reasoning continuation followed by
`</think><patch>JSON</patch>`; another opening `<think>` is explicitly rejected.
The official evaluator also accepts separated `reasoning_content`/`reasoning`
and patch-only `content`, synthesizing one closing tag. Avoid duplicated tags.
Safest deployment uses raw Engineer continuation content without a reasoning
parser; the target uses the original tokenizer, nonthinking template switch,
and `qwen3_xml` tool parser producing OpenAI `tool_calls` with JSON arguments.
Target requests have no assistant prefill; Engineer requests have no tools.
The original Qwen template may render an empty closed thinking block before
the explicit final assistant prefill; do not rewrite the tokenizer to remove
it. The completion itself must not restart `<think>`.

The official runner launches controller, assigner, and worker together and
hardcodes their mutual URLs to `127.0.0.1`. Keep all CPU orchestration on node3
and tunnel only the two model/load-balancer endpoints from node4. Alternatively,
run the **entire** CPU stack on node4 after provisioning its own isolated
environment and writable shadow runtime. A node4-controller/node3-worker split
is not supported by the unchanged runner. All CPU processes keep
`CUDA_VISIBLE_DEVICES=''`; model supervision remains separate.

Each run has a fresh, non-overwritable directory:
`assets/alfworld_harness_r1/runs/<run-id>/`. It retains baseline traces,
SHA256 baseline identity, immutable task selection manifest, failure packets,
raw official engineer patch and paired evaluations, and `adapter_summary.json`.
Both endpoints must advertise their requested aliases through `/v1/models`;
the response, source hashes, environment/asset hashes, and checkpoint provenance
record are captured in the immutable manifest. Authentication can be supplied
through `HARNESS_R1_ENGINEER_API_KEY` for the official engineer request.
The shadow AgentBench directory symlinks official code; generated configs and
assets live outside upstream.

## Data and interpretation

`cohort12.json` (or `cohort8.json`/`cohort32.json`) selects the first tasks of each of six task types in deterministic
round-robin order from the official 500-task `test_set.json` (mixed train and
validation origins). This is **transductive diagnostic, not an untouched test**
or a replication claim. A patch sees failures from the same tasks it reruns.
All selected original task IDs are retained; no resampling or task answer edits.

The official builder in this pinned revision omits sparse `task_ids` and computes
batch size from the index range. The adapter first requires complete, non-infra
baseline results, invokes the unchanged builder with `--include-incomplete`,
then adds verified task IDs to packet metadata. The unchanged evaluator supports
these task IDs natively. Prompts/engine remain official.

Official evaluator fallback scores for infrastructure failures are **not**
valid paired evidence. The external summary separates infrastructure errors,
invalid patches, compiled-configuration no-ops, and evaluated patches. No-op
detection checks exact compiled task-definition equality, not arbitrary code
semantic equivalence. Never interpret zero
delta from fallback as a failed edit. Baseline files are rehashed after evaluation.
CPU expert smoke validates real game transitions/solvability only: it is not
an LLM success score. It requests the official handcoded expert in train-mode
with domain randomization off and a 150-step smoke budget; model evaluations
keep the official 50-step budget. The bounded protocol smoke uses a local
CPU HTTP stub (always `take_action("look")`) for two steps and validates the
unchanged controller, worker, assigner, and tool-call plumbing end-to-end.
Its rows are explicitly labeled **not model evaluation**.
It also sends a synthetic strict engineer response with a real `make_pre_hint`
Python code hook through the unchanged official evaluator and reruns the same
task. This validates edit application, not released-checkpoint quality.

## First actual-checkpoint diagnostic (2026-09-30)

`assets/alfworld_harness_r1/runs/diagnostic8_released9b_20260930_1406/`
contains the first actual run: original Qwen3.5-9B and released Engineer revision
`f6a915953f8206f3f5c7ef0b40240e1f4d2b6676`, served separately on node4 while
all ALFWorld orchestration ran CPU-only on node3.

* Baseline **4/8 (50%)**; same-task patched result **4/8 (50%)**, **0 pp delta**.
* One valid, non-no-op patch added four real Python hooks; no infrastructure
  failures or invalid patches. No rescued or regressed tasks.
* Generated initialization guidance appears in all eight patched trajectories,
  and seven action sequences changed: this was an applied edit, not a mock.
* Three successful trajectories shortened (7→3, 20→14, 4→3 actions); the fourth
  successful trajectory stayed at 29 actions. This does not establish a general
  improvement or change the primary success result.
* Baseline stage took about 81 seconds; actual Engineer generation plus patched
  evaluation took about 81 seconds. Frozen baseline and control hashes were
  independently reverified.

`first_result.json` gives paired task-level results and execution evidence;
`adapter_summary.json` gives primary metrics, and `patched/baseline/patch.json`
retains the actual generated code. This remains a **transductive diagnostic,
not an untouched-test result or full Harness-R1 replication**.

The subsequent **cold-start-sft Engineer** comparison reused the exact frozen
failure packet and the **original Qwen3.5-9B target** (not Solver-SFT):
**4/8→5/8 (50%→62.5%, +12.5 pp)**. One valid four-hook patch rescued task8;
no regressions, infrastructure failures, invalid patches, or configuration
no-ops. Original baseline/control hashes and packet hash were reverified.
Engineer generation plus paired evaluation took about 76 seconds.

Evidence:
`runs/harness_r1_alfworld_20260930/node3/coldstart_samepacket8_20260930_1415/adapter_summary.json`.
This tiny transductive result does **not** establish a general cold-start versus
RL Engineer ranking. Solver-SFT standalone results are a separate Solver
checkpoint comparison and must never be counted as harness gains.

The parent also collected a different cold-start patch on this same packet:
it scores4/8, rescuing task8 but regressing task27, after an infrastructure-only
rerun that preserved that patch. Report both4/8 and5/8, not only the positive
proposal. Two cold-start proposals versus one RL proposal are not an
equal-proposal-budget algorithm ranking. See the committed
[project state](../../docs/PROJECT_STATE.md) for the complete evidence ledger.

The standalone **target-agent-sft Solver**, with all harness modules disabled
and no patch, scored **7/8 (87.5%)** on the same task IDs versus original
**4/8 (50%)**: **+37.5 pp Solver checkpoint gain**, **not harness gain**.
It rescued four original failures and regressed one original success (task1).
All eight episodes completed without infrastructure failures, taking about
53 seconds. Evidence:
`runs/harness_r1_alfworld_20260930/node3/solver_sft8_20260930_1418/solver_comparison.json`.

## Two-node ALFWorld SAO/MA-SAO campaign

The frozen protocol is
`examples/alfworld_sao_masao_campaign_20260930.json`. Both arms start from the
same released RL Engineer (`harness-r1`) and Agent-SFT Solver
(`target-agent-sft`). The official 500-task mixed list is locally partitioned
before training:

| Local split | Official IDs | Use |
|---|---:|---|
| train | 0–239 | optimizer packets and W episodes |
| reward | 240–299 | global-harness candidate selection |
| dev | 300–399 | stopping and candidate acceptance |
| sealed test | 400–499 | one final comparison after both arms freeze |

This is a local repartition of the official mixed list, not the paper's
untouched test protocol. Every Engineer reward packet remains a contiguous
10-task cohort because the pinned official reward assumes a continuous
`start:end` range. Formal protocol v2, split hashes, checkpoint-receipt hashes,
and task-type counts are frozen under
`assets/alfworld_sao_masao_20260930/protocol_v6/`. Earlier pre-optimizer
protocol receipts are retained with explicit supersession records.

### Full train-packet starting matrix

Six matched waves cover all 240 training IDs. Each wave ran sixteen jobs:
four contiguous packets for each of the 2×2 released checkpoint pairings.
All 96 jobs completed on the first infrastructure attempt.

| Engineer + Solver | Baseline | Same-packet patched | Net |
|---|---:|---:|---:|
| RL Engineer + Agent-SFT Solver (campaign start) | 166/240 (69.17%) | 184/240 (76.67%) | **+18 / +7.50 pp** |
| Cold-SFT Engineer + Agent-SFT Solver | 167/240 (69.58%) | 157/240 (65.42%) | −10 / −4.17 pp |
| RL Engineer + original Solver | 93/240 (38.75%) | 132/240 (55.00%) | **+39 / +16.25 pp** |
| Cold-SFT Engineer + original Solver | 97/240 (40.42%) | 103/240 (42.92%) | +6 / +2.50 pp |

The campaign-start pairing produced 23 valid patches and one invalid patch,
rescuing 27 task outcomes and regressing 9. These are **same-packet
transductive diagnostics**: the Engineer saw failures from the same ten tasks
that were rerun. They demonstrate a useful nonzero reward distribution and
justify optimization, but are not held-out harness gains. Deterministic
temperature-zero baselines still differed on one Agent-SFT task (ID 134) and
ten original-Solver tasks across independently concurrent pairings; the
task-level report preserves those mismatches rather than silently picking one.

Collection executed 1,750 physical target episodes and 96 Engineer proposals
in 1,432.49 seconds across the six sequential waves (about 1.22 target
episodes/s), with bounded-retry distribution `{1: 96}`. All sixteen GPUs held
real model replicas. Mean utilization was 28.06% on node3 and 29.93% on node4;
Solver replicas averaged roughly 30–50%, while Engineer replicas averaged
11–24%. This imbalance is expected for ten baseline plus patched Solver
episodes around only one Engineer proposal. It is collection telemetry, not
the expected optimizer layout or an algorithm-quality metric.

Primary receipts:

* `runs/alfworld_sao_masao_20260930/matrix_train_000_239.json`
* `runs/alfworld_sao_masao_20260930/matrix_infra_report.json`
* `runs/alfworld_sao_masao_20260930/w0_e0_train_packet_report.json`
* `assets/alfworld_sao_masao_20260930/data/manifest.json`

### Training and evaluation implementation

Engineer SAO uses three FSDP actor/critic GPUs, one vLLM rollout GPU, and four
frozen-Solver replicas. Solver SAO uses four FSDP actor/critic GPUs plus four
rollout GPUs and one official AgentBench worker per packet harness. Both paths
use the pinned SAO/AReaL PPO and critic losses; infrastructure fallback is
raised, never converted to model reward. The released Engineer's literal
assistant continuation (`<think>\n`) is preserved token-for-token.

The sequential arm on node4 trains H to its declared stopping rule, freezes H,
then trains W. The MA-SAO arm on node3 alternates H/W. Its critic receives only
pre-action joint state (role, partner checkpoint/version, harness identity, and
dev calibration); the actor stays decentralized. After W changes, H reward
caches are namespaced by W checkpoint and 25% of later H rows may reuse exact
Engineer behavior tokens/log-probabilities with refreshed counterpart reward
and interaction credit. Solver cross-role replay is deliberately disabled:
the official ALFWorld reward path does not expose exact Solver behavior
log-probabilities, so claiming full MAPPO replay would be unsupported.

Evaluation reports three distinct quantities rather than conflating them:
packet-transductive Engineer+Solver performance, Solver no-patch performance,
and one global harness selected only on reward IDs then applied unchanged to
unseen dev/test IDs. Dev acceptance requires positive paired gain, at least
one percentage point, and a positive one-sided 95% bootstrap lower bound.
Solver checkpoint acceptance compares per-task no-patch rewards against its
incumbent on the same 100 development IDs with that same gate. Global harness
selection is run separately under the resulting Solver state, so a Harness
change cannot be misreported as a weight improvement.
Sealed test was kept unused until both arm states were frozen.

### Optimizer qualification and scheduling fixes

AReaL originally treated Qwen3.5 as a vision-language generation model even
when `is_critic=true`, so the critic returned 248,320 vocabulary logits per
token instead of one value. The pinned local patch adds a Qwen3.5 token-value
head, preserves it across checkpoint save/reload, and rejects any value/mask
shape mismatch. A real H smoke then reached critic backward but 2-way FSDP
OOMed with only 82 MiB free. The frozen layout is therefore 3 actor FSDP
ranks plus 1 rollout rank; it completed one actor and two critic updates with
about 23 GiB peak headroom. A 20k-token microbatch bound prevents two long H
trajectories from being packed on one rank.

The real W smoke consumed 139 action trajectories and 1,386,174 tokens from
one four-episode update. Actor update took 10m32s and two critic updates took
21m47s; the full supervised phase, including model startup, export and hashes,
took 2,713 seconds. Actor GPUs averaged 77.74% utilization in the resident
window (median 100%); rollout GPUs averaged 0.55% because the one-step smoke
spent most of its lifetime in the long optimizer pass. This is an honest
single-step pipeline measurement, not eight-card simultaneous compute.

The first W smoke also exposed an environment scheduler bug: consecutive task
IDs shared one single-session packet worker, causing 3,870 HTTP 406 workflow
failures and repeated sampling. Solver rows are now round-robin interleaved
across all 24 packet assignments. Before any W process claims GPUs, it audits
every asynchronous active window (`batch × (staleness+1)`) and fails if a
worker repeats. The colliding smoke checkpoint is never a formal initializer.

Protocol v6 initially froze `2 steps × batch 4` per staged role round.
H training uses at most 4,096 generated tokens: all 23 valid
released-Engineer train-matrix responses were at most 3,014 tokens, while the
only response reaching 12,288 was invalid. Evaluation retains the official
12,288-token cap. Checkpoint export also restores missing Qwen processor
metadata with a separate receipt, while parameter-change checks hash only
the safetensors weights.

The first formal H1 rollout attempt revealed one more contract mismatch before
any optimizer update: the official multi-hook prompt did not state the existing
per-code limits of five helper functions and 1,200 AST nodes. Node4's first
four candidates were rejected only for those undeclared limits; node3 had not
generated a candidate. Both exact units were stopped with zero metric rows and
zero checkpoint files. Fresh training rows now append the literal existing
validator limits (they do not relax validation or add format reward), and both
arms restarted from untouched E0/W0. The invalid attempts and four physical
rollouts remain archived and are not formal checkpoints.

That H1-r2 restart produced four valid node3 patches with packet deltas
`+0.2`, `+0.1`, `0`, and `-0.1`, confirming the prompt fix. It then failed
before critic forward because AReaL's balanced three-way dispatch requires the
consumer batch length to be divisible by three, not merely at least three.
Node4 independently returned infrastructure HTTP 502 because its UID 1008
could not write a shared AgentBench generated-config directory created by
node3 UID 1009 with mode 0775. Both r2 runs have zero metric rows, zero
checkpoints, and zero optimizer updates.

Protocol v7 preserves every split hash, E0/W0, seed, reward, and optimizer
setting. It changes only the H batch to six, adds a pre-GPU divisibility check,
and gives every H reward or W environment run its own writable AgentBench
layout and scratch directory. Pinned upstream source and ALFWorld data remain
read-only symlink targets. Its manifest SHA256 is
`fa6c2c98315fcfba2a65edd48dfb17ff0c7f4380fea1f35cfa35e57188680eaa`.
Both H1-r3 arms started again from untouched E0/W0.

H1-r3 verified both fixes: node4 completed official run-local evaluations
without permission failures, and both arms entered critic value forward with a
six-item divisible batch. It then exposed that the 20,000-token microbatch
limit was below each sampled 20,309--21,139-token trajectory. AReaL failed
explicitly before `compute_logp` or any optimizer update; both r3 runs still
have zero metric rows and checkpoints.

The exact training chat template over all 64 H1 rows has a maximum 18,606-token
prompt. With the 4,096 generation cap, every possible fresh trajectory is at
most 22,702 tokens. Protocol v8 therefore uses a 23,040-token H microbatch:
large enough for one worst-case trajectory, but too small to combine two long
ones. The launcher now computes and persists this audit before stopping filler
or claiming GPUs. Protocol v8 manifest SHA256 is
`d952ef63f49f83dbb2c1bc730115cdcbfc870043e30253984e3541bed2d5a643`.
Both H1-r4 arms restarted from untouched E0/W0.

Because batch6 consumes 12 distinct rows over a two-step H round, the H
launcher also rejects datasets shorter than `steps * batch_size` or with
duplicate consumed IDs before claiming GPUs. The sequential H2 dataset was
regenerated with 64 unique sample IDs at offsets 64--127 instead of allowing
the old eight-row batch4 file to repeat across epochs.

H1-r4 subsequently completed on both nodes. Each run contains exactly two
native metric rows, two successful actor updates, four critic updates, actor
and critic checkpoints at global step 1, and a changed actor safetensors
digest. Wall time was 1,197.7 seconds on node4 and 1,197.4 seconds on node3.
The longest observed trajectories were 21,918 and 21,958 tokens, respectively,
so the 23,040-token capacity passed with 1,082 tokens of observed headroom.
The asynchronously prefetched second reward batch reduced measured step-1
rollout wait to about 1 ms on both nodes; final actor+critic serialization
still cost 190--207 seconds and is reported separately from optimization.

The dev refreshes produced no-patch/packet-transductive scores of 79/80 on
node4 and 78/81 on node3. Packet-transductive scores remain diagnostics, not
Harness acceptance evidence. The formal candidate sets contained only unique
valid patches from optimizer-consumed behavior-version-0 train receipts:
10 on node4 and 11 on node3. No dev-generated patch entered either pool.
Node4 accepted an H1 global patch at +8pp paired gain with a +2pp one-sided
95% lower bound (79 to 87); node3 accepted one at +11pp with a +5pp lower
bound (78 to 89).

The initial global pass was stopped before selection when the official
evaluator exposed packet-local reward keys 0--9 while the protocol metadata
used global IDs 300--399. `alfworld_global_harness.py` now accepts only the
exact global set or an exact local `0..batch_size-1` set and maps the latter
through the declared packet start. Interrupted retries may seed only from a
prior run with identical evaluation inputs; they exclude stale `.lock` files
and cached infrastructure-failure results. Node3 retry3 consequently reused
109 successful evaluations and recomputed one failed key in 253 seconds.

Node4 H2 completed two more actor and four critic updates in 1,320.9 seconds
but did not improve over the H1 incumbent. H3 restarted from accepted H1
rather than rejected H2 and completed in 1,207.4 seconds; it also produced
zero incremental gain. Two consecutive rejected rounds satisfy the Harness
patience rule, so the sequential arm freezes H1 patch
`60b43b8fd5f64852bd20234dbd4e50e71b46253a6f604adba86bc53f5f49a680`.

Train-packet refresh then reached exact 24-packet coverage through 21+3
partial recovery on node3 and 18+6 on node4. Each arm generated 512
assignment-aligned Solver rows. MA W1 had two zero-update infrastructure
failures before its formal run: first, a full 9.8 KB patch embedded in the
critic context exceeded the reserved 1,024 tokens; second, AReaL's DP4
dispatch supplied one-dimensional per-rank trajectories while the context
prefix assumed `[1, seq]`. Context now stores a stable patch SHA and is
token-audited before GPU allocation. Prefixing now supports 1-D, 2-D, and
general token-aligned leading dimensions and strips the prefix from returned
values.

After those fixes, MA W1 and sequential W1 each completed two actor and four
critic updates in 4,060.0 and 4,065.1 seconds. MA used a 224-token pre-action
centralized context; sequential W1 had no centralized context. Their final
weight digests were `258e52983e8b...` and `81bc16629ec3...`. Paired no-patch
dev gates rejected both: MA was 78 to 78 (gain 0, LCB 0), while sequential was
79 to 81 (+2 points, LCB 0). The associated packet-transductive 77 and 83 are
diagnostics only.

MA H2 then completed in 1,218.0 seconds from accepted E1/W0 with 25% authentic
Engineer replay and a decentralized actor. The two optimizer batches consumed
exactly nine fresh rows and three replay rows. Relaxing replay head lag also
allowed extra version-0 fresh prefetches, so behavior version alone was no
longer an exact consumption marker. The first selector launch was stopped
before a decision. Training now persists the first `steps * batch` ID ledger;
global selection requires exact receipt coverage for its fresh IDs and
excludes replay. The corrected nine-candidate gate rejected H2. H1 re-evaluated
at 90/100 versus a 78/100 no-patch baseline (+12 points, LCB +6); the one-task
change from its first 89/100 run is not credited to H2.

Both W1 checkpoints were therefore rejected. Sequential and MA W2 restarted
from W0, not W1, with non-overlapping Solver sample offsets 512--1023.
Sequential W2 completed two actor/four critic updates in 4,728.7 seconds and
changed its weight digest to `74eb852d15c2...`; its paired no-patch dev result
was 79 to 77 (−2 points, one-sided 95% LCB −4), so it was rejected. MA W2
completed the same update budget in 4,162.8 seconds with centralized critic
enabled and digest `76f476f8cfb6...`; its dev result tied W0 at 78/100, so it
was also rejected. W1/W2 are two consecutive rejected Solver rounds in each
arm, and both arms therefore freeze the released W0.

MA H3 restarted from accepted E1 rather than rejected H2, used a dataset with
25% authentic H2 replay, and completed two actor/four critic updates in
1,163.3 seconds. Its exact optimizer ledger contained nine fresh and three
replay rows; seven valid unique consumed-fresh candidates reached the global
gate, and none beat H1. H2/H3 satisfy the MA Harness patience rule, freezing
H1 patch `6e6750643c1b451c65c9be1db61f73db7e88d3ab9150ca005ec00e37d36a0234`.

The final development comparison is sequential 88/100 versus MA 90/100.
MA minus sequential is +2 points with a one-sided paired 95% lower bound of
zero. This misses the preregistered +5-point, positive-LCB threshold, so
cadence search is not permitted. Both final arms are H1+W0; the accepted
improvement came from global Harness H1, not Solver weight SAO.

### Frozen-arm sealed test

Test IDs 400--499 were first opened only after both H and W roles froze.
Node4 completed all ten packet metadata cohorts directly. Node3 completed nine
and had one infrastructure failure at start 470; only that cohort was rerun,
then the strict merger required exact, duplicate-free coverage of all ten
starts. No infrastructure failure became a model score.

The two independently collected metadata sources contain identical task IDs
and task paths, but their no-patch rewards differ on six tasks (67/100 versus
65/100). Before either frozen patch was evaluated, the symmetric rule was
frozen: evaluate both patches on both sources, average the two rewards within
each task and arm, then bootstrap the 100 task-level mean deltas. This avoids
choosing the more favorable stochastic metadata source after seeing packet
diagnostics.

| Sealed source | No patch | Sequential H1 | MA H1 | MA − sequential | One-sided 95% LCB |
|---|---:|---:|---:|---:|---:|
| node4 metadata | 67/100 | 78/100 | 82/100 | +4pp | −1pp |
| node3 metadata | 65/100 | 78/100 | 84/100 | +6pp | +1pp |
| per-task two-replicate mean | 66/100 | 78/100 | 83/100 | **+5pp** | **0pp** |

Against no patch, sequential gains +12pp with LCB +6pp and MA gains +17pp
with LCB +10.5pp. Thus both frozen H1 patches show positive sealed
generalization. The arm contrast reaches the +5pp point threshold exactly but
its lower bound is zero, so material MA superiority is **not established** and
cadence search remains forbidden. Test results are not used for checkpoint,
patch, cadence, or hyperparameter selection. The full task rows and frozen
rule are under
`runs/alfworld_sao_masao_20260930/final_sealed_arm_comparison.json` and
`assets/alfworld_sao_masao_20260930/configs/final_sealed_replication_rule.json`.

### Adaptive co-resident GPU filler

The standalone `/mnt/3fs/xudong/gpu-utilization/adaptive_gpu_filler.py`
replaces the static 55%-memory, 82%--94%-duty filler on node3/node4 while
retaining the authorized `bruli-h200-load.service`. Its complete unit,
lifecycle manager, tests and operations guide all live outside this repository
in `/mnt/3fs/xudong/gpu-utilization/`. `resources.stop_known()` still stops an
identity-checked legacy filler, but recognizes the exact external adaptive
script as cooperative and leaves it active when a training launcher claims the
node.

The supervisor samples both `nvidia-smi --query-compute-apps` and per-process
`nvidia-smi pmon`, excluding only its own exact child PIDs. Each GPU has an
independent state:

* `idle`: no external CUDA process; a 4,096 TF32 worker uses 192 MiB of matrix
  buffers and ramps slowly toward 86% total utilization.
* `yielded`: a new, active, reserved, hot, or memory-constrained external
  workload; the worker exits and releases its CUDA context and buffers.
* `shared`: an external process has remained at no more than 5% SM for at
  least six seconds; a 2,048 worker uses 48 MiB of matrix buffers. Duty caps
  rise from 15% to 30% after 30 seconds and 50% after 120 seconds of sustained
  external idleness.

Scale-down is immediate and scale-up is limited to five duty points per
two-second sample. Less than 8 GiB free memory, temperature at least 82 C, or
an explicit reservation yields the GPU. A reservation file may contain
`{"reserved_gpus":[7],"expires_epoch":...}`; malformed reservations fail safe
by yielding all GPUs. Three telemetry failures yield and restart the
supervisor. Workers have heartbeat recovery, and systemd retains
`Restart=always`.

Node4 canary evidence:

* Empty-node steady state reached 85.25% mean utilization across eight H200s
  at about 791 MiB total filler memory per GPU; the legacy service used about
  80 GiB per GPU.
* Under the final policy, a synthetic external process at 89% SM moved GPU0
  from idle to yielded, removed the filler worker and CUDA context entirely,
  and returned automatically to idle after exit.
* Terminating one verified worker changed its PID automatically. Terminating
  the verified supervisor incremented systemd `NRestarts` and rebuilt all
  eight workers.
* The final standalone policy's order-balanced 30-second TF32 ABBA measured
  84,221.5 mean batches active versus 84,900 stopped: 99.20% throughput, or
  0.80% observed overhead including detection handoff. This is a compute proxy,
  not a complete PPO/vLLM end-to-end A/B.
* The real sealed vLLM launch recorded `cooperative=true, stopped=false` and
  moved all eight filler workers to the small shared state during model-load
  idle periods. Filler duty remains separate from algorithm GPU telemetry.
* After sealed evaluation, the final full-yield/8-GiB-margin version was
  restarted on both nodes. Both units are active/enabled with the repo mount
  dependency loaded; their steady-state means were about 86.4% and 86.8%.
  Release audits classify eight verified filler PIDs and zero foreign PIDs on
  each node.

Status is written to
`/mnt/3fs/models/bruli/gpu-utilization-20260925/nodes/<host>/adaptive-status.json`.
The service never signals or terminates external PIDs. Its own GPU processes
remain constrained to the exact systemd cgroup and `bruli` service user.

The self-contained installer validates, deploys, enables boot start, starts the
service and checks its exact identity:

```bash
cd /mnt/3fs/xudong/gpu-utilization
sudo ./manage-adaptive-gpu-filler.sh install
```

Use the same manager for `status`, `verify`, `start`, temporary `stop`,
persistent `pause`, `resume`, `restart`, and `uninstall`. Full operating,
development, reservation, troubleshooting and expected-utilization guidance is
in `/mnt/3fs/xudong/gpu-utilization/README.md`. The effective service contains
no `coevolve` path or Python import dependency. Do not use a name-based kill to
remove GPU processes.

## Environment and official assets

`/dev/shm/harness-r1-alfworld-venv` is node3-local and intentionally isolated.
`assets/alfworld_harness_r1/{data,downloads,asset_manifest.json}` is shared 3FS.
Only these official ALFWorld release archives are used:

* `alfworld/alfworld` release `0.2.2`: `json_2.1.1_json.zip`,
  `json_2.1.1_pddl.zip`, `json_2.1.1_tw-pddl.zip`

All official 500-task directories are extracted, not the full training corpus.
The newer `0.4.2/json_2.1.3_tw-pddl.zip` was investigated but excludes three
selected official tasks, so it is **not** used as the authoritative game bundle.

CPU-only preparation deliberately omits THOR, detector weights, BUTLER training
checkpoints, and RL initialization. Archive and cohort game hashes are recorded.
`environment_freeze.txt` records installed packages. If reboot removes the shm
venv, restore it with the existing `virtualenv --copies` module, then install the recorded
packages with `TMPDIR` and pip cache under the owned asset directory.
The evaluator resolves `--agentbench-python` symlinks, which otherwise silently
escapes a symlink-based venv. A copied interpreter in this isolated environment
preserves the environment; the driver rejects symlink interpreters.
