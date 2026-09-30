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
Sealed test remains unused until both arm states are frozen.

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

Protocol v6 was frozen before formal optimization at `2 steps × batch 4` per
staged role round. H training uses at most 4,096 generated tokens: all 23 valid
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
