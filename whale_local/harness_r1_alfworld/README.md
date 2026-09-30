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
