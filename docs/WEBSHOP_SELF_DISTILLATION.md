# WebShop: isolated successful-context self-distillation pilot

Status, 2026-11-14: **The real CPU/search and corrected native protocol gates
pass. A real fixed32-task initial-model baseline is complete; matched
train-subset pilots are running. No learned improvement or full-benchmark
result is established.** This is an opt-in Solver-only experimental
path, not a change to the ALFWorld Engineer/Solver loop, a continuation of its
checkpoints, or permission to interrupt the primary SFT job.

## Initial state and recovered environment

The local checkout and `origin/main` contained WebShop's pinned Harness-R1
implementation, strict identity tests, and setup instructions, but no Coevolve
native WebShop training workflow, WebShop run/checkpoint, installed environment,
product corpus or Lucene index. The old project-state entry also says WebShop
was not prepared. ALFWorld-trained weights are not WebShop-trained checkpoints.
The existing WebShop identity suite passed **5 tests** on CPU before changes.

A real, model-free preflight with the existing 9B model path found:

- The Harness-R1 task directory is an **adapter**, not an installed
  `web_agent_site` environment.
- Missing product, attribute and human-instruction JSON files and Lucene index.
- Missing `gym`, `bs4`, `spacy`, `pyserini`, `jnius`, the spaCy English model,
  and JDK tools in the checked interpreter/host.
- The local 9B `config.json` exists; weights were not loaded or validated.

Consequently, a before/after benchmark score was **unavailable**, not zero.
Unit-test pass counts and toy-policy optimizer tests are not WebShop successes.
The initial CPU-only implementation pass performed no GPU workload, inference
server, resource waiter, SFT edit, package installation, or service reconfiguration.

An additional CPU preparation attempt created only a disposable isolated venv
and resolved the optional environment requirements with `pip --dry-run`.
Acquiring a complete public source checkout via `git clone` required an
interactive tool approval unavailable in this session and was denied. No
alternative acquisition was attempted at that point. The user then explicitly
authorized a read-only archive route and isolated dependency recovery:

- The pinned GitHub API source tar archive downloaded successfully; its
  SHA-256 is `d16d0e6ea2d2fa8647569b9358bb56db09945d8ed58bb10f23967a76ce17824f`.
  Only the WebShop source and root licenses/notices were safely extracted.
  The nested Princeton WebShop MIT license is retained alongside the source.
- The three public files came from
  `https://huggingface.co/datasets/HongbangYuan/webshop/resolve/main/`:
  `items_shuffle_1000.json`, `items_ins_v2_1000.json`, and `items_human_ins.json`.
  They occupy about 9.4 MiB; their content hashes are preserved in the environment receipt.
- Existing local Java probes found no JDK. A default-user node probe failed
  SSH authentication; the subsequently supplied authorized helper works.
  An isolated Temurin JDK 17.0.21+9 archive was downloaded from the public
  `adoptium/temurin17-binaries` release. Verified archive SHA-256:
  `8b9b611d917e3dfd8419d3566b5949226f69b052fed7e109c1368add1dfe8711`.
- A new `runs/webshop_self_distillation_20261114/env` venv uses the existing
  Coevolve venv's site-packages read-only, with all new packages confined to
  its own overlay. Neither the primary SFT environment nor the old Coevolve
  interpreter was modified. Pins are in `requirements/webshop-environment.txt`.
  Display-only Rich and HTML-registration Selenium use compatibility pins;
  WebShop text environment code is unchanged.
- Initial failures identified missing FAISS and Selenium imports. Both are now
  explicitly declared and installed. Import **Pyserini before jnius** so its
  classpath is configured before the JVM starts. Direct `WebAgentTextEnv`
  construction avoids the released Gym 0.24 `gym.make` warning path.
- The unmodified upstream converter and a single-thread Lucene indexer indexed
  **1,000 documents with zero errors**. `environment-check` constructed the
  actual in-process Flask/SimServer, loaded **6,910 goals**, and twice repeated
  task0/reset/`search[shoes]` at goal seed233 with exactly identical manifests
  and transitions, yielding12 product/navigation clickables. This checks the
  real environment, not a mock, and starts no HTTP daemon or model.

Persistent runtime paths, all under `runs/webshop_self_distillation_20261114/`:
`reference/source/agent_system/environments/env_package/webshop/webshop/`,
`env/`, `java/jdk-17.0.21+9/`, `PREFLIGHT_READY.json`, and
`REAL_ENVIRONMENT_CHECK.json`. Runtime assets are ignored, not committed.

The user-reported historical baseline is **success45, score74**. Its original
checkpoint, denominator/task IDs, corpus, goal ordering, sampling and units have
not yet been verified. Do not equate these numbers with a newly measured subset
or automatically assert that they mean0.45/0.74. New reports explicitly use
success fraction and environment mean task score on `[0,1]`; multiplying by100
is presentation only, never a new metric or a way to make mismatched runs comparable.

### Failed first protocol gate and reference-aligned correction

The first full-budget subset attempt was stopped as an **infrastructure/protocol
gate failure**, not scored against the historical45/74 result. Its first completed
task consumed15 actions,12 invalid, with repeated512-token truncated thoughts.
A completed376-token response also contained an example action inside thinking
and a duplicated final action. Since Qwen prefills the opening `<think>` in the
prompt, a response-only parser cannot assume that opening tag is present.

The pinned public launcher explicitly sets
`+data.apply_chat_template_kwargs.enable_thinking=False` at line491. The initial
pilot incorrectly used True. This is corrected to an explicit, serialized
`enable_thinking=false` default for **both** control and improved arms. The parser
now uses only text after the final `</think>` and rejects unfinished thoughts;
the prompt also records the absolute action step and remaining budget, rather
than losing step count when history is trimmed. The15-step/512-token budgets
are unchanged. Regression tests cover the actual prefilled/duplicated action
failure, not just a synthetic well-formed `<think>` block.

The failed v1 source snapshot, complete first trajectory, and process identities
remain in ignored runtime evidence. Only PID3965966/UID1008/start-ticks564151557
was signalled after exact identity checks; SIGKILL was needed after the SIGTERM
grace, and node25GPU0 was then verified empty. No other process was signalled.
The corrected four-action/two-task protocol gate must pass before restarting
the fixed32 comparison with one common frozen corrected source.

The second protocol probe then exposed a separate action-affordance defect:
showing the raw `{"has_search_bar": true, "clickables": ["search"]}` object led
the model to repeatedly choose `click[search]`. The underlying text environment
explicitly treats that UI button as a no-op; the wrapper also undercounted these
as valid clicks. The pinned reference's `env_manager.py` instead formats
executable `search[<your query>]`/`click[label]` strings. The pilot now does so,
omits the known non-executable search button, explains direct query submission,
and correctly marks `click[search]` invalid without altering environment reward.
Both failed probes and their original validity flags remain unchanged as raw
evidence; the second probe is not presented as a successful format gate.

Tasks0/1 are consequently **protocol-debug tasks**, not clean test tasks.
Before any successful comparison or training, the fixed32 comparison manifest
was moved to IDs32–63, never inspected or used for these repairs. The original
aborted0–31 proposal is retained in its frozen snapshot; no old/new scores are
mixed across these manifests. Train IDs remain500–503. This exclusion avoids
claiming a clean heldout result on tasks used to debug the interface.

The third native protocol probe, with both fixes, completed eight real actions,
zero invalid actions and no512-token truncations, using510 generated tokens.
Its actual actions searched, paginated, opened a product, selected an option,
and purchased. The four-action cap makes its0/2 full successes and0.42857
mean partial score **diagnostic only**, not a benchmark comparison.

Separately, a native node26 learner smoke completed two one-action train-only
episodes, real backpropagation, AdamW step and LoRA adapter/tokenizer save:
199.93seconds including setup,30.996GB peak allocation,51,265,024 trainable
parameters. Both returns were0, so loss/gradient and teacher tokens were0.
That validates the native training/save path, **not policy learning or improvement**.

## Public reference and license

Reviewed [aim-uofa/Self-Distillation-AgentRL](
https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86)
at commit `35a44f5c5647b1785ad23ddf0ffbd196b695bf86`:

- [Apache-2.0 license](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/LICENSE)
  was checked before implementation. This pilot is original code implementing
  algorithmic ideas; no upstream implementation or benchmark assets are copied
  into tracked files.
- [`run_skill_method_1gpu.sh`](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/launchers/run_skill_method_1gpu.sh):
  WebShop uses 15 environment steps, 512 response tokens, 4096 prompt/context
  scale, group size 8, validation sampling temperature 0.4, and KL coefficient 0.01.
- [`envs.py`](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/agent_system/environments/env_package/webshop/envs.py):
  train task IDs start at 500; evaluation uses IDs below 500. Full success
  requires termination with task score exactly 1.0; partial task score is not
  success. The released wrapper scales binary success reward to 10.
- [`rlsd_utils.py`](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/verl/trainer/ppo/rlsd_utils.py),
  [`sdar_utils.py`](https://github.com/aim-uofa/Self-Distillation-AgentRL/blob/35a44f5c5647b1785ad23ddf0ffbd196b695bf86/verl/trainer/ppo/sdar_utils.py)
  and OCSD/OVCSD helpers distinguish student-sampled actions, privileged teacher
  context, detached teacher signals, response masks, and state alignment.

The reference itself warns that self-distillation is setting-dependent and
does not consistently beat GRPO. Its skill-conditioned RLSD/SDAR and
observation-calibrated methods are **not identical** to this successful-peer
hindsight variant. This is a hypothesis worth testing against GRPO, not an
implementation-equivalence or performance claim.

## Mechanism and safeguards

New code lives only in `whale_local/webshop/`; existing training imports and
defaults are unchanged.

1. **Real policy rollouts.** For one fixed training task, collect eight
   independent student-policy trajectories with exact generated token IDs and
   actual normalized sampling log probabilities. Training samples at temperature
   1, top-p 1, with no top-k truncation. No teacher action is substituted for a
   policy action. Invalid responses consume an environment step.
2. **Success, not failure imitation.** The original strict pilot selects the shortest *other* completed
   successful trajectory with no invalid actions from the same current-policy/task-identity group.
   Never use evaluation episodes, partial purchases, cross-task trajectories,
   or the student's own success as its teacher example. No successful peer
   means no distillation signal.
   The separately preregistered filtered-path pilot additionally permits a
   real successful episode with invalid turns **only when every omitted
   invalid turn has exactly equal before/after observations**. It teaches only
   the remaining real valid actions and labels the omission explicitly.
   State-changing invalid turns, failed episodes and evaluation episodes are
   rejected. All original student turns/tokens/rewards/budgets remain intact.
3. **Feedback is teacher-only.** The teacher receives the successful
   observation/action/result path and terminal success feedback in addition
   to the student's current prompt. The student prompt is unchanged. If needed,
   compact the demonstration to its complete action path plus success receipt;
   if that still overflows, disable distillation for that turn and count it.
   Never truncate away the current shopping instruction or observation.
4. **On-policy objective.** Teacher-force those *same student-sampled response
   tokens* under the same frozen policy with privileged context. All teacher
   scores are detached and computed before the optimizer step. For each token:

   ```
   A_total = A_group
             + lambda * eligible * clip(log q_teacher - log pi_behavior, -5, 5)
   ratio = exp(log pi_student - log pi_behavior)
   L_policy = -min(ratio * A_total, clip(ratio, 0.8, 1.2) * A_total)
   ```

   `A_group` is the group-standardized binary terminal success return.
   The detached teacher log-ratio is an on-policy sampled reverse-KL
   policy-gradient signal, not supervised cross-entropy on arbitrary teacher
   outputs and not the full-vocabulary KL itself. Clipping is a deliberate
   stabilization/bias. Both arms also use the same coefficient-0.01 sampled
   nonnegative KL estimator against the initial base policy, scored with LoRA
   disabled. No second 9B reference model is allocated.
5. **Masks/weighting.** Only generated response tokens participate in loss.
   Prompt, observation and padding targets never enter it. Keep raw EOS/token
   alignment rather than decoding/re-tokenizing training responses. Average
   over tokens within the whole episode, then over episodes; long trajectories
   do not receive extra mass just because they have more action turns.
6. **Honest retesting.** Preserve goal seed 233, exact per-task goal,
   instruction, product-price and asset/source hashes across rollout seeds and
   both arms. Reject mismatched seeds, goals, corpora, source, task IDs, budgets,
   duplicate or missing results. Recompute success counts from per-episode
   terminal score, not a reported aggregate. Report partial task score
   separately, as well as invalid actions, token counts, seed mean/std, rescued
   and regressed pairs. Rollout seeds are not independent training seeds.
   The environment adapter also clears only its own previous task session before
   each reset, since upstream reset otherwise retains old `done`/`reward` keys.
   Initial model weights/tokenizer and pilot source are content-hashed;
   changed base weights at the same local path do not silently pass as the
   same initialization. Full-weight hash reads happen only in the explicitly
   leased real-model runtime, not the CPU preflight.

The teacher is the pre-update current policy, not a larger external model.
Base model weights stay frozen; both GRPO and `success_sd` train the same LoRA
architecture. This Solver-only pilot does not test the Engineer–Solver
co-evolution mechanism.

## CPU validation and preflight

Use an existing isolated interpreter with torch/pytest, or separately install
`requirements/webshop-pilot.txt`. Do **not** install anything into primary SFT's
environment. All scratch/cache paths below are inside the project.
The optional CPU environment pins are in
`requirements/webshop-environment.txt`; actual Python3.12/JDK17/JNI/Lucene
imports and a real deterministic search transition have now been validated
in the isolated overlay.

```bash
mkdir -p .cache
CUDA_VISIBLE_DEVICES="" TMPDIR="$PWD/.cache" \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m pytest -q -p no:cacheprovider --basetemp=.cache/pytest-webshop \
  tests/test_webshop_self_distillation.py

CUDA_VISIBLE_DEVICES="" python -m whale_local.webshop preflight \
  --env-root /absolute/path/to/installed/webshop \
  --model /absolute/path/to/local/Qwen3.5-9B \
  --output runs/webshop/preflight.json
```

Preflight exits 2 and preserves a JSON blocker list when prerequisites are
missing. It does not import/load a model, initialize CUDA, install anything,
download data, construct a search engine, or claim runtime validation.
It only checks files/dependencies/JDK presence; an actual environment reset and
model gate are still required. Output paths must be new to protect old evidence.

Validation on 2026-11-14: **78 pilot tests passed** on CPU, including a real
tiny Hugging Face transformer generation/log-probability/LoRA-gradient check
and the full train → adapter save → evaluate → paired-compare CLI round trip
with an explicitly labelled CPU toy environment. Five upstream deprecation
warnings were emitted. The existing pinned Harness-R1 WebShop identity suite
also passed **5 tests**. Neither test run is a benchmark evaluation.
The combined suite passed83 tests in46.81seconds with5 upstream deprecation warnings.

## Measured fixed-subset results

The initial model is the existing local BF16 `Qwen3.5-9B`, not a primary-SFT
QAT checkpoint and not the published ALFWorld97 Actor. Exact model/tokenizer
content hashes, the frozen source manifest, index/data hashes, per-task goals,
sampling and numeric process identities are in each runtime report.

All rows below use **the same registered test IDs32–63, seed0,
goal seed233, 1k corpus,15actions,512response tokens,4096total context, and
temperature0.4**. Protocol-debug IDs0/1 are excluded. Values are fractions on
`[0,1]` internally; percentage/100-point displays are explicitly labelled.

| Condition | Success | Mean task score | Evidence |
|---|---:|---:|---|
| Initial model | **15/32 =46.875%** | **0.55771645/1 =55.771645/100** | Real completed model/environment evaluation |
| Matched GRPO control | Pending | Pending | Train-subset pilot only; no result claimed |
| Successful-context arm | Pending | Pending | Train-subset pilot only; no result claimed |

Initial-model Wilson95% success interval: **30.8694–63.5505%**.
Task-bootstrap95% mean-score interval: **40.5850–71.4881/100**.
The run generated15,931 action/response tokens, recorded20 invalid actions,
took682.79seconds including setup, and peaked at21.01GB allocated CUDA memory.
Its GPU was verified empty after exit. These are32 fixed tasks and one
rollout/training setting, not a full500-task benchmark or multi-seed result.
In particular, this row is **not comparable to the user-reported45/74**
without the original checkpoint, manifest and metric-scale evidence.

The two matched pilots each use train IDs500–503,8rollouts/task and4optimizer
updates (32episodes), with identical initial base/LoRA layouts and seeds.
Only the successful-context coefficient/teacher scoring differs. Their native
training receipts must establish actual teacher-token and gradient signal;
the earlier zero-gradient one-action learner smoke does not count as learning.

The optional model-free evidence tool
`python3 scripts/reproduce_webshop_subset.py` recomputes stored compact per-task
results and paired confidence gates once the final bundle is emitted. It never
loads a model or opens a GPU/network connection; offline recomputation is not
a new model evaluation. Payload hashes detect changed evidence.

For the real CPU check with the prepared runtime (no GPU lease is needed):

```bash
BASE=/absolute/path/to/coevolve/runs/webshop_self_distillation_20261114
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  JAVA_HOME="$BASE/java/jdk-17.0.21+9" \
  PATH="$BASE/java/jdk-17.0.21+9/bin:$PATH" \
  JAVA_TOOL_OPTIONS="-Xmx2g -XX:ActiveProcessorCount=2 -XX:-UsePerfData -Djava.io.tmpdir=$BASE/.cache" \
  TMPDIR="$BASE/.cache" \
  "$BASE/env/bin/python" -m whale_local.webshop environment-check \
  --config configs/webshop_success_sd_pilot.json \
  --env-root "$BASE/reference/source/agent_system/environments/env_package/webshop/webshop" \
  --task-id 0 --action "search[shoes]" \
  --output runs/webshop/new_environment_check.json
```

## Preparing the environment, without touching primary SFT

Follow the pinned reference's `launchers/prepare_webshop.sh small` in a
**separate** source checkout and virtual environment, or prepare the original
WebShop environment yourself. It requires separately distributed product data,
the spaCy model, a working JDK/JNI stack and a built Lucene 1k index. Keep the
upstream license with that source checkout. Do not copy only Harness-R1's task
adapter and assume it contains the environment.

Pass the directory containing `web_agent_site/`, `data/`, and `search_engine/`
as `--env-root`. The runtime adapter loads the explicit 1000-product/attribute
files and fingerprints the actual index and environment Python files.
The 1k corpus is a pilot, not the full WebShop catalog. Dataset-size,
goal-order and reference-worker seeding differences must not be hidden.

## Real-model execution: blocked until an exact resource lease

No launcher requests GPUs, creates a global vLLM server, waits for idle cards,
or terminates another process. A user/coordinator must first release **one
specific GPU on one specific host**, after the primary SFT no longer needs it.
Every train/evaluate command and each model forward/generation checks an
explicit lease with these fields:

```json
{
  "purpose": "webshop-side-project",
  "hostname": "EXACT_HOSTNAME",
  "cuda_visible_devices": "EXACT_SINGLE_DEVICE",
  "expires_at": "EXPLICIT_FUTURE_UTC_TIME",
  "primary_sft_released": true,
  "modes": ["train", "evaluate"],
  "max_episodes": 32
}
```

This is a schema example, **not an authorization or usable lease**. Expired,
wrong-host, multi-GPU, unapproved-mode or over-budget leases fail closed before
model initialization. An already-running bounded operation can finish after
expiry; subsequent operations are rejected. Set a lease expiry earlier than
the actual resource hand-back deadline and allow shutdown margin.

The checked-in diagnostic profile uses:

| Item | Fixed value |
|---|---|
| Train IDs | 500–503, one task group per update |
| Training | 4 updates × 8 trajectories = 32 episodes per arm |
| Evaluation | IDs 0–7 × rollout seeds 0,1 = 16 episodes per arm |
| Per-episode cap | 15 actions × 512 generated tokens |
| Context cap | 4096 tokens including response |
| Shared goal seed / training seed | 233 / 1000043 |
| Arms | GRPO vs GRPO + successful-context signal, lambda 0.5 |
| Optimizer | AdamW LoRA, LR 1e-5, one full-group update, grad clip 1 |

This is an **eight-task diagnostic**, not the reference's 128-task/eight-seed
evaluation, 16-task training batches, full-finetuning recipe or paper result.
Up to 96 total episodes (two trainings plus two evaluations), at most 737,280
generated tokens, are permitted by this paired profile. Distillation adds
teacher-scoring compute; equal environment/token caps do not mean equal FLOPs
or wall time. A separate raw-base evaluation adds another 16 episodes.
No real-model throughput/runtime estimate has been measured.

After an approved exact lease and an environment/model smoke gate:

```bash
# On the explicitly leased host, make only that one GPU visible.
python -m whale_local.webshop train \
  --config configs/webshop_success_sd_pilot.json \
  --env-root "$WEBSHOP_ROOT" --model "$LOCAL_MODEL" --lease "$LEASE" \
  --method grpo --output runs/webshop/grpo
python -m whale_local.webshop train \
  --config configs/webshop_success_sd_pilot.json \
  --env-root "$WEBSHOP_ROOT" --model "$LOCAL_MODEL" --lease "$LEASE" \
  --method success_sd --output runs/webshop/success_sd
python -m whale_local.webshop evaluate \
  --config configs/webshop_success_sd_pilot.json \
  --env-root "$WEBSHOP_ROOT" --model "$LOCAL_MODEL" --lease "$LEASE" \
  --adapter runs/webshop/grpo/adapter --output runs/webshop/grpo_eval
python -m whale_local.webshop evaluate \
  --config configs/webshop_success_sd_pilot.json \
  --env-root "$WEBSHOP_ROOT" --model "$LOCAL_MODEL" --lease "$LEASE" \
  --adapter runs/webshop/success_sd/adapter --output runs/webshop/success_sd_eval
CUDA_VISIBLE_DEVICES="" python -m whale_local.webshop compare \
  --baseline runs/webshop/grpo_eval/REPORT.json \
  --improved runs/webshop/success_sd_eval/REPORT.json \
  --output runs/webshop/PAIRED_COMPARISON.json
```

The adapter, per-episode trajectories, training metrics, source/data hashes and
final report remain local under ignored `runs/`. The inference adapter is **not**
a resumable optimizer/RNG checkpoint; failures require a new bounded run, not
an automatic restart. Model loads are local-only with remote code disabled.
Inspect success/gradient/teacher-token metrics before escalating a budget.
If there are no successful groups, this method cannot manufacture a valid
success teacher; report that outcome instead of repeatedly relaunching.

## Required next result

CPU environment/search readiness is satisfied. The parent explicitly released
nodes25/26 for WebShop through2026-11-15 00:00UTC; nodes22/24 remain exclusively
for primary SFT and are not WebShop resources. Authorized probes found both
WebShop nodes idle. Runs use numeric UID/GID1008:1008, never a username-derived
chown, a separate cache/home, `PROCESS.json` PID/start-tick identity, and a hard
per-command timeout in addition to the model-forward lease checks.

The staged plan is:

1. `configs/webshop_model_gate.json`: two fixed tasks, one action each, at the
   actual512-token/4096-context settings. One GPU, hard600-second wall limit.
   This is a format/memory/throughput check, **not task success evaluation**.
   The first native gate on node25GPU0 passed: both responses executed valid
   search actions, zero invalid actions,330 generated tokens in11.33seconds
   (~29tokens/s),160.74seconds including setup,19.68GB peak CUDA allocation.
   The one-action cap makes its0/2 terminal successes uninformative about task
   success. Exact UID1008/start-tick identity was recorded; GPU0 was verified
   empty after exit. The coordinator's helper returns stdout as a string,
   not a CompletedProcess; an initial post-run status-formatting error did not
   invalidate the completed per-task/report files or leave a GPU worker.
   The subsequent first full-budget run exposed the thinking/parser defect
   described above; use `configs/webshop_protocol_gate.json` (four actions on
   the same two IDs) with the corrected reference-aligned mode before proceeding.
2. `configs/webshop_subset_gate.json`: fixed evaluation IDs32–63, seed0,
   15actions/512tokens/goal seed233; train IDs500–503, four groups×eight
   trajectories. Preserve an initial-model baseline, then match GRPO and
   `success_sd` initial weights, LoRA layout, training budget and evaluation.
   Use separate immutable source snapshots and new output directories.
3. Before any full-data expansion, require at least32 fixed evaluation tasks,
   matched training controls, an actually nonzero teacher signal and gradient,
   paired success-difference task-bootstrap95% CI strictly above0,
   score-difference CI not below0, and exact two-sided McNemar p≤0.025 for the
   single-seed paired binary outcomes. This is a conservative screening gate,
   not a guarantee of multi-training-seed robustness or a permission to
   automatically start another experiment. With multiple rollout seeds,
   cluster bootstrap by task and omit the naive McNemar p-value.

The family comprises two prespecified subset candidates: original strict
successful targets (test IDs32–63) and verified-no-op-filtered targets
(`configs/webshop_filtered_subset_gate.json`, untouched test IDs128–159).
Both use the same4 training tasks/32episodes and matched GRPO controls.
The second candidate was motivated solely by train-task501: its only real
success had two invalid turns with unchanged observations, causing the strict
pilot to discard every teacher target for that group. The new teacher summary
omits only these proven no-ops; it does not invent a successful student action
or remove student loss tokens.

Both candidates, fresh second test IDs and the Bonferroni threshold0.05/2=0.025
were declared **before either trained policy's heldout evaluation**. The first
raw initial-model baseline was already visible; it is not used as a promotion
comparison or a source of teacher targets. Do not rank candidate point scores
across different test subsets: only their respective paired-control effects
are meaningful. The second pilot is bounded to32training episodes/arm on
node25/26GPU1, not a full-data expansion.

Report partial task score separately from success and include confidence
intervals. A point improvement on eight questions, a mocked environment, an
all-failure/no-teacher update, or a differently seeded/corpused45/74 run cannot
pass this gate. Until an actual paired result exists, this remains a tested
experimental mechanism and retest pipeline, **not a demonstrated benchmark win**.
