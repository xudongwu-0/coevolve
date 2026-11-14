# WebShop: isolated successful-context self-distillation pilot

Status, 2026-11-14: **CPU implementation/contract validation only; no new WebShop
model score or training result.** This is an opt-in Solver-only experimental
path, not a change to the ALFWorld Engineer/Solver loop, a continuation of its
checkpoints, or permission to interrupt the primary SFT job.

## What was actually available

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

Consequently, a before/after benchmark score is **unavailable**, not zero.
Unit-test pass counts and toy-policy optimizer tests are not WebShop successes.
No GPU workload, inference server, resource waiter, SFT edit, package
installation, or service reconfiguration was performed.

An additional CPU preparation attempt created only a disposable isolated venv
and resolved the optional environment requirements with `pip --dry-run`.
Acquiring a complete public source checkout via `git clone` required an
interactive tool approval unavailable in this session and was denied. No
alternative acquisition was attempted to bypass that denial; no JDK, product
data, or WebShop dependencies were installed. The source/JDK/data setup remains
an explicit prerequisite, not a completed preparation.

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
2. **Success, not failure imitation.** Select the shortest *other* completed
   successful trajectory with no invalid actions from the same current-policy/task-identity group.
   Never use evaluation episodes, partial purchases, cross-task trajectories,
   or the student's own success as its teacher example. No successful peer
   means no distillation signal.
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
`requirements/webshop-environment.txt`; dependency resolution was checked, but
actual installation and Python/JNI/Lucene compatibility were not validated.

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

Validation on 2026-11-14: **64 pilot tests passed** on CPU, including a real
tiny Hugging Face transformer generation/log-probability/LoRA-gradient check
and the full train → adapter save → evaluate → paired-compare CLI round trip
with an explicitly labelled CPU toy environment. Five upstream deprecation
warnings were emitted. The existing pinned Harness-R1 WebShop identity suite
also passed **5 tests**. Neither test run is a benchmark evaluation.

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

First satisfy CPU environment/search readiness, then obtain a bounded exact
GPU lease without displacing primary SFT. Run the same initial-model GRPO and
successful-context arms with identical fixed task/seed/budget manifests.
Publish their actual per-task comparison, including regressions and failures.
Until then, the improvement is a tested experimental mechanism and retest
pipeline, **not a demonstrated benchmark win**.
