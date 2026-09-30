# Running the local WHALE extension

Start with [the current project state and agent handoff](PROJECT_STATE.md).
It separates historical algorithms, measured gains, invalid attempts and
current experiment gates. Official ALFWorld checkpoint reproduction is documented
in [its CPU runner](../whale_local/harness_r1_alfworld/README.md).

## Current Harness-R1 fast pilot

The current frozen-solver Engineer experiment is documented in
[the fast-4B README](../runs/searchqa_harness_r1_20260930/fast4b/README.md).
It records real infra timings, native 4B versus 9B distinctions, direct pinned
Harness-R1 objective reuse, backend adaptations, datasets, checkpoint receipts
and live pipeline status. The
[adapter method index](../whale_local/harness_r1_searchqa/README.md) links the
training, evaluation and performance implementation. These are preliminary
small-scale experiments, not the historical publication entry point below.

## What this release contains

The repository contains research source, a pinned upstream submodule, aggregate results and a new configuration entry point. It does not include model checkpoints, optimizer states, rollout records, benchmark data or prebuilt environments. Historical experiments ran on the original server. The publication entry point is checked separately; exporting the code is not another full GPU experiment.

The two roles use independent checkpoints. Native local checkpoints include Hugging Face model/tokenizer files plus `training_state.json`, `optimizer.pt`, and `rng.pt`. The runtime rehashes model tensors and verifies optimizer continuity when loading. A generic Hugging Face model directory alone is not a continuation checkpoint accepted by `prepare_run.py`.

## Environment

Use Linux, a compatible NVIDIA driver and Python 3.12. The recorded training setup used **modified 48 GB RTX 4090 cards**, FP32 parameters/Adam state and BF16 autocast. It is not evidence that full-parameter training fits a stock 24 GB 4090. The new example reserves one local training GPU and one or more separate sampling GPUs; the two trainable roles use the training GPU sequentially.

Install `bubblewrap` through the system package manager; unprivileged user namespaces must work. Keep coordinator and worker environments separate:

```bash
git submodule update --init --recursive
python3.12 -m venv .venv-whale-portable
.venv-whale-portable/bin/python -m pip install -r requirements/worker.lock
python3.12 -m venv .venv-whale-search
.venv-whale-search/bin/python -m pip install -r requirements/coordinator.lock

export PYTHONPATH="$PWD/whale_local:$PWD/WHALE/domains/math_reasoning:$PWD/WHALE/domains/chess_puzzles"
.venv-whale-search/bin/python scripts/check_install.py --role coordinator
.venv-whale-portable/bin/python scripts/check_install.py --role worker --sandbox
```

The `.txt` files record direct package versions from the working environments; `.lock` files resolve their transitive dependencies for Python 3.12 on Linux. These are not cross-platform locks or a complete snapshot of the historical overlay environments. The worker does not need vLLM or a separately installed verl: Python imports the pinned math-domain actor directly from WHALE. Math must precede chess on `PYTHONPATH`. `transformers==5.3.0` is intentional: the scoped Qwen3.5 padding compatibility fix checks that implementation before applying.

`check_install.py` checks the pinned upstream commit, imports, a strict correct/incorrect reward pair, the editor objective, and optionally a real isolated Python call. It does not allocate model weights or launch training.

## Data

Obtain the official `EleutherAI/hendrycks_math`, `number_theory` train/test parquet sources at revision `21a5633873b6a120296cce3e2df9d5550074f4a3`. Put them in `assets/math_source/train.parquet` and `assets/math_source/test.parquet`, then run:

```bash
.venv-whale-search/bin/python scripts/prepare_math_data.py \
  --source assets/math_source --output assets/math
```

The script reproduces the question selection rule: canonical integer boxed answers at Levels 2–4, sorted by problem SHA256. Official train supplies 128 training and the next48 development questions. Eligible official test questions exclude every official-train problem; indices128:256 supply the128 evaluation questions used in the published editor pilot. The first128 were reserved for the separate gate experiment.

The manifest includes file digests. Different parquet library versions can change file bytes, so question IDs/order—not parquet byte identity alone—define the historical cohort. The script records the expected revision but does not assert that user-supplied source files came from that revision.

## Initial models

For continuation of the published experiment, obtain both native checkpoints separately from the experiment owner:

| Role | Historical initialization |
|---|---|
| Solver W | `whale_math_mean8_native_20260915/round_01/checkpoint`: 512 prior training rollouts,120 RSFT updates |
| Editor E | `whale_dual2b_focused_20260916/editor_continuation/checkpoint`:132 prior SFT updates, zero prior GRPO updates |

Place them at `assets/solver/checkpoint` and `assets/editor/checkpoint`, or change the explicit paths in your config. They are intentionally excluded from Git. This repository does not claim to regenerate these exact warm starts with a single bootstrap command.

For a **new initialization**, the synthetic editor curriculum and SFT trainer are provided:

```bash
python3 whale_local/harness_bootstrap.py --out assets/editor_curriculum
mkdir -p runs/editor_sft
CUDA_VISIBLE_DEVICES=0 WHALE_PARAMETER_DTYPE=float32 \
WHALE_RUN_ROOT="$PWD/runs/editor_sft" \
WHALE_MODEL_ROOT="$PWD/assets/Qwen3.5-2B" \
.venv-whale-portable/bin/python whale_local/harness_learner.py train \
  --data assets/editor_curriculum/train.jsonl --epochs 2 --learning-rate 1e-5
```

The base model directory must already be downloaded. The generated curriculum contains synthetic interface demonstrations, not benchmark solutions or measured successful improvements. This command is an example of initializing E; it is not the full132-step historical curriculum. The solver's own RSFT path is `math_pipeline.py` / `alternating_solver_train.py`; historical preparation scripts require their corresponding manifests and run assets. For the supported new pilot entry point, supply a complete initialized solver checkpoint.

## Prepare and execute

```bash
cp examples/pilot.json examples/local.json
# Edit paths and GPU IDs in examples/local.json; paths are relative to repository root.
.venv-whale-search/bin/python scripts/prepare_run.py \
  --spec examples/local.json --run runs/pilot --check-only
.venv-whale-search/bin/python scripts/prepare_run.py \
  --spec examples/local.json --run runs/pilot
.venv-whale-search/bin/python -u whale_local/three_arm_hwh.py execute --run runs/pilot
```

Paths inside a data manifest are relative to that manifest. The check verifies required files, recorded checkpoint identities, data digests, question identities, nonoverlapping128/48/128 splits and separate training/sampling GPU assignments. Tensor hashes are checked by the runtime on model load. It does not check GPU availability; choose currently free devices.

The default arms are `fixed_editor` and `trained_editor`. Both use eight pre-update candidates, four matched post-update candidates,32 solver training trajectories, and H→W→H. Training the editor makes up to four GRPO steps per phase. To run only the learned editor set `arms` to `["trained_editor"]`; to include the external baseline add `"external"`.

The original `three_arm_hwh.py prepare` function preserves the historical hardcoded setup. **Use `scripts/prepare_run.py` for new configurations.** The release routes editor generation and training to the configured training GPU, rather than historical GPU6.

`STATUS.json` reports progress. On success, `COMPLETE.json` contains all stage model/harness paths and observed scores; `FROZEN.json` records the states before test evaluation. Checkpoint files alone are not a completion signal. Resume only after confirming the old workers have stopped; the pool deliberately refuses to silently start over a live or uncleared pool directory.

## External proposer

When the external arm is enabled, the coordinator starts a loopback FastAPI queue and waits for responses. A separate actual model client or active human/agent must read:

```text
runs/pilot/proposer_jobs/<job_id>/request.json
```

and atomically write `response.json` in the same directory:

```json
{
  "proposer": "your-model-or-agent",
  "candidates": [
    {"id": "reward_0_00", "group": "reward_0", "source": "<complete CandidateEnv Python module>", "hypothesis": "<reason>"}
  ]
}
```

Return exactly the IDs/groups and number of slots specified by that request (eight before, four after). Responses must obey the supplied code contract. The queue does not automatically call a commercial API or expose this Codex conversation as a model service. The internal-only configuration bypasses this queue.

## Optional second node

`pool_gpus` accepts local IDs and `node1:0`, `node1:1`, etc. These are independent **frozen sampling replicas**, not multi-node synchronized gradient training. Workers need the same repository, dependencies and absolute path layout; the transport copies checkpoints and returns run evidence using SSH/rsync. Configure before launch:

```bash
export COEVOLVE_REMOTE_HOST=user@your-worker-host
export COEVOLVE_SSH_CONTROL=/tmp/coevolve-ssh/control
mkdir -p /tmp/coevolve-ssh
ssh -MNf -S "$COEVOLVE_SSH_CONTROL" "$COEVOLVE_REMOTE_HOST"
ssh -MNf -S /tmp/coevolve-ssh/management "$COEVOLVE_REMOTE_HOST"
```

Use an already configured SSH identity; no login secrets belong in experiment configs. The management socket is the file named `management` beside the control socket. The shipped node adapter supports remote GPU indices0–9. Ports20000+slot and21000+slot must be free locally and remotely. Training remains on the selected local GPU.
