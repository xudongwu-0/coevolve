# ALFWorld reproducibility and artifact guide

This guide covers offline result verification, experiment provenance, training-source restoration, and checkpoint access. Historical files use the name **MA-SAO** for the joint Engineer/Solver implementation described as **JoMoHa** in the [paper](https://arxiv.org/abs/2611.11499).

For WebShop and DBBench runtimes, builders, and evaluation entry points, see the [benchmark guide](BENCHMARK_IMPLEMENTATIONS.md).

## Offline verification

From the repository root, using Python 3.10 or later:

```bash
python3 -B scripts/reproduce_ma_progress.py --best97
```

No dependency installation, submodule initialization, API access, model download, or GPU is required. The verifier:

- Checks every declared file against the [payload manifest](../reproducibility/ma_sao_20261012/MANIFEST.json).
- Reconstructs the joint/joint/Solver-only training inputs and compares them with the stored ledger.
- Audits task IDs and goal-group overlap across training, selection, and evaluation.
- Recomputes Full96 and Full144 scores from the original per-task reports.
- Checks that harness selection preceded the endpoint evaluation, with the recorded Solver identity and evaluation budget.
- Distinguishes recorded training metrics from complete paired recovery metadata.
- Verifies the seven frozen executing-source snapshots against their own manifests.
- Verifies the best97 harness and recomputes its 194/200 result, matching summary outcomes to the original per-task report.

The JSON output should report `status: "passed"`, 4,225 payload files, `training_allowed: false`, and `gpu_or_api_used: false`. These flags describe an offline evidence check, not a new experiment.

The `best97` object reports `selected.successes: 194`, `selected.accuracy: 0.97`, and `no_patch.successes: 185`. It also includes the matching harness path and SHA-256 digests for the harness and summary reports. Omit `--best97` to run the original Full96/Full144 archive audit on its own.

### Expected endpoint scores

| Archived endpoint | Selected harness + Solver | Same Solver without a harness |
|---|---:|---:|
| Full96 | 180/200 (90.0%) | 178/200 (89.0%) |
| Full144, fixed primary endpoint | 186/200 (93.0%) | 179/200 (89.5%) |

For `without_peer`, the verifier should find 76 metric rows per role, but the latest complete paired recovery metadata is at round **72**, not 76. The archive does not establish a completed endpoint comparison for that ablation. Recovery metadata verifies saved receipts; it does not mean model tensors are present in Git.

### Extract an evidence copy

To reconstruct the original relative directory layout in a new directory:

```bash
python3 -B scripts/reproduce_ma_progress.py --materialize /tmp/coevolve-evidence-copy
```

The destination must not already exist and must be outside the bundle. This copies only declared evidence, verifies the copied bytes, preserves recorded absolute paths as provenance, and does not execute launchers or copy model tensors.

The [frozen payload](../reproducibility/ma_sao_20261012/payload/) must not be edited. Adaptations belong in separate working configurations, not in the archived scientific records.

## Interpreting the results

### Data splits

The archive preserves the original 500-task catalogue and its ordering; task IDs must not be reassigned by sorting directories.

| Partition | Scope | Interpretation |
|---|---|---|
| Training | Task IDs 0-239, 240 tasks | No task-ID overlap with the 200-task evaluation |
| Source inputs | 768 rows, fixed seed 1000042 | Prompts and historical evidence are preserved |
| Scheduled inputs | 1,152 rows, 144 rounds x 8 episodes | Repeated inputs are not replayed generated trajectories |
| Harness selection | 48 explicit task IDs | All 48 are included in `main200` |
| `main200` | 200 tasks, one sample each | Reused development evaluation, not an untouched test set |
| `strict197` | `main200` excluding IDs 291, 414, 467 | Removes three goal-group overlaps with legacy training |
| Remaining 152 tasks | `main200` minus the selection tasks | Post-hoc descriptive subset, not a new primary test |

Training/evaluation task-ID separation does **not** eliminate selection/evaluation overlap. Neither `strict197` nor the remaining 152 tasks should be presented as a newly established independent test protocol. Upstream pretraining exposure cannot be independently ruled out.

These records describe a single-seed study on reused development data; they do not establish robust multi-seed superiority or a causal comparison with the historical 97% run.

### The 93% archive and the historical 97% run

The offline verifier's **Full144** result is 186/200 (93.0%). It uses a different initialization and run from the historical 194/200 (97.0%) result.

The historical 97% record pairs a selected harness with a Solver. The same Solver without that harness scored 185/200 (92.5%). It is therefore not a weight-only 97% result, and it was selected on reused development data rather than an untouched generalization test.

The historical [selected harness](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/patches/95-proposal07.json) is preserved in Git. The `--best97` verifier recomputes the archived score without running the model.

## Archived training configuration

The Full144 study used:

- Separate Engineer/Solver actors and separate critics, with a shared terminal task reward.
- H0 Engineer, ordinary SAO W1 Solver, and pre-MA critics; not the historical 97% run's trained critics.
- Bounded partner-trajectory weights, critic collaboration context, and per-role ValueNorm.
- Joint / joint / Solver-only scheduling, with critic-only warmup for each role's first three active rounds.
- Seed 1000043, batch size 8, 144 rounds, 32,768 context tokens, and a 4,096 action-token limit.
- Actor learning rate 5e-7, critic learning rate 2.5e-6, two critic updates per active round, and reference KL coefficient 0.02.
- Paired native saves every 12 rounds, with a shared commit record.

This accounts for 1,152 shared environment episodes and 2,304 generated role trajectories. It does not imply 144 actor updates for each role.

The original Full run used two eight-GPU nodes. A later partial ablation changed the rollout layout while retaining four learner ranks per role; those layouts do not guarantee token-identical sampling. These are historical configurations, not a tested minimum hardware requirement.

For implementation entry points, see the [README code overview](../README.md#code-overview). The archive's [state index](../reproducibility/ma_sao_20261012/STATE.json) identifies the exact study and source snapshots; current working-tree code must not be presented as the code that executed an older run.

## Restoring the training source

The following steps restore source dependencies, **not** a ready-to-run training environment.

### 1. Initialize pinned submodules

```bash
git submodule update --init --recursive
```

The exact revisions are recorded in [UPSTREAMS.json](../reproducibility/ma_sao_20261012/UPSTREAMS.json). Third-party copyright and license notices remain with their respective components.

### 2. Restore the patched AReaL tree

AReaL is not a top-level submodule. The historical runtime includes local modifications, preserved as a [complete working-tree patch](../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch) with [baseline metadata and file fingerprints](../reproducibility/ma_sao_20261012/upstream/AReaL.json).

From the repository root, with `third_party/sao/vendor/AReaL` **not yet present**:

```bash
git clone https://github.com/inclusionAI/AReaL.git third_party/sao/vendor/AReaL
git -C third_party/sao/vendor/AReaL checkout --detach 3cf0dfbd2b0fbeabd6977184980e189d1567747a
git -C third_party/sao/vendor/AReaL apply --check \
  ../../../../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch
git -C third_party/sao/vendor/AReaL apply \
  ../../../../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch
```

Do not run the SAO bootstrap's component-patch sequence before applying this complete patch. If the vendor directory already exists, inspect its revision and local changes rather than overwriting or resetting it.

### 3. Prepare the runtime and external assets

The recorded 9B environment is described in [ENVIRONMENT.json](../reproducibility/ma_sao_20261012/ENVIRONMENT.json) and the worker environment records in the payload:

- Python 3.12, PyTorch 2.10.0+cu129, vLLM 0.19.1, and Transformers 5.7.0.
- A separate learner overlay with FLA/fla-core 0.5.2 and Triton 3.7.1; see [training-acceleration.txt](../requirements/training-acceleration.txt).
- Separately obtained model weights and ALFWorld/AgentBench environment assets, under their providers' terms.

Use this inventory for the 9B joint-training stack; the older 2B coordinator/worker dependency files describe a separate experiment setup.

Historical launchers contain absolute paths, host/service assumptions, user/group IDs, and GPU mappings. Prepare a new configuration for your infrastructure. Do not globally replace paths inside frozen evidence or treat old orchestration scripts as a portable launch recipe. Offline verification and extraction never activate the archived training controls.

## Checkpoint access

The best97 package provides the Engineer/Solver actor pair, with the matching harness and per-task reports preserved in Git. Follow the [checkpoint download command](../README.md#checkpoints) using a Hugging Face account with access to the dataset.

The Engineer/Solver actor weights are stored in the Hugging Face dataset [wxd02/coevolve](https://huggingface.co/datasets/wxd02/coevolve). The [remote receipt](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json) records revision `d0e84414e8ef9e2b6a84585e1365189faf501051`; the [file manifest](../reproducibility/historical97_20261012/CHECKPOINT_UPLOAD_MANIFEST.json) records the file identities.

| Historical artifact | Dataset-relative path | Scope |
|---|---|---|
| Solver actor | `best97/solver/` | Actor weights, configuration, and tokenizer |
| Engineer actor | `best97/engineer/` | Actor weights, configuration, and tokenizer |

The 20 recorded checkpoint files total 37,679,454,196 bytes, approximately 35.09 GiB. This actor pair belongs to the historical 97% run, **not** the Full144 93% archive. The dataset is not a single model loadable from its root.

The upload did not include the harness, critics, optimizer, native continuation state, other checkpoints, or evaluation artifacts. The harness is available separately in the Git evidence linked above. Upload completion and remote metadata were checked; model download-back verification was not performed.

Resuming training requires both actors, both critics, Adam state, ValueNorm, RNG, schedule/data cursors, and matching initial/reference models. Small schedule/runtime configuration files inside an actor directory do not supply those missing states.

## Development checks

In an environment with `pytest` installed, run:

```bash
python3 -m pytest -q tests/test_ma_handoff.py tests/test_reproduce_ma_progress.py
python3 -B scripts/reproduce_ma_progress.py --best97
```

These checks validate evidence packaging and offline reconstruction. They are not GPU training or model-evaluation tests.

Keep credentials outside the checkout and never include them in commands, URLs, configuration snapshots, logs, or issue reports. Consult the [third-party notices](../THIRD_PARTY_NOTICES.md) for component attribution and licensing boundaries.
