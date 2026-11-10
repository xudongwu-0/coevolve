# DBBench, WebShop, and the historical ALFWorld checkpoint

## What is available

Both benchmarks have implementation code in the already-tracked
[official Harness-R1 submodule](../third_party/harness-r1-upstream).
Coevolve pins upstream commit
`94f2e087f573e1b82fc9568bb634d4ae9a887e28`; the links below use that exact version.
The source is not a new Coevolve implementation and should not be copied out
without its upstream licenses and notices. See
[third-party attribution](../THIRD_PARTY_NOTICES.md#official-harness-r1-runtime-and-objective).

This checkout has benchmark task adapters, harness hooks, dataset builders,
reward functions, evaluation entry points, and focused unit tests.
It does **not** establish that Coevolve's two-role MA-SAO trainer has been
ported or validated on DBBench or WebShop. We do not report local training
results for those environments here. The historical 97% result is ALFWorld,
not either of these benchmarks.

## Source index

Paths are relative to `third_party/harness-r1-upstream`.

| Component | DBBench | WebShop |
|---|---|---|
| Task/environment adapter | [DBBench task package](https://github.com/DeepExperience/Harness-R1/tree/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/life-harness/AgentBench/src/server/tasks/dbbench) | [WebShop task package](https://github.com/DeepExperience/Harness-R1/tree/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/life-harness/AgentBench/src/server/tasks/webshop) |
| Executable harness hooks | [dbbench.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/life-harness/AgentBench/src/server/harness/dbbench.py) | [webshop.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/life-harness/AgentBench/src/server/harness/webshop.py) |
| Dataset construction | [build_dbbench_rl_dataset.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/build_dbbench_rl_dataset.py) | [build_webshop_dataset.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/build_webshop_dataset.py) |
| Patch reward | [reward_dbbench_codepatch.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/reward_dbbench_codepatch.py) | [reward_webshop_patch.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/reward_webshop_patch.py) |
| Evaluation implementation | [eval_dbbench_gpt_patches.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/eval_dbbench_gpt_patches.py) | [eval_webshop_checkpoint_patches.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/code/Relax/examples/harness_r1/eval_webshop_checkpoint_patches.py) |
| Evaluation launcher | [scripts/eval_dbbench.sh](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/scripts/eval_dbbench.sh) | [scripts/eval_webshop.sh](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/scripts/eval_webshop.sh) |
| Focused tests | [test_dbbench_prompt_alignment.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/tests/test_dbbench_prompt_alignment.py) | [test_webshop_identity.py](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/tests/test_webshop_identity.py) |

## Get the pinned source without downloading models

From an existing Coevolve checkout, initialize only this submodule:

```bash
git submodule update --init -- third_party/harness-r1-upstream
git -C third_party/harness-r1-upstream rev-parse HEAD
```

The second command must print
`94f2e087f573e1b82fc9568bb634d4ae9a887e28`.
Do not use `git submodule update --remote`: that changes the upstream version.
A ZIP download or a clone without submodule initialization does not include
the benchmark source. The commands above retrieve the upstream repository,
including its small manifests/examples, but do not download model weights or
the separately distributed benchmark assets, install environments, or run
training/evaluation.

For CPU-only code checks, with pytest and the upstream Python dependencies
available in the chosen interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python -m pytest -q \
  third_party/harness-r1-upstream/tests/test_dbbench_prompt_alignment.py \
  third_party/harness-r1-upstream/tests/test_webshop_identity.py
```

These are synthetic unit tests, not benchmark score reproduction.

## Runtime prerequisites and scope

Follow the pinned upstream
[benchmark setup guide](https://github.com/DeepExperience/Harness-R1/blob/94f2e087f573e1b82fc9568bb634d4ae9a887e28/docs/BENCHMARK_SETUP.md)
before attempting real runs.

- **DBBench:** obtain the benchmark task file separately and provision either
  the Docker driver or a MySQL 8 instance for the manual driver. Use isolated
  schemas or a reset-capable controller for concurrency. Never point it at a
  shared production database.
- **WebShop:** install the external environment, product assets, and search
  index separately. The search stack needs a JDK with `javac`,
  `pyserini`/`jnius`, and compatible Python worker dependencies. Baseline and
  patched runs must share `webshop_goal_seed=233` and strict task manifests.
- Both launchers require prepared input and configured Engineer/target model
  endpoints. Merely fetching their code does not make the environment ready.
- No benchmark data, model tensors, credentials, or new experiment results
  are added by this documentation update. Nothing here authorizes starting
  training, inference, database services, or a resource waiter.

## Historical ALFWorld 97% checkpoint

- **Private Hugging Face dataset:**
  [wxd02/coevolve, immutable best97 revision](https://huggingface.co/datasets/wxd02/coevolve/tree/d0e84414e8ef9e2b6a84585e1365189faf501051/best97).
  Access requires an account/token authorized for that private repository.
- **Solver Actor:**
  [best97/solver](https://huggingface.co/datasets/wxd02/coevolve/tree/d0e84414e8ef9e2b6a84585e1365189faf501051/best97/solver).
- **Engineer Actor:**
  [best97/engineer](https://huggingface.co/datasets/wxd02/coevolve/tree/d0e84414e8ef9e2b6a84585e1365189faf501051/best97/engineer).
- **Lineage:** `alfworld_ma_convergence_v1r4/block-01/round0096`,
  cumulative round 144; both Actors are `epoch0epochstep95globalstep95`.
- **Result:** selected `proposal07` harness plus Solver achieved
  **194/200 (97.0%)**; the same Solver without that harness achieved
  **185/200 (92.5%)**. This is a reused-development-set peak, not an
  independent held-out or multi-training-seed result.

HF contains only the two Actor checkpoints and their tokenizer/configuration
files: 20 files, 37,679,454,196 bytes. It does not contain the harness, Critics,
optimizer, RNG, or complete native training recovery state. Upload completion
was verified using remote metadata, not by downloading the weights back.

The matching artifacts are already in the Git evidence bundle:

- [Selected proposal07 harness](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/patches/95-proposal07.json).
- [Original 194/200 per-task report](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/evaluation200/attempt-01/reports/af8f607201d8a33fc24700422ee1b1169e830df854ff3022a63e0842d2dcdb76.json).
- [Checkpoint upload manifest](../reproducibility/historical97_20261012/CHECKPOINT_UPLOAD_MANIFEST.json)
  and [remote verification receipt](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json).
- [Full reproduction and access guide](REPRODUCE_MA_SAO.md).
