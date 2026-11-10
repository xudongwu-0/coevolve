# Benchmark environments and checkpoints

JoMoHa studies household task completion, goal-directed shopping, and database interaction through **ALFWorld**, **WebShop**, and **DBBench**. This guide collects their source entry points and the matching best97 ALFWorld artifacts.

## Get the benchmark runtimes

The official Harness-R1 runtime is pinned at commit `94f2e087f573e1b82fc9568bb634d4ae9a887e28`:

```bash
git submodule update --init -- third_party/harness-r1-upstream
git -C third_party/harness-r1-upstream rev-parse HEAD
```

Use the pinned revision to keep the environment adapters, harness interfaces, and evaluation code aligned. The submodule preserves the upstream source, licenses, and notices; see [third-party attribution](../THIRD_PARTY_NOTICES.md#official-harness-r1-runtime-and-objective).

Follow the pinned [benchmark setup guide](../third_party/harness-r1-upstream/docs/BENCHMARK_SETUP.md) to prepare environment assets and model endpoints. All commands below run from the Coevolve repository root.

## ALFWorld

The local [joint-training implementation](../whale_local/sao_bridge/) pairs Engineer harness generation with Solver execution. Start with:

- [Paired rollout workflow](../whale_local/sao_bridge/joint_workflow.py).
- [ALFWorld runtime preparation](../whale_local/sao_bridge/alfworld_runtime.py).
- [Solver evaluation](../whale_local/sao_bridge/alfworld_solver_eval.py).
- [Training-source and environment setup](REPRODUCE_MA_SAO.md#restoring-the-training-source).

To verify archived ALFWorld outcomes without loading a model:

```bash
python3 -B scripts/reproduce_ma_progress.py --best97
```

## WebShop

The pinned Harness-R1 components cover shopping-task execution, executable harness hooks, dataset construction, patch rewards, and checkpoint evaluation:

| Component | Source |
|---|---|
| Task/environment adapter | [WebShop task package](../third_party/harness-r1-upstream/code/life-harness/AgentBench/src/server/tasks/webshop/) |
| Executable harness hooks | [webshop.py](../third_party/harness-r1-upstream/code/life-harness/AgentBench/src/server/harness/webshop.py) |
| Dataset builder | [build_webshop_dataset.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/build_webshop_dataset.py) |
| Patch reward | [reward_webshop_patch.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/reward_webshop_patch.py) |
| Evaluation implementation | [eval_webshop_checkpoint_patches.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/eval_webshop_checkpoint_patches.py) |
| Evaluation launcher | [eval_webshop.sh](../third_party/harness-r1-upstream/scripts/eval_webshop.sh) |

Prepare the WebShop environment, product assets, and search index using the upstream setup guide. The search stack uses a JDK with `javac`, `pyserini`/`jnius`, and compatible Python worker dependencies. Keep `WEBSHOP_GOAL_SEED=233` and task manifests aligned across baseline and patched runs.

With prepared input and running OpenAI-compatible Engineer/Solver endpoints:

```bash
ENGINEER_BASE_URL=http://127.0.0.1:8142/v1 \
TARGET_BASE_URL=http://127.0.0.1:8110/v1 \
bash third_party/harness-r1-upstream/scripts/eval_webshop.sh \
  "$PWD/data/webshop_eval.jsonl" "$PWD/runs/webshop_eval"
```

Use `ENGINEER_MODEL`, `TARGET_MODEL`, `AGENTBENCH_PYTHON`, and `WEBSHOP_WORKER_PYTHON` to select the served models and worker interpreters.

## DBBench

The pinned Harness-R1 components cover database tasks, code-patch hooks, RL dataset construction, rewards, and patch evaluation:

| Component | Source |
|---|---|
| Task/environment adapter | [DBBench task package](../third_party/harness-r1-upstream/code/life-harness/AgentBench/src/server/tasks/dbbench/) |
| Executable harness hooks | [dbbench.py](../third_party/harness-r1-upstream/code/life-harness/AgentBench/src/server/harness/dbbench.py) |
| Dataset builder | [build_dbbench_rl_dataset.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/build_dbbench_rl_dataset.py) |
| Patch reward | [reward_dbbench_codepatch.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/reward_dbbench_codepatch.py) |
| Evaluation implementation | [eval_dbbench_gpt_patches.py](../third_party/harness-r1-upstream/code/Relax/examples/harness_r1/eval_dbbench_gpt_patches.py) |
| Evaluation launcher | [eval_dbbench.sh](../third_party/harness-r1-upstream/scripts/eval_dbbench.sh) |

Prepare the benchmark task file and either the Docker driver or a MySQL 8 instance. Use a dedicated benchmark database with isolated schemas or a reset-capable controller.

With prepared input, model endpoints, and the manual MySQL driver:

```bash
ENGINEER_BASE_URL=http://127.0.0.1:8142/v1 \
TARGET_BASE_URL=http://127.0.0.1:8110/v1 \
DBBENCH_ENV_DRIVER=manual \
DBBENCH_MYSQL_HOST=127.0.0.1 \
DBBENCH_DATA_FILE="$PWD/data/dbbench/db_out_new.jsonl" \
bash third_party/harness-r1-upstream/scripts/eval_dbbench.sh \
  "$PWD/data/dbbench_eval.jsonl" "$PWD/runs/dbbench_eval"
```

Use `DBBENCH_WORKER_PYTHON` to select the worker interpreter. The upstream setup guide describes database provisioning and driver configuration.

## best97 checkpoints

The best97 artifact pair contains the Engineer and Solver actors associated with **194/200 (97.0%)** ALFWorld success using the selected `proposal07` harness.

| Artifact | Location |
|---|---|
| Solver actor | [Hugging Face: best97/solver](https://huggingface.co/datasets/wxd02/coevolve/tree/d0e84414e8ef9e2b6a84585e1365189faf501051/best97/solver) |
| Engineer actor | [Hugging Face: best97/engineer](https://huggingface.co/datasets/wxd02/coevolve/tree/d0e84414e8ef9e2b6a84585e1365189faf501051/best97/engineer) |
| Matching harness | [proposal07](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/patches/95-proposal07.json) |
| Per-task evaluation | [Original 194/200 report](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/evaluation200/attempt-01/reports/af8f607201d8a33fc24700422ee1b1169e830df854ff3022a63e0842d2dcdb76.json) |
| File identities | [Upload manifest](../reproducibility/historical97_20261012/CHECKPOINT_UPLOAD_MANIFEST.json) and [revision receipt](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json) |

See the [README download command](../README.md#checkpoints) for the pinned Hugging Face revision. Use an account with access to the dataset.

The checkpoint directories contain actor weights, configurations, and tokenizers. The full artifact pair comprises 20 files, approximately 35.09 GiB. See the [reproducibility guide](REPRODUCE_MA_SAO.md) for experiment lineage and native training-state details.

## Focused runtime tests

With `pytest` and the upstream Python dependencies installed:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES="" \
  python3 -m pytest -q \
  third_party/harness-r1-upstream/tests/test_dbbench_prompt_alignment.py \
  third_party/harness-r1-upstream/tests/test_webshop_identity.py
```

These CPU-only tests check DBBench prompt alignment and WebShop task identity/harness behavior.
