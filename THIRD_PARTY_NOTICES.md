# Third-party software and attribution

Coevolve is a research extension of **WHALE: A Simple Recipe for Joint Harness-Weight Optimization**, by the WHALE authors / KRAFTON AI. The upstream repository is https://github.com/krafton-ai/WHALE and is pinned as a separate Git submodule at `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`.

WHALE is distributed under Apache-2.0. Copies of its [license](licenses/WHALE-APACHE-2.0.txt) and [NOTICE](licenses/WHALE-NOTICE.txt) are included. Its own notices cover vendored verl and Search-R1 components; the submodule preserves those notices and source history. Our local modules adapt the execution interfaces and reuse the upstream environment, verifier, actor and selection code. We do not claim authorship of those components.

The editor update is inspired by Harness-R1's group-relative outcome optimization; the local implementation and its differences are described in [METHOD.md](docs/METHOD.md). Model weights (Qwen3.5-2B), MATH source data, and other datasets are not redistributed in this repository and remain subject to their providers' terms.

This private research repository does not add a license grant for the original Coevolve extensions. Third-party license files apply to their respective components.

## Optional independent SAO baseline

`third_party/sao` pins the Apache-2.0 **unofficial** SAO implementation by
fooSynaptic at `ac2728149041b5f15ceb75413f467ce0659179cf`, from
https://github.com/fooSynaptic/Single-rollout-async-Optimization. Its original
license and attribution remain in the submodule. It is not endorsed by the
authors of *Single-Rollout Asynchronous Optimization for Agentic Reinforcement
Learning* (arXiv:2607.07508).

Its bootstrap script downloads Apache-2.0 AReaL at
`3cf0dfbd2b0fbeabd6977184980e189d1567747a` and applies the submodule's published
SAO integration patch. The optional `whale_local/sao_bridge` task adapters call
that trainer; they do not implement a replacement PPO/SAO optimizer. This
baseline does not change the original RSFT/GRPO experiment.
# MAPPO design reference

The optional `third_party/mappo-reference` submodule is the official
`marlbenchmark/on-policy` repository, pinned at
`de66d7a4b23fac2513f56f96f73b3f5cb96695ac`. It is retained as a design reference
for centralized training/decentralized execution; its PPO implementation is
not substituted for the SAO training objective. Its upstream notices and license
remain in the submodule.

## Official Harness-R1 runtime and objective

`third_party/harness-r1-upstream` pins the official
[DeepExperience/Harness-R1](https://github.com/DeepExperience/Harness-R1)
repository at `94f2e087f573e1b82fc9568bb634d4ae9a887e28`.
Its Apache-2.0 license, NOTICE and component licenses remain in the submodule.
The SearchQA adapter directly imports checksum-verified official normalization
and PPO/KL primitives; its FSDP/vLLM backend is a local adaptation.
The ALFWorld adapter invokes unchanged official builders, evaluators and
AgentBench runtime through a writable shadow directory.
Released model weights and benchmark assets are downloaded separately and are
not redistributed by this repository. See the
[current project state](docs/PROJECT_STATE.md) for model-role distinctions and
reproduction limitations.

## Frozen MA-SAO reproduction evidence

The dated reproduction bundle includes the original executing Coevolve source
snapshots, generated research inputs/results, and metadata identifying model and
dataset dependencies. It does not contain model tensors or redistribute the
complete ALFWorld game archive.

The AReaL working-tree patch reconstructs the locally modified Apache-2.0
upstream at the pinned revision. Its original license is included at
[AReaL-LICENSE](reproducibility/ma_sao_20261012/upstream/AReaL-LICENSE).
It supplements, rather than replaces, the attribution and licenses of the
upstream submodules. Generated task evidence and task-path catalogues do not
provide new rights to underlying benchmark assets or model weights.
