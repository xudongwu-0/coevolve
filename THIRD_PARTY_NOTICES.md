# Third-party software and attribution

Coevolve is a research extension of **WHALE: A Simple Recipe for Joint Harness-Weight Optimization**, by the WHALE authors / KRAFTON AI. The upstream repository is https://github.com/krafton-ai/WHALE and is pinned as a separate Git submodule at `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`.

WHALE is distributed under Apache-2.0. Copies of its [license](licenses/WHALE-APACHE-2.0.txt) and [NOTICE](licenses/WHALE-NOTICE.txt) are included. Its own notices cover vendored verl and Search-R1 components; the submodule preserves those notices and source history. Our local modules adapt the execution interfaces and reuse the upstream environment, verifier, actor and selection code. We do not claim authorship of those components.

The editor update is inspired by Harness-R1's group-relative outcome optimization; the local implementation and its differences are described in [METHOD.md](docs/METHOD.md). Model weights (Qwen3.5-2B), MATH source data, and other datasets are not redistributed in this repository and remain subject to their providers' terms.

This private research repository does not add a license grant for the original Coevolve extensions. Third-party license files apply to their respective components.
