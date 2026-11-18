# WebShop MA-SAO: step 144

The Solver with the **default harness** succeeded on **37 of 64** held-out
tasks (57.81%; mean raw score 0.8232) in the 1,000-product preview. This is
not the [original WebShop paper](https://arxiv.org/abs/2207.01206)'s 500-task
benchmark. Development tasks 384–447 selected the harness (default 42/64,
learned 40/64); held-out tasks 512–575 were evaluated once after selection.
The development and held-out denominators must not be added.

The candidate resumed both roles from round 12 to cumulative round 144.
Training used strict success minus `0.1 * invalid_actions / environment_actions`;
evaluation used the unmodified benchmark score. Only the final round of the
continuation exported models and saved paired native state.

On the machine containing the committed run and its Python environment, use
`bash coevolve/scripts/upload_webshop144.sh` from the directory above the
checkout. Add `--check` for an offline input check. The command uploads only
the two final inference Actors and their inference assets, license and model
cards. It requires a Hugging Face write token for `wxd02`, checks the remote
weight hashes, and neither starts models nor runs evaluations.
