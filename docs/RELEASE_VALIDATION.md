# Publication checks — 2026-09-22

This records checks performed for the code export, separately from historical GPU experiment results.

| Check | Outcome |
|---|---|
| Python parsing | All144 Python files under `whale_local/` and `scripts/` parsed |
| Upstream | Clean WHALE checkout at `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`, registered as submodule |
| Dependency resolution | Both coordinator and GPU-worker direct requirements resolved for Python3.12; transitive versions saved in `requirements/*.lock` |
| Fresh coordinator environment | Installed the coordinator lock into a new isolated venv; native environment, three-arm orchestrator and strict reward checks passed |
| Existing GPU-worker environment | Actor import resolves to the pinned WHALE math actor; Qwen3.5 compatibility imports and editor objective checks passed |
| Python tool | A real bubblewrap-isolated `print(sum([1,2,3]))` execution returned6 |
| Data reconstruction | All128 train,48 development and128 evaluation question IDs/order and target answers matched historical source records |
| New run preparation | Validated complete existing native solver/editor checkpoints and generated a fresh config/protocol without GPU model loads |
| Internal-only control path | A worker-start sentinel confirmed the external proposer was not started |
| Editor GPU routing | A generation stub received the configured GPU rather than the historical hardcoded GPU6 |
| Documentation | Local Markdown links checked; aggregated results and figure correspond to the recorded draw0 pilot |
| Submission contents | Code, documentation, dependency pins, figures and aggregates only; no checkpoints, raw data, local credentials or venvs |

The GPU-worker check reused the working local environment; a fresh installation of the complete worker lock and a new GPU training run were **not** performed for this export. The preparation check verifies stored checkpoint metadata and required files; runtime tensor rehashing happens on actual model load. The internal-only and GPU-routing checks are control-path tests, not new model outcomes.

The original pilot remains the evidence for real E/W updates. Its interpretation limits are recorded in [EXPERIMENTS.md](EXPERIMENTS.md), and its configuration-specific selection rule in [METHOD.md](METHOD.md).
