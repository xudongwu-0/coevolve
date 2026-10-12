# Agent handoff

## Start here

1. Read [the reproduction guide](docs/REPRODUCE_MA_SAO.md).
2. Run `python3 -B scripts/reproduce_ma_progress.py` from this checkout.
   This uses only Python's standard library and local project code: no API,
   model download, GPU, training, or model evaluation.
3. Read [the algorithm](docs/MA_SAO_ALGORITHM.md), then
   [the current state](docs/PROJECT_STATE.md). Older dated plans are history,
   not permission to run them.

## Current boundary

- All experiments and the proposed Full144 extension are paused.
- Full144: selected Harness + Solver = 186/200; weight-only = 179/200.
- `without_peer`: 76 metric rows per role, but the latest complete paired
  native checkpoint is **72**, not76. The other three ablations are unstarted.
- The 48 selection questions are contained in the200 development questions.
  Training task IDs0-239 do not overlap those questions; strict197 additionally
  removes three task instances whose goal groups overlap legacy training.
- One training seed and reused development data do not prove a robust ceiling,
  universal SAO superiority, or a causal comparison with the historical97%.

## Evidence and restoration

- [The bundle](reproducibility/ma_sao_20261012/) preserves original paths,
  payload SHA-256 values, raw score reports, task order, input ledgers, native
  metadata and seven executing-source snapshots. Do not rewrite frozen files.
- The Git bundle contains no model tensors. Consult
  [MODEL_ARTIFACTS.json](reproducibility/ma_sao_20261012/MODEL_ARTIFACTS.json);
  this is the earlier local inventory, not the current upload authorization.
  The multi-checkpoint upload plan has been cancelled in favor of the single
  historical97 checkpoint described in [the reproduction guide](docs/REPRODUCE_MA_SAO.md).
  Its two Actor checkpoints have been published; use the actual private HF
  revision in [the remote receipt](reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json).
  No Harness or native continuation state was uploaded, and no download-back
  verification was performed.
- Native recovery needs both Actors, both Critics, Adam, ValueNorm, RNG,
  schedule/data cursors and matching initial/reference models. HF Actor
  weights alone are not a continuation checkpoint.
- Older launchers bind absolute paths, node names, UID/GID, services and GPU
  mappings. A new machine needs an explicitly reviewed execution adaptation,
  not global replacements inside evidence. Existing cancellation gates remain
  effective even after files are materialized.
- Do not start, evaluate, restart, alter seeds/budgets, or reactivate an old
  waiter without a new explicit user request. External termination must not
  cause automatic restart; internal failures require diagnosis first.

## Credentials and publication

Credentials are outside this Git checkout in `/mnt/3fs/xudong/api_keys/`,
with directory mode0700 and file mode0600. Never print, stage, archive, place
in command arguments, or copy their contents into configuration or reports.
Read them only for a platform action explicitly authorized by the user.
The GitHub credential currently uses SSH-key authentication despite its file
name. The latest2026-10-12 15:34 Beijing instruction authorizes uploading only
the historical97 Engineer/Solver Actor checkpoint pair to a private HF
dataset. Do not upload its Harness, Critics, optimizer/native state, other
checkpoints, or evaluation artifacts. Do not download the checkpoint back
for verification; use upload completion and remote revision/file metadata.
W&B remains read-only. This permission does not authorize training or model
evaluation, and a historical model inventory is not an upload queue.

For packaging changes, run:

```bash
python3 -m pytest -q tests/test_ma_handoff.py tests/test_reproduce_ma_progress.py
python3 -B scripts/reproduce_ma_progress.py
```

The native-runtime test receipts in the bundle describe their exact frozen
source, not every future change to the working tree.
