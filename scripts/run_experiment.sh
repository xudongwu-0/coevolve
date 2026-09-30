#!/usr/bin/env bash
# Usage: scripts/run_experiment.sh examples/SPEC.json
# Syncs the shared venv to node-local tmpfs (runtime cache only), then runs the experiment.
set -euo pipefail
cd "$(dirname "$0")/.."
LOCAL=/dev/shm/sao-venv-ppo
rsync -a --delete .venv-ppo/ "$LOCAL/"
export PYTHONPATH="$PWD/whale_local"
exec "$LOCAL/bin/python" -B -m sao_bridge.experiment --spec "$1"
