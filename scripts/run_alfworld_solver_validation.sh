#!/usr/bin/env bash
# Usage: scripts/run_alfworld_solver_validation.sh examples/SPEC.json
set -euo pipefail
cd "$(dirname "$0")/.."
LOCAL=/dev/shm/sao-venv-ppo
rsync -a --delete .venv-ppo/ "$LOCAL/"
export PYTHONPATH="$PWD/whale_local"
exec "$LOCAL/bin/python" -B -m sao_bridge.alfworld_solver_validation --spec "$1"
