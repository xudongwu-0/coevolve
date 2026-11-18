#!/usr/bin/env bash
set -euo pipefail
scripts="$(cd -- "$(dirname -- "$0")" && pwd)"
exec "$scripts/../runs/webshop_self_distillation_20261114/env/bin/python" "$scripts/upload_webshop144.py" "$@"
