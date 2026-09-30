#!/usr/bin/env bash
set -Eeuo pipefail
cd /mnt/3fs/xudong/coevolve
export PYTHONPATH="$PWD/whale_local"
export TMPDIR=/dev/shm PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cleanup() {
    code=$?
    trap - EXIT
    if gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits); then
        if [ -z "$gpu_pids" ]; then
            sudo -n systemctl start bruli-h200-load.service || {
                echo 'ERROR: authorized filler restoration failed'
                exit 1
            }
        else
            printf 'GPU processes remain; filler not started: %s\n' "$gpu_pids"
        fi
    else
        echo 'ERROR: cannot audit GPUs for filler restoration'
        exit 1
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
RUN="$PWD/runs/rewrite_timebox_20260929T0745Z"
PRIOR="$PWD/runs/rewrite_round1_calibration_node4_20260929T0740Z"
INITIAL="$PWD/runs/rewrite_progressive_start_node4_20260929T0808Z"
/dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.rewrite_progressive \
    --run "$RUN" \
    --queue "$PWD/assets/rewrite_teacher_round1_20260929/queue" \
    --fixed-data "$PWD/assets/rewrite_round1_calibration_512_v1/examples.json" \
    --control "$PRIOR/original_dev_evaluation" \
    --predecessor-exit "$INITIAL/exit_code" \
    --initial-training "$INITIAL/training" \
    --initial-evaluation "$INITIAL/dev" \
    --train-until 2026-09-29T09:45:00+00:00 \
    --finish-by 2026-09-29T10:45:00+00:00
