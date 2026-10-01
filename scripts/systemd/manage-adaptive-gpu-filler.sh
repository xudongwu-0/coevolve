#!/usr/bin/env bash
set -Eeuo pipefail

UNIT=bruli-h200-load.service
SERVICE_USER=bruli
SERVICE_GROUP=bruli
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${GPU_FILLER_REPO_ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}
BASE_DIR=${GPU_FILLER_BASE_DIR:-/mnt/3fs/models/bruli/gpu-utilization-20260925}
SYSTEMD_DIR=${GPU_FILLER_SYSTEMD_DIR:-/etc/systemd/system}
TEMPLATE="$SCRIPT_DIR/bruli-h200-load.service.in"
UNIT_PATH="$SYSTEMD_DIR/$UNIT"
NODE_DIR="$BASE_DIR/nodes/$(hostname)"
PAUSE_FILE="$NODE_DIR/PAUSED"
RESERVATION_FILE="$NODE_DIR/reservation.json"
LEGACY_DROPIN="$SYSTEMD_DIR/$UNIT.d/adaptive-coexist.conf"

usage() {
    cat <<'EOF'
Usage: manage-adaptive-gpu-filler.sh COMMAND

Commands:
  install    Install/update the complete unit, enable boot start, and start it.
  status     Show systemd state, controller summary, and current GPU metrics.
  verify     Verify unit identity and the current supervisor status receipt.
  start      Start now; refuses while a persistent PAUSED marker exists.
  stop       Stop now but keep boot start enabled.
  pause      Stop and persist the pause across reboot and reinstall.
  resume     Clear the persistent pause, enable boot start, and start now.
  restart    Restart now and verify the new supervisor.
  uninstall  Disable and remove the unit; preserve state, logs, and receipts.
  render     Render the complete unit to stdout without installing it.
EOF
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        echo "ERROR: '$1' changes systemd state; rerun with sudo." >&2
        exit 2
    fi
}

require_authorized_host() {
    case "$(hostname)" in
        dev-NDH200-025|dev-NDH200-026) ;;
        *)
            echo "ERROR: adaptive GPU filler deployment is authorized only on node3/node4." >&2
            exit 2
            ;;
    esac
}

render_unit() {
    [[ -f "$TEMPLATE" ]] || {
        echo "ERROR: missing unit template: $TEMPLATE" >&2
        return 1
    }
    local content
    content=$(<"$TEMPLATE")
    content=${content//@REPO_ROOT@/$REPO_ROOT}
    content=${content//@BASE_DIR@/$BASE_DIR}
    content=${content//@NODE_DIR@/$NODE_DIR}
    content=${content//@PAUSE_FILE@/$PAUSE_FILE}
    content=${content//@RESERVATION_FILE@/$RESERVATION_FILE}
    printf '%s\n' "$content"
}

validate_install_inputs() {
    id "$SERVICE_USER" >/dev/null
    [[ -r "$REPO_ROOT/whale_local/sao_bridge/adaptive_gpu_filler.py" ]] || {
        echo "ERROR: adaptive filler source is not readable below $REPO_ROOT." >&2
        return 1
    }
    for library in libcublasLt.so.13 libcublas.so.13 libcudart.so.13; do
        [[ -r "$BASE_DIR/vendor/$library" ]] || {
            echo "ERROR: missing CUDA runtime library: $BASE_DIR/vendor/$library" >&2
            return 1
        }
    done
    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | wc -l)
    [[ "$gpu_count" -eq 8 ]] || {
        echo "ERROR: expected exactly 8 GPUs, found $gpu_count." >&2
        return 1
    }
}

prepare_state() {
    install -d -m 0755 "$BASE_DIR"
    install -d -m 0755 -o "$SERVICE_USER" -g "$SERVICE_GROUP" \
        "$BASE_DIR/nodes" "$NODE_DIR" "$NODE_DIR/home" "$NODE_DIR/tmp" \
        "$NODE_DIR/cache" "$NODE_DIR/cache/cuda"
}

wait_for_supervisor() {
    local expected_pid status_file="$NODE_DIR/adaptive-status.json"
    expected_pid=$(systemctl show "$UNIT" --property=MainPID --value)
    for _ in $(seq 1 15); do
        if [[ -s "$status_file" ]] && \
            PYTHONPATH="$REPO_ROOT/whale_local" /usr/bin/python3 -B - \
                "$status_file" "$expected_pid" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
raise SystemExit(0 if status.get("supervisor_pid") == int(sys.argv[2]) else 1)
PY
        then
            return 0
        fi
        sleep 2
    done
    echo "ERROR: no current supervisor receipt appeared at $status_file." >&2
    return 1
}

verify_running() {
    if [[ -e "$PAUSE_FILE" ]]; then
        echo "ERROR: persistent pause marker exists: $PAUSE_FILE" >&2
        return 1
    fi
    systemctl is-enabled --quiet "$UNIT"
    systemctl is-active --quiet "$UNIT"
    systemctl show "$UNIT" --property=ExecStart --value |
        grep -Fq -- "-m sao_bridge.adaptive_gpu_filler supervisor"
    wait_for_supervisor
    PYTHONPATH="$REPO_ROOT/whale_local" /usr/bin/python3 -B - <<'PY'
from sao_bridge.resources import audit

state = audit()
if not state["verified_unit"] or state["filler_kind"] != "adaptive":
    raise SystemExit(f"adaptive unit identity check failed: {state}")
if any(
    process["filler_kind"] is not None and not process["verified_filler"]
    for process in state["processes"]
):
    raise SystemExit(f"unverified filler process found: {state}")
print("verified adaptive supervisor identity")
PY
}

install_service() {
    require_root install
    require_authorized_host
    validate_install_inputs
    prepare_state

    local rendered
    rendered=$(mktemp --suffix=.service)
    render_unit >"$rendered"
    if ! systemd-analyze verify "$rendered"; then
        rm -f "$rendered"
        return 1
    fi
    install -D -m 0644 "$rendered" "$UNIT_PATH"
    rm -f "$rendered"
    rm -f "$LEGACY_DROPIN"
    rmdir "$SYSTEMD_DIR/$UNIT.d" 2>/dev/null || true
    systemctl daemon-reload
    systemctl enable "$UNIT"
    if [[ -e "$PAUSE_FILE" ]]; then
        systemctl stop "$UNIT"
        echo "Installed and enabled $UNIT; persistent pause remains active."
        return 0
    fi
    systemctl restart "$UNIT"
    verify_running
    echo "Installed, enabled, and started $UNIT."
}

show_status() {
    systemctl show "$UNIT" \
        --property=ActiveState,SubState,UnitFileState,MainPID,NRestarts,Restart,RestartUSec \
        --no-pager
    if [[ -e "$PAUSE_FILE" ]]; then
        echo "PersistentPause=yes ($PAUSE_FILE)"
    else
        echo "PersistentPause=no"
    fi
    if [[ -s "$NODE_DIR/adaptive-status.json" ]]; then
        /usr/bin/python3 -B - "$NODE_DIR/adaptive-status.json" <<'PY'
import json
import sys

status = json.load(open(sys.argv[1]))
print(
    "Controller:"
    f" utc={status['utc']}"
    f" supervisor_pid={status['supervisor_pid']}"
    f" total_util_mean={status['total_utilization_mean_pct']:.2f}%"
    f" external_sm_mean={status['external_sm_mean_pct']:.2f}%"
    f" filler_duty_mean={status['filler_duty_mean']:.3f}"
)
for row in status["gpus"]:
    print(
        f"GPU{row['index']}: util={row['utilization_pct']}%"
        f" memory={row['memory_used_mib']}/{row['memory_total_mib']}MiB"
        f" mode={row['decision']['mode']}"
        f" reason={row['decision']['reason']}"
        f" duty={row['worker_duty']:.3f}"
        f" external_pids={row['external']['pids']}"
    )
PY
    fi
    nvidia-smi \
        --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits
}

command=${1:-}
case "$command" in
    install)
        install_service
        ;;
    status)
        show_status
        ;;
    verify)
        verify_running
        ;;
    start)
        require_root start
        require_authorized_host
        [[ ! -e "$PAUSE_FILE" ]] || {
            echo "ERROR: service is persistently paused; use 'resume'." >&2
            exit 2
        }
        systemctl start "$UNIT"
        verify_running
        ;;
    stop)
        require_root stop
        require_authorized_host
        systemctl stop "$UNIT"
        echo "Stopped $UNIT; it remains enabled for the next boot."
        ;;
    pause)
        require_root pause
        require_authorized_host
        prepare_state
        install -m 0644 -o "$SERVICE_USER" -g "$SERVICE_GROUP" /dev/null "$PAUSE_FILE"
        systemctl stop "$UNIT"
        echo "Persistently paused $UNIT."
        ;;
    resume)
        require_root resume
        require_authorized_host
        rm -f "$PAUSE_FILE"
        systemctl enable --now "$UNIT"
        verify_running
        echo "Resumed $UNIT and enabled boot start."
        ;;
    restart)
        require_root restart
        require_authorized_host
        [[ ! -e "$PAUSE_FILE" ]] || {
            echo "ERROR: service is persistently paused; use 'resume'." >&2
            exit 2
        }
        systemctl restart "$UNIT"
        verify_running
        ;;
    uninstall)
        require_root uninstall
        require_authorized_host
        systemctl disable --now "$UNIT" 2>/dev/null || true
        rm -f "$UNIT_PATH" "$LEGACY_DROPIN"
        rmdir "$SYSTEMD_DIR/$UNIT.d" 2>/dev/null || true
        systemctl daemon-reload
        systemctl reset-failed "$UNIT" 2>/dev/null || true
        echo "Removed $UNIT; state and logs under $NODE_DIR were preserved."
        ;;
    render)
        render_unit
        ;;
    -h|--help|help|"")
        usage
        ;;
    *)
        echo "ERROR: unknown command: $command" >&2
        usage >&2
        exit 2
        ;;
esac
