# Adaptive GPU filler operations

This folder contains the complete, agent-independent deployment for the
training-aware GPU filler on node3 (`dev-NDH200-025`) and node4
(`dev-NDH200-026`). After one installation, systemd starts it at boot and
restarts the supervisor after any exit. The supervisor separately restarts
failed per-GPU workers.

The filler is cooperative. It uses idle compute, but an external CUDA process
always has priority. It never signals or kills an external PID.

## Files and ownership

| File | Purpose |
|---|---|
| `bruli-h200-load.service.in` | Complete systemd unit template for fresh installation |
| `manage-adaptive-gpu-filler.sh` | Install and lifecycle command |
| `bruli-h200-load-adaptive.conf` | Compatibility drop-in for an older legacy unit; new installs do not require it |
| `../../whale_local/sao_bridge/adaptive_gpu_filler.py` | Supervisor, policy, telemetry, and CUDA worker |
| `../../whale_local/sao_bridge/resources.py` | Exact unit/PID identity and training-launch audit |

The installed unit remains named `bruli-h200-load.service` because launchers
already authorize that exact cgroup. It runs as the unprivileged `bruli` user.
Do not change the unit name, user, Python command shape, or cgroup without
updating and testing `resources.py`.

Runtime state is node-local below:

```text
/mnt/3fs/models/bruli/gpu-utilization-20260925/nodes/<hostname>/
```

Important files are:

- `adaptive-status.json`: current controller decision and per-GPU telemetry.
- `supervisor.log`: supervisor events.
- `adaptive-gpu-<index>.log`: worker events.
- `adaptive-worker-<index>.json`: worker heartbeat.
- `reservation.json`: optional per-GPU yield request.
- `PAUSED`: persistent operator pause marker.

## Install or update

Run this once on each authorized node. The command validates the source,
vendor CUDA libraries, GPU count, and rendered unit before installation. It
then enables boot start, starts the service, waits for a current status
receipt, and verifies the exact supervisor identity.

```bash
cd /mnt/3fs/xudong/coevolve
sudo scripts/systemd/manage-adaptive-gpu-filler.sh install
```

No agent has to keep a process alive after this command:

- `WantedBy=multi-user.target` plus `systemctl enable` starts it after reboot.
- `Restart=always` and `RestartSec=5` restart a failed supervisor.
- systemd `KillMode=control-group` cleans up all owned workers during a restart.
- heartbeat checks replace a failed or wedged worker.
- three consecutive telemetry failures yield all cards and exit, allowing
  systemd to restart from a clean state.
- `RequiresMountsFor` waits for both the repository and state mount.

Python-only updates take effect after `restart`. Unit-template or manager
updates should use `install` again. `install` preserves a persistent pause.

## Operate the service

All mutating commands require `sudo`. Read-only `status`, `verify`, `render`,
`systemctl show`, and `nvidia-smi` do not.

```bash
# Human-readable controller state plus live nvidia-smi values
scripts/systemd/manage-adaptive-gpu-filler.sh status

# Exact unit/supervisor identity and fresh status receipt
scripts/systemd/manage-adaptive-gpu-filler.sh verify

# Reload Python code or recover immediately
sudo scripts/systemd/manage-adaptive-gpu-filler.sh restart

# Stop now, but remain enabled and return after the next reboot
sudo scripts/systemd/manage-adaptive-gpu-filler.sh stop

# Persistently stop across reboot, systemd restart, and reinstall
sudo scripts/systemd/manage-adaptive-gpu-filler.sh pause

# Clear the persistent pause and restore boot start
sudo scripts/systemd/manage-adaptive-gpu-filler.sh resume

# Start after a temporary stop; refuses if PAUSED exists
sudo scripts/systemd/manage-adaptive-gpu-filler.sh start

# Remove the unit but retain all logs, state, and receipts
sudo scripts/systemd/manage-adaptive-gpu-filler.sh uninstall
```

Use `pause`, not only `stop`, when GPUs must remain unfilled across reboot or
unattended recovery. Never kill processes based only on a name match.

## Yield selected GPUs without stopping the service

The installed unit watches `reservation.json` every two seconds. Write JSON
atomically so the supervisor never reads a partial file:

```bash
node_dir=/mnt/3fs/models/bruli/gpu-utilization-20260925/nodes/$(hostname)
tmp=$(mktemp "$node_dir/reservation.json.XXXXXX")
printf '%s\n' '{"reserved_gpus":[0,1],"expires_epoch":1790816400}' >"$tmp"
sudo chown bruli:bruli "$tmp"
sudo mv "$tmp" "$node_dir/reservation.json"
```

`reserved_gpus` accepts indices 0--7. `expires_epoch` is optional; an expired
reservation is ignored. Remove the file to release all reservations:

```bash
sudo rm -f \
  /mnt/3fs/models/bruli/gpu-utilization-20260925/nodes/$(hostname)/reservation.json
```

Malformed JSON or an invalid index fails safe by yielding all eight GPUs.

## Policy and expected utilization

The control loop samples every two seconds:

| State | Condition | Filler behavior |
|---|---|---|
| `idle` | No external CUDA PID | 4,096 TF32 worker, slow ramp toward 86% total GPU utilization |
| `yielded` | New/busy external PID, less than 8 GiB free, reservation, or at least 82 C | No worker or filler CUDA context on that GPU |
| `shared` | External PID stays at no more than 5% SM for at least 6 seconds | 2,048 worker with 15%, 30%, then 50% duty caps |

On an otherwise empty node, expect approximately **85%--88% per GPU** rather
than an exact constant. The target is 86%, while `nvidia-smi` sampling and the
five-point ramp introduce small variation. After installing the self-contained units on 2026-10-01, the live snapshot
was:

- node3: 86, 86, 85, 85, 85, 85, 85, 85%; mean **85.25%**.
- node4: 86, 85, 85, 85, 86, 85, 86, 86%; mean **85.5%**.
- idle filler memory: about **791 MiB per H200**, versus about 80 GiB for the
  retired static filler.

When training is busy, filler utilization intentionally falls to zero and the
overall number is determined by training. Therefore a low aggregate value
during a memory-bound or communication-bound training phase is not a filler
failure. Algorithm utilization reports must exclude filler duty; use
`external_sm_mean_pct` and each row's `external` field in
`adaptive-status.json`.

The measured 20-second standalone TF32 throughput ratio was 98.53% with the
service active versus stopped, or about 1.47% observed handoff overhead. A
separate coexistence test at 89% external SM removed the filler worker and
CUDA context, then restored idle filling after the external process exited.

## Develop and validate

Safety invariants:

1. Exclude only exact child worker PIDs when measuring external work.
2. Never signal, terminate, or relabel an external process.
3. Scale down immediately; scale up gradually.
4. On uncertain telemetry, malformed reservations, heat, or memory pressure,
   yield rather than fabricate success.
5. Keep the supervisor and worker command lines compatible with
   `resources.py`.
6. Test only on node3/node4; do not deploy this unit to node0--2.

Focused validation:

```bash
cd /mnt/3fs/xudong/coevolve
bash -n scripts/systemd/manage-adaptive-gpu-filler.sh
scripts/systemd/manage-adaptive-gpu-filler.sh render >/tmp/bruli-h200-load.service
systemd-analyze verify /tmp/bruli-h200-load.service

CUDA_VISIBLE_DEVICES='' TMPDIR=/dev/shm PYTHONPATH=whale_local:. \
  /dev/shm/sao-venv-ppo/bin/python -B -m pytest -q \
  tests/test_sao_large.py whale_local/tests/test_alfworld_sao.py
```

After changing policy or lifecycle code, verify all of the following on one
node before rolling out to the other:

1. Empty-node utilization converges near 86%.
2. A new high-SM external process moves that GPU to `yielded` within one
   control cycle and leaves no filler worker.
3. The worker returns only after the external process exits or becomes
   eligible for shared mode.
4. Killing one verified worker replaces only that worker.
5. Killing the verified supervisor increments `NRestarts` and rebuilds the
   controller under systemd.
6. `resources audit` reports `filler_kind=adaptive`, a verified unit, and no
   filler PID as foreign.

For troubleshooting:

```bash
systemctl status --no-pager bruli-h200-load.service
systemctl show bruli-h200-load.service \
  -p ActiveState -p SubState -p UnitFileState -p MainPID -p NRestarts
tail -n 100 \
  /mnt/3fs/models/bruli/gpu-utilization-20260925/nodes/$(hostname)/supervisor.log
PYTHONPATH=whale_local python3 -B -m sao_bridge.resources audit
```

Do not recursively delete the state directory. Preserve logs and receipts when
reinstalling or uninstalling.
