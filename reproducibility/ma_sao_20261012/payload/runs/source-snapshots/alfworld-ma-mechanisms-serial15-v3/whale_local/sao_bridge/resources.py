"""Identity-check GPU work and explicitly pause only verified utilization fillers."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import file_sha, read_json, write_json

UNIT = "bruli-h200-load.service"
SCRIPT = "/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py"
ADAPTIVE_SCRIPT = "/mnt/3fs/xudong/gpu-utilization/adaptive_gpu_filler.py"
RESEARCH_HOSTS = frozenset({"dev-NDH200-025", "dev-NDH200-026"})
PAUSE_ROOT = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/nodes")
SERIAL_PAUSE_REASON = "User requested pausing GPU fillers on node3/node4 for serial MA-SAO training"
PAUSE_REASONS = frozenset({
    "User requested stopping GPU stress on s23-s26", SERIAL_PAUSE_REASON,
})
SCOPE_ENV = ("SAO_RESOURCE_AUTHORITY", "SAO_RESOURCE_AUTHORITY_SHA256")


def scope_environment() -> dict[str, str]:
    values = {name: os.environ[name] for name in SCOPE_ENV if name in os.environ}
    if values and (len(values) != len(SCOPE_ENV) or not all(values.values())):
        raise ValueError("Incomplete GPU partition authority environment")
    return values


def execution_gpu_indices() -> tuple[int, ...] | None:
    from .alfworld_ma_idle_resume import HOSTS, SERIAL_SCHEMA, verify_authority

    values = scope_environment()
    if not values:
        return None
    policy = verify_authority(Path(values[SCOPE_ENV[0]]), values[SCOPE_ENV[1]])
    if policy["schema"] != SERIAL_SCHEMA:
        raise ValueError("GPU partitioning requires explicit serial-training authority")
    node = next((name for name, host in HOSTS.items() if host == os.uname().nodename), None)
    if node is None:
        raise RuntimeError("GPU partition authority does not cover this host")
    return tuple(policy["gpu_indices"][node])


def logical_gpu_string() -> str:
    indices = execution_gpu_indices()
    return ",".join(map(str, range(len(indices) if indices is not None else 8)))


def physical_gpu_string(logical: str) -> str:
    indices = execution_gpu_indices()
    if indices is None or not logical:
        return logical
    devices = logical.split(",")
    if (
        any(not value.isdigit() or int(value) >= len(indices) for value in devices)
        or len(devices) != len(set(map(int, devices)))
    ):
        raise ValueError("GPU launch escaped its explicitly allocated logical devices")
    return ",".join(str(indices[int(value)]) for value in devices)


def runtime_gpu_counts(spec: dict) -> tuple[int, int]:
    actor, rollout = spec.get("actor_gpus", 2), spec.get("rollout_gpus", 2)
    indices = execution_gpu_indices()
    if indices is None:
        return actor, rollout
    from .alfworld_ma_idle_resume import HOSTS, spec_deadline

    binding = spec.get("rolling_recovery", {}).get("execution_authority")
    values = scope_environment()
    if (
        not binding or binding["authority"] != values[SCOPE_ENV[0]]
        or binding["authority_sha256"] != values[SCOPE_ENV[1]]
        or actor != 4 or rollout != 4
        or os.uname().nodename != HOSTS[{"editor": "node3", "solver": "node4"}[spec["role"]]]
    ):
        raise ValueError("Runtime GPU partition does not match the admitted native role")
    spec_deadline(spec)
    return actor, len(indices) - actor


def require_research_node() -> None:
    host = os.uname().nodename
    if host not in RESEARCH_HOSTS:
        raise RuntimeError(f"Research compute is authorized only on node3/node4, not {host}")


def process_identity(pid: int) -> dict:
    root = Path("/proc") / str(pid)
    return {
        "pid": pid, "uid": root.stat().st_uid,
        "cmd": [s.decode() for s in (root / "cmdline").read_bytes().split(b"\0") if s],
        "cgroup": (root / "cgroup").read_text().strip(),
    }


def filler_kind(record: dict, worker: bool = True) -> str | None:
    cmd = record["cmd"]
    suffix = cmd[3:] if cmd[:3] == ["/usr/bin/python3", "-B", SCRIPT] else []
    legacy = (
        len(suffix) == 3 and suffix[:2] == ["worker", "--gpu"]
        and suffix[2].isdigit() and 0 <= int(suffix[2]) < 8
    ) if worker else suffix == ["supervisor"]
    adaptive_suffix = (
        cmd[3:] if cmd[:3] == ["/usr/bin/python3", "-B", ADAPTIVE_SCRIPT] else []
    )
    adaptive = (
        len(adaptive_suffix) == 7
        and adaptive_suffix[0] == "worker"
        and adaptive_suffix[1] == "--gpu"
        and adaptive_suffix[2].isdigit()
        and 0 <= int(adaptive_suffix[2]) < 8
        and adaptive_suffix[3] == "--matrix-size"
        and adaptive_suffix[4] in {"2048", "4096"}
        and adaptive_suffix[5] == "--control-fd"
        and adaptive_suffix[6].isdigit()
    ) if worker else adaptive_suffix == ["supervisor"]
    if not record["cgroup"].endswith("/system.slice/" + UNIT):
        return None
    return "legacy" if legacy else "adaptive" if adaptive else None


def is_filler(record: dict, worker: bool = True) -> bool:
    return filler_kind(record, worker=worker) is not None


def audit() -> dict:
    import pwd

    expected_uid = pwd.getpwnam("bruli").pw_uid
    props = subprocess.check_output(
        ["systemctl", "show", UNIT, "--property=MainPID,ActiveState,User"], text=True,
    )
    unit = dict(line.split("=", 1) for line in props.splitlines())
    pids = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).split()
    main = process_identity(int(unit["MainPID"])) if int(unit["MainPID"]) else None
    main_kind = filler_kind(main, worker=False) if main else None
    records = []
    vanished_pids = []
    for pid in map(int, pids):
        try:
            records.append(process_identity(pid))
        except FileNotFoundError:
            vanished_pids.append(pid)
    for record in records:
        record["filler_kind"] = filler_kind(record)
        record["verified_filler"] = (
            record["uid"] == expected_uid
            and record["filler_kind"] is not None
            and record["filler_kind"] == main_kind
        )
    return {
        "host": os.uname().nodename, "time": time.time(), "unit": unit,
        "supervisor": main, "processes": records,
        "filler_kind": main_kind,
        "vanished_gpu_pids": vanished_pids,
        "verified_unit": bool(
            main and unit["User"] == "bruli" and main["uid"] == expected_uid
            and main_kind is not None
        ),
    }


def scoped_audit() -> dict:
    indices = execution_gpu_indices()
    state = audit()
    if indices is None:
        return state
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits",
    ], text=True, timeout=30)
    uuids = {}
    for line in devices.splitlines():
        index, uuid = (part.strip() for part in line.split(","))
        uuids[uuid] = int(index)
    if set(uuids.values()) != set(range(8)) or len(uuids) != 8:
        raise ValueError("Cannot identify all physical GPUs for the admitted partition")
    response = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits",
    ], text=True, timeout=30)
    allocations: dict[int, set[int]] = {}
    for line in response.splitlines():
        pid, uuid = (part.strip() for part in line.split(","))
        if uuid not in uuids:
            raise ValueError("A GPU process references an unidentified physical device")
        allocations.setdefault(int(pid), set()).add(uuids[uuid])
    records = {record["pid"]: record for record in state["processes"]}
    for pid in allocations.keys() - records.keys():
        try:
            record = process_identity(pid)
        except FileNotFoundError:
            state["vanished_gpu_pids"].append(pid)
            continue
        record["filler_kind"] = filler_kind(record)
        record["verified_filler"] = bool(
            state["verified_unit"] and record["uid"] == state["supervisor"]["uid"]
            and record["filler_kind"] == state["filler_kind"]
        )
        records[pid] = record
    included, excluded = [], []
    for record in records.values():
        physical = allocations.get(record["pid"])
        record["gpu_indices"] = sorted(physical) if physical is not None else None
        if physical is not None and physical.isdisjoint(indices):
            excluded.append({
                "pid": record["pid"], "uid": record["uid"],
                "gpu_indices": record["gpu_indices"],
                "verified_filler": record["verified_filler"],
            })
        else:
            included.append(record)
    values = scope_environment()
    return {
        **state, "processes": included,
        "resource_scope": {
            "host": state["host"], "gpu_indices": list(indices),
            "authority": values[SCOPE_ENV[0]], "authority_sha256": values[SCOPE_ENV[1]],
            "excluded_gpu_processes": excluded,
        },
    }


def gpu_release_audit() -> dict:
    state = scoped_audit()
    filler = [
        process["pid"] for process in state["processes"]
        if process["verified_filler"]
    ]
    foreign = [
        process["pid"] for process in state["processes"]
        if not process["verified_filler"]
    ]
    result = {
        "remaining_gpu_pids": filler + foreign,
        "verified_filler_pids": filler,
        "foreign_gpu_pids": foreign,
        "vanished_gpu_pids": state["vanished_gpu_pids"],
        "filler_kind": state["filler_kind"],
        "filler_unit_active": state["unit"]["ActiveState"] == "active",
    }
    if "resource_scope" in state:
        result["resource_scope"] = state["resource_scope"]
    if not result["filler_unit_active"]:
        pause = operator_pause_audit()
        if pause is not None:
            result["operator_pause"] = pause
    return result


def operator_pause_audit() -> dict | None:
    import pwd

    host = os.uname().nodename
    if host not in RESEARCH_HOSTS:
        return None
    marker = PAUSE_ROOT / host / "PAUSED"
    if not marker.exists():
        return None
    properties = subprocess.check_output([
        "systemctl", "show", UNIT,
        "--property=ActiveState,ConditionResult,MainPID,User,ExecStart",
    ], text=True, timeout=30)
    service = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    if (marker.is_symlink() or not marker.is_file()
            or marker.stat().st_uid != pwd.getpwnam("bruli").pw_uid):
        raise ValueError("The stress-filler pause marker has an unexpected identity")
    content = read_json(marker)
    if (content.get("paused") is not True
            or content.get("reason") not in PAUSE_REASONS
            or service["ActiveState"] != "inactive"
            or not (
                service["ConditionResult"] == "no"
                or content.get("reason") == SERIAL_PAUSE_REASON and service["MainPID"] == "0"
            )
            or service["User"] != "bruli"
            or f"argv[]=/usr/bin/python3 -B {ADAPTIVE_SCRIPT} supervisor ;" not in service["ExecStart"]):
        raise ValueError("Inactive stress filler does not match the explicit operator pause")
    return {
        "verified": True, "host": host, "path": str(marker), "sha256": file_sha(marker),
        "content": content, "service": service, "verified_at_unix": time.time(),
    }


def research_resources_released(release: dict) -> bool:
    if release.get("vanished_gpu_pids"):
        return False
    scope = release.get("resource_scope")
    if scope is not None:
        expected = {
            "dev-NDH200-025": list(range(8)), "dev-NDH200-026": list(range(1, 8)),
        }
        if (
            scope.get("host") not in expected
            or scope.get("gpu_indices") != expected[scope["host"]]
            or not Path(scope.get("authority", "")).is_absolute()
            or len(scope.get("authority_sha256", "")) != 64
            or any(
                not item.get("gpu_indices")
                or not set(item["gpu_indices"]).isdisjoint(scope["gpu_indices"])
                for item in scope.get("excluded_gpu_processes", [])
            )
        ):
            return False
    if (release["foreign_gpu_pids"]
            or not set(release["remaining_gpu_pids"]) <= set(release["verified_filler_pids"])):
        return False
    if release["filler_unit_active"]:
        return True
    pause = release.get("operator_pause")
    if not pause or release["remaining_gpu_pids"]:
        return False
    return (
        pause.get("verified") is True
        and pause["host"] in RESEARCH_HOSTS
        and pause["path"] == str(PAUSE_ROOT / pause["host"] / "PAUSED")
        and len(pause["sha256"]) == 64
        and pause["content"].get("paused") is True
        and pause["content"].get("reason") in PAUSE_REASONS
        and pause["service"]["ActiveState"] == "inactive"
        and (
            pause["service"]["ConditionResult"] == "no"
            or pause["content"].get("reason") == SERIAL_PAUSE_REASON
            and pause["service"].get("MainPID") == "0"
        )
        and pause["service"]["User"] == "bruli"
        and f"argv[]=/usr/bin/python3 -B {ADAPTIVE_SCRIPT} supervisor ;" in pause["service"]["ExecStart"]
    )


def stop_known(*, require_exclusive: bool) -> dict:
    require_research_node()
    state = scoped_audit()
    foreign = [p["pid"] for p in state["processes"] if not p["verified_filler"]]
    if require_exclusive and foreign:
        raise RuntimeError(f"Cannot allocate node; non-filler GPU processes: {foreign}")
    if state["unit"]["ActiveState"] == "active":
        if not state["verified_unit"]:
            raise RuntimeError("Active utilization unit identity did not match authorization")
        if state["filler_kind"] == "adaptive":
            state["stopped"] = False
            state["cooperative"] = True
            return state
        # Recheck the exact supervisor immediately before targeting its service.
        main = process_identity(state["supervisor"]["pid"])
        if main != state["supervisor"]:
            raise RuntimeError("Utilization supervisor changed during validation")
        subprocess.run(["sudo", "-n", "systemctl", "stop", UNIT], check=True, timeout=45)
        state["stopped"] = True
    else:
        state["stopped"] = False
    state["cooperative"] = False
    return state


def write_pause_marker() -> dict:
    import pwd
    from .alfworld_train_data import _immutable_json

    require_research_node()
    if os.getuid() != pwd.getpwnam("bruli").pw_uid:
        raise PermissionError("Only the verified filler owner may create its pause marker")
    marker = PAUSE_ROOT / os.uname().nodename / "PAUSED"
    if marker.is_symlink():
        raise ValueError("Refusing a symlinked filler pause marker")
    if marker.exists():
        value = read_json(marker)
        if value.get("paused") is not True or value.get("reason") not in PAUSE_REASONS:
            raise ValueError("An unrelated pause marker already exists")
        return value
    value = {
        "paused": True, "reason": SERIAL_PAUSE_REASON,
        "host": os.uname().nodename, "at_unix": time.time(),
    }
    _immutable_json(marker, value)
    return value


def pause_verified_fillers() -> dict:
    require_research_node()
    state = audit()
    if state["unit"]["ActiveState"] != "active" or not state["verified_unit"]:
        raise RuntimeError("Explicit pause requires the currently verified active filler unit")
    supervisor = state["supervisor"]
    stat_path = Path("/proc") / str(supervisor["pid"]) / "stat"
    start_ticks = stat_path.read_text().rsplit(") ", 1)[1].split()[19]
    subprocess.run([
        "sudo", "-n", "-u", "bruli", "env",
        f"PYTHONPATH={Path(__file__).resolve().parents[1]}",
        "PYTHONDONTWRITEBYTECODE=1", sys.executable, "-B", "-m",
        "sao_bridge.resources", "write-pause-marker",
    ], check=True, timeout=30)
    if (
        process_identity(supervisor["pid"]) != supervisor
        or stat_path.read_text().rsplit(") ", 1)[1].split()[19] != start_ticks
    ):
        raise RuntimeError("Filler supervisor identity changed before the explicit stop")
    subprocess.run(
        ["sudo", "-n", "kill", "-TERM", str(supervisor["pid"])],
        check=True, timeout=30,
    )
    # Restart=always can bypass start conditions during an automatic restart.
    subprocess.run(["sudo", "-n", "systemctl", "stop", UNIT], check=True, timeout=45)
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        after = audit()
        remaining = [p["pid"] for p in after["processes"] if p["verified_filler"]]
        if not remaining and after["unit"]["ActiveState"] == "inactive":
            pause = operator_pause_audit()
            if pause is None:
                raise RuntimeError("Stopped filler lacks its verified persistent pause")
            return {
                "status": "paused", "host": state["host"], "supervisor_pid": supervisor["pid"],
                "supervisor_start_ticks": start_ticks,
                "stopped_filler_pids": [p["pid"] for p in state["processes"] if p["verified_filler"]],
                "preserved_foreign_gpu_pids": [p["pid"] for p in after["processes"] if not p["verified_filler"]],
                "operator_pause": pause, "at_unix": time.time(),
            }
        time.sleep(1)
    raise TimeoutError("Verified filler did not release its GPUs within45 seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "stop-known", "pause-known", "write-pause-marker"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "pause-known":
        result = pause_verified_fillers()
    elif args.action == "write-pause-marker":
        result = write_pause_marker()
    else:
        result = audit() if args.action == "audit" else stop_known(require_exclusive=True)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
