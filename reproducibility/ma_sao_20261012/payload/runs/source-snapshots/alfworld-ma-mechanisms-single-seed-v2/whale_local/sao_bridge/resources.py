"""Identity-check the standalone filler and stop only its legacy implementation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .common import file_sha, read_json, write_json

UNIT = "bruli-h200-load.service"
SCRIPT = "/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py"
ADAPTIVE_SCRIPT = "/mnt/3fs/xudong/gpu-utilization/adaptive_gpu_filler.py"
RESEARCH_HOSTS = frozenset({"dev-NDH200-025", "dev-NDH200-026"})
PAUSE_ROOT = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/nodes")


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


def gpu_release_audit() -> dict:
    state = audit()
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
        "--property=ActiveState,ConditionResult,User,ExecStart",
    ], text=True, timeout=30)
    service = dict(line.split("=", 1) for line in properties.splitlines() if "=" in line)
    if (marker.is_symlink() or not marker.is_file()
            or marker.stat().st_uid != pwd.getpwnam("bruli").pw_uid):
        raise ValueError("The stress-filler pause marker has an unexpected identity")
    content = read_json(marker)
    if (content.get("paused") is not True
            or content.get("reason") != "User requested stopping GPU stress on s23-s26"
            or service["ActiveState"] != "inactive" or service["ConditionResult"] != "no"
            or service["User"] != "bruli"
            or f"argv[]=/usr/bin/python3 -B {ADAPTIVE_SCRIPT} supervisor ;" not in service["ExecStart"]):
        raise ValueError("Inactive stress filler does not match the explicit operator pause")
    return {
        "verified": True, "host": host, "path": str(marker), "sha256": file_sha(marker),
        "content": content, "service": service, "verified_at_unix": time.time(),
    }


def research_resources_released(release: dict) -> bool:
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
        and pause["content"].get("reason") == "User requested stopping GPU stress on s23-s26"
        and pause["service"]["ActiveState"] == "inactive"
        and pause["service"]["ConditionResult"] == "no"
        and pause["service"]["User"] == "bruli"
        and f"argv[]=/usr/bin/python3 -B {ADAPTIVE_SCRIPT} supervisor ;" in pause["service"]["ExecStart"]
    )


def stop_known(*, require_exclusive: bool) -> dict:
    require_research_node()
    state = audit()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "stop-known"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit() if args.action == "audit" else stop_known(require_exclusive=True)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
