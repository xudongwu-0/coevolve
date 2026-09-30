"""Identity-checked, temporary stopping of the explicitly authorized filler."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .common import write_json

UNIT = "bruli-h200-load.service"
SCRIPT = "/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py"
RESEARCH_HOSTS = frozenset({"dev-NDH200-025", "dev-NDH200-026"})


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


def is_filler(record: dict, worker: bool = True) -> bool:
    cmd = record["cmd"]
    suffix = cmd[3:] if cmd[:3] == ["/usr/bin/python3", "-B", SCRIPT] else []
    valid = (
        len(suffix) == 3 and suffix[:2] == ["worker", "--gpu"]
        and suffix[2].isdigit() and 0 <= int(suffix[2]) < 8
    ) if worker else suffix == ["supervisor"]
    return valid and record["cgroup"].endswith("/system.slice/" + UNIT)


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
    records = [process_identity(int(pid)) for pid in pids]
    main = process_identity(int(unit["MainPID"])) if int(unit["MainPID"]) else None
    for record in records:
        record["verified_filler"] = record["uid"] == expected_uid and is_filler(record)
    return {
        "host": os.uname().nodename, "time": time.time(), "unit": unit,
        "supervisor": main, "processes": records,
        "verified_unit": bool(
            main and unit["User"] == "bruli" and main["uid"] == expected_uid
            and is_filler(main, worker=False)
        ),
    }


def stop_known(*, require_exclusive: bool) -> dict:
    require_research_node()
    state = audit()
    foreign = [p["pid"] for p in state["processes"] if not p["verified_filler"]]
    if require_exclusive and foreign:
        raise RuntimeError(f"Cannot allocate node; non-filler GPU processes: {foreign}")
    if state["unit"]["ActiveState"] == "active":
        if not state["verified_unit"]:
            raise RuntimeError("Active utilization unit identity did not match authorization")
        # Recheck the exact supervisor immediately before targeting its service.
        main = process_identity(state["supervisor"]["pid"])
        if main != state["supervisor"]:
            raise RuntimeError("Utilization supervisor changed during validation")
        subprocess.run(["sudo", "-n", "systemctl", "stop", UNIT], check=True, timeout=45)
        state["stopped"] = True
    else:
        state["stopped"] = False
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
