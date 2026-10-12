"""Run concurrent matched SFT-initialized H-W-H arms and persist paired results."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time

from .analysis import (
    compare_runs, difference_in_differences, expected_sample_ledger, paired_difference,
    stage_reports, verify_phases, verify_sample_ledger,
)
from .budget import initial_model_digest
from .common import file_sha, read_json, write_json
from .launch import Launcher
from .resources import require_research_node, stop_known


def trial_spec(protocol: dict, host: str) -> dict:
    if host not in protocol["nodes"]:
        raise ValueError(f"Host is not assigned to this campaign: {host}")
    receipt = read_json(Path(protocol["initialization_record"]))
    if (receipt["status"] != "completed" or not receipt["hf_cpu_reload_verified"]
            or receipt["checkpoint"] != protocol["model"]
            or receipt["cumulative_train_records"] != 1152):
        raise ValueError("Campaign must start from the verified latest1152-record SFT checkpoint")
    node = protocol["nodes"][host]
    spec = {
        **protocol["common"], "model": protocol["model"], "data": protocol["data"],
        "seed": protocol["seed"], "method": node["method"],
        "run": str(Path(node["root"]) / "hwh"),
    }
    if spec["method"] == "ma_sao":
        spec.update(protocol["algorithm"])
        warm = (spec["critic_warmup_steps"] * spec["solver_batch_size"]
                * spec["warmup_recycled_per_row"])
        mixed = int(spec["solver_steps"] * spec["solver_batch_size"]
                    * spec["solver_recycle_fraction"])
        pool = ((spec["editor_steps"] + 1) * spec["editor_batch_size"]
                * 2 * spec["reward_questions"])
        if warm + mixed > pool or spec["solver_recycle_fraction"] != 1.0:
            raise ValueError("Unique partner pool cannot cover disjoint warmup and uniform mixing")
    elif spec["method"] != "independent":
        raise ValueError("Unknown paired method")
    if (spec["cycles"] != 1 or spec["proposal_source"] != "final_policy"
            or spec["evaluate_test"] or spec["formal_baseline"]):
        raise ValueError("This concurrent H-W-H diagnostic must keep AIME sealed")
    return spec


def verify_unique_recycling(run: Path, spec: dict) -> dict:
    rows = read_json(run / "w1_solver_data.json")
    warm = [eid for row in rows if row.get("warmup") for eid in row["recycled_ids"]]
    mixed = [eid for row in rows if not row.get("warmup") for eid in row["recycled_ids"]]
    pool_ids = [episode["id"] for episode in read_json(run / "w1_recycle.json")]
    expected_warm = (spec["critic_warmup_steps"] * spec["solver_batch_size"]
                     * spec["warmup_recycled_per_row"])
    expected_mixed = int(spec["solver_steps"] * spec["solver_batch_size"]
                         * spec["solver_recycle_fraction"])
    if (len(warm) != expected_warm or len(mixed) != expected_mixed
            or len(set(warm + mixed)) != len(warm + mixed)
            or len(set(pool_ids)) != len(pool_ids) or set(warm + mixed) != set(pool_ids)):
        raise ValueError("MA recycling must be disjoint, complete and once-per-episode")
    receipt = read_json(run / "w1/trained.json")
    if (receipt["actor_steps"] != spec["solver_steps"]
            or receipt["critic_only_warmup_steps"] != spec["critic_warmup_steps"]):
        raise ValueError("Actor/value update counts differ from the declared v6 protocol")
    return {"passed": True, "warmup_unique_episodes": len(warm),
            "actor_mixed_unique_episodes": len(mixed), "overlap": 0, "max_reuse": 1}


def publish(spec: dict, root: Path, identity: dict) -> None:
    run = Path(spec["run"])
    complete = read_json(run / "COMPLETE.json")
    if (read_json(run / "STATUS.json")["status"] != "training_frozen"
            or complete["method"] != spec["method"] or complete["test_evaluated"]):
        raise ValueError("Cannot publish an incomplete or protocol-mismatched result")
    ledger = verify_sample_ledger(run)
    write_json(run / "checkpoint_verification.json", verify_phases(run))
    files = ["COMPLETE.json", "FROZEN.json", "STATUS.json", "spec.json",
             "checkpoint_verification.json", "sample_budget_verification.json"]
    if spec["method"] == "ma_sao":
        write_json(run / "unique_recycling.json", verify_unique_recycling(run, spec))
        files.append("unique_recycling.json")
    reports = stage_reports(run)
    if set(reports) != {"h0w0", "h1w0", "h1w1", "h2w1"}:
        raise ValueError("Missing H-W-H stage reports")
    measurements = {
        "stages": {name: {k: v for k, v in read_json(path).items() if k != "results"}
                   for name, path in reports.items()},
        "final_minus_initial": paired_difference(
            read_json(reports["h0w0"]), read_json(reports["h2w1"])),
        "physical_sample_ledger": ledger,
        "sample_budget": read_json(run / "sample_budget_verification.json"),
        "elapsed_seconds": complete["elapsed_seconds"],
    }
    write_json(run / "measurements.json", measurements)
    files.extend(["measurements.json", *(p.name for p in reports.values()),
                  *(p.name for p in run.glob("h*_selection.json"))])
    for name in files:
        (run / name).chmod(0o644)
    run.chmod(0o755)
    write_json(root / "READY.json", {
        **identity, "run": str(run), "seed": spec["seed"], "method": spec["method"],
        "reports": {name: file_sha(run / name) for name in files},
    })
    (root / "READY.json").chmod(0o644)


def verify_peer(marker: Path, identity: dict, seed: int) -> dict:
    peer = read_json(marker)
    if peer["seed"] != seed or any(peer[key] != value for key, value in identity.items()):
        raise ValueError("Peer initialization/data/protocol/seed differs")
    run = Path(peer["run"])
    for name, digest in peer["reports"].items():
        if Path(name).name != name or file_sha(run / name) != digest:
            raise ValueError(f"Peer report changed: {name}")
    return peer


def worker(path: Path, resume: bool = False):
    require_research_node()
    protocol = read_json(path)
    host = os.uname().nodename
    spec = trial_spec(protocol, host)
    node = protocol["nodes"][host]
    root = Path(node["root"])
    root.mkdir(parents=True, exist_ok=True)
    if root.stat().st_uid != os.getuid():
        raise PermissionError("Each host needs its own user-owned output directory")
    root.chmod(0o755)
    if resume:
        previous = read_json(root / "WORKER_STATUS.json")
        claim = read_json(root / "worker_claim.json")
        if previous["phase"] != "failed" or claim["host"] != host:
            raise ValueError("Resume requires a failed worker on this same host")
        try:
            os.kill(claim["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Previous worker PID still exists; refusing concurrent recovery")
        archive = root / f"recovery_{time.time_ns()}"
        archive.mkdir()
        for name in ("worker_claim.json", "WORKER_STATUS.json", "trial.json"):
            shutil.copy2(root / name, archive / name)
        (root / "worker_claim.json").unlink()
    with (root / "worker_claim.json").open("x") as stream:
        json.dump({"pid": os.getpid(), "host": host, "protocol_sha256": file_sha(path)}, stream)
    write_json(root / "protocol.json", protocol)
    write_json(root / "trial.json", spec)
    write_json(root / "planned_samples.json", expected_sample_ledger(spec))
    launcher = None

    def status(phase, **extra):
        write_json(root / "WORKER_STATUS.json", {
            "host": host, "phase": phase, "method": spec["method"], "run": spec["run"],
            "updated_at": time.time(), **extra,
        })
        print(host, phase, extra, flush=True)

    try:
        status("verifying_initialization")
        manifest = read_json(Path(spec["data"]) / "manifest.json")
        for split in ("train", "reward", "dev"):
            if file_sha(Path(spec["data"]) / f"{split}.json") != manifest["splits"][split]["sha256"]:
                raise ValueError(f"Data hash mismatch: {split}")
        identity = {
            "protocol_sha256": file_sha(path),
            "initial_actor_sha256": initial_model_digest(Path(spec["model"])),
            "data_manifest_sha256": file_sha(Path(spec["data"]) / "manifest.json"),
        }
        write_json(root / "identity.json", identity)
        status("allocating")
        stop_known(require_exclusive=True)
        launcher = Launcher(root / "supervision")
        arguments = ["--spec", root / "trial.json"] + (["--resume"] if resume else [])
        process = launcher.module("sao_bridge.experiment", arguments,
                                  gpus="0,1,2,3,4,5,6,7", name="hwh")
        status("running")
        launcher.wait(process, "H-W-H experiment", timeout=protocol["budget_hours_per_run"] * 3600)
        publish(spec, root, identity)
        status("training_completed", initial_and_stage_results=str(Path(spec["run"]) / "measurements.json"))
    except BaseException as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if launcher:
            launcher.close()
        pids = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        if not pids:
            subprocess.run(["sudo", "-n", "systemctl", "start", "bruli-h200-load.service"],
                           check=True, timeout=45)
        else:
            print(f"GPU processes remain; filler restoration skipped: {pids}", flush=True)

    marker = Path(node["peer_root"]) / "READY.json"
    status("waiting_for_peer_results")
    deadline = time.monotonic() + protocol["budget_hours_per_run"] * 3600
    while not marker.is_file():
        peer_status = marker.parent / "WORKER_STATUS.json"
        if peer_status.is_file() and read_json(peer_status)["phase"] == "failed":
            status("peer_failed", own_training_completed=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Peer did not publish results; own measurements are preserved")
        time.sleep(15)
    peer = verify_peer(marker, identity, spec["seed"])
    own = Path(spec["run"])
    control, candidate = ((own, Path(peer["run"])) if spec["method"] == "independent"
                          else (Path(peer["run"]), own))
    result = compare_runs(control, candidate)
    left, right = stage_reports(control), stage_reports(candidate)
    result["difference_in_differences"] = difference_in_differences(
        read_json(left["h0w0"]), read_json(left["h2w1"]),
        read_json(right["h0w0"]), read_json(right["h2w1"]),
    )
    result["interpretation"] = "Single-seed cross-host matched diagnostic, not seed-robust superiority."
    result["physical_costs"] = {
        method: read_json(run / "measurements.json")["physical_sample_ledger"]
        for method, run in (("sao", control), ("ma_sao", candidate))
    }
    result["ma_sample_budget"] = read_json(candidate / "measurements.json").get("sample_budget")
    write_json(root / "PAIRED_COMPARISON.json", result)
    status("completed", comparison=str(root / "PAIRED_COMPARISON.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise InterruptedError(f"Campaign worker received signal {signum}")

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    worker(args.protocol.resolve(), resume=args.resume)


if __name__ == "__main__":
    main()
