"""Generate packet-specific harnesses from a trained Engineer on all eight GPUs."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import threading
import time

from .alfworld_h_train import monitor_gpus
from .alfworld_phase_data import assignments_from_matrix, solver_dataset
from .common import read_json, write_json
from .launch import Launcher
from .resources import stop_known
from harness_r1_alfworld.packet_report import aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    run = Path(config["run"]).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "config.json", config)
    write_json(run / "resource_audit.json", stop_known(require_exclusive=True))
    launch = Launcher(run / "supervision")
    telemetry_stop = threading.Event()
    telemetry = threading.Thread(
        target=monitor_gpus,
        args=(run / "gpu_samples.jsonl", telemetry_stop,
              float(config.get("monitor_interval_seconds", 5))),
        daemon=True,
    )
    telemetry.start()
    started = time.time()
    try:
        with launch.vllm(
            config["engineer_model"], port=config.get("engineer_port", 19601),
            gpus="0,1,2,3", name=f"{config['phase']}_engineer",
            data_parallel=4, max_model_len=49152, tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton", language_model_only=True,
            max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as engineer_urls, launch.vllm(
            config["solver_model"], port=config.get("solver_port", 19701),
            gpus="4,5,6,7", name=f"{config['phase']}_solver",
            data_parallel=4, max_model_len=32768, tool_call_parser="qwen3_xml",
            gdn_prefill_backend="triton", language_model_only=True,
            max_num_seqs=32,
            server_args=["--enable-prefix-caching"],
        ) as solver_urls:
            engineer_ports = [
                int(url.rsplit(":", 1)[1].split("/", 1)[0])
                for url in engineer_urls.split(",")
            ]
            solver_ports = [
                int(url.rsplit(":", 1)[1].split("/", 1)[0])
                for url in solver_urls.split(",")
            ]
            starts = config.get("starts", list(range(0, 240, 10)))
            packet_workers = int(config.get("packet_workers", min(len(starts), 8)))
            if not 1 <= packet_workers <= len(starts):
                raise ValueError("packet_workers must fit the declared packet count")
            process = launch.module(
                "whale_local.harness_r1_alfworld.packet_probe",
                [
                    "--prefix", config["phase"],
                    "--root", run / "probe",
                    "--run-root", run / "driver_runs",
                    "--starts", ",".join(map(str, starts)),
                    "--size", config.get("size", 10),
                    "--concurrency", config.get("rollout_concurrency", 4),
                    "--target-ports", ",".join(map(str, solver_ports)),
                    "--engineer-ports", ",".join(map(str, engineer_ports)),
                    "--target-record", config["solver_receipt"],
                    "--target-label", config["solver_version"],
                    "--engineer-record", config["engineer_receipt"],
                    "--attempts", config.get("attempts", 2),
                    "--timeout-seconds", config.get("job_timeout_seconds", 10800),
                    "--workers", packet_workers,
                ],
                gpus="", name=f"{config['phase']}_probe",
            )
            launch.wait(
                process, f"{config['phase']}_probe",
                timeout=float(config.get("phase_timeout_seconds", 21600)),
            )
            probe = read_json(run / "probe/COMPLETE.json")
            report = aggregate([run / "probe"])
            write_json(run / "report.json", report)
            assignments, task_names = assignments_from_matrix(
                [run / "probe"], Path(config["no_patch"]), config["harness_version"],
            )
            write_json(run / "assignments.json", assignments)
            write_json(run / "task_names.json", {
                str(task_id): name for task_id, name in sorted(task_names.items())
            })
            solver_data_output = config.get("solver_data_output")
            solver_rows = None
            if solver_data_output:
                solver_rows = solver_dataset(
                    sorted(task_names),
                    int(config["solver_data_count"]),
                    int(config.get("solver_data_offset", 0)),
                    config["harness_version"],
                    task_names,
                )
                write_json(Path(solver_data_output), solver_rows)
            write_json(run / "COMPLETE.json", {
                "status": "completed", "phase": config["phase"],
                "elapsed_seconds": time.time() - started,
                "packets": probe["completed"], "assignments": len(assignments),
                "task_count": len(task_names), "harness_version": config["harness_version"],
                "baseline_rate": report["baseline_rate"],
                "patched_rate": report["patched_rate"],
                "delta_rate": report["delta_rate"],
                "engineer_model": config["engineer_model"],
                "solver_model": config["solver_model"],
                "solver_data_output": solver_data_output,
                "solver_data_rows": len(solver_rows) if solver_rows is not None else None,
            })
    except BaseException as exc:
        write_json(run / "FAILED.json", {
            "status": "failed", "phase": config.get("phase"),
            "elapsed_seconds": time.time() - started, "error": repr(exc),
        })
        raise
    finally:
        launch.close()
        telemetry_stop.set()
        telemetry.join(timeout=10)
        write_json(run / "RELEASED.json", {
            "released_at": time.time(),
            "remaining_gpu_pids": subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid",
                 "--format=csv,noheader,nounits"], text=True,
            ).split(),
        })


if __name__ == "__main__":
    main()
