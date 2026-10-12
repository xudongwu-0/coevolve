"""Frozen four-cell evaluation; independent process and GPU per cell."""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pipeline
from search_runtime import BASE_HARNESS, evaluate, service


def read(path):
    return json.loads(Path(path).read_text())


def evaluate_four_cells(*, run, config, split, base_url, initial, final, final_harness, resume=False,
                        remote_cells=None, main_cell="Wf_Hf"):
    hardware_path = run / "eval_hardware.json"
    hardware = read(hardware_path) if hardware_path.exists() else {}
    gpus = hardware.get("confirmation_gpus", [])
    assert len(gpus) == 3, "Configure three free GPUs for the frozen comparison"
    manifest_path = run / "frozen_eval_manifest.json"
    pipeline.write_json(manifest_path, config["data"])
    jobs = []
    try:
        cells = remote_cells if remote_cells is not None else [
            ("Wf_H0", final, BASE_HARNESS), ("W0_H0", initial, BASE_HARNESS),
            ("W0_Hf", initial, final_harness)]
        for index, (cell, identity, harness) in enumerate(cells):
            output = run / split / cell
            if (output / "val.json").exists():
                value = read(output / "val.json")
                assert value["metadata"]["weight_digest"] == identity["weight_digest"]
                assert value["metadata"]["harness_sha256"] == pipeline.sha(harness)
                continue
            work = run / "frozen_eval_workers" / split / cell
            work.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(pipeline.ROOT / "whale_local/search_evaluation.py"),
                "--run-dir", str(work), "--manifest", str(manifest_path), "--model", identity["weights"],
                "--harness", str(harness), "--split", split, "--gpu", str(gpus[index]),
                "--port", str(hardware["confirmation_port_base"] + index),
                "--parameter-dtype", identity.get("parameter_dtype", "bfloat16"),
                "--expected-digest", identity["weight_digest"]]
            if resume:
                command.append("--resume")
            with (work.parent / (cell + ".log")).open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((cell, identity, harness, work, output, process))
        output = run / split / main_cell
        if not (output / "val.json").exists():
            asyncio.run(evaluate(base_url=base_url, identity=final, harness=final_harness,
                data_entry=config["data"]["splits"][split], output=output, resume=resume))
        for cell, identity, harness, work, output, process in jobs:
            assert process.wait() == 0, f"Frozen evaluator failed: {cell}; inspect its log"
            assert read(work / "complete.json")["frozen_weights_verified"]
            value = read(work / "results/val.json")
            assert value["metadata"]["weight_digest"] == identity["weight_digest"]
            assert value["metadata"]["harness_sha256"] == pipeline.sha(harness)
            assert value["num_examples"] == config["data"]["splits"][split]["count"]
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(work / "results", output)
    finally:
        for *_, process in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=45)


def evaluate_gate_comparison(*, run, config, base_url, initial, final, final_harness, resume=False):
    reference = Path(config["gate_reference"])
    assert pipeline.sha(reference / "verification.json") == config["qualification_sha256"]
    ref = read(reference / "summary.json")
    assert ref["status"] == "complete" and not ref["gate_enabled"]
    assert ref["initial_identity"]["weight_digest"] == initial["weight_digest"]
    cells = [("initial", initial, BASE_HARNESS),
             ("whale", ref["final_identity"], Path(ref["final_harness"]))]
    frozen = {name: {"model_identity": identity, "harness_path": str(harness),
                    "harness_sha256": pipeline.sha(harness)}
              for name, identity, harness in cells + [("gate", final, final_harness)]}
    pipeline.write_json(run / "frozen_comparison.json", frozen)
    evaluate_four_cells(run=run, config=config, split="test", base_url=base_url,
        initial=initial, final=final, final_harness=final_harness, resume=resume,
        remote_cells=cells, main_cell="gate")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--split", choices=["confirmation", "test"], required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--parameter-dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run, manifest = args.run_dir.resolve(), read(args.manifest)
    run.mkdir(parents=True, exist_ok=args.resume)
    if (run / "complete.json").exists():
        return
    with service(run, args.model, args.gpu, args.port, manifest,
                 parameter_dtype=args.parameter_dtype) as (base_url, identity):
        assert identity["weight_digest"] == args.expected_digest
        pipeline.write_json(run / "identity.json", identity)
        if not (run / "results/val.json").exists():
            asyncio.run(evaluate(base_url=base_url, identity=identity, harness=args.harness,
                data_entry=manifest["splits"][args.split], output=run / "results", resume=args.resume))
        assert pipeline.api(base_url, "/health")["weight_digest"] == identity["weight_digest"]
    pipeline.write_json(run / "complete.json", {"frozen_weights_verified": True})


if __name__ == "__main__":
    main()
