"""Frozen four-cell evaluation; independent process and GPU per cell."""
import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pipeline
from math_runtime import BASE_HARNESS, evaluate, service


def read(path):
    return json.loads(Path(path).read_text())


def evaluate_four_cells(*, run, config, split, base_url, initial, final, final_harness, resume=False,
                        remote_cells=None, main_cell="Wf_Hf"):
    hardware_path = run / "eval_hardware.json"
    hardware = read(hardware_path) if hardware_path.exists() else {}
    cells = remote_cells if remote_cells is not None else [
        ("Wf_H0", final, BASE_HARNESS), ("W0_H0", initial, BASE_HARNESS),
        ("W0_Hf", initial, final_harness)]
    draws = config.get("evaluation_rollouts", 1)
    temperature = config.get("evaluation_temperature", 0.0)
    assert 1 <= draws <= 8 and 0 <= temperature <= 2
    expected_rows = config["data"]["splits"][split]["count"] * draws
    gpus = hardware.get("confirmation_gpus", [])
    assert len(gpus) == len(cells), "Configure one local child GPU for each additional frozen arm"
    manifest_path = run / "frozen_eval_manifest.json"
    pipeline.write_json(manifest_path, config["data"])
    jobs = []
    replica_groups = hardware.get("confirmation_replica_groups", [[] for _ in range(len(cells) + 1)])
    assert len(replica_groups) == len(cells) + 1
    allocated = [str(g) for g in gpus] + [str(config["gpu"])] + [str(g) for group in replica_groups for g in group]
    assert len(set(allocated)) == len(allocated), "Evaluation replicas must have exclusive GPUs"
    try:
        for index, (cell, identity, harness) in enumerate(cells):
            output = run / split / cell
            if (output / "val.json").exists():
                value = read(output / "val.json")
                assert value["metadata"]["weight_digest"] == identity["weight_digest"]
                assert value["metadata"]["harness_sha256"] == pipeline.sha(harness)
                assert value["num_examples"] == expected_rows
                assert value["metadata"]["temperature"] == temperature
                assert value["metadata"]["dataset_path"] == config["data"]["splits"][split]["path"]
                continue
            work = run / "frozen_eval_workers" / split / cell
            work.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(pipeline.ROOT / "whale_local/math_evaluation.py"),
                "--run-dir", str(work), "--manifest", str(manifest_path), "--model", identity["weights"],
                "--harness", str(harness), "--split", split, "--gpu", str(gpus[index]),
                "--port", str(hardware["confirmation_port_base"] + index),
                "--parameter-dtype", identity.get("parameter_dtype", "bfloat16"),
                "--expected-digest", identity["weight_digest"],
                "--evaluation-rollouts", str(draws), "--evaluation-temperature", str(temperature),
                "--replica-gpus", json.dumps(replica_groups[index]),
                "--replica-port-base", str(hardware["confirmation_replica_port_base"] + index*100)]
            if resume:
                command.append("--resume")
            with (work.parent / (cell + ".log")).open("a") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((cell, identity, harness, work, output, process))
        output = run / split / main_cell
        if not (output / "val.json").exists():
            from math_sampling import sample_parallel
            phase = run / "frozen_sampling" / split / main_cell
            phase.mkdir(parents=True, exist_ok=True)
            sample_parallel(run=run, phase=phase, base_url=base_url, identity=final,
                harness=final_harness, data_entry=config["data"]["splits"][split], output=output,
                rollouts=draws, seed=42, temperature=temperature, split=split, resume=resume,
                replica_gpus=replica_groups[-1],
                port_base=hardware["confirmation_replica_port_base"] + len(cells) * 100)
        main_value = read(output / "val.json")
        assert main_value["metadata"]["weight_digest"] == final["weight_digest"]
        assert main_value["metadata"]["harness_sha256"] == pipeline.sha(final_harness)
        assert main_value["num_examples"] == expected_rows
        assert main_value["metadata"]["temperature"] == temperature
        assert main_value["metadata"]["dataset_path"] == config["data"]["splits"][split]["path"]
        for cell, identity, harness, work, output, process in jobs:
            assert process.wait() == 0, f"Frozen evaluator failed: {cell}; inspect its log"
            assert read(work / "complete.json")["frozen_weights_verified"]
            value = read(work / "results/val.json")
            assert value["metadata"]["weight_digest"] == identity["weight_digest"]
            assert value["metadata"]["harness_sha256"] == pipeline.sha(harness)
            assert value["num_examples"] == expected_rows
            assert value["metadata"]["temperature"] == temperature
            assert value["metadata"]["dataset_path"] == config["data"]["splits"][split]["path"]
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
    reference_config = read(reference / "config.json")
    for key, default in [("evaluation_rollouts", 1), ("evaluation_temperature", 0.0)]:
        assert config.get(key, default) == reference_config.get(key, default)
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
    parser.add_argument("--evaluation-rollouts", type=int, default=1)
    parser.add_argument("--evaluation-temperature", type=float, default=0.0)
    parser.add_argument("--replica-gpus", default="[]")
    parser.add_argument("--replica-port-base", type=int, required=True)
    args = parser.parse_args()
    assert 1 <= args.evaluation_rollouts <= 8 and 0 <= args.evaluation_temperature <= 2
    run, manifest = args.run_dir.resolve(), read(args.manifest)
    run.mkdir(parents=True, exist_ok=args.resume)
    if (run / "complete.json").exists():
        cached = read(run / "results/val.json")
        assert cached["num_examples"] == manifest["splits"][args.split]["count"] * args.evaluation_rollouts
        assert cached["metadata"]["temperature"] == args.evaluation_temperature
        assert cached["metadata"]["weight_digest"] == args.expected_digest
        assert cached["metadata"]["harness_sha256"] == pipeline.sha(args.harness)
        assert cached["metadata"]["dataset_path"] == manifest["splits"][args.split]["path"]
        return
    pipeline.write_json(run / "config.json", {"data": manifest,
        "evaluation_rollouts": args.evaluation_rollouts, "evaluation_temperature": args.evaluation_temperature})
    pipeline.write_json(run / "eval_hardware.json", {"sampler_gpus": json.loads(args.replica_gpus),
        "sampler_port_base": args.replica_port_base})
    with service(run, args.model, args.gpu, args.port, manifest,
                 parameter_dtype=args.parameter_dtype) as (base_url, identity):
        assert identity["weight_digest"] == args.expected_digest
        pipeline.write_json(run / "identity.json", identity)
        if not (run / "results/val.json").exists():
            from math_sampling import sample_parallel
            sample_parallel(run=run, phase=run, base_url=base_url, identity=identity,
                harness=args.harness, data_entry=manifest["splits"][args.split], output=run / "results",
                rollouts=args.evaluation_rollouts, seed=42, temperature=args.evaluation_temperature,
                split=args.split, resume=args.resume,
                port_base=args.replica_port_base)
        assert pipeline.api(base_url, "/health")["weight_digest"] == identity["weight_digest"]
    pipeline.write_json(run / "complete.json", {"frozen_weights_verified": True})


if __name__ == "__main__":
    main()
