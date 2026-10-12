"""Parallel native rollouts with fixed per-trajectory seeds and exact receipts."""
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pipeline
from math_runtime import evaluate


def read(path):
    return json.loads(Path(path).read_text())


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def sample_parallel(*, run, phase, output, base_url, identity, harness, data_entry,
                    rollouts=4, seed=43, schedule=None, resume=False,
                    temperature=0.8, split="train", replica_gpus=None, port_base=None):
    hardware = read(run / "eval_hardware.json")
    config = read(run / "config.json")
    manifest = config["data"]
    assert data_entry == manifest["splits"][split]
    gpus = hardware["sampler_gpus"] if replica_gpus is None else replica_gpus
    plan = ([[draw, index] for draw in range(rollouts) for index in range(data_entry["count"])]
            if schedule is None else schedule)
    assert plan and len({tuple(item) for item in plan}) == len(plan)
    width = len(gpus) + 1
    partitions = [plan[offset::width] for offset in range(width)]
    workspace = phase / ("sampling_" + output.name)
    workspace.mkdir(exist_ok=resume)
    manifest_path = workspace / "manifest.json"
    pipeline.write_json(manifest_path, manifest)
    jobs = []
    try:
        for offset, gpu in enumerate(gpus, 1):
            if not partitions[offset]:
                continue
            work = workspace / f"worker_{offset}"
            plan_path = workspace / f"plan_{offset}.json"
            pipeline.write_json(plan_path, partitions[offset])
            # Sequential phases finish and release these ports before reuse.
            stage_offset = {"rollouts": 0, "initial": 100, "followup": 200}.get(output.name, 0)
            port = (hardware["sampler_port_base"] + seed * 10 + stage_offset
                    if port_base is None else port_base) + offset
            command = [sys.executable, "-u", str(pipeline.ROOT / "whale_local/math_runtime.py"),
                "--run-dir", str(work), "--manifest", str(manifest_path), "--model", identity["weights"],
                "--harness", str(harness), "--gpu", str(gpu), "--port", str(port), "--split", split,
                "--schedule", str(plan_path), "--output-name", "rollouts", "--temperature", str(temperature),
                "--seed", str(seed), "--parameter-dtype", identity.get("parameter_dtype", "bfloat16")]
            if resume:
                command.append("--resume")
            with (workspace / f"worker_{offset}.log").open("a") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((work, child))
        local = workspace / "main"
        if not (local / "val.json").exists():
            asyncio.run(evaluate(base_url=base_url, identity=identity, harness=harness,
                data_entry=data_entry, output=local, temperature=temperature, seed=seed,
                schedule=partitions[0], resume=resume))
        rows = jsonl(local / "trajectories.jsonl")
        for work, child in jobs:
            assert child.wait() == 0, f"Sampling worker failed: {work}"
            assert read(work / "complete.json")["frozen_weights_verified"]
            worker_rows = jsonl(work / "rollouts/trajectories.jsonl")
            for row in worker_rows:
                assert row["sampling_identity"]["weight_digest"] == identity["weight_digest"]
                assert row["harness_sha256"] == pipeline.sha(harness)
                if row.get("context_rejection_id"):
                    name = row["context_rejection_id"] + ".json"
                    source, target = work / "context_rejections" / name, run / "context_rejections" / name
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():
                        assert target.read_bytes() == source.read_bytes()
                    else:
                        shutil.copy2(source, target)
                for call in row["policy_calls"]:
                    source = work / "calls" / (call["call_id"] + ".json")
                    assert read(source) == call
                    target = run / "calls" / source.name
                    target.parent.mkdir(exist_ok=True)
                    if target.exists():
                        assert target.read_bytes() == source.read_bytes()
                    else:
                        shutil.copy2(source, target)
            rows.extend(worker_rows)
        by_key = {(r["draw"], r["example_index"]): r for r in rows}
        assert len(by_key) == len(rows) == len(plan)
        rows = [by_key[tuple(item)] for item in plan]
        assert pipeline.api(base_url, "/health")["weight_digest"] == identity["weight_digest"]
        output.mkdir(parents=True, exist_ok=resume)
        text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        path = output / "trajectories.jsonl"
        if path.exists():
            assert path.read_text() == text
        else:
            temp = path.with_suffix(".jsonl.tmp")
            temp.write_text(text)
            temp.replace(path)
        value = {"num_examples": len(rows), "solved_examples": int(sum(r["reward"] for r in rows)),
            "success_rate": sum(r["reward"] for r in rows) / len(rows),
            "avg_metrics": {key: sum(r["metrics"].get(key, 0) for r in rows) / len(rows)
                            for key in rows[0]["metrics"]},
            "mean_turn_count": sum(len(r["policy_calls"]) for r in rows) / len(rows),
            "mean_assistant_tokens": sum(r["assistant_tokens"] for r in rows) / len(rows),
            "metadata": {"dataset_path": data_entry["path"], "seed": seed, "temperature": temperature,
                "weight_digest": identity["weight_digest"], "harness_sha256": pipeline.sha(harness),
                "harness_path": str(harness), "sampling_schedule": plan,
                "assistant_token_budget": 2048, "policy_max_tokens": 1024,
                "frozen_worker_gpus": gpus, "receipts_copied_without_changes": True}}
        pipeline.write_json(output / "val.json", value)
        return value
    finally:
        for _, child in jobs:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=45)
