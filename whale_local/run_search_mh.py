"""Native SearchQA MH with local policy and active-agent proposer transports."""
import asyncio
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from meta_harness import meta_harness_search_r1 as mh
import pipeline
from search_runtime import evaluate


def local_sweep(config, harnesses, logs_dir, **kwargs):
    local = config["local"]
    identity = pipeline.api(local["policy_base"], "/health")
    assert identity["weight_digest"] == local["weight_digest"]
    assert len(config["models"]) == len(config["task_profiles"]) == len(config["seeds"]) == 1
    assert config["seeds"] == [42]
    root_run = logs_dir.parents[2]
    hardware_path = root_run / "eval_hardware.json"
    if len(harnesses) > 1 and hardware_path.exists():
        hardware = json.loads(hardware_path.read_text())
        gpus = hardware["candidate_gpus"]
        assert len(gpus) >= len(harnesses)
        manifest = json.loads((root_run / "config.json").read_text())["data"]
        manifest_path = root_run / "eval_worker_manifest.json"
        pipeline.write_json(manifest_path, manifest)
        round_number = int(logs_dir.parent.name.rsplit("_", 1)[1])
        jobs = []
        try:
            for index, (name, _, path) in enumerate(harnesses):
                output = logs_dir / "mh_val" / name / (identity["model"].lower() + "_seed42")
                if (output / "val.json").exists():
                    continue
                work = root_run / "eval_workers" / logs_dir.parent.name / name
                if not (work / "complete.json").exists():
                    port = hardware["port_base"] + round_number * 20 + int(name[1:])
                    log_path = work.parent / (name + ".log")
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("a") as handle:
                        child = subprocess.Popen([sys.executable, "-u", str(pipeline.ROOT / "whale_local/search_runtime.py"),
                            "--run-dir", str(work), "--manifest", str(manifest_path), "--model", identity["weights"],
                            "--harness", str(path), "--gpu", str(gpus[index]), "--port", str(port)],
                            stdout=handle, stderr=subprocess.STDOUT)
                    jobs.append((name, path, work, output, child))
                else:
                    jobs.append((name, path, work, output, None))
            for name, path, work, output, child in jobs:
                if child is not None and child.wait():
                    raise RuntimeError(f"Parallel MH evaluator {name} failed")
                assert json.loads((work / "complete.json").read_text())["frozen_weights_verified"]
                value = json.loads((work / path.stem / "val.json").read_text())
                assert value["metadata"]["weight_digest"] == identity["weight_digest"]
                assert value["metadata"]["harness_sha256"] == pipeline.sha(path)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(work / path.stem, output)
            return [(name, True) for name, _, _ in harnesses]
        finally:
            for _, _, _, _, child in jobs:
                if child is not None and child.poll() is None:
                    child.terminate()
                    child.wait(timeout=30)
    results = []
    for name, _, path in harnesses:
        output = logs_dir / "mh_val" / name / (identity["model"].lower() + "_seed42")
        if not (output / "val.json").exists():
            asyncio.run(evaluate(base_url=local["policy_base"], identity=identity, harness=path,
                data_entry=local["data_entry"], output=output))
        result = json.loads((output / "val.json").read_text())
        assert result["metadata"]["weight_digest"] == identity["weight_digest"]
        assert result["metadata"]["harness_sha256"] == pipeline.sha(path)
        results.append((name, True))
    return results


def local_propose(*, task_prompt, iteration, run_dir, timeout_seconds, **kwargs):
    root_config = json.loads((run_dir.parents[1] / "config.json").read_text())
    expected = re.findall(r"harnesses/(h\d+)/harness\.py", task_prompt)
    if root_config.get("gate_proposal_mode") == "reference_pool":
        # Controlled sampling-gate comparison: exactly the reference's proposal
        # sources, freshly scored under this arm's weights by native MH.
        reference = Path(root_config["gate_reference"]) / "mh" / run_dir.name
        metadata = []
        for name in expected:
            source = reference / "harnesses" / name / "harness.py"
            values = list((reference / "logs/mh_val" / name).rglob("val.json"))
            assert len(values) == 1
            assert json.loads(values[0].read_text())["metadata"]["harness_sha256"] == pipeline.sha(source)
            directory = run_dir / "harnesses" / name
            directory.mkdir(exist_ok=True)
            (directory / "harness.py").write_text(source.read_text())
            metadata.append({"name": name, "axis": "shared-reference-pool",
                "hypothesis": "Evaluate the exact same external-proposer candidate as the qualified reference, under this arm's updated weights.",
                "changes": "Frozen candidate source copied without modification; only sampling allocation differs between arms.",
                "components": ["matched-proposal-pool"], "reference_source": str(source),
                "reference_sha256": pipeline.sha(source)})
        pipeline.write_json(run_dir / "pending_eval.json", {"candidates": metadata})
        return SimpleNamespace(exit_code=0, text="frozen shared external-proposer pool", stderr="", show=lambda: None)
    accepted = mh.get_accepted_harness(run_dir)
    payload = {"task_prompt": task_prompt, "run_dir": str(run_dir), "iteration": iteration,
        "accepted_harness_source": (run_dir / "harnesses" / accepted / "harness.py").read_text(),
        "frontier": json.loads((run_dir / "logs/frontier_val.json").read_text())}
    context_path = run_dir.parents[1] / run_dir.name / "gate/gate_harness_context.json"
    if context_path.exists():
        payload["gate_context"] = json.loads(context_path.read_text())
        payload["task_prompt"] += ("\nAdditional TRAIN failure-routing predictions are supplied in gate_context. "
            "Use them as hypotheses when designing candidates, alongside current development failures. "
            "They are not ground-truth causes and must not change the verifier or the search budget.")
    base = "http://127.0.0.1:" + os.environ["WHALE_PROPOSER_PORT"]
    job = pipeline.api(base, "/jobs", payload)
    print(f"Active Codex proposer job: {job['job_id']}", flush=True)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = pipeline.api(base, "/jobs/" + job["job_id"])
        if "candidates" in result:
            break
        time.sleep(1)
    else:
        raise TimeoutError("Active proposer did not respond")
    assert [c["name"] for c in result["candidates"]] == expected
    metadata = []
    for candidate in result["candidates"]:
        directory = run_dir / "harnesses" / candidate["name"]
        directory.mkdir(exist_ok=True)
        (directory / "harness.py").write_text(candidate["source"])
        metadata.append({k: v for k, v in candidate.items() if k != "source"})
    pipeline.write_json(run_dir / "pending_eval.json", {"candidates": metadata})
    return SimpleNamespace(exit_code=0, text="active Codex via local FastAPI", stderr="", show=lambda: None)


mh.RUNS_DIR = Path(os.environ["WHALE_MH_RUNS"])
mh.run_sweep = local_sweep
mh.propose_claude = local_propose

if __name__ == "__main__":
    mh.main()
