"""Bounded node-local H->W experiment using two independent upstream SAO trainers."""
from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .common import (
    INITIAL_SOURCE, file_sha, harness_spec, prepare_data, read_json,
    verify_upstreams, write_json,
)


def assert_idle(gpus: list[int]) -> None:
    result = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True,
    )
    found = {}
    for index, memory, util in csv.reader(io.StringIO(result)):
        found[int(index)] = (int(memory), int(util))
    for gpu in gpus:
        if gpu not in found or found[gpu][0] > 1024 or found[gpu][1] > 5:
            raise RuntimeError(f"GPU {gpu} is absent or not idle: {found.get(gpu)}")
    process_csv = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    if process_csv:
        raise RuntimeError("Node has active compute processes; allocate an idle node for this pilot")


def main():
    import httpx
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--train-gpus", default="0,1")
    parser.add_argument("--eval-gpu", type=int, default=2)
    parser.add_argument("--propose-gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=18641)
    parser.add_argument("--proposal-count", type=int, default=16)
    parser.add_argument("--proposal-temperature", type=float, default=0.3)
    args = parser.parse_args()
    upstream = verify_upstreams()
    train_gpus = [int(n) for n in args.train_gpus.split(",")]
    all_gpus = train_gpus + [args.eval_gpu, args.propose_gpu]
    if len(train_gpus) != 2 or len(set(all_gpus)) != 4 or args.steps < 2:
        parser.error("Choose four distinct GPU IDs and at least two update steps")
    if not 1 <= args.proposal_count <= 32 or not 0 <= args.proposal_temperature <= 2:
        parser.error("Use 1-32 proposals and temperature between 0 and 2")
    if args.run.exists():
        raise FileExistsError(f"Use a fresh run directory: {args.run}")
    assert_idle(all_gpus)
    args.run.mkdir(parents=True, mode=0o700)
    prepare_data(args.source_data, args.run / "data", 16, 4, 8)
    harness = args.run / "initial_harness.py"
    harness.write_text(INITIAL_SOURCE)
    write_json(args.run / "manifest.json", {
        "host": socket.gethostname(), "pid": os.getpid(), "upstream": upstream,
        "model": str(args.model.resolve()), "gpus": all_gpus,
        "mode": "independent SAO H->W, not MAPPO", "actor_steps_per_role": args.steps,
        "harness_scope": "literal prompts and MAX_TURNS; fixed text-only recovery",
        "code_execution": False, "external_model_api": False,
        "proposal_count": args.proposal_count,
        "proposal_temperature": args.proposal_temperature,
        "adapter_sha256": {
            path.name: file_sha(path) for path in Path(__file__).parent.glob("*.py")
        },
    })
    processes = []
    environment = dict(os.environ, PYTHONUNBUFFERED="1", SAO_CRITIC_K="2",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                       OMP_NUM_THREADS="4", LITELLM_LOCAL_MODEL_COST_MAP="True",
                       VLLM_PLUGINS="")

    def launch(module, arguments, gpus, name):
        command = [sys.executable, "-B", "-m", f"sao_bridge.{module}", *map(str, arguments)]
        print(f"{name}: starting on GPU(s) {gpus}; log={args.run / (name + '.log')}",
              flush=True)
        log = (args.run / f"{name}.log").open("wb")
        try:
            process = subprocess.Popen(
                command, env=dict(environment, CUDA_VISIBLE_DEVICES=str(gpus)),
                stdout=log, stderr=subprocess.STDOUT,
            )
        finally:
            log.close()
        processes.append({"pid": process.pid, "host": socket.gethostname(),
                          "command": command, "gpus": str(gpus), "log": f"{name}.log"})
        write_json(args.run / "processes.json", processes)
        return process

    def run(module, arguments, gpus, name):
        process = launch(module, arguments, gpus, name)
        code = process.wait()
        if code != 0:
            raise RuntimeError(f"{name} failed ({code}); see {args.run / (name + '.log')}")
        print(f"{name}: finished", flush=True)

    def evaluate(model, source, name):
        run("service", ["--model", model, "--data", args.run / "data/test.json",
                       "--harness", source, "--evaluate-output", args.run / f"{name}.json"],
            args.eval_gpu, name)

    evaluate(args.model, harness, "before")
    server = launch(
        "service", ["--model", args.model, "--data", args.run / "data/dev.json",
                    "--harness", harness, "--port", args.port],
        args.eval_gpu, "frozen_solver",
    )
    url = f"http://127.0.0.1:{args.port}"
    selected_path = args.run / "selected_harness.py"
    try:
        with httpx.Client(timeout=600, trust_env=False) as client:
            for _ in range(180):
                if server.poll() is not None:
                    raise RuntimeError("Frozen solver failed to start; see frozen_solver.log")
                try:
                    response = client.get(url + "/health", timeout=2)
                    response.raise_for_status()
                    if response.json()["ready"]:
                        break
                except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                    time.sleep(1)
            else:
                raise TimeoutError("Frozen solver did not become ready")
            baseline = client.post(url + "/score", json={"source": INITIAL_SOURCE})
            baseline.raise_for_status()
            feedback = (
                f"Development accuracy is {baseline.json()['accuracy']:.3f}. "
                "Improve general accuracy and complete answers within the fixed budget."
            )
            run("train", ["--role", "editor", "--model", args.model,
                          "--run", args.run, "--harness", harness,
                          "--steps", args.steps, "--evaluator-url", url,
                          "--feedback", feedback], args.train_gpus, "editor_train")
            editor = read_json(args.run / "editor_trained.json")
            run("propose", ["--model", editor["checkpoints"]["actor"],
                            "--output", args.run / "proposals.json",
                            "--count", args.proposal_count,
                            "--temperature", args.proposal_temperature],
                args.propose_gpu, "editor_proposals")
            proposals = read_json(args.run / "proposals.json")
            candidates = [{"source": INITIAL_SOURCE, **baseline.json(), "incumbent": True}]
            for row in proposals:
                if not row["valid"]:
                    continue
                response = client.post(url + "/score", json={"source": row["source"]})
                response.raise_for_status()
                candidates.append({**response.json(), "incumbent": False})
            if len(candidates) == 1:
                raise RuntimeError("Updated editor produced no valid harness; see proposals.json")
            chosen = max(candidates, key=lambda item: item["accuracy"])
            selected_path.write_text(harness_spec(chosen["source"])["source"])
            write_json(args.run / "selection.json",
                       {"candidates": candidates, "selected": chosen, "selection_split": "dev"})
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
    run("train", ["--role", "solver", "--model", args.model, "--run", args.run,
                  "--harness", selected_path, "--steps", args.steps],
        args.train_gpus, "solver_train")
    solver = read_json(args.run / "solver_trained.json")
    evaluate(solver["checkpoints"]["actor"], selected_path, "after")
    from .verify import verify
    checkpoint_checks = verify(args.run)
    write_json(args.run / "COMPLETE.json", {
        "editor": read_json(args.run / "editor_trained.json"), "solver": solver,
        "before": read_json(args.run / "before.json")["accuracy"],
        "after": read_json(args.run / "after.json")["accuracy"],
        "selected_harness_sha256": file_sha(selected_path),
        "checkpoint_changes_verified": checkpoint_checks,
        "test_count": 8, "claim": "integration smoke test, not a performance conclusion",
    })


if __name__ == "__main__":
    main()
