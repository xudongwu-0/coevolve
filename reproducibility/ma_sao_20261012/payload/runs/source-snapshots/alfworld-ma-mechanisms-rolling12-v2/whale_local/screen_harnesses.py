"""Evaluate frozen models/harnesses on development data with native WHALE.

This runner never trains or selects using a test split. One sequential worker
owns its GPU and port, so observation-bound alias parsers cannot cross tasks.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--harness", type=Path, action="append", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    split = json.loads(args.manifest.read_text())["splits"]["mh_val"]
    assert pipeline.sha(split["path"]) == split["sha256"]
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    pipeline.POLICY = f"http://127.0.0.1:{args.port}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu,
               WHALE_RUN_ROOT=str(run), WHALE_MODEL_ROOT=str(args.model.resolve()),
               WHALE_POLICY_PORT=str(args.port), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONPATH=str(pipeline.ROOT / "whale_local") + ":" + str(pipeline.DOMAIN))
    pipeline.write_json(run / "config.json", {
        "purpose": "development screening only; no test results or training",
        "model": str(args.model.resolve()), "gpu": args.gpu, "data": split,
        "harnesses": [{"path": str(p.resolve()), "sha256": pipeline.sha(p)} for p in args.harness],
        "batch_concurrency": 1, "temperature": 0,
    })
    with (run / "worker.log").open("w") as log:
        worker = subprocess.Popen([sys.executable, "-u", str(pipeline.ROOT / "whale_local/worker.py")],
                                  env=env, stdout=log, stderr=subprocess.STDOUT)
        pipeline.write_json(run / "services.json", {"worker_pid": worker.pid})
        try:
            identity = pipeline.wait_ready(worker, pipeline.POLICY)
            pipeline.write_json(run / "identity.json", identity)
            from autoharness_chess_puzzle.runner import evaluate_harness
            scores = []
            for harness in args.harness:
                result = evaluate_harness(harness_path=harness, dataset_path=split["path"],
                    llm_config=pipeline.llm_config(identity), output_dir=run / harness.stem,
                    limit=split["count"], seed=42, assistant_token_budget=256, policy_max_tokens=128)
                scores.append({"harness": str(harness), "result": result})
                pipeline.write_json(run / "scores.json", scores)
                print(json.dumps(scores[-1]), flush=True)
            assert pipeline.api(pipeline.POLICY, "/health")["weight_digest"] == identity["weight_digest"]
            pipeline.write_json(run / "complete.json", {"frozen_weights_verified": True})
        finally:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            pipeline.write_json(run / "services_stopped.json", {"pid": worker.pid, "returncode": worker.returncode})


if __name__ == "__main__":
    main()
