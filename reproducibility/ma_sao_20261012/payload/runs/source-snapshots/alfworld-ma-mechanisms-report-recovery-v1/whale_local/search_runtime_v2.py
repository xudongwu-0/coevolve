"""Execute the native WHALE SearchR1Env against a recording local worker."""
import argparse
import asyncio
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess

import httpx
from openai import AsyncOpenAI
import pandas as pd
from verifiers.legacy.clients.openai_chat_completions_client import OpenAIChatCompletionsClient
from verifiers.legacy.types import ResponseTokens

from environments.search_r1.env import load_environment, _last_assistant_content
from environments.search_r1 import search_r1_env
import pipeline

ROOT = pipeline.ROOT
DOMAIN = ROOT / "WHALE/domains/search_qa"
BASE_HARNESS = DOMAIN / "environments/search_r1/base_harness.py"


class RecordingClient(OpenAIChatCompletionsClient):
    def __init__(self, base_url):
        super().__init__(AsyncOpenAI(base_url=base_url + "/v1", api_key="local",
            http_client=httpx.AsyncClient(trust_env=False), max_retries=0, timeout=180))
        self.calls = []
        self.context_rejection_id = None

    async def get_native_response(self, *args, **kwargs):
        try:
            response = await super().get_native_response(*args, **kwargs)
        except Exception as exc:
            current = exc
            while current is not None:
                body = getattr(current, "body", None)
                if isinstance(body, dict):
                    detail = body.get("error", body)
                    if detail.get("code") == "context_length_exceeded":
                        self.context_rejection_id = detail["rejection_id"]
                current = current.__cause__
            raise
        record = response.model_extra["whale_record"]
        assert response.id == record["call_id"]
        self.calls.append(record)
        return response

    async def from_native_response(self, response):
        converted = await super().from_native_response(response)
        record = response.model_extra["whale_record"]
        converted.message.tokens = ResponseTokens(prompt_ids=record["prompt_ids"],
            prompt_mask=[0] * len(record["prompt_ids"]), completion_ids=record["response_ids"],
            completion_mask=[1] * len(record["response_ids"]),
            completion_logprobs=record["response_logprobs"])
        return converted


@contextmanager
def service(run, model, gpu, port, manifest, learning_rate=1e-6, parameter_dtype=None):
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    if str(gpu).startswith("node1:"):
        from search_remote_v2 import service as remote_service
        with remote_service(run, model, gpu, port, manifest, learning_rate, parameter_dtype or "float32") as pair:
            yield pair
        return
    # Native harness tools use httpx's default environment handling.
    # Keep their loopback traffic out of the user's outbound proxy.
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    run = Path(run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    base_url = f"http://127.0.0.1:{port}"
    assert pipeline.sha(manifest["corpora"]["path"]) == manifest["corpora"]["sha256"]
    if parameter_dtype is None:
        saved = Path(model) / "training_state.json"
        parameter_dtype = json.loads(saved.read_text()).get("parameter_dtype", "bfloat16") if saved.exists() else "float32"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), WHALE_RUN_ROOT=str(run),
        WHALE_MODEL_ROOT=str(model), WHALE_POLICY_PORT=str(port),
        WHALE_SEARCH_CORPORA=manifest["corpora"]["path"], WHALE_LEARNING_RATE=str(learning_rate),
        WHALE_PARAMETER_DTYPE=parameter_dtype,
        WHALE_CHECKPOINT_ALLOWLIST=os.environ.get("WHALE_CHECKPOINT_ALLOWLIST", "[]"),
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONPATH=":".join(str(p) for p in [ROOT / "whale_local", DOMAIN, pipeline.DOMAIN]),
        NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    with (run / "worker.log").open("a") as log:
        worker = subprocess.Popen([str(ROOT / ".venv-whale-portable/bin/python"), "-u",
            str(ROOT / "whale_local/search_worker_v2.py")], env=env, stdout=log, stderr=subprocess.STDOUT)
        pipeline.write_json(run / "services.json", {"worker_pid": worker.pid})
        try:
            identity = pipeline.wait_ready(worker, base_url)
            yield base_url, identity
        finally:
            if worker.poll() is None:
                worker.send_signal(signal.SIGTERM)
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            pipeline.write_json(run / "services_stopped.json", {"pid": worker.pid, "returncode": worker.returncode})


async def evaluate(*, base_url, identity, harness, data_entry, output, rollouts=1,
                   temperature=0, seed=42, limit=None, resume=False, schedule=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=resume)
    assert pipeline.sha(data_entry["path"]) == data_entry["sha256"]
    count = min(limit or data_entry["count"], data_entry["count"])
    env = load_environment(harness_path=str(harness), eval_dataset_path=data_entry["path"], num_eval_examples=count)
    env.set_assistant_token_budget(2048)
    metadata = pd.read_parquet(data_entry["path"]).to_dict("records")[:count]
    plan = ([(draw, index) for draw in range(rollouts) for index in range(count)]
            if schedule is None else [tuple(item) for item in schedule])
    assert plan and len(set(plan)) == len(plan)
    assert all(isinstance(draw, int) and 0 <= draw < 64 and isinstance(index, int)
               and 0 <= index < count for draw, index in plan)
    records = []
    stream_path = output / "trajectories.jsonl"
    if resume and stream_path.exists():
        # Resume only a complete, identity-checked prefix. Never pick retries by reward.
        records = [json.loads(line) for line in stream_path.read_text().splitlines()]
        assert len(records) <= len(plan)
        for position, record in enumerate(records):
            draw, index = plan[position]
            assert record["draw"] == draw and record["example_index"] == index
            assert record["task_id"] == metadata[index]["extra_info"]["task_id"]
            assert record["sampling_identity"]["weight_digest"] == identity["weight_digest"]
            assert record["harness_sha256"] == pipeline.sha(harness)
            assert record["rollout_seed"] == seed * 100003 + draw * 1009 + index
        print(f"Resuming {output}: retained {len(records)} complete trajectories", flush=True)
    retained = len(records)
    client = RecordingClient(base_url)
    try:
        with stream_path.open("a" if resume else "x") as stream:
            for position, (draw, index) in enumerate(plan):
                if position < retained:
                    continue
                example = env.eval_dataset[index]
                task_id = metadata[index]["extra_info"]["task_id"]
                os.environ["SEARCH_R1_RETRIEVAL_URLS"] = base_url + "/retrieve/" + task_id
                search_r1_env._url_cycle = None
                client.calls = []
                client.context_rejection_id = None
                row_seed = seed * 100003 + draw * 1009 + index
                # Reference answers are used solely by the native rubric.
                state = await env.rollout(input={"prompt": example["prompt"], "example_id": index,
                    "answer": example["answer"], "info": {}}, client=client, model=identity["model"],
                    sampling_args={"temperature": temperature, "top_p": 1.0,
                                   "max_tokens": 768, "seed": row_seed})
                if state.get("error") and client.context_rejection_id is None:
                    raise RuntimeError(f"Native rollout failed: {state['error']!r}")
                for trace in state.get("harness_trace", []):
                    feedback = trace.get("env_feedback", "")
                    if "Search error:" in feedback and any(term in feedback for term in
                        ["502 Bad Gateway", "503 Service Unavailable", "All connection attempts failed", "Connection refused"]):
                        raise RuntimeError("Retrieval transport failure invalidates this rollout: " + feedback[:500])
                await env.rubric.score_rollout(state)
                tokens = sum(len(c["response_ids"]) for c in client.calls)
                assert tokens <= 2048, "Native cumulative token cap was not enforced"
                for call in client.calls:
                    assert call["sampling_identity"]["weight_digest"] == identity["weight_digest"]
                record = {"task_id": task_id, "example_index": index, "draw": draw,
                    "reward": state["reward"], "metrics": state["metrics"],
                    "last_assistant": _last_assistant_content(state["completion"]),
                    "harness_trace": state.get("harness_trace", []),
                    "policy_calls": client.calls, "assistant_tokens": tokens,
                    "harness_sha256": pipeline.sha(harness), "sampling_identity": identity,
                    "rollout_seed": row_seed, "answer_redacted": True}
                if client.context_rejection_id:
                    record.update(termination_reason="context_length_exceeded", context_rejection_id=client.context_rejection_id)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(record)
                if len(records) % 8 == 0:
                    print(f"{output.name}: {len(records)}/{len(plan)}, successes={sum(r['reward'] for r in records):g}", flush=True)
    finally:
        await client.close()
    result = {"num_examples": len(records), "solved_examples": int(sum(r["reward"] for r in records)),
        "success_rate": sum(r["reward"] for r in records) / len(records),
        "avg_metrics": {key: sum(r["metrics"].get(key, 0) for r in records) / len(records)
                        for key in records[0]["metrics"]},
        "mean_turn_count": sum(len(r["policy_calls"]) for r in records) / len(records),
        "mean_assistant_tokens": sum(r["assistant_tokens"] for r in records) / len(records),
        "metadata": {"dataset_path": data_entry["path"], "seed": seed, "rollouts": rollouts,
            "harness_path": str(harness), "harness_sha256": pipeline.sha(harness),
            "weight_digest": identity["weight_digest"], "temperature": temperature,
            "assistant_token_budget": 2048, "policy_max_tokens": 768}}
    if schedule is not None:
        result["metadata"]["sampling_schedule"] = [list(item) for item in plan]
        result["metadata"]["rollouts"] = None
    pipeline.write_json(output / "val.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--harness", type=Path, action="append", required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--split", choices=["train", "mh_val"], default="mh_val")
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--output-name")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parameter-dtype", choices=["bfloat16", "float32"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=args.resume)
    if (run / "complete.json").exists():
        return
    manifest = json.loads(args.manifest.read_text())
    model = args.model or Path((ROOT / "runs/whale_setup/base_model_path.txt").read_text().strip())
    config = {"purpose": "scheduled training sampler" if args.split == "train" else "development screen only", "data": manifest,
        "model": str(model), "harnesses": [{"path": str(h), "sha256": pipeline.sha(h)} for h in args.harness]}
    if args.schedule:
        config.update(schedule_sha256=pipeline.sha(args.schedule), seed=args.seed,
                      temperature=args.temperature, parameter_dtype=args.parameter_dtype)
    if (run / "config.json").exists():
        assert json.loads((run / "config.json").read_text()) == config
    else:
        pipeline.write_json(run / "config.json", config)
    schedule = json.loads(args.schedule.read_text()) if args.schedule else None
    assert not args.output_name or len(args.harness) == 1
    with service(run, model, args.gpu, args.port, manifest, parameter_dtype=args.parameter_dtype) as (base_url, identity):
        pipeline.write_json(run / "identity.json", identity)
        for harness in args.harness:
            output = run / (args.output_name or harness.stem)
            if (output / "val.json").exists():
                continue
            result = asyncio.run(evaluate(base_url=base_url, identity=identity, harness=harness,
                data_entry=manifest["splits"][args.split], output=output, limit=args.limit,
                schedule=schedule, temperature=args.temperature, seed=args.seed, resume=args.resume))
            print(json.dumps(result), flush=True)
        assert pipeline.api(base_url, "/health")["weight_digest"] == identity["weight_digest"]
    pipeline.write_json(run / "complete.json", {"frozen_weights_verified": True})


if __name__ == "__main__":
    main()
