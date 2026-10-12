"""Local ReTool policy transport and native WHALE math RSFT.

Run in the original portable training environment. The separate verifiers
process owns the unmodified ReToolEnv rollout and strict boxed reward.
"""
import ast
import asyncio
from collections import Counter
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import time
from typing import Any
import uuid

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import AliasChoices, BaseModel, Field

import worker as w
from pipeline import sha
from qwen35_padding import install as install_padding_fix

PADDING_FIX = install_padding_fix()
GRADIENT_NORM_FIX = None
if os.environ.get('WHALE_SAFE_GRAD_NORM') == '1':
    import safe_gradient_norm
    def record_gradient_event(event):
        w.atomic_json(w.RUN / 'gradient_norm_events.json', safe_gradient_norm.EVENTS)
        print('GRADIENT_NORM_EVENT ' + json.dumps(event), flush=True)
        if event['nonfinite_gradient_tensor_indices']:
            # Preserve the last valid parameters for targeted diagnosis. This
            # is not a final training checkpoint and cannot enter test.
            directory = w.RUN / 'gradient_failure_checkpoint'
            directory.mkdir()
            hashes = w.parameter_hashes(w.state['model'])
            w.state['model'].save_pretrained(directory, max_shard_size='5GB')
            w.state['tokenizer'].save_pretrained(directory)
            torch.save(w.state['optimizer'].state_dict(), directory / 'optimizer.pt')
            torch.save({'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}, directory/'rng.pt')
            w.atomic_json(directory/'training_state.json', {'partial_diagnostic_only':True,
                'parameter_hashes':hashes,'weight_digest':w.digest(hashes),
                'optimizer_steps_after':w.optimizer_steps(w.state['optimizer']),
                'failed_minibatch':event['minibatch_index'],'parameter_dtype':'float32'})
    GRADIENT_NORM_FIX = safe_gradient_norm.install(record_gradient_event)

DOMAIN = w.ROOT / "WHALE/domains/math_reasoning"
spec = importlib.util.spec_from_file_location("whale_math_native_parser",
    DOMAIN / "verl/experimental/agent_loop/tool_parser.py")
native_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native_parser)
# Upstream uses eval for XML array arguments. Literal parsing preserves arrays
# while refusing executable expressions from model-generated tool arguments.
native_parser.eval = ast.literal_eval
from verl.tools.schemas import OpenAIFunctionToolSchema

from math_reward import binary_reward

assert Path(inspect.getfile(w.DataParallelPPOActor)).resolve() == DOMAIN / "verl/workers/actor/dp_actor.py"

app = FastAPI(lifespan=w.lifespan)
app.get("/health")(w.health)
app.post("/checkpoint/load")(w.reload_checkpoint)


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = Field(default_factory=list)
    max_tokens: int = Field(default=512, ge=1, le=1024,
                           validation_alias=AliasChoices("max_tokens", "max_completion_tokens"))
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = 1.0
    seed: int | None = None
    stream: bool = False


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    with w.LOCK:
        if request.model != w.state["model_id"] or request.stream:
            raise HTTPException(409, "Stale model identity or streaming request")
        if request.seed is not None:
            torch.manual_seed(request.seed)
            torch.cuda.manual_seed_all(request.seed)
        messages = json.loads(json.dumps(request.messages))
        # HF Qwen templates expect decoded argument mappings in prior tool
        # calls; the OpenAI wire protocol represents them as JSON strings.
        for message in messages:
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                if isinstance(function.get("arguments"), str):
                    function["arguments"] = json.loads(function["arguments"])
        try:
            content, usage, record = w.generate(messages, tools=request.tools,
                max_tokens=request.max_tokens, temperature=request.temperature, top_p=request.top_p,
                stop_at_chat_end=True, return_token_logprobs=True)
        except ValueError as exc:
            if str(exc) != "Context exceeds the configured 4096-token integration limit":
                raise
            rejection_id = "context-" + uuid.uuid4().hex
            prompt_text = w.state["tokenizer"].apply_chat_template(messages,
                tools=request.tools, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            prompt_ids = w.state["tokenizer"](prompt_text, add_special_tokens=False)["input_ids"]
            (w.RUN / "context_rejections").mkdir(exist_ok=True)
            w.atomic_json(w.RUN / "context_rejections" / (rejection_id + ".json"), {
                "rejection_id": rejection_id, "messages": messages, "tools": request.tools,
                "sampling_identity": w.identity(), "requested_max_tokens": request.max_tokens,
                "seed": request.seed, "generated_tokens": 0, "context_limit": 4096,
                "prompt_tokens": len(prompt_ids)})
            return JSONResponse(status_code=400, content={"error": {
                "message": "This request exceeds the maximum context length of 4096 tokens.",
                "type": "invalid_request_error", "code": "context_length_exceeded",
                "param": "messages", "rejection_id": rejection_id}})
        schemas = [OpenAIFunctionToolSchema.model_validate(t) for t in request.tools]
        parser = native_parser.Qwen3XMLToolParser(w.state["tokenizer"])
        # Use raw generated IDs, including tool delimiters, for native parsing.
        parsed_content, calls = asyncio.run(parser.extract_tool_calls(record["response_ids"], schemas))
        if calls:
            content = parsed_content
        message = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = [{"id": "call_" + uuid.uuid4().hex, "type": "function",
                "function": {"name": c.name, "arguments": c.arguments}} for c in calls]
        call_id = "chatcmpl-" + uuid.uuid4().hex
        record.update(call_id=call_id, tools=request.tools, sampling_identity=w.identity(),
                      parsed_message=message, usage=usage, stop_at_chat_end=True,
                      requested_max_tokens=request.max_tokens)
        (w.RUN / "calls").mkdir(exist_ok=True)
        w.atomic_json(w.RUN / "calls" / (call_id + ".json"), record)
        return {"id": call_id, "object": "chat.completion", "created": int(time.time()),
            "model": w.state["model_id"], "choices": [{"index": 0, "message": message,
                "finish_reason": "tool_calls" if calls else ("length" if len(record["response_ids"]) == request.max_tokens else "stop")}],
            "usage": usage, "whale_record": record}


class TrainRequest(BaseModel):
    round: int = Field(ge=1)
    rollouts: str
    train_data: str
    harness: str
    expected_weight_digest: str


@app.post("/rsft/train")
def train(request: TrainRequest):
    with w.LOCK:
        started = time.monotonic()
        source = Path(request.rollouts).resolve()
        assert source.is_relative_to(w.RUN)
        out = w.RUN / f"round_{request.round:02d}"
        out.mkdir(exist_ok=True)
        checkpoint = out / "checkpoint"
        assert not checkpoint.exists()
        assert w.state["weight_digest"] == request.expected_weight_digest
        before, steps_before, identity = w.parameter_hashes(w.state["model"]), w.optimizer_steps(w.state["optimizer"]), w.identity()
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        data = pd.read_parquet(request.train_data).to_dict("records")
        targets = {row["extra_info"]["task_id"]: row["reward_model"]["ground_truth"]["target"] for row in data}
        accepted, successes = [], 0
        for row in rows:
            assert row["task_id"] in targets
            reward = binary_reward(row["last_assistant"], targets[row["task_id"]])
            assert reward == row["reward"], "Native math reward differs between rollout and training"
            assert row["harness_sha256"] == sha(request.harness)
            for call in row["policy_calls"]:
                saved = json.loads((w.RUN / "calls" / (call["call_id"] + ".json")).read_text())
                assert saved == call and call["sampling_identity"]["weight_digest"] == request.expected_weight_digest
            if reward:
                successes += 1
                accepted.extend(row["policy_calls"])
        if not accepted:
            raise HTTPException(422, "No verified successful trajectories for RSFT")
        torch.cuda.reset_peak_memory_stats()
        batch = w.make_batch(accepted)
        torch.save({key: value.cpu() for key, value in batch.batch.items()}, out / "accepted_batch.pt")
        update_started = time.monotonic()
        update_count = 0
        def record_update(optimizer, hook_args, hook_kwargs):
            nonlocal update_count
            update_count += 1
            if update_count == 1 or update_count % 4 == 0:
                w.atomic_json(out / "training_progress.json", {"round": request.round,
                    "completed_optimizer_updates": update_count,
                    "planned_minibatches": (len(accepted) + 3) // 4,
                    "update_elapsed_seconds": time.monotonic() - update_started,
                    "status": "running"})
        hook = w.state["optimizer"].register_step_post_hook(record_update)
        try:
            metrics = w.state["actor"].update_sft_policy(batch)
        finally:
            hook.remove()
        w.atomic_json(out / "training_progress.json", {"round": request.round,
            "completed_optimizer_updates": update_count,
            "planned_minibatches": (len(accepted) + 3) // 4,
            "update_elapsed_seconds": time.monotonic() - update_started,
            "status": "updates_finished_pending_checkpoint_verification"})
        assert all(np.isfinite(np.array(v)).all() for v in metrics.values())
        after = w.parameter_hashes(w.state["model"])
        changed = [key for key in before if before[key] != after[key]]
        assert changed and metrics["actor/sft_updates"]
        info = {"round": request.round, "method": "native WHALE ReTool online RSFT",
            "actor_source": inspect.getfile(w.DataParallelPPOActor), "sampling_identity": identity,
            "harness": request.harness, "harness_sha256": sha(request.harness),
            "train_data": request.train_data, "train_data_sha256": sha(request.train_data),
            "rollouts": len(rows), "accepted_trajectories": successes, "accepted_policy_calls": len(accepted),
            "response_tokens": int(batch.batch["response_mask"].sum()), "metrics": metrics,
            "learning_rate": w.LEARNING_RATE, "sft_epochs": 1,
            "parameter_dtype": w.PARAMETER_DTYPE_NAME,
            "qwen35_padding_compatibility_fix": PADDING_FIX,
            "optimizer_state_dtypes": sorted({str(value.dtype) for state in w.state["optimizer"].state.values()
                for key, value in state.items() if key != "step" and torch.is_tensor(value)}),
            "optimizer_steps_before": steps_before, "optimizer_steps_after": w.optimizer_steps(w.state["optimizer"]),
            "weight_digest": w.digest(after), "parameter_hashes": after, "changed_parameter_tensors": changed,
            "native_math_reward_rechecked_every_trajectory": True, "exact_server_tokens_verified": True,
            "elapsed_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
        if GRADIENT_NORM_FIX:
            info['gradient_norm_precision_fix'] = GRADIENT_NORM_FIX
            info['gradient_norm_precision_events'] = safe_gradient_norm.EVENTS
        staging = out / ("checkpoint.staging-" + uuid.uuid4().hex)
        staging.mkdir()
        w.state["model"].save_pretrained(staging, max_shard_size="5GB")
        w.state["tokenizer"].save_pretrained(staging)
        torch.save(w.state["optimizer"].state_dict(), staging / "optimizer.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, staging / "rng.pt")
        w.atomic_json(staging / "training_state.json", info)
        staging.rename(checkpoint)
        w.state.update(parameter_hashes=after, weight_digest=w.digest(after), model_id="whale-2b-" + w.digest(after)[:16],
                       path=str(checkpoint), training_state=info)
        w.atomic_json(out / "training_result.json", info)
        return {"checkpoint": str(checkpoint), "accepted_trajectories": successes, "metrics": metrics}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ["WHALE_POLICY_PORT"]))
