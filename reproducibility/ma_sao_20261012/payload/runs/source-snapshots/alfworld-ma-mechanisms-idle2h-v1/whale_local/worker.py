"""Single-GPU WHALE backend: real policy sampling, native online RSFT, HF handoff.

Only localhost control endpoints are exposed. The upstream runner/verifier and
DataParallelPPOActor implement the task and loss; this module supplies transport,
exact response-token capture, and checkpoint lifecycle missing from the release.
"""
from contextlib import asynccontextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Literal
import gc
import hashlib
import json
import os
import random
import threading
import time
import uuid

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoTokenizer
from autoharness_textarena.config import LLMConfig
from autoharness_textarena.llm import Completion
from autoharness_chess_puzzle.harness import load_harness
from autoharness_chess_puzzle.runner import example_from_mapping, run_puzzle_rollout
from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ["WHALE_RUN_ROOT"]).resolve()
BASE = Path(os.environ["WHALE_MODEL_ROOT"]).resolve()
LEARNING_RATE = float(os.environ.get("WHALE_LEARNING_RATE", "1e-5"))
PARAMETER_DTYPE_NAME = os.environ.get("WHALE_PARAMETER_DTYPE", "bfloat16")
PARAMETER_DTYPE = {"bfloat16": torch.bfloat16, "float32": torch.float32}[PARAMETER_DTYPE_NAME]
LOCK = threading.Lock()
state = {}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def parameter_hashes(model):
    return {name: hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for name, p in model.named_parameters()}


def digest(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def optimizer_steps(optimizer):
    return sorted({int(v["step"].item()) for v in optimizer.state.values() if "step" in v})


def load_model(path):
    state.clear()
    gc.collect()
    torch.cuda.empty_cache()
    model = AutoModelForImageTextToText.from_pretrained(
        path, local_files_only=True, dtype=PARAMETER_DTYPE,
        device_map="cuda:0", attn_implementation="sdpa")
    # Chess is text-only. Preserve vision weights, train all language parameters.
    for name, p in model.named_parameters():
        p.requires_grad_("visual" not in name)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=LEARNING_RATE, weight_decay=0.01)
    actor = DataParallelPPOActor(FSDPActorConfig(
        rollout_n=1, ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=1,
        use_remove_padding=False, use_torch_compile=False,
        use_dynamic_bsz=False, grad_clip=1.0), model, optimizer)
    hashes = parameter_hashes(model)
    metadata = None
    if (path / "training_state.json").exists():
        metadata = json.loads((path / "training_state.json").read_text())
        if hashes != metadata["parameter_hashes"]:
            raise RuntimeError("Checkpoint tensor digest mismatch after reload")
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
        if optimizer_steps(optimizer) != metadata["optimizer_steps_after"]:
            raise RuntimeError("Optimizer continuity check failed")
        rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng["cpu"])
        torch.cuda.set_rng_state_all(rng["cuda"])
    weight_digest = digest(hashes)
    state.update(model=model.eval(), tokenizer=tokenizer, optimizer=optimizer, actor=actor,
                 path=str(path), parameter_hashes=hashes, weight_digest=weight_digest,
                 model_id="whale-2b-" + weight_digest[:16], training_state=metadata)
    return identity()


def identity():
    result = {"ready": bool(state), "model": state.get("model_id"), "weights": state.get("path"),
            "weight_digest": state.get("weight_digest"),
            "optimizer_steps": optimizer_steps(state["optimizer"]) if state else []}
    if PARAMETER_DTYPE_NAME != "bfloat16":
        result["parameter_dtype"] = PARAMETER_DTYPE_NAME
    return result


@asynccontextmanager
async def lifespan(app):
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method="file://" + str(RUN / ("rdzv_" + uuid.uuid4().hex)),
                            rank=0, world_size=1)
    try:
        print(json.dumps(load_model(BASE)), flush=True)
        yield
    finally:
        state.clear()
        gc.collect()
        torch.cuda.empty_cache()
        dist.destroy_process_group()


app = FastAPI(lifespan=lifespan)


def generate(messages, *, max_tokens, temperature, top_p=0.95, top_k=20, tools=None,
             stop_at_chat_end=False, return_token_logprobs=False):
    tokenizer, model = state["tokenizer"], state["model"]
    model.eval()
    template_args = {} if tools is None else {"tools": tools}
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False, **template_args)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    prompt = inputs["input_ids"][0].tolist()
    if len(prompt) + max_tokens > 4096:
        raise ValueError("Context exceeds the configured 4096-token integration limit")
    kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0)
    if stop_at_chat_end:
        eos = model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        chat_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if len(chat_end) != 1:
            raise ValueError("Expected a single Qwen message-end token")
        kwargs["eos_token_id"] = sorted(set(eos + chat_end))
    if temperature > 0:
        kwargs.update(temperature=temperature, top_p=top_p, top_k=top_k)
    if return_token_logprobs:
        kwargs.update(return_dict_in_generate=True, output_scores=True)
    precision_context = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                         if PARAMETER_DTYPE == torch.float32 else nullcontext())
    with torch.inference_mode(), precision_context:
        output = model.generate(**inputs, **kwargs)
    sequences = output.sequences if return_token_logprobs else output
    response = sequences[0, len(prompt):].tolist()
    content = tokenizer.decode(response, skip_special_tokens=True)
    usage = dict(prompt_tokens=len(prompt), completion_tokens=len(response),
                 total_tokens=len(prompt) + len(response))
    record = {"messages": messages, "prompt_ids": prompt, "response_ids": response, "content": content}
    if return_token_logprobs:
        record["response_logprobs"] = [float(torch.log_softmax(score[0].float(), dim=-1)[token].item())
                                        for score, token in zip(output.scores, response)]
        assert len(record["response_logprobs"]) == len(response)
    return content, usage, record


class RecordingPolicy:
    def __init__(self, temperature):
        self.config = LLMConfig(model=state["model_id"], temperature=temperature, max_tokens=128)
        self.calls = []

    def complete_response(self, messages, max_tokens=None):
        content, usage, record = generate([asdict(m) for m in messages], max_tokens=max_tokens or 128,
                                           temperature=self.config.temperature)
        self.calls.append(record)
        return Completion(content, usage)


def make_batch(calls):
    """Left-pad prompt, right-pad response; train only generated assistant tokens.

    Each policy call is a separate SFT item with the EXACT context used to sample
    it. Later turns include prior assistant/tool/user history in the prompt mask.
    """
    if not calls or any(not c["prompt_ids"] or not c["response_ids"] for c in calls):
        raise ValueError("RSFT requires nonempty accepted policy completions")
    pad = state["tokenizer"].pad_token_id
    pmax = max(len(c["prompt_ids"]) for c in calls)
    rmax = max(len(c["response_ids"]) for c in calls)
    ids = torch.full((len(calls), pmax + rmax), pad, dtype=torch.long)
    attn = torch.zeros_like(ids)
    response_mask = torch.zeros((len(calls), rmax), dtype=torch.long)
    for i, c in enumerate(calls):
        p, r = c["prompt_ids"], c["response_ids"]
        ids[i, pmax-len(p):pmax] = torch.tensor(p)
        ids[i, pmax:pmax+len(r)] = torch.tensor(r)
        attn[i, pmax-len(p):pmax+len(r)] = 1
        response_mask[i, :len(r)] = 1
        assert ids[i, pmax:pmax+len(r)].tolist() == r
    positions = (attn.cumsum(-1) - 1).clamp(min=0)
    return DataProto.from_dict(tensors={"input_ids": ids, "attention_mask": attn,
        "position_ids": positions, "responses": ids[:, pmax:].clone(), "response_mask": response_mask},
        meta_info={"temperature": 1.0, "pad_token_id": pad, "sft_epochs": 1,
                   "sft_mini_batch_size": 4, "sft_micro_batch_size_per_gpu": 1})


class ChatRequest(BaseModel):
    model: str
    messages: list[dict[str, str]] = Field(min_length=1)
    max_tokens: int = Field(default=128, ge=1, le=256)
    temperature: float = Field(default=0, ge=0, le=2)
    top_p: float = 1.0
    top_k: int = 20
    stream: bool = False
    chat_template_kwargs: dict = Field(default_factory=dict)


@app.get("/health")
def health():
    with LOCK:
        return identity()


@app.post("/v1/chat/completions")
def chat(request: ChatRequest):
    with LOCK:
        if request.model != state["model_id"] or request.stream:
            raise HTTPException(409, "Requested model identity is not the loaded checkpoint, or streaming requested")
        content, usage, _ = generate(request.messages, max_tokens=request.max_tokens,
            temperature=request.temperature, top_p=request.top_p, top_k=request.top_k)
        return {"id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": state["model_id"], "weight_digest": state["weight_digest"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "length" if usage["completion_tokens"] == request.max_tokens else "stop"}],
            "usage": usage}


class TrainRequest(BaseModel):
    round: int = Field(ge=1)
    harness: str
    train_data: str
    expected_weight_digest: str
    rollouts_per_puzzle: int = Field(default=4, ge=1, le=16)
    seed: int = 42
    matched_sampling: bool = False
    gate_probe_harness: str | None = None
    gate_probe_budget: int = Field(default=16, ge=0, le=128)
    gate_resamples: int = Field(default=2, ge=0, le=16)
    gate_mode: Literal["counterfactual", "teacher"] = "counterfactual"
    teacher_audit_pairs: int = Field(default=2, ge=0, le=8)
    teacher_equal_budget: bool = False


@app.post("/rsft/train")
def train(request: TrainRequest):
    with LOCK:
        if request.gate_mode == "teacher" and not request.gate_probe_harness:
            raise HTTPException(422, "Teacher gate requires a frozen-weight audit harness")
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        if request.expected_weight_digest != state["weight_digest"]:
            raise HTTPException(409, "Sampling checkpoint mismatch")
        out = RUN / f"round_{request.round:02d}"
        out.mkdir(exist_ok=True)
        checkpoint = out / "checkpoint"
        if checkpoint.exists():
            raise HTTPException(409, "Checkpoint already exists; reload it instead of repeating the update")
        harness_path = Path(request.harness).resolve()
        harness_digest = hashlib.sha256(harness_path.read_bytes()).hexdigest()
        before = state["parameter_hashes"]
        before_steps = optimizer_steps(state["optimizer"])
        sampling_identity = identity()
        rows = pd.read_parquet(request.train_data).to_dict("records")
        torch.manual_seed(request.seed)
        torch.cuda.manual_seed_all(request.seed)
        random.Random(request.seed).shuffle(rows)
        row_map = {hashlib.sha256(example_from_mapping(row).start_fen.encode()).hexdigest(): row for row in rows}

        def sample(task_key, path, rollout_seed):
            torch.manual_seed(rollout_seed)
            torch.cuda.manual_seed_all(rollout_seed)
            random.seed(rollout_seed)
            np.random.seed(rollout_seed % (2**32))
            policy = RecordingPolicy(temperature=0.8)
            result = run_puzzle_rollout(harness=load_harness(Path(path)),
                example=example_from_mapping(row_map[task_key]), llm=policy,
                assistant_token_budget=256, policy_max_tokens=128)
            result.update(task_key=task_key, rollout_seed=rollout_seed,
                          sampling_identity=sampling_identity,
                          harness_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                          policy_calls=policy.calls)
            return result

        accepted, trajectories = [], []
        with (out / "train_rollouts.jsonl").open("w") as stream:
            for n in range(1 if request.gate_mode == "teacher" else request.rollouts_per_puzzle):
                for row in rows:
                    task_key = hashlib.sha256(example_from_mapping(row).start_fen.encode()).hexdigest()
                    if request.gate_probe_harness or request.matched_sampling:
                        result = sample(task_key, harness_path, request.seed * 10000 + len(trajectories))
                    else:
                        policy = RecordingPolicy(temperature=0.8)
                        result = run_puzzle_rollout(harness=load_harness(harness_path),
                            example=example_from_mapping(row), llm=policy,
                            assistant_token_budget=256, policy_max_tokens=128)
                        result.update(task_key=task_key, policy_calls=policy.calls)
                    result.update(sample_index=len(trajectories), sampling_identity=sampling_identity,
                                  harness_sha256=harness_digest)
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                    stream.flush()
                    trajectories.append(result)
                    if result["reward"] > 0.5:
                        accepted.extend(result["policy_calls"])
                    print(f"round={request.round} sampled={len(trajectories)} "
                          f"reward={result['reward']} accepted_calls={len(accepted)}", flush=True)
        sampling_seconds = time.monotonic() - started
        selected = [t for t in trajectories if t["reward"] > 0.5]
        gate_summary = None
        if request.gate_probe_harness:
            if request.gate_mode == "teacher":
                if request.teacher_equal_budget:
                    from teacher_budget_gate import route_and_select
                else:
                    from teacher_gate import route_and_select
                selected, gate_summary = route_and_select(trajectories, replay=sample,
                    original_harness=harness_path, probe_harness=Path(request.gate_probe_harness).resolve(),
                    out=out, rollouts_per_puzzle=request.rollouts_per_puzzle,
                    audit_pairs=request.teacher_audit_pairs, seed=request.seed)
            else:
                from failure_gate import route_and_select
                selected, gate_summary = route_and_select(trajectories, replay=sample,
                    original_harness=harness_path, probe_harness=Path(request.gate_probe_harness).resolve(),
                    out=out, probe_budget=request.gate_probe_budget,
                    resamples=request.gate_resamples, seed=request.seed)
            # Verify no optimizer or model changes happened during the interventions.
            assert parameter_hashes(state["model"]) == before
            assert optimizer_steps(state["optimizer"]) == before_steps
            accepted = [c for t in selected for c in t["policy_calls"]]
        if not accepted and gate_summary is None:
            raise HTTPException(422, "No successful online trajectories: no weight update performed")
        sft_started = time.monotonic()
        response_tokens = 0
        metrics = {"actor/sft_updates": 0}
        if accepted:
            batch = make_batch(accepted)
            response_tokens = int(batch.batch["response_mask"].sum())
            torch.save({k: v.cpu() for k, v in batch.batch.items()}, out / "accepted_batch.pt")
            metrics = state["actor"].update_sft_policy(batch)
        if not all(np.isfinite(np.array(v)).all() for v in metrics.values()):
            raise RuntimeError("Nonfinite optimizer metrics; checkpoint refused")
        after = parameter_hashes(state["model"])
        changed = [n for n in before if before[n] != after[n]]
        if accepted and (not changed or not metrics["actor/sft_updates"]):
            raise RuntimeError("No actual language weight update")
        if not accepted:
            assert not changed and optimizer_steps(state["optimizer"]) == before_steps
        info = {"round": request.round, "method": "online rejection-sampling SFT",
                "actor_source": __import__("inspect").getfile(DataParallelPPOActor),
                "sampling_identity": sampling_identity, "harness": str(harness_path),
                "harness_sha256": harness_digest, "train_data": request.train_data,
                "train_data_sha256": hashlib.sha256(Path(request.train_data).read_bytes()).hexdigest(),
                "seed": request.seed, "rollouts": len(trajectories),
                "accepted_trajectories": len(selected),
                "original_successful_trajectories": sum(t["reward"] > 0.5 for t in trajectories),
                "accepted_policy_calls": len(accepted), "response_tokens": response_tokens,
                "training_temperature": 1.0, "sampling_temperature": 0.8, "learning_rate": LEARNING_RATE,
                "optimizer_steps_before": before_steps, "optimizer_steps_after": optimizer_steps(state["optimizer"]),
                "metrics": metrics, "changed_parameter_tensors": changed,
                "trainable_parameters": sum(p.numel() for p in state["model"].parameters() if p.requires_grad),
                "weight_digest": digest(after), "parameter_hashes": after,
                "gate": gate_summary,
                "update_status": "updated" if accepted else "gate_skipped_no_verified_weight_targets",
                "sampling_seconds": sampling_seconds,
                "sft_and_hash_seconds": time.monotonic() - sft_started,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()}
        staging = out / ("checkpoint.staging-" + uuid.uuid4().hex)
        staging.mkdir()
        state["model"].save_pretrained(staging, max_shard_size="5GB")
        state["tokenizer"].save_pretrained(staging)
        torch.save(state["optimizer"].state_dict(), staging / "optimizer.pt")
        torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}, staging / "rng.pt")
        info["elapsed_seconds"] = time.monotonic() - started
        atomic_json(staging / "training_state.json", info)
        staging.rename(checkpoint)
        info["checkpoint"] = str(checkpoint)
        state.update(parameter_hashes=after, weight_digest=digest(after),
                     model_id="whale-2b-" + digest(after)[:16], path=str(checkpoint), training_state=info)
        atomic_json(out / "training_result.json", info)
        print(f"round={request.round} checkpoint={checkpoint} updates={metrics['actor/sft_updates']}", flush=True)
        return {k: v for k, v in info.items() if k not in {"parameter_hashes", "changed_parameter_tensors"}} | {
            "changed_parameter_tensors": len(changed)}


class LoadRequest(BaseModel):
    checkpoint: str


@app.post("/checkpoint/load")
def reload_checkpoint(request: LoadRequest):
    with LOCK:
        path = Path(request.checkpoint).resolve()
        allowed = {Path(p).resolve() for p in json.loads(os.environ.get("WHALE_CHECKPOINT_ALLOWLIST", "[]"))}
        completed = all((path / name).is_file() for name in ["training_state.json", "optimizer.pt", "rng.pt"])
        if path != BASE and (not completed or not (path.is_relative_to(RUN) or path in allowed)):
            raise HTTPException(400, "Only the configured base, completed run checkpoints, or explicitly allowlisted completed checkpoints can be loaded")
        result = load_model(path)
        if path != BASE:
            if path.is_relative_to(RUN):
                proof = path.parent / "reload_verification.json"
            else:
                proof = RUN / "checkpoint_loads" / (result["weight_digest"] + ".json")
                proof.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(proof, result | {"all_parameter_hashes_equal": True})
        return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WHALE_POLICY_PORT", "18541")))
