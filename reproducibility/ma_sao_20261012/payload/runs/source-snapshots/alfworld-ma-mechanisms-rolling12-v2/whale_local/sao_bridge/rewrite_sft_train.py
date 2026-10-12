"""Bounded, train-only editing SFT using WHALE's native response-token update.

Example (from the repository root, on an externally allocated H200):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=whale_local .venv-ppo/bin/python \
    -m sao_bridge.rewrite_sft_train --model assets/Qwen3-4B-Instruct-2507 \
    --data /mnt/3fs/xudong/coevolve/edit_dataset/examples.json \
    --output /mnt/3fs/xudong/coevolve/edit_sft/run01

The output must be new. Individual sequences AND padded groups must fit
--max-length; nothing is truncated or silently dropped. Microbatches contain
one document, with one native token-weighted optimizer update per group.
--batch-size is GLOBAL, including under single-node torchrun (1-8 ranks).
Each global group is deterministically partitioned without repeats or drops;
its size must be divisible by world size. Each rank holds a full FP32 model
and AdamW state. Native local token-mean gradients are reduced once per group,
weighted by local/global response tokens, BEFORE native gradient clipping.
Use torchrun --standalone --local-addr=127.0.0.1 --nnodes=1 --max-restarts=0.
The explicit loopback avoids these nodes' link-local hostname resolution.
Rendezvous uses local
TCP (torchrun), or /dev/shm for a plain single-process invocation.
Only train targets receive edit-contract validation; held-out rows receive
schema/count checks only. Compute is restricted to authorized research nodes;
the external launcher, not this trainer, owns GPU allocation/filler lifecycle.
By default this saves fresh-optimizer HF weights only. For incremental stages,
use --save-training-state to also save OUTPUT/training_state.pt and its SHA256
sidecar. Start the next stage with --model PREVIOUS_OUTPUT/checkpoint and
--resume-state PREVIOUS_OUTPUT (or PREVIOUS_OUTPUT/training_state.pt), plus
--save-training-state if another stage will follow. Only load your own trusted
outputs: checksums detect corruption, not malicious replacement. State loading
uses weights_only=True and binds AdamW steps/moments to exact checkpoint files,
parameter probes, optimizer settings, batch size, max length and world size.
Opt-in stages require --epochs 1 and disjoint train IDs across the whole chain;
submit a frozen snapshot of NEW teacher data, never an accumulating dataset.
The external controller owns data arrival, dev evaluation/plateau stopping and
the 2-3h budget. This is a completed-stage optimizer continuation, not a
mid-stage/RNG-state restart; stochastic layers need not be bitwise continuous.
Completion is NOT held-out acceptance: evaluate legality/instruction fidelity
separately before using output/checkpoint as SAO's starting model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import socket
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
SHARED_ROOT = Path("/mnt/3fs/xudong")
SYSTEM_MESSAGE = (
    "Apply the requested edit to the supplied document. Return only the complete "
    "edited document, without Markdown fences or explanation. Preserve every "
    "unrequested setting and the original declaration structure."
)
USER_TEMPLATE = "Instruction:\n{instruction}\n\nDocument:\n{source}"
NATIVE_ACTOR = ROOT / "WHALE/domains/math_reasoning/verl/workers/actor/dp_actor.py"


def format_messages(record: dict) -> list[dict[str, str]]:
    """The exact training/evaluation prompt; target and changes never enter it."""
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": USER_TEMPLATE.format(**record)},
    ]


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON value: {value}")


def load_records(path: Path) -> tuple[list[dict], str]:
    """Validate all schemas, but inspect document semantics only for train."""
    from .rewrite_data import EditExample, validate_example

    payload = path.read_bytes()
    rows = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Data must be a nonempty JSON array")
    ids, families = set(), {}
    for index, row in enumerate(rows):
        try:
            item = EditExample.model_validate(row, strict=True)
            if item.split == "train":
                validate_example(item)
            if item.id in ids:
                raise ValueError(f"Duplicate record id: {item.id}")
            if item.family in families and families[item.family] != item.split:
                raise ValueError(f"Family leaked across splits: {item.family}")
            ids.add(item.id)
            families[item.family] = item.split
        except (ValueError, SyntaxError, TypeError) as exc:
            raise ValueError(f"Invalid data row {index}: {exc}") from exc
    if not any(row["split"] == "train" for row in rows):
        raise ValueError("Data contains no train records")
    return rows, hashlib.sha256(payload).hexdigest()


def tokenize_training_rows(rows: list[dict], tokenizer, max_length: int) -> list[dict]:
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError("Tokenizer must have an EOS token")
    calls = []
    for row in rows:
        if row["split"] != "train":
            continue
        prompt = tokenizer.apply_chat_template(
            format_messages(row), tokenize=True, add_generation_prompt=True,
            return_dict=False, truncation=False,
        )
        response = tokenizer.encode(row["target"], add_special_tokens=False, truncation=False)
        if not prompt or not response:
            raise ValueError(f"Empty prompt or target tokens for {row['id']}")
        if any(type(token) is not int or token < 0 for token in [*prompt, *response]):
            raise ValueError(f"Invalid token IDs for {row['id']}")
        if set(response) & set(tokenizer.all_special_ids):
            raise ValueError(f"Target contains reserved special tokens: {row['id']}")
        response = [*response, eos]
        if len(prompt) + len(response) > max_length:
            raise ValueError(
                f"Record {row['id']} needs {len(prompt) + len(response)} tokens "
                f"(max_length={max_length}); truncation is forbidden"
            )
        calls.append({
            "id": row["id"], "split": "train", "prompt_ids": prompt,
            "response_ids": response, "temperature": 1.0,
        })
    if not calls:
        raise ValueError("No train records to tokenize")
    return calls


def padded_length(calls: list[dict]) -> int:
    if not calls or any(not c["prompt_ids"] or not c["response_ids"] for c in calls):
        raise ValueError("Empty training group or token sequence")
    return max(len(c["prompt_ids"]) for c in calls) + max(len(c["response_ids"]) for c in calls)


def plan_batches(calls: list[dict], epochs: int, batch_size: int, seed: int, max_length: int,
                 world_size: int = 1):
    if epochs < 1 or batch_size < 1 or not calls:
        raise ValueError("Positive epochs, batch_size and nonempty train calls required")
    if any(c["split"] != "train" for c in calls):
        raise ValueError("Only train records may enter an optimizer group")
    if not 1 <= world_size <= 8 or batch_size % world_size or len(calls) % world_size:
        raise ValueError("Global batch-size and train count must be divisible by world_size (1-8)")
    rng, plan = random.Random(seed), []
    for epoch in range(1, epochs + 1):
        order = list(range(len(calls)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            length = padded_length([calls[i] for i in indices])
            if length > max_length:
                raise ValueError(
                    f"Epoch {epoch} padded group needs {length} tokens "
                    f"(max_length={max_length}); reduce batch-size or raise max-length"
                )
            plan.append({"epoch": epoch, "indices": indices})
    return plan


def rank_indices(indices: list[int], rank: int, world_size: int) -> list[int]:
    if not 0 <= rank < world_size or not indices or len(indices) % world_size:
        raise ValueError("Each global group must partition into nonempty equal-size rank groups")
    return indices[rank::world_size]


@dataclass
class DistributedRun:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    control_group: object = field(default=None, repr=False)

    @classmethod
    def from_environment(cls):
        keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        if not any(key in os.environ for key in keys):
            return cls()
        if not all(key in os.environ for key in keys):
            raise ValueError("Incomplete torchrun rank environment")
        rank, world, local_rank, local_world = (int(os.environ[key]) for key in keys)
        if (not 1 <= world <= 8 or local_world != world or rank != local_rank
                or not 0 <= rank < world or int(os.environ.get("GROUP_RANK", "0")) != 0):
            raise ValueError("Only single-node torchrun with 1-8 local ranks is supported")
        return cls(rank, world, local_rank)

    def gather(self, value):
        if self.world_size == 1:
            return [value]
        import torch.distributed as dist

        values = [None] * self.world_size
        dist.all_gather_object(values, value, group=self.control_group)
        return values

    def check(self, stage, error=None):
        failures = self.gather(None if error is None else f"{type(error).__name__}: {error}")
        if any(failures):
            details = "; ".join(f"rank {i}: {value}" for i, value in enumerate(failures) if value)
            raise RuntimeError(f"Distributed {stage} failed: {details}")

    def phase(self, stage, operation):
        """All ranks enter once; operation itself must not contain collectives."""
        if self.world_size == 1:
            return operation()
        value, error = None, None
        try:
            value = operation()
        except BaseException as exc:
            error = exc
        self.check(stage, error)
        return value

    def root_phase(self, stage, operation):
        return self.phase(stage, operation if self.rank == 0 else lambda: None)

    def agree(self, stage, value):
        values = self.gather(value)
        if any(item != values[0] for item in values):
            raise RuntimeError(f"Ranks disagree on {stage}")


@contextmanager
def distributed_process_group(run: DistributedRun, timeout_seconds=300, backend="nccl"):
    """Gloo control messages keep Python failures separate from gradient traffic.

    Failed runs deliberately avoid teardown barriers; torchrun terminates peers.
    Process-group timeouts bound waits if a worker dies before sending a status.
    """
    import torch.distributed as dist

    if run.world_size == 1:
        with single_rank_process_group():
            yield
        return
    if dist.is_initialized():
        raise RuntimeError("Run this CLI outside an existing process group")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    timeout = timedelta(seconds=timeout_seconds)
    dist.init_process_group(backend, init_method="env://", rank=run.rank,
                            world_size=run.world_size, timeout=timeout)
    succeeded = False
    try:
        run.control_group = dist.new_group(backend="gloo", timeout=timeout)
        run.agree("single-node hostname", socket.gethostname())
        yield
        succeeded = True
    finally:
        if succeeded:
            dist.destroy_process_group(run.control_group)
            dist.destroy_process_group()


def reduce_token_weighted_gradients(model, local_tokens: int, global_tokens: int,
                                    bucket_bytes: int = 64 * 1024**2):
    """Reduce native local token-means, not a mean of unequal-rank means.

    No DDP wrapper is used: communication happens once after all microbatch
    backwards. SUM_r (T_r / T_global) * grad_r equals the native global-batch
    gradient. The caller invokes the unchanged native clipping/AdamW step next.
    """
    import torch
    import torch.distributed as dist

    parameters = [p for p in model.parameters() if p.requires_grad]
    capacity = max(1, bucket_bytes // 4)
    buffer = torch.empty(capacity, dtype=torch.float32, device=parameters[0].device)
    views, used = [], 0

    def flush():
        nonlocal views, used
        dist.all_reduce(buffer[:used], op=dist.ReduceOp.SUM)
        offset = 0
        for view in views:
            view.copy_(buffer[offset:offset + view.numel()])
            offset += view.numel()
        views, used = [], 0

    with torch.no_grad():
        for parameter in parameters:
            flat = parameter.grad.view(-1)
            start = 0
            while start < flat.numel():
                count = min(capacity - used, flat.numel() - start)
                view = flat[start:start + count]
                buffer[used:used + count].copy_(view).mul_(local_tokens / global_tokens)
                views.append(view)
                used += count
                start += count
                if used == capacity:
                    flush()
        if used:
            flush()


def native_distributed_update(actor, batch, run: DistributedRun, global_tokens: int):
    """Keep update_sft_policy intact; intercept only its optimizer boundary."""
    import torch

    if run.world_size == 1:
        return actor.update_sft_policy(batch)
    local_tokens = int(batch.batch["response_mask"].sum())
    original_step = actor._optimizer_step
    reached_step = False
    ready_checked = False
    readiness_passed = False

    def step():
        nonlocal reached_step, ready_checked, readiness_passed
        if reached_step:
            raise RuntimeError("Expected exactly one native optimizer boundary per group")
        reached_step = True
        error = None
        try:
            parameters = [p for p in actor.actor_module.parameters() if p.requires_grad]
            if not parameters or actor.scaler is not None:
                raise RuntimeError("Replicated SFT requires unscaled FP32 master gradients")
            if any(p.dtype != torch.float32 or p.grad is None or p.grad.dtype != torch.float32
                   or p.grad.is_sparse or not p.grad.is_contiguous() for p in parameters):
                raise RuntimeError("Every trainable parameter must have a dense contiguous FP32 gradient")
        except BaseException as exc:
            error = exc
        ready_checked = True
        run.check("backward/gradient readiness", error)
        readiness_passed = True
        reduce_token_weighted_gradients(actor.actor_module, local_tokens, global_tokens)
        return original_step()

    actor._optimizer_step = step
    metrics, error = None, None
    try:
        metrics = actor.update_sft_policy(batch)
        if not reached_step:
            raise RuntimeError("Native update did not reach its optimizer boundary")
    except BaseException as exc:
        error = exc
    finally:
        actor._optimizer_step = original_step
    if not ready_checked:
        # A peer may be waiting at the hook after this rank failed in forward/backward.
        run.check("backward/gradient readiness", error)
    if ready_checked and not readiness_passed:
        raise error
    run.check("native optimizer update", error)
    return metrics


def make_sft_batch(calls: list[dict], pad: int, max_length: int):
    from .legacy_update import make_batch

    if any(c["split"] != "train" for c in calls):
        raise ValueError("Only train records may enter an optimizer group")
    if padded_length(calls) > max_length:
        raise ValueError("Padded training group exceeds max_length")
    batch = make_batch(calls, pad)
    batch.meta_info.update(
        temperature=1.0, sft_epochs=1, sft_mini_batch_size=len(calls),
        sft_micro_batch_size_per_gpu=1,
    )
    return batch


def make_actor(model, optimizer, batch_size: int):
    from .legacy_update import DataParallelPPOActor, FSDPActorConfig

    if Path(inspect.getfile(DataParallelPPOActor)).resolve() != NATIVE_ACTOR.resolve():
        raise RuntimeError("Imported actor is not the repository's native WHALE actor")
    config = FSDPActorConfig(
        rollout_n=1, ppo_mini_batch_size=batch_size, ppo_micro_batch_size_per_gpu=1,
        use_remove_padding=False, use_torch_compile=False, use_dynamic_bsz=False,
        use_fused_kernels=False, grad_clip=1.0,
    )
    if config.fsdp_config.get("dtype") != "bfloat16":
        raise RuntimeError("Native actor must use BF16 autocast with FP32 master parameters")
    return DataParallelPPOActor(config, model, optimizer)


def run_updates(actor, optimizer, calls, plan, pad, max_length, report=None, run=None,
                initial_updates=0):
    import torch

    run = run or DistributedRun()
    records = []
    for update, group in enumerate(plan, 1):
        started = time.monotonic()
        selected = [calls[i] for i in group["indices"]]
        shards = [rank_indices(group["indices"], rank, run.world_size)
                  for rank in range(run.world_size)]
        local = [calls[i] for i in shards[run.rank]]
        batch = run.phase("batch construction", lambda: make_sft_batch(local, pad, max_length))
        tokens = int(batch.batch["response_mask"].sum())
        global_tokens = sum(len(c["response_ids"]) for c in selected)
        metrics = native_distributed_update(actor, batch, run, global_tokens)

        def validate_update():
            for key in ("actor/sft_loss", "actor/sft_token_count", "actor/sft_updates"):
                if not math.isfinite(float(metrics[key])):
                    raise FloatingPointError(f"Nonfinite native metric: {key}")
            norms = metrics["actor/grad_norm"]
            if len(norms) != 1 or not all(math.isfinite(float(n)) for n in norms):
                raise FloatingPointError("Nonfinite or missing native gradient norm")
            if metrics["actor/sft_updates"] != 1.0 or metrics["actor/sft_token_count"] != tokens:
                raise RuntimeError("Native update did not perform exactly one token-weighted group")
            steps = {int(state["step"].item()) for state in optimizer.state.values() if "step" in state}
            expected_step = initial_updates + update
            if steps != {expected_step}:
                raise RuntimeError(f"Optimizer update count {steps}, expected {{{expected_step}}}")
            if any(state[key].dtype != torch.float32 for state in optimizer.state.values()
                   for key in ("exp_avg", "exp_avg_sq") if key in state):
                raise RuntimeError("AdamW moments are not FP32")

        run.phase("metric/optimizer validation", validate_update)
        rank_metrics = run.gather({
            "rank": run.rank, "response_tokens": tokens, "examples": len(local),
            "seconds": time.monotonic() - started, "native": metrics,
        })
        if sum(row["response_tokens"] for row in rank_metrics) != global_tokens:
            raise RuntimeError("Rank response-token counts do not cover the global group")
        global_metrics = {
            "actor/sft_loss": sum(row["native"]["actor/sft_loss"] * row["response_tokens"]
                                  for row in rank_metrics) / global_tokens,
            "actor/sft_token_count": global_tokens, "actor/sft_updates": 1.0,
            "actor/grad_norm": metrics["actor/grad_norm"],
        }
        records.append({
            "update": update, "optimizer_step": initial_updates + update,
            "epoch": group["epoch"], "ids": [c["id"] for c in selected],
            "rank_ids": [[calls[i]["id"] for i in shard] for shard in shards],
            "response_tokens": global_tokens, "examples": len(selected),
            "padded_length": padded_length(selected), "native": global_metrics,
            "rank_metrics": rank_metrics, "seconds": max(row["seconds"] for row in rank_metrics),
        })
        if report is not None:
            run.root_phase("metrics persistence", lambda: report(records))
    return records


@contextmanager
def single_rank_process_group():
    """Own a single-rank NCCL group, with rendezvous on local IPC storage."""
    import torch.distributed as dist

    if dist.is_initialized():
        raise RuntimeError("Run this single-process CLI outside an existing process group")
    rendezvous = Path("/dev/shm") / f"rewrite-sft-{os.getuid()}-{uuid.uuid4().hex}.rdzv"
    try:
        dist.init_process_group(
            "nccl", init_method=rendezvous.as_uri(), rank=0, world_size=1,
            timeout=timedelta(minutes=5),
        )
        yield
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            rendezvous.unlink(missing_ok=True)


def _sample(tensor):
    import torch

    flat = tensor.detach().reshape(-1)
    count = min(4096, flat.numel())
    indices = torch.arange(count, dtype=torch.long, device=flat.device)
    indices = indices * (flat.numel() - 1) // max(1, count - 1)
    return flat.index_select(0, indices).cpu().clone()


def parameter_probes(model) -> dict:
    import torch

    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad and p.numel()}
    if not parameters or any(p.dtype != torch.float32 for p in parameters.values()):
        raise RuntimeError("All trainable master parameters must be FP32")
    names = [name for name in parameters if name.endswith("self_attn.q_proj.weight")]
    if not names:
        names = list(parameters)
    chosen = dict.fromkeys((names[0], names[len(names) // 2], names[-1]))
    probes = {name: _sample(parameters[name]) for name in chosen}
    if any(not torch.isfinite(value).all() for value in probes.values()):
        raise FloatingPointError("Nonfinite initial parameter probe")
    return probes


def check_parameter_deltas(model, before: dict) -> tuple[dict, dict]:
    import torch

    after, deltas = {}, {}
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
            raise FloatingPointError(f"Nonfinite or non-FP32 parameter: {name}")
        if name in before:
            value = _sample(parameter)
            delta = value - before[name]
            if not torch.isfinite(delta).all():
                raise FloatingPointError(f"Nonfinite parameter delta: {name}")
            after[name] = value
            deltas[name] = {
                "sampled_elements": value.numel(), "changed_elements": int(delta.ne(0).sum()),
                "max_abs_delta": float(delta.abs().max()),
                "before_sha256": hashlib.sha256(before[name].numpy().tobytes()).hexdigest(),
                "after_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            }
    if set(after) != set(before) or not any(d["changed_elements"] for d in deltas.values()):
        raise RuntimeError("Training produced no checked parameter change")
    return deltas, after


def verify_saved_probes(checkpoint: Path, expected: dict) -> None:
    import torch
    from safetensors import safe_open

    remaining = set(expected)
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for name in remaining & set(tensors.keys()):
                value = _sample(tensors.get_tensor(name))
                if value.dtype != torch.float32 or not torch.equal(value, expected[name]):
                    raise RuntimeError(f"Persisted checkpoint probe mismatch: {name}")
                remaining.remove(name)
    if remaining:
        raise RuntimeError(f"Missing persisted checkpoint probes: {sorted(remaining)}")


def verify_hf_reload(checkpoint: Path, expected: dict, row: dict, call: dict, pad: int):
    """Reload on CPU, not a second GPU copy; verify weights and prompt contract."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    restored = AutoModelForCausalLM.from_pretrained(
        checkpoint, local_files_only=True, dtype=torch.float32,
        device_map="cpu", attn_implementation="sdpa",
    )
    if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in restored.parameters()):
        raise RuntimeError("Reloaded HF checkpoint must contain CPU FP32 master weights")
    probes = parameter_probes(restored)
    if probes.keys() != expected.keys() or any(
        not torch.equal(probes[name], expected[name]) for name in expected
    ):
        raise RuntimeError("Reloaded HF checkpoint parameter probes differ")
    del restored
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    restored_call = tokenize_training_rows([row], tokenizer,
                                           len(call["prompt_ids"]) + len(call["response_ids"]))[0]
    if restored_call != call or tokenizer.pad_token_id != pad:
        raise RuntimeError("Reloaded tokenizer changed exact prompt/response tokenization")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value) -> None:
    pending = path.with_name(path.name + ".incomplete")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def training_compatibility(args, run: DistributedRun) -> dict:
    return {
        "world_size": run.world_size, "global_batch_size": args.batch_size,
        "max_length": args.max_length, "micro_batch_size": 1, "grad_clip": 1.0,
        "master_dtype": "float32", "autocast_dtype": "bfloat16",
        "system_message": SYSTEM_MESSAGE, "user_template": USER_TEMPLATE,
    }


def training_state_path(path: Path) -> Path:
    path = path.resolve()
    return path / "training_state.pt" if path.is_dir() else path


def _checkpoint_identity(checkpoint: Path) -> dict:
    if not (checkpoint / "config.json").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise ValueError("Training state requires an HF safetensors checkpoint")
    return {
        str(path.relative_to(checkpoint)): _sha(path)
        for path in sorted(checkpoint.rglob("*")) if path.is_file()
    }


def _parameter_identity(model) -> list[dict]:
    return [
        {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
         "requires_grad": parameter.requires_grad}
        for name, parameter in model.named_parameters()
    ]


def _cpu_state(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_state(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_state(item) for item in value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"Unsupported training-state value: {type(value).__name__}")


def _validate_optimizer_state(state, optimizer, completed_updates: int) -> None:
    import torch

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Training state requires AdamW")
    if type(completed_updates) is not int or completed_updates < 1:
        raise ValueError("Invalid completed optimizer update count")
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise ValueError("Invalid AdamW state structure")
    if state["param_groups"] != optimizer.state_dict()["param_groups"]:
        raise ValueError("AdamW parameter groups/hyperparameters mismatch")
    parameters = {
        index: parameter
        for saved_group, group in zip(state["param_groups"], optimizer.param_groups, strict=True)
        for index, parameter in zip(saved_group["params"], group["params"], strict=True)
    }
    if not isinstance(state["state"], dict) or set(state["state"]) != set(parameters):
        raise ValueError("Missing or unexpected AdamW parameter state")
    for index, parameter in parameters.items():
        values = state["state"][index]
        if not isinstance(values, dict) or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"Invalid AdamW state for parameter {index}")
        step = values["step"]
        if (not isinstance(step, torch.Tensor) or step.device.type != "cpu"
                or step.dtype != torch.float32 or step.shape != torch.Size([])
                or step.item() != completed_updates):
            raise ValueError(f"AdamW step mismatch for parameter {index}")
        for key in ("exp_avg", "exp_avg_sq"):
            value = values[key]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.float32 or value.layout != torch.strided
                    or value.shape != parameter.shape or not torch.isfinite(value).all()
                    or (key == "exp_avg_sq" and (value < 0).any())):
                raise ValueError(f"Invalid FP32 AdamW {key} for parameter {index}")


def _validate_trained_ids(ids) -> None:
    if (not isinstance(ids, list) or not ids
            or any(type(identifier) is not str or not identifier for identifier in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("Training-state train IDs must be nonempty unique strings")


def save_training_state(path: Path, checkpoint: Path, model, optimizer,
                        compatibility: dict, completed_updates: int, trained_ids: list[str]) -> None:
    """Persist only CPU tensors/primitives, after the matching HF save succeeds."""
    import torch

    _validate_trained_ids(trained_ids)
    state = _cpu_state(optimizer.state_dict())
    _validate_optimizer_state(state, optimizer, completed_updates)
    probes = parameter_probes(model)
    verify_saved_probes(checkpoint, probes)
    payload = {
        "format": "whale-native-sft-adamw-v1",
        "checkpoint_files_sha256": _checkpoint_identity(checkpoint),
        "parameters": _parameter_identity(model), "probes": probes,
        "compatibility": compatibility, "completed_updates": completed_updates,
        "trained_ids": trained_ids, "optimizer": state,
    }
    checksum = path.with_name(path.name + ".sha256")
    pending = path.with_name(path.name + ".incomplete")
    pending_checksum = checksum.with_name(checksum.name + ".incomplete")
    if any(item.exists() for item in (path, checksum, pending, pending_checksum)):
        raise FileExistsError(f"Training state must be fresh: {path}")
    try:
        torch.save(payload, pending)
        pending_checksum.write_text(_sha(pending) + "\n")
        pending.replace(path)
        pending_checksum.replace(checksum)
    finally:
        pending.unlink(missing_ok=True)
        pending_checksum.unlink(missing_ok=True)


def restore_training_state(path: Path, checkpoint: Path, model, optimizer,
                           compatibility: dict, train_ids: list[str]) -> tuple[int, list[str]]:
    """Fail closed before mutating the optimizer; never fall back to fresh AdamW."""
    import torch

    path = training_state_path(path)
    checksum = path.with_name(path.name + ".sha256")
    expected = checksum.read_text().strip()
    if (len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected)
            or _sha(path) != expected):
        raise ValueError("Training-state checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    keys = {"format", "checkpoint_files_sha256", "parameters", "probes", "compatibility",
            "completed_updates", "trained_ids", "optimizer"}
    if (not isinstance(payload, dict) or set(payload) != keys
            or payload["format"] != "whale-native-sft-adamw-v1"):
        raise ValueError("Unsupported training-state format")
    if payload["checkpoint_files_sha256"] != _checkpoint_identity(checkpoint):
        raise ValueError("Training-state HF checkpoint identity mismatch")
    if payload["parameters"] != _parameter_identity(model):
        raise ValueError("Training-state model parameter identity mismatch")
    if payload["compatibility"] != compatibility:
        raise ValueError("Training-state training hyperparameters/world_size mismatch")
    probes = parameter_probes(model)
    saved_probes = payload["probes"]
    if (not isinstance(saved_probes, dict) or set(saved_probes) != set(probes)
            or any(not isinstance(saved_probes[name], torch.Tensor)
                   or saved_probes[name].dtype != torch.float32
                   or not torch.equal(saved_probes[name], value) for name, value in probes.items())):
        raise ValueError("Training-state model checkpoint probes mismatch")
    _validate_trained_ids(payload["trained_ids"])
    _validate_trained_ids(train_ids)
    overlap = set(train_ids) & set(payload["trained_ids"])
    if overlap:
        raise ValueError(f"Train IDs already consumed by previous stages: {sorted(overlap)[:8]}")
    _validate_optimizer_state(payload["optimizer"], optimizer, payload["completed_updates"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["completed_updates"], payload["trained_ids"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="GLOBAL examples per optimizer update, divided equally across ranks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--distributed-timeout", type=int, default=300,
                        help="Collective timeout in seconds, including rank-zero I/O phases")
    parser.add_argument("--save-training-state", action="store_true",
                        help="Save CPU AdamW state and checksum alongside the HF checkpoint (one epoch)")
    parser.add_argument("--resume-state", type=Path,
                        help="Previous OUTPUT or OUTPUT/training_state.pt; --model must be its HF checkpoint")
    args = parser.parse_args(argv)
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 2:
        parser.error("--epochs/--batch-size must be positive and --max-length >= 2")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    if args.distributed_timeout < 1:
        parser.error("--distributed-timeout must be positive")
    if (args.save_training_state or args.resume_state) and args.epochs != 1:
        parser.error("Incremental training-state stages require --epochs 1 (each train ID once)")
    return args


def validate_paths(args):
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Output must be fresh: {args.output}")
    for key in ("model", "data", "output"):
        path = getattr(args, key).resolve()
        if not path.is_relative_to(SHARED_ROOT):
            raise ValueError(f"--{key} must be on shared storage under {SHARED_ROOT}: {path}")
        setattr(args, key, path)
    if not args.model.is_dir() or not args.data.is_file():
        raise FileNotFoundError("Expected an existing local model directory and data file")
    if args.output.is_relative_to(args.model):
        raise ValueError("Output must not be inside the input model")
    if getattr(args, "resume_state", None) is not None:
        args.resume_state = training_state_path(args.resume_state)
        if not args.resume_state.is_relative_to(SHARED_ROOT):
            raise ValueError(f"--resume-state must be on shared storage under {SHARED_ROOT}")
        if not args.resume_state.is_file():
            raise FileNotFoundError(f"Missing training state: {args.resume_state}")


def main(argv=None):
    args = parse_args(argv)
    run = DistributedRun.from_environment()
    validate_paths(args)
    from .resources import require_research_node

    require_research_node()
    rows, data_sha = load_records(args.data)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no usable PAD/EOS token")
    model_config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    context = getattr(model_config, "max_position_embeddings", None)
    if context is not None and args.max_length > context:
        raise ValueError(f"--max-length exceeds the model context ({context})")
    calls = tokenize_training_rows(rows, tokenizer, args.max_length)
    plan = plan_batches(calls, args.epochs, args.batch_size, args.seed, args.max_length, run.world_size)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or torch.cuda.device_count() != run.world_size:
        raise RuntimeError(
            f"Select exactly {run.world_size} allocated GPU(s) externally with CUDA_VISIBLE_DEVICES "
            "(exactly one allocated GPU for a plain invocation)"
        )
    torch.cuda.set_device(run.local_rank)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This trainer requires a BF16-capable CUDA GPU (target: H200)")
    started = time.monotonic()
    output_claimed = False

    def prepare_output():
        nonlocal output_claimed
        args.output.mkdir(parents=True, exist_ok=False)
        output_claimed = True
        config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
        config.update(
            world_size=run.world_size, global_batch_size=args.batch_size,
            local_batch_size=args.batch_size // run.world_size,
            partition="global seeded shuffle, rank-strided partition, no repeats or drops",
            gradient_reduction="SUM(local_response_tokens/global_response_tokens * native_local_gradient)",
            master_dtype="float32", autocast_dtype="bfloat16", micro_batch_size=1,
            native_sft_epochs=1, native_grad_clip=1.0, planned_updates=len(plan),
            optimizer={"name": "AdamW", "betas": [0.9, 0.999], "weight_decay": 0.01, "foreach": False},
            system_message=SYSTEM_MESSAGE, user_template=USER_TEMPLATE,
            optimizer_reset=args.resume_state is None, heldout_acceptance="not evaluated",
        )
        _write(args.output / "config.json", config)
        source_files = [Path(__file__), Path(__file__).with_name("legacy_update.py"),
                        Path(__file__).with_name("rewrite_data.py"),
                        Path(__file__).with_name("resources.py"), NATIVE_ACTOR]
        _write(args.output / "provenance.json", {
            "data": str(args.data), "data_sha256": data_sha,
            "split_counts": dict(Counter(row["split"] for row in rows)),
            "train_format_counts": dict(Counter(row["format"] for row in rows if row["split"] == "train")),
            "train_ids": [c["id"] for c in calls], "optimizer_splits": ["train"],
            "train_record_count": len(calls), "expected_examples_seen": len(calls) * args.epochs,
            "dropped_records": 0, "repeated_records_within_epoch": 0,
            "model": str(args.model),
            "model_files_sha256": {p.name: _sha(p) for p in sorted(args.model.iterdir()) if p.is_file()},
            "source_sha256": {str(p.relative_to(ROOT)): _sha(p) for p in source_files},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "cuda_visible_devices": visible, "world_size": run.world_size,
            "host": socket.gethostname(), "writer_rank": 0,
            "teacher_authorship": "Supplied dataset; not independently attested by this trainer",
        })
        _write(args.output / "train_records.json", [row for row in rows if row["split"] == "train"])
        _write(args.output / "tokenized_train.json", calls)
        _write(args.output / "batch_plan.json", plan)

    def load_model():
        model = AutoModelForCausalLM.from_pretrained(
            args.model, local_files_only=True, dtype=torch.float32, attn_implementation="sdpa",
        ).to("cuda")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not model.is_gradient_checkpointing:
            raise RuntimeError("Model did not enable gradient checkpointing")
        return model

    try:
        with distributed_process_group(run, args.distributed_timeout):
            identity = hashlib.sha256(json.dumps(
                [data_sha, calls, plan, {k: str(v) for k, v in vars(args).items()}],
                sort_keys=True,
            ).encode()).hexdigest()
            run.agree("data, tokenization, global grouping and configuration", identity)
            run.root_phase("output initialization/provenance", prepare_output)
            random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            model = run.phase("model loading", load_model)
            if run.world_size > 1:
                # Replicated parameters/buffers start identically even for uninitialized HF keys.
                for tensor in [*model.parameters(), *model.buffers()]:
                    torch.distributed.broadcast(tensor.detach(), src=0)
            before = run.phase("initial FP32 probes", lambda: parameter_probes(model))
            optimizer = run.phase("optimizer construction", lambda: torch.optim.AdamW(
                model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01, foreach=False,
            ))
            initial_updates, previous_ids = 0, []
            compatibility = training_compatibility(args, run)
            if args.resume_state is not None:
                initial_updates, previous_ids = run.phase(
                    "optimizer state restore",
                    lambda: restore_training_state(
                        args.resume_state, args.model, model, optimizer, compatibility,
                        [call["id"] for call in calls],
                    ),
                )
                run.agree("optimizer continuation", [initial_updates, previous_ids])
            actor = run.phase("native actor construction",
                              lambda: make_actor(model, optimizer, args.batch_size // run.world_size))
            torch.cuda.reset_peak_memory_stats()
            training_started = time.monotonic()
            metrics = run_updates(
                actor, optimizer, calls, plan, tokenizer.pad_token_id, args.max_length,
                report=lambda records: _write(args.output / "metrics.json", records),
                run=run, initial_updates=initial_updates,
            )
            training_seconds = max(run.gather(time.monotonic() - training_started))
            deltas, after = run.phase("final finite parameters/deltas",
                                      lambda: check_parameter_deltas(model, before))
            run.agree("final replica probes", deltas)
            peak_memory = run.gather(torch.cuda.max_memory_allocated())
            checkpoint = args.output / "checkpoint"

            def save_checkpoint():
                model.config.use_cache = True
                model.save_pretrained(checkpoint, safe_serialization=True, max_shard_size="4GB")
                tokenizer.save_pretrained(checkpoint)
                verify_saved_probes(checkpoint, after)
                verify_hf_reload(checkpoint, after, next(row for row in rows if row["split"] == "train"),
                                 calls[0], tokenizer.pad_token_id)
                _write(args.output / "parameter_deltas.json", deltas)
                tokens = sum(row["response_tokens"] for row in metrics)
                examples = sum(row["examples"] for row in metrics)
                if examples != len(calls) * args.epochs:
                    raise RuntimeError("Observed example count differs from the complete global plan")
                if args.save_training_state:
                    save_training_state(
                        args.output / "training_state.pt", checkpoint, model, optimizer,
                        compatibility, initial_updates + len(metrics),
                        previous_ids + [call["id"] for call in calls],
                    )
                result = {
                    "status": "completed", "updated": True, "checkpoint": str(checkpoint),
                    "world_size": run.world_size, "optimizer_updates": len(metrics),
                    "response_tokens": tokens, "examples_seen": examples,
                    "train_records": len(calls), "dropped_records": 0,
                    "token_weighted_train_loss": sum(
                        row["native"]["actor/sft_loss"] * row["response_tokens"] for row in metrics
                    ) / tokens,
                    "all_parameters_finite": True, "sampled_parameters_changed": True,
                    "replica_probes_agree": True, "saved_probes_verified": True,
                    "hf_cpu_reload_verified": True, "tokenizer_reload_verified": True,
                    "heldout_acceptance": "not evaluated",
                    "peak_cuda_allocated_bytes": max(peak_memory),
                    "rank_peak_cuda_allocated_bytes": peak_memory,
                    "training_seconds": training_seconds,
                    "examples_per_training_second": examples / training_seconds,
                    "response_tokens_per_training_second": tokens / training_seconds,
                    "elapsed_seconds": time.monotonic() - started,
                }
                if args.save_training_state or args.resume_state is not None:
                    result.update(
                        optimizer_reset=args.resume_state is None,
                        initial_optimizer_updates=initial_updates,
                        cumulative_optimizer_updates=initial_updates + len(metrics),
                        cumulative_train_records=len(previous_ids) + len(calls),
                        training_state=(str(args.output / "training_state.pt")
                                        if args.save_training_state else None),
                    )
                _write(args.output / "result.json", result)
                print(json.dumps(result), flush=True)

            run.root_phase("checkpoint save/reload verification", save_checkpoint)
    except BaseException as exc:
        if run.rank == 0 and output_claimed:
            _write(args.output / "result.json", {
                "status": "failed", "updated": False, "error": f"{type(exc).__name__}: {exc}",
                "heldout_acceptance": "not evaluated",
            })
        raise


if __name__ == "__main__":
    main()
