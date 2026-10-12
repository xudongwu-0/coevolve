"""Fresh full-parameter Harness Engineer SFT (four/eight-rank torchrun, no adapters).

Default: Qwen3.5-9B, eight ranks, minimum 256 bootstrap examples. An explicit
Qwen3-4B-Instruct-2507 pilot supports minimum 32 with CLI-pinned provenance
and --engineer-think-prefill, without changing original base assets.
Only text-language-model parameters participate: the unused image tower and
upstream MTP head are deliberately not instantiated. Every language parameter
is trained, with FP32 AdamW master weights/states and BF16 computation.

JSON/JSONL rows contain id, split="train", and exactly system/user/assistant
messages, or verified bootstrap messages + positive train provenance with a
sibling COMPLETE.json. Assistant content is the literal continuation AFTER "<think>\\n",
including reasoning, "</think>", and one "<patch>... </patch>" block.
We do not re-render the assistant through the chat template, which would
normalize its reasoning whitespace. No truncation or dropped supervision.

--inspect-base-only verifies the official download receipt and tensor schema on
CPU. --preflight-only additionally verifies/tokenizes an actual supplied dataset.
Neither mode initializes CUDA. An externally approved SHA256 is mandatory for
training data. The output is an atomic, fresh HF text checkpoint plus provenance;
it is not an optimizer-resume checkpoint or an evaluation acceptance claim.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import timedelta
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "assets/Qwen3.5-9B"
PILOT_BASE = ROOT / "assets/Qwen3-4B-Instruct-2507"
BASE_RECEIPT_SHA256 = "e5b41d3fcd2d0211d7c7bb27ca0170352a0de5e5c2417fee9b9d29b82bfbb3ae"
if str(ROOT / "whale_local") not in sys.path:
    sys.path.insert(0, str(ROOT / "whale_local"))
from sao_bridge.rewrite_sft_train import _unique_object, _reject_constant, _sha


def read_json(payload):
    return json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)


def digest_map(files: dict[str, str]) -> str:
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_sha(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Expected a lowercase SHA256 (64 hexadecimal characters)")
    return value


def verify_bootstrap_completion(path: Path, data_sha: str, count: int, minimum_examples: int = 256):
    manifest_path = path.parent / "COMPLETE.json"
    complete = read_json(manifest_path.read_bytes())
    minimum = complete.get("minimum_examples")
    if (complete.get("stage") != "fresh_base_silver_collection"
            or complete.get("usable_for_sft") is not True
            or complete.get("dataset_sha256") != data_sha
            or complete.get("examples") != count
            or type(minimum) is not int or minimum < minimum_examples or count < minimum):
        raise ValueError(f"Bootstrap completion does not certify at least {minimum_examples} approved positive examples")
    teacher_sha = validate_sha(complete["input_identity"]["engineer_manifest_sha256"])
    return teacher_sha, _sha(manifest_path)


def load_records(path: Path, expected_sha: str, minimum_examples: int = 256):
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != validate_sha(expected_sha):
        raise ValueError("Dataset SHA256 mismatch")
    if path.suffix == ".jsonl":
        rows = [read_json(line) for line in payload.splitlines() if line.strip()]
    else:
        rows = read_json(payload)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Dataset must be a nonempty JSON array or JSONL")
    bootstrap = all(isinstance(r, dict) and set(r) == {"messages", "provenance"} for r in rows)
    teacher_sha = verify_bootstrap_completion(path, actual, len(rows), minimum_examples)[0] if bootstrap else None
    seen = set()
    for i, row in enumerate(rows):
        if bootstrap:
            provenance = row["provenance"]
            fields = {"generation_id", "engineer_manifest_sha256", "task_ids",
                      "reward_delta", "patch_sha256", "split"}
            if not isinstance(provenance, dict) or set(provenance) != fields:
                raise ValueError(f"Row {i}: invalid bootstrap provenance fields")
            delta, task_ids = provenance["reward_delta"], provenance["task_ids"]
            if (provenance["split"] != "train" or type(delta) not in (int, float)
                    or not math.isfinite(delta) or delta <= 0
                    or provenance["engineer_manifest_sha256"] != teacher_sha
                    or not isinstance(task_ids, list) or not task_ids
                    or any(not isinstance(t, str) or not t.strip() for t in task_ids)
                    or len(set(task_ids)) != len(task_ids)):
                raise ValueError(f"Row {i}: bootstrap must bind positive train-only rerun provenance")
            validate_sha(provenance["patch_sha256"])
            row = {"id": provenance["generation_id"], "split": "train", "messages": row["messages"]}
            rows[i] = row
        if not isinstance(row, dict) or set(row) != {"id", "split", "messages"}:
            raise ValueError(f"Row {i}: expected exactly id, split, messages")
        identifier = row["id"]
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError(f"Row {i}: empty or duplicate id")
        seen.add(identifier)
        if row["split"] != "train":
            raise ValueError(f"Row {i}: only train rows permitted; do not pass held-out data")
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Row {i}: expected three messages")
        for message, role in zip(messages, ("system", "user", "assistant")):
            if (not isinstance(message, dict) or set(message) != {"role", "content"}
                    or message["role"] != role or not isinstance(message["content"], str)
                    or not message["content"].strip()):
                raise ValueError(f"Row {i}: invalid {role} message")
            if any(token in message["content"] for token in ("<|im_start|>", "<|im_end|>", "<|endoftext|>")):
                raise ValueError(f"Row {i}: reserved conversation tokens")
        validate_continuation(messages[-1]["content"])
    return rows, actual


def validate_continuation(content: str):
    if "<think>" in content or content.count("</think>") != 1:
        raise ValueError("Assistant must continue the existing thinking prefill, closing it exactly once")
    reasoning, patch = content.split("</think>")
    if not reasoning.strip() or not patch.strip().startswith("<patch>") or not patch.strip().endswith("</patch>"):
        raise ValueError("Expected nonempty reasoning then </think> and a <patch> JSON block")
    patch = patch.strip()
    if patch.count("<patch>") != 1 or patch.count("</patch>") != 1:
        raise ValueError("Expected exactly one patch block")
    value = read_json(patch[len("<patch>"):-len("</patch>")])
    if not isinstance(value, dict):
        raise ValueError("Patch must contain a JSON object")


def tokenize_rows(rows, tokenizer, max_length: int):
    if tokenizer.eos_token != "<|im_end|>" or type(tokenizer.eos_token_id) is not int:
        raise ValueError("Expected the Qwen conversation EOS <|im_end|>")
    calls = []
    for row in rows:
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        if not prompt.endswith("<|im_start|>assistant\n<think>\n"):
            raise ValueError("Tokenizer did not produce the exact Qwen thinking prefill")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=False)
        target = messages[-1]["content"]
        target_ids = tokenizer.encode(target, add_special_tokens=False, truncation=False)
        # Independent encoding must preserve the same boundary as actual generation.
        if tokenizer.encode(prompt + target, add_special_tokens=False, truncation=False) != prompt_ids + target_ids:
            raise ValueError(f"{row['id']}: non-compositional prompt/continuation token boundary")
        if not prompt_ids or not target_ids:
            raise ValueError(f"{row['id']}: empty token sequence")
        if any(t in tokenizer.all_special_ids for t in target_ids):
            raise ValueError(f"{row['id']}: reserved tokens inside the assistant continuation")
        ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
        if len(ids) > max_length:
            raise ValueError(f"{row['id']}: {len(ids)} tokens exceed {max_length}; truncation forbidden")
        calls.append({"id": row["id"], "input_ids": ids,
                      "labels": [-100] * len(prompt_ids) + target_ids + [tokenizer.eos_token_id],
                      "target_tokens": len(target_ids) + 1})
    return calls


def configure_engineer_template(tokenizer, enabled: bool):
    original = tokenizer.chat_template
    if not isinstance(original, str) or not original:
        raise ValueError("Expected one explicit original chat template")
    before = hashlib.sha256(original.encode()).hexdigest()
    if enabled:
        tokenizer.chat_template = original + (
            "\n{%- if add_generation_prompt %}{{- '<think>\\n' }}{%- endif %}"
        )
    return {"engineer_think_prefill": enabled, "original_sha256": before,
            "exported_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}


def plan_batches(count: int, epochs: int, batch_size: int, world_size: int, seed: int):
    if min(count, epochs, batch_size, world_size) < 1 or batch_size % world_size:
        raise ValueError("Positive train count required; global batch must divide across ranks")
    rng = random.Random(seed)
    plan = []
    for epoch in range(1, epochs + 1):
        order = list(range(count))
        rng.shuffle(order)
        for start in range(0, count, batch_size):
            plan.append((epoch, order[start:start + batch_size]))
    return plan


def rank_slots(indices, rank: int, world_size: int):
    """Equal collective counts, with None denoting a zero-supervision idle slot."""
    return [indices[offset + rank] if offset + rank < len(indices) else None
            for offset in range(0, len(indices), world_size)]


def configure_caches():
    cache = ROOT / "assets/runtime-cache/harness-engineer-sft"
    cache.mkdir(parents=True, exist_ok=True)
    for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR",
                 "CUDA_CACHE_PATH", "TORCHINDUCTOR_CACHE_DIR"):
        os.environ[name] = str(cache / name.lower())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def verify_base(base: Path, expected_receipt_sha: str):
    native = base.resolve() == PILOT_BASE.resolve()
    receipt_path = base / ("provenance.json" if native else "download_receipt.json")
    if _sha(receipt_path) != validate_sha(expected_receipt_sha):
        raise ValueError("Official base download receipt SHA256 mismatch")
    receipt = read_json(receipt_path.read_bytes())
    repository = "Qwen/Qwen3-4B-Instruct-2507" if native else "Qwen/Qwen3.5-9B"
    if native and (receipt.get("repository") != repository or not receipt.get("revision")):
        raise ValueError("Native Qwen3 receipt must identify the original official repository/revision")
    entries = receipt.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Missing verified official base file inventory")
    files = {}
    for entry in entries:
        name = entry["path"]
        if (not isinstance(name, str) or Path(name).name != name or name in files
                or (not native and entry.get("verified") is not True)
                or not entry.get("url" if native else "source", "").startswith(
                    f"https://huggingface.co/{repository}/resolve/"
                    + (receipt["revision"] + "/" if native else ""))):
            raise ValueError("Receipt is not an official verified Qwen base download")
        files[name] = validate_sha(entry["sha256"])
        if _sha(base / name) != files[name]:
            raise ValueError(f"Base file checksum mismatch: {name}")
    required = {"config.json", "model.safetensors.index.json", "tokenizer.json",
                "tokenizer_config.json"}
    if not native:
        required.add("chat_template.jinja")
    if not required <= files.keys():
        raise ValueError("Receipt omits required base files")
    index = read_json((base / "model.safetensors.index.json").read_bytes())["weight_map"]
    if not set(index.values()) <= files.keys():
        raise ValueError("Receipt omits model shards")
    return {"receipt_sha256": expected_receipt_sha, "receipt_file": receipt_path.name,
            "repository": repository, "files_sha256": files,
            "tree_sha256": digest_map(files)}


def make_text_model(base: Path):
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    full = AutoConfig.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    if full.model_type not in {"qwen3_5", "qwen3"}:
        raise ValueError("Expected an official Qwen3.5 or native Qwen3 base config")
    config = full.text_config if full.model_type == "qwen3_5" else full
    config.use_cache = False
    config._attn_implementation = "sdpa"
    # Keep nonpersistent rotary buffers materialized; to_empty cannot recreate
    # their values from a completely-meta construction.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float32, trust_remote_code=False)
    # Accelerate's parameter registration can separate shared meta Parameters.
    if config.tie_word_embeddings:
        model.tie_weights()
    return model


def checkpoint_schema(base: Path, model):
    """Validate actual safetensors headers against every text model parameter."""
    from safetensors import safe_open

    index = read_json((base / "model.safetensors.index.json").read_bytes())["weight_map"]
    expected = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    native = model.config.model_type == "qwen3"
    mapping, ignored = {}, []
    for key in index:
        if native and key in expected:
            mapped = key
        elif not native and key.startswith("model.language_model."):
            mapped = "model." + key[len("model.language_model."):]
        elif key == "lm_head.weight":
            mapped = key
        elif not native and key.startswith(("model.visual.", "mtp.")):
            ignored.append(key)
            continue
        else:
            raise ValueError(f"Unrecognized base tensor: {key}")
        if mapped not in expected or mapped in mapping:
            raise ValueError(f"Unexpected/duplicate text tensor: {key}")
        mapping[mapped] = (key, index[key])
    missing = set(expected) - set(mapping)
    if model.config.tie_word_embeddings and "model.embed_tokens.weight" in mapping:
        missing -= {"lm_head.weight"}
    if missing:
        raise ValueError(f"Missing text tensors: {sorted(missing)}")
    for shard in sorted(set(index.values())):
        if Path(shard).name != shard:
            raise ValueError("Unsafe model shard path")
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            if set(handle.keys()) != {k for k, v in index.items() if v == shard}:
                raise ValueError(f"Shard/index inventory mismatch: {shard}")
            for mapped, (key, filename) in mapping.items():
                if filename == shard and tuple(handle.get_slice(key).get_shape()) != expected[mapped]:
                    raise ValueError(f"Model tensor shape mismatch: {key}")
    return mapping, ignored


def load_text_weights(base: Path, model, mapping):
    """Load only verified text weights, strictly, without HF's permissive remapping."""
    import torch
    from safetensors import safe_open

    for shard in sorted({v[1] for v in mapping.values()}):
        with safe_open(base / shard, framework="pt", device="cpu") as handle:
            values = {name: handle.get_tensor(key).to(torch.float32)
                      for name, (key, filename) in mapping.items() if filename == shard}
        model.load_state_dict(values, strict=False, assign=True)
    if model.config.tie_word_embeddings:
        if "lm_head.weight" in mapping:
            validate_tied_state(model.state_dict())
        model.tie_weights()
        if model.get_input_embeddings().weight is not model.get_output_embeddings().weight:
            raise ValueError("Native tied embedding/head integrity lost")
    if any(p.is_meta or p.dtype != torch.float32 or not p.requires_grad for p in model.parameters()):
        raise ValueError("Text model is not completely loaded, trainable FP32")
    return model


def validate_tied_state(weights):
    import torch
    if not torch.equal(weights["model.embed_tokens.weight"], weights["lm_head.weight"]):
        raise ValueError("Tied embedding and output-head tensors disagree")


def materialize_meta_module(module, device, cache: dict, shared_ids=None):
    """Preserve shared Parameter identity across FSDP's per-module init calls."""
    import torch
    for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
        if parameter.is_meta:
            identity = id(parameter)
            replacement = cache.get(identity)
            if replacement is None:
                replacement = torch.nn.Parameter(torch.empty_like(parameter, device=device),
                                                 requires_grad=parameter.requires_grad)
                if shared_ids is None or identity in shared_ids:
                    cache[identity] = replacement
            module._parameters[name] = replacement
    for name, buffer in module.named_buffers(recurse=False):
        if buffer.is_meta:
            raise ValueError(f"Unexpected meta buffer: {name}; preserve rotary initialization")


def chunked_loss(hidden, labels, head, chunk_size: int):
    """Exact summed causal CE; checkpoint the large vocabulary projection chunks."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    shifted = labels[:, 1:].reshape(-1)
    selected = shifted != -100
    targets = shifted[selected]
    states = hidden[:, :-1, :].reshape(-1, hidden.shape[-1])[selected]
    if not targets.numel():
        raise ValueError("No assistant target tokens")

    def project_loss(x, y):
        return F.cross_entropy(head(x).float(), y, reduction="sum")

    loss = states.new_zeros((), dtype=torch.float32)
    for start in range(0, len(targets), chunk_size):
        loss = loss + checkpoint(project_loss, states[start:start + chunk_size],
                                 targets[start:start + chunk_size], use_reentrant=False)
    return loss


def loss_module(model, chunk_size):
    import torch

    class AssistantLoss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model

        def forward(self, input_ids, labels):
            # Microbatch size one, no padding: no quadratic explicit attention mask.
            hidden = self.base.model(input_ids=input_ids, use_cache=False).last_hidden_state
            return chunked_loss(hidden, labels, self.base.lm_head, chunk_size)

    return AssistantLoss()


def configure_generation(model, tokenizer):
    # The upstream text config names endoftext; our chat target ends im_end.
    for config in (model.config, model.generation_config):
        config.eos_token_id = tokenizer.eos_token_id
        config.pad_token_id = tokenizer.pad_token_id


def source_identity():
    paths = [Path(__file__).resolve(), ROOT / "whale_local/sao_bridge/rewrite_sft_train.py"]
    config = ROOT / "configs/searchqa_engineer_sft.json"
    if config.exists():
        paths.append(config)
    return {str(p.relative_to(ROOT)): _sha(p) for p in paths}


def text_probes(weights):
    names = ("lm_head.weight", "model.norm.weight", "model.layers.0.input_layernorm.weight")
    return {name: weights[name].detach().reshape(-1)[:64].float().cpu().clone() for name in names}


def verify_changed_probes(before, weights):
    import torch

    after = text_probes(weights)
    result = {}
    for name, value in before.items():
        result[name] = {
            "changed": not torch.equal(value, after[name]),
            "before_sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest(),
            "after_sha256": hashlib.sha256(after[name].numpy().tobytes()).hexdigest(),
        }
    if not any(probe["changed"] for probe in result.values()):
        raise ValueError("Checkpoint probes unchanged; refusing to certify a no-op training run")
    return result


def verify_saved_checkpoint(directory: Path, expected_shapes: dict, tied_embeddings: bool = False):
    import torch
    from safetensors import safe_open

    found = set()
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in found or key not in expected_shapes:
                    raise ValueError(f"Unexpected saved tensor: {key}")
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != expected_shapes[key]:
                    raise ValueError(f"Saved shape mismatch: {key}")
                # Bound verifier scratch memory even for the 1B-element embedding.
                flat = tensor.reshape(-1)
                for start in range(0, flat.numel(), 1_000_000):
                    if not torch.isfinite(flat[start:start + 1_000_000]).all():
                        raise ValueError(f"Nonfinite saved tensor: {key}")
                found.add(key)
    missing = set(expected_shapes) - found
    if tied_embeddings and "model.embed_tokens.weight" in found:
        missing -= {"lm_head.weight"}
    if missing:
        raise ValueError("Saved checkpoint tensor coverage incomplete")
    if tied_embeddings and "lm_head.weight" in found:
        weights = {}
        for path in sorted(directory.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in ("model.embed_tokens.weight", "lm_head.weight"):
                    if name in handle.keys():
                        weights[name] = handle.get_tensor(name)
        validate_tied_state(weights)
    return {p.name: _sha(p) for p in sorted(directory.iterdir()) if p.is_file()}


def publish_checkpoint(staging: Path, output: Path, manifest: dict):
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output must be fresh: {output}")
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (staging / "COMPLETE").write_text("VERIFIED COMPLETE\n" + _sha(manifest_path) + "\n")
    for path in staging.iterdir():
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    staging.rename(output)
    fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=BASE)
    p.add_argument("--base-receipt-sha256", default=BASE_RECEIPT_SHA256,
                   help="Pinned receipt hash; 4B requires its original provenance.json SHA256 explicitly")
    p.add_argument("--data", type=Path)
    p.add_argument("--data-sha256", help="SHA256 of the actually approved, frozen dataset bytes")
    p.add_argument("--output", type=Path)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--purpose", choices=("training", "performance-profile"), default="training")
    p.add_argument("--global-batch-size", type=int, default=24)
    p.add_argument("--world-size", type=int, choices=(4, 8), default=8)
    p.add_argument("--pilot-minimum-examples", type=int, choices=(32, 256), default=256,
                   help="Explicit 32-example preliminary 4B pilot; default bootstrap minimum remains 256")
    p.add_argument("--engineer-think-prefill", action="store_true",
                   help="Append thinking prefill only to generation prompts of native 4B template in memory/export")
    p.add_argument("--max-length", type=int, default=32768)
    p.add_argument("--loss-chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--distributed-timeout", type=int, default=3600)
    p.add_argument("--inspect-base-only", action="store_true")
    p.add_argument("--preflight-only", action="store_true")
    args = p.parse_args(argv)
    if not args.inspect_base_only and (args.data is None or args.data_sha256 is None):
        p.error("--data and --data-sha256 are required (no mock/formal fallback)")
    if not (args.inspect_base_only or args.preflight_only) and args.output is None:
        p.error("--output is required for training")
    if (not math.isfinite(args.lr) or args.lr <= 0 or args.epochs < 1
            or args.global_batch_size < args.world_size or args.global_batch_size % args.world_size
            or not 2 <= args.max_length <= 32768 or args.loss_chunk_size < 1
            or args.distributed_timeout < 1 or not 0 <= args.seed < 2**32):
        p.error("Invalid learning rate, epochs, batch, context, chunk, timeout, or seed")
    native = args.model.resolve() == PILOT_BASE.resolve()
    if args.pilot_minimum_examples == 32 and not native:
        p.error("--pilot-minimum-examples 32 is restricted to the canonical native 4B pilot")
    if native and args.base_receipt_sha256 == BASE_RECEIPT_SHA256:
        p.error("4B requires an explicit --base-receipt-sha256 pin for original provenance.json")
    if native and not args.engineer_think_prefill and not args.inspect_base_only:
        p.error("Native 4B Engineer SFT requires --engineer-think-prefill")
    if not native and args.engineer_think_prefill:
        p.error("--engineer-think-prefill is only for the native 4B instruct template")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.model.resolve() not in {BASE.resolve(), PILOT_BASE.resolve()}:
        raise ValueError("Cold-start requires the canonical official 9B or native 4B base, never a trained editor")
    native = args.model.resolve() == PILOT_BASE.resolve()
    for name in ("data", "output"):
        path = getattr(args, name)
        if path is not None:
            path = path.resolve()
            if not path.is_relative_to(Path("/mnt/3fs/xudong")):
                raise ValueError(f"--{name} must be under /mnt/3fs/xudong")
            setattr(args, name, path)
    if args.output is not None:
        if args.output.exists() or any(args.output.is_relative_to(base.resolve()) for base in (BASE, PILOT_BASE)):
            raise ValueError("Output must be fresh and outside the base model")
    configure_caches()
    import torch
    import torch.distributed as dist
    import transformers
    from transformers import AutoTokenizer

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    inspect_only = args.inspect_base_only or args.preflight_only
    if not inspect_only:
        if world != args.world_size or not torch.cuda.is_available() or torch.cuda.device_count() != world:
            raise RuntimeError(f"Full SFT requires torchrun with exactly {args.world_size} allocated visible GPUs")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 GPU computation is required")
        dist.init_process_group("nccl", timeout=timedelta(seconds=args.distributed_timeout))
    try:
        # Only rank zero hashes the base; the authenticated inventory is broadcast.
        objects = [verify_base(args.model, args.base_receipt_sha256) if rank == 0 else None]
        if not inspect_only:
            dist.broadcast_object_list(objects, src=0)
        base_identity = objects[0]
        model = make_text_model(args.model)
        mapping, ignored = checkpoint_schema(args.model, model)
        shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
        parameter_count = sum(p.numel() for p in model.parameters())
        if native:
            if (model.config.model_type != "qwen3" or model.config.num_hidden_layers != 36
                    or not model.config.tie_word_embeddings or not 3_000_000_000 < parameter_count < 5_000_000_000):
                raise ValueError("Not the native tied-weight Qwen3-4B architecture")
        elif parameter_count < 8_000_000_000 or model.config.num_hidden_layers != 32:
            raise ValueError("Not the Qwen3.5-9B text parameter architecture")
        if args.inspect_base_only:
            print(json.dumps({"base": base_identity, "text_parameters": parameter_count,
                              "text_tensors": len(mapping), "unused_vision_mtp_tensors": len(ignored)}, indent=2))
            return
        rows, data_sha = load_records(args.data, args.data_sha256, args.pilot_minimum_examples)
        if len(rows) < args.pilot_minimum_examples:
            raise ValueError(f"Need at least {args.pilot_minimum_examples} approved train examples")
        bootstrap_complete = args.data.parent / "COMPLETE.json"
        bootstrap_sha = _sha(bootstrap_complete) if bootstrap_complete.is_file() else None
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        template_identity = configure_engineer_template(tokenizer, args.engineer_think_prefill)
        if native and tokenizer.eos_token_id != 151645:
            raise ValueError("Native Qwen3-4B conversation EOS must be 151645")
        configure_generation(model, tokenizer)
        calls = tokenize_rows(rows, tokenizer, args.max_length)
        plan = plan_batches(len(calls), args.epochs, args.global_batch_size, args.world_size, args.seed)
        sources = source_identity()
        if args.preflight_only:
            print(json.dumps({"data_sha256": data_sha, "rows": len(rows), "updates": len(plan),
                              "max_tokens": max(len(c["input_ids"]) for c in calls),
                              "preliminary": args.pilot_minimum_examples == 32,
                              "minimum_examples": args.pilot_minimum_examples,
                              "world_size": args.world_size, "template": template_identity,
                              "source_sha256": sources, "base_tree_sha256": base_identity["tree_sha256"]}))
            return
        # Ensure rank-local data, tokenizer outputs and sources agree before the first update.
        identity = digest_map({"base": base_identity["tree_sha256"], "data": data_sha,
                              "source": digest_map(sources),
                              "template": template_identity["exported_sha256"],
                              "tokens": hashlib.sha256(json.dumps(calls, sort_keys=True).encode()).hexdigest()})
        identities = [None] * world
        dist.all_gather_object(identities, identity)
        if len(set(identities)) != 1:
            raise ValueError("Ranks disagree on base, data, sources or tokenization")
        if rank == 0:
            load_text_weights(args.model, model, mapping)
            initial_probes = text_probes(model.state_dict())
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy,
            StateDictType, FullStateDictConfig,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl,
        )
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        decoder_type = Qwen3DecoderLayer if native else Qwen3_5DecoderLayer

        device = torch.device("cuda", local_rank)

        parameter_cache = {}
        parameter_counts = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
        shared_ids = {identity for identity, count in parameter_counts.items() if count > 1}

        def init_empty(module):
            materialize_meta_module(module, device, parameter_cache, shared_ids)

        trainer = FSDP(
            loss_module(model, args.loss_chunk_size),
            auto_wrap_policy=partial(transformer_auto_wrap_policy,
                                     transformer_layer_cls={decoder_type}),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                           reduce_dtype=torch.float32, buffer_dtype=torch.float32),
            param_init_fn=init_empty, sync_module_states=True, device_id=device,
            use_orig_params=False, limit_all_gathers=True,
        )
        parameter_cache.clear()
        apply_activation_checkpointing(
            trainer,
            checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
            check_fn=lambda module: isinstance(module, decoder_type),
        )
        optimizer = torch.optim.AdamW(trainer.parameters(), lr=args.lr, betas=(0.9, 0.999),
                                      eps=1e-8, weight_decay=0.0, foreach=False)
        trainer.train()
        completed = 0
        step_profiles = []
        for epoch, indices in plan:
            torch.cuda.synchronize(device)
            step_started = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            token_count = sum(calls[i]["target_tokens"] for i in indices)
            local_loss = torch.zeros((), device=device, dtype=torch.float32)
            for i in rank_slots(indices, rank, world):
                # A short final group may leave ranks idle. Borrow a REAL row
                # solely to execute matching FSDP collectives, with zero loss.
                # It contributes no targets, gradients, metrics or dataset row.
                call = calls[indices[0] if i is None else i]
                ids = torch.tensor([call["input_ids"]], device=device)
                labels = torch.tensor([call["labels"]], device=device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = trainer(ids, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at update {completed + 1}")
                # FSDP averages ranks on every backward; no no_sync full-gradient buffer.
                scale = 0.0 if i is None else world / token_count
                (loss * scale).backward()
                if i is not None:
                    local_loss += loss.detach()
            norm = trainer.clip_grad_norm_(1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            completed += 1
            dist.all_reduce(local_loss)
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - step_started
            input_tokens = sum(len(calls[i]["input_ids"]) for i in indices)
            profile = {
                "update": completed, "step_seconds": seconds,
                "input_tokens": input_tokens, "input_tokens_per_second": input_tokens / seconds,
                "supervised_tokens": token_count,
                "supervised_tokens_per_second": token_count / seconds,
                "peak_gpu_allocated_bytes_rank0": torch.cuda.max_memory_allocated(device),
            }
            if rank == 0:
                step_profiles.append(profile)
                print(json.dumps({"epoch": epoch, "update": completed,
                                  "assistant_token_loss": local_loss.item() / token_count,
                                  "grad_norm": norm.item(), **profile}), flush=True)
        if completed != len(plan):
            raise ValueError("Training did not complete all planned updates")
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        with FSDP.state_dict_type(trainer, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = trainer.state_dict()
        if rank == 0:
            if _sha(args.data) != data_sha or source_identity() != sources:
                raise ValueError("Dataset or training sources changed during training")
            if bootstrap_sha is not None and _sha(bootstrap_complete) != bootstrap_sha:
                raise ValueError("Bootstrap completion provenance changed during training")
            if verify_base(args.model, args.base_receipt_sha256) != base_identity:
                raise ValueError("Base changed during training")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            staging = args.output.with_name(args.output.name + ".incomplete-" + uuid.uuid4().hex)
            staging.mkdir()
            weights = {k.removeprefix("base."): v for k, v in state.items()}
            if set(weights) != set(shapes):
                raise ValueError("FSDP full checkpoint coverage mismatch")
            if model.config.tie_word_embeddings:
                validate_tied_state(weights)
            probes = verify_changed_probes(initial_probes, weights)
            model.save_pretrained(staging, state_dict=weights, safe_serialization=True, max_shard_size="4GB")
            tokenizer.save_pretrained(staging)
            saved_hashes = verify_saved_checkpoint(staging, shapes, model.config.tie_word_embeddings)
            manifest = {
                "status": "VERIFIED COMPLETE", "format": "harness-engineer-full-sft-v1",
                "data": {"path": str(args.data), "sha256": data_sha, "rows": len(rows),
                         "bootstrap_completion_sha256": bootstrap_sha},
                "source_files_sha256": sources, "base_model": base_identity,
                "checkpoint_files_sha256": saved_hashes,
                "parameter_update_probes": probes,
                "purpose": args.purpose,
                "performance": {"steps": step_profiles},
                "preliminary": args.pilot_minimum_examples == 32,
                "minimum_examples": args.pilot_minimum_examples,
                "tokenizer_template": template_identity,
                "training": {"lr": args.lr, "epochs": args.epochs, "global_batch_size": args.global_batch_size,
                             "micro_batch_size": 1, "gradient_accumulation": args.global_batch_size // world,
                             "final_batch_size": len(plan[-1][1]),
                             "idle_rank_slots": "real-row forwards with exactly zero supervised loss",
                             "world_size": world, "max_length": args.max_length, "seed": args.seed,
                             "updates": completed, "compute_dtype": "bfloat16", "master_dtype": "float32",
                             "optimizer": "AdamW", "full_text_parameters": parameter_count,
                             "model_type": model.config.model_type,
                             "tie_word_embeddings": model.config.tie_word_embeddings,
                             "eos_token_id": tokenizer.eos_token_id,
                             "unused_vision_mtp_tensors": ignored,
                             "loss": "global-assistant-token-mean-including-im_end",
                             "sharding": "FSDP FULL_SHARD"},
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            }
            publish_checkpoint(staging, args.output, manifest)
            print(f"VERIFIED COMPLETE: {args.output}", flush=True)
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
