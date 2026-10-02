"""Disk-synchronized vLLM rollout worker for online FSDP GRPO.

Every invocation creates fresh engines from the committed actor directory.
Reward exceptions propagate; infrastructure failures never become zero rewards.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace


UPSTREAM_COMMIT = "94f2e087f573e1b82fc9568bb634d4ae9a887e28"
BACKEND = "fsdp-vllm-adaptation-not-exact-relax"


def register_text_engine():
    from vllm import ModelRegistry
    # vLLM 0.19.1 ships this implementation but only registers the VLM wrapper.
    # Lazy registration uses the installed native text model, not HF fallback.
    if "Qwen3_5ForCausalLM" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "Qwen3_5ForCausalLM",
            "vllm.model_executor.models.qwen3_5:Qwen3_5ForCausalLM")


class Qwen35TextWorkerExtension:
    """Import hook ensures spawned vLLM TP workers register the text class."""


if os.environ.get("HARNESS_R1_REGISTER_VLLM_TEXT") == "1":
    register_text_engine()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: Path):
    files = sorted(p for p in path.rglob("*") if p.is_file()
                   and p.name != "actor_identity.json"
                   and (p.suffix in {".safetensors", ".bin", ".json", ".jinja"}
                        or p.name == "tokenizer.model"))
    if not files or not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError(f"no local HF model weights: {path}")
    hashes = {str(p.relative_to(path)): sha256(p) for p in files}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {"sha256": digest, "files": hashes}


def atomic_json(path: Path, value):
    staging = path.with_name(path.name + f".staging-{os.getpid()}")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(staging, path)


def thinking_prompt(tokenizer, messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt must be a nonempty message list")
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    # Do not inject or duplicate the opening; train the continuation only.
    if not text.rstrip().endswith("<think>"):
        raise ValueError("tokenizer does not implement the required thinking prefill")
    return tokenizer.encode(text, add_special_tokens=False)


def validate_rollouts(records, rows, expected_actor_sha, *, groups=4, responses=8):
    if len(rows) != groups or len(records) != groups * responses:
        raise ValueError("rollout must have four complete groups of eight")
    for index, record in enumerate(records):
        group = index // responses
        if record["group"] != group or record["metadata"] != rows[group]["metadata"]:
            raise ValueError("reward metadata/group mismatch")
        if record["actor_sha256"] != expected_actor_sha:
            raise ValueError("stale rollout actor")
        tokens, logprobs = record["response_ids"], record["rollout_logprobs"]
        if not record["prompt_ids"] or not tokens or len(tokens) != len(logprobs):
            raise ValueError("missing/misaligned generated ids and logprobs")
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("invalid generated token id")
        if any(not math.isfinite(value) for value in logprobs):
            raise ValueError("nonfinite rollout logprob")


def validate_rewards(results, count):
    if not isinstance(results, list) or len(results) != count:
        raise ValueError("reward function returned the wrong sample count")
    scores = []
    for result in results:
        if result.get("eval_status") not in {"ok", "invalid_patch_treated_as_no_patch"}:
            raise RuntimeError(f"reward infrastructure failed: {result}")
        score = float(result["score"])
        if not math.isfinite(score) or score != float(result["delta_score"]):
            raise ValueError("reward must be finite delta score without validity bonus")
        scores.append(score)
    return scores


def resident_signature(actor: Path, config: dict) -> dict:
    model_config = json.loads((actor / "config.json").read_text())
    for key in ("_name_or_path", "torch_dtype", "dtype", "transformers_version"):
        model_config.pop(key, None)
    tokenizer_files = {
        str(path.relative_to(actor)): sha256(path)
        for path in actor.rglob("*") if path.is_file()
        and ("tokenizer" in path.name or path.suffix == ".jinja"
             or path.name in {"vocab.json", "merges.txt", "special_tokens_map.json"})
    }
    return {"model": model_config, "tokenizer": tokenizer_files, "rollout_config": config}


class ResidentEngine:
    def __init__(self):
        self.engine = None
        self.signature = None
        self.actor_sha256 = None

    def prepare(self, actor: Path, identity: dict, config: dict, seed: int):
        signature = resident_signature(actor, config)
        if self.engine is None:
            from vllm import LLM
            self.engine = LLM(
                model=str(actor), tokenizer=str(actor), tensor_parallel_size=config["rollout_tp"],
                dtype="bfloat16", max_model_len=config["max_context_tokens"],
                gpu_memory_utilization=config["rollout_memory_fraction"],
                seed=seed, enable_prefix_caching=False, logprobs_mode="processed_logprobs",
                gdn_prefill_backend=config["rollout_gdn_prefill_backend"],
                worker_extension_cls="harness_r1_searchqa.workflow.Qwen35TextWorkerExtension",
                **({"language_model_only": True} if config.get("domain") == "alfworld" else {}),
            )
            self.signature = signature
        elif signature != self.signature:
            raise ValueError("Resident actor architecture/tokenizer/rollout configuration changed")
        elif identity["sha256"] != self.actor_sha256:
            replies = self.engine.collective_rpc(
                "reload_weights", timeout=config["rollout_timeout"],
                kwargs={"weights_path": str(actor), "is_checkpoint_format": True},
            )
            if len(replies) != config["rollout_tp"]:
                raise RuntimeError("Not all rollout workers acknowledged disk weight reload")
            if not self.engine.reset_prefix_cache():
                raise RuntimeError("Rollout cache invalidation failed after weight reload")
        self.actor_sha256 = identity["sha256"]
        return self.engine


def rollout_job(job_path: Path, resident: ResidentEngine | None = None):
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    register_text_engine()
    job = json.loads(job_path.read_text())
    config, rows = job["config"], job["rows"]
    actor = Path(job["actor"])
    identity = checkpoint_identity(actor)
    if identity["sha256"] != job["actor_sha256"]:
        raise ValueError("actor checkpoint changed before rollout engine reload")
    alfworld = job.get("domain", "searchqa") == "alfworld"
    if alfworld:
        from sao_bridge.alfworld_matched_grpo import validate_reward_rows, score_records
        validate_reward_rows(rows, job["reward_config"])
    else:
        from .reward import reward_func, _validate_metadata, _rows_by_id
        reward_args = SimpleNamespace(**job["reward_config"])
        train_ids = _rows_by_id(reward_args.harness_r1_searchqa_data)
        for row in rows:
            task_ids, *_ = _validate_metadata(reward_args, row["metadata"])
            if any(task not in train_ids for task in task_ids):
                raise ValueError("non-train task in reward metadata")
    tokenizer = AutoTokenizer.from_pretrained(str(actor), local_files_only=True)
    prompts = [thinking_prompt(tokenizer, row["prompt"]) for row in rows]
    if any(len(prompt) > config["max_prompt_tokens"] for prompt in prompts):
        raise ValueError("prompt exceeds upstream limit; refusing silent truncation")
    resident = resident if resident is not None else ResidentEngine()
    engine = resident.prepare(actor, identity, config, job["seed"])
    if job.get("verify_only"):
        atomic_json(Path(job["result"]), {"actor_sha256": identity["sha256"],
                                         "engine_reloaded_from_disk": True})
        return
    outputs = engine.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(n=8, temperature=0.7, top_p=0.95,
                       max_tokens=config["max_response_tokens"], logprobs=1,
                       seed=job["seed"], skip_special_tokens=False,
                       stop_token_ids=[tokenizer.eos_token_id]), use_tqdm=False)
    records = []
    if len(outputs) != 4:
        raise RuntimeError("rollout engine did not return four packets")
    for group, output in enumerate(outputs):
        if list(output.prompt_token_ids) != prompts[group] or len(output.outputs) != 8:
            raise RuntimeError("rollout engine changed prompt ids or sample count")
        for completion in output.outputs:
            if completion.finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"rollout failed: {completion.finish_reason}")
            tokens = list(completion.token_ids)
            if completion.logprobs is None or len(completion.logprobs) != len(tokens):
                raise RuntimeError("vLLM omitted generated token logprobs")
            logprobs = [float(lp[token].logprob)
                        for token, lp in zip(tokens, completion.logprobs, strict=True)]
            # Preserve reasoning markers, but terminal chat/EOS control tokens
            # are not part of the reward parser's patch envelope.
            text_tokens = tokens[:-1] if tokens and tokens[-1] == tokenizer.eos_token_id else tokens
            records.append({
                "group": group, "prompt_ids": prompts[group], "response_ids": tokens,
                "rollout_logprobs": logprobs,
                "response": tokenizer.decode(text_tokens, skip_special_tokens=False),
                "metadata": rows[group]["metadata"], "actor_sha256": identity["sha256"],
                "finish_reason": completion.finish_reason,
            })
    validate_rollouts(records, rows, identity["sha256"])
    if alfworld:
        results = asyncio.run(score_records(records, job["reward_config"], job["seed"]))
        scores = [result["score"] for result in results]
    else:
        results = asyncio.run(reward_func(
            reward_args, [SimpleNamespace(**record) for record in records]))
        scores = validate_rewards(results, len(records))
    for record, result, score in zip(records, results, scores, strict=True):
        record.update(reward=score, reward_result=result)
    atomic_json(Path(job["result"]), {"records": records, "actor_sha256": identity["sha256"],
                                     "engine_reloaded_from_disk": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rollout-job", type=Path)
    mode.add_argument("--resident-queue", type=Path)
    args = parser.parse_args()
    if args.rollout_job:
        rollout_job(args.rollout_job)
        return
    resident = ResidentEngine()
    queue = args.resident_queue
    cursor = 0
    while not (queue / "STOP").exists():
        request = queue / f"request-{cursor:06d}.json"
        if not request.is_file():
            time.sleep(0.1)
            continue
        envelope = json.loads(request.read_text())
        try:
            rollout_job(Path(envelope["job"]), resident)
        except (OSError, ValueError, RuntimeError, FloatingPointError) as exc:
            atomic_json(queue / f"failed-{cursor:06d}.json", {
                "error_type": type(exc).__name__, "error": str(exc),
                "reward_assigned": False,
            })
            raise
        atomic_json(queue / f"done-{cursor:06d}.json", {"cursor": cursor})
        cursor += 1


if __name__ == "__main__":
    main()
