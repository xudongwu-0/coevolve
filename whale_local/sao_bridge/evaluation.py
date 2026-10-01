"""Local vLLM tool evaluation and frozen-counterpart reward service."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import time
from pathlib import Path

import httpx
from fastapi import FastAPI
from pydantic import BaseModel

from .common import read_json, write_json
from .domain_api import DOMAIN_MODULES, get_domain
from .retool import ContextExhausted


class Sampler:
    def __init__(self, url: str, concurrency=32, temperature=1.0, record_tokens=False,
                 chat_template_kwargs: dict | None = None, request_timeout: float = 900):
        self.urls = [u.rstrip("/") for u in url.split(",")]
        self.temperature = temperature
        self.record_tokens = record_tokens
        self.chat_template_kwargs = chat_template_kwargs
        self.semaphore = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(timeout=request_timeout, trust_env=False)
        self.generated_tokens = 0
        self.requests = 0
        self.context_clipped = 0
        self.context_exhausted = 0

    async def __call__(self, messages, tools, max_tokens, seed):
        body = dict(model="policy", messages=messages,
                    temperature=self.temperature, top_p=1.0, top_k=-1, seed=seed)
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if self.chat_template_kwargs is not None:
            body["chat_template_kwargs"] = self.chat_template_kwargs
        if tools:
            body["tools"] = tools
        if self.record_tokens:
            body.update(return_token_ids=True, logprobs=True, top_logprobs=0)
        async with self.semaphore:
            url = self.urls[seed % len(self.urls)]
            response = await self.client.post(url + "/chat/completions", json=body)
            if response.status_code == 400 and "maximum context length" in response.text:
                # Same rule as the training client: generation gets the remaining context.
                count = await self.client.post(url.removesuffix("/v1") + "/tokenize", json={
                    "model": "policy", "messages": messages, "tools": tools,
                    "add_generation_prompt": True})
                count.raise_for_status()
                room = count.json()["max_model_len"] - count.json()["count"]
                if room <= 0:
                    self.context_exhausted += 1
                    raise ContextExhausted("Prompt fills the model context")
                self.context_clipped += 1
                response = await self.client.post(
                    url + "/chat/completions",
                    json=dict(body, max_tokens=room if max_tokens is None else min(max_tokens, room)))
            response.raise_for_status()
            result = response.json()
        self.requests += 1
        self.generated_tokens += result["usage"]["completion_tokens"]
        if self.record_tokens:
            choice = result["choices"][0]
            ids, probs = choice["token_ids"], choice["logprobs"]["content"]
            if not result["prompt_token_ids"] or not ids or len(ids) != len(probs):
                raise RuntimeError("Exact policy token transport failed")
            result["record"] = {
                "prompt_ids": result["prompt_token_ids"], "response_ids": ids,
                "response_logprobs": [p["logprob"] for p in probs],
                "temperature": self.temperature,
                "stop_reason": "length" if choice.get("finish_reason") == "length" else "stop",
            }
        return result

    async def close(self):
        await self.client.aclose()


def stable_id_seed_offset(sample_id: str) -> int:
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("Evaluation row ID must be a nonempty string")
    try:
        value = int(sample_id[:8], 16)
    except ValueError:
        value = int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
    return value % 1000000


async def evaluate(
    sampler,
    rows,
    source,
    *,
    tokens=4096,
    samples=1,
    seed=42,
    domain="math",
    episode_concurrency: int | None = None,
):
    if not rows or samples < 1:
        raise ValueError("Evaluation requires nonempty questions and samples")
    if episode_concurrency is not None and (
        type(episode_concurrency) is not int or episode_concurrency < 1
    ):
        raise ValueError("Episode concurrency must be a positive integer")
    task = get_domain(domain)
    spec = task.harness_spec(source)
    started = time.monotonic()
    task_specs = [
        (
            row,
            sample,
            seed + stable_id_seed_offset(row["id"]) + 1009 * sample,
        )
        for row in rows for sample in range(samples)
    ]
    semaphore = asyncio.Semaphore(episode_concurrency or len(task_specs))

    async def run_episode(row, rollout_seed):
        async with semaphore:
            return await task.run_episode(
                row,
                spec,
                sampler,
                total_tokens=tokens,
                seed=rollout_seed,
            )

    results = await asyncio.gather(*(
        run_episode(row, rollout_seed)
        for row, _, rollout_seed in task_specs
    ))
    for result, (_, sample, rollout_seed) in zip(results, task_specs):
        result["sample_index"] = sample
        result["rollout_seed"] = rollout_seed
    result = {
        "accuracy": sum(r["reward"] for r in results) / len(results),
        "question_count": len(rows), "samples_per_question": samples,
        "metric": f"mean@{samples}", "tokens": sum(r["tokens"] for r in results),
        "tool_calls": sum(r["tool_calls"] for r in results),
        "tool_errors": sum(r["tool_errors"] for r in results),
        "truncated_fraction": sum(r["truncated"] for r in results) / len(results),
        "boxed_fraction": sum(r["boxed"] for r in results) / len(results),
        "elapsed_seconds": time.monotonic() - started, "results": results,
    }
    if domain != "math":
        result.update(
            domain=domain, formatted_fraction=result["boxed_fraction"],
            harness_error_episodes=sum(bool(row.get("harness_errors") or row.get("hook_errors"))
                                       for row in results),
        )
    return result


class SourceRequest(BaseModel):
    source: str
    sample_id: str
    reference_source: str | None = None


def create_app(sampler, rows, incumbent, output: Path, tokens=4096, *, domain="math"):
    """Paired, uncached reward evaluations: exactly 2*len(rows) samples per action."""
    import itertools
    if output.exists() and any(output.glob("*.json")):
        raise FileExistsError("Reward service output must be a fresh directory")
    app = FastAPI()
    requests = itertools.count()
    task = get_domain(domain)

    @app.get("/health")
    def health():
        return {"ready": True, "questions": len(rows), "backend": "frozen local vLLM",
                "sampling": "paired_uncached"}

    @app.post("/score")
    async def score(request: SourceRequest):
        invalid_error = None
        try:
            source = task.harness_spec(request.source)["source"]
            reference = task.harness_spec(request.reference_source)["source"] if request.reference_source else incumbent
        except (ValueError, SyntaxError, TypeError) as exc:
            # Preserve the physical sample budget even for invalid editor actions.
            source = reference = incumbent
            invalid_error = str(exc)
        index = next(requests)
        # Candidate/reference use common random numbers. Seed assignment is bound to
        # the deterministic action ID, never concurrent request arrival order.
        pair_seed = 42 + int(hashlib.sha256(request.sample_id.encode()).hexdigest()[:12], 16) % 1_000_000_000
        baseline, candidate = await asyncio.gather(
            evaluate(sampler, rows, reference, tokens=tokens, seed=pair_seed, domain=domain),
            evaluate(sampler, rows, source, tokens=tokens, seed=pair_seed, domain=domain),
        )
        if candidate.get("harness_error_episodes", 0):
            invalid_error = (
                f"Executable harness failed in {candidate['harness_error_episodes']} episodes"
            )
        prefix = f"{index:06d}"
        for role, harness, result in (("reference", reference, baseline),
                                      ("candidate", source, candidate)):
            result["harness_source"] = harness
            result["pair_role"] = role
            result["request_index"] = index
            result["sample_id"] = request.sample_id
            name = hashlib.sha256(harness.encode()).hexdigest()
            await asyncio.to_thread(write_json, output / f"{prefix}-{role}-{name}.json", result)
        return {
            "valid": invalid_error is None,
            "reward": candidate["accuracy"] - baseline["accuracy"] if invalid_error is None else -1.0,
            "accuracy": candidate["accuracy"], "baseline": baseline["accuracy"],
            "source": source, "reference_source": reference, "request_index": index,
            "sample_id": request.sample_id,
            "error": invalid_error,
            "physical_generated_tokens_cumulative": sampler.generated_tokens,
        }
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--harness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--port", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=256)
    parser.add_argument("--record-tokens", action="store_true")
    parser.add_argument("--domain", choices=tuple(DOMAIN_MODULES), default="math")
    args = parser.parse_args()
    rows = read_json(args.data)[args.offset:]
    if args.limit:
        rows = rows[:args.limit]
    source = get_domain(args.domain).harness_spec(args.harness.read_text())["source"]
    if args.port:
        import uvicorn
        sampler = Sampler(args.url, concurrency=args.concurrency, record_tokens=args.record_tokens)
        app = create_app(sampler, rows, source, args.output, args.tokens, domain=args.domain)
        uvicorn.run(app, host="127.0.0.1", port=args.port)
    else:
        async def run():
            sampler = Sampler(args.url)
            try:
                return await evaluate(sampler, rows, source, tokens=args.tokens,
                                      samples=args.samples, domain=args.domain)
            finally:
                await sampler.close()
        write_json(args.output, asyncio.run(run()))


if __name__ == "__main__":
    main()
