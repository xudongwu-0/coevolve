"""Measure real generation throughput and exact-token transport before training."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import time

import httpx

from .common import write_json
from .launch import Launcher


async def probe(url: str, count: int):
    async with httpx.AsyncClient(timeout=180, trust_env=False) as client:
        async def call(seed):
            response = await client.post(url + "/chat/completions", json={
                "model": "policy", "messages": [{"role": "user", "content":
                    "Calculate the sum of integers from 1 through 100. Explain briefly."}],
                "max_tokens": 128, "temperature": 0.7, "top_p": 1.0, "top_k": -1,
                "return_token_ids": True, "logprobs": True, "top_logprobs": 0, "seed": seed,
            })
            response.raise_for_status()
            data = response.json()
            choice = data["choices"][0]
            ids, logps = choice["token_ids"], choice["logprobs"]["content"]
            if not data["prompt_token_ids"] or len(ids) != len(logps) or not ids:
                raise RuntimeError("Server token/logprob transport is inconsistent")
            return {"prompt_ids": data["prompt_token_ids"], "response_ids": ids,
                    "response_logprobs": [x["logprob"] for x in logps],
                    "usage": data["usage"], "message": choice["message"]}
        started = time.monotonic()
        rows = await asyncio.gather(*(call(42 + i) for i in range(count)))
        elapsed = time.monotonic() - started
        tokens = sum(row["usage"]["completion_tokens"] for row in rows)
        return {"count": count, "elapsed_seconds": elapsed,
                "generated_tokens": tokens, "tokens_per_second": tokens / elapsed,
                "exact_token_alignment": True, "records": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--gpu", default="4")
    parser.add_argument("--port", type=int, default=18751)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if args.run.exists():
        raise FileExistsError(args.run)
    launcher = Launcher(args.run)
    try:
        with launcher.vllm(str(args.model.resolve()), port=args.port,
                           gpus=args.gpu, name="vllm") as url:
            result = asyncio.run(probe(url, args.count))
            write_json(args.run / "calibration.json", result)
            print({k: v for k, v in result.items() if k != "records"}, flush=True)
    finally:
        launcher.close()


if __name__ == "__main__":
    main()
