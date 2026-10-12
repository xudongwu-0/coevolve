"""Loopback-only frozen local solver; never proxies the interactive assistant."""
from __future__ import annotations

import argparse
from pathlib import Path
import threading

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    INITIAL_SOURCE, RETRY_MESSAGE, binary_reward, harness_spec, read_json,
    solver_messages, write_json,
)


class SourceRequest(BaseModel):
    source: str


class FrozenSolver:
    def __init__(self, model_path: str, total_tokens: int, seed: int):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda:0").eval()
        self.total_tokens = total_tokens
        self.seed = seed

    def solve(self, spec: dict, row: dict) -> dict:
        messages = solver_messages(spec, row["question"])
        remaining = self.total_tokens
        torch.manual_seed(self.seed + int(row["id"][:8], 16))
        for _ in range(spec["MAX_TURNS"]):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self.tokenizer(prompt, return_tensors="pt",
                                    add_special_tokens=False).to("cuda:0")
            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=remaining, do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            tokens = outputs[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            remaining -= len(tokens)
            messages.append({"role": "assistant", "content": text})
            if "\\boxed{" in text or remaining <= 0:
                break
            messages.append({"role": "user", "content": RETRY_MESSAGE})
        return {"id": row["id"], "reward": binary_reward(text, row["answer"]),
                "tokens": self.total_tokens - remaining, "messages": messages}

    def evaluate(self, source: str, rows: list[dict]) -> dict:
        spec = harness_spec(source)
        results = [self.solve(spec, row) for row in rows]
        return {"accuracy": sum(r["reward"] for r in results) / len(results),
                "results": results}


def create_app(solver: FrozenSolver, rows: list[dict], incumbent: str) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()
    cache = {}

    @app.get("/health")
    def health():
        return {"ready": True, "backend": "local frozen HuggingFace solver",
                "data_count": len(rows), "self_chat_api": False}

    @app.post("/score")
    def score(request: SourceRequest):
        try:
            source = harness_spec(request.source)["source"]
        except (AssertionError, SyntaxError, ValueError, TypeError) as exc:
            # Invalid generated actions are an explicit environment outcome.
            return {"reward": -1.0, "valid": False, "error": str(exc)}
        with lock:
            if incumbent not in cache:
                cache[incumbent] = solver.evaluate(incumbent, rows)
            if source not in cache:
                cache[source] = solver.evaluate(source, rows)
            baseline, candidate = cache[incumbent], cache[source]
        return {"valid": True, "reward": candidate["accuracy"] - baseline["accuracy"],
                "accuracy": candidate["accuracy"], "baseline": baseline["accuracy"],
                "source": source}

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--harness", type=Path)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--port", type=int, default=18641)
    parser.add_argument("--evaluate-output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.tokens <= 4096:
        parser.error("--tokens must be between 1 and 4096")
    rows = read_json(args.data)
    if not rows:
        parser.error("Evaluation dataset is empty")
    source = harness_spec(
        args.harness.read_text() if args.harness else INITIAL_SOURCE
    )["source"]
    solver = FrozenSolver(args.model, args.tokens, args.seed)
    if args.evaluate_output:
        write_json(args.evaluate_output, solver.evaluate(source, rows))
        return
    import uvicorn
    uvicorn.run(create_app(solver, rows, source), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
