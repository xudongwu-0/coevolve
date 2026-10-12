"""Sample fresh harnesses from the trained editor for development-only selection."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import EDITOR_SYSTEM, INITIAL_SOURCE, harness_spec, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()
    if not 1 <= args.count <= 32 or not 0 <= args.temperature <= 2:
        parser.error("Use 1-32 candidates and temperature between 0 and 2")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()
    messages = [
        {"role": "system", "content": EDITOR_SYSTEM},
        {"role": "user", "content": (
            f"Current harness:\n{INITIAL_SOURCE}\n"
            "Improve only the three literal settings shown above. Keep the exact "
            "import and CandidateEnv class name. Return no methods or explanation."
        )},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    records = []
    for seed in range(1042, 1042 + args.count):
        torch.manual_seed(seed)
        with torch.inference_mode():
            result = model.generate(
                **inputs, max_new_tokens=512, do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else 1.0,
                top_p=1.0, top_k=0,
                pad_token_id=tokenizer.eos_token_id,
            )
        text = tokenizer.decode(result[0, inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
        record = {"seed": seed, "temperature": args.temperature, "text": text}
        try:
            record.update(valid=True, source=harness_spec(text)["source"])
        except (AssertionError, SyntaxError, TypeError, ValueError) as exc:
            record.update(valid=False, error=str(exc))
        records.append(record)
    write_json(args.output, records)


if __name__ == "__main__":
    main()
