"""Prepare token-bounded packets only from a completed provenance-bound baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from sao_bridge.common import file_sha, write_json
from .data import build_rl_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--engineer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-seconds", type=int, default=900)
    parser.add_argument("--maximum-packets", type=int, default=256)
    parser.add_argument("--packet-size", type=int, choices=(4, 8), default=8)
    parser.add_argument("--engineer-think-prefill", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + args.wait_seconds
    while not args.report.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Completed baseline not available: {args.report}")
        time.sleep(3)
    if args.output.exists():
        raise FileExistsError("Packet output must be fresh")
    report = json.loads(args.report.read_text())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.engineer, local_files_only=True)
    if args.engineer_think_prefill:
        from .sft_train import configure_engineer_template
        configure_engineer_template(tokenizer, True)

    def count(messages: list[dict]) -> int:
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=True,
        )
        return len(tokens)

    rows = build_rl_rows(
        report, incumbent_patch=report["incumbent_patch"],
        target_model_sha256=report["target_model_sha256"],
        data_manifest=args.data_root / "manifest.json",
        packet_size=args.packet_size, maximum_packets=args.maximum_packets,
        prompt_token_limit=20479, prompt_token_counter=count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".staging")
    with temporary.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    write_json(args.output.with_suffix(".verified.json"), {
        "status": "verified_train_packets",
        "report_sha256": file_sha(args.report),
        "dataset_sha256": file_sha(args.output), "packets": len(rows),
        "baseline_config": report["baseline_config"],
        "sft_prompt_limit": 20479, "reserved_response_tokens": 12288,
        "source": "real_completed_train_episodes",
        "tasks_per_packet": args.packet_size,
    })


if __name__ == "__main__":
    main()
