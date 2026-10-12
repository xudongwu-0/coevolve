"""Stress the exact campaign retrieval client; never retry or suppress failures."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import httpx

from .searchqa_domain import _retrieval_url, retrieve_documents

QUERIES = ("Alan Turing", "Marie Curie", "University of Oxford", "Albert Einstein",
           "Nile River", "Mount Everest", "Apollo 11", "William Shakespeare")


def _digest(documents):
    return hashlib.sha256(json.dumps(documents, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


async def stress(count: int, *, top_k: int = 1, deadline: float = 30.0) -> dict:
    expected = {}
    for query in QUERIES:
        expected[query] = _digest(await retrieve_documents(query, top_k))
    url = _retrieval_url().removesuffix("/retrieve") + "/health"
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.get(url)
        response.raise_for_status()
        before = response.json()

    async def one(index):
        query = QUERIES[index % len(QUERIES)]
        began = time.monotonic()
        try:
            documents = await retrieve_documents(query, top_k)
            digest = _digest(documents)
            if digest != expected[query]:
                raise RuntimeError("Concurrent retrieval changed exact ranking/content/score")
            return {"ok": True, "elapsed_seconds": time.monotonic() - began}
        except Exception as exc:
            return {"ok": False, "elapsed_seconds": time.monotonic() - began,
                    "exception": repr(exc), "cause": repr(exc.__cause__), "query": query}

    began = time.monotonic()
    results = await asyncio.gather(*(one(index) for index in range(count)))
    elapsed = time.monotonic() - began
    latencies = sorted(result["elapsed_seconds"] for result in results)
    failures = [result for result in results if not result["ok"]]
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.get(url)
        response.raise_for_status()
        after = response.json()
    return {
        "host": socket.gethostname(), "count": count, "top_k": top_k,
        "client": "unmodified searchqa_domain.retrieve_documents; one HTTP client per call; no retry",
        "deadline_seconds": deadline, "wall_seconds": elapsed,
        "successful": count - len(failures), "failures": failures,
        "throughput_requests_per_second": count / elapsed,
        "latency_p50_seconds": latencies[len(latencies) // 2],
        "latency_p95_seconds": latencies[min(len(latencies) - 1, int(len(latencies) * .95))],
        "latency_max_seconds": max(latencies),
        "exact_ranking_digests": expected, "health_before": before, "health_after": after,
        "passed": not failures and elapsed < deadline and max(latencies) < deadline,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--deadline", type=float, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(not 1 <= count <= 1024 for count in args.counts):
        parser.error("counts must be 1..1024")
    reports = []
    for count in args.counts:
        report = asyncio.run(stress(count, top_k=args.top_k, deadline=args.deadline))
        reports.append(report)
        print(json.dumps({key: value for key, value in report.items()
                          if key not in {"health_before", "health_after", "exact_ranking_digests"}}, indent=2), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    args.output.write_text(json.dumps({"reports": reports,
                                    "service_url": os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL")}, indent=2) + "\n")
    args.output.chmod(0o644)
    if not all(report["passed"] for report in reports):
        raise SystemExit("Retrieval stress failed; campaign remains blocked")


if __name__ == "__main__":
    main()
