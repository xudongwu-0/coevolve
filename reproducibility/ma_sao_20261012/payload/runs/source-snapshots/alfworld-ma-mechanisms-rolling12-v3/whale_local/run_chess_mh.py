"""Run native WHALE MH, replacing only the missing proposer transport."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import re
import time
from urllib.request import Request, build_opener, ProxyHandler
from meta_harness import meta_harness_chess_puzzle as mh

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:" + os.environ.get("WHALE_PROPOSER_PORT", "18542")
opener = build_opener(ProxyHandler({}))


def propose_local(*, task_prompt, iteration, run_dir, timeout_seconds, **kwargs):
    context = {}
    if os.environ.get("WHALE_GATE_CONTEXT"):
        context = json.loads(Path(os.environ["WHALE_GATE_CONTEXT"]).read_text())
        assert context["failures"], "Gate did not authorize harness search"
        probe_path = Path(context["probe_harness"])
        context["probe_source"] = probe_path.read_text()
    accepted = (run_dir / "logs/accepted_harness.txt").read_text().strip()
    payload = {"task_prompt": task_prompt, "run_dir": str(run_dir), "iteration": iteration,
               "gate_context": context,
               "accepted_harness_source": (run_dir / "harnesses" / accepted / "harness.py").read_text(),
               "frontier": json.loads((run_dir / "logs/frontier_val.json").read_text())}
    request = Request(BASE_URL + "/jobs", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with opener.open(request, timeout=10) as response:
        job = json.load(response)
    print(f"Local proposer job: {job['job_id']}; waiting for active Codex agent", flush=True)
    deadline = time.monotonic() + timeout_seconds
    result = None
    while time.monotonic() < deadline:
        with opener.open(BASE_URL + "/jobs/" + job["job_id"], timeout=10) as response:
            if response.status == 200:
                result = json.load(response)
                break
        time.sleep(1)
    if result is None:
        raise TimeoutError("Active-agent proposer did not answer the local API queue")
    expected = re.findall(r"harnesses/(h\d+)/harness\.py", task_prompt)
    candidates = result["candidates"]
    assert [c["name"] for c in candidates] == expected
    metadata = []
    for candidate in candidates:
        directory = run_dir / "harnesses" / candidate["name"]
        directory.mkdir(exist_ok=True)
        (directory / "harness.py").write_text(candidate["source"])
        metadata.append({k: v for k, v in candidate.items() if k != "source"})
    (run_dir / "pending_eval.json").write_text(json.dumps({"candidates": metadata}, indent=2) + "\n")
    return SimpleNamespace(exit_code=0, text="Candidate supplied by active Codex agent through local FastAPI", stderr="", show=lambda: print("Proposer: active Codex agent via FastAPI", flush=True))


mh.RUNS_DIR = Path(os.environ.get("WHALE_MH_RUNS", str(ROOT / "audit/whale/mh_runs")))
mh.propose_claude = propose_local
if __name__ == "__main__":
    mh.main()
