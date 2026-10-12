"""Bounded Modal reproduction of the audited local two-round gate integration.

The active agent answers the SAME FastAPI proposal queue, transported through a
private Modal Dict. Model/checkpoints persist in task-specific Modal Volumes.
No token files, existing model assets, or unrelated workspace paths are uploaded.
"""
from pathlib import Path
import json
import os
import modal

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "fbe125eb7abea7f760c99ab9acc1a6261e708fc6"
REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
APP_NAME = "whale-failure-gate-integration"
CACHE_NAME = "whale-gate-model-cache"
OUTPUT_NAME = "whale-gate-results"
QUEUE_NAME = "whale-gate-active-agent-jobs"
app = modal.App(APP_NAME)
cache = modal.Volume.from_name(CACHE_NAME, create_if_missing=True)
outputs = modal.Volume.from_name(OUTPUT_NAME, create_if_missing=True)
queue = modal.Dict.from_name(QUEUE_NAME, create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install(requirements=[str(ROOT / "whale_local/requirements-modal.lock")])
    .run_commands("git clone --no-checkout https://github.com/krafton-ai/WHALE.git /opt/rsi/WHALE",
                  "git -C /opt/rsi/WHALE checkout " + UPSTREAM)
    .env({"PYTHONPATH": "/opt/rsi/whale_local:/opt/rsi/WHALE/domains/chess_puzzles",
          "HF_HOME": "/cache/huggingface", "TOKENIZERS_PARALLELISM": "false",
          "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "8", "WANDB_MODE": "disabled"})
    .add_local_dir(ROOT / "whale_local", "/opt/rsi/whale_local",
                   ignore=["**/__pycache__/**", "*.pyc"])
    .add_local_dir(ROOT / "runs/whale_setup/data" if modal.is_local() else Path("/opt/rsi/input_data"), "/opt/rsi/input_data",
                   ignore=["source_*.json"]))


@app.function(image=image, cpu=4, memory=16384, timeout=1200,
              volumes={"/cache": cache}, max_containers=1, retries=0)
def prepare_model():
    from huggingface_hub import snapshot_download
    import hashlib
    import subprocess
    import sys
    path = snapshot_download("Qwen/Qwen3.5-2B", revision=REVISION,
        allow_patterns=["*.json", "*.jinja", "*.safetensors", "*.txt"])
    # Fail import compatibility on CPU before allocating the cloud GPU.
    env = dict(os.environ, WHALE_RUN_ROOT="/tmp/whale-import", WHALE_MODEL_ROOT=path)
    subprocess.run([sys.executable, "-c", "import worker; from meta_harness import meta_harness_chess_puzzle"],
                   cwd="/opt/rsi", env=env, check=True)
    files = {}
    for p in Path(path).glob("*.safetensors"):
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        files[p.name] = {"bytes": p.stat().st_size, "sha256": h.hexdigest()}
    assert files["model.safetensors"]["sha256"] == "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1", "Cloud base differs from verified local base"
    result = {"path": path, "revision": REVISION, "weights": files, "imports_passed": True}
    Path("/cache/model_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    cache.commit()
    return result


@app.function(image=image, gpu="L40S", cpu=8, memory=65536, timeout=3600,
              startup_timeout=600, max_containers=1, retries=0, scaledown_window=2,
              volumes={"/cache": cache, "/results": outputs})
def run_gate(run_name: str, rounds: int = 2):
    import re
    import signal
    import subprocess
    import sys
    import time
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", run_name) or rounds not in (1, 2):
        raise ValueError("This cloud integration entry point is bounded to 1–2 rounds")
    run = Path("/results") / run_name
    if run.exists():
        raise FileExistsError("Choose a fresh cloud run name; committed evidence is never overwritten")
    model = json.loads(Path("/cache/model_manifest.json").read_text())
    data = json.loads(Path("/opt/rsi/input_data/manifest.json").read_text())
    for value in data["splits"].values():
        value["path"] = str(Path("/opt/rsi/input_data") / Path(value["path"]).name)
    Path("/tmp/whale_data_manifest.json").write_text(json.dumps(data))
    env = dict(os.environ, WHALE_MODAL_QUEUE=QUEUE_NAME)
    command = [sys.executable, "-u", "/opt/rsi/whale_local/pipeline.py", "--run-dir", str(run),
               "--gpu", "0", "--rounds", str(rounds), "--rollouts-per-puzzle", "4",
               "--gate", "--gate-probe-budget", "16", "--gate-resamples", "4",
               "--learning-rate", "1e-6",
               "--base-model", model["path"], "--data-manifest", "/tmp/whale_data_manifest.json"]
    started = time.monotonic()
    process = None
    try:
        process = subprocess.Popen(command, cwd="/opt/rsi", env=env, start_new_session=True)
        code = process.wait(timeout=3300)
        if code:
            raise subprocess.CalledProcessError(code, command)
        subprocess.run([sys.executable, "/opt/rsi/whale_local/verify_run.py", str(run)],
                       cwd="/opt/rsi", env=env, check=True, timeout=120)
        result = {"summary": json.loads((run / "summary.json").read_text()),
                  "verification": json.loads((run / "verification.json").read_text()),
                  "cloud_elapsed_seconds": time.monotonic() - started,
                  "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total",
                                                  "--format=csv,noheader"], text=True).strip()}
        (run / "modal_result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        # Also terminate the worker/proposer descendants if the driver times out.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        outputs.commit()


@app.local_entrypoint()
def main(run_name: str = "gate_20260914", rounds: int = 2):
    queue.hydrate()
    print("Preparing fixed-revision model and checking portable imports on CPU.")
    print(json.dumps(prepare_model.remote(), indent=2))
    print("Starting one L40S, maximum one hour, no automatic retries; active proposer queue:", QUEUE_NAME)
    result = run_gate.remote(run_name, rounds)
    directory = ROOT / "runs" / ("modal_" + run_name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "modal_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
