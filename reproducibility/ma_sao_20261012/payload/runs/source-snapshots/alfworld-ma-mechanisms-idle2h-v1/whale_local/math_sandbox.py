"""SandboxFusion-compatible Python execution in isolated local namespaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile
import time
import uuid

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv-whale-portable"
# A venv may contain a copied executable rather than a symlink. Its binary
# alone does not identify the base standard-library installation.
PYTHON = Path(sys._base_executable).resolve()
PYTHON_ROOT = Path(sys.base_prefix).resolve()
OUTPUT_LIMIT = 1024 * 1024
app = FastAPI()
LOG_ROOT = None


def run_python(payload):
    if payload.get("language", "python") not in {"python", "python3"}:
        raise HTTPException(400, "Only Python execution is configured")
    if payload.get("files") or payload.get("fetch_files"):
        raise HTTPException(400, "File transfer is not configured")
    timeout = min(60, max(1, int(payload.get("run_timeout", 30))))
    memory_mb = min(4096, max(128, int(payload.get("memory_limit_MB", 1024))))
    code = str(payload.get("code", ""))
    command = ["/usr/bin/bwrap", "--unshare-all", "--die-with-parent", "--new-session"]
    for directory in [Path("/usr"), Path("/lib"), Path("/lib64"), PYTHON_ROOT, VENV]:
        command += ["--ro-bind", str(directory), str(directory)]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/work"]
    env = {"PATH": "/usr/bin:/bin", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024,) * 2)
        resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT,) * 2)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="whale-math-code-") as temporary:
        work = Path(temporary)
        script = work / "main.py"
        script.write_text(code)
        packages = VENV / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        bootstrap = "import sys,runpy;sys.path.insert(0," + repr(str(packages)) + ");runpy.run_path('/work/main.py',run_name='__main__')"
        command += ["--ro-bind", str(script), "/work/main.py", "--chdir", "/work",
                    "--", str(PYTHON.resolve()), "-I", "-c", bootstrap]
        timed_out = False
        with (work / "stdout").open("wb") as stdout, (work / "stderr").open("wb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr,
                                     env=env, preexec_fn=limits, start_new_session=True)
            try:
                child.communicate(str(payload.get("stdin") or "").encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(child.pid, signal.SIGKILL)
                child.communicate()
        output = (work / "stdout").read_bytes()[:OUTPUT_LIMIT].decode("utf-8", errors="replace")
        error = (work / "stderr").read_bytes()[:OUTPUT_LIMIT].decode("utf-8", errors="replace")
    duration = time.monotonic() - start
    result = {"status": "Success" if child.returncode == 0 and not timed_out else "Failed",
        "compile_result": {"status": "Finished", "stdout": "", "stderr": "", "return_code": 0},
        "run_result": {"status": "TimeLimitExceeded" if timed_out else "Finished",
                       "stdout": output, "stderr": error, "return_code": child.returncode,
                       "execution_time": duration}}
    if LOG_ROOT is not None:
        record = {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "code": code,
                  "timeout_seconds": timeout, "memory_limit_mb": memory_mb,
                  "isolated_filesystem_and_network": True, "result": result}
        (LOG_ROOT / (uuid.uuid4().hex + ".json")).write_text(json.dumps(record, indent=2) + "\n")
    return result


@app.get("/health")
def health():
    return {"ready": True, "sandbox": "bubblewrap; isolated filesystem/network; CPU Python"}


@app.post("/run_code")
def execute(payload: dict):
    return run_python(payload)


if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    args = parser.parse_args()
    LOG_ROOT = args.log_dir.resolve()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
