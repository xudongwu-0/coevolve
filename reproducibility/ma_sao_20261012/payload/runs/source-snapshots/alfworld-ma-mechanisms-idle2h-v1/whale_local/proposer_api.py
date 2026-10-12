"""Local work queue allowing the active Codex conversation to be the proposer.

This is explicitly human/agent-in-the-loop: it does not expose Codex as an
unattended model service. The active agent reads request.json and writes a
reviewed response.json. The API never synthesizes canned candidate responses.
"""
from pathlib import Path
from typing import Literal
import json
import os
import re
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

JOBS = Path(os.environ.get("WHALE_PROPOSER_JOBS", str(Path(__file__).resolve().parents[1] / "audit/whale/proposer_jobs")))
app = FastAPI()


def cloud_queue():
    name = os.environ.get("WHALE_MODAL_QUEUE")
    if not name:
        return None
    import modal
    return modal.Dict.from_name(name, create_if_missing=False)


@app.get("/health")
def health():
    return {"ready": True, "jobs": str(JOBS), "mode": "active_agent_queue"}


class ProposalRequest(BaseModel):
    task_prompt: str
    run_dir: str
    iteration: int
    gate_context: dict = Field(default_factory=dict)
    accepted_harness_source: str = ""
    frontier: dict = Field(default_factory=dict)


@app.post("/jobs")
def submit(request: ProposalRequest):
    return create_job(request.model_dump())


class GateAnnotationRequest(BaseModel):
    kind: Literal["failure_annotation"]
    request_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: dict


@app.post("/gate/jobs")
def submit_gate(request: GateAnnotationRequest):
    return create_job(request.model_dump())


def create_job(payload):
    job_id = uuid.uuid4().hex
    directory = JOBS / job_id
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(json.dumps(payload, indent=2) + "\n")
    queue = cloud_queue()
    if queue is not None:
        queue[job_id] = {"status": "awaiting_active_agent", "request": payload}
    return {"job_id": job_id, "status": "awaiting_active_agent"}


@app.get("/jobs/{job_id}")
def poll(job_id: str):
    if not re.fullmatch(r"[0-9a-f]{32}", job_id):
        raise HTTPException(400, "Invalid job ID")
    directory = JOBS / job_id
    if not (directory / "request.json").exists():
        raise HTTPException(404, "Unknown job")
    result = directory / "response.json"
    queue = cloud_queue()
    if queue is not None and not result.exists():
        remote = queue.get(job_id)
        if remote and "response" in remote:
            temp = directory / "response.json.tmp"
            temp.write_text(json.dumps(remote["response"], indent=2) + "\n")
            temp.replace(result)
            queue[job_id] = remote | {"status": "consumed"}
    if not result.exists():
        return JSONResponse(status_code=202, content={"status": "awaiting_active_agent"})
    return json.loads(result.read_text())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("WHALE_PROPOSER_PORT", "18542")))
