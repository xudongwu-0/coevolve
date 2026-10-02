"""Nonblocking, run-local forwarding for the official AgentRL worker protocol."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import itertools
import logging
import time
from urllib.parse import urlsplit

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
import httpx
import uvicorn

logger = logging.getLogger(__name__)


def worker_api(address: str, endpoint: str) -> str:
    parsed = urlsplit(address)
    if (
        parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username or parsed.password or parsed.query or parsed.fragment
        or parsed.path.rstrip("/") not in {"", "/api"}
    ):
        raise HTTPException(400, "Worker address must be a loopback HTTP API")
    base = address.rstrip("/")
    return f"{base}/{endpoint}" if base.endswith("/api") else f"{base}/api/{endpoint}"


def create_app(*, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    workers: dict[str, dict[str, dict]] = {}
    sessions: dict[int, str] = {}
    cancellations: dict[int, asyncio.Event] = {}
    counter = itertools.count(1)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(
            timeout=300, trust_env=False, transport=transport,
            limits=httpx.Limits(max_connections=512, max_keepalive_connections=256),
        ) as client:
            app.state.worker_client = client
            yield

    app = FastAPI(lifespan=lifespan)

    async def forward(address: str, endpoint: str, *, method: str = "POST",
                      payload: dict | None = None, timeout: float = 300) -> httpx.Response:
        url = worker_api(address, endpoint)
        try:
            return await app.state.worker_client.request(
                method, url, json=payload, timeout=timeout,
            )
        except httpx.RequestError as exc:
            logger.error("ALFWorld worker transport failed at %s: %r", url, exc)
            raise HTTPException(502, f"Worker transport failed at {url}: {exc!r}") from exc

    def alive(name: str) -> list[dict]:
        return [
            info for info in workers.get(name, {}).values()
            if time.time() - info["last_seen"] < 60
        ]

    def response_body(response: httpx.Response) -> dict:
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.text[:2000])
        try:
            body = response.json()
        except ValueError as exc:
            raise HTTPException(502, "Worker response is not JSON") from exc
        if not isinstance(body, dict):
            raise HTTPException(502, "Worker response must be a JSON object")
        return body

    @app.post("/api/receive_heartbeat")
    async def receive_heartbeat(payload: dict) -> dict:
        name, address = payload.get("name"), payload.get("address")
        if not isinstance(name, str) or not name or not isinstance(address, str):
            raise HTTPException(400, "Heartbeat requires name and address")
        worker_api(address, "worker_status")
        capacity = payload.get("concurrency")
        if type(capacity) is not int or capacity < 1 or not isinstance(payload.get("indices"), list):
            raise HTTPException(400, "Heartbeat requires capacity and task indices")
        body = response_body(await forward(address, "worker_status", method="GET", timeout=10))
        current = body.get("current")
        if type(current) is not int or current < 0:
            raise HTTPException(502, "Worker status omitted a nonnegative current count")
        workers.setdefault(name, {})[address] = {
            "address": address, "capacity": capacity, "current": current,
            "indices": payload["indices"], "status": "ALIVE", "last_seen": time.time(),
        }
        return {"ok": True}

    @app.get("/api/list_workers")
    async def list_workers() -> dict:
        return {
            name: {"workers": {
                address: {
                    **info, "status": "ALIVE" if time.time() - info["last_seen"] < 60 else "DEAD",
                } for address, info in task_workers.items()
            }} for name, task_workers in workers.items()
        }

    @app.get("/api/get_indices")
    async def get_indices(name: str) -> list:
        candidates = alive(name)
        if not candidates:
            raise HTTPException(404, f"No alive worker for task {name}")
        response = await forward(candidates[0]["address"], "get_indices", method="GET", timeout=30)
        if response.status_code != 200:
            raise HTTPException(response.status_code, response.text[:2000])
        values = response.json()
        if not isinstance(values, list):
            raise HTTPException(502, "Worker indices must be a JSON list")
        return values

    @app.post("/api/start_sample")
    async def start_sample(payload: dict) -> JSONResponse:
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise HTTPException(400, "start_sample requires name")
        candidates = alive(name)
        if not candidates:
            raise HTTPException(404, f"No alive worker for task {name}")
        sid, address = next(counter), candidates[0]["address"]
        sessions[sid] = address
        try:
            response = await forward(address, "start_sample", payload={
                "index": payload.get("index"), "custom_task": payload.get("custom_task"),
                "session_id": sid,
            })
            body = response_body(response)
        except BaseException:
            sessions.pop(sid, None)
            raise
        return JSONResponse(body, headers={"session_id": str(sid)})

    @app.post("/api/interact")
    async def interact(
        request: Request, session_id: str | None = Header(default=None, convert_underscores=False),
    ) -> JSONResponse:
        payload = await request.json()
        sid = session_id or payload.get("session_id")
        if sid is None:
            raise HTTPException(400, "interact requires session_id")
        try:
            sid = int(sid)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "Invalid session_id") from exc
        address = sessions.get(sid)
        if address is None:
            raise HTTPException(404, f"Unknown session {sid}")
        body = response_body(await forward(address, "interact", payload={
            "session_id": sid, "messages": payload.get("messages") or [],
        }))
        if isinstance(body.get("env_out"), dict):
            body = body["env_out"]
        return JSONResponse(body, headers={"session_id": str(sid)})

    @app.post("/api/cancel")
    async def cancel(
        request: Request, session_id: str | None = Header(default=None, convert_underscores=False),
    ) -> dict:
        payload = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
        sid = session_id or payload.get("session_id")
        if sid is None:
            return {"ok": True}
        try:
            sid = int(sid)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, "Invalid session_id") from exc
        address = sessions.get(sid)
        if address is not None:
            finished = cancellations.setdefault(sid, asyncio.Event())
            response_body(await forward(
                address, "cancel_with_notice", payload={"session_id": sid}, timeout=30,
            ))
            try:
                await asyncio.wait_for(finished.wait(), timeout=300)
            except TimeoutError as exc:
                logger.error("ALFWorld worker did not acknowledge cancellation of session %s", sid)
                raise HTTPException(504, f"Worker cancellation not completed for session {sid}") from exc
        return {"session_id": sid}

    @app.post("/api/cancel_notice")
    async def cancel_notice(payload: dict) -> dict:
        sid = payload.get("session_id")
        if sid is not None:
            try:
                session = int(sid)
                sessions.pop(session, None)
                finished = cancellations.pop(session, None)
                if finished is not None:
                    finished.set()
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "Invalid session_id") from exc
        return {"ok": True}

    @app.post("/api/calculate_overall")
    async def calculate_overall(payload: dict) -> JSONResponse:
        candidates = alive(str(payload.get("name") or ""))
        if not candidates:
            raise HTTPException(404, "No alive worker for overall calculation")
        return JSONResponse(response_body(await forward(
            candidates[0]["address"], "calculate_overall", payload=payload,
        )))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("The local controller must bind to loopback")
    uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
