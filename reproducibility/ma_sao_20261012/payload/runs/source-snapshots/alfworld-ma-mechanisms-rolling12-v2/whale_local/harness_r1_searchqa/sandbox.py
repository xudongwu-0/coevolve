"""Process-contained official hooks; infrastructure failures never become rewards.

No project/home/GPU/network mounts. CPU, address space, wall time, transport,
and JSON notebook limits apply even during compilation of helper defaults.
"""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from typing import Any

from sao_bridge import program_harness as isolation


SandboxInfrastructureError = isolation.SandboxInfrastructureError
MAX_NOTEBOOK_BYTES = 16384
MAX_REQUEST_BYTES = 256000
MAX_OUTPUT_BYTES = 70000
_READY = b"HARNESS_R1_READY\n"

_WORKER = r'''
import json, sys, signal

def checked(value, limit):
    pending = [(value, 0)]
    count = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        if count > 10000 or depth > 32:
            raise ValueError("JSON depth/value limit exceeded")
        if type(item) in (dict, list):
            if count + len(pending) + len(item) > 10000:
                raise ValueError("JSON value limit exceeded")
            if type(item) is dict and any(type(k) is not str for k in item):
                raise ValueError("JSON keys must be strings")
            pending.extend((x, depth + 1) for x in
                           (item.values() if type(item) is dict else item))
        elif type(item) not in (str, int, float, bool, type(None)):
            raise ValueError("Notebook/effect must contain only JSON values")
    text = json.dumps(value, allow_nan=False, ensure_ascii=True, separators=(",", ":"))
    if len(text.encode()) > limit:
        raise ValueError("JSON byte limit exceeded")
    return text

request = json.loads(sys.stdin.buffer.read(256001))
runner = {"__name__": "official_runner"}
exec(compile(request["runner"], "<official_runner>", "exec"), runner)
sys.stdout.write("HARNESS_R1_READY\n")
sys.stdout.flush()

class BudgetExceeded(BaseException):
    pass

lines = 0
def trace(frame, event, arg):
    global lines
    if event == "line":
        lines += 1
        if lines > 2000:
            raise BudgetExceeded("hook line budget exceeded")
    return trace

def timeout(signum, frame):
    raise BudgetExceeded("hook wall budget exceeded")

try:
    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, 0.2)
    fn = runner["compile_hook"](request["source"], benchmark="searchqa")
    if request["mode"] == "validate":
        reply = {"status": "ok", "effect": None, "notebook": {}}
    else:
        ctx = runner["_to_attrdict"](request["ctx"])
        nb = request["notebook"]
        sys.settrace(trace)
        try:
            raw = fn(ctx, nb)
        finally:
            sys.settrace(None)
        effect = runner["_normalize_hook_result"](request["hook"], raw, ctx)
        if raw is not None and raw != {} and effect is None:
            raise ValueError("invalid hook effect")
        checked(nb, 16384)
        checked(effect, 32000)
        reply = {"status": "ok", "effect": effect, "notebook": nb}
except BaseException as exc:
    sys.settrace(None)
    reply = {"status": "candidate_error", "message": type(exc).__name__ + ": " + str(exc)[:500]}
finally:
    signal.setitimer(signal.ITIMER_REAL, 0)
sys.stdout.write(checked(reply, 65536) + "\n")
'''


def _json(value: Any, limit: int) -> str:
    return isolation._json_text(value, limit)


def _transport(request: bytes) -> bytes:
    command = isolation._sandbox_command()
    command[-1] = _WORKER
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, close_fds=True, start_new_session=True)
    except OSError as exc:
        raise SandboxInfrastructureError(f"Cannot start hook sandbox: {exc}") from exc
    output = bytearray()
    diagnostic = bytearray()
    try:
        # Input is bounded, but may exceed pipe capacity. Multiplex all pipes.
        selector = selectors.DefaultSelector()
        for stream in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "input")
        selector.register(process.stdout, selectors.EVENT_READ, "output")
        selector.register(process.stderr, selectors.EVENT_READ, "error")
        position = 0
        deadline = time.monotonic() + isolation.WALL_SECONDS
        with selector:
            while selector.get_map():
                if time.monotonic() >= deadline:
                    if output.startswith(_READY):
                        return _READY + b'{"status":"candidate_error","message":"hook sandbox wall timeout"}'
                    raise SandboxInfrastructureError("Hook sandbox timed out before readiness")
                for key, _ in selector.select(0.1):
                    if key.data == "input":
                        try:
                            position += os.write(key.fd, request[position:position + 8192])
                        except BrokenPipeError:
                            position = len(request)
                        if position == len(request):
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                    else:
                        chunk = os.read(key.fd, 8192)
                        if not chunk:
                            selector.unregister(key.fileobj)
                        else:
                            destination = output if key.data == "output" else diagnostic
                            destination.extend(chunk)
                            if len(destination) > MAX_OUTPUT_BYTES:
                                if output.startswith(_READY):
                                    return _READY + b'{"status":"candidate_error","message":"hook output limit exceeded"}'
                                raise SandboxInfrastructureError("Oversized sandbox startup output")
        code = process.wait(timeout=1)
        if code:
            if output.startswith(_READY) and code in (-signal.SIGXCPU, 128 + signal.SIGXCPU):
                return _READY + b'{"status":"candidate_error","message":"hook CPU limit exceeded"}'
            raise SandboxInfrastructureError(
                f"Hook sandbox exited {code}: {diagnostic.decode(errors='replace')[:500]}")
        return bytes(output)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream and not stream.closed:
                stream.close()


def invoke(source: str, *, runner_source: str, mode: str = "execute",
           hook: str = "", ctx: dict | None = None, notebook: dict | None = None) -> dict:
    if (mode not in {"validate", "execute"} or type(source) is not str
            or type(runner_source) is not str
            or (ctx is not None and type(ctx) is not dict)
            or (notebook is not None and type(notebook) is not dict)):
        raise SandboxInfrastructureError("Invalid hook sandbox caller schema")
    try:
        _json(notebook or {}, MAX_NOTEBOOK_BYTES)
        request = _json({"runner": runner_source, "source": source, "mode": mode,
                         "hook": hook, "ctx": ctx or {}, "notebook": notebook or {}},
                        MAX_REQUEST_BYTES).encode()
    except (TypeError, ValueError, RecursionError) as exc:
        raise SandboxInfrastructureError(f"Invalid hook sandbox input: {exc}") from exc
    output = _transport(request)
    if not output.startswith(_READY):
        raise SandboxInfrastructureError("Hook sandbox readiness marker missing")
    try:
        reply = json.loads(output[len(_READY):])
        _json(reply, MAX_OUTPUT_BYTES)
    except (ValueError, TypeError, RecursionError, UnicodeError) as exc:
        raise SandboxInfrastructureError("Malformed hook sandbox response") from exc
    if type(reply) is not dict:
        raise SandboxInfrastructureError("Invalid hook sandbox envelope")
    if reply.get("status") == "candidate_error" and set(reply) == {"status", "message"}:
        if type(reply["message"]) is str:
            return reply
    elif reply.get("status") == "ok" and set(reply) == {"status", "effect", "notebook"}:
        if type(reply["notebook"]) is dict and (reply["effect"] is None or type(reply["effect"]) is dict):
            try:
                _json(reply["notebook"], MAX_NOTEBOOK_BYTES)
            except (ValueError, TypeError, RecursionError) as exc:
                raise SandboxInfrastructureError("Invalid sandbox notebook") from exc
            return reply
    raise SandboxInfrastructureError("Invalid hook sandbox status/schema")
