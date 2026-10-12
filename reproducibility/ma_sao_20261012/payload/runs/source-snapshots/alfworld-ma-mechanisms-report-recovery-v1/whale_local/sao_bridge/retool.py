"""Bounded ReTool actions and isolated Python execution; generated harnesses are never exec'd."""
from __future__ import annotations

import ast
import asyncio
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import tempfile
import time

from .common import ROOT, binary_reward

INITIAL_SOURCE = '''from environments.retool.base_harness import ReToolEnv
class CandidateEnv(ReToolEnv):
    MAX_TURNS = 4
    SYSTEM_PROMPT = "Solve the math problem. Use code_interpreter for exact calculations when helpful. Finish with a boxed answer."
    USER_PROMPT_TEMPLATE = "{question}"
    RETRY_ON_NO_TOOL = True
    TOOL_OUTPUT_LIMIT = 2048
    RETRY_PROMPT = "Finish your solution concisely and give the final answer in a box."
'''
EDITOR_SYSTEM = """Improve the reusable Python-tool math harness below. Return ONLY this
Python module structure, changing literal values, no functions or explanations:
""" + INITIAL_SOURCE + """
Keep the exact import, class name and six literal attributes. MAX_TURNS is 2-8.
SYSTEM_PROMPT and RETRY_PROMPT are single-line string literals. USER_PROMPT_TEMPLATE
contains {question} exactly once and no other format fields. RETRY_ON_NO_TOOL is a
boolean. TOOL_OUTPUT_LIMIT is an integer 128-4096. No new imports, methods, grading
changes or problem-specific answers. The tool and total token budget are fixed."""
TOOLS = [{"type": "function", "function": {
    "name": "code_interpreter", "description": "Execute Python for intermediate mathematical computation.",
    "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                   "required": ["code"], "additionalProperties": False},
}}]


class ContextExhausted(RuntimeError):
    """The conversation no longer fits the model context: the episode ends truncated.

    Mirrors the training client, which clips each turn's generation to the remaining
    context and refuses a prompt that already fills it.
    """


def harness_spec(content: str) -> dict:
    import string
    source = content.strip()
    if source.startswith("```python") and source.endswith("```"):
        source = source[len("```python"):-3].strip()
    tree = ast.parse(source)
    if len(tree.body) != 2:
        raise ValueError("Expected exactly one import and one class")
    imported, cls = tree.body
    if (
        not isinstance(imported, ast.ImportFrom) or imported.level
        or imported.module != "environments.retool.base_harness"
        or [(n.name, n.asname) for n in imported.names] != [("ReToolEnv", None)]
        or not isinstance(cls, ast.ClassDef) or cls.name != "CandidateEnv"
        or len(cls.bases) != 1 or not isinstance(cls.bases[0], ast.Name)
        or cls.bases[0].id != "ReToolEnv" or cls.decorator_list or cls.keywords
    ):
        raise ValueError("Unexpected harness declaration")
    values = {}
    for node in cls.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            raise ValueError("Only literal settings may be edited")
        key = node.targets[0].id
        if key in values:
            raise ValueError("Duplicate setting")
        values[key] = ast.literal_eval(node.value)
    if set(values) != {"MAX_TURNS", "SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE",
                       "RETRY_ON_NO_TOOL", "TOOL_OUTPUT_LIMIT", "RETRY_PROMPT"}:
        raise ValueError("Expected all six harness settings")
    if type(values["MAX_TURNS"]) is not int or not 2 <= values["MAX_TURNS"] <= 8:
        raise ValueError("Invalid turn limit")
    if type(values["TOOL_OUTPUT_LIMIT"]) is not int or not 128 <= values["TOOL_OUTPUT_LIMIT"] <= 4096:
        raise ValueError("Invalid tool-output limit")
    if type(values["RETRY_ON_NO_TOOL"]) is not bool:
        raise ValueError("Invalid recovery setting")
    for key in ("SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE", "RETRY_PROMPT"):
        if not isinstance(values[key], str) or not 1 <= len(values[key]) <= 1600:
            raise ValueError(f"Invalid prompt {key}")
    fields = [(f, s, c) for _, f, s, c in string.Formatter().parse(values["USER_PROMPT_TEMPLATE"]) if f is not None]
    if fields != [("question", "", None)]:
        raise ValueError("Only one plain question interpolation is permitted")
    return {"source": source + "\n", **values}


def prepare_python(code: str) -> str:
    matches = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", code, re.DOTALL)
    code = matches[0].strip() if matches else code
    lines = code.splitlines()
    # Match the upstream last-expression printing behavior, without changing statements.
    try:
        parsed = ast.parse(code)
        if parsed.body and isinstance(parsed.body[-1], ast.Expr):
            last = parsed.body[-1]
            if not (isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
                    and last.value.func.id == "print"):
                expression = ast.get_source_segment(code, last)
                code = "\n".join(lines[:last.lineno - 1]) + "\nprint(" + expression + ")"
    except SyntaxError:
        pass
    return code


def execute_python(code: str, timeout: int = 10) -> dict:
    if len(code) > 32000:
        return {"output": "Tool code exceeds 32000 characters.", "error": True}
    code = prepare_python(code)
    packages = ROOT / ".venv-ppo/lib/python3.12/site-packages"
    mounts = []
    for package in ("sympy", "mpmath"):
        path = packages / package
        if not path.is_dir():
            raise RuntimeError(f"Required isolated math package is missing: {package}")
        mounts.extend(["--ro-bind", str(path), f"/opt/math/{package}"])
    # Node-local tmpfs: some nodes have a full root disk, and 3FS lacks needed file semantics.
    with tempfile.TemporaryDirectory(prefix="sao-tool-", dir="/dev/shm") as directory:
        root = Path(directory)
        cmd = [
            "sudo", "-n", str(ROOT / "assets/tools/bwrap"), "--unshare-all", "--die-with-parent",
            "--uid", str(os.getuid()), "--gid", str(os.getgid()), "--cap-drop", "ALL",
            "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
            *mounts,
            "--tmpfs", "/tmp", "--tmpfs", "/work",
            "--chdir", "/work", "--clearenv", "--setenv", "PATH", "/usr/bin:/bin",
            "--", "/usr/bin/prlimit", "--as=2147483648", "--cpu=10", "--fsize=65536",
            "--nproc=32", "--nofile=64", "--", "/usr/bin/python3", "-I", "-c",
            "import sys; sys.path.insert(0, '/opt/math'); "
            "exec(compile(sys.stdin.read(), '<tool>', 'exec'), {'__name__': '__main__'})",
        ]
        with (root / "out").open("wb") as out, (root / "err").open("wb") as err:
            try:
                result = subprocess.run(cmd, input=code.encode(), stdout=out, stderr=err,
                                        timeout=timeout + 5)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Isolated Python exceeded hard execution deadline") from exc
        stdout = (root / "out").read_text(errors="replace")[:65536]
        stderr = (root / "err").read_text(errors="replace")[:65536]
        if "bwrap:" in stderr or "sudo:" in stderr:
            raise RuntimeError("Python sandbox infrastructure failed: " + stderr[:500])
        return {"output": stdout + stderr, "error": result.returncode != 0}


async def run_episode(row: dict, spec: dict, generate, total_tokens=4096, seed=42):
    messages = [
        {"role": "system", "content": spec["SYSTEM_PROMPT"]},
        {"role": "user", "content": spec["USER_PROMPT_TEMPLATE"].format(
            question=row["question"]) + "\nGive your final answer in \\boxed{...}."},
    ]
    remaining = total_tokens
    calls = []
    tool_count = tool_errors = 0
    last_text = ""
    started = time.monotonic()
    for turn in range(spec["MAX_TURNS"]):
        # Reserve tokens for observing tool results and emitting a final answer.
        cap = remaining if turn == spec["MAX_TURNS"] - 1 else min(remaining, max(256, total_tokens // 2))
        try:
            completion = await generate(messages, TOOLS, cap, seed + turn)
        except ContextExhausted:
            remaining = 0
            break
        choice = completion["choices"][0]
        used = completion["usage"]["completion_tokens"]
        if not 0 < used <= cap:
            raise RuntimeError("Invalid generation token accounting")
        remaining -= used
        message = dict(choice["message"])
        last_text = message.get("content") or ""
        messages.append(message)
        if completion.get("record"):
            calls.append(completion["record"])
        tools = message.get("tool_calls") or []
        if not tools:
            if "\\boxed{" in last_text or not spec["RETRY_ON_NO_TOOL"] or remaining <= 0:
                break
            messages.append({"role": "user", "content": spec["RETRY_PROMPT"]})
        else:
            for tool in tools:
                tool_count += 1
                try:
                    args = json.loads(tool["function"]["arguments"])
                    if tool["function"]["name"] != "code_interpreter" or set(args) != {"code"} or not isinstance(args["code"], str):
                        raise ValueError("Expected code_interpreter with a code string")
                except (ValueError, TypeError) as exc:
                    result = {"output": f"Invalid tool arguments: {exc}", "error": True}
                else:
                    result = await asyncio.to_thread(execute_python, args["code"])
                tool_errors += int(result["error"])
                messages.append({"role": "tool", "tool_call_id": tool["id"],
                                 "content": result["output"][:spec["TOOL_OUTPUT_LIMIT"]]})
        if remaining <= 0:
            break
    return {
        "id": row["id"], "reward": binary_reward(last_text, row["answer"]),
        "messages": messages, "calls": calls, "tokens": total_tokens - remaining,
        "truncated": remaining == 0, "boxed": "\\boxed{" in last_text,
        "tool_calls": tool_count, "tool_errors": tool_errors,
        "elapsed_seconds": time.monotonic() - started,
    }
