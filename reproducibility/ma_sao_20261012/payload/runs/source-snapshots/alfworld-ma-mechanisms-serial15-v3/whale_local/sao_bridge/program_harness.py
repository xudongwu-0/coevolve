"""Executable, JSON-only CandidateHarness programs for domain experiments.

``validate_program`` accepts one class with the supplied literal attribute names
and optional ``(self, observation)`` hooks. It returns canonical source, the
literal attributes at the top level, and an ``_hooks`` list. Domain code remains
responsible for attribute bounds, hook-result schemas, and constructing *public*
observations: never pass grading labels, private records, or model paths.

Hooks support branches, loops, comprehensions, arithmetic, JSON, bounded regex
operations, and the pure builtins/container methods listed below. Only direct
hook-local ``import re`` / ``import json`` statements are permitted; their
runtime importer exposes allowlisted facades, not unrestricted modules.
Reflection, I/O, other imports, nested functions, decorators, and private
attributes remain unavailable. Missing hooks raise rather than implicitly
substituting a trusted default. EDITOR_API_GUIDE describes the shared grammar.

Validation never executes candidate code. Every invocation starts a fresh
bubblewrap process using the isolation approach in ``retool.execute_python``:
unprivileged UID, private namespaces/network, cleared environment, read-only
system Python, and no project, home, model, package, or GPU mounts. Unlike the
math tool, this runner needs no math packages or temporary output files.
Bounded JSON is transported over pipes. A readiness marker distinguishes setup
failures from resource exhaustion during candidate execution.

Catch ProgramValidationError/HookExecutionError for candidate failures.
SandboxInfrastructureError must stop the campaign, not become a zero reward.
"""
from __future__ import annotations

import ast
import json
import keyword
import math
import os
from pathlib import Path
import signal
import subprocess


MAX_SOURCE_BYTES = 32000
MAX_OBSERVATION_BYTES = 128000
MAX_RESULT_BYTES = 65536
CPU_SECONDS = 10
WALL_SECONDS = 15
MEMORY_BYTES = 256 * 1024 * 1024
_READY = b"SAO_PROGRAM_READY\n"
_BWRAP = Path(__file__).resolve().parents[2] / "assets/tools/bwrap"
_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "chr", "dict", "enumerate", "float", "int", "isinstance",
    "len", "list", "max", "min", "ord", "range", "reversed", "round", "set",
    "sorted", "str", "sum", "tuple", "zip",
})
_METHODS = frozenset({
    "append", "capitalize", "casefold", "center", "clear", "copy", "count",
    "endswith", "expandtabs", "extend", "find", "get", "index", "insert",
    "isalnum", "isalpha", "isascii", "isdecimal", "isdigit", "islower",
    "isnumeric", "isspace", "istitle", "isupper", "items", "join", "keys",
    "ljust", "lower", "lstrip", "partition", "pop", "popitem", "remove",
    "removeprefix", "removesuffix", "replace", "reverse", "rfind", "rindex",
    "rjust", "rpartition", "rsplit", "rstrip", "setdefault", "sort", "split",
    "splitlines", "startswith", "strip", "swapcase", "title", "update",
    "upper", "values", "zfill",
})
_MODULE_FUNCTIONS = {
    "json": frozenset({"loads", "dumps"}),
    "re": frozenset({"search", "match", "fullmatch", "findall", "split", "sub", "subn", "escape"}),
}
_REGEX_FLAGS = frozenset({
    "IGNORECASE", "I", "MULTILINE", "M", "DOTALL", "S", "ASCII", "A",
    "VERBOSE", "X", "UNICODE", "U", "NOFLAG",
})
_MATCH_METHODS = frozenset({"group", "groups", "groupdict", "start", "end", "span"})
_MODULE_APIS = {
    name: functions | (_REGEX_FLAGS if name == "re" else frozenset())
    for name, functions in _MODULE_FUNCTIONS.items()
}
_NODES = (
    ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.If, ast.For, ast.While,
    ast.Break, ast.Continue, ast.Pass, ast.Name, ast.Constant, ast.List, ast.Tuple,
    ast.Set, ast.Dict, ast.Subscript, ast.Slice, ast.Attribute, ast.Call,
    ast.keyword, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.JoinedStr, ast.FormattedValue, ast.Load, ast.Store, ast.Add, ast.Sub,
    ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.BitAnd, ast.BitOr,
    ast.BitXor, ast.LShift, ast.RShift, ast.UAdd, ast.USub, ast.Not, ast.Invert,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Is, ast.IsNot, ast.In, ast.NotIn,
)

EDITOR_API_GUIDE = f"""Shared executable-harness Python contract:
- Return exactly one class CandidateHarness, without bases or decorators. Keep
  exactly the domain's required literal attributes and optional approved hooks.
  Every hook is synchronous with the exact signature (self, observation).
- Imports may appear ONLY INSIDE approved hooks: import re, import json, or
  import re, json. No aliases, from-imports, relative/dotted imports, or other
  modules (including random, os, sys, pathlib, and chess).
- Allowed builtins: {", ".join(sorted(_BUILTINS))}.
- Allowed string/list/dict methods: {", ".join(sorted(_METHODS))}.
- JSON API: json.loads and json.dumps only. json is also available without an
  explicit import for compatibility. No json.load/dump or encoder/decoder classes.
- Regex API after import re: {", ".join("re." + name for name in sorted(_MODULE_FUNCTIONS["re"]))}.
  Flags: {", ".join("re." + name for name in sorted(_REGEX_FLAGS))}.
  Match methods: {", ".join(sorted(_MATCH_METHODS))}. Check a match is not None
  before accessing its groups. No re.compile, re.DEBUG, or module internals.
- Branches, for/while loops, comprehensions, indexing, arithmetic, f-strings,
  and JSON/container mutation are supported. Read settings as self.<attribute>.
  No nested definitions, lambdas, try/except, raise, globals/nonlocals, arbitrary
  function calls, private/dunder attributes, attribute writes, reflection,
  str.format/format_map, eval/exec, files, processes, network, or other I/O.
- Return exactly the hook's domain-specific JSON schema. Match objects, modules,
  tuples, sets, functions, non-string dict keys, and non-finite numbers are not
  valid results. Convert internal regex groups/span/subn tuples to lists if needed.
- Only the public observation is available. Never embed problem-specific answers
  or request hidden labels. An absent hook may use a trusted domain default;
  an implemented hook's error is a candidate failure, NEVER a fallback.
- Source limit {MAX_SOURCE_BYTES} UTF-8 bytes; public observation limit
  {MAX_OBSERVATION_BYTES} ASCII-escaped JSON bytes; result limit {MAX_RESULT_BYTES}
  JSON bytes. JSON has at most 10000 values and 40 nesting levels. Each hook runs
  fresh in an isolated subprocess: {CPU_SECONDS}s CPU, {WALL_SECONDS}s wall time,
  {MEMORY_BYTES // (1024 * 1024)} MiB memory. Regex backtracking uses this same CPU
  budget. Invalid regex/JSON and resource exhaustion are explicit candidate errors.
"""
EDITOR_PROGRAM_GUIDE = EDITOR_API_GUIDE


class ProgramValidationError(ValueError):
    """The proposed source or serialized program spec is invalid."""


class HookExecutionError(ValueError):
    """A candidate hook failed, exceeded its budget, or returned non-JSON."""


class MissingHookError(HookExecutionError):
    """The candidate does not implement the requested optional hook."""


class SandboxInfrastructureError(RuntimeError):
    """Sandbox setup, transport, or caller input failed; stop the campaign."""


def _json_text(value: object, limit: int) -> str:
    pending = [(value, 0)]
    nodes = characters = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 10000 or depth > 40:
            raise ValueError("JSON exceeds 10000 values or 40 nesting levels")
        kind = type(item)
        if kind is str:
            characters += len(item)
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("JSON numbers must be finite")
        elif kind is int:
            if item.bit_length() > 14000:
                raise ValueError("JSON integer is too large")
        elif kind is list or kind is dict:
            if nodes + len(pending) + len(item) > 10000:
                raise ValueError("JSON exceeds 10000 values")
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("JSON object keys must be strings")
                characters += sum(len(key) for key in item)
                children = item.values()
            else:
                children = item
            pending.extend((child, depth + 1) for child in children)
        elif item is not None and kind is not bool:
            raise ValueError(f"Non-JSON value: {kind.__name__}")
        if characters > limit:
            raise ValueError(f"JSON exceeds {limit} bytes")
    text = json.dumps(value, ensure_ascii=True, allow_nan=False,
                      separators=(",", ":"), sort_keys=True)
    if len(text.encode("utf-8")) > limit:
        raise ValueError(f"JSON exceeds {limit} bytes")
    return text


def _public_name(name: object) -> bool:
    return (type(name) is str and name.isidentifier() and not keyword.iskeyword(name)
            and not name.startswith("_"))


class _MethodValidator(ast.NodeVisitor):
    def __init__(self, method: ast.FunctionDef, attributes: set[str]):
        self.attributes = attributes
        self.locals = {node.id for node in ast.walk(method)
                       if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        imported = {alias.name for node in ast.walk(method) if isinstance(node, ast.Import)
                    for alias in node.names if alias.name in _MODULE_APIS}
        self.names = self.locals | imported | _BUILTINS | {"self", "observation", "json"}
        self.loop_depth = 0

    def generic_visit(self, node):
        if not isinstance(node, _NODES):
            raise ProgramValidationError(f"Unsupported hook syntax: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if node.id.startswith("__") or node.id not in self.names:
            raise ProgramValidationError(f"Unavailable name: {node.id}")
        if isinstance(node.ctx, ast.Store) and node.id in _BUILTINS | {"self", "json", "re"}:
            raise ProgramValidationError(f"Cannot rebind reserved name: {node.id}")

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name not in _MODULE_APIS or alias.asname is not None:
                raise ProgramValidationError(
                    "Only direct hook-local imports of re/json without aliases are allowed: "
                    + ast.unparse(node)
                )

    def visit_Constant(self, node):
        if type(node.value) not in (str, int, float, bool, type(None)):
            raise ProgramValidationError("Only JSON constants are permitted")
        if isinstance(node.value, float) and not math.isfinite(node.value):
            raise ProgramValidationError("Numeric constants must be finite")

    def visit_Attribute(self, node):
        if not isinstance(node.ctx, ast.Load) or node.attr.startswith("_"):
            raise ProgramValidationError("Private attributes and attribute writes are forbidden")
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            allowed = node.attr in self.attributes
        elif isinstance(node.value, ast.Name) and node.value.id in _MODULE_APIS:
            allowed = node.attr in _MODULE_APIS[node.value.id]
        else:
            allowed = node.attr in _METHODS | _MATCH_METHODS
        if not allowed:
            raise ProgramValidationError(f"Unavailable attribute: {node.attr}")
        self.visit(node.value)

    def visit_Call(self, node):
        keywords = [arg.arg for arg in node.keywords if arg.arg is not None]
        if len(set(keywords)) != len(keywords):
            raise ProgramValidationError("Duplicate call keyword")
        function = node.func
        if isinstance(function, ast.Name):
            allowed = function.id in _BUILTINS
        elif isinstance(function, ast.Attribute):
            allowed = function.attr in _METHODS | _MATCH_METHODS or (
                isinstance(function.value, ast.Name) and function.value.id in _MODULE_FUNCTIONS
                and function.attr in _MODULE_FUNCTIONS[function.value.id])
        else:
            allowed = False
        if not allowed:
            raise ProgramValidationError("Only approved builtin, module, and container/regex calls are allowed")
        self.generic_visit(node)

    def visit_For(self, node):
        self.visit(node.target)
        self.visit(node.iter)
        self._loop(node)

    def visit_While(self, node):
        self.visit(node.test)
        self._loop(node)

    def _loop(self, node):
        self.loop_depth += 1
        for statement in node.body:
            self.visit(statement)
        self.loop_depth -= 1
        for statement in node.orelse:
            self.visit(statement)

    def visit_Break(self, node):
        if not self.loop_depth:
            raise ProgramValidationError("break requires an enclosing loop")

    def visit_Continue(self, node):
        if not self.loop_depth:
            raise ProgramValidationError("continue requires an enclosing loop")

    def visit_comprehension(self, node):
        if node.is_async:
            raise ProgramValidationError("Async comprehensions are forbidden")
        self.generic_visit(node)


def validate_program(content: str, *, attributes: set[str], hooks: set[str]) -> dict:
    """Validate without exec; normalize source and expose literal fields/hooks."""
    if (type(attributes) is not set or type(hooks) is not set
            or any(not _public_name(name) for name in attributes | hooks)
            or attributes & hooks or "source" in attributes | hooks
            or (attributes | hooks) & {"self", "observation", "json", "re", "CandidateHarness"}):
        raise ProgramValidationError("Invalid or overlapping attribute/hook names")
    if type(content) is not str:
        raise ProgramValidationError("Program source must be a string")
    try:
        if len(content.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ProgramValidationError(f"Program exceeds {MAX_SOURCE_BYTES} source bytes")
        source = content.strip()
        lines = source.splitlines()
        if lines and lines[0] in {"```python", "```py", "```"} and lines[-1] == "```":
            source = "\n".join(lines[1:-1]).strip()
        tree = ast.parse(source, mode="exec")
        if sum(1 for _ in ast.walk(tree)) > 6000:
            raise ProgramValidationError("Program exceeds 6000 AST nodes")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
            raise ProgramValidationError("Expected exactly one CandidateHarness class")
        cls = tree.body[0]
        if (cls.name != "CandidateHarness" or cls.bases or cls.keywords
                or cls.decorator_list or getattr(cls, "type_params", [])):
            raise ProgramValidationError("CandidateHarness cannot have bases, decorators, or type parameters")
        values, implemented = {}, []
        for node in cls.body:
            if isinstance(node, ast.Assign):
                if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                    raise ProgramValidationError("Each attribute needs one simple literal assignment")
                name = node.targets[0].id
                if name not in attributes or name in values:
                    raise ProgramValidationError(f"Unexpected or duplicate attribute: {name}")
                values[name] = ast.literal_eval(node.value)
            elif isinstance(node, ast.FunctionDef):
                args = node.args
                if (node.name not in hooks or node.name in implemented
                        or node.decorator_list or node.returns or node.type_comment
                        or getattr(node, "type_params", [])
                        or args.posonlyargs or args.kwonlyargs or args.defaults
                        or args.kw_defaults or args.vararg or args.kwarg
                        or [arg.arg for arg in args.args] != ["self", "observation"]
                        or any(arg.annotation or arg.type_comment for arg in args.args)):
                    raise ProgramValidationError("Hooks must be unique approved methods with (self, observation)")
                validator = _MethodValidator(node, attributes)
                for statement in node.body:
                    validator.visit(statement)
                implemented.append(node.name)
            elif not (isinstance(node, ast.Pass) and len(cls.body) == 1 and not attributes):
                raise ProgramValidationError("Class body allows only literal attributes and approved hooks")
        if set(values) != attributes:
            raise ProgramValidationError("Program must declare exactly the required literal attributes")
        _json_text(values, MAX_SOURCE_BYTES)
        canonical = ast.unparse(tree) + "\n"
        if len(canonical.encode("utf-8")) > MAX_SOURCE_BYTES:
            raise ProgramValidationError(f"Canonical program exceeds {MAX_SOURCE_BYTES} source bytes")
    except (SyntaxError, ValueError, TypeError, RecursionError, UnicodeError) as exc:
        if isinstance(exc, ProgramValidationError):
            raise
        raise ProgramValidationError(f"Invalid program: {str(exc)[:400]}") from exc
    return {"source": canonical, **values, "_hooks": sorted(implemented)}


# This trusted bootstrap runs only in the sandbox. The candidate's globals are
# separate from the bootstrap's imports, transport payload, and runtime helpers.
_WORKER = r'''
import builtins
import json
import math
import re
import sys
from types import SimpleNamespace

module_apis = __SAO_MODULE_APIS__
public_modules = {
    name: SimpleNamespace(**{key: getattr(module, key) for key in module_apis[name]})
    for name, module in (("json", json), ("re", re))
}

def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or fromlist or name not in public_modules:
        raise ImportError("Only direct imports of re/json are available in candidate hooks")
    return public_modules[name]

def checked_json(value):
    pending = [(value, 0)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > 10000 or depth > 40:
            raise ValueError("result exceeds JSON depth/value limits")
        kind = type(item)
        if kind is dict or kind is list:
            if nodes + len(pending) + len(item) > 10000:
                raise ValueError("result exceeds JSON value limit")
            if kind is dict:
                if any(type(key) is not str for key in item):
                    raise ValueError("result object keys must be strings")
                children = item.values()
            else:
                children = item
            pending.extend((child, depth + 1) for child in children)
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("result numbers must be finite")
        elif kind not in (str, int, bool, type(None)):
            raise ValueError("result must contain only JSON values")
    chunks = []
    size = 0
    for chunk in json.JSONEncoder(ensure_ascii=True, allow_nan=False,
                                  separators=(",", ":")).iterencode(value):
        size += len(chunk)
        if size > 65536:
            raise ValueError("result exceeds 65536 bytes")
        chunks.append(chunk)
    return "".join(chunks)

request = json.loads(sys.stdin.buffer.read(256001))
allowed = {name: getattr(builtins, name) for name in request["builtins"]}
allowed["__build_class__"] = builtins.__build_class__
allowed["__import__"] = restricted_import
namespace = {"__builtins__": allowed, "__name__": "candidate", "json": public_modules["json"]}
sys.stdout.write("SAO_PROGRAM_READY\n")
sys.stdout.flush()
try:
    exec(compile(request["source"], "<candidate>", "exec"), namespace)
    candidate = namespace["CandidateHarness"]()
    value = getattr(candidate, request["hook"])(request["observation"])
    result = checked_json(value)
except (ArithmeticError, AttributeError, ImportError, LookupError, MemoryError, NameError,
        RecursionError, RuntimeError, TypeError, ValueError, re.error) as exc:
    reply = {"status": "candidate_error",
             "message": type(exc).__name__ + ": " + str(exc)[:512]}
    sys.stdout.write(json.dumps(reply, ensure_ascii=True, separators=(",", ":")) + "\n")
else:
    sys.stdout.write('{"status":"ok","value":' + result + "}\n")
'''
_WORKER = _WORKER.replace("__SAO_MODULE_APIS__",
                          repr({name: sorted(api) for name, api in _MODULE_APIS.items()}))


def _sandbox_command() -> list[str]:
    uid, gid = os.getuid(), os.getgid()
    if uid == 0 or gid == 0:
        raise SandboxInfrastructureError("Candidate sandbox requires an unprivileged UID/GID")
    return [
        "sudo", "-n", str(_BWRAP), "--unshare-all", "--die-with-parent",
        "--new-session", "--uid", str(uid), "--gid", str(gid), "--cap-drop", "ALL",
        "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64", "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/work", "--chdir", "/work", "--clearenv",
        "--setenv", "PATH", "/usr/bin:/bin",
        # Distinct soft/hard limits make CPU exhaustion report SIGXCPU, so an
        # unrelated SIGKILL (for example a node OOM) remains an infrastructure error.
        "--", "/usr/bin/prlimit", f"--as={MEMORY_BYTES}",
        f"--cpu={CPU_SECONDS}:{CPU_SECONDS + 1}",
        "--core=0", "--fsize=65536", "--nproc=16", "--nofile=32",
        "--", "/usr/bin/python3", "-I", "-S", "-B", "-c", _WORKER,
    ]


def execute_hook(spec: dict, hook: str, observation: dict):
    """Execute a validated optional hook in a fresh bounded sandbox.

    Candidate failures raise HookExecutionError (including MissingHookError).
    Invalid specs raise ProgramValidationError. Invalid *caller* observations
    and sandbox failures raise SandboxInfrastructureError, never a fallback.
    """
    if (type(spec) is not dict or type(spec.get("_hooks")) is not list
            or any(not _public_name(name) for name in spec["_hooks"])
            or len(set(spec["_hooks"])) != len(spec["_hooks"])):
        raise ProgramValidationError("Expected a validated program spec with an _hooks list")
    validated = validate_program(
        spec.get("source"), attributes=set(spec) - {"source", "_hooks"},
        hooks=set(spec["_hooks"]),
    )
    try:
        if _json_text(validated, 256000) != _json_text(spec, 256000):
            raise ProgramValidationError("Program spec does not match its canonical source")
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ProgramValidationError):
            raise
        raise ProgramValidationError(f"Invalid program spec: {str(exc)[:400]}") from exc
    if type(hook) is not str or hook not in validated["_hooks"]:
        raise MissingHookError(f"Candidate does not implement hook {hook!r}")
    try:
        if type(observation) is not dict:
            raise ValueError("Public observation must be a JSON object")
        _json_text(observation, MAX_OBSERVATION_BYTES)
        request = _json_text({
            "source": validated["source"], "hook": hook, "observation": observation,
            "builtins": sorted(_BUILTINS),
        }, 256000).encode("utf-8")
    except (ValueError, TypeError, RecursionError) as exc:
        raise SandboxInfrastructureError(f"Invalid public observation: {str(exc)[:400]}") from exc
    try:
        completed = subprocess.run(
            _sandbox_command(), input=request, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=WALL_SECONDS, check=False, close_fds=True,
        )
    except subprocess.TimeoutExpired as exc:
        if (exc.stdout or b"").startswith(_READY):
            raise HookExecutionError(f"Hook {hook} exceeded {WALL_SECONDS}s wall deadline") from exc
        raise SandboxInfrastructureError("Sandbox timed out before candidate execution") from exc
    except OSError as exc:
        raise SandboxInfrastructureError(f"Cannot start candidate sandbox: {exc}") from exc
    stdout, stderr = completed.stdout, completed.stderr
    ready = stdout.startswith(_READY)
    if completed.returncode:
        # bwrap forwards the child's signal as 128 + signal on some versions.
        if ready and completed.returncode in {-signal.SIGXCPU, 128 + signal.SIGXCPU}:
            raise HookExecutionError(f"Hook {hook} exceeded sandbox CPU limit")
        diagnostic = stderr.decode("utf-8", errors="replace")[:500]
        raise SandboxInfrastructureError(
            f"Candidate sandbox exited {completed.returncode}: {diagnostic}"
        )
    if not ready or len(stdout) > MAX_RESULT_BYTES + 4096:
        raise SandboxInfrastructureError("Invalid or oversized sandbox response")
    try:
        reply = json.loads(stdout[len(_READY):])
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise SandboxInfrastructureError("Malformed sandbox JSON response") from exc
    if type(reply) is not dict:
        raise SandboxInfrastructureError("Invalid sandbox response envelope")
    if (set(reply) == {"status", "message"} and reply["status"] == "candidate_error"
            and type(reply["message"]) is str):
        raise HookExecutionError(f"Hook {hook} failed: {reply['message'][:600]}")
    if set(reply) != {"status", "value"} or reply["status"] != "ok":
        raise SandboxInfrastructureError("Unknown sandbox response status")
    try:
        _json_text(reply["value"], MAX_RESULT_BYTES)
    except (ValueError, TypeError, RecursionError) as exc:
        raise SandboxInfrastructureError("Sandbox returned an invalid JSON value") from exc
    return reply["value"]
