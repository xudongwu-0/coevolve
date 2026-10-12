import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import program_harness as harness


SOURCE = '''class CandidateHarness:
    MAX_TURNS = 3
    PROMPT = "Find relevant public evidence."

    def build_prompt(self, observation):
        terms = observation["question"].lower().split()
        snippets = []
        for item in observation.get("results", []):
            text = item.get("text", "").strip()
            if any(word in text.lower() for word in terms):
                snippets.append(text)
            if len(snippets) >= self.MAX_TURNS:
                break
        return self.PROMPT + "\\n" + "\\n".join(snippets)

    def select_action(self, observation):
        moves = sorted(observation.get("legal_moves", []))
        if not moves:
            return {"action": "stop"}
        return {"action": "move", "move": moves[0]}
'''
ATTRIBUTES = {"MAX_TURNS", "PROMPT"}
HOOKS = {"build_prompt", "select_action", "postprocess"}


def spec(source=SOURCE):
    return harness.validate_program(source, attributes=ATTRIBUTES, hooks=HOOKS)


def body_spec(body):
    return spec('class CandidateHarness:\n    MAX_TURNS = 3\n    PROMPT = "p"\n'
                '    def build_prompt(self, observation):\n'
                + "\n".join("        " + line for line in body.splitlines()) + "\n")


def response(value=None, *, error=None, returncode=0, stderr=b""):
    envelope = ({"status": "candidate_error", "message": error} if error is not None
                else {"status": "ok", "value": value})
    output = harness._READY + json.dumps(envelope, allow_nan=False).encode() + b"\n"
    return subprocess.CompletedProcess([], returncode, output, stderr)


class ProgramValidationTests(unittest.TestCase):
    def test_canonical_source_and_literal_fields(self):
        result = spec("```python\n" + SOURCE + "```")
        self.assertEqual(set(result), {"source", "MAX_TURNS", "PROMPT", "_hooks"})
        self.assertEqual(result["MAX_TURNS"], 3)
        self.assertEqual(result["_hooks"], ["build_prompt", "select_action"])
        self.assertEqual(spec(result["source"]), result)
        self.assertTrue(result["source"].startswith("class CandidateHarness:"))
        self.assertTrue(result["source"].endswith("\n"))

    def test_literal_attributes_need_no_methods_or_domain_bounds(self):
        result = spec('class CandidateHarness:\n    MAX_TURNS = -100\n    PROMPT = ""\n')
        self.assertEqual(result["_hooks"], [])
        self.assertEqual(result["MAX_TURNS"], -100)
        self.assertEqual(harness.validate_program(
            "class CandidateHarness:\n    pass", attributes=set(), hooks=set()
        )["_hooks"], [])

    def test_json_literal_settings(self):
        result = harness.validate_program(
            'class CandidateHarness:\n    SETTINGS = {"a": [1, True, None, -2.5]}\n',
            attributes={"SETTINGS"}, hooks=set(),
        )
        self.assertEqual(result["SETTINGS"], {"a": [1, True, None, -2.5]})

    def test_safe_imports_are_hook_local_and_preserved_canonically(self):
        candidate = body_spec('''
import re, json
response = observation.get("response", "")
if not isinstance(response, str):
    return None
match = re.search("<move>([^<]+)</move>", response, re.IGNORECASE)
return json.loads(json.dumps(match.group(1).strip())) if match else None
'''.strip())
        self.assertIn("import re, json", candidate["source"])
        self.assertEqual(spec(candidate["source"]), candidate)
        for prefix in ("import re\n", "import json\n"):
            with self.subTest(prefix=prefix), self.assertRaises(harness.ProgramValidationError):
                spec(prefix + SOURCE)
        with self.assertRaises(harness.ProgramValidationError):
            spec(SOURCE.replace("    MAX_TURNS = 3", "    import re\n    MAX_TURNS = 3"))

    def test_import_allowlist_does_not_enable_aliases_reflection_or_other_modules(self):
        bodies = [
            'import os\nreturn None',
            'import random\nreturn None',
            'import sys\nreturn None',
            'import pathlib\nreturn None',
            'import socket\nreturn None',
            'import subprocess\nreturn None',
            'import chess\nreturn None',
            'import re, os\nreturn None',
            'import re as regex\nreturn None',
            'import json as json\nreturn None',
            'import json.tool\nreturn None',
            'from re import search\nreturn None',
            'from json import loads\nreturn None',
            'from re import *\nreturn None',
            'from . import json\nreturn None',
            'return __import__("re")',
            'import re\nreturn re.__dict__',
            'import re\nreturn re._compiler',
            'import re\nreturn re.search.__globals__',
            'import re\nreturn re.DEBUG',
            'import re\nreturn re.compile("x")',
            'import re\nreturn re.purge()',
            'import re\nreturn re.search("x", "x").re',
            'import re\nreturn re.search("x", "x").__class__',
            'import re\nre.search = str\nreturn None',
            'import re\nre = observation\nreturn None',
            'import re\nalias = re\nreturn alias.search("x", "x")',
            'import json\nreturn json.load(observation)',
            'import json\nreturn json.dump(observation, observation)',
            'import json\nreturn json.JSONDecoder()',
            'import json\nreturn json.decoder',
            'import json\nreturn json.loads.__globals__',
            'return re.search("x", "x")',
        ]
        with patch.object(harness.subprocess, "run") as run:
            for body in bodies:
                with self.subTest(body=body), self.assertRaises(harness.ProgramValidationError):
                    body_spec(body)
            run.assert_not_called()

    def test_shared_editor_guide_tracks_the_executable_allowlists(self):
        guide = harness.EDITOR_API_GUIDE
        self.assertIs(guide, harness.EDITOR_PROGRAM_GUIDE)
        for module, names in harness._MODULE_APIS.items():
            for name in names:
                self.assertIn(f"{module}.{name}", guide)
        for name in harness._BUILTINS | harness._MATCH_METHODS:
            self.assertIn(name, guide)
        for constraint in ("INSIDE", "No aliases", "random", "NEVER a fallback",
                           "32000", "128000", "65536"):
            self.assertIn(constraint, guide)

    def test_class_structure_is_exact(self):
        invalid = [
            SOURCE + "\nprint('side effect')",
            "import os\n" + SOURCE,
            SOURCE.replace("CandidateHarness:", "Other:"),
            SOURCE.replace("CandidateHarness:", "CandidateHarness(object):"),
            "@staticmethod\n" + SOURCE,
            SOURCE.replace("CandidateHarness:", "CandidateHarness(metaclass=type):"),
            SOURCE.replace("CandidateHarness:", "CandidateHarness[T]:"),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS: int = 3'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = OTHER = 3'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = 3\n    MAX_TURNS = 2'),
            SOURCE.replace('MAX_TURNS = 3', 'UNEXPECTED = 3'),
            SOURCE.replace('MAX_TURNS = 3\n', ''),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = len([1, 2])'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = {1, 2}'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = (1, 2)'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = {1: "x"}'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = 1e999'),
            SOURCE.replace('MAX_TURNS = 3', 'MAX_TURNS = 3\n    "class docstring"'),
        ]
        for source in invalid:
            with self.subTest(source=source[:100]), self.assertRaises(harness.ProgramValidationError):
                spec(source)

    def test_hook_signature_and_declarations_are_exact(self):
        invalid = [
            SOURCE.replace("def build_prompt", "async def build_prompt"),
            SOURCE.replace("def build_prompt", "@staticmethod\n    def build_prompt"),
            SOURCE.replace("def build_prompt", "def unapproved"),
            SOURCE.replace("(self, observation)", "(self, observation=None)"),
            SOURCE.replace("(self, observation)", "(self, other)"),
            SOURCE.replace("(self, observation)", "(self, observation, extra)"),
            SOURCE.replace("(self, observation)", "(self, *, observation)"),
            SOURCE.replace("(self, observation)", "(self, observation, /)"),
            SOURCE.replace("(self, observation)", "(self, observation: dict)"),
            SOURCE.replace("(self, observation):", "(self, observation) -> str:"),
            SOURCE.replace("(self, observation)", "(self, observation, *args)"),
            SOURCE.replace("(self, observation)", "(self, observation, **kwargs)"),
            SOURCE + "\n    def build_prompt(self, observation):\n        return None\n",
        ]
        for source in invalid:
            with self.subTest(source=source[-100:]), self.assertRaises(harness.ProgramValidationError):
                spec(source)

    def test_dangerous_operations_never_reach_subprocess(self):
        invalid = [
            'import os\nreturn os.environ',
            'from pathlib import Path\nreturn Path("/")',
            'return open("/etc/passwd").read()',
            'return eval(observation["code"])',
            'return exec(observation["code"])',
            'return compile("1", "x", "eval")',
            'return __import__("os")',
            'return globals()',
            'return locals()',
            'return getattr(self, "__dict__")',
            'return vars(self)',
            'return self.__class__',
            'return observation.__class__.__mro__',
            'return observation._private',
            'return "{0.__class__}".format(observation)',
            'return "{x.__class__}".format_map(observation)',
            'return json.__dict__',
            'return json.load(observation)',
            'return json.dump(observation, observation)',
            'return observation.read()',
            'return observation.write("x")',
            'return observation.system("id")',
            'return observation.tofile("x")',
            'return (lambda: 1)()',
            'def nested():\n    return 1\nreturn nested()',
            'class Nested:\n    pass\nreturn Nested',
            'self.MAX_TURNS = 5\nreturn self.MAX_TURNS',
            'observation.attr = 5\nreturn 5',
            'global PROMPT\nreturn PROMPT',
            'try:\n    return 1\nexcept:\n    return 2',
            'raise ValueError("x")',
            'with observation:\n    return 1',
            'yield 1',
            'return [x async for x in observation]',
            'return (x := 1)',
            'return dict(x=1, x=2)',
            'return secret_answer',
            'return [*observation]',
            'return b"bytes"',
            'return 1j',
            'break',
            'continue',
            'len = 1\nreturn len',
            'self = observation\nreturn self',
            'json = observation\nreturn json',
        ]
        with patch.object(harness.subprocess, "run") as run:
            for body in invalid:
                with self.subTest(body=body), self.assertRaises(harness.ProgramValidationError):
                    body_spec(body)
            run.assert_not_called()

    def test_size_syntax_and_contract_limits(self):
        for source in ("not python", "", "\x00", "\ud800", "#" * 32001):
            with self.subTest(source=source[:30]), self.assertRaises(harness.ProgramValidationError):
                spec(source)
        for attributes, hooks in [
            ({"_hidden"}, set()), ({"source"}, set()), ({"x"}, {"x"}),
            ({"self"}, set()), ({"while"}, set()), ({"x"}, {"__call__"}),
            ({"a.b"}, set()), ({"observation"}, set()), ({"json"}, set()),
        ]:
            with self.subTest(attributes=attributes, hooks=hooks):
                with self.assertRaises(harness.ProgramValidationError):
                    harness.validate_program(SOURCE, attributes=attributes, hooks=hooks)
        with self.assertRaisesRegex(harness.ProgramValidationError, "6000 AST"):
            body_spec("x = [" + ",".join("1" for _ in range(6200)) + "]\nreturn x")


class ProgramExecutionMockTests(unittest.TestCase):
    def test_observation_is_only_data_and_sandbox_is_resource_limited(self):
        observation = {"question": '"); import os; #', "nested": ["public", 1]}
        with patch.object(harness.subprocess, "run", return_value=response({"query": "public"})) as run:
            self.assertEqual(harness.execute_hook(spec(), "build_prompt", observation), {"query": "public"})
        args, kwargs = run.call_args
        command = args[0]
        request = json.loads(kwargs["input"])
        self.assertEqual(request["observation"], observation)
        self.assertEqual(request["source"], spec()["source"])
        self.assertEqual(request["hook"], "build_prompt")
        self.assertEqual(kwargs["timeout"], harness.WALL_SECONDS)
        self.assertTrue(kwargs["close_fds"])
        for flag in ("--unshare-all", "--clearenv", "--cap-drop", "--new-session",
                     "--cpu=10:11", "--as=268435456", "--nproc=16", "--nofile=32",
                     "-I", "-S", "-B"):
            self.assertIn(flag, command)
        self.assertNotIn("eval", request["builtins"])
        self.assertNotIn("open", request["builtins"])
        mounts = [command[index + 1] for index, flag in enumerate(command)
                  if flag in {"--ro-bind", "--bind", "--dev-bind"}]
        self.assertEqual(mounts, ["/usr", "/lib", "/lib64"])
        self.assertNotIn(observation["question"], command[-1])

    def test_missing_hook_raises_without_starting_sandbox(self):
        with patch.object(harness.subprocess, "run") as run:
            with self.assertRaises(harness.MissingHookError):
                harness.execute_hook(spec(), "postprocess", {})
            run.assert_not_called()

    def test_spec_tampering_is_rejected(self):
        original = spec()
        altered = [
            {**original, "MAX_TURNS": 99},
            {**original, "MAX_TURNS": True},
            {**original, "_hooks": []},
            {**original, "_hooks": ["build_prompt", "build_prompt"]},
            {**original, "_hooks": ["__init__"]},
            {**original, "source": SOURCE + "\nimport os"},
            {**original, "source": original["source"].replace("terms =", "other =")},
        ]
        for candidate in altered:
            with self.subTest(candidate=candidate), self.assertRaises(harness.ProgramValidationError):
                harness.execute_hook(candidate, "build_prompt", {})

    def test_invalid_public_inputs_are_not_candidate_failures(self):
        cyclic = {}
        cyclic["cycle"] = cyclic
        invalid = [
            [], {"x": float("nan")}, {"x": object()}, {1: "not a string key"},
            {"x": (1, 2)}, {"x": "x" * 128001}, cyclic,
        ]
        with patch.object(harness.subprocess, "run") as run:
            for observation in invalid:
                with self.subTest(observation_type=type(observation)):
                    with self.assertRaises(harness.SandboxInfrastructureError):
                        harness.execute_hook(spec(), "build_prompt", observation)
            run.assert_not_called()

    def test_candidate_failure_is_not_silently_replaced(self):
        with patch.object(harness.subprocess, "run", return_value=response(error="KeyError: missing")):
            with self.assertRaisesRegex(harness.HookExecutionError, "build_prompt.*KeyError"):
                harness.execute_hook(spec(), "build_prompt", {})

    def test_setup_failure_and_malformed_protocol_stop_campaign(self):
        outputs = [
            subprocess.CompletedProcess([], 1, b"", b"bwrap: permission denied"),
            subprocess.CompletedProcess([], 1, b"", b"prlimit: cannot set limit"),
            subprocess.CompletedProcess([], 0, b"hello", b""),
            subprocess.CompletedProcess([], 0, harness._READY + b"not JSON", b""),
            subprocess.CompletedProcess([], 0, harness._READY + b"[]", b""),
            subprocess.CompletedProcess([], 0, harness._READY + b'{"status":"other"}', b""),
            subprocess.CompletedProcess([], 0, harness._READY + b'{"status":"ok","value":NaN}', b""),
            subprocess.CompletedProcess([], 0, harness._READY + b"x" * 70000, b""),
            response(returncode=1, stderr=b"Unexpected bootstrap failure"),
            response(returncode=-signal.SIGTERM),
            response(returncode=-signal.SIGKILL),
            response(returncode=137),
            subprocess.CompletedProcess([], -signal.SIGKILL, b"", b""),
        ]
        for completed in outputs:
            with self.subTest(returncode=completed.returncode, stdout=completed.stdout[:100]):
                with patch.object(harness.subprocess, "run", return_value=completed):
                    with self.assertRaises(harness.SandboxInfrastructureError):
                        harness.execute_hook(spec(), "build_prompt", {})

    def test_missing_binary_and_permissions_stop_campaign(self):
        for error in (FileNotFoundError("sudo"), PermissionError("bwrap")):
            with patch.object(harness.subprocess, "run", side_effect=error):
                with self.assertRaises(harness.SandboxInfrastructureError):
                    harness.execute_hook(spec(), "build_prompt", {})
        with patch.object(harness.os, "getuid", return_value=0):
            with self.assertRaisesRegex(harness.SandboxInfrastructureError, "unprivileged"):
                harness.execute_hook(spec(), "build_prompt", {})

    def test_only_post_readiness_resource_exhaustion_is_candidate_error(self):
        for code in (-signal.SIGXCPU, 152):
            with self.subTest(code=code):
                with patch.object(harness.subprocess, "run", return_value=response(returncode=code)):
                    with self.assertRaisesRegex(harness.HookExecutionError, "CPU limit"):
                        harness.execute_hook(spec(), "build_prompt", {})
        for output, expected in ((harness._READY, harness.HookExecutionError),
                                 (b"", harness.SandboxInfrastructureError)):
            error = subprocess.TimeoutExpired("sandbox", 15, output=output)
            with patch.object(harness.subprocess, "run", side_effect=error):
                with self.assertRaises(expected):
                    harness.execute_hook(spec(), "build_prompt", {})


class ProgramRealSandboxTests(unittest.TestCase):
    """CPU-only real bwrap tests: setup failures intentionally fail, not skip."""

    def test_real_branches_loops_and_domain_like_hooks(self):
        candidate = spec()
        result = harness.execute_hook(candidate, "build_prompt", {
            "question": "Paris river",
            "results": [{"text": " noise "}, {"text": " Paris is in France. "},
                        {"text": "The Seine river runs through Paris."}],
        })
        self.assertEqual(result, "Find relevant public evidence.\nParis is in France."
                         "\nThe Seine river runs through Paris.")
        self.assertEqual(harness.execute_hook(candidate, "select_action", {
            "legal_moves": ["g1f3", "e2e4", "b1c3"],
        }), {"action": "move", "move": "b1c3"})
        self.assertEqual(harness.execute_hook(candidate, "select_action", {}),
                         {"action": "stop"})

    def test_real_json_fstrings_comprehensions_and_collection_operations(self):
        candidate = body_spec('''
data = json.loads(observation["data"])
counts = {}
for word in data["words"]:
    key = word.lower().strip()
    counts[key] = counts.get(key, 0) + 1
pairs = [[key, value] for key, value in counts.items() if value > 1]
pairs.sort()
n = 0
while n < 3:
    n += 1
    if n < 2:
        continue
    pairs.append([f"step-{n}", n])
return {"summary": json.dumps(pairs), "total": sum(counts.values()),
        "unique": sorted(set(data["words"])), "prompt": self.PROMPT}
'''.strip())
        result = harness.execute_hook(candidate, "build_prompt", {
            "data": '{"words":["Apple","apple"," pear ","Apple"]}',
        })
        self.assertEqual(json.loads(result["summary"]), [["apple", 3], ["step-2", 2], ["step-3", 3]])
        self.assertEqual(result["total"], 4)
        self.assertEqual(result["unique"], [" pear ", "Apple", "apple"])

    def test_real_mutation_is_local_and_runs_are_fresh(self):
        candidate = harness.validate_program('''class CandidateHarness:
    VALUES = []
    def build_prompt(self, observation):
        self.VALUES.append("one")
        observation["changed"] = True
        return self.VALUES
''', attributes={"VALUES"}, hooks={"build_prompt"})
        observation = {"original": 1}
        for _ in range(2):
            self.assertEqual(harness.execute_hook(candidate, "build_prompt", observation), ["one"])
        self.assertEqual(observation, {"original": 1})
        self.assertEqual(candidate["VALUES"], [])

    def test_real_regex_json_imports_and_all_public_match_apis(self):
        candidate = body_spec(r'''
import re, json
text = observation["text"]
match = re.search(r"(?P<name>[a-z]+)-(\d+)", text, re.IGNORECASE | re.ASCII)
if not match:
    return None
return {
    "search": match.group(0), "named": match.group("name"),
    "groups": list(match.groups()), "groupdict": match.groupdict(),
    "start": match.start(), "end": match.end(), "span": list(match.span()),
    "fullmatch": bool(re.fullmatch(r"[a-z]+-\d+", text, re.I)),
    "match": bool(re.match("Widget", text)), "findall": re.findall(r"\d+", text),
    "split": re.split("-", text), "sub": re.sub(r"\d+", "N", text),
    "subn": list(re.subn(r"\d+", "N", text)),
    "escape": re.escape(text), "isinstance": isinstance(text, str),
    "json": json.loads(json.dumps({"ok": True})),
    "multiline": bool(re.search("^more", "one\nmore", re.MULTILINE)),
    "dotall": bool(re.fullmatch("a.b", "a\nb", re.DOTALL)),
    "verbose": bool(re.fullmatch("a b", "ab", re.VERBOSE)),
    "unicode": bool(re.fullmatch(r"\w+", "é", re.UNICODE)),
    "noflag": bool(re.fullmatch("x", "x", re.NOFLAG)),
}
'''.strip())
        result = harness.execute_hook(candidate, "build_prompt", {"text": "Widget-42"})
        self.assertEqual(result, {
            "search": "Widget-42", "named": "Widget", "groups": ["Widget", "42"],
            "groupdict": {"name": "Widget"}, "start": 0, "end": 9, "span": [0, 9],
            "fullmatch": True, "match": True, "findall": ["42"],
            "split": ["Widget", "42"], "sub": "Widget-N", "subn": ["Widget-N", 1],
            "escape": r"Widget\-42", "isinstance": True, "json": {"ok": True},
            "multiline": True, "dotall": True, "verbose": True, "unicode": True,
            "noflag": True,
        })
        self.assertIsNone(harness.execute_hook(candidate, "build_prompt", {"text": "no match"}))

    def test_real_regex_and_json_failures_remain_candidate_failures(self):
        for body in (
            'import re\nreturn re.search("[", observation["response"])',
            'import re\nreturn re.search("x", "x").group(2)',
            'import re\nreturn re.search("x", "x")',
            'import re\nreturn re',
            'import json\nreturn json.loads(observation["response"])',
        ):
            with self.subTest(body=body), self.assertRaises(harness.HookExecutionError):
                harness.execute_hook(body_spec(body), "build_prompt", {"response": "bad input"})

    def test_real_restricted_importer_and_module_facades_defend_beyond_ast(self):
        # Bypass the host validator deliberately, but never the isolated process.
        # This checks the worker's separate import gate and bounded module views.
        for body, expected in (
            ("import os\nreturn None", "ImportError"),
            ("import random\nreturn None", "ImportError"),
            ("import json.tool\nreturn None", "ImportError"),
            ("from re import search\nreturn None", "ImportError"),
            ("from . import json\nreturn None", "ImportError"),
            ('return __import__("subprocess")', "ImportError"),
            ("import re\nreturn re.compile('x')", "AttributeError"),
            ("import re\nreturn re.DEBUG", "AttributeError"),
            ("import json\nreturn json.load(None)", "AttributeError"),
            ("import json\nreturn json.JSONDecoder()", "AttributeError"),
        ):
            source = ("class CandidateHarness:\n    def build_prompt(self, observation):\n"
                      + "\n".join("        " + line for line in body.splitlines()) + "\n")
            request = {"source": source, "hook": "build_prompt", "observation": {},
                       "builtins": sorted(harness._BUILTINS)}
            with self.subTest(body=body):
                completed = subprocess.run(
                    harness._sandbox_command(), input=json.dumps(request).encode(),
                    capture_output=True, check=True, timeout=15,
                )
                self.assertTrue(completed.stdout.startswith(harness._READY))
                result = json.loads(completed.stdout[len(harness._READY):])
                self.assertEqual(result["status"], "candidate_error")
                self.assertIn(expected, result["message"])

    def test_real_runtime_errors_and_non_json_results_are_candidate_errors(self):
        bodies = [
            'return observation["missing"]',
            "return 1 / 0",
            "return x\nx = 1",
            "return {1, 2}",
            'return {1: "bad key"}',
            'return float("inf")',
            "return self",
            "return str",
            'return "x" * 70000',
            "result = []\nresult.append(result)\nreturn result",
            "return [None] * 10001",
        ]
        for body in bodies:
            with self.subTest(body=body):
                with self.assertRaises(harness.HookExecutionError):
                    harness.execute_hook(body_spec(body), "build_prompt", {})

    def test_real_observation_cannot_inject_program_source(self):
        value = "'; __import__('os').system('false'); #"
        result = harness.execute_hook(
            body_spec('return observation["question"]'), "build_prompt", {"question": value},
        )
        self.assertEqual(result, value)

    def test_real_memory_limit_is_candidate_error(self):
        with self.assertRaises(harness.HookExecutionError):
            harness.execute_hook(body_spec('return "x" * (1024 ** 3)'), "build_prompt", {})

    def test_real_cpu_limit_is_candidate_error(self):
        with patch.object(harness, "CPU_SECONDS", 1), patch.object(harness, "WALL_SECONDS", 5):
            with self.assertRaisesRegex(harness.HookExecutionError, "CPU limit"):
                harness.execute_hook(body_spec("while True:\n    pass"), "build_prompt", {})

    def test_real_regex_backtracking_obeys_cpu_limit(self):
        candidate = body_spec('import re\nreturn bool(re.search(r"(a+)+$", observation["text"]))')
        with patch.object(harness, "CPU_SECONDS", 1), patch.object(harness, "WALL_SECONDS", 5):
            with self.assertRaisesRegex(harness.HookExecutionError, "CPU limit"):
                harness.execute_hook(candidate, "build_prompt", {"text": "a" * 100000 + "!"})

    def test_real_isolation_and_limits(self):
        # Trusted test probe, NOT a candidate: inspect the same sandbox command.
        probe = '''import json, os, resource, socket
from pathlib import Path
print(json.dumps({
    "uid": os.getuid(), "gid": os.getgid(), "cwd": os.getcwd(),
    "env": dict(os.environ), "interfaces": socket.if_nameindex(),
    "private_paths": [p for p in ("/mnt", "/home", "/opt/math") if Path(p).exists()],
    "gpus": [str(p) for p in Path("/dev").glob("nvidia*")],
    "cpu": resource.getrlimit(resource.RLIMIT_CPU),
    "memory": resource.getrlimit(resource.RLIMIT_AS),
    "core": resource.getrlimit(resource.RLIMIT_CORE),
    "caps": [line for line in Path("/proc/self/status").read_text().splitlines()
             if line.startswith("CapEff:")],
}))
'''
        command = harness._sandbox_command()
        command[-1] = probe
        with patch.dict(os.environ, {"SAO_TEST_SECRET": "not-for-candidates"}):
            completed = subprocess.run(command, capture_output=True, timeout=15, check=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["uid"], os.getuid())
        self.assertEqual(result["gid"], os.getgid())
        self.assertNotEqual(result["uid"], 0)
        self.assertEqual(result["cwd"], "/work")
        self.assertNotIn("SAO_TEST_SECRET", result["env"])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", result["env"])
        self.assertEqual(result["private_paths"], [])
        self.assertEqual(result["gpus"], [])
        self.assertEqual(result["interfaces"], [[1, "lo"]])
        self.assertEqual(result["cpu"], [10, 11])
        self.assertEqual(result["memory"], [harness.MEMORY_BYTES] * 2)
        self.assertEqual(result["core"], [0, 0])
        self.assertEqual(result["caps"], ["CapEff:\t0000000000000000"])


class SavedChessPilotTests(unittest.TestCase):
    """Read-only regression over the entire stopped pilot, when locally present."""

    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1] / "runs/searchqa_chess_campaign_20260929"
        branches = ("node4/chess/sao", "node3/chess/ma_sao")
        directories = [root / branch / "blocks/0001/h1/receipts" for branch in branches]
        if not all(directory.is_dir() for directory in directories):
            raise unittest.SkipTest("Saved Chess H1 pilot receipts are not present")
        cls.rows = []
        for branch, directory in zip(branches, directories):
            for path in sorted(directory.glob("*.json")):
                raw = path.read_bytes()
                cls.rows.append((branch, path, json.loads(raw)["action_text"],
                                 hashlib.sha256(raw).hexdigest()))
        cls.attributes = {
            "MAX_TURNS", "SYSTEM_PROMPT", "FORMAT_RETRY_BUDGET", "ILLEGAL_MOVE_RETRY_BUDGET",
        }
        cls.hooks = {"format_observation", "parse_action", "recover_action"}

    def validated_rows(self):
        accepted, rejected = [], []
        for branch, path, source, digest in self.rows:
            try:
                candidate = harness.validate_program(
                    source, attributes=self.attributes, hooks=self.hooks,
                )
            except harness.ProgramValidationError as exc:
                rejected.append((branch, path.stem, str(exc)))
            else:
                accepted.append((branch, path, candidate, digest))
        return accepted, rejected

    def test_all_128_saved_programs_have_exactly_eight_forbidden_random_rejects(self):
        accepted, rejected = self.validated_rows()
        self.assertEqual(len(self.rows), 128)
        self.assertEqual(len(accepted), 120)
        self.assertEqual({(branch, identifier) for branch, identifier, _ in rejected}, {
            ("node4/chess/sao", "3fc3f26aeb384b7cbe7c788c4a2bd4bb"),
            ("node4/chess/sao", "8b0038d3229d42408e8bb6275d803d98"),
            ("node4/chess/sao", "a380068a4217434e8e9ab027568e6df9"),
            ("node3/chess/ma_sao", "0d352726c83a4fbc9544d5a016bba008"),
            ("node3/chess/ma_sao", "13523860e24647b8a462679bf1cdd4cd"),
            ("node3/chess/ma_sao", "266d4291f3224db9bf40adcf1d5fad23"),
            ("node3/chess/ma_sao", "a9a83f58653d4ff4946825f626347826"),
            ("node3/chess/ma_sao", "b321e06825b747e39de0346db395443a"),
        })
        for _, _, reason in rejected:
            self.assertIn("import random", reason)

    def test_every_accepted_saved_hook_executes_only_with_synthetic_public_data(self):
        accepted, rejected = self.validated_rows()
        self.assertEqual((len(accepted), len(rejected)), (120, 8))
        public = {
            "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "board_ascii": "r n b q k b n r\np p p p p p p p\n. . . . . . . .\n"
                           ". . . . . . . .\n. . . . . . . .\n. . . . . . . .\n"
                           "P P P P P P P P\nR N B Q K B N R",
            "side_to_move": "white",
            "legal_moves": [{"uci": "e2e4", "san": "e4"}, {"uci": "g1f3", "san": "Nf3"}],
            "last_opponent_move": None, "accepted_solver_moves": [],
            "solver_move_number": 1, "turn": 0, "turns_remaining": 5,
            "remaining_tokens": 2048, "response": "<move>e2e4</move>",
            "error": "illegal_move", "parsed_action": "a1a8", "retries_remaining": 1,
        }
        cases = [
            ("format_observation", {}),
            ("parse_action", {}),
            ("parse_action", {"response": "not a move"}),
            ("recover_action", {}),
            ("recover_action", {"error": "format_error", "parsed_action": None}),
        ]
        calls = 0
        parsed_moves = {}
        for branch, path, candidate, digest in accepted:
            self.assertEqual(set(candidate["_hooks"]), self.hooks)
            for hook, overrides in cases:
                with self.subTest(branch=branch, receipt=path.stem, hook=hook, overrides=overrides):
                    value = harness.execute_hook(candidate, hook, {**public, **overrides})
                    calls += 1
                    self.assertIsInstance(value, (str, type(None)))
                    if hook == "format_observation":
                        self.assertIsInstance(value, str)
                    if hook == "parse_action" and not overrides:
                        self.assertIn(value, ("e2e4", None))
                        parsed_moves[value] = parsed_moves.get(value, 0) + 1
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        self.assertEqual(calls, 600)
        # These 23 saved parsers have their own incorrect UCI predicates, e.g.
        # requiring the rank character to be a letter. Do not repair candidates.
        self.assertEqual(parsed_moves, {"e2e4": 97, None: 23})
        for _, path, _, digest in self.rows:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
