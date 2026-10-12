"""Native tool-call SearchQA, real fixed Wikipedia retrieval, and alias-EM reward."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
import re
import string
import time
from urllib.parse import urlsplit

import httpx

from . import program_harness
from .searchqa_prepare import ASSETS, question_key, sha256
from .searchqa_retrieval import CORPUS_SHA256, checked_query

METRIC = "normalized-answer-alias-exact-match-adaptation"
ATTRIBUTES = {"system_prompt", "max_steps", "max_tool_calls", "top_k",
              "max_query_chars", "document_char_limit"}
HOOKS = {"rewrite_query", "format_documents", "recovery"}
MAX_TOOL_OUTPUT_CHARS = 24000
INITIAL_SOURCE = '''class CandidateHarness:
    system_prompt = "Answer factual questions using Wikipedia search when needed."
    max_steps = 2
    max_tool_calls = 1
    top_k = 1
    max_query_chars = 512
    document_char_limit = 800
'''
EDITOR_SYSTEM = """Improve this reusable SearchQA harness. Return ONLY one Python
class CandidateHarness with no bases, decorators or module-level imports/code.
Only the hook-local imports and APIs described below are available.
Keep all six literal attributes in this initial structure:
""" + INITIAL_SOURCE + """
Allowed values: system_prompt is a nonempty string <=2000 characters; max_steps
2..8; max_tool_calls 1..8; top_k 1..5; max_query_chars 32..512;
document_char_limit 200..4000. Integer settings must be integers, not booleans.
Optionally implement any of these synchronous methods (self, observation):
rewrite_query: input {question,query,turn,previous_queries}; output {"query": str}.
format_documents: input {question,query,turn,documents,document_char_limit};
documents have {id,title,text,score}. Output {"documents":[{"id":str,"text":str}]}.
You may select, reorder and truncate actual documents, but each returned text
must be a nonempty substring of that id's original text; no invented passages.
recovery: input {question,turn,last_response,error,remaining_tokens};
output {"message": str} (nonempty, <=1600 characters).
Hooks receive no reference answers or dataset ids. No answer-specific rules,
verifier changes, file/network access, hidden state or retrieval URL changes.
The fixed full Wikipedia corpus, BM25 engine, five-document ceiling, and total
assistant token budget cannot change. Use the search function tool for retrieval.
The final answer must be inside <answer>...</answer>; no boxed-math format.
""" + program_harness.EDITOR_API_GUIDE
TOOLS = [{"type": "function", "function": {
    "name": "search",
    "description": "Search the fixed full 2018 English Wikipedia passage corpus.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                   "required": ["query"], "additionalProperties": False},
}}]


class RetrievalServiceError(RuntimeError):
    """A missing/broken trusted retriever is infrastructure failure, not a zero reward."""


def harness_spec(source: str) -> dict:
    values = program_harness.validate_program(source, attributes=ATTRIBUTES, hooks=HOOKS)
    ranges = {"max_steps": (2, 8), "max_tool_calls": (1, 8), "top_k": (1, 5),
              "max_query_chars": (32, 512), "document_char_limit": (200, 4000)}
    for key, (minimum, maximum) in ranges.items():
        if type(values[key]) is not int or not minimum <= values[key] <= maximum:
            raise ValueError(f"{key} must be an integer in {minimum}..{maximum}")
    if not isinstance(values["system_prompt"], str) or not 1 <= len(values["system_prompt"].strip()) <= 2000:
        raise ValueError("system_prompt must be a nonempty string of <=2000 characters")
    return values


def normalize_answer(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def extract_answer(text: str) -> str | None:
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if len(matches) != 1 or text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return None
    return matches[0].strip() or None


def alias_exact_match(text: str, answers: list[str]) -> float:
    answer = extract_answer(text)
    if answer is None:
        return 0.0
    normalized = normalize_answer(answer)
    return float(any(normalized == normalize_answer(alias) for alias in answers))


def load_splits(assets: Path = ASSETS) -> dict[str, list[dict]]:
    assets = Path(assets)
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest.get("metric") != METRIC:
        raise ValueError("Unexpected SearchQA verifier identity")
    splits = {}
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    for split in ("train", "reward", "dev", "test"):
        entry = manifest["splits"][split]
        path = assets / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen SearchQA split changed: {split}")
        rows = json.loads(path.read_text())
        if len(rows) != entry["count"]:
            raise ValueError("Split count mismatch")
        ids_digest = hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()
        if ids_digest != entry["ids_sha256"]:
            raise ValueError("Split ids changed")
        for row in rows:
            key = question_key(row["question"])
            if (not re.fullmatch(r"[0-9a-f]{64}", row["id"])
                    or row["id"] != hashlib.sha256(row["source_row_key"].encode()).hexdigest()):
                raise ValueError("SearchQA row id must be SHA256(source_row_key)")
            if row["id"] in seen_ids or key in seen_questions:
                raise ValueError("SearchQA split overlap")
            if not row["answers"] or row["answer"] != row["answers"][0]:
                raise ValueError("Invalid answer-alias compatibility")
            seen_ids.add(row["id"])
            seen_questions.add(key)
        splits[split] = rows
    return splits


def _retrieval_url() -> str:
    base = os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL", "").rstrip("/")
    parsed = urlsplit(base)
    if (parsed.scheme != "http" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path not in ("", "/retrieve")):
        raise RetrievalServiceError("Set SAO_SEARCHQA_RETRIEVAL_URL to the trusted HTTP service root or /retrieve endpoint")
    return base if parsed.path == "/retrieve" else base + "/retrieve"


async def retrieve_documents(query: str, top_k: int) -> list[dict]:
    url = _retrieval_url()
    manifest_path = ASSETS / "retrieval_manifest.json"
    if not manifest_path.is_file():
        raise RetrievalServiceError("Frozen retrieval_manifest.json is missing")
    identity = sha256(manifest_path)
    try:
        async with httpx.AsyncClient(timeout=30, trust_env=False, follow_redirects=False) as client:
            response = await client.post(url, json={"query": query, "top_k": top_k})
            response.raise_for_status()
            body = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise RetrievalServiceError(f"Trusted Wikipedia retrieval failed: {exc}") from exc
    if (not isinstance(body, dict) or body.get("corpus_sha256") != CORPUS_SHA256
            or body.get("index_identity") != identity):
        raise RetrievalServiceError("Retriever corpus/index identity does not match frozen local manifest")
    documents = body.get("documents")
    if not isinstance(documents, list) or len(documents) > top_k:
        raise RetrievalServiceError("Malformed retrieval document list")
    seen = set()
    for document in documents:
        if (not isinstance(document, dict) or set(document) != {"id", "title", "text", "score"}
                or not all(isinstance(document[key], str) for key in ("id", "title", "text"))
                or not 1 <= len(document["id"]) <= 128 or document["id"] in seen
                or not 0 < len(document["text"]) <= 4000
                or len(document["title"]) > 2000
                or type(document["score"]) not in (int, float)
                or not math.isfinite(document["score"])):
            raise RetrievalServiceError("Malformed retrieval passage")
        seen.add(document["id"])
    return documents


def _json_bounded_text(text: str, byte_limit: int) -> str:
    if len(json.dumps(text, ensure_ascii=True).encode()) <= byte_limit:
        return text
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if len(json.dumps(text[:middle], ensure_ascii=True).encode()) <= byte_limit:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def _public_hook_observation(observation: dict) -> dict:
    public = dict(observation)
    public["question"] = _json_bounded_text(public["question"], 16384)
    if "previous_queries" in public:
        public["previous_queries"] = list(public["previous_queries"][-8:])
    if "last_response" in public:
        public["last_response"] = _json_bounded_text(public["last_response"], 48000)
    if "documents" in public:
        public["documents"] = [
            {**document, "title": _json_bounded_text(document["title"], 1024),
             "text": _json_bounded_text(document["text"], 16000)}
            for document in public["documents"]
        ]
    if len(json.dumps(public, ensure_ascii=True, allow_nan=False).encode()) > 128000:
        raise program_harness.SandboxInfrastructureError("SearchQA public hook observation exceeds transport budget")
    return public


async def _hook(spec: dict, name: str, observation: dict):
    if name not in spec.get("_hooks", []):
        return None
    public = _public_hook_observation(observation)
    return await asyncio.to_thread(program_harness.execute_hook, spec, name, public)


def _rewritten_query(result: object, maximum: int) -> str:
    if not isinstance(result, dict) or set(result) != {"query"}:
        raise ValueError("rewrite_query must return exactly {query: string}")
    query = checked_query(result["query"])
    if len(query) > maximum:
        raise ValueError("Query exceeds harness query-character limit")
    return query


def _formatted_documents(result: object, documents: list[dict], limit: int) -> list[dict]:
    if not isinstance(result, dict) or set(result) != {"documents"} or not isinstance(result["documents"], list):
        raise ValueError("format_documents must return exactly {documents: list}")
    originals = {document["id"]: document for document in documents}
    selected = []
    used = set()
    for document in result["documents"]:
        if (not isinstance(document, dict) or set(document) != {"id", "text"}
                or not isinstance(document["id"], str) or not isinstance(document["text"], str)
                or document["id"] not in originals or document["id"] in used
                or not document["text"] or document["text"] not in originals[document["id"]]["text"]
                or len(document["text"]) > limit):
            raise ValueError("Formatted documents must be unique real ids with bounded verbatim passage substrings")
        used.add(document["id"])
        selected.append({**originals[document["id"]], "text": document["text"]})
    return selected


async def run_episode(row: dict, spec: dict, generate, total_tokens=4096, seed=42):
    from .retool import ContextExhausted

    if type(total_tokens) is not int or total_tokens <= 0:
        raise ValueError("total_tokens must be a positive integer")
    question = row["question"]
    messages = [
        {"role": "system", "content": spec["system_prompt"]},
        {"role": "user", "content": "Question: " + question
         + "\nUse the search function when needed. Put your final answer only inside <answer>...</answer>."},
    ]
    remaining = total_tokens
    records = []
    previous_queries = []
    harness_errors = []
    tool_count = tool_errors = output_chars = 0
    last_text = ""
    exhausted = False
    started = time.monotonic()
    for turn in range(spec["max_steps"]):
        cap = remaining if turn == spec["max_steps"] - 1 else min(remaining, max(128, total_tokens // 2))
        try:
            completion = await generate(messages, TOOLS, cap, seed + turn)
        except ContextExhausted:
            exhausted = True
            break
        used = completion["usage"]["completion_tokens"]
        if type(used) is not int or not 0 < used <= cap:
            raise RuntimeError("Invalid generation token accounting")
        remaining -= used
        if "record" in completion:
            records.append(completion["record"])
        message = dict(completion["choices"][0]["message"])
        last_text = message.get("content") or ""
        if not isinstance(last_text, str):
            raise RuntimeError("Assistant response content must be text")
        messages.append(message)
        tools = message.get("tool_calls") or []
        if not isinstance(tools, list) or len(tools) > 32:
            raise RuntimeError("Invalid assistant tool-call list")
        error = ""
        if tools:
            for tool in tools:
                tool_count += 1
                stage = "tool_arguments"
                try:
                    if harness_errors:
                        raise ValueError("Episode terminated by an invalid candidate hook")
                    if tool_count > spec["max_tool_calls"]:
                        raise ValueError("Episode search-call limit reached")
                    if output_chars >= MAX_TOOL_OUTPUT_CHARS:
                        raise ValueError("Episode retrieval-output budget reached")
                    function = tool["function"]
                    arguments = json.loads(function["arguments"])
                    if function["name"] != "search" or not isinstance(arguments, dict) or set(arguments) != {"query"}:
                        raise ValueError("Use search with exactly one query string")
                    query = _rewritten_query(arguments, spec["max_query_chars"])
                    stage = "rewrite_query"
                    rewritten = await _hook(spec, "rewrite_query", {
                        "question": question, "query": query, "turn": turn,
                        "previous_queries": list(previous_queries),
                    })
                    if "rewrite_query" in spec.get("_hooks", []):
                        query = _rewritten_query(rewritten, spec["max_query_chars"])
                    stage = "retrieval"
                    documents = await retrieve_documents(query, spec["top_k"])
                    previous_queries.append(query)
                    stage = "format_documents"
                    formatted = await _hook(spec, "format_documents", {
                        "question": question, "query": query, "turn": turn,
                        "documents": documents, "document_char_limit": spec["document_char_limit"],
                    })
                    if "format_documents" in spec.get("_hooks", []):
                        documents = _formatted_documents(formatted, documents, spec["document_char_limit"])
                    else:
                        documents = [{**document, "text": document["text"][:spec["document_char_limit"]]}
                                     for document in documents]
                    stage = "tool_output"
                    available = MAX_TOOL_OUTPUT_CHARS - output_chars
                    bounded = []
                    for document in documents:
                        if available <= 0:
                            break
                        text = document["text"][:available]
                        bounded.append({**document, "text": text})
                        available -= len(text)
                        output_chars += len(text)
                    result = {"query": query, "documents": bounded, "error": False}
                except (RetrievalServiceError, program_harness.SandboxInfrastructureError):
                    raise
                except (ValueError, TypeError, KeyError) as exc:
                    if (stage in ("retrieval", "tool_output")
                            or stage in HOOKS and stage not in spec.get("_hooks", [])):
                        raise
                    tool_errors += 1
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
                    if stage in HOOKS:
                        harness_errors.append({"hook": stage, "turn": turn, "error": error})
                    result = {"error": True, "stage": stage, "message": error}
                tool_id = tool.get("id") if isinstance(tool, dict) else None
                if not isinstance(tool_id, str) or not tool_id:
                    raise RuntimeError("Assistant tool call missing id")
                messages.append({"role": "tool", "tool_call_id": tool_id,
                                 "content": json.dumps(result, ensure_ascii=False)})
        elif extract_answer(last_text) is not None:
            break
        elif "recovery" not in spec.get("_hooks", []):
            break
        if harness_errors:
            break
        if remaining <= 0:
            exhausted = True
            break
        if (turn + 1 < spec["max_steps"] and (not tools or error)
                and "recovery" in spec.get("_hooks", [])):
            try:
                recovery = await _hook(spec, "recovery", {
                    "question": question, "turn": turn, "last_response": last_text,
                    "error": error or "missing_answer_format", "remaining_tokens": remaining,
                })
                if (not isinstance(recovery, dict) or set(recovery) != {"message"}
                        or not isinstance(recovery["message"], str)
                        or not 1 <= len(recovery["message"].strip()) <= 1600):
                    raise ValueError("recovery must return {message: nonempty string <=1600 chars}")
                retry = recovery["message"]
            except program_harness.SandboxInfrastructureError:
                raise
            except (ValueError, TypeError, KeyError) as exc:
                tool_errors += 1
                error = f"{type(exc).__name__}: {str(exc)[:500]}"
                harness_errors.append({"hook": "recovery", "turn": turn, "error": error})
                messages.append({"role": "user", "content": "Harness recovery failed; episode stopped: " + error})
                break
            messages.append({"role": "user", "content": retry})
    valid_format = extract_answer(last_text) is not None
    return {
        "id": row["id"], "reward": 0.0 if harness_errors else alias_exact_match(
            last_text, row["answers"] if "answers" in row else [row["answer"]]),
        "messages": messages, "calls": records, "tokens": total_tokens - remaining,
        "truncated": exhausted or remaining == 0 or not valid_format or bool(harness_errors),
        "boxed": valid_format, "formatted": valid_format, "answer_format": "answer-tags", "metric": METRIC,
        "tool_calls": tool_count, "tool_errors": tool_errors,
        "harness_errors": harness_errors,
        "elapsed_seconds": time.monotonic() - started,
    }
