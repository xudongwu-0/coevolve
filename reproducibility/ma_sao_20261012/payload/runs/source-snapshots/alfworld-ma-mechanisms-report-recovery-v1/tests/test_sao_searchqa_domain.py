import asyncio
from collections import Counter
import copy
from http.server import BaseHTTPRequestHandler
import json
import os
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge import searchqa_domain as domain
from sao_bridge import searchqa_prepare as prepare
from sao_bridge import searchqa_retrieval as retrieval
from sao_bridge.retool import ContextExhausted


ROW = {"id": "row-1", "question": "What is France's capital?",
       "answers": ["Paris", "the city of Paris"], "answer": "Paris"}
DOCUMENTS = [{"id": "real-1", "title": "Paris", "text": "Paris is the capital of France.", "score": 4.2}]


def completion(text="", query=None, tokens=10, record=None):
    message = {"role": "assistant", "content": text}
    if query is not None:
        message["tool_calls"] = [{"id": "call-1", "type": "function",
                                  "function": {"name": "search", "arguments": json.dumps({"query": query})}}]
    result = {"choices": [{"message": message}], "usage": {"completion_tokens": tokens}}
    if record is not None:
        result["record"] = record
    return result


class SearchQATests(unittest.TestCase):
    def setUp(self):
        self.spec = domain.harness_spec(domain.INITIAL_SOURCE)

    def test_normalized_alias_em_and_required_format(self):
        self.assertEqual(domain.alias_exact_match("<answer>The City of PARIS!</answer>", ROW["answers"]), 1)
        self.assertEqual(domain.alias_exact_match("Paris", ROW["answers"]), 0)
        self.assertEqual(domain.alias_exact_match("<answer>Paris</answer><answer>Lyon</answer>", ROW["answers"]), 0)
        self.assertEqual(domain.alias_exact_match("<answer>A</answer>", ["A"]), 1)
        self.assertEqual(domain.alias_exact_match("<answer> </answer>", ["the"]), 0)
        self.assertEqual(domain.alias_exact_match("<answer>Paris France</answer>", ROW["answers"]), 0)

    def test_harness_contract_and_numeric_bounds(self):
        self.assertEqual(self.spec["max_steps"], 2)
        self.assertEqual(self.spec["max_tool_calls"], 1)
        self.assertEqual(self.spec["top_k"], 1)
        self.assertEqual(self.spec["document_char_limit"], 800)
        self.assertFalse(self.spec["_hooks"])
        for old, new in [("top_k = 1", "top_k = 6"), ("max_steps = 2", "max_steps = True"),
                         ("max_query_chars = 512", "max_query_chars = 513"),
                         ("document_char_limit = 800", "document_char_limit = 9999")]:
            with self.assertRaises(ValueError):
                domain.harness_spec(domain.INITIAL_SOURCE.replace(old, new))
        with self.assertRaises(ValueError):
            domain.harness_spec("import os\n" + domain.INITIAL_SOURCE)

    def test_native_tool_observation_and_all_records(self):
        seen = []
        replies = [completion(query="France capital", record={"record_id": 1}),
                   completion("<answer>Paris</answer>", record={})]
        async def generate(messages, tools, max_tokens, seed):
            seen.append(copy.deepcopy(messages))
            self.assertEqual(tools[0]["function"]["name"], "search")
            return replies.pop(0)
        with patch.object(domain, "retrieve_documents", new=AsyncMock(return_value=DOCUMENTS)) as search:
            with patch.object(domain.program_harness, "execute_hook") as hook:
                result = asyncio.run(domain.run_episode(ROW, self.spec, generate, total_tokens=100))
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["tokens"], 20)
        self.assertEqual(result["calls"], [{"record_id": 1}, {}])
        self.assertTrue(result["boxed"])
        self.assertTrue(result["formatted"])
        self.assertEqual(result["answer_format"], "answer-tags")
        self.assertEqual(result["tool_calls"], 1)
        self.assertEqual(seen[1][-1]["role"], "tool")
        self.assertIn("capital of France", seen[1][-1]["content"])
        search.assert_awaited_once_with("France capital", 1)
        hook.assert_not_called()
        for message in seen[0]:
            self.assertNotIn("city of Paris", message["content"])
            self.assertNotIn("row-1", message["content"])

    def test_real_executable_query_rewrite_hook(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def rewrite_query(self, observation):
        return {"query": observation["query"].strip() + " capital"}
""")
        replies = [completion(query="France"), completion("<answer>Paris</answer>")]
        async def generate(*args):
            return replies.pop(0)
        with patch.object(domain, "retrieve_documents", new=AsyncMock(return_value=DOCUMENTS)) as search:
            result = asyncio.run(domain.run_episode(ROW, spec, generate, total_tokens=100))
        search.assert_awaited_once_with("France capital", 1)
        self.assertEqual(result["reward"], 1)
        self.assertEqual(result["tool_errors"], 0)

    def test_hook_observation_has_no_answers_and_rejects_fabricated_evidence(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def format_documents(self, observation):
        return {"documents": []}
""")
        replies = [completion(query="France"), completion("<answer>Paris</answer>")]
        async def generate(*args):
            return replies.pop(0)
        def hook(specification, name, observation):
            self.assertEqual(name, "format_documents")
            self.assertEqual(set(observation), {"question", "query", "turn", "documents", "document_char_limit"})
            return {"documents": [{"id": "real-1", "text": "An invented oracle document"}]}
        with patch.object(domain, "retrieve_documents", new=AsyncMock(return_value=DOCUMENTS)):
            with patch.object(domain.program_harness, "execute_hook", side_effect=hook):
                result = asyncio.run(domain.run_episode(ROW, spec, generate, total_tokens=100))
        self.assertEqual(result["tool_errors"], 1)
        self.assertEqual(result["reward"], 0)
        self.assertEqual(result["harness_errors"][0]["hook"], "format_documents")
        self.assertNotIn("invented oracle", json.dumps(result["messages"]))

    def test_valid_document_selection_is_verbatim_and_bounded(self):
        result = domain._formatted_documents({"documents": [{"id": "real-1", "text": "capital of France"}]},
                                             DOCUMENTS, 200)
        self.assertEqual(result[0]["text"], "capital of France")
        for value in [
            {"documents": [{"id": "fake-id", "text": "Paris"}]},
            {"documents": [{"id": "real-1", "text": "Berlin"}]},
            {"documents": [{"id": "real-1", "text": "Paris"}] * 2},
            {"documents": [{"id": "real-1", "text": ""}]},
        ]:
            with self.assertRaises(ValueError):
                domain._formatted_documents(value, DOCUMENTS, 200)

    def test_unicode_hook_observations_fit_transport_without_fabrication(self):
        documents = [{"id": str(index) + "😀" * 127, "title": "😀" * 2000,
                      "text": "😀" * 4000, "score": 1.0} for index in range(5)]
        observations = [
            {"question": "😀" * 20000, "query": "😀" * 511 + "q", "turn": 7,
             "documents": documents, "document_char_limit": 4000},
            {"question": "😀" * 20000, "query": "😀" * 511 + "q", "turn": 7,
             "previous_queries": ["😀" * 511 + "q"] * 8},
            {"question": "😀" * 20000, "turn": 7, "last_response": "😀" * 20000,
             "error": "😀" * 500, "remaining_tokens": 4000},
        ]
        for observation in observations:
            bounded = domain._public_hook_observation(observation)
            self.assertLessEqual(len(json.dumps(bounded, ensure_ascii=True).encode()), 128000)
            wire = json.dumps(bounded, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
            self.assertLessEqual(len(wire), 128000)
            self.assertTrue(observation["question"].startswith(bounded["question"]))
            if "query" in observation:
                self.assertEqual(bounded["query"], observation["query"])
            if "documents" in observation:
                self.assertEqual(len(bounded["documents"]), 5)
                for original, clipped in zip(documents, bounded["documents"]):
                    self.assertEqual(original["id"], clipped["id"])
                    self.assertTrue(original["text"].startswith(clipped["text"]))
            if "previous_queries" in observation:
                self.assertEqual(bounded["previous_queries"], observation["previous_queries"])
                self.assertEqual(len(bounded["previous_queries"]), 8)
        self.assertEqual(len(documents[0]["text"]), 4000)

    def test_invalid_query_and_call_limit_do_not_retrieve(self):
        spec = {**self.spec, "max_steps": 3, "max_tool_calls": 1}
        replies = [completion(query=""), completion(query="France"), completion("<answer>Paris</answer>")]
        async def generate(*args):
            return replies.pop(0)
        with patch.object(domain, "retrieve_documents", new=AsyncMock()) as search:
            result = asyncio.run(domain.run_episode(ROW, spec, generate, total_tokens=100))
        search.assert_not_awaited()
        self.assertEqual(result["tool_calls"], 2)
        self.assertEqual(result["tool_errors"], 2)

    def test_infrastructure_failure_is_not_silently_zero_reward(self):
        async def generate(*args):
            return completion(query="France")
        with patch.object(domain, "retrieve_documents",
                          new=AsyncMock(side_effect=domain.RetrievalServiceError("offline"))):
            with self.assertRaises(domain.RetrievalServiceError):
                asyncio.run(domain.run_episode(ROW, self.spec, generate))
        with patch.dict(os.environ, {"SAO_SEARCHQA_RETRIEVAL_URL": ""}):
            with self.assertRaises(domain.RetrievalServiceError):
                domain._retrieval_url()

    def test_sandbox_infrastructure_failure_aborts(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def rewrite_query(self, observation):
        return {"query": observation["query"]}
""")
        async def generate(*args):
            return completion(query="France")
        with patch.object(domain.program_harness, "execute_hook",
                          side_effect=domain.program_harness.SandboxInfrastructureError("sandbox unavailable")):
            with self.assertRaises(domain.program_harness.SandboxInfrastructureError):
                asyncio.run(domain.run_episode(ROW, spec, generate))

    def test_invalid_recovery_stops_without_default_fallback(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def recovery(self, observation):
        return {"message": ""}
""")
        async def generate(*args):
            return completion("Need more information", record={})
        result = asyncio.run(domain.run_episode(ROW, spec, generate))
        self.assertEqual(result["reward"], 0)
        self.assertEqual(len(result["calls"]), 1)
        self.assertEqual(result["harness_errors"][0]["hook"], "recovery")
        self.assertIn("episode stopped", result["messages"][-1]["content"])

    def test_recovery_sandbox_and_unknown_runtime_failures_propagate(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def recovery(self, observation):
        return {"message": "Continue"}
""")
        async def generate(*args):
            return completion("Need more information")
        for exception in [domain.program_harness.SandboxInfrastructureError("sandbox offline"),
                          RuntimeError("unexpected runtime failure")]:
            with patch.object(domain.program_harness, "execute_hook", side_effect=exception):
                with self.assertRaises(type(exception)):
                    asyncio.run(domain.run_episode(ROW, spec, generate))

    def test_unexpected_retrieval_runtime_failure_propagates(self):
        async def generate(*args):
            return completion(query="France")
        with patch.object(domain, "retrieve_documents", new=AsyncMock(side_effect=RuntimeError("bug"))):
            with self.assertRaises(RuntimeError):
                asyncio.run(domain.run_episode(ROW, self.spec, generate))

    def test_total_assistant_token_budget_across_turns(self):
        spec = domain.harness_spec(domain.INITIAL_SOURCE + """
    def recovery(self, observation):
        return {"message": "Finish the answer"}
""")
        caps = []
        async def generate(messages, tools, max_tokens, seed):
            caps.append(max_tokens)
            return completion("unfinished", tokens=max_tokens)
        result = asyncio.run(domain.run_episode(ROW, spec, generate, total_tokens=400))
        self.assertEqual(caps, [200, 200])
        self.assertEqual(result["tokens"], 400)
        self.assertTrue(result["truncated"])
        self.assertFalse(result["boxed"])
        self.assertEqual(result["reward"], 0)

    def test_h0_stops_on_no_tool_without_optimized_retry(self):
        calls = []
        async def generate(*args):
            calls.append(1)
            return completion("Paris")
        result = asyncio.run(domain.run_episode(ROW, self.spec, generate))
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(result["messages"]), 3)
        self.assertEqual(result["reward"], 0)
        self.assertFalse(result["formatted"])

    def test_h0_returns_one_800_character_passage_and_unchanged_query(self):
        replies = [completion(query="France capital"), completion("<answer>Paris</answer>")]
        async def generate(*args):
            return replies.pop(0)
        documents = [{**DOCUMENTS[0], "text": "Paris " * 500}]
        with patch.object(domain, "retrieve_documents", new=AsyncMock(return_value=documents)) as search:
            result = asyncio.run(domain.run_episode(ROW, self.spec, generate))
        search.assert_awaited_once_with("France capital", 1)
        observation = json.loads(result["messages"][3]["content"])
        self.assertEqual(len(observation["documents"]), 1)
        self.assertEqual(len(observation["documents"][0]["text"]), 800)
        self.assertEqual(result["tool_calls"], 1)

    def test_context_exhaustion_preserves_actual_consumption_and_records(self):
        replies = [completion(query="France", record={"id": "kept"})]
        async def generate(*args):
            if replies:
                return replies.pop()
            raise ContextExhausted("context full")
        with patch.object(domain, "retrieve_documents", new=AsyncMock(return_value=DOCUMENTS)):
            result = asyncio.run(domain.run_episode(ROW, self.spec, generate, total_tokens=100))
        self.assertTrue(result["truncated"])
        self.assertEqual(result["tokens"], 10)
        self.assertEqual(result["calls"], [{"id": "kept"}])

    def test_retrieval_corpus_format_and_query_safety(self):
        self.assertEqual(retrieval.corpus_document({"id": 3, "contents": '"Title"\nPassage text'}),
                         {"id": "3", "title": "Title", "text": "Passage text"})
        for query in ("", "   ", "???", "x" * 513, "a\x00b"):
            with self.assertRaises(ValueError):
                retrieval.checked_query(query)

    def test_http_transport_handles_bursts_with_bounded_workers(self):
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    time.sleep(.01)
                    self.send_response(200)
                    self.send_header("Content-Length", "2")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(b"{}")
                finally:
                    with lock:
                        state["active"] -= 1
            def log_message(self, *args):
                pass
        server = retrieval.BoundedHTTPServer(("127.0.0.1", 0), Handler, workers=4, pending=64)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        async def burst():
            async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
                return await asyncio.gather(*(client.get(f"http://127.0.0.1:{server.server_port}") for _ in range(64)))
        try:
            responses = asyncio.run(burst())
            self.assertTrue(all(response.status_code == 200 for response in responses))
            self.assertLessEqual(state["peak"], 4)
            self.assertGreater(state["peak"], 1)
            self.assertEqual(server.stats()["rejected"], 0)
            self.assertEqual(server.stats()["connection_errors"], 0)
            self.assertGreaterEqual(server.request_queue_size, 512)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        server = retrieval.BoundedHTTPServer(("127.0.0.1", 0), Handler, workers=1, pending=1)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01})
        thread.start()
        try:
            responses = asyncio.run(burst())
            statuses = {response.status_code for response in responses}
            self.assertIn(503, statuses)
            self.assertTrue(statuses <= {200, 503})
            self.assertGreater(server.stats()["rejected"], 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    @unittest.skipUnless(os.environ.get("SAO_SEARCHQA_STRESS") == "1", "Opt in to full service concurrency regression")
    def test_live_full_corpus_concurrency_64_and_256(self):
        from sao_bridge.searchqa_retrieval_stress import stress
        for count in (64, 256):
            result = asyncio.run(stress(count))
            self.assertTrue(result["passed"], result)

    @unittest.skipUnless((prepare.ASSETS / "manifest.json").is_file(), "Run data preparation first")
    def test_frozen_real_data_manifest(self):
        splits = domain.load_splits()
        self.assertEqual({key: len(rows) for key, rows in splits.items()},
                         {"train": 18946, "reward": 128, "dev": 256, "test": 700})
        self.assertEqual(Counter(row["data_source"] for row in splits["test"]),
                         dict.fromkeys(prepare.EVAL_SOURCES, 100))
        self.assertEqual(Counter(row["data_source"] for row in splits["train"]),
                         {"nq": 9473, "hotpotqa": 9473})
        questions = [prepare.question_key(row["question"]) for rows in splits.values() for row in rows]
        self.assertEqual(len(questions), len(set(questions)))
        for rows in splits.values():
            self.assertTrue(all(row["answers"] and row["answer"] == row["answers"][0] for row in rows))
            self.assertTrue(all(len(row["id"]) == 64 and int(row["id"][:8], 16) >= 0 for row in rows))
        schemas = {tuple(sorted(row)) for rows in splits.values() for row in rows}
        self.assertEqual(len(schemas), 1)

    @unittest.skipUnless(os.environ.get("SAO_SEARCHQA_RETRIEVAL_URL"), "Set service URL for full-wiki18 smoke")
    def test_live_full_wikipedia_retrieval(self):
        documents = asyncio.run(domain.retrieve_documents("Marie Curie Nobel Prize", 3))
        self.assertTrue(documents)
        self.assertTrue(any("Curie" in document["text"] or "Curie" in document["title"] for document in documents))
        self.assertTrue(all(document["score"] > 0 for document in documents))
        identity = json.loads((prepare.ASSETS / "retrieval_manifest.json").read_text())
        self.assertGreater(identity["passages"], 20_000_000)


if __name__ == "__main__":
    unittest.main()
