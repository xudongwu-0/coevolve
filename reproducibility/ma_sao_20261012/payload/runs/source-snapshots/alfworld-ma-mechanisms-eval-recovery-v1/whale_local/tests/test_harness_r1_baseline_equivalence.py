import asyncio
import copy
import json

import pytest

from harness_r1_searchqa import episode
from sao_bridge import searchqa_domain


@pytest.mark.parametrize("mode", ["answer", "search", "no_answer", "context"])
def test_no_hooks_matches_existing_h0(monkeypatch, mode):
    requests = [[], []]

    async def retrieve(query, top_k):
        return [{"id": "doc1", "title": "title", "text": "real evidence", "score": 1.0}]

    monkeypatch.setattr(searchqa_domain, "retrieve_documents", retrieve)
    monkeypatch.setattr(episode, "retrieve_documents", retrieve)

    class Generator:
        def __init__(self, slot):
            self.slot = slot
            self.calls = 0

        async def __call__(self, messages, tools, cap, seed):
            requests[self.slot].append(copy.deepcopy((messages, tools, cap, seed)))
            self.calls += 1
            if mode == "context":
                raise episode.ContextExhausted("full")
            message = {"role": "assistant", "content": "<answer>Paris</answer>"}
            if mode == "search" and self.calls == 1:
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call1", "type": "function",
                        "function": {"name": "search", "arguments": json.dumps({"query": "France capital"})},
                    }],
                }
            elif mode == "no_answer":
                message["content"] = "I don't know."
            return {"usage": {"completion_tokens": 4}, "choices": [{"message": message}]}

    row = {"id": "train1", "question": "Capital of France?", "answers": ["Paris"]}
    spec = searchqa_domain.harness_spec(searchqa_domain.INITIAL_SOURCE)
    old = asyncio.run(searchqa_domain.run_episode(row, spec, Generator(0), total_tokens=64, seed=42))
    new = asyncio.run(episode.evaluate_one(
        row, sampler=Generator(1), patch=None, total_tokens=64, rollout_seed=42,
    ))
    assert requests[0] == requests[1]
    for key in ["reward", "messages", "tokens", "truncated", "formatted", "tool_calls", "tool_errors"]:
        assert old[key] == new[key]
