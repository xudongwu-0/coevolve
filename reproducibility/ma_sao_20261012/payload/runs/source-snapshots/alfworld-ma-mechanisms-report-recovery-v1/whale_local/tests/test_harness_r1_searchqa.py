from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_r1_searchqa import episode
from harness_r1_searchqa.data import build_rl_rows
from harness_r1_searchqa.identity import baseline_config
from harness_r1_searchqa.patch import (
    PatchValidationError,
    base_patch,
    compile_patch,
    extract_and_normalize_patch,
    merge_patches,
    patch_digest,
)
from harness_r1_searchqa.prompt import build_messages
from harness_r1_searchqa import reward as reward_module
from harness_r1_searchqa.reward import PROMPT_VERSION, REWARD_PROTOCOL
from harness_r1_searchqa.runtime import SearchQAHarnessRuntime
from sao_bridge.searchqa_prepare import sha256


def training_rows(data_root: Path) -> list[dict]:
    manifest = json.loads((data_root / "manifest.json").read_text())
    return json.loads((data_root / manifest["splits"]["train"]["file"]).read_text())


def configuration(data_root: Path, incumbent: dict) -> dict:
    return baseline_config(
        target_model_sha256="target-digest",
        target_temperature=1.0,
        total_tokens=4096,
        retrieval_manifest=data_root / "retrieval_manifest.json",
        data_manifest=data_root / "manifest.json",
        incumbent_patch=incumbent,
    )


def patch_with(hook: str, code: str) -> dict:
    return {
        "benchmark": "searchqa",
        "description": "Reusable query guidance.",
        "actions": [{"type": "add_code_hook", "hook": hook, "code": code}],
    }


def test_extracts_prefilled_think_patch() -> None:
    patch = patch_with(
        "make_pre_hint",
        'def hook(ctx, nb):\n    return {"message": "Use concise entity queries."}',
    )
    response = "Reasoning.\n</think><patch>" + json.dumps(patch) + "</patch>"
    normalized = extract_and_normalize_patch(response)
    assert normalized["schema_version"] == "harness-r1-patch-v1"
    assert normalized["actions"][0]["hook"] == "make_pre_hint"


def test_rejects_second_think_block_and_unsafe_import() -> None:
    safe = patch_with(
        "make_pre_hint",
        'def hook(ctx, nb):\n    return {"message": "Use concise queries."}',
    )
    response = "<think>again</think><patch>" + json.dumps(safe) + "</patch>"
    with pytest.raises(PatchValidationError, match="second"):
        extract_and_normalize_patch(response)
    unsafe = patch_with(
        "on_init",
        "import os\ndef hook(ctx, nb):\n    return None",
    )
    with pytest.raises(PatchValidationError, match="sandbox"):
        compile_patch(unsafe)


def test_merge_overrides_only_named_hook() -> None:
    incumbent = patch_with(
        "on_init",
        'def hook(ctx, nb):\n    return {"tool_hint": "Old init."}',
    )
    candidate = patch_with(
        "make_pre_hint",
        'def hook(ctx, nb):\n    return {"message": "New pre-hint."}',
    )
    merged = merge_patches(incumbent, candidate)
    assert {action["hook"] for action in merged["actions"]} == {
        "on_init",
        "make_pre_hint",
    }


def test_base_patch_compiles_without_hooks() -> None:
    compiled = compile_patch(base_patch())
    assert compiled["hooks"] == {}
    assert compiled["notebook"] == {}


def test_runtime_persists_notebook_across_hooks() -> None:
    patch = {
        "benchmark": "searchqa",
        "description": "Track query state.",
        "actions": [
            {
                "type": "add_code_hook",
                "hook": "on_init",
                "code": (
                    "def hook(ctx, nb):\n"
                    '    nb["count"] = 1\n'
                    '    return {"tool_hint": "Search exact entities."}'
                ),
            },
            {
                "type": "add_code_hook",
                "hook": "make_pre_hint",
                "code": (
                    "def hook(ctx, nb):\n"
                    '    nb["count"] = nb.get("count", 0) + 1\n'
                    '    return {"message": "Hint " + str(nb["count"])}'
                ),
            },
        ],
    }
    runtime = SearchQAHarnessRuntime(
        patch,
        question="Who wrote Hamlet?",
        max_steps=2,
        max_tool_calls=1,
        top_k=1,
        document_char_limit=800,
    )
    _, tool_hint = runtime.initialize()
    hint = runtime.pre_hint(
        step=0,
        remaining_steps=2,
        previous_queries=[],
        last_observation="",
    )
    assert tool_hint == "Search exact entities."
    assert hint == "Hint 2"
    assert runtime.notebook["count"] == 2


def test_failure_packet_rejects_labels_and_task_identity() -> None:
    with pytest.raises(ValueError, match="forbidden"):
        build_messages([{"question": "q", "answers": ["secret"]}])
    with pytest.raises(ValueError, match="identity"):
        build_messages([{"task_id": "train-1", "question": "q"}])


def test_episode_applies_query_rewrite_and_post_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSampler:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages, tools, max_tokens, seed):
            self.calls += 1
            if self.calls == 1:
                return {
                    "usage": {"completion_tokens": 10},
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": json.dumps(
                                                {"query": "Bad Query"}
                                            ),
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                }
            return {
                "usage": {"completion_tokens": 4},
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "<answer>Paris</answer>",
                        }
                    }
                ],
            }

    async def fake_retrieve(query, top_k):
        assert query == "france capital"
        return [
            {
                "id": "p1",
                "title": "Paris",
                "text": "Paris is the capital of France.",
                "score": 1.0,
            }
        ]

    monkeypatch.setattr(episode, "retrieve_documents", fake_retrieve)
    patch = {
        "benchmark": "searchqa",
        "description": "Rewrite broad queries and preserve state.",
        "actions": [
            {
                "type": "add_code_hook",
                "hook": "on_before_action",
                "code": (
                    "def hook(ctx, nb):\n"
                    '    nb["rewritten"] = True\n'
                    '    return {"kind": "rewrite_action", "action": "France capital"}'
                ),
            },
            {
                "type": "add_code_hook",
                "hook": "on_post_step",
                "code": (
                    "def hook(ctx, nb):\n"
                    '    nb["saw_result"] = bool(ctx.observation)\n'
                    '    return {"kind": "inject_hint", "message": "Use the retrieved evidence."}'
                ),
            },
        ],
    }
    result = asyncio.run(
        episode.evaluate_one(
            {"id": "train-1", "question": "What is France's capital?", "answers": ["Paris"]},
            sampler=FakeSampler(),
            patch=patch,
            total_tokens=64,
            rollout_seed=123,
        )
    )
    assert result["reward"] == 1
    assert result["calls"][0]["query"] == "france capital"
    assert result["notebook"] == {"rewritten": True, "saw_result": True}
    assert result["rollout_seed"] == 123


def test_reward_uses_cumulative_patch_and_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SAO_SEARCHQA_RETRIEVAL_URL", "http://127.0.0.1:8120")
    data_root = Path("/mnt/3fs/xudong/coevolve/assets/searchqa_sao_20260929")
    row = training_rows(data_root)[0]
    incumbent = base_patch()
    metadata = {
        "benchmark": "searchqa",
        "reward_protocol": REWARD_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "task_ids": [row["id"]],
        "rollout_seeds": [123],
        "baseline_rewards": {row["id"]: 0.0},
        "incumbent_patch": incumbent,
        "incumbent_patch_sha256": patch_digest(incumbent),
        "target_model_sha256": "target-digest",
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "batch_tag": "test-batch",
        "split": "train",
        "source_baseline_sha256": "a" * 64,
        "baseline_config": configuration(data_root, incumbent),
    }
    candidate = patch_with(
        "make_pre_hint",
        'def hook(ctx, nb):\n    return {"message": "Use concise entity queries."}',
    )
    sample = SimpleNamespace(
        metadata=metadata,
        response="Reasoning.</think><patch>"
        + json.dumps(candidate)
        + "</patch>",
    )
    args = SimpleNamespace(
        harness_r1_target_model_sha256="target-digest",
        harness_r1_searchqa_data=str(data_root),
        harness_r1_response_protocol="prefill_think_patch",
        harness_r1_eval_root=str(tmp_path),
        harness_r1_retrieval_url="http://127.0.0.1:8120",
        harness_r1_target_urls=["http://127.0.0.1:8110/v1"],
        harness_r1_rollout_concurrency=1,
        harness_r1_rollout_max_tokens=4096,
        harness_r1_target_temperature=1.0,
        reward_max_concurrency=1,
    )
    evaluations = 0

    async def fake_evaluate(rows, **kwargs):
        nonlocal evaluations
        evaluations += 1
        merged = kwargs["patch"]
        assert merged["actions"][0]["hook"] == "make_pre_hint"
        return {
            "results": [
                {"task_id": row["id"], "rollout_seed": 123, "reward": 1.0}
            ]
        }

    monkeypatch.setattr(reward_module, "evaluate_rows", fake_evaluate)
    first = asyncio.run(reward_module.reward_func(args, sample))
    second = asyncio.run(reward_module.reward_func(args, sample))
    assert first["score"] == 1.0
    assert second["score"] == 1.0
    assert first["valid_patch"] is True
    assert evaluations == 1


def test_reward_invalid_patch_is_zero_but_identity_failure_raises(tmp_path: Path) -> None:
    data_root = Path("/mnt/3fs/xudong/coevolve/assets/searchqa_sao_20260929")
    row = training_rows(data_root)[0]
    incumbent = base_patch()
    metadata = {
        "benchmark": "searchqa",
        "reward_protocol": REWARD_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "task_ids": [row["id"]],
        "rollout_seeds": [123],
        "baseline_rewards": {row["id"]: 0.0},
        "incumbent_patch": incumbent,
        "incumbent_patch_sha256": patch_digest(incumbent),
        "target_model_sha256": "target-digest",
        "data_manifest_sha256": sha256(data_root / "manifest.json"),
        "batch_tag": "invalid-patch",
        "split": "train",
        "source_baseline_sha256": "a" * 64,
        "baseline_config": configuration(data_root, incumbent),
    }
    args = SimpleNamespace(
        harness_r1_target_model_sha256="target-digest",
        harness_r1_searchqa_data=str(data_root),
        harness_r1_eval_root=str(tmp_path),
        harness_r1_target_urls=["http://127.0.0.1:8110/v1"],
        harness_r1_retrieval_url="http://127.0.0.1:8120",
        harness_r1_target_temperature=1.0,
        harness_r1_rollout_max_tokens=4096,
    )
    result = asyncio.run(
        reward_module.reward_func(
            args,
            SimpleNamespace(metadata=metadata, response="not a patch"),
        )
    )
    assert result["score"] == 0.0
    assert result["valid_patch"] is False
    bad_metadata = dict(metadata, target_model_sha256="wrong")
    with pytest.raises(ValueError, match="target model identity"):
        asyncio.run(
            reward_module.reward_func(
                args,
                SimpleNamespace(metadata=bad_metadata, response="not a patch"),
            )
        )


def test_failure_packet_keeps_identity_only_in_metadata() -> None:
    data_root = Path("/mnt/3fs/xudong/coevolve/assets/searchqa_sao_20260929")
    results = []
    for index, actual in enumerate(training_rows(data_root)[:8]):
        task_id = actual["id"]
        results.append(
            {
                "task_id": task_id,
                "question": actual["question"],
                "reward": 0.0,
                "rollout_seed": 100 + index,
                "messages": [
                    {"role": "system", "content": "Use search."},
                    {"role": "user", "content": actual["question"]},
                    {"role": "assistant", "content": "<answer>wrong</answer>"},
                ],
                "tool_errors": 0,
                "formatted": True,
            }
        )
    rows = build_rl_rows(
        {"results": results, "split": "train",
         "baseline_config": configuration(data_root, base_patch())},
        incumbent_patch=base_patch(),
        target_model_sha256="target-digest",
        data_manifest=data_root / "manifest.json",
        packet_size=8,
    )
    assert len(rows) == 1
    prompt_text = json.dumps(rows[0]["prompt"])
    for result in results:
        assert result["task_id"] not in prompt_text
    assert rows[0]["metadata"]["task_ids"] == [
        result["task_id"] for result in results
    ]
