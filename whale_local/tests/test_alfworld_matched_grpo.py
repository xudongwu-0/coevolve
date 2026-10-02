import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from harness_r1_searchqa.grpo_train import packet_batch, validate_engineer_scale
from sao_bridge.alfworld_matched_grpo import matched_rows, score_records, validate_reward_rows


def records():
    return [
        {"messages": [{"role": "user", "content": f"packet-{i}"}],
         "metadata_json": json.dumps({"batch": i})}
        for i in range(24) for _ in range(8)
    ]


def test_exact_192_prompt_multiset_and_ninety_six_boundary():
    rows = matched_rows(records())
    batches = [packet_batch(rows, i, 42) for i in range(6)]
    assert len({r["metadata"]["batch"] for b in batches for r in b}) == 24
    assert sum(len(b) * 8 for b in batches[:3]) == 96
    assert sum(len(b) * 8 for b in batches) == 192


def test_changed_multiplicity_or_metadata_rejected():
    data = records()
    with pytest.raises(ValueError, match="24 prompts"):
        matched_rows(data[:-1])
    changed = copy.deepcopy(data)
    changed[0]["metadata_json"] = '{"batch": 99}'
    with pytest.raises(ValueError, match="inconsistent"):
        matched_rows(changed)
    changed = copy.deepcopy(data)
    changed[0]["messages"] = changed[-1]["messages"]
    changed[0]["metadata_json"] = changed[-1]["metadata_json"]
    with pytest.raises(ValueError, match="24 prompts"):
        matched_rows(changed)


def metadata():
    return {
        "batch_tag": "baseline", "split": "test_set", "start": 0, "end": 10,
        "batch_size": 10, "baseline_pass": 5,
        "baseline_rewards": {str(i): float(i % 2) for i in range(10)},
        "reward_threshold": 1, "target_agent_name": "qwen35-9b-nothink",
        "target_model": "policy", "response_protocol": "prefill_think_patch",
        "solver_version": "w0",
    }


def reward_config():
    return {"solver_version": "w0", "excluded_eval_task_ids": list(range(240, 440)),
            "reward_url": "http://local-test"}


def test_exclusion_and_solver_are_enforced():
    row = {"metadata": metadata()}
    validate_reward_rows([row], reward_config())
    bad = reward_config()
    bad["excluded_eval_task_ids"] = [5]
    with pytest.raises(ValueError, match="held-out"):
        validate_reward_rows([row], bad)
    bad = reward_config()
    bad["solver_version"] = "w1"
    with pytest.raises(ValueError, match="Solver"):
        validate_reward_rows([row], bad)


@pytest.mark.parametrize("status,score,delta,raises", [
    ("ok", 0.2, 0.2, False),
    ("invalid_patch", 0.0, 0.0, False),
    ("eval_failed", 0.0, 0.0, True),
    ("ok", 0.2, 0.1, True),
    ("ok", float("nan"), float("nan"), True),
])
def test_official_delta_and_infrastructure_failures(monkeypatch, status, score, delta, raises):
    class Client:
        def __init__(self, **kwargs):
            assert kwargs["trust_env"] is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, json):
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"reward": score, "evaluation": {
                    "score": score, "delta_score": delta, "eval_status": status}},
            )

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    work = score_records([{"metadata": metadata(), "response": "patch"}], reward_config(), 42)
    if raises:
        with pytest.raises(RuntimeError):
            asyncio.run(work)
    else:
        assert asyncio.run(work)[0]["score"] == score


def test_vlm_requires_explicit_alfworld_opt_in():
    with pytest.raises(ValueError):
        validate_engineer_scale("qwen3_5", 9_000_000_000)
    validate_engineer_scale("qwen3_5", 9_000_000_000, allow_vlm=True)
    with pytest.raises(ValueError):
        validate_engineer_scale("qwen3_5", 4_000_000_000, allow_vlm=True)
