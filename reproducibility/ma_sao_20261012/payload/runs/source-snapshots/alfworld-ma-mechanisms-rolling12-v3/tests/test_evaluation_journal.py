import asyncio
from collections import Counter
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_solver_eval import reuse_completed_reports
from sao_bridge.budget import contract_digest
from sao_bridge.common import file_sha, read_json, write_json
from sao_bridge.evaluation import Sampler, evaluate, stable_id_seed_offset


def outcome(row):
    return {
        "id": row["id"], "task_id": row["task_id"], "reward": float(row["task_id"] == 240),
        "tokens": 8, "tool_calls": 1, "tool_errors": 0, "truncated": False, "boxed": True,
        "messages": [{"role": "user", "content": "A real task observation in this synthetic unit fixture."}],
    }


def test_episode_commit_survives_peer_transport_failure_without_replaying_success(tmp_path):
    calls = Counter()
    rows = [{"id": f"alfworld-{task}", "task_id": task} for task in (240, 241)]

    async def episode(row, spec, sampler, *, total_tokens, seed):
        calls[row["task_id"]] += 1
        if row["task_id"] == 241 and calls[241] == 1:
            raise httpx.ReadError("connection closed before a response")
        return outcome(row)

    domain = SimpleNamespace(harness_spec=lambda source: {}, run_episode=episode)
    with patch("sao_bridge.evaluation.get_domain", return_value=domain):
        with pytest.raises(httpx.ReadError):
            asyncio.run(evaluate(None, rows, "same-policy", seed=42, domain="alfworld",
                                 episode_concurrency=1, episode_journal=tmp_path))
        committed = tmp_path / "episode-000000.json"
        digest = file_sha(committed)
        result = asyncio.run(evaluate(None, rows, "same-policy", seed=42, domain="alfworld",
                                      episode_concurrency=1, episode_journal=tmp_path))
    assert calls == {240: 1, 241: 2}
    assert file_sha(committed) == digest
    assert len(read_json(tmp_path / "episode-000001.attempts.json")) == 2
    assert result["question_count"] == 2 and result["accuracy"] == .5
    assert [row["rollout_seed"] for row in result["results"]] == [
        42 + stable_id_seed_offset(row["id"]) for row in rows
    ]


def test_journal_rejects_changed_seeds_and_tampered_results(tmp_path):
    rows = [{"id": "alfworld-240", "task_id": 240}]

    async def episode(row, spec, sampler, **kwargs):
        return outcome(row)

    domain = SimpleNamespace(harness_spec=lambda source: {}, run_episode=episode)
    with patch("sao_bridge.evaluation.get_domain", return_value=domain):
        asyncio.run(evaluate(None, rows, "policy", episode_journal=tmp_path))
        with pytest.raises(ValueError, match="frozen"):
            asyncio.run(evaluate(None, rows, "policy", seed=43, episode_journal=tmp_path))
        path = tmp_path / "episode-000000.json"
        cached = read_json(path)
        cached["result"]["reward"] = 0.
        write_json(path, cached)
        with pytest.raises(ValueError, match="journal changed"):
            asyncio.run(evaluate(None, rows, "policy", episode_journal=tmp_path))
        cached["result"]["id"] = "different-task"
        cached["result_sha256"] = contract_digest(cached["result"])
        write_json(path, cached)
        with pytest.raises(ValueError, match="journal changed"):
            asyncio.run(evaluate(None, rows, "policy", episode_journal=tmp_path))


def test_sampler_does_not_reuse_idle_connections_or_retry_ambiguous_posts():
    async def check():
        with patch("sao_bridge.evaluation.httpx.AsyncClient") as client:
            Sampler("http://127.0.0.1:19801/v1")
            limits = client.call_args.kwargs["limits"]
            assert limits.max_keepalive_connections == 0 and limits.max_connections == 100
        sampler = Sampler("http://127.0.0.1:19801/v1")
        await sampler.client.aclose()
        requests = []

        def fail(request):
            requests.append(request)
            raise httpx.ReadError("incomplete response", request=request)

        sampler.client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
        try:
            with pytest.raises(httpx.ReadError):
                await sampler([], [], 4096, 42)
            assert len(requests) == 1 and sampler.requests == 0
        finally:
            await sampler.close()

    asyncio.run(check())


def partial_matrix(tmp_path):
    previous, current = tmp_path / "previous", tmp_path / "current"
    config = {
        "run": str(previous), "solver_model": "unchanged-w0", "solver_version": "w0",
        "task_ids": [240], "samples": 1, "seed": 42, "temperature": 1.,
        "solver_tokens": 4096,
    }
    write_json(previous / "config.json", config)
    write_json(previous / "FAILED.json", {"error": "ReadError('')"})
    complete = {}
    for identity, reward in (("first", 1.), ("second", 0.)):
        row = {**outcome({"id": "alfworld-240", "task_id": 240}), "reward": reward,
               "sample_index": 0, "rollout_seed": 42 + stable_id_seed_offset("alfworld-240")}
        report = {"accuracy": reward, "question_count": 1, "samples_per_question": 1,
                  "metric": "mean@1", "results": [row]}
        path = previous / "reports" / f"{identity}.json"
        write_json(path, report)
        complete[identity] = {"path": str(path), "sha256": file_sha(path)}
    manifest = {
        "previous_attempt": str(previous), "previous_config_sha256": file_sha(previous / "config.json"),
        "previous_failure_sha256": file_sha(previous / "FAILED.json"),
        "reason": "inference-http-read-error", "complete_reports": complete,
    }
    path = tmp_path / "resume.json"
    write_json(path, manifest)
    new_config = {**config, "run": str(current), "episode_checkpointing": True, "resume_matrix": str(path)}
    assignments = [{"harness_sha256": identity} for identity in complete]
    return new_config, current, assignments, manifest, path


def test_resume_keeps_all_completed_positive_and_negative_reports_exactly(tmp_path):
    config, current, assignments, manifest, _ = partial_matrix(tmp_path)
    with patch("sao_bridge.alfworld_strong_phase.checked_release"):
        reused = reuse_completed_reports(config, current, assignments)
    assert reused["first"]["accuracy"] == 1 and reused["second"]["accuracy"] == 0
    for identity, metadata in manifest["complete_reports"].items():
        assert file_sha(current / "reports" / f"{identity}.json") == metadata["sha256"]
    assert read_json(current / "REUSED_REPORTS.json")["reused_complete_episodes"] == 2


@pytest.mark.parametrize("fault", ["drop_negative", "seed", "model", "tokens", "report_bytes"])
def test_resume_rejects_outcome_selection_or_scientific_input_changes(tmp_path, fault):
    config, current, assignments, manifest, path = partial_matrix(tmp_path)
    if fault == "drop_negative":
        manifest["complete_reports"].pop("second")
        write_json(path, manifest)
    elif fault == "report_bytes":
        report_path = Path(manifest["complete_reports"]["first"]["path"])
        report_path.write_text("{}")
    else:
        config[{"seed": "seed", "model": "solver_model", "tokens": "solver_tokens"}[fault]] = "different"
    with patch("sao_bridge.alfworld_strong_phase.checked_release"), pytest.raises(ValueError):
        reuse_completed_reports(config, current, assignments)
