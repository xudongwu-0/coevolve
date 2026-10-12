import argparse
import asyncio
import json

import httpx
import pytest

from harness_r1_searchqa import bootstrap


def setup_run(tmp_path, monkeypatch, failures):
    data = tmp_path / "packets.jsonl"
    data.write_text(json.dumps({
        "prompt": [{"role": "user", "content": "edit"}],
        "metadata": {"task_ids": ["train-1"]},
    }) + "\n")
    manifest = tmp_path / "engineer.json"
    manifest.write_text("{}")
    config = tmp_path / "reward.yaml"
    config.write_text("{}")
    output = tmp_path / "output"

    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"count": 10, "choices": [{
                "message": {"content": "response"}, "finish_reason": "stop",
            }], "usage": {"completion_tokens": 10}}

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, *args, **kwargs):
            return Response()

        async def post(self, *args, **kwargs):
            return Response()

    attempts = []
    posts = []

    async def reward(*args):
        attempts.append(True)
        if len(attempts) <= len(failures):
            raise failures[len(attempts) - 1]
        return {"eval_status": "ok", "score": 0.125, "hook_errors": 0}

    async def sleep(seconds):
        pass

    monkeypatch.setattr(bootstrap.httpx, "AsyncClient", lambda **kwargs: Client())
    monkeypatch.setattr(bootstrap, "reward_func", reward)
    monkeypatch.setattr(bootstrap, "_validate_metadata", lambda *args: None)
    monkeypatch.setattr(bootstrap, "extract_and_normalize_patch", lambda response: {})
    monkeypatch.setattr(bootstrap, "patch_digest", lambda patch: "a" * 64)
    monkeypatch.setattr(bootstrap.asyncio, "sleep", sleep)
    original_post = Client.post

    async def post(self, url, **kwargs):
        posts.append(url)
        return await original_post(self, url, **kwargs)

    monkeypatch.setattr(Client, "post", post)
    args = argparse.Namespace(
        data=data, engineer_urls="http://127.0.0.1:19001/v1",
        engineer_manifest=manifest, reward_config=config, output=output,
        packet_count=1, packet_start=0, samples=1, concurrency=1,
        reward_concurrency=1, minimum_examples=256, max_response_tokens=12288,
        max_prompt_tokens=20479, seed=42, generate_only=False, wait_for_data_seconds=0,
    )
    args.test_posts = posts
    return args, output, attempts


@pytest.mark.parametrize("error", [
    httpx.RemoteProtocolError("disconnected"),
    httpx.ReadError("connection reset"),
    httpx.WriteError("connection reset"),
])
def test_transient_target_disconnect_retries_without_zero_reward(tmp_path, monkeypatch, error):
    args, output, attempts = setup_run(
        tmp_path, monkeypatch, [error],
    )
    asyncio.run(bootstrap.run(args))
    assert len(attempts) == 2
    sample = json.loads((output / "samples" / "packet-000000-sample-00.json").read_text())
    assert sample["reward"]["score"] == 0.125
    errors = list((output / "infrastructure_errors").glob("*.json"))
    assert len(errors) == 1
    assert json.loads(errors[0].read_text())["reward_assigned"] is False


def test_exhausted_infrastructure_retries_raise_and_preserve_generation(tmp_path, monkeypatch):
    args, output, attempts = setup_run(
        tmp_path, monkeypatch, [httpx.RemoteProtocolError("disconnected") for _ in range(3)],
    )
    with pytest.raises(httpx.RemoteProtocolError):
        asyncio.run(bootstrap.run(args))
    assert len(attempts) == 3
    assert not (output / "COMPLETE.json").exists()
    sample = json.loads((output / "samples" / "packet-000000-sample-00.json").read_text())
    assert sample["status"] == "generated"
    assert "reward" not in sample


def test_provenance_error_is_never_retried_or_scored_zero(tmp_path, monkeypatch):
    args, output, attempts = setup_run(
        tmp_path, monkeypatch, [ValueError("wrong provenance")],
    )
    with pytest.raises(ValueError, match="provenance"):
        asyncio.run(bootstrap.run(args))
    assert len(attempts) == 1
    assert not (output / "infrastructure_errors").exists()
    assert not (output / "COMPLETE.json").exists()


def test_eight_identical_patches_use_one_tokenization_and_one_target_rerun(tmp_path, monkeypatch):
    args, output, attempts = setup_run(tmp_path, monkeypatch, [])
    args.samples = 8
    args.concurrency = 8
    asyncio.run(bootstrap.run(args))
    assert len(attempts) == 1
    assert sum(url.endswith("/tokenize") for url in args.test_posts) == 1
    assert sum(url.endswith("/chat/completions") for url in args.test_posts) == 8
    complete = json.loads((output / "COMPLETE.json").read_text())
    assert complete["current_invocation_unique_target_scores"] == 1
    assert complete["examples"] == 8
