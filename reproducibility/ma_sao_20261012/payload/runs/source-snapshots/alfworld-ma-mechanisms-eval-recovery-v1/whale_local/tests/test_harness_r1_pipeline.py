import hashlib
import json

import pytest

from harness_r1_searchqa.pipeline import merge_clean_silver, verify_grpo_commit, wait_for_silver
from sao_bridge.common import file_sha


def complete(directory, examples=256, usable=True):
    directory.mkdir()
    data = directory / "sft.jsonl"
    data.write_text("verified test data\n")
    receipt = {
        "dataset_sha256": file_sha(data), "examples": examples,
        "usable_for_sft": usable,
    }
    (directory / "COMPLETE.json").write_text(json.dumps(receipt))
    return receipt


def test_pipeline_requires_matching_dataset_hash(tmp_path):
    root = tmp_path / "silver"
    complete(root)
    (root / "sft.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="bytes"):
        wait_for_silver(root, "unit", 1)


def test_pipeline_rejects_insufficient_positive_data(tmp_path):
    root = tmp_path / "silver"
    complete(root, examples=255, usable=False)
    with pytest.raises(RuntimeError, match="Not enough"):
        wait_for_silver(root, "unit", 1)


def test_pipeline_accepts_only_complete_gate(tmp_path):
    root = tmp_path / "silver"
    expected = complete(root)
    assert wait_for_silver(root, "unit", 1) == expected


def test_pipeline_missing_receipt_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr("harness_r1_searchqa.pipeline.subprocess.check_output",
                        lambda *args, **kwargs: "failed\n")
    with pytest.raises(RuntimeError, match="without"):
        wait_for_silver(tmp_path, "unit", 1)


def grpo_receipt(root):
    checkpoint = root / "checkpoint-000001"
    (checkpoint / "actor").mkdir(parents=True)
    config = checkpoint / "actor" / "config.json"
    config.write_text("{}")
    weight = checkpoint / "actor" / "model.safetensors"
    weight.write_text("test weight bytes")
    files = {"config.json": file_sha(config), weight.name: file_sha(weight)}
    actor_sha = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    (checkpoint / "actor" / "actor_identity.json").write_text(json.dumps({
        "sha256": actor_sha, "files": files,
    }))
    states = {}
    for rank in range(4):
        state = checkpoint / f"rank-{rank:04d}.pt"
        state.write_text(f"rank {rank}")
        states[str(rank)] = file_sha(state)
    receipt = {
        "updates_completed": 1, "optimizer_steps": 4,
        "engine_reloaded_from_disk": True, "actor_sha256": actor_sha,
        "rank_state_sha256": states, "metrics_by_rank": [{
            "optimizer_steps": 4,
            "iterations": [{"loss": 0.0, "grad_norm": 1.0} for _ in range(4)],
        } for _ in range(4)],
    }
    (checkpoint / "COMMITTED.json").write_text(json.dumps(receipt))
    return checkpoint, receipt


def test_pipeline_verifies_real_commit_counts(tmp_path):
    _, receipt = grpo_receipt(tmp_path)
    assert verify_grpo_commit(tmp_path, 1) == receipt


@pytest.mark.parametrize("field,value", [
    ("optimizer_steps", 0),
    ("engine_reloaded_from_disk", False),
    ("metrics_by_rank", [{"loss": float("nan")}]),
])
def test_pipeline_rejects_success_shaped_grpo_receipt(tmp_path, field, value):
    checkpoint, receipt = grpo_receipt(tmp_path)
    receipt[field] = value
    (checkpoint / "COMMITTED.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):
        verify_grpo_commit(tmp_path, 1)


def silver_source(directory, count=256, seed=42):
    (directory / "samples").mkdir(parents=True)
    rows = []
    for index in range(count):
        identifier = f"packet-{index:06d}-sample-00"
        response = f"response {index}"
        sample = {
            "response": response, "seed": seed + index, "valid_patch": True,
            "finish_reason": "stop", "clean_execution": True,
            "reward": {"score": 0.125, "hook_errors": 0},
        }
        (directory / "samples" / f"{identifier}.json").write_text(json.dumps(sample))
        rows.append({
            "messages": [{"role": "user", "content": str(index)},
                         {"role": "assistant", "content": response}],
            "provenance": {"generation_id": identifier, "reward_delta": 0.125,
                           "split": "train", "engineer_manifest_sha256": "b" * 64,
                           "task_ids": [f"train-{index}"], "patch_sha256": "e" * 64},
        })
    data = directory / "sft.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    identity = {
        "data_sha256": "a" * 64, "engineer_manifest_sha256": "b" * 64,
        "reward_config_sha256": "c" * 64, "response_protocol": "prefill_think_patch",
        "temperature": 0.7, "top_p": 0.95, "max_response_tokens": 12288,
        "max_prompt_tokens": 20479, "seed": seed,
    }
    (directory / "COMPLETE.json").write_text(json.dumps({
        "examples": count, "dataset_sha256": file_sha(data), "input_identity": identity,
    }))


def test_merge_deduplicates_identical_teacher_messages(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    silver_source(first)
    silver_source(second, seed=1000042)
    receipt = merge_clean_silver([first, second], tmp_path / "merged")
    assert receipt["examples"] == 256
    assert receipt["usable_for_sft"] is True
    assert receipt["input_identity"]["engineer_manifest_sha256"] == "b" * 64
    row = json.loads((tmp_path / "merged" / "sft.jsonl").read_text().splitlines()[0])
    assert set(row["provenance"]) == {
        "generation_id", "engineer_manifest_sha256", "task_ids", "reward_delta",
        "patch_sha256", "split",
    }


def test_merge_rejects_positive_delta_with_hook_error(tmp_path):
    source = tmp_path / "source"
    silver_source(source)
    path = source / "samples" / "packet-000000-sample-00.json"
    sample = json.loads(path.read_text())
    sample["reward"]["hook_errors"] = 1
    path.write_text(json.dumps(sample))
    with pytest.raises(ValueError, match="clean positive"):
        merge_clean_silver([source], tmp_path / "merged")


def test_merge_does_not_invent_minimum_examples(tmp_path):
    source = tmp_path / "source"
    silver_source(source, count=2)
    with pytest.raises(RuntimeError, match="only 2"):
        merge_clean_silver([source], tmp_path / "merged")


def test_merge_rejects_different_frozen_source(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    silver_source(first)
    silver_source(second)
    path = second / "COMPLETE.json"
    receipt = json.loads(path.read_text())
    receipt["input_identity"]["data_sha256"] = "d" * 64
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="different frozen"):
        merge_clean_silver([first, second], tmp_path / "merged")
