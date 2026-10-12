import pytest

from sao_bridge.alfworld_matched_grpo_eval import choose_harness, wait_artifact


def test_selection_retains_invalid_slots_and_breaks_ties_by_index():
    proposals = [{"sample_index": i, "valid": i != 0} for i in range(8)]
    reports = {f"grpo-{i:02d}": {"accuracy": 0.5} for i in range(8)}
    reports["no_patch"] = {"accuracy": 0.9}
    assert choose_harness(reports, proposals) == "grpo-00"
    reports["grpo-07"]["accuracy"] = 0.8
    assert choose_harness(reports, proposals) == "grpo-07"
    with pytest.raises(ValueError, match="all eight"):
        choose_harness(reports, proposals[:-1])
    del reports["grpo-03"]
    with pytest.raises(ValueError, match="missing"):
        choose_harness(reports, proposals)


def test_failed_training_does_not_become_a_missing_result_timeout(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    (training / "FAILED.json").write_text('{"error": "GPU failure"}')
    with pytest.raises(RuntimeError, match="GPU failure"):
        wait_artifact(tmp_path / "never.json", training, tmp_path, timeout=1)


def test_existing_artifact_is_read_without_wait(tmp_path):
    path = tmp_path / "READY.json"
    path.write_text('{"ready": true}')
    assert wait_artifact(path, tmp_path, tmp_path) == {"ready": True}
