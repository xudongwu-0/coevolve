import json

import pytest

from harness_r1_searchqa.evaluate import load_shared_baseline
from sao_bridge.common import file_sha


def receipt(directory):
    report = directory / "baseline.json"
    report.write_text(json.dumps({"results": [
        {"task_id": "a", "rollout_seed": 1}, {"task_id": "b", "rollout_seed": 2},
    ], "mean": 0.5}))
    (directory / "COMPLETE.json").write_text(json.dumps({
        "status": "baseline_only", "baseline_signature": {"model": "frozen"},
        "baseline_report_sha256": file_sha(report),
    }))


def test_cache_preserves_exact_task_seed_pairing(tmp_path):
    receipt(tmp_path)
    assert load_shared_baseline(
        tmp_path, {"model": "frozen"}, [{"id": "a"}, {"id": "b"}], [1, 2],
    )["mean"] == 0.5


def test_cache_rejects_changed_configuration(tmp_path):
    receipt(tmp_path)
    with pytest.raises(ValueError, match="identity"):
        load_shared_baseline(tmp_path, {"model": "changed"}, [{"id": "a"}, {"id": "b"}], [1, 2])


def test_cache_rejects_changed_bytes(tmp_path):
    receipt(tmp_path)
    (tmp_path / "baseline.json").write_text("{}")
    with pytest.raises(ValueError, match="bytes"):
        load_shared_baseline(tmp_path, {"model": "frozen"}, [{"id": "a"}, {"id": "b"}], [1, 2])


def test_cache_rejects_reordered_or_duplicate_pairs(tmp_path):
    receipt(tmp_path)
    with pytest.raises(ValueError, match="pairing"):
        load_shared_baseline(tmp_path, {"model": "frozen"}, [{"id": "b"}, {"id": "a"}], [2, 1])
