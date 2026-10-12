from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_ma_crossplay import combinations, wait_predecessor
from sao_bridge.common import write_json


def record(rounds, model, harness):
    return {"selection": {"rounds": rounds, "harness": harness, "pair": {"solver": {"actor": model}}}}


def test_crossplay_uses_saved_patches_without_searching_new_candidates():
    models, patches = combinations(record(0, "w0", "h0"), record(64, "w64", "h64"), record(112, "w112", "h112"))
    assert models == {"matched": "w64", "late": "w112"}
    assert patches == {"h0": "h0", "h_matched": "h64", "h_late": "h112"}
    assert len(models) * (len(patches) + 1) * 200 == 1600
    with pytest.raises(ValueError, match="identical"):
        combinations(record(0, "w0", "h0"), record(64, "same", "h64"), record(112, "same", "h112"))
    with pytest.raises(ValueError, match="later"):
        combinations(record(0, "w0", "h0"), record(64, "w64", "h64"), record(64, "w112", "h112"))


def test_failed_predecessor_is_not_silently_ignored_even_with_stale_completion(tmp_path):
    write_json(tmp_path / "COMPLETE.json", {"status": "completed"})
    write_json(tmp_path / "RELEASED.json", {})
    protocol = {"predecessor": str(tmp_path), "queue_timeout_seconds": 1}
    with patch("sao_bridge.alfworld_ma_crossplay.gpu_release_audit", return_value={
        "foreign_gpu_pids": [], "filler_unit_active": True,
    }):
        wait_predecessor(protocol)
    write_json(tmp_path / "FAILED.json", {"error": "failed"})
    with pytest.raises(RuntimeError, match="predecessor failed"):
        wait_predecessor(protocol)
