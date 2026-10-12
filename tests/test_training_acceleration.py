from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.common import file_sha, write_json
from sao_bridge.training_acceleration import overlay_path, relative_rms


def test_no_overlay_keeps_existing_training():
    assert overlay_path({}) is None


def test_overlay_rejects_missing_versions_or_packages(tmp_path):
    with pytest.raises(ValueError, match="explicit pinned"):
        overlay_path({"training_kernel_overlay": str(tmp_path)})
    with pytest.raises(ValueError, match="declared packages"):
        overlay_path({
            "training_kernel_overlay": str(tmp_path),
            "training_kernel_versions": {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "triton": "3.7.1"},
        })


def test_relative_rms_detects_shape_nonfinite_and_error():
    expected = torch.tensor([1., -2.])
    assert relative_rms(expected, expected) == 0.
    assert relative_rms(expected * 1.02, expected) == pytest.approx(.02, abs=1e-7)
    with pytest.raises(ValueError):
        relative_rms(torch.tensor([float("nan"), 1.]), expected)
    with pytest.raises(ValueError):
        relative_rms(torch.ones(3), expected)


def test_overlay_requires_hash_bound_success(tmp_path):
    from types import SimpleNamespace

    versions = {"flash-linear-attention": "0.5.2", "fla-core": "0.5.2", "triton": "3.7.1"}
    distributions = [SimpleNamespace(metadata={"Name": k}, version=v) for k, v in versions.items()]
    validation = tmp_path / "VALIDATION.json"
    write_json(validation, {"status": "passed"})
    spec = {
        "training_kernel_overlay": str(tmp_path), "training_kernel_versions": versions,
        "training_kernel_validation": str(validation),
        "pinned_files": {str(validation): file_sha(validation)},
    }
    with patch("importlib.metadata.distributions", return_value=distributions):
        assert overlay_path(spec) == tmp_path
        write_json(validation, {"status": "failed"})
        with pytest.raises(ValueError, match="not pinned"):
            overlay_path(spec)
        spec["pinned_files"][str(validation)] = file_sha(validation)
        with pytest.raises(ValueError, match="did not pass"):
            overlay_path(spec)


def test_stage_failure_is_not_success():
    from sao_bridge.training_speed_campaign import check_stage

    stage = {"unit": "fixture", "remote": False}
    for state, result, expected in (("active", "success", False), ("inactive", "success", True)):
        with patch("sao_bridge.training_speed_campaign.unit_state", return_value={
            "ActiveState": state, "Result": result,
        }):
            assert check_stage(stage) is expected
    with patch("sao_bridge.training_speed_campaign.unit_state", return_value={
        "ActiveState": "failed", "Result": "exit-code",
    }):
        with pytest.raises(RuntimeError, match="Stage failed"):
            check_stage(stage)


def test_native_comparison_requires_same_workload_and_real_speedup(tmp_path):
    import json
    from sao_bridge.training_acceleration import compare_native

    baseline, fast = tmp_path / "baseline", tmp_path / "fast"
    for root, duration in ((baseline, 10), (fast, 2)):
        root.mkdir()
        write_json(root / "result.json", {"status": "passed"})
        write_json(root / "scope.json", {"length": 32768})
        for rank in range(4):
            for role in ("reference", "proximal", "values"):
                torch.save([torch.ones(1, 4)], root / f"{role}-rank{rank}.pt")
            for role in ("actor", "critic"):
                events = []
                for step in range(2):
                    events += [{"event": "ppo_begin", "time": 20 * step},
                               {"event": "ppo_end", "time": 20 * step + duration}]
                (root / f"{role}-rank{rank}.jsonl").write_text(
                    "\n".join(json.dumps(row) for row in events))
    result = compare_native(baseline, fast, tmp_path / "result.json")
    assert result["optimizer_speedup"] == 5
    write_json(fast / "scope.json", {"length": 4096})
    with pytest.raises(ValueError, match="workloads differ"):
        compare_native(baseline, fast, tmp_path / "bad.json")
