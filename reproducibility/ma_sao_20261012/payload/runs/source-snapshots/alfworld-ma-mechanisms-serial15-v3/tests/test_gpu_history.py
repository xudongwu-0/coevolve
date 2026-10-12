import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "whale_local"))

from sao_bridge import gpu_history as history


def row(index, *, time=0, state="idle", recorder="one"):
    timestamp = time + index * 300
    return {
        "schema": history.SCHEMA, "sequence": index + 1, "recorder_id": recorder,
        "recorded_at_unix": timestamp, "recorded_at_monotonic": timestamp,
        "recorded_at_beijing": history.beijing(timestamp),
        "nodes": {name: {"status": "ok", "classification": state} for name in history.HOSTS},
    }


def test_timestamps_and_daily_rotation_use_beijing(tmp_path):
    assert history.beijing(0) == "1970-01-01T08:00:00+08:00"
    sample = row(0, time=16 * 3600)
    path = history.append_sample(tmp_path, sample)
    assert path.name == "1970-01-02.jsonl"
    assert json.loads(path.read_text())["recorded_at_beijing"] == "1970-01-02T00:00:00+08:00"
    assert history.history_rows(tmp_path, 16 * 3600, 16 * 3600 + 1) == [sample]


def test_process_identity_records_programs_without_arguments_or_credentials():
    token = "hf_" + "A" * 34
    value = history.process_summary({
        "pid": 123, "uid": 1008,
        "cmd": ["/env/bin/python", "/work/train.py", "--token", token, "--url", "https://user:pass@example.invalid"],
    })
    serialized = json.dumps(value)
    assert value["executable"] == "python" and value["scripts"] == ["train.py"]
    assert token not in serialized and "user:pass" not in serialized and "--token" not in serialized
    assert history.safe_text("error " + token) == "error [REDACTED]"


def test_module_entrypoint_is_preserved_without_its_arguments():
    value = history.process_summary({"pid": 1, "uid": 1008, "cmd": ["python", "-m", "torch.distributed.run", "--nproc-per-node", "4"]})
    assert value["modules"] == ["torch.distributed.run"]
    assert "nproc" not in json.dumps(value)


def gpu_text():
    return "\n".join(f"{i}, GPU-{i}, NVIDIA H200, 0, 0, 143771, 79.5, 34" for i in range(8))


def test_gpu_parser_preserves_eight_gpu_metrics():
    gpus = history.gpu_rows(gpu_text())
    assert len(gpus) == 8 and gpus[7]["index"] == 7
    assert gpus[0]["memory_total_mib"] == 143771 and gpus[0]["power_watts"] == 79.5


@pytest.mark.parametrize("text", [
    "", gpu_text().replace("GPU-7", "GPU-0"), gpu_text().replace(", 0, 0,", ", 101, 0,", 1),
    gpu_text().replace(", 0, 0,", ", N/A, 0,", 1),
])
def test_invalid_gpu_identity_or_metrics_are_not_recorded_as_idle(text):
    with pytest.raises(ValueError):
        history.gpu_rows(text)


def node():
    return {
        "gpus": history.gpu_rows(gpu_text()), "unresolved_gpu_pids": [],
        "non_filler_gpu_pids": [], "distributed_launchers": [], "gpu_pids": [],
        "verified_filler_pids": [], "filler_unit_verified": False, "filler_kind": None,
    }


def test_unknown_memory_and_other_processes_are_not_filler():
    value = node()
    assert history.classification(value) == "idle"
    value["gpus"][0]["memory_used_mib"] = 500
    assert history.classification(value) == "unattributed_gpu_activity"
    value["gpu_pids"] = [7]
    assert history.classification(value) == "unverifiable"
    value.update(verified_filler_pids=[7], filler_unit_verified=True, filler_kind="adaptive")
    assert history.classification(value) == "verified_filler_only"
    value["non_filler_gpu_pids"] = [7]
    assert history.classification(value) == "other_workload"
    value["unresolved_gpu_pids"] = [7]
    assert history.classification(value) == "unverifiable"


def test_two_hours_require_25_samples_and_a_real_7200_second_span():
    rows = [row(index, state="idle" if index % 2 else "verified_filler_only") for index in range(25)]
    summary = history.summarize(rows, now=7201, hours=2)
    assert summary["required_samples"] == 25 and summary["meets_requested_window"]
    assert summary["trailing_idle_or_filler_seconds"] == 7200
    assert not history.summarize(rows[:-1], now=6901, hours=2)["meets_requested_window"]
    rows[-1]["recorded_at_unix"] = rows[-1]["recorded_at_monotonic"] = 7199
    assert not history.summarize(rows, now=7201, hours=2)["meets_requested_window"]


@pytest.mark.parametrize("change", ["busy", "error", "gap", "sequence", "restart", "clock"])
def test_historical_breaks_reset_the_trailing_eligible_window(change):
    rows = [row(index) for index in range(25)]
    target = rows[12]
    if change == "busy":
        target["nodes"]["node4"]["classification"] = "other_workload"
    elif change == "error":
        target["nodes"]["node3"].update(status="error", classification="probe_error")
    elif change == "gap":
        target["recorded_at_unix"] -= 61
        target["recorded_at_monotonic"] -= 61
    elif change == "sequence":
        target["sequence"] = 999
    elif change == "restart":
        target["recorder_id"] = "different-process"
    else:
        target["recorded_at_monotonic"] += 61
    summary = history.summarize(rows, now=7201, hours=2)
    assert not summary["meets_requested_window"]
    assert summary["trailing_idle_or_filler_samples"] < 25


def test_missing_stale_or_future_history_does_not_claim_current_availability():
    assert not history.summarize([], now=7200, hours=2)["meets_requested_window"]
    rows = [row(index) for index in range(25)]
    assert not history.summarize(rows, now=7561, hours=2)["latest_sample_fresh"]
    assert not history.summarize(rows, now=7199, hours=2)["latest_sample_fresh"]


@pytest.mark.parametrize("hours", [0, -1, float("inf"), float("nan")])
def test_invalid_query_window_fails_explicitly(hours):
    with pytest.raises(ValueError):
        history.summarize([], now=0, hours=hours)


def test_malformed_history_is_not_silently_skipped(tmp_path):
    directory = tmp_path / "samples"
    directory.mkdir()
    (directory / "1970-01-01.jsonl").write_text("{invalid\n")
    with pytest.raises(ValueError, match="Malformed GPU history"):
        history.history_rows(tmp_path, 0, 1000)


def test_one_unreachable_node_is_an_explicit_error_without_losing_the_other(monkeypatch, capsys):
    def run(command, **kwargs):
        if command[0] == "ssh":
            raise subprocess.TimeoutExpired(command, 90)
        value = node()
        value.update(host=history.HOSTS["node4"], observed_at_unix=history.time.time(), classification="idle", status="ok")
        return SimpleNamespace(stdout=json.dumps(value))
    monkeypatch.setattr(history.subprocess, "run", run)
    result = history.collect_nodes()
    assert result["node4"]["status"] == "ok"
    assert result["node3"]["status"] == "error"
    assert result["node3"]["classification"] == "probe_error"
    assert "PROBE ERROR node3" in capsys.readouterr().err


def test_five_minute_schedule_does_not_drift_with_probe_duration_or_catch_up_in_bursts():
    assert history.INTERVAL == 300
    assert history.next_sample_time(1000, 1004) == 1300
    assert history.next_sample_time(1300, 1320) == 1600
    assert history.next_sample_time(1300, 1700) == 2000


def test_probe_preserves_attribution_and_does_not_call_training_tools(monkeypatch):
    monkeypatch.setattr(history.os, "uname", lambda: SimpleNamespace(nodename=history.HOSTS["node4"]))
    record = {"pid": 42, "uid": 1008, "cmd": ["python", "/work/train_stage3.py"], "cgroup": "ordinary"}
    monkeypatch.setattr(history, "audit", lambda: {
        "verified_unit": False, "filler_kind": None, "vanished_gpu_pids": [],
        "processes": [{**record, "verified_filler": False}],
    })
    monkeypatch.setattr(history, "process_identity", lambda pid: record)
    monkeypatch.setattr(history, "launcher_inventory", lambda: [])
    calls = []
    def query(command):
        calls.append(command)
        if command[1].startswith("--query-gpu"):
            return gpu_text()
        if command[1].startswith("--query-compute-apps"):
            return "GPU-0, 42, 8000\n"
        assert command == ["nvidia-smi", "pmon", "-s", "um", "-c", "1"]
        return "0 42 C 35 10 0 0 0 0 python\n"
    monkeypatch.setattr(history, "query", query)
    result = history.probe_node()
    assert result["classification"] == "other_workload"
    assert result["processes"][0]["scripts"] == ["train_stage3.py"]
    assert result["processes"][0]["gpu_allocations"] == [{"gpu": 0, "pid": 42, "memory_mib": 8000}]
    assert result["processes"][0]["utilization_samples"][0]["sm_percent"] == 35
    assert all(command[0] == "nvidia-smi" for command in calls)


def test_launchers_with_no_gpu_memory_still_prevent_idle(monkeypatch):
    value = node()
    value["distributed_launchers"] = [{"pid": 99, "modules": ["torch.distributed.run"]}]
    assert history.classification(value) == "other_workload"
