from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.common import write_json, file_sha
from sao_bridge.validity_credit_study import audit_training, launch_unit, copy_initial_reports, wait_files


def fixture(root):
    for role in ("editor", "solver"):
        metrics = root / role / "optimizer/metrics.jsonl"
        metrics.parent.mkdir(parents=True)
        metrics.write_text('{"global_step": 0}\n')
        write_json(root / role / "COMPLETE.json", {
            "status": "completed", "steps": 1, "actor_updates": 1, "critic_updates": 2,
            "metrics_sha256": file_sha(metrics)})
        write_json(root / role / "RELEASED.json", {"foreign_gpu_pids": [], "filler_unit_active": True})
        write_json(root / f"{role}_spec.json", {"editor_reward_mode": "validity_gated"})
        for slot in range(8):
            row = {"version": 0, "slot": slot, "sample_id": str(slot),
                   "team_reward": 1, "model_error": "invalid", "reward": int(role == "solver"),
                   "editor_reward_mode": "validity_gated"}
            write_json(root / f"joint/round-0000/{role}-receipt-{slot:03d}.json", row)
            write_json(root / f"joint/round-0000/outcome-{slot:03d}.json", row)


def test_audit_real_counts_and_credit_instead_of_completion_label(tmp_path):
    fixture(tmp_path)
    assert audit_training(tmp_path, 1, 0)["editor"]["invalid_team_success"] == 8
    metrics = tmp_path / "editor/optimizer/metrics.jsonl"
    metrics.write_text('{"global_step": 99}\n')
    with pytest.raises(ValueError, match="metrics mismatch"):
        audit_training(tmp_path, 1, 0)


def test_extra_prefetched_round_is_not_a_successful_budget(tmp_path):
    fixture(tmp_path)
    write_json(tmp_path / 'joint/round-0001/editor-receipt-000.json',
               {"version": 1, "slot": 0, "sample_id": "extra"})
    with pytest.raises(ValueError, match="over budget"):
        audit_training(tmp_path, 1, 0)


def test_shared_success_cannot_leak_back_into_invalid_editor_reward(tmp_path):
    fixture(tmp_path)
    path = tmp_path / 'joint/round-0000/editor-receipt-000.json'
    import json
    row = json.loads(path.read_text()); row['reward'] = 1
    write_json(path, row)
    with pytest.raises(ValueError, match="credit differs"):
        audit_training(tmp_path, 1, 0)


def test_evaluation_launch_uses_existing_cli_and_is_not_duplicated(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('sao_bridge.validity_credit_study.subprocess.run',
                        lambda args, **kwargs: calls.append(args))
    study = tmp_path / 'runs/study'
    source = tmp_path / 'source'
    launch_unit(study, source, study / 'control', 'evaluate')
    launch_unit(study, source, study / 'control', 'evaluate')
    assert len(calls) == 1
    assert calls[0][-4:] == ['--mode', 'evaluate', '--role', 'matched']


def test_reused_initialization_carries_full_reports(tmp_path):
    parent, arm = tmp_path / 'parent', tmp_path / 'arm'
    for variant in ('joint_harness', 'no_patch'):
        write_json(parent / f'initial_{variant}_report.json', {'results': [{'reward': 1}]})
    copy_initial_reports(parent, arm)
    for path in parent.glob('*.json'):
        assert file_sha(path) == file_sha(arm / path.name)


def test_recovery_only_resolves_exact_failure_and_pinned_evidence(tmp_path):
    failure = tmp_path / 'FAILED.json'
    evidence = tmp_path / 'recovered.json'
    write_json(failure, {'error': 'missing_summary_input'})
    write_json(evidence, {'status': 'audited'})
    write_json(tmp_path / 'RESOLVED_FAILURES.json', {
        'failures': {str(failure.resolve()): file_sha(failure)},
        'evidence': {str(evidence): file_sha(evidence)},
    })
    # An accepted recovery keeps waiting; it does not fabricate completion.
    with pytest.raises(TimeoutError):
        wait_files(tmp_path, ['not_completed.json'], timeout=-1)
    write_json(failure, {'error': 'new_failure'})
    with pytest.raises(RuntimeError, match='dependency failed'):
        wait_files(tmp_path, ['not_completed.json'], timeout=-1)
    write_json(evidence, {'status': 'changed'})
    with pytest.raises(ValueError, match='evidence changed'):
        wait_files(tmp_path, ['not_completed.json'], timeout=-1)
