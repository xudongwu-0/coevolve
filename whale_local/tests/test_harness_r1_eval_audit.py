import pytest

from harness_r1_eval_audit import require_complete_task_report


def report():
    return {
        "total": 8,
        "validation": {"running": 0, "unknown": 0, "task error": 0},
        "custom": {"overall": {"total": 8, "pass": 4, "wrong": 4}},
    }


def test_complete_paired_cohort_is_reportable():
    require_complete_task_report(report(), 8)


def test_success_shaped_zero_reward_does_not_hide_task_errors():
    value = report()
    value["validation"]["task error"] = 1.0
    value["custom"]["overall"].update({"pass": 0, "wrong": 8})
    with pytest.raises(ValueError, match="infrastructure-failed"):
        require_complete_task_report(value, 8)


def test_partial_or_missing_execution_is_not_reportable():
    with pytest.raises(ValueError, match="task count"):
        require_complete_task_report(report(), 32)
    value = report()
    del value["validation"]["task error"]
    with pytest.raises(ValueError, match="validation"):
        require_complete_task_report(value, 8)


def test_incomplete_outcome_counts_are_rejected():
    value = report()
    value["custom"]["overall"]["wrong"] = 3
    with pytest.raises(ValueError, match="outcome counts"):
        require_complete_task_report(value, 8)
