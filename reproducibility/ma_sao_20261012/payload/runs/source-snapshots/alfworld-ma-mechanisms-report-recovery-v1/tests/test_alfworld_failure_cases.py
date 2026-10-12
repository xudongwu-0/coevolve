from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))

from sao_bridge.alfworld_failure_cases import action_trace, case_markdown, failure_categories, paired_cases


def row(task, reward):
    messages = [{"role": "user", "content": "Your task is to: put the apple on the table."}]
    for _ in range(4 if not reward else 1):
        messages += [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "take_action", "arguments": '{"action":"put apple 1 in/on table 1"}'}}
            ]},
            {"role": "tool", "content": "Nothing happens." if not reward else "Task completed."},
        ]
    return {
        "id": f"alfworld-{task}", "task_id": task, "sample_index": 0,
        "rollout_seed": 42 + task, "reward": reward, "tokens": 100,
        "messages": messages, "status": "completed",
    }


def test_reverse_pairs_and_initial_regressions_are_not_hidden():
    control = {"results": [row(1, 0), row(2, 0), row(3, 1), row(4, 0), row(5, 1)]}
    full = {"results": [row(1, 1), row(2, 1), row(3, 0), row(4, 0), row(5, 1)]}
    initial = {"results": [row(1, 0), row(2, 1), row(3, 1), row(4, 0), row(5, 1)]}
    result = paired_cases(control, full, initial=initial, limit=1)
    assert result["control_failure_full_success_ids"] == [1, 2]
    assert result["control_success_full_failure_ids"] == [3]
    assert result["both_fail"] == result["both_succeed"] == 1
    assert result["examples"][0]["task_id"] == 2
    assert result["category_population_counts"]["repeated_action_at_least_four_times"] == 2
    assert result["all_control_failure_category_counts"]["repeated_action_at_least_four_times"] == 3
    assert result["all_full_ma_failure_category_counts"]["repeated_action_at_least_four_times"] == 2
    text = case_markdown(result, title="Actual cases")
    assert "Nothing happens." in text and "Task completed." in text
    assert "Control succeeds, full MA fails | 1" in text


def test_absent_contrast_is_reported_without_a_fabricated_case():
    report = {"results": [row(1, 1)]}
    result = paired_cases(report, report)
    assert not result["examples"]
    assert "No control-failure" in case_markdown(result, title="No contrast")


@pytest.mark.parametrize("fault", ["seed", "task", "duplicate", "messages"])
def test_case_identity_and_original_trace_are_mandatory(fault):
    control = {"results": [row(1, 0)]}
    full = {"results": [row(1, 1)]}
    if fault == "seed":
        full["results"][0]["rollout_seed"] += 1
    elif fault == "task":
        full["results"][0]["task_id"] = 2
    elif fault == "duplicate":
        control["results"].append(control["results"][0])
    else:
        control["results"][0].pop("messages")
    with pytest.raises(ValueError):
        paired_cases(control, full)


def test_invalid_arguments_and_token_exhaustion_are_observable_not_inferred_goals():
    value = row(1, 0)
    value["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"action":'
    value["truncated"] = True
    categories = failure_categories(value)
    assert "invalid_tool_arguments" in categories and "token_budget_exhaustion" in categories
    assert action_trace(value)[0]["actions"][0]["raw_arguments"] == '{"action":'
