from collections import Counter
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party/sao/vendor/AReaL"))

from sao_bridge.alfworld_ma_mechanism_study import (
    ARMS, ROUNDS, arm_budgets, choose_harness, staged_assignments, staged_solver_rows,
)
from sao_bridge.alfworld_w_train import assignment_schedule_audit
from sao_bridge.common import write_json
from sao_bridge.joint_anchor import build_rows
from sao_bridge.joint_schedule import update_counts


def input_rows():
    order = list(range(240))
    random.Random(1000042).shuffle(order)
    return [{"id": f"source-{index}", "task_id": order[index % 240],
             "messages": [{"role": "user", "content": f"evidence-{order[index % 240] // 10}"}]}
            for index in range(768)]


def test_five_configs_one_fixed_primary_and_correct_update_counts():
    assert ARMS == ("full", "without_peer", "without_context", "without_value_norm", "always_joint")
    assert ROUNDS == (96, 144)
    spec = {"schedule": "joint_joint_anchor", "steps": 144, "critic_warmup_steps": 3,
            "critic_updates_per_step": 2}
    assert update_counts({**spec, "role": "editor"}, 144)["actor_steps"] == 93
    assert update_counts({**spec, "role": "editor"}, 144)["critic_steps"] == 192
    assert update_counts({**spec, "role": "solver"}, 144)["actor_steps"] == 141
    assert update_counts({**spec, "role": "solver"}, 144)["critic_steps"] == 288
    full, joint = arm_budgets("full"), arm_budgets("always_joint")
    assert full["shared_environment_episodes"] == joint["shared_environment_episodes"] == 1152
    assert full["generated_role_trajectories"] == joint["generated_role_trajectories"] == 2304
    assert full["roles"]["solver"] == joint["roles"]["solver"]
    assert full["roles"]["editor"]["consumed_trajectories"] == 768
    assert joint["roles"]["editor"]["consumed_trajectories"] == 1152
    assert joint["roles"]["editor"]["actor_steps"] == 141
    assert joint["roles"]["editor"]["critic_steps"] == 288
    with pytest.raises(ValueError, match="Unknown"):
        arm_budgets("without_team_learning")


def test_staged_solver_matches_every_task_count_without_active_worker_collisions(tmp_path):
    joint = build_rows(input_rows(), cycles=48, batch_size=8)
    staged = staged_solver_rows(joint)
    assert len(joint) == len(staged) == 1152
    assert Counter(row["task_id"] for row in joint) == Counter(row["task_id"] for row in staged)
    assert {row["source_joint_sample_id"] for row in staged} == {row["id"] for row in joint}
    data = tmp_path / "data.json"
    write_json(data, staged)
    audit = assignment_schedule_audit({"data": str(data), "steps": 144, "batch_size": 8, "staleness": 1})
    assert audit["required_rows"] == 1152 and audit["active_window"] == 16
    assignments = staged_assignments(staged, "chosen-harness")
    assert len(assignments) == 24
    assert all(len(row["task_ids"]) == 10 and row["harness"] == "chosen-harness" for row in assignments)


def test_impossible_population_order_fails_instead_of_repeating_or_dropping_tasks():
    with pytest.raises(ValueError, match="active-window"):
        staged_solver_rows([{"id": str(index), "task_id": 0} for index in range(20)])


def test_invalid_harness_cannot_masquerade_as_a_promoted_new_policy():
    values = [{"sample_index": i, "valid": i == 3} for i in range(8)]
    reports = {"incumbent_harness": {"accuracy": .8},
               **{f"new-proposal{i:02d}": {"accuracy": .95} for i in range(8)}}
    reports["new-proposal03"]["accuracy"] = .82
    assert choose_harness(values, reports, label="new", margin=2 / 48) == "incumbent_harness"
    reports["new-proposal03"]["accuracy"] = .9
    assert choose_harness(values, reports, label="new", margin=2 / 48) == "new-proposal03"
    values[3]["valid"] = False
    assert choose_harness(values, reports, label="new", margin=2 / 48) == "incumbent_harness"
    with pytest.raises(ValueError, match="exactly eight"):
        choose_harness(values[:-1], reports, label="new", margin=2 / 48)
