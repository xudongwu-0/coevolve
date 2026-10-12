import pytest

from harness_r1_searchqa.selection import paired_gain


def results(reward):
    return [
        {"task_id": f"q{i}", "rollout_seed": j, "reward": reward}
        for i in range(16) for j in range(8)
    ]


def test_positive_gain_accepted():
    evidence = paired_gain(results(0), results(1))
    assert evidence["absolute_gain"] == 1
    assert evidence["paired_bootstrap_ci95"] == [1, 1]
    assert evidence["accepted"] is True


def test_unchanged_harness_not_accepted():
    evidence = paired_gain(results(1), results(1))
    assert evidence["absolute_gain"] == 0
    assert evidence["accepted"] is False


def test_mismatch_rejected():
    after = results(1)
    after[0]["rollout_seed"] = 123456
    with pytest.raises(ValueError, match="identities"):
        paired_gain(results(0), after)


def test_duplicate_identity_rejected():
    before = results(0)
    before.append(before[0])
    with pytest.raises(ValueError, match="Duplicate"):
        paired_gain(before, results(1))
