from pathlib import Path

import pytest

from harness_r1_searchqa import rapid_screen


def test_screen_nested_questions_exclude_all_training_packet_ids(monkeypatch):
    rows = [{"id": f"task-{i:03d}", "dataset": str(i % 7)} for i in range(200)]
    monkeypatch.setattr(rapid_screen, "load_train_rows", lambda root: rows)
    excluded = {row["id"] for row in rows[:50]}
    small, _ = rapid_screen.select_heldout_train(Path("."), excluded, count=32)
    large, receipt = rapid_screen.select_heldout_train(Path("."), excluded, count=128)
    assert small == large[:32]
    assert not excluded.intersection(row["id"] for row in large)
    assert len({row["id"] for row in large}) == 128
    assert receipt["stratification_field"] == "dataset"
    assert len(receipt["selected_stratum_counts"]) == 7


def test_screen_refuses_exclusions_outside_canonical_train(monkeypatch):
    monkeypatch.setattr(rapid_screen, "load_train_rows", lambda root: [{"id": "train"}])
    with pytest.raises(ValueError, match="non-train"):
        rapid_screen.select_heldout_train(Path("."), {"test"}, count=1)


def test_screen_refuses_insufficient_independent_questions(monkeypatch):
    monkeypatch.setattr(rapid_screen, "load_train_rows", lambda root: [{"id": "train"}])
    with pytest.raises(ValueError, match="only 0"):
        rapid_screen.select_heldout_train(Path("."), {"train"}, count=32)
