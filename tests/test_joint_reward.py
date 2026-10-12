from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.joint_reward import role_reward


@pytest.mark.parametrize("team", [0.0, 1.0])
@pytest.mark.parametrize("error", [None, "invalid syntax"])
@pytest.mark.parametrize("role", ["editor", "solver"])
def test_only_invalid_editor_credit_changes(team, error, role):
    assert role_reward(role, team, error) == team
    result = role_reward(role, team, error, "validity_gated")
    assert result == (0.0 if role == "editor" and error is not None else team)


def test_unknown_credit_cannot_silently_change_objective():
    with pytest.raises(ValueError):
        role_reward("editor", 1, None, "unknown")
    with pytest.raises(ValueError):
        role_reward("solver", -1, None)
