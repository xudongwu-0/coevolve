from pathlib import Path
import copy
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.alfworld_harness_intervention import remove_action_guard, disable_wrong_object_block


def test_branch_intervention_preserves_remaining_guard_behavior():
    code = "def hook(target, a):\n    if target not in a:\n        return 'wrong'\n    return 'remaining'\n"
    patch = {"actions": [{"type": "add_code_hook", "hook": "on_before_action", "code": code}]}
    changed = disable_wrong_object_block(patch)
    assert patch["actions"][0]["code"] == code
    scope = {}
    exec(changed["actions"][0]["code"], scope)
    assert scope["hook"]('apple', 'put mug') == 'remaining'
    assert scope["hook"]('apple', 'put apple') == 'remaining'
    with pytest.raises(ValueError, match="exactly one"):
        disable_wrong_object_block(changed)


def test_single_hook_intervention_preserves_all_other_content_and_source():
    patch = {"benchmark": "alfworld", "description": "unchanged",
             "actions": [{"type": "add_code_hook", "hook": h, "code": h}
                         for h in ("on_init", "on_before_action", "on_post_step", "make_pre_hint")]}
    old = copy.deepcopy(patch)
    changed = remove_action_guard(patch)
    assert patch == old
    assert changed == {**old, "actions": [old["actions"][i] for i in (0, 2, 3)]}
    with pytest.raises(ValueError, match="exactly one"):
        remove_action_guard(changed)
    patch["actions"].append(copy.deepcopy(patch["actions"][1]))
    with pytest.raises(ValueError, match="exactly one"):
        remove_action_guard(patch)
