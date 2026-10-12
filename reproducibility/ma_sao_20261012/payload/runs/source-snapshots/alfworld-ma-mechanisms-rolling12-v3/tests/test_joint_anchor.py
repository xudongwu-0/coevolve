import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "whale_local"))
from sao_bridge.joint_anchor import build_rows, is_anchor_round, route_patch, source_round


def test_same_joint_sequence_extra_fresh_anchor_ids_and_exact_budget():
    original = [{"id": f"source-{i}", "task_id": i % 8,
                 "messages": [{"content": f"evidence-{i}"}]} for i in range(256)]
    before = copy.deepcopy(original)
    result = build_rows(original, cycles=16, batch_size=8)
    joint = [r for r in result if not r['anchor_round']]
    anchor = [r for r in result if r['anchor_round']]
    assert len(joint) == 256 and len(anchor) == 128
    assert len({r['id'] for r in result}) == 384
    assert [r['messages'] for r in joint] == [r['messages'] for r in original]
    assert [r['schedule_source_index'] for r in joint] == list(range(256))
    assert original == before
    result[0]['messages'][0]['content'] = 'mutated'
    assert original == before
    assert anchor[0]['messages'][0]['content'] == 'evidence-0'


def test_arms_differ_only_in_applied_patch_on_anchor_rounds():
    proposed = {"actions": ["learned"]}; fixed = {"actions": ["incumbent"]}
    for step in range(48):
        control, cm = route_patch(proposed, fixed, round_index=step, mode='sampled_control')
        treatment, tm = route_patch(proposed, fixed, round_index=step, mode='fixed_anchor')
        assert cm['engineer_update_eligible'] == tm['engineer_update_eligible'] == (step % 3 != 2)
        assert cm['solver_peer_weight_rule'] == tm['solver_peer_weight_rule']
        assert control == proposed
        assert treatment == (fixed if step % 3 == 2 else proposed)
    treatment['actions'].append('mutation')
    assert fixed == {"actions": ["incumbent"]}


def test_source_balances_extra_exposure_without_changing_joint_order():
    assert [source_round(i) for i in range(12)] == [0, 1, 0, 2, 3, 3, 4, 5, 4, 6, 7, 7]


@pytest.mark.parametrize('value', [-1, True, 1.5])
def test_invalid_round_rejected(value):
    with pytest.raises(ValueError):
        is_anchor_round(value)


def test_invalid_arm_and_missing_source_rejected():
    with pytest.raises(ValueError):
        route_patch({}, {}, round_index=0, mode='unknown')
    with pytest.raises(ValueError):
        build_rows([], cycles=16, batch_size=8)
