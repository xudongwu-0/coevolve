from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'whale_local'))
from sao_bridge.joint_probability_replay import replay_row, summarize


def receipt():
    return {'role': 'editor', 'version': 0, 'calls': [{
        'prompt_ids': [10, 11], 'response_ids': [12, 13],
        'response_logprobs': [-.5, -.25], 'versions': [0, 0]}]}


def test_native_prediction_alignment_excludes_prompt_and_last_unused_position():
    row = replay_row(receipt())
    value = torch.tensor([[99., -.5, -.25, 99.]])
    r = summarize(row, value)
    assert r['action_tokens'] == 2
    assert r['log_importance_sum'] == 0
    assert r['action_logprobs'] == [-.5, -.25]
    assert not r['clipped_at_bound2']
    value[0, 1] = -1.5
    assert summarize(row, value)['clipped_at_bound2']


def test_rejects_updated_policy_and_missing_behavior_tokens():
    r = receipt();r['calls'][0]['versions'][0] = 1
    with pytest.raises(ValueError, match='updated policy'):
        replay_row(r)
    r = receipt();r['calls'][0]['response_logprobs'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        replay_row(r)
