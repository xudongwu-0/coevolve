"""Independent receipt, token-mask, optimizer, and checkpoint diagnostic audit."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from safetensors import safe_open
import torch

from pipeline import ROOT, sha, write_json
from math_reward import binary_reward
from math_trajectory_rsft import accepted_groups


def read(path):
    return json.loads(Path(path).read_text())


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol, preflight = read(run / 'PROTOCOL.json'), read(run / 'PREFLIGHT.json')
    assert not preflight['check_only']
    assert sha(run / 'PREFLIGHT.json') == protocol['preflight_sha256']
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest
        assert sha(run / 'source_snapshot' / Path(path).relative_to(ROOT)) == digest
    source = Path(preflight['source'])
    assert sha(source / 'verification.json') == preflight['source_audit_sha256']
    source_path = source / 'round_01/rollouts/trajectories.jsonl'
    assert sha(source_path) == preflight['source_trajectories_sha256']
    rows = [json.loads(line) for line in source_path.read_text().splitlines()]
    config = read(source / 'config.json')
    entry = config['data']['splits']['train']
    assert sha(entry['path']) == entry['sha256']
    targets = {row['extra_info']['task_id']: row['reward_model']['ground_truth']['target']
               for row in pd.read_parquet(entry['path']).to_dict('records')}
    for row in rows:
        assert binary_reward(row['last_assistant'], targets[row['task_id']]) == row['reward']
        for call in row['policy_calls']:
            receipt = source / 'calls' / (call['call_id'] + '.json')
            assert call == read(receipt)
            assert sha(receipt) == preflight['receipt_sha256'][str(receipt)]
    groups = accepted_groups(rows, protocol['trajectory_minibatch'])
    tokens, call_count, all_keys = 0, 0, []
    for number, group in enumerate(groups, 1):
        phase = run / f'group_{number:03d}'
        record = read(phase / 'input.json')
        assert sha(phase / 'batch.pt') == record['batch_sha256']
        keys = [[row['draw'], row['example_index']] for row in group]
        assert record['trajectory_keys'] == keys
        all_keys.extend(keys)
        calls = [call for row in group for call in row['policy_calls']]
        assert record['call_ids'] == [call['call_id'] for call in calls]
        batch = torch.load(phase / 'batch.pt', map_location='cpu', weights_only=True)
        assert batch['input_ids'].shape[0] == len(calls)
        width = batch['input_ids'].shape[1] - batch['responses'].shape[1]
        for index, call in enumerate(calls):
            assert batch['input_ids'][index, :width][batch['attention_mask'][index, :width].bool()].tolist() == call['prompt_ids']
            assert batch['responses'][index][batch['response_mask'][index].bool()].tolist() == call['response_ids']
            assert batch['attention_mask'][index, width:].tolist() == batch['response_mask'][index].tolist()
        assert torch.equal(batch['position_ids'], (batch['attention_mask'].cumsum(-1) - 1).clamp(min=0))
        assert int(batch['response_mask'].sum()) == record['response_tokens']
        result = read(phase / 'result.json')
        assert {key: result[key] for key in record} == record
        assert result['metrics']['actor/sft_updates'] == 1
        assert result['metrics']['actor/sft_token_count'] == record['response_tokens']
        tokens += record['response_tokens']
        call_count += len(calls)
    assert len(all_keys) == len({tuple(key) for key in all_keys}) == preflight['accepted_trajectories']
    assert tokens == preflight['response_tokens'] and call_count == preflight['accepted_policy_calls']
    info = read(run / 'checkpoint/training_state.json')
    before = read(run / 'before_parameter_hashes.json')
    assert hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest() == protocol['expected_base_digest']
    actual = {}
    for path in (run / 'checkpoint').glob('*.safetensors'):
        with safe_open(path, framework='pt', device='cpu') as saved:
            for name in saved.keys():
                if name in info['parameter_hashes']:
                    tensor = saved.get_tensor(name)
                    assert tensor.dtype == torch.float32
                    actual[name] = tensor_hash(tensor)
    assert actual == info['parameter_hashes']
    assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == info['weight_digest']
    changed = [name for name in before if before[name] != actual[name]]
    assert changed == info['changed_parameter_tensors'] and changed
    assert all('visual' not in name for name in changed)
    assert info['optimizer_steps_before'] == [] and info['optimizer_steps_after'] == [len(groups)]
    optimizer = torch.load(run / 'checkpoint/optimizer.pt', map_location='cpu', weights_only=True)
    assert {int(value['step'].item()) for value in optimizer['state'].values()} == {len(groups)}
    assert {value.dtype for state in optimizer['state'].values() for key, value in state.items()
            if key != 'step' and torch.is_tensor(value)} == {torch.float32}
    assert all(group['lr'] == preflight['learning_rate'] for group in optimizer['param_groups'])
    reloaded = read(run / 'reload_verification.json')
    assert reloaded['weight_digest'] == info['weight_digest']
    assert reloaded['optimizer_steps'] == [len(groups)]
    assert read(run / 'complete.json')['checkpoint_reload_verified']
    result = {'passed_audit': True, 'weight_digest': info['weight_digest'],
        'sampled_trajectories_reused': len(rows), 'accepted_trajectories': len(all_keys),
        'accepted_policy_calls': call_count, 'response_tokens': tokens,
        'trajectory_minibatch': protocol['trajectory_minibatch'], 'optimizer_updates': len(groups),
        'learning_rate': preflight['learning_rate'], 'all_parameter_tensors_independently_hashed': True,
        'new_generation_calls': 0, 'confirmation_used': False, 'test_used': False,
        'qualified_for_gate': False, 'auditor_sha256': sha(__file__)}
    write_json(run / 'verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
