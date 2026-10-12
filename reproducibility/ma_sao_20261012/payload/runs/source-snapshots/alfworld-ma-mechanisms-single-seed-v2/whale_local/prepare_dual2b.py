"""Freeze the local dual-checkpoint pilot and a previously unused test slice."""
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
from harness_bootstrap import messages, START
from math_reward import native
from pipeline import ROOT, sha, write_json

RUN = ROOT / 'runs/whale_dual2b_20260916'
REFERENCE = ROOT / 'runs/whale_math_mean8_native_20260915'

def read(p): return json.loads(Path(p).read_text())

def feedback(rows):
    failed = [r for r in rows if not r['reward']]
    missing = sum('\\boxed{' not in r['last_assistant'] for r in failed)
    errors = sum(any('Traceback' in str(t.get('feedback', '')) or '[tool error]' in str(t.get('feedback', ''))
        or '[parse error]' in str(t.get('feedback', '')) for t in r.get('harness_trace', [])) for r in failed)
    return (f'Out of {len(rows)} development attempts, {len(failed)} failed. '
        f'{missing} failures ended without a boxed final answer; {errors} contained interpreter errors. '
        'A tool-free partial response currently stops immediately. The total assistant-token budget remains 2048. '
        'Address a likely general failure mechanism while retaining successful behavior. '
        'No reference answers or question-specific solutions are supplied.')

def main():
    old = read(REFERENCE / 'config.json')
    audit = read(REFERENCE / 'verification.json'); assert audit['passed_audit']
    data = copy.deepcopy(old['data'])
    excluded = {hashlib.sha256(r['problem'].strip().encode()).hexdigest()
        for r in pd.read_parquet(data['source_files']['train']['path']).to_dict('records')}
    for entry in data['splits'].values():
        assert sha(entry['path']) == entry['sha256']
        excluded.update(r['task_id'] for r in pd.read_parquet(entry['path'])['extra_info'])
    pool = {}
    for row in pd.read_parquet(data['source_files']['test']['path']).to_dict('records'):
        key = hashlib.sha256(row['problem'].strip().encode()).hexdigest()
        if key in excluded or row['level'] not in {'Level 2', 'Level 3', 'Level 4'}: continue
        boxed = native.last_boxed_only_string(row['solution'])
        target = native.remove_boxed(boxed).strip() if boxed else ''
        if not re.fullmatch(r'-?(0|[1-9]\d*)', target) or str(int(target)) != target: continue
        pool[key] = (row, target)
    assert len(pool) == 45
    chosen = sorted(pool)[:32]
    rows = []
    for i, key in enumerate(chosen):
        r, target = pool[key]
        rows.append({'data_source': 'math_number_theory_integer',
            'extra_info': {'question': r['problem'], 'task_id': key, 'index': i, 'level': r['level']},
            'reward_model': {'ground_truth': {'target': [target]}}})
    directory = RUN / 'data'; directory.mkdir()
    path = directory / 'fresh_test.parquet'; pd.DataFrame(rows).to_parquet(path, index=False)
    data['splits'] = {k: data['splits'][k] for k in ['train', 'mh_val']}
    data['splits']['test'] = {'path': str(path), 'sha256': sha(path), 'count': 32}
    data['sizes'] = {k: v['count'] for k, v in data['splits'].items()}
    data['selection'] = 'First32 question hashes among45 eligible official-test tasks not used in any previous registered split; no reward filtering.'
    config = {'scope': 'internal 2B harness editor + separate 2B task solver; native WHALE RSFT and MH selection',
        'data': data, 'base_model': old['base_model'], 'gpu': '8',
        'parameter_dtype': 'float32', 'learning_rate': 1e-6,
        'evaluation_rollouts': 4, 'evaluation_temperature': 1.0,
        'mh_iterations': 2, 'mh_proposals': 2,
        'solver_prefix_reference': str(REFERENCE), 'solver_prefix_reuse': True,
        'editor_initial_checkpoint': str(RUN / 'editor_sft/checkpoint'),
        'editor_online_learning_rate': 1e-6,
        'editor_online_acceptance': 'strictly higher MH solved count than current incumbent; all native task outcomes verified',
        'online_editor_update': 'one SFT epoch on exact generated tokens from strictly improving proposals; no teacher fixes',
        'teacher_runtime_calls': 0, 'test_open_after_frozen': True}
    write_json(RUN / 'config.json', config)
    write_json(RUN / 'eval_hardware.json', {
        'sampler_gpus': ['6','7','9'] + [f'node1:{i}' for i in range(10)],
        'sampler_port_base': 20100, 'mh_replica_port_base': 22000,
        'confirmation_gpus': ['6','7','9'], 'confirmation_port_base': 25000,
        'confirmation_replica_port_base': 27000,
        'confirmation_replica_groups': [['node1:0','node1:1','node1:2'], ['node1:3','node1:4','node1:5'],
            ['node1:6','node1:7'], ['node1:8','node1:9']]})
    protocol = {'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'reference_verification_sha256': sha(REFERENCE / 'verification.json'),
        'bootstrap_data_manifest': read(RUN / 'bootstrap_data/manifest.json'),
        'test_count': 32, 'test_draws': 4, 'remaining_unopened_pool': 13,
        'pilot_limits': ['small test, descriptive estimates only', 'bootstrap checks hold out wording, not mechanism families',
            'reuses exact qualified native solver update, not a new training claim',
            'repeated MH labels are meta-training feedback, not unbiased test',
            'one-cycle pilot cannot establish sustained self-improvement or superiority to external proposer'],
        'source_files': {str(p): sha(p) for p in (ROOT / 'whale_local').glob('harness_*.py')},
        'config_sha256': sha(RUN / 'config.json')}
    write_json(RUN / 'PROTOCOL.json', protocol)
    links = RUN / 'checkpoints'; links.mkdir()
    (links / 'harness_E0').symlink_to(RUN / 'editor_sft/checkpoint', target_is_directory=True)
    (links / 'solver_W0').symlink_to(old['base_model'], target_is_directory=True)
    (links / 'solver_W1').symlink_to(REFERENCE / 'round_01/checkpoint', target_is_directory=True)
    write_json(RUN / 'CHECKPOINT_ROLES.json', {'harness_E0': str(links / 'harness_E0'),
        'solver_W0': str(links / 'solver_W0'), 'solver_W1': str(links / 'solver_W1'),
        'shared_updates': False, 'optimizer_states_separate': True})
    print(json.dumps({'test_count': len(rows), 'unused_pool': len(pool), 'config': str(RUN / 'config.json')}))

if __name__ == '__main__': main()
