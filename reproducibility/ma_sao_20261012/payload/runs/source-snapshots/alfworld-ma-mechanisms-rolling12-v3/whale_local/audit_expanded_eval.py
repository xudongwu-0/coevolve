"""Independently check frozen identities, raw receipts, schedules and outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from alternating_pool import read
from math_context_audit import recheck_context
from math_reward import binary_reward, native
from pipeline import ROOT, sha, write_json


def audit(run):
    config = read(run / 'config.json')
    protocol = read(run / 'PROTOCOL.json')
    assert protocol['config_sha256'] == sha(run / 'config.json')
    assert read(run / 'COMPLETE.json')['trajectories'] == protocol['fixed_trajectories'] == 4152
    assert config['training_updates'] == 0
    prior = Path(config['reference'])
    assert sha(prior / 'verification.json') == config['reference_verification_sha256']
    assert read(prior / 'verification.json')['passed_audit']
    upstream = ROOT / 'WHALE'
    commit = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == 'fbe125eb7abea7f760c99ab9acc1a6261e708fc6'
    subprocess.run(['git', '-C', str(upstream), 'diff', '--quiet', 'HEAD'], check=True)
    for manifest in (run / 'source_snapshots').glob('*/manifest.json'):
        for source, digest in read(manifest).items():
            assert sha(source) == sha(manifest.parent / Path(source).name) == digest
    for item in config['models'].values():
        p = Path(item['model']) / 'training_state.json'
        assert sha(p) == item['training_state_sha256']
        assert read(p)['weight_digest'] == item['weight_digest']
    assert sha(config['harness']) == config['harness_sha256']
    datasets, all_ids = {}, set()
    for cohort, entry in config['data']['splits'].items():
        assert sha(entry['path']) == entry['sha256']
        data = pd.read_parquet(entry['path']).to_dict('records')
        assert len(data) == entry['count']
        ids = {r['extra_info']['task_id'] for r in data}
        assert len(ids) == len(data) and not ids & all_ids
        all_ids |= ids
        datasets[cohort] = data
    assert len(all_ids) == 173
    for entry in config['data']['source_files'].values():
        assert sha(entry['path']) == entry['sha256']
    original_train = pd.read_parquet(config['data']['source_files']['train']['path']).to_dict('records')
    train_ids = {hashlib.sha256(r['problem'].strip().encode()).hexdigest() for r in original_train}
    assert not all_ids & train_ids
    source = {hashlib.sha256(r['problem'].strip().encode()).hexdigest(): r for r in
              pd.read_parquet(config['data']['source_files']['test']['path']).to_dict('records')}
    for data in datasets.values():
        for r in data:
            raw = source[r['extra_info']['task_id']]
            assert r['extra_info']['question'] == raw['problem']
            assert raw['level'] in {'Level 2', 'Level 3', 'Level 4'}
            answer = native.remove_boxed(native.last_boxed_only_string(raw['solution'])).strip()
            assert list(r['reward_model']['ground_truth']['target']) == [answer]
    confirmation = config['data']['excluded_confirmation']
    assert sha(confirmation['path']) == confirmation['sha256']
    confirmation_ids = {x['task_id'] for x in pd.read_parquet(confirmation['path'])['extra_info']}
    assert not all_ids & confirmation_ids
    unused_ids = {r['extra_info']['task_id'] for r in datasets['unopened13']}
    for entry in protocol['registered_split_inventory']:
        assert sha(entry['path']) == entry['sha256']
        ids = {x['task_id'] for x in pd.read_parquet(entry['path'])['extra_info'] if 'task_id' in x}
        assert not unused_ids & ids
    receipt_files = {p.stem: p for p in (run / 'pool').rglob('calls/*.json')}
    assert len(receipt_files) == sum(1 for _ in (run / 'pool').rglob('calls/*.json'))
    used, contexts, summaries, total = set(), set(), {}, 0
    for name in config['model_order']:
        summaries[name] = {}
        for cohort in config['cohort_order']:
            directory = run / 'eval' / name / cohort
            value = read(directory / 'val.json')
            meta = value['metadata']
            entry = config['data']['splits'][cohort]
            assert meta['dataset_sha256'] == entry['sha256']
            assert meta['dataset_path'] == entry['path']
            assert meta['weight_digest'] == config['models'][name]['weight_digest']
            assert meta['harness_sha256'] == config['harness_sha256']
            assert meta['seed'] == config['seed'] and meta['temperature'] == config['temperature']
            plan = [[d, i] for d in range(config['draws']) for i in range(entry['count'])]
            assert meta['sampling_schedule'] == plan and value['num_examples'] == len(plan)
            solved, tokens, calls, actual = 0, 0, 0, 0
            with (directory / 'trajectories.jsonl').open() as stream:
                for pos, line in enumerate(stream):
                    row = json.loads(line)
                    draw, index = plan[pos]
                    task = datasets[cohort][index]
                    assert row['draw'] == draw and row['example_index'] == index
                    assert row['task_id'] == task['extra_info']['task_id']
                    assert row['rollout_seed'] == config['seed'] * 100003 + draw * 1009 + index
                    assert row['sampling_identity']['weight_digest'] == meta['weight_digest']
                    assert row['harness_sha256'] == meta['harness_sha256'] and row['answer_redacted']
                    assert row['reward'] == binary_reward(row['last_assistant'], task['reward_model']['ground_truth']['target'])
                    assert row['assistant_tokens'] == sum(len(c['response_ids']) for c in row['policy_calls']) <= 2048
                    physical = Path(row['physical_worker_run'])
                    for c in row['policy_calls']:
                        key = c['call_id']
                        assert key not in used
                        assert read(physical / 'calls' / (key + '.json')) == c == read(receipt_files[key])
                        assert c['sampling_identity']['weight_digest'] == meta['weight_digest']
                        assert c['stop_at_chat_end'] and len(c['response_ids']) <= c['requested_max_tokens'] <= 1024
                        assert len(c['response_ids']) == len(c['response_logprobs'])
                        assert np.isfinite(c['response_logprobs']).all()
                        used.add(key)
                    if row.get('context_rejection_id'):
                        key = row['context_rejection_id']
                        assert key not in contexts
                        recheck_context(physical / 'context_rejections' / (key + '.json'))
                        contexts.add(key)
                    solved += row['reward']; tokens += row['assistant_tokens']
                    calls += len(row['policy_calls']); actual += 1
            assert actual == len(plan) and solved == value['solved_examples']
            assert value['success_rate'] == solved / actual
            total += actual
            summaries[name][cohort] = {'solved': int(solved), 'attempts': actual,
                'questions': entry['count'], 'mean_at_8_percent': solved / actual * 100,
                'mean_assistant_tokens': tokens / actual, 'policy_calls': calls,
                'trajectories_sha256': sha(directory / 'trajectories.jsonl')}
    assert total == 4152 and used == set(receipt_files)
    assert contexts == {p.stem for p in (run / 'pool').rglob('context_rejections/*.json')}
    stopped = read(run / 'pool/pool_stopped.json')
    assert len(stopped['workers']) == len(config['pool_gpus'])
    assert all(w['returncode'] == 0 for w in stopped['workers'])
    loads = []
    for directory in sorted((run / 'pool/replicas').glob('slot_*')):
        ready = read(directory / 'ready.json')
        assert ready['weight_digest'] == config['models']['W2_initial']['weight_digest']
        for file in directory.glob('checkpoint_loads/*.json'):
            proof = read(file)
            assert proof['all_parameter_hashes_equal'] and proof['weights'] in config['checkpoint_allowlist']
            assert proof['weight_digest'] == read(Path(proof['weights']) / 'training_state.json')['weight_digest']
            loads.append(str(file))
        services, stopped_services = read(directory / 'services.json'), read(directory / 'services_stopped.json')
        assert all(services[k] == stopped_services[k] for k in ['worker_pid', 'sandbox_pid'])
        assert all(stopped_services[k] in (0, -15) for k in ['worker_returncode', 'sandbox_returncode'])
    report = {'passed_audit': True, 'verified_trajectories': total, 'verified_unique_calls': len(used),
        'context_rejections': len(contexts), 'summaries': summaries, 'training_updates': 0,
        'model_states_unchanged': True, 'checkpoint_reload_proofs': loads,
        'evaluation_ids_disjoint_from_entire_official_train_and_development_confirmation': True,
        'unused13_absent_from_previously_registered_splits': True,
        'sources_frozen': True, 'upstream_commit': commit, 'services_stopped': True,
        'config_sha256': sha(run / 'config.json'), 'protocol_sha256': sha(run / 'PROTOCOL.json')}
    write_json(run / 'verification.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ['summaries', 'checkpoint_reload_proofs']}), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    audit(p.parse_args().run.resolve())
