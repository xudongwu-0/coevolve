"""Frozen, larger evaluation of the three completed solver checkpoints.

No learning, model selection, or outcome-dependent stopping. Historical test
cohorts and the remaining unused questions are reported separately.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import pandas as pd

from alternating_common import identity, source_snapshot, spec
from alternating_pool import Pool, read
from math_reward import native
from pipeline import ROOT, sha, write_json

REFERENCE = ROOT / 'runs/whale_alternating_ready_20260916'
OLD = ROOT / 'runs/whale_math_mean8_native_20260915'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def task_ids(entry):
    assert sha(entry['path']) == entry['sha256']
    data = pd.read_parquet(entry['path']).to_dict('records')
    assert len(data) == entry['count']
    return {r['extra_info']['task_id'] for r in data}


def prepare(run, gpus):
    run.mkdir(parents=True, exist_ok=False)
    (run / 'data').mkdir()
    prior, old = read(REFERENCE / 'config.json'), read(OLD / 'config.json')
    assert read(REFERENCE / 'verification.json')['passed_audit']
    source = old['data']['source_files']
    for entry in source.values():
        assert sha(entry['path']) == entry['sha256']
    original_train_ids = {hashlib.sha256(r['problem'].strip().encode()).hexdigest()
        for r in pd.read_parquet(source['train']['path']).to_dict('records')}
    pool = {}
    for r in pd.read_parquet(source['test']['path']).to_dict('records'):
        key = hashlib.sha256(r['problem'].strip().encode()).hexdigest()
        if key in original_train_ids or r['level'] not in {'Level 2', 'Level 3', 'Level 4'}:
            continue
        boxed = native.last_boxed_only_string(r['solution'])
        answer = native.remove_boxed(boxed).strip() if boxed else ''
        if not re.fullmatch(r'-?(0|[1-9]\d*)', answer) or str(int(answer)) != answer:
            continue
        pool.setdefault(key, (r, answer))
    assert len(pool) == 301
    confirmation = old['data']['splits']['confirmation']
    confirmation_ids = task_ids(confirmation)
    existing = {'historical_test128': old['data']['splits']['test'],
                'reused_monitor32': prior['data']['splits']['monitor']}
    existing_ids = {name: task_ids(entry) for name, entry in existing.items()}
    excluded = confirmation_ids | set.union(*existing_ids.values())
    unused = sorted(set(pool) - excluded)
    assert len(unused) == 13
    new_rows = []
    for i, key in enumerate(unused):
        r, answer = pool[key]
        new_rows.append({'data_source': 'math_number_theory_integer',
            'extra_info': {'question': r['problem'], 'task_id': key, 'index': i, 'level': r['level']},
            'reward_model': {'ground_truth': {'target': [answer]}}})
    path = run / 'data/unopened13.parquet'
    pd.DataFrame(new_rows).to_parquet(path, index=False)
    splits = dict(existing, unopened13={'path': str(path), 'sha256': sha(path), 'count': len(new_rows)})
    ids = {name: task_ids(entry) for name, entry in splits.items()}
    union = set.union(*ids.values())
    assert len(union) == sum(map(len, ids.values())) == 173
    assert not union & (original_train_ids | confirmation_ids)
    # Check every previously registered data split, not just the reference run.
    registered = []
    for file in sorted((ROOT / 'runs').glob('*/config.json')):
        cfg = read(file)
        for name, entry in cfg.get('data', {}).get('splits', {}).items():
            if not isinstance(entry, dict) or 'path' not in entry:
                continue
            p = Path(entry['path'])
            if not p.exists() or p.suffix != '.parquet':
                continue
            frame = pd.read_parquet(p)
            if 'extra_info' not in frame:
                continue
            seen = {x['task_id'] for x in frame['extra_info'] if 'task_id' in x}
            assert not set(unused) & seen, ('Previously registered supposedly unused tasks', file, name)
            registered.append({'config': str(file), 'split': name, 'path': str(p), 'sha256': sha(p)})
    models = {
        'W2_initial': prior['solver_initial_checkpoint'],
        'W3_cycle1': str(REFERENCE / 'cycle_01/solver_update/round_01/checkpoint'),
        'W4_cycle2': str(REFERENCE / 'cycle_02/solver_update/round_01/checkpoint'),
    }
    identities = {name: {**identity(path), 'training_state_sha256': sha(Path(path) / 'training_state.json')}
                  for name, path in models.items()}
    harness = prior['initial_harness']
    assert sha(harness) == read(REFERENCE / 'FINAL_MODELS.json')['harness_sha256']
    config = {'reference': str(REFERENCE), 'reference_verification_sha256': sha(REFERENCE / 'verification.json'),
        'models': identities, 'solver_initial_checkpoint': models['W2_initial'],
        'checkpoint_allowlist': list(models.values()), 'pool_gpus': gpus,
        'harness': harness, 'harness_sha256': sha(harness), 'draws': 8, 'seed': 42, 'temperature': 1.0,
        'assistant_token_budget': 2048, 'policy_max_tokens': 1024, 'context_limit': 4096,
        'data': {'source_files': source, 'source_revision': old['data']['source_revision'],
                 'splits': splits, 'excluded_confirmation': confirmation},
        'cohort_order': list(splits), 'model_order': list(models),
        'training_updates': 0, 'primary_comparison': 'W4_cycle2 minus W2_initial on all173',
        'metric': 'mean_at_8; question-paired bootstrap, 20000 resamples, seed 20260914',
        'limitations': ['160 questions have prior evaluation exposure; 13 were absent from prior registered splits',
            'No fresh training seeds; inference repeats do not measure training variability',
            'H3 is identical at all three stages; this measures solver weight changes conditional on H3',
            'This is not a gate-versus-WHALE comparison',
            'No claim of fresh independent confirmation from the pooled173 outcome']}
    write_json(run / 'config.json', config)
    source_snapshot(run, 'frozen', extra=['expanded_alternating_eval.py', 'audit_expanded_eval.py', 'report_expanded_eval.py'])
    write_json(run / 'PROTOCOL.json', {'registered_utc': utc(), 'config_sha256': sha(run / 'config.json'),
        'fixed_trajectories': 173 * 8 * 3, 'no_outcome_dependent_stopping': True,
        'dataset_selection': 'All301 eligible official-test tasks minus128 prior development confirmation tasks; no outcome filtering',
        'registered_split_inventory': registered,
        'prior_exposure': {'historical_test128': 'Earlier WHALE/gate evaluation',
                          'reused_monitor32': 'Two-cycle monitoring and preceding dual-model pilot',
                          'unopened13': 'Absent from all earlier config-declared splits'}})
    write_json(run / 'STATUS.json', {'stage': 'prepared', 'time': utc()})
    print(json.dumps({'prepared': str(run), 'questions': 173, 'draws': 8,
                      'models': list(models), 'trajectories': 4152, 'gpus': gpus}), flush=True)


def execute(run):
    config = read(run / 'config.json')
    assert sha(run / 'config.json') == read(run / 'PROTOCOL.json')['config_sha256']
    assert not (run / 'pool').exists(), 'Use a separately documented recovery for interrupted runs'
    start = time.monotonic()
    write_json(run / 'START.json', {'utc': utc(), 'config_sha256': sha(run / 'config.json')})
    try:
        with Pool(run / 'pool', config) as pool:
            for name in config['model_order']:
                write_json(run / 'STATUS.json', {'stage': 'evaluating', 'model': name, 'time': utc()})
                jobs = [spec(config['models'][name]['model'], config['harness'], config['data']['splits'][cohort],
                    run / 'eval' / name / cohort, draws=config['draws'], seed=config['seed'],
                    temperature=config['temperature']) for cohort in config['cohort_order']]
                results = pool.evaluate_many(jobs)
                print(json.dumps({'finished_model': name, 'scores': [
                    {'cohort': c, 'solved': v['solved_examples'], 'total': v['num_examples']}
                    for c, v in zip(config['cohort_order'], results)],
                    'elapsed_seconds': time.monotonic() - start}), flush=True)
        write_json(run / 'COMPLETE.json', {'utc': utc(), 'elapsed_seconds': time.monotonic() - start,
            'models': config['model_order'], 'trajectories': 4152, 'training_updates': 0})
        write_json(run / 'STATUS.json', {'stage': 'evaluation_complete_pending_audit', 'time': utc()})
    except BaseException as exc:
        write_json(run / 'STATUS.json', {'stage': 'failed', 'time': utc(), 'error': repr(exc)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--gpus', nargs='+', default=['6', '7', '8', '9'] + [f'node1:{i}' for i in range(4, 10)])
    args = p.parse_args()
    if args.action == 'prepare':
        prepare(args.run.resolve(), args.gpus)
    else:
        execute(args.run.resolve())


if __name__ == '__main__':
    main()
