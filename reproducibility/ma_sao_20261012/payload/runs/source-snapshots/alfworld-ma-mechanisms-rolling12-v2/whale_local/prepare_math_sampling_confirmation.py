"""Prepare a blind conditional mean@8 confirmation; performs no model calls."""
import argparse
import copy
import datetime
import hashlib
from pathlib import Path
import re

import pandas as pd

from pipeline import ROOT, sha, write_json
from verify_math_run import read
from math_reward import native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    assert not out.exists()
    reference = ROOT / 'runs/whale_math_onecycle_fixed_native_20260915'
    development = ROOT / 'runs/whale_math_harness_sampling_diagnostic_20260915'
    assert not (development / 'verification.json').exists(), 'Register before four-cell development result'
    ref_audit = read(reference / 'verification.json')
    assert ref_audit['passed_audit'] and not ref_audit['qualified_for_gate']
    data = read(reference / 'config.json')['data']
    summary = read(reference / 'summary.json')
    for entry in data['source_files'].values():
        assert sha(entry['path']) == entry['sha256']
    # Exclude the entire official training source, not just the selected training
    # questions. This also excludes both already-used confirmation partitions.
    original_train_hashes = {hashlib.sha256(row['problem'].strip().encode()).hexdigest()
        for row in pd.read_parquet(data['source_files']['train']['path']).to_dict('records')}
    reserved_test = data['splits']['test']
    assert sha(reserved_test['path']) == reserved_test['sha256']
    reserved_hashes = {info['task_id'] for info in pd.read_parquet(reserved_test['path'])['extra_info']}
    assert len(reserved_hashes) == 128
    excluded = original_train_hashes | reserved_hashes
    pool = {}
    for row in pd.read_parquet(data['source_files']['test']['path']).to_dict('records'):
        key = hashlib.sha256(row['problem'].strip().encode()).hexdigest()
        if key in excluded or row['level'] not in {'Level 2', 'Level 3', 'Level 4'}:
            continue
        boxed = native.last_boxed_only_string(row['solution'])
        target = native.remove_boxed(boxed).strip() if boxed else ''
        if not re.fullmatch(r'-?(0|[1-9]\d*)', target) or str(int(target)) != target:
            continue
        pool.setdefault(key, (row, target))
    assert len(pool) == 173
    chosen = sorted(pool)[:128]
    assert not set(chosen).intersection(excluded)
    for parent in [ROOT / 'runs/whale_math_fixed_native_20260915', reference]:
        for entry in read(parent / 'config.json')['data']['splits'].values():
            assert sha(entry['path']) == entry['sha256']
            assert not set(chosen).intersection(info['task_id'] for info in pd.read_parquet(entry['path'])['extra_info'])
    out.mkdir(parents=True)
    rows = []
    for index, key in enumerate(chosen):
        row, target = pool[key]
        rows.append({'data_source': 'math_number_theory_integer',
            'extra_info': {'question': row['problem'], 'task_id': key, 'index': index, 'level': row['level']},
            'reward_model': {'ground_truth': {'target': [target]}}})
    path = out / 'confirmation.parquet'
    pd.DataFrame(rows).to_parquet(path, index=False)
    manifest = copy.deepcopy(data)
    manifest['splits']['confirmation'] = {'path': str(path), 'sha256': sha(path), 'count': 128}
    manifest['selection'] = ('Original TRAIN/MH retained. Confirmation uses the next128 eligible official-test '
        'questions in SHA order after excluding all official-train questions and the untouched final-test128. '
        'No policy-reward filtering.')
    manifest['confirmation_selection'] = manifest['selection']
    manifest['confirmation_candidate_pool_count'] = len(pool)
    manifest['reserved_final_test_unchanged'] = True
    write_json(out / 'manifest.json', manifest)
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'source': str(Path(__file__).resolve()), 'source_sha256': sha(__file__),
        'condition': 'Launch only if the frozen four-cell mean@8 DEVELOPMENT audit passes and both conditional gains are at least5pp. Development does not itself qualify a gate.',
        'development_run': str(development), 'development_protocol_sha256': sha(development / 'PROTOCOL.json'),
        'reference_native_prefix': str(reference), 'reference_audit_sha256': sha(reference / 'verification.json'),
        'candidate_weight_digest': summary['final_identity']['weight_digest'],
        'candidate_harness_sha256': sha(summary['final_harness']),
        'manifest': str(out / 'manifest.json'), 'manifest_sha256': sha(out / 'manifest.json'),
        'new_confirmation_questions': 128, 'temperature': 1.0, 'draws_per_question': 8,
        'top_p': 1.0, 'top_k': 20, 'seed': 42, 'thinking': False,
        'policy_max_tokens': 1024, 'assistant_token_budget': 2048,
        'metric': 'mean@8, with question-level paired bootstrap intervals; not pass@8',
        'four_cells': ['W0_H0', 'W1_H0', 'W0_Hf', 'W1_Hf'],
        'minimum_conditional_gain_pp': 5.0, 'new_confirmation_trajectories': 4096,
        'new_training_and_MH_generations': 0, 'reuse_frozen_native_prefix': True,
        'logical_training_trajectories': 512, 'logical_MH_trajectories': 240,
        'all_prior_screening_costs_retained': True, 'reserved_final_test_unchanged': True,
        'confirmation_used': False, 'test_used': False, 'gate_started': False,
        'scope': 'Sequential scenario discovery; all prior negative greedy confirmations remain reported. A prospective sampling-metric confirmation cannot rewrite those results.'}
    write_json(out / 'CONDITIONAL_PROTOCOL.json', protocol)
    print({'prepared_only': True, 'unused_pool_size': len(pool), 'new_confirmation_questions': len(chosen),
           'new_generations': 0, 'out': str(out)})


if __name__ == '__main__':
    main()
