"""Rebuild the pilot question splits from local official MATH parquet sources."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'whale_local'))
from math_reward import native
from pipeline import sha, write_json

REVISION = '21a5633873b6a120296cce3e2df9d5550074f4a3'


def prepare(source, output):
    import pandas as pd
    if output.exists():
        raise FileExistsError(output)
    pools = {}
    all_train = set()
    for split in ['train', 'test']:
        pool = {}
        for row in pd.read_parquet(source / f'{split}.parquet').to_dict('records'):
            key = hashlib.sha256(row['problem'].strip().encode()).hexdigest()
            if split == 'train':
                all_train.add(key)
            elif key in all_train:
                continue
            if row['level'] not in {'Level 2', 'Level 3', 'Level 4'}:
                continue
            boxed = native.last_boxed_only_string(row['solution'])
            answer = native.remove_boxed(boxed).strip() if boxed else ''
            if not re.fullmatch(r'-?(0|[1-9]\d*)', answer) or str(int(answer)) != answer:
                continue
            pool.setdefault(key, (row, answer))
        pools[split] = pool
    train, test = sorted(pools['train']), sorted(pools['test'])
    if len(train) < 176 or len(test) < 256:
        raise ValueError('Official sources are too small for the recorded split rule')
    # The original first128 official-test questions belong to the separate gate
    # experiment. The H/W pilot used the next128 (historically "confirmation").
    selections = {'train': train[:128], 'mh_val': train[128:176], 'test': test[128:256]}
    output.mkdir(parents=True)
    manifest = {'source': 'EleutherAI/hendrycks_math', 'config': 'number_theory',
        'expected_source_revision': REVISION, 'source_revision_verified': False,
        'source_files': {s: {'sha256': sha(source / f'{s}.parquet')} for s in ['train', 'test']},
        'filter': 'Level2-4, canonical boxed integer; no policy-reward filtering',
        'test_selection': 'SHA256 order of eligible official test, indices128:256, excluding all official train',
        'splits': {}}
    for name, keys in selections.items():
        pool = pools['test' if name == 'test' else 'train']
        records = []
        for index, key in enumerate(keys):
            row, answer = pool[key]
            records.append({'data_source': 'math_number_theory_integer',
                'extra_info': {'question': row['problem'], 'task_id': key, 'index': index, 'level': row['level']},
                'reward_model': {'ground_truth': {'target': [answer]}}})
        path = output / f'{name}.parquet'
        pd.DataFrame(records).to_parquet(path, index=False)
        manifest['splits'][name] = {'path': path.name, 'sha256': sha(path), 'count': len(records)}
    write_json(output / 'manifest.json', manifest)
    print(json.dumps({'manifest': str(output / 'manifest.json'), 'counts': {k:len(v) for k,v in selections.items()}}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Directory with official train.parquet and test.parquet')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source.resolve(), args.output.resolve())
