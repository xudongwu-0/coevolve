"""Prepare a fresh H -> W -> H run from explicit local assets; no model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'whale_local'))
from pipeline import sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def resolve(path, base=ROOT):
    value = Path(path).expanduser()
    return (value if value.is_absolute() else base / value).resolve()


def checkpoint(path):
    path = resolve(path)
    for name in ['config.json', 'tokenizer_config.json', 'training_state.json', 'optimizer.pt', 'rng.pt']:
        if not (path / name).is_file():
            raise ValueError(f'Missing {path / name}; use a complete native local training checkpoint')
    state = read(path / 'training_state.json')
    if not re.fullmatch(r'[0-9a-f]{64}', state.get('weight_digest', '')):
        raise ValueError(f'Invalid checkpoint weight digest: {path}')
    if not state.get('parameter_hashes') or 'optimizer_steps_after' not in state:
        raise ValueError(f'Incomplete checkpoint provenance: {path}')
    if not list(path.glob('*.safetensors')):
        raise ValueError(f'Missing safetensors weights: {path}')
    return {'model': str(path), 'weight_digest': state['weight_digest']}


def prepare(spec_path, run, check_only=False):
    import pandas as pd
    spec = read(spec_path)
    arms = spec.get('arms', ['fixed_editor', 'trained_editor'])
    valid = {'external', 'fixed_editor', 'trained_editor'}
    if not arms or len(arms) != len(set(arms)) or set(arms) - valid:
        raise ValueError(f'Choose distinct arms from {sorted(valid)}')
    w, e = checkpoint(spec['solver_checkpoint']), checkpoint(spec['editor_checkpoint'])
    manifest_path = resolve(spec['data_manifest'])
    manifest = read(manifest_path)
    splits = {}
    seen = set()
    for name, count in [('train', 128), ('mh_val', 48), ('test', 128)]:
        entry = manifest['splits'][name]
        path = resolve(entry['path'], manifest_path.parent)
        if sha(path) != entry['sha256']:
            raise ValueError(f'Dataset digest mismatch: {path}')
        rr = pd.read_parquet(path).to_dict('records')
        if len(rr) != count or entry['count'] != count:
            raise ValueError(f'{name} must have {count} questions for this pilot')
        ids = [r['extra_info']['task_id'] for r in rr]
        if len(set(ids)) != count or seen.intersection(ids):
            raise ValueError(f'Duplicate/overlapping questions in {name}')
        for r in rr:
            question = r['extra_info']['question']
            if hashlib.sha256(question.strip().encode()).hexdigest() != r['extra_info']['task_id']:
                raise ValueError(f'Question identity mismatch in {name}')
            if len(r['reward_model']['ground_truth']['target']) != 1:
                raise ValueError('Expected exactly one strict boxed target')
        seen.update(ids)
        splits[name] = {'path': str(path), 'sha256': sha(path), 'count': count}
    gpus = [str(g) for g in spec.get('pool_gpus', ['1'])]
    gpu = str(spec.get('training_gpu', '0'))
    if not gpus or len(set(gpus)) != len(gpus) or not gpu.isdigit():
        raise ValueError('Provide distinct pool_gpus and a numeric local training_gpu')
    if any(not re.fullmatch(r'(node1:)?[0-9]+', g) for g in gpus) or gpu in gpus:
        raise ValueError('Reserve the training GPU separately from persistent sampling replicas')
    lr = float(spec.get('solver_rsft_lr', 1e-7))
    if not 0 < lr <= 1e-3:
        raise ValueError('solver_rsft_lr must be in (0, 1e-3]')
    port = int(spec.get('proposer_port', 18542))
    if not 1024 <= port <= 65535 or port in {*range(20000, 20000+len(gpus)), *range(21000, 21000+len(gpus))}:
        raise ValueError('proposer_port must be unprivileged and separate from sampler ports')
    harness = resolve(spec.get('initial_harness', 'examples/initial_harness.py'))
    from harness_code_contract import validate_source
    validate_source(harness.read_text())
    if run.exists():
        raise FileExistsError(f'Use a fresh run directory: {run}')
    cfg = dict(arms=arms, solver_initial_checkpoint=w['model'], editor_initial_checkpoint=e['model'],
        initial_solver=w, initial_editor=e, initial_harness=str(run/'initial_harness.py'),
        data={'splits': splits}, pool_gpus=gpus, editor_gpu=gpu, solver_gpu=gpu, solver_rsft_lr=lr,
        checkpoint_allowlist=[w['model']]+[str(run/a/'weight_phase/solver_update/round_01/checkpoint') for a in arms],
        train_rollouts=32, train_schedule=[[0,i] for i in range(32)], editor_grpo_updates=4,
        proposal_temperature=1.0, pre_candidates=8, post_candidates=4, harness_phases=2,
        reward_tasks=48, evaluation_tasks=128, seed=42, proposer_port=port,
        logical_harness_rollout_cap_per_phase=624,
        selection='Original WHALE Pareto accuracy/turn rule, incumbent + unique valid post candidates',
        order='H1 -> W1 -> H2; freeze all stages before test',
        limits=['Single seed, draw0; no monotonicity or total-cost superiority claim.',
            'Warm-start solver/editor training costs are separate and must be reported.',
            'Only post candidates and incumbent enter final selection; see docs/METHOD.md.',
            'External arm requires an active proposer queue consumer.'])
    if check_only:
        print(json.dumps({'checked': True, 'arms': arms, 'data_sizes': {k:v['count'] for k,v in splits.items()},
                          'gpu_launches': 0, 'tensor_hashes_rechecked': False}))
        return cfg
    run.mkdir(parents=True)
    shutil.copy2(harness, run/'initial_harness.py')
    write_json(run/'config.json', cfg)
    write_json(run/'PROTOCOL.json', {'config_sha256': sha(run/'config.json'),
        'source_spec_sha256': sha(spec_path), 'test_feedback_used_for_selection': False,
        'phase_points': ['start', 'H1', 'W1', 'H2'], 'prepared_without_model_calls': True})
    snapshot = run/'source_snapshots/release'; snapshot.mkdir(parents=True)
    for path in (ROOT/'whale_local').glob('*.py'):
        shutil.copy2(path, snapshot/path.name)
    write_json(run/'STATUS.json', {'stage': 'prepared'})
    print(json.dumps({'run': str(run), 'arms': arms, 'gpu_launches': 0}))
    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    prepare(args.spec.resolve(), args.run.resolve(), args.check_only)
