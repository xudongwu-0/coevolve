"""Prepare a preregistered first-cycle fallback after the primary fails qualification."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

import pandas as pd
from pipeline import sha, write_json


def read(p):
    return json.loads(Path(p).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setup', type=Path, required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    setup, run = args.setup.resolve(), args.run_dir.resolve()
    conditional = read(setup / 'CONDITIONAL_PROTOCOL.json')
    timing = read(setup / 'REGISTRATION_TIMING_VERIFICATION.json')
    assert timing['conditional_protocol_sha256'] == sha(setup / 'CONDITIONAL_PROTOCOL.json')
    assert timing['registered_timestamp'] < timing['final_MH_comparison_timestamp']
    assert sha(timing['final_MH_comparison_path']) == timing['final_MH_comparison_sha256']
    parent = Path(conditional['primary'])
    audit, config = read(parent / 'verification.json'), read(parent / 'config.json')
    assert audit['passed_audit'] and audit['fp32_master_and_moments']
    assert audit['development_selected_training_stage_reuse_verified']
    assert not audit['qualified_for_gate'], 'The primary passed; do not run this fallback'
    assert sha(parent / 'PROTOCOL.json') == conditional['primary_protocol_sha256']
    assert sha(setup / 'manifest.json') == conditional['manifest_sha256']
    data = read(setup / 'manifest.json')
    for key in ['train', 'mh_val', 'test']:
        assert data['splits'][key] == config['data']['splits'][key]
    old_keys = {x['task_id'] for entry in config['data']['splits'].values()
                for x in pd.read_parquet(entry['path'])['extra_info']}
    new_keys = {x['task_id'] for x in pd.read_parquet(data['splits']['confirmation']['path'])['extra_info']}
    assert len(new_keys) == 128 and not old_keys.intersection(new_keys)
    training = read(parent / 'round_01/checkpoint/training_state.json')
    mh = read(parent / 'round_01/mh_result.json')
    assert training['weight_digest'] == conditional['candidate_weights_digest']
    assert mh['selected'] == 'h3' and mh['harness_sha256'] == conditional['candidate_harness_sha256']
    assert not run.exists(), 'Use a fresh directory'
    run.mkdir(parents=True)
    originals = {}

    def copy(relative):
        source, dest = parent / relative, run / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(source, dest)
        digest = sha(source)
        assert sha(dest) == digest
        originals[str(relative)] = digest

    for directory in ['round_01/checkpoint', 'round_01/rollouts', 'mh/round_01']:
        for source in sorted((parent / directory).rglob('*')):
            if source.is_file():
                copy(source.relative_to(parent))
    for name in ['initial_identity.json', 'round_01/accepted_batch.pt', 'round_01/training_result.json']:
        copy(Path(name))
    rows = [json.loads(x) for x in (parent / 'round_01/rollouts/trajectories.jsonl').read_text().splitlines()]
    for path in mh['evaluation_files']:
        rows += [json.loads(x) for x in Path(path).with_name('trajectories.jsonl').read_text().splitlines()]
    for row in rows:
        for call in row['policy_calls']:
            copy(Path('calls') / (call['call_id'] + '.json'))
        if row.get('context_rejection_id'):
            copy(Path('context_rejections') / (row['context_rejection_id'] + '.json'))
    relocated = dict(mh)
    relocated['harness_path'] = str(run / Path(mh['harness_path']).relative_to(parent))
    relocated['evaluation_files'] = [str(run / Path(p).relative_to(parent)) for p in mh['evaluation_files']]
    write_json(run / 'round_01/mh_result.json', relocated)
    write_json(run / 'MH_PATH_RELOCATION.json', {'original': mh, 'relocated': relocated,
        'only_changed_fields': ['harness_path', 'evaluation_files']})
    selection = {'passed_audit': True, 'selection_split': 'mh_val', 'selected_round': 1,
        'rule': 'Preregistered first native cycle, conditionally confirmed only after primary failure',
        'conditional_protocol': str(setup / 'CONDITIONAL_PROTOCOL.json'),
        'conditional_protocol_sha256': sha(setup / 'CONDITIONAL_PROTOCOL.json'),
        'registration_timing': str(setup / 'REGISTRATION_TIMING_VERIFICATION.json'),
        'registration_timing_sha256': sha(setup / 'REGISTRATION_TIMING_VERIFICATION.json'),
        'parent_audit_sha256': sha(parent / 'verification.json'),
        'selected_weight_digest': training['weight_digest'], 'selected_harness_sha256': mh['harness_sha256']}
    write_json(run / 'development_selection.json', selection)
    write_json(run / 'prefix_reuse.json', {'parent_run': str(parent),
        'parent_audit_sha256': sha(parent / 'verification.json'), 'completed_rounds': 1,
        'new_rollouts_in_prefix': 0, 'reuse_mode': 'completed_stopping_point',
        'development_selection': {'path': str(run / 'development_selection.json'),
                                  'sha256': sha(run / 'development_selection.json')},
        'original_relative_file_sha256': originals, 'confirmation_from_parent_not_reused': True})
    hardware = read(parent / 'eval_hardware.json')
    write_json(run / 'eval_hardware.json', hardware)
    protocol = read(parent / 'PROTOCOL.json')
    names = list(protocol['source_sha256']) + ['whale_local/prepare_math_onecycle_baseline.py',
                                             'whale_local/verify_math_onecycle_run.py']
    protocol.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='registered_conditional_first_cycle_confirmation', rounds=1,
        total_training_trajectories=512, total_mh_trajectories=240,
        manifest_sha256=sha(setup / 'manifest.json'), hardware=hardware,
        formal_start_weights='original base; reuse the complete audited first native cycle with its full logical budget',
        conditional_protocol=selection['conditional_protocol'],
        conditional_protocol_sha256=selection['conditional_protocol_sha256'],
        source_sha256={name: sha(root / name) for name in names})
    write_json(run / 'PROTOCOL.json', protocol)
    for name, digest in protocol['source_sha256'].items():
        dest = run / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
        assert sha(dest) == digest
    (run / 'PROTOCOL.md').write_text(
        '# WHALE 第一轮独立确认\n\n仅在两轮主实验审计通过但双轴资格失败后启动。'
        '完整复用已保存的第一轮训练和原生 MH；512 条训练与240 条 MH 均计入本轮逻辑预算，'
        '没有重新生成。使用提前准备的另一批128 题确认，排除主实验所有分区；'
        '两个条件增益仍须各达到5 个百分点。最终 test 保留。\n')
    command = [str(root / '.venv-whale-search/bin/python'), '-u', str(root / 'whale_local/math_pipeline.py'),
        '--run-dir', str(run), '--manifest', str(setup / 'manifest.json'), '--gpu', '8',
        '--policy-port', '30041', '--proposer-port', '30042', '--rounds', '1', '--rollouts', '4',
        '--learning-rate', str(config['learning_rate']), '--parameter-dtype', 'float32', '--resume']
    write_json(run / 'launch_command.json', {'argv': command,
        'PYTHONPATH': str(root / 'whale_local') + ':' + str(root / 'WHALE/domains/math_reasoning')})
    print(json.dumps({'prepared': str(run), 'cached_training_trajectories': 512,
                      'cached_MH_trajectories': 240, 'new_generations': 0}))


if __name__ == '__main__':
    main()
