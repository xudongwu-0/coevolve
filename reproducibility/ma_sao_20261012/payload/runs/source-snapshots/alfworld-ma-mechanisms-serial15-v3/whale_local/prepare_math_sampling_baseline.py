"""Materialize the frozen native prefix only after the mean@8 dev guard passes."""
import argparse
import datetime
import json
from pathlib import Path
import shutil

import pandas as pd

from pipeline import ROOT, sha, write_json
from verify_math_run import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    setup, run = args.setup.resolve(), args.run_dir.resolve()
    assert not run.exists()
    conditional = read(setup / 'CONDITIONAL_PROTOCOL.json')
    development = Path(conditional['development_run'])
    dev_audit = read(development / 'verification.json')
    assert dev_audit['passed_audit'] and dev_audit['development_both_axes_above_5pp']
    assert not dev_audit['qualified_for_gate']
    assert sha(development / 'PROTOCOL.json') == conditional['development_protocol_sha256']
    assert (setup / 'CONDITIONAL_PROTOCOL.json').stat().st_mtime < (development / 'verification.json').stat().st_mtime
    parent = Path(conditional['reference_native_prefix'])
    parent_audit = read(parent / 'verification.json')
    assert sha(parent / 'verification.json') == conditional['reference_audit_sha256']
    assert parent_audit['passed_audit'] and parent_audit['fp32_master_and_moments']
    assert parent_audit['preregistered_conditional_first_cycle_verified']
    assert sha(setup / 'manifest.json') == conditional['manifest_sha256']
    config, data = read(parent / 'config.json'), read(setup / 'manifest.json')
    for name in ['train', 'mh_val', 'test']:
        assert data['splits'][name] == config['data']['splits'][name]
    new_keys = {info['task_id'] for info in pd.read_parquet(data['splits']['confirmation']['path'])['extra_info']}
    for previous in [parent, ROOT / 'runs/whale_math_fixed_native_20260915']:
        old = {info['task_id'] for entry in read(previous / 'config.json')['data']['splits'].values()
               for info in pd.read_parquet(entry['path'])['extra_info']}
        assert not new_keys.intersection(old)
    assert len(new_keys) == 128
    training, mh = read(parent / 'round_01/checkpoint/training_state.json'), read(parent / 'round_01/mh_result.json')
    assert training['weight_digest'] == conditional['candidate_weight_digest']
    assert mh['harness_sha256'] == conditional['candidate_harness_sha256']
    run.mkdir(parents=True)
    originals = {}

    def copy(relative):
        source, destination = parent / relative, run / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
        digest = sha(source)
        assert sha(destination) == digest
        originals[str(relative)] = digest

    for directory in ['round_01/checkpoint', 'round_01/rollouts', 'mh/round_01']:
        for path in sorted((parent / directory).rglob('*')):
            if path.is_file():
                copy(path.relative_to(parent))
    for name in ['initial_identity.json', 'round_01/accepted_batch.pt', 'round_01/training_result.json']:
        copy(Path(name))
    rows = [json.loads(line) for line in (parent / 'round_01/rollouts/trajectories.jsonl').read_text().splitlines()]
    for path in mh['evaluation_files']:
        rows += [json.loads(line) for line in Path(path).with_name('trajectories.jsonl').read_text().splitlines()]
    for row in rows:
        for call in row['policy_calls']:
            copy(Path('calls') / (call['call_id'] + '.json'))
        if row.get('context_rejection_id'):
            copy(Path('context_rejections') / (row['context_rejection_id'] + '.json'))
    relocated = dict(mh)
    relocated['harness_path'] = str(run / Path(mh['harness_path']).relative_to(parent))
    relocated['evaluation_files'] = [str(run / Path(path).relative_to(parent)) for path in mh['evaluation_files']]
    write_json(run / 'round_01/mh_result.json', relocated)
    write_json(run / 'MH_PATH_RELOCATION.json', {'original': mh, 'relocated': relocated,
        'only_changed_fields': ['harness_path', 'evaluation_files']})
    write_json(run / 'prefix_reuse.json', {'parent_run': str(parent),
        'parent_audit_sha256': sha(parent / 'verification.json'), 'completed_rounds': 1,
        'new_rollouts_in_prefix': 0, 'reuse_mode': 'completed_stopping_point_new_sampling_metric',
        'original_relative_file_sha256': originals, 'confirmation_from_parent_not_reused': True})
    config.update(data=data, rounds=1, gate_enabled=False, evaluation_temperature=1.0,
        evaluation_rollouts=8, evaluation_metric='mean_at_8', minimum_conditional_gain_pp=5.0)
    write_json(run / 'config.json', config)
    hardware = {'W0': {'gpu': 6, 'replica_gpus': [7, 'node1:4', 'node1:5', 'node1:6'],
                      'port': 30101, 'replica_port_base': 21100},
                'W1': {'gpu': 8, 'replica_gpus': [9, 'node1:7', 'node1:8', 'node1:9'],
                      'port': 30102, 'replica_port_base': 21300}}
    write_json(run / 'eval_hardware.json', hardware)
    frozen = {'W0': {'model': config['base_model'],
                     'weight_digest': read(run / 'initial_identity.json')['weight_digest']},
              'W1': {'model': str(run / 'round_01/checkpoint'), 'weight_digest': training['weight_digest']},
              'H0': str(ROOT / 'WHALE/domains/math_reasoning/environments/retool/base_harness.py'),
              'Hf': relocated['harness_path']}
    write_json(run / 'frozen_evaluation.json', frozen)
    # The copied native prefix is covered by its parent's complete execution
    # audit and byte hashes above. The new executable phase runs only frozen
    # sampling and confirmation auditing; it never imports these gate/MH tools.
    # Keep future gate implementation independent of this running confirmation.
    not_executed = {'whale_local/math_pipeline.py', 'whale_local/math_evaluation.py',
        'whale_local/math_gate.py', 'whale_local/verify_math_gate.py',
        'whale_local/verify_math_gate_complete.py', 'whale_local/math_common_probe.py',
        'whale_local/run_math_mh.py'}
    names = [name for name in read(parent / 'PROTOCOL.json')['source_sha256']
             if name not in not_executed] + [
        'whale_local/prepare_math_sampling_confirmation.py', 'whale_local/prepare_math_sampling_baseline.py',
        'whale_local/math_sampling_qualification.py', 'whale_local/verify_math_sampling_qualification.py',
        'whale_local/assess_viability.py',
        'WHALE/domains/math_reasoning/verl/experimental/agent_loop/tool_parser.py',
        'WHALE/domains/math_reasoning/verl/tools/schemas.py']
    protocol = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'New independent mean@8 confirmation of the complete frozen first native WHALE cycle',
        'conditional_protocol': str(setup / 'CONDITIONAL_PROTOCOL.json'),
        'conditional_protocol_sha256': sha(setup / 'CONDITIONAL_PROTOCOL.json'),
        'development_audit': str(development / 'verification.json'),
        'development_audit_sha256': sha(development / 'verification.json'),
        'data_manifest': str(setup / 'manifest.json'), 'data_manifest_sha256': sha(setup / 'manifest.json'),
        'frozen_evaluation_sha256': sha(run / 'frozen_evaluation.json'),
        'source_sha256': {name: sha(ROOT / name) for name in names},
        'nonexecuted_gate_and_MH_tools_excluded_from_confirmation_scope': sorted(not_executed),
        'native_prefix_execution_source_evidence': str(parent / 'verification.json'),
        'hardware': hardware, 'logical_training_trajectories': 512, 'logical_MH_trajectories': 240,
        'new_confirmation_trajectories': 4096, 'final_test_used': False, 'gate_started': False}
    write_json(run / 'PROTOCOL.json', protocol)
    for name, digest in protocol['source_sha256'].items():
        destination = run / 'source_snapshot' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
        assert sha(destination) == digest
    write_json(run / 'STATUS.json', {'stage': 'prepared_frozen_sampling_confirmation',
        'confirmation_used': False, 'final_test_used': False, 'gate_started': False})
    print({'prepared': str(run), 'new_generations': 0, 'cached_training': 512, 'cached_MH': 240})


if __name__ == '__main__':
    main()
