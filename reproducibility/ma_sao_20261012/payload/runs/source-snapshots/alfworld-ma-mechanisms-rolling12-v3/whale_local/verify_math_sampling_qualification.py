"""Audit a fresh mean@8 confirmation of a byte-identical native WHALE prefix."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, sha, write_json
from verify_math_run import read, jsonl, native_em_scope, recheck_rewards
from math_context_audit import recheck_context
from math_reward import native


def change(before, after):
    delta = after - before
    boot = np.random.default_rng(20260915).choice(delta, size=(20000, len(delta)), replace=True)
    return {'difference_pp': float(100 * delta.mean()),
        'paired_question_bootstrap_95_percent_pp': (100 * np.quantile(boot.mean(axis=1), [.025, .975])).tolist(),
        'question_means_increased': int((delta > 0).sum()),
        'question_means_decreased': int((delta < 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    config, protocol = read(run / 'config.json'), read(run / 'PROTOCOL.json')
    assert not config['gate_enabled'] and config['rounds'] == 1
    assert config['evaluation_rollouts'] == 8 and config['evaluation_temperature'] == 1.0
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT / name) == sha(run / 'source_snapshot' / name) == digest
    assert sha(protocol['conditional_protocol']) == protocol['conditional_protocol_sha256']
    conditional = read(protocol['conditional_protocol'])
    assert sha(protocol['development_audit']) == protocol['development_audit_sha256']
    development = read(protocol['development_audit'])
    assert development['passed_audit'] and development['development_both_axes_above_5pp']
    assert Path(protocol['conditional_protocol']).stat().st_mtime < Path(protocol['development_audit']).stat().st_mtime
    assert sha(protocol['data_manifest']) == protocol['data_manifest_sha256']
    assert config['data'] == read(protocol['data_manifest'])
    prefix = read(run / 'prefix_reuse.json')
    parent = Path(prefix['parent_run'])
    assert sha(parent / 'verification.json') == prefix['parent_audit_sha256'] == conditional['reference_audit_sha256']
    parent_audit = read(parent / 'verification.json')
    assert parent_audit['passed_audit'] and parent_audit['fp32_master_and_moments']
    assert parent_audit['preregistered_conditional_first_cycle_verified']
    assert prefix['completed_rounds'] == 1 and prefix['new_rollouts_in_prefix'] == 0
    for relative, digest in prefix['original_relative_file_sha256'].items():
        assert sha(run / relative) == sha(parent / relative) == digest
    relocation = read(run / 'MH_PATH_RELOCATION.json')
    assert relocation['original'] == read(parent / 'round_01/mh_result.json')
    expected = dict(relocation['original'])
    expected['harness_path'] = str(run / Path(expected['harness_path']).relative_to(parent))
    expected['evaluation_files'] = [str(run / Path(path).relative_to(parent)) for path in expected['evaluation_files']]
    assert expected == relocation['relocated'] == read(run / 'round_01/mh_result.json')
    keys = {}
    for split, entry in config['data']['splits'].items():
        assert sha(entry['path']) == entry['sha256']
        keys[split] = {info['task_id'] for info in pd.read_parquet(entry['path'])['extra_info']}
        assert len(keys[split]) == entry['count']
    assert len(set.union(*keys.values())) == sum(map(len, keys.values()))
    for previous in [parent, ROOT / 'runs/whale_math_fixed_native_20260915']:
        previous_keys = {info['task_id'] for entry in read(previous / 'config.json')['data']['splits'].values()
                         for info in pd.read_parquet(entry['path'])['extra_info']}
        assert not keys['confirmation'].intersection(previous_keys)
    entry = config['data']['splits']['confirmation']
    metadata = pd.read_parquet(entry['path']).to_dict('records')
    source_entry = config['data']['source_files']['test']
    assert sha(source_entry['path']) == source_entry['sha256']
    originals = {hashlib.sha256(row['problem'].strip().encode()).hexdigest(): row
                 for row in pd.read_parquet(source_entry['path']).to_dict('records')}
    for row in metadata:
        original = originals[row['extra_info']['task_id']]
        assert original['problem'] == row['extra_info']['question']
        assert original['level'] in {'Level 2', 'Level 3', 'Level 4'}
        target = native.remove_boxed(native.last_boxed_only_string(original['solution'])).strip()
        assert row['reward_model']['ground_truth']['target'] == [target]
    task_ids = [row['extra_info']['task_id'] for row in metadata]
    assert len(task_ids) == 128
    assert sha(run / 'frozen_evaluation.json') == protocol['frozen_evaluation_sha256']
    frozen = read(run / 'frozen_evaluation.json')
    assert frozen['W1']['weight_digest'] == conditional['candidate_weight_digest']
    assert sha(frozen['Hf']) == conditional['candidate_harness_sha256']
    assert frozen['W0']['weight_digest'] == read(run / 'initial_identity.json')['weight_digest']
    training = read(run / 'round_01/checkpoint/training_state.json')
    assert training['weight_digest'] == frozen['W1']['weight_digest']
    means, cells, costs = {}, {}, {'trajectories': 0, 'policy_calls': 0, 'generated_tokens': 0, 'prompt_tokens': 0}
    all_ids = set()
    by_weight = {weight: {'calls': set(), 'contexts': set()} for weight in ['W0', 'W1']}
    identities = {}
    for weight in ['W0', 'W1']:
        worker = run / 'frozen_eval_workers' / weight
        assert read(worker / 'complete.json')['frozen_weights_verified']
        identities[weight] = read(worker / 'identity.json')
        assert identities[weight]['weight_digest'] == frozen[weight]['weight_digest']
        assert identities[weight]['optimizer_steps'] == ([] if weight == 'W0' else training['optimizer_steps_after'])
        for harness_name in ['H0', 'Hf']:
            name = weight + '_' + harness_name
            report, rows = read(run / 'confirmation' / name / 'val.json'), jsonl(run / 'confirmation' / name / 'trajectories.jsonl')
            assert len(rows) == report['num_examples'] == 1024
            assert report['metadata']['temperature'] == 1.0 and report['metadata']['seed'] == 42
            assert report['metadata']['dataset_path'] == entry['path']
            assert report['metadata']['weight_digest'] == frozen[weight]['weight_digest']
            assert report['metadata']['harness_sha256'] == sha(frozen[harness_name])
            schedule = [[draw, index] for draw in range(8) for index in range(128)]
            assert report['metadata']['sampling_schedule'] == schedule
            assert [[row['draw'], row['example_index']] for row in rows] == schedule
            recheck_rewards(rows, entry, native_em_scope())
            calls = []
            for row in rows:
                assert row['task_id'] == task_ids[row['example_index']] and row['answer_redacted']
                assert row['harness_sha256'] == sha(frozen[harness_name])
                assert row['sampling_identity']['weight_digest'] == frozen[weight]['weight_digest']
                assert row['rollout_seed'] == 42 * 100003 + row['draw'] * 1009 + row['example_index']
                assert row['assistant_tokens'] == sum(len(c['response_ids']) for c in row['policy_calls']) <= 2048
                if row.get('context_rejection_id'):
                    identifier = row['context_rejection_id']
                    assert identifier not in by_weight[weight]['contexts']
                    by_weight[weight]['contexts'].add(identifier)
                    recheck_context(worker / 'context_rejections' / (identifier + '.json'))
                for call in row['policy_calls']:
                    assert call['call_id'] not in all_ids
                    all_ids.add(call['call_id'])
                    by_weight[weight]['calls'].add(call['call_id'])
                    assert call == read(worker / 'calls' / (call['call_id'] + '.json'))
                    assert call['sampling_identity']['weight_digest'] == frozen[weight]['weight_digest']
                    assert len(call['response_ids']) == len(call['response_logprobs'])
                    assert len(call['response_ids']) <= call['requested_max_tokens'] <= 1024
                    assert len(call['prompt_ids']) + call['requested_max_tokens'] <= 4096
                    assert call['stop_at_chat_end']
                    assert not any(token in (248044, 248046) for token in call['response_ids'][:-1])
                    calls.append(call)
            rewards = np.asarray([row['reward'] for row in rows], dtype=float).reshape(8, 128)
            assert set(np.unique(rewards)) <= {0, 1} and int(rewards.sum()) == report['solved_examples']
            means[name] = rewards.mean(axis=0)
            cells[name] = {'mean_at_8_percent': float(100 * rewards.mean()),
                'successful_draws': int(rewards.sum()), 'total_draws': 1024, 'questions': 128}
            costs['trajectories'] += len(rows)
            costs['policy_calls'] += len(calls)
            costs['generated_tokens'] += sum(len(c['response_ids']) for c in calls)
            costs['prompt_tokens'] += sum(len(c['prompt_ids']) for c in calls)
        assert {p.stem for p in worker.rglob('calls/*.json')} == by_weight[weight]['calls']
        assert {p.stem for p in worker.rglob('context_rejections/*.json')} == by_weight[weight]['contexts']
    for path in (run / 'frozen_eval_workers').rglob('services.json'):
        started, stopped = read(path), read(path.with_name('services_stopped.json'))
        for service_name in ['worker', 'sandbox']:
            assert started[service_name + '_pid'] == stopped[service_name + '_pid']
            assert stopped[service_name + '_returncode'] in {0, -15}
    contrasts = {'harness_at_final_weights': change(means['W1_H0'], means['W1_Hf']),
        'weights_at_final_harness': change(means['W0_Hf'], means['W1_Hf']),
        'weights_at_base_harness': change(means['W0_H0'], means['W1_H0']),
        'harness_at_base_weights': change(means['W0_H0'], means['W0_Hf']),
        'joint': change(means['W0_H0'], means['W1_Hf'])}
    qualified = all(contrasts[name]['difference_pp'] >= conditional['minimum_conditional_gain_pp']
                    for name in ['harness_at_final_weights', 'weights_at_final_harness'])
    h0_audit = read(read(Path(conditional['development_run']) / 'PROTOCOL.json')['H0_diagnostic'] + '/verification.json')
    additional = {key: value for key, value in parent_audit.items() if key.startswith('additional_')}
    additional['earlier_onecycle_greedy_confirmation'] = parent_audit['costs']['confirmation']
    additional['H0_development_sampling'] = h0_audit['cells']
    additional['Hf_development_sampling'] = {key: value for key, value in development['cells'].items() if key.endswith('Hf')}
    result = {'passed_audit': True, 'qualified_for_gate': qualified, 'fp32_master_and_moments': True,
        'scores': cells, 'contrasts': contrasts, 'evaluation_metric': 'mean_at_8',
        'minimum_conditional_gain_pp': conditional['minimum_conditional_gain_pp'],
        'native_prefix_exact_reuse_verified': True, 'parent_audit_sha256': prefix['parent_audit_sha256'],
        'training': parent_audit['training'], 'costs': {'training': parent_audit['costs']['training'],
            'harness_search': parent_audit['costs']['harness_search'], 'confirmation': costs},
        'additional_scenario_screening_costs': additional, 'independent_gate_test_not_evaluated': True,
        'new_training_and_MH_generations': 0, 'prior_greedy_failures_preserved': True,
        'source_snapshot_verified': True, 'auditor_sha256': sha(__file__)}
    write_json(run / 'verification.json', result)
    summary = read(parent / 'summary.json')
    summary.update(initial_identity=identities['W0'], final_identity=identities['W1'],
        final_harness=frozen['Hf'], gate_enabled=False, status='complete',
        scope='Exact completed native first-cycle prefix with a new frozen independent mean@8 confirmation',
        elapsed_seconds_native_prefix=summary.get('elapsed_seconds'), evaluation_metric='mean_at_8')
    summary.pop('elapsed_seconds', None)
    write_json(run / 'summary.json', summary)
    write_json(run / 'frozen_final.json', {'model_identity': identities['W1'],
        'harness_path': frozen['Hf'], 'harness_sha256': sha(frozen['Hf'])})
    write_json(run / 'round_01/reload_verification.json', identities['W1'] | {'all_parameter_hashes_equal': True,
        'evidence': 'Byte-identical checkpoint copy plus worker load_model hash equality and frozen health checks'})
    lines = ['# WHALE 独立 mean@8 确认', '',
        f'执行审计通过；双轴资格：{qualified}。128 道新确认题，每题 8 次温度 1.0 采样。', '',
        '| 组合 | 成功采样 / 总采样 | mean@8 |', '|---|---:|---:|']
    lines += [f"| {name} | {value['successful_draws']}/1024 | {value['mean_at_8_percent']:.2f}% |"
              for name, value in cells.items()]
    lines += ['', '| 条件比较 | 变化 pp | 按题配对 bootstrap 95% 区间 |', '|---|---:|---|']
    lines += [f"| {name} | {value['difference_pp']:+.2f} | {value['paired_question_bootstrap_95_percent_pp']} |"
              for name, value in contrasts.items()]
    lines += ['', '训练与原生 MH 完整复用已审计的一轮，逻辑预算为 512 条训练轨迹和 240 条 MH 轨迹。'
              '此前学习率、场景筛选及两组贪心确认成本另列；新的采样指标结果不覆盖旧的负面结果。',
              '这仍是 thinking=False、短预算的本地 WHALE 缩小版，非论文完整复现。最终 test 128 题尚未使用。', '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    write_json(run / 'STATUS.json', {'stage': 'complete_audited', 'qualified_for_gate': qualified,
        'confirmation_used': True, 'final_test_used': False, 'gate_started': False})
    print(result)


if __name__ == '__main__':
    main()
