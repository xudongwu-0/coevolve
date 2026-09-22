"""Audit development mean@8 of all four frozen weight/harness combinations."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, sha, write_json
from verify_math_run import read, jsonl, native_em_scope, recheck_rewards
from math_context_audit import recheck_context


def change(before, after):
    delta = after - before
    samples = np.random.default_rng(20260915).choice(delta, size=(20000, len(delta)), replace=True)
    return {'difference_pp': float(100 * delta.mean()),
        'paired_question_bootstrap_95_percent_pp': (100 * np.quantile(samples.mean(axis=1), [.025, .975])).tolist(),
        'question_means_increased': int((delta > 0).sum()),
        'question_means_decreased': int((delta < 0).sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = read(root / 'PROTOCOL.json')
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest
        assert sha(root / 'source_snapshot' / Path(path).relative_to(ROOT)) == digest
    reference = Path(protocol['H0_diagnostic'])
    assert sha(reference / 'verification.json') == protocol['H0_audit_sha256']
    ref_audit, ref_protocol = read(reference / 'verification.json'), read(reference / 'PROTOCOL.json')
    assert ref_audit['passed_audit']
    assert protocol['settings'] == ref_protocol['settings']
    assert protocol['data'] == ref_protocol['data']
    assert sha(protocol['harness']) == protocol['harness_sha256']
    entry = protocol['data']['splits']['mh_val']
    assert sha(entry['path']) == entry['sha256']
    metadata = pd.read_parquet(entry['path']).to_dict('records')
    task_ids = [row['extra_info']['task_id'] for row in metadata]
    assert len(task_ids) == 48
    means, cells, all_ids = {}, {}, set()
    for name in ['W0_H0', 'W1_H0', 'W0_Hf', 'W1_Hf']:
        old = name.endswith('H0')
        run = (reference if old else root) / name
        spec = (ref_protocol if old else protocol)['cells'][name]
        assert spec['weight_digest'] == ref_protocol['cells'][name.replace('Hf', 'H0')]['weight_digest']
        assert read(run / 'complete.json')['frozen_weights_verified']
        assert read(run / 'identity.json')['weight_digest'] == spec['weight_digest']
        report, rows = read(run / 'mh_val/val.json'), jsonl(run / 'mh_val/trajectories.jsonl')
        expected_harness = sha(ROOT / 'WHALE/domains/math_reasoning/environments/retool/base_harness.py') if old else protocol['harness_sha256']
        assert report['metadata']['harness_sha256'] == expected_harness
        assert report['metadata']['weight_digest'] == spec['weight_digest']
        assert report['metadata']['dataset_path'] == entry['path']
        assert report['metadata']['temperature'] == 1.0 and report['metadata']['seed'] == 42
        schedule = [[draw, index] for draw in range(8) for index in range(48)]
        assert report['metadata']['sampling_schedule'] == schedule
        assert [[row['draw'], row['example_index']] for row in rows] == schedule
        assert len(rows) == report['num_examples'] == 384
        recheck_rewards(rows, entry, native_em_scope())
        calls, contexts = [], set()
        for row in rows:
            assert row['answer_redacted'] and row['task_id'] == task_ids[row['example_index']]
            assert row['rollout_seed'] == 42 * 100003 + row['draw'] * 1009 + row['example_index']
            assert row['sampling_identity']['weight_digest'] == spec['weight_digest']
            assert row['harness_sha256'] == expected_harness
            assert row['assistant_tokens'] == sum(len(c['response_ids']) for c in row['policy_calls']) <= 2048
            if row.get('context_rejection_id'):
                identifier = row['context_rejection_id']
                assert identifier not in contexts
                contexts.add(identifier)
                recheck_context(run / 'context_rejections' / (identifier + '.json'))
            for call in row['policy_calls']:
                assert call['call_id'] not in all_ids
                all_ids.add(call['call_id'])
                assert call == read(run / 'calls' / (call['call_id'] + '.json'))
                assert call['sampling_identity']['weight_digest'] == spec['weight_digest']
                assert len(call['response_ids']) == len(call['response_logprobs'])
                assert len(call['response_ids']) <= call['requested_max_tokens'] <= 1024
                assert len(call['prompt_ids']) + call['requested_max_tokens'] <= 4096
                assert call['stop_at_chat_end']
                calls.append(call)
        assert {p.stem for p in run.rglob('calls/*.json')} == {c['call_id'] for c in calls}
        assert {p.stem for p in run.rglob('context_rejections/*.json')} == contexts
        rewards = np.asarray([row['reward'] for row in rows], dtype=float).reshape(8, 48)
        assert set(np.unique(rewards)) <= {0, 1}
        assert int(rewards.sum()) == report['solved_examples']
        means[name] = rewards.mean(axis=0)
        cells[name] = {'mean_at_8_percent': float(100 * rewards.mean()),
            'successful_draws': int(rewards.sum()), 'total_draws': 384, 'questions': 48,
            'policy_calls': len(calls), 'generated_tokens': sum(len(c['response_ids']) for c in calls),
            'prompt_tokens': sum(len(c['prompt_ids']) for c in calls),
            'context_limited_trajectories': len(contexts), 'new_generations_in_this_run': not old}
        for path in run.rglob('services.json'):
            started, stopped = read(path), read(path.with_name('services_stopped.json'))
            for service in ['worker', 'sandbox']:
                assert started[service + '_pid'] == stopped[service + '_pid']
                assert stopped[service + '_returncode'] in {0, -15}
        assert not list(run.rglob('training_progress.json'))
    contrasts = {'weights_at_base_harness': change(means['W0_H0'], means['W1_H0']),
        'weights_at_final_harness': change(means['W0_Hf'], means['W1_Hf']),
        'harness_at_final_weights': change(means['W1_H0'], means['W1_Hf']),
        'harness_at_base_weights': change(means['W0_H0'], means['W0_Hf']),
        'joint': change(means['W0_H0'], means['W1_Hf'])}
    promising = all(contrasts[key]['difference_pp'] >= 5 for key in
                     ['weights_at_final_harness', 'harness_at_final_weights'])
    result = {'passed_audit': True, 'cells': cells, 'contrasts': contrasts,
        'development_both_axes_above_5pp': promising, 'qualified_for_gate': False,
        'scope': 'Exploratory development factorial check; fresh confirmation still required',
        'new_optimizer_updates': 0, 'confirmation_used': False, 'test_used': False,
        'new_trajectories_this_run': 768, 'H0_trajectories_reused_for_analysis': 768,
        'H0_audit_sha256': protocol['H0_audit_sha256'], 'auditor_sha256': sha(__file__),
        'protocol_sha256': sha(root / 'PROTOCOL.json')}
    write_json(root / 'verification.json', result)
    lines = ['# 开发集 mean@8 四格诊断', '',
        '48 道原开发题，每题 8 次温度 1.0 采样；权重与 harness 均已冻结。'
        '这是 mean@8，不是至少一次成功的 pass@8。', '',
        '| 组合 | 成功采样 / 总采样 | mean@8 |', '|---|---:|---:|']
    lines += [f"| {name} | {value['successful_draws']}/384 | {value['mean_at_8_percent']:.2f}% |"
              for name, value in cells.items()]
    lines += ['', '| 配对比较 | 变化 pp | 按题 bootstrap 95% 区间 |', '|---|---:|---|']
    lines += [f"| {name} | {value['difference_pp']:+.2f} | {value['paired_question_bootstrap_95_percent_pp']} |"
              for name, value in contrasts.items()]
    lines += ['', f'开发双轴均超过 5 pp：{promising}。独立 gate 资格：未取得。',
              '没有新增训练、确认集或最终测试集使用。H0 两格引用已审计的前一项诊断，未重复生成。', '']
    (root / 'REPORT.md').write_text('\n'.join(lines))
    write_json(root / 'STATUS.json', {'stage': 'complete_audited_development_only',
        'development_both_axes_above_5pp': promising, 'gate_started': False,
        'confirmation_used': False, 'test_used': False})
    print(result)


if __name__ == '__main__':
    main()
