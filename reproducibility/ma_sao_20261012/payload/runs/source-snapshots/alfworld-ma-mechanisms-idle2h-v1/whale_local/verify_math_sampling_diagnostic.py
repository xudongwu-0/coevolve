"""Audit the frozen development sampling diagnosis, without gate qualification."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import ROOT, sha, write_json
from verify_math_run import read, jsonl, native_em_scope, recheck_rewards
from math_context_audit import recheck_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    protocol = read(root / 'PROTOCOL.json')
    assert protocol['settings']['draws_per_question'] == 8
    assert protocol['settings']['temperature'] == 1.0
    assert protocol['settings']['allowed_split'] == 'mh_val'
    parent = Path(protocol['parent'])
    assert sha(parent / 'verification.json') == protocol['parent_audit_sha256']
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest
        assert sha(root / 'source_snapshot' / Path(path).relative_to(ROOT)) == digest
    entry = protocol['data']['splits']['mh_val']
    assert sha(entry['path']) == entry['sha256'] and entry['count'] == 48
    metadata = pd.read_parquet(entry['path']).to_dict('records')
    task_ids = [row['extra_info']['task_id'] for row in metadata]
    scope, cells, question_scores, all_calls = native_em_scope(), {}, {}, set()
    for name in ['W0_H0', 'W1_H0']:
        run = root / name
        assert read(run / 'complete.json')['frozen_weights_verified']
        assert read(run / 'config.json')['data'] == protocol['data']
        assert read(run / 'identity.json')['weight_digest'] == protocol['cells'][name]['weight_digest']
        report = read(run / 'mh_val/val.json')
        rows = jsonl(run / 'mh_val/trajectories.jsonl')
        assert len(rows) == report['num_examples'] == 384
        assert report['metadata']['temperature'] == 1.0
        assert report['metadata']['seed'] == 42
        assert report['metadata']['dataset_path'] == entry['path']
        assert report['metadata']['harness_sha256'] == sha(
            ROOT / 'WHALE/domains/math_reasoning/environments/retool/base_harness.py')
        schedule = [[draw, index] for draw in range(8) for index in range(48)]
        assert report['metadata']['sampling_schedule'] == schedule
        assert [[row['draw'], row['example_index']] for row in rows] == schedule
        recheck_rewards(rows, entry, scope)
        calls, contexts = [], set()
        for row in rows:
            assert row['task_id'] == task_ids[row['example_index']]
            assert row['answer_redacted']
            assert row['rollout_seed'] == 42 * 100003 + row['draw'] * 1009 + row['example_index']
            assert row['sampling_identity']['weight_digest'] == protocol['cells'][name]['weight_digest']
            assert row['harness_sha256'] == report['metadata']['harness_sha256']
            assert row['assistant_tokens'] == sum(len(c['response_ids']) for c in row['policy_calls']) <= 2048
            if row.get('context_rejection_id'):
                identifier = row['context_rejection_id']
                assert identifier not in contexts
                contexts.add(identifier)
                recheck_context(run / 'context_rejections' / (identifier + '.json'))
            for call in row['policy_calls']:
                assert call['call_id'] not in all_calls
                all_calls.add(call['call_id'])
                assert call == read(run / 'calls' / (call['call_id'] + '.json'))
                assert call['sampling_identity']['weight_digest'] == row['sampling_identity']['weight_digest']
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
        question_scores[name] = rewards.mean(axis=0)
        cells[name] = {'mean_at_8_percent': float(100 * rewards.mean()),
            'successful_draws': int(rewards.sum()), 'total_draws': 384, 'questions': 48,
            'policy_calls': len(calls), 'generated_tokens': sum(len(c['response_ids']) for c in calls),
            'prompt_tokens': sum(len(c['prompt_ids']) for c in calls),
            'context_limited_trajectories': len(contexts)}
        for path in run.rglob('services.json'):
            started, stopped = read(path), read(path.with_name('services_stopped.json'))
            for service in ['worker', 'sandbox']:
                assert started[service + '_pid'] == stopped[service + '_pid']
                assert stopped[service + '_returncode'] in {0, -15}
        assert not list(run.rglob('training_progress.json'))
    delta = question_scores['W1_H0'] - question_scores['W0_H0']
    rng = np.random.default_rng(20260915)
    bootstrap = rng.choice(delta, size=(20000, 48), replace=True).mean(axis=1)
    contrast = {'difference_pp': float(100 * delta.mean()),
        'paired_question_bootstrap_95_percent_pp': (100 * np.quantile(bootstrap, [.025, .975])).tolist(),
        'question_means_increased': int((delta > 0).sum()),
        'question_means_decreased': int((delta < 0).sum())}
    result = {'passed_audit': True, 'scope': 'Exploratory development decoding diagnostic',
        'cells': cells, 'weight_contrast': contrast,
        'new_optimizer_updates': 0, 'confirmation_used': False, 'test_used': False,
        'qualified_for_gate': False, 'reason': 'Development reuse cannot qualify the gate',
        'protocol_sha256': sha(root / 'PROTOCOL.json'), 'auditor_sha256': sha(__file__)}
    write_json(root / 'verification.json', result)
    (root / 'REPORT.md').write_text('\n'.join([
        '# 开发集采样诊断', '',
        '这是已用于开发的 48 题，固定原始 harness，每题采样 8 次。'
        '指标是 8 次单次准确率的平均值，不能解读为至少一次成功的 pass@8。', '',
        '| 权重 | 成功采样 / 总采样 | mean@8 |', '|---|---:|---:|',
        *[f"| {name} | {value['successful_draws']}/384 | {value['mean_at_8_percent']:.2f}% |"
          for name, value in cells.items()], '',
        f"权重变化：{contrast['difference_pp']:+.2f} pp；按题配对 bootstrap 95% 区间："
        f"{contrast['paired_question_bootstrap_95_percent_pp']}。", '',
        '该结果仅诊断解码方式下的变化；无新训练、无确认集或最终 test 使用，不能据此启动 gate。', '']))
    write_json(root / 'STATUS.json', {'stage': 'complete_audited_development_only',
        'gate_started': False, 'confirmation_used': False, 'test_used': False})
    print(result)


if __name__ == '__main__':
    main()
