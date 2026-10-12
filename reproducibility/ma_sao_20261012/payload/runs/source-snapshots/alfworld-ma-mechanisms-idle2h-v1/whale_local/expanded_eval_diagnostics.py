"""Descriptive length diagnostics; does not generate or select any samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from pipeline import sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def token_signature(row):
    return hashlib.sha256(json.dumps([c['response_ids'] for c in row['policy_calls']]).encode()).hexdigest()


def main(run):
    cfg = read(run / 'config.json')
    assert read(run / 'verification.json')['passed_audit']
    old_curve = read(Path(cfg['reference']) / 'monitor_curve.json')
    result, repeat = {}, {}
    for name in cfg['model_order']:
        counts = Counter()
        current_prefix = {}
        for cohort in cfg['cohort_order']:
            for r in rows(run / 'eval' / name / cohort / 'trajectories.jsonl'):
                failed = not bool(r['reward'])
                total_cap = r['assistant_tokens'] >= cfg['assistant_token_budget']
                context = bool(r.get('context_rejection_id'))
                last_call_cap = bool(r['policy_calls']) and len(r['policy_calls'][-1]['response_ids']) >= r['policy_calls'][-1]['requested_max_tokens']
                counts.update(attempts=1, failures=int(failed), assistant_tokens=r['assistant_tokens'],
                    total_output_cap_hits=int(total_cap), context_rejections=int(context),
                    last_call_at_requested_cap=int(last_call_cap),
                    failures_with_any_budget_flag=int(failed and (total_cap or context or last_call_cap)),
                    failures_without_boxed_output=int(failed and '\\boxed{' not in r['last_assistant']))
                if cohort == 'reused_monitor32' and r['draw'] < 2:
                    current_prefix[(r['draw'], r['example_index'])] = {
                        'reward': r['reward'], 'tokens': token_signature(r), 'task_id': r['task_id'],
                        'seed': r['rollout_seed'], 'weight_digest': r['sampling_identity']['weight_digest'],
                        'harness_sha256': r['harness_sha256']}
        assert counts['attempts'] == 1384
        result[name] = dict(counts, mean_assistant_tokens=counts['assistant_tokens'] / counts['attempts'])
        prior = next(p for p in old_curve if p['weight_digest'] == cfg['models'][name]['weight_digest'])
        previous = {(r['draw'], r['example_index']): r for r in rows(prior['trajectories']) if r['draw'] < 2}
        assert len(previous) == len(current_prefix) == 64
        equal_reward = equal_tokens = 0
        for key, old in previous.items():
            cur = current_prefix[key]
            assert cur['task_id'] == old['task_id'] and cur['seed'] == old['rollout_seed']
            assert cur['weight_digest'] == old['sampling_identity']['weight_digest']
            assert cur['harness_sha256'] == old['harness_sha256']
            equal_reward += cur['reward'] == old['reward']
            equal_tokens += cur['tokens'] == token_signature(old)
        repeat[name] = {'matched_attempts': 64, 'same_reward': equal_reward,
            'same_response_token_sequences': equal_tokens,
            'previous_successes': sum(r['reward'] for r in previous.values()),
            'current_successes': sum(r['reward'] for r in current_prefix.values()),
            'prior_trajectories_sha256': sha(prior['trajectories'])}
    out = {'descriptive_only': True, 'same_fixed_generation_limits': True,
        'model_diagnostics': result, 'original_first_two_draws_repetition': repeat,
        'verification_sha256': sha(run / 'verification.json'), 'analysis_source_sha256': sha(__file__),
        'interpretation': 'A budget flag is an observation, not proof that a larger budget would fix the failure. No generation budget was changed.'}
    write_json(run / 'length_diagnostics.json', out)
    lines = ['# 生成长度与重复运行诊断', '',
        '本轮只扩大题量和采样次数，每题生成预算仍为 2048 token，每次调用 1024 token，上下文 4096 token。以下是描述性检查，不用于选择模型或修改本轮方案。', '',
        '| 模型 | 平均输出 token | 触及总输出上限 | 上下文超限 | 失败中出现任一预算标记 |',
        '|---|---:|---:|---:|---:|']
    for name, s in result.items():
        lines.append(f"| {name} | {s['mean_assistant_tokens']:.1f} | {s['total_output_cap_hits']}/1384 | {s['context_rejections']}/1384 | {s['failures_with_any_budget_flag']}/{s['failures']} |")
    lines += ['', '“任一预算标记”包括总输出达到上限、最后一次调用达到其请求上限或上下文超限，三者取并集。触及上限不等于失败由上限导致，也不保证增加长度能解决。', '',
        '| 原 32 题前 2 次采样的重复运行 | 上次正确数 | 本次正确数 | 逐题评分一致 | 完整生成 token 一致 |',
        '|---|---:|---:|---:|---:|']
    for name, s in repeat.items():
        lines.append(f"| {name} | {int(s['previous_successes'])}/64 | {int(s['current_successes'])}/64 | {s['same_reward']}/64 | {s['same_response_token_sequences']}/64 |")
    lines += ['', '固定 checkpoint、harness、输入任务、随机种子和预算不保证跨进程/设备的完整轨迹逐 token 完全一致；主比较使用本轮重新评测的全部 8 次采样。', '']
    (run / 'LENGTH_DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(json.dumps(out, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args().run.resolve())
