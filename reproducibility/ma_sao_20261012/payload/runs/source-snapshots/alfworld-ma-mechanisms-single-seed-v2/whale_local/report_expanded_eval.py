"""Question-paired mean-at-eight comparisons, including prior exposure labels."""
import argparse
import json
from pathlib import Path

import numpy as np

from alternating_pool import read
from assess_viability import paired_change
from pipeline import sha, write_json


def report(run):
    config, audit = read(run / 'config.json'), read(run / 'verification.json')
    assert audit['passed_audit']
    outcomes, question_ids = {}, {}
    for name in config['model_order']:
        outcomes[name], question_ids[name] = {}, {}
        for cohort in config['cohort_order']:
            count = config['data']['splits'][cohort]['count']
            arr = np.full((count, config['draws']), np.nan)
            ids = [None] * count
            with (run / 'eval' / name / cohort / 'trajectories.jsonl').open() as stream:
                for line in stream:
                    r = json.loads(line)
                    arr[r['example_index'], r['draw']] = r['reward']
                    ids[r['example_index']] = r['task_id']
            assert np.isfinite(arr).all()
            outcomes[name][cohort] = arr.mean(axis=1)
            question_ids[name][cohort] = ids
        outcomes[name]['all173'] = np.concatenate([outcomes[name][c] for c in config['cohort_order']])
    assert all(question_ids[n] == question_ids['W2_initial'] for n in config['model_order'])
    comparisons = {}
    for cohort in ['all173', *config['cohort_order']]:
        comparisons[cohort] = {f'{after}_minus_{before}': paired_change(outcomes[before][cohort], outcomes[after][cohort])
            for before, after in [('W2_initial', 'W3_cycle1'), ('W3_cycle1', 'W4_cycle2'), ('W2_initial', 'W4_cycle2')]}
    result = {'metric': config['metric'], 'primary': comparisons['all173']['W4_cycle2_minus_W2_initial'],
        'comparisons': comparisons, 'scores': audit['summaries'],
        'elapsed_seconds': read(run / 'COMPLETE.json')['elapsed_seconds'],
        'verification_sha256': sha(run / 'verification.json'), 'limitations': config['limitations']}
    write_json(run / 'statistics.json', result)
    labels = {'W2_initial': '起点 W2', 'W3_cycle1': '第一轮后 W3', 'W4_cycle2': '第二轮后 W4'}
    lines = ['# 扩大固定 checkpoint 评测', '',
        '同一 MATH 数论场景，173 题 × 每题 8 次采样 × 3 个 checkpoint，共 4,152 条真实轨迹。',
        '所有组合固定 H3、temperature=1.0、配对种子、总生成上限 2048 token、单次调用上限 1024 token。没有新训练或候选选择。', '',
        '| 模型阶段 | 正确次数 / 总采样 | mean@8 |', '|---|---:|---:|']
    for name in config['model_order']:
        solved = sum(v['solved'] for v in audit['summaries'][name].values())
        lines.append(f'| {labels[name]} | {solved}/1384 | {solved / 1384 * 100:.2f}% |')
    lines += ['', 'mean@8 是每题 8 次生成的平均正确率，不是“8 次中至少答对一次”。', '',
        '| 比较 | 净变化（百分点） | 按题配对 bootstrap 95% 区间 | 提升 / 下降题数 |', '|---|---:|---:|---:|']
    for key, v in comparisons['all173'].items():
        low, high = v['paired_bootstrap_95_percent_pp']
        lines.append(f"| {key} | {v['difference_pp']:+.2f} | [{low:+.2f}, {high:+.2f}] | {v['new_successes']} / {v['regressions']} |")
    lines += ['', '| 数据来源 | 题量 | W2 | W3 | W4 | W4−W2（百分点，95% 区间） |', '|---|---:|---:|---:|---:|---|']
    cohort_labels = {'historical_test128': '历史测试集（曾评测）', 'reused_monitor32': '原监控集（反复使用）', 'unopened13': '此前登记 split 未使用题'}
    for cohort in config['cohort_order']:
        scores = [audit['summaries'][n][cohort] for n in config['model_order']]
        v = comparisons[cohort]['W4_cycle2_minus_W2_initial']; lo, hi = v['paired_bootstrap_95_percent_pp']
        lines.append(f"| {cohort_labels[cohort]} | {scores[0]['questions']} | " +
            ' | '.join(f"{s['solved']}/{s['attempts']} ({s['mean_at_8_percent']:.2f}%)" for s in scores) +
            f" | {v['difference_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ['',
        '主比较预先固定为全 173 题上的 W4−W2；分组和相邻轮次比较用于描述，不作多重检验后的独立确认。',
        '置信区间按题重采样（20,000 次），保留同一题的 8 次采样在同一簇内。区间不包含重新训练不同随机种子的方差。',
        '173 题中 160 题已有评测使用历史；13 道未使用题另列。扩大评测提高测量精度，但不会把历史使用的数据变成独立测试集。',
        '三个阶段使用完全相同的 H3，因此这里只能衡量固定 harness 下的求解器权重变化。没有证明 harness 学习持续上升、gate 优于 WHALE 或样本效率优势。', '',
        f"原始轨迹、原始模型响应、checkpoint 身份、评分、数据隔离和服务退出均通过审计；核验 {audit['verified_unique_calls']} 次模型调用。",
        f"完整评测耗时 {result['elapsed_seconds'] / 60:.1f} 分钟，使用 {len(config['pool_gpus'])} 张空闲 4090。",
        f"审计 SHA256：`{result['verification_sha256']}`。", '']
    (run / 'REPORT.md').write_text('\n'.join(lines))
    write_json(run / 'STATUS.json', {'stage': 'evaluation_complete_audited', 'verification_sha256': sha(run / 'verification.json')})
    print(json.dumps({'primary': result['primary'], 'report': str(run / 'REPORT.md')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    report(p.parse_args().run.resolve())
