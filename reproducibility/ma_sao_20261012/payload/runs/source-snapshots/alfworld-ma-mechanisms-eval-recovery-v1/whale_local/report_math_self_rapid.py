"""Lightweight checks and a first-result plot; no additional model evaluation."""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]


def main(run):
    cfg = read(run / 'config.json'); frozen = read(run / 'FROZEN.json')
    state = read(run / 'TEST_STATES.json')['W1_H_internal']
    rr = rows(Path(state['output']) / 'trajectories.jsonl')
    assert len(rr) == 128 and all(r['draw'] == 0 for r in rr)
    final = read(Path(state['output']) / 'val.json')
    assert sum(r['reward'] for r in rr) == final['solved_examples']
    assert final['metadata']['dataset_sha256'] == cfg['data']['splits']['test']['sha256']
    assert all(r['sampling_identity']['weight_digest'] == cfg['models']['W1']['weight_digest'] for r in rr)
    assert all(r['harness_sha256'] == frozen['internal_harness_sha256'] for r in rr)
    baseline = read(run / 'BASELINE_QUALIFICATION.json')['first_draw']
    original = rows(baseline['W1_Hf']['source'])
    original = [r for r in original if r['draw'] == 0]
    assert [r['task_id'] for r in rr] == [r['task_id'] for r in original]
    before = rows(run / 'pre_proposals/scored.jsonl'); after = rows(run / 'post_proposals/scored.jsonl')
    training = read(run / 'editor_grpo/complete.json')
    updates = 0
    if training['updated']:
        t = read(Path(training['identity']['weights']) / 'training_state.json')
        assert training['reload_verified'] and t['weight_digest'] != cfg['editor_initial_digest']
        assert t['changed_parameter_tensors']; updates = t['epochs']
    actual_dev = 48
    for p in (run / 'outcome_cache').glob('*/val.json'): actual_dev += read(p)['num_examples']
    for p in (run / 'selection').glob('*/remaining/val.json'): actual_dev += read(p)['num_examples']
    assert actual_dev <= 240
    solved = {'initial': baseline['W0_H0']['solved'], 'weight_only': baseline['W1_H0']['solved'],
        'external_whale': baseline['W1_Hf']['solved'], 'internal_whale': final['solved_examples']}
    steps = [0, 1, 2]; fig, ax = plt.subplots(figsize=(8.3, 4.5))
    for name, color, label in [('external_whale', '#3b68b4', 'External proposer (historical draw 0)'),
                                ('internal_whale', '#d77928', 'Internal 2B proposer (new single draw)')]:
        y = [solved['initial'], solved['weight_only'], solved[name]]
        ax.plot(steps, y, 'o-', linewidth=2, color=color, label=label)
        ax.annotate(f'{y[-1]}/128', (2, y[-1]), xytext=(10, 7 if name.startswith('external') else -17), textcoords='offset points', color=color)
    for i, key in enumerate(['initial', 'weight_only']): ax.annotate(f"{solved[key]}/128", (i, solved[key]), xytext=(0, 10), textcoords='offset points', ha='center')
    ax.set_xticks(steps, ['W0 + H0', 'W1 + H0\nRSFT', 'W1 + selected H\nHarness search'])
    ax.set_ylabel('Solved questions / 128'); ax.set_xlim(-.15, 2.4)
    ax.set_ylim(max(0, min(solved.values()) - 10), min(128, max(solved.values()) + 13))
    ax.set_title('Rapid WHALE proposer replacement: MATH number theory')
    ax.grid(alpha=.2); ax.legend(loc='lower right', fontsize=8)
    fig.text(.5, .01, 'Single draw; historical baseline reused. Screening result, not an independent confirmation.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .05, 1, 1])
    for ext in ['png', 'svg', 'pdf']: fig.savefig(run / ('rapid_curve.' + ext), dpi=160)
    ended = datetime.datetime.fromisoformat(read(run / 'COMPLETE.json')['utc'])
    started = datetime.datetime.fromisoformat(read(run / 'START.json')['utc'])
    elapsed = (ended - started).total_seconds()
    changed = sum(a.get('source') != b.get('source') for a, b in zip(before[:4], after))
    selected_source = Path(frozen['internal_harness']).read_text()
    existed_before = any(r.get('source') == selected_source and r['outcome_kind'] == 'valid' for r in before)
    unused_file = run / 'unused_generation_mode/generations.jsonl'
    unused = rows(unused_file) if unused_file.exists() else []
    result = {'basic_checks_passed': True, 'exhaustive_audit': False, 'scores': solved, 'questions': 128,
        'internal_harness_gain_pp': 100 * (solved['internal_whale'] - solved['weight_only']) / 128,
        'internal_minus_external_pp': 100 * (solved['internal_whale'] - solved['external_whale']) / 128,
        'internal_dev_trajectories': actual_dev, 'external_dev_trajectories': 240,
        'shared_weight_training_trajectories_per_arm': 512, 'shared_solver_updates': 120,
        'new_editor_updates': updates, 'editor_updated': training['updated'],
        'unused_editor_generations_due_to_entrypoint_error': len(unused),
        'unused_editor_tokens': sum(len(r['response_ids']) for r in unused),
        'pre_valid': sum(r['outcome_kind'] != 'invalid' for r in before),
        'pre_unique_sources': len({r.get('source') for r in before if r['valid_source']}),
        'post_valid': sum(r['outcome_kind'] != 'invalid' for r in after),
        'post_unique_sources': len({r.get('source') for r in after if r['valid_source']}),
        'pre_rewards': [r['reward_delta'] for r in before], 'post_rewards': [r['reward_delta'] for r in after],
        'matched_sources_changed': changed, 'elapsed_seconds_including_restart': elapsed,
        'selected_harness_already_generated_before_grpo': existed_before,
        'final_harness': frozen['internal_harness'], 'editor_checkpoint': frozen['editor_checkpoint']}
    (run / 'RAPID_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    text = f'''# 内部 2B 替换外部 WHALE 改写器：第一版快速结果

| 阶段 | 128 题解题数 |
|---|---:|
| 初始 W0/H0 | {solved['initial']} |
| 更新权重 W1/H0 | {solved['weight_only']} |
| 外部模型改写 harness | {solved['external_whale']} |
| 内部 2B 改写 harness | {solved['internal_whale']} |

![阶段曲线](rapid_curve.png)

内部 harness 相比仅更新权重为 **{result['internal_harness_gain_pp']:+.2f} 个百分点**，相比外部版本为 **{result['internal_minus_external_pp']:+.2f} 个百分点**。

这是快速筛选：前三项复用已审计实验的第一条采样；内部结果在同样128题、相同 seed42、每题一次采样上新测。没有新增多 seed、重复采样、交叉消融或全面审计。基本检查确认数据、权重和 harness 身份一致，分数由已有 verifier 计算。历史基线与新实验运行时间和硬件分配不同，细小差距暂不作优势结论。

权重训练复用原 WHALE 的512条轨迹、120次真实 RSFT 更新，两臂同样计入；本轮没有重复训练求解器。外部 harness 原搜索花费240条任务轨迹，内部实际花费 **{actual_dev}** 条，上限同为240；重复候选复用评分。内部另外生成12次代码提案，已有 SFT 初始化的成本不包含在任务轨迹中。

改写器：训练前8次提案中{result['pre_valid']}次代码合法、{result['pre_unique_sources']}种代码；训练后4次提案中{result['post_valid']}次合法、{result['post_unique_sources']}种代码。新增 outcome-GRPO 更新 **{updates}** 次，同提示/采样配置下4个提案中 **{changed}** 个代码发生变化。训练前奖励差值为 `{result['pre_rewards']}`，训练后为 `{result['post_rewards']}`。若更新数为0，是候选奖励无差异，不能声称改写器学到了东西。

最终选中代码在 GRPO 前是否已经生成：**{existed_before}**。因此应区分“内部2B可以生成有收益的harness”和“GRPO进一步提高了改写能力”。本轮主要观察到工具调用判断错误被修正，没有看到新改写机制。当前提示明确聚焦答案补全，下一轮先放开机制选择，再考虑增加训练量。

本轮总耗时约 **{elapsed / 60:.1f} 分钟**，含按用户要求缩减评测并续跑的时间。使用本机及内网空闲4090，最多13张并行执行求解任务，另一张运行改写器。

工程额外开销：训练入口漏传模式产生了{len(unused)}个额外提案，已保留，但未评测或用于训练/选择；修复后才执行上述真实更新。它们计入时间开销，不计作算法收益。

最终代码：[harness]({frozen['internal_harness']})；改写器 checkpoint：`{frozen['editor_checkpoint']}`。

这是 MATH number_theory Level2–4 的小规模算法实验，尚非论文完整 DAPO/AIME 复现，也尚未证明多轮持续自提高。当前测试题有历史使用记录。内部支持提示词和恢复钩子改写，但代码语法比外部原生全代码搜索更受限。
'''
    (run / 'REPORT.md').write_text(text)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); args = p.parse_args(); main(args.run.resolve())
