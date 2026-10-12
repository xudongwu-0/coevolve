"""Basic run checks and a raw stage curve, without new model evaluations."""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def save(p, v): p.write_text(json.dumps(v, indent=2, ensure_ascii=False) + '\n')


def main(run):
    cfg = read(run / 'config.json'); curve = read(run / 'CURVE.json')
    assert len(curve) == 5
    ids = None
    for point in curve:
        rr = rows(Path(point['output']) / 'trajectories.jsonl')
        assert len(rr) == 64 and sum(r['reward'] for r in rr) == point['solved']
        assert all(r['draw'] == 0 for r in rr)
        keys = [r['task_id'] for r in rr]
        if ids is None: ids = keys
        else: assert ids == keys
        digest = read(Path(point['model']) / 'training_state.json')['weight_digest']
        assert all(r['sampling_identity']['weight_digest'] == digest for r in rr)
    final = read(run / 'final128/val.json'); rr = rows(run / 'final128/trajectories.jsonl')
    assert len(rr) == 128 and sum(r['reward'] for r in rr) == final['solved_examples']
    old_rows = rows(Path(cfg['initial_monitor']) / 'trajectories.jsonl')
    assert [r['task_id'] for r in rr] == [r['task_id'] for r in old_rows]
    old_score = int(sum(r['reward'] for r in old_rows)); rounds = []
    for n in [1, 2]:
        cycle = run / f'cycle_{n:02d}'
        e = read(cycle / 'editor_grpo/complete.json')
        editor_result = read(cycle / 'EDITOR_RESULT.json')
        editor_updates = 0
        if e['updated']:
            state = read(Path(e['identity']['weights']) / 'training_state.json')
            assert e['reload_verified'] and state['changed_parameter_tensors']
            assert state['weight_digest'] != state['sampling_identity']['weight_digest']
            editor_updates = state['optimizer_steps_after'][0] - (max(state['optimizer_steps_before']) if state['optimizer_steps_before'] else 0)
        solver = read(cycle / 'solver_update/complete.json')
        ws = read(Path(solver['identity']['weights']) / 'training_state.json')
        assert solver['reload_verified'] and ws['changed_parameter_tensors']
        assert ws['weight_digest'] != ws['sampling_identity']['weight_digest']
        assert ws['native_math_reward_rechecked_every_trajectory'] and ws['exact_server_tokens_verified']
        assert ws['rollouts'] == 128 and ws['harness_sha256'] == read(cycle / 'selection/decision.json')['harness_sha256']
        h = read(cycle / 'selection/decision.json')
        start = read(cycle / 'START.json')
        assert ws['sampling_identity']['weight_digest'] == start['solver']['weight_digest']
        assert editor_result['before']['weight_digest'] == start['editor']['weight_digest']
        if n == 2:
            prev = read(run / 'cycle_01/complete.json')
            assert start['solver'] == prev['solver'] and start['editor'] == prev['editor']
        rounds.append({'cycle': n, 'editor_updates': editor_updates,
            'solver_updates': ws['optimizer_steps_after'][0] - max(ws['optimizer_steps_before']),
            'accepted_train_trajectories': ws['accepted_trajectories'], 'solver_training_seconds': ws['elapsed_seconds'],
            'harness_changed': h['changed'], 'selected_harness': h['accepted'],
            'development_scores': {k: v['solved_examples'] for k, v in h['results'].items()},
            'pre_unique_sources': editor_result['pre_unique_sources'], 'post_unique_sources': editor_result['post_unique_sources'],
            'pre_valid': editor_result['pre_valid'], 'post_valid': editor_result['post_valid'],
            'extra_exploration': editor_result['extra_exploration']})
    changes = [{'stage': curve[i]['label'], 'change_questions': curve[i]['solved'] - curve[i - 1]['solved'],
        'change_pp': 100 * (curve[i]['solved'] - curve[i - 1]['solved']) / 64} for i in range(1, 5)]
    start = datetime.datetime.fromisoformat(read(run / 'START.json')['utc'])
    end = datetime.datetime.fromisoformat(read(run / 'COMPLETE.json')['utc'])
    y = [p['solved'] for p in curve]
    fig, ax = plt.subplots(figsize=(9, 4.7))
    for i in range(4):
        ax.plot([i, i + 1], y[i:i + 2], 'o-', color='#ce7920' if i % 2 == 0 else '#3266b5', linewidth=2.5)
    for i, value in enumerate(y):
        ax.annotate(f'{value}/64', (i, value), xytext=(0, 10), textcoords='offset points', ha='center')
    stage_labels = ['Start', 'Round 1\nEditor + harness', 'Round 1\nSolver weights', 'Round 2\nEditor + harness', 'Round 2\nSolver weights']
    for i, r in enumerate(rounds):
        if not r['harness_changed']:
            stage_labels[1 + i * 2] = f'Round {i + 1}\nHarness unchanged'
    ax.set_xticks(range(5), stage_labels)
    ax.set_ylabel('Solved questions / 64'); ax.set_title('Two internal 2B checkpoints: alternating updates')
    ax.set_ylim(max(0, min(y) - 8), min(64, max(y) + 8)); ax.grid(alpha=.2)
    ax.legend(handles=[Line2D([0], [0], color='#ce7920', lw=2, label='Editor GRPO + harness selection'),
        Line2D([0], [0], color='#3266b5', lw=2, label='Solver RSFT')], fontsize=8)
    fig.text(.5, .01, 'Same 64 questions, one draw each. Raw stage scores; unchanged states reuse evaluation.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    for ext in ['png', 'pdf', 'svg']: fig.savefig(run / ('alternating_curve.' + ext), dpi=160)
    result = {'basic_checks_passed': True, 'full_audit': False, 'curve': [{k: p[k] for k in ['label', 'solved', 'total']} for p in curve],
        'stage_changes': changes, 'initial128': old_score, 'final128': final['solved_examples'],
        'final_gain_questions': final['solved_examples'] - old_score,
        'final_gain_pp': 100 * (final['solved_examples'] - old_score) / 128,
        'rounds': rounds, 'elapsed_minutes': (end - start).total_seconds() / 60,
        'checkpoints': read(run / 'CURRENT_MODELS.json')}
    result['evaluation_halves'] = {
        'first64': {'initial': curve[0]['solved'], 'final': curve[-1]['solved']},
        'remaining64': {'initial': old_score - curve[0]['solved'], 'final': final['solved_examples'] - curve[-1]['solved']}}
    diagnostic = run / 'diagnostic_strategy_request/RESULT.json'
    if diagnostic.exists():
        result['strategy_diagnostic'] = read(diagnostic)
    save(run / 'RAPID_RESULT.json', result)
    labels = ['起点', '第1轮：改写器训练 + harness选择', '第1轮：求解器训练', '第2轮：改写器训练 + harness选择', '第2轮：求解器训练']
    for i, r in enumerate(rounds):
        if not r['editor_updates'] and not r['harness_changed']:
            labels[1 + i * 2] = f'第{i + 1}轮：harness不变，改写器训练跳过'
    table = '\n'.join(f'| {label} | {point["solved"]}/64 |' for label, point in zip(labels, curve))
    updates = '\n'.join(f'| {r["cycle"]} | {r["editor_updates"]} | {r["solver_updates"]} | {r["accepted_train_trajectories"]}/128 | {r["harness_changed"]} |' for r in rounds)
    text = f'''# 两个内部2B checkpoint：两轮快速交替

同128题最终结果：**{old_score}/128 → {final['solved_examples']}/128**，变化 **{result['final_gain_questions']:+d}题（{result['final_gain_pp']:+.2f}个百分点）**。总耗时约 **{result['elapsed_minutes']:.1f}分钟**。

任务沿用 MATH number_theory、Level 2–4 的小规模整数答案场景。起点继承上一轮已训练的求解器与内部改写器，并非从 base model 重新开始。

每轮顺序：内部改写器生成8个候选 → 真实执行反馈做4次GRPO（无奖励差异时提高温度再探索8个，仍无差异则跳过训练）→ 改写器再生成4个候选 → 选择harness → 在所选harness下采样128条轨迹 → 成功轨迹训练求解器。两个角色的checkpoint和优化器分别连续继承。

| 阶段 | 固定64题单次解题数 |
|---|---:|
{table}

![交替曲线](alternating_curve.png)

| 轮次 | 改写器新增更新 | 求解器新增更新 | 用于RSFT的成功轨迹 | harness是否变化 |
|---|---:|---:|---:|---|
{updates}

改写提示已允许模型自己选择推理指导、精确计算、Python校验、工具错误反馈、答案恢复等机制。候选由内部模型生成，没有外部模型代写或修正候选代码。

使用本机和内网第二台机器共10张当时空闲的4090：9张分担采样/评测，1张依次训练两个角色；这次没有做跨机梯度联合训练。求解器两轮共新增{sum(r['solver_updates'] for r in rounds)}次优化器更新，改写器共新增{sum(r['editor_updates'] for r in rounds)}次。

曲线使用相同64题的每题一次采样，最终再补齐余下64题；未增加多seed或交叉消融。同一W/H状态复用评测，起点复用上一轮结果，下降也原样保留。题目有历史使用记录，这是一版迭代结果。64题监测分数没有反馈给本轮训练或候选选择。

基本检查确认了两个角色权重实际变化、checkpoint重载、连续继承、求解器在选中harness下自采样训练，以及各阶段评测题目相同。未进行全面审计。

最终两份checkpoint见 [CURRENT_MODELS.json](CURRENT_MODELS.json)，逐轮候选数与开发成绩见 [RAPID_RESULT.json](RAPID_RESULT.json)。

本轮现象、复制诊断与下一步建议见 [INTERPRETATION.md](INTERPRETATION.md)。
'''
    (run / 'REPORT.md').write_text(text)
    save(run / 'STATUS.json', {'stage': 'complete', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'basic_checks_passed': True, 'report': str(run / 'REPORT.md')})
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); args = p.parse_args(); main(args.run.resolve())
