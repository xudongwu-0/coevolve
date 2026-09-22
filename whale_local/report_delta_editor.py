"""Small result report: actual editor learning, alternating stages, no new evals."""
import argparse
import datetime
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')


def main(run):
    cfg = read(run / 'config.json'); done = read(run / 'COMPLETE.json'); curve = read(run / 'CURVE.json')
    assert len(curve) == 4
    keys = None
    for point in curve:
        rr = rows(Path(point['output']) / 'trajectories.jsonl')
        assert len(rr) == 64 and sum(r['reward'] for r in rr) == point['solved']
        current_keys = [(r['task_id'], r['draw']) for r in rr]
        if keys is None: keys = current_keys
        else: assert current_keys == keys
        digest = read(Path(point['model']) / 'training_state.json')['weight_digest']
        assert all(r['sampling_identity']['weight_digest'] == digest for r in rr)
    final = rows(run / 'final128/trajectories.jsonl'); old = rows(Path(cfg['reference']) / 'final128/trajectories.jsonl')
    assert len(final) == len(old) == 128
    assert [(r['task_id'], r['draw']) for r in final] == [(r['task_id'], r['draw']) for r in old]
    initial_score = int(sum(r['reward'] for r in old)); final_score = int(sum(r['reward'] for r in final))
    phases = []; total_updates = 0
    for n in [1, 2]:
        d = run / f'cycle_{n:02d}'; outcome = read(d / 'EDITOR_RESULT.json'); trained = read(d / 'editor_update/complete.json')
        pre = rows(d / 'pre/scored.jsonl'); post = rows(d / 'post/scored.jsonl')
        assert [g['seed'] for g in pre[:4]] == [g['seed'] for g in post]
        for a, b in zip(pre, post): assert a['messages'] == b['messages']
        updates = 0
        if trained['updated']:
            state = read(Path(trained['identity']['weights']) / 'training_state.json')
            assert trained['reload_verified'] and state['changed_parameter_tensors']
            assert state['sampling_identity']['weight_digest'] == outcome['before']['weight_digest']
            assert state['weight_digest'] != state['sampling_identity']['weight_digest']
            updates = state['new_updates']; total_updates += updates
        target = read(d / 'START.json')['solver']['weight_digest']
        assert all(g['reward_provenance']['solver_weight_digest'] == target for g in pre + post)
        for g in pre + post:
            r = g['reward_provenance']
            assert abs(g['reward_delta'] - (r['rescued'] - r['regressed']) / r['task_count']) < 1e-9
        decision = read(d / 'selection/decision.json')
        phases.append({'phase': n, 'editor_updates': updates, 'pre_matched4': outcome['pre_matched4'], 'post': outcome['post'],
            'pre_all8': outcome['pre'], 'harness_changed': decision['changed'], 'chosen': decision['accepted'],
            'changed_matched_outputs': sum(a['content'] != b['content'] for a, b in zip(pre, post)),
            'selection_scores': {k: v['solved_examples'] for k, v in decision['results'].items()}})
    solver = read(run / 'cycle_01/solver_update/complete.json')
    state = read(Path(solver['identity']['weights']) / 'training_state.json')
    assert solver['reload_verified'] and state['changed_parameter_tensors']
    assert state['rollouts'] == 32 and state['native_math_reward_rechecked_every_trajectory'] and state['exact_server_tokens_verified']
    solver_updates = state['optimizer_steps_after'][0] - max(state['optimizer_steps_before'])
    assert read(run / 'cycle_02/START.json')['solver']['weight_digest'] == state['weight_digest']
    warm = run / 'format_warmup/complete.json'
    warm_updates = read(warm)['new_updates'] if warm.exists() else 0
    probe = rows(run / 'interface_probe/generations.jsonl')
    start = datetime.datetime.fromisoformat(read(run / 'START.json')['utc']); end = datetime.datetime.fromisoformat(done['utc'])
    result = {'basic_checks_passed': True, 'full_audit': False, 'initial128': initial_score, 'final128': final_score,
        'gain_questions': final_score - initial_score, 'gain_pp': 100 * (final_score - initial_score) / 128,
        'curve': [{k: p[k] for k in ['label', 'solved', 'total']} for p in curve], 'editor_phases': phases,
        'new_editor_rl_updates': total_updates, 'new_editor_format_sft_updates': warm_updates,
        'new_solver_updates': solver_updates, 'new_solver_rollouts': 32, 'accepted_solver_rollouts': state['accepted_trajectories'],
        'interface_probe': {'slots': len(probe), 'valid_changed': sum(g['valid_source'] and g['edit_changed'] for g in probe),
            'unique_changed': len({g['source'] for g in probe if g['valid_source'] and g['edit_changed']})},
        'elapsed_minutes': (end - start).total_seconds() / 60, 'checkpoints': read(run / 'CURRENT_MODELS.json')}
    save(run / 'RAPID_RESULT.json', result)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1.65, 1]})
    yy = [p['solved'] for p in curve]
    for i in range(3): axes[0].plot([i, i + 1], yy[i:i + 2], 'o-', color='#356cb5' if i == 1 else '#cf7d21', lw=2.5)
    for i, y in enumerate(yy): axes[0].annotate(f'{y}/64', (i, y), xytext=(0, 10), textcoords='offset points', ha='center')
    tick_labels = ['Start', 'Editor 1\n+ harness', 'Solver\nweights', 'Editor 2\n+ harness']
    for phase in phases:
        if not phase['harness_changed']:
            tick_labels[1 if phase['phase'] == 1 else 3] = f'Editor {phase["phase"]}\nHarness unchanged'
    axes[0].set_xticks(range(4), tick_labels)
    axes[0].set_ylim(min(yy) - 5, max(yy) + 6); axes[0].set_ylabel('Solved / 64'); axes[0].grid(alpha=.2)
    axes[0].set_title('Raw alternating stage scores')
    axes[1].bar(['Start', 'Final'], [initial_score, final_score], color=['#9ba7b4', '#356cb5'], width=.55)
    for i, y in enumerate([initial_score, final_score]): axes[1].text(i, y + 2, f'{y}/128', ha='center')
    axes[1].set_ylim(0, 128); axes[1].set_title('Full 128 questions'); axes[1].set_ylabel('Solved / 128')
    fig.suptitle('Current-solver gains train a compact harness editor')
    fig.text(.5, .01, 'One draw per question. Historical initial evaluation reused. No test scores used for selection.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, .04, 1, 1])
    for ext in ['png', 'pdf', 'svg']: fig.savefig(run / ('editor_curve.' + ext), dpi=160)
    labels = ['起点', '第一次改写器训练 + harness选择', '小量求解器训练', '刷新失败后第二次改写器训练 + harness选择']
    table = '\n'.join(f'| {label} | {point["solved"]}/64 |' for label, point in zip(labels, curve))
    learning = '\n'.join(f'| {r["phase"]} | {r["editor_updates"]} | {r["pre_matched4"]["positive_gain"]}/4 → {r["post"]["positive_gain"]}/4 | {100*r["pre_matched4"]["mean_reward"]:+.2f} → {100*r["post"]["mean_reward"]:+.2f} | {r["pre_matched4"]["unique_changed"]}/4 → {r["post"]["unique_changed"]}/4 |' for r in phases)
    (run / 'REPORT.md').write_text(f'''# 改写器权重更新方法：短修改与当前求解器增益

完整128题：**{initial_score}→{final_score}，{final_score-initial_score:+d}题（{result['gain_pp']:+.2f}个百分点）**。实验运行约{result['elapsed_minutes']:.1f}分钟。

| 阶段 | 固定64题 |
|---|---:|
{table}

![阶段曲线与完整评测](editor_curve.png)

| 改写器阶段 | 实际RL更新 | 相同4个提案seed：正收益候选数 | 平均候选解题增益，百分点 | 不同有效修改数 |
|---|---:|---:|---:|---:|
{learning}

上表只在每个阶段内比较相同输入和提案seed。两个阶段对应不同求解器/当前harness，奖励任务也重新选择，不能把两行直接当作同一固定测验上的学习曲线。开发12题的候选收益与最终评测分开记录，尚非新的独立测试。

改写器新增RL优化{total_updates}步、接口SFT优化{warm_updates}步；求解器采样32条新轨迹，{state['accepted_trajectories']}条成功轨迹用于{solver_updates}次RSFT更新。模型变化和重载、奖励正负、前后输入/提案seed一致、目标求解器身份刷新均通过基本检查。未增加多seed或全面审计。

本版把完整模块生成改成短JSON编辑，当前可改三处提示内容，保留已有可执行钩子。用当前求解器相对当前harness的实际解题增益训练编辑tokens，无效候选不再通过人为失败惩罚制造组内差异。动作表达、奖励及失败输入同时变化，不能据此拆分各改动的独立贡献。方法与预算见 [METHOD.md](METHOD.md)。

完整机器可读结果见 [RAPID_RESULT.json](RAPID_RESULT.json)，最终两份模型见 [CURRENT_MODELS.json](CURRENT_MODELS.json)。
''')
    save(run / 'STATUS.json', {'stage': 'complete', 'basic_checks_passed': True, 'report': str(run / 'REPORT.md')})
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('run', type=Path); main(p.parse_args().run.resolve())
