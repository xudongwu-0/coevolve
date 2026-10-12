"""Plot existing audited half-step outcomes; performs no sampling or training."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from pipeline import ROOT, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    training = ROOT / 'runs/whale_alternating_ready_20260916'
    evaluation = ROOT / 'runs/whale_expanded_eval_20260916'
    out = evaluation / 'attribution'
    out.mkdir(exist_ok=True)
    training_audit = read(training / 'verification.json')
    audit = read(evaluation / 'verification.json')
    assert training_audit['passed_audit'] and audit['passed_audit']
    cfg = read(evaluation / 'config.json')
    curve = read(training / 'monitor_curve.json')
    assert len(curve) == 5
    names = {v['weight_digest']: k for k, v in cfg['models'].items()}
    labels = ['起点\nW2 + H3', '第1轮\nharness选择后', '第1轮\nweight更新后',
              '第2轮\nharness选择后', '第2轮\nweight更新后']
    stages = []
    for i, point in enumerate(curve):
        assert point['harness_sha256'] == cfg['harness_sha256']
        name = names[point['weight_digest']]
        scores = audit['summaries'][name]
        solved = sum(v['solved'] for v in scores.values())
        count = sum(v['attempts'] for v in scores.values())
        assert count == 1384
        same = i in (1, 3)
        if same:
            assert point['weight_digest'] == curve[i-1]['weight_digest']
            assert point['harness_sha256'] == curve[i-1]['harness_sha256']
        stages.append({'stage': point['label'], 'model': name, 'weight_digest': point['weight_digest'],
            'harness_sha256': point['harness_sha256'], 'correct_samples': solved, 'total_samples': count,
            'accuracy_percent': solved / count * 100,
            'phase': 'initial' if i == 0 else 'harness' if same else 'weight',
            'same_execution_state_score_reused': same})
    editor = []
    for name in ['E1c', 'E2', 'E3']:
        path = training / 'editor_probes' / name / 'score_summary.json'
        value = read(path)
        editor.append({'checkpoint': name, 'proposals': value['proposals'],
            'mean_reward_delta_pp': value['mean_reward_delta'] * 100,
            'source': str(path), 'source_sha256': sha(path)})
    data = {'execution_curve': stages, 'fixed_solver_editor_probe': editor,
        'training_verification_sha256': sha(training / 'verification.json'),
        'evaluation_verification_sha256': sha(evaluation / 'verification.json'),
        'source_monitor_curve_sha256': sha(training / 'monitor_curve.json'),
        'plot_script_sha256': sha(__file__), 'matplotlib_version': matplotlib.__version__,
        'new_model_calls': 0, 'new_training_updates': 0,
        'notes': ['Uses the already completed 173-question mean-at-eight evaluation.',
                  'Harness selection accepted no new code; H stages reuse the identical W/H execution state score.',
                  'The editor probe is a separate fixed-development-packet measurement, not test accuracy.',
                  '160 evaluation questions had prior exposure; 13 were new in that completed evaluation.']}
    write_json(out / 'curve_data.json', data)
    with (out / 'execution_curve.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stages[0]))
        writer.writeheader(); writer.writerows(stages)
    font_path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    assert font_path.exists()
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=str(font_path)).get_name(),
        'axes.unicode_minus': False, 'font.size': 11, 'axes.spines.top': False,
        'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, (ax, ed) = plt.subplots(1, 2, figsize=(13.5, 6.3), gridspec_kw={'width_ratios': [1.85, 1]})
    fig.subplots_adjust(left=.07, right=.975, bottom=.29, top=.76, wspace=.30)
    fig.suptitle('两轮交替更新：任务成绩与改写器能力', y=.965, fontsize=19, fontweight='bold')
    blue, orange, teal = '#2166ac', '#d9822b', '#258778'
    values = [p['accuracy_percent'] for p in stages]
    for i in range(1, 5):
        is_h = stages[i]['phase'] == 'harness'
        ax.plot([i-1, i], values[i-1:i+1], color=orange if is_h else blue,
                linestyle='--' if is_h else '-', linewidth=2.6)
        delta = values[i] - values[i-1]
        ax.text(i-.5, (values[i]+values[i-1])/2 + (-1.3 if is_h else .7),
                f"Δ{'H' if is_h else 'W'} = {delta:+.2f} pp", ha='center',
                color=orange if is_h else blue, fontsize=11)
    for i, value in enumerate(values):
        is_h = stages[i]['phase'] == 'harness'
        ax.scatter(i, value, s=75, facecolors='white' if is_h else blue,
                   edgecolors=orange if is_h else blue, linewidth=2, zorder=4)
        ax.text(i, value+.45, f'{value:.2f}%', ha='center', fontsize=12)
    ax.set(title='执行系统：先 H 阶段，再 W 阶段', ylabel='任务平均正确率（%）',
           ylim=(65, 80), xlim=(-.35, 4.35), xticks=range(5), xticklabels=labels,
           yticks=[65, 70, 75, 80])
    ax.grid(axis='y', alpha=.20)
    ed.plot(range(3), [v['mean_reward_delta_pp'] for v in editor], color=teal, marker='o', linewidth=2.5, markersize=7)
    for i, value in enumerate(editor):
        ed.text(i, value['mean_reward_delta_pp']+.45, f"+{value['mean_reward_delta_pp']:.2f}", ha='center', color=teal)
    ed.set(title='改写器：固定 W2 的开发检查', ylabel='候选平均任务收益（百分点）',
           xticks=range(3), xticklabels=['E1c\nGRPO前', 'E2\n第1轮后', 'E3\n第2轮后'],
           ylim=(0, 10), xlim=(-.35, 2.35))
    ed.grid(axis='y', alpha=.20)
    ed.text(1, 1.4, '改写器权重已更新\n这组固定检查的平均收益未上升', ha='center', fontsize=11, color='#555555')
    handles = [Line2D([0], [0], color=orange, linestyle='--', linewidth=2.5, label='H阶段：未接受新代码，复用相同执行状态的分数'),
               Line2D([0], [0], color=blue, linewidth=2.5, label='W阶段：更新求解器权重')]
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.065, .90), frameon=False, ncol=2, fontsize=11)
    fig.text(.07, .17, '左：已有173题 × 每题8次采样；当前新增训练、模型调用均为0。以后评测按单seed、每题一次执行。', fontsize=10.5)
    fig.text(.07, .115, '右：每个改写器使用相同2个开发题包、共8个提案；是小规模能力检查，不是独立测试成绩。', fontsize=10.5)
    fig.text(.07, .06, '任务测评中160题有历史使用记录。曲线展示实际更新顺序和观测结果，不保证每一步或两个方向都提高。', fontsize=10.5, color='#555555')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(out / f'alternating_curve.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    print(json.dumps({'figure': str(out / 'alternating_curve.png'), 'stages': values,
                      'new_model_calls': 0}, ensure_ascii=False))


if __name__ == '__main__':
    main()
