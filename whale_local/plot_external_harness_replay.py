"""Plot audited single-draw external/internal harness comparison, without inference."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', type=Path, required=True)
    run = parser.parse_args().run.resolve()
    stats = json.loads((run / 'statistics.json').read_text())
    assert stats['audit']['passed_audit']
    font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams.update({'font.family': font.get_name(), 'font.size': 12, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'})
    data = stats['stages']; x = list(range(len(data)))
    internal = [p['internal_percent'] for p in data]; external = [p['percent'] for p in data]
    fig, ax = plt.subplots(figsize=(11.5, 6.7))
    fig.subplots_adjust(left=.10, right=.97, bottom=.30, top=.81)
    ax.plot(x, internal, '--s', color='#8d949c', label='内部改写器：执行代码保持 H3', linewidth=2, markersize=7)
    ax.plot(x, external, '-o', color='#176baa', label='外部改写器：本次选择的 harness', linewidth=2.5, markersize=8)
    for i, (a, b) in enumerate(zip(internal, external)):
        ax.annotate(f'{a:.2f}%', (i, a), xytext=(0, -23), textcoords='offset points', ha='center', color='#63676b')
        ax.annotate(f'{b:.2f}%', (i, b), xytext=(0, 12), textcoords='offset points', ha='center', color='#176baa')
    ax.set_xticks(x, ['起点\nW2 + H3', '第1轮\nharness 选择后', '载入既有\nW3 权重', '第2轮\nharness 选择后', '载入既有\nW4 权重'])
    ax.set_ylabel('173 题单次正确率（%）')
    lo, hi = min(internal + external), max(internal + external)
    ax.set_ylim(max(0, lo - 6), min(100, hi + 6)); ax.grid(axis='y', alpha=.20)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1.22), frameon=False, ncol=2)
    fig.suptitle('外部 harness 改写对照：相同搜索预算，复用冻结权重', fontsize=18, fontweight='bold', y=.98)
    fig.text(.10, .16, '同一批 173 题，每题只取 seed 42 的一次采样；内部基线复用已审计的同 seed 记录。', fontsize=11)
    fig.text(.10, .105, '两轮不同候选—题包评测上限为 11 / 10；每组 8 题，另各 2 个候选进入 24 题选择集。', fontsize=11)
    fig.text(.10, .05, '本轮训练更新为 0。W3/W4 来自旧 H3 下的训练；图中换权重的阶段不代表本次重新训练。', fontsize=11, color='#666666')
    for suffix in ['png', 'pdf', 'svg']:
        fig.savefig(run / ('comparison_curve.' + suffix), dpi=180)
    plt.close(fig)
    print(str(run / 'comparison_curve.png'))


if __name__ == '__main__': main()
