"""Verify recorded identities and draw the completed three-arm pilot."""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, v): Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=2) + '\n')


def historical_alignment(run, cfg, curves):
    historical_run = Path(cfg['solver_initial_checkpoint']).parents[1]
    old_cfg = read(historical_run / 'config.json')
    split = cfg['data']['splits']['test']
    assert split['sha256'] == old_cfg['data']['splits']['confirmation']['sha256'] == sha(split['path'])
    current = {r['task_id']: r for r in rows(Path(curves['external'][0]['output']) / 'trajectories.jsonl')}
    configs = []
    labels = {'W0_H0': '初始weight＋初始harness', 'W1_H0': '只提高weight',
              'W0_Hf': '只提高harness', 'W1_Hf': '两者结合（历史harness）'}
    for key, label in labels.items():
        source = historical_run / 'confirmation' / key / 'trajectories.jsonl'
        rr = rows(source)
        by_draw = {d: {r['task_id']: r for r in rr if r['draw'] == d} for d in range(8)}
        assert len(rr) == 1024 and all(set(group) == set(current) and len(group) == 128 for group in by_draw.values())
        counts = [int(sum(r['reward'] for r in group.values())) for group in by_draw.values()]
        if key == 'W1_H0':
            assert all(by_draw[0][k]['reward'] == current[k]['reward'] for k in current)
            assert all(by_draw[0][k]['rollout_seed'] == current[k]['rollout_seed'] for k in current)
            assert all(by_draw[0][k]['sampling_identity']['weight_digest'] == current[k]['sampling_identity']['weight_digest'] for k in current)
        configs.append({'configuration': key, 'label': label, 'source': str(source),
            'source_sha256': sha(source), 'solved_by_draw': counts, 'draw': 0, 'solved': counts[0],
            'total': 128, 'accuracy_pct': counts[0] / 128 * 100,
            'historical_mean_at_8_pct': sum(counts) / 1024 * 100})
    alignment = {'metric': 'single_sample_accuracy', 'draw': 0, 'tasks': 128,
        'split_sha256': split['sha256'], 'same_tasks_verified': True,
        'start_weight_digest_and_per_task_rewards_match_historical_draw0': True,
        'historical_points_recomputed_from_saved_trajectories': True, 'new_model_calls': 0,
        'configurations': configs}
    save(run / 'STATISTICAL_ALIGNMENT.json', alignment)
    with (run / 'historical_draw0_data.csv').open('w', newline='') as f:
        fields = ['configuration', 'label', 'draw', 'solved', 'total', 'accuracy_pct', 'source']
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(configs)
    return alignment


def main(run):
    cfg = read(run / 'config.json'); done = read(run / 'COMPLETE.json'); curves = done['curves']
    assert read(run / 'STATUS.json')['stage'] == 'complete'
    names = {'external': '外部模型改写', 'fixed_editor': '固定内部2B改写器', 'trained_editor': '持续训练内部2B改写器'}
    ids = None; records = []; summary = {}; start_scores = []
    for arm, points in curves.items():
        assert [p['label'] for p in points] == ['start', 'H1', 'W1', 'H2']
        assert points[0]['model'] == cfg['solver_initial_checkpoint']
        assert sha(points[0]['harness']) == sha(cfg['initial_harness'])
        assert points[0]['model'] == points[1]['model']
        assert points[2]['model'] == points[3]['model']
        assert sha(points[1]['harness']) == sha(points[2]['harness'])
        for p in points:
            rr = rows(Path(p['output']) / 'trajectories.jsonl')
            assert len(rr) == p['total'] == 128 and sum(r['reward'] for r in rr) == p['solved']
            current = [(r['task_id'], r['draw']) for r in rr]
            if ids is None: ids = current
            assert current == ids and all(r['draw'] == 0 for r in rr)
            digest = read(Path(p['model']) / 'training_state.json')['weight_digest']
            assert all(r['sampling_identity']['weight_digest'] == digest for r in rr)
            assert all(r['harness_sha256'] == sha(p['harness']) for r in rr)
            records.append({'arm': arm, 'stage': p['label'], 'solved': p['solved'], 'total': 128,
                'accuracy': p['solved']/128, 'accuracy_pct': p['solved']/128*100,
                'metric': 'single_sample_accuracy', 'draw': 0,
                'solver_digest': digest, 'harness_sha256': sha(p['harness'])})
        start_scores.append(points[0]['solved'])
        train = read(Path(points[2]['model']) / 'training_state.json')
        assert train['rollouts'] == 32
        assert train['sampling_identity']['weight_digest'] == cfg['initial_solver']['weight_digest']
        assert train['harness_sha256'] == sha(points[1]['harness'])
        assert train['changed_parameter_tensors']
        solver_steps = max(train['optimizer_steps_after']) - max(train['optimizer_steps_before'] or [0])
        cache_paths = set(); phases = []; editor_steps = 0; editor_tokens = 0
        for n in (1, 2):
            c = run / arm / f'phase_{n:02d}'
            start = read(c / 'START.json'); er = read(c / 'EDITOR_RESULT.json'); decision = read(c / 'selection/decision.json')
            assert start['model']['weight_digest'] == read(Path(points[0 if n==1 else 2]['model']) / 'training_state.json')['weight_digest']
            groups = {}
            for stage in ['pre', 'post']:
                rs = rows(c / stage / 'scored.jsonl')
                assert len(rs) == (8 if stage == 'pre' else 4)
                if arm != 'external': editor_tokens += sum(len(r['response_ids']) for r in rs)
                for r in rs:
                    assert r['reward_provenance']['task_count'] == 48
                    assert r['reward_provenance']['solver_weight_digest'] == start['model']['weight_digest']
                    if r['outcome_kind'] == 'valid':
                        val = read(Path(r['reward_provenance']['candidate']) / 'val.json')
                        assert abs(r['reward_delta'] - (val['success_rate'] - r['reward_provenance']['baseline_score'])) < 1e-9
                groups[stage] = {'slots': len(rs), 'valid_edits': sum(r['outcome_kind'] == 'valid' for r in rs),
                    'unique_valid_edits': len({r['source'] for r in rs if r['outcome_kind'] == 'valid'}),
                    'positive_gain': sum(r['reward_delta'] > 0 for r in rs), 'rewards': [r['reward_delta'] for r in rs],
                    'matched4_mean_delta_questions': sum(r['reward_delta'] * 48 for r in rs[:4]) / 4}
                if arm != 'external':
                    assert all(r['sampling_identity']['weight_digest'] == er['before' if stage == 'pre' else 'after']['weight_digest'] for r in rs)
            updates = 0
            if er['training']['updated']:
                state = read(Path(er['training']['identity']['weights']) / 'training_state.json')
                assert er['training']['reload_verified'] and state['changed_parameter_tensors']
                updates = max(state['optimizer_steps_after']) - max(state['optimizer_steps_before'] or [0]); editor_steps += updates
            if arm == 'fixed_editor':
                assert er['before'] == er['after'] == cfg['initial_editor']
            if arm != 'external':
                pre, post = rows(c / 'pre/scored.jsonl'), rows(c / 'post/scored.jsonl')
                assert all(a['messages'] == b['messages'] and a['seed'] == b['seed'] for a,b in zip(pre[:4],post))
            for cache in [c / 'incumbent/CACHE.json', *c.glob('pre/scores/*/CACHE.json'), *c.glob('post/scores/*/CACHE.json')]:
                cache_paths.add(read(cache)['source'])
            phases.append({'phase': n, 'editor_updates': updates, 'harness_changed': decision['changed'],
                'selected': decision['accepted'], 'development_scores': {k:v['solved_examples'] for k,v in decision['results'].items()},
                'proposal_groups': groups, 'matched4_changed_sources': er['matched4_changed_sources']})
        search_rows = [r for p in cache_paths for r in rows(Path(p) / 'trajectories.jsonl')]
        rollout_rows = rows(run / arm / 'weight_phase/solver_rollouts/trajectories.jsonl')
        def tokens(rr): return sum(len(c['response_ids']) for r in rr for c in r['policy_calls'])
        summary[arm] = {'scores': [p['solved'] for p in points], 'stage_deltas': [points[i]['solved'] - points[i-1]['solved'] for i in range(1,4)],
            'search_rollouts_distinct_within_arm': len(search_rows), 'search_generated_tokens': tokens(search_rows),
            'solver_rollouts': len(rollout_rows), 'solver_rollout_tokens': tokens(rollout_rows),
            'successful_sft_trajectories': train['accepted_trajectories'], 'sft_response_tokens': train['response_tokens'],
            'editor_proposal_tokens_logical_slots': editor_tokens if arm != 'external' else None,
            'solver_updates': solver_steps, 'editor_updates': editor_steps, 'phases': phases}
        assert len(search_rows) <= cfg['logical_harness_rollout_cap_per_phase'] * 2
    assert len(set(start_scores)) == 1
    alignment = historical_alignment(run, cfg, curves)
    with (run / 'curve_data.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    elapsed = (datetime.datetime.fromisoformat(done['utc']) - datetime.datetime.fromisoformat(read(run/'START.json')['utc'])).total_seconds()/60
    initialization = read(run / 'EDITOR_INITIALIZATION.json')
    matched = read(run / 'EDITOR_MATCHED_COMPARISON.json')
    assert matched['same_h1_source'] and matched['same_w1_digest']
    assert all(p['same_fixed_and_trained_editing_inputs_and_seeds'] for p in matched['phases'])
    limits = [s.replace('historical SFT/outcome training', '132 historical SFT updates, zero prior RL updates') for s in cfg['limits']]
    save(run / 'RESULT.json', {'basic_checks_passed': True, 'full_audit': False, 'elapsed_minutes_including_restarts': elapsed,
        'same_test128_verified': True, 'shared_start_verified': True, 'training_under_own_harness_verified': True,
        'summary': summary, 'editor_initialization': initialization, 'matched_editor_comparison': matched,
        'statistical_alignment': alignment, 'limitations': limits})
    plt.rcParams.update({'font.family':'Noto Sans CJK SC','font.size':11,'axes.unicode_minus':False,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    colors={'external':'#2563B5','fixed_editor':'#818A98','trained_editor':'#C76A16'}
    fig, grid = plt.subplots(2, 2, figsize=(14.8, 10.5), sharey=True)
    axes = grid.ravel()
    history = alignment['configurations']
    for i, point in enumerate(history):
        axes[0].plot(i, point['accuracy_pct'], 's', color='#2563B5' if i == 1 else '#64748B', markersize=8)
        axes[0].annotate(f'{point["solved"]}/128\n{point["accuracy_pct"]:.2f}%',
            (i, point['accuracy_pct']), xytext=(0, 12), textcoords='offset points', ha='center', fontsize=12, weight='bold')
    axes[0].set_xticks(range(4), ['初始weight\n初始harness', '只提高weight\n（交替共同起点）',
        '只提高\nharness', '两者结合\n（历史harness）'], fontsize=10)
    axes[0].set_title('历史四种配置：第1次采样（独立组合）', pad=14, fontsize=14, color='#475569')
    for ax,(arm,s) in zip(axes[1:],summary.items()):
        yy=[v/128*100 for v in s['scores']]; ax.plot(range(4),yy,'o-',color=colors[arm],lw=2.7,markersize=7)
        for i,y in enumerate(yy): ax.annotate(f'{s["scores"][i]}/128\n{y:.2f}%',(i,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=12,weight='bold')
        for i,d in enumerate(s['stage_deltas']): ax.annotate(f'{d:+d}题' if d else '0题',(i+.5,(yy[i]+yy[i+1])/2),xytext=(0,-21),textcoords='offset points',ha='center',color=colors[arm])
        ticks=['已训练weight\n＋初始harness','第一次\nharness',f'求解器+{s["solver_updates"]}步\n权重训练','第二次\nharness']
        if arm == 'trained_editor':
            ticks[1]=f'改写器+{s["phases"][0]["editor_updates"]}步\n第一次harness'
            ticks[3]=f'改写器+{s["phases"][1]["editor_updates"]}步\n第二次harness'
        ax.set_xticks(range(4),ticks,fontsize=10)
        ax.set_title(names[arm] + '：H → W → H',pad=14,fontsize=14,color=colors[arm])
    for ax in axes:
        ax.set_ylim(35,85);ax.set_xlim(-.3,3.3)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.axhline(start_scores[0]/128*100, color='#B6C3D2', linestyle='--', linewidth=1, zorder=0)
        ax.grid(axis='y',alpha=.16);ax.tick_params(axis='x',length=0,pad=10)
    axes[0].set_ylabel('成功率（答对题数 / 128）')
    axes[2].set_ylabel('成功率（答对题数 / 128）')
    fig.suptitle('MATH 数论：同一128题，每题采样1次',fontsize=21,weight='bold',y=.98)
    fig.text(.5,.935,'统一采用 draw 0，temperature=1，seed=42；每点同时标注题数和成功率',ha='center',fontsize=12)
    fig.text(.06,.085,'虚线＝交替共同起点 78/128（60.94%）；已有求解器完成512条历史训练采样、120步更新。',fontsize=10,color='#526174')
    fig.text(.06,.057,'左上取历史原始记录的第1次采样，其余为本轮逐阶段实测；历史harness与本轮重新搜索的harness不同。',fontsize=10,color='#526174')
    fig.text(.06,.029,'三组提案数和采样上限相同，实际采样量不同；这是本地WHALE式快速对照，非论文完整复现。',fontsize=10,color='#526174')
    fig.tight_layout(rect=[.015,.12,.995,.905], h_pad=3.0, w_pad=3.0)
    for ext in ['png','pdf','svg']:fig.savefig(run/f'three_arm_curve.{ext}',dpi=180)
    plt.close(fig)
    table='\n'.join(f'| {names[a]} | '+ ' | '.join(f'{x}/128（{x/128*100:.2f}%）' for x in s['scores'])+' |' for a,s in summary.items())
    history_table='\n'.join(f'| {p["label"]} | {p["solved"]}/128 | {p["accuracy_pct"]:.2f}% | {p["historical_mean_at_8_pct"]:.2f}% |' for p in history)
    budget='\n'.join(f'| {names[a]} | {s["search_rollouts_distinct_within_arm"]} | {s["solver_rollouts"]} | {s["successful_sft_trajectories"]} | {s["solver_updates"]} | {s["editor_updates"]} |' for a,s in summary.items())
    token_table='\n'.join(f'| {names[a]} | {s["search_generated_tokens"]:,} | {s["solver_rollout_tokens"]:,} | '
        f'{s["editor_proposal_tokens_logical_slots"] if s["editor_proposal_tokens_logical_slots"] is not None else "未计量"} | '
        f'{s["sft_response_tokens"]:,} |' for a,s in summary.items())
    editor_table='\n'.join(
        f'| {names[a]} H{phase["phase"]} | {phase["editor_updates"]} | '
        f'{phase["proposal_groups"]["pre"]["matched4_mean_delta_questions"]:+.2f} | '
        f'{phase["proposal_groups"]["post"]["matched4_mean_delta_questions"]:+.2f} | '
        f'{phase["matched4_changed_sources"]}/4 |'
        for a,s in summary.items() if a != 'external' for phase in s['phases'])
    trained_final=summary['trained_editor']['scores'][-1]
    fixed_final=summary['fixed_editor']['scores'][-1]
    external_final=summary['external']['scores'][-1]
    extra_editor_gain=trained_final-fixed_final
    diagnosis=read(run/'INTERNAL_H1_TEST_DIAGNOSIS.json')
    report=f'''# 三组同起点的 H → W → H 快速对照

三组都从已有WHALE训练得到的同一求解器checkpoint与初始harness出发。内部两组使用同一份已有132步SFT初始化、尚未经过RL的改写器，区别是本轮是否继续更新它；“固定”不等于从未训练。初始化链条见EDITOR_INITIALIZATION.json。每个阶段均在相同128题、每题一次采样上实测，没有用64题结果补成128题。

数据是原始MATH的`number_theory`本地子集：保留Level2–4、标准boxed答案为规范整数的题目；官方train中选128题训练、48题开发，官方test中选128题confirmation测试。见[划分清单](../whale_math_sampling_confirmation_setup_20260915/manifest.json)。这组题历史上已使用，属于快速迭代评测；不是新的盲测，也不是完整MATH或MATH-500分数。

外部提案由当前Codex对话通过本机FastAPI队列提供，推理token和费用未计量；本轮没有配置独立、无人值守的外部模型API。

| 方法 | 共同起点 | 第一次harness后 | 求解器训练后 | 第二次harness后 |
|---|---:|---:|---:|---:|
{table}

本次内部训练组比固定编辑器最终多解{extra_editor_gain}题，即{extra_editor_gain/128*100:.2f}个百分点。在这两个内部臂中，第二轮改写前的求解器、harness、编辑输入与采样种子一致，因此这是本轮编辑器训练有效的初步信号。与外部模型组相比只多{trained_final-external_final}题，单次小实验不足以判断普遍优劣。内部曲线第一轮存在明显回落，仍不能称为稳定、单调的自我改进。

![三组交替曲线](three_arm_curve.png)

图中所有面板统一为同一128题、每题一次采样（draw 0）的成功率，同时标出题数；左上是历史四种独立配置，其余为本轮H→W→H交替。共同起点是已训练weight＋初始harness。历史mean@8只保留在下面的口径对照列，不进入图中。

| 历史配置 | 第1次采样答对题数 | 图中统一口径：单次成功率 | 历史mean@8（仅供对照） |
|---|---:|---:|---:|
{history_table}

历史只提高weight的8次采样依次答对78、73、63、69、80、72、63、71题，平均569/1024＝55.57%；第1次为78/128＝60.94%，与本轮共同起点逐题成败一致。图中的历史点来自已保存的原始轨迹，这次整理没有新增训练或模型评测。历史最终harness与本轮重新搜索的harness不同，不能把四种配置顺次解读为本轮交替阶段。核对数据见[统计口径与来源](STATISTICAL_ALIGNMENT.json)、[历史第1次采样CSV](historical_draw0_data.csv)。

| 方法 | 两阶段开发搜索rollout | 新求解器rollout | 成功SFT轨迹 | 求解器更新 | 改写器更新 |
|---|---:|---:|---:|---:|---:|
{budget}

开发搜索按每臂实际依赖的不同模型/harness/题目/采样结果计数，跨臂共享缓存仍分别计入各臂。每个harness阶段预算上限624条（48条incumbent＋8个训练前候选×48＋4个训练后候选×48），相同提案槽位不意味着实际rollout或总算力完全相同。内部改写器另有历史初始化成本和本轮梯度训练成本；外部模型推理token/费用未计量，不能用这张表直接宣称总费用或样本效率胜出。

| 方法 | 搜索生成token | 求解器训练采样生成token | 改写器提案生成token | 求解器SFT响应token |
|---|---:|---:|---:|---:|
{token_table}

内部改写器提案token按每个逻辑提案槽位计入，两阶段共24个槽位，跨分支缓存复用仍分别计入各臂。这里只报告各类用量，不把不同模型推理token、训练token和费用直接相加。

三个分支各在自己选中的harness下生成32条新轨迹，再用成功响应进行RSFT，学习率1e-7。不能把外部分支的harness直接配上内部分支训练的权重冒充其交替结果。完全相同W/H和原始轨迹可复用确定的训练checkpoint，复用记录单独保留。

改写器训练使用当前求解器在完整48道开发题上的相对成功率反馈，8个提案组成一组，最多4步outcome-GRPO；无奖励差异则跳过。训练前后4个匹配提案使用相同提示与采样seed。外部模型可根据自己第一批候选的开发反馈再提案；内部固定组权重不更新，内部训练组用这些反馈更新权重。三组都只从后4个合法候选与incumbent中按原生WHALE规则选择。代码接口、工具、总回答token和判题器相同。

求解器每道题总生成上限2048 token、单次调用上限1024 token；harness最多3轮。增加恢复轮次仍受相同总token上限约束。每题测试一次，temperature=1、seed=42；阶段数值是这次固定协议的实测点，没有估计跨seed方差。

| 内部改写阶段 | 实际梯度更新 | 训练前4候选平均增量题数 /48 | 训练后匹配4候选平均增量题数 /48 | 源码变化 |
|---|---:|---:|---:|---:|
{editor_table}

第二轮训练组的4个匹配提案，开发增量从0、4、0、0题变为8、4、8、8题：原样复制从3次变成0次，较强的恢复机制从0次变成3次。这支持“更新E能在当前反馈下改变并改善提案分布”。但42/48的强候选在训练前8个提案中已经出现；如果直接保留全部已评估候选，第二次E更新并不是获得它的必要条件。不能把更常重现已发现候选，写成首次发明了一个新机制或提高了本阶段所有已搜索候选的最高分。

这张表衡量同一输入与采样seed下的提案变化，invalid/no-op按协议计零增量；它是训练所用开发集上的反馈，不能单独证明改写器泛化。本轮两内部臂H1选中相同源码，W更新复用同一checkpoint，H2改写提示和种子也完全相同。因此内部两臂的末端差异能对应到E权重更新造成的提案变化，不是混入了不同的求解器；具体核对见[匹配比较](EDITOR_MATCHED_COMPARISON.json)。但不能把每个上升点都归因于E学到了新能力，也不能从这一次小实验推广到所有编辑任务。

本轮改写器梯度训练可用的反馈上下文最多两个，每个上下文8个提案；重复源码还会减少不同干预的数量。4次优化器更新不等于4份新的独立执行反馈。这里检验的是一次很小的在线更新能否改善当前编辑过程，没有单独测试它对新任务、新harness的编辑泛化能力。

为了比较更新前后的相同4个提案槽位，本协议的最终选择池只包含post候选。因此固定内部组前8个提案中、未出现在后4个槽位的提案也不会被选择。这是预先固定的比较协议，不代表冻结编辑器在同等搜索预算下的最佳策略；本实验不能据此给出一般性的编辑器样本效率优劣结论。

固定组在后4个槽位复用了训练前的提示和种子，因此这些提案会重复；这样便于隔离权重变化的作用，但没有给固定组4次新的独立探索机会。若要比较最优搜索效率，需要另行比较固定组使用新采样种子或更多不同候选的策略，本轮没有做这一实验。

第一轮内部候选的泛化失败需要单独保留：它把“修复当前错误、比较整数”的指导写成了全局系统提示，开发集从28/48升到38/48，测试却从78/128降到69/128。按题对齐，有{diagnosis['gained']}题从错变对、{diagnosis['lost']}题从对变错；最终响应缺少boxed标记的题数从{diagnosis['initial_harness']['no_box_marker']}变为{diagnosis['internal_H1']['no_box_marker']}。全局提示可能过度贴合少数失败上下文，但这仍是机制假设，没有做单独因果消融。记录见[失败诊断](INTERNAL_H1_TEST_DIAGNOSIS.json)。

这轮对后续方法的直觉是：求解器更新后，harness的收益和失败分布会变化，改写器应使用当前求解器重新执行得到的奖励；同时要防止把某次失败的局部描述直接推广成所有问题的系统提示。第二轮训练成功强化了条件触发的未完成答案恢复，第一轮训练却没有改善匹配提案的平均收益，这两种结果都应保留。

起点求解器的历史512条训练轨迹、120次RSFT更新是共享前缀，未重新训练。当前曲线不把历史54/128与新阶段混作同时重测的结果。本轮总耗时约{elapsed:.1f}分钟（含启动修复和重启）。

基本核对包括：题目与采样序号一致、W/H身份匹配、真实权重变化、加载验证、改写器前后输入匹配、求解器在各自harness下训练、先冻结后评测。没有新增多seed或全面审计。这是受限全代码接口的本地WHALE式复现，不是论文完整DAPO/AIME实验。

数据：[RESULT.json](RESULT.json)、[CSV](curve_data.csv)、[PDF](three_arm_curve.pdf)、[最终checkpoint与harness路径](COMPLETE.json)。Gate的完整流程与样本效率失败分析另见[报告](../whale_alternating_overview_20260918/GATE_METHOD_AND_FAILURE_ANALYSIS.md)，不将本三组编辑器对照视为gate实验。
'''
    (run/'REPORT.md').write_text(report)
    print(json.dumps({'scores':{a:s['scores'] for a,s in summary.items()},'minutes':elapsed,'report':str(run/'REPORT.md')},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run.resolve())
