"""Summarize the three rapid editor-method trials, preserving negative results."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from harness_delta import gain_rank_advantages


def read(p): return json.loads(Path(p).read_text())
def rows(p): return [json.loads(s) for s in Path(p).read_text().splitlines()]
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value): p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main(run):
    cfg = read(run / 'config.json'); reference = Path(cfg['source_run']); rank = Path(cfg['rank_run'])
    original = read(reference / 'RAPID_RESULT.json'); rr = read(rank / 'RESULT.json'); full = read(run / 'RESULT.json')
    read(run / 'COMPLETE.json'); assert original['basic_checks_passed']
    for branch in [rank, run]:
        done = read(branch / 'editor_update/complete.json'); state = read(Path(done['identity']['weights']) / 'training_state.json')
        assert done['updated'] and done['reload_verified'] and state['changed_parameter_tensors'] and state['new_updates'] == 8
        assert state['weight_digest'] != state['sampling_identity']['weight_digest']
        assert state['sampling_identity']['weights'] == cfg['editor_initial_checkpoint']
        assert not state['optimizer_reset'] and state['optimizer_steps_before'] == [8] and state['optimizer_steps_after'] == [16]
        calls = rows(branch / 'editor_update/requests.jsonl')
        measured = read(branch / 'editor_update/advantages.json')['advantages']
        assert all(abs(a-b) < 1e-9 for a,b in zip(measured,gain_rank_advantages(calls)))
    previous_pre = rows(Path(cfg['source_replay_cycle']) / 'pre/scored.jsonl'); corrected = rows(run / 'pre/scored.jsonl')
    assert len(previous_pre) == len(corrected) == 8
    for a, b in zip(previous_pre, corrected):
        for key in ['prompt_ids','response_ids','response_logprobs','sampling_identity']: assert a[key] == b[key]
        r = b['reward_provenance']; assert r['task_count'] == 48
        assert abs(b['reward_delta'] - (r['rescued'] - r['regressed'])/48) < 1e-9
    after = rows(run / 'post/scored.jsonl')
    assert len(after) == 2
    assert [g['seed'] for g in after] == [g['seed'] for g in corrected[:2]]
    assert all(a['messages'] == b['messages'] for a,b in zip(corrected,after))
    final = rows(run / 'final128/trajectories.jsonl'); old = rows(reference / 'final128/trajectories.jsonl')
    assert len(final) == len(old) == 128
    assert [(r['task_id'],r['draw']) for r in final] == [(r['task_id'],r['draw']) for r in old]
    assert sum(r['reward'] for r in final) == full['final128']
    target = read(Path(full['solver']) / 'training_state.json')['weight_digest']
    assert all(r['sampling_identity']['weight_digest'] == target and r['harness_sha256'] == digest(full['harness']) for r in final)
    same_rank_harness = digest(rr['harness']) == digest(original['checkpoints']['harness']) and rr['solver'] == original['checkpoints']['solver']
    rank128 = original['final128'] if same_rank_harness else rr['rank_solved128']
    calibration = read(run / 'CALIBRATION.json'); valid = [r for r in calibration['labels'] if r['kind']=='valid']
    end = datetime.datetime.fromisoformat(read(run / 'COMPLETE.json')['utc']); start = datetime.datetime.fromisoformat(read(reference / 'START.json')['utc'])
    result = {'basic_checks_passed': True, 'initial128': original['initial128'], 'paired128': original['final128'],
        'rank128': rank128, 'rank128_reused_identical_state': same_rank_harness, 'full_reward128': full['final128'],
        'full_reward_gain_over_paired': full['final128']-original['final128'],
        'full_reward_gain_over_turn_start': full['final128']-original['initial128'],
        'calibration': valid, 'full_reward_pre_matched2_mean': sum(g['reward_delta'] for g in corrected[:2])/2,
        'full_reward_post_matched2_mean': sum(g['reward_delta'] for g in after)/2,
        'selected_dev48': full['selection_scores'][full['selected']], 'old_dev48': full['selection_scores']['incumbent'],
        'new_editor_sft_updates': 24, 'new_editor_rl_updates_across_branches': 32, 'new_solver_updates': 12,
        'new_calibration_rollouts': calibration['new_rollouts'], 'elapsed_minutes': (end-start).total_seconds()/60,
        'current_models': {k:full[k] for k in ['solver','editor','harness']}}
    save(run / 'COMBINED_RESULT.json', result)
    labels = ['Bounds', 'Python check', 'Tool verification', 'Side lengths']
    fig, axes = plt.subplots(1,2,figsize=(11,4.6))
    xx = list(range(len(valid)))
    axes[0].bar([x-.19 for x in xx],[100*r['sample12_gain'] for r in valid],width=.38,label='Failure-enriched 12')
    axes[0].bar([x+.19 for x in xx],[100*r['full48_gain'] for r in valid],width=.38,label='All 48 development tasks')
    axes[0].set_xticks(xx,labels,rotation=18,ha='right');axes[0].axhline(0,color='#777',lw=.8)
    axes[0].set_ylabel('Gain over same solver / harness (pp)');axes[0].set_title('The reward target can reverse edit rankings');axes[0].legend(fontsize=8)
    vals=[original['initial128'],original['final128'],full['final128']]
    axes[1].bar(['Turn start','First version','Full-reward\nrevision'],vals,color=['#9ba7b4','#ce7920','#356cb5'],width=.6)
    axes[1].set_ylim(0,128);axes[1].set_ylabel('Solved / 128');axes[1].set_title('Single-draw final task performance')
    for i,v in enumerate(vals):axes[1].text(i,v+2,f'{v}/128',ha='center')
    fig.text(.5,.01,'Same current solver for all edit-label comparisons. Final models selected on development tasks only.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,1])
    for ext in ['png','pdf','svg']:fig.savefig(run/('reward_calibration.'+ext),dpi=160)
    table='\n'.join(f'| {label} | {r["sample12_gain"]*12:+.0f}/12 | {r["full48_gain"]*48:+.0f}/48 |' for label,r in zip(['搜索边界','Python核验','工具结果核验','边长推导'],valid))
    (run/'REPORT.md').write_text(f'''# 改写器权重更新：三次快速方法迭代

本轮起点{original['initial128']}/128；第一版交替结束{original['final128']}/128；修正整体奖励后{full['final128']}/128。相对第一版{result['full_reward_gain_over_paired']:+d}题，相对本轮起点{result['full_reward_gain_over_turn_start']:+d}题。总墙钟时间约{result['elapsed_minutes']:.1f}分钟，包含两条复用样本的更新对照。

| 方法 | 最终128题 |
|---|---:|
| 本轮起点 | {original['initial128']} |
| 短修改 + 绝对增益更新，交替两次 | {original['final128']} |
| 相同样本改用有效候选收益排序 | {rank128 if rank128 is not None else '仅测64题'} |
| 用完整48题奖励重新训练改写器 | {full['final128']} |

排序分支若选择了与第一版相同的W/H，整表成绩复用第一版同一状态的128题评测，未重复采样。完整奖励分支固定同一求解器，只重训改写器。各分支的模型分别保留，不把不同分支的更新算成同一checkpoint连续训练。

![奖励修正与最终成绩](reward_calibration.png)

已跑通的改变：改写器输出短JSON修改，编译器保留已有执行钩子；48条已有合成编辑示例做24次格式SFT后，首组8/8可安装、4种不同修改。原来4次接口探测均失败。主流程每个改写器阶段都按当前求解器重新采集失败和计算候选奖励；两阶段各真实更新8次，求解器在中间更新12次。固定64题曲线为40→43→41→41，见[主实验报告](../whale_editor_delta_20260917/REPORT.md)。

第一阶段相同4个提案seed，训练前后正收益候选4/4→2/4；第二阶段1/4→4/4。然而第二阶段最优搜索边界候选在小批12题上多解3题，在完整48题上反而少解3题。仅改变优势排序也未解决这一失真，见[排序试验结果](../whale_editor_delta_rank_20260917/RESULT.json)。

| 同一候选、同一当前求解器 | 原12题奖励批次净变化 | 完整48题净变化 |
|---|---:|---:|
{table}

奖励批次的失败比例是50%，完整当前开发集是18.75%；少量按顺序选取的成功题也不能代表其余成功题。这说明改写器可以优化批次奖励，却同时损害整体任务表现。此次只补{calibration['new_rollouts']}条轨迹，复用已有执行结果，修正了同8个原始候选的训练标签。无需重新生成这些候选。

完整奖励版本沿用相同改写器起点和优化器状态，做8次收益排序更新，再用原前2个seed生成新候选。按完整48题计，匹配的训练前2个提案平均增益为{100*result['full_reward_pre_matched2_mean']:+.2f}个百分点，训练后为{100*result['full_reward_post_matched2_mean']:+.2f}个百分点；所选harness的开发成绩{result['old_dev48']}→{result['selected_dev48']}/48。2个提案用于快速判断，不与另外两条分支的4个提案混称同预算。

与一般固定目标的Harness-R1相比，交替训练需要以当前W、当前H和新失败重新标定编辑收益；保留历史样本时要重算收益。原论文已经要求同一批任务包含原本成功的任务，见[Harness-R1方法](https://arxiv.org/html/2608.02276v1)。本地试验中额外挑选失败题的做法引入了偏差，不能把这个问题归因于原论文算法。

实际执行共24次改写器格式SFT更新、跨分支32次RL更新、12次求解器更新。完成了权重变化和重载、原始采样token复用、奖励净变化、当前求解器身份及评测题目一致性的基本检查。没有多seed、全面审计或新的独立测试；题目有历史使用记录。当前动作仅修改解题和两处恢复提示，不是任意Python执行逻辑编辑。

方法细节见[METHOD.md](METHOD.md)，机器可读结果见[COMBINED_RESULT.json](COMBINED_RESULT.json)，最后一版两份checkpoint见[CURRENT_MODELS.json](CURRENT_MODELS.json)。
''')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run.resolve())
