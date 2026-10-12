"""Report measured checkpoint quality and each coordinate of the fixed trial."""
import argparse
import datetime
import json
from pathlib import Path

from pipeline import sha,write_json


def read(path):return json.loads(Path(path).read_text())
def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--initial-only',action='store_true')
    args=p.parse_args();run=args.run.resolve();config=read(run/'config.json')
    audit=read(run/('initial_verification.json' if args.initial_only else 'verification.json'));assert audit['passed_audit']
    lines=['# 同一 MATH 任务上的双 checkpoint 交替更新','',
        '任务保持为 MATH number_theory、Level 2–4、整数答案和 ReTool Python 工具环境。内部 harness 模型与求解模型独立保存参数、优化器和版本。在线代码全部由内部 2B 生成。','',
        '本轮参考 [Harness-R1 的 GRPO 目标](https://arxiv.org/html/2608.02276v1#S3.SS2)：固定求解模型，在同一批任务上评估候选相对当前 harness 的解题率变化；每包 8 个候选形成组内相对优势，再优化改写模型。无格式奖励、无外部模型评分。保留 WHALE 的 Python harness 接口和原生求解器 RSFT；这是本地适配，不是原版 Relax/Megatron 后端复现。','',
        '## 首先检查两个角色的 checkpoint','',
        '改写模型对照使用同样的两个失败包（各 4 题）、每包 4 个提案、相同生成种子、固定求解模型 W2。E0 是 focused SFT 后的模型；E1 是上一轮两条正收益提案 SFT 后的模型。后续 E2/E3 沿用相同评估输入、温度 0.7 和种子。','',
        '| 改写 checkpoint | AST 合法 / 提案 | 不同合法代码 | 正收益提案 | 负收益提案 | 平均解题率变化 |','|---|---:|---:|---:|---:|---:|']
    for path in sorted((run/'editor_probes').glob('E*/score_summary.json')):
        name=path.parent.name;summary=read(path);generated=rows(path.parent/'generations.jsonl')
        lines.append(f'| {name} | {sum(g["valid_source"] for g in generated)}/{len(generated)} | {summary["unique_valid_sources"]} | {summary["positive_delta"]}/{len(generated)} | {summary["negative_delta"]}/{len(generated)} | {100*summary["mean_reward_delta"]:+.2f} pp |')
    lines+=['','两个预先固定检查包的分项均值如下；每包只有 4 道题、4 个提案，分项用于描述能力变化。','',
        '| 改写 checkpoint | 从简单 harness 添加恢复能力：平均收益 | 在 H3 上修改 guidance：平均收益 |','|---|---:|---:|']
    for file in sorted((run/'editor_probes').glob('E*/scored.jsonl')):
        proposals=rows(file);means=[]
        for group in ['probe_finalize','probe_concise']:
            values=[r['reward_delta'] for r in proposals if r['group']==group]
            means.append(100*sum(values)/len(values))
        lines.append(f'| {file.parent.name} | {means[0]:+.2f} pp | {means[1]:+.2f} pp |')
    lines+=['','这些是两个固定失败包的小规模提案检查；不同候选复用同一组任务，不是更多独立测试题。合法代码、不同代码和真实收益分别统计；完全相同的候选按相同模型、任务、种子缓存结果，仍计入提案预算。','',
        f'初测 E0/E1 输出高度重复，均未显示提案收益。本次训练探索温度预先设为 {config.get("training_proposal_temperature",1.2)}；仅在两个奖励组都无差异时，允许额外一个温度 {config.get("fallback_proposal_temperature",1.5)} 的候选批次。固定 checkpoint 对照的温度保持 0.7。','',
        '求解器起点使用已有 W2。本轮监测使用原 32 题的前两次采样，初始数据完全复用此前已审计记录：','',
        '| 初始求解组合 | 成功采样 / 64 |','|---|---:|']
    initial=read(run/'initial_checkpoint_results.json')
    for name,point in initial['solver_monitor'].items():lines.append(f'| {name} | {int(point["successful_draws"])}/64 |')
    lines+=['','分母 64 = 32 道题 × 2 次采样。它是此前用过的重复监测集，不是新鲜独立测试集；不进入奖励、选候选或调整本轮超参数。','']
    if config.get('composition_bootstrap'):
        bridge=Path(config['composition_bootstrap']);ready=read(bridge/'READY.json')
        states=[read(p) for p in sorted(bridge.glob('editor_sft*/checkpoint/training_state.json'))]
        lines+=[f'两个前置尝试都在任何 E/W 更新之前停止：贪心奖励包已满分；改用随机采样后虽有失败题，但旧 E1 主要复制 incumbent 或产生无效代码，组内收益仍无差异。记录完整保留。随后在 E1 上继续进行 64 条纯合成代码编辑 SFT，共 {sum(s["epochs"] for s in states)} 个 epoch、{32*len(states)} 次更新，得到 E1c。没有改变求解器 W2 起点。',
            f'补训只教在已有恢复逻辑上增改 SYSTEM_PROMPT，不含 MATH 题目或答案，也不把示例标成实际收益。预先固定的代码开发检查得到 {ready["executable_preserving_changed_guidance"]}/8 可执行、保留 hooks 且改变 guidance 的提案，含 {ready["distinct_system_prompts"]} 种不同提示。E1c 也接受上表完全相同的固定 checkpoint 检查。',
            '在线请求只指定通用 guidance 编辑接口；具体策略由内部模型选择，再由真实任务收益优化。候选生成温度为 1.0，零方差时至多额外使用一次 1.2；任务奖励与候选选择使用温度 1.0、相同种子的配对采样。','']
        retention=[]
        for probe in sorted((run/'editor_probes').glob('E*/runtime_checks.json')):
            values=[r for r in read(probe)['results'] if r['id'].startswith('probe_finalize_')]
            retention.append(f'{probe.parent.name} {sum(r.get("recovers_partial_response",False) for r in values)}/{len(values)}')
        lines+=['本轮在线编辑范围是为已有 H3 增改 SYSTEM_PROMPT；恢复 hooks 来自此前模型生成的 H3。额外的旧能力检查要求从简单 harness 新增 partial-response recovery，各 checkpoint 实际做到的提案数为：'+'；'.join(retention)+'。这项接口遵循率与解题收益单独记录，避免用平均奖励掩盖旧编辑能力的退化。','']
    elif config.get('initial_probe_origin'):
        lines+=['首次交替尝试的两个贪心奖励包均为 8/8，16 个常规提案只有一种合法代码且全部为 no-op。该尝试在任何 GRPO 或求解器更新之前停止，保留记录。本次沿用完全相同的 E1/W2 和任务划分，将奖励及 harness 选择改为 temperature=1 的同种子配对采样，并提高代码探索温度；初测结果直接复用。','']
    if not args.initial_only:
        lines+=['## 两轮交替轨迹','','每轮按 E 的 GRPO → 更新后的 E 重新生成候选 → 在 24 道选择题上选 harness → 128 道 TRAIN 各采样一次 → W 的原生 RSFT 顺序执行。奖励题、提案检查题和选择题在本轮内互不重叠。','',
            '| 阶段 | 成功采样 / 64 | 平均成功率 |','|---|---:|---:|']
        for point in audit['monitor_curve']:
            lines.append(f'| {point["label"]} | {int(point["successful_draws"])}/64 | {100*point["successful_draws"]/64:.2f}% |')
        lines+=['','| 相邻阶段比较 | 变化 | 按题配对 bootstrap 95% 区间 |','|---|---:|---:|']
        for contrast in audit['adjacent_monitor_changes']:
            lo,hi=contrast['paired_bootstrap_95_percent_pp']
            lines.append(f'| {contrast["from"]} → {contrast["to"]} | {contrast["difference_pp"]:+.2f} pp | [{lo:+.2f}, {hi:+.2f}] |')
        total=audit['total_monitor_change'];lo,hi=total['paired_bootstrap_95_percent_pp']
        lines+=['',f'最终相对本轮起点的变化为 {total["difference_pp"]:+.2f} pp，按题配对 bootstrap 95% 区间 [{lo:+.2f}, {hi:+.2f}]；32 题中平均成功率上升 {total["new_successes"]} 题、下降 {total["regressions"]} 题。该区间只描述这组重复监测数据的抽样不确定性。','']
        lines+=['','## 改写权重是否真的更新并交接','','| 轮次 | E 更新次数 | 同预算提案：更新前 → 更新后平均收益 | 执行 harness 选择 | W 更新次数 |','|---|---:|---:|---|---:|']
        for cycle in sorted(run.glob('cycle_*')):
            done=read(cycle/'complete.json');trained=read(cycle/'editor_grpo/complete.json')
            e_updates=read(cycle/'editor_grpo/checkpoint/training_state.json')['epochs'] if trained['updated'] else 0
            source=read(cycle/'TRAIN_SOURCE.json');before=rows(Path(source['directory'])/'scored.jsonl')
            after=rows(cycle/'editor_post_proposals/scored.jsonl');post_ids={r['id'] for r in after};before=[r for r in before if r['id'] in post_ids]
            assert len(before)==len(after)==8
            before_reward=100*sum(r['reward_delta'] for r in before)/8;after_reward=100*sum(r['reward_delta'] for r in after)/8
            choice=read(cycle/'selection/decision.json')['accepted'];state=read(Path(done['solver'])/'training_state.json')
            lines.append(f'| {cycle.name} | {e_updates} | {before_reward:+.2f} → {after_reward:+.2f} pp | {choice} | {int(state["metrics"]["actor/sft_updates"])} |')
        lines+=['','| 轮次 | 原 harness：24 题 | 更新后模型的入围候选：24 题 | 接受结果 |','|---|---:|---|---|']
        for cycle in sorted(run.glob('cycle_*')):
            decision=read(cycle/'selection/decision.json');values=decision['results']
            candidates='；'.join(f'{name}: {value["solved_examples"]}/24' for name,value in values.items() if name!='incumbent')
            lines.append(f'| {cycle.name} | {values["incumbent"]["solved_examples"]}/24 | {candidates} | {decision["accepted"]} |')
        lines+=['','该表中的前后提案对照使用相同 8 个请求/种子，但任务是 GRPO 奖励题，因此只说明训练题上的变化；独立于本轮奖励的固定失败包结果见上表 E0/E1/E2/E3。','',
            'GRPO 的第一次更新从 SFT 权重初始化新的 AdamW 状态（betas 0.9/0.98、weight decay 0.1）；之后延续该优化器。裁剪为 0.2/0.28，重要性权重上限 2，每批 4 次更新。求解器保留原 AdamW 状态，学习率在加载后明确设为 3e-7。','',
            '所有通过数值核验的求解器更新都会交接，即使监测成绩下降；harness 选择包含 incumbent。没有按监测结果回滚或挑最好版本来制造上升曲线。','']
        editor_seconds=sum(read(p)['elapsed_seconds'] for p in run.glob('cycle_*/editor_grpo/checkpoint/training_state.json'))
        solver_seconds=sum(read(Path(read(c/'complete.json')['solver'])/'training_state.json')['elapsed_seconds']
            for c in run.glob('cycle_*'))
        started=datetime.datetime.fromisoformat(read(run/'PROTOCOL.json')['registered_utc'])
        ended=datetime.datetime.fromtimestamp((run/'verification.json').stat().st_mtime,datetime.timezone.utc)
        discarded=[]
        for previous in config.get('previous_stopped_attempts',[]):
            directory=Path(previous)
            count=sum(len(p.read_text().splitlines()) for p in directory.glob('cycle_*/editor_pre*/generations.jsonl'))
            discarded.append(f'{directory.name}: {count} 条已记录提案，0 次 E/W 更新')
        lines+=['## 本轮成本','',
            f'固定交替运行至审计结束约 {(ended-started).total_seconds()/60:.1f} 分钟（不含前置尝试和补充 SFT）。实际 E-GRPO 训练记录共 {editor_seconds/60:.1f} 分钟、W-RSFT 共 {solver_seconds/60:.1f} 分钟；求解器采样预算为 {config["cycles"]*128} 条训练轨迹。',
            f'审计覆盖 {audit["verified_task_trajectories"]} 条任务轨迹、{audit["verified_unique_task_calls"]} 个去重原始求解调用，包含复用的初测记录；复用初测调用为 {audit.get("reused_initial_task_calls",0)} 个。因此该数不是全部新增训练样本数。',
            '前置尝试的已完成生成记录：'+'；'.join(discarded)+'。未把它们计入本轮 GRPO 更新。','']
        if (run/'TRANSPORT_AMENDMENT.json').exists():
            lines+=['第一轮 W 训练后的首次服务交接遇到 HTTP 400：新 checkpoint 不在单个采样服务的原目录范围内。修复为明确列举本轮模型的加载白名单，服务消费者分别保存重载证明；保留失败服务和全部记录，在尚未产生该阶段监测轨迹时恢复。已完成 E/W 训练未重复。详见 `TRANSPORT_AMENDMENT.json`。','']
        final=read(run/'FINAL_MODELS.json');directory=run/'checkpoints';directory.mkdir(exist_ok=True)
        current={'harness_generator':{'checkpoint':final['editor']['model'],'weight_digest':final['editor']['weight_digest']},
            'task_solver':{'checkpoint':final['solver']['model'],'weight_digest':final['solver']['weight_digest']},
            'execution_harness':final['harness'],'verification_sha256':sha(run/'verification.json'),
            'selection_note':'Latest numerically audited alternating updates; not selected using monitoring scores.'}
        for name,role in [('harness_current','harness_generator'),('solver_current','task_solver')]:
            link=directory/name;target=Path(current[role]['checkpoint'])
            if link.is_symlink():assert link.resolve()==target.resolve()
            else:link.symlink_to(target,target_is_directory=True)
        write_json(run/'CURRENT_MODELS.json',current)
        lines+=['## 当前两个 checkpoint','',f'- Harness：`{directory / "harness_current"}`',
            f'- Solver：`{directory / "solver_current"}`','',
            '两角色都保留完整语言权重与优化器，视觉参数冻结。逐参数哈希、实际 optimizer 步数、成功轨迹原始 token、组内奖励、生成器版本交接和服务退出均已独立核验。','']
    lines+=['## 解释边界','',
        '这次检验的是现有小规模任务上的交替行为。奖励差异为零就没有 GRPO 学习信号，不能把一次空更新称为改进；参数改变也不等于提案质量提高。多个阶段重复查看同一监测集，不能据此证明总体泛化或保证单调上升。',
        '当前 SFT 的编辑范围仍有限，输入包含预定义编辑接口；本轮没有证明开放式诊断、同预算优于外部 WHALE proposer，或样本效率提升。',
        '推理最多使用本机 4 张空闲卡加内网 10 张；模型训练分别用本机单张改装 48GB 4090，FP32 参数/优化器、BF16 autocast。没有进行跨机梯度同步。首次启动时 SSH 控制连接超时退出，发生在新任务评分前，已保留启动日志并重新连接。','',
        f'审计文件：`{"initial_verification.json" if args.initial_only else "verification.json"}`。','']
    if not args.initial_only:
        first,last=audit['monitor_curve'][0],audit['monitor_curve'][-1]
        e_steps=sum(read(p)['epochs'] for p in run.glob('cycle_*/editor_grpo/checkpoint/training_state.json'))
        w_steps=sum(int(read(Path(read(c/'complete.json')['solver'])/'training_state.json')['metrics']['actor/sft_updates'])
            for c in run.glob('cycle_*'))
        selected=sum(read(c/'selection/decision.json')['accepted']!='incumbent' for c in run.glob('cycle_*'))
        lines[2:2]=[f'两轮交替已完成并通过审计：改写器真实进行 {e_steps} 次 GRPO 更新，求解器真实进行 {w_steps} 次 RSFT 更新。重复监测集（32 题 × 2 次采样）从 {int(first["successful_draws"])}/64 变为 {int(last["successful_draws"])}/64；两轮共接受 {selected} 次新 harness。参数更新、提案奖励改善与最终任务提升在下文分别列出。','']
    target=run/('INITIAL_REPORT.md' if args.initial_only else 'REPORT.md');target.write_text('\n'.join(lines));print(target)


if __name__=='__main__':main()
