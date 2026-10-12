"""Publish audited local results and two unambiguous current checkpoint links."""
import argparse
import datetime
import json
from pathlib import Path
from pipeline import write_json,sha

def read(p):return json.loads(Path(p).read_text())

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();run=args.run.resolve()
    audit=read(run/'verification.json');assert audit['passed_audit']
    frozen=read(run/'FROZEN.json');selection=read(run/'BOOTSTRAP_SELECTION.json')
    editor=Path(frozen['editor_checkpoint']);solver=Path(frozen['solver_identity']['weights'])
    models={'harness_generator':{'checkpoint':str(editor),'weight_digest':read(editor/'training_state.json')['weight_digest']},
        'task_solver':{'checkpoint':str(solver),'weight_digest':frozen['solver_identity']['weight_digest']},
        'execution_harness':{'path':frozen['harness'],'sha256':frozen['harness_sha256']},
        'verification_sha256':sha(run/'verification.json'),'role_optimizers_separate':True,
        'new_solver_update_difference_pp':audit['contrasts']['new_solver_update']['difference_pp'],
        'note':'Latest audited training states, not checkpoints selected for best test score. E1 versus frozen E0 proposal quality remains untested.'}
    for name,target in [('harness_current',editor),('solver_current',solver)]:
        link=run/'checkpoints'/name
        if link.is_symlink():assert link.resolve()==target.resolve()
        else:link.symlink_to(target,target_is_directory=True)
    write_json(run/'CURRENT_MODELS.json',models)
    state=read(solver/'training_state.json');e=read(editor/'training_state.json');mh=read(run/'MH_RESULT.json')
    names={'W0_H0':'原始 2B + 原始 harness','Wf_H0':'更新后的解题模型 + 原始 harness',
        'W0_Hf':'原始 2B + 内部生成的 harness','W1_Hf':'已验证的 WHALE 解题起点 + 内部 harness',
        'Wf_Hf':'本轮更新后的解题模型 + 内部 harness'}
    harness_gain=audit['contrasts']['harness_at_final_weights'];weight_gain=audit['contrasts']['new_solver_update']
    h_interval=harness_gain['paired_bootstrap_95_percent_pp'];w_interval=weight_gain['paired_bootstrap_95_percent_pp']
    weight_conclusion='显示正向增益' if w_interval[0]>0 else '未显示明确提升'
    lines=['# 内部 2B 双 checkpoint WHALE：本机试验结果','',
        '已运行内部模型生成候选、真实开发题验证、harness 生成器在线 RSFT、解题模型在线 RSFT，以及冻结后的新测试。运行时外部大模型调用为 0。','',
        f'本轮通过了执行与训练审计。固定最终权重，内部 harness 的测试增益为 {harness_gain["difference_pp"]:+.2f} 个百分点（95% 按题配对区间 [{h_interval[0]:+.2f}, {h_interval[1]:+.2f}]）；新增求解器训练为 {weight_gain["difference_pp"]:+.2f} 个百分点（[{w_interval[0]:+.2f}, {w_interval[1]:+.2f}]），{weight_conclusion}。模型可作为受限场景的继续实验起点，但尚未验证持续自我改进。','',
        '## 两个当前 checkpoint','',f'- Harness 生成器：`{run / "checkpoints/harness_current"}`',
        f'- 解题模型：`{run / "checkpoints/solver_current"}`',
        f'- 实际执行的 harness：`{frozen["harness"]}`','',
        '两份模型都是 Qwen3.5-2B，独立保存全量语言参数和 AdamW 状态，保留原始模型与中间版本。E1 再次生成的代码通过接口检查；本次测试使用的是先前完成开发集选择的代码，未擅自替换。','',
        '## 热启动检查（开发集）','','| 模型 | 可执行接口检查 | 指定 hook / 保留行为检查 |','|---|---:|---:|']
    for key,label in [('base','原始 2B'),('editor_continuation','继续 SFT'),('editor_from_base','从原始 2B 直接 SFT')]:
        s=selection['scores'][key];lines.append(f'| {label} | {s["code_contract_passed"]}/20 | {s["requested_hook_behavior_passed"]}/8 |')
    lines += ['', '新课程有 60 条合成教学样本、8 个机制族，包含保留已有恢复逻辑的修改；检查的是同类机制的新表述，不是任意代码修改能力。两条训练分支各训练 4 epochs / 60 步；按事先规则选用继续 SFT 的分支。第一版笼统反馈热启动失败，已单独保留，未打开本次测试集。','',
        '选中的 E0 还继承了此前 48 条通用示范、累计 6 epochs / 72 步的初始化，再做本轮 60 步，累计 132 步。直接从 base 训练的分支也通过开发行为检查，但本轮真实 harness 搜索使用的是 continuation 分支；不能把其全部能力归因于仅训练 60 条新示范。','',
        '运行时由失败计数选择预定义修改方向，2B 自己生成完整代码。这仍是有明确修改目标的受限起点，尚未验证开放式故障诊断。','',
        '## 开发集 harness 选择','','| 候选 | 解题数 |','|---|---:|']
    for name,s in mh['results'].items():lines.append(f'| {name} | {s["solved_examples"]}/{s["num_examples"]} |')
    duplicates=list(run.glob('editor_proposals_*/*_duplicate.json'))
    lines += ['',f'共生成 4 个候选，{len(duplicates)} 个与已评估代码完全重复而跳过。最终选用 {mh["accepted"]}。',
        '选中代码改变了停止条件：没有调用工具但尚未给出 boxed 答案时，保留进度并提示继续；工具返回错误时也追加继续完成的提示。最大交互轮次从 2 调整为 3，但总响应预算仍为 2048 token、每次调用最多 1024 token。',
        f'{mh["positive_proposals"]} 条严格改善开发解题数的原始代码生成记录用于 E 的 RSFT；优化器从 {e["optimizer_steps_before"]} 到 {e["optimizer_steps_after"]}，参数哈希改变并通过重新加载核验。',
        f'解题模型的新一轮采样 256 次，{state["accepted_trajectories"]} 次成功，训练 {state["accepted_policy_calls"]} 条真实助手调用 / {state["response_tokens"]} 个响应 token，执行 {int(state["metrics"]["actor/sft_updates"])} 次更新。',
        '解题起点 W1 复用了之前已审计的 WHALE 第一轮 checkpoint；它的 512 次训练采样属于已有预算，不是这次免费产生的提升。','',
        '## 冻结后的新测试','','场景为 MATH number_theory、Level 2–4、标准整数答案，使用 ReTool Python 工具环境。32 道此前未用过的题，每题 4 次采样，temperature=1。下表分母 128 是采样次数，不是 128 道独立题。','',
        '| 组合 | 成功采样 / 128 | mean@4 |','|---|---:|---:|']
    for key,label in names.items():
        s=audit['scores'][key];lines.append(f'| {label} | {s["successful_draws"]}/128 | {s["mean_at_4_percent"]:.2f}% |')
    lines += ['', '| 成对比较 | 变化（百分点） | 按题 bootstrap 95% 区间 |','|---|---:|---:|']
    labels={'harness_at_final_weights':'固定最终权重，替换内部 harness','new_solver_update':'固定内部 harness，新增一次权重训练',
        'all_solver_updates':'固定内部 harness，全部解题权重训练','joint_change':'相对原始模型与原始 harness'}
    for key,label in labels.items():
        v=audit['contrasts'][key];lo,hi=v['paired_bootstrap_95_percent_pp']
        lines.append(f'| {label} | {v["difference_pp"]:+.2f} | [{lo:+.2f}, {hi:+.2f}] |')
    numeric=audit.get('solver_numerical_validation',{})
    if numeric.get('failed_attempt'):
        failure=numeric['failed_attempt'];events=numeric['fp64_fallback_events']
        lines += ['', '## 训练数值与重试','',
            f'求解器首次尝试出现 4 次非有限梯度范数，原生 actor 跳过了 4 步，仅完成 {failure["completed_optimizer_updates"]}/{failure["planned_minibatches"]} 次更新。该次尝试被本地检查拒绝，没有保存最终 checkpoint，也没有开始测试；失败日志完整保留。',
            '随后从原 W1 重启，复用完全相同的 310 条成功调用和 256 条采样轨迹，没有新增训练采样。增加的数值保护仅在每个梯度元素均有限、但 FP32 范数求和溢出时用 FP64 计算裁剪范数；真实非有限梯度仍立即失败。',
            f'有效重试完成 {numeric["completed_updates"]}/78 次更新，所有记录的训练指标有限；FP64 范数回退触发 {len(events)} 次。最终模型与优化器重新加载和独立审计均通过。',
            '原首次异常的根因尚未确定；重试成功本身不能证明数值保护修复了原问题。失败计算成本不能算作免费。']
    started=datetime.datetime.fromisoformat(read(run/'BOOTSTRAP_PROTOCOL.json')['registered_utc'])
    completed=datetime.datetime.fromisoformat(read(run/'STATUS.json')['updated_utc'])
    update_seconds=read(run/'round_01/training_progress.json')['update_elapsed_seconds']
    lines += ['', '## 本机用时','',
        f'本轮 focused 协议登记到审计结束约 {(completed-started).total_seconds()/60:.1f} 分钟，包含候选生成、评估、数值诊断与重试、测试和文件核验；不包含之前失败的通用提示热启动试验。有效求解器更新本身为 {update_seconds/60:.1f} 分钟（78 步），额外还有加载、保存、重载开销。',
        '两条 focused 初始化分支并行，各约 5 分钟训练；在线 harness 更新只有两条有收益的提案、一个优化器步，其训练时间不代表一次完整交替所需时间。']
    lines += ['', '## 结论边界','',
        '本次验证了内部 2B 可以提出有开发收益的 harness，并且代码生成器与解题器都实际更新了参数。生成器 E1 相对冻结 E0 的提案质量尚未做独立对照；这次参数更新不能单独证明持续自我改进或样本效率更高。',
        '本轮没有同预算外部 proposer 的新测试组，因此不能据此声称优于原始 WHALE。开发集择优和成功轨迹 RSFT 也不保证测试集单调提高。',
        '采样与评估最多使用本机 4 卡加内网 10 卡；每个权重训练过程为单张 48GB 改版 RTX 4090，全量语言参数和优化器保持 FP32、计算用 BF16 autocast。并非多机梯度同步实验。','',
        f'审计：`verification.json`，SHA256 `{sha(run / "verification.json")}`。独立复核了模型张量、优化器状态、真实生成 token、所有评分、测试划分和服务退出。','']
    (run/'REPORT.md').write_text('\n'.join(lines))
    print(str(run/'REPORT.md'))

if __name__=='__main__':main()
