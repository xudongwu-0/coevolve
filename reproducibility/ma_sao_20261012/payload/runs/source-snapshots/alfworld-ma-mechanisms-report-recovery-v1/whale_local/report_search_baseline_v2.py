"""Write the baseline qualification report from audited numbers only."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args();run=args.run.resolve()
v=json.loads((run/'verification.json').read_text());assert v['passed_audit']
status=json.loads((run/'STATUS.json').read_text());scores=v['scores']
labels=['W0H0','W1H0','W1H1','W2H1','W2H2']
lines=['# WHALE 小规模 HotpotQA 基线验证','',
    ('本轮通过预先设定的双轴开发集收益门槛。' if v['qualified_on_development'] else '本轮未通过预先设定的双轴开发集收益门槛；不能据此宣称已复现稳定的共同上升。'),
    '结果来自 64 道未参与本轮训练或 harness 选择的场景验证题，单 seed 42、每阶段每题一次。它是任务筛选用开发集，不是最终独立测试。128 道测试题未评测。','',
    '| 阶段 | 正确题数 | 严格匹配率 |','|---|---:|---:|']
for k in labels:
 s=scores[k];lines.append(f"| {k} | {s['solved']}/{s['total']} | {100*s['solved']/s['total']:.2f}% |")
lines+=['','![真实交替曲线](alternation_curve.png)','','![固定一轴的曲线](fixed_axis_curves.png)','',
    '图中保留每个阶段的原始分数，没有用 best-so-far 包络替换回落。','',
    '| 固定条件下的比较 | 增益（百分点） | 从错变对 / 从对变错 |','|---|---:|---:|']
for key,name in [('weight_at_final_harness','最终 H：W0 → W2'),('harness_at_final_weights','最终 W：H0 → H2'),('joint','起点 → 最终联合状态')]:
 c=v['contrasts'][key];lines.append(f"| {name} | {c['difference_pp']:+.2f} | {c['wins']} / {c['losses']} |")
lines+=['','预设门槛为两个条件增益都至少 5 个百分点。小样本的开发集资格筛选不等于已经证明测试集提升。配对变化的精确检验保存在 verification.json。','',
    '训练与成本：','',
    f"- 有效基线执行耗时：{status['elapsed_seconds']/60:.1f} 分钟，含当前对话生成提案的等待时间；不代表商业模型 API 延迟。",
    '- 两轮分别采集 512 条轨迹；仅原模型自己生成且通过固定 verifier 的成功轨迹用于 RSFT。']
for t in v['training']:lines.append(f"- 第 {t['cycle']} 轮：{t['accepted_trajectories']} 条成功轨迹，{int(t['updates'])} 次 optimizer 更新，训练及保存/重载 {t['elapsed_seconds']/60:.1f} 分钟。")
for key,c in v['cost'].items():lines.append(f"- {key}：{c['trajectories']} 条轨迹、{c['policy_calls']} 次模型调用、{c['assistant_tokens']} 个生成 token。")
lines+=['','本机 3 张和内网 10 张空闲卡承担并行采样/评测，另一张本机卡训练；没有使用本机被其他任务占用的卡。',
    'harness 搜索由当前对话经本地 FastAPI 队列提出完整 Python 候选，使用上游 MH/Pareto 选择；它不是无人值守的外部模型 API。外部 proposer 的推理 FLOPs 未计量。','',
    '实现核验包含原始服务回执、native EM 重算、SFT 输入/响应 token 与 mask、FP32 参数/Adam 状态、padding 修复、参数变化及 checkpoint 重载。','',
    '与论文的差异：','']
lines += ['- '+s for s in v['limits']]
lines+=['- 严格 EM 存在别名或长参考答案导致的语义正确但记错案例；本轮没有事后修改 verifier。',
    '- 先前一次内网检索被代理转发产生 502 的尝试已作废；它的 560 条任务轨迹和 12 次更新保留在 INFRASTRUCTURE_RESTART.json 所指目录，不计入有效提升。',
    '- 发现第二轮开发集回落后，另注册了仅将该轮 LR 降到论文 1e-7 的检查，复用同一批真实 on-policy 轨迹。它是额外校准成本，不能算作免费或相同预算的改进。','',
    '原论文：[WHALE](https://arxiv.org/html/2609.00196v1)。这里验证小规模算法行为，不声称复现论文全量榜单数值。','']
(run/'REPORT.md').write_text('\n'.join(lines))
print(run/'REPORT.md')
