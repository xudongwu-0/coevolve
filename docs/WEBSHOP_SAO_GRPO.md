# SAO WebShop Solver 的 GRPO 改进：无蒸馏

本次目标是修复SAO的WebShop强化学习链路，不再尝试自蒸馏。代码位于独立分支，旧ALFWorld/MA-SAO工作树、历史实验及已交付的Aether-QAT checkpoint均不改动。

## 实现边界

新增 [SAO Solver训练入口](../whale_local/sao_bridge/webshop_grpo.py)：

```bash
python -m whale_local.sao_bridge.webshop_grpo train \
  --config configs/webshop_sao_grpo_subset.json \
  --env-root /path/to/pinned/webshop \
  --model /path/to/local/solver \
  --lease /path/to/current-single-gpu-lease.json \
  --output /new/experiment/path
```

此入口是**SAO的Solver阶段可选择的GRPO后端**：固定Engineer/Harness，更新Solver；不声称已经训练Engineer、不改变全局SAO的DIS/critic配置，也不将历史ALFWorld93/97分数转称WebShop分数。若有已选Harness，可通过协议的`solver_harness`提供冻结指导，其hash进入训练回执；未指定时是无Harness的Solver控制实验。

SAO原有基于AReaL的异步训练器不被此后端静默替换。选择本入口是明确的同步、同题8rollout、无critic的GRPO变体，不是声称其等价于原单rollout SAO。当前首先验证Solver学习是否改善，然后才有证据研究Engineer/Solver交替阶段。

## 已核实的参考差异与修复

参考源固定为Apache-2.0：
[`aim-uofa/Self-Distillation-AgentRL@35a44f5`](https://github.com/aim-uofa/Self-Distillation-AgentRL/tree/35a44f5c5647b1785ad23ddf0ffbd196b695bf86)。

1. **环境raw score与训练reward分开。** 原始商品匹配score仍在0–1，仅用于评测展示。参考环境对终局全成功奖励10，否则0；不能为“学习更密集”而未经声明改用partial score或制造成功奖励。
2. **无效动作处罚。** 参考trainer在动作对应的奖励上扣0.1。新profile对每个真实动作使用`10 * terminal_success - 0.1 * invalid_action`，保留invalid动作的token、行为概率和原始环境结果，不免费重试、不删掉失败动作。
3. **同题组身份严格一致。** 8条真实rollout必须具有相同task manifest、policy version，独立seed；不能把8个不同购物任务当成一个GRPO组。
4. **跨动作组内统计。** 参考`compute_grpo_outcome_advantage`默认`compute_mean_std_cross_steps=True`，使用各动作上的episode outcome统计均值与sample std。此前pilot用一次episode平均/population std，并给所有动作相同advantage。新`reference_step`实现跨动作统计和动作级处罚；它会与每episode归一化有不同的长度权重，属于明确声明的参考策略，不是普遍更优的数学定理。
5. **有效生成token聚合。** 参考默认`loss_agg_mode="token-mean"`。新profile按照整个同题组的真实generated token数归一化，不再让一个极短episode与一个很长episode强制拥有相同总梯度权重。
6. **无蒸馏。** SAO入口要求`distill_coefficient=0`，拒绝teacher/SD模式；模型不构造、不评分privileged teacher上下文。GRPO晋级门禁也改为实际reward variance/gradient信号，不要求不存在的teacher信号。
7. **保持可归因。** `legacy_episode`默认保留不变；独立control配置与candidate仅改GRPO profile。模型、goal seed、task IDs、rollout预算、optimizer、KL、history和生成参数保持一致，不用更长预算的候选冒充方法改进。

所用环境是已完成真实reset/search校验的WebShop1k，goal seed233，1,000商品及6,910 synthetic goals。每次reset清除该session的残留done/reward，原始asset与goal/price指纹绑定。源码/env正常运行不等于历史45/74或参考66已经可复现；旧45/74的完整模型/数据/评测manifest仍缺失。

## 子集门禁

新实验用8个训练任务`600,604,...,628`，每题8rollout、8个optimizer更新，共64个训练episode。新固定评测IDs256–287，32题单seed，不复用此前32–63/128–159来选方法。两组都从同一local Qwen3.5-9B初始化，不从之前退化的SD checkpoint继续训练。

先实际比较initial、legacy GRPO与reference-step GRPO；不直接启动完整100步。增加训练任务覆盖只是这轮共同控制条件，不单独算算法贡献。子集结果要报告success、raw score、invalid率、实际action/token数、reward variance和梯度，以及逐题paired win/loss；若置信区间或未触碰数据上的效果不可靠，不扩大训练、不宣称45→66。

与参考完整实验仍有差异：pilot是1k corpus、LoRA、8步、单训练seed及单评测seed；history3/total context4096、KL0.01/entropy未新增与论文完整设置不完全相同。此次只针对经过核实的reward/advantage/loss-aggregation差异，不宣称完整复现参考GRPO成绩。

## 验证

- CPU测试覆盖：reference动作统计、sample std、invalid处罚、partial score不冒充success、失败token保留、group identity、真实optimizer的token-mean权重、旧路径行为不变，以及无需teacher的GRPO comparison gate。
- 原始WebShop回归继续验证动作解析、环境身份、response-only因果对齐、真实tiny-transformer/LoRA梯度和显式GPU lease。
- `webshop_sao_grpo_subset.json`与`webshop_sao_grpo_control.json`除`grpo_profile`外相同。
- 新SAO入口仅允许<=16updates、<=64eval IDs、<=2eval seeds；代码不会根据某个好看的子集分数自动启动完整训练。
