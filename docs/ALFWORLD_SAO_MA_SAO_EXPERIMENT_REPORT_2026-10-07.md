# ALFWorld：SAO、MA-SAO 与本地 WHALE 适配实验报告

**证据版本：2026-10-07，已纳入H3原始200、新MA总240轮早期结果，以及WHALE时间截断运行的原始200。** WHALE主评在23:31 UTC前完成并通过独立核验，为163/200=81.5%；实际640个Solver episode、216个MH提案，**没有完成要求的640/384同预算复现**。用户硬截止为 **2026-10-07 23:37:12 UTC**；MA早期结果已交付，原状态续训在WHALE释放资源后接续，不能把早期下降写成最终平台。后续进展见[项目状态](./PROJECT_STATE.md)。

本文中的“SAO / Harness-R1 式交替训练”指本项目在 ALFWorld 上的具体实现与配置，不代表对原方法所有实现、超参数和训练预算的穷尽检验。

## 结论摘要

目前最合适的总括是：

> 在已经测试的配置和追加预算内，固定 Harness 继续训练 Solver、以及固定强 Solver 再训练一阶段 Engineer，都没有可靠突破历史 SAO 表现。前者曾恢复、随后再次退化；后者改善了候选格式合法性，但没有可靠改善任务效果。因此，可以说这两种直接续训方案尚未解决当前瓶颈，不能说已经证明两个模型或 SAO 方法无法继续学习。

| 研究问题 | 已有结果 | 当前结论 |
| --- | --- | --- |
| 相同 Engineer rollout 数，SAO 是否比 GRPO 好？ | W1 部署：90.5% 对 88%；W0 迁移：74% 对 83.5% | W1 上 SAO 点估计略高但不确定；不支持普遍优于 GRPO，也没有做统计等效检验 |
| 固定 Harness，延长 Solver 训练能否突破？ | 带 h2：63.5% → 71% → 79.5% → 38% → 49% | 曾回升，但没有恢复历史 92%，随后再次明显退化 |
| 固定 W1，再训练 Engineer 能否突破？ | 合法候选 2/8 → 8/8；旧 h2 为 178/200；**真正 H3 新候选为 173/200 = 86.5%** | 已完成新候选的完整 200 题，不是拿旧策略回退结果代替 H3 |
| 原 MA 联合框架是否自然优于 SAO？ | 原始 H0/W0 分支：85% → 87.5% → 82.5% | 不是；已有多个负面配置 |
| 调优后的 MA 长期高度如何？ | 旧混合调度最高 97%，最终 95%，最后四点 93%/94.5%/94.5%/95% | 满足预声明的观测平台，但不是稳定 97%；相对历史 92% 尚未通过严格优势门槛 |
| 97% 是否完全是无效提示或评测漏洞？ | 固定 Solver 下，自动策略带来 9 改善/0 退化；交叉复测也有正向策略收益 | 支持保存策略具有实际效用；不足以证明 Engineer 整体分布改善或独立泛化 |
| 从 97% 改为每轮双方更新，能否更高、更稳定？ | 总 144/192/240 轮：97% → 93% → 92%；空策略 92.5% → 90.5% → 89% | 早期没有提升，反而下降；尚未达到预声明的平台判断长度 |
| WHALE 同预算表现如何？ | 时间截断运行主200为**163/200=81.5%**；实际640个Solver episode、216个MH提案 | **未完成640/384同预算复现**；对历史SAO92%差-10.5pp，CI[-16.5,-4.5]，但不是完整同预算算法比较 |

## 统一术语、评测口径与证据边界

- **H / Engineer**：生成 Harness 的模型；**W / Solver**：执行环境任务的模型。
- **h2、P48、P144 等 Harness**：模型生成后保存的具体策略产物，不等于 Engineer 模型本身。评测一个好策略，不等于证明 Engineer 的全部生成分布更好。
- **成功率**：除特别说明外，指固定开发 200 题上的 `mean@1`，成功依据环境奖励/胜利信号，不以程序正常退出或输出合法 JSON 代替。
- **pp** 表示百分点；差值与置信区间均用百分点，不是相对百分比。
- **历史 92%** 是历史 W1+h2 结果。本次旧 h2 同期复测为 89%，两者分别保留，不能用较低复测值覆盖历史结果或降低 MA 的目标。
- **区间**通常为按题目配对的 10,000 次 bootstrap 95% 区间；它不包含训练种子不确定性，不是校正了多次调优/高点选择的确认性区间。
- **开发集已经反复使用**。MA 的 selection48 全部包含在 report200；本次 Engineer 续训的 selection24 也包含在 report200。`strict197` 只排除三个训练目标文本重合项，不能消除选择偏差。
- **训练消费轨迹、实际生成轨迹、环境 episode 和计算量分开计费**。普通 SAO 的一条 Engineer 轨迹可调用十个环境任务评分；MA 的一对角色轨迹可来自一个环境 episode。“角色轨迹总数一样”不等于完整样本成本一样。
- 主线长期曲线和本次补充续训均不是多训练种子验证。已有的历史训练重复、两种推理种子检查，也不能充当当前新 MA 分支的训练种子稳健性证据。

完整历史配置与变更记录见[研究协议](./MA_SAO.md)和[实验日志](./EXPERIMENT_LOG.md)。

## 研究对象、算法与复现方法

### M1. 本报告分别估计什么？

| 估计对象 | 对应证据 | 不能替代它的量 |
| --- | --- | --- |
| 固定采样预算的任务效果 | 固定预算、固定选择规则后的系统成功率 | 每种方法事后最高分；相同训练步数但不同 batch |
| 延长训练的变化 | 预先列出的全部中间点及最终点 | 只报告恢复段或最高点 |
| 有限预算内的稳定高度 | 满足预声明平台条件的连续端点 | 一次 97%；训练 loss 下降；某阶段保存成功 |
| 保存 Harness 的贡献 | 固定 Solver，替换/删除具体策略的配对评测 | Engineer 模型更新了多少步 |
| Engineer 生成分布的变化 | 同 prompt/seed/候选数的槽均分及合法率 | 从不同规模搜索中各挑一个最好策略 |
| 算法因果优势 | 进一步控制初始化、数据、选择及计算差异的比较 | 本报告中所有异构历史分数的排序 |

本文将“有模型参数更新”“有保存策略收益”“有系统分数改善”“有统计支持的稳健优势”分开使用。四者不是同义词。

### M2. 模型、初始化与检查点谱系

两个角色使用本地 9B 级 Qwen3.5 系列模型，配置架构名为 `Qwen3_5ForConditionalGeneration`；文本主干 hidden size 4096、32 层、16 个 attention heads、4 个 KV heads、intermediate size 12288、词表 248320。本实验是文本环境交互，不是视觉任务。

| 名称 | 实际来源 | 解释 |
| --- | --- | --- |
| H0 / E0 | 官方发布的 `harness-r1` | **已经经过官方 RL 的 Engineer**，不是从裸基座或仅 SFT 模型开始 |
| W0 | 官方发布的 `target-agent-sft` | Agent-SFT Solver |
| 官方 target-specific Engineer | 与 Agent-SFT 配套的另一个官方 Engineer | 只在固定候选比较中使用；不能和 H0 混为同一初始化 |
| H1 / E15 | 本地 H0 续训阶段选中的 step15 | 选中模型有 11 次 Actor 更新；整阶段实际完成 27 次 Actor 更新 |
| W1 / W79 | 单角色 fast-v4 的 step79 | 80 native step、5 次 Critic-only warmup、75 次 Actor 更新 |
| H2 / h2 | W1 条件下的后续 Engineer 模型 / 选中的具体策略 | H2 与 h2 是两个不同对象 |
| W2 | 后续被选择门槛拒绝的实际 Solver 检查点 | 仍单独评测，不能用保留的 W1 冒充 W2 |
| H3 | 从真实 H2-step31 再续训 32 步 | W1 完全冻结；新策略另测完整 200 题 |
| MA64 | 原 H0/W0 联合分支第 64 轮 | 后续部分实验使用其 Solver 或 Critic；继承训练不是免费初始化 |
| 平衡 LR 的 MA48 | H0/W1、MA64 Critics 起点的 48 轮候选 | 自动策略 P48，93.5%；不是纯 H0/W0 从零训练 |
| MA144 / P144 | 旧混合调度保存的第 144 轮组合 | 原评测 97.0%；新每轮联合分支的真实父检查点 |

主要谱系为：

```text
官方 H0 ----> 本地 H1/E15 ----> 在 W1 条件下训练 H2 ----> 固定 W1 继续 H3
官方 W0 ----> 单角色 W1/W79 --> W2 被拒绝 ---------> 固定 h2 继续 W2

官方 H0/W0 联合 --> MA64（包括后来复用的 Critics）
H0 + W1 + MA64 Critics --> 平衡 LR MA48/P48
                           └─ 旧 2:3 更新调度 --> MA144/P144 --> 旧曲线最终 MA432
                                                  └─ 原生状态恢复、每轮联合 --> MA192/MA240 ...
```

“总轮数”是谱系位置，不等于当前 Adam 的 `step`。从 MA48 进入旧长期续训时有一次明确 fresh Adam；此后保存 FP32 model-only DCP、逐 rank Adam 与 ValueNorm。MA144 的原生 H/W Actor 步数为 64/96，新 96 轮后为 160/192。仅加载 BF16 HF 导出权重不能代替这种恢复。MA 的 Adam 指纹是每个 tensor 最多 64 个元素的采样核验，不应描述为所有 Adam 元素逐位哈希；WHALE 原生恢复验证另使用完整本地 tensor 指纹。

模型绑定见[原 H0/W0 身份](../runs/alfworld_joint_h0w0_8h_v1/INITIAL_MODEL_IDENTITIES.json)、[W1 训练配置](../examples/alfworld_solver_fast_v4_s42.json)、[新 MA 初始化](../runs/alfworld_ma_simultaneous_from97_v1/block-01/INITIAL_STATE.json)。

### M3. 数据、选择集与报告集

| 数据用途 | 规模/构造 | 与当前主 200 题的关系 |
| --- | --- | --- |
| 官方本地 catalogue | 500 个 game instance | 多次用于开发与诊断；不是当前封存测试 |
| 早期缓存证据、MA 训练 population | 其中 legacy task IDs 0–239 | 与主 200 无 instance 重合，但有 3 个目标文本家族相关项 |
| 单角色强 W1 的 train-only pool | 2296 个训练 game；W1 消费前 640 个训练实例 | 独立构建，instance/goal 与官方 catalogue 隔离 |
| 早期 Solver 选择/报告 | 59 题 / 91 题，`mean@2` | 是早期不同协议，不能与 core200 直接拼成曲线 |
| Engineer/交替实验的原选择/报告 | 24 题 / 70 题 | 两者原本互斥；后续已反复使用 |
| 当前 core200 | 原 24 题的 sample0 + 新增 176 题 | 不是 25+175；不取原两次采样中较好的一次 |
| strict197 | 从 core200 去掉 291、414、467 | 仅去掉目标家族相关项，不能消除模型/策略选择偏差 |
| MA selection48 | 48 题、通常 `mean@1` | 完全包含于 core200；选策略后才运行主评，但主评仍非独立测试 |
| H3 selection24 | 24 题、`mean@2` | 同样包含于 core200 |
| 当前 WHALE 的 MH 开发任务 | 从 train-only catalogue 固定挑 10 个不同、goal-disjoint 任务 | 排除 WHALE 训练目标及 core200 目标；不使用 core200 选择策略 |

WHALE 当前训练取的是**原 MA 记录的前 640 条**，包含 240 个不同 legacy instance 的重复；不是历史强 W1 的 640 个不同 train-only 实例。相同 episode 数并不代表相同 population、课程或反馈难度。MA 新分支也保留原输入次序：每三轮使用两个 source batch，第三轮按 cycle 奇偶重采其中一个；是相同任务输入的新轨迹，不是重放旧行为 token。每 96 轮有 512 个 source 记录、768 个新环境 episode，不能写成 768 个不同任务。

完整 ID 与 goal-group 规则见[core200 协议](../runs/alfworld_core_200_oneshot_v2/protocol.json)、[训练集构建](../whale_local/sao_bridge/alfworld_train_data.py)、[WHALE 冻结计划](../runs/alfworld_whale_sao_matched_v10/FROZEN.json)、[MA 输入排程](../whale_local/sao_bridge/joint_anchor.py)。

### M4. 环境、策略接口与错误语义

- 使用原生 AgentBench ALFWorld worker；任务奖励来自环境，不由 Engineer 修改。成功为二元 0/1。
- Harness 是带 `benchmark=alfworld`、`actions` 的 JSON patch，代码 hook 在原生运行时执行；可提供初始化/逐步提示、维护状态、检查或干预动作。策略产物本身不是神经网络权重。
- Solver 通过工具调用执行动作；合法 JSON、合法 patch、进程正常完成都不等于环境成功。一个语法合法 hook 仍可能错误地阻止恢复动作。
- core200 主评为 `mean@1`、temperature 1.0、基础 seed 6000045，并结合稳定题目 ID 得到 rollout seed；使用全部相同的 task/sample/seed 身份进行配对。
- 当前原生 Solver 的 **4096 是整条 episode 的累计生成 token 上限**；模型上下文最大 32768；环境另有 50 步限制。三者不能混为一谈。早期官方 target-runner 协议单独标注，不并入该原生预算。
- 已知模型失败（例如 token limit、环境 invalid action、无法往返渲染的工具参数）保留原轨迹并按明确合同计零。未知状态、HTTP/传输错误、worker 故障仍然抛错，不能转成成功或普通零奖励来凑齐题数。
- WHALE v7 曾把 `AGENT_INVALID_ACTION` 错判为基础设施错误；修复后仅接受该状态的零奖励结果，仍拒绝“invalid action 却 reward=1”的不一致记录。

接口及状态处理见[ALFWorld 适配器](../whale_local/sao_bridge/alfworld_domain.py)、[原生多策略服务](../whale_local/sao_bridge/alfworld_env_service.py)、[WHALE episode 校验](../whale_local/sao_bridge/alfworld_whale_baseline.py)。

### M5. 本地 SAO 的实际优化目标，而非仅用方法名称代替实现

本项目延续“SAO”名称，但底座是明确声明为非官方实现的 SAO/AReaL 集成：SAO revision `ac2728149041b5f15ceb75413f467ce0659179cf`，AReaL revision `3cf0dfbd2b0fbeabd6977184980e189d1567747a`，并有本地原生训练补丁。论文结论应限定为**该本地实现和配置**，不能宣称穷尽复现 SAO 或 Harness-R1 的所有算法细节。

对普通 Engineer 续训，合法策略由固定 Solver 在其 packet 对应的 10 个训练任务上评分，使用官方 Harness-R1 的平均奖励增量合同；不是只看代码合法性。Solver 续训使用整条真实任务轨迹的终局奖励。Actor 与 Critic 独立，Critic-only warmup 期间仍生成并消费样本，但不更新 Actor。

令行动 token 集为 \(\mathcal A\)，行为策略为 \(\mu\)，重新计算概率的 proximal 策略为 \(\pi_p\)，待更新策略为 \(\pi_\theta\)：

\[
c_t=\frac{\pi_p(a_t\mid s_t)}{\mu(a_t\mid s_t)},\quad
r_t=\frac{\pi_\theta(a_t\mid s_t)}{\pi_p(a_t\mid s_t)},\quad
m_t=\mathbf1[0.3\le c_t\le5].
\]

原生 decoupled PPO 对行动 token 使用上述行为校正与双侧剔除；超出范围是 mask，不是把比率夹到边界后继续贡献梯度。记 batch 标准化后的优势为 \(\bar A_t\)，普通单角色的核心 Actor 目标为：

\[
L_{\rm actor}=-\frac1{|\mathcal A|}
\sum_{t\in\mathcal A}m_t c_t
\min\!\left(r_t\bar A_t,\,
\operatorname{clip}(r_t,1-\epsilon_l,1+\epsilon_h)\bar A_t\right).
\]

分母保留过滤前的行动 token 数。普通 SAO 模板 \((\epsilon_l,\epsilon_h)=(0.3,5.0)\)，新 MA 原生路径覆盖为 \((0.2,0.2)\)；这不是同一个 clipping 设置。奖励可含固定参考 KL 惩罚，普通历史主配置为 0，新 MA 为 0.02。

GAE 沿行动 token 回传，跳过环境 observation token。当前实际配置的原生默认是 **\(\gamma=1,\lambda=1\)**，完整 episode 的终端 bootstrap 明确置零。仓库虽有 length-adaptive lambda 的独立辅助函数，本次实际调用的原生优势路径没有调用它；不能在论文方法节把辅助函数的存在写成已启用的算法特性。实现核验依据是实际加载的原生 Actor 和最终配置，而不是仅阅读上游 README。

每轮先进行 Actor 更新，再以相同 `adv_batch` 对 Critic 做 \(K\) 次更新；不是每次 Critic 更新后重新采样。普通 H/W 本阶段 \(K=1\)，当前 MA \(K=2\)。普通 Engineer 的无效策略按官方 `invalid_patch_treated_as_no_patch` 合同得到零增量奖励，通常不额外启动目标环境评测；这与 MA 的无效策略仍可获得正团队奖励不同。基础设施异常由本地 reward service 检出并显式失败，不接受上游的错误回退作为正常训练奖励。

```text
固定当前队友与训练数据
for native step:
    采样完整行为轨迹，保存真实 token / log-prob / 版本
    用固定队友/原生环境计算奖励
    重新计算 proximal log-prob，计算终局修正后的优势/returns
    warmup 之外更新 Actor；更新 Critic K 次
    发布新版本，按预声明位置保存
固定候选生成/选择规则，再评测保存策略 + Solver
```

依据：[模板](../third_party/sao/configs/phase2_hard_sao.yaml)、[实际原生优势路径](../third_party/sao/vendor/AReaL/areal/trainer/ppo/actor.py)、[实际损失函数](../third_party/sao/vendor/AReaL/areal/utils/functional/functional.py)、[原生更新顺序](../third_party/sao/vendor/AReaL/areal/trainer/rl_trainer.py)、[终局与队友权重包装](../whale_local/sao_bridge/terminal_actor.py)。

### M6. GRPO 对照到底匹配了什么？

使用同一官方 RL H0、同一 24 个 prompt 各出现 8 次、同一 W0 奖励 Solver，共 192 条 Engineer 轨迹。GRPO 是 6 个 32 条的 rollout bundle，总 24 个优化小步；SAO 为 32×6，其中 30 条为 Critic-only warmup，162 条进入 Actor 更新。两者 Actor 有效曝光量、Critic、优化步数、token/FLOPs 不是严格相同。

GRPO 复用固定 Relax/Harness-R1 的 reward-group 标准化及 PPO primitives，而不是换名使用 SAO loss。每组 8 个 completion，按组奖励构造标准化优势；clip low/high 为 0.2/0.28，token-level TIS 上界 2，KL 系数 0。**先对每个 response 的 token loss 取均值，再平均 response**，不等同于 WHALE RSFT 的全 response-token 均值。AdamW LR `1e-6`、weight decay 0.1、betas 0.9/0.98、grad clip 1；采样 temperature 0.7、top-p 0.95、最大生成 4096/context 32768。

这是固定原生目标的 FSDP/vLLM 后端适配，不宣称与完整官方 Relax 系统逐项一致。SAO 选中了历史 step15，而 GRPO 使用预声明最终 192 条端点；两者的选择过程也不相同。

依据：[匹配输入与奖励校验](../whale_local/sao_bridge/alfworld_matched_grpo.py)、[GRPO 参数](../configs/alfworld_engineer_matched_grpo.json)、[目标函数](../whale_local/harness_r1_searchqa/grpo_objective.py)、[真实训练完成记录](../runs/alfworld_matched_grpo_192_v4/TRAINING_COMPLETE.json)。

### M7. MA-SAO：一个环境 episode，两个真实动作轨迹

每个配对样本有 `sample_id/version/slot`。Engineer 先基于该行的训练 evidence prompt 生成代码策略，Solver 再在指定任务中使用它完成多轮交互。该 episode 产生：

1. Engineer 的生成 token、行为 log-prob、版本；
2. Solver 全部生成调用拼接后的行动 token、行为 log-prob、版本；
3. 同一个终局环境奖励 \(R\)，以及两侧可追溯 receipt。

**不是把同一串 token 假装成两种角色的行为；也不是两个网络共享参数。** Engineer 文本证据来自冻结训练 packet；当前 Solver 的实际行为通过当轮环境结果和训练奖励形成耦合，不能误写为每轮都重新收集并重写所有 Engineer 输入 evidence。

当前 `team` 模式下，两方都拿实际 \(R\)。无效 Engineer 输出按既有合同回退为空策略，Engineer 仍可能收到 Solver 成功的团队奖励；`validity_gated` 是另一个已经失败的消融，不是当前主配置。没有实施反事实差分奖励。

两个 Critic 也独立：Engineer Critic 额外看到 Solver 任务的预动作信息及版本/lag；Solver Critic 额外看到 Engineer 预动作 evidence、策略有效状态和 Solver 任务信息。Actor 不加入这些 privileged prefix。额外 Critic 上下文最多 1024 token。ValueNorm 复用固定 MAPPO 参考实现，在同一角色全部 DP rank 的 active returns 上更新，并保存/恢复统计；不是各 rank 独立统计，更不是共享一个中央网络。

队友的整条行动轨迹校正为：

\[
w_{-i,b}=\exp\left[
\operatorname{clip}\left(
\sum_{t\in\mathcal A_{-i,b}}(\log\pi_{p,-i,t}-\log\mu_{-i,t}),
-\log2,\log2\right)\right]\in[0.5,2].
\]

原生优势计算与标准化后，当前角色的优势再乘 \(w_{-i,b}\)。自身的 token-level 校正仍保留；两者不是重复统计同一个比率。该有界队友校正是实验性、带偏的稳定化手段，不应宣传成标准 MAPPO 的完整理论实现。相同权重下很小的逐 token 概率偏差也会在整轨迹求和中放大，后文单列了复现与负面消融。

```text
for 配对 round（batch=8，最大行为版本 lag=1）:
    H 生成8条策略轨迹
    W 在8个真实任务中使用对应策略，产生8条完整 Solver 轨迹
    双方校验 ID/slot/version、真实 token、行为概率与相同终局结果
    各自计算 proximal 概率；交换同槽队友概率摘要
    各自 Critic 用额外协作上下文估值，Actor 计算对应优势
    本轮被调度激活的 Actor 先更新一次
    各自 Critic 随后更新两次；ValueNorm 用相应 active returns 更新统计
    发布版本，保存原生状态与审计记录
```

旧长期分支每三轮是 `joint/joint/Solver-only`，H:W 更新比 2:3；第三轮仍生成 H 提案，但不训练 H，且该轮 W 队友权重为 1。新分支每轮两方都更新、所有轮都用 bounded 校正。因此新旧比较同时改变更新频率及第三轮校正规则，**不是纯粹的频率单因素消融，也不是“首次开始联合训练”**。

| 当前新 MA 参数 | Engineer | Solver |
| --- | ---: | ---: |
| Actor LR / Critic LR | `5e-7` / `2.5e-6` | `5e-7` / `2.5e-6` |
| 每轮 batch / Critic 更新次数 | 8 / 2 | 8 / 2 |
| temperature / top-p | 0.7 / 1.0 | 1.0 / 1.0 |
| 新块 warmup / 最大版本 lag | 0 / 1 | 0 / 1 |
| 生成 token / Actor context | 4096 / 32768 | 4096 / 32768 |
| Actor clip / 固定 KL 系数 | ±0.2 / 0.02 | ±0.2 / 0.02 |
| 固定 reference | 原 H0 | 原 W1 |
| Learner / rollout GPU | 4 / 4，node3 | 4 / 4，node4 |
| master / 计算 / reference 存储 | FP32 / BF16 / BF16 | FP32 / BF16 / BF16 |
| 数据 seed / 保存间隔 | 1000043 / 48 轮 | 1000043 / 48 轮 |

两个原生 Adam 使用 constant LR、betas 0.9/0.999、epsilon `1e-8`、weight decay 0、grad clip 1。梯度 checkpointing 开启；每个 microbatch 一条独立完整轨迹，避免 Qwen3.5 recurrent-state 跨打包样本泄漏。

依据：[联合 workflow](../whale_local/sao_bridge/joint_workflow.py)、[队友比率](../whale_local/sao_bridge/joint_updates.py)、[独立 Critic/ValueNorm](../whale_local/sao_bridge/joint_critic.py)、[实际 H 配置](../runs/alfworld_ma_simultaneous_from97_v1/block-01/editor_spec.json)、[实际 W 配置](../runs/alfworld_ma_simultaneous_from97_v1/block-01/solver_spec.json)。

### M8. WHALE 本地适配：在线 RSFT 与 Meta-Harness

上游 WHALE revision 为 `fbe125eb7abea7f760c99ab9acc1a6261e708fc6`。官方域包含 SearchQA、数学和棋类，本项目没有可直接引用的官方 ALFWorld 成绩。当前适配以固定本地 RL H0 为 proposer、原 W0 为 Solver、原自动选择的 H0 策略为初始 Harness；**没有调用本研究助手、外部 FastAPI 模型或隐藏的人工作者来写候选**。

固定两周期：`320 Solver episode → 192 MH proposals → 320 Solver episode → 192 MH proposals`。原计划每个 RSFT 段为10×32 episode；实测模型加载、状态保存的吞吐不足后，在完整160题/43个Adam小步边界冻结v12：第一段为`32/32/32/32/32/160`，第二段为`160/160`。仍在320和640题边界交替，前160题不重采，FP32/Adam不重置。**该修订减少段内rollout权重刷新次数，是学习超参数变化，不是纯运行时等价加速**；决定发生在任何MH或最终200题分数出现前。每个 MH 段仍为8次适应性迭代，每次24提案。迭代内提案共用该迭代开始的incumbent和反馈；完整评分、选择后才进入下一轮，不是一次独立best-of-192。

RSFT 只学习 reward>0.5 的成功 episode 中**真实采样的生成调用**。对每个至多 64 调用的优化小批：

\[
L_{\rm RSFT}=-\frac{\sum_{j\in B}\sum_{t\in\mathrm{response}(j)}
\log p_\theta(a_{jt}\mid x_j,a_{j,<t})}
{\sum_{j\in B}|\mathrm{response}(j)|}.
\]

原始路径使用上游 `DataParallelPPOActor.update_sft_policy`，单epoch、microbatch1、AdamW LR `1e-6`、weight decay0.01、grad clip1；没有Critic或PPO advantage。原生FP32参数与Adam跨batch保存/恢复；BF16用于计算与推理导出。原v10单调用在8个FSDP rank复制计算，后续v11/v12改为下面明确验证的样本分片目标；最终成本必须保留这两种实际执行模式。梯度小步数量随成功调用数量变化，不能与SAO的native step数等同。

上述为已执行v10的原生路径。实测首批552成功调用的请求至COMMITTED约1626秒，
其中原生`elapsed_seconds=1250.31`还没包含最后串行哈希开销；不能拿该字段
当完整批墙钟。第二批也完整保存，合计64个episode。为保留原预算而非删减
学习，v11将同一个64调用全局小批按rank分片，FSDP平均梯度前使用
\(T_{\rm global}/8\)作每rank分母；尾部只补零loss-mask的占位调用。
CPU两rank对1/3/5/7条不等长调用验证参数、Adam、目标和token计数一致。
这是同一全局token均值目标的本地并行实现，不再称逐字调用未修改的上游函数，
也不保证浮点训练轨迹逐位一致。
同时并行完整rank-state哈希，并把独立“加载后立即关闭”验证改为实际下一次
rollout/MH/主评成功后才标记reload verified，未使用之前保持pending。
v11实际32K/4096、warm Adam、FP32恢复和vLLM重载验证通过，耗时1210.03秒。
但原先预声明的**跨多步loss/grad norm数值门槛失败**，失败收据保留，未改大旧阈值。
随后另行预声明固定权重、零optimizer update的全梯度诊断：原复制路径重复的
gradient relative RMS为0.0027486333，分片对参考为0.0032532655；
门槛`min(0.01,max(0.001,2×同路径重复误差))=0.0054972666`，通过，
且三次loss逐值相同。该78.19秒诊断支持同权重目标/梯度范围，不证明多步轨迹等价。
实际续训保持旧64题与Adam18，后续三个32题批保存到160题/Adam43。
三个新native批耗时686.26、474.26、447.63秒，包含末尾完整文件hash；
第一新批明确恢复Adam18→27，未fresh reset。

每个合法 MH 提案都在同一组 10 个 goal-disjoint 开发任务上评分。保留官方 success 优先、turn 数辅助的 Pareto 选择及 incumbent tie 规则；无效生成保留在 384 提案账本中。proposer 不更新权重，所以 384 次搜索提案不能称为 384 次 Engineer 梯度训练。

v10 只改变评分运行方式：迭代内复用 Solver 服务，并用独立命名的原生 worker 至多并行评分 8 个策略。候选数、seed、任务、反馈边界与 Pareto 函数不变。56 个 CPU 测试及 48 个 subtest 通过；还用两个真实官方 worker、同两题、4 个显式无模型调用的合成 episode 验证了新路径，不能将该 probe 计作模型成功率。

v12进一步让H0与Solver各8个replica在同一8张GPU上分端口常驻，单模型每GPU
memory fraction0.4，跨整个MH段复用；并发环境episode由32改为64。
实际执行了两模型共128个并发请求，每个32767 prompt token+1生成token，
首token逐请求区分以避免相同前缀缓存掩盖负载，全部通过，293.05秒。
这是128次合成模型调用、0环境episode、0optimizer update，不能计入640/384。
六个native计算/恢复函数AST和分片helper SHA与v11相同，故复用已披露范围的
原生证明；没有伪造一次v12的新原生训练smoke。最新48测试/38subtest通过，
与此前测试有重合，不能相加成独立总数。

原生 RSFT 已在 v9 实际通过 32768 context / 4096 response、warm Adam、完整 FP32/Adam 恢复和 vLLM 重载验证，耗时 1822.47 秒。v10 **没有伪称又执行一次 GPU smoke**：仅在全部非评分代码 AST、模型、native 配置、依赖与 kernel 身份一致后，显式复用该原生组件证明；变化的评分组件单独做真实 worker 验证。见[组件验证收据](../runs/alfworld_whale_sao_matched_v10/COMPONENT_VALIDATION.json)。

提前停止只有两个明确类别：

1. 一完整周期后，选中策略开发准确率比初始 W0+h0 低至少 20 pp：预声明性能停止，不用 core200 作停止判断。
2. RSFT新批准入仍截止22:52:12 UTC；v12另将MH新迭代准入冻结为23:12:12 UTC，保留25分钟主评窗口。用户硬截止仍为23:37:12 UTC，没有延长。这是时间限制，若未满640/384必须报告 **partial budget**，不能改称性能停止或完整同预算结果。

本轮匹配参照是历史保留的 **H1(192)+W1(640)+H2(192)** 阶段预算，不包含其后被拒绝的 W2 额外 640 条；后者属于历史真实研究成本，不能消失。WHALE 两个 W 段各 320、合计 640，不是每段 640。匹配的是角色相关采样维度，**不是整个 SAO 两个完整 W 阶段的实际研究总预算**，也不匹配训练 population、token/FLOPs、梯度步数、评分任务或选择成本。

依据：[v12批量与服务修订](../runs/alfworld_whale_sao_matched_v12/DESIGN.json)、[本地适配](../whale_local/sao_bridge/alfworld_whale_baseline.py)、[上游RSFT](../WHALE/domains/search_qa/verl/workers/actor/dp_actor.py)、[v11原生GPU验证](../runs/alfworld_whale_sao_matched_v11/gpu-smoke/SMOKE_COMPLETE.json)、[失败的旧多步门槛](../runs/alfworld_whale_sao_matched_v11/EXECUTION_FAILED.json)、[独立固定权重验证修订](../runs/alfworld_whale_sao_matched_v11/VALIDATION_AMENDMENT.json)、[双模型真实服务门控](../runs/alfworld_whale_sao_matched_v12/DUAL_SERVING_VALIDATION.json)、[新评分的实际CPU集成验证](../runs/alfworld_whale_sao_matched_v10/SCORING_CPU_INTEGRATION.json)。

## 实验 1：相同 Engineer rollout 数下，SAO 是否与 GRPO 相当或更好？

### 研究问题

在相同 Engineer 训练轨迹消费量下，SAO 是否具有比 GRPO 更好的任务效果？这种关系是否能从强 Solver W1 迁移到 W0？

### 实验设计

- 两者使用同一 H0 起点，训练总消费均为 **192 条 Engineer 轨迹**。
- SAO 沿用已选出的 step15 检查点；GRPO 使用预先规定的最终 192 条消费检查点。训练总预算相同，不代表检查点选择过程相同。
- GRPO 在 W1 上用 selection24、`mean@2` 从八个候选中选定一个策略；该策略原样部署到 W0，后者是迁移评测。
- 主评测为相同开发 200 题；不将历史消耗了额外 Engineer 阶段的 h2=92% 当作这个 192 条比较的同预算 SAO 端点。

**结果表：**

| 部署 Solver | SAO | GRPO | SAO − GRPO | 配对 95% 区间 |
| --- | ---: | ---: | ---: | --- |
| W1 | 90.5% | 88.0% | +2.5 pp | [−1.5, +6.5] pp |
| W0，迁移部署 | 74.0% | 83.5% | −9.5 pp | [−14, −5] pp |

区间由原报告的 GRPO−SAO 对比换向；没有把换向前后的改善/退化数混用。

### 实验结论

1. **在 W1 上，SAO 点估计略高，但差异不明确。** “没有显著差异”不等于证明效果相同。
2. **W0 迁移结果反而支持 GRPO 更好。** 因此不能概括为 SAO 普遍优于 GRPO。
3. 这回答的是一个特定的 192 条 Engineer 轨迹对照，不是完整 H/W 系统的同环境成本、同 token 或同算力比较。
4. 现有证据应保留，不因后续研究目标而重复选择有利结果。

证据：[完整匹配评测](../runs/alfworld_matched_grpo_200_v3/COMPLETE.json)、[评测实现](../whale_local/sao_bridge/alfworld_matched_grpo_eval.py)。

## 实验 2：固定 h2，延长已经退化的 Solver W2，能否恢复并超过 92%？

### 研究问题

SAO 后期表现下降后，继续训练 Solver，是始终下降、先下降后恢复，还是最终可以超过历史 92%？

### 实验设计

- 从实际退化的 W2 继续，而不是重新初始化一个 Solver。
- 固定 h2，追加最多 **80 个 Solver 更新步、640 条消费轨迹**。
- 在额外 0/20/40/60/80 步评估，同时报告带 h2 和空策略。
- 该边界有 **fresh Adam**，并使用已修复轨迹打包问题的后端。因此它不是从历史 92% 系统开始、保留所有状态的完整 H/W 交替延长实验。

**结果表：**

| 额外 Solver 步数 | 额外消费轨迹 | 带固定 h2 | 空策略 |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 63.5% | 49.0% |
| 20 | 160 | 71.0% | 74.5% |
| 40 | 320 | 79.5% | 83.0% |
| 60 | 480 | 38.0% | 46.0% |
| 80 | 640 | 49.0% | 44.5% |

历史 W1+h2 的 92% 仅作为历史参考，不是表中额外 0 步的起点。

### 实验结论

- **确实发生过恢复**：带 h2 从 63.5% 回升到 79.5%，所以“继续训练完全不会涨”不成立。
- **恢复没有持续**：随后降到 38%，最终 49%；空策略也出现恢复后再次下降。
- 当前追加预算内没有恢复、更没有超过历史 92%。这支持“简单继续更新 Solver 没有解决该配置的后期瓶颈”。
- 不能据此证明 92% 是 SAO 的理论上限，也不能把这次固定 h2 实验称为所有交替延长策略的穷尽检验。

证据：[续训协议](../runs/alfworld_w2_recovery_v2/protocol.json)、[完整曲线和配对统计](../runs/alfworld_w2_recovery_v2/COMPLETE.json)、[历史 92% 报告](../runs/alfworld_core_200_oneshot_v2/merged/w1_h2.json)。

## 实验 3：固定强 Solver W1，再训练一阶段 Engineer，能否继续提升？

### 研究问题

实验 2 只验证了 Solver 方向。若冻结历史较强的 W1，继续训练 H2，Engineer 能否生成更好的策略，并改善最终系统表现？

### 实验设计

**训练：**

- 从真实 H2-step31 的 Actor/Critic 权重继续，W1 权重完全冻结，训练前后校验其摘要。
- 新增一个历史大小的 Engineer 阶段：**32 个 native step，batch6，192 条消费轨迹**。
- 前五步 Critic 预热；实际 **27 次 Actor、32 次 Critic 更新**。27 次 Actor 梯度均非零。
- Actor LR `2e-6`，Critic LR `1e-5`，K1，温度 0.7，32K 上下文、4096 生成 token 上限。
- H2 没有保存 Adam，因此明确使用 fresh Adam；采用修正后的普通 SAO 独立轨迹/行为概率处理，不加入 MA、KL 或 ValueNorm。
- 复用原 W1 失败证据；样本编号从 384 继续到 575，生成新 rollout，不重放旧训练轨迹。

**选择与评测：**

- 只评最终 extra32 检查点，不搜索中间训练终点。
- 前后各八组相同 prompt、seed 和生成预算；同时报告所有候选槽的平均任务质量与合法率。
- 用原 selection24、`mean@2` 选择；两侧共享旧 h2 incumbent，必须达到原定至少 +2 pp 的提升门槛才替换。
- 即使新候选没有替换旧 h2，也必须在主 200 题报告新模型最好合法候选，避免用回退旧策略掩盖新模型表现。

**训练与成本表：**

| 指标 | 核验结果 |
| --- | ---: |
| native step / Actor 更新 / Critic 更新 | 32 / 27 / 32 |
| Solver 更新 | 0 |
| 声明消费 Engineer 轨迹 | 192 |
| 实际记录生成轨迹 | 198 |
| 训练生成 token | 503,089 |
| 已完成训练奖励环境 episode | 1,120 |
| 选择和主评新增环境 episode | 1,176 |
| 评测候选生成 | 16 次，34,481 token |

198 条生成中有六条异步预取额外记录，不能任意挑选某条重复记录冒充精确消费映射。其中 112 条合法、86 条无效，62 条以 `length` 结束；这些分母是 **198 条生成记录**，不是 192 条消费样本。已完成的奖励评测条目没有失败尝试，但未持久化/未完成工作不能由该账本恢复。

**候选质量表：**

| selection24 指标 | 续训前 H2 | 续训后 H3 |
| --- | ---: | ---: |
| 合法候选数 | 2/8 | 8/8 |
| 八个候选槽平均成功率 | 88.8021% | 89.5833% |
| 最好新合法候选 | 93.75% | 93.75% |
| 保存的旧 h2 incumbent | 95.8333% | 95.8333% |
| 最终部署选择 | 保留旧 h2 | 保留旧 h2 |

候选槽均分差为 **+0.78125 pp，95% 区间 [−4.4271, +5.7292] pp**。该指标是八种策略各自 `mean@2` 的平均，不是同一种策略独立采样 16 次。无效候选按既有合同映射为空策略，不从分母删除。

**主 200 题结果表：**

| 系统/评测别名 | 成功数 | 成功率 | 解释 |
| --- | ---: | ---: | --- |
| W1 + 空策略 | 173/200 | 86.5% | 本次同期对照 |
| 续训前按规则选中的系统 | 178/200 | 89.0% | 旧 h2 |
| 续训后按规则选中的系统 | 178/200 | 89.0% | 仍是同一旧 h2 |
| 历史保存 h2 的本次复测 | 178/200 | 89.0% | 与上两行是同一物理报告的别名 |
| W1 + H3 最好合法新候选 | 173/200 | 86.5% | `after-proposal02`，没有被隐藏 |

| 对比 | 差值 | 配对 95% 区间 | 改善 / 退化题数 |
| --- | ---: | --- | ---: |
| H3 新候选 − 旧 h2 | −2.5 pp | [−7, +2] pp | 8 / 13 |
| H3 新候选 − 空策略 | 0 pp | [−4.5, +4.5] pp | 10 / 10 |

前后部署、旧 h2 三个别名共用同一策略和同一物理评测。其差值 0、区间 [0,0] 只反映数据相同，**不能作为两个不同模型统计等效的证据**。主评实际只有三个不同策略，而不是五次独立实验。

### 实验结论

1. **这一阶段确实更新了模型，而且这八槽候选的格式合法性改善了。** 不能写成 Engineer 完全没学到东西。
2. **没有可靠的任务效果提升。** 候选均分区间跨零；最好新候选未超过 incumbent；部署系统没有变；新候选在主评点估计还低于旧 h2。
3. 新候选 −2.5 pp 的区间跨零，也不能写成已经统计确认退化。
4. 最好新候选主要提供通用初始化提示；其 `on_post_step` 更新阶段变量但返回 `None`，没有其他 hook 消费这些变量来提供后续动态提示。更短、更容易合法，不等于更有效的任务策略。
5. **这补齐了 SAO 的另一方向续训检验**：在本配置和再训练一阶段的预算下，没有发现能突破当前瓶颈的证据。不是任意训练时长或超参数下的绝对上限证明。
6. 本次旧 h2=89% 不覆盖历史 92%，也不能据此把 MA 优势目标降低到只超过 89%。

证据：[冻结协议](../runs/alfworld_fixed_w1_h3_v1/protocol.json)、[训练审计](../runs/alfworld_fixed_w1_h3_v1/TRAINING_AUDIT.json)、[选择结果](../runs/alfworld_fixed_w1_h3_v1/SELECTION.json)、[完整结果](../runs/alfworld_fixed_w1_h3_v1/COMPLETE.json)、[最终独立审计](../runs/alfworld_fixed_w1_h3_v1/FINAL_AUDIT.json)、[真正 H3 的 200 题原始报告](../runs/alfworld_fixed_w1_h3_v1/reports/after_candidate.json)、[最好新候选](../runs/alfworld_fixed_w1_h3_v1/selection/patches/after-proposal02.json)。

## 实验 4：SAO 的后期下降，在真实行为上表现为什么？

### 研究问题

Solver 成功率下降，主要是评测程序错误、输出格式错误，还是实际任务执行行为退化？能否据此确定训练崩落的唯一原因？

### 实验设计

只读审计实验 2 已有原始轨迹，不新增训练或模型 rollout。重点配对 extra40 与 extra60 的同题结果，检查失败终止、50 步上限、动作语义和已成功任务的回归。

**结果表：**

| 策略 | extra40 | extra60 | 改善题数 | 退化题数 | 退化中跑满 50 步 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定 h2 | 159/200，79.5% | 76/200，38.0% | 5 | 88 | 69/88 |
| 空策略 | 166/200，83.0% | 92/200，46.0% | 4 | 78 | 72/78 |

已检查的失败案例包括反复搜索/导航、引用不存在的物体或位置、语义不正确的动作。`status=completed` 只表示执行流程结束，不代表环境任务成功；合法 JSON 也不代表合法或有效的环境动作。

### 实验结论

- 下降在配对任务的实际执行中可见，不是单靠汇总分数形成的假象。
- 大量回归以跑满环境步数结束，反映长程行为失效；**50 个环境步与 4096 个生成 token 不是同一个限制**。
- 现有审计描述了失败模式，尚未隔离导致参数更新不稳定的唯一因果机制。不能未经实验就断言全部由梯度爆炸、熵塌缩或 token 耗尽造成。

证据：[SAO 失败行为审计](../runs/alfworld_ma_convergence_v1r4/SAO_FAILURE_MODE_AUDIT.json)。

## 实验 5：从 H0/W0 开始，原始 MA 联合训练是否自然改善？

### 研究问题

共享团队奖励、引入协作上下文的联合训练框架，是否无需进一步调优就能稳定提高表现？

### 实验设计

保留早期 H0/W0 原始联合分支，报告初始、64 轮和后期选中端点。64 轮的 1024 条总角色消费只构成名义轨迹总数匹配，不能代替按角色、环境交互和选择成本的完整匹配。

**结果表：**

| 端点 | 消费角色轨迹 | 带保存 Harness | 空策略 |
| --- | ---: | ---: | ---: |
| 初始 H0/W0 | 0 条新增训练 | 85.0% | 74.0% |
| 第 64 轮 | 1,024 | 87.5% | 78.5% |
| 后期选中第 112 轮 | 1,792 | 82.5% | 80.5% |

整次运行实际消费 1,840 条、记录完整生成 1,872 条；不能仅按最后选中的 1,792 条计算研究成本。后期带策略端点相对初始为 **−2.5 pp，95% 区间 [−7.5, +2.5] pp**。

### 实验结论

- 原始 MA 框架不是“训练越久必然越好”：早期上升、后期组合表现下降。
- 空策略下的 Solver 点估计改善，但最终策略组合未能保持收益；不能把整个系统的退化简单归咎于 Solver 完全没学习。
- 这一负面分支必须与后续最强分支同时报告，不能只展示调优后 97% 的高点。

证据：[原始 H0/W0 联合训练最终审计](../runs/alfworld_joint_h0w0_8h_v1/FINAL_AUDIT.json)。

## 实验 6：已经尝试的 MA 细节与初始化调整，哪些有用、哪些没有？

### 研究问题

无效策略的 credit 处理、队友权重、H0 锚定、较强 Solver 初始化和学习率平衡，能否解决 MA 早期退化或突破历史表现？

### 实验设计

整理已有冻结对照，不新增大批消融。下表各行属于不同实验，具有各自初始化、预算和选择协议，不能把跨行分数当成单因素顺序增益。

**结果表：**

| 实验 | 主要设计 | 结果 | 支持的解释 |
| --- | --- | --- | --- |
| 无效策略 credit 处理 | 修改无效生成的训练 credit，保留对照及诊断复测 | 实验组 78%；原对照 87.5%，诊断复测 89% | 当前处理未改善；固定 H0/空策略的劣势也存在，不只是新策略选错 |
| 队友权重消融 | 将 bounded teammate 权重替换为 1 | 84% 对复用对照 87.5%；差 −3.5 pp，区间 [−8, +1] pp | 没有改进证据；未隔离权重异质性与平均缩放的影响 |
| 固定 H0 锚定交错 | 对照与 H0 锚定在各自固定终点评估 | 带策略 91% 对 90%；空策略 87% 对 83% | 当前锚定方案没有端点收益，均未超过历史 92% |
| Solver 初始化 | MA64 与 W1 初始化，各 48 轮、640 条消费；最后均保留 H0 策略 | 90% 对 92%；W1−MA64 为 +2 pp，区间 [−1.5, +5.5] pp | 强初始化有正向点估计，尚未证实可靠优势，更不能归因于新 Engineer 策略 |
| 平衡学习率整体候选 | 同一既有种子；H LR 从 `1e-7` 升到 `5e-7`，W LR 从 `1e-6` 降到 `5e-7`；固定 48 轮 | 原同种子分支 + H0 为 92.5%；新分支 + 自动策略为 93.5%；新 Solver 空策略 90.5% | 首次为这一候选保留自动新策略，但两种 LR 与运行时一起改变，不是单因素归因 |

平衡学习率分支的 48 轮消费/生成/训练环境成本分别为 **640 / 768 / 384**，最终选中的是自动生成的 `proposal06`，不是人工恢复提示。后续长期 MA 从这一分支继续。

### 实验结论

- 多个直观修正没有带来收益，部分点估计明显更差；当前没有证据支持把这些修正都叠加使用。
- 更强初始化与学习率平衡提供了继续研究的候选，但不能据此宣称 MA-SAO 方法已得到普遍验证。
- 全部失败结果保留，后续一次只推进一个明确冻结的候选，避免通过大量隐性搜索只报告最佳结果。

证据：[credit 对照审计](../runs/alfworld_validity_credit_study_v1/FINAL_AUDIT.json)、[队友权重审计](../runs/alfworld_peer_weight_ablation_v1/FINAL_AUDIT.json)、[H0 锚定审计](../runs/alfworld_anchor_retention_study_v2/FINAL_AUDIT.json)、[初始化对照](../runs/alfworld_solver_initialization_study_v1/PRIMARY_COMPLETE.json)、[平衡学习率结果](../runs/alfworld_ma_balanced_fast_v1r2/RESULT.json)、[平衡学习率设计](../runs/alfworld_ma_balanced_fast_v1r2/DESIGN.json)。

## 实验 7：人工目标恢复提示的高分，是否构成 MA 自动学习优势？

### 研究问题

人工提示曾带来较高开发分数，这种收益能否通过固定复测保持？能否归因于 Engineer 自动学习？

### 实验设计

固定已有候选和历史 SAO 对照，用两个推理 seed 复测并按题目聚类合并统计。这不是两次独立训练，也没有在验证时重新训练 Engineer。

**结果表：**

| 推理 seed | 历史 W1+h2 | MA Solver + 固定人工提示 |
| --- | ---: | ---: |
| 20261005 | 92.0% | 93.5% |
| 20261006 | 90.5% | 90.5% |
| 合并 | 91.25% | 92.0% |

合并差为 **+0.75 pp，95% 区间 [−2.5, +4] pp**，未通过预设优势验证。此前的 94.5% 保留为探索性开发结果，不能代替固定验证。

### 实验结论

- 人工提示的探索结果没有转化为可靠的验证优势。
- 无论其分数如何，人工提示改动都不能归因于 Engineer 自动学习。
- 这一支与后续自动生成 P144 的 97% 不是同一策略来源，不能混淆。

证据：[双推理种子验证](../runs/alfworld_explicit_hint_verification_v1/RESULT.json)、[历史实验记录](./EXPERIMENT_LOG.md)。

## 实验 8：调优后的 MA 继续训练，最终能达到什么平台？

### 研究问题

从短程平衡学习率分支继续，MA 是否持续上升，能否形成高于历史 SAO 的稳定长期平台？

### 实验设计

- 从总 48 轮、93.5% 的已保存分支继续，按每 48 轮评估一次。
- 旧调度始终是 **joint / joint / Solver-only**：每三轮双方更新两轮，再只更新 Solver 一轮，即 H:W 更新比 2:3。第三轮 Engineer 仍生成提案，计入生成成本。
- 起始第 48 轮边界因父模型没有 Adam 而 fresh Adam 一次；之后使用原生状态恢复，不在每块重置优化器。
- 四个完整 96 轮续训块，共新增 **5120 条消费角色轨迹、6144 条生成角色轨迹、3072 个训练环境 episode**；这些不是包含所有上游初始化和失败尝试的总研究成本。
- 预声明的观测平台：至少 192 新增轮；最后四个 48 轮间隔点跨度不超过 2 pp；连续三个间隔没有至少 0.5 pp 的新高。

**完整结果表：**

| 总轮数 | 保存 Harness + Solver | 同一端点 Solver + 空策略 |
| ---: | ---: | ---: |
| 48 | 93.5% | 90.5% |
| 96 | 93.5% | 91.5% |
| 144 | **97.0%** | 92.5% |
| 192 | 96.0% | 90.0% |
| 240 | 94.5% | 92.5% |
| 288 | 93.0% | 92.0% |
| 336 | 94.5% | 93.0% |
| 384 | 94.5% | 94.5% |
| 432 | **95.0%** | 95.5% |

最后四个带策略点是 **93% / 94.5% / 94.5% / 95%**，跨度恰为 2 pp，满足原定观测平台。第四块已完整结束，未启动第五块；最终审计通过。

| 最终端点对历史 SAO | 结果 |
| --- | --- |
| MA 95% − 历史 SAO 92% | +3 pp |
| 题目配对 95% 区间 | [0, +6.5] pp |
| 改善 / 退化 | 9 / 3 题 |
| “区间下界严格大于 0”的预设门槛 | 未通过 |

### 实验结论

- **曾达到 97%，但没有稳定保持。** 旧混合调度的最终观测平台约为 93%–95%，不是 97%。
- 最终 95% 比历史 92% 高 3 pp，最后四个点也都高于 92%，这是值得保留的正向开发证据。
- 区间下界为 0，不满足严格大于 0 的门槛；不能报告为已经可靠证明收敛后优于 SAO。
- 最终空策略 95.5% 甚至略高于保存策略组合 95%，提示早期有益策略不一定始终适配后期 Solver；这只是端点现象，不是确定的因果归因。
- “观测平台”是有限预算内的预定义停止条件，不是数学收敛或理论上限。

证据：[完整长期结果](../runs/alfworld_ma_convergence_v1r4/RESULT.json)、[最终审计与优势门槛](../runs/alfworld_ma_convergence_v1r4/FINAL_AUDIT.json)。

## 实验 9：97% 高点是否真的学到了有效协作策略？

### 研究问题

97% 是自动 Engineer 的真实策略收益、Solver 自身变强，还是仅由选择噪声、手工提示或评测捷径造成？

### 实验设计

分两个层次，不把保存策略与 Engineer 分布混为一谈。

**A. 审计原高点。** 核查总 144 轮的原始生成记录、冻结策略、运行时动作和逐题奖励；比较同一 Solver 的空策略与自动 P144。P144 来自 `95-proposal07`，没有手工修改。

**B. 固定交叉复测。** 冻结 W48/W144 与空策略、初始 H0 策略、P48、P144，进行 2×4 个组合的同题评测，共 **1600 个环境 episode、零新增训练**。

**原高点结果表：**

| 同一 W144 的比较 | 空策略 | P144 | 改善 / 退化 |
| --- | ---: | ---: | ---: |
| 全部 200 题 | 185/200，92.5% | 194/200，97.0% | 9 / 0 |
| 排除 selection48 后的 152 题 | 141/152 | 146/152 | 5 / 0 |
| 再排除三个训练目标重合项后的 149 题 | 139/149 | 143/149 | 4 / 0 |

全部 200 题上的策略收益为 **+4.5 pp，95% 区间 [+2, +7.5] pp**。九个改善任务中七个涉及热/冷/清洗后放置，两个涉及取两个物体；实际策略消息包含变换顺序与阻止过早放置等提示。

环境成功仍由 `info['won']` 判定；被策略阻断的动作仍消耗 50 步循环预算，没有通过扩大环境步数上限获得额外机会。

**固定交叉复测结果表：**

| 固定策略 | W48 | W144 |
| --- | ---: | ---: |
| 空策略 | 90.0% | 91.5% |
| 初始 H0 策略 | 91.5% | 92.5% |
| P48 | 95.0% | 95.0% |
| P144 | 95.0% | 96.5% |

| 复测对比 | 差值 | 配对 95% 区间 | 改善 / 退化 |
| --- | ---: | --- | ---: |
| 同一 W144：P144 − 空策略 | +5 pp | [+2, +8] pp | 10 / 0 |
| 完整组合：W144+P144 − W48+P48 | +1.5 pp | [−0.5, +4] pp | 4 / 1 |
| 同一 W144：P144 − P48 | +1.5 pp | [−1, +4] pp | 5 / 2 |

前两项复测对比由保存的逐题报告用项目配对统计函数重算，bootstrap seed 为 20261007；第三项沿用原交叉结果。不要把“换整个组合”和“只换策略”的配对统计混用。

### 实验结论

1. **支持自动保存的 P144 具有真实任务效用**，不是仅靠格式合法或程序正常结束获得分数；固定 Solver 的两次观察均有正向策略收益。
2. **整体继续训练收益仍不明确**：新的完整组合比旧组合高 1.5 pp，但区间跨零。
3. 新旧复测与原曲线分数不完全相同，不能拼接成一条伪装为同批次的单调曲线。
4. 排除 selection48 和三个目标重合项后仍有收益，是有用的事后敏感性证据；但高点是在看过开发曲线后选择的，剩余 152/149 题也不是事前封存的独立测试。
5. 这回答“保存的自动策略有没有用”，尚未证明整个 Engineer 生成分布更强，也未确立跨训练种子的泛化能力。

证据：[原高点行为审计](../runs/alfworld_peak97_crossplay_v1/BEHAVIOR_AUDIT_v2.json)、[固定交叉结果](../runs/alfworld_peak97_crossplay_v1/COMPLETE.json)、[原 97% 端点](../runs/alfworld_ma_convergence_v1r4/block-01/round0096_COMPLETE.json)、[原自动策略](../runs/alfworld_ma_convergence_v1r4/block-01/round0096/patches/95-proposal07.json)、[配对统计实现](../whale_local/sao_bridge/analysis.py)。

## 实验 10：从真实 97% 点恢复，每轮双方更新，能否更高、更稳定？

### 研究问题

旧框架已经联合训练，但 Engineer 更新较少。将 H:W 从 2:3 调整为 1:1，并利用现有 MAPPO-like 协作训练机制，能否保持或超过高点？

### 实验设计

- 从**总 144 轮、97.0%** 的真实检查点恢复，而不是不存在的 97.5% 检查点。原先拟从 93.5% 起跑的候选在开训前取消，零研究轨迹，不算一次失败训练。
- 恢复两角色原生 FP32 Actor/Critic、Adam 与 ValueNorm；不以会舍入参数的 HF 导出代替 learner 原生状态。
- 原生 Adam 初始 Actor/Critic step：H 为 64/128，W 为 96/192；本分支没有新的 fresh Adam。它们是原生优化器计数，不是所有上游训练的总更新数。
- 保持原 H0/W1 固定 reference、数据顺序和 source index768 续接。
- H/W Actor LR 均为 `5e-7`，Critic LR `2.5e-6`，KL0.02，K2，lag1，batch8，32768 上下文和 4096 动作 token 预算。
- 改为每轮双方更新，同时把原 Solver-only 第三轮的队友校正统一为 bounded。**两项一起改变，不是纯更新频率的单变量消融。**

**调度与成本表：**

| 每 96 轮 | 旧混合调度 | 新每轮双方更新 |
| --- | ---: | ---: |
| H Actor 更新 | 64 | 96 |
| W Actor 更新 | 96 | 96 |
| H / W Critic 更新 | 128 / 192 | 192 / 192 |
| 消费角色轨迹 | 1,280 | 1,536 |
| 生成角色轨迹 | 1,536 | 1,536 |
| 训练环境 episode | 768 | 768 |

生成量和环境量相同，但消费量增加 **20%**，因此不是严格同消费预算的效果对照。97% 起点前的成功谱系已消费 1920 条角色轨迹；上游 W1/MA64 初始化与其他研究成本另计。

**MAPPO-like 的确切含义：**

| 属性 | 当前实现 |
| --- | --- |
| Actor 参数 | Engineer 与 Solver 各自独立 |
| Critic 参数/主干/优化器 | 各自独立，不共享网络参数 |
| 共享信息 | 团队奖励与协作/队友上下文 |
| 训练机制 | CTDE 式上下文、ValueNorm、bounded teammate correction |
| 动作执行顺序 | Engineer 先提出策略，再由 Solver 执行；联合更新不是同时发生物理动作 |

因此本实验是**既有联合框架内的调度候选**，不是第一次联合训练，也不宣称逐项复现了标准 MAPPO。

**已核验早期结果：**

| 项目 | 状态 |
| --- | --- |
| 首个新增 96 轮原生训练 | 完成，总 240 轮状态已保存 |
| 首块结束原生 Actor step | H160 / W192 |
| 首块结束原生 Critic step | H320 / W384 |
| 总 192 轮，即新增 48 轮端点 | 带策略 186/200 = 93%；空策略 181/200 = 90.5% |
| 总 240 轮，即新增 96 轮端点 | 带策略 184/200 = 92%；空策略 178/200 = 89% |
| 新分支平台/最终优势 | 尚不能判断 |

| 配对比较 | 差值 | 题目 bootstrap 95% 区间 | 改善 / 退化 |
| --- | ---: | --- | ---: |
| 总192 − 父点97% | −4 pp | [−7, −1] pp | 1 / 9 |
| 总240 − 父点97% | −5 pp | [−8.5, −2] pp | 0 / 10 |
| 总240 − 历史SAO92% | 0 pp | [−3.5, +3.5] pp | 6 / 6 |

总240的八个新候选都合法；最好新候选在 selection48 为47/48，incumbent为46/48，未达到多两题的替换门槛，故保留新增48轮的策略。没有利用200题换候选。

H3 插入期间只暂停过 CPU 协调器，两个 learner 完成并保存了原96轮块。随后总240主评的第一次启动，被本次一个未屏蔽CUDA的“CPU测试”进程意外创建的GPU上下文触发安全 watchdog；当时 **0 个环境episode**。这是我们的测试隔离失误，不是训练 collapse。原失败证据保留，后续测试全部屏蔽CUDA，只重试相同模型/策略/任务/seed的主评，不重训、不重选。第二次主评的完整原始报告已核验。

原协调器已经退出，不能继续称“原PID仍在暂停”。已启动并核验新的续训协调器，等待本轮WHALE完成并出具两节点GPU释放收据后，从block02恢复原生FP32/Adam/ValueNorm；不重做block01、不重置Adam、不热改算法。插入WHALE时间仍计入原24小时安全上限；不足三小时不再开启新96轮块，时间停止不等于收敛。

平台条件相对总 144 轮起点计算：至少新增 192 轮，最早总 336 轮才有资格判断；最后四点跨度不超过 2 pp，三个间隔无至少 0.5 pp 新高。上限为八个 96 轮块或 24 小时。Serving/RNG 会重启，原生状态恢复不等于逐位随机状态续跑。

### 实验结论

**早期结果明确为负：97% → 93% → 92%，尚没有超过97%的证据。** 这已经允许按用户期限安排WHALE资源窗口，但不能宣布MA研究结束或已到长期平台。父点又是选中过的开发峰值，区间也不修正该峰值选择，不能把早期下降外推为所有联合调度永远无效。

后续必须报告完整曲线与预算停止原因，并同时考虑历史 92% 和本次固定 W1 的同期 SAO 证据；原冻结门槛不改写，新增比较另列。

证据：[冻结设计](../runs/alfworld_ma_simultaneous_from97_v1/DESIGN.json)、[早期结果与原生状态核验](../runs/alfworld_ma_simultaneous_from97_v1/EARLY_RESULT.json)、[首块原生记录](../runs/alfworld_ma_simultaneous_from97_v1/block-01)、[零episode评测恢复](../runs/alfworld_ma_simultaneous_from97_v1/ENDPOINT_RECOVERY_COMPLETE.json)、[实际续训协调器状态](../runs/alfworld_ma_simultaneous_from97_v1/RESUME_MONITOR.json)、[取消的 93.5% 起点候选](../runs/alfworld_ma_simultaneous_convergence_v1/CANCELLED_BEFORE_TRAINING.json)。

## 实验 11：大规模训练前的加速是否有效，倍率应如何报告？

### 研究问题

在保留训练语义和正确性检查的前提下，能否明显降低 MA 训练成本？合成优化器加速能否代表完整 RL 的加速？

### 实验设计

- 使用隔离的 FLA0.5.2 / Triton3.7.1 加速路径。
- 仅将固定 reference 存储改为 BF16；Actor/Critic master 保持 FP32，保留完整 32K 上下文。
- 验证原生前向、有限非零梯度、参数变化、状态恢复和资源释放。
- 分别统计 32K 合成更新、真实稳态轮次、成功尝试的训练墙钟；保留旧 Triton 正确性拒绝与首个 FLA 反向 OOM 的失败记录。

**结果表：**

| 度量 | 之前 | 加速后 | 倍率 |
| --- | ---: | ---: | ---: |
| 合成 Actor + 两次 Critic 稳态更新 | 346.94 秒 | 39.17 秒 | 8.86× |
| 真实联合训练相同稳态轮次均值 | 96.90 秒 | 49.56 秒 | 1.96× |
| 成功尝试从两角色启动到训练完成 | 101.09 分钟 | 60.00 分钟 | 1.68× |

### 实验结论

- 已取得实质性的训练速度改善，但**不能把 8.86× 写成端到端 RL 加速**。
- 真实轨迹长度与中间检查点保存频率也发生变化，全部墙钟收益不能只归因于 FLA。
- 成功训练墙钟不包含失败尝试、加速验证、输入重哈希和完整主评测；它不是整个研究周期的总耗时。
- 加速通过正确性验证，不等于后续训练轨迹逐位相同，也不构成 MA 算法效果更好的证据。

证据：[加速验证](../runs/alfworld_training_speed_v4/COMPLETE.json)、[真实训练与时间独立审计](../runs/alfworld_ma_balanced_fast_v1r2/INDEPENDENT_AUDIT.json)、[隔离依赖](../requirements/training-acceleration.txt)。

## 实验 12：本轮 WHALE 在明确的 SAO 采样参照下表现如何？

### 研究问题

在原W0和固定本地H0下，在线RSFT与Meta-Harness交替能否取得有竞争力的ALFWorld效果？在640个Solver episode与384个提案的名义预算下，其系统效果相对历史SAO如何？

### 实验设计

完整方法、停止条件与不匹配项见M8。以原W0+h0的开发10题结果作为性能停止参照，最终仅评价预先冻结的部署模型/策略；主200不得参与候选选择、学习率调整或提前停止。

| 顺序 | 实验 | 当前状态 | 必须保留的比较口径 |
| --- | --- | --- | --- |
| 1 | 固定 W1 继续 Engineer，补齐 SAO 瓶颈检验 | 已完成并审计，见实验 3 | 同期旧策略、空策略、新原始候选、候选均分与合法率全部报告 |
| 2 | 从 97% 继续 MA，检验高点后的稳定高度 | 早期97→93→92%，待WHALE释放后继续 | 完整曲线、原生状态、预算停止、历史和同期 SAO 参考 |
| 3 | WHALE 在 ALFWorld 上与明确 SAO 参照做匹配样本比较 | 主200为81.5%，实际640/216；原始结果已核验，但未完成384提案 | 批量修订、时间截断、角色消费、环境评分、搜索和初始化成本分别披露 |
| 此后三项完成后 | MA 与 SAO 的样本效率和更细机制归因 | 延后 | 不只比较各自最佳点；对齐样本口径和选择过程，先固定可解释的配置 |

**当前实际观测，不是最终分数：**

| 阶段 | 样本与结果 | 可以说明什么 |
| --- | --- | --- |
| 旧v7部分执行 | 224个Solver episode、3553个成功轨迹调用、60个Adam小步；随后中止 | 没有MH和最终200题，不是完成基线 |
| v9原生32K验证 | warm Adam、FP32恢复与vLLM重载通过，1822.47秒，0研究episode | 验证计算/恢复路径，不验证任务能力 |
| v10新评分验证 | 56测试、48subtest；4个真实环境、合成回复的CPU probe | 验证隔离和错误传播，0模型调用，不是准确率 |
| 本轮初始开发评测 | W0+h0：6/10=60%，mean turns29.5 | 仅是固定开发任务上的训练前停止参照 |
| v10已保存训练前缀 | 64个Solver episode，两批分别25/32、27/32成功；552、555个成功调用，各9个Adam小步 | 真实训练证据，不是主评或完整WHALE结果 |
| v11原生与数值门控 | 32K/warm-Adam通过；旧跨多步门槛失败；独立固定权重全梯度RMS0.0032533低于预声明门槛0.0054973 | 保留失败，不宣称多步数值轨迹一致 |
| v11已保存训练前缀 | 继承64题后新增96题，合计160题、43个Adam小步 | 不是五条独立训练重复，原64题不重采 |
| v12批量修订与服务验证 | 保留160+后续480题，仍计划384提案；128路完整32K并发通过，293.05秒 | online刷新频率已变，合成服务调用不算研究样本 |
| v12首个160题后缀 | 134/160成功，2495个成功调用、79092响应token；39个Adam小步，43→82，751.58秒 | 原生梯度39次均有限非零，norm范围[0.3391,1.8376]；是训练曝光，不是主评 |
| 首个完整交替周期 | 320个Solver episode+192个MH提案；125个合法，104组不同actions；开发9/10=90% | 训练前6/10=60%；8次搜索全部保留h0，未见MH选择增益；随后进入第二周期 |
| 第二个Solver段与截断的MH段 | 再完成320题，Adam82→159；MH仅完成1/8迭代、24提案，6个合法；开发10/10=100% | 第二个RSFT段完整，第二个MH段不完整；仍保留原h0，不能称完成两次完整交替 |
| 本轮训练后主200 | 163/200=81.5%；strict197为160/197=81.2183% | 原始200、模型/策略/seed、原生Adam、适应性反馈和实际成本均通过核验 |

#### 已核验的最终预算边界

23:14 UTC左右冻结最终模型/策略，之后才启动主200。最终Solver是第640条
训练episode之后的检查点，159个连续Adam小步；两个MH段均保留初始h0。
实际预算为**640个Solver episode、216个MH提案、131个合法提案**，
即一个完整交替周期，加第二个完整RSFT段和1/8个MH段。
停止原因是`reporting_reserve_not_performance`，**不是性能差导致提前停止**。
原目标384个MH提案还差168个，不能宣称本轮已经完成用户要求的同预算复现；
后续主200只能作为这套明确时间截断的本地适配结果。

| 系统 | 主200成功数 | 主200准确率 | 实际Solver episode | 实际MH提案 | 完整交替周期 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WHALE v12最终部署 | 163/200 | 81.5% | 640 | 216 | 1 |

| 历史参考 | 参考主200 | WHALE减参考 | 题目配对bootstrap 95%区间 | 改善 / 退化题 |
| --- | ---: | ---: | ---: | ---: |
| SAO W1+h2 | 184/200=92.0% | -10.5pp | [-16.5,-4.5]pp | 9 / 30 |
| SAO W1+h1 | 181/200=90.5% | -9.0pp | [-15.5,-2.5]pp | 14 / 32 |
| 原W0+h0 | 170/200=85.0% | -3.5pp | [-8.5,+1.0]pp | 8 / 15 |

上述历史参考共用相同主200 ID/seed口径，不是新重训对照；WHALE的提案预算、
学习目标、训练population及选择过程存在已披露差异，不能把该配对区间解释成
严格同计算/同完整采样条件下的纯算法效应。

| 实际成本维度 | 已核验数量 |
| --- | ---: |
| Solver训练episode / 其中成功并进入RSFT | 640 / 529 |
| MH评分episode / 训练前开发episode / 主评episode | 1330 / 10 / 200 |
| 全部本轮Solver环境episode | 2180 |
| Solver生成token / H0提案生成token | 1,411,074 / 693,416 |
| H0提案prompt token | 2,681,646 |
| 成功调用的逻辑SFT forward / response token | 9,972 / 316,420 |
| 实际物理SFT forward / response-token位置 | 17,744 / 562,943 |
| 零loss-mask占位调用 / 连续Adam小步 | 23 / 159 |

物理forward保留前64题的8-rank复制以及后续真实样本分片两种执行模式，
不能把复制当新数据。v12执行墙钟7989.83秒，约2小时13分10秒；它复用了
此前160题的检查点，因此**不是完整实验从零开始的总耗时**。旧v7失败成本、
v9/v11原生验证、固定权重诊断、v12的128次合成服务调用及继承的H0选择成本
均另计，不从账本中删除，也不将复用前缀重复计为新采样。

### 实验结论

- 本轮时间截断的WHALE本地适配得到**81.5%**，没有达到历史SAO的92%。相对原W0+h0的85%也没有明确提升证据；其-3.5pp区间[-8.5,+1]包含0。
- 首周期开发60%→90%发生在RSFT更新后，MH前后仍使用同一h0；只能说这10个开发任务出现了权重更新的正向信号，不能推断主200提升。192个提案中55个截断、11个缺完整patch标记、1个代码语法错误；47个合法候选的逐题reward/turns与h0相同，但全部125个合法候选仍实际评分，没有以去重代替规定采样。
- 第二次RSFT之后开发达到100%，但搜索仍未替换h0；这是10题开发集上的现象，不是主200或独立泛化证据。640/384目标未完成的直接限制是实际吞吐与时间窗口，不是已满足预声明的20pp退化停止条件。
- **交付边界必须承认：按时取得了完整200题结果，但没有按时完成要求的完整同预算WHALE复现。** 不能把216提案说成384，也不能将主20081.5%称为完整两周期WHALE的最终结论。双节点资源已核验释放，MA在23:31 UTC实际启动从原生状态恢复的block02；[恢复边界](../runs/alfworld_ma_simultaneous_from97_v1/NATIVE_RESUME_BOUNDARY.json)通过，原24h截止不延长。
- 结果即使较好，也只能先归于这套本地适配系统。训练population、搜索/选择和损失均不同，不能用一个配对区间宣布纯算法普遍优越。

证据：[独立最终结果与配对统计](../runs/alfworld_whale_sao_matched_v12/RESULT.json)、[原始200题](../runs/alfworld_whale_sao_matched_v12/execution/final200/report.json)、[实际成本](../runs/alfworld_whale_sao_matched_v12/execution/COST.json)、[最终审计](../runs/alfworld_whale_sao_matched_v12/TERMINAL_AUDIT.json)、[冻结设计](../runs/alfworld_whale_sao_matched_v12/DESIGN.json)、[160题保存边界](../runs/alfworld_whale_sao_matched_v11/THROUGHPUT_PAUSED.json)、[640题/159步原生核验](../runs/alfworld_whale_sao_matched_v12/NATIVE640_AUDIT.json)、[首周期192提案与反馈核验](../runs/alfworld_whale_sao_matched_v12/MH0_AUDIT.json)、[实际640/216预算边界](../runs/alfworld_whale_sao_matched_v12/SEARCH_BUDGET_AUDIT.json)、[评测前最终选择](../runs/alfworld_whale_sao_matched_v12/execution/FINAL_SELECTION.json)、[训练前开发报告](../runs/alfworld_whale_sao_matched_v10/execution/initial-development/report.json)、[旧v7真实成本](../runs/alfworld_whale_baseline_v7/execution/COST.json)、[本地WHALE方法仓库](../WHALE)。

## 实验 13：基础单角色学习是否真实发生，哪些早期方向失败了？

### 研究问题

在讨论联合方法之前，9B Solver和Engineer能否分别取得可复核的学习收益？早期负面和未晋升模型是否被保留？

### 实验设计

早期Solver v1/v2各用两个训练seed，30 native step、5 Critic-only warmup；按各自冻结选择规则选checkpoint，再在100题报告。v2修正终端语义，不能只按名字把跨版本差异归于一个理论机制。随后fast-v4同时改变训练池、预算和学习率，是单seed探索包；fast-v5又改变GPU分配、batch、LR和warmup，不是干净的单因素速度实验。

| Solver实验 | 报告协议 | 初始→报告系统 | 配对差及95%区间 |
| --- | --- | ---: | --- |
| v1，seed42，完整concat | 100题，mean@1 | 74%→72% | −2 pp，[−7,+3] |
| v1，seed43 | 100题，mean@1 | 79%→77% | −2 pp，[−11,+7] |
| v2，seed42，终端修正 | 100题，mean@1 | 74%→81% | +7 pp，[0,+15] |
| v2，seed43 | 100题，mean@1 | 81%→74% | −7 pp，[−17,+3] |
| fast-v4，seed42 | 91题，mean@2 | 78.0220%→89.5604% | +11.5385 pp，[+3.2967,+19.7802] |
| fast-v5，seed42 | 59题选择；不打开新模型报告 | W0 66.9492%；step23 59.3220%；step47 67.7966% | 最好只高0.8474 pp，未过2 pp门槛，保留W0 |

v1的双seed平均变化−2 pp，v2平均0 pp，均未通过原重复验证门槛。v2 seed43初始复测81%与v1的79%不同，不能当作逐位配对的实现消融。fast-v4的80个native step/75个Actor更新以及原始报告均通过审计，成为后来W1；其总墙钟199.82分钟包含恢复，不是仅22.96分钟的控制器重启段。

Engineer基础实验使用H0、固定W0、32 native step、batch6、5步warmup。E0/step15/step31各生成相同8组prompt/seed候选，在24题选择；选中step15，随后沿用这些候选评70个不重合目标任务。此处为**旧官方target runner，temperature0、每turn4096、最多50步**；不是当前整episode4096的core200协议。

| Engineer阶段 | 合法候选 | selection24八槽均分 | report70八槽均分 |
| --- | ---: | ---: | ---: |
| E0 | 8/8 | 69.2708% | 72.1429% |
| step15 | 8/8 | 76.5625% | 80.0000% |
| step31 | 5/8 | 67.1875% | 未作为选中模型打开主报告 |
| 同期空策略 | 不适用 | 不作为Engineer分布 | 75.7143% |

step15相对E0 +7.8571 pp，题目聚类区间[+4.4643,+11.25]，44题均分改善/11题退化；相对空策略也高4.2857 pp，因此不只是“少做有害编辑”。完整评测含失败与修复104.63分钟，18.59分钟只是恢复段。step15实际含11次Actor更新，不能把整阶段27次都算进它的权重谱系。

### 实验结论

- 确有单角色学习的正面证据，同时v1/v2的负面与相反seed方向保留。
- H和W各自学会不等于它们联合后一定更好；不能相加两个不同协议上的增益。
- fast-v5的最终报告0差异来自**保留同一W0**，不是训练后模型被证明等效。后期Engineer选择退化也不能隐藏。
- Engineer的证据是固定8个训练evidence prompt下对未见目标任务的改善，不是未见packet或多训练seed稳健性。

证据：[Solver v1 seed42](../runs/alfworld_solver_v1_exact_concat_s42/COMPLETE.json)、[seed43](../runs/alfworld_solver_v1_exact_concat_s43/COMPLETE.json)、[Solver v2 seed42](../runs/alfworld_solver_v2_terminal_s42/COMPLETE.json)、[seed43](../runs/alfworld_solver_v2_terminal_s43/COMPLETE.json)、[fast-v4](../runs/alfworld_solver_fast_v4_s42/COMPLETE.json)、[fast-v5](../runs/alfworld_solver_fast_v5_r1_s42/COMPLETE.json)、[Engineer完整结果](../runs/alfworld_engineer_fast_v1_s42/ENGINEER_VALIDATION_COMPLETE.json)、[Engineer原协议](../runs/alfworld_engineer_fast_v1_s42/evaluation_protocol.json)。

## 实验 14：早期“大阶段交替/CTDE”比较与失败后恢复支持什么？

### 研究问题

早期标记为MA的实验是否已经检验了当前“同episode双方训练”的方法？被拒绝模型、未训练fallback和失败后checkpoint是否被误当成联合改进？

### 实验设计

按算法版本而非统一名称区分三类：

| 版本/实验 | 实际设计 | 原始报告/结果 | 关键边界 |
| --- | --- | --- | --- |
| strong SAO/MA v1 | 共同E15/W0/H1；SAO按H2-W1-W2，早期CTDE分支按W1-H2-W2；无当前同episode联合机制 | 70题mean@2：87.8571% / 89.2857%；差+1.4286 pp，[−1.4286,+4.2857] | 两臂最后是**同一模型与策略**，没有有效协适应 |
| E15/W79交替活动 | 强W1下继续H2、再W2 | 正式最终70题mean@2为96.4286% | H2有效，W2未获晋升 |
| 该活动的MA臂 | 因Git ownership启动错误未进入native训练 | 95%是未训练W1/H1 fallback | 训练预算未匹配，不能作为MA训练分数 |
| 后续真正同episode短程联合 | 最终修正尝试实际提交6轮；OOM后只恢复共同已保存round4 | 200题：新Solver空86%，新策略组合90% | round4每角色仅2次Actor更新，不是round6内存模型或预期round12 |

strong SAO/MA中，SAO W1的step39/79在selection24 mean@2均0%，48个样本全部达到4096 token上限；原生平均任务奖励在steps5–19约70.83%，steps20–59变为0。两个坏checkpoint被拒绝，**没有故意部署0%对照让MA获胜**。SAO W2选择分数62.5%/56.25%，也未替换初始系统。早期MA W1的75%/77.08%同样未晋升。

强W1交替活动的另一诊断矩阵为：空策略91.4286%、H1 95%、H2 97.8571%；H2−H1 +2.8571 pp，[−1.4286,+7.1429]。这些是另一服务运行，**不替换正式96.4286%结果，更不是当前core200的97%高点**。该阶段H2初始/step15/step31合法槽为7/8、1/8、3/8；后续W2完成80/75 native/Actor更新，却仍不通过选择。

真正联合短程最后一次因24075-token轨迹的Actor反向OOM失败，colocated reference有大量未释放reserved cache。已保存round4组合90%对历史92%：−2 pp，[−5.5,+1.5]，5改善/9退化；固定新Solver加策略为+4 pp，[−0.5,+8.5]，16改善/8退化。被评checkpoint消费64条角色轨迹；最后修正尝试实际消费96，加上之前失败尝试72，共168已提交新MA轨迹、至少260个生成动作；继承共同前缀832后是1000已知消费，而不是总研究成本的完整上界。

### 实验结论

- “MA”历史名字下曾有不同实现，不能把早期CTDE交替、当前同episode联合和后来的2:3调度合并为一个算法。
- 同一最终模型不同服务复测的1.43 pp差别不是联合学习收益。
- OOM、端口/Git/通信失败与真正模型退化分开；没有保存的状态不能假称成功恢复。
- 修复后的短程90%仍没达到92%目标，但它只是短程、部分预算证据，不证明联合方法永远较差。

证据：[strong两臂完整报告](../runs/alfworld_strong_sao_ma_v1/PAIRED_REPORT.json)、[强W1交替报告](../runs/alfworld_alternating_e15_w79_8h_v1/PAIRED_REPORT.json)、[恢复后的真实联合审计](../runs/alfworld_joint_async_6h_v5_recovered/FINAL_AUDIT.json)、[历史过程与失败账目](./EXPERIMENT_LOG.md)。

## 实验 15：原始MA后期下降，是权重、策略还是兼容性问题？

### 研究问题

实验5的87.5%下降到82.5%，能否简单解释为Solver权重collapse？固定权重互换策略能否定位问题？

### 实验设计

冻结MA64/MA112两个Solver，分别配空、初始h0、h64、h112，完整2×4交叉复测；无新训练，共1600环境episode。以下是同次诊断，不覆盖实验5的原始端点。

| 固定Solver | 空策略 | h0 | h64 | h112 |
| --- | ---: | ---: | ---: | ---: |
| MA64 | 77.5% | 90.0% | 89.5% | 78.5% |
| MA112 | 79.5% | 86.5% | 87.5% | 83.5% |

固定MA64，h112−h64为−11 pp，区间[−16,−6]。固定h0，MA112−MA64为−3.5 pp，区间[−7,0]；空策略下权重变化为+2 pp，[−2,+6]。原训练的后期selection24候选均分依次为round104 75.7813%、108 73.6979%、112 60.1563%，但max-candidate规则选中了round112的单个最好候选。此时Solver也变化，所以这些均分不是Engineer-only因果比较。

### 实验结论

- 后期策略确实可以在相同Solver上显著有害，不能把全部退化归于Solver或“交替更新”。
- 权重更换在不同固定策略下方向不同，提示兼容性，而非一个全局“模型能力数值”能解释全部结果。
- 好的单候选不代表整个生成分布变好；selection48/24的max选择存在过拟合风险。

证据：[完整交叉矩阵与配对统计](../runs/alfworld_ma_crossplay_v1/COMPLETE.json)。

## 实验 16：h112的错误物品护栏是否阻碍任务恢复？

### 研究问题

后期策略是否在持有干扰物品时，阻止必要的“先放下再继续”动作？具体哪个分支解释了损失？

### 实验设计

先进行固定MA64、原h112/删除整个`on_before_action` hook/空策略的三组200题；再冻结更窄的四组实验，只禁用错误物品阻拦分支，同时保留其他代码。都是**人工机制干预**，零新训练，不作为学习算法成绩。

| 实验 | 空策略 | 原h112 | 仅禁用错误物品阻拦 | 删除整hook |
| --- | ---: | ---: | ---: | ---: |
| 整hook干预 | 79.0% | 81.0% | 不适用 | 87.0% |
| 单分支干预主比较 | 78.0% | 78.0% | 85.5% | 辅助组后续恢复为86.5% |

整hook收益+6 pp，[+2.5,+10]，14改善/2退化。单分支收益+7.5 pp，[+4,+11.5]，15改善/0退化。删整hook相对单分支仅+1 pp，[−1,+3]，不能证明额外收益，也不是等效证明。

第二实验的第四组曾发生传输失败；前三份完整600 episode报告全部保留，只恢复缺失辅助组，并增加空策略复测。恢复新增400 episode；新空策略77.5%不覆盖原78%。辅助差异跨服务会话，必须注明。已保存完整工作至少1000 episode，原失败尝试还可能有未完整记录的工作；不能把缺失进度当零成本。

后验轨迹审计显示15个获救任务中14个出现目标阻拦模式，13个原先跑满环境步数。原生动作规范化后，表面请求的`take`可能变成恢复用的`put`，hook检查的是最终动作，不能只按原始字符串解释。另以三种目标物体×四类任务构造12个CPU恢复状态：h112阻拦12/12，h0/h64/h2/guarded新策略/单分支禁用版均0/12。该合成检查0环境episode、0训练，且“不阻拦”本身不证明策略语义完整正确。

### 实验结论

- 有较强、局部的机制证据：该分支在这些固定权重与复用开发题上有害。
- 这不能被包装成Engineer自动学习成功或MA已突破92%。
- 后来88%的guarded策略本来就通过12个恢复状态检查，因此只加这一个guard并不能解释/解决全部剩余差距。

证据：[整hook实验](../runs/alfworld_h112_guard_ablation_v2/COMPLETE.json)、[单分支实验](../runs/alfworld_h112_branch_ablation_v2/COMPLETE.json)、[单分支最终审计](../runs/alfworld_h112_branch_ablation_v2/FINAL_AUDIT.json)、[CPU恢复检查](../runs/alfworld_recovery_contract_probe_v1/RESULT.json)。

## 实验 17：低LR、KL与保留强策略的方法包能否改善协适应？

### 研究问题

从H0/MA64出发，低Engineer LR、固定参考KL和强策略incumbent能否超过92%？权重收益与策略收益能否分开？

### 实验设计

H/W LR为`1e-7`/`1e-6`、KL0.02、K2、3步warmup、同episode联合、lag1；预定64轮，round32辅助评测。v1在8轮后因可空`tool_calls`边界异常失败，消费128条、记录158条完整生成；v2从同一起点fresh Adam重启，不能称原生续跑。

v2在预算末尾循环提交时发生重复version0请求，实际提交H62/W63，共1000条消费，生成1024条；共同保存点最后为56轮。恢复按预先冻结的规则在40/48/56中选主端点，最终选round40，**计划64轮的完整结果不存在**。

| 端点/矩阵 | 空策略 | h0 | 新选中策略 |
| --- | ---: | ---: | ---: |
| 原主端点round40 | 81.5% | 未在该主矩阵单独测 | 88.0% |
| 原辅助round32 | 83.0% | 未在该主矩阵单独测 | 88.5% |
| 诊断起点Solver | 78.0% | 89.0% | 85.5% |
| 诊断选中Solver | 82.0% | 85.0% | 88.0% |
| 同期W1对照 | 87.5% | 不适用 | 旧h2为89.5% |

主88%相对历史92%：−4 pp，[−9,+1]，9改善/17退化；不能拿稍高的辅助88.5%替换主结果。诊断相对同期W1+h2为−1.5 pp，[−6.5,+3.5]。

固定h0的权重变化为−4 pp，[−8.5,+0.5]，空策略为+4 pp，[−2,+10]。策略相对h0的差分交互为+6.5 pp，[+0.5,+13]。另一个事后对比中，h0相对空策略的收益由11 pp变为3 pp，差分−8 pp，[−15,−1]。完整诊断1600环境episode、零新增训练。

### 实验结论

- 方法包没有超过历史基线，且完整64轮预算未完成，必须按部分训练解释。
- 结果支持“组件兼容性会变化”的研究方向，不支持单个组件全面变强。
- 预算wrap、可空字段等实现故障已被定位，但修复执行流程不是自动获得任务收益；后续修复另经真实两轮GPU训练验证。

证据：[完整方法包/交叉诊断](../runs/alfworld_ma_improvement_study_v2/COMPLETE.json)、[h0兼容性交互](../runs/alfworld_ma_improvement_study_v2/H0_COMPATIBILITY_INTERACTION.json)、[预算修复的真实GPU验证](../runs/alfworld_joint_budget_gpu_smoke_v1/FINAL_AUDIT.json)。

## 实验 18：队友重要性权重的大幅裁剪是不是数值/实现现象？

### 研究问题

大量队友轨迹比率触及0.5/2边界，是否只是因为Actor更新导致off-policy？去掉该权重是否就能提高任务效果？

### 实验设计

先审计前三个Critic-only warmup round，Actor尚未更新。再冻结第一轮8条Engineer行为token，在相同初始权重的原生后端中按原顺序和循环置换各重算一次，无optimizer、无新生成或环境交互。随后另立unit-peer训练消融，不修改既有对照。

| 检查 | 结果 |
| --- | --- |
| warmup的Engineer轨迹比率 | 18/24触界，14低于0.5、4高于2 |
| warmup的Solver轨迹比率 | 0/24触界 |
| 固定权重8条重算 | 7/8仍触界 |
| 原proximal logsum与重放logsum最大差 | `1.34937e-6` |
| 置换batch顺序导致的最大逐token差 | 0 |
| 成本 | 成功重算16条+前一失败尝试已forward8条；0更新、0新生成、0环境episode |
| unit-peer后续32轮任务结果 | 84%对复用bounded对照87.5%，−3.5 pp，[−8,+1] |

Engineer整轨迹比率作用于**Solver**优势，反之亦然；不能把角色对应弄反。3000左右token上很小的每token偏差经求和后就可能有较大的序列比率。去掉权重也改变平均梯度缩放，不只移除异质性。

### 实验结论

- Actor参数变化不能解释全部偏差；当前重放排除了被测试的batch顺序效应。
- 尚未完整隔离backend数值、采样处理或其他实现因素，更未证明其是准确率下降的唯一原因。
- unit-peer负面结果不支持“只把权重设为1就能解决”。当前保留bounded方案，同时明确它的经验性和局限。

证据：[warmup概率审计](../runs/alfworld_validity_credit_study_v1/WARMUP_LIKELIHOOD_AUDIT.json)、[真实重放审计](../runs/alfworld_fixed_weight_probability_replay_v5/AUDIT.json)、[unit-peer任务审计](../runs/alfworld_peer_weight_ablation_v1/FINAL_AUDIT.json)。

## 实验 19：基础六系统表和官方Engineer比较是否支持简单“越训越强”？

### 研究问题

使用相同题目与推理合同，W1、H1、H2各自提供多少系统收益？官方已发布Engineer与本地续训Engineer的比较是否一致？

### 实验设计

先完成24题mean@2快速表，再冻结所有模型/策略，用原24题的sample0加176题组成core200，不按新结果重选模型。

| 系统 | 24题快速表，mean@2 | core200，mean@1 | strict197 |
| --- | ---: | ---: | ---: |
| W0 | 68.7500% | 147/200=73.5% | 144/197=73.10% |
| W0+官方target-specific策略 | 最好8候选85.4167% | 166/200=83.0% | 163/197=82.74% |
| W0+历史固定H1策略 | 75.0000% | 148/200=74.0% | 146/197=74.11% |
| W1 | 87.5000% | 174/200=87.0% | 172/197=87.31% |
| W1+H1 | 91.6667% | 181/200=90.5% | 178/197=90.36% |
| W1+H2 | 97.9167% | 184/200=92.0% | 181/197=91.88% |

core200中，W1−W0为+13.5 pp，[+6.5,+20.5]；H1在W0上仅+0.5 pp，[−4.5,+5.5]；H2对W1+H1为+1.5 pp，[−1.5,+5]。最强组合相对官方策略组合为+9 pp，[+3.5,+15]，但初始化/训练/search不同，并非相同总成本的算法对照。

另以相同8个W79历史evidence prompt、相同候选数和W0的24题原生评测比较两个Engineer：

| Engineer | 合法候选 | 八槽平均 | best-of-8 |
| --- | ---: | ---: | ---: |
| 官方target-specific版本 | 8/8 | 80.2083% | 85.4167% |
| 本地E15 | 7/8 | 69.5313% | 81.2500% |

该协议下本地E15并不优于官方版本，与实验13的不同target-runner/evidence条件不可混合。官方论文500题的分数也不能直接当成本地200题对照。

### 实验结论

- 在本地原生core200上，W1权重收益明显；策略的增量取决于Solver，H1在W0上几乎没有点估计收益。
- 92%是完整可追溯的历史组合成绩，不能被后续89%、89.5%或90%的复测覆盖。
- 24题上的接近98%不能当成200题成绩，best-of-8不能改称候选均分，已扩展/反复使用的集合不能改称封存测试。

证据：[core200完整表与对比](../runs/alfworld_core_200_oneshot_v2/COMPLETE.json)、[快速表与额外Engineer诊断记录](./EXPERIMENT_LOG.md)、[core200固定协议](../runs/alfworld_core_200_oneshot_v2/protocol.json)。

## 统计、预算与工程有效性附录

### A1. 统计方法与解释规则

对同题两个系统，先在每题内部平均其固定samples，得到 \(d_q=\bar R_{\rm candidate,q}-\bar R_{\rm control,q}\)；对题目有放回抽样10000次，报告均值差的percentile 95%区间。bootstrap seed写在各结果JSON，通常使用项目固定值；实验9的交叉统计也用同一实现重算。两推理seed合并时以题目聚类，不将400条当成400个独立题目。多策略槽均分同样先在题目内汇总，不将8个策略×题数错误地作为独立样本。

这些区间不包含训练seed方差，也未校正重复调优、选中最高checkpoint、共享目标家族或多重探索。不同训练seed的Solver v1/v2是有限的真实重复；人工提示的两个seed只是**推理重复**。独立训练重复分支另得到H0组合92.5%、空策略90.5%，对历史92%差+0.5 pp，区间[−3,+4]；后续人工提示扩展被暂缓，不是已完成的多训练seed人工提示验证。

所有“区间包含0”只能解释为未得到明确差异证据，不是统计等效；平台规则也不是数学收敛。最终严格优势门槛要求配对区间**下界大于0**，所以旧长期95%的[0,+6.5]不通过。代码见[配对统计](../whale_local/sao_bridge/analysis.py)，独立训练重复见[该分支端点](../runs/alfworld_explicit_hint_training_replication_v1/repeat/matched_COMPLETE.json)。

### A2. 必须分开的成本维度

| 实验/阶段 | 优化器消费的角色轨迹 | 完整生成/提案 | 训练环境episode | 另计的选择/主评/诊断 |
| --- | ---: | ---: | ---: | --- |
| 匹配GRPO Engineer | 192 | 192训练生成 | 合法候选依官方10任务奖励另计 | 8候选选择与W0/W1主评另计 |
| 固定W1的H3 | 192 | 198训练生成；16评测提案 | 1120个已完成奖励episode | 1176个选择/报告episode |
| 原H0/W0 MA整次 | 1840 | 1872完整角色轨迹 | 随实际联合outcome账本核算 | 初评、matched与best另计 |
| guarded方法包v2 | 1000，H496/W504 | H512+W512 | 512完整配对生成 | 1600诊断另计，旧失败128消费另计 |
| validity-credit两臂 | 1024 | 每臂H256+W256 | 两臂合计512 | 选择912、主评800、诊断1400，共3112 |
| unit-peer主臂 | 512 | H256+W256 | 256 | 复用对照、smoke及诊断另计 |
| 48轮混合调度一臂 | 640 | 768角色轨迹 | 384 | 按合法/去重策略实际评分量计算 |
| 旧长期4块新增 | 5120 | 6144角色轨迹 | 3072 | 每48轮8提案、selection48与两组200题另计 |
| 新每轮联合首96轮 | 1536 | 1536角色轨迹 | 768 | 两个指定端点另计；失败启动0episode |
| WHALE完整计划 | 无Engineer优化；Solver按成功调用做SFT | 640个Solver rollout、384个MH提案 | 640 Solver episode | MH最多3860开发episode，另有初始10题及主200 |
| WHALE实际时间截断结果 | 529个成功episode中的9972调用；159个Adam小步 | 640个Solver rollout、216个MH提案 | 640 Solver episode | MH1330、初始开发10、主评200；合计2180 |
| 97%交叉复测 | 0 | 0训练生成 | 0训练episode | 1600环境episode |

WHALE的640个episode不等于640个SFT调用：一次成功episode包含多次模型调用，全部成功调用才进入response-token目标。前64题的8-rank复制不能算成8倍新的数据；后续分片的占位forward也须单列。Actor warmup、prefetch、丢弃checkpoint、模型/策略选择、失败后的真实工作和预训练分别标注。特别是被拒绝W2的640条属于实际研究开销，不在“保留W1+h2名义1024”中自动消失。

本报告不编造一个覆盖所有历史工作的GPUh总和。已有账本包含部分失败成本下界、物理共享的共同前缀、只读复用及各自运行时间；简单相加各臂“含共同前缀”数字会重复计费。已公布的早期共同H1学习/选择窗口分别23.124939/12.825886 GPUh，合计35.950825 GPUh是每个逻辑臂继承、**物理只发生一次**的成本。其他域早期SFT/GRPO实验不并入ALFWorld性能表。

### A3. 重要失败、修复与未完成比较

| 事件 | 实际影响与处理 | 不允许的解释 |
| --- | --- | --- |
| 终端bootstrap、processed sampling log-prob、recurrent packing | 分别修正终局值、temperature处理、跨轨迹状态污染；历史与修复后结果区分 | 将跨版本全部差异归于一个算法 |
| DP ValueNorm不一致 | 改为同角色全DP active returns统计 | 声称原先各rank独立统计也等价 |
| 端口冲突、controller心跳饥饿、取消超时、HTTP keepalive | 独立保留失败，只恢复未完成工作，不重试不明确的非幂等动作 | 将transport失败记成任务失败0 |
| 联合短程OOM | 恢复共同已保存round4，后续验证释放unused cache | 使用未保存round6或称OOM为策略collapse |
| finite-budget dispatcher wrap | CPU复现24输入却提交32；本地限额后真实两轮GPU验证通过 | 把基础设施收尾修复当准确率提升 |
| ValueNorm继承计数审计缺陷、结果包装缺报告 | 修复审计/总结器，保留原始native结果与失败哈希 | 重跑选有利分数，或覆盖原失败 |
| 旧长期v1r3传输失败 | 30轮后失败；400消费、491完整角色回执、246完整环境结果保留；v1r4重启 | 从未保存状态“精确续训”或把失败成本归零 |
| WHALE v7状态分类错误 | 224完成episode/7批后失败；无MH、无最终200 | 称已完成同预算WHALE复现 |
| WHALE v8权限失败 | 原输入未被修改，v9复制同字节可读输入 | 将其计作一次负面任务性能试验 |
| WHALE v11多步数值门槛失败 | 原失败保留；另行预声明同权重全梯度/重复性门槛并通过 | 偷改旧阈值、宣称多步轨迹逐位等价 |
| WHALE v12批量调整 | 160题/Adam43完整状态继续，保留320/640交替边界但降低段内刷新频率 | 将学习超参数修订伪称纯运行时加速 |
| 新MA主评被CPU测试GPU上下文干扰 | 已承认本次隔离失误；0episode重试矩阵，未重训或重选 | 隐去故障或称训练导致collapse |
| 较早修正SAO/交替MA控制队列 | 未形成可用于本报告的完整匹配终点；新请求不再重复启动这些控制 | 把已准备配置当执行结果 |
| 新97%起点之前的93.5%全联合候选 | 用户改起点后，在训练前取消，0新增轨迹 | 把取消候选计作失败学习试验 |

失效尝试不是从结果集中删除：能恢复多少真实episode、token、梯度或墙钟就记录多少，不以一个“FAILED”将工作归零。反之，只有权重导出/服务启动/梯度有限也不能当成算法有效。

### A4. 复现入口与审计层次

每个实验先读其冻结protocol/DESIGN和对应source snapshot，再使用该snapshot的入口；**不要用当前main代码无条件重跑旧协议**，也不要直接覆盖既有run目录。模型、输入、patch、解码合同和原生报告SHA都在相应JSON中，完整checkpoint通常比Markdown摘要更权威。

| 内容 | 复现/核验入口 |
| --- | --- |
| 单角色H/W与终局处理 | [训练配置转换](../whale_local/sao_bridge/large_train.py)、[原生Actor包装](../whale_local/sao_bridge/terminal_actor.py) |
| H3一阶段与候选分布 | [固定Solver实验入口](../whale_local/sao_bridge/alfworld_fixed_solver_engineer.py)、[冻结协议](../runs/alfworld_fixed_w1_h3_v1/protocol.json) |
| 旧/新MA联合与长期审计 | [joint workflow](../whale_local/sao_bridge/joint_workflow.py)、[研究审计](../whale_local/sao_bridge/joint_research_audit.py) |
| 原生Adam/DCP恢复 | [恢复实现](../whale_local/sao_bridge/joint_continuation.py)、[新MA的恢复设计](../runs/alfworld_ma_simultaneous_from97_v1/ENDPOINT_RECOVERY_DESIGN.json) |
| WHALE原生训练与适配 | [主实现](../whale_local/sao_bridge/alfworld_whale_baseline.py)、[本轮冻结源](../runs/source-snapshots/alfworld-whale-sao-matched-v12) |
| 七小时窗口与MA后续衔接 | [WHALE准入](../runs/alfworld_whale_sao_matched_v12/run_validated.py)、[原始200/采样/Adam独立核验](../runs/alfworld_whale_sao_matched_v12/final_audit.py)、[MA续训协调器](../runs/alfworld_ma_simultaneous_from97_v1/resume_after_whale.py) |
| 配对统计与原始结果 | [统计实现](../whale_local/sao_bridge/analysis.py)、每节列出的COMPLETE/RESULT与原始report |

执行验证分三层：CPU合同/错误路径测试；真实原生长上下文与状态恢复验证；完整任务/模型/策略/seed/token/结果身份核验。三层都通过才支持“这个结果按所述方法产生”，仍不能替代独立研究泛化或多训练seed证据。

## 可用于汇报的结论措辞

**可以说：**

> 在本项目当前配置和已测试的追加预算下，SAO的两种固定一方续训都未可靠突破历史表现：Solver延长训练曾部分恢复，随后再次退化；Engineer再训练一阶段提高了候选合法性，但真正新策略主评为173/200=86.5%，没有可靠提高任务效果。调优后的MA-SAO曾达到97%，旧混合调度最终形成约93%–95%的观测开发平台，但尚未通过相对历史92%的严格优势门槛。从97%原生检查点改为每轮联合后，早期为93%和92%，未提升；block02已继续，尚非最终平台结论。本地WHALE时间截断运行得到163/200=81.5%，对历史SAO差-10.5pp，配对95%[-16.5,-4.5]；实际640个Solver episode、216个MH提案，没有完成640/384的完整同预算复现，不能作完整同预算的算法优劣结论。

**目前不能说：**

- SAO 与 GRPO 已被证明等效，或 SAO 普遍更好。
- Engineer 和 Solver 已经完全停止学习，或 SAO 不可能超过 92%。
- MA-SAO 已经稳定达到 97%/97.5%，或已证明同样本量优于 SAO。
- 两个 Critic 共享网络参数，或新的 1:1 调度才首次实现联合学习。
- 所有高分都来自 Engineer 自动学习，或 97% 是未见测试集上的确认性泛化结果。
- WHALE 已经完成 ALFWorld 复现或同预算测评。
