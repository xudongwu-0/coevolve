# MA-SAO 算法说明：从分阶段优化到 Engineer–Solver 协作训练

**版本：2026-10-09。** 本文只整理算法、相对普通 SAO 的改动及其实现边界，不启动训练。
2026-10-12补充了实际切分重叠与滚动恢复边界；当前可执行的离线证据复算入口见
[复现交接](REPRODUCE_MA_SAO.md)，该入口不运行训练或模型评测。
描述以当前机制研究的[冻结源码](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/)
和[v2 协议](../runs/alfworld_ma_mechanisms_single_seed_v2/PROTOCOL.md)为准。
历史实验的不同分支不能混成同一个算法配置。

> **核心思路：** 将“固定 Solver 训练 Engineer，再固定选中的 Harness 训练 Solver”
> 改为“同一个环境 episode 同时产生两方的真实动作轨迹，以同一个团队终局奖励学习”。
> 在此基础上，使用协作上下文 Critic、队友轨迹重要性校正、按角色 ValueNorm，
> 并以“两轮双方更新、一轮只更新 Solver”控制协同学习节奏。

这里的“改进”指相对本项目普通 SAO 路径的设计变化及预期作用，
**不等于已经证明每个组件都有正向效果，也不等于这些基础技术都是首次提出。**

## 1. 任务定义与角色

系统包含两个独立的 Actor 和两个独立的 Critic：

| 对象 | 职责 | 可训练参数 |
| --- | --- | --- |
| Engineer Actor | 根据任务证据生成 Harness/代码策略 | \(\theta_H\) |
| Solver Actor | 使用该 Harness 与环境交互，执行任务 | \(\theta_W\) |
| Engineer Critic | 估计 Engineer 自身行动轨迹的回报，额外使用协作上下文 | \(\phi_H\) |
| Solver Critic | 估计 Solver 自身行动轨迹的回报，额外使用协作上下文 | \(\phi_W\) |

需要区分 **Engineer 权重**与 **生成出来的 Harness**：
前者是产生策略的模型，后者是某次采样得到、实际部署给 Solver 的代码产物。
训练 Engineer 不是直接优化一个固定 Harness 的字符串。

记任务为 \(x_b\)，Engineer 证据输入为 \(e_b\)，生成的 Harness 为 \(h_b\)，
Solver 在环境中的整条轨迹为 \(\tau_{W,b}\)。团队目标可写为：

\[
J(\theta_H,\theta_W)
=\mathbb E_{x_b,e_b;\,h_b\sim\pi_H;\,\tau_{W,b}\sim P(\cdot\mid x_b,h_b,\pi_W)}
\left[R_b\right].
\]

ALFWorld 的环境终局奖励 \(R_b\in\{0,1\}\)。
两个 Actor 参数独立、没有跨模型的可微反向传播；
协作通过实际 Harness、环境交互、共享奖励和训练时统计实现。
实际优化还保留固定参考策略 KL 正则，不能把上式当作全部实现细节。

## 2. 相比普通 SAO，我们改变了什么？

本文的“普通 SAO”指本项目已有的本地 SAO/AReaL 分阶段实现，
不是声称对官方 SAO 或 Harness-R1 的所有配置作了穷尽复现。

| 维度 | 本项目普通分阶段 SAO | 当前 MA-SAO | 设计意图 |
| --- | --- | --- | --- |
| 训练组织 | 固定 Solver 训练 Engineer；选定 Harness 后再训练 Solver | 小轮次耦合，joint / joint / Solver-only | 缩短双方适应的间隔，同时保留 Solver 单独适应的轮次 |
| 训练数据 | 两个阶段分别采样；Engineer 评分时的 Solver 执行通常只供奖励计算 | 一个 episode 产生 H/W 各自的真实轨迹 | 让同一次协作的结果直接用于两方学习 |
| Engineer 奖励 | 固定 Solver 下，一个候选在 10 个任务上的平均奖励增量，相对无策略基准 | 同一个 episode 的终局奖励 \(R_b\) | 从固定队友下的局部改进转为当前团队的任务成功 |
| Solver 奖励 | 自己执行任务的终局奖励 | 与 Engineer 共享的同一个 \(R_b\) | 保持两方环境目标一致 |
| Critic 信息 | 主要根据本角色轨迹估值 | 两个独立 Critic，额外读取预动作协作信息 | 帮助价值估计条件化于队友和任务状态 |
| 策略分布校正 | 本角色的行为/proximal 概率校正 | 保留自身校正，再乘有界队友轨迹权重 | 缓解队友策略变化带来的采样分布偏移 |
| 数值尺度 | 保留原生 Actor 优势标准化，没有本套联合 Critic ValueNorm | 额外对各角色 Critic 的回报目标作 ValueNorm | 稳定价值学习的尺度 |

普通 Engineer 的奖励概念上是：

\[
r_H^{\mathrm{stage}}(h)
=\frac{1}{10}\sum_{x\in\mathcal T_b}
\left[R(x;h,\pi_W^{\mathrm{fixed}})
-R(x;\varnothing,\pi_W^{\mathrm{fixed}})\right].
\]

MA-SAO 的主配置则是：

\[
r_{H,b}=r_{W,b}=R_b.
\]

因此，我们不仅改变了“更新频率”，也改变了 Engineer 的奖励口径和两方的数据组织。
不能把 MA-SAO 相对普通 SAO 的所有差异都解释成某一个组件的作用。
不同历史分支的 PPO clipping、KL 系数和 Critic 更新频率也存在差异；
第 9 节列出的是当前实例化参数，不能声称它与所有旧普通 SAO 实验严格同超参数。
四项机制消融则在同一个新完整 MA 控制配置上逐项改变。

## 3. 一个 episode，两个真实动作轨迹

对每个配对样本，保存共同的 `sample_id / version / slot`：

1. Engineer 从该行的证据 prompt 生成 Harness，记录真实输出 token、行为 log-prob 和版本。
2. 验证并解析 Harness，交给本轮 Solver，在指定 ALFWorld 任务中实际执行。
3. Solver 的多轮生成调用拼成一条完整轨迹，保留其全部真实行动 token 和行为概率。
4. 环境给出终局 \(R_b\)，分别绑定到两方各自的轨迹。
5. 按本轮调度决定哪些轨迹进入参数更新；未更新的角色仍可能参与生成和配对。

这不是复制同一串 token 给两个模型，也不是从一个外部奖励数字“伪造”Solver 训练样本。
Engineer 的文本证据来自冻结的训练 packet；当前 Solver 的行为主要通过实际执行和奖励形成耦合，
**不是每轮都重新收集、改写全部 Engineer 输入证据。**

行为概率必须描述真实采样分布：当前生成使用 `processed_logprobs`，
尤其不能把 Engineer 温度 0.7 采样时的概率替换成温度变换前的原始概率。
这是保证后续重要性比率有意义的数据合同，不是额外的一种奖励或归一化机制。

### 无效策略与错误语义

主配置采用 `editor_reward_mode="team"`：

- 无效或运行接口不兼容的 Engineer 提案按明确合同作为空 Harness 执行，并记录模型错误。
- Solver 仍实际执行任务，双方仍获得真实团队奖励 \(R_b\)。
- 因而，即使提案无效，只要 Solver 独立完成任务，Engineer 也可能得到正奖励。
- 基础设施、通信或进程失败显式终止并留证，不伪装成正常的成功奖励。

这体现了共享团队目标，但也有信用分配局限：
**共享奖励不等于识别了 Harness 的因果贡献。**
当前主方法没有反事实差分奖励；此前的无效提案奖励门控是另一个实验变体，不应混入主算法。

实现：[配对 workflow](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_workflow.py)、
[角色奖励](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_reward.py)。

## 4. 两个独立 Critic 与预动作协作上下文

对角色 \(i\in\{H,W\}\)，价值估计写作：

\[
V_{\phi_i}(z_{i,t},c_i),
\]

其中 \(z_{i,t}\) 是本角色当时的轨迹状态，\(c_i\) 是仅用于 Critic 的协作信息。

| Critic | 额外上下文的主要内容 |
| --- | --- |
| Engineer Critic | Solver 任务的预动作信息、交互协议、共同轮次、允许的行为版本 lag、调度/路由信息 |
| Solver Critic | Engineer 的预动作证据、已生成提案的有效状态、Solver 任务信息、共同轮次和调度/路由信息 |

关键约束：

- 两个 Critic 不共享参数，也不是一个统一的中央价值网络。
- 额外前缀最多 1024 token，只对 Critic 展开，并从行动 loss mask 中排除。
- Actor 不新增这部分特权输入；训练和部署的 Actor 输入接口保持一致。
- 前缀不放入本 episode 的终局奖励、未来 Solver 轨迹或未来成功标签。
- Solver 开始行动时，Engineer 提案及其有效状态已经存在，因此它们不是 Solver 的未来信息。

这是**受集中训练、分散执行思路启发的价值估计设计**，不是完整 MAPPO 的逐项复现。
其目标是减少未建模的协作状态造成的估值混淆，实际收益仍需独立消融检验。

实现：[Critic 上下文展开与归一化](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_critic.py)。

## 5. 优势估计、队友校正与 Actor 更新

### 5.1 沿真实行动 token 计算 GAE

对本角色行动 token 序列，以 KL 正则化后的 token 奖励 \(r_{i,t}\) 计算：

\[
\delta_{i,t}=r_{i,t}+\gamma V_{\phi_i}(z_{i,t+1},c_i)
-V_{\phi_i}(z_{i,t},c_i),
\qquad
A_{i,t}=\delta_{i,t}+\gamma\lambda A_{i,t+1}.
\]

这里的相邻位置指沿有效行动位置回传；环境 observation、prompt 和 padding 不作为 Actor 行动。
当前原生路径使用 \(\gamma=1,\lambda=1\)，完整 episode 的终端 bootstrap 明确置零。
没有在这条实际路径上启用 length-adaptive lambda。
角色自己的 KL 奖励可能不同，因此“共享环境奖励”不意味着两方的全部 token 回报或优势相同。

然后分别对两个角色的有效 batch 优势作标准化：

\[
\bar A_{i,t}
=\frac{A_{i,t}-\mu_i^A}{\sigma_i^A+\varepsilon}.
\]

这是原生 Actor 优势标准化，不是下面的 Critic ValueNorm。
Critic 使用的 returns 在优势标准化之前形成，不用队友权重重写监督目标。

### 5.2 有界队友轨迹重要性校正

令 \(\mu_j\) 为队友采样时的行为策略，\(\pi_j^p\) 为训练端重新计算概率的 proximal 策略。
对同一个 episode 中队友 \(j\ne i\) 的全部行动 token：

\[
\ell_{j,b}
=\sum_{t\in\mathcal A_{j,b}}
\left[\log\pi_j^p(a_{j,t}\mid z_{j,t})
-\log\mu_j(a_{j,t}\mid z_{j,t})\right],
\]

\[
w_{j,b}
=\exp\!\left(\operatorname{clip}(\ell_{j,b},-\log 2,\log 2)\right)
\in[0.5,2].
\]

将角色 \(i\) 的整条轨迹优势乘以同一个队友权重：

\[
\widetilde A_{i,b,t}=w_{j,b}\bar A_{i,b,t}.
\]

**顺序是先计算并标准化本角色优势，再乘队友权重；之后不再标准化一次来抵消该权重。**
该权重作为固定系数，不反向传播到队友模型。
主调度的每第三轮使用 \(w_{j,b}=1\)，其余轮次使用上述有界校正。

设计动机是：两方都在变化，仅校正自己行为策略的陈旧性，不能覆盖队友造成的联合分布变化。
但该实现有明确边界：

- 它是整条行动轨迹的概率比求积后截断，不是逐 token 比率的算术平均。
- 它不减均值、不除标准差，也不保证 batch 内权重平均为 1，因此不应称为“伙伴归一化”。
- 截断引入偏差，且长轨迹会放大微小 log-prob 差异；不能宣传为严格无偏的联合策略梯度。
- 它不完整修正整个交互过程的所有状态分布变化，也不保证单调改进。

实现：[队友概率摘要与权重](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_updates.py)、
[优势加权位置](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/terminal_actor.py)。

### 5.3 保留本角色的原生概率校正

本角色仍保留 decoupled PPO 的两种比率：

\[
b_{i,t}=\frac{\pi_i^p(a_t\mid z_t)}{\mu_i(a_t\mid z_t)},
\qquad
q_{i,t}=\frac{\pi_{\theta_i}(a_t\mid z_t)}{\pi_i^p(a_t\mid z_t)}.
\]

前者校正本角色采样策略与 proximal 策略的差异，后者用于 PPO 参数更新约束。
当前配置以
\(m_{i,t}=\mathbf1[0.3\le b_{i,t}\le5]\)
剔除超出范围的行动 token，核心 Actor loss 为：

\[
L_i^{\mathrm{actor}}
=-\frac{1}{|\mathcal A_i|}
\sum_{t\in\mathcal A_i}
m_{i,t}b_{i,t}
\min\!\left(
q_{i,t}\widetilde A_{i,t},
\operatorname{clip}(q_{i,t},0.8,1.2)\widetilde A_{i,t}
\right).
\]

分母使用剔除前的有效行动 token 数；越界 token 是被 mask，不是夹到边界后继续贡献梯度。
自身校正和队友校正针对不同角色，不是对同一比率重复乘两次。
固定参考 KL 系数为 0.02，通过原生奖励/优势路径进入目标，不在上式中再次重复添加。

实际路径见[原生优势计算](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/third_party/sao/vendor/AReaL/areal/trainer/ppo/actor.py)、
[PPO 损失](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/third_party/sao/vendor/AReaL/areal/utils/functional/functional.py)。

## 6. 按角色 ValueNorm：稳定 Critic，而不是再次归一化队友

H/W 分别维护自己的回报统计 \((\mu_i^V,\sigma_i^V)\)。
Critic 对归一化目标学习：

\[
\widehat G_{i,t}=\frac{G_{i,t}-\mu_i^V}{\sigma_i^V},
\qquad
V_{i,t}^{\mathrm{raw}}=\sigma_i^V\widehat V_{\phi_i,t}+\mu_i^V.
\]

具体实现：

1. 使用固定版本的 MAPPO ValueNorm 参考实现维护运行统计。
2. 只收集有效行动位置的 returns，排除 prompt、padding 和 Critic 特权前缀。
3. 在**同一角色的数据并行 rank 之间**汇总目标，再更新一致统计；不把 H/W 混在一起统计。
4. Critic 在归一化坐标下使用原生 clipped MSE；计算 GAE 前先反归一化到原回报尺度。
5. 保存/恢复 ValueNorm 状态；首次校准前保留初始 raw-unit Critic 的尺度。

这不是 PopArt 的价值头参数重标定，也不是换成 MAPPO 默认 Huber loss。
关闭 ValueNorm 时必须使用相容的原尺度 Critic，不能直接把一个已归一化价值头当作原尺度模型。

### 与 Dr. MAS 的关系

[Dr. MAS 原文 Algorithm 1（B2）](https://arxiv.org/html/2602.08847v1)
按 agent 的有效参与样本计算奖励均值/标准差，构造
\(A_{b,i}^{\mathrm{DrMAS}}=(R_b-\mu_i^R)/(\sigma_i^R+\varepsilon)\)。
与本方法的关系必须分开表述：

| 机制 | 归一化/加权对象 | 与 Dr. MAS 的关系 |
| --- | --- | --- |
| 本角色 Actor 优势标准化 | Critic/GAE 产生的优势 | 最接近“按角色校准梯度尺度”的思路，但不是直接使用同一个奖励归一化公式 |
| 按角色 ValueNorm | Critic 的回报目标和价值尺度 | 稳定化方向相关，具体实现来自 MAPPO，作用位置不同 |
| 队友重要性校正 | 队友轨迹的策略概率比 | 不是均值/方差归一化，针对采样分布偏移 |

若双方使用相同 episode、共享奖励且每个 episode 等权，奖励均值/方差本来就相同。
两角色的 GAE、行动长度、token 权重和更新轮次仍可能不同，但不能直接照搬 Dr. MAS 的动机
来宣称我们这里必然需要不同的奖励统计。
**“去 ValueNorm”消融仍保留 Actor 优势标准化，并不等于“去掉 Dr. MAS 式归一化”。**

## 7. 更新节奏：joint / joint / Solver-only

主配置采用三轮一个周期：

| 周期位置 | Engineer 生成提案 | Solver 执行环境 | Engineer 参数更新 | Solver 参数更新 | 有效队友权重 |
| --- | --- | --- | --- | --- | --- |
| 第 1 轮 | 是 | 是 | 激活 | 激活 | bounded |
| 第 2 轮 | 是 | 是 | 激活 | 激活 | bounded |
| 第 3 轮 | 是 | 是 | 不更新 Actor/Critic | 激活 | unit，即 1 |

“激活”不意味着预热期间也更新 Actor：每个角色自己的前 3 个激活轮次只更新 Critic。
之后，每个激活轮次更新 Actor 一次，并在同一批 returns 上更新 Critic 两次。
原生更新顺序是 Actor 在前、Critic 在后；Critic 的两次更新不是两次新 rollout。
优化器在同一训练尝试的各轮之间持续保留，不按三轮周期重置。

第三轮重复本周期某一批任务输入，交替选择前两批之一，但重新生成提案和环境轨迹，
**不是回放上一轮的旧轨迹**。主配置为 `anchor_mode="sampled_control"`：
虽然代码中的调度名含 `anchor`，第三轮仍使用当前 Engineer 新生成的提案，
不是固定使用 H0 或历史 Harness。

其设计意图是给 Solver 更多适应机会，同时限制 Engineer 更新频率；
这只是待验证的稳定化假设，不是已证明最优的 2:3 比例。
允许最大一轮行为版本滞后；共同采样版本不等于 Actor 实际更新次数。
“共同更新”也不要求两个模型的反向传播发生在同一物理时刻。

实现：[输入重复与路由](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_anchor.py)、
[角色更新调度](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/joint_schedule.py)。

## 8. 完整算法流程

以下是机制级伪代码，省略分布式调度和进程管理：

```text
初始化独立 H/W Actors、各自固定参考策略和两个独立 Critics
初始化按角色 ValueNorm；准备固定有序任务输入和可审计的随机种子

for collection round r:
    按 joint / joint / Solver-only 确定 H/W 是否激活
    取本轮 8 个输入；第三轮重复输入但重新采样

    对每个 slot:
        H 生成真实提案轨迹；解析并记录有效性
        W 使用该提案（无效时为空策略）执行真实任务
        为 H/W 各自轨迹记录同一个终局 R、行为概率、版本和配对身份
        构造不含终局结果的 Critic-only 协作上下文

    两角色计算并交换同轮/同槽 proximal 概率摘要
    校验行为版本滞后不超过一轮，校验 action mask 与真实输出 token 一致

    对本轮被激活的角色 i:
        Critic 用协作上下文估值，并将 ValueNorm 输出恢复到原回报尺度
        在完整终局轨迹上计算 returns、GAE 和本角色 batch 标准化优势
        优势乘有界队友权重；第三轮改用单位权重
        若已完成本角色 3 个激活轮次的 Critic 预热:
            保留自身行为校正，执行一次 clipped PPO Actor 更新
        对同一批 returns:
            更新/同步本角色 ValueNorm，执行 Critic 更新，共两次

    未激活角色保留必要的配对/概率握手，但不执行参数更新
    发布采样版本，保存指标和真实轨迹回执；按预声明位置导出检查点
```

## 9. 当前实例化配置与精确预算

算法本身与初始化谱系应分开。当前四机制研究共同使用 H0 Engineer Actor、
普通 SAO 的 W1 Solver Actor、前 MA 的普通 H1/W1 Critics，fresh Adam，
不继承历史 97% 点的 MA Critic/ValueNorm。

| 项目 | 当前固定设置 |
| --- | --- |
| 训练种子 | 1000043，单 seed |
| 输入任务 | 原训练任务 0–239；相同的 1152 条有序训练输入 |
| 主端点 | 144 collection rounds；96 为诊断端点 |
| batch | 每轮 8 个配对 episode |
| 上下文/生成预算 | 32768 context；每角色每 episode 最多 4096 action tokens |
| H/W 采样温度 | 0.7 / 1.0 |
| Actor / Critic 学习率 | \(5\times10^{-7}\) / \(2.5\times10^{-6}\) |
| Critic 更新频率 | 每激活轮次 2 次；每角色先预热 3 个激活轮次 |
| Actor PPO clip / 参考 KL | 对称 0.2 / 0.02 |
| 行为版本滞后上限 | 1 round |
| 保存与选择 | 96/144各生成8个候选，在selection48上选择；当前滚动原生恢复每12轮保存 |
| 报告评测 | 选择锁定后报告原200题；selection48全部包含在这200题中，不是独立测试集 |

| 144 轮训练预算 | 完整 MA / 三个机制移除臂 | 每轮共同更新臂 |
| --- | ---: | ---: |
| 共享环境 episodes | 1152 | 1152 |
| 生成的角色轨迹总数 | 2304 | 2304 |
| H/W 实际消费轨迹 | 768 / 1152 | 1152 / 1152 |
| H/W Actor 更新 | 93 / 141 | 141 / 141 |
| H/W Critic 更新 | 192 / 288 | 288 / 288 |

每轮共同更新臂的总消费轨迹由 1920 增为 2304，多 20%；
不能因此声称同优化器工作量、同 token、同 FLOPs 或同墙钟时间。
预热、校准、上游初始化、候选选择、评测和中断尝试也必须分别计费。

原始main200矩阵本身不用于事后选择checkpoint或重抽候选，但这不意味着题目
未参与选择：selection48与main200重叠48题。训练0–239与main200无任务ID重叠；
strict197排除291、414、467的goal-group重叠，仍保留选择集重叠。
因此主结果是复用开发集上的固定终点评分，不能表述为独立未触碰测试结果。

依据：[实际 H 配置](../runs/alfworld_ma_mechanisms_single_seed_v2/full/editor_spec.json)、
[实际 W 配置](../runs/alfworld_ma_mechanisms_single_seed_v2/full/solver_spec.json)、
[预算与评测实现](../runs/source-snapshots/alfworld-ma-mechanisms-single-seed-v2/whale_local/sao_bridge/alfworld_ma_mechanism_study.py)。

## 10. 四个关键机制的消融口径

| 消融 | 唯一计划干预 | 保持不变的关键内容 |
| --- | --- | --- |
| 去队友校正 | 队友权重全设为 1 | 自身概率校正、Critic 上下文、ValueNorm、主调度 |
| 去协作上下文 | 关闭 Critic 的额外前缀 | 独立 Critics、Actor 输入、队友校正、ValueNorm |
| 去 ValueNorm | Critic 在原回报尺度训练 | Actor 优势标准化、协作上下文、队友校正 |
| 每轮共同更新 | 第三轮也激活 Engineer 参数更新 | 输入、真实提案路由、共享奖励、第三轮单位队友权重 |

旧的“从 97% 出发每轮双方更新”实验同时改变了第三轮队友校正，
不能替代当前只改变更新开关的调度消融。
本轮不再额外训练分阶段 SAO 对照；共享 episode/团队奖励属于共同基础设计，
这四个消融不会单独分离它们各自的因果贡献。

## 11. 哪些是已有技术，哪些是我们的适配？

**不应单独包装成原创：**

- PPO/GAE、自身 off-policy 概率校正、独立 Actor–Critic。
- 集中训练、分散执行的思想，以及训练期的协作价值信息。
- MAPPO ValueNorm、按角色维护尺度、Actor 优势标准化。
- Dr. MAS 已明确研究的 agent-wise advantage normalization 思路。

**我们实际完成的算法适配与组合：**

1. 将 Harness 生成与 Solver 执行组织成可配对的两条真实行为轨迹，并共享环境终局奖励。
2. 为两个独立 Critic 定义角色相关的预动作协作信息，而不向 Actor 注入该额外前缀。
3. 在本角色原生校正之外，接入有界队友轨迹校正，并明确处理异步版本和第三轮例外。
4. 结合角色级价值尺度和 2:3 更新节奏，形成可逐项关闭、可审计的共同训练协议。

这些可以准确表述为“本项目对 Engineer–Solver 共同训练的设计和实现”，
但其相对整个文献的首次性、各组件必要性和稳定收益，仍需文献定位及实验验证。
我们没有实现完整反事实信用分配，也没有得到单调改进或收敛保证。

### 与系统加速的区别

已验证的 FLA/Triton 路径、BF16 固定参考存储与保持 FP32 learner masters，
属于训练系统优化，不改变共享奖励或增加一种新 RL 目标。
既有测量的 warm-round 均值由约 96.90 秒降至 49.56 秒，约 1.96 倍；
这是特定配置下的已有证据，不保证所有消融具有相同速度。
改变生成结果的更快采样布局没有被当作无损优化采用。
见[速度策略及证据](../runs/alfworld_ma_mechanisms_single_seed_v2/SPEED_POLICY.json)。

## 12. 当前能支持的结论与论文表述

历史保存系统曾达到 194/200，即 97%，但它使用了已经训练过的 MA64 Critics/ValueNorm、
48 轮边界的 Adam 重启、不同候选机会和开发集峰值选择。
不能将这个分数直接归因为上述任何单一机制，也不能把新共同初始化控制当作该历史路线的精确复现。

截至本文整理时，完整/每轮共同更新的两项原生六轮检查已通过；
它们验证实际更新次数和关键数据合同，不是机制性能消融。
新完整 MA 已于2026-10-09 14:22 UTC通过144轮训练审计，
2026-10-10恢复剩余评测后，预声明144轮主终点为186/200（93.0%），
同一Solver在空Harness下为179/200（89.5%）；96轮诊断为90.0%和89.0%。
恢复复用全部82条已完成的筛选结果，不重训、不补抽候选、不重复完成题。
最终CPU报告误读历史紧凑摘要的问题已通过解析原始、逐字段匹配的历史轨迹解决，
不影响已保存的主终点分数，也没有把缺失消息虚构为真实行为。
新93%没有复现历史97%；历史初始化、优化器边界和选择机会不同，不能作组件因果对照。
四项正式消融尚未完成，不能把单个完整MA分数或训练步数当成机制必要性的结论。
见[固定终点结果](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/RESULT.md)
及[历史路径审计](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/BASELINE_PATH_AUDIT.json)。
用户2026-10-09 08:34 UTC要求再次暂停，至少两小时后才尝试恢复；每12轮完整保存不变。
暂停时首次12轮保存尚未配对提交；延迟到期后重新完成校验，
11:16 UTC实际通过了双节点12轮保存/停止/原生恢复至15轮，无Adam重置或重复预热。
这不是144轮正式训练结果，也不能用作机制性能消融证据。
两小时冷却和周期恢复都是执行层改动，不增加RL目标。原生恢复校验及实时执行状态见项目状态，
不能将代码测试或恢复校验当作正式机制消融结果。

2026-10-10首项without_peer在24轮保存后暴露共享文件系统跨节点flock不互斥的问题。
修正为双方各自保存状态、仅Solver发布共享配对元数据，Engineer等待并验证最新配对；
这不等于“只更新Solver”，也不改变2:3学习调度。修复通过并发回调及真实双节点CPU发布测试，
实际24轮原生状态有效，按两小时策略安排恢复。该执行层修复和24轮进度都不是组件必要性证据。

可用于方法部分的谨慎总结：

> 我们将 Harness–Solver 的长阶段交替优化重组为共享终局反馈的配对多智能体训练。
> 每个环境 episode 为 Engineer 与 Solver 提供各自真实的行为轨迹，
> 并以独立的协作上下文 Critic、保留自身校正的有界队友轨迹加权、
> 按角色价值归一化和非对称更新调度支持联合学习。
> 这些设计分别针对反馈脱节、协作状态未建模、策略陈旧性与价值尺度问题；
> 它们的实际收益和必要性通过预声明、共同初始化的消融进行检验，而不是由机制描述或历史峰值直接推定。

历史结果与原生优化路径的更完整背景见[实验报告](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md)；
实时执行状态以[项目状态](./PROJECT_STATE.md)为准。
