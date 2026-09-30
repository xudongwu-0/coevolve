# 项目状态与 Agent 接手指南

更新：2026-09-30 UTC。这是当前研究主线的入口；历史细节见
[实验日志](EXPERIMENT_LOG.md)，算法定义见 [MA-SAO](MA_SAO.md)、
[Harness-R1 SearchQA适配](../whale_local/harness_r1_searchqa/README.md)和
[ALFWorld复现入口](../whale_local/harness_r1_alfworld/README.md)。

## 1. 目标与当前结论

目标不是只提高解题模型的权重，而是训练一个 Engineer，使其能够根据真实
失败轨迹提出可执行、可采纳、能在未见任务上提高表现的 harness 修改。
先验证 Harness-R1 的官方方法，再在相同起点、奖励和预算下比较 SAO / MA-SAO。
最终希望小模型（优先4B）具有这种能力。

**当前还没有证明稳定、可泛化的 RL harness 改写收益。**

- 旧 SAO / MA-SAO 实验的主要收益来自 Solver 权重；多次交替实验始终保留 h0。
- 32条数据的4B Harness-R1小试未通过输出/执行门槛，不能评价有效RL学习。
- 官方9B Engineer在ALFWorld能够生成并实际应用修改，解决了执行能力的校准问题。
- 官方冷启动SFT Engineer两次小样本提案分别得到4/8与5/8；存在一个正收益实例，
  但还不是稳定收益、未见任务收益或RL优于SFT的证据。
- Agent-SFT Solver得到7/8是Solver收益，绝不能算作harness改写收益。

不要把“启动成功”“参数发生变化”“合法patch”“训练reward为正”
或“GPU显示100%”替代最终实验目标。

## 2. 模型角色与算法边界

| 状态 | 职责 | 必须区分 |
|---|---|---|
| Solver / W | 解题、调用环境工具 | 权重更新收益与harness收益分开 |
| Engineer / E | 读取失败packet，生成patch | 合法输出、实际执行、正收益、泛化四个门槛 |
| Harness / H | 模型周围的提示、工具和控制逻辑 | 修改后的代码是否真正被采纳，不能用h0收益冒充 |

### Harness-R1

官方代码固定在 `94f2e087f573e1b82fc9568bb634d4ae9a887e28`。
训练流程为冻结Solver、缓存原始任务轨迹、提取失败packet、冷启动SFT、
在线采样patch、同任务重跑、用结果差值训练Engineer。

- 官方SFT：Qwen3.5-9B，877条teacher-filtered样本，2 epochs；
  样本须可执行、完成重跑且非退步，并非必须严格正收益。
- 官方GRPO：每packet 8个候选，每更新4个packet；组内reward归一化、
  非对称PPO clip、TIS cap 2；不加格式奖励或显式KL。
- Reward为完整任务批次的平均收益变化，包括原本成功的任务，因而会惩罚回归。
- Patch是四类可执行hook：`on_init`、`make_pre_hint`、
  `on_before_action`、`on_post_step`；不是任意重写全部框架。
- 论文主训练是独立批次的固定基线编辑，不跨批次积累patch记忆；
  我们持续采纳、累积H的设计是额外扩展，不是已经复现的论文结论。
- SearchQA训练直接调用SHA固定的官方reward normalization和PPO/KL函数；
  FSDP/vLLM执行后端是本地适配，不是完整Megatron/SGLang复现。

### SAO / MA-SAO

SAO基线复用固定的**非官方**SAO/AReaL开源实现，不是自行替换PPO目标。
MA-SAO的主要机制是：

1. 保留Engineer奖励评估中产生的真实Solver candidate/reference轨迹；
2. 在下一Solver阶段分为不相交的critic warmup和actor/critic混合训练池；
3. 保留原始token、behavior log-probabilities与版本，使用原生DIS校正；
4. Solver改变后，重新评估旧Engineer动作及其原incumbent，刷新奖励。

这不是已证明的MAPPO，也不是对Solver更新求梯度。领先版本没有使用已验证的
集中式critic；CTDE-style critic、跨角色critic迁移和interaction bonus是实验分支。
相同生成样本上限不等于相同optimizer工作量或GPU时间。

## 3. 主要实测结果

所有数字须连同模型、数据口径、预算、阶段、harness身份和不确定性解释。
不同benchmark/判分/模型的百分比不能放在一起直接排名。

### 3.1 历史小数学试验

本地MATH number-theory 128题、每题一次采样：训练内部2B改写器最终93/128，
固定内部改写器84/128，外部改写器92/128。
这是候选再生成频率的初步信号，历史题已使用且实际搜索成本不等；
不能凭一题差异宣称优于外部模型或稳定自我改进。
详见[历史实验说明](EXPERIMENTS.md)与根目录[README](../README.md)。

### 3.2 SAO / MA-SAO双域探索

| 实验 | 实测 | 解释 |
|---|---|---|
| Chess共同起点 | 原始4B 34/2048=1.6602%；SFT1152 31/2048=1.5137% | 选择原始模型，不是显著SFT效应 |
| Chess首个H-W-H后 | SAO 423/2048=20.6543%；MA-SAO 217/2048=10.5957% | 两者H仍为h0；MA−SAO −10.0586pp，题级区间[-14.0137,-6.3477]pp |
| SearchQA共同起点 | SFT1152 362/2048=17.6758% | 本地BM25/alias EM，不是WHALE原判分 |
| SearchQA第四块 | SAO 549/2048=26.8066%；MA-SAO 600/2048=29.2969% | 两者仍为h0；+2.4902pp区间[-0.3906,+5.3223]pp跨零 |

Chess训练还出现同seed、匹配配置下的大幅采样/格式变异；题级bootstrap不能
覆盖训练seed变异。因此不能据这些单seed、不同compute结果宣称MA-SAO普遍更强。
SearchQA harness-only曾保留改变步数/搜索预算的literal配置，但不等于在固定预算下
学会有效可执行hook改写。旧权重、算法、分析保留；旧实验未继续扩展。

### 3.3 本地4B Harness-R1快速训练

Engineer是 **Qwen3-4B-Instruct-2507**，不是Qwen3.5-4B。
Teacher是fresh Qwen3.5-9B；Solver冻结为SearchQA MA-SAO W4。

- 修正9B协议后，728条生成中102合法、95 clean、32 clean-positive。
- 从其中32条唯一程序建立小SFT集，真实hook执行568次，无错误。
- 4B full SFT：1 epoch、4次optimizer更新；这是小试，不是官方877条/2 epoch。
- 实际在线GRPO：1个更新bundle，每actor rank 4步；32个训练候选全部非法，
  四组reward全零，policy loss和policy gradient均为零。
- 匹配raw/SFT/RL三个对照，各32个提案，均0合法；没有候选dev/test结果。
- 实际GRPO为11:33:36至11:52:31 UTC，18分55秒。
  **没有合法patch重跑成本，不能把它外推为有效RL的完整训练速度。**

结论是这份小数据、协议与基础模型组合没有准备好，不是4B或Harness-R1能力上限。
不追加零信号RL，不靠放宽parser、手写patch或validity bonus制造成功。
服务器侧证据：`runs/searchqa_harness_r1_20260930/fast4b/`（不进入Git）。

### 3.4 官方9B checkpoint：ALFWorld首轮

模型仓库 `ShaoShuai0605/Harness-R1` 固定revision：
`f6a915953f8206f3f5c7ef0b40240e1f4d2b6676`。

| Checkpoint | 角色 |
|---|---|
| `cold-start-sft` | SFT-only Engineer，官方RL初始化 |
| `harness-r1` | 针对原始Solver训练的RL Engineer |
| `agent-sft-harness-r1` | 针对Agent-SFT Solver训练的Engineer，当前未下载/测评 |
| `target-agent-sft` | Agent-SFT Solver，不是Engineer |

三个Engineer不是三个benchmark分别一个模型，而是混合benchmark训练并按训练阶段、
目标Solver区分。三个训练环境是ALFWorld/WebShop/DBBench，与WHALE主实验的
SearchQA/数学/Chess不重叠；跨未见Solver的泛化不等于跨任意新runtime的泛化。

当前只测官方500题split中确定性的8题、覆盖六个任务类型，
IDs为 `0,1,2,3,8,17,27,29`。失败packet来自同一批题，
所以是**transductive diagnostic，不是未见测试集复现**。
原始Qwen3.5-9B Solver冻结，temperature 0、非thinking、50步、
4096 target tokens；Engineer temperature 0、12288 tokens、严格think/patch协议。

| 条件/独立提案 | 成功数 | 净变化 | 证据解释 |
|---|---:|---:|---|
| 原始Solver、默认H | 4/8 | 基线 | 无patch |
| 官方RL Engineer的一次patch | 4/8 | 0 | 真实修改提示和7条动作序列，非no-op |
| Cold-start SFT提案A | 4/8 | 0 | 救回task8，回归task27；25条匹配的生成提示 |
| Cold-start SFT提案B（并发子任务） | 5/8 | +1题/+12.5pp | 救回task8，无回归；独立patch，不是A的基础设施重跑 |
| Agent-SFT Solver、默认H | 7/8 | +3题/+37.5pp | Solver权重收益；四题救回、一题回归 |

**必须同时保留SFT A/B，不挑最好一条汇报。**早期主汇总仅包含A；
读取并发子任务完整结果后补入B。两个不同SFT提案和一个RL提案不构成
同提案预算算法比较，也不是独立training seeds。A因/tmp失败的恢复重跑
保留了同一个patch，不算第二个提案。

两种Engineer都通过可执行编辑门槛；SFT B提供一个小正例，但没有稳定或
独立held-out净收益证据，也没有RL优于SFT证据。官方9B权重可作9B继续RL起点，
不能直接初始化4B；4B仍需要独立初始化或蒸馏。

论文的SFT-only Engineer报告：ALFWorld 40.6%→39.4%，WebShop 31.2%→38.6%，
DBBench 61.0%→61.3%，平均44.3%→46.4%。平均收益不代表ALFWorld收益。
这些是论文数字，不能与本地8题结果直接比较。

服务器侧证据（均不进入Git）：

- 原始基线/RL：`assets/alfworld_harness_r1/runs/diagnostic8_released9b_20260930_1406/`。
- SFT A：`assets/alfworld_harness_r1/runs/diagnostic8_coldstart9b_20260930_1413_r2/`。
- SFT B：`runs/harness_r1_alfworld_20260930/node3/coldstart_samepacket8_20260930_1415/`。
- Agent-SFT：`assets/alfworld_harness_r1/runs/diagnostic8_targetagentsft_20260930_1414_r2/`
  及独立核验 `runs/harness_r1_alfworld_20260930/node3/solver_sft8_20260930_1418/`。

### 3.5 ALFWorld双节点SAO/MA-SAO起点

正式协议固定在
`examples/alfworld_sao_masao_campaign_20260930_v6.json`。两臂共同从官方RL
Engineer `harness-r1`和Agent-SFT Solver `target-agent-sft`开始；node4为
顺序SAO（H收敛后W），node3为H/W交替MA-SAO。官方500题mixed列表在本地固定为
train 0–239、reward 240–299、dev 300–399、sealed test 400–499。
这是本地研究划分，不是论文原始untouched test。

0–239已完成六轮2×2匹配矩阵，共96 jobs、全部首尝试成功：

| Engineer + Solver | baseline | 同packet patched | 净变化 |
|---|---:|---:|---:|
| RL + Agent-SFT（正式共同起点） | 166/240 | 184/240 | **+18/+7.50pp** |
| Cold-SFT + Agent-SFT | 167/240 | 157/240 | −10/−4.17pp |
| RL + Original | 93/240 | 132/240 | **+39/+16.25pp** |
| Cold-SFT + Original | 97/240 | 103/240 | +6/+2.50pp |

共同起点有23个合法patch、1个非法patch，救回27题、回归9题。它证明官方
reward非零且有优化信号，但Engineer看过同十题失败，因此只能称
**train packet transductive diagnostic**。Agent-SFT在两种Engineer并发配对下
仅task134出现baseline不一致；Original有10题不一致，均已在task-level receipt保留。

两节点16卡完成1,750个真实target episodes与96个Engineer proposals，
六轮合计1,432.49秒、约1.22 target episodes/s。node3/node4全时段平均GPU利用率
分别28.06%/29.93%；Solver约30–50%，Engineer约11–24%。这是“十题baseline+
patched围绕一次Engineer生成”的采集结构，不是optimizer阶段上限。正式H布局为
3 actor FSDP+1 rollout+4 frozen Solver，W为4 actor FSDP+4 rollout。

正式更新前的真实optimizer smoke已经通过。Qwen3.5原先被AReaL按VLM
language-model分支加载，critic输出`[batch,seq,248320]`词表logits；
专用token-value head修复后严格输出`[batch,seq]`并通过save/reload测试。
H的2-way FSDP随后在一条约18k-token轨迹的critic backward OOM；3-way
FSDP完成1次actor和2次critic更新，actor/critic grad norm分别9.37/226.21，
actor权重SHA256从发布权重`7d16d5...`变为`cb06bf...`，峰值仍有约23 GiB
余量。正式H训练另用20k-token microbatch上限避免同rank打包两条长轨迹。

W smoke完成1次actor和2次critic更新：139条action trajectories、
1,386,174 tokens，actor/critic grad norm为1.06/348.28。总耗时
2,713秒；其中actor更新10分32秒、两次critic更新21分47秒。
旧连续task顺序使并发首批都落到同一个单会话packet worker，产生3,870次
HTTP 406工作流失败；数据现按24个packet assignment round-robin，运行前还会
审计整个异步active window，碰撞直接失败而不是进入训练。旧smoke只用于通路
与计时，不作为正式checkpoint。H训练生成上限4096（23个历史合法提案均
≤3014 tokens，唯一12288-token样本是非法runaway），正式评测仍保留12288。
首个正式H1 rollout随后暴露提示/验证器契约缺口：提示要求多hook，却未声明
每个hook code最多5个helper和1200 AST nodes；node4前4个候选全因此无reward
被拒，node3尚未产生候选。两臂均在`metrics=0/checkpoints=0`时按精确unit停止，
原attempt不计正式更新。训练提示现只补充已有验证器限制，任务、reward、seed、
模型、优化器均不变。

H1-r2证明修正后的契约有效：node3首批4/4候选合法，packet reward分别
`+0.2/+0.1/0/-0.1`。但它也在任何critic forward/update前证明AReaL的3-way
actor DP不仅要求`batch>=3`，还要求batch可被3整除；v6的batch4触发
`balanced_greedy_partition`异常。与此同时，node4因共享AgentBench
`configs/generated`由node3 UID1009以0775创建，而node4 UID为1008，官方评测
在模型判分前触发`PermissionError`并返回HTTP502。两臂r2仍为
`metrics=0/checkpoints=0/optimizer updates=0`。protocol v7保持全部数据split、
E0/W0、seed、reward和优化器不变，只把H轮次改为`2 steps × batch6`并新增
batch/DP整除预检；每个H reward与W环境现在使用run-local可写AgentBench布局和
scratch，只把固定upstream源码与ALFWorld数据作为只读symlink目标。v7 manifest
SHA256为`fa6c2c98315fcfba2a65edd48dfb17ff0c7f4380fea1f35cfa35e57188680eaa`。
两臂H1-r3随后都完成了合法的run-local官方评测并进入critic value forward，
证明UID隔离与batch6 dispatch修复有效；但20k microbatch小于实际
20,309–21,139-token轨迹，AReaL在`compute_logp`前显式拒绝。r3两臂仍为
`metrics=0/checkpoints=0/optimizer updates=0`。对全部64行使用训练时相同chat
template的审计得到prompt最大18,606，叠加4096生成上限后单轨迹最坏22,702。
protocol v8把H microbatch冻结为23,040（可容纳任一单轨迹但不能打包两条长轨迹），
并在停止filler/领取GPU前持久化capacity audit。v8 manifest SHA256为
`d952ef63f49f83dbb2c1bc730115cdcbfc870043e30253984e3541bed2d5a643`；
两臂已从原始E0/W0同步启动H1-r4。

正式MA分支不是虚构完整MAPPO：采用去中心actor、pre-action joint-state critic、
W版本隔离reward cache、W变化后的H reward refresh，以及25%真实Engineer
behavior-token/logprob replay+interaction credit。官方ALFWorld reward没有提供精确
Solver behavior logprob，所以Solver cross-role replay明确关闭。最终同时报告：
packet-transductive、Solver no-patch、reward区选择后原样用于未见题的global harness。

关键索引：

- `runs/alfworld_sao_masao_20260930/matrix_train_000_239.json`
- `runs/alfworld_sao_masao_20260930/matrix_infra_report.json`
- `assets/alfworld_sao_masao_20260930/protocol_v3/manifest.json`
- `assets/alfworld_sao_masao_20260930/protocol_v6/manifest.json`
- `assets/alfworld_sao_masao_20260930/protocol_v7/manifest.json`
- `assets/alfworld_sao_masao_20260930/protocol_v8/manifest.json`
- `assets/alfworld_sao_masao_20260930/data/manifest.json`
- `whale_local/harness_r1_alfworld/README.md`

## 4. 性能优化：测量而不是利用率印象

| 实际小工作负载测量 | 结果 |
|---|---|
| SearchQA Solver 8卡→4卡 | 2022.98→2027.74 tokens/s，吞吐基本不变 |
| 活跃Solver卡平均利用率 | 33.68%→65.68%；释放4卡给Engineer |
| Shared dev baseline复用 | 140.04→74.32秒，1.884倍；首次baseline另需71.68秒 |
| SFT loss chunk128→1024 | 端到端261.46→255.99秒，约2.09%缩短 |
| ALFWorld双节点矩阵 | 1,750 target episodes/1,432.49秒；96/96 jobs首尝试成功 |
| H-SAO 3-way FSDP smoke | 1 actor+2 critic更新；765.73秒；峰值约116.7/139.8 GiB |
| W-SAO单更新smoke | 139 action trajectories/1.386M tokens；2,713秒 |

以上是单次小测，不是复现稳定的全规模加速率。SFT平均利用率没有提高；
盲目增大并发反而减速。NCCL等待可能显示GPU100%，不能冒充有效计算。
ALFWorld模型/环境就绪后的首轮8题基线+RL patch重跑约几分钟；
环境安装、下载、SHA校验、模型初始化必须另计。每阶段报告目的与更新后的ETA。

## 5. 困难与已验证修复

| 问题 | 根因/处理 | 以后必须做 |
|---|---|---|
| 4B格式失败、零RL信号 | cold-start/协议未准备好 | 未见packet可执行生成先过关，再启动RL |
| 独立vLLM rollout卡TCPStore | 继承torchrun的`TORCHELASTIC_USE_AGENT_STORE` | 子进程清除rank/master及全部`TORCHELASTIC_*`；actor环境不变 |
| Health通过，生成HTTP500 | FlashInfer GDN JIT需要缺失的nvcc | 显式Triton backend；做真实请求/工具调用probe |
| CPU episode全task error却进程exit0 | FastDownward把库复制到满盘/tmp | 独立TMPDIR放/dev/shm或3FS；审计overall.validation，不接受伪0分 |
| ALFWorld sparse ID丢失 | 官方builder用连续range推断size | 完整baseline校验后添加真实IDs；同任务身份配对 |
| ALFWorld packet metadata HTTP422 | Arrow把不同task-id字典并成含None的稀疏struct | optimizer数据把metadata存为稳定JSON列，workflow边界严格解码 |
| node3 Git dubious ownership | 两节点UID不同、共享submodule所有者不同 | 每个固定仓库命令显式使用对应`safe.directory`，不修改全局Git配置 |
| Python venv逃逸 | 官方脚本resolve了symlink解释器 | 用copied interpreter，不把系统python当已装环境 |
| Packed-SDPA跨序列污染 | 上游SDPA忽略packed边界 | 受控block-diagonal mask补丁；旧受污染实验无效 |
| Qwen3.5 critic变成词表logits | VLM分支无条件加载生成模型 | 专用token-value head；shape guard与save/reload回归 |
| H 2-way FSDP OOM | 单条约18k token不能再切；critic backward仅余82 MiB | 3-way actor FSDP+1 rollout；20k microbatch防止双长序列打包 |
| H batch4无法3-way dispatch | AReaL balanced partition要求consumer batch可被DP整除 | protocol v7 H batch6；领取GPU前显式整除校验 |
| H单轨迹超过20k microbatch | 4096输出下实际轨迹为20.3k–21.1k；旧512-token smoke未覆盖 | 全64行模板审计上界22,702；v8容量23,040并在GPU分配前校验 |
| 双节点AgentBench写权限冲突 | 同名用户在node3/node4分别为UID1009/1008，共享generated目录为0775 | 每个H/W run建立独立可写runtime/scratch；固定源码和数据只读链接 |
| W并发出现HTTP 406 | 连续task IDs共用同一单会话packet worker | 跨24 workers round-robin；启动前审计完整异步active window |
| AReaL导出缺preprocessor metadata | 权重checkpoint不一定复制全部Qwen runtime文件 | 从输入checkpoint补齐并单独记录；权重变化使用纯safetensors digest |
| H训练提示漏掉执行上限 | 多hook提示未声明每个code的helper/AST硬限制 | 零更新停止；显式声明已有validator契约并从E0/W0同步重启 |
| 异步receipt重复/缺失 | 消费序号与真实task身份不一致 | exact multiset审计，隔离无效attempt并计物理成本 |
| 接口/预算/判分不一致 | 本地proxy和论文协议不同 | 显式披露，不宣称同起点或绝对分数复现 |

ALFWorld资产使用官方`0.2.2/json_2.1.1`三份archive，保留官方500题目录。
更新版游戏bundle缺少部分指定任务，不能静默替换。
Engineer API须正确继续最后的`<think>\n`，避免重复generation prefix；
target必须返回真正的OpenAI `tool_calls`，文本看似XML不是等价证据。

## 6. 下一步与停止条件

1. H-SAO与W-SAO真实optimizer smoke已完成；正式checkpoint均不从smoke继承。
   E0/W0 dev为no-patch79/100、packet-transductive76/100。packet370基础设施
   失败已仅重跑该packet。r1/r2/r3均为零optimizer更新且已隔离；protocol v8下
   两臂H1-r4均已完成2个actor/4个critic更新并产生权重变化；100-task dev
   refresh/global Harness门控正在运行。
2. node4按H至平台→W至平台执行顺序SAO；node3按H1-W1-H2-W2执行MA-SAO。
   每个角色至少两轮、patience=2、最多八轮，基础设施失败不算模型0分。
3. 每轮只用dev 300–399停止/接受；要求≥1pp、paired gain>0且one-sided
   bootstrap 95% LCB>0。两臂冻结前禁止读取test 400–499结果。
   Solver权重门控使用同一100个task的no-patch paired reward，避免把同时变化的
   Harness收益误归因给W更新；global Harness另行在该Solver状态下选择。
4. 只有MA-SAO在sealed test比顺序SAO至少+5pp且LCB>0，才搜索交替频率。
5. WebShop尚未准备；小商品库预览与论文配置须区分，Java/JNI/检索索引和
   `webshop_goal_seed=233`/task manifest匹配后才能报告。
6. WHALE同起点复现须恢复原始Solver/h0/检索器/判分/数据协议；
   当前SearchQA W4+BM25+alias EM不满足。部分原始中间数据未公开，需披露重建限制。

正式SearchQA接受门槛：clean执行、至少+0.5pp独立dev收益、
题级bootstrap 95%下界>0，并有正的RL-vs-SFT配对证据。
小样本正例仅支持继续调查；无合法候选、全零reward或基础设施错误不能通过门槛。

## 7. 接手与持久化规则

- 代码、方法、主要聚合结果进入Git；`assets/`、`runs/`、权重、optimizer、
  原始任务/轨迹、缓存与凭据不进入Git。
- 固定WHALE、SAO、MAPPO参考和Harness-R1 submodule；遵循
  [第三方说明](../THIRD_PARTY_NOTICES.md)，不可称非官方SAO为官方实现。
- 首先读本文件，再读对应方法README和实际JSON receipt；启动日志不是完成receipt。
- 当前服务器仅授权node3/node4。CPU测试显式`CUDA_VISIBLE_DEVICES=''`，
  每节点GPU进程由单个ownership supervisor管理，不杀未知/其他用户进程。
- 目前两节点运行H1-r4的正式dev门控；开始新任务前仍须重新核验实际service、PID、端口和
  GPU占用，不能只依赖本文判断运行状态。
- 保存已有权重、算法和分析；清理中间产物必须先有经过验证的保留清单。
  旧约826.5GB checkpoint inventory仅做过盘点，不能声称已安全清理。
- 使用既有测试；仓库导出不是新的GPU实验。提交前禁止带入任何模型文件或凭据。

外部来源：[Harness-R1](https://github.com/DeepExperience/Harness-R1)、
[官方模型卡](https://huggingface.co/ShaoShuai0605/Harness-R1)、
[WHALE复现说明](https://github.com/krafton-ai/WHALE/blob/main/docs/reproducing.md)。

## 8. 本次代码导出验证

提交范围的CPU回归：**448 passed、5 skipped、843 subtests passed**。
跳过项不代表通过GPU训练测试；本次没有重新训练模型。存在上游依赖的弃用警告。
服务器Python环境为torch 2.10、transformers 5.7、vLLM 0.19.1；
旧发布环境的requirements不能静默视为当前环境完全相同。

```bash
cd /path/to/coevolve
git submodule update --init --recursive
# 按对应方法README准备SAO/AReaL与ALFWorld的独立环境后：
CUDA_VISIBLE_DEVICES='' TMPDIR=/dev/shm PYTHONPATH=whale_local:. \
  /path/to/current/python -B -m pytest -q \
  tests whale_local/tests whale_local/harness_r1_alfworld/test_driver.py
```

Artifact路径和原始数据只存在实验服务器；Git中的本文保留主要聚合结论与限制。
补丁文件中的空白context行是统一diff格式，不能为了普通源码的尾空白检查破坏补丁
或改变已冻结的补丁SHA；其他提交源码通过`git diff --check`检查。
