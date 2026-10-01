# 项目状态与 Agent 接手指南

更新：2026-10-01 UTC。这是当前研究主线的入口；历史细节见
[实验日志](EXPERIMENT_LOG.md)，算法定义见 [MA-SAO](MA_SAO.md)、
[Harness-R1 SearchQA适配](../whale_local/harness_r1_searchqa/README.md)和
[ALFWorld复现入口](../whale_local/harness_r1_alfworld/README.md)。

## 1. 目标与当前结论

目标不是只提高解题模型的权重，而是训练一个 Engineer，使其能够根据真实
失败轨迹提出可执行、可采纳、能在未见任务上提高表现的 harness 修改。
先验证 Harness-R1 的官方方法，再在相同起点、奖励和预算下比较 SAO / MA-SAO。
最终主证明目标是9B Solver与9B Engineer。按2026-10-01 11:20 UTC的新指令，
4B Solver修复加扩规模已通过更强双seed门槛，当前转向9B Solver，
随后验证9B Engineer；不能把某次4B校准失败说成推翻历史4B正向结果。

**2026-10-01 16:07 UTC用户更新优先级：快速找到有效配置优先，取消探索阶段
的多seed/消融前置要求。** node3/node4用于不同候选或Solver/Engineer并行，
不再重复同一配置的两个seed。已经开始的9B v2对照正常收尾并保留原gate，
不得追溯性放宽旧结果；尚未启动的9B v3双seed扩规模计划暂停。后续单seed
结果明确标记exploratory，继续保留固定起点、独立任务评测和基础设施失败记录。
Engineer准备与验证不再等待Solver的双seed门槛；MA-SAO仍须先有两侧学习证据。

**ALFWorld已经证明train候选到未见dev任务的global Harness提升，但尚未证明
SAO更新后的Engineer权重比E0更会改Harness，也尚未证明9B Solver权重侧SAO
能通过held-out gate；在这两项顺序基线成立前，MA-SAO继续锁定。**

- 旧 SAO / MA-SAO 实验的主要收益来自 Solver 权重；多次交替实验始终保留 h0。
- 32条数据的4B Harness-R1小试未通过输出/执行门槛，不能评价有效RL学习。
- 官方9B Engineer在ALFWorld能够生成并实际应用修改，解决了执行能力的校准问题。
- 官方冷启动SFT Engineer两次小样本提案分别得到4/8与5/8；存在一个正收益实例，
  但还不是稳定收益、未见任务收益或RL优于SFT的证据。
- Agent-SFT Solver得到7/8是Solver收益，绝不能算作harness改写收益。
- 正式ALFWorld H1在顺序/MA两臂分别得到+8pp（LCB+2pp）和+11pp
  （LCB+5pp）的未见dev收益；这是clean global Harness证据。
- 两个被采纳H1都来自第一次optimizer更新前的behavior-version-0。因此该结果
  证明E0候选搜索与筛选有效，不能归因于训练后的E1变强；E1产生的后续候选
  没有超过H1。
- 两臂W1/W2均未通过paired no-patch dev gate；最终都冻结为H1+官方W0。
- 最终dev上MA为90/100、顺序为88/100，但MA−顺序仅+2pp且LCB=0，
  未达到预注册的+5pp，因此禁止搜索交替频率。
- 新的fixed-no-patch 9B Solver双seed验证已于2026-10-01 06:40:35 UTC
  全部完成。seed42独立report为74%→72%，seed43为79%→77%，平均-2pp；
  两个题级95%区间都跨零，预声明pair gate失败。速度修复和非零梯度
  不能替代held-out学习收益，当前仍不能宣称SAO已经提升9B Solver。

不要把“启动成功”“参数发生变化”“合法patch”“训练reward为正”
或“GPU显示100%”替代最终实验目标。

### 历史顺序验证阶梯（2026-10-01；后续探索按上述新优先级执行）

1. 固定数学H0，只训练4B Solver。双seed各用8步critic-only warm-up、
   64步actor、batch64，共4,608条真实轨迹；在独立reward split上从
   3个预声明checkpoint中保守选择，再在512题mean@2 report上配对比较。
2. 4B v2仍按原冻结gate诚实报告：两个seed都选中训练checkpoint且不退步，
   平均至少+3pp，并至少一个seed的题级95%区间下界大于0。
   该实验现已结束且gate失败：两个seed均未通过selection LCB，故都保留W0、
   未打开训练checkpoint的独立report。仅作探索性描述，最终selection点
   seed42为+1.37pp、seed43为+5.08pp，后者LCB仍为-0.20pp。
3. 固定空Harness的9B ALFWorld Solver首轮双seed验证已完成，但gate失败。
   每seed为5步critic-only warm-up、25步actor、batch8，共240条不同train
   任务；actor实际更新使用后200条。主线仍是诊断和加强顺序SAO，不得从
   本轮selection正差值推断泛化提升，也不得把这次失败视为SAO理论不可训练。
4. 9B Solver通过后，才扩大9B Engineer SAO，并在完全相同packet、prompt、
   decoding seed、候选数和全局held-out任务上做E0/E1能力A/B。
5. 若SAO-H失败，先按官方Harness-R1的组内GRPO目标做同预算复现；只有
   9B Solver和Engineer两侧都证明可训练后，才解锁MA-SAO与交替频率研究。

### 9B Solver v1结果（fixed no-patch，2026-10-01）

| seed | W0 selection | step9 / step19 / step29 selection | 选中step | 独立report W0→候选 | 配对delta及题级95%区间 |
|---|---|---|---|---|---|
| 42 | 61.67% | 65.00% / 56.67% / 56.67% | 9 | 74%→72% | -2pp，[-7,+3]pp |
| 43 | 58.33% | 60.00% / 60.00% / 61.67% | 29 | 79%→77% | -2pp，[-11,+7]pp |

selection为240–299的60题，report为300–399的100题；本轮未打开400–499。
两个seed均清除selection的+2pp floor，但独立report均未满足不退步要求，
平均delta和显著seed数也未过gate。selection的正差值只是模型选择信号。
seed42后期selection下降是退化信号，尚不足以单独确认过拟合或确定根因。
正式结果见服务器
`runs/alfworld_solver_v1_pair_gate_result.json`及两个
`runs/alfworld_solver_v1_exact_concat_s4{2,3}/COMPLETE.json`。

两个训练阶段分别耗时52.36/52.50分钟；每个actor batch精确为8条concat
轨迹，三个actor/critic checkpoint均完整保存。optimizer窗口中真实实验SM
为40.21%/41.44%，filler为6.63%/6.60%，filler全过程active；
显存峰值为134,988/129,332MiB。短smoke的约66倍episode吞吐不是完整
训练加评测的端到端加速比，正式长episode的吞吐和显存须使用这些实测receipt。

seed42的step29评估曾把单题4096-token预算耗尽误当作基础设施异常而重启。
现已按其它域的既有语义返回零分、`truncated=true`并取消环境session；
恢复后60题中3题被正确记为预算耗尽失败。训练未重跑，任务、seed、预算、
候选checkpoint和选择规则未变。源码哈希迁移及旧manifest保存在该run的
`TOKEN_BUDGET_SEMANTICS_MIGRATION.json`中，不能描述为全程同一源码哈希。
此前完成的对照/selection及seed43评测没有预算耗尽，原结果保留。

预声明有限工作由系统级systemd service执行，写原子`STATUS.json`、
训练attempt、checkpoint选择、`COMPLETE.json`/`FAILED.json`和GPU释放审计；
客户端断线不终止这些工作，但队列完成不等于研究目标完成，也不会自动设计下一轮。
检查这些unit必须使用`systemctl`，不是`systemctl --user`。
历史11:08 UTC快照中，当时训练/评测队列均已结束，没有研究GPU任务在跑。
两节点adaptive filler receipt仍新鲜，平均总利用率node3为87.13%、node4为
86.50%，外部实验SM均为0；这是补载，不是算法利用率。

### 4B v3修复与扩规模协议（已完成）

已用原生AReaL GAE复现一个边界错误：同一个已经结束、reward=1的episode，
若末尾critic value=3，在batch中未padding时return=4，有padding时return=1。
原因是上游把`seqlen == batch_width`当作未终止依据；9B v1每rank两个episode，
`no_eos_ratios`恰为0.5，冷启动critic target也曾超过二元奖励范围。
这证明边界处理存在bug，但不能直接证明它解释了全部质量差异。

新`terminal_episodes=true`只接受完整fresh concat Solver轨迹或完整Engineer
动作，在worker端RTensor实化后、GAE前把最终动作之后的state value置零；
有效action-state value、
行为logprob和loss mask不改。旧默认false用于兼容历史配置，critic-free模式
无需额外修改。回归覆盖padding不变性、0/1/-1 terminal reward、工具观察mask
和输入不被修改；原生SAO loss及上游patch保持不变。

4B v3双seed已预声明：

- 保持原4B的W0/h0、LR、batch64、8步critic warm-up、learned critic K=2、
  temperature和staleness；启用exact concat及terminal boundary修复。
- 共240步，其中232步actor；15,360条不同train题，比v2的4,608条增加3.33倍。
  不以重复相同题伪装扩数据；保留384题train尾部report，不碰AIME sealed split。
- 每24步保存，共10个checkpoint。step23/47/71与v2提供同预算的探索性比较，
  其余点用于观察scale曲线；这不是将两个修复因素完全分离的因子实验。
- selection为256题mean@2，对10个checkpoint作Bonferroni校正；只对选中的
  checkpoint打开独立512题mean@2 report。已有report被使用过，因此属于
  development复验，不能称为新、未触碰的confirmatory test。
- 更强门槛：两个seed均选中训练checkpoint、每seed至少+2pp、平均至少+5pp、
  两个题级95%区间下界均大于0。通过后才运行已准备的9B v2协议。

配置见`examples/dev_solver_only_v3_terminal_scale_s42.json`、
`examples/dev_solver_only_v3_terminal_scale_s43.json`和
`examples/dev_solver_only_v3_pair_gate.json`。控制器冻结边界修复等全部相关源码
哈希，记录真实optimizer窗口GPU利用率，不停止filler。
首次live启动在第一个optimizer更新前暴露RTensor controller/worker边界错误；
已停止训练unit并保留filler，失败root完整归档。修复移到FSDP worker端，
原生batch/unpack回归加入后为65 tests、9 subtests通过；正式协议从新root重启，
不复用失败attempt的rollout。
worker修复已完成真实GPU验证：`7d765e5`代码在新root完成15/16个optimizer
step（seed42/43），全部critic target严格在[0,1]；前8步没有actor更新，
随后7/8步actor梯度均为非零，每batch恰为64条concat轨迹、critic K=2。
证据：`runs/dev_solver_only_v3_live_terminal_validation.json`。
两节点optimizer窗口真实实验SM为62.40%/64.07%，filler仅0.49%/0.56%
（58/60次采样，无采样错误；只是早期窗口，非全程数据）。
训练系统unit为`coevolve-4b-solver-v3-s42-r1`（node4）和
`coevolve-4b-solver-v3-s43-r1`（node3），均零重启；filler始终active。
完整240步和模型选择、report现已结束；见下方最终结果。
补充controller远程handle不可运算的工厂接线回归，新增测试单独运行通过；
不修改正在运行的冻结训练源码。

### 4B v3最终结果：更强Solver门槛通过（2026-10-01 14:15 UTC）

固定H0，数学development report为512题、每题2次生成：

| seed | selection选中step | report W0 | report候选 | 配对提升 | 题级95%区间 |
|---|---|---|---|---|---|
| 42 | 143 | 41.50% | 56.54% | +15.04pp | [+11.62,+18.46]pp |
| 43 | 191 | 38.18% | 51.56% | +13.38pp | [+9.86,+16.99]pp |

平均**+14.21pp**；两seed均选中训练checkpoint、均超过+2pp、
平均超过+5pp、两题级95%下界均大于0，预声明强门槛全部通过。
结果：`runs/dev_solver_only_v3_pair_gate_result.json`。
这提供了比前一轮同数学协议更强的Solver学习证据，但修复、concat与规模
一起变化，尚不能把收益逐项归因；跨benchmark也不能直接比较提升幅度。
report已有使用记录，仍是development复验，不能冒充全新confirmatory test，
题级bootstrap不能替代大量训练seed上的统计显著性。

两节点均完成240步、15360个不同train任务及10对actor/critic checkpoint，
全部critic target在[0,1]，8步warm-up隔离正确，后续actor确实更新。
optimizer窗口为117.69/117.52分钟，真实实验SM为58.40%/57.55%，
filler为1.82%/1.94%，全程active。不是用filler占用率充当算法利用率。

seed42在step119评测模型启动时发生一次replica退出，systemd自动恢复，
没有重训、改seed、改题或改gate，旧评测缓存保留。原replica stderr被旧
Launcher的`wb`重启日志覆盖，底层原因未知；不要把恢复后的tool-parser
警告误当成启动故障原因。恢复receipt在该run的
`evaluation_recovery_20261001T135423Z.json`。
该run最终`elapsed_seconds`只覆盖重启后的controller，不能当全程耗时。
后续Launcher会把同名旧日志先归档到`previous_logs/`，不再覆盖故障证据。

同预算探索点也已完成：step71的selection提升为+6.45/+6.64pp，
十重校正LCB为0/+0.98pp；继续到选中的step143/191后，selection提升
为+12.89/+14.06pp。这支持在修复后的协议中继续增加数据/训练步数有用，
但不是对早期checkpoint另做独立report、也不是把bug修复与scale分离的消融。

### 9B v2控制已完成：正负各一，平均无提升

强4B gate通过后，于2026-10-01 14:41 UTC启动两个持久系统service：
node4的`coevolve-9b-solver-v2-s42`与node3的`coevolve-9b-solver-v2-s43`。
使用`2fed1fd`源码，维持旧9B的30步、5步critic warm-up、batch8、K1、
原240题、解码预算和选择门槛；启用正确的worker端terminal修复。
两个baseline API均返回200，14:56 UTC进入training阶段，约16:20 UTC全部结束。
每个run的`launch_receipt.json`冻结首次启动时间、unit、spec和4B gate哈希，
避免重启后仅用controller当前uptime计算总耗时。

| seed | 选中step | 独立report W0 | 训练后 | delta | 题级95%CI |
|---|---:|---:|---:|---:|---|
|42|29|74%|81%|+7pp|[0,+15]pp|
|43|29|81%|74%|-7pp|[-17,+3]pp|

原双seed gate失败，平均delta为0，不能仅摘取seed42宣称9B已稳定学会。
两个selection都从61.67%升到70%，再次说明selection上涨不足以证明泛化。
seed43的W0与v1的79%有2pp差异：相同名义解码设置并不意味着逐题bitwise复现。
两边均完成30步/25次非零Actor更新，targets始终[0,1]，三个actor/critic
checkpoint完整，观察到的controller重启数均为0。optimizer窗口43.36/46.65
分钟，真实实验SM40.25%/42.16%，filler为6.55%/6.32%，全过程active。
结果与完成时源码/原生更新审计分别持久化在
`runs/alfworld_solver_v2_pair_gate_result.json`、
`runs/alfworld_solver_v2_final_native_audit.json`。之后的合法源码修改不能用来
追溯否定完成时已核对的冻结身份；旧gate结果也不会因改用单seed探索而重标。

旧9B实测计时还表明，记录到的阶段时间中actor/critic更新约占62%--64%，
保存约占21%--22%，GAE只占1.6%，不能把向量化GAE当主要提速方向。
learner卡显存曾达约125--132GiB，不能盲目翻倍batch。上游异步HF saver
只支持Archon，FSDP设置async会退回同步，不能声称已经启用异步保存。

### 后续9B goal隔离扩规模准备（scale尚未启动）

已解除旧ALFWorld训练适配器只能使用500题catalogue、其中240题训练的限制。
新增显式`training_split=sao_train_v1`，**默认`test_set`行为保持不变**。
本地3605个train游戏在排除官方500题及全部`valid_seen`、`valid_unseen`、
`valid_train`的goal-group后，剩余**2296个game实例、880个goal-group**。
不同实例可能属于同一训练goal，不能把2296说成2296个不同goal。
训练catalogue位于独立输出目录，运行时通过run-local overlay接入，
不修改共享的只读benchmark目录，也不覆盖原`test_set.json`。

元数据审计还发现：旧9B协议虽然任务ID不重叠，但train/selection、
train/report、selection/report分别共享17/24/5个goal-group。
这不是相同game实例重复，但意味着旧“held-out”仅指任务ID层面，
不是goal-unseen泛化，原题级bootstrap也未作goal簇修正；旧负结果不改写。
新scale协议固定59个selection goal、91个report goal，各取一个实例，
两者goal互斥、均用mean@2，不使用400--499区间的评测结果。
这些题以前已有使用记录，仍是development复验，不是全新confirmatory test。

[seed42](../examples/alfworld_solver_v3_train_only_scale_s42.json)与
[seed43](../examples/alfworld_solver_v3_train_only_scale_s43.json)预声明：
从原始W0重新开始、固定H0、batch8、5步critic warmup、235步actor，
共1920个不同game实例（覆盖880个训练goal），每30步保存，8个checkpoint。
模型、LR、K1、concat、terminal修复、GC和单步显存预算沿用9B v2；
此试验同时改变训练池、规模和评测隔离，不能声称是纯step数消融。
后续仍复用原9B双seed promotion gate，不因扩大数据而放宽门槛。
4B强门槛通过后，先做9B v2同预算控制；更强9B结论再看goal隔离scale。

上述240步双seed配置保留作协议记录，但按最新指令不再排队启动。
新的[单seed快速方案](../examples/alfworld_solver_fast_v4_s42.json)使用fresh W0、
相同goal隔离数据与59/91题mean@2评测，5步critic warmup加75步actor，
batch8，共640个不同train实例/goal，Actor实际使用600个；仅保存step39/79。
LR改为Actor 2e-6、critic 1e-5，保持learned critic、K1、concat和terminal修复。
这是面向有效性能的联合配置探索，不是LR或步数的单因素消融。
[单seed探索门槛](../examples/alfworld_solver_fast_exploration_gate.json)要求独立
report至少+3pp、题级95%下界大于0；通过也不代表多seed稳定或正式确认，
更不能隐藏既有负结果。另一节点优先准备/验证Engineer，不重复这个seed。

复现数据准备（仓库根目录，无模型调用）：
```bash
PYTHONPATH=whale_local python3 -m sao_bridge.alfworld_train_data \
  --output assets/alfworld_solver_train_only_v1
```
生成文件不可静默覆盖；规范化spec冻结catalogue/data校验和，训练进程只收到
精确预算的行，避免异步预取触及未分配的后缀任务。两个节点均通过CPU预检。
51项回归与7个subtests通过；真实AgentBench task512完成start/step/cancel
验证，使用手写`look`而非LLM，零GPU，不计入算法准确率。
证据：`runs/alfworld_train_catalogue_cpu_smoke_v1/validation.json`及
`assets/alfworld_solver_train_only_v1/protocol_preflight.json`。
当前4B两个run的冻结源码校验和保持不变。

当前顺序验证入口：

- `whale_local/sao_bridge/math_solver_validation.py`：4B固定H0双split单seed控制器；
  `math_solver_pair_gate.py`聚合双seed。
- `whale_local/sao_bridge/alfworld_solver_validation.py`：9B固定no-patch
  baseline、训练、checkpoint selection与独立report控制器；
  `alfworld_solver_pair_gate.py`聚合双seed。
- `whale_local/sao_bridge/alfworld_solver_eval.py`：显式seed、官方环境、
  exact no-patch的W0/W1 paired evaluator。
- `whale_local/sao_bridge/gpu_telemetry.py`：按精确PID区分实验/filler，
  同时记录峰值显存；算法利用率必须取`mean_experiment_sm_percent`。

新增只读的`alfworld_solver_monitor.py`供预声明9B单seed或多seed长期监管，不修改训练
配置、不停止GPU进程，也不负责替代controller的systemd重启策略。它检查
完整JSONL行、连续step、critic-only warmup、真实Actor更新、concat batch、
terminal target范围、冻结源码/数据、最终actor/critic checkpoint及GPU释放。
短暂controller失败给既有自动恢复留出10分钟，持续失败或不变量错误写出
`attention_required`并退出；不能将失败算作零分或通过gate。

在controller已生成`spec.json`及`source_manifest.json`后运行当前单seed版本：
```bash
PYTHONPATH=whale_local python3 -B -m sao_bridge.alfworld_solver_monitor \
  --protocol examples/alfworld_solver_fast_exploration_gate.json \
  --run runs/alfworld_solver_fast_v4_s42 \
  --output runs/alfworld_solver_fast_v4_result.json
```
默认每60秒写原子`*_monitor.json`，24小时超时，预声明gate和run身份存入
不可变的`*_monitor_protocol.json`，重启监管也不能放宽gate。只有所有声明run完整
结束、当前尝试释放GPU且filler仍active后才发布pair result。应由持久系统
service执行此CPU命令，不依赖编辑器连接。运行中源码应保持冻结；实验结束
后合法修改源码，不应再对旧run重启这项“当前源码一致”检查。

### 9B Engineer快速并行准备

[数据准备器](../whale_local/sao_bridge/alfworld_engineer_data.py)复用24个已冻结的
train failure cohorts，通过官方builder压缩观测证据，不重新采集Solver rollout。
基线文件及reward/task identity在压缩前后检查；验证器限制仍放入E0/E1相同
system prompt。192行训练数据的精确chat-template prompt为7,448--7,733 tokens，
加4,096生成预算最坏11,829，低于16,384 microbatch；不是依靠截断训练轨迹提速。

[快速H配置](../examples/alfworld_engineer_fast_v1_s42.json)为5步critic warmup+
27步actor、batch6、K1、terminal修复，使用3 learner+1 Engineer rollout+
4 frozen Solver GPU，fresh官方E0/W0，不从历史被拒checkpoint接着训练。
E0/E1使用相同8个压缩packet、相同decode种子和候选数。metadata-only过滤排除
全部legacy train goal及selection/report相互亲缘后，得到24个selection和70个
report goal。主要目标是固定证据下候选质量的平均E1−E0变化，而不是挑出一个
比no-patch好的E0提案；不声称测试了未见failure packet的迁移能力。
数据位于`assets/alfworld_engineer_fast_v1/`。评测流水线仍需完成实际GPU验证。
[评测协议](../examples/alfworld_engineer_fast_evaluation_v1.json)在训练前固定：
仅比较step15/31，selection至少+2pp才打开该候选的report，report目标为
平均E1−E0至少+3pp且题级95%下界大于0；重复和非法提案均保留在8个候选
的分母内。协议是快速探索，不作多seed稳定性或独立confirmatory声明。

[有限H训练worker](../whale_local/sao_bridge/alfworld_engineer_training.py)供系统
service执行：冻结配置、数据、provenance与源码，最多两个训练attempt，失败
attempt原样归档；完成训练只写`TRAINING_COMPLETE.json`和
`training_complete_pending_evaluation`，不冒充Engineer能力验证完成。
```bash
PYTHONPATH=whale_local /dev/shm/sao-venv-ppo/bin/python -B \
  -m sao_bridge.alfworld_engineer_training \
  --config examples/alfworld_engineer_fast_v1_s42.json
```
systemd需保留`Restart=on-failure`、`RestartSec=30s`、
`RestartPreventExitStatus=78`与精确unit级进程所有权。三个重试/归档/完成恢复
回归测试通过，训练与评测为独立阶段，后者尚不能因训练service退出0而跳过。

另已把train-only catalogue接入Engineer launcher和官方reward adapter：可选
`training_split/training_catalogue/training_catalogue_sha256/solver_weights_sha256`
贯穿预检、run-local overlay、请求metadata和cache namespace。每次请求先检查
catalogue hash；新模式拒绝错误Solver权重或既有未绑定cache。默认`test_set`
行为保留，本次快速H先复用上述旧train证据，再排除相关评测goal以节省采集时间。
Solver telemetry现在直接导入共享`gpu_telemetry.monitor`，不再依赖Engineer
launcher文件，便于两条研究线独立开发。准备阶段61项测试及22个subtests通过。

GPU利用率必须分开报告。新增per-process telemetry把精确识别的adaptive
filler PID与其它实验PID分开。4B启动阶段曾观测到表面72.75% SM中仅
1.94%来自实验、71.88%来自filler；进入真实optimizer后，node3短窗为
总50.0%、实验48.22%、filler 0%，同时filler unit仍保持active。由此证明
不能用`nvidia-smi`总利用率冒充算法利用率，也证明训练活跃时filler会自动
完全让载，无需停止用户的服务。隔离吞吐测量结束后应恢复adaptive shared
模式：高SM训练阶段duty自动归零，异步空档只做低幅补载；只在严格A/B窗口
临时使用有时限的reservation。恢复shared后的匹配5分钟稳态窗口中，node4
实验/filler SM为58.10%/3.29%，node3为58.73%/3.55%，两边49个样本均有效。

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

两臂H1-r4随后均完成2个actor/4个critic更新。dev refresh在node4得到
no-patch79/100、packet-transductive80/100，在node3得到78/100与81/100；
transductive数字只作诊断。正式global候选仅来自optimizer实际消费的
behavior-version-0 train receipts（node4 10个、node3 11个合法唯一候选），
不使用dev生成patch。node4正式接受一个+8pp、one-sided95% LCB+2pp的H1
candidate（79→87）；node3接受+11pp、LCB+5pp（78→89）。这是本项目首次
通过clean train-candidate→dev gate的global Harness提升。

首次global pass还暴露官方packet evaluator返回局部reward ID0–9，而协议任务
ID为300–399。selector在作出决定前停止；修复后只接受精确global集合或完整
packet-local连续集合，并将后者按packet start映射回协议ID。中断cache retry
现在只复制评测身份完全一致的结果，排除`.lock`并丢弃缓存的infra-failure
`reward.json`。node3 retry3复用了109个成功评测，只重算1个失败key，
253秒完成。完整`whale_local/tests`为135 passed，后续失败cache过滤定向回归为
18 passed。

node4 H2又完成2个actor/4个critic更新（1,320.9秒），但没有超过H1 incumbent；
H3从接受的H1而非被拒绝的H2重启，1,207.4秒后同样增量为0。H2/H3连续两轮
未接受，顺序臂Harness达到patience=2并冻结H1 patch
`60b43b8fd5f64852bd20234dbd4e50e71b46253a6f604adba86bc53f5f49a680`。

两臂随后分别用21+3（node3）和18+6（node4）的严格partial recovery覆盖全部
24个train packets，各生成512行assignment-aligned Solver数据。MA W1首次因把
完整9.8KB patch源码放入critic context而在GPU分配前超出1024-token预算；改为
稳定patch SHA后，第二次又在首个value forward暴露AReaL DP4实际发送1-D单轨迹，
而prefix逻辑只接受`[1, seq]`。两次均为零optimizer update。修复后MA W1和顺序
W1均完成2个actor/4个critic更新，分别耗时4,060.0秒和4,065.1秒；MA使用
224-token pre-action centralized context，顺序臂不使用。MA权重digest从
`435cb03ebd10`变为`258e52983e8b`，顺序臂变为`81bc16629ec3`。

W1的paired no-patch dev gate均拒绝：node3为78→78、gain 0、LCB 0；node4为
79→81、gain +2pp但LCB 0。node3 dev packet用8+2恢复，node4用9+1恢复；
packet-transductive结果77/100和83/100只作诊断。MA H2随后在E1/W0上完成
2个actor/4个critic更新（1,218.0秒），64行数据中有25% authentic replay；
实际两个batch精确消费9条fresh与3条replay。最终权重digest为`a68594777e7d`。
其9个已消费fresh候选都未超过H1 incumbent；H1本轮重测为90/100，相对
no-patch 78/100为+12pp、LCB+6pp，这一题运行波动不归因于H2。H2拒绝。
两臂W2均从接受的W0（不是被拒绝的W1）重启，使用sample offsets512–1023。

顺序W2完成2个actor/4个critic更新，耗时4,728.7秒，最终权重digest
`74eb852d15c2...`；paired dev为79→77、−2pp、LCB−4pp，拒绝。MA W2同样完成
2个actor/4个critic更新，耗时4,162.8秒，centralized critic保持开启，最终
digest为`76f476f8cfb6...`；paired dev为78→78、gain/LCB均为0，拒绝。
两臂W1/W2连续两轮不接受，因此Solver均冻结为官方W0。

MA H3从接受的E1而非被拒绝的H2重启，25% authentic replay，完成2个actor/
4个critic更新（1,163.3秒）；exact ledger为9 fresh+3 replay，7个合法唯一候选
均未超过H1。H2/H3连续拒绝后，MA Harness也冻结H1 patch
`6e6750643c1b451c65c9be1db61f73db7e88d3ab9150ca005ec00e37d36a0234`。
两臂最终dev重测为顺序88/100、MA90/100；MA−顺序+2pp、one-sided 95%
LCB=0，未达到+5pp/LCB>0的cadence-search门槛。

两臂冻结后才首次解封sealed test 400–499。node4一次完成10 packets；node3
首次9/10，仅串行恢复start470后严格合并为10/10，基础设施失败未记模型0分。
两份metadata的task/path完全一致，但no-patch为67/100和65/100且6题不同；
在任何fixed-patch结果产生前冻结为对称双replicate、按task均值bootstrap。
最终no-patch均值66/100，顺序H1为78/100（+12pp、LCB+6pp），MA H1为
83/100（+17pp、LCB+10.5pp）。MA−顺序恰好+5pp，但one-sided 95% LCB=0；
因此两个Harness的sealed提升成立，MA的material superiority不成立，
`cadence_search_permitted=false`。

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
- `/mnt/3fs/xudong/gpu-utilization/adaptive_gpu_filler.py`
- `/mnt/3fs/xudong/gpu-utilization/README.md`
- `/mnt/3fs/xudong/gpu-utilization/manage-adaptive-gpu-filler.sh`
- `/mnt/3fs/xudong/gpu-utilization/bruli-h200-load.service.in`
- `runs/alfworld_sao_masao_20260930/final_sealed_arm_comparison.json`

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
| 自适应filler空闲态 | node4 8卡平均85.25% util；约791MiB/卡，旧静态实现约80GiB/卡 |
| 自适应filler共存开销 | 20秒TF32对照55,584 vs 56,414 batches；吞吐比98.53%，约1.47%开销 |

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
| dev packet仍写共享cohort | 旧driver把generated config/cohort留在共享asset目录 | AgentBench、config、cohort、task ID和scratch全部run-local；exit0 summary仍检查infra failure并有界重试 |
| global reward返回packet局部ID | 官方子评测输出0–9，selector基线使用300–399 | 只接受精确global集合或完整局部集合；后者按packet start显式映射，其他形状失败 |
| 中断cache retry等待7200秒 | SIGKILL遗留`.lock`，失败`reward.json`又会成为cache hit | resume排除锁并丢弃infra-failure entry，只复用评测身份一致的成功结果 |
| centralized critic context超预算 | 把完整Harness源码放进joint context，tokenizer后远超预留1024 | 只存稳定patch SHA；H/W领取GPU前审计context token数 |
| DP4 value输入为1-D | 每rank单轨迹没有显式batch维，旧prefix只接受`[1,seq]` | prefix沿最后一维扩展全部token-aligned字段，返回actor前剥离context |
| replay轮次version-0候选过量 | 放宽replay head lag后异步预取也产生version-0 fresh receipts | 训练持久化前`steps×batch` ID ledger；global gate精确限定已消费fresh ID并排除replay |
| W并发出现HTTP 406 | 连续task IDs共用同一单会话packet worker | 跨24 workers round-robin；启动前审计完整异步active window |
| AReaL导出缺preprocessor metadata | 权重checkpoint不一定复制全部Qwen runtime文件 | 从输入checkpoint补齐并单独记录；权重变化使用纯safetensors digest |
| H训练提示漏掉执行上限 | 多hook提示未声明每个code的helper/AST硬限制 | 零更新停止；显式声明已有validator契约并从E0/W0同步重启 |
| 异步receipt重复/缺失 | 消费序号与真实task身份不一致 | exact multiset审计，隔离无效attempt并计物理成本 |
| 接口/预算/判分不一致 | 本地proxy和论文协议不同 | 显式披露，不宣称同起点或绝对分数复现 |
| 静态filler与训练争抢 | 旧实现固定55%显存且duty下限82%，只能训练前停服务 | always-on逐卡状态机；外部负载出现时快速yield，持续空闲才分级恢复 |

ALFWorld资产使用官方`0.2.2/json_2.1.1`三份archive，保留官方500题目录。
更新版游戏bundle缺少部分指定任务，不能静默替换。
Engineer API须正确继续最后的`<think>\n`，避免重复generation prefix；
target必须返回真正的OpenAI `tool_calls`，文本看似XML不是等价证据。

## 6. 下一步与停止条件

1. 顺序和MA两臂已按patience=2完整冻结为H1+W0；W1/W2都没有产生可接受的
   Solver权重提升。不得从被拒绝的W/H checkpoint继续，也不得用sealed结果回调。
2. sealed test 400–499只在冻结后解封。对称双replicate最终为顺序78/100、
   MA83/100；MA−顺序+5pp但LCB=0，material superiority未建立。
3. MA−顺序须至少+5pp且one-sided 95% LCB>0才允许cadence搜索；sealed结果
   未通过，因此不搜索频率，也不能据此选择checkpoint或超参数。
4. 当前主算法瓶颈是9B Solver weight-side样本效率与反向吞吐。历史每轮仅
   8个长工具episode；稳定step中actor约885秒、K=2 critic约1801秒，而
   异步rollout等待已接近0。更深一层的根因是Solver按turn导出时反复训练
   累计对话前缀；历史4个episode可膨胀到166条sequence和约216万actor
   token。AReaL原生exact-token concat资格测试已把`n_seqs`从166降到8、
   actor token降到42,996；actor为12.84秒、K=1 critic为13.16秒，首批
   rollout为50.18秒，峰值显存102,540MiB。相对历史非checkpoint核心step，
   batch翻倍后的episode吞吐约66倍。GC-on时反向已短于rollout，因此正式
   4+4基线保留gradient checkpointing，避免为几秒收益增加OOM风险。
5. 主9B基线采用learned critic；critic-free terminal-return只能作为明确
   标注的速度/归因ablation，不能冒充原始learned-critic SAO。Engineer侧
   因单回合terminal reward可并行比较该ablation。
6. WebShop尚未准备；小商品库预览与论文配置须区分，Java/JNI/检索索引和
   `webshop_goal_seed=233`/task manifest匹配后才能报告。
7. WHALE同起点复现须恢复原始Solver/h0/检索器/判分/数据协议；
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
- 截至2026-10-01，两臂训练已冻结；node3/node4部署always-on adaptive filler。
  开始新任务前仍须重新核验service、PID、端口和GPU占用，不能只依赖本文判断。
- `bruli-h200-load.service`保持同一授权unit；adaptive supervisor不停止外部任务，
  只管理自己精确PID/cgroup下的worker。算法利用率报告必须排除filler duty。
- 完整unit、运行代码和生命周期管理器位于仓库外的
  `/mnt/3fs/xudong/gpu-utilization/`；有效service不含coevolve路径/PYTHONPATH。
  一次`install`后由systemd开机自启、5秒崩溃重启，支持临时`stop`、跨重启
  `pause`、`resume`与身份`verify`。后续agent须先读外部README，不能把运行
  脚本移回仓库或回退到手工legacy drop-in流程。
- 保存已有权重、算法和分析；清理中间产物必须先有经过验证的保留清单。
  旧约826.5GB checkpoint inventory仅做过盘点，不能声称已安全清理。
- 使用既有测试；仓库导出不是新的GPU实验。提交前禁止带入任何模型文件或凭据。

外部来源：[Harness-R1](https://github.com/DeepExperience/Harness-R1)、
[官方模型卡](https://huggingface.co/ShaoShuai0605/Harness-R1)、
[WHALE复现说明](https://github.com/krafton-ai/WHALE/blob/main/docs/reproducing.md)。

## 8. 本次代码导出验证

提交范围的CPU回归：**448 passed、5 skipped、843 subtests passed**。
本轮global-selector/cache恢复改动另通过完整`whale_local/tests`：
**135 passed**；其后失败cache过滤的定向回归为**18 passed**。
centralized-context、1-D trajectory、partial packet和精确消费ledger相关定向
回归最新为**32 passed**。
本轮ALFWorld预算耗尽、evaluator及domain集成回归为**34 passed**；
live step29评估也已验证预算耗尽按零分计入，而不是使整批崩溃。
外部filler策略/unit为**9 passed**，coevolve身份校验与ALFWorld回归为
**35 passed**，合计仍为**44 passed**；实际GPU又通过
空闲85.25%利用率、外部负载让载/恢复、worker与supervisor故障恢复和
98.53%合成训练吞吐比验证。最终full-yield/8GiB版本已在node3/node4
active+enabled；多次空闲快照约85%–88%，完整unit安装后的最新均值分别为
85.25%和85.5%。release audit各识别8个verified filler且无foreign PID；
node4外部SM=89%时本卡worker/context归零，外部进程退出后自动恢复。最终
standalone full-yield策略的30秒TF32 ABBA为active均值84,221.5、off均值
84,900，吞吐比99.20%、观测开销0.80%；这是计算代理而非完整训练A/B。
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
