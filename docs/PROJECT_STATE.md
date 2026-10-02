# 项目状态与 Agent 接手指南

## 当前进行：Matched GRPO，然后联合异步MA-SAO

2026-10-02 17:09 UTC的新请求尚未完成，不能因旧测评完成而结束。
先从原SAO的同一个H0续训Harness-R1 GRPO，精确匹配24提示*8=192条；
固定W0，6个32条rollout、24个optimizer minibatch step，保存96/192条点。
新增`alfworld_matched_grpo`复用SearchQA现有FSDP trainer与官方hash-pinned
Relax objective，显式适配原生Qwen3.5 VLM checkpoint和既有ALFWorld reward服务。
属于后端适配，不是原作者完整Relax运行栈。实现验证中，尚无新训练成绩。

之后必须完成新Engineer在W0/W1上的200题单次测评，再继续联合MA。
MA从H1/W1起步，第一目标超过W1+h2的92%，第二目标同H+W累计预算超过；
需要真实联合更新、异步行为概率审核、崩溃预警与incumbent保护。
旧joint试跑没有有效native更新，staleness=0骨架不能冒充已完成异步MAPPO。

**18:34 UTC恢复记录：** GRPO v2在TP2 custom-allreduce启动阶段失败；
v3的TP1引擎卡在TCPStore，py-spy及TCP探针确认网卡地址拒绝连接而127.0.0.1可达。
均没有完成首轮或optimizer commit，失败目录保留。修复`3ebdcfa`已推送，
当前源码`runs/source-snapshots/alfworld-matched-grpo-v4`，
训练`runs/alfworld_matched_grpo_192_v4`，自动评测`runs/alfworld_matched_grpo_200_v3`。
node3服务`coevolve-matched-grpo192-v4-target`、`coevolve-matched-grpo200-v3-w0`；
node4对应`coevolve-matched-grpo192-v4-train`、`coevolve-matched-grpo200-v3-w1`和summary。
启动命令成功不代表真实训练；下一步须核验首个rollout、非零更新及192消费收据。

新[joint_campaign](../whale_local/sao_bridge/joint_campaign.py)接入等待GRPO比较、
联合异步训练、健康止损、共同checkpoint恢复、双节点匹配/后期200题报告及持久deadline guard。
用户17:41:31要求6小时结论：**23:41:31 UTC报告、22:41:31训练截止**。
48项CPU测试及14个subtests通过，含原生PPO配置解析与同温度固定reference、
joint元数据批处理、健康条件、已提交checkpoint过滤。MA仍未GPU验证，不得称为完成。
准备新运行目录时先执行`joint_campaign --mode prepare`冻结数据与截止时间，
随后持久部署editor/solver、matched/best及guard，不能只写计划不启动。

## 最新完成：实际W2+h2的200题补测

2026-10-02 16:48:39 UTC完成，约11分钟。输出
`runs/alfworld_w2_h2_200_v1`。实际候选W2最终step79，75次新增Actor更新，
不是此前被拒绝后保留的W1。两节点各100题，完整重新测200题，每题一次，
同seed/temperature1/4096总token、同一冻结h2；W2/W1权重hash确认不同。

- W2无harness：**50.00%（100/200）**；严格197题50.76%。
- W2+h2：**63.50%（127/200）**；严格197题63.96%。
- 参考W1+h2为92.00%；差**-28.50pp**，题级95%区间[-35.50,-21.50]pp。
- 旧选择阶段拒绝W2的决定得到较大样本支持；不能声称W2完成有效进化。
  原因仍未定位。两节点报告来源/seed/token/覆盖和释放审计通过，filler保留。

用户也问样本效率和代码来源：H1实际32native消费192条（162Actor+30warmup）。
官方GRPO文档示例174*4*8=5568条，仅是名义参考配置；局部1/29比例
不证明效率更高，尤其我们的起点继承官方已RL训练Engineer。
SAO上游fooSynaptic/Single-rollout-async-Optimization明确是**非官方实现**，
基于固定版本inclusionAI/AReaL，加本项目适配/修复，不是原作者官方代码。
模型起点及各阶段checkpoint表已补入根README。MA仍暂停。

## 前次完成：200题每题一次，排除本项目训练实例

2026-10-02 16:04:13 UTC完成。输出`runs/alfworld_core_200_oneshot_v2`，
新测176题，加旧24题固定sample0，共200题；冻结模型/harness，不重选。
W0 **73.50%**、W0+官方 **83.00%**、W0+固定H1 **74.00%**、
W1 **87.00%**、W1+H1 **90.50%**、W1+H2 **92.00%**。
严格剔除训练目标组的197题分别为
73.10%/82.74%/74.11%/87.31%/90.36%/91.88%。

全部200题无本项目训练实例重叠，3题291/414/467仍有训练目标组亲缘；
单列197题，不把200题都说成goal-unseen。197题含179个不同目标组。
原24题参与选择，其他任务也有开发评测历史；83个原封存任务本次已开封。
不追溯改写旧报告，也不把这些83题在以后重新称为封存测试。
目前核心结论：权重收益+13.5pp最明确，固定H1对W0仅+0.5pp；
H2比H1高1.5pp但区间[-1.5,+5]pp跨零，尚无稳定显著优势。

v1两节点均因worker连接ReadError->502失败，未变成零奖励。
一个同配置W1重试在启动期中止，避免继续消耗；原失败记录全部保留。
57942d1关闭worker连接复用而非重试不确定动作，真实TCP复现/修复测试通过，
18测试+6subtest通过；相同题目/模型/harness/seed/预算/并发在v2完成。
总计从首次启动30.5分钟，恢复轮16.6分钟。两个GPU服务和CPU合并器均结束，
GPU释放审计通过，filler继续active。汇总器应在两侧LAUNCH收据存在后启动，
跨节点读取只开放本次专用结果目录的读取/遍历权限，不修改历史目录。
`COMPLETE.json`、`COHORT_AUDIT.json`、`RECOVERY.json`与`merged/`可核验。
MA仍暂缓，没有新增联合训练成绩。

## 此前完成：24题同口径核心数据表

2026-10-02用户将起点、官方Harness-R1、单独H1/W1、W1+H1/H2的核心表
提到第一优先级，并明确授权node3/node4全力并行补测。
输出`runs/alfworld_core_quick_20261002_v1`，统一24题mean@2、
temperature1、总completion预算4096、seed6000045。
node4的`coevolve-core-quick-20261002-v1-w0-w1`已完成：
W0空harness **68.75%**，W0+固定H1 **75.00%**，
W1空harness **87.50%**，W1+同一固定H1 **91.67%**，
W1+固定H2 **97.92%**。题目与sample/rollout_seed配对校验已通过。

node3的`coevolve-core-quick-20261002-v1-official`使用已校验的
`agent-sft-harness-r1`，8个新提案全部合法。与冻结E15的8提案使用同样
W79历史失败输入、提案seed及decode预算；在同一W0上重新测24题。
源码d2e306a，[快速比较入口](../whale_local/sao_bridge/alfworld_core_quick.py)；
无效和重复提案槽保留，候选均分与8选1的选择分分别报告。
全部8个官方候选已完成必需测评：平均**80.21%**、最优
**85.42%（41/48，proposal05；proposal06同分）**。
节点3复测的W0空harness68.75%、固定H1 75%与节点4完全一致。
核心表已完成并存为`CORE_TABLE.json`，包含所有48条样本覆盖验证、
配对seed校验、来源hash及题级区间。
15:08 UTC额外E15候选诊断完成并释放GPU：H1的8提案均分**69.53%**、
最优**81.25%**、合法7/8；官方对应为80.21%/85.42%/8/8。
相同输入/提案预算/目标Solver下，H1本次仍不如官方，不能将总系统收益
解释为H1 Engineer超越官方。`ENGINEER_MATCHED_DIAGNOSTIC.json`保存完整
摘要，18个逻辑variant的原生覆盖/token/seed检查通过。
旧H1的75.71%→80%来自70题、greedy/per-turn预算和8提案均值，
不能并入本次固定harness、总token预算的快速列。

MA两次joint启动均未产生native更新：首试在生成提案后被嵌套Launcher
误判同一实验的兄弟GPU进程为foreign而停止；dd543db修复为共享父Launcher，
并以文件锁保留首个错误，28测试及3subtest通过。r2于14:42 UTC因用户
新优先级主动停止，当时0个native更新、0个配对完成episode。
两次原始记录全部保留，不自动重用部分目录。两节点filler始终保持active。

## 暂缓任务：E15/W79 联合轮次MA-SAO首版

2026-10-02 14:21 UTC用户要求尽快得到与已验证H1/W1可比的MA-SAO结果，
不再把长阶段H→W交替加静态critic标签称为MAPPO式联合训练。
新入口是[联合实验控制器](../whale_local/sao_bridge/joint_experiment.py)，
输出`runs/alfworld_joint_ma_pilot_v1`，冻结源码`1b2fd3f`。
14:31 UTC已启动持久systemd服务：node3的
`coevolve-joint-ma-pilot-v1-editor`，node4的
`coevolve-joint-ma-pilot-v1-solver`及等待两侧完成的
`coevolve-joint-ma-pilot-v1-evaluation`。GPU filler保持开启。
启动不等于训练完成：后续启动故障及主动暂停见上面的最新优先级记录。

- E15/C15与W79/C79起步；128个不同train任务，16轮×8个合作episode；
  每角色4轮critic预热、12次Actor更新，轮间不重置Adam。
- 每轮必须等待两个rollout策略版本都发布；Engineer真实采样harness，
  Solver使用实际应用该harness的官方worker执行，双方获得相同最终奖励。
  本版staleness=0，保留各自完整行为概率及concat action masks。
- Engineer Actor LR1e-6、Solver5e-7，critic5e-6；每轮一次原生SAO critic更新。
  动态动作前critic特权上下文不进入Actor。尚无MAPPO ValueNorm，
  不能宣称完整MAPPO复现或MA优于SAO。
- 自动生成最终Engineer的8个候选，在24题mean@2选择harness，
  随后对冻结H1/W1和联合候选重新测70题mean@2；负结果也必须报告。
  两角色服务各有3小时上限，等待评测服务5小时，不自动重启部分训练。
- CPU测试26项及3项subtest通过；Git校验对确切canonical路径逐命令授权，
  真实Launcher改写HOME的节点3子进程已通过。不得依赖交互用户Git配置。
- 接回首先检查`joint/*FAILED.json`、双方`optimizer/metrics.jsonl`、
  `joint/round-*/{editor,solver}-receipt-*.json`及最终
  `evaluation/COMPLETE.json`，不要把GPU filler利用率当作训练进度。

### 已结束的8小时交替实验：不是有效MA对照

窗口05:22:03–13:22:03 UTC，实际训练与报告在09:33:45结束。
普通SAO H2完成32 native/27 Actor更新并选择step31；
W2完成80/75，但step39/79分别81.25%/75.00%，低于W79+H2的
97.92%选择分，两者均拒绝，最终仍用W79。
预声明70题mean@2终评为**W79+H2 96.43%（135/140）**。
另一次同W79矩阵为no-patch91.43%、H1 95.00%、H2 97.86%；
H2−H1为+2.86pp，题级95%区间[-1.43,+7.14]pp，不显著。
96.43%和97.86%是同一最终策略的不同serving结果，不挑高值替代主报告。

MA两次启动均因Launcher改写HOME后的canonical Git ownership校验失败，
没有任何native更新。其95%是回退H1/W1，不是训练后的MA成绩，
两臂完成预算不匹配，禁止据此推断MA算法优劣。
普通H2的合法提案数从初始7/8降到step15的1/8、step31的3/8；
非法槽执行no-patch并计分，候选均分提高不等于编辑可靠性提高。
W2此次没有4096-token耗尽，不能套用更早的截断崩坏解释。

## 历史部署记录：E15/W79 的8小时自动交替实验

用户最新要求优先部署离线也能完成的实验，而非继续延长旧W0训练。新入口是
[自动控制器](../whale_local/sao_bridge/alfworld_alternating_campaign.py)：node4
`sequential`为普通SAO对照，node3为`ma`。两臂均从已验证的E15 Engineer、
v4-W79 Solver及各自匹配critic起步，都执行H2 → W2。不要把旧实验保留的
W0或旧的不同更新顺序带入本轮。

运行目录为`runs/alfworld_alternating_e15_w79_8h_v1`。冻结协议保存绝对8小时
截止和服务名；先共同收集W79真实train-only失败证据及E15初始harness。
H2为32/27 native/Actor更新，W2为80/75；至少留最后1小时做70题mean@2终评。
独立systemd timer在协调器失效时仍能终止本实验计算服务，GPU filler始终
不在停止列表。截止仅保留已验证状态，部分预算未完成时不宣称MA胜出。
不做多seed/参数扫/交叉harness消融，不使用node0–node2。
实际启动时间及接回方式见根[README](../README.md)与运行收据。
本次窗口已冻结为2026-10-02 **05:22:03–13:22:03 UTC**，12:22:03训练截止。
源码`1b365c4`，node4协调器`coevolve-alt8h-e15-w79-v1-sequential-coordinator`、
node3协调器`coevolve-alt8h-e15-w79-v1-ma-coordinator`均已启动，5个独立expiry
timer已注册。05:30 UTC真实验证：W79的8/8服务健康，15/24个官方10题训练
baseline批次已完成；MA在等待同一份共同初始化，不是已开始H2训练。

旧两臂均已完成native预算但保留同一W0/E15/H1。MA W2选择为45.83%/75.00%，
均低于83.33%的incumbent；step39端口/取消修复后获得真实分数，step79的
一次worker ReadError在原有第二次尝试中恢复。相同最终策略的分差不能视作
MA进化收益。官方对口`agent-sft-harness-r1`权重已校验下载完毕，但其GPU
基线评测推迟到本次8小时实验之后。
旧`PAIRED_REPORT.json`已完成：87.86%对89.29%，差+1.43pp，题级95%区间
[-1.43,+4.29]pp；两臂权重与harness完全相同，gate为false，不能称为MA提升。
旧顺序臂成本审计已完成；旧MA的`cost-r2`退出失败，成本审计仍待修复，
不能把缺失成本记成0。它不阻塞当前8小时实验，也不应占用当前实验GPU。

更新：2026-10-02 UTC。这是当前研究主线的入口；历史细节见
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

**ALFWorld现已有单seed匹配评测证据：SAO更新后的9B Engineer在独立70个
任务上，候选mean@8由E0的72.14%提升到E15的80.00%，+7.86pp，题级95%区间
[+4.46,+11.25]pp，且优于no-patch的75.71%。这不是只找到一个好的E0 patch。
9B Solver v4也已通过原单seed门槛：固定空harness，独立91题mean@2从
78.02%提升到89.56%，+11.54pp，题级95%区间[+3.30,+19.78]pp。
两侧学习证据现在成立，可以进入新的公平SAO/MA-SAO对比；尚未证明MA更强。**

**新的固定H1续训也暴露出SAO稳定性问题，不能把上述正证据说成“任意配置
都稳定”。** 顺序臂W1的step39/79在24题mean@2上，无论no-patch还是H1
均为0%，全部评测episode耗尽4096-token预算。原W0/H1没有被替换，最终
顺序基线不会被偷换成这个崩坏checkpoint。MA同预算W1为75%/77.08%，
同样未超过83.33%控制组。最终比较仍须等完整W2和独立于本轮选择的report；
即使有最终MA收益，也不能借此声称顺序SAO已收敛或已解决稳定性。

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
  不能替代held-out学习收益；这项早期失败不因后来v4成功而改写。
- 最新9B Solver v4的75次Actor更新通过goal隔离的独立report；23个题级均值
  改善、8个退步。这是快速单seed探索的正证据，不能覆盖早期双seed负结果，
  也不代表对训练seed稳健，或已校正自适应方法搜索。
- 最新9B Engineer快速实验选择step15，即11次Actor更新后的权重；E0/E15
  各8/8合法候选，在相同训练证据与decode预算下通过独立report门槛。
  这是单seed、固定8个训练failure prompts下的未见任务收益，不是多seed
  稳健性或未见failure-packet迁移结论。step31退化与3/8非法输出也保留。

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

Engineer验证完成后，node3的下一项为不同方向的
[v5保守更新/资源重平衡](../examples/alfworld_solver_fast_v5_s42.json)，不是v4的
重复seed：fresh W0/H0，8步critic校准+40步Actor，batch12，6 learner+
2 rollout GPU，Actor LR1e-6，critic LR1e-5，checkpoint23/47。共576个不同
train goal，Actor使用480个；评测任务、mean@2与探索gate不变。
动机来自v4原生计时：Actor/critic更新分别均值28.37/27.92秒，是串行阶段；
rollout等待均值4.26秒、中位数约0.001秒。应先把更多GPU给学习端，而不是
继续增加rollout卡。6/2的速度和质量提升尚须实测，不能用配置推断成功。
这是同时调整分配、batch、LR、warmup和步数的探索，不是单因素消融。
若复用v4的共同W0评测，必须绑定原配置/模型/结果SHA，保存显式reuse receipt，
不得将缓存当作新采样或新计算；训练始终从官方W0重新开始。
v5使用独立的detached Git worktree执行冻结版本，主仓库仍可开发下一个方案。
worktree仅共享只读模型/数据、已固定的upstream依赖以及节点runtime cache；
其controller、子进程和CPU monitor的`PYTHONPATH`都必须指向同一worktree。
upstream依赖仍是共享的，运行期间不得修改；不能只冻结入口却让子进程加载
主仓库的新实现。运行收据记录worktree、commit和所共享依赖的身份。
首次v5启动的两个attempt均在任何native update之前失败：通过symlink导入
AReaL时，Hydra按真实文件定位caller，却按symlink路径计算相对config目录。
[Launcher](../whale_local/sao_bridge/launch.py)现先canonicalize依赖的PYTHONPATH；
主worktree的本地代码路径仍保留，回归测试用真实Hydra解析外部配置。
旧attempt与错误保留；恢复须用新的冻结源码/显式运行收据，不重置旧失败记录。
node3的CPU monitor输出必须放在其拥有的run目录内，而非node4拥有的runs根。
两节点只给当前process设置精确共享upstream的Git信任，不改global配置。
修复后的显式retry在node3运行：
`runs/alfworld_solver_fast_v5_r1_s42/`，service
`coevolve-9b-solver-fast-v5-r1-s42`和`coevolve-9b-solver-fast-v5-r1-monitor`；
冻结代码为`runs/source-snapshots/solver-fast-v5-r1-31c66f4/`。
`INFRASTRUCTURE_RETRY.json`绑定旧失败记录并最多授权两个新attempt。
环境和模型已ready，20:32UTC累计20个native step，其中12次Actor更新全部
非零，critic target仍在[0,1]，manifest匹配、没有controller重启。
前12次Actor阶段均值35.33s、critic34.84s、rollout等待11.72s；
batch从8增到12，不能只按单步时间判断快慢，也不能提前宣称完整6/2实验
加速/质量成功。结果输出在该run的`result.json`，不是runs根。

v4已完成80个native step、75次非零Actor更新，critic targets始终[0,1]，
训练只有一个attempt。step79评测连续因HTTP400失败，根因不是权重加载：
生成的tool-call arguments含不完整JSON，下一轮vLLM重渲染历史时触发
`JSONDecodeError`。修复前native审计与完整源码存于run下的
`TRAINING_NATIVE_AUDIT_BEFORE_EVAL_RECOVERY.json`和
`recoveries/invalid-tool-arguments-r1/`，未重新训练。

[ALFWorld适配器](../whale_local/sao_bridge/alfworld_domain.py)现将无法往返
传输的非JSON/object工具参数记为当前episode的零分模型格式错误，保留原始
assistant消息、token数和错误，并取消session；不修补动作、不把HTTP/环境
异常填零。**这是显式、保守的格式错误终止规则，不是纯基础设施修复**：
官方worker本来可反馈无效工具调用并继续，但该历史无法通过当前vLLM模板。
合法工具调用和既有token-limit行为不变。已逐项审计完成的W0与step39缓存，
分别7,886/3,099次工具调用没有这种格式错误，故缓存不受新规则影响。
源码/monitor身份必须显式迁移并保留旧版本，不能声称全程源码哈希一致。
51项CPU回归和25个subtests通过。恢复后selection中1条、report中2条格式
错误轨迹均按零分保留，并非删除样本；其它HTTP/环境错误仍导致显式失败。

**v4完整结果已发布并由独立monitor通过原gate：**

| split | W0 | step39 | step79 |
|---|---:|---:|---:|
| selection59题mean@2 | 66.95% | 66.10% | 79.66% |
| report91题mean@2 | 78.02% | 未打开 | 89.56% |

step79独立report +11.54pp，题级95%区间[+3.30,+19.78]pp。首次launch到完成
199.82分钟，包含训练、评测和失败修复；恢复controller的22.96分钟不是整体
耗时。最终审计
`runs/alfworld_solver_fast_v4_s42/FINAL_NATIVE_EVIDENCE_AUDIT.json`独立重算配对
report，并确认原训练、缓存SHA未变化，80步/75次Actor更新、两个完整
checkpoint及当前GPU释放。主gate结果为`runs/alfworld_solver_fast_v4_result.json`。
此前v1双seed平均-2pp、v2平均0pp仍是必须报告的负结果；v4同时改变了训练池、
预算和LR，不能据此识别唯一关键因素。当前已可研究MA-SAO，但不能把两侧
分别变强直接当作交替训练更强，更不能与旧低预算顺序臂作不匹配比较。

进入MA准备阶段，node4已完成
[强Solver训练证据probe](../examples/alfworld_strong_solver_train_probe_v1.json)
从已验证的W79采集legacy train 0--239的当前失败轨迹，与node3训练并行。
240/240条轨迹完整，191/240成功（79.58%，仅训练诊断），耗时17.97分钟；
1条环境`agent invalid action`保留，完成与释放收据齐全，剩余GPU进程只有
用户filler。它不是相对W0的配对提升结果。
固定空harness、greedy、每episode总4096 tokens，只用于训练反馈与
counterpart-refresh实现，不打开新的report/test、不算作MA优势。
这些train goal已从Engineer的24/70评测中排除。旧packet baseline的token/
serving协议不同，不能直接把两个train准确率相减；若需要配对必须重采相同
协议的W0。新的
[强SAO/MA共同前缀探索协议](../examples/alfworld_strong_sao_ma_v1.json)
声明相同E15/W0起点、H2-W1-W2对W1-H2-W2、每W块80步和H2的32步。
两臂optimizer重置次数相同，MA critic只用train calibration。
**不启用旧replay/interaction/recycling**：正确terminal模式明确要求fresh
完整轨迹，不能为接入旧MA组件重新打开已修正的bootstrap错误。
24/70仍是用过的development数据，不是新sealed test；当前只是预算匹配的
强续训探索，不预先声称到收敛或MA更强。完整说明见[MA方案](MA_SAO.md)。
协议已冻结到`runs/alfworld_strong_sao_ma_v1/frozen_protocol.json`。
第一次共同H1选择启动在import阶段失败：新worktree遗漏根目录WHALE link，
没有启动GPU模型或采样；`common/INITIAL_STARTUP_FAILURE.json`保留完整记录。
另已补齐matrix与环境服务的显式shared-cohort接线，普通训练仍强制ID分区。
85项CPU测试和32个subtests通过；恢复必须保留9个标签/8个原E15提案、
原seed/预算和24题cohort，不能把零计算启动失败当作算法负结果。
`selection-r1`真实注册9个worker、启动模型并通过HTTP健康检查，但随后并发
start_sample触发404。根因是旧minimal controller在async路由中同步请求worker，
阻塞heartbeat，超过60秒alive期限；不是9B能力问题，未产生完整候选分数。
新bridge async controller不改共享upstream、不把错误填零。
真实官方worker CPU压力验证：128次start+128次cancel全部成功，
burst48.32秒、heartbeat持续ALIVE、残留session=0、Solver生成次数=0；
证据在`common/async_controller_preflight/COMPLETE.json`。下一次matrix恢复必须
沿用原提案/seed/预算并保留上述失败目录。

`selection-r2`已完成9个harness x24题x2样本，13.44分钟，全部432条结果的
ID/seed/token cap/二元reward已独立核验，且已释放GPU、filler active。
按事前规则选择`proposal04`为共同H1：selection83.33%，no-patch68.75%；
这是**用于选优的selection结果**，不能当独立report增益。
共同E15/W0/H1及原始前缀成本发布到`common/INITIAL_STATE.json`，
`FINAL_MATRIX_AUDIT.json`保存审计。顺序臂H2和MA臂W1的持久phase入口已实现，
真实E15 tokenizer容量验证为11829<=16384 tokens；MA W1的train-only critic
context为198<=1024 tokens。尚不能声称两臂已训练完成或MA已更强。

v5速度/稳定性分支已完成，**学习gate未通过**：selection W0为66.95%，
step23为59.32%，step47为67.80%（仅+0.85pp，低于+2pp选择门槛），因此保留W0，
**没有打开训练checkpoint的新report**。其`paired_report`的0pp只是W0对W0，
不能说训练后report测出0pp。48个native step/40次非零Actor更新已审计，
optimizer活动窗口76.05分钟，含模型启动的training为79.78分钟。
6/2、batch12、更小LR和更少Actor更新共同改变，不能归因于唯一因素；新强
SAO/MA仍使用已验证的v4级80步/75Actor配方。

两臂第一块已真实启动（不是只准备配置）：node4
`coevolve-strong-sao-v1-sequential-h2`从E15/critic15对W0续训，
node3`coevolve-strong-ma-v1-w1`用共同H1与train-only centralized critic训练W0。
顺序H2已经完成32步/27次非零Actor更新并释放GPU；MA W1已观察到41步/
36次非零Actor更新，仍在运行。两侧filler始终active。
整臂持久控制器会按冻结顺序继续proposal/checkpoint选择、必要的train-only
counterpart refresh与后续W块，不重复首块，不提前打开report。
这些执行器的46项CPU测试/23个subtests通过；两臂的最终比较尚未完成。
MA W1随后已完成80步/75次非零Actor更新，critic target和warmup隔离审计通过；
这证明当前CTDE+concat+terminal接线真实跑通，但还不是质量提升结论。
顺序H2的匹配生成保留全部结果：incumbent6/8合法、续训step15为4/8、
step31为0/8，出现明显过训退化，不能只展示较好checkpoint。
仍按冻结规则选择，最终report尚未打开。
报告执行器已实现并测试：必须两臂同时冻结后才能打开70题mean@2 report；
真实未冻结实验上的拒绝门也已验证。整套相关50测试/23 subtests通过。

最新phase选择结果：顺序H2的incumbent/step15/step31 proposal mean分别为
73.44%/68.75%/68.75%，没有提升，保留E15和共同H1（83.33% selection）。
MA W1在固定H1上的step39/79分别为75.00%/77.08%，低于83.33%控制组，
因此保留W0；**不能因为训练75次真实更新就宣称MA提升**。W没有被接受，
MA H2正确沿用W0原packet，并没有伪称发生新的counterpart refresh。
顺序W1现已完成80个native step/75次非零Actor更新；MA H2也已完成
32步/27次非零Actor更新，先前较旧的状态文件不代表训练卡死。
顺序W1两个候选均出现真实token-limit失效，故保留W0/H1，现已启动完整W2。
MA H2的incumbent/step15/step31分别8/8、8/8、2/8合法提案；
其matrix首试完成16/20个unique harness后遇到worker transport ReadError/
HTTP502，没有把错误记作零分。原训练与16份完整报告保留，systemd已自动
启动第二次matrix；没有重新训练，也没有重采样Engineer提案。
第二次matrix现已完成：incumbent/step15/step31均值为
71.61%/71.35%/69.79%，H2未被采纳，继续保留E15/H1。MA也已进入W2。
两臂第一轮都没有接受counterpart变化，不能把阶段顺序不同等同于已经
发生了有效共适应；两侧W2目前都从同一个W0/E15/H1起点训练。
**04:11 UTC快照：顺序W2已完成80步/75次非零Actor更新并进入checkpoint
选择；MA W2记录79步/74次非零Actor更新，仍在原native阶段。**
两臂尚未最终冻结，report仍未打开。该快照不是实时状态；接手时先读
`ARM_STATUS.json`及完成/释放收据，不重复已完成的W2训练。
**04:41 UTC更新：顺序W2选择完成，step39/79为62.50%/56.25%，均低于
83.33%的W0/H1控制组，整臂冻结为原W0/E15/H1。** MA W2也完成80步/
75次非零Actor更新，但选择先后遭遇两次vLLM内部端口`EADDRINUSE`、
一次真实取消超时后的worker容量HTTP406；这些没有作为模型零分。
当前MA为`coevolve-strong-ma-v1-arm-r3`，冻结`a4eef56`源码，
复用原W2，尚无最终选择/配对report。`c0aa909`隔离replica内部端口；
`a4eef56`使用官方`cancel_with_notice`并等待完成通知，避免取消未完成
就释放客户端episode槽位，且不再忽略取消HTTP502。51测试/28 subtests、
128个真实worker start/cancel后0剩余session验证通过，无模型生成。
`STARTUP_RECOVERY.json`及`CANCELLATION_RECOVERY.json`分别授权一次额外
基础设施尝试，绑定请求与失败log/hash；原自动max2确实增加了两次特批
恢复，不改变native预算/模型/任务/seed/采纳门槛。全部失败和部分生成
仍计成本，不重写历史收据。

当前report/cost均已在未打开report时迁移至`a4eef56`：
node4`coevolve-strong-sao-v1-sequential-report-r2`与
`coevolve-strong-sao-v1-sequential-cost-r2`，node3
`coevolve-strong-ma-v1-report-r2`与`coevolve-strong-ma-v1-cost-r2`；
旧report/cost及r1仅为历史服务。每臂`REPORT_PORT_MIGRATION.json`、
`REPORT_CANCELLATION_MIGRATION.json`保留迁移，最终器仍是
`coevolve-strong-sao-ma-v1-finalizer-r1`。下文早先的服务名是历史快照。

完整pipeline已独立于agent运行：两臂`*-arm.service`推进到冻结，
node4`coevolve-strong-sao-v1-sequential-report`与node3
`coevolve-strong-ma-v1-report`等待两臂冻结才评测，node4
`coevolve-strong-sao-ma-v1-finalizer-r1`等待两侧report与release后汇总。
最终输出为`runs/alfworld_strong_sao_ma_v1/PAIRED_REPORT.json`。
所有systemd启动参数、代码worktree和输入SHA均有run内收据；不依赖本次
终端观察器存活。report尚未打开，最终MA增益尚无结论。
成本也已自动排队：node4`coevolve-strong-sao-v1-sequential-cost`与node3
`coevolve-strong-ma-v1-cost`使用冻结`42a521f`源码，仅CPU等待各自report
service自然结束，再验证freeze/report/release并发布每臂`COST_AUDIT.json`。
启动记录在每臂`COST_AUDIT_LAUNCH.json`，不停止其他unit，不提前写成功结果。
这两个等待器已验证无GPU进程；它们不是仍需人工发起的评测或训练。

**当前顺序臂unit为`coevolve-strong-sao-v1-sequential-arm-r2`，不是旧的
`...-arm`或`...-arm-r1`。** W1训练已成功，但原精确区间审计在step26的
`-3.725290298461914e-9` FP32残差上拒绝完成。现在仅审计容差为`1e-6`，
保留原始极值，不裁剪target、不改reward/optimizer。第一次CPU恢复因新增
审计字段与旧H2选择identity形状不一致而停止；兼容修复逐项验证旧绑定，
只另写`target_range_audit.json`，不改旧identity。34测试/21 subtests及
真实H2历史收据的临时副本验证通过。`55a5b31`冻结源码已部署；
原W1的metrics SHA不变、attempt仍为1，已经真实开始step39选择，无重训。
所有失败及两次恢复均保存在顺序臂的`TARGET_ROUNDOFF_RECOVERY*.json`和
failure归档；这些停顿也应计入成本。node3的MA原unit继续运行，未打断。

node3另已排队`coevolve-strong-ma-v1-arm-r1`（冻结`42a521f`源码），使用
`--after-unit coevolve-strong-ma-v1-arm.service`等待原controller自然停止，
不会stop/kill原训练。旧controller若正常冻结，新controller直接返回已有
freeze，不增加训练或评测；若旧精确审计遇到同类FP32边界残差，则在原有
输入和max2 native attempts内恢复。兼容处理也覆盖非首阶段的旧
`PHASE_COMPLETE.json`：重新验证原始native证据，但保持历史phase和selection
收据逐字节不变，不允许借schema升级重写科学绑定。45测试/29 subtests及
真实MA H2 phase+selection临时副本验证通过，9个训练/轨迹实现文件与旧
`1a9af61`相同。`ma/DEFERRED_AUDIT_HANDOFF.json`保留启动/源码迁移记录；
启动后已确认新unit仅CPU等待、原MA PID未变、W2继续非零Actor更新。

最终器在任何report打开前增加了policy identity防误报：
若两臂最终Solver权重SHA和harness SHA均相同，独立serving得到的分差只能
称为评测波动，不能通过MA方法优势gate；原始配对分数仍保留。
`7067440`及`FINALIZER_IDENTITY_GUARD_MIGRATION.json`记录该纯解释性迁移。
上述恢复中用户filler一直active，没有停止或更改参数。
选择门槛应准确描述：当前`select_checkpoint`计算Bonferroni LCB供诊断，
实际采纳条件是至少+2pp点差，不额外要求该LCB为正；最终MA门才要求
至少+5pp且题级配对95%下界为正。这是原冻结实现，不是本次放宽门槛。

成本工具`alfworld_strong_cost`已加入并在真实已完成阶段上验证：
顺序H2声明192行，实际保存198次生成，额外6次是prefetch重复输入；
W1声明640行，实际648次生成、2,156,802个completion tokens，
其中495次token-limit、142次completed、7次非法tool arguments、
4次environment invalid action，全部保留。H2另有116次真实官方reward
evaluation，共1,160个task episodes。不能把声明行数当全部采样成本，
也不能凭相同ID任挑一个receipt作为optimizer实际消费的行为。
最终成本审计要等两臂冻结和report释放；无收据的中断生成只能报告下界，
原始启动至report释放之间的恢复/等待仍全部计入8卡分配时间。
已发布的共同成本片段为：原Engineer训练/筛选/report及中断尝试的完整
窗口23.124939 GPUh，共同H1选择含启动失败、修复、间隙和r2的窗口
12.825886 GPUh，合计35.950825 GPUh。它们物理上只执行一次，但完整
计入每个逻辑臂，不把失败或丢弃的step31搜索视为免费。另行Solver-v5与
strong-Solver probe属于额外R&D，不混作免费的共同前缀。
`W2_MATCHED_INPUT_AUDIT.json`已验证两臂实际W2的640行、模型/critic起点、
harness内容、80步/batch8/seed1000042/LR/4:4布局一致，只有MA启用集中
critic上下文；W1/W2无ID重叠，共1,280个不同权重训练输入。这是接线审计，
不是W2完成或学习成功的证据。

利用率必须区分filler：顺序H2整个phase实验SM25.95%、filler15.28%；
MA W1整个phase实验SM43.87%、filler7.79%。这些是每进程pmon采样均值，
不是8卡始终占满，也不是纯optimizer窗口。filler全程active。
v5/v4活动时间按Actor实际消费episode归一化仅从9.685降至9.506秒/episode
（约1.85%），按全部optimizer episode则从9.080降至7.922；不同warmup比例
与样本/更新预算导致口径差异，不能称为明显速度突破，且v5质量门未过。

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
数据位于`assets/alfworld_engineer_fast_v1/`。评测流水线现已完成实际GPU验证。
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

本轮H训练已完成32个native step，其中27次真实非零Actor更新；两个actor/
critic checkpoint完整，单attempt，完成时源码身份匹配。critic targets范围
约[-0.70000005,+0.40000001]，符合本轮有符号delta reward，而不是Solver的
[0,1]区间。optimizer窗口真实实验SM33.16%、filler7.99%，filler始终active；
峰值显存127,844MiB，不能据压缩prompt就盲目把3-way learner改成2-way。
证据为`runs/alfworld_engineer_fast_v1_s42/TRAINING_NATIVE_AUDIT.json`。
这些native数据只证明训练机械过程正确；能力结论来自下述独立匹配评测。

[匹配评测器](../whale_local/sao_bridge/alfworld_engineer_eval.py)已实现独立持久入口：
```bash
PYTHONPATH=whale_local /dev/shm/sao-venv-ppo/bin/python -B \
  -m sao_bridge.alfworld_engineer_eval \
  --spec examples/alfworld_engineer_fast_evaluation_v1.json
```
它等待native training audit，再以3个Engineer单卡实例并行生成E0/step15/step31
的相同8个提案，同时5卡Solver采集94个稀疏任务的共同baseline；随后释放
Engineer并补齐8卡Solver完成selection与selected report。使用官方
`evaluate_patch`的原生`task_ids`支持，不必为24/70个目标评测整个连续superset。
同一patch的确定性target缓存可共用，但重复提案仍按原次数计入均值。
传输或环境失败只做最多两次有记录的重试，绝不采用官方脚本的infra no-patch
fallback作为分数；非法模型输出则明确记为no-patch并保留在分母。
另报告相对no-patch的收益和`useful_editing_evidence`诊断，避免把纯粹学会
输出非法patch/弃权、少伤害的收益误说成更强修改能力；这不重定义原相对gate。
单元测试覆盖mean@8、70题report形状、稀疏ID、缓存/重试及controller流程。
真实CPU环境用mock API验证了任务240/247的稀疏路由；零模型/GPU调用，不是
准确率证据。收据在`runs/alfworld_engineer_eval_cpu_smoke_v1/validation.json`。

首次匹配评测完成5个baseline shard、全部24个Engineer提案及15个selection
target job后，部分worker因`EADDRINUSE`启动失败。已确认固定worker端口
46172/46196/46244等落在机器的ephemeral范围32768--60999；没有证据确定
当时占用每个端口的具体进程，不能把该故障当作模型零分或归因于某个用户。
修复使用官方已存在的free-port/文件锁allocator，强制避开ephemeral范围，
不修改系统端口设置、不杀未知进程。额外重试仅授权给6个未完成target job，
原attempt、失败日志、旧源码identity和所有成功cache的SHA保留；不重采样
任何Engineer输出，也不修改任务、候选数、seed、token或模型选择门槛。
恢复时跳过已缓存的Engineer模型加载；耗尽重试返回78，避免重复加载GPU模型
后才再次发现预算耗尽。`infrastructure_recovery.json`冻结这次迁移及有限授权。
实际恢复已启动为node3的`coevolve-9b-engineer-fast-v1-eval-r1.service`，
保留并核验91个文件SHA；六个未完成job使用12xxx--15xxx/22xxx--25xxx的
已锁定非ephemeral端口后全部完成。24题mean@8 selection为E0 69.27%、
step15 76.56%、step31 67.19%；按原规则选择step15。step31三条非法候选仍保留
在分母，所有24个提案均未重新采样。

**独立70题report已完成，原gate与useful-editing诊断均通过：**

| 指标 | E0 | step15（11次Actor更新） |
|---|---:|---:|
| 相同8个提案的目标任务mean成功率 | 72.14% | 80.00% |
| 相对共同no-patch（75.71%） | -3.57pp | +4.29pp |
| 合法提案 | 8/8 | 8/8 |

E15−E0为+7.86pp，题级bootstrap 95%区间[+4.46,+11.25]pp；44个题级均值
改善、11个退步。不是只修复输出格式、丢弃失败提案或变得更会弃权的收益。
但bootstrap条件于这8个提案，不覆盖训练seed或未见failure packet。
恢复段18.59分钟，首次评测身份建立到完成共104.63分钟，后者包含失败、重启
和修复等待，不能用恢复段elapsed冒充端到端耗时。
66个完成job、142个证据文件SHA及独立重算的paired result保存在
`runs/alfworld_engineer_fast_v1_s42/evaluation/FINAL_EVIDENCE_AUDIT.json`；
`COMPLETE.json`和更新于其后的`RELEASED.json`确认GPU已交还仍active的filler。
本轮不需要触发“SAO-H失败后再复现GRPO”的条件分支；未运行GRPO，
更不能把这个结果称为GRPO复现。9B Solver随后也已通过v4门槛；当前主要未解
目标转为匹配起点/预算的SAO与MA-SAO harness-weight-harness比较。

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
