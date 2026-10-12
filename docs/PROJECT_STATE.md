# 项目状态与 Agent 接手指南

### 2026-10-12 15:34 北京时间：只上传历史97%的checkpoint

用户取消此前多checkpoint方案，只保存达到194/200的历史时点：
`alfworld_ma_convergence_v1r4/block-01/round0096`，
两角色Actor均为`epoch0epochstep95globalstep95`，对应该路线累计144轮。
不是共同初始化Full144的93%版本，也不是未完成消融native72。
实际范围是Engineer/Solver两个Actor各10份文件，共37679454196字节，
约35.09GiB；模型路径、原评测身份及已有摘要已核对。

15:44:40已完成这20份checkpoint文件的私有HF dataset上传，仓库为
[wxd02/coevolve](https://huggingface.co/datasets/wxd02/coevolve)，不可变revision为
`d0e84414e8ef9e2b6a84585e1365189faf501051`。
远端清单、大小及服务器对象摘要全部匹配本地文件，见
[发布回执](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json)。
用户明确不上传Harness、不下载回验；Critic/Adam/native、其他模型和评测归档也不上传。
97%是Solver与原选定Harness组成的系统成绩，该Solver单独是185/200（92.5%）；
原Harness留在既有Git证据中，不包含在本次HF上传。
完成状态、不可变revision和访问方法见[复现说明](REPRODUCE_MA_SAO.md)。
W&B不写入，所有训练和Full144延长仍暂停。

### 2026-10-12 15:14 北京时间：代码发布与无GPU复现交接

当时工作仅为代码/数据交接和平台凭据验证，实验保持暂停。
入口为[复现说明](REPRODUCE_MA_SAO.md)、
[Git证据manifest](../reproducibility/ma_sao_20261012/MANIFEST.json)和
[模型库存](../reproducibility/ma_sao_20261012/MODEL_ARTIFACTS.json)。
4225份文件、182643105字节的证据已逐文件核验；标准库Python离线命令
`python3 -B scripts/reproduce_ma_progress.py`已重算Full96的180/178和Full144的186/179，
核验双角色76行metrics与native72边界、七份冻结运行快照。

切分复核明确：selection48全部位于main200之内；训练任务0–239与main200无实例重叠，
strict197排除goal-related的291、414、467。这个93%不是独立未触碰测试集成绩。
所有原协议与结果未修改，额外152题统计只作为事后描述，不替换原主终点。

GitHub用户提供的凭据实际是SSH私钥，已验证xudongwu-0身份及目标仓库读取；
HF身份wxd02、W&B身份2373025856w验证通过。当时HF/W&B只允许只读认证，
尚未创建远端仓库/run或上传模型。最新上传范围见上节；本地模型清单
不等于远端备份。

15:14:47北京时间，主交接提交
[d3fb85964c55c705518118d098057e46f8d28ac2](https://github.com/xudongwu-0/coevolve/commit/d3fb85964c55c705518118d098057e46f8d28ac2)
已实际推送到GitHub的main；当时`ls-remote`返回的SHA与本地提交完全一致。
从该提交导出的独立目录已通过无依赖离线复算，37项打包/复算测试通过；
最终4807个Git索引条目核验包含全部4225份payload及13份包元数据，
1249个不同blob已做凭据排除检查。三份训练取消门控仍存在，没有恢复训练。

### 2026-10-12 14:32 北京时间：用户暂停全部实验及延长训练准备

**最新指令是不再准备、启动或恢复实验，只检查node3/node4当前状态。**
此前提出的Full144延长方案尚未实施，不应继续准备或自动插入队列。
当前serial15 self_fix01入口已增加持久
[CANCELLED.json](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/CANCELLED.json)，
实际调用原冻结代码的授权检查已确认拒绝启动；旧idle2h和首个serial入口也保持取消。
下面12:45的计划和绝对完成时间均为历史记录，不是当前执行授权或有效排期。

without_peer在14:25:40收到SIGTERM；最后双角色metrics均76轮，
最新完整原生配对为72轮，已核验。76是已记录训练进度，不能当作76轮可恢复状态。
Full144仍为186/200，未修改。控制器、worker及阶段观察器已退出，
node3对应训练service为inactive/MainPID0。控制目录旧STATUS心跳已停止，
不能只凭其中的training字样认定仍在训练。

14:31:57实际探测：node3八卡均0%利用率、0MiB显存、无GPU计算进程；
node4 GPU1–7同样空闲，GPU0仅其他用户hongyuanzhu的PID1138383，
入口为O-Voxel Stage3训练脚本，显存8092MiB，三次pmon观测SM为21%、18%、19%。
两节点filler service均inactive/MainPID0。未修改或终止其他用户进程。
独立5分钟只读GPU记录器仍存活，不负责启动训练。

### 2026-10-12 12:45 北京时间：串行15卡续训计划与新故障规则

用户要求先报告整体计划，然后立即继续训练并监控。当前优先级如下；T0为首个
实际训练启动时间，表中时长包含恢复、训练及原96/144两点评测，不包含排障停机。

| 顺序 | 实验与剩余轮数 | 检验目的 | 单项估计 | 累计估计 |
|---|---|---|---|---|
| 1 | without_peer，24→144，剩余120轮 | 队友重要性权重贡献 | 3–4.5小时 | T0+3–4.5小时 |
| 2 | without_context，0→144 | Critic队友上下文贡献 | 3.5–5小时 | T0+6.5–9.5小时 |
| 3 | without_value_norm，0→144 | 价值归一化与稳定性 | 3.5–5小时 | T0+10–14.5小时 |
| 4 | always_joint，0→144 | 始终联合更新对比联合/锚定调度 | 4–5.5小时 | T0+14–20小时 |

Full144已有186/200=93.0%，直接保留。always_joint角色轨迹/更新更多，
不是等计算量对照。统一seed1000043、batch8、32768上下文、4096输出上限、
原优化器和损失设置、固定144主终点保持不变。每项训练结束后顺序评测96和144，
各8候选、48题选择集、200题主评测，完成审计再进入下一项。

**分配与可比性：**

- node3/Engineer：8卡，4个训练rank（Actor/Critic及原reference共置）+4个rollout副本。
- node4/Solver：仅GPU1–7，4个训练rank+3个rollout副本；GPU0保留给其他用户，
  不因对方暂时低利用率而将其视作空闲。GPU启动统一逻辑到物理映射，
  占用监控仅排除有明确GPU0归属的进程；跨入GPU1–7、归属缺失或探测失败仍阻止运行。
- 终点评测在node3执行，维持原8副本布局，不为多跑候选而改种子、预算或选择规则。
- 原四rank原生Adam、ValueNorm、RNG和游标恢复；科学spec/hash不改写，
  单独保存实际资源分配及执行授权。Solver rollout由4变3可能改变采样，
  **不能宣称逐token复现或相对原Full完全排除了运行布局的影响**。

**时间和中间状态：**

已完成Full144训练实测10540.58秒（2.928小时），常规轮约48–50秒，
含12轮保存的平均轮间隔约68.7秒。15卡尚未测量，14–20小时只是排期估计。
保持原4-rank learners且每角色至少1个独立rollout副本时，一个任务至少10卡，
两个任务至少20卡；现有15卡不能直接做保持原生布局的双任务并行。
若假设可以做两个均衡小任务，且大分配相对小分配加速S倍，串行/并行总时长约2/S；
S=2、1.8、1.5时的例示损失为0%、11.1%、33.3%，**不是本项目实测值**。
详见[耗时依据](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012/TIMING_PLAN.json)。

当前一个paired native状态约384.36GiB。沿用每12轮保存、先核验新配对再淘汰
运行目录内前一个滚动payload的策略，保存过程中会短暂有两份，约768.7GiB；
最终native及96/144 HF保留。已有交接归档/硬链接不删除，因此原24轮归档仍占空间，
不能把运行目录删除旧链接说成释放了全部物理容量。最终各实验的native仍会累计。

**故障规则与执行状态：**

- 外部kill或资源冲突：停止队列，不自动重启，不复活旧两小时等待器。
- 自身训练错误：保留失败日志/未提交成本，先查明并修复，再从最后完整原生断点继续。
  未知错误不自动循环重试；已完成训练后的评测失败不能触发重训或重新抽取候选。
- 原旧暂停/取消记录保留，当前有效授权单独位于
  [serial15_20261012_self_fix01](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/)。
  在新控制目录创建CANCELLED.json可停止新入口。
- 两节点维持程序已暂停且MainPID=0，PID1138383保留；
  [双节点回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012/FILLERS_PAUSED_VERIFIED.json)。
  初次仅TERM后systemd自动拉起，已经补充显式停止service和持久pause。
- 本段写入时处于冻结验证阶段，尚未创建新的训练attempt；
  实际状态以后续STARTUP_VERIFIED及原PROGRAM_MONITOR为准，不把计划写成完成。

13:20北京时间更新：第一轮attempt-0002在原生启动前置检查中以code1退出，
原因为冻结打包漏掉4份`areal-*.patch`运行时文件，不是外部kill。
新采样/新提交轮次均为0，原24轮checkpoint未改动。原控制目录持久取消，
失败日志与COST保留；使用新的
[self_fix01控制目录](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/)、
新完整serial15-v3快照，不原地修改旧快照/旧attempt。
该快照348项测试、13项subtests通过，追加实际verify_upstreams启动检查，
node3真实UID1009/GID1008的相同检查亦通过。
attempt-0003于约13:14进入双节点原生启动，GPU资源映射实测正确：
node4 GPU1–4为四rank学习器及reference，GPU5–7为三个rollout；
GPU0仅保留原PID1138383。node3八卡已分配给本实验。
**13:20时正在初始化/读取模型，尚未核验双角色RECOVERY_LOADED或新增训练轮次；
复制出的24行历史metrics不是新训练进度。**
控制器为会话托管进程，不是开机自启/自动复活服务；五分钟只读GPU记录仍独立运行。

**13:37:52北京时间更新：双角色已从24轮原生状态恢复并完成36轮，
第36轮新配对checkpoint已发布、核验，当前可恢复状态不再停留在24轮。**
这是单发布者修复后首次真实双节点GPU续训及新配对保存成功，
并非仅CPU或合成验证。两个角色均有RECOVERY_LOADED；
Engineer的Actor/Critic Adam步数从13/32恢复，Solver从21/48恢复，
四rank一致，没有重置优化器或重复warmup。第36轮COMMITTED摘要为
`4bbe5102598bcc336d8a88f4a5fb2cd3e9589625721671061566b37038681fe1`。
核验范围为全部分片大小、小文件哈希、优化器/ValueNorm/RNG与游标元数据，
不冒称对全部大tensor重新做了全文件哈希。
见[首次新增配对核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/FIRST_NEW_CHECKPOINT_VERIFIED.json)、
[原生恢复与新进度](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/FIRST_PROGRESS_AUDIT.json)。

实际T0为10月12日13:14:30。保持原保守排期，预计各项完成窗口依次为：
without_peer当日16:14–17:44，without_context当日19:44–22:44，
without_value_norm当日23:14至次日03:44，always_joint次日03:14–09:14；
均为北京时间，包含终点评测、不含排障停机，**不是完成承诺**。
首批常规轮平均约44–48秒，但不包含保存/评测，不能据此宣称15卡比16卡更快。
见[按实际启动时间换算的排期](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/serial15_20261012_self_fix01/TIMING_PLAN.json)。
当前仍只有训练进度，尚无新的消融终点评分，四项串行实验尚未完成。

### 2026-10-12 11:32 北京时间：独立于对话的5分钟GPU历史记录

用户转为要求独立、只读的GPU历史记录，以便以后请求启动时判断近期占用情况，
并继续自由对话。本次不启动训练；旧两小时自动等待器已经退出，
其控制目录增加持久CANCELLED.json，不会在聊天过程中自行开训。
完整full144和without_peer24轮状态不变。

- 源码：[gpu_history.py](../whale_local/sao_bridge/gpu_history.py)，仅调用只读的
  nvidia-smi、进程身份查询和node3 SSH探测，不导入训练控制器或执行停止GPU程序的命令。
- 数据目录：[gpu_history_node3_node4](../runs/gpu_history_node3_node4/)。
  [RECORDER.json](../runs/gpu_history_node3_node4/RECORDER.json)记录PID、启动身份及程序摘要；
  [STATUS.json](../runs/gpu_history_node3_node4/STATUS.json)记录上次/下次采样时间；
  [LATEST.json](../runs/gpu_history_node3_node4/LATEST.json)记录两节点最新完整样本。
  `samples/YYYY-MM-DD.jsonl`按北京时间日期追加保存，不自动删除历史。
- 每300秒并行探测两节点，各8张卡记录利用率、已用/总显存、功率和温度。
  进程记录PID、UID/用户名、可执行程序、脚本/模块名、每卡分配显存及可用的pmon利用率；
  不保存完整命令参数、环境变量、认证URL或令牌。
- 所有可读时间均为`Asia/Shanghai`、ISO8601 `+08:00`，另保留Unix/单调时钟用于间隔核验。
  真实工作、已核验维持程序、身份不明活动和检测失败分别记录，失败不伪装为空闲。
  已核验维持程序仍要求原UID、路径、cgroup和supervisor身份一致。
- 独立后台进程不会因为当前对话结束而停止；未安装开机自启。
  主机重启、运行环境丢失或进程被外部终止后，需要重新启动。
  新旧进程的记录通过recorder_id区分，空闲连续性不跨进程重启、漏检或检测错误继承。
- 查询最近两小时需要至少25次有效采样且跨度不少于7200秒，最新样本不能过期。
  该结果只是历史证据，不授予自动开训权限，也不能排除两次采样间的短时作业。
  真正启动实验之前仍须复查实时资源。

查询最近两小时：

```bash
PYTHONPATH=/mnt/3fs/xudong/coevolve/whale_local \
  /dev/shm/sao-venv-ppo/bin/python -B -m sao_bridge.gpu_history summary \
  --root /mnt/3fs/xudong/coevolve/runs/gpu_history_node3_node4 --hours 2
```

将`--hours`改为所需历史时长；没有足够历史、出现其他作业或检测失败时，
`meets_requested_window`为false，不把未知时间算作空闲。
创建数据目录中的`STOP.json`可在当前有界探测结束后停止记录器；
不要以它停止其他用户GPU任务。再次运行`record --root ...`前需确认旧记录器已停止，
并明确解除该STOP标记；协调锁防止同一节点出现两个记录器。
历史日志保持原样，不要用人工补写样本制造连续空闲。

CPU单元验证43项通过；首次实测看到node3空闲，node4存在真实GPU工作，
不是只运行利用率维持程序。实际部署PID和第一份样本以
[启动回执](../runs/gpu_history_node3_node4/STARTUP_VERIFIED.json)为准。

### 2026-10-12 02:56 UTC：重新授权，先观察两小时再自动续训

用户要求每10分钟检查node3/node4；两小时内的检查均为空闲或已核验的GPU利用率维持程序，
再自动启动后续训练。这是对10月10日暂停的**新、独立、有限实验范围授权**，
不是删除旧取消记录，也不是重新运行历史队列。

- 控制目录：[idle2h_20261012](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/)。
  [新授权](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/AUTHORIZATION.json)保留旧暂停文件摘要，
  旧文件发生变化或新控制目录出现`CANCELLED.json`即停止此入口。
- 每600秒并行只读检查两个节点，至少13次有效采样并跨越7200秒。
  空闲和已核验维持程序可以交替；其他GPU任务、未分配显存的分布式启动器、
  无法归属的显存/利用率、权限/SSH错误均不合格。间隔超过660秒或系统时钟跳变重新计时。
  控制器重启也从零计时，不把10月12日早先的手工检查当作已等待时间。
- 达标后重新核验代码、原科学输入、断点和当前资源；没有独占资源或预约保证。
  仅认可已有路径、UID、systemd cgroup和supervisor一致的维持程序，
  保留原adaptive协作策略，不结束其他人的训练或任意同名程序。
- 先从without_peer第24轮恢复Actor/Critic、Adam、ValueNorm、RNG和游标，
  再按原顺序运行without_context、without_value_norm、always_joint。
  每12轮保存，96/144保留评测点；健康训练和成功完成后的下一项不人为停两小时。
  若有明确外部终止/资源冲突证据，则清理自有进程后重新观察完整两小时。
  未知代码错误、已完成训练后的评测/报告中断停止诊断，绝不自动重训或重新抽候选。
- 原10月10日截止时间已过。原协议/spec/科学摘要不改写；
  仅在新attempt的`rolling_recovery.execution_authority`中绑定新授权和两小时准入回执，
  让新入口完成剩余四项固定144轮实验或在取消时停止。原每阶段8小时和无进展保护仍有效。
- 监控是本会话托管的CPU后台进程，不是开机自启或脱离会话的服务。
  会话退出/控制器被结束后不会自行复活；再次启动必须重新观察两小时。
  10分钟离散采样不能证明采样之间没有短时作业，报告必须保留这一限制。

运行是否已开始，以[STATUS.json](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/STATUS.json)、
[SAMPLES.jsonl](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/SAMPLES.jsonl)
及原[训练进度](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PROGRAM_MONITOR.json)为准。
暂停新入口可在新控制目录创建`CANCELLED.json`；不要删除历史门控以尝试恢复旧入口。

03:22:54 UTC已核验CPU控制PID1137442存活、持有单控制器锁，首个有效样本为两节点空闲，
计数1/13，GPU工作尚未启动，attempt-0002仍不存在。下一次检查约03:32:54 UTC；
在后续检查一直合格的前提下，准入时间不早于05:22:54 UTC（北京时间13:22:54），
之后还需完成源代码/输入/资源核验。精确执行快照179项测试通过。
见[启动验证回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/idle2h_20261012/STARTUP_VERIFIED.json)。

### 2026-10-10 08:17 UTC：按用户要求停止自动恢复，禁止H200训练，转为本地交接

用户08:16 UTC要求暂停自动重启、不再使用H200；不执行原定08:52续训。
原等待控制/追踪/只读观察PID均已退出，第二次attempt不存在，
两节点GPU列表和分布式启动器均为空，对应原生服务inactive/not-found。
已写入持久取消门控，并实际调用两个恢复入口验证它们会拒绝继续。
无关的既有CPU数据服务未停止，其他用户进程未修改。

完整MA144轮93.0%及所有原始评测保留；without_peer只有24轮已提交原生状态，
没有最终分数，其余三项未启动。当前任务改为归档代码、原始结果、数据及checkpoint，
不是继续消融。GitHub/HF远端同步尚未执行，不能把本地准备写成已上传。

本地交接包含完整工作树代码和依赖改动、5个执行/科学快照、原始ALFWorld数据、
当前实验的可读取证据、历史97%原始评测，以及19组checkpoint的保留/校验清单。
两份原生恢复状态共约768.71 GiB；全部19组约1051.84 GiB。
硬链接/本地复制不是独立异地备份，最终产物以归档校验回执为准。
详见[本地交接和存储说明](./MA_SAO_HANDOFF_2026-10-10.md)。

- [禁止H200及交接指令](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/USER_NO_H200_AND_HANDOFF.json)
- [实际暂停核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/NO_H200_PAUSE_VERIFIED.json)
- [两个自动恢复入口拒绝执行的验证](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PAUSE_GATES_VERIFIED.json)

### 2026-10-10 07:15 UTC：首项消融停在24轮；共享断点并发写入已修复，等待两小时后续训

without_peer在06:52 UTC提交了24轮配对原生断点，随后停止，尚无96/144端点评测。
首个异常是Solver保存日志中的COMMITTED.json.tmp重命名FileNotFoundError，
不是后续日志中的CancelledError；后者发生在退出清理时。控制进程没有记录外部终止信号。
用实际两节点、与训练相同的UID/GID实测发现：node4持有排他flock时，
node3仍能同时获得同一共享文件的排他锁。原实现让两角色同时发布共享元数据，
因此这次是已定位的保存并发问题，不能写成已证实的外部kill。

24轮原生配对已独立核验：提交192个共享episode，H/W Actor更新13/21、Critic更新32/48；
实际落盘199个共享结果、398条角色生成回执，额外7个结果属于未提交预取，不删除或漏记成本。
两个角色的保存指标均连续覆盖0–23，科学身份与原spec一致。
原FAILED、旧源码和所有attempt保留；24轮轻量元数据另行归档，后续仍按协议滚动清理大状态。

新的独立rolling12-v3快照只修改joint_recovery.py：
双方仍各自保存完整原生状态，但仅Solver发布共享OWNER/COMMITTED/history/LATEST及清理旧代，
Engineer等待并验证已经发布的配对指针。Actor/Critic、目标、数据、seed、预算和恢复格式均不改。
66项断点/原生/时序测试、28项驱动/时序测试通过（两套测试有重叠）；
实际双节点CPU合成断点的12→24发布、双边确认及删除旧12轮也通过。
这不是新增GPU验收或性能结果；未改动的载入/优化器路径继续绑定此前真实GPU验收，
首次实际24轮恢复及36轮新配对保存另有只读观察器核验。

新控制PID4102113、信号跟踪PID4102109及只读观察PID4102927均已验证存活、CPU-only，
尚无attempt-0002。按真实退出时间06:52:33.772817起冷却7200秒，
最早**08:52:33 UTC（北京时间16:52:33）**重新验资源及原科学/原生门控，再从24轮续训。
这是明确修复已知代码错误后的单次恢复，不篡改旧失败记录为“可自动重试的外部终止”。
原截止时间仍为10:10:44 UTC（北京时间18:10:44），恢复窗口最多约78分钟；
只能继续推进，不能承诺这项144轮或四项消融能全部完成。完整MA的93%结果不受影响。

- [真实跨节点锁行为](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/CROSS_NODE_FLOCK_PROBE.json)
- [已保存24轮的完整检查](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/SAVED24_AUDIT.json)
- [修复、证据和两小时边界绑定](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/PREPARED.json)
- [新等待器启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/STARTUP_VERIFIED.json)
- [冷却实时心跳](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_MONITOR.json)

### 2026-10-10 06:18 UTC：93%基线报告后，首项“关闭队友校正”消融已实际启动

完整MA分数、历史97%路线和真实失败样例已报告，精确BASELINE_RELEASE门控通过。
首项without_peer使用原rolling12-v2训练源码、同一seed及共同初始化，
没有使用本次93%或历史97%的权重作为起点，也没有修改任何科学配置。
自有控制PID4062818、本地Solver PID4063504及node3自有服务MainPID1731201均已验证存活，
远端active/running；06:18进度仍为H/W 0/0，当前是已启动后的加载阶段，不是消融已完成。

仍每12轮配对保存、验证新代后删除旧代，保留96/144评测节点；
非计划训练中断至少冷却两小时，未知代码错误不盲目重启，不修改其他用户任务。
其他三项尚未启动，原10:10:44 UTC截止时间保持不变，只能在剩余窗口内推进。
基线发布时的三份说明文档已按原摘要留存快照，后续状态更新不覆盖当时报告证据。

- [首项消融本地/远端实际启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/WITHOUT_PEER_NATIVE_START_VERIFIED.json)
- [实时训练进度](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PROGRAM_MONITOR.json)
- [首项消融控制日志](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/WITHOUT_PEER_CONTROL.log)
- [基线精确报告快照](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/postprocess_resume01/baseline-report-snapshot/MANIFEST.json)

### 2026-10-10：完整MA主终点已核验为93.0%，恢复故障只发生在CPU报告阶段

资源等待于05:15 UTC结束，原selection48的82条完整结果全部原样复用，
没有重训、重复已完成题或增加候选。144轮原8个提案仍为7/8有效；
05:40 UTC按原selection48规则固定proposal01，随后才打开main200，05:55 UTC完成。

| 固定保存节点 | 选定Harness + Solver | 同一Solver、空Harness |
| --- | ---: | ---: |
| 96轮诊断 | 180/200（90.0%） | 178/200（89.0%） |
| **144轮主终点** | **186/200（93.0%）** | **179/200（89.5%）** |

**93.0%是本次共同初始化的固定终点结果，没有复现历史97%，也不能据此证明任何组件必要。**
与历史194/200相比，新系统改善1题、退化9题，净差−4个百分点；
配对问题bootstrap区间为[−7,−1]个百分点，但这不是独立训练种子区间，也不是组件因果对照。
历史路线核验确认：H0/W1 Actor起点和1152条有序训练输入一致；
历史使用已训练的MA64 Critics/ValueNorm、48轮边界Adam重启、24次而非16次提案机会，
且97%是开发集选出的峰值，不是本次的预声明144轮终点。
不能仅因分数较低就认定实现损坏，也不能修改初始化或重训来追分。

本次新系统失败14题，历史系统失败6题；9题“新失败/历史成功”、1题反向、5题共同失败。
新系统失败中12题出现至少4次相同行动、13题达到至少50次模型调用；
这些是保存轨迹的可观察现象，不是队友校正、Critic或ValueNorm的因果证明。
4个展示样例按既定规则选择为270、329、342、356，反向题305同样披露。

原评测恢复控制在约06:02 UTC返回1，**不是又一次已证实的外部kill**：
144轮结果与最终结果审计均已落盘，错误是历史失败样例分析误读了不含messages的紧凑摘要。
历史原始200题轨迹实际存在，已通过原matrix审计，并逐字段验证其紧凑投影与冻结97%摘要完全一致。
独立报告源码修正只改CPU审计/数据来源，不改训练或评测；67项测试在精确快照通过。
旧FAILED、旧历史审计、所有结果/日志均保留，新的历史审计另存；结果报告已补齐。
原等待器已经退出，不能再以其旧心跳宣称仍在监控。

selection实际记录494次开始：432个逻辑episode加上此前62个未完成尝试，
复用82条完成结果、重复完成题0次、补抽候选0次；未落盘在途token成本仍为未知。
正式四项消融尚未完成。原截止时间仍为2026-10-10 10:10:44 UTC；
06:12两节点资源空闲，但仅余约4小时，不能承诺在原窗口内完成四组。

- [完整MA结果与两端点评测](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/RESULT.md)
- [93%与历史97%的完整路径审计](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/BASELINE_PATH_AUDIT.json)
- [真实配对失败样例与反向题](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/failure-cases-vs97/CASES.md)
- [只恢复CPU报告的完成证据](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/postprocess_resume01/COMPLETE.json)
- [完整MA及四项消融总表，未完成项明确保留](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/RESULT.md)

### 2026-10-09 15:39 UTC：完整训练已完成，144轮评测中断；等待两小时后只恢复未完成评测

完整MA双方已完成144/144轮，最终配对原生断点已验证，
14:22:34 UTC训练审计通过1152个共享episode；96轮端点评测已完成并保留。
15:01检查时确实处于144轮selection48，不是等待资源。
随后原控制PID3626493及评测PID3792500退出，附着式控制返回1。
推理日志约15:03:12同时结束，但没有原控制/评测FAILED回执或Python traceback。
**目前无法确认终止信号及发送者，不将其写成已证实的外部kill或已知代码异常。**

144轮八个原候选已固定，因重复候选去重后有9个独立Harness，selection48共432个逻辑episode。
原attempt记录144次episode开始，其中82次完整结果已落盘，62次尚未完成；
尚无任何完整Harness报告，144轮SELECTED及最终200题结果均未发布。
所有82条完整结果（含失败）及原始候选/日志/配置已摘要绑定，不重新生成候选、不重跑已完成题，
只恢复剩余350个selection episode，然后按原规则选择Harness并打开最终main200。
未完成62次的实际token成本未知，不记为0，也不漏记这些额外尝试。

首次确认退出时间15:04:50.346 UTC起保守冷却7200秒；
最早2026-10-09 17:04:50 UTC（北京时间10月10日01:04:50）重新检查资源。
15:36两节点已有其他分布式任务，不能占用其GPU；到期仍忙则继续等，不修改外部进程。
附着式等待PID3819570（自有信号跟踪PID3819566）已于15:39验证存活、心跳正常、
没有原生/GPU子进程，也未创建selection attempt-02。原绝对截止时间不变。

评测恢复使用独立源码快照，原训练源码与科学配置不改。
67项针对性测试和13项子测试通过：完整episode跨进程复用、失败结果保留、物理成本计数、
两小时门槛和截止时间，以及恢复后身份审计只接受预声明的attempt路径转换。
同时修复了原完整结果审计把合法rolling路径转换误判为科学配置变化的问题；
实际两角色身份和最终144轮配对状态已独立通过新检查。
新增恢复运行遇到明确错误会停止，不用伪造ReadError或清除旧attempt强行重跑。
正式144主成绩与四项消融仍未完成，等待器或CPU测试不是研究结果。

- [中断观察、固定证据和两小时计划](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/SCHEDULE.json)
- [等待器启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/STARTUP_VERIFIED.json)
- [实时评测恢复心跳](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/MONITOR.json)
- [评测专用源码](../runs/source-snapshots/alfworld-ma-mechanisms-eval-recovery-v1/SOURCE_MANIFEST.json)
- [精确源码测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/evaluation_resume01/CPU_TESTS.log)

### 2026-10-09 11:28 UTC：正式完整MA控制已启动，当前为模型加载

正式full144的attempt-0001于11:26:14 UTC实际启动，恢复位置为0，
不是从历史97%或原生校验权重续训。共同H0/W1、普通SAO raw Critics、
fresh Adam/ValueNorm、seed1000043及冻结科学配置保持不变。
控制PID3626493存活；11:28核实node3自有服务active/running、MainPID1302169，
当前已提交轮次H/W仍为0/0，不能称为已完成训练。

继续每12轮配对保存、验证新代后删除旧代，保留96/144的HF评测节点。
中断后按新两小时冷却规则恢复，未知代码错误不盲目重试；没有修改其他用户任务。
原生校验测得每角色完整原生保存约259至290秒，仅两次保存样本，不保证所有轮次相同。
正式完整结果与历史路径差异须先审阅/报告，再放行四项单seed消融。

- [实际启动回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/attempts/attempt-0001/STARTED.json)
- [本地与远端启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/FULL_NATIVE_START_VERIFIED.json)
- [实时正式进度](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PROGRAM_MONITOR.json)
- [正式控制日志](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/FULL_CONTROL.log)

### 2026-10-09 11:22 UTC：两小时延迟已执行，双节点原生12→15恢复验收通过

等待器没有在10:34:57 UTC之前提交新GPU工作；之后两节点资源检查通过。
原生attempt-0002从原始校验初始化训练至12并完成配对保存，主动停止自有角色后，
attempt-0003实际载入同一个12轮断点，继续至15。11:16:20 UTC原生门控通过，
控制器正常退出；两次校验的node3自有服务均已释放。

11:18独立核验了输入/源码/执行层摘要、两角色加载与完成回执摘要和实际指标：
恢复游标96，global_step连续0至14，无Adam重置、无重复预热；
H/W Actor更新7/12，Critic更新20/30，消费80/120条轨迹，均符合15轮原协议。
两角色载入的配对断点摘要相同。只保留最终15轮大状态，12轮旧代和先前暂停半成品已移除，
轻量历史与所有attempt的指标、结果、成本仍保留。

成功逻辑路径为120个共享环境结果；连同用户暂停的首次96个结果，
本轮原生校验共记录216个共享结果、432条生成角色轨迹。
未落盘的在途成本仍未知，不能把120写成全部物理成本。
这验证恢复执行，不是正式消融成绩；校验权重不用于正式初始化。

11:22资源核验两节点均无GPU进程或分布式启动器。接下来提交正式full144：
H0/W1、原始普通SAO Critics、fresh Adam/ValueNorm、seed1000043、
每12轮配对保存，96/144保留；完整结果报告和审阅之前不释放四项消融。
首次正式启动仍以full/STARTED及attempt回执为准，不将控制程序准备误写成训练已完成。
非计划中断后7200秒冷却和原绝对截止时间保持不变。

- [实际原生恢复门控](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/ROLLING_NATIVE_GATE.json)
- [校验结论](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/rolling-canary/RESULT.md)
- [报告与资源准入核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/NATIVE_GATE_REPORT_AND_ADMISSION.json)

### 2026-10-09 08:37 UTC：按用户要求暂停，重提交间隔改为两小时

用户08:34:57 UTC要求“可以暂停，改为两小时间隔再开始训练”。
已中断自有控制PID3514619，并停止只读观察PID3518079；两者均已退出。
node3当前自有校验服务已为inactive/not-found、MainPID=0，未改动任何其他用户任务。
本次是用户主动暂停，不是外部kill，原30分钟自动重提交控制器不再运行。

暂停时双方各提交12轮：H/W Actor更新5/9、Critic更新16/24；
保留96个共享环境结果、192条生成角色轨迹。**同轮双角色完整保存尚未提交，
没有可验证的LATEST配对断点，不能声称可从第12轮无损恢复。**
局部保存及原attempt、指标和成本全部保留；后续必须重跑恢复校验，不能跳过验收直接开正式MA。
正式144轮和四项消融仍无新完成结果。

按“从本条用户指令计时，至少两小时后再尝试；以后非计划中断也冷却至少两小时”执行，
不是把保存频率改为两小时，也不是要求GPU连续两小时空闲。
首次最早为2026-10-09 10:34:57.344 UTC，即北京时间18:34:57；
届时仍须重新排除两节点GPU占用和分布式启动器，资源繁忙则继续等，不清理外部工作。
校验内部预声明的12轮停止/恢复不额外增加两小时冷却。
每12轮完整保存、新配对验证成功后才删除旧代、保留96/144评测节点均不变；
原绝对截止时间2026-10-10 10:10:44 UTC不变。

旧授权和冻结源码不覆盖修改。新的执行层控制器直接复用冻结版本的单次训练/原生恢复，
不调用其硬编码30分钟的重试循环；只在明确终止证据下允许两小时后的新attempt，
未知代码/数据错误仍停止。新控制器已通过30项精确运行时测试，覆盖7200秒门槛、
撤销/修改授权、原截止时间、旧失败记录不能授权新错误，以及恢复门控不能把错误游标、
重置Adam、重复预热、不同角色断点或不完整预算当作成功。
08:50核实附着式等待PID3543367存活、心跳正常、没有原生子进程，未创建新的native attempt；
10:34:57之前不会提交GPU任务。等待器依附本会话，不宣称能在整个会话被关闭后继续运行。

- [最新暂停/延迟指令](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/USER_PAUSE_AND_DELAY.json)
- [暂停后的实际指标与断点审计](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/PAUSED_STATE.json)
- [两小时执行层控制器](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/resume_after_two_hours.py)
- [等待器存活及不提前启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_STARTUP_VERIFIED.json)
- [两小时控制测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_TESTS.log)
- [实时等待心跳](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/TWO_HOUR_MONITOR.json)

### 2026-10-09 08:20 UTC：双节点原生恢复校验已启动，当前为模型加载

r2输入绑定及两节点空闲/分布式启动器门控已通过。
原生校验attempt-0001在08:16:39 UTC启动，
双方角色身份回执已写入，08:20核验node3自有服务处于active/running，
无角色FAILED回执；最新控制心跳中双方已提交轮次仍为0，不能描述为已训练完成。

[实际启动回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/rolling-canary/full/attempts/attempt-0001/STARTED.json)、
[原生进度](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/rolling-canary/PROGRAM_MONITOR.json)
及[控制日志](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/CANARY_CONTROL.log)已持久化。
主控制PID3514619；只读首个配对断点观察PID3518079均已核实存活。
本次是12轮保存、停止自有校验进程、恢复至15轮的验收，不是正式144轮控制实验，
校验通过后再启动正式MA，校验权重不继承。

### 2026-10-09 08:08 UTC：补齐中断边界，120项测试通过，r2执行准备中

此前rolling12-v1等待器在尚无任何原生attempt时被我们主动取消，
不是再次遭遇外部kill，也没有损失训练轮次；
[取消回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12/CANARY_ADMISSION_CANCELLED.json)保留。
补齐了最终一轮已保存、完成回执尚未写入时的纯CPU恢复发布，以及新断点成功后
清理旧attempt未完成的大型保存文件，避免中断残片无限占盘。删除操作保留独立归属/清理日志，
不得越出本实验逐轮恢复目录。训练中的原生状态仍按同一实现保存和恢复。

新的[rolling12-v2源码快照](../runs/source-snapshots/alfworld-ma-mechanisms-rolling12-v2/SOURCE_MANIFEST.json)
通过[120项精确快照测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/CPU_VALIDATION.log)。
[r2控制程序](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/BOOTSTRAP_STARTED.json)
PID3514619已核实存活，当前进行输入/模型摘要绑定，随后自动进入资源门控及12→15轮原生恢复校验。
资源门控同时排除GPU占用和仍在启动的torch.distributed/vLLM任务，
不再仅凭瞬时0MiB就认定已有分布式任务的节点可用。

截至此条记录没有新的原生训练attempt，正式MA也未启动。
原始科学v2和历史证据保持不变；原绝对截止时间不延长。

### 2026-10-09 07:58 UTC：12轮完整断点代码已通过113项测试，原生恢复校验等待资源

用户07:40进一步授权首次启动只需新鲜的双节点空闲检查，不额外等待30分钟；
被kill后的重提交仍需两节点连续空闲30分钟。两节点资源在短时检查之间反复变化，
07:49曾每卡约120GB占用，07:53识别到其他账号的4节点NCCL通信校验；
已知利用率填充服务处于inactive，不能将其他进程当作填充程序清理。

[独立rolling12运行目录](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12/PROTOCOL.md)
已完成输入/原科学配置对照及准备，冻结v2源码没有修改。
新的操作性源码快照通过[113项针对性测试](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12/CPU_VALIDATION.log)，
包括真实CPU原生DCP/Adam/RNG保存恢复后的下一次更新一致性、12轮保存频率、
新旧配对提交/删除顺序、损坏拒绝、数据游标和2:3/每轮共同更新的恢复计数。

附着式原生校验控制器PID3508842已启动，当前处于资源等待；
[控制心跳](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12/PROGRAM_MONITOR.json)
及[启动回执](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12/CANARY_CONTROL_STARTED.json)可复核。
它将先运行12轮、主动停止自有校验进程，再从完整断点恢复至15轮；
校验权重不会进入正式训练。原生大模型恢复检查尚未通过，不能称为正式MA已开跑。
正式完整MA仍需通过这一额外门控；没有新的144轮或200题结果。

保存包含同轮次双方Actor/Critic/Adam、ValueNorm、worker RNG/LR和已提交输入/指标游标。
只有校验后的配对提交成功才删除前一份本实验恢复状态；保留96/144轮HF检查点及最终原生状态。
为兼容原有审计，HF检查点同时保留Actor和Critic，不仅是此前估算中的Actor。
进程中断后推理服务会重建，未提交的异步工作废弃并记成本，不宣称逐位相同的rollout重放。

### 2026-10-09 07:30 UTC：用户重新授权12轮完整断点及空闲30分钟后恢复

用户已要求开始训练，采用每12轮保存、成功提交新完整断点后才删除旧断点；
被kill后保留证据，等待两节点连续空闲至少30分钟再提交，不干预其他任务。
[本次授权](../runs/alfworld_ma_mechanisms_single_seed_v2/resume_12round/AUTHORIZATION.json)
取代此前暂停及“不再恢复”的执行限制，历史记录不删除。

目前正在实现并验证周期性双角色恢复，尚未启动新的正式训练。
现有48轮HF保存不能替代完整Adam/ValueNorm及训练位置恢复；
必须先通过针对性测试和实际恢复检查，不将新Adam或重复预热冒充无损续训。
冻结v2科学源码保持不变，继续使用原seed、144轮固定终点和原200题，
完整MA先行，结果审阅后再放行四项消融。原绝对截止时间仍为2026-10-10 10:10:44 UTC。

### 2026-10-09 05:16 UTC：按用户要求暂停恢复，转为整理独立算法说明

用户05:11曾授权连续空闲30分钟后恢复；只读观察于05:14启动。
05:15:48用户随后要求“先暂停”，改为单独总结MA-SAO算法及改进。
已停止并核实观察PID3444039退出，attempt03尚未准备或启动；
不因文档完成而自动恢复训练，也不把此前稀疏空闲快照当作已满足30分钟门控。

[当前暂停指令](../runs/alfworld_ma_mechanisms_single_seed_v2/resume_after_idle/PAUSED_BY_USER.json)
覆盖此前恢复授权的执行，历史授权及全部中断证据均保留。
独立[MA-SAO 算法说明](./MA_SAO_ALGORITHM.md)整理了相对普通SAO的变化、真实配对轨迹、
协作Critic、双重策略校正、ValueNorm、2:3调度、精确预算和四项消融，
并区分Dr. MAS相关归一化思想、已有技术和本项目的适配，不把尚未完成的消融写成正向结论。

### 2026-10-09 03:18 UTC：恢复尝试再次收到SIGTERM，按用户指令结束队列

attempt02在03:13:12 UTC记录SIGTERM并退出，正式144轮没有完成。
**当前无训练控制器、角色协调器或资源等待器在运行；不会再重启或新建恢复尝试。**
03:17已核实自有进程退出、node3自有服务关闭以及两节点资源释放。
即使GPU空闲，也遵守03:07:51用户指令，不再恢复。

实际指标文件显示双方各提交2轮，均为Critic预热：H/W Actor更新0/0、Critic更新4/4。
保留22个共享环境结果、44条生成角色轨迹，其中32条由已提交更新消费；
第三轮已有6个结果但没有提交更新，未落盘的在途成本未知。
未保存新的HF检查点，未获得原200题成绩，四项正式消融均未启动。
首轮观察曾成功通知，但那只是中间里程碑，不能误认为实验完成。

本次只能确认SIGTERM，尚未识别发送者；不能直接沿用上一次中断的账号归因。
node3的终止日志也可能包含本次协调器退出后的正常自有清理，不单独证明外部来源。
所有失败、校准及部分训练成本均已保留。

- [本次中断报告](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/full/INTERRUPTED.md)
- [实际轮次、轨迹和信号审计](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/full/INTERRUPTED.json)
- [按用户指令终止队列](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/STOPPED_BY_USER_POLICY.json)
- [自有进程退出及资源核验](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/RESOURCE_AFTER_TERMINATION.json)

### 2026-10-09 03:08 UTC：用户明确要求再次被kill后不再重启

03:07:51 UTC用户要求：如果对方再次终止我们的任务，不需要继续重启。
**当前正常运行的attempt02继续；若再次被终止，停止剩余实验队列，保留日志、
检查点和成本证据并汇报，不再自动重启，也不再由助手新建恢复尝试。**
这一要求取代之前“空闲后可再次显式恢复”的后续授权，不改变科学配置或正常完成后的消融顺序。
现有控制器本就配置零自动重试；新增用户约束也禁止后续人工式恢复。

本次只读检查确认控制器和首轮观察仍在运行，full的两个角色进程存活；
当时的进度快照尚无已提交轮次。没有因这条新指令停止当前健康运行的进程。
[不再恢复的用户指令](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/NO_RESTART_AFTER_TERMINATION.json)
已持久保存，后续接手必须遵守。

### 2026-10-09 03:00 UTC：同源恢复控制程序已启动，正式训练尚待门控

[独立v2 attempt02](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/PROTOCOL.md)
已完成实际输入/原生证据重绑定和历史路径复核；14个准备文件、10份角色配置的科学内容保持不变，
两项原生校准不重跑。node3通过既有训练服务接口核实UID1009/GID1008可读写新目录。
直接sudo切换用户的命令需要密码，没有修改授权；既有受控服务接口的CPU检查正常完成。

02:59:17 UTC启动附着式恢复控制程序，PID3401882，外部有界监督PID3401880。
03:00:23核验进程存活，首轮通知观察PID3402105心跳正常。
此时为输入摘要复核阶段，正式full/STARTED尚不存在，不能描述为已经开始提交训练轮次。
后续自动进行三次稳定空闲门控，再启动完整MA；完整结果审阅后才放行四项消融。
原绝对截止时间保持2026-10-10 10:10:44 UTC，任何新中断均停下留证，不循环重试。

- [实际准备核验](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/PREPARATION_AUDIT.json)
- [进程与观察心跳核验](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/MONITOR_START_AUDIT.json)
- [恢复说明与原成本](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/RECOVERY.md)
- [新控制日志](../runs/alfworld_ma_mechanisms_single_seed_v2_attempt02/CONTROL_PROGRAM.log)

### 2026-10-09 02:55 UTC：新空闲窗口已确认，准备一次显式同源恢复

只读观察在02:45:42 UTC取得连续三次双节点空闲并正常退出；
02:47与02:54复查两节点均无GPU计算进程，02:47也未见此前账号的训练启动器。
这是新的可用窗口，不是GPU预约，也不说明此前SIGTERM的生命周期问题已经修复。

按用户此前“释放后继续”的要求，准备独立v2 attempt02，只改变运行目录和自有服务唯一后缀。
H0/W1、原始Critic、种子1000043、144轮、原200题及四个消融的科学配置保持不变。
继承已通过的96个原生校准episode，不重跑校准，也不使用校准权重。
旧v1的22轮、v2的零正式轮中断、全部信号证据和启动成本均保留。
仍使用最初的绝对截止时间2026-10-10 10:10:44 UTC，不因恢复重新获得40小时。

[显式恢复驱动](../runs/alfworld_ma_mechanisms_single_seed_v2/recover_v2_attempt.py)
及已有路径等价辅助逻辑的23项针对性测试已通过；
[验证回执](../runs/alfworld_ma_mechanisms_single_seed_v2/RECOVERY_CODE_VALIDATION.json)已固定。
正在校验并重绑定实际模型、数据、14个配置文件和原生证据，尚未启动新的正式训练。
恢复启动仍会重新通过稳定资源门控；任何新中断均停下留证，不自动循环重试。
完整MA结果仍必须先汇报并复核，之后才能放行四个消融。

### 2026-10-09 02:32 UTC：两项原生检查通过，正式MA启动被SIGTERM中断

完整MA及每轮共同更新的六轮原生检查均已通过，总计96个共享校准episode；
完整臂H/W Actor为1/3、Critic为8/12，每轮共同更新为3/3和12/12，
第三轮队友矫正规则及实际更新预算核验通过。校准检查点未用于正式训练。

正式完整MA于02:20:30 UTC启动，02:22在模型载入阶段被SIGTERM中断。
两个角色均无已提交指标、正式轨迹/环境回执或检查点，尚无144轮/200题成绩。
控制程序已经退出，不是仍在后台训练或等待，也没有自动重试。

本次取得了来源证据：node3的sudo日志在02:22:19.462记录`hongyuanzhu`账号以root
运行指向MA训练模块的Python命令；紧接着02:22:19.718，信号跟踪记录UID0/PID1148490
发送SIGTERM。我们自己的服务清理在02:22:28才发生，晚于首个远端信号。
不能据此区分人工操作、自动脚本或其他会话，也不能将以往中断或node4的信号发送者
直接归因到该账号。02:31检查时，两节点已由该账号02:26启动的12K上下文压力测试占用。
没有停止或修改对方任务。

[中断报告及原生成本](../runs/alfworld_ma_mechanisms_single_seed_v2/full/INTERRUPTED.md)、
[信号来源证据与限制](../runs/alfworld_ma_mechanisms_single_seed_v2/full/SIGNAL_PROVENANCE.json)
已经保留。需要稳定、无清理冲突的GPU窗口后再作显式恢复，不能盲目重启争用资源；
原生通过结果可供同源恢复核验，不应无说明地重跑，正式四组消融仍未完成。

02:37 UTC另启了附着式只读资源观察，PID3392264，已核实心跳和占用状态；
该观察已在02:45:42 UTC取得稳定空闲并退出，不再是运行中的观察进程。
它不启动训练、不清除故障回执；只在重新出现稳定空闲窗口时通知恢复复核。
该观察最多12小时且不超过原40小时边界，状态位于
[中断后的只读观察](../runs/alfworld_ma_mechanisms_single_seed_v2/post-interruption-watch/PROGRAM_MONITOR.json)。

### 2026-10-09 01:44 UTC：资源已准入，v2完整MA原生检查开始

现有监控连续三次确认两节点空闲后，自动通过资源门控和冻结输入校验。
01:42:53 UTC写入首个六轮完整MA原生检查的
[开始回执](../runs/alfworld_ma_mechanisms_single_seed_v2/schedule-calibration/full/STARTED.json)，
01:43收到启动通知，01:44核对双角色进程存活、阶段心跳更新及GPU模型载入。
node3的自有Engineer服务以UID1009/GID1008运行；此前观察到的外部GPU进程已不在两节点快照中。
没有停止外部任务，也不将资源释放推断为对方实验成功完成。

当前是`native-full`校准阶段，尚无六轮通过结论，更不是正式144轮训练已经完成。
后续仍是每轮共同更新六轮检查、完整MA144及评测、结果/路线复核、四项消融。
[原生训练实时状态](../runs/alfworld_ma_mechanisms_single_seed_v2/schedule-calibration/full/MONITOR.json)
与[控制程序日志](../runs/alfworld_ma_mechanisms_single_seed_v2/CONTROL_PROGRAM.log)持续保留。

### 2026-10-08 18:11 UTC：四项消融v2已启动真实资源监控，尚未开训

按最新要求取消分阶段SAO行，改为纯粹的“每轮共同更新”消融。
五配置为完整MA、去队友矫正、去协作上下文、去ValueNorm、每轮共同更新；
仍只用种子1000043、原始200题和固定144轮。新的调度只改变Engineer更新开关，
保留第三轮unit队友权重及原有采样/输入，避免旧每轮更新实验的双重改动。
各臂环境/生成预算相同，但每轮共同更新总消费轨迹多20%，不能称同算力。

[v2协议与预算](../runs/alfworld_ma_mechanisms_single_seed_v2/PROTOCOL.md)已冻结。
精确冻结源上115项CPU测试、3个子测试通过，320个运行文件及所有模型/数据摘要、
10份角色配置的单一科学差异、原生检查的前48条输入均核验通过。
[代码和输入核验](../runs/alfworld_ma_mechanisms_single_seed_v2/CODE_VALIDATION.json)
不等于新的原生门控或正式训练完成；六个旧32K证据按摘要继承，
另需完整MA/每轮共同更新两次六轮原生检查，合计96个校准episode，权重不用于正式臂。

**有界、附着式监控已真实启动，而不是仅准备了脚本。**
18:10:44 UTC启动，协调PID为`3224405`；18:11核对进程身份与心跳成功，
状态为`waiting_for_resources`，两节点尚未同时释放，原生/正式STARTED回执均为0。
[启动核验](../runs/alfworld_ma_mechanisms_single_seed_v2/MONITOR_START_AUDIT.json)
及[实时阶段与资源记录](../runs/alfworld_ma_mechanisms_single_seed_v2/PROGRAM_MONITOR.json)已落盘。
每60秒只读检查，要求连续3次空闲；每段等待最多12小时，总程序40小时，
不会停止其他用户的任务，也不会遇错自动重训。node3以UID1009/GID1008访问共享目录
的真实CPU服务检查已通过，没有扩大权限或启动GPU工作。

执行顺序：两次原生检查 → 完整MA144及评测 → 汇报和历史路径审查 → 四组消融。
完整MA结束后控制程序会停在审查门，需绑定结果摘要的复核回执才可放行后四组；
不是因低分自动改配置重跑。完整MA和四个消融目前都没有新成绩。
[历史路线审计](../runs/alfworld_ma_mechanisms_single_seed_v2/HISTORICAL_PATH_AUDIT.json)
确认H0/W1 Actor、1152条有序输入和144轮更新预算一致，但历史97%使用MA64 Critic/ValueNorm、
round48的Adam重置、更多候选槽和开发峰值选择，只作为补充参考。

保留已实测约1.955倍加速，不声明新的加速测量。
资源可用后预计3--4小时先获得完整MA结果，总计约14--20小时完成五配置，
不含资源等待/基础设施恢复，当前不能承诺日历完成时间。
旧attempt02已在零开训状态[显式取消](../runs/alfworld_ma_mechanisms_single_seed_v1_attempt02/CANCELLED_BEFORE_TRAINING.json)，
原22轮中断及184个环境结果仍计入历史成本；此前SIGTERM发送方仍未查明。
本轮须保留正向、无差异和反向结果，不能事先认定四个组件均为必要。

### 2026-10-08 17:11 UTC：W1/H0消融重启准备已验证，GPU再次被占用，未开训

本次不是从97%检查点继续训练。五配置统一从H0/W1 Actor及普通SAO的原始尺度
H1/W1 Critic起跑，固定144轮；完整MA是同期主对照，历史97%只作补充参考。
历史97%的Actor也是H0/W1，但使用MA64 Critic/ValueNorm、round48的Adam重置和
48/96/144三次候选选择，不能只凭“同144轮”就称完全相同的初始化或搜索预算。
历史成功MA段的H/W消费768/1152、Actor93/141、Critic192/288与新联合臂匹配，
不包含上游W1、MA64及失败尝试；分阶段方法的环境成本另列。

[重启与对照计划](../runs/alfworld_ma_mechanisms_single_seed_v1/RESTART_PLAN.md)已经记录
四项消融、旧2:3对1:1调度的归因限制及条件耗时16--24小时。
独立[attempt02](../runs/alfworld_ma_mechanisms_single_seed_v1_attempt02/)准备完成：
17份准备输入仅重定位输出/输入路径，科学配置规范化后完全一致；原始/新摘要、
六份原生结果及中断指标摘要通过实查，12项CPU测试通过。原22轮及其成本未覆盖。
准备了遇错停止、无自动重试的串行驱动，以及原97%固定检查点的一次补充复测绑定。
这些是准备和验证，不是已经运行的训练或原生门控重跑。

16:52两节点确实空闲；随后`hongyuanzhu`的新进程于16:58启动，17:11检查时
两节点全部GPU利用率100%。新启动前资源门控拒绝执行，没有改动这些任务。
attempt02没有QUEUE/角色STARTED回执，新增训练和环境episode均为0，
没有自动等待/重启监控在后台运行。当前仍需可用GPU窗口，且旧SIGTERM发送方仍未确定。
[最新资源阻塞回执](../runs/alfworld_ma_mechanisms_single_seed_v1_attempt02/RESOURCE_BLOCKED.json)
保留了两节点记录和零开训核验；16--24小时是资源可用后的工作量估计，不是日历完成承诺。

### 2026-10-08 15:52 UTC：资源阻塞已核实，正式消融未在后台运行

只读检查确认node4和node3的全部16张GPU都有其他用户的分配，当前资源门控不允许启动。
没有停止、修改或挤占这些任务；当前占用不能解释此前SIGTERM的发送方。
两个自有正式训练服务随后核对均为inactive、MainPID=0，没有自动重启监控在后台运行。
1500秒无GPU生命周期诊断正常结束，未复现终止，因此不能把问题归咎于会话等待，
也不能据此认为GPU长任务的生命周期问题已经修复。

完整MA仍只有22/144轮；原始指标SHA及连续轮次再次通过核验。
持久化episode回执为Engineer184份、Solver184份，共享184个环境终局结果，
不同于已提交更新消费的120/176条，也不是368个环境episode或完整的尝试/token成本。
六个原生门控结果SHA及冻结输入摘要一致；后续四臂仍未启动。
资源快照、生命周期诊断及恢复约束已写入
[阻塞记录](../runs/alfworld_ma_mechanisms_single_seed_v1/RESOURCE_BLOCKED.json)和
[中断报告](../runs/alfworld_ma_mechanisms_single_seed_v1/full/INTERRUPTED.md)。
正式消融仍待稳定、授权的两节点GPU窗口及明确的生命周期恢复安排，不能给出可靠的完成日期。
恢复必须另记尝试、保持同一科学配置与种子并计入已发生开销，不覆盖旧记录或伪称Adam续训。

### 正式消融完整臂在22/144轮被SIGTERM中断，尚无正式消融结论

六个32K/4096-token原生机制门控全部通过后，已启动完整MA正式对照。
协调进程随后被SIGTERM终止，发送方未确认；不能把systemd的`success`等同于实验完成。
两个角色各提交22轮：Engineer消费120条、Actor12次/Critic30次更新；
Solver消费176条、Actor19次/Critic44次更新。没有96/144终点、HF检查点或主200成绩，
因此不能按同一Adam状态接着续跑，也不能把中断当作算法表现差。

保留记录：[正式完整臂中断报告](../runs/alfworld_ma_mechanisms_single_seed_v1/full/INTERRUPTED.md)。
后续四个消融尚未启动，先排查反复进程终止的生命周期原因，避免无说明地重跑并扩大预算。
已经完成并交付的W3及九阶段各200题结果不受影响。

### 前两项已经交付，进入单种子机制原生门控

用户明确实验2是九个阶段各自完整200题，不要求把8个候选全部做200题。
九行新列已完成并独立核验：[结果表及条件贡献分析](../runs/alfworld_ma_engineer_weights_w0_v1/RESULT.md)。
固定W0的新Harness列按48--432轮依次为81%、82%、82%、80.5%、85.5%、85.5%、85.5%、85.5%、85%；
本次W0空策略150/200=75%，W0+原保存脚本为79.5%（48/96）和83.5%（144及以后）。
每阶段新策略都是完整200题，不是8题；8只指此前已生成的候选数量。

用户要求的9份加3份控制共12份完整报告/2400个episode已经保留。
确认这些报告的SHA、200题覆盖及模型身份后，按本次用户缩小范围的指令停止剩余候选队列；
累计主评启动2750个，额外350个已发生的工作如实记录，不报告未完成的8候选均分。
原69策略大矩阵没有冒标完成，[范围限定最终审计](../runs/alfworld_ma_engineer_weights_w0_v1/FINAL_AUDIT.json)
只认证九阶段200题和三个控制，结果SHA为
`8bd5355b485f6fa899ea0f08871fded5207492b79a7fcd80ceea0090eaa1f28b`。

144轮的原保存Harness在W0上增益8.5pp，在对应Solver上增益4.5pp；
固定原保存Harness的weight差为13.5pp，空策略weight差为17.5pp，交互为-4pp。
新生成Harness与原保存脚本不同，不能把97%-82%冒充纯weight贡献。
历史两列和新W0列不在同次服务运行，不能过度声称严格因果分解。

W3负结果、同题失败案例及九行贡献表已在对话汇报并写入MD。
研究级消融仍只用种子1000043；当前先做H0/W0各自的完整32K/4096-token
full、去上下文、去ValueNorm原生门控，没有引入第二训练种子或新的研究轨迹。
冻结路径：[五配置研究](../runs/alfworld_ma_mechanisms_single_seed_v1/)。

### W3完整负结果已核验：48.0%，不是未完成训练

早停W2到W3的普通SAO周期已按用户要求完成，结果和失败案例写入
[完整报告](../runs/alfworld_w3h3_from_w2_early79_v1/RESULT.md)。
H3完成32轮/192条消费轨迹，W3完成80轮/640条；Actor分别27/75次更新，
Critic分别32/80次。H3仍0/8合法，执行empty且没有回退旧h2。

| 端点 | 原h2 | 无效H3的empty语义 |
| --- | ---: | ---: |
| 早停W2同口径新测 | 157/200=78.5% | 165/200=82.5% |
| W3额外80轮终点 | 62/200=31.0% | 96/200=48.0% |

同一empty策略下，追加Solver训练下降34.5个百分点，题目配对95%区间[-41.5,-27.5]pp：
4题改善、73题退化。换回旧h2也未修复，W3+h2只有31%。
同题W3失败而历史MA144成功99题，反向1题；其中72题原先早停W2+empty也能完成。
例244已取到并清洗土豆且走到垃圾桶，却不放下，随后在地点间循环到50步；
历史MA144同题7步成功。104个W3失败都出现重复动作，MA的6个失败也有此类现象，
因此不能把“这个实例MA解决了”夸大成“MA从不循环”。

保留报告覆盖800个主评episode；失败的末次评测另启动339个，实际记录的主评环境启动共1139。
额外339个的部分token成本未知，未用于择优重测。运输恢复只修连接和落盘，
没有重训、增提案或更换终点。[最终审计](../runs/alfworld_w3h3_from_w2_early79_v1/FINAL_AUDIT.json)
绑定结果SHA `25635fe08529a45084eef4bd66a9bf67bd5a015afdd3062c2d3f074fe3ba906e`。

### 2026-10-08 11:30 UTC执行更新

- **W3的80轮训练、75次Actor/80次Critic更新以及最终检查点已完成。**
  起点同口径新测W2+h2为157/200=78.5%，W2+empty为165/200=82.5%；
  历史79.5%仍单独保留。H3仍为0/8合法，实际执行empty，不冒充合法新策略。
  最后200题交叉评测遇到推理HTTP `ReadError`，不是再次训练失败；旧尝试无完整策略报告，
  已保留并开始只恢复评测，不重训W3或重生成H3。尚不能在最终审计前宣称新W3成绩。
- 九个Engineer的72个候选已全部生成，选择矩阵有69份规范化后不同策略。
  首次选择评测在28份完整报告/1344 episode后遇到同类HTTP读取错误：
  控制器记录1461次环境启动，另117次开始执行但未保存完整策略报告，部分token成本未知。
  完整报告按原始SHA保留，不择优重测；候选不重生成。
  独立恢复源码只改变HTTP连接复用与评测落盘/复用路径，
  51项测试及13项子测试通过，实际恢复已超过原中断点。
- 新推理客户端禁用空闲连接复用，不自动重试结果不明的POST；逐episode原子落盘，
  成功与失败都保留。恢复不声称逐样本bitwise等价，未持久化的重复工作单独计成本。
  主200将优先完成用户要求的新列和控制组，再补全全部8候选均分。
- 提速探针：GPU分片生成虽实测1.710倍，但15/16输出不一致，按事前门槛拒绝。
  完整文件SHA并行校验实测4.294倍且全部摘要一致，已采用；不是推理/训练4.294倍。
  [提速正负结果](../runs/alfworld_ma_engineer_weights_w0_v1/throughput-probe/RESULT.md)。
- 五配置、单训练种子1000043的机制研究已经CPU冻结：
  完整MA、去队友权重、去协作上下文、去ValueNorm、分阶段SAO方法级对照。
  共同H0/W1 Actor及普通SAO原始尺度H1/W1 Critic，不继承已受MA处理的Critic/ValueNorm；
  因此不是历史97%初始化的严格复刻。固定144轮主终点，H/W消费768/1152条，
  Actor93/141次、Critic192/288次；两方法Solver每个训练实例的暴露次数一致。
  161项测试及16项子测试通过，[实际冻结验证](../runs/alfworld_ma_mechanisms_single_seed_v1/CODE_VALIDATION.json)；
  **尚未开始正式消融训练**，仍需完成前两项交付与原生32K机制门控。

## 2026-10-08 08:00 UTC：单种子四机制消融与继续执行授权

用户要求先完成早停W2到W3的普通SAO追加周期、九个MA Engineer权重配固定W0的新策略评测，
再做完整MA加四个机制移除的单种子实验；取消第二种子，协作上下文和ValueNorm必须分开。
联合采样与团队学习的比较是方法级对照，不能冒称只改一个独立开关。
每个大实验启动前在对话通知，不等待批准；每个结果写入项目Markdown并汇报。
原200题是反复使用的开发集，不据其成绩挑消融终点或重试负面结果。

H3的0/8合法是已确认的负面结果。用户已授权不按合法率停止，沿用原无效输出执行为空策略
的规则完成80轮W3及原200题交叉评测，不回退旧h2、不增加H3训练或提案。
继续控制器于首次评测服务加载阶段被停止；全部八个服务日志和环境日志均无推理调用、
无环境episode，尚无W3训练。恢复脚本保留原attempt，使用相同模型、任务、seed和预算。
九Engineer评测尚未产生新候选，CPU哈希准备被中断，协议和W0状态保留、不重选检查点。

大规模执行前检查吞吐：W3继续使用已通过数值/原生验证的FLA0.5.2/Triton3.7.1，
此前同预算MA端到端暖步实测95.912秒降至48.725秒，约1.96倍，并非本轮新测量。
九Engineer另做有界速度探针，比较八副本逐检查点加载与按GPU分配不同检查点；
保留完全相同的提示、模型精度和4096-token上限，只有响应一致且实测更快才采用。
探针不训练、不评分、生成结果不进入正式候选选择，失败的优化也保留记录。

失败案例须来自同题真实行动/观察/终局回报：同时报告“SAO或消融失败、完整MA成功”和
反向配对的总数，再按明确规则选代表例子，不保证MA一定解决，不用案例冒充机制因果证明。
当前执行证据见[W3继续目录](../runs/alfworld_w3h3_from_w2_early79_v1/)和
[九Engineer及速度探针目录](../runs/alfworld_ma_engineer_weights_w0_v1/)。

## 2026-10-08：从早停 W2 的79.5%点再做一轮普通SAO交替

用户指定保留原训练集、原200题评测，从W2恢复曲线约79%的版本和原h2继续到W3+h3。
本次起点是`alfworld_w2_recovery_v2`额外40步的Actor/Critic，即原生globalstep39、
历史h2主200为159/200=79.5%；不是原始globalstep79的63.5%版本，也不是MA检查点。
这是用户看过开发曲线后的回溯选点，不能称为预注册的独立测试早停。

- 按原SAO次序先固定早停W2训练H3，再固定实际新H3训练W3，不启动MA联合更新。
- H3追加32轮、192条消费轨迹；W3追加80轮、640条。H3使用原train0--239范围内
  新收集的真实W2失败证据；W3复用恢复实验的同640题和顺序，仅更换轨迹ID与队友版本。
- H3仅评额外32轮终点，前后各8个候选在原selection24 mean@2比较。实验W3使用
  最佳合法新H3，即使原2个百分点部署门槛保留h2，也不把回退策略冒称新H3。
- W3保存20步间隔检查点，但事前固定额外80轮为本轮终点，不用主200再挑最高点。
  同一原200题上分开报告W2+h2、W2+h3、W3+h2、W3+h3及空策略对照。
- 两个所选检查点只有HF Actor/Critic权重，故各有一次明确fresh Adam与原5步
  Critic warmup。采用已修正的普通SAO独立轨迹和采样概率处理；无MA/CTDE、KL或ValueNorm。
- 不使用新100题训练、选候选或选检查点。原MA已保存的第三块继续保持暂停，
  不修改其历史源码或失败证据。尊重操作者已暂停GPU填充服务的状态，
  资源审计仍拒绝任何外部GPU负载，不强制恢复填充负载。

**首次执行结果：在H3阶段按当时规则早停；用户随后授权继续，状态见上。**
H3完成32轮、27次非零Actor更新和32次Critic更新，192条消费轨迹、198条已记录生成。
同一组8候选的合法数从训练前5/8降为训练后0/8；训练后8次均耗尽4096生成token，
没有输出完整`<patch>...</patch>`。这是真实模型输出失败，不是GPU、传输或优化器故障。
不追加候选、不临时延长解码预算，也不把旧h2回退冒称新h3，因此W3更新为0，
本轮新增主200评测episode也为0；159/200=79.5%仍只是起点的历史参考。
selection中训练后79.1667%的数值是无效策略按no-patch计分的结果，不是学到的H3表现。

冻结和执行证据位于[W3/H3实验目录](../runs/alfworld_w3h3_from_w2_early79_v1/)；
[停止结果](../runs/alfworld_w3h3_from_w2_early79_v1/STOPPED_RESULT.json)与
[独立停止审计](../runs/alfworld_w3h3_from_w2_early79_v1/STOPPED_AUDIT.json)已保存。
审计重新解析全部16份原始响应，核验模型、prompt、seed、预算、原始输入SHA及资源释放。
99项CPU回归测试、20项子测试及冻结源码上的19项定向测试通过；原RUN_FAILED证据保留。

## 最新实验报告与七小时交付窗口（2026-10-07）

正在扩充同一份[ALFWorld SAO / MA-SAO / WHALE实验报告](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md)，
已增加模型谱系、实际损失、超参数、数据划分、采样口径和新MA早期结果。
已纳入WHALE时间截断运行的真实主200：163/200=81.5%，严格197为81.2183%。
实际640个Solver episode、216个MH提案，未完成要求的384提案；不冒充同预算复现完成。
所有主分数保留开发集、选择重合、真实成本和统计解释的边界。

## 当前任务优先级（用户于2026-10-07 16:37:12 UTC更新）

H3原始200题与WHALE结果的硬截止为**2026-10-07 23:37:12 UTC**。
MA在此前给出早期结果，用户明确允许其之后继续；不能再要求MA先完成长期收敛才启动WHALE。
不插入新的SAO基线重训、训练种子扫描或大批消融。

1. **真正H3+W1的200题已完成并重新核验。** 新策略`after-proposal02`为
   **173/200=86.5%**，不是旧h2回退别名。同期旧h2为178/200=89%，
   新策略相对它为-2.5pp，CI[-7,+2]。保存原始报告和策略/模型SHA；
   不重复占用GPU重测已完整完成的同一请求，也不用89%覆盖历史92%。
2. **从保存的高点继续MA-SAO，检验更高且稳定的水平。** 用户本条称97.5%，
   但实际完整曲线、检查点及新冻结设计均对应总144轮的**97.0%**；
   不擅自将成绩改成97.5%，也不创造不存在的检查点。
   新每轮联合分支已完成首96轮原生训练和两次主评：总144/192/240轮为
   **97%→93%→92%**，空策略92.5%→90.5%→89%。总240相对父点-5pp，
   CI[-8.5,-2]，0改善/10退化；不是新的更高成绩，也尚不满足平台判断长度。
   原生Actor H160/W192、Critic320/384保存完整。独立Critic/CTDE、ValueNorm及
   bounded队友校正均保留，不能称为参数共享或第一次联合训练。
3. **利用早期MA未改善的完整块边界执行WHALE。** v12保留v10/v11的160题/Adam43，
   现已完成两个Solver段共640题/Adam159；MH实际192+24=216提案、131合法，
   两段搜索均保留原h0。开发6/10→9/10→10/10，不能代替主200，
   原W0、固定本地RL H0 proposer，未调用外部助手API。两周期各320 RSFT
   episode+192 MH提案，共640/384；只匹配明确的角色采样维度，不匹配所有计算、
   population或整个历史研究预算。RSFT新批截止22:52:12 UTC，MH新迭代截止23:12:12 UTC，
   已因该时间预留规则停止搜索，少168个MH提案；不是性能退化停止，
   **尚未完成640/384同预算复现**。主200已核验为163/200=81.5%，
   对历史SAO92%差-10.5pp，95%区间[-16.5,-4.5]；不是完整同预算算法比较。
   最终原始报告在硬截止前完成，双节点已释放，MA续训协调器接续。

MA总240首次主评启动被本次未屏蔽CUDA的CPU测试进程触发watchdog，中断时0episode。
原协调器已退出；保留其失败证据，仅恢复完全相同的评测矩阵，没有重训或重选。
所有后续CPU测试均屏蔽CUDA。WHALE原始200和双节点释放已核验；
续训协调器在23:31 UTC实际启动从原生状态恢复的block02，
恢复边界通过，FP32/Adam/ValueNorm不重置，首块不重训。
原24小时安全上限不延长；不能把临时资源插入或早期下降写成研究终止/最终收敛。

WHALE v8因跨节点读取旧私有报告的权限在训练前失败；v9只复制同字节输入，
未改旧目录权限，并实际通过完整32K/warm-Adam/vLLM原生验证（1822.47秒）。
v10仅修改迭代内评分的共享worker/并发路径：56测试、48subtest及真实CPU worker
四episode/零模型调用验证通过。非评分代码AST、模型、kernel和native配置一致，
故显式复用v9的原生组件证明；没有伪造一次v10 GPU smoke。
这些门控本身不是性能成绩，最终时间截断的主200已在v12取得。

- [MA早期结果](../runs/alfworld_ma_simultaneous_from97_v1/EARLY_RESULT.json)
- [MA实际续训协调器状态](../runs/alfworld_ma_simultaneous_from97_v1/RESUME_MONITOR.json)
- [WHALE v12设计](../runs/alfworld_whale_sao_matched_v12/DESIGN.json)
- [WHALE组件验证](../runs/alfworld_whale_sao_matched_v10/COMPONENT_VALIDATION.json)
- [WHALE独立最终结果](../runs/alfworld_whale_sao_matched_v12/RESULT.json)

### WHALE吞吐修订：实际640/216，主200为81.5%，未满计划预算

实际首32题产生552个成功调用。原生COMMITTED的1250秒计时尚未包含之后
约6分钟的逐rank串行文件哈希；整个请求到保存约1626秒，另有rollout/服务开销。
不能用MA的32K合成8.86倍直接预估这条RSFT流水线，原实现无法在窗口内满额。

v10在第二个完整native批后暂停CPU协调器，保留64个真实episode、两个完整
FP32/Adam检查点，未启动第三份rollout报告。v11不重采这64条，而是准备
按rank分片成功调用、保持全局response-token均值，以及并行完整文件SHA、
由下一次实际推理验证导出而不重复空载服务。两rank不同长度/尾部padding测试
的参数、Adam、loss和token计数与原复制实现一致；46测试/35subtest通过。
8个真实rank-state文件的哈希在39.58秒重算，全部与原摘要一致。

新原生32K/warm-Adam验证实际通过，1210.03秒；原跨多步loss/grad门槛失败且保留。
随后另行事前声明固定权重、零更新的全梯度诊断：同路径重复RMS0.0027486，
分片RMS0.0032533，小于规则`min(.01,max(.001,2×repeat))`给出的0.0054973。
三次loss相同；只支持固定权重目标/梯度范围，不能称多步轨迹逐值等价。
v11继承64题/Adam18后保存到160题/Adam43，没有重采或重置。

实际状态I/O仍慢，故v12在任何MH或最终200分数出现前冻结批量修订：
第一周期32/32/32/32/32/160、第二周期160/160，仍320/640边界和384提案；
**online rollout刷新频率改变，是学习超参数修订，不是纯运行时等价**。
H0/Solver分别8个replica、每模型每GPU显存0.4，跨MH段常驻；
实际128路并发、每路32768总token的双模型门控通过，293.05秒，资源已释放。
该门控128次合成调用、0环境episode、0优化更新，不能计入研究采样。
v12新原生计算/恢复代码与v11一致，显式复用已有原生证明而不伪造新smoke。
RSFT准入22:52:12、MH准入23:12:12，硬截止仍23:37:12 UTC。
v10/v11的已验证路径与失败/暂停证据均保留。若剩余时间不足，仍须在截止前给出
实际预算和主评，明确哪些等样本目标未完成，不能改成“性能差提前停”。

- [v11冻结设计](../runs/alfworld_whale_sao_matched_v11/DESIGN.json)
- [实际160题暂停收据](../runs/alfworld_whale_sao_matched_v11/THROUGHPUT_PAUSED.json)
- [数值门控修订及边界](../runs/alfworld_whale_sao_matched_v11/VALIDATION_AMENDMENT.json)
- [双模型实际服务验证](../runs/alfworld_whale_sao_matched_v12/DUAL_SERVING_VALIDATION.json)
- [320题原生状态核验](../runs/alfworld_whale_sao_matched_v12/NATIVE320_AUDIT.json)
- [首周期192提案核验](../runs/alfworld_whale_sao_matched_v12/MH0_AUDIT.json)

## 2026-10-07新增优先实验：固定W1，再训练一阶段Engineer

用户指出实验2只有“固定Harness续训Solver”一支，不能据此声称SAO两个方向的
继续学习都已完成验证。新增互补实验：固定历史较强W1，从真实H2-step31
Actor/Critic继续一个历史大小的Engineer阶段，即32 native step、batch6、
192消费轨迹、前5步Critic预热、27 Actor/32 Critic更新；W1零更新。
H2仅保存HF参数、没有Adam，故明确fresh Adam边界，不伪称精确优化器续跑。
沿用修正后的普通SAO独立轨迹与采样概率处理，不启用MA、KL或ValueNorm。

续训前后分别生成相同8组prompt/seed的候选，用原selection24 mean@2比较
全部8槽平均质量及合法率；相同保存h2 incumbent和2个百分点门槛选择部署策略。
固定最终extra32检查点，不搜索多个训练终点；主200题同时报告before/after、
历史h2、空策略，以及新Engineer最好合法候选（即使未获替换也照报）。
因此不会把“旧策略被保留”误写成Engineer学习提升，也不隐藏新候选退化。
主评测依旧是重复开发集，选择重合和题目配对区间须披露。

已在MA首个96轮原生块结束后插入并完成该单阶段：仅暂停MA的CPU协调器，
两个独立原生learner继续完成整块与状态保存；node3执行新实验并释放GPU后，
已恢复同一个MA协调器，不改变其冻结算法/数据/Adam。墙钟插入单独记录，
不延长原MA的24小时上限。
新训练与比较逻辑73项测试通过；资源暂停/恢复与候选比较定向19项测试通过，
两组有重合，不能相加当成独立测试数。

最终训练32/32步、27次非零Actor更新、32次Critic更新；192条声明消费、
198条实际生成（6条异步预取额外），1120个训练奖励环境episode、
1176个选择/主评环境episode。八槽候选均分88.8021%→89.5833%，
差+0.78125个百分点，95%区间[-4.4271,+5.7292]，没有可靠提升。
新最好候选对旧h2为-2.5个百分点，区间[-7,+2]，8改善/13退化；
对空策略为0个百分点，区间[-4.5,+4.5]，10改善/10退化。
前后部署和旧h2是同一策略的别名，共用物理报告，不是三个独立模型的等效验证。
结论是格式合法性改善、任务效果未可靠突破，而非完全没有学习或已证实绝对上限。

- [固定W1续训与比较实现](../whale_local/sao_bridge/alfworld_fixed_solver_engineer.py)
- [单阶段实验目录](../runs/alfworld_fixed_w1_h3_v1)
- [最终结果](../runs/alfworld_fixed_w1_h3_v1/COMPLETE.json)
- [最终审计](../runs/alfworld_fixed_w1_h3_v1/FINAL_AUDIT.json)
- [MA恢复收据](../runs/alfworld_fixed_w1_h3_v1/MA_RESUMED.json)

## 实验报告的术语与组织

后续按“研究问题—实验设计—实验结论”逐项报告，不把不同问题混成一个总分结论。
原MA-SAO本来就是联合学习：每三轮H/W联合更新两轮，再Solver-only一轮，
即H:W更新比2:3；不是先前没有联合学习、现在才开始联合学习。
新候选改为每轮双方更新，即1:1，并同时统一第三轮的bounded队友校正。
这是联合框架内部的调度候选，不是纯粹单变量的频率消融，更不是新的“联合对交替”对照。
H指生成Harness的Engineer模型，W指Solver；保存的P48/P144只是具体Harness产物，
交叉评测这些产物不能直接证明整个Engineer生成分布改善。
当前主线的完整三段式报告见[独立实验报告](./ALFWORLD_SAO_MA_SAO_EXPERIMENT_REPORT_2026-10-07.md)；
较早版本保留在[实验日志](./EXPERIMENT_LOG.md)的“按研究问题组织的实验报告”一节。

## 2026-10-07最新要求：保留97%检查点，从该点每轮H/W联合续训

用户随后明确要求不要从93.5%重新起跑，而是保留总round144的97%点，
并核查是否真正学到了能力。原93.5%起点协调器已在开训前停止：
没有STARTED、没有block、零新增训练轨迹；其冻结设计保留，
另存[取消收据](../runs/alfworld_ma_simultaneous_convergence_v1/CANCELLED_BEFORE_TRAINING.json)，
不能将它记为一次失败训练结果。

用户明确要求Weight和Harness worker通过MA-SAO一起更新。核验第四块第70轮时，
Engineer47次Actor更新、Solver70次，Engineer最近联合轮次梯度非零：
不是后期完全冻结H，而是一直沿用joint/joint/Solver-only的2:3更新配比。
该安排来自旧H0锚定对照的采样控制组，不是MA-SAO必需设计，也未证实最优。

已按用户要求设置明确实验边界：在第四块第73轮时仅停止旧CPU协调器，
两节点原生训练服务保持active。新的边界协调器完成该块原配置96轮及
总round384/432两个指定评测，不再启动第五块。只有原平台规则确实满足才称平台；
否则记录`user_requested_schedule_change_not_converged`，不能冒充算法收敛或基础设施失败。

新的[97%起点每轮联合候选](../runs/alfworld_ma_simultaneous_from97_v1/DESIGN.json)已独立冻结，
恢复总round144的两角色Actor、Critic、ValueNorm和原生FP32参数/Adam状态，
而不是仅加载会舍入参数的HF导出。初始Adam步数H为64/128、W为96/192
（Actor/Critic），本次没有新的fresh Adam。保留原H0/W1固定reference。
从source index768继续，与旧混合方案block02的任务/prompt顺序对应。
平台规则的“至少192额外轮”从144计算，最早总round336才有资格判定；
不能沿用旧48起点而提前宣布平台。

两角色每轮都更新Actor和Critic，全部轮次使用bounded队友权重，
移除anchor路由及影子-only Engineer轮次。LR/KL/K2/奖励/原始reference、
任务与prompt顺序、评测和平台规则保持不变。
每96轮H/W各96Actor、192Critic，消费1536角色轨迹（旧方案1280），
生成仍1536、环境仍768；额外20%消费成本必须披露，不是严格同消费量对照。
每轮额外激活H与第三轮队友权重规则一起改变，不能单独归因于其中一项。

70项定向测试通过，新增覆盖97%起点、两角色不同Adam偏移、首块原生恢复、
source768续接、完整Critic状态和相对起点的平台门槛。
增强后的Adam审计还通过了旧方案前三块的真实检查。
本次12:41 UTC状态询问期间，旧方案完成了总432轮末评和完整最终审计：
带保存策略95%，空策略95.5%；最后四点93/94.5/94.5/95%，跨度2个百分点，
满足事先规定的观测开发平台，但不是稳定97%或数学收敛。
最终相对历史92%为+3个百分点，配对95%区间[0,+6.5]；
下界不严格大于0，尚未通过预设优势门槛。
新CPU协调器已完成输入核验，交叉验证也已完成并释放node3；
旧FINAL_AUDIT已经生成并通过，队列依赖解除，按原冻结设计转入97%原生续训。
没有重跑SAO，没有新增训练种子，也不会启动旧方案第五块。
新分支暂未有评测成绩，启动/恢复不能当作已经取得新的训练效果。
两个Critic参数仍独立，仅共享协作信息。

97%之前这条成功谱系消费1920角色轨迹（640+1280），上游W1/MA64初始化成本另计；
后续旧混合曲线、失败尝试和诊断都是真实研究成本，不能因回到高点而删除。
用户从已观察曲线挑选97%最高点，必须披露开发集选择偏差，
不能将它当作未经筛选的确认性初始分数。

### 97%的现有行为证据与正在进行的核验

- 同一W144，空策略185/200，新自动Harness194/200：9题改善、0题退化，
  +4.5个百分点，题目配对95%区间[+2,+7.5]个百分点。
  9题中7题涉及加热/冷却/清洗后放置，2题涉及取两个物体；
  原始消息能看到停止无效搜索、拿取目标、先变换再放置等提示实际出现。
- 原200题有3题不满足严格目标文本隔离；在strict197上仍为191/197对183/197，
  不是仅由这3题产生的高分。
- **进一步核查发现，selection48全部包含在报告200题中；二者不是独立测试集。**
  排除这48题后，新策略146/152=96.05%、空策略141/152=92.76%，仍5题改善、
  0退化；再排除3个训练目标文本重合项为143/149对139/149，仍4题改善、0退化。
  这些是事后敏感性分析，且97%检查点本身是看过200题曲线后选择的，
  不能据此声称获得了新的独立测试确认。
- 空策略从父W48的90.5%到W144的92.5%，+2个百分点，
  区间[-1.5,+5.5]个百分点，单独的Weight能力提升尚不明确。
- 两个Engineer检查点使用完全相同的8组生成消息，合法提案由5/8变为8/8；
  但W144下仍有仅35/48的差提案。一个被选中的有效Harness，
  不等于已经证明Engineer整体生成分布因RL而稳定变好。
- [固定检查点交叉验证已完成](../runs/alfworld_peak97_crossplay_v1/COMPLETE.json)：
  W48/W144分别搭配空策略、初始H0策略、P48、P144，共8个系统、1600环境episode。
  使用原推理seed，重测对角组合；无新增训练种子，不据此重新选检查点或训练起点。
  两份原生矩阵、八份原生报告与compact行/哈希、1600次环境成本已再次核验。

| 固定策略 | W48复测 | W144复测 |
| --- | --- | --- |
| 空策略 | 90% | 91.5% |
| 初始H0策略 | 91.5% | 92.5% |
| P48旧自动策略 | 95% | 95% |
| P144新自动策略 | 95% | 96.5% |

原97%组合复测96.5%，支持该固定组合的高分不是仅出现一次，但不是长期平台证据。
W144下新策略对空策略+5个百分点，10改善/0退化，题目配对95%区间[+2,+8]；
排除selection48后仍6改善/0退化，+3.95个百分点，区间[+1.32,+7.24]。
但旧组合也从原93.5%复测到95%：新旧完整组合差只剩+1.5个百分点，
4改善/1退化，区间[-0.5,+4]。固定旧策略时W48/W144都为95%，
固定W144时新旧策略差+1.5个百分点、区间[-1,+4]。
因此尚未确立继续训练带来的明确增益，也未证明Engineer整体生成分布改善；
当前较可靠的证据是保存下来的自动Harness在开发集上确有作用。

- [旧实验用户指定边界](../runs/alfworld_ma_convergence_v1r4/REQUESTED_BOUNDARY.json)
- [旧实验边界执行器](../runs/alfworld_ma_convergence_v1r4/finish_requested_boundary.py)
- [旧混合方案最终审计](../runs/alfworld_ma_convergence_v1r4/FINAL_AUDIT.json)
- [97%原生续训执行器](../runs/alfworld_ma_simultaneous_from97_v1/driver.py)
- [97%行为与来源审计程序](../runs/alfworld_peak97_crossplay_v1/audit_behavior.py)

## 2026-10-07：核实SAO延长曲线，启动MA-SAO长训练研究

用户要求不再停留于短预算比较，而是延长MA-SAO观察绝对高度和稳定性。
随后进一步明确研究顺序：先调优到稳定更强，再冻结方法比较相同样本量，
最后做细节归因。若当前方案不够好，可据实际失败机制调整MA-SAO算法/实现，
但每个候选独立冻结、保留全部结果，不热改当前实验或筛选种子。
[阶段门控研究计划](../runs/alfworld_ma_convergence_v1/RESEARCH_PROGRAM.json)
记录后续比较所需的初始化/继承成本、消费/生成/环境预算和评测匹配约束。
历史SAO仅在合同匹配时复用；否则在优势候选冻结后才开展必要的单种子匹配对照。
历史SAO延长实验已完成，并非“尚未启动”：

| 从真实退化W2额外训练 | 0步 | 20步 | 40步 | 60步 | 80步 |
| --- | --- | --- | --- | --- | --- |
| 固定h2的200题准确率 | 63.5% | 71% | 79.5% | 38% | 49% |
| 空策略准确率 | 49% | 74.5% | 83% | 46% | 44.5% |

2026-10-07重新核验10份原生报告及8项配对比较。结论是先恢复、后再次崩落，
没有回到历史92%；不是持续单调下降，也不是已经证实的稳定收敛。
延长实验使用fresh Adam并修正独立轨迹打包，固定h2、只续训W；
不是继续若干完整H/W交替周期，不能推出所有交替SAO的收敛上限。
权威来源：[W2延长结果](../runs/alfworld_w2_recovery_v2/COMPLETE.json)。

新MA-SAO方案从93.5%的第48轮Actor/Critic及ValueNorm出发，初始Adam状态未保存，
故明确有一次fresh Adam边界；之后通过原生DCP保留FP32参数、
原生逐rank优化器分片保存/恢复Adam步骤/矩，
不在每次评测后重置Adam。固定原H0/W1 reference及原学习率/KL/K2/训练预算单位，
不重新做critic warmup。每96轮释放GPU评测第48、96轮两个检查点，
serving进程会重启，不能称逐位RNG续跑。

至少192额外轮；最后4个48轮间隔的观测点范围<=2个百分点，
且3个间隔未刷新至少0.5个百分点的最高分，才称观测开发平台。
最大768额外轮/24小时是安全上限，触顶必须报告未确认收敛。
每个点8提案、selection48、至少比继承策略多2题才替换，200题完整曲线全部保留。
新起始incumbent是自动proposal06；不靠200题选提示，不新增SAO训练、种子扫描或组件消融。
最高点会作为探索性筛选结果，不作为确认性显著优势。

原生Adam保存/恢复及下一步更新一致性的CPU测试和定向回归共86项通过。
首次32K验证中，原生组合DCP加载因纯文本未使用的视觉参数没有Adam状态而失败；
零研究轨迹，保留失败。不能用strict=False吞掉缺失状态，也不能以HF导出替代
FP32 master参数（HF导出会舍入到BF16）。恢复方案使用model-only DCP加原生Adam分片，
并验证所有已更新参数/矩的采样指纹和步骤计数；新增懒初始化状态CPU往返测试通过。
随后定位并修复synthetic控制器未设loopback而worker绑定loopback的地址不一致；
无研究轨迹损失，W0通过结果被复用，未重复已通过的验证。
v1r3两节点完整32K原生handoff均通过，但正式block-01在完成30轮后，
Solver到环境controller的交互POST发生ReadError，未到第48轮保存点。
已消费400角色轨迹；有248条提案、246个完整环境结果，
491条完整角色回执（包括11条未提交更新回执）。这些保留为失败研发成本，
不能计入成功优化器lineage，也不是模型性能终点。
日志与HTTP keepalive关闭竞态一致；沿用controller到worker层已有做法，
禁用Solver到controller的连接复用，不对可能已执行的动作POST盲目重试。
相关28项测试及13个subtest通过。

2026-10-07 04:41 UTC核验：恢复分支v1r4首个96轮块训练完成，
Editor64 Actor/128 Critic、Solver96 Actor/192 Critic更新及原生Adam保存审计通过。
未复现第30轮传输故障。首个指定终点（总round96）仍为187/200=93.5%，
空策略183/200=91.5%，保留父自动proposal06；相对历史SAO为+1.5pp，
配对95%区间[-2,5.5]，相对父93.5%为0pp[-2.5,3]。尚未建立整体继续上升。
05:02 UTC新增总round144终点：194/200=97%，空策略185/200=92.5%。
自动proposal07在selection48获得48/48，继承proposal06为46/48，
达到既定多2题门槛。相对历史SAO为+5pp[2,8.5]（11改善/1退化），
相对父93.5%为+3.5pp[1,6.5]（7改善/0退化）；原生矩阵、选择和配对统计复核通过。
完整曲线目前为93.5%、93.5%、97%。这是长期提升的正向开发证据，
不是已经确认的稳定平台，不能因单个高点提前停止。
第二个96轮块已完成，实际Adam加载及累计步骤通过审计：
H Actor128/Critic256，W Actor192/Critic384，四rank一致。
07:13 UTC总round192终点为96%，空策略90%，保留round144的自动proposal07；
相对历史SAO为+4pp[1,7]。完整已评曲线为93.5/93.5/97/96%，
没有持续单调上升，也没有确认平台。07:41 UTC总round240终点为94.5%，
空策略92.5%，仍保留proposal07；相对历史差+2.5pp[-1,6.5]。
两个第二块终点均完成原生矩阵与配对复核。完整曲线为
93.5/93.5/97/96/94.5%，最后四点范围3.5pp，平台和优势门槛均未满足。
第三块续训已开始，不以97%最高点冒充长期稳定结论。
09:46 UTC总round288为93%，空策略92%，继续保留proposal07；
相对历史差+1pp[-2.5,4.5]。第三块训练已完成，正在评测总round336。
当前曲线97%高点后为96/94.5/93%，尚未建立更高稳定平台。
现有选择集记录显示round240/288新提案最好47/48，incumbent46/48，
均未达到多2题门槛；这只是待诊断的选择限制，不能靠主评测反向更换提示。
从相同父round48重新开始，保持种子、数据、算法及评测规则；复用已通过的
W0/H0原生状态验证。之后每96轮块沿用原生Adam，不再重置。

架构澄清：当前两个Actor和两个Critic参数均独立。Critic输入加入同episode的
协作上下文，但没有共享Critic主干、价值头或隐藏表示；这是MAPPO启发的
信息集中训练，不是共享参数的团队Critic。当前调度为两轮联合更新、
一轮只更新Solver。用户同意先完成冻结曲线，再依据诊断决定是否改变架构。

- [固定设计及恢复说明](../runs/alfworld_ma_convergence_v1r4/DESIGN.json)
- [持续执行器](../runs/alfworld_ma_convergence_v1r4/driver.py)
- [本次运行目录](../runs/alfworld_ma_convergence_v1r4)
- [总round144的97%结果](../runs/alfworld_ma_convergence_v1r4/block-01/round0096_COMPLETE.json)
- [失败研发成本](../runs/alfworld_ma_convergence_v1r3/FAILED_COST.json)
- [历史SAO延长曲线重新核验](../runs/alfworld_ma_convergence_v1r3/SAO_TAIL_RECHECK.json)

三项研究问题的当前答案边界：
1. 同192条Engineer轨迹，W1下SAO90.5%对GRPO88%，尚无明确差异；
   W0迁移下SAO74%对GRPO83.5%。不能把“不显著”称为等效，
   也不能概括SAO普遍不差于GRPO。当前复用已有结果，不重复训练。
2. 已测的固定h2、退化W2延长训练先恢复再崩落，没有超过历史92%；
   不是“继续训练完全不涨”，也未证明所有交替SAO的收敛上限。
3. 当前积极检验MA的长期上升及更高观测平台。若冻结曲线优势不足，
   才按诊断选择一个算法/实现改进；不能预先保证正结论。

[长期结果核验器](../whale_local/sao_bridge/joint_research_audit.py)在最终结果落盘后
重新核验原生报告、选择、完整训练成本和Adam步骤，并计入失败前序400消费成本。
只有观测平台、末四点均高于92%、末端配对区间下界>0同时满足才解锁样本效率阶段；
仍标注探索性单种子开发证据。13项定向测试通过，真实整条曲线尚待完成后核验。

## 2026-10-06 收尾：加速调优MA-SAO达到93.5%，自动新策略入选，但优势尚未确立

不再启动新的SAO对照或训练种子扫描。历史W1+h2=92%继续作为参考，
但历史对照不是新匹配的因果实验。已运行的种子1000043保持原配置完成，
随后仅先评估原final48主终点；原人工提示扩展暂缓，不能将其记为已完成。

本轮完整结果、审计和结论已落盘，node3/node4训练与评测服务均成功退出，
两节点无剩余研究GPU进程，adaptive filler保持运行。没有新SAO训练或种子扫描。

| 固定200题主终点 | 成功数 | 准确率 |
| --- | --- | --- |
| 历史SAO W1+h2 | 184/200 | 92% |
| 同种子上一MA-SAO配置，保留H0 | 185/200 | 92.5% |
| 加速、平衡学习率MA-SAO，自动proposal06 | 187/200 | **93.5%** |
| 同一新Solver，不用策略提示 | 181/200 | 90.5% |

相对历史SAO为+1.5个百分点，配对95%区间[-1.5, 5]；相对同种子上一配置
为+1个百分点，区间[-2.5, 4.5]。这是正向点估计，**预设优势验证仍未通过**；
不能宣称MA-SAO普遍优于交替SAO，也不能据此证伪整个方向。
全200题仍是反复开发使用的题目，区间不是训练种子稳健性或确认性泛化证据。

新Engineer生成8个提案，5个有效；proposal06在选择48题上48/48，
H0为46/48，达到至少多2题的既定替换门槛。该选择集100%不是主成绩。
这次确实使用自动新策略，而非旧H0或人工恢复提示；尚无相同新Solver配H0的
200题主评测，不能把全部改进单独归因于Engineer学习。

原种子1000043训练与固定final48主评测已完成：H0组合185/200（92.5%），
无提示181/200（90.5%）。仍未选出达到替换门槛的新Engineer策略。
相对历史SAO92%仅多1题，配对差+0.5个百分点，95%区间[-3, 4]；
尚未确立MA-SAO优势，不能将保留H0的结果归因于Engineer自动学习。

加速验证已完成：隔离FLA0.5.2、Triton3.7.1，并仅将固定reference存储改为BF16；
Actor/Critic master weights仍为FP32，原BF16计算不变。
完整32K原生Actor+两次Critic稳态更新由346.94秒降至39.17秒，约8.86倍。
这是合成优化器工作负载，不是端到端RL加速倍数。真实训练相同稳态轮次
由平均96.90秒降至49.56秒（1.96倍）；包含成功尝试冷启动、编译和保存的
训练总墙钟由101.09分钟降至60.00分钟（1.68倍）。
真实轨迹token数会变化，并同时减少了中间保存，不能将端到端收益全部归因于内核。
总墙钟比较不含失败尝试、加速验证、驱动输入重哈希和主评测。
H0/W0原生更新、数值误差、有限非零梯度、参数变化和释放审计通过；最新CPU定向45项测试通过。
旧Triton被正确性guard拒绝、首次FLA原生Actor反向OOM均保留，未绕过检查或删除失败。

新候选仅一个整体调优方案：Engineer LR由1e-7提高至5e-7，Solver LR由1e-6降低至5e-7；
同种子1000043、H0/W1起点、数据及48轮/640消费/768生成/384环境预算不变。
使用已验证加速，只保存最终48轮；仍按48题选择集和至少多2题的门槛决定是否替换H0。
无人工提示、无新SAO训练、无新训练种子。输入、模型和源码已重验并冻结，
首尝试在rollout冷编译期间超过原300秒启动等待而失败，零轨迹/零优化更新；
失败和资源释放记录完整保留。仅通过可选`rollout_setup_timeout=1800`
延长启动健康检查等待（默认不变），同配置从初始权重重启。
恢复后的48轮完整训练审计通过：Editor29次Actor/64次Critic更新，
Solver45次Actor/96次Critic更新；选择与主评测共736次新增环境任务。
所有原生矩阵、完整题目覆盖、解码种子、token预算、提示哈希、选择规则和配对统计独立复核通过。
历史对照、重复使用开发200题、两项LR与运行时一起变更均限制因果解释。

- [加速验证完整结果](../runs/alfworld_training_speed_v4/COMPLETE.json)
- [已完成重复的主结果](../runs/alfworld_explicit_hint_training_replication_v1/repeat/matched_COMPLETE.json)
- [新候选固定设计及恢复说明](../runs/alfworld_ma_balanced_fast_v1r2/DESIGN.json)
- [本轮实验结论](../runs/alfworld_ma_balanced_fast_v1r2/CONCLUSION.md)
- [新候选主结果](../runs/alfworld_ma_balanced_fast_v1r2/RESULT.json)
- [独立复核及实际速度](../runs/alfworld_ma_balanced_fast_v1r2/INDEPENDENT_AUDIT.json)
- [新候选运行目录](../runs/alfworld_ma_balanced_fast_v1r2)
- [首次冷启动失败](../runs/alfworld_ma_balanced_fast_v1/FAILED.json)
- [速度实验执行器](../whale_local/sao_bridge/training_speed_campaign.py)
- [隔离加速依赖](../requirements/training-acceleration.txt)

以下为历史记录；运行状态以本页顶部和对应运行目录为准。

## 2026-10-06 早期汇总：人工提示候选未通过双种子验证，总目标未完成

已重新读取四个完整评测矩阵，核验原生报告、题目/种子/token预算及释放记录，
并使用预先固定的分析代码独立重算，结果与 RESULT.json 一致。

| 固定评测种子 | 历史 W1+h2 | MA-SAO Solver + 固定人工提示 |
| --- | --- | --- |
| 20261005 | 92% | 93.5% |
| 20261006 | 90.5% | 90.5% |
| 两种子合并 | 91.25% | 92% |

合并配对差值为 +0.75 个百分点，题目聚类 bootstrap 95% 区间为 [-2.5, 4] 个百分点。
候选未严格超过历史92%，区间下界也未大于0，因此预设优势验证失败。
此前开发集94.5%保留为探索性结果，不能替代验证结果；人工提示改动不能归因于Engineer学习。
验证完成1600次环境任务，无新增训练；服务已成功退出。

- 原联合训练初始化对照：MA64初始化最终H0组合90%，W1初始化92%；两组各48轮、640条消费训练轨迹。
  两组均未选出达到替换门槛的新Engineer策略。累计研究消费训练轨迹7528，非总token/FLOPs口径。
- 独立训练重复：新种子1000043已于2026-10-06 14:34 UTC启动。驱动与Solver在node4、
  Editor在node3均由systemd托管；启动前释放审计、冻结输入/模型/源码哈希及训练前置检查通过。
  两端已进入原生训练栈，暂无训练终点或评测结果，不得据此主张复现或效果改善。
- SAO–GRPO旧对照复核：同192条Engineer训练轨迹，W1上SAO90.5%对GRPO88%，差异区间包含零；
  W0迁移上SAO74%对GRPO83.5%。不支持SAO普遍优于GRPO；checkpoint选择、环境交互及计算量不完全匹配。
- 全部200题为反复使用的开发题，不构成未见任务泛化证据。总目标仍未完成。

权威结果与执行记录：

- [联合训练主结果](../runs/alfworld_solver_initialization_study_v1/PRIMARY_COMPLETE.json)
- [人工提示开发结果](../runs/alfworld_target_explicit_hint_study_v1/RESULT.json)
- [双种子完整验证](../runs/alfworld_explicit_hint_verification_v1/RESULT.json)
- [独立重复设计与运行目录](../runs/alfworld_explicit_hint_training_replication_v1/DESIGN.json)
- [算法比较证据复核](../runs/alfworld_algorithm_evidence_audit_v1/RECHECK.json)

以下为历史进度记录，不能作为当前运行状态。

## 2026-10-05T06:30:13.238620+00:00 anchor选择完成，保留h0，配对主评测配置一致

ANCHOR_SELECTION_AUDIT.json重算全部9份原生48题报告，对应10逻辑变体。
最佳新补丁43/48，h0为42/48，空补丁40/48，未达到多2题门槛；7/8提案有效，
无效proposal06按既定规则复用空补丁报告，实际选择成本432环境。两组选择共912环境。
两组主评测使用相同h0及相同200题、seed、temperature、token/并发预算，final47 Solver。
实验组200题已提交，无准确率结果。两组均选h0，后续固定h0诊断可复用主结果。


## 2026-10-05T06:15:35.904515+00:00 anchor正式48轮训练独立审计通过，终点评测启动

ANCHOR_FINAL_TRAINING_AUDIT.json独立重算与驱动一致，两组训练预算相同。
Editor29Actor/64Critic、Solver45Actor/96Critic；ValueNorm192/224（初始128）。
384环境路由和奖励、640消费/768生成/128影子轨迹及训练释放记录通过。
分支累计消费6088（包含失败保守计费），恢复无新增评测。
Engineer和Solver服务成功退出；驱动4033201持续运行，anchor评测PID4126635 active。
固定final48选补丁及200题主评测正在运行；目前仅control成绩91%/87%，anchor无准确率。


## 2026-10-05T05:51:28.000495+00:00 anchor正式实验组前36轮独立审计通过

ANCHOR_ROUND36_AUDIT.json核验Editor21Actor/48Critic、Solver33Actor/72Critic；
检查点及ValueNorm176/200（初始128）通过。288环境、576角色回执、实际路由及奖励一致。
前缀480消费/576生成/96Editor影子轨迹；累计已审计消费5928，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。


## 2026-10-05T05:26:40.684705+00:00 anchor正式实验组前24轮独立审计通过

ANCHOR_ROUND24_AUDIT.json核验Editor13Actor/32Critic、Solver21Actor/48Critic；
检查点及ValueNorm160/176（初始128）通过。192环境、384角色回执、实际路由及奖励一致。
前缀320消费/384生成/64Editor影子轨迹；累计已审计消费5768，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。


## 2026-10-05T05:00:57.873556+00:00 anchor正式实验组前12轮独立审计通过

ANCHOR_ROUND12_AUDIT.json核验Editor5Actor/16Critic、Solver9Actor/24Critic；
检查点及ValueNorm144/152（初始128）通过。96环境、192角色回执、实际路由及奖励一致。
前缀160消费/192生成/32Editor影子轨迹；累计已审计消费5608，未来在途另计。
驱动4033201与Solver4033353经systemctl确认继续active；保持final48评测，无新准确率。


## 2026-10-05T04:40:09.849809+00:00 anchor首个三轮排程核验通过

ANCHOR_FIRST_CYCLE_AUDIT.json核验两角色已提交前3轮指标；Editor0Actor/4Critic，
Solver0Actor/6Critic（仍处warmup）。第三轮Editor不更新Actor/Critic且无ValueNorm更新指标，
两端peer为unit。关联ANCHOR_FIRST_ROUTING_AUDIT的8/8实际h0和16奖励回执。
前缀40消费/48生成/24环境/8影子，累计已审计消费5488；后续在途工作另计。
尚未审计检查点或参数逐位不变，不主张效果改善。驱动4033201、Solver4033353、
Editor2283910经systemctl确认运行；完整48轮终点不变。


## 2026-10-05T04:29:43.890026+00:00 control离线恢复完成，正式anchor已启动

原始200题h0=182/200(91%)、空补丁=174/200(87%)，完整覆盖、逐题seed/token、
哈希绑定及分数均核验。相对历史92%为-1pp，配对95%CI[-5.5,+3.5]，9改善/11退化；
未达到总目标，不将选择集93.75%当主成绩。
recover_control_endpoint.py离线发布control/matched_COMPLETE.json并绑定matched_RECOVERED.json；
原始根FAILED.json、control/matched_FAILED.json及内层RELEASED均保留，未重跑评测。
第一次离线断言发现原生config额外默认evaluation_mode/monitor_interval字段，确认原生代码后
显式纳入校验；无模型调用或部分终点发布。恢复新增训练0、评测0。
ORCHESTRATION_RECOVERY.json固定恢复脚本、原失败、结果与全部原输入哈希。
原driver.py及冻结v3源码不变，driver_recovery.py只执行未开始的anchor然后原定比较/诊断。
新驱动alfworld-anchor-study-r2-recovery-v1 PID4033201；anchor Editor node3 PID2283910，
Solver node4 PID4033353，经systemctl确认全部active。原驱动failed属历史记录；
监测新的RECOVERY_FAILED.json判断恢复分支异常。保持原48轮/640消费预算和最终选择规则。
当前已完成分支消费5448，anchor在途新增另计；两组全部完成预计6088。


## 2026-10-05T04:26:42.257959+00:00 control原生200题完成，包装层GPU释放检查失败

原生h0=182/200(91%)、空补丁=174/200(87%)；逐题覆盖、seed、token预算及准确率核验通过。
selection480+主评测400环境均保留。内层RELEASED将PID4029906列为foreign，
此前gpu_process_samples明确将其识别为filler；外层及重新采样均foreign=[]且filler active。
驱动与control evaluate已经failed，anchor未启动；不得继续声称任务仍在运行。
证据CONTROL_NATIVE_RESULT_RELEASE_FAILURE_AUDIT.json。原失败文件和报告保持不变，
下一步恢复编排并复用原生结果，不重新抽样或改写失败历史。尚未完成两组比较。


## 2026-10-05T04:10:06.230642+00:00 anchor control选择完成，保留h0

CONTROL_SELECTION_AUDIT.json独立重算10份原生48题报告，覆盖及分数一致。
最佳新补丁45/48，h0为44/48，空补丁43/48；未达到至少多2题门槛，保留h0。
全部8候选有效，选择成本480环境；proposal05为0/48，仍保留其结果。
200题主评测request使用final47 Solver及h0，seed6000045、temperature1、4096tokens；
目前未完成主终点，不将选择分数当作准确率或超过92%的证据。


## 2026-10-05T03:58:51.913134+00:00 anchor正式control完整48轮训练独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_FINAL_TRAINING_AUDIT.json。
独立重算与驱动TRAINING_AUDIT完全一致：Editor29Actor/64Critic，Solver45Actor/96Critic；
最终ValueNorm192/224（均含初始128）。完整384环境的补丁路由、奖励来源、
生成/消费账本及训练资源释放记录通过；640消费、768生成、128Editor影子轨迹。
含既有失败保守计费后累计5448消费角色轨迹，后续实验另计。
驱动3868829与control评测4005561均经systemctl确认active。
selection-47实际request/config均使用globalstep47的最终Solver及8份最终Engineer提案；
48题、seed6000045、temperature1、solver_tokens4096。当前尚无all200主终点结果。
固定h0实验组正式训练尚未开始；保持既定顺序与选择规则，不以中途结果调参。


## 2026-10-05T03:31:11.983892+00:00 anchor正式control前36轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND36_AUDIT.json。
Editor21Actor/48Critic、Solver33Actor/72Critic，ValueNorm176/200（初始128）；
36轮指标、288环境回执、补丁路由与奖励来源、检查点通过核验。
前缀480消费/576生成角色轨迹，其中96Editor影子轨迹；未来在途工作另计。
含此前失败保守计费累计5288消费。Editor2191253、Solver3911557、驱动3868829
经systemctl确认持续运行。未评测中间检查点，保持final48和后续配对比较。


## 2026-10-05T03:04:55.677281+00:00 anchor正式control前24轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND24_AUDIT.json。
Editor13Actor/32Critic、Solver21Actor/48Critic，ValueNorm160/176（初始128）；
24轮指标、192环境回执、实际补丁及奖励来源、检查点均通过核验。
此前缀320消费/384生成角色轨迹、64Editor影子轨迹，未来在途工作不混入。
包含此前失败保守计费的分支累计5128消费，后续另计。
node3 Editor2191253、node4 Solver3911557、驱动3868829持续运行。
仅执行训练审计，未评测中途检查点；固定第48轮主要终点不变。


## 2026-10-05T02:40:45.341112+00:00 anchor正式control前12轮独立审计通过

证据 runs/alfworld_anchor_retention_study_v2/CONTROL_ROUND12_AUDIT.json：
Editor5Actor/16Critic、Solver9Actor/24Critic，累计ValueNorm144/152（初始128）；
两端检查点、更新排程、96环境回执及奖励来源一致。前缀160消费/192生成角色轨迹，
其中32Editor影子轨迹，未来在途轨迹不混入。此前4808消费+此前缀160=4968，后续另计。
node3 Editor2191253、node4 Solver3911557、驱动3868829继续active。
未评测中途检查点；固定48轮终点、两组比较及92%目标不变。


## 2026-10-05T02:11:58.941758+00:00 两组anchor联调完整通过，正式对照48轮已启动

当前根 runs/alfworld_anchor_retention_study_v2，冻结源码 anchor-retention-v3。
PAIRED_SMOKE_FINAL_AUDIT.json独立重算两组完整审计：两组均Editor3Actor/8Critic、
Solver5Actor/12Critic；最终ValueNorm136/140（含初始128）；补丁路由、奖励、
影子提案及资源释放全部通过。两组共160消费/192生成角色轨迹、96环境；
含之前失败80保守收费后累计4808消费，正式新增单列。
正式control已从原H0/MA64及对应critic重新初始化，未使用smoke权重；
node3 Editor2191253、node4 Solver3911557均active，驱动3868829。
48轮、32joint+16Solver-only，每组640消费/768生成/384环境；固定final48评测。
control训练评测后自动anchor训练评测及固定h0诊断。尚无正式准确率；不挑中途检查点。


## 2026-10-05T01:48:44.844271+00:00 anchor 对照smoke完整通过，固定H0实验组smoke已启动

当前 runs/alfworld_anchor_retention_study_v2；源码 anchor-retention-v3。
独立重算对照TRAINING_AUDIT通过：6轮，Editor 3Actor/8Critic、Solver 5Actor/12Critic；
最终ValueNorm为136/140（均含初始128），路由、团队奖励、影子提案及资源释放符合协议。
证据SMOKE_CONTROL_FINAL_AUDIT.json。新增80消费/96生成角色轨迹、48环境；
含此前失败保守计成本后累计4728消费，实验组新增另计。
固定H0实验组服务已启动并核实active：node3 Editor2173233，node4 Solver3890349，
驱动3868829继续运行。两组smoke通过后自动执行正式48轮配对实验；无新准确率。


## 2026-10-05T01:24:55.619876+00:00 anchor 审计修复冻结为 v3，恢复研究 v2 已启动

当前运行根 runs/alfworld_anchor_retention_study_v2，源码 runs/source-snapshots/alfworld-anchor-retention-v3。
仅修正初始Critic继承ValueNorm累计数的检查，学习算法及输入逐字节不变；63项测试通过。
RECOVERY_MANIFEST.json 固定输入、参数等价性证据；原始 v1 失败记录不改写。
已复用32768 token验证：实际Actor/Critic引擎文件哈希不变，零新增memory计算。
驱动 alfworld-anchor-study-r2-v1 PID3868829；对照smoke Editor node3 PID2154960、
Solver node4 PID3868981，均经systemctl确认active。完成两组smoke并审计后自动正式配对比较。
此前成本按4568+失败smoke80=4648消费轨迹保守结转；全部新实验完成预计6088。
原失败smoke为72已提交消费、96生成、48环境，最后Solver未提交更新的成本差异单列。
无新准确率结论，历史92%目标及预先定义的验证协议不变。


## 2026-10-05T01:22:20.023279+00:00 anchor smoke 因继承 ValueNorm 计数审计缺陷失败

Checkpoint audit omitted inherited ValueNorm update count 128. Editor native training completed but wrapper falsely failed; peer failure interrupted Solver final checkpoint save.

修正 joint_health.py：预期累计数=初始critic统计updates+本次排程critic_steps；
2个回归案例通过，并继续拒绝额外更新。原生学习算法未修改，冻结v2源保持原样。
原始smoke_control保留：48环境、96生成、72条已提交消费，保守按80条计成本。
两节点研究GPU已释放，filler保留。恢复需新根目录及新冻结源，重新运行完整配对smoke。
证据 runs/alfworld_anchor_retention_study_v1/VALUE_NORM_AUDIT_FAILURE.json；当前无活跃研究训练。


## 2026-10-05T01:01:22.648337+00:00 anchor 内存验证通过，对照组原生联调已启动

32768 token 内存验证已通过：每 rank Actor 2 次、Critic 6 次更新，梯度有限且非零，
reference 输出一致性审计通过；memory_recovery/result.json 确认 GPU 释放与 filler 保留。
24 条合成验证轨迹单列，不计入环境研究训练。MEMORY_COMPLETE.json 已由驱动发布。
当前驱动 alfworld-anchor-study-recovery-v1 PID3832655 active；smoke_control 的
node3 Editor PID2135770、node4 Solver PID3845623 均经 systemctl 确认 active，
两端正在初始化 rollout 服务。尚无联调 TRAINING_AUDIT、正式终点或新准确率。
下一门禁为两组各 6 轮原生联调：实际补丁路由、inactive Engineer 更新隔离、
peer 权重规则、生成/消费成本、模型及 GPU 释放全部通过，才进入正式比较。
冻结协议、输入和源码不变；保留原始网络故障记录，不因观察超时重启任务。


## 2026-10-05T00:44:18.103254+00:00 当前：anchor保留实验已冻结，内存验证恢复运行

上一项单位队友权重消融已完整结束：h0=168/200(84%)，空补丁=145/200(72.5%)。
相对原对照87.5%/81.5%，差异分别-3.5pp[-8,+1]、-9pp[-14.5,-3.5]；
未超过历史92%。独立审计：runs/alfworld_peer_weight_ablation_v1/FINAL_AUDIT.json。
该分支此前累计消费4568条研究角色轨迹；负面结果和全部成本保留。

当前 runs/alfworld_anchor_retention_study_v1：固定h0经验是否保留兼容性？
两组均为joint/joint/Solver-only排程；只在Solver-only回合比较采样补丁与固定h0。
Engineer在这些回合不更新Actor/Critic/ValueNorm/LR，但仍生成影子提案并计成本。
两组联合回合bounded peer；Solver-only回合unit peer；真实proximal交换保留。
每正式组48轮、640消费/768生成角色轨迹、384环境回合；每smoke组6轮、80消费/
96生成。两组各一份smoke，全部通过后按control训练/评测、anchor训练/评测执行。
固定final48，8候选selection48、h0需多2题才替换；all200主终点及fixed-h0诊断。
协议protocol.json、四组spec/data已冻结，勿重跑prepare_runs.py或改写源快照。

正式代码源为runs/source-snapshots/alfworld-anchor-retention-v2；内存检查用v1，
两者terminal_actor和memory引擎哈希一致，v2只更新离线审计及测试。
原始memory因节点IP部分端口不可达停在配置阶段，零模型前向/更新，已停止释放；
证据MEMORY_NETWORK_RECOVERY.json。恢复服务node4 alfworld-anchor-memory-recovery-v1
PID3832650，显式AREAL_HOST_IP=127.0.0.1，已进入32768 token前向与Critic更新。
等待驱动alfworld-anchor-study-recovery-v1 PID3832655；完整内存审计通过后自动启动。
恢复输出memory_recovery，驱动driver_recovery.py，原始记录保留。当前无smoke/
正式轨迹，无新准确率。仅node3/node4授权，不干预bruli-h200-load。

## 2026-10-04 22:47 unit消融完成首个8轮检查点

两端均提交8轮指标、完成第8轮检查点并进入下一轮。每角色64条消费、
5轮Actor更新；已检查指标有限，所有实际peer权重为1。运行中的服务仍为
node3 editor PID2030120、node4 solver PID3719553；driver3691683和
secondary3698522等待各自门禁。PROGRESS.json记录当前健康检查。
不评估/挑选中途检查点，保持第32轮固定终点；当前无新准确率结论。

## 2026-10-04 22:25 unit权重smoke通过，正式32轮训练已启动

`SMOKE_FINAL_AUDIT.json`独立重验：两角色各16生成=16消费、1Actor/4Critic
更新，2轮实际peer_weight=1、clipped=0、effective_fraction=1；原始队友
诊断仍保留，Solver两轮raw clipping均75%。正常结束且GPU释放。新增32角色
训练轨迹/16联合环境，当前研究路径累计消费4056（主训练新消费另计）；
24合成memory轨迹单列，smoke模型不用于正式初始化。

正式 `alfworld-peer-weight-unit-editor-v1`（node3 PID2030120）和
`alfworld-peer-weight-unit-solver-v1`（node4 PID3719553）均active。冻结32轮
512角色轨迹、3轮critic warmup、每角色29Actor/64Critic更新，sole learning
change=unit队友权重，team奖励不变。driver3691683自动等待账本审计/第32轮
8候选选择/200题主评估；secondary3698522等待后续固定h0诊断。
当前只有执行验证通过，没有新准确率结论；历史92%与后续优越性门槛不变。

## 2026-10-04 22:10 奖励归因实验全部结束；unit权重smoke已启动

credit-study `FINAL_AUDIT.json`核验全部诊断报告哈希/题目/seed/token预算及
18项配对比较。同期W1+h2为90%，empty87.5%；实验selected78%比同期
基线低12pp，95%[-18,-6]，9改善33退化。历史92%阈值不变。
原主对照87.5%、实验78%保留；诊断对照selected89%、h087.5%、empty81%，
实验h0/selected78%、empty73%。固定h0差-9.5pp[-15,-4]，排除仅由
候选选择解释下降；并未证明单一训练机制。单训练种子/重复开发题限制保留。
成本：两组训练1024角色轨迹，研究路径此前累计4024；选择912+主评估800+
诊断1400=3112环境回合，不含训练环境回合。恢复未重跑原终点。

node3诊断正常结束并释放GPU，driver已自动启动
`alfworld-peer-weight-smoke-editor-v1`（node3 PID2016561，native2016647）和
`alfworld-peer-weight-smoke-solver-v1`（node4 PID3704269，native3704357）。
两轮unit smoke进行中，待实际指标/账本验证后自动进入unit32轮训练。
主driver3691683、secondary3698522继续活跃，勿重复启动。

## 2026-10-04 22:06 新Actor全上下文原生验证通过

`alfworld_peer_weight_ablation_v1/MEMORY_COMPLETE.json`已由driver独立审计生成，
执行源哈希匹配：4个rank各2Actor/6Critic更新，Actor梯度5.4123/2.9354，
参数样本变化、参考输出bitwise一致；32768上下文+1024 Critic前缀。
峰值Actor119.073GiB、Critic98.250GiB。node4 GPU释放且filler保留，
memory服务正常结束。24条合成轨迹，无模型checkpoint/环境回合。

主driver已进入等待node3诊断阶段（同期baseline PID2005989仍活跃），
随后自动执行smoke2轮及unit32轮；正式unit训练尚未开始。secondary等待
PRIMARY_COMPLETE。勿重复启动这些服务或重写冻结配置。

## 2026-10-04 21:59 unit消融后续固定h0诊断已排队

`alfworld-peer-weight-secondary-v1`等待新消融PRIMARY_COMPLETE后执行secondary.py。
SECONDARY_PROTOCOL在主训练前明确：selected与h0哈希相同则复用主报告（0新增）；
否则原生matrix评估h0+empty共400回合。原协议只计h0的200上限已在该补充协议
更正，主终点/选择/训练预算不变。对比此前control固定h0及本次credit诊断基线；
不将复测或较低同期基线替代历史92%。内存检查与node3基线仍运行。

## 2026-10-04 21:54 固定h0诊断同样显示Solver下降

诊断control/treatment已完成1000环境回合：固定h0 87.5%→78%，
差-9.5pp，配对95%[-15,-4]，7改善26退化；空补丁81%→73%，
差-8pp，95%[-14,-2]；所选系统复测89%→78%，差-11pp，
95%[-16.5,-6]。`diagnostics/PARTIAL_CONTROL_TREATMENT.json`固定报告哈希。
固定相同h0后仍下降，不能仅以候选选择解释；单训练实现/重复开发题限制保留。
同期baseline已在node3启动（PID2005989），诊断尚未全部结束。

新unit消融memory服务仍活跃，已完成critic-only warmup两次更新，
rank0梯度35.34/98.76且各426张量样本改变；Actor更新与剩余Critic更新
及释放尚待验证。driver仍等待memory和原诊断门禁，正式训练未开始。

## 2026-10-04 21:51 新消融原生验证与门禁调度已启动

冻结源 `runs/source-snapshots/alfworld-peer-weight-ablation-v1` 固定5个模块改动，
实现unit队友权重并让memory工具使用node4/独立缓存；学习默认bounded不变。
11项相关测试通过（缓存路径断言随独立缓存改动更新，首轮该断言失败已记录）。
`VALIDATION_ADDENDUM.json`明确新增费用：全上下文合成24轨迹、2Actor/6Critic
更新但无checkpoint/环境；两角色smoke2轮32角色轨迹/16环境，正式32轮512角色
轨迹。smoke模型不进入正式初始化，单独记账。

node4 `alfworld-peer-weight-memory-v1`运行H0、32768上下文+1024 Critic前缀检查；
`alfworld-peer-weight-study-v1`已启动，依次等待新memory结果并独立审计、等待
node3诊断COMPLETE+RELEASED，再执行smoke/严格unit指标审计/正式训练/固定终点
评估。smoke与unit输入已准备，勿重跑prepare_runs.py；driver脚本与配置已在
DRIVER_MANIFEST.json锁定。当前正式训练尚未开始。旧诊断FAILED历史记录不会
作为当前运行失败证据；用服务和最终完成/释放门禁。

## 2026-10-04 21:45 固定权重异常复现，队友权重消融已立项

`runs/alfworld_fixed_weight_probability_replay_v5/AUDIT.json`独立复算与校验通过：
8条初始Engineer轨迹7条越过factor2阈值，原记录/重算logsum最大差1.35e-6，
循环移位后tokenwise差为0。optimizer=None，GPU释放已验证。完成16条重算，
加v4失败前向8条，共24条前向，0生成/参数更新/新增环境回合。此证据排除
本批次的Actor更新与所试批次顺序作为偏差解释，不证明准确率因果影响。

新冻结协议 `runs/alfworld_peer_weight_ablation_v1/protocol.json`：team奖励不变，
只把两角色队友权重设为1；保留原始比率交换/诊断、PPO自身比率、KL及Critic。
复用credit/control作既有同初始化/数据/种子/预算对照（原主结果87.5%），新增
32轮512角色轨迹，仍第32轮8候选、selection48至少+2题才替换h0，再评估200题。
明示非同期单训练实现，诊断复测不替换主结果，>92后仍须既有优越性验证。
主代码新增peer_weight_mode可选unit，默认bounded不改；7个相关测试通过。
原生smoke/训练调度尚未完成，勿将方案视作运行完成。node3诊断仍在运行。

## 2026-10-04 21:41 对照诊断完成；概率重算v5运行

node3已完成control诊断600回合，compact reports记录selected89%、h087.5%、
empty81%，各200题。它们是诊断复测，不替换主结果87.5%/81.5%。当前进入
treatment-attempt-1（PID1994757），同期基线尚未完成。

node4概率v4完成第一遍8轨迹前向，但返回RTensor，CPU摘要调用detach失败；
8条前向成本保留在COST.json，无结果取回、0参数更新/环境回合，GPU已释放。
v5冻结源使用框架RTensor.localize读取结果，继续相同轨迹/权重/指标；启动
`alfworld-fixed-weight-probability-replay-v5`，输出同下划线名称目录。仅v5当前
运行，不重启v1–v4。

## 2026-10-04 21:38 概率重算v4与权重分组审计

v3创建4个RPC工作进程，但第4个端口通过节点IP无法访问，回环health返回200。
停止自有v3服务并验证GPU释放，无模型初始化/重算/参数更新/新增环境回合。
v4沿用冻结v2源与输入哈希，预创建目录、controller也设置AREAL_HOST_IP=127.0.0.1；
服务 `alfworld-fixed-weight-probability-replay-v4` 已启动，结果待产生。主模块已
补上目录/回环设置，但运行的冻结v2未改动。

`IMPORTANCE_GROUP_AUDIT.json` 对两组各256条/角色做事后描述性分组。
Actor更新阶段全部232条均含滞后版本；Engineer权重均值对照0.8204、实验0.8653，
下截断133/232、129/232，上传递给Solver。成功/失败平均权重对照0.848/0.754、
实验0.827/0.961；加权成功比例由70.69%到73.07%、71.55%到68.39%。这只是
已观察奖励重加权，不能替代真实优势/梯度或因果准确率证据。Solver自产权重
接近1（均值0.9903/0.9850）。0新增训练/环境回合。

## 2026-10-04 21:33 概率重算当前运行版本v3

v2在LocalScheduler构造前发现engine目录尚不存在，0工作进程/重算/更新/
环境回合，FAILED与RELEASED保留。v3预创建engine与name_resolve目录，
沿用冻结v2代码与原始输入哈希，服务 `alfworld-fixed-weight-probability-replay-v3`
已启动，输出 `runs/alfworld_fixed_weight_probability_replay_v3`。不要重启旧版本。
node3诊断仍在control-attempt-1（PID1983909），结果未齐。

## 2026-10-04 21:31 概率诊断配置恢复

概率重算v1在配置校验阶段失败，未创建GPU工作进程，0重算/更新/环境回合；
原因是框架调度策略不可设为None。已保留日志与FAILED.json。恢复v2保留
原模型/轨迹哈希、指标和顺序，仅沿用默认调度策略并消除未使用的rollout
凭据环境插值依赖；实际结构化配置校验通过。新冻结源与输出分别为
`alfworld-fixed-weight-probability-replay-v2` / `alfworld_fixed_weight_probability_replay_v2`，
同名systemd服务已启动。结果仍待原生GPU执行，不把启动当作完成。

## 2026-10-04 21:28 固定权重概率重算诊断已启动

训练日志审计 `TRAINING_DYNAMICS_AUDIT.json` 所查指标均有限，但 Solver 使用的
队友重要性权重平均约68–70%被截断。首三轮只更新Critic期间，Engineer的
24条轨迹已有18条越过[0.5,2]阈值，不能全部归因于Actor更新。
这只是机制线索，尚未证明对准确率的影响。

新协议 `runs/alfworld_fixed_weight_probability_replay_v1/protocol.json` 已锁定
原始首轮8条Engineer单次调用的tokens、行为概率、训练配置与初始权重哈希。
node4服务 `alfworld-fixed-weight-probability-replay-v1` 用原生FSDP以原顺序和
循环移位顺序各重算一次，共16条；Optimizer=None，0新增训练/环境回合。
检查原始偏差能否复现以及批次顺序敏感性；2个对齐/数据完整性测试通过。
结果尚未产生。node3固定h0/同期基线诊断继续运行，勿重复启动。

## 2026-10-04 21:16 奖励归因主实验为负结果，诊断继续

`runs/alfworld_validity_credit_study_v1/PRIMARY_AUDIT.json`独立核验两组报告的
题目覆盖、随机种子、token预算、模型轮次、哈希和GPU释放，主比较一致：
对照组175/200=87.5%，实验组156/200=78%；实验组减对照组-9.5pp，
题目配对95%区间[-15,-4]pp，8题改善、27题退化。空补丁对照81.5%、
实验73%，差值-8.5pp，区间[-14.5,-2.5]pp。report152同样为负，仍为
重复使用的开发题，并非未见测试集。只有一个训练种子，不能外推为跨种子结论。

研究问题“取消无效补丁成功奖励能否改善最终系统”在本次实验中未得到支持。
空补丁也下降，故不能只归因于候选选择；下一步用已预定的固定h0与同期
W1+h2矩阵区分Solver变化、补丁兼容性和推理波动。node3诊断服务
`alfworld-validity-credit-diagnostics-recovery-v2`已确认仍活跃，正在接续。
两组主评估与训练均已结束，勿重跑。历史失败记录保留，恢复已明确记账。
本实验两组训练1024条，累计研究路径4024条；选择+主评估1712环境回合，
诊断新增回合待完成后计入。没有>92候选，不触发优越性验证，目标仍未达成。

## 2026-10-04 20:39 两组训练均完成，实验组候选评估已启动

`runs/alfworld_validity_credit_study_v1/BOTH_TRAINING_AUDIT.json`：两组各512条
生成/消费角色轨迹，每角色32轮、29次Actor及64次Critic更新，均正常结束。
实验组无效补丁31/256=12.11%，对照组20/256=7.81%；实验组19个无效补丁
对应任务成功，但Engineer奖励之和为0，对照组相应14个样本奖励之和为14。
干预实际执行正确，但训练无效率没有显示改善；单训练种子，不能替代终点评估。
本研究路径累计训练消费4024条，含此前失败路径及32条预算smoke。

node4 `validity-credit-treatment-evaluate-v1`运行，固定第32轮候选生成/选择/
200题评估；对照组正式结果仍为87.5%。恢复调度与node3诊断等待器继续运行。
两组训练服务已正常退出，不要重新启动。尚无实验组最终准确率，目标未达成。

## 2026-10-04 对照组终点评估完成，汇总故障已恢复，实验组已启动

`runs/alfworld_validity_credit_study_v1/control/matched_COMPLETE.json`：固定32轮，
proposal03为175/200=87.5%，空补丁163/200=81.5%。对历史W1+h2的92%差值
-4.5pp，配对95%区间[-10,+0.5]pp；对沿用的起点90%差值-2.5pp，
区间[-6.5,+1.5]pp。单训练种子、重复使用开发题，不能宣称达到总目标。
对照组训练预算512条角色轨迹、选择432回合、终点评估400回合已完成。

原评估已完成后，汇总因缺少initial_joint_harness_report.json失败。
`SUMMARY_RECOVERY.json`及`recover_summary.py`保留输入哈希，补齐复用报告，
仅执行原冻结代码的汇总段；新增训练/环境回合均为0，原matched_FAILED和
总调度FAILED保留。`RESOLVED_FAILURES.json`只豁免哈希匹配的已恢复故障。
诊断等待器首次恢复因跨节点目录权限失败；权限已按uid1009/gid1008验证，
失败与恢复日志保留。不要直接据历史FAILED文件重启当前正在运行的实验。

当前：node3 `validity-credit-treatment-editor-v1`、node4
`validity-credit-treatment-solver-v1`均运行，仍使用原study-v1冻结训练代码。
调度：node4 `alfworld-validity-credit-study-recovery-v1`。
诊断等待：node3 `alfworld-validity-credit-diagnostics-recovery-v2`。
新恢复快照仅更改validity_credit_study.py的报告依赖/等待逻辑，学习代码不变。
等待实验组32轮、相同选择与终点评估，再做预定固定h0/同期基线诊断。

## 2026-10-04 17:25 GPU预算修复验证通过，正式对照训练已启动

runs/alfworld_joint_budget_gpu_smoke_v1/FINAL_AUDIT.json已核验：H/W均完整2步，
各1次Actor、4次Critic更新；各16条唯一生成/消费轨迹，16个配对环境结果；
无额外采样、无第0轮回绕故障，保存后正常退出并释放GPU。
最终Actor梯度范数H1.6631/W0.7823，有限非零；Critic梯度也有限非零。
这证明预算修复在真实原生训练中收尾成功，不是准确率提升证据。
累计本研究路径训练消费2968+32=3000条，未将生成数重复加上。

自动接续control32轮已启动：node3 validity-credit-control-editor-v1，
node4 validity-credit-control-solver-v1，MainPID与joint_native均核验在线。
root runs/alfworld_validity_credit_study_v1/control。仍用原H0/MA64起点，
不继承smoke权重；control奖励team，treatment奖励validity_gated只改非法H奖励。
主终点固定round32、selection48规则不变；当前尚无新准确率。
顺序：control训练/评估→treatment训练/评估→node3固定h0和W1+h2诊断。
驱动alfworld-validity-credit-study-v1与诊断队列保持active；继续监控失败/真实进度。


## 2026-10-04 17:10 辅助对照完成，GPU预算修复验证已启动

branch_ablation_v2/COMPLETE与FINAL_AUDIT已核验协议/报告SHA及GPU释放：
原h11278%，仅禁用错误物品阻拦85.5%，删除整钩子86.5%。
主比较+7.5pp[+4,+11.5]；整钩子相对单分支+1pp[-1,+3]，不支持额外收益，
亦非等价证明。辅助差异跨服务会话；空harness重测78→77.5%，-0.5pp[-2,+1]。
原3组600完整episode复用；恢复新增400，失败尝试最多800不得漏记。
这是固定权重的人工机制实验，不是新学习模型达标成绩。

GPU smoke队列自动接续成功：17:08启动node3 editor/node4 solver，
units alfworld-joint-budget-gpu-smoke-v1-editor/solver。两节点MainPID均已核验active，
日志显示真实AReaL训练引擎初始化，尚无训练完成结论。
后续credit-study驱动仍等待精确两轮/各1actor与4critic更新、各16唯一轨迹核验。
通过后自动执行control32轮→固定终点评估→treatment32轮→固定终点评估，
再由node3诊断服务执行固定h0和W1+h2重测。继续监控，勿重复启动或改冻结源。


## 2026-10-04 单分支主比较完成，辅助组恢复中

单分支v1在第四变体推理ReadError后整体FAILED；前三份200题native报告已完整落盘，
经checked_native_report校验后保留，未按分数选择：空78%，原h11278%，
仅禁用错误物品阻拦85.5%。主比较+7.5pp[+4,+11.5]，15改善/0退步，
PRIMARY_PARTIAL_RESULT.json明确primary_complete_auxiliary_failed，原FAILED不删除。
这支持该分支在当前固定MA64/复用开发题条件下有害；不是学习成绩或>92%证据。

只恢复缺失的without_guard辅助组，runs/alfworld_h112_branch_ablation_v2，
node4 unit alfworld-h112-branch-ablation-v2；沿用冻结推理源码及config，
resume.py和所有复用报告/输入固定SHA。额外跑without_guard+fresh_no_patch共400回合；
原3组完整600回合保留，fresh_no_patch只作跨次波动诊断，不覆盖原空对照。
辅助差异跨服务会话，必须披露。失败尝试环境成本上限800，另加恢复400。

原GPU smoke队列因前置失败而退出，0训练启动；新unit
alfworld-joint-budget-gpu-smoke-v1-queue-recovery等待branch v2 COMPLETE/RELEASED，
训练spec和源码不变，QUEUE_RECOVERY.json记录修订。后续credit study与diagnostics
服务仍active等待，不需重启或重复启动。两臂配置比对通过：数据完全相同，
除奖励模式外只有路径/实验名/截止时间差异；ARMS_COMPARABILITY_AUDIT.json。
总目标保持active。下一步完成辅助组、核验GPU smoke、继续32轮两臂与基线诊断。


## 持续总目标（2026-10-04用户明确授权）

持续提出研究问题、设计/运行实验并分析，直到MA-SAO超过历史W1+h2=92%，
且比较公平性/可信度经过核验。不能为达标选择性报告、改口径或把人工消融当学习成绩。
Codex active goal已创建；不要仅启动队列就把总目标标为完成，持续监控并推进。

当前单分支消融alfworld_h112_branch_ablation_v1在node4运行。下一研究问题是
非法patch回退空harness却获得成功奖励是否妨碍Engineer学习。
冻结协议runs/alfworld_validity_credit_study_v1/protocol.json：team对照与
validity_gated处理，两组各32轮、512条H+W，总1024条；同H0/MA64起点、
数据/种子/超参，仅非法patch的Editor奖励置零，Solver/环境不变。
固定round32为终点，8proposal+incumbent、selection48与2/48门槛不变，
不根据200题结果换模型。对照和处理都使用预算修复，旧v2仅作历史支持。
相关49测试通过。源码冻结runs/source-snapshots/alfworld-validity-credit-v1。

正式两臂尚未启动。先跑runs/alfworld_joint_budget_gpu_smoke_v1两轮原生GPU验证：
各16轨迹、1actor/4critic更新、总32条角色训练轨迹；必须无重复/故障且完整收尾。
队列alfworld-joint-budget-gpu-smoke-v1-queue等待单分支消融完成和GPU释放，
自动启动node3 editor/node4 solver。两个spec、协议、queue.py及日志可核验。
下一步：收单分支结果，核验GPU smoke的两role COMPLETE、metrics与完整唯一receipt；
通过后准备并依次运行同快照的32轮对照和处理，比较主终点及非法patch率。
若失败，修复并保留失败成本，不启动依赖训练。不得修改已冻结源码。


## 2026-10-04：改进分支与机制消融结果已完成

研究问题1：低学习率/KL/保留h0的方法包能否超过92%？主终点selection48选中
round40/proposal02，all200=88%，相对历史92%为-4pp，配对95%[-9,+1]pp；
round32辅助终点88.5%，不能因较高就替换主终点。训练仍是H62/W63部分完成。
同环境交叉重测：起点W+空/h0/新patch=78/89/85.5%；选中W=82/85/88%；
W1空/h2=87.5/89.5%。相对重测基线-1.5pp[-6.5,+3.5]，没有优越性证据。
固定h0权重变化-4pp[-8.5,+0.5]；空harness+4pp[-2,+10]；兼容性交互
+6.5pp[+0.5,+13]，提示协适应而非单一组件全面提升。单训练种子、复用开发题。

研究问题2：h112动作护栏是否损害表现？固定MA64，同次重测原h112=81%，
删除整钩子=87%，空=79%；+6pp[+2.5,+10]，14题改善/2题退步。
支持该钩子在当前条件下有害，但尚不能归因到其中某一个条件；不是学习方法成绩。
两组诊断分别1600/600环境episode，0新增训练；分支累计训练消费2968条。
完整结果：runs/alfworld_ma_improvement_study_v2/COMPLETE.json及
runs/alfworld_h112_guard_ablation_v2/COMPLETE.json。历史92%不被重测89.5%覆盖。

下一问题：仅禁用“错误物品阻拦”分支能否解释整钩子消融收益？已冻结并启动
runs/alfworld_h112_branch_ablation_v1，node4服务alfworld-h112-branch-ablation-v1。
固定MA64，原h112/仅禁用该分支/删除整钩子/空harness，共800episode，0训练。
主比较为单分支减原版；次比较为删整钩子减单分支。后者不显著不代表等价。
源码独立快照，相关6测试通过；其他动作及两物体任务防护代码保持原样。


## 2026-10-04 末尾预取故障与研究队列恢复

MA improve v2未完成64轮：H提交62步/496条，W提交63步/504条；
双方完整生成各512条。共同checkpoint止于round56（另有32/40/48）。
最终环境轮结束后出现version0/sample00003重入，随后ConnectError；
上游prepare_batch使用cycle_dataloader，dispatcher持续预取，可在末尾回绕。
CPU调用真实上游dispatcher复现：预算24条时提交32条；新增本地预算适配器后
只提交/交付24条。修复只在主工作树，未改运行快照，尚未GPU重训验证。
原训练失败文件保留，SHARED明确标记recovered_committed_checkpoints_after_training_failure。
既有matched/best评估已自动接管，仍按原规则选择，不能将部分训练写成完成64轮。

诊断v1遇训练失败按规则停止，未启动GPU/环境评估。终点评分出现前，已冻结
显式恢复修订到`runs/alfworld_ma_improvement_study_v2`和
`runs/alfworld_h112_guard_ablation_v2`：保留原始协议、比较/选择规则，固定
失败记录、SHARED、metrics哈希及实际消费。只放行已核验的训练失败；
任何新评估失败或记录变化仍停止。相关5测试通过。
新源码`runs/source-snapshots/alfworld-ma-diagnostics-v2`，未改正在运行的冻结训练源。
node3服务`alfworld-ma-improvement-study-v2`、node4服务
`alfworld-h112-guard-ablation-v2`等待双方原评估完成并释放GPU后执行。
研究路径累计实际训练消费1840+128+1000=2968条（不是全项目总成本）。
下方早先条目是历史状态；诊断v1已被上述显式修订取代。目前仍无新终点成绩。

## 2026-10-04 研究推进方式：问题、干预、结论

用户要求按研究问题组织工作，而非只汇报训练进度。当前MA improve v2已越过
原HTTP500故障点，双方约39轮，未完成训练后评测。原冻结训练/选择代码不变。

- RQ1：约束H/W变化并保留初始h0，是否改善组合？v2为方法包实验，
  best的selection48选择结果为主要终点，round32为辅助；不能看all200后换终点。
- RQ2：改善/退化来自W、harness还是兼容性？新冻结
  `runs/alfworld_ma_improvement_study_v1`，node3服务
  `coevolve-ma-improvement-study-v1`等待v2双方评测成功/释放后做2x2交叉诊断、
  空harness对照和同环境W1+h2重测，最多1600环境episode、0训练。
  分别报告all200/selection48/其余152题；都不是全新sealed测试集。
- RQ3：晚期h112的动作阻拦是否伤害恢复？CPU复现“建议放下错误物品却阻拦
  放到非目标位置”的矛盾；旧25个退步题中14个出现阻拦提示，尚非因果结论。
  `runs/alfworld_h112_guard_ablation_v1`、node4
  `coevolve-h112-guard-ablation-v1`等待同一前置后，固定MA64权重，只删除
  h112单个on_before_action钩子，原h112/删钩子/空harness共600episode。
  这是人工机制消融，不能当成新的学习方法成绩。
- 待验证奖励假设：v2前32轮24/256个非法patch回退空harness，其中17个获得
  团队成功奖励。奖励不等于PPO正优势，不能直接归因；下一方法候选为固定W
  的配对边际harness收益，尚未实施，不得宣称有效。

所有诊断源码与训练快照隔离，协议在v2终点评测前冻结。新诊断相关测试13项通过
（12项study/analysis/crossplay，加1项单钩子不变量）；仍需实际运行结果。
已保留具体证据JSON与分析规则，详见EXPERIMENT_LOG末尾。下一步按结果决定
单因素消融或独立重复，不能把队列/healthy状态当成超越92%。

## 2026-10-04 接手：MA改进v2已启动，尚无新成绩

仅接手最近两个Copilot会话，继续“MA-SAO超过W1+h2=92%，WHALE暂停”。
原改进v1双方各完成8轮后，round9/task190的纯文本回复带
`tool_calls: null`，被AgentRL `AgentOutput`拒绝为HTTP500；不是OOM或性能崩塌。
v1保留失败记录和checkpoint，已消费128条H+W、另有158条完整生成角色轨迹
（生成与消费有重叠，不相加）。没有训练后200题成绩。

本次只修复ALFWorld边界的可空字段：省略null `tool_calls`，保留原文本、
token成本和环境判分；未把HTTP错误转成零分。70测试及30子测试通过；真实
task190/原失败patch的CPU环境验证通过，纯文本按原环境处理至token预算零分，
无HTTP500（合成回复，1个环境episode，不是模型成绩）。

新root `runs/alfworld_ma_improve_v2`；node3 editor/matched、node4 solver/best/guard，
unit前缀`coevolve-ma-improve-v2`。独立源码快照
`runs/source-snapshots/alfworld-ma-improve-v2`，基于0bbb1cf加本次字段修复，
完整文件SHA见`SOURCE_MANIFEST.json`。启动前两节点GPU释放审计及filler身份通过。
这是从相同H0/MA64起点重新尝试，fresh Adam，**不是v1优化器无缝续训**；
保留原64轮、round32探针、低学习率/KL及incumbent保护方案。
父实验实际1840条、继承路径1024条、失败v1成本单独披露，见
`RESTART_PROVENANCE.json`。不得将此分支当成从零同预算或独立测试集证明。
下一步检查真实更新、round32与最终200题报告；目前尚未超过92%。

## 当前最高优先级：让MA-SAO超过92%，暂停WHALE

用户明确改为优先超越W1+h2=92%，WHALE暂不继续。不要启动WHALE、
旧修正SAO或调度消融队列。WHALE未完成的DP/续训代码修改保留但不纳入新MA冻结源。

交叉测评已完成并在其owner节点核验：
`runs/alfworld_ma_crossplay_v1/matched/FINAL_AUDIT.json`。
同MA64权重下：空77.5%、初始h0 90%、h64 89.5%、晚期h112 78.5%。
同MA112权重下：空79.5%、初始h0 86.5%、h64 87.5%、h112 83.5%。
固定MA64权重换h64→h112下降11pp，配对区间[-16,-6]pp，
比“跨checkpoint比较后直接说W崩了”更直接地支持harness有问题。
但h0下换成晚期W也未改善，兼容性不能忽略；重测分数不替换原始MA报告。

新MA改进分支使用`joint_campaign --from-initial --improve-from <原MA根>`：
原H0 Actor＋MA64 Solver，两个critic/ValueNorm继承MA64，新Adam；
H学习率1e-7、W学习率1e-6，均以各自分支起点为reference、KL0.02，
K2、三轮critic预热、同轮双方更新/lag1保持不变。
冻结初始h0作为部署候选，selection48 mean@1上新patch须至少多2题才替换；
与旧selection24×2预算接近，但增加开发集暴露，必须披露。
最多64新轮、每8轮保存，round32为中点探针。继承路径1024条，父实验实际
已消费1840条，不能将该分支冒充从零同1024比较。第一目标先取得>92%的真实结果。

## 最新：MA正式结果已完成，正在恢复后续研究队列

MA最终报告于2026-10-03 08:12:17UTC完成，早于08:49:31期限：
`runs/alfworld_joint_h0w0_8h_v1/FINAL_AUDIT.json`已核验完整结果。
原始W0空74%、H0+h0/W0组合85%；第64轮（1024条）空78.5%、组合87.5%；
后期选中第112轮（1792条）空80.5%、组合82.5%。历史W1+h2=92%仍保留。
匹配点相对起点+2.5pp[-2,7]；后期相对匹配点-5pp[-10,0]。
后期空harness Solver相对W0+6.5pp[1.5,11.5]，所以不是简单的W权重崩塌证据：
harness加成从匹配点9pp降为后期2pp，应优先研究harness与选择流程。
三个后期checkpoint候选均值75.78/73.70/60.16%，max选择却选中了第三个；
需要同Solver交叉harness诊断，不能凭跨模型均值直接归因Engineer。
训练实际115轮/各110次Actor、230次critic，消费1840，完整生成1872；
无OOM，按计划停止。此结果没有证明MA超过SAO或永不退化。

**后续当前状态：**
- W2恢复v1在初始评测开始前因重写只读`runtime_supplements.json`失败，0采样/更新。
  修复`2ed6700`复用`supplement_runtime`，5测试和node3实际只读路径验证通过。
  活跃恢复为`runs/alfworld_w2_recovery_v2`、node3 `coevolve-w2-recovery-v2`，
  源码快照`runs/source-snapshots/alfworld-research-controls-v4`；已完成初始评测并启动训练。
- WHALEv6在完整32K烟测`smoke-1`阶段失败，未进入baseline研究训练；
  `queue/FAILED.json`和`execution/FAILED.json`正确传播。具体根因正在node4单独排查，
  不得未经日志证实就称为OOM。失败v6及冻结源不改写。
- 修正SAO v4/v5和调度消融v2/v3都没有开始GPU训练，前置失败后已暂停。
  不要恢复旧队列；Whale修复后需新冻结/新明确依赖。
  v5 SAO协议绑定W2v2与旧Whalev6，不应在Whale失败状态下重启。

以下为此前准备/排队历史，不代表当前所有旧队列仍在运行。

## 新研究任务：恢复、WHALE基线、稳定性对照

**06:02UTC当前接手状态：** 原始H0/W0 MA双方91轮均健康、无新OOM。
`MATCHED_POINT_AUDIT.json`已核验共同global_step63/round64的完整Actor/critic，
H/W各512、合计1024条消费，各59次Actor和128次critic更新；继续训练中。
所有后续研究队列均已持久激活，当前仍只有原始MA实际占用研究GPU。

| 实验 | 活跃根目录 | 固定源码 | 持久服务/前置 |
|---|---|---|---|
| W2恢复 | `runs/alfworld_w2_recovery_v1` | `e527e19` | node3 `coevolve-w2-recovery-v1`，等待原MA matched |
| WHALE适配 | `runs/alfworld_whale_baseline_v6` | `be92302` / controls-v2快照 | node4 `coevolve-alfworld-whale-v6`，等待原MA best，再32K smoke，再完整baseline |
| 修正普通SAO | `runs/alfworld_sao_corrected_v4` | `a05800e` / controls-v3快照 | node3 `coevolve-sao-control-v4-w`等W2；node4 `coevolve-sao-control-v4-h`等WHALE完整execution |
| MA调度消融 | `runs/alfworld_ma_alternating_blocks_v2` | `a05800e` / controls-v3快照 | 两节点`coevolve-ma-schedule-v2-*`，等待修正SAO完整报告/释放 |

WHALE生产冻结是`PRODUCTION_FROZEN.json`，SHA
`dd165caffb22b85340e5047daf9cb320d71e74022d930d74300ce9dea9301933`；
已从实际冻结快照核验57个本地源文件和原始模型权重，无GPU动作。
标准`.pth`附加main路径是潜在fallback，**没有证据表明原MA实际导入了main**；
新控制严格做模块来源隔离，不因此改写旧实验有效性。
修正SAO生产协议SHA`ab78c4c9e453b6af50e3b53b5bc60586bded8cb29abefde9920c8643b7e15d42`，
已在两个实际UID下CPU验证。原v3只在准备时误拒合法AReaL依赖链接，
0训练/0GPU，保留`PREPARE_FAILED.json`；由v4替代。调度消融旧v1也未启动，由v2替代。

**时间安排调整：** node4新增`coevolve-h0w0-evaluation-reserve.timer`，
07:10UTC通过原冻结代码发布STOP，给最多3对候选选择+200题留约100分钟。
原报告截止08:49:31UTC不变，第64轮已保存；不是依据200题结果挑停训时刻。
原07:49:31仍保留为训练协议兜底。后续队列有独立超时，不能保证同日同一报告窗口
全部完成；其CPU准备、排队和smoke都不是研究性能结论。

下面为本轮研究任务的准备历史：

02:14UTC用户要求持续监督，并按研究问题组织实验：
1. 继续真实退化W2+固定h2，看有限预算内是否回升。
2. 给出WHALE的ALFWorld基线，不把Harness-R1/SAO改名成WHALE。
3. 比较MA与SAO，并检验交替更新是否与崩塌有关。

当前H0/W0 MA保持运行；02:44左右双方19轮均健康，已经越过12轮/192条，
`ROUND12_REACHED.json`只是消费里程碑，不是性能结果。
node3后续**已持久排队**`coevolve-w2-recovery-v1`，root
`runs/alfworld_w2_recovery_v1`，冻结源码`e527e19`，
快照`runs/source-snapshots/alfworld-w2-recovery-v1`。
它等待当前MA的matched完成/释放再做liveGPU审计，不抢占正在训练的MA。
从真实W2-step79及critic起步，固定原h2；新Adam（不是精确优化器续跑），
保留原LR/K，启用修正后的独立轨迹microbatch。640个未用train-only实例
1280–1919；先重测起点，再报告额外20/40/60/80步完整200题曲线。
24项针对性回归通过。不能用有限80步没恢复来证明“永远不能恢复”。

WHALE适配在CPU准备中，官方没有可引用的ALFWorld分数；需真实GPU验证后
在node4当前MA best评测释放之后运行。其RSFT/Meta-Harness与本地H0 proposer
替换要明确披露，不能称为官方原样复现。
另准备真正的修正后端普通SAO H1/W1/H2对照（192+640+192=1024），
以及保持MA其它组件不变的H16/W16阶段调度消融。后者不是普通SAO。
预计队列：node3 W2恢复、node4 WHALE之后，再做修正SAO，然后调度消融；
最终实际绑定以各run的QUEUE_LAUNCH/LAUNCH为准，CPU准备不等于GPU实验完成。
禁止热改当前MA冻结源码。所有控制必须记录额外生成、环境评奖和选择成本，
单seed结果不能推出MA“永不崩塌”；还需后续多seed验证。

## 当前请求：H0/W0独立MA，先验证长序列OOM修复

**02:05UTC正式训练已核验，当前仍在继续：** root为
`runs/alfworld_joint_h0w0_8h_v1`，训练冻结源码`b481fb0`，
快照`runs/source-snapshots/alfworld-joint-h0w0-v2`。
node3 `coevolve-joint-h0w0-v2-editor`、node4 `coevolve-joint-h0w0-v2-solver`已各完成
6个native步骤：5轮critic-only后1次Actor更新，K2共12次critic更新，
各消费48、H+W96；不是synthetic显存测试的更新。
首个Actor梯度H2.50648/W0.53345，有限非零；每role48条中40条真实lag1。
`FIRST_ACTOR_AUDIT.json`、`INITIAL_POLICY_CHECK.json`及`TRAINING_LAUNCH.json`
保存精确证据。初始概率偏差上限H0.000810/W0.000222nats/token，小于0.01工程阈值。
这证明真实更新与版本协议，不证明性能已提升；训练后200题尚未完成。

持久评测：node3 `coevolve-joint-h0w0-v2-matched`（round64/1024条）、
node4 `coevolve-joint-h0w0-v2-best`；node4原`coevolve-joint-h0w0-v1-guard`
控制07:49:31UTC训练截止及08:49:31UTC报告截止。
初始基线是W0空harness74%、H0新h0+W0为85%，不是旧W1/H1。
两节点filler保留。下一步监督有效checkpoint与完整matched/best结果；
在完成真实结果核验前不能把本轮任务标为全部完成。

以下为本轮准备历史：

2026-10-03 00:49UTC用户要求从原始H0/W0训练，不继承H1/W1。
新目录`runs/alfworld_joint_h0w0_8h_v1`，初始评测源码冻结`415fd64`，
快照`runs/source-snapshots/alfworld-joint-h0w0-v1`。node4持久服务
`coevolve-joint-h0w0-v1-initial`和`coevolve-joint-h0w0-v1-guard`已启动。
**此时只是初始评测，不是训练启动。** node3独立验证完整32768-token原生PPO
反向/optimizer显存；原先8GiB缓存释放smoke test不能放行长训练。
显存收据预定`runs/alfworld_joint_memory_20261003/MEMORY_COMPLETE.json`。

**初始评测已完成：** W0空harness148/200=74%，H0新h0+W0为170/200=85%；
strict197分别146/197和168/197。8候选7合法，slot01快测40/48，
选定冻结后测200题。`INITIAL_AUDIT.json`核验原始模型、完整原生报告、
compact hashes、题/seed/token合同、训练预算0和node4释放。
这才是新独立MA的组合起点，不能用旧W1+h1=90.5%代替。

`INITIAL_MODEL_IDENTITIES.json`核验原始模型权重：
H0=`7d16d5ad5734af16b1fd0619caeb04270a024df86ca6e6c6e9bdb60e45f39af7`，
W0=`9f52da7a3e4166e3af6b6ba8c0722a8abb396a7ac876e1cb4f83d2453c7bc2fb`。
H/W Actor、critic、reference都从原始对应模型开始；数据和critic历史证据
均来自原W0的train0-239，不再用W79历史。H0本身仍是官方已RL的Engineer。

预算从0计：batch8配对episode/轮，H+W16条；5轮critic-only预热，K2，
Actor LR5e-7、critic LR2.5e-6、固定reference KL0.005、真实lag1。
每4轮保存，round12/52/64分别192/832/1024条；正式matched改为round64，
与历史H1+W1+H2的1024条比较。128轮为最高预算，不保证截止前完成。
预定训练截止07:49:31UTC，报告截止08:49:31UTC；两个前置检查也占用此窗口。
先测H0生成并选定的新h0与W0的200题（以及W0空harness），再开始联合训练。
不能把历史官方对口Engineer或旧h1的分数冒充本轮H0/W0起点。

后续训练入口还增加显式失败后的已提交checkpoint发布：保留原FAILED并以
`recovered_committed_checkpoints_after_training_failure`供独立评测，绝不把训练
异常算成reward0或改写为训练完成。无完整checkpoint则明确RECOVERY_FAILED。
实际训练源码需在显存验证结束后另行冻结，以TRAINING_LAUNCH收据为准，
不要热改正在运行的初始评测快照。

01:44UTC显存验证完成：W0完整32768/33792-token、4DP、两次Actor/critic更新通过；
H0的K2验证为1轮critic-only预热、2次Actor和6次critic更新，**不是5轮预热**。
两个角色都有真实长序列反传证据，H0/W0文件未被测试改写。
进一步复现暖Adam状态下reference OOM，修复为joint learner在PPO更新后也释放
unused cache（reference-only释放不够）。执行源hash必须匹配显存收据才放行。
具体peak/配置/source/释放证据在`runs/alfworld_joint_memory_20261003/`两份报告。
集成38测试+3subtests通过；随后冻结训练源码并启动两角色，不能把上述
synthetic optimizer steps当本轮研究训练。

## 当前结论：六小时窗口已报告，MA目标未达成

**2026-10-02 23:07:27 UTC：** GRPO同192条训练/评测已完成；
MA从失败前的唯一共同持久检查点恢复评测，也已完成并核验200题。
MA组合为**180/200=90%**，未超过原W1+h2的**184/200=92%**。
第12轮未达到，同预算比较仍不可用；不能把本轮称为完整成功训练。
本窗口训练/评测均已退出，两个节点研究GPU释放、filler保持开启；
不以filler利用率冒充训练进度，不自动覆盖W1+h2。

| 系统 | all200 | strict197 |
|---|---:|---:|
| W0 + 新GRPO h1 | 167/200 = 83.50% | 165/197 = 83.76% |
| W1 + 新GRPO h1 | 176/200 = 88.00% | 174/197 = 88.32% |
| MA第4轮Solver，空harness | 172/200 = 86.00% | 170/197 = 86.29% |
| MA第4轮Solver + 新harness | 180/200 = 90.00% | 177/197 = 89.85% |
| 原W1+h2，继续保留 | 184/200 = 92.00% | 181/197 = 91.88% |

MA八提案六合法；01/02/03快测并列45/48，按预定规则选01。
同一个冻结补丁随后做200题，每题sample0一次；没有按200题重选。
MA比W1+h2低2pp，配对95%区间[-5.5,1.5]pp，5题改善、9题退步。
新harness比同一MA Solver空harness高4pp，区间[-0.5,8.5]pp；
并未证明稳定增益，也不证明MA算法长期必然较差。
单seed开发评测、已有选择暴露及与历史后端实现差异都保留。

最终入口：`runs/alfworld_joint_async_6h_v5_recovered/FINAL_AUDIT.json`、
`best_COMPLETE.json`及两份compact report。模型是global_step3/round4，
每角色2次Actor更新，不是未保存的第6轮模型，也不是旧h2替代。
报告hash、原生完整报告一致性、200唯一题/单样本、seed/token预算、
训练题排除、strict197、选中模型路径/补丁及两节点最终GPU释放已核验。
首次释放快照中PID2016329未被识别；最终核验时已退出，未手工杀进程，
旧快照保留，最终两节点只剩已核验filler。
`DEADLINE_REPORT.json`的`incomplete`指matched缺失，不能改写成整个协议完成。

预算：所选点新增64；有效run提交96；前期失败尝试另提交72，
已知MA总提交168、生成动作下界260。加共同前缀832后已知提交1000；
未提交工作、补丁生成/选择及评测另计，不是完整研究成本恰好1000。
评测源为`f3563e6`；后续RPC修复`5d3c0b9`和reference缓存修复`837a67e`
均未热改本次训练/评测。后者通过41测试+3subtests和真实CUDA释放8GiB验证，
**尚未通过完整24075-token PPO重跑验证**。

本次六小时结论已在期限前给出；性能超越/同预算目标未达成。
后续若继续，需先完整验证长序列训练恢复，再从明确检查点按完整预算执行，
不能自动把失败目录重启或把旧200题用于反复择优。

### 以下为17:09–22:46 UTC历史执行记录，不代表当前仍在运行

**22:46 UTC最新接手入口：** 正确性修复后的训练v5在22:33:19发生Solver Actor
CUDA OOM，不能继续称训练中。H/W各有6条完整指标行（各4次Actor、12次critic更新），
但共同持久检查点只有global_step3（第4轮，各2次Actor更新）。
该检查点的真实权重张量及ValueNorm8次更新已经验证。
初始HF/vLLM action平均logprob差最大H0.001488、W0.007527nats/token，
通过显式0.01工程阈值；前6轮每role48条轨迹中40条实际有一轮滞后，
配对ID/reward/slot一致。失败前固定reference KL约H0.000335/W0.000028，
value MSE约0.170/0.172；健康loss没有预报显存耗尽，不是性能崩溃结论。

**当前只做评测：** `runs/alfworld_joint_async_6h_v5_recovered`，
node4 `coevolve-joint-recovered-v1-best`及guard；
node3 matched入口已明确写`unavailable`，不伪造round12。
旧v5的FAILED和原生OOM日志完整保留；恢复结果标
`recovered_committed_checkpoints_after_training_failure`。
评测沿用冻结`f3563e6`/`runs/source-snapshots/alfworld-joint-async-6h-v7`，
当前main后续修复没有热改运行源。23:41:31报告期限不变。
新尝试实际消费96条，加前期72条已提交消费共168条；仅所选点为64条。
生成至少260个policy action（包含未获reward的H动作），另有未完成W工作，
以及后续补丁选择/评测成本；不能把模型路径成本当整个搜索成本。

OOM发生时reference已分配8.80GiB却保留40.40GiB缓存，actor反传需11.25GiB、
仅余6.04GiB。main已加reference-only缓存释放，41项测试及3个subtests通过；
node3真实CUDA检查释放8GiB且保留全部活跃输出。此证据只验证缓存释放，
**不是已重跑通过完整24075-token PPO**，不改写此次OOM失败。

**21:42 UTC恢复：** 首次native MA生成了16个Engineer动作、12个Solver终止outcome，
但尚无已提交optimizer指标行。核查发现ValueNorm按DP分片独立更新，已停止该尝试，
不保留其模型；两节点GPU释放及filler已确认。真实双进程回归复现了不一致，
现改为先收集整个DP组的有效目标，再在各rank更新相同的官方统计。
旧尝试目录`runs/alfworld_joint_async_6h_v1`保留。v2因快照的空gitlink目录内
出现嵌套链接，在模型加载前provenance失败，0采样；修复精确链接并在两个UID下验证后，
v3完成Editor两轮critic预热、Solver两轮预热和一次Actor更新。
但概率审计确认默认raw logprob不包含Engineer采样温度，v3也已停止弃用：
32个H动作、32个W终止outcome，H消费16、W消费24；未提交工作不能假定为零。
v4显式`processed_logprobs`后，初始policy审计仍发现packed microbatch的第二条
轨迹偏差；安装的HF Qwen3.5循环层没有packed边界重置。已停止v4（各角色2轮
critic预热，24个H动作/24个W终止outcome），采用原生allocator每microbatch一条，
有效batch及梯度累积不变。最新恢复目录为`runs/alfworld_joint_async_6h_v5`，
起点、温度、数据与截止时间不变。以上失败采样不是免费成本，
不能将新的round12称为包含全部失败成本的严格同采样实验。
最新启动绑定以恢复目录的`LAUNCH.json`为准，截止时间不变。

2026-10-02 17:09 UTC的新请求尚未完成，不能因旧测评完成而结束。
先从原SAO的同一个H0续训Harness-R1 GRPO，精确匹配24提示*8=192条；
固定W0，6个32条rollout、24个optimizer minibatch step，保存96/192条点。
新增`alfworld_matched_grpo`复用SearchQA现有FSDP trainer与官方hash-pinned
Relax objective，显式适配原生Qwen3.5 VLM checkpoint和既有ALFWorld reward服务。
属于后端适配，不是原作者完整Relax运行栈。GRPO训练和200题对照已完成，
联合MA-SAO正在初始化；整个请求仍未完成。

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
20:57已核验最终192条生成/消费、6个commit、24次optimizer更新，
各rank梯度有限且非零，最终actor identity
`8f2c62db0003175e275b7a957543478ee667fc7008322f4492af40e7a6e6f0d9`。

新[joint_campaign](../whale_local/sao_bridge/joint_campaign.py)接入等待GRPO比较、
联合异步训练、健康止损、共同checkpoint恢复、双节点匹配/后期200题报告及持久deadline guard。
用户17:41:31要求6小时结论：**23:41:31 UTC报告、22:41:31训练截止**。
CPU回归含原生PPO配置解析、同温度固定reference、joint元数据批处理、
健康条件、已提交checkpoint过滤及跨UID仅共享checkpoint的权限边界。
MA仍未GPU验证，不得称为完成。运行目录`runs/alfworld_joint_async_6h_v1`，
实际源码`43c3de0`，快照`runs/source-snapshots/alfworld-joint-async-6h-v4`。
node3持久服务`coevolve-joint-async6h-v4-editor`、`coevolve-joint-async6h-v4-matched`；
node4对应solver、best、guard。**21:31两个native训练进程开始初始化**，
分别PID294695与1939420；下一步须核验同轮配对轨迹、真实版本滞后、
双方proximal交换及warmup后非零Actor梯度，不能把初始化或filler当作训练成果。
真实部署绑定见该目录`LAUNCH_v4.json`；协议继承的旧source_commit不是当前执行源码。
之前v1/v2/v3 MA等待队列都在0样本时停止换版，没有隐藏的MA更新。
18:48:13，GRPO v4首个32条真实rollout成功，reward均值0.009375，
范围[-0.6,0.2]；随后py-spy确认进入native autograd backward，四个训练GPU有负载。
这仍不是最终192条完成收据；等待`progress.json`中的已提交更新，勿把预取条数当消费。

19:55:39已提交第3/6轮：**96条消费、128条已生成、12次optimizer更新**，
前3轮梯度均有限且非零。GRPO最终训练实测速率预计20:50–21:05 UTC，
原19:41对照完成目标偏乐观，主要成本是完整FP32优化器状态保存/校验；
MA的22:41训练截止与23:41报告截止不变。
在MA仍未创建optimizer目录、0条消费时，修正两项监督边界：
长自回归轨迹的高重要性截断率只预警，不单独当作崩溃；
明确STOP之后的worker退出记CANCELLED，不让正常停训污染checkpoint恢复。
缺失固定reference、非有限值仍硬停，连续3轮KL>0.10或value MSE>4仍软停；
STOP之前的真实首个FAILED不会被覆盖。实际队列版本以最新LAUNCH收据为准。

**21:30:47 UTC同预算GRPO对照完成：**

| 系统 | 新GRPO h1的200题 | strict197 | 此前SAO h1的200题 |
|---|---:|---:|---:|
| W0 | 167/200 = 83.50% | 165/197 = 83.76% | 74.00% |
| W1 | 176/200 = 88.00% | 174/197 = 88.32% | 90.50% |

最终192点预先固定；8提案7合法，slot00/07快测同为46/48，按预定规则选00。
补丁digest `fd04c0f9147d0e03364ffab2c5cf2f4d982e2d528fdd5717da7c6bc587f94dde`，
两节点部署完全相同，W0是迁移测试。完整报告的hash、200唯一题/单样本、
seed、token上限及GPU释放收据已核验。
相对SAO h1，W0 +9.5pp，题级95%区间[5,14]；W1 -2.5pp，[-6.5,1.5]。
相对W1+h2的92%，新GRPO组合低4pp，[-8.5,0]，未超过当前系统。
新空harness重测为W0 75%、W1 88%，与旧73.5%/87%有随机运行差异，必须分别标识。
24题选点95.83%未延续到200题；selection24包含在200中，不能称独立测试。
此阶段输出`runs/alfworld_matched_grpo_200_v3/COMPLETE.json`，不要重跑已完成训练。

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


### Validity-credit execution queued after native GPU gate

Prepared both32-round arms with identical inherited H0/MA64 models, task order,
seeds, selection48 and optimizer settings. Only Editor invalid-patch reward mode
differs. Both use bounded submission. Frozen driver/learner snapshot:
runs/source-snapshots/alfworld-validity-credit-study-v1. Main service
alfworld-validity-credit-study-v1 is active, waiting for the two-round GPU smoke.
It requires exact update counts,16 unique receipts per role, paired outcomes,
correct role/team rewards and clean release before launching control training,
its fixed round32 evaluation, then treatment training/evaluation. Driver/audit
regressions4 passed. PRIMARY_RESULTS will remain explicitly diagnostics-pending;
it is not a full research completion flag. DIAGNOSTIC_PROTOCOL.json freezes
fixed-h0 and contemporaneous W1+h2 comparisons, max1600 environment episodes,
before primary scores exist. No new training beyond two512-role-trajectory arms.
Study helpers read committed artifacts and preserve all failed attempts; snapshots
must not be edited. Continue monitoring predecessor branch ablation, GPU smoke,
then study units. The overall>92% goal remains active and unachieved.


### Prespecified credit diagnostics now persistently queued

Implemented frozen diagnostics for the two credit arms: control/treatment Solver
with no patch, fixed h0, own selection48-chosen patch, plus contemporary W1
with no patch/h2. Same200 questions and inference contract, maximum1600 episodes.
Original primary results remain separate from remeasurement; all200/selection48/
report152 paired contrasts are reported. The analysis regression demonstrates
that fixed-h0 weight gains need not imply a selected-system gain and rejects
mismatched rollout identities. Five diagnostic/driver tests passed.
Node3 unit alfworld-validity-credit-diagnostics-v1 is verified active, waiting for
both primary endpoints. Source: runs/source-snapshots/alfworld-validity-credit-diagnostics-v1.
No new score yet; live branch-ablation inference continues normally as of16:41UTC.


### Mechanism localization from completed one-branch reports

MECHANISM_TRACE_AUDIT.json:15 rescued tasks,14 with the targeted blocker;
13 previously reached task step limit,1 completed unsuccessfully,1 invalid action.
Families: two-object3, simple4, cool5, heat1, clean2. This is post-hoc trace
interpretation, separate from the prespecified primary causal contrast.

A misleading surface trace was investigated: task324 requests `take pan 1 from
stoveburner 4` while holding pot1; that request is inadmissible. Calling the
unchanged upstream process_action on the actual preceding admissible list maps
it to `put pot 1 in/on stoveburner 4`. The learned guard examines final_action
after stock normalization, then rejects wrong-object placement. Thus raw tool
requests alone do not identify what the guard saw. NORMALIZATION_MECHANISM_PROBE.json
pins code/report hashes and records this CPU replay; no environment reward changed.
This supports a future recovery-preserving semantic patch contract, but that
method is not implemented or claimed effective. First finish the already frozen
validity-credit comparison; syntactic/runtime validity alone cannot detect this
kind of harmful but executable hook. No new GPU experiment was added from this probe.


### Prospective success verification, before any>92% candidate

Frozen runs/alfworld_ma_goal_validation_v1/protocol.json. Historical point target
requires at least185/200 at the predeclared primary endpoint. Bind the chosen
weights/harness before verification; never switch using diagnostic outcomes.
Verify with inference seeds20261005/20261006 against unchanged W1+h2 under the
same runtime/token/temperature contract. Publish both seed results and a question-
clustered paired interval; pooled candidate accuracy>92% and positive paired lower
bound are required for supported inference superiority. Repeat training with a
new seed before calling the gain robust. All200/report152 remain development data;
unseen-task claims require a separately audited unopened cohort. Costs and failed
attempts remain disclosed. No new compute was launched by this validation protocol.
It has not been triggered: current best new learned primary score remains88%.


### CPU recovery-contract feasibility probe while control trains

runs/alfworld_recovery_contract_probe_v1 pins source/patch hashes and12 constructed
states:3 target types crossed with simple/heat/cool/clean task families, carrying
a distractor with an admissible put-away action away from the destination.
Native compile_hook/run_hook executes initialization, observation update, hint,
and before-action hook in sequence. h112 blocks12/12; h0,h64,h2,the guarded
round40 selected patch,and branch-disabled h112 block0/12. No rewrite/force effects.
Zero environment episodes and zero training trajectories. This is a post-hoc CPU
counterexample suite, not a test of accuracy or proof of semantic correctness;
the native runner suppresses runtime errors, which may also yield no effect.

Interpretation: a small behavioral check can detect the known h112 defect, but
the current88% selected patch already passes. Therefore adding only this check
cannot explain/solve the remaining gap to92%. It is a candidate regression guard,
not an asserted successful learning method. Frozen credit-control training is
unchanged; do not add semantic filtering to either existing arm mid-run.


### Fixed-h0 compatibility interaction in the completed guarded matrix

H0_COMPATIBILITY_INTERACTION.json under alfworld_ma_improvement_study_v2 pins
four existing reports. At initial weights h0-minus-empty=11pp; after continuation
it is3pp. The paired question difference-in-differences is-8pp,95%[-15,-1]pp.
This is a post-hoc exploratory contrast, not a new primary endpoint or an isolated
causal mechanism. It strengthens the question of retaining compatibility with a
strong fixed harness during Solver adaptation. It does not prove that adding
anchor rollouts will help. The currently running validity-credit arms remain
unchanged. As of17:49UTC control H/W both committed9 steps, healthy, and resumed
after the round8 checkpoint save; no new evaluation result exists.

补充：新消融改变terminal_actor.py调用，因此joint_campaign的内存验证源哈希
门禁要求重跑全上下文native memory check，不可沿用旧MEMORY_COMPLETE。
详见新实验IMPLEMENTATION_STATUS.json；原生验证费用需在运行前补充记账。
