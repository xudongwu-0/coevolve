# MA-SAO：当前进展复现与模型交接

更新日期：2026-10-12。**训练保持暂停。本文的校验、下载和上传计划均不授予训练启动权限。**

## 1. 新 agent 的最短接手路径

取得此代码版本后，在仓库根目录执行：

```bash
python3 -B scripts/reproduce_ma_progress.py
```

不需要安装训练依赖、提供API key、下载模型或使用GPU。这条命令逐文件核验
[证据包](../reproducibility/ma_sao_20261012/MANIFEST.json)，然后：

- 从固定任务目录和源样本重新构造joint/joint/anchor训练输入，逐项对比原始ledger。
- 核对训练、选择、主评测和strict197的真实重叠关系。
- 从原始逐题报告重算Full96、Full144的成功数，而不抄汇总表。
- 核对Harness选择先于该端点主评测、Solver身份及评测预算。
- 区分without_peer的76轮metrics与72轮完整配对恢复元数据。
- 按各自原manifest验证七个冻结源码快照；不把当前代码倒称为历史执行代码。

预期核心输出：

| 对象 | 选定Harness + Solver | 无Harness、同一Solver |
|---|---:|---:|
| Full96 | 180/200，90.0% | 178/200，89.0% |
| Full144，预先固定主终点 | **186/200，93.0%** | **179/200，89.5%** |

without_peer应显示两角色各76个连续metrics step、恢复元数据72轮；
`training_allowed`与`gpu_or_api_used`均为false。
**这是既有证据的确定性复算，不是新跑一次模型后得到93%。**

需要查看原运行相对目录时，可复制到一个尚不存在的新目录：

```bash
python3 -B scripts/reproduce_ma_progress.py --materialize /tmp/coevolve-evidence-copy
```

禁止覆盖现有目录，不修改原记录中的绝对路径，不复制模型、不执行任何启动器。
在任意checkout目录均可直接做第一条离线核验；原训练脚本则不是任意路径即插即用。

## 2. 算法和冻结配置

完整机制、公式及设计局限见[算法说明](MA_SAO_ALGORITHM.md)；
历史分支与协议见[MA_SAO](MA_SAO.md)。本次共同初始化控制：

- H0 Engineer、普通SAO W1 Solver，以及原始pre-MA Critics；不是历史97%那组已训练MA Critics。
- 两套Actor、两套Critic，同一环境episode的团队终局奖励。
- bounded队友重要性权重、Critic协作上下文、各角色ValueNorm。
- joint / joint / Solver-only调度；每角色前3个活跃轮只训练Critic。
- seed1000043、batch8、144轮、32768上下文、4096动作token上限。
- Actor LR5e-7、Critic LR2.5e-6、每活跃轮2次Critic更新、固定参考KL0.02。
- 每12轮成对原生保存，只有Solver发布共享COMMITTED/LATEST。

Full训练共1152个共享环境episode、2304条生成角色轨迹；不是两个角色
都各做144次Actor更新。原规范、两角色spec、metrics及训练审计均在证据包中。

Full原训练使用两节点各8卡；最近without_peer续训使用node3八卡、node4 GPU1–7：
每角色保持4个learner rank，rollout副本分别4和3。该调整保持原生优化器布局，
但不保证采样逐token等价。端点评测维持node3八副本；不能将其缩成七副本后
声称与原评分完全相同。

## 3. 数据切分：必须准确表达的限制

原始500任务目录及顺序被保存，不按目录排序重新编号：

| 用途 | 实际范围/数量 | 注意事项 |
|---|---|---|
| 本次MA训练 | 官方目录中的ID0–239，共240个任务 | 不与下面200题发生任务ID重叠 |
| 基础源输入 | 768行，固定种子1000042产生任务顺序 | prompt及历史W0证据逐字保留 |
| joint/joint/anchor输入 | 1152行，144轮×8 | anchor重复的是输入，不是重放已生成轨迹 |
| Harness选择 | 显式manifest中的48题 | **这48题全部包含在下面main200中** |
| main200 | 固定200题，每题一次 | 是反复使用的开发集，不是独立未触碰测试集 |
| strict197 | main200排除291、414、467 | 这三题的实例没训练过，但goal group与旧训练任务有关 |
| 其余152题 | main200减selection48 | 只做事后描述性统计，不能包装成新预注册测试终点 |

虽然训练与评测的任务ID分离，**选择与主评测并不分离**。93.0%因此不能称为
独立泛化测试成绩。strict197也不消除选择题重叠；官方预训练数据暴露无法独立核验。
同样，不把另一个2296-game训练目录或更早selection24协议误套到本次MA输入。

证据包同时保存任务目录、原selection/core200协议、完整训练输入、训练packet、
选择候选及原始逐题报告。任何新的未污染测试需要另行设计和授权，不能改写旧200题。

## 4. 代码、依赖和环境如何恢复

顶层Git包含完整项目代码和测试；上游仓库继续使用固定commit的Git submodule，
不重新声称拥有其版权：

```bash
git submodule update --init --recursive
```

版本见[UPSTREAMS.json](../reproducibility/ma_sao_20261012/UPSTREAMS.json)。
AReaL不是这些submodule之一的普通已提交子目录；它位于SAO的忽略vendor目录，
有实际本地修改。为避免只提交指针而遗漏运行代码，本次额外保存：

- [完整AReaL工作树补丁](../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch)。
- [基准revision及1180个文件的指纹](../reproducibility/ma_sao_20261012/upstream/AReaL.json)。
- 原AReaL许可证，补丁同时包含tracked改动与untracked新增源文件。

在**尚不存在**的vendor目录中准备原基准版本，然后应用完整补丁：

```bash
git clone https://github.com/inclusionAI/AReaL.git third_party/sao/vendor/AReaL
git -C third_party/sao/vendor/AReaL checkout --detach 3cf0dfbd2b0fbeabd6977184980e189d1567747a
git -C third_party/sao/vendor/AReaL apply --check \
  ../../../../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch
git -C third_party/sao/vendor/AReaL apply \
  ../../../../reproducibility/ma_sao_20261012/upstream/AReaL-working-tree.patch
```

不要先运行自动应用SAO补丁的bootstrap再重复应用这份**完整**补丁。
若目录已存在，先核对状态，不覆盖或reset。准备工具已经从干净基准重建并逐文件
校验了该工作树；没有用修改中的仓库来冒充干净安装测试。

实际9B原生环境以[ENVIRONMENT.json](../reproducibility/ma_sao_20261012/ENVIRONMENT.json)
和包内ALFWorld worker环境清单为准，不直接照抄历史2B的worker requirements：

- Python3.12；实际训练环境记录torch2.10.0+cu129、vLLM0.19.1、transformers5.7.0。
- learner另有隔离覆盖层：FLA/fla-core0.5.2、Triton3.7.1；基础venv的Triton版本不是
  实际learner覆盖层版本。见[覆盖层依赖](../requirements/training-acceleration.txt)。
- ALFWorld AgentBench的原数据、环境和本地内核覆盖层另在大型assets归档中。
- 原协议含绝对路径、固定host/UID/GID、systemd及GPU映射。完整GPU重放前须恢复
  对应依赖和模型，并单独准备合法的新执行配置，**不能改写已冻结的科学记录**。

本轮没有重装训练环境、没有新开推理服务、没有重训。远端模型/归档未提供前，
新机器可以复算证据，但不能声称已经具备完整GPU续训材料。

## 5. Hugging Face checkpoint上传计划：仅计划，尚未执行

凭据只读验证到个人命名空间`wxd02`；细粒度token在该命名空间声明了repo.write等权限。
`wxd02/coevolve`的dataset查询返回404，**没有创建仓库**。W&B身份只读验证为
`2373025856w`，没有创建run或上传artifact。

建议使用私有dataset仓库`wxd02/coevolve`，因为需要同时容纳HF Actor、
原生DCP/Adam/RNG和数据归档，而不是一个可直接from_pretrained的单模型repo。
最终远端repo ID、类型、不可变revision和每组上传回执必须另行登记；
Git里的[模型清单](../reproducibility/ma_sao_20261012/MODEL_ARTIFACTS.json)
现在是本地库存，不是远端可下载承诺。

| 优先级 | 对象 | 约体积 | 作用 |
|---|---|---:|---|
| 1 | Full144 Solver Actor | **17.55GiB** | 配合Git里的选定Harness及固定协议重跑93%系统 |
| 2 | Full144 Engineer Actor | **17.55GiB** | 保存主算法最终两方策略；前两项合计35.09GiB |
| 3 | without_peer完整native72 | **384.36GiB** | 保住未完成消融的最近可靠恢复位置；不是76轮 |
| 4 | Full144完整native | **384.36GiB** | 保留主方法全部训练状态，供未来另行授权续训 |
| 必需依赖 | H0、W1、两个原始Critic、历史packet W0 | **83.94GiB** | 原初始化/参考模型及完整恢复依赖，按manifest去重 |
| 可选后置 | Full96、额外HF Critic、历史97%参考 | 依清单 | 诊断/历史对照，不优先重复上传所有大对象 |

两份native约768.71GiB；上述主要Actor、native和五组依赖合计约887.75GiB，
还未计数据/轨迹归档。完整19组为1051.84GiB，约1.129TB（十进制）。
**先核对私有存储可用额度，不因额度不足将研究数据改为公开。**

上传执行须另获许可，并按以下顺序验收：

1. 先冻结清单，确认源文件不变；小元数据和大张量分别记录校验范围。
2. 第一组17.55GiB Actor做端到端试传，记录真实吞吐后再估计总时间。
3. 使用可续传的大文件上传；训练checkpoint不得原地压缩、改写或删掉。
4. 为每个待传大文件生成/验证完整SHA-256。已有尺寸检查、Adam采样指纹、
   或历史全文件摘要不能冒充本次传输全字节校验。
5. 每组完成后固定HF commit SHA，用只读下载抽检/完整校验确认大小和摘要，
   验证tokenizer/config及模型角色没有串组，再登记远端回执。
6. 一组上传完成并不代表另一组完成；暂停/失败保留可恢复上传状态，
   本地原checkpoint在远端验证完毕前不得删除。
7. 同时上传固定输入、任务数据和必要轨迹归档；保留原目录映射及取消记录。
   Git中的元数据不能替代384GiB原生状态。

建议相对布局：

```text
releases/20261012/
  checkpoints/full-144-solver-actor/
  checkpoints/full-144-editor-actor/
  checkpoints/without_peer-native-0072/
  checkpoints/full-native-0144/
  checkpoints/initial-*/
  archives/
  MODEL_ARTIFACTS.json
  TRANSFER_VERIFIED.json
```

只有远端回执存在后才使用实际`HF_REPO_ID`和不可变`HF_REVISION`按需下载；
不要使用虚构revision或默认main。W&B后续更适合保存指标、版本索引和验证回执，
本次不重复上传两份384GiB状态，也不创建任何run。

## 6. 凭据与Git发布

用户指定的三个凭据文件在仓库之外的`/mnt/3fs/xudong/api_keys/`。
目录0700、文件0600；GitHub文件实际是SSH私钥，不是GitHub REST token。
API key/私钥不进入Git、命令参数、URL、配置快照或日志；不读取旧聊天中的令牌。
仓库额外忽略api_keys及相应文件名，打包器也拒绝凭据路径和已知令牌/私钥内容。

GitHub仅发布代码、固定数据切分/输入、原始核心结果和元数据；
大模型放HF的动作尚未授权执行。私有Git仓库本身也不能替代模型备份。
