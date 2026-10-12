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

本轮没有重装训练环境、没有新开推理服务、没有重训。历史Actor权重不包含Critic、
Adam、RNG等原生训练状态；取得它们也不代表已经具备完整GPU续训材料。

## 5. Hugging Face：只保存历史97%的Actor checkpoint

2026-10-12 15:34北京时间，用户将范围限定为历史97%对应的checkpoint，
并明确**不上传Harness、不下载回验**。此前Full96/144、native72/144和初始化模型
的批量上传计划全部取消。[旧模型清单](../reproducibility/ma_sao_20261012/MODEL_ARTIFACTS.json)
仅保留为当时的本地库存，不能按其中的旧priority继续上传。

已于15:44:40北京时间完成上传，唯一目标为私有dataset
[wxd02/coevolve](https://huggingface.co/datasets/wxd02/coevolve)，不可变revision为
`d0e84414e8ef9e2b6a84585e1365189faf501051`。
[精确文件清单](../reproducibility/historical97_20261012/CHECKPOINT_UPLOAD_MANIFEST.json)
记录本地SHA256，[远端回执](../reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json)
记录相同revision下的文件、大小及服务器对象摘要核对结果。
**没有执行下载回验。** W&B仍只读，没有创建run或上传artifact。

### 精确模型身份与范围

历史路线为`alfworld_ma_convergence_v1r4/block-01/round0096`，
两个Actor目录均为`epoch0epochstep95globalstep95`；这是该路线累计144轮的峰值，
**不是**共同初始化研究Full144的93%模型。

| 内容 | HF相对目录 | 大小 |
|---|---|---:|
| 历史97% Solver Actor及原配置/tokenizer | `best97/solver/` | 18839727095字节，约17.55GiB |
| 同时点Engineer Actor及原配置/tokenizer | `best97/engineer/` | 18839727101字节，约17.55GiB |

共20份checkpoint文件、37679454196字节，约35.09GiB。
**没有Harness、Critic、Adam/native状态、其他checkpoint或评测数据。**
checkpoint目录里的原`joint_schedule.json`和`runtime_supplements.json`是小型配置元数据，
不包含优化器状态，也不是可续训原生快照。HF自动维护的`.gitattributes`不计入模型文件数。

原始报告为选定Harness与Solver合计194/200（97%），同一Solver无Harness为185/200
（92.5%）。97%是复用开发集选出的历史峰值，不是独立泛化测试，不能称为纯权重成绩。
原选定Harness仍在[既有Git证据](../reproducibility/ma_sao_20261012/payload/runs/alfworld_ma_convergence_v1r4/block-01/round0096/patches/95-proposal07.json)
中；按用户要求不复制到HF。

### 发布与访问约定

- 每份待传文件均已由上传SDK计算完整SHA256，并核对原模型身份、大小及已有摘要。
- 上传完成后已核对远端revision、精确文件清单、大小及服务器对象摘要；**没有下载模型回验**。
- 本地原checkpoint保留不动，文件权限/所有权不改写，不因为容量或权限将仓库改为公开。
- 访问需要有该私有仓库权限的HF token，并显式指定`repo_type="dataset"`和实际不可变revision。
- 只需Solver时选择`best97/solver/*`；需要Engineer时选择`best97/engineer/*`。
  不能把这个双角色dataset当成一个根目录可直接`from_pretrained`的单模型repo。
- 不使用虚构revision或默认`main`冒充已锁定的模型版本。

以下仅供未来按需访问，本次没有执行这些下载代码。安装有`huggingface_hub`的环境中，
将`HF_TOKEN_FILE`设置为仓库之外的HF凭据文件路径，再从Git仓库根目录运行：

```python
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

release = json.loads(Path(
    "reproducibility/historical97_20261012/HF_REMOTE_VERIFIED.json"
).read_text())
model_root = snapshot_download(
    repo_id=release["repo_id"],
    repo_type=release["repo_type"],
    revision=release["revision"],
    token=Path(os.environ["HF_TOKEN_FILE"]).read_text().strip(),
    allow_patterns=["best97/solver/*"],
    local_dir="runs/hf_best97_download",
)
print(Path(model_root) / "best97/solver")
```

访问Engineer时将allow_patterns改为`["best97/engineer/*"]`；需要两者时显式列出
两个模式。得到本地Actor目录不代表获准启动推理、评测或训练。

## 6. 凭据与Git发布

用户指定的三个凭据文件在仓库之外的`/mnt/3fs/xudong/api_keys/`。
目录0700、文件0600；GitHub文件实际是SSH私钥，不是GitHub REST token。
API key/私钥不进入Git、命令参数、URL、配置快照或日志；不读取旧聊天中的令牌。
仓库额外忽略api_keys及相应文件名，打包器也拒绝凭据路径和已知令牌/私钥内容。

GitHub仅发布代码、固定数据切分/输入、原始核心结果和元数据；
HF仅授权上述历史97% Actor对，W&B不写入。私有Git仓库本身也不能替代模型备份，
历史上传计划不能覆盖用户最新限定的范围。
