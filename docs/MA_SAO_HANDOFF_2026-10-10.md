# MA-SAO 本地交接与存储说明

日期：2026-10-10。**自动重启已暂停，禁止使用H200训练。本文不是训练启动授权。**

> 本文保留10月10日历史交接，不代表最新without_peer进度。10月12日已经恢复后再次
> 暂停：最后76轮metrics，完整native72；代码、切分核验、当前平台认证及未来上传计划
> 以[最新复现交接](REPRODUCE_MA_SAO.md)为准。下文native24及“未执行GPU续训”
> 是当时事实，不应覆盖最新记录。

## 1. 当前保留的研究结果

| 实验 | 已完成工作 | 选定Harness + Solver | 空Harness Solver |
| --- | --- | ---: | ---: |
| 共同初始化full，96轮诊断 | 训练及200题评测完成 | 180/200，90.0% | 178/200，89.0% |
| 共同初始化full，144轮主终点 | 训练及200题评测完成 | **186/200，93.0%** | 179/200，89.5% |
| without_peer | **24轮配对原生状态已保存**，没有端点评测 | 未完成 | 未完成 |
| without_context / without_value_norm / always_joint | 尚未启动 | 未完成 | 未完成 |
| 历史MA参考 | 开发集选出的历史峰值，不是本次因果对照 | 194/200，97.0% | 185/200，92.5% |

新93%相对历史97%净差为-4个百分点：新系统改善1题、退化9题，
问题级配对bootstrap95%区间[-7,-1]个百分点。不是独立训练种子不确定性。
历史使用已训练MA64 Critics/ValueNorm、48轮边界Adam重启、24次而非16次提案机会，
并经过开发集峰值选择；不得把差异归因于某个组件。

完整训练逻辑预算为1152个共享环境结果、2304条生成角色轨迹；
144轮selection恢复复用82条已完成结果，432个逻辑episode共记录494次物理开始，
其中62次来自首次未完成尝试；不重跑完成题、不补抽候选。
without_peer提交192个共享结果，实际落盘199个结果、398条角色回执；
额外7个结果为未提交预取，未知在途token成本不记为0。

## 2. 暂停和恢复边界

- 用户08:16 UTC要求停止自动重启、不再使用H200。持久取消门控已写入原运行目录。
- 初检时旧等待控制、信号跟踪和只读观察器均已退出；两节点没有GPU进程或分布式启动器，
  两个对应原生服务均inactive/not-found，attempt-0002不存在。
- 不删除原结果、旧FAILED、原协议、原源码快照或24轮断点。
- 原10:10:44 UTC截止时间不会因交接而延长。任何以后使用其他硬件的训练都需要新的明确授权和执行方案。

最后一次without_peer停止的根因是共享文件系统跨节点flock不互斥，
导致两个角色竞争写入同一COMMITTED临时文件。修复快照仅让Solver发布共享元数据，
双方仍各自保存和训练Actor/Critic。修复经过66项恢复/时序测试、
28项驱动/时序测试及真实双节点CPU合成断点验证；两套测试有重叠。
**新的GPU24轮恢复、36轮保存尚未执行**，不能声称这一步已完成。

## 3. 本地交接目录和产物

目录：[ma_sao_handoff_20261010T081628Z](../runs/ma_sao_handoff_20261010T081628Z/)。
产物是否最终完整，以目录中的`VERIFIED.json`和`DELIVERY.json`为准；本说明不代替校验回执。

| 产物 | 内容 |
| --- | --- |
| `source.tar.gz`及manifest | 当前工作树的完整受版本控制/未忽略代码；包括此前未提交的新文件、6个代码仓库、5个科学/执行快照、运行目录内的恢复脚本 |
| `evidence.tar.gz`及manifest | 当前运行可读取的JSON/JSONL、指标、原始结果、完整episode、配置和日志；历史97%原始评测、固定数据和关键依赖 |
| `supplemental-logs.tar.gz`及manifest | 补充保存先前5个权限不可读Engineer监督目录中的10个日志文件；原权限和原排除记录保持不变 |
| `assets.tar.gz`及manifest | 原始ALFWorld数据和实际FLA/Triton运行覆盖层，避免只留代码却遗漏环境数据 |
| `CHECKPOINTS.json` | 19组关键checkpoint的用途、原路径、实际文件大小、保留位置及校验范围 |
| `checkpoint-pins/` | 大多数文件采用同文件系统硬链接；所有者权限不允许链接的HF文件明确记录后只读复制，不修改原文件权限 |
| `GIT_PROVENANCE.json` | 主仓库及依赖的原revision和dirty状态；原工作树、原索引不被批量提交或清理 |
| `repository/`、`code-results.bundle` | 单独的本地交接Git仓库及可离线传递的Git包；含代码、主要原始结果/训练输入和大数据包索引 |
| `*.exclusions.json` | 未跟随的符号链接、未纳入证据包的二进制训练状态、缓存，以及权限不可读的可选子目录；不把这些遗漏隐藏为完整备份 |

源代码与证据包按原项目相对路径保存，不覆盖原文件。大checkpoint不进入Git提交。
环境模板仅保留空凭据字段，私钥/token特征检查不等于公开发布许可。
原始训练记录和模型可能包含敏感项目内容；没有直接外传。

### 明确的存储限制

**本地归档、Git提交、硬链接均不等于远端备份。**
硬链接与原checkpoint共享相同inode；它能在原目录项被正常删除时保留文件，
但不能抵御底层存储损坏或原文件被原地改写。不得编辑任何pin文件或原权重。
因权限复制的文件也是同一3fs上的本地副本，不是异地容灾。

小元数据重新计算SHA-256；已存在的权重全文件摘要注明来源，
未重新读取的大型原生张量只记录大小、inode及已有恢复验证证据。
不得把采样优化器指纹或尺寸检查冒充新计算的全部大权重SHA-256。
归档压缩包及其中每个成员会逐字节重读验证。

## 4. checkpoint应该如何选择

| 保存目标 | 必需内容 | 约占用 |
| --- | --- | ---: |
| 复现93%系统推理 | full144 Solver Actor、tokenizer/config、选定Harness及原评测协议 | **17.55 GiB**加少量文本 |
| 保留full144两个Actor | Engineer Actor和Solver Actor | **35.09 GiB** |
| 恢复without_peer第24轮训练 | 该轮双角色完整原生状态、原始初始化/参考模型、相同数据与代码 | 原生状态本身 **384.36 GiB** |
| 保留full144完整训练状态 | 双角色Actor/Critic、Adam、RNG、ValueNorm、游标及元数据 | **384.36 GiB** |
| 上述两份完整原生状态 | full144和without_peer24 | **768.71 GiB** |
| 19组关键原生/HF/初始化/历史参考 | 包含96/144诊断、共同初始化、历史97%Actor/Critic | **1051.84 GiB，约1.129 TB十进制** |

只有Actor的HF权重不能恢复Adam/ValueNorm/RNG/训练游标。
同样，原生DCP不是可以直接交给通用HF推理加载器的Actor导出。
每个目录的精确字节数以`CHECKPOINTS.json`为准。

## 5. GitHub与Hugging Face分工

用户希望GitHub保存完整代码、实验结果和数据，HF保存大权重及中间恢复状态。
合理的组织方式是：经批准的私有代码仓库保存代码、协议、核心原始评测和数据索引，
经批准的大对象存储保存重型训练状态、完整轨迹数据包及原始校验记录。
**本次只完成本地准备，不执行GitHub push或HF上传，不声称已经远端同步。**
用户指定的HF项目名为`coevolve`；尚未核验命名空间或创建远端仓库，访问约定见第7节。

根据2026-10-10查看的[HF官方存储限制](https://huggingface.co/docs/hub/storage-limits)：
免费用户/组织私有空间为100GB；PRO为1TB加超额计费；组织配额取决于订阅席位。
19组全部保存约1.129TB，超过1TB，且还要考虑既有占用和版本历史。
即使只保留一份384GiB原生状态，也超过免费私有空间。
不要为了额度将含敏感状态的仓库改为公开。平台额度、模型/数据许可和组织外传要求需先确认。

## 6. 在其他机器恢复前必须处理的事项

1. 保留源报告中的绝对路径和SHA证据；不要在冻结原文件上全局替换路径。
   新机器应采用相同挂载布局，或另立有记录的执行层路径映射/恢复方案。
2. 代码快照中的`runs`、`assets`、`third_party`等链接不会被归档器递归跟随。
   链接目标见排除清单；数据和依赖在独立包中提供，重新建立链接后再验证。
3. 原生训练依赖AReaL本地改动、固定FLA/Triton版本、原始模型/参考模型、原有ALFWorld实例。
   这些源文件/数据已纳入对应包；不复制凭据或假设本机虚拟环境会自动存在。
4. 原始脚本含固定节点名、UID、systemd、GPU拓扑和原截止时间检查。
   交接包不是“在任意显卡直接运行”的承诺，尤其不能据此在H200自动启动。
5. full93%和历史97%不是组件因果对照；without_peer24轮不是最终消融结果。
   所有未完成实验、额外成本和负面结果都必须继续保留。

## 7. Hugging Face `coevolve`访问约定（未部署）

本节是本地交接后的文档补充，**没有执行认证、创建远端仓库、上传或远程下载**。
已经校验的原始归档、Git离线包和交付清单保持不变；不能声称本节已经包含在那个冻结Git包中。
以下说明用于今后经批准建立存储后的读取，不构成训练或上传授权。

### 仓库标识和凭据

- 项目名为`coevolve`，完整标识必须是`<个人或组织命名空间>/coevolve`，不能只写`coevolve`。
  不根据GitHub用户名猜测HF命名空间，也不使用聊天中提供的令牌查询账户。
- 以下示例约定使用**私有dataset仓库**，用于混合保存checkpoint、恢复状态和实验归档，
  不是可直接调用`from_pretrained("<namespace>/coevolve")`的单模型仓库。
  实际仓库建立后需记录完整ID、类型和不可变HF提交SHA；目前这些远端信息均未核验。
- 下载凭据由执行者自己的凭据管理提供，使用仅有必要仓库读取权限的令牌。
  命令沿用HF客户端已有的本地认证，不传`--token`，不在项目配置、脚本、日志或Git中写入令牌。
  出现在聊天或日志中的令牌应立即在[HF令牌设置](https://huggingface.co/settings/tokens)撤销，
  不继续使用，也不要把替换令牌发到对话中。
- 固定`HF_REVISION`为远端完整提交SHA，不依赖会变化的`main`。
  本地代码提交`b9ea7aa`不是HF存储提交，不能拿它代替HF revision。

### 约定的远端相对路径

| 路径 | 用途 |
| --- | --- |
| `CHECKPOINTS.json`、`DELIVERY.json`、`VERIFIED.json`、`SHA256SUMS` | 冻结交付范围和明确的校验能力边界 |
| `checkpoint-pins/full-144-solver-actor/` | 93%系统的Solver HF Actor、配置和tokenizer，约17.55 GiB |
| `checkpoint-pins/without_peer-native-0024/` | 第24轮双角色完整原生恢复状态，约384.36 GiB |
| `checkpoint-pins/full-native-0144/` | 已完成full144的完整原生恢复状态 |
| `checkpoint-pins/initial-*/` | 原初始化和参考模型；具体用途以checkpoint清单为准 |
| `evidence.tar.gz`、`assets.tar.gz`、`supplemental-logs.tar.gz`及各自manifest | 实验记录、固定资源和补充日志；按需要读取 |

路径以本地[checkpoint清单](../runs/ma_sao_handoff_20261010T081628Z/CHECKPOINTS.json)的
`local_pin`字段为准。保留原相对布局，不改写冻结记录中的`source`绝对路径。
源码、核心结果和数据索引属于代码交接；大模型文件不应塞进普通Git提交。

### 按需下载，不默认拉取全部1.129 TB

命令要求读取端已安装提供`hf`命令的`huggingface_hub`。
执行者需先配置下列非秘密环境变量；本说明不提供令牌赋值示例：

| 变量 | 必须提供的值 |
| --- | --- |
| `HF_REPO_ID` | 经核验的`<namespace>/coevolve` |
| `HF_REVISION` | 经核验的不可变HF提交SHA |
| `COEVOLVE_ARTIFACTS` | 新机器上的绝对下载目录；不要指向原始checkpoint或本地硬链接保留目录 |

先获取清单；未设置变量时直接报错，不回退到公共仓库或默认版本：

```bash
: "${HF_REPO_ID:?Set HF_REPO_ID to namespace/coevolve}"
: "${HF_REVISION:?Set HF_REVISION to an immutable HF commit SHA}"
: "${COEVOLVE_ARTIFACTS:?Set COEVOLVE_ARTIFACTS to a new absolute local directory}"

hf download "$HF_REPO_ID" CHECKPOINTS.json DELIVERY.json VERIFIED.json SHA256SUMS \
  --repo-type dataset --revision "$HF_REVISION" --local-dir "$COEVOLVE_ARTIFACTS"
```

仅需要93%推理模型时，在同一已配置终端中读取这一组，不下载完整训练状态：

```bash
hf download "$HF_REPO_ID" \
  --repo-type dataset --revision "$HF_REVISION" --local-dir "$COEVOLVE_ARTIFACTS" \
  --include "checkpoint-pins/full-144-solver-actor/*"
```

模型目录是`$COEVOLVE_ARTIFACTS/checkpoint-pins/full-144-solver-actor`。
仍需使用冻结结果中已选定的Harness及原评测协议，不能只换权重后声称复现93%。
取得文件不会启动推理或训练。

需要准备第24轮续训材料时，分别获取原生状态和原始模型：

```bash
hf download "$HF_REPO_ID" \
  --repo-type dataset --revision "$HF_REVISION" --local-dir "$COEVOLVE_ARTIFACTS" \
  --include "checkpoint-pins/without_peer-native-0024/*"

hf download "$HF_REPO_ID" \
  --repo-type dataset --revision "$HF_REVISION" --local-dir "$COEVOLVE_ARTIFACTS" \
  --include "checkpoint-pins/initial-*/*"
```

这还需要第6节所列的固定输入、源码、依赖和路径映射；不是一条直接启动训练的命令。
原生DCP必须走项目的双角色恢复路径，不能用`from_pretrained`加载。
只想分析实验记录时，可以按名称下载独立归档及其manifest，无需拉取模型。

### 下载后的核验和失败处理

2026-10-10冻结版`CHECKPOINTS.json`的SHA-256为
`a650b4126e120fc557ee17173dfa6474d6e582793fd1f02031d98f915a5104dc`。
针对上述冻结交付，可先验证这一独立保存的校验值：

```bash
(
  cd -- "${COEVOLVE_ARTIFACTS:?Set COEVOLVE_ARTIFACTS first}" || exit 1
  printf '%s  %s\n' \
    'a650b4126e120fc557ee17173dfa6474d6e582793fd1f02031d98f915a5104dc' \
    'CHECKPOINTS.json' | sha256sum --check -
)
```

- 对照清单检查所选目录中的全部文件及字节数，再核对条目提供的全文件SHA-256。
  只有尺寸或原inode的条目不能证明跨机器传输的字节完整性；新机器的inode不同是正常现象。
  大型native张量尚未全部重新计算全文件哈希，不能把这些检查写成“全部权重SHA验证通过”。
  原`verify_pins`核验的是源机器硬链接关系，不应直接当作跨机器下载校验器。
- 压缩归档应同时核验压缩文件SHA-256和manifest中的成员记录。
  `SHA256SUMS`覆盖37项顶层交付文件，不覆盖全部大型checkpoint张量；
  只有清单对应文件全部到齐时才对整份清单运行`sha256sum --check SHA256SUMS`。
  按需下载时只核验对应目标，并明确其他文件尚未下载/核验。
- 后续远端传输若补充大张量摘要或发布新版本，应另立带版本的传输回执，
  不修改这份冻结清单，也不要无记录地替换本节的可信校验值。
- 401/403、仓库或revision不存在、缺文件、尺寸或哈希不匹配均应停止并明确报告。
  不把私有仓库改公开，不静默改用`main`，不以另一组模型替代目标checkpoint。

CLI参数依据[HF官方下载说明](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download)；
本次只补充本地访问文档，远端存在性、权限和下载行为仍未验证。

入口：[项目状态](./PROJECT_STATE.md)、[算法说明](./MA_SAO_ALGORITHM.md)、
[完整MA结果](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/full/RESULT.md)、
[24轮保存审计](../runs/alfworld_ma_mechanisms_single_seed_v2_rolling12_r2/without_peer/commit_recovery01/SAVED24_AUDIT.json)。
