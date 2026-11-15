# SAO WebShop：GRPO-only改造与真实子集结果

## 结论

**SAO Solver的GRPO后端改造和真实子集验证已完成；未证明成功率提高，不能宣称45→66。**

本次不加蒸馏：同预算的reference-step GRPO相对legacy GRPO恢复了一部分退化，但没有超过初始模型。置信区间跨0，预先定义的扩大训练门禁未通过，未启动完整100步或完整评测。

算法和SAO接线边界见 [WEBSHOP_SAO_GRPO.md](./WEBSHOP_SAO_GRPO.md)。这是可选的**SAO Solver阶段GRPO后端**，Engineer/Harness保持冻结；不声称完整Engineer–Solver联合训练已被测试，也不修改旧ALFWorld/MA-SAO或Aether-QAT实验。

## 控制条件

- 源码：`72a554342dac85a806db895c363f5319e72e8663`，immutable snapshot `frozen/grpo-v1`。
- Reference：`aim-uofa/Self-Distillation-AgentRL@35a44f5c5647b1785ad23ddf0ffbd196b695bf86`，Apache-2.0及WebShop自身MIT边界保留。
- 基座：同一local BF16 Qwen3.5-9B、LoRA r16/alpha32、all-linear；不使用先前SD checkpoint。
- 环境：实际WebShop1k，1,000商品、6,910 synthetic goals，goal seed233，Lucene索引/源码/商品/价格/goal指纹固定。
- 训练：8个任务IDs`600,604,608,612,616,620,624,628`，每题8rollout，8updates，共64个真实episode/arm。
- 测试：新的固定IDs256–287，32题，eval seed0，temperature0.4，每题最多15actions，每action最多512generated tokens，total context4096/history3。
- 两组的模型、task IDs、生成参数、optimizer/KL、Harness和预算相同；只改GRPO profile。
- legacy：episode binary reward、population std、episode mean loss。
- reference：动作上`10*terminal_success - 0.1*invalid`，跨动作same-task group/sample std，token-mean loss；无teacher上下文、teacher tokens为0。
- initial模型单独重测同32题，不以历史45/74替代。

以上profile按固定reference的实际代码适配，不代表完整复现论文：仍有1k corpus、LoRA、8updates、history/total-context、KL、entropy和单seed等限制。与参考100updates/128goals/8eval seeds不能直接排名。

## 真实结果

success定义为`terminated && raw_task_score == 1`；score是raw环境分数平均值×100，不能将partial score当作成功率。

| 模型 | 成功 | 成功率 | Mean score /100 | 无效动作 | 生成token |
|---|---:|---:|---:|---:|---:|
| Initial | 8/32 | 25.000% | 47.2693 | 16 | 18,409 |
| 同预算legacy GRPO | 6/32 | 18.750% | 38.3160 | 4 | 7,829 |
| 新SAO Solver reference-step GRPO | 7/32 | 21.875% | 47.1701 | 4 | 6,193 |

### Reference minus legacy GRPO

- 成功率净+3.125pp：3题救回、2题退化；95% paired task-bootstrap CI **[-9.375,+15.625]pp**。
- score净+8.8542/100；95% paired CI **[-4.1667,+21.8750]**。
- Exact McNemar双侧p=1.0。
- 有真实reward variance/gradient信号，但performance gate **FAIL**。

### Reference minus initial

- 成功率 **-3.125pp**。
- score **-0.0992/100**，基本持平但没有正向证明。
- 两个训练版本都降低了无效动作，相对initial的这一变化不能单独归因于新的reference profile。
- 输出变短/评测变快也不能替代任务完成；不将少生成token包装成成功率改善。

历史45%/score74没有对应完整模型/题集/环境manifest。本次同一个基座在新32题上为25%/47.27，说明子集难度不能与旧汇总直接相减；没有证据支持“当前只需多训就一定达到66”。

## 已验证的机制

第一训练group的两组8条rollout、成功6/8、无效动作3次相同；更新前输入/采样条件匹配。之后策略变化导致轨迹不同，不能要求后续环境episode仍相同。

训练完成：

| 项目 | legacy | reference |
|---|---:|---:|
| 真实训练episodes | 64 | 64 |
| optimizer updates | 8 | 8 |
| 训练group中的成功数 | 23 | 21 |
| 无效动作数 | 59 | 68 |
| 零reward-variance groups | 3 | 0 |
| teacher-scored tokens | 0 | 0 |

reference确实让无成功但动作有效性不同的group得到非零学习信号；legacy这类group基本只剩固定reference KL。**非零梯度不等于更好策略**，此次heldout没有可靠收益。

93项CPU回归测试通过，包括reference统计/sample std、invalid处罚但raw score不改、失败token/behavior logprobs保留、同题/同policy identity、token-mean真实optimizer权重、无蒸馏的SAO入口/晋级门禁，以及既有WebShop response-only/native tiny-transformer/LoRA测试。真实环境重复reset/search也通过。

每个GPU任务有current explicit single-GPU lease、PID/uid/start_ticks及完成receipt。实际训练/测试均正常完成，没有OOM或nonfinite指标，使用node25/26，未触及22/24。每个任务退出后查询对应GPU为空，最后进一步核查节点无GPU残留。

## 产物与复算

实验目录：

```text
/mnt/3fs/xudong/coevolve/runs/webshop_sao_grpo_20261115/
  initial_eval32/REPORT.json
  control_train64/adapter/
  control_train64/TRAINING.json
  reference_train64/adapter/
  reference_train64/TRAINING.json
  control_eval32/REPORT.json
  reference_eval32/REPORT.json
  PAIRED_RESULT.json
  INITIAL_COMPARISON.json
  per_task.csv
```

Adapter是已验证能加载评测的**推理产物，不是完整optimizer/RNG恢复checkpoint**。不从这些adapter静默假装续训。

已保存 [小型结果/provenance复算包](../reproducibility/webshop_sao_grpo_20261115/)，不包含模型或完整原始生成轨迹；这些保留在上述实验目录。无模型/GPU/API的确定性复算：

```bash
python3 scripts/reproduce_webshop_sao_grpo.py
```

下一步应先明确原45/74的模型/环境/评测身份，以及此次成功/失败的具体商品属性、选择顺序与停止行为，再用更有代表性的训练开发子集/多seed验证学习曲线。若单靠动作处罚只学会合法输出而不完成购物，应调整信用分配或探索机制并重新定义未触碰确认集，而不是启动盲目的完整100步。
