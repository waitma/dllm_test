# 工作池监督中的诊断发现

本文件保留解释性发现；逐任务状态与指标由[自动监督记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md)持续更新。以下均未触发活跃训练源码修改。

## 全 mask 的可识别性上限

对旧首轮实际 6 条 toy、实际 collator 和 `measure()` 掩码协议做 CPU 验证：全部 7 个 denoise 输入特征的分组为 `[[0,1,2,3],[4],[5]]`，前四条 TCR 在全 mask 后输入相同但标签不同。逐组逐位置计算，AA-only 最多正确 `46/82=56.10%`；所有目标 token 联合最多正确 `51/93=54.84%`。两个上限分别优化相应指标，不声称能同时达到。

因此“AA/end 同时达到 100%”不能作为该 toy 的通过条件。提高到 100 步仍有必要，但应加入固定部分 mask 与条件链诊断，并将合成记忆能力与真实数据泛化分开。

证据：[full_mask_identifiability.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/full_mask_identifiability.json)。

## R08 的辅助 loss 不能单独排除表征塌缩

首轮 R08 step5 / step15 的 VICReg 辅助 loss 约为 `0.9688 / 0.9492`，数值和梯度有限；当前日志未单独给出 invariance、variance、covariance 或 embedding 标准差。仅凭辅助 loss 总值不能排除表征塌缩，也不能证明表征优于对照。若下一阶段保留 R08，应记录这些分量及 embedding 标准差。

首轮 R08 最终全 mask loss `2.04524`、AA acc `19.51%`、end acc `0`；运行成功属于工程结论。

## R12–R19 使用新的比较协议

[run_immune_architecture.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_architecture.py) 在初始化后统一重设训练噪声 RNG。R13–R19 应与 R12 对照，不应直接根据它们与旧 R00 的轨迹差异宣称架构增益。

新输出包括 `mask25/mask50/mask75/mask100` 的 selected/AA/end 计数与 loss/acc；`mask_mean` 是每个指标对非空 mask level 的等权平均。旧顶层指标仍是 full mask 语义。监督表按 job_id、目标 step 和 mask level 保留独立记录。

## 新实现的显存开销：待本轮结束后验证的等价优化

实测生产尺寸、A100 80GB、同一 profile 协议：

| 方法 | micro-bs | 峰值 GiB |
| --- | --- | --- |
| 旧 R00 | 2 | 3.6362 |
| 新 R12 | 2 | 5.8003 |
| 旧 R00 | 8 | 11.1955 |
| 新 R12 | 8 | 20.4573 |

代码阅读发现，[ArchitectureModel.denoise](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_next/model.py) 每层用 `base_bias.masked_fill(...)` 创建新的 `[B,heads,L,L]` attention bias，再将其作为 activation checkpoint 的输入。R12 的 global mask 每层相同，但此实现仍创建多个 bias 张量。重复保留大张量是显存增长的合理解释，尚未通过内存归因实验确证。

待当前被冻结的实验结束后，可以验证：在单次 forward 内按 global / chain / local mask 类型构造并复用 bias，禁止原地修改。先做 logits、梯度及 checkpoint 等价验证，再重新 profile；不沿用本轮显存证据。该项属于实现优化，不应与架构增益混为一谈。

当前不热改代码；R12–R19 按自己的实测 profile 选择 batch，global batch 仍固定 256。

实测 profile 现已完成：R12–R19 均选择 micro16 / accumulation16，micro16 峰值约 40.02–40.12 GiB；micro32 是预期 OOM 上界。源文件 35 项指纹均匹配。上述工程优化仍未应用。

## R12–R19 的 20/100 步中间观测

以下为历史中间点，最终结果以自动监督记录为准；所有数值来自六条合成记录，不能作为生物学泛化结论。

| 方法 | mask_mean loss | mask_mean AA acc | mask_mean end acc |
| --- | --- | --- | --- |
| R12 对照 | 1.8529 | 28.28% | 0% |
| R13 可学习 head gate | 1.8477 | 26.46% | 0% |
| R14 固定 half attention | 1.8518 | 25.85% | 0% |
| R15 可学习 LayerScale | 1.8071 | 27.76% | 0% |
| R16 固定 residual scale | 1.8051 | 27.76% | 0% |
| R17 chain late fusion | 1.6853 | 36.21% | 0% |
| R18 chain local/global | 1.4810 | 44.92% | 38.35% |
| R19 value residual | 1.7574 | 43.09% | 0% |

R18 此时出现较明显的联合改善信号，需看 100 步轨迹是否持续。其 full-mask AA/end 分别为 23.17% / 90.91%，不能只看终止符准确率。各 mask level 的 end 计数仅 3/4/8/11，单个预测变化即可造成较大比例跳动。

R15 与固定缩放对照 R16 的结果近乎一致，此时不能将改善归因于“可学习”的 LayerScale。R13 的 mean loss 略低于 R12，但 mean AA 准确率较低，也不能只据总 loss 宣布有效。

## R18 收益可能受短链与固定容量的组合影响

当前 toy 的实际受体链只有约 5–10 aa，而每条 full-scope target 分配 256 个位置。R18 的局部层采用 chain-local radius32，因而限制了大量 MASK 尾部作为 key 的参与；这可能是早期优化改善的部分来源。该机制使用位置与链，不依赖隐藏的参考长度，但短 toy 可能放大其收益。

这属于待验证解释，不是已确认的因果归因。后续应增加多种长度的合成组，或在清理后的真实长度分布上验证，且保持 R12 为同协议对照。当前监督不自行新增实验。

## 50/100 步：排序改变与对照的作用

| 方法 | mask_mean loss | mask_mean AA acc | mask_mean end acc |
| --- | --- | --- | --- |
| R12 对照 | 0.36302 | 86.41% | 86.08% |
| R13 可学习 head gate | 0.40583 | 82.34% | 95.45% |
| R14 固定 half attention | 0.32141 | 86.41% | 95.45% |
| R15 可学习 LayerScale | 0.30717 | 87.38% | 93.18% |
| R16 固定 residual scale | 0.28648 | 87.75% | 93.18% |
| R17 chain late fusion | 0.42266 | 85.94% | 74.05% |
| R18 chain local/global | 0.33151 | 86.47% | 100% |
| R19 value residual | 0.35632 | 85.89% | 95.45% |

R18 的早期优势已缩小，50 步最低均值 loss 是固定缩放 R16；20 步排序不能当成稳定的方法排序。R13 此时未优于固定 gate R14，R15 也未优于固定缩放 R16。

full-mask 的联合 argmax 也存在取舍：R18 是 AA `40/82` 加 end `11/11`；R15/R16 是 AA `43/82` 加 end `8/11`，三者均为 `51/93`，已经达到该固定 toy 的联合 argmax 上限。它们的 AA/end 差异不直接代表生物学优劣；CE 仍可能因概率校准不同而变化。

## 四档可识别性：独立 CPU 复算证据

使用实际 collator、实际 `fixed_validation_mask` 与所有七个 denoise 特征做逐输入分组；未加载模型或 checkpoint。计算假设为 eval-mode 协议中的输入确定性 argmax：相同输入、相同位置的预测必须相同。

| mask level | 输入组 | AA 可正确数 / 总数 | 全目标可正确数 / 总数 |
| --- | --- | --- | --- |
| 25% | 6 个互异输入 | 18/18 | 21/21 |
| 50% | 6 个互异输入 | 48/48 | 52/52 |
| 75% | 6 个互异输入 | 69/69 | 77/77 |
| 100% | `[[0,1,2,3],[4],[5]]` | 46/82 | 51/93 |

因此该协议下 `mask_mean AA` 的独立上限是 `(1+1+1+46/82)/4=89.0244%`；四档全目标 token accuracy 的独立上限为 `(1+1+1+51/93)/4=88.7097%`。这些是固定六条 toy 的信息可识别性结果，不能推广为真实数据上限；AA、end、联合目标分别优化，不能假定各自上限可同时达到。

可复算程序：[audit_immune_toy_identifiability.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/audit_immune_toy_identifiability.py)。证据含实际分组、各档计数、源文件 SHA 与审计脚本 SHA：[four_mask_identifiability.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/four_mask_identifiability.json)。

```bash
USE_TF=0 python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/audit_immune_toy_identifiability.py --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor
```

先激活 `pllm`；之后如需复算历史 release，可另外提供 `--source-root` 指向其冻结源码目录。

## Refined 缓存修复的首次实际显存证据

2026-09-30 21:04 UTC，R20–R27 新 profile 仍在扫描；以下只比较已经 passed 的同一 micro-batch 档位，不提前给推荐 batch 或模型质量结论。

| 同档比较 | 旧 R12 ArchitectureModel | 新 R20 RefinedModel | 变化 |
| --- | --- | --- | --- |
| micro2 峰值 GiB | 5.8003 | 3.6360 | 约降低37.3% |
| micro8 峰值 GiB | 20.4573 | 11.1955 | 约降低45.3% |

新 R20 同档与原 R00 全局注意力显存相符，支持先前关于逐层重复 bias 引起内存膨胀的解释。R25/R26 分别保留两种拓扑 bias，micro2为约4.019/4.371 GiB。80项CPU对照验证了缓存重用的数值等价性，GPU显存扫档独立验证了资源改善；两者均不能替代广长度诊断与真实下游测评。

旧证据：[R12 profile](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r12_architecture_control/summary.json)；新已过档证据：[R20 micro2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r20_refined_control/bs2/measurement.json)、[R20 micro8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles/r20_refined_control/bs8/measurement.json)。后续完整扫描和25步诊断以实时文档为准。
