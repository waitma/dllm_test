# 指标审计：高 CDR AAR / AUROC 的含义与评测缺口

完成时间：2026-11-04T07:00:19+00:00。只读检查现有数据与最终权重，没有训练、改写输入或修改冻结评分器。

用户的质疑成立：三项数字衡量不同任务；AUROC 还受到明确的缺链捷径影响，现有隔离只按受体组合做精确去重。之前把高 AUROC 作为强表征证据的说法需要降级。此审计没有发现 CDR 评分掺入框架/padding 或关系标签直接进入编码器的错误。

## 1. 评分口径与 CDR 分区域复算

| 指标 | A0 两 seed 均值 | 任务条件 |
|---|---:|---|
| 整链 AAR | 7.92% | 117 条链/4 来源；整条定容区域全遮蔽，生成 EOS 和长度，按完整参考长度计分 |
| 全 CDR AAR | 79.71% | 64 条链：OAS 抗体 H/L 各16，OTS TCR β/α 各16；只遮 CDR，框架/另一条链/EOS 保留 |
| CDR1 AAR | 94.95% | 相同64条 CDR 补全任务，按来源等权平均链级 AAR |
| CDR2 AAR | 92.40% | 相同64条 CDR 补全任务，按来源等权平均链级 AAR |
| CDR3 AAR | 65.10% | 相同64条 CDR 补全任务，按来源等权平均链级 AAR |
| 抗体重链 CDR3 | 40.72% | OAS 中16条重链，每个种子固定同一组 |

CDR 是保留框架、已知各 CDR 位置/长度的补全。框架可提供 V/J 相关信息；CDR1/2 很高不等于未知长度整链生成好，也不等于 CDR3 或新结合特异性好。64条任务不是64条抗体：一半来自 OTS TCR。

A0 整链提前 EOS：seed42/43 的来源宏平均预测长度分别为37.10/11.49，参考114.18；参考覆盖分别33.04%/10.17%。A5 同口径为95.39%/94.04%。原评分把未输出的尾部计为不匹配，因此整链 AAR 显著更低；已覆盖位置准确率与长度问题都应分开报告。

即使只比较相同 OAS/OTS 来源，A0 整链 AAR 仍分别只有13.72%/4.38%，CDR 为79.80%/79.62%。因此来源差异不是唯一原因，遮蔽/长度条件是实质差异。

## 2. AUROC 的明确捷径

A0 随机权重加监督线性探针的宏 AUROC 已为 0.8206，训练后为 0.8434（增加0.0228）。随机的是骨干；探针仍在 probe_train 用真实标签训练。不能把随机骨干基线称为随机猜测。

当前探针只有 tcr_papers 与 TRAIT 两个来源，分别独立拟合 StandardScaler + LogisticRegression，再等权取两个 AUROC。train383、valid106；tcr_papers valid33中仅7正例，TRAIT valid73中9正例。不是通用蛋白/抗体表征排行榜。

| 来源 | 未训练 A0 + 探针 | 训练 A0 + 探针 | 仅缺 α 链指示器 | 仅布局/长度特征 + 探针 |
|---|---:|---:|---:|---:|
| tcr_papers | 0.9093 | 0.9533 | 0.9286 | 0.8791 |
| trait | 0.7318 | 0.7335 | 0.5556 | 0.4323 |

tcr_papers train：30正例中12缺 α 链，112负例全部观测到 α 链；valid：7正例中6缺 α 链，26负例全部观测到 α 链。缺链在 receptor_vectors 中产生零向量，输入本身也保留观测缺失模式。这是标签与采样/缺失模式相关的捷径，不是把 labels 字段直接喂给模型。元数据子来源与正负类也不平衡，详见 cpu_audit.json。

只保留两链均有观测的 tcr_papers valid，原 A0 探针 AUROC seed42/43变成0.6538/0.6923；但剩下27条且只有1正例，不能把这个事后子集作为可靠的新泛化估计。TRAIT 对应72条、8正例，A0 为0.7051/0.7012。

## 3. 全部固定10%训练记录的精确重叠检查

实际读取 653,288 条已选训练记录。评测与预训练完整受体组合的精确重叠为0，probe_train/probe_valid组合重叠也为0。但是：

- 117条整链任务中 16 条目标单链在训练中出现过；其搭配链不同，因而未被组合去重排除。
- 64条 CDR 任务中 10 条完整目标单链在训练中出现过；20 条目标 CDR3 在训练中出现过。
- CDR3 重叠按角色：OAS L=13、H=1，OTS α=4、β=2；不能将这些任务统称为未见过序列的泛化验证。
- probe_train/valid 无相同非空 β 链；TRAIT 有1条非空 α 链重复。空 α 链不能当作相同蛋白序列重复计数。
- 此检查仅是精确身份检查，没有证明同源、克隆型、V/J 或研究来源隔离；重复也不足以单独解释全部高分。

## 4. 已验证的计算与复现限制

- 117条整链和64条 CDR 的索引与真实序列对齐；CDR 分母只含被遮蔽的真实残基，框架/EOS/padding 未计分。
- 489条探针记录逐条翻转关系标签后，全部7个编码器输入特征与 pooling residue_mask 完全一致；没有直接关系标签穿透。
- 10个模型端点及初始评测的40个来源 AUROC/AUPRC，均由保存的逐样本概率独立重算一致；所有保存整链 AAR 也由预测序列和完整参考长度重算一致。
- A0 seeds42/43 与 A5 seed42重新推理，各64条 CDR评分与原产物一致；A0 seed43、A5 seed42全181条观察也完全一致。
- A0 seed42有2/117整链解码与原产物不同（原始来源均为ASD，CDR0差异）。该细小复现问题未定位，不将本次A0 seed42整链推理声称为完全精确重放；原始长度统计仍来自原始保存序列。详情 replay_A0_42.json / gpu_audit.json。

## 5. 结论与后续使用规则

保留所有历史原始数字，但名称必须写明“未知长度整链重建”“已知框架/位置的CDR补全”“来源内小样本结合探针”。旧 AUROC 架构排序暂停用于泛化能力结论；A5 对 A0 的长度/整链重建改善只保留为当前冻结任务上的描述性结果。C2 等候选需要在控制缺链模式、研究来源及单链/CDR3/同源隔离后的评测再比较，现有带捷径探针不足以支持稳定表征提升。

后续修正应另建带版本的评测清单与评分报告，按原V8 reader/collator读取，保留原始记录形式及历史权重。应分别报告 CDR1/2/3 与 H/L/β/α，加入随机骨干、缺链指示器和布局基线，并为新泛化结论准备独立样本。此次未启动新训练、没有改变实验筛选/显著性门槛，也不把事后子集当作新确认集。

![审计图](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/audit.png)

[CPU输入/概率检查](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/cpu_audit.json) · [GPU逐区域核对](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/gpu_audit.json) · [训练精确重叠](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/training_overlap.json) · [逐CDR CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/cdr_regions.csv) · [AUROC控制CSV](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/arch70m_results/metric_audit_20261104/probe_controls.csv)

复现代码：/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/metric_audit_20261104.py 与 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/render_metric_audit_20261104.py。原seed42/43绑定源码SHA依然为258dd11e8895c3b6d4fbb9c00e526de67e6bc9ccbef43948970790f5a00d8fed / 28609d7f776013915561eb038396d54181dcee7b1c32d071be0a9da5dc12faad。
