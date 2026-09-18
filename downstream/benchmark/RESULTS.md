# AB / TCR 下游评测结果

更新：2026-09-17。上半部分 AB，下半部分 TCR；每个任务先列 baseline，Ours 在表末，checkpoint 与协议不混用。

**v5 159000 AB＋TCR 最终结果已更新（2026-09-17）**：AB 六项、TCR 十项均完成，使用独立留存 [v5 159000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_159000_full_20260917)，SHA256 `01b3d1446b21f56b3a3658528a405cb34dee7dc889eb1f162e423d64141faf3e`。当前 Ours 主行是 **159000**，49000/92000/140000 历史结果保留。Pairing 固定 **p0 / CFG=0 / iter=16**、reference 长度、gumbel_argmax、500×8；140000 为124步，因此差异不能只归因于权重。CDR 为 argmax / iter=2，Specificity 为五折固定末轮100 epoch。TCR 沿用现有协议（T4仍为32步，不随AB pairing改成16步）。[AB产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917)、[TCR任务索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/all_159000_tcr_submissions.json)、[汇总核对记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/results_document_audit.json)。本轮核对产物身份、数量、折完整性及部分聚合，不重跑前向、IM评分器或完整逐预测重评分；不宣称严格论文复现或预训练去污染。

49000 整矩阵仍保留为已验收对照。**AB pairing 159000 主行为p0（无前三残基、16步），140000 p0为124步**；49000 主表行仍是 ESMC 反馈修复后的 **p3（有前三残基）**。TCR 49000 完整性以 [最终汇总 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.json) 为准；92000/140000/159000 数字来自各任务产物 JSON/聚合日志，不是该 49000 comparison 的重算。

- **AB**：[CDR 补全](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr) · [轻链配对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing) · [Specificity / GDPa1 / m396](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-probes)
- **TCR**：[Binding 四轨](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-binding) · [Clustering A/B](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-clustering) · [Representation broad/deep](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-representation) · [Generation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-generation)

读表约定：`[P]` 论文报告值；`[A]` 官方预测重评分；`[R]` 官方代码/权重本地复跑；`[L]` 本地实现；`[C]` 控制。Ours 均为本地评测。论文值优先，本地结果不冒称论文值；“未获取”不等于 0。审计通过不自动证明预训练无重叠、与论文完全同协议或实验功能有效。

已完成对照快照：[v5 49000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913)。历史非 pairing 快照：[v5 92000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915)（SHA256 `2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c`）。历史矩阵快照：[v5 140000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_140000_llada_20260917)（SHA256 `a482afa425ef4f058d88e38d316aca096341af471b504efe948434d3ebc2dfe2`）。TCR 29k 仅作已存在的 others 历史对照，其他任务不借旧结果补空。 当前快照：[v5 159000](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_159000_full_20260917)。

这里只保留 AB/TCR 范围内的结果与必要限制。旧排队状态、重复表、失效协议、历史故障分析及已冻结任务移入 [整理前全文存档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)；原始预测、日志、权重和审计均未删除。任务定义/操作手册见 [任务目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/README.md)。保留 §0.x 节号以兼容其他文档的引用；展示顺序已改为 AB → TCR。

## 一、AB 结果

<a id="ab-cdr"></a>

### §0.5 AB CDR infilling

当前 Ours 为 v5 159000，固定 argmax、**iter=2**；指标为 AAR %。Kong 为 10 折宏平均，SAb23H2 为 60 条抗体。49000/92000 行仍是 **iter=1**，与 140000/159000 不是同一解码协议，不能直接当步数消融。论文模型的训练方式、预训练重叠和解码设置并不完全一致，不据此宣布“干净 zero-shot 超过 baseline”。协议：[AB CDR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)；论文值：[Ophiuchus 论文 Table 1–5 / Figure 4](/root/oph_paper/oph.pdf)。

#### SAbDab Kong：H1 / H2 / H3

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | 来源 / 设置 |
|---|---|---|---|---|
| AntiBERTy | 76.70 | 71.10 | 42.70 | [P] |
| AbLang2 | 76.30 | 70.60 | 42.70 | [P] |
| Ophiuchus-Ab | 75.50 | 70.18 | 43.55 | [P] |
| RADD | 67.64 | 58.87 | 37.71 | [P] fine-tuned |
| ADesigner | 64.34 | 55.52 | 37.37 | [P] fine-tuned |
| dyMEAN | 63.52 | 55.41 | 37.19 | [P] fine-tuned |
| MEAN | 58.29 | 47.15 | 36.38 | [P] fine-tuned |
| AbBFN | 70.30 | 64.90 | 31.50 | [P] |
| Ophiuchus-Ab 官方 checkpoint | 75.69 | 69.96 | 43.70 | [R] 历史固定 iter=1 |
| Ours v5 49000 | 75.9802 | 69.5445 | 42.9223 | 历史对照，固定 iter=1 |
| Ours v5 92000 | 76.2245 | 69.7462 | 43.2944 | 历史对照，固定 iter=1 |
| Ours v5 140000 | 76.9329 | 71.1631 | 42.6172 | 历史，固定 argmax、iter=2 |
| Ours v5 159000 | 76.8701 | 71.5733 | 42.8128 | 当前，固定 argmax、iter=2 |

140000 折间 SD（ddof=0）：H1 4.2600、H2 4.0469、H3 2.6097。92000 折间 SD：H1 4.2175、H2 4.0271、H3 2.2569。来源：[159000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/cdr-kong)、[140000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/cdr-kong)、[92000 三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/cdr-kong)；49000：[三份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/cdr-kong)。表值为 `Average AAR all folds`。没有保存逐序列预测，因此是日志/参数核对，不称独立重算全部 AAR。YAML 写明 `--sampling-strategy argmax --max-iter 2`。

#### SAb23H2：六个 CDR

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | L1 ↑ | L2 ↑ | L3 ↑ |
|---|---|---|---|---|---|---|
| Ophiuchus-Ab [P] | 74.8 | 68.6 | 36.8 | 81.1 | 80.1 | 73.7 |
| IgGM (IgFold) [P] | 74.0 | 64.4 | 36.0 | 75.0 | 74.3 | 63.5 |
| IgGM (AF3) [P] | 73.9 | 63.9 | 33.0 | 73.7 | 73.5 | 60.2 |
| dyMEAN [P] | 74.2 | 62.7 | 29.4 | 63.3 | 63.4 | 57.0 |
| DiffAb (AF3) [P] | 63.7 | 39.4 | 22.6 | 60.8 | 59.9 | 42.4 |
| Ophiuchus-Ab [R]，历史 iter=1 | 73.81 | 68.32 | 35.40 | 80.88 | 80.68 | 72.36 |
| Ours v5 49000，iter=1 | 75.0000 | 67.2718 | 34.4950 | 80.2964 | 78.9304 | 70.9363 |
| Ours v5 92000，iter=1 | 74.2857 | 67.5516 | 34.5570 | 80.7383 | 79.6447 | 70.6512 |
| Ours v5 140000，iter=2 | 75.2381 | 68.9266 | 34.7017 | 80.7319 | 79.8828 | 72.3347 |
| Ours v5 159000，iter=2 | 74.7619 | 68.8710 | 35.0377 | 81.0982 | 79.6447 | 73.2870 |

来源：[159000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/cdr-sab23)、[140000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/cdr-sab23)、[92000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/cdr-sab23)、[49000 六份聚合日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/cdr-sab23)。140000 主行是 iter=2，不是 49000/92000 的 iter=1，也不是下方 92000 多步对照里逐 CDR 挑最优。未用每个 CDR 在测试集上挑选的 best-iteration 数值作 Ours 主行；历史多步 sweep 见 [归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)。

<details>
<summary>SAb23H2：ESMC 反馈修复后的 iteration 对照（2026-09-16，本地已完成）</summary>

同一 v5 92000 留存权重，完整 60 条抗体，每次掩一个 CDR；argmax、batch=1、CFG=0、temperature=1、seed=42。仅改变 `max_iter`，共 24 组、1,440 条预测；短 CDR 可提前填完，实际 forward 次数不一定等于上限。指标为逐序列 AAR 的均值，单位 %；下列论文行仅作参考，论文解码步数未明确披露，不等于相同采样协议。

| 方法 / checkpoint | H1 ↑ | H2 ↑ | H3 ↑ | L1 ↑ | L2 ↑ | L3 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Ophiuchus-Ab [P] | 74.8 | 68.6 | 36.8 | 81.1 | 80.1 | 73.7 |
| Ours v5 92000，iter=1 | 74.2857 | 67.5516 | 34.5570 | 80.7383 | 79.6447 | 70.6512 |
| Ours v5 92000，iter=2 | 74.2857 | 66.9405 | 34.8933 | 80.6666 | 79.6447 | 70.9038 |
| Ours v5 92000，iter=4 | 74.5238 | 66.9405 | 32.8751 | 80.4677 | 79.6447 | 70.6089 |
| Ours v5 92000，iter=8 | 74.5238 | 67.2183 | 32.3089 | 80.3289 | 79.8828 | 70.3961 |

结论：本组没有随步数增加而稳定提升。H3 的 iter=2 比 iter=1 高 0.3364 个百分点，但 iter=8 低 2.2481 个百分点；不能用逐 CDR 最优值拼成新主行，原固定 iter=1 主结果保持不变。本轮没有运行 Kong 多步对照，也没有同权重“修复前／后”多步配对实验，不能量化修复本身带来的性能增益。

验收：1,440 条预测均完整且为标准氨基酸，逐条匹配数和 24 格 AAR 独立重算通过；六模式 iter=1 与旧日志完全一致。每模式首条样本、每种步数检查真实 ESMC 输入（共 24 组轨迹），已接受生成残基可见、未接受位置 MASK；另六次隐藏参考替换检查输出不变。这不是对全部 1,440 条进行逐层抓取，也不等于预训练去污染验收。

来源：[完整 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/summary.json)、[逐条预测目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916)、[ESMC 输入审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/feedback_audit.json)、[权重／数据／源码身份](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916/manifest.json)；baseline 仍来自 [论文 Table 1](/root/oph_paper/oph.pdf)，协议见 [AB CDR §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)。

</details>

<a id="ab-pairing"></a>

### §0.6 AB Light-chain pairing

500 条重链，每条生成8候选；IM为全部4,000候选均值。159000 固定 **p0 / CFG=0 / iter=16**，140000 为 **p0 / CFG=0 / iter=124**；均为reference长度、gumbel_argmax、seed42。没有159000/140000的p3或CFG网格结果。49000主表行仍为修复后p3。前缀、步数、权重不同需明确区分。

| 方法 / checkpoint | IM ↑ | 有效率 ↑ | 轻链类型匹配 ↑ | Diversity |
|---|---|---|---|---|
| Ophiuchus-Ab，论文 [P]，有前三残基 | 0.695 | 1.000 | 0.997 | 0.335 |
| Ophiuchus-Ab，论文 [P]，无前缀 | 0.701 | 1.000 | 0.556 | 0.676 |
| Ours v5 49000 p3（ESMC 反馈修复后） | 0.442264 | 1.000000 | 0.996500 | 0.203651 |
| Ours v5 140000 p0 / CFG=0 | 0.504681 | 1.000000 | 0.783750 | 0.389908 |
| Ours v5 159000 p0 / CFG=0 / iter=16 | 0.543686 | 1.000000 | 0.792000 | 0.433252 |

**2026-09-16 修复后重评已完成并验收（49000 p3）：** ESMC 每轮读取模型已接受的生成残基，pending 保持 MASK，不读取真实后缀；固定同一 49000 权重、数据、reference 长度、iter124、gumbel_argmax、CFG0、seed42。有前缀的 IM 从修复前 0.476541 降至 0.442264，不能把输入错误修复等同于性能必然提升；有效率与轻链类型匹配改善，多样性下降。两组 p0/p3 均有完整 4,000 条生成与评分。验收边界见 [修正协议 §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

140000 p0 使用同一修复后 encoder 状态（`committed_generated_residues_pending_mask_no_reference_targets`）。与 49000 p3 的前缀条件不同，也不同于论文原生生成长度。同权重 49000 无前缀 124 步诊断 IM=0.442087 见下方折叠表，不能与 140000 混称为步数扫描。来源：[Ophiuchus-Ab 论文 Table 3](/root/oph_paper/oph.pdf)、[159000 p0 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/pair-p0-cfg0/holdout500_metrics.json)、[140000 p0 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/pair-p0-cfg0/holdout500_metrics.json)、[修复后 49000 p3 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_encoder_sync_20260916/pair-p3-cfg0/holdout500_metrics.json)、[修复前历史 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/pair-p3-cfg0/holdout500_metrics.json)；详细协议见 [AB pairing](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

<details>
<summary>无前缀 pairing：49000 步数诊断完整结果（2026-09-17 验收）</summary>

六档均为平台Success，24,000条新增候选全部验收；每档同一500条重链×8候选，固定v5 49000、ESMC反馈修复后、无真实轻链初始残基、reference长度、CFG0、gumbel_argmax、temperature1、seed42。124步为此前已验收的同协议独立参照。IM及各率均以全部候选为分母，不删除无效候选、不选best-of-8。论文行来自Table3无前三残基部分，长度为原生生成，与Ours已知长度不同；以下是步数诊断，不替换上方p3主表，也不与140000矩阵混用。

| 方法 / max_iter | IM ↑ | 有效率 ↑ | κ/λ类型匹配 ↑ | Diversity | worker耗时（分钟） |
|---|---:|---:|---:|---:|---:|
| Ophiuchus-Ab [P]，无前缀 | 0.701 | 1.000 | 0.556 | 0.676 | 未报告 |
| Ours v5 49000，8 | 0.485744 | 0.997250 | 0.805250 | 0.553055 | 30.06 |
| Ours v5 49000，16 | 0.496521 | 0.999500 | 0.780250 | 0.497163 | 39.67 |
| Ours v5 49000，32 | 0.479923 | 0.999750 | 0.758750 | 0.480879 | 67.03 |
| Ours v5 49000，64 | 0.462169 | 1.000000 | 0.775750 | 0.458670 | 93.52 |
| Ours v5 49000，96 | 0.447975 | 1.000000 | 0.759250 | 0.457356 | 130.17 |
| Ours v5 49000，124（原参照） | 0.442087 | 1.000000 | 0.766750 | 0.450097 | 未记录同口径worker耗时 |
| Ours v5 49000，128 | 0.438446 | 1.000000 | 0.762750 | 0.452593 | 164.96 |

本轮最高IM出现在16步：比124步高0.054434（相对+12.31%），但仍低于论文参考0.204479。16→32→64→96→128的IM逐档下降，不能假设增加步数必然改善；有效率已接近100%，并未同步解释IM下降。worker耗时含门禁、生成、评分及验收，不含平台排队，不能直接称纯推理吞吐。固定seed不意味着不同步数的逐候选随机数一致；尚无多seed置信区间，也没有在独立验证集选定16步，不能将本次最高分称为普遍最优或直接升级主结果。

2026-09-17只读复核：六组实际反馈gate通过、每组4,000条预测与全部annotation齐全，样本身份／顺序和124参照一致；重新核对指标文件／CSV／manifest哈希、匹配率及有效率分母、CSV审计均通过，已固定源码／数据／参考产物没有漂移。IM与Diversity来自已完成的相同评分器，本次没有重跑评分模型。来源：[完整汇总（含Better、V/J及家族匹配）](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916/summary.json)、[实验身份与逐条产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_pairing_iters_p0_20260916)、[论文Table3](/root/oph_paper/oph.pdf)；协议见[AB pairing §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。

</details>

<a id="ab-probes"></a>

### §0.6a AB 附加表征任务

Ours 编码原生完整 H/L，冻结 ESMC+LLaDA，使用 post-LLaDA 有效残基 global mean；不同任务按既定头/回归流程训练。协议与输入/选模差异见 [AB Native Probes](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。

#### Specificity：HD-Flu-CoV 分类

4,398 条样本、五折；Ours 固定取末轮，不按留出折曲线挑最好 epoch。下面是各折均值，不是跨 epoch 平均。

| 方法 / checkpoint | 头训练预算 | Accuracy ↑ | F1 ↑ | MCC ↑ |
|---|---|---|---|---|
| Ophiuchus-Ab [P] | 论文，最终选头配置未披露 | 0.6796 | 0.6790 | 0.5203 |
| Ophiuchus-Ab [R] | 历史独立 100 轮 | 0.677804 | 0.676818 | 0.517054 |
| Ours v5 49000 | 100 | 0.618000 | 0.616288 | 0.428139 |
| Ours v5 49000 | 200 | 0.627325 | 0.625782 | 0.441693 |
| Ours v5 92000 | 200 | 0.643468 | 0.641730 | 0.466119 |
| Ours v5 140000 | 100 | 0.644152 | 0.642641 | 0.466953 |
| Ours v5 159000 | 100 | 0.641652 | 0.640057 | 0.463362 |

140000/159000 均是标准 100-epoch native、固定末轮，不是 92000 的 200-epoch 预算，也不能当作 200 轮曲线上的中间点。Ours F1 为 macro；论文 Table 5 未标明 averaging。相同头结构不等于相同输入宽度/参数量；历史 baseline 未按本轮全部保存头门禁重跑。来源：[159000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/specificity/metrics.json)、[140000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/specificity/metrics.json)、[49000 / 100轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/specificity/metrics.json)、[49000 / 200轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915/metrics.json)、[92000 / 200轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915/metrics.json)（均保存 fold SD / OOF），以及 [Ophiuchus 论文 Table 1–5 / Figure 4](/root/oph_paper/oph.pdf)。

<details>
<summary>训练预算诊断：完整末轮 loss / 指标对照</summary>

92000/300、400、500轮是独立预算实验；以下新增三行只在平台Success且完整产物验收通过后回填，不用中间轮次。预算与fixed-last规则见[AB Native Probes §4.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)；[实时状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915/status.json)与[逐项验收/最终比较目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_budget_audit_20260915)提供来源。

<!-- ab92000-budget-results:start -->
| 本地诊断运行 | 末轮训练loss | 末轮留出折loss | Accuracy | macro-F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Ophiuchus [R] 50轮 | 0.638430 | 0.774160 | 0.665071 | 0.664036 | 0.498153 |
| Ophiuchus [R] 100轮 | 0.540101 | 0.777501 | 0.677804 | 0.676818 | 0.517054 |
| Ophiuchus [R] 200轮 | 0.407478 | 0.841788 | 0.670304 | 0.669584 | 0.505715 |
| Ophiuchus [R] 300轮 | 0.317621 | 0.945798 | 0.667571 | 0.666570 | 0.501624 |
| Ours v5 step49000 100轮 | 0.790130 | 未记录 | 0.618000 | 0.616288 | 0.428139 |
| Ours v5 step49000 200轮 | 0.742728 | 0.838495 | 0.627325 | 0.625782 | 0.441693 |
| **Ours v5 step92000 200轮** | **0.707176** | **0.821146** | **0.643468** | **0.641730** | **0.466119** |
| Ours v5 step92000 300轮 | 0.670705 | 0.823670 | 0.648926 | 0.647276 | 0.474183 |
| Ours v5 step92000 400轮 | 0.641995 | 0.830413 | 0.648926 | 0.647463 | 0.473956 |
| Ours v5 step92000 500轮 | 0.618489 | 0.839541 | 0.647563 | 0.646208 | 0.471843 |
| Ours v5 step92000 100轮 | 0.759260 | 0.829224 | 0.632781 | 0.631154 | 0.450079 |
<!-- ab92000-budget-results:end -->

不同训练预算是独立运行，会改变 scheduler，不能把独立 100 轮结果当作 200 轮运行的中间点。留出曲线仅作诊断，正式成绩固定末轮。数据/曲线与历史 baseline 路径见 [学习曲线协议与验收 §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#45-specificity-200-epoch-学习曲线诊断2026-09-15)；本节不是采样消融。

</details>

#### GDPa1：属性回归，参考流程 CV

| 属性 | n | Ophiuchus-Ab论文 [P] | Ours v5 49000 参考CV | Ours v5 92000 参考CV | Ours v5 140000 参考CV | Ours v5 159000 参考CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AC-SINS pH7.4 | 242 | 0.511 | 0.490733 | 0.490949 | 0.512870 | 0.526944 |
| HIC | 242 | 0.550 | 0.389040 | 0.472238 | 0.527719 | 0.517856 |
| PR_CHO | 197 | 0.460 | 0.361299 | 0.476785 | 0.360072 | 0.398506 |
| Titer | 239 | 0.359 | 0.267836 | 0.301977 | 0.349780 | 0.357980 |
| Tm2（附加） | 193 | 未报告 | 0.140118 | 0.070526 | 0.131592 | 0.104690 |

来源：`legacy_reference.cv_mean_spearman`；[159000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/gdp_a1/metrics.json)、[140000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/gdp_a1/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/gdp_a1/metrics.json)、[92000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/gdp_a1/metrics.json)；论文 Table 4。参考流程存在全标签变换与同 CV 选参/报分局限，不当作独立外层测试；不与 raw-label pooled OOF 混用。按用户决定只展示参考流程，历史 nested 诊断已移出主文档，原产物保留。

#### m396：突变体计算能量预测

五个计算能量目标的平均 Spearman，不是实测 Kd。论文 Figure 4 没有可直接转录的六档精确数值，保留缺项，不由本地复跑代填。

| 方法 / checkpoint | 0.5% | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|---|
| Ophiuchus-Ab [P] | 未获取 | 未获取 | 未获取 | 未获取 | 未获取 | 未获取 |
| Ophiuchus-Ab [R] | 0.930065 | 0.944155 | 0.952616 | 0.961872 | 0.967179 | 0.970591 |
| Ours v5 49000 | 0.899690 | 0.921741 | 0.934626 | 0.945228 | 0.950440 | 0.954311 |
| Ours v5 92000 | 0.904707 | 0.925033 | 0.934510 | 0.946026 | 0.951057 | 0.953988 |
| Ours v5 140000 | 0.909132 | 0.927589 | 0.937812 | 0.948321 | 0.952641 | 0.955300 |
| Ours v5 159000 | 0.910117 | 0.928021 | 0.937246 | 0.947397 | 0.952155 | 0.954771 |

有效突变体 86,929；独立完整 H/L 序列 82,311。按 Ophiuchus-Ab 原始代码采用按行随机划分，训练比例按行计、划分 seed=2023，其余行用于测试；各档存在同序列跨 split。按用户决定只展示上述原始划分，附加分组产物不列主表。baseline 特征窗口/pooling 与 Ours 不同。来源：[159000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_159000_full_20260917/m396/metrics.json)、[140000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_140000_full_20260917/m396/metrics.json)、[92000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_full_20260915/m396/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_llada_20260913/m396/metrics.json)、[历史 Ophiuchus metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/affinity_m396_metrics.json)、[原始划分代码](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/in_silico/finetune_in_silico.py:38)。

## 二、TCR 结果

当前 TCR Ours 主行是 **v5 159000**，共 10 项：T1 四轨、T2 两项、T3 两项、T4 两数据集。49000/92000/140000 同行保留对照；49000 完整性仍以 [最终 comparison](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.json) 为准。159000/140000/92000 数字直接取各任务 summary/metrics/fewshot JSON。T1 为保存预测的五折均值，T2/T3 为保存产物聚合，T4 为完整保存序列评分；本轮没有对已写入数字再做一次独立前向重算。

<a id="tcr-binding"></a>

### §0.1 T1 Binding：四种输入分别比较

NM2025 官方 AS 五折；Ours 冻结基础模型、每折训练 MLP，不是全参数微调，也不是关系 token 生成评分。**recognition 位置使用 mask，不给定 `<binding>` 或样本标签。** 原始数据读取后运行时补成 v5 格式，未知区域保留 X；仅观察到的残基参与池化。

每格为 **AUROC / AUPRC（precrec）** 的五折均值；完整折间 sample SD 在各轨 summary.json。论文 [P] 出自 Supplementary Table 7-1…7-9；最终 comparison 保留逐行转录、原表计数与来源 SHA。本地公开文件与论文部分样本数/正负比例不同，尤其影响 AUPRC，不能作为同一行集的严格排名。协议和方法训练方式：[T1 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)。

#### T1-1 大数据 cdr3b：表位＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TEPCAM [P] | 0.8008 / 0.8163 | 0.6116 / 0.5922 | 0.5053 / 0.5075 |
| epiTCR [P] | 0.8142 / 0.8345 | 0.6556 / 0.6976 | 0.5075 / 0.5106 |
| TEIM [P] | 0.7766 / 0.7932 | 0.6090 / 0.6224 | 0.5326 / 0.5218 |
| AttnTAP [P] | 0.5856 / 0.5911 | 0.5216 / 0.5189 | 0.4939 / 0.4981 |
| ERGO-AE [P] | 0.7388 / 0.7519 | 0.5636 / 0.5711 | 0.5099 / 0.5106 |
| TCRGP [P] | 0.7829 / 0.7982 | 0.6317 / 0.6742 | 未报告 |
| ATM-TCR [P] | 0.7728 / 0.7800 | 0.6028 / 0.5838 | 0.5192 / 0.5136 |
| NetTCR [P] | 0.7372 / 0.7482 | 0.5626 / 0.5563 | 0.5203 / 0.5156 |
| TCR-BERT [P] | 0.7645 / 0.7821 | 0.6208 / 0.6157 | 未报告 |
| ERGO-lstm [P] | 0.5022 / 0.5012 | 0.4987 / 0.4981 | 0.5009 / 0.5049 |
| vibtcr [P] | 0.4995 / 0.5007 | 0.5040 / 0.5056 | 0.4981 / 0.4994 |
| VitTCR [P] | 0.7092 / 0.7277 | 0.5675 / 0.5663 | 0.5126 / 0.5129 |
| TCR-H [P] | 0.6865 / 0.6904 | 0.5593 / 0.5562 | 0.5269 / 0.5267 |
| PiTE [P] | 0.6121 / 0.6256 | 0.5581 / 0.5724 | 0.5127 / 0.5065 |
| TEINet [P] | 0.5770 / 0.5887 | 0.5111 / 0.5164 | 0.5025 / 0.5057 |
| ImRex [P] | 0.5895 / 0.5894 | 0.5295 / 0.5331 | 0.5050 / 0.5041 |
| SETE [P] | 0.5457 / 0.5487 | 0.5412 / 0.5381 | 未报告 |
| DeepTCR [P] | 0.5200 / 0.5091 | 0.5280 / 0.5320 | 0.5113 / 0.5118 |
| TPBTE [P] | 0.4999 / 0.5001 | 0.5011 / 0.5001 | 0.5009 / 0.5008 |
| MCMC [P] | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| TITAN [P] | 0.5041 / 0.5021 | 0.5025 / 0.4983 | 0.5039 / 0.4979 |
| DLpTCR-RESNET [P] | 0.3669 / 0.4079 | 0.4668 / 0.4712 | 0.4918 / 0.4918 |
| DLpTCR-FULL [P] | 0.2535 / 0.3598 | 0.4097 / 0.4341 | 0.4916 / 0.4947 |
| DLpTCR-CNN [P] | 0.2976 / 0.3764 | 0.4394 / 0.4541 | 0.4942 / 0.4941 |
| Ours v5 49000 | 0.7450 / 0.7607 | 0.5755 / 0.5583 | 0.5068 / 0.5181 |
| Ours v5 92000 | 0.7510 / 0.7658 | 0.5822 / 0.5630 | 0.5226 / 0.5236 |
| Ours v5 140000 | 0.7528 / 0.7682 | 0.5814 / 0.5701 | 0.5394 / 0.5357 |
| Ours v5 159000 | 0.7562 / 0.7716 | 0.5909 / 0.5709 | 0.5318 / 0.5312 |

来源：[159000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_cdr3b/cdr3b/AS/summary.json)、[140000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_cdr3b/cdr3b/AS/summary.json) 的 `eval_sets.*.local`、[92000 五折 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_cdr3b/cdr3b/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_cdr3b/cdr3b/AS/summary.json)、[49000 逐预测审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/binding_cdr3b.json)。未找到 v5 29k 的同协议大数据结果；不把 others β-only 当成本项旧 checkpoint 对照。历史官方 checkpoint 复跑保留于 [归档 §0.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，其中 ERGO-AE 有丢行，不能当逐行对齐；论文 seen 两列与公开复跑存在偏移。

#### T1-2 others β-only：表位＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| epiTCR [P] | 0.6667 / 0.6835 | 0.5993 / 0.6315 | 0.5168 / 0.5198 |
| TEPCAM [P] | 0.6697 / 0.6611 | 0.5677 / 0.5697 | 0.5266 / 0.5225 |
| TCRGP [P] | 0.6372 / 0.6517 | 0.6023 / 0.6323 | 未报告 |
| TEIM [P] | 0.6488 / 0.6457 | 0.5977 / 0.6154 | 0.5228 / 0.5196 |
| TCR-BERT [P] | 0.6311 / 0.6388 | 0.5972 / 0.6366 | 未报告 |
| ATM-TCR [P] | 0.6398 / 0.6287 | 0.5602 / 0.5714 | 0.5277 / 0.5323 |
| ERGO-AE [P] | 0.6277 / 0.6218 | 0.4798 / 0.5068 | 0.4890 / 0.4933 |
| TCR-H [P] | 0.6244 / 0.6064 | 0.5362 / 0.5358 | 0.5292 / 0.5203 |
| VitTCR [P] | 0.5834 / 0.5817 | 0.5375 / 0.5482 | 0.5143 / 0.5131 |
| NetTCR [P] | 0.5681 / 0.5643 | 0.5021 / 0.5058 | 0.4883 / 0.4946 |
| ImRex [P] | 0.5472 / 0.5532 | 0.4940 / 0.4909 | 0.4976 / 0.5038 |
| PiTE [P] | 0.5382 / 0.5472 | 0.5394 / 0.5461 | 0.4997 / 0.5012 |
| AttnTAP [P] | 0.5450 / 0.5462 | 0.4880 / 0.5010 | 0.5015 / 0.4992 |
| SETE [P] | 0.5217 / 0.5169 | 0.5350 / 0.5482 | 未报告 |
| TEINet [P] | 0.5129 / 0.5146 | 0.4924 / 0.4941 | 0.4940 / 0.5005 |
| TITAN [P] | 0.5003 / 0.5015 | 0.4736 / 0.4878 | 0.4990 / 0.4950 |
| TPBTE [P] | 0.5001 / 0.5012 | 0.5076 / 0.5058 | 0.5009 / 0.4996 |
| DLpTCR-RESNET [P] | 0.5006 / 0.5005 | 0.4979 / 0.4984 | 0.5012 / 0.5021 |
| MCMC [P] | 0.5000 / 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| ERGO-lstm [P] | 0.4995 / 0.4997 | 0.5635 / 0.5513 | 0.4992 / 0.5000 |
| vibtcr [P] | 0.4870 / 0.4891 | 0.4841 / 0.4846 | 0.5012 / 0.4981 |
| DLpTCR-CNN [P] | 0.3990 / 0.4251 | 0.5012 / 0.5056 | 0.4865 / 0.4852 |
| DLpTCR-FULL [P] | 0.3846 / 0.4209 | 0.4735 / 0.4867 | 0.4991 / 0.4969 |
| DeepTCR [P] | 0.3214 / 0.3916 | 0.2778 / 0.3753 | 0.5102 / 0.5113 |
| Ours v5 29000（历史） | 0.6212 / 0.6162 | 0.5811 / 0.5920 | 0.5230 / 0.5187 |
| Ours v5 49000 | 0.6335 / 0.6345 | 0.5961 / 0.6051 | 0.5236 / 0.5149 |
| Ours v5 92000 | 0.6314 / 0.6245 | 0.5924 / 0.5976 | 0.5246 / 0.5175 |
| Ours v5 140000 | 0.6362 / 0.6303 | 0.6020 / 0.6149 | 0.5149 / 0.5151 |
| Ours v5 159000 | 0.6346 / 0.6280 | 0.5817 / 0.5989 | 0.5114 / 0.5125 |

来源：[159000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_others_cdr3b/others_cdr3b/AS/summary.json)、[140000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_others_cdr3b/others_cdr3b/AS/summary.json)、[92000 others β-only summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_others_cdr3b/others_cdr3b/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_others_cdr3b/others_cdr3b/AS/summary.json)。这一轨来自 others 行集，不与上表的大数据 cdr3b 合并。

#### T1-3 others CDR3αβ：表位＋CDR3α＋CDR3β

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TCRGP-AB [P] | 0.6372 / 0.6517 | 0.6023 / 0.6323 | 未报告 |
| NetTCR-AB [P] | 0.5034 / 0.5063 | 0.4976 / 0.5021 | 0.5023 / 0.5047 |
| vibtcr-AB [P] | 0.4964 / 0.4982 | 0.4967 / 0.4920 | 0.5003 / 0.5023 |
| DeepTCR-ABVJ [P] | 0.4876 / 0.4811 | 0.5002 / 0.4944 | 0.5019 / 0.5046 |
| Ours v5 29000（历史） | 0.6590 / 0.6593 | 0.5414 / 0.5458 | 0.5193 / 0.5111 |
| Ours v5 49000 | 0.6556 / 0.6550 | 0.5717 / 0.5790 | 0.5205 / 0.5143 |
| Ours v5 92000 | 0.6755 / 0.6729 | 0.5472 / 0.5518 | 0.5085 / 0.4998 |
| Ours v5 140000 | 0.6685 / 0.6666 | 0.5472 / 0.5536 | 0.5196 / 0.5178 |
| Ours v5 159000 | 0.6765 / 0.6723 | 0.5540 / 0.5601 | 0.5189 / 0.5092 |

DeepTCR-ABVJ 额外使用 V/J，不是相同输入对照。来源：[159000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_cdr3ab/cdr3ab/AS/summary.json)、[140000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_cdr3ab/cdr3ab/AS/summary.json)、[92000 CDR3αβ summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_cdr3ab/cdr3ab/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_cdr3ab/cdr3ab/AS/summary.json)。

#### T1-4 others LongAB：表位＋LongA＋LongB

| 方法 / checkpoint | Seen test | Seen independent | Unseen independent |
|---|---|---|---|
| TCRconv-fullAB [P] | 0.6915 / 0.7609 | 0.5704 / 0.7321 | 未报告 |
| TCRconv-fullA [P] | 0.6373 / 0.7051 | 0.5342 / 0.6891 | 未报告 |
| TCRconv-fullB [P] | 0.6373 / 0.7051 | 0.5342 / 0.6891 | 未报告 |
| Ours v5 29000（历史） | 0.7419 / 0.7336 | 0.5099 / 0.5105 | 0.5071 / 0.5063 |
| Ours v5 49000 | 0.7595 / 0.7537 | 0.5204 / 0.5380 | 0.5051 / 0.5095 |
| Ours v5 92000 | 0.7619 / 0.7524 | 0.5535 / 0.5429 | 0.4767 / 0.4876 |
| Ours v5 140000 | 0.7732 / 0.7626 | 0.5559 / 0.5431 | 0.5129 / 0.5187 |
| Ours v5 159000 | 0.7737 / 0.7597 | 0.5464 / 0.5354 | 0.5119 / 0.5187 |

fullA / fullB 分别是仅长链 α / β；论文重复数字按原表保留。Ours 的 CDR3α/β 只用于核验长链，不额外拼接；MHC、V/J 原字段没有输入。来源：[159000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_159000_20260917_others_longab/others_longab/AS/summary.json)、[140000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_140000_20260917_others_longab/others_longab/AS/summary.json)、[92000 LongAB summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_92000_20260915_others_longab/others_longab/AS/summary.json)、[49000 summary](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_tcr_v5_49000_20260913_others_longab/others_longab/AS/summary.json)。92000 的 unseen independent AUROC 低于 0.5，按原值记录，不解释成有效反向预测。

others 三轨均使用同一公开 AS 行集，每轨 18,868 个唯一输入。论文 TCRconv seen-independent 原表计数为 834（524 正/310 负），当前公开 others 为 626（313/313），不能因 LongAB seen_test AUROC 较高就宣称超过 TCRconv。长链在两个 independent 集上没有体现优势。

29k/49k 的 others 行身份/标签和输入语义已核对，但源码指纹不同，**仅为历史 checkpoint 对照，不是只改权重的因果实验**。140000 / 92000 / 49000 使用同一冻结协议与官方 AS 行集，但仍不是只改 step 的受控消融。全部 49000 差值、逐折预测 SHA 与配对检查见最终 comparison 的 `old_checkpoint_comparison`；新汇总不混入旧协议或其他系列 Ours 数值。

<a id="tcr-clustering"></a>

### §0.2 T2 Clustering

Ours 两项均仅编码 CDR3β，**不输入表位或 binding 标签**；表位仅用于聚类评分。协议：[T2 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)。

#### T2A：阈值聚类，purity–retention

4,779 个配对身份单位；Ours v5 159000 完整曲线 AUC = **0.416182**（140000：0.424461；92000：0.431322；49000：0.439354）。下表比较各baseline retention附近的Ours点，不将单独purity当作整体排名；AUC不与baseline单点purity混比。

| baseline | 单位 | `[P]` retention / purity | `[A]` purity | Ours 92000 实际 retention / purity | Ours 140000 实际 retention / purity | Ours 159000 实际 retention / purity |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HD | chain | 0.27 / 0.69 | 0.694 | 0.270140 / 0.736638 | 0.270140 / 0.725794 | 0.270140 / 0.759876 |
| LD | chain | 0.31 / 0.57 | 0.573 | 0.310316 / 0.670937 | 0.310525 / 0.656334 | 0.310316 / 0.697235 |
| TCRMatch | chain | 0.09 / 0.82 | 0.824 | 0.092279 / 0.988662 | 0.090395 / 0.981481 | 0.087675 / 0.988067 |
| iSMART | chain | 0.18 / 0.75 | 0.748 | 0.180372 / 0.940835 | 0.180372 / 0.947796 | 0.180163 / 0.950058 |
| GIANA | chain | 0.25 / 0.65 | 0.649 | 0.250471 / 0.766082 | 0.250052 / 0.767364 | 0.250262 / 0.785953 |
| clusTCR | pair | 0.09 / 0.99 | 0.993 | 0.092279 / 0.988662 | 0.090395 / 0.981481 | 0.087675 / 0.988067 |
| GLIPH2 | pair | 0.22 / 0.80 | 0.798 | 0.220548 / 0.851044 | 0.220130 / 0.866920 | 0.220130 / 0.889734 |
| DeepTCR | pair | 0.92 / 0.63 | 0.632 | 0.920067 / 0.180805 | 0.920067 / 0.183534 | 0.920276 / 0.187358 |
| TCRdist3 | pair | 0.44 / 0.42 | 0.422 | 0.440050 / 0.453162 | 0.440050 / 0.439848 | 0.440469 / 0.474584 |

92000、140000 与159000 各自九点 gap 均小于预设 0.02，但统计单位 chain/pair 和各方法实际输入范围不同，数值可对齐不代表完全同协议。[P] 为论文 Fig 3A，[A] 为作者聚类产物的既有计数校准；精确 tau / gap / comparison_allowed 与完整曲线来源：[159000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_159000_20260917_runtime_v2/metrics.json)、[140000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_140000_20260917_runtime_v2/metrics.json)、[92000 T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_92000_20260915_runtime_v2/metrics.json)；49000 对照：[T2A metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/ours_tcr_v5_49000_20260913_runtime_v2/metrics.json)。校准表来源见最终 comparison。

#### T2B：embedding → KMeans

9,033 条 β / 25 表位；K=10…100、步长 5，seed=0；报告 **19 个 K 的均值**，不在评价集上选 best K。

| 方法 / checkpoint | mean ARI ↑ | mean NMI ↑ | mean purity ↑ | 输入 / 来源 |
|---|---|---|---|---|
| SCEPTR [L] | 0.0328 | 0.1587 | 0.3388 | β＋TRBV，历史 |
| TCR-BERT MLM-only [L] | 0.0161 | 0.0948 | 0.2869 | β-only，历史 |
| ESM2-150M [L] | 0.0068 | 0.0676 | 0.2584 | β-only，历史 |
| ProtBERT [L] | 0.0056 | 0.0606 | 0.2478 | β-only，历史 |
| k-mer [C] | 0.0102 | 0.0793 | 0.2680 | β-only，历史 |
| Ours v5 49000 | 0.0204 | 0.1243 | 0.3052 | β-only，历史对照 |
| Ours v5 92000 | 0.0197 | 0.1261 | 0.3065 | β-only，历史对照 |
| Ours v5 140000 | 0.0188 | 0.1241 | 0.3047 | β-only，历史对照 |
| Ours v5 159000 | 0.0191 | 0.1237 | 0.3042 | β-only，当前 |

本 K-sweep 的对应论文 [P] 精确数值未获取；本地 baseline 是独立参考层，不拿 T2A 或 T3 数字代填。[L]/[C] 未在本轮重跑。来源：[159000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_159000_20260917_runtime_v2/metrics.json)、[140000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_140000_20260917_runtime_v2/metrics.json) 的 `kmeans_*_mean`、[92000 T2B metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_92000_20260915_runtime_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_49000_20260913_runtime_v2/metrics.json) / [49000 全部 K 点](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering_embed/ours_tcr_v5_49000_20260913_runtime_v2/curve.csv)；历史 baseline 原始路径/SHA 在最终 comparison。没有 v5 29k 同协议 T2 结果。

<a id="tcr-representation"></a>

### §0.3 T3 Representation：few-shot NN AUROC

冻结受体表征、不训练分类头。Ours 输入 CDR3αβ，不输入表位/binding；表位标签用于 support/query 抽样和 AUROC。**本地 broad/deep 不是 SCEPTR 论文同版数据复现。** 协议与 baseline 输入核查：[T3 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。

#### T3-broad：24 表位本地任务

Ours 唯一受体 train/test=6,885/1,721，每个有效表位/shot 使用 5 seeds。表内均为 macro AUROC。

| 方法 / checkpoint | k=1 | k=2 | k=5 | k=10 | k=20 | k=50 | k=100 |
|---|---|---|---|---|---|---|---|
| SCEPTR [L] | 0.5869 | 0.6219 | 0.6552 | 0.6919 | 0.7110 | 0.7152 | 0.7413 |
| TCRdist [L] | 0.5772 | 0.6197 | 0.6439 | 0.6740 | 0.6870 | 0.6939 | 0.7282 |
| CDR3-Levenshtein [C] | 0.5545 | 0.5757 | 0.5971 | 0.6265 | 0.6414 | 0.6543 | 0.6834 |
| k-mer [C] | 0.5425 | 0.5702 | 0.5967 | 0.6231 | 0.6432 | 0.6479 | 0.6734 |
| ESM2-150M [L] | 0.5613 | 0.5830 | 0.5894 | 0.6191 | 0.6381 | 0.6419 | 0.6706 |
| ProtBERT [L] | 0.5496 | 0.5761 | 0.5870 | 0.6073 | 0.6273 | 0.6282 | 0.6506 |
| TCR-BERT MLM-only [L] | 0.5470 | 0.5671 | 0.5773 | 0.6034 | 0.6231 | 0.6244 | 0.6414 |
| Ours v5 49000 | 0.5435 | 0.5720 | 0.5949 | 0.6127 | 0.6335 | 0.6180 | 0.6447 |
| Ours v5 92000 | 0.5422 | 0.5748 | 0.5925 | 0.6073 | 0.6258 | 0.6237 | 0.6479 |
| Ours v5 140000 | 0.5445 | 0.5727 | 0.5958 | 0.6126 | 0.6302 | 0.6210 | 0.6497 |
| Ours v5 159000 | 0.5443 | 0.5732 | 0.5949 | 0.6121 | 0.6299 | 0.6226 | 0.6492 |

Ours 有效表位数依次为 24/24/24/24/24/20/15，高 shot 的均值不是同一个固定人群。历史 baseline 尚未用本轮唯一受体/多标签规则重跑，**只能并列参考，不能据此给修正后协议排名**；其输入字段差异见下方 deep 说明。本地 broad 对应的论文 [P] 数值未获取。

来源：[159000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_159000_20260917_runtime_v2/fewshot.json)、[140000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_140000_20260917_runtime_v2/fewshot.json) 的 `auroc_by_shot` / `n_epitopes_by_shot`、[92000 broad fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_92000_20260915_runtime_v2/fewshot.json)、[49000 fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation/ours_tcr_v5_49000_20260913_runtime_v2/fewshot.json)、[49000 support / 聚合核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_short.json)。std 是表位间 population SD，不是置信区间；完整 std 保留原 JSON。

#### T3-deep：六表位本地任务

25,816 条 universe；8 档 shots × 100 seeds × 6 targets。以下先保留论文代表值，再列本地完整曲线。

| 论文模型 [P] | Table SI，k=200 六表位宏均值 |
|---|---|
| CDR3-Levenshtein | 0.736833 |
| ESM2-T6-8M | 0.716667 |
| ProtBERT | 0.705333 |
| SCEPTR | 0.798167 |
| TCR-BERT | 0.734667 |
| TCRdist | 0.783167 |

论文宏均值由六个印刷三位小数值计算，来源与逐表位转录在最终 comparison 的 `sceptr_paper_baselines`。ESM2 论文版本为 **T6-8M＋重建长链**，不是本地 ESM2-150M＋CDR3。论文数据与本地不同，不能直接相减解释模型能力差距。

| 本地方法 / checkpoint | k=1 | k=2 | k=5 | k=10 | k=20 | k=50 | k=100 | k=200 |
|---|---|---|---|---|---|---|---|---|
| SCEPTR [L] | 0.6401 | 0.6673 | 0.7020 | 0.7182 | 0.7375 | 0.7602 | 0.7748 | 0.7903 |
| TCRdist [L] | 0.6466 | 0.6724 | 0.7008 | 0.7156 | 0.7325 | 0.7516 | 0.7642 | 0.7770 |
| CDR3-Levenshtein [C] | 0.6256 | 0.6480 | 0.6712 | 0.6836 | 0.6958 | 0.7110 | 0.7223 | 0.7334 |
| k-mer [C] | 0.6074 | 0.6334 | 0.6653 | 0.6777 | 0.6901 | 0.7059 | 0.7173 | 0.7284 |
| ESM2-150M [L] | 0.5914 | 0.6055 | 0.6315 | 0.6429 | 0.6563 | 0.6738 | 0.6848 | 0.6958 |
| ProtBERT [L] | 0.5847 | 0.5992 | 0.6237 | 0.6343 | 0.6480 | 0.6664 | 0.6808 | 0.6944 |
| TCR-BERT MLM-only [L] | 0.5786 | 0.5989 | 0.6309 | 0.6423 | 0.6563 | 0.6699 | 0.6806 | 0.6916 |
| Ours v5 49000 | 0.5810 | 0.6056 | 0.6327 | 0.6452 | 0.6550 | 0.6700 | 0.6792 | 0.6895 |
| Ours v5 92000 | 0.5753 | 0.5985 | 0.6282 | 0.6406 | 0.6514 | 0.6668 | 0.6773 | 0.6871 |
| Ours v5 140000 | 0.5784 | 0.5990 | 0.6297 | 0.6426 | 0.6542 | 0.6692 | 0.6795 | 0.6890 |
| Ours v5 159000 | 0.5794 | 0.5994 | 0.6281 | 0.6413 | 0.6526 | 0.6678 | 0.6785 | 0.6882 |

历史 SCEPTR / TCRdist 有额外 V 或 V/J 信息；Levenshtein/k-mer/ESM2/ProtBERT 为 CDR3αβ 控制，TCR-BERT MLM-only 为 β-only 控制。历史 baseline 的输入/运行身份未按本轮全部门禁重新锁定，不把 [L]/[C] 升级为论文复现。

来源：[159000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_159000_20260917_runtime_v2/fewshot.json)、[140000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_140000_20260917_runtime_v2/fewshot.json)、[92000 deep fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_92000_20260915_runtime_v2/fewshot.json)、[49000 fewshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/ours_tcr_v5_49000_20260913_runtime_v2/fewshot.json)、[49000 deep 身份/覆盖/聚合核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_deep.json)。未重跑未保存的全部 NN 距离，不能声称独立重算所有 AUROC。v5 29k 的 broad/deep 同协议成绩均未获取。

<a id="tcr-generation"></a>

### §0.4 T4 Generation：表位条件 CDR3β

**2026-09-16 共享 sampler 修正提示：** 49000/92000 多步 Ours 也调用相同的 ESMC+LLaDA 采样路径，数值仍为修复前协议记录，不作修正版 headline。140000 是修复后代码路径上的新生成（pairing 同日矩阵已标 `committed_generated_residues_pending_mask`）；本轮没有单独做 T4 ESMC 逐轮输入审计，不把 140000 写成已完成 T4 反馈验收。详见 [T4 §7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。 159000 为修复后路径，真实 GPU gate 通过目标占位符不变性、固定context及评分JSON检查；这不等于对全部正式候选逐轮抓取ESMC输入。

当前 Ours 为 v5 159000，输入表位＋MHC pseudo（34 aa）；未知框架、CDR1/2 和 α 保持 X，仅 CDR3β core 参与生成。每个目标 **K=100 随机候选＋1 独立 greedy**，max_iter=32，长度来自训练先验，不读取测试参考 CDR3 长度。140000为修复后历史对照；49000/92000为修复前协议记录，不作修正版headline。当前不是 αβ 配对生成，后者仍延期。

表中 d_edit 越低越好，seq-recovery / F1@100 越高越好。序列相似度不等于实验结合，F1=0 只说明未精确命中参考集合。相同目标集合仍可能有 MHC、参考长度、候选筛选和采样预算差异；[T4 协议 / baseline 条件审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) 中的限制继续有效。

#### T4-1 benchmark14：全部 14 个目标

| 方法 / checkpoint | 覆盖 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|---|
| TCRT5 | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| GRATCR | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| ER | 14目标未重评分 | — | — | — | [A] 当前正式评分为13目标，见下表 |
| TcrDesign | 14/14 | 4.322143 | 0.613834 | 0.001082 | [R] 历史原生权重管线 |
| TCRDiff | 14/14 | 4.719761 | 0.587113 | 0.000922 | [R] 历史原生权重管线 |
| OLGA | 14/14 | 5.931429 | 0.500620 | 0.000000 | [C] 无条件 |
| Ours v5 49000 | 14/14 | 5.674286 | 0.539441 | 0.000000 | 历史对照，完整 β-only |
| Ours v5 92000 | 14/14 | 5.722857 | 0.538980 | 0.000000 | 历史对照，完整 β-only |
| Ours v5 140000 | 14/14 | 4.923571 | 0.579153 | 0.000000 | 历史完整 β-only |
| Ours v5 159000 | 14/14 | 4.882857 | 0.582537 | 0.000000 | 当前完整 β-only |

**不是所有模型只测试了六个目标。** 14 是完整集合；13 是排除 `RVRAYTYSK_HLA-A*03:01` 模拟保留目标的论文主视图；6 是其中本轮 v5 训练/验证均未出现的表位子集。官方三种方法的原始存储预测也含 RVR，但当前正式评分产物未对其补成完整 14，不填造分数。来源：[159000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_159000_20260917_benchmark14_pmhc_beta_v2/metrics.json)、[140000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_140000_20260917_benchmark14_pmhc_beta_v2/metrics.json)、[92000 benchmark14 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_92000_20260915_benchmark14_pmhc_beta_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_benchmark14_pmhc_beta_v2/metrics.json)。

#### 论文主视图 sparse13：官方预测重评分

| 方法 / checkpoint | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|
| TCRT5 | 4.467385 | 0.601754 | 0.000296 | [A] 官方预存预测 |
| GRATCR | 4.396615 | 0.595101 | 0.000000 | [A] 官方预存预测 |
| ER | 6.067805 | 0.336948 | 0.000000 | [A] 官方预存预测 |
| Ours v5 49000 | 5.781538 | 0.530099 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 92000 | 5.840000 | 0.529980 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 140000 | 5.053077 | 0.567773 | 0.000000 | 同一14目标产物的13目标视图 |
| Ours v5 159000 | 5.018462 | 0.571268 | 0.000000 | 同一14目标产物的13目标视图 |

这里是论文主目标集合上的 **[A] 重评分**，不把重评分的小数冒充独立转录的 [P]。最终收集器没有单独的生成论文 [P] 精确指标项；TcrDesign / TCRDiff 的原生论文协议也不同，对应 [P] 比较值保留“未获取/未对齐”，不以 held20 本地结果替代论文测试成绩。

#### common6：未见表位子集

| 方法 / checkpoint | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|
| TCRT5 | 4.696333 | 0.584639 | 0.000329 | [A] |
| GRATCR | 4.604667 | 0.583890 | 0.000000 | [A] |
| ER | 6.074833 | 0.320566 | 0.000000 | [A] |
| TcrDesign | 4.856667 | 0.573352 | 0.000000 | [R] 历史管线 |
| TCRDiff | 5.157347 | 0.543338 | 0.000000 | [R] 历史管线 |
| OLGA | 6.343333 | 0.469924 | 0.000000 | [C] 无条件 |
| Ours v5 49000 | 5.890000 | 0.519255 | 0.000000 | 历史对照，β-only |
| Ours v5 92000 | 5.975000 | 0.519036 | 0.000000 | 历史对照，β-only |
| Ours v5 140000 | 5.328333 | 0.544756 | 0.000000 | 历史 β-only |
| Ours v5 159000 | 5.211667 | 0.552989 | 0.000000 | 当前 β-only |

6 个目标是相同目标比较，不是全部模型的最大覆盖交集，也不证明所有生成条件已对齐。来源：最终 comparison 的 `generation_baselines` 及同一 benchmark14 metrics。

#### T4-2 held20：已知表位补充测试

| 方法 / checkpoint | 覆盖 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 来源 |
|---|---|---|---|---|---|
| TcrDesign | 20/20 | 1.474000 | 0.870222 | 0.152686 | [R] 历史管线 |
| TCRDiff | 20/20 | 2.953920 | 0.731759 | 0.011777 | [R] 历史管线 |
| Ours v5 49000 | 20/20 | 4.100500 | 0.658046 | 0.000500 | 历史对照，β-only |
| Ours v5 92000 | 20/20 | 4.198000 | 0.650140 | 0.000000 | 历史对照，β-only |
| Ours v5 140000 | 20/20 | 2.983500 | 0.737351 | 0.001500 | 历史 β-only |
| Ours v5 159000 | 20/20 | 2.971000 | 0.740938 | 0.002500 | 当前 β-only |

held20 来自 TCRT5 官方验证参考，不是“我们的 20 条验证样本”，也不是新表位测试。当前正式 TCRT5/GRATCR/ER 官方预测及 OLGA 评分产物未覆盖本集合，不从其他集合借数。来源：[159000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_159000_20260917_held20_pmhc_beta_v2/metrics.json)、[140000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_140000_20260917_held20_pmhc_beta_v2/metrics.json)、[92000 held20 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_92000_20260915_held20_pmhc_beta_v2/metrics.json)、[49000 metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_held20_pmhc_beta_v2/metrics.json)、[49000 完整序列重评分审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/held20.json)。

#### 训练重叠与视图解释（所有 T4 表必读）

| Ours v5 92000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 5.722857 | 0.538980 | 0.000000 | 0.653244 |
| 其中训练已见表位（不含 RVR） | 7 | 5.724286 | 0.539362 | 0.000000 | 0.650969 |
| 其中未见表位（common6） | 6 | 5.975000 | 0.519036 | 0.000000 | 0.612743 |
| held20 全部已见表位 | 20 | 4.198000 | 0.650140 | 0.000000 | 0.912557 |

| Ours v5 140000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 4.923571 | 0.579153 | 0.000000 | 0.739736 |
| 其中训练已见表位（不含 RVR） | 7 | 4.817143 | 0.587502 | 0.000000 | 0.802045 |
| 其中未见表位（common6） | 6 | 5.328333 | 0.544756 | 0.000000 | 0.623664 |
| held20 全部已见表位 | 20 | 2.983500 | 0.737351 | 0.001500 | 0.947890 |

| Ours v5 159000 视图 | 目标数 | d_edit ↓ | seq-rec ↑ | F1@100 ↑ | 独立 greedy Char-BLEU |
|---|---|---|---|---|---|
| benchmark14 全量 | 14 | 4.882857 | 0.582537 | 0.000000 | 0.763420 |
| 其中训练已见表位（不含 RVR） | 7 | 4.852857 | 0.586936 | 0.000000 | 0.801371 |
| 其中未见表位（common6） | 6 | 5.211667 | 0.552989 | 0.000000 | 0.687139 |
| held20 全部已见表位 | 20 | 2.971000 | 0.740938 | 0.002500 | 0.948652 |

这是分层统计，不是额外训练或采样消融。T4 去污染以“β core＋表位”参考配对为键，不是删除整个表位：benchmark14 训练集仍有 7 个表位，但参考 β core / 配对精确命中均为 0。held20 的 20 个表位全部出现，且很多参考 β core 在其他上下文中出现，但参考配对精确命中为 0；不能称全新 β / 全新表位。

完整 source / train / valid 数字、七表位名单及数据成员≠实际 batch 消费的边界，唯一详见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) 和 [最终 overlap 报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/audit_v5_overlap_20260913/full_scan_v2/report.json)。不覆盖 ESMC 更早预训练或近同源。

Ours Char-BLEU 使用独立 greedy，不能与历史 `gen[0]` 替代 greedy 的行直接排序。缺少 conditioning 参考时 novelty 为 null，不是新颖性为 0；主指标覆盖完整。没有 v5 29k 同协议 T4 结果；没有当前 αβ 配对质量表。

## 文档与历史证据

- 数字只在本文件维护；协议/缺陷见 [AB 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)、[TCR 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 和各任务文档。
- 本次整理前全文（含旧模型、失效采样、PPI/NbBench/FLAb 和历史完整 baseline 表）：[RESULTS_ARCHIVE_20260915.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)。归档不作为当前排行榜，不继续追加新数字。
- 执行时间/提交取消日志见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)；文档整理记录见 [PROGRESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md)。2026-09-17 已加入 v5 159000 全部 AB/TCR 最终汇总（AB pairing p0/CFG0/16步；CDR 2步；Specificity 100 epoch；TCR T4 32步），49000/92000/140000历史数字保留。核对边界与30个来源文件SHA见 [汇总核对记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_159000_20260917/results_document_audit.json)。本轮没有重新训练、推理、评分、submit或cancel。
