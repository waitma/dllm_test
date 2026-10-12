# R20–R27：500步人工诊断终态

生成时间：2026-10-01T01:21:14.079228+00:00。8/8任务均succeeded，500步COMPLETE、summary SHA、worker returncode=0、39份源码及对应profile证据全部核验；没有执行失败或NaN。8个run目录没有模型权重或checkpoint。

这是576条人工训练/72条heldout、单seed=42的诊断，mode=train不代表真实生物语料训练。使用LLaDA270原d768/L8/H12/MLP3072主干及紧凑蛋白词表；500步×global256=128,000次样本呈现/方法。完整目标字符串分离，但共享周期8人工规则，内部字母表为AGSTNQDE，并保留C/F/W边界；条件显式给出代码与长度档，全部target scope为full。

[完整JSON证据及原始artifact链接](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r20_r27_stage500_summary.json)；[逐25步发现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_FINDINGS.md)。

## 判读与下一阶段依据

最近100步的R21（head gate）、R26（local/global）、R24（fixed scale）mean loss均值为0.186520、0.186906、0.187263，极差只有0.000743；这不足以宣布唯一胜出架构。R24的conditional full-AA均值最高（99.00%），R26为98.77%、R21为98.53%，对照R20为97.01%。R26在这个窗口的条件AA范围最窄（98.06%–99.44%），但2048长度profile峰值比对照高28.51%；R21高0.45%，R24基本相同。保留R20对照，R21/R24适合作为低额外显存候选，R26适合作为较高显存成本的候选继续比较；这些是人工诊断支持的后续比较方向，不是生物任务结论。

学习速度和终点排名需要分开。按每25步的已记录点，R26从300步起条件full-AA一直至少95%，R20从325步起、R21从350步起；R24/R23在350步先超过95%，375步又低于95%，400步后恢复。R22直到475步达到95%，R27虽在425步达到95%，475步仍有回落；R25终态为91.03%，仍在追近。500单点最低mean loss是R22，但其最近100步loss均值0.204979，不能据该单点替代整条曲线的判断。

partial-mask三档AA终态已在99.70%–99.74%，条件full-AA多项也接近100%，说明这份可重复人工规则的主要测试已接近饱和。无条件full-mask项贡献终态总mean loss的81.73%–95.41%（R26为93.36%）；计算为该组loss×该组selected_count/全组selected_count/4，逐项贡献保存在JSON中。这个有输入冲突的分组会明显左右总体loss，不应把降低总体loss直接解释为条件生成更好。没有必要为了从此toy中选出唯一赢家而统一延长全部8项。

终态训练probe与heldout的条件AA差值在−2.91至+2.64个百分点，且probe只有固定前72条。这可用于发现明显异常，不能证明真实数据无过拟合。三类条件任务组在终态仍有不同排序，例如R24的人工ab_antigen为100%，R26的人工tcr_pmhc为99.42%，R27为99.48%；分组名称不代表真实下游性能。

2026-10-01优先级更新：下一阶段保留当前release及完整证据，使用现有attention核对修正后的Trainer和新评估；按用户确认，独立CUDA算子测试不再是主线前置条件。新增首次END、误报率和长度指标，才能回答终止位置是否正确。真实语料的短程训练仍以覆盖审计及所需证书完成为前提。当前监督未添加任务、扩步、重试或放宽任何门禁。

## 500步固定heldout指标

mean为25/50/75/100%四档等权均值；每档内部按token计数汇总。END为真实端点位置的召回率，首次END位置/长度准确率和END precision本轮未测。

| 方法 | mean loss | mean AA | mean END召回 | 条件full-AA | 条件full-END召回 | 无条件full-AA | 无条件full-END召回 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R20 | 0.206563 | 91.99% | 100.00% | 95.27% | 100.00% | 37.12% | 100.00% |
| R21 | 0.195153 | 92.48% | 100.00% | 99.13% | 100.00% | 36.83% | 100.00% |
| R22 | 0.190045 | 92.32% | 100.00% | 97.29% | 100.00% | 37.64% | 100.00% |
| R23 | 0.190681 | 92.34% | 99.24% | 98.71% | 100.00% | 36.13% | 93.33% |
| R24 | 0.190901 | 92.36% | 100.00% | 99.45% | 100.00% | 35.31% | 100.00% |
| R25 | 0.215730 | 91.44% | 99.24% | 91.03% | 100.00% | 37.16% | 93.33% |
| R26 | 0.193827 | 92.38% | 100.00% | 98.33% | 100.00% | 36.91% | 100.00% |
| R27 | 0.192513 | 92.45% | 99.24% | 98.44% | 100.00% | 37.17% | 93.33% |

## 最近100步稳定性

固定窗口包含400、425、450、475、500五个检查点。均值与范围只是同一条训练曲线的描述性统计，不是独立种子或显著性检验。

| 方法 | loss均值 | loss范围 | 条件full-AA均值 | 条件full-AA范围 | 条件full-END召回范围 |
| --- | --- | --- | --- | --- | --- |
| R20 | 0.193457 | 0.180679–0.206563 | 97.01% | 95.27%–98.27% | 91.67%–100.00% |
| R21 | 0.186520 | 0.176513–0.195153 | 98.53% | 96.47%–99.61% | 97.22%–100.00% |
| R22 | 0.204979 | 0.190045–0.217519 | 93.12% | 87.76%–97.29% | 86.11%–100.00% |
| R23 | 0.190259 | 0.182640–0.201471 | 97.86% | 96.34%–98.85% | 98.61%–100.00% |
| R24 | 0.187263 | 0.178409–0.191695 | 99.00% | 97.74%–99.85% | 98.61%–100.00% |
| R25 | 0.238029 | 0.215730–0.259769 | 84.47% | 76.97%–91.03% | 100.00%–100.00% |
| R26 | 0.186906 | 0.183305–0.193827 | 98.77% | 98.06%–99.44% | 95.83%–100.00% |
| R27 | 0.210674 | 0.192513–0.235941 | 94.73% | 89.56%–98.44% | 94.44%–100.00% |

## 成本与资源

全部方法实测推荐micro-batch32、梯度累积8、global256。profile峰值来自actual_length=2048的A100 80GB测试；500步峰值来自较短人工输入，不能用后者替换长输入profile。worker时长含启动/初始评估/训练/周期及终态评估；runner时长不含启动和初始评估。

| 方法 | 参数M | 2048 profile峰值GiB | 500诊断峰值GiB | worker分钟 | runner分钟 |
| --- | --- | --- | --- | --- | --- |
| R20 | 75.601 | 41.995 | 5.413 | 238.54 | 238.37 |
| R21 | 75.675 | 42.184 | 5.470 | 239.05 | 238.85 |
| R22 | 75.601 | 41.995 | 5.413 | 238.65 | 238.43 |
| R23 | 75.613 | 41.996 | 5.413 | 238.63 | 238.43 |
| R24 | 75.601 | 41.995 | 5.413 | 238.55 | 238.35 |
| R25 | 75.601 | 48.120 | 5.967 | 238.75 | 238.55 |
| R26 | 75.601 | 53.968 | 6.453 | 238.77 | 238.55 |
| R27 | 75.601 | 42.183 | 5.469 | 238.66 | 238.44 |

8项按每job独占1GPU的worker wall折算合计31.83 GPU·小时，包含上述验证开销，不含此前profile及工作池待机；这不是测得的CUDA内核活跃时间。

## 训练probe对照

训练probe仅为训练集前72条固定记录，不是完整576条训练集的评估；与heldout同样保留18个分层。

| 方法 | heldout mean loss | train probe mean loss | heldout条件full-AA | train probe条件full-AA | probe−heldout |
| --- | --- | --- | --- | --- | --- |
| R20 | 0.206563 | 0.199209 | 95.27% | 97.91% | +2.64 pp |
| R21 | 0.195153 | 0.191696 | 99.13% | 99.09% | -0.04 pp |
| R22 | 0.190045 | 0.195139 | 97.29% | 97.60% | +0.31 pp |
| R23 | 0.190681 | 0.177173 | 98.71% | 99.53% | +0.82 pp |
| R24 | 0.190901 | 0.178828 | 99.45% | 98.62% | -0.82 pp |
| R25 | 0.215730 | 0.220531 | 91.03% | 88.95% | -2.08 pp |
| R26 | 0.193827 | 0.183285 | 98.33% | 99.07% | +0.74 pp |
| R27 | 0.192513 | 0.207426 | 98.44% | 95.53% | -2.91 pp |

## 各人工任务组的full-mask AA

每个任务组内部按三个长度档的AA token数汇总。任务名描述人工上下文组织，不是生物下游任务分数；完整18分层仍在JSON和实时监督MD中。

| 方法 | ab_antigen | tcr_peptide | tcr_pmhc | unconditional_ab | unconditional_tcr | unconditional_single |
| --- | --- | --- | --- | --- | --- | --- |
| R20 | 94.09% | 99.69% | 92.03% | 36.92% | 37.87% | 36.02% |
| R21 | 99.26% | 99.66% | 98.46% | 36.02% | 37.96% | 36.21% |
| R22 | 97.60% | 97.41% | 96.86% | 36.21% | 39.04% | 37.68% |
| R23 | 99.91% | 97.75% | 98.46% | 36.02% | 36.76% | 35.10% |
| R24 | 100.00% | 99.75% | 98.58% | 33.25% | 36.76% | 36.51% |
| R25 | 89.69% | 86.85% | 96.55% | 36.64% | 37.56% | 37.38% |
| R26 | 97.48% | 98.09% | 99.42% | 35.65% | 37.81% | 37.62% |
| R27 | 96.86% | 98.98% | 99.48% | 34.24% | 39.93% | 37.50% |

## 四档完整指标

| 方法 | mask档 | loss | AA | END召回 | AA数 | END数 |
| --- | --- | --- | --- | --- | --- | --- |
| R20 | mask25 | 0.000122 | 100.00% | 100.00% | 4428 | 39 |
| R20 | mask50 | 0.003412 | 99.86% | 100.00% | 8819 | 69 |
| R20 | mask75 | 0.018543 | 99.27% | 100.00% | 13330 | 107 |
| R20 | mask100 | 0.804176 | 68.84% | 100.00% | 17864 | 132 |
| R21 | mask25 | 0.000038 | 100.00% | 100.00% | 4428 | 39 |
| R21 | mask50 | 0.003464 | 99.82% | 100.00% | 8819 | 69 |
| R21 | mask75 | 0.026481 | 99.27% | 100.00% | 13330 | 107 |
| R21 | mask100 | 0.750630 | 70.81% | 100.00% | 17864 | 132 |
| R22 | mask25 | 0.000181 | 100.00% | 100.00% | 4428 | 39 |
| R22 | mask50 | 0.003552 | 99.85% | 100.00% | 8819 | 69 |
| R22 | mask75 | 0.018603 | 99.23% | 100.00% | 13330 | 107 |
| R22 | mask100 | 0.737844 | 70.17% | 100.00% | 17864 | 132 |
| R23 | mask25 | 0.000050 | 100.00% | 100.00% | 4428 | 39 |
| R23 | mask50 | 0.003879 | 99.83% | 100.00% | 8819 | 69 |
| R23 | mask75 | 0.025669 | 99.27% | 100.00% | 13330 | 107 |
| R23 | mask100 | 0.733126 | 70.26% | 96.97% | 17864 | 132 |
| R24 | mask25 | 0.000076 | 100.00% | 100.00% | 4428 | 39 |
| R24 | mask50 | 0.003842 | 99.84% | 100.00% | 8819 | 69 |
| R24 | mask75 | 0.021001 | 99.30% | 100.00% | 13330 | 107 |
| R24 | mask100 | 0.738684 | 70.29% | 100.00% | 17864 | 132 |
| R25 | mask25 | 0.000234 | 100.00% | 100.00% | 4428 | 39 |
| R25 | mask50 | 0.004378 | 99.84% | 100.00% | 8819 | 69 |
| R25 | mask75 | 0.016702 | 99.38% | 100.00% | 13330 | 107 |
| R25 | mask100 | 0.841605 | 66.54% | 96.97% | 17864 | 132 |
| R26 | mask25 | 0.000045 | 100.00% | 100.00% | 4428 | 39 |
| R26 | mask50 | 0.004120 | 99.82% | 100.00% | 8819 | 69 |
| R26 | mask75 | 0.021563 | 99.30% | 100.00% | 13330 | 107 |
| R26 | mask100 | 0.749581 | 70.41% | 100.00% | 17864 | 132 |
| R27 | mask25 | 0.000043 | 100.00% | 100.00% | 4428 | 39 |
| R27 | mask50 | 0.003938 | 99.86% | 100.00% | 8819 | 69 |
| R27 | mask75 | 0.018720 | 99.35% | 100.00% | 13330 | 107 |
| R27 | mask100 | 0.747350 | 70.59% | 96.97% | 17864 | 132 |

## 学习曲线与证据边界

![完整500步曲线](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/figures/immune_refined_synthetic500_progress.png)

[SVG](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/figures/immune_refined_synthetic500_progress.svg)；[曲线job/SHA证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/figures/immune_refined_synthetic500_progress.json)。

无条件full-mask存在相同输入对应多种目标代码/长度的冲突；独立审计得到该heldout AA上限60.5911%、首次END上限33.3333%。这些是有限集合可识别性界限，不是模型一定可学到或应拟合的泛化目标。END召回100%可以通过在多个候选位置同时输出END达到，不能解释为长度正确。

[可识别性审计及逐输入计数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/expanded_identifiability.json)。

本轮不提供真实语料、下游结合任务、未见长度/片段scope、独立种子或多步生成质量的证据。结果不自动解除数据门禁，不自动延长训练或提交新任务。

## 逐任务原始证据

| 方法 | job_id | summary | COMPLETE | worker结果 |
| --- | --- | --- | --- | --- |
| R20 | 1790803305045499169_r20_refined_control_a37dc5ab | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305045499169_r20_refined_control_a37dc5ab/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305045499169_r20_refined_control_a37dc5ab/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305045499169_r20_refined_control_a37dc5ab.json) |
| R21 | 1790803305051081571_r21_refined_head_gate_9cc7449e | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305051081571_r21_refined_head_gate_9cc7449e/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305051081571_r21_refined_head_gate_9cc7449e/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305051081571_r21_refined_head_gate_9cc7449e.json) |
| R22 | 1790803305054490194_r22_refined_fixed_gate_72a52fe0 | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305054490194_r22_refined_fixed_gate_72a52fe0/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305054490194_r22_refined_fixed_gate_72a52fe0/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305054490194_r22_refined_fixed_gate_72a52fe0.json) |
| R23 | 1790803305057942502_r23_refined_layerscale_56d8467a | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305057942502_r23_refined_layerscale_56d8467a/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305057942502_r23_refined_layerscale_56d8467a/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305057942502_r23_refined_layerscale_56d8467a.json) |
| R24 | 1790803305061375040_r24_refined_fixed_scale_4edeef26 | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305061375040_r24_refined_fixed_scale_4edeef26/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305061375040_r24_refined_fixed_scale_4edeef26/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305061375040_r24_refined_fixed_scale_4edeef26.json) |
| R25 | 1790803305064798754_r25_refined_late_fusion_5cd5e385 | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305064798754_r25_refined_late_fusion_5cd5e385/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305064798754_r25_refined_late_fusion_5cd5e385/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305064798754_r25_refined_late_fusion_5cd5e385.json) |
| R26 | 1790803305068196112_r26_refined_local_global_5bf50a1c | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305068196112_r26_refined_local_global_5bf50a1c/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305068196112_r26_refined_local_global_5bf50a1c/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305068196112_r26_refined_local_global_5bf50a1c.json) |
| R27 | 1790803305071555086_r27_refined_value_residual_52f013e5 | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305071555086_r27_refined_value_residual_52f013e5/summary.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790803305071555086_r27_refined_value_residual_52f013e5/COMPLETE.json) | [JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/results/1790803305071555086_r27_refined_value_residual_52f013e5.json) |
