# R20–R27：Refined profile 与25步人工诊断终态

观测：2026-09-30T21:21:55.738501+00:00。

8项profile和25步人工诊断全部成功，39个生产源码指纹及完成证书的summary摘要全部匹配，无执行失败或NaN。池内目前累计28项成功。25步各约718秒；这只验证工程、显存与早期人工规则学习，不是生物语料初训或下游评测。

沿用实际 LLaDA270 的 d768/L8/H12/MLP3072 主干，采用蛋白词表。8项micro32 × accumulation8 = global256，25步即6400次人工样本曝光；576条train/72条heldout valid、6任务组×3长度组，真实残基长度24/127/255。

## 25步终态

四档均值对25/50/75/100% mask等权；每档内部按对应token计数汇总。end指监督的结束标记，与实际序列末尾F/W残基不同。

| 方法 | 初始mean loss | 最终mean loss | 最终mean AA | 最终mean end | Full-mask AA | Full-mask end | train-probe mean loss | 秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r20_refined_control | 3.952055 | 1.907822 | 30.06% | 0.00% | 26.56% | 0.00% | 1.879378 | 717.85 |
| r21_refined_head_gate | 4.088158 | 1.842231 | 32.48% | 0.00% | 31.72% | 0.00% | 1.786728 | 719.18 |
| r22_refined_fixed_gate | 4.105877 | 1.850581 | 32.31% | 0.00% | 30.56% | 0.00% | 1.797944 | 717.59 |
| r23_refined_layerscale | 3.826016 | 1.845777 | 32.04% | 0.00% | 31.17% | 0.00% | 1.784199 | 717.97 |
| r24_refined_fixed_scale | 3.826016 | 1.845360 | 32.10% | 0.00% | 31.24% | 0.00% | 1.783512 | 717.65 |
| r25_refined_late_fusion | 4.041661 | 1.898732 | 31.14% | 0.00% | 29.52% | 0.00% | 1.873512 | 717.99 |
| r26_refined_local_global | 4.013172 | 1.768326 | 36.38% | 0.00% | 29.73% | 0.00% | 1.705406 | 718.47 |
| r27_refined_value_residual | 4.129529 | 1.819044 | 32.69% | 0.00% | 28.69% | 0.00% | 1.798192 | 717.98 |

train_probe只取训练集合前72条，不能当作全训练集loss；其作用是与72条heldout一起观察人工规则学习和过拟合迹象。

R26局部/全局注意力的mean loss与mean AA最佳，但full-mask AA最佳为R21；R26优势不能概括为所有掩码条件都更强。R23可学习尺度与R24固定尺度最终loss分别1.845777/1.845360，当前没有可学习尺度优于固定尺度的证据。

全部方法end准确率仍为0。全mask下结束标记只有132/17996个监督位置（约0.73%），总loss下降并不代表结束位置已学会；该项保留为学习告警，不算程序失败。后续长步数要分别看有条件/无条件组的end与AA，不允许只凭总loss晋级。无条件full-mask输入隐藏样本代码与真实长度，不能要求所有指标同时100%。

人工目标主要重复长度8的可见代码/角色motif，局部注意力与这项规则天然相关；heldout精确序列不重复也不等于生物泛化证据。仍需真实训练/下游/few-shot清理证书和独立评测。

## 显存测量

A100-SXM4-80GB；实测长度2048；FP32参数与Adam状态、BF16 autocast、整层activation checkpointing；每档3次优化步。旧R12–R19推荐micro16，新8项均micro32。

| 方法 | 推荐micro | profile峰值 GiB | 预期OOM档 | 人工诊断峰值 GiB |
| --- | --- | --- | --- | --- |
| r20_refined_control | 32 | 41.995 | 64 | 5.413 |
| r21_refined_head_gate | 32 | 42.184 | 64 | 5.470 |
| r22_refined_fixed_gate | 32 | 41.995 | 64 | 5.413 |
| r23_refined_layerscale | 32 | 41.996 | 64 | 5.413 |
| r24_refined_fixed_scale | 32 | 41.995 | 64 | 5.413 |
| r25_refined_late_fusion | 32 | 48.120 | 64 | 5.967 |
| r26_refined_local_global | 32 | 53.968 | 64 | 6.453 |
| r27_refined_value_residual | 32 | 42.183 | 64 | 5.469 |

旧R12与新R20同micro8峰值从20.457降至11.196 GiB，约减少45.3%；80项CPU等价性测试和GPU显存扫档分别提供行为等价与资源改善证据。人工诊断输入短于2048，因此运行峰值不能替代最坏长度profile。

## 各档mask终态

| 方法 | mask | loss | AA acc | end acc | AA count | end count |
| --- | --- | --- | --- | --- | --- | --- |
| r20_refined_control | mask25 | 1.874350 | 31.26% | 0.00% | 4428 | 39 |
| r20_refined_control | mask50 | 1.870665 | 31.41% | 0.00% | 8819 | 69 |
| r20_refined_control | mask75 | 1.865973 | 31.00% | 0.00% | 13330 | 107 |
| r20_refined_control | mask100 | 2.020301 | 26.56% | 0.00% | 17864 | 132 |
| r21_refined_head_gate | mask25 | 1.792641 | 32.72% | 0.00% | 4428 | 39 |
| r21_refined_head_gate | mask50 | 1.802892 | 32.74% | 0.00% | 8819 | 69 |
| r21_refined_head_gate | mask75 | 1.815707 | 32.75% | 0.00% | 13330 | 107 |
| r21_refined_head_gate | mask100 | 1.957686 | 31.72% | 0.00% | 17864 | 132 |
| r22_refined_fixed_gate | mask25 | 1.799174 | 32.84% | 0.00% | 4428 | 39 |
| r22_refined_fixed_gate | mask50 | 1.805964 | 32.79% | 0.00% | 8819 | 69 |
| r22_refined_fixed_gate | mask75 | 1.816477 | 33.06% | 0.00% | 13330 | 107 |
| r22_refined_fixed_gate | mask100 | 1.980707 | 30.56% | 0.00% | 17864 | 132 |
| r23_refined_layerscale | mask25 | 1.796758 | 32.00% | 0.00% | 4428 | 39 |
| r23_refined_layerscale | mask50 | 1.800183 | 32.28% | 0.00% | 8819 | 69 |
| r23_refined_layerscale | mask75 | 1.810449 | 32.69% | 0.00% | 13330 | 107 |
| r23_refined_layerscale | mask100 | 1.975718 | 31.17% | 0.00% | 17864 | 132 |
| r24_refined_fixed_scale | mask25 | 1.796337 | 32.07% | 0.00% | 4428 | 39 |
| r24_refined_fixed_scale | mask50 | 1.799615 | 32.46% | 0.00% | 8819 | 69 |
| r24_refined_fixed_scale | mask75 | 1.809966 | 32.65% | 0.00% | 13330 | 107 |
| r24_refined_fixed_scale | mask100 | 1.975525 | 31.24% | 0.00% | 17864 | 132 |
| r25_refined_late_fusion | mask25 | 1.805553 | 31.50% | 0.00% | 4428 | 39 |
| r25_refined_late_fusion | mask50 | 1.813761 | 31.77% | 0.00% | 8819 | 69 |
| r25_refined_late_fusion | mask75 | 1.827042 | 31.75% | 0.00% | 13330 | 107 |
| r25_refined_late_fusion | mask100 | 2.148573 | 29.52% | 0.00% | 17864 | 132 |
| r26_refined_local_global | mask25 | 1.605613 | 39.79% | 0.00% | 4428 | 39 |
| r26_refined_local_global | mask50 | 1.651628 | 38.55% | 0.00% | 8819 | 69 |
| r26_refined_local_global | mask75 | 1.731497 | 37.44% | 0.00% | 13330 | 107 |
| r26_refined_local_global | mask100 | 2.084567 | 29.73% | 0.00% | 17864 | 132 |
| r27_refined_value_residual | mask25 | 1.764883 | 34.26% | 0.00% | 4428 | 39 |
| r27_refined_value_residual | mask50 | 1.764933 | 34.28% | 0.00% | 8819 | 69 |
| r27_refined_value_residual | mask75 | 1.781155 | 33.53% | 0.00% | 13330 | 107 |
| r27_refined_value_residual | mask100 | 1.965203 | 28.69% | 0.00% | 17864 | 132 |

## 任务组终态

下表把各组的三个长度按每档对应token数合并，再平均四档，不能与直接平均三个长度的数值混用。全部组的end均为0。完整18组每档计数和initial/latest/train_probe保存在监督MD与原summary。

| 方法 | 任务组 | mean loss | mean AA | Full-mask AA |
| --- | --- | --- | --- | --- |
| r20_refined_control | ab_antigen | 1.856463 | 31.77% | 25.80% |
| r20_refined_control | tcr_peptide | 1.980945 | 25.22% | 22.97% |
| r20_refined_control | tcr_pmhc | 1.715954 | 33.28% | 32.85% |
| r20_refined_control | unconditional_ab | 2.016422 | 31.40% | 31.83% |
| r20_refined_control | unconditional_single | 1.931423 | 27.26% | 17.12% |
| r20_refined_control | unconditional_tcr | 1.959735 | 29.87% | 24.08% |
| r21_refined_head_gate | ab_antigen | 1.846109 | 31.92% | 29.19% |
| r21_refined_head_gate | tcr_peptide | 1.925150 | 27.26% | 26.79% |
| r21_refined_head_gate | tcr_pmhc | 1.583155 | 39.87% | 39.75% |
| r21_refined_head_gate | unconditional_ab | 1.975124 | 31.25% | 31.83% |
| r21_refined_head_gate | unconditional_single | 1.866614 | 30.32% | 31.28% |
| r21_refined_head_gate | unconditional_tcr | 1.872331 | 33.03% | 31.25% |
| r22_refined_fixed_gate | ab_antigen | 1.845855 | 33.49% | 31.34% |
| r22_refined_fixed_gate | tcr_peptide | 1.951979 | 27.10% | 26.29% |
| r22_refined_fixed_gate | tcr_pmhc | 1.583832 | 39.03% | 38.89% |
| r22_refined_fixed_gate | unconditional_ab | 1.973130 | 31.77% | 31.83% |
| r22_refined_fixed_gate | unconditional_single | 1.901320 | 31.97% | 31.28% |
| r22_refined_fixed_gate | unconditional_tcr | 1.876333 | 30.15% | 24.08% |
| r23_refined_layerscale | ab_antigen | 1.855807 | 31.77% | 31.53% |
| r23_refined_layerscale | tcr_peptide | 1.945049 | 25.07% | 22.20% |
| r23_refined_layerscale | tcr_pmhc | 1.600089 | 38.31% | 38.98% |
| r23_refined_layerscale | unconditional_ab | 1.965806 | 32.62% | 31.83% |
| r23_refined_layerscale | unconditional_single | 1.860899 | 33.38% | 31.28% |
| r23_refined_layerscale | unconditional_tcr | 1.858073 | 31.48% | 31.25% |
| r24_refined_fixed_scale | ab_antigen | 1.856027 | 31.92% | 31.59% |
| r24_refined_fixed_scale | tcr_peptide | 1.943919 | 25.06% | 22.20% |
| r24_refined_fixed_scale | tcr_pmhc | 1.599035 | 38.48% | 39.29% |
| r24_refined_fixed_scale | unconditional_ab | 1.965769 | 32.64% | 31.83% |
| r24_refined_fixed_scale | unconditional_single | 1.860166 | 33.36% | 31.28% |
| r24_refined_fixed_scale | unconditional_tcr | 1.858131 | 31.55% | 31.25% |
| r25_refined_late_fusion | ab_antigen | 1.918968 | 28.64% | 19.89% |
| r25_refined_late_fusion | tcr_peptide | 2.045583 | 26.48% | 27.00% |
| r25_refined_late_fusion | tcr_pmhc | 1.676084 | 36.84% | 36.76% |
| r25_refined_late_fusion | unconditional_ab | 1.979783 | 32.14% | 31.83% |
| r25_refined_late_fusion | unconditional_single | 1.900857 | 31.02% | 31.28% |
| r25_refined_late_fusion | unconditional_tcr | 1.876591 | 31.45% | 31.25% |
| r26_refined_local_global | ab_antigen | 1.758048 | 36.27% | 27.49% |
| r26_refined_local_global | tcr_peptide | 1.888039 | 32.21% | 27.03% |
| r26_refined_local_global | tcr_pmhc | 1.581607 | 42.17% | 30.26% |
| r26_refined_local_global | unconditional_ab | 1.843993 | 34.69% | 31.83% |
| r26_refined_local_global | unconditional_single | 1.779811 | 38.13% | 31.28% |
| r26_refined_local_global | unconditional_tcr | 1.767922 | 35.51% | 31.25% |
| r27_refined_value_residual | ab_antigen | 1.809345 | 33.36% | 29.40% |
| r27_refined_value_residual | tcr_peptide | 1.920383 | 27.68% | 24.69% |
| r27_refined_value_residual | tcr_pmhc | 1.605053 | 36.12% | 32.94% |
| r27_refined_value_residual | unconditional_ab | 1.920074 | 32.75% | 30.79% |
| r27_refined_value_residual | unconditional_single | 1.830834 | 34.62% | 31.28% |
| r27_refined_value_residual | unconditional_tcr | 1.838486 | 32.43% | 24.32% |

后续若同release从头进行500步人工诊断，按本次25步速度线性估算约4小时；实际受评测和吞吐影响。25步不保存checkpoint，不能把500步重新初始化称为续训。

完整阶段证据：[r20_r27_profile25_summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r20_r27_profile25_summary.json)；逐job动态记录：[IMMUNE_RESEARCH_SUPERVISION.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md)。后台daemon继续运行，不取消平台或自动入队。
