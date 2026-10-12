# R12–R19：100 步合成诊断终态

八项均完成 100 步，全部成功；源文件与 profile 证据一致，无执行失败、NaN 或源码错位。该结果验证工程与合成优化链路，不能据此敲定最佳免疫受体架构。

记录时间：2026-09-30T20:50:33.963006+00:00。

沿用既有 LLaDA270 的 d768/L8/H12/MLP3072 主干；蛋白词表减去了无关的大词表参数。R12–R19 为统一的 v2 噪声/评测协议，以 R12 为对照。所有方法 micro16 × accumulation16 = global256；本组 100 步即 25,600 次合成样本曝光，唯一 toy 记录仅六条。

## 终态比较

四档均值是25/50/75/100%各level非空指标的等权平均。

| 方法 | 四档 mean loss | 四档 AA acc | 四档 end acc | Full-mask loss | Full-mask AA acc | Full-mask end acc | Full-mask 联合正确数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r12_architecture_control | 0.249980 | 87.20% | 100.00% | 0.993626 | 48.78% | 100.00% | 51/93 |
| r13_head_output_gate | 0.293260 | 88.11% | 90.91% | 1.171010 | 52.44% | 63.64% | 50/93 |
| r14_fixed_half_attention | 0.304166 | 86.53% | 100.00% | 1.110675 | 47.56% | 100.00% | 50/93 |
| r15_layerscale | 0.225457 | 87.80% | 95.45% | 0.901605 | 51.22% | 81.82% | 51/93 |
| r16_fixed_residual_scale | 0.287032 | 87.20% | 100.00% | 1.146744 | 48.78% | 100.00% | 51/93 |
| r17_chain_late_fusion | 0.261464 | 87.80% | 95.45% | 1.023912 | 51.22% | 81.82% | 51/93 |
| r18_chain_local_global | 0.286785 | 88.72% | 80.30% | 0.990162 | 54.88% | 54.55% | 51/93 |
| r19_value_residual | 0.249806 | 88.11% | 93.18% | 0.990644 | 52.44% | 72.73% | 51/93 |

## 学习曲线

下表为四档 mean loss，保留相同步数，避免只选最优点。

| 方法 | step5 | step20 | step25 | step50 | step75 | step100 | 首次 mean loss <0.5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| r12_architecture_control | 3.8112 | 1.8529 | 1.7019 | 0.3630 | 0.2322 | 0.2500 | 50 |
| r13_head_output_gate | 2.7991 | 1.8477 | 1.6550 | 0.4058 | 0.2508 | 0.2933 | 45 |
| r14_fixed_half_attention | 3.5205 | 1.8518 | 1.6545 | 0.3214 | 0.2304 | 0.3042 | 45 |
| r15_layerscale | 3.3868 | 1.8071 | 1.5877 | 0.3072 | 0.2467 | 0.2255 | 45 |
| r16_fixed_residual_scale | 3.3836 | 1.8051 | 1.5846 | 0.2865 | 0.3201 | 0.2870 | 45 |
| r17_chain_late_fusion | 3.3609 | 1.6853 | 1.3467 | 0.4227 | 0.2507 | 0.2615 | 45 |
| r18_chain_local_global | 3.5196 | 1.4810 | 1.0094 | 0.3315 | 0.2830 | 0.2868 | 35 |
| r19_value_residual | 2.9211 | 1.7574 | 1.4322 | 0.3563 | 0.2242 | 0.2498 | 35 |

最低 mean loss 的方法从 step50 的 R16，到 step75 的 R19，再到 step100 的 R15，多次换位。R18 在早期收敛更快，但100步并非最低loss；短步数排序不是稳定的架构排序。

R15 的100步loss低于固定尺度对照 R16，但20/25步近乎一致、50步固定尺度更好，单seed/小toy不足以证明可学习尺度的泛化收益。R13与固定gate R14的最终loss/AA/end也存在取舍，不能忽略控制组或只看某一个数字。

## 哪些测试已饱和，下一步应验证什么

25/50/75%部分mask的AA在多项方法上已达到100%；四档mean-AA已接近可识别上限89.0244%。full-mask下四条TCR输入完全相同而标签不同，联合argmax最多正确51/93；多项最终已达到这一上限，其AA/end分配差异包含等价最优取舍。该条件下继续延长同六条toy主要测过拟合、概率校准与稳定性，不能提供更多真实泛化证据。

后续应在独立release中扩展链长、氨基酸组成、家族/条件链/多链组合和噪声分布，并保留相同协议的控制组；对较广数据观察稳定性与收敛时才考虑增加steps。最终晋级初步真实训练仍需训练数据/下游测试及few-shot范围的清理证书，合成成功不能替代数据门禁。

R18局部注意力的早期收益可能部分来自极短toy配256容量的MASK尾部，必须用较广长度分布检验。所有结果为单seed42、六条合成样本，未做下游评测。

可识别性证据与复算程序见 [监督发现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION_FINDINGS.md)；完整四档指标、计数、全部trace和源文件证据见 [阶段JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r12_r19_stage100_summary.json)。

## 实际运行与资源证据

| 方法 | job_id | 100步用时秒 | profile峰值GiB | synthetic峰值GiB | source root |
| --- | --- | --- | --- | --- | --- |
| r12_architecture_control | 1790799044322727901_r12_architecture_control_37b2b106 | 1835.91 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r13_head_output_gate | 1790799044331991663_r13_head_output_gate_72718882 | 1839.41 | 40.12 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r14_fixed_half_attention | 1790799044337931939_r14_fixed_half_attention_8643a4ca | 1835.59 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r15_layerscale | 1790799044341624231_r15_layerscale_d0115c63 | 1837.30 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r16_fixed_residual_scale | 1790799044345097609_r16_fixed_residual_scale_6111de03 | 1838.02 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r17_chain_late_fusion | 1790799044348827183_r17_chain_late_fusion_2a3a70e9 | 1835.36 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r18_chain_local_global | 1790799044352237695_r18_chain_local_global_a0a72b83 | 1837.33 | 40.02 | 4.01 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |
| r19_value_residual | 1790799044355869324_r19_value_residual_3bcfd14a | 1835.93 | 40.07 | 4.03 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r12_r19_20260930 |

GPU工作池未被本监督任务取消；后台daemon继续记录后续任务，平台与worker的实时状态见 [实时记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)。
