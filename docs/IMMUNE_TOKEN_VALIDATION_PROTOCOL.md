# 真实训练前的评估计数原型（v2）

2026-09-30。状态：隔离原型，尚未接入运行中的 R20–R27 或真实训练入口。源码：[validation_metrics.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/validation_metrics.py)。不改变当前 39 个冻结文件，不读取真实序列或启动 GPU 任务。协议为 `immune.token_validation.v2.aa_end_relation_and_full_mask_first_terminal`；下述 v2 取代早期26项测试的 v1，旧验收记录保留为历史证据。

## 指标定义

沿用每条记录固定的 25/50/75/100% 遮蔽；模型只收到既有七项 denoise 输入，标签、AA/end 评分掩码不进入模型特征。token 指标只评分被选中且有效的监督位置，排除 padding、可见真值和非监督条件 token。

逐记录收集可加的 CE 总和、选中 token 数、AA/end/relation 三组 CE/计数及正确数，再聚合；避免先求每 batch 均值再平均造成批大小依赖。v3 的 layout191 可以监督 `<binding>`/`<nonbinding>` 前缀，故 v2 明确接受目标链槽位以外、chain_id<0 的这两类标签并单列 relation 指标；其他未分类监督 token 拒绝。三组 CE/计数之和必须等于总数。某组缺席时返回 `null`，同时报告计数，不当成零损失或正确率为零。CE 与正确率是两种指标。

每档 mask 同时报告：

- `source_macro`：每个来源内部先按 token 聚合，再对有定义的来源等权平均，附有效来源数。
- `token_micro`：跨来源合并分子与 token 计数，长记录及大来源权重更高。
- `mask_mean`：对上述两种口径的九项 token 指标分别做四档 mask 等权平均，附有效档位数。四档目标存在重叠，不把计数相加当成独立样本量。仅在 full-mask 有定义的三个首次 END 指标不加入这个平均。
- 顶层 `eval_loss` 等沿用 full-mask/source-macro 的含义和名称。**其来源内部计数方式已修正，与旧 HF batch 均值不保证数值相同；新旧评估协议不能混作同一条曲线。**

全 mask 单次预测和部分 mask 重建均不等于迭代生成质量。人工任务的多解、可识别性上限以及 END/AA 取舍仍须分开解释。

## END 召回与实际停止位置

旧 `terminal_accuracy` 只问真实 END 位置有没有预测 END，实质是 recall。即使在多个位置都预测 END，该值仍可能100%。v2 保留该名称以便辨认，同时报告以下指标与分母：

| 指标 | 定义及范围 |
|---|---|
| `terminal_precision` | 真实 END 上的正确 END /（正确 END + 选中 AA 位置误报 END）；仅覆盖选中的 AA/end 监督位置 |
| `terminal_false_positive_rate` | 选中 AA 位置误报 END / 选中 AA 数 |
| `first_terminal_accuracy` | 各目标链的第一个预测 END 恰在真实位置的链数 / 全部目标链数 |
| `terminal_missing_rate` | 所有有效目标槽位都未预测 END 的链数 / 全部目标链数 |
| `length_mae` | 有预测 END 的链上，首次 END 与真实 END 的槽位距离绝对值之和 / 有预测 END 的链数 |

后三项只在100% mask评分，25/50/75%时为 `null`、链计数为0，避免可见 END 泄露答案。首次 END 搜索覆盖该链**所有 attended target slots，包括未监督的容量尾部**，不能根据真值长度裁短搜索区间。每条评分链必须有且仅有一个真实 END；padding和非目标上下文排除，链间不混淆。未预测 END 时只计 missing，长度误差无定义，不能记成0。relation 前缀预测 END 不混入目标 AA 的误报数。

当前人工集三个长度的无条件输入分别相同；任意确定性单次预测的首次 END 总体准确率至多1/3，尽管逐真值位置的 END recall 可以100%。这是人工任务的信息限制，详见[可识别性审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_EXPANDED_IDENTIFIABILITY.md)。**运行中的500步没有保存完整预测，不能从现有 END recall 反推或补造首次 END 成绩。** 本原型的反例测试证明指标能够区分两者，不代表模型已经改善。

## Trainer 集成与已纠正问题

独立审阅发现初稿逐 source/mask 调 `Trainer.evaluate` 会反复触发局部回调，完整汇总没有统一提交给 callback；只保护 Torch RNG 也不足以保护 Python/NumPy 及最终日志回调。最终实现一次 memory tracking、逐 source/mask 的 `evaluation_loop`、一次完整日志与 `on_evaluate`，并在成功和异常路径恢复三类 RNG。

直接调用解包模型的 `denoise` 会绕过 Accelerate 的 forward AMP 包装，故显式使用 `accelerator.autocast()`。训练 `compute_loss` 完全调用父类；没有重写训练、保存或采样流程。CPU 实测训练 loss 与全部参数梯度逐值一致。

当前只接受单进程评估、非 persistent workers、非 legacy loop、无自定义 `compute_metrics`、无 FSDP/DeepSpeed；要求 `disable_tqdm=True`，避免默认进度回调混合多个 loader。HF 4.57.6 对直接传入 Dataset 的持久 loader 缓存可能跨来源复用，因此不允许该配置。分布式、CUDA BF16 与真实训练中的行为仍需新 release 实测，不由 CPU 通过推定。

## 固定验证子集

新 [selection 工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_VALIDATION_SELECTION.md) 提供按来源 seed42、最多256条的精确索引及 SHA。未来 trainer 用 `Subset(PreparedImmuneDataset(..., split='valid', source=source), source_local_indices)` 消费，并把清单 SHA 写入 run provenance。原型尚未接入真实训练，该清单不替代训练认证。

当前独立候选的 selection 共1288条；TRAIT/native 各仅4条、4个 family。等权 source macro 会给小来源相同权重，因此必须同时展示样本/family数量与 token micro，不能只按一个总数选模，也不静默移除小来源或拆 family 凑验证集。

## 验收

最终联合验收在两套环境均 **90项通过：本评估原型42项（28 token + 14 endpoint），独立 pilot 适配器48项**。本地 `pllm` 8.60秒，训练 worker 对应 Python 环境10.90秒。前者 Torch2.8.0+cu126 / Transformers4.57.6，后者 Torch2.8.0 / Transformers4.48.1；Accelerate均1.14.0。均为 CPU，未声称 GPU 已验证。

覆盖人工数组解析计数、padding/visible隔离、非法标签/分组拒绝、空指标、宏微差异、四mask平均、真实 tiny 模型 FP32/BF16、batch1/3与尾批一致性、训练梯度等价、完整回调恰好一次、三类RNG含异常恢复、memory tracker清理后重试、CPU autocast及persistent worker拒绝。新增反例覆盖提前 END、仅尾部 END、无 END、多链、非法终点 metadata，以及实际 collator 的 binding/nonbinding 前缀计分。首轮 v2 测试误将 HF `model_preparation_time` 计时字段纳入 batch 不变性比较，现仅排除非确定性时间字段；没有放宽指标数值容忍。初始失败日志保留，后续全部通过。五个自有临时目录共2,322,199 bytes已清理，没有保留测试checkpoint。

[token 测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_token_validation.py) · [END 测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_token_endpoints.py) · [最终验收 JSON、源码 SHA 和39文件冻结复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_token_validation_v2_20260930/final_acceptance.json) · [pilot适配器及实际未认证数据拒绝](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CERTIFIED_PILOT_ADAPTER.md)。

复现：激活 `pllm` 后运行 `USE_TF=0 python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_token_validation.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_token_endpoints.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_certified_pilot.py`。下一 release 才能把已审阅的评估与 selection 接入，并重新记录源码、profile、指标协议及数据证书。
