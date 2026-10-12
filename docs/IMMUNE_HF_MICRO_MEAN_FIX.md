# 新 Trainer 的完整梯度累积窗口修正

2026-10-01。实际 CPU 复现发现，尚未部署的 `TokenValidationTrainer` 在完整梯度累积窗口中会将各 microbatch 均值相加，未按预期取平均。已对该独立原型作最小修正，两套环境验收通过；**当前 R20–R27 的 39 份训练源码及 GPU 500 步任务未修改**。

## 原因与影响范围

`ResearchModel.forward(token_t=None, **batch)` 的可变关键字签名使两版 Hugging Face Trainer 将 `model_accepts_loss_kwargs` 判为 true；但 `ImmuneTrainer.compute_loss` 只调用 `model(**inputs)`，返回 microbatch loss 均值，没有使用框架提供的窗口 `num_items_in_batch`。在当前没有自定义 `compute_loss_func` 的配置下，错误的 true 标记让框架跳过梯度累积除法。

修正位于 [validation_metrics.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/validation_metrics.py)：`TokenValidationTrainer.__init__` 在父类初始化后明确设置 `model_accepts_loss_kwargs=False`。模型前向、microbatch 内 CE、corruption、sampler 和参数架构均未改变；训练 factory 与后续人工 GPU producer 使用同一个 Trainer 类。没有运行时 monkeypatch，也没有将新 record-mean 目标混入此修正。

当前500步走的是 `run_immune_refined.expanded_diagnostic` 的手动循环，其中已经显式计算 `loss=result['loss']/accumulation`，不受该 HF 接入错误影响。此前小型 `compute_loss(...).backward()` 等价测试绕过了真实 `Trainer.training_step`，因此不能发现这一问题。不能把该发现追溯解释为所有旧模型或所有既有训练结果失效。

## 实际复现与修正验收

[hf_micro_mean_audit.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/hf_micro_mean_audit.py)使用固定8条人工数值记录、`Linear(3,2)`、MSE、SGD和完整单次更新窗口。人工模型与 `ResearchModel.forward` 的参数名称及参数种类一致；使用真实 `TokenValidationTrainer.train`、原生 `Trainer.training_step`、原生自然采样器和 `Accelerator.backward`，没有构造真实 backbone、读取序列或使用 GPU。

独立 NumPy 闭式公式计算完整 batch 的 weight/bias 梯度；等大 microbatch 下，它也等于各 microbatch 均值的平均。`on_pre_optimizer_step` 比较全部实际梯度，更新后再比较全部模型参数。关闭梯度裁剪、weight decay、混合精度和额外 loss callback，框架 Accelerator GA 实测为1；记录真实消费的每个 record ID，确认无重复或遗漏。

| 配置 | 修正前：实际梯度/预期均值 | 修正后：实际梯度/预期均值 |
|---|---:|---:|
| micro8 × GA1 | 1 | 1 |
| micro4 × GA2 | 2 | 1 |
| micro2 × GA4 | 4 | 1 |
| micro4 × GA2，显式False控制 | 1 | 1 |
| micro2 × GA4，显式False控制 | 1 | 1 |

pllm（Transformers 4.57.6）和 worker 对应环境（4.48.1）的结论一致，各执行修正前5项和修正后5项真实单步实验。修正后相对于预期均值的最大梯度绝对误差为 `1.4901161193847656e-08`，更新后参数最大差为0。这里的2/4倍描述**梯度及本次SGD测试**，不能据此声称 AdamW 的参数更新也必然放大2/4倍。

修正后原评估/pilot回归分组执行，各环境共90项通过：token+pilot 76项（8.98/10.64秒）及 endpoint 14项（6.58/7.08秒）。没有重复运行已通过的76项。旧7份原型/测试源码精确字节及修正前两份报告保存在 [before_fix_manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/before_fix_manifest.json) 与相邻 `before_fix_source/`；历史验收JSON不覆写为新SHA。

证据：[pllm修正前](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/pllm.json)、[worker修正前](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/worker.json)、[pllm修正后](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/pllm_after_fix.json)、[worker修正后](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/worker_after_fix.json)。新报告额外记录 ResearchTrainer、ImmuneTrainer 和 Accelerator 的来源文件SHA；worker解释器路径的 symlink 解析后可能显示 `flow`，其实际包版本单独记录。

## 仍待验证

此验收只覆盖完整窗口、CPU FP32、人工Linear/SGD和 `compute_loss_func=None`。HF 4.48.1 对尾窗按配置GA除法，4.57.6按当前实际窗口长度除法；此修正不证明不满窗口、跨epoch或恢复时两版行为一致。真实 backbone、BF16/AdamW、实际CUDA梯度、完整训练/评估显存及续训仍需后续独立验收。当前未部署、未签发GPU release或数据证书，未启动真实1000/5000步训练。

未来的完整GPU验收必须使用修正后的同一来源版本，实际观察窗口、optimizer状态、四档评估计数和RNG。旧CPU评估或旧显存profile不能代替新训练/评估的GPU证据；[producer方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_GPU_RELEASE_PRODUCER_PLAN.md)记录具体缺口。
