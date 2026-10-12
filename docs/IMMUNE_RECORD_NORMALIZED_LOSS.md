# 每记录归一化 token CE：独立数学原型

2026-09-30 UTC。**46 项人工 CPU 测试通过；没有接入当前训练目标、Trainer 或 39 个冻结源文件。** 本原型只处理整数类别标签、浮点 logits 和显式布尔 mask，不读取生物序列、prepared 数据或 benchmark，不运行 GPU，不创建 checkpoint。

## 目标与分母

依据 [噪声与目标诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_NOISE_OBJECTIVE_DIAGNOSTIC.md)，定义一个 accumulation 窗口的实际记录数为 N，第 i 条记录的被选 token 集合为 S_i，K_i=|S_i|，逐 token CE 为 ℓ_ij：

\[
c_i=\frac{1}{K_i}\sum_{j\in S_i}\ell_{ij},\qquad
L=\frac{1}{N}\sum_{i=1}^{N}c_i.
\]

一个 microbatch G 的反向贡献是 `sum(c_i for i in G)/N`。所有 microbatch 的贡献直接相加；不能再除 microbatch 数或 gradient accumulation steps。每个被选 token 的 CE 系数为 `1/(N*K_i)`，每条记录的总系数相同。这与按整个窗口全部被选 token 求均值是不同目标，也与先求各 microbatch 的 token 均值再平均不同。

N 必须是该窗口实际呈现的记录数。若容量是 8、实际尾窗口有 7 条，分母就是 7；同一个窗口的所有 microbatch 必须使用相同 N。重复呈现的记录按呈现次数计入；此 helper 不做去重，也不推断采样权重。它不改变 mask 分布、强制补选策略或某类 token 的权重，不能称为官方 LLaDA loss 或 ELBO。

## 接口与拒绝约定

[record_loss.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/record_loss.py) 只有两个无状态函数：

- `actual_global_record_count(microbatch_record_counts)`：相加整个窗口的实际 microbatch 记录数；拒绝空窗口、非整数、布尔值、零和负数。
- `record_normalized_ce(logits, labels, selected_mask, valid_mask, global_record_count=N)`：返回当前 microbatch 的标量反向贡献、仍保留计算图的每记录 CE、整数选中计数，以及本地/全局记录数。logits 为 `[B,L,C]`，labels 为 int64 `[B,L]`，两个 mask 为 bool `[B,L]`，设备必须相同。

无有效位置的记录、零 selected token 的记录、选中 padding、非法被选类别索引、非法分母和非有限被选 CE 均明确拒绝。没有 `clamp_min(1)`、自动补 token、跳过记录或隐式缩小分母。分母必须是严格正整数且不小于当前 microbatch 的记录数；局部函数无法检查尚未看到的完整窗口，调用方仍须确认每批大小、总数以及每条呈现仅贡献一次。

仅 gather 被选 logits/labels 后计算 CE；未选位置允许 `-100` 或其他占位标签，且对这些 logits 的梯度严格为零。该结论不保证任意上游模型中的 NaN 输入都安全。被选 logits 中 NaN、正无穷直接拒绝；负无穷可排除非目标类别，但目标 CE 必须有限。FP16/BF16 CE 提升为 FP32，FP32/FP64 保留原精度。先除各记录 token 数再累加、先除 N 再求和，避免有限均值的中间总和溢出；最终归约仍检查有限性。

## CPU 验证范围

[测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_record_loss.py) 使用 seed 固定的人工整数标签与 `Linear(5,7)`。七条记录有效长度为 `[1,3,11,8,4,10,6]`，选中数为 `[1,1,8,2,4,3,6]`。完整 batch oracle 独立逐记录执行 `log_softmax + gather + mean`；不调用被测 reducer。

FP32 与 FP64 分别验证完整 batch、七个单记录 microbatch、`2+2+2+1`、`1+6`、以及记录重排后的 `2+1+4`。每个 microbatch 只 padding 到自身最长记录。累计贡献与 oracle 的 loss，以及模型的所有参数（weight、bias）梯度均一致；绝对/相对容差分别为 FP32 `3e-7`、FP64 `2e-14`，不声称跨运算次序 bitwise 相同。另验证实际尾批分母、错误再次平均的反例、record mean 与 token mean 的区别、未选位置零梯度、低精度 CE 提升、负无穷类别排除、极大有限均值和非法输入拒绝。

最终 **46 passed，2.71 秒**，环境为 pllm。详见 [日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_20260930/pytest.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_20260930/pytest.xml) 和 [SHA/环境证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_20260930/validation.json)。本测试没有 optimizer step；数学归一化正确不代表新目标提高实验效果。

这里的分组不变性要求 logits、标签、mask 与记录呈现相同，且记录间前向独立。真实模型的随机遮蔽、dropout、不同 kernel 舍入或跨样本运算仍可能改变前向结果，不能由本测试推导整个训练轨迹对 microbatch 完全不变。

## 为什么暂不接入 HF/Accelerate

只读审阅当前已安装框架发现明确的待验证接口差异：

1. 本地 Transformers 4.57.6 的 [training_step](/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/lib/python3.10/site-packages/transformers/trainer.py:4060) 在相关分支把 loss 除以 `current_gradient_accumulation_steps`；[窗口构造](/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/lib/python3.10/site-packages/transformers/trainer.py:2621) 将该值设为实际窗口 batch 数。
2. Worker Transformers 4.48.1 的 [training_step](/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib/python3.12/site-packages/transformers/trainer.py:3709) 在对应分支除以固定 `args.gradient_accumulation_steps`。因此不能假设两版本的尾窗口缩放一致。
3. [本地 num_items_in_batch](/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/lib/python3.10/site-packages/transformers/trainer.py:5620) 和 [worker 同字段](/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib/python3.12/site-packages/transformers/trainer.py:5160) 按非 `-100` 标签计 token，不能充当实际记录数。`model_accepts_loss_kwargs` / `compute_loss_func` 会改变相关分支。
4. 两环境的 Accelerate 1.14.0 在非 DeepSpeed [backward](/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/lib/python3.10/site-packages/accelerate/accelerator.py:2838) 中也按自身 `gradient_accumulation_steps` 缩放。常规 Trainer 构造通常令该值为 1；仅看到两处除法不能直接断言发生了双重归一化，必须检查实际运行对象。
5. 分布式梯度平均与 `average_tokens_across_devices` 还涉及进程数缩放。当前原型只验证单进程显式反向累积，不支持以此证明 DDP/FSDP/DeepSpeed 接入正确。

安全接入还需要在独立 release 中，用真实 `training_step`、`accelerator.backward` 和实际 dataloader 窗口，分别验证完整窗口与尾窗口的所有参数梯度及一次 optimizer 更新；明确观测实际记录数的时机、全部缩放因子，以及是否发生重采样或随机状态变化。在两套框架环境完成该验证前，本 helper **不接入**现有 Trainer。

## 作为独立消融

后续候选只改变上述目标归一化，其余架构、初值、数据呈现顺序、随机 mask、microbatch、优化器、学习率日程和训练预算保持一致；比较现目标与每记录均值目标，另用改变 microbatch 的软件一致性检查单独验证后者。记录实际记录数、被选 token 数、各来源占比、原始未加权 CE，以及既有分来源与分 mask 评估。目标变化应获得独立方法标识与源码/协议绑定，不能混入 backend 或架构改动后统一归因，也不能凭这组 CPU 数学测试宣称收益。
