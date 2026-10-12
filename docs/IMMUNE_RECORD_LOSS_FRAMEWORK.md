# Record mean 的真实 Trainer/Accelerate CPU 验收

2026-09-30 UTC。**两套环境各 14 项通过。** 验证对象是新的人工 CPU [隔离适配器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/record_loss_framework.py)，没有改动当前训练 loss、模型、Trainer、已安装框架或 39 个冻结源文件，也没有读取任何生物序列、prepared 数据或 benchmark。

这次执行真实 `Trainer.train`、Trainer dataloader、`Trainer.compute_loss`、`Trainer.training_step` 和 `Accelerator.backward`；不是手动调用 helper 后直接反向的替代实验。测试断言 train/compute_loss/training_step/backward 的实际方法身份没有被覆盖或 mock。输入是固定人工整数标签和浮点特征，模型为 `Linear(5,7)`，无 dropout、随机 mask 或 GPU，使用真实 SGD 更新。

## 隔离适配器改了什么

1. 使用 Trainer 公开的 `compute_loss_func` 扩展点，计算 [每记录 selected-token CE 均值](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_NORMALIZED_LOSS.md)之和，再除整个 accumulation 窗口实际记录数。两版 Trainer 在有这个回调时均不执行其额外 GA 除法。回调是真正被原生 `Trainer.compute_loss` 调用，不是只设标记后绕开调用。
2. **规范化预取窗口上限**：重载 `get_batch_samples`，仍调用父类真实迭代器预取，但把上限统一为配置 GA，由真实 EOF 返回不足窗口。这也是适配改动。HF 4.48.1 的末次请求使用 `num_examples % GA`，不能保证等于实际 dataloader batch 数的余数；仅修 loss 分母还不够。
3. 根据真实 prefetched batches 的 batch 维取得实际记录数，以独立 `_record_window` 元数据传给人工模型及损失回调。模型只把元数据原样转递，不用它计算 logits。原生 `num_items_in_batch` 继续表示非 `-100` 标签 token 数；既不重定义这个字段，也不把 token 数作为 record 分母。
4. 每个窗口记录 microbatch 大小、record IDs、记录数、原生 token 数、配置 GA、实际 Accelerator GA、框架原始预取请求和适配后的上限；每个反向贡献检查窗口索引、消费顺序、IDs、记录数和 token 证据。

适配器严格限制为本次已测试的 Transformers 4.57.6 / 4.48.1、Accelerate 1.14.0、单进程 CPU、Accelerator GA=1、无 AMP/梯度裁剪/DS/FSDP/编译、零 dataloader workers、无 drop_last / label smoother / token 跨设备平均。人工模型不接受 loss kwargs。它是训练窗口原型；评估和真实模型不能直接复用这个人工输入/输出协议。

因此，通过的数值证明**上述隔离适配器**在测试范围内正确，不证明原生旧 HF 的预取窗口正确，也不表示现有 Trainer 已经修复。当前运行代码完全没有应用这些改动。

## 实际窗口与独立 oracle

每个环境都在 FP32、FP64 各运行下面七种情况。数据由固定 seed 构造，有不同记录长度和 selected-token 数；每个 microbatch padding 到自身最长记录。测试只用顺序 sampler 固定呈现顺序，dataloader 本身仍来自真实 Trainer / Accelerate。

| 记录数 | micro | 配置 GA | 实际每次更新的记录数 | 覆盖情况 |
|---:|---:|---:|---|---|
| 4 | 4 | 1 | 4 | 单一完整 batch |
| 5 | 3 | 1 | 3、2 | GA=1 的不满 microbatch |
| 6 | 2 | 3 | 6 | 完整 accumulation 窗口 |
| 9 | 2 | 3 | 6、3 | 最后仅两个 microbatch，大小 2、1 |
| 7 | 1 | 3 | 3、3、1 | 最后仅一个 microbatch |
| 7 | 2 | 5 | 7 | 整个 epoch 少于配置 GA |
| 7 | 7 | 1 | 7 | 与上一行相同七条记录的一次完整 batch |

特别是 `N=7, micro=2, GA=5`：本地 HF 原始请求 4 个 batch，旧 worker HF 原始请求 2 个；适配器预取上限均为 5，真实 EOF 后均得到 `[2,2,2,1]`。窗口实际记录数为 **7**，原生标签 token 数为 **25**。一次 optimizer 更新覆盖全部七条记录；测试还确认整次训练中 IDs 恰为 `0…N-1`，没有丢弃、重复或重排。这份证据保留两个计数的不同含义。

每个真实 `on_pre_optimizer_step` 回调都用独立模型，对当前窗口原始人工记录逐条执行 `log_softmax + gather + mean`，再求完整窗口记录均值。它比较真实累积后的 **weight 和 bias 全部梯度**、累计 loss，并更新自己的 SGD oracle。随后 `on_optimizer_step` 比较真实模型和 oracle 的全部参数。学习率为 0.03、常数 scheduler、无 momentum/weight decay/裁剪。每个环境共检查 22 次 optimizer 更新，而非只核对最后一次标量 loss。

## 数值结果与可复核证据

| 环境 | Python / Transformers | 测试 | 耗时 |
|---|---|---:|---:|
| pllm | 3.10.19 / 4.57.6 | 14 passed | 11.03 s |
| worker Python | 3.12.12 / 4.48.1 | 14 passed | 12.17 s |

两环境的最大误差相同：

| 精度 | 预先固定的绝对/相对容差 | 最大参数梯度差 | 最大更新后参数差 |
|---|---:|---:|---:|
| FP32 | 3e-7 | 8.94e-8 | 2.98e-8 |
| FP64 | 2e-14 | 8.33e-17 | 2.78e-17 |

没有为取得通过而放宽容差，也不声称浮点逐位相同。真实窗口、token/record 数、框架实际 GA、每一步误差和版本均保留在小型 JSON 中。

- [总体验收、SHA 与清理记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/validation.json)
- [pllm 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/pllm/pytest.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/pllm/pytest.xml)、[环境及框架方法 SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/pllm/environment.json)
- [worker 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/worker/pytest.log)、[JUnit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/worker/pytest.xml)、[环境及框架方法 SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/worker/environment.json)
- 短 epoch 的逐步证据：[pllm](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/pllm/case_n7_micro2_ga5_float64.json)、[worker](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_record_loss_framework_20260930/worker/case_n7_micro2_ga5_float64.json)
- [独立测试代码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_record_loss_framework.py)

复现只需激活 pllm 后运行文件 docstring 中的 pytest 命令；worker 使用该环境对应的 Python。没有重新运行先前 90 项测试，也没有保存 checkpoint。临时 pytest 输出目录已清理，仅保留上述日志/JSON/JUnit。

## 仍然不代表什么

本次没有验证真实 backbone、BF16/梯度 scaler、AdamW 状态、分布式归约、checkpoint 恢复、IterableDataset、随机采样器、跨 epoch 恢复或评估循环，也不证明该目标提升任何真实任务效果。预取元数据依赖明确的人工模型协议，不能把这个类直接替换现有 corpus Trainer。

若后续把目标作为独立消融接入，需要单独 release、真实模型输入协议适配、对应精度和优化器的窗口/续训验证，以及原有数据认证和实验准入条件。新旧目标对照组应使用相同的窗口读取规则，避免把修正旧框架尾窗口与改变目标混成一个因素。当前 loss 和实验仍保持原样；此目标不是官方 LLaDA/ELBO 的复现。
