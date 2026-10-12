# Attention 后端软件审阅：保持可学习 bias 与 mask 语义

更新：2026-09-30 UTC。范围为模型数学代码、PyTorch 官方实现、已有合成 profile 元数据和随机 CPU 张量。本轮没有读取生物序列、数据集或 benchmark，没有提交或执行 GPU kernel，没有修改当前 39 个冻结源码文件。当前 500 步实验保持原 release；以下候选需要下一个独立 release 验证。

**结论：当前配置很可能强制使用 CUDA math attention。最先值得验证的是开启 PyTorch memory-efficient SDPA，保持现有 bias 输入；FlexAttention 是另一条候选，但当前 CPU 原型已暴露编译复用限制，不能直接部署。** 当前 GPU profile 没有记录实际 kernel，不能将上述源码推断写成已测 GPU 结论，也不能预报加速比例。

## 1. 已确认的本地代码路径

1. [LLaDA 构造函数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py:1130) 全局启用 Flash SDPA，同时调用 `enable_mem_efficient_sdp(False)`。这是进程级设置；仅在构造模型前启用 efficient 会被覆盖。历史注释认为该实现慢，不能代表当前 A100、PyTorch 2.8、可学习 bias 组合的实测结论。
2. [attention 调用](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py:633) 只有 `attn_mask is None` 才调用外部 `flash_attn_func`；非空 bias 进入 `F.scaled_dot_product_attention`，使用非因果语义。安装外部 FlashAttention 本身不会绕过此条件。
3. [bias 转换](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py:619) 按 autocast dtype 转换 bias，必要时把负无穷调整为有限最小值。逐层调用可能重复建立转换张量；是否保留、峰值多少需 profiler。后续优化不能对共享 autograd 张量做破坏性原地修改，也不能把转换置于 `no_grad`。
4. 可学习关系表展开为 `[B,H,L,L]` bias；global、链内、local 的可见性与 padding 合并。参数即使初始化为零，也仍需梯度，不能为了触发融合路径把 bias 替换成 `None` 或 `detach()`。

## 2. PyTorch 2.8 的后端约束：源码证据与推断

本地 CPU 验证环境是 `torch==2.8.0+cu126`；已有 GPU measurement 没有记录其运行时 PyTorch/CUDA 版本，下一次必须补记。官方源码按 `v2.8.0` 固定，下载来源和 SHA 见 [source manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_attention_backend_review_20260930/sources/manifest.json)。

| 路径 | 2.8 源码中的相关条件 | 对当前可学习非空 bias 的含义 |
|---|---|---|
| CUDA Flash SDPA | `check_for_attn_mask` 拒绝非空 `attn_mask` | 不可选；不能只看 flash enabled flag |
| CUDA cuDNN SDPA | `check_attn_mask_shape` 拒绝 `attn_mask.requires_grad` | 训练 bias 时不可选；固定 bias 是另一个问题 |
| CUDA memory-efficient SDPA | 有 additive bias backward，返回 `grad_bias`；仍有 dtype、stride、硬件等条件 | 原理上可保留 bias 梯度，但被本地构造函数禁用 |
| math SDPA | 支持通用浮点 bias 和梯度 | 在上述条件与 math enabled 成立时应被选择 |

Flash/cuDNN 条件来自 [公共选择器](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h) 与 [CUDA 调度源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp)。efficient 的 bias 梯度来自 [CUDA backward 实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/cuda/attention_backward.cu)：`bias_requires_grad` 控制梯度分配，CUDA 实现要求 bias 最后维 stride 为 1，梯度缓冲有对齐处理。支持梯度不等于特定布局一定能运行或一定更快。

**推断而非 GPU 实测：**若 GPU 容器使用该版本和这些默认训练条件，当前可学习 dense bias 应回落到 math。math 对半精度输入保留 float 中间结果，融合实现可能有不同数值误差；这些是潜在内存和耗时来源。[SDPA 官方说明](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)

## 3. 已有合成 GPU profile 能证明什么

下表都是已有的三步 profile，不是本轮新增 GPU 试验。A100-SXM4-80GB，长度 2048，8 层、768 hidden、12 heads、3072 MLP，FP32 参数与 Adam、BF16 autocast。全部原始路径、摘要 SHA 和精确数字保存在 [profile 摘录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_attention_backend_review_20260930/existing_synthetic_profiles.json)。

| variant | bs8 峰值 GiB | bs8 三步秒数 | bs16 峰值 GiB | bs16 三步秒数 |
|---|---:|---:|---:|---:|
| R00 control | 11.196 | 28.224 | 21.462 | 55.948 |
| R12 architecture control | 20.457 | 28.856 | 40.021 | 55.780 |
| R20 refined control | 11.196 | 27.901 | 21.462 | 54.100 |
| R21 head gate | 11.243 | 28.363 | 21.557 | 55.063 |
| R25 late fusion | 12.727 | 27.658 | 24.525 | 54.958 |
| R26 local/global | 14.215 | 27.830 | 27.466 | 55.026 |
| R27 value residual | 11.242 | 28.022 | 21.556 | 54.514 |

R20 的 bs8 峰值比 R12 低约 9.26 GiB，表明先前 bias 复用修正有明显内存收益。这些记录没有实际 backend/kernel 名，也没有受控的后端切换；约 28 秒的相近总时间不能证明 attention 已加速或没有优化空间。

[profile_candidate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py:158) 在三次 forward/backward/clip/optimizer step 外计时，结束时 synchronize，未单列 warmup，也没有分解 attention 时间。数字包含初始化影响、首次 optimizer 状态分配与其他层成本，不是稳态单步 attention latency。不同 release 的 loss/RNG 轨迹也不能被这些数字证明等价。

## 4. 保留数学行为的候选与风险

### A. 优先测试 memory-efficient SDPA

在新 release 中用 `sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)` 包住实际 attention 调用，先强制执行以暴露不支持的输入；通过后再设计包含 math 的回退策略。上下文必须在模型构造后生效，退出会恢复设置，避免向其他模型传播全局改动。[官方上下文管理器](https://docs.pytorch.org/docs/2.8/generated/torch.nn.attention.sdpa_kernel.html)

保留相同的 Q/K/V、加性 bias、梯度、scale、非因果配置和 dropout。未来 GPU probe 应记录 `can_use_efficient_attention(..., debug=True)` 诊断与实际 kernel。[后端诊断 API](https://docs.pytorch.org/docs/2.8/backends.html)

这是改动最少的候选，但 dense bias 和它的梯度仍占二次方空间。以 bs8、12 heads、长度 2048 计算，单份 `[B,H,L,L]` 为 BF16 0.75 GiB 或 FP32 1.50 GiB；这里只是张量大小计算，不是预测实际峰值。不要把 efficient 名称当作一定更省显存或更快的证据。

### B. FlexAttention：通用 dense bias 与小参数表两种实现

`score_mod` 可以在每个 attention score 上加 bias；对任意 dense bias，可捕获该张量并索引。对当前两类别、逐 head 的小参数表，可按整数 group metadata 选择参数，避免先展开完整可学习表。结构 mask 可以先保持 dense boolean 索引；后续单独研究 `mask_mod`/BlockMask 的构建成本与稀疏收益。[FlexAttention 官方接口](https://docs.pytorch.org/docs/2.8/nn.attention.flex_attention.html)

**2.8 已支持 score_mod 捕获张量的梯度。**[高阶算子源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/_higher_order_ops/flex_attention.py) 回传 `grad_score_mod_captured`；要求梯度的 `mask_mod` 捕获值仍不被支持。不能沿用较早版本“所有可学习 Flex bias 都不支持”的结论。CPU 随机实验确认了 dense 参数和小参数表的梯度；boolean group/valid metadata 不参与梯度。

风险是 prototype API、编译时间、形状变化、参数梯度累加、block mask 成本及 CUDA kernel 的版本限制。**本次已经出现实际限制：不同 score_mod 闭包共用 eager Dynamo 编译状态时，部分用例报 `aten._local_scalar_dense`，首轮 10 failed / 28 passed。**本轮对每个独立数学用例 reset Dynamo 后通过；这只是测试隔离，不是热路径解决方案。初始失败日志保留在 [initial_shared_dynamo_pytest.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_attention_backend_review_20260930/initial_shared_dynamo_pytest.log)。新 GPU release 必须验证可复用闭包、动态/静态 shape 策略和编译缓存，不能把本原型直接接入训练。

此外，本次 CPU 路径是 [Python eager 高阶实现](https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/torch/nn/attention/flex_attention.py)，没有执行 CUDA/Triton 编译 kernel；因此没有证据声称其 GPU 显存或速度改善。

### C. 按 query 行分块的 math attention

每个 query 的 softmax 只沿 key 轴计算，故沿 query 轴分块并完整保留 K/V 与对应 bias 行，在当前 dropout=0 时保持相同数学运算。可对每个块使用 non-reentrant checkpoint，减少保存的中间结果。该路径对任意可学习 bias 适用，输出及 Q/K/V/bias 梯度已做 CPU 对照。

单纯分块后拼接，autograd 仍可能保留所有块的中间张量；本原型也仍传入完整 bias，因此不能声称消除了二次方内存。checkpoint 增加重计算，更多小 kernel 可能变慢。这是通用回退候选，不优先作为提速方案。若将来 dropout 非零，需要重新验证随机状态及重计算；相同分布不代表相同随机轨迹。

### 不属于等价替换的做法

删除或冻结可学习 bias、把全部 mask 改成 causal、让 padding key 可见、合并 prefix 与有效 group、把不同 group 直接拼成无 mask Flash 输入，都会改变现有行为。将 `finfo.min` 替换为 `-inf` 也必须检查空行和数值边界；当前有效 query 不可访问 padding key，padding query 的 fallback key 规则必须保留。当前纯链内 attention 中某些逐行常数 bias 的梯度数学上可以为零，不能用“所有参数必须非零梯度”作为通用验收条件。

## 5. 本轮 CPU 验证与产物

原型：[attention_backends.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/attention_backends.py)。测试：[test_attention_backends_cpu.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_attention_backends_cpu.py)。均不导入项目模型、tokenizer、数据或 benchmark；只构造随机 float32 CPU 张量和整数 mask metadata。

36 个等价用例 = 3 方法 × 2 bias 参数形式 × 3 mask 类型 × 有/无 padding；随机 seed 8421，`[B,H,L,D]=[2,3,17,8]`，分块长度 5，包含非整除尾块。每例比较输出和 Q/K/V/原始 bias 参数的全部梯度，`atol=2e-6, rtol=1e-5`。另有 padding query fallback/有效 key 隔离及全无效输入拒绝测试。

| 方法 | 最大输出绝对差 | 全部 Q/K/V/bias 梯度中最大绝对差 |
|---|---:|---:|
| query chunk | 2.3842e-7 | 5.9605e-7 |
| query chunk + checkpoint | 2.3842e-7 | 5.9605e-7 |
| Flex eager，独立 Dynamo 状态 | 4.7684e-7 | 1.9968e-6 |

最终 **38 passed**，见 [pytest.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_attention_backend_review_20260930/pytest.log)；逐例数值、环境、原型 SHA 见 [cpu_equivalence.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_attention_backend_review_20260930/cpu_equivalence.json)。

另一次 CPU profiler 的相同形状实验观察到：固定 dense bias 选择 `aten::_scaled_dot_product_flash_attention_for_cpu`，可学习 dense bias 选择 `aten::_scaled_dot_product_attention_math`，后者存在 bias 梯度。**这是 CPU dispatch 证据，不能替代 CUDA profiler。**未运行 BF16 GPU、memory-efficient CUDA、Flex CUDA、模型级 checkpoint 双 view 或 optimizer 更新对照。

复现命令：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python scripts/architecture_prototypes/attention_backends.py
python -m pytest -q scripts/tests/immune_llada/test_attention_backends_cpu.py
```

## 6. 下一独立 release 的 GPU 验证要求

先做纯随机张量 probe，保持 math 对照、权重/输入/mask seed 和精度一致；不改变当前 500 步实验源码。第一轮以 MATH vs 强制 EFFICIENT 为主，Flex 编译复用问题修复后再加入。

- 实际记录容器 PyTorch/CUDA/cuDNN/GPU 版本、构造后 backend flags、bias 的 dtype/shape/stride/requires_grad、选择器诊断和实际 forward/backward kernel 名。
- 使用 head_dim 64、12 heads、多个 batch 与长度，覆盖当前长度 2048 及非整齐长度；global/chain/local、padding/no-padding、dense/table 参数与梯度全部覆盖。保留当前 mask 和有限最小值策略。
- 对照输出、Q/K/V/bias 梯度、参数单步更新；先用 FP32 找错误，再在生产 BF16 autocast 下预先制定误差容忍标准。检查梯度有限、padding 不泄漏、checkpoint 双 view 不共享错误状态。
- 与旧三步容量测试分开报告：至少 3–5 次 warmup 后测 20 次，用 CUDA event/synchronize 分别报告 forward、backward、optimizer 与总时间，记录 allocated/reserved 峰值；编译时间单列。不要通过增加占卡负载伪造利用率。
- 数学/梯度通过且 GPU 实测有收益后，冻结新的源码指纹和 profile，再做独立短训练对照。新的后端训练结果要标明 release，不能和旧轨迹混作逐步严格一致基线。

共享根 MD 由父会话同步；本轮只新增此独立审阅文档、未部署原型、CPU 测试及证据目录。
