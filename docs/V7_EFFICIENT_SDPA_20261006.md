# V7 高效 SDPA 接入

更新：2026-10-06T12:01:37.551Z

## 实现与使用

工作区 RefinedModel 和 FactorialModel（包括 f011）默认 IMMUNE_SDPA_BACKEND=auto。ArchitectureBlock 内每次 attention 调用使用局部 sdpa_kernel([EFFICIENT_ATTENTION, MATH])，CUDA 支持时由 PyTorch 调度高效后端，不支持时允许 math；CPU 使用 math。上下文位于 block 内，非重入 checkpoint 重计算也执行相同策略，退出后恢复调用者全局 flags。旧 ArchitectureModel 未配置此策略时保持原行为。没有更改参数名、state_dict、mask、loss 或链偏置梯度路径。

- auto：默认，CUDA efficient + math 回退。允许回退不代表所有形状都获得加速，也不吞掉 OOM 或其他运行异常。
- math：设置 export IMMUNE_SDPA_BACKEND=math，使用原数学后端作对照。
- efficient：设置 export IMMUNE_SDPA_BACKEND=efficient，CUDA 上强制高效后端，不支持时报错；适合启动验收。CPU 仍用 math。

此开关在模型构造时读取，所有 rank 及恢复运行须一致设置。它是运行环境策略，不写进模型权重；若要沿用原 math 数值路径，须显式设 math。日志记录策略而非假称实际 kernel；实际 kernel 在 GPU profiler 测试中验证。

## 环境

实测 NVIDIA A100-SXM4-80GB，PyTorch 2.8.0、torch.version.cuda=12.9，原生 BF16 可用，解释器 /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python。使用 PyTorch 内置 SDPA，不要求额外安装 flash-attn 或 xformers。其他 GPU/版本须重新验证 kernel、BF16、可训练 attention bias、实际长度与 checkpoint/FSDP；未宣称其他版本的最低支持矩阵。驱动需兼容所用 PyTorch CUDA 构建，不要求用户另行编译 CUDA 扩展。

官方接口：https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html

## 验证和数值差异

最终 validation.log：98 passed、16 deselected，22.78 秒。涵盖原 refined/factorial CPU 回归、保存重载（临时权重自动删除），新 CPU 策略/非法配置检查，以及 r20、r26、f011 的 FP32/BF16 CUDA 全模型小尺寸测试。每种 CUDA 模型比较 math/auto/efficient，包含非重入 whole-layer checkpoint；profiler 确认 efficient 前向和反向，全部参数梯度有限、chain_bias 非零、全局后端 flags 恢复。未停止或重启已有任务，此轮未做吞吐测量。

首轮 BF16 r20 对照的 chain_bias 梯度相对 L2 差异约 3.27%，超过预设 3% 界限（最大绝对差约 3e-4）。补充 FP32 严格检验后通过：FP32 每张量相对 L2<1e-4 或最大绝对差<2e-7；BF16 界限记录为 5% 或 2e-5，loss atol/rtol 2e-3。这个检查说明小尺寸测试下计算接近，不是逐位等价或完整训练收敛等价证据。不能将此前 where-only 的梯度等价数值移用到 efficient 后端。

已有完整 75.6M 原生 V7 单卡计时中，where-only 0.5450 秒/更新，where+efficient 0.3103 秒/更新，额外约 1.76 倍；相对最初 6.6288 秒约 21.36 倍。该历史实验不包含 CPU 取数、多卡通信、验证与保存，本次未重复计时，不承诺八卡整体倍率。

## 发布边界

本次修改工作区模型代码；原运行任务的冻结源码未改，既有 where-only FSDP YAML、adapter 和 manifest 未改。因此旧任务和该封版包不会因设置环境变量就自动纳入本实现。用于新的正式任务时须将这三份模型源码重新纳入绑定快照，并完成真实多卡验收。未提交、取消或重启训练，未产生保留的测试 checkpoint。

代码：
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_next/model.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/model.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/factorial.py
- /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_refined_sdpa.py

验证日志：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_efficient_sdpa_20261006/validation.log
