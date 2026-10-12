# 遮蔽概率、END 权重与 microbatch 归一化的数学诊断

2026-09-30 UTC。本轮仅审阅采样与 loss 数学代码，使用人工整数数组和解析概率；不读取真实序列、数据集或 benchmark，不运行模型或 GPU，不修改当前 39 个冻结文件。

**当前全遮蔽概率约为 `1/(N+1)`，并非 `2^-N`。部分遮蔽恢复改善仍不能直接证明全遮蔽生成改善。另一个明确问题是当前训练 loss 随 microbatch 分组改变权重；本轮八项均为 micro32，比较口径一致，不因此推翻已有结果。**

## 1. 实际采样分布

[sample_bioseq_diffusion_noise](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py:310) 默认 `epsilon=0.001`、纯 mask 替换。每条记录独立采样一个 `t~Uniform(epsilon,1)`，该记录所有 eligible token 共用 t，再独立 Bernoulli(t) 遮蔽。如果一条有 eligible token 的记录一个都没选中，则均匀补选一个。[ResearchModel.corrupt](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py:154) 没有传入其他 t 分布；本轮 R20–R27 的 options 没有启用旧 chain/span mask 分支。

令 N 为记录中可遮蔽 token 总数、K 为其遮蔽数。补选前：

\[
P(K=k)=\frac{\binom Nk}{1-\epsilon}\int_\epsilon^1 t^k(1-t)^{N-k}\,dt
=\frac{1-I_\epsilon(k+1,N-k+1)}{(N+1)(1-\epsilon)}.
\]

因此 N>1 时：

\[
P(K=N)=\frac{1-\epsilon^{N+1}}{(N+1)(1-\epsilon)}.
\]

epsilon=0 时，K 在 0…N 上均匀；epsilon=0.001 仅略微修正。强制补选把 K=0 的质量移动到 K=1，不改变这里列出的全遮蔽或接近全遮蔽事件。`t` 很少恰好等于 1，不代表全遮蔽事件没有概率。

## 2. 长度 24/127/255 的单、双目标

为避免长度歧义，下表 **L 表示每个目标的内容 token 数，每个目标另有一个 END**；m 个同长度目标时 `N=m(L+1)`。全遮蔽指该记录所有目标同时全遮蔽，固定条件不计入 N。接近全遮蔽按实际 K/N 定义，取整会使短长度比例跳变。

| L | 目标数 m | N | 全遮蔽 | 最多剩 1 个可见 token | 遮蔽 ≥95% | 遮蔽 ≥90% | END 系数份额 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 24 | 1 | 25 | 3.8500% | 7.7000% | 7.7000% | 11.5500% | 4.00000% |
| 24 | 2 | 50 | 1.9627% | 3.9255% | 5.8882% | 11.7765% | 4.00000% |
| 127 | 1 | 128 | 0.7760% | 1.5519% | 5.4318% | 10.0876% | 0.78125% |
| 127 | 2 | 256 | 0.3895% | 0.7790% | 5.0634% | 10.1269% | 0.78125% |
| 255 | 1 | 256 | 0.3895% | 0.7790% | 5.0634% | 10.1269% | 0.390625% |
| 255 | 2 | 512 | 0.1951% | 0.3903% | 5.0733% | 10.1466% | 0.390625% |

若 24/127/255 已经包含 END，应直接令单目标 N=24/127/255；对应全遮蔽概率约为 4.0040%/0.7820%/0.3910%。两种长度约定的精确数字都保存在 [probabilities.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_noise_math_20260930/probabilities.json)。这些是人工长度条件下的概率，不代表真实语料长度分布。

双目标共用 t，因此不能把单目标概率相乘。每目标 n 个 eligible token 时：指定一个目标全遮蔽为 E[t^n]，两者同时全遮蔽为 E[t^(2n)]，至少一个全遮蔽为 `2E[t^n]-E[t^(2n)]`。例如 L=127，两者同时全遮蔽约 0.3895%，远高于错误的独立乘积约 0.0060%。

## 3. END 的相对份额，以及恢复与生成的区别

普通 selected-token 平均 CE 中，每个被选 token 系数相同。由于遮蔽和补选均具有 token 交换对称性，记录非空遮蔽条件下 END 系数的期望为 `m/N`；相同长度单、双目标均为 `1/(L+1)`。全遮蔽时这就是精确系数份额；随机遮蔽的单次实际份额会波动，且有时没有选中 END。

**系数份额不是实际 loss 份额或梯度范数份额。** END 可能更难，CE 或梯度可能更大。总 CE 下降也可能掩盖 END 退化：人工例子 L=127，内容 CE 从 2 降到 1、END CE 从 1 升到 10，总 CE 仍从 255/128 降到 137/128。应分别记录两类原始 CE、正确率和选中计数。

部分 mask 评估包含可见的真值 token；END 也经常可见。全 mask 则缺少这些信息，条件熵不同。迭代生成后续步骤还以模型自己的预测作条件，而重建评估保留的是干净真值。因此：

- 四档重建指标能诊断不同噪声条件下的预测能力；它不是生成质量的替代指标。
- 全 mask 单次 argmax 恢复率也不等于迭代生成质量，尤其一个条件允许多个合理输出时。
- 不能把全 mask 较差简单归因于“从未见过全 mask”；上述概率和实际训练抽样次数需要一起考虑。

## 4. microbatch 分组改变当前 loss：明确反例

[ResearchModel.forward](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py:200) 在一个 microbatch 内计算 `sum(masked CE)/sum(masked count)`；累积多个 microbatch 时再平均这些局部 loss。这与整个 global batch 的 token 平均通常不同。

固定四条人工记录，令逐记录平均 CE 为 `c=[1,3,1,3]`，masked count 为 `K=[1,9,1,9]`。不改变任何预测、mask 或记录，只改分组：

| microbatch 分组（记录从 1 编号） | 当前累积 loss | 对 c1,c2,c3,c4 的梯度系数 |
|---|---:|---|
| (1,2)、(3,4) | 2.8 | (0.05,0.45,0.05,0.45) |
| (1,3)、(2,4) | 2.0 | (0.25,0.25,0.25,0.25) |

全局 token 平均是 `(1+27+1+27)/20=2.8`。全局 per-record mean 是 `(1+3+1+3)/4=2.0`，无论上述哪种分组，其梯度系数都是四个 0.25。它们是两种不同且可明确定义的目标，不能混为一谈。

要实现 per-record 分组不变，应累积 `sum(c_i)/实际全局记录数`；只有 microbatch 大小相同时，才等价于简单平均各 microbatch 的记录均值。对不整尾批仍须按实际记录数加权。CPU 自动微分已验证上述两种分组，以及 (1)、(2,3,4) 的不整分组。

本轮八项均使用 micro32，因此该发现不推翻本轮同口径结果。后续 backend 或架构导致最大 microbatch 不同时，release 协议必须明确采用哪种训练目标、如何累积归一化；不能在切换 backend 时顺便修改目标，再把结果全部归因于 backend。

## 5. 与原论文/官方代码的对应关系

- **LLaDA**：[论文](https://arxiv.org/abs/2502.09992)与[官方训练核心代码](https://github.com/ML-GSAI/LLaDA/blob/main/GUIDELINES.md)使用随机遮蔽率，并对被遮蔽 token 的 CE 除以其遮蔽概率，再按固定 B×L 归一化。当前 `sum CE / 当前被选 token 数` 不是同一个目标。当前强制补选又改变了低 t 处的 mask 分布；不能简单补一个 `1/t` 就宣称复现原论文的 likelihood bound。
- **MDLM**：[论文](https://arxiv.org/abs/2406.07524)与[官方 diffusion.py](https://raw.githubusercontent.com/kuleshov-group/mdlm/master/diffusion.py)将连续时间 SUBS loss 与噪声导数/遮蔽率权重联系起来；`_sample_t` 支持按区间分层的 antithetic 抽样，`_loss` 按 attention token count 聚合。这支持把“噪声抽样”和“目标归一化”分开审计，不支持把任意加权目标都称作原始 diffusion ELBO。
- **MaskGIT**：[论文](https://arxiv.org/abs/2202.04200)与[官方 schedule](https://raw.githubusercontent.com/google-research/maskgit/main/maskgit/libml/mask_schedule.py)包括 cosine mask schedule；[官方 decoder](https://raw.githubusercontent.com/google-research/maskgit/main/maskgit/libml/parallel_decode.py)根据采样置信度与剩余 mask 计划迭代更新。可以借鉴噪声范围覆盖和独立 rollout 评估，但其图像 tokenizer、固定网格及具体调度效果不能直接转用为当前任务的结论。官方仓库未公开完整训练实现，本轮没有声称复现其训练。

已读代码的 URL 与下载 SHA 在 [source manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_noise_math_20260930/sources/manifest.json)。

## 6. 下一独立 release 的三个可分离 ablation

以下仅为待验证候选，不修改当前实验，不依据 50 步决定最终架构。

| 候选 | 单独改变什么 | 对照与判定 |
|---|---|---|
| A：训练按记录归一化 | 保持相同噪声，比较当前 microbatch token mean 与全局 per-record selected-token mean | 先用固定人工 logits/mask 比较 loss 与梯度，证明分组性质；训练固定架构、microbatch、记录顺序、初值、步数，同时报告 source macro/token micro。不可称作 LLaDA ELBO 复现。 |
| B：显式全 mask 曝光 | 预先指定小比例（例 10%）记录采用 t=1，其余保持原 U 分布 | 基线之外，可加平均遮蔽率匹配的连续高噪声对照，区分总体遮蔽增加与端点质量；同记录/步数报告实际监督 token 数和全 mask 次数。该混合是有意改变目标，不能宣称无偏保留原连续目标。 |
| C：独立 END 权重 | 保持原噪声，比较普通 token mean 与内容/END 分组均值的凸组合（例 END 权重 0.05） | 对不存在的组不填零，应明确按存在组重归一化；保持 loss 尺度可比，单列两类未加权 CE/正确率。检查 END 改善是否以内容退化或塌缩为代价。 |

B 的数值例子：10% 全 mask 混合的 E[t]=0.55045；连续对照 `t=0.001+0.999*u^(9/11)`、u~U(0,1) 有相同 E[t]，但没有 t=1 点质量。两者不具有相同完整分布，这只是平均噪声匹配对照，仍应报告完整 mask-count 直方图。

每次先验证一个因素，继续使用独立的 25/50/75/100% 重建指标；另在固定的人工序列任务上测全 mask 迭代 rollout、END 有效率、长度误差及输出分布。保持采样步数与解码规则一致，避免把 decoder 更改混入训练改动。短 smoke 只检查数值、梯度和实现；结论需要预先确定预算、多 seed 与独立评估，不能由几十步 loss 排名推导普适收益。

## 7. 可复核的 CPU 证据

[noise_probability_diagnostics.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/noise_probability_diagnostics.py) 只 AST 载入现有文件中的三个纯采样函数，不导入模型或数据模块。6 组长度/目标数，每组 200,000 次条件 Binomial 人工模拟；24 个高遮蔽事件的解析结果与独立数值积分一致，Monte Carlo 均在预定 6 个标准误范围内；每组另用 2,048 条人工整数数组调用真实采样函数，核对 context/padding 排除、标签与强制非空，并检查显式 t=1。

标量归一化反例使用 CPU autograd 验证。全部断言通过，详见 [diagnostics.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_noise_math_20260930/diagnostics.log)与 [完整 JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_noise_math_20260930/probabilities.json)。解析概率采用连续实数 t；当前 float32 随机数离散化对这些量级的影响未单独测量。

复现：激活 pllm 后运行 `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/noise_probability_diagnostics.py`。本次没有训练或生成实际序列，没有扩大 GPU 试验。
