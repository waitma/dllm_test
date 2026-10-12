# no-ESMC 序列架构第二轮调研：注意力、链间融合、状态空间与残差通路

更新：2026-09-30 UTC。性质：文献与代码只读调研、待验证方案；本文没有修改训练代码、部署新架构或生成实验成绩。正在运行的 R00–R11 以 [首轮研究记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md) 和 [实验进展](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 为准。

## 1. 对当前 A4 的判断与下一步优先级

已阅读 [research/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py)、[v3/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/model.py)、[layout.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py) 和 [LLaDA blocks](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py)。当前底座是 d=768、8 层、12 heads、SwiGLU、RMSNorm、双向 attention、链内 RoPE、role/scope embedding、同/异链偏置；R00–R11 已覆盖 QK norm、register、局部输入卷积、不同 masking/objective、来源采样和 role-pair bias。再次添加 SwiGLU/RMSNorm/RoPE 不构成新的消融。

**首选三个小改动：SDPA 后 headwise gate、早层独立链编码后层融合、交替链内窗口与全局层。** 它们可以保持同一词表、观测输入、去噪目标和输出接口，不需要外部 protein teacher。先做单项，不和现有 R01–R11 的改动捆绑。Value Residual Learning 是下一项低参数量备选。双向 SSM、Gated DeltaNet、Perceiver bottleneck、HC/mHC 值得进入研究储备，但需要更多接口/内核与公平性工作，当前不应因其名称新就优先于小改动。

这里的优先级是实现成本、与本项目故障假设的对应关系及可归因性判断，**不是对下游收益的预测**。语言或图像论文报告的收益、吞吐和长上下文能力均未在本项目复现。

## 2. 原论文索引与阅读范围

日期采用 arXiv 首次提交日期；更新版本另列。代码链接只指向作者/机构实现或论文原附录，未将第三方复刻标为官方。正文代码结论来自实际读取具体文件；未安装/执行外部仓库，也未下载模型权重。

| 工作 | 日期 / 版本 | 阅读重点与优先级 |
|---|---|---|
| [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708) | 2025-05-10；NeurIPS 2025 | §3 gate 位置、粒度和稀疏性消融；官方 Qwen3 attention；P0 |
| [Differential Transformer](https://arxiv.org/abs/2410.05258) | 2024-10-07；ICLR 2025 | §2 双 softmax 差分、参数匹配和每 head norm；官方 V1 及 2026-01-20 V2 代码；P1/P2 |
| [Value Residual Learning](https://arxiv.org/abs/2410.17897) | 2024-10-23；v5=2025-06-08 | §3 及 8 层消融；早期标题含 “For Alleviating Attention Concentration…”；作者 ResFormer 实现；P1 |
| [Hyper-Connections](https://arxiv.org/abs/2409.19606) | 2024-09-29；v3=2025-03-18，ICLR 2025 | §2、Appendix J 作者 PyTorch 风格代码；P2 |
| [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) | 2025-12-31；v2=2026-01-05 | §3 的约束、Sinkhorn 与系统开销；本轮未确认作者独立实现仓库；P2 |
| [Transformers are SSMs / Mamba-2](https://arxiv.org/abs/2405.21060) | 2024-05-31；ICML 2024 | SSD 的 chunk 分解、官方 Mamba2 与 sequence boundary 参数；P2 |
| [Jamba: A Hybrid Transformer-Mamba Language Model](https://arxiv.org/abs/2403.19887) | 2024-03-28；v2=2024-07-03 | attention:Mamba 比例与 MoE 是独立维度；AI21 官方模型文件；P2 |
| [Caduceus](https://arxiv.org/abs/2403.03234) | 2024-03-05；v2=2024-06-05，ICML 2024 | BiMamba 与 DNA reverse-complement 等变性的区别；官方 wrapper；P2 |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | 2024-12-09；v3=2025-03-06，ICLR 2025 | §3 gated delta rule、chunk 训练及 hybrid；NVIDIA 官方 layer；P2 |
| [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) | 2018-10-01；ICML 2019 | ISAB/PMA 及集合不变性；作者 modules.py；P1/P2 |
| [Perceiver IO](https://arxiv.org/abs/2107.14795) | 2021-07-30；v3=2022-03-15，ICLR 2022 | read–process–write、output query 和 mask；DeepMind 官方代码；P2 |
| [Attention Bottlenecks for Multimodal Fusion / MBT](https://arxiv.org/abs/2107.00135) | 2021-06-30；NeurIPS 2021 | mid-fusion 与 bottleneck 是不同轴；Google Scenic Encoder；P0/P2 |
| [Gemma 2](https://arxiv.org/abs/2408.00118) | 2024-07-31；v3=2024-10-02 | local/global 层交替；Google 官方配置；P1 |
| [Longformer](https://arxiv.org/abs/2004.05150) | 2020-04-10；v2=2020-12-02 | 同一层的滑窗与少数 global token；AllenAI 官方 attention；P1 的设计参照 |

## 3. 最小改动的可执行设计

以下 S1–S3 是本文方案编号，不替代运行中的 R 编号。实现应放入隔离的新版本；需要同版本无改动 control。现有 GPU 工作池中的代码指纹不能在任务进行时改变。

### S1：输入相关的 attention head 输出门，P0

论文机制是在每个 head 的 SDPA 结果与输出投影之间施加 sigmoid gate。作者 [Qwen3 实现](https://github.com/qiuzh20/gated_attention/blob/f9fe37c55e2232d866c370b8f23cbb8f1446c56b/modeling_qwen3.py) 扩展 `q_proj`，分离 query 和 `gate_score`，在 SDPA 后、`o_proj` 前相乘；headwise 为每 token/head 一个标量，elementwise 为每 token/head/dimension 一个值。

**本地落点：** 在 `LLaDALlamaBlock.forward` 的 `x_normed` 上产生 12 个 gate logits，传给 attention；在 `attn_out` 前乘 gate。保留链内 RoPE、双向 SDPA 和原 attention bias。禁止把 gate 放到最终 logits 上后称为同一机制。无 bias 的独立 headwise gate 共增加 `8×768×12=73,728` 参数；elementwise 会增加约 4.72M，不能混为一个方案。

**假设：** 当前多个 MASK 槽、元信息 token、长条件链可能形成过度聚合，gate 让模型按 query 选择各 head 的贡献。需要测 gate 按 role、遮蔽比例、链长的分布，避免稀少链被长期关掉。

**初始化与对照：** 零初始化 gate logits 对应 sigmoid=0.5，会改变 attention 分支尺度；应保留 0.5 常数 gate 的尺度对照。若改用 `2×sigmoid` 实现初始恒等，需明确这是本项目适配，不等同原版稀疏 sigmoid gate。首轮保持学习率和 schedule，不把增大学习率带来的收益归给架构。

### S2：早层链内独立、后层完整融合，P0

最贴切引用是 MBT 的 **mid-fusion** 设计。作者 [Scenic Encoder](https://github.com/google-research/scenic/blob/693290c7b7927b27c9539561fca4ae8c8a84f98e/scenic/projects/mbt/model.py) 在 `lyr < fusion_layer` 时分别编码各输入；当 `share_encoder=True` 且不启用 bottleneck 时，后层拼接输入做完整 attention。这与其“少量 bottleneck token 交换信息”的另一个分支不同。

**本地适配：** 8 层中先固定前 2 层仅同链 attention、后 6 层全局；所有链共用权重，role/scope 继续区分语义。只修改允许的 attention 边，参数量保持一致。若验证支持，再用前 4 层隔离作为第二个预先登记的深度。不是一开始为每种链建立独立 encoder，也不是受体 MSA 的 row/column 对齐。

**必须解决的边界：** A4 的 `<tcr>/<ab>` 和关系 prefix 使用 `chain_id=-1`，padding 也是 -1。若 prefix 在早层双向连接全部链，第二层即可通过 prefix 把 A 链传到 B 链，早层就不再独立。严格版本应让有效 prefix 自成组、链之间完全隔离，后层才开放通信；padding 按 `attention_mask` 屏蔽。若需要所有链早期感知 prefix，可另设只读静态元信息路径，单独记录该变化。

**风险与验证：** 整链被遮蔽时早层拿不到配对链条件，可能损害补全；观察 100% 遮蔽诊断和条件任务。对短片段与全长分别报告。单链输入在保持元信息策略一致时应接近 control；数值等价与否要依据实际 mask，而不是口头假定。CPU 验证应在早层检查 `d h_chain_B / d x_chain_A = 0`，在首次全局层确认存在交互路径。

### S3：交替链内窗口层与全局层，P1

[Gemma 2 官方配置](https://github.com/google/gemma_pytorch/blob/cb7c0152a369e43908e769eb09e1ce6043afe084/gemma/config.py) 的 `get_config_for_2b_v2` 明确交替 `LOCAL_SLIDING` 和 `GLOBAL`。这是隔层 local/global 的直接出处。Longformer 则在同一层设置局部窗口和少数全局 token，作者 [实现](https://github.com/allenai/longformer/blob/7a3a454cee25247f9cacc1b4ef05d6b8951d1ed6/longformer/longformer.py) 包含额外 global Q/K/V 与滑窗运算；不能将两者等同。Transformer-XL 的跨片段递归缓存也不是这一改动。

**本地适配：** 层模式 `local,global` 重复 4 次；local 的内容边满足同链且 `abs(position_i-position_j) <= 32`，这里 **32 是半径，最大含自身 65 个位置**。如最终代码用总宽 32，应另行命名和记录，避免配置歧义。最后一层保留全局。条件链和目标链都遵守同样定义，不能按拼接下标让两条链的边界接壤。

**公平性与风险：** 先用 dense mask 检验归纳偏置，不声称线性复杂度或吞吐改善；mask 仍可能产生 `B×H×L×L` 开销。短 CDR3 可能整个都在窗口内，而长条件链被明显限制，必须按链长分层报告。local 层可以接收上一 global 层已融合的特征，因此这不是“整个模型只有窗口范围”。prefix/register 是否全局应固定；若存在中继路径，则写成 local-plus-global-token，不能叫严格链内隔离。

## 4. 下一批低成本机制：ResFormer 与差分注意力

### 4.1 Value Residual Learning，P1

作者 [ResFormer 代码](https://github.com/Zcchill/Value-Residual-Learning/blob/55ae7bed8cae9719bd3533483d64b81be705ddbc/src/modeling/modeling_llama_resformer.py) 在第一个 attention 层保存 `value_states`，后层使用第一层与当前层 value 的各 0.5 混合，再执行 attention。论文最新版本还讨论可学习比例和稀疏后层应用，不能把早期代码的固定混合称为包含全部新版方法。

**A4 方案：** 在同一次 `denoise` 内显式传递 `V_first`，保留各层 Q/K 和所有 V 投影；先做固定 0.5 混合。它没有新增主投影矩阵，仍会增加一个跨层 value 张量的生命周期。可检验深层 residue 特征是否保留更多输入信息；对均被 MASK 的目标链，也可能主要保留无信息的 MASK 初始表示。

**实现陷阱：** 不 `detach(V_first)`，否则改变梯度路径；不存成跨 batch 的模块缓存。activation checkpoint 重算必须拿到该次 forward 对应的 value，VICReg 两视图与生成不同迭代之间绝不能共享。A4 `use_cache=False`，因此论文 SVFormer 的 KV cache 节省不是本项目收益。观察同一层深度的冻结表征与有效秩，加入单纯缩放当前 V 的对照。

### 4.2 Differential Transformer V1 / V2，P1/P2

[V1 作者实现](https://github.com/microsoft/unilm/blob/b3783e7d6c080cd682345aa80f3b2322e447da38/Diff-Transformer/multihead_diffattn.py) 使用两张 softmax attention 的差，带指数参数化 lambda、每差分 head RMSNorm 和层相关缩放。其注释要求差分 head 数设为普通模型的一半，以维持投影规模。A4 若尝试 V1，应从普通 12 heads 对应 6 个差分 heads 出发，并重新解释原来的每 head chain bias。

**2026 进展：** [官方 V2 文件，2026-01-20 提交](https://github.com/microsoft/unilm/blob/335fa4fd9e855b03c1f01e05a2173f89db6fe6e1/Diff-Transformer/Diff-Transformer-V2/multihead_flashdiffv2.py) 改为双倍 query heads、共享相应 K/V、每 token/head 的 sigmoid lambda，并移除差分后的 per-head norm；这是作者代码更新，本轮未核实独立 V2 论文，不能把它记作新论文。

**适配成本：** V2 要对同一 GQA 组中相邻 query-head 输出作差，不能随意前半/后半分组。代码写死 `causal=True` 且没有 A4 的完整 bias/mask 接口，不能直接导入；须改为双向并正确复制 padding、role/chain bias、RoPE。保持 12 个输出 heads 时，额外 Q 参数约 4.72M，另有约 0.074M lambda 参数。应与同参数量 Transformer 以及 headwise gate 比较，不引用其 autoregressive 解码速度作为 diffusion 去噪收益。

## 5. 残差通路：HC 与 mHC

### 5.1 Hyper-Connections，P2

[原论文 Appendix J](https://arxiv.org/html/2409.19606v3#A10) 给出作者 PyTorch 风格代码：残差流扩展为 `B×L×streams×d`，静态 alpha 包含选定流与 identity，动态 alpha/beta 经归一化、tanh 和小尺度参数调节；先汇聚成单个 block 输入，再分配 block 输出。这里直接读论文作者代码，不把常见的第三方 `hyper-connections` 包当官方实现。

**A4 落地：** 若稳定的 8 层模型有明显层间特征高度相似、扩深收益不足，可做 2-stream 原型，分别包装 attention 与 MLP 残差分支，最终汇聚回 d=768。它扩的是残差存储通路，不是把每个 block 全部宽度翻倍。现有 hidden-state、checkpoint 和表征输出接口均需适配。

**风险/公平性：** 多流增加 activation 保存与内存访问，参数少不等于便宜；必须保持主干宽度，并同时做等 GPU 时间对照。当前 8 层还没有证据表明深度瓶颈主导，优先级低于 gate 和链结构实验。不能把单个 scalar residual gate 称为完整 HC。

### 5.2 mHC，P2

[原论文 v2 §3](https://arxiv.org/html/2512.24880v2) 把 residual stream mixing 限制为非负、行列和均为 1 的矩阵，通过 Sinkhorn 归一化近似实现；pre/post 映射采用不同的 sigmoid 尺度。论文还专门设计 FP32 小系数与融合内核以控制开销。**本轮未找到并确认作者发布的独立 mHC 实现，不采用搜索到的社区仓库作为官方代码证据。**

**A4 方案与限制：** 先有 2-stream HC 的可测基线，再只替换 residual mixing 约束。记录行/列和误差、梯度、activation 峰值、迭代次数与 BF16/FP32边界。双随机 residual mapping 的非扩张性质不等于整个含 attention/MLP 的网络获得训练稳定性保证；简单 Python Sinkhorn 的吞吐也不等于论文融合内核。对当前小模型，这一机制是待验证研究题，而不是默认升级。

## 6. 长序列架构：双向 SSM、hybrid 与 Gated DeltaNet

### 6.1 Mamba-2 / SSD，P2

原论文从 semiseparable 矩阵的块结构建立 SSM 与 attention 的联系，并设计 SSD chunk 算法。官方 [Mamba2 module](https://github.com/state-spaces/mamba/blob/6b72c122713bb769cc82c6b8e6d019c53d27d6a1/mamba_ssm/modules/mamba2.py) 包含输入投影、短卷积、状态扫描、gated normalization，以及 `seq_idx` / `cu_seqlens` 路径；仅把 `attention` 类名换成 Mamba 不会得到正确双向去噪。

**A4 方案：** 把链内编码层替换为双向扫描，链间交互继续保留 dense attention；先单独处理每条有效链，状态和卷积在链边界重置。逐样本 padding 被置零不一定等于“不会传播状态”，必须验证 packed 与逐链运行等价。不同 denoise step 不复用有目标上下文的递归 state。先比较 2/8、4/8 层替换，而非立即全替换。

**风险：** 双向扫描有两次计算，短链与当前 2048 总长度可能不能摊薄内核/分段成本；A100、现有 PyTorch/CUDA 和 fused kernel 要重新验证。报告参数量、真实链长吞吐、长条件层的收益；不照搬论文 2–8 倍 core-layer 速度。

### 6.2 Jamba：混合比例与专家机制分开，P2

论文把 attention:Mamba 比例、MoE 出现频率和每 token expert 数作为独立设计轴。[AI21 官方 Jamba-v0.1 模型文件](https://huggingface.co/ai21labs/Jamba-v0.1/blob/main/modeling_jamba.py) 按 `layers_block_type` 选择 Attention/Mamba decoder layer，并按 `layers_num_experts` 在 dense MLP 与 MoE 间选择；Mamba 的 dt/B/C 还有内部 RMSNorm。

**A4 适配：** 先只借用 hybrid 的设计方式：双向链内 mixer + 周期性全局 attention，MLP 保持 dense。角色多不自动意味着应该为每个角色建稀疏专家。若后来加入 MoE，需单独评估总参数与激活参数、每来源每专家样本数及路由负载。

**边界：** 官方模型为 autoregressive causal LM，不带本项目的双向 masked diffusion、链内 reset 和输入 feature 白名单。其 KV cache 成本、长上下文解码优势不能直接迁移到本项目。禁止把 attention、Mamba、MoE 同时变更后只归因于某一个组件。

### 6.3 Caduceus：可借双向 wrapper，不借 DNA 互补等变性，P2

作者 [BiMambaWrapper](https://github.com/kuleshov-group/caduceus/blob/2593c807ebfc0aede05e7a991474b1a0e60fdc93/caduceus/modeling_caduceus.py) 正向扫描一次，反转序列扫描后再反转回来；支持相加等组合方式，并可共享正反向的 in/out projection。论文的 DNA reverse-complement parameter sharing 是另一机制。

**A4 适配：** 只借双向上下文设计，每条链单独逆转有效位置、再恢复原位置；保留 N→C 的语义和原位输出。不能把蛋白序列倒过来视为同一个样本，也没有 DNA 那样的互补字符映射。共享投影但分开的动态参数与“所有参数共享”也不同，应明确记录。

**验证：** padding 长度和 batch 编排不应改变有效残基输出；两条链交换顺序不应让状态串入其他链。等参数量和等计算两套比较都需要，避免把双向的额外处理量藏起来。

### 6.4 Gated DeltaNet，P2

论文用 gate 控制状态衰减、delta rule 做有针对性的关联更新，并给出 chunk 并行算法。[NVIDIA 官方 layer](https://github.com/NVlabs/GatedDeltaNet/blob/fba56c5351ed34e747e43818a0708a13dc4c28d2/lit_gpt/gated_delta_net.py) 实际包含 Q/K/V 的短卷积、Q/K normalization、独立 beta 与衰减 gate、chunk gated-delta kernel、输出 normalization/gate。

**A4 方案：** 与 Mamba-2 共用“链内双向 mixer、间隔跨链 attention”的实验接口；先只替换少数链内层。输入链、padding、正反向都必须显式定义 state reset；该作者文件不是即插即用的双向模块。

**风险/公平性：** 输出 gate 不等于 S1 的 SDPA gate，delta rule 也不能简化成普通 attention 乘 sigmoid。先验证所依赖内核与数值稳定性，再谈算力节省。至少与同参数 Mamba-2 hybrid 和 Transformer 对照，另记每一步实际时间；当前短序列不保证有工程收益。

## 7. 链集合、层次结构与 latent bottleneck

### 7.1 Set Transformer，P1/P2

[作者 modules.py](https://github.com/juho-lee/set_transformer/blob/6fdae7fb8d085d502326ecb8deea8f1c38eb9cdf/modules.py) 的 ISAB 先用可学习 inducing points 读取集合，再让元素读取 inducing summary；PMA 用可学习 seed query 池化。代码是固定形状集合示例，没有本项目变长/缺失链的 padding 语义，不能直接无 mask 套用。

**A4 适配：** 链内保留有序编码，在每链若干 summary 上做集合交互，再回写到 residue。所有 summary 保留 role/scope；不得把 alpha、beta、重链、轻链互换后要求语义不变。可先用每链 2 个 summary、一个共享小交互层作为机制试验，再决定是否替代部分 dense 层。

**已有能力不能重复宣称：** A4 已用链内位置、role 和 chain equality，而不是显式 chain-index embedding；在所有特征随 token 一起置换、无跨链绝对位置泄漏时，主体 attention 应具备相应置换等变性。因此先做 whole-chain permutation 检查，不能不经测试就声称 A4 完全不懂链集合。当前 collator 的 canonical sorting 会掩盖该检查，需要在合法 batch 上同步置换全部 feature 后逆置换输出。

**风险：** 单链平均池化会丢掉局部相互作用，少数 summary 可能成为信息瓶颈。区分等参数附加模块与真正替换 dense attention；特征池化不应按真实 target 终点删掉生成容量尾部。

### 7.2 Perceiver IO，P2

[DeepMind 官方实现](https://github.com/google-deepmind/deepmind-research/blob/9176a9f23ced8e3d6024718d757be8d67cfb6927/perceiver/perceiver.py) 让固定数量 latent 通过 cross-attention 读取输入，主要深度在 latent 上处理，decoder 的每个 output query 再读取 latent。`input_mask` 和 `query_mask` 分开；decoder query 可由位置及输入特征构造。

**A4 方案：** 先把长条件链压到固定数量 latent，受体目标保留完整 token 分辨率；query 仅由可观测的 role/scope、链内位置、当前 noised token 和容量构造。保留原形状 logits 及 residue 表征返回。目标全 MASK 时仍有合法 query，不能依赖真实目标长度或 `<protd>` 位置恢复输出。

**风险/公平性：** 低频局部条件可能被压缩损失；逐残基任务与全局分类必须都测。register 在原序列后多加几个 token 不会自动获得 Perceiver 的计算节省。只有重层在固定 latent 上运行才改变主复杂度；输入读取与输出写回成本仍在。若条件编码始终不读取 target，推理可讨论跨 denoise step 缓存；当前 joint A4 的条件 hidden 会随 target 变化，不能直接缓存。

### 7.3 MBT bottleneck 的进一步适配，P2

S2 只借 mid-fusion；若要完整研究少量跨链通信，可按 [官方 MBT Encoder](https://github.com/google-research/scenic/blob/693290c7b7927b27c9539561fca4ae8c8a84f98e/scenic/projects/mbt/model.py) 的 bottleneck 分支，让每链和共享 latent 交互，再聚合各链更新。它与 R05 的“全局 register 和所有 residue 都做 dense attention”不同。

受体链不是音频/视频模态。为了控制参数，应先共享链 encoder，并区分 missing chain、空链和有效链；聚合不能让缺失链的占位符等权稀释有效链。跨链 token 对关系可能需要直接通信，故应留完整全局层作为旁路或对照。是否提速取决于是否真正减少 residue 对 residue 的 dense 计算，不取决于加入 latent 的名字。

## 8. 统一落地和验证约束

本节是本项目设计建议，并非任何一篇论文的结论。

| 必须固定/观测的项 | 执行要求 |
|---|---|
| 输入语义 | 保留真实 C/F/W 残基、role/scope、链内 position、目标容量和 `<protd>`；feature 白名单不接受 labels、真实 residue/end masks 作为网络输入 |
| 训练比较 | 同数据清理证书、source sampler、seed、global256、token/样本预算和 schedule；训练目标不同者仍用统一验证诊断；新增版本需要自己的 control |
| 计算公平 | 同参数量与等 GPU 时间分开报；gate、双向扫描、extra Q、latent read/write、checkpoint 重算均计入；dense sparse-mask 不声称 sparse-kernel 提速 |
| 工程检查 | 自己的同 flavor 最坏长度 profile；有限 loss/grad、新增参数梯度、save/load、checkpoint 重算、padding/batch/链重排不变性、reference-free generation |
| 架构特有检查 | S1 gate 分布和尺度 control；S2 前层跨链梯度为零与后层通路；S3 链边界和窗口半径；ResFormer 同 forward 的 V_first；SSM reset；latent query 不用真终点 |
| 阶段预算 | 3-step profile 与20-step合成诊断只验证工程；真实数据1000步初筛、候选累计5000步；每100步同一25/50/75/100%固定遮蔽验证。更长训练若仍无趋势/或趋势反转，明确记录，不以短训敲定最终架构 |
| 表征与生成 | 按来源、链长、单链/配对链、短片段/全长报告共同AA/end指标；候选再验证冻结特征、few-shot、配对、条件生成；不用 test 选择结构 |
| 数据 | 真实训练继续要求新 downstream-scope 证书；包括 test、few-shot support/query 暴露边界，以及已有 TCR/条件序列清理协议。资源就绪和合成验证成功不等于数据门禁通过 |
| 工程发布 | 不改动正在运行 release 的 fingerprints；新脚本/变体在隔离目录开发，测试后冻结快照，再投递新任务；最终部署状态由主记录维护 |

小批顺序建议：同版本 control + S1 + S2 + S3；如果预算有限，先 S1/S2，S3 随后。LayerScale 可作为另一维优化稳定性消融，但需要与这些单项分开。只有明确各单项的收益后，才组合 gate 与局部/全局布局。ResFormer 次轮加入；SSM、mHC 或 bottleneck 每次只开一个较大架构分支。

## 9. 可追溯的代码阅读证据

2026-09-30 实际读取下列原始文件并核对 SHA-256。这里保存元数据，不复制外部仓库、不保留额外 checkpoint。固定 commit 链接见对应机制小节；commit 日期表示该文件最新变更，不等于论文发表日期。

| 文件 | 固定 revision 前缀 | SHA-256 |
|---|---|---|
| gated_attention/modeling_qwen3.py | f9fe37c55e22 | `fd1eff3afaeff708f8506d6952655ee834af3c97054ca124633ff3536548ff0d` |
| unilm/Diff-Transformer/multihead_diffattn.py | b3783e7d6c08 | `a49056a3227df0730d61c3ee547198a887539ca99e7aacb59585a3ca5ecece5d` |
| unilm/Diff-Transformer-V2/multihead_flashdiffv2.py | 335fa4fd9e85 | `c64742b3a3eb1e9271b245dc86cac0750b21dbabd12eb2cfb0b3897a653dd524` |
| Value-Residual-Learning/modeling_llama_resformer.py | 55ae7bed8cae | `ba2b508d667c4d2fbf2822683bf239c68022ae77e35c9f7a6f88376e8e12c027` |
| mamba/mamba_ssm/modules/mamba2.py | 6b72c122713b | `605e4439ff0baec8d8acaf4a191d9f0570eea9900065a065909124c472b08707` |
| caduceus/caduceus/modeling_caduceus.py | 2593c807ebfc | `e1b45c6d97a123a9657056832fa7a018eb509d1a06ddc1c20792124770ee8f54` |
| GatedDeltaNet/lit_gpt/gated_delta_net.py | fba56c5351ed | `8aaee1dcff184742148bb694730e5c1da0d246bbba92d2a0a462e77963f35fbc` |
| set_transformer/modules.py | 6fdae7fb8d08 | `26c11c9f18209cd2ddcf52073a28b0052b356673ab356a1a5bf8ec41e9c95787` |
| deepmind-research/perceiver/perceiver.py | 9176a9f23ced | `b68017bf5fd20f0de77b2068a2fee2548eb0b6f24c938bf21fcf35f56aa7b6ef` |
| scenic/scenic/projects/mbt/model.py | 693290c7b792 | `008fbd39222513e81684a51a33f0b96abc0c088fdc33ab06c547fc71606511f3` |
| gemma_pytorch/gemma/config.py | cb7c0152a369 | `43bcc8e024815e18e41045fd168509669ffc2af506061fa598db795e021dafd1` |
| longformer/longformer/longformer.py | 7a3a454cee25 | `8e9e76b4dbae20cc97f726fdc112491db0e4a214146933f7303e8aa676f35282` |

另读 AI21 官方 Hugging Face `modeling_jamba.py` 和 HC 论文 Appendix J；前者本轮读取 `main`、未冻结 revision，因此实现前需要固定版本。mHC 仅以作者论文公式及系统说明为依据；没有用社区实现填补“官方代码已验证”的空缺。
