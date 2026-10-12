# no-ESMC 架构扩展：图像、多模态与自监督机制

更新：2026-09-30。本轮在已有 [18 篇文献与 R00–R11](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md) 之外新增阅读十项工作，并深入核对 DINOv3 的 Gram 实现。本文记录的是文献证据和本项目待检验方案，**没有新增模型质量实验结果**。现有运行中的模型/runner保持冻结；本文不修改它们，不提交任务。

最适合先做的两个小改动是 **LayerScale 残差缩放**和 **SDPA 输出的按 head 门控**。第三个结构方向是**前几层链内编码、后几层跨链融合**；它有明确的信息流对照，先于更复杂的瓶颈 token。LeJEPA/SIGReg 值得作为独立表征目标分支，但需要专门处理统计批、投影随机性与恢复一致性，暂不与前三项捆绑。

## 阅读范围与可追溯证据

先读了当前 [ResearchModel](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py) 和 [LLaDA attention](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py)。下列官方代码共 16 份、约 300 KB，只下载文本阅读，未安装或执行外部代码；每份均保存 commit、URL、SHA256。目录为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_vision_expansion_literature_20260930`，索引见 [manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_vision_expansion_literature_20260930/manifest.json)。不保存模型权重。

| 工作 | 阅读范围 | 外部预训练依赖 | 本项目优先级 |
|---|---|---|---|
| CaiT / LayerScale，2021 | 原论文 §2、深度消融；官方残差 block/初始化 | 缩放机制不需要 | P0，小结构改动 |
| Gated Attention，2025 | 原论文机制与消融；Qwen3 attention 三种后端 | 门控机制不需要 | P0，小结构改动 |
| MBT，2021 | 原论文 §3–4；Scenic 融合分支 | 机制不需要；论文主实验用了 ImageNet21K 初始化 | P1，先试融合层位置 |
| LeJEPA，2025 | 原论文 §3–5；官方 minimal、切片和 EP 统计 | 不需要 teacher | P1，独立目标分支 |
| data2vec 2.0，2023 | 原论文 §3；官方 teacher/target/multimask | 不需要外部 teacher，使用在线 EMA teacher | P1，先纯 latent 辅助目标 |
| iBOT，2022 | 摘要；官方训练循环、iBOTLoss、EMA/centering | 不需要外部 teacher，使用在线 EMA tokenizer | P2；全文 web 抓取超大小限制，OpenReview遇验证页，不声称完整论文阅读 |
| AIMv2，2024 | 原论文伪代码/方法/消融；Apple encoder 和 layers | 方法可从头训练，但依赖真实图文配对监督 | P2，结构与目标分工启发 |
| Perceiver IO，2022 | 原论文 §3；DeepMind encoder/decoder query | 不需要 | P2，长条件输入方向 |
| Token Merging，2023 | 原论文匹配/消融；官方 merge/timm patch | 机制不需要 | 暂缓，目标残基不直接合并 |
| Representation Autoencoders，2025 | 原论文 §3–4；官方 stage1 RAE | 原方案依赖冻结的预训练表征 encoder | 暂缓，不直接符合当前 scratch 主线 |
| DINOv3，2025，补充阅读 | 原论文 §4；官方 GramLoss、teacher 生命周期 | 原始 Gram 阶段依赖已有较好 teacher | 后期局部表示退化时再考虑 |

以上“机制不需要”与“论文已证明随机初始化也有效”是不同陈述。本项目只有经过相同数据、计算和验证协议的实测，才能判断这些机制是否适用。

## 1. LayerScale：最小的残差结构改动

原方法在 attention 和 FFN 两个残差分支各放一个可学习的逐通道乘子，保留原有归一化。论文浅层实验到 18 层使用初值 0.1，24 层用 1e-5，更深用 1e-6；不能直接把深层工厂配置的极小值用于当前 8 层模型。[原论文 §2](https://arxiv.org/pdf/2103.17239)。

已读 [官方 cait_models.py，固定 commit](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_vision_expansion_literature_20260930/cait.py:129)：`LayerScale_Block` 建立 `gamma_1/gamma_2` 两个 `(dim,)` 参数，分别乘 attention/MLP 的输出，再进入 drop_path 和残差相加。block 默认 `init_values=1e-4`，XXS24 工厂传入 1e-5。来源：[官方源码](https://github.com/facebookresearch/deit/blob/7e160fe43f0252d17191b71cbb5826254114ea5b/cait_models.py#L129)。

**本项目方案**：在隔离的新 block 中，仅实现两个逐通道缩放；不复制 class attention、talking-head 或图像分类头。8×768×2=12,288 个参数，初始化 0.1。对照至少包括 R00、两个分支固定 0.1、可学习向量初始 0.1；若向量有效，再比较每分支单一可学习标量，辨别逐通道自由度是否必要。无需改 tokenizer、链角色、RoPE、`<protd>` 或损失。

关注每层残差输出/输入范数比、gamma 范数、梯度峰值，以及共同 AA/end 验证。0.1 会缩小早期梯度，短训变慢不等于长期无效；反之不允许通过独立增大学习率后与未调参 R00 比较，声称纯架构收益。当前只有 8 层，深层稳定性论文收益可能并不显著。

## 2. Attention 输出门控：有选择地使用聚合结果

Gated Attention 的核心是在 SDPA 输出之后、输出线性映射之前加 query-dependent sigmoid gate；论文同时研究不同放置位置与粒度。这是语言模型机制，纳入本轮是因为它不依赖图像假设、适用于现有 attention 接口。[原论文](https://arxiv.org/abs/2505.06708)。

已读 [官方 Qwen3 实现](https://github.com/qiuzh20/gated_attention/blob/f4c2a5f6ffd6ec709e0c60072c95ed4f5ce5b5d2/modeling_qwen3.py#L276)：headwise 分支把 `q_proj` 输出扩到 `d+H`；从 attention 输入 hidden_states 投影得到 H 个 gate logit，另 H×head_dim 为 Q。gate 不经过 Q/K norm 或 RoPE。输出形状 `(B,L,H,head_dim)` 乘 `sigmoid(gate)` 后 flatten，再 `o_proj`。参数沿普通 Linear 正态初始化，bias 为零；没有特殊的零权重初始化，也没有乘 2。

**本项目方案**：每层独立 `Linear(d,H,bias=False)` 从 pre-attention norm 后的 x 产生门控，等价于原版扩展投影中的额外行；d=768、H=12、8 层新增 73,728 参数。保持本项目双向 attention、chain bias 和位置逻辑。先测普通 sigmoid，不把 `2*sigmoid` 或饱和为 1 的变体混入同一名称。

R00、固定 0.5 输出缩放、可学习 query gate 三项能分开判断“单纯减幅”与“依输入选择”。额外记录按层/链角色的 gate 均值、分位数和低门控比例、attention entropy、首个特殊 token 接收的注意力份额。原论文的因果语言 attention sink 不等于本项目已存在的问题；不提前称该机制消除了受体模型的 sink，也不直接预测生成质量会提高。

## 3. MBT：链内表征与跨链融合分开安排

MBT 研究两种限制：跨模态融合从某一层开始；或者不同输入只能经少量共享 bottleneck 交换信息。原论文主实验使用预训练 ViT，机制本身并不要求外部 teacher。[原论文 §3–4](https://arxiv.org/pdf/2107.00135)。

已读 [Scenic model.py](https://github.com/google-research/scenic/blob/8c113c501c9f700b69899c55a69e65bb46727da6/scenic/projects/mbt/model.py#L339)：`lyr < fusion_layer` 时每个输入流独立编码；bottleneck 分支将同一组潜在 token 分别拼到每个流，分别更新，再均值合并各流的 bottleneck；支持共享编码器权重。它不是简单在全部 dense token 后追加 register。

**本项目第一步**：同一套层参数和总层数，试“前 2 层 chain-local，后 6 层 global”，与全部 global 对照。另一个预先指定的方案可为每 2 层中 1 层 global，但这属于本项目假设，不是 MBT 原配方。初期只改信息可达性，不同时加 latent、卷积或新目标。所有层仍保留完整 token 输出，末端有 global 交互。多条链共享同层参数，不按角色复制整套 backbone。

局部 mask 应按真实 `chain_ids` 区分，不能只按 role 合并两条同角色链。特殊 token 的可见规则必须显式：一个同时读取各链的全局 token 会在两层内重新传递信息，使“完全链内”对照失效；padding 也不能成为通道。条件整体仍可在后面的 global 层进入 target 推断，避免整链遮蔽时目标永远看不到条件。

**第二步才考虑 bottleneck**：每条链保留自身 token，跨链只经 4/8/16 个共享 latent；latent 数量、融合层位置独立消融，并保留 R05 register 的对照。输入链不是音频/视频模态，因而“跨模态”只提供计算分组机制，不提供受体互作的生物学等价假设。注意：在 dense attention 上写屏蔽矩阵只改变信息流，通常不减少矩阵乘法；只有分组执行/有效稀疏 kernel 的测量才能支持加速声称。

## 4. LeJEPA/SIGReg：无教师的表征分布约束

LeJEPA 将多视图一致性与 SIGReg 联合，后者经随机方向投影，用特征函数统计约束表示分布。方法不依赖负样本、stop-gradient 或 EMA teacher；论文理论建立在特定数据和下游风险假设下，不能写成对本项目所有下游的无条件保证。[原论文 §3–5](https://arxiv.org/html/2511.08544v3)。

已读 [官方 minimal](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/MINIMAL.md)、[slicing](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/multivariate/slicing.py) 与 [EppsPulley](https://github.com/galilai-group/lejepa/blob/c293d291ca87cd4fddee9d3fffe4e914c7272052/lejepa/univariate/epps_pulley.py)：Gaussian 随机方向按列单位化；每次 forward 改变方向；当前库维护 global_step；cos/sin 先在样本轴求均值再形成差值平方，统计量乘实际 N（DDP版另计world_size）。minimal 用 17 个积分点、256 个投影方向，是一个示例配置，不能省略这些口径后只记录“SIGReg”。

**本项目方案**：先保持 R08 的两次 masking、相同 pooling/projector 和两次 forward，把辅助项从 VICReg 换为同视图一致性 + SIGReg；这应叫 CE+SIGReg adaptation，不能叫完整 LeJEPA 复现。先用 FP32 统计、固定投影数和明确 lambda；投影 RNG 独立于训练遮蔽，状态进入 checkpoint，验证使用固定方向。辅助池只包含该样本的目标受体残基，避免把共享 antigen/MHC 作为 shortcut。

不能把累积 global batch=256 当成统计批=256：先在每个 microbatch 计算 cos/sin 再求非线性统计，与跨 256 个样本聚合后计算不是同一损失。方向数增加也不能制造更多独立样本。复制同一条人工记录的 smoke 尤其不适于评估 Gaussian 分布目标。对照 R08 必须匹配真实统计 microbatch、projector、mask views、辅助预算；同时报告有效秩、方差、pair cosine 与冻结探针，不能只用 SIGReg 下降宣称通用表征变好。

## 5. data2vec 2.0：先学上下文表示，再考虑稀疏编码

data2vec 2.0 用同一次训练产生的 EMA teacher 对未遮蔽样本编码，student 预测其上下文化表示；通过仅编码可见位置、较轻 decoder、多次遮蔽复用 teacher 降低成本。论文明确区分 unique sample 数与多 mask 次数。[原论文 §3](https://arxiv.org/pdf/2212.07525)。

已读 [官方 data2vec2.py](https://github.com/facebookresearch/fairseq/blob/3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99/examples/data2vec/models/data2vec2.py)：`make_targets` 对最后 K 层表示按配置归一化后平均；主路径取 block 返回的 layer result；`clone_batch` 重复 target 对应多个 mask；源码记录 target/pred variance 并设坍塌下限。teacher 有单独 EMA 更新和恢复状态。直接平均本地最终 hidden_states 不等于该实现。

**本项目分解**：先保留当前 dense denoise，仅加 masked-residue latent 预测辅助头；在线 EMA 初始化为同一个随机 student，teacher stop-gradient，只将干净表示用作 loss target，不输入 student。局部位置必须依 chain ID 和链内坐标对齐，padding/容量尾部不计目标；teacher 的目标真实长度不得变成生成时额外输入。最后 K 层选择、EMA 时点和更新频率写入方法配置与完整恢复状态。

第二阶段再比较可见 encoder +轻 decoder，不能第一轮同时更换结构、mask、teacher和损失后归因。全MASK target 时可见 encoder 的输入不一定足够，仍需明确由 condition 和角色/位置 query 生成的机制。与 R00 比较时同时给出每步 unique records、student views、teacher forwards 和 GPU 秒；“复用一次 teacher”不代表多次 student 免费。

## 6. iBOT：在线软目标适合表示辅助，不替换氨基酸字母表

iBOT 使用在线 tokenizer 的 patch 软目标与全局 self-distillation 目标。论文摘要和官方训练实现已核对；本轮全文抓取受到大小/浏览器验证限制，未声称读完所有消融。[原论文入口](https://arxiv.org/abs/2111.07832)。

已读 [官方 main_ibot.py](https://github.com/bytedance/ibot/blob/da316d82636a7a7356835ef224b13d5f3ace0489/main_ibot.py#L478)：teacher 初始化复制 student；teacher 读未mask global views、student 读masked views；patch loss在同视图mask位置计算，全局 loss在跨视图计算；teacher softmax前 centering/temperature，输出 detach；每步optimizer后EMA更新；center状态也需要恢复。

**本项目候选**：在不改 AA CE 的前提下，对 masked residue hidden 增加软原型预测；另设 chain pooling 的全局目标。原型分布是上下文表示辅助，不能据此把有限氨基酸 vocabulary 改成图像 learned tokenizer，也不能把近似片段自动当同义正样本。需要监控原型占用熵、teacher/student分布、稀有来源使用率；和 data2vec 的连续目标在相同teacher成本下比较。多项temperature/centering/EMA超参会增加筛选成本，所以排在SIGReg与简单结构改动之后。

## 7. AIMv2：表示 backbone 与生成 decoder 的任务分工

AIMv2 将视觉 encoder 与生成图像patch和caption的decoder配对，encoder采用prefix attention，图像与文本目标分开计算。原论文依赖真实配对文本，而非任意“多任务标签”都可替换caption。[原论文方法与伪代码](https://arxiv.org/html/2411.14402v1)。

已读 Apple 的 [models.py](https://github.com/apple/ml-aim/blob/a018ae3208ef3289cae9aa024947e0bb3b71d3ff/aim-v2/aim/v2/torch/models.py) 和 [layers.py](https://github.com/apple/ml-aim/blob/a018ae3208ef3289cae9aa024947e0bb3b71d3ff/aim-v2/aim/v2/torch/layers.py)：公开代码提供RMSNorm/SwiGLU视觉encoder、可选attention-pool头及文本encoder；本次文件中没有完整原论文multimodal pretraining decoder/训练循环，不能宣称已核对完整训练复现。

**本项目启发**：共享表示主干后接较小去噪decoder，使通用pooling特征与强重建要求不完全挤在最后一层。先做纯序列版本，与相同总参数/FLOPs的单体8层backbone比较；若引入V/J或链角色预测，只能用真实可用训练元数据并检查是否仅从输入tag复制。用户的抗原/受体配对是条件信息，不是caption语义同义视图，不能任意套CLIP式全局匹配。

AIMv2的prefix因果训练与当前双向masked diffusion不同。直接混用其attention mask会改变训练问题与生成器，应该独立命名和验证；RMSNorm/SwiGLU等如果底座已具备也不能再算新的创新。

## 8. Perceiver IO：将长条件输入压到固定数量潜在 token

Perceiver IO 通过input-to-latent cross-attention、latent自注意力和output-query到latent的cross-attention，将主要深层计算与输入/输出长度分离。[原论文 §3](https://arxiv.org/pdf/2107.14795)。

已读 [DeepMind官方 perceiver.py](https://github.com/google-deepmind/deepmind-research/blob/f5de0ede8430809180254ee957abf36ed62579ef/perceiver/perceiver.py)：encoder使用可学习latent位置初始值；decoder显式构造输出query；encoder/decoder均支持输入或query mask，输出数量由query确定。并非向原Transformer追加几个token就获得相同复杂度。

**本项目优先迁移条件侧**：完整保留受体target槽，长且已观测的条件链先编码为固定 K 个 latent，再让 target 对 latent cross-attend。输入侧必须保留role和链内位置；不同条件链是否共享latent作为单独消融。先比较 K=32/64 的精度–时间曲线，不同时将target压缩。条件为短epitope的样本可能完全不获益，因此报告按条件长度分层的吞吐和AA/end验证。

若完整改成Perceiver，target output query仍需表达角色、容量和位置，未知结束点仍由模型预测；用训练真实长度创建query会泄漏答案。本项目仍有精确残基恢复需求，过强瓶颈可能抹掉细粒度信息；需要逐位/逐链错误、长条件依赖和条件缺失诊断。复杂度上的优势需包含输入编码和decoder成本实测。

## 9. ToMe：暂不合并需要逐残基预测的 token

ToMe依据token相似性做二部匹配合并，并传播token大小以调整后续attention。图像patch可以含冗余内容，氨基酸位置具有离散输出含义，二者不能直接等同。[原论文](https://arxiv.org/pdf/2210.09461)。

已读 [merge.py](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/merge.py) 与 [timm patch](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/patch/timm.py)：先归一化metric，交替分为两组，取相似度最大的匹配；merge发生在attention和MLP之间；metric来自head平均的K；proportional attention在logit加log(token size)。unmerge只是把聚合值放回对应位置，不恢复被平均掉的细节。

**本项目判断**：不把目标受体/结束标记/capacity槽做ToMe合并。若后续确有长条件瓶颈，可仅在同一条件链内试有限压缩，并与固定分块pool和Perceiver latent对照；不得跨链、跨特殊token或根据真实目标mask决定合并。RoPE位置在合并后没有天然唯一值，必须明确保持的坐标/区间表示。即便embedding相似，也没有证据说明两个残基可互换；吞吐收益不能掩盖该误差。

## 10. RAE：有价值，但不是当前scratch主线的立即替代

Representation Autoencoders使用冻结的预训练表示encoder加训练的重建decoder，然后在高维表示空间训练生成模型；论文另分析高维latent下的宽度、噪声与decoder鲁棒性。[原论文 §3–4](https://arxiv.org/html/2510.11690v1)。

已读 [官方 stage1/rae.py](https://github.com/bytetriper/RAE/blob/a4d18c4db766419cbe7cb8c02cd9f7ceb0ec9041/src/stage1/rae.py)：`encode` 被no_grad包裹，默认是DINOv2 encoder配置，训练时可向latent加噪声；decoder负责重建，并有latent归一化状态。不能仅把这个类拿来就声称“所有部分从零训练”。

**本项目后续可行路径**：先在合格语料上训练本项目表示encoder，再冻结它训练离散序列decoder，最后研究latent生成；整条路线的前序预训练成本全部计入。它比当前改一两个block多出数据/训练/推理阶段，暂缓。图像LPIPS/GAN和连续Gaussian噪声假设不直接适配离散AA目标；论文高维continuous diffusion的宽度下界也不等于当前离散masked-CE模型必须加宽的证明。

## 11. DINOv3 Gram：补充实现边界与进入条件

原论文Gram anchoring用较早、局部表示较好的teacher约束student的局部相似度矩阵，保留关系而非强制每个向量固定；作者后期才启用并定期更新teacher。[原论文 §4.2](https://arxiv.org/html/2508.10104v1)。

已读 [GramLoss](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/loss/gram_loss.py) 与 [teacher生命周期](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/train/ssl_meta_arch.py)：FP32归一化后计算样本内或批内Gram MSE；有截断负相似度的选项；teacher加载/更新、tokens_used和损失schedule均显式配置。原文简洁公式与实际带开关代码需要区分。

**本项目进入条件**：已有可靠scratch checkpoint，且固定valid显示训练越久、局部表征越退化，才考虑同链残基Gram约束。teacher选点只看validation，记录来源和SHA。禁止把token embedding Gram解释为真实接触图。两视图需要同链、同位置对应，去掉padding/end/capacity尾部；不直接对所有不同链的残基做全局强一致。完整Gram是O(L²)，长条件需限制到受体或固定采样pair，后者属于新的近似。若没有退化证据，不为增加论文元素而引入第三份网络。

## 下一轮最小可解释实验集

这里的 V 编号只是设计标签，不占用已有 R00–R11 ID，不代表实现或提交状态。实际新模块和任务清单由主实验记录维护。

| 设计标签 | 改动 | 必要对照 | 首先回答的问题 |
|---|---|---|---|
| V1 | attention/FFN逐通道LayerScale，init0.1 | R00、固定0.1两分支 | 学到的残差幅度是否改善稳定性和共同验证 |
| V2 | pre-attn输入→head sigmoid，SDPA后应用 | R00、固定0.5 attention输出 | query依赖的门控是否比单纯减幅有效 |
| V3 | 前2层链内、后6层全局 | 全8层global；必要时每2层全局的预先指定方案 | 延迟链间交互是否改善表示，不切断条件依赖 |
| V4 | CE+双mask一致性+SIGReg | R08，匹配统计批/两视图/投影头；双forward CE控制 | 表征分布约束是否提高冻结探针且不损伤AA/end |
| V5 | CE+EMA连续latent辅助目标 | 纯CE、同计算量CE；单层/多层target消融 | 上下文表示目标是否补足仅预测AA的信号 |

首轮不直接测试 V1+V2+V3 的混合体；先单项，再预先指定少量组合。分支需要独立源码指纹、checkpoint schema、profile证书和恢复测试。新设计未开始真实训练前，不改动既有运行中的代码。

统一沿用已确定的 clean-data 门禁、seed42初筛、global batch256、相同样本顺序/数据版本/词表、warmup100和horizon10000；1000步筛选、优选后到5000步，但这些步数只提供阶段性趋势，不保证收敛。按每来源固定validation，25/50/75/100%固定遮蔽下报告标准AA/end CE和准确率，另按长度、chain role、输入组合分层。不同训练目标的总loss不进入横向排行榜。

基础表征验证应同时看冻结mean-pooling线性探针、相同预算few-shot与既定条件任务；probe层和pooling只在validation选择。对新增pooling头增加均值pool对照，避免把下游头容量提升误写成backbone改进。通用探针优先复用既定官方协议，不能临时从测试集挑最有利的指标。最终比较至少多个seed，报告配对差异/不确定性，不用一次0.1%的变动宣布胜出。

同样steps只能保证部分样本呈现预算相同，不能保证计算量相同：额外teacher、双视图、decoder、latent投影均单列forward数、实际token数、峰值显存、GPU秒、有效样本/秒。结构选择同时看等样本与等GPU时间两组比较。模型高利用率应来自有效任务吞吐，不能把显存占用或无效计算当成科学结果。

共同失败信号包括非有限值、有效秩明显塌缩、只学end不学AA、只学AA不学结束、条件置空几乎不影响配对/条件表现、某一来源提升但多个来源明显退化。受体链/条件的view augmentation只用同一记录的合法遮蔽；图像裁剪、翻转、mixup、任意残基替换均不自动保留生物学含义。完整C/F/W输入、未知target终点、链角色和capacity边界继续遵循A4接口，不能为提高数字暴露目标真实长度。

## 本轮记录与状态

- 完成原论文/官方代码阅读与16份固定commit文本快照；没有下载权重或执行外部实现。
- 已向主实验线程提供LayerScale和headwise gate的精确位置、初始化、参数量及固定缩放对照，并补充MBT的融合层依据。
- 架构模型和 GPU profile/训练结果由主线程另行记录；本文末尾补充本轮独立 runner 的 CPU 验证，不把它视为 GPU 验证。
- 当前数据证书/运行状态以主账本和实时报告为准；本轮研究不能解除真实数据去重门禁。

## R12–R19 独立 runner 与评估协议（2026-09-30）

已新增 [run_immune_architecture.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_architecture.py) 和 [test_research_next_runner.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_next_runner.py)。runner 导入旧 research runner，在自身进程的 `configure()` 内替换模型、variant 清单、fingerprint、synthetic measure 和 profile child 入口；不修改仍在运行的旧 runner/model。`base.main()` 继续负责参数检查、真实语料门禁、训练、续跑及证书验证。源码指纹保留全部原项目条目，额外覆盖新 runner 和 `research_next` 下所有 Python 源文件。

profile 保留原有互斥文件锁、1/2/4/8/16/32/64/128/256 倍增扫档、独立子进程、900 秒单档超时、预期 OOM 停扫、80% 扫描停止线、85% 可选档上限和 global batch256；所有 profile child 改为调用新 runner。已有 COMPLETE 仍由原 `verify_profile()` 检查源码、summary SHA、生产配置、GPU 型号和精度。没有绕过测量直接写入推荐 microbatch。

synthetic 每次评估用既定的逐记录确定性遮蔽，分别评估 25/50/75/100% 四档。四档复用同一组逐位置随机数，因此遮蔽集合嵌套，且不依赖批次 padding 或训练 RNG。每档仅在 `selected ∩ eligible ∩ attention` 上计算 CE，在其中的 AA/end 子集上计算对应准确率；可见 token 不计为正确答案。传给 `denoise` 的仍只有 7 项合法 feature，标签和真实 end 位置只用于损失与指标汇总。条件链和未选中的容量尾部保持原样。整个评估恢复 torch CPU/对应 CUDA RNG，以及评估前的 train/eval 状态，异常路径也恢复。

输出顶层 `loss/aa_loss/terminal_loss/aa_accuracy/terminal_accuracy` 沿用 100% 遮蔽定义，保证旧 synthetic 汇总和 `loss_decreased` 可读；新增 `mask_levels.mask25/mask50/mask75/mask100`，每档附 selected/AA/end 数量，以及四档等权宏平均 `mask_mean`。某档没有某类目标时该指标写 JSON `null`，宏平均排除该档，避免 NaN 或凭空设为 0。四档是固定概率采样，实际数量不强制恰好为 25/50/75/100%。本轮未新增条件链补全专门评估。

全遮蔽 toy 存在不可辨识性：当前 6 条人工记录里，4 条 TCR 的全遮蔽合法输入特征完全一致但标签不同。既有审计的 AA-only 上限是 46/82=56.10%，全部目标联合上限是 51/93=54.84%；两个上限分别优化相应指标，不声称能同时达到。参见 [监督与审计记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md)。新 runner 因而不把全遮蔽 100% 正确率作为诊断通过条件。这是当前人工数据的上限，不推广为真实任务上限。

CPU 验证在激活 `pllm` 后执行：

```bash
USE_TF=0 python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_next_runner.py -q
```

结果 **16 passed，6.84 秒**；96 条 warning 均来自已有底层 `torch.is_autocast_cpu_enabled()` 的弃用提示。覆盖 R12–R19 全部 variant 的四档有限指标、全遮蔽旧五指标一致（绝对容差 2e-6）、只统计遮蔽位、7 项 feature 白名单、条件/容量尾部不改、四档嵌套、评估与异常 RNG/模式恢复、空子集 JSON、源码指纹、self 子进程、预期 OOM 与非 OOM 失败区分、既有 profile 复用及源码变动拒绝。初次测试发现新代码误用了不存在的 `model.vocab` 属性；已改为项目既有 `model.tokenizer.mask_token_id`，再跑上述测试全部通过，旧模型未改。

另用新脚本完整入口执行 R13 两步 CPU synthetic：d64/L2/H4、90,312 参数、global batch2、6 条人工记录。结果 [summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_cpu_20260930/runner_r13_cli/summary.json) 和 [COMPLETE.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_cpu_20260930/runner_r13_cli/COMPLETE.json) 均已写出，状态 `completed`，训练/评估循环 0.129 秒（不含进程导入和建模），全遮蔽 CE 3.528018→3.377347，四档平均 CE 3.526903→3.353860；两步末 AA/end 准确率均为 0，不据此评判结构效果。四档 selected 数分别 27/48/72/93，全遮蔽为 82 AA+11 end。目录只包含两个 JSON，没有产生 checkpoint。CPU 小模型验证接口和数值流转，不能替代 A100 生产配置 profile、100 步诊断或 clean corpus 1000/5000 步比较。

本轮没有 submit/enqueue，没有修改旧 runner、旧研究模型或当前看护进程。实际 GPU 运行状态、选择和共享项目账本由主线程同步。
