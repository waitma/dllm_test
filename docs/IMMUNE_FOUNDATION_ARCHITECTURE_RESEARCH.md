# no-ESMC 从零训练：架构与方法探索记录

更新：2026-09-30。本文是本项目的持续研究记录；区分论文证据、本项目假设、已实现代码与实际实验结果。最新用户决定：将先前六卡方案改为c20250601单实例8×A100非闲时工作池，已提交并Running。先前queue012提交建议仅作历史记录。

## 当前判断

用户已有从零训练达到一定效果的观察，足以支持继续做受控架构实验，但还不能证明通用预训练没有价值。首轮应该同时测试 **训练任务、表征目标、链结构归纳偏置、优化稳定性、数据采样**，不要只加深 Transformer 或只比较 reconstruction loss。

最值得投入的假设是：一个受体 foundation model 应同时理解链内序列模式、链间条件关系，并提供可迁移表征。当前 A4 已有紧凑词表、角色/范围 embedding、链内 RoPE、同/异链 attention bias、观测序列布局和单链结束标记；这些是新探索的固定底座。旧 A0–A4 累加比较不等于下面的新方法比较。

首轮已实现 **12 个变体、36 个 YAML**；当前又扩展R12–R19共8个架构配置，总计20种注册配置。第二轮运行与研究见 [架构扩展v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。首轮提交说明见 [提交说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)。R00–R11的八卡GPU显存/合成检查现已开始，尚无可据以比较下游质量的新结果，没有证明任何变体能提高生物学下游效果。合成诊断仅检查实现与优化是否工作。

## 文献范围与适用性

以下直接链接原论文或作者实现，避免把图像成绩当成序列模型收益。主要方法阅读包括论文方法/消融及官方代码；较远方向只作范围筛查，未声称复现全部工作。

| 工作 | 可迁移的机制 | 本项目判断与状态 |
|---|---|---|
| [MAE, CVPR 2022](https://arxiv.org/abs/2111.06377) | 高遮蔽率、仅处理可见内容的 encoder、轻量 reconstruction decoder | 启发遮蔽任务设计；真正的非对称 encoder/decoder 暂缓，R01 **不是完整 MAE**。氨基酸信息密度不同，不能照搬 75% 最优结论 |
| [SpanBERT, TACL 2020](https://aclanthology.org/2020.tacl-1.5/) | 连续片段遮蔽与 span 表征 | R01 测试连续遮蔽；没有实现原文 span-boundary objective，不称 SpanBERT 复现 |
| [DiT, ICCV 2023](https://arxiv.org/abs/2212.09748) | Transformer 扩散骨干、条件调制、零初始化 | R07 仅测试可观测 MASK 比例的输入 FiLM；不是逐层 adaLN-Zero，也不是连续图像扩散 |
| [MDLM, NeurIPS 2024](https://arxiv.org/html/2406.07524v2) | masked diffusion 的目标推导和工程方法；明确比较 time conditioning | 附录 E.5 显示时间条件未必有收益，因此 R07 为较低优先级消融。当前项目 masked CE 也不能直接宣称实现了其 ELBO |
| [DINOv2, 2023](https://arxiv.org/abs/2304.07193) | 全局/局部自监督表征，以及训练数据选择的重要性 | 推动我们同时考虑表示质量与来源分布；R09 为数据采样对照，不是 DINOv2 训练配方复现 |
| [DINOv3, 2025](https://arxiv.org/html/2508.10104v1) | Gram anchoring 约束局部特征关系，改善长训练中的 dense feature 退化 | 适合作后续 residue-level 关系保持；原文依赖较好早期 teacher，不能从随机 teacher 直接外推效果；本轮未实现 |
| [Vision Transformers Need Registers, ICLR 2024](https://arxiv.org/abs/2309.16588) | 增加独立工作 token，减少内容 token 承担内部计算的压力 | R05 增加 4 个全局 register；是否对短受体有益未知，特别关注 residue embedding 的变化 |
| [ViT-22B, ICML 2023](https://arxiv.org/html/2302.05442v1) | Q/K 归一化控制 attention logit 尺度 | R04 借用已有 LLaDA Q/K norm 插槽；实现为整个投影维度 RMSNorm，不能与原论文 LayerNorm 等同 |
| [ConvNeXt, CVPR 2022](https://arxiv.org/abs/2201.03545) | 深度可分卷积的局部归纳偏置仍有价值 | R06 为链内 kernel=7 的一维残差混合；不会跨链卷积，没有完整复制 ConvNeXt block |
| [Swin, ICCV 2021](https://arxiv.org/abs/2103.14030) | 局部窗口与跨窗口通信兼顾成本及范围 | 可启发长 antigen/MHC 的局部编码、间隔全局层；首轮保留 dense attention，避免把跨链交互直接截断 |
| [VICReg, ICLR 2022](https://arxiv.org/abs/2105.04906) | 两视图一致性、方差下限和去相关，无需负样本 | R08 使用同一记录的两次遮蔽，联合 masked CE 与小权重表征正则；无 ESMC teacher |
| [I-JEPA, CVPR 2023](https://arxiv.org/abs/2301.08243) | 从上下文预测目标区域的潜在表征；EMA target encoder | 后续可比较 masked residue latent prediction。需要 teacher/predictor、表示坍塌诊断和额外显存，本轮只形成设计 |
| [REPA, ICLR 2025](https://arxiv.org/abs/2410.06940) | 用高质量外部表征辅助生成模型学习 | 原版依赖预训练 encoder；为保持 scratch 主线，本轮不把 ESMC/ESM2 作为教师。未来可用本项目早期 scratch teacher，但这是新假设 |
| [MSA Transformer, ICML 2021](https://proceedings.mlr.press/v139/rao21a.html) | 序列集合与行/列 attention 带来的上下文建模 | 提示链集合和层次交互方向；受体链对不是同源 MSA，不能直接解释为列对齐关系 |
| [ProtMamba, Bioinformatics 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12206526/) | 状态空间模型与长同源上下文，混合 FIM 训练 | 较适合长条件输入探索；首轮短链可能收益有限，替换双向 diffusion 骨干需单独适配，未提供伪 Mamba 开关 |
| [ProGen3, 2025 preprint](https://www.biorxiv.org/content/10.1101/2025.04.15.649055v1.full) | 稀疏专家和大规模蛋白预训练的数据/计算扩展 | 本轮仅通过作者摘要/搜索展示核对方向，全文抓取失败；不据此声称阅读完整消融。MoE 涉及负载均衡、稀有链专家训练不足及总/激活参数公平性，暂缓 |
| [Feature Reuse and Scaling, ICML 2024](https://proceedings.mlr.press/v235/li24a.html) | 蛋白模型迁移需要按任务、深度、训练程度分别考察 | 支持冻结表征/线性探针、微调和生成分开评估，不能仅看一次总分 |
| [ImmSET, PMLR 2026](https://proceedings.mlr.press/v297/noceda26a.html) | 把多条变长输入作为序列集合建模 | 提醒 role-aware 交互值得测；R10 仅是有向 role-pair attention bias，不是其完整架构或结果复现 |

## 实际阅读的代码与本地接口

7 份作者代码的只读快照、URL、SHA 和抓取结果已保存：[manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_literature_20260930/manifest.json)。快照只用于阅读，没有安装或执行外部代码；本项目实现复用自己的模块。

| 官方代码 | 阅读的实现点 | 本地落地与差异 |
|---|---|---|
| [MAE models_mae.py](https://github.com/facebookresearch/mae/blob/main/models_mae.py) | random_masking 的可见 token gather 和 decoder restore | 当前仍让所有容量槽进入 backbone；不能声称拿到了 MAE 稀疏编码的计算节省 |
| [DiT models.py](https://github.com/facebookresearch/DiT/blob/main/models.py) | timestep embedding、六路逐层调制、输出/调制零初始化 | 只借鉴条件调制的受控初始化；R07 用输入可观测 MASK 比例，不用真实目标长度 |
| [VICReg main_vicreg.py](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py) | variance/covariance loss，以及跨 rank gather 的统计口径 | R08 为单 GPU/microbatch 统计，25:25:1 归一化为 1:1:0.04 后再乘 0.05；没有跨累积步汇总统计，必须披露这个差异 |
| [DINOv2 vision_transformer.py](https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py) | register 参数初始化与加入输入的位置 | R05 追加 4 个向量，参与注意力，输出时移除；不修改氨基酸词表或给 register 添加预测标签 |
| [MDLM diffusion.py](https://github.com/kuleshov-group/mdlm/blob/master/diffusion.py) | time_conditioning 关闭路径 | 没有时间条件不是代码遗漏；作为假设独立验证 |
| [I-JEPA train.py](https://github.com/facebookresearch/ijepa/blob/main/src/train.py) | no-grad target encoder、上下文 predictor 与表示损失 | 未来 teacher 要明确来源和更新时刻；本轮 R08 没有 teacher |
| [ConvNeXt convnext.py](https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py) | depthwise kernel=7、残差尺度 | R06 是轻量的链内输入局部混合，避免在每层引入新计算 |

本地重点阅读：[v3 模型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/model.py)、[布局](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py)、[噪声采样](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py)、[LLaDA attention](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py)、[来源采样](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/balanced_sampling.py) 和 [训练入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/immune_pretrain_v3.py)。

## 首轮 12 个可执行变体

所有变体从头初始化，底座为 A4，d=768、8 layers、12 heads、MLP=3072。方法代码隔离在 [research/model.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research/model.py)，不会把旧 v3 checkpoint 静默当成新架构加载。

| ID / 文件前缀 | 实际改动 | 要检验的假设 | 优先级 |
|---|---|---|---|
| R00 `r00_a4_control` | 新 runner 中的 A4 对照；测试确认相同权重/RNG 时训练 loss 与现有 A4 一致 | 排除 runner 与 schedule 的影响 | 必跑 |
| R01 `r01_span_mask` | 每行 50% 概率使用链内连续 span，长度 geometric(p=.33)，预算按 t×观测残基数；另一半保持 Bernoulli | 避免只利用邻近单个残基的容易重建，促进 motif/长程建模 | P0 |
| R02 `r02_chain_completion` | 每行 50% 概率整条遮蔽随机目标链；其余目标链变为该步可见上下文 | 让配对与条件补全成为预训练任务的一部分 | P0 |
| R03 `r03_chain_balanced_loss` | 每个被监督链分别计算 AA mean + 0.1×end mean，然后跨链/样本平均；其他监督 token 单独计入 | 长链/短链以及结束目标的梯度预算可能不合理 | P0 |
| R04 `r04_qk_norm` | 每层 Q/K 投影后、分 head/RoPE 前增加 RMSNorm | 小数据 scratch 优化是否更稳定，是否改善注意力过度集中 | P0 |
| R05 `r05_register4` | 4 个可学习全局 register，原 token logits/表示长度不变 | 给内部聚合计算独立空间，改善可迁移 residue/chain 表征 | P0 |
| R06 `r06_local_conv7` | 零初始化 7-position depthwise residual；屏蔽不同 chain ID、padding 与非内容 token | 局部 motif 偏置能否改善样本效率 | P1 |
| R07 `r07_mask_ratio_film` | 可见 MASK/target-slot 比例→MLP→输入 scale/shift，最后一层零初始化 | 高/低遮蔽状态的适配是否有额外价值 | P2 |
| R08 `r08_vicreg` | 两次遮蔽、同一个模型，共享 projector；masked CE + 0.05×VICReg-style 辅助损失；pool 仅用 receptor target 残基 | 不依赖预训练教师，能否同时提高 few-shot 表征与重建 | P1 |
| R09 `r09_equal_source` | 模型与 R00 相同；复用来源轮转且来源内无放回遍历的 sampler | 来源长尾是否比架构本身更限制跨任务迁移 | P0，数据对照 |
| R10 `r10_role_pair_bias` | 每个 head 有向 role→role attention bias，零初始化，叠加既有同/异链 bias | 区分多类链之间的交互是否优于只有同链/异链两类 | P1 |
| R11 `r11_span_qknorm_register` | R01+R04+R05 预先指定组合 | 方法之间是否互补 | 最后看；不能用于单项归因 |

R03 同时改变链权重与 end 权重，是一个训练目标方案；若有效，需要追加两者各自单独消融。R11 的组合事先固定，没有根据 test 挑选。R08 辅助表示池使用训练监督侧的 residue mask，**该 mask 不传入 denoise**；推理仍是无参考长度输入。

## 本项目可继续形成的方法贡献

以下是待验证的组合假设，不是已证明的论文创新性声明：

1. **链结构约束的多粒度学习**：token/span/whole-chain 遮蔽，配合角色范围、链内位置、有向跨链关系。优势假设是统一单链、配对链和条件输入；必须按任务分别看收益。
2. **表征与生成联合学习**：用同一 scratch 模型的一致性目标减少“会重建但不适合 few-shot”的差距。比较纯 CE、VICReg-style、后续 EMA latent prediction，并防止 condition-only shortcut。
3. **内容 token 与计算 token 分离**：register 配合链局部混合，让全局聚合与局部序列表示分工；检查是否改善 pooling、attention 分布和 residue-level 特征，而不只看总 loss。
4. **分布优先的架构评估**：先把来源与数据暴露控制好，再解释架构收益。若 R09 提升大于 R04/R05，下一轮应该优先研究覆盖率和来源配比。

真正值得后续实现但本轮未提供假开关的方向：

| 方向 | 为什么值得做 | 进入下一轮的条件 |
|---|---|---|
| 稀疏可见 encoder + 轻量 diffusion decoder | MAE 式减少大量 MASK/容量尾部计算 | 先解决 zero-visible 的整链生成、角色/位置恢复、训练推理一致性，并重新 profile |
| 每链局部编码 + 间隔全局 cross-attention | 为长 antigen/MHC 降低 quadratic 成本 | 对长上下文任务有实测收益需求；不能让跨链交互断开 |
| EMA teacher / JEPA / Gram 关系约束 | 兼顾局部 residue 与全局 chain 表征 | 先有稳定 scratch baseline/teacher，明确 warmup、EMA、stop-gradient、坍塌监测 |
| 双向 SSM–attention 混合 | 长条件序列下的吞吐和显存 | 需要兼容双向去噪的实现、依赖验证、匹配参数/计算预算 |
| 角色专家/稀疏 MoE | 缓解不同链/来源之间的任务干扰 | 每专家有效训练量、路由熵、负载均衡和总/激活参数都可测 |
| 高级采样/自条件化 | 减少生成步数，提高迭代稳定性 | 与表示学习质量分开比较，额外 forward 纳入预算 |

## 如何判断是否提高 foundation model

不能按这 12 个模型的训练总 loss 排名。R03、R08 改了目标，R01/R02 改了遮蔽难度，R09 改了来源频率。通用验证使用**相同的25/50/75/100%遮蔽下的标准AA/end CE**，关闭额外训练目标；这是统一诊断口径，不等于完整验证NLL。每条记录的验证遮蔽固定，不随microbatch/padding或训练RNG变化，验证也不扰动之后的训练随机流。

真实数据实验现统一：seed42、global batch256、lr1e-4、warmup100、cosine horizon10000。全部方法先跑1000步，候选及R00再到累计5000步；每100步保存和验证，scratch训练前记录step0。每来源固定seed42抽至多256条valid，报告各遮蔽率的来源宏平均AA/end；旧200步/warmup20/horizon2000方案由本预算替代，不能直接拼表或无校验续训。当前只有单 seed 筛查；候选扩大训练后至少多个 seed、同数据/样本预算和另外一组等 GPU 时间对照。R08 多一次训练 forward，不能只报等 steps 优勢。

下游阶段按已有官方协议，至少分别覆盖冻结特征/线性探针、few-shot、配对、条件生成与多类受体任务。先在独立 validation 上选配置，再一次性查看 test；所有 test/support/query 暴露边界沿用 [数据审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md)。不把官方 few-shot support 的合法适配当成预训练权限。现有下游加载器需要显式使用 `ResearchModel.from_pretrained`，本轮未冒充已接通全部正式 benchmark。

建议淘汰标准：非有限梯度、明显更慢且无表征收益、对单一来源改善但多个来源退化、纯总 loss 降低而 AA/end 共同指标恶化、表示坍塌或极低有效秩、超过参数/计算预算的收益未能在对齐后保持。R08 microbatch 统计与 accumulation 不等价，需单列实际 microbatch；若太小先调小 projector 或用固定统计批方案，不能说 global256 就是 VICReg 的统计 batch256。

## 数据与资源边界

当前最新 prepared 候选是 [immune_v7_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1)，来自并行任务的 MHC 输入更新。它可被接口读取不等于通过新下游去重；本轮真实数据 YAML 默认指向该候选，但仍强制证书门禁，最终清理目录确定后重新生成。没有复制旧证书，没有修改原始 benchmark。

本轮不改旧 A0–A4 配置，不操作其他会话已经提交的任务。新架构、新数据代码可能改变显存/代码指纹，因此 **每个 R 变体独立重新 profile**，不能套用旧 A4 的测量。新 source SHA 包括 research、base model、collator、数据 loader/sampler、训练和门禁代码。

同 flavor 的单卡 profile 在每个第一组 YAML 内先执行，batch 从小到大；每档独立子进程、3 个含 backward/Adam 的步骤，最长输入2048，register 方案内部长度2052。选不超过85%显存且整除256的最大通过档，再做20步合成验证。VICReg 从 batch2 起。batch 上限256；若下一档 OOM 或上限已到，记录原因，不为了利用率改变 global batch。只有 profile 证书通过，才允许该方法的后续短训。

## 实验记录与待办

- 2026-09-30：12 个实现的前向/反向、关键新增参数梯度、保存重载、A4 control 等价、链局部隔离、条件/容量尾部保护、无参考生成、坍塌惩罚及 profile 指纹/显存门禁等 22 项 CPU 测试通过。
- 2026-09-30：最终代码在 worker 环境 protenix_abtcr 中以 CPU 完成 R08、R11 两步入口检查；详细验证记录见 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 后续检查为准。没有把两步结果写成下游提升。
- 生成的 GPU 配置尚未由本会话提交。首次真实 GPU 显存、20步诊断结果、task ID 和新的优先级判断，后续回填本节或链接对应实验报告。
- 决策必须保留：测试集不参与方法筛选；完整真实残基 C/F/W 保留；A4 仍学习 `<protd>`；每条 target 的容量边界保留；没有引入 ESMC 教师；论文启发不等于原版复现。


## 2026-09-30 首版结果确认与验证预算升级

已直接复核平台状态及磁盘COMPLETE/summary SHA：`t-20261001020115-zvt6f` 为 **Success**，Elapsed936s，End=2026-09-30T18:16:52Z；COMPLETE=passed。扫描中的OOM是找到容量边界，不是整个任务Failed。它是合成显存profile，不是1M语料训练。另一个任务`t-20260930225817-8r8sx`最新平台状态 **Killed**，End=18:32:04Z；“仍Queue”是取消前的旧快照。

| 轴 | 推荐micro-bs | accumulation（global256） | 峰值GiB | OOM档 |
|---|---:|---:|---:|---|
| A0 | 16 | 16 | 58.24 | 32 |
| A1 | 256 | 1 | 43.59 | 扫到256未OOM |
| A2 | 256 | 1 | 43.59 | 扫到256未OOM |
| A3 | 32 | 8 | 41.80 | 64 |
| A4 | 32 | 8 | 41.76 | 64 |

A0@16的3步loss为12.4864→2.8479→2.0846，有限值通过。该结果为后续资源规划提供依据，不证明架构泛化。记录见 [baseline_profile.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_budget_update_20260930/baseline_profile.json)。新R变体继续按自己的代码实测，历史证书不会因调大步数被篡改或冒用。

| 阶段 | optimizer steps | 能回答的问题 | 不能据此声称 |
|---|---:|---|---|
| 每档显存扫描 | 3 | backward/Adam能否运行、该输入的显存档位、初步有限值检查 | 长期稳定、收敛、真实数据收益 |
| 合成短训 | 20 | 方法分支、梯度、loss记录、运行流程是否工作 | foundation representation变好 |
| 清洁真实数据初筛 | 1000 | 多个验证点的学习趋势、来源差异、训练速度与早期退化 | 排名已稳定、最终效果确定 |
| R00和2–3个候选 | 累计5000 | 收益是否持续，是否只在warmup/早期出现 | 已覆盖全语料或所有下游、单seed即可定论 |

1000步对应256,000次样本呈现，5000步对应1,280,000次；这是样本呈现预算而非去重后的独立样本数。现有候选train约684.9万条，因此两种预算都不是全量训练一轮；最终清理后样本量也会改变。架构和R08额外forward成本不同，应结合wall time比较；单看相同步数不等于等计算量。

**没有一个固定的小步数可以保证验证充分。** 若最后数个验证点仍明显变化、AA/end学得不平衡、方法排序翻转或跨来源变化不一致，就不能结束效果判断。保留后续多seed、固定协议下游评测和更长训练的必要性。此前合成20步A1出现总loss下降而AA accuracy仍为0，就是不能只看总loss的本地例子；具体数字仍以旧实验记录为准。

训练支持从完整checkpoint续到更大的累计step，强制相同数据/源码/方法/batch/schedule/验证协议，恢复optimizer、scheduler、RNG和采样进度；仅停止步可以改变。自然采样使用按epoch确定的种子，跨epoch续训可重建顺序。不能把旧200步schedule的checkpoint直接接进本方案。配置入口仍见 [README](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)，第三组默认scratch，显式--resume-map才注入选定checkpoint。

最终24项CPU测试通过，新增验证遮蔽的batch/padding/RNG不变性以及真实Trainer“连续4步”与“2步保存后续到4步”参数一致性检查，包含跨epoch和中间验证。[测试日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_budget_update_20260930/pytest_final.log)。本次只查询状态、更新代码/配置/文档，没有submit/cancel或新GPU训练；真实数据门禁继续保留。


## 2026-09-30 八卡池已运行

任务`t-20261001033327-fmnql`，c20250601，单实例8卡，非闲时；8实验正在并行、4待领。控制、保留和续训见 [工作池说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/README.md)，结果自动进入 [LIVE](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)。维护负载单独标识，不是训练结果。


## 2026-09-30T20:19:38.413413+00:00 第二轮架构扩展

原R00–R11首轮12项全部成功，后续R12–R19的8项已在同一八卡池做100步诊断。新的文献、官方代码、115项集成验证、四级mask及批量训练前条件，见 [架构扩展v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。源码按版本冻结；不把工程通过写成下游收益。
