# CV多对象关系与理解/生成联合建模探索

## 2026-10-01T11:47:52.931165+00:00 优先级修订

保持共享主干；G02末端小adapter为有条件的结构候选，G01作为读出/优化对照，硬瓶颈和EMA暂缓。[决策与具体测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)、[三方审查摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)、[禁用决策矩阵YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。本次只读设计，没有改动或启用固定包内8份配置；CPU/GPU工程成绩不作真实架构收益。旧论文/设计优先级由本节和决策页覆盖。

更新：2026-10-01T11:29:06.399847+00:00。G00–G07八个原型的CPU语义检查已完成；新增GPU引擎CPU控制44个唯一案例/环境分阶段通过，固定15文件源码包与8配置核对完成。现有8卡执行接入、GPU结果及实际训练数据比较仍未完成。[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。R20/R21/R24/R26工程验收保持，不与新8方法混排行；以下文献阅读范围不变。

## 结论和优先级

“一个模型”可保留一个共享主干、同一词表和统一checkpoint，同时让理解与生成有少量不同参数。图像论文提示两类目标可能偏好不同表征，但这个冲突在本项目尚未实测。优先比较低成本任务专用norm/末端适配层，再比较跨对象瓶颈及自身EMA表示辅助。所有迁移均是待验证假设。

已有[CV扩展](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_VISION_EXPANSION.md)、[序列扩展](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_SEQUENCE_EXPANSION.md)已经覆盖MBT、Perceiver、Set Transformer、register、JEPA、REPA、Gram等；本轮不把它们列作首次发现。区别是加入同模型理解/生成的联合评价、任务分工对照及关系依赖诊断。

## 原论文证据、借用范围与限制

| 工作及本次阅读范围 | 原工作支持什么 | 本项目判断/当前状态 |
|---|---|---|
| [Transfusion](https://arxiv.org/abs/2408.11039)，摘要 | 共享Transformer联合离散自回归与连续扩散目标，主干从头训练 | 支持考虑共享主干与不同目标；没有证明离散多段任务必须加入连续扩散。只作设计依据。 |
| [Show-o](https://arxiv.org/abs/2408.12528)，摘要＋作者forward | 同主干处理自回归理解与离散去噪生成；代码分开计算图像、语言、理解损失 | 明确分开损失及指标，不能只用总loss判定两个能力都提高。原实现使用已有语言模型。 |
| [Janus](https://arxiv.org/abs/2410.13848)，摘要＋模型构造 | 理解/生成编码通路分开，共享语言Transformer | 可研究任务专用小模块；输入本身同为离散符号，不直接搬两套视觉编码器。 |
| [UniFork](https://arxiv.org/html/2506.17202v1)，§3及作者backbone | 浅层共享、深层任务分支；原实现复制后半层，并有独立末端norm | 优先从小适配层开始。原文初始化自Qwen，增加总参数；不把论文数字当scratch证据。 |
| [Show-o2](https://arxiv.org/pdf/2506.15564)，§3.1＋作者融合实现 | 语义与低层特征两路融合，专用输出头 | 借鉴保留局部细节和全局语义的分工；原方案使用VAE/SigLIP及预蒸馏，不能称无外部编码器方案。 |
| [BAGEL](https://arxiv.org/abs/2505.14683)，摘要＋作者README | 同系统支持理解与生成；公开模型区分活跃与总参数 | 大规模专家结构暂低优先级。HTML方法页本次失败，未据此声称已审阅其全部MoT代码。 |
| [MultiMAE](https://arxiv.org/abs/2204.01678)，摘要＋作者mask/gather/adapter | 多输入、多输出，采样可见token预算并用任务输出适配器重建 | 后续研究缺失输入鲁棒性；真正稀疏编码须单独解决全遮蔽和输出位置恢复。只改dense mask不算MAE加速。 |
| [Slot Attention](https://arxiv.org/abs/2006.15055)，摘要 | 交换对称的slot通过竞争注意力归纳对象表示 | 已知分段边界时，不需要先重新发现对象；先测试显式段池化和关系latent，更复杂slot暂缓。 |
| [SRA](https://arxiv.org/html/2505.02831v1)，摘要＋作者loss/train | EMA模型在较低噪声、较深层提供目标，指导较高噪声的较早层 | 可借机制做人工离散任务对照；增加teacher前向且原图像代码仍加载预训练VAE。G06是新适配假设。 |
| [SRA2](https://arxiv.org/abs/2601.17830)，标题/摘要/作者发布说明 | 用VAE自身表示提供更轻量的对齐指导 | 本项目没有同等VAE表示基础，暂不直接实现；不能因无需另加encoder就宣称完全scratch可照搬。 |
| [REPA](https://arxiv.org/abs/2410.06940)，摘要＋已有项目记录复核 | 外部高质量表示辅助扩散生成 | 已有方向，原版依赖外部表示；保持不使用ESMC/ESM2教师。 |

上述图像结果不是本项目性能保证。尤其生成要保留逐位置信息，而理解可能鼓励压缩与不变性；对全部token强制一致，可能损害生成，需实测两端权衡。

## 已实际核读的作者代码

7份快照均固定commit/URL/SHA；只读文本，没有安装或执行外部代码、没有下载模型权重。[来源清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/source_snapshots.json)。

- [Show-o forward](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/showo_model.source.txt:61)：共享logits后按任务切分CE，图像mask预测与文本移位预测分开。
- [Janus构造](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/janus_model.source.txt:190)：独立vision/gen-vision及aligner，后接同一个language_model。
- [UniFork构造](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/unifork_backbone.source.txt:1166)：num_shared_layers取总层数一半，新增gen_fork_layers及norm_gen；不是零参数开销。
- [Show-o2融合](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/showo2_model.source.txt:199)：语义通路和生成通路拼接后投影；不能简化成“同一个embedding自动兼顾”。
- [SRA loss](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/sra_loss.source.txt:82)与[训练](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/sra_train.source.txt:139)：同噪声样本构造更低噪声teacher输入，EMA不求梯度；读取代码确认额外前向、EMA更新及VAE依赖。
- [MultiMAE](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cv_unified_literature_20261001/multimae_model.source.txt:320)：显式gather可见token，再加global token并经encoder、各输出adapter解码。

## 已写好的8个实验配置

[配置目录说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。YAML是实验设计配置，当前全部execution_enabled=false、prototype_CPU_verified_GPU_entry_and_learning_curves_pending；**不是可直接提交的Volc资源YAML**。固定GPU入口与原池接入待完成，随后复用现有8卡，不新申请资源。

| ID | 方案 | 核心比较 |
|---|---|---|
| G00 | 共享主干＋理解/生成输出头 | 全部方案共同对照 |
| G01 | 任务专用末端norm | 小参数分工是否足够 |
| G02 | 末端两层小适配模块 | 共享表示与任务特化的权衡 |
| G03 | 全局register4 | 已有机制在双目标评价中的复查 |
| G04 | 主干跨段attention仅经4个register | 关系信息压缩是否损伤生成 |
| G05 | 与G04相同，16个latent | 检查瓶颈容量，不能只比较“有/无latent” |
| G06 | 自身EMA低噪声深层表示辅助 | 同时观察理解、生成及表征方差 |
| G07 | G06同teacher计算，辅助权重0 | 排除额外计算/训练路径带来的混杂 |

已实现d192/L4/H6的独立通用整数对象原型，非当前d768/L8免疫模型；不把其结果混入R20–R27排行。8配置CPU各3窗口/24人工记录只验功能，global8/micro3/固定LR与规划不同。计划500步与每50步曲线尚未执行；只看validation决定是否延长到1000，候选确认用3个seed。G04/G05仅限制主干跨角色attention，关系头仍拼接两角色池化特征；给定长度的人工原型不验证EOS/停止能力。真实任务是否提升仍须审核通过的数据和既定下游协议。

## 人工实验和评价合同

使用随机整数符号、多段内容及可复算的数学关系，不读取真实生物序列。理解端评价段间关系分类；生成端评价遮蔽预测和固定前向次数的迭代生成。理解输入中目标段完全遮蔽，标签不会成为输入特征；生成标签只用于损失与评价。

同角色/内位置随整段内容一起置换，检查等变/不变性；不能交换内容却不带role，误把语义改动当等价变换。去掉必要条件后的生成差异用于观察条件依赖，不把保留旧标签的任意反事实输入当新的有效准确率样本。分开报告IID、未见长度、干扰段和缺失输入，避免只记容易饱和的partial-mask平均值。

同时记录关系准确率、token CE、迭代生成准确率、条件移除差异、置换偏差、表示方差、可训练/总参数、显存和GPU秒。任务分支、teacher和更多latent都必须计入成本。dense注意力加mask不会自动减少矩阵计算，不宣称未测量的加速。

## 接续点与门禁

当前新方法进度：文献/源码阅读完成到上述范围 → 8份YAML计划完成 → 8原型实现完成 → CPU功能验收通过 → GPU入口与500步曲线待完成。CPU两环境各35个唯一案例覆盖，首轮worker的1项FP32汇总差异经定向修正通过，阈值和训练目标未放宽；原始packet独立重算和只读监督已完成。下一步实现固定通用GPU入口、核对来源并接到现有池，先测显存再按既定预算比较；已通过的Trainer运行包源码保持固定，不修改或重跑。训练/验证独立随机流和ID不认证内容严格去重。

现有R21/R24/R26验收看[统一接手状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。当前真实数据training_ready=false、两份审核证书缺失。此前停止的序列注释分支不恢复，独立CUDA算子测试仍暂停。关键阶段写总账、当前状态及本文件；未完成不标已完成。
