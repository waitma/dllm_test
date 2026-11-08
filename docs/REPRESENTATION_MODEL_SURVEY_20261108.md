# 蛋白、抗体与TCR表征模型：借鉴点与本项目适用范围

调研日期：2026-11-08。依据原始论文、作者官方代码和模型卡；本次没有下载模型权重、安装依赖或启动训练/推理。目标承接 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_REPRESENTATION_PRIORITY_20261108.md。

## 结论

有成熟的序列基础模型，也有直接面向免疫受体表征的小模型。现有文献支持把“骨干结构”“如何汇聚成向量”“向量接受什么训练信号”分别研究。它们没有证明任何一项移植后必定超过V9。

优先借鉴SCEPTR的表征目标与读出消融、ProteinBERT的局部/全局表示交换、AMPLIFY的深宽配置；ESM-2/ESMC及IgBert/AbLang2适合作外部表征参照。新增训练目标必须与纯架构消融分开报告，不能同时修改后把收益全部记在架构上。

## 1. SCEPTR：最直接对应TCR表征问题

[作者论文v2](https://arxiv.org/html/2406.06397v2)；[正式论文](https://doi.org/10.1016/j.cels.2024.12.006)；[作者代码](https://github.com/yutanagano/sceptr)。

BERT式双向编码器，论文以TCR六个CDR为输入，用专门CLS向量表示受体；联合MLM与自对比学习。其few-shot特异性预测消融中，MLM-only明显弱于联合训练，而联合训练的mean-pooling版本仍较强，说明不能把收益只归于CLS。论文与官方仓库说明模型本身很小，不能将其结果解释为大模型规模优势。

借鉴：在固定骨干下分别比较读出方式与序列级目标，使用冻结表征的近邻任务检验距离空间。当前原模型dropout为0，同一输入无随机性地编码两遍不能直接形成有效训练视图。

约束：不照搬其CDR-only tokenization、CDR编号或新增输入CLS；保留现有全链V8输入与词表，可将可学习query汇聚头作为内部模块单独验证。训练目标若引入同记录多视图，只在内存中构造，原记录不改；增强是否保留任务语义需先核实。相同V-family不是已知功能正对，不同受体也不能默认功能不同；避免把来源、缺链和家族捷径当表征学习。

证据适用于其TCR任务；不自动外推至抗体或本项目真实下游。

## 2. ProteinBERT：明确区分局部残基表示与全局蛋白向量

[论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9386727/)；[官方代码](https://github.com/nadavbra/protein_bert)。

原模型约16M，维护局部表示与全局向量两条路径，通过全局query对局部位置做attention汇聚，再把全局信息广播回局部。该global attention不同于A5每个token对所有token的稠密自注意力。原预训练还包含GO注释预测。

借鉴：把链/受体级latent表示显式纳入骨干，让其逐层汇聚并与残基特征交换；可保留现有token输入与数据reader。只增加末端pooling头和在各层加入全局反馈是两个不同因素，应分别消融。

约束：没有GO标签时不声称复现ProteinBERT的训练目标或收益；原论文的全模型线性复杂度也不能直接套到仍含稠密自注意力的改造版上。约75M的重分配与性能需实测。

## 3. AMPLIFY：接近当前预算的现代纯序列编码器

[120M官方模型卡](https://huggingface.co/chandar-lab/AMPLIFY_120M)；[官方代码](https://github.com/chandar-lab/AMPLIFY)。

官方120M配置为24层、隐藏维640，使用RMSNorm、SwiGLU、RoPE，支持残基和蛋白向量。我们V11为8层/768维，已经具有这些归一化、激活和位置编码组件。

借鉴：约75M同参数下比较较深较窄与现有较浅较宽结构，而不是重新宣布已有组件为新架构。参数量接近不等于FLOPs/速度相同；下游指标、训练token数与GPU时间一起报告。

约束：其训练数据规模和配方不同；官方checkpoint可作参照，同预算自训才是深宽结构的受控比较。

## 4. IgBert/IgT5：原生配对与下游线性读出

[正式论文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012646)；[作者模型](https://huggingface.co/Exscientia)。

支持配对与不配对抗体。论文使用冻结语言模型embedding与正则化线性回归评估实际测量任务，配对版本的平均读出已能取得收益；不同性质任务并非始终由同一专用模型胜出。

借鉴：把是否联合编码双链、是否分链汇聚、整体平均与分链拼接分别比较。我们已经有联合编码与按角色分链mean再拼接的探针接口，不能把“使用配对链”本身当成尚未实现的新功能。

约束：原论文包含不同初始化、预训练数据和模型尺寸，其成绩不是对全局注意力或pooling单因素的因果证明。外部模型输入适配保持其官方词表，不把我们的自定义token ID直接喂给外部权重。

## 5. AbLang2：种系偏差是表征评测的重要提醒

[作者官方代码与研究摘要](https://github.com/oxpig/AbLang2)。

研究抗体序列中大量保守种系残基造成的语言模型偏差；支持配对/不配对抗体，并提供序列与残基编码。它主要不是一个“所有下游任务均胜出”的表征证明。

借鉴：区分家族可分性、保守残基拟合与功能表征；按V-family、CDR3及克隆/同源分组评估。我们已有family辅助头，但家族分类准确率不能替代功能表征指标。

待检验假设：family目标权重可能与某些功能任务存在取舍，需要独立消融才知道方向。目前没有证据认定它导致V11低于V9，也不在本次修改该损失。

## 6. ESM-2 / ESM C：通用表征参照

[ESM-2官方仓库](https://github.com/facebookresearch/esm)；[2024版ESMC-300M权重页](https://huggingface.co/biohub/esmc-300m-2024-12)。

ESM-2提供35M、150M等尺寸，适合充当规模附近的外部参照；ESMC-300M为另一通用编码器参照。官方仓库和权重组织目前已有迁移，应固定具体模型revision、分词器和提取层，不以名称代表同一份权重。

借鉴：固定任务协议比较中间层/末层、分链/整体汇聚与归一化的读出。外部模型的预训练数据和预算远不相同，外部胜负用于衡量实用表征差距，不作纯架构因果归因；也不假定其测试集已去污染。

## 7. 新近线索：recAMPLIFY（2026-10预印本）

[预印本v1](https://arxiv.org/abs/2610.26747v1)。

论文以155M独立参数的共享Transformer核心多次迭代，研究循环隐状态的汇聚和门控；与AMPLIFY-350M对照使用相同2T训练token。可借鉴的是参数共享下的有效深度与多层状态读出。当前仅作第二轮线索，未在我们数据上验证；减少独立参数不保证减少FLOPs或训练时间，不直接用论文收益预测75M/固定10%数据结果。

## 与当前实现的对应

- 代码 /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/dllm/pipelines/immune_llada/fixed_chain/unbinding_vhh.py 的receptor_vectors已经对H/VHH、L、beta、alpha逐槽mean并拼接；缺链产生零向量，需保留缺链控制。
- /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/dllm/pipelines/immune_llada/fixed_chain/v_family.py 有chain-mean上的family分类监督，不能声称当前完全没有序列级监督；它没有直接约束功能性受体距离空间。
- /mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/experiments/arch70m/evaluate.py 的旧representation是逐来源StandardScaler＋LogisticRegression，不能把它等同于上述论文的官方few-shot/任务矩阵。

## 推荐的验证次序（未启动新实验）

1. 先对齐V9/V11正式评测与检查点，保留现有权重，建立分层/分链读出和外部参照的冻结表征比较。
2. 纯架构阶段分别比较：专门全局汇聚/局部反馈；约75M深宽重分配。仍使用原MLM/family目标，统一读出。
3. 训练目标阶段固定骨干和读出，单独比较序列级自对比信号；与family权重消融分别报告。训练视图和采样须避免已有捷径与无效正负对。
4. 参数共享递归或结构对齐列为后续候选；本轮不要求新增结构、GO标签或输入格式。

上述均是待验证假设，尚无新的本项目性能数字。继续原V8记录、固定10%索引与探索分支，单卡大batch探索/多卡独立trial的执行预算待实际协议冻结，不自动开始长训。
