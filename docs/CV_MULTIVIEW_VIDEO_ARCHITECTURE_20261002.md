# 多图、多帧架构研究与下一轮候选

2026-10-02；状态：阅读论文、检查作者实现与形成方案，未将下列新机制加入正在运行的八模型。当前R20/R21/R24/R26均按同一十万条子集跑完整epoch。用户新增context数据要求另见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。

## 与既有研究的关系

已有文档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_VISION_EXPANSION.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_SEQUENCE_EXPANSION.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md 已覆盖MBT、Set Transformer、MultiMAE、register、EMA及adapter等。本次新增重点是多视角/视频组织方法，旧项只复核，不能重复记成首次发现。文献在视觉上的结果不证明迁移有效；以下候选是本项目推断。

## 原文与作者代码观察

|工作|读到的机制|可借鉴部分与限制|
|---|---|---|
|[VGGT](https://arxiv.org/abs/2503.11651)|多图共同建模；作者aggregator交替frame与global attention|分段内部计算和跨段计算交替。它的frame层覆盖整帧，当前R26同段半径32窗口并非同一设计；不能把2D位置编码、相机token或视觉预训练直接当成本项目已具备能力。|
|[TimeSformer](https://arxiv.org/abs/2102.05095)|空间/时间attention分解；代码把同一空间位置跨帧排成序列|可借分解计算思路。但不同链同编号位置没有天然对应，不能照搬固定位置跨链attention；应使用内容寻址或整段共享信息。|
|[SiamMAE](https://arxiv.org/abs/2305.14344)|两帧不对称遮蔽，可见帧作为条件，cross-attention预测另一帧缺失内容|与context保持可见、只监督目标的输入约定相近；不推导出链间存在视觉运动/逐位置对应，也不直接套95%遮蔽率。|
|[CroCo](https://arxiv.org/abs/2210.10716)|从同场景两视角进行cross-view completion；代码第一输入有mask、第二输入无mask|借用共享编码与目标查询读取条件的结构；原任务依赖两视角相关性，这与任意两链不能等同。需要单独检验条件是否被使用。|
|[VideoMAE](https://arxiv.org/abs/2203.12602)|视频冗余条件下高比例tube masking|可提示比较遮蔽模式，但属于训练目标轴，应与架构轴分开；不能对不对应的链复制时空tube假设。|
|[MBT](https://arxiv.org/abs/2107.00135)|延后融合或通过少数latent通信|先复用已有R25作融合层位置对照；硬瓶颈容量会限制细粒度信息，延后到更大改动组。|
|[Set Transformer](https://arxiv.org/abs/1810.00825)|集合交互与置换不变/等变归纳偏置|对完整“内容＋role”块重排检查应保持语义；不能把交换角色、打乱链内顺序也视为不变变换。|
|[MultiMAE](https://arxiv.org/abs/2204.01678)|多输入模态及多输出任务的遮蔽预训练|区分可见context与被监督target，为多读出头提供参考；不照搬其伪标签生成过程，不开展新序列注释。|

读取范围：以上论文摘要/方法描述；VGGT、CroCo、TimeSformer另外阅读作者代码的实际attention/encoder/decoder组织。VGGT的CVF PDF抓取失败，因此不宣称读完该PDF；以arXiv页面和固定作者代码为依据。源码只读存档，未执行或装依赖。

作者代码固定证据：

- [VGGT aggregator](https://github.com/facebookresearch/vggt/blob/a288dd0f14786c93483e45524328726ab7b1b4ce/vggt/models/aggregator.py)：frame路径用(B×S,P,C)，global路径用(B,S×P,C)，由aa_order轮流执行；另有首视角特殊token，不能据此声称全置换不变。
- [CroCo](https://github.com/naver/croco/blob/5d4dbc920b4cc0dac66bef0ce6876b58f1c82deb/models/croco.py)：同一_encode_image分别处理mask/no-mask两输入；decoder恢复第一输入查询并结合第二输入特征，输出第一输入patch目标。
- [TimeSformer](https://github.com/facebookresearch/TimeSformer/blob/a5ef29a7b7264baff199a30b3306ac27de901133/timesformer/models/vit.py)：temporal分支按同(h,w)收集各帧，再做逐帧spatial分支。这一位置对应前提不能直接用于多链。
- 本地SHA/commit清单：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/cv_multiview_research_20261002/sources.json。

## root取舍与测试顺序

第一优先仍是完成正在运行的8项，而非用早期loss提前删掉方法。共同250步中R26两个seed的AA CE较低，但仅消费8000/100000条，尚未完整epoch。当前验证只有37条context记录；总体指标主要反映其他记录，不能直接解释为关系理解增强。

下一轮优先考虑两类小改动：**分段全长attention与global交替**，用于区分R26收益来自分组还是局部窗口；**已有R25延后融合**，用于比较信息从第几层开始交流。保持同一总层数、宽度、输入、标签和loss掩码；层数量不因加入一个空间/时间子层而悄悄翻倍。若代码只改变dense mask，报告实际显存/耗时，不宣称稀疏加速。

其次评估 **target查询读取context特征** 的共享编码/cross-attention方案。context始终无序列CE，关系标签仍可作为监督目标；必须防止预测关系时把真实关系token原样留在输入形成答案泄漏。context应可接受目标任务反传，不因“context不算loss”而detach。参数量、额外前向、context缺失和多target情况要有对照。此项需要实现与独立验证，当前不属于可直接运行的R编号。

MBT小latent通信、任务专用norm/adapter、遮蔽策略作为后续储备，分别成组，不能一次叠加后归因。已有G00–G07是独立整数原型，不能把它们的结果补进当前真实模型图。视觉重建方法也不能证明理解/生成双目标已经都提升；至少要保留冻结的表征评价和独立生成协议，当前本轮没有该结论。

## 数据与实验轴

新context增强子集若成功为120000条，则同一组全部候选用相同120000条、同valid、相同真实训练seed，global32→3750步，重新固定3750步调度与源码版本。当前100000/3125组保持历史对照；跨组数据/步数/调度都变，不能称架构单因素收益。补充有/无context和binding标签分母；正例-only时不能报告二分类区分能力。不得以test选架构。

每轮8卡可安排共同baseline＋三个候选，各seed42/43；至少跑完一个epoch。仅因慢或初期loss落后不提前终止，遇非有限loss、OOM或语义错误则记录真实失败并修正后另开明确定义的对照版本。当前没有把新设计自动入队，也没有重启资源、修改运行中样本/源码或取消旧测试。
