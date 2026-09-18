# TCR baseline 与下游评测纠错记录

> 更新：2026-09-15（UTC）。
> 最新用户决定：结果页按 AB 在前、TCR 在后整理，逐任务列 baseline、明确 checkpoint 的 Ours 和单列消融；旧全文完整归档，见 §9.28。本文继续只维护关键问题、协议与决策，不复制排行榜数字。
> 当前本轮终态：指定 v5 49000 的原定 10/10 项已完成并通过各自结果核对，最终 comparison 已收口；精确时间与执行证据见 [任务账本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，数字见 [RESULTS 下半部分](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-binding)。旧 8/10、9/10、失败/恢复和运行中记录均为历史快照，不是当前状态。
> 输入准则不变：待预测关系只输入 mask；读取原始评测文件后运行时复用训练侧补全，不落地预处理后的评测副本。任务专属 gate 和已构建 T1–T4 的逐项审查要求见 §9.8 / §9.21，不能把 T1 的 mask 策略无条件套到其他任务。
> 边界不变：T3 是本地对照，不冒称论文同版复现；T4 当前只生成 β core，未知 α/框架/CDR1/2 保持 X，配对 αβ 生成仍延期。全参数微调、直接关系 token 评分和完整预训练去污染未因本轮完成而被标为完成；T4 去污染范围解释见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
> 历史 others/AS 三轨与 29k 对照的实现/验证见 §9.2 / §9.6 / §9.7 / §9.27；历史 §0.1a–c 与旧协议数字移入 [RESULTS 归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，不用于补当前 checkpoint 的缺项。后续 TCR 关键改动仍须在本文追加登记。

## 1. 本文负责什么

本文按用户要求，作为 TCR 下游纠错问题与修正方案的集中记录。已执行协议的入口仍见
[T1 任务文档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)，旧结果仍保存在
[RESULTS 整理前归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)；当前数字见 [RESULTS][results]。这些位置只链接本文，不复制问题台账。

本文中的“已确认”指代码或既有产物能直接支持的事实；“待验证”指还需要运行或追溯训练 provenance。
“建议协议”是待实现的实验设计，不能写成已经运行、已经变好或已经获得论文可比结果。
台账中的 12 个编号是审计/整改事项，包含历史记录、现有设计限制和待验证风险，**不是 12 个新发现的 bug**。
历史事实沿用原文权威记录，本文只维护 TCR 纠错增量、代码变更、验收与未完成项。

本轮结论：

- 固定正类 recognition token 是输入语义问题。虽然所有样本都放同一 token 不会泄露逐条真值，仍然把“已结合”的条件交给了模型；在 v5 已训练关系 token 的背景下尤其需要纠正。
- 现有 Ours T1 是冻结模型后的监督探针，不能代表端到端监督适配后的性能，更不能直接推导“预训练模型的 binding 能力上限”。
- 官方 baseline 采用不同训练方式，不能笼统称作“全部全参数微调”。需逐模型记录哪些模块训练、哪些冻结。
- 建议补充 Ours 与 ESM2 的同数据全参数微调对照，同时保留纠正输入后的冻结探针；直接预测关系 token 单列。
- 对既有固定 recognition-token Ours T1 结果保留原始数字和 provenance，但不作为纠正后 v5 协议的 headline。不能据此断言旧数字必然虚高或必然虚低。

## 2. 已核实的当前实现

### 2.1 Ours T1 实际计算链（已切换运行时修正协议）

入口是 [run_retrained_ours.py][ours-run]，wrapper 是 [run_immune_fusion_repr.sh][repr-run]。

1. 直接读取官方 NM2025 retrain.zip 的五折数据；默认 `--track cdr3b` 输入 Epitope、CDR3B；`--track cdr3ab --neg-source AS` 读取 others，再加入其真实 CDR3A，其他字段忽略。Affinity 仅作二分类监督标签。
2. [TCRBindingQueryProtocol][query] 在内存中调用训练侧 `_receptor_chains`，补为 beta-first β/α 双链；paired 轨道保留两条真实 CDR3，β-only 的 α 则全为占位。只读取冻结 profile，不读取 prepared 训练样本、不预处理/改写 benchmark CSV。
3. grammar collator 完成编码/词表 remap 后，[T1 专用适配层][query] 将唯一 peptide→TCR recognition 位置替换为 **decoder 空间**的 mask，并移除 reconstruction labels。真实 Affinity 不进入记录/encoder/模型输入。
4. 完整 ESMC＋LLaDA 做单次前向；提取最后一层，只对 attention AND residue AND NOT synthetic 的位置全局平均；当前 270m decoder 得到 768 维向量。
5. 每折仅从训练子集拟合 z-score，再训练 768→256→128→1 的 MLP；ESMC、fusion 和 LLaDA 均冻结。
6. train 内分层留出 10% validation；BCE loss，Adam，验证 loss 早停；各折分别评 seen_test、seen_independent、unseen_independent，最后取 mean±sample SD。

当前输入示意（正负样本相同；X 表示缺失区域的 synthetic 占位）：

```text
<prots> <pep> {peptide} <protd> <mask> <prots> <tcr> {β补全链} <chainsep> {α补全链或占位链} <protd>
```

`<mask>` 在 fusion 模型中使用其 `_decoder_mask_token_id`，不是硬编码的 ESMC mask ID。
修正前通过 [GrammarEmbedder._record][record] 直接构造单条 CDR3β，且 [renderer][recognition] 默认回退 `<binding>`；
该共享路径未全局修改，旧协议及其他任务不被本次 T1 修正静默重定义。
若存在 MHC→peptide presentation relation，它是另一条边；当前 T1 适配器会拒绝这种扩展布局，不能一把全遮。

### 2.2 ESM2 当前做了什么

[ESM2Embedder][esm-embed] 使用 eval 模式及 no_grad 提取固定特征。
[旧 T1 run.py][legacy-run] 分别编码各列再拼接，训练 StandardScaler＋LogisticRegression；
[run_nm2025.py][nm-run] 则提供冻结特征＋logreg/MLP 路径。这些都不是端到端微调。

盘上存在旧 [ESM2-150M 产物][esm-old]，其 columns 为 cdr3b、cdr3a、peptide；
不能拿它冒充 NM2025 cdr3b/AS 五折、同 MLP 预算的 ESM2 对照。
本轮在 retrained 输出顶层未发现 ESM2 命名目录；这不排除其他自定义 tag，完成性仍需逐份 manifest 核查。
现有五折 Ours runner 也没有通用 ESM2 backbone 参数，不能只换 checkpoint 路径就宣称接入成功。

ESM 官方提供“提取 embedding 后训练监督预测器”的示例，但示例是 ESM-1 的变体预测，
不能用它证明“所有 ESM2 TCR baseline 都冻结”或“ESM2 不适合全参数微调”。[ESM 官方说明](https://github.com/facebookresearch/esm#supervised-variant-prediction---training-a-classifier-on-the-embeddings)

### 2.3 NM2025 不止 peptide＋CDR3β：官方输入与本地接入范围

论文将数据分为 **CDR3β-only** 和 **CDR3β+others**，各自有 original-model 与 retrained-model 设置。
retrained 的两类数据都有 seen test、seen independent、unseen independent；不能把不同数据轨道的数字当成同一测试集。
β-only 还比较 AS/PS/HS 负样本来源；多特征轨道使用 AS。
[官方论文](https://www.nature.com/articles/s41592-025-02910-0)；
[官方输入字段说明](https://github.com/SuoLab-GZLab/TCREpitopeBenchmark#input-file-format)

| 数据轨道 | 文件提供的信息 | 当前 Ours 五折 runner |
|---|---|---|
| CDR3β-only | Epitope、CDR3B、Affinity；CDR3B 是 β 的 CDR3 片段，不是完整 β 链 | 已接入；默认 AS，可选 PS/HS |
| CDR3β+others | 上述字段＋CDR3A、TRAJ/TRBJ/TRBV/TRAV、MHC、LongB、LongA | 已接入并完成 AS 五折，仅使用 Epitope/CDR3B/CDR3A；不等于使用全部 others 特征 |

本地只读核查对象：[retrain.zip](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark/retrain.zip)，
MD5 `5cf77befd7a07e0cb359540f050fb7b4`（与原下载记录一致）。其中 `_others_seen/AS/` 有 5 个 train、5 个 test 和 1 个共享 independent CSV；
`_others_unseen/AS/` 有 1 个共享 independent CSV。多特征数据实际核查如下（行数含正负，不是唯一 TCR 数）：

| 多特征 AS member 后缀 | 行数 | 正例数 | SHA256 |
|---|---:|---:|---|
| `_others_seen/AS/1_1_1train.csv` | 6,824 | 3,412 | `06a7d852ab1bbcca3d127992958ccbbce65a5b165d51897178b61edd44ad2402` |
| `_others_seen/AS/1_1_1test.csv` | 1,710 | 855 | `024d311d1e8833c74ad1b610dc4164c08b36ba43f3a318c6d4d202923cc325d5` |
| `_others_seen/AS/1_1_1independent_test.csv` | 626 | 313 | `103857c58d94f0aa27065ec4f9ff6d3d05ca701850d2b609889b800cd456297a` |
| `_others_unseen/AS/1_1_1independent_test.csv` | 1,378 | 689 | `cb4d5d98bbe96de049e65163d758e4f18813e8a40fb61bf1462f7bfe29360e87` |

接入边界：

- 单独设多特征 track，不擅自给 β-only 数据检索真实 α/MHC，也不与其合并分数。
- LongA/LongB 是官方提供的长链序列字段；是否重建、来源及缺失值规则仍需审计，不能仅凭字段名就宣称是实验测得完整序列。
- MHC 列可能是等位基因/类别文本，不是可直接喂给蛋白 encoder 的氨基酸序列；如转序列，需固定映射版本、缺失策略与 provenance。
- 真实可用 α/β CDR3 保留，缺失部分才用训练侧 X 补全；paired 无 MHC 布局已做 CPU recognition-mask 验收，pMHC 布局仍未接入。
- β-only 的补全 α 全是占位符，并不等于已经利用了多特征轨道的真实 α 信息。

## 3. baseline 训练方式：按模型核实

先区分两件事：

- **官方发布前怎样训练**：依据官方 notebook/源码和对应 artifact。
- **本项目这次做了什么**：当前 [run_retrained_baseline.py][baseline-run] 加载官方每折 retrained checkpoint 做推理，未在本地重新训练 baseline。推理时 eval/no_grad 不说明它发布前也冻结编码器。

NM2025 论文包含 original-model 与 retrained-model 两条设置，并区分传统机器学习与深度模型。
官方仓库也把 original prediction、model retraining、retrained prediction 分成独立模块；
不能从“retrained”这个名称推断统一的全参数训练规则。
[论文](https://www.nature.com/articles/s41592-025-02910-0)；
[官方仓库](https://github.com/SuoLab-GZLab/TCREpitopeBenchmark)

以下是本轮核对的八个当前 retrained baseline。结论基于签入的训练代码，
并不等于已经恢复了每个发布权重的完整训练日志。

| 模型 | 已核实的训练代码 | 能下的结论与限制 |
|---|---|---|
| ATM-TCR | [notebook][atm-nb] 用 Adam(model.parameters())；[attention.py][atm-attn] 的 embedding 显式 freeze=False | 序列表示网络＋分类部分参与监督训练；不是冻结特征＋外部头 |
| NetTCR | [notebook][net-nb] 构建 CNN，compile 后 mdl.fit | 训练任务网络；不等同于微调一个通用蛋白预训练大模型 |
| TEINet | [notebook][tein-nb] 加载两套 TCRpeg encoder 后用 Adam(model.parameters())；[model.py][tein-model] 将 encoder 注册为子模块，forward 无 detach/no_grad | 编码器＋投影分类网络有可训练路径；不能把它描述成固定 embedding 探针 |
| ERGO-lstm | [notebook][ergo-nb] 创建 DoubleLSTMClassifier，用 Adam(model.parameters()) | 从任务网络学习序列表示与分类 |
| ERGO-AE | 同一 [notebook][ergo-nb] 的 retraining 分支设 train_ae=True；[ERGO_models.py][ergo-model] 仅在 False 时冻结 AE | 本地签入的 retraining 路径允许预训练 AE 与肽 LSTM/分类器共同训练；不能只看 inference 构造时的 False 就认定训练时也冻结 |
| TEIM | [notebook][teim-nb] 的 SeqLevelSystem 训练 scripts.model_raw.TEIM；[model_raw.py][teim-model] 冻结 epitope AE，包含可训练卷积/交互/输出模块 | 是部分模块冻结的任务训练；不是“全模型冻结”，也不是严格全参数更新 |
| epiTCR | [notebook][epi-nb] 对 BLOSUM 特征拟合 RandomForestClassifier | 监督训练随机森林；没有“解冻蛋白 LM 全参数”的对应操作 |
| TCR-H | [notebook][tcrh-nb] 对描述符做处理、选择特征并拟合 SVC/GridSearchCV | 监督训练 SVM；当前推理 wrapper 另需训练 CSV 重建选择出的特征列 |
| ESM2（项目通用对照，非上述八模型之一） | 见 §2.2 | 已有冻结特征路径；同 NM2025 五折的冻结/全参数两套对照仍需接入与验收 |
| Ours（当前实现） | [run_retrained_ours.py][ours-run] 明确冻结整个 backbone，仅 MLP 更新 | 仅代表当前 readout＋监督探针，不是端到端 binding 微调 |

后续每条实验必须记录 backbone 名称/版本、预训练来源、总参数与可训练参数、模块冻结表、
训练数据及选模规则。不能单凭 optimizer 收到 parameters() 就认定每个参数实际有梯度/发生更新。
Ours 参数预算应包含 ESMC＋decoder＋fusion＋head，不能用“270m decoder”代替总模型规模。

## 4. 关键问题台账

| ID | 优先级 | 问题 | 当前状态 | 关闭条件 |
|---|---|---|---|---|
| TCR-01 | P0 | recognition 输入固定为正类；unknown 也会回退成 binding | T1 专用路径已修，CPU 通过；真实模型评测待做 | label-blind 输入，relation 查询位置明确 mask；通过标签置换不变性检查 |
| TCR-02 | P0 | T1 CDR3β 单链构造未接 v5 受体补全；补全后旧 pool 又会平均 synthetic X | 运行时补全＋observed pooling 已实现，CPU 通过 | 固定 profile、CDR3 定义及 beta-first 布局；只 pool 真实观测残基 |
| TCR-03 | P0 | 输入/renderer/pooling 改动后旧特征缓存仍可能命中 | 特征/头/预测/汇总协议隔离已实现，CPU 通过；不是已发生误复用的证据 | 缓存及 head provenance 含协议/模板/补全/pooling 版本与指纹 |
| TCR-04 | P1 | 冻结探针与不同训练范围的任务模型放在一起，容易被解释为预训练方法优劣 | 已确认，解释与实验设计待修正 | 冻结、全参数、原生监督和关系预测分列；补同预算适配对照 |
| TCR-05 | P1 | ESM2 含 CLS/EOS 的 pool 与 Ours residue-only 不一致；分链特征与联合特征不同 | 已确认，未修 | 显式 pooling/interaction 配置；不能把差异都归因于 backbone |
| TCR-06 | P0 | 直接套现有提特征函数无法做全参数微调 | 冻结探针关闭梯度是正确实现；全参数新路径尚未实现 | 骨干参数有梯度且实际更新；冻结对照参数不变 |
| TCR-07 | P0 | unseen 只说明下游训练未见表位，不能证明预训练未见；近重复与补全 provenance 仍需追踪 | 历史审计存在；本轮未重跑 v5 全量审计 | 两层泄露报告＋输入记录/数据/补全 profile 版本 |
| TCR-08 | P1 | AUPRC 估计量、fold 聚合、正负比例与测试行覆盖存在混用风险 | 部分已有实现/历史审计，待统一验收 | 固定主指标，发布逐行预测、覆盖率、fold 统计 |
| TCR-09 | P1 | 部分官方 notebook 把 test 参数接入 validation/选模；发布 artifact 实际选模过程不完整 | 代码风险已确认，artifact 影响待证 | 逐 baseline 记录选模数据；新本地适配仅用 train 内 validation |
| TCR-10 | P1 | 关系 token 脚本默认测 prepared valid，与独立 T1 benchmark 不同 | 已确认，独立数据接入未完成 | 相同 benchmark 行集、label-blind mask 与完整指标 |
| TCR-11 | P1 | 文档/注释与实际实现有漂移，旧 headline 容易被继续引用 | 已确认；本轮新增状态警示与链接 | 旧结果正确标注，新协议验收后替换有效结论 |
| TCR-12 | P2 | T2/T3/T4 与 T1 共用组件，修改有跨任务影响 | 待专项审计 | 各任务输入、监督范围、池化/生成、去重与 baseline 训练方式分别验收 |

### TCR-01：预测位置不能预先断言结合

用户已明确拒绝将待预测 recognition 的 `<binding>` 作为输入。T1 新路径已实现：
保留关系位置，在训练分类器和推理时一致使用 decoder 的 `<|mdm_mask|>`；
真实标签只交给监督 loss/metric，不能先生成带真值的模型可见特征。

不要只改 `labels={"relation": "unknown"}`：
[recognition fallback][recognition] 会再次把它变成 binding。
也不要直接删 token 或全局修改训练 renderer 的默认值；这会改变位置/训练语义，
尤其共享工作区里仍有 v5 长跑。应在下游的查询构造层定义明确的 recognition query mask。

v5 当前只有明确监督记录才有 relation_target_mask；[现有 _record][record] 没有设置这类监督 metadata。
因此“直接套 relation_target_mask”可能得到空 mask，必须从查询结构确定 peptide→TCR 目标边并断言其数量，
不能从 y 值决定目标位置，也不能用包含所有 relation token 的 relation_token_mask 一把全遮。
实现先断言链角色恰为 peptide/beta/alpha、grammar 为 tcr_peptide，再断言唯一 relation 位置；
只有在这个受限布局下才将 relation mask 解释为 recognition query。含 MHC 的记录直接拒绝，不错误遮盖 presentation。

验证要求：

- 同一序列对分别附 y=0、y=1 或隐藏 y，模型实际收到的 input_ids、encoder inputs、所有可见 masks/positions 完全相同。
- recognition 位置必须是 mask；gold 只用于 loss。MHC→peptide presentation token 按任务语义处理，不能被误当分类目标。
- 特征抽取与预测对外部标签置换保持不变；指标会随标签置换改变。
- 保留旧 fixed-binding 版本仅作诊断消融，使用独立 tag，不能继续作为纠正后的正式结果。

### TCR-02：补全必须保持信息边界

[修正前 T1 record][record] 直接把 CDR3β 塞入 tcr_beta；
v5 的 [complete_tcr_chain / _receptor_chains][completion] 会把残缺受体补全为 beta-first 双链。
这些输入不是同一种布局。

现已在读取 CSV 之后、构造模型 batch 之前复用训练补全函数，不允许引入测试样本未提供的真实 α、V/J、MHC 或检索来的配对信息。
NM2025 的官方字段示例及论文的 C/F 规范化说明支持 junction 约定；默认显式 `--cdr3-format junction`，其他来源可指定 core。
这是数据集级配置，不按单条首尾字符推断 convention；junction 内部的锚点安置复用训练函数，不重复添加 C/F。
冻结与 v5 产物一致的 region-length profile，记录其来源和 hash；不从测试集拟合补全长度。
每条样本的补全随机性由与标签无关的稳定 key/seed 决定，不能正负分支或读入顺序不同就采出不同占位模式。

[旧共享 pooling][pool] 只用 residue_mask；T1 新适配层已显式排除 synthetic_residue_mask：
attention_mask AND residue_mask AND NOT synthetic_residue_mask。
占位 X 可作为缺失上下文进入 attention，但不应在 readout 时与真实残基等权平均。
mask-query 是否纳入 readout 是另一项配置，不能偷偷改变旧 global-mean 基线。

### TCR-03：缓存需要编码评测协议

[缓存检查][cache-check] 修正前仅核对 checkpoint hash、序列对 hash、行数及固定 feature_source/pooling 字符串；
缺少 renderer/template、recognition policy、completion profile、synthetic pooling 等版本，存在协议改动后复用旧产物的风险。
没有证据证明历史运行已发生这种误复用。

新路径已给特征与 head 统一加入 protocol_id、输入实现/renderer 等源码指纹、补全 profile/seed 规则、
pooling mask、模型权重及 tokenizer/config/data hash；预测缓存校验 head/测试 member，汇总拒绝混合旧协议。
上述协议身份改变会拒绝复用旧缓存；新实验应使用独立 tag。
不删除旧产物，以便量化纠错前后差异。

### TCR-05 / TCR-06：公平读出与可训练路径

[ESM2 embed][esm-embed] 当前 attention-mask mean 包含 CLS/EOS；
短 CDR3/peptide 上不能凭注释认为其贡献可忽略。新的 residue-only 读出需要正确移除 special tokens，
记录截断/未知字符/空串策略，拒绝静默用 A 冒充缺失输入。
ESM2 分别编码 β/peptide 后 concat 与 Ours 联合 attention 也是架构差异，必须披露。
冻结探针控制 MLP 深度、正则、搜索预算，并报告因输入维数不同导致的 head 参数量差异。

[load_fusion_for_eval][loader] 和 FusionGrammarEmbedder 显式冻结全部参数；
[model.last_hidden_state][no-grad] 还带 no_grad 装饰器，外层 pool 也关闭梯度。
仅调用 model.train() 或加一个“全参数”命令参数无法建立训练图。
全参数适配需要独立可求导 forward、明确解冻模块、optimizer 参数组和梯度/参数变化验收；
不能缓存训练前的固定 embedding 再只训头，并称其为 full finetuning。

### TCR-07 / TCR-08 / TCR-09：切分、选模与指标

- 检查两层泄露：下游 train→test，与预训练 corpus→下游 test。
  exact pair、CDR3 exact/near-duplicate、epitope overlap 分别报告；unseen 的定义限定到具体层。
  历史污染与近重复证据见 [T1 历史审计][old-audit] 和 [RESULTS][results]，本轮未重新计算。
- AS 在论文中指 antigen-specific 来源的负样本方案。新对照直接复用官方给定行，
  不自行换负样本、正负比例或重划 unseen。AS 负例不能全部宣称是实验验证的不结合。
- 五个 heads/finetuned models 分别在三套 test 上打分；不先把五折预测混起来当主指标。
  AUPRC 主列统一取项目 precrec 实现，sklearn AP 单独报告；AUROC、样本量与正例率并列。
  AUROC 随机参考为 0.5，AUPRC 随机参考随正例率变化，不能无条件写 0.5。
- 逐行验证 row_id、多重集合、预测数量、有限值、正负覆盖。ERGO-AE 的官方 tail-drop 已在
  [baseline runner][baseline-run] 明确配置；官方复现保留并披露，对等本地比较另列统一覆盖版本，不能暗删难例。
- 本轮确认 [TEIM notebook][teim-nb] 将 testfile_path/test_name 构成的 val_set 用于 valid/auc_avg early stopping。
  这证明源代码存在把传入 test 用于选模的路径，尚不能证明每个发布 fold artifact 实际用的是哪份文件；
  不据此宣布论文整体失效，也不以“严格复现”为由让新实验在测试集选 epoch。
- 原生 baseline 的论文值与本地官方 artifact 结果保留 provenance；seen 两列的既有系统偏移不能在未证实前归因为某一种原因。
- 新本地微调的 optimizer/LR、epoch/early stopping、阈值、readout、mask policy、补全策略都只在训练内 validation 决定。
  预训练 checkpoint 可按预先定义规则或预训练 validation 选；不能看独立 test 逐个挑 checkpoint。

## 5. 纠正后实验协议（others 冻结探针已完成；其余对照待做）

执行状态：Ours 的 others/AS CDR3αβ 冻结探针已完成，具体配置与验证见 §9.2；同数据 ESM2 对照、全参数监督适配、直接关系 token 预测仍未执行。不能把一个已完成实验扩大成整张对照矩阵已完成，也不能继续统称“实验尚未执行”。

全参数微调值得作为正式的监督适配实验补充，但不保证效果一定改善。冻结结果仍有研究价值，
前提是纠正输入并把它明确标成 frozen probe。先修输入/切分/缓存，再比较训练方式。

| 实验 | Ours | ESM2 对照 | 回答的问题 |
|---|---|---|---|
| 冻结探针 | label-blind masked recognition；冻结 ESMC＋fusion＋LLaDA，仅训 MLP | 冻结 ESM2，真实残基 pooling 后同类 MLP | 冻结表示能否支持下游分类 |
| 全参数监督适配 | 从相同预训练 checkpoint 开始，训练 ESMC＋fusion＋LLaDA 及分类头；mask 查询输入一致 | ESM2 encoder＋分类头共同更新 | 给模型同等下游适配机会后的任务性能 |
| 直接关系 token 预测 | mask recognition，score=logit(binding)−logit(nonbinding)，不训练下游头 | ESM2 没有原生关系 token，不强行伪造同类 zero-shot 分数 | 预训练的关系预测能力 |
| 原生监督 baseline | 保留官方模型/论文结果，列清原生冻结范围与训练方式 | epiTCR/ATM-TCR/TEIM 等按原生协议 | 与任务方法的性能对照，不能视作控制变量实验 |

可选的 decoder-only/LoRA 属单独适配臂，不能标为 full finetuning。
全参数主张需核查 ESMC、fusion projection/norm、LLaDA 与新 head 的实际梯度/更新；
未参与该前向的输出参数应明确披露，不要求为制造“全参数”名义强加无关 loss。

建议首批通用 LM 对照包括 ESM2-150M、ESM2-650M，另可用 ESMC-only 隔离 decoder 的增益。
这些是待安排候选，不表示权重路径、运行环境或显存已验证。
同折、同训练行、相同输入可用信息、相同验证选择原则及相当超参搜索预算是核心；
不强求所有模型同学习率，记录学习率网格、总参数、可训练参数、训练时间及种子。
模型名/任务名必须编码 frozen/full/partial/relation-zero-shot，不能只写“retrained”。

[score_binding_relation.py][relation-score] 可复用 mask/打分机制，但目前默认从 prepared valid 抽样，
并可能按来源与正负类配额抽取，不能当作完整独立 benchmark。
接入 NM2025 后需覆盖原行集、补 AUPRC、报告真实正例率；TCR 专项不能把抗体记录混入总分。
“zero-shot”仅指没有新增下游监督适配，预训练污染仍需单独检查。

## 6. 实施顺序与验收清单

### 阶段 A：固定问题定义与数据（others 第一版已锁定；预训练 overlap 待核验）

- [x] 为本次独立 v5 checkpoint 建评测快照并验证 hash，避免长跑 top-k 清理中途删除输入。
- [x] 锁定本次 others 官方五折、三套 test 与 AS 行集；记录 archive/member hash、row_id、正负统计。
- [ ] 确认 CDR3 provenance、补全 profile 来源、测试数据与预训练数据 overlap。
- [x] 冻结本次 protocol_id 及训练内 validation/选模规则。

### 阶段 B：修输入与缓存（CPU 与真实 GPU 前向通过；显存/批次一致性专项待做）

- [x] T1 query 不输入 gold 或固定正类 recognition；unknown fallback 无法绕过此约束。
- [x] 查询构造不接收标签；CPU 检查补全/打分对标签置换、批次顺序不变（打分前向使用测试替身，未加载真实模型）。
- [x] 补全 synthetic 标记贯穿 collator；readout 排除 synthetic 和 special/pad。
- [x] 协议变更使旧 features/head cache 拒绝加载；预测及跨折汇总也检查身份。
- [x] 160 条官方正负样本检查渲染后的 token、encoder 输入、关系位置；长度超限报错，不截断。
- [x] 固定真实 v5 checkpoint 的 GPU 前向；完整特征为有限值，batch16 未触发 OOM。
- [ ] 显存峰值测量及真实模型不同 batch size 的数值一致性专项验收（本轮未做，不能用无 OOM 代替）。

### 阶段 C：建立训练方式对照（尚未完成）

- [ ] 修正后的 Ours frozen＋ESM2 frozen 同折跑通。
- [ ] 独立 full-finetuning 路径通过可训练参数、梯度、实际更新验证；不复用 no_grad 提特征路径。
- [x] 本次 frozen track 每个 fold 独立初始化 MLP，从同一冻结预训练 checkpoint 的特征开始，不继承前一 fold 的 head/优化器。
- [x] 本次 head/特征/预测写入独立下游目录，不覆盖预训练 checkpoint 或旧结果；full-finetuning optimizer 路径仍待实现。
- [ ] frozen/full/partial/zero-shot 各自使用明确 tag，完成后再分别收数。
- [x] 本次正式 GPU 作业通过项目 eval_jobs 提交并完成；其他训练方式不因此视作已完成。

### 阶段 D：收数与解释（others 第一版已完成；因果对照待做）

- [x] 本次五折全部完成才报告 mean±sample SD；缺 fold 不填完整结果。
- [x] 本次发布逐行预测、AUPRC 估计量、AUROC、样本覆盖、正例率与资源预算。
- [x] 旧 fixed-binding Ours T1 仅作历史/消融；不删除其原始 artifact。
- [x] 本次所有指标来自同一 checkpoint、协议和 summary。
- [ ] 公布纠错前后差异后，才讨论 mask/补全/full finetuning 是否改善性能。

## 7. T2 / T3 / T4 后续审计边界

| 任务 | 后续需核对 | 本轮边界 |
|---|---|---|
| T2 clustering | 表位标签不能进入 embedding；全局/分链池化、聚类超参的选取集、近重复、baseline 是否用标签适配 | 输入/关系与标签边界 CPU 初查通过；v5 补全、真实模型、预训练 overlap 和完整 baseline 审计待做，见 §9.9 |
| T3 representation / few-shot | 支持/查询集与预训练去重；每任务 k、采样、种子；是否在 query/test 上拟合 scaler/PCA；baseline 的训练标签范围 | 输入接线、support/query 评分 CPU 初查完成；v5 适配、字段特定去重、真实模型与完整 baseline 审计待做，见 §9.10 |
| T4 generation | 条件表位与生成目标是否隔离；encoder 是否看到待生成真值；长度/解码预算/重排标准；oracle 独立性与训练重叠 | 当前链路静态初查及部分旧 metrics 核对完成，双路动态输入/版本/重评等 gate 尚未完成，见 §9.12 |

T4 的任务可能是“给定期望 binding 条件生成 TCR”，其中使用期望关系 token 可以是合法条件；
T1 是“预测未知 recognition”，不能直接照搬该条件。不能为了修 T1 一概禁止所有生成任务使用 binding token。

## 8. 审计快照与证据

以下是首次静态审计的历史快照，不代表 §9 修改后的源码 SHA256。
本地官方基准 checkout：ec832b47ec3a566d76ecdddb99e5510a6d96ec4e。
主仓库 HEAD：d3daa3585669be0b0aebb6e5a3d474347b35cfbb；工作区已有未提交改动，
故 HEAD 不能单独重现本轮读取的代码。以下为读取时的 SHA256（不是测试通过证明）：

| 文件 | SHA256 |
|---|---|
| [model_api.py][record] | e839326cb1e188e2a3717d366c8875a1e9a5a316712723e3a02d59b95a63fb80 |
| [run_retrained_ours.py][ours-run] | ae5e9f5da9aa14746fdac4da2f57b7304e16b8c8561f82fc3128b65eb6131db0 |
| [grammar.py][recognition] | 4670fcb4d2ee9e98af506566548e8e57a403fd7ffc626553513a32ac5f3044e7 |
| [protein_fusion_model.py][no-grad] | faef335c2f758fc4745caaba3b499badbb355fee3f6eb0ecd995b7ba6d9a20cc |

首次静态审计只审阅代码/notebook 的 source（未把 notebook 输出视作重新执行结果）、
既有 ESM2 metrics，以及官方论文/README。没有运行模型 forward、梯度检查或全量泄露审计。
历史 RESULTS 中的“瓶颈必然在 readout”“绝对指标是下界”等推断，不能替代上述待做对照。

## 9. 关键改动日志（后续持续追加）

### 9.1 2026-09-13：T1 运行时输入修正与 NM2025 数据范围核查

本节保留第一阶段的实施快照；其 CPU 测试数、代码摘要与“尚未运行 GPU”说明只对应当时状态。随后新增的 others 输入及已完成 GPU 测试见 §9.2。

用户决策：relation 位置只输入 mask；原始文件读入后即时补全到训练范式，不预处理评测数据；所有关键变更在本文登记。

**实现范围**：

1. 新增 [query_protocol.py][query]，由 [run_retrained_ours.py][ours-run] 使用。复用训练 `_receptor_chains`，不修改训练 renderer、source adapter 或 prepared 数据；T2/T3/T4 不切换协议。
2. 只读 v5 manifest 的冻结 profile，摘要为 `be50d06bb612b6559625b66ce812df9163675351748c1b699cc2b920dd168353`。缺文件、摘要不符、不完整分布均报错，不回退另一套 profile。
3. 补全随机 key 由规范化 CDR3β 的 SHA256 决定，再调用训练采样器；不含标签、epitope、fold 或读入行号。相同受体在不同 batch/顺序下占位一致；这是相同训练范式，不要求与某条训练行的随机占位逐字相同。
4. collator/remap 后、模型前向前替换 recognition 为 decoder mask；移除 collator 的 reconstruction labels，监督标签只交给 MLP/loss/metric。已知序列不随机加噪。
5. X 占位参与 attention，但不进入 observed-residue mean；真实输入中的歧义 X 与 synthetic X 按 metadata 区分。空/全 X/非法序列报错，不用 A 替代、不静默裁剪。
6. 特征 manifest 增加完整 input_protocol 及指纹；head 记录 feature_identity；已有预测与跨折汇总也必须匹配身份，避免换输入后沿用旧输出。原始旧缓存/权重/指标未删除或覆盖。
7. runner 增加 `--completion-manifest`、`--cdr3-format {junction,core}`、`--max-length`。wrapper 支持 `T1_COMPLETION_MANIFEST`、`T1_CDR3_FORMAT`、`T1_MAX_LENGTH` 环境变量，只影响 T1。
8. 官方论文/README 与本地 archive 确认多特征轨道，字段与接入边界见 §2.3。**本次不声称已支持多特征评测**。

**验证与边界**：

- pllm 环境运行 [新增回归测试][query-test] 加既有 receptor-completion / relation / role-resolution 测试，**54 passed**。
- 覆盖：真实残基守恒、训练补全函数一致、junction/core 区分、顺序/标签不变、native RemapCollator 的 decoder mask 空间、synthetic/pad/query 不参与 pooling、非法输入/长度拒绝、旧缓存拒绝、feature cache 写入/恢复、汇总混用拒绝。
- 实际从 AS β-only fold 1 train/test 与两个 independent CSV 各取正负 20 条，共 160 条：全部守恒，三链布局成立，recognition 全部 masked，token 长度 233–255。此处是 CPU 输入检查，不是模型性能测试。
- AS β-only 五折全部 train/test/independent 的并集 **462,883 个唯一序列对**，逐条运行补全与真实 β 残基守恒检查，零错误；长度 230–263 tokens，约 91.8 秒。这是全量语义记录检查，不是全量模型前向或全量 token 编码。
- 使用真实本地 ESMC/LLaDA tokenizer 与原生 RemapCollator 做 CPU 检查：grammar mask ID=32，decoder mask ID=126336（`<|mdm_mask|>`）；实际 recognition 位置写入 126336，encoder 保持三链，未误用 ESMC ID。未加载模型权重。
- `--help` 和 wrapper 的 `bash -n` 已通过。初次测试暴露 pllm 缺少 sklearn；已把 runner 的 sklearn/metric import 放到实际训练/评分函数，使 CPU 输入与 provenance 检查可独立运行。另发现导入训练入口会因缺少 accelerate 失败；真实 tokenizer 检查改为独立加载 tokenizer/collator 完成。未安装包；正式模型/head 运行前仍需核验评测环境依赖。
- 未加载真实模型做推理、未训练 MLP/全参数模型、未提交/取消作业、未产生新 AUROC/AUPRC；正式使用需新 tag 并完成 GPU smoke。
- 源码交付注意：仓库现有 `/downstream` gitignore 规则覆盖 T1 runner/新增 query 模块；本次未改忽略规则或执行 git add，文件已保存在工作区，归档提交时需按项目现有流程显式纳入这些源码。

**复验入口**（仓库根目录，pllm 环境）：

本次实现快照 SHA256（区别于 §8 的首次审计快照）：

| 文件 | SHA256 |
|---|---|
| [query_protocol.py][query] | `22976a4b9c3f9046445dff7a39b7efd01841394cae7f64879ee5b30d64b2d010` |
| [run_retrained_ours.py][ours-run] | `72fb9f851cd7839b3a2e9e539b7221490480562922e3e5cebe23d07581e09fd5` |
| [test_tcr_binding_query.py][query-test] | `bf2a1bdd073beae75a36347ad11b4ea4c616a8fec8f65819a6a703e4ba771474` |

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python -m pytest -q scripts/tests/immune_llada/test_tcr_binding_query.py scripts/tests/immune_llada/test_receptor_completion.py scripts/tests/bioseq/test_grammar_tcr_relation.py scripts/tests/bioseq/test_grammar_tcr_role_resolution.py
```

### 9.2 2026-09-13：others/AS 的真实 CDR3αβ 第一版

**用户决策与输入边界**：在保留 epitope＋CDR3β 的基础上，只加入 others 提供的 CDR3α；不使用 MHC、TRAJ/TRBJ/TRBV/TRAV、LongA/LongB。这里不是丢弃 β 的 alpha-only 实验，也不是全部多特征 baseline 的复现。

**实现**：

1. [official loader][protocol]、[Ours runner][ours-run] 增加 `--track {cdr3b,cdr3ab}`，默认仍为 β-only；paired 明确选择 `_others_seen/AS/` / `_others_unseen/AS/`，其他负采样组合报错，不回退 β-only。
2. runtime query 协议升级为 `tcr_t1_runtime_v5_masked_recognition_observed_mean_v2`。paired 占位采样 key 仅由规范化 `[CDR3B, CDR3A]` 的 SHA256 决定；缺失/非法 α 报错。无标签/表位/行号参与补全，不读取 LongA/LongB 重建链。
3. 特征去重、内存/磁盘缓存和行映射都使用 `(Epitope, CDR3B, CDR3A)`，不能按 β＋peptide 错合并不同 α。provenance 含 track、输入列、忽略列与 loader 代码摘要；输出 `<tag>/cdr3ab/AS/` 与 β-only 隔离，predictions 保留 row_id 和 cdr3a。
4. wrapper 增加 `T1_TRACK`，默认 cdr3b。paired 汇总不生成与旧 β-only baseline 的混轨排行榜；旧数字与文件不覆盖。
5. 本版仍为 frozen ESMC＋LLaDA 的 observed-residue global mean＋每折 MLP，recognition mask 只作未知条件，不是关系 token logits 直接预测；未实现全参数微调。

**数据与 CPU 验证**：

- AS 五折 train 行数依次 6,824 / 6,852 / 6,866 / 6,886 / 6,908；seen_test 为 1,710 / 1,658 / 1,648 / 1,652 / 1,640；共享 seen/unseen independent 分别 626 / 1,378。12 个原始 CSV 均正负各半，CDR3A 均非空且非全 X；每文件无重复或冲突输入三元组。
- 全并集 **18,868 个唯一三元组**运行实际补全；α/β 观察残基全部保留，长度 232–257 tokens，零错误。没有落地预处理后的 benchmark 副本。
- 五折 audit 均为 train/test 精确输入三元组和配对 α/β clonotype 重叠 0；unseen epitope 交集 0。此项不是 v5 预训练去污染或近同源验收，不据此宣称预训练从未见过测试数据。
- `test_tcr_binding_query.py`、receptor completion、relation、role-resolution 合计 **63 passed**；新增测试覆盖 paired 残基、α 缺失拒绝、忽略字段不影响输入、不同 α 不复用缓存、轨道 loader/输出隔离及拒绝跨轨排名。YAML 解析与 bash 语法检查通过。
- 另跑 `test_retrained_ours_protocol.py` 的 5 项既有 β-only/五折回归均通过（与 query 测试同跑为 42 passed）；本轮不同用例共 **68**，不将两轮重复 query 用例相加。
- 收尾将上述五个测试文件合并重新运行，**68 passed in 5.78s**；wrapper bash 语法与本次相关 tracked 文件的 `git diff --check` 通过。T1 源码及 eval YAML 保存在工作区；现有忽略规则覆盖这些路径，未修改 gitignore 或执行 git add，后续归档提交需显式纳入。

**首次模型测试配置**：

- 使用当前 c20250601 v5 预训练验证集最优 `checkpoint-29000`（eval_loss=0.7530768225356571），不是按下游测试挑 checkpoint。不替换既有 headline 的历史 checkpoint。
- 已将权重及 tokenizer 四文件复制到 [独立评测快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_checkpoints/fusion_v5_c20250601_step29000_tcr_others_20260913/snapshot_manifest.json)，避免训练 top-k 清理；源/副本权重 SHA256 一致：`31828c1260a9d88f538e5678964ee994cc7d57d6e7e75c87f312303c98bea474`。
- 作业入口：[eval_tcr_others_cdr3ab_v5_29000.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000.yml)。单卡，五折×三个测试集；embedding batch16，head 最多 200 epochs、patience15，训练内 10% 分层 validation，禁止用 test 早停。
- tag=`ours_v5_29000_others_cdr3ab_masked_runtime_v2`；原始 audit 位于 [protocol_audit.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2/cdr3ab/AS/protocol_audit.json)。GPU 提交与终态记 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，新性能值产生后只写 RESULTS 独立 paired 表。
- 已核验既有 `protenix_abtcr` 环境具备 torch/sklearn/accelerate/transformers/esm；不向共享环境安装包。正式作业先激活 pllm，再显式调用该既有评测 Python。
- 作业 `t-20260913180748-44599` 首次查询为 Running；随后 **Success**，完整五折收数与复核见下方。β-only baseline 数字不得直接比较本轨道；预训练去污染、全参数对照、直接关系 token 预测均仍待做。

**GPU 结果验收（同日追加）**：

- 平台 Start=10:07:48Z、End=10:10:40Z，任务用时 172 秒；未取消/重提，无 OOM。冻结特征抽取 119.352 秒，最终 batch16，`(18868, 768)` float32 全部有限，约 158.1 条/秒；未单独测显存峰值或真实模型跨 batch 数值一致性。
- 五个 MLP 的 seed=1/2/3/4/5，分别运行 29/22/25/25/25 epochs 后按各自 validation loss 早停。每折 head 及 metrics 均记录 backbone_trainable_parameters=0，未更新 ESMC/fusion/LLaDA。
- 15 个测试均完成。回读原始官方 CSV，对每份预测逐行检查 row_id、peptide、cdr3b、cdr3a、Affinity、行数和分数有限/范围；核对 head hash、test-member hash、feature identity，全部通过。用落盘预测重新计算所有指标及五折 mean/sample SD，与落盘 metrics/summary 一致。
- 实际输入协议 SHA256=`cbd1d063eb0b73bbc507aae292fd355f98ee66dda6c62fdb4f84626896c91014`；feature identity=`fcb8852d8447f47d56c873120522b80d1cbf51e6e7feff5207c5d968fb036eda`。作业运行期间 input manifest 登记的源码摘要与工作区匹配；完整代码/模型/配置指纹在 [run_config.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2/cdr3ab/AS/run_config.json)。
- 权威指标及解释只维护在 [RESULTS §0.1a][results]，同目录 [summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2/cdr3ab/AS/summary.json) 为机器来源；本次不能证明 α 增益、纠错前后性能变化或 baseline 相对名次。
- 仍需单独开展：同一 others 行集的 β-only 消融、同字段/同训练预算 baseline、全参数适配及直接关系 token 预测。没有把这些未做项解释为本次五折作业失败。

**文档复核（2026-09-13）**：按用户要求再次核对本节与落盘 summary/结果表，确认上述关键改动、已完成状态和结果引用均已登记；修正 §5 遗留的“实验尚未执行”标题，区分 §9.1 历史快照与 §9.2 最新完成记录。本次仅更新文档，不修改评测代码、不重跑或提交作业，也不产生新指标。

### 9.3 2026-09-13：提交 others/AS 闲时独立重评

本节为提交阶段记录；用户随后取消该作业，终态与新的执行选择见 §9.4，不再视作在途任务。

**用户要求**：提交闲时任务开始测试。沿用 §9.2 已实现的 others/AS CDR3αβ 第一版；不是全参数微调、关系 token 直接预测或额外 baseline 套件。此前非闲时作业已 Success，无同范围在途作业需要取消。

**资源与入口变更**：

- 新增 [eval_tcr_others_cdr3ab_v5_29000_spot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_spot.yml)。队列 c20250601、1×ml.pni2.3xlarge，**Preemptible: true**；原非闲时 YAML/结果保留，不修改或取消预训练与其他下游任务。
- checkpoint 沿用 §9.2 的独立 29k 快照，数据、输入字段、mask/补全/pooling、五折×三个测试集、MLP 超参、seed 及 batch 均不变。已逐字比较两个 Entrypoint：除 tag 与日志路径/追加方式外相同；未改 Python 评测代码。
- 新 tag=`ours_v5_29000_others_cdr3ab_masked_runtime_v2_spot_20260913`；提交前确认其输出及特征缓存目录均不存在，确保实际重新抽特征、训练 head 和预测，而非只重读已完成的 §9.2 缓存。
- 结果根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2_spot_20260913/cdr3ab/AS/`；特征缓存位于同一 retrained 根的 `_feature_cache/ours_v5_29000_others_cdr3ab_masked_runtime_v2_spot_20260913/cdr3ab/AS/`。
- 日志：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_tcr_others_cdr3ab_v5_29000_spot_20260913.log`；使用 `tee -a` 保留平台重试的先前日志。
- YAML 设置 4 小时运行上限；仅 `InstanceReclaimed` 自动重试，最多 3 次、间隔 180 秒。沿用 runner 的特征块续写和已完成 head/预测身份检查；中断于未保存的特征块或未完成 head 时重算该块/该折，不宣称支持 MLP 中途 optimizer 精确续训，也不新增长期看护进程。

**提交与验证**：

- YAML/batch/入口等价性与 bash 语法检查通过；首次 CLI 解析发现旧模板的 RetryOptions 列表不兼容当前 CLI，未创建远端任务。改为映射后提交成功；详细失败/成功账本见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- 新 task_id=`t-20260913182826-c4pw2`，平台 CreateTime=2026-09-13 10:28:26；首次查询 **Queue**，`LaunchTime` 为空；平台已确认 `Preemptible=true` 和 c20250601 队列。此时是提交成功、等待调度，不是模型已开始计算。
- 本轮未重新运行 CPU 模型回归或产生新 AUROC/AUPRC；§9.2 的 68 项测试和既有五折分数仅作为此前已完成证据，不能当作闲时新作业的完成凭据。新作业完成后必须检查自己的 summary、逐折 metrics/预测及来源，再更新结果状态；不因重跑挑选更高测试分数替换旧结果。

### 9.4 2026-09-13：取消闲时排队，改用本地测试

本节保留当时的资源选择；用户随后授权的单卡非闲时方案见 §9.5，未启动本地评测，也未自动恢复旧闲时任务。

- **用户决定**：本次 TCR others 测试改为本地执行，不再提交 Volc 作业，并取消 §9.3 排队任务。该明确选择仅适用于本次 TCR 测试，不改动其他训练/下游任务的执行方式。
- **取消确认**：仅对 `t-20260913182826-c4pw2` 执行 cancel；取消前 Queue、LaunchTime 为空，平台最终 **Killed**，End=2026-09-13T11:16:17Z，LaunchTime 仍为空。任务未启动过模型计算，没有该闲时实验的新结果；操作与资源身份见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **保留/禁止重提**：此前 §9.2 的完整五折结果、29k 快照和评测代码均保留；闲时 YAML 作为历史配置保留，不自动重提，不取消任何其他任务。本地后续若重跑，需独立 local tag，不能将非闲时已完成的缓存或取消的 spot tag 误报为本地新结果。
- **本地只读检查**：可见 1 张 A100-SXM4-80GB，已有 2 个计算进程；11:16Z 快照为显存已用 15,220 MiB、空闲 65,829 MiB、GPU 利用率 88%。显存可用不代表计算资源空闲；未终止既有进程，也未立即叠加评测。本地共享 GPU 是否允许待用户确认。
- **本次执行边界**：只取消远端任务、核对终态与本地资源并同步文档；未改模型/评测协议，未启动本地 GPU 前向、MLP 重训或新 CPU 回归，未产生新指标。

### 9.5 2026-09-13：本地繁忙，授权单卡非闲时独立重评

- **用户授权与原因**：用户明确允许本地不合适时提交 1 张非闲时 GPU。本地仍有 `flow_antibody_tcr/occupy.py` 占卡脚本及 FABind 训练，11:23Z GPU 利用率约 84%；没有停止这些进程、没有叠加本地重跑。提交前查询无同范围在途 TCR others 任务。
- **新增配置**：[eval_tcr_others_cdr3ab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_nonspot.yml)，c20250601、1×ml.pni2.3xlarge、**Preemptible: false**、4 小时运行上限。旧非闲时/闲时 YAML、模型快照、已有五折结果均不覆盖，不启用抢占重试或额外看护进程。
- **协议不变**：沿用 §9.2 的 v5 29k 独立快照、others/AS 三字段输入、masked recognition、运行时补全、observed-residue pool、冻结 ESMC＋LLaDA/每折 MLP；五折×三测试集、seed、batch 和超参相同。本次是独立复跑，不是 full finetuning、直接关系预测、新 checkpoint 或新增 baseline。
- **独立产物**：tag=`ours_v5_29000_others_cdr3ab_masked_runtime_v2_nonspot_20260913`；结果根为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2_nonspot_20260913/cdr3ab/AS/`，特征根为同一 retrained 目录下的 `_feature_cache/ours_v5_29000_others_cdr3ab_masked_runtime_v2_nonspot_20260913/cdr3ab/AS/`；日志为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_tcr_others_cdr3ab_v5_29000_nonspot_20260913.log`。
- **提交前验证**：checkpoint 权重存在；新 tag 对应输出和特征目录均不存在；YAML/bash 语法通过。归一化 tag/日志路径后 Entrypoint 与 §9.2 作业逐字一致，资源规格也一致；没有修改 Python 源码，未重复运行模型 CPU 回归。不用旧缓存命中冒充本次重跑。
- **任务状态**：`t-20260913192406-hxhfb`，CreateTime=2026-09-13 11:24:07，LaunchTime=11:24:33，首次查询 **Running**；平台确认 c20250601 队列和 `Preemptible=false`。账本见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。首次提交确认时尚无本次新指标；旧 §9.2 分数不作为这次完成凭据，收数后核对独立产物，不挑更高分替换旧结果。

**终态与独立产物复核（11:28Z 追加）**：

- 平台确认 **Success**，End=2026-09-13T11:27:24Z，Elapsed=197 秒；实际 LaunchTime 到结束约 171 秒。日志显示重新抽取 18,868 个输入、训练五个 MLP 并产生全部 15 份测试结果；特征抽取 117.981 秒，batch16，未触发 OOM。
- 本次 `(18868, 768)` 特征全部有限；features.npy 与首轮 SHA256 相同，15 份 predictions.csv 也逐份 SHA256 相同，三测试集的五折 mean/sample SD 与首轮完全一致。两次 protocol_audit JSON 完全相同；feature identity 为 `fcb8852d8447f47d56c873120522b80d1cbf51e6e7feff5207c5d968fb036eda`。
- 另回读原始官方行集，对本次 15 份预测核查行数、row_id、peptide/CDR3A/CDR3B/标签、head hash、test-member hash；从预测重新计算各项指标，全部通过。没有重复运行 68 项模型 CPU 回归，不能把此前的测试次数算成本轮新增。
- 新 [summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3ab_masked_runtime_v2_nonspot_20260913/cdr3ab/AS/summary.json) 与首轮独立保存；指标唯一归档位置仍为 [RESULTS §0.1a][results]，增加复跑来源，不重复增加模型榜单行。这只确认本配置本次复跑一致，不证明 α 增益、无预训练污染或全参数适配效果。

### 9.6 2026-09-13：v5 29k 在相同 others 行集上的 β-only 对照

- **用户请求及解释**：用户要求使用 v5 checkpoint 测试；承接上一轮同数据对照讨论，本次使用 §9.5 相同 29k 快照与 others/AS 五折，仅向模型提供 Epitope/CDR3B。不是切回较大的官方 β-only 数据集，不是全参数微调，不替换旧 αβ 结果。
- **实现**：[runner][ours-run] 新增 `--track others_cdr3b`，显式区分数据轨道（官方 loader 的 `cdr3ab`，即 others）和模型输入模式（`cdr3b`）。所有输入键、特征、pool 只用 peptide/beta；原始 CSV、行序、标签、五折、训练内验证切分种子保持不变。预测 CSV 中仍保留原始 `cdr3a` 供逐行核验，但它不进入模型。`--track cdr3b` 和 `--track cdr3ab` 的既有选择含义不变。
- **输入边界**：关系 token 仍为 decoder mask；缺失 α 全为 synthetic X，observed pool 排除；MHC、V/J、LongA/B 忽略。补全仍以可见受体作为确定性 seed：β-only 用 β，αβ 用 β/α。因此两种模式的占位长度/位置可能不同，不能把本对照称为“完全固定 token 布局、仅替换 α 残基”的严格消融。
- **隔离**：新 track 独立输出与缓存路径，禁止与旧大 β-only 数据集的 baseline 自动混排；summary/metrics/run_config 标明 data_track、input_mode、input_columns。源码 hash 更新会使旧缓存拒绝复用，不覆盖或删除旧产物。
- **配置入口**：[eval_tcr_others_cdr3b_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3b_v5_29000_nonspot.yml)。沿用前次单卡非闲时资源与全部 MLP 超参；tag=`ours_v5_29000_others_cdr3b_masked_runtime_v2_nonspot_20260913`，输出根为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3b_masked_runtime_v2_nonspot_20260913/others_cdr3b/AS/`。
- **提交前验证**：本轮实际运行 query 回归与新增 track 回归，共 **43 项通过**（37 既有＋6 新增；不是重复计入此前的 68 项）。YAML/bash 语法通过。真实 others 数据投影后仍为 18,868 个唯一 peptide/beta 输入；五折无训练重复/冲突配对，各训练/测试精确配对交集为零。此检查不等于预训练去污染或近同源无泄露。
- **全量运行时验收**：18,868 个记录真实 β 残基全部保留，α 全为 synthetic X；完成后长度 242–275 tokens。20 次 fold/split 读取的原始 member metadata 与 §9.5 完全相同；29k 权重 SHA256 与前次一致（`31828c1260a9d88f538e5678964ee994cc7d57d6e7e75c87f312303c98bea474`）。新输出/缓存目录提交前均不存在。
- **执行状态**：已提交 `t-20260913202402-c9hf5`，12:24Z 首次查询 Staging，12:25Z 确认 Running/重新抽取特征；当时尚无本 β-only 对照的新分数。本地 GPU 利用率 95%，不干扰占卡/FABind 进程；沿用用户已允许的单卡非闲时评测方式。任务账本唯一入口为 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，新指标仅写 [RESULTS][results]。

**终态及收数（12:28Z 追加）**：

- 平台 **Success**；Launch=12:24:32Z，End=12:27:40Z，提交至结束 218 秒、实际运行 188 秒。重新抽取特征 121.513 秒；五折 head 实际 epoch 为 29/22/21/29/33，验证集选择 epoch 为 14/7/6/14/18，未跑满上限 200；head 训练循环总计 2.362 秒。骨干可训练参数为零，未执行全参数微调。
- `(18868,768)` 特征全部有限；checkpoint SHA、全部原始 train/test member metadata、head 配置及逐折种子与前次 αβ 一致。15 份预测逐行比对 row_id、peptide、CDR3B、原始 CDR3A、标签；head/test-member hash 均相符，AUROC/precrec-AUPRC/AP 从预测重算通过。原始 CDR3A 只保存在预测审计列，feature manifest 明确模型输入只有 Epitope/CDR3B。
- 新 feature identity=`98c9ca311f3142da5c8e3c6131298104865da686c9523c1b89eee7a7bfe4d089`；[summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_cdr3b_masked_runtime_v2_nonspot_20260913/others_cdr3b/AS/summary.json) 与同目录 run_config/protocol_audit/head/predictions 构成证据链。新指标与比较解释唯一维护于 [RESULTS §0.1b][results]。
- 同数据 β-only 控制现已完成，不再列为未执行项；官方 others baseline 本地复跑、ESM2、全参数适配、直接关系 token 评分和预训练去污染仍未完成。本结果不足以宣称 α 有稳定增益，也不替换历史大 β-only 数据集分数。

### 9.7 2026-09-13：同 others 行集的 LongA/LongB 版本

- **用户决定**：批准增加官方 others 数据所提供的长链版本，用相同 v5 c20250601@29000 checkpoint、相同五折/三个测试集和冻结骨干＋MLP 测试。不替换 β-only 或 CDR3αβ 版本，不增加新 baseline 或全参数微调。
- **输入实现**：[runner][ours-run] 新增 `--track others_longab`，数据选择仍为官方 loader 的 `cdr3ab`（others/AS），模型模式为 `longab`，输入列严格为 `Epitope, LongB, LongA`。在读取原始 CSV 后调用训练侧 `_receptor_chains(beta_fv=LongB, alpha_fv=LongA)`，保持 peptide→beta→alpha 布局和长链原序列；不预处理数据副本、不运行 ANARCI、不猜测区域边界、不拼接 CDR3、不生成 synthetic X。无 MHC/V/J 特征。
- **校验与信息边界**：CDR3B/CDR3A 只验证其在对应 LongB/LongA 中恰好出现一次，且作为 prediction CSV 审计列保留；它们不是额外模型输入。缺列、缺链、非法序列、CDR3 不匹配/重复出现均报错，不丢行、不回退；完整输入超 `max_length=1024` 直接拒绝（含 audit-only），不截断。标签仅用于 head 监督及评分，recognition 仍为 decoder mask。
- **读出语义**：沿用 post-LLaDA 全局 observed-residue mean；现在 peptide 和两条长链的所有残基均参与池化，排除 special/pad。没有人为标注 synthetic 残基或 CDR3-only pooling。长链上下文及池化范围同时改变，因此不是只增加 framework 信息的严格消融，也不等价于 TCRconv-fullAB 的上下文编码后 CDR3 读出/训练流程。
- **“长链”的边界**：这是数据集提供的可变区长序列，不宣称一定是实验直接测得的完整 TCR 蛋白；LongA/B 的逐条来源仍未全面审计。训练完整 Fv 路径保留序列但不发明 IMGT 区域标记，这与 CDR3 缺失区域补全路径不同。
- **隔离与溯源**：protocol=`tcr_t1_provided_longab_masked_recognition_observed_mean_v1`，input_record=`peptide_mask_provided_long_beta_alpha_runtime`。特征键改为 `(Epitope,LongB,LongA)`；manifest/缓存身份编码 input mode、实际列、读出、源码 hash 及模型 hash，拒绝复用旧 CDR3 缓存。预测 CSV 新增 `longb/longa` 可回查原始行；禁止合入旧大 β-only 排名。
- **执行入口**：[eval_tcr_others_longab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml)，c20250601 单卡、Preemptible=false；tag=`ours_v5_29000_others_longab_masked_runtime_v1_20260913`。独立结果根为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_longab_masked_runtime_v1_20260913/others_longab/AS/`；特征在同 retrained 根的 `_feature_cache/<tag>/others_longab/AS/`，不修改旧产物。日志为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/eval_tcr_others_longab_v5_29000_nonspot_20260913.log`。
- **CPU 回归**：本轮实际运行 query、others-beta 与新增 [longab 测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_binding_long.py)，**55 项通过（43 既有＋12 新增，6.56 秒）**。覆盖真 collator 的 mask/残基范围、完整链原样保留、非法/重叠匹配拒绝、长链 embedder/缓存 round-trip、模式隔离与源行不变。没有重复计入先前的 68 项。
- **真实数据 gate**：18,868 个唯一 `(Epitope,LongB,LongA)` 全部通过真实 runtime record 构造，token 长度 205–259；没有 synthetic mask 或猜测区域。所有 fold/split 原始 member metadata 与 §9.5 完全相同；五折训练无重复/冲突输入，训练/测试精确输入对交集为零。checkpoint SHA256 仍为 `31828c1260a9d88f538e5678964ee994cc7d57d6e7e75c87f312303c98bea474`；retrain.zip 大小/MD5 与官方登记一致。新输出和缓存提交前不存在；YAML/bash 语法通过，除 track/tag/log 外 Entrypoint 与 §9.6 相同。
- **提交状态（16:56Z）**：`t-20260914005618-vr2nb` 首次查询 Running，平台确认 Preemptible=false；这时尚无 LongAB 新分数。任务状态唯一账本为 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，收数后指标仅写 [RESULTS §0.1c][results]。

**终态及验收（17:00Z 追加）**：

- 平台 **Success**，Launch=16:56:37Z，End=16:59:36Z，提交至结束 197 秒、实际运行 179 秒。重新抽取特征 118.097 秒；五折 MLP 运行 epoch 为 **35/49/51/31/42**，按原有 `1e-4` 改善阈值选出的 epoch 为 **20/34/36/16/27**，均因 patience=15 提前停止。head 训练循环共 3.149 秒；未全参数微调，骨干可训练参数为零。
- `(18868,768)` 特征全部有限，feature identity=`5dbad1062c5c80cbc73ed5a7781b43e60a2eaf83013e914db4a6cdec1c45c8a1`；协议/源码 hash 与本次代码吻合。缓存 pairs.csv 与全部长链输入键逐项一致，run_config 与 feature manifest 一致。
- 所有原始 fold/split member metadata、checkpoint SHA、head 配置/种子与 §9.5 一致。另按每折原始训练行和相同 stratified seed 重算训练/验证划分，feature mean/std 与 **仅训练子集**重算值逐元素一致；未使用验证/测试特征拟合标准化。
- 全部 **15 份预测**逐行核对 row_id、Epitope、CDR3B/A、LongB/A、Affinity，无丢行，分数全部有限且在 [0,1]；head/train-member/test-member hash 与身份链一致。从预测重算 AUROC/precrec-AUPRC/AP 及阈值分类指标，逐项误差 <1e-10；五折 mean/sample SD 重算误差 <1e-12。
- 独立 [summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/ours_v5_29000_others_longab_masked_runtime_v1_20260913/others_longab/AS/summary.json)、同目录 run_config/protocol_audit/各 fold head 与预测，以及独立缓存构成证据链。指标与三输入版本比较只维护于 [RESULTS §0.1c][results]。旧 β-only/CDR3αβ 结果保留；不能据此宣称独立集泛化改善、超越论文 baseline、预训练无污染，或完成了直接关系 token 预测评测。

### 9.8 2026-09-13：全部已构建 TCR 任务的测试准则确认

用户希望测试 TCR 已构建的全部任务，并明确要求后续遵守本文准则。本条将该要求固定为后续工作的必读约束，不代表全套实验已经启动或完成。

| 任务 | 已有任务手册 | 执行前重点 |
|---|---|---|
| T1 binding | [TCR_T1_BINDING.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md) | 未知 recognition 必须 mask；冻结探针、全参数适配、关系 token 预测分开报告 |
| T2 clustering | [TCR_T2_CLUSTERING.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md) | 表位标签不得进入 embedding；基础 A/B 协议分开，核查聚类参数选择与去重 |
| T3 representation / few-shot | [TCR_T3_REPRESENTATION.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md) | 支持/查询集隔离；主 few-shot 与 broad probe 分开；不在 query/test 拟合 scaler/PCA 或选择超参 |
| T4 generation | [TCR_T4_GENERATION.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) | encoder 不得看到生成目标真值；核对长度、解码与重排预算、oracle 独立性；不同生成设置不混表 |

持续执行要求：

1. 先核查实际模型可见的序列、关系、位置/mask 和 encoder 条件，不能只看模板文字。读取原始数据后运行时适配训练范式；保留真实可用信息，缺失部分才按已冻结规则补全，不引入标签或未提供的真实配对信息。
2. 关系按任务语义处理：T1 未知关系查询用 decoder mask；T2/T3 禁止注入目标表位标签；T4 指定期望 binding 可以是合法生成条件，不能把 T1 修正规则机械推广成所有任务都必须遮蔽关系。生成目标与条件必须隔离。
3. 固定 checkpoint/hash、数据/split、输入字段、pooling/解码、种子和选模规则；缓存及输出按协议隔离。保留旧结果，不把先前 checkpoint 或协议的完成状态当成当前验收证据。T1 已完成三种 others 输入，并不表示 T2–T4 已通过本次 v5 专项验收。
4. 各任务都需对照 baseline，并明确论文值、官方 artifact 重评分、本地复跑/控制及各自训练范围；数据行集、正负比例、字段或指标口径不一致时显式披露，不作直接同协议排名。缺失 baseline 结果保留待做标记，不用 Ours 自身消融替代 baseline 对照。
5. 新运行前通过对应手册的 gate，先做输入/CPU 检查与必要的 GPU 小样验收，再做完整评测；记录失败和未完成项。关键决定、代码修正和验收同回合追加本文；数值唯一归档 [RESULTS][results]，提交/取消和平台状态记 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。未实现的全参数/ESM2/关系评分分支不能写成现有任务已跑通。

本轮仅完成要求登记与四项任务文档入口/提交 gate 的只读盘点；未修改 Python、重新测试模型、提交/取消作业或产生新指标。T2–T4 的专项审计状态仍以 §7 为准。后续逐项核验再执行，不直接用旧 wrapper 一次性提交未经验证的整套任务。

### 9.9 2026-09-13：开始 T2 clustering 输入与 relation 审计

- **用户请求**：开始审计，优先确认 clustering 是否需要给定 binding 标签。本轮只做诊断，不将审计请求当成共享输入代码修改或 GPU 提交授权。
- **结论及唯一详细记录**：当前主路径不存在 T1 那种固定正类 recognition 问题；T2 的无条件 `<null>/<unknown>` 前缀与 T1 关系查询用途不同。完整实际链路、标签使用例外、已发现问题和新 v5 提交 gate 维护于 [T2 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)。不要给 T2 添加真实 epitope/binding 条件，也不要直接套用带 peptide 的 T1 adapter。
- **实际验证边界**：调用真实 T2 特征入口和共享 record/pooling 方法，用本地 v5 tokenizer＋原生 remap collator 检查 A-beta/A-paired/B-beta 三条路径；每条用前 8 行进行评价标签改变前后比较。所有实际输入字段一致，relation 为无条件 unknown。使用替身 decoder 的 features/聚类不变；未加载权重、未使用 GPU，不把此检查称为模型推理验收或全量聚类重跑。另只读统计两份完整 CSV 的序列合法性；没有数据副本或新生产指标。
- **待做项**：任务 §7.1 登记的 v5 运行时补全、补全后的 synthetic 排除、旧已知-K 分支解释、严格输入验证、外部 embedding 行身份/provenance，及后续 baseline/预训练 overlap 核查。没有证明发生 binding 标签泄露、缓存误复用，或历史结果必然偏高/偏低。
- **数据身份**：基础 A [tcrs.csv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_clustering/tcrs.csv) SHA256=`1fd8dc255cd648a2d80f1d82630e120bd583e6001bb77d17a7ffd0e3b125eb85`；基础 B [tcrs.csv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_clustering_embed/tcrs.csv) SHA256=`1a6241393c1cbc3c4c8869a4da67787b6ecb233bbf00eefa5f6b05072966fb8d`。
- **代码身份**：本轮未修改源码。[run.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_clustering/run.py) SHA256=`6c2fa21e93e5222d39a2d334313eae28765e25ffd273e857a1935fdc1bcc4d21`；[run_embed_bench.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_clustering/run_embed_bench.py) SHA256=`d41dc7aa8530c0b5221fbcf7f0799d936a929ad6df50be76d967338778a96834`；shared model_api / grammar 与 §8 同 SHA256。
- **同步与结果状态**：T2 任务手册记录缺陷与验收；[RESULTS §0.2][results] 加 v5 引用限制但保留原数值及官方 baseline 状态；项目计划、过程记录同步。未提交/取消任务、未自动修复或产生新的 T2 模型性能指标；T3/T4 不因本次检查视作完成。

### 9.10 2026-09-13：T3 representation / few-shot 专项初查

- **用户请求与范围**：继续审计 T3。详细协议、发现与验收唯一维护于 [T3 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。区分模型输入的 binding token 与合法的 support 标签；不将审计解释为改生产代码或启动 GPU 任务的授权。
- **验证记录**：实际 deep 特征入口的 spy 参数隔离检查通过；生产 broad / deep 评分函数各 8 个合成 episode 检查通过；四份实际 CSV 全量身份与缺链统计完成。检查仅覆盖入口参数/评分逻辑和数据，不加载 checkpoint，不生成真实 decoder hidden，不宣称 GPU 或模型性能通过。首次临时诊断错误地把 `_embed_vectors` 的 `(vectors, name)` 返回整体当数组比较，修正诊断解包后通过；未修改被测代码。
- **后续边界**：v5 运行时补全与 pooling 尚未实现；成对去重不能替代 β-only 身份检查；缺链、多表位、预训练重叠等按任务 §7.1 gate 处理。没有证明所有历史分数存在 binding 泄露，也没有保证修正后提分或排序不变。
- **代码身份**：本轮未修改源码。`common/fewshot.py` SHA256=`2844c3eda55144ad0075275fdfb6e8a964a3d02ca9d5314299f3fadcac72280b`；`tcr_representation/run.py`=`5a6aeb07638170270ed0c0439a384593b5d51e184f5fd8d99cdea541a1932665`；`run_paper6.py`=`4b7c565283082e9e88d4cc7e01bd1844ef588bad8554a10a7c54a0136614c558`；`common/model_api.py`=`e839326cb1e188e2a3717d366c8875a1e9a5a316712723e3a02d59b95a63fb80`。文件均位于 [benchmark](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark)。
- **数据身份**（SHA256；数据根与完整统计见任务 §7.1）：broad `train.csv`=`b603734cccc1de8329c8dc69c26fb604d6f43b3f97e7222492147c005886160e`；`test.csv`=`efb366329cecb6634635d84d8f502756ed91bfbe17359f09c81885da051454fa`；deep `target_binders.csv`=`880ea00d83fa4e53b4d67dd151112b486326275558afd1b4c3303a2e1195ea15`；`background_pool.csv`=`e50dd8d9713f63a5a9b9ae1b97e76c586f058768f665558f6e42ed39f7db3c50`。
- **结果与同步**：[RESULTS §0.3][results] 增加纠正后 v5 引用边界，原数字保留；项目计划、过程记录同步。未修生产代码/数据、提交/取消作业或新增模型指标，T4 尚未专项验证。

### 9.11 2026-09-13：按原论文重新核对 T3 baseline

- **用户请求**：仔细查看论文，解释 T3 baseline 怎么跑。本轮阅读 SCEPTR 正式发表版、预印本补充方法，核对作者数据 notebook 和官方 TCR-BERT 模型卡，并只读检查本地源数据/已有产物。详细协议、证据与纠错的唯一 owner 为 [T3 任务 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。
- **关键结论**：保留“T3 主 NN 不输入 peptide/binding 真值”的结论；撤回把本地 CDR3 输入/β-only TCR-BERT、数据范围和 checkpoint 说成论文 recipe 的表述。明确区分论文冻结 NN、辅助 SVC、SCEPTR 部分层微调，与本地控制。已有监督不自动等于测试泄露，MLM-only 也不能未经说明替换论文 checkpoint。
- **证据身份**：正式文献 DOI=`10.1016/j.cels.2024.12.006`（Cell Systems 2025）；补充方法对照为 `arXiv:2406.06397v2`。作者 `tcrlm` notebook 固定 commit=`70ef27939051937a8a5c1d8edd52b51ea4326706`，访问日期 2026-09-13。本地四份数据与入口代码身份沿用 §9.10；原始数据位置从现有 `meta.json.source` 读取，未生成评测副本或重建数据。
- **已执行验证**：只读匹配源数据的 reference 与导出 αβ pair，确认存在与作者排除来源对应的样本；准确计数、解释边界及方法见任务 §7.2。检查 ESM2/ProtBert/TCR-BERT 的已有 `fewshot.json` 实际 columns，与当前 adapter 的模型/层/池化接线交叉核对。未把当前源码当作旧运行权重 hash 的证明；原始模型 snapshot/完整作者 episode 仍待追溯。
- **引用处置**：[RESULTS §0.3 及历史校准节][results] 撤回等同论文的解释，不改原分数；README、BASELINE_VERIFICATION 改为失配/待重评并链接任务 owner；项目计划与进程同步。不改历史 `[P]` 原值或静默重写 registry，旧机器标记不得凌驾本次审计。
- **实施边界**：本次仅研究、诊断和文档纠错；未改生产 Python、训练 renderer、原始 CSV/JSON 或权重，未提交/取消任务，没有新增模型指标。修复后必须独立命名产物并重评，不能由旧分数推断修正收益。

#### 9.11.1 关键理解：后续 T3 实验如何判断与比较

按用户要求，将关键理解直接固化在总审计中。这里维护**解释和执行准则**；逐模型输入/层数/权重对照表、数据计数与原文证据仍以 [T3 任务 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md:188) 为唯一详细记录，不复制第二份问题台账。

1. **文件有表位，不代表必须把表位输入模型。** 标准 TCR-only 评测将表位用于组织 support 和评价 query。若新增表位条件版，必须独立命名和比较；不能根据 query 真标签为它挑选表位条件，也不能替换后仍宣称与原论文是同一任务。
2. **合法 support 标签与 query 真值输入是两回事。** 固定 support 后，改变 query 真值不得改变模型输入或打分。不要把 T1 的未知关系查询、T2 的无监督聚类与 T3 的有标签 support 混成一种任务；也不要机械给所有任务添加 binding 或关系 mask。
3. **评测时冻结，不等于 checkpoint 历史上从未接受监督。** 必须分别记录模型既往训练和本次下游更新范围。冻结 NN、训练读出头、部分层微调、全参数微调是不同设置。论文指定 checkpoint 与 MLM-only 控制要分臂，不能静默替换；已有监督也不能直接判作测试泄露，仍需 overlap 证据。
4. **比较输入必须追溯序列来源。** CDR3、V 基因推导的其他 CDR、V/J/CDR3 重建完整链、实测长链、v5 运行时合成补全不是同一种信息条件。不能把完整链重建说成原始数据提供全长，也不能把我们的补全当作已复现 Stitchr。链范围之外，还要核对模型大小、取哪层、如何池化及特殊 token 处理。
5. **数据集名称、目标表位相同或分数接近，都不是协议对齐证明。** 需核对来源过滤、去重单位、reference/query、抽样和误差条。不额外导入某来源文件，不代表数据库内部已排除该来源；αβ pair 去重也不能外推为 β-only 不重叠。已撤回的兼容判定不能继续支持“超过论文”或模型能力归因，修正也不保证提分。
6. **后续分开三个实验目的。** ① 论文原协议复现：恢复各 baseline 的原始 recipe；② 同输入公平对照：统一允许的信息并披露与论文的差别；③ 我们的 v5 适配：独立验收运行时构造、信息隔离及真实残基读出。三者可共存，但不能混成一张未披露差异的“论文 baseline 对比”。

**本次仅固化理解，不表示已批准或完成全部实现。** 未改变评测代码、数据、权重或历史分数；后续关键实现和验收仍需持续追加到本文。

### 9.12 2026-09-13：T4 generation 当前实现与结果边界初查

- 用户询问 TCR generation 目前情况。已只读核对 wrapper、条件采样入口、mask/共享 sampler、eval JSON 与三个已有 common-6 metrics；详细发现与 gate 统一见 [T4 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。未执行原生 tokenizer/collator 或真实模型测试，不称全套审计通过。
- 关键理解：期望结合的生成条件与 T1 待预测 binding 标签不同；不能机械删除生成任务的期望关系。参考相似度与精确匹配不是实验结合率，F1=0 不等于证明所有新序列不结合。
- 引用状态：共享多步 sampler 修复不能自动更新旧 T4 成绩；在 RESULTS §0.4 提示逐产物版本核对及重评，不改旧数字或独立官方 baseline。当前已列历史模型不作为 v5 29k 生成结果。
- 本轮只更新文档，不改代码、数据、权重或任务状态，没有新增性能指标。输入/长度隔离、v5 适配、预训练 overlap、生成动态不变量和 baseline 原文核对仍待完成。

### 9.13 2026-09-13：T4 baseline 原生生成与本地产物核查

- 用户询问 baseline 如何生成；已核对 TCRT5 正式论文、相关官方仓库及本地调用。逐模型机制、条件与新问题唯一详表见 [T4 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
- 关键理解：作者预存预测重评分不等于现场生成；同K不等于同预算、同条件。生成期望关系不等于实验结合标签；专用 baseline 不必使用我们的关系 token。
- 新问题及处置：TCRDiff 本地路径存在参考长度信息与默认 Ours 条件不一致，详见任务 owner；RESULTS §0.4 标注不作严格同信息 headline，保留历史数字和来源标记。不能据此断言其优势都来自长度条件，亦不能将当前源码泛化为全部旧产物版本。
- 本轮仅文档纠错与只读研究，未改 Python、数据、权重或任务状态，无新增性能指标；后续修正和验收继续追加本文。

### 9.14 2026-09-13：held20 / benchmark14 对 v5 实际 train、valid 的全量核查

- 用户要求核查两份生成数据是否进入我们的训练/验证。两条v5 diffusion启动metadata与prepared manifest已核对；只读扫描全部选中分片，分开表位、beta核心、完整junction证据、参考配对及关系/MHC子集。权威数表、方法、边界与产物见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
- 关键理解：TCRT5作者的validation归属不是我们的split身份；但held20也不能因此被当作对我们全新、独立的表位/受体集合。最终发现参考配对未残留，同时存在表位与beta分别重叠；benchmark14应区分已见与未见表位。公共6目标的v5未见身份本轮重新实证，不能倒推所有历史模型。
- 验收：新增可复跑的只读审计脚本及合成自测；第一版漏读无regions真实长链时被完整性门禁拒绝，最终v2补查长链、合成X伴侣与缺identifier行后全量通过，未通过报告明确保留为无效中间产物。参考非标准字符另行记录，未静默清洗。
- 本轮只新增诊断脚本、审计产物与文档；未修改prepared数据、blocklist、训练/生成生产代码、权重或模型分数，未submit/cancel。精确成员匹配不等于完成近似同源、ESMC历史预训练或逐checkpoint已消费样本审计；T4输入/sampler/完整重评gate仍未关闭。

### 9.15 2026-09-13：联合 αβ 生成后取 β 比较的提议

- 用户提出条件联合生成αβ、最终以CDR3β与现有baseline比较；候选协议、代码依据及信息/候选预算边界见 [T4 §7.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
- 关键澄清：训练的未知X占位与生成MASK不是同一语义；用户所述“非框架区域”是否包含CDR1/2，尚待确认。联合采样能力不等于已验证配对功能；β-only参考数据无法单独验证αβ真实性。
- 本轮仅记录方向、静态核对与提问，没有实施生产改动、提交任务或新指标；不得把此条当作已完成的联合生成协议。

### 9.16 2026-09-13：49000 全套 TCR 重评要求与提交前复核

用户要求“所有下游任务确认没有问题后”测试指定的最新训练模型 ckpt49000；本条登记条件与范围，**不是全套验收通过或任务已提交**。

- **模型与汇总**：本轮固定 c20250601 的 v5 diffusion 49000，不自动替换为其他队列同一步数、后续保存点或验证最优 checkpoint。新汇总的 Ours 行只接收本轮 49000 的产物，缺项明确写未完成；不拿 29k/42k 等历史分数补齐。历史文件保留，baseline 的论文/官方来源分层规约不变。现存快照身份、SHA256 复核和训练源路径状态唯一记录在 [PROJECT_PROCESS：TCR 49000 提交前复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **T1 范围**：保留既有 NM2025 大 β-only `cdr3b`；另在同一官方 others/AS 行集上覆盖 `cdr3ab`（Epitope＋CDR3B/A）、`others_longab`（Epitope＋LongB/A），并保留已构建的 `others_cdr3b` 同行 β-only 控制。三个 others 版本都按现有五折与三个测试集执行，各自独立协议/缓存/输出；不把大 β-only 与 others 混成同数据消融。沿用已纠正的 masked recognition，非长链读取后运行时补全，长链保留真实提供序列；这仍是冻结骨干＋MLP，不暗中切为全参数微调或直接关系 token 评分。
- **T2 / T3 / T4 范围**：沿用 §9.8 与各任务 owner 的已构建任务边界；T2 基础 A/B、T3 few-shot/broad、T4 生成设置分别报告，不合成一个指标。T4 同时保留 benchmark14 与 held20 的不同来源/seen 状态，β-only 控制和用户提出的条件联合 αβ 版不能互相冒充；后者的目标区域语义仍待确认（§9.15 / T4 §7.4）。

本轮只读源码复核与 CPU 验证：

| 项目 | 本次状态 | 未关闭项的权威位置 |
|---|---|---|
| T1 输入回归 | 在 pllm 重跑三个现有测试文件，55 passed；没有加载 49000 权重前向或重跑五折 | 本文 §6：真实模型批次一致性/显存专项及预训练 overlap 等仍待做；历史 GPU 通过不自动覆盖新 checkpoint |
| T2 | 实际 wrapper 仍调用共享旧单链/双链 embedder，未接入 v5 运行时补全与 observed-only pooling | [T2 §7.1 / §8.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md) |
| T3 | 同一旧 embedder 路径仍在使用；论文数据、baseline recipe 与本地评测的差异没有通过文档登记自动消失 | [T3 §7.1–7.2 / §8.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md) |
| T4 | `conditional_cdr3b` 仍只有 β；`fulllength_pairs` 是无表位条件的另一模式，不是拟议联合版；任务特定双路输入/sampler 验收仍待做 | [T4 §7.1 / §7.4 / §8.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) |

**执行结论**：尚未满足用户的前置条件，本轮没有运行旧 wrapper 来替代纠正后的全套评测，没有提交/取消 GPU 作业、修改生产代码或生成模型性能指标，也没有改写历史 RESULTS。后续先完成任务专属修正与验收；涉及 T4 CDR1/2 是否参与生成的实质选择，不从本次条件性运行请求推断答案。

### 9.17 2026-09-13：确认 CDR1/2 保持未知，并核查测试数据字段

- **用户确认**：未知框架和CDR1/2保留训练的X占位，仅CDR3α/β使用MASK参与联合生成。关闭 §9.15–9.16 的区域语义待确认项；不意味着T2–T4所有门禁已通过，也不代表联合生成已经实现。
- **关键理解纠正**：当前held20/benchmark14评测产物确实都未提供CDR1/2及α/长链，但“评测产物没有”不等于“作者源文件完全没有”。本轮全量核查发现held20原CSV还含部分CDR3α与stitched长链；详细字段、计数、来源证据及边界唯一见 [T4 §7.4.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。此前“只有β参考”的表述应限定于当前导出的评测产物，不能外推到全部上游资料。
- **执行边界**：表位条件生成不因上游存在参考受体字段就自动读取它们；未知区域继续X占位，真实受体信息只在另行明确的条件任务中使用。联合生成仍保存αβ对应、投影β比较，当前参考集合不能直接验证αβ配对功能。
- **本轮动作**：只读检查全部评测条目和两个官方源CSV，复核文件SHA并登记身份清单，同步任务协议及本日志；未修改生产代码/数据、未加载权重/提交GPU、无新模型指标。49000与latest-only汇总要求继续按 §9.16 执行。

### 9.18 2026-09-13：benchmark14 数据形式与 TCRT5 输出范围

- 用户追问benchmark14的格式及β-only原因；对照原CSV、当前JSON/准备脚本、官方collator和论文Methods，详细映射与实物例子见 [T4 §7.4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
- 关键理解：benchmark14是一行一个pMHC及其参考β集合，而非14条受体或正负分类样本；作者候选列表与参考真值是不同字段。TCRT5已发表生成任务输出CDR3β，不是整条β链；这是研究选用的表示，不是架构只能生成β。该参考集合可以评价我们联合生成的β投影，不能独立证明α或αβ功能。
- 本輪仅解释、核对与文档更新，无生产代码/数据变更、GPU任务或新模型分数；不改变已确认的X/MASK区域语义、49000选择或生成条件。

### 9.19 2026-09-13：调研双链生成 baseline，登记 β-only / αβ 两组

- **用户决策**：生成比较分成β-only与配对αβ两组；先核实真实生成能力和可用资产，再接入。逐模型论文/官方代码证据、两组清单、就绪状态和验收门禁统一见 [T4 §7.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。这是已有T4内的候选/方法登记，不表示新配对榜已经实现或现有任务范围解冻。
- **关键理解**：存在联合生成与顺序配对生成两种可用方向；不能把β投影误认作模型只能生成β，也不能把使用配对数据/配对打分误认作支持双链生成。现成TCRT5仍为β-only。全长组装、CDR3双链生成、给定真实伴链补全是不同任务，须分别命名。
- **当前缺口**：本轮发现的双链输出保留、测试参考长度、候选预算/筛选与full模式语义等实现问题见任务owner；这些未因加入清单而解决。benchmark14能评价双链输出的β部分，不能独立验证α或αβ配对；真正配对评测的数据冻结与重叠审计仍待做。
- **不变约束**：Ours继续固定未知FR/CDR1/2为X、仅CDR3αβ使用MASK；新汇总只使用指定v5 49000。外部baseline用自己的原生表示，并显式披露MHC/V基因/长度/筛选条件，不强制复制Ours token。
- **本轮动作**：只读研究论文/官方仓库，核对本地源码、相关权重存在性和配对数据候选表头，同步文档；没有生产实现、数据改写、权重下载/重新SHA核验、GPU任务或新模型指标。候选可用性不等于通过完整运行验收，后续实现继续追加本文。

### 9.20 2026-09-13：复用已有生成模型的真实配对测试数据

- **用户方向**：保留benchmark14评价联合输出β的做法，另找已有模型做过生成实验的真实αβ配对数据。来源优先级、论文/官方脚本依据、全行核查及接入限制见 [T4 §7.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。推荐IMMREP23作首版双链序列来源，STCRDab作后续结构补充；本轮是研究建议，不是已完成新配对评测。
- **关键理解更新**：benchmark14本身无α，与它的部分上游来源有真实配对并不矛盾；必须从原始配对行保留对应，不能单靠β任意找α。新来源确实提供CDR1/2及重建长链，但字段存在不授权将参考受体信息喂给本轮pMHC-only生成；X/MASK决定不变。
- **复用边界**：已有模型在同数据做过实验，不等于同输入/同预算；显式V基因、参考长度、结构条件和真实伴链补全与无参考受体双链生成须分开。原论文分数不能直接充当修改条件后的复跑结果，官方测试集也不能直接称为我们v5未见数据。
- **本轮验证**：全行解析本地配对文件、读取官方solutions做字段多重集对照、确认结构子集排除规则、检查benchmark14交集及数据SHA；详细数字唯一见任务owner。没有生产代码/源数据改写、GPU作业、新模型分数或新配对集落盘；输入适配、训练重叠和完整评分验收仍待做。后续新Ours汇总继续只用指定49000。

### 9.21 2026-09-13：双链生成延后，实施并提交当前 49000 任务

- **用户最新决定优先**：双链生成作为后续实现，现阶段开始实施与提交。执行子任务范围唯一见 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)；没有提交 IMMREP23/αβ 生成，也没有把“占位 α 链”冒称成对生成。
- **模型与结果纪律**：固定用户指定 c20250601 v5 49000 的既有评测快照；源 checkpoint 已被训练保留策略清理，因此使用已核验 SHA、含完整 fusion 的独立快照，不切换其他步数。快照名称带 `ab` 是保存用途，不代表 AB 微调模型。新 Ours 汇总只收此权重的完整、可追溯结果；旧 29k/42k 等保留历史，不补缺项。
- **关键实现登记**：T1 四输入轨、全折全测试集及 cache/GPU gate 见 [T1 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)；T2 无标签 runtime adapter 与 observed-only pooling 见 [T2 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)；T3 缺 α、同受体多表位、support 身份和本地对照解释见 [T3 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)；T4 β-only 原生布局、C/F 锚点、长度先验、候选预算、双路遮蔽与本地 gate 修复见 [T4 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。
- **工程与科学结论分开**：每项先通过专属输入/真实模型执行 gate 再提交，未直接调用旧 wrapper。当前可以收集修正后本地结果；但预训练成员/近同源、T3 论文同版数据、原生 baseline 输入与 episode 对齐仍有未关闭项。不是“所有审计问题都已解决”，不能把 local-control 改名为 paper-exact 或无污染泛化。
- **提交与资源**：10 项独立单卡非闲时任务均已提交（T1×4、T2×2、T3×2、T4×2）；准确 ID、YAML、初始/后续状态和时间唯一见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。无取消、重训、全参数微调或共享 AB sampler 修改；没有覆盖原始数据及旧实验产物。
- **验收证据**：[binding gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/binding_preflight/passed.json)、[representation gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/repr_preflight/passed.json)、[generation gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/generation_preflight/passed.json)。预检仅执行正确性，不属于正式模型质量结果；后续完成/失败及关键修正继续追加本文，不拿排队状态当完成。

### 9.22 2026-09-13：全任务执行与评分复查，参考 AB 非闲时提交

- **用户要求及范围**：对现阶段全部已构建 TCR 任务重新审查、验证后提交；参考 AB 的非闲时配置。沿用 §9.21 的指定 49000 与当前 10 项矩阵，不新增 IMMREP23/双链生成、全参数微调或未实现的关系预测分支；不把“所有任务已提交”表述成“所有科学审计已关闭”。
- **发现并修复实际遗漏**：T2A 的 alignment 锚点仍取旧 universe 数值；自动 cosine 阈值取 5 位、输出取 4 位，会合并不同边或使保存的阈值无法重建聚类。新增回归先复现，再修为精确阈值与已核实 Fig 3A 九方法锚点；最近点显式记录 retention 差值和不可对齐状态。详细协议、容差与代码 owner 见 [T2 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)，不在本文另抄 baseline 性能表。
- **验证不仅是输入前向**：新增 [preflight_tcr_suite.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_tcr_suite.py)，校验三类专属 gate 当前身份、原始文件 SHA、数据边界、真实模型提特征/训练/评分与生成评分。不修改原始 CSV/训练语料，不使用测试参考选择生成长度，不把预检产物放入正式结果目录。

| 方向 | 本轮实际复核 / 验收 | 不能据此声称 |
|---|---|---|
| T1 四输入轨 | recognition MASK、真实观测池化/缓存身份；others 三轨全折原始行集一致；每轨 32 train / 16 test 的 GPU 特征→2 epoch MLP→评分；训练内标准化且特征不被改写；AUROC 正反向/平分 oracle | 完成全参数微调、直接 relation-token 预测，或全部预训练去污染 |
| T2 A/B | 4,779 / 9,033 原始行数；无 peptide/binding 输入；精确阈值与存盘重建、label-blind sweep；真实特征→阈值簇/KMeans→评分 | 同链范围/同数据版本的公平论文排名；最近 retention 点一律可比 |
| T3 deep/broad | αβ 身份交集为零；保留多标签与未知 α；deep support 排除 oracle、真实 few-shot；broad 多标签 NN＋独立 probe/kNN 路径 | 论文同版数据、相同 baseline 输入/episode、预训练无重叠 |
| T4 两数据集 | 14/20 pMHC、34-aa MHC、原始参考不传入 sampler；各一目标 batch=8 / 32 步＋独立 argmax→评分；raw/unique/greedy/实际评分分母 oracle | 已实现 αβ 生成、held20 是 v5-unseen，或序列指标证明真实结合 |

- **通过证据**：实际 protenix_abtcr 环境联合回归 **81 passed**；修正后 [repr_preflight_v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/repr_preflight_v2/passed.json) 全行输入长度、最长/缺 α、batch=1/8 和倒序检查通过。独立 [suite_audit_local_v2/passed.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/suite_audit_local_v2/passed.json) 覆盖上表全部链路，A100-80GB，peak allocated 21.12 GiB，`quality_evaluation=false`。T1/T4 原专属 gate 仍与当前源码/数据匹配。
- **失败不隐藏**：首版 suite 在完成四轨 T1 和 T2A 后，因预检脚本将 T2B 的 `epitope` 列误作 `peptide` 而失败；原生产 T2B 已正确使用 `epitope`，这不是新的生产任务输入 bug。仅修预检中的标签列适配，重跑完整 suite 后通过。[首版 failed report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/suite_audit_local/report.json) 保留，未覆盖/改成 passed；CPU 测试初版调用了不存在的 metric 名也在正式通过前修正，不计为模型缺陷。
- **提交/取消纪律**：T2/T3 共享源码验收指纹，T2 修正会使四项旧 gate 过期，因此在它们均 Queue 时精确取消旧四项，确认 Killed；旧 YAML/gate/记录保留。修正后通过新 gate 再以 `*-v2` 名称重提，正式输出 `runtime_v2`，不重复在途运行；T1×4、T4×2 未取消/重复提交。另提交一项独立云端 suite 复验，仍 Queue，不冒称云端验收已通过。精确 ID、YAML 与首次/最新状态唯一见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **AB 对照**：已核对 [AB 预检配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_llada_20260913_preflight.yml) 和生成器，沿用 c20250601 / ml.pni2.3xlarge / 1 worker / `Preemptible:false`、同镜像/挂载/环境、独立预检和分任务正式作业；五份新 YAML 的 bash 语法、资源字段一致性与代码身份通过校验。未触碰 AB/预训练作业或共享 sampler。
- **当前未完成**：正式全量模型结果、云端 suite 复验、T1–T3 对实际 v5 的完整成员/近同源核验、T3 论文同版数据和各 baseline 同输入/episode 对照。T4 仍保留已核实的 held20 重叠分层和 51 条非标准参考记录披露；不偷偷过滤以制造干净结果。新 Ours 汇总只收指定 **49000**，不拼接历史步数；本次小样本产物不是可发表质量数字。

### 9.23 2026-09-13：短任务转本地，重任务保留云端

- **用户决定**：简单、耗时短的任务可以在当前机器执行；执行位置不再一律提交 Volc。任务是否“短”根据输入规模、既有同类运行耗时与当前资源判断，不通过抽样、降低 epoch 或候选预算来制造短任务。
- **本轮迁移**：T2 A/B、T3 broad、others 的 β-only / CDR3αβ / LongAB 六项转本地串行；大数据 β-only binding、T3 deep 和 T4 两数据集保留非闲时云端。六个原云端正式任务取消前均 Queue，随后确认 Killed，再启动本地，避免重复运行。额外云端 suite 同样取消，保留 §9.22 已完成的本地 suite，不把取消说成云端验证通过。
- **模型和协议不变**：仍为指定 49000 的同一完整 fusion 快照；已核对三个工程 gate 当前指纹。原 YAML Entrypoint、全部原始数据、fold、训练参数、seed/batch、输出 tag 与身份均不变；T1 仍 masked recognition、运行时补全/observed-only pooling，T2/T3 仍不输入表位或 binding。others 长链/非长链都继续测，不改成预处理评测副本。
- **本地执行入口与保护**：[run_tcr_local_short.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_local_short.py) 读取冻结 [local_plan_v1.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_plan_v1.json)，复用现有 YAML，不再实现一份评测算法。启动前验证 task ID/名称/Killed、YAML SHA/bash 语法、输出目录不存在和输入 gate；运行锁保证串行队列不重复启动。每项检查至少 24 GiB 空闲 GPU 显存；子任务失败则停止，单项实测超过 10 分钟则完成该项后暂停后续，等待重新安排。输出成功先记为 `executed_pending_result_audit`，不自动标指标验收通过。
- **当前状态与验证**：本机 A100-80GB 原有约 1.7 GiB 常驻占用，未干预原进程；新 tmux 本地队列已启动，T2A 已读取完整 4,779 行、启动前 10 项评分回归通过，其余排本地串行队列。新调度入口 py_compile / check-only 通过，没有更改模型、源数据、评测计算代码或已有性能数值。
- **追踪入口**：实时本地状态与逐任务日志见 [local_short_v1/status.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_short_v1/status.json)；准确取消 ID、原 YAML、tmux/PID/时间与云端剩余任务见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。旧 submissions.json / submissions_v2.json 保留历史，不再作为当前运行地点/状态。
- **结果纪律继续有效**：本地执行不等于放宽论文比较/去污染 gate；完整结果需要检查覆盖、来源和重算指标后才进入 RESULTS。新 Ours 只汇总 49000，双链生成仍是后续；本轮没有取消/修改 AB 或预训练作业。
- **首项实测**：T2A 本地完整执行约 79 秒，预期结果文件已落盘，标记为待独立结果验收；这支持本轮将该类小规模任务放本地的判断，不以执行快慢推断模型性能。后续串行进度以状态文件为准，本节之前的“T2A 已启动”保留为启动时快照。

### 9.24 2026-09-13：others 后继续本地运行，并持续监控

- **用户决定**：开始并持续监控；others 完成后，将剩余仍排队的任务转到本地执行，再取消对应云端排队的资源需求。实际执行顺序为先确认云端取消至 Killed，再启动本地，避免重复执行；已经 Running 的云端任务保留，不中断已投入的计算。
- **前置条件和范围**：既有六项本地队列（包括 others β-only、CDR3αβ、LongAB）全部执行成功且完整产物落盘后，才允许接续其余 T3 deep、T4 benchmark14/held20、大数据 T1 β-only。不重跑已完成的 others；不触碰 AB、预训练和原常驻占卡进程。
- **实现边界**：新增独立接续入口 [run_tcr_local_remaining.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_local_remaining.py)，不热修改已在运行的短任务入口；原 YAML Entrypoint 原样使用，49000 权重、数据/fold、训练和生成预算不变。前置队列锁、输出拒绝覆盖、云端 ID/名称/Killed、YAML/gate 指纹、GPU 空闲显存检查继续有效；本轮用户已允许接续较大任务，不再用短任务的十分钟阈值暂停后续，但失败即停止。
- **监控及结果纪律**：跟踪状态、逐任务日志、GPU 使用和预期产物，区分执行完成与独立结果复核。异常时保留日志，不自动降 batch、减少数据、改协议或覆盖重跑；关键状态更新至 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。本节开始时 others β-only 已完整执行，CDR3αβ 在跑、LongAB 待跑，四项云端仍 Queue；这是时间快照，具体迁移以逐项取消记录为准。

- **迁移已执行**：others 最后一项 LongAB 于 23:36:37Z 完成；四个剩余任务逐项再次确认 Queue 后取消、确认 Killed，云端当前列表为空。本地接续于 23:39:40Z 启动，原 YAML 完整执行。新入口 py_compile、6 个拒绝路径检查及完整 check-only 通过，具体 ID/时间/PID/冻结计划与状态见 PROJECT_PROCESS；旧 §9.23 的“重任务保留云端”已被本节接续决定覆盖。
- **others 完整结果复核**：新增只读 CPU 验收 [audit_tcr_49000_binding_results.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/audit_tcr_49000_binding_results.py)，三轨 45 份逐样本预测对应原始行/标签、五折×三测试集、特征/head/checkpoint/成员指纹、训练内标准化均值、逐预测指标及五折 mean/sample-std 均通过；[report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/others.json) 保留证据。只新增本轮 49000 数值至 [RESULTS §0.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，不复制旧 29k 分数，不把 artifact 一致性审计称为预训练去污染完成。

- **T2/T3 已完成部分的结果核对**：新增只读 [audit_tcr_49000_repr_results.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/audit_tcr_49000_repr_results.py)，T2 A/B 与 T3 broad 的源码/模型/数据身份、保存阈值曲线/19 个 K 点、775 个 support episode 与聚合统计一致，见 [repr_short report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_short.json)。该验证只重算保存结果的聚合及抽样身份，不重新运行未保存的嵌入/距离/聚类，不扩大为预训练无重叠或 paper-exact 证明。新数值仅见 RESULTS §0.2/§0.3；broad 高 shot 可用表位减少、std 是表位间离散度，已同步解释。

- **deep 和持久监控**：T3 deep 已完整执行，25,816 universe、8 shots×100 seeds×6 targets 的覆盖/聚合核对通过，证据 [repr_deep report](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v1/repr_deep.json)，数值仅见 RESULTS §0.3。新增 [monitor_tcr_49000_local.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/monitor_tcr_49000_local.py) 和冻结 [monitor_plan_v2.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_plan_v2.json)：每 30 秒检查两队列、producer 身份、启动器/计划/YAML 指纹、GPU/日志；输入 gate 仍由原正式任务入口校验；新完成任务触发固定只读 CPU 审计，全部执行和审计通过才算完成。监控/审计异常记 attention_required，不自行杀其他任务或改参数重试。实时 [monitor status](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v2/status.json)、[49000-only partial/final JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v2/results_49000.json)；旧 v1 索引保留历史。
- **监控自检发现的正常提前停止解释**：新增 [generation artifact auditor](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/audit_tcr_49000_generation_results.py) 首次自检误把 max_iter=32 当成必须完整 32 次前向，观察到 30/31 步后核对 [sampler break](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py:441)：所有目标提交后会正常提前结束。只修验收为 1≤actual_steps≤32，仍核对 100 完整候选、无残留非法字符、固定上下文、长度先验、独立 greedy 和原始参考重评分；没有改 sampler 或生成预算。只停止自建旧 monitor，再以新指纹计划 v2 启动，生产持续；[首版检查失败记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v1/auditor_check_failure.json) 未隐藏。语法/边界检查、三个已生成 pMHC 检查和完整 monitor check-only 均在修正后通过。

### 9.25 2026-09-14：状态核查发现 T4 评分写出失败，未自动恢复

- **本次请求与动作**：用户查询当前结果；核查任务/监控终态、原始生成产物并做只读 CPU 重评分。没有修改生产或监控代码、重启/重提任务、重新生成或覆盖产物。
- **状态纠正**：当前是 7 项通过相应结果核对，benchmark14 生成完成但评分写出失败，后两项未启动；监控已报错退出，不能把 §9.24 的运行中快照当作现在仍在跑。精确时间/进程与证据见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **问题已定位、尚未修复**：非有限值只在 novelty_vs_conditioning 及其聚合，和严格 JSON 写出不兼容；不是模型输出 NaN、无效候选或显存失败。完整 14×100 设计和 14 个独立 greedy 已保存，可在修复后复用。问题定义/后续验收边界唯一见 [T4 §7.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，不因只读重评分能计算部分值就标整个 T4 成功。
- **数字纪律**：继续使用 RESULTS §0.1–§0.3 的 49000 已核对结果；本次没有将临时重算数字写入正式 T4 表。尚未启动两项不能以历史 ckpt 或旧测试结果补齐。

### 9.26 2026-09-14：修复 T4 评分写出，复用已生成序列并接续本地队列

- **用户决定**：“修复一下”；按此前授权恢复同一 49000 矩阵，不扩展到 αβ 生成、baseline 重跑或 AB 任务。已完成七项不重跑，benchmark14 不重新采样，剩余仅 held20 → 大数据 T1 β-only；没有云端 submit/cancel，没有改评测 YAML 参数、模型/原始数据/共享采样器。
- **修复与验收 owner**：缺失 novelty 的 null/原因/有效分母契约、不可放宽的主指标断言、恢复身份白名单与原始产物保护，唯一详见 [T4 §7.7 / §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。实现涉及 native generation runner、新 TCR result_io、generation artifact auditor；新增 19 项回归加原 81 项共 100 passed，新 GPU gate 与 benchmark14 独立全量重评分通过。未把修复前的 suite/gate 重新标成已覆盖写出边界。
- **追溯**：修复前源码保存于 [source_before](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/serialization_repair_v2/source_before)；v1 的 failed manifest、raw、旧队列/监控终态未覆盖。v2 记录旧 generation_identity 和源文件 SHA，恢复的 designs 与原件逐字节一致；新 [结果审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/benchmark14.json) 才是正式收数依据。JSON helper 通过精确 gitignore 例外跟踪，不放开数据/权重/输出目录。
- **恢复调度**：新增 [resume_tcr_49000_local.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/resume_tcr_49000_local.py) 与冻结 [local_resume_plan_v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/local_resume_plan_v2.json)。七个历史完成任务只从已哈希状态、通过的审计报告及一致产物继承；benchmark14 以显式 CPU 恢复模式接入，不将旧 failed 状态改为成功。两待执行任务启动前重查对应云端 Killed、代码/输入 gate/配置哈希、输出不存在及 GPU 空闲显存；持有旧/新队列锁，不影响其他 GPU 进程。完整真实计划 check-only 通过。
- **持续监控**：复用原 monitor 代码，新 [monitor_plan_v3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_plan_v3.json) 绑定接续队列及新 T4 auditor 身份，30 秒轮询、完成后只读 CPU 审计、异常停止并记录 attention_required、不自主改参/重试。实时 [status](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v3/status.json) 与 [49000-only results](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/monitor_v3/results_49000.json)，进程/时间快照唯一见任务账本。
- **结果与剩余项**：benchmark14 的正式 14-target control、paper_sparse13、common-6 及 v5 overlap 分层进入 [RESULTS §0.4][results]；只用指定 49000，不混入历史 ckpt。held20 和大数据 β-only 完整执行/审计仍待结束；配对生成继续后置。工程恢复不意味着论文条件对齐、全局去污染或实验功能结合已验证。

### 9.27 2026-09-14：继续收口当前矩阵，并建立 baseline / 旧 checkpoint 分层对照

- **用户要求**：完成未完成任务，列出与 baseline 以及最近旧 checkpoint 的比较。继续使用已授权、已运行的本地接续队列，不重复提交、不取消健康运行进程，不扩展配对生成或全参数微调。旧模型先按同一 v5 训练系列最近已有评测的 29000 整理；该选择已向用户作非阻塞确认，未收到其他指定前不借用 v3/allch 或其他训练系列。
- **对照事实 owner**：T1 出版社 others 工作簿分组、公开行集/正负比例差异、29k/49k 样本与协议核验见 [T1 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)；T3 论文 ESM2 版本标签与历史/当前评分分层见 [T3 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。新增官方原件的 URL/SHA 进入下载清单，出处入口同步 README §3.1 / BASELINE_VERIFICATION；不把论文转录升级成新复跑。
- **关键实现**：新增只读 [收集器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_tcr_49000_comparison.py)，复用既有 audit 的文件身份核验，分开当前 49000、29k 历史控制、论文值和既有 baseline。`--require-complete` 防止未审计任务被标为完成；`--wait-for-complete` 只等待既有健康 monitor，完成后生成新 JSON，异常/死进程/PID 复用/重复输出直接报错，不自主重试模型或改预算。新脚本不进入生产输入/gate，未修改运行中的评测源码/配置。
- **验证**：10 项收集/等待器 CPU 测试、py_compile、真实完整 monitor plan check-only 通过；真实新旧 others 逐行身份核对及分层转录成功。部分汇总明确 `complete=false`，不因代码检查通过而提前宣称余下 GPU 测试结束。测试/进程与时间记录见 [任务账本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **展示与边界**：所有当前成绩和 baseline / 29k 数字仅进入 [RESULTS §0.1–§0.4][results]；只增加比较，不改原始模型分数。T2 不忽略 retention/输入单位，T3 历史 baseline 不冒称多标签修正后复跑，T4 common-6 仅对齐目标而非全部输入/采样条件。旧模型缺失的大数据 T1/T2/T3/T4 结果明确空缺，29k/49k 源码差异保留，不作只改变权重的因果结论。
- **完成进展**：held20 已完整生成并通过 [独立重评分审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/held20.json)，新 49000 成绩和相同数据集的既有 baseline 对照进入 RESULTS §0.4。当前 [九项已审计快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/audited9_20260914.json) 仍明确 incomplete，唯一缺项为大数据 β-only；该任务已经接续本地、正常提特征。新增 [收口状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/comparison_v1/final_49000_vs_baselines_and_29000.status.json) 只有在全部任务通过后才生成 final comparison。工程验收不等于论文同版复现、完整预训练去污染或功能结合验证。

### 9.28 2026-09-15：按 AB / TCR 重排结果页，补齐最终收口与解释边界

- **用户决定**：上半部分 AB、下半部分 TCR；每任务集中 baseline 和 Ours 的明确 checkpoint，采样/协议消融单列；移除主文档中的过期状态、重复表和无关冻结任务。数字唯一维护在 [RESULTS][results]，旧正文完整保存在 [整理前归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，不删除任何原始实验产物。
- **收口补漏**：此前最终 JSON 已完成但 Markdown 停留在 8/10 或 9/10；本次补齐 T1 大数据 β-only 结果及各入口状态。完整当前矩阵沿用指定 49000，29k 只作已存在的 others 历史控制，不把未运行的旧模型任务填成完成；最终 comparison 与时间/来源核验见 [任务账本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- **比较口径**：T1 四轨分表；T2 保留 retention/输入单位；T3 broad/deep 分表并保留论文与历史本地差异；T4 分开全14、sparse13、common6、held20及已见/未见分层。common6 不是所有模型的最大覆盖交集。论文 [P]、官方预测 [A]、历史复跑和当前结果沿用既有证据层，不用缺失的论文值或其他任务数字补空。
- **上一轮用户疑问的归档位置**：T4 去污染以参考 core＋表位配对为键，不代表表位级整体排除；七个已见表位的来源及训练语料成员/实际消费边界补充在 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)，不在本台账复制整张 overlap 表。本次未更改去污染规则或训练数据。
- **本次操作范围**：文档整理及既有 JSON/日志只读核验，未训练、重采样、重新推理、submit/cancel 或改变生产配置。AB 的 92000 仅用于已有 Specificity 对照，不扩散到其他 AB/TCR 任务；AB 采样矩阵与历史混杂控制不宣称是 TCR 消融。工程完成不解除任务文档中的科学可引用性限制。

后续每次关键改动追加：用户决策、涉及代码/协议版本、数据与模型身份、验证结果、是否产生新指标、尚未完成项；只在其他入口文档放链接/状态，不复制台账。

[query]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/query_protocol.py
[protocol]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/retrained_protocol.py
[query-test]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_binding_query.py
[ours-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py
[repr-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_repr.sh
[record]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py:676
[recognition]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:511
[pool]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py:751
[cache-check]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py:243
[esm-embed]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py:63
[legacy-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run.py:57
[nm-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py:254
[esm-old]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding/esm2_150m/metrics.json
[baseline-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py
[completion]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py:223
[loader]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/load_fusion_checkpoint.py:193
[no-grad]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py:577
### 2026-09-15 直接关系 token 本地评测完成

原 prepared-validation scorer 不能直接代表 NM2025，已新增官方测试行入口，先跑 others β-only AS 全集。严格复用已验证的无标签 query adapter；训练 target mask 不用于从未标注记录自动寻找查询。新实现从结构确认的唯一 recognition MASK 取两 token logit 差。五个 seen test 分区不是五次头训练，independent 只算一次；具体契约/入口统一见 [任务 §4.1a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)，执行状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

本轮 others β-only 全集已完成，55 项输入协议测试及保存预测的重新评分检查通过。按用户决定，该诊断实验不列入 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 正式结果；原始分数保留在任务文档所列输出目录。未测试其他三轨，不把本次现象外推为全任务结论。

[relation-score]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_binding_relation.py:307
[old-audit]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/audit_2026_08_29/T1_binding.md
[results]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md
[atm-nb]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/ATM_TCR/ATM-TCR.ipynb
[atm-attn]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/ATM_TCR/attention.py
[net-nb]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/NetTCR/NetTCR.ipynb
[tein-nb]: </vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/TEINet(TEINet-large_TEINet-small)/TEINet.ipynb>
[tein-model]: </vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/TEINet(TEINet-large_TEINet-small)/model.py>
[ergo-nb]: </vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)/ERGO.ipynb>
[ergo-model]: </vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/ERGO(ERGO-AE-vdj_ERGO-AE-mc_ERGO-lstm-vdj_ERGO-lstm-mc)/ERGO_models.py>
[teim-nb]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/TEIM/TEIM.ipynb
[teim-model]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/TEIM/scripts/model_raw.py
[epi-nb]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/epiTCR/epiTCR.ipynb
[tcrh-nb]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark/TCR-H/TCR-H.ipynb
