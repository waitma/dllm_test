# AB-AG — 抗原条件抗体 benchmark：官方比较协议与复现台账

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | AB-AG-GEN / AB-AG-BIND / AB-AG-KD |
| 范围权威 | [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) |
| 目标 | 抗原序列→配对 VH/VL；抗原＋VH/VL→是否结合；抗原＋VH/VL→实验 KD/pKD |
| 结果权威 | [RESULTS：AB-AG](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen) |
| 入口 | [official.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/official.py)，只校验和检查官方 CLI，无模型/评分实现 |
| 官方代码根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/ab_antigen_official` |
| 版本及文件哈希 | [sources.lock.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sources.lock.json) |
| 最后更新 | 2026-09-24 |
| 可引用性 | 协议调研及复现准备；尚无本任务模型推理结果，不能主张能力或论文复现成功 |

## 1. 任务定义与范围

本轮研究对象为常规配对抗体可变区 VH/VL。**不评测 VHH，不跑混有 VHH 的默认全数据入口。** 分别保留序列生成、结合分类、实验 KD 回归三条目标。文献中的 affinity ranking、突变 ΔΔG、neutralization IC50、enrichment 不自动成为 KD regression。真实负例与随机错配/构造负例分别报告。

### 1.1 模型适用性矩阵

| 模型臂 | 生成 | 分类 / 回归 |
|---|---|---|
| Ours-Diffusion | 可适配，须证实抗原进入实际解码上下文；长度协议另定 | 可适配，须先冻结官方任务与训练/选模协议 |
| Ours-BERT | 不进入生成任务 | 可作为表征模型；新增 head 不冒充作者 baseline |

### 1.2 相邻任务

CDR infilling、H→L pairing 不替代完整配对生成。提供 PDB、表位或 framework 的设计方法单列条件。旧 FLAb/MINT 本地 Ridge 与 NbBench 仍保持冻结；FLAb2 已按用户要求完全排除，不再做代码补全、环境安装或试跑；历史审计仅留在原始归档。

## 2. 参考文献调研：原论文到底比较什么

### 2.1 逐 benchmark 的比较协议

BioMap 的论文、官方源码、各 baseline 发布缺口及数据问题详见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md)；不将该 ΔG 任务等同于实验 KD/pKD。

用户新增 S²ALM 的逐任务/产物审计见 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)；当前只有论文值，未取得可执行模型。它的显式 antigen affinity 回归与固定 HER2 分类、CDR-H3 infilling 分开判定。对应数据及 baseline 追溯到 PALM/A2binder，未新建自写 scorer。

| Benchmark / 原始证据 | 比较对象与输入 | 划分、训练、选择与统计 | 本轮判定 |
|---|---|---|---|
| [MAGE：Cell](https://doi.org/10.1016/j.cell.2025.10.006)、[官方生成脚本](https://github.com/IGlab-VUMC/MAGE_ab_generation/blob/3953077baf87c8afb3fb6c15ecbeda160c9d8533/generate_antibodies.py) | 抗原序列条件生成 H/L；论文案例为 RBD、RSV、H5。不是多个生成模型在同一测试集上的统一 leaderboard | 训练留出用于 epoch 选择；发布脚本固定 WT RBD，epoch4，top-p 0.9、temperature 1、生成长度为 prompt 字符数＋250。作者另用 notebook 编号、过滤、做多样性/新颖性和候选筛选，再进行湿实验 | 最接近完整配对序列生成；stock demo 不能称未见抗原 benchmark；实验结合率不能用计算评分复现 |
| [AbBiBench 论文](https://arxiv.org/abs/2506.04235)、[官方 leaderboard](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/README.md) | 序列 LM、逆折叠、结构生成、结构预测与物理评分；对实验突变体打分 | 主要是 zero-shot affinity ranking；分别算各 assay 的 Spearman，再按官方 benchmark 聚合。不同 assay 的标注方向、缺失值、模型条件须逐项保持 | 排序附轨；不能宣称预测绝对 pKD。PDB 定义的抗原上下文也不等于自由输入任意全长抗原 |
| [AbRank 原论文及 Appendix B](https://arxiv.org/html/2506.17857v1)、[Makefile](https://github.com/biochunan/AbRank-WALLE-Affinity/blob/216a112d8f93b5011e5c2040dc2c02e95cd43f68/Makefile) | WALLE-Affinity、ESM2＋AntiBERTy、MINT 分别用 ranking/regression 训练；ANTIPASTI、GearBind、PBEE、FoldX 分别配 Boltz1/AF3 结构 | Balanced、Hard Ab、Hard Ag 训练轨；Unrelated Complex 与 Local Perturbation 测试轨。原 Table 2 用 pairwise affinity preference ROC-AUC，regression 模型也在此排序任务上比较 | 重要排序 benchmark；不把该 AUROC 写成 binder/nonbinder AUROC。结构方法与纯序列方法分列 |
| [AntiBinder 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC11744619/)、[官方代码](https://github.com/fdu-wangfeilab/AntiBinder/tree/f765f9b39732073b5615e4ab78a833ad12691ea8) | AntiBinder 对 TAGPPI、MARPPI、DNN-PPI、DeepPPISP、PIPR、GraphPPIS、AbAgIntPre、AttABseq、DeepAAI、DG-affinity、PESI；代码抗体侧为 H 链、抗原侧为 ESM 特征 | COVID/HIV/BioMap/MET；交叉验证、未见抗原及物种迁移；分类 AUC、MCC、F1 等。单抗原 MET 不能用于论证未见抗原泛化 | H-only，不进入 paired 主轨。公开 main_test 路径/权重留空，不能据此宣称全部论文对照可直接复跑 |
| [MochiBind 官方发布](https://github.com/amazon-science/ab-binding-eval/tree/0c693d815f7f0bf0a3dd5494205dca10fbc20c04)、[论文](https://www.sciencedirect.com/science/article/pii/S2589004226018973) | MochiBind、Graphinity fine-tuning、GeoDock、Boltz2；pairwise Δaffinity 与 candidate retrieval | 官方代码有 held-out target 和验证 target；最小 validation loss 选模。默认 abs_weight=0，pairwise correctness 使用预测/真实差值乘积>0；另有候选检索 notebook | 排序/检索附轨；默认包含 VHH72，禁止直接全量执行。绝对输出头不等于训练好的 pKD 模型 |
| [CoV-UniBind 论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC12800777/)、[官方数据与分数](https://huggingface.co/datasets/InstaDeepAI/CoVUniBind) | AF2 系列、Boltz1、Chai1；ProteinMPNN/MPNNsol/AbMPNN/FAMPNN/AntiFold/ESM-IF1；IgBERT/AbLang2 | 固定抗体给抗原排序、固定抗原给抗体排序；按抗体或抗原 lineage 分组，分别用分类 AUROC 或连续标签 Spearman。论文有最小组大小和方向约定；并非随机切分后 pooled AUROC | 两个 paired 数据子集值得作为候选；公开 loader 与已算分数不等于公开完整评测程序；尚不能宣称原代码复跑 |
| [Agate 原论文](https://openreview.net/pdf?id=pjCFQfWcCu) | Agate 对 PLM-Interact、MINT、MAGE、Boltz2；分别用分类头、PPI head、抗原条件 antibody perplexity、ipTM | antigen cluster 留出；构造负例与真实实验非结合不同；ROC/PR 和逐抗原相关，另有 Spike 排除轨 | 输入和分类任务很相关，但本轮未定位到作者完整代码/权重/固定 split；不能自写 MAGE PPL 代替 Agate 作者的比较实现 |
| [DuaDeep-SeqAffinity](https://arxiv.org/html/2512.22007) | antigen、H、L 三路 frozen ESM2 表征，Transformer/CNN 及融合回归；ESM-T/ESM-C 为消融 | 使用 AbRank；当前版本 data availability 指向 AbRank，未定位到作者实现/权重/可复查 split。不同论文中的回归数字不得直接并排 | 任务高度相关，列为源码待获取。不能根据方法图自行实现后标 official |
| [AbLWR Appendix E](https://arxiv.org/html/2604.11272) | AbRank、GraphDTA、MINT＋MLP、AntiBERTy＋MLP、LambdaMART、FoldX、ANTIPASTI | 作者对 baseline 做过任务适配；GraphDTA 调整训练预算，MINT/AntiBERTy 加 head，FoldX/ANTIPASTI 用 HDOCK；按 K=5 列表计算再平均的排序指标 | 列表排序相邻轨；下载各模型原仓库不等于拿到 AbLWR 作者的适配实现，当前待获取 |

结构生成补充：PALM-H3 是 H3 生成；DiffAb、IgGM、RFantibody 的结构/表位/framework 条件需单列。AB-Bind 是突变 ΔΔG 数据，不等于绝对 KD；数据汇编 ABPairDB/ANDD 需逐来源核查实验/预测标注；AlphaBind 可用于亲和力相关研究，但其数据重叠、链字段及官方任务版本仍需单独审计。这些线索不计入已核验官方仓库，也不默认全量运行。

### 2.2 为什么同名模型不能共用一个自写 scorer

以下是已读到的代码行为；即使行为有疑点，也先记录而不悄悄修正后声称原样复现。

| Benchmark / baseline | 官方代码行为 | 必须保留的差异 |
|---|---|---|
| AbBiBench / ESM2 | [脚本](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/models/ESM-2/get_model_log_likelihood.py)：3B 模型；突变 H＋L＋PDB 抗原；不 mask，平均 residue log-score | 不是 masked PLL；特殊 token 的索引对齐须单独核查，不自行改后仍标同版 |
| AbBiBench / CurrAb | [脚本](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/models/CurrAb/get_model_log_likelihood.py) 中整段 complex sequence 评分 | 不统一替换成 antibody-only 条件分数 |
| AbBiBench / AntiBERTy | [脚本](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/models/AntiBERTy/get_model_log_likelihood.py)：Ab 与 Ab-Ag 分支 | Ab 分支引用未定义变量/参数，异常可能写成空分数；退出码 0 仍须检查有效行数 |
| AbBiBench / ProGen2 | [脚本](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/models/ProGen2/get_model_log_likelihood.py)：支持 HLA/AHL/LAH 链顺序 | 默认 base 与 leaderboard large 不同；必须核实具体模型、顺序、评分残基范围 |
| AbBiBench / ProteinMPNN | [HC 入口](https://github.com/MSBMI-SAFE/AbBiBench/blob/7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8/models/ProteinMPNN/get_model_log_likelihood.py) 与独立 LC 入口 | HC 入口固定 native light/antigen，不能直接套 LC 突变体；固定骨架与采样结构也不能混用 |
| AbRank / regression | [配置](https://github.com/biochunan/AbRank-WALLE-Affinity/blob/216a112d8f93b5011e5c2040dc2c02e95cd43f68/waffle/config/train-abrank-regression.yaml)：max100 epoch、lr1e-5、val/mse 选模、patience10 | 测试代码导出预测；没有可直接引用的本地 KD 测试指标。registry 的 logAff 不能未经标签审核当 pKD |
| Agate / MAGE 与 MAGE 原生成 | 前者作为 binding scorer，后者采样生成 | 生成 script 不提供 Agate 的 PPL 评估定义；不能仅用同一权重认定协议一致 |

### 2.3 未公开或尚未核实的细节

缺失信息一律登记为 pending，不填猜测值：确切权重 revision、paper commit、所有 fold/seed、分数符号与缺失处理、测试集去重、预训练污染、baseline 适配 head、结构准备参数。默认分支的当前 commit 是可定位源码快照，**不是已证明与论文发表时完全相同的版本**。

## 3. 官方产物 release 与获取状态

| 来源 | 版本 | 获取状态 / 许可证证据 |
|---|---|---|
| PALM / A2binder | `f44090b3fbe5404ae0a9d196f5429a48ef4de5c5` | 原代码/数据/初始化权重已校验；单 seed BioMap 作业状态见 §8.6；不含五个对照方法的 BioMap adapter |
| MAGE | `3953077baf87c8afb3fb6c15ecbeda160c9d8533` | 代码已取；本轮查询无 GitHub Release。HF 权重 revision `dad1a221e992771666ba9d36b59de5bbe7015321` 首次 gated 403；用户确认同意条款后复查成功，完整模型权重/config/tokenizer 已落地并完成 SHA256 记录 |
| AbBiBench | `7d9c73e23c3c535fbb73ef073cf2fb9633e1fda8` | 代码已取；本轮查询无 Release。数据/权重未完整取得，逐产物 license 待核 |
| AbRank | `216a112d8f93b5011e5c2040dc2c02e95cd43f68` | 代码已取；本轮查询无 Release；大型图表征未下载，未跑训练 |
| AntiBinder | `f765f9b39732073b5615e4ab78a833ad12691ea8` | 代码已取；本轮查询无 Release；H-only 排除 paired 主轨 |
| MochiBind | `0c693d815f7f0bf0a3dd5494205dca10fbc20c04` | 代码已取；本轮查询无 Release；README 声明 CC-BY-NC-4.0。Git tree 中数据/权重 LFS 条目不等于已下载对象 |
| CoV-UniBind | HF dataset revision `1a3d3a135c21e2a07bf75c9eb7015585b168bf65` | 官方 README/loader 与 binding、SPR 子集单独获取；无重切分、无自写 baseline 评分。完整 evaluator 未定位 |
| Agate / DuaDeep / AbLWR | 尚未锁定作者评测仓库 | 检索未定位不是证明不存在；保留候选行和缺失项 |

逐文件 SHA256 与状态统一登记到 [获取清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv)。代码采用 sparse checkout；数据文件在本地不存在时先查 `git ls-tree`，不能直接断言作者没有发布。子模块、LFS 对象和外部权重需要额外锁定，父仓库 SHA 不替代它们的哈希。

## 4. 我们的协议

### 4.1 原代码优先的执行顺序

1. 按 benchmark 拆开每一个 baseline，记录作者仓库、tag/commit、脚本、数据、split、权重、输入、训练/选模、评分与聚合。
2. 用作者原入口复跑其原任务。只允许外围的下载、路径映射、环境隔离和 provenance；路径映射清单必须存档。
3. 原比较中有二次适配时，要取得 **benchmark 作者的适配代码**。仅下载基模型仓库不够。
4. 源码缺失、权重受限或原脚本有缺陷时保留失败记录。不用自写 head、分数或 split 补成 official baseline。
5. 将原 benchmark 限定为 conventional VH/VL 子集、补 antigen、重切分或转换标签都会形成新协议；必须单列，不能沿用原论文分数宣称同表可比。

### 4.2 口径切割线

| 分组 A | 分组 B | 分列原因 |
|---|---|---|
| affinity ranking / pairwise AUC | bind/nonbind 分类 | 标签和问题不同 |
| Spearman 排序 | 实验 pKD 回归 RMSE/MAE | 排序对单调变换不敏感，无法证明绝对量纲正确 |
| KD | IC50 / enrichment / ΔΔG / 模拟能量 | 测量意义与单位不同 |
| measured nonbinder | shuffled/naive constructed negative | 不能把构造标签称为实验真值 |
| full antibody + antigen sequence | PDB/epitope/framework 条件 | 可用信息不对等 |
| benchmark 作者正式适配 | 我们自写适配 | 来源不同；本任务不以自写代码替代官方 baseline |
| 原始全数据 | 去 VHH 后的 VH/VL 子集 | 测试样本变化；不能无说明借用原论文汇总值 |

KD 转换须逐行明确 M/nM 等单位、测定体系与 censoring。CoV-UniBind SPR 的重复上界值须核对是否检测限；没有证据前不生成 pKD 主榜。不能用 logAff、ΔΔG 或模型 score 填补实测 KD 缺口。

## 5. Baseline 清单与排行榜构造

### 5.1 来源规则

沿用 `[P]` 论文值、`[A]` 官方 artifact、`[R]` 原代码复跑；不把未跑通行删除，也不把 help/静态检查标为 `[R]`。只在确切协议与数据子集一致时并排。每个论文值须来自原表并带版本/指标；尚未逐格转录的写“未获取”，不填臆测数字。本任务不新建 `[L]` 代替缺失 baseline。

### 5.2 每个方法的可审查入口

[comparisons.tsv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv) 为逐项方法索引，包含原代码链接、协议组、复现限制。AbBiBench 的 leaderboard 各行均保留；FLAb2 各入口均已从本清单移除。候选表不是运行成功列表。

## 6. 结果写入位置

| 内容 | 唯一位置 |
|---|---|
| 任务数字与无结果状态 | [RESULTS AB-AG](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen) |
| 源码原始审计与 CLI 日志 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924` |
| 当前权重获取记录（初次失败另存 acquisition_initial_403.json） | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/ab_antigen_official/MAGE_V1/acquisition.json` |
| 数据来源与哈希 | [获取清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv) |
| 过程 / 作业 | [PROGRESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md)、[PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) |

## 7. 已知缺陷与不可比口径

2026-09-30 endpoint 状态拆分：SPR ranking 的 complete 不授予 pKD regression complete。单位和删失问题未解决时 pKD 保持 blocked；不同 endpoint 分别见 [独立完整性报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。当前修复不调整 binding 划分的统计支撑、不修改现有数据，也不改运行中评测锁定的源码。

| 项目 | 当前状态与影响 | 源码证据 |
|---|---|---|
| MAGE 权重 | 完整权重已取得并核验；官方单样本生成检查已提交，尚无成功结果 | §6 acquisition.json；官方脚本第35行附近模型路径 |
| AbBiBench 聚合 | README 指向的 `notebooks/figure.ipynb` 不在锁定 Git tree；不能自写聚合冒充作者最终结果 | README evaluation 段，source audit required-files 字段 |
| AbBiBench wrapper | 仅支持一部分模型；相对目录依赖；不得用 wrapper 支持数推断全部 baseline 已可复跑 | `scripts/eval_seq.py` VALID_MODELS / script_path |
| AbBiBench AntiBERTy | Ab 分支引用 `args.scoring_col` / `mutated_chain_seq`，存在生成空分数风险 | `models/AntiBERTy/get_model_log_likelihood.py` |
| AbBiBench AF3/Boltz2 | 有 score artifacts，不等于发布对应 inference pipeline | 锁定 Git tree；尚无原推理复跑 |
| AbRank README | Evaluation 的 ImageNet 示例不适用；须追踪 Makefile/Hydra | `README.md` 与 `Makefile`；回归测试只导出分数 |
| AntiBinder | H-only；main_test 权重/路径为空 | `antigen_antibody_emb.py` / `main_test.py` |
| MochiBind 默认任务 | 含 VHH72；不能在本范围直接执行默认训练/全 notebook | 官方 dataset/train 配置；未调用 |
| 运行环境 | 初次 CLI 在 pllm 缺 Bio/biotite/hydra/abnumber 等；这是环境失败，不是算法成绩差 | cli_probe 原始日志；不能据此断言模型不可复现 |
| 污染与新靶点 | 当前模型训练来源和公开集合可能重叠，尚未做完整同源去污染 | 正式 Ours 测试前必须冻结 train/val/test provenance |

### 7.1 代码缺口分类（授权复查后）

- **锁定 Git tree 确认缺文件**：AbBiBench README 承诺的 `notebooks/figure.ipynb`。
- **只有部分 baseline 链路**：AbBiBench 有 AF3/Boltz2 分数文件，但未定位对应推理脚本；AbRank 有 WALLE 主方法，未定位论文中其他对照的完整适配/评分链路。
- **尚未定位完整作者发布**：Agate 的模型/训练评测/split；DuaDeep 的实现/权重/split；AbLWR 的 benchmark baseline 适配；CoV-UniBind 有数据、分数和 loader，但完整 ranking evaluator 未定位。这些是检索状态，不断言作者没有发布。
- **不是缺代码**：MAGE 主生成代码已取得、授权复查成功；MochiBind 有训练和评测 notebook，但默认含 VHH，不能直接全量运行；AntiBinder 有代码，但 H-only 且测试权重/路径不完整。
- **另列运行障碍**：CLI 的 Bio/biotite/hydra 等 import 失败是本地依赖问题；AntiBERTy 未定义变量是上游代码质量问题，不能混称“没开源”。

### 7.2 BioMap 原代码回归候选

执行 gate 以 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md) §3–5、§7 为准：初始化权重/环境已验证并提交首轮运行，原 H/L 列存在不一致，完整重复实验划分及五个 baseline adapter 缺失。原入口为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/ab_antigen_official/palm_a2binder/Code/bert_finetuning_er_main.py`，配置为同仓库 BioMap 专用 JSON；须先在独立运行目录映射作者配置的权重路径，再使用原脚本。具体作业与已验证条件见 §8.6；当前状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。

## 8. Agent 执行手册

### 8.1 提交前 gate

- [ ] 固定官方 tag/commit，校验 origin、工作区与所有代码哈希；依赖/子模块/权重版本另存。
- [ ] 每条数据明确 conventional VH/VL 且含实验对应 antigen；禁止 VHH 默认全量运行。
- [ ] 正式比较使用作者 split、训练配置、选模准则、seed、原 scorer/aggregation；新增过滤单列协议。
- [ ] 权重拿到且匹配架构；数据/单位/censoring 明确；无用假数据通过 gate。
- [ ] 所有输出样本与预期集合对齐，空值/异常行登记；退出码 0 不足以判成功。
- [ ] Ours 生成与 BERT 适用性、抗原上下文、长度与去污染协议已固定。

### 8.2 当前可直接执行的检查

先获取锁定源码；本地已有源码无需重复 clone。以下不下载权重、不运行模型评分：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python -m downstream.benchmark.ab_antigen.official audit --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_audit_new
python -m downstream.benchmark.ab_antigen.official probe --source mage --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_mage_help_new
python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/test_ab_antigen_official.py -q
```

输出目录必须不存在，避免覆盖证据。audit 退出码 1=来源校验失败，3=上游静态缺陷；probe 退出码 2=CLI 失败。`source_verified` 仅说明原样取得代码，`static_checks_passed` 才表示列出的静态检查通过，二者都不表示可以完成论文评测。

### 8.3 从空目录恢复官方源码

按 lock 中每个 source 的 URL clone，checkout 到确切 commit；可使用与现有 acquisition 一致的 sparse 模式排除数据/权重。**不能从 main 最新版本重新生成 lock 然后说校验通过。** git checkout 后运行 §8.2 audit。外部子模块和 LFS 数据须显式获取并登记；原作者 README 的环境只装到新环境，不修改项目训练用 pllm。

### 8.4 正式运行与收数

当前未提供“全 benchmark 一键出数”命令，因为其依赖条件尚未满足。MAGE 权重授权、完整文件下载及哈希已完成。现已在独立工作目录以软链提供作者要求的 `model_epoch4` 与 `progen` 包布局，保留脚本字节不变；已提交 Volc eval 作业调用官方 `generate_antibodies.py -n 1 -o <absolute-output.csv>`。这是 WT-RBD smoke，只登记生成/加载是否成功，不作为实验 binder 成功率。

其他每一行必须先解决对应 §7 gate，再编写只调用原入口的 Volc 配置。GPU 作业统一 `volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/<job>.yml`，提交/取消同步 PROJECT_PROCESS。输出保留 argv/cwd、环境包版本、输入/权重/源码哈希和原始预测，使用作者原统计程序；没有作者统计程序时标 pending，不自动改写。

### 8.5 填表与同步

只将真实 scores 写入 RESULTS；论文表 `[P]` 与复跑 `[R]` 分列，协议不一致单列。无成绩保留“未运行/阻塞”及原因。同步 RESULTS、PROGRESS、本文档及 PROJECT_PROCESS；范围变更同步 README §0 的四个镜像点。

### 8.6 BioMap A2binder 原代码运行

- 发布代码由 source lock 固定到 `f44090b3fbe5404ae0a9d196f5429a48ef4de5c5`；入口 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/ab_antigen_official/palm_a2binder/Code/bert_finetuning_er_main.py`。
- 初始化资产：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/model_weights/ab_antigen_official/PALM_BioMap/acquisition.json`；包含作者预训练 H/L RoFormer 与 ESM2 revision `a695f6045e2e32885fa60af20c13cb35398ce30c`。ZIP 按成员 CRC＋本地 SHA256 验证，未下载完整归档，不能声称通过整个 Model.zip MD5。
- Python：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/palm_env/bin/python`；使用 Python 3.9 和作者 requirements 的核心版本；额外安装 TensorBoard。完整 freeze 在 run 目录。
- 运行目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3`；`config.json` 仅修改路径，`path_mappings.json` 显式列出差异；`Code/facebook/esm2_t30_150M_UR50D` 为硬编码 tokenizer ID 提供本地权重别名，不修改源码。
- GPU 编排：[run_biomap.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/run_biomap.py) 校验来源/权重/原协议、拒绝 CPU 和覆盖结果，然后启动未改动的作者入口；结束后调用作者 metric 函数汇总原始测试预测。
- 作业：[biomap_seed3.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/biomap_seed3.yml)；实际提交文件 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_biomap_a2binder_official_seed3_20260924.yml`；**已经提交，不重复执行 submit**。状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。
- 若失败，保留 report/log 和已有 checkpoint；新尝试使用新 run 目录并另记协议差异，不能删除失败痕迹或擅自换模型。GPU 性能尚未验证，CPU preflight 不能证明整轮训练成功。

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-24 | 按用户新增要求，把 benchmark 作者原实现、原划分/权重/评分作为基线复现依据；排除 VHH；新增源码锁、方法索引、真实失败台账；未生成 benchmark 成绩 |

### 2026-09-24 后续执行

FLAb2 已排除出 active sources 及方法清单；历史快照位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/before_flab2_exclusion`。MAGE 作业 `t-20260924115209-csmbz` 已提交，1 GPU、c20250601 闲时，平台当前 Queue。启动配置：[mage_smoke.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/mage_smoke.yml)；官方代码原样执行，仅做依赖包路径映射和 provenance 包装。Python/torch 版本与作者建议不同，记录在 generation_report 中；此次 smoke 不等于完整 benchmark。

## 8.7 三任务 / 四子集纯序列协议（2026-09-27）

当前构建方案见[研究文档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ANTIGEN_CONDITIONED_AB_RESEARCH.md)。本节为本项目派生协议，不能标成 CoV-UniBind / CHIMERA 官方模型复现。外部 baseline 仍必须使用作者代码；本模型评分和探针是 Ours 实现。

- 数据根：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/ab_antigen_sequence_v1`。四个 `binding/spr/abbibench/chimera.jsonl` 保存稳定 ID、Ag/VH/VL、标签及来源；`build_audit.json` 保存逐项排除原因。原始资产和 SHA 在 `raw/acquisition.json`。AbBiBench 分数方向核验：[论文 §3.1](https://arxiv.org/html/2506.04235v2#S3.SS1)，负 log KD / log enrichment 均为越高越强，仍禁止混合做 KD 回归。
- VH/VL：CoV-UniBind 使用官方 VH/VL 字段；CHIMERA 使用官方 IMGT 编号定位唯一连续可变区并同步裁剪参考接触映射；AbBiBench 使用 ANARCI 验证 PDB WT 的可变区边界，变体只在边界外与 WT 一致或已经是同长 Fv 时映射。缺链、非标准残基、无法核验的边界、模型溢出明确排除。
- 输入上限 2048，H/L canvas 为 167/135（含 EOS）；抗原不裁剪。训练数据预算曾为 1024，因此超过训练预算的输入须作为长度外推披露，不能声称全部处于训练长度分布内。
- CoV-UniBind 使用未按表位裁剪的 `antigen_sequence`。SPR `log10_kd` 取负用于来源内排序；单位与 censoring 未核验前没有 pKD。默认分组为固定 antibody sequence pair / 固定 antigen lineage；项目宏平均分类组至少两行且含两类，相关组至少三行且标签与预测非恒定，不冒充未公开的完整官方 evaluator。
- 探针划分：MMseqs easy-cluster、connected component cluster mode 1；抗原 70% identity、H/L 90%、双向 coverage 80%，seed42，80/10/10（至少一组 val/test）。配对抗体按 H/L cluster 二元组划分。低于三组保留 zero-shot，不生成伪监督三分。`split_audit.json` 和 `*_splits.json` 固定成员。
- 模型：无 ESMC step150000，独立快照及 SHA 见研究文档。单残基 mask、H/L residue mean PLL；冻结三链联合前向均值拼接，LogisticRegression / Ridge，训练集拟合 scaler，验证集选正则，测试集不选模。
- CHIMERA 保留官方 antigen_fold split，排除相同抗原跨 split；纯序列派生测试仅给 Ag，固定 K20 / iter32 / seed42 / temperature1 / CFG0。正确与替换抗原共享候选 seed，替换采用排序后循环移位；不当作真实非结合标签。
- 生成评分独立进程读取参考：IMGT 对齐 CDR 27–38、56–65、105–117；缺位计错。CAAR 采用参考 CA–CA <8Å 接触，生成器不接收这些信息。编号失败记录为失败，全部生成尝试保留。
- [sequence_data.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sequence_data.py) 构建数据；[sequence_audit.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sequence_audit.py) 划分/训练重叠；[sequence_eval.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sequence_eval.py) GPU 评分/探针/生成；[sequence_metrics.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/sequence_metrics.py) 独立编号/生成评分。

运行顺序（先激活 pllm，再使用 protenix_abtcr Python）：`acquire_sequence --features` → `sequence_data` → `sequence_audit --training` → `sequence_eval --preflight` → Volc smoke → 正式 GPU eval → `sequence_metrics`。每个模块可通过 `python -m downstream.benchmark.ab_antigen.<module>` 调用，eval 必须指定 `--task` 和新建的 `--out-dir`。断点缓存要求数据/代码/checkpoint provenance 完全匹配。

当前 smoke YAML：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_ag_sequence_v1_noesmc150000_smoke.yml`；提交时覆盖 `--resource_queue_name c20250601`。状态和分数唯一来源见[结果页](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-ag-sequence-v1)。训练重叠未完成的样本不得标 clean；原始来源和 H3 未充分核验时保留 unknown。

#### 自动执行与审计补充

- 原始下载已完成。AbBiBench metadata 中 AAYL50 PDB 大小写问题已按唯一同名文件解决；仍缺作者引用的 `1mlc_LC_benchmarking_data.csv` 和 `1mhp_benchmarking_data.csv`，当前公开仓库无该两文件。17 个实际公开 CSV 均已处理。
- 正式作业拆成 binding、SPR、CHIMERA 和 17 个 AbBiBench assay，共20份 YAML，按单GPU顺序推进，单作业上限86400秒；首轮必须先通过四子集 smoke。任何作业失败/被抢占时停止自动推进并保留可恢复分数，不无限重试。
- 执行计划 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000/execution_plan.json`；看护 `sequence_pipeline.py --watch`，状态 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000/pipeline_state.json`，日志 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000/pipeline.log`。创建 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000/STOP` 仅停止后续提交，不取消既有作业。
- 精确重叠通过 `sequence_overlap.py` 可重跑，输出 `audit/training/exact_final.json` 与逐样本 `overlap_rows.json`；H3 使用 IMGT105–117 与 prepared heavy_h3_core，原始未进入 prepared 的来源仍属 unknown。
- MMseqs 原方向大库索引发生崩溃；改用训练序列为 query、benchmark 为 target 后继续检索，输出列换回 benchmark-first。每训练序列最多100个命中，属于启发式同源审计，不把无命中称为不存在同源。生成新颖性同样分别报告精确匹配和检索到的最近命中，搜索失败只阻断该项。
- CPU 数值/协议测试10项通过；真实 ANARCI native fixture 的 paired validity / CDR-AAR / H3-AAR / CAAR 已验证自恢复为1。这是评分程序检查，**不是模型生成成绩**。

## 8.8 统一接口（2026-09-27）

新增四子集 validate/export/evaluate 与 Python adapter；具体命令和 I/O 契约见 [ab_antigen/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/README.md)。预测按稳定 ID 对齐，缺失结果须显式标 partial；生成器不接收参考 VH/VL，正确/替换条件共享 seed。断点续跑锁定数据、代码、模型和采样配置。SPR 仅核验 pKD 子集进入定量回归；AbBiBench 保持 WT 差分。native_adapter 接入指定 no-ESMC 150000 zero-shot，冻结探针仍由 sequence_eval 管理划分与训练。本轮仅 CPU 校验和接口测试，未新增 GPU 提交。

### 2026-09-28 本地执行覆盖

用户明确要求本地测试、不提交任务；已停止自动提交并取消 Volc smoke（`t-20260928062524-wls4h`，终态 Killed）。本地 A100 80GB 上启动固定 no-ESMC150000 四子集 smoke，PID833743，日志 [ab_ag_sequence_v1_noesmc150000_local_20260928/run.log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/run.log)。沿用 K20 与正确/替换条件，未改指标协议；启动不等于完成，未获得有效完整模型成绩。

## 2026-09-30 PLL 批量调整并恢复本地评测

用户要求扩大批量。以剩余4fqi_h1/h3的6条代表记录实测批量2/16/32/64/128；平均每条1.926/1.334/1.379/1.313/1.396秒，分数最大绝对差均为0；batch64峰值allocated显存31.08GiB。选择64，约1.47倍加速（小样本计时，不保证全程同比）。保留4fqi_h1已有20404条完整预测，原评分代码和checkpoint不变，运行参数变化记录在各输出目录batch_execution_change.json。本地流水线PID1614629支持读取既有状态并跳过完成项，续跑h1及h3；未提交集群任务。验证记录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928/batch_benchmark.json`。


### 2026-10-09 评测修复

ANARCI check_for_j出现NoneType+int时递归拆批，隔离单序列并记录numbering.errors.json；该候选继续计入失败分母。其他类型异常继续抛出，不静默吞掉环境故障。CHIMERA仅对保留预测重新编号与评分，不重新生成。
