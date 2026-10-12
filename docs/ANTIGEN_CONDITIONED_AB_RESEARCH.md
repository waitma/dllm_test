# 抗原–抗体 benchmark：构建方案与历史调研

## 当前方案：三个任务、四个数据子集（2026-09-27）

本节取代下文早期 A/B/C 的实施顺序，保留历史调研与代码审计作为依据。只测 conventional paired VH/VL，不测 VHH；FLAb2 排除，MAGE 保留。完整抗体序列在本项目指配对可变区，不包含 IgG 恒定区。

| 任务 / 子轨 | 数据来源 | 模型输入 → 输出 | 核心指标 |
|---|---|---|---|
| Binding prediction | CoV-UniBind binding | Ag + VH + VL → 条件分数；监督探针输出概率 | 固定抗体比较抗原、固定抗原比较抗体：ROC-AUC、AUPRC |
| Affinity prediction | CoV-UniBind SPR | Ag + VH + VL → 强弱排序；核验后才输出 pKD | 双向 Spearman；定量子集 MAE、RMSE |
| Affinity landscape evaluation（亲和力子轨） | AbBiBench | Ag + 配对抗体变体 → 条件分数相对 WT 的差值 | 同一实验内 Spearman |
| Antigen-conditioned generation | CHIMERA-Bench 配对序列 | 仅 Ag → 完整配对 VH/VL | CDR-AAR、H3-AAR、CAAR、双链有效率、多样性、新颖性、可开发性问题基序数 |

数据与协议出处：[CoV-UniBind](https://huggingface.co/datasets/InstaDeepAI/CoVUniBind)、[AbBiBench](https://github.com/MSBMI-SAFE/AbBiBench)、[CHIMERA-Bench](https://github.com/mansoor181/chimera-bench)。清洗后数量以实际数据审计为准，不能直接采用论文总数。

### 协议边界

- CoV-UniBind 按方向分组统计，AbBiBench 按 assay 排序；不汇总不同实验终点为统一 KD。原始 SPR 字段是 `log10_kd`，单位、重复测量和检测限未经核验不得直接转换 pKD。Enrichment、IC50、ΔG、ΔΔG 均不进入 KD 回归。
- CHIMERA 原任务是表位条件 CDR 序列/结构设计。本项目建立独立的 `chimera_sequence_only_v1` 派生协议，不复用原结构条件榜单比较。生成端不能访问参考 framework、H/L 长度、表位或结构；CAAR 的参考接触位点只在评分端使用。
- 参考恢复是序列相似性诊断，不等同于真实结合成功率。多个参考形成同抗原参考集合；对候选取最大参考恢复后平均，best-of-20 仅作附表。无有效接触编号映射时 CAAR 标不可计算。
- 不使用 RMSD、DockQ、EpiF1。正确/替换抗原采用相同随机种子与预算，替换抗原仅作为条件敏感性消融，不称实验 nonbinder。
- 固定划分并审核实际预训练数据的抗原、VH/VL、H3 和配对重叠。原始划分、过滤后派生划分及 seen/overlap 状态分别保留；不能仅凭随机切分宣称未见抗原泛化。

### 已确认的首轮测试

用户选择无 ESMC **150000 步独立快照**，权重 SHA256 `a6ce5c2c22b2e938f4573eb552b6a8ce1c2f42a3647a345adc98e6a5a052b048`；身份以内部 step/config/hash 核验，不由目录日期推断。

- **Zero-shot**：冻结骨干，对 VH/VL 实际残基逐位 mask，计算条件伪对数似然的残基均值；不计 EOS/padding。AbBiBench 另减同一 assay 的 WT 分数。此分数不是结合概率或 pKD。
- **冻结骨干探针**：联合 Ag/H/L 前向，分别池化三个链的有效残基后拼接；binding 使用 StandardScaler + LogisticRegression，核验后 pKD 使用 StandardScaler + Ridge。只在 train 拟合标准化，正则网格 `{0.01,0.1,1,10,100}`；按 validation AUPRC / RMSE 选模，不使用 test 选模。与 zero-shot 分表。
- **生成**：每个唯一抗原 20 个候选，32 次去噪，CFG=0，temperature=1，seed=42；使用当前固定槽位和链 EOS，自主生成双链，不读取参考长度。保存全部尝试与失败原因。
- 外部 baseline 使用作者原始 GitHub 实现；本项目的数据适配、公共指标和本模型评分单独注明 provenance，不冒充作者原始复现。

先完成文档、数据与协议冻结，再进行 CPU 检查、GPU smoke 和正式评测。pKD 或 CAAR 等单项未通过核验只阻断该项，不阻断已就绪子轨。执行细节和实际状态分别以[任务手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)与[结果页](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)为准。

## 历史调研与审计记录

2026-09-24 新增用户指定的 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)：亲和力回归明确使用 antigen，但原仓库不可访问，同名快照没有实现，checkpoint 链接实际为 ProtET；当前不纳入可执行 baseline。其 BioMap/A2binder 原代码作为回归候选继续核验，不能等同于实验 pKD。

**2026-09-24 当前决定：FLAb2 已排除；MAGE 保留。** FLAb2 已从 active source lock 和方法清单移除，不再处理其缺失代码或依赖。范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准。

**2026-09-24 实施更新**：本页早期建议已扩展为逐 benchmark 的官方协议审计，权威说明见 [AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)，每个 baseline 的入口与待核项见 [comparisons.tsv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/comparisons.tsv)，实际执行见 [AB-AG 执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)。只做 conventional VH/VL，不测 VHH；未完成原模型评测。下文 A/B/C 为先前建议，不代表已经实现或可以用自写 scorer 替代作者 baseline。

现有公开工作中，MAGE 最接近完整配对生成；Agate 接近 antigen＋paired 抗体分类；DuaDeep 接近三路序列亲和力回归，但后二者本轮尚未定位到完整官方实现。AbBiBench、AbRank、CoV-UniBind、MochiBind 多在比较 affinity ranking；FLAb2 因该任务不输入抗原已排除。不能以这些相邻任务宣称已补齐绝对 KD 回归。原论文和逐模型比较细节均收敛到上述任务手册。

历史记录（2026-09-15，不表示当前执行状态）。状态：代码核查与公开论文/作者仓库调研；下述实验是建议，尚未接入评测、下载新基准或运行生成。不改变现有 headline 范围。

## 1. 当前能力与测评缺口

既有模型方案已提出 antigen→抗体及 antigen+framework→CDR 设计，不能称作此前完全未考虑。训练的 `antibody_antigen` grammar 固定抗原并监督受体生成块，提供表达这些任务的能力，但不能由此推断已经学到抗原特异性。

当前 [CDR 入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/cdr_infill.py:45) 的 Dataset 读取 heavy/light、目标 CDR 和位置；`build_antibody_record` 只构造抗体链，正式循环第 215 行调用它。因而当前 CDR AAR 测评没有 antigen 输入，不能验证 antigen-conditioning。已有 H→L pairing 也不是 antigen→VH/VL 的替代。历史 CDR 协议、成绩和去污染状态以 [AB-CDR 手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md) 为准，本调研不重新判定旧成绩。

模型输出应称配对抗体可变区 VH/VL，不能默认包含完整 IgG 恒定区、表达或功能验证。只给抗原序列与同时给结构/表位/抗体 framework，是不同输入协议。

## 2. 最相关的公开方法

| 方法 | 实际场景 | 对本项目的用途 |
|---|---|---|
| MAGE | 抗原氨基酸序列→配对 VH/VL，无起始抗体模板；论文报告 RBD、H5、RSV-A 的实验验证 | 最接近当前序列模型的目标任务，优先研究其数据和候选筛选流程 |
| PALM-H3 | 抗原序列→CDR-H3；生成 H3 再置入指定重链，保留轻链进行后续评估 | H3 生成参考；与给定 framework 的 infilling、整对 VH/VL 生成分开报告 |
| DiffAb | 抗体–抗原结构上下文中 CDR 序列/结构设计；antigen-only 工具仍使用抗原 PDB 和默认或指定抗体结构模板进行 docking | 可参考 SAbDab 和 CDR 协议，但输入结构优势应单独披露 |
| IgGM / RFantibody | 结构、framework 或表位等条件下设计，具体条件随版本/模式不同；RFantibody 有实验筛选与结合/结构验证 | 结构条件对照和功能验收参考，不直接等同于 antigen-sequence-only |

证据：

- [MAGE 作者代码](https://github.com/IGlab-VUMC/MAGE_ab_generation)、[MAGE Cell 论文](https://doi.org/10.1016/j.cell.2025.10.006)。作者仓库公开生成脚本、训练数据压缩包和分析目录，并链接模型权重；早期调研时未下载；2026-09-24 已锁定原代码，权重首次遇到 HF gated 403，用户同意条款后已验证访问成功、完整权重已下载并记录哈希，当前状态以新增任务手册为准。MAGE 的实验成功也不等于对任意未见抗原的泛化已得到证明。
- [PALM-H3 原论文](https://www.nature.com/articles/s41467-024-50903-y)、[作者代码与输出定义](https://github.com/TencentAILabHealthcare/PALM)。作者说明 origin heavy/light 用于将生成 H3 放回抗体并评估，不应擅自解释为生成器的 framework 条件。
- [DiffAb 作者仓库](https://github.com/luost26/diffab)：`design_pdb.py` / `design_dock.py` 区分复合物设计和未结合抗原设计；后者默认使用抗体结构模板。
- [IgGM 作者仓库及版本说明](https://github.com/TencentAI4S/IgGM)、[RFantibody 原论文](https://www.nature.com/articles/s41586-025-09721-5)。
- [AgForce 2026 预印本](https://arxiv.org/abs/2605.21610) 报告某些设计模型忽略抗原的现象，可作为条件消融的研究动机；不把其理论概括直接外推为本模型已被证明忽略条件。

## 3. 建议分层验证

### A. 先验证模型是否利用 antigen

使用有真实 antigen–VH/VL 配对的、与本模型训练集核查过重叠的留出样本。固定抗体 framework/非目标 CDR，在同一批 H3 目标上比较正确抗原、另一真实抗原、null-context 三种输入；两路 encoder/decoder 同步改变条件，并保持目标位置、随机 mask、生成预算一致。错配抗原不能未经验证标为实验 nonbinder，优先挑选有测量支持的非结合条件；否则只作条件敏感性对照。

报告 H3 masked NLL/AAR 及正确条件相对对照的增益，按抗原先聚合再宏平均，并给出按抗原重采样的不确定性。可增加实验结合标签支持的候选排序（AUROC/AUPRC）作为诊断，但它与新序列生成功能是不同任务。framework 本身可能提示靶点，因此即便 infilling 好也需进入下一层验证。

### B. 再评估抗原序列→完整 VH/VL

只提供抗原序列和明确声明的生成约束，H/L 序列均待生成。当前固定槽位采样需要单独制定长度协议：第一阶段如提供 reference H/L 长度，必须标 known-length；这不是纯 antigen-only。纯抗原任务需使用不读取测试抗体的训练长度先验或经验证的自主终止方案。既有 pairing 的 reference-length 授权不自动变成此新任务的默认决定。

按每个抗原固定相同候选数、seed 和采样预算，保存全部候选及失败原因，分别报告原始与筛选后的结果。基本指标：成对有效率、编号成功率、CDR 多样性、唯一率、对全部训练抗体及 CDR 的最近邻距离。单个已知抗体不是唯一正确答案，不把与该参考的 AAR 当作主要生成功能指标。

结合能力需有独立证据：外部 binding/结构评分只是代理，不能用模型自己的 `<binding>` 概率自证；报告 off-target 面板、独立候选排序，并在可行时进行实验结合率、SPR/BLI 亲和力及表达/特异性验证。结构预测的高置信度不能直接换算成实验结合成功率。首个同输入 baseline 候选是 MAGE；PALM-H3 与结构模型按任务条件另组。

### C. 数据集候选与划分

SAbDab / SAb23H2 完整抗原配对可用于条件 H3 基准；现有仅抗体字段的转换入口需要补 antigen。MAGE 公布的靶点案例和生成抗体可用于目标案例与已有候选评分，不能把它们直接当成已去污染的公开统一测试集。

至少区分已见抗原/新抗体与未见抗原家族两层。与 prepared v5 全部来源核查抗体、H3 和抗原同源重叠；不能仅检查 ASD 或精确字符串。只有在训练中未出现的靶点族留出上评估，才支持新靶点泛化。若当前 checkpoint 已见过这些抗原，应报告 seen-target 或另选数据，不能靠改测试标签修复。

## 4. 与 eval 均值改动的关系

本轮基础模型 eval 改为各来源等权平均，只改变 checkpoint 的跨来源汇总。生成能力验证是独立下游评测，尚未接入 checkpoint 指标。后续若将其用于选模，必须独立 validation/test，预先固定指标；不能在最终测试集上选 checkpoint。

## 2026-09-24 BioMap 官方实现追踪

最新数据与 baseline 可复现性结论见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md)。

### 2026-09-27 评测接口补充

四子集已增加独立数据校验、输入导出、离线评分和可续跑适配器，使用说明见 [ab_antigen/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/README.md)。保留既有研究及官方 baseline 审计，接口测试不作为模型成绩。
