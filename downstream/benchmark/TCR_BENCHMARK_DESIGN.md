> ⚠️ **ARCHIVE / 非现行入口（2026-09-02）**：协议与执行手册已迁到 [`tasks/`](../tasks/)。数字只写 [`RESULTS.md`](RESULTS.md) §0。
> **T2 universe 已修正为 4,779**（pandas/dplyr NA 语义差，不是数据库漂移）。下文若仍写 5,368，以 `RESULTS.md` 与 `tasks/TCR_T2_CLUSTERING.md` 为准。
> 不要给本文填新数字。

# TCR Benchmark 设计主文档（文献驱动 · 源码-only baseline · 实时更新）

> 状态文档，**随进度实时更新**（见文末「变更记录」与每任务「状态」列）。
> 定位：TCR 领域**缺一个统一、防泄露、可复现**的下游基准 → 我们从最新高引文献里挑「权威评测口径」迁移过来，自建 IRBench-TCR。
> 配套：PPI 维度对齐 **MINT (protein)**、抗体维度对齐 **Ophiuchus-Ab**，两者不并入 TCR 四类（见 §6）。
>
> **硬性原则**
> 1. **baseline 只用官方源码 / 官方权重**，不自行复现；每个 repo 记录 clone commit（沿用 `baselines/README.md`）。
>    T1 external baseline 只走 original checkpoint + 官方推理 + `original.zip`，不读 `train.csv`、不 retrain；不可运行时引用论文 original-model 值并标“论文值，未本地复现”。
> 2. 每个任务只保留 **1 主数据集 + 1 主指标**（贵精不贵多），其余为辅报 / 消融，不另开榜。
> 3. **防泄露断言是核心卖点**：克隆型 / 序列相似性去重、seen vs unseen 必须报告，落 `leakage_report.json`。
> 4. 我们的模型 **headline 口径 = post-LLaDA**（`bioseq-llada:` / `grammar:decoder:` → `EsmcLladaBioSeqEmbedder` / `GrammarEmbedder(feature_source="decoder")`）；`bioseq:`（`EsmcBioSeqEmbedder`，仅 ESMC encoder mean-pool）**仅作 encoder-only 对照**，不得写入 headline。详见 §5。

---

## 0. 四类任务一览（投稿口径）

> **TCR 下游 = 四大方向：T1 Binding · T2 Clustering · T3 Representation · T4 Generation。**
> 这四类由最新文献一致支撑，无第五大类：binding（Nat Methods 2025，50 模型/21 库）、clustering（NAR-GAB 2025，190k TCR/2313 表位）、representation（SCEPTR, Cell Systems 2024）、generation（TCRT5 Nat Mach Intell 2025 / TcrDesign / TCRTransBench / LSMTCR）、以及 PLM 综述（Int Immunol 2025 `dxaf048`）都围绕这四个方向组织。
> **Generation 是一个方向**，其下按"是否给定表位条件"分两种**评测 setting**（非两个独立榜）：**无条件**（repertoire 分布匹配）与**表位条件 / design**（给 epitope 生成 binder）。二者是**同一个生成任务的两种 conditioning 设定**，共享生成器主干与序列级指标，只是条件设定与功能性指标不同。

| 类 | 主任务 | 主数据集 | 主指标 | 防泄露要点 | 状态 |
|----|--------|----------|--------|-----------|------|
| **T1 Binding** | TCR–epitope 二分类，seen/**unseen** | **Nature Methods 2025 官方 split**（主）；IMMREP23 辅历史 | **unseen overall AUPRC**（辅 macro-AUPRC/AUC0.1/AUROC） | **近似去重（Lev≤k, §L1）**+ unseen 表位 + 负样本来源(AS/PS/HS §L2)受控 | 🟢 **original-only 来源选择已落地（2026-07-21）**：15 个论文 original-model 行中 13 个使用官方 checkpoint/code/`original.zip` 本地结果，SETE/TEPCAM 明示论文 fallback；另有 ERGO-II、kNN/Random 与 Ours controls；L1/L2 保留为本地 sensitivity |
| **T2 Clustering** | 无监督聚类 → 表位特异性 | **双基础**：A=NAR-GAB 2025 / i3-unit（配对 **4,779**/374 表位；旧文 5,368 是 `curate()` bug）；B=Brief Bioinformatics 2025 / TCREmbedding（单链 9,033 CDR3β/25 表位） | A: **Purity–Retention 权衡**（辅 Sensitivity/NMI）；B: **ARI/NMI/Purity**（K 扫描） | 表位标签只用于评测，不参与聚类 | 🟢 **已结项（双基础）**：A 官方 9 方法复算(**9/9** 与 Fig 3A 吻合，max \|Δ\| 0.005)‖B embedding→clustering；基础A **Ours-BioSeq** 领先 ESM2/k-mer，基础B **专用对比学习 SCEPTR 最强、Ours 次之**（领先 ESM2/k-mer/ProtBERT/TCR-BERT） |
| **T3 Representation** | 表位特异性 **few-shot per-epitope NN**（deep 6-pMHC 主 + broad 24-epitope 互补 + probe 辅） | **SCEPTR (Cell Systems 2024)**：deep=Table SI III.1 原始协议(era-matched VDJdb 2023-10)；broad=24 表位 split | **few-shot per-epitope AUROC**（deep 100 seeds/k≤200/全库负样本；broad macro）+ kNN top-1 | 克隆型隔离；deep 逐 pMHC 与 Table SI Δ 校验 | 🟢 **deep SCEPTR Table SI 严格复现全跑通**（SCEPTR/TCRdist/CDR3-Lev/ESM2/ProtBERT/TCR-BERT/kmer，k=200 Δ≈0.006–0.024 忠实吻合）；🟢 **我们 3 模型口径最终定为真双链 tcr_pair(LLaDA 末层+无 peptide 配对记录+跨链联合前向, 结构性无泄漏)** 并重跑（`grammar_tcrpair_*`，如实：伪双链→真双链未提升，未超旧 encoder-only）；🟢 broad 六模型对照全跑通 |
| **T4 Generation** | CDR3β 序列生成（**两种 setting：无条件 / 表位条件**） | 无条件: OTS paired-clean（OLGA/SONIA 口径）；表位条件: IMMREP23 VDJdb 分层 common/rare/novel（TcrDesign/TCRT5 口径） | 无条件: **k-mer JSD↓**（辅 novelty/NN/Pgen）；表位条件: **recovery + exact-match**（≈ TCRT5 SeqRec/F1@K） | 无条件: holdout 去重、novelty vs train；表位条件: 表位分层 zero-shot（novel 不进 conditioning）、ref∩conditioning=0 | 🟢 无条件 Markov/PWM + 条件 Markov/CondPool 均已跑通（含 oracle）；🟡 待接官方 OLGA/SONIA/TCRT5/TcrDesign + Pgen 校准 |

> 图例：🟢 已有 baseline 数字 / 🟡 进行中 / ⚪ 待建。`Ours-BioSeq`(3×grammar_v2) 已接入 **T1 / T2 / T3**（真实数字）；仅 **T4** 待接入。
> **实现映射**：T4 无条件 setting = `tcr_generation/run.py`、表位条件 setting = `tcr_design/run.py`（两个脚本入口，但同属 T4 Generation 一类的两种 conditioning 设定）。

> ⚠️ **本次（2026-07-03）广泛文献复核发现的构建短板见 §7「文献校准与已知构建局限」**——不只是命名，涉及 T1 近似去重/负样本来源、T2 规模、T3 few-shot 口径与缺失的序列比对 baseline。

---

## 1. 文献调研（2024–2026，高引 / 前沿）

### T1 Binding — 主依据
| 文献 | 出处 / 年份 | 迁移到我们基准的口径 |
|------|-----------|----------------------|
| **Assessment of computational methods in predicting TCR–epitope binding recognition** 【**T1 主构建基础**】 | **Nature Methods 2025** ([s41592-025-02910-0](https://www.nature.com/articles/s41592-025-02910-0))；官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark` commit `ec832b47` | ① **主指标 = AUPRC**；② **负样本来源**（AS/PS/HS）显著影响分数 → 必须固定并报告（§L2 已复现）；③ **独立测试集** + CD-HIT 去相似 TCR；④ seen / unseen 分开；⑤ 多特征(αβ+MHC)>CDR3β-only 但 unseen 仍近随机。**外部 baseline 迁移**：`Original_model/` 官方权重 + 官方推理代码 + `original.zip` seen/unseen（AS 主协议），不使用 GitHub retraining `train.csv`。`train.csv` 只服务 Ours/探针/kNN 等本地诊断及 L1 sensitivity，不属于 external original-model baseline 协议。 |
| **New Benchmarking…Limited Generalization** 【**T1 补强口径**】 | arXiv 2026 ([2606.04994](https://arxiv.org/html/2606.04994v1))，Liao et al. (UMBC/UPenn/CHOP) | ① **Levenshtein≤3 近似去重**（CDR3β 允许≤3 替换，剔除 test 中 **40–70%** 近重复防记忆泄露）→ 迁移为 §L1（k=1/2/3 敏感性；我们 k≤2 去掉 unseen 67.4%，落其区间）；② **Macro-AUC0.1**（低 FPR pAUC）作辅指标（已加进 metrics）；③ unseen 定义 = 公共库(VDJdb/IEDB)完全未见 |
| **Assessing TCR-epitope data-size requirements**（点突变 pan-allelic ligandome）【**T1 佐证**】 | Sci Rep 2025 ([s41598-025-26454-7](https://www.nature.com/articles/s41598-025-26454-7)) | published models 仅在与训练表位 **edit distance≤1（至多2）** 时优于随机；估计需 1–100M 表位才泛化 → 支持"按到训练集编辑距离分层"框架（§L1 的 k 敏感性即此思路的 TCR 侧对应） |
| **Lessons learned from IMMREP23** | ImmunoInformatics 2024 ([100045](https://doi.org/10.1016/j.immuno.2024.100045)) | unseen 表位「基本未解决」；负样本构造导致 test 泄露 → 我们的 clonotype 去重直接对症 |
| **ePytope-TCR** | Cell Genomics 2025 ([100946](https://doi.org/10.1016/j.xgen.2025.100946)) | 18+3 预测器统一接口范式 → 我们的 `--method`/`--embedder` wrapper 思路一致；低频表位普遍失败 |

**共识与选型**：主榜必须报 **unseen + AUPRC**；负样本来源与（近似）克隆型泄露对分数的影响 > 模型结构。**选定 NM2025 为 T1 主构建基础（不替换）+ arXiv2606.04994 的 Lev≤k 近似去重/AUC0.1 补强 + Sci Rep 2025 佐证编辑距离分层**——理由：NM2025 是唯一同时给官方 train/test + AS/PS/HS + 50 模型官方权重的可复现基座；2026 新数据(TetTCR-SeqHD/Fingerprinting)是 perspective/新实验、未打包成带官方 baseline 的可复现 split，故作方法论补强而非新主基座。

### T2 Clustering — 主依据（**双基础**）
| 文献 | 出处 / 年份 | 口径 | 角色 |
|------|-----------|------|------|
| **Benchmarking unsupervised methods for inferring TCR specificity** | NAR Genomics & Bioinformatics 2025 ([lqaf150](https://doi.org/10.1093/nargab/lqaf150) / [biorxiv](https://www.biorxiv.org/content/10.1101/2024.10.26.620398)) | 9 方法统一 190k TCR / 2313 表位；指标 **Purity / Retention / Sensitivity**；DeepTCR 保留率最高，clusTCR/TCRMatch/GLIPH2 纯度最高 | **基础A（主）**：官方方法级 baseline（预计算输出） |
| **A comprehensive benchmarking for evaluating TCR embeddings in modeling TCR-epitope interactions** | Brief Bioinformatics 2025 ([bbaf030](https://doi.org/10.1093/bib/bbaf030), PMID 39883514；[deepomicslab/TCREmbedding](https://github.com/deepomicslab/TCREmbedding) commit `d5bf911`) | **embedding→clustering**：固定标注集（GIANA curated，**9,033 CDR3β / 25 表位**）→ K∈{10,…,100} 扫描 → **ARI / NMI / Purity**；评 19 个嵌入方法，结论"专用/特异性嵌入 > 数据驱动，通用 ESM 在 CDR3β 偏弱" | **基础B（第二）**：Ours-BioSeq 最对口的嵌入→聚类同口径 + 第二数据集 + ARI/NMI |
| **A comparison of clustering models for TCR antigen specificity** | ImmunoInformatics 2024 ([S266711902400003X](https://www.sciencedirect.com/science/article/pii/S266711902400003X)；[hudsondan/tcr-scapes](https://github.com/hudsondan/tcr-scapes)) | clusTCR/GIANA/GLIPH2/iSMART/tcrdist3 + Hamming/V-gene/random 基线，F1 加权 + purity/consistency/retention | 文献参照（与基础A方法重叠，不新建榜） |
| TouCAN (bioRxiv 2024, 对比学习 pLM+ESM 聚类) / G2VTCR (bioRxiv 2025, 原子级图嵌入聚类) | bioRxiv 2024/2025 | 新聚类**方法/模型**（非统一基准），各需独立 env/权重 | 可选扩展（"嵌入参照候选"，本轮不接入） |

**方法可用性**：clusTCR / GIANA / iSMART / tcrdist3 / TCRMatch / DeepTCR 有开源；**GLIPH2 仅 web server → 不收录**。

### T3 Representation — 主依据
| 文献 | 出处 / 年份 | 口径 |
|------|-----------|------|
| **Contrastive learning of TCR representations (SCEPTR)** ★主锚 | **Cell Systems 2024** ([S2405-4712(24)00369-7](https://www.cell.com/cell-systems/fulltext/S2405-4712(24)00369-7)) | few-shot **per-epitope NN AUROC**（Methods III.1 / Table SI：6 target pMHC、全 VDJdb 背景负样本、k≤200、per-pMHC macro）；对比 CDR3-Lev / TCRdist / ProtBert / ESM2 / TCR-BERT，证明通用/专用 PLM < 序列比对法，专用对比学习才反超；源码 [github.com/yutanagano/sceptr](https://github.com/yutanagano/sceptr)。**T3-deep 逐 pMHC 与其 Table SI 校验** |
| Feng et al. — Comprehensive benchmarking of TCR embeddings（辅助/未来扩展） | Briefings in Bioinformatics 2025 ([bbaf030](https://doi.org/10.1093/bib/bbaf030)) | 19 种 CDR3 嵌入 + `deepomicslab/TCREmbedding`；结论"**手工嵌入(BLOSUM/k-mer/理化)总体优于数据驱动 PLM 嵌入**"，呼应本 track"kmer 追平/略胜多数 PLM"；可作未来把 T3 baseline 面扩到 19 种嵌入的方向 |
| Drost et al. — ePytope-TCR（泛化性 caveat 参照） | Cell Genomics 2025 ([xgen.2025.100946](https://doi.org/10.1016/j.xgen.2025.100946)) | 统一评测 18+3 TCR-epitope 预测器，强调未见表位近随机；偏 binding（贴 T1），此处作泛化 caveat，非 T3 主协议 |

### T4 Generation — 主依据（2024–2026 调研；采纳标准 = 官方代码+权重可复现）
> 完整选型论证见 `TCR_GENERATION_BENCHMARK.md §0`。**主构建基础 = TCRT5**（唯一 turnkey 可复现基准：官方 held-out 划分 + 竞品官方预存预测 + 官方评测代码，我们已 42 block 0 mismatch 对齐）；**补 TcrDesign(全长 SOTA) + TCRDiff/TCR-epiDiff(与 BioSeq 同族扩散 baseline)**；LSMTCR/TCRGen 无公开权重仅引用。

| 文献 | 出处 / 年份 | 口径 | 官方代码/权重 | 采纳 |
|------|-----------|------|---------------|------|
| **TCRT5** (TCR-TRANSLATE) | **Nat Mach Intell 2025** ([s42256-025-01096-6](https://www.nature.com/articles/s42256-025-01096-6)) | 条件生成 held-out 表位；exact matches / F1@K / seq-recovery / Char-BLEU | `pirl-unc/tcr_translate` + HF `dkarthikeyan1/tcrt5_ft_tcrdb` | ✅ **主锚** |
| **TcrDesign** | bioRxiv **2026**.01 ([699824](https://www.biorxiv.org/content/10.64898/2026.01.15.699824v1.full-text)) | **全长** epitope-specific（CDR3→VJ 拼全长）；common/rare/novel 分层 | `XSLiuLab/TcrDesign` + Zenodo 14545852 | ✅ Setting C 全长 SOTA |
| **TCRDiff** | bioRxiv **2026**.06 ([730756](https://sciety.org/articles/activity/10.64898/2026.06.10.730756)) | **条件离散扩散(DPLM)** pMHC+V基因→CDR3αβ；MAGE-A3 体外验证 | `zhangyumeng1sjtu/TCRDiff` + Zenodo 20586708 | ✅ **新增 2026 扩散 baseline** |
| **TCR-epiDiff** | Bioinformatics **2025** ([btaf202](https://doi.org/10.1093/bioinformatics/btaf202)) | **DDPM(U-Net+ProtT5-XL 条件)** epitope→CDR3β + binding 双任务 | `seoseyeon/TCR-epiDiff` + 仓库内 `Model/*.pth` | ⛔ **仅引用**：官方生成代码非 turnkey（broken import + 未发布 pkl 输入 + denoise-from-real-TCR + 4通道编码歧义），不 reimplement（详见 `TCR_GENERATION_BENCHMARK.md §0.3/§7`） |
| **LSMTCR** | arXiv 2025 ([2509.07627](https://doi.org/10.48550/arxiv.2509.07627)) | **全长 α/β** epitope-conditioned；diffusion-BERT + 条件 GPT + gene-aware Transformer | ❌ 无公开仓库(截至 2026-07) | ⛔ 仅引用（不可复现） |
| **TCRGen** | bioRxiv 2025 (Lee-CBG) | in-context learning 生成新表位 TCR | `Lee-CBG/TCRGen`（权重 "soon"，未发布） | ⛔ 仅引用（权重未发布） |
| **TCRTransBench** | arXiv 2026 ([2605.04762](https://arxiv.org/html/2605.04762)) | **双向** TCR2PEP / PEP2TCR；Binding Specificity + Novelty | 基准工具 | ○ 方法论参照（非生成器） |
| **OLGA / SONIA / soNNia** | Bioinformatics 2019 / PLoS CB 2020 / PNAS 2021 | repertoire Pgen / Ppost；KL + Pearson 分布匹配 | [statbiophys](https://github.com/statbiophys) | ✅ Setting A 参照 |

### 综述 / 总体依据
| 文献 | 出处 / 年份 | 对四类划分的支撑 |
|------|-----------|------------------|
| **TCR representation learning with protein language models: a comprehensive review** | Int Immunol 2025 ([dxaf048](https://doi.org/10.1093/intimm/dxaf048)) | 系统综述确认 TCR 下游围绕 **表征 / 特异性(binding) 分类 / 聚类(metaclonotype) / 生成设计** 组织；生成与设计同属 decoder 的 de novo generation，不单列 |

---

## 2. 逐任务详细设计

### T1 — TCR-Epitope Binding
- **任务**：给定 (TCRβ, TCRα, peptide[, MHC])，二分类是否结合。
- **数据（主）**：IMMREP23 paired-chain VDJdb 正样本 + 参考负样本；测试用官方 `solutions.csv`。已落地 `data/tcr_binding/{train,test}.csv`（train=61008 / test=3484；seen 13 表位/2418 行、unseen 7 表位/1066 行）。
- **迁移来源**：`baselines/IMMREP23`（justin-barton/IMMREP23, commit 06d85be）。
- **主指标（改）**：unseen **macro-AUPRC**（对齐 Nat Methods 2025）；辅报 macro-AUROC / macro-AUC0.1、seen 三指标。
- **防泄露**：删除与测试克隆型(CDR3a|CDR3b)重叠的 60 条训练正样本 → clonotype/pair overlap=0（`leakage_report.json`）。**负样本**：参考 TCR、Levenshtein>3、5:1，来源固定并记录。
- **源码-only baseline**：
  - 训练free：CDR3-kNN(TCRdist 式)、Random（已跑）。
  - 官方权重：**NetTCR-2.2**、**ERGO-II**、**epiTCR**、**PanPep**（已 clone，待各自 env 出数）；候选 **ATM-TCR / TEIM**（Nat Methods 榜首附近，有源码）。
  - 通用 PLM probe：ESM2-150M（已跑）；SCEPTR few-shot（见 T3）。
- **接入点**：`tcr_binding/run.py --method embed --embedder bioseq:/abs/final.pt --columns cdr3b cdr3a peptide`。
- **状态**：🟢 baseline(kNN/ESM2/Random) 已跑通并有 `outputs/tcr_binding/*`；🟡 待办 = 主指标切 AUPRC + 官方 baseline wrapper 出数。

### T2 — TCR Clustering（**双基础**：NAR-GAB 2025 主 + Brief Bioinformatics 2025 第二基础）
- **任务**：无监督聚类 TCR（CDR3β[+α]），与 held-out 表位标签比对。经更充分的 2024–2026 文献调研，采用两个互补基础，**分区呈现、互不混表**。
- **基础A（主）数据**：**直接复用 NAR-GAB 2025 开源仓库 [`i3-unit/TCR_Unsupervised_Benchmark`](https://github.com/i3-unit/TCR_Unsupervised_Benchmark)（commit `ac767882`）的 curated pooled DB**（IEDB+McPAS+VDJdb），照其 `database_processing.R` 复算；**现行 universe 是 4,779 配对 / 8,395 序列**（旧文 5,368/9,370 是 `curate()` 对 `PubMed_ID` NA 的 pandas/dplyr 语义差，已修）。论文口径四个常量现已逐一命中。**评测单元 = 唯一配对 TCR（`pair_id`）**；chain-separate 方法用 β 链簇映射到配对。
- **基础A 主指标**：**Purity–Retention 权衡曲线**；辅 Sensitivity / %clusters purity>0.9 / %seqs in high-purity clusters / NMI / ARI，并做 cluster-size>3/5/10 复算（`common/metrics.py::clustering_metrics`，NAR-GAB 口径，向后兼容）。
- **基础A 源码-only baseline（只用官方源码/结果）**：**9 个官方方法直接复用 NAR-GAB 2025 作者仓库的预计算聚类输出** = LD/HD/TCRMatch/TCRdist3/GIANA/iSMART/clusTCR/GLIPH2/DeepTCR（`Data/Output_methods/`；HD/LD 为论文脚本用 stringdist 现算的汉明/编辑距离连通分量，我们同法现算）。**我们只解析官方输出、按论文口径复算指标，绝不重写方法、不 clone 各工具重跑**。指标口径交叉验证：**6/9 逐一复现论文 Purity/Retention**（clusTCR 0.09/0.99、DeepTCR 0.92/0.63、TCRMatch 0.09/0.82、GLIPH2 0.22/0.80、TCRdist3 0.44/0.42、GIANA 0.25/0.66；HD/LD、iSMART 因 DB 漂移/仓库输出规模差异略偏，已标注）→ `outputs/tcr_clustering/calibration_report.csv`。**编辑距离方法族以官方 LD（Levenshtein，来自仓库）为代表**；早前自写的 `editdist(t=1)` 与官方 LD 重复，按"只用官方源码/结果"原则已移除。
- **基础B（第二）数据/指标**：取 Brief Bioinformatics 2025《A comprehensive benchmarking for evaluating TCR embeddings...》（[deepomicslab/TCREmbedding](https://github.com/deepomicslab/TCREmbedding) commit `d5bf911`）的**固定评测数据集** `dataset/clustering/TCRantigenData_unique_test.csv`（GIANA curated，**9,033 唯一 CDR3β / 25 表位**，`data/tcr_clustering_embed/`），跑**embedding→clustering** 口径：嵌入 CDR3β → L2 归一 → K-means(主)+层次(PCA-50 Ward,辅) 扫 K∈{10,…,100} → **ARI / NMI / Purity**（mean/best over K）。补齐基础A缺的 ARI/NMI + 第二独立数据集。**只引论文定性结论，不擅自转录其未复核的逐方法数值**（在 Supplementary Table 2）。脚本 `scripts/prepare_tcrembedding_clustering.py` / `tcr_clustering/run_embed_bench.py` / `scripts/summarize_tcrembedding_clustering.py`。
- **防泄露**（两基础一致）：表位标签只用于评测，不进聚类。
- **补充 baseline**：TCR-VALID（Nat Commun 2024, `peterghawkins-regn/tcrvalid` commit `475ed964`，官方权重）原生 DBSCAN 协议 + 我们 embed-threshold 口径（基础A）。
- **接入点 / 我们的模型（与官方方法分区）**：无官方聚类输出的模型/参照嵌入走**我们统一评测框架**——基础A `tcr_clustering/run.py --method embed-threshold`（Purity–Retention），基础B `tcr_clustering/run_embed_bench.py`（ARI/NMI/Purity）。参照 = ESM2-150M / k-mer / TCR-VALID latent；**Ours-BioSeq** = 3 个 `grammar_v2_*_llada` ESMC-300M/600M checkpoint（`EsmcBioSeqEmbedder` 0 missing/0 unexpected 载入）。**GPU 嵌入走火山引擎作业**（`eval_jobs/eval_t2_clustering_embed_bioseq.yml`，task `t-20260705211729-crvhj` Success），本地只跑轻量参照 + 解析/汇总。这些**归属"我们的模型/参照嵌入"，不与官方 9 方法混栏**。
- **状态**：🟢 **已结项（双基础）**——基础A：官方 9 方法复算并与论文交叉验证（6/9），我们框架下 Purity–Retention 曲线全产出（Ours-BioSeq 同 retention 纯度显著优于 ESM2/k-mer，如 ret≈0.25 esmc600m 0.917 vs ESM2 0.860，略逊专用 TCR-VALID）；基础B：9,033/25 数据集上 ARI/NMI/Purity 独立复核，**Ours-BioSeq post-LLaDA cmp500k 领先所有通用/自监督 PLM（ESM2/k-mer/ProtBERT/TCR-BERT，ARI 0.024 vs 0.007/0.010）**，但**低于专用对比学习 SCEPTR（ARI 0.033，2026-07-08 新增参照）**——复现论文"通用/自监督表征 < 专用对比学习"趋势，与 T3 次序吻合。详见 `RESULTS.md` T2 (a)–(d) 与 `downstream.md`。

### T3 — TCR Representation
- **任务**：冻结主干 → 表位特异性判别。**锚定论文 = SCEPTR (Cell Systems 2024)**。两条互补 track，均复现"通用/专用 PLM 打不过序列比对法、只有对比学习 SCEPTR 反超"：
  - **T3-deep（主口径，严格照 SCEPTR Methods III.1 / Table SI）**：`tcr_representation/run_paper6.py` → `outputs/tcr_representation_paper6/`。6 target pMHC、**背景全 VDJdb 负样本**、k∈{1,2,5,10,20,50,100,200}、**100 seeds**、per-pMHC AUROC macro，并逐 pMHC 与论文 Table SI 原始数字做 Δ 校验。
  - **T3-broad（互补大表位面视图）**：`tcr_representation/run.py` → `outputs/tcr_representation/`。24 表位 clonotype 隔离 split（train=6887/test=1722，负样本仅其余表位），k∈{1,2,5,10,20,50,100}；辅协议 = 24-way linear probe + 1-NN（`metrics.json`）。
- **数据**：deep = era-matched **VDJdb 2023-10**（`data/tcr_representation_paper6/`，universe 25,816 = 3,941 binder + 21,875 背景，6 pMHC 均 >200 binder）；broad = `data/tcr_representation/`（与 T2 共享 split）。
- **few-shot 引擎**：`common/fewshot.py`——三后端统一成 query×ref 距离矩阵：native cdist（SCEPTR L2/TCRdist3）、embedding cosine（ESM2/ProtBERT/TCR-BERT/kmer/Ophiuchus/BioSeq）、editdist（CDR3β+α Levenshtein）。min-NN 取负打分、seed 均值 → pMHC macro（mean±std）。**oracle 自检**（支持集含 query 正样本→AUROC≈1.0）：mean=1.0000 / min=0.9997。列口径：嵌入类 `cdr3b cdr3a`，**TCR-BERT 官方用法仅 CDR3β**（`--columns cdr3b`）。
- **关键 baseline（文献强制，只用官方源码/权重）已全部落地（Table SI 六模型对照 + kmer + 3×Ours-BioSeq）**：SCEPTR / TCRdist(tcrdist3) / CDR3-Lev / ESM2 / ProtBERT(`Rostlab/prot_bert`) / TCR-BERT(`wukevin/tcr-bert-mlm-only`)。
- **deep 真实数字（`outputs/tcr_representation_paper6/*/fewshot.json`，k=200 macro）**：**SCEPTR 0.790 ≳ TCRdist 0.777 > CDR3-Lev 0.733 ≈ kmer 0.728 ≈ Ours-BioSeq esmc600m/300m-cmp500k 0.727/0.724 > ESM2 0.696 ≈ ProtBERT 0.694 ≈ TCR-BERT 0.692 > Ours-BioSeq mint 0.686**。**口径校验（vs SCEPTR Table SI k=200 逐 pMHC Δ）**：Lev/TCRdist/SCEPTR/ProtBERT/ESM2 平均 |Δ|≈0.006–0.024、单点 |Δ|≤0.052（忠实复现）；TCR-BERT 逐点偏差偏大（平均 |Δ|≈0.057，源自 MLM-only 编码器 mean-pool 与论文取法差异），但 macro 落 PLM 带内、排序不变。
- **broad 真实数字（k=100 macro）**：SCEPTR 0.741 > TCRdist 0.728 > Lev 0.683 ≥ ESM2 0.671 > kmer 0.673 ≥ ProtBERT 0.651 ≥ TCR-BERT 0.641；排序与 deep 一致（更大表位面稳健性佐证）。
- **接入点**：`tcr_representation/run_paper6.py`（deep）/ `run.py`（broad）`--method {levenshtein,tcrdist,sceptr,embed} [--embedder …]`；我们的模型 `--method embed --embedder bioseq:/abs/best.pt --columns cdr3b cdr3a`。集群一键脚本 `scripts/run_paper6_all.sh`（+ `eval_jobs/eval_paper6_tcr_representation.yml`）。
- **我们模型接入（最终口径 = 真双链 tcr_pair，deep）**：主表用 **Ours** = LLaDA 扩散解码器末层 + β+α 渲染成**无 peptide 的配对记录 `tcr_pair`**（`grammar.py` 对无 peptide 双链落 tcr_pair）+ 干净无噪声**跨链联合**前向（`GrammarEmbedder(feature_source=decoder).embed_pairs(peptide=None)`），`position_ids_chain` 分段 {0:alpha,1:beta} mean-pool 拼 `[α‖β]`。**无表位泄漏是结构性的**（根本无 peptide 输入）。**deep k=200：cmp500k(300m/600m) 0.711/0.718，仍超三个官方 PLM ~0.02、逼近 kmer/Lev**，仍低于 TCRdist/SCEPTR 约 0.06–0.08；mint 0.661 最弱。**如实：从伪双链(encoder-only) → 真双链(tcr_pair) few-shot 未提升，比 encoder-only cmp500k 略降 ~0.01、mint 持平，与占位 peptide 版差 ≤0.005**。**诚实口径**：paper6 仅 CDR3 环 → 喂 CDR3β+CDR3α、tcr_pair、无 peptide，最接近但不完全等同任一训练渲染。对照行保留：`grammar_dec_*`（占位 peptide 中间态）、`bioseq_grammar_v2_*`（encoder-only 伪双链）。
- **状态**：🟢 **deep（SCEPTR Table SI 严格复现）全跑通**——baseline SCEPTR/TCRdist/Lev/ESM2/ProtBERT/TCR-BERT/kmer（Volc `t-20260705204639-l56dp`）+ **3×Ours 最终真双链 tcr_pair 口径重跑**（Volc `t-20260705221018-trjnv`, ExitCode 0，`grammar_tcrpair_*`）+ 中间态 `grammar_dec_*`（`t-20260705214312-k8ktb`）+ 旧 encoder-only 对照，汇总 `_summary.json`；🟢 broad 六模型对照 + Ophiuchus + 3×BioSeq 全跑通；⚪ catELMo 未接（GitHub clone 超时 + allennlp↔torch2.8 依赖冲突，记 blocker，非阻塞）。

### T4 — TCR Generation（生成；**以 TCRT5 为锚 · 三 setting**）
> **2026-07-05 重构**：主口径改为 **TCRT5 (Nat Mach Intell 2025)**，一个方向下 **三种 conditioning setting**（无条件 / 表位条件 / 全长）。完整口径见 **`TCR_GENERATION_BENCHMARK.md`**；统一 harness = `tcr_generation_bench/run.py`，统一指标 = `common/metrics.py` 的 `tcrt5_*` 套件（与官方 `src/evaluation.py` 逐函数一致，42 block 交叉验证 0 mismatch）。
> 旧口径（`tcr_generation/run.py` 无条件 + `tcr_design/run.py` common/rare/novel）降为辅视图，保留不动。

**Setting A — 无条件（repertoire 分布匹配）**
- **数据**：OTS paired-clean CDR3β，`data/tcr_generation/{train,holdout}_cdr3b.txt`（200k train + 9837 holdout）。
- **指标**：k-mer(3) **JSD↓** + novelty + mean-NN-editdist + **OLGA pgen 校准**（生成序列可重组比例 + log10 pgen）。
- **baseline**：OLGA(human_T_beta) / soNNia / TCRT5-uncond；`file` 接 BioSeq uncond 采样。
- **接入**：`tcr_generation_bench/run.py --setting A --method {olga,sonnia,tcrt5,file}`。

**Setting B — 表位条件 CDR3β 设计 ★主榜**
- **数据（对齐 TCRT5）**：held-out **20 pMHC**（`topk_val_df.csv`，target-rich）+ **14 sparse benchmark pMHC**（`benchmark_data_w_preds.csv`，含官方 TCRT5/GRATCR/ER 预测）。`data/tcr_generation_bench/eval_conditional.json`。
- **指标（TCRT5 套件）**：Precision@K / Recall@K / F1@K / native seq-recovery / Char-BLEU / mean edit dist / diversity。harness 自动聚合 `overall/held20/benchmark14/bioseq_seen/bioseq_unseen` 5 视图。
- **防泄露（核心，`data/tcr_generation_bench/leakage_report.json`）**：held20 有 18/20 表位、~48% ref binder 泄露进 BioSeq 训练语料；benchmark14 有 7/14 表位泄露（其余干净）。**BioSeq 主榜用 benchmark14/bioseq_unseen 干净集；held20 对 BioSeq 仅作 seen 对照并显式标注偏乐观**。TCRT5/GRATCR/ER/TcrDesign 非我们训练，held20/benchmark14 对它们照常报（held20 = TCRT5 论文官方 held-out）。ref binder 与 TCRT5 conditioning pool 重叠 = 0。
- **baseline**：TCRT5(HF beam / 官方预存) / TcrDesign-G(beta) / OLGA·soNNia(uncond 下限)；`file` 接 BioSeq。
- **接入**：`tcr_generation_bench/run.py --setting B --method {tcrt5,olga,sonnia,tcrdesign,file} --eval-set {held20,benchmark14,all}`。

**Setting C — 全长 α/β 组装**
- **数据**：同 Setting B 表位集。
- **指标**：全长 validity（合法 AA 比例 + CDR3β 可提取率，ANARCI/CASS…F）→ 提取 CDR3β 后套 Setting B 指标 + pgen。
- **baseline**：**TcrDesign 全长 pipeline**（`tcrdesign.py`）作 SOTA；`file` 接 BioSeq 无条件全长（BioSeq 只在 ots 全长源上训练、无表位，故只做无条件全长 + 提取，不做 epitope 条件全长）。
- **接入**：`tcr_generation_bench/run.py --setting C --method {tcrdesign,file}`。

> **旧辅视图保留**：`tcr_design/run.py`（common/rare/novel + Markov/CondPool，`outputs/tcr_design/`）与 `tcr_generation/run.py`（Markov/PWM，`outputs/tcr_generation/`）不删除，作历史对照。TCRTransBench 双向 / LSMTCR 全长仍为可选扩展子协议。

---

## 3. 源码-only baseline 总表（官方 repo + commit）

> 已 clone 见 `baselines/README.md`；下表含**待 clone**项。仓库本体不纳入 git（`.gitignore`），只留清单 + wrapper。

| 任务 | baseline | 官方 repo | 状态 |
|------|----------|-----------|------|
| T1 | NetTCR-2.2 / ERGO-II / epiTCR / PanPep | mnielLab / IdoSpringer / ddiem-ri-4D / bm2-lab | ✅ 已 clone（epiTCR 已出数，其余待建 env） |
| T1 | ATM-TCR / TEIM | 官方 GitHub | ⚪ 候选待 clone |
| T2（基础A） | **9 方法 = LD/HD/TCRMatch/TCRdist3/GIANA/iSMART/clusTCR/GLIPH2/DeepTCR**（复用官方预计算输出）+ **TCR-VALID** | **i3-unit/TCR_Unsupervised_Benchmark**（commit `ac767882`，含 9 方法 output_methods）；peterghawkins-regn/tcrvalid（`475ed964`） | ✅ 已导入并 6/9 复现论文；TCR-VALID 独立 TF env 已跑 |
| T2（基础B） | 固定评测数据集（9,033 CDR3β/25 表位）；参照嵌入 ESM2-150M（官方 HF 权重）/ k-mer(训练free) | **deepomicslab/TCREmbedding**（commit `d5bf911`，Brief Bioinformatics 2025 bbaf030） | ✅ 数据已下载；embedding→clustering ARI/NMI/Purity 已跑（Ours-BioSeq ×3 走 Volc 作业，ESM2/k-mer 本地） |
| T3 | **TCRdist / CDR3-Levenshtein-NN（主对照）** / **SCEPTR** / **ProtBERT** / **TCR-BERT** / catELMo(可选) | kmayerb/tcrdist3、yutanagano/sceptr(8c1fa02ec4)、Rostlab/prot_bert、wukevin/tcr-bert-mlm-only、Lee-CBG/catELMo | ✅ **DONE：TCRdist(tcrdist3 0.3) / Lev-NN / SCEPTR(1.2.0) / ProtBERT / TCR-BERT 全跑通**（主 env，few-shot 主协议，见 `common/fewshot.py` + `common/model_api.py::{ProtBertEmbedder,TcrBertEmbedder}`；ProtBERT/TCR-BERT 官方 HF 权重 mean-pool 薄封装，复现 SCEPTR Table SI 六模型对照）；⚪ catELMo 未接（GitHub clone 超时 + allennlp 2.10↔torch 2.8 依赖冲突，blocker，非阻塞） |
| T4 无条件(A) | OLGA / SONIA / soNNia | statbiophys/OLGA、statbiophys/SONIA | ✅ 已 clone + 跑通（官方 `human_T_beta`；Setting A JSD 0.043–0.061） |
| T4 表位条件(B) | TCRT5(主锚) / GRATCR / ER-Transformer / **TCRDiff(2026 扩散)** / ~~TCR-epiDiff(2025 扩散)~~仅引用 | `dkarthikeyan1`(TCRT5,HF+官方预存) / benchmark 文件(GRATCR·ER) / `zhangyumeng1sjtu/TCRDiff`(e5d2ecba) / `seoseyeon/TCR-epiDiff`(d7b726ff) | ✅ TCRT5/GRATCR/ER/**TCRDiff(benchmark14)** 跑通；⛔ TCR-epiDiff 官方生成代码非 turnkey → 仅引用（见 `TCR_GENERATION_BENCHMARK.md §0.3/§7`） |
| T4 全长(C) | **TcrDesign(全长 SOTA)** / Ours-BioSeq(无条件全长) | `XSLiuLab/TcrDesign` + Zenodo 14545852 | ✅ BioSeq 无条件全长跑通(Volc)；🔄 TcrDesign 权重下载中 |
| 参考 | IMMREP22 官方 17 方法 | viragbioinfo/IMMREP_2022_TCRSpecificity | ✅ 已导入 `outputs/external/` |

**环境策略**：各 baseline 框架差异大（TF1.x / Keras / PyTorch-Lightning / sklearn），按官方 requirements 建独立 conda env，wrapper 只做「读统一输入 → 调官方推理 → 写统一预测文件」。

---

## 4. 数据与目录（现状）

```
benchmark/
├── common/            # schema / leakage / negatives / metrics / model_api / featurizers（已实现）
├── data/{tcr_binding,tcr_clustering,tcr_representation,tcr_generation,ppi}/  # 已落地
├── scripts/prepare_*.py   # 数据准备（已实现）
├── baselines/         # 官方源码(不纳入 git) + wrappers/
├── outputs/           # 预测 + metrics.json（可复现）
├── {tcr_binding,tcr_clustering,tcr_representation,tcr_generation,ppi}/run.py
├── README.md / RESULTS.md / PROGRESS.md   # 总览 / 排行榜 / 过程
├── TCR_TASK_TAXONOMY.md                    # 早期精简方案（本文件为其升级版）
└── TCR_BENCHMARK_DESIGN.md ← 本文件（主设计 + 实时状态）
```

## 5. 我们模型接入点（Ours-BioSeq · post-LLaDA headline 口径）

### 5.1 post-LLaDA 特征定义（2026-07-05 方法学修正，用户指示）

**一句话**：对每个输入，渲染为 **clean、全可见、无扩散噪声** 的 grammar 记录，经 **ESMC 编码 → condition 替换 LLaDA token embedding → LLaDA backbone 双向前向**，读出 **`decoder(..., output_hidden_states=True).hidden_states[-1]`**（post-`ln_f` 末层），再按任务池化。

| 维度 | 定义 |
|------|------|
| **checkpoint** | `output/grammar_v2_*_llada/best.pt` = `BioSeqLLaDAEncoderDiffusionModel`（`modeling_bioseq.py`）：`encoder.esmc.*` + `decoder.*`（`LLaDAModelLM`） |
| **载入** | `load_grammar_checkpoint` strict load，**0 missing / 0 unexpected**（含 ESMC+LLaDA+condition） |
| **读out 层** | `decoder.hidden_states[-1]`（post-`ln_f`，与 tied 读出头对接） |
| **单链任务**（T2 聚类 CDR3β） | `task_type="tcr"`, role `tcr_beta`，**无 peptide**；残基位 **mean-pool** → `[dim]` |
| **成对任务**（T1 binding β+peptide；T3 broad β+α） | `embed_pairs(beta, alpha?, peptide?)`：多链联合前向，分段 mean-pool 后 concat；T3 broad 用**常量占位 peptide**（`peptide=None`→`""`）防表位泄漏 |
| **T3 deep** | 真双链 `tcr_pair`（β+α 无 peptide，跨链注意力）——已是 post-LLaDA，见 §2 T3 |

**embedder spec（等价）**：
- `bioseq-llada:/abs/best.pt` → `EsmcLladaBioSeqEmbedder`（推荐，语义明确）
- `grammar:decoder:/abs/best.pt` → `GrammarEmbedder(feature_source="decoder")`
- ~~`bioseq:/abs/best.pt`~~ → `EsmcBioSeqEmbedder`（**encoder-only 对照**，跳过 LLaDA decoder）

实现：`common/model_api.py`（`EsmcLladaBioSeqEmbedder` + `GrammarEmbedder._final_hidden`）；冒烟：`scripts/smoke_bioseq_llada.py`。

### 5.2 任务接入速查

| 任务 | 推荐 spec | 池化/拼接 | tag 约定 |
|------|-----------|-----------|----------|
| T2 基础A/B（单链 CDR3β） | `bioseq-llada:` | mean-pool | `*_postllada` |
| T1 binding（β+peptide） | `grammar:decoder:` 或 `bioseq-llada:` + `embed_pairs` | `[emb(β)‖emb(pep)]` | `ours_postllada_*_mlp` |
| T3 broad（β+α） | `bioseq-llada:` + `grammar_pair_embed` | `[emb(β)‖emb(α)]` | `bioseq_llada_*` |
| T3 deep | `grammar:decoder:` + `tcr_pair` | `[α‖β]` 分段 pool | `grammar_tcrpair_*` |
| T4 generation | checkpoint 直接扩散解码 | 生成原生（天然 post-LLaDA） | — |
| NbBench | `grammar:decoder:` / `bioseq-llada:` | 标量 mean-pool / 残基级 | `ours_*_llada` |

- 已训 checkpoint：`output/grammar_v2_*`（esm2_650m / esmc600m / esmc300m / no_encoder_1b，均有 `best.pt`）。
- **生成任务**另需模型侧导出采样序列文件，再 `--method file` 接入。

## 6. 非 TCR 维度（独立，不并入四类）
- **PPI**：完全对齐 **MINT (protein)** 测试口径 → 见 `../MINT_MIGRATION.md`（迁移 GeneralPPI：Bernett/HumanPPI/YeastPPI/SKEMPI/PDBBind/mutational）。当前自研 STRING 90/90（`ppi/run.py`）已跑通作占位。
- **抗体**：对齐 **Ophiuchus-Ab** → 见 `../README.md`（comp_chain / infill / humanization / flab / dev / specificity），及 NbBench A1（`nbbench/`）。

---

## 7. 文献校准与已知构建局限（2026-07-03 广泛复核）

> 结论先行：**四大方向（binding/clustering/representation/generation）划分正确、被最新文献一致支撑**；generation 的"条件/无条件"是**同一方向的两种 setting**（已在 §0/§2 修正，不再当第五类）。但本轮复核发现**若干实质构建短板**（非命名问题），按优先级列出，供后续 fix：

| # | 方向 | 现状 | 文献要求 | 差距 / 待修 | 优先级 |
|---|------|------|----------|-------------|--------|
| L1 | T1 Binding | ✅ **已修复（2026-07-05）** | Nat Methods 2025 / arXiv2606.04994 用 **CD-HIT 或 Lev≤3 近似去重**，可剔除 test 中 40–70% 近似序列 | `scripts/nm2025_near_dedup.py`：train↔test CDR3β Levenshtein≤k（rapidfuzz，k=1/2/3 敏感性），在同一硬子集对**所有方法**复算 overall AUPRC。**去掉 unseen 19.0%/67.4%/92.6%（k=1/2/3）**（k≤2 落 arXiv 40–70% 区间）；去重后 unseen 仍全体≈0.5（泛化失败是真结论），但 **kNN 表观优势 k≤1 即坍缩 0.544→0.487**（距离法吃近重复红利被证实）。产物 `data/tcr_binding_nm2025/near_dedup_report.json` + `outputs/.../near_dedup_summary.csv`，原口径保留 | ✅ **DONE** |
| L2 | T1 Binding | ✅ **已修复（2026-07-05）** | Nat Methods 2025：**负样本来源(AS/PS/HS)对分数影响 > 模型结构**，须固定且报告；HS/PS 引入混杂 | `scripts/summarize_nm2025.py`：AS/PS/HS × seen/unseen 主表汇总（`summary_neg_source.csv`）。**复现现象**：同一模型换负样本，unseen AUPRC 摆动 ~0.13–0.15（Ours-cmp500k PS 0.469→HS 0.623），大于固定 AS 时模型间差异（0.479–0.544）；HS 系统性抬分（来源混杂）。**主榜坚持 AS**（作者 refined cross-matching，假阴性最少） | ✅ **DONE** |
| L3 | T2 Clustering | ~~仅 24 表位 / 8609 条~~ → **已结项 + 双基础加固（2026-07-05）** | NAR-GAB 2025 统一库；**新增** Brief Bioinformatics 2025 / TCREmbedding 第二基础 | ✅ **基础A**：NAR-GAB 2025 / i3-unit 重建 **4,779** 配对（旧 5,368 为 curate bug）、官方 9 方法 **9/9** 与 Fig 3A 吻合。✅ **基础B（回应"调研不充分"）**：新增 embedding→clustering 口径（9,033 CDR3β/25 表位 + **ARI/NMI/Purity** + 第二独立数据集），**Ours-BioSeq(cmp500k) 领先通用/自监督 PLM（ESM2/k-mer/ProtBERT/TCR-BERT），但低于专用对比学习 SCEPTR（2026-07-08 新增参照）**，复现论文"通用/自监督 < 专用对比学习"趋势。GPU 嵌入走 Volc 作业。见 §2 T2、§1 T2 双基础表 | ✅ 结项 |
| L4 | T3 Representation | ✅ **已修复+深化+口径两轮修正（2026-07-05）**：先建 broad few-shot + 六模型对照；补 **T3-deep = SCEPTR Table SI III.1 严格复现**；再修正我们模型口径 encoder-only→decoder；**最终定为真双链 `tcr_pair`**——β+α 渲染成**无 peptide 的配对记录**(grammar_name=tcr_pair, 训练 OTS 分布内)、经 LLaDA 解码器跨链联合前向、分段 {0:alpha,1:beta} pooling，结构性无泄漏。harness(`run_paper6/run/fewshot`)加 grammar-decoder 分支；`_record` 去掉占位 peptide、`_segment_slots` 改无 peptide 映射；3 模型用 `grammar_tcrpair_*` 重跑，旧 `grammar_dec_*`/`bioseq_*` 保留对照 | SCEPTR 核心结论：**通用/专用 LM（含 TCR-BERT）打不过 TCRdist / CDR3-Levenshtein，只有对比学习 SCEPTR 反超**；主口径 = deep 6-pMHC few-shot | **deep 已复现方向（k=200）**：SCEPTR 0.790 ≳ TCRdist 0.777 > Lev 0.733 ≈ kmer 0.728 > **Ours(tcr_pair) cmp500k 0.718/0.711** > ESM2/ProtBERT/TCR-BERT ~0.69 > Ours(tcr_pair) mint 0.661。**Table SI Δ 校验**：5/6 baseline 平均\|Δ\|≈0.006–0.024 忠实吻合。**口径修正如实**：伪双链→真双链未提升（tcr_pair 比 encoder-only cmp500k 略降 ~0.01/mint 持平，与占位 peptide 版差 ≤0.005）；仍为最强 PLM 嵌入档但未超比对法/SCEPTR | ✅ **DONE** |
| L5 | T4 Generation | 无条件 + 条件已跑通；指标 JSD / recovery/exact | 文献视二者为**同一生成任务的两种 conditioning**；条件用 SeqRec%/F1@K，无条件建议加 **OLGA Pgen 校准** | 文档已收敛为一类（本次修正）；仍缺 Pgen 校准 + 双向(TCR↔pep)/全长扩展 | P1 |
| L6 | 全局 | 🟢 **T1/T2/T3 完成**：`Ours-BioSeq`(3×grammar_v2) 已接入 **T1**/T2/T3-broad/T3-deep；**T1 官方 baseline 6 个全补齐**（ATM-TCR/NetTCR-2.2/TEIM/ERGO-II/epiTCR/PanPep，官方权重）；T3 deep+broad 六模型对照+kmer | 榜需真实对照 | 仅剩 **T4** 待接 `grammar_v2_*` + 官方 OLGA/SONIA/TCRT5/TcrDesign | P1 |

> **一句话**：命名已理顺（四类 + generation 两 setting）；**P0 全部修复**——L4（T3）2026-07-05 修复，**L1（T1 近似去重）+ L2（T1 AS/PS/HS 负样本来源）本轮 2026-07-05 落地**（真实数字，见 §2 T1 / `downstream.md` / `RESULTS.md`），T1 补齐 6 个官方 baseline + 3×Ours-BioSeq。剩余 P1 = L5（T4 Pgen/扩展）、L6（T4 接入）。

---

## 变更记录（changelog）
- **2026-07-02**：创建本主文档。完成 T1–T4 文献调研（Nat Methods 2025 binding、SCEPTR、NAR-GAB clustering、TCRT5/OLGA generation），确立主指标口径（T1 改用 unseen-AUPRC）、源码-only baseline 清单（含待 clone 项）。当前 baseline 均已跑通有真实数字；`Ours-BioSeq` 全部待接入。
- **2026-07-02（补 T5）**：把 Generation 拆成 T4（无条件分布匹配）+ **T5（表位条件 TCR design）**。**新落地并跑通**：`common/metrics.py::{design_metrics,aggregate_design}`（含 oracle 校验：exact=recovery=1.0/JSD=0）、`scripts/prepare_tcr_design.py`（common13/rare24/novel7，防泄露断言 leak=0）、`tcr_design/run.py`（Markov / CondPool 训练free baseline + `--method file` 接入点）。产出落 `outputs/tcr_design/{markov3,cond_pool}/metrics.json`。
- **2026-07-02（全量 baseline sweep）**：`scripts/run_all_baselines.sh` 一键复跑全部 baseline，**全部 exit=0**，结果写入 `RESULTS.md`：
  - T1: Random / CDR3-kNN / ESM2-150M / **epiTCR 官方 RF**（unseen AUPRC 依次 0.192/0.167/0.189/**0.195**；epiTCR 在主 env sklearn1.6 下直接跑通，seen AUROC 0.612）。
  - T2: editdist(Purity 0.800) / ESM2(0.273)。 T3: kmer 0.801 / ESM2 0.806 / **Ophiuchus 0.825**（probe-AUROC）。
  - T4: Markov3 JSD 0.041(最优) / Markov2 0.042 / PWM 0.176。 T5: Markov3 & CondPool recovery≈0（下限），oracle 上界 1.0。
  - PPI: k-mer AUROC 0.562 / **ESM2-150M 0.815**。 A1/NbBench(8 任务全跑): nanobody_type acc 0.9997、polyreaction AUROC 0.865、hIL6 0.898、SARS 0.868、thermo-tm ρ0.658 等。
  - 唯一仍 pending = `Ours-BioSeq`（等模型接入）；`output/grammar_v2_*` 已有 ckpt，接入即出对比数字。
- **2026-07-03（分类收敛：T5 并回 T4）**：修正上一版把 Generation 拆成 T1–T5「五榜」的口径。**无条件生成（原 T4）与表位条件设计（原 T5）本质同为"采样生成 CDR3β 序列"的生成任务**，共享生成器主干与序列级指标，差异仅在"是否给定表位条件"。故重新收敛为 **TCR 四大方向：T1 Binding / T2 Clustering / T3 Representation / T4 Generation**。**仅文档口径调整，代码/数据/指标/`outputs/` 全部不变**（`tcr_generation/` = 无条件 setting、`tcr_design/` = 表位条件 setting，均照常跑通）。
- **2026-07-03（广泛文献复核 + 进一步收敛为"方向+setting"）**：按用户要求再做一轮广泛调研（Nat Methods 2025 binding、NAR-GAB 2025 clustering、SCEPTR/Cell Systems 2024 representation、TCRT5/Nat Mach Intell 2025 + TcrDesign + TCRTransBench + LSMTCR generation、Int Immunol 2025 PLM 综述），**确认四大方向划分正确、无第五大类**。把 generation 内部从"T4a/T4b 两个子任务"进一步表述为 **T4 一个方向下的两种 conditioning setting（无条件 / 表位条件）**，去掉子榜感。**新增 §7「文献校准与已知构建局限」**，诚实列出非命名类的实质短板并定优先级：**P0 = T1 近似去重(CD-HIT/Lev≤3, L1)、T1 负样本来源(AS/PS/HS)口径(L2)、T3 缺 TCRdist/Levenshtein-NN 主对照(L4)**；P1 = T2 数据规模(190k/2313, L3)、T4 Pgen 校准+双向/全长扩展(L5)、接入 `Ours-BioSeq`+官方 baseline(L6)。§1 文献表补 LSMTCR / TCRTransBench / PLM 综述；§3 baseline 表 T3 增列 TCRdist/Lev-NN 为主对照。仍为文档层改动，`outputs/` 数字不变。
- **2026-07-05（P0-L4 落地：T3 few-shot 主协议 + SCEPTR/TCRdist/Levenshtein 主对照 + Ours-BioSeq 接入）**：以 SCEPTR (Cell Systems 2024) 为权威口径，**重构 T3 并跑出真实数字**（代码 + `outputs/`，非文档层）。
  - **新增 `common/fewshot.py`**：SCEPTR few-shot per-epitope NN AUROC 引擎，三后端（native cdist / embedding-cosine / editdist）统一成 query×ref 距离矩阵；shots k∈{1,2,5,10,20,50,100}、R=5 seed、min-NN、表位 macro（mean±std）、train binder 不足则跳过。**oracle 自检通过**（支持集含 query 正样本 → mean AUROC 1.0000 / min 0.9997，`python -m common.fewshot`）。
  - **`common/model_api.py`**：新增 `SceptrDistance`（native cdist，tidytcells 标准化 junction+V）、`TcrdistDistance`（tcrdist3 矩形距离）、`build_distance_source` 工厂；**修复 `BioSeqEmbedder` 支持 ESMC 骨架**——新增 `EsmcBioSeqEmbedder` 自动识别 `grammar_v2_*_llada` 检查点，载入微调后 `encoder.esmc.*`（**0 missing/0 unexpected**）做 mean-pool（旧 Ophiuchus 检查点走 `OphiuchusBioSeqEmbedder`）。
  - **重写 `tcr_representation/run.py`**：`--protocol {fewshot,probe,both}` + `--method {levenshtein,tcrdist,sceptr,embed}` + `--shots/--seeds/--columns`；few-shot 落 `fewshot.json`、probe 辅报落 `metrics.json`（向后兼容）。新增 `scripts/summarize_tcr_representation.py`。
  - **真实结果（few-shot AUROC k=20）**：SCEPTR **0.711** > TCRdist 0.687 > Ours-BioSeq esmc300m-cmp500k 0.667 > Ophiuchus 0.664 / kmer 0.643 / Lev 0.641 / ESM2 0.638 / esmc600m 0.648 / esmc300m-mint 0.639。**复现 SCEPTR 方向：SCEPTR ≥ 序列比对法(TCRdist/Lev) ≥ 通用 PLM(ESM2)**；我们的模型在 embedding 类里领先但未超过比对法/SCEPTR（诚实：扩散自监督表征尚未针对表位特异性优化）。
  - **安装**：主 env 直接 pip 装通 `sceptr==1.2.0`（+libtcrlm 1.1.3/tidytcells 2.2.1/blosum 2.2.0）与 `tcrdist3==0.3`（+parasail/pwseqdist/olga），离线权重可用。**无 blocker**。
  - §0 T3 状态、§2 T3 段、§7 L4/L6、§3 baseline 表同步更新；`RESULTS.md` T3 改为 few-shot 主表 + probe 辅表。
- **2026-07-05（补齐 T3 官方 baseline：ProtBERT + TCR-BERT，凑齐 SCEPTR Table SI 六模型对照）**：按"官方权重 + 薄封装"原则补齐 SCEPTR Table SI 中尚缺的两条通用/专用 LM 对照，完成六模型对照（SCEPTR / TCRdist / CDR3-Lev / ESM2 / ProtBERT / TCR-BERT）。
  - **`common/model_api.py`**：新增 `ProtBertEmbedder`（`Rostlab/prot_bert`，空格分隔 AA、mean-pool 末层）与 `TcrBertEmbedder`（`wukevin/tcr-bert-mlm-only`——纯自监督 MLM，避免用 `wukevin/tcr-bert` 分类微调权重造成表位监督泄漏；空格分隔 AA、64 上下文、mean-pool），均为 `HFProteinLMEmbedder` 薄子类；`build_embedder` 注册 `protbert`/`tcrbert`。官方 HF 权重经 `HF_ENDPOINT=hf-mirror.com` 下载并本地缓存，`/c20250601/mj/model_weights/` 下无本地快照。
  - **真实结果（few-shot AUROC）**：k=5/20/100 — ProtBERT 0.587/0.627/0.651、TCR-BERT 0.577/0.623/0.641。**完整复现 SCEPTR 核心论点**：ProtBERT/TCR-BERT/ESM2 均 ≤ CDR3-Levenshtein(0.597/0.641/0.683)，且显著低于 TCRdist(0.644/0.687/0.728)、SCEPTR(0.655/0.711/0.741)；即使 TCR-BERT 为 TCR 专用 MLM，其 mean-pool 表征也未反超朴素编辑距离。probe 辅报 ProtBERT 0.793 / TCR-BERT 0.748（probe-AUROC）。口径参照 Table SI(k=200,6表位均值)：ProtBert≈0.705 / TCR-BERT≈0.735（本 split 24 表位更严，绝对值偏低但相对排序一致）。
  - **catELMo（可选，未接）**：GitHub `Lee-CBG/catELMo` clone 无代理超时；且其 ELMo 栈依赖 `allennlp 2.10`（要求 torch<1.13）与主 env torch 2.8 冲突，需独立老环境 + 外部 ELMo 权重下载。记 **blocker** 跳过，不阻塞主线。
  - `RESULTS.md`（few-shot 主表 + probe 辅表增行）、`downstream.md`（T3 关键表）、`scripts/summarize_tcr_representation.py`（ORDER 增 tcrbert/protbert）、`baselines/README.md`（权重来源）同步更新。`outputs/tcr_representation/{protbert,tcrbert}/` 新增，`_summary.json` 重建。
- **2026-07-05（T3-deep 深化：SCEPTR Table SI III.1 严格复现补齐，deep 升为 T3 主口径）**：把 T3 从"broad 24-epitope few-shot"深化为**双 track**——新增 **deep 6-pMHC** 严格照 SCEPTR Methods III.1 / Table SI 原始协议（6 target pMHC、**背景全 VDJdb 负样本**、k∈{1,2,5,10,20,50,100,200}、**100 seeds**、per-pMHC AUROC macro、era-matched **VDJdb 2023-10**），逐 pMHC 与论文 Table SI 原始数字做 Δ 校验，并把 deep 作为**更"充分"的主口径**、broad 保留为互补大表位面视图。
  - **补齐 deep 缺失方法**：此前 deep 只有 5 baseline（sceptr/tcrdist/levenshtein/esm2_150m/protbert）；本次补 **TCR-BERT(仅CDR3β) + kmer + 3×Ours-BioSeq(grammar_v2 esmc300m/600m)** + 汇总 `_summary.json`。**计算改由 Volcengine ML 单卡作业执行**（本地 A100 100% 占用），新增 `scripts/run_paper6_all.sh`（自包含容错 runner）+ `eval_jobs/eval_paper6_tcr_representation.yml`；task `t-20260705204639-l56dp`，ExitCode 0，输出直写共享 vepfs `outputs/tcr_representation_paper6/`。
  - **deep 真实结果（k=200 macro）**：SCEPTR **0.790** ≳ TCRdist 0.777 > CDR3-Lev 0.733 ≈ kmer 0.728 ≈ Ours-BioSeq esmc600m/300m-cmp500k **0.727/0.724** > ESM2 0.696 ≈ ProtBERT 0.694 ≈ TCR-BERT 0.692 > Ours-BioSeq mint 0.686。**Table SI k=200 逐 pMHC 校验**：Lev/TCRdist/SCEPTR/ProtBERT/ESM2 平均\|Δ\|≈0.006–0.024、单点\|Δ\|≤0.052（忠实复现）；TCR-BERT 平均\|Δ\|≈0.057（TFE −0.113/SPR −0.069，源自官方 MLM-only 编码器 mean-pool 与论文取法差异），macro 仍落 PLM 带内、排序不变。
  - **Ours-BioSeq 诚实定位**：cmp500k(300m/600m) 是**最强 mean-pool PLM 嵌入**（超三个官方 PLM ~0.03、追平 kmer、逼近 CDR3-Lev），但仍低于 TCRdist/SCEPTR 约 0.05–0.06；mint 最弱。接入真实性 CPU 独立核验：`encoder.esmc.*` 0 missing/0 unexpected（300m 308 键 / 600m 368 键）。
  - **文献**：web 复核确认 **SCEPTR (Cell Systems 2024) 仍是 T3 唯一精确锚**（给出 6-pMHC few-shot NN + Table SI 参照）；补 2 篇辅助/未来扩展新文——Feng et al. *Briefings in Bioinformatics 2025* (bbaf030, 19 种 TCR 嵌入基准，"手工嵌入总体优于数据驱动 PLM"，呼应 kmer 表现) 与 Drost et al. *ePytope-TCR, Cell Genomics 2025* (泛化 caveat)。§0 T3 状态、§1 T3 文献表、§2 T3 段、§7 L4/L6 已更新；`downstream.md`/`RESULTS.md` T3 改为 deep 主表 + broad 互补。
- **2026-07-05（P0-L1/L2 落地：T1 近似去重 + AS/PS/HS 负样本来源对照 + 补齐 6 官方 baseline + 3×Ours-BioSeq）**：把 T1 从"NM2025 主协议已跑通但 §7 L1/L2 待修"推进到**结项**，全部真实数字、可复现、禁止编造。
  - **文献确认**：主构建基础仍选 **Nature Methods 2025**（官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark` commit `ec832b47`；不替换），补强 **arXiv 2606.04994 (2026)** 的 Lev≤k 近似去重 + Macro-AUC0.1，佐证 **Sci Rep 2025 (s41598-025-26454-7)** 的编辑距离分层。2026 新数据(TetTCR-SeqHD/Fingerprinting)是 perspective/新实验、未打包成带官方 baseline 的可复现 split → 方法论补强而非新主基座（§1 T1 文献表已更新）。
  - **L1 近似去重**（`scripts/nm2025_near_dedup.py`，本地训练free）：train↔test CDR3β Levenshtein≤k（rapidfuzz cdist），k=1/2/3。**去掉 unseen 19.0%/67.4%/92.6%、seen 9.5%/57.3%/87.5%**；在同一硬子集对所有方法复算 overall AUPRC。结论：k≤2 去 unseen 67.4% 落 arXiv 40–70% 区间；去重后 unseen 仍全体≈0.5（泛化失败为真），但 **kNN 表观优势 k≤1 即坍缩 0.544→0.487**（距离法吃近重复红利）；Ours/ATM-TCR 变化温和。产物 `data/tcr_binding_nm2025/near_dedup_report.json` + `outputs/.../near_dedup_summary.csv`，原口径全保留。
  - **L2 AS/PS/HS 对照**（`scripts/summarize_nm2025.py`）：cdr3b track 六套 test（AS/PS/HS×seen/unseen）主指标成表（`summary_neg_source.csv`）。**复现"负样本来源影响>模型结构"**：同一模型换负样本，unseen AUPRC 摆动 ~0.13–0.15（Ours-cmp500k PS 0.469→HS 0.623；TEIM AS 0.518→HS 0.596），大于固定 AS 的模型间差异（0.479–0.544）；HS 系统性抬分。主榜坚持 AS。
  - **补齐官方 baseline（6 个，只用官方权重）**：除已有 ATM-TCR/epiTCR/PanPep，新增 **NetTCR-2.2(b)**、**TEIM(seq)**、**ERGO-II(vdjdb)**。TEIM 用 `Original_model_prediciton/ckpt/{teim_seq,epi_ae}.ckpt` 在主 env 跑通（`torch.load(weights_only=False)` + 主 env ANARCI/hmmscan 做 CDR3 IMGT numbering）；ERGO-II 用 `Original_model/ERGOII_vdj` + `Models/AE` 在主 env 经**框架兼容 shim**（PL-0.7→1.8.6：`data_loader` 装饰器 / `pytorch_lightning.logging` / 可写 `hparams` / CUDA→CPU map_location）载入，**0 missing/0 unexpected**（AE 权重 `ln -s Models/AE TCR_Autoencoder`）。均**不改模型数学**。wrapper `atmtcr.py`/`nettcr.py` 修复 `LD_LIBRARY_PATH` 隔离（避免主 env libtorch 串味）。
  - **我们的模型（3 checkpoint）**：esmc300m-cmp500k 已有；**esmc600m-cmp500k / esmc300m-mint 经 Volcengine ML 单卡作业 `t-20260705204753-5jwgf`**（`eval_jobs/eval_tcr_binding_nm2025_ours_extra_ckpts.yml`，输出直写共享 vepfs）跑出六套 test。3 者 encoder 均 0 missing/0 unexpected。unseen overall AUPRC 最好 0.534（esmc300m-cmp500k）。
  - **主表（cdr3b·AS·overall AUPRC）**：seen ATM-TCR 0.696 / TEIM 0.676 / NetTCR 0.611；unseen 全体 ≈0.48–0.54（Ours-cmp500k 0.534 最高，kNN 0.544 去重后坍缩）。ERGO-II 因仅喂 CDR3β（原生需 α/V/J/MHC）seen 近随机属预期，已标注。
  - §0 T1 状态、§1 T1 文献表、§7 L1/L2/L6、`downstream.md` T1、`RESULTS.md` T1、`baselines/README.md`、`PROGRESS.md` 同步更新；`outputs/tcr_binding_nm2025/{nettcr,teim,ergo2,ours_biseq_mlp_esmc600m,ours_biseq_mlp_esmc300m_mint,near_dedup_summary.csv,summary_*.csv}` 新增。
- **2026-07-05（T3-deep 口径修正：我们模型改用正确的 LLaDA decoder 多链表征，只重跑 3 个模型）**：应用户"重要修正"——T3-deep 里 `Ours-BioSeq` 用 ESMC encoder-only mean-pool（`bioseq:…`）是**对我们模型的低估**（每链孤立过编码器、未用 LLaDA 生成主干）。正确口径 = **LLaDA 解码器末层 + 训练时多链 grammar 输入格式(β+α 联合) + 无噪声(干净全可见前向)**。**baseline 全部不动、不重跑**，仅改 3 个模型并重跑。
  - **harness 改动（最小侵入，不破坏其它 embedder 逐列路径）**：`common/fewshot.py` 新增 `grammar_pair_embed()` + 在 `embedding_distance_matrix` 加分支；`tcr_representation/run_paper6.py::_embed_vectors` 与 `run.py`（broad 一并改保持一致）对 `getattr(emb,"feature_source",None)=="decoder"` 的 embedder 改走 `embed_pairs(beta=cdr3b, alpha=cdr3a, peptide=None, order=("beta","alpha"))`；`_tag_for` 支持 `grammar:decoder:` → tag `grammar_dec_*`；`summarize` 的 ORDER 加 3 个新 tag（旧 `bioseq_*` 保留对照）。判定依据：`GrammarEmbedder.feature_source=="decoder"`（仅我们的 grammar 模型有此属性，其余 esm2/protbert/tcrbert/kmer/bioseq 保持原逐列路径）。
  - **防表位泄漏**（few-shot NN 协议要求）：peptide 槽喂**恒定占位**（`peptide=None`→"A"，对所有行一致，无判别性泄漏），输出**只取 β+α 段、丢 peptide 段**。**冒烟校验**（540 行子集/shots{5,20}/seeds20）：dim=1920=2×decoder_hidden（≠encoder-only），macro AUROC 0.64/0.66，远低于 1.0 且低于 SCEPTR/TCRdist → **确认无泄漏**。
  - **提交**：新建 `scripts/run_paper6_ours_decoder.sh` + `eval_jobs/eval_paper6_ours_decoder.yml`，no-proxy `ml_task submit`，task **`t-20260705214312-k8ktb`, ExitCode 0**，输出直写 `outputs/tcr_representation_paper6/grammar_dec_*`，未覆盖旧 encoder-only。
  - **新 decoder vs 旧 encoder-only（k=200 macro，如实）**：cmp500k-300m **0.714** vs 0.724；cmp500k-600m **0.721** vs 0.727；mint **0.660** vs 0.686。**改用生成主干后 few-shot 并未提升，与 encoder-only 基本持平、略降**（cmp500k ~0.01、mint ~0.026）。我们的模型在 baseline 中的位置不变：仍是最强 PLM 嵌入档（超 ESM2/ProtBERT/TCR-BERT ~0.02–0.03、逼近 kmer/Lev），但低于 TCRdist(0.777)/SCEPTR(0.790)。§0、§2 T3、§7 L4/L6 与 `downstream.md`/`RESULTS.md` T3 同步更新。
- **2026-07-05 22:50（UTC+8）post-LLaDA 方法学修正（用户指示 · 边做边写 MD）**：
  - **§5 重写**：post-LLaDA 特征定义表（读out 层、单链/成对池化、embedder spec）；headline = `bioseq-llada:` / `grammar:decoder:`；`bioseq:` 仅 enc-only 对照。
  - **实现**：`EsmcLladaBioSeqEmbedder`（`common/model_api.py`）；`embed_pairs` β-only 行零填充修复（`model_api.py:780-788`）。
  - **T2 ✅**：Volc `t-20260705215519-nck2s` T2A+T2B 完成；post-LLaDA vs enc-only 对照已写入 `downstream.md`/`RESULTS.md`（基础A 纯度略降、基础B ARI 略升）。
  - **T1+T3broad 🔄**：续跑 `t-20260705224442-rn9lx`（`eval_ours_postllada_downstream_resume.yml`）。
