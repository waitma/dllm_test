# IRBench — 免疫受体基础模型下游评测基准

> 根目录: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`
>
> 为 `dllm_test` 的扩散式免疫受体基础模型 (BioSeq) 搭建统一、可复现、防数据泄露的下游评测，
> 覆盖 TCR 的 binding / clustering / representation / generation、PPI 与抗体任务。baseline 按论文值、官方 artifact、官方代码重跑和本地重实现分层记录。
>
> **Agent 必读**：[`PROJGUIDE.md`](PROJGUIDE.md) — 主模型中心论、post-LLaDA 强制口径、文档实时更新规则。

## 1. 当前定位
- **状态（2026-07-21）**：已完成 baseline provenance audit；本轮不修改 Ours 的模型、checkpoint 或输出。T1 采用 original-only 决策：完整的官方 checkpoint + 官方推理 + `original.zip` 本地结果优先，否则采用论文 original-model 值并标“论文值，未本地复现”。其余任务继续按 `[P]/[A]/[R]/[L]/[C]` 分层，不把协议不同的数字混排。
- T1–T4、MINT、NbBench、Ophiuchus CDR 与 FLAb 均已有论文锚点；本地 pipeline 继续作为复现审计或 diagnostic。完整判定见 `BASELINE_VERIFICATION.md`，结果见 `RESULTS.md`，过程见 `PROGRESS.md`。
- **A1/NbBench**：论文 Table 5 是 **11 个任务、MLP head、3 seeds** 的 `[P]` 主表；本地 sklearn single-seed probe 含一个额外 hTNFa 任务，仅列作 `[L]` diagnostic。Paratope 论文主指标为 AUROC，本地 AUPRC 不与其同列比较。

### 快速开始
```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark
# GPU 任务前置（避免 libstdc++ 链接问题）：LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

# T1 binding（防泄露 seen/unseen）
$ENV/bin/python scripts/prepare_tcr_binding.py
$ENV/bin/python tcr_binding/run.py --method knn          # TCRdist 式距离
$ENV/bin/python tcr_binding/run.py --method embed --embedder esm2_150m
# NM2025 external original-model baseline（不读 train.csv、不 retrain）
$ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name atmtcr --track cdr3b --neg-source AS
$ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name teinet --tag teinet_small --track cdr3b --neg-source AS
$ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py  # canonical 本地/论文 fallback 选择
# NM2025 retrained checkpoint artifact audit（独立诊断，不进入 original-only 主表）
$ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py --name all --fold all --eval-set all --neg-source AS
# T2 clustering（论文 Retention/Purity 为 [P] 主表；官方输出本地复算 + 阈值曲线为审计）
$ENV/bin/python scripts/prepare_tcr_clustering.py                          # curated 数据集（需 baselines 仓库）
$ENV/bin/python scripts/import_clustering_baselines.py                     # 导入 9 官方方法输出 + 口径校验
$ENV/bin/python tcr_clustering/run.py --method precomputed --all          # 官方输出的本地复算；不自动等同论文值
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder esm2_150m  # 我们的参照嵌入/模型（非官方 baseline）
# T3 representation：few-shot per-epitope NN AUROC 主协议（SCEPTR 口径）+ 24-way probe 辅报
$ENV/bin/python scripts/prepare_tcr_repertoire.py                          # T3 数据准备
$ENV/bin/python -m common.fewshot                                          # few-shot 引擎 oracle 自检
$ENV/bin/python tcr_representation/run.py --method sceptr                   # 或 levenshtein / tcrdist
$ENV/bin/python tcr_representation/run.py --method embed --embedder esm2_150m  # 或 kmer / ophiuchus / bioseq:/abs/best.pt
# T4 generation
$ENV/bin/python scripts/prepare_tcr_generation.py
$ENV/bin/python tcr_generation/run.py --method markov --order 3
# P1 PPI
$ENV/bin/python scripts/prepare_ppi.py
$ENV/bin/python ppi/run.py --embedder kmer
# A1 NbBench（全 12 任务：9 序列级 + 3 位点级）
$ENV/bin/python nbbench/run.py --task all --embedder esm2_150m        # 序列级 + --embedder kmer 对照
$ENV/bin/python nbbench/run_residue.py --task all --embedder esm2_150m # 位点级 + --embedder onehot 对照

# 接入基础模型（训练完成后，对任一任务）：
$ENV/bin/python tcr_binding/run.py --method embed --embedder bioseq:/abs/path/final.pt
$ENV/bin/python nbbench/run.py --task all --embedder bioseq:/abs/path/final.pt
```

## 2. 整合的现有基准（优先复用，不重造）
| 维度 | 整合的现成基准 | 本地状态 |
|------|----------------|----------|
| 框架范式 + 抗体/纳米抗体 | **NbBench**（论文 Table 5 共 11 tasks `[P]`；本地另含 hTNFa `[L]`） | 已在 `data/nanobody_raw/nbbench`（数据+脚本齐全） |
| TCR 数据与防泄露 split | **TCRpMHCdataset (PIRD)** | 已在 `data/tcr/PIRD_alt` |
| TCR binding | **IMMREP23/22**（自带 seen/unseen split + 全长TCR + 官方评测/结果） | 已 clone 到 `baselines/` |
| PPI | **STRING 90/90**（Bernett 式 gold standard） | 已在 `data/ppi` |
| 抗体 CDR / 属性回归 | Ophiuchus-Ab Table 2、MINT Figure 3b | 论文值 `[P]` 已导入；历史本地 CDR/FLAb sweep 仅为 `[L]` diagnostic |

> 框架骨架直接复用 NbBench 的「模型 wrapper + 统一 train/val/test + head-only probing」，
> 把 TCR/PPI 任务接进同一范式，并新增 BioSeq/Ophiuchus 模型 wrapper。

## 3. 任务总览

> **TCR 四类精简方案**见 `TCR_TASK_TAXONOMY.md`（文献调研 2023–2026 → 贵精不贵多，每类 1 主任务）。

> **TCR 基准主设计入口 = `TCR_BENCHMARK_DESIGN.md`**（文献驱动、含 T1–T4 四大方向口径，其中 T4 Generation 含无条件 + 表位条件设计两种 setting；附源码-only baseline 清单、实时状态、以及 §7 已知构建局限）。下表为速览。

> **TCR = 四大方向（T1–T4）**。Generation 是一个方向，按"是否给定表位条件"分**两种 setting**：无条件分布匹配 + 表位条件设计（"TCR design"）；二者本质同为"生成 CDR3β 序列"，共享生成器与序列级指标，是同一类的两种设定、不另立第五榜。

| # | 任务 | 整合基准 | 类型 | 主指标 |
|---|------|----------|------|--------|
| T1 | TCR Binding | Nature Methods 2025 original CDR3β/AS `[P]` | 二分类 | seen/unseen overall AUPRC（辅 AUROC） |
| T2 | TCR Clustering | NAR-GAB Retention/Purity `[P]` + TCREmbedding `[L]` | 无监督聚类 | Retention / Purity；本地 ARI/NMI |
| T3 | TCR Representation | SCEPTR Table SI `[P]` + compatible local audit `[L]` | few-shot NN + linear probe | per-pMHC AUROC |
| T4 | **TCR Generation** | TCRT5 sparse-13 official artifact `[A]`; 14-pMHC 仅 control | 采样 / 表位条件设计 | F1 / edit distance / sequence recovery / BLEU |
| P0 | MINT GeneralPPI | MINT Source Data Figure 2c–h `[P]` | 分类 / 回归 | 论文 task-specific primary metric |
| P1 | PPI | STRING 90/90 | 二分类 | AUROC / AUPRC |
| A1 | 抗体/纳米抗体 | NbBench Table 5（11 tasks）`[P]`; 本地第 12 项另列 `[L]` | 分类/回归/位点级/生成 | 论文各任务 metric；Paratope=AUROC |
| AB | 抗体生成 | Ophiuchus-Ab CDR Table 2 `[P]` + OAS pairing `[L]` | 生成 | CDR=AAR；pairing=ImmunoMatch |
| AB-FLAB | 抗体属性回归 | MINT Figure 3b `[P]` | frozen embed + nested Ridge | R²（10 outer × 5 inner CV） |

> AB / AB-FLAB 详情见 [`../downstream.md`](../downstream.md) AB 段 · [`../flab/README.md`](../flab/README.md) · [`../infill/README.md`](../infill/README.md) · 结果 `output/downstream_generation/flab_baselines/`、`cdr_baselines/`。

## 4. 数据泄露防控（核心）
| 任务 | 防控 |
|------|------|
| T1 | unseen-epitope 与 seen-epitope 两套 split；CDR3β 去重且跨 split 互斥；负样本用 reference-TCR 随机配对 |
| T2 | 仅用测试表位的 TCR；表位标签只用于打分，不参与聚类 |
| T3 | 复用 T1 unseen split；冻结主干只训 probe |
| T4 | 用 OTS holdout；报告与训练集 CDR3 的新颖度/最近邻距离 |
| P1 | 直接用 STRING 90/90 split（两端蛋白互相 <90% 相似） |

## 5. 数据来源（本地）
| 任务 | 路径 |
|------|------|
| T1/T2/T3 | `data/tcr/`（VDJdb / McPAS / MIRA / IEDB / PIRD） |
| T4 | `data/ots_paired_clean/final`、`data/downstream/cdr_infilling/tcr` |
| P1 | `data/ppi/string_model_org_90_90_split` |
| A1 | `data/nanobody_raw/nbbench/hf_data`（NbBench 自带） |

## 6. baseline 证据规则

> **原则**：协议一致且能验证时采用官方 artifact/代码；无法等价复现或本地结果不一致时，主表引用论文值 `[P]` 并明确标注。任何本地重实现 `[L]` 或控制 `[C]` 都必须与论文主表分开，不以“趋势接近”冒充复现。统一字段与判定规则见 `common/baseline_provenance.py` 和 `BASELINE_VERIFICATION.md`。

| 任务 | 外部 baseline / 可用性 |
|------|-----------------------------------------|
| Binding | NetTCR-2.2 `mnielLab/NetTCR-2.2`、ERGO-II `IdoSpringer/ERGO-II`、pMTnet `tianshilu/pMTnet`、epiTCR `ddiem-ri-4D/epiTCR`、PanPep `bm2-lab/PanPep` |
| Clustering | tcrdist3 `kmayerb/tcrdist3`、GIANA `s175573/GIANA`、clusTCR `svalkiers/clusTCR`、DeepTCR `sidhomj/DeepTCR` |
| Representation | TCR-BERT `wukevin/tcr-bert`、SCEPTR `yutanagano/sceptr`、catELMo `Lee-CBG/catELMo`；NbBench 自带 ESM2/ProtBERT/IgBERT/AntiBERTa2/AbLang/AntiBERTy/VHHBert/NanoBERT |
| Generation · 无条件 setting (T4) | OLGA `statbiophys/OLGA`、soNNia `statbiophys/soNNia` |
| Generation · 表位条件 setting (T4) | TCRT5 sparse-13 artifact `[A]`；TcrDesign 可本地评分；TCR-epiDiff 因官方生成路径不完整仅引用，不宣称已复现 |
| PPI | D-SCRIPT/Topsy-Turvy `samsledje/D-SCRIPT`、SENSE-PPI `AlbertMolina/SENSE_PPI`；口径对齐 MINT(protein)（`../MINT_MIGRATION.md`） |

> clone 到 `baselines/<name>` 时记录 commit/许可证；即使缺少可运行代码，论文报告值仍可作为 `[P]` 收录，但不得标成 `[R]`。

## 7. 目录结构（已实现）
```
benchmark/
├── README.md / RESULTS.md / PROGRESS.md
├── common/          # 已实现：schema, leakage, negatives, metrics(含 BLOSUM62 BR), model_api, featurizers
├── scripts/         # 已实现：prepare_tcr_binding / _repertoire / _generation / _ppi, import_immrep22_results, import_nbbench_results
├── tcr_binding/run.py      tcr_clustering/run.py   tcr_representation/run.py
├── tcr_generation/run.py   tcr_design/run.py       ppi/run.py
├── nbbench/         # A1 抗体/纳米抗体：tasks.py + run.py(序列级) + run_residue.py(位点级/生成, 含 onehot 对照)
├── baselines/wrappers/  # 外部 baseline 薄封装；每项单独声明 provenance
├── baselines/       # 公开 baseline（git clone，本体不入库）
├── data/            # 各任务生成的数据集 + leakage_report.json（gitignore）
└── outputs/         # 预测文件 + metrics.json + external/{immrep22,nbbench}_official.csv（gitignore）
```

## 8. 统一模型接入接口（核心抽象）
所有模型实现一个极简接口即可上榜：
```python
class SequenceEmbedder:
    name: str; dim: int
    def embed(self, seqs: list[str]) -> np.ndarray  # [N, dim]
```
任务用 `build_embedder(spec)` 构造（`esm2_150m` / `ophiuchus` / `bioseq:/path/final.pt`），
按需嵌入相关列（CDR3β/CDR3α/peptide/全长链）并拼接，再接 probe / 距离 / 聚类。
**基础模型训练完成后，唯一要做的就是让 `BioSeqEmbedder` 指向其 backbone checkpoint。**

## 9. 已落地决策（原"待确认"）
1. **Binding 粒度**：默认特征 = CDR3β + CDR3α + peptide（`--columns` 可调）；距离基线用 CDR3β(+α)。
2. **GPU 时机**：边搭边跑——轻量(kNN/editdist/kmer/Markov)在 CPU 即时跑；嵌入类用本机单 GPU；
   大规模/基础模型嵌入后续可 `volc submit`。
3. **优先级**：已覆盖 T1–T4、PPI 与 NbBench；“已覆盖”不等于所有本地值均 paper-comparable，具体以 provenance audit 为准。
4. **后续可扩展**：clone clustering/representation/generation/PPI 的官方深度 baseline（tcrdist3/GIANA/SCEPTR/D-SCRIPT 等）、
   PIRD unseen-pHLA split、paired-αβ binding 子集、TCR-pMHC 结构 oracle；A1 可接入更多 NbBench 抗体专用 LM（AbLang/AntiBERTa2/NanoBERT 等）作 wrapper 对照。
