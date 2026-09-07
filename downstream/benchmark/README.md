# IRBench — 免疫受体基础模型下游评测基准

> 根目录: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`
>
> 为 `dllm_test` 的扩散式免疫受体基础模型 (BioSeq) 搭建统一、可复现、防数据泄露的下游评测，
> baseline 按论文值、官方 artifact、官方代码重跑和本地重实现分层记录。
>
> **Agent 阅读顺序（唯一）**：[`PROJGUIDE.md`](PROJGUIDE.md)（怎么做）→ [`../tasks/<TASK>.md`](../tasks/)（做什么、怎么填）→ [`RESULTS.md`](RESULTS.md) §0（数字）。
>
> **现役 v3 diffusion 评哪个 ckpt / 怎么提交**：见 [`examples/llada/README.md`](../../examples/llada/README.md)「下游评测（v3 diffusion）」。

## 0. 评测范围（2026-08-29 收窄，强制）

> **当前只做 AB（抗体）与 TCR 两个维度的任务，其余一律不考虑。**

**在范围内**

| 维度 | 任务 |
|---|---|
| TCR | T1 Binding · T2 Clustering · T3 Representation · T4 Generation |
| AB | CDR infilling（SAbDab Kong / SAb23H2）· Light-chain pairing |
| AB（有 harness、无本地数据或按计划排除） | Humanization · Developability (GDPa1) · Specificity (CoV/Flu/HD) · Binding affinity (m396) |

**已冻结，不在范围内**

| 维度 | 任务 | 冻结说明 |
|---|---|---|
| 通用蛋白 PPI | P0 MINT GeneralPPI（`../mint_tasks/`） | 非免疫受体任务 |
| 通用蛋白 PPI | P1 STRING 90/90 PPI（`ppi/`） | 非免疫受体任务 |
| 纳米抗体 | A1 NbBench 11/12 任务（`nbbench/`） | 与 AB + TCR 主线分开 |
| 抗体属性回归 | AB-FLAB（MINT Figure 3b 锚点，`../flab/`） | 锚点是 MINT 而非 AB 主线 |

冻结含义：**代码、数据与已落盘产物全部保留**，但不再更新数字、不进 headline、不写入论文表、不作为结论依据。
本文件下方各表中标 ⛔ 的行即属此列，保留是为了记录已做过的工作与出处，不代表当前口径。

**本节是范围口径的唯一定义处。** 同一声明已同步到另外四处，改范围时五处必须一起改：
[`PROJGUIDE.md`](PROJGUIDE.md) §0.1.1 · [`RESULTS.md`](RESULTS.md) 顶部横幅 · [`../README.md`](../README.md)「评测范围」段 · [`../tasks/README.md`](../tasks/README.md)。
[`../AB_TCR_EVAL_SUMMARY.md`](../AB_TCR_EVAL_SUMMARY.md) 已 ARCHIVE，不再作为范围同步点。

## 1. 当前定位
- **状态（2026-07-21）**：已完成 baseline provenance audit；本轮不修改 Ours 的模型、checkpoint 或输出。各任务按 `[P]/[A]/[R]/[L]/[C]` 分层，不把协议不同的数字混排。
- **🔻 排行榜构造规约已于 2026-09-02 反转为「论文值优先」**：每个已列 baseline 必须带上它自己的最佳已发表值作为 `[P]` 主榜行；`[R]` 官方代码本地重跑降为**附行**，永不取代 `[P]`；复跑失败也不得删行（标 `未获取` + 原因）；协议不一致时同行披露协议差。**T1 的 original-only 决策同步反转**——旧口径是「官方环节可用则取 `[R]`，不可用才 fallback 到 `[P]`」，现为 `[P]` 主榜 + `[R]` 附行。完整规则见 [`PROJGUIDE.md`](PROJGUIDE.md) §0.2.2 与 [`.cursor/rules/downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc) §3。
- T1–T4 与 Ophiuchus CDR / pairing 均已有论文锚点；本地 pipeline 继续作为复现审计或 diagnostic。完整判定见 `BASELINE_VERIFICATION.md`，结果见 `RESULTS.md`，过程见 `PROGRESS.md`。
- ⛔ **MINT / NbBench / FLAb 已按 §0 冻结**，其论文锚点与历史 provenance 判定保留在 `BASELINE_VERIFICATION.md`，但不再推进。
- ⛔ **A1/NbBench（已冻结）**：论文 Table 5 是 **11 个任务、MLP head、3 seeds** 的 `[P]` 主表；本地 sklearn single-seed probe 含一个额外 hTNFa 任务，仅列作 `[L]` diagnostic。Paratope 论文主指标为 AUROC，本地 AUPRC 不与其同列比较。

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

# AB CDR infilling / light pairing（Ophiuchus-Ab 官方 ckpt [R]）
bash ../ophiuchus_eval/run_eval.sh

# ⛔ 以下已按 §0 冻结，保留仅为存档，不要用于出数：
#   P1 PPI      : scripts/prepare_ppi.py + ppi/run.py --embedder kmer
#   A1 NbBench  : nbbench/run.py --task all / nbbench/run_residue.py --task all
#   AB-FLAB     : ../flab/run_flab_all.sh

# 接入基础模型（训练完成后，对范围内的任一任务）：
$ENV/bin/python tcr_binding/run.py --method embed --embedder bioseq:/abs/path/final.pt
```

## 2. 整合的现有基准（优先复用，不重造）
| 维度 | 整合的现成基准 | 本地状态 |
|------|----------------|----------|
| TCR 数据与防泄露 split | **TCRpMHCdataset (PIRD)** | 已在 `data/tcr/PIRD_alt` |
| TCR binding | **IMMREP23/22**（自带 seen/unseen split + 全长TCR + 官方评测/结果） | 已 clone 到 `baselines/` |
| 抗体 CDR | Ophiuchus-Ab Table 1 / Table 2 / Table 3 | 论文值 `[P]` 已导入；历史本地 CDR sweep 仅为 `[L]` diagnostic |
| ⛔ 框架范式 + 纳米抗体 | **NbBench**（论文 Table 5 共 11 tasks `[P]`；本地另含 hTNFa `[L]`） | 已冻结（§0）；数据仍在 `data/nanobody_raw/nbbench` |
| ⛔ PPI | **STRING 90/90**（Bernett 式 gold standard） | 已冻结（§0）；数据仍在 `data/ppi` |
| ⛔ 抗体属性回归 | MINT Figure 3b | 已冻结（§0） |

> 框架骨架直接复用 NbBench 的「模型 wrapper + 统一 train/val/test + head-only probing」，
> 把 TCR 任务接进同一范式，并新增 BioSeq/Ophiuchus 模型 wrapper。**沿用 NbBench 的代码范式不等于 NbBench 任务在范围内**——后者已按 §0 冻结。

## 3. 任务总览

> **TCR / AB 现行协议入口 = [`../tasks/`](../tasks/)**。`TCR_TASK_TAXONOMY.md` 与 `TCR_BENCHMARK_DESIGN.md` 已 ARCHIVE，只作历史。

> **TCR = 四大方向（T1–T4）**。Generation 是一个方向，按"是否给定表位条件"分**两种 setting**：无条件分布匹配 + 表位条件设计（"TCR design"）；二者本质同为"生成 CDR3β 序列"，共享生成器与序列级指标，是同一类的两种设定、不另立第五榜。

**在范围内（AB + TCR）**

| # | 任务 | 整合基准 | 类型 | 主指标 |
|---|------|----------|------|--------|
| T1 | TCR Binding | Nature Methods 2025 original CDR3β/AS `[P]` | 二分类 | seen/unseen overall AUPRC（辅 AUROC） |
| T2 | TCR Clustering | NAR-GAB Retention/Purity `[P]` + TCREmbedding `[L]` | 无监督聚类 | Retention / Purity；本地 ARI/NMI |
| T3 | TCR Representation | SCEPTR Table SI `[P]` + compatible local audit `[L]` | few-shot NN + linear probe | per-pMHC AUROC |
| T4 | **TCR Generation** | TCRT5 sparse-13 official artifact `[A]`; 14-pMHC 仅 control | 采样 / 表位条件设计 | F1 / edit distance / sequence recovery / BLEU |
| AB | 抗体生成 | Ophiuchus-Ab CDR Table 1/2 `[P]` + Table 3 pairing `[P]` | 生成 | CDR=AAR；pairing=ImmunoMatch |

> AB 详情见 [`../tasks/AB_CDR_INFILLING.md`](../tasks/AB_CDR_INFILLING.md) · [`../tasks/AB_LIGHT_CHAIN_PAIRING.md`](../tasks/AB_LIGHT_CHAIN_PAIRING.md)。结果只写 `RESULTS.md` §0.5 / §0.6。

**⛔ 已冻结（按 §0 不在范围内，仅存档）**

| # | 任务 | 整合基准 | 冻结原因 |
|---|------|----------|----------|
| ⛔ P0 | MINT GeneralPPI | MINT Source Data Figure 2c–h `[P]` | 通用蛋白 PPI，非免疫受体 |
| ⛔ P1 | PPI | STRING 90/90 | 通用蛋白 PPI，非免疫受体 |
| ⛔ A1 | 纳米抗体 NbBench | NbBench Table 5（11 tasks）`[P]` | 纳米抗体，与 AB + TCR 主线分开 |
| ⛔ AB-FLAB | 抗体属性回归 | MINT Figure 3b `[P]` | 锚点是 MINT 而非 AB 主线 |

> 冻结任务的历史资料保留在 [`../mint_tasks/README.md`](../mint_tasks/README.md) · [`nbbench/`](nbbench/) · [`../flab/README.md`](../flab/README.md) · [`../downstream.md`](../downstream.md)，**不再更新、不进 headline**。登记表 `outputs/external/paper_reported_baselines.csv` 里这三个基准的 197 行（NbBench 121 + MINT 48 + FLAb 28）**保留为存档但已打 `in_scope=False`**——转录的论文值重新取回代价很高，故不删除，只标记；任何消费方按 `in_scope` 过滤即可，不要另建第二份文件。在范围内为 360 行。

### 3.1 锚点登记表（唯一权威；其余文档只允许链接到此，不得复述）

每个在范围任务在此登记它的**锚点论文、数据集版本、切分、指标定义、baseline 出处层、已披露偏离、可引用性**。任何对外数字都必须能落到这张表的某一行上。出处层记号：`[P]` 论文报告值 · `[A]` 官方 artifact 重评分 · `[R]` 官方代码本地重跑 · `[L]` 本地重实现 · `[C]` 本地控制。

| 任务 | 锚点论文（DOI） | 数据集版本 | 切分 | 主指标定义 | baseline 出处层 | 可引用性 |
|---|---|---|---|---|---|---|
| **T1** Binding | Nature Methods 2025 · [10.1038/s41592-025-02910-0](https://doi.org/10.1038/s41592-025-02910-0) · repo `SuoLab-GZLab/TCREpitopeBenchmark`@`ec832b47` | figshare `original.zip`（seen 3 表位 / unseen 40，neg=AS）；retrained 协议用 `retrain.zip` 五个 AS fold | ① original-only（现成官方 ckpt，零训练）② retrained 五折（冻骨干 + 每 fold MLP）。**禁止混排** | overall AUPRC，**估计量 = R `precrec::evalmod`**（非 sklearn AP，后者单侧偏高 +0.001/+0.002） | 13×`[R]` + SETE/TEPCAM `[P]` + kNN/Random `[C]`；58 行 `[P]` 出自 Supplementary Table 4-1/4-2 | ⚠️ **有条件**：original unseen 须附 `TTAATHREK` 污染注；retrained unseen 可引用；retrained seen 仅作同口径相对比较 |
| **T2** Clustering | NAR Genom Bioinform 2025 · [10.1093/nargab/lqaf150](https://doi.org/10.1093/nargab/lqaf150) · repo `i3-unit/TCR_Unsupervised_Benchmark`@`ac767882` | curated pooled DB（IEDB+McPAS+VDJdb），本地 curation **已精确复现论文四常量**（4,779 配对 / 8,395 序列 / 4,103 α / 4,292 β）；HD/LD 本地重建 | 表位标签只用于评测，不参与聚类 | Retention / Purity（Th=1，簇大小 > 1）；本地辅报 ARI/NMI | 36 行 `[P]` 出自 Figure 3A（9 方法 × 4 指标）；本地为 `[L]` | ⚠️ **`[P]` 为主表**：本地校准 **9/9** 落在 \|Δ\|≤0.03（最大 \|Δ\| 0.005）；HD/LD 存在 Fig 3A 与补充表 S2 矛盾（已披露）；(b) 区为塌缩标签口径（purity 低估 ≤0.0023，已量化）；**(c) 区嵌入曲线仍在旧 5,368 universe 上，待重跑，不可引用** |
| **T3** Representation | SCEPTR · [arXiv:2406.06397](https://arxiv.org/abs/2406.06397) | 六个 pMHC，k=200 | few-shot per-epitope NN AUROC 为**主**任务；24-way linear probe 仅辅报 | per-pMHC AUROC | 36 行 `[P]` 出自 Table SI；本地重建 `[L]` | ⚠️ **须按输入字段分层**：SCEPTR / TCRdist 额外使用 V(/J) 基因，我方与其他 PLM 只有 CDR3β+CDR3α；本地跑数对论文值有系统性负偏（已披露） |
| **T4** Generation | TCRT5 · [10.1038/s42256-025-01096-6](https://doi.org/10.1038/s42256-025-01096-6) | 官方 `benchmark_data_w_preds.csv`；主榜 = sparse-13 | 表位条件设计。`RVRAYTYSK/HLA-A*03:01` 是论文 simulation 保留项，已剔除；held20 是**验证**集非测试集 | F1 / edit distance / sequence recovery / Char-BLEU | sparse-13 `[A]`（官方 artifact 重评分）+ 84 行 `[P]`（Supplementary Table 2，**另一套协议，不可与 sparse-13 合表**） | ⚠️ **跨模型只能用 `bioseq_unseen_common`（固定 6 个 pMHC）**；`bioseq_unseen` 成员随 run 变化不可比；Char-BLEU 的 greedy/sampled 口径跨模型不一致；Setting A 的 `ours_bioseq` 存在出处缺陷（已记录） |
| **AB** CDR infilling | Ophiuchus-Ab · [10.64898/2026.02.02.703197](https://doi.org/10.64898/2026.02.02.703197) | Table 1 = SAb23H2（`IgGM_Test_set`，n=60）；Table 2 = SAbDab Kong 划分（n=3,127，十折） | 六个 CDR 分别掩码回填 | AAR（%），**固定 `iter=1`**（此前"1/2/4/8 各取最优"等于测试集调参，已撤回） | 86 行 `[P]`（Table 1 + Table 2）+ 官方 ckpt `[R]` | 🚨 **Ours 行只能作上界**：抗体侧去污染被禁用，实测 SAbDab Kong 折 **54.4%** 测试抗体的 CDR-H3 原样在训练语料中（胚系背景对照 0.83%）。SAb23H2 污染面 25%，较可引用 |
| **AB** Light pairing | Ophiuchus-Ab · [10.64898/2026.02.02.703197](https://doi.org/10.64898/2026.02.02.703197) | OAS holdout500，每条重链生成 8 条轻链 | Table 3 有**两种条件化设定**（仅重链 / 重链+轻链前 3 残基），方向相反，必须先声明属于哪一种 | ImmunoMatch（主）+ 9 项辅助 | 60 行 `[P]`（Table 3）+ 官方 ckpt `[R]`（Ophiuchus-Ab / p-IgGen / LICHEN） | ✅ 已复现：LICHEN 九项均在 0.015 内，p-IgGen 除 ImmunoMatch/Better 外在 0.023 内。`W_property` 本地未实现，指标覆盖 9/10。机制已定位：我方仅捕获官方 ~30% 的重链条件化强度 |

> **配套产物**：论文值全表 `outputs/external/paper_reported_baselines.csv`（557 行，`in_scope` 列区分范围）；机器可读策略 `outputs/external/baseline_registry.json`；出处审计 `outputs/external/baseline_provenance_audit.json`；逐任务审计报告 [`audit_2026_08_29/`](audit_2026_08_29/)（T1/T2/T3/T4/AB-CDR/AB-pairing 各一份）。

## 4. 数据泄露防控（核心）
| 任务 | 防控 |
|------|------|
| T1 | unseen-epitope 与 seen-epitope 两套 split；CDR3β 去重且跨 split 互斥；负样本用 reference-TCR 随机配对 |
| T2 | 仅用测试表位的 TCR；表位标签只用于打分，不参与聚类 |
| T3 | 复用 T1 unseen split；冻结主干只训 probe |
| T4 | 用 OTS holdout；报告与训练集 CDR3 的新颖度/最近邻距离（**参考集有已知缺陷，见 §4.1**） |
| AB | CDR：SAbDab 用论文口径的 Kong 冻结快照（3,127 条）+ 官方 10 折切分脚本；SAb23H2 用官方 `IgGM_Test_set`。pairing：OAS holdout500，长度走与参考无关的 train 先验 |
| ⛔ P1 | 已冻结（§0）。原防控：直接用 STRING 90/90 split（两端蛋白互相 <90% 相似） |

### 4.1 ⚠️ T4 Setting-A 参考集构造缺陷（2026-08-28 发现，未修正打分）

`scripts/prepare_tcr_generation.py` 的 `--train-cap` 默认 **200,000**，
而 OTS train 有 **2,102,700** 行 —— 只覆盖 **9.5%**。这个参数**同时**决定两件事：
① novelty 参照集（"生成的序列算不算新"的判据）；② holdout 去重的比对对象。

| | cap=200k（现行线上文件） | cap=0（全量） |
|---|---:|---:|
| novelty 参照（唯一 core） | 180,918 | 1,718,935 |
| holdout 保留（唯一 core） | 9,767 | 7,874 |

即 **19.4% 的 holdout 序列其实躺在训练数据里**。逐条核验：多出的 1,893 条
全部（1,893/1,893）确认存在于 OTS train 全量中，无一误杀。

两个方向都在**虚高**分数：

- **novelty 虚高** —— 训练集第 200k–2.1M 行的任意序列，模型原样吐出来也算"新"
- **JSD / 分布匹配虚高** —— 目标分布里混了 19.4% 训练集内序列，
  模型越是记住训练集，反而越"匹配" held-out

修正版参考已生成，**未覆盖线上文件**：

```bash
python downstream/benchmark/scripts/prepare_tcr_generation.py \
    --train-cap 0 --out-dir downstream/benchmark/data/tcr_generation_fullref
```

验证：修正版 holdout ∩ OTS-train 全量 = **0**。

**现有 Setting A 的 novelty / JSD 数字是在旧参考（`data/tcr_generation/`）上算的。**
换用 `tcr_generation_fullref/` 会改变这些数值（预期 novelty 下降）。
要不要重跑、论文报哪一版，尚未决定 —— 引用 T4 Setting A 数字前请先确认口径。

发现路径：`scripts/data/dedup/audit_downstream_leakage.py`（训练侧去重审计的副产品）。
背景见 [`examples/llada/DATA_PIPELINE_README.md`](../../examples/llada/DATA_PIPELINE_README.md) §6。

## 5. 数据来源（本地）
| 任务 | 路径 |
|------|------|
| T1/T2/T3 | `data/tcr/`（VDJdb / McPAS / MIRA / IEDB / PIRD） |
| T4 | `data/ots_paired_clean/final`、`data/downstream/cdr_infilling/tcr` |
| AB CDR | `data/downstream/cdr_infilling/sabdab_kong`（Kong 3,127）；SAb23H2 eval harness = `data/downstream/cdr_infilling/sab23h2_converted/`（raw 仍是 `sab23h2/` 的 `IgGM_Test_set`） |
| AB pairing | `data/downstream/comp_chain/test_data_oas_holdout.csv` + `oas_train_light_length_prior.json` |
| ⛔ P1 | `data/ppi/string_model_org_90_90_split`（已冻结） |
| ⛔ A1 | `data/nanobody_raw/nbbench/hf_data`（已冻结） |

## 6. baseline 证据规则

> **原则（2026-09-02 论文值优先）**：每个已列 baseline 必须带自己的最佳已发表值作 `[P]` 主榜行；官方 artifact/代码复跑是 `[A]`/`[R]` **附行**，永不取代 `[P]`。协议不一致时同行披露协议差，不靠删行回避。`[L]`/`[C]` 必须与论文主表分开，不以“趋势接近”冒充复现。完整规则见 [`PROJGUIDE.md`](PROJGUIDE.md) §0.2.2。字段与判定见 `common/baseline_provenance.py` 和 `BASELINE_VERIFICATION.md`。

| 任务 | 外部 baseline / 可用性 |
|------|-----------------------------------------|
| Binding | NetTCR-2.2 `mnielLab/NetTCR-2.2`、ERGO-II `IdoSpringer/ERGO-II`、pMTnet `tianshilu/pMTnet`、epiTCR `ddiem-ri-4D/epiTCR`、PanPep `bm2-lab/PanPep` |
| Clustering | tcrdist3 `kmayerb/tcrdist3`、GIANA `s175573/GIANA`、clusTCR `svalkiers/clusTCR`、DeepTCR `sidhomj/DeepTCR` |
| Representation | TCR-BERT `wukevin/tcr-bert`、SCEPTR `yutanagano/sceptr`、catELMo `Lee-CBG/catELMo` |
| Generation · 无条件 setting (T4) | OLGA `statbiophys/OLGA`、soNNia `statbiophys/soNNia` |
| Generation · 表位条件 setting (T4) | TCRT5 sparse-13 artifact `[A]`；TcrDesign 可本地评分；TCR-epiDiff 因官方生成路径不完整仅引用，不宣称已复现 |
| AB CDR / pairing | Ophiuchus-Ab 官方 ckpt `[R]`（Zenodo 18478480）；AntiBERTy / AbLang2 / IgGM / dyMEAN / DiffAb / p-IgGen / LiChen 取论文值 `[P]` |
| ⛔ PPI | 已冻结（§0）。原清单：D-SCRIPT/Topsy-Turvy `samsledje/D-SCRIPT`、SENSE-PPI `AlbertMolina/SENSE_PPI`（`../MINT_MIGRATION.md`） |

> clone 到 `baselines/<name>` 时记录 commit/许可证；即使缺少可运行代码，论文报告值仍可作为 `[P]` 收录，但不得标成 `[R]`。

## 7. 目录结构（已实现）
```
benchmark/
├── README.md / RESULTS.md / PROGRESS.md
├── common/          # 已实现：schema, leakage, negatives, metrics(含 BLOSUM62 BR), model_api, featurizers
├── scripts/         # 已实现：prepare_tcr_binding / _repertoire / _generation / _ppi, import_immrep22_results, import_nbbench_results
├── tcr_binding/run.py      tcr_clustering/run.py   tcr_representation/run.py
├── tcr_generation/run.py   tcr_design/run.py       tcr_generation_bench/run.py
├── ppi/             # ⛔ 已冻结（§0）：STRING 90/90
├── nbbench/         # ⛔ 已冻结（§0）：纳米抗体 tasks.py + run.py + run_residue.py
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
3. **优先级**：当前范围 = T1–T4 + AB CDR/pairing（见 §0）。PPI 与 NbBench 虽已跑过但已冻结。“已覆盖”不等于所有本地值均 paper-comparable，具体以 provenance audit 为准。
4. **后续可扩展（仅限 AB + TCR 范围内）**：clone clustering/representation/generation 的官方深度 baseline（tcrdist3/GIANA/SCEPTR 等）、
   PIRD unseen-pHLA split、paired-αβ binding 子集、TCR-pMHC 结构 oracle。**不再向 PPI / NbBench / FLAb 方向扩展。**
