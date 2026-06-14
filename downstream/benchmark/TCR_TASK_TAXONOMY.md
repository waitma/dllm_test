# TCR 下游任务精简方案（文献调研 → 四类嵌合）

> 原则：**贵精不贵多**。每个大类只保留 1 个主任务 + 最多 1 个辅任务；其余文献任务作为指标变体或后续扩展，不单独开榜。
>
> 四类：T1 Binding · T2 Clustering · T3 Representation · T4 Generation

---

## 文献里常见什么（2023–2026）

| 论文/基准 | 年份 | 核心评测 | 对我们四类的启示 |
|-----------|------|----------|------------------|
| **IMMREP23** (Nielsen et al.) | 2024 | TCR–pMHC 二分类；seen vs **unseen epitope**；强调数据泄露 | **T1 金标准**；主排名必须用 unseen |
| **Nature Methods 大基准** (50 模型) | 2025 | AUPRC 主指标；独立测试集；负样本来源；paired αβ+MHC | T1 辅指标用 AUPRC/AUC0.1；列配置支持 αβ+peptide(+MHC) |
| **ePytope-TCR** (Drost et al.) | 2025 | 21 个预测器统一接口；低频表位全面失败 | T1 只接一个主数据集即可，不堆多个相似 binding 榜 |
| **SCEPTR** (Yermanos et al.) | 2024 | 嵌入质量 + **per-epitope NN binding**；对比 TCRdist/ESM2 | **T3 主任务** + T1 上可作为 few-shot NN 子协议 |
| **Clustering 对比** (Valkiers et al. 2024) | 2024 | Purity / Retention / Consistency；ClusTCR·tcrdist3·GLIPH2 | **T2 指标协议**已对齐 |
| **DecoderTCR** (Lai et al.) | 2026 | 零样本 binding AUROC + **表位特异性 TCR 识别** + 条件生成 | T1 零样本分、T3 池内排序、T4 条件 infill 的文献依据 |
| **IMMREP22** | 2023 | 17 表位 MicroAUC | 仅作**外部参考行**，不与 IRBench-T1 混排 |

**共识（三篇 binding 综述的一致结论）**
1. Seen 表位上方法尚可（AUC 0.6–0.8）；**unseen 表位普遍接近随机** → 主榜必须报 unseen。
2. CDR3β-only 不够；**paired αβ + peptide (+MHC)** 是论文标配特征粒度。
3. 负样本构造与克隆型泄露对分数影响大于模型结构 → 我们的 leakage 断言是核心卖点，不能省。

---

## 嵌合到四类：保留 vs 砍掉

### T1 — Binding（结合预测）

| 状态 | 任务 | 数据 | 主指标 | 文献依据 |
|------|------|------|--------|----------|
| **保留·主榜** | IMMREP23 unseen-epitope | 已落地 `data/tcr_binding/` | macro-AUROC / **macro-AUC0.1** / AUPRC | IMMREP23, ePytope-TCR, Nat Methods 2025 |
| **保留·辅报** | IMMREP23 seen-epitope | 同上 | 同上 | 对照「记忆表位」能力 |
| **保留·协议变体** | Per-epitope kNN binding（SCEPTR 式） | 复用 T1 数据 | per-epitope AUROC 均值 | SCEPTR Fig.1 |
| **砍掉** | IMMREP22 / TetTCR-SeqHD / Fingerprinting 独立榜 | — | — | 与 IMMREP23 同质，仅外部参考 |
| **砍掉** | PIRD unseen-pHLA 单独任务 | — | — | 并入 T1 的 `--columns` 扩展，不新开任务 |
| **延后** | 零样本 binding（DecoderTCR 协议） | 需无训练重叠 split | AUROC | 等 BioSeq 有无监督/零样本接口再接 |

**T1 最终形态（精）**
```
主排名 = unseen-epitope macro-AUC0.1
辅报   = seen-epitope + macro-AUPRC
特征   = CDR3β + CDR3α + peptide（默认）；+MHC 为 ablation 列
基线   = CDR3-kNN（训练free）+ 官方 NetTCR/ERGO/epiTCR（wrapper）+ ESM2 probe
```

---

### T2 — Clustering（ repertoire 聚类 → 特异性推断）

| 状态 | 任务 | 数据 | 主指标 | 文献依据 |
|------|------|------|--------|----------|
| **保留·主榜** | 表位标注 repertoire 聚类 | VDJdb ≥50 TCR/表位（已落地） | **Purity** + Retention + NMI | ClusTCR 2021; Valkiers 2024 |
| **保留·基线** | CDR3 edit-distance (t=1) | 同上 | 同上 | clusTCR 式强基线 |
| **延后·基线** | tcrdist3 / GIANA 官方实现 | clone 后 wrapper | 同上 | 文献标配，但不新开任务 |
| **砍掉** | GLIPH2 | 仅 web server | — | 无法复现，不收录 |
| **砍掉** | scRNA+TCR 联合聚类（TESSA/TCRclub） | 需转录组 | — | 超出序列基础模型范围 |

**T2 最终形态（精）**
```
单任务：无监督聚类 → 与表位标签比对
不增加「多数据集聚类榜」；换数据只改 prepare 脚本，不增任务行
```

---

### T3 — Representation（表征质量）

| 状态 | 任务 | 数据 | 主指标 | 文献依据 |
|------|------|------|--------|----------|
| **保留·主榜** | 克隆型隔离的 **epitope linear probe** | 与 T2 共享 24 表位 split | macro probe-AUROC + Acc | NbBench 范式; SCEPTR 对比 PLM |
| **保留·辅报** | **kNN top-1** 表位检索 | 同上 | kNN top-1 Acc | SCEPTR nearest-neighbour |
| **砍掉** | 单独「CDR3 二分类」等小任务 | — | — | 被 T1/T3 覆盖 |
| **延后** | 池内 TCR 识别（DecoderTCR: 给定表位从库中找 binder） | 需结合子集构造 | AUROC | 与 T1 kNN 协议合并为一个「retrieval」子指标 |

**T3 最终形态（精）**
```
主任务 = 冻结嵌入 + 线性 probe（24-way）
辅指标 = kNN top-1
特征列 = CDR3β + CDR3α（与文献一致）
```

---

### T4 — Generation（生成/补全）

| 状态 | 任务 | 数据 | 主指标 | 文献依据 |
|------|------|------|--------|----------|
| **保留·主榜** | CDR3β **无条件**生成（库分布） | OTS holdout（已落地） | k-mer JSD↓ + novelty + NN distance | OLGA/soNNia 式 repertoire 生成 |
| **保留·次榜** | CDR3 **条件 infill**（masked 恢复） | `data/downstream/cdr_infilling/tcr` 10-fold | AAR（氨基酸恢复率） | 抗体 CDR infill 范式迁移到 TCR |
| **砍掉** | 全链 αβ 配对生成 | 数据/评测不成熟 | — | 延后 |
| **砍掉** | 实验验证型 design（DecoderTCR wet-lab） | 超出计算基准 | — | 不纳入 IRBench |
| **延后** | 表位条件 TCR 设计 | 需 epitope-conditioned 采样接口 | 结合子集 enrichment | DecoderTCR 生成分支 |

**T4 最终形态（精）**
```
主任务 = 无条件 CDR3β 分布匹配（JSD）
次任务 = CDR infill AAR（一条脚本，不拆第三个大类）
```

---

## 与当前 IRBench 的差距（待补，仍不涉及训练）

| 优先级 | 动作 | 对应类 |
|--------|------|--------|
| P0 | T1 官方 baseline wrapper（NetTCR/ERGO/epiTCR）出数 | T1 |
| P0 | T4 CDR infill 脚本接入 `tcr_generation/run.py --method infill` | T4 |
| P1 | T1 增加 SCEPTR 式 per-epitope kNN 子指标 | T1/T3 |
| P1 | T2 clone tcrdist3 作官方聚类基线 | T2 |
| P2 | T1 `--columns +mhc` ablation 行 | T1 |
| 不做 | 新增第 5/6/7 个 TCR 任务榜 | — |

---

## 一张表总览（投稿用）

| 类 | 唯一主任务 | 主指标 | 防泄露要点 |
|----|-----------|--------|------------|
| **T1 Binding** | IMMREP23 paired-chain | **unseen macro-AUC0.1** | 克隆型去重 + unseen 表位 |
| **T2 Clustering** | VDJdb 表位 repertoire | **Purity** (+ Retention) | 标签不参与聚类 |
| **T3 Representation** | 24-way epitope probe | **probe-AUROC** | 克隆型隔离 split |
| **T4 Generation** | OTS CDR3β + infill | **JSD** / **AAR** | holdout 新颖度 |

> PPI（P1）与抗体（A1 NbBench）保持独立维度，不并入 TCR 四类。
