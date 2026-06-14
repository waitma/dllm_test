# IRBench — 免疫受体基础模型下游评测基准

> 根目录: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`
>
> 为 `dllm_test` 的扩散式免疫受体基础模型 (BioSeq) 搭建统一、可复现、防数据泄露的下游评测，
> 覆盖 TCR 的 binding / clustering / representation / generation 与 PPI。baseline 仅用公开代码/权重。

## 1. 当前定位
- 免疫受体基础模型**尚未训练**。现阶段目标：**整合现有公开基准** + 跑公开 baseline + 预留模型接入接口。
- 模型训练完成后，只需 `build_embedder("bioseq:/path/final.pt")`，各任务 `--embedder bioseq:...` 即并入排行榜。
- **状态（2026-06-13）**：`common/` 框架 + 五任务（T1–T4 + P1）数据准备与运行脚本**全部实现并跑出真实可复现数字**，
  防泄露断言通过。结果见 `RESULTS.md`，过程见 `PROGRESS.md`。

### 快速开始
```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark
# GPU 任务前置（避免 libstdc++ 链接问题）：LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

# T1 binding（防泄露 seen/unseen）
$ENV/bin/python scripts/prepare_tcr_binding.py
$ENV/bin/python tcr_binding/run.py --method knn          # TCRdist 式距离
$ENV/bin/python tcr_binding/run.py --method embed --embedder esm2_150m
# T2 clustering + T3 representation（共享数据准备）
$ENV/bin/python scripts/prepare_tcr_repertoire.py
$ENV/bin/python tcr_clustering/run.py --method editdist --threshold 1
$ENV/bin/python tcr_representation/run.py --embedder esm2_150m
# T4 generation
$ENV/bin/python scripts/prepare_tcr_generation.py
$ENV/bin/python tcr_generation/run.py --method markov --order 3
# P1 PPI
$ENV/bin/python scripts/prepare_ppi.py
$ENV/bin/python ppi/run.py --embedder kmer

# 接入基础模型（训练完成后，对任一任务）：
$ENV/bin/python tcr_binding/run.py --method embed --embedder bioseq:/abs/path/final.pt
```

## 2. 整合的现有基准（优先复用，不重造）
| 维度 | 整合的现成基准 | 本地状态 |
|------|----------------|----------|
| 框架范式 + 抗体/纳米抗体 | **NbBench** (8 任务, 9 开源 baseline, head-only probing) | 已在 `data/nanobody_raw/nbbench`（数据+脚本齐全） |
| TCR 数据与防泄露 split | **TCRpMHCdataset (PIRD)** | 已在 `data/tcr/PIRD_alt` |
| TCR binding | **IMMREP23/22**（自带 seen/unseen split + 全长TCR + 官方评测/结果） | 已 clone 到 `baselines/` |
| PPI | **STRING 90/90**（Bernett 式 gold standard） | 已在 `data/ppi` |
| 抗体 binding/developability | 现有 `downstream/`（FLAb 等） | 已迁移 |

> 框架骨架直接复用 NbBench 的「模型 wrapper + 统一 train/val/test + head-only probing」，
> 把 TCR/PPI 任务接进同一范式，并新增 BioSeq/Ophiuchus 模型 wrapper。

## 3. 任务总览

> **TCR 四类精简方案**见 `TCR_TASK_TAXONOMY.md`（文献调研 2023–2026 → 贵精不贵多，每类 1 主任务）。

| # | 任务 | 整合基准 | 类型 | 主指标 |
|---|------|----------|------|--------|
| T1 | TCR Binding | **IMMREP23**（主）；文献变体：SCEPTR kNN | 二分类 | **unseen** macro-AUC0.1 / AUPRC |
| T2 | TCR Clustering | VDJdb + clusTCR/Valkiers 协议 | 无监督聚类 | Purity / Retention / NMI |
| T3 | TCR Representation | 共享 T2 split + SCEPTR 对比 | linear probe + kNN | probe-AUROC / kNN top-1 |
| T4 | TCR Generation | OTS 分布 + **CDR infill** | 采样/补全 | JSD↓ / AAR |
| P1 | PPI | STRING 90/90 | 二分类 | AUROC / AUPRC |
| A1 | 抗体/纳米抗体 | NbBench(8 任务) | 分类/回归 | 各任务自带 |

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

## 6. 公开 baseline
> **原则：只用作者 report 的官方代码 + 官方预训练权重，绝不自行复现；无公开代码的方法不收录。**

| 任务 | baseline（官方 repo，均含权重/可复现） |
|------|-----------------------------------------|
| Binding | NetTCR-2.2 `mnielLab/NetTCR-2.2`、ERGO-II `IdoSpringer/ERGO-II`、pMTnet `tianshilu/pMTnet`、epiTCR `ddiem-ri-4D/epiTCR`、PanPep `bm2-lab/PanPep` |
| Clustering | tcrdist3 `kmayerb/tcrdist3`、GIANA `s175573/GIANA`、clusTCR `svalkiers/clusTCR`、DeepTCR `sidhomj/DeepTCR` |
| Representation | TCR-BERT `wukevin/tcr-bert`、SCEPTR `yutanagano/sceptr`、catELMo `Lee-CBG/catELMo`；NbBench 自带 ESM2/ProtBERT/IgBERT/AntiBERTa2/AbLang/AntiBERTy/VHHBert/NanoBERT |
| Generation | OLGA `statbiophys/OLGA`、soNNia `statbiophys/soNNia` |
| PPI | D-SCRIPT/Topsy-Turvy `samsledje/D-SCRIPT`、SENSE-PPI `AlbertMolina/SENSE_PPI` |

> clone 到 `baselines/<name>` 记录 commit/许可证；统一调用接口可参考 `ePytope-TCR`。
> GLIPH2 等仅提供 web server、无可复现源码的工具暂不收录（如后续确认有官方 CLI 代码再补）。

## 7. 目录结构（已实现）
```
benchmark/
├── README.md / RESULTS.md / PROGRESS.md
├── common/          # 已实现：schema, leakage, negatives, metrics, model_api, featurizers
├── scripts/         # 已实现：prepare_tcr_binding / _repertoire / _generation / _ppi, import_immrep22_results
├── tcr_binding/run.py      tcr_clustering/run.py   tcr_representation/run.py
├── tcr_generation/run.py   ppi/run.py
├── nbbench/         # A1 抗体/纳米抗体：tasks.py + run.py（冻结主干 + head-only probe）
├── baselines/wrappers/  # 官方 baseline 薄封装（epiTCR 等）
├── baselines/       # 公开 baseline（git clone，本体不入库）
├── data/            # 各任务生成的数据集 + leakage_report.json（gitignore）
└── outputs/         # 预测文件 + metrics.json + external/immrep22_official.csv（gitignore）
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
3. **优先级**：已先完成 T1(binding) 与全部五任务族；NbBench 抗体维度与更多 baseline 作为后续扩展。
4. **后续可扩展**：clone clustering/representation/generation/PPI 的官方深度 baseline（tcrdist3/GIANA/SCEPTR/D-SCRIPT 等）、
   接入 NbBench 8 任务、PIRD unseen-pHLA split、paired-αβ binding 子集、TCR-pMHC 结构 oracle。
