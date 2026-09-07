> ⚠️ **ARCHIVE — 范围已冻结（NbBench 不在评测范围内），仅历史/复现用。**
> 范围定义：[`../README.md`](../README.md) §0。不要给本文填新数字。归档于 2026-09-02。

# A1 — NbBench 抗体/纳米抗体维度

> 复用 NbBench 官方 train/val/test split + 冻结主干 + head-only probe 范式。
> **不涉及主干微调或基础模型训练**。
> **全部 12 个 NbBench 任务已跑通**（9 序列级标量 + 3 位点级/生成），结果与官方参照见 `../RESULTS.md` A1 节。

## 序列级标量任务（``nbbench/run.py``）

| 任务 | 类型 | 主指标 | ESM2-150M | k-mer |
|------|------|--------|-----------|-------|
| nanobody_type | 多分类 | probe_acc | 0.9997 | 0.9978 |
| polyreaction | 二分类 | probe_auroc | 0.865 | 0.869 |
| thermo-tm | 回归 | spearman | 0.658 | 0.836⚠ |
| thermo-seq | 回归 | spearman | 0.625 | 0.733⚠ |
| vhh_affinity-seq | 回归 | spearman | 0.195 | 0.113 |
| vhh_affinity-score | 回归 | spearman | 0.165 | 0.062 |
| hTNFa / hIL6 / SARS-CoV-2 | 二分类(VHH+抗原) | probe_auroc | 0.788 / 0.898 / 0.868 | 0.538 / 0.926 / 0.858 |

> ⚠ thermo 数据存在近似重复（~52%/32% 测试样本到训练集编辑距离≤2/较小），组成敏感的 k-mer 会虚高，须连同注解报告，见 `../RESULTS.md`。

## 位点级 / 生成任务（``nbbench/run_residue.py``）

| 任务 | 类型 | 主指标 | ESM2-150M | one-hot(±3) |
|------|------|--------|-----------|-------------|
| VRClassification | 逐残基 4 区多分类(FR/CDR1/2/3) | acc | 0.9989 | 0.8708 |
| Paratope | 逐残基二分类 | auprc（AUROC 0.923） | 0.682 | 0.449 |
| CDRInfilling | masked 位恢复 | BR（EM 辅） | 1.467（EM 0.396） | 1.124（EM 0.329） |

> - 位点级 runner 用冻结嵌入的 **逐残基表征**（`embed_residues`）+ 轻量 sklearn head，仍属 head-only probing。
> - **CDRInfilling 主指标 = BLOSUM62 Recovery(BR)**，对齐官方主榜（masked 位平均替换分，越大越好），同时报硬恢复 EM(`aar_masked`)。BR 由 Biopython BLOSUM62 计算，无 Biopython 时自动回退到 EM。
> - **`onehot` 训练free 对照**（滑窗 ±3 逐残基 one-hot，纯局部组成、无预训练上下文）是标量 `kmer` 的位点级类比；ESM2 三项均显著胜出，量化"预训练主干相对局部 motif 的增益"。用 `--embedder onehot` 或 `onehot:W` 指定窗宽。

## 快速开始

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

# 单任务（标量）
$ENV/bin/python nbbench/run.py --task nanobody_type --embedder esm2_150m
# 全部标量任务 + k-mer 训练free 对照
$ENV/bin/python nbbench/run.py --task all --embedder esm2_150m
$ENV/bin/python nbbench/run.py --task all --embedder kmer

# 位点级 / 生成任务 + 训练free one-hot 对照
$ENV/bin/python nbbench/run_residue.py --task all --embedder esm2_150m
$ENV/bin/python nbbench/run_residue.py --task all --embedder onehot

# 固化官方 NbBench 参照数字（outputs/external/nbbench_official.csv）
$ENV/bin/python scripts/import_nbbench_results.py

# 基础模型接入（训练完成后）
$ENV/bin/python nbbench/run.py --task all --embedder bioseq:/abs/path/final.pt
$ENV/bin/python nbbench/run_residue.py --task all --embedder bioseq:/abs/path/final.pt
```

数据路径：`data/nanobody_raw/nbbench/hf_data/<task>/`。
