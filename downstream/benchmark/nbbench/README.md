# A1 — NbBench 抗体/纳米抗体维度

> 复用 NbBench 官方 train/val/test split + 冻结主干 + head-only probe 范式。
> **不涉及主干微调或基础模型训练**。

## 已接入（标量标签，``nbbench/run.py``）

| 任务 | 类型 | 主指标 |
|------|------|--------|
| nanobody_type | 多分类 | probe_acc |
| polyreaction | 二分类 | probe_auroc |
| thermo-tm | 回归 | spearman |
| thermo-seq | 回归 | spearman |
| vhh_affinity-score | 回归 | spearman |
| hTNFa / hIL6 / SARS-CoV-2 | 二分类(VHH+抗原) | probe_auroc |

## 待扩展（需专用 runner）

| 任务 | 原因 |
|------|------|
| VRClassification | 115 位点区域标注（非标量标签） |
| CDRInfilling | 生成/补全任务 |
| Paratope | 表位定位（序列对齐标注） |
| vhh_affinity-seq | 序列级亲和力分类（待核对标签格式） |

## 快速开始

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark
LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH

# 单任务
$ENV/bin/python nbbench/run.py --task nanobody_type --embedder esm2_150m

# 全部标量任务
$ENV/bin/python nbbench/run.py --task all --embedder esm2_150m
```

数据路径：`data/nanobody_raw/nbbench/hf_data/<task>/`。
