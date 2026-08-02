# FLAb 抗体属性回归（MINT Figure 3b 口径）

在 [Graylab/FLAb](https://github.com/Graylab/FLAb) 四个公开数据集上，用 frozen backbone 嵌入 + Ridge 回归预测抗体 fitness（结合亲和力 / 表达）。

**论文主指标 `[P]`**：R²，nested 10 outer × 5 inner CV。

**本地诊断 `[L]`**：历史结果是 Spearman（5-fold pooled OOF），不能替代或混入论文主表。paper-aligned runner 已修为 fold 内 transformer + nested Ridge，重跑 pending。
**结果汇总**：`downstream.md` AB-FLAB 段 · `benchmark/RESULTS.md` · `benchmark/outputs/external/paper_reported_baselines.csv`。

### 论文结果速览 `[P]`（R² mean±std ↑）

| Model `[P]` | trastuzumab | d44 | g6 Kd | g6 ER |
|-------|:-----:|:-----:|:-----------:|:------:|
| AbLang | 0.293±0.117 | 0.246±0.038 | 0.244±0.034 | 0.439±0.027 |
| AntiBERTy | 0.239±0.102 | 0.217±0.056 | 0.199±0.025 | 0.401±0.032 |
| IgBert | 0.306±0.114 | 0.131±0.047 | 0.174±0.032 | 0.400±0.023 |
| IgT5 | 0.274±0.070 | 0.297±0.057 | 0.179±0.014 | 0.548±0.067 |
| MINT | **0.398±0.078** | **0.379±0.058** | **0.253±0.041** | **0.657±0.023** |

历史 32/32 的 Spearman sweep 保留在 `output/downstream_generation/flab_baselines/`，只作 `[L] diagnostic`，不用于论文排名。

---

## 数据

2026-07-07 起，数据从 GitHub 官方仓库重新下载（原 `datasets/` symlink 已失效）：

| 文件 | 任务 | n |
|------|------|---|
| `data/downstream/flab/flab_raw/koenig2017mutational_kd_g6.csv` | 结合 (Kd) | 1836 |
| `data/downstream/flab/flab_raw/koenig2017mutational_er_g6.csv` | 表达 (ER) | 4275 |
| `data/downstream/flab/flab_raw/shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv` | 结合 (Kd) | 422 |
| `data/downstream/flab/flab_raw/warszawski2019_d44_Kd.csv` | 结合 (Kd) | 2048 |

列：`heavy`, `light`, `fitness`（及原始 Kd/ER 列）。所有 CSV 在 `flab_raw/` 下也有同名 symlink 便于脚本引用。

---

## 脚本

| 脚本 | 用途 |
|------|------|
| `run_flab_baselines.py` | 单 (dataset, embedder) 跑通：嵌入 + nested 10×5 Ridge CV → `metrics.json` |
| `run_flab_all.sh` | 四数据集 × 全部 embedder 批量 sweep |
| `summarize_ab_baselines.py` | 聚合 `flab_baselines/*.json` 为 markdown 表 + `_summary.csv` |
| `finetune_flab.py` | **Legacy**：仅 Ophiuchus-Ab checkpoint 嵌入（AirGen 迁移版） |

### 支持的 embedder（`--embedder`）

| spec | 说明 |
|------|------|
| `onehot` | Descriptor tier：逐残基 one-hot |
| `esm2_650m` | ESM-2 650M（本地 `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`） |
| `ophiuchus` | Ophiuchus-Ab 官方 ckpt（joint paired embed，`embed_pairs`） |
| `antiberty` | AntiBERTy（`antiberty` 包内置权重） |
| `igbert` | IgBERT（HuggingFace `Exscientia/IgBert`） |
| `protbert` | ProtBERT（HuggingFace `Rostlab/prot_bert`） |
| `bioseq-llada:/abs/best.pt` | **我们的** grammar_v2 post-LLaDA global mean-pool |

所有非 one-hot 模型经 `benchmark/common/model_api.py::build_embedder` 加载。

---

## 复现

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
export HF_ENDPOINT=https://hf-mirror.com
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test

# 单条：Ophiuchus-Ab × trastuzumab
$ENV/bin/python downstream/flab/run_flab_baselines.py \
  --dataset data/downstream/flab/flab_raw/shanehsazzadeh2023unlocking_zerokd_trastuzumab.csv \
  --embedder ophiuchus --device cuda \
  --output output/downstream_generation/flab_baselines/trastuzumab_kd__ophiuchus.json

# 全量 sweep（跳过已有 json）
bash downstream/flab/run_flab_all.sh

# 汇总表
$ENV/bin/python downstream/flab/summarize_ab_baselines.py
```

---

## 已知 blocker

| 项目 | 原因 |
|------|------|
| paper-aligned nested-R² 全量重跑 | runner 已修；尚未生成可与 Figure 3b 对照的本地结果 |
| AbLang2 FLAb embedder 行 | 权重已下载（CDR 已跑通），但 FLAb runner 尚未加 ablang2 embedder（可选补） |

> g6_er × onehot 已跑完（Spearman 0.602，~21 min）；不再列为 blocker。AbLang2 权重已从正确 Zenodo 地址下载并跑通 CDR baseline。

Legacy `finetune_flab.py` 仍可用于 Ophiuchus-Ab 单模型评测；新 baseline 对比请用 `run_flab_baselines.py`。
