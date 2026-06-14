# IRBench 结果排行榜 (Leaderboard)

> 持续更新。每个任务一张表；空缺表示尚未运行。
> `Ours-BioSeq` 列在免疫受体基础模型训练完成后填充；当前以公开 baseline + 通用蛋白 LM(ESM2) 作参考。
> **所有数字可由 `outputs/` 下的预测文件复现**，并标注 split / 数据版本 / 负采样比 / 随机种子。
>
> 复现命令见每个表下方。数据版本：IMMREP23 paired-chain VDJdb (commit 06d85be)，neg ratio 5:1，seed 0。

---

## T1 — TCR-Epitope Binding

**数据集（IRBench 主基准）**：IMMREP23 paired-chain VDJdb 正样本 + 自生成参考负样本(Lev>3, 5:1) 作训练；
官方 `solutions.csv`(含负样本) 作测试。**防泄露**：训练集中与测试克隆型(CDR3a|CDR3b)重叠的 60 条正样本已剔除，
最终 clonotype/pair overlap = 0（见 `data/tcr_binding/leakage_report.json`）。
测试按表位是否在训练集出现分为 **seen(13 表位/2418 行)** 与 **unseen(7 表位/1066 行)**。
主排名 = **unseen-epitope**（泛化到新表位，最难也最有意义）。

### Unseen-epitope split (主排名)
| Method | macro-AUROC | macro-AUPRC | macro-AUC0.1 | 类型 | 备注 |
|--------|-------------|-------------|--------------|------|------|
| CDR3-kNN (TCRdist式) | 0.500 | 0.167 | 0.500 | 距离/训练free | 无参考结合子→必然随机 |
| ESM2-150M + linear probe | 0.470 | 0.188 | 0.495 | 通用蛋白LM | 均值池化拼接，线性不建模交互 |
| Random | 0.473 | 0.192 | 0.504 | floor | |
| **Ours-BioSeq** | _pending_ | | | 扩散免疫基础模型 | 训练完成后填 |

### Seen-epitope split
| Method | macro-AUROC | macro-AUPRC | macro-AUC0.1 | 备注 |
|--------|-------------|-------------|--------------|------|
| CDR3-kNN (TCRdist式) | **0.694** | **0.539** | **0.689** | 有同表位参考结合子 |
| Random | 0.512 | 0.207 | 0.511 | |
| ESM2-150M + linear probe | 0.489 | 0.191 | 0.494 | 线性 probe 不足以建模 binding |
| **Ours-BioSeq** | _pending_ | | | |

> **关键结论**：CDR3-kNN 在 seen 表位有效(AUROC 0.69)，但在 unseen 表位**恰好随机(0.50)**——
> 这正是泄露受控的 unseen-split 的价值：它如实暴露了距离类方法无法泛化到新表位。
> 通用 ESM2 均值嵌入 + 线性 probe 在 binding 上不足（接近随机），说明 binding 需要建模 TCR×表位交互
> 或专门预训练——这是免疫受体基础模型的动机。
>
> 复现：
> ```bash
> python scripts/prepare_tcr_binding.py --neg-ratio 5 --seed 0
> python tcr_binding/run.py --method random
> python tcr_binding/run.py --method knn --knn-k 5
> python tcr_binding/run.py --method embed --embedder esm2_150m
> ```

### 外部参考：IMMREP22 官方排行榜（不同数据集，仅作定位）
> 来源：`baselines/IMMREP_2022_TCRSpecificity/methods_results/`（17 个表位，指标 MicroAUC / Average Rank）。
> **注意**：这是 IMMREP22 数据集上的官方结果，**与上方 IRBench-T1(IMMREP23) 不可直接比较**，仅用于展示传统方法量级。

| Method (官方) | avg MicroAUC | avg Rank |
|---------------|--------------|----------|
| tcrexab | 0.847 | 3.28 |
| TCRGP | 0.847 | 3.49 |
| tcrdist3 | 0.840 | 3.58 |
| netTCR_cdr123ab | 0.825 | 3.82 |
| TCRAI | 0.823 | — |
| sonia_paired | 0.806 | 4.54 |
| pMTnet | 0.774 | — |
| TCR-BERT | 0.754 | — |
| TITAN | 0.746 | 9.12 |
| SETE | 0.739 | — |
| random | 0.490 | 9.12 |

> 完整 22 行见 `outputs/external/immrep22_official.csv`（由 `scripts/import_immrep22_results.py` 生成）。

---

## T2 — TCR Clustering
> 数据：VDJdb 正样本中每表位 >=50 条 TCR 的 24 个表位 / 8609 条（`data/tcr_clustering/tcrs.csv`）。
> **防泄露**：表位标签**只用于评测**，不参与聚类。指标 Purity / Retention / NMI / ARI。

| Method | Purity | Retention | NMI | ARI | #clusters | 备注 |
|--------|--------|-----------|-----|-----|-----------|------|
| CDR3-editdist (t=1, clusTCR式) | **0.800** | 0.431 | **0.475** | **0.310** | 549 | 紧致高纯度，部分保留 |
| ESM2-150M agglomerative (24簇) | 0.273 | 1.000 | 0.091 | 0.026 | 24 | 通用嵌入聚类区分度低 |
| **Ours-BioSeq** | _pending_ | | | | | |

> 结论：编辑距离聚类(Purity 0.80)远胜通用 ESM2 嵌入聚类(0.27)——通用 PLM 嵌入不擅长按表位特异性分组 TCR。
> 复现：`python scripts/prepare_tcr_repertoire.py` → `python tcr_clustering/run.py --method editdist --threshold 1` / `--method embed --embedder esm2_150m`

## T3 — TCR Representation (linear probe / kNN)
> 任务：24-way 表位特异性分类（冻结主干 + 线性 probe / 1-NN）。**防泄露**：克隆型隔离 split（CDR3a|CDR3b 不跨 train/test，overlap=0）。
> 随机基线 acc≈1/24≈0.04。特征列 = CDR3β+CDR3α。

| Method | probe-AUROC(macro-OVR) | probe-Acc | kNN top-1 | 备注 |
|--------|------------------------|-----------|-----------|------|
| ESM2-150M | **0.806** | **0.438** | 0.409 | 通用蛋白 LM 嵌入 |
| Ophiuchus-Ab | **0.825** | **0.452** | 0.443 | 抗体权重作 TCR 参考嵌入 |
| k-mer(3) 组成 | 0.800 | 0.430 | **0.447** | 训练free 经典基线 |
| **Ours-BioSeq** | _pending_ | | | |

> 结论：ESM2 与 k-mer 组成在 24-way 表位分类上量级相近(AUROC≈0.80)，均远超随机——为免疫基础模型留出明确的提升空间。
> 复现：`python tcr_representation/run.py --embedder esm2_150m` / `--embedder kmer`

## T4 — TCR Generation (CDR3β, 无条件采样)
> 数据：OTS paired-clean CDR3β，train(抽样 200k) 建模 + 新颖度参考，holdout(9837, 与 train 去重) 作分布参考。
> 指标：novelty(不在 train 中的比例)↑、mean NN edit distance(到最近 train)、k-mer(3) JSD(对 holdout 分布)↓、unique 比例↑。

| Method | novelty | NN距离 | k-mer JSD↓ | unique | 备注 |
|--------|---------|--------|------------|--------|------|
| Markov(order=3) | 0.971 | 2.68 | **0.0413** | 4679/5000 | 局部依赖最优分布匹配 |
| Markov(order=2) | 0.987 | 2.88 | 0.0421 | 4308/5000 | |
| PWM(per-length) | 0.999 | 3.87 | 0.1759 | 5000/5000 | 位置独立→分布偏差大 |
| **Ours-BioSeq (infill)** | _pending_ | | | | 训练后可用 `--method file` 接入采样 |

> 结论：捕捉残基依赖的 Markov(3) 分布最接近真实库(JSD 0.041)，位置独立 PWM 最差(0.176)；三者新颖度均 >0.97。
> 复现：`python scripts/prepare_tcr_generation.py` → `python tcr_generation/run.py --method markov --order 3`（/ `--order 2` / `--method pwm`）

## P1 — PPI (STRING 90/90)
> 数据：STRING model-org 90/90 正样本 + 同 split 内平衡随机负样本(1:1)。**防泄露**：train/test 蛋白重叠=0（90/90 序列相似性划分，见 `data/ppi/leakage_report.json`）。

| Method | AUROC | AUPRC | n_test | 备注 |
|--------|-------|-------|--------|------|
| k-mer(3) + Hadamard + LR | 0.563 | 0.606 | 2644 | 训练free 组成特征 |
| **Ours-BioSeq** | _pending_ | | | |

> 结论：在严格 90/90 防泄露划分下，朴素 k-mer PPI 仅 AUROC 0.56——正是 Bernett 式基准的意义：消除序列相似性泄露后任务显著变难。
> 复现：`python scripts/prepare_ppi.py` → `python ppi/run.py --embedder kmer`

---

## A1 — 抗体/纳米抗体（NbBench，冻结主干 + 线性 probe）

> 数据：`data/nanobody_raw/nbbench/hf_data/<task>/` 官方 train/test split。
> 方法：ESM2-150M 均值池化嵌入 + 线性 probe（**无主干微调**）。
> 复现：`python nbbench/run.py --task <name> --embedder esm2_150m` 或 `--task all`。

| 任务 | 类型 | ESM2-150M | 主指标 | 备注 |
|------|------|-----------|--------|------|
| nanobody_type | 多分类 | probe_acc **0.999** | probe_acc | 3 类纳米抗体类型 |
| polyreaction | 二分类 | probe_auroc **0.865** | probe_auroc | n_train=101854 |
| thermo-tm | 回归 | spearman **0.658** | spearman | 熔解温度 |
| thermo-seq | 回归 | spearman **0.625** | spearman | 热稳定性序列 |
| vhh_affinity-score | 回归 | spearman 0.165 | spearman | 亲和力连续分数（难） |
| hTNFa | 二分类 | probe_auroc **0.779** | probe_auroc | VHH+抗原分列嵌入 |
| hIL6 | 二分类 | probe_auroc **0.896** | probe_auroc | n_test=449234（官方 split） |
| SARS-CoV-2 | 二分类 | probe_auroc **0.867** | probe_auroc | VHH+抗原分列嵌入 |
| **Ours-BioSeq** | — | _pending_ | | 训练完成后填 |

> 待扩展：VRClassification（115 位点标注）、CDRInfilling、Paratope 需专用 runner。
