# IRBench 结果排行榜 (Leaderboard)

> 持续更新。每个任务一张表；空缺表示尚未运行。
> `Ours-BioSeq` 列在免疫受体基础模型训练完成后填充；当前以公开 baseline + 通用蛋白 LM(ESM2) 作参考。
> 本地运行数字可由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/` 下的预测文件复算；论文 fallback 数字没有本地 predictions，必须标注“论文值，未本地复现”并给出表格位置。所有行同时标注 split / 数据版本 / 负采样比 / 随机种子或论文来源。
>
> 复现命令见每个表下方。数据版本：IMMREP23 paired-chain VDJdb (commit 06d85be)，neg ratio 5:1，seed 0。
>
> **呈现规约**：所有对比表 / 图中，**我们的模型（Ours-BioSeq）一律排在 baseline 之后** —— 行式对比 = 表末的 Ours 行块、列式 = 最右列、图 = 图例末项；**即使分数领先也不与 baseline 穿插排序**（靠加粗 / 高亮强调）。详见 `PROJGUIDE.md §2.4`。
>
> **Baseline 来源规约（2026-07-21）**：`[P]`=论文报告值，`[A]`=官方 artifact 重评分，`[R]`=官方代码本地重跑，`[L]`=本地重实现，`[C]`=本地控制。协议不一致时 `[P]` 是主 baseline，`[L]/[C]` 只作独立 diagnostic。T1 进一步采用 original-only 决策：官方 checkpoint+code+`original.zip` 完整可运行时取 `[R]`，否则取 `[P]` 并标“论文值，未本地复现”。来源表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/external/paper_reported_baselines.csv`；审计：同目录 `baseline_provenance_audit.json`。旧 MINT 回归 Pearson/RMSE、旧 FLAb Spearman 与 NbBench sklearn probe 均不属于“论文可比”层。

---

## T1 — TCR-Epitope Binding

### T1 · Nature Methods 2025 官方 split（主协议）

> **文献锚定**：主基础 Nature Methods 2025《Assessment of computational methods in predicting TCR–epitope binding recognition》([s41592-025-02910-0](https://www.nature.com/articles/s41592-025-02910-0)，官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark` commit `ec832b47`)；近似去重/低-FPR 补强 arXiv 2606.04994 (2026)。**主指标 = overall（all_values）AUPRC**；macro-AUPRC / AUROC / macro-AUC0.1 辅报。
> **数据边界**：external original-model baseline 只使用 figshare `original.zip`（seen 3 表位 / unseen 40 表位；主 neg=AS）派生 test，不读取 GitHub `train.csv`、不 retrain。`train.csv`（cdr3b 82,065 行）只供 Ours frozen-head、kNN 与 L1 sensitivity 等本地诊断，不属于 external baseline 协议。
> **我们的模型（headline = post-LLaDA globalfeat）**：冻结 grammar_v2-LLaDA decoder 末层 → **完整** `tcr_peptide` record → **全局 mean-pool 单向量** → 官方 train 上 **MLP head**（`run_nm2025.py --method embed --embedder grammar:decoder:global: --head mlp`，符合 PROJ_GUIDE.md post-LLaDA 强制口径）；3 checkpoint（esmc300m-cmp500k / esmc600m-cmp500k / esmc300m-mint），tag `ours_globalfeat_*`。encoder-only（`grammar:encoder:`）与 segment-concat（`ours_postllada_*`）仅作 control。
> **baseline**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv` 是 canonical 来源表：完整可用时取官方 original checkpoint + 官方推理代码 + `original.zip` 的 `[R]`；不可运行时取 Supplementary Table 4 `[P]` 并标 **“论文值，未本地复现”**。当前 13 个模型为 `[R]`，SETE/TEPCAM 为 `[P]` fallback。同目录 `summary_local_runs.csv` 与 `baseline_paper_audit.csv` 仅作运行/差异诊断。kNN / Random 为 `[C]` 控制。

#### cdr3b · AS · overall AUPRC（主排名，随机≈0.5；Ours = post-LLaDA globalfeat headline）

| Method | Seen AUPRC↑ | Unseen AUPRC↑ | Seen AUROC | Unseen AUROC | Seen AUC0.1 | Unseen AUC0.1 | 类型 |
|--------|:-----------:|:-------------:|:----------:|:------------:|:---:|:---:|------|
| **ATM-TCR** | **0.6959** | 0.5223 | **0.6480** | 0.5218 | 0.663 | 0.674 | [R] original-only |
| **TEIM** | 0.6758 | 0.5182 | 0.6025 | 0.5161 | 0.676 | 0.622 | [R] original-only |
| **TEPCAM** | 0.6693 | 0.5128 | 0.5988 | 0.5120 | — | — | [P] 论文值，未本地复现 |
| **NetTCR** | 0.6114 | 0.5048 | 0.5903 | 0.4971 | 0.569 | 0.642 | [R] original-only |
| **SETE** | 0.5629 | — | 0.5463 | — | — | — | [P] 论文值，未本地复现 |
| epiTCR | 0.5572 | 0.4824 | 0.5357 | 0.4822 | 0.517 | 0.626 | [R] original-only |
| PanPep | 0.4987 | 0.4791 | 0.5065 | 0.4826 | 0.518 | 0.669 | [R] original-only |
| CDR3-kNN (k=5) | 0.553 | 0.544 | 0.507 | 0.504 | 0.551 | 0.511 | 训练free |
| Random | 0.512 | 0.503 | 0.517 | 0.520 | 0.496 | 0.694 | floor |
| **Ours-BioSeq** esmc300m-mint | 0.596 | 0.501 | 0.574 | 0.529 | 0.578 | 0.696 | [L] LLaDA-dec globalfeat+MLP |
| **Ours-BioSeq** esmc300m-cmp500k | 0.583 | 0.507 | 0.552 | 0.502 | 0.585 | 0.712 | [L] LLaDA-dec globalfeat+MLP |
| **Ours-BioSeq** esmc600m-cmp500k | 0.576 | 0.496 | 0.543 | 0.494 | 0.593 | 0.624 | [L] LLaDA-dec globalfeat+MLP |

#### 口径对照：headline globalfeat vs 两组 control（Ours-BioSeq · cdr3b · AS · 消融对照）

| Checkpoint | 口径 | Seen AUPRC | Unseen AUPRC | 状态 |
|------------|------|:----------:|:------------:|------|
| esmc300m-cmp500k | **post-LLaDA globalfeat（headline，`grammar:decoder:global:`）** | 0.583 | 0.507 | ✅ |
| esmc600m-cmp500k | **post-LLaDA globalfeat（headline）** | 0.576 | 0.496 | ✅ |
| esmc300m-mint | **post-LLaDA globalfeat（headline）** | 0.596 | 0.501 | ✅ |
| esmc300m-cmp500k | post-LLaDA segment-concat（control-A，`ours_postllada_*`） | 0.618 | 0.514 | ✅ |
| esmc600m-cmp500k | post-LLaDA segment-concat（control-A） | 0.538 | 0.515 | ✅ |
| esmc300m-mint | post-LLaDA segment-concat（control-A） | 0.494 | 0.517 | ✅ |
| esmc300m-cmp500k | encoder-only（control-B，`grammar:encoder:`） | 0.564 | 0.534 | ✅ |
| esmc600m-cmp500k | encoder-only（control-B） | 0.579 | 0.511 | ✅ |
| esmc300m-mint | encoder-only（control-B） | 0.569 | 0.520 | ✅ |

> **诚实结论**：globalfeat 是 PROJ_GUIDE.md 规定的 headline 口径，但其 unseen AUPRC（0.496–0.507）实际**略低于** encoder-only control（0.511–0.534）与 segment-concat control（0.514–0.517）——三口径 unseen 全部落在近随机带（0.49–0.53），post-LLaDA 相比 encoder-only 在 T1 unseen 上**未见提升**（如实标注）；seen 上 globalfeat（0.58–0.60）与其他口径相当。源：`outputs/tcr_binding_nm2025/{ours_globalfeat_*,ours_postllada_*,ours_biseq_mlp*}` + `summary_main.csv`。

> **caveat**：① macro-AUPRC / macro-AUC0.1 unseen 40 表位、~17 行/表位方差大，不作主排名（如 Random unseen AUC0.1=0.694 属小样本假象）。② **ERGO-II** 原生用 CDR3β+α+V/J+MHC，本 track 仅喂 CDR3β（其余 UNK，同论文 partial-feature 口径），seen 近随机属预期。③ 汇总 `outputs/tcr_binding_nm2025/summary_main.csv`。

#### Original-model baseline 扩充：TEINet / ERGO 变体 / TPBTE（cdr3b · AS · overall AUPRC）

> 下表均已通过官方 original checkpoint + 官方推理代码 + `original.zip` 本地重跑，故选择 `[R]`；论文数值与 delta 仍保存在 audit 表中。

| Method | Seen AUPRC↑ | Unseen AUPRC↑ | Seen AUROC | Unseen AUROC | 类型 |
|--------|:-----------:|:-------------:|:----------:|:------------:|------|
| TEINet-large | 0.6665 | 0.5047 | 0.6006 | 0.5023 | [R] original-only |
| TEINet-small | 0.6561 | 0.5011 | 0.5923 | 0.5053 | [R] original-only |
| ERGO-lstm (vdj) | 0.6281 | 0.5191 | 0.5770 | 0.5116 | [R] original-only |
| ERGO-AE (vdj) | 0.6152 | 0.5066 | 0.5428 | 0.4994 | [R] original-only |
| ERGO-lstm (mc) | 0.6094 | 0.5089 | 0.5614 | 0.5108 | [R] original-only |
| ERGO-AE (mc) | 0.5801 | 0.5232 | 0.5152 | 0.5051 | [R] original-only |
| TPBTE (mc) | 0.5009 | 0.5009 | 0.5055 | 0.5037 | [R] original-only |
| TPBTE (vdj) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | [R] original-only |

> **结论（与 NM2025 一致）**：TEINet/ERGO 变体 seen 有信号（AUPRC 0.58–0.67）、**unseen 全部回落近随机（0.50–0.52）**；TPBTE 两变体 seen/unseen 均≈随机（0.50）。SETE 的官方 pickle 在 seen/AS 报输入 433 维而模型期待 1 维，故使用论文 seen 值；论文未给 SETE unseen 值，保持空缺而不补造。TEPCAM wrapper 尚未接通，seen/unseen 均明确使用论文值。

#### L2 · 负样本来源对照（AS/PS/HS · overall AUPRC；`summary_neg_source.csv`）

| Method | seen AS | seen PS | seen HS | unseen AS | unseen PS | unseen HS |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| ATM-TCR | 0.696 | 0.682 | 0.666 | 0.522 | 0.508 | 0.519 |
| NetTCR-2.2(b) | 0.611 | 0.622 | 0.578 | 0.505 | 0.572 | 0.526 |
| TEIM | 0.676 | 0.700 | 0.741 | 0.518 | 0.536 | 0.596 |
| ERGO-II | 0.545 | 0.556 | 0.554 | 0.510 | 0.507 | 0.509 |
| epiTCR(noMHC) | 0.557 | 0.541 | 0.543 | 0.482 | 0.500 | 0.490 |
| PanPep | 0.499 | 0.504 | 0.590 | 0.479 | 0.448 | 0.510 |
| CDR3-kNN(k=5) | 0.553 | 0.526 | 0.649 | 0.544 | 0.544 | 0.544 |
| Ours esmc300m-cmp500k (globalfeat) | 0.583 | 0.560 | 0.657 | 0.507 | 0.484 | 0.547 |
| Ours esmc600m-cmp500k (globalfeat) | 0.576 | 0.521 | 0.650 | 0.496 | 0.471 | 0.518 |
| Ours esmc300m-mint (globalfeat) | 0.596 | 0.535 | 0.639 | 0.501 | 0.478 | 0.508 |

> **本地 sensitivity `[L]`**：换负样本来源会明显移动 AUPRC。该表不替代 canonical original-only 主表；主 baseline 固定 AS，并逐模型选择完整官方重跑 `[R]` 或带标签的论文 fallback `[P]`。

#### L1 · 近似去重（train↔test CDR3β Levenshtein≤k；`near_dedup_summary.csv`）

去掉与官方 train.csv 任一 CDR3β 距离 ≤k 的 test 行后在同一硬子集复算 overall AUPRC：

| | k=0(原) | k≤1 | k≤2 | k≤3 |
|---|:---:|:---:|:---:|:---:|
| 去掉 unseen 比例 | 0%(690) | 19.0%(559) | 67.4%(225) | 92.6%(51) |
| 去掉 seen 比例 | 0%(1956) | 9.5%(1770) | 57.3%(836) | 87.5%(244) |
| kNN unseen AUPRC | 0.544 | 0.487 | 0.499 | 0.490 |
| ATM-TCR unseen | 0.522 | 0.523 | 0.564 | 0.542 |
| TEIM unseen | 0.518 | 0.507 | 0.528 | 0.560 |
| Ours-cmp500k unseen (enc-only control) | 0.534 | 0.520 | 0.511 | 0.534 |

> **结论**：k≤2 去掉 unseen 67.4%（落 arXiv 2606.04994 的 40–70%）；去重后 unseen 仍全体≈0.5（"泛化失败"是真结论）；但 **kNN 表观优势 k≤1 即坍缩**（0.544→0.487），距离法吃近重复红利被证实。k≤3 仅剩 51 行不作主张，原口径全保留。

> 复现：
> ```bash
> ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
> export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH; cd benchmark
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_nm2025_binding.py --tests-only
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py --method random --track cdr3b
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py --method knn --track cdr3b
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name epitcr --tag epitcr_nomhc --track cdr3b --neg-source AS
> $ENV/bin/python tcr_binding/run_nm2025.py --method embed --track cdr3b \
>     --embedder grammar:encoder:/abs/grammar_v2_esmc300m_cmp500k_llada/best.pt \
>     --head mlp --tag ours_biseq_mlp
> for b in atmtcr nettcr teim panpep; do
>   $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name $b --track cdr3b --neg-source AS; done
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py --name teinet --tag teinet_small --track cdr3b --neg-source AS
> $ENV/bin/python scripts/nm2025_near_dedup.py --splits seen unseen --neg AS --dedup-ref all
> $ENV/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py
> ```

#### T1 retrained 五折 · baseline 与 Ours（独立诊断，不替换 original-only 主表）

> 数据直接取 Figshare `retrain.zip` 的五个 AS fold。8 个 baseline 使用发布的 official retrained checkpoint 本地推理；Ours 每 fold 用自己的 `i_1_1train.csv` 训练 frozen-backbone head。两张表分别展示 seen test 与 seen independent；baseline 均按 headline AUPRC 从高到低排列，Ours 固定在最后。当前展示先不包含 unseen。

##### Seen test

| Method | AUROC | AUPRC |
|---|---:|---:|
| epiTCR | **0.8047±0.0015 (#1)** | **0.8253±0.0016 (#1)** |
| TEIM | 0.7641±0.0029 (#3) | 0.7802±0.0035 (#2) |
| ATM-TCR | 0.7644±0.0017 (#2) | 0.7707±0.0044 (#3) |
| ERGO-AE | 0.7286±0.0040 (#4) | 0.7405±0.0043 (#4) |
| NetTCR | 0.7271±0.0050 (#5) | 0.7371±0.0056 (#5) |
| TCR-H | 0.6786±0.0044 (#6) | 0.6805±0.0054 (#6) |
| TEINet | 0.5818±0.0768 (#8) | 0.5906±0.0848 (#8) |
| ERGO-lstm | 0.5016±0.0018 (#9) | 0.4999±0.0007 (#9) |
| **Ours BioSeq frozen global-mean + MLP** | **0.6532±0.0026 (#7)** | **0.6699±0.0026 (#7)** |

##### Seen independent

| Method | AUROC | AUPRC |
|---|---:|---:|
| epiTCR | **0.6662±0.0030 (#1)** | **0.7080±0.0007 (#1)** |
| TEIM | 0.6171±0.0042 (#2) | 0.6303±0.0073 (#2) |
| ATM-TCR | 0.6067±0.0071 (#3) | 0.5881±0.0083 (#3) |
| ERGO-AE | 0.5701±0.0104 (#5) | 0.5788±0.0090 (#4) |
| NetTCR | 0.5713±0.0116 (#4) | 0.5630±0.0099 (#5) |
| TCR-H | 0.5674±0.0067 (#6) | 0.5596±0.0075 (#6) |
| TEINet | 0.5226±0.0211 (#8) | 0.5250±0.0245 (#8) |
| ERGO-lstm | 0.4998±0.0022 (#9) | 0.4981±0.0038 (#9) |
| **Ours BioSeq frozen global-mean + MLP** | **0.5462±0.0026 (#7)** | **0.5558±0.0020 (#7)** |

> 分表产物：`comparison_with_ours_seen_test_local_release_AS.{md,csv}` 与 `comparison_with_ours_seen_independent_local_release_AS.{md,csv}`。CSV 中保留 numeric mean/SD/rank、来源和 ERGO-AE official tail-drop 行数注释。复现入口：`tcr_binding/run_retrained_ours.py`。

---

### T1 · IMMREP23 辅基准（历史，与 NM2025 不可直接比较）

**数据集（IRBench 辅基准）**：IMMREP23 paired-chain VDJdb 正样本 + 自生成参考负样本(Lev>3, 5:1) 作训练；
官方 `solutions.csv`(含负样本) 作测试。**防泄露**：训练集中与测试克隆型(CDR3a|CDR3b)重叠的 60 条正样本已剔除，
最终 clonotype/pair overlap = 0（见 `data/tcr_binding/leakage_report.json`）。
测试按表位是否在训练集出现分为 **seen(13 表位/2418 行)** 与 **unseen(7 表位/1066 行)**。
主排名 = **unseen-epitope**（泛化到新表位，最难也最有意义）。

> **主指标口径（IMMREP23 历史口径）**：unseen-epitope macro-AUPRC 为主榜，AUROC/AUC0.1 辅报。

### Unseen-epitope split (主排名，2026-07-02 全量复跑)
| Method | macro-AUPRC | macro-AUROC | macro-AUC0.1 | 类型 | 备注 |
|--------|-------------|-------------|--------------|------|------|
| **epiTCR (官方 RF)** | **0.195** | **0.525** | **0.509** | 官方源码+权重 | 迄今 unseen 最好，仍近随机 |
| Random | 0.192 | 0.473 | 0.504 | floor | |
| ESM2-150M + linear probe | 0.189 | 0.470 | 0.496 | 通用蛋白LM | 均值池化拼接，线性不建模交互 |
| CDR3-kNN (TCRdist式) | 0.167 | 0.500 | 0.500 | 距离/训练free | 无参考结合子→必然随机 |
| **Ours-BioSeq** | _pending_ | | | 扩散免疫基础模型 | 训练完成后填 |

### Seen-epitope split
| Method | macro-AUROC | macro-AUPRC | macro-AUC0.1 | 备注 |
|--------|-------------|-------------|--------------|------|
| CDR3-kNN (TCRdist式) | **0.694** | **0.539** | **0.689** | 有同表位参考结合子 |
| epiTCR (官方 RF) | 0.612 | 0.352 | 0.584 | 官方源码 baseline |
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

## T2 — TCR Clustering（双基础：NAR-GAB 2025 主 + Brief Bioinformatics 2025 第二基础）

> **两个构建基础**：**基础A = NAR-GAB 2025**（配对 TCR + 官方 9 方法 + Purity–Retention–Sensitivity，见 (a)(b)(c)）；**基础B = Brief Bioinformatics 2025 / TCREmbedding**（9,033 唯一 CDR3β/25 表位 + embedding→clustering 的 ARI/NMI/Purity，见 (d)）。基础A 提供"官方方法级 baseline"权威来源；基础B 提供 Ours-BioSeq 最对口的"嵌入→聚类"同口径 + 第二独立数据集 + ARI/NMI 完整指标。两基础**分区呈现、互不混表**（配对口径 vs 单链 CDR3β 口径）。

> **数据（基础A，主）**：直接复用 NAR Genomics & Bioinformatics 2025《Benchmarking unsupervised methods for inferring TCR specificity》开源仓库 [`i3-unit/TCR_Unsupervised_Benchmark`](https://github.com/i3-unit/TCR_Unsupervised_Benchmark)（commit `ac767882`）的 curated pooled DB（IEDB+McPAS+VDJdb），照其 `database_processing.R` 复算 CD8/VS=2/AIS>4.3/配对唯一口径，得 **5,368 唯一配对 TCR / 9,370 唯一单链序列 / 374 表位**（`data/tcr_clustering/{tcrs.csv,meta.json}`）。论文口径为 4,779 配对 / 8,395 序列——仓库 README 明确警告数据库版本漂移；**6 个高亮表位的配对/链计数与论文逐一完全一致**（GIL 706/484/456…）。
> **评测单元**：唯一**配对 TCR**（`pair_id`）。paired 方法（clusTCR/GLIPH2/DeepTCR/TCRdist3）用其配对簇；chain-separate 方法（HD/LD/TCRMatch/iSMART/GIANA）用 **β 链簇映射到配对**（同一 β 归同簇）。**防泄露**：表位标签只用于评测，不参与聚类。
> **指标**（`common/metrics.py::clustering_metrics`，NAR-GAB 口径）：Purity / Retention / Sensitivity / %clusters purity>0.9 / %seqs in high-purity clusters（+ NMI/ARI + cluster-size>3/5/10 复算）。
>
> **口径归属（重要）**：论文原生 Retention/Purity `[P]` 是主 baseline。七种方法有作者 assignment artifact；HD/LD 是本地距离图重建。九种方法又被投影到漂移的 5,368-pair universe，因此“复算”和统一 pair 表都是 `[A]/[L], mismatched` diagnostic。
> - **我们的模型 / 参照嵌入（(c) 区）**：ESM2/k-mer/TCR-VALID/**Ours-BioSeq** 无"官方聚类输出"可用（论文本身也是对 ESM/tcrBERT 做 embedding+聚类），故用**我们统一评测框架**的阈值式聚类壳（`embed-threshold`）。这些是"我们的模型/参照嵌入"，**与官方 9 方法分区呈现、不混为一栏**。
> - HD/LD 明确标作 `local_reimplementation`；2026-07-14 修复了 LD indel 分支曾静默退化成 HD 的错误，旧 LD 数值不再引用。

### (a) 论文主 baseline `[P]` + 本地校准 diagnostic（`outputs/tcr_clustering/calibration_report.csv`）

| 方法 | 复算 Ret/Pur | 论文 Ret/Pur | 一致 |
|------|-------------|-------------|------|
| clusTCR | 0.09 / 0.99 | 0.09 / 0.99 | ✓ |
| DeepTCR | 0.92 / 0.63 | 0.92 / 0.63 | ✓ |
| TCRMatch | 0.09 / 0.82 | 0.09 / 0.82 | ✓ |
| GLIPH2 | 0.22 / 0.80 | 0.22 / 0.80 | ✓ |
| TCRdist3 | 0.44 / 0.42 | 0.44 / 0.42 | ✓ |
| GIANA | 0.25 / 0.65 | 0.25 / 0.66 | ✓ |
| HD | 0.303 / 0.686 | 0.25 / 0.66 | ~（DB 漂移） |
| LD | 0.347 / 0.563 | 0.25 / 0.66 | ~（DB 漂移；indel bug 已修） |
| iSMART | 0.18 / 0.75 | 0.25 / 0.66 | ~（仓库输出文件为较小一次运行） |

> **6/9 完全吻合**（跨 purity 0.42–0.99、retention 0.09–0.92 全范围）→ 指标实现与输出解析正确。HD/LD 在论文里由 stringdist **现算**（仓库无预计算文件），我们同样现算，受复算数据库比论文多 ~975 序列影响略偏高；iSMART 仓库预计算输出仅 ~1,538 序列（< 论文 0.25 retention 对应量），如实标注。

### (b) 本地 hybrid pair-universe diagnostic（N=5,368；非论文原生）

七种为作者 assignment artifact 投影；HD/LD 为本地重建。所有行 `protocol_alignment=mismatched`：

| Method（官方输出来源） | Purity | Retention | Sens(top6) | %clu pur>0.9 |
|--------|--------|-----------|-----------|--------------|
| clusTCR | **0.991** | 0.079 | 0.135 | 0.964 |
| TCRMatch | 0.966 | 0.105 | 0.127 | 0.866 |
| GIANA | 0.931 | 0.175 | 0.290 | 0.714 |
| iSMART | 0.917 | 0.174 | 0.281 | 0.683 |
| GLIPH2 | 0.904 | 0.186 | 0.277 | 0.638 |
| HD（本地重建） | 0.860 | 0.241 | 0.342 | 0.615 |
| LD（本地重建，indel 已修） | 0.760 | 0.267 | 0.207 | 0.549 |
| TCRdist3 | 0.421 | 0.387 | 0.004 | 0.070 |
| DeepTCR | 0.630 | **0.821** | **0.376** | 0.194 |

> 该 hybrid 表在排序上与论文有相似处，但因 universe 漂移而不作为论文复现或主 baseline。LD 是本地 Levenshtein≤1 连通分量重建，不称为官方 artifact。

### (c) 我们的模型 / 参照嵌入 —— 统一评测框架的阈值式聚类（**非论文官方方法 baseline**，与 (a)(b) 分区）

> 这些方法无论文聚类输出，使用本地 `embed-threshold` 壳。由于本地 universe 与论文不同，只在本地曲线内部比较；表中的 retention 坐标仅用于取点，不构成与论文九方法的数值排名。

| 嵌入方法 | Pur@ret≈0.19 (GLIPH2) | Pur@ret≈0.25 (HD/LD/GIANA) | Pur@ret≈0.39 (TCRdist3) | Pur@ret≈0.82 (DeepTCR) | 曲线AUC |
|----------|:---:|:---:|:---:|:---:|:---:|
| k-mer(3) 组成(训练free) | 0.931 | 0.842 | 0.664 | 0.188 | 0.420 |
| ESM2-150M（通用蛋白 LM） | 0.949 | 0.860 | 0.633 | 0.180 | 0.408 |
| TCR-VALID（我们口径复用其 latent） | **0.983** | **0.953** | **0.815** | 0.263 | **0.521** |
| **Ours-BioSeq** esmc300m-cmp500k (enc-only 消融) | 0.964 | 0.912 | 0.661 | 0.207 | 0.441 |
| **Ours-BioSeq** esmc600m-cmp500k (enc-only 消融) | 0.968 | 0.917 | 0.666 | 0.184 | 0.419 |
| **Ours-BioSeq** esmc300m-mint (enc-only 消融) | 0.959 | 0.874 | 0.602 | 0.176 | 0.390 |
| **Ours post-LLaDA** esmc300m-cmp500k | 0.963 | 0.853 | 0.613 | 0.211 | 0.426 |
| **Ours post-LLaDA** esmc600m-cmp500k | 0.969 | 0.888 | 0.615 | 0.193 | 0.434 |
| **Ours post-LLaDA** esmc300m-mint | 0.936 | 0.824 | 0.572 | 0.184 | 0.392 |

> Ophiuchus-Ab 嵌入按逐序列路径极慢（占位抗体权重），本轮未在时限内产出曲线，如实标注（其余两个通用 LM 参照 esm2_150m/kmer 已产出）。ret≈0.08（clusTCR 的超低保留/超高纯度区）阈值法不可达（这些嵌入的余弦图最低保留约 0.10，仅剩完全重复序列的簇），如实留空。

### (d) 第二基础 —— embedding→clustering 的 ARI/NMI/Purity（Brief Bioinformatics 2025 / TCREmbedding）

> **为何加第二基础**：NAR-GAB 2025（基础A）给的是**配对 TCR + 官方 9 方法 + Purity–Retention 权衡**，但缺 ARI/NMI 完整一致性指标、也只有一个数据集。Brief Bioinformatics 2025《A comprehensive benchmarking for evaluating TCR embeddings in modeling TCR-epitope interactions》（bbaf030, PMID 39883514；官方仓库 [`deepomicslab/TCREmbedding`](https://github.com/deepomicslab/TCREmbedding) commit `d5bf911`）正是**"嵌入→聚类"评测口径**：固定标注数据集（GIANA 项目 curated，**9,033 唯一 CDR3β / 25 抗原表位**，≥110/类，`dataset/clustering/TCRantigenData_unique_test.csv`）→ 标准 sklearn 聚类（**K∈{10,15,…,100} 扫描**取均值以消目标簇数偏置）→ **ARI / NMI / Purity**。这正是 Ours-BioSeq embed 方法最对口的同口径基准，补齐了基础A缺的 ARI/NMI + 第二独立数据集。
> **数据/脚本**：`scripts/prepare_tcrembedding_clustering.py` → `data/tcr_clustering_embed/{tcrs.csv,meta.json}`；`tcr_clustering/run_embed_bench.py`（嵌入 CDR3β → L2 归一 → K-means 主 + 层次(PCA-50 Ward) 辅 → ARI/NMI/Purity）；`scripts/summarize_tcrembedding_clustering.py`。**防泄露**：表位标签仅评测。**GPU 嵌入在火山引擎作业跑**（task_id `t-20260705211729-crvhj`，State Success；ESM2/k-mer 参照本地跑）。
> **口径与官方方法的关系**：基础B 是"我们的模型/参照嵌入"分区（与基础A 官方 9 方法互不混表）。参照 = **ESM2-150M / ProtBERT（通用蛋白 LM，官方 HF 权重）** + **TCR-BERT（TCR 专用 LM，仅 CDR3β）** + **k-mer(2) 组成（训练free）**；论文本身评的 19 个嵌入方法（GIANA/clusTCR/catELMo/理化…）在 Supplementary Table 2，其**定性结论**是"专用/特异性感知嵌入 > 数据驱动，通用 ESM 在 CDR3β 短序列上不占优"——我们的数字复现了这一趋势（ESM2/ProtBERT 垫底，TCR-BERT 介于 k-mer 与 Ours 之间），未擅自转录论文未复核的逐方法数值。

| 分区 | 嵌入方法 | ARI (mean/best) | NMI (mean/best) | Purity (mean/best) |
|------|----------|:---:|:---:|:---:|
| 参照 | **SCEPTR**（专用对比学习 TCR 模型，CDR3β+TRBV partial） | **0.033 / 0.063** | **0.159 / 0.186** | **0.339 / 0.370** |
| 参照 | TCR-BERT（TCR 专用 LM，仅 CDR3β） | 0.016 / 0.021 | 0.095 / 0.120 | 0.287 / 0.304 |
| 参照 | k-mer(2) 组成（训练free） | 0.010 / 0.013 | 0.079 / 0.112 | 0.268 / 0.294 |
| 参照 | ESM2-150M（通用蛋白 LM） | 0.007 / 0.008 | 0.068 / 0.093 | 0.258 / 0.279 |
| 参照 | ProtBERT（通用蛋白 LM） | 0.006 / 0.008 | 0.061 / 0.087 | 0.248 / 0.269 |
| **Ours post-LLaDA** | esmc300m-cmp500k（headline） | 0.024 / 0.045 | 0.140 / 0.152 | 0.317 / 0.332 |
| **Ours post-LLaDA** | esmc600m-cmp500k | 0.022 / 0.036 | 0.129 / 0.147 | 0.306 / 0.322 |
| **Ours post-LLaDA** | esmc300m-mint | 0.004 / 0.006 | 0.055 / 0.076 | 0.242 / 0.257 |
| **Ours** | **BioSeq** esmc300m-cmp500k（enc-only 对照） | 0.021 / 0.036 | 0.128 / 0.146 | 0.306 / 0.321 |
| **Ours** | **BioSeq** esmc600m-cmp500k（enc-only） | 0.018 / 0.027 | 0.109 / 0.136 | 0.293 / 0.315 |
| **Ours** | **BioSeq** esmc300m-mint（enc-only） | 0.006 / 0.009 | 0.064 / 0.089 | 0.257 / 0.274 |

> post-LLaDA 行来自 Volc `t-20260705215519-nck2s`（T2 部分成功，`bioseq-llada:` tag `*_postllada`）；**SCEPTR 行本地跑**（`run_embed_bench.py --embedder sceptr`，14.7s，`outputs/tcr_clustering_embed/sceptr/`）。源：`outputs/tcr_clustering_embed/_summary.csv`。

> 表内为 **K-means** mean/best over K；层次(Ward on PCA-50)趋势一致（esmc300m-cmp500k ARI 0.025/NMI 0.138/Purity 0.314），完整曲线见 `outputs/tcr_clustering_embed/<tag>/curve.csv`。**绝对值偏低是单链 CDR3β 表位聚类的固有难度**（论文各方法 ARI 亦普遍在 0.0x 量级）；关键是**同数据同协议下的相对次序**。
>
> **加入 SCEPTR 后的诚实次序（2026-07-08）**：专用对比学习 **SCEPTR 三项均最高（ARI 0.033 / NMI 0.159 / Purity 0.339），超过我们的模型**；我们的 post-LLaDA cmp500k（ARI 0.024 headline）领先所有通用/自监督 PLM（TCR-BERT 0.016 > k-mer 0.010 > ESM2 0.007 > ProtBERT 0.006）但**低于 SCEPTR**——与 T3 及 SCEPTR 论文"对比学习专用模型 > 通用/自监督表征"的结论一致、互相印证。**口径注**：SCEPTR 用 CDR3β+TRBV（其原生 partial 输入口径，α 缺失；比纯 CDR3β 多 V 基因信息），其余参照仅 CDR3β。

### 核心结论
- **口径校验通过**（6/9 官方方法逐一复现论文 Purity/Retention）→ 我们的 NAR-GAB 指标实现可信；数字全部可由 `outputs/tcr_clustering/` 复现。
- **Ours-BioSeq post-LLaDA headline 已接入 T2**（`bioseq-llada:`，Volc `t-20260705215519-nck2s`）：基础A 仍优于 ESM2/k-mer 但**略低于 enc-only 对照**（ret≈0.25：600m post-LLaDA 0.888 vs enc 0.917）；基础B cmp500k post-LLaDA 领先所有通用/自监督 PLM（ARI 0.024 vs ESM2 0.007/k-mer 0.010），但**低于专用对比学习 SCEPTR（0.033，2026-07-08 新增参照）**。
- **两类范式各擅胜场（与论文一致的权衡）**：阈值式嵌入聚类（含 Ours-BioSeq）在**中低保留区**能达到甚至超过 GLIPH2/GIANA 的纯度，并在 TCRdist3 的 retention（0.39）处纯度远超之（0.66 vs 0.42）；但**高保留区**（DeepTCR 的 0.82）仍是 DeepTCR 独有优势（阈值法此时坍缩为巨簇、纯度骤降）。

### 复现命令
```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
cd benchmark
# 1) 数据（需先放置 baselines/TCR_Unsupervised_Benchmark，gitignore）
$ENV/bin/python scripts/prepare_tcr_clustering.py
# 2) 导入 9 官方方法（作者预计算输出）+ 口径校验
$ENV/bin/python scripts/import_clustering_baselines.py
$ENV/bin/python tcr_clustering/run.py --method precomputed --all
# 3) 我们的模型 / 参照嵌入的统一阈值曲线（非官方 baseline）
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder esm2_150m --tag esm2_150m_thr
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder kmer --tag kmer_thr
$ENV/bin/python tcr_clustering/run.py --method embed-threshold --embedder bioseq:/abs/best.pt --tag ours_esmc300m_cmp500k
# 4) TCR-VALID（独立 tcrvalid TF env）：原生协议 + 我们口径
conda run -n tcrvalid python scripts/tcrvalid_embed_cluster.py
$ENV/bin/python tcr_clustering/run.py --method embed-threshold \
    --embedder emb:TCR-VALID:$(pwd)/outputs/tcr_clustering/tcrvalid/latents.npy --tag tcrvalid
# 5) 汇总
$ENV/bin/python scripts/summarize_tcr_clustering.py

# ===== 第二基础 (d)：Brief Bioinformatics 2025 / TCREmbedding embedding→clustering =====
# 6) 数据（需先放置 baselines/TCREmbedding/dataset/clustering/TCRantigenData_unique_test.csv，gitignore）
$ENV/bin/python scripts/prepare_tcrembedding_clustering.py
# 7) 参照嵌入本地跑（轻量）
$ENV/bin/python tcr_clustering/run_embed_bench.py --embedder kmer      --tag kmer      --algos kmeans,hierarchical
$ENV/bin/python tcr_clustering/run_embed_bench.py --embedder esm2_150m --tag esm2_150m --algos kmeans,hierarchical
# 8) Ours-BioSeq 嵌入（GPU）→ 提交火山引擎作业（no-proxy 包装脚本）
/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_t2_clustering_embed_bioseq.yml
# 9) 汇总（ours vs 参照）
$ENV/bin/python scripts/summarize_tcrembedding_clustering.py
```

### 补充 baseline：TCR-VALID（Nat Commun 2024, `peterghawkins-regn/tcrvalid`, commit `475ed964`, Apache-2.0）
- **原生协议**（`outputs/tcr_clustering/tcrvalid_native/`）：其 VAE latent 上 DBSCAN(manhattan) 扫 eps + 其自带 `clustering_scoring`（mean_purity/well_clustered/percent_clustered，均为 %）。如 eps=3.0：mean_purity 80.5% / retention 36.8%；eps=0.5：98.4% / 18.1%。
- **本地统一口径 `[L]`**（上表 (c)）：把 TCR-VALID 的 16 维 latent 灌入相同 embed-threshold 后，其 AUC 为 0.521；这只是在该本地阈值曲线中的比较，不作为 NAR-GAB 论文协议下的领先结论。
- **诚实标注的差异**：TCR-VALID 用 **CDR2β-CDR3β_core** 特征（V 基因经 `TRBV_reference.csv` → CDR2，99.5% 配对可映射），与 NAR-GAB 的 **CDR3-only** 聚类口径不同，故列为**补充表征 baseline**、非 NAR-GAB 协议复跑；其原生 mean-purity 也与 NAR-GAB 加权 purity 定义不同。

## T3 — TCR Representation (few-shot per-epitope NN AUROC)

**锚定论文 = SCEPTR (Cell Systems 2024)**。两条互补 track：**deep 6-pMHC（主口径，严格照 Table SI III.1）** + **broad 24-epitope（互补的大表位面视图）**。

### T3-deep — SCEPTR-compatible 本地评测 `[L]`（6-pMHC deep few-shot）

> **协议**：脚本 `tcr_representation/run_paper6.py`，输出 `outputs/tcr_representation_paper6/`。6 target pMHC（GILGFVFTL/NLVPMVATV/SPRWYFYYL/TFEYVSQPFLMDLE/TTDPSFLGRY/YLQPRTFLL），universe=**25,816**（3,941 binder + 21,875 背景 VDJdb 全库负样本），k∈{1,2,5,10,20,50,100,200}、**100 seeds**、min-NN、per-pMHC AUROC → 6-pMHC macro。数据 = era-matched **VDJdb 2023-10**（与论文同期，`prepare_tcr_representation_paper6.py`）。列口径同 broad track：嵌入类 `cdr3b cdr3a`，**TCR-BERT 仅 CDR3β**。
> **提交方式**：baseline 由 `eval_paper6_tcr_representation.yml`（task `t-20260705204639-l56dp`, ExitCode 0，其 encoder-only 伪双链输出已按 PROJ_GUIDE.md 删除）；decoder+占位peptide 中间态由 `eval_paper6_ours_decoder.yml`（task `t-20260705214312-k8ktb`, ExitCode 0）；**我们 3 个模型的最终正确口径(真双链 tcr_pair)**由 `eval_jobs/eval_paper6_ours_tcrpair.yml`（`scripts/run_paper6_ours_tcrpair.sh`，task `t-20260705221018-trjnv`, ExitCode 0）重跑，输出直写共享 vepfs。
>
> **我们模型的表征口径（headline = post-LLaDA globalfeat）**：主表 **Ours** = `GrammarEmbedder` + `grammar:decoder:global:/abs/best.pt`——β+α 渲染为无 peptide 的 `tcr_pair`，干净联合前向 LLaDA 解码器，**整条 record global mean-pool** → 单向量（960 维 @300m）。tag `ours_globalfeat_<run>`；`fewshot.json` 含 `checkpoint` 溯源（`common/provenance.py`）。旧 `grammar_tcrpair_*` / `bioseq_llada_*`（07-05 ckpt）仅历史参照。

#### deep few-shot macro AUROC vs shots（真实数字，`_summary.json`；随机=0.5；"我们的模型"=post-LLaDA globalfeat · Volc `t-20260707172056-fwktl`）
| Method | 类型 | k=1 | k=5 | k=20 | k=100 | k=200 |
|--------|------|-----|-----|------|-------|-------|
| **SCEPTR** | 专用对比学习(官方权重) | **0.640** | **0.702** | **0.738** | **0.775** | **0.790** |
| TCRdist (tcrdist3) | 序列比对法(官方源码) | 0.647 | 0.701 | 0.733 | 0.764 | 0.777 |
| CDR3-Levenshtein-NN | 序列比对法(训练free) | 0.626 | 0.671 | 0.696 | 0.722 | 0.733 |
| k-mer(3) 组成 | 训练free 组成基线 | 0.607 | 0.665 | 0.690 | 0.717 | 0.728 |
| ESM2-150M | 通用蛋白LM(官方权重) | 0.591 | 0.631 | 0.656 | 0.685 | 0.696 |
| ProtBERT | 通用蛋白LM(官方权重) | 0.585 | 0.624 | 0.648 | 0.681 | 0.694 |
| TCR-BERT | TCR专用LM(官方权重,仅CDR3β) | 0.579 | 0.631 | 0.656 | 0.681 | 0.692 |
| **Ours** esmc300m-cmp500k | post-LLaDA globalfeat | 0.594 | 0.646 | 0.678 | 0.707 | **0.719** |
| **Ours** esmc600m-cmp500k | post-LLaDA globalfeat | 0.585 | 0.626 | 0.654 | 0.689 | 0.703 |
| **Ours** esmc300m-mint | post-LLaDA globalfeat | 0.557 | 0.585 | 0.606 | 0.633 | 0.645 |

> ckpt：300m-cmp500k `best.pt @2026-07-07 02:08`（sha256 `abb85171c7b7…`，稳定）；600m `@2026-07-06 06:55`（`c6cc944bf223…`，稳定）；mint 快照 `best.t3snap_20260707T110558Z.pt`（源 `best.pt @2026-07-07 09:24`，sha256 `900176b10f94…`）。源：`outputs/tcr_representation_paper6/ours_globalfeat_*/fewshot.json`（mint 本地重刷 @2026-07-07 12:25 UTC）。
>
> ⚠️ **mint = 当前最新 ckpt 快照**：训练仍 Running（Volc `t-20260707105105-9x4hq`），T3 数字（deep 0.645 / broad 0.595）对应该时刻冻结快照，不必等训练结束；`best.pt` 更新后 `FORCE=1 bash scripts/downstream/rerun_t3_mint.sh` 可重刷。cmp500k ckpt 已固定。

#### 旧 ckpt 对照（resume 前，仅历史参照）
| Checkpoint | 旧 deep k=200 | 新 deep k=200 | 旧 broad k=100 | 新 broad k=100 |
|------------|---------------|---------------|----------------|----------------|
| esmc300m-cmp500k | 0.710 (`grammar_tcrpair_seq_*`) | **0.719** | 0.657 (`bioseq_llada_*`) | **0.662** |
| esmc600m-cmp500k | 0.718 | **0.703** | 0.643 | **0.613** |
| esmc300m-mint | 0.645 | 0.645 | 0.613 | **0.595** |

#### 口径校验：本地 baseline vs SCEPTR Table SI `[P]`（k=200 per-pMHC，Δ=local−paper）
| 复现 baseline | Δ 范围 | 平均 \|Δ\| | 判定 |
|------|------|------|------|
| CDR3-Levenshtein | −0.016 … +0.006 | ≈0.006 | ✓ 高度吻合 |
| TCRdist | −0.013 … +0.003 | ≈0.007 | ✓ 高度吻合 |
| SCEPTR | −0.017 … −0.000 | ≈0.008 | ✓ 高度吻合 |
| ProtBERT | −0.029 … +0.010 | ≈0.014 | ✓ 吻合 |
| ESM2-150M | −0.052 … +0.009 | ≈0.024 | ○ 基本吻合 |
| TCR-BERT | −0.113 … +0.042 | ≈0.057 | △ 排序一致，逐点更宽松 |

> **5/6 baseline 的本地偏差较小**（逐 pMHC 平均约 0.006–0.024）；TCR-BERT 偏差明显更大。该结果说明 harness 大体兼容，但主表仍标 `[L]`，不会把重跑值冒充论文原值。
>
> **deep track 核心结论（诚实）**：每 shots 上 **SCEPTR ≥ TCRdist/CDR3-Lev ≥ ESM2/ProtBERT/TCR-BERT**（k=200：SCEPTR 0.790 ≳ TCRdist 0.777 > Lev 0.733 > 三个 PLM ~0.69）。**Ours globalfeat**：300m-cmp500k k=200 **0.719**（超 PLM ~0.03、逼近 Lev 0.733，仍低于 TCRdist/SCEPTR ~0.06–0.07）；600m k=200 **0.703**（resume 后略低于旧 ckpt 0.718）；mint **0.645**。

复现：
```bash
cd benchmark
$ENV/bin/python tcr_representation/run_paper6.py --method sceptr        # 或 tcrdist / levenshtein
$ENV/bin/python tcr_representation/run_paper6.py --method embed --embedder tcrbert --columns cdr3b   # TCR-BERT 官方口径仅 CDR3β
# 我们的模型 —— headline：post-LLaDA globalfeat
$ENV/bin/python tcr_representation/run_paper6.py --method embed \
  --embedder grammar:decoder:global:/abs/best.pt --columns cdr3b cdr3a \
  --tag ours_globalfeat_<run>
# Volc 批量：eval_jobs/eval_t3_representation_globalfeat.yml
$ENV/bin/python scripts/summarize_tcr_representation_paper6.py          # 汇总 + Table SI k=200 逐 pMHC Δ
bash scripts/run_paper6_all.sh              # 仅 baseline（tcrbert/kmer）+ 汇总；encoder-only ours 已移除
bash scripts/run_paper6_ours_decoder.sh     # 3×decoder+占位peptide 中间态（t-20260705214312-k8ktb）
bash scripts/run_paper6_ours_tcrpair.sh     # 3×真双链 tcr_pair 最终口径 + 汇总（t-20260705221018-trjnv）
```

### T3-broad — 24-epitope few-shot（互补视图，更大表位面）

> **主协议 = SCEPTR (Cell Systems 2024) few-shot per-epitope NN AUROC**：对每个目标表位、每个 shots k，从 train 池随机抽 k 个该表位参考 binder（R=5 seed），把整个 test 集按到支持集的最近邻距离打分（min-NN 取负），算该表位 AUROC，对表位内 seed 取均值、再对 24 表位 macro 平均（mean±std）。距离后端：native cdist（SCEPTR / TCRdist 真实序列/距离矩阵）、embedding cosine（ESM2 / ProtBERT / TCR-BERT / kmer / Ophiuchus / Ours-BioSeq）、editdist（CDR3β+α Levenshtein）。ProtBERT/ESM2 编码 CDR3β+CDR3α（与 SCEPTR Table SI 通用 PLM 口径一致）；TCR-BERT 按官方用法仅编码 CDR3β（`--columns cdr3b`，其上下文窗口 64、专为 CDR3 训练）。
> **数据**：与 T2 共享 clonotype 隔离 split（train=6887 / test=1722 / 24 表位，overlap=0）。k∈{1,2,5,10,20,50,100}；k≥50 时部分表位 train binder 不足被跳过（k=50 剩 20 表位、k=100 剩 15 表位，如实标注 n_epitopes）。
> **引擎自检（oracle）**：支持集含 query 正样本 → mean AUROC=1.0000 / min=0.9997（`python -m common.fewshot`，口径正确）。

#### few-shot AUROC vs shots（主表，macro mean over epitopes；随机=0.5）
| Method | 类型 | k=1 | k=5 | k=20 | k=100 |
|--------|------|-----|-----|------|-------|
| **SCEPTR** | 专用对比学习(官方权重) | **0.587** | **0.655** | **0.711** | **0.741** |
| TCRdist (tcrdist3) | 序列比对法(官方源码) | 0.577 | 0.644 | 0.687 | 0.728 |
| CDR3-Levenshtein-NN | 序列比对法(训练free) | 0.555 | 0.597 | 0.641 | 0.683 |
| k-mer(3) 组成 | 训练free | 0.543 | 0.597 | 0.643 | 0.673 |
| ESM2-150M | 通用蛋白LM | 0.561 | 0.589 | 0.638 | 0.671 |
| ProtBERT | 通用蛋白LM(官方权重) | 0.550 | 0.587 | 0.627 | 0.651 |
| TCR-BERT | TCR专用LM(官方权重) | 0.547 | 0.577 | 0.623 | 0.641 |
| Ophiuchus-Ab | 抗体权重 | 0.576 | 0.612 | 0.664 | 0.677 |
| **Ours** esmc300m-cmp500k | post-LLaDA globalfeat | 0.569 | 0.609 | 0.648 | **0.662** |
| **Ours** esmc600m-cmp500k | post-LLaDA globalfeat | 0.556 | 0.580 | 0.616 | 0.613 |
| **Ours** esmc300m-mint | post-LLaDA globalfeat | 0.541 | 0.555 | 0.593 | 0.595 |

> Ours 三行 = **post-LLaDA headline**（`grammar:decoder:global:`，tag `ours_globalfeat_*`；cmp500k/600m Volc `t-20260707172056-fwktl`，mint 本地快照重刷 @2026-07-07 12:25 UTC；probe AUROC 300m **0.793** / 600m **0.796** / mint **0.811**）。旧 `bioseq_llada_*`（07-05 ckpt）仅历史参照。

> 全部 7 个 shots（k=1/2/5/10/20/50/100）见 `outputs/tcr_representation/_summary.json` 与各 `<tag>/fewshot.json`。
>
> **本地结论 `[L]`（与 SCEPTR 论文方向一致，但不是论文数值复现）**：
> 1. **SCEPTR ≥ 序列比对法（TCRdist / Levenshtein）**：每个 shots 上 SCEPTR 均居首（如 k=20：0.711 vs TCRdist 0.687 vs Lev 0.641），符合 SCEPTR 论文"专用对比学习反超"结论。
> 2. **通用/专用 LM（ProtBERT / TCR-BERT / ESM2）≤ CDR3 Levenshtein，低于 TCRdist/SCEPTR**；这只描述本地 broad split 的排序。
> 3. **我们的模型（Ours-BioSeq, post-LLaDA）**：esmc300m-cmp500k k=20 达 0.641（≈ Lev 0.641，高于 ESM2 0.638 / ProtBERT 0.627 / TCR-BERT 0.623），但仍低于 TCRdist/SCEPTR——与"通用/自监督 PLM 表征打不过专门为 TCR 特异性设计的对比/距离法"一致；扩散预训练过 LLaDA 解码器的 mean-pool 表征尚未针对表位特异性优化。
>
> **口径对照 SCEPTR Table SI `[P]`**（论文 k=200、6 表位均值，仅供量级参照，split/表位数不同）：ProtBERT≈0.705 / TCR-BERT≈0.735 / ESM2(8M)≈0.717 / CDR3-Lev≈0.737 / TCRdist≈0.783 / SCEPTR≈0.798。不得与 24 表位本地 k=100 数值直接排名。
>
> 复现：
> ```bash
> python -m common.fewshot                                   # oracle 自检
> python tcr_representation/run.py --method levenshtein       # 零依赖主对照
> python tcr_representation/run.py --method tcrdist           # tcrdist3
> python tcr_representation/run.py --method sceptr            # SCEPTR
> python tcr_representation/run.py --method embed --embedder protbert                    # ProtBERT (Rostlab/prot_bert)
> python tcr_representation/run.py --method embed --embedder tcrbert --columns cdr3b     # TCR-BERT (wukevin/tcr-bert-mlm-only)
> python tcr_representation/run.py --method embed --embedder esm2_150m   # 或 kmer / ophiuchus
> python tcr_representation/run.py --method embed --embedder grammar:decoder:global:/abs/best.pt --tag ours_globalfeat_<run>
> python scripts/summarize_tcr_representation.py             # 汇总主表
> ```

### 辅报：24-way linear probe / 1-NN（向后兼容，非主榜）
> 冻结主干 + 线性 probe（逻辑回归）/ 1-NN，24-way 表位分类（随机 acc≈0.04）。距离法（Lev/TCRdist）无 probe 特征，故不列。

| Method | probe-AUROC(macro-OVR) | probe-Acc | kNN top-1 |
|--------|------------------------|-----------|-----------|
| ESM2-150M | 0.806 | 0.438 | 0.409 |
| ProtBERT | 0.793 | 0.405 | 0.393 |
| TCR-BERT | 0.748 | 0.416 | 0.431 |
| Ophiuchus-Ab | 0.825 | 0.452 | 0.443 |
| k-mer(3) 组成 | 0.800 | 0.430 | 0.447 |
| SCEPTR (64-d) | 0.803 | 0.425 | **0.506** |
| **Ours-BioSeq** esmc300m-cmp500k | **0.825** | 0.451 | 0.438 |
| **Ours-BioSeq** esmc600m-cmp500k | **0.825** | **0.468** | 0.432 |
| **Ours-BioSeq** esmc300m-mint | 0.811 | 0.433 | 0.411 |

> 辅报里各法 probe-AUROC 量级相近（0.80–0.825）：24-way probe 有充足监督，能从任意合理嵌入里线性读出表位特异性，区分度弱于 few-shot（这正是补 few-shot 主协议的原因）。SCEPTR 的 kNN-top1（0.506）显著领先，与其表征更"表位聚类"一致。
> **BioSeq 接入说明**：`grammar_v2_*_llada` 为 ESMC-300M/600M 扩散(LLaDA)微调 checkpoint，接入时把微调后 `encoder.esmc.*` 权重载入独立 ESMC 骨架（**0 missing / 0 unexpected 键**，非随机初值），mean-pool 末层表征。见 `common/model_api.py::EsmcBioSeqEmbedder`。

## T4 — TCR Generation（TCRT5 锚定 · 三 setting）

> **2026-07-05 重构**：主口径改为 **TCRT5 (Nat Mach Intell 2025)** 三 setting（A 无条件 / B 表位条件 / C 全长 α/β）。统一 harness = `tcr_generation_bench/run.py`；指标 = `common/metrics.py` 的 `tcrt5_*` 套件（与官方 `src/evaluation.py` 逐函数一致，**42 block 交叉验证 0 mismatch**）。完整设计见 **`TCR_GENERATION_BENCHMARK.md`**。
> 旧 IMMREP23/Markov/PWM/CondPool 口径（`tcr_generation/run.py` + `tcr_design/run.py`）保留为辅视图，本表不再作为主榜。
> **论文 baseline 主榜 = sparse-13 `[A]`**。`RVRAYTYSK/HLA-A*03:01` 是论文 simulation 保留项；旧 benchmark14 只作 `[14-pMHC control]`。held20 对 BioSeq 有 ~48% 训练 pair 泄露，仅作 seen control。

### Setting A — 无条件 CDR3β repertoire（n=5000）

> 数据：OTS paired-clean holdout(9837) 作分布参考；指标：k-mer(3) JSD↓ · novelty · NN 距离 · OLGA pgen 校准。

| Method | 类型 | JSD↓ | novelty | NN dist | pgen+ | med log10 pgen | unique |
|--------|------|------|---------|---------|-------|----------------|--------|
| **soNNia/SONIA** | 官方 post-select | **0.0428** | 0.992 | 2.37 | 1.00 | -8.84 | 4987/5000 |
| **OLGA** human_T_beta | 官方 VDJ 模型 | 0.0612 | 0.996 | 3.07 | 1.00 | -9.89 | 4998/5000 |
| **Ours-BioSeq** | 我们(扩散) | 0.295 | **1.000** | 6.27 | 0.786 | -21.28 | 5000/5000 |

> 结论：官方 soNNia/OLGA 分布最接近真实库（JSD 0.04–0.06）；BioSeq 无条件采样多样性高但分布偏离（JSD 0.30），pgen 校准显示大量序列重组概率极低——符合扩散模型未针对 VDJ 重组统计建模的预期。

### Setting B — 表位条件 CDR3β 设计 ★主榜（K=100）

> 数据：论文 sparse-13 + 单独的 reserved-simulation control；held20 为 target-rich 补充。

**论文 sparse-13（官方 stored predictions，主 baseline）**

| Method | 类型 | F1@100 | prec | recall | d_edit↓ | seq-rec | Char-BLEU | div |
|--------|------|--------|------|--------|---------|---------|-----------|-----|
| **TCRT5** | [A] paper sparse-13 | 0.0003 | 0.0002 | 0.0075 | 4.467 | 0.602 | 0.701 | 1.000 |
| **GRATCR** | [A] paper sparse-13 | 0.0000 | 0.0000 | 0.0000 | 4.397 | 0.595 | 0.677 | 0.081 |
| **ER-Transformer** | [A] paper sparse-13 | 0.0000 | 0.0000 | 0.0000 | 6.068 | 0.337 | 0.594 | 0.994 |

> TCRDiff/TcrDesign/OLGA/Ours 的现有数值属于 14-pMHC local control；统一重评分到 sparse-13 前不与上述论文 baseline 合表。

**held20（TCRT5 论文 held-out；BioSeq ⚠️ ~48% pair 泄露）**

| Method | F1@100 | prec | recall | d_edit↓ | seq-rec | Char-BLEU |
|--------|--------|------|--------|---------|---------|-------------|
| **TCRT5** (HF beam, 本次生成) | **0.0855** | 0.0855 | 0.0855 | 1.82 | 0.834 | 0.947 |
| **TcrDesign-G** (beta, 本次生成) | **0.1527** | 0.1540 | 0.1515 | 1.47 | 0.870 | 0.986 |
| **Ours-BioSeq** ⚠️seen | 0.000 | 0.000 | 0.000 | 6.36 | 0.454 | 0.411 |

> † `bioseq_unseen` = epitope 不在 BioSeq 训练集的 pMHC（7 benchmark14 + 2 held20 = 9 个），零 pair 泄露，是 BioSeq 能力主张的诚实口径。
> 结论：TCRT5 beam 在 held20 上 F1=0.086、seq-recovery=0.83；**TcrDesign-G 同 split F1=0.153、seq-rec 0.87**，为当前最强官方 baseline。BioSeq 表位条件 CDR3β 设计尚未找回任何真 binder（F1=0），但 seq-recovery 仍优于 OLGA 下限——扩散条件生成有信号但未达 TCRT5/TcrDesign 水平。benchmark14 上所有方法 F1≈0（稀疏 ref），以 seq-recovery / edit dist 为主读数；**TCRDiff**（pMHC-only 扩散对照）F1=0.0009、seq-rec 0.59。

### Setting C — 全长 α/β 组装

> 两种口径：(1) **无条件全长**（BioSeq，OTS 分布）：validity（合法 AA + CDR3β 可提取率）→ ANARCI 提取 CDR3β 后套 Setting A 分布指标；(2) **表位条件全长**（TcrDesign，SOTA）：官方 `tcrdesign.py` pipeline，全长 α/β + CDR3β 提取后套 Setting B 指标。

**(1) 无条件全长（n=500，BioSeq；Volc 任务 `t-20260705205817-8mvdh` 已完成）**

| Method | cond | frac_valid_aa (β/α) | frac_cdr3_extracted | 提取CDR3β JSD↓ | novelty | NN dist | pgen+ | med log10 pgen |
|--------|------|---------------------|----------------------|----------------|---------|---------|-------|----------------|
| **Ours-BioSeq** fulllength | 无条件 | 0.998 / 0.996 | **0.038** (19/499) | 0.846 | 1.000 | 8.47 | 0.789 | -23.99 |

> 结论（诚实标注能力边界）：BioSeq 全长 α/β 的**单残基合法率高（99.6–99.8% 合法 AA）**，但**只有 3.8% 的 β 链能被 ANARCI 提取出规范 CDR3β**——说明模型能产出"看起来像蛋白"的氨基酸串，却**尚不能组装出带保守 Cys/J-motif 的真实 TCR 全长结构**（生成物富含重复 Ala，见 `outputs/.../setting_C/ours_bioseq/full.jsonl`）。可提取出的 19 条 CDR3β 相对 OTS holdout 的 k-mer JSD=0.85、多为低重组概率序列，进一步印证全长生成远未达真实库分布。**BioSeq 的强项是 CDR3 级、非全长 de novo**——这是 TcrDesign（表位条件全长 SOTA）的定位。

**(2) 表位条件全长（TcrDesign，SOTA baseline）**

| Method | 状态 | 备注 |
|--------|------|------|
| **TcrDesign-G** (`tcrdesign_G.py -mode beta`) | ✅ | Setting B：34 pMHC k=100；held20 F1=0.153 / seq-rec 0.870 |
| **TcrDesign** (`tcrdesign.py` 全长) | ✅ | Setting C：34 pMHC k=100 已评分 → overall F1 **0.061** / prec 0.133 / recall 0.040 / seq-rec **0.772** / d_edit **2.58** / Char-BLEU 0.84（`setting_C/tcrdesign_full/metrics.json`；`bash scripts/run_tcrdesign_baseline.sh full all 100 tcrdesign_full`） |
| **TCRDiff** (2026 扩散) | ✅ benchmark14 | pMHC-only 公平口径；F1=0.0009 / seq-rec 0.587；held20 待补 |
| **TCR-epiDiff** (2025 扩散) | ⛔ 仅引用 | 官方生成代码非 turnkey-reproducible（`from TCR-epiDiff_model import *` 非法模块名；硬编码未发布 pkl 输入；采样为 denoise-from-real-TCR 非 de-novo；4 通道编码 + 未定义 `nucleotide_to_index`）→ 跑通需 reimplement，违反"官方代码 only"；本地权重仅作 provenance，不接入、不编造数字（同 LSMTCR/TCRGen）。详见 `TCR_GENERATION_BENCHMARK.md §0.3/§7` |

> 扩散 baseline 与 BioSeq 同族对照；TCRDiff 因 eval 集无 V 基因信息，wrapper 使用官方 **pMHC-only** 模式（V 留空 → CDR1/2 特征置零），与 TCRT5/Ours 条件一致。

> 口径交叉验证：`scripts/test_tcrt5_metrics.py` gold 单测全 PASS + 对官方 `ModelEvaluator` **42 block 0 mismatch**。
> 复现：`python scripts/prepare_tcr_generation_bench.py` → `python tcr_generation_bench/run.py --setting {A,B,C} --method {olga,sonnia,tcrt5,file}` → Setting C 无条件：`bash scripts/run_bioseq_setC.sh 500`（或提交 `eval_jobs/eval_tcr_generation_setC_bioseq.yml`）→ `python scripts/summarize_tcr_generation_bench.py`

## P1 — PPI (STRING 90/90)

### P0 — MINT GeneralPPI

> **当前五任务范围（2026-07-21）**：HumanPPI、YeastPPI、Gold-standard PPI、MutationalPPI、SKEMPI。选表规则是前三项使用论文 Source Data `[P]`，后两项使用本地固定 split 的全 8 模型重跑 `[L]`；论文未公开精确 fold，因此 `[L]` 不冒充论文复现，也不拿论文值覆盖。审计见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。
>
> | 方法 / 来源 | HumanPPI Accuracy `[P]` | YeastPPI Accuracy `[P]` | Gold-standard AUPRC `[P]` | MutationalPPI AUPRC `[L]` | SKEMPI Pearson `[L]` |
> |---|---:|---:|---:|---:|---:|
> | MINT | **0.8796±0.0069** | **0.6870±0.0106** | **0.6872±0.0033** | 0.6769±0.0683 | 0.3491±0.1908 |
> | ESM2-150M | 0.8537±0.0026 | 0.5973±0.0012 | 0.6775±0.0040 | 0.7349±0.0811 | **0.3493±0.1324** |
> | ESM2-650M | 0.8704±0.0146 | 0.6074±0.0084 | 0.6639±0.0065 | 0.7294±0.0824 | 0.3035±0.1658 |
> | ESM-1b | 0.8407±0.0172 | 0.6074±0.0032 | 0.6630±0.0032 | 0.7336±0.0738 | 0.3476±0.1397 |
> | ESM2-3B | 0.8759±0.0094 | 0.6218±0.0136 | 0.6513±0.0033 | **0.7520±0.0731** | 0.2965±0.1372 |
> | ProGen2-Large | 0.8204±0.0343 | 0.5838±0.0110 | 0.6146±0.0014 | 0.7140±0.0880 | 0.2733±0.0631 |
> | ProtT5-UniRef | 0.8704±0.0131 | 0.5880±0.0093 | 0.6546±0.0097 | 0.7264±0.0778 | 0.2736±0.1689 |
> | ProtT5-BFD | 0.8537±0.0146 | 0.6125±0.0176 | 0.6536±0.0039 | 0.7165±0.0812 | 0.3067±0.0883 |
> | BioSeq step121000 `[C]` | 0.6778±0.0181 | 0.6024±0.0078 | 0.5921±0.0014† | — | — |
>
> `[L]` 最终任务：MutationalPPI `t-20260722032251-pd4qq`、SKEMPI `t-20260722032319-9wfwt`，均为 **Success**；早期任务的成功格由 manifest 保留，最终 retry 只补缺失模型。机器可读混合表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_selected_baseline_results.csv`；audit 为 expected/observed=`16/16`、missing=`[]`。
>
> BioSeq `[C]` 使用用户指定的冻结 `step121000`、joint two-chain PPI grammar record、每链最多 1,024 residues、final decoder residue global mean，以及论文 Methods 的 960→640→1 / 100 epochs / 3 repetitions probe。HumanPPI 与 YeastPPI 是 exact public fixed split；†Gold-standard 使用公开 notebook 的 163,192 个 train rows，比论文 Table 1 的 163,019 多 173，因此该格必须视为可审计本地 control、不是 paper-exact reproduction。三任务完整的 8×`[P]`+Ours 表与逐任务差值见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_ours_step121000_official3ppi/REPORT.md` 和 `ours_vs_best_paper.csv`。
>
> **历史本地 port `[L]`**：`max_train=3000`、不同 head/数据；Bernett=IRBench subset，非论文 Gold-standard PPI。不得把下面的本地表称作 MINT 论文复现，也不得将它当成当前官方重建数据上的结果。
> **HumanPPI** = SaProt PEER split（test 237）；**Bernett** = IRBench STRING 90/90 split（test 2644，train/test 蛋白零重叠）。

#### 本地 diagnostic：AUROC / AUPRC（3-rep mean）

| 任务 ↓ \ 方法 → | MINT checkpoint `[L]` | ESM2-650M `[L]` | Ours-BioSeq `[L]` |
|-----------------|:-------------:|:---------:|:-----------:|
| **HumanPPI** | 0.938 / 0.910 | **0.939** / **0.941** | **0.830** / **0.811** |
| **Bernett** (90/90) | 0.739 / **0.748** | **0.786** / **0.767** | **0.609** / **0.616** |

#### 辅表：Accuracy / F1（3-rep mean）

| 任务 | MINT checkpoint `[L]` | ESM2-650M `[L]` | Ours-BioSeq `[L]` |
|------|:-------------:|:---------:|:-----------:|
| HumanPPI Acc / F1 | **0.876** / **0.874** | 0.851 / 0.855 | **0.727** / **0.675** |
| Bernett Acc / F1 | **0.661** / **0.625** | **0.709** / **0.686** | **0.531** / **0.165** |

> **运行位置**：MINT checkpoint local port = Volc `t-20260705210235-5k5zz`；ESM2 = 本地 sep；Ours = local globalfeat。三者仅在本地 capped grid 内比较。
> 复现：Volc → `eval_jobs/eval_mint_ppi_benchmark.yml`；本地 → `downstream/mint_tasks/_run_bench.sh` 或 `mint_tasks/README.md` Quickstart。

#### 本地 PLM diagnostic（非 MINT Fig.2 论文口径）

> Volc **`t-20260707165326-5qg4d`**（Success，2026-07-07，`eval_mint_ppi_baselines.yml`）。数字溯源 `mint_tasks/<task>/<stem>_sep_t3000/*_metrics.json`。

| 方法 | HumanPPI AUROC / AUPRC | Bernett AUROC / AUPRC |
|------|:----------------------:|:---------------------:|
| ESM-1b | 0.937 / 0.931 | 0.727 / 0.711 |
| ESM2-3B | 0.938 / 0.938 | 0.711 / 0.694 |
| **ProtT5-XL** | **0.939** / **0.941** | **0.799** / **0.777** |
| ProGen2-base | 0.925 / 0.906 | 0.622 / 0.603 |

> 该排序只描述本地 cap/Bernett 数据，不作“符合 MINT 论文趋势”的复现主张。

#### GeneralPPI 新任务（YeastPPI / MutationalPPI / SKEMPI）

> Volc **`t-20260707172352-znrkl`**（Success，2026-07-07，`eval_mint_ppi_newtasks.yml`；首轮 `t-20260707165937-zz8gr` 因 esm2 cache 路径 bug Failed 已修复重提）。9/9 格，`--max_train 3000`。Ours = post-LLaDA global pool（无 `--sep_chains`）；MINT/ESM2 = `--sep`。

**YeastPPI**（binding 分类，MLP head，test `best_test_*`）

| 方法 | AUROC | AUPRC |
|------|:-----:|:-----:|
| **MINT checkpoint `[L]`** | **0.678** | **0.717** |
| ESM2-650M | 0.648 | 0.674 |
| Ours esmc300m-cmp500k (globalfeat) | 0.601 | 0.609 |
| Ours esmc600m-cmp500k (globalfeat) | 0.567 | 0.579 |
| Ours esmc300m-mint (globalfeat) | 0.582 | 0.590 |

**MutationalPPI**（10-fold CV；类别极不平衡 pos/neg≈1473/11139 → 看 AUPRC，F1≈0 无参考价值）

| 方法 | AUROC | AUPRC |
|------|:-----:|:-----:|
| ESM2-650M | **0.725** | **0.250** |
| MINT checkpoint `[L]` | 0.643 | 0.186 |
| Ours esmc600m-cmp500k (globalfeat) | 0.621 | 0.183 |
| Ours esmc300m-mint (globalfeat) | 0.619 | 0.178 |
| Ours esmc300m-cmp500k (globalfeat) | 0.614 | 0.177 |

> ⚠️ **口径脚注**：本地 MutationalPPI 的 Y2H 标签**已是 {0,1} 二值**（来自 SWING CSV 内嵌列），与 MINT 论文 Methods（Y2H 0–4 + cutoff=2 二值化 + UniProt 序列 lookup）不完全同构；数值仅在本口径内可比。

**SKEMPI 回归状态**：旧 Pearson/RMSE 在 fold-specific PowerTransformer 空间，已撤回；旧 Spearman 只作 local diagnostic。新的完整数据、raw-unit、固定 complex folds 的 8-baseline `[L]` 重跑已全部完成并进入上方主表；每个模型为三个预定义 complex-held-out fold + weighted aggregate，共覆盖 6,706 个 test rows。

#### 排除项历史记录：PDB-Bind

> PDB-Bind 已从当前 MINT 五任务面板移除。以下只保留历史原因：HF `proteinea/ppb_affinity` 的 3267 行本地子集与论文数据构建不同，旧 Pearson/RMSE 已撤回，不得进入当前排行榜。

---

> 数据：STRING model-org 90/90 正样本 + 同 split 内平衡随机负样本(1:1)。**防泄露**：train/test 蛋白重叠=0（90/90 序列相似性划分，见 `data/ppi/leakage_report.json`）。

| Method | AUROC | AUPRC | n_test | 备注 |
|--------|-------|-------|--------|------|
| ESM2-150M + Hadamard + LR | **0.815** | **0.826** | 2644 | 通用蛋白LM 嵌入（max-train 8000） |
| k-mer(3) + Hadamard + LR | 0.562 | 0.602 | 2644 | 训练free 组成特征 |
| **Ours-BioSeq** | _pending_ | | | |

> 结论：90/90 防泄露划分下 k-mer 仅 AUROC 0.56，而 ESM2-150M 嵌入升到 0.82——预训练蛋白表征在消除相似性泄露后仍显著有效，为免疫基础模型设定了强参照。
> 复现：`python scripts/prepare_ppi.py` → `python ppi/run.py --embedder kmer` / `--embedder esm2_150m --max-train 8000`

---

## A1 — 抗体/纳米抗体（NbBench，冻结主干 + 线性 probe）

> 数据：`data/nanobody_raw/nbbench/hf_data/<task>/` 官方 train/val/test split（NbBench, Zhang & Tsuda, *Mach. Learn.: Sci. Technol.* **6**(4):040502, 2025; DOI 10.1088/2632-2153/ae20ec; arXiv:2505.02022）。
> 方法：冻结嵌入 + sklearn probe（分类 LogReg / 回归 Ridge / 位点级逐残基 head），**无主干微调**。
> **全部 12 个 NbBench 任务已跑通**（8 序列级标量 + vhh_affinity-seq + 3 位点级/生成：VRClassification / Paratope / CDRInfilling）。
> 复现：`python nbbench/run.py --task all --embedder esm2_150m`（标量）、`python nbbench/run_residue.py --task all --embedder esm2_150m`（位点级）。

### 论文主表 `[P]`：NbBench Table 5（MLP head，3 seeds）

| Task | Paper metric | ESM2-150M `[P]` | Best of 11 `[P]` |
|---|---|---:|---:|
| VRClassification | Accuracy | 0.9989 | 0.9989 (ESM2-150M/AntiBERTa2) |
| CDRInfilling | BLOSUM62 recovery | 1.499 | 1.551 (ESM2-650M) |
| SARS-CoV-2 | AUROC | 0.834 | 0.884 (AbLang-H) |
| hIL6 | AUROC | 0.850 | 0.925 (AbLang-H) |
| Paratope | AUROC | 0.922 | 0.939 (AntiBERTa2-CSSP) |
| thermo-seq | Spearman | 0.389 | 0.587 (AntiBERTa2) |
| thermo-tm | Spearman | 0.301 | 0.593 (AntiBERTa2-CSSP) |
| polyreaction | AUROC | 0.833 | 0.842 (ESM2-650M) |
| nanobody_type | Accuracy | 0.994 | 0.999 (AntiBERTy) |
| vhh_affinity-seq | Spearman | 0.170 | 0.184 (IgBert) |
| vhh_affinity-score | Spearman | 0.063 | 0.128 (AbLang-L) |

> 本地 ESM2/k-mer/Ours 使用 sklearn 单 seed probe（validation 未用于训练），不是论文 MLP/3-seed 协议，统一放在 `_leaderboard.json.local_diagnostic`，不得和上表排名。Paratope 本地 primary=AUPRC、论文=AUROC，尤其不能混列。hTNFa 是本地额外任务，论文 Table 5 不含。

> **⚠ thermo 近似重复告警（诚实标注）**：`thermo-tm` 有 **52%** 测试样本到训练集最短编辑距离 ≤2、`thermo-seq` 约 32%（NbBench 采用 75% 相似度切分，非严格去冗余）。故组成敏感的 **k-mer 在 thermo 上 spearman 虚高（0.84/0.73）**，甚至超过官方所有 PLM——这是数据近似重复而非表征优势。回归任务的可信信号仍以 **affinity**（无此问题、各法均低）为准；报告 thermo 数字时须连同该注解。


### Ours-BioSeq post-LLaDA（NbBench · 12 探针 + 生成原生 · 2026-07-05 23:01（UTC+8））
> embedder = `grammar:decoder:/abs/best.pt`；SARS-CoV-2 抗原 **head-1024 截断**。Resume Volc：`t-20260705224330-dcd5w` / `t-20260705224329-984qk` / `t-20260705224330-zf7zf` → **Success**；源：`outputs/nbbench/**/ours_*_llada/metrics.json` + `CDRInfilling/gen_ours_*_llada/`。

| Checkpoint | VRCls acc | Paratope AUPRC | SARS-CoV-2 AUROC | CDRInf (probe BR) | CDRInf gen (EM / BR) |
|------------|-----------|----------------|------------------|-------------------|----------------------|
| esmc600m-cmp500k | 0.9975 | 0.6843 | **0.847** | 1.406 | 0.400 / **1.569** |
| esmc300m-cmp500k | 0.9982 | 0.6819 | **0.874** | 1.440 | 0.375 / 1.476 |
| esmc300m-mint | 0.9959 | **0.7106** | 0.852 | 1.473 | **0.409** / **1.594** |

> 全榜见 `_leaderboard.json.local_diagnostic`。这些值只用于同一 local sklearn harness 内的诊断，不与论文 Table 5 作数值高低结论。

### AB 抗体生成（grammar_v2 · SabDab CDR + OAS light pairing · post-LLaDA 扩散解码）

> 协议：`scripts/downstream/run_grammar_v2_variant_downstream_eval.sh`；CDR = SabDab 10-fold Average AAR all folds；pairing = OAS holdout500 prompt3 n=8，ImmunoMatch + ANARCI。源：`output/downstream_generation/grammar_v2_esmc{300,600}m_cmp500k_llada_*`。7-05 Volc：`768fw`（300M Failed @ tf-keras）/ `zwgdk`（600M Success）/ `mrmzv`（300M metrics-only Success）。

#### CDR infilling（Average AAR all folds ↑）

| Checkpoint | CDR-H1 | CDR-H2 | CDR-H3 | 状态 |
|------------|:------:|:------:|:------:|:----:|
| **Ours** esmc300m-cmp500k | **68.76** | **63.64** | **44.68** | ✅ |
| **Ours** esmc600m-cmp500k | **71.88** | **67.27** | **45.25** | ✅ |

> vs 7-04 旧 ckpt：300M 64.72/57.93/40.54 → 上表（Δ +4.0/+5.7/+4.1 pp）；600M 68.95/62.34/41.80 → 上表（Δ +2.9/+4.9/+3.5 pp）。

#### CDR infilling 论文 baseline `[P]`（Ophiuchus-Ab Table 2）

| Model | CDR-H1 | CDR-H2 | CDR-H3 | 来源/注记 |
|-------|:------:|:------:|:------:|:----:|
| AntiBERTy | 76.70 | 71.10 | 42.70 | [P] overlap flagged |
| AbLang2 | 76.30 | 70.60 | 42.70 | [P] overlap flagged |
| Ophiuchus-Ab | 75.50 | 70.18 | **43.55** | [P] no overlap; zero-shot |

> 本地 AntiBERTy/AbLang2 masked-fill 和 Ours 是另一协议，保留在 `cdr_baselines/_summary.csv`，不覆盖上表。Ophiuchus 路径已修，local rerun pending。

#### Light-chain pairing（holdout500 · prompt3 · n=8）

| Checkpoint | gen ImmunoMatch↑ | ref ImmunoMatch | gen>ref ratio | chain match | V gene match | diversity | 状态 |
|------------|:----------------:|:---------------:|:-------------:|:-----------:|:------------:|:---------:|:----:|
| **Ours** esmc300m-cmp500k | **0.633** | 0.699 | **0.393** | **1.000** | **0.835** | 0.163 | ✅ `mrmzv` |
| **Ours** esmc600m-cmp500k | **0.607** | 0.699 | **0.368** | **1.000** | **0.835** | 0.140 | ✅ |

> vs 7-04：300M gen ImmunoMatch **0.430 → 0.633**；600M **0.455 → 0.607**；V gene 均 **→ 0.835**（旧 0.577）。源：`grammar_v2_esmc{300,600}m_cmp500k_llada_light_pairing_holdout500_prompt3_metrics.json`（mtime 2026-07-05）。

#### Light pairing baseline（Ophiuchus-Ab Table 3 口径 · OAS holdout500 × n=8）

外部 baseline **均为自跑**（论文未提供 p-IgGen/LICHEN 数字）；`de-novo`（无 light prompt）与 `prompt3`（前 3 个 reference light 残基作 prompt，与 grammar_v2/Ophiuchus 同口径）两套。grammar_v2 亦补跑 de-novo(prompt0) 以隔离 chain-match 成因。

| Model | 口径 | gen ImmunoMatch↑ | ref | chain match | V(exact/fam) | J | diversity | 状态 |
|-------|:----:|:----------------:|:---:|:-----------:|:------------:|:---:|:---------:|:----:|
| **p-IgGen** | de-novo | **0.686** | 0.699 | 0.552 | 0.050/0.179 | 0.150 | 0.695 | ✅ |
| **p-IgGen** | **prompt3** | **0.683** | 0.699 | **0.990** | 0.320/0.880 | 0.282 | 0.368 | ✅ |
| **LICHEN** | de-novo | 0.666 | 0.699 | 0.622 | 0.053/0.206 | 0.152 | 0.519 | ✅ |
| **LICHEN** | **prompt3** | 0.632 | 0.699 | **0.998** | 0.292/0.887 | 0.276 | 0.255 | ✅ |
| **Ours** esmc300m-cmp500k | **de-novo** | 0.604 | 0.699 | **1.000** | **0.846**/0.964 | 0.693 | 0.168 | ✅ |
| **Ours** esmc300m-cmp500k | prompt3 | 0.633 | 0.699 | **1.000** | **0.835**/0.964 | 0.628 | 0.163 | ✅ |
| **Ours** esmc600m-cmp500k | **de-novo** | 0.629 | 0.699 | **1.000** | **0.882**/0.964 | 0.735 | 0.147 | ✅ |
| **Ours** esmc600m-cmp500k | prompt3 | 0.607 | 0.699 | **1.000** | **0.835**/0.965 | 0.703 | 0.140 | ✅ |

> **全部自跑**（p-IgGen/LICHEN 非论文数字）。外部 baseline de-novo 未上 prompt3，故 chain 仅 0.55/0.62；**补 prompt3 后 → 0.990/0.998**。**⚠️ grammar_v2 的 chain=1.000/V≈0.85 不是 prompt3 造成的**：de-novo(prompt0) 下仍 chain 1.000 / V 0.85+ / diversity 0.15，因为 mask-fill 用 mask 数=reference 长度，泄漏目标长度→生成 light 与 ref 长度 100% 一致、同一性 0.92（定长近重建），prompt3 对 chain/V 几乎无增量。与外部模型真变长 de-novo 是范式差异，匹配率不可横比。ImmunoMatch：外部专用模型最强（0.68），grammar_v2 0.60-0.63。源：`output/downstream_generation/{pairing_baselines/{piggen,lichen}_holdout500{,_prompt3}, grammar_v2_esmc*_light_pairing_holdout500_{denovo,prompt3*}}_metrics.json`。grammar_v2 de-novo=当前 best.pt，prompt3=7-05 ckpt（同 ckpt prompt3 复跑待回填）。

### AB-FLAB 论文 baseline `[P]`（MINT Figure 3b · nested 10×5-fold R²）

> 论文主指标是 R²；历史 local sweep 是 5-fold pooled Spearman，不能替代本表。

| Model `[P]` | g6 Kd R² | g6 ER R² | trastuzumab R² | d44 R² |
|-------|:-----:|:-----:|:--------------:|:------:|
| AbLang | 0.244 | 0.439 | 0.293 | 0.246 |
| AntiBERTy | 0.199 | 0.401 | 0.239 | 0.217 |
| IgBert | 0.174 | 0.400 | 0.306 | 0.131 |
| IgT5 | 0.179 | 0.548 | 0.274 | 0.297 |
| MINT | **0.253** | **0.657** | **0.398** | **0.379** |

> 完整 7-method mean±std 见 `paper_reported_baselines.csv`。历史 32-run local Spearman sweep 仅为 diagnostic；paper-aligned nested-R² runner 已修，尚待重跑。

### 位点级任务：ESM2 vs 训练free one-hot 对照（可复现）
> **动机**：位点级任务此前只有 ESM2 一列，缺训练free 下限。新增 `onehot`（滑窗 ±3 逐残基 one-hot，纯局部组成、无预训练上下文）作对照，隔离"预训练主干相对局部 motif 的增益"——是标量 k-mer 基线的位点级类比。

| 任务 | 本地主指标 | ESM2-150M `[L]` | one-hot `[L]` | 论文 ESM2-150M `[P]` | 本地增益 |
|------|--------|-----------|-------------|----------------|------------------------|
| VRClassification | acc | **0.9989** | 0.8708 | 0.9989 | +0.128（区域判别强依赖上下文） |
| Paratope | AUPRC（AUROC） | **0.6817**（0.9231） | 0.4485（0.7923） | **AUROC 0.922** | +0.233 AUPRC |
| CDRInfilling | BR（EM） | **1.467**（0.396） | 1.124（0.329） | 1.499（—） | +0.34 BR / +0.067 EM |

> 结论：三项位点级任务上 ESM2 均显著超训练free one-hot（尤以 VRCls +0.13 acc、Paratope +0.23 AUPRC 最明显），说明**位点级任务确实需要预训练主干的上下文表征**，与标量任务里 thermo 之外 ESM2>k-mer 的趋势一致——这为基础模型在位点级任务留出了明确、真实的提升空间。

### 逐任务全指标（ESM2-150M，可复现）
| 任务 | 全部指标 |
|------|----------|
| VRClassification | acc 0.9989 / macro-P 0.9982 / macro-R 0.9985 / macro-F1 0.9983（n_pos=345679, 4 类） |
| Paratope | auprc 0.6817 / auroc 0.9231 / acc 0.8861 / P 0.670 / R 0.605 / F1 0.636（pos_rate 0.164） |
| CDRInfilling | br_masked 1.467 / aar_masked 0.3959 / exact_seq_rate 0.000（n_masked_pos=89440, n_seq=2846） |

> 复现命令：
> ```bash
> ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
> LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
> $ENV/bin/python nbbench/run.py --task all --embedder esm2_150m     # 9 标量
> $ENV/bin/python nbbench/run.py --task all --embedder kmer          # k-mer 对照
> $ENV/bin/python nbbench/run_residue.py --task all --embedder esm2_150m  # VR/Paratope/CDRInf
> $ENV/bin/python nbbench/run_residue.py --task all --embedder onehot     # 位点级 训练free 对照
> $ENV/bin/python scripts/import_nbbench_results.py                       # 固化官方参照 CSV
> ```
> 官方数字来源：NbBench (Zhang & Tsuda, *Mach. Learn.: Sci. Technol.* **6**(4):040502, 2025, DOI 10.1088/2632-2153/ae20ec; arXiv:2505.02022 v1, Tables 5–8) → `outputs/external/nbbench_official.csv`。
