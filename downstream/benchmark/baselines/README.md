# Baselines（官方代码 + 官方权重）

> 原则：只用作者 report 的官方代码/权重，**不自行复现**。各仓库记录 clone commit。
> 仓库本体不纳入 git（见 `../.gitignore`），仅保留本清单。
> 获取登记：[`_download_manifest.tsv`](_download_manifest.tsv)（`sha256` = dest 内 README/LICENSE 探针哈希，**不是** git-archive；同 license 文本会碰撞）。
> 排行榜构造：论文值 `[P]` 优先，见 [`../PROJGUIDE.md`](../PROJGUIDE.md) §0.2.2。

## 现成 benchmark（数据 + 官方评测/结果）
| 仓库 | commit | 用途 |
|------|--------|------|
| `IMMREP23` (justin-barton/IMMREP23) | 06d85be | TCR binding 主测试集；含全长 TCRα/β + CDR1/2/3 + V/J + peptide + HLA + Label，seen/unseen 由 Usage(Public/Private) 区分 |
| `IMMREP_2022_TCRSpecificity` (viragbioinfo/...) | 8942484 | TCR binding（17 表位）；自带 `evaluation/` 官方评测脚本 + `methods_results/` 17+ 方法官方结果（MicroAUC/Average Rank），可直接对比 |

## TCR binding baseline（T1，Nature Methods 2025 主基础）

**主来源 = `TCREpitopeBenchmark` (SuoLab-GZLab/TCREpitopeBenchmark) commit `ec832b47ec3a566d76ecdddb99e5510a6d96ec4e`（2026-01-21）**：NM2025 官方仓库，含各模型官方代码目录（`ATM_TCR/`、`NetTCR/`、`TEIM/` …）+ **`Original_model/` 50 个官方预训练权重**（`ATM_TCR.ckpt` / `NetTCR.h5` / `TEIM.ckpt` / `ERGOII_vdj` / `epiTCR.pickle` / `PanPep.pt` …）+ `environment/*.yml`（各模型 env，注意其 yml 为 **aarch64**，x86 需手动建/兼容 shim）。**我们只调用官方代码+官方权重**（wrapper 见 `wrappers/*.py`），不复现模型数学。

**T1 original-only（2026-09-02 论文值优先）**：外部 baseline 禁止使用 `train.csv` 或 retraining。主榜取论文 original-model `[P]`（Supplementary Table 4）。官方 original checkpoint + 官方推理代码可用时，在 figshare `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip` 上重跑产出 `[R]` **附行**，不取代 `[P]`。normalized tests 必须匹配 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_binding_nm2025/original_test_manifest.json`。无法获得或无法运行时行保留并标 **“论文值，未本地复现”**。机器可读登记为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`；本地运行诊断另存同目录 `summary_local_runs.csv`，不得替代主表。完整规则见 [`../PROJGUIDE.md`](../PROJGUIDE.md) §0.2.2。

| baseline | 官方权重 | env / 接入方式 | 状态 |
|----------|----------|----------------|:---:|
| **ATM-TCR** | `Original_model/ATM_TCR.ckpt` | 独立 env `tcrbench_atmtcr`（torch1.10+torchtext0.11）；`wrappers/atmtcr.py`（LD_LIBRARY_PATH 隔离到本 env） | ✅ |
| **NetTCR-2.2 (b)** | `Original_model/NetTCR.h5` | 独立 env `tcrbench_nettcr`（TF/Keras）；`wrappers/nettcr.py`（单链 chain='b'） | ✅ |
| **TEIM (seq)** | runtime `TEIM/Original_model_prediciton/ckpt/{teim_seq,epi_ae}.ckpt`；`teim_seq.ckpt` 与 bundle `Original_model/TEIM.ckpt` sha256 相同 | **主 env**（torch2.8+PL1.8.6+ANARCI）；`wrappers/teim.py`（`torch.load(weights_only=False)`；ANARCI+hmmscan 走主 env） | ✅ canonical `[R]` |
| **ERGO-II (vdjdb)** | `ERGO-II/Models/version_1veajht/checkpoints/_ckpt_epoch_10.ckpt` + `Models/AE/*.pt` | **主 env**（PL-0.7 代码经 `wrappers/ergo2_official_infer.py::_install_pl_shims` 兼容 shim 载入，0 missing/0 unexpected）；AE 权重 `ln -s Models/AE TCR_Autoencoder` | ⚪ L2/local diagnostic，不进 canonical 主表 |
| **epiTCR (noMHC)** | `Original_model/epiTCR.pickle` | 独立 env `tcrbench_epitcr`（sklearn RF）；`run_official_baseline.py --name epitcr --tag epitcr_nomhc` | ✅ |
| **PanPep (zero-shot)** | runtime `PanPep/Requirements/{model.pt,Content_memory.pkl,Query.pkl}`；`model.pt` 与 bundle `Original_model/PanPep.pt` sha256 相同 | 主 env（PyTorch）；`wrappers/panpep.py` | ✅ canonical `[R]` |

> **ERGO-II caveat**：其原生输入含 CDR3β+α+V/J+MHC；本 cdr3b track 仅喂 CDR3β（α/V/J/MHC=UNK，同 NM2025 partial-feature 口径），故 seen 近随机属预期。
> **兼容 shim 说明**：TEIM/ERGO-II 仅对**已移除的框架 API**（torch2.6 `weights_only`、PL 的 `data_loader` 装饰器 / `pytorch_lightning.logging` / 可写 `hparams`、CUDA→CPU map_location）做 shim，**不改动任何模型数学**；权重 100% 官方。
> 早年独立 clone 记录（仅备查，实际不用）：`ERGO-II` (IdoSpringer/ERGO-II) 85d320a、`NetTCR-2.2` (mnielLab) 7cead3f、`epiTCR` (ddiem-ri-4D) 7919d64、`PanPep` (bm2-lab) b44ffb1。

### T1 已实现的评测强化（L1/L2）
- **Original-only 来源选择**：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/original_protocol.py` 定义 15 个 Supplementary Table 4 original-model 行与 checkpoint/wrapper 对应；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py` 优先完整官方 rerun，否则回退论文值。当前 13 行已由新 runner 重跑 AS seen/unseen 并写入完整 provenance；**SETE** 因官方 pickle 运行时报特征维数不兼容、**TEPCAM** 因官方 wrapper 尚未接通，使用“论文值，未本地复现”。
- **L1 近似去重**：`scripts/nm2025_near_dedup.py` —— train↔test CDR3β Levenshtein≤k（rapidfuzz），产 `data/tcr_binding_nm2025/near_dedup_report.json` + `outputs/.../near_dedup_summary.csv`；k≤2 去掉 unseen 67.4%（复现 arXiv 2606.04994 的 40–70%）。
- **L2 负样本来源对照**：`scripts/summarize_nm2025.py` —— AS/PS/HS × seen/unseen 主表汇总（`summary_main.csv` / `summary_neg_source.csv`）。

## TCR representation baseline（T3，few-shot 主对照 + 专用模型）
| 仓库 / 包 | commit / version | 框架 / 备注 |
|------|--------|-----------|
| `sceptr` (yutanagano/sceptr) | clone commit **8c1fa02ec4**（见 `_download_manifest.tsv`）；pip `sceptr==1.2.0` | MIT。对比学习 TCR 表征模型（Cell Systems 2024），153k 参数，自带全部权重 `src/sceptr/_model_saves/*/state_dict.pt`（离线可用）。**主 env 直接装通**：`pip install --no-deps sceptr libtcrlm tidytcells blosum`（依赖版本：libtcrlm 1.1.3 / tidytcells 2.2.1 / blosum 2.2.0）。接入见 `common/model_api.py::SceptrDistance`（native cdist）。API：`calc_cdist_matrix(anchors_df, comparisons_df)` / `calc_vector_representations(df)`，输入列 `TRAV,CDR3A,TRBV,CDR3B`，CDR3 需 IMGT junction（用 tidytcells `junction.standardize(fix_missing_conserved=True)` 补 C…F；V 基因 `tr.standardize(enforce_functional=True)`，非功能性→None 走 partial）。 |
| `tcrdist3` (kmayerb/tcrdist3) | clone 在 `baselines/tcrdist3/`（无独立 .git）；pip `tcrdist3==0.3` | 序列比对法主对照（Dash 2017 / Mayer-Blackwell 2021）。**主 env 直接装通**：`pip install tcrdist3`（拉入 parasail 1.3.4 / pwseqdist 0.6 / olga 1.3.0 等）。接入见 `common/model_api.py::TcrdistDistance`（`TCRrep.compute_rect_distances` 矩形距离，α+β 加权 CDR1/2/2.5/3）。输入同样先 tidytcells 标准化；缺失/非功能 V/J 用功能性占位符补齐以推断 germline loop。 |
| **ProtBERT** (`Rostlab/prot_bert`) | HF 权重（无版本 pin，本地缓存 snapshot `7a894481acdc…` / `ebb451a407f2…`；`/c20250601/mj/model_weights/` 下无本地快照，走 `HF_ENDPOINT=https://hf-mirror.com`） | SCEPTR Table SI 的通用蛋白 PLM 对照（Elnaggar 2021，UniRef100/BFD MLM，hidden 1024）。**薄封装**：`common/model_api.py::ProtBertEmbedder`（`HFProteinLMEmbedder` 子类，`AutoTokenizer/AutoModel`，空格分隔 AA、mean-pool 末层排除 padding）。用法 `--embedder protbert`，编 CDR3β+CDR3α。 |
| **TCR-BERT** (`wukevin/tcr-bert-mlm-only`) | HF 权重（snapshot `ef65ddcb4e54…`，10 文件；同走 hf-mirror） | SCEPTR Table SI 的 TCR 专用 LM 对照（Wu 2021，~90k TRB/TRA CDR3 上做 masked-AA 预训练，BERT hidden 768 / 上下文 64）。**用 MLM-only 权重**（纯自监督）而非 `wukevin/tcr-bert`（后者额外在 PIRD 表位标签上分类微调，用于表位特异性任务会泄漏监督）。**薄封装**：`common/model_api.py::TcrBertEmbedder`（`HFProteinLMEmbedder` 子类，空格分隔 AA、mean-pool）。用法 `--embedder tcrbert --columns cdr3b`（官方以 CDR3β 为主）。 |

> **CDR3-Levenshtein-NN**（另一主对照）无需 clone：纯 `common/leakage.edit_distance`（rapidfuzz），见 `common/fewshot.py::edit_distance_matrix`。
> **ProtBERT / TCR-BERT** 与 **SCEPTR / TCRdist / CDR3-Lev / ESM2** 共同构成 SCEPTR (Cell Systems 2024) Table SI 的六模型对照，全部已跑通（见 `../RESULTS.md` T3）。
> **few-shot 引擎** `common/fewshot.py`：三后端（native cdist / embedding-cosine / editdist）统一成 query×ref 距离矩阵；oracle 自检 `python -m common.fewshot`（支持集含 query 正样本 → AUROC≈1.0）。

## TCR clustering baseline（T2，锚定 NAR-GAB 2025 / i3-unit 仓库）
| 仓库 | commit | 框架 / 备注 |
|------|--------|-----------|
| `TCR_Unsupervised_Benchmark` (i3-unit) | **ac767882f6bb7a23ccbbe973c5487ff58add04cf** | NAR Genomics & Bioinformatics 2025《Benchmarking unsupervised methods for inferring TCR specificity》官方仓库。含 curated pooled DB（IEDB+McPAS+VDJdb，`Data/Database/database_pooled_human_2023_03_15.txt`）+ **9 方法预计算聚类输出**（`Data/Output_methods/`：clusTCR/DeepTCR/GIANA/iSMART/TCRMatch/TCRdist3/GLIPH2；HD/LD 论文用 stringdist 现算，我们同法复现）。**用途**：数据集重建（`scripts/prepare_tcr_clustering.py` → `data/tcr_clustering/`，复算 5,368 配对/374 表位）+ 官方 baseline 导入与指标口径交叉验证（`scripts/import_clustering_baselines.py` → `outputs/tcr_clustering/`，6/9 逐一复现论文 Purity/Retention）。经代理不通，用 `ghfast.top` 镜像下载 tarball。 |
| `TCREmbedding` (deepomicslab/TCREmbedding) | **d5bf911f5baa6e8d45251eae1f217ad6af0b65f7** | **T2 第二构建基础**（Brief Bioinformatics 2025《A comprehensive benchmarking for evaluating TCR embeddings...》bbaf030, PMID 39883514）。它是 **embedding→clustering** 基准：固定标注数据集 `dataset/clustering/TCRantigenData_unique_test.csv`（GIANA 项目 curated，**9,033 唯一 CDR3β / 25 抗原表位**）+ 标准 sklearn 聚类（K∈{10,15,…,100} 扫描）+ **ARI/NMI/Purity**，评 19 个嵌入方法。**用途**：仅取其**固定评测数据集**作为 T2 第二基础，用**我们统一框架** `tcr_clustering/run_embed_bench.py` 跑 Ours-BioSeq + ESM2/k-mer 参照的 embedding→clustering（GPU 嵌入走 Volc 作业 `eval_jobs/eval_t2_clustering_embed_bioseq.yml`）；论文的逐方法数值在其 Supplementary Table 2，**只引定性结论不擅自转录未复核数字**。经代理不通，数据用 `api.github.com` base64 下载到 `baselines/TCREmbedding/`（gitignore）。 |
| `tcrvalid` (peterghawkins-regn/tcrvalid) | **475ed9647f924d60fa49ac6e83288f89b06275b1** | Apache-2.0。TCR-VALID（Nat Commun 2024）capacity-controlled VAE，含预训练 TRB/TRA keras 权重（`tcrvalid/logged_models/`，co-packaged 离线可用）。**独立 conda env `tcrvalid`**（TF-cpu 2.13 / py3.8；`minimal_requirements.txt` 的 TF2.8 在本平台不可得，改用 tensorflow-cpu 2.13 + `pip install -e . --no-deps`，模型 h5 正常载入、复现 README 参考 latent）。**用途**：`scripts/tcrvalid_embed_cluster.py` 在同一 curated 数据集上跑（a）原生 DBSCAN 聚类协议 → `outputs/tcr_clustering/tcrvalid_native/`；（b）导出 16D latent 供我们 embed-threshold 口径 → `outputs/tcr_clustering/tcrvalid/`。输入按 TCR-VALID 的 CDR2β-CDR3β_core 格式（V 基因→CDR2 映射 99.5%），与 NAR-GAB CDR3-only 口径差异已在 RESULTS.md 标注。经代理不通，用 `ghfast.top` 镜像下载 tarball。 |

## 待 clone
- pMTnet (tianshilu/pMTnet)：TF1.x + Keras2.2，含 library 权重（环境较老，单独处理）
- catELMo (Lee-CBG/catELMo)：T3 可选对照。**blocker**：GitHub clone 无代理超时；ELMo 栈依赖 `allennlp 2.10`（要求 torch<1.13）与主 env torch 2.8 冲突，需独立老环境 + 外部 ELMo 权重下载。暂跳过（非阻塞）。
- 其他维度（clustering / generation / PPI）：在对应任务推进时再 clone。

> 备注：TCR-BERT 早前 manifest `DL_FAIL` 的是**仓库源码 clone**；本轮直接用官方 **HF hub 权重**（`wukevin/tcr-bert-mlm-only`）经 transformers 载入，已接入并跑通（见上表）。

## 环境说明
各 baseline 框架差异大（TF1.x / Keras / PyTorch / sklearn），按官方 requirements 各建独立 conda 环境，wrapper 只做「读统一输入 → 调官方推理 → 写统一预测文件」。

| baseline | 环境注意 |
|----------|----------|
| epiTCR | RF pickle 用 sklearn 1.2 训练；主 env sklearn>=1.3 需单独 env 或 ``scripts/epitcr_infer.py`` |
| ATM-TCR | 独立 env `tcrbench_atmtcr`（torch1.10+torchtext0.11）；wrapper 把 `LD_LIBRARY_PATH` 隔离到本 env 的 lib，避免主 env libtorch 串味 |
| NetTCR-2.2 | 独立 env `tcrbench_nettcr`（Keras/TF）；同样隔离 `LD_LIBRARY_PATH` |
| TEIM | **主 env**（torch2.8+PL1.8.6）跑通；官方 ckpt 经 `torch.load(weights_only=False)` 载入，CDR3 IMGT numbering 用主 env 的 ANARCI+hmmscan |
| ERGO-II | **主 env** 经框架兼容 shim 跑通（PL-0.7 → PL1.8.6），`Original_model/ERGOII_vdj` 权重 0 missing/0 unexpected；AE 权重需 `ln -s Models/AE TCR_Autoencoder`；CPU 推理（`CUDA_VISIBLE_DEVICES=""`） |
| PanPep | PyTorch few/zero-shot（主 env） |
| sceptr / libtcrlm | 主 env 直接装通（`pip install --no-deps sceptr libtcrlm tidytcells blosum`）；torch 前置 `LD_LIBRARY_PATH=$ENV/lib`（CXXABI） |
| tcrdist3 | 主 env 直接装通（`pip install tcrdist3`）；numba/parasail 均落地成功 |
| ProtBERT / TCR-BERT | 主 env transformers 直接载（无需装包）；官方 HF 权重经 `HF_ENDPOINT=https://hf-mirror.com` 下载，torch 前置 `LD_LIBRARY_PATH=$ENV/lib` |
| TCR-VALID | **独立 env `tcrvalid`**（`conda create -n tcrvalid python=3.8`；`pip install tensorflow-cpu==2.13.1 mlflow-skinny numba weblogo biopython "pandas<2" "numpy<1.25" scipy scikit-learn matplotlib Pillow` 后 `pip install -e baselines/TCR-VALID --no-deps`）。requirements 里 TF2.8 的 `tf-estimator-nightly` 已从 PyPI 下架，故用 TF-cpu 2.13；keras h5 权重照常载入，复现 README 参考 latent。`scripts/tcrvalid_embed_cluster.py` 用 `conda run -n tcrvalid` 跑，产 latent 后回主 env 跑 embed-threshold。 |
