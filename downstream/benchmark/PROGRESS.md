# IRBench 过程记录

## 2026-07-21 T1 retrained checkpoint 五折审计

- Figshare `retrain.zip` / `Retraining_model.zip` 已完整下载并按发布 MD5 校验；新增 retrained protocol/runner、`precrec` PR-AUC port、checkpoint 嵌套提取与逐层 CRC/SHA provenance。
- 8 个模型（ATM-TCR、NetTCR、epiTCR、TEIM、TCR-H、TEINet、ERGO-AE、ERGO-lstm）完成 AS CDR3β-only 的 5 folds × 3 eval sets。统一 local↔paper 表：`outputs/tcr_binding_nm2025_retrained/comparison_AS.csv`/`.json`。
- unseen 双指标四位小数同时复现：ATM-TCR、NetTCR、epiTCR、TCR-H、ERGO-AE、ERGO-lstm；TEIM 接近但不相同，TEINet 差约 0.006。所有 seen 格均不精确一致，显示发布 seen 数据与论文内部版本的系统偏移。
- TCR-H 明确记录 inference-time train descriptor filter（无 fit）；ERGO-AE 明确记录官方 50-row batching 丢尾；测试 22 passed。retrained audit 与 canonical original-only 主表严格分离。

## 2026-07-21 T1 external baseline original-only 修复

- External baseline runner 已与 retraining 拆开：只加载 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip` 派生 test，校验固定 sha256，并记录 `train_csv_loaded=false` / `retrained=false`；`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_nm2025.py` 明确保留为 Ours/knn/local probe 路径。
- 数据准备新增 `--tests-only`：不检查、不打开、不改写 `train.csv`，生成 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_binding_nm2025/original_test_manifest.json`。runner 逐文件校验 normalized test SHA256，13 个可运行模型已全部重新推理并写入该哈希；实测两个 train 文件前后 SHA256 不变。
- 15 个论文 original-model baseline 建立 canonical catalog 与来源选择器。当前 13 个已由新 runner 重跑 AS seen/unseen，全部产物内嵌 checkpoint path / variant / original.zip checksum / `retrained=false`；SETE 官方 pickle 推理维数不兼容、TEPCAM wrapper 未接通，自动取论文值并标“论文值，未本地复现”。SETE unseen 论文无值，保持空缺。
- 修复多权重 wrapper 的 tag→checkpoint 时序：TEINet/ERGO/TPBTE 在 import 前设置 canonical variant，避免“目录 tag 正确但实际权重错误”。
- 补上 runtime artifact 对账：每个 wrapper 必须返回推理时真正打开的模型 artifact，runner 与 catalog 逐路径比对，不一致立即回退论文值。TEIM 现记录 `teim_seq.ckpt + epi_ae.ckpt`，PanPep 记录 `model.pt + Content_memory.pkl + Query.pkl`，ERGO-AE 记录 classifier + TCR autoencoder；受影响的 AS seen/unseen 已重跑并刷新 provenance。
- 修复 ERGO inference 原地 tokenization 污染输出 epitope 标识；四个 ERGO 变体 AS seen/unseen 已重跑，metrics 含完整 original-only provenance。
- 新产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`（source decision）、同目录 `summary_main.csv`（消费 canonical decision）、`summary_local_runs.csv`（local diagnostic）、`<tag>/cdr3b/original_run_status.json`（成功/失败状态）。
- 测试：`protenix_abtcr` 环境 15 passed；`pllm` 环境缺 `scikit-learn`，只能完成 py_compile，不能 import 既有 benchmark metrics。

## 2026-06-13 立项

**现状**
- BioSeq 免疫受体基础模型尚未训练；现有 `downstream/` 只有抗体任务。
- 本地数据齐全：`data/tcr/`(VDJdb/McPAS/MIRA/IEDB)、`data/ots_paired_clean`(全长 TCR)、`data/ppi`(STRING 90/90)、`data/downstream/cdr_infilling/tcr`(CDR 10-fold)。
- 坑：`data/downstream/tcr_binding`、`data/tcr_downstream` 是断链/空目录 → 从 `data/tcr/` 重建。

**产出**
- `README.md` 设计总览、`RESULTS.md` 排行榜骨架、目录骨架。

## 2026-06-13 整合现有基准（用户指示）

**可直接整合的现成基准**
- **NbBench**(`data/nanobody_raw/nbbench`)：8 个免疫受体任务 + 9 开源 baseline + head-only probing 脚本，数据齐全 → 作为框架范式与抗体维度。
- **TCRpMHCdataset/PIRD**(`data/tcr/PIRD_alt`)：自带 epitope/TCR 防泄露 split → TCR 数据与切分引擎。
- **STRING 90/90**(`data/ppi`)：PPI gold standard，直接用。
- **IMMREP**：TCR binding 权威 benchmark，待对接。

**策略**：复用 NbBench 框架，把 TCR/PPI 接进同一范式，新增 BioSeq 模型 wrapper。

## 2026-06-13 baseline 原则与调研（用户指示）

**原则**：只用作者 report 的官方代码 + 官方权重，不自行复现；无公开代码者不收录。

**调研确认（均有官方 git + 权重）**
- TCR binding 现成 benchmark：**IMMREP23** `justin-barton/IMMREP23`（seen/unseen 二分类, 带标签）、**IMMREP22** `viragbioinfo/IMMREP_2022_TCRSpecificity`（全长 TCR）。
- binding baseline：NetTCR-2.2 `mnielLab/NetTCR-2.2`、ERGO-II `IdoSpringer/ERGO-II`、pMTnet `tianshilu/pMTnet`、epiTCR `ddiem-ri-4D/epiTCR`、PanPep `bm2-lab/PanPep`。
- 统一接口参考 `ePytope-TCR`。
- 本地未发现现成 TCR baseline 代码（AirGen-Dev 仅抗体任务），故 baseline 从官方 git clone。
- GLIPH2 仅 web server，暂不收录。

## 2026-06-13 clone 现成基准与 baseline

- 网络：经 socks5 代理可访问 GitHub。
- 已 clone（commit 见 `baselines/README.md`）：
  - 现成 benchmark：IMMREP23、IMMREP_2022_TCRSpecificity（后者自带官方 `evaluation/` + 17+ 方法 `methods_results/`）。
  - binding 官方 baseline（含权重）：NetTCR-2.2、ERGO-II、epiTCR、PanPep。
- IMMREP 数据含**全长 TCRα/β + CDR1/2/3 + V/J + peptide + HLA + Label**，既适合 foundation model 也适合传统 baseline。
- pMTnet（TF1.x）待单独处理。

**下一步**
- 写 `common/`(metrics/schema/model_api) + IMMREP 数据加载与评测脚本。
- 整合 IMMREP22 `methods_results` 官方结果到 `RESULTS.md`。
- 待用户确认：binding 粒度、GPU 时机、优先级。

## 2026-06-13 框架落地 + 五任务全部跑通（真实数字）

**环境**：`/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`（torch 2.8.0 + sklearn + 1×GPU）。
torch 在并发 pip 后偶发 `libstdc++ CXXABI_1.3.15` 链接错误，GPU 任务前置 `LD_LIBRARY_PATH=$ENV/lib`。

**`common/` 框架（全部实现并冒烟通过）**
- `schema.py` 规范记录 + IMMREP23 加载器 + JSONL/预测 IO；`clonotype_key`(CDR3a|CDR3b)。
- `leakage.py` 编辑距离(rapidfuzz)、`epitope_holdout_split`/`seen_epitope_split`、`dedup_against`、`leakage_report`。
- `negatives.py` 参考 TCR 负采样（Lev>3 距离守卫、不复用正克隆型）。
- `metrics.py` binding(per-epitope AUROC/AUPRC/Macro-AUC0.1)、clustering(purity/retention/NMI/ARI)、representation(linear probe + kNN)、generation(AAR/novelty/NN/k-mer JSD)。
- `model_api.py` 统一 `SequenceEmbedder` 接口 + `ESM2Embedder`(本地快照)/`OphiuchusEmbedder`/`BioSeqEmbedder`(基础模型接入点) + `build_embedder` 工厂。
- `featurizers.py` `DistanceKNNScorer`(TCRdist 式)、`KmerFeaturizer`。

**数据准备脚本（`scripts/`）**：`prepare_tcr_binding.py`、`prepare_tcr_repertoire.py`(T2+T3)、`prepare_tcr_generation.py`、`prepare_ppi.py`、`import_immrep22_results.py`。

**防泄露（已落地并断言）**
- T1：剔除与测试克隆型重叠的 60 条训练正样本 → clonotype/pair overlap=0；测试按表位是否见过分 seen(13)/unseen(7)。
- T3：克隆型隔离 split，overlap=0。
- P1：STRING 90/90 → train/test 蛋白重叠=0。
- 关键证据：原始 IMMREP23 train↔test 有 46 克隆型 / 39 pair 重叠（真实泄露），去重后归零。

**真实结果（详见 `RESULTS.md`，均可由 `outputs/` 复现）**
- T1 binding（主排名 unseen）：CDR3-kNN unseen AUROC=0.500 / seen=0.694；ESM2-150M probe ~0.47；Random ~0.5。**距离法无法泛化到新表位**。
- T2 clustering：editdist(t=1) Purity=0.800/NMI=0.475；ESM2 agglomerative Purity=0.273。
- T3 representation（24-way）：ESM2-150M probe-AUROC=0.806/Acc=0.438；k-mer 0.800/0.430（随机 0.04）。
- T4 generation：Markov(3) JSD=0.041(最优)、PWM 0.176；novelty 均 >0.97。
- P1 PPI（90/90）：k-mer+Hadamard AUROC=0.563（严格防泄露下任务变难）。
- 外部参考：IMMREP22 官方 22 方法 MicroAUC 表 → `outputs/external/immrep22_official.csv`。

**基础模型接入点**：训练完成后只需 `build_embedder("bioseq:/path/final.pt")`，各任务 `--embedder bioseq:...` 即并入排行榜（`Ours-BioSeq` 行）。

**结论**：五个任务族（binding/clustering/representation/generation/PPI）均端到端跑通、有真实可复现数字、防泄露断言通过。baseline 仅用公开数据/代码（IMMREP/STRING/OTS/VDJdb + ESM2 本地权重 + 训练free 经典法）。

## 2026-06-14 下游完善（不涉及训练）

**新增**
- `baselines/wrappers/epitcr.py`：epiTCR 官方 RF 薄封装（BLOSUM62 编码内联，无 imblearn 依赖）；`tcr_binding/run.py --method epitcr`。
- `nbbench/`：NbBench A1 维度统一 runner（8 个标量标签任务，冻结 ESM2 + 线性 probe，无主干微调）。
- `scripts/epitcr_infer.py`：sklearn 版本隔离推理辅助脚本。

**已知限制**
- epiTCR pickle 需 sklearn~=1.2（主 env 为 1.6.x，待建独立 env 后出数）。
- VRClassification / CDRInfilling / Paratope 需位点级或生成式 runner，列入后续扩展。

**下一步**
- 按 `TCR_TASK_TAXONOMY.md` 精简：T4 CDR infill 接入、T1 官方 baseline 出数、不新增冗余 binding 榜。
- clone tcrdist3 作 T2 官方聚类基线。
- 建 epiTCR 独立 conda env（sklearn 1.2）补齐 T1 官方 baseline 行。

## 2026-07-02 文献驱动重构：新增 T5 Design + 全量 baseline sweep（不涉及训练）

**背景（用户指示）**：PPI 仿 MINT(protein)、抗体仿 Ophiuchus-Ab、TCR 从头搭 benchmark（binding/representation/clustering/generation + **TCR design**）；baseline 只用官方源码；先写并**实时更新** MD。

**文献调研（2024–2026 高引/前沿，详见 `TCR_BENCHMARK_DESIGN.md`）**
- T1 Binding：**Nature Methods 2025《Assessment of computational methods...》**（50 模型/21 数据集）→ 主指标改 **unseen macro-AUPRC**；负样本来源(AS/PS/HS)与克隆型泄露影响 > 模型结构。IMMREP23 / ePytope-TCR / arXiv2606.04994(AUC0.1)。
- T2 Clustering：**NAR-GAB 2025** 9 方法（Purity/Retention/Sensitivity）；GLIPH2 仅 web → 不收录。
- T3 Representation：**SCEPTR (Cell Systems 2024)** few-shot probe+kNN。
- T4/T5 Generation：**TCRT5 (Nat Mach Intell 2025)** / TcrDesign / TCR-epiDiff（扩散）/ OLGA·SONIA。

**新增文档**
- `TCR_BENCHMARK_DESIGN.md`：五榜（T1–T5）主设计 + 文献依据 + 源码-only baseline 清单（含待 clone）+ 实时状态 + changelog。**本文件（`TCR_TASK_TAXONOMY.md` 的升级版）为 TCR 基准主设计入口。**

**新增任务 T5 — 表位条件 TCR Design（从零实现并跑通）**
- `scripts/prepare_tcr_design.py` → `data/tcr_design/`：IMMREP23 VDJdb 按表位分层 **common 13 / rare 24 / novel 7**（novel 完全 held-out = zero-shot）。防泄露断言：novel-in-train=0、ref∩conditioning CDR3β=0。
- `common/metrics.py::{design_metrics, aggregate_design}`：recovery / exact-match / mean_nn_to_ref / k-mer JSD / novelty / diversity，按 common/rare/novel 分层。**oracle 校验 exact=recovery=1.0 / JSD=0**。
- `tcr_design/run.py`：训练free baseline `markov` / `cond_pool` + `--method file`（JSONL 每行 `{"epitope","sequences"}`）接入我们模型。

**全量 baseline sweep（`scripts/run_all_baselines.sh`，全部 exit=0，结果入 `RESULTS.md`）**
- **T1**: Random / CDR3-kNN / ESM2-150M / **epiTCR 官方 RF**。unseen-AUPRC = 0.192 / 0.167 / 0.189 / **0.195**；seen-AUROC kNN 0.694、epiTCR 0.612。
- **T2**: editdist Purity 0.800 / ESM2 0.273。**T3**: Ophiuchus **0.825** > ESM2 0.806 > kmer 0.801（probe-AUROC）。
- **T4**: Markov3 JSD 0.041(最优)/Markov2 0.042/PWM 0.176。**T5**: Markov3 & CondPool recovery≈0（下限），oracle 1.0。
- **PPI**: k-mer AUROC 0.562 / **ESM2-150M 0.815**（AUPRC 0.826）。
- **A1/NbBench(8 任务全跑)**: nanobody_type acc 0.9997、hIL6 AUROC 0.898(n_test449k)、polyreaction 0.865、SARS 0.868、hTNFa 0.779、thermo-tm ρ0.658、thermo-seq ρ0.625、vhh_affinity ρ0.165。

**关键修正 / 发现**
- ✅ **epiTCR 官方 RF 在主 env（sklearn 1.6）直接跑通**（原以为需 sklearn 1.2 独立 env）——unseen 主榜目前最佳但仍近随机，印证 Nat Methods 2025 结论。
- ✅ T1 主指标口径由 AUROC 切换为 **unseen-AUPRC**（对齐 Nat Methods 2025）。
- ⏳ 唯一 pending = `Ours-BioSeq`：`output/grammar_v2_*`（esm2_650m/esmc600m/esmc300m/no_encoder_1b）已有 `best.pt`，`--embedder bioseq:/abs/final.pt`（或 design `--method file`）接入即出全部对比数字。

**下一步**
- 接入 `grammar_v2_*` checkpoint 到 TCR 四大方向（T4 含无条件/表位条件两 setting）+ PPI + NbBench，填 `Ours-BioSeq` 行。
- clone 官方 baseline 补齐：**T3 TCRdist/Levenshtein-NN（主对照，优先）**、T2(clusTCR/GIANA/tcrdist3/DeepTCR)、T3(SCEPTR/TCR-BERT)、T4 无条件(OLGA/SONIA)、T4 表位条件(TCRT5/TcrDesign/TCR-epiDiff)、T1(NetTCR-2.2/ERGO-II/PanPep)。
- PPI 落地 MINT 迁移（`../MINT_MIGRATION.md`）。

## 2026-07-03 分类收敛：T5 并回 T4（Generation 一个方向、两个子任务）

**背景（用户指示）**：generation 与 design **本质是同一件事——都是生成任务**，之前拆成 T4/T5「五榜」口径有问题，需 fix。

**结论与改动（仅文档口径，零代码/数据/指标变更）**
- 重新明确 **TCR = 四大方向：T1 Binding / T2 Clustering / T3 Representation / T4 Generation**。
- **T4 Generation 内含两个子任务**（都是"采样生成 CDR3β 序列"，共享生成器主干与序列级指标 k-mer JSD/novelty/NN 距离，差异仅"是否给定表位条件"）：
  - **T4a 无条件分布匹配** → `tcr_generation/run.py`（主指标 JSD↓）。
  - **T4b 表位条件设计**（"TCR design"）→ `tcr_design/run.py`（主指标 recovery + exact-match，common/rare/novel 分层）。
- 同步更新 4 份 MD：`TCR_BENCHMARK_DESIGN.md`（§0 表 + §2 章节改 T4a/T4b + §3 baseline 表 + changelog）、`README.md`（状态行 + §3 总览表 + baseline 表）、`RESULTS.md`（T5 章并入 T4 作 T4a/T4b 两子节）、`TCR_TASK_TAXONOMY.md`（T4 表补 T4b、标注"已实现"）。
- **物理目录/脚本保持不变**：`tcr_generation/`（无条件）与 `tcr_design/`（表位条件）仍各自独立跑通，`outputs/` 结果与指标口径完全不动。

## 2026-07-03 广泛文献复核：确认四类 + generation 收为"方向+setting" + 诚实列构建局限

**背景（用户指示）**：感觉 TCR benchmark 构建仍有问题，要求**广泛再调研**；重申仍是四类（binding / clustering / representation / generation），条件/无条件生成只是**子分类（setting）**。

**调研（2024–2026，均查证原文）**
- Binding：**Nat Methods 2025**（50 模型/21 库/762 表位）→ 主指标 AUPRC、**CD-HIT 去相似**、**负样本来源 AS/PS/HS 影响 > 模型结构**、unseen 近随机；arXiv 2606.04994 用 **Lev≤3 去重剔除 40–70% test**。
- Clustering：**NAR-GAB 2025** 统一 **190,670 TCR / 2313 表位**，9 方法，**Purity–Retention 权衡**（DeepTCR 高保留，clusTCR/TCRMatch/GLIPH2 高纯度）。
- Representation：**SCEPTR (Cell Systems 2024)** few-shot per-epitope NN AUROC；**通用 PLM 打不过 TCRdist / CDR3-Levenshtein**，对比学习才反超。
- Generation：**TCRT5 (Nat Mach Intell 2025)** SeqRec%/F1@K；**TcrDesign / LSMTCR** 全长条件设计；**TCRTransBench** 双向 TCR↔pep；OLGA/SONIA 无条件 Pgen。PLM 综述 **Int Immunol 2025 dxaf048** 佐证四类划分。

**结论**
- **四大方向划分正确、无第五类**；generation 的条件/无条件是**同一方向的两种 conditioning setting**（进一步去掉上一条的"T4a/T4b 子任务"措辞）。
- 但发现**非命名类的实质构建短板**（写入 `TCR_BENCHMARK_DESIGN.md` §7）：
  - **P0-L1** T1 用精确克隆型去重，缺 **CD-HIT/Lev≤3 近似去重** → unseen 分数可能虚高。
  - **P0-L2** T1 负样本仅 AS 一种，未按 Nat Methods 2025 显式标注/对照 **AS/PS/HS** 来源。
  - **P0-L4** T3 缺 **TCRdist / Levenshtein-NN 主对照**（SCEPTR 证明通用 PLM 打不过它，无此基线结论失真）。
  - **P1-L3** T2 仅 24 表位/8609 条，远小于 190k/2313；应扩规模 + 报权衡曲线。
  - **P1-L5** T4 无条件缺 OLGA Pgen 校准；可补双向/全长扩展。
  - **P1-L6** `Ours-BioSeq` 全 pending、官方深度 baseline 多数待 clone。

**改动**：更新 `TCR_BENCHMARK_DESIGN.md`（§0 四类表 + generation 两 setting、§1 补 LSMTCR/TCRTransBench/综述、§2 T4 合一含 setting A/B + T3 强调 TCRdist/Lev-NN、§3 baseline 表、**新增 §7 局限清单**、changelog）、`README.md` / `RESULTS.md` / `TCR_TASK_TAXONOMY.md`（措辞由"T4a/T4b 子任务"改为"两种 setting"）。**仍为文档层，`outputs/` 数字不变**；实质修复（L1/L2/L4…）留作后续代码任务。

## 2026-07-03 NbBench(A1) 补齐：12 任务全跑通 + k-mer 对照 + 官方参照对齐（不涉及训练）

**背景（用户指示）**：继续完善下游 **NbBench（nb）** 这块的结果。

**产出（全部 exit=0，结果入 `outputs/nbbench/` 可复现）**
- **补齐缺失 3 个位点级/生成任务**（`nbbench/run_residue.py`，ESM2-150M 逐残基表征 + 轻量 head）：
  - **VRClassification** acc **0.9989**（macro-F1 0.9983, n_pos=345679, 4 区）→ 与官方 ESM2-150M 0.9989 **完全一致**。
  - **CDRInfilling** aar_masked **0.3959**（masked 位精确恢复, n_masked=89440）→ 对齐官方 CDRs-EM 0.402。
  - **Paratope** auprc 0.6817 / **auroc 0.9231**（≈官方 0.922）。
- **补齐序列级 vhh_affinity-seq**：ESM2-150M spearman **0.1953**（≈官方 0.170，本地略高）。
- **新增 k-mer(3) 训练free 对照**（`run.py --embedder kmer`，9 标量任务全跑）。
- **接入官方 NbBench 参照**（*Mach. Learn.: Sci. Technol.* 6(4):040502, 2025, DOI 10.1088/2632-2153/ae20ec；= arXiv:2505.02022 v1, Tables 5–8，3-seed mean）：`RESULTS.md` A1 改为 **ours(ESM2-150M/k-mer) vs 官方 ESM2-150M vs 官方最佳(11 模型)** 主表 + 逐任务全指标表 + 复现命令。

**关键发现 / 诚实标注**
- ✅ **口径交叉验证通过**：VRCls / NbType / Paratope-AUROC / CDRInf-EM / affinity-seq 我们的 ESM2-150M 与官方几乎重合 → 数据加载与评测口径正确。
- ⚠ **thermo 近似重复告警**：`thermo-tm` 52% / `thermo-seq` 32% 测试样本到训练集编辑距离≤2（NbBench 75% 相似度切分，非严格去冗余）。**k-mer 因此在 thermo 虚高**（spearman 0.836/0.733，超官方所有 PLM）——是数据近似重复而非表征优势，已在 `RESULTS.md` 明确标注；回归可信信号以 **affinity**（无此问题）为准。

**改动**：`nbbench/README.md`（12 任务分序列级/位点级两表 + 数字 + thermo 告警）、`RESULTS.md`（A1 节重写为官方对照主表）、重建 `outputs/nbbench/_summary_{esm2_150m,kmer,residue_esm2_150m}.json`。**A1 全 12 任务闭环，唯一 pending 仍是 `Ours-BioSeq`**（`--embedder bioseq:/abs/final.pt` 接入即出对比行）。

## 2026-07-03 NbBench(A1) 任务代码持续完善：BR 指标 + one-hot 位点级对照 + 官方数字固化（不涉及训练）

**背景（用户指示）**：继续完善**下游任务代码**（持续性工作）。

**代码改动（`common/metrics.py` + `nbbench/run_residue.py` + `scripts/`）**
- **`infilling_recovery` 补 BLOSUM62 Recovery(BR)**：新增 `blosum62_score`/`_blosum62`（Biopython 标准 BLOSUM62，无依赖时 BR→NaN 优雅降级），CDRInfilling 主指标由 EM 切换为 **BR（对齐官方主榜口径）**，同时保留硬恢复 EM(`aar_masked`)。单元校验：perfect→BR=平均自替换分、W→F 相似替换给部分分，口径正确。
- **位点级训练free 基线 `OneHotResidueEmbedder`**：滑窗 ±W 逐残基 one-hot（纯局部组成、无预训练上下文），`run_residue.py` 新增 `build_residue_embedder` 工厂（`onehot` / `onehot:W`）。是标量 `kmer` 基线的**位点级类比**，隔离"预训练主干 vs 局部 motif"的增益。
- **`scripts/import_nbbench_results.py`**：把官方 NbBench 主表（*Mach. Learn.: Sci. Technol.* 6(4):040502, 2025, DOI 10.1088/2632-2153/ae20ec；= arXiv:2505.02022 v1 Table 5，11 模型×11 任务）**转录固化**为 `outputs/external/nbbench_official.csv`（只转录不重算），使 `RESULTS.md` 的"官方 ESM2-150M / 官方最佳"列**可追溯**。

**结果（真实、可复现）**
- **位点级 ESM2 vs one-hot 对照**：VRCls acc 0.9989 vs 0.871、Paratope AUPRC 0.682 vs 0.449（AUROC 0.923 vs 0.792）、CDRInf BR 1.467 vs 1.124（EM 0.396 vs 0.329）。→ **三项 ESM2 均显著胜出**，证明位点级任务需预训练上下文，为基础模型留出真实提升空间。
- **CDRInf 双口径交叉验证**：ours BR **1.467** ≈ 官方 ESM2-150M **1.499**；EM 0.396 ≈ 官方 CDRs-EM 0.402 → 口径正确。

**改动**：`common/metrics.py`（BR）、`nbbench/run_residue.py`（onehot + BR primary）、`scripts/import_nbbench_results.py`（新）、`RESULTS.md`（A1 加位点级对照子表 + BR 口径注 + 官方 CSV 追溯）、`nbbench/README.md`（BR/onehot 说明 + 命令）。基础模型接入点扩到位点级：`run_residue.py --embedder bioseq:/abs/final.pt`。

## 2026-07-05 T3 重构：以 SCEPTR 为权威口径落地 few-shot 主协议 + 主对照 + 接入我们的模型

**背景（用户指示）**：以 **SCEPTR (Cell Systems 2024)** 为权威基础重构 T3 "TCR representation"，跑通 baseline 再接入我们的模型；这是 §7 局限清单里 **P0-L4**（T3 缺 few-shot 口径 + 缺 TCRdist/Levenshtein-NN 主对照）。全程真实可复现，禁止编造。

**环境 / 安装（主 env `protenix_abtcr`，torch 前置 `LD_LIBRARY_PATH=$ENV/lib`）**
- **SCEPTR 装通**：`pip install --no-deps sceptr==1.2.0 libtcrlm==1.1.3 tidytcells==2.2.1 blosum==2.2.0`（清华源，秒级；离线权重已在 `baselines/sceptr/src/sceptr/_model_saves/`）。clone commit **8c1fa02ec4**（`_download_manifest.tsv`）。
- **TCRdist 装通**：`pip install tcrdist3==0.3`（拉入 parasail 1.3.4 / pwseqdist 0.6 / olga 1.3.0 / numba 等，均成功）。**无 blocker**。

**新增 / 改动代码**
- **`common/fewshot.py`（新）**：SCEPTR few-shot per-epitope NN AUROC 引擎。三后端（native cdist / embedding-cosine / editdist）统一成 query(test)×ref(train) 距离矩阵；协议 shots k∈{1,2,5,10,20,50,100}、R=5 seed、min-NN 取负打分、表位内 seed 均值→24 表位 macro(mean±std)、train binder 不足 k 的表位跳过（如实报 n_epitopes）。**oracle 自检**（支持集含 query 正样本→AUROC≈1.0）：mean **1.0000** / min **0.9997**（`python -m common.fewshot`）。
- **`common/model_api.py`**：新增 `SceptrDistance` / `TcrdistDistance` / `build_distance_source`（native cdist 源，tidytcells 标准化 junction+V/J，非功能 V→partial 或占位）；**修复 `BioSeqEmbedder` 支持 ESMC 骨架**——`EsmcBioSeqEmbedder` 自动识别 `grammar_v2_*_llada` 检查点，载微调后 `encoder.esmc.*`（**0 missing / 0 unexpected 键**，真加载非随机）→ mean-pool；旧 Ophiuchus 检查点走 `OphiuchusBioSeqEmbedder`。
- **`tcr_representation/run.py`（重写）**：`--protocol {fewshot,probe,both}`（默认 both）+ `--method {levenshtein,tcrdist,sceptr,embed}`（embed 配 `--embedder`）+ `--shots/--seeds/--columns`。few-shot→`fewshot.json`，probe 辅报→`metrics.json`（向后兼容）。
- **`scripts/summarize_tcr_representation.py`（新）**：汇总所有 `fewshot.json`/`metrics.json` → 主表 + `outputs/tcr_representation/_summary.json`。

**真实结果（`outputs/tcr_representation/*/fewshot.json`，few-shot AUROC）**
| method | k=1 | k=5 | k=20 | k=100 |
|--------|-----|-----|------|-------|
| **SCEPTR** | 0.587 | 0.655 | **0.711** | **0.741** |
| TCRdist | 0.577 | 0.644 | 0.687 | 0.728 |
| Levenshtein-NN | 0.555 | 0.597 | 0.641 | 0.683 |
| k-mer(3) | 0.543 | 0.597 | 0.643 | 0.673 |
| ESM2-150M | 0.561 | 0.589 | 0.638 | 0.671 |
| Ophiuchus-Ab | 0.576 | 0.612 | 0.664 | 0.677 |
| **Ours** esmc300m-cmp500k | 0.579 | 0.617 | 0.667 | 0.682 |
| **Ours** esmc600m-cmp500k | 0.560 | 0.602 | 0.648 | 0.668 |
| **Ours** esmc300m-mint | 0.560 | 0.589 | 0.639 | 0.661 |

- **复现 SCEPTR 核心方向**：**SCEPTR ≥ 序列比对法(TCRdist/Lev) ≥ 通用 PLM(ESM2)**（每个 shots 稳定成立；TCRdist 逼近 SCEPTR）。
- **Ours-BioSeq 接入成功、无 blocker**：3× grammar_v2 均 0 missing/0 unexpected；embedding 类里 esmc300m-cmp500k 最强（k=20 0.667，>Ophiuchus/ESM2），但仍低于比对法与 SCEPTR——诚实标注：扩散自监督 mean-pool 表征未针对表位特异性优化。probe-AUROC 辅报 0.825（并列最高）。

**遗留项**：TCR-BERT 待 clone（manifest DL_FAIL，非阻塞）；T1/T2/T4 接入 Ours-BioSeq + 官方深度 baseline。

**改动文件**：`common/fewshot.py`(新)、`common/model_api.py`、`tcr_representation/run.py`、`scripts/summarize_tcr_representation.py`(新)、`RESULTS.md`、`TCR_BENCHMARK_DESIGN.md`、`baselines/README.md`、`PROGRESS.md`；产出 `outputs/tcr_representation/{levenshtein,tcrdist,sceptr,esm2_150m,kmer,ophiuchus,bioseq_grammar_v2_*}/fewshot.json` + `_summary.json`。

## 2026-07-05 T3 补齐官方 baseline：ProtBERT + TCR-BERT（凑齐 SCEPTR Table SI 六模型对照）

**背景（用户指示）**：给 T3 补齐 SCEPTR (Cell Systems 2024) Table SI 中尚缺的两条通用/专用 LM 对照，凑齐六模型（SCEPTR / TCRdist / CDR3-Lev / ESM2 / ProtBERT / TCR-BERT）。原则：baseline 只用官方源码/权重，薄封装（读数据→官方 API→出表征），不复现模型逻辑。用现有 split（24 表位 / train 6887 / test 1722）。

**权重来源 / 安装（主 env `protenix_abtcr`，`HF_ENDPOINT=https://hf-mirror.com`）**
- **ProtBERT**：官方 `Rostlab/prot_bert`（HF，本地已缓存于 `conda/cache/huggingface/hub`）。`/c20250601/mj/model_weights/` 下无本地快照，走 HF 镜像。空格分隔 AA、mean-pool 末层。
- **TCR-BERT**：官方 `wukevin/tcr-bert-mlm-only`（纯 MLM 预训练，非 `wukevin/tcr-bert` 分类微调权重——后者见过 PIRD 表位标签，用于表位特异性任务会泄漏监督）。HF 镜像下载成功（10 文件，~10s）。空格分隔 AA、64 上下文、mean-pool；按官方用法仅编 CDR3β（`--columns cdr3b`）。
- **catELMo（可选）**：**blocker**——GitHub `Lee-CBG/catELMo` 无代理 clone 超时（60s×多次）；且 ELMo 栈依赖 `allennlp 2.10`（要求 torch<1.13）与主 env torch 2.8 冲突，需独立老环境 + 外部 ELMo 权重。记 blocker 跳过，不阻塞主线。

**新增代码**：`common/model_api.py` 加 `ProtBertEmbedder` / `TcrBertEmbedder`（均 `HFProteinLMEmbedder` 薄子类），`build_embedder` 注册 `protbert`/`tcrbert`；`scripts/summarize_tcr_representation.py` ORDER 增 `tcrbert`/`protbert`。

**真实结果（`outputs/tcr_representation/{protbert,tcrbert}/fewshot.json`，few-shot AUROC）**
| method | k=1 | k=5 | k=20 | k=100 |
|--------|-----|-----|------|-------|
| ProtBERT | 0.550 | 0.587 | 0.627 | 0.651 |
| TCR-BERT | 0.547 | 0.577 | 0.623 | 0.641 |
| （对照）CDR3-Lev | 0.555 | 0.597 | 0.641 | 0.683 |
| （对照）TCRdist | 0.577 | 0.644 | 0.687 | 0.728 |
| （对照）SCEPTR | 0.587 | 0.655 | 0.711 | 0.741 |

probe 辅报：ProtBERT probe-AUROC 0.793 / acc 0.405 / kNN 0.393；TCR-BERT 0.748 / 0.416 / 0.431。

- **复现论文核心结论**：ProtBERT / TCR-BERT / ESM2 的 few-shot AUROC 均 **≤ CDR3 Levenshtein、显著低于 TCRdist/SCEPTR**（k=100：ProtBERT 0.651 / TCR-BERT 0.641 ≤ Lev 0.683 ≪ TCRdist 0.728 < SCEPTR 0.741）。即使 TCR-BERT 为 TCR 专用 MLM，mean-pool 表征也未反超朴素编辑距离——即论文的"通用 PLM（含 TCR-BERT）打不过 CDR3 Levenshtein，只有 SCEPTR 反超"。口径参照 Table SI(k=200,6表位均值)：ProtBert≈0.705 / TCR-BERT≈0.735，本 split 表位更多更严绝对值偏低但排序完全一致。

**改动文件**：`common/model_api.py`、`scripts/summarize_tcr_representation.py`、`RESULTS.md`、`downstream.md`、`TCR_BENCHMARK_DESIGN.md`、`baselines/README.md`、`PROGRESS.md`；产出 `outputs/tcr_representation/{protbert,tcrbert}/{fewshot,metrics}.json` + 重建 `_summary.json`。

## 2026-07-05 T2 文献锚定重建：以 NAR-GAB 2025 / i3-unit 仓库为唯一基础，9 官方 baseline + Ours-BioSeq（不涉及训练）

**背景（用户指示）**：以 **NAR Genomics & Bioinformatics 2025《Benchmarking unsupervised methods for inferring TCR specificity》**为**唯一构建基础**重建 T2 "TCR clustering"（§7 局限清单 **P1-L3**：原仅 24 表位/8609 条）。策略：**直接复用该仓库的 curated 数据集 + 9 种方法预计算聚类输出作官方 baseline**，按论文口径复算并交叉验证，再用阈值式聚类接入 BioSeq 模型。只用官方源码/结果，禁止编造。

**网络 / 数据获取**：GitHub 直连与常见 socks5 端口均不通（env 已 unset 代理），经 `ghfast.top` 镜像下载 tarball。仓库 clone 到 `baselines/TCR_Unsupervised_Benchmark`（commit **`ac767882`**，gitignore）。

**数据重建（`scripts/prepare_tcr_clustering.py` → `data/tcr_clustering/{tcrs.csv,meta.json}`）**：照仓库 `database_processing.R` 复算 curated pooled DB（IEDB+McPAS+VDJdb，CD8/VS=2/AIS>4.3/有 V·J/CDR3 6–23/表位≥2 唯一配对）得 **5,368 唯一配对 / 9,370 唯一单链序列 / 374 表位**。论文口径 4,779/8,395——仓库 README 警告 DB 版本漂移，但 **6 高亮表位配对/链计数与论文逐一完全一致**（GIL 706/484/456…）。**评测单元 = 唯一配对 TCR**（`pair_id`）；chain-separate 方法 β 链簇映射到配对。

**指标扩展（`common/metrics.py::clustering_metrics`，向后兼容）**：新增 **sensitivity**（表位特异性簇 size>3 且 该表位≥2×其他 的覆盖率 macro）、**pct_clusters_purity_gt90**、**pct_seqs_in_high_purity_clusters**、**min_cluster_size 过滤**（>3/>5/>10 复算）；现有 purity/retention/nmi/ari 键名与语义不变。单元校验通过。

**导入 9 官方方法 + 口径交叉验证（`scripts/import_clustering_baselines.py` → `outputs/tcr_clustering/`）**：解析 clusTCR/DeepTCR/GIANA/iSMART/TCRMatch/TCRdist3/GLIPH2 预计算输出（HD/LD 论文现算、同法复现）→ 每方法 pair-universe `assignments.csv`（singleton=-1）+ 论文原生协议 `calibration_report.csv`。**指标口径 6/9 逐一复现论文 Purity/Retention**：

| 方法 | 复算 Ret/Pur | 论文 Ret/Pur | | 方法 | 复算 | 论文 |
|------|-----|-----|---|------|-----|-----|
| clusTCR | 0.09/0.99 | 0.09/0.99 ✓ | | GIANA | 0.25/0.65 | 0.25/0.66 ✓ |
| DeepTCR | 0.92/0.63 | 0.92/0.63 ✓ | | HD/LD | 0.30/0.69 | 0.25/0.66 ~现算 |
| TCRMatch | 0.09/0.82 | 0.09/0.82 ✓ | | iSMART | 0.18/0.75 | 0.25/0.66 ~仓库输出偏小 |
| GLIPH2 | 0.22/0.80 | 0.22/0.80 ✓ | | TCRdist3 | 0.44/0.42 | 0.44/0.42 ✓ |

**阈值式聚类（`tcr_clustering/run.py` 加 `--method precomputed` / `embed-threshold`）**：嵌入 CDR3β→余弦近邻图→阈值 τ 连通分量（一次并查集扫边定 τ 命中目标 retention）→ Purity–Retention 曲线。产出（匹配 retention 取点，`summarize_tcr_clustering.py` → `embed_threshold_summary.csv`）：

| 嵌入 | Pur@ret≈0.19 | Pur@ret≈0.25 | Pur@ret≈0.39 | AUC |
|------|:--:|:--:|:--:|:--:|
| k-mer(3) | 0.931 | 0.842 | 0.664 | 0.420 |
| ESM2-150M | 0.949 | 0.860 | 0.633 | 0.408 |
| TCR-VALID(我们口径) | 0.983 | 0.953 | 0.815 | 0.521 |
| **Ours** esmc300m-cmp500k | 0.964 | 0.912 | 0.661 | 0.441 |
| **Ours** esmc600m-cmp500k | 0.968 | 0.917 | 0.666 | 0.419 |
| **Ours** esmc300m-mint | 0.959 | 0.874 | 0.602 | 0.390 |

- **Ours-BioSeq 已真实接入**（`EsmcBioSeqEmbedder` 0 missing/0 unexpected 键载 `encoder.esmc.*`）：**同 retention 上纯度显著优于通用蛋白 LM**（ret≈0.25 esmc600m 0.917 vs ESM2 0.860 vs k-mer 0.842），略逊专用 TCR-VALID。
- **两范式权衡（与论文一致）**：阈值嵌入法中低保留区比肩 GLIPH2/GIANA、TCRdist3 保留处纯度远超（0.66 vs 0.42），高保留区仍 DeepTCR 独占（ret 0.82/pur 0.63）。编辑距离方法族以官方 **LD**（Purity 0.86/Ret 0.24）为代表。

**补充 baseline TCR-VALID（Nat Commun 2024, `peterghawkins-regn/tcrvalid` commit `475ed964`, Apache-2.0）**：独立 conda env `tcrvalid`（TF-cpu 2.13/py3.8；requirements 的 TF2.8 `tf-estimator-nightly` 已下架，改 TF2.13，h5 权重照常载入并复现 README 参考 latent）。`scripts/tcrvalid_embed_cluster.py`：在同一 curated 数据集上跑（a）原生 DBSCAN 协议（`tcrvalid_native/`，mean_purity 80.5%/ret 36.8% @eps3.0）+（b）导出 16D latent 供我们 embed-threshold（`tcrvalid/`）。诚实标注 CDR2β-CDR3β_core 特征与 NAR-GAB CDR3-only 口径的差异。

**遗留 / 诚实标注**：Ophiuchus-Ab 嵌入逐序列路径极慢（占位抗体权重），本轮未在时限内产出曲线（另两个通用 LM 参照 esm2/kmer 已产出）；ret≈0.08（clusTCR 超低保留区）阈值法不可达，如实留空。

**改动文件**：`scripts/prepare_tcr_clustering.py`(新)、`scripts/import_clustering_baselines.py`(新)、`scripts/tcrvalid_embed_cluster.py`(新)、`scripts/summarize_tcr_clustering.py`(新)、`common/metrics.py`(扩展)、`tcr_clustering/run.py`(重写)、`data/tcr_clustering/{tcrs.csv,meta.json}`、`RESULTS.md`(T2 重写)、`TCR_BENCHMARK_DESIGN.md`(§0/§2/§3/§7 L3 结项)、`downstream.md`(加 T2 节)、`baselines/README.md`(记 i3-unit + tcrvalid repo+commit)、`PROGRESS.md`；产出 `outputs/tcr_clustering/{calibration_report.csv, baseline_comparison.csv, embed_threshold_summary.csv, <method>/assignments.csv, <embed>_thr/curve.csv, tcrvalid_native/*}`。

## 2026-07-05 T2 清理：移除自写 editdist baseline + 明确口径归属（只用官方源码/结果）

**背景（用户强调原则）**：baseline 只用官方源码/官方结果，不用我们自己写的实现。

**改动（其余 T2 成果不变）**
- **彻底移除自写 `editdist(t=1)`**：`tcr_clustering/run.py` 删掉 `--method editdist` 选项、`cluster_editdist` 函数、`--threshold/--min-size` 参数及 `from common.leakage import edit_distance` 导入（`-h` 正常）；删 `outputs/tcr_clustering/editdist_t1/`；`RESULTS.md`/`downstream.md`/`TCR_BENCHMARK_DESIGN.md` 去掉 editdist 行/脚注/命令。**理由**：editdist 是自写实现且与官方 **LD（Levenshtein，来自 i3-unit 仓库预计算输出）重复**——编辑距离方法族以官方 LD 为代表。
- **官方 9 方法来源保持现状**：复用 NAR-GAB 2025 作者仓库（commit `ac767882`）的官方预计算聚类输出，我们只解析 + 按论文口径复算指标，不 clone 各工具重跑；文档已明确标注。
- **明确口径分区**：ESM2/k-mer/TCR-VALID/Ours-BioSeq 的 `embed-threshold` 是**我们统一评测框架**（对无官方聚类输出的模型/嵌入；论文本身也对 ESM/tcrBERT 做 embedding+聚类），文档中标注为"我们的模型/参照嵌入"、**与官方 9 方法分区呈现，不混栏**。
- 自检：`run.py -h` 正常；`rg editdist tcr_clustering/run.py` 无残留；`summarize_tcr_clustering.py` 重跑，`embed_threshold_summary.csv`/`baseline_comparison.csv` 均不含 editdist。

## 2026-07-05 T1 文献锚定：Nature Methods 2025 官方 binding split 迁移 + 文档结项

**背景**：T1 binding 主协议从 IMMREP23 迁移至 **Nature Methods 2025 官方 split**（figshare `original.zip` + GitHub `train.csv`）。主指标改为 **overall（all_values）AUPRC**（非 unseen macro-AUPRC——后者在 40 表位/~17 行每表位时方差过大）。

**已跑通方法**（`outputs/tcr_binding_nm2025/`，cdr3b·AS）：

| Method | Seen AUPRC | Unseen AUPRC |
|--------|:----------:|:------------:|
| ATM-TCR | **0.696** | 0.522 |
| Ours-BioSeq esmc300m+MLP | 0.564 | **0.534** |
| epiTCR | 0.557 | 0.482 |
| kNN | 0.553 | 0.544 |
| Random | 0.512 | 0.503 |
| PanPep | 0.499 | 0.479 |

**结论**：unseen 全体近随机（0.48–0.54），复现论文核心发现；Ours-BioSeq 已真实接入（grammar encoder 0 missing key）。NetTCR/ERGO-II 无 env/输出，skip。

**改动文件**：`downstream.md`（新增 T1 节，T1→T2→T3 排序）、`RESULTS.md`（NM2025 主表 + IMMREP23 降为辅基准）、`PROGRESS.md`。

## 2026-07-05 T4 文献锚定重构：TCRT5 三 setting + 官方 baseline + Ours-BioSeq

**背景（用户指示）**：以 **TCRT5 (Nat Mach Intell 2025)** 为锚重构 T4 Generation，一个方向三种 conditioning setting（A 无条件 / B 表位条件 / C 全长 α/β）。硬性约束：**baseline 只用官方源码/权重**，跑不通标 pending，不用自制近似顶替；BioSeq 主榜用 benchmark14，held20 标注训练泄露。

**已完成**
- 指标：`common/metrics.py` 移植 TCRT5 全套 `tcrt5_*`；`scripts/test_tcrt5_metrics.py` gold 单测 + 对官方 `ModelEvaluator` **42 block 0 mismatch**。
- 数据：`scripts/prepare_tcr_generation_bench.py` → held20 + benchmark14 + `leakage_report.json` + `bioseq_unseen_pmhc.json`。
- Harness：`tcr_generation_bench/{run,scoring,generators,olga_beta,score_official}.py`；BioSeq adapter `downstream/grammar/tcr_generation.py` + `masks.py` 扩展。
- **baseline 审计**（全部官方，无自制近似冒充）：
  - **OLGA**：`baselines/OLGA` → `SequenceGenerationVDJ.gen_rnd_prod_CDR3`（`olga_beta.py`）
  - **SONIA/soNNia**：`baselines/SONIA` → `SequenceGeneration.generate_sequences_post` + bundled `human_T_beta/model.h5`（已从 OLGA fallback 替换为官方 API）
  - **TCRT5**：HF `dkarthikeyan1/tcrt5_ft_tcrdb` + `model.generate()`；benchmark14 另用 `benchmark_data_w_preds.csv` 官方预存
  - **GRATCR/ER**：官方预存预测（`score_official.py`）
  - **TcrDesign**：wrapper 调官方 `tcrdesign_G.py`/`tcrdesign.py` CLI — ⏳ **blocker**：Zenodo 2.48GB 权重下载中 + 独立 env `tcrdesign`(py3.8/torch1.12 CPU)
- **真实跑通结果**（`outputs/tcr_generation_bench/`）：

| Setting | 方法 | 状态 | 主指标 |
|---------|------|------|--------|
| A | OLGA / SONIA / Ours-BioSeq | ✅ | SONIA JSD=0.043 最优；BioSeq JSD=0.295 |
| B | TCRT5 official / GRATCR / ER / OLGA / Ours-BioSeq | ✅ | TCRT5 held20 F1=0.086；BioSeq benchmark14 F1=0, seq-rec=0.368 |
| B | TCRT5 held20 beam | ✅ | `setting_B/tcrt5/metrics.json` |
| C | Ours-BioSeq fulllength | ⏳ | `run_bioseq_all.py` 后台生成 n=500 |
| C | TcrDesign | ⏳ pending | 权重/env |

- 文档：`TCR_GENERATION_BENCHMARK.md`(新)、`TCR_BENCHMARK_DESIGN.md §T4` 更新、`RESULTS.md` T4 重写、`downstream.md` T4 节（T3 排版）。

**遗留 / blocker**
1. Setting C BioSeq fulllength 生成中（~500 对 α/β，max_iter=64，GPU 密集，预计数小时）。
2. TcrDesign Zenodo 权重下载 ~99% + 解压 + 独立 env 冒烟。
3. `_summary.json` 待 Setting C 完成后重跑 `summarize_tcr_generation_bench.py`。

## 2026-07-05 T1 重做执行计划（先出 plan 再执行 · 用户指示）

**背景（用户指示）**：把 T1 TCR binding 下游 benchmark 重新做扎实——充分文献调研（含 2026 新文章）确认/选定开源构建基础论文、正面解决 §7 的 L1（近似去重）/ L2（AS/PS/HS 负样本来源对照），补齐官方 baseline、测我们的模型、实时更新 `downstream.md`。GPU 重活提交 Volc 跑，轻活本地跑，全程真实可复现禁止编造。

**Step 0 · 文献调研结论（已核实原文，见 §1 文献表更新）**
- **主构建基础仍选 Nature Methods 2025《Assessment of computational methods in predicting TCR–epitope binding recognition》**（Lu, Wang, Xu, Xie, Yang, Xu, Suo. Nat Methods 2026, 23(1):248-259；官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark`）。理由：① 唯一同时提供 **官方 curated train.csv + 官方 seen/unseen 独立测试集 + AS/PS/HS 三套负样本 + 50 模型官方权重（`Original_model/`）** 的可复现基座；② 主指标 AUPRC、CD-HIT 去相似、负样本来源(AS/PS/HS)影响>模型结构、unseen 近随机——我们已如实迁移（ATM-TCR seen AUPRC 0.696≈论文 0.70、unseen 0.522≈论文 0.52，口径吻合）。
- **补强口径来自 arXiv 2606.04994（2026, Liao/Li/Jiang/Li/Chen, UMBC·UPenn·CHOP）《New Benchmarking Shows Limited Generalization Power...》**：核实其 ① **Lev≤3 近似去重**（CDR3β 允许≤3 氨基酸替换，剔除 test 中 **40–70%** 近似序列，防记忆型泄露）；② **Macro-AUC0.1**（低 FPR 区 pAUC）辅指标；③ unseen 严格定义为公共库(VDJdb/IEDB)完全未见；④ 复述"epitope novelty 比 TCR novelty 更影响性能"。→ 落地为我们的 **L1 近似去重 + AUC0.1 辅报**。
- **佐证 Sci Rep 2025（Delaunay et al., s41598-025-26454-7）**：published models 仅在与训练表位 **edit distance ≤1（至多2）** 时优于随机；估计需 1–100M 表位才可泛化。支持"按到训练集的编辑距离分层评测"的框架。
- **结论**：NM2025 仍是最合适的主基础，**不替换**；用 arXiv2606.04994 的近似去重/AUC0.1 思路补强（L1），并把 AS/PS/HS 对照做实（L2）。新 2026 文章（arXiv2606.04994 的 TetTCR-SeqHD/Fingerprinting）是 perspective/新实验数据，未打包成带官方 baseline 的可复现 split，故作方法论补强而非新主基座。

**Step 1 · L1 近似去重（本地，训练free，快）**
- 新增 `scripts/nm2025_near_dedup.py`：对 cdr3b track 每个 (split, neg) test，按 rapidfuzz 计算每条 test CDR3β 到 **官方 train.csv 全部 CDR3β**（及仅训练正样本）的最小 Levenshtein 距离；对 k=1/2/3 生成"硬 unseen"子集（移除 minLev≤k 的 test 行）。产出 `data/tcr_binding_nm2025/near_dedup_report.json`（去重比例/n_before/after/pos/neg/表位数）。
- 复算主指标：所有方法在**同一去重子集**上重算 overall AUPRC/AUROC（run_nm2025 系方法按 `id` join test 拿 cdr3b 过滤；官方 baseline 因输出重排→在过滤后的 test.csv 上重跑 wrapper）。产出 `outputs/tcr_binding_nm2025/near_dedup_summary.csv`。**向后兼容：原口径 outputs 不删**。

**Step 2 · L2 AS/PS/HS 负样本来源对照（本地，数据已就位）**
- cdr3b track 的 AS/PS/HS × seen/unseen 六套 test 已生成，各方法 outputs 已含六目录。新增 `scripts/summarize_nm2025.py` 汇总 AS/PS/HS 主指标成表，复现"负样本来源(AS/PS/HS)影响 > 模型结构"，显式说明主榜用 AS 的理由（与论文主表一致、AS 是 refined cross-matching 最少 FN）。

**Step 3 · baseline（官方源码/权重）**
- 已跑通：ATM-TCR / epiTCR(noMHC) / PanPep / **NetTCR-2.2(b)**（envs：tcrbench_atmtcr/epitcr/nettcr + 主 env panpep）。
- 尝试补 **TEIM**（`Original_model/TEIM.ckpt` + `environment/TEIM.yml`）；**ERGO-II**（wrapper 已在，env `tcrbench_ergo2` 未建）。跑不通如实标 blocker（env/权重/依赖哪一环）。kNN/Random 训练free 下限保留。

**Step 4 · 我们的模型（GPU→Volc）**
- esmc300m_cmp500k（`ours_biseq_mlp`）已跑通六套 test。
- 提交 **Volc eval job** 跑 esmc600m_cmp500k / esmc300m_mint 两个 checkpoint 的 `run_nm2025.py --method embed --head mlp`（cdr3b track 六套 test），输出写 vepfs 共享 `outputs/tcr_binding_nm2025/ours_biseq_mlp_{esmc600m,esmc300m_mint}/`。近似去重后 unseen 本地按 id-join 重报。task_id 记于本文件。

**Step 5 · 文档实时更新**
- `downstream.md` T1 节（主协议+基础论文理由、AS 主表、AS/PS/HS 对照、近似去重前后对照、结论、复现命令）、`TCR_BENCHMARK_DESIGN.md`（§1 补 2026 文章、§7 L1/L2 → 已修复+证据）、`RESULTS.md`（T1 主表补对照）、`baselines/README.md`（新增官方 baseline commit/env）、本 `PROGRESS.md` changelog。数字必须与 `outputs/` 一致。

## 2026-07-05 T2 加固执行计划（plan-first · 双基础 · Volc 提交作业 · 用户指示）

**背景（用户指示）**：用户认为 T2 TCR clustering 的**调研不够充分**，要求先做一轮更充分、含 2026 的文献调研，判断 NAR-GAB 2025 作为唯一构建基础是否足够、是否应引入更权威/更新的开源基准或第二篇，然后据此**加固或扩展** T2。硬性原则不变：baseline 只用官方源码/预计算/权重；我们的模型走统一 `embed-threshold` 框架、与官方方法分区呈现；防泄露（表位标签只评测不进聚类）；结果真实可复现禁止编造；文档实时更新。**追加两条**：① 先出结构化 plan 落到文件再执行；② 重算力（BioSeq 全序列嵌入、较大规模聚类/曲线、耗时 baseline）优先**提交 Volc ML 平台作业**（`volc ml_task submit`，走 no-proxy 包装脚本），本地只做轻量解析/指标复算/文档更新。

**Step 0 · 文献调研结论（已核实原文，2024–2026；见 §1 T2 文献表更新）**
- **主构建基础保留 = NAR-GAB 2025《Benchmarking unsupervised methods for inferring TCR specificity》**（lqaf150；`i3-unit/TCR_Unsupervised_Benchmark` commit `ac767882`）。它是唯一同时提供 **curated pooled DB（190,670 TCR/2,313 表位）+ 9 官方方法预计算聚类输出（`Data/Output_methods/`）+ purity/retention/sensitivity 权衡口径** 的可复现基座——**"官方方法级 baseline"的权威来源**，不替换。
- **新增互补第二基础 = Brief Bioinformatics 2025《A comprehensive benchmarking for evaluating TCR embeddings in modeling TCR-epitope interactions》**（Feng, Huo et al., bbaf030, PMID 39883514；官方仓库 `deepomicslab/TCREmbedding` commit `d5bf911`）。它正是 **"embedding→clustering" 评测口径**：固定标注数据集（GIANA 项目 curated，9,033 唯一 CDR3β / 25 抗原表位，≥100/类，`dataset/clustering/TCRantigenData_unique_test.csv`）+ 标准 sklearn 聚类（K-means/spectral/hierarchical，K 从 10 扫到 100 step 5，"以缓解目标簇数影响"）+ **ARI / NMI / Purity** 三指标，评 19 个嵌入方法（含通用 **ESM** baseline + 手工 BLOSUM/理化 + 数据驱动）。**这是 Ours-BioSeq embed 方法最对口的同口径基准**，补齐 NAR-GAB 缺的两点：(1) ARI/NMI 完整聚类一致性指标；(2) 一个**独立于 pooled DB 的第二标注聚类数据集** + 一批"嵌入参照方法"（论文结论：手工 > 数据驱动，通用 ESM 在 CDR3 短序列上不占优——正是我们要对照的方向）。开源程度：官方代码（`pip tcrembedding`）+ 数据（GitHub `dataset/clustering/`，已 API 下载到 `baselines/TCREmbedding/`，gitignore）+ 论文 Supplementary Table 2 报告每方法 ARI/NMI/Purity。
- **其余候选（诚实说明为何不作主基础）**：
  - ImmunoInformatics 2024《A comparison of clustering models for TCR antigen specificity》（`hudsondan/tcr-scapes`）：F1 加权口径 + purity/consistency/retention，5 UCM + Hamming/V-gene/length/random 基线，开源。与 NAR-GAB 方法高度重叠、无预计算输出规模优势，作**文献校准参照**不新建榜。
  - TouCAN（bioRxiv 2024，对比学习 pLM+ESM 聚类）、G2VTCR（bioRxiv 2025，原子级图嵌入聚类）：是**新聚类方法/模型**而非统一基准，各需独立 env/权重；作"嵌入参照候选"，本轮不接入（记为可选扩展）。
  - TCR-VALID（Nat Commun 2024）：已作补充 baseline（独立 TF env，CDR2β-CDR3β_core，原生 DBSCAN + 我们口径 latent）。
  - Feng et al. 里 catELMo/pMTnet 等数据驱动嵌入：与 T3 相同 blocker（allennlp/老 TF 依赖），本轮不接入。
- **相对旧做法的变化**：旧 = 仅 NAR-GAB 单基础、主看 purity–retention 曲线（ARI/NMI 只算未突出、无第二数据集）。新 = **双基础**：基础A（NAR-GAB）继续给 9 官方方法 + purity–retention–sensitivity（**关键结果 1/2 不动，向后兼容**）；基础B（TCREmbedding）新增 **embedding→clustering 的 ARI/NMI/Purity 标准口径 + 第二独立数据集**，Ours-BioSeq 同口径接入，对照 ESM2/k-mer 参照 + 论文报告的嵌入方法（关键结果 3，新增分区）。

**Step 1 · 主基础（NAR-GAB）加固与核验（本地，轻量）**
- 输入：`data/tcr_clustering/tcrs.csv`（5,368 配对/374 表位）+ `baselines/TCR_Unsupervised_Benchmark/Data/Output_methods/`。
- 动作：重跑 `scripts/import_clustering_baselines.py`（校准 6/9，已核验可复现）+ `tcr_clustering/run.py --method precomputed --all`（9 方法全指标含 **ARI/NMI/sensitivity**，已在 `baseline_comparison.csv`）。**把 ARI/NMI 提升为 RESULTS/downstream 明列**（原来偏埋在 CSV）。诚实保留 9/9 校准现状：6/9 逐一吻合，HD/LD/iSMART 因 DB 版本漂移（5,368 vs 论文 4,779）偏移，已在 `calibration_report.csv` + meta 标注（非实现错误，是数据版本；无法在不改数据版本下"凑齐 9/9"，如实说明）。
- 产物：`outputs/tcr_clustering/{calibration_report.csv, baseline_comparison.csv, <method>/metrics.json}`（复现，不变）。

**Step 2 · 第二基础（TCREmbedding）数据准备与脚本（本地，轻量）**
- 输入：`baselines/TCREmbedding/dataset/clustering/TCRantigenData_unique_test.csv`（9,033/25，已下载）。
- 新增 `scripts/prepare_tcrembedding_clustering.py`：解析→`data/tcr_clustering_embed/{tcrs.csv,meta.json}`（列 cdr3b/vgene/epitope/antigen；记录 repo+commit、每表位计数、来源 GIANA）。防泄露：表位标签只评测。
- 新增 `tcr_clustering/run_embed_bench.py`（**我们统一评测框架**，embedding→clustering，与官方方法分区）：对给定 embedder 嵌入 9,033 CDR3β → 标准化 → **K-means（+hierarchical）K∈{10,15,…,100} 扫描** → 每 K 算 ARI/NMI/Purity → 报 mean-over-K + best；写 `outputs/tcr_clustering_embed/<tag>/{curve.csv,metrics.json}`。指标复用 `common/metrics.py`（purity/nmi/ari）；口径对齐论文（连续 K 扫描取均值）。
- 新增 `scripts/summarize_tcrembedding_clustering.py`：汇总各 embedder 的 ARI/NMI/Purity 主表 → `outputs/tcr_clustering_embed/_summary.csv`。

**Step 3 · 计算作业（重活提交 Volc，轻活本地）**
- **提交 Volc 作业**（`eval_jobs/eval_t2_clustering_embed.yml`，1×GPU，no-proxy 包装脚本提交）：在共享 vepfs 上跑 `run_embed_bench.py`，覆盖 **两基础** 的 BioSeq 嵌入+聚类：
  - 基础B（TCREmbedding 9,033）：embedder = `esm2_150m` / `kmer`（参照）+ `bioseq:.../grammar_v2_esmc300m_cmp500k_llada/best.pt` / `esmc600m` / `esmc300m_mint`（Ours ×3）→ ARI/NMI/Purity。
  - 基础A（NAR-GAB 5,368）：如需重算 BioSeq embed-threshold 曲线亦在同作业内跑（已存在则跳过；输出写同一 vepfs 目录）。
  - 产物写 `outputs/tcr_clustering_embed/*` 与（如重算）`outputs/tcr_clustering/*`。日志 tee 到 `output/downstream_generation/eval_t2_clustering_embed_volc.log`。task_id 记于本文件 changelog。
- **本地轻量**：ESM2/k-mer 参照若 GPU 空闲也可直接本地跑（几分钟）；官方 9 方法解析/校准、指标复算、汇总、文档更新全部本地。
- **pending 规则**：若 Volc 排队/失败/依赖缺失，如实标 pending 并说明（不本地阻塞、不编造）。

**Step 4 · 我们模型接入核验（随作业产出）**
- `EsmcBioSeqEmbedder` 载 `encoder.esmc.*`（0 missing/0 unexpected，已验证）；3× grammar_v2 在两基础同口径出 ARI/NMI/Purity（基础B）与 purity–retention（基础A，已存在）。与官方方法分区、与 ESM2/k-mer 参照同栏对照。

**Step 5 · 文档实时更新**
- `downstream.md` T2 节：加"文献调研结论（双基础）"+ 基础B 的 ARI/NMI/Purity 关键表 + 复现命令（基础A 关键结果 1/2 保留）。
- `RESULTS.md` T2：新增 (d) 第二基础 embedding→clustering ARI/NMI/Purity 表；`TCR_BENCHMARK_DESIGN.md`（§1 T2 文献表补 Brief Bioinformatics 2025 + tcr-scapes/TouCAN/G2VTCR；§2 T2 主依据改"双基础"；§7 L3 补第二基础）；`baselines/README.md`（记 `deepomicslab/TCREmbedding` commit `d5bf911`）；本 `PROGRESS.md` 追加执行 changelog。数字必须与 `outputs/` 一致。

**依赖/风险**：GitHub 直连时通时断，数据用 `api.github.com` base64 下载已成功（9,033 行）。Volc 队列可能排队（提交后轮询）。第二基础是 β 链-only（无配对 α），与基础A 配对口径不同——如实标注为"CDR3β 单链聚类口径"，不与基础A 混表。

**执行落地（2026-07-05，同日完成）**
- **Step 0–1 ✅**：文献结论落 §1 T2 双基础表 + §2 T2；基础A 校准 6/9 保持（未改数据版本，如实标注）。
- **Step 2 ✅**：`prepare_tcrembedding_clustering.py` 产 `data/tcr_clustering_embed/{tcrs.csv,meta.json}`（9,033 CDR3β/25 表位/9 抗原，min 110/表位）；`tcr_clustering/run_embed_bench.py`（K-means 主 + PCA-50 Ward 辅，K∈{10,…,100}）；`summarize_tcrembedding_clustering.py`。
- **Step 3 ✅（Volc 提交）**：`eval_jobs/eval_t2_clustering_embed_bioseq.yml` → **task `t-20260705211729-crvhj`**（no-proxy 包装脚本提交，1×`ml.pni2.3xlarge`）；**State=Success, ExitCode=0**。作业内跑 3× BioSeq 嵌入(15–20s/模型)+聚类，写 `outputs/tcr_clustering_embed/ours_*`。ESM2-150M(embed 72.9s)/k-mer(2) 参照本地跑。
- **Step 4–5 ✅**：结果（K-means mean-over-K）——**ours_esmc300m_cmp500k ARI 0.021 / NMI 0.128 / Purity 0.306（全场最高）** > ours_esmc600m_cmp500k(0.018/0.109/0.293) > k-mer(0.010/0.079/0.268) > esm2_150m(0.007/0.068/0.258) ≈ ours_esmc300m_mint(0.006/0.064/0.257)。复现论文"通用 ESM 在 CDR3β 偏弱"，且与基础A 强弱次序吻合（cmp500k>mint）。已同步 `RESULTS.md §T2 (d)`、`downstream.md §T2 关键结果 3`、`TCR_BENCHMARK_DESIGN.md §0/§1/§2/§3/§7`、`baselines/README.md`。数字全部可由 `outputs/tcr_clustering_embed/{_summary.csv,<tag>/curve.csv,metrics.json}` 复现。
- **无编造/无 pending 顶替**：论文 Supplementary Table 2 的逐方法 ARI/NMI/Purity 未复核，故只引其定性结论、不转录数值；参照嵌入仅用官方 HF 权重(ESM2)/训练free(k-mer)，未自造 baseline 数字。

## 2026-07-05 T4 加固执行计划（plan-first · 充分调研含2026 · Volc 提交作业 · 用户指示）

**背景（用户指示，两轮）**：把 **T4 TCR Generation** 下游 benchmark 做扎实。用户认为前期文献调研不够，要求**重新做一次广泛、扎实的调研，尤其纳入 2026 年刚挂出的 TCR 生成相关文章**，据此**选定一篇/多篇有官方开源代码+权重的文章作为主构建基础**（可维持 TCRT5、也可调整/补充，关键是"可复现、开源、权威"），再构建任务→测 baseline→测我们的模型，全程实时更新文档。**追加执行方式**：① 先出结构化 plan 落到文件再执行（本条）；② 重活（生成/大权重下载/长推理）优先**提交 Volc ML 作业**（`volc ml_task submit`，走 no-proxy 包装脚本），本地只做轻量解析/评分/文档。硬性原则不变：baseline 只用官方源码/权重、禁止编造数字、结果可由 `outputs/` 复现、诚实标注泄露与能力边界。

**主线依赖图**：Step 0 文献调研 → Step 1 选型论证（是否维持 TCRT5 / 是否补 TcrDesign 等）→ Step 2 补齐 Setting C（BioSeq 全长 [Volc] + TcrDesign 全长 [本地/独立 env]）与 Setting B（TcrDesign beta CDR3b baseline）→ Step 3 重跑 summarize、口径交叉验证 → Step 4 文档实时更新。Step 0/1 与 Step 2 的下载/生成可并行。

**Step 0 · 广泛文献调研（2024–2026，重点 2026 新挂出，已核实原文 + 官方代码/权重可得性）** — 已完成
调研 10 个候选生成/设计方法（均核实原文 + GitHub/Zenodo 可得性，2026-07-05）：

| 方法 | 出处/年份 | 任务口径 | 架构 | 官方代码 | 官方权重 | 采纳 |
|------|-----------|----------|------|----------|----------|------|
| **TCRT5** (TCR-TRANSLATE) | Nat Mach Intell **2025** (s42256-025-01096-6) | Setting B: pMHC→CDR3β | seq2seq / T5 | `pirl-unc/tcr_translate` `97aeab10` | HF `dkarthikeyan1/tcrt5_ft_tcrdb` + Zenodo 15068617 | ✅ **主构建基础/锚** |
| **TcrDesign** | bioRxiv **2026**.01.15 (XSLiuLab) | Setting C: 全长 epitope→α/β | BERT + Seq2Seq (+ 结构筛) | `XSLiuLab/TcrDesign` | Zenodo 14545852 (2.48GB) | ✅ Setting C 全长 SOTA baseline |
| **TCRDiff** | bioRxiv **2026**.06.10 (SJTU/Monash) | Setting B: pMHC+V基因→CDR3αβ | **条件离散扩散 (DPLM + cross-attn)** | `zhangyumeng1sjtu/TCRDiff` `e5d2ecba` | Zenodo 20586708 (ckpt 195MB + data 536MB) | ✅ **新增 2026 扩散 baseline**（与我们 masked-diffusion 同族，MAGE-A3 体外验证） |
| **TCR-epiDiff** | Bioinformatics **2025** (btaf202) | Setting B: epitope→CDR3β | **DDPM (U-Net + ProtT5-XL 条件)** | `seoseyeon/TCR-epiDiff` | 仓库内 `Model/*.pth` + Zenodo 15094766/67 | ✅ **新增 2025 扩散 baseline**（可跑则接入） |
| GRATCR | 2024 | Setting B: pMHC→CDR3β | BERT+GPT | (官方预存预测) | benchmark_data_w_preds.csv | ✅ 经 TCRT5 benchmark 文件 |
| ER-Transformer | 2023 | Setting B | encoder-decoder | (官方预存预测) | 同上 | ✅ 同上 |
| OLGA / SONIA / soNNia | 2019/2021 | Setting A: 无条件 CDR3β + pgen | 生成式统计 (VDJ 重组/后选择) | `statbiophys/*` | bundled `human_T_beta` | ✅ Setting A 参照 |
| **LSMTCR** | arXiv **2509.07627** (2025) | Setting C: 全长 α/β | Epitope-BERT(扩散增强)+GPT+基因 Transformer | ❌ 无公开仓库(截至 2026-07) | ❌ | ⛔ 仅引用（无代码/权重，不可复现接入） |
| **TCRGen** | bioRxiv **2025** (Lee-CBG) | Setting B: ICL 新表位 | in-context LM | `Lee-CBG/TCRGen` | ❌ "provided soon"(未发布) | ⛔ 仅引用（权重未发布） |
| TCRTransBench | arXiv 2026 | 双向 TCR↔pep 基准 | — | (基准工具) | — | ○ 方法论参照（非生成器） |

**选型论证（写入 `TCR_GENERATION_BENCHMARK.md §0/§1`）**：
- **维持 TCRT5 为"主构建基础/锚"不变**。它是候选里**唯一同时提供**「官方 held-out split（held20 target-rich + 14 sparse benchmark pMHC）+ 其他方法官方预存预测（GRATCR/ER）+ 官方评测代码（`src/evaluation.py`，我们已逐函数对齐**42 block 0 mismatch**）」的**turnkey 可复现基准**。其余方法给的是"方法+权重"（baseline），不是"基准协议"。→ 用 TCRT5 定 Setting B 主榜口径与数据划分最权威、可交叉验证。
- **补两个"最对口的近期扩散 baseline"**：**TCRDiff (2026, 离散扩散 DPLM)** 与 **TCR-epiDiff (2025, DDPM)**。理由：我们的 BioSeq 是 **masked/离散扩散**，与这两者同族；用它们作 Setting B baseline 是对我们模型**最公平、最当代**的对照（回应用户"纳入 2026 新方法"）。二者均有官方代码+权重。
- **补 TcrDesign (2026) 为 Setting C 全长 SOTA**：唯一有官方权重的"全长 epitope-specific"设计方法，且有体外验证。→ Setting C 的权威 baseline。
- **诚实排除**：LSMTCR（无公开代码/权重）、TCRGen（权重未发布）→ 只在文献表引用，标注"不可复现接入"，不编造其数字。
- **验收**：文献表每行可核实（DOI/arXiv/GitHub commit）；主构建基础=TCRT5、新增扩散 baseline=TCRDiff+TCR-epiDiff、全长 baseline=TcrDesign 的选型论证成立。

**Step 1 · 重估主构建基础（选型论证）**
- 判断是否维持 TCRT5 为主锚 + TcrDesign 为全长 SOTA baseline，或补充/替换。给出明确理由（可复现性/开源/权威/与我们扩散模型的对口度）。
- 验收：结论落 `TCR_GENERATION_BENCHMARK.md`；若引入新论文口径/数据/baseline，落到脚本与数据准备。

**Step 2 · 补齐遗留 blocker（Setting C + TcrDesign）**
- 2a **BioSeq Setting C 全长（Volc 提交）**：`eval_jobs/eval_tcr_generation_setC_bioseq.yml` → `scripts/run_bioseq_setC.sh`（生成 n=500 无条件全长 α/β + `scripts/score_bioseq_fulllength.py` 诚实评分：validity + ANARCI 提取 CDR3β → Setting A 分布 JSD/novelty/NN/pgen）。**task_id 记于 changelog**。能力边界诚实：BioSeq 全长只能无条件（ots 源无表位）。
- 2b **TcrDesign 官方 baseline（本地独立 env `tcrdesign` py3.8/torch1.12）**：Zenodo `tcrdesign_weights.tar.gz`(2.48GB) 用 aria2c 多连接下载（原 wget 87KB/s→aria2c 恢复），解压到 `weights/`；冒烟 `tcrdesign_G.py -mode beta`；跑 Setting B（每 pMHC beta CDR3b k=100，官方 `tcrdesign_generate.py --mode beta`）+ Setting C（全长 `tcrdesign.py`，epitope 条件，官方 pipeline）。跑不通如实标 blocker（哪一环）。
- 验收：`outputs/tcr_generation_bench/setting_{B,C}/tcrdesign*/metrics.json` 真实产出；或如实 blocker。

**Step 3 · 重跑汇总 + 口径交叉验证**
- `scripts/summarize_tcr_generation_bench.py` 重建 `_summary.json`（含 Setting C）；`scripts/test_tcrt5_metrics.py` 保持 42 block 0 mismatch；Setting C 新增 validity/分布口径与 Setting A 复用同一 `scoring` 实现（无新指标漂移）。
- 验收：`_summary.json` 三 setting 齐全、数字与各 `metrics.json` 一致。

**Step 4 · 文档实时更新**
- `downstream.md` T4 节（选型论证摘要 + 三 setting 关键表 + 复现命令）、`TCR_GENERATION_BENCHMARK.md`（§0/§1 调研+选型、§7 baseline 表状态、Setting C 口径）、`RESULTS.md` T4（Setting C 补 TcrDesign+BioSeq 真实行）、`TCR_BENCHMARK_DESIGN.md`（§1 T4 文献表补 2026、§7 L5 结项相关）、本 `PROGRESS.md` 追加执行 changelog。数字必须与 `outputs/` 一致。

**执行方式确认**：GPU 重活（BioSeq 全长生成）→ **已提交 Volc**（task `t-20260705205817-8mvdh`）；大权重下载（TcrDesign 2.48GB）→ 本地 aria2c 后台；TcrDesign 推理（CPU/GPU，独立老 env）→ 本地独立 env；文献调研/评分/汇总/文档 → 本地轻量。所有长跑后台+轮询，不空等。

**执行记录（滚动更新）**
- ✅ **Step 0/1 文献调研+选型完成**：维持 TCRT5 主锚 + 补 TcrDesign(2026 全长)/TCRDiff(2026 扩散)/TCR-epiDiff(2025 扩散)；LSMTCR/TCRGen 无公开权重仅引用。写入 `TCR_GENERATION_BENCHMARK.md §0`、`downstream.md T4`、本文件 Step 0。
- ✅ **BioSeq Setting C（Volc `t-20260705205817-8mvdh`，State=Success）**：n=500 无条件全长 α/β + 诚实评分。结果：validAA(β)=0.998 / validAA(α)=0.996，**frac_cdr3β_extracted=0.038(19/499)**，提取 CDR3β JSD=0.846、pgen+=0.789。→ 能力边界：BioSeq 全长合法但难组装真实 CDR3β（重复 Ala），强项在 CDR3 级。新脚本 `scripts/score_bioseq_fulllength.py`（validity + ANARCI 提取 → Setting A 分布复用 `scoring.score_unconditional`）；`scripts/run_bioseq_setC.sh` + `eval_jobs/eval_tcr_generation_setC_bioseq.yml` 一体化生成+评分。`summarize_tcr_generation_bench.py` 增加 Setting C 无条件 validity/分布表。`_summary.json` 已重建。
- 🔄 **下载中**：TcrDesign Zenodo `tcrdesign_weights.tar.gz`(2.48GB，aria2c -x16，~44%)；TCRDiff Zenodo ckpt(195MB ✅)+data(511MB 下载中)。TCRDiff deps（rotary_embedding_torch/warmup_scheduler）已装到隔离 `baselines/TCRDiff/_extra_deps`（不污染 protenix_abtcr），torch2.8 复用主 env。TCRDiff/TCR-epiDiff repo 已 clone（commit e5d2ecba / d7b726ff）。
- ⏳ **待跑**：TcrDesign B(beta CDR3b)+C(full-length)；TCRDiff Setting B（冒烟通过后）。


## 2026-07-06 T4 pending 批量执行（用户「都跑」）

- ✅ **Volc MINT globalfeat**：`t-20260706004005-tqw46`（`eval_mint_ppi_globalfeat.yml`）已提交，State=Running；日志 `output/downstream_generation/eval_mint_ppi_globalfeat_volc.log`；6 格 metrics 根目录 `output/downstream_generation/mint_tasks/<task>/grammar_*_globalfeat_t3000/*_metrics.json`（完成前暂无新 metrics）。
- ✅ **TCRDiff held20**（`run_tcrdiff_baseline.sh held20 100 tcrdiff_held20`）：exit **0**；held20 F1=**0.0118** / seq-rec=**0.732** / div=**0.996** → `setting_B/tcrdiff_held20/metrics.json`；`_summary.json` 已更新。
- ❌ **TcrDesign Setting C full**（`run_tcrdesign_baseline.sh full all 100 tcrdesign_full`）：脚本 exit **0** 但 **0 designs**；blocker：wrapper 调 `TcrDesign/tcrdesign.py` 路径错误（实际为 `baselines/TcrDesign/tcrdesign.py`），34/34 pMHC `tcrdesign.py` exit 1。
- ⚠️ **download_tcr_generation_baselines.sh**：exit **3**；TcrDesign 权重 OK；TCR-epiDiff **Zenodo 15094766 410 GONE**（Best_model.pth 404）；源码已手动从 ghfast zip 同步（`Tutorial_Code/.../TCR-epiDiff_main.py` ✅）。

## 2026-07-06 T4 baseline 接线 + 文档同步

- ✅ **TcrDesign Setting B**：权重+`tcrdesign` env 就绪；34 pMHC k=100 已评分（held20 F1=**0.153** / seq-rec **0.870**）。脚本 `scripts/run_tcrdesign_baseline.sh`。
- ✅ **TCRDiff Setting B (benchmark14)**：smoke + 全 14 pMHC 跑通（F1=**0.0009** / seq-rec **0.587**；pMHC-only 公平口径）。脚本 `scripts/run_tcrdiff_baseline.sh`；`_summary.json` 已更新。
- ⏳ **TcrDesign Setting C**：wrapper 路径 blocker；**TCR-epiDiff**：权重 Zenodo 下架 + wrapper/env 待接。
- 文档：`downstream.md` T4 / `RESULTS.md` T4 / `TCR_GENERATION_BENCHMARK.md` §7–§8 状态表已同步。

## 2026-07-05 MINT GeneralPPI（P0）六格闭环

**权重与提交**
- MINT 官方 `mint.ckpt`（3.25GB）经 hf-mirror 下载至 `downstream/mint_tasks/_mint_weights/mint.ckpt`。
- MINT-official 两格（HumanPPI + Bernett）提交 Volc 作业 **`t-20260705210235-5k5zz`**（`eval_jobs/eval_mint_ppi_benchmark.yml`），**State=Success**。
- ESM2-650M + Ours-BioSeq grammar 四格本地 `_run_bench.sh`（09:40 起，Bernett esm2 抽取 ~14:13 完成 finetune，grammar ~15:08 完成）。

**真实结果（3-rep mean，`*_metrics.json`）**

| 格 | AUROC | AUPRC | Acc | F1 | 位置 |
|----|:-----:|:-----:|:---:|:--:|------|
| HumanPPI × MINT | 0.938 | 0.910 | 0.876 | 0.874 | Volc |
| HumanPPI × ESM2 | 0.939 | 0.941 | 0.851 | 0.855 | 本地 |
| HumanPPI × Ours | 0.904 | 0.880 | 0.831 | 0.830 | 本地 |
| Bernett × MINT | 0.739 | 0.748 | 0.661 | 0.625 | Volc |
| Bernett × ESM2 | 0.786 | 0.767 | 0.709 | 0.686 | 本地 |
| Bernett × Ours | 0.618 | 0.617 | 0.550 | 0.298 | 本地 |

**结论**：HumanPPI 上 Ours AUROC 0.904 略逊 MINT(0.938)/ESM2(0.939)；Bernett 90/90 更难，Ours 0.618 低于 ESM2 0.786 / MINT 0.739，F1 方差大。已同步 `downstream.md` P0 节、`RESULTS.md` P0 子表、`mint_tasks/EXECUTION_PLAN.md` 状态表。

## 2026-07-05 T1 重做执行记录（L1/L2 落地 + 6 官方 baseline + 3×Ours）

**Volc 作业**：`t-20260705204753-5jwgf`（`eval_jobs/eval_tcr_binding_nm2025_ours_extra_ckpts.yml`）—— Ours-BioSeq esmc600m-cmp500k / esmc300m-mint 两 checkpoint 的 embed+MLP，六套 test，State=Success，输出直写共享 vepfs `outputs/tcr_binding_nm2025/ours_biseq_mlp_{esmc600m,esmc300m_mint}/`。

**执行记录**
- ✅ **文献选型**：主基础维持 **NM2025**（官方仓库 `SuoLab-GZLab/TCREpitopeBenchmark` `ec832b47`），补强 **arXiv 2606.04994**（Lev≤k 近似去重 / Macro-AUC0.1）+ **Sci Rep 2025**（编辑距离分层佐证）。2026 新数据集(TetTCR-SeqHD/Fingerprinting)无可复现带官方 baseline 的 split → 方法论补强而非替换。
- ✅ **L1 近似去重**（`scripts/nm2025_near_dedup.py`，本地）：train↔test CDR3β Lev≤k(1/2/3)。去掉 unseen 19.0%/67.4%/92.6%、seen 9.5%/57.3%/87.5%；同硬子集复算所有方法 overall AUPRC。**kNN unseen 0.544→0.487(k≤1) 坍缩**证实距离法吃近重复红利；去重后 unseen 仍≈0.5。产物 `near_dedup_report.json` + `near_dedup_summary.csv`。
- ✅ **L2 AS/PS/HS**（`scripts/summarize_nm2025.py`）：`summary_neg_source.csv`。复现"负样本来源(AS/PS/HS)影响>模型结构"（Ours-cmp500k unseen PS 0.469→HS 0.623，摆动 ~0.15 > 固定 AS 模型间 0.065）。主榜用 AS。
- ✅ **官方 baseline 6 个**：ATM-TCR / epiTCR / PanPep（已有）+ **NetTCR-2.2(b) / TEIM(seq) / ERGO-II(vdjdb)** 新增。
  - TEIM：主 env（torch2.8+PL1.8.6）+ 官方 `teim_seq.ckpt`/`epi_ae.ckpt`；`torch.load(weights_only=False)` + 主 env ANARCI/hmmscan。新 `wrappers/teim.py` + `teim_official_infer.py`。seen/AS 0.676（第二好）。
  - ERGO-II：主 env 经框架兼容 shim（`_install_pl_shims`：PL `data_loader`/`.logging`/可写 hparams + CUDA→CPU）载入官方 `ERGOII_vdj`，**0 missing/0 unexpected**；AE 权重 `ln -s Models/AE TCR_Autoencoder`；修复原 infer 脚本 `output` 变量被循环遮蔽的 bug。seen/AS 0.545（仅喂 CDR3β，原生需 α/V/J/MHC，seen 近随机属预期）。
  - `wrappers/atmtcr.py`/`nettcr.py`：修复子进程 `LD_LIBRARY_PATH` 隔离（防主 env libtorch 串味导致去重重跑失败）。
- ✅ **Ours-BioSeq 3 checkpoint**：cmp500k 已有；esmc600m/esmc300m_mint 经 Volc 作业跑通。encoder 均 0 missing/0 unexpected。unseen overall AUPRC 最高 0.534（cmp500k）。
- ✅ **文档**：`downstream.md` T1、`RESULTS.md` T1（主表+L2+L1+AUC0.1）、`TCR_BENCHMARK_DESIGN.md`（§0/§1/§7 L1/L2→已修复+changelog）、`baselines/README.md`（T1 官方权重表+env）同步更新。**所有数字来自 `outputs/`，可复现。**
- ⚠️ **caveat/blocker**：ERGO-II 仅 CDR3β 的 partial-feature 口径（同论文）；ERGO-II 近似去重（k>0）子集重跑未纳入（其为该 track 弱 baseline，非阻塞）；k≤3 去重后 unseen 仅 51 行方差大，不作主张。

## 2026-07-05 方法学修正：我们模型下游特征改为 post-LLaDA 口径（plan-first · Volc · 用户指示）

**背景（用户指示）**：在下游任务测试**我们自己的模型**时，**所有 feature 必须是"经过 LLaDA 之后"的表征**，而非只到 ESMC 编码器。此前 `EsmcBioSeqEmbedder`(`bioseq:`) 与 `GrammarEmbedder.embed()`(`grammar:encoder:`) 的**单序列** `embed()` 都只跑到 **ESMC encoder mean-pool 末层**（`encoder.esmc.*`），未过 LLaDA 扩散去噪 backbone → 不符合要求。（注：`GrammarEmbedder.embed_pairs(feature_source="decoder")` 已是 post-LLaDA，但仅用于成对/带 peptide 的 T1；单序列 `embed()` 一律回退到 ESMC encoder。）

**架构与"LLaDA 之后的特征"定义（已查清代码，可复现）**
- checkpoint `grammar_v2_esmc*_llada` = `BioSeqLLaDAEncoderDiffusionModel`（`dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py:1443`）：`encoder.esmc.*`=ESMC 编码器；`decoder.*`=**LLaDA 去噪 backbone**（`LLaDAModelLM`，双向、RoPE、无 timestep/RADD）；可选 `condition_norm`/`condition_proj`。
- forward（clean/无 mask）：ESMC 编码每条链 → 每残基 condition `[B,S,E]`；LLaDA `wte(input_ids)` 在残基位被 ESMC condition **替换**（grammar 特殊 token 保留 wte）→ LLaDA 双向去噪 → **final hidden state（过 `ln_f` 后）= `decoder(...,output_hidden_states=True).hidden_states[-1]`**，即与 tied 读出头对接的张量（`modeling_llada.py:1414-1417`）。
- **默认 post-LLaDA 口径（本次选定）**：把单条序列渲染成 **clean、完全可见（不加扩散噪声；LLaDA 无 timestep）** 的**单链** grammar 记录（`task_type="tcr"`、role=`tcr_beta`，**不带 peptide/MHC** → CDR3β 单独嵌入、聚类无表位泄露），跑完整 ESMC→LLaDA，取 **LLaDA final hidden（post-`ln_f`）在残基位 mean-pool**。理由：这是模型端到端"过完 LLaDA"后实际用于去噪读出的表征，与 encoder-only 严格区分。备选（留口）：可切换到某 mask 比例/noise level 输入，或换 `feature_source`；默认用 clean。
- 载入稳健性：整模型经 `load_grammar_checkpoint` **strict load（0 missing/0 unexpected，含 ESMC+LLaDA+condition 模块）**；embedder 记录 `_load_info`（取的层=`llada_decoder_final_hidden_state_post_lnf`、池化=`mean_over_residues`）。

**受影响 `Ours-BioSeq` 条目全清单（均走单序列 `embed()` → 当前为 encoder-only，须改 post-LLaDA）**
| 任务 | 脚本 | 当前口径 | ckpt 数 | 本轮 |
|------|------|----------|---------|------|
| **T2 基础A**（NAR-GAB embed-threshold, Purity/Retention/Sensitivity） | `tcr_clustering/run.py --method embed-threshold --embedder bioseq:` | ESMC-encoder | 3（esmc300m_cmp500k / esmc600m_cmp500k / esmc300m_mint） | ✅ **本轮重跑** |
| **T2 基础B**（TCREmbedding ARI/NMI/Purity, K-sweep） | `tcr_clustering/run_embed_bench.py --embedder bioseq:` | ESMC-encoder | 3 | ✅ **本轮重跑** |
| **T3 broad**（SCEPTR few-shot + probe） | `tcr_representation/run.py --method embed --embedder bioseq:` | ESMC-encoder | 3 | ✅ **本轮一并（同作业）** |
| **T1 binding**（NM2025 embed+MLP, AS/PS/HS×seen/unseen） | `tcr_binding/run_nm2025.py --method embed --embedder grammar:encoder:` | ESMC-encoder(condition) | 3 | ⏳ **本轮一并（同作业，cdr3b track）** |
| **T3 deep**（paper6 Table SI 6-pMHC, 100 seeds） | `tcr_representation/run_paper6.py --embedder bioseq:` | ESMC-encoder | 3 | ⬜ **TODO**（100 seeds 较重，先出上面四项；确认无误再补） |

**实现计划**
1. 新增 embedder `EsmcLladaBioSeqEmbedder`（spec 前缀 `bioseq-llada:`），继承 `GrammarEmbedder`（复用已冒烟的 `_final_hidden` post-LLaDA 前向），新增单序列 `embed()`：渲染单链 clean 记录→跑 ESMC→LLaDA→`hidden_states[-1]` 残基 mean-pool；接入 `build_embedder`。
2. 本地 smoke：strict load 0/0、finite、与 `bioseq:`(encoder-only) 输出确实不同、维度=decoder hidden。
3. **Volc 作业**（重活 GPU 嵌入）：一次作业跑 T2基础A+基础B+T3broad+T1(cdr3b) 的 3×ckpt post-LLaDA；参照/baseline 不动（ESM2/kmer/官方方法保持）。tag 加 `_postllada` 后缀，**与旧 ESMC-encoder 结果并列不混表**。
4. 本地：summarize + 新旧对比表 + 文档更新（`downstream.md`/`RESULTS.md`/`TCR_BENCHMARK_DESIGN.md`/本文件）。

**本地 vs 火山分工**：embedder 实现/smoke（本地轻量）；3×ckpt GPU 嵌入+聚类/few-shot/MLP（**Volc 提交**，no-proxy）；解析/summarize/新旧对比/文档（本地）。

**执行记录（滚动更新）**
- ✅ **架构查清**：`grammar_v2_esmc*_llada` = `BioSeqLLaDAEncoderDiffusionModel`；LLaDA backbone(`decoder`=`LLaDAModelLM`) 支持 `output_hidden_states=True`，`hidden_states[-1]`=过 `ln_f` 的 final hidden（`modeling_llada.py:1414-1417`）。既有 `GrammarEmbedder(feature_source="decoder")` 的单序列 `embed()`（`_embed_single_decoder`）已是 post-LLaDA 单链路径 → 新 `bioseq-llada:` 复用之，语义更清晰并强制 decoder 源。
- ✅ **实现**：`common/model_api.py` 新增 `EsmcLladaBioSeqEmbedder`（继承 `GrammarEmbedder`，强制 `feature_source="decoder"`，`_load_info` 记录取的层/池化/输入口径）；`build_embedder` 加 `bioseq-llada:` 分支（置于 `bioseq:` 前）。
- ✅ **本地 smoke**（`scripts/smoke_bioseq_llada.py`，esmc300m_cmp500k）：strict load **0 missing/0 unexpected**；embed→(40,960) finite；与 `bioseq:`(ESMC-encoder) 输出**相对 L2 差异=40.6**（确非同一张量）；dim=960（=decoder hidden，无投影）。
- ❌ **Volc 作业 `t-20260705215519-nck2s`**（`eval_jobs/eval_ours_postllada_downstream.yml`）**Failed**：T2基础A+基础B **✅ 完成**；T3broad 在首 ckpt 步 **`KeyError: 'alpha'`**（β-only 行混 paired 数据；**已在 `model_api.py:780-788` 修复**，本地 `grammar_pair_embed` smoke 通过）；T1 **未跑到**。日志：`output/downstream_generation/eval_ours_postllada_downstream_volc.log`。
- ✅ **T2 post-LLaDA 真实数字已落盘**（`outputs/tcr_clustering/*_postllada/` + `outputs/tcr_clustering_embed/*_postllada/` + `embed_threshold_summary.csv` + `_summary.csv`）：
  - **基础A**（Pur@ret≈0.25）：300m-cmp500k post-LLaDA **0.853** vs enc-only 0.912；600m **0.888** vs 0.917；mint **0.824** vs 0.874 → **post-LLaDA 在中保留区纯度略低于 encoder-only**（诚实）。
  - **基础B**（kmeans ARI mean）：300m-cmp500k post-LLaDA **0.024** vs enc 0.021；600m **0.022** vs 0.018；mint **0.005** vs 0.006 → cmp500k 略升、mint 略降，整体同量级。
- 🔄 **续跑已提交 `t-20260705224442-rn9lx`**（`eval_jobs/eval_ours_postllada_downstream_resume.yml`）：仅 T3broad + T1 × 3 ckpt（`bioseq-llada:`）；T2 不重跑。并行 yaml 亦备：`eval_tcr_binding_nm2025_postllada.yml`（T1，`grammar:decoder:`）、`eval_t3_representation_postllada.yml`（T3 broad+deep）。
- ⬜ **T3 deep post-LLaDA 重跑**：deep 已用 `grammar_tcrpair_*`（真双链 post-LLaDA），与本次单链 `bioseq-llada:` 改造**不重复**；待 broad/T1 落盘后评估是否需额外条目。

- **2026-07-05 22:44（UTC+8）** 续跑作业已 no-proxy 提交：**`t-20260705224442-rn9lx`**（`eval_ours_postllada_downstream_resume.yml`）；范围 T3broad + T1 only；`embed_pairs` fix 本地 smoke 通过（`grammar_pair_embed` shape (20,1920)）。

## 2026-07-05 evening — NbBench(A1) 执行 pivot + post-LLaDA 口径 + 实时 MD 同步

**背景（用户指示）**：NbBench 下游 benchmark 的关键决策须**实时写入 MD**，不等 GPU 作业结束；GPU 重活走 **Volc ML Platform**，本地只做代码/文档/读 Vepfs 结果。

**基准锚点**
- **主干**：NbBench（Zhang & Tsuda, *Mach. Learn.: Sci. Technol.* 6(4):040502, 2025; DOI 10.1088/2632-2153/ae20ec; arXiv:2505.02022）。
- **2026 补强**（设计/注解）：NbBayesLM（thermo 外部参照）、Aiki-GeNano（developability 6 指标）、AbBiBench（binding 方法论）、nanoBERT-infilling（生成口径参照）。

**口径纠正（CRITICAL）**
- Ours-BioSeq **探针**特征必须 **post-LLaDA**（`grammar:decoder:` / `bioseq-llada:`），不能只用 `bioseq:` → `EsmcBioSeqEmbedder`（ESMC encoder mean-pool）。
- `common/model_api.py` 已实现：`GrammarEmbedder`（含 `embed_residues`）、`EsmcLladaBioSeqEmbedder`；生成原生 `run_generative.py` 本就扩散解码，与 probing 分开。
- 早期 encoder-only Volc 跑法产出 `bioseq_esmc*_encoder` 目录，**不作 headline**；第三轮改 tag `ours_*_llada` 重投。

**执行**
- 3 份 YAML 已创建：`eval_jobs/eval_grammar_v2_{esmc600m_cmp500k,esmc300m_cmp500k,esmc300m_mint}_llada_nbbench.yml`。
- Volc task IDs（post-LLaDA 第三轮）：600m `t-20260705220330-q8qqn`、300m `t-20260705220334-64xm8`、mint `t-20260705220337-mtcz9`；smoke 全绿，全量 sweep **运行中**（进度表见 `nbbench/NBBENCH_EXECUTION_PLAN.md`）。
- `--tag` 输出目录修复已在 `run.py` / `run_residue.py` / `run_generative.py` 落地。

**文档（Live update rule）**
- 新增/更新：`nbbench/NBBENCH_EXECUTION_PLAN.md`（Volc 命令、进度表、Live update rule）、`NBBENCH_DESIGN.md` §5 + changelog、`downstream.md` A1 stub、本条目。
- 规则：任何协议变更 / task ID / blocker / 新结果 → 立即写入上述 MD + `RESULTS.md`（数字就绪时）。

**Agent 分工**：简单任务 → `composer-2.5-fast`；复杂 LLaDA/embedder → thorough agent。

- **2026-07-05 22:29（UTC+8）** NbBench post-LLaDA 轮询：Volc `t-20260705220330-q8qqn` / `t-20260705220334-64xm8` / `t-20260705220337-mtcz9` 均为 **Running**；Vepfs 新增 `hTNFa/ours_esmc600m_llada/metrics.json`（600m hTNFa → done）。仍缺：三 ckpt 的 SARS-CoV-2、Paratope、CDRInfilling(probing)；600m hIL6。

- **2026-07-05 22:45（UTC+8）** NbBench **resume 轮询**：Volc `t-20260705224330-dcd5w` / `t-20260705224329-984qk` / `t-20260705224330-zf7zf` 均为 **Running**（当前步 scalar SARS-CoV-2，head-1024 截断已生效）。Vepfs **10/12** 探针/ckpt（缺 SARS-CoV-2、Paratope、CDRInfilling probing；`gen_*` 仍 smoke n=16）。**P2 未执行**（未 Success + 未 12×3 全量）。

- **2026-07-05 22:43（UTC+8）** NbBench **blocker 已修复 + resume 已提交**：`nbbench/run.py::featurize` 对 post-LLaDA grammar embedder（`grammar:decoder:` / `bioseq-llada:`）在 binding 任务 `Ag_sequence` 列 embed 前 **head-1024 截断**（对齐 L3 长抗原截断注解；`GrammarRenderer` 单链上限 1024 aa）。冒烟 `scripts/_smoke_nbbench_antigen_truncate.py`（1273 aa 抗原）通过。Resume YAML ×3 已 no-proxy 提交：`t-20260705224330-dcd5w`（600m）、`t-20260705224329-984qk`（300m cmp500k）、`t-20260705224330-zf7zf`（300m mint）；补跑 SARS-CoV-2 + `run_residue.py --task all` + full generative。**P2 待 resume Success**。

- **2026-07-05 22:40（UTC+8）** NbBench post-LLaDA 第三轮 **Failed**：三 Volc 作业均在 `SARS-CoV-2` 步崩溃——`GrammarEmbedder` 对 >1024 aa 抗原 hard raise（SARS spike ~1273 aa），`run.py --task all` 中断，**未跑** Paratope / CDRInfilling(probing) / full generative。Partial 落盘：三 ckpt × 8/9 标量 + VRClassification + gen smoke（n=16）。Blocker + remediation（抗原 head-1024 截断后 resume 补跑）已写入 `NBBENCH_EXECUTION_PLAN.md` §Blocker。**P2 未执行**（未达 12×3 全量 metrics）。

- **2026-07-05 22:50（UTC+8）** 轮询 `t-20260705215519-nck2s` → **Failed**（ExitCode 0）：**T2A+T2B ✅**（3×ckpt post-LLaDA 曲线/summary 已落盘）；**T3broad ❌** `KeyError:'alpha'`（β-only 行，`embed_pairs` 已修 `model_api.py:780-788`）；**T1 未执行**。NbBench blocker：`nbbench/run.py` head-1024 截断已实现；各 ckpt **8/9 标量 + VR** 已有，缺 SARS-CoV-2/residue/gen。
- **2026-07-05 22:50（UTC+8）补跑提交**：
  - T1 binding post-LLaDA：`t-20260705224454-5tfcf`（`eval_tcr_binding_nm2025_postllada.yml`）
  - T3 representation post-LLaDA：`t-20260705224457-m7ndf`（`eval_t3_representation_postllada.yml`）
  - NbBench resume：600m `t-20260705224501-b7ksp` / 300m `t-20260705224504-5dbh9` / mint `t-20260705224508-v4sh2`
- **文档**：T2 post-LLaDA 数字已写入 `downstream.md` T2 段 + `RESULTS.md` (c)(d)；T1 300m post-LLaDA 已回填（unseen 0.514 vs enc 0.534），600m/mint pending。

- **2026-07-05 22:55（UTC+8）Agent 分流 + 续跑**（用户指示 composer-2.5-fast 处理简单子任务）：
  - **composer-2.5-fast**：Volc 轮询（`ml_task list -n --output json`）、产出目录盘点、T4 `_summary.json` 数字摘录、TcrDesign/TCRDiff 下载状态。
  - **主代理**：`eval_ours_postllada_t3_resume.yml` 新建、T1 yaml skip-if-done、Volc 重投。
  - **Volc**：T3 **`t-20260705224455-thzv6`**；T1 **`t-20260705224455-ndr5z`**（与 22:50 批次 `5tfcf`/`m7ndf` 并行存在，以 ndr5z/thzv6 为准）。
  - **T3 broad 口径澄清**：`bioseq_grammar_v2_*_llada` = encoder-only 误标；headline = `bioseq_llada_*`（待 thzv6）或 `grammar:decoder:` joint。

## 2026-07-05 22:50 — P0 MINT GeneralPPI 6 格收尾 + post-LLaDA 状态快照

**P0 MINT GeneralPPI（6/6 ✅，2026-07-05 15:08 收尾）**
- **Volc mint-official ✅**：`t-20260705210235-5k5zz` → HumanPPI AUROC **0.9378** / Bernett **0.7386**。
- **本地四格 ✅**：HumanPPI esm2 **0.9394** / grammar **0.9042**；Bernett esm2 **0.7858** / grammar **0.6182**（rep=3 mean）。
- 完整 2×3 表已写入 `downstream.md` P0 段 + `RESULTS.md` P0 子表 + `EXECUTION_PLAN.md` 状态表。

**post-LLaDA 方法学修正（T1–T3）**
- **代码 ✅**：`GrammarEmbedder.embed()` 默认 `feature_source=decoder`（单链 clean grammar → LLaDA `hidden_states[-1]` mean-pool）；`EsmcLladaBioSeqEmbedder`（`bioseq-llada:`）已接入；MINT PPI 的 `grammar:` 路径天然合规。
- **Volc 补跑（拆分 yaml，新 tag 与 encoder-only 并列）**：
  - T1 binding：`t-20260705215728-n4xqj`（`eval_tcr_binding_nm2025_postllada.yml`）⏳
  - T2 clustering：`t-20260705215733-f7k8q`（`eval_t2_clustering_postllada.yml`）— T2A+T2B 部分已落盘
  - T3 representation：`t-20260705220618-qdw4j`（`eval_t3_representation_postllada.yml`，`embed_pairs` fix 后重投）⏳
- **文档**：`downstream.md` P0 段 + `RESULTS.md` P0 子表 + `EXECUTION_PLAN.md` 状态表已同步（Bernett grammar pending 如实标注）。

- **2026-07-05 23:00（UTC+8）** follow-up 轮询（`volc ml_task get -o json`，no-proxy）：
  - **T1** `t-20260705224454-5tfcf` / `t-20260705224455-ndr5z` → **Running**（log：600m-cmp500k 嵌入中；300m 6/6 已有 skip）；无新 binding metrics。
  - **T3** `t-20260705224457-m7ndf` / `t-20260705224455-thzv6` / 续跑 `t-20260705224442-rn9lx` → **Running**；Vepfs **T3broad 300m post-LLaDA ✅**（`bioseq_llada_esmc300m_cmp500k_llada`：k=100 AUROC **0.657**，24-way probe **0.820**）；600m/mint broad 进行中。
  - **NbBench resume** `t-20260705224501-b7ksp` / `t-20260705224504-5dbh9` → **Running**（residue 3 tasks）；`t-20260705224508-v4sh2` → **Queue**；Vepfs **SARS-CoV-2 ×3 ckpt ✅**（600m **0.847** / 300m **0.874** / mint **0.852**，head-1024 截断）；Paratope / CDRInfilling(probing) / full gen 仍 pending。
  - 注：NbBench 完整 task id 为 `…24504-5dbh9`、`…24508-v4sh2`（非 `24501-*` 前缀）。

## 2026-07-05 Volc T1/T3 续跑 Success · 文档同步

- **T1 binding post-LLaDA** Volc `t-20260705224455-ndr5z` → **Success**（3/3 ckpt）：600m seen/unseen AUPRC **0.538/0.515**；mint **0.494/0.517**（`grammar:decoder:` joint，源 `outputs/tcr_binding_nm2025/ours_postllada_*`）。
- **T3 broad post-LLaDA** Volc `t-20260705224455-thzv6` → **Success**（3/3 ckpt）：600m k=20/k=100 **0.629/0.643** probe **0.811**；mint **0.606/0.613** probe **0.816**（`bioseq-llada:`，源 `outputs/tcr_representation/bioseq_llada_*`）。
- **文档**：`downstream.md` T1 对照表 + 进展段、`RESULTS.md` T1/T3-broad pending 行已同步。

- **2026-07-05 23:01（UTC+8）** follow-up 轮询（`VOLC_DEFAULT_OUTPUT=json`，no-proxy）：
  - **T1** `t-20260705224455-ndr5z` → **Success**（Elapsed ~447s，Exit 0；Vepfs `ours_postllada_*` 三 ckpt 均 **6/6** splits metrics.json）。
  - **T3 broad** `t-20260705224455-thzv6` → **Success**（Elapsed ~362s；Vepfs `bioseq_llada_*` 三 ckpt `fewshot.json`+`metrics.json` 已齐，与 RESULTS 表一致）。
  - **续跑** `t-20260705224442-rn9lx` → **Running**（Volc 未 End；Vepfs `eval_ours_postllada_downstream_resume_volc.log` 已印 `RESUME done 2026-07-05T15:00:35Z`，与 ndr5z/thzv6 专用 yaml 重叠，待 Volc 终态）。
  - **文档**：`downstream.md` / `RESULTS.md` / PROGRESS「Success · 文档同步」段已含上述数字，本轮**未改**结果表，仅追加本快照。

- **2026-07-05 23:01（UTC+8）** NbBench resume 轮询：`t-20260705224330-dcd5w`/`t-20260705224329-984qk`/`t-20260705224330-zf7zf` → **Success**；metrics **12×3 + gen 2847**；`summarize_nbbench.py` + A1 回填。

## 2026-07-05 整段 feature 口径（global whole-feature）实现 + Volc 重跑

- **方法学决策**：Ours-BioSeq 下游 headline **禁止** segmented/piecewise 特征（per-chain segment mean-pool 再 concat、`sep_chains` 分链拼接等）。统一为：完整 grammar record → post-LLaDA clean forward → **全局 mean-pool** → 单向量 `[H]`。
- **代码**：
  - `model_api.py::GrammarEmbedder`：`pool_mode="global"`（默认）；legacy `segment_concat`；`_record` 现含 peptide→`tcr_peptide`；spec `grammar:decoder:global:` / `:wholefeat:`。
  - `mint_tasks/embedders.py::GrammarEmbedder`：默认 `sep_chains=False`（MINT official 仍 `True`）；缓存名 `*_globalfeat_*`。
  - `tcr_binding/run_nm2025.py`、`tcr_clustering/run.py`（`--use-alpha`+decoder→`embed_pairs`）、`tcr_representation/run.py`（probe→`grammar_pair_embed`）、`fewshot.py` 同步。
- **冒烟**（`scripts/smoke_globalfeat_pool.py`，ckpt 300m-cmp500k）：T1 bind global `(2,960)` vs segment `(2,1920)`；T3 global ≠ 各 segment pool（max|Δ|≈0.48）。
- **Volc 提交**（新 tag，不覆盖 `ours_postllada_*` / `*_sep_*`）：
  - T1 `eval_tcr_binding_nm2025_globalfeat.yml` → `t-20260705230306-mk5jr`
  - T2 `eval_t2_clustering_globalfeat.yml` → `t-20260705230310-p97r7`
  - T3 `eval_t3_representation_globalfeat.yml` → `t-20260705230313-s7xsw`
  - PPI `eval_mint_ppi_globalfeat.yml` → `t-20260705230316-wwphr`
- **文档**：`downstream.md` 关键进展、`mint_tasks/EXECUTION_PLAN.md`（PPI headline 待 globalfeat 落盘）、本段。

## 2026-07-05 AB 抗体下游重跑（CDR + light pairing · resume 后 best.pt）

**动机**：7-04 `output/downstream_generation/*_downstream_summary.txt` 与 `*_light_pairing_*_metrics.json` 基于 resume 前旧 `best.pt`（300M/600M 均在 7-5 被覆盖）。在 current `best.pt` 上重跑 SabDab CDR 10-fold + OAS holdout500 light pairing（post-LLaDA 扩散解码，非 encoder-only）。

**Volc 提交**（`eval_jobs/eval_grammar_v2_esmc{300,600}m_cmp500k_llada_downstream.yml`，no-proxy）：
- 300M → `t-20260705220152-768fw` → **Failed** @ 2026-07-05T14:58:48Z（CDR ✅；light pairing 生成 125/125 ✅；ImmunoMatch 崩溃：`transformers` + Keras 3，需 `pip install tf-keras`）
- 600M → `t-20260705220153-zwgdk` → **Success** @ 2026-07-05T15:33:44Z（全流程 ✅）

**CDR Average AAR all folds（新 vs 7-04 旧 ckpt）**

| ckpt | H1 | H2 | H3 | Δ H1/H2/H3 (pp) |
|------|:--:|:--:|:--:|:----------------|
| 300m | **68.76** | **63.64** | **44.68** | +4.0 / +5.7 / +4.1（旧 64.72/57.93/40.54） |
| 600m | **71.88** | **67.27** | **45.25** | +2.9 / +4.9 / +3.5（旧 68.95/62.34/41.80） |

**Light pairing（300M + 600M 新 metrics）**

| ckpt | gen ImmunoMatch | V gene match | gen>ref | 备注 |
|------|:---------------:|:------------:|:-------:|------|
| 300m | **0.633**（旧 0.430） | **0.835**（旧 0.577） | **0.393**（旧 0.263） | metrics-only `t-20260705233630-mrmzv` Success @ 16:07:55Z |
| 600m | **0.607**（旧 0.455） | **0.835**（旧 0.577） | **0.368**（旧 0.263） | 主作业 `zwgdk` Success @ 15:33:44Z |

**文档回填**：`downstream.md` AB 段、`RESULTS.md` A1 下 AB 子表、本条目、`PROJECT_PROCESS.md` 完成记录。

- **2026-07-06 00:08（UTC+8）** 300M pairing metrics-only **`t-20260705233630-mrmzv` → Success** @ 2026-07-05T16:07:55Z；gen ImmunoMatch **0.633** / V gene **0.835** / gen>ref **0.393** / chain **1.000** / diversity **0.163**。AB 下游 eval **全部落盘**。

## 2026-07-06 follow-up（MINT globalfeat / TcrDesign C / TCR-epiDiff）

- **MINT PPI globalfeat ✅**：Volc **`t-20260706004005-tqw46`** Success → 6× `*_globalfeat_t3000/*_metrics.json`；headline Ours = **esmc600m_cmp500k**（HumanPPI AUROC **0.8300** / Bernett **0.6089**）。文档：`downstream.md` P0、`mint_tasks/EXECUTION_PLAN.md` §3.1。
- **TcrDesign Setting C 全量 🔄 running**：`run_tcrdesign_baseline.sh full all 100 tcrdesign_full`，pid **1399291** (`run_tcrdesign_baseline.sh`; nohup shell 1399290)，log `/tmp/tcrdesign_full_clean.log`（冒烟 metrics 已存在于 `outputs/tcr_generation_bench/setting_C/tcrdesign_full/metrics.json`，全量完成前勿覆盖 headline）。
- **TCR-epiDiff**：**BLOCKED** — Zenodo 15094766 返回 **410 Gone**，官方 `Best_model.pth` 不可得；不伪造权重、不接 wrapper 直至有合法权重源。

## 2026-07-07 T1 binding headline 收尾：globalfeat 汇总入主表（文档层，零重算）

**背景（用户指示）**：T1 的 headline 口径按 PROJ_GUIDE.md「Downstream Feature Extraction (post-LLaDA, mandatory)」+ 2026-07-05 整段 global whole-feature 决策应为 **post-LLaDA globalfeat**（`grammar:decoder:global:`，完整 record 全局 mean-pool）。此前 3×ckpt 六套 test 的 globalfeat 结果已由 Volc `t-20260705230306-mk5jr`（`eval_tcr_binding_nm2025_globalfeat.yml`）跑通并落盘 `outputs/tcr_binding_nm2025/ours_globalfeat_*`，但 **`summary_main.csv` 与文档主表仍停留在 encoder-only 消融口径**，未收尾。

**改动（纯文档 + 汇总脚本，`outputs/` 各 metrics.json 数字不动、不重算）**
- `scripts/summarize_nm2025.py`：`METHODS` 重排——**headline = `ours_globalfeat_*`**（3 ckpt），`ours_postllada_*` 标 **control-A（post-LLaDA segment-concat）**、`ours_biseq_mlp*` 标 **control-B（encoder-only）**。重跑生成 `summary_main.csv` / `summary_neg_source.csv`（全部读自 metrics.json）。
- **headline 数字（cdr3b · AS · overall AUPRC，seen/unseen）**：esmc300m-cmp500k **0.583/0.507**、esmc600m-cmp500k **0.576/0.496**、esmc300m-mint **0.596/0.501**。
- `downstream.md` T1：主表 Ours 行换为 globalfeat headline；新增「口径对照（headline globalfeat vs 两组 control · 3×ckpt）」表；L2 表 Ours 行换 globalfeat；顶部「关键进展」T1 行 + 最近更新日期同步；复现命令改 `grammar:decoder:global:`。
- `RESULTS.md` T1：主表 + 消融对照表 + L2 表同步为 globalfeat headline；L1 近似去重行标注为 **enc-only control**（near-dedup 仅在 enc-only 上算过，未对 globalfeat 重算）。

**诚实结论（写入两文档）**：globalfeat（guideline 强制 headline）unseen AUPRC 0.496–0.507，实际**略低于** encoder-only control（0.511–0.534）与 segment control（0.514–0.517）——三口径 unseen 全落近随机带（0.49–0.53），**post-LLaDA 相比 encoder-only 在 T1 unseen 未见提升**，如实标注不夸大；seen 上 globalfeat（0.58–0.60）与其他口径相当，仍低于专用 binding 模型（ATM-TCR/TEIM 0.68–0.70）。

**遗留（非阻塞）**：L1 近似去重未对 globalfeat 重算（enc-only 已足以证明"kNN 表观优势 k≤1 坍缩"论点，globalfeat 变化预计温和）；如需 globalfeat 的 L1 曲线，跑 `scripts/nm2025_near_dedup.py` 时把 tag 指到 `ours_globalfeat_*` 即可。

## 2026-07-07 T1 官方 baseline 扩充（plan-first · 派 composer-2.5-fast · 主代理监督）

**背景（用户指示）**：NM2025 官方仓库 `Original_model/` 的权重我们**已全部下载**，但只复现了 6 个官方 baseline（ATM-TCR/NetTCR/TEIM/ERGO-II/epiTCR/PanPep）；要求把参考文献里其余**已下载权重 + 有官方代码 + 有 env yml** 的模型也构建 + 重跑复现，进 T1 主表。先出 plan → 调 composer-2.5-fast 执行 → 主代理监督验收。

**Plan**：`benchmark/T1_BASELINE_EXPANSION_PLAN.md`（分 Tier-1/2/3 + 每模型 6 步执行协议 + 收尾 + 硬性原则：只用官方源码/权重、禁编造、blocker 如实标）。
- **Tier-1（现代 torch2.0.x，cdr3b 口径）**：TEINet(large/small) / ERGO(AE·lstm × vdj·mc, 4 变体) / TPBTE(mc/vdj) / SETE。
- **Tier-2**：TITAN / VitTCR / TEPCAM / AttnTAP / PISTE / TCRconv / TCR-H。
- **Tier-3（keras/TF 旧栈）**：iTCep / MCMC / TCRGP / ImRex / NetTCR-AB / DeepAIR(TF1.2 最难)。
- **跳过**（README 声明不可接入）：DLpTCR / PiTE / vibtcr / pMTnet(_omni) / MixTCRpred / TCRfinder。

**执行记录（滚动）**
- 2026-07-07 立计划 + 派 **composer-2.5-fast**（background）跑 **Tier-1**（TEINet/ERGO/TPBTE/SETE）。主代理监督：完成后验 `outputs/tcr_binding_nm2025/<tag>/cdr3b/{seen,unseen}/metrics.json` 存在、AUPRC 合理、n_scored==n_input，再重跑 `summarize_nm2025.py` + 回填文档。

## 2026-07-07 MINT PPI 覆盖扩展：GeneralPPI 新任务 ×3 + baseline PLM ×4（两 composer-2.5 子代理并行，均提交成功 Running）

**背景（用户指示）**：只聚焦 MINT 的 PPI 任务（casestudy 不管），对齐论文补齐「一：GeneralPPI 更多任务」+「二：更多 baseline PLM」，用 composer-2.5 子代理执行。派两路并行子代理（新任务数据/backend 一路、baseline PLM 一路），避免文档写冲突。

**路 A — GeneralPPI 新任务扩展**（`prepare_data.py` 新增 `prepare_yeastppi/mutationalppi/skempi`；本地 grammar esmc300m_cmp500k `--test_run` 冒烟 extract+finetune 全过）
- **YeastPPI** ✅ `data/downstream/mint/yeast-ppi/processed_data_{train,validation,test}.csv`（4945/95/394；复用 `_read_humanppi_lmdb` 读 `peer_yeastppi/*.lmdb`，与 MINT/PEER 官方同源）。
- **MutationalPPI** ✅ `data/downstream/mint/mutational-ppi/processed_data.csv`（12612 行，pos/neg 1473/11139，10-fold CV）。**口径差异脚注**：本地 SWING CSV 的 Y2H 已是 {0,1} 二值（非 MINT Methods 0–4 + cutoff=2 原始刻度），序列取自内嵌列而非 UniProt lookup。
- **SKEMPI** ✅ `data/downstream/mint/SKEMPI_v2/processed_data.csv`（6706 行，ΔΔG=(8.314/4184)·298.15·ln(Aff_mut/Aff_wt)；列 `seq1,seq2,seq1_mut,seq2_mut,target,split_0/1/2`；3-fold complex-held-out random_state=2023，blocklist 1KBH；逻辑复刻官方 notebook，未与 published csv 逐行核对故不编造对比数字）。
- **PDB-Bind** ⛔ **blocker**：本地仅 `data/ppi_task_raw/raw/pdbbind/site_assets`，全 vepfs 搜索无真实序列/标签 → 需 pdbbind-plus.org.cn 注册下载，**未纳入网格**。
- Volc **`t-20260707165937-zz8gr`**（`eval_mint_ppi_newtasks.yml`）：3 任务 × 3 backend（MINT sep / ESM2-650M sep / grammar×3 globalfeat）= **9 格**，**Running**；日志 `output/downstream_generation/eval_mint_ppi_newtasks_volc.log`。

**路 B — baseline PLM 补齐**（`baselines.py` 集成 ESM-1b / ESM2-3B / ProtT5-XL / ProGen2）
- ESM2-3B 本地 smoke ✅（`/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`）；ESM-1b/ProtT5/ProGen2 经 HF 镜像下载（Volc 内）。
- Volc **`t-20260707165326-5qg4d`**（`eval_mint_ppi_baselines.yml`）：4 baseline × (HumanPPI+Bernett) = **8 格**，**Running**。
- 状态表：`mint_tasks/EXECUTION_PLAN.md` 新增 §7（baseline）。

**校验（本轮）**：两 yaml 存在；数据 csv 行数与 SKEMPI 列头 (`seq1,seq2,seq1_mut,seq2_mut,target,split_0/1/2`) 核对通过；`volc ml_task get -o json` 确认两作业 **Status=Running**（newtasks Elapsed≈128s / baselines≈499s）。

**待办（数字纪律）**：两作业 Success 后回填 `RESULTS.md`（新任务表 + baseline 表，每格必须溯源 `outputs/mint/*/*_metrics.json`）+ `mint_tasks/EXECUTION_PLAN.md` §6/§7 状态；MutationalPPI 对比行加二值化口径脚注；不编造未落盘数字。

### 2026-07-07 收尾修复：newtasks 作业 esm2 缓存路径 bug → 重提

- **作业 A `t-20260707165937-zz8gr` Failed @9.5min**（作业 B baselines 无此问题、正常 Running）。根因：`eval_mint_ppi_newtasks.yml::esm2_cell` 的 **finetune 步误把完整绝对路径 `${ESM2}` 传给 `--model`**。`extract_embeddings_baselines.py` 抽取时做 `model_name.split("/")[-1]` 取 basename 写缓存（`YeastPPI/esm2_t33_650M_UR50D_sep_t3000/` 已正确落盘 train/val/test.pt），但 `finetune_general.py` 经 `model_cache_name()` 对非 grammar/mint spec 直接 `base=spec`，绝对路径参与 `Path(RESULTS_ROOT)/task/<abs>` 时被绝对分量重置 → `emb_dir` 指向 `model_weights/esm2/..._sep_t3000`（无嵌入）→ `FileNotFoundError` → `SystemExit(1)` + `set -e` 整作业挂。对照 `eval_mint_ppi_baselines.yml` finetune 传干净 stem（`--model "${stem}"`），故 B 不受影响。
- **修复（两处）**：① `extract_embeddings.py::model_cache_name` else 分支 `base = spec.split("/")[-1]`（路径型 spec 一律取 basename，extract 已 split 故幂等、baselines yaml 传 stem 无副作用）；② `eval_mint_ppi_newtasks.yml::esm2_cell` 新增 `ESM2_STEM=esm2_t33_650M_UR50D`，finetune 改传 `--model "${ESM2_STEM}"`。
- **本地验证**：`model_cache_name(<abs esm2 path>|<stem>, sep=True, 3000)` 均解析为 `esm2_t33_650M_UR50D_sep_t3000` 且 `train.pt` 命中。清理泄漏产生的垃圾目录 `model_weights/esm2/esm2_t33_650M_UR50D_sep_t3000`（4.5K，非真实权重）。
- **重提**：Volc **`t-20260707172352-znrkl`**（`eval_mint_ppi_newtasks.yml`，9 格）**Running**；已抽好的 YeastPPI mint 指标 + esm2 嵌入会被复用（extract isfile 跳过）。旧 `t-20260707165937-zz8gr` 作废不回填。

### 2026-07-07 两 MINT PPI 作业 ✅ Success，全部数字已回填（RESULTS.md P0 + EXECUTION_PLAN §6.4.1/§7 + downstream.md 关键进展）

- **作业 A 新任务** `t-20260707172352-znrkl` **Success @65min**，9/9 格全落盘；**作业 B baseline** `t-20260707165326-5qg4d` **Success @89min**，8/8 格全落盘。接线无 blocker（esm1b AutoModel 路由、ProGen2 trust_remote_code、ProtT5/ESM2-3B hf-mirror 均正常）。
- **headline 结论（数字溯源 `output/downstream_generation/mint_tasks/<task>/<cache>/*_metrics.json`）**：
  - **【已撤回；由 2026-07-14 baseline audit 取代】SKEMPI 历史本地结果**：曾记录 Ours post-LLaDA globalfeat 三 ckpt Spearman 0.34–0.43 / RMSE 1.32–1.73，以及 MINT sep（0.21 / 5.92）和 ESM2-650M（0.25 / 3.69）。后续审计确认 Pearson/RMSE 位于 fold-specific PowerTransformer 空间，不能解释成论文原始 ΔΔG 单位，也不能据此作“领先”结论；论文值改由 `[P]` 表单独引用，raw-unit 本地重跑 pending。
  - **YeastPPI（binding 分类）**：MINT sep 最强（AUROC 0.678），ESM2-650M 0.648，Ours globalfeat 0.57–0.60（落后）。
  - **MutationalPPI（10-fold CV，极不平衡）**：看 AUPRC，ESM2-650M 0.250 > MINT 0.186 > Ours 0.177–0.183；F1≈0（少数类几乎全预测负）。⚠️ 本地 Y2H 预二值化，与 MINT Methods（0–4+cutoff2 + UniProt lookup）口径不完全同构，已在表中脚注。
  - **baseline PLM 补齐（HumanPPI / Bernett）**：ProtT5-XL 两任务最强（0.939 / 0.799），ProGen2 最弱（0.622 Bernett），ESM-1b ≈ ESM2-3B ≈ 0.93 / 0.71–0.73；符合 MINT Fig.2 趋势。
- **文档**：`RESULTS.md` P0 新增「baseline PLM 补齐」表 + 「GeneralPPI 新任务」三子表；`mint_tasks/EXECUTION_PLAN.md` §6.4.1（新任务结果）+ §7（baseline 结果 + 接线成功）；`downstream.md` 关键进展 MINT 条目改为 Success + headline 结论。**PDB-Bind 仍为唯一 blocker**（本地无数据）。数字纪律：全部可溯源，未臆造。

### 2026-07-07 收尾：T2 基础B 补 ProtBERT/TCR-BERT 参照 + P0 表补齐 Ours-globalfeat AUPRC/Acc/F1

- **T2 基础B 参照嵌入补齐（本地跑，无 Volc）**：
  - `run_embed_bench.py --embedder protbert --tag protbert` → `outputs/tcr_clustering_embed/protbert/{curve.csv,metrics.json}`。K-means mean/best：**ARI 0.006/0.008 · NMI 0.061/0.087 · Purity 0.248/0.269**（embed 285.3s）。
  - `run_embed_bench.py --embedder tcrbert --tag tcrbert` → `outputs/tcr_clustering_embed/tcrbert/{curve.csv,metrics.json}`。K-means mean/best：**ARI 0.016/0.021 · NMI 0.095/0.120 · Purity 0.287/0.304**（embed 133.1s；首跑 hf-mirror ReadTimeout 后重试成功；pooler 权重未初始化警告，与 T3 口径一致）。
  - `summarize_tcrembedding_clustering.py` 重建 `_summary.csv`（11 行，含 protbert/tcrbert）。
  - 文档：`RESULTS.md` T2 (d) 表 + `downstream.md` 关键结果 3 表各增 ProtBERT/TCR-BERT 两行。
- **P0 MINT GeneralPPI Ours-globalfeat 指标补齐**（仅 HumanPPI + Bernett，headline = esmc600m_cmp500k globalfeat，3-rep mean）：
  - HumanPPI：**AUROC 0.8300 / AUPRC 0.8109 / Acc 0.7271 / F1 0.6747** ← `.../HumanPPI_grammar_grammar_v2_esmc600m_cmp500k_llada_globalfeat_t3000_metrics.json`
  - Bernett：**AUROC 0.6089 / AUPRC 0.6160 / Acc 0.5314 / F1 0.1645** ← `.../Bernett_grammar_grammar_v2_esmc600m_cmp500k_llada_globalfeat_t3000_metrics.json`
  - 文档：`downstream.md` P0 2×3 网格 + `RESULTS.md` P0 主/辅表 Ours 列由 legacy sep 值更正为 globalfeat headline；legacy sep 移脚注。

## 2026-07-07 TCR-epiDiff（2025 DDPM）Setting B 接入 → 端到端审计后**降级为仅引用（cite-only）**

**背景（用户指示）**：把 TCR-epiDiff（Bioinformatics 2025，DDPM 扩散）接入 Setting B（epitope→CDR3β），用官方代码 + 本地已有权重 `Model/TCR-epiDiff_Best_model.pth`（12MB）生成 benchmark14 k=100、评分、回填。中途用户下 **STOP + 改方针**指令：先只判定官方生成代码是否 turnkey-reproducible，若否则不 reimplement、降级为 cite-only（同 LSMTCR/TCRGen），并回填文档。

**turnkey 判定：NO（官方生成代码非 turnkey-reproducible）**。端到端核对 `Tutorial_Code/TCR-epiDiff/{TCR-epiDiff_main.py,TCR-epiDiff_model.py}`、`Tutorial_Code/Data_Processing.py`、`diff_neo_valid_practice.ipynb`（无 requirements.txt，官方 GitHub 也无）后确认四处硬伤：
1. `TCR-epiDiff_main.py:24` `from TCR-epiDiff_model import *` —— 模块名带连字符，**非法 import，脚本无法直接运行**。
2. 硬编码**未随仓库发布**的绝对路径 pkl 输入 `/tcr_epitope_peptide.pkl`（L27/L63）、`/CDR3_epi_onehot_emb_covid.pkl`（L48，作者自注"数据太大需自建"）；`Data_Processing.py` 又依赖未发布的 `/data.csv`（L94）+ ProtT5-XL。
3. 采样 `denoise_with_trained_model`（main L100-112）是 **denoise-from-a-real-TCR**：`TCR=original_TCR[0]` → `add_noise` → 去噪，需真实 TCR one-hot 种子 batch，**非 clean de-novo 表位→CDR3β 生成**。
4. `UNet1D(in_channels=4)`（main L60）用**非显然的 4 通道核苷酸/密码子编码**；`Data_Processing.py:35` 的 one-hot 构造引用**从未定义**的 `nucleotide_to_index`（全仓库无赋值，另一处 broken reference）。
跑通它需重写模型 I/O + 预处理管线，违反"官方代码 only，不 reimplement 模型/预处理"铁律。

**动作**：
- **不接入生成路径、不编造数字**。`wrappers/tcr_epidiff_generate.py` 恢复为 cite-only 骨架：`CITE_ONLY=True`，`check_ready()`/`--check-only` 返回 `BLOCKED: cite-only ...`（已验证 exit 3），`main()` 生成路径故意不实现。
- **文档降级**：`TCR_GENERATION_BENCHMARK.md`（§0 结论、§0.1 文献表、§0.3 论证、§1 Setting B baseline 列、§7 baseline 表、§8 环境）+ `RESULTS.md`（T4 Setting B 扩散行）全部把 TCR-epiDiff 从 ⏳/✅ 改为 ⛔ **仅引用**，并写明四点具体 blocker；TCRDiff 保持为已复现的 2026 扩散 Setting B baseline。
- **清理**：本轮改方针**前**（STOP 指令到达前）已创建过 `tcr_epidiff` conda env 并下载 ProtT5-XL（22GB 到 `.cache/huggingface/hub/models--Rostlab--prot_t5_xl_uniref50`）；判定 NO 后**均已删除**（`conda env remove -n tcr_epidiff` + `rm -rf` ProtT5 缓存），删除临时生成的 `outputs/.../setting_B/tcr_epidiff/`（smoke + 部分 designs）与临时 `requirements.txt`。本地官方权重 `Model/TCR-epiDiff_Best_model.pth` 仅作 provenance 保留。

## 2026-07-07 PDB-Bind 多链任务：数据解除 blocker + grammar/mint 适配 + Volc 重提

**背景（用户确认「需要」）**：MINT PDB-Bind 为蛋白-蛋白复合物结合亲和力回归（2–6 链）；我们 `GrammarEmbedder` 原先仅支持 2 链 `ppi_conditional`，需扩展 N 链并补齐数据与评测网格。

**数据（解除 blocker）**
- 官方 `INDEX_general_PP.2020` 需注册；改从 HF **`proteinea/ppb_affinity`**（CC-BY-4.0）拉 PDBbind v2020 PP 子集。
- `prepare_data.py::prepare_pdbbind` → `data/downstream/mint/pdb-bind/processed_data.csv`：**3267** 行（原始 3593 → 去 >6 链 24 + seq 去重）；链分布 2:1857 / 3:915 / 4:372 / 5:85 / 6:38；`target=-log10(KD)` ∈ [1.60, 15.70]。
- 截断对齐 MINT `PDBBindCollateFn(max_length=1024)`（`per_chain = 1024 // n_chains - 2`）。⚠️ 与 MINT 论文数字不逐字节可比（链提取来源差异），三 backend 同 CSV 公平对比。

**代码适配**
- `GrammarEmbedder`：`complex_mode=True`（Pdb-bind 自动）→ `single_entity` 多链 grammar record + post-LLaDA 全局 mean-pool；`complex_total_budget=1024`。
- `MintEsm2Embedder`：`use_pdbbind_layout=True`（Pdb-bind 自动）→ 固定槽位 1024-token 布局（复刻 `PDBBindCollateFn`）。
- `extract_embeddings.py`：Pdb-bind 强制 bs=1、每 32 步 `cuda.empty_cache()`、CUDA 错误单次重试。

**本地冒烟**：grammar esmc300m complex_mode extract+finetune（`--test_run` 20 行）✅；`train.pt` shape `(20, 960)` finite；10-fold Ridge CV 可跑通。

**Volc**
- YAML：`eval_jobs/eval_mint_ppi_pdbbind.yml`；5 backend（mint global / ESM2-650M global / grammar×3 globalfeat）× 10-fold Ridge CV。
- 首轮 **`t-20260707201254-26rk7` Failed** @ mint extract ~127/3267（`CUDA error: unspecified launch failure`；样本编码长≤1020，判为长程 GPU 状态问题 + 非 pdbbind 布局）。
- 修复后重提 **`t-20260707203251-thcvf` Running**；日志 `output/downstream_generation/eval_mint_ppi_pdbbind_volc.log`。

**待办**：作业 Success 后从 `output/downstream_generation/mint_tasks/Pdb-bind/<cache>/*_metrics.json` 回填 `RESULTS.md` / `downstream.md` / `EXECUTION_PLAN.md` §6.3.1 结果表（Spearman/Pearson/RMSE）。

## 2026-07-07 三处已完成结果的文档回填（纯文档 + 汇总脚本，零 GPU、零重算）

**背景（用户指示）**：盘上已有三处跑完但文档未回填的下游结果，本轮按 plan 补齐；数字全部溯源已落盘 `metrics.json`，不重算/不臆造。GPU 重活维持 Volc（本轮无新作业）；机械性文档回填派 composer-2.5-fast 子代理（按文件切分避免写冲突），脚本/串联/收尾主代理自做。

- **【已撤回；由 2026-07-14 baseline audit 取代】PDB-Bind 历史本地结果**（Volc `t-20260707203251-thcvf`）：曾记录 Ours Spearman 0.660–0.676、所谓 “MINT-official” 0.615、ESM2-650M 0.570，以及 transformed-space RMSE。审计确认本地 HF 子集、特征/训练协议和论文 PDB-Bind 构建不一致，且旧 Pearson/RMSE 未 inverse-transform；这些值仅保留为历史 artifact，不再称为 MINT official 复现或“领先”。论文 `[P]` 结果与本地 `[L]` diagnostic 已拆表。
- **T1 官方 baseline 扩充 Tier-1（历史状态，已由 2026-07-21 original-only 记录覆盖）**：`summarize_nm2025.py::METHODS` 当时新增 teinet_large/small、ergo_{ae,lstm}_{vdj,mc}、tpbte_{mc,vdj}。当时 SETE 尚无输出；当前精确失败状态与论文 fallback 已写入 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/sete/cdr3b/original_run_status.json`，canonical 状态见本文件后续 2026-07-21 记录。
- **TcrDesign Setting C 全长（✅ 已评分回填）**：34 pMHC k=100 → overall F1 **0.061** / seq-rec **0.772** / d_edit **2.58** / Char-BLEU 0.84（`setting_C/tcrdesign_full/metrics.json`）。回填 `downstream.md` T4 + `RESULTS.md` T4 + `TCR_GENERATION_BENCHMARK.md`（⏳→✅）。
- **数字纪律**：三处均只引用已落盘 metrics.json；无编造。本轮不在范围：T1 Tier-2/3 baseline、SETE 修复、TCRDiff held20、mint ckpt 训练完成后 T3 重刷、模型能力短板（T1 unseen 近随机 / T4 Setting B F1=0，属诚实结论）。

## 2026-07-08 T2 clustering 方法调研 + 基础B 补 SCEPTR 参照（零成本，本地）

**背景（用户指示）**：调研 TCR clustering 是否还有其他可纳入方法。web 复核确认基础A（NAR-GAB 官方 9 方法）已收全该领域经典法，增量都在 embedding→clustering（基础B 范畴）。查到候选：**SCEPTR / TouCAN / Metaclonotypist / TCRemP / G2VTCR**。评估：
- **SCEPTR**（Cell Systems 2024，`yutanagano/sceptr` 已装 v1.2.0）：零成本、最该补——基础B 此前缺 TCR 领域最重要的对比学习 SOTA 的聚类对照。✅ **本轮已接入**。
- **TouCAN**（Brief Bioinf 2024，`LSSI-ETH/TouCAN`）：最对口（pLM+对比学习+聚类），但 TF1.x/keras 老栈 + 无发布权重需训练 + paired αβ V-domain 口径。中优先级，待接。
- **Metaclonotypist**（bioRxiv 2025，pip/MIT）：Leiden 图聚类 + TCRdist/SCEPTR 后端，但输出是 metaclone-HLA 关联，需适配。可选。
- **TCRemP**（antigenomics，prototype+DBSCAN）：可选待核。
- **G2VTCR**（`princello/G2VTCR`，1 star）：仓库已重构为 binding 分类器，clustering 仅 legacy v1，训练数据不含 → 可复现性差，**观望不接**。

**动作（SCEPTR 接入基础B，纯本地零 GPU 作业）**
- `tcr_clustering/run_embed_bench.py`：新增 `sceptr` 分支——SCEPTR 是 native TCR 模型（非 plain SequenceEmbedder），走 `build_distance_source("sceptr").embed(df)`，喂 CDR3β+TRBV（**β-only partial 输入**，α 留空，SCEPTR 官方支持 partial；基础B 本就单链 CDR3β + 有 vgene 列）。冒烟：50 行 → (50,64) finite ✅。
- 全量跑通（本地 14.7s 嵌入 9033 序列 + K∈{10..100} kmeans+hierarchical 双算法）→ `outputs/tcr_clustering_embed/sceptr/{curve.csv,metrics.json}`。**K-means mean/best over K：ARI 0.033/0.063 · NMI 0.159/0.186 · Purity 0.339/0.370**。
- `summarize_tcrembedding_clustering.py` 重建 `_summary.csv`（12 行含 sceptr）。

**诚实结论（回填 RESULTS.md / downstream.md / TCR_BENCHMARK_DESIGN.md，遵守 PROJGUIDE §2.4 Ours 置末规约）**：加入 SCEPTR 后基础B 次序 = **SCEPTR（ARI 0.033，最强）> Ours post-LLaDA cmp500k（0.024）> TCR-BERT 0.016 > k-mer 0.010 > ESM2 0.007 > ProtBERT 0.006**。即我们的模型**领先所有通用/自监督 PLM，但低于专用对比学习 SCEPTR**——与 T3（SCEPTR 反超 PLM）及 SCEPTR 论文核心论点一致、互相印证。此前文档"基础B 全面领先"的表述已更正（SCEPTR 未纳入时 Ours 确为最高，纳入后退居第二）。**口径注**：SCEPTR 用 CDR3β+TRBV（原生 partial，比纯 CDR3β 多 V 基因信息），其余参照仅 CDR3β，已标注。canvas `irbench-results` T2 表/结论同步。
- **数字纪律**：SCEPTR 数字来自本地 `outputs/tcr_clustering_embed/sceptr/metrics.json`，可复现（`run_embed_bench.py --embedder sceptr`）；无编造。TouCAN/Metaclonotypist/TCRemP 待用户确认是否接（有工程成本）。

## 2026-07-08 AB 任务 baseline 补全（Ophiuchus-Ab 口径对齐）

**背景（用户指示）**：`downstream/` 除 `benchmark/` 外均为 AB 抗体任务；参考 Ophiuchus-Ab 论文补 baseline。锚点模型 = **grammar_v2 post-LLaDA**（与全 benchmark 一致）。

**已完成**
- **FLAb 数据修复**：4 CSV 自 `Graylab/FLAb` GitHub 重新下载至 `data/downstream/flab/flab_raw/`（原 dead symlink 指向已删 AirGen-Dev 路径）。
- **统一 runner**：`downstream/flab/run_flab_baselines.py`（paired [H|L] concat + PowerTransform + Ridge 5-fold CV；复用 `build_embedder`）。
- **FLAb sweep**：`downstream/flab/run_flab_all.sh` → **32/32 json ✅**（4 数据集 × 8 embedder，含 g6_er + Ophiuchus-Ab；g6_er × onehot ~21 min 完成 Spearman 0.602）。
- **CDR infilling 历史本地 diagnostic `[L]`**：`downstream/infill/run_cdr_baselines.py` + `run_cdr_all.sh` 得 AntiBERTy H1 **76.20** / H2 **71.61** / H3 **41.23**，同一本地设置下 grammar_v2 H3 **45.25**。该数据路径/切分未证明与 Ophiuchus-Ab Table 2 完全一致，故撤回 paper-level “领先”表述；主 baseline 现引用论文 Table 2 `[P]`，本地数值只另表展示。
- **文档回填**：`downstream.md` AB 段 + 新 AB-FLAB 段；`RESULTS.md` CDR baseline + FLAb 子表。

**Blocker（如实标注，未编造）**
| 任务 | Blocker |
|------|---------|
| in_silico m396（论文 binding 口径） | `Desautels_insilico_data.csv` dead symlink，定向搜索无副本 |
| dev GDPa1（论文 Table 4 developability） | 无 GDPa1 数据 → 论文 Table 4 真实任务无法复现 |
| specificity HD_Flu | 无数据 |
| CDR dyMEAN/IgGM | 需结构预测 + 独立 repo |
| humanization HuDiff/IgCraft | 需独立生成 pipeline |
| ~~AbLang2（CDR）~~ | ✅ 已跑通（见下 2026-07-08 更新） |
| ~~pairing p-IgGen/LiChen~~ | ✅ 已跑通（见下 2026-07-08 更新） |

**FLAb 结论（32/32 完成）**：g6 Kd grammar_v2 600M **0.617** 略领先；g6 ER ESM-2 **0.820** 最高；trastuzumab Ophiuchus **0.612** 最高；四数据集 mean ESM-2 **0.647** > Ophiuchus **0.645** > grammar_v2 600M **0.635**。**口径修正**：论文 Table 4 = GDPa1 developability，本地 FLAb 四数据集是独立 sanity-check，非 Table 4 复现。

**2026-07-08 外部 baseline 接入 ✅**
- **AbLang2 CDR**：正确 Zenodo 地址 `records/10185169/files/ablang2-weights.tar.gz` 经代理 `100.68.162.212:3128` 下载（159MB），解压到 `ablang2/model-weights-ablang2-paired/`。`run_cdr_all.sh ablang2` 跑通：H1 **76.29** / H2 **70.69** / H3 **41.66**（n_total=3320）。修复 restore 输出 `<`/`>` token 剥离（`run_cdr_baselines.py`）。
- **p-IgGen pairing**：`pip install -e p-IgGen`（clone `oxpig/p-IgGen`），HF 权重 `ollieturnbull/p-IgGen` 经代理拉取。OAS holdout500 × n=8 → gen ImmunoMatch **0.686** / gen>ref 0.482 / chain 0.552 / diversity 0.695。
- **LICHEN pairing**：`pip install -e LICHEN`（clone `oxpig/LICHEN`），Zenodo `records/15917096/files/Model.zip` 权重。6 shard 并行生成 → gen ImmunoMatch **0.666** / gen>ref 0.498 / chain 0.622 / diversity 0.519。
- **接入代码**：`downstream/pairing_baselines/run_pairing_baselines.py`（heavy→light，`--baseline piggen|lichen`，支持 `--num-shards/--shard-id/--resume`）；打分复用 `downstream/comp_chain/eval_scripts/generation_eval.py`（需 `PATH` 含 hmmscan、`HF_DATASETS_CACHE` 可写）。
- 源：`output/downstream_generation/{cdr_baselines/*__ablang2.json, pairing_baselines/{piggen,lichen}_holdout500_metrics.json}`。

**文档（2026-07-08）**：README 同步更新 —
- `downstream/README.md`（AB 任务总览、baseline sweep、ckpt 路径、blocker）
- `downstream/flab/README.md`（runner / embedder 列表 / 复现命令）
- `downstream/infill/README.md`（CDR baseline + grammar_v2 + Ophiuchus 入口）
- `data/downstream/README.md`（flab_raw 数据路径）
- `benchmark/README.md`（AB / AB-FLAB 任务行）

## 2026-07-08 NbBench(A1) 收尾：删 encoder-only 误跑 + 引用统一期刊版 + 状态回填（纯文档/清理，零 GPU、零重算）

**背景（用户指示）**：确认 NbBench 是否搞定、baseline 是否全复现、encoder-only 可删、引用改成中稿的期刊版。

**已完成**
- **删除 encoder-only 误跑产物**：`outputs/nbbench/*/bioseq_esmc*_encoder/`（33 探针目录）+ `CDRInfilling/gen_bioseq_esmc*/`（3 生成目录）+ 5 个 `_summary_{,residue_}bioseq_esmc*.json`。这些是 `bioseq:` encoder-only 口径（只到 ESMC encoder mean-pool，未过 LLaDA），按 PROJGUIDE **不作 headline**，已按用户指示清除。
- **`scripts/summarize_nbbench.py`**：移除 encoder-only / `gen_bioseq_*` 的 `OURS_LABELS` 与发现逻辑，只保留 `ours_*_llada` + `gen_ours_*_llada`；重跑 → `_leaderboard.json` 不再含 encoder-only 行。
- **引用统一为期刊正式版**：NbBench 已 peer-reviewed 发表（*Mach. Learn.: Sci. Technol.* **6**(4):040502, 2025-11-27, DOI 10.1088/2632-2153/ae20ec；预印本 arXiv:2505.02022）。改 `RESULTS.md`(×2)、`NBBENCH_DESIGN.md`(×2)、`NBBENCH_EXECUTION_PLAN.md`(×1)、`PROGRESS.md`(×2)、`scripts/import_nbbench_results.py`(×2 注释/docstring，数字未动)。
- **状态回填**：`RESULTS.md` A1 主表 `Ours-BioSeq _pending_` → ✅ 已跑；`NBBENCH_DESIGN.md` §3 Ours 🟡 partial → 🟢 done；`NBBENCH_EXECUTION_PLAN.md` P1 resume 表 Running → Success、P2 全勾选；`downstream.md` A1 收尾注 + baseline 10/11（AbLang-L 有意跳过）。

**baseline 复现确认（磁盘核对）**：官方 10/11 模型（ESM2-150M/650M、ProtBERT、AbLang-H、AntiBERTa2(-CSSP)、IgBERT、AntiBERTy、NanoBERT、VHHBERT）× 12 任务全 `metrics.json` 齐；k-mer 9 标量 + one-hot 3 位点（分工互补）；AbLang-L 轻链有意跳过（官方数字转录 `outputs/external/nbbench_official.csv` 留档）。**A1 baseline + Ours 全闭环，无 pending 实验**。

**未做（可选补强，非阻塞，待用户定夺）**：NbBayesLM 外部 thermo 测试集接入、抗原条件 de novo VHH 生成（NbBench 无此任务，需另建实验）。


## 2026-07-10 Humanization 跑通（Ours grammar + Ophiuchus-Ab）

**背景**：`DOWNSTREAM_STATUS` 将 humanization 标为 blocked（HuDiff/IgCraft 未接）。数据其实已在 `data/downstream/humanization/humanisation/`（28-pair exact split），缺的是 grammar_v2 生成适配器。

**已完成**
- 新增 `downstream/grammar/humanization.py` + `masks.framework_generation_partial_mask`（IMGT FR1–FR4 mask，CDR 固定，light C-term 3 保留）。
- `humanization/README.md`；接入 `run_grammar_v2_variant_downstream_eval.sh`；修复 `humanize.py` 输出目录 mkdir；修复 smoke_test unpack。
- **Ours** `grammar_v2_esmc300m_integrated_llada` 全量 28×n=8：H/L FR AAR **62.53 / 63.61%**，VH/VL id **70.45 / 70.30%** → `output/downstream_generation/grammar_v2_esmc300m_integrated_llada_humanization_n8*`。
- **Ophiuchus-Ab** smoke 2 结构：H/L FR AAR **91.13 / 95.93%** → `output/downstream_generation/ophiuchus_ab/humanization_smoke2.csv`。
- 状态回填：`DOWNSTREAM_STATUS.md` / `downstream.md` / `downstream/README.md` — humanization 从 blocked → **complete**（HuDiff/IgCraft 仍为可选 backlog）。

## 2026-07-10 Humanization 接入本地 OASis

**已完成**
- 新增 `downstream/humanization/oasis_score.py`，复用 `project/oasis`（`score_csv.sh` + `OASis_9mers_v1.db` + `biophi` env）。
- `grammar/humanization.py` 默认 `--run-oasis`（可用 `--no-run-oasis` 关闭）。
- 修复：FASTA id 含 `sample_id_v{variant}` 防碰撞；`score_csv.sh` 强制 `biophi` env 绝对路径；biophi 非零退出但 xlsx 已写出时视为成功。
- Ours n8 打分：OASis Identity **53.67%**（H 50.49 / L 57.20；220/224）→ `oasis/output/grammar_v2_esmc300m_integrated_llada_humanization_n8_oasis.xlsx`，已合并进 `*_humanization_n8_metrics.json`。

## 2026-07-14 Baseline paper-alignment audit（不改模型、无新 Volc 作业）

**目标**：先修 baseline，不修改模型/checkpoint。协议不一致时以论文原值为主，并显式标注来源。

**已完成**
- 新增统一 provenance：`[P] paper_reported` / `[A] official_artifact_rescored` / `[R] official_code_rerun` / `[L] local_reimplementation` / `[C] local_control`，并生成 282 行论文值注册表 `outputs/external/paper_reported_baselines.csv`。
- T1 主 baseline 改为 Nature Methods Supplementary Table 4 原值；T2 主 baseline 改为 NAR-GAB Retention/Purity 原值。修复 LD indel 分支反向 lookup；本地重评分 6/9 在 ±0.03 内，其余明确保留差异。
- T4 删除 sparse-13 与旧 benchmark14 混表；TCRT5/GRATCR/ER 统一在论文 sparse-13 上重评分，14-pMHC 仅作 control。
- MINT 主 baseline 改为 Source Data Figure 2c–h / Table S1；修复回归 target inverse-transform。旧 SKEMPI/PDB-Bind Pearson/RMSE 与“领先”结论全部撤回，raw-unit rerun pending。
- CDR infilling 改引 Ophiuchus-Ab Table 2；修复 Ophiuchus fold 路径兼容。FLAb 改引 MINT Figure 3b nested 10×5-fold R²；runner 已修 nested CV，重跑 pending。
- NbBench 改为 Table 5 MLP/3-seed 主表；本地 sklearn single-seed probe 与论文分离，Paratope AUROC/AUPRC 不再混列。
- 论文正文/附录与 downstream 文档完成同源修订；跨协议 Ours-vs-paper 领先结论移除。历史日志中的相反结论由本条覆盖。

**验证**
- `scripts/test_baseline_protocols.py`：4 passed（`protenix_abtcr`）。
- `scripts/audit_baseline_provenance.py --strict`：paper registry、T2 LD、T4 sparse-13、MINT regression audit 全通过，0 error / 0 warning。
- Python compile 全通过。环境无 LaTeX engine；已做 brace、table/tabular 配对与列数静态检查，0 error。

**仍待 baseline-only 重跑**：MINT raw-unit regression、FLAb nested-R²、Ophiuchus-Ab exact local CDR；NbBench exact MLP/3-seed 是可选补齐。未提交任何新 GPU/Volc 作业。

## 2026-07-21 Public beta-generation Track A（14 pMHC × 1,000）

- 新增统一实现 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/{track_a.py,public_generators.py,giana.py,run_public_beta.py}`，权威说明为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`。
- benchmark 审计完成：14 targets / 1,312 references；RVR 895；已有 prediction columns 未进入 reference。
- 10 条 smoke：TCRT5、GRATCR、TcrDesign beta-only 均从官方权重成功生成；TCR-epiDiff 被 `reconstruction_only` guard 正确拒绝。
- RVR 1,000 与全量 14×1,000 均真实运行完成；最终 42,000 行、42/42 block 完整，未使用作者预存预测替代本地全量运行。
- 统一汇总：TCRT5 Valid/Unique/Exact1000/Recall/Recovery90/GIANA=`1/1/10/0.007622/235/335`；TcrDesign=`0.999643/1/4/0.003049/197/402`；GRATCR=`1/0.074214/0/0/7/4`。TCR-epiDiff 不排名。
- GIANA 42/42 `ok`；协议单测 16/16 passed。最终产物目录为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results`。
- TcrDesign α addendum：官方单条/双条 batch 等价 smoke 通过；对全部 14,000 条本地 generated β 各解码 10 条 α，`tcrdesign_alpha_generations.csv` 共 140,000 行，回连 β 0 missing/0 mismatch。官方未发布可运行 α/αβ evaluator 或 pairing checkpoint，因此仅保存定性 pair，不计算 α 指标、不改变 β 排名。协议单测随之更新为 18/18 passed。
