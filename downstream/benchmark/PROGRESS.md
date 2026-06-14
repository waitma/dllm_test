# IRBench 过程记录

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
