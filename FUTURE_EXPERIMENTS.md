# FUTURE_EXPERIMENTS

后置的模型/目标/采样增强清单。本轮整合数据训练**不实现**这些；骨干（每链 ESMC/ESM2 encoder + LLaDA masked-diffusion decoder）保持不动，理解沿用 post-LLaDA global mean-pool。每项在真正启动时，按 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` 的条目模板补 run 记录（原因/目的/效果）。

相关现状锚点：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`。

---

## 判定原则

- 只有当整合数据版的 headline（干净数据 vs 现状）跑出来、且明确指向某个瓶颈时，才启动对应实验。
- 每项独立开关、小权重起步、单独消融；不改默认 baseline 行为。
- 需要重训的实验（L1/L2/L3/L4/L5、3.5）成本高，优先做零重训的 3.6 采样。

---

## L1 · 关系型位置特征（AF-Multimer relpos）

- 原因：当前多链信号只有 `position_ids_chain`（链序号）+ `position_ids_inner`（链内位置），缺 `entity_id`/`same_entity`/相对链索引/跨链 bin。逐链独立 encoder 是多链关系学习的最大缺口之一。
- 落地要点：在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/grammar.py` 的 collator 产出 `entity_id`（同序列/同实体）、相对链索引、"跨链" bin；在 LLaDA decoder 注入为 attention bias 或加性 embedding。破同源对称性，直接告诉模型两残基是否跨链/同实体。
- 预期验证：PPI（MINT GeneralPPI）与 binding 表征、条件生成质量；relpos 开/关消融。
- 成本：低（仅特征 + 注入），需重训。

## L2 · 理解目标进预训练

- 原因：现训练只有 masked 去噪一个目标，无任何理解头；理解全靠下游 post-LLaDA pooling 的"涌现"。
- 落地要点（各自可开关、小权重）：
  - relation 类型分类头（复用 grammar relation token）；
  - 二元 interact 头（PLM-interact 式，需显式负样本，与去重协同）；
  - cognate 对比对齐（InfoNCE，CALM/CLIP 式：Ab↔Ag、TCR↔pMHC、PPI A↔B 拉近、错配推远）。
- 预期验证：当 post-LLaDA pooling 的理解 headline 不足时，量化各头的增益；需负样本策略确认（STRING 置信度阈值 vs MINT 负采样）。
- 成本：中，需重训 + 负样本工程。

## L3 · REPA/REED 表示对齐（对齐 Protenix pairformer）

- 原因：DPLM-2.1 / REPA / REED 表明把扩散模型隐状态对齐到结构模型表示可显著提质（蛋白反折叠约 3.6x 提速、recovery/RMSD 更好）。这是"与结构预测 pairformer 对齐"的落地，无需从零训 pairformer。
- 落地要点：对同一复合物，用**冻结**的 Protenix pairformer（`/vepfs-mlp2/c20250601/251105016/project/Protenix-v2/protenix/model/modules/pairformer.py`，AF3 Algorithm 17，single c_s=384 + pair c_z=128）产出 clean single/pair 表示作 target，对齐 decoder 的（带噪输入）single 隐状态与新增 pair 隐状态（sim 损失，仅前几层）。
- 预期验证：PPI/binding 表征与生成增益；对齐层数/权重扫描。
- 开放项：用 Protenix-v2 现成 checkpoint 在线出 target 表示，还是离线蒸馏预测（算力/耦合度权衡）。
- 成本：中高，需重训 + 结构模型推理。

## L4 · 轻量 pairformer trunk + inter-chain contact 头

- 原因：显式建模"哪些跨链残基相互作用"，为 PPI/界面/binding 提供 2D 归纳偏置。
- 落地要点：decoder single 表示经 `OuterProductMean` 升 2D 得 `z_ij`，跑少量 triangle mult/attention 块，再（a）pair-biased attention 回注 token 流，（b）接 inter-chain contact/interface 头（D-SCRIPT/Boltz 式）。监督用 SAbDab2/STCRDab/PDB 复合物或 Protenix 蒸馏。
- 预期验证：inter-chain contact/interface 指标 + 对生成的增益。
- 成本：高（新增 2D 模块 + 结构监督数据），v2。

## L5 · encoder 内跨链注意力（MINT 式）

- 原因：把逐链独立 encoder 升级为跨链注意力（MINT：ESM-2 + cross-chain attention），从 encoder 侧建模多链关系。
- 落地要点：改造 encoder 为 MINT 式跨链注意力，或直接以 MINT 权重作可选 encoder。
- 预期验证：MINT GeneralPPI；与 L1 relpos 的增益是否重叠。
- 成本：最高（改 encoder + 显存/速度），v2 消融。

## 3.5 · BERT/GIDD 混合腐蚀

- 原因：现 `sample_bioseq_diffusion_noise`（`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py` 约 254-312 行）是纯 absorbing-mask（腐蚀位一律置 `<mask>`）。文献里 absorbing-mask 只在大词表文本明显占优；小词表（grammar vocab ESM2=49 / ESMC=80，残基本体仅 20）下 UDLM/uniform-noise 与 mask 差距可忽略、部分数据反超。蛋白正是小词表。
- 落地要点：加 `random_sub_ratio`（被腐蚀位里一部分替换为随机氨基酸而非 `<mask>`，一部分保持原样），`labels` 覆盖被替换/保持位；`compute_masked_cross_entropy` 的 loss mask 相应扩展；仅腐蚀 residue 位，结构/relation token 不动。
- 预期验证：纯 mask vs 混合腐蚀 A/B——held-out 去噪 loss、生成 recovery/validity、下游表征。参考 GIDD、UDLM、DiffusionBERT。
- 成本：中，需重训。

## 3.6 · Flexibility-Trap 采样与 remask（最易先试）

- 原因：现 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py` 是 `confidence-deterministic-linear`、严格单调不可回改。依据 ICML 2026 Outstanding Paper《The Flexibility Trap》（arXiv 2601.15165）：任意顺序/低置信度解码会绕开高熵"分叉 token"、过早坍缩解空间。映射到蛋白 = 系统性推迟 CDR3/paratope/TCR 特异性决定位。
- 落地要点：做 semi-AR block/顺序约束、区域感知顺序（骨架先、CDR/界面后并升温）、可回改 remask（ReMDM）。**落地约束（已定）**：做成可控开关——在 `BioSeqGenerateConfig` 加 `sampling_mode`（或扩 `decoding_strategy`），**默认保持现有 `confidence-deterministic-linear`**，新模式仅显式开启时生效，绝不改动默认 baseline 行为。
- 预期验证：用 diversity / novelty / nearest-neighbor 距离 / recovery@k（Pass@k 蛋白类比）对比 confidence-first vs 顺序/区域感知，看功能残基处熵是否被压。
- 成本：低——**纯推理零重训**，可在现有 ckpt 上直接试，是后续里最易先做的一项。

## D1 · 全长五链 TCR-pMHC 数据重建（整合数据版待办）

- 原因：当前两个 TCR 源都够不到 grammar 支持的完整 TCR-pMHC 布局（`<prots> MHC . B2M <protd> <binding> <prots> <pep> PEPTIDE <protd> <binding> <prots> <tcr> ALPHA . BETA <protd>`）。审计（见 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md` §"TCR / TCR-pMHC Data State Audit"）确认四个问题：**① CDR3-only 非全长；② 无 MHC+B2M（PISTE 仅 34aa HLA 伪序列）；③ 仅 β 无 α；④ PISTE binding/nonbinding 标签在渲染时被写死 `<binding>` 吞掉**。
- 现状锚点（本地已有、可重建的原料）：VDJdb `data/tcr/vdjdb_full.txt`（α/β CDR3 + V/D/J 基因 + `mhc.a` 等位基因 + `mhc.b`=B2M + `mhc.class` + epitope）、McPAS `data/tcr/McPAS-TCR.csv`（同级信息）、OTS paired clean `data/ots_paired_clean/final`（2.1M 全长配对 α/β，但无 pMHC）。缺 IMGT/HLA 与 Stitchr 资源；STCRDab summary 下载损坏（HTML 错误页）需重下。
- 落地要点：
  1. **全长 α/β 重建**：用 Stitchr + IMGTgeneDL（`pip install stitchr IMGTgeneDL`，`stitchrdl -s human`，批量走 Thimble）把 VDJdb/McPAS 的 `V/J 基因 + CDR3` 拼成全长可变域，替换现在的 CDR3 片段。
  2. **全长 MHC 重建**：用 IMGT/HLA 把 `mhc.a` 等位基因名（如 `HLA-A*02:01`）映射成全长 MHC-I 重链序列，并补上人 B2M 常量序列 → 渲染成 `<prots> MHC . B2M <protd>` 双链固定上下文块。
  3. **配对优先**：优先保留 VDJdb `complex.id` 关联的 α+β 配对记录，构造 `[mhc, b2m, peptide, tcr_alpha, tcr_beta]` 五实体 `bioseq.v1` 记录。
  4. **负样本处理**：修 `GrammarRenderer.encode` 的 TCR 分支使其读取 `record.labels["relation"]`（区分 `<binding>`/`<nonbinding>`），或在整合数据版里对生成式预训练**只用结合对**、把 PISTE 负样本留给判别式下游（IMMREP 式 AUPRC）。
- 外部可选源：TCR3d（结构复合物 + 亲和力，周更，作 gold/oracle）、EPACT（编译好的配对 αβ–pMHC，含 Full-TCR k-fold）、PT-recognition（配对 6-CDR + HLA + label）、10x dextramer（大规模配对 α/β + pMHC）。
- 预期验证：TCR-pMHC 条件生成 recovery/novelty、IMMREP 式 unseen-epitope 结合排序；全长 vs CDR3-only、有无 MHC 块的消融。
- 成本：中（数据工程为主：Stitchr/IMGT-HLA 重建 + 渲染器负样本修复 + 混合权重），需在整合数据版一并重训。
