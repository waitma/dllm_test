# TCR T1 Binding

> 2026-09-30 历史作业配置整理：已归档 YAML 的位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；文中的旧提交命令及原路径文字保留作历史记录，复用配置前请先按索引定位。

<a id="uptcr-binding"></a>

## 2026-09-30 新增 UpTCR binding benchmark（当前用户决定）

Embedding澄清：作者已有预计算特征确实已发布，9份分类表所需文件名覆盖率100%，每类1个.pt已实测可加载；并非本benchmark阻碍。仅包外新序列的同款特征生成流程尚未定位，不影响当前Ours原序列评测。详见 [内容抽查报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/embedding_release_recheck.json)。

作为当前训练的 foundation model 的另一项 binding 评测，与既有 NM2025 独立列示。完全沿用作者发布数据的 majority/few-shot/zero-shot 各 train/valid/test；保留原始行、标签与顺序，不清洗、去重、隔离冲突、重采负例或重新划分。此决定覆盖此前统一数据池清洗重划分的建议。格式适配须保持每条记录及其所属split可追溯；遇到不支持记录报告错误，不静默过滤。

数据入口：[原样9份CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/released_tables/data/finetune/abpm)。majority train/valid/test=88315/15585/25993；few-shot=13345/2356/4044；zero-shot=132180/23326/8958。原SHA和审计证据见 [manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/audit_manifest.json)。

baseline 直接引用 UpTCR 论文中公开的对应任务成绩，标 [P]/paper_reported，记录来源及输入条件；不要求重新训练、运行 UpTCR 或补齐上游 embedding 才能评测我们的序列模型。UpTCR、DLpTCR、NetTCR2.0、ERGO-AE/LSTM、pMTnet、PanPep、EPACT 的未提取数字保持待获取，不以本地重评分或估算冒充论文值。

Ours 使用原 train 做需要的下游监督适配、valid 选模、test 报告；仅推理评分与监督适配明确分列。模型版本/checkpoint及具体适配方式在执行前从当前模型状态锁定，不自动沿用此前 AB 的 no-ESMC150000 或旧 TCR checkpoint。主指标 AUROC/AUPRC；如沿用作者代码的 AUPR，应明确为 average_precision_score，不与 NM2025 precrec 混用。输入字段包含 FVα/FVβ、CDR3α/β、peptide及HLA伪序列，实际使用条件需披露。

原始重复/冲突问题仅留作说明，不是清洗前置条件。已知发布 majority test=25993 而论文曲线 Source Data=16882；发布 zero-shot test=8958，而论文30%/50%曲线各4014。照常在原发布表评测，但在表下注明差异；只有一个发布zero-shot不能人为复制成两套已复现30%/50%结果，论文行和Ours发布集行分开标注。few-shot行数相同也尚未核验样本身份。允许并列文献对照，不宣称已证实完全同测试行集或据此作严格优劣排名。

2026-09-30 已启动首轮本地评测，入口 [run_uptcr_ours.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_uptcr_ours.py)。固定 no-ESMC150000（SHA256 `a6ce5c2c22b2e938f4573eb552b6a8ce1c2f42a3647a345adc98e6a5a052b048`），首轮完整FVα/FVβ+peptide，不含HLA，明确标αβp轨。冻结骨干，复用masked recognition/observed-residue mean和256/128 MLP；seed0、dropout0.3、Adam lr0.001/weight_decay1e-5、batch512、最多200epoch/patience15，使用原valid BCE选模，归一化只用原train。单seed不称论文多次均值。

9表314102行已逐条通过输入兼容性检查，未删行/改标签；39项测试通过。特征batch16/chunk512，逐行保存及断点续跑。输出 [uptcr_noesmc150000_original_20260930](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/uptcr_noesmc150000_original_20260930)，顺序few-shot/majority/zero-shot；已确认首批真实特征落盘，尚无完成指标。进展以status.json/run.log为准，不依赖baseline重跑。

> **2026-09-13 纠错更新**：当前入口的实现、运行时配置、NM2025 数据轨道与验收记录统一见
> [TCR baseline 纠错记录 §2 / §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。下方历史结果不等于新协议已评测；正式运行仍需通过 §8 gate 并使用新 tag。
> others/AS 第一版 GPU 五折已完成并复核，独立结果见 [RESULTS §0.1a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)；输入子集、track/输出和验收边界均见上述记录，不得沿用 β-only 榜单解释其结果。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.1 / §T1。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | T1 |
| 锚点论文 | Deng et al., *Assessment of computational methods in predicting TCR–epitope binding recognition*, Nature Methods 2025, [10.1038/s41592-025-02910-0](https://doi.org/10.1038/s41592-025-02910-0) |
| RESULTS 锚点 | `RESULTS.md` **§0.1**（主协议 = retrained 五折）+ **§T1**（副协议 = original-only） |
| 主指标 | overall AUPRC ↑（估计量 = R `precrec::evalmod` 移植，非 sklearn AP） |
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark/`（`retrain.zip` + `original.zip`） |
| 我们的入口 | `python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py`；Volc 走 `scripts/downstream/run_immune_fusion_repr.sh` 的 `t1` stage |
| 可引用性 | ⚠️ 有条件：retrained **unseen_independent 可引用**；retrained seen 两列仅作同口径相对比较（系统偏移未定论）；original unseen 须附 `TTAATHREK` 污染注 |
| 最后更新 | 2026-09-13 |

---

## 1. 任务定义与范围

给定 CDR3β + 肽表位，预测是否结合。输入是一对序列，输出是标量分数；headline 协议冻骨干、只训每折 MLP 头。两条协议（retrained 五折 vs original-only）**禁止混排**。

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | 当前输入/readout 以 [纠错记录 §2.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 为准；冻结骨干＋MLP |
| Ours-BERT | yes | 同上；BERT 目标产出的是表征，本任务不走迭代解码 |

### 1.2 不在本文档范围内的相邻任务

- IMMREP23 / IMMREP 2022 MicroAUC 主榜：已被 NM2025/AUPRC-retrain 取代，旧口径只在归档文档里。
- T3 few-shot NN：那是表征检索，不是结合分类；见 `TCR_T3_REPRESENTATION.md`。
- original-only 的 kNN / Random 控制行：diagnostic，不进 §0.1 主榜。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| Supplementary Table 4 original-model | figshare `original.zip`：seen 3 表位 / unseen 40 表位，neg=AS/PS/HS | AUROC / AUPRC / AUC0.1 | 现成官方模型零训练；多数方法 unseen ≈ 随机 | 是（副协议 §T1） | — |
| Supplementary Table 7 retrained 五折 | `retrain.zip` 五个 AS fold + 官方 Retraining_model 权重 | 同上，mean±sample-std over 5 folds | epiTCR 最强；unseen 全体贴随机 | 是（**主协议 §0.1**） | — |
| 全长 TCR / MHC 特征消融 | 同基准，部分方法吃 α/V/J/MHC | 同上 | 额外字段有时有用 | 否 | 我们 cdr3b track 只喂 CDR3β+肽，对齐论文 partial-feature 口径 |
| 近似去重敏感性 | 论文讨论 40–70% 近重复 | AUROC 塌缩 | 近重复抬高 seen | 部分（L1 诊断，不进 headline） | — |

### 2.2 指标定义（逐字对齐）

| 指标 | 论文定义 | 我们的实现 | 已知差异 |
|---|---|---|---|
| AUROC | 逐折算再平均，sample-std | sklearn `roc_auc_score`，同聚合 | unseen 上 6/8 基线与论文逐位相同（`RESULTS.md:583`） |
| AUPRC | R `precrec::evalmod` + `auc`（tie 感知非线性 PR 插值 + 梯形积分） | `common/metrics.py::precrec_pr_auc`；产物同时记 `auprc_precrec_mean` 与 `average_precision_mean` | sklearn AP 单侧偏高；表取 precrec。Ours 上梯形法再偏约 +0.0012（`RESULTS.md:560`） |
| 聚合 | 五折 mean±sample-std | 同 | 已排除「汇总池化 vs 逐折平均」作为 seen 偏移原因 |

### 2.3 论文未公开的关键细节

- Supplementary Table 7 那次训练是否等于现发布的 `Retraining_model.zip`：**未证实**。这是 §0.1 seen 系统偏移的剩余候选。
- 论文 AUPRC 是否 100% 等于 `precrec`：unseen 吻合到 5×10⁻⁵ 支持「是」；seen 因偏移不能当独立证据。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| 代码 | yes | 仓库内 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCREpitopeBenchmark`（commit `ec832b47`） | 见 `_download_manifest.tsv` | GitHub / ghfast.top | 已落地 |
| Retrained 权重 | yes | 随 figshare | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark/Retraining_model.zip` | md5 在 `retrained_protocol.py::EXPECTED_ARCHIVES` | Figshare DOI `10.6084/m9.figshare.27020455` | 已落地；**我们不重训它们** |
| 数据 | yes | 同上 | `retrain.zip`（md5 `5cf77befd7a07e0cb359540f050fb7b4`）+ `original.zip` | 同上 | Figshare | 已落地 |
| 官方评测脚本 | yes | 同上 | `tcr_binding/run_retrained_baseline.py` + `baselines/wrappers/*.py` | — | 本地 wrapper 调官方推理 | 已跑通 `[R]` |

论文值硬编码：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/retrained_protocol.py` `PAPER_EXPECTED_AS`（Supplementary Table 7-1/7-2/7-3）。全表：`.../outputs/external/paper_reported_baselines.csv`。

登记表：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv`

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark/` |
| split 构造 | 官方 zip 原样；`resolve_test_member` 无歧义。neg-source headline = **AS** |
| 入口命令 | 见 §8.2。embedder spec 必须是 `grammar:decoder:global:<ckpt_dir>` |
| Volc 作业 | `eval_jobs/eval_immune_{270m,8b}_{bert,diff}_repr.yml`（`REPR_STAGES` 默认含 t1）；v3：`eval_v3_*_repr.yml` |
| 输出目录 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/<tag>/cdr3b/AS/summary.json` |

Headline ckpt：270m BERT `@49000`、270m diffusion `@42000`、8B BERT `@43000`、8B diffusion `@45000`。v3 七源另标，≠ 旧 270m。

### 4.1a 直接 recognition token 评分（2026-09-15）

新增入口：[run_relation_token.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_relation_token.py)。使用独立留存的 v5 49000 权重，本轮本地先测试 `others_cdr3b` 完整 AS 测试行集。调用现有 `TCRBindingQueryProtocol`，运行时 X 补全、recognition 强制 MASK；输入记录不带标签。两路遮盖后单次 `_denoise`，从该位置取 `logit(<binding>)-logit(<nonbinding>)`，不训练 MLP、不池化、不用 CFG。AUROC/precrec AUPRC 用原始 margin（避免 sigmoid 饱和造成并列）；阈值分类用两 token 条件概率 0.5，概率未校准。

五个 seen test 分区分别评价后取均值及样本 SD；两个 independent 集校验跨折原始内容 SHA256 相同后只评分一次。这不是训练五个模型，也不重复算五次独立试验。保留官方全部测试行及原始类比例，只缓存相同输入的推理结果，不去掉重复评价行。无下游拟合不代表排除了预训练重叠。

运行：激活 `pllm` 后切换 `protenix_abtcr` 环境，在仓库根执行 `python downstream/benchmark/tcr_binding/run_relation_token.py --checkpoint /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_49000_llada_20260913 --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_relation_token/ours_v5_49000_20260915_others_cdr3b`。输出目录必须不存在，防止覆盖历史；保存 manifest、逐行预测、独立输入分数和 summary。该实验按用户决定不列入 `RESULTS.md` 正式结果；原始数值只保留在输出目录中用于诊断。

### 4.2 口径切割线（⚠️ 引用前必读）

| 分组 A | 分组 B | 为什么不可同表 | 证据 |
|---|---|---|---|
| §0.1 retrained 五折 | §T1 original-only | 数据、训练、负采样、表位数全不同 | `RESULTS.md:507`；审计 `audit_2026_08_29/T1_binding.md` |
| retrained seen_test / seen_independent | retrained unseen_independent | seen 上 8 个 `[R]` 相对论文系统偏移（test 偏低 ~0.010、ind 偏高 ~0.008），机制未定 | `RESULTS.md:528-547` |
| sklearn AP | precrec AUPRC | 不是同一个估计量；AP 单侧偏高 | `RESULTS.md:560`；审计 §B |
| 精确配对重叠=0 | Lev≤1 近重复 | seen_test 12–13% 近重复；「零重叠」高估洁净度 | `RESULTS.md:549-558` |
| 跨 checkpoint 拼接的旧 8B 行 | 单一 ckpt `summary.json` | 已 VOID | `RESULTS.md:526` |

---

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

1. **每个已列 baseline 必须带上它自己的最佳已发表值，作为 `[P]` 主榜行。**
2. **`[R]` 官方代码/权重本地复跑作附行或附列，永不取代 `[P]` 主榜行。**
3. **不得因为复跑失败就把 baseline 从表里删掉。** 标 `未获取` + 原因。
4. **论文值协议与我们不一致时，同行披露协议差**，不靠删行回避。

检索范围：只覆盖本表已列 baseline，**不做全文献 SOTA 搜寻**。Ours 整块置于表末（`PROJGUIDE.md` §2.4）。

> ✅ **2026-09-02 已补**：§0.1 主榜 baseline 现为 8 行 `[P]`（`PAPER_EXPECTED_AS` / Supp Table 7），原 8 行 `[R]` 降为附行。seen 偏移仍在附行上展示。数字只看 `RESULTS.md` §0.1，此处不复述。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| epiTCR | `[P]` 主 + `[R]` 附 | Supp Table 7 `PAPER_EXPECTED_AS["epitcr"]` | yes | yes | unseen 吻合；seen 复跑偏低 |
| TEIM | `[P]` + `[R]` | 同上 `teim` | yes（`Original_model/TEIM.ckpt`） | yes | unseen Δ AUROC +0.0011 |
| ATM-TCR | `[P]` + `[R]` | 同上 `atmtcr` | yes | yes | seen 偏移 |
| ERGO-AE | `[P]` + `[R]` | 同上 `ergo_ae` | yes | yes | **静默丢行**（seen_test 每折少 10–34 行） |
| NetTCR | `[P]` + `[R]` | 同上 `nettcr` | yes | yes | seen 偏移 |
| TCR-H | `[P]` + `[R]` | 同上 `tcrh` | yes | yes | 推理需官方 train CSV 做特征选择，不拟合 |
| TEINet | `[P]` + `[R]` | 同上 `teinet` | yes | yes | unseen Δ −0.0056，唯二未逐位吻合 |
| ERGO-lstm | `[P]` + `[R]` | 同上 `ergo_lstm` | yes | yes | ≈随机 |
| SETE / TEPCAM（original-only） | `[P]` fallback | Supp Table 4 | 部分 | 否：SETE pickle 维数不兼容；TEPCAM wrapper 未接通 | 标「论文值，未本地复现」 |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | [RESULTS §0.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-binding)（四轨当前结果）；旧 §T1 original-only 位于[整理前归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS_ARCHIVE_20260915.md)，不再作为当前副榜 |
| 任务锚点 / 出处层 | `benchmark/README.md` §3.1 T1 行 |
| 过程 changelog | `benchmark/PROGRESS.md` |
| Volc submit/cancel | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` |
| 本任务协议变更 | **本文件** |

原始产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/`

---

## 7. 已知缺陷与不可比口径

2026-09-30 汇总规则已执行：原 original-only 选择器的本地优先分支撤回，所有已列 baseline 的论文 `[P]` 保留为主行；本地 `[R]` 附列。论文缺项保持 missing，禁止本地填空。数据/输入/指标/聚合/选模等价性缺证时禁止跨协议 delta/rank。下游 unseen 与预训练 unseen 分开记录，见 [独立完整性报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。UpTCR 的当前执行状态及成绩仅见 [UpTCR 当前结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#uptcr-binding-status)。

2026-09-13 审计/整改事项 TCR-01–TCR-12（不是 12 个新 bug）统一维护在 [TCR baseline 纠错记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) §4，
不在本表复制。下表是历史缺陷记录；新旧结论冲突时以带日期的纠错登记为准。

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | §0.1 主表缺 `[P]` 行 | 未修（本轮文档轮补） | 只摆复跑会抬高 Ours 身位 | `RESULTS.md:509-518` vs `PAPER_EXPECTED_AS` |
| b | seen 两列 `[R]` 相对论文系统偏移 | 未定论 | seen 不得声称与论文对齐 | `RESULTS.md:528-547` |
| c | original unseen `TTAATHREK` 30 条精确污染 | 已披露，主表仍报官方 shipped split | kNN unseen 信号来自该表位 | 审计 T1 §A |
| d | AUPRC 估计量 | retrained 已用 precrec；original 曾用 sklearn，已换 | 混列会单侧抬高 | 审计 T1 §B |
| e | Lev≤1 近重复 | 已量化 | seen 泛化声称必须带 12–13% | `RESULTS.md:549-558` |
| f | 读出各向异性（全局 mean-pool、不重标定） | 已量化、决定不改口径 | 绝对值是下界；不能据此判 BERT vs diffusion 谁表征更好 | `RESULTS.md` §0.0(f) |
| g | 禁止跨 ckpt 拼接一行 | 已修 | 旧 8B 行 VOID | `RESULTS.md:526` |

### 7.1 2026-09-14：others 论文对照与 29k 历史对照的边界

- 新获取出版社 Supplementary Table 7 工作簿；官方 URL、原件 SHA 唯一登记在 [下载清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv)。Table 7-1/2/3 是大数据 β-only；7-4/6/8 是 others 多字段模型；7-5/7/9 是 β-only 模型在 others 上的结果。不能用前者代替后两组，也不能把论文转录说成 baseline 新复跑。
- 工作簿的 others seen-independent 混淆计数与公开数据存在差别：TCRconv-fullAB 为 524 正／310 负（合计 834），另一些方法为 581／581；公开 others 为 313／313（626）。seen_test 也存在计数差异。计数可能经过折平均/取整，不能直接反推出具体样本身份或断言整个公开数据错误；但尚无“论文与本地同一行集”的证据，尤其不应跨正例比例直接排名 AUPRC。fullA/fullB 的重复数值照录，未报告的 unseen 保持缺失。
- 29k 对照仅指**同一 v5 训练系列最近已有评测的 29000**，不是所有旧模型里步数数字最大的那个。三轨新旧 90 个预测文件逐列（除 score）、逐行及顺序一致；archive/唯一输入 SHA、头参数、fold/test 配置与声明的 input_protocol 语义一致。代码指纹存在差别，脚本明确输出 `strict_weight_only_causal_comparison=false`；故只作为历史控制，不声称重新锁定源码后的单变量实验。T2/T3/T4 与大数据 β-only 没有此旧模型的完整同协议结果，不以其他 run 拼补。
- 收数 helper [collect_tcr_49000_comparison.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_tcr_49000_comparison.py) 只读取已有产物，并调用现有 monitor 的报告/文件指纹核验。它区分当前 49000、旧模型、论文及历史 baseline，输出新 JSON；`--require-complete` 在任一当前任务未通过验收时拒绝最终收口，输出路径已存在也拒绝覆盖。不是新的评测协议，不改生产代码、模型、原始数据或 live gate。
- 10 项新增 CPU 测试通过；真实数据收集也已通过（当前未完成任务保持 missing）。`--wait-for-complete` 可在独立 tmux 中等现有 monitor 全部通过后收数；monitor 异常、消失/PID 复用或重复输出即报错，不启动/重试评测。完整数字与逐来源证据仅见 [RESULTS §0.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。多训练步数下的变化不自动代表统计显著、去污染泛化提升或全参数微调效果。

---

## 8. Agent 执行手册

2026-09-30 汇总使用独立新目录；命令、CPU 环境和 source-copy 验证见 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)。历史原始结果保持只读。

### 8.0 2026-09-13：指定 v5 49000 本轮执行入口

执行地点追加：others 三轨本机完整执行并通过独立结果复核；用户继续授权大数据 β-only 在本地接续，原云端 Queue 已取消至 Killed。直接复用已冻结 YAML 完整 Entrypoint，不改参数/数据/tag。逐预测验收与调度保护见 [审计 §9.24](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，新数值仅见 RESULTS §0.1。

追加执行/评分验收：独立 [preflight_tcr_suite.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_tcr_suite.py) 复核 others 三输入轨的全折原始行集一致、四轨真实特征→MLP→预测→指标链路、标准化仅使用内部训练子集且不改写特征。小样本每轨 32 train / 16 test、2 epoch，只验证执行，不能作为正式性能。CPU oracle 覆盖正向/反向/平分 AUROC；总体验收与云端状态见 [审计 §9.22](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。T1 正式配置/输入/旧专属 gate 未因 T2 修正而改变。

- 本轮四轨：大表 `cdr3b`、同 others 行集的 `cdr3ab / others_cdr3b / others_longab`；全部 AS、5 folds × 3 test sets。输入仍为 masked recognition，冻结完整 fusion 后训练 MLP；不改原始数据，不进行全参微调。
- [binding preflight](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_tcr_native.py) 已遍历每轨全部唯一输入并做真实 GPU batch=1/16、倒序及最长样本前向，均通过；55 项现有 CPU 回归通过。权重固定为指定 49000 快照，不追随训练后续保存。
- [passed.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/binding_preflight/passed.json) 固化权重、输入协议与数据身份。生产启动再次校验权重及协议 SHA；独立新 tag 不读取旧评测缓存。此门禁不证明完整预训练去污染。
- 用 [prepare_tcr_native_jobs.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/prepare_tcr_native_jobs.py) `--stage binding` 生成配置（已生成，不可重复覆盖）。提交与实时状态唯一见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)；正式数值只回填 RESULTS §0.1，旧 29k 等不进入本轮汇总。

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

- [ ] 已完成 [TCR baseline 纠错记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) §6 中对应实验的输入、缓存、数据与梯度验收；不得把下方历史命令跑通视作新协议完成

- [ ] embedder spec 含 `decoder` 或 `bioseq-llada`，不是 `bioseq:` / `grammar:encoder:`
- [ ] 协议是 **retrained 五折**（`--neg-source AS --fold all --eval-set all`），不是 original-only
- [ ] AUPRC 读 `auprc_precrec_mean`，不是 sklearn `average_precision_mean`
- [ ] 一个 Ours 行的 AUROC 与 AUPRC 来自**同一个** `summary.json`（禁止跨 ckpt 拼接）
- [ ] BERT/diffusion 都允许（表征任务）
- [ ] strict load 报告 0 missing / 0 unexpected
- [ ] `test -f ${CKPT}/model.safetensors`

### 8.2 提交命令

最新新增 LongAB 输入：`--track others_longab`，运行入口为 [eval_tcr_others_longab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml)。激活 pllm 后执行 `bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/volc-no-proxy.sh ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_longab_v5_29000_nonspot.yml`；执行前确认同 tag 无在途任务，勿重复提交。字段、长链校验/池化边界和独立产物以 [TCR 审计 §9.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 为准；状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，指标见 [RESULTS §0.1c](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。旧模式仍可用，不能复用旧缓存运行新协议。

2026-09-13 新增同 others 行集的 v5 β-only 对照：`--track others_cdr3b`，不是默认 `cdr3b` 大数据集。协议、验证、独立 tag 与当前状态统一见 [TCR 审计 §9.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；作业入口 [eval_tcr_others_cdr3b_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs/eval_tcr_others_cdr3b_v5_29000_nonspot.yml)。提交前查询同范围在途任务，避免重复；使用 `bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/volc-no-proxy.sh ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_tcr_others_cdr3b_v5_29000_nonspot.yml`。

2026-09-13 用户随后授权在本地繁忙时提交 1 张非闲时 GPU，当前入口为 [eval_tcr_others_cdr3ab_v5_29000_nonspot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_nonspot.yml)，已提交任务及独立 tag 见 [TCR 审计 §9.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；再次提交前先核查任务状态，不能重复启动同 tag。此前取消闲时与本地选择为 §9.4 历史记录，没有自动重提已取消任务。

此前 others 闲时配置为 [eval_tcr_others_cdr3ab_v5_29000_spot.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs/eval_tcr_others_cdr3ab_v5_29000_spot.yml)；历史作业、输出 tag 和抢占恢复边界见 [TCR 审计 §9.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

此前 others/AS 非闲时入口为 [eval_tcr_others_cdr3ab_v5_29000.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs/eval_tcr_others_cdr3ab_v5_29000.yml)；作业与输出身份统一见 [TCR 审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。下方为历史 β-only 示例，不是 paired 配置；常规 GPU 作业遵循根 AGENTS.md 的 eval_jobs 流程，当前单卡非闲时执行以 §9.5 为准，§9.4 仅保留先前未执行的本地选择。

历史 wrapper 命令（供核对配置）：

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
source "${ENV}/../protenix_abtcr/bin/activate" 2>/dev/null || source activate "${ENV}"
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000
TAG=ours_fusion_270m_diff_42000
bash scripts/downstream/run_immune_fusion_repr.sh "${CKPT}" "${TAG}" 16
# 只要 T1：REPR_STAGES=t1 bash scripts/downstream/run_immune_fusion_repr.sh ...
```

Volc：

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_repr.yml
# 提交后立刻记 task_id → PROJECT_PROCESS.md
```

官方 baseline 复跑（不进 `[P]` 主榜，只产 `[R]` 附行）：

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py \
  --name all --fold all --eval-set all --neg-source AS
```

### 8.3 收数

```bash
python -c "
import json, pathlib
p=pathlib.Path('/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained')
for s in p.glob('*/cdr3b/AS/summary.json'):
    d=json.loads(s.read_text())
    print(s.parent.parent.parent.name, {k:d.get(k) for k in d if 'auroc' in k.lower() or 'auprc' in k.lower() or 'precrec' in k.lower()})
"
```

读每个 eval_set 的 `*_mean` / `*_std`。论文对照：`PAPER_EXPECTED_AS`。

### 8.4 填表

1. 数字写入 `RESULTS.md` §0.1。baseline 先 `[P]`（Supp Table 7）再 `[R]` 附行，Ours 置表末。
2. 每个数字必须能指到对应 `summary.json`。**禁止编造、禁止跨 ckpt 拼接。**
3. 追加 changelog 到 `benchmark/PROGRESS.md`。
4. 历史 original-only 不并入当前 retrain 五折表；原数字保存在 RESULTS_ARCHIVE_20260915.md，归档不再追加新结果。新授权的不同协议必须独立标注，不借用旧 §T1 的“主排名”文字。

### 8.5 同步哪些文档

按 `.cursor/rules/downstream-doc-sync.mdc`。最小集合：`RESULTS.md` §0.1 + `PROGRESS.md` + 本文件（若协议有变）+ `PROJECT_PROCESS.md`（Volc）。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-02 | 初版。主协议锁定 retrained 五折。同日 RESULTS §0.1 已补 `[P]` 主榜行，本文 §5.1 同步改为「已补」。 |

### 2026-09-27 v2 launch provenance repin

The eight v2 T1 launch guards now pin the current query-protocol hash including source-file fingerprints; checkpoint SHA guards remain enabled. All eight guard commands passed locally. The job generator computes the expected hash when preparing YAML, rather than copying legacy constants. A later code change still fails the launch check. Full provenance is saved at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v2_immune_v6_eval_20260927/t1_protocol_repin_20260927.json`. No query labels, splits, features or pooling rules were changed in this repair.


### 2026-09-30 Foundation预训练暴露门禁

新增UpTCR应同时检查CDR3α/β、完整FV exact、epitope及联合键，并保护few-shot train支持池和各test/valid；详细方法、来源范围、实测证据与待办统一见 [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) 的“新增TCR下游暴露审计”。此门禁作用于新foundation训练数据和unseen结论，**不修改作者原split/标签、不阻塞按原数据接入附注比较**。published zero-shot不自动代表对foundation预训练也未见；历史checkpoint需独立披露。


2026-09-30更新：与foundation v6/v7的全量暴露审计已完成，结果有受体、表位及同一联合键精确重叠。详见 [完整审计记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#tcr-exposure-20260930)；运行中的原样UpTCR评测继续按既定协议，但不能把作者zero-shot直接解释为foundation预训练epitope-unseen，不追溯改写已有checkpoint。


### 2026-10-09 评测修复

native V5 suite的独立protocol_outputs原先没有comparison_AS.csv，五折推理已完成但write_rankings失败。修复时复制原基线表并记录SHA，仅恢复排名，不重训探针、不改变原始预测。
