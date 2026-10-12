# TCR T4 Generation

> 2026-09-30 历史作业配置整理：已归档 YAML 的位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；文中的旧提交命令及原路径文字保留作历史记录，复用配置前请先按索引定位。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.4。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | T4 |
| 锚点论文 | Karthikeyan et al., TCRT5 (TCR-TRANSLATE), Nature Machine Intelligence 2025, [10.1038/s42256-025-01096-6](https://doi.org/10.1038/s42256-025-01096-6) |
| RESULTS 锚点 | `RESULTS.md` **§0.4**（Setting B，跨模型主对照 = `bioseq_unseen_common`） |
| 主指标 | mean edit distance ↓；辅：seq-recovery ↑ / F1@K / Char-BLEU（后者跨组不可比） |
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_generation_bench/` |
| 我们的入口 | `python -m downstream.grammar.tcr_generation` 经 `scripts/downstream/run_immune_fusion_gen.sh` `GEN_STAGES=t4`；打分 `tcr_generation_bench/run.py --method file` |
| 可引用性 | ⚠️ 同集合不等于同信息/同预算，见 §7.2；旧 Ours 的 sampler 与结论边界见 §7.1。v5 精确数据成员核查已完成（§7.3），不是生成重评或所有历史语料无泄露证明 |
| 最后更新 | 2026-09-13（双链候选分组见 §7.5，已有模型使用的配对数据实物核查见 §7.6；完整生成审计/重评仍待做） |

---

## 1. 任务定义与范围

给定 pMHC（肽 ± MHC），生成能结合的 CDR3β。Headline 是 **Setting B 表位条件设计**。**BERT 禁止进入本任务。**

用户新增的两组生成清单见 §7.5：保留 β-only，并登记条件生成配对 CDR3αβ 的接入方案。双链输出的β投影仍可参与既有β指标；真正的配对指标尚待冻结配对参考数据，不把此登记当成新榜单已运行，也不与下述无条件全长 Setting C 混同。

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | 迭代 mask-fill 解码 |
| Ours-BERT | **no** | BERT 目标不产生可用的迭代解码路径；已产 BERT T4 数字不进 RESULTS、不当 headline |

### 1.2 不在本文档范围内的相邻任务

- Setting A 无条件 repertoire：历史/诊断；TCRT5 不做无条件；参考集 cap=200k 缺陷见 `benchmark/README.md` §4.1。
- Setting C 全长 α/β：TcrDesign SOTA；BioSeq 全长无条件，不进 Setting B 主榜。
- Track-A（`TCR_BETA_PUBLIC_TRACK_A.md`）：14 pMHC × **1000 raw candidates**，7–24 aa C/F-or-W 合法性，另一套指标（exact/recall/GIANA/Jaccard）。与 Setting B K=100 **禁止合表**。
- TCR-epiDiff：cite-only，官方生成路径非 de novo。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| sparse-13 主榜 | `benchmark_data_w_preds.csv` 13 pMHC（排除 simulation 保留项 `RVRAYTYSK/HLA-A*03:01`） | F1 / edit / seq-rec / Char-BLEU；官方预存 TCRT5/GRATCR/ER | 条件生成器可打到 d_edit≈4.4 | 是（`[A]` 重评分；**尚未与 Setting B 主表合排**） | 协议：K 与候选来源不同 |
| Supplementary Table 2 | 另一套协议 | 84 行 `[P]` | — | 登记，**不可与 sparse-13 合表** | 协议差 |
| held-out top-20 | `topk_val_df.csv` | 同上 | **这是验证集** | 我们当 target-rich 补充，不当论文主榜 | — |
| 独立 simulation | `RVRAYTYSK` | — | 保留给 simulation | 从 `bioseq_seen/unseen` 剔除；benchmark14 control 保留 | — |

### 2.2 指标定义（逐字对齐）

| 指标 | 论文定义 | 我们的实现 | 已知差异 |
|---|---|---|---|
| F1@K | unique_correct / 与 min(K, n_ref) 相关 | `common/metrics.py` TCRT5 套件 | Ours 全视图 F1=0 |
| d_edit | 到最近参考的平均编辑距离 | 同 | 跨模型主指标 |
| seq-recovery | native sequence recovery | 同 | — |
| Char-BLEU | 论文用 greedy 假设 | `bleu_hyp = greedy if greedy else gen[0]` | **跨组不可比**：官方行有 greedy，Ours/TcrDesign/TCRDiff/OLGA 用 `gen[0]` |
| n_pmhc | 易把类别成员数当成参与平均的数量 | 现印 `scored/total`（`n_pmhc_scored`） | 旧表 benchmark14 印 14 实为 7 |

### 2.3 论文未公开的关键细节

- Track-A 的 1000-candidate 分母 vs Setting B 的 K=100：两套活协议，适用范围不同。
- v5 train/valid vs held20 的精确成员核查已完成，见 §7.3；旧 grammar_v2 标注仍不能替代当前核查，也不能用 v5 结果外推所有历史模型。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| TCRT5 代码 | yes | — | `.../baselines/TCRT5_tcr_translate` commit `97aeab1008` | 见 manifest | GitHub | 已落地 |
| TCRT5 权重 | yes | — | HF `dkarthikeyan1/tcrt5_ft_tcrdb` + Zenodo 15068617 | — | HF / Zenodo | 已用于 `[R]` beam |
| 官方预存预测 | yes | 随论文数据 | `benchmark_data_w_preds.csv` | — | 随 TCRT5 数据 | `[A]` 重评分已跑 |
| TcrDesign | yes | — | `.../baselines/TcrDesign` + Zenodo 14545852 | 见 manifest | GitHub + Zenodo | `[R]` 已跑 |
| TCRDiff | yes | — | `.../baselines/TCRDiff` + `TCRDiff_zenodo` | — | GitHub + Zenodo 20586708 | `[R]` 已跑 |
| GRATCR / ER | 预存预测 | — | 同 csv | — | 论文 artifact | `[A]`；本地生成也成功（Track-A） |
| TCR-epiDiff | 代码+权重有 | — | `.../baselines/TCR-epiDiff` | — | 曾 DL_FAIL 后恢复 | **不进排名**（reconstruction_only） |
| OLGA | yes | GPL-3.0 | `.../baselines/OLGA` | 见 manifest | GitHub | `[C]` 无条件下限 |

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_generation_bench/` |
| eval json | `.../data/tcr_generation_bench/eval_conditional.json` |
| 入口 | `GEN_STAGES=t4 bash scripts/downstream/run_immune_fusion_gen.sh <ckpt_dir> <tag> <heavy_bs> <t4_bs>` |
| Volc | `eval_jobs/eval_immune_{270m,8b}_diff_t4.yml`；v3：`eval_v3_*_t4.yml`（BERT yml 存在但数字不进 headline） |
| 输出 | `output/downstream_generation/` 的 `designs.jsonl` + `outputs/tcr_generation_bench/` |
| 离线重评分 | `scripts/rescore_tcr_generation_offline.py`（不跑模型） |

Headline 解码：diffusion mask-fill，K=100。ckpt：270m diff `@42000`、8B diff `@45000`。

### 4.2 口径切割线（⚠️ 引用前必读）

| 分组 A | 分组 B | 为什么不可同表 | 证据 |
|---|---|---|---|
| Track-A 14×1000 | Setting B K=100 | 候选数、合法性规则、指标全不同 | `TCR_GENERATION_BENCHMARK.md` 顶部 override；`TCR_BETA_PUBLIC_TRACK_A.md` |
| sparse-13 `[A]` / Supp Table 2 `[P]` | Setting B 主表 | 另一套协议 | `README.md` §3.1 T4 行 |
| `bioseq_unseen` | `bioseq_unseen_common` | 前者成员随 run 变（2/6/8） | `RESULTS.md:728-730` |
| held20 | 论文测试集 | held20 = 验证集，且 ~48% BioSeq pair 泄露 | `RESULTS.md:768-770` |
| Char-BLEU ᵍ | Char-BLEU `gen[0]` | greedy vs 随机一条采样 | `RESULTS.md:725` |
| Setting A | Setting B | 无条件 vs 条件 | `README.md` §4.1 |
| BERT T4 | diffusion T4 | BERT 不适用 | §1.1 |
| `*_t4_unseen` 的 held20 视图 | 独立 `*_t4_held20` run | 不同 run | `RESULTS.md:783`；审计 T4 §2.2 |

`bioseq_unseen_common` 固定 6 个 pMHC：`FTDALGIDEY_A*01:01` / `KINMPMSVK_A*03:01` / `NENLDLQEL_B*40:01` / `SALPTNADLY_A*01:01` / `TPSVSSSISSL_B*07:02` / `TSDACMMTMY_A*01:01`。只在 run 覆盖全部 6 个时输出。

---

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

1. 每个已列生成器必须有其**最佳已发表值**作 `[P]`（或论文预存 `[A]` 若那就是论文主榜）。
2. `[R]` 本地生成附行，不取代。
3. TCR-epiDiff 不可复现也**保留 cite-only 行**，标 `未获取` 生成路径原因。
4. sparse-13 `[P]`/`[A]` 与 Setting B 同行披露协议差，**不合并排序**。

> ✅ **2026-09-02 已补**：`RESULTS.md` §0.4 在 Setting-B 主表之上插入了「已列生成器最佳已发表值」表（TCRT5/GRATCR/ER 的 sparse-13 `[P]`/`[A]`）。**sparse-13 与 Setting-B K=100 禁止合表 / 禁止合并排序。** 数字只看 RESULTS，此处不复述。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| TCRT5 | `[P]` sparse-13；主表现用 `[A]` 预存 | Nat Mach Intell + csv | yes | yes（HF beam `[R]` 在 held20） | sparse-13 ≠ K=100 Setting B |
| GRATCR | `[A]` 预存 | 同 csv | yes 预存；ckpt 本地也有 | Track-A 生成成功 | 同上 |
| ER-Transformer | `[A]` | 同 csv | 预存 yes | — | 同上 |
| TcrDesign-G | 论文 + `[R]` | bioRxiv 2026 / 本地 | yes | yes | held20 上极强，验证集不可当测试 |
| TCRDiff | 论文 + `[R]` | bioRxiv 2026 | yes | yes | 曾误把 benchmark14 行填进 held20 |
| OLGA | `[C]` | 无条件 | yes | yes | 不看表位；用于「有无条件信号」 |
| TCR-epiDiff | cite-only | Bioinformatics 2025 | 权重有 | **否** de novo | reconstruction_only |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | `RESULTS.md` §0.4 |
| 任务锚点 | `benchmark/README.md` §3.1 T4 行 |
| changelog | `benchmark/PROGRESS.md` |
| Volc | `PROJECT_PROCESS.md` |
| 本任务协议 | **本文件** |

产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/` + `.../outputs/tcr_generation_bench/`

---

## 7. 已知缺陷与不可比口径

### 2026-10-02T18:40:43+00:00 当前 reference 保护覆盖核对

读取当前 eval_conditional.json 的实际 ref_binders（benchmark14/held20 共 34 个条目），得到 67,013 个唯一 beta-core＋epitope 测试键；Setting B eval.json 的 44 个条目含 2,792 个测试键。两者与现有 data/tcr_native/dataset/t4_refbinder_blocklist.txt 比较，缺失均为 0；未将 train_binders 纳入测试键。当前最终数据已按既有 beta-core＋epitope 条件键完成全量复核，规则内残留为 0；不外推为全局单链近似隔离。最终结果以[本轮范围验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准。未修改评测输入、划分、评分或排行榜。


**2026-09-16 共享 sampler 依赖提醒**：`FusionBetaSampler.sample_records` 直接调用 `generate_bioseq`（默认32步）。本次 AB pairing 修复了该共享路径中 ESMC 不读取已接受生成残基的问题，准确状态约束及防参考泄露实现见 [AB pairing §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。T4 既有多步结果未按新 sampler 重评，仅保留修复前历史；RESULTS 已就地提示，不作修正版 headline。此轮没有启动 TCR 重评、改 TCR 输入协议或覆盖原始预测／分数，后续复跑需新 provenance 与产物目录，不能从旧 progress 混续。

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | Ours vs OLGA 无统计差别 | 已用配对 t 检验收紧 | 不得写「略优于随机」 | `RESULTS.md:747-764` |
| b | F1@100 = 0 | 历史结果 | 未找回参考集合中的精确序列；不等于证明所有新序列不结合 | `RESULTS.md` §0.4 |
| c | 训练语料与评测重叠 | v5 精确成员核查已完成，见 §7.3；更早模型/预训练与近似重叠仍未核验 | held20 不能称新表位测试，v5 结果不能自动外推到旧模型 | §7.3 |
| d | Setting A cap=200k | 修正版未覆盖线上 | novelty/JSD 虚高 | `README.md` §4.1 |
| e | Char-BLEU 跨组 | 已灰标 | 不可比 | `RESULTS.md:725` |
| f | n_pmhc 虚报 | 已修打印 | 旧表 n 不可信 | 审计 T4 §2 |
| g | infill 诊断：无条件 PWM 都输 | 已做 | 拟合失败，不是「学到先验」 | `RESULTS.md` §0.0(h) |

---

### 7.1 2026-09-13：当前生成链路与引用边界初查

本轮只读 [T4 入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/tcr_generation.py)、[运行脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh)、共享 mask/sampler、真实 eval JSON 与部分已有 metrics；未做 tokenizer/collator 动态检查、权重前向、完整论文核查或 GPU 重评。

- **主任务**：Setting B，给定 epitope 生成 CDR3β；不是主报完整 αβ，也不是在已知真 CDR3 上局部补空。无条件 repertoire、完整链模式和 infill 是另外的路径，不与主任务混排。
- **当前条件**：默认 wrapper 不传 `--with-mhc`，只用 `entry.epitope`；可选参数才传 `entry.mhc_pseudo`。没有输入 α、V/J 或参考 binder 序列。renderer 对该条件 record 输出固定 recognition `<binding>`，mask helper 保留关系/结构/条件。这在“期望生成结合者”的任务中是合法设计条件，不是 T1 的未知关系标签，也不保证实际结合。
- **目标与长度**：先按代码内经验分布抽取 9–23 的槽位预算，构建 `C A...A F` 常量 placeholder；目标 β 的所有残基位置都被设为生成位置，默认不保留 C/F anchor。不从 `ref_binders` 取目标长度或真序列；这也不是模型自主学习何时停止的已验证结果。最终输出长度/终止符与有效候选数须另外验收，不能仅凭槽位数断言输出恒定等长。
- **默认解码**：K=100/表位，seed=42，`max_iter=32`，`gumbel_argmax`，temperature=1，共享调用 cfg=0；冻结 checkpoint 采样，不在此训练生成头。输出非空候选即加入列表，并无凑足 100 条的重试，因此 K 是请求数，不是每个目标实际有效/去重候选数的保证。
- **模型输入边界**：CLI 虽读取包含 `ref_binders` 的 eval JSON，但条件采样调用只传 epitope、可选 MHC 和采样参数。共享 `_model_logits` 当前对整个目标生成区域施加 encoder corruption；这只是静态接线证据，仍需验证 remap 后 decoder/ESMC 两路每一步的实际输入，不能将其写成所有历史结果均已通过无泄露测试。
- **本地数据**：实际 eval JSON 共 34 条目（held20 20 / benchmark14 14）；`--eval-set unseen` 与现有标记文件交集为 9，不能把这 9 当作主表公共 6 个目标。held20 是验证来源；`bioseq_unseen_common` 的共同目标视图及训练 seen/unseen 身份是两件事，旧命名不证明对当前 v5 预训练无重叠。
- **共享 sampler 版本风险**：T4 的 `run_grammar_generate → generate_bioseq` 与 [AB pairing §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md) 已修复的多步状态路径相同。不能因别的任务修好了就把旧 T4 当作新代码成绩；逐产物代码版本和受控重评仍缺。未确认版本的历史多步 Ours 暂不作纠正后 headline；不因此宣称所有历史 artifact 必然受影响，也不波及走独立路径的官方 baseline。
- **结果解释**：已核对旧 270m@42000、v3 allchains@151k、genonly@44k 的 common-6 metrics，数值只在 [RESULTS §0.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 维护。这不是当前 v5 29k 的 T4 成绩。编辑距离/seq-recovery/精确匹配衡量与参考集合的接近程度，不能当作实验结合率；精确匹配为零也不等于所有新序列都不能结合。

**下一步审计 gate**：在任务特定生成语法下核验 v5 输入适配（不能直接复用 T1/T3 adapter）、参考真值与长度隔离、逐轮 sampler 状态/双路 encoder 输入、有效候选分母、训练 overlap 及 baseline 原协议。本轮不实施这些改动。总审计日志见 [§9.12](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

### 7.2 2026-09-13：baseline 如何生成，以及本地条件差异

本轮核对论文 Methods、官方仓库与本地 wrapper；仅研究/静态诊断，未加载权重或重跑。模型原生机制、本地配置与历史产物身份分开描述；当前源码不自动证明所有旧产物使用相同版本。

| baseline | 原生生成机制 | 本地接入边界 |
|---|---|---|
| TCRT5 | T5 encoder-decoder；输入 `[PMHC]{peptide}[SEP]{MHC pseudo}[EOS]`，自回归生成 CDR3β，beam search 保留候选 | 主表官方行为预存预测重评分；另有 HF 现场生成路径，默认 K=100、300 beams、max_new_tokens=25，不能混作同一产物 |
| GRATCR | Epitope-BERT 编码表位，TCR-GPT 解码器经交叉注意力逐 token 生成 β；beam search | TCRT5 论文 benchmark 使用作者 beam=1000 实现，以返回序号作为排名；本地主表预存行不是本轮新生成 |
| ER-Transformer | ER-BERT 的 EpitopeBERT / ReceptorBERT 经 seq2seq 训练，表位输入、β 自回归 beam 输出 | TCRT5 论文采用单氨基酸 tokenizer，另报补首 C / 尾 F 的 ER-Transformer+；本地 er 列与 + 身份须核查原 CSV 构建。作者完整示例还含独立 discriminator，不能推断本地主表经过该筛选 |
| TcrDesign-G | β模式：表位经 BERT/GRU encoder、带 attention 的 GRU decoder、beam search 生成 CDR3β | beta wrapper 不传 MHC/真实β；full 模式另走 β→α（条件含β）→V/J→完整链装配/结合评分，不是当前 β-only 行 |
| TCRDiff | 条件离散扩散；支持 pMHC 与 V 基因导出的 CDR1/2 特征，指定 α/β 长度后迭代去噪 | wrapper 将 V 留空，不提供真实α/β残基；联合生成αβ，仅导出β。默认α长14、10轮、temperature=0.1；β长度来自各目标参考 binder 长度分布 |
| OLGA | 按 V(D)J 重排概率模型随机生成 CDR3β，无表位条件 | 无条件控制，不等同 TCRT5 论文的 soNNia |
| TCR-epiDiff | 现有接入仍为 reconstruction-only，不能视为已跑通的 de novo 生成器 | 保留 cite-only 行；本轮未新增验证 |

**新登记的比较问题（未修复）**：

1. [TCRDiff wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/wrappers/tcrdiff_generate.py) 从 `ref_binders` 提取长度8–20的频数，按 `max(1, round(K * cnt / total))` 分配候选，总数不保证恰为K。虽未输入真实残基，却获得测试目标特定的参考长度信息，与 §7.1 Ours 固定先验不一致。“已知参考长度分布”可合法独立报告，但不能混称无参考 de novo 公平对照；未验证此差异实际贡献多少分数。
2. 该 wrapper 注释声称与 TCRT5 / Ours 相同 pMHC 条件不够准确：TCRT5 输入 pseudo 序列；TCRDiff 经 allele 库编码 MHC、提供 pseudo mask；Ours 默认仅表位，显式 `--with-mhc` 才加 MHC。本轮只纠正解释，不改生产行为或代码注释。
3. 同目标集合、同K不等于同协议。必须记录条件、参考长度访问、beam宽度/采样预算、EOS/槽位长度、C/F后处理、有效/去重数量、排序及外部 binding 筛选。原文 sparse-13 benchmark 生成1000条/目标，不等于本地K=100视图；保留论文成绩但不合并排序。
4. baseline 不需要统一的 `<binding>` token；期望结合通常由专用条件训练任务定义。我们的生成关系是设计条件，不是某条候选经实验确认结合的证据；T1 未知关系查询的 mask 规则不能机械套到 T4。

**证据**：[TCRT5 论文 Methods](https://www.nature.com/articles/s42256-025-01096-6)、[GRATCR](https://github.com/zhzhou23/GRATCR)、[ER-BERT 官方生成代码](https://github.com/TencentAILabHealthcare/ER-BERT/blob/main/Code/evaluate_seq2seq_MIRA.py)、[TcrDesign](https://github.com/XSLiuLab/TcrDesign)、[TCRDiff](https://github.com/zhangyumeng1sjtu/TCRDiff)。本地核对 [预存评分入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/score_official.py)、[HF / OLGA 入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/generators.py)、[TcrDesign wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/wrappers/tcrdesign_generate.py)、[原生 GRU 生成](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TcrDesign/utils/generate_tcr.py)、[TCRDiff 条件构造](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRDiff/scripts/generate/generate_pmhc_binding_tcr.py)。

**后续 gate**：逐产物核对配置后，分开“论文原协议”和“统一允许信息”的实验，固定预算与长度来源并测试参考信息隔离。本轮无模型指标，不改历史数字或来源层级。

### 7.3 2026-09-13：TCRT5 两份数据与 v5 train / valid 的实物重叠核查

**对象与身份**：不是旧 BioSeq 报告。两条 v5 diffusion 的真实启动 metadata 均指向 [immune_v5_receptor_completion](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion/dataset_manifest.json)，选择全部七源。当前训练入口按 manifest 加载 train=7,626,885、valid=102,308 行；`max_eval_rows_per_source` 是不生效的兼容参数，不把 valid 限成每源2,000。运行 metadata、入口和 helper 的路径/SHA 见 [本次 provenance](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/audit_v5_overlap_20260913/PROVENANCE.md)。TCRT5 的 `held20` 不因命名就成为我们的 valid；必须按实际内容判断重叠。

**方法与验收**：新增只读 [audit_tcr_generation_overlap.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/audit_tcr_generation_overlap.py)，复用现有 CDR3 anchor-normalization helper，不调用旧 source adapter 重建数据。扫描 train / valid 全部 manifest 分片，无抽样；逐片记录 SHA、行数和大小/mtime稳定性，来源以 manifest 为准而非保留旧别名的行级 source。真实 beta 的 region CDR3 优先，真实长链没有 regions 时用其已标注 core、逐条检查是否实际出现在 beta 链中；不把合成X链计作已观察序列。参考 junction 去掉首C/末F或W，prepared core 不重复裁剪。合成自测覆盖 split 隔离、β与配对的区分、重复去重、锚点、X、长链回退及缺 identifier 的 region 路径。

**计数口径**：表位命中、beta-core命中、(表位,beta-core)命中分别报告；配对检查不要求MHC相同且涵盖所有关系标签，比仅检查正例/pMHC配对更保守。额外保存真实C/F锚点可恢复时的完整junction精确命中、同MHC配对和binding正例子集。参考条目按目标计数，同core跨目标或不同junction可重复，不能把条目数称作全局唯一beta数。`held20` 有1条带X的参考序列，另标质量字段，未静默删去分母。

**最终数值（已完成）**：通过完整性门禁的 [full_scan_v2/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/audit_v5_overlap_20260913/full_scan_v2/report.json)，SHA256 `3848531613f0108e6c3978906809fc5eedb9d1afbfa810dbaf1b5dea10a6c072`。85个分片全部读完，逐片行数与manifest一致，总计7,729,193行，扫描约193秒；无未解析beta核心、identifier/region矛盾或split错误，当前blocklist覆盖全部参考配对。第一版 `full_scan/report.json` 漏查缺 regions 的真实长链，已被自身门禁拒绝，不得引用；v2已核验并补入训练38,055、验证2,984条真实长链，另2条OTS虽缺identifier但保留了可直接读取的region。

| 数据 | 对照v5分片 | 同表位目标数 | 参考β核心命中条目 | 同表位＋β核心参考配对 |
|---|---|---:|---:|---:|
| held20 | train | 20/20 | 44,204/68,482 | 0/68,482 |
| held20 | valid | 20/20 | 639/68,482 | 0/68,482 |
| benchmark14 | train | 7/14 | 0/1,312 | 0/1,312 |
| benchmark14 | valid | 4/14 | 0/1,312 | 0/1,312 |
| paper sparse-13 | train | 7/13 | 0/417 | 0/417 |
| paper sparse-13 | valid | 4/13 | 0/417 | 0/417 |
| 公共6目标 | train / valid 分别 | 0/6 | 0/129 | 0/129 |

- held20 的核心命中来自 OTS、TRAIT、tcr_native、tcr_papers，不能跨源简单加总。全局唯一命中core分别为train=37,790、valid=475；上表按参考条目计的比例为64.55% / 0.93%。确有完整C…F/W junction证据的命中条目分别为13,779 / 319，不把只有core的证据夸大为已见完整junction。
- 所有数据的binding正例配对及同MHC参考配对子集同样零命中。**held20不是整份被用作我们的valid，但其表位全部出现、很多参考beta曾在其他表位或无表位上下文中出现；它不支持“全新表位、全新beta”的生成主张。**
- benchmark14 中训练见过的7个表位为 `TDLGQNLLY / HPNGYKSLSTL / QEIRTFSF / QMMVKAGL / RTATKAYNV / RTATKQYNV / YERMCNIL`；valid见过其中 `HPNGYKSLSTL / QMMVKAGL / RTATKQYNV / YERMCNIL`。另外7个（公共6＋RVR simulation）在train和valid均未出现。公共6的旧名字虽来自历史划分，本次已由v5实物重新确认，不再只依赖旧标记文件。
- 参考质量另有发现：held20 共51条core含非标准字符，其中50条带 `*`、1条带 `X`；benchmark14为0。原文件和评分分母未改动；后续若过滤须另立协议/版本，不能静默改变与baseline比较的参考集合。

**后续评测解释**：held20作已知表位/参考配对未见的补充；benchmark14整体不是全部表位未见。保留论文sparse-13原视图，同时单列其中训练见过的7个与未见的公共6个；RVR simulation继续独立。上述建议不表示已启动v5生成重评。

**边界**：这是所选 v5 数据集的成员审计，不是逐 checkpoint 已消费批次回放；不审计 ESMC 更早的预训练语料，不覆盖近似序列/同源性或所有历史模型。现有 T4 blocklist 为 exact-pair 模式，不是表位级或全β级一刀切。即使参考配对零命中，也不能把已知表位/已知β重组成新的条件配对称为全新表位生成。没有改变训练数据、blocklist、权重、生成代码或既有模型分数。

**2026-09-15 用户疑问核对（去污染不等于表位排除）**：[T4 blocklist 构建代码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/build_t4_refbinder_blocklist.py) 的键是 `CDR3β core|epitope`，[prepared 过滤链](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/preprocessing/filters.py) 对带表位来源按该配对键排除；可选 cluster 扩展也在同一表位内扩展参考 core，并非删除该表位的全部记录。上述最终报告中七个 train 已见表位全部命中 manifest 来源 `tcr_papers`；`HPNGYKSLSTL` 另命中 `trait`，`RTATKQYNV` 与 `YERMCNIL` 另命中 `tcr_native`。这是报告的表位来源成员判定，不自动证明每次命中都带目标 TCR 或 49k 已消费全部记录。如果要求全14新表位泛化，必须在新训练前跨源排除对应表位并重建语料；事后删数据不能追溯性改变现有权重。当前仍按已见/未见分层解释，未执行新的清洗或训练。

### 7.4 2026-09-13：联合生成 αβ、投影 β 对照（区域语义已确认，待实现）

- **用户方向**：在表位条件下发挥模型的双链能力，生成α/β，再取CDR3β与β-only baseline比较；不是输入真实α帮助预测β，也不把既有无条件full-length模式直接冒充条件生成。当前conditional入口仍只构造β，尚未实现本提议。
- **代码依据**：[renderer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py) 按β→α发射；[目标mask接口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/masks.py) 可接收多个链索引，架构支持多链采样，但不等于已证明抗原特异性或配对功能。区域补全复用 [complete_tcr_chain](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py)：synthetic X在训练中不受corruption和loss监督，与待预测的MASK位置不是同一语义。
- **用户已确认**：未知FR及CDR1/2按训练保留固定 `X`，只有CDR3α/β使用生成 `[MASK]`；不是把CDR1/2也作为生成目标。X占位不能误入目标mask，CDR3目标也不能错误标为synthetic而排除生成。读取原文件后在运行时构造，不生成预处理评测副本。C/F/W锚点与core/junction边界、长度来源和MHC条件须在实现时明确登记；不能读取测试参考长度或真实α。
- **比较原则**：K个αβ候选只贡献K个β候选，保存pair对应关系，不凭真实α筛选、不按参考答案挑β或隐性增加best-of预算；额外计算量与其他baseline解码预算披露。当前两份**评测产物**仅保留β参考，能比较β生成质量，不能由此宣称验证了αβ配对或结合；这不等于上游源文件完全没有α/长链，见 §7.4.1。保留现有β-only独立对照，联合版独立命名产物；不覆盖原成绩。
- **状态**：目标区域的用户选择已确定；生产生成实现、CPU/真实GPU验收与完整评测仍待做。此次只确认协议、只读核查数据并更新文档，没有修改生成/训练代码或提交任务。

#### 7.4.1 当前评测输入与上游 CDR1/2 信息核查

本轮完整检查 [eval_conditional.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_generation_bench/eval_conditional.json) 的全部34条目，并以CSV解析器扫描两个上游文件所有行。不能将“评测产物未保留”说成“上游从未提供”。

| 数据层 | 实物核查 | CDR1/2 / 受体条件边界 |
|---|---|---|
| 当前 held20 / benchmark14 评测JSON | 各20 / 14个pMHC；每项字段均为 `epitope, allele, mhc_pseudo, category, dataset, ref_binders, n_ref`；参考分别68,482 / 1,312条CDR3β字符串 | 没有CDR1/2、CDR3α、V/J或TCR长链；参考β只用于评分，不进入生成 |
| [benchmark14 源CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRT5_tcr_translate/data/benchmark_data_w_preds.csv) | 14行，含参考β列表、作者生成预测与pMHC/来源等信息 | 没有CDR1/2、α、TCR长链或V/J字段；不能从其中读取真实CDR1/2 |
| [held20 源CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRT5_tcr_translate/data/topk_val_df.csv) | 92,674行；CDR3β全部非空，CDR3α非空25,364行；`TRA_stitched` 非空23,615行、`TRB_stitched` 非空87,676行，另有V/J等列 | 没有独立CDR1/2字段，但存在可供另行区域注释的长链信息；不能笼统说上游完全没有α/框架信息，也不将它们自动作为本轮生成条件 |

源CSV行数不是按pMHC去重后的参考条数；上述非空计数将空串/NA/NaN/None/null视作缺失，不表示序列质量或配对真值已通过验收。两个stitched列非空字符串长分别259–288 / 297–333，均包含同行给出的CDR3；[官方loader说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRT5_tcr_translate/src/dataset.py:101) 将其定义为full TRA/TRB。此次未做CDR1/2区域注释，也没有核实这些长链的测序/重建来源，不能直接称为实测真全长。

[准备脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_tcr_generation_bench.py:63) 将held20按表位/allele分组，仅提取CDR3β参考；benchmark14提取 `reference_translations`。因此当前联合生成保持未知CDR1/2的依据既是**所选输入没有提供**，也是**本轮定义为表位条件生成，而不是给定参考受体框架的受限设计**。以后若另建V基因/框架条件版本，须独立协议/输入列/结果，不能悄悄使用held20参考受体的V/J、stitched链或CDR3α来帮助主对照。

评测JSON SHA256仍为 `44af250d510254ea34e8f128e18df6508dd2f05814967d70c50fd400b10708b0`；两个官方源文件的本次SHA复核登记在 [下载/身份清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv) 的单文件条目。没有下载、改写数据或改变参考集合。

#### 7.4.2 benchmark14 的行结构与 β-only 任务定义

对照 [论文 Methods：Sequence representation](https://pmc.ncbi.nlm.nih.gov/articles/PMC12460172/) 与 [本地官方collator](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRT5_tcr_translate/src/dataset.py:356)：TCRT5在本研究生成方向使用peptide＋MHC pseudo-sequence作为条件，目标取 `tcr.cdr3b`。**输出是CDR3β，不是整条β链，也不是配对αβ。** 论文同时说明该表示可以扩展到双链、多CDR或全链；所以β-only是所发表实验的任务/数据表示选择，不是T5架构天然不能处理α。作者benchmark文件本身已经按此任务整理，不能把字段缺失归因于我们丢弃了benchmark14已有的α/CDR1/2列，也不能据此推断所有原始数据库记录均无这些信息。

原 [benchmark_data_w_preds.csv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRT5_tcr_translate/data/benchmark_data_w_preds.csv) 是**一行一个pMHC目标**，不是一行一条TCR，也不是T1式正负样本分类表。主要字段及当前接入：

| 作者CSV字段 | 含义 | 本地评测处理 |
|---|---|---|
| `Epitope` / `Allele` | 目标肽与HLA等位基因 | 对应JSON的 `epitope` / `allele`；另从作者HLA表查得 `mhc_pseudo`，不是从参考TCR推断 |
| `reference_translations` | 同一目标的已知CDR3β参考列表；CSV单元格为列表字符串 | 解析、规范化并按目标去重，存为 `ref_binders`，重算 `n_ref`；仅用于评分，不是生成输入 |
| `tcrt5_translations` / `gratcr_translations` / `er_translations` | 作者预先生成的各模型候选列表，不是参考真值 | 单独存入 `benchmark_preds.json`；不输入Ours |
| `Dataset` / `Reference` 等 | 来源数据库与研究来源元信息 | JSON保留 `dataset`；其余来源仍可回查原CSV |

实物示例：`FTDALGIDEY` / `HLA-A*01:01`，来源 `IMMREP23`；12条唯一参考β，当前JSON中包括 `CAISESQGSSNTEAFF`、`CASRGTASSYEQYF`、`CASSFGSSFSGELFF`。本轮另确认该行三列作者候选各有1,000条，不能将储存候选数当作我们主评测实际使用的K或新增现场生成结果。原 `n_references` 为如 `(12, 12)` 的字符串，评测使用的是实际解析/去重后的参考数量。

本地名称benchmark14覆盖14个目标；论文另将其中RVRAYTYSK保留为模拟筛选任务，其余13个为稀疏目标比较。因此应继续区分14/13/公共6视图及候选预算。联合生成αβ后投影β可沿用其参考评分，但该基准不能独立判定生成α的正确性或αβ配对功能。这里只补充字段解释与方法依据，不修改当前采样/评分协议或数据。

### 7.5 β-only / 配对 αβ baseline 调研与两组登记（2026-09-13）

**结论：存在可优先接入的双链条件生成 baseline，首选 TCRDiff 与 XSLiuLab/TcrDesign-G。** “双链”在本轮首先指配对CDR3αβ；能再组装全长不等于直接从未知框架生成完整受体。以下为方法/接入清单，非新增性能榜；不改变已冻结的任务范围和历史分数。

#### 7.5.1 原始方法与可用性证据

| 方法身份 | 双链生成机制 / 条件 | 官方资料与本轮接入判断 |
|---|---|---|
| **TCRDiff**，Zhang et al.，2026预印本 | pMHC条件的联合CDR3αβ离散扩散；官方还支持V基因衍生CDR1/2条件和后续筛选。无V基因控制须明确披露，不能冒充完整论文设计管线 | [论文](https://www.biorxiv.org/content/10.64898/2026.06.10.730756v1.full)、[官方代码](https://github.com/zhangyumeng1sjtu/TCRDiff)、[权重/数据](https://zenodo.org/records/20586708)。本地生成权重及MHC/V库存在；优先接入，双链新协议未通过运行验收 |
| **XSLiuLab/TcrDesign-G**，Diao et al.，2026预印本 | 表位→CDR3β，再由表位＋**生成的β**→CDR3α，属于顺序配对生成；另一模块预测V/J，完整管线可用stitchr组装长链 | [论文](https://www.biorxiv.org/content/10.64898/2026.01.15.699824v1)、[官方代码](https://github.com/XSLiuLab/TcrDesign)、[权重](https://zenodo.org/records/14545852)。本地β、β→α及α/β VJ权重存在；优先接入，不能把给定测试真β的α生成当成双链de novo |
| **TAPIR 的生成扩展**，Fast et al.，2023预印本 | 先生成α，再以目标＋生成α等条件生成β；依赖V/J选择，并用TAPIR打分筛选。TAPIR结合预测器本身不是生成器 | [论文及Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC10515850/)。本轮未确认可直接使用的官方生成代码/权重；列候选，暂不列可执行复跑 |
| **LSMTCR**，Zhang & Liu，2025预印本 | 声称包含表位条件α/β CDR3模块、V/J感知全长组装与结构联合建模；公开证据不足以把两个链模块直接等同于联合学习的配对分布 | [论文](https://arxiv.org/abs/2509.07627)、[作者页面](https://thuzrh.github.io/publications/)。本轮作者页Code指向 `github.com/username/LSMTCR` 占位地址并返回404；未确认可运行官方代码/权重，仅候选 |

**同名消歧**：[BiB 2025 的 TCRdesign（bbaf691）](https://academic.oup.com/bib/article/26/6/bbaf691/8405050) 不是XSLiuLab项目。其使用αβ资料/配对打分不代表生成器输出双链；该文生成目标为CDR3β，不能误放入αβ生成组。TCRT5发表版本亦为β-only，见 §7.4.2；没有因底层T5可扩展就把现成权重登记为双链。

#### 7.5.2 本地实现差距（仅静态核对，尚未修复）

- **TCRDiff已生成双链，但当前适配只留β**：[wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/wrappers/tcrdiff_generate.py) 将官方解码的α/β分隔结果取最后一段写入 `sequences`；旧β产物不能恢复已丢弃的α。β长度来自每个测试目标的 `ref_binders`，且逐长度 `max(1, round(...))` 分配不保证总数恰好K。新版本须保留原始配对，并改用与测试参考隔离的冻结长度先验和精确候选预算。
- **TcrDesign的full模式不是“已完成双链评测”**：[wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/wrappers/tcrdesign_generate.py) 可保留 `full_rows`，但公共 `sequences` 仍只取去重β；`check_ready` 只检查β权重，不能证明双链依赖齐全。本輪只读确认 `weights/rnn/beta/RNN_tcr.ep`、`beta_alpha/RNN_tcr.ep`、`alphaVJ/RNN_tcr.ep`、`betaVJ/RNN_tcr.ep` 实际存在，没有重新计算权重SHA或加载模型。
- **TcrDesign预算/长链语义**：[官方组合函数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TcrDesign/utils/generate_whole.py) 包含生成、清洗、V/J预测和结合评分排序。当前wrapper默认30个β×每β100个α，不能把排序后的前100与另一模型原始100个候选称为同预算；[官方CLI](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TcrDesign/tcrdesign.py) 的 `coding_seq` 默认False，只有开启相应选项才执行全长组装。不能仅凭 `mode=full` 声称输出已有αAA/βAA；历史产物是否实际含长链须逐份验列，不从当前默认反推全部历史运行。
- **X/MASK规则只约束Ours适配**：仍遵守 §7.4：未知FR/CDR1/2固定X，仅CDR3αβ参与MASK生成。外部模型沿用原生tokenizer/掩码，不强制把相同字面X解释成我们的固定未知；TCRDiff原生X可表示待生成位置。

#### 7.5.3 两组清单与报告边界

| 清单 | 计划纳入的方法（不是完成状态） | 状态 / 比较边界 |
|---|---|---|
| **β-only生成** | TCRT5、GRATCR、ER-Transformer、TcrDesign-G β；Ours v5 49000 β-only控制置末 | 既有作者预测重评分与现场生成继续分层；ER后处理另标。OLGA单列无条件控制，不称表位条件模型 |
| **配对αβ生成** | TCRDiff（联合）、XSLiuLab/TcrDesign-G（β→α）；Ours v5 49000联合αβ置末 | 前两者有本地接入基础，配对输出/指标尚未验收；Ours待实现。TAPIR / LSMTCR单列论文候选，代码/权重未确认，不填造双链复跑结果 |

TCRDiff和Ours联合版可以另报 **“αβ生成→β投影”** 以接入既有β参考比较，但保留原生生成组别，不能因此声称它们只生成β。新Ours汇总只用用户指定49000；缺项标待完成，禁止挪用旧checkpoint或β分数充当αβ分数。后续正式数值只写RESULTS §0.4；论文值与本地复跑及协议差异仍分层呈现。

**接入顺序与验收要求：**

1. 先适配TCRDiff配对导出及无参考长度先验；再组合TcrDesign-G原生β、α模块建立未结合打分筛选的顺序配对控制。可采用K个β候选各生成1个α的首版预算，具体beam/去重/失败处理在运行前冻结；这是本地组合协议，非原论文整条设计/筛选管线复现。完整大预算筛选版如保留，单列并披露生成量和打分量。
2. 每个候选必须保留稳定配对ID、目标pMHC、CDR3α和β、有效性/失败原因及原始次序；记录代码/权重/数据身份、长度先验、随机种子、实际条件和raw/valid/scored数。不能把两个链集合任意zip，不能为提高分数重配αβ，也不能使用测试真α/β、V/J、CDR1/2、结构或按目标统计的参考长度作为隐含输入。
3. **β对照沿用现有参考边界**：benchmark14及held20仍只有当前导出的β参考；可计算β投影的既有指标，不能据此给出真实αβ配对恢复率。common-6、sparse-13、held20与Track-A的原限制不变；pMHC条件和仅表位条件、长度条件、原始生成和reranking必须明列，不因为K相同就称完全公平。
4. **配对质量需要独立冻结配对参考**：本地TCRDiff官方 [IMMREP23测试正样本文件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRDiff/data/tcrpmhc/immrep23_test_pos_data.csv) 表头包含 `CDR3A, CDR3B, Peptide, MHC, TRAV, TRBV, CDR1A/B, CDR2A/B` 等，可作为后续来源候选；本輪只核查文件/表头，未完成全表质量、官方split、各模型训练重叠和相同目标覆盖审计，**尚未建立新配对测试集**。held20源文件有部分α也不自动获得独立测试身份。
5. 配对报告应同时给α/β边际指标与保留真实配对关系的指标。若使用最近参考距离，须对同一参考对求 `min_(a,b)[d(生成α,a)+d(生成β,b)]`，不可把不同参考受体的最佳α和最佳β拼成“匹配”。这些序列指标及预测器得分仍不等于实验结合功能。

本轮只完成研究、静态证据核对和文档登记；未改生成生产代码/数据、未下载模型、未执行新GPU前向或提交任务，没有新的单链/双链性能结果。

### 7.6 已有生成模型使用的配对测试数据：优先复用 IMMREP23（2026-09-13）

用户接受benchmark14只评联合输出中的β，同时希望另用**已有模型做过生成评测的真实配对数据**。本节推进 §7.5 的数据候选核查，不再仅停留于表头；不表示已建立新评测产物或全部去污染通过。

#### 7.6.1 两份可复用的官方来源

| 数据 / 优先级 | 已有生成实验依据 | 本次实物核查与推荐用途 |
|---|---|---|
| **IMMREP23，优先双链序列评测** | TCRDiff在该集做过CDR3αβ生成；论文Fig. 3 / Extended Data讨论恢复、相似性、motif等；[官方生成入口](https://github.com/zhangyumeng1sjtu/TCRDiff/blob/main/scripts/generate/generate_immrep23_tcrs.py) 明确读取正样本配对文件 | 本地 [immrep23_test_pos_data.csv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRDiff/data/tcrpmhc/immrep23_test_pos_data.csv)：598行正样本，20个pMHC、6个HLA，全为human；CDR3α/β、Peptide/MHC及四个CDR1/2字段均无缺失。适合作为第一版配对参考来源 |
| **TCRDiff 的 STCRDab 结构测试集，补充** | [论文Methods：STCRDab structural test set](https://www.biorxiv.org/content/10.64898/2026.06.10.730756v1.full) 用于TCRDiff / TCRDiff+ / ProteinMPNN的双链设计比较；结构来源为2022年以后收录复合物并按作者规则去训练重叠 | 本地 [STCRDab-22-test-data-rm-dup.csv](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/TCRDiff/data/tcrpmhc/STCRDab-22-test-data-rm-dup.csv) 原始38行；[作者提取脚本](https://github.com/zhangyumeng1sjtu/TCRDiff/blob/main/scripts/analyze/extract_stcrdab_generation_results.py) 排除 `7rm4, 8vcx, 8vcy, 8vd2` 后恰为论文34条、27个pMHC（21 human / 13 mouse）。有PDB ID及配对CDR，适合后续结构/跨物种补充，不与纯pMHC无框架条件直接混排 |

两个CSV均已在本地，无需重新下载大数据包；本次SHA登记在 [官方产物身份清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv) 的新增单文件项。数据来源存在且有生成实验先例，不等于现有Ours/TcrDesign/TCRDiff已在该新协议完成重跑。

#### 7.6.2 IMMREP23 上游身份、边界和字段

- **来源不是生成模型凭空构造的配对**：[IMMREP23官方仓库](https://github.com/justin-barton/IMMREP23/) 原任务是结合预测，公布 `data/solutions.csv` 的配对受体、标签和测试划分；TCRDiff复用正例做生成评测。官方 `VDJdb_paired_chain.csv` 是示例**训练**数据，不能误当测试集。
- 本轮通过HTTP只读解析官方 `solutions.csv` 全部3,484行（598正 / 2,886负），与本地正例按 `(Peptide, normalized_HLA, CDR3a_extended, CDR3b_extended, Usage)` 做**计数多重集**对照。只统一HLA前缀后完全相等，双方差集均0；这是选定字段/重数相等，不是整文件字节相等。最初未统一 `A*02:01` / `HLA-A*02:01` 的检查不能用来认定数据不同。
- **CDR3边界必须一致**：本地 `CDR3A/B` 对应官方 `CDR3a/b_extended`，包含边界锚点；不是官方不含两端的 `CDR3a/b` 核心列。实物首个正例：`GLCTLVAML / HLA-A*02:01`，α=`CAESIGKLIF`，β=`CSVGTGGTNEKLFF`。不能混用core与extended制造恢复率差异。
- **划分与重数**：598行包含548 Private、50 Public，并非全部Private；按 `(pMHC, CDR3α, CDR3β)` 有596个唯一项，两个项各重复一次。保留原始ID/Usage/记录关系，若评分去重必须明确并另存映射，不能静默把598改成596。官方README介绍提及21个pHLA，而此次实际正例文件覆盖20个；复用应锁文件版本，不用介绍文字替代成员清单。
- **CDR1/2确实提供，但有来源限定**：官方文件还含TCRa/TCRb、V/J和各CDR。按作者数据准备说明，长链由V/J/CDR3经stitchr构造，再用ANARCI注释区域；不能一概称这些长链/CDR1/2为独立实测序列。本轮主协议仍仅输入pMHC，参考受体字段不因存在而自动成为输入；Ours未知FR/CDR1/2继续固定X。若用V基因/框架条件，另列条件设计任务。
- **基础合法性检查**：本地正例CDR3α/β均为20种标准氨基酸，长度分别7–24 / 10–22；不能沿用某个baseline的8–20长度过滤而不披露丢弃覆盖。这里只查字符与缺失，未完成全部生物学质量/来源独立性验证。

**与benchmark14的直接衔接**：两份当前文件共有4个pMHC。其去重β参考集合逐目标完全相同，配对记录来自IMMREP23原行，不靠β序列猜配α：

| 表位（均 HLA-A*01:01） | IMMREP23配对行数 | benchmark14唯一β参考数 |
|---|---:|---:|
| FTDALGIDEY | 12 | 12 |
| SALPTNADLY | 18 | 16 |
| TDLGQNLLY | 33 | 32 |
| TSDACMMTMY | 11 | 11 |

同β可对应不同α，因此单用β去重/join会损失或错配真实配对。以上只证明这4个目标可追溯原始配对，不表示benchmark14其余目标均能补α；更不能把20目标配对集直接当成原common-6。

#### 7.6.3 “直接复用数据”不等于照搬所有实验条件

1. 推荐先用IMMREP23的正例建立独立配对参考视图，按原行读取后运行时组装Ours输入；原文件不覆盖，真实αβ只供评分。benchmark14的β评测保留，两份结果分别汇总；αβ序列指标仍不等于实验结合功能。
2. TCRDiff官方IMMREP23脚本按参考TRAV/TRBV与α/β真实长度分组，生成组内参考数的10倍；它是显式V/长度条件设计，不应笼统说作者“泄露答案”，也不能与我们的无参考受体主协议称同信息。原生协议复现与去掉参考V/长度、统一候选预算的控制应分列；后者采用官方权重但属于改动条件的本地重评，不能照搬论文分数。
3. STCRDab的34条成员已解释，但本轮没有下载/验收PDB与结构预测链路，也没运行作者评分脚本；结构引导baseline读取复合物骨架，与未知框架X版本的信息边界不同。两个已读分析脚本对38→34的处理并不一致，不能只凭文件名宣称完整原生harness已验收。
4. 尚需核查IMMREP23配对参考对我们v5实际train/valid，以及各baseline训练资料的重叠；先前benchmark14的β/表位核查不自动覆盖新增αβ。官方测试身份也不等于2026年模型未见。ID、Public/Private、重复项、core/extended、生成K与失败覆盖在运行前冻结。

本轮仅论文/官方代码研究、完整CSV解析、上游字段对照、现有数据SHA核验和文档更新；未创建新评测数据副本、未修改生产代码、未运行模型/提交GPU，无新增模型性能指标。

### 7.7 2026-09-14：完整生成后的缺失辅助指标与严格 JSON 写出冲突（已修复并恢复评分）

- 原故障：49000 benchmark14 完成全部 14 个 pMHC、每个 100 个有效候选和 1 个独立 greedy，随后在 [修复前 runner 快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/serialization_repair_v2/source_before/run_tcr_native_generation.py:143) 保存 metrics.json 时失败。原始 [designs.jsonl](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_benchmark14_pmhc_beta_v1/designs.jsonl) 与 failed manifest 原样保留；v1 没有正式 metrics，不直接收数。
- 只读 CPU 重评分与原日志一致：非有限字段仅为每个 pMHC 的 novelty_vs_conditioning 及其宏聚合。该辅助指标在缺少对应 conditioning 参考时按 [scoring.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/scoring.py:118) 返回 NaN；共用 [write_json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/ab_features.py:35) 使用 allow_nan=False，因而拒绝序列化。不是模型 logits/残基输出 NaN，也不是全体指标无法计算。
- 现有 suite 只到评分计算/有限主指标断言，没有覆盖完整正式结果的严格 JSON 序列化这一边界；这处执行验收缺口需要补回归，不能称此前工程预检已涵盖所有端到端失败路径。
- 修复边界：新增 [TCR 专属结果 JSON 边界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/result_io.py)。仅当真实 conditioning bank 缺少该表位参考且 novelty 为 NaN 时，将 novelty_vs_conditioning 写为 null；聚合仅在该指标有效计数为 0 时允许 null。新增 metric_availability.protocol=`tcr_generation_metrics_nullable_novelty_v1`、逐目标/视图不可用原因；保留全部主指标、pMHC、候选和 n_scored_by_metric。不填 0、不删除成员、不泛化清洗 NaN/Inf；其他非有限值仍报错。共享 scoring.py、AB write_json、sampler、训练数据与 checkpoint 未改。
- 恢复边界：入口 `--rescore-from` 不加载模型、不生成候选；先核对旧 failed manifest、旧 gate、指定旧 runner SHA，再逐项校验原模型/协议/采样器/评分器/参考数据身份及完整 raw。仅允许已登记 runner/JSON 边界的代码变化。新目录保留逐字节一致 designs，manifest/metrics 同时保存原 generation_identity、来源文件 SHA 与 CPU-rescore 标记；旧失败目录不覆盖，正式新输出一律 `_pmhc_beta_v2`。
- 验收：原有 81 项加新 19 项 CPU 回归共 **100 passed**，覆盖无/部分/全 conditioning、严格 writer roundtrip、非法非有限主指标、恢复不构造 sampler、拒绝不完整 raw/身份漂移/覆盖。首轮测试 fixture 只有一条 unique 序列触发 diversity_edit 无定义，已用两条不同序列修正 fixture；没有放宽生产断言。新版真实 GPU gate 保留原输入/目标/最长长度/占位扰动检查，另加真实 scorer→严格 JSON 写出→重读闭环，见 [generation_preflight_v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/generation_preflight_v2/passed.json)。原 suite 对写出边界的验收缺口由此次补充覆盖，不追改旧 gate 为成功新版本。
- benchmark14 已完成 CPU 恢复及 [独立全量产物审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/benchmark14.json)：14×100 raw 与 14 greedy 全覆盖，所有原参考评分/聚合重新计算一致。正式数值唯一见 [RESULTS §0.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)；这不解除 baseline 条件差异、held20 已见或序列分数不等价结合的限制。
- 后续 held20 完整生成及 [独立全量产物审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/result_audit_v2/held20.json) 也已通过：全部目标的原始候选、独立 greedy、固定上下文断言、训练长度先验及原参考重评分一致。沿用上述 v2 代码/gate，没有另一次生产修复；当前 β-only 两数据集均收口。时间状态见任务账本，数字与 baseline 边界仅见 RESULTS §0.4，不将 held20 已见表位控制称为未见泛化测试。
- 队列恢复与持续监控状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)；用户决定及实施登记见 [总审计 §9.26](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

## 8. Agent 执行手册

### 8.0 2026-09-13：现阶段 β-only v5 生成（双链延后）

执行地点追加：others 完成后两项生成任务均从云端 Queue 取消至 Killed，排入本机串行接续；原 YAML、完整 100 stochastic + 1 greedy 预算不变。实时状态和调度保护见 [审计 §9.24](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，不把任务取消说成生成完成。

运行解释与产物验收：max_iter=32 是上限；原 sampler 在所有目标解掩码后提前退出，实际 30/31 步属于正常完成。新增结果审计检查完整 raw/greedy 预算、训练先验长度、固定上下文记录、来源与原始参考的全部重评分；不人为补齐空余迭代、不为凑 32 步重跑。持久监控见审计 §9.24，未来完整结果需要审计通过才进入 49000 汇总。

**当前入口与条件**：[run_tcr_native_generation.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_native_generation.py) `--task preflight / benchmark14 / held20`；[生成适配器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/tcr_generation_v5.py)。固定指定 49000 的完整 ESMC+LLaDA 快照，只传 `epitope + mhc_pseudo(34 aa)`，不向生成 API 传参考 CDR3、V/J、长链或参考长度。MHC 输入条件须在与外部 baseline 比较时披露，不能冒称与 peptide-only 消融相同。

**2026-09-14 修复后入口**：默认 gate 已切换 `generation_preflight_v2/passed.json`，可显式传 `--gate /绝对路径/passed.json`，身份必须一致；输出固定新 `_pmhc_beta_v2`，拒绝覆盖。先激活 pllm、再 protenix_abtcr，设置 PYTHONPATH 为仓库绝对路径及该环境的 LD_LIBRARY_PATH。无 GPU 的恢复命令为 `CUDA_VISIBLE_DEVICES='' python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_native_generation.py --task benchmark14 --rescore-from /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_generation_bench/setting_B/ours_tcr_v5_49000_20260913_benchmark14_pmhc_beta_v1`；本次已执行，不能再次对相同 v2 输出重放。正常 held20 使用原 YAML 的 `--task held20`，默认走新 gate；参数、数据、候选预算不变。接续/监控启动器与固定计划见总审计 §9.26。

- **运行时布局**：MHC → peptide → completed β → unknown α，复用冻结 v5 profile 和训练补全 helper；未知 FR/CDR1/2 与整条 α 为固定 X。仅 β 的 CDR3 **core** 为生成 MASK；IMGT 框架边界的通用 C/F 锚点固定，最终输出为 `C + generated_core + F`。这两个格式锚点不是参考残基读取，不声称所有框架字符都为 X。α 占位不表示生成双链。
- **关系语义**：presentation/recognition 固定 binding 表示“生成期望结合给定 pMHC 的受体”条件，不是分类任务的真实标签；因此与 T1 的待预测 recognition MASK 不同。T2/T3 则使用 null/unknown，无结合条件。
- **长度与候选预算**：β core 长度只从冻结 v5 训练 profile 采样，按 seed/目标键确定；不查看 eval 参考长度。每个 pMHC 100 个随机候选，batch=8、seed=42、gumbel_argmax、max_iter=32、CFG=0；另生成 1 个独立 argmax（训练 profile 众数长度）用于 greedy/char-BLEU。共享 sampler 的 gumbel noise 固定，本版不宣称 temperature sweep。不是可变长度 EOS 解码，也不生成全长真实受体。
- **输出/评分**：100 个原始候选及 token IDs、合法性、索引、batch seed、固定上下文检查均保存；不去重补采、不 rerank、不按测试分数筛选。出现非法候选时保留原始输出并阻断评分。保存每目标的独立 greedy。benchmark14 控制、排除 RVR 的 paper_sparse13、held20 和成员明确的 v5 seen/unseen 分开；原 `bioseq_*` 分层重命名为 historical，不能冒充 v5 去污染。
- **v5 overlap**：逐目标绑定 §7.3 已完整核验的 full_scan_v2 身份，保留 train/valid 的 epitope、参考 β、pair 三类 seen 标记。held20 不称为未见独立验证；精确 pair 未重叠不意味着所有组分未见。指标只衡量序列相似性，不等价功能结合验证。
- **门禁**：真实 GPU A/G 占位扰动不变、decoder/encoder 遮蔽、全部采样步骤固定上下文、最长 core=82、正式 stochastic/32-step 上限路径通过；修复后另含严格评分 JSON 闭环，见 [当前 generation gate v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/generation_preflight_v2/passed.json)。预检不进入正式候选或质量表。
- **实施期间修正**：新 runner 最初把 overlap 报告“每源空错误字典”的外层 dict 误判为有错，改为检查子字典；新解码最初直接用 `GrammarTokenizer.token()`，实际 HF adapter 返回 `<base:n>`，现用真实 `encode_residues` 的 20-AA 逆表严格解码，并加入回归。两次失败均在本地 gate 阶段，未产生正式评分；没有更改共享 AB sampler 或预训练。
- **提交历史 / 当前执行**：原 `prepare_tcr_native_jobs.py --stage generation` 为独立非抢占单卡任务；对应云端任务现已取消，当前接续本地，不重复提交。运行前核验权重、数据、协议、代码及 overlap 报告 SHA。修复后完整产物写独立 `ours_tcr_v5_49000_20260913_<dataset>_pmhc_beta_v2`，v1 保留失败证据，既有输出不覆盖。任务 ID/status 见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

本版仅放行现有 Setting B 的 β-only 评测；§7.4–7.6 配对生成、IMMREP23 与结构实验仍是后续。下方旧执行命令/gate 不自动成为本轮 runner；原生 baseline 的条件/预算/数据对齐与可引用门禁继续单独核验。

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

当前 β-only 追加评分验收：独立 [suite preflight](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_tcr_suite.py) 对两个数据集各一个 pMHC 执行 batch=8、32-step、独立 argmax 与正式评分函数；预检候选写专用目录，不加入正式 K=100。CPU oracle 检查 raw/unique 计数、重复不补采、独立 greedy 的传递、空设计不计入 scored 分母。整体状态见 [审计 §9.22](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)，不能把云端排队当 gate 通过。此项不放行延后的 αβ 任务。

- [ ] 完成 §7.1 的协议/版本/输入隔离 gate；旧多步产物不充当新 sampler 或 v5 的验收结果
- [ ] 条件联合αβ版按 §7.4 保持FR/CDR1/2为X，逐步验证只有声明的CDR3目标参与生成；两路均不输入参考受体信息，不能以用户已确认区域语义代替实现验收
- [ ] 按 §7.5 分别标记β-only、αβ及αβ→β投影；配对ID、长度/条件隔离、候选预算和外部双链权重加载通过验收，真实配对指标另需冻结配对参考；候选登记不等于可执行或已有分数
- [ ] **模型是 diffusion，不是 BERT**
- [ ] embedder/生成路径走 decoder mask-fill
- [ ] 本轮要进 §0.4 主对照则必须覆盖 `bioseq_unseen_common` 全部 6 个 pMHC
- [ ] 不把 held20 当论文测试集
- [ ] 不把 Track-A 与 Setting B 合表
- [ ] `n` 用 scored/total
- [ ] `test -f ${CKPT}/model.safetensors`

### 8.2 提交命令

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
export GEN_STAGES=t4
bash scripts/downstream/run_immune_fusion_gen.sh \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000 \
  ours_fusion_270m_diff_42000 4 8
```

Volc：`bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_t4.yml`

### 8.3 收数

优先读 `bioseq_unseen_common` 的 `d_edit` / `seq-rec` / `F1@100` / `n=6/6`。离线重算：

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/rescore_tcr_generation_offline.py
```

### 8.4 填表

1. 跨模型只填 `bioseq_unseen_common`。`[P]`/`[A]` 与 `[R]` 分层。Ours 表末。
2. Char-BLEU 标 ᵍ 或灰。
3. changelog → `PROGRESS.md`。

### 8.5 同步哪些文档

`RESULTS.md` §0.4 + `PROGRESS.md` + 本文件 + `PROJECT_PROCESS.md`。

---

### 8.6 2026-10-06T23:07:56.130738+00:00 V7 50082原生A4 T4协议与执行（全量生成和评分完成）

当前 2026-10-07T00:39:13.894414+00:00：34条件/K100＋独立argmax全量生成和CPU评分均实际exit0，已完成，不重放。本轮完整报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json、评分根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/score_actual_completion_root_review.json；分数唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#tcr-generation。原始候选、无效项与控制均保留；没有修复、筛选或按test重新调参。全部主视图覆盖完整，optional novelty无条件参考时保持null并明确原因。评分很快结束，未捕获运行时PID；真实runner argv/done/exit0及完成产物已核对，不伪造PID。GPU完整生成的逐候选原始token/trace保留，输入扰动验收仍为既定CPU/GPU各两例，不冒充对全部候选做隐藏参考替换。下方按日期保留历史准备和运行状态。

2026-10-06T23:50:17.965727+00:00 V5比较来源固定：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/v5_comparison_binding_v1.json，实际只读核对cd00cd rc0。同一eval_conditional字节及主评分源码，旧benchmark14/sparse13/common6/held20成员与每目标reference数量、K100及主指标宏平均已核实。V5 per-target candidate_count=100/greedy_count=1，不能误当整轮总数。V5生成中间core并固定C/F、训练长度先验、batch8；V7生成含锚点/终止符的64槽位、batch1。V5未提供的optional novelty不作跨版本对比。保持原模型/训练预算/输入差异说明，历史common6不等于V7未见，尚未有完整V7 T4成绩。

当前 2026-10-06T23:39:56.588650+00:00：首次真实GPU输入检查通过e2be72 rc0，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_gpu_preflight_root_review.json；完整预测单次启动，实际PID397587/start2459151386、已见80/3434原始行，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_local_launch_root_observation.json。全量输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_all_local_gpu_20261006；未取得实际退出，评分尚未运行，不重放。新增CPU评分入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_tcr_generation.py，固定方案 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scoring_plan_v1.json；使用原序列指标，绑定sources/data/实际exit/全部原始行和独立argmax导出。准备验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scorer_preparation_root_review.json。下方按时间戳为历史。

2026-10-06T23:35:21.048373+00:00 评分辅助准备：/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_tcr_scoring.py 直接复用原score_conditional，不调用模型或注释。人工控制b643f1 rc0见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scoring_controls_root_review.json。保留每目标原始K100（含空串），明确原评分器对非空串及独立greedy为空时首非空候选回退的口径；每项指标记录实际分母，证明无定义时仅写null/原因，不删除目标。旧BioSeq视图统一标historical，不作为V7训练曝光结论。实际GPU与全量评分尚未执行。

2026-10-06T23:18:02.210187+00:00 全量入口已准备 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_tcr_generation.py；计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_generation_plan_v1.json，实际尚未启动。原34条目标×100随机＋34独立argmax=3434 raw rows，seed=42+原目标index×101+candidate_index（0–99随机，100 argmax）。同设备真实preflight、原源/data门禁和checkpoint绑定通过后才可执行。predictions逐候选保留raw64/终止/合法性/输入trace；designs保留每目标原100候选和单独greedy，空串与非法候选不删除。新人工artifact检查229451 rc0通过，202行、10类损坏拒绝、顺序/空值/独立预算/重复封存覆盖，临时文件已移除；根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_generation_preparation_root_review.json。

当前更新 2026-10-06T23:12:31.503718+00:00：真实50082 CPU输入验收103160 rc0通过，2类别各1例、32轮和扰动复算共128次forward，实际输入/输出不受隐藏参考内容/数量/长度影响；根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_cpu_preflight_root_review.json。新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_v7_tcr_generation.py --device cpu --output-root /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_input_audit_cpu_20261006 已实际运行，不可重复。GPU尚未验收、完整benchmark/评分未开始；下方此前仅合成准备记录按日期保留。

当前模型为no-ESMC A4/f011；原生训练布局按observed scope区分full与junction，直接支持无参考GenerationRequest。本次Setting B选择β junction目标，保留benchmark14/held20原34条目与100个随机候选＋1个独立argmax控制/目标，不扩展双链任务。现有基准清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/benchmark_inventory_v1.json；参考只供评分，输入API只有model、epitope、provided 34-residue mhc_pseudo。

模型输入：MHC/peptide条件原样可见、期望binding为固定生成条件；alpha为原生missing/null。β整个junction64槽位全部MASK，包括待预测protd，槽位容量只来自checkpoint config。没有参考长度、真实β/α或V/J输入。直接预测C/F-or-W锚点与终止符，不事后补锚点；所有raw64 token及解析序列、终止/合法性/失败原因均保存，无空候选删除、补采或rerank。解码复用已有原生提交token路径，32步上限、gumbel_argmax、temperature1、CFG0、candidate batch1；seed映射须在正式入口固定，不能依test挑选。

与旧成绩的区别必须并列披露：V5 core生成使用训练长度先验并固定C/F；V6使用full-beta后提取。本版是原生junction＋learned terminal，不能将差异仅归因架构，也不能把64容量说成参考长度。MHC仍使用该基准提供的34残基pseudo条件；不将其冒称完整MHC序列。语法合法性采用既有scoring.is_valid_cdr3b（仅格式，非功能结合），不调用序列注释。

实现 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_tcr_generation.py；人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_generation_adapter.py。4af089 rc0通过，根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/synthetic_root_review.json：tiny frozen f011、人工pMHC、30次forward、全部条件固定和目标遮蔽、5类有效/非法/空/缺终止候选原样保留、4类非法条件拒绝。未加载正式权重或执行实际T4生成/评分，不能作为质量结果。真实准入继续要求原两轮/源/data门禁、已绑定基准和真实checkpoint输入核查；不得沿用v5 overlap标记声明V7未见。现有数据只认证声明的test-only exact pair与T4 beta-core＋epitope规则；不新签审核、恢复已停VNAR/source-H3分支或宣称全query/few-shot/相似度隔离。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-02 | 初版。写死 Track-A vs Setting-B 禁止合表；BERT 禁入；主对照锁 `bioseq_unseen_common`。同日 RESULTS §0.4 已插入 sparse-13 published-best 表。 |
| 2026-09-13 | 新增 §7.1 当前入口/数据与结果状态初查，登记共享 sampler 版本边界并纠正 F1=0 的解释；仅文档，无新模型结果。 |

### 2026-09-27 incomplete v2 output restart

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_v2_generation_eval.py --restart-incomplete` checks checkpoint identity and archives an incomplete output directory under a timestamped sibling before a fresh evaluation. It is not target-level resume. Existing successful manifests or metrics are protected from replacement. Monitor YAML parsing now uses parsed Entrypoint content, preserving correct TCR tags and directory paths.


### 2026-10-09 评测修复

原native V5 step45000无约束采样在单β任务生成多个dot，链结构全部拒绝。新增single_chain_no_dot_v2协议：在logits层禁止dot，保留原冻结采样器、302容量、learned protd、温度、步数和种子，不截断/修补原样本。与旧协议分列；所有失败计入请求数，空候选质量指标为null并明确分母，不冒充0分或通过质量门禁。

补充解释：当前抽取器允许ANARCI不可用时使用regex；v2的CDR3有效率仅表示提取及C..F/W语法检查，不证明整条生成对象是单个完整β域。不得以禁止dot后的长对象直接宣称全长β有效。
