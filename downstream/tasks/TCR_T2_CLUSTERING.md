# TCR T2 Clustering

> 2026-09-30 历史作业配置整理：已归档 YAML 的位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；文中的旧提交命令及原路径文字保留作历史记录，复用配置前请先按索引定位。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.2 / §T2。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | T2 |
| 锚点论文 | **基础 A** Hudson et al., *Benchmarking unsupervised methods for inferring TCR specificity*, NAR Genom Bioinform 2025, [10.1093/nargab/lqaf150](https://doi.org/10.1093/nargab/lqaf150)。**基础 B** *A comprehensive benchmarking for evaluating TCR embeddings...*, Briefings in Bioinformatics 2025, [10.1093/bib/bbaf030](https://doi.org/10.1093/bib/bbaf030) |
| RESULTS 锚点 | `RESULTS.md` **§0.2**（Ours embed 曲线 / K-means）+ **§T2(a)(b)**（官方 9 方法校准） |
| 主指标 | 基础 A：Retention / Purity（Th=1，簇大小 > 1）↑；基础 B：ARI mean-over-K ↑ |
| 数据根 | 基础 A：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_clustering/`（universe **4,779** 配对）。基础 B：TCREmbedding `TCRantigenData_unique_test.csv`（9,033 unique CDR3β / 25 表位） |
| 我们的入口 | `tcr_clustering/run.py --method embed-threshold`（基础 A）+ `tcr_clustering/run_embed_bench.py`（基础 B）；Volc 走 `run_immune_fusion_repr.sh` 的 `t2` stage |
| 可引用性 | ⚠️ 有条件：官方 9 方法 `[P]` 与 §T2(a)(b) 在 4,779 universe 上 **9/9 校准可引用**；§0.2 基础 A **旧 embed 曲线仍在 5,368 universe 上，不可引用**；v3 基础 A 新 universe 可引用但勿与旧表混排 |
| 最后更新 | 2026-09-13（输入/标签审计；未改评测实现） |

---

## 1. 任务定义与范围

无监督地把 TCR 聚成特异性簇。表位标签**只用于打分，不参与聚类**。两个独立基础：NAR-GAB 阈值聚类（官方 9 方法）与 TCREmbedding 的 embedding→K-means。

当前主路径的输入/关系及旧分支例外见 §7.1；不能因 `GrammarEmbedder` 的泛化注释提到 binding，就断言本任务给定了 binding 标签。当前 v5 专项仅完成输入审计，不代表补全协议或模型性能已验收。

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | 表征：global mean-pool 后做阈值聚类或 K-means |
| Ours-BERT | yes | 同上；不走生成 |

### 1.2 不在本文档范围内的相邻任务

- T3 few-shot NN：有监督检索，不是聚类。
- TCR-VALID 原生 DBSCAN：已接过作对照，不是 headline 基础。
- 官方 9 方法排序：**Ours 不并入**该排序（协议是 embed 曲线在同 retention 处取点）。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| NAR-GAB Fig 3A Th=1 | curated IEDB+McPAS+VDJdb；**4,779 配对 / 8,395 序列 / 4,103 α / 4,292 β** | retention / purity / perc_clusters / perc_CDR3s | DeepTCR retention 最高；clusTCR purity 最高；TCRdist3 最均衡 | 是（官方 9 方法 `[P]` + 本地复算） | — |
| NAR-GAB 补充表 S2 | 同上，给聚类序列计数 | 由计数反算 retention | 与 Fig 3A 在 HD/LD 上**互换** | 披露矛盾；**采信 Fig 3A** | 不另开表 |
| NAR-GAB 其它 Th | Th>1 | 同上 | 敏感性 | 否 | headline 锁 Th=1 |
| bbaf030 embedding→K-means | 9,033 CDR3β / 25 抗原，K∈{10…100} | ARI / NMI / Purity | 评 19 个嵌入 | 是（用我们统一 harness，不擅自转录其 Supp Table 2 未复核数字） | 论文逐方法数字只引定性 |

### 2.2 指标定义（逐字对齐）

| 指标 | 论文定义 | 我们的实现 | 已知差异 |
|---|---|---|---|
| Retention | 被聚进 size>1 簇的序列比例；**官方 retention 不做 th 过滤** | `import_clustering_baselines.py` 复制该不对称 | 与 purity 的簇集合不同，属论文原口径 |
| Purity | 官方 R：多标签分子 / 单标签分母，理论上可 >1 | 校准路径用 multi-label，与官方一致 | shipped `build_pairs` 塌缩多数标签只影响 (b)(c)，purity 低估 ≤0.0023 |
| ARI/NMI | sklearn，mean-over-K 与 best-K | `run_embed_bench.py` | 基础 B 专用，不与 NAR-GAB Retention/Purity 合表 |
| 校准判据 | 实现是 `\|Δ\|≤0.03`，不是「吻合到两位小数」 | 同 | `RESULTS.md` 已改 |

### 2.3 论文未公开的关键细节

- Fig 3A vs S2 在 HD/LD retention 上互相矛盾。采信 Fig 3A：正文主图 + LD 容许 indel 应有更高 retention + 修复 universe 后我们的复算独立站在 Fig 3A 一侧。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| NAR-GAB 代码+预计算输出 | yes | 仓库内 | `.../baselines/TCR_Unsupervised_Benchmark` commit `ac767882` | 见 manifest | ghfast.top tarball | 已落地 |
| 9 方法预计算簇 | yes（仓库 `Data/Output_methods/`） | 同上 | 同上 | — | 随仓库 | 已导入；HD/LD 无预计算，本地 stringdist 现算 |
| TCREmbedding 数据 | yes | 仓库内 | `.../baselines/TCREmbedding/` commit `d5bf911f` | 见 manifest | api.github.com base64（代理不通） | 已落地数据；不转录未复核论文数字 |
| clusTCR / GIANA 源码 | yes | — | GIANA 本地已有；clusTCR dest 2026-09-02 由 ghfast 恢复 | 见 manifest | ghfast.top + 代理 | 校准仍用 NAR-GAB 预计算输出 |

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| 数据根 A | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_clustering/` |
| split 构造 | `scripts/prepare_tcr_clustering.py`；`curate()` **必须** `& d["PubMed_ID"].notna()`，否则 universe 变 5,368 |
| 入口 A | `python tcr_clustering/run.py --method embed-threshold --embedder grammar:decoder:global:<ckpt> --tag <tag>` |
| 入口 B | `python tcr_clustering/run_embed_bench.py --embedder ... --tag <tag> --algos kmeans` |
| 官方导入 | `python scripts/import_clustering_baselines.py` |
| Volc | `eval_jobs/eval_immune_*_repr.yml`（t2 stage）；旧 `eval_t2_clustering_globalfeat.yml` 指向 grammar_v2 ckpt，**VOID** |
| 输出 | `outputs/tcr_clustering/` ；embed 曲线 `outputs/tcr_clustering_embed/` |

### 4.2 口径切割线（⚠️ 引用前必读）

| 分组 A | 分组 B | 为什么不可同表 | 证据 |
|---|---|---|---|
| universe 4,779 | universe 5,368 | pandas/dplyr NA 语义差，不是数据库漂移 | `RESULTS.md:19`；审计 T2 §4.2 |
| 官方 9 方法 Retention/Purity `[P]` | Ours embed 曲线取点 | 协议不同；Ours 不进官方排序 | `RESULTS.md:627-628` |
| §0.2 基础 A 旧表 | v3 基础 A 新表 | 旧表待重跑，不可引用 | `RESULTS.md:630` vs `641` |
| 基础 A | 基础 B | 数据、算法、指标全不同 | `RESULTS.md:594` vs `627` |
| HD/LD 组均值 0.25/0.66 | Fig 3A 逐方法值 | 组均值误用 | 审计 T2 §1.1 |

---

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

1. 每个已列 baseline 必须有 `[P]` 主榜行。
2. `[R]` 复跑只作附行。
3. 复跑失败不得删行。
4. 协议差同行披露。

本任务官方 9 方法的 `[P]` 已是 Fig 3A 逐方法值；本地复算是校准，不是主榜替代。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| Hamming_Distance | `[P]` | Fig 3A；S2 计数与 Fig3A 互换已披露 | n/a（现算） | yes 复算 | 采信 Fig 3A 0.27/0.69 |
| Levenshtein_Distance | `[P]` | Fig 3A | n/a | yes | 采信 0.31/0.57 |
| TCRMatch | `[P]` | Fig 3A | yes 预计算 | yes | 无 |
| iSMART | `[P]` | Fig 3A 0.18/0.75 | yes | yes | 旧「仓库输出偏小」注已撤回 |
| GIANA | `[P]` | Fig 3A 0.25/0.65 | 预计算 yes；源码曾 DL_FAIL | 校准 yes | 0.66 是组均值污染 |
| clusTCR | `[P]` | Fig 3A 0.09/0.99 | 预计算 yes；源码 dest 已恢复 | 校准 yes | 源码不重跑 headline |
| GLIPH2 | `[P]` | Fig 3A 0.22/0.80 | yes | yes | 无 |
| DeepTCR | `[P]` | Fig 3A 0.92/0.63 | yes | yes | 无 |
| TCRdist3 | `[P]` | Fig 3A 0.44/0.42 | yes | yes | 无 |
| SCEPTR / TCR-BERT / ESM2 / ProtBERT | `[L]` 参照 | 基础 B 本地 | 部分 | yes | **不是** bbaf030 论文值 |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | `RESULTS.md` §0.2 + §T2 |
| 任务锚点 | `benchmark/README.md` §3.1 T2 行 |
| changelog | `benchmark/PROGRESS.md` |
| Volc | `PROJECT_PROCESS.md` |
| 本任务协议 | **本文件** |

原始产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_clustering/`

---

## 7. 已知缺陷与不可比口径

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | §0.2 基础 A embed 曲线在旧 5,368 universe | 未重跑 | **不可引用**；v3 新表勿与之混排 | `RESULTS.md:630` |
| b | Fig 3A vs S2 HD/LD 矛盾 | 已披露，采信 Fig 3A | 引用须带取舍 | 审计 T2 §1.3 |
| c | `build_pairs` 多数标签塌缩 | 已量化 | (b)(c) purity 低估 ≤0.0023 | 审计 T2 §4.1 |
| d | 读出各向异性 | 已量化 | 8B BERT 阈值曲线 tau 分辨率退化 | `RESULTS.md:650` |
| e | 训练-评测泄露（旧语料） | 数据侧已修，数字多为旧 ckpt | 引用须看是否 v2/v3 去污后 ckpt | `RESULTS.md:20` |

### 7.1 2026-09-13：binding 条件与标签使用专项审计

**结论**：T2 是受体自身的无监督表征/聚类，不需要给定 binding/nonbinding，也不应把待评估的真实 epitope 当作模型条件。保留训练格式的无条件占位前缀是另一回事，不等于给出结合标签。

当前实际路径与证据：

- [wrapper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_repr.sh:61) 的 T2 默认不传 `--use-alpha`。基础 A 的 [_embed_features](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_clustering/run.py:179) 只取 CDR3β；显式启用 alpha 时，decoder 后端调用 `embed_pairs(beta, alpha, peptide=None)`。基础 B 的 [_embed](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_clustering/run_embed_bench.py:107) 对 Ours 只取 CDR3β，标签列不传入后端；SCEPTR 的 V gene 是另外披露的输入信息，不能说所有 baseline 输入完全相同。
- [FusionGrammarEmbedder](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py:1105) 继承共享 `_record` / `_pool_single_decoder`，构造的记录不含 peptide 或 relation gold。当前真实 grammar 为：`<prots><null><protd><unknown><prots><tcr>{CDR3β}[链分隔符+CDR3α]<protd>`。方括号部分仅在 alpha 模式存在。训练 [append_null_context_prefix](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:418) 明确用固定 `<unknown>` 表示无条件关系，不是 T1 的 masked recognition 查询，不要擅自改成 `<binding>`、删掉前缀或一律 mask。
- 主阈值路径先生成 features/similarity/connected components，随后才用 `peptide` 标签评分；自动阈值根据无标签的 similarity/retention 选取，不按 purity 选优。基础 B 固定 K 扫描先聚类再读 `epitope` 评分。单独报告的 `best-over-K` 使用评价标签取最大值，只能作扫描诊断，不代表独立验证选出的最佳 K。

本次实际 CPU 检查：激活 pllm 后使用既有 protenix_abtcr 环境，加载 v5@29k 快照的真实 decoder tokenizer、本地 ESMC tokenizer 与原生 `RemapCollator`；调用实际 A-beta、A-alpha/beta、B-beta 特征函数，每种模式用源数据前 8 行，再改变所有评价标签并加入不同 Affinity 值。记录、decoder/encoder inputs、全部 batch 张量及非张量字段逐项一致；每行恰有一个 decoder `<unknown>`（本快照 ID=126386），无 binding/nonbinding，无 relation target，无 synthetic 残基。decoder 前向使用确定性替身：其 features、阈值/聚类分配及 K=2/3 的 K-means 分配不变；另确认固定簇下改评价标签会改变 ARI。**这验证接线和标签边界，不是加载真实权重后的 embedding 一致性或性能评测。** 检查段约 0.89 秒（不含环境/import 启动），未使用 GPU、未生成生产预测。

仍需纠正/验收的事项（不等于 binding 泄露）：

| 项 | 已确认事实 | 对新 v5 测试的限制 |
|---|---|---|
| T2-input | 直接将 CDR3 当链，没有调用训练侧 `_receptor_chains` / 冻结 profile；beta-only 没有 alpha 占位 | 尚未满足用户指定的运行时 v5 补全协议；须单独接入，不能照搬带 peptide 的 T1 adapter |
| T2-pool | 共享读出目前仅用 `residue_mask`，未排除 synthetic | 当前没有补全，不能称已污染；新增 X 补全时须同步 observed-only pool |
| T2-legacy-K | `run.py --method embed` 将真实 `peptide.nunique()` 传给 `n_clusters` | 是已知类别数的旧参照，不是完全 label-blind；当前 wrapper 不用此分支 |
| T2-input-validation | `_sanitize_chain('')→'A'`，`_sanitize_chain('AA?--')→'AA'`；会静默替换/删除 | 新适配应拒绝缺失/非法观测输入并记录缺链策略，不能冒充真实残基 |
| T2-artifact | 两个 `emb:` 入口只校验数组行数；主输出未保存 T1 等强度的输入协议/模型/行映射身份链 | 存在同长度错序/错协议无法检出的风险；未发现实际误复用证据，重评需独立 tag 和指纹 |

原始输入只读检查：基础 A 当前 4,779 行，CDR3β 长 8–21、CDR3α 长 6–23；基础 B 9,033 行，CDR3β 长 10–30。以上列均无空值、无被当前 sanitizer 改写的序列；所以 silent fallback 是代码风险，不是这两份数据已经发生的数据损坏。未审计 baseline 全部内部训练路径、v5 预训练重叠/近重复或全部生成任务，也未量化本次发现对历史分数的影响。

关键日志与本次数据/代码身份见 [TCR 审计 §9.9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮只审计和文档登记，不修改 Python、wrapper 或训练 renderer；共享组件修复需另行实施并做跨任务回归。

---

## 8. Agent 执行手册

### 8.0 2026-09-13：v5 runtime 本地重评入口（优先于下方旧命令）

执行地点追加：A/B 已从云端排队迁往本机串行；v2 YAML/gate/完整数据与评分参数不变，结果写原独立 v2 tag。迁移与运行入口见 [审计 §9.23](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

- 新入口：[run_tcr_native_repr.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_native_repr.py) `--task t2a / t2b`；使用 `tcr-v5:<49000快照目录>`，底层仍为 post-LLaDA decoder。A 取 β-only，B 取 β-only；没有输入表位、binding 或逐样本关系标签。
- [tcr_features.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/tcr_features.py) 在读取原始 CSV 后复用 v5 训练 profile/补全代码，按受体身份确定框架；缺失 α 为固定 X 伙伴。null prefix 的 `<unknown>` 是训练结构占位，不是待预测结合标签。池化仅包含真实观测残基，排除 X 补全、特殊 token 与 padding；不做 whitening。
- A 仍用 label-blind retention sweep，B 仍用 MiniBatchKMeans 固定 K sweep，不按测试分数选配置；best-K 只能作为诊断。A 使用 4,779-pair universe，B 使用现有 9,033 条 β。下述 v2 修正评分，不改变输入或模型。
- **评分复查修正（v2）**：[run.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_clustering/run.py) 原自动阈值取 5 位、CSV/JSON 取 4 位，会合并接近的 cosine 边，甚至用产物 tau 无法重建原簇；现在计算与序列化均保留精确 float 值。原 alignment 使用旧 5,368-universe 的锚点，现与已核实 Fig 3A 逐方法值一致，补齐 9 方法与 0.92 高 retention。最近点额外记录目标、实际 retention、差值；差值大于 **0.02（本地预先声明的容差，并非论文标准）** 时 `comparison_allowed=false`，只作最近点诊断，不能称同 retention 比较；即使通过也不证明输入 universe/链范围相同。
- 新增评分回归先复现旧阈值合并和 CSV 不可重建，再验证修正、label-blind 阈值、锚点及不可达声明，见 [test_tcr_scoring_audit.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_scoring_audit.py)。联合回归 81 passed；真实 GPU 全量长度、batch=1/8、倒序一致性重新通过，见 [v2 repr gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/repr_preflight_v2/passed.json)。旧 gate 保留但已失效；配置入口改为 `prepare_tcr_native_jobs.py --stage repr-v2`，输出 tag 为 `ours_tcr_v5_49000_20260913_runtime_v2`。
- 本轮是输入修正后的本地重评；预训练成员/近同源未完整排除，不能宣称无污染泛化或加入官方九方法排名。工程输入 gate 与论文可引用 gate 分开记录，后者仍未全部关闭。下方 §8.1 原“全部 gate 后再提交”仅适用于可引用实验；本次 local-control 作业不自动获得该资格。

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

- [ ] 新 v5 重评先完成 §7.1 的运行时受体补全、observed-only pool 和输入/产物身份检查；不得因旧 T2 分数存在而跳过。
- [ ] 无真实 epitope 或 binding gold 进入 Ours 输入；固定无条件 `<unknown>` 与 T1 recognition mask 明确区分；选定分支的标签不变性通过。

- [ ] embedder spec 含 `decoder` / `bioseq-llada`
- [ ] `data/tcr_clustering/` 配对常数为 **4,779 / 8,395 / 4,103 / 4,292**（否则 `curate()` 回归）
- [ ] 基础 A 与基础 B 分表；不把 Ours 插入官方 9 方法排序
- [ ] 不把 5,368 旧曲线数字填进可引用表
- [ ] BERT 允许（表征）
- [ ] strict load 0 missing / 0 unexpected

### 8.2 提交命令

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
REPR_STAGES=t2 bash scripts/downstream/run_immune_fusion_repr.sh \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000 \
  ours_fusion_270m_diff_42000 16
```

Volc：`bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_repr.yml`（或只含 t2 的 yml）。提交后记 `PROJECT_PROCESS.md`。

官方校准（CPU）：

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark
python scripts/prepare_tcr_clustering.py
python scripts/import_clustering_baselines.py
```

### 8.3 收数

基础 B：`outputs/tcr_clustering_embed/` 下该 tag 的 ARI/NMI/Purity mean-over-K。
基础 A：按 Fig 3A 逐方法锚点取点（例如 GLIPH2 0.22 / TCRdist3 0.44），同时报告实际 retention 与差值；`comparison_allowed=false` 不进同 retention 对照。确认产物声明 universe=4779，阈值可以从 CSV/JSON 精确重建簇。

### 8.4 填表

1. 官方 9 方法保持 `[P]`（Fig 3A）。Ours 只进 §0.2，表末。
2. 新 embed 曲线若仍在 5368 上，标「不可引用」。
3. changelog → `PROGRESS.md`。

### 8.5 同步哪些文档

`downstream-doc-sync.mdc` 最小集合：`RESULTS.md` §0.2 + `PROGRESS.md` + 本文件 + `PROJECT_PROCESS.md`。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-13 | T2 输入/关系审计与原生 collator CPU 检查，详见 §7.1；登记 v5 补全等待做项，未改实现/重跑指标。 |
| 2026-09-02 | 初版。锁定 universe 4,779；旧 5,368 曲线不可引用。 |
