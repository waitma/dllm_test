# TCR beta-only 双链补全与 relation diffusion 训练计划

日期：2026-09-12
状态：需求 1–6 + wandb/null 前缀已完成并在 v4 全量数据上验证；
§2.5（表位源补全）与 §2.6（all-X 占位符）**已在全量语料落地并核验**，
产物为 `data/prepared/immune_v5_receptor_completion/`（13G）。
§3.1 item 13 计数**代码已落地**；数字是对已发布 JSONL 的 post-hoc 推导，见 §4.2
（已发布 v5 `filter_report` 未重写）。
v5 逐源行数 / 与 v4 对照 / 监督 token 份额 / 不变量 / item 13 计数见 §4.2。
训练提交仍未执行（等平台配额/队列确认）。

操作入口：[`dllm/pipelines/immune_llada/README.md`](../dllm/pipelines/immune_llada/README.md)。
布局字段：[`DATA_FORMAT_AUDIT.md`](../DATA_FORMAT_AUDIT.md)。
进展账本：[`PROJECT_PROCESS.md`](../PROJECT_PROCESS.md)。

## 0. 进度快照

| 步骤 | 状态 |
|---|---|
| 链顺序统一为 beta-first（渲染层生效） | 完成 |
| `ARCH_AUDIT.md` 骨架 token / relation 描述修正 | 完成 |
| anchor 判定改为 provenance 驱动 | 完成 |
| config 切到 `tcr_repertoire_junc80` | 完成 |
| smoke 验证（含真实 `nonbinding`） | 完成，零不变量违规 |
| focused pytest / freshness / alphabet | 完成 |
| 全量 preprocessing v4（7,768,293 records，12G） | 完成，validation status `passed`（§4.1） |
| 全量审计：零不变量违规 | 完成 |
| 文档同步（链顺序 / 骨架 token / Gap 1） | 完成 |
| 无条件布局加固定 `<null>` context 前缀 | 完成（renderer 改动，**不需重跑预处理**） |
| wandb 切 `online`（带不可达时自动回退 offline） | 完成 |
| §2.5 表位源 beta-only 补全 | **已在全量语料落地并核验**（产物 `data/prepared/immune_v5_receptor_completion/`，§4.2） |
| §2.6 all-X epitope 删除 / all-X MHC 降级 | **已在全量语料落地并核验**（与 §2.5 同一次 v5 重跑，§4.2） |
| §3.1 item 13 计数（`downgraded_all_x_mhc`、各源 `beta_only_completed` / `alpha_only_completed`） | **代码已落地**（post-hoc 数字见 §4.2；已发布 v5 报告未重写） |
| 训练提交 | **未执行**（v5 已发布；等平台配额/队列确认） |

## 1. 当前真实训练数据源

本轮基于当前工作区的 `immune_llada` 热路径，不直接接入尚未完成 runtime gate 的 `immune_receptor_v2`：

```text
raw CSV/JSONL
  -> dllm/pipelines/immune_llada/data/sources.py
  -> BioSeqRecord
  -> scripts/data/preprocess_immune_dataset.py
  -> prepared semantic JSONL
  -> GrammarRenderer / GrammarBioSeqCollator
  -> examples/llada/protein_pretrain_esmc.py
  -> LLaDAEsmcFusion diffusion
```

当前 active mix 为：`oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`。
其中 `tcr_repertoire` 指向 `data/tcr_repertoire_junc80/dataset`（`schema_version=tcr_repertoire.junc80.v1`，`fv_source=tcrdesign2026_cdr3_junc80`），该构建保留**完整 IMGT junction**（`C...F/W`）。

> 注意：旧的 `data/tcr_repertoire/dataset` 存的是**剥掉锚点的 core**，两者 `cdr3b` 语义不同，不可互换使用。adapter 通过 provenance 字段（而非首尾字符）判定属于哪一种。

其余 TCR 数据按当前 adapter 和过滤器处理。`immune_receptor_v2` 当前文档状态为 `runtime_views_ready=false` / `training_ready=false`，本轮不把它假设成训练输入。

各源 raw 行数（train / valid）：

| source | train | valid |
|---|---:|---:|
| `oas` | 2,486,442 | 12,553 |
| `ots` | 2,102,715 | 10,619 |
| `asd_antibody` | 850,134 | 47,230 |
| `trait` | 69,251 | 3,050 |
| `tcr_native` | 143,391 | 4,390 |
| `tcr_papers` | 682,383 | 7,336 |
| `tcr_repertoire` (junc80) | 2,000,000 | 22,121 |
| 合计 | 8,334,316 | 107,299 |

这是**过滤前**的 raw 行数；各源保留率差别很大（decontamination blocklist），最终数量以新 manifest 为准，不能引用旧 v3 的表。

## 2. 目标行为

### 2.1 beta-only 数据统一成双链

只针对 `tcr_repertoire` beta-only 行：

- 生成统一的双链记录，**beta 在前、alpha 在后**；理由不只是与 `_ots` / `_oas`（heavy 先）保持一致：`sample_chain_conditioned_timesteps` 与 `generated_heavy_light_masks` 都把**最小 chain id 当作 heavy 类比链**，alpha-first 会让模型把 alpha 误当 heavy 链；
- 该顺序由 grammar renderer 的 receptor 组装决定（`grammar.py` 中 `receptor = [beta, alpha]`）。`record.chains` 的顺序**不会**进入 token 流，renderer 按角色重新取链，因此只改 adapter 是无效的；
- beta 的 CDR3 保留原始真实序列；若输入是 `C...F/W` junction，则 FR3 末端的 `C` 和 FR4 起始的 `F/W` 按 IMGT 语义放回 framework 区域；
- alpha 的 FR/CDR 区域和 beta 的非 CDR3 区域按照冻结的 region-length profile 采样；
- 所有补全区域使用 `X`；
- 记录 synthetic/completed region metadata，不能通过字符 `X` 猜测监督范围；
- decontamination key 继续由原始 CDR3β core 计算，不受随机补全影响；
- 相同 seed 和 profile 必须可复现。

### 2.2 synthetic X 不参与 loss

补全的 `X` 仍保留在模型输入中作为结构占位，但：

- `residue_mask=1`；
- `diffusion_loss_mask=0`；
- `diffusion_eligible_mask=0`；
- 不被 diffusion corruption；
- 不进入任何 reconstruction loss 或 token weighting。

真实的 CDR3β 继续作为 generated target。

### 2.3 relation token 纳入 diffusion

对明确标注为 `binding` / `nonbinding` 的 recognition relation：

- relation token 不再作为固定 context；
- 可被 diffusion mask；
- 进入 `diffusion_loss_mask` 和 `diffusion_eligible_mask`；
- clean relation token 作为 label；
- 推理时允许恢复 relation token。

无明确 relation supervision 的 relation token 继续固定可见，不产生 relation loss。pMHC 中 MHC→peptide 的 presentation `<binding>` 固定，只有 peptide→TCR 的 recognition relation 才是 target。

### 2.3.1 generated block 的骨架 token 也进训练

`<prots>`、类型标记（`<tcr>`/`<ab>`/`<nb>`）、`<protd>` 与 relation target 一并加噪并计入 loss。机制：renderer 输出显式 `diffusion_eligible_mask = not fixed and not synthetic`，`sample_bioseq_diffusion_noise` **信任该 mask**，不再与 `residue_mask` 取交（corruption 调用传 `require_residue=False`），因此非 residue 的骨架 token 同样被 corrupt 并进 loss。已用强制 `t=1` 实测确认，无需改动代码。

### 2.3.2 无条件布局加固定 `<null>` context 前缀

2026-09-11 追加需求：让 antibody pair / tcr pair 这类**无条件**布局在前面也跟一段固定块，与有
condition 的布局形状对齐。已批准的前缀形式：

```text
<prots> <null> <protd> <unknown>   <prots> <ab> *[heavy] <chainsep> *[light] <protd>
```

- 覆盖三个无条件分支：`antibody_pair` / `tcr_pair` / `nanobody`，即 **6.58M / 7.67M ≈ 86% 语料**；
- 前缀四个 token 全部 `is_fixed=True`：`loss=0`、`eligible=0`、`relation_target=0`，只作结构占位；
- 真正的收益是 **CFG 式条件强度控制** —— null context 与真实 context 占据同一结构槽位，
  推理时可以跑两支做外推。

三个必须遵守的实现约束（都踩过或推演过）：

1. **`<null>` 追加到 `GRAMMAR_STRUCTURE_TOKENS` 末尾（id 49），不插入。** token id 是
   `base_vocab_size + enumerate(GRAMMAR_TOKENS)`，结构 token 占 33–38、relation 占 39–48。
   插进结构 tuple 会让 `<binding>` 39→40，全部 relation 移位，破坏 v3 checkpoint 兼容性和
   下游硬编码 id。实测确认 `<binding>` 仍是 39。
2. **null 前缀块不消耗 chain index。** `<protd>` 原本会 `chain_index += 1`；若不特殊处理，
   抗体两链的 `position_ids_chain` 会从 `[0,1]` 变 `[1,2]`。而下游多处**按索引硬编码**寻址
   （`downstream/grammar/tcr_generation.py:178` 传 `0`、`:339` 传 `{0,1}`、
   `downstream/benchmark/nbbench/run_generative.py:108` 取 chain0），偏移会**静默指向错链**：
   不报错，但评测结果变垃圾。选择让前缀不占 index，而不是改所有调用方。实测残基链索引仍是 `[0,1]`。
3. **不能用空 block 代替。** `append_protein_block` 有
   `raise ValueError("Protein block requires at least one chain")`，且空跨度是别处不产生的畸形产生式。

连带修了 `downstream/grammar/masks.py` 的 token-scan fallback：它原本锁定**第一个** `<prots>`、
遇第一个 `<protd>` 就 `break`，加前缀后会撞上空前缀块返回零位置。已改为跳过 null 前缀块，
并实测两条寻址路径（`position_ids_chain` / token-scan）结果一致。

> 诚实标注：这是**未做 ablation 的建模赌注**，且会让 parity 基线重新定基。
> 条件布局（`antigen_antibody` / `tcr_pmhc` / `tcr_peptide`）完全不受影响，已实测确认。

**关键前提：prepared JSONL 是语义层的，不含 `input_ids`**（顶层 key 为
`chains`/`chain_roles`/`regions`/`metadata`/…），渲染在训练时才发生。所以加前缀是纯 renderer
改动，那 12G 预处理产物**不需要重跑**。

### 2.4 只保留一个正式 diffusion 版本

本轮唯一正式训练版本：

```text
diffusion
+ generated-only receptor residues
+ relation target token diffusion
+ beta-only -> alpha/beta dual-chain completion
+ synthetic X excluded from loss
+ relation_aux=none
+ diffusion_all_chains=false
```

BERT、all-chain、chain-ratio、cognate auxiliary 等旧配置保留作历史/对照，不作为本轮提交目标。

### 2.5 表位源里的 beta-only 也要补全（已在全量语料落地并核验）

**问题**：上一轮只给 `tcr_repertoire` 做了 beta-only → 双链补全。带表位的三个源
（`tcr_papers` / `tcr_native` / `trait`）**完全没走补全**，beta-only 行直接渲染成
单链受体块（`tcr_papers` 第一条 chain_roles 就是 `['peptide','tcr_beta']`，无
`beta_only_completion` 标记、无 `X`）。

prepared train 实测占比（**v4 动机快照**，不是 v5 行数；v5 核验见 §4.2）：

| 源 | 总数 | beta-only | 占比 | alpha-only |
|---|---:|---:|---:|---:|
| `tcr_papers` | 681,444 | 516,162 | **75.7%** | 0 |
| `tcr_native` | 96,552 | 50,021 | **51.8%** | 4,467 |
| `trait` | 31,515 | 5,155 | 16.4% | 4,418 |
| 合计 | 809,511 | **571,338** | 70.6% | 8,885 |

valid：`tcr_papers` 5,448/7,336（74.3%）、`trait` 386/1,393（27.7%）、`tcr_native` 237/3,855（6.1%）。

**这不只是布局一致性，是真实歧义 bug**：`GrammarTokenizer` 没有 `<tcra>`/`<tcrb>`，
单链受体块靠**位置**编码链身份，而单链没有位置可言 —— 于是 8,885 条 alpha-only 和
571,338 条 beta-only 渲染成**完全无法区分**的 token 序列。我们以为补全 `tcr_repertoire`
就消灭了 `tcr_single` 布局，它从表位源后门又进来了。

**处置**：补全成 α/β 双链，与 `tcr_repertoire` 同一套 region-length profile 与 `X` 语义
（synthetic 不进 loss、不加噪）。

**实现（已落地，相对 §3.1 原计划的故意偏离）**：

- `complete_tcr_chain(...)` 泛化自 repertoire 逻辑；RNG seed 按源派生，每源独立可复现。
- `_receptor_chains(...)` 返回 **beta-first**。真实全长 Fv 原样保留、不发明 region
  标注；只对缺失/残缺链合成。
- **偏离 (a)：按实际 `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` 列内容分流，不按
  `sequence_scope` 标签。** 825,774 行粗粒度一致（0 分歧），但 351 条 `tcr_native`
  标 `sequence_scope='fv'` 却只有 `beta_fv`、没有 `alpha_fv`。按标签分流会把它们
  留成单链，重新引入本节要消灭的歧义。`tcr_papers` 100% `scope='cdr3'`（682,383 行），
  全长 Fv 只出现在 `tcr_native`。
- Junction vs core 由 provenance 决定：`trait` = IMGT junction；`tcr_native` /
  `tcr_papers` = 去锚 core。不用首尾字符猜。
- 补全过的表位行仍是**条件**布局，**不该**拿 §2.3.2 的 null 前缀。2026-09-12 审计
  2,240 条 prepared 行，0 缺口；`grammar.py` 未改。

v5 全量核验（双链、真实 Fv 守恒、1024 预算）见 §4.2。
§4.2 的 post-hoc `alpha_only_completed` train 合计再次得到 8,885
（trait 4,418 + `tcr_native` 4,467），与本表在补全落地前用另一套方法测到的数字一致。

### 2.6 all-X 占位符记录的处置（已在全量语料落地并核验）

**根因是我们的 ingest bug，不是上游数据损坏。** 数据来源已核实为真实 release：
Zenodo record 14545852（DOI `10.5281/zenodo.14545852`，XSLiuLab/ShanghaiTech，CC-BY-4.0，
`tcrdesign_data.tar.gz` 501.2MB，Created 2024-12-24，与盘上文件时间戳一致），
repo `github.com/XSLiuLab/TcrDesign`。`X` 是上游**官方约定**：其 README 明写缺失值用
`'X'` 占位，CLI 示例本身带 `-alphaj X`。

原始 `data/tcr_papers/raw/tcrdesign2026/tcrdesign_B/pMHC_TCR_train.tsv`（74MB，无表头）
全量扫描 743,147 行：`epitope` 全 X 78,839 条（长度一律 21）、`mhc` 全 X 82,014 条
（长度一律 34）、两者同时全 X **0** 条。

bug 在 `scripts/data/tcr_native/ingest_papers.py::ingest_td26_pmhc`：`cdr3b`/`cdr3a` 都过了
`_drop_placeholder`，`epitope` 和 `mhc_pseudo` **漏用**。两道校验都拦不住，因为 **`X` 在我们
字母表里是合法残基**：`is_valid_protein_sequence('X'*21)` 为 `True`，`ok_epitope` 的长度区间
`[6,25]` 恰好放过 21，`len(pseudo) != 34` 对 `'X'*34` 也正好通过。信号其实被记录过但没人读：
`pseudo2allele` 对 `X*34` 查不到，`stats["pseudo_allele_unmapped"]` 累计到 15,932。

**连带后果**：`_tier(...)` 里 `has_mhc` 写死 `True`，那 15,932 条无 MHC 行被标成 tier B /
`task_type=tcr_pmhc`，tier 标签和 `by_task_type` 计数都错。

final CSV / prepared 实测：

| 层 | all-X epitope | all-X MHC |
|---|---:|---:|
| raw TSV | 78,839 | 82,014 |
| final CSV 全 splits | 39,521 | 15,932 |
| prepared train | 38,689 | 15,583 |

raw → final 的减半来自 dedup + 负样本 cap，**没有任何一步是占位符过滤**。all-X 只出现在
`tcrdesign26_pmhc` 一个子源（`trait` / `tcr_native` 均为 none）。

**处置**：

| 项 | prepared train | 处置 | 理由 |
|---|---:|---|---|
| all-X epitope | 38,689 | **删** | 没有表位可条件化；relation 现在是可训 target，留着等于教模型「epitope 全 X → nonbinding」这条捷径 |
| all-X MHC | 15,583 | 改渲染成 `tcr_peptide`（不 emit MHC 链），**不反推 MHC** | 见下 |

**为什么不反推 MHC**：`mhc_all.dat` 有 18,639 条 `allele→34aa pseudo` 映射且**零 all-X 条目**
（反证 `X*34` 不是某个 allele）。在 582,294 条两栏都真实的行上测条件熵：

```text
H(pseudo)           = 2.553 bits
H(pseudo | epitope) = 0.030 bits
mutual information  = 2.523 bits   -> epitope 解释 98.8% 的 MHC 熵
1,673 epitopes / 96 distinct pseudos；1,625 个 epitope 只对应 1 个 pseudo
mhcX 反推：69,755 唯一映射（85.1%）+ 12,259 歧义（14.9%，avg 6.59 候选）
唯一映射的 support：<=10 的一条都没有（11-100: 5,211 行，>100: 64,544 行）
```

所以「统计共现不可靠」这个理由是**错的** —— 反推整体准确率约 99.7%，且不存在
low-support 假象。真正的理由是反过来读同一组数字：**给定 epitope 后 MHC 只剩 0.030 bits**，
反推的信息增益近乎零。为 0.03 bits 制造一批无出处的标注不划算。

all-X epitope 则**信息论上恢复不了**：那 78,839 行只覆盖 70 个 distinct pseudo，而 epitope
空间是几千量级。

**顺带发现（不阻塞训练，但影响评测解读）**：96 个 pseudo 覆盖 1,673 个 epitope、近乎 1:1，
这是语料**组装方式**的产物（每个 epitope 基本只在一个 allele 语境下被收集），不是生物学
（真实情况一个 epitope 可被多个 HLA 呈递）。推论：`tcr_pmhc` 布局里的 MHC 块教给模型的
比我们以为的少，模型可以走捷径从 peptide 推出 MHC，不必学 MHC 限制性。**该测量仅针对
`tcrdesign26_pmhc` 子源**，`tcr_native`（PISTE）未做同样测量。读 pMHC 条件生成的评测数字时
要记住这一点。

**净代价**：train 负样本减少 24,623 条（占 `tcr_papers` 负样本 11.1%）。mhcX 里的 12,693 条
负样本因为改成降级渲染而**被救回**，所以比早先估的 16.8% 低。

**修复层级决策**：在 **adapter / offline-filter 层修**，ingest 根因修复仍记为「下次重建语料时做」。
理由不变：根因在 ingest（`_drop_placeholder` 漏用），但在那儿修要重建
`data/tcr_papers_v2/dataset/`，连带重跑 dedup + 去污染；v4 YAML 要求该目录对 v2
checkpoint 保持 byte-identical，只能写新 root。

**偏离 (b)：all-X epitope 走命名 filter `quality.blank_epitope`，不是 adapter 返回 `None`。**
这样 drop 会计入 `filter_report.json` 并可归因。all-X MHC 仍剥离链并降为
`tcr_peptide`，`task_type` 由 MHC 链是否幸存推导，不再写死。

v5 实删 `quality.blank_epitope` **38,689 train + 411 valid**（全部 `tcr_papers`），
剩余 all-X peptide / MHC 链均为 0；见 §4.2。

§3.1 item 13 的 `downgraded_all_x_mhc` 与各源 `beta_only_completed` /
`alpha_only_completed` **已落地**。权威数字（含「朴素 JSONL 启发式会把 MHC
降级数错记约 10 倍」的陷阱）见 §4.2；已发布 v5 `filter_report` 尚未写入这些键。
§4.2 的 live-definition train 降级数再次得到 15,583，与上表事先记下的独立数字一致。

## 3. 实施步骤

1. 增加可冻结、可审计的 region-length profile 配置/生成方式，并在 preprocessing manifest/report 中记录 profile provenance。
2. 扩展 `BioSeqChain` / `BioSeqRecord` 的 metadata 传递，使 synthetic region mask 能从 prepared JSONL 传到 renderer/collator。
3. 修改 `tcr_repertoire` adapter，只对 beta-only 行生成 alpha/beta 双链，并保留原始 CDR3β key。
4. 修改 grammar renderer，输出 synthetic residue mask 和 relation target mask；保证旧源默认行为不变。
5. 修改 diffusion noise / inference partial-mask 逻辑，让 relation target 可被加噪和恢复，但 synthetic X 不进入 target。
6. 新增唯一 v4 diffusion YAML，明确 `train_objective=diffusion`、`diffusion_all_chains=false`、`relation_aux=none`。
7. 增加 adapter、profile、mask、relation 和配置测试。
8. 运行 focused pytest、diagnostics，并记录实际命令、退出码和产物。
9. 更新数据管线与训练进度文档，明确当前源、profile、mask 和单一 diffusion 口径。

### 3.1 §2.5 + §2.6 执行步骤

全部集中在 `dllm/pipelines/immune_llada/data/`（未纳入 git，不要用 git 术语讲这段历史）：

10. **泛化补全**（已做）：`complete_tcr_chain` 按源入参；RNG seed 按源派生；
    `tcr_repertoire` 对 v4 冻结 profile **byte-identical**（3,000 行 0 mismatch）。
11. **按实际 fv/CDR3 列分流**（已做；**偏离原「按 `sequence_scope`」计划**，见 §2.5）。
12. **all-X 处置**（已做；**偏离原「adapter `None`」计划**，见 §2.6）：
    epitope → `quality.blank_epitope`；MHC → 剥链降为 `tcr_peptide`。
13. **计数落到 `filter_report`**（**已做**）：每个 source 与 `totals` 写入三个
    kept-record transformation 计数 —— `downgraded_all_x_mhc`、
    `beta_only_completed`、`alpha_only_completed`。与六个 first-failure drop
    计数分开，`count_immune_drops.py` 不受影响。`dropped_all_x_epitope` 不重复，
    仍只走 `quality.blank_epitope`。已发布 v5 报告是重跑前写的，没有这些键；
    post-hoc 数字见 §4.2。下次 preprocess 会原生写入。
14. **补测试**（已做）：`scripts/tests/immune_llada/test_receptor_completion.py`、
    `test_null_context_prefix.py`。套件 **198 passed / 5 既有无关失败**
    （`test_full_parity` ×3、`test_profiling` autocast ×2）。
15. 重跑全量预处理（**已完成并审计**）：新 root
    `data/prepared/immune_v5_receptor_completion/`，不覆盖
    `immune_v4_beta_relation`。数字见 §4.2。

## 4. 验证清单

- beta-only 行输出恰好 alpha/beta 两条链；
- 原始 CDR3β 完整保留；补全长度满足 profile；
- synthetic X 的 loss/eligible mask 为 0；真实 CDR3β mask 为 1；
- binding/nonbinding relation token 可被 diffusion，且 clean label 正确；
- 未标注 relation 仍固定；
- OAS/OTS/native/papers 旧行为 parity；
- prepared JSONL round-trip 不丢 metadata；
- 新 YAML 只启用 diffusion；
- 当前 blocklist freshness、residue alphabet 和数据计数审计仍通过。

§2.5 / §2.6 额外验证项：

- prepared 里 **`tcr_single` 布局归零**（三个表位源 + `tcr_repertoire` 全部双链）；
- 表位源补全行的真实 CDR3 逐残基守恒，synthetic 段 loss/eligible mask 全 0；
- 表位源补全行**没有**拿到 null 前缀（仍是条件布局）；
- all-X epitope 行在 prepared 里计数为 0；
- all-X MHC 行的 `grammar_name` 为 `tcr_peptide` 且 chain_roles 不含 `mhc`；
- 同一 seed 下 `tcr_repertoire` 输出与旧产物 byte-级一致（确认 §2.5 没误伤已验证路径）。

## 4.1 全量实测结果（2026-09-11）

> 这一节的数字是 **v4-only** 快照（prepared root `immune_v4_beta_relation`）。
> 不要覆盖。v5 快照与 v4 对照见 §4.2。

focused pytest：**307 passed, 1 failed**（含 null 前缀改动后的重跑；`test_null_context_prefix.py` 7 passed，
parity 子集单独重跑 11 passed）。唯一失败 `test_gidd_ophiuchus_edit.py::test_ophiuchus_ratio_buckets_and_val_independent`
是**既有失败**，与本轮无关（该文件对 grammar renderer 零引用，`chain_ids` 全手工构造）。

> 跑测试必须 `--ignore` 掉 `scripts/tests/attention` 与 4 个依赖 `lm_eval` 的文件，
> 否则会出现大批**由缺包引起的连带失败**，不是回归。

预处理产物：`data/prepared/immune_v4_beta_relation/`，profile `source_policy=current_complete_tcr_rows`，
源 `['ots','tcr_native','tcr_papers']`，digest `3149948651967f952f32`，kept 2,894,002 / raw 2,950,834，**非 fallback**。

| source | train | valid |
|---|---:|---:|
| oas | 2,485,433 | 12,551 |
| ots | 2,094,218 | 10,597 |
| asd_antibody | 276,412 | 44,866 |
| trait | 31,515 | 1,393 |
| tcr_native | 96,552 | 3,855 |
| tcr_papers | 681,444 | 7,336 |
| tcr_repertoire | 2,000,000 | 22,121 |
| **TOTAL** | **7,665,574** | **102,719** |

raw 8,441,615 → kept 7,768,293。约 780 万是**过滤后**的量级。

审计结论：

- **negative 路径已在真实全量数据跑通**（此前 smoke 全是 binding 只因取前 N 行的采样偏差）：
  train nonbinding —— tcr_papers 222,757 / asd_antibody 77,820 / tcr_native 43,735 / trait 19,828。
  注意 `valid/tcr_native` 仅 7 条 nonbinding。
- **链顺序 beta-first 生效**：`ots` 与 `tcr_repertoire` 全部 `tcr_beta|tcr_alpha`。
- `tcr_repertoire` loss token ≈ 8%，synthetic ≈ 92%。
- **anchor 守恒**（400 条真实 junc80 行）：junction 逐残基守恒 400/400，`C` 落 FR3 末位、
  `F/W` 落 FR4 首位，alpha 全 synthetic `X`，零 mismatch。
- `assert_residue_alphabet.py` train + valid 均 OK；`assert_corpus_fresh.py` junc80 PASS
  （mode `exact_core+hamming_junc80_hobohm1`）。

`test_full_parity` 的 mismatch 是**已批准的偏离**，不是 bug：冻结基线里没有 beta-only 双链补全，
审计如实报告差异。过滤决策完全一致（`legacy_kept == canonical_kept == 3`，`count_mismatches == 0`）。
已按同文件既有先例更新断言（期望 exit 2），**未削弱审计**。

## 4.2 全量实测结果（2026-09-12，v5）

> 这一节是 **v5** 快照，也是 v5 行数 / v5-vs-v4 对照 / 监督 token 份额 /
> 渲染 token 上限 / 不变量的**唯一权威**。§4.1 仍是 v4-only，不要覆盖。

产物：`data/prepared/immune_v5_receptor_completion/`（**13G**），
config `configs/data/immune_v5_receptor_completion.yaml`，
`tcr_region_profile.completion_sources: [trait, tcr_native, tcr_papers, tcr_repertoire]`。
manifest `schema_version: immune_llada.semantic.v1`，
region-profile digest
`be50d06bb612b6559625b66ce812df9163675351748c1b699cc2b920dd168353`，
profile 学习源 / policy 未变（`ots`, `tcr_native`, `tcr_papers` /
`current_complete_tcr_rows`）。

digest 相对 v4 的 `31499486…` **只**因为 `observations.dropped_filters`
从 56,832 升到 95,932（多出来的是 blank-epitope drops）。
`region_distributions` 与 v4 **逐项相同**，`tcr_repertoire` train/valid
shard 与 v4 **byte-identical**。digest 变化是记账，不是分布漂移。

| source | split | v4 | v5 | Δ |
|---|---|---:|---:|---:|
| oas | train | 2,485,433 | 2,485,433 | 0 |
| ots | train | 2,094,218 | 2,094,218 | 0 |
| asd_antibody | train | 276,412 | 276,412 | 0 |
| trait | train | 31,515 | 31,515 | 0 |
| tcr_native | train | 96,552 | 96,552 | 0 |
| tcr_papers | train | 681,444 | 642,755 | −38,689 |
| tcr_repertoire | train | 2,000,000 | 2,000,000 | 0 |
| oas | valid | 12,551 | 12,551 | 0 |
| ots | valid | 10,597 | 10,597 | 0 |
| asd_antibody | valid | 44,866 | 44,866 | 0 |
| trait | valid | 1,393 | 1,393 | 0 |
| tcr_native | valid | 3,855 | 3,855 | 0 |
| tcr_papers | valid | 7,336 | 6,925 | −411 |
| tcr_repertoire | valid | 22,121 | 22,121 | 0 |

v5 train **7,626,885**。valid 七源之和 **102,308**（v4 valid 102,719 − 411）。
唯一 drop 是 `quality.blank_epitope`：**38,689 train + 411 valid**，全部在
`tcr_papers`。`trait` / `tcr_native` 的 raw CSV 没有全 X 表位，行数不变。

v5 raw→kept 总量**未另给**，不要用 v4 的 8,441,615 → 7,768,293 减一下充数。

核验不变量：

- 剩余 all-`X` peptide 链 0；all-`X` MHC 链 0。
- 每条 `trait` / `tcr_native` / `tcr_papers` 恰好 2 条受体链，**`tcr_beta` 在 `tcr_alpha` 前**。
- 0 条未剥皮的 `trait` junction。朴素「FR3 以 C 结尾 AND CDR3 以 C 开头」会标出
  1 train + 1 valid，但两条都是正确剥皮的 `CC…F` junction，剩下的 core 本来就以 C
  开头（`CSGGSNYKLT`、`CTHNAGGTSYGKLT`）——不是剥皮 bug。首尾字符启发式正好踩进这个陷阱。
- `tcr_native` 真实全长 Fv 保留：train 75,865 条 real-Fv 链（valid 5,926），
  相对 raw `alpha_fv`/`beta_fv` 0 missing、0 发明 region。
- 0 条超过 1024-token grammar 预算。

各源最大渲染 token：oas 305，ots 305，asd_antibody 1016，trait 323，
tcr_native 296，tcr_papers 306，tcr_repertoire 274。

监督 token 份额（train，`diffusion_loss_mask` 求和）：

| source | v5 records | rec % | v5 supervised tokens | v5 sup % | v4 sup %（重测） |
|---|---:|---:|---:|---:|---:|
| oas | 2,485,433 | 32.59% | 585,414,754 | 49.38% | 49.36% |
| ots | 2,094,218 | 27.46% | 482,547,783 | 40.71% | 40.69% |
| asd_antibody | 276,412 | 3.62% | 53,930,078 | 4.55% | 4.55% |
| trait | 31,515 | 0.41% | 911,861 | 0.08% | 0.08% |
| tcr_native | 96,552 | 1.27% | 9,874,487 | 0.83% | 0.83% |
| tcr_papers | 642,755 | 8.43% | 12,849,684 | 1.08% | 1.13% |
| tcr_repertoire | 2,000,000 | 26.22% | 39,926,266 | 3.37% | 3.37% |
| **total** | **7,626,885** | | **1,185,454,913** | | |

**若别处把 v4 监督份额写成 `asd_antibody 5.4%` / `tcr_native 0.13%`，那是错的**，
应以本表重测值 `4.55%` / `0.83%` 为准。根因：早先把 `tcr_native` 的监督 token
算进了 `asd_antibody`（4.55 + 0.83 = 5.38 ≈ 误记的 5.4，留下 0.13 当 native 残差）；
其余五源与重测相差不超过 0.17pp。

结论：补全几乎不移动监督预算 —— 合成 `X` 不进 loss，受体源多了记录和 raw token，
但几乎不加监督。

valid 监督份额、v5 DataLoader / GPU 吞吐、v5 六布局 `gen%` **未测**，不要估算。

### Item 13 计数（post-hoc；不在已发布 v5 `filter_report` 里）

> 本节是 `downgraded_all_x_mhc` / `beta_only_completed` / `alpha_only_completed`
> 的**唯一权威数字**。别处只许链接，不许复述表。

`filter_report.json` 现对每个 source 以及 `totals` 写入三个 **kept-record
transformation** 计数，与六个 first-failure drop 计数分开
（`count_immune_drops.py` 仍只比原来的六键）：

- `downgraded_all_x_mhc` — 一条保留行的 raw MHC 全为 `X`，因此被剥掉
- `beta_only_completed` — 一条保留行补了一条全合成 alpha（synthetic mask 全 1）
- `alpha_only_completed` — 一条保留行补了一条全合成 beta

`dropped_all_x_epitope` **故意不重复** —— 仍只记在 `quality.blank_epitope`
filter 归因里。

**这些键不在已发布的 v5 `filter_report.json` 里**（v5 13G 未重跑）。下表由已发布
JSONL **事后推导**。下次 preprocess 会原生写入。

Post-hoc v5 伴侣补全（全量 = train + valid）：

| source | records | `beta_only_completed` | `alpha_only_completed` | both observed | one real + one synthetic |
|---|---:|---:|---:|---:|---:|
| trait | 32,908 | 5,541 | 4,418 | 22,949 | 9,959 |
| tcr_native | 100,407 | 50,258 | 4,646 | 45,503 | 54,904 |
| tcr_papers | 649,680 | 511,791 | 0 | 137,889 | 511,791 |
| tcr_repertoire | 2,022,121 | 2,022,121 | 0 | 0 | 2,022,121 |

Train / valid 拆分：trait 5,155 / 386 beta-only 与 4,418 / 0 alpha-only；
tcr_native 50,021 / 237 与 4,467 / 179；tcr_papers 506,433 / 5,358 beta-only，
0 alpha-only；tcr_repertoire 2,000,000 / 22,121 beta-only。

自洽核验：每个 completed 计数 ≤ 该源 record 数；每个补全源
`both_observed + one_real_one_synth = records`；没有一条记录两条链都全合成；
`tcr_layout_other = 0`。

**与 §2.5 交叉验证（不要丢）**：train alpha-only = trait 4,418 + tcr_native
4,467 = **8,885**，与 §2.5 在补全落地前用另一套方法测到的 8,885 **完全一致**。
这是补全记账正确的最强旁证。

#### `downgraded_all_x_mhc`：朴素启发式会错约 10 倍

不要用「prepared JSONL 有 peptide 链、没有 `mhc` role」去重数这个计数。

- 该朴素启发式给出 **153,069**（trait 1,638 / tcr_native 182 / tcr_papers
  151,249）。**这是错的。**
- 它**高估**了。`tcr_papers` raw `mhc_seq` 有 15,777 条全 `X`，另有
  **135,840 条本来就是空的** —— 空行从未有过 allele，本来就是 `tcr_peptide`，
  不是「降级」。trait / tcr_native raw `mhc_seq` 含 **零** 条全 `X`，它们的
  peptide-without-MHC 行也从未降级。
- 按 live 定义（只对 raw 全 `X` 行重放 adapter + filters）才是对的：
  **15,762 = 15,583 train + 179 valid，全部在 `tcr_papers`；trait 0，
  tcr_native 0。**
- **与 §2.6 交叉验证**：15,583 train 与 §2.6 事先记下的独立数字完全一致。

以后不要再推导出 153,069 并把它当成 `downgraded_all_x_mhc`。

## 5. 风险与边界

- 如果当前原始数据没有足够完整 region 标注，profile 必须报告样本数和缺失情况，不能静默伪造统计来源。
- `tcr_repertoire.cdr3b` 可能是 core 或完整 junction；判定**必须依据 provenance 字段**（`fv_source` / `provenance` / `schema_version`），不能用首尾字符猜：anchor-free core 也可能天然以 `C` 开头或 `F/W` 结尾（如 `AAAATGAGEQF`），字符启发式会静默吃掉一个真实 residue。
- relation token 的 target 语义不能把默认兼容性 binding 当成真实监督；无标签记录必须区分。
- 不改正在运行的 2M 训练目录、resume 逻辑或旧 checkpoint。
- 不把 `immune_receptor_v2` 宣称为本轮已接入数据，除非另行关闭其 runtime/training gate。
- **`assert_corpus_fresh.py` 输出含一条 `UNVERIFIED dataset: builder records no blocklist provenance`**：
  非致命（脚本 exit 0），但该 corpus 的 staleness 无法校验。
- **beta-only 的算力代价 —— 已批准，不是待优化项**：`tcr_repertoire` 占记录数约 26%，
  其中约 92% token 是不算 loss、不加噪的 `X`，但仍吃满 attention。这是「统一双链布局」的
  必然成本。2026-09-11 已确认接受这个交换：**换来的是所有 TCR 源共享同一种 alpha/beta
  双链布局**，下游不需要为 beta-only 记录准备第二条渲染/解码路径，`tcr_single` 布局也不再
  是训练里从未出现过的分布。
  
  > 后续若有人想「优化掉」这些填充位（例如让 beta-only 退回单链、或把 `X` 段截短），
  > 那不是性能修复，而是**推翻本轮的布局统一决策**，需重新走设计评审。
  > 注意单链退路另有硬约束：`GrammarTokenizer` 没有 `<tcra>`/`<tcrb>`，
  > `len(receptor) == 1` 时 alpha 与 beta 渲染成完全相同的 token 序列（见
  > `examples/llada/PROTEIN_PRETRAIN_PROGRESS.md` §7.3）。
- **null 前缀是未做 ablation 的建模赌注**，影响 86% 语料，且会让 parity 基线重新定基。
  若日后要回退，注意 `<null>` 的 id 49 位于词表末尾，移除它不会移位 relation token。
- **wandb online 的 egress 未在计算节点验证**：本地探到 `api.wandb.ai` TCP:443 OK，但本 shell 有
  `SAND_REMOTE_PROXY_URL`，走的可能是沙箱出口，**不能证明计算节点能连**。因此 YAML 里加了
  一次性 socket 探测 + 自动回退 offline：online 下若节点无出口，wandb 可能重试甚至阻塞，
  比 offline 更糟。回退后日志仍在盘上，可事后 `wandb sync`。
- **无 metric 可观测 synthetic X 是否真被排除**：只在预处理审计里离线验过，训练时看不到。
  要可见性需改训练脚本加 metric，本轮未做。
- **`grammar.py` 的 `receptor = [beta, alpha]`（L494）才是链顺序的真正排序点**；
  `record.chains` 的顺序**不进入** token 流，renderer 按角色重新取链。只改 adapter 无效。
  beta-first 不只是一致性问题：`sample_chain_conditioned_timesteps` 和
  `generated_heavy_light_masks` 都把最小 chain id 当 heavy，alpha-first 会让模型把 alpha 误当 heavy。
- **§2.5 会把计算代价从 26% 语料扩到更大范围**：又有 571,338 条（叠加后约占总语料 7.4%）
  会长出一整条全 `X` 的 alpha 链。这是 §2.5 的已知、已接受成本，理由与上一条 beta-only
  条目相同（布局统一 > 算力）。
- **§2.6 的 ingest 根因未修（当前最重要的未修项）**：`ingest_papers.py` 里
  `_drop_placeholder` 仍漏用在 `epitope` / `mhc_pseudo` 两栏。adapter /
  `quality.blank_epitope` 拦住了当前 v5 语料，但**下次有人重建
  `tcr_papers_v2/dataset/` 时 all-X 会重新进来**。已在本节留档，不要当它已修。
  细节权威：[`examples/llada/DATA_PIPELINE_README.md`](../examples/llada/DATA_PIPELINE_README.md) §6.0.1。
- **manifest `filter_names`：代码已修，已发布 v5 产物未重写。** 旧缺陷比「漏写
  `quality.blank_epitope`」更大（硬编码 `BLOCKLIST_NAMES` 配置键，不是实际跑过的
  filter）。缺陷描述与新语义的权威登记：[`DATA_FORMAT_AUDIT.md`](../DATA_FORMAT_AUDIT.md)。
  不要把代码修复读成已发布 v5 manifest 已更新。
- **已发布 v5 `filter_report.json` / `dataset_manifest.json` 仍是旧字段**：没有
  item 13 计数，`filter_names` 仍是旧硬编码。只有下次 preprocess 才会写出新字段。
  数字权威：§4.2（post-hoc）。
- **`tcr_papers` 记录 identity 的 `source` 仍是 `"tcr_native"`**（既存 wart）：
  `metadata.dataset_source` 和 shard 名是对的 `tcr_papers`。不要标已修。
  布局侧登记：[`DATA_FORMAT_AUDIT.md`](../DATA_FORMAT_AUDIT.md)。
- **Circular-profile guard 已落地**：`add_tcr_region_lengths` 跳过
  `synthetic_regions`。profile 学习源含 `tcr_native` / `tcr_papers`，v5 把它们变成
  补全源；无此守卫会把合成长度学回去。
- **`sample_region_lengths` 必须按数值排序**：manifest `sort_keys=True` 会让
  `'10'` 排在 `'2'` 前。分布一直对，给定 seed 抽到的长度曾不可复现（只影响
  单双位数并存的区域，即 CDR3）。v4 现可从自身 manifest 再生。
- **`tcrdesign26_pmhc` 里 MHC 对 epitope 几乎冗余（条件熵 0.030 bits）**：不阻塞训练，
  但意味着 pMHC 条件生成的评测分数可能高估了模型对 **MHC 限制性**的掌握 —— 模型可以
  从 peptide 直接推出 MHC。该测量**仅针对这一个子源**，`tcr_native`（PISTE）未测。
