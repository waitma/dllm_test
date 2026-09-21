# 免疫受体原始语料：数据源、去污与审计

> 本文是 **raw corpus audit** 的权威文档：有哪些源、原始语料如何去污、三次
> 「过滤看起来生效了其实没有」的事故。不要在这里复述 prepared 布局或 v4/v5 行数。
> 正式离线入口 / 当前 prepared 版本：
> [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)。
> 布局字段：[`DATA_FORMAT_AUDIT.md`](../../DATA_FORMAT_AUDIT.md)。
> 补全/all-X 设计：[`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](../../docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md)。
> 训练任务账本：[`PROJECT_PROCESS.md`](../../PROJECT_PROCESS.md)。
> 最近更新：2026-09-12

这条管线上出现过三次同一类事故 —— **过滤看起来生效了，其实没有**。每次都不是过滤
逻辑写错，而是"过滤依据"和"被过滤的数据"之间没有强制关联。第 5 节写全三次的根因和
修法，新增数据源前请先读。

---

## 0. 当前 prepared 版本（指针，不复述布局）

| 版本 | 状态 | 入口 |
|---|---|---|
| v4 `data/prepared/immune_v4_beta_relation` | 已发布 | [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md) |
| v5 `data/prepared/immune_v5_receptor_completion` | **已发布**（行数：plan §4.2） | 同上 |
| v3 / `immune_v3_heterotypic` | 盘上仍在；部分 checkpoint 仍指向 | 验收史：[`docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md`](../../docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md) |

v4 逐源行数：plan §4.1。v5：plan §4.2。junc80 vs core 的 **raw 构建**约定仍见
[`scripts/data/tcr_native/TCR_REPERTOIRE_PROCESSING.md`](../../scripts/data/tcr_native/TCR_REPERTOIRE_PROCESSING.md)
与下文 §3。§6.0 的表位源补全 **已在 v5 全量核验**；§6.0.1 的 ingest 根因 **仍未修**。

---

## 1. 当前数据源与 loss 预算

```
--dataset_args oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire
```

下表是 **v4 之前** 直读 CSV 的 loss-budget 口径（`scripts/count_immune_mix.py`
2026-09-07）。**不要把它当成 v4/v5 prepared 行数，也不要把它当成监督 token 份额。**
v4 入训量见 plan §4.1；v5 见 plan §4.2。本节 `gen%` 未按 v4/v5 重算（v4/v5
`diffusion_loss_mask` 监督份额在 plan §4.2，口径不同）。记录占比 ≠ loss 预算
这一点仍然成立。

| 源 | 内容 | grammar 布局 | 记录数 | 记录% | **gen%** | 去污层 |
|---|---|---|---:|---:|---:|---|
| `oas` | 抗体配对重/轻链 | `antibody_pair` | 2,485,471 | 31.89% | **50.01%** | 加载期 |
| `ots` | TCR 配对全长 α/β | `tcr_pair` | 2,094,231 | 26.87% | **41.21%** | 加载期 |
| `tcr_repertoire` | **无表位 CDR3β**（v4 补全成双链） | ~~`tcr_single`~~ → `tcr_pair` 布局 | 2,128,750 | 27.31% | **2.42%** | 构建 **+** 加载 |
| `asd_antibody` | 抗原 → 抗体 | `antigen_antibody` | 276,412 | 3.55% | **4.57%** | 加载期 |
| `tcr_papers` | 论文表位层（`tcr_papers_v2`） | `tcr_pmhc` 80.3% / `tcr_peptide` 19.7% | 681,444 | 8.74% | **0.91%** | 构建 **+** 加载 |
| `tcr_native` | TCR 全长 + pMHC | `tcr_pmhc` 99.9% / `tcr_peptide` 0.1% | 96,552 | 1.24% | **0.82%** | 构建 **+** 加载 |
| `trait` | TCR CDR3 + 表位 | `tcr_pmhc` 95.2% / `tcr_peptide` 4.8% | 31,515 | 0.40% | **0.07%** | 加载期 |

**记录占比 ≠ loss 预算。** `oas` + `ots` 两源吃掉 **91%** 的残基预算；`tcr_repertoire`
占 27% 记录却只有 2.4%。根因是每条样本待预测残基的数量级差异：`tcr_repertoire` 约 13 个、
`oas` 约 232、`asd_antibody` 约 523，而 `tcr_pmhc`/`tcr_peptide` 只生成 CDR3（13–31 个）。
**表位条件生成三源合计只有 1.79% 的 loss 预算，靠加行数改不动** —— `tcr_papers` 翻 3 倍
也只抬到约 3%。要真正抬高必须做 per-sample loss 加权；当前 source weight 已随 prepared `BioSeqRecord` 落盘，但是否用于 loss 仍是独立建模决策（§6.3）。

`tcr_repertoire` 曾是唯一渲染成 `tcr_single` 的源（T4 Setting-A 无条件生成用该布局）。
v4 起它被补成双链；v5 起表位源单链行同样补全。§6.2 的 grammar 硬约束仍在：没有
`<tcra>`/`<tcrb>` 就不能安全摄入单链 α。

valid split 同口径 **203,647 条**，但构成与 train 完全不同（`asd_antibody` 占 22% 记录、
70% 残基），不可直接类比 —— 见 §6.4。

> ⚠️ **占比必须全量或蓄水池采样测，不能取前缀。** `tcr_papers_v2` 是 7 个论文语料首尾
> 拼接的，前 3 万行 100% 是 `tcr_peptide`，全量却是 80.3% `tcr_pmhc` —— 前缀采样曾把
> 547,274 条 `tcr_pmhc` 误记成 `tcr_peptide`（低估约 4 倍，2026-08-29 已修）。

> 🔴 **不要用 `build_report.json::split_counts` 查行数**，它是构建期快照，而 2026-08-29
> 的 valid 近重复搬迁（`move_near_dup_eval_rows.py`）只改 CSV 不重写报告。行数以 CSV
> 为准，搬迁记录见 `near_dup_eval_move_report.json`；**该报告仍是「去污染」口径的权威
> 来源**（`PASS` / `decontam_mode` / `blocklist_provenance`）。raw 行数
> （train/valid/holdout，2026-09-07 复核）：`tcr_papers_v2` = 682,383 / 7,336 / 7,201，
> `tcr_repertoire` = 2,128,750 / 123,049 / 122,669。

### 1.1 去污发生在哪一层，决定了它会不会过期

当前训练读的是 prepared JSONL，不再在 DataLoader 里套 `with_exclusion_filter`。
下面说的「加载期」是 **raw 语料 / 旧 CSV loader** 的一层；offline preprocess 把同一套
规则写进 prepared。黑名单事后重建**不会**自动回溯已写出的 CSV 或 prepared shard，
必须靠 `assert_corpus_fresh.py` 发现、再重跑构建或 preprocess。

三个 unified-schema 的源两层都有 —— 建库时按全套黑名单去污，加载期又在
`build_immune_specs` 里叠一层：

| 源 | 加载期叠的过滤 |
|---|---|
| `tcr_native` / `tcr_papers` | `_tcr_native_row_to_record`：`t4_refbinder` + `t2t3_eval`（配对键） |
| `tcr_repertoire` | `repertoire_core_exclusions` = `t4` ∪ `t2t3` **投影成裸 core** ∪ `ots_benchmark` |

`tcr_repertoire` 无表位，配对键永远匹配不上，所以必须投影成裸 core。这是**故意过挡**：
一条 benchmark CDR3β 不论当初是哪个表位的答案都会被丢掉。补上这层后实测三条 T4 参考
binder（`ASSFGGRSYEQY` / `ASSFLAGQETQY` / `SAPTDTQY`）全部挡住。

**双层不是冗余，它是唯一能兜住"黑名单重建了但语料没跟着重建"的机制** —— 断言只能
*发现*脱节，加载期过滤才能*挡住*后果。

> 新增数据源时先决定放哪一层。构建期更省启动开销，但欠一份 provenance 就等于欠一个
> 静默失效的坑；能加一层加载期过滤就加上。

---

## 2. 黑名单清单

全部在 `data/tcr_native/dataset/` 下。键数为 2026-08-28 实测，启动时入口会打印实际
加载的键数，和下表对不上就说明黑名单被改过。

| 文件 | 键数 | 键空间 | 保护对象 |
|---|---:|---|---|
| `trait_benchmark_blocklist.txt` | 59,212 | 裸 core | NM2025 seen/unseen + public（hard）+ full_bank |
| `t4_refbinder_blocklist.txt` | 68,846 | `(core\|epitope)` | T4 表位条件生成的答案键 |
| `t2t3_eval_blocklist.txt` | 8,377 | `(core\|epitope)` | T2 聚类 / T3 表征评测集 |
| `ots_benchmark_blocklist.txt` | 4,494 | 裸 core | TCR binding 基准（OTS 侧） |
| `replaces_trait_blocklist.txt` | 11,843 | `(core\|epitope)` | 非去污：native 全长取代的 TRAIT 行 |
| `asd_antibody_benchmark_blocklist.txt` | 521,073 | CDR-H3 / 重+轻 | 抗体基准（Kong 等） |
| `asd_nanobody_benchmark_blocklist.txt` | 2,086 | CDR-H3 | 纳米抗体基准 |
| `oas_benchmark_blocklist.txt` | 661 | CDR-H3 / 重+轻 | 抗体基准（OAS 侧） |

```
Blocklist key counts: replaces_trait=11843 trait_benchmark=59212 ots=4494 oas=661
  asd_ab=521073 asd_nb=2086 t4_refbinder=68846 t2t3_eval=8377
```

---

## 3. 两个键空间，别混用

读任何去污结果时最容易出错的地方。

- **裸 core 键**：保护集就是 CDR3β core 本身，命中一条就是泄漏答案，零容忍。只有
  `NM2025_seen` / `NM2025_unseen` / `public_trackA` 属于这类，即
  `decontam.py::load_benchmark_sets()` 返回的 `binding_benchmark`（docstring 写明是
  "the HARD-requirement gate"）。
- **`(core|epitope)` 配对键**：保护"某条 CDR3β 配某个表位"这个组合。同一条 CDR3β 配
  **不同**表位是合法训练信号，不是泄漏。`t4_refbinder` / `t2t3_eval` /
  `replaces_trait` 都是这类。

后果：把配对键投影成裸 core 去比对会**按构造过报**。实测 `tcr_papers_v2` 约 7.7 万、
`tcr_native` 约 7.4 万裸 core 命中 T4 黑名单，但 `(core|epitope)` 配对层命中是 **0**
（`t4_refbinder_blocklist.txt` 的 68,846 个键 100% 含 `|`）。审计脚本因此区分
"必须为 0 的 hard-requirement 行"和"非零也合理的信息行"，**不要看总和**。

另外 **anchor 约定不统一**：CDR3 可能存成完整 IMGT junction（`C..[FW]`）或去锚的 loop，
直接比会静默漏掉全部命中。各源约定声明在 `audit_downstream_leakage.py::_SOURCES`：

| 源 | 列 | `has_anchors` |
|---|---|---|
| `ots` | 按 `anarci_type` 取 β | False（loop） |
| `trait` | `cdr3b` | **True**（完整 junction） |
| `tcr_native` / `tcr_papers` / `tcr_repertoire` | `cdr3b` | False（统一 schema 存 core） |

`tcr_repertoire` 的抽样 / Hamming / junction 处理见
[`scripts/data/tcr_native/TCR_REPERTOIRE_PROCESSING.md`](../../scripts/data/tcr_native/TCR_REPERTOIRE_PROCESSING.md)。
v4/v5 prepared 用 `data/tcr_repertoire_junc80/`；旧 core 目录仍在盘上，供 v3 artifact 引用。

---

## 4. 运行手册

### 4.1 训练前必做：语料新鲜度断言

```bash
# 位置参数是构建报告的 JSON 路径（注意两个语料的报告文件名不同）
python scripts/data/tcr_native/assert_corpus_fresh.py \
    data/tcr_papers_v2/dataset/finalize_report.json \
    data/tcr_repertoire/dataset/build_report.json
```

检查两件事：① 报告里 `PASS: true`；② 报告 `blocklist_provenance` 记录的黑名单 sha1 与
**当前磁盘上的活文件**一致（无 provenance 字段则报 `UNVERIFIED`，不静默放过）。
`train_jobs/protein_esmc_llada270m_{diffusion,bert}_immune_v3.yml` 都把它作为 pre-flight
步骤，失败即中止。**不要**退回只看 `PASS` 的写法，原因见 §5.1。

### 4.2 残留泄漏矩阵审计

```bash
python scripts/data/dedup/audit_downstream_leakage.py   # 约 100 秒；--limit N 快速抽查
```

把每一行推过该源**真实的** `row_to_record`（含所有加载期过滤器），只统计存活的行 ——
也就是训练真正看到的数据。输出 5 源 × 11 基准矩阵，结论行只统计 hard-requirement：
`HARD-REQUIREMENT hits = 0 -> PASS`。

**运行前确认没有其他会话在重写语料**（本项目多会话并行，§5.2 那次审计跑到一半语料就被
重写了）：用 `stat -c '%y %s %n'` 连查两次 train.csv，mtime 与大小都不变再开跑。
**审计报出命中时，先确认文件在整个扫描期间没被动过再下结论。**

### 4.3 重建语料

```bash
python scripts/data/tcr_native/build_repertoire.py --max-train 2000000   # 无表位单链 CDR3β
python scripts/data/tcr_native/finalize_papers.py                        # 论文表位层
```

两者都会把 `blocklist_provenance`（每个依赖黑名单的 mtime + sha1）写进构建报告
（分别是 `build_report.json` / `finalize_report.json`）。**任何新建语料的脚本都必须写
这个字段**，否则 `assert_corpus_fresh.py` 只能报 `UNVERIFIED`。

### 4.4 重建黑名单

```bash
python scripts/data/tcr_native/decontam_extra.py --source trait   # 需要 mmseqs，较慢
python scripts/data/tcr_native/build_t4_refbinder_blocklist.py
python scripts/data/tcr_native/build_t2t3_eval_blocklist.py
```

`decontam_extra.py --source trait` 输出 `簇级交集 ∪ binding_benchmark ∪ full_bank`（§5.2）。

> **重建黑名单后，所有依赖它的语料都必须跟着重建。** 二者之间没有自动关联，
> `assert_corpus_fresh.py` 只能*发现*不一致，不会替你修（§5.1 就是漏了这一步）。
> 重建完跑一遍 §4.2 确认 `HARD-REQUIREMENT hits = 0`。

### 4.5 数据配比与 loss 预算

```bash
# 精确计数（全量流式，产出 §1 那张表）
python scripts/count_immune_mix.py train \
    "oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire"

# 布局分布（蓄水池采样，能拆出 tcr_pmhc / tcr_peptide）
python scripts/count_grammar_layouts.py \
    --tokens oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire \
    --tcr-papers-dir data/tcr_papers_v2/dataset \
    --tcr-repertoire-dir data/tcr_repertoire/dataset
```

`count_immune_mix.py` 复用训练入口自己的 `build_immune_specs`，所以数出来的就是训练
看到的数据；**务必传整个 token 列表**，因为 `replaces_trait` 的取代键只在取代源
（`tcr_native`）也被请求时才组装，单独数 `trait` 会偏多。`count_grammar_layouts.py` 看 **mix-weighted**
视图，不要看 per-source 截断视图（后者会把小 TCR 源虚高一个数量级）。

### 4.6 训练前必做：残基字母表断言

```bash
python scripts/data/assert_residue_alphabet.py                  # 七源 × train/valid/holdout
python scripts/data/assert_residue_alphabet.py --paths some.csv  # 单文件
```

发现坏行 **exit 1**，可与 §4.1 并列做 pre-flight gate。默认扫训练实际读取的 21 个文件。

**为什么需要它**：`RemapCollator` 的映射表只覆盖 `RESIDUES`
（`LAGVSERTIDPKQNFYMHWCXBUZO`）加 grammar / chainsep / pad，ESMC 词表里
**id 29 `.`、id 30 `-`、id 31 `|`** 没有像。撞上任何一个都会在 DataLoader worker 里抛
`AssertionError: unmapped decoder grammar ids` 杀死一个 rank，其余 rank 阻塞在
all-gather 直到 NCCL watchdog 1800s 超时，整个任务 Failed ——
**它已经报废过一次 84% 完成度的训练**（§5.4）。

> 脚本里的 `RESIDUES` 常量必须与 `examples/llada/protein_fusion_model.py` 同步。

---

## 5. 已修事故

### 5.1 语料过期：`PASS` 只代表"对构建时那份黑名单干净"

2026-08-28，`tcr_repertoire` 建在 67,013 键的 T4 黑名单上；48 分钟后另一会话把黑名单
重建成 68,846 键，语料没跟着重建，3 个 T4 参考 binder 留在 `train.csv` 里，而
`build_report.json` 依然写着 `PASS: true`。

根因不是那 3 条，而是 `PASS` 的语义：它只表示"对**构建时那份**黑名单干净"。黑名单是活
文件会被重建，语料不会自动跟随 —— 所以过期语料和干净语料在报告上长得**一模一样**。

修法两层：① 报告记 `blocklist_provenance`（mtime + sha1），`assert_corpus_fresh.py`
比对活文件 hash 并做 pre-flight；② 给 `tcr_repertoire` 补加载期过滤（§1.1）。

### 5.2 黑名单构造缺陷：`语料 ∩ benchmark` 挡不住语料重建

`decontam_extra.py::decontaminate_trait` 原本先扫**语料**收集 core 再与 benchmark 求交集，
于是黑名单只对构建时那份语料完备 —— 语料一重建，新增行的 core 从没进过候选集，黑名单
**结构上不可能**挡住它们。实测那 862 个键只覆盖 hard-requirement 保护集的 **4.2%**
（277 / 6,570），即 **TRAIT 此前 0 命中是运气，不是过滤起了作用**；2026-08-28 重建 TRAIT
加的 600 行里就有 4 行直接落在保护 core 上。

修法：改为 `簇级交集 ∪ binding_benchmark ∪ full_bank` = 59,212 键。这不是新发明 ——
`build_repertoire.py` / `finalize_papers.py` 早就用 `binding_benchmark | full_bank` 做与
语料无关的精确黑名单，`decontaminate_trait` 是唯一的例外。簇级那一半仍只能语料派生
（近重复是语料的属性），限制已写进 docstring。代价：TRAIT 保留行 35,472 → 31,515
（−11.2%），附带把 T2/T3 四列全部归零。

> **新增数据源的规则**：精确黑名单必须**benchmark 派生**，不能写成 `语料 ∩ benchmark`。
> 前者对任何语料都完备，后者只对一份快照完备。

### 5.3 审计脚本自己的结论行会误导

原来它把矩阵所有非零格子加总，报 `TOTAL residual hits = 75929 -> LEAKAGE`，但绝大多数
非零格子是 §3 说的配对键投影，按构造就该非零。这是 §5.1 的镜像（那次报告显示干净但不
干净，这次显示脏但不脏），危害相同 —— 一个天天喊狼来了的指标，真出事时没人会信。
修法：区分 hard-requirement 行（必须 0，驱动结论）和信息行（标注为何非零合理）。

### 5.4 比对空位符混进训练 CSV

`records.VALID_PROTEIN_CHARS` 含 `.-`，ingest 把比对空位当合法残基。2026-08-30，
`protein_esmc_llada270m_diffusion_immune_v3_4gpu` 在 **step 42782（50000 的 84%）**被一个
字符打死：`data/tcr_papers_v2/dataset/train.csv` 第 **3046** 行
`cdr3b=ASSKVAARVP-TLKLS`（机制见 §4.6）。

2026-08-31 按训练实际路径全量复扫 21 个 split、约 880 万行，**只这一格**；就地改为
`ASSKVAARVPTLKLS`（行数不变），`finalize_report.json` 记 `out_of_band_edits`，
`assert_corpus_fresh.py` 仍 PASS。ingest / finalize / `cdr3_core` 现在写盘前剥 `.-|`，
没有把这三个 token 映射到 `<res_X>`。提交前 gate：§4.6。

---

## 6. 未解决项

### 6.0 表位源里的 beta-only 行（2026-09-12：v5 已全量核验）

v4 发布时补全只覆盖 `tcr_repertoire`；表位三源的单链行因此仍按位置编码链身份
（`GrammarTokenizer` 无 `<tcra>`/`<tcrb>`）。**补全已在 v5 全量语料落地并核验**
（plan §2.5 / §4.2）。动机、v4 单链占比表、偏离（按实际 fv 列分流）见 plan §2.5 /
§3.1，不要在此复述。
下面这张表是 **v4 prepared 动机快照**（不是 v5 数字），保留是为了说明为什么必须改：

| 源 | train 总数 | beta-only | 占比 | alpha-only |
|---|---:|---:|---:|---:|
| `tcr_papers` | 681,444 | 516,162 | **75.7%** | 0 |
| `tcr_native` | 96,552 | 50,021 | **51.8%** | 4,467 |
| `trait` | 31,515 | 5,155 | 16.4% | 4,418 |
| 合计 | 809,511 | **571,338** | 70.6% | 8,885 |

这不只是布局不一致，是一个**真实的歧义 bug**（§6.2）：8,885 条 alpha-only 与
571,338 条 beta-only 渲染成无法区分的单链受体块。

**2026-09-12：补全已在 v5 全量核验**（按实际 fv/CDR3 列分流，不按 `sequence_scope`；
351 条 `tcr_native` 标 `fv` 却只有 `beta_fv`）。权威实现与偏离记录：plan §2.5 /
§3.1；v5 不变量：plan §4.2。不要在这里复述函数名、config 默认值或 v5 行数。

### 6.0.1 TcrDesign-2026 的 `X` 占位符（2026-09-12：adapter/filter 已拦；ingest 根因未修）

与 §5.4 同类根因：上游用非残基 token 表达“这一栅没有”，而我们的字母表把它当合法残基收下了。

**上游是有意约定，不是数据损坏。** 数据取自 Zenodo `10.5281/zenodo.14545852`
（XSLiuLab / ShanghaiTech，CC-BY-4.0，`tcrdesign_data.tar.gz` 501.2 MB，盘上时间戳
2024-12-24 与 Zenodo 一致），其 README 明写 *"For missing values, please use `'X'` as a
placeholder"*，CLI 示例本身就带 `-alphaj X`。原始 `tcrdesign_B/pMHC_TCR_train.tsv`
743,147 行全量扇描：

```text
epitope 全 X   78,839   长度一律 21
mhc     全 X   82,014   长度一律 34
两者同时全 X        0
```

**bug 在我们这边**：`ingest_papers.py::ingest_td26_pmhc` 的 `_drop_placeholder` 只用在
`cdr3a`/`cdr3b`，**漏在 `epitope` 和 `mhc_pseudo` 上**。两道校验都拦不住，因为 `X` 在
`is_valid_protein_sequence` 里是合法残基（unknown AA）：`ok_epitope('X'*21)` 长度落在
`[6,25]` 内 → 通过；`len(pseudo) != 34` 这道检查对 `'X'*34` 正好等于 34 → 通过。

信号其实**被记录过**：`pseudo2allele.get(pseudo, "")` 对 `X*34` 查不到，
`stats["pseudo_allele_unmapped"]` 每次 +1，最终 15,932 条 `mhc_allele_norm` 为空，
只是没人把它读成“这些是占位符”。连带后果：`_tier(has_mhc=True)` 是**写死的**，那
15,932 条没有 MHC 的行被标成 tier B / `task_type=tcr_pmhc`，tier 标签和
`by_task_type` 计数都是错的。

流到 prepared 的量（`tcr_papers` train 54,272 条，全部来自 `tcrdesign26_pmhc` 单一子源）：
`pepX` 38,689 + `mhcX` 15,583。**这批 relation 分布严重偏斜**：37,313 nonbinding /
16,959 binding（69% 负），而 `tcr_papers` 整体只有 33% 负 —— 而 v4 的 relation 是**可训
target**，留着就是教模型学“表位是一串 X → nonbinding”的人工捷径。

处置结论（plan §2.6：**已在 v5 全量落地并核验**；ingest `_drop_placeholder` 根因**仍未修**）：

- **`mhcX` 可以救，不该删。** MHC 缺失但 peptide 与 CDR3β 都是真的，本该渲染成
  `tcr_peptide` 布局（不带 MHC 块），我们却塞了个全 X 的假 MHC 块。改成不 emit MHC 链
  即可，数据一条不丢，顺带修 tier 标签。
- **`epiX` 应该删**，理由是**没有表位可条件化**（不是“脏”）：只剩 MHC + CDR3β + 标签，
  单靠 MHC 限制性判断结合与否在生物学上没有意义。补齐 `_drop_placeholder` 后
  `ok_epitope("")` 会自然挡掉，删得有据。
- **不反推 MHC。** 交叉扇描确认 `mhcX` 行的 epitope 100% 在别处配着真实 pseudo 出现过，
  其中 85.1%（69,755）映射唯一、14.9%（12,259）有歧义；上游也给了
  `data/mhc_pseudo/mhc_all.dat`（18,639 条 allele→34aa，**无** all-X 条目）。但这是
  **统计共现而非记录级真值** —— 同一 epitope 由多个 HLA 呈递是真实生物学（那 14.9% 就是
  证据），填充等于制造看似可信却无出处的标注。`epiX` 则**信息论上无法恢复**：这批只有
  70 个 distinct pseudo，而 epitope 空间是千级。

净代价：train 负样本少 24,623 条（占 `tcr_papers` 负样本 11.1%）。

> 根因在 ingest 层，但在那儿修要重建 `tcr_papers_v2/dataset/`（连带重跑 dedup + 去污染，
> 且 v4 yml 要求该目录对 v2 checkpoint 保持 byte-identical，只能写新 root）。当前计划是
> 在 adapter 层等价拦截（§6.0 反正要改那里，只需重跑 prepared），并把 ingest 的
> `_drop_placeholder` 补齐记为**下次重建语料时的根因修复**。

### 6.1 T4 Setting-A 参考集自身有构造缺陷（既存问题，非新增数据引入）

`prepare_tcr_generation.py` 的 `--train-cap` 默认 200,000，而 OTS train 有 2,102,700 行 ——
只覆盖 **9.5%**。这个参数同时决定 novelty 参照集和 holdout 的去重比对对象：

| | cap=200k（现行） | cap=0（全量） |
|---|---:|---:|
| novelty 参照（唯一 core） | 180,918 | 1,718,935 |
| holdout 保留（唯一 core） | 9,767 | 7,874 |

即 **19.4% 的 holdout 序列其实躺在训练数据里**（逐条核验：被剔除的 1,893 条全部确认
存在于 train 全量中，无一误杀）。两个方向都在虚高分数。

已做（**只生成不覆盖线上文件**）：`--train-cap 0 --out-dir
downstream/benchmark/data/tcr_generation_fullref`，验证修正版 holdout ∩ OTS-train 全量
= **0**。`tcr_repertoire` 不受影响（建库时用的是 OTS holdout 原始 CSV 的全部 10,516 个
core，是修正后 7,874 的超集）。

**待决策**：没有用修正版参考重跑 T4 打分。现有 Setting A 的 novelty / JSD 是在旧参考上
算的，换参考后会变（预期 novelty 下降），论文里报哪一版需要人定。

### 6.2 单链 α 无法摄入：grammar 不区分 α/β

`tcr_repertoire` 只有 CDR3β。单链 α 数据**不是没找到，是渲染层放不下** ——
`grammar.py::GrammarRenderer.encode` 的受体块按 `[beta, alpha]` 顺序拼接，
`GrammarTokenizer` 只有一个 `<tcr>` 标记，没有 `<tcra>` / `<tcrb>`。配对时靠**位置**
编码身份（第一条 β、第二条 α），所以 `tcr_pair` / `tcr_pmhc` 没问题；但
`len(receptor) == 1` 时两者渲染成**完全相同的输出** —— 实测同一条序列分别标
`role="tcr_alpha"` / `"tcr_beta"`，`grammar_name` 都是 `tcr_single`，全部 11 个输出字段
逐字节相同（输出里根本没有 `chain_ids` 这个键）。

> `GRAMMAR_V1.md` 原写"链身份由 `position_ids_chain` / `chain_ids` 编码"。该说法**只对
> 多链块成立**（身份来自块内位置），单链时不成立；已于 2026-08-29 加上限制说明。

后果：单链 α 摄入 `tcr_single` 会和 2.13M 条 β 混进同一个分布，模型无从知道该生成哪条
链，且会污染 T4 Setting-A 测的 β 分布。要摄入必须先扩 grammar，那是**改词表**，会使现有
checkpoint 不兼容 —— 属于建模决策，未做。

### 6.3 表位条件生成的 loss 预算抬不上去

见 §1：`tcr_pmhc` + `tcr_peptide` 三源合计仅 **1.79%** 的待预测残基。
prepared `BioSeqRecord` 保留 source weight，但 collator 和 loss 计算目前仍不读它；七源默认权重为 `1.0`，所以这是尚未决策的 loss-budget 功能，不是动态 loader 遗留。

### 6.4 加载期过滤在 train / valid 上的丢弃率严重不对称（ASD 根因已查清并修复，2026-09-19）

同一份黑名单、同一个 `row_to_record`，只是 split 不同（2026-09-07 全量实测）：

| 源 | train 丢弃率 | valid 丢弃率 |
|---|---:|---:|
| `asd_antibody` | **67.5%**（850,134 → 276,412） | **5.0%**（47,230 → 44,866） |
| `tcr_native` | 32.7%（143,391 → 96,552） | 12.2%（4,390 → 3,855） |
| `trait` | 54.5%（69,251 → 31,515） | 54.3%（3,050 → 1,393） |

后果：**ASD 的 valid loss 不是可靠的模型选择信号**。valid 保留了 95% 的行，其中包含大量
被从 train 里剥掉的抗体家族（Kong 基准的 CDRH3 0.8 相似簇），即 valid 在测一个训练时被
刻意屏蔽掉的分布；它还占 valid 70% 的残基，会主导整体 `eval_loss`。`trait` 对称说明这不
是通病。没有泄漏风险（valid 是训练期验证集，不是 Kong 基准本身），但拿它做 early-stopping
会误导。

**根因（2026-09-19）：blocklist 的输入就是切分的输出，循环依赖。**
`scripts/data/tcr_native/decontam_extra.py:52` 只读 `step6_final/antibody/train.csv`
生成名单，所以 52 万条 `H:` 键装的是 train 行的**精确重链**；而 step6 按 0.90 整簇装箱，
train/valid **共享 0 条精确重链**——valid 那个近 0 的命中率结构上不可能非零，
它那 3% 全部来自 963 条 `h3:` 键（唯一能跨簇命中的桶）。
此前「机制未查清 / 可能是簇级切分」的猜测**已作废**。

**修复：`downstream/asd/scripts/step6_symmetric.py` 先去污染再切分**，名单从切分的输入
变成副产物。重建后用真实 loader + `build_filters` 实测，train/valid/holdout
三侧去污染命中率均为 **0.00%**。新语料 `downstream/asd/step6_symmetric/`，
新配置 `configs/data/immune_v6_binding_only.yaml`；`step6_final/` 与 v5 配置保持不动。

细节与代价（含 binding-only、抗原 20aa 下限、标签冲突整组丢弃）见
[`downstream/asd/README.md`](../../downstream/asd/README.md)。
**注意 `eval_asd_antibody_loss` 与任何 v5 run 不可横向比**（valid 也被过滤且 split 重建）。

---

## 7. 相关脚本

| 脚本 | 作用 |
|---|---|
| `scripts/count_immune_mix.py` | 七源精确记录数 / 残基数 / loss 预算（§1、§4.5） |
| `scripts/count_grammar_layouts.py` | 布局分布（可拆 `tcr_pmhc` / `tcr_peptide`） |
| `scripts/data/dedup/audit_downstream_leakage.py` | 残留泄漏矩阵（5 源 × 11 基准），查出 §5.1 / §5.2 / §6 |
| `scripts/data/tcr_native/assert_corpus_fresh.py` | 语料新鲜度 + 黑名单 provenance 断言 |
| `scripts/data/assert_residue_alphabet.py` | 残基字母表断言，越界字符 exit 1（§4.6） |
| `scripts/data/tcr_native/build_repertoire.py` | 建无表位单链 CDR3β 语料 |
| `scripts/data/tcr_native/finalize_papers.py` | 建论文表位层语料 |
| `scripts/data/tcr_native/decontam_extra.py` | 建 trait / 抗体黑名单 |
| `scripts/data/tcr_native/build_t4_refbinder_blocklist.py` | 建 T4 答案键黑名单 |
| `scripts/data/tcr_native/build_t2t3_eval_blocklist.py` | 建 T2/T3 评测集黑名单 |
| `scripts/data/tcr_native/assess_candidate.py` | 候选新数据源的净增量评估 |
| `scripts/data/tcr_native/score_layout_diag.py` | 布局诊断打分（官方精确 F1 + 软指标） |

更细的数据扩充过程与决策记录：`data/tcr_papers/EXPANSION_AUDIT_2026_08_28.md`。
语料侧导航：[`data/tcr_papers_v2/README.md`](../../data/tcr_papers_v2/README.md)。
