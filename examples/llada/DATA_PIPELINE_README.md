# 免疫受体数据管线：数据源、去污与审计

> 配套入口脚本 [`protein_pretrain_esmc.py`](protein_pretrain_esmc.py)。
> 训练/模型侧的进展记在 [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md)，
> 本文只管**数据**：有哪些源、什么被过滤掉了、怎么验证过滤真的生效。
> 最近更新：2026-08-29（修 `count_grammar_layouts.py` 前缀采样偏差 → §1；
> 新增按布局的 loss 预算表 §1.1；新增未解项 §6.2 单链 α、§6.4 split 丢弃率不对称）

本文档存在的理由：这条管线上出现过三次同一类事故 —— **过滤看起来生效了，其实没有**。
每次都不是过滤逻辑写错，而是"过滤依据"和"被过滤的数据"之间没有强制关联。
第 5 节把三次事故的根因和修法写全，新增数据源前请先读。

---

## 1. 当前数据源

`--dataset_args oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`

| 源 | 内容 | grammar 布局（实测占比） | 去污发生在 |
|---|---|---|---|
| `oas` | 抗体配对重/轻链 | `antibody_pair` 100% | 加载期 |
| `ots` | TCR 配对全长 α/β | `tcr_pair` 100% | 加载期 |
| `asd_antibody` | 抗体–抗原 | `antigen_antibody` 100% | 加载期 |
| `trait` | TCR CDR3 + 表位（多数带 MHC） | `tcr_pmhc` 95.2% / `tcr_peptide` 4.8% | 加载期 |
| `tcr_native` | TCR 全长 + pMHC | `tcr_pmhc` 99.9% / `tcr_peptide` 0.1% | 构建期 **+** 加载期 |
| `tcr_papers` | 论文表位层（→ `tcr_papers_v2`） | `tcr_pmhc` 80.3% / `tcr_peptide` 19.7% | 构建期 **+** 加载期 |
| `tcr_repertoire` | **无表位单链 CDR3β** | `tcr_single` 100% | 构建期 **+** 加载期 |

> ⚠️ 布局占比必须**全量或蓄水池采样**测，不能取前缀。`tcr_papers_v2` 是 7 个论文语料
> 首尾拼接的，它的**前 3 万行 100% 是 `tcr_peptide`**，而全量是 80.3% `tcr_pmhc`。
> `count_grammar_layouts.py` 原先用前缀采样（`break`），把 547,274 条 `tcr_pmhc`
> 误记成 `tcr_peptide`，把 `tcr_pmhc` 低估约 4 倍；2026-08-29 已改成蓄水池采样。
> 布局是「行填了哪些字段」的函数，源内不同质时前缀就不具代表性。

语料行数（2026-08-29 13:30 复核）：

| 语料 | train | valid | holdout |
|---|---:|---:|---:|
| `data/tcr_papers_v2/dataset` | 682,383 | 7,336 | 7,201 |
| `data/tcr_repertoire/dataset` | 2,128,750 | 123,049 | 122,669 |

> ⚠️ **`tcr_repertoire` 近期反复在变，但三次变动的性质不同，别混为一谈**
> （train 1,971,136 → 1,971,794 → 2,128,750）：
>
> | 时间 | 性质 | 动作 |
> |---|---|---|
> | 08-28 11:50 | blocklist 重建后**重跑构建** | `build_repertoire.py` |
> | 08-28 19:35 | 同上 | `build_repertoire.py` |
> | 08-29 03:35 | **valid 近重复搬迁**，非重建 | `move_near_dup_eval_rows.py --apply` |
>
> 🔴 **因此 `build_report.json::split_counts` 已过期，不能再用它查行数**——它记的是构建期
> 的 1,971,794 / 201,504 / 201,170，而 03:35 的搬迁只改 CSV、不重写该报告。
> **行数以 CSV 和 `data/tcr_native/dataset/near_dup_eval_move_report.json` 为准；
> `build_report.json` 仍是「去污染」口径的权威来源**（`PASS` / `decontam_mode` /
> `blocklist_provenance`）。
> 每次变动后跑 §4.1 的新鲜度断言 + §4.2 的泄漏审计；03:35 那次两者均通过
> （本源对 11 个基准全 0，且 `count_immune_mix.py` 测得 kept=raw、剔 0）。

加载期过滤后的七源实际配比（记录数，2026-08-29）：
`oas` 31.89% / `tcr_repertoire` 27.31% / `ots` 26.87% / `tcr_papers` 8.74% /
`asd_antibody` 3.55% / `tcr_native` 1.24% / `trait` 0.40%，合计 **7,794,375** 条。
注意**记录占比不等于 loss 预算占比**：`tcr_repertoire` 每条只有约 13 个待预测残基，
`oas` 约 232、`asd_antibody` 约 523，所以它占 27% 记录却只占约 2% 残基预算。

### 1.1 按 grammar 布局看 loss 预算

这张表才是「模型实际在学什么」。`gen%` = 该布局在全部待预测残基里的占比
（`scripts/count_grammar_layouts.py`，蓄水池采样 3 万行/源后按保留行数放大）：

| 布局 | 条件 → 生成 | 记录数 | 记录% | **gen%** |
|---|---|---:|---:|---:|
| `antibody_pair` | 无条件 → 抗体 H+L | 2,485,471 | 31.9% | **49.64%** |
| `tcr_pair` | 无条件 → TCR α+β 全长 | 2,094,231 | 26.9% | **40.92%** |
| `antigen_antibody` | 抗原 → 抗体 H+L | 276,412 | 3.6% | **4.55%** |
| `tcr_single` | 无条件 → 单链 CDR3β | 2,128,750 | 27.3% | **2.90%** |
| `tcr_pmhc` | MHC+表位 → TCR | 671,678 | 8.6% | **1.78%** |
| `tcr_peptide` | 仅表位 → TCR | 137,833 | 1.8% | **0.20%** |

三个无条件/抗体布局吃掉 **95.1%** 的 loss 预算；**表位条件生成合计只有 1.98%**。
根因是残基数量级差异而非行数：`antibody_pair` 每条约 232 个待预测残基、
`tcr_pair` 约 231，而 `tcr_pmhc`/`tcr_peptide` 每条只有 13–31 个（只生成 CDR3）。
**靠加数据改不动这个比例**——`tcr_papers` 行数翻 3 倍也只把 1.98% 抬到约 3%。
要真正抬高必须做 per-sample loss 加权，而 `ImmuneSourceSpec.weight` 目前
只存不用（见 §6 未解决项）。

`tcr_repertoire` 是 2026-08-28 新增的，来自 TcrDesign-2026 的 `pretrain/bCDR3_train.csv`。
它是**唯一**渲染成 `tcr_single` 的源，而 T4 Setting-A 无条件生成基准恰好用这个布局解码 ——
在它之前该布局的训练覆盖率是 0%。

### 去污发生在哪一层，决定了它会不会过期

**加载期过滤**：每次训练启动时用 `with_exclusion_filter` 读黑名单现场过滤，
所以**天然新鲜** —— 黑名单改了下一次训练自动生效，不存在过期问题。
代价是它的正确性完全押在黑名单文件的**完备性**上，这就是 §5.2 把修法选在
黑名单层（而不是去改语料）的原因。

**构建期过滤**：去污在建库时一次性做完，写死进 CSV。黑名单事后重建
**不会**回溯到已生成的语料，所以必须靠 `assert_corpus_fresh.py` 显式比对
（§4.1、§5.1）。

三个 unified-schema 的源两层都有 —— 建库时已按全套黑名单去污，
加载期又在 `build_immune_specs` 里叠了一层：

| 源 | 加载期叠的过滤 |
|---|---|
| `tcr_native` / `tcr_papers` | `_tcr_native_row_to_record`：`t4_refbinder` + `t2t3_eval`（配对键） |
| `tcr_repertoire` | `repertoire_core_exclusions` = `t4` ∪ `t2t3` **投影成裸 core** ∪ `ots_benchmark` |

`tcr_repertoire` 无表位，配对键永远匹配不上，所以必须投影成裸 core。
这是**故意过挡**：一条 benchmark CDR3β 不论当初是哪个表位的答案都会被丢掉。

这层是 §5.1 事故的修复产物。`tcr_repertoire` 最初**没有**加载期过滤
（当时的代码注释写着"构建期已去污，无表位可作键，故不叠过滤器"），
所以那 3 条 T4 参考 binder 落在磁盘上就等于会进训练。补上这层之后实测：

```
ASSFGGRSYEQY -> 挡住   ASSFLAGQETQY -> 挡住   SAPTDTQY -> 挡住
```

**双层不是冗余，它是唯一能兜住"黑名单重建了但语料没跟着重建"的机制。**
新鲜度断言只能*发现*脱节，加载期过滤才能*挡住*后果。
但只有构建期那三个源需要断言，纯加载期的源不需要，也不在 provenance 里。

> 新增数据源时先决定放哪一层。构建期更省启动开销，但欠一份 provenance
> 就等于欠一个静默失效的坑；能加一层加载期过滤就加上。

---

## 2. 黑名单清单

全部在 `data/tcr_native/dataset/` 下。键数为 2026-08-28 实测。

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

启动时入口会打印实际加载的键数，和上表对不上就说明黑名单被改过：

```
Blocklist key counts: replaces_trait=11843 trait_benchmark=59212 ots=4494 oas=661
  asd_ab=521073 asd_nb=2086 t4_refbinder=68846 t2t3_eval=8377
```

---

## 3. 两个键空间，别混用

这是读任何去污结果时最容易出错的地方。

**裸 core 键** —— 保护集就是 CDR3β core 本身。命中一条就是泄漏答案，零容忍。
只有三个基准属于这一类：`NM2025_seen`、`NM2025_unseen`、`public_trackA`。
代码里 `decontam.py::load_benchmark_sets()` 返回的 `binding_benchmark`
就是这三者的并集，其 docstring 写明是 "the HARD-requirement gate"。

**`(core|epitope)` 配对键** —— 保护的是"某条 CDR3β 配某个表位"这个组合。
同一条 CDR3β 配**不同**表位是合法训练信号，不是泄漏。
`t4_refbinder` / `t2t3_eval` / `replaces_trait` 都是这一类。

后果：把配对键投影成裸 core 去比对，会**按构造过报**。实测
`tcr_papers_v2` 有约 7.7 万、`tcr_native` 有约 7.4 万裸 core 命中 T4 黑名单，
但 `(core|epitope)` 配对层命中是 **0**。`t4_refbinder_blocklist.txt`
的 68,846 个键 100% 含 `|`，全是配对键。

审计脚本因此区分"必须为 0 的 hard-requirement 行"和"非零也合理的信息行"，
不要看总和。

另外，**anchor 约定**不统一：CDR3 可能存成完整 IMGT junction（`C..[FW]`）
或去锚的 loop。直接比会静默漏掉全部命中。各源的约定声明在
`audit_downstream_leakage.py::_SOURCES`：

| 源 | 列 | `has_anchors` |
|---|---|---|
| `ots` | 按 `anarci_type` 取 β | False（loop） |
| `trait` | `cdr3b` | **True**（完整 junction） |
| `tcr_native` / `tcr_papers` / `tcr_repertoire` | `cdr3b` | False（统一 schema 存 core） |

---

## 4. 运行手册

### 4.1 训练前必做：语料新鲜度断言

```bash
# 位置参数是构建报告的 JSON 路径（注意两个语料的报告文件名不同）
python scripts/data/tcr_native/assert_corpus_fresh.py \
    data/tcr_papers_v2/dataset/finalize_report.json \
    data/tcr_repertoire/dataset/build_report.json
```

它检查两件事：① 构建报告里 `PASS: true`；② 报告中
`blocklist_provenance` 记录的黑名单内容 sha1 与**当前磁盘上的活文件**一致。
没有 provenance 字段的语料报 `UNVERIFIED` 而不是静默放过。

`train_jobs/protein_esmc_llada270m_diffusion_immune_v3.yml` 与
`train_jobs/protein_esmc_llada270m_bert_immune_v3.yml` 都把它作为
pre-flight 步骤，失败即中止训练。两臂数据参数逐字节一致，唯一差异是目标函数
（见 [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2）。
**不要**退回只看 `PASS` 的写法，原因见 §5.1。

### 4.2 残留泄漏矩阵审计

```bash
python scripts/data/dedup/audit_downstream_leakage.py
# 约 100 秒。--limit N 可快速抽查
```

它把每一行都推过该源**真实的** `row_to_record`（即 `build_immune_specs`
返回的那个，含所有加载期过滤器），只统计存活下来的行 —— 也就是训练真正看到的数据。
输出是 5 源 × 11 基准的矩阵，结论行只统计 hard-requirement：

```
HARD-REQUIREMENT hits = 0   -> PASS
```

**运行前确认没有其他会话在重写语料**，否则可能读到半写文件。
本项目是多会话并行的，这不是理论风险 —— §5.2 那次审计跑到一半，
另一会话正好重写了 TRAIT 和 `tcr_papers_v2`：

```bash
stat -c '%y %s %n' downstream/trait/step4_final/train.csv \
    data/tcr_papers_v2/dataset/train.csv data/tcr_repertoire/dataset/train.csv
```

连查两次、mtime 与大小都不变，再开始跑。**审计报出命中时，
先确认文件在整个扫描期间没被动过，再下结论** —— §5.2 那 4 条是等
mtime 稳定 40 秒后复核确认为真泄漏的，不能直接采信第一次的结果。

### 4.3 重建语料

```bash
# 无表位单链 CDR3β
python scripts/data/tcr_native/build_repertoire.py --max-train 2000000
# 论文表位层
python scripts/data/tcr_native/finalize_papers.py
```

两者都会把 `blocklist_provenance`（每个依赖黑名单的 mtime + sha1）写进构建报告
（`build_repertoire.py` → `build_report.json`，`finalize_papers.py` →
`finalize_report.json`）。**任何新建语料的脚本都必须写这个字段**，
否则 `assert_corpus_fresh.py` 只能报 `UNVERIFIED`。

### 4.4 重建黑名单

```bash
python scripts/data/tcr_native/decontam_extra.py --source trait   # 需要 mmseqs，较慢
python scripts/data/tcr_native/build_t4_refbinder_blocklist.py
python scripts/data/tcr_native/build_t2t3_eval_blocklist.py
```

`decontam_extra.py --source trait` 需要 `mmseqs`（簇级那一半），较慢；
它输出的是 `簇级交集 ∪ binding_benchmark ∪ full_bank`（§5.2）。

> **重建黑名单后，所有依赖它的语料都必须跟着重建。** 二者之间没有自动关联，
> `assert_corpus_fresh.py` 只能*发现*不一致，不会替你修。§5.1 就是漏了这一步。
> 重建完记得跑一遍 §4.2 的审计确认 `HARD-REQUIREMENT hits = 0`。

### 4.5 布局分布（真实 loss 预算）

```bash
python scripts/count_grammar_layouts.py \
    --tokens oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire \
    --tcr-papers-dir data/tcr_papers_v2/dataset \
    --tcr-repertoire-dir data/tcr_repertoire/dataset
```

看 **mix-weighted** 视图，不要看 per-source 截断视图 —— 后者会把小 TCR 源的
占比虚高一个数量级。

### 4.6 训练前必做：残基字母表断言

```bash
python scripts/data/assert_residue_alphabet.py                 # 七源 × train/valid/holdout
python scripts/data/assert_residue_alphabet.py --split train
python scripts/data/assert_residue_alphabet.py --paths some.csv  # 单文件
```

发现坏行时 **exit 1**，所以可以和 §4.1 并列做 pre-flight gate。默认扫训练实际读取的
21 个文件（含 `tcr_native` 与 OAS `*_oas_label.csv`）。

**为什么需要它。** `examples/llada/protein_fusion_model.py::RemapCollator` 把 ESMC grammar id
映射进扩展后的 LLaDA 词表，而映射表只覆盖 `RESIDUES`
（`LAGVSERTIDPKQNFYMHWCXBUZO`，25 个字符，含 XBUZO 模糊码）加 grammar / chainsep / pad。
ESMC 词表里剩下三个单字符 token **没有像**：**id 29 `.`、id 30 `-`、id 31 `|`**。
撞上任何一个都会在 DataLoader worker 里抛
`AssertionError: unmapped decoder grammar ids`，杀死一个 rank，其余 rank 阻塞在 all-gather
直到 NCCL watchdog 1800s 超时，整个任务 Failed。

**这个断言不是假想威胁 —— 它已经报废了一次接近完成的训练。** 2026-08-30，
`protein_esmc_llada270m_diffusion_immune_v3_4gpu` 在 **step 42782（50000 的 84%）**
被一个字符打死：`data/tcr_papers_v2/dataset/train.csv` 第 **3046** 行
`cdr3b` = `ASSKVAARVP-TLKLS`。2026-08-31 全量复扫 21 个 split、约 880 万行，
**只这一格**；已就地改为 `ASSKVAARVPTLKLS`（行数不变）。ingest / finalize /
`cdr3_core` 写盘前剥 `.-|`。完整分析见
[`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.6。

> 脚本里的 `RESIDUES` 常量必须与 `examples/llada/protein_fusion_model.py` 同步。

---

## 5. 已修事故：过滤看起来生效了其实没有；外加一次字母表漏网

### 5.1 语料过期：`PASS` 只代表"对构建时那份黑名单干净"

2026-08-28，`tcr_repertoire` 建在 67,013 键的 T4 黑名单上；48 分钟后
另一会话把黑名单重建成 68,846 键，但语料没跟着重建。
结果 3 个 T4 参考 binder（`ASSFGGRSYEQY`、`ASSFLAGQETQY`、`SAPTDTQY`）
留在 `train.csv` 里，而 `build_report.json` 依然写着 `PASS: true`。

根因不是那 3 条，而是 `PASS` 的语义：它只表示"对**构建时那份**黑名单干净"。
黑名单是活文件会被重建，语料不会自动跟随，两者之间没有任何关联 ——
所以一份过期语料和一份干净语料在报告上长得**一模一样**。

修法两层：① 报告记 `blocklist_provenance`（mtime + 内容 sha1），
新增 `assert_corpus_fresh.py` 比对活文件 hash，v3 配置改用它做 pre-flight；
② 给 `tcr_repertoire` 补上加载期过滤（`repertoire_core_exclusions`，见 §1）——
断言只能*发现*脱节，加载期过滤才能*挡住*后果。实测那 3 条现在会被挡下。

### 5.2 黑名单构造缺陷：交集写法挡不住语料重建

`decontam_extra.py::decontaminate_trait` 原本先扫**语料**收集 core，
再与 benchmark 求交集，把交集写成黑名单。于是黑名单是 `语料 ∩ benchmark`，
只对构建时那份语料完备 —— 语料一重建，新增行的 core 从没进过候选集，
黑名单**结构上不可能**挡住它们。

实测那 862 个键只覆盖 hard-requirement 保护集的 **4.2%**（277 / 6,570）：

| 基准 | 保护 core | 被覆盖 | 覆盖率 |
|---|---:|---:|---:|
| NM2025_seen | 4,254 | 36 | 0.8% |
| NM2025_unseen | 1,091 | 70 | 6.4% |
| public_trackA | 1,306 | 172 | 13.2% |

也就是说 **TRAIT 此前 0 命中是运气，不是过滤起了作用**。
2026-08-28 17:16 另一会话重建 TRAIT，加了 600 行，其中 4 行直接落在保护 core 上
（2 条 NM2025_unseen、2 条 public_trackA，全是无表位的裸 CDR3b 行）。

修法：黑名单改为 `簇级交集 ∪ binding_benchmark ∪ full_bank` = 59,212 键。
这不是新发明 —— `build_repertoire.py` 和 `finalize_papers.py` 早就直接用
`binding_benchmark | full_bank` 做与语料无关的精确黑名单，
`decontaminate_trait` 是唯一的例外。
簇级那一半仍只能语料派生（近重复是语料的属性而非 benchmark 的属性），
限制已写进 docstring。

代价：TRAIT 保留行 35,472 → 31,515（**−3,957 / −11.2%**）。
附带好处：TRAIT 在 T2/T3 四列也从 162/2,332/287/208 全部归零。
原 862 键文件备份为 `trait_benchmark_blocklist.txt.pre_union_bak`。

> **新增数据源时的规则**：精确黑名单必须**benchmark 派生**，
> 不能写成 `语料 ∩ benchmark`。前者对任何语料都完备，后者只对一份快照完备。

### 5.3 审计脚本自己的结论行会误导

原来它把矩阵里所有非零格子加总，报
`TOTAL residual hits = 75929 -> LEAKAGE`。但绝大多数非零格子是 §3
说的配对键投影，按构造就该非零。

这是 §5.1 的镜像：那次是报告显示干净但其实不干净，这次是报告显示脏但其实不脏。
两种都会训练出"看报告"的习惯而不是看证据，危害相同 ——
一个天天喊狼来了的指标，等真出事时没人会信。

修法：区分 hard-requirement 行（必须 0，驱动结论）和信息行
（标注为何非零合理），并在脚注写明 T4 黑名单全是配对键、配对层命中为 0。

### 5.4 比对空位符混进训练 CSV（2026-08-31 修）

`records.VALID_PROTEIN_CHARS` 含 `.-`，ingest 把比对空位当合法残基。
`RemapCollator` 映射表不覆盖 ESMC id 29/30/31（`.` / `-` / `|`），
4 卡 diffusion 在 step 42782 反复崩在
`data/tcr_papers_v2/dataset/train.csv` 第 3046 行
`cdr3b=ASSKVAARVP-TLKLS`。

2026-08-31 按训练实际路径扫七源 × train/valid/holdout（21 个文件、约 880 万行），
**只这一格**。就地改为 `ASSKVAARVPTLKLS`，行数不变；`finalize_report.json` 记
`out_of_band_edits`；`assert_corpus_fresh.py` 仍 PASS。ingest / finalize /
`cdr3_core` 写盘前剥 `.-|`。没有把这三个 token 映射到 `<res_X>`。

语料导航：[`data/tcr_papers_v2/README.md`](../../data/tcr_papers_v2/README.md)
「残基字母表」。训练侧经过见 [`PROTEIN_PRETRAIN_PROGRESS.md`](PROTEIN_PRETRAIN_PROGRESS.md) §4.2.6。
提交前 gate：`python scripts/data/assert_residue_alphabet.py`（§4.6）。

---

## 6. 未解决项

### 6.1 T4 Setting-A 参考集自身有构造缺陷

**这是既存问题，与新增数据无关**，去重审计的副产品。

`downstream/benchmark/scripts/prepare_tcr_generation.py` 的 `--train-cap`
默认 200,000，而 OTS train 有 2,102,700 行 —— 只覆盖 **9.5%**。
这个参数同时决定两件事：novelty 参照集，以及 holdout 的去重比对对象。

| | cap=200k（现行） | cap=0（全量） |
|---|---:|---:|
| novelty 参照（唯一 core） | 180,918 | 1,718,935 |
| holdout 保留（唯一 core） | 9,767 | 7,874 |

即 **19.4% 的 holdout 序列其实躺在训练数据里**。逐条核验：被剔除的 1,893 条
全部（1,893/1,893）确认存在于 train 全量中，无一误杀。
两个方向都在虚高分数 —— 训练集第 200k–2.1M 行的序列被原样吐出也算"新"；
目标分布里混了 19.4% 训练集内序列，模型越记住训练集反而越"匹配" held-out。

已做（**只生成不覆盖线上文件**）：

```bash
python downstream/benchmark/scripts/prepare_tcr_generation.py \
    --train-cap 0 --out-dir downstream/benchmark/data/tcr_generation_fullref
```

验证：修正版 holdout ∩ OTS-train 全量 = **0**。
`tcr_repertoire` 不受影响 —— 它建库时用的是 OTS holdout **原始 CSV** 的全部
10,516 个 core，是修正后 7,874 的超集，实测命中 0。

**未做 / 待决策**：没有用修正版参考重跑 T4 打分。现有 Setting A 的
novelty / JSD 数字是在旧参考上算的，换参考后会变（预期 novelty 下降）。
要不要重跑、论文里报哪一版，会改动已有结论，需要人决定。

### 6.2 单链 α 无法摄入：grammar 不区分 α/β

`tcr_repertoire` 只有 CDR3β。单链 α 数据**不是没找到，是渲染层放不下**。

`grammar.py` 的受体块按 `[alpha, beta]` 顺序拼接，`GrammarTokenizer` 只有一个
`<tcr>` 标记，没有 `<tcra>` / `<tcrb>`：

```466:472:dllm/pipelines/qwen3_vl_arch/data/grammar.py
                receptor = [chain for chain in (alpha, beta) if chain is not None]
                if len(receptor) == 1:
                    append_protein_block(
                        receptor,
                        type_marker="<tcr>",
                        is_fixed=False,
                    )
```

配对时靠**位置**编码身份（第一条是 α、第二条是 β），所以 `tcr_pair` / `tcr_pmhc`
没问题。但 `len(receptor) == 1` 时两者渲染成**完全相同的输出**。

实测（同一条序列分别标 `role="tcr_alpha"` 和 `role="tcr_beta"`）：
两者 `grammar_name` 都是 `tcr_single`，且**全部 11 个输出字段逐字节相同** ——
`input_ids` / `position_ids_chain` / `position_ids_inner` / `token_class_ids`
以及三个 mask 全部一致。渲染输出里根本没有 `chain_ids` 这个键。

> `GRAMMAR_V1.md` 原写"链身份由 `position_ids_chain` / `chain_ids` 编码，
> 不靠 per-role token 名"。该说法**只对多链块成立**（身份来自块内位置），
> 单链时不成立。已于 2026-08-29 在该文件加上限制说明。

后果：单链 α 摄入 `tcr_single` 会和 2.13M 条 β 混进同一个分布，
模型无从知道该生成哪条链，且会污染 T4 Setting-A 测的 β 分布。

要摄入 α 必须先扩 grammar（加 `<tcra>`/`<tcrb>` 或单链 role 标记），
那是**改词表**，会使现有 checkpoint 不兼容。属于建模决策，未做。

### 6.3 表位条件生成的 loss 预算抬不上去

见 §1.1：`tcr_pmhc` + `tcr_peptide` 合计仅 **1.98%** 的待预测残基。
`ImmuneSourceSpec.weight` 字段存在且已写进 `BioSeqRecord`，但
collator 和 loss 计算**都不读它** —— 七源当前全是默认 `1.0`，
所以这是缺失功能而非静默 bug。不实现加权就只能接受这个比例。

### 6.4 加载期过滤在 train / valid 上的丢弃率严重不对称

2026-08-29 实测（同一份黑名单、同一个 `row_to_record`，只是 split 不同）：

| 源 | train 丢弃率 | valid 丢弃率 | 差 |
|---|---:|---:|---:|
| `asd_antibody` | **67.5%**（850,134 → 276,412） | **5.0%**（47,230 → 44,866） | 13x |
| `tcr_native` | 32.7%（143,391 → 96,552） | 12.2%（4,390 → 3,855） | 2.7x |
| `trait` | 54.5%（69,251 → 31,515） | 54.3%（3,050 → 1,393） | 对称 ✓ |

后果：**ASD 的 valid loss 不是可靠的模型选择信号**。valid 保留了 95% 的行，
其中包含大量被从 train 里剥掉的抗体家族（Kong 基准的 CDRH3 0.8 相似簇），
即 valid 在测一个训练时被刻意屏蔽掉的分布。`trait` 对称说明这不是通病。

没有泄漏风险（valid 是训练期验证集，不是 Kong 基准本身），
但拿 ASD valid loss 做 early-stopping 会误导。机制未查清 —— 可能是
ASD 的簇级切分让 Kong 相似簇集中落在 train，需要单独确认。
**待决策**：要么重建 ASD 语料使两个 split 去污一致，要么在监控里
把 ASD valid loss 标注为不可比。

---

## 7. 相关脚本

| 脚本 | 作用 |
|---|---|
| `scripts/data/dedup/audit_downstream_leakage.py` | 残留泄漏矩阵（5 源 × 11 基准），查出 §5.1 / §5.2 / §6 |
| `scripts/data/tcr_native/assert_corpus_fresh.py` | 语料新鲜度 + 黑名单 provenance 断言 |
| `scripts/data/assert_residue_alphabet.py` | 残基字母表断言：序列里出现 `RESIDUES` 之外的字符就 exit 1（§4.6） |
| `scripts/data/tcr_native/build_repertoire.py` | 建无表位单链 CDR3β 语料 |
| `scripts/data/tcr_native/finalize_papers.py` | 建论文表位层语料 |
| `scripts/data/tcr_native/decontam_extra.py` | 建 trait / 抗体黑名单 |
| `scripts/data/tcr_native/build_t4_refbinder_blocklist.py` | 建 T4 答案键黑名单 |
| `scripts/data/tcr_native/build_t2t3_eval_blocklist.py` | 建 T2/T3 评测集黑名单 |
| `scripts/count_grammar_layouts.py` | 布局分布 / loss 预算占比 |
| `scripts/data/tcr_native/assess_candidate.py` | 候选新数据源的净增量评估 |
| `scripts/data/tcr_native/score_layout_diag.py` | 布局诊断打分（官方精确 F1 + 软指标） |

更细的数据扩充过程与决策记录：
`data/tcr_papers/EXPANSION_AUDIT_2026_08_28.md`。
