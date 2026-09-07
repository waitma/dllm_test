# Data Format Audit

Date: 2026-06-14

Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`

## 当前训练语料实测快照（2026-08-28）

本节取代下方「当前训练数据单一入口（2026-07-23）」。那一节描述的是
`data/bioseq_grammar_v1` 的 7 源 Arrow 配方（7L step389500 lineage），**与现在
`examples/llada/protein_pretrain_esmc.py` 实际读取的语料无关**。当前唯一训练入口为：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`

它按 `--dataset_args` 的 `+` token 列表直读 CSV，不经过 `bioseq_grammar_v1`，也不经过
`data/immune_receptor_v2`（后者仍 `training_ready=false`）。

### 七源实测（v3 mix，`max_length=max_protein_length=1024`，全部 blocklist 生效）

2026-08-29 05:30 由
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/count_immune_mix.py`
实测，非文档转抄。**这是 valid 近重复搬迁全部落地后的口径。**

| source | 目录 | 原始 | 保留 | 剔除 | 记录% | 残基% | 生成残基% | res/rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `oas` | `data/oas_previous_clean/splits` | 2,486,442 | 2,485,471 | 0.04% | 31.89% | 45.17% | 50.01% | 232 |
| `tcr_repertoire` | `data/tcr_repertoire/dataset` | 2,128,750 | 2,128,750 | 0.00% | 27.31% | 2.19% | 2.42% | **13** |
| `ots` | `data/ots_paired_clean/final` | 2,102,715 | 2,094,231 | 0.40% | 26.87% | 37.22% | 41.21% | 226 |
| `tcr_papers` | `data/tcr_papers_v2/dataset` | 682,383 | 681,444 | 0.14% | 8.74% | 2.84% | 0.91% | 53 |
| `asd_antibody` | `downstream/asd/step6_final/antibody` | 850,134 | 276,412 | **67.49%** | 3.55% | **11.35%** | 4.57% | **523** |
| `tcr_native` | `data/tcr_native/dataset` | 143,391 | 96,552 | 32.67% | 1.24% | 1.06% | 0.82% | 140 |
| `trait` | `downstream/trait/step4_final` | 69,251 | 31,515 | **54.49%** | 0.40% | 0.16% | 0.07% | 66 |
| **合计** | | | **7,794,375** | | | | | |

总残基 1,273,914,576，生成链残基 1,150,746,721。

相对 2026-08-28 11:50 那一版（合计 7,626,737）的两处变化，**成因不同，不要混为一谈**：

1. **valid 近重复搬迁**（本会话所做）把 eval 行移入 train，抬高了三个源的 raw：
   `tcr_repertoire` 1,971,794→2,128,750（**+7.96%**）、`tcr_papers` 668,331→682,383、
   `trait` 67,842→69,251。`tcr_repertoire` 保留数 = raw（剔 0），说明搬进 train 的
   156,956 行没有任何 blocklist 命中——它们本来就在同一个去污染过的 pool 里。
2. **`trait_benchmark` blocklist 被并发会话重建**（文件 mtime 2026-08-28 17:25:35Z，
   862 → **59,212** 键），使 `trait` 的剔除率从 48.60% 升到 54.49%、保留数从 34,872 掉到
   31,515。**这不是搬迁造成的**：搬迁只增加 raw。`trait` 是加载期过滤，所以无需重建语料
   即自动跟上新 blocklist。

> ⚠️ `tcr_repertoire` 的 train 现在是 **2,128,750**，超过了当初刻意设的 **200 万** token
> 预算上限（见 `data/tcr_repertoire/README.md`「为什么 train 截到 200 万」）。超出 6.4%，
> 残基占比从 2.04% 升到 2.19%，判断为可接受，未回切。若日后重建该语料，注意 2,000,000
> 这个上限会把搬迁回来的行重新截掉。

v2 mix 与 v3 只差 `tcr_papers` 一个目录：指 v1 的 `data/tcr_papers/dataset` 时
raw 408,037 / kept 407,112，合计 **7,366,444** 条，总残基 1,258,656,430，
生成链残基 1,144,047,600。

按监督类型（当前口径）：无标签配对（`oas`+`ots`）58.76%，无标签单链 CDR3β
（`tcr_repertoire`）27.31% → **无标签合计 86.07%**；有标签 TCR 识别
（`trait`+`tcr_native`+`tcr_papers`）**10.39%**；抗体-抗原（`asd_antibody`）**3.55%**。

#### 按 grammar 布局的 loss 预算（2026-08-29 蓄水池采样实测）

上表是**按源**的，但一个源可以产出多种布局，所以它答不了"模型在学哪些任务"。
按布局看（`gen%` = 占全部待预测残基）：

| 布局 | 条件 → 生成 | 来源 | 保留行数 | 记录% | **gen%** |
|---|---|---|---:|---:|---:|
| `antibody_pair` | 无条件 → 抗体 H+L | `oas` | 2,485,471 | 31.9% | **49.64%** |
| `tcr_pair` | 无条件 → TCR α+β 全长 | `ots` | 2,094,231 | 26.9% | **40.92%** |
| `antigen_antibody` | 抗原 → 抗体 H+L | `asd_antibody` | 276,412 | 3.6% | **4.55%** |
| `tcr_single` | 无条件 → 单链 CDR3β | `tcr_repertoire` | 2,128,750 | 27.3% | **2.90%** |
| `tcr_pmhc` | MHC+表位 → TCR | `trait`+`tcr_native`+`tcr_papers` | 671,678 | 8.6% | **1.78%** |
| `tcr_peptide` | 仅表位 → TCR | 同上三源里无 MHC 的部分 | 137,833 | 1.8% | **0.20%** |

三源到布局的实测拆分（全量扫描）：`trait` 95.2% pmhc / 4.8% peptide、
`tcr_native` 99.9% / 0.1%、`tcr_papers` 80.3% / 19.7%。

三个无条件/抗体布局吃掉 **95.1%** 的 loss 预算，表位条件生成合计仅 **1.98%**。
根因是残基数量级（232 vs 13–31）而非行数，**加数据改不动这个比例**。

> 🔴 **2026-08-29 前所有布局占比数字都是错的。** `count_grammar_layouts.py` 当时读满
> `--per-source` 就 `break`，取的是**前缀**。`tcr_papers_v2` 是 7 个论文语料首尾拼接的，
> 前 3 万行 **100% 是 `tcr_peptide`**，全量却是 80.3% `tcr_pmhc` —— 547,274 条被归错，
> `tcr_pmhc` 低估约 4 倍（旧值 8.10%/2.47%，实为 1.8%/8.6%）。已改蓄水池采样 + `--seed`，
> 新数字与独立全量扫描一致（671,678 vs 673,686）。
> **规则：源内不同质时任何抽样都必须随机抽，并与一次全量扫描对齐过。**

#### 剔除率在 train 与 valid 上不对称（2026-08-29 实测）

上表的「剔除%」只是 **train** 的。同一份 blocklist 换到 valid 上差别很大：

| 源 | train 剔除 | valid 剔除 |
|---|---:|---:|
| `asd_antibody` | **67.5%** | **5.0%**（47,230 → 44,866） |
| `tcr_native` | 32.7% | 12.2%（4,390 → 3,855） |
| `trait` | 54.5% | 54.3%（3,050 → 1,393） |

`trait` 对称，另两个不对称。后果是 **ASD 的 valid loss 不适合做模型选择** ——
valid 保留了大量被从 train 剥掉的 Kong 相似簇。详见
`downstream/asd/README.md` 与 `examples/llada/DATA_PIPELINE_README.md` §6.4。

<details>
<summary>到达当前口径的三步（历史，展开看）</summary>

| 时点（UTC） | 合计 | 变动原因 |
|---|---:|---|
| 08-28 11:50 | 7,626,737 | 首次七源实测基线 |
| 08-29 01:45 | 7,637,419 | `trait_benchmark` blocklist 862→59,212 键（`trait` 34,872→31,515）；TRAIT 与 `tcr_papers_v2` 语料被重建（`tcr_papers` 667,405→681,444） |
| 08-29 05:30 | **7,794,375** | **`tcr_repertoire` valid 近重复搬迁**（train 1,971,794→2,128,750） |
| 08-31 | **7,794,375（行数不变）** | `tcr_papers_v2/train.csv` 第 3046 行 `cdr3b` 剥空位 `ASSKVAARVP-TLKLS`→`ASSKVAARVPTLKLS`；全量 21 个 split 仅此一格。见该目录 README「残基字母表」 |

**第三步的成因曾被误记为「语料被并行会话重建」，实为 `move_near_dup_eval_rows.py --apply`
把 valid/holdout 中与 train Lev≤1 的行移入 train**：`build_repertoire.py` 未重跑、
blocklist 未变动，这也是 `assert_corpus_fresh.py` 依然通过的原因。并发会话的泄漏审计在
新语料上给出 `HARD-REQUIREMENT hits = 0 -> PASS`（本源对 11 个基准全 0），与
`count_immune_mix.py` 测得的 kept=raw、剔 0 一致。

</details>

> 🔴 **不要用 `data/tcr_repertoire/dataset/build_report.json::split_counts` 查行数。**
> 它记的是构建期的 1,971,794 / 201,504 / 201,170，而近重复搬迁只改 CSV、不重写该报告。
> 行数以 CSV 与 `data/tcr_native/dataset/near_dup_eval_move_report.json` 为准；
> `build_report.json` 仍是**去污染**口径的权威来源（`PASS` / `decontam_mode` /
> `blocklist_provenance`）。

> ~~⚠️ **`tcr_papers` 的默认目录是 v1，不是 v2。**~~
> **已于 2026-08-29 修正：`TCR_PAPERS_DEFAULT_DIR` 现指向
> `data/tcr_papers_v2/dataset`。** 不传 `--tcr_papers_dir` 的脚本现在量到的就是
> v3 口径。改默认的原因正是本条警告描述的坑：所有临时统计、布局计数、泄漏审计
> 都在静默地量 v1，而 v3 训的是 v2，两者差 274k 行。
>
> 三个 job 配置都显式传目录，不受影响：`bert_immune` / `diffusion_immune`
> 钉 v1（保已跑 checkpoint 可复现），`diffusion_immune_v3` 钉 v2。
> `count_immune_mix.py` 仍支持尾随 `field=value` 覆盖，用于反过来复现 v1：
>
> ```bash
> python scripts/count_immune_mix.py train \
>   "oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire" \
>   tcr_papers_dir=/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_papers_v2/dataset
> ```

> ⚠️ **`tcr_repertoire` 于 2026-08-28 11:50 被重建过，本表已是重建后的口径。**
> 重建把 `t4_refbinder` blocklist 从 61,136 core 刷新到 62,893（并集 123,135 →
> 124,176），train 从 1,971,136 变为 **1,971,794**。**引用本源规模前请先看
> `data/tcr_repertoire/dataset/build_report.json` 的当前内容**，该语料近期在变。

### 语料新鲜度：`PASS: true` 不是充分证据

同日发现的事故：构建报告的 `PASS` 只表示「对**构建时那份** blocklist 干净」。
blocklist 是活文件会被重建，语料不会自动跟着重建，于是**过期语料与干净语料在报告上
完全同形**。`tcr_repertoire` 09:04 建库、其 T4 黑名单 09:52 被重建（67,013 → 68,846
键），语料未重跑而 `PASS` 仍为 true，实测 3 个 T4 参考 binder 留在 train 中。

结构性修复（已生效）：

- `build_repertoire.py` / `finalize_papers.py` 的报告新增
  **`blocklist_provenance`**：每个依赖 blocklist 的 path + mtime + 内容 sha1。
- 新增 `scripts/data/tcr_native/assert_corpus_fresh.py`：校验 `PASS` **并且**逐个比对
  blocklist 活文件 hash；无 provenance 的语料报 UNVERIFIED 而非静默放过。
- `train_jobs/protein_esmc_llada270m_diffusion_immune_v3.yml` 启动前调用该脚本
  （原来是只看 `PASS` 的内联断言）。

**规则：任何 blocklist 重建后，依赖它的语料必须重跑并重新校验。** 事故全过程见
`data/tcr_papers/EXPANSION_AUDIT_2026_08_28.md` §3.8。

补一半（2026-08-29）：**断言只能*发现*脱节，*挡住*后果的是加载期过滤。**
`tcr_repertoire` 原先没有加载期过滤（注释写"构建期已去污、无表位可作键，故不叠"），
所以那 3 条落在磁盘上就等于会进训练。已在 `build_immune_specs` 补上
`repertoire_core_exclusions`（`t4` ∪ `t2t3` **投影成裸 core** ∪ `ots_benchmark`——
本源无表位，配对键永远匹配不上，故必须投影；这是故意过挡）。
实测那 3 条现在全部被挡下。**双层不是冗余，它是唯一能兜住"黑名单重建了但语料没跟着
重建"的机制。**

### 黑名单自己也可能不是充分证据：`语料 ∩ benchmark` 的交集写法

同一类问题的第三个面，2026-08-29 发现。前两节问的是"报告绿了能不能信"，
这节问的是"黑名单本身够不够"。

`decontam_extra.py::decontaminate_trait` 原本先扫**语料**收集 CDR3β core，
再与 benchmark 求交集，把交集写成黑名单。于是黑名单是 `语料 ∩ benchmark`，
**只对构建时那份语料完备**——语料一重建，新增行的 core 从没进过候选集，
黑名单**结构上不可能**挡住它们。实测那 862 个键只覆盖 hard-requirement
保护集的 **4.2%**（277 / 6,570）：

| 基准 | 保护 core | 被旧黑名单覆盖 | 覆盖率 |
|---|---:|---:|---:|
| `NM2025_seen` | 4,254 | 36 | 0.8% |
| `NM2025_unseen` | 1,091 | 70 | 6.4% |
| `public_trackA` | 1,306 | 172 | 13.2% |

**也就是说 TRAIT 此前 0 命中是运气，不是过滤起了作用。** 17:16 UTC 那次语料重建
加了 600 行，其中 4 行直接落在保护 core 上（`ASSVGGISPLH`、`ASSYGGPEQF` →
`NM2025_unseen`；`ASSVGTGYEQY`、`ASSVGRNTEAF` → `public_trackA`，全是无表位的裸
CDR3b 行）。等 mtime 稳定 40 秒后复核确认是真泄漏，不是读到半写文件。

修复：改为 `簇级交集 ∪ binding_benchmark ∪ full_bank` = **59,212 键**（原 862）。
这不是新发明——`build_repertoire.py` / `finalize_papers.py` 早就直接用
`binding_benchmark | full_bank` 做与语料无关的精确黑名单，`decontaminate_trait`
是唯一的例外。簇级那一半仍只能语料派生（近重复是语料的属性而非 benchmark 的属性），
限制已写进 docstring。代价 trait 保留行 **−11.2%**；附带 T2/T3 四列
从 162/2,332/287/208 全部归零。原文件备份 `*.pre_union_bak`。

**规则：精确黑名单必须 benchmark 派生，不能写成 `语料 ∩ benchmark`。**
前者对任何语料都完备，后者只对一份快照完备。

验证工具：`scripts/data/dedup/audit_downstream_leakage.py`（5 源 × 11 基准矩阵，
把每行推过该源真实的 `row_to_record` 后再比对）。当前
**`HARD-REQUIREMENT hits = 0 -> PASS`**。读矩阵注意：非零格子不等于泄漏，
只有上表三个基准按裸 core 保护，其余按 `(core|epitope)` 配对键保护，
投影成裸 core 会按构造过报（`tcr_papers_v2` 约 7.7 万裸 core 命中 T4，
配对层命中为 **0**）。详见
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md) §3。

### valid/train 近重复：`split_disjoint_PASS` 也不是充分证据

同一类问题的第二个面：各源报告里的 `split_disjoint_PASS=true` 只保证三 split 的 CDR3β
**精确**互斥。编辑距离 1 的变体不算 overlap，于是 valid 里可以塞满 train 序列的单点突变
体，报告照样全绿。ckpt 是按 valid 的 `eval_loss` 选的，这直接影响选模。

**先说度量单位，这是这件事最容易做错的地方**：近重复必须按**该源的生成目标**来量，不是
一律按 CDR3。`ImmuneSourceSpec` 的 `roles` 决定哪些链是固定上下文、哪些是去噪目标：

- `tcr_native` / `tcr_papers` / `trait` / `tcr_repertoire`：epitope 和 MHC 是固定上下文，
  **CDR3 才是生成目标** → 按 CDR3β core 量。
- `oas` / `ots`：两条链 100% 都是生成目标（实测 `gen_res == res`）→ 按**全长配对 Fv** 量。
- `asd_antibody`：抗原是固定上下文（生成残基仅占 36%）→ 按 **heavy_fv + light_fv** 量。

按错单位会得出完全相反的结论。`ots` 若按 CDR3 量是 exact 27.25% / Lev≤1 69.90%，看着像
重大泄漏；但它按全长配对量是 exact 0.00% / Lev≤1 1.85%。差异不是切分失败，而是天然
repertoire 里同一条 CDR3β 本来就会与不同 α 链配对——而模型要生成的是整条链，CDR3 复现
不构成答案键。

2026-08-28 晚七源全测（valid 抽样 2,000–全量，train 全量为参考）：

| source | 度量单位 | exact | Lev≤1 | 处置 |
|---|---|---:|---:|---|
| `tcr_papers`(v2，v3 在用) | CDR3β core | — | **49.2%** | 已搬迁 |
| `tcr_papers`(v1) | CDR3β core | — | 46.8% | 已搬迁（当日早先） |
| `tcr_native` | CDR3β core | — | 40.9% | 已搬迁（当日早先） |
| `tcr_repertoire` | CDR3β core | 0.00% | **38.9%** | 已搬迁 |
| `trait` | CDR3β 全 junction | — | **19.1%** | 已搬迁 |
| `asd_antibody` | heavy_fv+light_fv | 0.00% | 4.20% | **无需处理** |
| `ots` | 全长配对 Fv | 0.00% | 1.85% | **无需处理** |
| `oas` | 全长配对 Fv | 0.00% | 0.45% | **无需处理** |

即：占训练**残基** 94% 的三个全长源（`oas`+`ots`+`asd_antibody`）在自己的生成目标上
几乎没有 valid/train 近重复，问题**集中在 CDR3 生成类的四个源**，且都已处理。

`tcr_papers_v2` 漏做的原因值得记：v1 当天早些时候做过这一步，但 v2 是
`finalize_papers.py --out-root data/tcr_papers_v2` 从零重建的，**不继承 v1 已清洗的
split**。「某个源已经处理过」不能推广到它的重建版本。

处理方式是**搬进 train 而非丢弃**（`scripts/data/tcr_native/move_near_dup_eval_rows.py`），
迭代到收敛——搬入 train 的行会成为新参考，单轮不够（v2 用 3 轮，trait 用 5 轮）。

**这不是答案键泄漏。** 所有已处理源中，被搬走的 eval 行里 `(CDR3β, epitope)` 组合在
train 中精确出现的都是 **0 行**。修的是选模信号的可信度，不是补泄漏窟窿。

连带后果：新旧 `eval_loss` 不可比（valid 变小、变难、构成变了），不要与上一轮的
0.2578 / 0.5638 横向对比。

**顺带纠正一个常被引用的过期数字**：「valid 里 73% 是 `tcr_pmhc_fulllength`，train 只
8%」描述的是「单一混合 eval + 前缀截断」时代。现在 `subsample_seed=0` 走 reservoir 抽样、
eval 分源各截 `max_eval_rows_per_source=2000` 行，实测 eval 构成是**七源近等权**（各
14.93%，`trait` 因只有 1,393 行占 10.40%），而非任何单源占 73%。逐源实测表与「等权 vs
按 train 比例加权哪个才是对的选模口径」的论证见
`downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md` §3c。

**record 与 residue 口径严重脱节，且当前无法调节**：`ImmuneSourceSpec.weight` 在
`ImmuneBioSeqDataset.__getitem__` 中被丢弃，混合比例纯由磁盘行数决定。
`tcr_repertoire` 占 25.85% 记录但仅 2.03% 残基（裸 CDR3β，均长 13）；
`asd_antibody` 反向，3.62% 记录吃掉 11.37% 残基（长抗原，均长 523）。

### 长度上限口径（2026-08-28 修正）

`DataArguments.max_length` / `max_protein_length` 的默认值已由 **512 改为 1024**，
与全部 `train_jobs/protein_esmc_*immune*.yml` 显式传参一致。基类
`dllm/utils/configs.py::DataArguments.max_length` 本来就是 1024，是两个 protein 入口
把它往下覆盖了。

过滤语义为 `_record_chain_lengths_ok`：每条链 ≤ `max_protein_length`，**且**
`sum(len(chain)) + 3*n_chains + 8 ≤ max_length`。第二个（总长）约束才是实际生效的那个。

**只有 `asd_antibody` 对该参数敏感**，因为它是 antigen+heavy+light 三链，抗原长度分位数
为 q0.25=395、q0.5=q0.75=q0.9=q0.95=**607**（超半数行共用同一条 607 aa 抗原，
来自 `buzz`/trastuzumab 突变库）：

| `max_length` | 抗原预算中位数 | `asd_antibody` 保留 |
|---:|---:|---:|
| 512 | ~268 aa | 159,331（24.3%） |
| 768 | ~524 aa | 295,583（34.8%） |
| **1024** | ~780 aa | **276,412**（32.5%，叠加去污染后） |

607 正好夹在 512 与 1024 的预算之间，故该参数是悬崖式的。其余六源在 512 与 1024 下
逐行相同。历史报告若未显式传 1024，其 `asd_antibody` 行数不可用——
`RETRAIN_PLAN.md` §7.2b/§7.8 初版即因此误记为 159,331，已修正为 276,412。

### 新增数据源（2026-08-28）

- **`data/tcr_repertoire/dataset`** — TcrDesign-2026 `pretrain/bCDR3_train.csv` 的无标签
  单链 CDR3β。读 40,308,610 行，池 40,257,598，train 上限截到 2,000,000、簇级去污染
  再剔 28,206 后为 **1,971,794**（valid 201,504 / holdout 201,170）。
  `build_report.json` 记录 `decontam_mode=exact+cluster_0.80_0.80`，blocklist 并集
  124,176 核心，`residual_blocked_in_train=0`，三 split 互斥。**唯一渲染为
  `tcr_single` 布局的源**，T4 Setting-A 无条件生成 benchmark 用该布局解码，此前训练
  覆盖为 0。详见 `data/tcr_repertoire/README.md`。
- **`data/tcr_papers_v2/dataset`** — v1 四源（`tcrt5` 295,316 / `tcrdiff` 100,590 /
  `gratcr_tep` 17,768 / `epidiff` 2,484）加 TcrDesign-2026 三层（`tcrdesign26_beta`
  98,536 / `tcrdesign26_paired` 18,383 / `tcrdesign26_pmhc` 163,843）；以上为
  `finalize_report.json::by_source`，即去重后跨 split 的 final rows，合计 696,920。
  split 为 train 668,331 / valid 14,449 / holdout 14,140，唯一 epitope 3,218，
  相对现役语料净新 epitope **+1,763**（v1 为 +881）。`decontam_mode
  =exact+cluster_0.80_0.80`，`residual_binding_exact_hits=0`、`residual_t4_hits=0`、
  `split_disjoint_PASS=true`。v3 通过把 `TCR_PAPERS_DIR` 指向该目录接入，
  **未新增 dataset token**，因此 `--dataset_args` 里仍写 `tcr_papers`。

### 新落盘但未接线的原始数据（2026-08-28）

由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_missing_tcr_sources.sh`
下载，该脚本可重跑（已校验文件跳过）。

| 路径 | bytes | 校验 |
|---|---:|---|
| `data/tcr/vdjdb-2026-06-03.zip` | 41,017,409 | 与 GitHub `content-length` 一致 |
| `data/ots_tcrlang/raw/TCRLang_Datasets.tar.gz` | 126,348,093 | MD5 `f2d5cbdcbc518c7f4b73f27515ecb37d` ✓ 官方 |
| `data/ots_tcrlang/raw/tcrlang-weights.tar.gz` | 166,045,148 | MD5 `8bbade5ba653096a490467cbc68b4034` ✓ 官方 |
| `data/ots_tcrlang/raw/OTS_CoherenceCode.tar.gz` | 86,914 | MD5 `53900bf6864a5f4a2cd07876698eba8c` ✓ 官方 |

**VDJdb 2026-06-03**：`vdjdb_full.txt` 由 139,745 行（2025-12-29）增至 192,754 行
（+53,009，+37.9%），已解包至 `data/tcr/vdjdb_2026_06_03/vdjdb-2026-06-03/`。发布包
结构变了：2026 版不再含 `*_scored.txt` / `*_broken.txt` / `_filtered`，只有 10 个核心
文件，因此 zip 从 72.8 MB 缩到 41 MB——**不代表数据变少**。
`scripts/data/tcr_native/ingest_papers.py` 中「+3 net-new epitopes for UniPMT /
VDJdb / McPAS / GLIPH combined」是针对旧版的判断，**已过期**，需用
`assess_candidate.py` 重评。

**Zenodo 11208211（OTS/TCRLang，CC-BY-4.0）**：解包至
`data/ots_tcrlang/TCRLang_Data/`，格式为每行 `<BETA_FV>|<ALPHA_FV>` 全长可变域。
train paired 1,361,284 / test paired 100,000 / eval paired 100,000；
train heavy 4,624,002 / light 4,585,975（其 "heavy" 指 β、"light" 指 α）。

> ⚠️ **该记录的官方 test/eval 不能作为本项目的 held-out benchmark。** 按全长 β+α 精确
> 配对比对 `data/ots_paired_clean/final/train.csv`（即 `OTS_DEFAULT_DIR`，2,102,715 行）：
> test 命中 98,221/100,000 = **98.2%**，eval 命中 98,304/100,000 = **98.3%**。两边同源于
> OTS 而本项目自行重切 split，把对方测试集切进了训练集。反向亦然：不能拿本项目模型
> 与 TCRLang 论文在该 test 上报告的数字比较。其 train（136 万）小于本项目 OTS
> （210 万）且同源，预计不增序列覆盖。可用价值在 `tcrlang-weights.tar.gz`
> （配对模型权重，可作 baseline）与 `OTS_CoherenceCode`（α/β V-gene / V-allele
> coherence 校验函数 + 两个测试 pkl）。

## AB/TCR canonical v2（2026-08-04）

下一版免疫受体数据的权威执行记录为
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`，
机器可读数据根为
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2`。
本轮只处理 antibody H/L、TCR alpha/beta、antibody-antigen 与
TCR-peptide/pMHC；不包含 MINT/general-PPI、nanobody/VHH/VNAR 或无 specificity
的 bulk TCR，也没有改变模型或启动训练。

- `bioseq.v2` TCR union：922,479 records，SHA256
  `41b74432ec71baafd9fd1600706cac774ceab0732dbf6e41ee32c9d07da52b0e`；
  其中严格 split core 为 41,489。
- `bioseq.v2` antibody union：470,949 records，SHA256
  `3c7b5d679bdd98d7571e16b3821b1e918da7b92d3333838b136f58fed7cc42f0`；
  antibody-antigen interaction core 为 308,267，含 property 后总 core 为
  328,248。
- canonical record 显式保留 sequence scope、observed/reconstructed、V(D)J/CDR/FR、
  assay value/unit/censor/direction、negative type、target mapping confidence、
  evidence tier、source row/hash、PMID/DOI、original split 与稳定 group ID。
- 九套 receptor/peptide/pMHC/study、antibody/antigen/joint-hard、property split
  均通过 disjoint audit；原始官方 split 只保留为 provenance。
- exact/near benchmark quarantine、cluster-disjoint split、train-only TCR
  negatives 与 immutable artifact manifests 已物化，当前
  `technical_data_export_ready=true`。core export build 为
  `ir2exp_f7a60484c7e3a20db6a2`，strict OAS/OTS pairing build 为
  `ir2pair_6a1a5b62752caccd2cd5`，候选 recipe 为
  `ir2recipe_1e3eac44551e88aedc6e`。
技术产物的 active real-record 规模为：

| plane | train | valid | test |
|---|---:|---:|---:|
| OAS strict H/L pairing | 1,468,754 | 15,295 | 14,800 |
| OTS strict alpha/beta pairing | 1,445,804 | 14,732 | 14,741 |
| antibody recognition | 146,998 | 8,167 | 8,166 |
| TCR recognition | 3,986 | 222 | 221 |
| antibody properties | 12,227 | 722 | 637 |
| total | 3,077,769 | 39,138 | 38,565 |

TCR recognition 另有 3,970 条 train-only synthetic negatives，未并入 real-record
total；所有 export 的 residual benchmark cluster match 为 0。
- `rights_review_complete=false`、`runtime_views_ready=false`、
  `sampling_weights_frozen=false`，因此 `export_ready=false`、
  `training_ready=false`、`training_started=false`。canonical JSONL 与旧 split 仍不能
  直接作为训练输入；训练只允许从候选 recipe 引用的 immutable export 开始，并须先
  关闭上述 gate。

## 当前训练数据单一入口（2026-07-23）— 已被取代

> 🔴 **2026-08-28：本节标题中的「当前」已失效。** 它描述的是 `bioseq_grammar_v1`
> Arrow 配方与 7L step389500 lineage；现役训练入口是
> `examples/llada/protein_pretrain_esmc.py` 直读 CSV 的七源 mix，见本文件顶部
> 「当前训练语料实测快照（2026-08-28）」。本节保留为历史 lineage 事实，不得据此
> 判断当前训练数据。

当前 7L step389500 真正使用的 7 源配方、实际 Arrow 行数、runtime
fixed/target 语义、source weight、重复采样强度、resume 数据流问题，以及相对
2026-07-21 canonical benchmark 的新去污染复核，统一见：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/TRAINING_DATA_CATALOG.md`

当前活跃 train 精确总数为 16,044,966；活跃源为
`oas,ots,nanobody,tcr_piste,tcr_pmhc_fulllength,ppi,neutralization`。
`tcr,mint_ppi,mint_actions` 虽已落盘，但没有进入该 checkpoint 的 recipe。

下一版 active training scope 已收敛为四类：
`antibody H/L`、`TCR α/β`、`antibody-antigen`、`TCR-epitope/pMHC`。
nanobody/VHH、MINT/STRING/general-PPI、当前无 antigen sequence 的
`neutralization`，以及无 specificity 的 bulk TCR 不进入下一版 recipe。旧 7 源
checkpoint 只作为 lineage 事实保留，不能与下一版目标配方混称。

重要口径更新：现有 downstream decontamination bank 生成于 2026-07-08，早于
2026-07-21 固定的 MINT official、NM2025 official 和 Public TCR Track-A
artifact。它不能继续作为“当前全部 benchmark 已去污染”的充分证据。当前 exact
复核已发现 Public Track-A 1,306 个唯一参考 CDR3β 中有 395 个命中活跃 TCR
训练源，MINT Gold test 与活跃 PPI 有 498 个 exact pair overlap；下一版训练前
必须重建并版本化 bank。

## MINT 五任务官方 notebook 数据（2026-07-21）

当前 MINT benchmark 的唯一数据根是
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`。
总 manifest 状态为 `validated`，固定 MINT commit 为
`06694b7606e2d00b76ec58daf5c7aecdaf7cd283`。旧根
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint`
是历史本地处理数据，不能与本次结果混用。

| 目录 | schema | 行数 |
|---|---|---:|
| `ppi/Intra1_seqs.csv` | `seq1,seq2,labels` | 163,192（train） |
| `ppi/Intra0_seqs.csv` | `seq1,seq2,labels` | 59,260（validation） |
| `ppi/Intra2_seqs.csv` | `seq1,seq2,labels` | 52,048（test） |
| `human-ppi/processed_data_{train,validation,test}.csv` | index, `sequence_1,sequence_2,target` | 26,319 / 234 / 180 |
| `yeast-ppi/processed_data_{train,validation,test}.csv` | index, `sequence_1,sequence_2,target` | 4,945 / 95 / 394 |
| `mutational-ppi/processed_data.csv` | index, `seq1,seq2,seq1_mut,seq2_mut,target` | 3,406 |
| `SKEMPI_v2/processed_data.csv` | index, four sequence columns, `target,complex,split_0..2` | 6,706 |

每个任务目录都包含 `manifest.json`、`execution_metadata.json` 和 `execution.log`；
manifest 记录原始输入绝对路径和 SHA256、notebook 路径和 SHA256、运行环境版本、输出
SHA256 与校验统计。完整的来源和论文差异解释见
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。

两个未公开论文 fold 的任务另有本地固定评测协议：

| 任务 | 本地 split artifact | 协议 | 防泄漏单元 |
|---|---|---|---|
| MutationalPPI | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/protocols/mutationalppi_pair_group10.csv` | `mutppi_pair_group10_sgkf_seed0_v1`；3,406 行、1,490 个 canonical unordered WT pair group、10 folds | 同一 WT pair group 不跨 fold；sidecar SHA256 `c7ec1c3621757fc2e08fed9f72531a42056a25808767de1ad394435827597296` |
| SKEMPI | `SKEMPI_v2/processed_data.csv` 的 `split_0..2` | `skempi_notebook_complex3_mt19937_v1`；6,706 行、343 complexes、3 folds | complex 不跨同一 outer fold 的 train/test，且三个 test complex 集互斥 |

这两项只定义 `[L]` 本地可重复评测，不能称为论文未公开 fold。新重跑 cache 还必须带
`localfixed-v1-l2048`，其含义是 run-level 最高 cap=2048；每条 metrics 记录实际
`max_sequence_tokens`。ESM-1b 因 absolute-position config
`max_position_embeddings=1026` 使用 1024；ProGen2-Large 因 config `n_positions=1024`
和固定 causal mask 也使用 1024；其余六个模型使用 2048。WT 与 mutant 的每条链
使用相同左侧窗口，避免未突变 partner 因随机裁剪产生伪差分。

## High-Level Inventory

Top-level data size:

| Path | Size | Current role |
|---|---:|---|
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed` | 27G | cleaned nanobody/VHH pretraining CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean` | 22G | cleaned paired antibody heavy/light CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_raw` | 19G | mixed nanobody raw sources |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw` | 18G | TCRdb2.0 bulk repertoire raw zips/metadata |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean` | 12G | cleaned paired TCR beta/alpha CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_raw` | 7.5G | OTS raw paired TCR CSV.gz |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_paired_raw` | 3.4G | OAS raw paired antibody CSV.gz |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr` | 903M | VDJdb, McPAS, MIRA, IEDB/PIRD-related TCR specificity resources |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2` | 664M | existing JSONL mix: PPI + TCR-epitope |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi` | 653M | STRING-style PPI Hugging Face Arrow dataset |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed` | 500M | existing JSONL mix capped to max chain length 512 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream` | 43M | downstream benchmark data and symlinks |

Main file extensions under the data root are `.gz`, `.fasta`, `.json`, `.csv`, `.zip`, `.pdb`, `.py`, `.txt`, `.npy`, `.tsv`, `.parquet`, `.jsonl`, and Hugging Face `.arrow`.

## Closest Existing Unified JSONL

Current `processed` and `processed_v2` are the closest existing multi-chain JSONL format.

Record shape:

```json
{
  "chains": ["SEQUENCE_A", "SEQUENCE_B"],
  "types": ["other", "other"],
  "targets": [0, 1],
  "source": "ppi"
}
```

Stats:

| Dataset | Train rows | Val rows | Sources | Notes |
|---|---:|---:|---|---|
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed` | 805,095 | 14,405 | ppi, vdjdb, mira, mcpas | safer for current model; max chain length 512 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2` | 803,591 | 14,377 | ppi, vdjdb, mira, mcpas | preserves PPI chains up to 32,000 aa; needs cropping/bucketing |

`processed_v2/train.jsonl` distribution:

| Source | Rows |
|---|---:|
| ppi | 639,866 |
| vdjdb | 91,122 |
| mira | 46,824 |
| mcpas | 25,779 |

Chain-count distribution:

| Number of chains | Rows |
|---:|---:|
| 1 | 13,162 |
| 2 | 719,311 |
| 3 | 71,118 |

Top type combinations:

| Types | Rows |
|---|---:|
| `["other", "other"]` | 639,866 |
| `["beta", "antigen"]` | 73,532 |
| `["alpha", "beta", "antigen"]` | 71,118 |
| `["beta"]` | 13,162 |
| `["alpha", "beta"]` | 5,913 |

Important limitation: this JSONL does not include the large cleaned OAS, OTS, nanobody, or TCRdb2.0 pools yet. It is not the full foundation-model pretraining corpus.

## Cleaned Paired Antibody: OAS

Final/current path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{train,valid,holdout}_oas_label.csv`

Rows:

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 2,486,443 | 2,486,442 |
| valid | 12,554 | 12,553 |
| holdout | 12,654 | 12,653 |

CSV schema:

```text
h_sequence, l_sequence, species, l_locus,
h_v_call, h_d_call, h_j_call, l_v_call, l_j_call,
source,
h_fwr1, h_cdr1, h_fwr2, h_cdr2, h_fwr3, h_cdr3, h_fwr4,
l_fwr1, l_cdr1, l_fwr2, l_cdr2, l_fwr3, l_cdr3, l_fwr4,
cleaned_h_sequence, cleaned_l_sequence,
H_cluster_id, L_cluster_id, ab_cluster_key, ab_cluster_id,
ab_cluster_id_counts, split, h_region_labels, l_region_labels
```

Semantics:

- `source=OAS`
- `cleaned_h_sequence` is the heavy chain sequence
- `cleaned_l_sequence` is the paired light-chain-side sequence; `l_locus` is usually K or L
- FR/CDR fields preserve region-level segmentation
- cluster fields support leakage-aware split/grouping

For BioSeq foundation, this should map to:

- `task_type="antibody"`
- `complex_type="<type_ab>"`
- `chains=[heavy, light]` after role-oriented ordering
- `chain_roles=["antibody_heavy", "antibody_light"]`
- `targets=[0,1]` by default
- `regions` and V/J metadata preserved but not necessarily tokenized in v1

## Cleaned Paired TCR: OTS

Final/current path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`

Rows:

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 2,102,716 | 2,102,715 |
| valid | 10,620 | 10,619 |
| holdout | 10,622 | 10,621 |

CSV schema is aligned with OAS:

```text
cleaned_chain1_seq, cleaned_chain2_seq,
chain1_cdr3, chain2_cdr3,
chain1_anarci_type, chain2_anarci_type,
chain1_FR1, chain2_FR1, chain1_CDR1, chain2_CDR1,
chain1_FR2, chain2_FR2, chain1_CDR2, chain2_CDR2,
chain1_FR3, chain2_FR3, chain1_CDR3, chain2_CDR3,
chain1_FR4, chain2_FR4,
species, data_type,
chain1_type, chain2_type,
chain1_v, chain1_j, chain2_v, chain2_j,
source_file,
chain1_cluster, chain2_cluster, pair_cluster, cluster_id, split
```

Semantics:

- `data_type=tcr`
- `chain*_type` is beta/alpha
- `chain*_anarci_type` is B/A
- V/J metadata and FR/CDR segmentation are available

For BioSeq foundation, this should map to:

- `task_type="tcr"`
- `complex_type="<type_tcr>"`
- `chains=[beta, alpha]` after role-oriented ordering
- `chain_roles=["tcr_beta", "tcr_alpha"]`
- `targets=[0,1]` by default

## Cleaned Nanobody/VHH

**训练用最终路径（去泄漏后，2026-07-08 起用这个）**：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean`

由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/clean_nanobody_training.py` 从 `step6_final` 产出。两处清洗：(1) 删除所有 `nbbench_*` 源行——它们是 NbBench 下游基准的样本，被 step0 解析器混进了训练集，留着即对下游 nanobody 评测直接泄漏；(2) 对 `cleaned_seq` 施加 [90,160] aa 长度窗（真实 VHH 约 110-130 aa，p1=97/p99=132，此窗保留 99.9% 真样本、只切离群）。**不做**序列相似度去重（由独立的跨测试集去重工具负责）。

| Split | step6_final rows | step7_clean rows | drop nbbench | drop len |
|---|---:|---:|---:|---:|
| train | 11,649,792 | 11,525,884 | 109,644 | 14,264 |
| valid | 58,862 | 58,311 | 479 | 72 |
| holdout | 58,981 | 58,336 | 577 | 68 |

报告：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean/clean_report.json`。

上游/原始路径（含 nbbench，勿直接训练）：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final`

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 11,649,793 | 11,649,792 |
| valid | 58,863 | 58,862 |
| holdout | 58,982 | 58,981 |

CSV schema:

```text
vhh_seq, source, cleaned_seq, anarci_chain_type,
FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4,
cluster_id, split
```

For BioSeq foundation, this should map to:

- `task_type="antibody"`
- `complex_type="<type_nb>"` or `<type_ab>` with `chain_roles=["nanobody_vhh"]`
- `chains=[cleaned_seq]`
- `targets=[0]`
- FR/CDR regions preserved

## TCR Specificity Resources

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr`

Representative formats:

- `vdjdb_full.txt`: TSV with `cdr3.alpha`, `v.alpha`, `j.alpha`, `cdr3.beta`, `v.beta`, `d.beta`, `j.beta`, `species`, `mhc.a`, `mhc.b`, `mhc.class`, `antigen.epitope`, antigen metadata, method metadata, tissue/donor metadata, and score.
- `McPAS-TCR.csv`: CSV with `CDR3.alpha.aa`, `CDR3.beta.aa`, species/category/pathology, antigen protein, `Epitope.peptide`, `MHC`, tissue/T cell type, TRAV/TRAJ/TRBV/TRBD/TRBJ, PubMed ID, and remarks.
- `MIRA/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv`: CSV with TCR beta bioidentity/nucleotide sequence, experiment, ORF coverage, peptide amino acids, and genome coordinates.
- PIRD-related code/reference files are present under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr/PIRD_alt`.

These are not in a single unified schema yet. Existing `processed_v2` already contains a subset converted from VDJdb/McPAS/MIRA into `chains/types/targets/source`.

For BioSeq foundation, these should map to:

- `task_type="tcr_pmhc"`
- `complex_type="<type_tcr_pmhc>"`
- `chains=[beta]`, `[alpha,beta]`, `[beta,peptide]`, or `[alpha,beta,peptide]` depending on availability
- future extension: add MHC chain or MHC allele as metadata/conditioning, not necessarily as sequence in v1
- `targets` should usually include receptor chains, while peptide/MHC may be fixed context depending on task

## TCRdb2.0 Bulk Repertoire Raw Data

Root:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`

Downloaded structure:

- 263 project zips under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/project_zips`
- 263 project metadata CSVs under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/metadata`
- 1 healthy reference zip under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/healthy`
- validation manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests`

Metadata CSV schema example:

```text
CellSource, CellType, Condition, Chain, SampleId, ExperimentId, ProjectId,
RunId, Species, Comment, Gender, Instrument, LibraryLayout,
LibrarySelection, LibraryStrategy, Length, Spots
```

Project zip CSV schema example:

```text
AASeq, cloneCount, cloneFraction, Vregion, Dregion, Jregion,
NNSeq, Length, RunId, Chain
```

Healthy reference CSV schema:

```text
AASeq, Vregion, Dregion, Jregion, cloneFraction, cloneCount
```

For BioSeq foundation, this should map to single-chain or beta/alpha repertoire records first:

- `task_type="tcr_repertoire"` or `task_type="tcr"`
- `complex_type="<type_tcr>"`
- `chains=[AASeq]`
- `chain_roles=["tcr_beta"]`, `["tcr_alpha"]`, or chain-specific role from `Chain`
- `targets=[0]`
- clone count/fraction and disease/source metadata preserved

This source should be capped or downsampled during mixture training so it does not drown paired-chain learning.

## PPI

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`

Hugging Face Arrow schema:

```text
IDs: string
score: float64
OrgA: string
OrgB: string
SeqA: string
SeqB: string
```

Splits:

| Split | Examples |
|---|---:|
| train | 645,692 |
| valid | 5,854 |
| test | 1,322 |

Current `processed` converts this to:

- `chains=[SeqA, SeqB]`
- `types=["other", "other"]`
- `targets=[0,1]`
- `source="ppi"`

For BioSeq foundation:

- `task_type="ppi"`
- `complex_type="<type_ppi>"`
- `chain_roles=["protein_a", "protein_b"]`
- keep `score`, organism IDs, and pair IDs as labels/metadata if needed

## Downstream Benchmark Data

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream`

Main formats:

- CDR infilling: JSON/JSONL-style records with full chain sequences plus `{cdr_mode}_seq` and `{cdr_mode}_pos`.
- TCR binding: VDJdb/McPAS/ATLAS/IEDB/DeepInsight/TCRDesign/Nature Methods style task files.
- FLAb/in-silico/comp-chain: CSV/FASTA-style task files, some paths are symlinks to older AirGen locations.
- Humanization: documented as incomplete.

These should be treated as evaluation/fine-tuning data, not first-pass pretraining mixture data.

## Current Code-Level Schemas

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` defines the intended `bioseq.v1` JSONL:

Required:

```text
chains, task_type, source
```

Preferred optional fields:

```text
chain_roles, targets, split, labels, regions, metadata, schema_version
```

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/datasets.py` currently streams only three clean CSV corpora for Ophiuchus-style training:

- OAS paired antibody
- OTS paired TCR
- nanobody/VHH

It normalizes rows into a minimal training record:

```python
{"chains": [...], "task_type": "...", "source": "...", "weight": ...}
```

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/data.py` collates multi-chain examples into:

```text
input_ids, labels, chain_ids, attention_mask, loss_mask, task_type_ids
```

The current collator does not yet emit:

- `position_ids` split into inner residue position and outer chain index
- explicit `target_chain_mask`
- fixed-context mask for antigen/MHC/context chains
- BioSeq foundation complex header token ids
- `chain_role_ids`

## Implication for Adapting Qwen

The model-side input should not be designed around raw CSV columns. The stable boundary should be a unified BioSeq JSONL/example object. Chain-level `targets` is only a coarse default; complex conditional generation must be resolved through a task view or `generation_spec` into token-level masks.

```json
{
  "schema_version": "bioseq.v1",
  "task_type": "tcr_pmhc",
  "complex_type": "type_tcr_pmhc",
  "chains": ["TRA_SEQUENCE", "TRB_SEQUENCE", "PEPTIDE_OR_ANTIGEN"],
  "chain_roles": ["tcr_alpha", "tcr_beta", "peptide"],
  "targets": [0],
  "generation_spec": {
    "name": "beta_epitope_to_alpha",
    "fixed": [
      {"chain": 1, "scope": "full_chain"},
      {"chain": 2, "scope": "full_chain"}
    ],
    "generate": [
      {"chain": 0, "scope": "full_chain"}
    ]
  },
  "regions": {
    "0": {"CDR3": "..."}
  },
  "metadata": {
    "species": "HomoSapiens",
    "v_gene": "...",
    "j_gene": "...",
    "mhc_allele": "..."
  }
}
```

For the first Qwen-derived diffusion model, the canonical tensor batch should be:

```text
input_ids             [B, L]
labels                [B, L]
attention_mask        [B, L]
visible_mask          [B, L]
diffusion_loss_mask   [B, L]
fixed_context_mask    [B, L]
diffusion_target_mask [B, L]
chain_ids             [B, L]
chain_role_ids        [B, L]
task_type_ids         [B]
position_ids_inner    [B, L]
position_ids_chain    [B, L]
```

The view sampler should support at least these target constructions:

- Chain completion: fixed heavy generates light, fixed light generates heavy, fixed beta+epitope generates alpha.
- Antibody-antigen receptor design: fixed antigen generates antibody heavy/light or nanobody VHH; fixed antigen plus one antibody chain generates the paired antibody chain.
- Antigen-conditioned CDR design: fixed antigen plus antibody/nanobody FR residues generates all CDR regions or one selected CDR.
- MHC-conditioned TCR-pMHC denoising: fixed MHC/HLA generates or denoises peptide plus available TCR alpha/beta chains.
- Peptide design: fixed TCR alpha/beta plus MHC/HLA generates peptide or epitope.
- TCR design: fixed peptide or epitope plus MHC/HLA generates TCR alpha/beta.
- pMHC-conditioned TCR CDR design: fixed peptide/epitope plus MHC/HLA and TCR FR residues generates all TCR CDR regions or one selected CDR.
- Region infilling: fixed antibody/TCR FR regions generate all CDR regions.
- Single-region infilling: fixed all other residues generate one selected CDR.
- Inverse region infilling: fixed six CDR regions generate FR regions.
- Conditional receptor generation: fixed antigen/peptide/MHC/PPI partner generates selected receptor chains.

`full_denoise` in the BioSeq foundation loader should be read as full denoising over eligible target chains, not all biological chains. Antigen, peptide, MHC, and HLA-like chains are fixed context by default. They are visible conditioning residues but should not be remasked or included in `diffusion_loss_mask`.

## Encoder Tokenizer Boundary

The BioSeq foundation loader under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` should keep the canonical biological record independent from any one encoder tokenizer. Tokenization is a collator/encoder concern.

Local tokenizer verification:

- ESM2 snapshots under `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D` use id 31 for `<null_1>`.
- ESMC snapshots under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` use id 31 for `|`, and mark `|` as an additional special token.
- Standard amino-acid token ids and `<mask>` id 32 match between the verified ESM2 and ESMC local tokenizers.

Implementation rule:

- Use the ESM2/MINT-compatible tokenizer for Ophiuchus-Ab and no-encoder MINT paths.
- Use the encoder's own local Hugging Face tokenizer when an ESMC encoder is active.
- Do not infer multi-chain interaction capability from the `|` token alone. The ESMC/ESMFold2 paper places explicit multi-chain complex modeling in ESMFold2, where each chain is encoded independently by frozen ESMC 6B and then fused through downstream pair/folding/diffusion modules.

The immediate gap is data normalization: large clean OAS/OTS/nanobody and TCRdb2.0 should be converted into the same `bioseq.v1` format before building a BioSeq foundation architecture around them.

---

## 训练↔下游测试去重（integrated data 版，2026-07-08）

**原因**：整合全部数据训一个统一版前，必须保证训练语料不泄漏进任一下游 test，否则 headline 全线虚高。通用蛋白 50% 阈值不适用于抗体/TCR 设计（CDR3 是克隆型主键、框架区高度保守），按类定键与阈值。

**工具**（`scripts/data/dedup/`，env `protenix_abtcr`；mmseqs 用 `flow` env 的 18 版）：
1. `build_downstream_banks.py` → `data/dedup/banks/`：把所有下游 **test/eval** 读成按生物类型的 bank（跳过 benchmark 自带 train/background split，避免过删）。规模：`ab_cdrh3=9,619`、`ab_heavy=106,806`、`ab_light=6,329`、`tcr_cdr3b=32,887`、`tcr_cdr3a=16,969`、`antigen=221`、`ppi_proteins=15,051`。覆盖 OAS-pairing / SAbDab-CDR-infill(H1/H2/H3) / FLAb / OTS-holdout / T4-gen / NbBench(12) / IRBench-PPI / MINT(6) / NM2025 / TCR-clustering·representation。归一化与 shard 侧一致（`records.normalize_sequence`：去空格+大写+J→L）。
2. `dedup_train_vs_downstream.py --source <s>` → `data/dedup/reports/<s>.json` + `blocklists/<s>.jsonl`：
   - **tcr**（ots/tcr/tcr_piste）：CDR3β 精确单连接（set 交）。
   - **antibody**（oas/nanobody/neutralization）：CDRH3 精确 + CDRH3 70% linclust 聚类 + 全长 heavy/VHH 95% linclust；grammar shard 无 CDR 区段的源（neutralization）仅全长。
   - **ppi**（ppi/mint_ppi/mint_actions）：全局 40% linclust。
   - 相似度统一用 `mmseqs easy-linclust`（test bank + train 并集聚类；train 落到含 test 成员的簇即判泄漏）——线性时间，可扩到千万级；早期用 `search -s 5.7` 在 250 万 OAS 上 >12min 未完，已弃。
3. `apply_blocklists.py --source <s> [--promote]`：按 blocklist row index 过滤 Arrow shard（迭代序与 extractor 一致，且**硬断言** `n_rows` 与 report 一致防错位）。默认产 `<s>/train_dedup`；`--promote` 时 `train→train_prededup`、`train_dedup→train`（原件留底、可回滚）。
4. `inventory_integrated.py [--write-manifest]`：盘点 8 源整合混合的 raw/deduped 行数、weight、采样占比（=weight/Σweight，与行数无关）。

**逐源泄漏结果**（train 行数 / 泄漏行 / 比例）：

| source | 域 | 键/阈值 | rows | leaked | frac |
|---|---|---|---|---|---|
| oas | ab | CDRH3 精确+70% / 全长95% | 2,486,442 | 1,684 | 0.07% |
| ots | tcr | CDR3β 单连接 | 2,102,715 | 17,301 | 0.82% |
| nanobody | ab | CDRH3 精确+70% / 全长95% | 11,525,884 | 603,398 | 5.24% |
| tcr (processed_v2) | tcr | CDR3β 单连接 | 163,725 | 18,215 | **11.13%** |
| tcr_piste | tcr | CDR3β 单连接 | 284,144 | 63,117 | **22.21%** |
| ppi (STRING 90/90) | ppi | 全局40% | 319,429 | 57,528 | **18.01%** |
| mint_actions | ppi | 全局40% | 9,237,455 | 133,494 | 1.45% |
| neutralization | ab | 全长95%（无CDR区段）| 12,346 | 872 | 7.06% |
| mint_ppi | ppi | 全局40% | (重建中) | — | pending |

结论：TCR 源（tcr/tcr_piste）与 PPI(STRING 90/90) 与基准重叠最重（同源公库），必须去重后再入整合版；抗体/nanobody 泄漏比例低但绝对量不小（nanobody 60 万行）。

**mint_ppi 说明**：`rebuild_mint_training_shards.sh`（v12 binding 重建）完成后再跑其去重（40% 全局），并入整合 manifest。2026-07-08 已完成 promote（81,717,793 行）。

**train↔valid 去重（2026-07-09）**：整合训练提交前，用 `scripts/data/dedup/check_valid_in_train.py` 检查各源 `valid` 记录是否出现在 `train`（whole-record sorted-chain key）。8 源中仅 `tcr_piste` 命中 7 行（0.010% valid keys），经 `apply_validleak.py --promote` 从 train 删除；其余 7 源零重叠。`neutralization` 无 valid shard。工具与 blocklist 落 `data/dedup/reports/valid_in_train_<src>.json`、`data/dedup/blocklists/validleak_<src>.jsonl`。

**SAbDab2 说明（2026-08-03 校正）**：早先“归档无 `abag_split.csv`”的判断来自
一个不完整/损坏的本地下载，已作废。经官方 MD5
`0dbb4cc499e9eb77f14008b232f2c38c` 验证的 Zenodo 20083995 完整
`splits.tar.gz`（876,381,859 bytes）同时包含 `ab_split.csv`、
`ab_split_sd.csv`、`abag_split.csv` 与 `abag_split_sd.csv`。v2 adapter 只读取
paired VH/VL 的 `abag_split.csv`，排除 VHH/VNAR，并只接受 resolved
protein/peptide antigen；得到 6,412 个 antigen-component records，其中 3,363 个
single-polymer core、3,049 个 multi-component aux。官方 `ab_ag_split` 与 cluster
只保留为 provenance；本轮另建 antibody/antigen/joint-hard disjoint split。

**FLAb/AbRank 说明**：本地
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/flab/FLAb/data/binding/AbRank_dataset.csv.zip`
有 342,356 rows。严格 H/L/Ag sequence QC 后 149,782 rows 合法；76,515 rows 同时
满足当前单链 ≤1024，75,483 rows 另有 `fitness`。192,559 个 RBD-escape row 的
`Ag_seq` 是 mutation expression 而不是序列；AlphaSeq/AbCoV 的 full-spike context
大多超过 1024。不能把这些字段当氨基酸或静默截断。可用子集还需规范
affinity/IC50/censor，并按 antibody cluster 与 antigen cluster 的连通分量划分。

**TCR specificity merge 说明**：PISTE、TDC、TEIM 和 legacy
VDJdb/MIRA/McPAS 的 positive `(CDR3β,epitope)` 简单相加为 224,488，exact union
只有 142,456。TEIM 与 legacy exact overlap 40,912。PISTE random-train 若保留
`HLA_type`，284,144 个 triplet 无标签冲突；忽略 HLA 后只有 209,632 个 pair，
其中 13,901 个出现正负冲突。canonical schema 必须保留 HLA sequence scope、
assay、donor 和 source provenance，不能按无 HLA pair 多数投票。

---

## 全长 TCR-pMHC 五实体重建（2026-07-09，ARCH_AUDIT gap2 / FUTURE D1）

**原因**：旧 TCR 源是 CDR3 片段、无全长可变域、无 MHC+B2M，grammar 设计的五实体布局 `<prots> MHC . B2M <protd> <binding> <prots> <pep> PEP <protd> <binding> <prots> <tcr> α . β <protd>` 从未落地。

**产物**：`data/tcr_pmhc_fulllength/`（`scripts/data/build_fulllength_tcr_pmhc.py`，env `protenix_abtcr` + `PATH` 含 Stitchr/thimble）：
- 输入：VDJdb `data/tcr/vdjdb_full.txt` + McPAS `data/tcr/McPAS-TCR.csv`（human、MHC class I）。
- 全长 α/β：Stitchr/thimble（HUMAN IMGT ref，`stitchrdl -s human`）从 V/J 基因 + CDR3 拼全长可变+恒定域。
- 全长 MHC-I 重链：IMGT/HLA `imgt_hla/hla_prot.fasta`（45,762 等位基因，`HLAResolver` 精确→2-field→gene 级回退）；B2M = 成熟人 B2M（UniProt P61769 去信号肽，99 aa）。
- schema：`bioseq.v1`，`chains=[mhc, b2m, peptide, tcr_alpha, tcr_beta]`，`chain_roles=[mhc, mhc, peptide, tcr_alpha, tcr_beta]`（MHC+B2M 同 role `mhc` → 渲染成一个 `<prots> MHC . B2M <protd>` 块），`targets=[3,4]`，`labels.relation="binding"`（curated binder）。

**统计**（`build_stats.json`）：human MHCI 解析 134,526 → 配对可拼 79,040 → thimble OK 78,740 → **下游 CDR3β 去重去掉 15,324** → 唯一 59,093 → train **57,913** / valid 590 / holdout 590。

**shard**：`data/bioseq_grammar_v1/tcr_pmhc_fulllength/{train,valid}`（`build_bioseq_grammar_v1.py --sources tcr_pmhc_fulllength`，新增 `iter_tcr_pmhc_fulllength` reader 保留 roles + relation）。渲染验证：五实体骨架正确、2× `<binding>`、589 扩散目标 token（仅 α/β 残基，pMHC 固定上下文）、valid/train/holdout 整记录 0 重叠。

**待办（B4）**：并入整合 manifest + 权重后重训。数据布局/方法同步见 `ARCH_AUDIT.md`、`FUTURE_EXPERIMENTS.md` D1。

---

## T1 original-model 结果来源 schema（2026-07-21）

Canonical external-baseline 表位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`，一行对应一个 Nature Methods 2025 Supplementary Table 4 original model。关键字段：

- `selected_source` / `evidence_type`：`official_code_rerun`、`paper_reported` 或 `unavailable`。
- `result_label`：本地完整复现为“本地复现：官方 checkpoint + 官方推理代码 + original.zip”；论文 fallback 必须精确写“论文值，未本地复现”。
- `reproduced_locally`, `retrained`, `checkpoint_paths`, `checkpoint_available`, `checkpoint_aliases_verified`, `official_runner_available`, `local_metadata_status`：记录推理时实际打开的全部模型 artifact；只有带 `embedded_original_protocol`、路径与 catalog 逐项一致且 runtime copy 与 bundle alias 哈希一致的产物可进入本地选择，旧的无 metadata 产物视为 `not_complete`。
- `test_artifact` / `test_artifact_sha256`：固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip` 及 sha256 `906e84ebd4b071d7cb9eec04294f6bfaeb7f967dad0008fd3a758910afef13d9`。
- `test_manifest`：固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_binding_nm2025/original_test_manifest.json`；每个本地 `metrics.json` 的 `normalized_test` / `normalized_test_sha256` 必须与该 manifest 对应，证明实际评分 CSV 由当前 `original.zip` 展开而来。
- `seen_*` / `unseen_*` 是最终选中值；`local_*` 与 `paper_*` 保留两套原始来源供 audit，禁止把两套值无标记混合。
- `fallback_reason`, `citation`, `source_location` 记录不可运行原因和论文位置。论文未报告的格（当前 SETE unseen）保持空值。

每个新官方 rerun 的 `metrics.json` 还必须包含 `baseline_protocol`，其中 `train_csv_loaded=false`、`retrained=false`、实际 `wrapper` / `variant_tag`、runtime `checkpoint_paths` 和 original.zip checksum。wrapper 未披露路径、路径与 catalog 不一致或缺少任一 artifact 时均判失败；失败写入同一模型 track 下的 `original_run_status.json`，不生成伪 `predictions.csv`。

## T1 retrained artifact 数据与输出 schema（2026-07-21）

- 原始数据固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/nat_methods_tcr_benchmark/retrain.zip`，MD5 `5cf77befd7a07e0cb359540f050fb7b4`；权重固定为同目录 `Retraining_model.zip`，MD5 `aa6ea6c175738675692848d036c2244c`。校验缓存为 `retrained_artifact_integrity.json`，但缓存仅在 size/mtime/expected MD5 三者稳定时可复用。
- CDR3β-only member 以乱码不可依赖的父目录 + 稳定后缀解析：fold seen test=`_only_seen/<neg>/<fold>_1_1test.csv`；seen independent=`_only_seen/<neg>/1_1_1independent_test.csv`；unseen independent=`_only_unseen/<neg>/1_1_1independent_test.csv`；TCR-H fold preprocessing train=`_only_seen/<neg>/<fold>_1_1train.csv`。必需列均为 `Epitope,CDR3B,Affinity`。
- 输出根为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained/<tag>/cdr3b/AS/fold_<N>/<eval_set>/`。`predictions.csv` 固定列 `peptide,cdr3b,label,score`；`metrics.json` 保存 `n_input,n_scored,n_unscored_by_official_protocol`、precrec AUPRC、sklearn AP diagnostic，以及 archive/nested archive/checkpoint/test/train（若有）SHA256。
- 每模型五折汇总位于 `<tag>/cdr3b/AS/summary.json`；跨模型来源表为根目录 `comparison_AS.csv`/`.json`。comparison 不覆盖本地值，而是同时保留 `local_*`,`paper_*`,`delta_*`,`local_matches_paper_4dp`,`selected_source`,`selected_result_label`。当前 8 模型×3 eval sets=24 行，其中 6 行满足 AUROC/AUPRC 双指标四位小数一致。
- `train_csv_loaded` 通常为 false；TCR-H 是唯一当前例外，值为 true 且 `train_csv_use=inference_time_feature_selection_only`,`training_performed=false`。其 feature cache 必须绑定 train SHA 与 NumPy/Pandas/peptides/sklearn/SciPy 版本，且保留列数必须等于 checkpoint `n_features_in_=130`。
- 默认要求 `n_scored==n_input`。只有 spec 明确声明官方 drop-remainder 时可少行；当前仅 ERGO-AE 为 batch 50 截尾（unseen `3150/3162`），metadata 必须写差额。ERGO-lstm 虽同属 ERGO，但官方 LSTM batching 保留 partial batch，不得误套截尾规则。
- Ours frozen-head runner 固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_ours.py`。五个 official AS train 行数依次为 `150634/150832/151008/151150/151264`；seen test 为 `30670/30660/30384/30282/30268`，共享 seen-independent=`5882`，共享 unseen-independent=`3162`。所有 fold/split 的 `(Epitope,CDR3B)` 并集为 `462883`，按字典序固定行号并以 SHA256 `24f7de5f97346de0962ced427f7cbd8d0e293211b14219f00462f48dae22283c` 绑定 frozen-feature mmap。
- 每一 fold 的 official train 内 `(Epitope,CDR3B)` 都是唯一且无标签冲突。对该 fold 的三套 test，`shared_unique_clonotypes=0`、`shared_unique_exact_pairs=0`；unseen-independent 另有 `shared_unique_epitopes=0`。因此之前旧本地构造 `train.csv` 对 unseen 出现的 `1 epitope / 41 clonotypes / 30 exact pairs` 不属于本次 official fold-train 协议，也不能带入本次微调。逐 fold 真值写入 `<ours-tag>/cdr3b/AS/protocol_audit.json`。
- Ours feature cache 位于 `outputs/tcr_binding_nm2025_retrained/_feature_cache/<tag>/AS/`：`pairs.csv` 保存行号，`features.npy` 为 `462883 × 960 float32`，`feature_manifest.json` 绑定 checkpoint/data/pair SHA、`post_llada_final_hidden_state` 与 `global_mean_over_all_joint_record_residue_tokens`；中断时由 `features.partial.npy + feature_state.json::next_index` 恢复。每 fold 的 `head.pt` 同时保存 train-only mean/std、MLP state、fold seed、train member SHA 和 checkpoint SHA。
- Ours 五折完成后在 `<ours-tag>/cdr3b/AS/` 生成两张长表：`ranking_local_release_AS.csv` 是相同 release 数据上的主要排名；`ranking_paper_reference_AS.csv` 是 Ours 本地值与论文 baseline 值的 mixed-source 辅助排名，字段 `mixed_source=true`。ERGO-AE 的官方 batch-50 tail-drop 在三套 eval 的主要排名 `n_note` 中均显式标注：seen-test 按 fold 为 `30650/30670,30650/30660,30350/30384,30250/30282,30250/30268`，seen-independent=`5850/5882`，unseen-independent=`3150/3162`。
- 同目录的 `comparison_with_ours_local_release_AS.{csv,md}` 与 `comparison_with_ours_paper_reference_AS.{csv,md}` 是便于直接查看的 9-row 宽表：每个模型一行，每个 eval×metric 保存 numeric `mean,std_sample,rank`；CSV 另保留逐 split `n_note`、`result_source`、`comparison_scope` 与 `mixed_source`。baseline 保持 catalog 顺序，Ours 始终置于最后一行。
- 当前 seen-only 展示另物化为 `comparison_with_ours_seen_test_local_release_AS.{csv,md}` 与 `comparison_with_ours_seen_independent_local_release_AS.{csv,md}`。两者均不含 unseen 列；8 个 baseline 按 `auprc_mean` 降序（再以 AUROC/method 稳定破同分），Ours 不参与展示排序并强制为第 9 行。CSV 固定列含 `display_order,display_rule,tag,method,auroc_mean,auroc_std_sample,auroc_rank,auprc_mean,auprc_std_sample,auprc_rank,result_source,n_note`。

## Public CDR3β Track-A format audit（2026-07-21）

- Canonical prepared data root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_beta_public_benchmark`.
- `targets.csv` has one row per pMHC with `target_id,source_target_id,peptide,mhc_allele,mhc_pseudosequence,n_references,reference_source,dataset,year`; `references.csv` has one target-specific `cdr3b_reference` per row.
- Source references come only from `benchmark_data_w_preds.csv::reference_translations`. `tcrt5_translations`, `gratcr_translations`, `er_translations`, and greedy prediction columns are explicitly excluded. Audited cardinality is 14 targets / 1,312 distinct references; RVR has 895.
- Canonical generation columns are `model,target_id,peptide,mhc_allele,rank,cdr3b,raw_score`; provenance columns are `generation_mode,source,run_id`. Final `generations.csv` has 56,000 rows, 56 model-target blocks, exactly 1,000 ranks per block. `raw_score` is model-specific provenance: for TCRT5 it is cumulative generated-token transition log-likelihood and determines the paper-compatible rank; for BioSeq it is empty because the sampler exposes no candidate score, and BioSeq rank is seeded emission order. Neither value may be interpreted as a common score across models. BioSeq rows use `run_id=full14x1000_step117000_seed42_iter32` and `source=local_bioseq_checkpoint_runtime`.
- Canonical result root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results`, containing `generations.csv`, `per_target_metrics.csv`, `summary_metrics.csv`, `cross_epitope_jaccard.csv`, `reproduction_status.md`, `tcrt5_paper_consistency.csv`, `tcrt5_paper_consistency.md`, BioSeq-focused `bioseq_step117000_report.md`, `bioseq_step117000_generations.csv`, `bioseq_step117000_per_target_metrics.csv`, `bioseq_step117000_top_recoveries.csv`, `bioseq_step117000_giana_hits.csv`, and per-block GIANA inputs/outputs/logs. `generations.pre_tcrt5_paper_rank.csv` is the immutable pre-correction backup whose TCRT5 scores used length-normalized beam values.
- Raw candidate strings are preserved apart from whitespace/case normalization. No anchor repair is permitted; absent rows remain missing candidates in the requested denominator.
- TcrDesign qualitative paired output is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv` with required columns `target_id,epitope,generated_cdr3b,generated_cdr3a,beta_rank,alpha_rank,raw_beta_score,raw_alpha_score` and provenance columns `mhc_allele,generation_mode,source,run_id`.
- The completed paired file has 140,000 rows: 14 targets × 1,000 generated-beta ranks × 10 alpha ranks. Every `(target_id,beta_rank,generated_cdr3b)` maps back exactly to a `model=TcrDesign,generation_mode=de_novo` row in canonical `generations.csv`; 0 conditions are missing or mismatched. Every beta condition has contiguous `alpha_rank=1..10`.
- `raw_alpha_score` is empty because the official beam-search caller does not return scores. Empty is distinct from zero and must not be imputed. The file contains no reference-alpha columns and is not a quantitative alpha benchmark.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_evaluation_audit.json` records `alpha_generation_available=true`, `official_alpha_evaluation_available=false`, `included_in_quantitative_benchmark=false`, and the audited official code/input paths.

## BioSeq step189000 single-target diagnostic format（2026-07-22）

- Output root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/checkpoint_comparison/bioseq_step189000_rvr_epitope_only`. This root is intentionally outside canonical `results/`.
- `generations.csv` contains exactly 1,000 rows and one `(model,target_id)` block: `model=BioSeq-7L-step189000`, `target_id=RVRAYTYSK_HLA-A*03:01`, contiguous ranks `1..1000`, `generation_mode=de_novo_epitope_conditioned`, `source=local_bioseq_checkpoint_runtime`, and `run_id=rvr1000_step189000_seed42_iter32_epitope_only`. `raw_score` is empty by design.
- The raw candidate schema and evaluator are identical to canonical Track A. The focused additions are `top_recoveries.csv`, `invalid_generations.csv`, `rvr_comparison.csv`, and `report.md`; the standard `per_target_metrics.csv`, `summary_metrics.csv`, GIANA files, status, and reproduction report are retained.
- There are 998 legal raw candidates and two illegal candidates; no repair is applied. Since this artifact contains one target, `cross_epitope_jaccard.csv` has no target pair and no defined mean. The separately reported step117000-vs-step189000 Jaccard compares checkpoint candidate sets for the same target and is not a cross-epitope metric.

## TCRT5 full-eval output format（2026-07-23）

- Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval`. Top-level files are `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/paper_correspondence.csv`, and the persistent 95,871-sequence `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/pgen_cache.csv`.
- Author main root `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/author_release/main_top20_k100` contains: `per_target` 240 rows, `summary` 12, `jaccard` 2,280, `positional_entropy` 5,868, `kmer_jsd` 2,640, `length_distribution` 1,776, `polyspecificity_per_target` 240, `polyspecificity_summary` 12, `crosscheck` 11, `paper_main_comparison` 84, `pgen` 42, `pgen_aggregation_sensitivity` 8, and `olga_protocol_audit` 8.
- Author sparse root `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/author_release/sparse13_plus_rvr_k1000` contains: `per_target` 56 rows, `summary` 4, `jaccard` 364, `positional_entropy` 1,053, `kmer_jsd` 616, `length_distribution` 468, `polyspecificity_per_target` 56, and `polyspecificity_summary` 4.
- Current root `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/current_track_a` contains: `per_target` 56 rows, `summary` 4, `jaccard` 364, `positional_entropy` 1,114, `kmer_jsd` 616, `length_distribution` 579, `polyspecificity_per_target` 56, `polyspecificity_summary` 4, and `pgen` 60.
- Manifests are `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/author_release/main_top20_k100/manifest.json`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/author_release/sparse13_plus_rvr_k1000/manifest.json`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/current_track_a/manifest.json`. Each records protocol, UTC creation time, absolute source paths, source existence, Pgen status, and every table's absolute path/row count/columns. CSV names are the dictionary keys plus `.csv`.
- `per_target.csv` retains `model,target_id,protocol,k_requested,n_generated,n_unique_generated,n_references`, core P/R/F1/edit/recovery/Char-BLEU fields, rank evidence, exact ranks and cutoff-specific occurrence/unique/Hit@K fields. Empty ranks remain null; semicolon-separated exact ranks are 1-based.
- `summary.csv` separates `map_prefix` from `map_hit_rank_diagnostic`, and includes `map_rank_is_paper_compatible` plus `map_evidence`. The legacy-named `map_prefix_ordered_proxy` remains for paper-release comparison, but must be interpreted through the evidence columns.
- Entropy columns specify `gap_inclusive` versus `residue_only` and `_nats`; k-mer rows specify `k` and `js_divergence_nats`. Jaccard stores both similarity and dissimilarity. Pgen tables state population, counts, positive fraction and positive log10 moments; sensitivity rows additionally state aggregation population and log10 window.
