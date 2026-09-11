# Data Format Audit

Date: 2026-09-12

Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`

How to run preprocess:
[`dllm/pipelines/immune_llada/README.md`](dllm/pipelines/immune_llada/README.md).
Design rationale / risk register:
[`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md).
Raw-corpus decontam accidents (PASS is not enough; `语料 ∩ benchmark`; key spaces):
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).
Long-lived rules: [`PROJ_GUIDE.md`](PROJ_GUIDE.md).

## Current immune LLaDA data contract（2026-09-12）

The only current immune data implementation is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada` and the formal
training entry is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
The runtime contract is:

```text
raw source files
  -> source adapter
  -> canonical BioSeqRecord
  -> offline deterministic filters
  -> prepared semantic JSONL shards + manifest
  -> training-time grammar rendering
  -> padding/mask/tensor assembly
  -> per-chain encoder input reconstruction
  -> diffusion/MLM corruption and loss
```

Raw CSV/JSONL parsing, source conversion, receptor completion (when configured),
blacklist/decontamination, deterministic validity and length rejection, and bad-row
replacement are offline operations. The prepared loader builds byte-offset indexes and
decodes semantic rows; it does not create or consume a model-ready token cache. Grammar
encoding, padding, encoder reconstruction, and stochastic masking remain runtime
operations.

Current prepared roots: **v4 published**, **v5 published**. v3 /
`immune_v3_heterotypic` still exist on disk and some checkpoints/eval jobs still
point at them; they are not the current training default. See the pipeline README
version table. Counts: plan §4.1 (v4) / §4.2 (v5).

### Shared layout (v4 and v5)

Seven-source mix: `oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`.
Before completing listed sources, the offline pipeline scans complete TCR rows from
the selected `ots` / `tcr_native` / `tcr_papers` sources, applies the same
adapter/filter policy, and stream-counts FR/CDR lengths. The frozen profile, seed,
source policy, observation counts, fallback regions, digest, and (once written)
`completion_sources` are recorded in `dataset_manifest.json`.

Receptor completion (when a source is in `PreprocessConfig.completion_sources`):

- Chains are assembled **beta-first**. A real full-length Fv is kept verbatim with
  no invented region annotation; only missing/fragmentary chains are synthesized.
- Missing regions are `X` at profile-sampled lengths. Synthetic `X` is visible
  input but excluded from diffusion eligibility and loss (`synthetic_residue_mask`).
  The original CDR3β core remains the decontamination/benchmark identifier.
- Junction vs core is decided from provenance (`fv_source` / `provenance` /
  `schema_version`), **never** from the first/last character: an anchor-free core
  can legitimately begin with `C`. `trait` is IMGT junction form (`C...F/W`,
  anchors peeled into FR3/FR4); `tcr_native` / `tcr_papers` are anchor-free core.
- Branching is on the actual `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` column
  contents, **not** the `sequence_scope` label (351-row `tcr_native` counterexample
  and the grammar-ambiguity rationale: plan §2.5). Full-length Fv exists only in
  `tcr_native`; `tcr_papers` is CDR3-scope.
- Default `completion_sources` is `("tcr_repertoire",)`. Withholding a source
  from the list is what disables completion for it.

all-X placeholders (upstream Zenodo 14545852 / TcrDesign: `'X'` means missing):

- all-X **epitope** rows are dropped by the named offline filter
  `quality.blank_epitope` (applies to `trait` / `tcr_native` / `tcr_papers`) so
  the drop count lands in `filter_report.json`. Not a silent adapter `None`.
- all-X **MHC** chains are stripped; the record degrades to the `tcr_peptide`
  layout. `task_type` is derived from whether the MHC chain survived.

Circular-profile guard: `add_tcr_region_lengths` skips regions listed in a
chain's `synthetic_regions` metadata. The profile's learning sources are
`['ots','tcr_native','tcr_papers']`; once those sources can themselves be
completed, an unguarded profile would learn its own synthesized lengths.

`profiles.py::sample_region_lengths` sorts support lengths **numerically**.
`atomic_json_dump` writes the manifest with `sort_keys=True`, so a reloaded
profile iterates lexicographically (`'10'` before `'2'`). Weights were always
looked up by length (the distribution was correct); a given seed drew different
lengths, so a published dataset could not be regenerated from its own manifest.
Only regions whose support spans single and double digits are affected (CDR3);
framework regions are all two-digit and coincidentally escaped.

Relation / grammar (unchanged vs v4): only explicit `binding` / `nonbinding`
labels are relation targets. For pMHC, MHC→peptide presentation `<binding>` is
fixed; peptide→TCR recognition is the supervised target. Generated-block
skeleton tokens (`<prots>`, type marker, `<protd>`) are eligible with the
relation target. Unconditional layouts emit a fixed null-context prefix
`<prots> <null> <protd> <unknown>` at render time (not stored in JSONL).
Audited 2026-09-12: that prefix already applies to exactly the unconditional
layouts (`antibody_pair` / `tcr_pair` / `tcr_single` / `nanobody`) and not to
conditioned ones; `grammar.py` unchanged. Open item: unused `single_entity`
fallback has no prefix.

### v4 published counts

Root: `data/prepared/immune_v4_beta_relation` from
`configs/data/immune_v4_beta_relation.yaml` (completion = `tcr_repertoire` only).
Authoritative per-source table: plan §4.1. Headline: train **7,665,574** /
valid **102,719**; raw 8,441,615 → kept 7,768,293. Formal v4 job:
`train_jobs/protein_esmc_llada270m_diffusion_immune_v4_4gpu.yml`.

v4 `tcr_repertoire` is byte-reproducible from its own manifest after the
numeric-sort fix: 3,000 raw-CSV rows vs the on-disk shard, **0 mismatches**.

### v5 published

Root: `data/prepared/immune_v5_receptor_completion` from
`configs/data/immune_v5_receptor_completion.yaml`
(`completion_sources: [trait, tcr_native, tcr_papers, tcr_repertoire]`).
Authoritative per-source table, v4 Δ, supervised-token shares, max rendered
tokens, profile digest, and invariants: plan §4.2. Headline: train
**7,626,885** / valid **102,308** (13G; valid is the sum of the seven audited
source rows). Do not copy that table here.

### Manifest `filter_names` and `filter_report` counters (2026-09-12)

Code is fixed (`95c04a96`). **Published v5 artifacts were not rewritten** — a
future preprocess writes the new fields. Do not read "code fixed" as "the
13G v5 manifest / `filter_report` already carry them".

**`filter_names` was actively wrong, not merely incomplete.** The field was
hardcoded to `BLOCKLIST_NAMES` plus an optional `quality.homotypic_pair` —
i.e. the list of blocklist **config keys**, not the filters that ran. It
therefore omitted `quality.blank_epitope`, both budget filters
(`budget.max_protein_length`, `budget.max_length`), and
`decontam.repertoire_core_projection`, and it **listed
`asd_nanobody_benchmark`, a source that was never processed**. Manifest
filter provenance was wrong, not just missing one name.

**New semantics:** dataset-level `filter_names` is the **union**, in
first-seen order, of the `RecordFilter` names that `build_filters` actually
constructed for the processed sources. Filters remain per-source
(`quality.blank_epitope` exists only for `trait` / `tcr_native` /
`tcr_papers`), so the union must **not** be read as "every source ran every
name"; each `filter_report.json` source entry also carries its own
`filter_names`. Disabled/empty blocklists never become filters and are
omitted; a filter that ran and dropped zero rows is still listed.

The union v5 *would* have had (published v5 manifest was not rewritten):
`decontam.oas_benchmark`, `budget.max_protein_length`, `budget.max_length`,
`quality.homotypic_pair`, `decontam.ots_benchmark`,
`decontam.asd_antibody_benchmark`, `dedup.replaces_trait`,
`decontam.trait_benchmark`, `decontam.t4_refbinder`, `decontam.t2t3_eval`,
`quality.blank_epitope`, `decontam.repertoire_core_projection`.

**Plan §3.1 item 13 counters are implemented.** `filter_report.json` now
carries, per source and in `totals`, three kept-record transformation
counters — `downgraded_all_x_mhc`, `beta_only_completed`,
`alpha_only_completed` — distinct from the six first-failure drop counters
(`count_immune_drops.py` still compares the original six keys).
`dropped_all_x_epitope` is not duplicated; it stays the
`quality.blank_epitope` filter attribution. **Authoritative post-hoc v5
numbers, including the naive-vs-correct `downgraded_all_x_mhc` caveat
(do not recount MHC-strip with a peptide-without-MHC heuristic): plan §4.2.**
The published v5 `filter_report.json` does not contain these keys.

### Still open (do not mark done)

- `tcr_papers` records still carry `source: "tcr_native"` in record identity
  (pre-existing). `metadata.dataset_source` and the shard name are correctly
  `tcr_papers`.
- The published v5 `dataset_manifest.json` / `filter_report.json` still have
  the old `filter_names` and no item-13 counters.

The ingest root cause (`ingest_papers.py` `_drop_placeholder` not applied to
`epitope` / `mhc_pseudo`) is owned by
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md)
§6.0.1 and remains **the single most important unfixed item**. Rebuilding
`tcr_papers_v2/dataset/` would re-admit all-X rows.

The former `qwen3_vl_arch/data` alias tree, old `training` tree, `GRAMMAR_V1.md`, old dataset
implementation, and retired builders/tests/jobs are deleted. `refactor_baseline` and
`docs/archive` remain historical evidence and are not current runtime inputs.

## Historical pre-refactor notes（不可作为当前入口）

The old direct-CSV seven-source mix (2026-08-29, after near-dup moves) kept
**7,794,375** train rows. Those counts are **not** v4/v5 prepared counts.
v4 published counts: plan §4.1. Decontam accidents, key spaces, and the
prefix-sampling / `语料 ∩ benchmark` / `PASS: true` rules:
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).

Still-true format pitfalls (do not drop):

- **Do not use** `data/tcr_repertoire/dataset/build_report.json::split_counts`
  for row counts. Near-dup relocation rewrote CSVs, not that report. Counts
  come from the CSV + `data/tcr_native/dataset/near_dup_eval_move_report.json`.
  `build_report.json` remains the **decontam** authority (`PASS` /
  `decontam_mode` / `blocklist_provenance`).
- **`TCR_PAPERS_DEFAULT_DIR` points at `data/tcr_papers_v2/dataset`**
  (fixed 2026-08-29). Pre-fix stats may silently have measured v1 (−274k rows).
- **ASD valid loss is not a model-selection signal**: same blocklist, train
  drop 67.5% vs valid drop 5.0%. Detail: DATA_PIPELINE_README §6.4.
- **Layout shares must be reservoir-sampled**, never a prefix.
  `tcr_papers_v2` is seven corpora concatenated; the first 30k rows are 100%
  `tcr_peptide`, the full set is 80.3% `tcr_pmhc`.
- Record share ≠ residue share. `ImmuneSourceSpec.weight` was historically
  discarded at load; mixture was disk-row-count. Prepared `BioSeqRecord`
  now stores source weight; whether loss reads it is a separate modelling
  decision (plan / DATA_PIPELINE_README §6.3).

Decontam accidents (`PASS` ≠ freshness; exact blocklists must be benchmark-derived,
never `语料 ∩ benchmark`; dual-layer load-time filter) are owned by
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md) §5.

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
  124,176 核心，`residual_blocked_in_train=0`，三 split 互斥。当时是唯一渲染为
  `tcr_single` 的源（v4 起该源已补成双链）。详见 `data/tcr_repertoire/README.md`。
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

## AB/TCR canonical v2（2026-08-04；不是现役训练输入）

Authoritative record: [`IMMUNE_RECEPTOR_DATA_V2.md`](IMMUNE_RECEPTOR_DATA_V2.md).
`export_ready=false` / `training_ready=false`. Do not treat `data/immune_receptor_v2`
as the current LLaDA prepared mix.

## 当前训练数据单一入口（2026-07-23）— 已被取代

Historical 7L / `bioseq_grammar_v1` Arrow lineage is owned by
[`TRAINING_DATA_CATALOG.md`](TRAINING_DATA_CATALOG.md). It is not the current
prepared semantic JSONL mix. Do not restore a raw-CSV training loader.

The 2026-07-08 downstream decontam bank predates the 2026-07-21 MINT official /
NM2025 / Track-A artifacts and is not proof that the current prepared mix is
clean. Current decontam rules live in
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).

## MINT 五任务官方 notebook 数据（2026-07-21）

Downstream-owned. Canonical root and protocol:
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`
and `data/downstream/mint_official`. Do not restate task schemas or row counts here.

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

The old raw-CSV dataset and generic collator described in this historical section were deleted.
The current semantic record contract is implemented by
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/records.py`,
with prepared loading in `dataset.py`, grammar/tokenization in `grammar.py` and `esm_encoding.py`,
and batch assembly in `collator.py`. The current training model receives semantic records only
after offline preparation; no raw source dataset is imported at runtime.

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

The active immune loader under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data` keeps canonical biological records independent from the ESMC/decoder tokenizer. Tokenization remains a collator/encoder concern. The retained model-layer utilities under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch` do not imply that its deleted `data` tree still exists.

Local tokenizer verification:

- ESM2 snapshots under `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D` use id 31 for `<null_1>`.
- ESMC snapshots under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` use id 31 for `|`, and mark `|` as an additional special token.
- Standard amino-acid token ids and `<mask>` id 32 match between the verified ESM2 and ESMC local tokenizers.

Implementation rule:

- Use the ESM2/MINT-compatible tokenizer for Ophiuchus-Ab and no-encoder MINT paths.
- Use the encoder's own local Hugging Face tokenizer when an ESMC encoder is active.
- Do not infer multi-chain interaction capability from the `|` token alone. The ESMC/ESMFold2 paper places explicit multi-chain complex modeling in ESMFold2, where each chain is encoded independently by frozen ESMC 6B and then fused through downstream pair/folding/diffusion modules.

The current normalization boundary is already the prepared `BioSeqRecord`/semantic JSONL contract. Future source additions must implement an offline adapter and manifest under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; they must not reintroduce a raw-CSV runtime loader or the deleted qwen data alias.

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

**历史 shard**：`data/bioseq_grammar_v1/tcr_pmhc_fulllength/{train,valid}`（旧 `build_bioseq_grammar_v1.py --sources tcr_pmhc_fulllength`，新增 `iter_tcr_pmhc_fulllength` reader 保留 roles + relation）。其渲染验证数字仅为历史证据，不属于当前 prepared semantic JSONL 训练线。

**待办（B4）**：并入整合 manifest + 权重后重训。数据布局/方法同步见 `ARCH_AUDIT.md`、`FUTURE_EXPERIMENTS.md` D1。

---

## Downstream format schemas (not owned here)

T1 / Track-A / TCRT5 output schemas and paper-correspondence rules are owned by
`downstream/benchmark/` (`RESULTS.md`, `README.md`, task docs, and
`outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`). Do not copy headline
numbers or protocol tables into this file.

