# BioSeq Data Processing Pipelines

Permanent scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/`.
Split policies: `dllm/pipelines/qwen3_vl_arch/data/ppi_splits.py`.

## 现役训练语料在哪里（2026-08-28）

**本文件描述的 `bioseq_grammar_v1` / `immune_receptor_v2` 管线都不是现役训练输入。**
现役唯一训练入口是
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`，
它按 `--dataset_args` 的 `+` token 直读 CSV。七个 token 与其目录、实测行数、长度上限
口径见
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`
的「当前训练语料实测快照（2026-08-28）」。

行数复核（不要转抄文档，直接跑）：

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/count_immune_mix.py \
  train "oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire" \
  tcr_papers_dir=/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_papers_v2/dataset
```

`count_immune_mix.py` 与 `count_immune_drops.py` 一律使用 `DataArguments` 默认值
（`max_length=max_protein_length=1024`、所有 blocklist 走真实路径）。必须知道的坑：

1. **不要为了对齐旧报告把上限降到 512**：`asd_antibody` 的抗原中位数 607 aa 卡在两个
   预算之间，512 会让该源读成 159,331 行而非真实的 276,412。
2. **`tcr_papers_dir` 的默认值已于 2026-08-29 从 v1 改到 v2**，所以现在不传该参数量到的
   就是 v3 口径。**但 2026-08-29 之前产出的任何统计都可能是 v1 口径**，读旧数字时要留意。
   需要复现特定 job 时用尾随 `field=value` 覆盖。v3 当前合计 **7,794,375** 条
   （2026-08-29 05:30 实测；此前的 7,626,737 / 7,637,419 是 `tcr_repertoire` 近重复搬迁
   之前的口径）。
3. **单源计数 ≠ 该源在混合中的计数**。`replaces_trait` 抑制键只在超越源同时被请求时
   才装配，所以 `trait` 单跑与放进完整 mix 的行数不同。要得到与训练日志一致的分源数字，
   必须传该 job 的完整 token 列表。
4. **语料在变，报告不一定跟着变**。`trait` 当前在 mix 中是 **31,515** 行（曾为 34,872，
   因 `trait_benchmark` blocklist 于 08-28 17:25:35Z 从 862 键重建到 59,212 键而下降）；
   `tcr_repertoire` 的 `build_report.json::split_counts` 已被近重复搬迁改得过期。
   **量数字就现跑一次，不要转抄任何文档或旧报告。**

5. **加载期丢弃率在 train 与 valid 上可能差一个数量级**（2026-08-29 实测）。
   同一份黑名单、同一个 `row_to_record`，只换 split：

   | 源 | train 丢弃率 | valid 丢弃率 |
   |---|---:|---:|
   | `asd_antibody` | **67.5%**（850,134 → 276,412） | **5.0%**（47,230 → 44,866） |
   | `tcr_native` | 32.7%（143,391 → 96,552） | 12.2%（4,390 → 3,855） |
   | `trait` | 54.5%（69,251 → 31,515） | 54.3%（3,050 → 1,393） |

   所以**不能用 train 的丢弃率推断 valid**，要分 split 各量一次。
   后果：ASD 的 valid loss 不适合做 early-stopping，详见
   `examples/llada/DATA_PIPELINE_README.md` §6.4。

6. **黑名单构建脚本各自维护一份训练源清单，会漂移**（2026-08-29 发现）。
   `build_t4_refbinder_blocklist.py` 与 `build_t2t3_eval_blocklist.py` 都有自己的
   `TRAINING_SOURCES`，用途是把「训练语料中与受保护 eval 对相近的 core」也扩进黑名单。
   前者列了 `tcr_papers` 与 `tcr_papers_v2`，后者**只列了 v1**——而 v3 训练的是 v2。
   已补上 v2；实测补上后 T2/T3 黑名单 8,377 → 9,177 键（+800，对应 v2 train 890 行、
   0.13%，精确泄漏为 0，全是 Lev-1 邻居）。**加新源到 mix 时必须同步这两处清单。**
   现役黑名单刻意未重建，原因（会打破 `tcr_repertoire` 新鲜度断言）见
   `downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md` §7.9。

要显式关闭某个 blocklist 只能传 `none`（`""` 会被 `load_exclusion_keys` 直接抛错）。

### 布局分布必须蓄水池采样，不能取前缀（2026-08-29 修）

```bash
python scripts/count_grammar_layouts.py \
    --tokens oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire \
    --per-source 30000 --seed 0
```

`count_grammar_layouts.py` 原先读满 `--per-source` 就 `break`，取的是**前缀**。
布局是「行填了哪些字段」的函数，而 `tcr_papers_v2` 是 7 个论文语料首尾拼接的，
**前 3 万行 100% 是 `tcr_peptide`**，全量却是 80.3% `tcr_pmhc` —— 547,274 条被归错，
`tcr_pmhc` 低估约 4 倍。同类陷阱也坑过"取各源真实首行"举例的做法：
`trait` 首行没有 MHC，但全量 `trait` 95.2% 是 `tcr_pmhc`。

**规则：源内不同质时，任何"抽样看看"都必须随机抽，且结论要与一次全量扫描对齐过。**
现已改蓄水池采样 + `--seed`，实测 671,678 vs 全量 673,686（噪声内）。
看 **mix-weighted** 视图，不要看 per-source 截断视图（后者把小 TCR 源虚高一个数量级）。
输出的 `kept=` 列是**加载期过滤后**的行数，不是磁盘行数（旧版叫 `rows=`，会误读）。

## 补齐缺失原始源（2026-08-28）

```bash
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_missing_tcr_sources.sh
```

经实验室代理 `http://100.68.162.212:3128` 用 aria2c 断点续传，逐文件核对官方 MD5，
已校验文件跳过，可安全重跑。落盘：

- `data/tcr/vdjdb-2026-06-03.zip` — VDJdb 2026-06-03 release（盘上原先只有 2025-12-29）
- `data/ots_tcrlang/raw/{TCRLang_Datasets,tcrlang-weights,OTS_CoherenceCode}.tar.gz`
  — Zenodo 11208211（OTS/TCRLang，CC-BY-4.0）

两者**均未接入任何训练配置**。TCRLang 的官方 test/eval 与本项目 OTS train 有
98.2%/98.3% 精确配对重叠，不可作为 held-out benchmark；详见 `DATA_FORMAT_AUDIT.md`。

## Active AB/TCR canonical pipeline (`bioseq.v2`)

The next immune-receptor recipe is limited to paired antibody H/L, paired TCR
alpha/beta, antibody-antigen, and TCR-peptide/pMHC. MINT/general PPI,
nanobody/VHH, and structure benchmark targets are outside this pipeline. OAS
and OTS remain the already-cleaned pairing corpora; this v2 build canonicalizes
the recognition, affinity, and property sources that complement them.

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_immune_receptor_v2_sources.py \
  --sources iedb,catnap,sabdab2
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py inventory
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py build-tcr
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py build-antibody
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py finalize
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py export --threads 64 --negative-ratio 1.0
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py pairing-export --threads 64
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py recipe
```

Canonical records, source/download registries, split manifests, and reports are
written below
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2`.
The source's released train/test field is retained only as provenance. Exact
measurements may be provenance-merged, but conflicting labels or independent
assay values are retained separately; no majority voting is allowed. The
technical immutable exports are now materialized as core build
`ir2exp_f7a60484c7e3a20db6a2`, strict OAS/OTS pairing build
`ir2pair_6a1a5b62752caccd2cd5`, and candidate recipe
`ir2recipe_1e3eac44551e88aedc6e`. Benchmark exact/near quarantine and
train-only TCR negative generation pass, but `export_ready` remains false until
the SAbDab2/CATNAP rights review is complete; training additionally requires
runtime view/render support and frozen sampling/token-budget policy. Do not
train from `canonical/` or `splits/` directly.

The sections below describe the historical PPI/grammar-v1 pipeline. They are
not part of the active AB/TCR-only v2 build.

## Historical PPI/grammar-v1 overview

```text
Immune receptor data (OAS / OTS)          Interaction / PPI data
────────────────────────────────          ────────────────────────
External clean pipeline                   scripts/data/download_stringdb_assets.sh
  → data/oas_previous_clean/                → data/ppi_task_raw/raw/stringdb_mint/
  → data/ots_paired_clean/                scripts/data/build_ppi_unified_csv.py
                                            → processed/interaction_records_unified.csv
build_bioseq_grammar_v1.py (oas, ots)     build_mint_string_splits.py (MMseqs2 clu50)
  → data/bioseq_grammar_v1/oas|ots/         → processed/mint_string_pretrain_v1/
                                          build_mint_grammar_shards.py
                                            → bioseq_grammar_v1/mint_ppi/
                                          build_supervised_grammar_shards.py
                                            → bioseq_grammar_v1/neutralization/
                                          build_bioseq_grammar_v1.py (ppi, mix)
                                            → bioseq_grammar_v1/ppi/
```

## Step-by-step commands

### Step 0 — Immune CSV (already on disk)

OAS / OTS final splits are immutable inputs (same role as external preprocessing):

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final/`

### Step 1 — Unified interaction CSV (~3GB, long)

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_ppi_unified_csv.py
# subset rebuild:
python .../scripts/build_ppi_interaction_csv.py --sources covabdab_neutralization
```

Output: `data/ppi_task_raw/processed/interaction_records_unified.csv` with `grammar_relation`.

### Step 2 — STRING download

```bash
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_stringdb_assets.sh
# optional functional channel subscores (~190GB):
bash .../download_stringdb_assets.sh --with-detailed
```

Physical links for MINT are already present locally.

### Step 3 — MINT official splits (requires MMseqs2)

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/stringdb_mint
gunzip -k protein.sequences.v12.0.fa.gz
mmseqs createdb protein.sequences.v12.0.fa DB100
mmseqs cluster DB100 clu50 /tmp/mmseqs --min-seq-id 0.50 --remove-tmp-files
mmseqs createtsv DB100 DB100 clu50 clu50.tsv

python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_mint_string_splits.py
```

Policy: `mint_string_pretrain_v1` (~96M train / 250k valid, cluster-disjoint).

### Step 4 — MINT grammar Arrow shards (~96M pairs)

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_mint_grammar_shards.py --split train
python .../build_mint_grammar_shards.py --split valid
```

Output: `data/bioseq_grammar_v1/mint_ppi/{train,valid}/`

### Step 5 — Supervised shards (`<neutralization>`, etc.)

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_supervised_grammar_shards.py \
  --sources covabdab_neutralization
```

Output: `data/bioseq_grammar_v1/neutralization/train/`

Runtime grammar form (via `GrammarRenderer`): `<prots> <ab> HEAVY . LIGHT <protd>` with `<neutralization>` relation fixed in context-heavy layouts when neutralization shards are enabled.

### Step 6 — STRING functional channel sample (optional)

After `--with-detailed` download:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_string_functional_edges.py \
  --max-records 1000000
```

### Step 7 — Mixed grammar_v1 cache (OAS + OTS + TCR + PPI + neutralization)

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_bioseq_grammar_v1.py \
  --sources oas,ots,tcr,ppi,neutralization \
  --splits train,valid \
  --ppi-split-policy bernett_string_90_90_hf
```

For MINT-scale PPI pretraining, use `mint_ppi` shards separately (do not mix 96M into small Bernett cache).

### Step 8 — Audit

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_training_data_scale.py \
  --json-out data/ppi_task_raw/processed/training_data_audit.json
```

See also: `dllm/pipelines/qwen3_vl_arch/data/DATA_SCALE_AND_SPLITS.md`

## Script index

| Script | Purpose |
|--------|---------|
| `download_missing_tcr_sources.sh` | VDJdb 2026-06-03 + Zenodo 11208211 (OTS/TCRLang), md5-verified, resumable |
| `download_stringdb_assets.sh` | Download STRING sequences + physical links (+ optional detailed) |
| `build_ppi_unified_csv.py` | Step 1: unified CSV with `grammar_relation` |
| `build_mint_string_splits.py` | Step 3: MINT train/valid link files |
| `build_mint_grammar_shards.py` | Step 4: ~96M PPI Arrow shards |
| `build_supervised_grammar_shards.py` | Step 5: CoV-AbDab `<neutralization>` shards |
| `build_string_functional_edges.py` | Step 6: functional channel sample from detailed links |
| `build_bioseq_grammar_v1.py` | Step 7: OAS/OTS/TCR/PPI/neutralization mixed cache |
| `audit_ppi_sources.py` | Inventory JSON |
| `dedup/audit_downstream_leakage.py` | **训练混合 × 下游测试集的残留泄漏矩阵**（见下） |
| `tcr_native/assert_corpus_fresh.py` | 训练前断言语料未因黑名单重建而过期 |
| `assert_residue_alphabet.py` | 七源 × 3 split 残基字母表断言（`.`/`-`/`|` 会崩 RemapCollator） |

### 下游泄漏审计（训练前应跑）

```bash
python scripts/data/dedup/audit_downstream_leakage.py   # 约 100 秒
```

输出 5 个 TCR 源 × 11 个下游测试集的矩阵。**结论行只统计 hard-requirement**：

```
HARD-REQUIREMENT hits = 0   -> PASS
```

它把每行都推过该源**真实的** `row_to_record`（含所有加载期过滤器），
只统计存活的行，即训练真正看到的数据。

**读矩阵有个坑：非零格子不等于泄漏。** 只有 `NM2025_seen` /
`NM2025_unseen` / `public_trackA` 是按裸 CDR3β core 保护的，命中即泄漏；
其余基准按 `(core|epitope)` 配对键保护，把它投影成裸 core 去比会按构造过报
（实测 `tcr_papers_v2` 约 7.7 万裸 core 命中 T4，但配对层命中为 **0**）。

完整的键空间说明、黑名单清单与去污事故复盘：
[`examples/llada/DATA_PIPELINE_README.md`](../../examples/llada/DATA_PIPELINE_README.md)。

### 残基字母表（训练前应跑）

```bash
python scripts/data/assert_residue_alphabet.py                 # 七源 × train/valid/holdout
```

发现 `.` / `-` / `|` 等 `RESIDUES` 之外的字符就 **exit 1**。
`RemapCollator` 映射不了这三个 ESMC token（id 29/30/31），会在 DataLoader 里
把一个 rank 打死、其余卡到 NCCL 1800s 超时。

2026-08-31 全量复扫 21 个文件、约 880 万行，只命中
`data/tcr_papers_v2/dataset/train.csv` 第 3046 行一格
（`cdr3b` `ASSKVAARVP-TLKLS` → `ASSKVAARVPTLKLS`，行数不变）。
语料记录：[`data/tcr_papers_v2/README.md`](../../data/tcr_papers_v2/README.md)
「残基字母表」；事故复盘：
[`examples/llada/DATA_PIPELINE_README.md`](../../examples/llada/DATA_PIPELINE_README.md) §5.4。

## Split policy (never invent random splits)

See `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/PPI_DATA.md`.
