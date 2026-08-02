# Data Format Audit

Date: 2026-06-14

Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`

## 当前训练数据单一入口（2026-07-23）

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

**SAbDab2 说明**：Zenodo 20083995 仅含 `splits.tar.gz`，解包后只有
`ab_split.csv`（15,641 行，抗体相似度划分）+ 427 CIF，**无
`abag_split.csv`**（抗原感知划分）。`ab_split.csv` 有抗原列
（`agtypes`/`agresolvedseqs`，8,455 行名义上含 antigen）。严格要求 paired
VH/VL、每个 antigen component 都是 `PROTEIN/PEPTIDE`、序列合法且每链 ≤1024 后，
实际剩 3,980 rows、2,489 个唯一 `(VH,VL,antigen chains)`。原 train/test 中仍共享
60 个完整 antigen tuple、78 个 antigen component sequence，所以下一版必须重新做
antigen-aware group split。

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
