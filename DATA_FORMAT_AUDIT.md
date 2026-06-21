# Data Format Audit

Date: 2026-06-14

Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`

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

Final/current path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final`

Rows:

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
