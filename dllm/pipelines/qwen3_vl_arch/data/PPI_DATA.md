# PPI and Interaction Data

This note maps local interaction assets to `grammar_v1` relation tokens and
training tiers. Regenerate the machine-readable inventory with:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py
```

## Grammar relation vocabulary

`grammar_v1` relation tokens (see `GRAMMAR_V1.md`):

| Token | Meaning | Typical sources |
|-------|---------|-------------------|
| `binding` | Physical or functional binding / interaction | STRING physical, HumanPPI, antibody-antigen, TCR-epitope |
| `activation` | Upstream activates downstream | STRING activation channel (not downloaded yet) |
| `inhibition` | Upstream inhibits downstream | STRING inhibition channel |
| `catalysis` | Enzymatic catalysis | STRING catalysis channel |
| `reaction` | Biochemical reaction edge | STRING reaction channel |
| `expression` | Expression regulation | STRING expression channel |
| `ptmod` | Post-translational modification | STRING ptmod channel |
| `neutralization` | Antibody neutralizes antigen/virus | CoV-AbDab neutralization |
| `nonbinding` | Explicit negative / loss of binding | Negative PPI pairs, mutational loss labels |
| `unknown` | Relationship between the two chains is not specified | FLAb, SKEMPI, TEIM, generic multi-source pairs |

Implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/ppi_relations.py`

## Data tiers (how assets relate)

```text
pretraining_candidate   STRING-DB physical links (MINT-scale, ~96M curated pairs)
grammar_v1_current      string_model_org_90_90_split -> bioseq_grammar_v1/ppi (~319k train)
supervised_eval         Bernett gold-standard, HumanPPI, YeastPPI, SKEMPI, TCR benchmarks
supervised_finetune     FLAb antibody binding regression
case_study              oncoPPI Y2H validation sets
```

Current formal `grammar_v1` training uses only the **grammar_v1_current** tier for PPI.
All those rows serialize at **encode time** as (PPI conditional):

```text
<prots>PROTEIN_A<protd><binding><prots>PROTEIN_B<protd>
```

## Local inventory (2026-06-21)

| Asset | Scale | Default relation | Role |
|-------|------:|------------------|------|
| `data/ppi_task_raw/raw/stringdb_mint/*.gz` | ~13G physical links + ~11G sequences | `binding` (physical) | MINT-scale pretraining candidate |
| `data/ppi/string_model_org_90_90_split` | 645k train / 5.8k valid / 1.3k test | `binding` | Eval + grammar_v1 PPI source |
| `data/bioseq_grammar_v1/ppi` | 319k train / 2.9k valid | `binding` | Current training cache |
| `data/ppi_task_raw/processed/interaction_records_unified.csv` | ~6.27M rows | inferred per `task_family` | Audit / future sharded adapters |
| `data/processed_v2` PPI JSONL | ~640k train PPI rows | `binding` | Legacy; excluded from grammar_v1 PPI |

### Unified interaction sources (`ppi_task_raw`)

| source_id | task_family | Default relation | Records (manifest) |
|-----------|-------------|------------------|-------------------:|
| stringdb_mint | ppi_pretraining | binding | 0 (raw files partial/complete on disk) |
| figshare_gold_standard | ppi_binary | binding | 274,500 |
| saprot_humanppi | ppi_binary | binding | 68,945 |
| peer_yeastppi | ppi_binary | binding | 38,158 |
| skempi | ppi_mutation_affinity | binding | 7,085 |
| swing_mutint | mutational_ppi | binding / nonbinding by label | 12,612 |
| flab | antibody_binding | binding | 4,606,793 |
| covabdab_neutralization | antibody_neutralization | **neutralization** | 12,918 |
| tdc_tcr_epitope | tcr_epitope_binding | binding | 47,182 |
| piste_tcr_epitope_hla | tcr_epitope_hla | binding | 1,051,227 |
| teim_interface | tcr_epitope_interface | binding | 45,603 |
| oncoppi | oncogenic_ppi | binding | 106,536 |

Negative binary pairs (`label=0`) map to `nonbinding` when rebuilding unified CSV.

## Canonical split policies (do not invent random splits)

Implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/ppi_splits.py`

| Policy ID | Source | Allowed splits | Use |
|-----------|--------|----------------|-----|
| `mint_string_pretrain_v1` | STRING physical links | `train`, `valid` | MINT-scale foundation pretraining (~96M / 250k) |
| `bernett_string_90_90_hf` | `string_model_org_90_90_split` | `train`, `valid`, `test` | Grammar mix + IRBench P1 (test eval-only) |
| `bernett_gold_standard_figshare` | Figshare Intra0/1/2 | `intra0`, `intra1`, `intra2` | Supervised eval (MINT gold-standard) |
| `peer_humanppi_lmdb` | SaProt HumanPPI LMDB | `train`, `valid`, `test`, `cross_species_test` | MINT HumanPPI benchmark |
| `peer_yeastppi_lmdb` | PEER YeastPPI LMDB | same | MINT YeastPPI benchmark |
| `skempi_complex_3fold` | SKEMPI | `fold0`, `fold1`, `fold2` | Complex-held-out CV |

### Recommended training tiers

1. **Foundation pretraining (PPI)** — follow MINT exactly:
   - Download: `scripts/data/download_stringdb_assets.sh`
   - MMseqs2 50% cluster → `build_mint_string_splits.py`
   - Use `training_filtered` for train, `validation` for valid
   - Never merge Bernett / HumanPPI / YeastPPI test rows

2. **Current grammar_v1 mixed training** — PPI portion uses Bernett 90/90 published `train` + `valid` only (`bernett_string_90_90_hf`). The builder rejects `test`.

3. **Downstream eval** — keep each benchmark's published test split:
   - IRBench P1: `string_model_org_90_90_split/test`
   - HumanPPI / YeastPPI: LMDB `test` and `cross_species_test`
   - Bernett gold-standard: `intra0` pos/neg lists

```bash
# MINT STRING splits (after MMseqs2 clu50.tsv exists)
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_mint_string_splits.py

# Grammar PPI cache — only published train/valid
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_bioseq_grammar_v1.py \
  --splits train,valid --sources ppi --ppi-split-policy bernett_string_90_90_hf
```

## MINT comparison (Nat Commun 2026)

| | MINT STRING pretraining | This project (today) |
|---|--:|--:|
| Curated PPI pairs | ~95.8M | ~319k (grammar cache) |
| Link type | physical (binding) | physical only |
| Functional edges | separate STRING channels | not ingested yet |

STRING functional link files do not exist as separate per-channel downloads.
Channel subscores are in ``protein.links.detailed.v12.0.txt.gz`` (~190GB):

```bash
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_stringdb_assets.sh --with-detailed
```

Parse the dominant channel subscore to grammar tokens (`activation`, `inhibition`, etc.)
once the detailed file is wired into the CSV builder.

## Rebuild commands

Unified audit CSV (adds `grammar_relation`, `string_channel`):

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py
```

Grammar PPI cache (STRING 90/90 only today):

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_bioseq_grammar_v1.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1 \
  --splits train,valid \
  --sources ppi
```

Inventory JSON:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_ppi_sources.py
```

Full step-by-step pipeline: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/README.md`
