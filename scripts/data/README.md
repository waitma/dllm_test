# BioSeq / PPI Data Processing Pipeline

Permanent scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/`.
Split policies: `dllm/pipelines/qwen3_vl_arch/data/ppi_splits.py`.

## Overview

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
| `download_stringdb_assets.sh` | Download STRING sequences + physical links (+ optional detailed) |
| `build_ppi_unified_csv.py` | Step 1: unified CSV with `grammar_relation` |
| `build_mint_string_splits.py` | Step 3: MINT train/valid link files |
| `build_mint_grammar_shards.py` | Step 4: ~96M PPI Arrow shards |
| `build_supervised_grammar_shards.py` | Step 5: CoV-AbDab `<neutralization>` shards |
| `build_string_functional_edges.py` | Step 6: functional channel sample from detailed links |
| `build_bioseq_grammar_v1.py` | Step 7: OAS/OTS/TCR/PPI/neutralization mixed cache |
| `audit_ppi_sources.py` | Inventory JSON |

## Split policy (never invent random splits)

See `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/PPI_DATA.md`.
