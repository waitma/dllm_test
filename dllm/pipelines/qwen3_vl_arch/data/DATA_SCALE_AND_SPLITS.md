# Training Data Scale and Split Audit

Last verified by `scripts/data/audit_training_data_scale.py`.

## Executive summary

| Source | Grammar v1 train rows | Scale vs target | Split policy | Verdict |
|--------|----------------------:|-----------------|--------------|---------|
| **OAS** | 2,486,442 | Full | Project-owned (`oas_previous_clean/splits`) | OK |
| **OTS** | 2,102,715 | Full | Project-owned (`ots_paired_clean/final`) | OK |
| **TCR** | 163,725 | **Partial (~58% of PISTE train alone)** | `processed_v2` ad-hoc | **Replace with PISTE / Nat Methods splits** |
| **PPI** | 319,429 | **~0.33% of MINT pretrain** | Bernett 90/90 (eval tier) | **Add MINT 96M pretrain; keep 90/90 test for eval** |

Formal grammar_v1 Volc jobs (`train_jobs/qwen3_vl_bioseq_grammar_v1_*.yml`) use **full** OAS/OTS/TCR/PPI shards (no `--limit-per-source`). TCR/PPI are **not smoke-limited**, but they are **undersized and on non-recommended splits**.

Legacy `processed_v2` path (non-grammar jobs) mixes ~640k PPI + ~164k TCR in one JSONL without published split semantics.

---

## TCR: what we have vs what literature uses

### Current grammar_v1 TCR (~163k train)

- Built from `data/processed_v2/{train,val}.jsonl`
- Sources: VDJdb (~91k) + MIRA (~47k) + McPAS (~26k) after dedup
- **Not** aligned with PISTE, NetTCR-2, ERGO-II, or Nat Methods 2025 benchmark splits

### Already on disk (not in grammar_v1)

| Dataset | Local path | Unified CSV rows | Official split |
|---------|------------|-----------------:|----------------|
| **PISTE random** | `ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/` | ~358k total | `train_data.csv` / `val_data.csv` / `test_data.csv` |
| TDC Weber | `ppi_task_raw/raw/tdc_tcr_epitope/` | 47k | TDC export |
| TEIM | `ppi_task_raw/raw/teim_interface/` | 46k | Author TSV splits |
| VDJdb full | `data/tcr/vdjdb*.txt` | — | Use VDJdb confidence + benchmark protocols |
| IEDB T-cell | `data/tcr/iedb_tcell_full.zip` | — | IEDB assay tables |
| McPAS / MIRA | `data/tcr/`, `processed_v2` | — | Merge via published pipelines |

### Recommended TCR split policies (see `ppi_splits.py`)

1. **`piste_tcr_random`** — PISTE author splits (~284k / 71k / 3k train/val/test)
2. **`nat_methods_2025_tcr_benchmark`** — [Nat Methods 2025](https://www.nature.com/articles/s41592-025-02910-0), [figshare 27020455](https://doi.org/10.6084/m9.figshare.27020455): 19 databases, seen/unseen epitope tests
3. **NetTCR-2 / TChard** — strict epitope-held-out (for eval, not random merge)

### Large TCR corpora to add (literature-standard)

| Corpus | Scale | Reference |
|--------|------:|-----------|
| PISTE pMHC-TCR | ~1M+ rows in unified CSV | Armilius/PISTE |
| IEDB + VDJdb + McPAS merged | ~113k pairs (tpp); up to ~200k+ in benchmarks | Frontiers Immunol 2023 tpp |
| Nat Methods 2025 merged | Multi-source with published train/test | s41592-025-02910-0 |
| 10x / ImmuneCODE | Repertoire-scale | NetTCR-2, TChard |
| dbPepNeo / TCRdb2.0 | Neoantigen TCR | Nat Methods benchmark supp |

---

## PPI: what we have vs MINT

### Current grammar_v1 PPI (~319k train)

- Source: `string_model_org_90_90_split` (Bernett 90/90)
- Raw: 645k train → filtered (same-organism, dedup, 1024 aa crop) → 319k
- **Tier mistake for foundation pretraining**: this is IRBench / MINT **eval** gold standard

### MINT-scale pretraining (target ~95.8M train)

| Stage | Status |
|-------|--------|
| STRING sequences + physical links | Downloaded (~11G + ~13G) |
| MMseqs2 `clu50.tsv` | **Not built** |
| `build_mint_string_splits.py` | **Not run** |
| `build_mint_grammar_shards.py` | **Not run** |

### Other PPI assets on disk (eval / supervised)

| Dataset | Rows (unified CSV) | Split policy |
|---------|-------------------:|--------------|
| Figshare Bernett gold | 274k | `intra0/1/2` eval |
| HumanPPI LMDB | 69k | PEER/SaProt train/valid/test |
| YeastPPI LMDB | 38k | PEER splits |
| SKEMPI | 7k | 3-fold complex CV |
| oncoPPI | 107k | Case study |
| FLAb antibody binding | 4.6M | Supervised finetune (`<unknown>` relation) |

---

## Expansion priority

```bash
# 1. Audit (machine-readable)
python scripts/data/audit_training_data_scale.py \
  --json-out data/ppi_task_raw/processed/training_data_audit.json

# 2. Download / refresh
bash scripts/data/download_expansion_datasets.sh

# 3. PPI pretrain (MINT protocol)
# mmseqs ... -> build_mint_string_splits.py -> build_mint_grammar_shards.py

# 4. TCR (PISTE official split) — builder TODO: build_tcr_grammar_shards.py

# 5. Rebuild unified CSV (in progress)
python scripts/data/build_ppi_unified_csv.py
```

---

## Split rule (project policy)

- **OAS / OTS**: only project-owned splits (already done)
- **Everything else**: use `ppi_splits.py` registry; never random `train_test_split` in training builders
- **Eval test sets**: never merge into pretraining (Bernett 90/90 test, PISTE test, HumanPPI test, etc.)
