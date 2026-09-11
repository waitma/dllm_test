# Immune LLaDA data pipeline

> Last updated: 2026-09-12
>
> This directory owns the **operational** data path for the ESMC-conditioned
> LLaDA immune model (how to preprocess, what the prepared contract is, which
> prepared roots are current). It is independent of `qwen3_vl_arch`.
>
> | Fact | Owner (do not restate here) |
> |---|---|
> | Prepared layout / completion / all-X / profile fields | [`DATA_FORMAT_AUDIT.md`](../../../DATA_FORMAT_AUDIT.md) |
> | Design rationale and risk register | [`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](../../../docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md) |
> | Experiment / preprocess progress | [`PROJECT_PROCESS.md`](../../../PROJECT_PROCESS.md) |
> | Long-lived rules | [`PROJ_GUIDE.md`](../../../PROJ_GUIDE.md) |
> | Raw-corpus decontam accidents | [`examples/llada/DATA_PIPELINE_README.md`](../../../examples/llada/DATA_PIPELINE_README.md) |
> | v4 published row counts | plan §4.1 (same PLAN file) |

`dllm/pipelines/immune_llada/` is workspace-local (never added to git, not
gitignored). Do not describe this tree's history in git terms.

## Prepared versions

| Root | Config | Status |
|---|---|---|
| `data/prepared/immune_v4_beta_relation` | `configs/data/immune_v4_beta_relation.yaml` | **Current published.** Completion default: `tcr_repertoire` only. |
| `data/prepared/immune_v5_receptor_completion` | `configs/data/immune_v5_receptor_completion.yaml` | **进行中.** `completion_sources: [trait, tcr_native, tcr_papers, tcr_repertoire]`. Dataset-level manifest is not published yet — **do not invent final row counts, sizes, or token shares.** |
| `data/prepared/immune_v3/` and `immune_v3_heterotypic/` | `configs/data/immune_v3.yaml` | Live on-disk artifacts. Some v3 checkpoints and eval YAMLs still point here. **Not** a current training default. |

Qualitative expectation for v5 (not an audit number): total rows drop by roughly
39k (all-X epitope drops via `quality.blank_epitope`); the three epitope sources
grow per-row because each completed record gains a full synthetic chain plus
`synthetic_residue_mask`.

**v5 final audit numbers — placeholder (fill only from the published manifest):**

| split | records | notes |
|---|---:|---|
| train | *待 v5 manifest* | |
| valid | *待 v5 manifest* | |
| dropped `quality.blank_epitope` | *待 `filter_report.json`* | named filter; not a silent adapter `None` |

## Boundary

```text
raw CSV/JSONL
  -> source adapter (`data/sources.py`)
  -> canonical `BioSeqRecord` (`data/records.py`)
  -> offline filters (`data/preprocessing/filters.py`)
  -> semantic JSONL shards + manifest
  -> training-time `GrammarRenderer`
  -> batch padding/tensors
  -> per-chain encoder input reconstruction
  -> diffusion model batch
```

Preprocessing performs every operation that can reject a row: source schema
conversion, receptor completion (when the source is in `completion_sources`),
invalid-sequence handling, benchmark/decontamination blocklists, and
protein/grammar length budgets. A rejected row is counted in `filter_report.json`.
The prepared loader does not parse raw columns, reapply filters, retry, replace,
or silently skip rows. It still scans the prepared JSONL shards once at dataset
construction to build byte-offset indexes, then decodes and validates one semantic
row per access. Removing raw preprocessing is not a zero-overhead loading claim.

The following remain deliberately in the training hot path because they depend
on the current batch and diffusion state:

- grammar rendering from `BioSeqRecord`;
- batch padding and masks/tensor assembly;
- per-chain encoder input reconstruction;
- random diffusion/MLM masking and noise.

No model-ready token cache is produced.

## Prepared output

A successful run publishes shards first and writes the dataset-level manifest only
after all selected sources and splits finish:

```text
<data-dir>/
├── dataset_manifest.json
├── filter_report.json
├── validation_report.json
├── schema.json
└── train/
    ├── oas-00000.jsonl
    └── ots-00000.jsonl
```

Each JSONL row is a canonical semantic record plus `schema_version`. The manifest
is the only source of shard membership. Each shard is written to a temporary
file and atomically renamed after completion. The manifest records
`completion_sources` (once a run that knows the field finishes) so a prepared
record's layout is reconstructable.

## Latest pipeline changes (2026-09-12)

Code lives under `dllm/pipelines/immune_llada/data/`. Layout facts and the
deliberate deviations are owned by
[`DATA_FORMAT_AUDIT.md`](../../../DATA_FORMAT_AUDIT.md) and
[`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](../../../docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md).
Short operational summary:

1. **Receptor completion now covers the epitope-conditioned sources** when listed
   in `tcr_region_profile.completion_sources`. `complete_tcr_chain(...)` (RNG
   seed derived per-source), `_receptor_chains(...)` (beta-first; a real
   full-length Fv is kept verbatim), and `COMPLETION_SOURCES`. Junction vs core
   is decided from provenance, never from the first/last character. Branching is
   on actual `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` contents, not the
   `sequence_scope` label.
2. **all-X placeholders** (Zenodo 14545852 / TcrDesign `'X'` = missing): all-X
   epitope rows drop via named filter `quality.blank_epitope`; all-X MHC chains
   are stripped and the record degrades to `tcr_peptide`, with `task_type`
   derived from whether the MHC chain survived.
3. **Completion is config-gated.** `PreprocessConfig.completion_sources` defaults
   to `("tcr_repertoire",)` so an existing config keeps producing the dataset it
   produced before. Withholding the profile from a source disables completion.
4. **Circular-profile guard.** `add_tcr_region_lengths` skips regions listed in a
   chain's `synthetic_regions` metadata.
5. **`sample_region_lengths` now sorts lengths numerically** so a published
   dataset can be regenerated from its own manifest (`atomic_json_dump` writes
   `sort_keys=True`).
6. **Verification:** 3,000 `tcr_repertoire` rows regenerated from the raw CSV
   against the v4-frozen profile vs the on-disk shard: **0 mismatches**. Suite:
   **194 passed, 5 failed** (the 5 are pre-existing: `test_full_parity` ×3,
   `test_profiling` autocast ×2). New tests:
   `scripts/tests/immune_llada/test_receptor_completion.py`,
   `test_null_context_prefix.py`.
7. **Null-context prefix: audited, no code change.** `<prots> <null> <protd>
   <unknown>` is already applied to unconditional layouts only
   (`antibody_pair` / `tcr_pair` / `tcr_single` / `nanobody` → `oas`, `ots`,
   `tcr_repertoire`) and not to conditioned layouts. 2,240 prepared rows, 0 gaps;
   `grammar.py` unchanged. Open item: unused `single_entity` fallback has no
   prefix and no current source reaches it.

## Configuration and command

Use `configs/data/immune_v4_beta_relation.yaml` to reproduce the published v4
root, or `configs/data/immune_v5_receptor_completion.yaml` for the in-progress
completion mix. Paths in production configs should be absolute.

```bash
python scripts/data/preprocess_immune_dataset.py \
  --config configs/data/immune_v5_receptor_completion.yaml \
  --output-dir data/prepared/immune_v5_receptor_completion \
  --split train
```

For a bounded validation run:

```bash
python scripts/data/preprocess_immune_dataset.py \
  --config configs/data/immune_v5_receptor_completion.yaml \
  --output-dir /tmp/immune_llada_smoke \
  --source oas+ots+trait+tcr_native \
  --split train \
  --max-rows 100 \
  --dry-run
```

To audit legacy-to-canonical adapter parity without touching prepared data or
training, compare a bounded raw CSV prefix per source:

```bash
python scripts/data/check_immune_adapter_parity.py \
  --split train \
  --max-rows 5000 \
  --output refactor_baseline/immune_adapter_parity_train_5000.json
```

This audit compares keep/drop decisions, chain sequences, task type, source, and
roles/relation where the legacy adapter exposes them. It is intentionally an
offline migration check, not a runtime dataset dependency. Strict historical
parity vs the frozen v3 baseline is an **approved deviation** (homotypic-pair
drop + later completion), not a silent relabel.

Load prepared records in training code with:

```python
from dllm.pipelines.immune_llada.data import load_prepared_dataset

dataset = load_prepared_dataset("/absolute/path/to/prepared", split="train")
```

Pass those `BioSeqRecord` objects to the existing immune grammar collator. The
collator continues to build `encoder_input_ids` and its masks for every batch.

v3-era acceptance evidence (51 homotypic pairs, `quality.homotypic_pair`) remains
in [`docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md`](../../../docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md).
That document is not the current v4/v5 contract.
