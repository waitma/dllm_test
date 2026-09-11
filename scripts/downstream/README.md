# Immune LLaDA evaluation entry points

The active evaluation surface is intentionally small and targets the current
Immune LLaDA fusion checkpoints produced by:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`

All commands below are run from:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test`

## Representative evaluations

- `run_immune_fusion_gen.sh <checkpoint> <tag>`: one driver for antibody CDR
  infilling, OAS light-chain pairing, and T4 conditional TCR generation. Use
  `GEN_STAGES=cdr|pairing|t4` to run one stage.
- `run_immune_fusion_pairing.sh <checkpoint> <tag>`: pairing generation with
  the reference-independent light-length prior and leakage diagnostics.
- `run_immune_fusion_repr.sh <checkpoint> <tag>`: T1/T2/T3 frozen-representation
  suite through `downstream/benchmark`.
- `run_pairing_pll.sh <checkpoint> <tag>`: model-native `p(L|H)-p(L)` pairing
  score; this is complementary to generation-based ImmunoMatch metrics.
- `score_binding_relation.py`: offline teacher-forced score of the supervised
  recognition-relation token (`<binding>` / `<nonbinding>`) from a fusion
  checkpoint. Does not touch the training entrypoint. Score **only**
  `relation_target_mask == 1`. Do **not** use `relation_token_mask`: it also
  lights up MHC→peptide presentation `<binding>` (fixed context) and the
  OAS/OTS null-prefix `<unknown>`. Accuracy is almost meaningless on full
  valid — class balance is asd 39866/5000, tcr_papers 4602/2323,
  tcr_native 3848/7, trait 540/853; the 7 tcr_native negatives make that
  source's AUROC noise. `tcr_papers` rows still store `source="tcr_native"`;
  the script stratifies by shard name (the field itself is unfixed).
  Example (from repo root, `protenix_abtcr` env)::

      python scripts/downstream/score_binding_relation.py \
        --checkpoint <fusion_ckpt_dir> \
        --prepared-data-dir data/prepared/immune_v5_receptor_completion \
        --max-records 400 --device cuda
- `prepare_tcr_cdr_data.py`: rebuilds the current OTS TCR CDR infilling split
  used by the retained TCR CDR evaluator.
- `pairing_leakage_diagnostic.py`: checks generated pairing outputs for target
  length leakage, copying, and candidate collapse.

The underlying generation/evaluation implementation is in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/`.
It is an evaluation backend, not a training data layer; its checkpoint loader
accepts only current fusion checkpoint directories.

## Retired surface

The following were removed because they targeted the retired standalone
BioSeq grammar-v1/v2 trainer or the abandoned PPI/STRING/MINT corpus:

- `run_grammar*`, `gen_eval_grammar*`, and integrated grammar-v2 dispatchers;
- MINT/PPI embedding, summary, validation, and shard-specific evaluators;
- checkpoint-specific retry/sweep/watch scripts and generated historical jobs;
- the old humanization evaluator and one-off sampling probes.

Historical result files and parity evidence remain under the explicitly marked
`docs/archive/` and `refactor_baseline/` directories. They are provenance only,
not runnable evaluation entry points.
