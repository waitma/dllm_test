> ⚠️ **ARCHIVE / 非现行入口（2026-09-02）**：T4 现行入口 [`../tasks/TCR_T4_GENERATION.md`](../tasks/TCR_T4_GENERATION.md)。本文是 **Track-A 14×1000** 协议说明，**禁止**与 Setting-B K=100 或 sparse-13 合表。数字只写 [`RESULTS.md`](RESULTS.md) §0.4。

# Public epitope-conditioned CDR3β Track A

This is the authoritative first-stage protocol for comparing actual CDR3β
generation by TCRT5, GRATCR, beta-only TcrDesign, the local BioSeq 7L
step117000 checkpoint, and TCR-epiDiff. It supersedes
older T4 validity/ranking rules for this track. Paired αβ generation is a
separate Track B and is intentionally outside the quantitative ranking. A
post-benchmark TcrDesign generated-β→generated-α run is retained below as a
qualitative artifact only.

## Scope and model interface

Every de-novo adapter exposes:

```python
generate(
    peptide,
    mhc_allele=None,
    mhc_pseudosequence=None,
    num_candidates=1000,
)
```

The row-level output schema is:

```text
model,target_id,peptide,mhc_allele,rank,cdr3b,raw_score
```

Optional provenance columns are `generation_mode`, `source`, and `run_id`.
Raw candidates are retained in rank order. No C/F/W anchor is inserted and no
invalid sequence is repaired.

For TCRT5 specifically, `rank` follows the paper's cumulative model
log-likelihood ordering and `raw_score` is the sum of generated-token transition
log-probabilities. Hugging Face's length-normalized beam `sequences_scores` is
not used directly as the paper rank. Other models retain their official raw
ordering and score semantics, so `raw_score` is provenance rather than a
cross-model metric.

BioSeq does not expose a comparable candidate likelihood. Its `rank` is the
deterministic emission order under the recorded random seed and `raw_score` is
empty, not zero. The sampler uses only epitope conditioning; MHC fields remain
in the common table for target identity but are not passed to the model.

| Model | Conditioning | Runtime status | Track-A treatment |
|---|---|---|---|
| TCRT5 | peptide + 34-aa MHC pseudo-sequence | source, HF checkpoint, 10/1000/full-14 generation succeeded | comparable de novo |
| GRATCR | epitope | source, three official checkpoints, 10/1000/full-14 generation succeeded | comparable de novo |
| TcrDesign | epitope | only `tcrdesign_G.py -mode beta`; 10/1000/full-14 generation succeeded | comparable de novo beta-only |
| BioSeq-7L-step117000 | epitope | immutable local checkpoint, smoke/full-14 generation succeeded | comparable local de novo |
| TCR-epiDiff | epitope + real-TCR seed in released inference | source/checkpoint present, but no pure-random reverse sampler | `reconstruction_only`; excluded from ranking |

For TCR-epiDiff, the released inference assigns
`TCR = original_TCR[0]`, applies `add_noise(TCR, ...)`, and denoises that real
TCR tensor. The benchmark never supplies a test reference as a seed and never
turns this reconstruction path into a de-novo result.

## Public benchmark audit

The immutable reference field is `reference_translations` from
`benchmark_data_w_preds.csv`. Existing TCRT5/GRATCR/ER prediction columns are
explicitly excluded from references. The companion `IMMPREP_test_set.csv` is
audited for fields only and is not substituted for the 14-pMHC reference sets.

- 14 pMHC targets.
- 1,312 distinct target-specific CDR3β references in total.
- `RVRAYTYSK_HLA-A*03:01` has 895 distinct references.
- Every target has peptide, normalized HLA allele, 34-aa MHC pseudo-sequence,
  and an explicit reference count.

Machine-readable audit:
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_beta_public_benchmark/benchmark_audit.json`.

## Strict validity and metrics

A raw candidate is valid if and only if it contains only the 20 standard amino
acids, has length 7–24 aa, begins with `C`, and ends with `F` or `W`.

- `ValidRate = valid raw candidates / 1000`.
- `UniqueRate = distinct valid candidates / valid raw candidates`.
- `ExactHit@100` and `ExactHit@1000` count distinct generated sequences that
  exactly match a target reference within the raw-rank prefix.
- `ReferenceRecall@1000 = ExactHit@1000 / target reference count`.
- Recovery is `1 - Levenshtein / max(generated_length, reference_length)`;
  `Recovery90Hit@1000` counts distinct valid generated sequences whose best
  target-reference recovery is at least 0.9.
- GIANA clusters each model-target's distinct valid candidates together with
  that target's distinct references. `GIANAHit` counts distinct generated
  sequences in reference-containing clusters; reference-cluster and distinct
  reference coverage are reported separately. GIANA4.1 is run without V genes
  and with the same parameters for every model.
- Cross-epitope Jaccard is computed over each target's distinct valid sequence
  set. The full pairwise table is retained rather than reducing it to one number
  only.

Missing candidates count as invalid in the requested denominator. Reference
sets remain target-specific throughout all exact, recovery, and GIANA metrics.

## Reproduction commands

Prepare and audit:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py prepare
```

Generate all 14 targets with TCRT5:

```bash
LD_LIBRARY_PATH=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib:/usr/local/cuda/lib64 \
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
  generate --model tcrt5 --targets all --num-candidates 1000 \
  --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv \
  --run-id full14x1000_paper_loglik_rank
```

Append GRATCR and beta-only TcrDesign by changing `--model` to `gratcr` and
`tcrdesign` and adding `--append`. Append the retained BioSeq checkpoint:

```bash
LD_LIBRARY_PATH=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib:/usr/local/cuda/lib64 \
PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test \
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
  generate-bioseq \
  --checkpoint /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt \
  --targets all --num-candidates 1000 --seed 42 --batch-size 100 \
  --max-iter 32 --temperature 1.0 --sampling-strategy gumbel_argmax \
  --run-id full14x1000_step117000_seed42_iter32 \
  --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv \
  --results-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results \
  --append
```

Score all metrics:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
  score \
  --generations /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv \
  --results-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results \
  --num-candidates 1000 --with-giana
```

Audit TCRT5 against the authors' released generations and the paper values:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/audit_tcrt5_paper.py \
  --generations /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv \
  --per-target /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/per_target_metrics.csv \
  --output-csv /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.csv \
  --output-md /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.md
```

The retained pre-fix table can be converted reproducibly with
`migrate-tcrt5-paper-ranking`; the original length-normalized-score table is
preserved as `generations.pre_tcrt5_paper_rank.csv` in the result directory.

Protocol regression test:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m pytest -q \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_public_beta_benchmark.py
```

## Completed local result, 2026-07-21

All three public comparable models and the retained local BioSeq checkpoint
were run for 14 × 1,000 candidates. No author-released prediction rows are in
the final 56,000-row generation table.

| Model | ValidRate | UniqueRate | ExactHit@1000 | ReferenceRecall | Recovery90Hit | GIANAHit |
|---|---:|---:|---:|---:|---:|---:|
| BioSeq-7L-step117000 | 0.998786 | 0.999714 | 0 | 0.000000 | 0 | 1 |
| GRATCR | 1.000000 | 0.074214 | 0 | 0.000000 | 7 | 4 |
| TCRT5 | 1.000000 | 1.000000 | 10 | 0.007622 | 235 | 335 |
| TcrDesign | 0.999643 | 1.000000 | 4 | 0.003049 | 197 | 402 |

`ReferenceRecall` above is the micro total: summed distinct exact hits divided
by the 1,312 target-specific references. `GIANAHit` is also summed over targets.
TcrDesign's larger total GIANA hit is concentrated in some targets (notably
TDLGQNLLY with 289); it should not be read as a universal per-target win.

For the requested RVR target:

| Model | ValidRate | UniqueRate | ExactHit@1000 | ReferenceRecall@1000 | Recovery90Hit@1000 | GIANAHit |
|---|---:|---:|---:|---:|---:|---:|
| BioSeq-7L-step117000 | 1.000 | 1.000 | 0 | 0.000000 | 0 | 0 |
| GRATCR | 1.000 | 0.087 | 0 | 0.000000 | 2 | 4 |
| TCRT5 | 1.000 | 1.000 | 8 | 0.008939 | 166 | 231 |
| TcrDesign | 1.000 | 1.000 | 1 | 0.001117 | 108 | 103 |

The result supports a metric-specific conclusion: TCRT5 leads exact and ≥90%
recovery; TcrDesign has the largest summed GIANA hit; BioSeq has very high raw
validity/diversity and the lowest mean cross-target Jaccard, but essentially no
target-specific reference recovery; GRATCR suffers severe beam duplication and
does not recover an exact reference. TCR-epiDiff is not ranked because its
released path is reconstruction-only.

## BioSeq 7L step117000 diagnostic

The selected checkpoint is the retained Top-K `step117000` snapshot, which has
validation loss 0.4430236166 and SHA256
`e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`.
The run used seed 42, batch size 100, 32 masked-diffusion iterations,
temperature 1.0, and `gumbel_argmax`. All target-chain residue positions were
generated; the placeholder only sets candidate length, and no C/F/W anchor was
kept visible or repaired afterward.

Of 14,000 raw sequences, 13,983 are valid. All 17 invalid rows fail only the
terminal F/W rule. Within-target uniqueness is 13,979/13,983. Only 8 of 91
cross-target comparisons share even one sequence, giving mean Jaccard
0.000044 and maximum 0.000501.

This diversity does not translate into specificity recovery: exact hit and
Recovery≥90% are both zero on every target. The sole GIANA hit is
`CALSESGGGELDF`, clustered with reference `CASSESGAGELFF` for
`HPNGYKSLSTL_HLA-B*07:02`. On RVR, the best two normalized-Levenshtein
recoveries are 0.785714 (`CASSDGGAGELTF` versus `CASSRLGGAGELFF`) and 0.785714
(`CASSVGGQGGELTF` versus `CASSGGQGGKLFF`); median recovery is 0.588235.

The result therefore indicates a specific failure mode for step117000: it has
learned a strong CDR3β-like syntax and broad sequence diversity, but the current
epitope-only sampling interface does not recover the benchmark's
epitope-specific reference neighborhoods. The focused report is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_report.md`.

## TCRT5 paper-consistency audit

The local official-weight TCRT5 candidate set matches the authors' released
prediction artifact exactly for every target: 14,000/14,000 sequences, with a
set Jaccard of 1.0 in all 14 target blocks. After ranking by cumulative
log-likelihood, 13,990/14,000 integer ranks are identical; the remaining 10 are
adjacent floating-point near-ties outside every top-100 prefix, and all 14
top-100 sets are identical.

The canonical TCRT5 rows come from the completed clean 14-target rerun after
the adapter correction, with direct transition-score sums. The migration tool
and pre-fix backup are retained only for auditability.

| Paper check | Local result | Paper result | Agreement |
|---|---:|---:|---|
| Sparse 13-pMHC exact hits @1,000 | 2 | 2 | yes |
| RVR exact hits @1,000 | 8 | 8 | yes |
| RVR GIANA clustered generations (`t`) | 231 | 231 | yes |
| RVR reference-containing clusters (`c`) | 23 | 23 | yes |

The two sparse-set exact hits occur for
`FTDALGIDEY_HLA-A*01:01` and `HPNGYKSLSTL_HLA-B*07:02`, as stated in the
paper. The eight RVR exact-hit ranks are 11, 17, 83, 258, 259, 430, 535, and
603, matching the released artifact. The all-14 summary above reports 10 exact
hits because it combines the paper's sparse 13-target set (2) with its separate
RVR deep-simulation target (8); the paper does not treat those as one aggregate.

Recovery must be kept definition-specific. On RVR, the paper's same-length
identity/Hamming rule gives 109 candidates at ≥90%, while the user-specified
Track-A normalized-Levenshtein rule gives 166. This difference is expected and
is not a failed reproduction. Figure 5's `c=23,t=231,r=8` labels reproduce
exactly. The article prose calls 23 unique reference sequences, but the released
GIANA output contains 33 distinct reference rows in those 23 clusters; the
reproducible meaning of `c=23` is therefore reference-containing cluster count.

The detailed comparison is stored in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.md`
and its per-target CSV companion. A direct compatibility check also found
identical RVR sequences, order, and scores under the repository-pinned
Transformers 4.40.2 and the local 4.48.1 runtime, excluding that version change
as the source of the former rank discrepancy.

## TcrDesign generated-β-conditioned α addendum

After completing the public β leaderboard, the released TcrDesign paired CDR3
path was run end to end:

```text
epitope -> generated CDR3β
(epitope, generated CDR3β) -> generated CDR3α
```

All 14,000 locally generated TcrDesign β rows (14 targets × 1,000 raw ranks)
were used as conditions. Ten α beams were decoded per β with the official
`beta_alpha/RNN_tcr.ep` checkpoint, temperature 0.8, maximum length 20, and
official beam search, yielding 140,000 ordered αβ rows. No reference/oracle β
was supplied. A one-condition and a two-condition batched smoke test both
matched `tcrdesign_G.py -mode alpha` exactly, including sequence order.

The adapter loads the released model once and batches conditions; it does not
change the model, checkpoint, tokenization, beam search, cleanup, or candidate
order. It does not filter, repair, deduplicate, or score α sequences.

Official-release audit outcome:

- Alpha generation: available.
- Official alpha evaluation: unavailable.
- Included in quantitative benchmark: no.

The paper and supplement show αCDR3 BLOSUM62 similarity, minimum edit distance
to natural αCDR3, and TcrDesign-B binding score. However, the public repository
and Zenodo archives do not contain a directly runnable alpha-reference figure
script/CLI with its parameters. The repository's `clip/` directory contains an
in-batch α/β contrastive training model and expects CSV columns `CDR3a,CDR3b`,
but no pairing checkpoint, evaluation CLI, retrieval metric, or pairing-score
release. TcrDesign-B is a pMHC/full-TCR binding model, not a common independent
αβ pairing evaluator. Therefore no custom α exact-hit, recovery, BLOSUM62, or
pairing metric was invented, and the β leaderboard above is unchanged.

Reproduction command:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_public_beta.py \
  generate-tcrdesign-alpha \
  --generations /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv \
  --targets all --beta-top-k 1000 --alpha-per-beta 10 \
  --temperature 0.8 --max-len 20 --batch-size 64 --device cuda \
  --run-id full14x1000beta_x10alpha \
  --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv \
  --results-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results
```

## BioSeq step189000 RVR checkpoint diagnostic

The retained 7-layer step189000 checkpoint was tested on the focal
`RVRAYTYSK_HLA-A*03:01` target under the same BioSeq protocol as step117000:
epitope-only input `RVRAYTYSK`, seed 42, batch 100, 32 iterations, temperature
1.0, `gumbel_argmax`, and no raw-sequence repair. This is an isolated
single-target checkpoint diagnostic, not a replacement for the canonical
all-14 step117000 row.

| Model | ValidRate | UniqueRate | ExactHit@1000 | ReferenceRecall | Recovery90Hit | Median recovery | GIANAHit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BioSeq-7L-step117000 | 1.000 | 1.000 | 0 | 0.000 | 0 | 0.588235 | 0 |
| BioSeq-7L-step189000 | 0.998 | 1.000 | 0 | 0.000 | 0 | 0.600000 | 0 |

Step189000's maximum best-reference recovery is `0.800000`; neither checkpoint
produces an exact hit, a >=90% recovery hit, or a GIANA reference-cluster hit.
The two legal unique candidate sets have zero intersection (`Jaccard=0`). This
same-target checkpoint overlap is not the cross-epitope Jaccard metric; the
latter is undefined for a one-target run. The canonical 56,000-row generation
table is unchanged.

Focused report:

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/checkpoint_comparison/bioseq_step189000_rvr_epitope_only/report.md`

## Result paths

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/per_target_metrics.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/summary_metrics.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/cross_epitope_jaccard.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/reproduction_status.md`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_report.md`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_generations.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_per_target_metrics.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_top_recoveries.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_giana_hits.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.md`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrt5_paper_consistency.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/generations.pre_tcrt5_paper_rank.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_evaluation_audit.json`

## Complete TCRT5-eval addendum（2026-07-23）

The canonical Track-A candidate file is now also evaluated by the full
paper-facing suite at
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/current_track_a`.
This addendum does not overwrite the leaderboard's normalized-Levenshtein
Recovery90 metric. It adds the paper-native two-stage recovery, ranked AP,
length, positional entropy, k-mer JSD, Pgen and polyspecificity views.

| Model | Exact | paper-native ≥90% | GIANA hit | Global unique | AP(P@1…1000) | rank evidence | Pgen+ | mean log10 Pgen+ |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| BioSeq-7L-step117000 | 0 | 0 | 1 | 13,988 | 0.000000 | diagnostic | 0.9798 | -17.9253 |
| GRATCR | 0 | 4 | 4 | 629 | 0.000000 | diagnostic | 1.0000 | -6.8664 |
| TCRT5 | 10 | 154 | 335 | 7,788 | 0.001379 | cumulative-likelihood exact | 1.0000 | -6.9504 |
| TcrDesign | 4 | 142 | 402 | 13,083 | 0.000772 | diagnostic | 0.9928 | -8.8010 |

On RVR specifically, TCRT5 has 8 exact hits, paper-native Recovery90 count
109, AP `0.015847`, and GIANA `t=231,c=23`; its exact ranks remain
11, 17, 83, 258, 259, 430, 535 and 603. The old Track-A normalized-Levenshtein
RVR Recovery90 count remains 166 and is intentionally not replaced.

The complete paper correspondence—including the 240-block author-release
regression, Supplementary Table 2 comparison and explicit mAP/Pgen
non-reproducibility boundaries—is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`.
