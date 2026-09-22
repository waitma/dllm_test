# Variable-length generation v2

> Status (2026-09-21): **implemented, committed (`cf36477` + `955a1ec` on
> `origin/main`), CPU smoke-trained end to end, retested in
> `conda/envs/protenix_abtcr` (411 passed / 3 pre-existing failed), not yet
> trained for real.** Everything below is gated behind
> `--fixed_receptor_lengths` (default `False`); legacy checkpoints are
> byte-for-byte unaffected. Baseline before this work: commit `c234b07`,
> pushed to `origin/main`.

## Decoder-only ablation (2026-09-22)

v2 also supports `--residue_cond_mode token`: new training skips ESMC weight
loading and constructs no encoder, feature projection, or condition norm. The
same tokenizer assets, grammar, fixed decoder canvases, EOS targets, corruption,
and loss remain; the decoder receives only `wte(x_t)`. This is not merely a frozen
or bypassed encoder occupying the checkpoint. Feature replacement is still
forbidden in v2. Existing encoder-bearing token checkpoints remain loadable for
evaluation, but cannot be resumed into the new encoder-free training architecture.

Both 2-node × 4-GPU configs now exist (not submitted):

- [Fusion, 200k steps](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu.yml).
- [Decoder-only, proposed 1M steps](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml).

The latter extends the cosine horizon without changing peak LR or warmup steps.
A longer schedule does not guarantee matching pretrained-encoder performance;
comparisons must report both step and compute budgets. The historical validation
and gaps below refer to the initial September 21 implementation.

## Scope

The v2 recipe uses a fixed encoder canvas while retaining variable-length decoder
outputs for antibody and TCR receptor chains:

- heavy / beta: ESMC length `168` including `<cls>`, decoder canvas `167` slots;
- light / alpha: ESMC length `136` including `<cls>`, decoder canvas `135` slots;
- the same heavy/light canvas is used for antibody and TCR;
- one decoder slot is reserved for the first EOS, so the practical maximum
  residue count is `166` for heavy/beta and `134` for light/alpha.

The renderer emits real residues followed by repeated EOS tokens. EOS is an
attention-visible token, not padding. The decoder has a separate
`chain_eos_mask`; EOS slots are eligible for corruption and loss, but are not
included in `residue_mask` or the amino-acid ESMC residue-slot mapping.

## Data and model flow

```text
BioSeqRecord
  -> GrammarRenderer fixed decoder canvas
  -> [B,S] grammar ids + residue_mask + chain_eos_mask + chain ids
  -> [B,C,L] ESMC streams
       heavy/beta: <cls> residues <eos> <eos> ...  (L=168)
       light/alpha: <cls> residues <eos> <eos> ...  (L=136)
  -> ESMC encode [B*C,L] -> [B,C,L,E]
  -> gather amino-acid conditions onto decoder residue slots
  -> additive fusion: wte(x_t) + condition_proj(ESMC) * condition_mask
  -> LLaDA bidirectional denoising logits [B,S,V]
```

Flattening `[B,C,L]` to `[B*C,L]` is a batch operation, not concatenation:
each chain is an independent ESMC batch row and therefore self-attention cannot
cross chain boundaries. Cross-chain interaction happens in the decoder's flat
grammar stream.

## EOS and padding

Decoder EOS is supervised through the normal diffusion/MLM loss. v2 enables EOS
as a legal target with `predict_eos=True`; legacy configs continue to forbid EOS
as a denoising target. The local LLaDA tokenizer aliases native EOS and PAD, so
v2 adds `<chain_eos>` when necessary and records the policy in
`fusion_config.json`. Legacy checkpoints keep their old alias behavior.

Repeated EOS positions remain in decoder attention and the fixed ESMC stream.
They are not treated as encoder amino-acid residues, so they cannot shift the
residue condition alignment or leak a target through the ESMC path.

## Leakage contract

During training, every corrupted decoder residue is mirrored as encoder MASK
before ESMC encoding. During inference, pending generated residues remain MASK
and only committed generated residues are copied into the ESMC stream. The
reference suffix is never used to choose the v2 canvas length or to construct
encoder feedback. After the final denoising step, downstream extraction returns
only residues before each chain's first decoder EOS; the sampler does not stop
one chain early and does not remove the fixed canvas during denoising.

## Compatibility and files

- `dllm/pipelines/immune_llada/data/grammar.py`: fixed renderer/collator,
  EOS masks, fixed ESMC canvases.
- `examples/llada/protein_fusion_model.py`: additive fusion, EOS policy,
  configurable EOS loss target.
- `examples/llada/protein_pretrain_esmc.py`: v2 training defaults and sidecar
  configuration writing.
- `examples/llada/load_fusion_checkpoint.py`: sidecar-aware reconstruction,
  including fixed-canvas and EOS policy.
- `dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py`: generation state remains
  leakage-safe; first-EOS truncation is downstream extraction behavior.
- `downstream/grammar/light_chain_pairing.py`: v2 pairing uses the fixed light
  canvas instead of reference length or a sampled prior.
- `downstream/grammar/metrics.py`: first-EOS output truncation.

The optional native ESMC prediction-head auxiliary loss is intentionally not part
of the minimum v2 change. If enabled later, it must target only clean amino-acid
slots (excluding CLS, EOS, and PAD), and a frozen ESMC cannot receive gradient
updates from that loss.

## Decoder chain EOS token

The local LLaDA tokenizer resolves `eos_token_id == pad_token_id == 126081`
(`<|endoftext|>`), and `_load_llada_tokenizer` assigns `pad_token = eos_token`
when `pad_token` is unset. Reusing that id as the chain terminator would make
"this chain ended" indistinguishable from "this position is padding", both in
the decoder targets and in the inverse grammar remap. v2 therefore adds a
dedicated `<chain_eos>` special token and calls `resize_token_embeddings`.

`expand_llada_tokenizer_for_esmc_grammar(..., allow_chain_eos_token=True)`
records the outcome on the tokenizer as `_fusion_decoder_chain_eos_token_id` and
`_fusion_decoder_chain_eos_policy`, which takes one of three values:

- `native_eos` — native EOS is already distinct from PAD/MASK, so it is reused
  and the vocabulary does not grow. Not the case for the current LLaDA weights.
- `chain_eos` — the v2 path with the current LLaDA tokenizer: `<chain_eos>` is
  added and the vocabulary grows by one.
- `legacy_eos_alias` — legacy checkpoints only. The ambiguity is retained
  deliberately, because growing the vocabulary would invalidate old weights.

Both ids and the policy string are persisted in `fusion_config.json` and are
re-checked on resume and on eval load; a mismatch is a hard error rather than a
silent reinterpretation of the terminator.

## Rows the fixed canvas cannot render

Padding both receptor chains to `167 + 135` slots makes a small tail of the
prepared corpus unrenderable, in two ways:

- a receptor chain longer than its canvas (`ReceptorCanvasOverflow`), and
- a row whose padded grammar length now exceeds `max_sequence_length`, even
  though it passed the build-time `budget.max_length` filter under the old
  variable-length estimate.

The prepared shards cannot be re-filtered without rebuilding the 13 GB corpus,
and the renderer runs inside the collator, which has no per-record skip path.
`GrammarBioSeqCollator(drop_overflow_records=True)` therefore drops and counts
these rows; `protein_pretrain_esmc.py` enables it exactly when
`--fixed_receptor_lengths` is on, so legacy runs still raise.

Counts land in `collator.drop_counts` / `collator.drop_counts_by_source`, keyed
by `canvas.receptor_overflow` and `budget.max_length` — the latter deliberately
reuses the preprocessing filter name so the runtime ledger is comparable with
`filter_report.json`. `CanvasDropLogCallback` mirrors them into the Trainer logs
as `canvas_drop/*`, and the collator logs a warning with exponential backoff.
Counts are per-process and cover only the shards a rank actually read, so they
are a monitoring signal, not a corpus-wide filter report.

A batch in which *every* record was dropped raises rather than reaching the
model as a silent no-op. At the measured rate this is ~1e-13 per batch of four.

## Verification performed (2026-09-21, CPU)

- Retest in the requested env
  `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr`
  (Python 3.12.12, torch 2.8.0, nltk 3.9.4): `scripts/tests/immune_llada/`
  **411 passed / 3 failed** (`logs/v2_abtcr_pytest/`, 25s). The remaining 3
  are `test_full_parity.py` (`KeyError: 'prepared'` / `StopIteration`); they
  fail on the pre-v2 baseline too.
- First test pass was **not** in this env and reported 394 passed / 20 failed:
  17 extra failures were `ModuleNotFoundError: nltk` in the TCR
  scoring/result-IO tests. Those 17 pass in `protenix_abtcr`.
- `test_v2_inference_adapters.py`, `test_v2_checkpoint_policy.py`, and
  `test_v2_canvas_overflow.py` — 84 passed in `protenix_abtcr`.
- Canvas layout smoke on antibody and TCR pair fixtures: `encoder_input_ids` is
  `[B, 2, 168]`, per-row decoder slots are `{chain 0: 167, chain 1: 135}`,
  `<chainsep>` appears exactly once between the two canvases, and the encoder
  attention mask is fully ones over each fixed canvas (`[168, 136]`), so chain
  length does not leak through the attention shape.
- Full read-only data audit over `immune_v6_binding_only` (7,381,499 rows) and
  `immune_v3_heterotypic` (7,997,971 rows) via
  `scripts/debug/audit_v2_canvas_data.py`; report in
  `logs/v2_canvas_data_audit/audit.json` (`status=complete`,
  `code_unchanged_during_scan=true`, `renderer_mapping_issue_rows=0`). Overflow
  counts: 5 receptor-canvas rows and 4,575 over-budget rows in v6
  (4,580 / 7,381,499 = 0.062%); 7,044 / 7,997,971 (0.088%) in v3_heterotypic.
  All over-budget rows are `asd_antibody` with long antigens.
- Drop path against real corpus rows: a 223-row smoke corpus spliced with the
  three offending rows the audit named (`oas-00017:8523`, a 186-residue heavy;
  `ots-00008:69012`, a 148-residue alpha; `asd_antibody-00000:1695`, grammar
  length 1029) keeps 220 / 223 with
  `{canvas.receptor_overflow: 2, budget.max_length: 1}` attributed to the right
  sources, and aborts exactly 3 batches with the flag off.
- **End-to-end CPU smoke training** (`logs/v2_fixed_canvas_smoke/`, `rc=0`):
  12 steps on a shrunken decoder (d256/L2/h4) over all seven sources with
  `--fixed_receptor_lengths True --residue_cond_mode add --freeze_encoder False`.
  Loss 4.22 → ~3.1 with finite grad norms, per-source eval on all seven sources,
  checkpoints at 6/12/final. The written `fusion_config.json` carries
  `fixed_receptor_lengths=true`, `predict_eos=true`, `residue_cond_mode="add"`,
  `decoder_chain_eos_policy="chain_eos"`, `decoder_chain_eos_token_id=126389`
  (distinct from pad/eos `126081` and mask `126336`), and the six canvas
  constants.
- Reloading that checkpoint through `load_fusion_for_eval` restores the policy,
  `resolve_light_length_mode(model, "auto")` returns `fixed_v2`, the collator
  emits `[1, 2, 168]` with `encoder_slot_mask`, and denoising all 135 light
  canvas slots yields a variable-length chain (18 residues for the antibody
  fixture, 2 for the TCR fixture) terminated by a generated EOS. The sequences
  are meaningless after 12 steps; what this shows is that output length is set
  by the model, not by a reference or a sampled prior.

The smoke ran on CPU because the local A100 is unusable: torch `2.11.0+cu130`
needs a newer driver than the installed `535.129.03` (CUDA 12.2). The vendored
`LLaDAModelLM.__init__` also hard-codes `init_device = "cuda"`, which was
monkey-patched in the temporary harness rather than changed in the repo. No
real-scale (d768/L8) GPU run or loss curve exists yet.

## Known gaps

1. v2 train YAMLs now exist (see above); real-scale GPU training has not been validated.
2. `tcr_generation_v5` (CDR3-only) and the CDR3-only mode of `tcr_generation`
   explicitly raise on a `fixed_receptor_lengths` checkpoint. Those two
   downstream evals are unavailable for v2 until they are ported to the
   full-length protocol.
3. Repeated EOS runs of a chain whose residues are partly synthetic (`X`
   completion, e.g. `tcr_repertoire` alpha) are marked synthetic wholesale and
   therefore excluded from EOS supervision, since a completed synthetic chain
   carries no observed termination label.
4. The ESMC native prediction-head auxiliary loss is still not implemented.
5. Dropping over-budget `asd_antibody` rows removes the longest-antigen tail
   from training. It is 1.9% of that source rather than 0.06% of the corpus, so
   antigen-conditioned metrics on very long antigens are worth watching.
