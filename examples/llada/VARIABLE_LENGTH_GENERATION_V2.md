# Variable-length generation v2

> Status (2026-09-21): **implemented and committed, never trained.** Everything
> below is gated behind `--fixed_receptor_lengths` (default `False`); legacy
> checkpoints are byte-for-byte unaffected. CPU validation passed; two data
> gaps in "Known gaps" will abort a real training run and are still open.
> Baseline before this work: commit `c234b07`, pushed to `origin/main`.

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

## Verification performed (2026-09-21, CPU)

- `scripts/tests/immune_llada/` — 385 passed, 20 failed. All 20 failures
  reproduce on the baseline code after `git stash`, so none is a v2 regression:
  17 are `ModuleNotFoundError: nltk` in the TCR scoring/result-IO tests, 3 are
  the pre-existing legacy-vs-canonical mismatch in `test_full_parity.py`.
- `test_v2_inference_adapters.py` + `test_v2_checkpoint_policy.py` — 75 passed.
- Canvas layout smoke on antibody and TCR pair fixtures: `encoder_input_ids` is
  `[B, 2, 168]`, per-row decoder slots are `{chain 0: 167, chain 1: 135}`,
  `<chainsep>` appears exactly once between the two canvases, and the encoder
  attention mask is fully ones over each fixed canvas (`[168, 136]`), so chain
  length does not leak through the attention shape.
- Full read-only data audit over `immune_v6_binding_only` (7,381,499 rows) and
  `immune_v3_heterotypic` (7,997,971 rows) via
  `scripts/debug/audit_v2_canvas_data.py`; report in
  `logs/v2_canvas_data_audit/audit.json` (`status=complete`,
  `code_unchanged_during_scan=true`, `renderer_mapping_issue_rows=0`).

No GPU forward/backward, no smoke training run, and no loss curve exist for v2.

## Known gaps

Two of these will abort a real training run and are deliberately left open
pending a policy decision, because each possible fix changes what the model is
trained on:

1. **Receptor chains longer than the canvas.** 5 rows in `immune_v6_binding_only`
   exceed the fixed canvas (oas 1, ots 2, tcr_repertoire 1, trait 1; worst cases
   a 186-residue `antibody_heavy` and a 148-residue `tcr_alpha`). The renderer
   raises `ValueError`, and it runs inside the collator, which has no
   per-record skip path — so hitting one of these rows kills the job.
2. **Grammar length over `--max_length 1024`.** Padding both receptor chains to
   `167 + 135` slots inflates total grammar length, so 4,289 train and 286 valid
   rows in `immune_v6_binding_only` (5,008 + 2,033 in `immune_v3_heterotypic`)
   now exceed 1024; all of them are `asd_antibody` rows with long antigens. The
   prepared shards were filtered by `budget.max_length` under the *old*
   variable-length estimate, so the filter no longer holds. The collator raises
   `ValueError` rather than dropping the row.

   Combined: 4,580 / 7,381,499 rows (0.062%) for v6, 7,044 / 7,997,971 (0.088%)
   for v3_heterotypic.

Non-blocking follow-ups:

3. No v2 train job yml exists yet.
4. `tcr_generation_v5` (CDR3-only) and the CDR3-only mode of `tcr_generation`
   explicitly raise on a `fixed_receptor_lengths` checkpoint. Those two
   downstream evals are unavailable for v2 until they are ported to the
   full-length protocol.
5. Repeated EOS runs of a chain whose residues are partly synthetic (`X`
   completion, e.g. `tcr_repertoire` alpha) are marked synthetic wholesale and
   therefore excluded from EOS supervision, since a completed synthetic chain
   carries no observed termination label.
6. The ESMC native prediction-head auxiliary loss is still not implemented.
