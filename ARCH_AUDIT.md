# Model Architecture Audit

> Does the implemented model match the intended design (`BIOSEQ_MODEL_PLAN.md`,
> `AGENTS.md`)? Verdict per component + the concrete design↔implementation gaps
> to fix. Code refs: `dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`,
> `.../data/grammar.py`, `.../training/trainer.py`.
>
> Last updated: 2026-07-09.

## Verdict: the active model matches the design

Active class = `BioSeqLLaDAEncoderDiffusionModel` (`--model-type llada`), used by
the integrated run. Design intent vs implementation:

| Design requirement (BIOSEQ_MODEL_PLAN / AGENTS) | Implementation | OK |
|---|---|:--:|
| Per-chain ESMC/ESM encoder on the current diffusion state `x_t` | `encode_chain_tokens` flattens `[B,C,L]`→`[B*C,L]`, one encoder forward, reshapes back; `compute_loss` mirrors decoder corruption onto encoder input (`apply_decoder_corruption_to_encoder`) | ✅ |
| Gathered encoder features **replace** decoder residue embeddings (no projection by default) | `inputs_embeds = wte(x_t)*(1-mask) + condition*mask` at residue positions; `use_condition_projection=False` default | ✅ |
| `hidden_size` forced to encoder latent dim when replacing | ctor raises unless `d_model == encoder_hidden_size` (or projection on) | ✅ |
| LLaDA bidirectional masked-diffusion denoiser (RoPE, no timestep) | `build_llada_backbone` sets `rope=True, alibi=False, is_causal=False`; timestep/pos args accepted but unused | ✅ |
| Grammar-v2 token stream; entity roles via boundary tokens (no chain-role/task embeddings) | decoder has no role/task embeddings; roles are grammar tokens | ✅ |
| Diffusion loss **only** on eligible target residues; fixed context (antigen/MHC/peptide/relation) excluded | `sample_bioseq_diffusion_noise` masks only `diffusion_loss_mask ∧ eligible ∧ residue_mask`; `compute_masked_cross_entropy` on `labels!=-100`; forbidden special-token logits suppressed | ✅ |
| Trainable encoder with separate LR | YAML `--encoder-lr 2e-5` vs `--lr 1e-4`; encoder not frozen | ✅ |
| `condition_norm` to rescale tiny ESMC condition into the residual stream | integrated YAML sets `--condition-norm` (LayerNorm on condition); matches the plan's ESMC-scale note | ✅ |
| DDP shard by rank+worker, `num_workers=0/1` only | `mixture.py` shards by `rank*nw+worker`; `trainer.py` warns on nw>0; run uses nw=1 (keeps nw=0 shard identity) | ✅ |
| ≥1 diffusion-eligible token per record (no zero-loss microbatch) | `sample_bioseq_diffusion_noise` forces ≥1 masked token per row | ✅ |

Loss objective, corruption schedule (`t~U(eps,1)` absorbing-mask), weight tying,
RMSNorm/SwiGLU, and the checkpoint-driven downstream reload
(`load_grammar_checkpoint` rebuilds config from the ckpt's own args) are all
consistent with the design. No architectural deviation found.

## Design↔implementation gaps (to fix — B line)

### Gap 1 — TCR-pMHC negative labels are swallowed (`<binding>` hardcoded)
`grammar.py` `GrammarRenderer.encode`, TCR branch: the relation token linking the
pMHC context to the TCR receptor is `special("<binding>", is_fixed=True)`
(lines ~436 and ~439) — it ignores `record.labels["relation"]`, so a PISTE
`nonbinding` pair renders identical to a binder. `_relation_token` and the
`<nonbinding>` vocab exist and are honored on the antibody-antigen branch (line
~408), so this is a TCR-branch omission. Impact: the pMHC→TCR relation (a fixed
conditioning token) carries no binding/nonbinding signal. **Fix = B1** (honor
the record relation for the peptide→TCR link; keep MHC→peptide as presentation
`<binding>`; default unlabeled pretraining pairs to binding for back-compat).

### Gap 2 — TCR-pMHC is CDR3-only, no full-length + no MHC/B2M
Current TCR sources are CDR3 fragments (or β-only), and PISTE ships a 34-aa HLA
pseudo-sequence, not the grammar's intended full-length five-entity layout
`<prots> MHC . B2M <protd> <binding> <prots> <pep> PEP <protd> <relation> <prots>
<tcr> A . B <protd>`. **Fix = B2**: rebuild full-length α/β (Stitchr + IMGTgeneDL)
and full MHC-I heavy + B2M (IMGT/HLA), prefer `complex.id`-paired records.

### Gap 3 — ESMC flash-attention disabled
The training encoder path (`input_ids`) supports flash-attn and the
`--encoder-use-flash-attn` flag is wired through `from_esmc(use_flash_attn=...)`,
but the integrated YAML does not set it, so the per-chain ESMC forward runs
without flash. **Fix = B3**: enable after a bf16 parity check (see
`SPEED_ANALYSIS.md` lever 1).

## Scope note
Per the user decision, all three gaps are fixed this round (B1→B2→B3) and folded
into the integrated retrain (B4). Larger method extensions (relpos, understanding
heads, REPA alignment, mixed corruption, flexibility-trap sampling) stay deferred
in `FUTURE_EXPERIMENTS.md`.
