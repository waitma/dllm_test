# Model Architecture Audit

> Does the implemented model match the intended design (`BIOSEQ_MODEL_PLAN.md`,
> `AGENTS.md`)? Verdict per component + the concrete design↔implementation gaps
> to fix. Code refs: `dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`,
> `.../data/grammar.py`, `.../training/trainer.py`.
>
> Last updated: 2026-09-12. Active grammar/masks live in
> `dllm/pipelines/immune_llada/data/grammar.py`, not the deleted
> `qwen3_vl_arch/data` tree. Receptor-completion / relation-target design:
> `docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`.

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
| Diffusion loss on the generated block; fixed context (antigen/MHC/peptide, unsupervised relation) and synthetic padding excluded | `grammar.py` emits an explicit `diffusion_eligible_mask = not fixed and not synthetic`, which `sample_bioseq_diffusion_noise` **trusts as-is** — it is *not* intersected with `residue_mask` unless `require_residue=True` (the corruption call passes `False`). So the generated block's grammar skeleton (`<prots>`, `<tcr>`/`<ab>`/`<nb>`, `<protd>`) and supervised relation targets are corrupted **and** counted in the loss, alongside residues; `compute_masked_cross_entropy` on `labels!=-100` | ✅ |
| Supervised `binding`/`nonbinding` relation is itself a denoising target; `unknown`/absent relation and pMHC presentation stay visible context | `relation_target_mask` drives it: target relation tokens get `is_fixed=False` and enter both corruption and loss; the MHC→peptide presentation `<binding>` and unsupervised relations keep `fixed_context_mask=1` | ✅ |
| beta-only completion padding (`X`) never supervised | `synthetic_residue_mask` is subtracted from both `diffusion_loss_mask` and `diffusion_eligible_mask` in `grammar.py`, so synthetic `X` is neither corrupted nor scored | ✅ |
| Trainable encoder with separate LR | YAML `--encoder-lr 2e-5` vs `--lr 1e-4`; encoder not frozen | ✅ |
| `condition_norm` to rescale tiny ESMC condition into the residual stream | integrated YAML sets `--condition-norm` (LayerNorm on condition); matches the plan's ESMC-scale note | ✅ |
| DDP shard by rank+worker, `num_workers=0/1` only | `mixture.py` shards by `rank*nw+worker`; `trainer.py` warns on nw>0; run uses nw=1 (keeps nw=0 shard identity) | ✅ |
| ≥1 diffusion-eligible token per record (no zero-loss microbatch) | `sample_bioseq_diffusion_noise` forces ≥1 masked token per row | ✅ |

Loss objective, corruption schedule (`t~U(eps,1)` absorbing-mask), weight tying,
RMSNorm/SwiGLU, and the checkpoint-driven downstream reload
(`load_grammar_checkpoint` rebuilds config from the ckpt's own args) are all
consistent with the design. No architectural deviation found.

## Design↔implementation gaps (to fix — B line)

### Gap 1 — resolved: TCR relation labels are honored and supervised
`grammar.py` `GrammarRenderer.encode`, TCR branch, now derives
`recognition_relation` from `record.labels["relation"]` (falling back to
`<binding>` only when the label is absent, keeping unlabeled pretraining pairs
byte-identical to the old behaviour). The MHC→peptide link stays presentation
`<binding>` and fixed; the peptide→TCR recognition token carries the
binding/nonbinding label and, when `relation_supervised` is set, becomes a
diffusion target (`relation_target_mask=1`, corrupted and scored) rather than
fixed context. A PISTE `nonbinding` pair therefore no longer renders identically
to a binder. Covered by `scripts/tests/immune_llada/test_beta_only_relation.py`
(`test_presentation_binding_stays_fixed_while_recognition_relation_is_target`,
`test_supervised_binding_relation_is_a_diffusion_target`,
`test_unknown_relation_is_unsupervised`).

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
