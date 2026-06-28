# BioSeq Grammar (v2)

This documents the **active grammar** used by the BioSeq foundation training input
path for joint biological sequence generation. The Arrow cache directory is still
named `data/bioseq_grammar_v1` for historical/path compatibility; `GrammarRenderer`
applies the v2 layout at encode time.

Implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/grammar.py`.

## Tokens

- **Structure tokens** (appended after the base vocab): `<ab>`, `<tcr>`, `<nb>`,
  `<pep>`, `<prots>`, `<protd>`.
- **Relation tokens**: `binding`, `activation`, `inhibition`, `catalysis`,
  `reaction`, `expression`, `ptmod`, `neutralization`, `nonbinding`, `unknown`
  (rendered as `<binding>` … `<unknown>`). Sources without a typed relationship
  use `<unknown>`.
- **Residue tokens** come from the active base tokenizer (ESM2 / ESMC).
- **Chain separator**: the literal `.` from the base vocabulary (ESM2 id 29). It
  is classified as a structure token and receives a learned embedding; it joins
  multiple chains inside one `<prots>…<protd>` block.

Legacy tokens are **not** used: v1 `<fixs>`, `<fixd>`, `<generate>`, `<prote>`,
`<pairs>`; v2 Round-1 extras `<protbs>`, `<protbd>`, `<chainsep>`, `<peptides>`,
`<peptided>`.

Default `GrammarTokenizer(Esm2SequenceTokenizer())` vocab size: **49** (33 base +
16 grammar). With ESMC tokenizer: **80** (64 base + 16 grammar).

## Record Forms

Type markers sit **inside** `<prots>…<protd>`. Multi-chain objects use a single
block with `.` between chains. Separate objects (context vs target, PPI partners)
use separate `<prots>…<protd>` blocks.

```text
OAS (antibody):       <prots> <ab> HEAVY . LIGHT <protd>
OTS (TCR):            <prots> <tcr> ALPHA . BETA <protd>
Nanobody:             <prots> <nb> VHH <protd>
TCR+peptide:          <prots> <pep> PEPTIDE <protd> <binding> <prots> <tcr> ALPHA . BETA <protd>
TCR-pMHC:             <prots> MHC . B2M <protd> <binding> <prots> <pep> PEPTIDE <protd> <binding> <prots> <tcr> ALPHA . BETA <protd>
PPI (conditional):    <prots> PROTEIN_A <protd> <REL> <prots> PROTEIN_B <protd>
AB-antigen:           <prots> ANTIGEN <protd> <binding> <prots> <ab> HEAVY . LIGHT <protd>
NB-antigen:           <prots> ANTIGEN <protd> <binding> <prots> <nb> VHH <protd>
```

- Chain identity (heavy vs light, alpha vs beta, partner A vs B) is expressed by
  `position_ids_chain` / `chain_ids` embedding indices plus object boundaries,
  not by per-role token names.
- `<REL>` is the inferred relation token for the PPI edge.
- Antigen-conditioned tasks distinguish antibody vs nanobody via `<ab>` vs `<nb>`
  inside the generated receptor block.

## Denoising (fixed vs generated)

The renderer marks `fixed_context_mask`; `diffusion_loss_mask = NOT fixed`.
`diffusion_eligible_mask` equals `diffusion_loss_mask` (structure tokens and
residues in generated regions are both eligible).

### Masking rules by task

| Task | Fixed (not diffused) | Generated (diffused, including type markers) |
|------|----------------------|-----------------------------------------------|
| OAS / OTS / nanobody | — | Entire `<prots> <type> … <protd>` block |
| AB-antigen / NB-antigen | `<prots> ANTIGEN <protd> <binding>` | `<prots> <ab>/<nb> H . L <protd>` |
| TCR+peptide | `<prots> <pep> PEPTIDE <protd> <binding>` | `<prots> <tcr> α . β <protd>` |
| TCR-pMHC | MHC block + `<binding>` + peptide block + `<binding>` | `<prots> <tcr> α . β <protd>` |
| PPI conditional | A block + `<REL>` | B block (no type marker) |

Details:

- **Relation tokens** (`<binding>`, `<activation>`, …) in conditional tasks are
  always fixed.
- **Type markers** (`<ab>`, `<tcr>`, `<nb>`, `<pep>`) are fixed **only inside
  fixed context blocks** (e.g. `<pep>` in a fixed peptide block). Inside a
  **generated** block they participate in diffusion like any other structure
  token (OAS `<ab>`, receptor `<ab>`/`<nb>`/`<tcr>`, etc.).
- **Unconditional** tasks (OAS, OTS, nanobody) have no fixed context; the
  whole record is generated.

### TCR-pMHC example

```text
<prots> MHC . B2M <protd> <binding>   ← fixed (3 encoder chains: MHC, B2M, —)
<prots> <pep> PEPTIDE <protd> <binding> ← fixed (1 encoder chain; <pep> fixed)
<prots> <tcr> ALPHA . BETA <protd>      ← generated (2 encoder chains; <tcr> diffused)
```

### Noise sampling

Per sequence, sample `t ~ Uniform(ε, 1)`; each **eligible** token is masked
independently with probability `t` (`sample_bioseq_diffusion_noise`). Fixed
context is never masked. This is token-level masking, not chain-level
all-or-nothing.

PPI training is **conditional only**: protein A block and `<REL>` are fixed;
protein B block is generated. There is no `ppi_joint` mode.

All token classes use the same token-normalized cross-entropy. Training logs
expose separate residue, structure-token, and relation-token losses
(`TOKEN_CLASS_NAMES`).

Padding is applied after the complete record; records are never truncated after
serialization. STRING PPI / MINT pairs with either protein longer than **1024**
residues are **dropped** at Arrow shard build time (not cropped); see
``grammar_builders.ppi_record`` and ``build_mint_grammar_shards.py``.

## Decoder ↔ Encoder data flow

Training step (encoder models):

```text
BioSeqRecord (Arrow)
  → GrammarRenderer.encode     flat decoder stream + fixed/diffusion masks
  → GrammarBioSeqCollator      per-chain encoder_input_ids [B,C,L], chain_ids, position_ids_inner
  → sample_bioseq_diffusion_noise   decoder x_t (per-token mask on eligible positions)
  → apply_decoder_corruption_to_encoder   mirror mask onto encoder residue positions
  → encoder forward            [B,C,L] → [B,C,L,E] (ESMC/ESM2 per chain)
  → gather_token_condition     scatter residue features to decoder positions [B,S,E]
  → decoder forward            embedding replacement at residue sites only
```

- **Decoder**: one flat grammar token stream per record.
- **Encoder**: each biological chain → `<cls> + residues + <eos>` (split on `.`
  inside a block and on `<protd>` between blocks). Padding chain rows use a
  minimal valid `<cls><eos>` stream; `encoder_chain_mask` zeroes their condition.
- **Corruption mirror**: if decoder residue at chain `c`, inner index `i` is
  masked, the matching encoder residue slot is set to `<mask>`. Fixed context
  chains stay clean on both sides.
- **No encoder signal** on structure/relation tokens (`<prots>`, `<ab>`,
  `<binding>`, `.`, etc.): `gather_token_condition` only fills residue
  positions; others keep learned decoder embeddings.

### Encoder align fix (2026-06-28)

v2 per-chain collator originally rebuilt encoder chains via amino-acid string
round-trip (`_residue_ids_to_sequence`). Local ESMC/ESM2 adapters loaded through
`HuggingFaceEsmTokenizerAdapter` did not expose `id_to_token`, so every chain
silently became all-`X`. Fixed by `_encode_chain_from_residue_ids`: copy decoder
residue ids directly into `<cls> residues <eos>`. Grammar-v1 proxy stream did
not have this bug (it mapped decoder ids directly).

## Encoder Mode (per-chain, embedding replacement)

Encoder variants (`--model-type encoder` for ESMC, `--model-type esm2` for ESM2)
encode **each chain independently**:

- The collator emits `encoder_input_ids` of shape `[batch, max_chains, chain_len]`,
  one `<cls> seq <eos>` stream per chain (chains split on `.` and `<protd>`).
- Corrupted target residues are masked in `encoder_input_ids` before the encoder
  forward. Fixed context chains stay clean.
- Per-residue encoder features are gathered to decoder positions via `chain_ids` +
  `position_ids_inner`, then **replace** the decoder residue token embeddings
  (no condition projection by default). Special/structure/relation tokens keep
  learned embeddings.
- `decoder.hidden_size` must match the encoder latent dim (e.g. ESMC-300M → 960).
  For no-encoder ablations, pass `--align-hidden-size-to-encoder` to match the
  same width.

Grammar boundary tokens are the sole entity-role representation; grammar models
do not instantiate chain-role embeddings.

## Data

Build semantic Arrow shards with:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_bioseq_grammar_v1.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/bioseq_grammar_v1 \
  --splits train,valid \
  --sources oas,ots,tcr,ppi,mint_ppi,mint_actions
```

The cache includes paired OAS, paired OTS, non-PPI TCR/epitope records from
`processed_v2`, canonicalized STRING PPI pairs (Bernett 90/90), **v12 MINT
binding** (`mint_ppi`, ~96M train), and **v11 actions modes** (`mint_actions`,
~9.2M train). Rebuild mint shards::

```bash
bash /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/rebuild_mint_training_shards.sh
```

- **`mint_ppi`**: v12 physical binding from `mint_string_pretrain_v1` — relation
  token is always `<binding>`.
- **`mint_actions`**: v11 functional/regulatory edges from `mint_string_actions_v11.0`
  — relation token equals the STRING **mode** (`<catalysis>`, `<activation>`, etc.).

Antibody-antigen / nanobody-antigen records are supported by the renderer but are
not in the default training mix yet.

### STRING actions mode and direction (v11)

Actions edges come from `protein.actions.v11.0` (separate from physical binding
links). Split files use three columns:

```text
target_id  actor_id  mode
```

Direction follows STRING `a_is_acting=t` rows: **actor** is the acting protein,
**target** is the acted-upon protein. In grammar records this maps to:

```text
<prots> TARGET_SEQ <protd> <MODE> <prots> ACTOR_SEQ <protd>
```

- **target** → `protein_a` (fixed context, not diffused)
- **actor** → `protein_b` (generated partner)
- **`<MODE>`** → one of `<binding>`, `<activation>`, `<inhibition>`, `<catalysis>`,
  `<reaction>`, `<expression>`, `<ptmod>` (fixed, not diffused)

For **binding** mode in actions (undirected), partner order is canonicalized by
sorted protein id; both orientations may appear during augmentation. Directed modes
(catalysis, expression, reaction, activation, inhibition, ptmod) preserve actor→target
semantics via the column order above.

Physical MINT splits (`mint_ppi`) remain binding-only and use the same PPI template
with `<binding>` as the relation token.

## Inference Contract

Generation allocates a task-specific maximum record and denoises all non-fixed
positions. Final decoding must be constrained to one of the record forms above,
with valid open/close-token order and residue-only sequence spans. Conditional
tasks (chain completion, antigen-conditioned receptor generation, CDR infilling,
PPI partner generation) are expressed as inference-time partial-mask prompts over
this same grammar; see `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/masks.py`.
