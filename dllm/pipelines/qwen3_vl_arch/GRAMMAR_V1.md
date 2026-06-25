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

- **Fixed** (not diffused): type markers (`<ab>`/`<tcr>`/`<nb>`/`<pep>`) when
  present, all relation tokens, and any **context** object (antigen, MHC, peptide)
  including its `<prots>`, residues, `.`, and `<protd>`.
- **Generated** (diffused): target object structure tokens (`<prots>`, `.`,
  `<protd>`) and target residues. Type markers inside a **generated** block are
  still fixed (e.g. `<ab>` in OAS, `<ab>`/`<nb>` in ab-ag/nb-ag).

PPI training is **conditional only**: protein A block and `<REL>` are fixed;
protein B block is generated. There is no `ppi_joint` mode.

All token classes use the same token-normalized cross-entropy. Training logs
expose separate residue, structure-token, and relation-token losses
(`TOKEN_CLASS_NAMES`).

Padding is applied after the complete record; records are never truncated after
serialization. STRING proteins are deterministically cropped to 1024 residues
during Arrow preprocessing; full-record capacity is 2112 tokens.

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
  --sources oas,ots,tcr,ppi
```

The cache includes paired OAS, paired OTS, non-PPI TCR/epitope records from
`processed_v2`, and canonicalized STRING PPI pairs. Antibody-antigen /
nanobody-antigen records are supported by the renderer but are not in the default
training mix yet.

## Inference Contract

Generation allocates a task-specific maximum record and denoises all non-fixed
positions. Final decoding must be constrained to one of the record forms above,
with valid open/close-token order and residue-only sequence spans. Conditional
tasks (chain completion, antigen-conditioned receptor generation, CDR infilling,
PPI partner generation) are expressed as inference-time partial-mask prompts over
this same grammar; see `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/masks.py`.
