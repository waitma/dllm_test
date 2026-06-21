# BioSeq Grammar V1

`grammar_v1` is a parallel input path for joint biological sequence generation.
The legacy chain-concatenation path remains available through
`--input-format legacy`.

## Record Forms

The canonical forms are:

```text
OAS: <generate><proas>HEAVY<proae><binding><probs>LIGHT<probd>
OTS: <generate><proas>ALPHA<proae><binding><probs>BETA<probd>
TCR: <peptides>PEPTIDE<peptided><generate><proas>ALPHA<proae><binding><probs>BETA<probd>
pMHC: <fixs>MHC<fixd><binding><peptides>PEPTIDE<peptided><generate><proas>ALPHA<proae><binding><probs>BETA<probd>
PPI: <protas>PROTEIN_A<protad><binding><protbs>PROTEIN_B<protbd>
AB-antigen: <fixs>ANTIGEN<fixd><generate><proas>HEAVY<proae><binding><probs>LIGHT<probd>
```

`<proae>` is the intentionally retained spelling for the protein-A closing
token. The relation vocabulary contains `binding`, `activation`, `inhibition`,
`catalysis`, `reaction`, `expression`, `ptmod`, `neutralization`,
`nonbinding`, and `unknown_relation`. Sources without a typed relationship use
`<binding>`.

## Denoising

Only the complete `<fixs>...<fixd>` span is protected. Every other non-padding
token is eligible for masked diffusion, including `<generate>`, sequence
boundaries, relation tokens, peptide residues, and protein residues. All token
classes use the same token-normalized cross-entropy. Training logs also expose
separate residue, structure-token, and relation-token losses.

Padding is applied after the complete record. Records are never truncated after
serialization because that could create invalid grammar. STRING proteins are
deterministically cropped to 1024 residues during Arrow preprocessing, and the
full-record capacity is 2112 tokens.

## Encoder Mode

Encoder variants receive the current noisy decoder stream, not the clean target
or ground-truth chain boundaries. Residues and `<mask>` retain ESMC ids. Every
grammar-only token maps to the ESMC separator id (`|`, id 31). Decoder
corruption is copied position-for-position into this single proxy stream before
ESMC feature extraction.

Grammar boundary tokens are the sole entity-role representation. Grammar-v1
models do not instantiate or consume chain-role embeddings. The legacy input
path keeps its existing chain-role embedding behavior.

## Data

Build semantic Arrow shards with:

```bash
python scripts/data/build_bioseq_grammar_v1.py \
  --output-dir data/bioseq_grammar_v1 \
  --splits train,valid \
  --sources oas,ots,tcr,ppi
```

The cache includes paired OAS, paired OTS, non-PPI TCR/epitope records from
`processed_v2`, and canonicalized STRING PPI pairs. Reverse STRING duplicates
and cross-organism rows are removed. The duplicated `processed_v2` PPI rows are
excluded.

## Inference Contract

Generation allocates a task-specific maximum record and denoises all
non-fixed positions. Final decoding must be constrained to one of the record
forms above, with a valid opening/closing-token order and residue-only sequence
spans. A grammar FSA/Viterbi projection is the intended final decoder. Optional
post-projection refinement uses `--refine-steps`; the initial training release
uses zero refinement steps. A separate final repair pass can be added later if
projection error analysis shows it is necessary.
