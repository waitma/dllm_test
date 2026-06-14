# BioSeq Data Loading

This package is the first Qwen3-VL-style BioSeq data entry point. It is meant
for training-time loading, not offline conversion into one fixed JSONL format.

Supported first-version sources:

- OAS paired antibody CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/compat_for_current_loader_oasrule`
- OTS paired TCR CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`
- Nanobody/VHH CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final`
- Existing processed JSONL from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2`
- Optional PPI Arrow source from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`

The loading path is:

```text
source loader -> BioSeqRecord -> task batcher -> view sampler -> ESM-compatible collator -> model batch
```

`BioSeqRecord` stores the complete biological sample: chains, chain roles,
regions, source, task type, metadata, labels, and weight. It does not decide
which positions are generated for a specific training step.

`TaskHomogeneousBatchDataset` is the recommended training wrapper. It groups a
record stream into microbatches with one BioSeq task group per batch. It does
not deduplicate by default because dataset processing should already handle
global sample de-duplication; set `deduplicate_within_batch=True` only as a
defensive debugging option for untrusted or overlapping streams. Use it with
`DataLoader` by setting `batch_size=None`; the dataset already emits a list of
records:

```python
records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=seed)
batches = TaskHomogeneousBatchDataset(records, batch_size=micro_batch_size)
loader = DataLoader(
    batches,
    batch_size=None,
    collate_fn=BioSeqQwenDataCollator(require_homogeneous_task=True),
)
```

For multi-node/multi-GPU training, do not use `DistributedSampler` with these
iterable sources. `SequentialMultiSourceDataset` and `WeightedMixtureDataset`
already shard records by both DDP rank and DataLoader worker:

```text
global_shard_index = rank * num_workers + worker_id
num_shards = world_size * num_workers
```

The DDP trainer at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
uses this path directly.

`BioSeqQwenDataCollator` samples one shared generation view for the whole batch
by default. This keeps a physical microbatch from mixing objectives such as
full-chain receptor generation and CDR infilling. For stable multi-task updates,
the training loop should accumulate gradients across several task-homogeneous
microbatches before `optimizer.step()` rather than treating each single-task
microbatch as an isolated optimizer update.

View sampling uses one simple default rule. `full_denoise` is sampled with
`full_denoise_probability=0.5`; the remaining compatible condition views share
the other 0.5 uniformly. If no condition view is compatible with the current
record or batch, sampling falls back to `full_denoise`.

`BioSeqViewSampler` creates a training view such as:

- full denoising over eligible target chains
- heavy-to-light
- light-to-heavy
- antigen-to-antibody heavy/light
- antigen-to-nanobody
- heavy+antigen-to-light
- light+antigen-to-heavy
- antigen+antibody/nanobody FR-to-CDR
- beta+context-to-alpha
- alpha+context-to-beta
- MHC-to-peptide+TCR denoising
- alpha+beta+MHC-to-peptide
- pMHC-to-alpha+beta
- pMHC+TCR FR-to-CDR
- FR-to-CDR infilling
- single-CDR infilling
- CDR-to-FR infilling

Default view sampling is task-specific rather than one flat list. Antibody-only
records use antibody chain completion and FR/CDR infilling. Antibody-antigen
records use antigen-conditioned receptor generation plus antigen-conditioned
CDR infilling. TCR-pMHC records use pMHC-conditioned TCR generation plus
pMHC-conditioned TCR CDR infilling. Peptide-generation and co-design views are
supported but should be enabled intentionally through `allowed_views` or a
future weighted profile.

`BioSeqQwenDataCollator` then emits token-level masks:

- `visible_mask`
- `fixed_context_mask`
- `diffusion_target_mask`
- `diffusion_loss_mask`

`full_denoise` is not all chains unconditionally. It honors explicit record
`metadata["targets"]` when present and still keeps antigen, peptide, MHC, and
HLA-like chains as fixed context by default. Those context residues are visible
conditioning tokens and do not receive diffusion loss.

`mhc_to_peptide_tcr` is the complementary TCR-pMHC training view for the case
where MHC/HLA is fixed context while peptide plus available TCR chains are
diffusion targets. If a TCR epitope dataset stores the peptide as `antigen`,
that chain is treated as a target only for `tcr_epitope` or `tcr_pmhc` records.

`tcr_mhc_to_peptide` fixes available TCR chains plus MHC/HLA and uses peptide
or epitope as the diffusion target. `pmhc_to_tcr` fixes peptide/epitope plus
MHC/HLA and uses available TCR alpha/beta chains as diffusion targets.

Antibody-antigen and nanobody-antigen views follow the same fixed/target
pattern. `antigen_to_antibody` fixes antigen and targets available heavy/light
chains. `heavy_antigen_to_light` fixes heavy plus antigen and targets light.
`light_antigen_to_heavy` fixes light plus antigen and targets heavy.
`antigen_to_nanobody` fixes antigen and targets the VHH chain. Antibody and
nanobody training views do not generate the antigen by default; antigen is
clean conditioning context for receptor-side diffusion loss. `antigen_fr_to_cdr`
and `antigen_single_cdr` fix antigen plus receptor framework residues and target
all CDRs or one selected CDR.

The FR/CDR views are not antibody-specific. They operate on any chain with
`FR*` and `CDR*` region annotations, including TCR full-chain records when the
source adapter preserves those regions. For TCR-pMHC records, `pmhc_fr_to_cdr`
and `pmhc_single_cdr` require peptide/epitope plus MHC/HLA context and target
TCR CDR spans only.

The collator uses ESM-family token ids by default via `Esm2SequenceTokenizer`.
It also emits per-chain encoder tensors:

- `encoder_input_ids`
- `encoder_attention_mask`
- `encoder_residue_mask`
- `encoder_chain_mask`
- `encoder_chain_role_ids`

These tensors are intended for future ESM2/ESMC encoder-conditioned training.
If the encoder is loaded from a local Hugging Face ESM-family snapshot,
`HuggingFaceEsmTokenizerAdapter` can be used to make token ids match that
encoder tokenizer.

## ESM Tokenizer Compatibility

The default `Esm2SequenceTokenizer` matches the local ESM2/MINT snapshots under
`/c20250601/mj/model_weights/esm2`: `<cls>` id 0, `<pad>` id 1, `<eos>` id 2,
`<unk>` id 3, standard amino-acid ids 4-28, `.` id 29, `-` id 30, `<null_1>`
id 31, and `<mask>` id 32.

The local ESMC snapshots under `/c20250601/mj/model_weights/esmc` use the same
ids for normal amino acids and `<mask>`, but id 31 is `|` instead of
`<null_1>`. Their `special_tokens_map.json` marks `|` as an additional special
token. For ESMC encoder-conditioned batches, use `HuggingFaceEsmTokenizerAdapter`
or an ESMC-specific tokenizer loaded from the target ESMC snapshot.

The ESMC/ESMFold2 paper supports this implementation split: ESMC is the
per-protein sequence representation model, while ESMFold2 handles multi-chain
complexes by encoding each protein chain independently with frozen ESMC 6B and
then building complex-level pair/folding/diffusion representations downstream.
