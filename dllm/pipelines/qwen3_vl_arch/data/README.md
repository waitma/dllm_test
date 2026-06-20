# BioSeq Data Loading

This package is the first BioSeq foundation-model data entry point. It is meant
for training-time loading, not offline conversion into one fixed JSONL format.

Supported first-version sources:

- OAS paired antibody CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{train,valid,holdout}_oas_label.csv`
- OTS paired TCR CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`
- Nanobody/VHH CSV from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final`
- Existing processed JSONL from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2`

For structured joint generation, `grammar_v1` uses semantic Arrow shards under
`data/bioseq_grammar_v1`; see `../GRAMMAR_V1.md`.
- Optional PPI Arrow source from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`

The loading path is:

```text
source loader -> BioSeqRecord -> weighted mixed stream -> full-denoise target masks -> ESM-compatible collator -> model batch
```

`BioSeqRecord` stores the complete biological sample: chains, chain roles,
regions, source, task type, metadata, labels, and weight. It does not decide
which positions are generated for a specific training step.

The recommended foundation-training path keeps physical microbatches mixed.
`WeightedMixtureDataset` emits a source-weighted stream of records, and
`DataLoader(batch_size=N)` forms batches directly from that stream. This allows
one batch to contain records such as antibody, antibody-antigen, TCR, and PPI.
Batch-level de-duplication is not part of the default path because dataset
processing owns global de-duplication:

```python
records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=seed)
loader = DataLoader(
    records,
    batch_size=micro_batch_size,
    collate_fn=BioSeqQwenDataCollator(
        single_view_per_batch=False,
        require_homogeneous_task=False,
    ),
    drop_last=True,
)
```

`TaskHomogeneousBatchDataset` remains available only as an ablation/debugging
wrapper when a run intentionally wants one BioSeq task group per physical
microbatch.

For multi-node/multi-GPU training, do not use `DistributedSampler` with these
iterable sources. `SequentialMultiSourceDataset` and `WeightedMixtureDataset`
already shard records by both DDP rank and DataLoader worker:

```text
global_shard_index = rank * num_workers + worker_id
num_shards = world_size * num_workers
```

The DDP trainer at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`
uses this path directly.

Foundation training uses a simple default objective: every record resolves to
`full_denoise`, so all eligible target residues are diffusion targets and the
loss is computed only on corrupted target residues. Mixed physical batches can
still contain antibody, antibody-antigen, TCR, pMHC, PPI, and other task types;
the mask, not the batch grouping, defines the generated part.

The DDP trainer hard-codes `allowed_views=["full_denoise"]`; it does not expose
the ablation view path during foundation pretraining. `BioSeqViewSampler` keeps
conditional views available only for separate scripts or downstream fine-tuning
that intentionally pass `allowed_views` such as:

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

The task-specific conditional profiles are not part of the DDP foundation
training objective. They are supported so that antibody chain completion,
antigen-conditioned receptor generation, peptide design, pMHC-conditioned TCR
generation, and FR/CDR infilling can be enabled intentionally outside the main
foundation trainer.

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

These tensors are used by ESM2/ESMC feature-conditioned training to form the
per-chain diffusion state `x_t` for the biological encoder.
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
token. For ESMC feature-conditioned batches, use `HuggingFaceEsmTokenizerAdapter`
or an ESMC-specific tokenizer loaded from the target ESMC snapshot.

The ESMC/ESMFold2 paper supports this implementation split: ESMC is the
per-protein sequence representation model, while ESMFold2 handles multi-chain
complexes by encoding each protein chain independently with frozen ESMC 6B and
then building complex-level pair/folding/diffusion representations downstream.
