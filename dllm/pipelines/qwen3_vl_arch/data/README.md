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
PPI / interaction asset inventory and relation taxonomy: `PPI_DATA.md`.
- Optional PPI Arrow source from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`

The raw CSV/JSONL sources above are consumed offline by
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_bioseq_grammar_v1.py`,
which writes the semantic Arrow shards under `data/bioseq_grammar_v1`. Training
reads only those Arrow shards.

## Training Loading Path (grammar_v1, the only path)

```text
GrammarArrowSource -> WeightedMixtureDataset -> TaskHomogeneousBatchDataset -> GrammarBioSeqCollator -> model batch
```

`BioSeqRecord` stores the complete biological sample: chains, chain roles,
regions, source, task type, metadata, labels, and weight. The grammar renderer
turns each record into a flat token stream and decides which positions are
fixed context versus diffusion targets.

- `GrammarArrowSource` streams `BioSeqRecord`s from one source's Arrow shard
  (with an in-process `load_from_disk` cache so iterator restarts are free).
- `WeightedMixtureDataset` interleaves the per-source streams by sampling
  weight.
- `TaskHomogeneousBatchDataset` groups the stream so each physical microbatch
  holds one BioSeq task group (e.g. all antibody, or all PPI). This keeps the
  rendered grammar layout uniform inside a batch.
- `GrammarBioSeqCollator` renders/pads each record into the grammar token
  stream and the ESMC proxy stream.

```python
sources = [SourceWithWeight(GrammarArrowSource(cfg), weight=cfg.weight) for cfg in configs]
records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=source_seed)
batches = TaskHomogeneousBatchDataset(records, batch_size=N, drop_last=True)
loader = DataLoader(
    batches,
    batch_size=None,  # the dataset already yields complete batches
    collate_fn=GrammarBioSeqCollator(tokenizer, max_sequence_length=2112),
    num_workers=0,
)
```

### DDP sharding and the `num_workers` footgun

For multi-node/multi-GPU training, do not use `DistributedSampler` with these
iterable sources. `SequentialMultiSourceDataset` and `WeightedMixtureDataset`
already shard records by both DDP rank and DataLoader worker:

```text
global_shard_index = rank * num_workers + worker_id
num_shards = world_size * num_workers
```

Because each DataLoader worker is a separate process that independently shards
the infinite weighted stream (and memory-maps its own Arrow copy), using
`num_workers > 0` under `torchrun` can desync the first batch across ranks and
trigger NCCL collective timeouts. The validated default is therefore
`--num-workers 0`; the trainer warns when a distributed run sets it higher.

### Diffusion masks

`GrammarRenderer` marks only the `<fixs>...<fixd>` fixed-context block (antigen,
peptide, MHC/HLA conditioning) as non-target. Everything else — structure
tokens, relation tokens, and non-fixed residues — is a diffusion target. The
collator emits the token-level masks consumed by `sample_bioseq_diffusion_noise`:

- `fixed_context_mask`
- `diffusion_loss_mask`
- `diffusion_eligible_mask`
- `residue_mask`, `structure_token_mask`, `relation_token_mask`, `token_class_ids`

The collator also emits the single-stream ESMC proxy tensors used by the
encoder-conditioned path to build the per-chain diffusion state `x_t`:

- `encoder_input_ids`
- `encoder_attention_mask`
- `encoder_residue_mask`
- `encoder_chain_mask`
- `encoder_position_ids`

If the encoder is loaded from a local Hugging Face ESM-family snapshot,
`HuggingFaceEsmTokenizerAdapter` makes token ids match that encoder tokenizer.

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
