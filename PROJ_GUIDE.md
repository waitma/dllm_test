# dllm_test Project Guide

## Long-Term Rules

- The project root is `/vepfs-mlp2/c20250601/251105016/project/dllm_test`.
- The model weight root is `/c20250601/mj/model_weights`.
- All project docs, configs, examples, and scripts must use absolute paths.
- Plan changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`.
- Process changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Long-term project rules must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`.

## BioSeq Pipeline Boundary

- The BioSeq pipeline lives at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq`.
- The BioSeq foundation-model path lives at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`.
- Training examples live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq`.
- Tests live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq`.
- BioSeq code must not import `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core` or reuse old diffusion trainers.
- The pipeline may stay inside the `dllm` namespace so that the current package layout still works.
- Public docs should call this path `BioSeq foundation` or `BioSeq foundation-model`. Existing identifiers such as `qwen3_vl_arch`, `qwen3_vl_bioseq_*`, and `bioseq-qwen3-vl` are compatibility names for paths, task IDs, output directories, and historical runs.
- BioSeq foundation training models must use the loader/grammar masks from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` and compute diffusion loss only on `diffusion_loss_mask` / `diffusion_eligible_mask`.
- BioSeq foundation training uses a single grammar path: `GrammarArrowSource` -> `WeightedMixtureDataset` -> `TaskHomogeneousBatchDataset(batch_size=N)` -> `DataLoader(batch_size=None)` -> `GrammarBioSeqCollator`. The legacy chain-concatenation collator/view-sampler has been removed. The active grammar token layout (grammar-v2) is documented in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/GRAMMAR_V1.md`.
- Encoder-conditioned BioSeq foundation training must mask the same corrupted target residues in `encoder_input_ids` before the encoder forward pass. Fixed context chains remain clean and visible. The encoder runs per chain/sequence (`encoder_input_ids [batch, max_chains, chain_len]`, flattened to `[batch * max_chains, chain_len]`); gathered features **replace** decoder residue embeddings (no projection by default). `--model-type encoder|esm2` forces `hidden_size` to the encoder latent dim; no-encoder ablations may pass `--align-hidden-size-to-encoder`. Backends: `--model-type encoder` (ESMC via `from_esmc`) or `--model-type esm2` (ESM2 via `from_hf_encoder`).
- BioSeq foundation multi-node/multi-GPU training must use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` with `torchrun`.
- Iterable BioSeq foundation data streams must be sharded by DDP rank plus DataLoader worker, not by `DistributedSampler`. Run DDP with `--num-workers 0`: each extra worker process re-shards the infinite weighted stream independently and can desync the first batch across ranks, causing NCCL collective timeouts.
- The old lightweight generic BioSeq backend must not be used as the antibody/Ophiuchus-Ab implementation or as the BioSeq foundation-model implementation.

## Weight Layout

```text
/c20250601/mj/model_weights/esmc/ESMC-300M
/c20250601/mj/model_weights/esmc/ESMC-600M
/c20250601/mj/model_weights/esmc/ESMC-6B
/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D
/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab
```

The environment variable `BIOSEQ_MODEL_WEIGHTS_ROOT` may override the root, but its default must remain `/c20250601/mj/model_weights`.

`/c20250601/mj/model_weights/esm2/esm2_t48_15B_UR50D` is optional and is not part of the current default download set.

Current ESMC environment note: local ESMC checkpoints declare `model_type="esmc"`, but `transformers==4.48.1` does not recognize that model type. Use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::BioSeqEncoderDiffusionModel.from_esmc` or `load_local_esmc_encoder`; those paths fall back to Biohub `esm==3.2.3` and load the local safetensors without relying on `AutoModel` alone. For ESMC tokenization, use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/esm_encoding.py::HuggingFaceEsmTokenizerAdapter`, which falls back to local `tokenizer.json` when `AutoTokenizer` cannot import `ESMCTokenizer`.

Formal BioSeq foundation stage-1 training uses offline wandb by default. Keep run outputs, checkpoints, and wandb run files under absolute output paths such as `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc300m_stage1`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc600m_stage1`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1`. Do not use `/tmp` for ESMC training outputs or checkpoint tests because the root filesystem can be full and ESMC checkpoints are multi-GB.

## Data Layout

- PPI and interaction task raw data root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- PPI and interaction task processed outputs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- Rebuild command:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py
```

- `interaction_records_unified.csv` is an audit/consolidation table. Training should use task-specific sharded `bioseq.v1` records rather than reading the 3GB CSV directly.
