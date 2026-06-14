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
- The Qwen3-VL-style BioSeq foundation-model path lives at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`.
- Training examples live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq`.
- Tests live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq`.
- BioSeq code must not import `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core` or reuse old diffusion trainers.
- The pipeline may stay inside the `dllm` namespace so that the current package layout still works.
- Qwen3-VL-style training models must use the loader/view masks from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` and compute diffusion loss only on `diffusion_loss_mask` / `diffusion_target_mask`.
- Encoder-conditioned Qwen3-VL-style training must mask the same corrupted target residues in `encoder_input_ids` before the ESMC/ESM forward pass. Fixed context chains remain clean and visible.
- Qwen3-VL-style multi-node/multi-GPU training must use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py` with `torchrun`.
- Iterable Qwen3-VL-style data streams must be sharded by DDP rank plus DataLoader worker, not by `DistributedSampler`.
- The old lightweight generic BioSeq backend must not be used as the antibody/Ophiuchus-Ab implementation or as the Qwen3-VL-style foundation-model implementation.

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

Current ESMC environment note: local ESMC checkpoints declare `model_type="esmc"`, but `transformers==4.48.1` does not recognize that model type. Use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::BioSeqEncoderDiffusionModel.from_esmc` or `load_local_esmc_encoder`; those paths fall back to Biohub `esm==3.2.3` and load the local safetensors without relying on `AutoModel` alone.

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
