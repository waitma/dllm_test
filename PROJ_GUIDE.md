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
- Training examples live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq`.
- Tests live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq`.
- BioSeq code must not import `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core` or reuse old diffusion trainers.
- The pipeline may stay inside the `dllm` namespace so that the current package layout still works.

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
