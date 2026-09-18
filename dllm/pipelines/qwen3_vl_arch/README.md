# Qwen3-VL / BioSeq model-layer snapshot

This directory is retained as a model-layer namespace and compatibility location. It is **not**
the current immune data or training pipeline. The current immune data implementation is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`, and the formal
training entry is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.

The former `data/` alias tree, former `training/` tree, `GRAMMAR_V1.md`, and old BioSeq training
examples were deleted. Do not restore them or use their historical commands. Historical audit
snapshots remain under
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/refactor_baseline`; archived historical
evidence remains under
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive`.

## Qwen3-VL architecture snapshot

The copied Qwen source files live under:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/Qwen3-VL/qwen-vl-finetune/qwenvl/model`

The retained snapshot contains the dense and MoE Qwen3-VL configuration/model/modular files:

- `qwen3_vl/configuration_qwen3_vl.py`
- `qwen3_vl/modeling_qwen3_vl.py`
- `qwen3_vl/modular_qwen3_vl.py`
- `qwen3_vl_moe/configuration_qwen3_vl_moe.py`
- `qwen3_vl_moe/modeling_qwen3_vl_moe.py`
- `qwen3_vl_moe/modular_qwen3_vl_moe.py`

Processors, data loaders, finetuning scripts, demo assets, cookbooks, evaluation scripts, and
Docker files were not migrated into this snapshot. The copied model files use `transformers.*`
for shared Hugging Face utilities and keep relative imports between local Qwen3-VL files.

The local environment has `transformers==4.48.1`, while this Qwen3-VL snapshot expects newer
internal APIs. Syntax checks may pass even when runtime model import requires a newer compatible
Transformers environment or local compatibility shims.

## Retained BioSeq model layer

The retained BioSeq model-layer implementation is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`

It provides reusable diffusion/model primitives used by the current fusion implementation in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_fusion_model.py`:

- `BioSeqEncoderDiffusionModel`: ESMC/ESM-conditioned masked diffusion model primitives.
- `BioSeqNoEncoderDiffusionModel`: no-encoder bidirectional masked-diffusion primitive retained
  for compatibility and controlled diagnostics; it is not the current immune training entry.
- `BioSeqDiffusionTransformerConfig`: configuration for the retained BioSeq decoder stack.
- `load_local_esmc_encoder` and `BioSeqEncoderDiffusionModel.from_esmc`: local ESMC loading with
  a Biohub `esm==3.2.3` fallback when `transformers==4.48.1` cannot resolve `model_type="esmc"`.
- `sample_bioseq_diffusion_noise`, `sample_chain_conditioned_timesteps`, and
  `apply_decoder_corruption_to_encoder`: corruption and encoder-state mirroring utilities.

The current fusion data path supplies prepared semantic records and batch tensors from
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. That
package owns the active records, grammar, tokenizer adapter, prepared loader, collator, and
per-chain encoder input construction. This directory does not provide a replacement data loader.

## Sampling and relation auxiliaries

Sampling utilities are retained in:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py`

The default decoding behavior remains `confidence-deterministic-linear`; future remasking or
region-aware strategies must be explicit opt-in changes and must not silently change the current
fusion baseline.

Relation auxiliary helpers are retained in:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/relation_aux.py`

They support the current fusion relation/pairing diagnostics where enabled. The model-layer
functions are not evidence that the deleted qwen data/training implementation remains available.

## Active downstream boundary

The current fusion evaluation continues to use the shared public implementations under:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar`

In particular, CDR infilling, light-chain pairing, and TCR generation remain active. Typical
retained runners are:

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_pairing.sh`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_repr.sh`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_pairing_pll.sh`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_pairing_pll.py`

The old evaluation wrappers, sweeps, and retry jobs are historical/retired. External baseline
protocols are outside this model-layer cleanup and must not be changed.

## Validation boundary

A documentation update does not claim that the latest full validation passed. Main-agent follow-up
must record the actual command, exit code, duration, and absolute artifact path in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` for any fresh validation.

Useful targeted checks, when their environment is available, should start from the active files:

```bash
/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m py_compile \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/sampling_bioseq.py \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/relation_aux.py
```

The path above is intentionally absolute; correct the environment executable if the local
machine exposes the project environment at another approved absolute path. Do not run deleted
`qwen3_vl_bioseq` DDP tests or deleted grammar-v1 tests as current validation.
