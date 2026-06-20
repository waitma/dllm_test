# Qwen3-VL Architecture Snapshot

Source root:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/Qwen3-VL/qwen-vl-finetune/qwenvl/model`

This directory keeps only the latest Qwen architecture files needed before
adapting the model for BioSeq:

- `qwen3_vl/configuration_qwen3_vl.py`
- `qwen3_vl/modeling_qwen3_vl.py`
- `qwen3_vl/modular_qwen3_vl.py`
- `qwen3_vl_moe/configuration_qwen3_vl_moe.py`
- `qwen3_vl_moe/modeling_qwen3_vl_moe.py`
- `qwen3_vl_moe/modular_qwen3_vl_moe.py`

Intentionally not migrated:

- Qwen2/Qwen2.5 model files
- processors and video processors
- finetuning scripts
- data loaders
- demo assets
- cookbooks
- evaluation scripts
- Docker files

The copied model files were adjusted so imports of shared Hugging Face utility
modules use `transformers.*` instead of the original relative package path.
Local imports between the migrated Qwen3-VL dense and MoE files remain relative.

The current environment has `transformers==4.48.1`, which is older than this
Qwen3-VL snapshot and does not provide several newer internal APIs such as
`transformers.masking_utils`, `transformers.modeling_layers`,
`transformers.vision_utils`, `transformers.utils.output_capturing`,
`transformers.initialization`, `RopeParameters`, and `auto_docstring`. Syntax
validation passes, but runtime model import requires a matching newer
Transformers version or local compatibility shims.

## BioSeq Training Models

Training models for this path live in:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`

Current exported entry points:

- `BioSeqNoEncoderDiffusionModel`: no-encoder masked diffusion over the token stream emitted by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`. This path uses bidirectional self-attention and has no ESM/ESMC encoder; it is not a causal/autoregressive LM.
- `BioSeqEncoderDiffusionModel`: ESMC/ESM feature-conditioned masked diffusion. ESMC runs on the current per-chain diffusion state `x_t` and returns token-level features; the BioSeq denoiser still runs over the concatenated multi-chain token stream and models chain-chain denoising jointly.
- `BioSeqDiffusionTransformerConfig`: vocabulary/model/loss config. The local
  ESM2 and ESMC sequence tokenizers expose 33 active ids. `grammar_v1` appends
  23 structural/relation ids for a decoder vocabulary of 56.
- `load_local_esmc_encoder`: local Biohub ESMC loader for `/c20250601/mj/model_weights/esmc/<model>` checkpoints when Hugging Face `AutoModel` does not recognize `model_type="esmc"`.
- `sample_bioseq_diffusion_noise`: BioSeq diffusion corruption over `diffusion_loss_mask`.
- `apply_decoder_corruption_to_encoder`: builds the per-chain ESMC `x_t` by mapping the decoder corruption state back to encoder residue positions.

The encoder model keeps the biological encoder trainable by default. Use `freeze_encoder=True` only for ablations or debugging.

The structured multi-entity representation and Arrow preprocessing contract
are documented in [GRAMMAR_V1.md](GRAMMAR_V1.md).

Known environment constraint: local ESMC checkpoints under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` declare `model_type="esmc"`, but this environment uses `transformers==4.48.1`, which does not recognize that type through `AutoModel`. The project loader handles this by falling back to Biohub `esm==3.2.3`, instantiating native `esm.models.esmc.ESMC`, and converting local safetensor keys before strict state-dict loading.

Verification:

```bash
python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py
python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q
```

## DDP Training

The BioSeq foundation-model DDP trainer is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`

It should be launched with `torchrun` for multi-node/multi-GPU training. The
streaming data path is rank-aware: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py`
combines DDP rank and DataLoader worker id when sharding iterable source files.
The default foundation objective is `full_denoise`: every eligible target
residue is part of the diffusion target mask, and train/validation loss is
computed only on corrupted target residues. The DDP trainer hard-codes that
view and does not route foundation training through conditional-view ablations.

Local CPU DDP smoke:

```bash
torchrun --standalone --nproc_per_node=2 \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py \
  --device cpu --model-type no_encoder --sources oas --limit-per-source 16 \
  --batch-size 2 --max-steps 2 --max-chain-length 64 --max-sequence-length 256 \
  --hidden-size 32 --num-hidden-layers 1 --num-attention-heads 4 --intermediate-size 64 \
  --num-workers 0 --save-interval 1 --resume none --wandb-mode disabled \
  --output-dir /tmp/qwen3_vl_bioseq_ddp_smoke
```

Cluster template:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/qwen3_vl_bioseq_16gpu_smoke.yml`
