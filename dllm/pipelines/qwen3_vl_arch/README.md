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

- `BioSeqNoEncoderDiffusionModel`: decoder-only masked diffusion over the token stream emitted by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/collator.py`.
- `BioSeqEncoderDiffusionModel`: ESMC/ESM-conditioned masked diffusion. Decoder target residues that are corrupted are also masked in the per-chain encoder input before the encoder forward pass, so clean target residues cannot leak through encoder features.
- `BioSeqDiffusionTransformerConfig`: vocabulary/model/loss config. Use `vocab_size=33` for the local ESM2/MINT tokenizer and `vocab_size=64` when training with ESMC tokenization.
- `load_local_esmc_encoder`: local Biohub ESMC loader for `/c20250601/mj/model_weights/esmc/<model>` checkpoints when Hugging Face `AutoModel` does not recognize `model_type="esmc"`.
- `sample_bioseq_diffusion_noise`: BioSeq diffusion corruption over `diffusion_loss_mask`.
- `apply_decoder_corruption_to_encoder`: maps corrupted decoder residues back to per-chain encoder residues.

The encoder model keeps the biological encoder trainable by default. Use `freeze_encoder=True` only for ablations or debugging.

Known environment constraint: local ESMC checkpoints under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` declare `model_type="esmc"`, but this environment uses `transformers==4.48.1`, which does not recognize that type through `AutoModel`. The project loader handles this by falling back to Biohub `esm==3.2.3`, instantiating native `esm.models.esmc.ESMC`, and converting local safetensor keys before strict state-dict loading.

Verification:

```bash
python -m py_compile /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py
python -m pytest /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_data_loader.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_qwen3_vl_bioseq_model.py -q
```

## DDP Training

The Qwen3-VL-style BioSeq DDP trainer is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/train_qwen3_vl_bioseq_ddp.py`

It should be launched with `torchrun` for multi-node/multi-GPU training. The
streaming data path is rank-aware: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data/mixture.py`
combines DDP rank and DataLoader worker id when sharding iterable source files.

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
