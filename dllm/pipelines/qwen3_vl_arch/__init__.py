"""Qwen3-VL architecture snapshot for BioSeq adaptation.

This package intentionally vendors only the dense and MoE Qwen3-VL architecture
source files needed as a starting point for the immune-receptor diffusion model.
"""

SOURCE_ROOT = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/"
    "Qwen3-VL/qwen-vl-finetune/qwenvl/model"
)

from .modeling_bioseq import (
    BioSeqDiffusionDecoder,
    BioSeqDiffusionOutput,
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
    LocalESMCEncoder,
    apply_decoder_corruption_to_encoder,
    compute_masked_cross_entropy,
    load_local_esmc_encoder,
    sample_bioseq_diffusion_noise,
)

__all__ = [
    "SOURCE_ROOT",
    "BioSeqDiffusionDecoder",
    "BioSeqDiffusionOutput",
    "BioSeqDiffusionTransformerConfig",
    "BioSeqEncoderDiffusionModel",
    "BioSeqNoEncoderDiffusionModel",
    "LocalESMCEncoder",
    "apply_decoder_corruption_to_encoder",
    "compute_masked_cross_entropy",
    "load_local_esmc_encoder",
    "sample_bioseq_diffusion_noise",
]
