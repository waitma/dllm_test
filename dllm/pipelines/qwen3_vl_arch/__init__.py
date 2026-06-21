"""Qwen3-VL architecture snapshot for BioSeq adaptation.

This package intentionally vendors only the dense and MoE Qwen3-VL architecture
source files needed as a starting point for the immune-receptor diffusion model.
"""

from __future__ import annotations

from typing import Any

SOURCE_ROOT = (
    "/vepfs-mlp2/c20250601/251105016/project/dllm_test/base_model/"
    "Qwen3-VL/qwen-vl-finetune/qwenvl/model"
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
    "forbidden_diffusion_target_token_ids",
    "load_local_esmc_encoder",
    "mask_forbidden_target_logits",
    "sample_bioseq_diffusion_noise",
]

_LAZY_EXPORTS = {
    "BioSeqDiffusionDecoder",
    "BioSeqDiffusionOutput",
    "BioSeqDiffusionTransformerConfig",
    "BioSeqEncoderDiffusionModel",
    "BioSeqNoEncoderDiffusionModel",
    "LocalESMCEncoder",
    "apply_decoder_corruption_to_encoder",
    "compute_masked_cross_entropy",
    "forbidden_diffusion_target_token_ids",
    "load_local_esmc_encoder",
    "mask_forbidden_target_logits",
    "sample_bioseq_diffusion_noise",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import modeling_bioseq as _modeling

    return getattr(_modeling, name)
