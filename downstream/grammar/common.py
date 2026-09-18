"""Shared backend for current Immune LLaDA fusion evaluation.

Representative smoke checks:
  python -m downstream.grammar.cdr_infill --smoke --device cuda
  python -m downstream.grammar.light_chain_pairing --smoke --device cuda

Only current fusion checkpoint directories are supported; the standalone
BioSeq grammar-v1/v2 trainer and its checkpoint format are retired.
"""

from __future__ import annotations

import sys

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dllm.pipelines.immune_llada.data import (
    BioSeqChain,
    BioSeqRecord,
    GrammarBioSeqCollator,
    GrammarTokenizer,
    Esm2SequenceTokenizer,
)
from dllm.pipelines.qwen3_vl_arch.modeling_bioseq import (
    BioSeqDiffusionTransformerConfig,
    BioSeqEncoderDiffusionModel,
    BioSeqNoEncoderDiffusionModel,
)
from dllm.pipelines.qwen3_vl_arch.sampling_bioseq import BioSeqGenerateConfig, generate_bioseq

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
# Kept only for the optional record-fixture smoke path. Current evaluation
# datasets are supplied explicitly by each retained evaluator.
DEFAULT_GRAMMAR_DATA_DIR = PROJECT_ROOT / "data" / "prepared" / "immune_v3_heterotypic"


def build_grammar_tokenizer() -> GrammarTokenizer:
    return GrammarTokenizer(Esm2SequenceTokenizer())


def build_grammar_collator(tokenizer: GrammarTokenizer | None = None) -> GrammarBioSeqCollator:
    return GrammarBioSeqCollator(tokenizer or build_grammar_tokenizer())


def antibody_pair_record(
    heavy_sequence: str,
    light_sequence: str,
    *,
    heavy_regions: dict[str, str] | None = None,
    light_regions: dict[str, str] | None = None,
    source: str = "oas_paired",
) -> BioSeqRecord:
    return BioSeqRecord(
        chains=[
            BioSeqChain(heavy_sequence, "antibody_heavy", regions=heavy_regions or {}),
            BioSeqChain(light_sequence, "antibody_light", regions=light_regions or {}),
        ],
        task_type="antibody",
        source=source,
    )


def tcr_pair_record(
    alpha_sequence: str,
    beta_sequence: str,
    *,
    alpha_regions: dict[str, str] | None = None,
    beta_regions: dict[str, str] | None = None,
    source: str = "ots_paired",
) -> BioSeqRecord:
    """Paired TCR record. Grammar renders ``tcr_pair`` as alpha=chain 0, beta=chain 1."""

    return BioSeqRecord(
        chains=[
            BioSeqChain(alpha_sequence, "tcr_alpha", regions=alpha_regions or {}),
            BioSeqChain(beta_sequence, "tcr_beta", regions=beta_regions or {}),
        ],
        task_type="tcr",
        source=source,
    )


def collate_records(records: list[BioSeqRecord], collator: GrammarBioSeqCollator | None = None) -> dict[str, Any]:
    collator = collator or build_grammar_collator()
    return collator(records)



def load_grammar_checkpoint(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, GrammarTokenizer]:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from examples.llada.load_fusion_checkpoint import is_fusion_checkpoint, load_fusion_for_eval

    if is_fusion_checkpoint(checkpoint_path):
        bundle = load_fusion_for_eval(checkpoint_path, device=device)
        model = bundle.model
        model._fusion_eval_collator = bundle.collator
        return model, bundle.grammar_tokenizer

    raise ValueError(
        "Only Immune LLaDA fusion checkpoints are supported. "
        "The historical BioSeq grammar-v1/v2 trainer and its .pt checkpoint loader "
        "have been retired; pass a fusion checkpoint directory or model.safetensors. "
        f"Received: {checkpoint_path}"
    )


def build_eval_collator(model: nn.Module, tokenizer: GrammarTokenizer) -> GrammarBioSeqCollator:
    """Collator matching the loaded checkpoint (fusion remaps decoder ids)."""

    fusion_collator = getattr(model, "_fusion_eval_collator", None)
    if fusion_collator is not None:
        return fusion_collator
    return build_grammar_collator(tokenizer)


def load_untrained_no_encoder(vocab_size: int, **overrides: Any) -> BioSeqNoEncoderDiffusionModel:
    config = BioSeqDiffusionTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=int(overrides.pop("hidden_size", 32)),
        num_hidden_layers=int(overrides.pop("num_hidden_layers", 2)),
        num_attention_heads=int(overrides.pop("num_attention_heads", 4)),
        intermediate_size=int(overrides.pop("intermediate_size", 64)),
        dropout=float(overrides.pop("dropout", 0.0)),
        max_position_embeddings=int(overrides.pop("max_position_embeddings", 512)),
        mask_token_id=int(overrides.pop("mask_token_id", 32)),
        **overrides,
    )
    return BioSeqNoEncoderDiffusionModel(config).eval()


def run_grammar_generate(
    model: nn.Module,
    batch: dict[str, Any],
    *,
    partial_mask: torch.Tensor | None = None,
    max_iter: int = 32,
    sampling_strategy: str = "gumbel_argmax",
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    config = BioSeqGenerateConfig(
        max_iter=max_iter,
        sampling_strategy=sampling_strategy,
        temperature=temperature,
        cfg_scale=cfg_scale,
    )
    output_tokens, scores = generate_bioseq(
        model, batch, partial_mask=partial_mask, config=config
    )
    return _inverse_remap_llada_tokens(model, output_tokens), scores


def _inverse_remap_llada_tokens(model: nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    inverse = getattr(model, "llada_to_grammar_ids", None)
    if inverse is None:
        return tokens
    inverse = inverse.to(device=tokens.device)
    clamped = tokens.clamp(min=0, max=int(inverse.numel()) - 1)
    mapped = inverse[clamped]
    return torch.where(mapped.ge(0), mapped, tokens)


def load_sample_oas_record(
    split: str = "valid",
    index: int = 0,
    data_dir: Path | None = None,
) -> BioSeqRecord:
    from dllm.pipelines.immune_llada.data.grammar import GrammarArrowSource, GrammarArrowSourceConfig

    source = GrammarArrowSource(GrammarArrowSourceConfig(name="oas", path=data_dir or DEFAULT_GRAMMAR_DATA_DIR, split=split))
    for offset, record in enumerate(source.iter_records()):
        if offset == index:
            return record
    raise IndexError(f"OAS grammar split {split} has fewer than {index + 1} records")
