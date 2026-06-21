"""Shared helpers for grammar-v1 BioSeq downstream evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from dllm.pipelines.qwen3_vl_arch.data import (
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
DEFAULT_GRAMMAR_DATA_DIR = PROJECT_ROOT / "data" / "bioseq_grammar_v1"


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


def collate_records(records: list[BioSeqRecord], collator: GrammarBioSeqCollator | None = None) -> dict[str, Any]:
    collator = collator or build_grammar_collator()
    return collator(records)


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
    return generate_bioseq(model, batch, partial_mask=partial_mask, config=config)


def load_sample_oas_record(
    split: str = "valid",
    index: int = 0,
    data_dir: Path | None = None,
) -> BioSeqRecord:
    from dllm.pipelines.qwen3_vl_arch.data.grammar import GrammarArrowSource, GrammarArrowSourceConfig

    source = GrammarArrowSource(GrammarArrowSourceConfig(name="oas", path=data_dir or DEFAULT_GRAMMAR_DATA_DIR, split=split))
    for offset, record in enumerate(source.iter_records()):
        if offset == index:
            return record
    raise IndexError(f"OAS grammar split {split} has fewer than {index + 1} records")
