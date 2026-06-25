"""Grammar-v1 data module for BioSeq foundation training.

This centralizes the single grammar data path so training, validation, and
downstream evaluation construct identical batches::

    GrammarArrowSource -> WeightedMixtureDataset -> TaskHomogeneousBatchDataset
        -> DataLoader(batch_size=None) -> GrammarBioSeqCollator

It also bakes in the validated DDP defaults (``num_workers=0``) so the data
layer cannot silently re-introduce the per-worker first-batch desync that
caused NCCL collective hangs.

Run (import-only module; exercised through the trainer and tests)::

    from dllm.pipelines.qwen3_vl_arch.data import GrammarDataModule
    dm = GrammarDataModule.from_args(args)
    tokenizer = dm.build_tokenizer()
    loader = dm.train_loader(tokenizer)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .esm_encoding import Esm2SequenceTokenizer, HuggingFaceEsmTokenizerAdapter
from .grammar import (
    DEFAULT_GRAMMAR_DATA_DIR,
    GrammarArrowSource,
    GrammarArrowSourceConfig,
    GrammarBioSeqCollator,
    GrammarTokenizer,
)
from .mixture import SourceWithWeight, TaskHomogeneousBatchDataset, WeightedMixtureDataset

DEFAULT_MAX_SEQUENCE_LENGTH = 2112
VAL_SOURCE_SEED_OFFSET = 10_000


@dataclass
class GrammarDataModule:
    """Owns the grammar-v1 loading path, decoupled from argparse.

    Construct directly for programmatic use, or via :meth:`from_args` from the
    DDP trainer CLI namespace.
    """

    sources: Sequence[str]
    batch_size: int
    grammar_data_dir: Path = DEFAULT_GRAMMAR_DATA_DIR
    source_weights: Mapping[str, float] = field(default_factory=dict)
    split: str = "train"
    source_seed: int = 0
    epoch_size: int | None = None
    limit_per_source: int | None = None
    max_sequence_length: int | None = None
    deduplicate_within_batch: bool = False
    num_workers: int = 0
    tokenizer_path: Path | None = None
    val_split: str = "valid"
    val_interval: int = 0
    val_batches: int = 0

    @classmethod
    def from_args(cls, args: Any) -> "GrammarDataModule":
        # ``getattr`` defaults keep this tolerant of the minimal namespaces that
        # debug/probe scripts build (they only set the data fields they exercise).
        return cls(
            sources=[item.strip() for item in args.sources.split(",") if item.strip()],
            batch_size=args.batch_size,
            grammar_data_dir=getattr(args, "grammar_data_dir", DEFAULT_GRAMMAR_DATA_DIR),
            source_weights={
                "oas": getattr(args, "oas_weight", 1.0),
                "ots": getattr(args, "ots_weight", 1.0),
                "nanobody": getattr(args, "nanobody_weight", 1.0),
                "processed_v2": getattr(args, "processed_v2_weight", 1.0),
                "tcr": getattr(args, "tcr_weight", 1.0),
                "ppi": getattr(args, "ppi_weight", 1.0),
            },
            split=getattr(args, "split", "train"),
            source_seed=getattr(args, "source_seed", 0),
            epoch_size=getattr(args, "epoch_size", None),
            limit_per_source=getattr(args, "limit_per_source", None),
            max_sequence_length=getattr(args, "max_sequence_length", None),
            deduplicate_within_batch=getattr(args, "deduplicate_within_batch", False),
            num_workers=getattr(args, "num_workers", 0),
            tokenizer_path=getattr(args, "tokenizer_path", None),
            val_split=getattr(args, "val_split", "valid"),
            val_interval=getattr(args, "val_interval", 0),
            val_batches=getattr(args, "val_batches", 0),
        )

    def source_weight(self, name: str) -> float:
        return float(self.source_weights.get(name, 1.0))

    def build_tokenizer(self) -> GrammarTokenizer:
        base_tokenizer = (
            Esm2SequenceTokenizer()
            if self.tokenizer_path is None
            else HuggingFaceEsmTokenizerAdapter.from_pretrained(self.tokenizer_path, local_files_only=True)
        )
        return GrammarTokenizer(base_tokenizer)

    def loader(
        self,
        tokenizer: Any,
        *,
        split: str | None = None,
        source_seed: int | None = None,
        epoch_size: int | None = None,
    ) -> DataLoader:
        split = self.split if split is None else split
        source_seed = self.source_seed if source_seed is None else source_seed
        epoch_size = self.epoch_size if epoch_size is None else epoch_size
        configs = [
            GrammarArrowSourceConfig(
                name=name,
                path=self.grammar_data_dir,
                split=split,
                weight=self.source_weight(name),
                max_records=self.limit_per_source,
            )
            for name in sorted({item for item in self.sources if item})
        ]
        sources = [
            SourceWithWeight(GrammarArrowSource(config), weight=config.weight)
            for config in configs
        ]
        records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=source_seed)
        batches = TaskHomogeneousBatchDataset(
            records,
            batch_size=self.batch_size,
            drop_last=True,
            deduplicate_within_batch=self.deduplicate_within_batch,
        )
        collator = GrammarBioSeqCollator(
            tokenizer=tokenizer,
            max_sequence_length=self.max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH,
        )
        return DataLoader(
            batches,
            batch_size=None,
            collate_fn=collator,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def train_loader(self, tokenizer: Any) -> DataLoader:
        return self.loader(
            tokenizer,
            split=self.split,
            source_seed=self.source_seed,
            epoch_size=self.epoch_size,
        )

    def val_loader(self, tokenizer: Any) -> DataLoader | None:
        if self.val_interval <= 0 or self.val_batches <= 0:
            return None
        return self.loader(
            tokenizer,
            split=self.val_split,
            source_seed=self.source_seed + VAL_SOURCE_SEED_OFFSET,
            epoch_size=None,
        )
