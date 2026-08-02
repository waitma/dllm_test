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
from .records import DEFAULT_MAX_PROTEIN_LENGTH

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
    shuffle_buffer_size: int = 0
    epoch_size: int | None = None
    limit_per_source: int | None = None
    max_sequence_length: int | None = None
    max_protein_length: int = DEFAULT_MAX_PROTEIN_LENGTH
    deduplicate_within_batch: bool = False
    num_workers: int = 0
    prefetch_factor: int = 4
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
                "mint_ppi": getattr(args, "mint_ppi_weight", 1.0),
                "mint_actions": getattr(args, "mint_actions_weight", 1.0),
                "tcr_piste": getattr(args, "tcr_piste_weight", 1.0),
                "tcr_pmhc_fulllength": getattr(args, "tcr_pmhc_fulllength_weight", 1.0),
                "neutralization": getattr(args, "neutralization_weight", 1.0),
                "sabdab2_abag": getattr(args, "sabdab2_abag_weight", 1.0),
            },
            split=getattr(args, "split", "train"),
            source_seed=getattr(args, "source_seed", 0),
            shuffle_buffer_size=getattr(args, "shuffle_buffer_size", 0),
            epoch_size=getattr(args, "epoch_size", None),
            limit_per_source=getattr(args, "limit_per_source", None),
            max_sequence_length=getattr(args, "max_sequence_length", None),
            max_protein_length=getattr(args, "max_protein_length", DEFAULT_MAX_PROTEIN_LENGTH),
            deduplicate_within_batch=getattr(args, "deduplicate_within_batch", False),
            num_workers=getattr(args, "num_workers", 0),
            prefetch_factor=getattr(args, "prefetch_factor", 4),
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
        names = sorted({item for item in self.sources if item})
        # A source may legitimately lack a validation split (e.g. small
        # supervised aux sources like `neutralization`). For non-train splits we
        # skip such sources rather than hard-fail; the train split stays strict
        # (a missing train shard still raises in GrammarArrowSource).
        if split != self.split:
            names = [n for n in names if (self.grammar_data_dir / n / split).exists()]
        configs = [
            GrammarArrowSourceConfig(
                name=name,
                path=self.grammar_data_dir,
                split=split,
                weight=self.source_weight(name),
                max_records=self.limit_per_source,
                shuffle_buffer_size=self.shuffle_buffer_size,
                # Derive from source_seed (already offset per split) so train and
                # val reshuffle independently while staying reproducible per rank.
                shuffle_seed=source_seed,
            )
            for name in names
        ]
        sources = [
            SourceWithWeight(GrammarArrowSource(config), weight=config.weight)
            for config in configs
        ]
        records = WeightedMixtureDataset(sources, epoch_size=epoch_size, seed=source_seed)
        loader_max_protein = None if self.max_protein_length <= 0 else self.max_protein_length
        batches = TaskHomogeneousBatchDataset(
            records,
            batch_size=self.batch_size,
            drop_last=True,
            deduplicate_within_batch=self.deduplicate_within_batch,
            max_protein_length=loader_max_protein,
        )
        collator = GrammarBioSeqCollator(
            tokenizer=tokenizer,
            max_sequence_length=self.max_sequence_length or DEFAULT_MAX_SEQUENCE_LENGTH,
            max_protein_length=self.max_protein_length,
        )
        # With num_workers>0 the DataLoader runs the grammar encode (the expensive
        # CPU step in the collator) in a side process, overlapping it with the GPU
        # forward/backward. persistent_workers keeps that process (and its warm
        # Arrow mmap + iterator state) alive across epochs; prefetch_factor lets it
        # stage batches ahead of the training loop. Both are only valid for
        # num_workers>0. NOTE: num_workers=1 keeps the exact shard identity of
        # num_workers=0 (distributed_worker_shard -> (rank, world_size)); use >1
        # only with a DDP-consistent batch order (see mixture.distributed_worker_shard).
        loader_kwargs: dict[str, Any] = {
            "batch_size": None,
            "collate_fn": collator,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(batches, **loader_kwargs)

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
