"""BioSeq foundation training infrastructure.

This package holds the model-agnostic DDP training infrastructure (distributed
setup, LR schedule, the grad-accumulation loop, checkpointing, wandb logging,
and validation orchestration). Model/data semantics are injected through
:class:`TrainStepFns`, so the same loop can drive the grammar qwen3_vl path
without importing any entry-point script.
"""

from __future__ import annotations

from .checkpointing import ValLossTopKCheckpointManager
from .trainer import (
    BioSeqTrainer,
    DistributedContext,
    TrainStepFns,
    any_rank_nonfinite,
    count_names,
    cuda_memory_metrics,
    format_name_counts,
    is_main,
    log,
    lr_at,
    move_batch,
    setup_distributed,
    setup_wandb,
    unwrap_model,
    wandb_count_metrics,
)

__all__ = [
    "BioSeqTrainer",
    "DistributedContext",
    "TrainStepFns",
    "ValLossTopKCheckpointManager",
    "any_rank_nonfinite",
    "count_names",
    "cuda_memory_metrics",
    "format_name_counts",
    "is_main",
    "log",
    "lr_at",
    "move_batch",
    "setup_distributed",
    "setup_wandb",
    "unwrap_model",
    "wandb_count_metrics",
]
