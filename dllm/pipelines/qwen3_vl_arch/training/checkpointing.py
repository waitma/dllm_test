"""Validation-loss top-k checkpoint retention for BioSeq training.

Keeps up to ``save_top_k`` full training checkpoints ranked by ``val/loss``
(lower is better), writes a manifest for inspection, and refreshes ``best.pt``
as a stable alias to the current best checkpoint.

Run (import-only; exercised via :class:`~.trainer.BioSeqTrainer` and unit tests)::

    from dllm.pipelines.qwen3_vl_arch.training.checkpointing import ValLossTopKCheckpointManager
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

VOCAB_WEIGHT_SUFFIXES = (
    ".token_embeddings.weight",
    ".lm_head.weight",
)


def _is_vocab_weight_key(key: str) -> bool:
    return any(key.endswith(suffix) for suffix in VOCAB_WEIGHT_SUFFIXES)


def merge_vocab_expanded_weight(current: torch.Tensor, checkpoint: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Copy overlapping vocab rows from ``checkpoint`` into a clone of ``current``.

    Returns the merged tensor and whether vocab grew (checkpoint rows < current rows).
    """
    if current.shape == checkpoint.shape:
        return checkpoint, False
    if current.dim() != 2 or checkpoint.dim() != 2 or current.shape[1] != checkpoint.shape[1]:
        raise RuntimeError(
            f"Cannot adapt vocab weight with incompatible shapes: "
            f"model {tuple(current.shape)} vs checkpoint {tuple(checkpoint.shape)}"
        )
    if current.shape[0] < checkpoint.shape[0]:
        raise RuntimeError(
            f"Checkpoint vocab ({checkpoint.shape[0]}) is larger than the current model "
            f"({current.shape[0]}); refusing to truncate embeddings on resume."
        )
    merged = current.clone()
    rows = int(checkpoint.shape[0])
    merged[:rows].copy_(checkpoint)
    return merged, True


def adapt_model_state_dict_for_resume(
    module: torch.nn.Module,
    checkpoint_state: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], bool]:
    """Align a checkpoint state dict with the live module, expanding vocab rows if needed."""
    current_state = module.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    vocab_expanded = False
    for key, checkpoint_value in checkpoint_state.items():
        if key not in current_state:
            continue
        current_value = current_state[key]
        if current_value.shape == checkpoint_value.shape:
            adapted[key] = checkpoint_value
            continue
        if _is_vocab_weight_key(key):
            merged, expanded = merge_vocab_expanded_weight(current_value, checkpoint_value)
            adapted[key] = merged
            vocab_expanded = vocab_expanded or expanded
            continue
        raise RuntimeError(
            f"Checkpoint/model size mismatch for {key}: "
            f"checkpoint {tuple(checkpoint_value.shape)} vs model {tuple(current_value.shape)}"
        )
    return adapted, vocab_expanded


def load_resume_payload(
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    payload: dict[str, Any],
    *,
    log: Callable[[str], None] | None = None,
) -> tuple[int, bool]:
    """Load model (+ optimizer when compatible) from a training checkpoint payload."""
    emit = log or (lambda _message: None)
    checkpoint_state = payload["model_state_dict"]
    adapted_state, vocab_expanded = adapt_model_state_dict_for_resume(module, checkpoint_state)
    missing, unexpected = module.load_state_dict(adapted_state, strict=False)
    if missing:
        raise RuntimeError(f"[resume] missing keys after vocab-adapted load: {missing}")
    if unexpected:
        raise RuntimeError(f"[resume] unexpected keys after vocab-adapted load: {unexpected}")
    if vocab_expanded:
        old_rows = next(
            iter(
                value.shape[0]
                for key, value in checkpoint_state.items()
                if _is_vocab_weight_key(key)
            ),
            None,
        )
        new_rows = next(
            iter(
                value.shape[0]
                for key, value in adapted_state.items()
                if _is_vocab_weight_key(key)
            ),
            None,
        )
        emit(
            f"[resume] expanded vocab embeddings from {old_rows} to {new_rows} rows; "
            "new token rows keep current initialization"
        )

    if vocab_expanded:
        emit("[resume] skipped optimizer state because vocab size changed")
    else:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    step = int(payload.get("step", 0))
    return step, vocab_expanded


MANIFEST_NAME = "topk_manifest.json"
CHECKPOINTS_SUBDIR = "checkpoints"
BEST_CHECKPOINT_NAME = "best.pt"


@dataclass(frozen=True)
class TopKCheckpointEntry:
    step: int
    val_loss: float
    path: Path

    def to_json(self, *, output_dir: Path | None = None) -> dict[str, Any]:
        path = self.path
        if output_dir is not None:
            try:
                path = self.path.relative_to(output_dir)
            except ValueError:
                pass
        return {
            "step": self.step,
            "val_loss": self.val_loss,
            "path": str(path),
        }

    @classmethod
    def from_json(cls, item: dict[str, Any], output_dir: Path) -> "TopKCheckpointEntry":
        raw_path = Path(str(item["path"]))
        path = raw_path if raw_path.is_absolute() else output_dir / raw_path
        return cls(step=int(item["step"]), val_loss=float(item["val_loss"]), path=path)


def checkpoint_filename(step: int, val_loss: float) -> str:
    return f"step_{step:07d}_val_{val_loss:.4f}.pt"


class ValLossTopKCheckpointManager:
    """Retain the K lowest-val-loss checkpoints on disk."""

    def __init__(
        self,
        output_dir: Path,
        *,
        save_top_k: int = 10,
        metric_key: str = "val/loss",
    ) -> None:
        if save_top_k < 0:
            raise ValueError("save_top_k must be >= 0")
        self.output_dir = output_dir
        self.save_top_k = save_top_k
        self.metric_key = metric_key
        self.checkpoint_dir = output_dir / CHECKPOINTS_SUBDIR
        self.manifest_path = self.checkpoint_dir / MANIFEST_NAME
        self.entries: list[TopKCheckpointEntry] = []
        if save_top_k > 0:
            self._load_manifest()

    @property
    def enabled(self) -> bool:
        return self.save_top_k > 0

    def _load_manifest(self) -> None:
        if not self.manifest_path.is_file():
            return
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.entries = [
            TopKCheckpointEntry.from_json(item, self.output_dir)
            for item in payload.get("checkpoints", [])
        ]
        self.entries.sort(key=lambda entry: entry.val_loss)
        self.entries = self.entries[: self.save_top_k]

    def _write_manifest(self) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric": self.metric_key,
            "save_top_k": self.save_top_k,
            "checkpoints": [entry.to_json(output_dir=self.output_dir) for entry in self.entries],
        }
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.manifest_path)

    def should_save(self, val_loss: float) -> bool:
        if not self.enabled:
            return False
        if len(self.entries) < self.save_top_k:
            return True
        return val_loss < max(entry.val_loss for entry in self.entries)

    def maybe_save(
        self,
        step: int,
        val_loss: float,
        payload: dict[str, Any],
        *,
        save_payload: Callable[[Path, dict[str, Any]], None],
    ) -> TopKCheckpointEntry | None:
        if not self.should_save(val_loss):
            return None

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        target = self.checkpoint_dir / checkpoint_filename(step, val_loss)
        save_payload(target, payload)

        self.entries.append(TopKCheckpointEntry(step=step, val_loss=val_loss, path=target))
        self.entries.sort(key=lambda entry: entry.val_loss)

        while len(self.entries) > self.save_top_k:
            removed = self.entries.pop()
            if removed.path.is_file():
                removed.path.unlink()

        self._write_manifest()
        self._refresh_best_alias(save_payload)
        return TopKCheckpointEntry(step=step, val_loss=val_loss, path=target)

    def _refresh_best_alias(self, save_payload: Callable[[Path, dict[str, Any]], None]) -> None:
        best = self.best_entry
        if best is None or not best.path.is_file():
            return
        payload = torch.load(best.path, map_location="cpu")
        save_payload(self.output_dir / BEST_CHECKPOINT_NAME, payload)

    @property
    def best_entry(self) -> TopKCheckpointEntry | None:
        if not self.entries:
            return None
        return self.entries[0]
