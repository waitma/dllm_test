"""Atomic writers for semantic immune dataset shards.

Run through ``scripts/data/preprocess_immune_dataset.py``; this module is kept
small so preprocessing can stream large CSV/JSONL sources without materializing
records in memory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class JsonlShardWriter:
    """Write fixed-size JSONL shards and atomically publish each completed shard."""

    def __init__(self, output_dir: Path, split: str, source: str, shard_size: int) -> None:
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = output_dir
        self.split = split
        self.source = source
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._tmp_path: Path | None = None
        self._shard_index = 0
        self._rows_in_shard = 0
        self._published: list[dict[str, Any]] = []

    @property
    def published(self) -> list[dict[str, Any]]:
        return list(self._published)

    def _open(self) -> None:
        final_name = f"{self.source}-{self._shard_index:05d}.jsonl"
        final_path = self.output_dir / final_name
        self._tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        self._handle = self._tmp_path.open("w", encoding="utf-8")
        self._rows_in_shard = 0

    def write(self, row: dict[str, Any]) -> None:
        if self._handle is None:
            self._open()
        assert self._handle is not None
        self._handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._rows_in_shard += 1
        if self._rows_in_shard >= self.shard_size:
            self._publish_current()

    def _publish_current(self) -> None:
        if self._handle is None or self._tmp_path is None:
            return
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        final_path = self._tmp_path.with_suffix("")
        os.replace(self._tmp_path, final_path)
        self._published.append({
            "path": final_path.name,
            "source": self.source,
            "split": self.split,
            "records": self._rows_in_shard,
        })
        self._handle = None
        self._tmp_path = None
        self._shard_index += 1
        self._rows_in_shard = 0

    def close(self) -> None:
        self._publish_current()

    def abort(self) -> None:
        if self._handle is not None:
            self._handle.close()
        if self._tmp_path is not None:
            self._tmp_path.unlink(missing_ok=True)
        self._handle = None
        self._tmp_path = None

    def __enter__(self) -> "JsonlShardWriter":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()



def atomic_json_dump(path: Path, payload: dict[str, Any]) -> None:
    """Publish one JSON document only after the complete payload is serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


__all__ = ["JsonlShardWriter", "atomic_json_dump"]
