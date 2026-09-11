"""Map-style loader for prepared semantic immune JSONL shards.

The loader indexes byte offsets once during construction. ``__getitem__`` only
reads one already-prepared JSON object and restores ``BioSeqRecord``; it never
parses a source-specific row, loads blocklists, filters, retries, or replaces a
bad sample.
"""

from __future__ import annotations

import json
import os
from array import array
from bisect import bisect_right
from collections.abc import Iterator
from pathlib import Path
from typing import Any, TextIO

from torch.utils.data import Dataset

from .preprocessing.validators import validate_manifest, validate_prepared_row
from .records import BioSeqRecord


class PreparedImmuneDataset(Dataset[BioSeqRecord]):
    def __init__(self, dataset_dir: str | Path, *, split: str = "train", source: str | None = None) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest_path = self.dataset_dir / "dataset_manifest.json"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Prepared dataset manifest not found: {self.manifest_path}")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        split_info = manifest["splits"].get(split)
        if split_info is None:
            raise KeyError(f"Prepared split {split!r} is not present in {self.manifest_path}")
        self.split = split
        self.source = source
        # Keep one compact uint64 offset array per shard instead of one Python
        # tuple per record. This matters for multi-million-row datasets: the
        # index remains in memory, but does not approach the size of the JSONL.
        self._shards: list[tuple[Path, array]] = []
        self._cumulative_counts: list[int] = []
        total = 0
        for shard in split_info.get("shards", []):
            if source is not None and shard.get("source") != source:
                continue
            path = self.dataset_dir / str(shard["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Prepared shard listed by manifest is missing: {path}")
            offsets = array("Q")
            with path.open("r", encoding="utf-8") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if line.strip():
                        offsets.append(offset)
            if offsets:
                self._shards.append((path, offsets))
                total += len(offsets)
                self._cumulative_counts.append(total)
        if not self._shards:
            selected = f" for source {source!r}" if source else ""
            raise ValueError(f"Prepared split {split!r}{selected} contains no records")
        self._length = total
        # Handles are deliberately created lazily, so the dataset remains
        # picklable for spawned DataLoader workers. Each worker gets its own
        # handle cache and never shares a seek position with another worker.
        self._handles: dict[Path, TextIO] = {}
        self._handle_pid = os.getpid()

    def __len__(self) -> int:
        return self._length

    def _location(self, index: int) -> tuple[Path, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(f"Prepared dataset index out of range: {index}")
        shard_index = bisect_right(self._cumulative_counts, index)
        previous = self._cumulative_counts[shard_index - 1] if shard_index else 0
        path, offsets = self._shards[shard_index]
        return path, int(offsets[index - previous])

    def _read_line(self, path: Path, offset: int) -> str:
        # A fork can inherit a handle opened by the parent after an earlier
        # direct access. Drop inherited handles when the process changes.
        pid = os.getpid()
        if pid != self._handle_pid:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
            self._handle_pid = pid
        handle = self._handles.get(path)
        if handle is None:
            handle = path.open("r", encoding="utf-8")
            self._handles[path] = handle
        handle.seek(offset)
        return handle.readline()

    def __getitem__(self, index: int) -> BioSeqRecord:
        path, offset = self._location(index)
        row: dict[str, Any] = json.loads(self._read_line(path, offset))
        return validate_prepared_row(row)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except OSError:
                pass

    def iter_records(self) -> Iterator[BioSeqRecord]:
        for index in range(len(self)):
            yield self[index]


def load_prepared_dataset(dataset_dir: str | Path, *, split: str = "train", source: str | None = None) -> PreparedImmuneDataset:
    return PreparedImmuneDataset(dataset_dir, split=split, source=source)


__all__ = ["PreparedImmuneDataset", "load_prepared_dataset"]
