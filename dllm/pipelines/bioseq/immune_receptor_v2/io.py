"""Streaming and atomic I/O helpers for immune-receptor v2 artifacts.

The main builder calls these helpers; they can also be tested with::

    conda run -n pllm python -m pytest \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/bioseq/test_immune_receptor_v2.py -q
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from .schema import CanonicalRecord


def sha256_file(path: Path | str, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    return value.to_dict() if hasattr(value, "to_dict") else value


def write_json(path: Path | str, value: Any) -> None:
    """Atomically write a deterministic, human-readable JSON artifact."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=_jsonable)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)


def write_jsonl(path: Path | str, records: Iterable[Any]) -> int:
    """Atomically stream records into JSONL and return the number written."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            value = _jsonable(record)
            handle.write(
                json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    default=_jsonable,
                )
                + "\n"
            )
            count += 1
    os.replace(temporary, output)
    return count


def iter_jsonl(
    path: Path | str, *, as_records: bool = False
) -> Iterator[dict[str, Any] | CanonicalRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row: Mapping[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            yield CanonicalRecord.from_dict(row) if as_records else dict(row)
