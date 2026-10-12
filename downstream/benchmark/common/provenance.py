"""Checkpoint provenance helpers for downstream benchmark outputs."""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path


def _extract_checkpoint_path(spec_or_path: str) -> Path | None:
    """Return absolute checkpoint path if *spec_or_path* references a ``.pt`` file."""
    if not spec_or_path:
        return None
    spec = spec_or_path.strip()
    if spec.endswith(".pt") and (spec.startswith("/") or spec.startswith(".")):
        p = Path(spec)
        return p.resolve() if p.is_file() else None
    if spec.startswith("grammar:"):
        rest = spec.split(":", 1)[1]
        parts = rest.split(":")
        if parts and parts[0] in ("decoder", "encoder"):
            if len(parts) >= 3 and parts[1] in (
                    "global", "wholefeat", "whole", "sequence", "seq",
                    "segment", "seg", "segment_concat"):
                path = ":".join(parts[2:])
            else:
                path = ":".join(parts[1:])
        else:
            path = rest
        if path.endswith(".pt"):
            p = Path(path)
            return p.resolve() if p.is_file() else None
        return None
    for prefix in ("bioseq-llada:", "bioseq:"):
        if spec.startswith(prefix):
            path = spec.split(":", 1)[1]
            if path.endswith(".pt"):
                p = Path(path)
                return p.resolve() if p.is_file() else None
            return None
    return None


def checkpoint_provenance(spec_or_path: str | None) -> dict | None:
    """Build provenance dict for a checkpoint referenced by an embedder spec."""
    if not spec_or_path:
        return None
    ckpt = _extract_checkpoint_path(spec_or_path)
    if ckpt is None:
        return None
    st = ckpt.stat()
    mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    h = hashlib.sha256()
    with ckpt.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {
        "path": str(ckpt),
        "mtime": mtime,
        "size_bytes": int(st.st_size),
        "sha256": h.hexdigest(),
    }
