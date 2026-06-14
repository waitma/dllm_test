"""Canonical IRBench data records and dataset loaders.

The benchmark normalizes every external TCR / PPI source into a small set of
columns so that splits, negative sampling, models, and metrics all speak the
same language regardless of the upstream file format.

TCR binding / clustering / representation records share the ``TCR_COLUMNS``
schema (paired alpha/beta when available). Prediction files always carry an
``id``, a ``score`` in ``[0, 1]`` (higher = more likely positive / binder), and
the grouping key (epitope or split) needed to reproduce the leaderboard number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Sequence

import json

import pandas as pd

# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------

# Core paired-chain TCR columns used across T1/T2/T3.
TCR_COLUMNS = [
    "cdr3a", "cdr3b",
    "va", "ja", "vb", "jb",
    "cdr1a", "cdr2a", "cdr1b", "cdr2b",
    "tcra", "tcrb",          # full-length amino-acid chains (when available)
    "peptide", "mhc",
]

# A binding example = one TCR paired with one epitope + a binary label.
BINDING_COLUMNS = TCR_COLUMNS + ["label", "source", "split"]

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
BENCH_ROOT = PROJECT_ROOT / "downstream" / "benchmark"
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass
class Prediction:
    """One row of a model prediction file."""

    id: str
    score: float
    label: int | None = None
    group: str | None = None          # epitope (binding) or cluster/label group
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = {"id": self.id, "score": float(self.score)}
        if self.label is not None:
            row["label"] = int(self.label)
        if self.group is not None:
            row["group"] = self.group
        row.update(self.extra)
        return row


# ---------------------------------------------------------------------------
# JSONL / CSV helpers
# ---------------------------------------------------------------------------

def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_predictions(preds: Sequence[Prediction] | pd.DataFrame, path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(preds, pd.DataFrame):
        preds.to_csv(path, index=False)
        return len(preds)
    df = pd.DataFrame([p.to_row() for p in preds])
    df.to_csv(path, index=False)
    return len(df)


# ---------------------------------------------------------------------------
# IMMREP23 loaders (paired-chain VDJdb, positives-only train + labelled test)
# ---------------------------------------------------------------------------

_IMMREP23_RENAME = {
    "Peptide": "peptide", "HLA": "mhc",
    "Va": "va", "Ja": "ja", "Vb": "vb", "Jb": "jb",
    "TCRa": "tcra", "TCRb": "tcrb",
    "CDR1a": "cdr1a", "CDR2a": "cdr2a", "CDR3a": "cdr3a",
    "CDR1b": "cdr1b", "CDR2b": "cdr2b", "CDR3b": "cdr3b",
}

IMMREP23_DIR = BENCH_ROOT / "baselines" / "IMMREP23" / "data"


def _normalize_tcr_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_IMMREP23_RENAME)
    for col in TCR_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Strip whitespace / coerce NaN to empty for sequence fields.
    for col in TCR_COLUMNS:
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def load_immrep23_train(path: str | Path | None = None) -> pd.DataFrame:
    """VDJdb paired-chain positives (Target == 1). Returns normalized columns."""
    path = Path(path or (IMMREP23_DIR / "VDJdb_paired_chain.csv"))
    df = pd.read_csv(path)
    df = _normalize_tcr_frame(df)
    df["label"] = pd.to_numeric(df.get("Target", 1), errors="coerce").fillna(1).astype(int)
    df["source"] = "immrep23_vdjdb"
    return df


def load_immrep23_test(path: str | Path | None = None) -> pd.DataFrame:
    """IMMREP23 solutions.csv: labelled test with Public/Private Usage column."""
    path = Path(path or (IMMREP23_DIR / "solutions.csv"))
    df = pd.read_csv(path)
    df = _normalize_tcr_frame(df)
    df["label"] = pd.to_numeric(df["Label"], errors="coerce").astype(int)
    df["usage"] = df.get("Usage", "")
    df["source"] = "immrep23_solutions"
    if "ID" in df.columns:
        df["id"] = df["ID"].astype(str)
    else:
        df["id"] = [str(i) for i in range(len(df))]
    return df


def clonotype_key(df: pd.DataFrame, level: str = "ab") -> pd.Series:
    """Clonotype identity used for cross-split dedup.

    ``ab``: paired CDR3 alpha+beta (default, strictest for paired data).
    ``b`` : CDR3 beta only (use when alpha is missing / single-chain).
    """
    if level == "b":
        return df["cdr3b"].astype(str)
    if level == "a":
        return df["cdr3a"].astype(str)
    return df["cdr3a"].astype(str) + "|" + df["cdr3b"].astype(str)


__all__ = [
    "TCR_COLUMNS", "BINDING_COLUMNS", "Prediction",
    "PROJECT_ROOT", "BENCH_ROOT", "DATA_ROOT", "IMMREP23_DIR",
    "write_jsonl", "read_jsonl", "write_predictions",
    "load_immrep23_train", "load_immrep23_test", "clonotype_key",
]
