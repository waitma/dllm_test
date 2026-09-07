#!/usr/bin/env python3
"""Shared helpers for the tiered tcr_native build: unified schema, HLA pseudo
mapping, allele normalization, relation canonicalization, CDR3 core key."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DATA = PROJECT_ROOT / "data"
COMMON_HLA = DATA / "ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/raw_data/common_hla_sequence.csv"
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# Unified tier schema (column order is the on-disk CSV header).
UNIFIED_COLUMNS = [
    "record_id",
    "source",
    "fv_source",
    "tier",
    "task_type",
    "relation",
    "sequence_scope",
    "alpha_fv",
    "beta_fv",
    "cdr3a",
    "cdr3b",
    "epitope_seq",
    "mhc_seq",
    "mhc_allele_norm",
    "donor",
    "provenance",
]


def _load_records_module():
    path = PROJECT_ROOT / "dllm/pipelines/qwen3_vl_arch/data/records.py"
    spec = importlib.util.spec_from_file_location("_tcrnative_records", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tcrnative_records"] = mod
    spec.loader.exec_module(mod)
    return mod


_REC = _load_records_module()
normalize_sequence = _REC.normalize_sequence
is_valid_protein_sequence = _REC.is_valid_protein_sequence

# ESMC singles that RemapCollator cannot map (ids 29/30/31). Ingest used to
# accept them because records.VALID_PROTEIN_CHARS includes ".-". Strip before
# a sequence is written to a training CSV.
ALIGNMENT_GAPS = ".-|"


def strip_alignment_gaps(seq: str) -> str:
    return (seq or "").translate(str.maketrans("", "", ALIGNMENT_GAPS))


# --------------------------------------------------------------------------- #
# Relation (mirrors datasets.canonical_recognition_relation, standalone copy so
# this module has no torch dependency).
# --------------------------------------------------------------------------- #
_POS = {"binding", "binder", "positive", "pos", "true", "1", "yes"}
_NEG = {"nonbinding", "non_binding", "nonbinder", "non_binder", "negative", "neg", "false", "0", "no"}


def canonical_relation(value: Any, *, default: str = "binding") -> str:
    raw = str(value if value is not None else "").strip().lower().replace(" ", "_").replace("-", "_")
    if not raw:
        return default
    if raw in _POS:
        return "binding"
    if raw in _NEG:
        return "nonbinding"
    raise ValueError(f"Unrecognized recognition relation {value!r}")


# --------------------------------------------------------------------------- #
# HLA allele normalization + 34aa pseudo-sequence mapping
# --------------------------------------------------------------------------- #

def norm_allele(allele: Any) -> str:
    """Normalize an HLA allele string to the ``HLA-A02:01`` two-field form used
    by common_hla_sequence.csv (drop the ``*`` and any 3rd/4th field)."""

    a = str(allele or "").strip()
    if not a:
        return ""
    a = a.replace("*", "")
    if ":" in a:
        parts = a.split(":")
        a = ":".join(parts[:2])
    return a


def load_hla_pseudo() -> dict[str, str]:
    table: dict[str, str] = {}
    with COMMON_HLA.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = norm_allele(row["HLA_type"])
            if key and key not in table:
                table[key] = row["HLA_sequence"].strip()
    return table


def cdr3_core(seq: Any, *, has_anchors: bool) -> str:
    """Anchor-free CDR3 core (IMGT loop). ``has_anchors``: full junction C..[FW]."""

    s = strip_alignment_gaps(normalize_sequence(seq))
    if not s:
        return ""
    if has_anchors and len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def record_id(source: str, *parts: Any) -> str:
    digest = hashlib.sha1(("|".join(str(p) for p in (source, *parts))).encode()).hexdigest()[:16]
    return f"{source}_{digest}"


def empty_row() -> dict[str, str]:
    return {c: "" for c in UNIFIED_COLUMNS}
