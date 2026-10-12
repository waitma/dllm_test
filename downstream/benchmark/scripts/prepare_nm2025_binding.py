#!/usr/bin/env python
"""Prepare the official Nature Methods 2025 TCR-epitope binding splits.

Consumes the official figshare ``original.zip`` (paper test sets S_Data/U_Data)
and, unless ``--tests-only`` is passed, the GitHub ``train.csv`` /
``train_CDR3B_others.csv`` retraining sets. It normalizes files into the
IRBench ``common.schema`` columns and writes:

    data/tcr_binding_nm2025/
        train/{cdr3b,others}/train.csv
        seen/{cdr3b/{AS,PS,HS},others/AS}/test.csv
        unseen/{cdr3b/{AS,PS,HS},others/AS}/test.csv
        original_test_manifest.json
        leakage_report.json

The zip member names are NOT UTF-8 (the greek beta is mojibake), so members are
always indexed via ``zipfile.namelist()`` and matched on the ASCII path tokens
that survive the corruption (``seen``/``unseen``, ``only``/``others``,
``AS``/``PS``/``HS``, ``.csv``). CD-HIT similarity dedup was already applied by
the paper authors (seen and unseen epitope sets are disjoint); this script only
records that fact, it does not re-dedup.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import schema, leakage  # noqa: E402
from tcr_binding.original_protocol import (  # noqa: E402
    ORIGINAL_ZIP,
    ORIGINAL_ZIP_SHA256,
)

OFFICIAL_REPO = BENCH / "data" / "tcr_binding_nm2025" / "official_repo"
OUT_ROOT = BENCH / "data" / "tcr_binding_nm2025"
TEST_MANIFEST = OUT_ROOT / "original_test_manifest.json"

# Columns we persist (schema TCR columns + labels/grouping/provenance).
OUT_COLUMNS = [
    "id", "cdr3a", "cdr3b", "va", "ja", "vb", "jb",
    "tcra", "tcrb", "peptide", "mhc", "label", "group", "seen",
    "track", "neg_source", "source",
]


def _norm(df: pd.DataFrame, *, split: str, track: str, neg_source: str,
          source: str) -> pd.DataFrame:
    """Map an official CSV to the IRBench schema."""
    n = len(df)

    def col(name, default=""):
        if name in df.columns:
            return df[name].fillna("").astype(str).str.strip()
        return pd.Series([default] * n, index=df.index)

    out = pd.DataFrame({
        "cdr3a": col("CDR3A"),
        "cdr3b": col("CDR3B"),
        "va": col("TRAV"),
        "ja": col("TRAJ"),
        "vb": col("TRBV"),
        "jb": col("TRBJ"),
        "tcra": col("LongA"),
        "tcrb": col("LongB"),
        "peptide": col("Epitope"),
        "mhc": col("MHC"),
    })
    out["label"] = pd.to_numeric(df["Affinity"], errors="coerce").astype("Int64")
    out["group"] = out["peptide"]
    out["seen"] = split
    out["track"] = track
    out["neg_source"] = neg_source
    out["source"] = source
    out["id"] = [f"{source}-{i}" for i in range(n)]
    return out[OUT_COLUMNS]


def _match(names: list[str], *, split: str, others: bool, neg: str) -> str:
    """Find the single zip member for a (split, track, neg-source) triple."""
    want_only = "others" if others else "only"
    cands = []
    for m in names:
        if not m.endswith(".csv"):
            continue
        if f"/{neg}/" not in m:
            continue
        # split token: the leaf dir prefix is "seen_" / "unseen_"; the raw/
        # positives-only tree lacks the AS/PS/HS layer so it is excluded above.
        low = m.lower()
        if split == "seen" and "unseen" in low:
            continue
        if split not in low:
            continue
        if want_only == "others" and "others" not in low:
            continue
        if want_only == "only" and "only" not in low:
            continue
        cands.append(m)
    if len(cands) != 1:
        raise RuntimeError(f"expected 1 member for split={split} others={others} "
                           f"neg={neg}, got {cands}")
    return cands[0]


def _read_zip_csv(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as fh:
        raw = fh.read()
    df = pd.read_csv(io.BytesIO(raw))
    # Drop pandas index dumps ("Unnamed: 0", "Unnamed: 0.1", ...).
    drop = [c for c in df.columns if str(c).startswith("Unnamed")]
    return df.drop(columns=drop)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(*, tests_only: bool = False) -> dict:
    if not ORIGINAL_ZIP.exists():
        raise FileNotFoundError(ORIGINAL_ZIP)
    zip_sha256 = _sha256(ORIGINAL_ZIP)
    if zip_sha256 != ORIGINAL_ZIP_SHA256:
        raise RuntimeError(
            f"unexpected original.zip sha256={zip_sha256}; expected={ORIGINAL_ZIP_SHA256}"
        )
    zf = zipfile.ZipFile(ORIGINAL_ZIP)
    names = zf.namelist()

    written: dict[str, dict] = {}
    test_files: dict[str, dict] = {}
    # (track, others, neg-sources)
    plans = [
        ("cdr3b", False, ["AS", "PS", "HS"]),
        ("others", True, ["AS"]),
    ]
    frames: dict[tuple[str, str, str], pd.DataFrame] = {}
    for track, others, negs in plans:
        for split in ("seen", "unseen"):
            for neg in negs:
                member = _match(names, split=split, others=others, neg=neg)
                df = _read_zip_csv(zf, member)
                norm = _norm(df, split=split, track=track, neg_source=neg,
                             source=f"nm2025_{split}_{track}_{neg}")
                out_dir = OUT_ROOT / split / track / neg
                out_dir.mkdir(parents=True, exist_ok=True)
                norm.to_csv(out_dir / "test.csv", index=False)
                frames[(track, split, neg)] = norm
                relative = str((out_dir / "test.csv").relative_to(OUT_ROOT))
                entry = {
                    "member": member,
                    "n": int(len(norm)),
                    "n_pos": int((norm["label"] == 1).sum()),
                    "n_neg": int((norm["label"] == 0).sum()),
                    "n_epitopes": int(norm["peptide"].nunique()),
                    "sha256": _sha256(out_dir / "test.csv"),
                }
                written[relative] = entry
                test_files[relative] = entry
    zf.close()

    test_manifest = {
        "schema_version": 1,
        "source_artifact": str(ORIGINAL_ZIP),
        "source_artifact_sha256": zip_sha256,
        "files": test_files,
    }
    TEST_MANIFEST.write_text(json.dumps(test_manifest, indent=2) + "\n")

    # Training sets (GitHub retraining CSVs).
    train_specs = [
        ("cdr3b", OFFICIAL_REPO / "train.csv"),
        ("others", OFFICIAL_REPO / "train_CDR3B_others.csv"),
    ]
    train_frames: dict[str, pd.DataFrame] = {}
    if not tests_only:
        for track, path in train_specs:
            if not path.exists():
                raise FileNotFoundError(path)
            df = pd.read_csv(path)
            df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
            norm = _norm(df, split="train", track=track, neg_source="official_train",
                         source=f"nm2025_train_{track}")
            out_dir = OUT_ROOT / "train" / track
            out_dir.mkdir(parents=True, exist_ok=True)
            norm.to_csv(out_dir / "train.csv", index=False)
            train_frames[track] = norm
            written[str((out_dir / "train.csv").relative_to(OUT_ROOT))] = {
                "path": str(path),
                "n": int(len(norm)),
                "n_pos": int((norm["label"] == 1).sum()),
                "n_neg": int((norm["label"] == 0).sum()),
                "n_epitopes": int(norm["peptide"].nunique()),
            }

    # Leakage report: assert seen ∩ unseen epitopes == 0 per track; record
    # train ↔ test clonotype overlap (CD-HIT already applied by the authors).
    report: dict = {"files": written, "tracks": {}}
    for track in ("cdr3b", "others"):
        seen_ep = set(frames[(track, "seen", "AS")]["peptide"])
        unseen_ep = set(frames[(track, "unseen", "AS")]["peptide"])
        overlap = sorted(seen_ep & unseen_ep)
        clono_keys = ("cdr3b",) if track == "cdr3b" else ("cdr3a", "cdr3b")
        rep_track = {
            "n_seen_epitopes": len(seen_ep),
            "n_unseen_epitopes": len(unseen_ep),
            "seen_unseen_epitope_overlap": len(overlap),
            "seen_unseen_overlap_list": overlap,
        }
        if not tests_only:
            tr = train_frames[track]
            rep_track["train_seen"] = leakage.leakage_report(
                tr, frames[(track, "seen", "AS")], clono_keys=clono_keys)
            rep_track["train_unseen"] = leakage.leakage_report(
                tr, frames[(track, "unseen", "AS")], clono_keys=clono_keys)
        assert len(overlap) == 0, f"{track}: seen∩unseen epitopes must be 0, got {overlap}"
        report["tracks"][track] = rep_track

    report["mode"] = "tests_only" if tests_only else "tests_and_train"
    report["original_test_manifest"] = str(TEST_MANIFEST)
    if not tests_only:
        (OUT_ROOT / "leakage_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tests-only",
        action="store_true",
        help="extract and manifest original.zip tests without opening train.csv",
    )
    args = ap.parse_args()
    report = prepare(tests_only=args.tests_only)
    print(json.dumps(report, indent=2)[:4000])
    print(f"\n-> {OUT_ROOT}")


if __name__ == "__main__":
    main()
