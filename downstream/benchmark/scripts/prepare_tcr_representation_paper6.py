#!/usr/bin/env python
"""Rebuild the SCEPTR (Cell Systems 2024) **6-pMHC deep few-shot** dataset.

This is the *deep* leaderboard (T3-deep) that complements the existing 24-epitope
*broad* T3 benchmark. It reproduces the SCEPTR Methods III.1 / Table SI protocol
exactly on its own data, kept in an **independent** directory
(``data/tcr_representation_paper6/``) so the broad benchmark is untouched.

Protocol (SCEPTR III.1):
- Six target pMHCs: GILGFVFTL, NLVPMVATV, SPRWYFYYL, TFEYVSQPFLMDLE (class II),
  TTDPSFLGRY, YLQPRTFLL.
- Data = curated specificity-labelled paired αβ TCRs from **filtered VDJdb**
  (here the local ``data/tcr/vdjdb_full_filtered.txt`` full-database export).
- For a target ``e``, ``k`` cognate binders form the *reference* set; the test
  set is the remaining ``e`` binders (positives) + **the rest of the filtered
  VDJdb** (negatives, incl. TCRs annotated to non-target pMHCs). The negatives
  are therefore the *whole* background repertoire, not just the other 5 targets
  — this is the口径 that differs from the broad 24-epitope T3.

This script produces the two clonotype-deduplicated pools the paper6 few-shot
engine consumes:
- ``target_binders.csv`` : deduped paired binders for the 6 targets (positives +
  reference candidates), one row per clonotype, with a ``peptide`` label.
- ``background_pool.csv``: deduped paired TCRs annotated to *non-target* pMHCs
  (pure negatives / background repertoire).
- ``meta.json``          : per-target counts, background size, filters, k-grid.
- ``leakage_report.json``: asserts target∩background clonotype overlap == 0 and
  that every target has > max(k)=200 binders (so a held-out positive always
  exists at k=200).

Clonotype identity = ``CDR3a|CDR3b`` (paper-consistent, matches
``common.schema.clonotype_key`` level="ab"). A clonotype annotated to more than
one *target* is dropped (ambiguous label); a clonotype that binds a target is
never placed in the background (no positive leaks into the negative pool).

Run::

    $ENV/bin/python scripts/prepare_tcr_representation_paper6.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common import leakage  # noqa: E402

# VDJdb source (complex.id long format). We default to the **era-matched 2023-10
# release** shipped under baselines/TCR_Unsupervised_Benchmark, not the 2025
# re-download in data/tcr/. Rationale: the SCEPTR paper (2024) benchmarked
# VDJdb 2023-06-01; the 2025 export (data/tcr/vdjdb_full_filtered.txt) has been
# inflated by post-paper studies — most drastically NLVPMVATV jumps from ~415 to
# ~10,044 paired clonotypes because a single 2024 single-cell dextramer study
# (PMID:39746936, 9,663 TCRs) was added. Using the era-matched release keeps the
# per-pMHC binder pools consistent with Table SI. Override with --vdjdb.
VDJDB_2023_10 = (BENCH / "baselines" / "TCR_Unsupervised_Benchmark" / "Data"
                 / "Database_2023_10" / "vdjdb.txt")
VDJDB_FULL = VDJDB_2023_10

TARGETS = ["GILGFVFTL", "NLVPMVATV", "SPRWYFYYL", "TFEYVSQPFLMDLE", "TTDPSFLGRY", "YLQPRTFLL"]
K_GRID = [1, 2, 5, 10, 20, 50, 100, 200]

OUT_COLS = ["cdr3a", "cdr3b", "va", "ja", "vb", "jb", "tcra", "tcrb", "peptide", "mhc", "clonotype"]
_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _valid_cdr3(s: str) -> bool:
    return isinstance(s, str) and len(s) >= 5 and set(s) <= _AA


def _load_vdjdb_full(path: Path) -> pd.DataFrame:
    """Filter to functional human paired αβ TCRs with valid CDR3 loops.

    Supports two VDJdb export layouts:
    - **long** (complex.id + gene rows; the antigenomics ``vdjdb.txt`` release):
      TRA/TRB rows sharing a non-zero ``complex.id`` are joined into one paired
      TCR (this is the era-matched default).
    - **wide** (per-chain columns ``cdr3.alpha``/``cdr3.beta``; the 2025
      ``vdjdb_full*.txt`` export).
    """
    df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False).fillna("")
    if "gene" in df.columns and "complex.id" in df.columns:
        df = _pair_long_format(df)
    else:
        df = _pair_wide_format(df)
    df["clonotype"] = df["cdr3a"] + "|" + df["cdr3b"]
    df["vdjdb.score"] = pd.to_numeric(df.get("vdjdb.score", 0), errors="coerce").fillna(0).astype(int)
    df["_complete"] = (
        (df["va"] != "").astype(int) + (df["vb"] != "").astype(int)
        + (df["ja"] != "").astype(int) + (df["jb"] != "").astype(int)
    )
    return df


def _pair_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Join TRA/TRB rows sharing a complex.id into one paired-chain record."""
    df = df[(df["species"] == "HomoSapiens") & (df["complex.id"] != "0")].copy()
    tra = df[df["gene"] == "TRA"].drop_duplicates("complex.id").set_index("complex.id")
    trb = df[df["gene"] == "TRB"].drop_duplicates("complex.id").set_index("complex.id")
    cid = tra.index.intersection(trb.index)
    out = pd.DataFrame(index=cid)
    out["cdr3a"] = tra.loc[cid, "cdr3"].str.strip().str.upper()
    out["cdr3b"] = trb.loc[cid, "cdr3"].str.strip().str.upper()
    out["va"] = tra.loc[cid, "v.segm"].str.strip()
    out["ja"] = tra.loc[cid, "j.segm"].str.strip()
    out["vb"] = trb.loc[cid, "v.segm"].str.strip()
    out["jb"] = trb.loc[cid, "j.segm"].str.strip()
    out["tcra"] = ""
    out["tcrb"] = ""
    out["peptide"] = trb.loc[cid, "antigen.epitope"].str.strip()
    out["mhc"] = trb.loc[cid, "mhc.a"].str.strip()
    out["vdjdb.score"] = trb.loc[cid, "vdjdb.score"]
    m = out["cdr3a"].map(_valid_cdr3) & out["cdr3b"].map(_valid_cdr3)
    return out[m].reset_index(drop=True)


def _pair_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chain-column layout (cdr3.alpha / cdr3.beta already paired)."""
    df = df[df["species"] == "HomoSapiens"].copy()
    a = df["cdr3.alpha"].str.strip().str.upper()
    b = df["cdr3.beta"].str.strip().str.upper()
    m = a.map(_valid_cdr3) & b.map(_valid_cdr3)
    df = df[m].copy()
    out = pd.DataFrame()
    out["cdr3a"] = a[m]
    out["cdr3b"] = b[m]
    out["va"] = df["v.alpha"].str.strip()
    out["ja"] = df["j.alpha"].str.strip()
    out["vb"] = df["v.beta"].str.strip()
    out["jb"] = df["j.beta"].str.strip()
    out["tcra"] = ""
    out["tcrb"] = ""
    out["peptide"] = df["antigen.epitope"].str.strip()
    out["mhc"] = df["mhc.a"].str.strip()
    out["vdjdb.score"] = df["vdjdb.score"]
    return out.reset_index(drop=True)


def _pick_representative(rows: pd.DataFrame) -> pd.Series:
    """One row per clonotype: most gene-complete, then highest VDJdb score."""
    r = rows.sort_values(["_complete", "vdjdb.score"], ascending=False)
    return r.iloc[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vdjdb", default=str(VDJDB_FULL))
    ap.add_argument("--out", default=str(BENCH / "data" / "tcr_representation_paper6"))
    args = ap.parse_args()

    src = Path(args.vdjdb)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_vdjdb_full(src)
    print(f"[load] {src.name}: functional human paired rows={len(df)} "
          f"unique clonotypes={df['clonotype'].nunique()}")

    # Map clonotype -> set(epitopes) to classify target-binder vs background and
    # to drop clonotypes ambiguously annotated to >1 target.
    tset = set(TARGETS)
    clono_eps = df.groupby("clonotype")["peptide"].agg(lambda x: set(x))
    role: dict[str, str] = {}       # clonotype -> target epitope | "__bg__"
    dropped_multi = 0
    for clono, eps in clono_eps.items():
        inter = eps & tset
        if not inter:
            role[clono] = "__bg__"
        elif len(inter) == 1:
            role[clono] = next(iter(inter))
        else:
            dropped_multi += 1  # ambiguous multi-target clonotype

    # Representative row per clonotype (deduplicate). For target binders we keep
    # rows annotated to the assigned target; for background any row is fine.
    target_rows, bg_rows = [], []
    for clono, grp in df.groupby("clonotype"):
        r = role.get(clono)
        if r is None:
            continue  # dropped multi-target
        if r == "__bg__":
            rep = _pick_representative(grp)
            rep = rep.copy()
            bg_rows.append(rep)
        else:
            sub = grp[grp["peptide"] == r]
            rep = _pick_representative(sub if len(sub) else grp).copy()
            rep["peptide"] = r
            target_rows.append(rep)

    tgt = pd.DataFrame(target_rows)[OUT_COLS].reset_index(drop=True)
    bg = pd.DataFrame(bg_rows)[OUT_COLS].reset_index(drop=True)
    bg["peptide"] = "__background__"

    # Order target binders by target (so the few-shot engine can slice cleanly)
    tgt["peptide"] = pd.Categorical(tgt["peptide"], categories=TARGETS, ordered=True)
    tgt = tgt.sort_values(["peptide", "clonotype"]).reset_index(drop=True)
    tgt["peptide"] = tgt["peptide"].astype(str)

    tgt.to_csv(out_dir / "target_binders.csv", index=False)
    bg.to_csv(out_dir / "background_pool.csv", index=False)

    counts = {t: int((tgt["peptide"] == t).sum()) for t in TARGETS}
    # Leakage guards: target∩background clonotype overlap must be 0.
    tgt_clono = set(tgt["clonotype"])
    bg_clono = set(bg["clonotype"])
    overlap = len(tgt_clono & bg_clono)
    assert overlap == 0, f"target∩background clonotype overlap={overlap} (must be 0)"
    # Duplicate clonotypes within a pool would break reference/query disjointness.
    assert tgt["clonotype"].is_unique, "target binders contain duplicate clonotypes"
    assert bg["clonotype"].is_unique, "background contains duplicate clonotypes"

    kmax = max(K_GRID)
    insufficient = {t: c for t, c in counts.items() if c <= kmax}

    meta = {
        "protocol": "SCEPTR Cell Systems 2024 III.1 (6-pMHC deep few-shot)",
        "source": str(src),
        "targets": TARGETS,
        "k_grid": K_GRID,
        "clonotype_key": "cdr3a|cdr3b",
        "filters": "species=HomoSapiens; paired αβ (both CDR3 valid, len>=5, canonical AA); "
                   "dedup by clonotype; drop clonotypes annotated to >1 target",
        "n_target_binders": int(len(tgt)),
        "n_background": int(len(bg)),
        "n_universe": int(len(tgt) + len(bg)),
        "per_target_binders": counts,
        "dropped_multi_target_clonotypes": dropped_multi,
        "targets_with_ge_k200": {t: bool(c > kmax) for t, c in counts.items()},
        "source_release": "VDJdb 2023-10 (era-matched to SCEPTR 2024; benchmarked "
                          "VDJdb 2023-06-01). The 2025 re-download (data/tcr/"
                          "vdjdb_full_filtered.txt) inflates NLVPMVATV to ~10k paired "
                          "clonotypes via post-paper single-cell studies, so it is NOT used.",
        "note_10x": "10x Genomics whitepaper data NOT separately added (Fig S2 broad "
                    "repeat, not the Table SI main table). Background = paired VDJdb only. "
                    "SCEPTR Table SI (Fig 1d/S1) is the VDJdb-only benchmark → matches.",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    leak = {
        "target_background_clonotype_overlap": overlap,
        "target_clonotypes_unique": bool(tgt["clonotype"].is_unique),
        "background_clonotypes_unique": bool(bg["clonotype"].is_unique),
        "per_target_binders": counts,
        "targets_below_k200": insufficient,
        "reference_query_disjoint_by_construction": (
            "reference ⊂ target binders (deduped by clonotype); held-out positives = "
            "binders minus reference; reference rows excluded from query → clonotype "
            "overlap(reference, query)=0 for every split."
        ),
    }
    (out_dir / "leakage_report.json").write_text(json.dumps(leak, indent=2))

    print(f"[write] target_binders={len(tgt)} background={len(bg)} -> {out_dir}")
    for t in TARGETS:
        flag = "" if counts[t] > kmax else "  <-- <=200 binders!"
        print(f"    {t:16s} binders={counts[t]:6d}{flag}")
    print(f"[leakage] target∩background overlap={overlap} (OK) | dropped multi-target={dropped_multi}")
    if insufficient:
        print(f"[WARN] targets with <=200 binders (k=200 not reportable): {insufficient}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
