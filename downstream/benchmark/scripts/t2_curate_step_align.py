#!/usr/bin/env python
"""T2 -- step-by-step alignment of ``curate()`` against ``database_processing.R``.

The question this settles
-------------------------
Our pipeline reproduces 5,368 unique paired TCRs where the paper reports 4,779
(+589). RESULTS.md previously floated "database drift" as the cause, but the
audit rejected that: the per-epitope counts of all six highlighted epitopes match
the paper exactly, which a drifted database would not produce. The remaining
hypothesis was a filtering-implementation difference between R and pandas.

First, an important fact about where 4,779 comes from. It is **not** computed from
the database inside the analysis; it is a hardcoded constant in
``Analysis/benchmark_tools_pooled_database_CD8.Rmd`` lines 223-226
(``/4779`` with the comment "divided by the number of unique pairs"), alongside
``/8395`` on lines 217-221. So the number was produced by some earlier run and
pasted in, and cannot be re-derived from the shipped file by definition.

What this script does
---------------------
Runs the R filter chain twice over the shipped database and prints the surviving
row count and unique-pair count after every step:

* ``pandas`` -- exactly what ``curate()`` does today;
* ``r_faithful`` -- the same chain with R/dplyr semantics restored:

  1. ``fread`` leaves empty character fields as ``""`` and only treats the literal
     ``NA`` as missing, whereas ``pd.read_csv`` turns ``""``, ``N/A``, ``null`` and
     friends into ``NaN``. Emulated with ``keep_default_na=False`` plus an explicit
     ``NA``-only missing set.
  2. ``dplyr::filter`` drops rows whose predicate evaluates to ``NA``. So
     ``filter(PubMed_ID != <10x url>)`` drops rows with a missing ``PubMed_ID``,
     while pandas' ``!=`` returns ``True`` for ``NaN`` and keeps them.
  3. ``drop_na(V_alpha)`` removes ``NA`` only and keeps ``""``.
  4. ``subset(nchar(...) >= 6 & <= 23)`` drops rows whose condition is ``NA``.

If ``r_faithful`` also lands on 5,368 then the gap is not an implementation
difference and the +589 must live in the constant or the database version. If it
lands on 4,779 then ``curate()`` has a bug.

Output:
    outputs/tcr_clustering/curate_step_align.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH / "scripts"))

DB = (BENCH / "baselines" / "TCR_Unsupervised_Benchmark" / "Data" / "Database"
      / "database_pooled_human_2023_03_15.txt")
OUT = BENCH / "outputs" / "tcr_clustering" / "curate_step_align.json"

TENX = ("https://pages.10xgenomics.com/rs/446-PBO-704/images/"
        "10x_AN047_IP_A_New_Way_of_Exploring_Immunity_Digital.pdf")
DROP_PAIRS = {"CASSDSRGTEAFF_CASSDSRGTEAFF", "CASSPITGTGAYGYTF_CASSSVNEQYF"}
KEY = ["CDR3_beta", "CDR3_alpha"]


def _n(d: pd.DataFrame) -> tuple[int, int]:
    if not len(d):
        return 0, 0
    return len(d), int(d[KEY].drop_duplicates().shape[0])


def run(mode: str) -> tuple[list[dict], pd.DataFrame]:
    """Execute the R chain under either ``pandas`` or ``r_faithful`` semantics."""
    r = mode == "r_faithful"
    if r:
        # fread: only the literal "NA" is missing; "" stays an empty string.
        db = pd.read_csv(DB, sep="\t", low_memory=False, encoding="latin-1",
                         keep_default_na=False, na_values=["NA"])
        for c in ("Verified_score", "Identification_score"):
            db[c] = pd.to_numeric(db[c], errors="coerce")
    else:
        db = pd.read_csv(DB, sep="\t", low_memory=False, encoding="latin-1")

    steps = [{"step": "0_raw", "rows": len(db),
              "pairs": int(db[KEY].astype(str).drop_duplicates().shape[0])}]

    def rec(name, d):
        rows, pairs = _n(d)
        steps.append({"step": name, "rows": rows, "pairs": pairs})
        return d

    epi_missing = db["Epitope"].isna() | (db["Epitope"] == "") if r else db["Epitope"].isna()
    d = rec("1_score_and_epitope",
            db[(db["Verified_score"] == 2) & (db["Identification_score"] > 4.3)
               & ~epi_missing].copy())

    if r:
        # dplyr::filter drops NA predicates, so a missing PubMed_ID is removed.
        keep = (d["PubMed_ID"] != TENX) & d["PubMed_ID"].notna()
    else:
        keep = d["PubMed_ID"] != TENX
    d = rec("2_drop_10x_publication", d[keep])

    d = rec("3_cd8", d[d["Cell_subset"] == "CD8"])

    for c in ["V_alpha", "J_alpha", "V_beta", "J_beta"]:
        if r:
            d = d[d[c].notna()]          # drop_na keeps ""
        else:
            d = d[d[c].notna()]          # but "" already became NaN on read
    d = rec("4_drop_na_vj", d)

    if r:
        la = d["CDR3_alpha"].where(d["CDR3_alpha"].notna()).str.len()
        lb = d["CDR3_beta"].where(d["CDR3_beta"].notna()).str.len()
        cond = (la.between(6, 23) & lb.between(6, 23)).fillna(False)
    else:
        la = d["CDR3_alpha"].astype(str).str.len()
        lb = d["CDR3_beta"].astype(str).str.len()
        cond = la.between(6, 23) & lb.between(6, 23)
    d = rec("5_cdr3_length", d[cond])

    d = d.copy()
    d["pair"] = d["CDR3_beta"].astype(str) + "_" + d["CDR3_alpha"].astype(str)
    npe = d.groupby("Epitope")["pair"].transform("nunique")
    d = rec("6_epitope_has_2plus_pairs", d[npe != 1])

    d = rec("7_drop_two_doublons", d[~d["pair"].isin(DROP_PAIRS)])
    d = rec("8_unique_rows", d.drop_duplicates())
    return steps, d


def main() -> None:
    res = {"database": str(DB),
           "paper_constants": {"pairs": 4779, "unique_sequences": 8395,
                               "source": "Analysis/benchmark_tools_pooled_database_"
                                         "CD8.Rmd lines 217-226, hardcoded "
                                         "denominators (not computed from the "
                                         "shipped database)"},
           "modes": {}}

    frames = {}
    for mode in ("pandas", "r_faithful"):
        steps, d = run(mode)
        seqs = len(set(d["CDR3_alpha"].astype(str)) | set(d["CDR3_beta"].astype(str)))
        res["modes"][mode] = {"steps": steps, "final_pairs": steps[-1]["pairs"],
                              "final_unique_sequences": seqs,
                              "final_epitopes": int(d["Epitope"].nunique())}
        frames[mode] = d
        print(f"\n=== {mode} ===")
        for s in steps:
            print(f"  {s['step']:28s} rows {s['rows']:8,}  unique pairs {s['pairs']:6,}")
        print(f"  -> pairs {steps[-1]['pairs']:,} / unique seqs {seqs:,} / "
              f"epitopes {d['Epitope'].nunique()}")

    a, b = frames["pandas"], frames["r_faithful"]
    pa = set(map(tuple, a[KEY].astype(str).drop_duplicates().to_numpy()))
    pb = set(map(tuple, b[KEY].astype(str).drop_duplicates().to_numpy()))
    res["pandas_vs_r_faithful"] = {
        "pairs_pandas_only": len(pa - pb),
        "pairs_r_only": len(pb - pa),
        "pairs_shared": len(pa & pb),
    }
    print(f"\npandas-only pairs {len(pa - pb)}, r-only pairs {len(pb - pa)}, "
          f"shared {len(pa & pb)}")

    for mode in ("pandas", "r_faithful"):
        gap = res["modes"][mode]["final_pairs"] - 4779
        print(f"{mode:12s}: {res['modes'][mode]['final_pairs']:,} pairs, "
              f"gap vs paper constant {gap:+,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
