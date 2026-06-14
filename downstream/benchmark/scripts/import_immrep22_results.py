#!/usr/bin/env python
"""Import IMMREP22 official method results into a single reproducible CSV.

Reads ``baselines/IMMREP_2022_TCRSpecificity/methods_results/*_stats.txt`` (the
authors' published per-epitope MicroAUC + Average Rank) and aggregates the
``_Average`` rows. This is an *external reference* table (different dataset than
IRBench-T1); we only re-tabulate official numbers, never recompute them.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
SRC = BENCH / "baselines" / "IMMREP_2022_TCRSpecificity" / "methods_results"
OUT = BENCH / "outputs" / "external" / "immrep22_official.csv"


def main():
    rows = []
    for f in sorted(glob.glob(str(SRC / "*_stats.txt"))):
        name = os.path.basename(f).replace("_stats.txt", "")
        try:
            df = pd.read_csv(f, sep="\t", index_col=0)
        except Exception as e:  # pragma: no cover
            print("skip", name, e)
            continue
        auc = df.loc["_Average", "MicroAUC"] if (
            "_Average" in df.index and "MicroAUC" in df.columns) else None
        rank = df.loc["_Average", "Average Rank"] if (
            "_Average" in df.index and "Average Rank" in df.columns) else None
        rows.append({"method": name, "avg_MicroAUC": auc, "avg_Rank": rank})

    out = pd.DataFrame(rows).sort_values(
        "avg_MicroAUC", ascending=False, na_position="last")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} methods -> {OUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
