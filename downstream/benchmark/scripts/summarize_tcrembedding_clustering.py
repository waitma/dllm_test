#!/usr/bin/env python
"""Consolidate T2 second-basis (TCREmbedding) embedding->clustering results.

Reads every ``outputs/tcr_clustering_embed/<tag>/metrics.json`` and writes a
single leaderboard CSV (ARI / NMI / Purity, mean & best over the K sweep) split
into our-model rows vs reference-embedding rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
OUT = BENCH / "outputs" / "tcr_clustering_embed"

# Which tags are "ours" vs generic reference embeddings.
OURS_PREFIX = ("ours", "bioseq")


def _kind(tag: str) -> str:
    t = tag.lower()
    if t.startswith(OURS_PREFIX):
        return "ours"
    return "reference"


def main():
    rows = []
    for mj in sorted(OUT.glob("*/metrics.json")):
        d = json.loads(mj.read_text())
        tag = d.get("tag", mj.parent.name)
        algos = d.get("algos", ["kmeans"])
        row = {"kind": _kind(tag), "tag": tag, "model": d.get("model", ""),
               "n_units": d.get("n_units"), "n_epitopes": d.get("n_epitopes"),
               "embed_sec": d.get("embed_sec")}
        for algo in algos:
            for met in ("ari", "nmi", "purity"):
                row[f"{algo}_{met}_mean"] = d.get(f"{algo}_{met}_mean")
                row[f"{algo}_{met}_best"] = d.get(f"{algo}_{met}_best")
        rows.append(row)

    if not rows:
        print(f"no metrics.json under {OUT}")
        return

    df = pd.DataFrame(rows)
    sort_key = "kmeans_ari_mean" if "kmeans_ari_mean" in df else df.columns[-1]
    df = df.sort_values(["kind", sort_key], ascending=[True, False])
    out_csv = OUT / "_summary.csv"
    df.to_csv(out_csv, index=False)

    cols = [c for c in ("kind", "tag", "kmeans_ari_mean", "kmeans_nmi_mean",
                        "kmeans_purity_mean") if c in df.columns]
    print(f"=== T2 basis-B (TCREmbedding) embedding->clustering, mean over K ===")
    print(df[cols].to_string(index=False))
    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
