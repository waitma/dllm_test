#!/usr/bin/env python
"""Build the T2 SECOND-BASIS clustering dataset from Brief Bioinformatics 2025.

Second construction basis for T2 (complementary to NAR-GAB 2025):
*A comprehensive benchmarking for evaluating TCR embeddings in modeling
TCR-epitope interactions* (Feng, Huo et al., Brief Bioinform 2025, bbaf030,
PMID 39883514). Official repo ``deepomicslab/TCREmbedding`` (commit ``d5bf911``).

Unlike NAR-GAB 2025 (which supplies nine OFFICIAL clustering-method outputs on a
paired pooled DB with a Purity-Retention protocol), this benchmark standardises
the **embedding -> clustering** evaluation: a fixed labelled dataset of unique
CDR3beta sequences (curated from the GIANA project) is embedded by each method,
clustered with standard sklearn algorithms over a K sweep, and scored with
**ARI / NMI / Purity**. This is exactly the same-caliber protocol our BioSeq
embedding models plug into, and it adds the two things NAR-GAB lacks: full
ARI/NMI cluster-agreement metrics and an independent second labelled dataset.

We only reproduce the fixed evaluation dataset here (``TCRantigenData_unique_test.csv``
downloaded from the official repo). The dataset is CDR3beta-only (single chain);
epitope labels are held out for evaluation only and never seen by clustering.

Output:
- ``data/tcr_clustering_embed/tcrs.csv`` -- one row per unique CDR3beta with
  columns ``seq_id, cdr3b, vgene, antigen, epitope``.
- ``data/tcr_clustering_embed/meta.json`` -- sizes, per-epitope counts, source
  repo/commit, label-use note.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
SRC = (BENCH / "baselines" / "TCREmbedding" / "dataset" / "clustering"
       / "TCRantigenData_unique_test.csv")
OUT_DIR = BENCH / "data" / "tcr_clustering_embed"
REPO_COMMIT = "d5bf911f5baa6e8d45251eae1f217ad6af0b65f7"


def main():
    if not SRC.exists():
        raise SystemExit(
            f"second-basis dataset not found: {SRC}\n"
            "Download dataset/clustering/TCRantigenData_unique_test.csv from "
            "github.com/deepomicslab/TCREmbedding (gitignored) first.")

    df = pd.read_csv(SRC)
    # Columns: CDR3b, Vgene, Antigen-epitope, Antigen, Epitope_type_num, TCR_num
    df = df.rename(columns={"CDR3b": "cdr3b", "Vgene": "vgene",
                            "Antigen": "antigen"})
    # Antigen-epitope column format: "Antigen:<name>_<EPITOPE>"; epitope is the
    # trailing peptide after the last underscore.
    df["epitope"] = df["Antigen-epitope"].astype(str).str.rsplit("_", n=1).str[-1]
    df = df.drop_duplicates("cdr3b").reset_index(drop=True)
    df["seq_id"] = [f"seq_{i:05d}" for i in range(len(df))]

    cols = ["seq_id", "cdr3b", "vgene", "antigen", "epitope"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT_DIR / "tcrs.csv", index=False)

    epi_counts = Counter(df["epitope"])
    meta = {
        "task": "T2 TCR clustering -- SECOND BASIS (embedding -> clustering)",
        "source": {
            "paper": ("A comprehensive benchmarking for evaluating TCR "
                      "embeddings in modeling TCR-epitope interactions "
                      "(Brief Bioinformatics 2025, bbaf030, PMID 39883514)"),
            "repo": "github.com/deepomicslab/TCREmbedding",
            "commit": REPO_COMMIT,
            "dataset_file": "dataset/clustering/TCRantigenData_unique_test.csv",
            "raw_origin": "GIANA project antigen-labelled CDR3beta (curated by the authors)",
        },
        "evaluation_unit": "unique CDR3beta (single chain); seq_col=cdr3b",
        "protocol": ("embedding -> standardise -> K-means (+ hierarchical) over "
                     "K in {10,15,...,100} -> ARI / NMI / Purity, mean over K "
                     "(paper's continuous-K strategy to remove target-K bias)"),
        "label_use": "epitope labels held out for evaluation only; clustering never sees them",
        "n_sequences": int(len(df)),
        "n_epitopes": int(df["epitope"].nunique()),
        "n_antigens": int(df["antigen"].nunique()),
        "min_per_epitope": int(min(epi_counts.values())),
        "per_epitope_counts": dict(sorted(epi_counts.items(),
                                          key=lambda kv: -kv[1])),
        "note": ("CDR3beta-only single-chain clustering caliber; NOT mixed with "
                 "the paired NAR-GAB 2025 basis (5,368 paired TCRs). Reported "
                 "separately as the second construction basis."),
    }
    with (OUT_DIR / "meta.json").open("w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[T2 basis-B] wrote {len(df)} unique CDR3beta over "
          f"{df['epitope'].nunique()} epitopes -> {OUT_DIR}/tcrs.csv")
    print(f"     antigens={df['antigen'].nunique()} "
          f"min/epitope={min(epi_counts.values())}")
    print(f"     meta -> {OUT_DIR}/meta.json")


if __name__ == "__main__":
    main()
