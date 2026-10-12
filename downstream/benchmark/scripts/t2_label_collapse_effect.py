#!/usr/bin/env python
"""T2 -- how much does the shipped dataset's majority-label collapse move purity?

`data/tcr_clustering/tcrs.csv` stores one row per unique (CDR3b, CDR3a) pair with a
single ``peptide``, chosen by ``prepare_tcr_clustering.py::build_pairs`` as the
majority epitope with a lexicographic tie-break. The anchor paper's own
``purity_function`` instead keeps every epitope a pair was observed against: its
``merge`` expands a polyspecific CDR3 into one row per epitope, so the numerator
counts multi-label occurrences while the denominator stays the deduplicated
cluster size (which is why paper purity can in principle exceed 1).

The paper-native calibration path in ``import_clustering_baselines.py`` already
uses the multi-label convention, so section T2(a) is like-for-like. The collapse
only affects the local hybrid pair-universe tables, T2(b) and T2(c). This script
measures that effect instead of leaving it as a qualitative caveat: it recomputes
purity for every stored clustering under both conventions on the same clusters.

Outputs:
    outputs/tcr_clustering/label_collapse_effect.csv
    outputs/tcr_clustering/label_collapse_effect.json
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

DATA = BENCH / "data" / "tcr_clustering"
OUT = BENCH / "outputs" / "tcr_clustering"
DB = (BENCH / "baselines" / "TCR_Unsupervised_Benchmark" / "Data" / "Database"
      / "database_pooled_human_2023_03_15.txt")


NOISE_CLUSTER = -1   # run.py marks unclustered pairs with -1, not a real cluster


def purity(assign: pd.DataFrame, node_epis: dict, th: int = 1) -> tuple[float, int, int]:
    """Paper ``purity_function``: over clusters with size > ``th``,
    ``sum(majority epitope count) / sum(cluster size)``."""
    cluster_nodes = defaultdict(set)
    for node, c in zip(assign["pair_id"], assign["cluster"]):
        if c == NOISE_CLUSTER:
            continue
        cluster_nodes[c].add(node)
    num = den = n_clusters = 0
    for members in cluster_nodes.values():
        if len(members) <= th:
            continue
        n_clusters += 1
        counts = Counter()
        for node in members:
            for e in node_epis.get(node, ()):
                counts[e] += 1
        num += max(counts.values()) if counts else 0
        den += len(members)
    return (num / den if den else float("nan")), n_clusters, den


def main() -> None:
    pairs = pd.read_csv(DATA / "tcrs.csv")
    collapsed = {p: (e,) for p, e in zip(pairs["pair_id"], pairs["peptide"])}

    # Rebuild the multi-label map: every epitope each pair was observed against.
    multi: dict[str, set[str]] = defaultdict(set)
    if DB.exists():
        sys.path.insert(0, str(BENCH / "scripts"))
        from prepare_tcr_clustering import _pair_id, curate  # noqa: E402
        cur = curate(DB).rename(columns={"CDR3_alpha": "cdr3a", "CDR3_beta": "cdr3b",
                                         "Epitope": "peptide"})
        for a, b, e in zip(cur["cdr3a"], cur["cdr3b"], cur["peptide"]):
            multi[_pair_id(b, a)].add(str(e))
        multi = {k: v for k, v in multi.items() if k in collapsed}
        src = str(DB)
    else:
        print(f"[warn] {DB} missing; multi-label map unavailable")
        return

    n_poly = sum(1 for v in multi.values() if len(v) > 1)
    n_tie = 0
    for p, epis in multi.items():
        if len(epis) > 1 and collapsed[p][0] not in epis:
            n_tie += 1
    print(f"pairs {len(collapsed)}, multi-label map covers {len(multi)}, "
          f"polyspecific {n_poly} ({n_poly/len(collapsed)*100:.2f}%), "
          f"collapsed label outside its own epitope set {n_tie}")

    rows = []
    for adir in sorted(OUT.iterdir()):
        f = adir / "assignments.csv"
        if not f.exists():
            continue
        a = pd.read_csv(f)
        if not {"pair_id", "cluster"} <= set(a.columns):
            continue
        p_col, nc, den = purity(a, collapsed)
        p_mul, _, _ = purity(a, multi)
        rows.append({"method": adir.name, "n_clusters_gt1": nc, "n_nodes_scored": den,
                     "purity_collapsed": p_col, "purity_multilabel": p_mul,
                     "delta": p_mul - p_col})

    df = pd.DataFrame(rows).sort_values("method")
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "label_collapse_effect.csv", index=False)
    summary = {
        "database": src,
        "n_pairs": len(collapsed),
        "n_polyspecific_pairs": n_poly,
        "polyspecific_frac": round(n_poly / len(collapsed), 4),
        "tie_break": "majority epitope; ties broken lexicographically "
                     "(build_pairs); no collapsed label falls outside its own "
                     "observed epitope set" if n_tie == 0 else
                     f"{n_tie} collapsed labels fall outside their epitope set",
        "max_abs_delta": float(df["delta"].abs().max()) if len(df) else None,
        "mean_delta": float(df["delta"].mean()) if len(df) else None,
    }
    (OUT / "label_collapse_effect.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200)
    print(df.round(4).to_string(index=False))
    print(f"\nmax |delta| {summary['max_abs_delta']:.4f}, "
          f"mean delta {summary['mean_delta']:+.4f}")
    print(f"-> {OUT/'label_collapse_effect.csv'}")


if __name__ == "__main__":
    main()
