#!/usr/bin/env python
"""Import the NAR-GAB 2025 nine-method clustering outputs into IRBench.

Two deliverables, both anchored *only* on
``baselines/TCR_Unsupervised_Benchmark`` (i3-unit):

1. Per-method **pair-universe assignments** -- ``outputs/tcr_clustering/<method>/
   assignments.csv`` with one row for *every* paired TCR in
   ``data/tcr_clustering/tcrs.csv`` and columns ``pair_id,cdr3b,cluster,epitope``.
   ``cluster`` is the method's cluster id or ``-1`` (singleton / not clustered).
   Pairing-capable methods (clusTCR/GLIPH2/DeepTCR/TCRdist3) use their native
   paired clusters; chain-separate methods (HD/LD/TCRMatch/iSMART/GIANA) map each
   pair to the cluster of its CDR3beta (beta-chain clustering), so all nine sit on
   the same unit for an apples-to-apples comparison with our embedding models.
   Pair-clusters of size 1 collapse to ``-1``.

2. A **calibration report** -- ``outputs/tcr_clustering/calibration_report.csv``
   that recomputes each method under the *paper's native protocol* (chain methods
   on the 8,395-unique-sequence universe, pairing methods on the 4,779-pair
   universe, using the paper's fixed denominators and multi-label epitope
   purity) and compares to the paper's reported Purity/Retention. This validates
   that our metric implementation and output parsing are correct, independently
   of the small database drift that affects the shipped dataset size.

HD/LD have no precomputed file in the repo (the paper computes them on the fly in
``output_processing.R`` with stringdist); we reproduce them the same way
(connected components over CDR3 Hamming / Levenshtein distance == 1).
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(BENCH / "scripts"))

from prepare_tcr_clustering import curate, DB_FILE, REPO_COMMIT  # noqa: E402
from rapidfuzz.distance import Levenshtein  # noqa: E402

REPO = BENCH / "baselines" / "TCR_Unsupervised_Benchmark"
OM = REPO / "Data" / "Output_methods"
DATA = BENCH / "data" / "tcr_clustering"
OUT = BENCH / "outputs" / "tcr_clustering"

# Paper fixed denominators (Rmd) and reported (Retention, Purity) at size>1.
PAPER_TOTAL_PAIRS = 4779.0
PAPER_TOTAL_SEQS = 8395.0
CHAIN_METHODS = ["HD", "LD", "TCRMatch", "iSMART", "GIANA"]
PAIR_METHODS = ["clusTCR", "GLIPH2", "DeepTCR", "TCRdist3"]
# Per-method (Retention, Purity) read off the value labels printed on the Fig 3A
# bars of the anchor paper.  Earlier revisions carried (0.25, 0.66) for HD / LD /
# iSMART / GIANA: that is the *group* summary the Results text quotes for those
# four methods jointly ("retention 0.25 +/- 0.06 ... purity 0.66 +/- 0.09"), not
# their individual values.  Cross-checks on the values below: group mean is
# 0.2525 -> 0.25 and 0.665 -> 0.66; purity half-range is 0.09; every value falls
# in the qualitative bin Table 1 assigns it.
#
# Known conflict inside the paper: Supplementary Table S2's clustered-sequence
# counts divided by the denominators above give HD 2564/8395 = 0.31 and
# LD 2248/8395 = 0.27, i.e. HD and LD swapped relative to Fig 3A.  The other
# seven methods agree exactly across both sources.  We follow Fig 3A (the
# main-text result figure, and the ordering consistent with LD admitting indels
# and therefore clustering more sequences than HD).
PAPER_REPORTED = {
    "HD": (0.27, 0.69), "LD": (0.31, 0.57), "TCRMatch": (0.09, 0.82),
    "iSMART": (0.18, 0.75), "GIANA": (0.25, 0.65), "clusTCR": (0.09, 0.99),
    "GLIPH2": (0.22, 0.80), "DeepTCR": (0.92, 0.63), "TCRdist3": (0.44, 0.42),
}


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------
class UF:
    def __init__(self):
        self.p: dict = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def _components_from_edges(edges):
    uf = UF()
    nodes = set()
    for a, b in edges:
        uf.union(a, b)
        nodes.add(a)
        nodes.add(b)
    return {n: uf.find(n) for n in nodes}


def _distance_components(seqs, metric):
    """Connected components over CDR3 with Hamming/Levenshtein distance == 1.

    Substitution edges via positional masking; Levenshtein additionally links
    indel neighbours (length-differing pairs) via deletion keys.
    """
    uf = UF()
    nodes = set()
    bylen = defaultdict(list)
    for s in seqs:
        bylen[len(s)].append(s)
    for L, grp in bylen.items():
        for p in range(L):
            buckets = defaultdict(list)
            for s in grp:
                buckets[s[:p] + "*" + s[p + 1:]].append(s)
            for members in buckets.values():
                if len(members) > 1:
                    first = members[0]
                    for o in members[1:]:
                        uf.union(first, o)
                        nodes.add(first)
                        nodes.add(o)
    if metric == "lev":
        # An insertion/deletion neighbour is found by deleting one residue from
        # the *longer* observed sequence and looking up the resulting shorter
        # sequence directly.  The previous implementation grouped two original
        # strings that shared a deletion key and then required their lengths to
        # differ by one.  Members of such a bucket always have equal length, so
        # that condition was impossible and LD silently collapsed to HD.
        seq_set = set(seqs)
        for longer in seq_set:
            for i in range(len(longer)):
                shorter = longer[:i] + longer[i + 1:]
                if shorter not in seq_set:
                    continue
                if Levenshtein.distance(longer, shorter, score_cutoff=1) == 1:
                    uf.union(longer, shorter)
                    nodes.add(longer)
                    nodes.add(shorter)
    return {n: uf.find(n) for n in nodes}


# ---------------------------------------------------------------------------
# Parse each method into a canonical (node -> cluster) mapping
# ---------------------------------------------------------------------------
def parse_chain_methods(seqs_unique):
    """Return {method: {CDR3 -> cluster_id}} for chain-separate methods."""
    out = {}

    # HD / LD -- recomputed (no precomputed file exists in the repo).
    out["HD"] = _distance_components(seqs_unique, "hamming")
    out["LD"] = _distance_components(seqs_unique, "lev")

    # TCRMatch -- whitespace-delimited edge list (input, match, score); score==1
    # is the self-match diagonal and is dropped.
    tm = pd.read_csv(OM / "TCRMatch_output.txt", sep=r"\s+", header=None,
                     names=["a", "b", "score"], engine="python")
    tm = tm[tm["score"] != 1]
    out["TCRMatch"] = _components_from_edges(list(zip(tm["a"], tm["b"])))

    # iSMART -- CDR3 <tab> group (header line starts with '#').
    ism = pd.read_csv(OM / "iSMART_output.txt", sep="\t", comment="#",
                      header=None, names=["CDR3", "clusters"])
    out["iSMART"] = dict(zip(ism["CDR3"], ism["clusters"]))

    # GIANA -- CDR3 <tab> cluster <tab> TRBV (comment header lines start '#').
    gi = pd.read_csv(OM / "GIANA_output.txt", sep="\t", comment="#",
                     header=None, names=["CDR3", "clusters", "TRBV"])
    out["GIANA"] = dict(zip(gi["CDR3"], gi["clusters"]))
    return out


def parse_pair_methods(valid_pairs):
    """Return {method: [(CDR3b_CDR3a, cluster_id), ...]} membership lists.

    Membership lists (not dicts) preserve GLIPH2's overlapping motif clusters, in
    which one pair can belong to several patterns -- exactly what the paper's
    purity computation sees. ``valid_pairs`` restricts clusTCR to real pairs, as
    the paper does.
    """
    out = {}

    # clusTCR -- junction (=CDR3b+CDR3a) -> cluster; keep only real pairs.
    cl = pd.read_csv(OM / "clusTCR_output.csv")
    cl.columns = ["CDR3b_CDR3a", "clusters"]
    cl = cl[cl["CDR3b_CDR3a"].isin(valid_pairs)]
    out["clusTCR"] = list(zip(cl["CDR3b_CDR3a"], cl["clusters"]))

    # DeepTCR -- Beta/Alpha sequences + cluster.
    dt = pd.read_csv(OM / "DeepTCR_output.csv")
    dt["CDR3b_CDR3a"] = dt["Beta_Sequences"].astype(str) + dt["Alpha_Sequences"].astype(str)
    dt = dt.drop_duplicates("CDR3b_CDR3a")
    out["DeepTCR"] = list(zip(dt["CDR3b_CDR3a"], dt["cluster"]))

    # TCRdist3 -- ';'-delimited hierarchical-clustering output; drop singletons.
    td = pd.read_csv(OM / "TCRdist3_hclust_output.csv", sep=";")
    td["size"] = td.groupby("clusters")["clusters"].transform("size")
    td = td[td["size"] != 1]
    td["CDR3b_CDR3a"] = td["CDR3b"].astype(str) + td["CDR3a"].astype(str)
    td = td.drop_duplicates("CDR3b_CDR3a")
    out["TCRdist3"] = list(zip(td["CDR3b_CDR3a"], td["clusters"]))

    # GLIPH2 -- keep motif/global patterns (drop 'single'); TcRb/TcRa -> pair.
    # A pair may legitimately appear in several patterns; keep every membership.
    # The paper's cluster ``size`` is the raw row count per pattern (before
    # de-duplicating pairs), so we return that separately for a faithful purity.
    gl = pd.read_csv(OM / "GLIPH2_output_cluster.csv").iloc[:, :18].dropna()
    gl = gl[gl["pattern"] != "single"]
    gl["CDR3b_CDR3a"] = gl["TcRb"].astype(str) + gl["TcRa"].astype(str)
    gliph_size = gl.groupby("index").size().to_dict()      # raw row count / pattern
    gl = gl.drop_duplicates(["CDR3b_CDR3a", "index"])
    out["GLIPH2"] = list(zip(gl["CDR3b_CDR3a"], gl["index"]))
    return out, {"GLIPH2": gliph_size}


# ---------------------------------------------------------------------------
# Calibration (paper-native protocol, fixed denominators, multi-label purity)
# ---------------------------------------------------------------------------
def _paper_purity(memberships, node_epis: dict, th: int = 1,
                  size_override: dict | None = None) -> float:
    """Paper purity_function: clusters with size>th; sum(majority)/sum(size).

    ``memberships``: iterable of (node, cluster). A node may belong to several
    clusters (GLIPH2's overlapping motif/global patterns), and it is counted once
    per cluster it belongs to -- exactly as the paper's per-cluster ``merge``
    does. Cluster ``size`` is the number of distinct nodes in the cluster unless
    ``size_override`` supplies the paper's raw per-cluster row count (GLIPH2, where
    the denominator uses the row count while the majority uses distinct pairs).
    """
    cluster_nodes = defaultdict(set)
    for node, c in memberships:
        cluster_nodes[c].add(node)
    num = 0
    den = 0
    for c, members in cluster_nodes.items():
        size = size_override[c] if size_override is not None else len(members)
        if size <= th:
            continue
        epi_counts = Counter()
        for node in members:
            for e in node_epis.get(node, ()):  # multi-label
                epi_counts[e] += 1
        maj = max(epi_counts.values()) if epi_counts else 0
        num += maj
        den += size
    return num / den if den else float("nan")


def calibrate(chain_maps, pair_maps, pair_sizes, chain_epi_multi, pair_epi_multi):
    rows = []
    for m in CHAIN_METHODS:
        nc = chain_maps[m]                       # node -> cluster (single membership)
        ret = len(set(nc.keys())) / PAPER_TOTAL_SEQS
        pur = _paper_purity(nc.items(), chain_epi_multi, th=1)
        rows.append((m, "chain", ret, pur))
    for m in PAIR_METHODS:
        memb = pair_maps[m]                      # list of (pair, cluster) memberships
        ret = len({p for p, _ in memb}) / PAPER_TOTAL_PAIRS
        pur = _paper_purity(memb, pair_epi_multi, th=1, size_override=pair_sizes.get(m))
        rows.append((m, "pair", ret, pur))

    recs = []
    for m, kind, ret, pur in rows:
        pr, pp = PAPER_REPORTED[m]
        recs.append({
            "method": m, "unit": kind,
            "recomputed_retention": round(ret, 3),
            "recomputed_purity": round(pur, 3),
            "paper_retention": pr, "paper_purity": pp,
            "d_retention": round(ret - pr, 3),
            "d_purity": round(pur - pp, 3),
            "match": bool(abs(ret - pr) <= 0.03 and abs(pur - pp) <= 0.03),
        })
    order = {n: i for i, n in enumerate(
        ["HD", "LD", "TCRMatch", "iSMART", "GIANA", "clusTCR", "GLIPH2", "DeepTCR", "TCRdist3"])}
    return pd.DataFrame(recs).sort_values("method", key=lambda s: s.map(order))


# ---------------------------------------------------------------------------
# Pair-universe assignments (unified comparison unit)
# ---------------------------------------------------------------------------
def _relabel_drop_singletons(pair_to_raw: dict, all_pairs) -> dict:
    """raw cluster ids -> contiguous ints; pair-clusters of size 1 -> -1."""
    sizes = Counter(pair_to_raw.values())
    remap = {}
    nxt = 0
    out = {}
    for pid in all_pairs:
        raw = pair_to_raw.get(pid)
        if raw is None or sizes[raw] < 2:
            out[pid] = -1
            continue
        if raw not in remap:
            remap[raw] = nxt
            nxt += 1
        out[pid] = remap[raw]
    return out


def build_assignments(pairs_df, chain_maps, pair_maps):
    """Return {method: DataFrame[pair_id,cdr3b,cluster,epitope]} on the pair universe."""
    all_pairs = pairs_df["pair_id"].tolist()
    cdr3b_of = dict(zip(pairs_df["pair_id"], pairs_df["cdr3b"]))
    epi_of = dict(zip(pairs_df["pair_id"], pairs_df["peptide"]))
    junction_of = dict(zip(pairs_df["pair_id"],
                           pairs_df["cdr3b"].astype(str) + pairs_df["cdr3a"].astype(str)))

    assignments = {}

    # Pairing methods: pair -> its native cluster (via CDR3b+CDR3a key). GLIPH2
    # motifs overlap; for the partition-style pair universe we take each pair's
    # first cluster deterministically.
    for m in PAIR_METHODS:
        nc = {}
        for pair, c in pair_maps[m]:
            nc.setdefault(pair, c)
        raw = {pid: nc[junction_of[pid]] for pid in all_pairs if junction_of[pid] in nc}
        labels = _relabel_drop_singletons(raw, all_pairs)
        assignments[m] = labels

    # Chain methods: pair -> cluster of its CDR3beta (beta-chain clustering).
    for m in CHAIN_METHODS:
        nc = chain_maps[m]
        raw = {}
        for pid in all_pairs:
            c = nc.get(cdr3b_of[pid])
            if c is not None:
                raw[pid] = f"{m}:{c}"
        labels = _relabel_drop_singletons(raw, all_pairs)
        assignments[m] = labels

    frames = {}
    for m, labels in assignments.items():
        frames[m] = pd.DataFrame({
            "pair_id": all_pairs,
            "cdr3b": [cdr3b_of[p] for p in all_pairs],
            "cluster": [labels[p] for p in all_pairs],
            "epitope": [epi_of[p] for p in all_pairs],
        })
    return frames


def main():
    if not DB_FILE.exists():
        raise SystemExit(f"pooled DB not found: {DB_FILE}")
    pairs_df = pd.read_csv(DATA / "tcrs.csv").fillna("")

    # data_human reproduction for calibration multi-label maps + valid pairs.
    d = curate(DB_FILE)
    d["CDR3b_CDR3a"] = d["CDR3_beta"].astype(str) + d["CDR3_alpha"].astype(str)
    valid_pairs = set(d["CDR3b_CDR3a"])
    seqs_unique = sorted(set(d["CDR3_alpha"].astype(str)) | set(d["CDR3_beta"].astype(str)))

    # multi-label maps: node -> set(epitopes)
    pair_epi_multi = d.groupby("CDR3b_CDR3a")["Epitope"].agg(set).to_dict()
    chain_epi_multi = defaultdict(set)
    for cdr, epi in zip(d["CDR3_alpha"], d["Epitope"]):
        chain_epi_multi[cdr].add(epi)
    for cdr, epi in zip(d["CDR3_beta"], d["Epitope"]):
        chain_epi_multi[cdr].add(epi)

    print("parsing method outputs ...")
    chain_maps = parse_chain_methods(seqs_unique)
    pair_maps, pair_sizes = parse_pair_methods(valid_pairs)

    # 1) calibration report
    calib = calibrate(chain_maps, pair_maps, pair_sizes, chain_epi_multi, pair_epi_multi)
    OUT.mkdir(parents=True, exist_ok=True)
    calib.to_csv(OUT / "calibration_report.csv", index=False)
    n_match = int(calib["match"].sum())
    print("\n=== calibration vs paper (native protocol, fixed denominators) ===")
    print(calib.to_string(index=False))
    print(f"matched (|dRet|<=.03 & |dPur|<=.03): {n_match}/9")
    with (OUT / "calibration_report.json").open("w") as fh:
        json.dump({"n_matched": n_match, "rows": calib.to_dict(orient="records"),
                   "paper_total_pairs": PAPER_TOTAL_PAIRS,
                   "paper_total_seqs": PAPER_TOTAL_SEQS,
                   "repo_commit": REPO_COMMIT,
                   "ld_indel_implementation": "delete-longer-and-lookup-shorter (bug-fixed)",
                   "baseline_provenance": {
                       "evidence_type": "official_artifact_rescored",
                       "protocol_alignment": "mismatched",
                       "paper_comparable": False,
                       "citation": "doi:10.1093/nargab/lqaf150",
                       "source_location": "authors' precomputed assignments + drifted pooled DB",
                       "notes": "paper-reported rows remain primary until the paper-era DB is recovered",
                   }}, fh, indent=2)

    # 2) pair-universe assignments
    print("\nwriting pair-universe assignments ...")
    frames = build_assignments(pairs_df, chain_maps, pair_maps)
    summary = []
    for m, df in frames.items():
        mdir = OUT / m
        mdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(mdir / "assignments.csv", index=False)
        n_clustered = int((df["cluster"] != -1).sum())
        n_clusters = int(df.loc[df["cluster"] != -1, "cluster"].nunique())
        summary.append((m, n_clustered, len(df), n_clusters))
        print(f"  {m:10s} clustered {n_clustered:5d}/{len(df)} pairs "
              f"in {n_clusters:4d} clusters -> {mdir}/assignments.csv")

    with (OUT / "import_summary.json").open("w") as fh:
        json.dump({
            "unit": "paired TCR (pair_id); chain methods beta-mapped",
            "n_pairs": int(len(pairs_df)),
            "methods": {m: {"n_clustered": nc, "n_total": nt, "n_clusters": ncl}
                        for m, nc, nt, ncl in summary},
            "repo_commit": REPO_COMMIT,
        }, fh, indent=2)


if __name__ == "__main__":
    main()
