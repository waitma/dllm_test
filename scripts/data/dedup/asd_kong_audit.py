#!/usr/bin/env python3
"""Audit ASD-antibody decontamination against the Kong CDR-infilling benchmark.

Answers three questions:
  1. Do the two CDR-H3 extractors agree? (SAbDab IMGT ``cdrh3_seq`` loop vs the
     regex junction core ``ab_h3_core`` used on ASD heavy chains.)
  2. How much of the Kong benchmark is covered by the *old* SAbDab bank that
     ``ab_cdrh3.txt`` was actually built from?
  3. Exact-match contamination of ASD-antibody train against each snapshot.

Cluster-level (0.80) numbers are produced by ``--cluster``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/data/tcr_native"))
sys.path.insert(0, str(PROJECT_ROOT))

from decontam import (  # noqa: E402
    BANK_DIR,
    DATA,
    MMSEQS_BIN,
    canon_core,
    is_cdr3,
    is_valid_protein_sequence,
    normalize_sequence,
)
from decontam_extra import ab_h3_core  # noqa: E402

from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    ClusterRule,
    cluster_with_mmseqs,
)

csv.field_size_limit(2**31 - 1)

ASD_AB_TRAIN = PROJECT_ROOT / "downstream/asd/step6_final/antibody/train.csv"
OLD_SABDAB = DATA / "downstream/cdr_infilling/sabdab"
KONG_SABDAB = DATA / "downstream/cdr_infilling/sabdab_kong"


def load_snapshot(base: Path, loop: str = "cdrh3") -> dict[str, dict]:
    """Union of all 10 test folds, keyed by pdb(+chain). The two snapshots use
    different field names: Kong has ``pdb``/``heavy_chain``, the older dump has
    ``pdb_id`` only."""
    rows: dict[str, dict] = {}
    for fold in range(10):
        p = base / loop / f"fold_{fold}" / "test.json"
        if not p.is_file():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pdb = d.get("pdb") or d.get("pdb_id") or ""
            chain = d.get("heavy_chain") or ""
            rows[f"{pdb}_{chain}"] = d
    return rows


def extractor_agreement(rows: dict[str, dict], label: str) -> dict:
    """Does ab_h3_core(heavy) reproduce the annotated cdrh3_seq loop?"""
    n = agree = agree_after_canon = 0
    span_ok = 0
    examples: list[dict] = []
    for d in rows.values():
        heavy = normalize_sequence(d.get("heavy_chain_seq"))
        loop = normalize_sequence(d.get("cdrh3_seq"))
        if not heavy or not loop:
            continue
        n += 1
        # is cdrh3_pos a plain 0-indexed slice into heavy_chain_seq?
        try:
            lo, hi = json.loads(d.get("cdrh3_pos", "[]"))
            if heavy[lo : hi + 1] == loop:
                span_ok += 1
        except Exception:
            pass
        regex_core = ab_h3_core(heavy)
        if regex_core == loop:
            agree += 1
        if regex_core == canon_core(loop, has_anchors=True):
            agree_after_canon += 1
        elif len(examples) < 5:
            examples.append({"pdb": d.get("pdb"), "annot_loop": loop, "regex_core": regex_core})
    return {
        "snapshot": label,
        "n": n,
        "cdrh3_pos_is_0indexed_slice": span_ok,
        "regex_core == annotated_loop": agree,
        "regex_core == canon_core(loop)": agree_after_canon,
        "agreement_rate": round(agree / n, 4) if n else 0.0,
        "mismatch_examples": examples,
    }


def bank_tokens() -> tuple[set[str], set[str], int]:
    """ab_cdrh3.txt both raw and as ``decontam_extra.load_ab_banks`` consumes it.

    The bank stores IMGT loops that are ALREADY anchor-free, so applying
    ``canon_core(has_anchors=True)`` over-strips any loop that happens to start
    with C or end in F/W. ``n_altered`` quantifies that.
    """
    raw: set[str] = set()
    canon: set[str] = set()
    p = BANK_DIR / "ab_cdrh3.txt"
    n_altered = 0
    for tok in p.read_text().split():
        s = normalize_sequence(tok)
        if s:
            raw.add(s)
        c = canon_core(tok, has_anchors=True)
        if c:
            canon.add(c)
        if s and c and s != c:
            n_altered += 1
    return raw, canon, n_altered


def snapshot_cores(rows: dict[str, dict]) -> set[str]:
    """Benchmark-side CDR-H3 keys, matched to the ASD-side definition."""
    out: set[str] = set()
    for d in rows.values():
        loop = normalize_sequence(d.get("cdrh3_seq"))
        if loop and is_cdr3(loop, max_len=45):
            out.add(loop)
    return out


def asd_cores() -> tuple[set[str], list[str], list[str], int]:
    cores: set[str] = set()
    per_row: list[str] = []
    per_row_ds: list[str] = []
    n_rows = 0
    with ASD_AB_TRAIN.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            h3 = ab_h3_core(normalize_sequence(row.get("heavy_fv")))
            per_row.append(h3)
            per_row_ds.append(row.get("dataset", ""))
            if h3 and is_cdr3(h3, max_len=45):
                cores.add(h3)
    return cores, per_row, per_row_ds, n_rows


def cluster_hits(query: set[str], bank: set[str], tag: str, threads: int,
                 min_id: float = 0.80) -> set[str]:
    if not query or not bank:
        return set()
    out_dir = DATA / "dedup/clusters_asd_kong" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res = cluster_with_mmseqs(
        rule=ClusterRule("ab_h3", min_id, 0.80),
        query_sequences=query,
        benchmark_sequences=bank,
        output_dir=out_dir,
        mmseqs_bin=Path(MMSEQS_BIN),
        threads=threads,
    )
    return {q for q in query if res.sequence_to_cluster.get(q, "") in res.benchmark_clusters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cluster", action="store_true", help="run mmseqs 0.80 buckets")
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    old = load_snapshot(OLD_SABDAB)
    kong = load_snapshot(KONG_SABDAB)

    report: dict = {}
    report["extractor_agreement"] = [
        extractor_agreement(old, "sabdab_old"),
        extractor_agreement(kong, "sabdab_kong"),
    ]

    old_cores = snapshot_cores(old)
    kong_cores = snapshot_cores(kong)
    bank_raw, bank, bank_altered = bank_tokens()

    old_pdbs = {k.split("_")[0] for k in old}
    kong_pdbs = {k.split("_")[0] for k in kong}

    report["snapshots"] = {
        "old_records": len(old),
        "kong_records": len(kong),
        "old_pdbs": len(old_pdbs),
        "kong_pdbs": len(kong_pdbs),
        "pdb_intersection": len(old_pdbs & kong_pdbs),
        "kong_only_pdbs": len(kong_pdbs - old_pdbs),
        "old_h3_cores": len(old_cores),
        "kong_h3_cores": len(kong_cores),
        "h3_core_intersection": len(old_cores & kong_cores),
        "kong_only_h3_cores": len(kong_cores - old_cores),
    }

    report["bank_coverage"] = {
        "bank_tokens_raw": len(bank_raw),
        "bank_tokens_after_canon_core": len(bank),
        "bank_tokens_altered_by_canon_core": bank_altered,
        "old_cores_in_bank_raw": len(old_cores & bank_raw),
        "old_cores_missing_from_bank_raw": len(old_cores - bank_raw),
        "kong_cores_in_bank_raw": len(kong_cores & bank_raw),
        "kong_cores_missing_from_bank_raw": len(kong_cores - bank_raw),
        "kong_coverage_rate_raw": round(len(kong_cores & bank_raw) / len(kong_cores), 4) if kong_cores else 0.0,
    }

    cores, per_row, per_row_ds, n_rows = asd_cores()
    report["asd"] = {
        "n_rows": n_rows,
        "rows_with_h3": sum(1 for h in per_row if h),
        "unique_h3_cores": len(cores),
        "exact_hits_vs_kong": len(cores & kong_cores),
        "exact_hits_vs_old": len(cores & old_cores),
        "exact_hits_vs_bank": len(cores & bank_raw),
    }

    def rows_for(hit_cores: set[str]) -> int:
        return sum(1 for h in per_row if h in hit_cores)

    report["asd"]["rows_exact_vs_kong"] = rows_for(cores & kong_cores)
    report["asd"]["rows_exact_vs_old"] = rows_for(cores & old_cores)

    if args.cluster:
        c_kong = cluster_hits(cores, kong_cores, "asd_vs_kong", args.threads)
        c_old = cluster_hits(cores, old_cores, "asd_vs_old", args.threads)
        c_bank = cluster_hits(cores, bank_raw, "asd_vs_bank", args.threads)
        from collections import Counter

        ds_hit: Counter = Counter()
        ds_tot: Counter = Counter()
        for h, ds in zip(per_row, per_row_ds):
            ds_tot[ds] += 1
            if h in c_kong:
                ds_hit[ds] += 1

        report["asd_cluster_080"] = {
            "cores_hit_vs_kong": len(c_kong),
            "rows_removed_vs_kong": rows_for(c_kong),
            "cores_hit_vs_old": len(c_old),
            "rows_removed_vs_old": rows_for(c_old),
            "cores_hit_vs_bank": len(c_bank),
            "rows_removed_vs_bank": rows_for(c_bank),
            "kong_only_extra_cores": len(c_kong - c_bank),
            "kong_only_extra_rows": rows_for(c_kong - c_bank),
            "per_dataset_vs_kong": {
                ds: {
                    "rows": ds_tot[ds],
                    "hit": ds_hit[ds],
                    "rate": round(ds_hit[ds] / ds_tot[ds], 4) if ds_tot[ds] else 0.0,
                }
                for ds, _ in ds_tot.most_common()
            },
        }
        out = DATA / "dedup/clusters_asd_kong/asd_h3_cores_hit_kong.txt"
        out.write_text("\n".join(sorted(c_kong)) + "\n")
        report["asd_cluster_080"]["hit_cores_path"] = str(out)

        # The Kong benchmark defines its own folds by clustering CDR-H3 at
        # --min-seq-id 0.4, so anything above 0.4 identity to a test loop is
        # "the same cluster" by the benchmark's own standard.
        sweep = {}
        for mid in (0.40, 0.50, 0.60, 0.70):
            hits = cluster_hits(cores, kong_cores, f"asd_vs_kong_{int(mid * 100)}",
                                args.threads, min_id=mid)
            sweep[f"{mid:.2f}"] = {
                "cores_hit": len(hits),
                "rows_removed": rows_for(hits),
                "row_frac": round(rows_for(hits) / n_rows, 4),
            }
        sweep["0.80"] = {
            "cores_hit": len(c_kong),
            "rows_removed": rows_for(c_kong),
            "row_frac": round(rows_for(c_kong) / n_rows, 4),
        }
        report["threshold_sweep_vs_kong"] = sweep

    print(json.dumps(report, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
