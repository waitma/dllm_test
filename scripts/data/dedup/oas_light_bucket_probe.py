#!/usr/bin/env python3
"""Probe: does re-adding the light-chain bucket reproduce the ~378k-row OAS
removal that the pre-2026-08-16 ``oas_benchmark_blocklist.txt`` performed?

``decontam_extra.py`` currently decontaminates OAS on CDR-H3 (0.80/0.80) OR
heavy chain (0.95/0.80) and explicitly drops the light-chain bucket, because
shared germline light chains are benchmark-identity false positives. The
resumed 270m-diffusion segment (steps 9k-50k) saw 377,408 MORE records than
the three 2026-08-15 runs, and OAS is the only source whose blocklist changed
in that window. This script quantifies the light bucket to test whether it
explains the gap.
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
    DATA,
    MMSEQS_BIN,
    is_cdr3,
    is_valid_protein_sequence,
)
from decontam_extra import OAS_TRAIN, keys_oas, load_ab_banks  # noqa: E402

from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    ClusterRule,
    cluster_with_mmseqs,
)

csv.field_size_limit(2**31 - 1)


def cluster_hits(query: set[str], bank: set[str], tag: str, threads: int,
                 min_id: float, cov: float = 0.80) -> set[str]:
    if not query or not bank:
        return set()
    out_dir = DATA / "dedup/clusters_oas_light" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    res = cluster_with_mmseqs(
        rule=ClusterRule(tag, min_id, cov),
        query_sequences=query,
        benchmark_sequences=bank,
        output_dir=out_dir,
        mmseqs_bin=Path(MMSEQS_BIN),
        threads=threads,
    )
    return {q for q in query if res.sequence_to_cluster.get(q, "") in res.benchmark_clusters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    banks = load_ab_banks()

    rows: list[tuple[str, str, str]] = []
    set_h3: set[str] = set()
    set_h: set[str] = set()
    set_l: set[str] = set()

    with OAS_TRAIN.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if args.max_rows is not None and i >= args.max_rows:
                break
            h3, h, l = keys_oas(row)
            rows.append((h3, h, l))
            if h3 and is_cdr3(h3, max_len=45):
                set_h3.add(h3)
            if h and is_valid_protein_sequence(h):
                set_h.add(h)
            if l and is_valid_protein_sequence(l):
                set_l.add(l)

    n_rows = len(rows)
    c_h3 = cluster_hits(set_h3, banks["h3"], "ab_h3", args.threads, 0.80)
    c_h = cluster_hits(set_h, banks["heavy"], "ab_heavy", args.threads, 0.95)

    report: dict = {
        "n_rows": n_rows,
        "unique_h3": len(set_h3),
        "unique_heavy": len(set_h),
        "unique_light": len(set_l),
        "bank_h3": len(banks["h3"]),
        "bank_heavy": len(banks["heavy"]),
        "bank_light": len(banks["light"]),
        "current_criteria_h3_or_heavy": {},
        "light_bucket_sweep": {},
    }

    hit_cur = sum(1 for h3, h, _l in rows if h3 in c_h3 or h in c_h)
    report["current_criteria_h3_or_heavy"] = {
        "contaminated_h3_cores": len(c_h3),
        "contaminated_heavy": len(c_h),
        "rows_removed": hit_cur,
    }

    for mid in (0.95, 0.90, 0.80):
        c_l = cluster_hits(set_l, banks["light"], f"ab_light_{int(mid * 100)}",
                           args.threads, mid)
        hit_all = sum(1 for h3, h, l in rows if h3 in c_h3 or h in c_h or l in c_l)
        hit_light_only = sum(
            1 for h3, h, l in rows if l in c_l and h3 not in c_h3 and h not in c_h
        )
        report["light_bucket_sweep"][f"{mid:.2f}"] = {
            "contaminated_light_chains": len(c_l),
            "rows_hit_by_light": sum(1 for _h3, _h, l in rows if l in c_l),
            "rows_hit_by_light_only": hit_light_only,
            "rows_removed_h3_or_heavy_or_light": hit_all,
            "delta_vs_current": hit_all - hit_cur,
        }

    report["target_gap_to_explain"] = 377408
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
