#!/usr/bin/env python3
"""Build the T4 epitope-conditioned-generation reference-binder blocklist.

The existing tcr_native decontamination (``decontam.py``) only protects the
NM2025 *binding* benchmark and the public TCR-beta benchmark. It does not know
about the T4 **generation** benchmark, whose answer key is the ``ref_binders``
list stored per pMHC in ``eval_conditional.json``. As a result a large fraction
of that answer key currently sits inside the training mix.

Emits newline-delimited ``{cdr3b_core}|{epitope}`` exclusion keys in the same
format as ``replaces_trait_blocklist.txt`` so they plug straight into
``load_exclusion_keys`` + ``with_exclusion_filter``.

With ``--cluster`` the blocklist is widened from exact pairs to
"any CDR3b core that co-clusters (MMseqs2 0.80/0.80) with a reference binder
for the same epitope", matching the criteria decontam.py uses elsewhere.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import DATA, PROJECT_ROOT, cdr3_core, normalize_sequence

EVAL_JSON = (
    PROJECT_ROOT
    / "downstream/benchmark/data/tcr_generation_bench/eval_conditional.json"
)
# Setting B (epitope-conditioned TCR design, IMMREP23-stratified common/rare/novel).
# A SECOND, mostly-disjoint answer key: 44 epitopes of which only 13 appear in
# eval_conditional.json, so 1,345 of its 2,792 (core|epitope) pairs used to be
# protected by nothing at all. Measured 2026-08-28: that gap leaked 423 rows into
# tcr_native/train.csv, 163 into tcr_papers and 352 into tcr_papers_v2.
# Keyed by epitope rather than pMHC, and only ``ref_binders`` are answers --
# ``train_binders`` are handed to the model as in-context examples at eval time,
# so blocking those would remove legitimate inputs.
DESIGN_EVAL_JSON = PROJECT_ROOT / "downstream/benchmark/data/tcr_design/eval.json"
DEFAULT_OUT = DATA / "tcr_native/dataset/t4_refbinder_blocklist.txt"
MMSEQS_BIN = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs")


def load_ref_binders() -> tuple[dict[str, set[str]], dict]:
    """Return ``{epitope -> set(cdr3b_core)}`` plus a provenance summary.

    Merges both epitope-conditioned-generation answer keys (T4 + Setting B);
    they share the ``(cdr3b_core|epitope)`` key space, so one blocklist covers
    both and every consumer picks the fix up for free.
    """

    by_epitope: dict[str, set[str]] = defaultdict(set)

    # -- T4: keyed by pMHC, epitope stored in the entry --------------------- #
    payload = json.loads(EVAL_JSON.read_text())
    by_category = Counter()
    pmhc_meta = {}
    for pmhc, entry in payload.items():
        epitope = normalize_sequence(entry["epitope"])
        by_category[entry.get("category", "?")] += 1
        cores = set()
        for raw in entry.get("ref_binders", []):
            core = cdr3_core(raw, has_anchors=True)
            if core:
                cores.add(core)
        by_epitope[epitope] |= cores
        pmhc_meta[pmhc] = {
            "epitope": epitope,
            "allele": entry.get("allele", ""),
            "category": entry.get("category", ""),
            "n_ref_binders": len(entry.get("ref_binders", [])),
            "n_unique_cores": len(cores),
        }
    t4_pairs = sum(len(v) for v in by_epitope.values())

    # -- Setting B: keyed by epitope --------------------------------------- #
    design_category = Counter()
    design_epitopes = 0
    if DESIGN_EVAL_JSON.is_file():
        design = json.loads(DESIGN_EVAL_JSON.read_text())
        design_epitopes = len(design)
        for raw_epitope, entry in design.items():
            epitope = normalize_sequence(raw_epitope)
            design_category[entry.get("category", "?")] += 1
            for raw in entry.get("ref_binders", []):
                core = cdr3_core(raw, has_anchors=True)
                if core:
                    by_epitope[epitope].add(core)

    summary = {
        "eval_json": str(EVAL_JSON),
        "design_eval_json": str(DESIGN_EVAL_JSON),
        "n_pmhc": len(payload),
        "by_category": dict(by_category),
        "design_n_epitopes": design_epitopes,
        "design_by_category": dict(design_category),
        "n_epitopes": len(by_epitope),
        "n_unique_pairs": sum(len(v) for v in by_epitope.values()),
        "n_pairs_from_t4_only": t4_pairs,
        "n_pairs_added_by_design": sum(len(v) for v in by_epitope.values()) - t4_pairs,
        "per_pmhc": pmhc_meta,
    }
    return by_epitope, summary


# ``has_anchors`` per source: the unified tcr_native ``cdr3b`` column already
# stores the anchor-free IMGT core (see ingest_repo.py), while TRAIT ships the
# full ``C..[FW]`` junction (which is why trait_exclusion_key strips it).
TRAINING_SOURCES = [
    (DATA / "tcr_native/dataset/train.csv", "cdr3b", "epitope_seq", False),
    (
        PROJECT_ROOT / "downstream/trait/step4_final/train.csv",
        "cdr3b",
        "epitope_seq",
        True,
    ),
    # Published-paper corpora, same unified schema (anchor-free core). Listed so
    # the cluster widening below can also block *their* near-duplicates of a
    # reference binder, not just tcr_native's. Missing entries are skipped, so a
    # corpus that has not been built yet is not an error.
    (DATA / "tcr_papers/dataset/train.csv", "cdr3b", "epitope_seq", False),
    (DATA / "tcr_papers_v2/dataset/train.csv", "cdr3b", "epitope_seq", False),
]


def _training_cores() -> dict[str, set[str]]:
    """CDR3b cores present in the current epitope-conditioned training sources,
    grouped by epitope. Used to widen the blocklist by clustering and to report
    how much of the answer key is currently inside the mix."""

    out: dict[str, set[str]] = defaultdict(set)
    for path, ccol, ecol, has_anchors in TRAINING_SOURCES:
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                core = cdr3_core(row.get(ccol), has_anchors=has_anchors)
                epitope = normalize_sequence(row.get(ecol))
                if core and epitope:
                    out[epitope].add(core)
    return out


def _cluster_expand(
    ref_by_epitope: dict[str, set[str]],
    train_by_epitope: dict[str, set[str]],
    threads: int,
) -> dict[str, set[str]]:
    """Per epitope, add training cores that co-cluster with a reference binder."""

    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
        ClusterRule,
        cluster_with_mmseqs,
    )

    widened: dict[str, set[str]] = {}
    cluster_dir = DATA / "tcr_native/clusters_t4_refbinder"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    for epitope, ref_cores in sorted(ref_by_epitope.items()):
        train_cores = train_by_epitope.get(epitope, set())
        extra = train_cores - ref_cores
        if not extra or not ref_cores:
            widened[epitope] = set(ref_cores)
            continue
        res = cluster_with_mmseqs(
            rule=ClusterRule("tcr_beta_cdr", 0.80, 0.80),
            query_sequences=extra,
            benchmark_sequences=ref_cores,
            output_dir=cluster_dir / epitope,
            mmseqs_bin=MMSEQS_BIN,
            threads=threads,
        )
        hit = {
            c
            for c in extra
            if res.sequence_to_cluster.get(c, "") in res.benchmark_clusters
        }
        widened[epitope] = set(ref_cores) | hit
    return widened


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--cluster",
        action="store_true",
        help="widen from exact pairs to MMseqs2 0.80/0.80 co-clusters",
    )
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    ref_by_epitope, summary = load_ref_binders()
    train_by_epitope = _training_cores()

    exact_keys = {
        f"{core}|{epitope}"
        for epitope, cores in ref_by_epitope.items()
        for core in cores
    }
    current_leak = {
        f"{core}|{epitope}"
        for epitope, cores in train_by_epitope.items()
        for core in cores
    } & exact_keys

    final_by_epitope = ref_by_epitope
    if args.cluster:
        final_by_epitope = _cluster_expand(
            ref_by_epitope, train_by_epitope, args.threads
        )
    keys = {
        f"{core}|{epitope}"
        for epitope, cores in final_by_epitope.items()
        for core in cores
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(keys)) + "\n")

    report = {
        "schema_version": "t4_refbinder_blocklist.v1",
        "blocklist_path": str(args.out),
        "mode": "cluster_0.80_0.80" if args.cluster else "exact_pair",
        "n_exact_pair_keys": len(exact_keys),
        "n_final_keys": len(keys),
        "n_cluster_widened": len(keys) - len(exact_keys),
        "current_training_leak_pairs": len(current_leak),
        "current_training_leak_frac_of_answer_key": round(
            len(current_leak) / len(exact_keys), 4
        )
        if exact_keys
        else 0.0,
        **summary,
    }
    report_path = args.out.parent / "t4_refbinder_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    slim = {k: v for k, v in report.items() if k != "per_pmhc"}
    print(json.dumps(slim, indent=2, sort_keys=True))
    print(f"\nWROTE {args.out}  ({len(keys):,} keys)")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
