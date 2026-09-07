#!/usr/bin/env python3
"""Finalize the tcr_papers corpus: dedup, decontaminate, split, write CSVs.

Pipeline (mirrors unify.py + decontam.py + cluster_split.py, kept in one place
because this corpus has a single consumer):

  1. concatenate the selected ``unified/*.csv`` shards
  2. cross-source exact dedup on ``(cdr3b, cdr3a, epitope, allele, relation)``
     keeping the richest row (tier A > B > C, paired alpha/beta over beta-only)
  3. drop rows already present in the *existing* tcr_native + TRAIT training
     corpus, so mixing this source in does not silently re-weight records the
     model already sees
  4. T4 generation decontamination: drop any ``(cdr3b_core | epitope)`` that is
     a reference binder in the epitope-conditioned generation benchmark
  5. NM2025 / public binding-benchmark decontamination: exact CDR3b core match,
     plus MMseqs2 0.80/0.80 co-clustering unless ``--no-cluster``
  6. group-aware split by CDR3b core (a clonotype never straddles splits)

``gratcr_mira`` is excluded by default: it is tier C (no MHC), covers only 151
epitopes, and ImmuneCODE/MIRA carries a click-through licence that should be
reviewed before it enters a published training corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    DATA,
    PROJECT_ROOT,
    UNIFIED_COLUMNS,
    cdr3_core,
    normalize_sequence,
    strip_alignment_gaps,
)

_SEQ_COLS = ("alpha_fv", "beta_fv", "cdr3a", "cdr3b", "epitope_seq", "mhc_seq")

CORPUS = DATA / "tcr_papers"
UNIFIED_DIR = CORPUS / "unified"
ALL_SOURCES = [
    "tcrt5", "tcrdiff", "gratcr_tep", "epidiff", "gratcr_mira",
    "tcrdesign26_beta", "tcrdesign26_paired", "tcrdesign26_pmhc",
]
DEFAULT_SOURCES = ["tcrt5", "tcrdiff", "gratcr_tep", "epidiff"]
T4_BLOCKLIST = DATA / "tcr_native/dataset/t4_refbinder_blocklist.txt"
MMSEQS_BIN = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs")

_TIER_RANK = {"A": 3, "B": 2, "C": 1, "": 0}

# Existing epitope-conditioned training sources, with the per-source CDR3b
# anchor convention (unified tcr_native stores the anchor-free core; TRAIT
# stores the full C..[FW] junction).
EXISTING_SOURCES = [
    (DATA / "tcr_native/dataset/train.csv", False),
    (PROJECT_ROOT / "downstream/trait/step4_final/train.csv", True),
]


def _stable_rank(*parts) -> int:
    return int(hashlib.sha1(("|".join(str(p) for p in parts)).encode()).hexdigest(), 16)


def _dedup_key(row: dict) -> tuple:
    return (
        row.get("cdr3b", ""),
        row.get("cdr3a", ""),
        row.get("epitope_seq", ""),
        row.get("mhc_allele_norm", ""),
        row.get("relation", ""),
    )


def _richness(row: dict) -> tuple:
    return (_TIER_RANK.get(row.get("tier", ""), 0), 1 if row.get("cdr3a") else 0)


def _existing_keys() -> tuple[set[tuple], set[str]]:
    """``(dedup keys, cdr3b cores)`` already in the live training corpus."""

    keys: set[tuple] = set()
    cores: set[str] = set()
    for path, has_anchors in EXISTING_SOURCES:
        if not path.is_file():
            continue
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                core = cdr3_core(row.get("cdr3b"), has_anchors=has_anchors)
                alpha = cdr3_core(row.get("cdr3a"), has_anchors=has_anchors)
                epitope = normalize_sequence(row.get("epitope_seq"))
                keys.add(
                    (
                        core,
                        alpha,
                        epitope,
                        row.get("mhc_allele_norm", "") or "",
                        row.get("relation", "") or "",
                    )
                )
                if core:
                    cores.add(core)
    return keys, cores


def _binding_benchmark_cores() -> set[str]:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from decontam import load_benchmark_sets  # noqa: E402

    bench = load_benchmark_sets()
    return set(bench["binding_benchmark"]) | set(bench["full_bank"])


# Files whose content decides what gets filtered out. Recorded in the report so
# assert_corpus_fresh.py can refuse to train on a corpus built against a
# blocklist that has since been rebuilt (this happened on 2026-08-28: the T4
# list grew 67,013 -> 68,846 keys 45 minutes after a build, and the stale
# report still said PASS).
_PROVENANCE_FILES = {
    "t4_refbinder": T4_BLOCKLIST,
    "downstream_bank_tcr_cdr3b": DATA / "dedup/banks/tcr_cdr3b.txt",
    "public_benchmark_references": (
        PROJECT_ROOT / "downstream/benchmark/data/tcr_beta_public_benchmark/references.csv"
    ),
}


def _provenance() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for name, path in _PROVENANCE_FILES.items():
        if not path.is_file():
            out[name] = {"path": str(path), "exists": False}
            continue
        out[name] = {
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest()[:16],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sources", nargs="+", choices=ALL_SOURCES, default=DEFAULT_SOURCES)
    ap.add_argument("--out-root", default=str(CORPUS),
                    help="corpus root to write dataset/ into. Defaults to the "
                         "existing tcr_papers root; point it elsewhere (e.g. "
                         "data/tcr_papers_v2) when adding sources, so the corpus "
                         "the current checkpoints trained on stays byte-identical.")
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--valid-frac", type=float, default=0.02)
    ap.add_argument("--holdout-frac", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    dataset_dir = Path(args.out_root) / "dataset"

    report: dict = {"schema_version": "tcr_papers.v1", "sources": args.sources,
                    "out_root": str(args.out_root),
                    "blocklist_provenance": _provenance()}

    # -- 1/2. concatenate + cross-source dedup ----------------------------- #
    best: dict[tuple, dict] = {}
    read = Counter()
    collapsed = 0
    for name in args.sources:
        path = UNIFIED_DIR / f"{name}.csv"
        if not path.is_file():
            raise SystemExit(f"missing shard {path}; run ingest_papers.py first")
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                read[name] += 1
                key = _dedup_key(row)
                if key not in best:
                    best[key] = row
                else:
                    collapsed += 1
                    if _richness(row) > _richness(best[key]):
                        best[key] = row
    rows = list(best.values())
    report["input_rows_read"] = dict(read)
    report["cross_source_duplicates_collapsed"] = collapsed
    report["after_dedup"] = len(rows)

    # -- 3. drop what the live corpus already has --------------------------- #
    existing_keys, existing_cores = _existing_keys()
    before = len(rows)
    rows = [r for r in rows if _dedup_key(r) not in existing_keys]
    report["dropped_already_in_live_corpus"] = before - len(rows)
    report["live_corpus_keys"] = len(existing_keys)

    # -- 4. T4 generation reference-binder decontamination ------------------ #
    t4_keys = {
        line.strip()
        for line in T4_BLOCKLIST.read_text().splitlines()
        if line.strip()
    }
    before = len(rows)
    rows = [
        r
        for r in rows
        if f"{r.get('cdr3b', '')}|{r.get('epitope_seq', '')}" not in t4_keys
    ]
    report["t4_refbinder_keys"] = len(t4_keys)
    report["dropped_t4_refbinder"] = before - len(rows)

    # -- 5. binding-benchmark decontamination ------------------------------- #
    bench_cores = _binding_benchmark_cores()
    before = len(rows)
    rows = [r for r in rows if r.get("cdr3b", "") not in bench_cores]
    report["binding_benchmark_cores"] = len(bench_cores)
    report["dropped_binding_exact"] = before - len(rows)

    cluster_dropped = 0
    if not args.no_cluster:
        import sys

        sys.path.insert(0, str(PROJECT_ROOT))
        from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
            ClusterRule,
            cluster_with_mmseqs,
        )

        cores = {r["cdr3b"] for r in rows if r.get("cdr3b")}
        out_dir = CORPUS / "clusters_binding"
        out_dir.mkdir(parents=True, exist_ok=True)
        res = cluster_with_mmseqs(
            rule=ClusterRule("tcr_beta_cdr", 0.80, 0.80),
            query_sequences=cores,
            benchmark_sequences=bench_cores,
            output_dir=out_dir,
            mmseqs_bin=MMSEQS_BIN,
            threads=args.threads,
        )
        bad = {
            c
            for c in cores
            if res.sequence_to_cluster.get(c, "") in res.benchmark_clusters
        }
        before = len(rows)
        rows = [r for r in rows if r.get("cdr3b", "") not in bad]
        cluster_dropped = before - len(rows)
        report["contaminated_clusters_cores"] = len(bad)
    report["dropped_binding_cluster"] = cluster_dropped
    report["decontam_mode"] = "exact" if args.no_cluster else "exact+cluster_0.80_0.80"

    # residual verification
    residual_t4 = sum(
        1
        for r in rows
        if f"{r.get('cdr3b', '')}|{r.get('epitope_seq', '')}" in t4_keys
    )
    residual_bind = sum(1 for r in rows if r.get("cdr3b", "") in bench_cores)
    report["residual_t4_hits"] = residual_t4
    report["residual_binding_exact_hits"] = residual_bind
    report["PASS"] = residual_t4 == 0 and residual_bind == 0

    # -- 6. group-aware split by CDR3b core --------------------------------- #
    by_core: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_core[r.get("cdr3b", "")].append(r)
    splits: dict[str, list[dict]] = {"train": [], "valid": [], "holdout": []}
    denom = 10**6
    valid_cut = int(args.valid_frac * denom)
    holdout_cut = valid_cut + int(args.holdout_frac * denom)
    for core, group in by_core.items():
        bucket = _stable_rank(args.seed, core) % denom
        if bucket < valid_cut:
            splits["valid"].extend(group)
        elif bucket < holdout_cut:
            splits["holdout"].extend(group)
        else:
            splits["train"].extend(group)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for name, group in splits.items():
        group.sort(key=lambda r: r["record_id"])
        with (dataset_dir / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
            writer.writeheader()
            for row in group:
                writer.writerow(
                    {
                        c: (
                            strip_alignment_gaps(row.get(c, "") or "")
                            if c in _SEQ_COLS
                            else row.get(c, "")
                        )
                        for c in UNIFIED_COLUMNS
                    }
                )

    # cross-split clonotype disjointness audit
    core_sets = {
        k: {r.get("cdr3b", "") for r in v if r.get("cdr3b")} for k, v in splits.items()
    }
    overlaps = {
        f"{a}_vs_{b}": len(core_sets[a] & core_sets[b])
        for a, b in (("train", "valid"), ("train", "holdout"), ("valid", "holdout"))
    }
    report["split_counts"] = {k: len(v) for k, v in splits.items()}
    report["split_cdr3b_overlap"] = overlaps
    report["split_disjoint_PASS"] = all(v == 0 for v in overlaps.values())
    report["final_rows"] = len(rows)
    report["by_source"] = dict(Counter(r["source"] for r in rows))
    report["by_tier"] = dict(Counter(r["tier"] for r in rows))
    report["by_relation"] = dict(Counter(r["relation"] for r in rows))
    report["by_task_type"] = dict(Counter(r["task_type"] for r in rows))
    report["unique_epitopes"] = len({r["epitope_seq"] for r in rows})
    report["paired_alpha_beta"] = sum(1 for r in rows if r.get("cdr3a"))
    report["new_epitopes_vs_live_corpus"] = len(
        {r["epitope_seq"] for r in rows}
        - {
            normalize_sequence(k[2])
            for k in existing_keys
        }
    )
    report["dataset_dir"] = str(dataset_dir)

    (dataset_dir / "finalize_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
