#!/usr/bin/env python3
"""Build the T2-clustering + T3-representation evaluation-set blocklist.

``decontam.py`` only protects the NM2025 *binding* benchmark and the public
TCR-beta benchmark; ``build_t4_refbinder_blocklist.py`` added the T4 generation
answer key. Nothing ever protected the **clustering** and **representation**
evaluation sets, which is why the 2026-08-27 audit measured 86.6% / 64.7%
effective pair leakage against them (see
``downstream/benchmark/audit_2026_08_27/D_leakage_audit.md``).

Protected sets (both ship the full ``C..[FW]`` junction, so ``has_anchors=True``):
  * ``tcr_clustering/tcrs.csv``                  -- NAR-GAB 2025 clustering
  * ``tcr_representation_paper6/target_binders.csv`` -- SCEPTR-protocol few-shot

``background_pool.csv`` is deliberately NOT protected: its ``peptide`` column is
uniformly ``__background__``, so it carries no epitope answer key. Per the field
convention (SCEPTR / TCRT5 both pretrain on unlabeled TCR corpora without
excluding benchmark sequences) it is the *label pairing* that must not leak, not
the bare receptor sequence.

Emits newline-delimited ``{cdr3b_core}|{epitope}`` keys, the same format as
``replaces_trait_blocklist.txt`` / ``t4_refbinder_blocklist.txt``, so the file
plugs straight into ``load_exclusion_keys`` + ``with_exclusion_filter`` and is
consumed correctly by BOTH ``trait_exclusion_key`` (strips anchors from TRAIT's
full junction) and ``tcr_native_exclusion_key`` (unified schema already stores
the core).

``--lev N`` widens the blocklist from exact pairs to "any training core within
edit distance N of a protected core for the same epitope". Edit distance is used
rather than MMseqs2 easy-linclust because MMseqs2's k-mer prefilter is unreliable
on 12-18 aa CDR3 cores -- the same audit found the receptor cluster split missed
~41% of Lev<=1 near-duplicates while catching exact matches fine.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from common import DATA, PROJECT_ROOT, cdr3_core, normalize_sequence

BENCH = PROJECT_ROOT / "downstream/benchmark/data"
DEFAULT_OUT = DATA / "tcr_native/dataset/t2t3_eval_blocklist.txt"

# (label, csv path, cdr3b column, epitope column, has_anchors)
PROTECTED_SETS = [
    ("clustering_nargab", BENCH / "tcr_clustering/tcrs.csv", "cdr3b", "peptide", True),
    (
        "representation_paper6",
        BENCH / "tcr_representation_paper6/target_binders.csv",
        "cdr3b",
        "peptide",
        True,
    ),
]

# Same two epitope-labelled training sources build_t4_refbinder_blocklist.py uses.
# ``has_anchors`` differs per source: the unified tcr_native ``cdr3b`` column
# already stores the anchor-free IMGT core (ingest_repo.py), while TRAIT ships
# the full ``C..[FW]`` junction.
TRAINING_SOURCES = [
    ("tcr_native", DATA / "tcr_native/dataset/train.csv", "cdr3b", "epitope_seq", False),
    (
        "trait",
        PROJECT_ROOT / "downstream/trait/step4_final/train.csv",
        "cdr3b",
        "epitope_seq",
        True,
    ),
    (
        "tcr_papers",
        DATA / "tcr_papers/dataset/train.csv",
        "cdr3b",
        "epitope_seq",
        False,
    ),
    # v2 is what v3 actually trains on (``TCR_PAPERS_DEFAULT_DIR`` points here)
    # and it carries 228,700 pair keys absent from v1. Omitting it meant the
    # ``--lev`` widening below never saw those cores, so a v2-only near-duplicate
    # of a T2/T3 eval pair was never added to the blocklist -- and the leak
    # report measured a corpus we no longer train on. The T4 builder already
    # lists both; this one had not been updated.
    (
        "tcr_papers_v2",
        DATA / "tcr_papers_v2/dataset/train.csv",
        "cdr3b",
        "epitope_seq",
        False,
    ),
]


def _read_pairs(
    path: Path, ccol: str, ecol: str, has_anchors: bool
) -> dict[str, set[str]]:
    """``{epitope -> set(cdr3b_core)}`` from one CSV."""

    out: dict[str, set[str]] = defaultdict(set)
    if not path.is_file():
        return out
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            core = cdr3_core(row.get(ccol), has_anchors=has_anchors)
            epitope = normalize_sequence(row.get(ecol))
            if core and epitope and epitope != "__BACKGROUND__":
                out[epitope].add(core)
    return out


def _merge(dicts: list[dict[str, set[str]]]) -> dict[str, set[str]]:
    merged: dict[str, set[str]] = defaultdict(set)
    for d in dicts:
        for epitope, cores in d.items():
            merged[epitope] |= cores
    return merged


def _lev_expand(
    protected: dict[str, set[str]],
    training: dict[str, set[str]],
    max_dist: int,
) -> dict[str, set[str]]:
    """Per epitope, add training cores within ``max_dist`` of a protected core."""

    from rapidfuzz import process as rf_process
    from rapidfuzz.distance import Levenshtein

    widened: dict[str, set[str]] = {}
    for epitope, prot_cores in sorted(protected.items()):
        extra = sorted(training.get(epitope, set()) - prot_cores)
        if not extra or not prot_cores:
            widened[epitope] = set(prot_cores)
            continue
        ref = sorted(prot_cores)
        matrix = rf_process.cdist(
            extra,
            ref,
            scorer=Levenshtein.distance,
            score_cutoff=max_dist,
            workers=-1,
        )
        hit = {c for c, d in zip(extra, matrix.min(axis=1)) if d <= max_dist}
        widened[epitope] = set(prot_cores) | hit
    return widened


def _keys(by_epitope: dict[str, set[str]]) -> set[str]:
    return {
        f"{core}|{epitope}"
        for epitope, cores in by_epitope.items()
        for core in cores
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--lev",
        type=int,
        default=0,
        help="widen to training cores within this edit distance (0 = exact only)",
    )
    args = ap.parse_args()

    per_set: dict[str, dict[str, set[str]]] = {}
    for label, path, ccol, ecol, anchors in PROTECTED_SETS:
        pairs = _read_pairs(path, ccol, ecol, anchors)
        if not pairs:
            raise SystemExit(f"protected set is empty or missing: {path}")
        per_set[label] = pairs

    protected = _merge(list(per_set.values()))
    exact_keys = _keys(protected)

    training: dict[str, dict[str, set[str]]] = {}
    for label, path, ccol, ecol, anchors in TRAINING_SOURCES:
        training[label] = _read_pairs(path, ccol, ecol, anchors)
    training_all = _merge(list(training.values()))

    # How much of each protected set currently sits in each training source.
    leak_report: dict[str, dict[str, object]] = {}
    for set_label, pairs in per_set.items():
        set_keys = _keys(pairs)
        entry: dict[str, object] = {"n_pairs": len(set_keys)}
        for src_label, src in training.items():
            hit = set_keys & _keys(src)
            entry[src_label] = {
                "leaked_pairs": len(hit),
                "frac": round(len(hit) / len(set_keys), 4) if set_keys else 0.0,
            }
        total = set_keys & _keys(training_all)
        entry["combined"] = {
            "leaked_pairs": len(total),
            "frac": round(len(total) / len(set_keys), 4) if set_keys else 0.0,
        }
        leak_report[set_label] = entry

    final = protected
    if args.lev > 0:
        final = _lev_expand(protected, training_all, args.lev)
    keys = _keys(final)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sorted(keys)) + "\n")

    report = {
        "schema_version": "t2t3_eval_blocklist.v1",
        "blocklist_path": str(args.out),
        "mode": f"exact_pair+lev_{args.lev}" if args.lev else "exact_pair",
        "protected_sets": {
            label: {
                "path": str(path),
                "n_epitopes": len(per_set[label]),
                "n_pairs": sum(len(v) for v in per_set[label].values()),
            }
            for label, path, _, _, _ in PROTECTED_SETS
        },
        "n_exact_pair_keys": len(exact_keys),
        "n_final_keys": len(keys),
        "n_lev_widened": len(keys) - len(exact_keys),
        "current_leak_by_set": leak_report,
    }
    report_path = args.out.parent / "t2t3_eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nWROTE {args.out}  ({len(keys):,} keys)")
    print(f"WROTE {report_path}")


if __name__ == "__main__":
    main()
