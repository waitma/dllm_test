#!/usr/bin/env python3
"""Move near-duplicate valid/holdout rows into train.

The receptor-cluster split (MMseqs2 easy-linclust 0.80/0.80, seed 42) is only
effective at the exact level: the 2026-08-27 audit measured 0.1% exact CDR3b-core
overlap between valid and train but **40.9% at edit distance <= 1** (holdout
41.7%), versus <0.1% for both null controls (within-sequence shuffle and
composition-matched resampling). The near-duplication is therefore real and not
an artefact of CDR3 cores being short and convergent -- MMseqs2's k-mer prefilter
is simply unreliable on 12-18 aa sequences.

Rather than delete those rows, they are moved into train: they carry real signal,
and the only thing wrong with them is that they cannot serve as held-out
evaluation data. This keeps eval_loss and any holdout metric honest at a cost of
about +2% train rows.

Threshold is edit distance <= 1 by default. Rationale: the field convention
(arXiv 2606.04994) allows up to 3 substitutions in the CDR3b region and discards
40-70% of the test set; Lev<=1 discards ~41% here, at the lower end of that band,
while Lev<=2 would discard ~74% and leave only ~2,000 eval rows.

Iterates to a fixed point. This matters: a valid row at distance 2 from train may
sit at distance 1 from *another* valid row that was just moved into train, so a
single pass does not actually guarantee the stated separation.

Distances come from rapidfuzz with a score cutoff, which is exact. A hash-based
deletion-neighbourhood index was tried first and rejected: intersecting deletion
neighbourhoods is necessary but *not* sufficient for edit distance <= 1 (equal
length strings deleted at different positions collide, e.g. "AB" vs "BA" is
distance 2), and it over-reported 49.2% against rapidfuzz's 40.8% on
tcr_native/valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from common import DATA, PROJECT_ROOT, cdr3_core, normalize_sequence

# (dataset root, whether ``cdr3b`` carries the C..[FW] anchors).
#
# The anchor convention differs per source and getting it wrong silently
# corrupts every count -- it already happened once in this corpus's history.
# tcr_native / tcr_papers store the anchor-free IMGT core; TRAIT stores the full
# junction. Comparisons only ever happen within one dataset, so a wrong flag
# would not mix conventions, but it would leave the constant anchors in the
# string and dilute the edit distance.
#
# tcr_papers_v2 is a separate root because it was rebuilt from scratch by
# ``finalize_papers.py --out-root data/tcr_papers_v2`` *after* the 2026-08-28
# near-dedup pass, so it does not inherit v1's cleaned splits and has to be
# processed on its own. v3 trains on v2, not v1.
#
# tcr_repertoire is included but is the expensive one: 1.97M train cores against
# 201k valid + 201k holdout. Its group-aware split is exact-disjoint by
# construction (measured: 0.00% exact on a 3,000-row sample) yet still 36.7% at
# Lev<=1, i.e. the same order as tcr_native's 40.9%, so it does need the same
# treatment. An earlier version of this comment guessed the filter would "strip
# most of valid"; measurement says 36.7%, leaving ~127k rows -- and eval only
# ever samples ``max_eval_rows_per_source=2000`` of them anyway.
DATASETS = {
    "tcr_native": (DATA / "tcr_native/dataset", False),
    "tcr_papers": (DATA / "tcr_papers/dataset", False),
    "tcr_papers_v2": (DATA / "tcr_papers_v2/dataset", False),
    "tcr_repertoire": (DATA / "tcr_repertoire/dataset", False),
    "trait": (PROJECT_ROOT / "downstream/trait/step4_final", True),
}
EVAL_SPLITS = ("valid", "holdout")


def _core(row: dict[str, str], has_anchors: bool = False) -> str:
    return cdr3_core(row.get("cdr3b"), has_anchors=has_anchors)


def _near_dup_flags_rapidfuzz(
    queries: list[str], reference: set[str], max_dist: int
) -> list[bool]:
    from rapidfuzz import process as rf_process
    from rapidfuzz.distance import Levenshtein

    ref = sorted(reference)
    flags = [False] * len(queries)
    # cdist allocates the full block x reference int32 matrix regardless of
    # score_cutoff, so the block has to shrink as the reference grows. A fixed
    # 2000 was fine for tcr_native's 60k reference (0.5 GB) but would ask for
    # ~16 GB against tcr_repertoire's 1.97M cores. Cap the matrix at ~1.6 GB.
    target_cells = 4 * 10**8
    block = max(1, min(2000, target_cells // max(1, len(ref))))
    for start in range(0, len(queries), block):
        chunk = queries[start : start + block]
        matrix = rf_process.cdist(
            chunk, ref, scorer=Levenshtein.distance, score_cutoff=max_dist, workers=-1
        )
        for offset, best in enumerate(matrix.min(axis=1)):
            flags[start + offset] = bool(chunk[offset]) and best <= max_dist
    return flags


def _near_dup_flags(
    queries: list[str], reference: set[str], max_dist: int
) -> list[bool]:
    if max_dist <= 0:
        return [bool(q) and q in reference for q in queries]
    return _near_dup_flags_rapidfuzz(queries, reference, max_dist)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def process(
    name: str,
    root: Path,
    max_dist: int,
    max_rounds: int,
    apply: bool,
    has_anchors: bool = False,
) -> dict:
    train_path = root / "train.csv"
    fields, train = _read(train_path)
    splits = {s: _read(root / f"{s}.csv") for s in EVAL_SPLITS}

    report: dict = {
        "dir": str(root),
        "max_dist": max_dist,
        "has_anchors": has_anchors,
        "before": {"train": len(train), **{s: len(v[1]) for s, v in splits.items()}},
        "rounds": [],
    }

    moved: dict[str, list[dict[str, str]]] = {s: [] for s in EVAL_SPLITS}
    kept = {s: list(v[1]) for s, v in splits.items()}
    reference = {c for c in (_core(r, has_anchors) for r in train) if c}

    for rnd in range(1, max_rounds + 1):
        round_moved = 0
        entry: dict = {"round": rnd}
        for split in EVAL_SPLITS:
            rows = kept[split]
            cores = [_core(r, has_anchors) for r in rows]
            flags = _near_dup_flags(cores, reference, max_dist)
            hit = [r for r, f in zip(rows, flags) if f]
            kept[split] = [r for r, f in zip(rows, flags) if not f]
            moved[split].extend(hit)
            reference |= {c for c, f in zip(cores, flags) if f and c}
            entry[split] = len(hit)
            round_moved += len(hit)
        report["rounds"].append(entry)
        if round_moved == 0:
            break

    # Diagnostics: how many moved rows share their epitope with a train row, i.e.
    # were an actual (CDR3b, epitope) answer-key leak rather than just a similar
    # receptor. Reported, not used as the criterion -- the decision was CDR3b-only.
    train_pairs = {
        f"{_core(r, has_anchors)}|{normalize_sequence(r.get('epitope_seq'))}"
        for r in train
    }
    diagnostics: dict = {}
    for split in EVAL_SPLITS:
        rows = moved[split]
        exact_pair = sum(
            1
            for r in rows
            if f"{_core(r, has_anchors)}|{normalize_sequence(r.get('epitope_seq'))}"
            in train_pairs
        )
        blank = sum(1 for r in kept[split] if not _core(r, has_anchors))
        diagnostics[split] = {
            "moved": len(rows),
            "moved_sharing_train_epitope_pair": exact_pair,
            "moved_by_sub_source": dict(
                Counter(r.get("source", "?") for r in rows).most_common()
            ),
            "remaining_with_unusable_cdr3b": blank,
            "remaining_by_sub_source": dict(
                Counter(r.get("source", "?") for r in kept[split]).most_common()
            ),
        }
    report["diagnostics"] = diagnostics

    new_train = train + [r for s in EVAL_SPLITS for r in moved[s]]
    report["after"] = {
        "train": len(new_train),
        **{s: len(kept[s]) for s in EVAL_SPLITS},
    }
    report["moved_total"] = sum(len(moved[s]) for s in EVAL_SPLITS)
    report["train_growth_frac"] = (
        round((len(new_train) - len(train)) / len(train), 4) if train else 0.0
    )

    if apply:
        for path in [train_path] + [root / f"{s}.csv" for s in EVAL_SPLITS]:
            backup = path.with_suffix(".csv.pre_neardedup")
            if not backup.exists():
                shutil.copy2(path, backup)
        _write(train_path, fields, new_train)
        for split in EVAL_SPLITS:
            _write(root / f"{split}.csv", splits[split][0], kept[split])
        report["applied"] = True
    else:
        report["applied"] = False
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=sorted(DATASETS))
    ap.add_argument("--max-dist", type=int, default=1)
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the CSVs (originals saved as *.csv.pre_neardedup)",
    )
    args = ap.parse_args()

    out: dict = {"schema_version": "near_dup_eval_move.v1", "datasets": {}}
    for name in args.datasets:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name}; known: {sorted(DATASETS)}")
        root, has_anchors = DATASETS[name]
        print(f"\n=== {name} (has_anchors={has_anchors}) ===", flush=True)
        rep = process(
            name, root, args.max_dist, args.max_rounds, args.apply, has_anchors
        )
        out["datasets"][name] = rep
        b, a = rep["before"], rep["after"]
        for split in EVAL_SPLITS:
            frac = (b[split] - a[split]) / b[split] if b[split] else 0.0
            print(
                f"  {split:8s} {b[split]:>7,} -> {a[split]:>7,} "
                f"(moved {b[split] - a[split]:,}, {frac:.1%})"
            )
        print(
            f"  {'train':8s} {b['train']:>7,} -> {a['train']:>7,} "
            f"(+{rep['train_growth_frac']:.2%})"
        )
        print(f"  rounds: {rep['rounds']}")
        for split in EVAL_SPLITS:
            d = rep["diagnostics"][split]
            print(
                f"  {split} moved rows that were a real (CDR3b,epitope) train pair: "
                f"{d['moved_sharing_train_epitope_pair']:,}/{d['moved']:,}"
            )

    # Merge rather than overwrite: the report is keyed by dataset, and running a
    # subset (e.g. only tcr_papers_v2) used to wipe the record of every dataset
    # processed in earlier runs.
    report_path = DATA / "tcr_native/dataset/near_dup_eval_move_report.json"
    if report_path.is_file():
        try:
            previous = json.loads(report_path.read_text())
        except json.JSONDecodeError:
            previous = {}
        merged = previous.get("datasets", {})
        merged.update(out["datasets"])
        out["datasets"] = merged
    report_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    print(f"\nWROTE {report_path}")
    if not args.apply:
        print("DRY RUN -- rerun with --apply to rewrite the CSVs")


if __name__ == "__main__":
    main()
