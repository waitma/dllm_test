#!/usr/bin/env python3
"""Length histograms for unlabeled CDR3β (cores) vs OTS / current repertoire.

Reports both raw TcrDesign junctions (C..[FW]) and the anchor-free cores used
in training CSVs. OTS ``chain1_cdr3`` is already an ANARCI loop (no anchors).

Does not modify any training corpus. Writes JSON + TSV under
``data/tcr_repertoire_cluster80/length_stats/`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_DLLM_ROOT = HERE.parents[3]
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))

from build_repertoire import SRC_TRAIN, SRC_VAL  # noqa: E402
from common import DATA, cdr3_core, is_valid_protein_sequence, normalize_sequence  # noqa: E402

csv.field_size_limit(2**31 - 1)

OTS_FINAL = DATA / "ots_paired_clean/final"
REP_FINAL = DATA / "tcr_repertoire/dataset"
NATIVE = DATA / "tcr_native/dataset"
DEFAULT_OUT = DATA / "tcr_repertoire_cluster80/length_stats"


def _hist_stats(counts: Counter[int]) -> dict:
    n = sum(counts.values())
    if n == 0:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None, "histogram": {}}
    running = 0
    median = None
    mid = (n + 1) // 2
    weighted = 0
    for length in sorted(counts):
        c = counts[length]
        weighted += length * c
        running += c
        if median is None and running >= mid:
            median = length
    return {
        "n": n,
        "min": min(counts),
        "max": max(counts),
        "mean": round(weighted / n, 4),
        "median": median,
        "histogram": {str(k): counts[k] for k in sorted(counts)},
    }


def _read_column(path: Path, column: str, *, has_anchors: bool) -> Counter[int]:
    hist: Counter[int] = Counter()
    if not path.is_file():
        return hist
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            core = cdr3_core(row.get(column), has_anchors=has_anchors)
            if core:
                hist[len(core)] += 1
    return hist


def _bCDR3_hists() -> tuple[Counter[int], Counter[int], Counter[int], dict[str, int]]:
    """Return junction-len, unique-core-len, all-core-len (with multiplicity), counts."""
    junction: Counter[int] = Counter()
    unique_core: Counter[int] = Counter()
    all_core: Counter[int] = Counter()
    seen: set[str] = set()
    stats = Counter()
    for src in (SRC_TRAIN, SRC_VAL):
        if not src.is_file():
            stats[f"missing_{src.name}"] += 1
            continue
        with src.open() as handle:
            for line in handle:
                stats["read"] += 1
                seq = normalize_sequence(line.strip())
                if not is_valid_protein_sequence(seq):
                    stats["drop_invalid"] += 1
                    continue
                junction[len(seq)] += 1
                core = cdr3_core(seq, has_anchors=True)
                if not core or len(core) < 4:
                    stats["drop_short"] += 1
                    continue
                all_core[len(core)] += 1
                if core in seen:
                    stats["dup_core"] += 1
                    continue
                seen.add(core)
                unique_core[len(core)] += 1
    stats["unique_cores"] = len(seen)
    return junction, unique_core, all_core, dict(stats)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[bCDR3] reading TcrDesign junctions + unique cores …", flush=True)
    junction, unique_core, all_core, read_stats = _bCDR3_hists()

    series = {
        "tcrdesign_bcdr3_junction": _hist_stats(junction),
        "tcrdesign_bcdr3_core_all_reads": _hist_stats(all_core),
        "tcrdesign_bcdr3_core_unique": _hist_stats(unique_core),
        "tcr_repertoire_train_core": _hist_stats(
            _read_column(REP_FINAL / "train.csv", "cdr3b", has_anchors=False)
        ),
        "tcr_repertoire_valid_core": _hist_stats(
            _read_column(REP_FINAL / "valid.csv", "cdr3b", has_anchors=False)
        ),
        "ots_train_beta_core": _hist_stats(
            _read_column(OTS_FINAL / "train.csv", "chain1_cdr3", has_anchors=False)
        ),
        "ots_valid_beta_core": _hist_stats(
            _read_column(OTS_FINAL / "valid.csv", "chain1_cdr3", has_anchors=False)
        ),
        "ots_holdout_beta_core": _hist_stats(
            _read_column(OTS_FINAL / "holdout.csv", "chain1_cdr3", has_anchors=False)
        ),
        "tcr_native_train_cdr3b_core": _hist_stats(
            _read_column(NATIVE / "train.csv", "cdr3b", has_anchors=False)
        ),
    }

    report = {
        "schema_version": "tcr_repertoire_length_stats.v1",
        "length_unit": "amino_acids",
        "core_definition": "anchor-free IMGT loop (strip leading C and trailing F/W when has_anchors)",
        "read_stats": read_stats,
        "series": series,
    }
    (out_dir / "length_stats.json").write_text(json.dumps(report, indent=2, sort_keys=True))

    with (out_dir / "length_hist.tsv").open("w") as handle:
        handle.write("series\tlength\tcount\tfraction\n")
        for name, payload in series.items():
            n = payload["n"] or 1
            for length_s, count in payload["histogram"].items():
                handle.write(f"{name}\t{length_s}\t{count}\t{count / n:.6f}\n")

    print(f"\n{'series':<36} {'n':>12} {'min':>4} {'med':>4} {'mean':>7} {'max':>4}", flush=True)
    for name, payload in series.items():
        print(
            f"{name:<36} {payload['n']:>12,} "
            f"{str(payload['min']):>4} {str(payload['median']):>4} "
            f"{payload['mean'] if payload['mean'] is not None else '':>7} "
            f"{str(payload['max']):>4}",
            flush=True,
        )
    print(f"\n[wrote] {out_dir / 'length_stats.json'}", flush=True)
    print(f"[wrote] {out_dir / 'length_hist.tsv'}", flush=True)


if __name__ == "__main__":
    main()
