#!/usr/bin/env python3
"""Exact CDR3β-core overlap: OTS full-length train vs TcrDesign repertoire.

Compares OTS ``chain1_cdr3`` (already anchor-free) against:
  * TcrDesign unique clean cores (~40.3M)
  * clean MMseqs cluster representatives (~31.0M)
  * cluster80 sampled train (2.0M)
  * previous random repertoire train (~2.13M)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_DLLM_ROOT = HERE.parents[3]
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))

from common import DATA, cdr3_core  # noqa: E402

OUT_DIR = DATA / "tcr_repertoire_cluster80/overlap"
OTS_TRAIN = DATA / "ots_paired_clean/final/train.csv"
CLEAN = DATA / "tcr_repertoire_cluster80/work/clean_cores.txt"
BLOCKED = DATA / "tcr_repertoire_cluster80/work/blocked_cores.txt"
TSV = DATA / "tcr_repertoire_cluster80/work/cluster_cluster.tsv"
C80_TRAIN = DATA / "tcr_repertoire_cluster80/dataset/train.csv"
OLD_TRAIN = DATA / "tcr_repertoire/dataset/train.csv"


def _load_lines(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open() as handle:
        for line in handle:
            s = line.strip()
            if s:
                out.add(s)
    return out


def _csv_cdr3b(path: Path) -> tuple[int, set[str]]:
    seqs: set[str] = set()
    n = 0
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            n += 1
            core = (row.get("cdr3b") or "").strip()
            if core:
                seqs.add(core)
    return n, seqs


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] load OTS train CDR3β cores …", flush=True)
    ots_rows = 0
    ots_empty = 0
    ots: set[str] = set()
    with OTS_TRAIN.open(newline="") as handle:
        for row in csv.DictReader(handle):
            ots_rows += 1
            core = cdr3_core(row.get("chain1_cdr3"), has_anchors=False)
            if not core:
                ots_empty += 1
                continue
            ots.add(core)
    print(f"    ots_train_rows={ots_rows:,} unique_cores={len(ots):,} empty={ots_empty}", flush=True)

    print("[2] blocked ∩ OTS train …", flush=True)
    blocked = _load_lines(BLOCKED)
    blocked_hit = len(ots & blocked)
    print(f"    blocked={len(blocked):,} ∩ ots_train={blocked_hit:,}", flush=True)

    print("[3] TcrDesign unique clean cores ∩ OTS …", flush=True)
    n_clean = 0
    hit_clean = 0
    with CLEAN.open() as handle:
        for line in handle:
            s = line.strip()
            if not s:
                continue
            n_clean += 1
            if s in ots:
                hit_clean += 1
    print(
        f"    clean_cores={n_clean:,} exact_hit={hit_clean:,} "
        f"({100 * hit_clean / max(len(ots), 1):.2f}% of OTS unique, "
        f"{100 * hit_clean / max(n_clean, 1):.4f}% of TcrDesign clean)",
        flush=True,
    )

    print("[4] cluster80 2M train ∩ OTS …", flush=True)
    n80, s80 = _csv_cdr3b(C80_TRAIN)
    hit80 = len(s80 & ots)
    print(f"    n={n80:,} unique={len(s80):,} ∩ ots={hit80:,}", flush=True)

    print("[5] old random 2.13M train ∩ OTS …", flush=True)
    nold, sold = _csv_cdr3b(OLD_TRAIN)
    hit_old = len(sold & ots)
    print(f"    n={nold:,} unique={len(sold):,} ∩ ots={hit_old:,}", flush=True)

    print("[6] 31M clean cluster reps ∩ OTS (stream tsv) …", flush=True)
    dirty: set[str] = set()
    reps: set[str] = set()
    n_edges = 0
    with TSV.open() as handle:
        for line in handle:
            n_edges += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            reps.add(rep)
            if member in blocked or rep in blocked:
                dirty.add(rep)
    clean_reps = reps - dirty
    hit_reps = len(clean_reps & ots)
    print(
        f"    clusters={len(reps):,} dirty={len(dirty):,} "
        f"clean_reps={len(clean_reps):,} ∩ ots={hit_reps:,}",
        flush=True,
    )

    payload = {
        "ots_train_rows": ots_rows,
        "ots_train_unique_cdr3b": len(ots),
        "ots_empty": ots_empty,
        "blocklist_n": len(blocked),
        "blocklist_intersect_ots_train": blocked_hit,
        "tcrdesign_clean_cores": n_clean,
        "tcrdesign_clean_intersect_ots": hit_clean,
        "cluster80_train_n": n80,
        "cluster80_train_intersect_ots": hit80,
        "old_repertoire_train_n": nold,
        "old_repertoire_train_intersect_ots": hit_old,
        "clean_cluster_reps": len(clean_reps),
        "clean_cluster_reps_intersect_ots": hit_reps,
        "n_clusters": len(reps),
        "n_dirty_clusters": len(dirty),
        "n_edges": n_edges,
    }
    out_json = OUT_DIR / "ots_vs_repertoire.json"
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    print(f"WROTE {out_json}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
