#!/usr/bin/env python3
"""Cluster-then-pick unlabeled CDR3β repertoire (does NOT overwrite v3).

Replaces the random ``sample(2_000_000)`` in ``build_repertoire.py`` with:

  1. exact-core dedup + exact blocklist (same as the v3 builder)
  2. MMseqs2 easy-linclust 0.80 id / 0.80 cov on the full unique pool + blocklist
  3. drop every cluster that touches a benchmark core
  4. group-disjoint hash split by cluster representative (0.5% / 0.5% eval)
  5. keep the MMseqs representative of each clean cluster
  6. if train reps exceed ``--max-train``, sample clusters (not raw sequences)

Writes to ``data/tcr_repertoire_cluster80/`` by default so the live v3 corpus
at ``data/tcr_repertoire/`` stays untouched.

Resume: each stage writes a stamp under ``work/``. Re-run the same command;
completed stages are skipped unless ``--force`` is set.

Typical background launch::

    nohup python cluster_sample_repertoire.py --threads 32 \\
        > /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_repertoire_cluster80/run.log 2>&1 &
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_DLLM_ROOT = HERE.parents[3]
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))

from build_repertoire import (  # noqa: E402
    MMSEQS_BIN,
    PROVENANCE,
    SRC_TRAIN,
    SRC_VAL,
    build_blocked,
    _stable_rank,
)
from common import (  # noqa: E402
    DATA,
    UNIFIED_COLUMNS,
    cdr3_core,
    is_valid_protein_sequence,
    normalize_sequence,
    record_id,
)

csv.field_size_limit(2**31 - 1)

DEFAULT_OUT = DATA / "tcr_repertoire_cluster80"
MIN_ID = 0.80
COVERAGE = 0.80


def _stamp(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _done(path: Path) -> bool:
    return path.is_file()


def _write_lines(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for item in items:
            handle.write(item)
            handle.write("\n")


def _read_lines(path: Path) -> list[str]:
    with path.open() as handle:
        return [line.strip() for line in handle if line.strip()]


def _hist(seqs: list[str]) -> dict[str, int]:
    counts: Counter[int] = Counter(len(s) for s in seqs)
    return {str(k): counts[k] for k in sorted(counts)}


def stage_prepare(work: Path, force: bool) -> dict:
    stamp = work / "prepare.done.json"
    if _done(stamp) and not force:
        print(f"[prepare] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    blocked, blocked_parts, blocked_prov = build_blocked()
    print(f"[prepare] blocklist {len(blocked):,}  {blocked_parts}", flush=True)

    stats = Counter()
    cores: list[str] = []
    seen: set[str] = set()
    for src in (SRC_TRAIN, SRC_VAL):
        if not src.is_file():
            print(f"[prepare] MISSING {src}", flush=True)
            continue
        with src.open() as handle:
            for line in handle:
                stats["read"] += 1
                seq = normalize_sequence(line.strip())
                if not is_valid_protein_sequence(seq):
                    stats["drop_invalid"] += 1
                    continue
                core = cdr3_core(seq, has_anchors=True)
                if not core or len(core) < 4:
                    stats["drop_short"] += 1
                    continue
                if core in blocked:
                    stats["drop_benchmark"] += 1
                    continue
                if core in seen:
                    stats["drop_dup"] += 1
                    continue
                seen.add(core)
                cores.append(core)
    stats["pool"] = len(cores)
    print(f"[prepare] unique clean cores {len(cores):,}  {dict(stats)}", flush=True)

    _write_lines(work / "clean_cores.txt", cores)
    _write_lines(work / "blocked_cores.txt", sorted(blocked))
    payload = {
        "counts": dict(stats),
        "blocklist_cores": blocked_parts,
        "blocklist_cores_union": len(blocked),
        "blocklist_provenance": blocked_prov,
        "n_clean": len(cores),
    }
    _stamp(stamp, payload)
    return payload


def stage_cluster(work: Path, threads: int, force: bool) -> dict:
    stamp = work / "cluster.done.json"
    if _done(stamp) and not force and (work / "cluster_cluster.tsv").is_file():
        print(f"[cluster] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    if not MMSEQS_BIN.is_file():
        raise FileNotFoundError(f"mmseqs not found: {MMSEQS_BIN}")

    fasta = work / "input.fasta"
    print("[cluster] writing fasta (clean + blocked) …", flush=True)
    n_fasta = 0
    seen: set[str] = set()
    with fasta.open("w") as handle:
        for path in (work / "clean_cores.txt", work / "blocked_cores.txt"):
            for core in _read_lines(path):
                if core in seen:
                    continue
                seen.add(core)
                handle.write(">")
                handle.write(core)
                handle.write("\n")
                handle.write(core)
                handle.write("\n")
                n_fasta += 1

    scratch = work / "mmseqs_tmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    prefix = work / "cluster"
    for leftover in (
        work / "cluster_cluster.tsv",
        work / "cluster_rep_seq.fasta",
        work / "cluster_all_seqs.fasta",
    ):
        if leftover.is_file():
            leftover.unlink()

    command = [
        str(MMSEQS_BIN),
        "easy-linclust",
        str(fasta),
        str(prefix),
        str(scratch),
        "--min-seq-id",
        str(MIN_ID),
        "-c",
        str(COVERAGE),
        "--cov-mode",
        "0",
        "--cluster-mode",
        "1",
        "--threads",
        str(max(1, threads)),
        "--remove-tmp-files",
    ]
    print("[cluster] " + " ".join(command), flush=True)
    completed = subprocess.run(command, check=True, text=True)
    version = subprocess.run(
        [str(MMSEQS_BIN), "version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "n_fasta": n_fasta,
        "command": command,
        "mmseqs_version": version,
        "cluster_tsv": str(work / "cluster_cluster.tsv"),
        "returncode": completed.returncode,
    }
    _stamp(stamp, payload)
    return payload


def stage_pick(work: Path, *, seed: int, valid_frac: float, holdout_frac: float,
               max_train: int, force: bool) -> dict:
    stamp = work / "pick.done.json"
    if _done(stamp) and not force and (work / "train_reps.txt").is_file():
        print(f"[pick] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    blocked = set(_read_lines(work / "blocked_cores.txt"))
    tsv = work / "cluster_cluster.tsv"
    if not tsv.is_file():
        raise FileNotFoundError(tsv)

    dirty: set[str] = set()
    reps: set[str] = set()
    n_edges = 0
    print("[pick] streaming cluster.tsv …", flush=True)
    with tsv.open() as handle:
        for line in handle:
            n_edges += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0], parts[1]
            reps.add(rep)
            if member in blocked or rep in blocked:
                dirty.add(rep)

    clean = sorted(rep for rep in reps if rep not in dirty)
    print(
        f"[pick] clusters={len(reps):,} dirty={len(dirty):,} clean={len(clean):,} "
        f"edges={n_edges:,}",
        flush=True,
    )

    denom = 10**6
    valid_cut = int(valid_frac * denom)
    holdout_cut = valid_cut + int(holdout_frac * denom)
    splits: dict[str, list[str]] = {"train": [], "valid": [], "holdout": []}
    for rep in clean:
        bucket = _stable_rank(seed, "cluster80", rep) % denom
        if bucket < valid_cut:
            splits["valid"].append(rep)
        elif bucket < holdout_cut:
            splits["holdout"].append(rep)
        else:
            splits["train"].append(rep)

    before_cap = {k: len(v) for k, v in splits.items()}
    rng = random.Random(seed)
    if max_train and len(splits["train"]) > max_train:
        splits["train"] = rng.sample(splits["train"], max_train)
        splits["train"].sort()
    else:
        splits["train"].sort()
    splits["valid"].sort()
    splits["holdout"].sort()

    for name, seqs in splits.items():
        _write_lines(work / f"{name}_reps.txt", seqs)

    payload = {
        "n_clusters": len(reps),
        "n_dirty_clusters": len(dirty),
        "n_clean_clusters": len(clean),
        "n_edges": n_edges,
        "split_before_cap": before_cap,
        "split_after_cap": {k: len(v) for k, v in splits.items()},
        "max_train": max_train,
        "length_hist_train": _hist(splits["train"]),
        "length_hist_valid": _hist(splits["valid"]),
        "length_hist_holdout": _hist(splits["holdout"]),
        "min_seq_id": MIN_ID,
        "coverage": COVERAGE,
        "pick": "mmseqs_representative_one_per_clean_cluster",
    }
    _stamp(stamp, payload)
    return payload


def stage_write(out_root: Path, work: Path, prepare: dict, cluster: dict, pick: dict) -> dict:
    dataset = out_root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    splits = {
        name: _read_lines(work / f"{name}_reps.txt")
        for name in ("train", "valid", "holdout")
    }
    blocked = set(_read_lines(work / "blocked_cores.txt"))

    def write(name: str, cores: list[str]) -> None:
        with (dataset / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
            writer.writeheader()
            for core in cores:
                writer.writerow({
                    **{c: "" for c in UNIFIED_COLUMNS},
                    "record_id": record_id("tcr_repertoire_cluster80", core),
                    "source": "tcr_repertoire",
                    "fv_source": "tcrdesign2026_cdr3_cluster80",
                    "tier": "D",
                    "task_type": "tcr",
                    "relation": "unknown",
                    "sequence_scope": "cdr3b_only",
                    "cdr3b": core,
                    "provenance": PROVENANCE + "+cluster80_linclust",
                })

    for name, cores in splits.items():
        write(name, cores)

    residual = sum(1 for core in splits["train"] if core in blocked)
    overlaps = {
        f"{a}_vs_{b}": len(set(splits[a]) & set(splits[b]))
        for a, b in (("train", "valid"), ("train", "holdout"), ("valid", "holdout"))
    }
    report = {
        "schema_version": "tcr_repertoire.cluster80.v1",
        "PASS": residual == 0 and all(v == 0 for v in overlaps.values()),
        "residual_blocked_in_train": residual,
        "split_overlap": overlaps,
        "split_disjoint_PASS": all(v == 0 for v in overlaps.values()),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "decontam_mode": "exact+cluster_0.80_0.80_then_one_rep",
        "provenance": PROVENANCE + "+cluster80_linclust",
        "dataset_dir": str(dataset),
        "does_not_replace": str(DATA / "tcr_repertoire"),
        "prepare": prepare,
        "cluster": {k: v for k, v in cluster.items() if k != "command"} | {
            "command": cluster.get("command")
        },
        "pick": pick,
    }
    (dataset / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps({k: report[k] for k in (
        "PASS", "split_counts", "residual_blocked_in_train", "split_disjoint_PASS",
        "dataset_dir",
    )}, indent=2), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--max-train", type=int, default=2_000_000)
    ap.add_argument("--valid-frac", type=float, default=0.005)
    ap.add_argument("--holdout-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stage", choices=("all", "prepare", "cluster", "pick", "write"),
                    default="all")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    work = out_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    prepare = cluster = pick = {}
    if args.stage in ("all", "prepare"):
        prepare = stage_prepare(work, args.force)
    if args.stage in ("all", "cluster"):
        cluster = stage_cluster(work, args.threads, args.force)
    if args.stage in ("all", "pick"):
        pick = stage_pick(
            work,
            seed=args.seed,
            valid_frac=args.valid_frac,
            holdout_frac=args.holdout_frac,
            max_train=args.max_train,
            force=args.force,
        )
    if args.stage in ("all", "write"):
        if not prepare:
            prepare = json.loads((work / "prepare.done.json").read_text())
        if not cluster:
            cluster = json.loads((work / "cluster.done.json").read_text())
        if not pick:
            pick = json.loads((work / "pick.done.json").read_text())
        stage_write(out_root, work, prepare, cluster, pick)


if __name__ == "__main__":
    main()
