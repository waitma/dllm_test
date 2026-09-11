#!/usr/bin/env python3
"""Cluster-then-pick unlabeled CDR3β by length-bucketed Hamming on full junctions.

Does NOT overwrite ``data/tcr_repertoire/`` or the core-based
``data/tcr_repertoire_hamming80/``.

Similarity is computed on the **full junction** (C…[FW] kept). Equal-length
junctions are neighbors when Hamming identity ≥ 0.80, i.e. substitutions
d ≤ floor(0.20 × L). A typical 15-mer junction allows d=3 (exactly 80%).
Sequences with identity < 0.80 stay separate. Hobohm-1 (lex order) picks
representatives so HD=1 chains do not collapse into one mega-cluster.

Resume: each stage stamps ``work/*.done.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
_DLLM_ROOT = HERE.parents[3]
if str(_DLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(_DLLM_ROOT))

from build_repertoire import (  # noqa: E402
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

DEFAULT_OUT = DATA / "tcr_repertoire_junc80"
IDENTITY = 0.80


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


def hamming_radius(length: int, identity: float = IDENTITY) -> int:
    """Max substitutions so (L-d)/L ≥ identity on the clustered string.

    Junction L=15 → d=3 (12/15 = 0.80). Identity below 0.80 does not merge.
    """
    if length <= 0:
        return 0
    return int(length * (1.0 - identity) + 1e-12)


def _blocked_junctions(cores: list[str]) -> list[str]:
    """Expand core blocklist to junction form for same-length Hamming."""
    out: list[str] = []
    seen: set[str] = set()
    for core in cores:
        if not core:
            continue
        for junc in (core, "C" + core + "F", "C" + core + "W"):
            if junc not in seen:
                seen.add(junc)
                out.append(junc)
    return out


def _junction_core(seq: str) -> str:
    if len(seq) >= 5 and seq[0] == "C" and seq[-1] in "FW":
        return seq[1:-1]
    return cdr3_core(seq, has_anchors=True) or seq


def _deletion_key(seq: str, positions: tuple[int, ...]) -> str:
    if not positions:
        return seq
    parts: list[str] = []
    prev = 0
    for pos in positions:
        parts.append(seq[prev:pos])
        prev = pos + 1
    parts.append(seq[prev:])
    return "".join(parts)


def _iter_key_hashes(seq: str, radius: int):
    if radius <= 0:
        yield hash((0, seq))
        return
    for positions in combinations(range(len(seq)), radius):
        yield hash((positions, _deletion_key(seq, positions)))


def _hits_index(seq: str, radius: int, index: set[int]) -> bool:
    if not index:
        return False
    for key in _iter_key_hashes(seq, radius):
        if key in index:
            return True
    return False


def _add_index(seq: str, radius: int, index: set[int]) -> None:
    for key in _iter_key_hashes(seq, radius):
        index.add(key)


def hobohm1_equal_length(clean: list[str], blocked: list[str], radius: int) -> list[str]:
    """Greedy packing: keep lex-smallest cores that are >radius from blocked and from kept.

    Sharing a radius-deletion key ⇔ Hamming ≤ radius on equal-length strings.
    Single-linkage is intentionally avoided: a path of HD=1 pairs would otherwise
    collapse a public-clonotype chain into one mega-cluster.
    """
    blocked_index: set[int] = set()
    for seq in blocked:
        _add_index(seq, radius, blocked_index)
    kept: list[str] = []
    kept_index: set[int] = set()
    n_block_near = 0
    n_absorbed = 0
    t0 = time.time()
    for i, seq in enumerate(sorted(clean)):
        if _hits_index(seq, radius, blocked_index):
            n_block_near += 1
            continue
        if _hits_index(seq, radius, kept_index):
            n_absorbed += 1
            continue
        kept.append(seq)
        _add_index(seq, radius, kept_index)
        if (i + 1) % 1_000_000 == 0:
            print(
                f"    scanned={i+1:,}/{len(clean):,} kept={len(kept):,} "
                f"near_block={n_block_near:,} absorbed={n_absorbed:,} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
    print(
        f"    done kept={len(kept):,} near_block={n_block_near:,} "
        f"absorbed={n_absorbed:,} elapsed={time.time() - t0:.1f}s",
        flush=True,
    )
    return kept


def _self_check() -> None:
    # Hobohm-1, lex order: absorb HD≤d to an already-kept seq, do not chain.
    kept = hobohm1_equal_length(
        ["AAAAAA", "AAAAAB", "AAAABB", "BBBBBB"], blocked=[], radius=1,
    )
    assert kept == ["AAAAAA", "AAAABB", "BBBBBB"], kept
    a, b, c = "A" * 10, "A" * 8 + "BB", "A" * 7 + "BBB"
    kept = hobohm1_equal_length([a, b, c], blocked=[], radius=2)
    assert a in kept and b not in kept and c in kept, kept
    kept = hobohm1_equal_length([a, b], blocked=[a], radius=2)
    assert kept == [], kept
    assert hamming_radius(13) == 2
    assert hamming_radius(15) == 3
    assert hamming_radius(10) == 2
    # junction 15-mer: d=3 merges exactly 80%, d=4 (below 80%) stays split
    j15 = "C" + "A" * 13 + "F"
    j15_d3 = "C" + "A" * 10 + "BBB" + "F"
    j15_d4 = "C" + "A" * 9 + "BBBB" + "F"
    kept = hobohm1_equal_length([j15, j15_d3, j15_d4], blocked=[], radius=3)
    assert j15 in kept and j15_d3 not in kept and j15_d4 in kept, kept
    print("[self-check] Hamming radius + Hobohm-1 OK", flush=True)


def stage_prepare(work: Path, force: bool) -> dict:
    stamp = work / "prepare.done.json"
    if _done(stamp) and not force and (work / "clean_junctions.txt").is_file():
        print(f"[prepare] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    blocked, blocked_parts, blocked_prov = build_blocked()
    print(f"[prepare] blocklist {len(blocked):,}  {blocked_parts}", flush=True)

    stats = Counter()
    clean: list[str] = []
    seen: set[str] = set()
    work.mkdir(parents=True, exist_ok=True)
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
                if len(seq) < 6:
                    stats["drop_short"] += 1
                    continue
                if seq in seen:
                    stats["drop_dup"] += 1
                    continue
                seen.add(seq)
                if _junction_core(seq) in blocked:
                    stats["drop_benchmark"] += 1
                    continue
                clean.append(seq)
    stats["pool"] = len(clean)
    print(f"[prepare] unique clean junctions {len(clean):,}  {dict(stats)}", flush=True)

    _write_lines(work / "clean_junctions.txt", clean)
    _write_lines(work / "blocked_cores.txt", sorted(blocked))
    payload = {
        "counts": dict(stats),
        "blocklist_cores": blocked_parts,
        "blocklist_cores_union": len(blocked),
        "blocklist_provenance": blocked_prov,
        "n_clean": len(clean),
        "cluster_on": "full_junction",
        "write_as": "full_junction",
        "identity": IDENTITY,
        "radius": "floor(L_junction * (1-identity)); merge iff identity>=0.80",
    }
    _stamp(stamp, payload)
    return payload


def stage_cluster(work: Path, force: bool) -> dict:
    stamp = work / "cluster.done.json"
    kept_path = work / "kept_junctions.txt"
    if _done(stamp) and not force and kept_path.is_file():
        print(f"[cluster] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    blocked_cores = _read_lines(work / "blocked_cores.txt")
    blocked = _blocked_junctions(blocked_cores)
    clean = _read_lines(work / "clean_junctions.txt")
    print(
        f"[cluster] Hobohm-1 on junctions clean={len(clean):,} "
        f"blocked_cores={len(blocked_cores):,} blocked_junc={len(blocked):,}",
        flush=True,
    )

    clean_by_len: dict[int, list[str]] = defaultdict(list)
    blocked_by_len: dict[int, list[str]] = defaultdict(list)
    for seq in clean:
        clean_by_len[len(seq)].append(seq)
    for seq in blocked:
        blocked_by_len[len(seq)].append(seq)

    kept: list[str] = []
    length_stats = []
    for length in sorted(set(clean_by_len) | set(blocked_by_len)):
        radius = hamming_radius(length)
        bucket = clean_by_len.get(length, [])
        near = [s for s in blocked_by_len.get(length, []) if len(s) == length]
        print(f"[cluster] L={length} clean={len(bucket):,} blocked={len(near):,} d={radius}", flush=True)
        t0 = time.time()
        bucket_kept = hobohm1_equal_length(bucket, near, radius)
        elapsed = time.time() - t0
        kept.extend(bucket_kept)
        print(f"[cluster] L={length} kept={len(bucket_kept):,} elapsed={elapsed:.1f}s", flush=True)
        length_stats.append({
            "length": length,
            "n_clean": len(bucket),
            "n_blocked": len(near),
            "radius": radius,
            "n_kept": len(bucket_kept),
            "elapsed_s": round(elapsed, 2),
        })

    kept.sort()
    _write_lines(kept_path, kept)
    payload = {
        "kept_junctions": str(kept_path),
        "method": "length_bucket_hamming_hobohm1_junction",
        "identity": IDENTITY,
        "n_clean": len(clean),
        "n_blocked_cores": len(blocked_cores),
        "n_blocked_junctions": len(blocked),
        "n_kept": len(kept),
        "kept_frac": round(len(kept) / max(len(clean), 1), 4),
        "by_length": length_stats,
    }
    _stamp(stamp, payload)
    print(f"[cluster] kept={len(kept):,} / clean={len(clean):,} ({100 * len(kept) / max(len(clean), 1):.1f}%)", flush=True)
    return payload


def stage_pick(work: Path, *, seed: int, valid_frac: float, holdout_frac: float,
               max_train: int, force: bool) -> dict:
    stamp = work / "pick.done.json"
    if _done(stamp) and not force and (work / "train_reps.txt").is_file():
        print(f"[pick] resume {stamp}", flush=True)
        return json.loads(stamp.read_text())

    blocked_cores = set(_read_lines(work / "blocked_cores.txt"))
    clean = [j for j in _read_lines(work / "kept_junctions.txt")
             if _junction_core(j) not in blocked_cores]
    print(f"[pick] kept clean junction reps={len(clean):,}", flush=True)

    denom = 10**6
    valid_cut = int(valid_frac * denom)
    holdout_cut = valid_cut + int(holdout_frac * denom)
    splits: dict[str, list[str]] = {"train": [], "valid": [], "holdout": []}
    for rep in clean:
        bucket = _stable_rank(seed, "junc80", rep) % denom
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
        "n_clean_reps": len(clean),
        "split_before_cap": before_cap,
        "split_after_cap": {k: len(v) for k, v in splits.items()},
        "max_train": max_train,
        "length_hist_train": _hist(splits["train"]),
        "length_hist_valid": _hist(splits["valid"]),
        "length_hist_holdout": _hist(splits["holdout"]),
        "identity": IDENTITY,
        "pick": "hobohm1_lexmin_then_hash_split",
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
    blocked_cores = set(_read_lines(work / "blocked_cores.txt"))
    missing = 0

    def write(name: str, seqs: list[str]) -> None:
        with (dataset / f"{name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
            writer.writeheader()
            for junc in seqs:
                writer.writerow({
                    **{c: "" for c in UNIFIED_COLUMNS},
                    "record_id": record_id("tcr_repertoire_junc80", junc),
                    "source": "tcr_repertoire",
                    "fv_source": "tcrdesign2026_cdr3_junc80",
                    "tier": "D",
                    "task_type": "tcr",
                    "relation": "unknown",
                    "sequence_scope": "cdr3b_only",
                    "cdr3b": junc,
                    "provenance": PROVENANCE + "+junc80",
                })

    for name, seqs in splits.items():
        write(name, seqs)

    residual = sum(1 for junc in splits["train"] if _junction_core(junc) in blocked_cores)
    overlaps = {
        f"{a}_vs_{b}": len(set(splits[a]) & set(splits[b]))
        for a, b in (("train", "valid"), ("train", "holdout"), ("valid", "holdout"))
    }
    report = {
        "schema_version": "tcr_repertoire.junc80.v1",
        "PASS": residual == 0 and all(v == 0 for v in overlaps.values()) and missing == 0,
        "residual_blocked_in_train": residual,
        "missing_junction_fallback": missing,
        "split_overlap": overlaps,
        "split_disjoint_PASS": all(v == 0 for v in overlaps.values()),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "decontam_mode": "exact_core+hamming_junc80_hobohm1",
        "cdr3b_stored_as": "full_junction",
        "cluster_on": "full_junction",
        "provenance": PROVENANCE + "+junc80",
        "dataset_dir": str(dataset),
        "does_not_replace": str(DATA / "tcr_repertoire"),
        "prepare": prepare,
        "cluster": cluster,
        "pick": pick,
    }
    (dataset / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in (
        "PASS", "split_counts", "residual_blocked_in_train",
        "missing_junction_fallback", "split_disjoint_PASS", "dataset_dir",
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
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--stage", choices=("all", "prepare", "cluster", "pick", "write"),
                    default="all")
    args = ap.parse_args()

    if not args.skip_self_check:
        _self_check()

    out_root = Path(args.out_root)
    work = out_root / "work"
    work.mkdir(parents=True, exist_ok=True)

    prepare = cluster = pick = {}
    if args.stage in ("all", "prepare"):
        prepare = stage_prepare(work, args.force)
    if args.stage in ("all", "cluster"):
        cluster = stage_cluster(work, args.force)
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
