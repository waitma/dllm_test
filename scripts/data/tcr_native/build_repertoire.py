#!/usr/bin/env python3
"""Build the unlabeled single-chain CDR3b repertoire corpus (``tcr_repertoire``).

Source: TcrDesign-2026 ``pretrain/bCDR3_train.csv`` -- 32.2M bare CDR3b junctions
with no epitope and no MHC. These render as the ``tcr_single`` grammar layout,
which currently has **zero** training records (OTS renders as ``tcr_pair``), and
which is exactly what the T4 Setting-A unconditional benchmark decodes with.

Why this needs its own builder rather than ingest_papers.py: the unified tier
schema keys on ``epitope_seq`` and ``tcr_native_row_to_record`` drops any row
without one, so unlabeled repertoire cannot go through that path.

Decontamination is the whole point of this script. The raw file overlaps the
Setting-A distribution reference badly -- 4,326 of the 10,516 OTS-holdout CDR3b
cores (41.1%) appear in it -- and Setting A's JSD / novelty / NN metrics are all
computed against that holdout. Ingesting it unfiltered would collapse novelty
(currently 1.000) and improve JSD for the wrong reason.

Anchor conventions differ per source and getting this wrong silently corrupts
every count (it did once already in this corpus's history):
  * ``bCDR3_train.csv``      full junction, 100% ``C..[FW]``  -> has_anchors=True
  * OTS ``chain*_cdr3``      ANARCI loop, already anchor-free -> has_anchors=False
  * unified ``cdr3b``        anchor-free core                 -> has_anchors=False
  * TRAIT ``cdr3b``          full junction                    -> has_anchors=True

Pipeline:
  1. read + normalize + validate, dedup on the anchor-free core
  2. exact decontamination against every benchmark core set (cheap, on all 32M)
  3. group-aware hash split by core
  4. subsample train to ``--max-train`` (full 31.6M would take 31% of the
     generated-token budget and crush every other layout; 2M gives ~2.8%)
  5. MMseqs2 0.80/0.80 cluster decontamination on the sampled train only --
     clustering 32M sequences is not worth it when only the sample ships
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))

from common import (  # noqa: E402
    DATA,
    PROJECT_ROOT,
    UNIFIED_COLUMNS,
    cdr3_core,
    is_valid_protein_sequence,
    normalize_sequence,
    record_id,
)

csv.field_size_limit(2**31 - 1)

SRC_TRAIN = DATA / "tcr_papers/raw/tcrdesign2026/pretrain/bCDR3_train.csv"
SRC_VAL = DATA / "tcr_papers/raw/tcrdesign2026/pretrain/bCDR3_val.csv"
CORPUS = DATA / "tcr_repertoire"
OTS_FINAL = DATA / "ots_paired_clean/final"
T4_BLOCKLIST = DATA / "tcr_native/dataset/t4_refbinder_blocklist.txt"
T2T3_BLOCKLIST = DATA / "tcr_native/dataset/t2t3_eval_blocklist.txt"
OTS_BLOCKLIST = DATA / "tcr_native/dataset/ots_benchmark_blocklist.txt"
MMSEQS_BIN = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs")

PROVENANCE = "tcrdesign2026:pretrain/bCDR3_train.csv"


def _stable_rank(*parts) -> int:
    return int(hashlib.sha1(("|".join(str(p) for p in parts)).encode()).hexdigest(), 16)


def _core_only(blocklist: Path) -> set[str]:
    """Project a ``(cdr3b_core|epitope)`` blocklist down to bare cores.

    Unlabeled repertoire rows have no epitope, so the composite key cannot
    match. Dropping the epitope half is deliberately over-aggressive: it removes
    a benchmark CDR3b regardless of which epitope it was an answer for.
    """
    cores: set[str] = set()
    if not blocklist.is_file():
        return cores
    for line in blocklist.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        cores.add(line.split("|", 1)[0])
    return cores


def _ots_cores(name: str) -> set[str]:
    path = OTS_FINAL / f"{name}.csv"
    cores: set[str] = set()
    if not path.is_file():
        return cores
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            # OTS chain1 is beta, chain2 alpha; chain*_cdr3 is already anchor-free.
            core = cdr3_core(row.get("chain1_cdr3"), has_anchors=False)
            if core:
                cores.add(core)
    return cores


def _binding_benchmark_cores() -> set[str]:
    from decontam import load_benchmark_sets  # noqa: E402

    bench = load_benchmark_sets()
    return set(bench["binding_benchmark"]) | set(bench["full_bank"])


def _provenance(paths: dict[str, Path]) -> dict[str, dict]:
    """Record mtime + content hash of every blocklist this build depended on.

    Without this the corpus silently goes stale: the 2026-08-28 build ran against
    a 67,013-key T4 blocklist, the blocklist was rebuilt to 68,846 keys 45
    minutes later, and 3 reference binders were left sitting in train.csv with
    the report still claiming PASS. Consumers can now compare these mtimes
    against the live files and refuse to train on a stale corpus.
    """
    out: dict[str, dict] = {}
    for name, path in paths.items():
        if not path.is_file():
            out[name] = {"path": str(path), "exists": False}
            continue
        out[name] = {
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "sha1": hashlib.sha1(path.read_bytes()).hexdigest()[:16],
        }
    return out


def build_blocked() -> tuple[set[str], dict[str, int], dict[str, dict]]:
    """Every CDR3b core that must not enter training, with counts + provenance."""
    parts = {
        "ots_holdout": _ots_cores("holdout"),
        "ots_valid": _ots_cores("valid"),
        "t4_refbinder": _core_only(T4_BLOCKLIST),
        "t2t3_eval": _core_only(T2T3_BLOCKLIST),
        "ots_benchmark": _core_only(OTS_BLOCKLIST),
        "binding_benchmark": _binding_benchmark_cores(),
    }
    blocked: set[str] = set()
    for s in parts.values():
        blocked |= s
    prov = _provenance({
        "t4_refbinder": T4_BLOCKLIST,
        "t2t3_eval": T2T3_BLOCKLIST,
        "ots_benchmark": OTS_BLOCKLIST,
        "ots_holdout": OTS_FINAL / "holdout.csv",
        "ots_valid": OTS_FINAL / "valid.csv",
    })
    return blocked, {k: len(v) for k, v in parts.items()}, prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-train", type=int, default=2_000_000,
                    help="rows written to train.csv. 0 = no cap (not advised: the "
                         "full pool takes ~31%% of the generated-token budget)")
    ap.add_argument("--valid-frac", type=float, default=0.005)
    ap.add_argument("--holdout-frac", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-cluster", action="store_true")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out-root", default=str(CORPUS))
    ap.add_argument("--write-pool", action="store_true",
                    help="also write the full decontaminated pool (~1GB)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    dataset_dir = out_root / "dataset"
    report: dict = {"schema_version": "tcr_repertoire.v1", "provenance": PROVENANCE}

    blocked, blocked_parts, blocked_prov = build_blocked()
    report["blocklist_cores"] = blocked_parts
    report["blocklist_cores_union"] = len(blocked)
    report["blocklist_provenance"] = blocked_prov
    print(f"[blocklist] {len(blocked):,} cores  {blocked_parts}", flush=True)

    # -- 1/2. read, dedup on core, exact decontam --------------------------- #
    stats = Counter()
    pool: dict[str, str] = {}  # core -> full junction (as shipped)
    for src in (SRC_TRAIN, SRC_VAL):
        if not src.is_file():
            print(f"[read] MISSING {src}", flush=True)
            continue
        with src.open() as fh:
            for line in fh:
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
                if core in pool:
                    stats["drop_dup"] += 1
                    continue
                pool[core] = seq
    stats["pool"] = len(pool)
    report["counts"] = dict(stats)
    print(f"[pool] {len(pool):,} unique clean cores  {dict(stats)}", flush=True)

    # -- 3. group-aware hash split (one core == one clonotype group) -------- #
    denom = 10**6
    valid_cut = int(args.valid_frac * denom)
    holdout_cut = valid_cut + int(args.holdout_frac * denom)
    splits: dict[str, list[str]] = {"train": [], "valid": [], "holdout": []}
    for core in pool:
        bucket = _stable_rank(args.seed, core) % denom
        if bucket < valid_cut:
            splits["valid"].append(core)
        elif bucket < holdout_cut:
            splits["holdout"].append(core)
        else:
            splits["train"].append(core)
    report["split_pool_counts"] = {k: len(v) for k, v in splits.items()}

    # -- 4. subsample train -------------------------------------------------- #
    rng = random.Random(args.seed)
    if args.max_train and len(splits["train"]) > args.max_train:
        splits["train"] = rng.sample(splits["train"], args.max_train)
    report["train_after_cap"] = len(splits["train"])

    # -- 5. cluster decontam on the shipped train only ----------------------- #
    cluster_dropped = 0
    if not args.no_cluster:
        sys.path.insert(0, str(PROJECT_ROOT))
        from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
            ClusterRule,
            cluster_with_mmseqs,
        )

        cores = set(splits["train"])
        work = out_root / "clusters_binding"
        work.mkdir(parents=True, exist_ok=True)
        res = cluster_with_mmseqs(
            rule=ClusterRule("tcr_beta_cdr", 0.80, 0.80),
            query_sequences=cores,
            benchmark_sequences=blocked,
            output_dir=work,
            mmseqs_bin=MMSEQS_BIN,
            threads=args.threads,
        )
        bad = {c for c in cores
               if res.sequence_to_cluster.get(c, "") in res.benchmark_clusters}
        splits["train"] = [c for c in splits["train"] if c not in bad]
        cluster_dropped = len(bad)
    report["dropped_cluster"] = cluster_dropped
    report["decontam_mode"] = "exact" if args.no_cluster else "exact+cluster_0.80_0.80"

    # -- write --------------------------------------------------------------- #
    dataset_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, cores: list[str]) -> None:
        cores = sorted(cores)
        with (dataset_dir / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=UNIFIED_COLUMNS)
            w.writeheader()
            for core in cores:
                w.writerow({
                    **{c: "" for c in UNIFIED_COLUMNS},
                    "record_id": record_id("tcr_repertoire", core),
                    "source": "tcr_repertoire",
                    "fv_source": "tcrdesign2026_cdr3",
                    "tier": "D",
                    "task_type": "tcr",
                    "relation": "unknown",
                    "sequence_scope": "cdr3b_only",
                    # Stored anchor-free, matching the unified cdr3b convention.
                    "cdr3b": core,
                    "provenance": PROVENANCE,
                })

    for name, cores in splits.items():
        write(name, cores)

    residual = sum(1 for c in splits["train"] if c in blocked)
    report["residual_blocked_in_train"] = residual
    report["PASS"] = residual == 0
    report["split_counts"] = {k: len(v) for k, v in splits.items()}
    overlaps = {
        f"{a}_vs_{b}": len(set(splits[a]) & set(splits[b]))
        for a, b in (("train", "valid"), ("train", "holdout"), ("valid", "holdout"))
    }
    report["split_overlap"] = overlaps
    report["split_disjoint_PASS"] = all(v == 0 for v in overlaps.values())
    report["dataset_dir"] = str(dataset_dir)

    if args.write_pool:
        with (out_root / "pool.txt").open("w") as fh:
            for core in sorted(pool):
                fh.write(core + "\n")
        report["pool_file"] = str(out_root / "pool.txt")

    (dataset_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
