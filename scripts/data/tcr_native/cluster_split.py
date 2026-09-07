#!/usr/bin/env python3
"""Global CDR3 clustering + group-disjoint split + final benchmark decontamination
for the unified tcr_native corpus (Ophiuchus-Ab style).

Steps (run ONCE on the full union so native rows are included):
  1. MMseqs CDR3b cluster at 0.80 id / 0.80 cov (repo ``cluster_with_mmseqs`` with
     the ``tcr_beta_cdr`` rule); CDR3a cluster for alpha-only rows.
  2. FINAL decontamination (HARD requirement): drop every tcr_native row whose
     CDR3b core is a benchmark test sequence (exact) OR co-clusters (0.80) with
     one -> ``residual_benchmark_cluster == 0`` and exact hits == 0.
  3. union-find group-disjoint 90/5/5 split (seed=42) via repo
     ``build_split_manifest``; a shared cluster->split map is emitted.
  4. replaces_trait blocklist: TRAIT rows superseded by a full-length-Fv
     tcr_native row (same CDR3b cluster + epitope) -> stateless (core|epitope)
     exclusion keys for the TRAIT loader.

Outputs under ``data/tcr_native/dataset/``: train/valid/holdout.csv,
cluster_split_map.json, decontam_report.json, replaces_trait_blocklist.txt.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

sys.path.insert(0, str(Path(__file__).parent))
from common import DATA, PROJECT_ROOT, UNIFIED_COLUMNS, cdr3_core, normalize_sequence  # noqa: E402
from decontam import load_benchmark_sets  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    ClusterRule,
    cluster_with_mmseqs,
)
from dllm.pipelines.bioseq.immune_receptor_v2.splits import (  # noqa: E402
    SplitProtocol,
    build_split_manifest,
)

MMSEQS_BIN = Path("/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs")
UNIFIED = DATA / "tcr_native/unified/all_unified.csv"
TRAIT_TRAIN = PROJECT_ROOT / "downstream/trait/step4_final/train.csv"
OUT = DATA / "tcr_native/dataset"
CLUSTER_DIR = DATA / "tcr_native/clusters"
FV_SCOPE = "fv"


@dataclass
class _SplitRec:
    record_id: str
    groups: Mapping[str, str]
    biological_key: str = ""


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--unified", type=Path, default=UNIFIED)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--with-trait", action="store_true", default=True,
                    help="Include TRAIT CDR3b in clustering for replaces_trait.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CLUSTER_DIR.mkdir(parents=True, exist_ok=True)

    rows = _read_csv(args.unified)
    for i, r in enumerate(rows):
        r["_idx"] = i
        r["_cb"] = cdr3_core(r.get("cdr3b") or "", has_anchors=False)
        r["_ca"] = cdr3_core(r.get("cdr3a") or "", has_anchors=False)

    trait_rows: list[dict] = []
    if args.with_trait and TRAIT_TRAIN.is_file():
        for r in _read_csv(TRAIT_TRAIN):
            # TRAIT cdr3b is the full IMGT junction (C..F); strip anchors so its
            # core matches the native/loader core space AND the replaces_trait
            # blocklist keys match datasets.trait_exclusion_key (which strips too).
            r["_cb"] = cdr3_core(r.get("cdr3b") or "", has_anchors=True)
            trait_rows.append(r)

    bench = load_benchmark_sets()
    bench_beta = bench["binding_benchmark"]  # NM2025 seen/unseen + public (cores)

    # ------------------------------------------------------------------ #
    # 1) CDR3b + CDR3a clustering (0.80/0.80) including benchmark cores
    # ------------------------------------------------------------------ #
    beta_query = {r["_cb"] for r in rows if r["_cb"]} | {r["_cb"] for r in trait_rows if r["_cb"]}
    alpha_query = {r["_ca"] for r in rows if r["_ca"]}

    beta_res = cluster_with_mmseqs(
        rule=ClusterRule("tcr_beta_cdr", 0.80, 0.80),
        query_sequences=beta_query,
        benchmark_sequences=bench_beta,
        output_dir=CLUSTER_DIR,
        mmseqs_bin=MMSEQS_BIN,
        threads=args.threads,
    )
    seq2clu_b = beta_res.sequence_to_cluster
    bench_clusters_b = beta_res.benchmark_clusters

    seq2clu_a: dict[str, str] = {}
    if alpha_query:
        alpha_res = cluster_with_mmseqs(
            rule=ClusterRule("tcr_alpha_cdr", 0.80, 0.80),
            query_sequences=alpha_query,
            benchmark_sequences=set(),
            output_dir=CLUSTER_DIR,
            mmseqs_bin=MMSEQS_BIN,
            threads=args.threads,
        )
        seq2clu_a = alpha_res.sequence_to_cluster

    def group_cluster(r: dict) -> str:
        if r["_cb"] and r["_cb"] in seq2clu_b:
            return seq2clu_b[r["_cb"]]
        if r["_ca"] and r["_ca"] in seq2clu_a:
            return seq2clu_a[r["_ca"]]
        return f"singleton_{r['_idx']}"

    # ------------------------------------------------------------------ #
    # 2) FINAL decontamination (delete benchmark exact + cluster hits)
    # ------------------------------------------------------------------ #
    exact_before = cluster_before = 0
    kept: list[dict] = []
    blocked: list[dict] = []
    for r in rows:
        cb = r["_cb"]
        is_exact = bool(cb) and cb in bench_beta
        clu = seq2clu_b.get(cb, "")
        is_cluster = bool(clu) and clu in bench_clusters_b
        if is_exact:
            exact_before += 1
        if is_cluster:
            cluster_before += 1
        if is_exact or is_cluster:
            blocked.append({
                "record_id": r.get("record_id", ""),
                "source": r.get("source", ""),
                "cdr3b_core": cb,
                "reason": "exact" if is_exact else "cluster",
            })
        else:
            kept.append(r)

    # residual audit on kept
    residual_exact = sum(1 for r in kept if r["_cb"] and r["_cb"] in bench_beta)
    residual_cluster = sum(
        1 for r in kept if seq2clu_b.get(r["_cb"], "") in bench_clusters_b and r["_cb"]
    )

    # ------------------------------------------------------------------ #
    # 3) union-find group-disjoint 90/5/5 split (seed=42)
    # ------------------------------------------------------------------ #
    split_recs = [
        _SplitRec(
            record_id=r.get("record_id") or f"row_{r['_idx']}",
            groups={"tcr_cluster": group_cluster(r)},
            biological_key=r.get("record_id") or f"row_{r['_idx']}",
        )
        for r in kept
    ]
    id2row = {sr.record_id: r for sr, r in zip(split_recs, kept)}
    protocol = SplitProtocol(
        "tcr_native_receptor_cluster_disjoint",
        group_fields=("tcr_cluster",),
        ratios=(0.9, 0.05, 0.05),
        split_names=("train", "valid", "holdout"),
        seed=args.seed,
        description="CDR3b(0.80)/CDR3a receptor-cluster-disjoint 90/5/5 split.",
    )
    manifest = build_split_manifest(split_recs, protocol)
    assign = {a["record_id"]: a["split"] for a in manifest["assignments"]}

    # cluster -> split map (shared)
    cluster_to_split: dict[str, str] = {}
    for sr in split_recs:
        cluster_to_split[sr.groups["tcr_cluster"]] = assign[sr.record_id]

    # write splits
    split_rows: dict[str, list[dict]] = {"train": [], "valid": [], "holdout": []}
    for sr in split_recs:
        split_rows[assign[sr.record_id]].append(id2row[sr.record_id])
    for split, srows in split_rows.items():
        with (OUT / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=UNIFIED_COLUMNS)
            writer.writeheader()
            for r in srows:
                writer.writerow({c: r.get(c, "") for c in UNIFIED_COLUMNS})

    (OUT / "cluster_split_map.json").write_text(
        json.dumps({"protocol": "tcr_native_receptor_cluster_disjoint",
                    "seed": args.seed, "cluster_to_split": cluster_to_split},
                   sort_keys=True)
    )

    # ------------------------------------------------------------------ #
    # 4) replaces_trait blocklist (TRAIT rows superseded by native full Fv)
    # ------------------------------------------------------------------ #
    # (beta_cluster, epitope) covered by a full-Fv tcr_native row
    fv_cluster_epitope: set[tuple[str, str]] = set()
    for r in kept:
        if r.get("sequence_scope") == FV_SCOPE and r["_cb"]:
            clu = seq2clu_b.get(r["_cb"], "")
            ep = normalize_sequence(r.get("epitope_seq"))
            if clu and ep:
                fv_cluster_epitope.add((clu, ep))
    replaces_keys: set[str] = set()
    replaced_trait_rows = 0
    for r in trait_rows:
        cb = r["_cb"]
        if not cb:
            continue
        clu = seq2clu_b.get(cb, "")
        ep = normalize_sequence(r.get("epitope_seq"))
        if clu and (clu, ep) in fv_cluster_epitope:
            replaces_keys.add(f"{cb}|{ep}")
            replaced_trait_rows += 1
    (OUT / "replaces_trait_blocklist.txt").write_text(
        "\n".join(sorted(replaces_keys)) + ("\n" if replaces_keys else "")
    )

    # ------------------------------------------------------------------ #
    # reports
    # ------------------------------------------------------------------ #
    by_source = Counter(r["source"] for r in kept)
    by_tier = Counter(r["tier"] for r in kept)
    by_scope = Counter(r["sequence_scope"] for r in kept)
    split_counts = {k: len(v) for k, v in split_rows.items()}
    # Embed the OTS-clean state (separate loader-level decontamination) so the
    # single decontam_report.json covers both tcr_native and the oas+ots base.
    ots_report = None
    ots_report_path = OUT / "ots_decontam_report.json"
    if ots_report_path.is_file():
        ots_report = json.loads(ots_report_path.read_text())
    extra_report = None
    extra_report_path = OUT / "extra_decontam_report.json"
    if extra_report_path.is_file():
        extra_report = json.loads(extra_report_path.read_text())
    decontam = {
        "schema_version": "tcr_native_decontam.v1",
        "input_rows": len(rows),
        "benchmark": {
            "binding_benchmark_cores": len(bench_beta),
            "benchmark_beta_clusters": len(bench_clusters_b),
        },
        "removed": {
            "exact_hits": exact_before,
            "cluster_hits": cluster_before,
            "total_removed_rows": len(blocked),
        },
        "kept_rows": len(kept),
        "residual_exact_hits": residual_exact,
        "residual_benchmark_cluster": residual_cluster,
        "PASS": residual_exact == 0 and residual_cluster == 0,
        "active_mix": ["oas", "ots", "tcr_native", "trait", "asd_antibody"],
        "antibody_bank": "nbbench excluded (comp_chain OAS holdout + FLAb + SAbDab CDR-infilling only)",
        "antibody_criteria": "CDR-H3 0.80/0.80 | heavy 0.95/0.80 (light-only matching dropped)",
        "tcr_criteria": "CDR3b 0.80/0.80",
        "asd_nanobody_status": "dropped from active mix; blocklist/report retained on disk but unused",
        "ots_benchmark_decontamination": ots_report,
        "ots_PASS": (ots_report or {}).get("PASS"),
        "extra_source_decontamination": extra_report,
        "all_sources_PASS": (
            residual_exact == 0
            and residual_cluster == 0
            and bool(ots_report) and ots_report.get("PASS") is True
            and bool(extra_report)
            # active mix only: asd_nanobody is dropped, so it is NOT required
            and all((extra_report.get(s) or {}).get("PASS") is True
                    for s in ("oas", "asd_antibody", "trait"))
        ),
        "per_source_rows_removed": {
            "tcr_native_build": len(blocked),
            "ots": (ots_report or {}).get("rows_removed"),
            **{k: (extra_report or {}).get(k, {}).get("rows_removed")
               for k in ("oas", "asd_antibody", "trait")},
        },
        "split_counts": split_counts,
        "split_audit_passed": manifest["audit"]["passed"],
        "by_source": dict(by_source),
        "by_tier": dict(by_tier),
        "by_scope": dict(by_scope),
        "replaces_trait": {
            "fv_cluster_epitope_keys": len(fv_cluster_epitope),
            "trait_rows_superseded": replaced_trait_rows,
            "blocklist_keys": len(replaces_keys),
        },
    }
    (OUT / "decontam_report.json").write_text(json.dumps(decontam, indent=2, sort_keys=True))
    with (OUT / "benchmark_blocklist.jsonl").open("w") as handle:
        for b in blocked:
            handle.write(json.dumps(b) + "\n")
    print(json.dumps(decontam, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
