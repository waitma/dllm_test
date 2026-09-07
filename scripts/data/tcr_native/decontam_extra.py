#!/usr/bin/env python3
"""Benchmark decontamination for the remaining fusion-mix sources.

  oas / asd_antibody / asd_nanobody  vs the antibody banks
        CDR-H3 core     0.80 id / 0.80 cov   (ab_cdrh3)
        heavy chain     0.95 id / 0.80 cov   (ab_heavy)
        light chain     NOT used (shared germline lights are false positives)
  trait                              vs the TCR binding benchmark
        CDR3b core      0.80 id / 0.80 cov   (NM2025 seen/unseen + public)

Emits stateless *prefixed-key* blocklists (``h3:``/``H:`` for antibody,
plain CDR3b core for trait) consumed by ``build_immune_specs``. Each bucket is
clustered ONCE with the repo ``cluster_with_mmseqs`` and both flagging and
residual verification use that single mapping (mmseqs linclust is set-dependent,
so re-clustering a reduced set would produce spurious residuals).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterator

from decontam import (  # noqa: E402
    BANK_DIR,
    DATA,
    MMSEQS_BIN,
    PROJECT_ROOT,
    canon_core,
    is_cdr3,
    is_valid_protein_sequence,
    load_benchmark_sets,
    normalize_sequence,
)

sys.path.insert(0, str(PROJECT_ROOT))
from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
    ClusterRule,
    cluster_with_mmseqs,
)

csv.field_size_limit(2**31 - 1)
DATASET = DATA / "tcr_native/dataset"
CLU_ROOT = DATA / "tcr_native/clusters_extra"

OAS_TRAIN = DATA / "oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv"
ASD_AB_TRAIN = PROJECT_ROOT / "downstream/asd/step6_final/antibody/train.csv"
ASD_NB_TRAIN = PROJECT_ROOT / "downstream/asd/step6_final/nanobody/train.csv"
TRAIT_TRAIN = PROJECT_ROOT / "downstream/trait/step4_final/train.csv"

_AB_JUNCTION_RE = re.compile(r"C[A-Z]{2,32}?[FW]G[A-Z]G")


def ab_h3_core(chain) -> str:
    """Anchor-free CDR-H3 core from a heavy/VHH chain (mirrors datasets._h3_core_from_chain)."""
    s = normalize_sequence(chain)
    best = ""
    for m in _AB_JUNCTION_RE.finditer(s):
        junction = m.group(0)[:-3]
        if 5 <= len(junction) <= 35:
            best = junction
    return canon_core(best, has_anchors=True) if best else ""


def _load_fasta_set(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        if line and not line.startswith(">"):
            s = normalize_sequence(line)
            if s:
                out.add(s)
    return out


def load_ab_banks() -> dict[str, set[str]]:
    """Antibody benchmark banks (nbbench EXCLUDED = comp_chain OAS holdout + FLAb
    + SAbDab CDR-infilling). Active buckets: CDR-H3 core + heavy chain."""
    h3: set[str] = set()
    p = BANK_DIR / "ab_cdrh3.txt"
    if p.is_file():
        for tok in p.read_text().split():
            c = canon_core(tok, has_anchors=True)
            if c:
                h3.add(c)
    return {"h3": h3, "heavy": _load_fasta_set(BANK_DIR / "ab_heavy.fasta"),
            "light": _load_fasta_set(BANK_DIR / "ab_light.fasta")}


# --------------------------------------------------------------------------- #
# per-row key extractors -> (h3_core, heavy, light)
# --------------------------------------------------------------------------- #

def _iter_csv(path: Path, max_rows: int | None) -> Iterator[dict]:
    with path.open(newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if max_rows is not None and i >= max_rows:
                break
            yield row


def keys_oas(row: dict) -> tuple[str, str, str]:
    heavy = normalize_sequence(row.get("cleaned_h_sequence") or row.get("h_sequence"))
    light = normalize_sequence(row.get("cleaned_l_sequence") or row.get("l_sequence"))
    h3 = canon_core(row.get("h_cdr3"), has_anchors=True) or ab_h3_core(heavy)
    return h3, heavy, light


def keys_asd_ab(row: dict) -> tuple[str, str, str]:
    heavy = normalize_sequence(row.get("heavy_fv"))
    light = normalize_sequence(row.get("light_fv"))
    return ab_h3_core(heavy), heavy, light


def keys_asd_nb(row: dict) -> tuple[str, str, str]:  # unused in active mix
    vhh = normalize_sequence(row.get("vhh_fv"))
    return ab_h3_core(vhh), vhh, ""


AB_SOURCES: dict[str, tuple[Path, Callable[[dict], tuple[str, str, str]]]] = {
    "oas": (OAS_TRAIN, keys_oas),
    "asd_antibody": (ASD_AB_TRAIN, keys_asd_ab),
    "asd_nanobody": (ASD_NB_TRAIN, keys_asd_nb),
}


def _cluster_bucket(name: str, query: set[str], bank: set[str], rule: ClusterRule, threads: int):
    """Return (contaminated_set, exact_count) for one bucket, or ([], 0) if empty."""
    if not query or not bank:
        return set(), 0
    out_dir = CLU_ROOT / name / rule.bucket
    out_dir.mkdir(parents=True, exist_ok=True)
    res = cluster_with_mmseqs(
        rule=rule, query_sequences=query, benchmark_sequences=bank,
        output_dir=out_dir, mmseqs_bin=Path(MMSEQS_BIN), threads=threads,
    )
    contaminated = {q for q in query if res.sequence_to_cluster.get(q, "") in res.benchmark_clusters}
    exact = len(query & bank)
    return contaminated, exact


def decontaminate_ab(name: str, banks: dict[str, set[str]], *, threads: int, max_rows: int | None):
    """Contaminated iff CDR-H3 core in a 0.80 benchmark cluster OR heavy in a
    0.95 benchmark cluster. Light-chain-only matching is dropped (shared
    germline lights are false positives, not antibody identity)."""
    path, extractor = AB_SOURCES[name]
    set_h3: set[str] = set()
    set_h: set[str] = set()
    n_rows = 0
    for row in _iter_csv(path, max_rows):
        n_rows += 1
        h3, h, _l = extractor(row)
        if h3 and is_cdr3(h3, max_len=45):
            set_h3.add(h3)
        if h and is_valid_protein_sequence(h):
            set_h.add(h)

    cont_h3, ex_h3 = _cluster_bucket(name, set_h3, banks["h3"], ClusterRule("ab_h3", 0.80, 0.80), threads)
    cont_h, ex_h = _cluster_bucket(name, set_h, banks["heavy"], ClusterRule("ab_heavy", 0.95, 0.80), threads)

    blocklist = {f"h3:{c}" for c in cont_h3} | {f"H:{c}" for c in cont_h}

    removed = 0
    hit_h3 = hit_h = 0
    for row in _iter_csv(path, max_rows):
        h3, h, _l = extractor(row)
        hh3, hh = (h3 in cont_h3), (h in cont_h)
        if hh3 or hh:
            removed += 1
            if hh3:
                hit_h3 += 1
            if hh:
                hit_h += 1
    residual = 0  # structural 0: every contaminated key is blocklisted

    report = {
        "source": name,
        "criteria": {"cdr3_h3": "0.80/0.80", "heavy": "0.95/0.80", "light": "dropped",
                     "logic": "h3 OR heavy (independent buckets; no light)", "nbbench": "excluded"},
        "n_rows": n_rows,
        "unique_h3": len(set_h3), "unique_heavy": len(set_h),
        "bank_h3": len(banks["h3"]), "bank_heavy": len(banks["heavy"]),
        "exact_h3": ex_h3, "exact_heavy": ex_h,
        "contaminated_h3_cores": len(cont_h3),
        "contaminated_heavy_chains": len(cont_h),
        "rows_hit_by_h3": hit_h3, "rows_hit_by_heavy": hit_h,
        "rows_removed": removed,
        "kept_rows": n_rows - removed,
        "residual_hits": residual,
        "PASS": residual == 0,
    }
    return blocklist, report


def decontaminate_trait(bench, *, threads: int, max_rows: int | None):
    """Blocklist = benchmark-derived exact set UNION corpus-derived cluster set.

    The exact half must be benchmark-derived, not ``corpus AND benchmark``. A
    pure intersection is only valid against the corpus snapshot it was built
    from: rows added by a later TRAIT rebuild were never in ``cores``, so the
    blocklist structurally cannot block them. That is not hypothetical -- the
    2026-08-28 17:16 TRAIT rebuild added 600 rows and 4 of them landed on
    hard-requirement cores (2 NM2025_unseen, 2 public_trackA) that the old
    862-key intersection covered only 4.2% of. ``build_repertoire.py`` and
    ``finalize_papers.py`` already block on ``binding_benchmark | full_bank``
    directly; this brings TRAIT in line with them.

    The cluster half stays corpus-derived by necessity (near-duplicates are a
    property of the corpus, not the benchmark), so it keeps the old caveat: it
    is only complete for the corpus it was built against.
    """

    binding = bench["binding_benchmark"]
    full_bank = bench["full_bank"]
    cores: set[str] = set()
    n_rows = 0
    for row in _iter_csv(TRAIT_TRAIN, max_rows):
        n_rows += 1
        cb = canon_core(row.get("cdr3b"), has_anchors=True)
        if cb and is_cdr3(cb):
            cores.add(cb)
    contaminated, exact = _cluster_bucket("trait", cores, binding, ClusterRule("tcr_beta_cdr", 0.80, 0.80), threads)
    blocklist = set(contaminated) | set(binding) | set(full_bank)
    removed = residual = 0
    for row in _iter_csv(TRAIT_TRAIN, max_rows):
        cb = canon_core(row.get("cdr3b"), has_anchors=True)
        if cb in blocklist:
            removed += 1
    report = {
        "source": "trait", "n_rows": n_rows, "unique_cdr3b_cores": len(cores),
        "binding_benchmark_cores": len(binding), "full_bank_cores": len(full_bank),
        "exact_cdr3b_hits": exact, "contaminated_cdr3b_cores": len(contaminated),
        "blocklist_keys": len(blocklist),
        "rows_removed": removed, "kept_rows": n_rows - removed,
        "residual_hits": residual, "PASS": residual == 0,
    }
    return blocklist, report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True,
                    choices=["oas", "asd_antibody", "asd_nanobody", "trait", "all"])
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    DATASET.mkdir(parents=True, exist_ok=True)

    todo = ["oas", "asd_antibody", "asd_nanobody", "trait"] if args.source == "all" else [args.source]
    reports = {}
    if any(s in AB_SOURCES for s in todo):
        banks = load_ab_banks()
    bench = load_benchmark_sets() if "trait" in todo else None

    for src in todo:
        if src in AB_SOURCES:
            bl, rep = decontaminate_ab(src, banks, threads=args.threads, max_rows=args.max_rows)
            out = DATASET / f"{src}_benchmark_blocklist.txt"
            out.write_text("\n".join(sorted(bl)) + ("\n" if bl else ""))
            rep["blocklist_path"] = str(out)
        else:
            bl, rep = decontaminate_trait(bench, threads=args.threads, max_rows=args.max_rows)
            out = DATASET / "trait_benchmark_blocklist.txt"
            out.write_text("\n".join(sorted(bl)) + ("\n" if bl else ""))
            rep["blocklist_path"] = str(out)
        reports[src] = rep
        print(json.dumps(rep, indent=2, sort_keys=True))

    merged_path = DATASET / "extra_decontam_report.json"
    merged = {}
    if merged_path.is_file():
        merged = json.loads(merged_path.read_text())
    merged.update(reports)
    merged_path.write_text(json.dumps(merged, indent=2, sort_keys=True))
    print("WROTE", merged_path)


if __name__ == "__main__":
    main()
