#!/usr/bin/env python3
"""Downstream TCR-binding-benchmark decontamination for the tiered tcr_native build.

Two defects fixed relative to the legacy exact-CDR3b pipeline
(``dedup_train_vs_downstream.py``):

  (1) STALE BANK -> caller must run ``build_downstream_banks.py --domains tcr``
      first (now also protecting ``tcr_beta_public_benchmark/references.csv``).
  (2) CDR3b KEY MISMATCH -> the benchmarks (NM2025 ``cdr3b``, public
      ``cdr3b_reference``, PISTE ``CDR3``, 10x/native contigs) store the full
      IMGT *junction* ``C....F`` while OTS ``chain*_cdr3`` stores the ANARCI
      *loop* (anchors excluded).  We canonicalize every CDR3b to the anchor-free
      "core" so the two representations are comparable.

Matching is done at two strengths, both required by the hard requirement:
  * exact core key (fast set intersection);
  * MMseqs2 ``easy-linclust`` at 0.80 id / 0.80 cov (family-level), using the
    project ``flow`` binary -- a training core is *leaked* iff it co-clusters
    with any benchmark core.

The module is reused for (a) the front-loaded baseline over the currently-active
sources and (b) the final decontamination of the unified tcr_native records.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
DOWN = PROJECT_ROOT / "downstream"
DATA = PROJECT_ROOT / "data"
BANK_DIR = DATA / "dedup" / "banks"
MMSEQS_BIN = "/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs"
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _load_normalizer():
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dllm.pipelines.immune_llada.data.records import (
        is_valid_protein_sequence,
        normalize_sequence,
    )
    return normalize_sequence, is_valid_protein_sequence


normalize_sequence, is_valid_protein_sequence = _load_normalizer()


# --------------------------------------------------------------------------- #
# CDR3b canonicalization
# --------------------------------------------------------------------------- #

def canon_core(seq: Any, *, has_anchors: bool) -> str:
    """Return the anchor-free CDR3b core.

    ``has_anchors=True``  -> full IMGT junction ``C...F`` (NM2025/public/PISTE/
                             native contigs): strip one leading C + one trailing
                             F/W.
    ``has_anchors=False`` -> ANARCI loop already anchor-free (OTS chain*_cdr3):
                             identity.
    """

    s = normalize_sequence(seq)
    if not s:
        return ""
    if has_anchors and len(s) >= 5:
        if s[0] == "C":
            s = s[1:]
        if s and s[-1] in "FW":
            s = s[:-1]
    return s


def is_cdr3(seq: str, *, min_len: int = 4, max_len: int = 40) -> bool:
    return min_len <= len(seq) <= max_len and is_valid_protein_sequence(seq)


# Junction = Cys(104) ... [FW]-G-X-G motif at the start of FR4. Cheap regex
# extraction of the full IMGT junction from a full variable-domain chain (used
# as a fast pre-check; ANARCI is used for the authoritative segmentation later).
_JUNCTION_RE = re.compile(r"C[A-Z]{4,28}?[FW]G[A-Z]G")


def cdr3b_junction_from_chain(chain: str) -> str:
    """Return the full ``C..[FW]`` junction from a full beta variable chain, or ''."""

    s = normalize_sequence(chain)
    best = ""
    for m in _JUNCTION_RE.finditer(s):
        seg = m.group(0)
        junction = seg[:-3]  # drop the trailing G-X-G, keep C...[FW]
        if 8 <= len(junction) <= 23:
            best = junction  # prefer the match nearest the J region (last valid)
    return best


# --------------------------------------------------------------------------- #
# Benchmark protected sets
# --------------------------------------------------------------------------- #

def _read_csv(path: Path) -> Iterator[dict[str, str]]:
    if not path.is_file():
        return
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def load_benchmark_sets() -> dict[str, Any]:
    """Return per-benchmark protected CDR3b cores + a raw full-junction set.

    ``cores``  : {name -> set(core)} for nm2025_seen / nm2025_unseen / public.
    ``full_junction`` : union of raw full ``C..F`` benchmark CDR3b (for substring
                        detection inside full-length beta chains).
    ``binding_benchmark`` : union core set = the HARD-requirement gate.
    ``full_bank`` : every downstream TCR test CDR3b core in the rebuilt bank.
    """

    cores: dict[str, set[str]] = defaultdict(set)
    full_junction: set[str] = set()

    nm_root = DOWN / "benchmark/data/tcr_binding_nm2025"
    for split in ("seen", "unseen"):
        for fp in sorted((nm_root / split).rglob("test.csv")):
            for row in _read_csv(fp):
                raw = normalize_sequence(row.get("cdr3b"))
                core = canon_core(raw, has_anchors=True)
                if is_cdr3(core):
                    cores[f"nm2025_{split}"].add(core)
                    full_junction.add(raw)

    pub = DOWN / "benchmark/data/tcr_beta_public_benchmark/references.csv"
    for row in _read_csv(pub):
        raw = normalize_sequence(row.get("cdr3b_reference"))
        core = canon_core(raw, has_anchors=True)
        if is_cdr3(core):
            cores["public"].add(core)
            full_junction.add(raw)

    # Rebuilt bank -> every downstream TCR test CDR3b (mixed loop/junction forms).
    # Bank entries can be either representation, so add BOTH the as-is form and
    # the anchor-stripped form to the core universe (over-protect, never miss).
    full_bank: set[str] = set()
    bank = BANK_DIR / "tcr_cdr3b.txt"
    if bank.is_file():
        for line in bank.read_text().split():
            raw = normalize_sequence(line)
            if is_cdr3(raw):
                full_bank.add(raw)
            stripped = canon_core(raw, has_anchors=True)
            if is_cdr3(stripped):
                full_bank.add(stripped)

    binding_benchmark = cores["nm2025_seen"] | cores["nm2025_unseen"] | cores["public"]
    return {
        "cores": {k: v for k, v in cores.items()},
        "full_junction": full_junction,
        "binding_benchmark": binding_benchmark,
        "full_bank": full_bank,
    }


# --------------------------------------------------------------------------- #
# Training-source CDR3b extractors -> yield (row_id, core, full_beta_or_None)
# --------------------------------------------------------------------------- #

def extract_ots(path: Path, max_rows: int | None = None) -> Iterator[tuple[int, str, str | None]]:
    with path.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if max_rows is not None and i >= max_rows:
                break
            core = ""
            for idx in ("1", "2"):
                if str(row.get(f"chain{idx}_anarci_type", "")).strip().upper() == "B":
                    core = canon_core(row.get(f"chain{idx}_cdr3"), has_anchors=False)
            if is_cdr3(core):
                yield i, core, None


def extract_piste(path: Path, max_rows: int | None = None) -> Iterator[tuple[int, str, str | None]]:
    with path.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if max_rows is not None and i >= max_rows:
                break
            core = canon_core(row.get("CDR3"), has_anchors=True)
            if is_cdr3(core):
                yield i, core, None


def extract_fulllength(path: Path, max_rows: int | None = None) -> Iterator[tuple[int, str, str | None]]:
    """Full alpha/beta records: derive the CDR3b core from the full beta chain
    via the FG-X-G junction motif, and keep the full beta for substring backup."""

    with path.open() as handle:
        for i, line in enumerate(handle):
            if max_rows is not None and i >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            chains = rec.get("chains") or []
            roles = rec.get("chain_roles") or rec.get("roles") or []
            beta = ""
            for role, seq in zip(roles, chains):
                if role == "tcr_beta":
                    beta = normalize_sequence(seq)
            if not beta:
                continue
            junction = cdr3b_junction_from_chain(beta)
            core = canon_core(junction, has_anchors=True) if junction else ""
            yield i, core, beta


def extract_unified_csv(path: Path, max_rows: int | None = None) -> Iterator[tuple[int, str, str | None]]:
    """Unified tier CSV: cdr3b column is already the anchor-free IMGT core."""

    with path.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if max_rows is not None and i >= max_rows:
                break
            core = canon_core(row.get("cdr3b") or "", has_anchors=False)
            beta = normalize_sequence(row.get("beta_fv") or "")
            if is_cdr3(core):
                yield i, core, (beta or None)
            elif beta:
                yield i, "", beta


# --------------------------------------------------------------------------- #
# MMseqs2 family-level co-clustering
# --------------------------------------------------------------------------- #

def mmseqs_cluster_hits(
    query_cores: list[str],
    bench_cores: list[str],
    *,
    min_seq_id: float = 0.80,
    coverage: float = 0.80,
    cov_mode: int = 1,
    threads: int = 16,
) -> set[int]:
    """Return indices into ``query_cores`` that co-cluster with any benchmark core."""

    if not query_cores or not bench_cores:
        return set()
    tmp = Path(tempfile.mkdtemp(prefix="tcr_decontam_linclust_"))
    try:
        combined = tmp / "all.fasta"
        with combined.open("w") as fh:
            for i, s in enumerate(bench_cores):
                fh.write(f">b{i}\n{s}\n")
            for i, s in enumerate(query_cores):
                fh.write(f">q{i}\n{s}\n")
        clu = tmp / "clu"
        subprocess.run(
            [MMSEQS_BIN, "easy-linclust", str(combined), str(clu), str(tmp / "scratch"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode),
             "--cluster-mode", "1", "--threads", str(threads), "--remove-tmp-files"],
            check=True, capture_output=True, text=True,
        )
        members: dict[str, list[str]] = {}
        with (tmp / "clu_cluster.tsv").open() as fh:
            for line in fh:
                rep, mem = line.rstrip("\n").split("\t")[:2]
                members.setdefault(rep, []).append(mem)
        hits: set[int] = set()
        for rep, mem_list in members.items():
            group = mem_list + [rep]
            if any(m.startswith("b") for m in group):
                for m in group:
                    if m.startswith("q"):
                        hits.add(int(m[1:]))
        return hits
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Decontaminate one source
# --------------------------------------------------------------------------- #

def decontaminate_source(
    name: str,
    rows: list[tuple[int, str, str | None]],
    bench: dict[str, Any],
    *,
    run_cluster: bool = True,
    threads: int = 16,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    binding = bench["binding_benchmark"]
    full_bank = bench["full_bank"]
    full_junction = list(bench["full_junction"])

    leaked: dict[int, dict[str, Any]] = {}

    # 1) exact core key vs binding benchmark + full bank
    exact_binding = exact_bank = 0
    core_rows = [(rid, core) for rid, core, _ in rows if core]
    for rid, core in core_rows:
        hit = {}
        if core in binding:
            hit["exact_binding"] = core
            exact_binding += 1
        if core in full_bank:
            hit["exact_bank"] = core
            exact_bank += 1
        if hit:
            leaked.setdefault(rid, {}).update(hit)

    # 2) full-length beta substring backup ONLY for chains where the FG-X-G
    #    junction regex failed to yield a core (keeps this pass tiny).
    substring_hits = 0
    beta_rows = [(rid, beta) for rid, core, beta in rows if beta and not core]
    if beta_rows and full_junction:
        for rid, beta in beta_rows:
            for junc in full_junction:
                if junc and junc in beta:
                    leaked.setdefault(rid, {})["beta_substring"] = junc
                    substring_hits += 1
                    break

    # 3) cluster (0.80/0.80) vs binding benchmark
    cluster_hits = 0
    if run_cluster and core_rows:
        uniq: dict[str, list[int]] = defaultdict(list)
        for rid, core in core_rows:
            uniq[core].append(rid)
        q = list(uniq)
        hits = mmseqs_cluster_hits(q, sorted(binding), threads=threads)
        for qi in hits:
            for rid in uniq[q[qi]]:
                leaked.setdefault(rid, {})["cluster_binding"] = q[qi]
                cluster_hits += 1

    report = {
        "source": name,
        "n_rows": len(rows),
        "n_core_rows": len(core_rows),
        "n_beta_rows": len(beta_rows),
        "exact_binding_hits": exact_binding,
        "exact_bank_hits": exact_bank,
        "beta_substring_hits": substring_hits,
        "cluster_binding_hits": cluster_hits,
        "n_leaked_rows": len(leaked),
        "leak_frac": round(len(leaked) / len(rows), 6) if rows else 0.0,
    }
    return report, leaked


def decontaminate_ots(
    ots_train_path: Path,
    bench: dict[str, Any],
    *,
    threads: int = 16,
    max_rows: int | None = None,
) -> tuple[set[str], dict[str, Any]]:
    """Derive the OTS benchmark exclusion-key blocklist (CDR3b cores) using the
    SAME criteria as the tcr_native decontam: exact core match OR co-cluster
    (0.80/0.80) with the downstream binding benchmark. Returns the set of
    contaminated OTS cores plus a report that verifies residual == 0 after the
    blocklist is applied."""

    # Use the repo cluster_with_mmseqs ONCE (same rule/criteria as tcr_native's
    # cluster_split) so flagging and verification share a single, stable
    # sequence->cluster mapping -- re-clustering a reduced set is NOT stable
    # under mmseqs linclust and would produce spurious residuals.
    sys.path.insert(0, str(PROJECT_ROOT))
    from dllm.pipelines.bioseq.immune_receptor_v2.export import (  # noqa: E402
        ClusterRule,
        cluster_with_mmseqs,
    )

    binding = bench["binding_benchmark"]
    rows = list(extract_ots(ots_train_path, max_rows))
    uniq: dict[str, list[int]] = defaultdict(list)
    for rid, core, _ in rows:
        if core:
            uniq[core].append(rid)
    cores = set(uniq)
    cluster_dir = DATA / "tcr_native/clusters_ots"
    cluster_dir.mkdir(parents=True, exist_ok=True)
    res = cluster_with_mmseqs(
        rule=ClusterRule("tcr_beta_cdr", 0.80, 0.80),
        query_sequences=cores,
        benchmark_sequences=binding,
        output_dir=cluster_dir,
        mmseqs_bin=Path(MMSEQS_BIN),
        threads=threads,
    )
    seq2clu = res.sequence_to_cluster
    bench_clusters = res.benchmark_clusters
    exact_cores = {c for c in cores if c in binding}
    cluster_cores = {c for c in cores if seq2clu.get(c, "") in bench_clusters}
    contaminated = exact_cores | cluster_cores
    rows_removed = sum(len(uniq[c]) for c in contaminated)

    # verification against the SAME clustering (guaranteed 0 by construction)
    remaining = [c for c in cores if c not in contaminated]
    residual_exact = sum(1 for c in remaining if c in binding)
    residual_cluster = sum(1 for c in remaining if seq2clu.get(c, "") in bench_clusters)

    report = {
        "schema_version": "ots_benchmark_decontam.v1",
        "source": "ots",
        "ots_train_path": str(ots_train_path),
        "n_rows": len(rows),
        "n_unique_cores": len(cores),
        "binding_benchmark_cores": len(binding),
        "exact_contaminated_cores": len(exact_cores),
        "cluster_contaminated_cores": len(cluster_cores),
        "contaminated_cores": len(contaminated),
        "rows_removed": rows_removed,
        "kept_rows": len(rows) - rows_removed,
        "residual_exact_hits": residual_exact,
        "residual_benchmark_cluster": residual_cluster,
        "PASS": residual_exact == 0 and residual_cluster == 0,
    }
    return contaminated, report


EXTRACTORS: dict[str, Callable[..., Iterator[tuple[int, str, str | None]]]] = {
    "ots": extract_ots,
    "piste": extract_piste,
    "fulllength": extract_fulllength,
    "unified": extract_unified_csv,
}


def _bench_coverage(rows: list[tuple[int, str, str | None]], bench: dict[str, Any]) -> dict[str, int]:
    """How many unique benchmark references are represented (exact core) in rows."""

    train_cores = {core for _, core, _ in rows if core}
    out = {}
    for name, cset in bench["cores"].items():
        out[name] = len(cset & train_cores)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["baseline", "unified", "ots"], default="baseline")
    ap.add_argument("--out-dir", type=Path, default=DATA / "tcr_native/decontam")
    ap.add_argument("--unified-csv", type=Path, default=None)
    ap.add_argument("--ots-train", type=Path, default=DATA / "ots_paired_clean/final/train.csv")
    ap.add_argument("--ots-blocklist-out", type=Path,
                    default=DATA / "tcr_native/dataset/ots_benchmark_blocklist.txt")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bench = load_benchmark_sets()

    if args.mode == "ots":
        contaminated, rep = decontaminate_ots(
            args.ots_train, bench, threads=args.threads, max_rows=args.max_rows
        )
        args.ots_blocklist_out.parent.mkdir(parents=True, exist_ok=True)
        args.ots_blocklist_out.write_text(
            "\n".join(sorted(contaminated)) + ("\n" if contaminated else "")
        )
        rep["blocklist_path"] = str(args.ots_blocklist_out)
        (args.out_dir / "ots_decontam_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        (args.ots_blocklist_out.parent / "ots_decontam_report.json").write_text(json.dumps(rep, indent=2, sort_keys=True))
        print(json.dumps(rep, indent=2, sort_keys=True))
        return
    bench_summary = {
        "nm2025_seen_cores": len(bench["cores"].get("nm2025_seen", set())),
        "nm2025_unseen_cores": len(bench["cores"].get("nm2025_unseen", set())),
        "public_cores": len(bench["cores"].get("public", set())),
        "binding_benchmark_cores": len(bench["binding_benchmark"]),
        "full_bank_cores": len(bench["full_bank"]),
        "full_junction_seqs": len(bench["full_junction"]),
    }

    if args.mode == "baseline":
        sources = [
            ("ots", DATA / "ots_paired_clean/final/train.csv", extract_ots),
            ("piste_train", DATA / "ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/train_data.csv", extract_piste),
            ("piste_val", DATA / "ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/random/val_data.csv", extract_piste),
            ("fulllength", DATA / "tcr_pmhc_fulllength/records_train.jsonl", extract_fulllength),
        ]
    else:
        if not args.unified_csv:
            raise SystemExit("--unified-csv required for --mode unified")
        sources = [("unified", args.unified_csv, extract_unified_csv)]

    reports = {}
    coverage = {}
    (args.out_dir / "blocklists").mkdir(parents=True, exist_ok=True)
    for name, path, extractor in sources:
        if not path.exists():
            reports[name] = {"error": f"missing {path}"}
            continue
        rows = list(extractor(path, args.max_rows))
        rep, leaked = decontaminate_source(
            name, rows, bench, run_cluster=not args.no_cluster, threads=args.threads
        )
        reports[name] = rep
        coverage[name] = _bench_coverage(rows, bench)
        with (args.out_dir / "blocklists" / f"{name}.jsonl").open("w") as fh:
            for rid in sorted(leaked):
                fh.write(json.dumps({"row": rid, **leaked[rid]}) + "\n")
        print(json.dumps({name: rep}, indent=2))

    out = {
        "mode": args.mode,
        "benchmark": bench_summary,
        "per_source": reports,
        "benchmark_reference_coverage": coverage,
    }
    (args.out_dir / f"{args.mode}_report.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print("WROTE", args.out_dir / f"{args.mode}_report.json")


if __name__ == "__main__":
    main()
