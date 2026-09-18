#!/usr/bin/env python3
"""Deduplicate a training source against downstream-test banks.

Run ``build_downstream_banks.py`` first. This script scans ONE training source,
extracts the biological keys that define leakage for that source's domain, and
matches them against the banks with domain-appropriate rules::

    antibody / nanobody : CDR-H3 loop cluster (>= --cdrh3-id) OR full-length
                          heavy/VHH identity (>= --ab-full-id)
    tcr                 : CDR3-beta clonotype exact match (single-linkage)
    ppi                 : global protein identity (>= --ppi-id) on either chain

Exact-key matches (TCR CDR3b, and the fast CDR-H3 exact pre-pass) use in-memory
set intersection. Similarity thresholds (CDR-H3 < 100%, full-length, PPI global)
use MMseqs2 ``search`` with the small test bank as *target* and training seqs as
*query*, mirroring ``data/pipeline/step2_decontaminate.py``.

Outputs (under ``data/dedup/``):
  * ``reports/<source>.json``       -- overlap counts per key + rule.
  * ``blocklists/<source>.jsonl``   -- one row per leaked training example with
                                       the offending keys (consumed by the
                                       shard-filter step).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


def _load_normalizer():
    path = PROJECT_ROOT / "dllm/pipelines/immune_llada/data/records.py"
    name = "_dedup_records2"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclass processing needs the module registered
    spec.loader.exec_module(mod)
    return mod.normalize_sequence, mod.is_valid_protein_sequence


normalize_sequence, is_valid_protein_sequence = _load_normalizer()

DATA = PROJECT_ROOT / "data"
BANK_DIR = DATA / "dedup" / "banks"
REPORT_DIR = DATA / "dedup" / "reports"
BLOCK_DIR = DATA / "dedup" / "blocklists"
MMSEQS_BIN = "/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs"


def _norm(v: Any) -> str:
    return normalize_sequence(v)


# --------------------------------------------------------------------------- #
# Training-source row extractors: yield dict of keys per row.
# Each yields: {"row": int, "cdrh3": str|None, "full": [seqs], "tcr_cdr3b": str|None}
# --------------------------------------------------------------------------- #

def _iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as handle:
        yield from csv.DictReader(handle)


def extract_oas(path: Path) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(_iter_csv(path)):
        yield {
            "row": i,
            "cdrh3": _norm(row.get("h_cdr3")) or None,
            "full": [s for s in (_norm(row.get("cleaned_h_sequence")),) if s],
        }


def extract_nanobody(path: Path) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(_iter_csv(path)):
        yield {
            "row": i,
            "cdrh3": _norm(row.get("CDR3")) or None,
            "full": [s for s in (_norm(row.get("cleaned_seq")),) if s],
        }


def extract_ots(path: Path) -> Iterator[dict[str, Any]]:
    for i, row in enumerate(_iter_csv(path)):
        cdr3b = None
        for idx in ("1", "2"):
            if str(row.get(f"chain{idx}_anarci_type", "")).strip().upper() == "B":
                cdr3b = _norm(row.get(f"chain{idx}_cdr3")) or None
        yield {"row": i, "tcr_cdr3b": cdr3b}


def _iter_arrow(path: Path) -> Iterator[dict[str, Any]]:
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    for i, row in enumerate(ds):
        yield i, row


def extract_grammar_ppi(path: Path) -> Iterator[dict[str, Any]]:
    """mint_ppi / mint_actions / ppi grammar shards: chains list of protein seqs."""
    for i, row in _iter_arrow(path):
        chains = [_norm(c) for c in (row.get("chains") or [])]
        yield {"row": i, "full": [c for c in chains if c]}


def extract_grammar_tcr(path: Path) -> Iterator[dict[str, Any]]:
    """tcr / tcr_piste grammar shards: CDR3-only chains keyed by role."""
    for i, row in _iter_arrow(path):
        roles = row.get("roles") or []
        chains = row.get("chains") or []
        cdr3b = None
        for role, seq in zip(roles, chains):
            if role == "tcr_beta":
                cdr3b = _norm(seq) or None
        yield {"row": i, "tcr_cdr3b": cdr3b}


def extract_grammar_ab(path: Path) -> Iterator[dict[str, Any]]:
    """Antibody grammar shards (e.g. neutralization): full heavy/VHH chains.

    Grammar shards do NOT store CDR regions, so only full-length identity is
    available here (CDR-H3 exact/cluster passes are skipped for these sources).
    """
    for i, row in _iter_arrow(path):
        roles = row.get("roles") or []
        chains = row.get("chains") or []
        full = [
            _norm(seq)
            for role, seq in zip(roles, chains)
            if role in ("antibody_heavy", "nanobody_vhh") and _norm(seq)
        ]
        yield {"row": i, "cdrh3": None, "full": full}


SOURCES: dict[str, dict[str, Any]] = {
    "oas": {"domain": "antibody", "path": DATA / "oas_previous_clean/splits/cleaned_merged_data_step_clustered_train_oas_label.csv", "extract": extract_oas},
    "nanobody": {"domain": "antibody", "path": DATA / "nanobody_processed/step7_clean/train.csv", "extract": extract_nanobody},
    "ots": {"domain": "tcr", "path": DATA / "ots_paired_clean/final/train.csv", "extract": extract_ots},
    "tcr": {"domain": "tcr", "path": DATA / "bioseq_grammar_v1/tcr/train", "extract": extract_grammar_tcr},
    "tcr_piste": {"domain": "tcr", "path": DATA / "bioseq_grammar_v1/tcr_piste/train", "extract": extract_grammar_tcr},
    "ppi": {"domain": "ppi", "path": DATA / "bioseq_grammar_v1/ppi/train", "extract": extract_grammar_ppi},
    "mint_ppi": {"domain": "ppi", "path": DATA / "bioseq_grammar_v1/mint_ppi/train", "extract": extract_grammar_ppi},
    "mint_actions": {"domain": "ppi", "path": DATA / "bioseq_grammar_v1/mint_actions/train", "extract": extract_grammar_ppi},
    "neutralization": {"domain": "antibody", "path": DATA / "bioseq_grammar_v1/neutralization/train", "extract": extract_grammar_ab},
}


# --------------------------------------------------------------------------- #
# MMseqs2 similarity: query seqs -> hit set, targeting the (small) test bank.
# --------------------------------------------------------------------------- #

def _load_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    if not path.is_file():
        return seqs
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                seqs.append("".join(cur))
                cur = []
        else:
            cur.append(line.strip())
    if cur:
        seqs.append("".join(cur))
    return seqs


def mmseqs_query_hits(
    query_seqs: list[str],
    target_fasta: Path,
    *,
    min_seq_id: float,
    coverage: float,
    threads: int,
    cov_mode: int = 0,
) -> set[int]:
    """Return indices into ``query_seqs`` that co-cluster with any test sequence.

    Uses linear-time ``mmseqs easy-linclust`` on the union of the (small) test
    bank + the (large) training queries. A training query is *leaked* when it
    lands in a cluster that also contains a test sequence. linclust scales to
    tens of millions of sequences, unlike all-vs-all ``search`` at high -s.
    """
    targets = _load_fasta(target_fasta)
    if not query_seqs or not targets:
        return set()
    tmp = Path(tempfile.mkdtemp(prefix="dedup_linclust_"))
    try:
        combined = tmp / "all.fasta"
        with combined.open("w") as fh:
            for i, s in enumerate(targets):
                fh.write(f">t{i}\n{s}\n")
            for i, s in enumerate(query_seqs):
                fh.write(f">q{i}\n{s}\n")
        clu = tmp / "clu"
        subprocess.run(
            [MMSEQS_BIN, "easy-linclust", str(combined), str(clu), str(tmp / "scratch"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", str(cov_mode),
             "--threads", str(threads)],
            check=True, capture_output=True,
        )
        # clu_cluster.tsv: <rep_id>\t<member_id>, grouped by representative.
        tsv = tmp / "clu_cluster.tsv"
        members: dict[str, list[str]] = {}
        with tsv.open() as fh:
            for line in fh:
                rep, mem = line.rstrip("\n").split("\t")
                members.setdefault(rep, []).append(mem)
        hits: set[int] = set()
        for rep, mem_list in members.items():
            group = mem_list + [rep]
            if any(m.startswith("t") for m in group):
                for m in group:
                    if m.startswith("q"):
                        try:
                            hits.add(int(m[1:]))
                        except ValueError:
                            pass
        return hits
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #

def run_source(name: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = SOURCES[name]
    domain = cfg["domain"]
    path = cfg["path"]
    if not path.exists():
        raise FileNotFoundError(f"training source not found: {path}")

    # Load banks
    ab_cdrh3 = set((BANK_DIR / "ab_cdrh3.txt").read_text().split()) if (BANK_DIR / "ab_cdrh3.txt").is_file() else set()
    tcr_cdr3b = set((BANK_DIR / "tcr_cdr3b.txt").read_text().split()) if (BANK_DIR / "tcr_cdr3b.txt").is_file() else set()

    rows = list(cfg["extract"](path))
    n = len(rows)
    leaked: dict[int, dict[str, Any]] = {}

    report: dict[str, Any] = {"source": name, "domain": domain, "path": str(path), "n_rows": n}

    if domain == "tcr":
        exact = 0
        for r in rows:
            key = r.get("tcr_cdr3b")
            if key and key in tcr_cdr3b:
                leaked.setdefault(r["row"], {})["tcr_cdr3b"] = key
                exact += 1
        report["tcr_cdr3b_exact_hits"] = exact
        report["bank_tcr_cdr3b"] = len(tcr_cdr3b)

    elif domain == "antibody":
        # 1) fast exact CDR-H3
        exact = 0
        for r in rows:
            key = r.get("cdrh3")
            if key and key in ab_cdrh3:
                leaked.setdefault(r["row"], {})["cdrh3_exact"] = key
                exact += 1
        report["cdrh3_exact_hits"] = exact
        report["bank_ab_cdrh3"] = len(ab_cdrh3)
        # 2) CDR-H3 cluster (< 100%) via mmseqs, only on rows not already exact
        if not args.exact_only and args.cdrh3_id < 1.0:
            uniq_cdrh3: dict[str, list[int]] = {}
            for r in rows:
                key = r.get("cdrh3")
                if key and r["row"] not in leaked:
                    uniq_cdrh3.setdefault(key, []).append(r["row"])
            q = list(uniq_cdrh3)
            hits = mmseqs_query_hits(q, _cdrh3_fasta(),
                                     min_seq_id=args.cdrh3_id, coverage=args.cdrh3_cov,
                                     threads=args.threads)
            cl = 0
            for qi in hits:
                for row_id in uniq_cdrh3[q[qi]]:
                    leaked.setdefault(row_id, {})["cdrh3_cluster"] = q[qi]
                    cl += 1
            report["cdrh3_cluster_hits"] = cl
        # 3) full-length heavy/VHH identity via mmseqs
        if not args.exact_only:
            uniq_full: dict[str, list[int]] = {}
            for r in rows:
                for s in r.get("full", []):
                    if r["row"] not in leaked or True:
                        uniq_full.setdefault(s, []).append(r["row"])
            q = list(uniq_full)
            hits = mmseqs_query_hits(q, BANK_DIR / "ab_heavy.fasta",
                                     min_seq_id=args.ab_full_id, coverage=args.ab_full_cov,
                                     threads=args.threads)
            fl = 0
            for qi in hits:
                for row_id in uniq_full[q[qi]]:
                    leaked.setdefault(row_id, {})["heavy_full"] = q[qi]
                    fl += 1
            report["heavy_full_hits"] = fl

    elif domain == "ppi":
        if args.exact_only:
            report["skipped"] = "ppi requires mmseqs (global id); rerun without --exact-only"
        else:
            uniq_full: dict[str, list[int]] = {}
            for r in rows:
                for s in r.get("full", []):
                    uniq_full.setdefault(s, []).append(r["row"])
            q = list(uniq_full)
            hits = mmseqs_query_hits(q, BANK_DIR / "ppi_proteins.fasta",
                                     min_seq_id=args.ppi_id, coverage=args.ppi_cov,
                                     threads=args.threads)
            pl = 0
            for qi in hits:
                for row_id in uniq_full[q[qi]]:
                    leaked.setdefault(row_id, {})["ppi_chain"] = q[qi]
                    pl += 1
            report["ppi_chain_hits"] = pl

    report["n_leaked_rows"] = len(leaked)
    report["leak_frac"] = round(len(leaked) / n, 6) if n else 0.0

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (BLOCK_DIR / f"{name}.jsonl").open("w") as fh:
        for row_id in sorted(leaked):
            fh.write(json.dumps({"row": row_id, **leaked[row_id]}) + "\n")
    return report


def _cdrh3_fasta() -> Path:
    """Materialize the CDR-H3 bank as FASTA for mmseqs (built on demand)."""
    src = BANK_DIR / "ab_cdrh3.txt"
    out = BANK_DIR / "ab_cdrh3_fasta.fasta"
    seqs = src.read_text().split() if src.is_file() else []
    with out.open("w") as fh:
        for i, s in enumerate(seqs):
            fh.write(f">t{i}\n{s}\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=sorted(SOURCES))
    parser.add_argument("--exact-only", action="store_true",
                        help="Skip all MMseqs2 similarity passes (fast exact-key report).")
    parser.add_argument("--cdrh3-id", type=float, default=0.70)
    parser.add_argument("--cdrh3-cov", type=float, default=0.80)
    parser.add_argument("--ab-full-id", type=float, default=0.95)
    parser.add_argument("--ab-full-cov", type=float, default=0.80)
    parser.add_argument("--ppi-id", type=float, default=0.40)
    parser.add_argument("--ppi-cov", type=float, default=0.50)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()

    report = run_source(args.source, args)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
