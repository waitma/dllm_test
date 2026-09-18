#!/usr/bin/env python3
"""Dedicated, scale-aware leakage dedup for the ``mint_ppi`` source.

``mint_ppi`` is the largest training source (~82.4M physical-binding pairs over
~16.4M unique STRING proteins). The generic per-row Arrow extractor is far too
slow here, and the 40%-identity PPI rule must run on the *protein universe*, not
per pair. So this path:

1. **Cluster once.** Run ``mmseqs easy-linclust`` on the union of the PPI test
   bank (``banks/ppi_proteins.fasta``) and the mint_ppi protein sequence map
   (``training_filtered.seqs.txt.gz``) at ``--min-seq-id`` (default 0.40). Any
   mint protein that co-clusters with a test protein is *leaked*.
2. **Map to Arrow rows.** Stream the links file in the SAME order and with the
   SAME filter the shard builder used (``ppi_record``: both sequences valid, both
   <= --max-protein-length), counting only kept rows so the emitted index equals
   the Arrow row index. A row is leaked if either protein is in the leaked set.

Outputs (matching the generic tool so ``apply_blocklists.py`` consumes them):
  * ``data/dedup/reports/mint_ppi.json``
  * ``data/dedup/blocklists/mint_ppi.jsonl``
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")

DATA = PROJECT_ROOT / "data"
BANK_DIR = DATA / "dedup" / "banks"
REPORT_DIR = DATA / "dedup" / "reports"
BLOCK_DIR = DATA / "dedup" / "blocklists"
MMSEQS_BIN = "/vepfs-mlp2/c20250601/251105016/conda/envs/flow/bin/mmseqs"
MINT_DIR = DATA / "ppi_task_raw/processed/mint_string_pretrain_v1"


def _load_records_mod():
    path = PROJECT_ROOT / "dllm/pipelines/immune_llada/data/records.py"
    name = "_dedup_records_mint"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rec = _load_records_mod()
normalize_sequence = _rec.normalize_sequence
is_valid_protein_sequence = _rec.is_valid_protein_sequence


def _load_fasta_seqs(path: Path) -> list[str]:
    seqs, cur = [], []
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


def cluster_leaked_proteins(
    seqs_gz: Path,
    bank_fasta: Path,
    *,
    min_seq_id: float,
    coverage: float,
    threads: int,
    tmp_root: Path,
) -> set[str]:
    """Return the set of mint protein *names* that co-cluster with a test protein.

    Sequences are keyed by their mint name (``q<name>``); test bank entries use
    ``t<i>``. Normalization matches the shard builder (J->L, upper).
    """
    tmp = Path(tempfile.mkdtemp(prefix="dedup_mint_linclust_", dir=str(tmp_root)))
    try:
        combined = tmp / "all.fasta"
        n_targets = n_query = 0
        with combined.open("w") as out:
            for i, s in enumerate(_load_fasta_seqs(bank_fasta)):
                s = normalize_sequence(s)
                if s:
                    out.write(f">t{i}\n{s}\n")
                    n_targets += 1
            with gzip.open(seqs_gz, "rt") as fh:
                for line in fh:
                    parts = line.split(None, 1)
                    if len(parts) != 2:
                        continue
                    name, seq = parts[0], normalize_sequence(parts[1])
                    if is_valid_protein_sequence(seq):
                        # mmseqs id must be single token; STRING names are token-safe
                        out.write(f">q{name}\n{seq}\n")
                        n_query += 1
        print(f"linclust input: {n_targets:,} test + {n_query:,} mint proteins", flush=True)

        clu = tmp / "clu"
        subprocess.run(
            [MMSEQS_BIN, "easy-linclust", str(combined), str(clu), str(tmp / "scratch"),
             "--min-seq-id", str(min_seq_id), "-c", str(coverage), "--cov-mode", "0",
             "--threads", str(threads)],
            check=True,
        )
        tsv = tmp / "clu_cluster.tsv"
        # Group members by representative, flag clusters containing a test seq.
        members: dict[str, list[str]] = {}
        with tsv.open() as fh:
            for line in fh:
                rep, mem = line.rstrip("\n").split("\t")
                members.setdefault(rep, []).append(mem)
        leaked: set[str] = set()
        for rep, mem_list in members.items():
            group = mem_list + [rep]
            if any(m.startswith("t") for m in group):
                for m in group:
                    if m.startswith("q"):
                        leaked.add(m[1:])  # strip 'q' prefix -> mint protein name
        print(f"leaked mint proteins: {len(leaked):,}", flush=True)
        return leaked
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_seq_map(seqs_gz: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    with gzip.open(seqs_gz, "rt") as fh:
        for line in fh:
            parts = line.split(None, 1)
            if len(parts) == 2:
                seqs[parts[0]] = parts[1].strip()
    return seqs


def map_leaked_rows(
    links_gz: Path,
    seqs: dict[str, str],
    leaked_proteins: set[str],
    *,
    max_protein_length: int,
) -> tuple[list[tuple[int, str, str]], int]:
    """Stream links; replicate shard filter; emit (arrow_row, target, actor) leaks.

    The shard builder keeps a row iff both proteins normalize to valid protein
    sequences within ``max_protein_length``. The Arrow row index therefore
    advances only on kept rows -- we mirror that exactly here.
    """
    arrow_row = 0
    leaks: list[tuple[int, str, str]] = []
    with gzip.open(links_gz, "rt") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 2:
                continue
            target, actor = parts[0], parts[1]
            seq_t = normalize_sequence(seqs.get(target, ""))
            seq_a = normalize_sequence(seqs.get(actor, ""))
            if not is_valid_protein_sequence(seq_t) or not is_valid_protein_sequence(seq_a):
                continue
            if len(seq_t) > max_protein_length or len(seq_a) > max_protein_length:
                continue
            # This row is kept by the shard builder -> it has arrow index arrow_row.
            if target in leaked_proteins or actor in leaked_proteins:
                leaks.append((arrow_row, target, actor))
            arrow_row += 1
    return leaks, arrow_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-seq-id", type=float, default=0.40)
    parser.add_argument("--coverage", type=float, default=0.50)
    parser.add_argument("--max-protein-length", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--tmp-root", type=Path, default=DATA / "dedup")
    args = parser.parse_args()

    seqs_gz = MINT_DIR / "training_filtered.seqs.txt.gz"
    links_gz = MINT_DIR / "training_filtered.links.txt.gz"
    bank = BANK_DIR / "ppi_proteins.fasta"
    for p in (seqs_gz, links_gz, bank):
        if not p.exists():
            raise FileNotFoundError(p)

    print("== step 1: cluster protein universe vs PPI test bank ==", flush=True)
    leaked_proteins = cluster_leaked_proteins(
        seqs_gz, bank,
        min_seq_id=args.min_seq_id, coverage=args.coverage,
        threads=args.threads, tmp_root=args.tmp_root,
    )

    print("== step 2: load seq map + stream links -> arrow-row blocklist ==", flush=True)
    seqs = load_seq_map(seqs_gz)
    leaks, n_rows = map_leaked_rows(
        links_gz, seqs, leaked_proteins, max_protein_length=args.max_protein_length,
    )

    report = {
        "source": "mint_ppi",
        "domain": "ppi",
        "path": str(DATA / "bioseq_grammar_v1/mint_ppi/train"),
        "n_rows": n_rows,
        "n_leaked_rows": len(leaks),
        "leak_frac": round(len(leaks) / n_rows, 6) if n_rows else 0.0,
        "leaked_proteins": len(leaked_proteins),
        "min_seq_id": args.min_seq_id,
        "coverage": args.coverage,
        "rule": "ppi_global_linclust_40",
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    BLOCK_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "mint_ppi.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    with (BLOCK_DIR / "mint_ppi.jsonl").open("w") as fh:
        for row_id, target, actor in leaks:
            fh.write(json.dumps({"row": row_id, "ppi_pair": f"{target}|{actor}"}) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
