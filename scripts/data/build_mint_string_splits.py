#!/usr/bin/env python3
"""Build MINT-compatible STRING physical-link train/valid splits.

This follows the published procedure in Ullanat et al. (Nat Commun 2026) and the
reference implementation ``stringdb.py`` in https://github.com/VarunUllanat/mint

Prerequisites (already on disk under ``data/ppi_task_raw/raw/stringdb_mint``)::

    protein.sequences.v12.0.fa.gz
    protein.physical.links.full.v12.0.txt.gz   # MINT uses the *full* physical links file

MMseqs2 clustering at 50% sequence identity (run once, ~hours)::

    cd /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/stringdb_mint
    gunzip -k protein.sequences.v12.0.fa.gz
    mmseqs createdb protein.sequences.v12.0.fa DB100
    mmseqs cluster DB100 clu50 /tmp/mmseqs --min-seq-id 0.50 --remove-tmp-files
    mmseqs createtsv DB100 DB100 clu50 clu50.tsv

Then build splits::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_mint_string_splits.py

Outputs (under ``data/ppi_task_raw/processed/mint_string_pretrain_v1/``)::

    validation.links.txt.gz
    validation.seqs.txt.gz
    training_filtered.links.txt.gz
    training_filtered.seqs.txt.gz
    manifest.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
RAW_ROOT = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint"
OUT_ROOT = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1"

NUM_VALID = 250_000
FILTER_SHUFFLE_SEED = 137
SPLIT_SHUFFLE_SEED = 731


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--sequences-fa", type=Path, default=None, help="Uncompressed FASTA (default: raw-root/protein.sequences.v12.0.fa)")
    parser.add_argument("--cluster-tsv", type=Path, default=None, help="MMseqs clu50.tsv (default: raw-root/clu50.tsv)")
    parser.add_argument(
        "--links-gz",
        type=Path,
        default=None,
        help="Default: raw-root/protein.physical.links.full.v12.0.txt.gz",
    )
    parser.add_argument("--num-valid", type=int, default=NUM_VALID)
    parser.add_argument("--max-links", type=int, default=0, help="Debug cap on links read (0 = all).")
    return parser.parse_args()


def read_sequences(fasta_path: Path) -> dict[str, str]:
    seqs: dict[str, str] = {}
    with fasta_path.open() as handle:
        name: str | None = None
        chunks: list[str] = []
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(chunks)
                name = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if name is not None:
            seqs[name] = "".join(chunks)
    return seqs


def read_cluster_map(cluster_tsv: Path) -> dict[str, str]:
    reps: dict[str, str] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            rep, seq = line.strip().split()[:2]
            reps[seq] = rep
    return reps


def read_links(links_gz: Path, max_links: int) -> list[str]:
    links: list[str] = []
    with gzip.open(links_gz, "rt") as handle:
        header = next(handle)
        if "protein1" not in header:
            raise ValueError(f"Unexpected links header: {header!r}")
        for index, line in enumerate(handle, start=1):
            links.append(line.strip())
            if max_links and index >= max_links:
                break
    return links


def filter_unique_cluster_pairs(links: list[str], reps: dict[str, str], seed: int) -> list[str]:
    random.seed(seed)
    random.shuffle(links)
    linked_clusters: set[tuple[str, str]] = set()
    filtered: list[str] = []
    for link in links:
        name1, name2 = link.split()[:2]
        clu1, clu2 = reps[name1], reps[name2]
        key = tuple(sorted((clu1, clu2)))
        if key in linked_clusters:
            continue
        linked_clusters.add(key)
        filtered.append(link)
    return filtered


def write_links_and_seqs(
    links: list[str],
    seqs: dict[str, str],
    links_path: Path,
    seqs_path: Path,
) -> tuple[int, int]:
    written_seqs: set[str] = set()
    with gzip.open(links_path, "wt") as links_file, gzip.open(seqs_path, "wt") as seqs_file:
        for link in links:
            links_file.write(link + "\n")
            name1, name2 = link.split()[:2]
            for name in (name1, name2):
                if name in written_seqs:
                    continue
                seqs_file.write(f"{name} {seqs[name]}\n")
                written_seqs.add(name)
    return len(links), len(written_seqs)


def filter_training_disjoint_clusters(
    training_links: list[str],
    validation_links: list[str],
    reps: dict[str, str],
    links_path: Path,
    seqs_path: Path,
    seqs: dict[str, str],
) -> tuple[int, int, int]:
    val_clusters = set()
    for link in validation_links:
        name1, name2 = link.split()[:2]
        val_clusters.add(reps[name1])
        val_clusters.add(reps[name2])

    kept: list[str] = []
    scanned = 0
    for link in training_links:
        scanned += 1
        name1, name2 = link.split()[:2]
        if reps[name1] in val_clusters or reps[name2] in val_clusters:
            continue
        kept.append(link)

    written_links, written_seqs = write_links_and_seqs(kept, seqs, links_path, seqs_path)
    return scanned, written_links, written_seqs


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root
    sequences_fa = args.sequences_fa or (raw_root / "protein.sequences.v12.0.fa")
    cluster_tsv = args.cluster_tsv or (raw_root / "clu50.tsv")
    links_gz = args.links_gz or (raw_root / "protein.physical.links.full.v12.0.txt.gz")

    for path in (sequences_fa, cluster_tsv, links_gz):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input {path}. See docstring for MMseqs2 and download steps."
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading sequences from {sequences_fa}", flush=True)
    seqs = read_sequences(sequences_fa)
    print(f"Loaded {len(seqs):,} sequences", flush=True)

    print(f"Reading clusters from {cluster_tsv}", flush=True)
    reps = read_cluster_map(cluster_tsv)
    print(f"Loaded {len(reps):,} sequence->cluster mappings ({len(set(reps.values())):,} clusters)", flush=True)

    print(f"Reading links from {links_gz}", flush=True)
    links = read_links(links_gz, args.max_links)
    print(f"Read {len(links):,} raw links", flush=True)

    print("Filtering to one link per cluster pair (MINT step 1)", flush=True)
    filtered = filter_unique_cluster_pairs(links, reps, FILTER_SHUFFLE_SEED)
    print(f"Kept {len(filtered):,} cluster-unique links", flush=True)

    random.seed(SPLIT_SHUFFLE_SEED)
    random.shuffle(filtered)
    validation = filtered[: args.num_valid]
    training = filtered[args.num_valid :]
    print(f"Split: valid={len(validation):,}, train_pool={len(training):,}", flush=True)

    val_links = args.output_dir / "validation.links.txt.gz"
    val_seqs = args.output_dir / "validation.seqs.txt.gz"
    val_nlinks, val_nseqs = write_links_and_seqs(validation, seqs, val_links, val_seqs)
    print(f"Wrote validation: {val_nlinks:,} links, {val_nseqs:,} seqs", flush=True)

    train_links = args.output_dir / "training_filtered.links.txt.gz"
    train_seqs = args.output_dir / "training_filtered.seqs.txt.gz"
    scanned, train_nlinks, train_nseqs = filter_training_disjoint_clusters(
        training, validation, reps, train_links, train_seqs, seqs
    )
    print(
        f"Wrote training_filtered: scanned={scanned:,}, kept={train_nlinks:,} links, {train_nseqs:,} seqs",
        flush=True,
    )

    manifest = {
        "policy_id": "mint_string_pretrain_v1",
        "reference": "Ullanat et al., Nat Commun 2026; github.com/VarunUllanat/mint stringdb.py",
        "inputs": {
            "sequences_fa": str(sequences_fa),
            "cluster_tsv": str(cluster_tsv),
            "links_gz": str(links_gz),
        },
        "params": {
            "num_valid": args.num_valid,
            "filter_shuffle_seed": FILTER_SHUFFLE_SEED,
            "split_shuffle_seed": SPLIT_SHUFFLE_SEED,
            "mmseqs_min_seq_id": 0.50,
        },
        "outputs": {
            "valid": {"links": str(val_links), "seqs": str(val_seqs), "n_links": val_nlinks, "n_seqs": val_nseqs},
            "train": {
                "links": str(train_links),
                "seqs": str(train_seqs),
                "n_links": train_nlinks,
                "n_seqs": train_nseqs,
            },
        },
        "forbidden": [
            "Do not merge Bernett gold-standard test pairs into mint_string_pretrain train.",
            "Do not merge HumanPPI/YeastPPI test or cross_species_test into pretraining.",
        ],
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
