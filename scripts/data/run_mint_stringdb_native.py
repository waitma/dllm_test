#!/usr/bin/env python3
"""Run MINT ``stringdb.py`` in-memory split pipeline with local paths.

This mirrors https://github.com/VarunUllanat/mint/blob/main/stringdb.py
(seed 137 filter shuffle, seed 731 split shuffle, 250k validation, cluster
dedup + train/valid cluster disjoint filter). Only change: configurable paths
and stdlib logging instead of ``mint.utils.logging``.

Usage::

    python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/run_mint_stringdb_native.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from pathlib import Path

from Bio import SeqIO

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
RAW_ROOT = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint"
OUT_ROOT = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v1"

NUM_VALID = 250_000
FILTER_SHUFFLE_SEED = 137
SPLIT_SHUFFLE_SEED = 731

logger = logging.getLogger("mint_stringdb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--sequences-fa", type=Path, default=None)
    parser.add_argument("--cluster-tsv", type=Path, default=None)
    parser.add_argument("--links-gz", type=Path, default=None)
    parser.add_argument("--num-valid", type=int, default=NUM_VALID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    sequences_fa = args.sequences_fa or (args.raw_root / "protein.sequences.v12.0.fa")
    cluster_tsv = args.cluster_tsv or (args.raw_root / "clu50.tsv")
    links_gz = args.links_gz or (args.raw_root / "protein.physical.links.full.v12.0.txt.gz")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for path in (sequences_fa, cluster_tsv, links_gz):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    logger.info("===Reading seqs====")
    seqs: dict[str, str] = {}
    for seq in SeqIO.parse(sequences_fa.open(), "fasta"):
        seqs[seq.name] = str(seq.seq)
        if len(seqs) % 1_000_000 == 0:
            logger.info("%s million seqs read", len(seqs) / 1e6)
    logger.info("Done, %s seqs total", len(seqs))

    logger.info("===Reading reps====")
    reps: dict[str, str] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            rep, seq = line.strip().split()[:2]
            reps[seq] = rep
            if len(reps) % 1_000_000 == 0:
                logger.info("%s million reps read", len(reps) / 1e6)
    logger.info("Done, %s reps total, %s clusters", len(reps), len(set(reps.values())))

    logger.info("===Reading links====")
    handle = gzip.open(links_gz, "rt")
    handle = iter(handle)
    next(handle)
    links: list[str] = []
    index = 0
    while True:
        try:
            line = next(handle).strip()
        except StopIteration:
            break
        index += 1
        links.append(line)
        if index % 1_000_000 == 0:
            logger.info("%s million links read", index / 1e6)
    logger.info("Done, %s links total", len(links))

    logger.info("===Shuffling links===")
    random.seed(FILTER_SHUFFLE_SEED)
    random.shuffle(links)
    logger.info("Done shuffling links")

    logger.info("===Filtering links===")
    linked_clusters: set[tuple[str, str]] = set()
    filtered_links: list[str] = []
    scanned = 0
    for link in links:
        scanned += 1
        name1, name2 = link.split()[:2]
        clu1, clu2 = reps[name1], reps[name2]
        key = tuple(sorted((clu1, clu2)))
        if key not in linked_clusters:
            linked_clusters.add(key)
            filtered_links.append(link)
        if scanned % 1_000_000 == 0:
            logger.info(
                "%s million links filtered, %s million kept",
                scanned / 1e6,
                len(filtered_links) / 1e6,
            )
    links = filtered_links
    logger.info("Done, %s links filtered, %s kept", scanned, len(links))

    logger.info("===Shuffling links===")
    random.seed(SPLIT_SHUFFLE_SEED)
    random.shuffle(links)
    logger.info("Done shuffling links")

    filtered_path = args.output_dir / "filtered.links.txt.gz"
    with gzip.open(filtered_path, "wt") as links_file:
        for link in links:
            links_file.write(link + "\n")
    logger.info("Wrote intermediate %s", filtered_path)

    validation = links[: args.num_valid]
    training = links[args.num_valid :]

    val_links = args.output_dir / "validation.links.txt.gz"
    val_seqs = args.output_dir / "validation.seqs.txt.gz"
    logger.info("===Writing validation links===")
    written_seqs: set[str] = set()
    with gzip.open(val_links, "wt") as links_file, gzip.open(val_seqs, "wt") as seqs_file:
        for link in validation:
            links_file.write(link + "\n")
            name1, name2 = link.split()[:2]
            if name1 not in written_seqs:
                seqs_file.write(name1 + " " + seqs[name1] + "\n")
                written_seqs.add(name1)
            if name2 not in written_seqs:
                seqs_file.write(name2 + " " + seqs[name2] + "\n")
                written_seqs.add(name2)
    logger.info("Done, %s validation links written, %s seqs", args.num_valid, len(written_seqs))

    train_links = args.output_dir / "training.links.txt.gz"
    train_seqs = args.output_dir / "training.seqs.txt.gz"
    logger.info("===Writing training links===")
    written_count = 0
    written_seqs = set()
    with gzip.open(train_links, "wt") as links_file, gzip.open(train_seqs, "wt") as seqs_file:
        for link in training:
            written_count += 1
            links_file.write(link + "\n")
            name1, name2 = link.split()[:2]
            if name1 not in written_seqs:
                seqs_file.write(name1 + " " + seqs[name1] + "\n")
                written_seqs.add(name1)
            if name2 not in written_seqs:
                seqs_file.write(name2 + " " + seqs[name2] + "\n")
                written_seqs.add(name2)
            if written_count % 1_000_000 == 0:
                logger.info(
                    "%s million training links written, %s million seqs",
                    written_count / 1e6,
                    len(written_seqs) / 1e6,
                )
    logger.info("Done, %s training links written, %s seqs", written_count, len(written_seqs))

    logger.info("===Extracting validation clusters===")
    val_clus: set[str] = set()
    for link in validation:
        name1, name2 = link.split()[:2]
        val_clus.add(reps[name1])
        val_clus.add(reps[name2])
    logger.info("Done, %s validation clusters", len(val_clus))

    train_filtered_links = args.output_dir / "training_filtered.links.txt.gz"
    train_filtered_seqs = args.output_dir / "training_filtered.seqs.txt.gz"
    logger.info("===Writing filtered training links===")
    scanned_train = 0
    kept_train = 0
    written_seqs = set()
    with gzip.open(train_filtered_links, "wt") as links_file, gzip.open(
        train_filtered_seqs, "wt"
    ) as seqs_file:
        for link in training:
            scanned_train += 1
            name1, name2 = link.split()[:2]
            clu1, clu2 = reps[name1], reps[name2]
            if clu1 not in val_clus and clu2 not in val_clus:
                kept_train += 1
                links_file.write(link + "\n")
                if name1 not in written_seqs:
                    seqs_file.write(name1 + " " + seqs[name1] + "\n")
                    written_seqs.add(name1)
                if name2 not in written_seqs:
                    seqs_file.write(name2 + " " + seqs[name2] + "\n")
                    written_seqs.add(name2)
            if scanned_train % 1_000_000 == 0:
                logger.info(
                    "%s million training links filtered, %s million written, %s million seqs",
                    scanned_train / 1e6,
                    kept_train / 1e6,
                    len(written_seqs) / 1e6,
                )
    logger.info(
        "%s training links filtered, %s kept, %s seqs",
        scanned_train,
        kept_train,
        len(written_seqs),
    )

    manifest = {
        "policy_id": "mint_string_pretrain_v1",
        "reference": "VarunUllanat/mint stringdb.py (in-memory native run)",
        "runner": str(Path(__file__).resolve()),
        "inputs": {
            "sequences_fa": str(sequences_fa),
            "cluster_tsv": str(cluster_tsv),
            "links_gz": str(links_gz),
        },
        "params": {
            "num_valid": args.num_valid,
            "filter_shuffle_seed": FILTER_SHUFFLE_SEED,
            "split_shuffle_seed": SPLIT_SHUFFLE_SEED,
        },
        "outputs": {
            "valid": {"links": str(val_links), "seqs": str(val_seqs), "n_links": len(validation)},
            "train_filtered": {
                "links": str(train_filtered_links),
                "seqs": str(train_filtered_seqs),
                "n_links": kept_train,
            },
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
