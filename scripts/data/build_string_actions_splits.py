#!/usr/bin/env python3
"""Build MINT-style train/valid splits from STRING protein.actions (v11).

Each actions row with ``a_is_acting=t`` defines actor=item_id_a acting on
item_id_b (target). Output link lines are::

    target_id actor_id mode

Grammar rendering maps target -> ``protein_a`` (fixed context) and actor ->
``protein_b`` (generated), matching ``GrammarRenderer`` PPI conditional mode.

Prerequisites::

    protein.actions.v11.0.txt.gz
    protein.sequences.v11.0.fa
    clu50.v11.0.tsv   # MMseqs2 50% clustering on v11 sequences

Usage::

    python scripts/data/build_string_actions_splits.py \\
      --actions-gz data/ppi_task_raw/raw/stringdb_mint/protein.actions.v11.0.txt.gz \\
      --sequences-fa data/ppi_task_raw/raw/stringdb_mint/protein.sequences.v11.0.fa \\
      --cluster-tsv data/ppi_task_raw/raw/stringdb_mint/clu50.v11.0.tsv \\
      --output-dir data/ppi_task_raw/processed/mint_string_actions_v11.0

Human-only smoke test (no full v11 sequences download)::

    python scripts/data/build_string_actions_splits.py \\
      --actions-gz data/ppi_task_raw/raw/stringdb_mint/9606.protein.actions.v11.0.txt.gz \\
      --sequences-fa data/ppi_task_raw/raw/stringdb_mint/protein.sequences.v12.0.fa \\
      --cluster-tsv data/ppi_task_raw/raw/stringdb_mint/clu50.tsv \\
      --output-dir /tmp/mint_string_actions_human_smoke \\
      --max-records 50000
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data.ppi_relations import normalize_relation

RAW_ROOT = PROJECT_ROOT / "data/ppi_task_raw/raw/stringdb_mint"
DEFAULT_OUT = PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_actions_v11.0"

NUM_VALID = 250_000
FILTER_SHUFFLE_SEED = 137
SPLIT_SHUFFLE_SEED = 731

BINDING_MODE = "binding"
DIRECTIONAL_MODES = frozenset(
    {"catalysis", "expression", "activation", "inhibition", "ptmod", "reaction"}
)

logger = logging.getLogger("string_actions_splits")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions-gz", type=Path, default=RAW_ROOT / "protein.actions.v11.0.txt.gz")
    parser.add_argument("--sequences-fa", type=Path, default=RAW_ROOT / "protein.sequences.v11.0.fa")
    parser.add_argument("--cluster-tsv", type=Path, default=RAW_ROOT / "clu50.v11.0.tsv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--num-valid", type=int, default=NUM_VALID)
    parser.add_argument(
        "--binding-keep-frac",
        type=float,
        default=0.05,
        help="Fraction of binding-mode rows to keep (binding is abundant and generic).",
    )
    parser.add_argument("--binding-seed", type=int, default=42)
    parser.add_argument("--max-records", type=int, default=None, help="Debug cap on parsed action rows.")
    parser.add_argument(
        "--physical-valid-links",
        type=Path,
        default=PROJECT_ROOT / "data/ppi_task_raw/processed/mint_string_pretrain_v11.0/validation.links.txt.gz",
        help="Optional physical MINT valid links; clusters from this split are excluded from actions train.",
    )
    return parser.parse_args()


def mode_to_relation(mode: str, action: str) -> str:
    mode_norm = normalize_relation(mode)
    action_norm = normalize_relation(action) if action else ""
    if mode_norm == "expression" and action_norm == "inhibition":
        return "inhibition"
    if mode_norm == "expression" and action_norm == "activation":
        return "activation"
    return mode_norm


def parse_action_edges(
    actions_gz: Path,
    *,
    binding_keep_frac: float,
    binding_seed: int,
    max_records: int | None,
) -> list[tuple[str, str, str, int]]:
    """Return deduped edges as (target_id, actor_id, relation, score)."""
    binding_rng = random.Random(binding_seed)
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str, int]] = []
    parsed = 0

    with gzip.open(actions_gz, "rt") as handle:
        header = handle.readline().strip().split("\t")
        if header[:3] != ["item_id_a", "item_id_b", "mode"]:
            raise ValueError(f"Unexpected actions header: {header}")

        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 7:
                continue
            item_a, item_b, mode, action, _is_dir, acting, score_text = parts[:7]
            if acting != "t":
                continue
            parsed += 1
            if max_records is not None and parsed > max_records:
                break

            relation = mode_to_relation(mode, action)
            score = int(score_text)

            if relation == BINDING_MODE:
                if binding_rng.random() > binding_keep_frac:
                    continue
                target_id, actor_id = sorted((item_a, item_b))
            else:
                target_id, actor_id = item_b, item_a

            key = (target_id, actor_id, relation)
            if key in seen:
                continue
            seen.add(key)
            edges.append((target_id, actor_id, relation, score))

    logger.info(
        "Parsed %s acting rows -> %s unique (target, actor, relation) edges",
        parsed,
        len(edges),
    )
    return edges


def load_sequence_subset(sequences_fa: Path, needed: set[str]) -> dict[str, str]:
    seqs: dict[str, str] = {}
    missing: set[str] = set()
    for record in SeqIO.parse(sequences_fa.open(), "fasta"):
        if record.id not in needed:
            continue
        seqs[record.id] = str(record.seq)
        if len(seqs) == len(needed):
            break
    missing = needed - set(seqs)
    if missing:
        logger.warning(
            "Missing %s/%s requested sequences in %s (e.g. %s)",
            len(missing),
            len(needed),
            sequences_fa,
            sorted(missing)[:5],
        )
    return seqs


def filter_edges_with_reps_and_seqs(
    edges: list[tuple[str, str, str]],
    reps: dict[str, str],
    seqs: dict[str, str],
) -> list[tuple[str, str, str]]:
    kept: list[tuple[str, str, str]] = []
    dropped_reps = 0
    dropped_seqs = 0
    for target_id, actor_id, relation in edges:
        if target_id not in reps or actor_id not in reps:
            dropped_reps += 1
            continue
        if target_id not in seqs or actor_id not in seqs:
            dropped_seqs += 1
            continue
        kept.append((target_id, actor_id, relation))
    logger.info(
        "Edge filter: kept=%s dropped_missing_cluster=%s dropped_missing_sequence=%s",
        len(kept),
        dropped_reps,
        dropped_seqs,
    )
    return kept


def load_reps(cluster_tsv: Path) -> dict[str, str]:
    reps: dict[str, str] = {}
    with cluster_tsv.open() as handle:
        for line in handle:
            rep, seq = line.strip().split()[:2]
            reps[seq] = rep
    return reps


def cluster_key(target_id: str, actor_id: str, relation: str, reps: dict[str, str]) -> tuple:
    clu_target = reps[target_id]
    clu_actor = reps[actor_id]
    if relation == BINDING_MODE:
        return (relation, tuple(sorted((clu_target, clu_actor))))
    return (relation, clu_target, clu_actor)


def validation_clusters_from_physical(physical_valid_links: Path, reps: dict[str, str]) -> set[str]:
    if not physical_valid_links.exists():
        logger.warning("Physical valid links not found (%s); skip cross-split cluster exclusion", physical_valid_links)
        return set()
    clusters: set[str] = set()
    with gzip.open(physical_valid_links, "rt") as handle:
        for line in handle:
            name1, name2 = line.strip().split()[:2]
            clusters.add(reps[name1])
            clusters.add(reps[name2])
    logger.info("Loaded %s validation clusters from %s", len(clusters), physical_valid_links)
    return clusters


def write_links_and_seqs(
    links: list[tuple[str, str, str]],
    seqs: dict[str, str],
    links_path: Path,
    seqs_path: Path,
) -> None:
    written_seqs: set[str] = set()
    with gzip.open(links_path, "wt") as links_file, gzip.open(seqs_path, "wt") as seqs_file:
        for target_id, actor_id, relation in links:
            links_file.write(f"{target_id} {actor_id} {relation}\n")
            for name in (target_id, actor_id):
                if name in written_seqs:
                    continue
                seqs_file.write(f"{name} {seqs[name]}\n")
                written_seqs.add(name)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
        force=True,
    )

    for path in (args.actions_gz, args.sequences_fa, args.cluster_tsv):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges_scored = parse_action_edges(
        args.actions_gz,
        binding_keep_frac=args.binding_keep_frac,
        binding_seed=args.binding_seed,
        max_records=args.max_records,
    )
    edges = [(t, a, r) for t, a, r, _ in edges_scored]

    logger.info("Loading cluster map from %s", args.cluster_tsv)
    reps = load_reps(args.cluster_tsv)

    needed_ids = {name for t, a, _ in edges for name in (t, a) if name in reps}
    logger.info("Loading sequences for %s cluster-mapped proteins", len(needed_ids))
    seqs = load_sequence_subset(args.sequences_fa, needed_ids)
    edges = filter_edges_with_reps_and_seqs(edges, reps, seqs)
    if not edges:
        raise RuntimeError("No action edges remain after cluster/sequence filtering.")

    random.seed(FILTER_SHUFFLE_SEED)
    random.shuffle(edges)

    linked_clusters: set[tuple] = set()
    filtered: list[tuple[str, str, str]] = []
    for edge in edges:
        target_id, actor_id, relation = edge
        key = cluster_key(target_id, actor_id, relation, reps)
        if key in linked_clusters:
            continue
        linked_clusters.add(key)
        filtered.append(edge)
    logger.info("Cluster dedup: %s -> %s edges", len(edges), len(filtered))

    random.seed(SPLIT_SHUFFLE_SEED)
    random.shuffle(filtered)

    validation = filtered[: args.num_valid]
    training = filtered[args.num_valid :]

    val_clusters: set[str] = set()
    for target_id, actor_id, _relation in validation:
        val_clusters.add(reps[target_id])
        val_clusters.add(reps[actor_id])

    physical_val_clusters = validation_clusters_from_physical(args.physical_valid_links, reps)
    excluded_clusters = val_clusters | physical_val_clusters

    train_filtered = [
        edge
        for edge in training
        if reps[edge[0]] not in excluded_clusters and reps[edge[1]] not in excluded_clusters
    ]
    logger.info(
        "Split: valid=%s train=%s train_filtered=%s (excluded %s clusters)",
        len(validation),
        len(training),
        len(train_filtered),
        len(excluded_clusters),
    )

    val_links = args.output_dir / "validation.links.txt.gz"
    val_seqs = args.output_dir / "validation.seqs.txt.gz"
    train_links = args.output_dir / "training_filtered.links.txt.gz"
    train_seqs = args.output_dir / "training_filtered.seqs.txt.gz"

    write_links_and_seqs(validation, seqs, val_links, val_seqs)
    write_links_and_seqs(train_filtered, seqs, train_links, train_seqs)

    manifest = {
        "policy_id": "mint_string_actions_v11",
        "inputs": {
            "actions_gz": str(args.actions_gz),
            "sequences_fa": str(args.sequences_fa),
            "cluster_tsv": str(args.cluster_tsv),
            "physical_valid_links": str(args.physical_valid_links),
        },
        "params": {
            "binding_keep_frac": args.binding_keep_frac,
            "binding_seed": args.binding_seed,
            "num_valid": args.num_valid,
            "filter_shuffle_seed": FILTER_SHUFFLE_SEED,
            "split_shuffle_seed": SPLIT_SHUFFLE_SEED,
        },
        "outputs": {
            "valid": {"links": str(val_links), "seqs": str(val_seqs), "n_links": len(validation)},
            "train_filtered": {
                "links": str(train_links),
                "seqs": str(train_seqs),
                "n_links": len(train_filtered),
            },
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
