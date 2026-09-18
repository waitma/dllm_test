"""Count the published prepared immune corpus, grouped by manifest source.

Usage::

    python scripts/count_immune_mix.py --prepared-data-dir data/prepared/immune_v3 \
        --split train --sources oas+trait

By default, read semantic records through the immune_llada prepared loader.
--counts-only reads manifest counts without opening shards; it does not verify
shard contents. Raw counts always come from preprocessing's manifest, not CSVs.
Generated-chain residues exclude antigen/mhc/pmhc/hla/peptide/epitope roles;
these are sequence characters (including '.'/'-'), not grammar/loss tokens.
Counts include duplicate records, ignore record weights, and describe one pass
over the selected corpus, not a sampler's training exposure or loss share.

--sources selects from an already prepared mix; it never recomputes supersession
or filters. Omit it to use all sources in the split. Raw positional arguments,
field=value overrides and trainer flags are retired; prepare a separate dataset
to compare another filtering policy. --json writes the report to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data.dataset import load_prepared_dataset
from dllm.pipelines.immune_llada.data.preprocessing.validators import validate_manifest
from dllm.pipelines.immune_llada.data.registry import parse_sources

_FIXED_ROLES = {"antigen", "mhc", "pmhc", "hla", "peptide", "epitope"}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def nonnegative_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer, got {value!r}")
    return value


def _source_names(value: str) -> list[str]:
    if any(not part.strip() for part in value.split("+")):
        raise ValueError("--sources requires nonempty source names separated by '+'")
    return parse_sources(value)


def load_manifest(dataset_dir: Path) -> dict[str, Any]:
    """Shared manifest checks for the three counting CLIs; never open raw data."""
    manifest = read_json(dataset_dir / "dataset_manifest.json")
    validate_manifest(manifest)
    names = manifest.get("sources")
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        raise ValueError("Dataset manifest requires a nonempty sources list")
    if parse_sources(names) != names:
        raise ValueError("Manifest sources must be unique canonical registry names")
    for split, info in manifest["splits"].items():
        if not isinstance(split, str) or not split.strip() or not isinstance(info, dict):
            raise ValueError("Manifest splits must map nonempty names to objects")
        sources = info.get("sources")
        shards = info.get("shards")
        if not isinstance(sources, dict) or not sources or not set(sources) <= set(names):
            raise ValueError(f"Manifest {split}: missing or unknown source statistics")
        if not isinstance(shards, list):
            raise TypeError(f"Manifest {split}: shards must be a list")
        shard_counts = dict.fromkeys(sources, 0)
        paths: set[str] = set()
        for shard in shards:
            if not isinstance(shard, dict) or shard.get("source") not in sources:
                raise ValueError(f"Manifest {split}: shard has an unknown source")
            path = shard.get("path")
            if not isinstance(path, str) or not path or path in paths:
                raise ValueError(f"Manifest {split}: missing or duplicate shard path {path!r}")
            if shard.get("split", split) != split:
                raise ValueError(f"Manifest {split}: shard {path} belongs to another split")
            paths.add(path)
            shard_counts[shard["source"]] += nonnegative_count(shard.get("records"), f"{split}/{path} records")
        total = 0
        for source, stats in sources.items():
            if not isinstance(stats, dict):
                raise TypeError(f"Manifest {split}/{source}: source statistics must be an object")
            records = nonnegative_count(stats.get("records"), f"{split}/{source} records")
            raw = nonnegative_count(stats.get("raw_rows"), f"{split}/{source} raw_rows")
            if records > raw or records != shard_counts[source]:
                raise ValueError(f"Manifest {split}/{source}: inconsistent raw/record/shard counts")
            total += records
        if nonnegative_count(info.get("records"), f"{split} records") != total:
            raise ValueError(f"Manifest {split}: total does not match source record counts")
    return manifest


def selected_sources(manifest: dict[str, Any], split: str, value: str | None) -> list[str]:
    info = manifest["splits"].get(split)
    if info is None:
        raise ValueError(f"Prepared split {split!r} is not present in dataset_manifest.json")
    available = [name for name in manifest["sources"] if name in info["sources"]]
    selected = _source_names(value) if value is not None else available
    missing = [name for name in selected if name not in available]
    if missing:
        raise ValueError(f"Sources {missing!r} are not present in split {split!r}; available: {available}")
    return selected


def load_source_dataset(root: Path, manifest: dict[str, Any], split: str, source: str):
    declared = manifest["splits"][split]["sources"][source]["records"]
    if declared == 0:
        # The prepared loader rejects empty selections. A published empty source
        # normally has no shards; an empty shard still needs its existence checked.
        for shard in manifest["splits"][split]["shards"]:
            if shard["source"] == source:
                path = root / shard["path"]
                with path.open(encoding="utf-8") as handle:
                    if any(line.strip() for line in handle):
                        raise ValueError(f"Manifest declares an empty shard containing records: {path}")
        return None
    dataset = load_prepared_dataset(root, split=split, source=source)
    if len(dataset) != declared:
        raise ValueError(
            f"Manifest record count mismatch for {split}/{source}: "
            f"manifest={declared}, readable={len(dataset)}"
        )
    return dataset


def count_mix(
    dataset_dir: str | Path, *, split: str = "train", sources: str | None = None,
    counts_only: bool = False,
) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest = load_manifest(root)
    selected = selected_sources(manifest, split, sources)
    rows: list[dict[str, Any]] = []
    for source in selected:
        declared = manifest["splits"][split]["sources"][source]
        row = {"source": source, "raw_rows": declared["raw_rows"], "records": declared["records"]}
        if not counts_only:
            dataset = load_source_dataset(root, manifest, split, source)
            residues = generated_residues = chains = 0
            if dataset is not None:
                for record in dataset.iter_records():
                    chains += len(record.chains)
                    residues += sum(len(chain.sequence) for chain in record.chains)
                    generated_residues += sum(
                        len(chain.sequence) for chain in record.chains
                        if chain.role.lower() not in _FIXED_ROLES
                    )
                del dataset
            row.update(residues=residues, generated_chain_residues=generated_residues, chains=chains)
        rows.append(row)
    keys = ["raw_rows", "records"]
    if not counts_only:
        keys += ["residues", "generated_chain_residues", "chains"]
    return {
        "prepared_data_dir": str(root), "split": split, "sources": selected,
        "basis": "manifest_only" if counts_only else "prepared_records",
        "raw_rows_basis": "preprocessing_manifest",
        "budget": manifest.get("budget"), "rows": rows,
        "totals": {key: sum(row[key] for row in rows) for key in keys},
    }


def percentage(value: float, total: float) -> float:
    return 100 * value / total if total else 0.0


def _print_report(result: dict[str, Any]) -> None:
    print(f"prepared={result['prepared_data_dir']}  split={result['split']}  basis={result['basis']}")
    print("Raw counts: preprocessing manifest. Record weights and sampler exposure are not applied.")
    if result["basis"] == "manifest_only":
        print("Manifest-only: shard contents were not read or verified.")
    total = result["totals"]
    for row in result["rows"]:
        text = (
            f"{row['source']:16s} raw={row['raw_rows']:>10,} records={row['records']:>10,} "
            f"rec%={percentage(row['records'], total['records']):6.2f}"
        )
        if result["basis"] == "prepared_records":
            text += (
                f" residues={row['residues']:,} gen_chain_res={row['generated_chain_residues']:,}"
                f" chains={row['chains']:,} res%={percentage(row['residues'], total['residues']):.2f}"
                f" gen_res%={percentage(row['generated_chain_residues'], total['generated_chain_residues']):.2f}"
            )
        print(text)
    print("TOTAL " + "  ".join(f"{key}={value:,}" for key, value in total.items()))


def counting_parser(description: str | None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description, allow_abbrev=False)
    parser.add_argument("--prepared-data-dir", required=True, type=Path, help="published semantic dataset directory")
    parser.add_argument("--split", default="train", help="exact manifest split name (default: train)")
    parser.add_argument("--sources", help="registry source names separated by '+'; default: every source in the split")
    parser.add_argument("--json", action="store_true", dest="as_json", help="emit JSON to stdout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = counting_parser(__doc__)
    parser.add_argument("--counts-only", action="store_true", help="read manifest counts only; do not verify shards")
    args = parser.parse_args(argv)
    try:
        result = count_mix(args.prepared_data_dir, split=args.split, sources=args.sources, counts_only=args.counts_only)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
