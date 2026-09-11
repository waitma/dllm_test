"""Stream raw immune sources into prepared semantic JSONL shards.

Run with::

    python scripts/data/preprocess_immune_dataset.py --config configs/data/immune_v3.yaml

This stage performs parsing, canonical record construction, decontamination,
and length filtering once before training. It deliberately does not render
grammar tokens or create model-ready token caches.
"""

from __future__ import annotations

import csv
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..registry import SourceSpec, parse_sources, source_spec, source_split_path
from ..sources import COMPLETION_SOURCES, row_to_record

from ..profiles import add_tcr_region_lengths, empty_tcr_region_lengths, fallback_region_profile, finalize_region_profile, profile_digest, profile_summary
from .filters import BLOCKLIST_NAMES, build_filters, filter_reason, load_blocklists
from .reports import add_filter_drop, new_source_stats, totals
from .validators import SCHEMA_VERSION, add_schema_version, validate_prepared_row
from .writers import JsonlShardWriter, atomic_json_dump

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - optional outside the training environment.
    OmegaConf = None  # type: ignore[assignment]

try:
    import yaml
except ImportError:  # pragma: no cover - CLI reports a clearer dependency error.
    yaml = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PreprocessConfig:
    sources: list[dict[str, Any]]
    blocklists: dict[str, str | None]
    max_protein_length: int = 1024
    max_length: int = 1024
    shard_size: int = 100_000
    splits: tuple[str, ...] = ("train",)
    schema_version: str = SCHEMA_VERSION
    region_profile_seed: int = 42
    region_profile_source_policy: str = "current_complete_tcr_rows"
    #: Sources whose receptor chains get completed from the region profile.
    #: Defaults to tcr_repertoire alone so an existing config reproduces the
    #: dataset it produced before; widening it is an explicit, versioned choice
    #: because it changes every affected record's layout.
    completion_sources: tuple[str, ...] = ("tcr_repertoire",)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> PreprocessConfig:
        source_value = config.get("sources")
        if source_value is None:
            raise ValueError("Preprocessing config requires 'sources'")
        if isinstance(source_value, Mapping):
            source_items = [{"name": name, **(value if isinstance(value, Mapping) else {"path": value})} for name, value in source_value.items()]
        else:
            source_items = [dict(item) for item in source_value]
        blocklists = dict(config.get("blocklists") or {})
        missing = [name for name in BLOCKLIST_NAMES if name not in blocklists]
        if missing:
            raise KeyError(f"Preprocessing config missing blocklists: {missing}")
        budget = config.get("budget") or {}
        output = config.get("output") or {}
        region_profile = config.get("tcr_region_profile") or {}
        return cls(
            sources=source_items,
            blocklists={name: blocklists[name] for name in BLOCKLIST_NAMES},
            max_protein_length=int(budget.get("max_protein_length", config.get("max_protein_length", 1024))),
            max_length=int(budget.get("max_length", config.get("max_length", 1024))),
            shard_size=int(output.get("shard_size", config.get("shard_size", 100_000))),
            splits=tuple(str(item) for item in (config.get("splits") or ("train",))),
            schema_version=str(config.get("schema_version", SCHEMA_VERSION)),
            region_profile_seed=int(region_profile.get("seed", 42)),
            region_profile_source_policy=str(region_profile.get("source_policy", "current_complete_tcr_rows")),
            completion_sources=_completion_sources(region_profile.get("completion_sources")),
        )


def _completion_sources(value: Any) -> tuple[str, ...]:
    if value is None:
        return ("tcr_repertoire",)
    names = tuple(str(item).strip() for item in value)
    if not all(names):
        raise ValueError("tcr_region_profile.completion_sources must not contain empty names")
    unknown = [name for name in names if name not in COMPLETION_SOURCES]
    if unknown:
        raise ValueError(
            f"tcr_region_profile.completion_sources has non-completable sources {unknown}; "
            f"choose from {list(COMPLETION_SOURCES)}"
        )
    return names


def load_preprocess_config(path: str | Path) -> PreprocessConfig:
    """Load a plain YAML config without requiring the training stack.

    OmegaConf is used when available so existing project conventions continue
    to work. The standalone preprocessing CLI also supports PyYAML-only
    environments, which keeps data preparation independent of transformers and
    the training launcher.
    """
    if OmegaConf is not None:
        loaded = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    elif yaml is not None:
        with Path(path).open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    else:
        raise RuntimeError("Reading preprocessing YAML requires omegaconf or pyyaml")
    if not isinstance(loaded, Mapping):
        raise TypeError("Preprocessing config root must be a mapping")
    normalized = {str(key): value for key, value in loaded.items()}
    return PreprocessConfig.from_mapping(normalized)


def _iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson", ".json"}:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number} JSON row must be an object")
                yield row
        return
    csv.field_size_limit(2**31 - 1)
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def _configured_sources(config: PreprocessConfig, selected: str | None) -> list[SourceSpec]:
    selected_names = set(parse_sources(selected)) if selected else None
    specs: list[SourceSpec] = []
    for item in config.sources:
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError(f"Source entry missing name: {item!r}")
        if selected_names is not None and name not in selected_names:
            continue
        specs.append(source_spec(name, path=item.get("path"), weight=float(item.get("weight", 1.0))))
    if not specs:
        raise ValueError(f"No configured sources selected by {selected!r}")
    return specs


def _split_path(spec: SourceSpec, split: str) -> Path:
    path = source_split_path(spec, split)
    if path.is_file():
        return path
    if spec.path.is_dir():
        for candidate in (spec.path / f"{split}.jsonl", spec.path / f"{split}.ndjson", spec.path / f"{split}.json"):
            if candidate.is_file():
                return candidate
    return path


def _preflight_inputs(specs: list[SourceSpec], splits: list[str]) -> None:
    if not splits:
        raise ValueError("At least one split must be selected")
    for split in splits:
        if not split.strip():
            raise ValueError("Split names must not be empty")
        for spec in specs:
            path = _split_path(spec, split)
            if not path.is_file():
                raise FileNotFoundError(f"Raw immune source not found: {path}")


def _process_source(
    spec: SourceSpec,
    split: str,
    output_dir: Path,
    config: PreprocessConfig,
    blocklists: dict[str, set[str]],
    mix: list[str],
    max_rows: int | None,
    dry_run: bool,
    region_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = _split_path(spec, split)
    if not path.is_file():
        raise FileNotFoundError(f"Raw immune source not found: {path}")
    stats = new_source_stats(spec.name, split, str(path))
    filters = build_filters(spec.name, mix, blocklists, config.max_protein_length, config.max_length)
    writer = None if dry_run else JsonlShardWriter(output_dir / split, split, spec.name, config.shard_size)
    # Withholding the profile is what disables completion for a source, so only
    # configured sources receive it.
    source_profile = region_profile if spec.name in config.completion_sources else None
    try:
        for raw_index, row in enumerate(_iter_rows(path)):
            if max_rows is not None and raw_index >= max_rows:
                break
            stats["raw_rows"] += 1
            record = row_to_record(
                spec.name,
                dict(row),
                split=split,
                weight=spec.weight,
                profile=source_profile,
                row_index=raw_index,
            )
            if record is None:
                stats["dropped_schema"] += 1
                continue
            stats["converted_rows"] += 1
            reason = filter_reason(record, filters)
            if reason is not None:
                add_filter_drop(stats, reason)
                continue
            prepared = add_schema_version(record.to_dict())
            validate_prepared_row(prepared)
            stats["kept_rows"] += 1
            if writer is not None:
                writer.write(prepared)
        if writer is not None:
            writer.close()
            stats["shards"] = [
                {**shard, "path": f"{split}/{shard['path']}"}
                for shard in writer.published
            ]
    except BaseException:
        if writer is not None:
            writer.abort()
        raise
    return stats


def preprocess_dataset(
    config: PreprocessConfig | Mapping[str, Any],
    output_dir: str | Path,
    *,
    splits: Iterable[str] | None = None,
    source: str | None = None,
    max_rows: int | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Preprocess selected source/splits and publish reports after all shards finish."""
    if not isinstance(config, PreprocessConfig):
        config = PreprocessConfig.from_mapping(config)
    if config.schema_version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version {config.schema_version!r}")
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if resume:
        raise NotImplementedError(
            "resume is not implemented; use --overwrite only after verifying the output directory"
        )
    output = Path(output_dir)
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output}")
    manifest_path = output / "dataset_manifest.json"
    if output.exists() and any(output.iterdir()) and not (overwrite or dry_run):
        raise FileExistsError(f"Output directory is not empty: {output}; use --overwrite")
    specs = _configured_sources(config, source)
    source_names = [spec.name for spec in specs]
    configured_splits = config.splits or ("train",)
    requested_splits = configured_splits if splits is None else splits
    split_names: list[str] = []
    for split in requested_splits:
        normalized = str(split).strip()
        if normalized and normalized not in split_names:
            split_names.append(normalized)
    _preflight_inputs(specs, split_names)
    if overwrite and output.exists() and not dry_run:
        for child in output.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output.mkdir(parents=True, exist_ok=True)
    blocklists = load_blocklists(config.blocklists)
    region_profile = None
    # Any configured completion source needs the profile, not just tcr_repertoire:
    # the epitope-conditioned sources carry bare CDR3 loops that are expanded the
    # same way. The learning pass below deliberately calls row_to_record without a
    # profile so that synthesized scaffolding cannot feed back into the profile.
    if any(name in source_names for name in config.completion_sources):
        profile_counts = empty_tcr_region_lengths()
        profile_observations: Counter[str] = Counter()
        profile_sources_seen: list[str] = []
        profile_filters = {
            profile_source: build_filters(
                profile_source, source_names, blocklists,
                config.max_protein_length, config.max_length,
            )
            for profile_source in ("ots", "tcr_native", "tcr_papers")
            if profile_source in source_names
        }
        for profile_source in ("ots", "tcr_native", "tcr_papers"):
            if profile_source not in source_names:
                continue
            profile_spec = next(spec for spec in specs if spec.name == profile_source)
            for profile_split in split_names:
                profile_path = _split_path(profile_spec, profile_split)
                for profile_index, profile_row in enumerate(_iter_rows(profile_path)):
                    if max_rows is not None and profile_index >= max_rows:
                        break
                    profile_observations["raw_rows"] += 1
                    profile_record = row_to_record(
                        profile_source,
                        dict(profile_row),
                        split=profile_split,
                        weight=profile_spec.weight,
                        row_index=profile_index,
                    )
                    if profile_record is None:
                        profile_observations["dropped_schema"] += 1
                        continue
                    profile_observations["converted_rows"] += 1
                    if filter_reason(profile_record, profile_filters[profile_source]) is not None:
                        profile_observations["dropped_filters"] += 1
                        continue
                    add_tcr_region_lengths(profile_counts, profile_record)
                    profile_observations["kept_rows"] += 1
                    if profile_source not in profile_sources_seen:
                        profile_sources_seen.append(profile_source)
        if not profile_filters:
            # No complete-TCR source is configured at all (e.g. a tcr_repertoire-only
            # run). There is nothing to learn from, so fall back to the shared
            # default profile and record that policy in the manifest instead of
            # failing. Production configs always carry ots/tcr_native/tcr_papers,
            # so they take the strict branch below.
            region_profile = fallback_region_profile(seed=config.region_profile_seed)
            region_profile["source_policy"] = "default_profile_no_complete_tcr_source_configured"
            region_profile["observations"] = dict(profile_observations)
        else:
            if not profile_observations["kept_rows"]:
                raise ValueError("Cannot build tcr_repertoire completion profile: no complete TCR records after filters")
            region_profile = finalize_region_profile(
                profile_counts,
                seed=config.region_profile_seed,
                sources=profile_sources_seen,
                source_policy=config.region_profile_source_policy,
                observations=dict(profile_observations),
            )
    all_stats: list[dict[str, Any]] = []
    manifest_splits: dict[str, Any] = {}
    for split in split_names:
        split_stats: list[dict[str, Any]] = []
        for spec in specs:
            stats = _process_source(
                spec,
                split,
                output,
                config,
                blocklists,
                source_names,
                max_rows,
                dry_run,
                region_profile,
            )
            split_stats.append(stats)
            all_stats.append(stats)
        manifest_splits[split] = {
            "records": sum(item["kept_rows"] for item in split_stats),
            "sources": {item["source"]: {"records": item["kept_rows"], "raw_rows": item["raw_rows"]} for item in split_stats},
            "shards": [shard for item in split_stats for shard in item["shards"]],
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "dry_run": dry_run,
        "splits": all_stats,
        "totals": totals(all_stats),
    }
    if not dry_run:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "format": "jsonl",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": source_names,
            "splits": manifest_splits,
            "filter_names": list(BLOCKLIST_NAMES) + (["quality.homotypic_pair"] if {"oas", "ots"} & set(source_names) else []),
            "budget": {"max_protein_length": config.max_protein_length, "max_length": config.max_length},
            "shard_size": config.shard_size,
            "tcr_region_profile": ({
                "profile": region_profile,
                "digest": profile_digest(region_profile),
                "summary": profile_summary(region_profile),
                "observations": region_profile.get("observations", {}),
                "source_policy": region_profile.get("source_policy"),
                # Which sources actually had their receptor chains completed;
                # without this the layout of a prepared record cannot be
                # reconstructed from the manifest alone.
                "completion_sources": [name for name in config.completion_sources if name in source_names],
            } if region_profile is not None else None),
        }
        atomic_json_dump(output / "filter_report.json", report)
        atomic_json_dump(output / "validation_report.json", {"schema_version": SCHEMA_VERSION, "status": "passed", "records": report["totals"]["kept_rows"]})
        atomic_json_dump(output / "schema.json", {"schema_version": SCHEMA_VERSION, "format": "semantic_jsonl", "token_cache": False})
        # Publish the manifest last: the loader must never observe an incomplete
        # source/split shard set.
        atomic_json_dump(manifest_path, manifest)
    return {"manifest": str(manifest_path) if not dry_run else None, "report": report, "splits": manifest_splits}


__all__ = ["PreprocessConfig", "load_preprocess_config", "preprocess_dataset"]
