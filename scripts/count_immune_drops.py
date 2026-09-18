"""Report first-failure drops from a published immune LLaDA prepared dataset.

Usage::

    python scripts/count_immune_drops.py --prepared-data-dir data/prepared/immune_v3 \
        --split train --sources oas+trait

Read dataset_manifest.json and filter_report.json only; shards and raw inputs
are not scanned. Report counts are checked against the manifest. --sources
selects an existing prepared mix, without recomputing supersession or filters.

filter_reasons records the FIRST rejection in the offline preprocessing pass.
The current order is schema conversion, source-specific dedup/decontamination,
length budgets, then quality checks. A row failing decontamination AND length
is attributed only to decontamination. This differs from the retired script's
schema -> length -> supersession -> decontamination stage differences, obtained
by evaluating several raw configurations. No counterfactual stage counts can be
recovered from this report. --scenario and raw/trainer overrides are retired;
prepare a separate dataset to compare another policy. --json emits to stdout.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dllm.pipelines.immune_llada.data.preprocessing.reports import (
    totals as report_totals,
)
from scripts.count_immune_mix import (
    counting_parser,
    load_manifest,
    nonnegative_count,
    read_json,
    selected_sources,
)

_COUNT_KEYS = ("raw_rows", "converted_rows", "kept_rows", "dropped_schema", "dropped_filters", "errors")


def load_drop_report(dataset_dir: str | Path, *, split: str = "train", sources: str | None = None) -> dict[str, Any]:
    root = Path(dataset_dir)
    manifest = load_manifest(root)
    selected = selected_sources(manifest, split, sources)
    report = read_json(root / "filter_report.json")
    if report.get("schema_version") != manifest["schema_version"]:
        raise ValueError("filter_report.json and dataset_manifest.json have different schema versions")
    if report.get("dry_run") is not False:
        raise ValueError("filter_report.json must declare dry_run=false for a published dataset")
    stats_list = report.get("splits")
    if not isinstance(stats_list, list):
        raise TypeError("filter_report.json requires a splits list")
    stats_by_source: dict[tuple[str, str], dict[str, Any]] = {}
    for stats in stats_list:
        if not isinstance(stats, dict):
            raise TypeError("filter_report.json source statistics must be objects")
        split_name, source = stats.get("split"), stats.get("source")
        if not isinstance(split_name, str) or not isinstance(source, str):
            raise TypeError("filter_report.json requires source and split names")
        key = (split_name, source)
        if key in stats_by_source:
            raise ValueError(f"Duplicate filter report entry for {split_name}/{source}")
        declared = manifest["splits"].get(split_name, {}).get("sources", {}).get(source)
        if declared is None:
            raise ValueError(f"Filter report entry absent from manifest: {split_name}/{source}")
        label = f"{split_name}/{source}"
        counts = {name: nonnegative_count(stats.get(name), f"{label} {name}") for name in _COUNT_KEYS}
        if counts["errors"]:
            raise ValueError(f"Filter report contains preprocessing errors for {label}")
        if (
            counts["raw_rows"] != counts["converted_rows"] + counts["dropped_schema"]
            or counts["converted_rows"] != counts["kept_rows"] + counts["dropped_filters"]
        ):
            raise ValueError(f"Inconsistent first-failure counts for {label}")
        reasons = stats.get("filter_reasons")
        if not isinstance(reasons, dict) or not all(isinstance(name, str) and name for name in reasons):
            raise ValueError(f"filter_reasons for {label} must map nonempty names to counts")
        reasons = {name: nonnegative_count(count, f"{label} {name}") for name, count in reasons.items()}
        if sum(reasons.values()) != counts["dropped_filters"]:
            raise ValueError(f"First-failure reason total does not match dropped_filters for {label}")
        if declared["records"] != counts["kept_rows"] or declared["raw_rows"] != counts["raw_rows"]:
            raise ValueError(f"Manifest/report raw or kept count mismatch for {label}")
        stats_by_source[key] = {"source": source, **counts, "filter_reasons": reasons}
    expected = {(name, source) for name, info in manifest["splits"].items() for source in info["sources"]}
    if set(stats_by_source) != expected:
        raise ValueError(f"filter_report.json has no stats for: {sorted(expected - set(stats_by_source))}")
    declared_totals = report.get("totals")
    if not isinstance(declared_totals, dict):
        raise TypeError("filter_report.json requires totals")
    declared_totals = {name: nonnegative_count(declared_totals.get(name), f"report totals {name}") for name in _COUNT_KEYS}
    if declared_totals != report_totals(list(stats_by_source.values())):
        raise ValueError("filter_report.json totals do not match its source/split entries")

    rows = [stats_by_source[(split, source)] for source in selected]
    totals: dict[str, Any] = report_totals(rows)
    reason_totals: dict[str, int] = {}
    for row in rows:
        for reason, count in row["filter_reasons"].items():
            reason_totals[reason] = reason_totals.get(reason, 0) + count
    totals["filter_reasons"] = dict(sorted(reason_totals.items()))
    return {
        "prepared_data_dir": str(root), "split": split, "sources": selected,
        "attribution": "first_failure_from_offline_preprocessing",
        "basis": "manifest_and_filter_report", "budget": manifest.get("budget"),
        "rows": rows, "totals": totals,
    }


def _print_report(result: dict[str, Any]) -> None:
    print(f"prepared={result['prepared_data_dir']}  split={result['split']}  sources={'+'.join(result['sources'])}")
    print("Drop attribution: FIRST FAILURE recorded offline; not old length-first stage differences.")
    print("Manifest/report counts only; raw inputs and shard contents were not scanned.\n")
    for row in result["rows"]:
        print(
            f"{row['source']:16s} raw={row['raw_rows']:>9,} converted={row['converted_rows']:>9,} "
            f"kept={row['kept_rows']:>9,} schema_drop={row['dropped_schema']:>8,} "
            f"filter_drop={row['dropped_filters']:>8,}"
        )
        for reason, count in sorted(row["filter_reasons"].items()):
            print(f"  {reason}: {count:,}")
    totals = result["totals"]
    print("\nTOTAL " + "  ".join(f"{key}={totals[key]:,}" for key in _COUNT_KEYS))
    print(f"first-failure reasons: {totals['filter_reasons']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = counting_parser(__doc__)
    args = parser.parse_args(argv)
    try:
        result = load_drop_report(args.prepared_data_dir, split=args.split, sources=args.sources)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
