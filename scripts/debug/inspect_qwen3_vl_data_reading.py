from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data import (  # noqa: E402
    BioSeqQwenDataCollator,
    BioSeqRecord,
    BioSeqViewSampler,
    PpiArrowSourceConfig,
    default_source_configs,
    source_from_config,
)
from dllm.pipelines.qwen3_vl_arch.data.records import CHAIN_ROLE_TO_ID, TASK_TYPE_TO_ID  # noqa: E402


FIXED_CONTEXT_ROLE_NAMES = {
    "antigen",
    "peptide",
    "epitope",
    "mhc",
    "pmhc",
    "hla",
}


def is_fixed_context_role(role: str) -> bool:
    normalized = role.strip().lower()
    return normalized in FIXED_CONTEXT_ROLE_NAMES or normalized.startswith("mhc") or normalized.startswith("hla")


def batches(records: list[BioSeqRecord], batch_size: int) -> Iterable[list[BioSeqRecord]]:
    for offset in range(0, len(records), batch_size):
        yield records[offset : offset + batch_size]


def context_role_ids() -> set[int]:
    return {
        role_id
        for role, role_id in CHAIN_ROLE_TO_ID.items()
        if is_fixed_context_role(role)
    }


def mask_count_for_role_ids(batch: dict[str, torch.Tensor], mask_name: str, role_ids: set[int]) -> int:
    role_mask = torch.zeros_like(batch["chain_role_ids"], dtype=torch.bool)
    for role_id in role_ids:
        role_mask |= batch["chain_role_ids"].eq(role_id)
    return int((batch[mask_name] & role_mask).sum().item())


def summarize_records(records: list[BioSeqRecord]) -> dict[str, object]:
    role_counter: Counter[str] = Counter()
    task_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    chain_count_counter: Counter[int] = Counter()
    metadata_targets_counter: Counter[str] = Counter()
    max_chain_length = 0
    max_total_residues = 0
    unknown_roles: Counter[str] = Counter()
    unknown_tasks: Counter[str] = Counter()

    for record in records:
        task_counter[record.task_type] += 1
        source_counter[record.source] += 1
        chain_count_counter[len(record.chains)] += 1
        metadata_targets_counter[str(record.metadata.get("targets", "<missing>"))] += 1
        total_residues = 0
        for chain in record.chains:
            role_counter[chain.role] += 1
            total_residues += len(chain.sequence)
            max_chain_length = max(max_chain_length, len(chain.sequence))
            if chain.role not in CHAIN_ROLE_TO_ID:
                unknown_roles[chain.role] += 1
        max_total_residues = max(max_total_residues, total_residues)
        if record.task_type not in TASK_TYPE_TO_ID:
            unknown_tasks[record.task_type] += 1

    return {
        "records": len(records),
        "tasks": dict(task_counter),
        "sources": dict(source_counter),
        "roles": dict(role_counter),
        "chain_counts": dict(chain_count_counter),
        "metadata_targets_top": dict(metadata_targets_counter.most_common(5)),
        "max_chain_length": max_chain_length,
        "max_total_residues": max_total_residues,
        "unknown_roles": dict(unknown_roles),
        "unknown_tasks": dict(unknown_tasks),
    }


def check_full_denoise(records: list[BioSeqRecord], max_chain_length: int | None, batch_size: int) -> dict[str, object]:
    collator = BioSeqQwenDataCollator(
        view_sampler=BioSeqViewSampler(allowed_views=["full_denoise"]),
        max_chain_length=max_chain_length,
    )
    view_counter: Counter[str] = Counter()
    loss_tokens = 0
    fixed_context_tokens = 0
    empty_loss_batches = 0
    context_loss_tokens = 0
    context_ids = context_role_ids()
    errors: list[str] = []

    for batch_records in batches(records, batch_size):
        try:
            batch = collator(batch_records)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        view_counter.update(batch["view_names"])
        loss_tokens += int(batch["diffusion_loss_mask"].sum().item())
        fixed_context_tokens += int(batch["fixed_context_mask"].sum().item())
        context_loss_tokens += mask_count_for_role_ids(batch, "diffusion_loss_mask", context_ids)
        per_example_loss = batch["diffusion_loss_mask"].flatten(1).sum(dim=1)
        empty_loss_batches += int(per_example_loss.eq(0).sum().item())

    return {
        "views": dict(view_counter),
        "loss_tokens": loss_tokens,
        "fixed_context_tokens": fixed_context_tokens,
        "context_loss_tokens": context_loss_tokens,
        "empty_loss_examples": empty_loss_batches,
        "errors": errors[:5],
    }


def check_default_views(records: list[BioSeqRecord], max_chain_length: int | None, batch_size: int) -> dict[str, object]:
    collator = BioSeqQwenDataCollator(
        view_sampler=BioSeqViewSampler(seed=17),
        max_chain_length=max_chain_length,
    )
    view_counter: Counter[str] = Counter()
    loss_tokens = 0
    fixed_context_tokens = 0
    empty_loss_examples = 0
    errors: list[str] = []

    for batch_records in batches(records, batch_size):
        try:
            batch = collator(batch_records)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        view_counter.update(batch["view_names"])
        loss_tokens += int(batch["diffusion_loss_mask"].sum().item())
        fixed_context_tokens += int(batch["fixed_context_mask"].sum().item())
        per_example_loss = batch["diffusion_loss_mask"].flatten(1).sum(dim=1)
        empty_loss_examples += int(per_example_loss.eq(0).sum().item())

    return {
        "views": dict(view_counter),
        "loss_tokens": loss_tokens,
        "fixed_context_tokens": fixed_context_tokens,
        "empty_loss_examples": empty_loss_examples,
        "errors": errors[:5],
    }


def inspect_source(config, args: argparse.Namespace) -> dict[str, object]:
    source_name = getattr(config, "name", type(config).__name__)
    result: dict[str, object] = {
        "source": source_name,
        "path": str(getattr(config, "path", "")),
        "split": getattr(config, "split", ""),
    }
    try:
        source = source_from_config(config)
        records = list(islice(iter(source), args.limit_per_source))
    except Exception as exc:
        result["fatal_error"] = f"{type(exc).__name__}: {exc}"
        if args.traceback:
            result["traceback"] = traceback.format_exc()
        return result

    result["record_summary"] = summarize_records(records)
    if not records:
        result["issues"] = ["no records yielded"]
        return result

    result["full_denoise_check"] = check_full_denoise(records, args.max_chain_length, args.batch_size)
    result["default_view_check"] = check_default_views(records, args.max_chain_length, args.batch_size)

    issues: list[str] = []
    summary = result["record_summary"]
    assert isinstance(summary, dict)
    if summary.get("unknown_roles"):
        issues.append(f"unknown roles: {summary['unknown_roles']}")
    if summary.get("unknown_tasks"):
        issues.append(f"unknown tasks: {summary['unknown_tasks']}")

    full_check = result["full_denoise_check"]
    assert isinstance(full_check, dict)
    if full_check.get("context_loss_tokens"):
        issues.append(f"context residues in full_denoise loss: {full_check['context_loss_tokens']}")
    if full_check.get("empty_loss_examples"):
        issues.append(f"full_denoise empty-loss examples: {full_check['empty_loss_examples']}")
    if full_check.get("errors"):
        issues.append(f"full_denoise collate errors: {full_check['errors']}")

    default_check = result["default_view_check"]
    assert isinstance(default_check, dict)
    if default_check.get("empty_loss_examples"):
        issues.append(f"default-view empty-loss examples: {default_check['empty_loss_examples']}")
    if default_check.get("errors"):
        issues.append(f"default-view collate errors: {default_check['errors']}")

    result["issues"] = issues
    return result


def build_configs(args: argparse.Namespace):
    configs = default_source_configs(split=args.split, max_records=args.limit_per_source)
    if args.include_ppi_arrow:
        configs.append(PpiArrowSourceConfig("ppi_arrow", split=args.split, max_records=args.limit_per_source))
    if args.only:
        requested = set(args.only)
        configs = [config for config in configs if getattr(config, "name", "") in requested]
    return configs


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect Qwen3-VL BioSeq data loading on real local sources.")
    parser.add_argument("--split", default="train", help="Split to inspect for sources that support splits.")
    parser.add_argument("--limit-per-source", type=int, default=64, help="Number of yielded records to inspect per source.")
    parser.add_argument("--batch-size", type=int, default=8, help="Small diagnostic collator batch size.")
    parser.add_argument("--max-chain-length", type=int, default=512, help="Tokenizer truncation length per chain. Use -1 for no truncation.")
    parser.add_argument("--include-ppi-arrow", action="store_true", help="Also inspect optional PPI Arrow dataset.")
    parser.add_argument("--only", nargs="*", help="Restrict to source names such as oas ots nanobody processed_v2.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--traceback", action="store_true", help="Include tracebacks for fatal source errors.")
    args = parser.parse_args()

    if args.max_chain_length is not None and args.max_chain_length < 0:
        args.max_chain_length = None

    reports = [inspect_source(config, args) for config in build_configs(args)]
    if args.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"\n## {report['source']} [{report.get('split')}]")
            print(f"path: {report.get('path')}")
            if "fatal_error" in report:
                print(f"FATAL: {report['fatal_error']}")
                if "traceback" in report:
                    print(report["traceback"])
                continue
            print(json.dumps(report["record_summary"], indent=2, sort_keys=True))
            print("full_denoise:", json.dumps(report["full_denoise_check"], sort_keys=True))
            print("default_views:", json.dumps(report["default_view_check"], sort_keys=True))
            issues = report.get("issues") or []
            print("issues:", "none" if not issues else json.dumps(issues, ensure_ascii=False))

    has_issues = any(report.get("fatal_error") or report.get("issues") for report in reports)
    return 1 if has_issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
