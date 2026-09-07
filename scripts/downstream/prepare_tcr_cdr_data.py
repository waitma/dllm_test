#!/usr/bin/env python3
"""Rebuild the TCR CDR-infilling 10-fold set from OTS holdout.

The original AirGen files under ``data/downstream/cdr_infilling/tcr/`` were
broken symlinks. This recreates the documented protocol:

  OTS holdout → locate each IMGT CDR on the full chain → sklearn KFold(10, seed=42)

Output schema matches ``data/downstream/README.md`` §1b.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import KFold

PROJECT_ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dllm.pipelines.qwen3_vl_arch.data.records import (  # noqa: E402
    REGION_ORDER,
    BioSeqChain,
    is_valid_protein_sequence,
    normalize_sequence,
)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "ots_paired_clean" / "final" / "holdout.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "downstream" / "cdr_infilling" / "tcr"
MODES = (
    "cdr_beta1",
    "cdr_beta2",
    "cdr_beta3",
    "cdr_alpha1",
    "cdr_alpha2",
    "cdr_alpha3",
)
_MODE_TO_CDR = {
    "cdr_beta1": "CDR1",
    "cdr_beta2": "CDR2",
    "cdr_beta3": "CDR3",
    "cdr_alpha1": "CDR1",
    "cdr_alpha2": "CDR2",
    "cdr_alpha3": "CDR3",
}


def _is_beta(row: dict[str, str], prefix: str) -> bool:
    anarci = str(row.get(f"{prefix}_anarci_type", "")).strip().upper()
    chain_type = str(row.get(f"{prefix}_type", "")).strip().lower()
    return anarci == "B" or chain_type in {"beta", "trb"}


def _regions(row: dict[str, str], prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for region in REGION_ORDER:
        seq = normalize_sequence(row.get(f"{prefix}_{region}"))
        if seq:
            out[region] = seq
    return out


def _locate_cdr(sequence: str, regions: dict[str, str], cdr_name: str) -> tuple[list[int], str] | None:
    """Return inclusive ``[start, end]`` plus the CDR substring, or None."""

    if not sequence or cdr_name not in regions:
        return None
    chain = BioSeqChain(sequence, "tcr_beta", regions=regions)
    span = chain.region_span(cdr_name)
    if span is not None:
        start, end = span
        cdr = sequence[start:end]
        if cdr == regions[cdr_name]:
            return [start, end - 1], cdr
    cdr = regions[cdr_name]
    start = sequence.find(cdr)
    if start < 0:
        return None
    return [start, start + len(cdr) - 1], cdr


def _assign_chains(row: dict[str, str]) -> tuple[dict[str, object], dict[str, object]] | None:
    chain1 = normalize_sequence(row.get("cleaned_chain1_seq"))
    chain2 = normalize_sequence(row.get("cleaned_chain2_seq"))
    if not is_valid_protein_sequence(chain1) or not is_valid_protein_sequence(chain2):
        return None
    first = {
        "seq": chain1,
        "regions": _regions(row, "chain1"),
        "is_beta": _is_beta(row, "chain1"),
    }
    second = {
        "seq": chain2,
        "regions": _regions(row, "chain2"),
        "is_beta": _is_beta(row, "chain2"),
    }
    if first["is_beta"] == second["is_beta"]:
        return None
    beta, alpha = (first, second) if first["is_beta"] else (second, first)
    return alpha, beta


def load_holdout_pairs(path: Path) -> list[dict[str, object]]:
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    pairs: list[dict[str, object]] = []
    skipped = Counter()
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            assigned = _assign_chains(row)
            if assigned is None:
                skipped["bad_chains"] += 1
                continue
            alpha, beta = assigned
            record = {
                "alpha_chain_seq": alpha["seq"],
                "beta_chain_seq": beta["seq"],
            }
            found_any = False
            for mode, cdr_name in _MODE_TO_CDR.items():
                chain = beta if "beta" in mode else alpha
                located = _locate_cdr(str(chain["seq"]), chain["regions"], cdr_name)
                if located is None:
                    skipped[f"missing_{mode}"] += 1
                    continue
                pos, seq = located
                record[f"{mode}_seq"] = seq
                record[f"{mode}_pos"] = pos
                found_any = True
            if not found_any:
                skipped["no_cdr"] += 1
                continue
            pairs.append(record)
    print(f"loaded {len(pairs)} paired TCRs from {path}")
    if skipped:
        print("skipped:", dict(skipped))
    return pairs


def write_folds(pairs: list[dict[str, object]], output_dir: Path, n_folds: int, seed: int) -> None:
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    indices = list(range(len(pairs)))
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, list[int]] = {mode: [] for mode in MODES}
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(indices)):
        for mode in MODES:
            rows = []
            for i in test_idx:
                rec = pairs[i]
                if f"{mode}_seq" not in rec:
                    continue
                rows.append(
                    {
                        "alpha_chain_seq": rec["alpha_chain_seq"],
                        "beta_chain_seq": rec["beta_chain_seq"],
                        f"{mode}_seq": rec[f"{mode}_seq"],
                        f"{mode}_pos": rec[f"{mode}_pos"],
                    }
                )
            dest = output_dir / mode / f"fold_{fold}"
            dest.mkdir(parents=True, exist_ok=True)
            out_path = dest / "test.json"
            with out_path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[mode].append(len(rows))
    for mode, fold_ns in counts.items():
        print(f"  {mode}: {sum(fold_ns)} rows across {len(fold_ns)} folds "
              f"(min={min(fold_ns)} max={max(fold_ns)})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir.exists():
        for leftover in args.output_dir.iterdir():
            if leftover.is_symlink() or leftover.name.startswith("._"):
                leftover.unlink()

    pairs = load_holdout_pairs(args.input_csv)
    if len(pairs) < args.n_folds:
        raise SystemExit(f"only {len(pairs)} pairs; cannot make {args.n_folds} folds")
    write_folds(pairs, args.output_dir, args.n_folds, args.seed)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
