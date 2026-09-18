#!/usr/bin/env python3
"""Fail if any corpus sequence contains a character outside ``RESIDUES``.

Why this exists
---------------
``RemapCollator`` (examples/llada/protein_fusion_model.py) maps ESMC grammar ids
into the expanded LLaDA vocabulary. That table only covers the 25 ``RESIDUES``
characters plus the grammar / chainsep / pad specials, so the three remaining
single-character ESMC tokens have no image:

    id 29 '.'   id 30 '-'   id 31 '|'

Hitting one of them raises ``AssertionError: unmapped decoder grammar ids`` from
inside a DataLoader worker. That kills one rank; the others then block in
all-gather until the NCCL watchdog fires at 1800 s and the whole job fails.

On 2026-08-30 this took down ``..._diffusion_immune_v3_4gpu`` at step 42782 --
84 % of the way through a 50k-step run -- because of a *single* stray gap
character in a single cell out of 8.5M corpus rows
(``data/tcr_papers_v2/dataset/train.csv`` row 3046, ``cdr3b`` =
``ASSKVAARVP-TLKLS``). The cell was stripped in-place on 2026-08-31
(``ASSKVAARVPTLKLS``); this script is the gate that keeps it from coming back.

Since HF Trainer replays the same sample order on resume, a leftover gap
crashes at the same step forever and burns the whole ``RetryOptions`` budget.
See examples/llada/PROTEIN_PRETRAIN_PROGRESS.md 4.2.6.

Run this before submitting, and after any corpus rebuild.

Usage
-----
    python scripts/data/assert_residue_alphabet.py              # all 7 sources × 3 splits
    python scripts/data/assert_residue_alphabet.py --split train
    python scripts/data/assert_residue_alphabet.py --split holdout
    python scripts/data/assert_residue_alphabet.py --paths a.csv b.csv

Exit code is 1 if any offending row is found, so it can gate a submit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Must stay in sync with examples/llada/protein_fusion_model.py RESIDUES.
RESIDUES = "LAGVSERTIDPKQNFYMHWCXBUZO"
BAD_CHAR = re.compile(f"[^{RESIDUES}]")
RESIDUE_ONLY = re.compile(f"[{RESIDUES}]+")

# Paths used to rebuild the prepared v3 dataset from raw sources. The trainer
# itself reads prepared semantic JSONL; these CSV paths are only for an offline
# pre-processing preflight.
_OAS = "data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{split}_oas_label.csv"
_STD = {
    "ots": "data/ots_paired_clean/final/{split}.csv",
    "asd_antibody": "downstream/asd/step6_final/antibody/{split}.csv",
    "trait": "downstream/trait/step4_final/{split}.csv",
    "tcr_native": "data/tcr_native/dataset/{split}.csv",
    "tcr_papers_v2": "data/tcr_papers_v2/dataset/{split}.csv",
    # junc80 build (full IMGT junctions), matching configs/data/immune_v4_beta_relation.yaml.
    "tcr_repertoire": "data/tcr_repertoire_junc80/dataset/{split}.csv",
}


def _split_files(split: str) -> dict[str, str]:
    files = {"oas": _OAS.format(split=split)}
    files.update({name: tmpl.format(split=split) for name, tmpl in _STD.items()})
    return files


SPLIT_FILES: dict[str, dict[str, str]] = {
    split: _split_files(split) for split in ("train", "valid", "holdout")
}

SEQ_HINTS = ("seq", "cdr", "chain", "heavy", "light", "alpha", "beta", "peptide", "mhc", "antigen")

# A column counts as sequence data only if most of its non-empty values are
# residue-only; otherwise ids like "chain_id" or "peptide_source" false-positive.
RESIDUE_COLUMN_THRESHOLD = 0.5


def _looks_like_seq_name(col: str) -> bool:
    lc = col.lower()
    return any(hint in lc for hint in SEQ_HINTS)


def scan_file(path: Path) -> tuple[int, int, dict[str, dict[str, int]], list[tuple[int, str, str]]]:
    """Return (n_rows, n_bad_rows, {col: {char: count}}, [(row, col, value)])."""

    df = pd.read_csv(path, dtype=str, low_memory=False)
    hits: dict[str, dict[str, int]] = {}
    examples: list[tuple[int, str, str]] = []
    bad_rows = pd.Series(False, index=df.index)

    for col in (c for c in df.columns if _looks_like_seq_name(c)):
        values = df[col].fillna("")
        nonempty = values[values.str.len() > 0]
        if nonempty.empty:
            continue
        if nonempty.str.fullmatch(RESIDUE_ONLY).mean() < RESIDUE_COLUMN_THRESHOLD:
            continue
        mask = values.str.contains(BAD_CHAR, na=False) & (values.str.len() > 0)
        if not mask.any():
            continue
        chars: dict[str, int] = {}
        for value in values[mask]:
            for ch in set(BAD_CHAR.findall(value)):
                chars[ch] = chars.get(ch, 0) + 1
        hits[col] = chars
        bad_rows |= mask
        for row in df.index[mask][:5]:
            examples.append((int(row), col, str(df.at[row, col])))

    return len(df), int(bad_rows.sum()), hits, examples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", *sorted(SPLIT_FILES)],
        help="which canonical split to scan (default: all 7 sources × 3 splits)",
    )
    parser.add_argument("--paths", nargs="*", help="scan these CSVs instead of the known split")
    args = parser.parse_args()

    if args.paths:
        work: list[tuple[str, str, str]] = [("", Path(p).name, p) for p in args.paths]
    elif args.split == "all":
        work = [
            (split, name, rel)
            for split, files in SPLIT_FILES.items()
            for name, rel in files.items()
        ]
    else:
        work = [(args.split, name, rel) for name, rel in SPLIT_FILES[args.split].items()]

    total_bad = 0
    missing: list[str] = []

    for split, name, rel in work:
        path = Path(rel)
        if not path.is_absolute():
            path = ROOT / rel
        label = f"{name}/{split}" if split else name
        if not path.exists():
            missing.append(f"{label} -> {rel}")
            print(f"  {label}: MISSING {rel}")
            continue

        n_rows, n_bad, hits, examples = scan_file(path)
        total_bad += n_bad
        verdict = "OK" if n_bad == 0 else f"** {n_bad} BAD ROWS **"
        print(f"  {label}: rows={n_rows} -> {verdict}")
        for col, chars in hits.items():
            top = sorted(chars.items(), key=lambda kv: -kv[1])[:6]
            print("      " + col + ": " + ", ".join(f"{ch!r}x{n}" for ch, n in top))
        for row, col, value in examples:
            shown = value if len(value) <= 80 else value[:80] + "..."
            print(f"      row {row} {col} = {shown!r}")

    if missing:
        print(f"\n{len(missing)} file(s) missing; scan incomplete")
    if total_bad:
        print(
            f"\nFAIL: {total_bad} row(s) carry characters outside RESIDUES.\n"
            "These will crash training with 'unmapped decoder grammar ids' the moment\n"
            "they are sampled, and the crash is reproducible across resumes."
        )
        return 1

    scope = args.split if not args.paths else "paths"
    print(f"\nresidue alphabet OK ({scope}): every sequence is within {RESIDUES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
