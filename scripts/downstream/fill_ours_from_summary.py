#!/usr/bin/env python
"""Backfill Ours numbers from run_all_downstream summary into docs / paper.

Reads (or builds) ``output/downstream_generation/summary_<run>.json`` via the
same collectors as ``run_all_downstream.py collect``, then emits a human-readable
report and optional targeted LaTeX replacements for ``\\OursCkpt{}`` rows.

Usage::

    python scripts/downstream/fill_ours_from_summary.py \\
        --checkpoint output/grammar_v2_esmc300m_integrated_llada/best.pt --report

    python scripts/downstream/fill_ours_from_summary.py \\
        --summary output/downstream_generation/summary_grammar_v2_esmc300m_integrated_llada.json \\
        --apply-paper --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
PAPER = ROOT / "paper" / "sections"
REPORT = ROOT / "output" / "downstream_generation" / "C1_FILL_REPORT.md"


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _collect(checkpoint: Path, out: Path) -> dict[str, Any]:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "downstream" / "run_all_downstream.py"),
            "collect",
            "--checkpoint",
            str(checkpoint),
            "--out",
            str(out),
        ],
        check=True,
        cwd=str(ROOT),
    )
    return _load_summary(out)


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _task_metrics(summary: dict[str, Any], key: str) -> dict[str, Any]:
    block = summary.get("tasks", {}).get(key, {})
    return block.get("metrics") or {}


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# C1 fill report",
        "",
        f"- checkpoint: `{summary.get('checkpoint', '?')}`",
        f"- run: `{summary.get('run', '?')}`",
        f"- tcr_tag: `{summary.get('tcr_tag', '?')}`",
        "",
    ]

    t1 = _task_metrics(summary, "t1")
    if t1:
        lines += [
            "## T1 (NM2025 cdr3b · AS · globalfeat)",
            "",
            "| Seen AUPRC | Unseen AUPRC | Seen AUROC | Unseen AUROC |",
            "|:----------:|:------------:|:----------:|:------------:|",
            f"| {_fmt(t1.get('seen_AUPRC'))} | {_fmt(t1.get('unseen_AUPRC'))} "
            f"| {_fmt(t1.get('seen_AUROC'))} | {_fmt(t1.get('unseen_AUROC'))} |",
            "",
        ]

    t2 = _task_metrics(summary, "t2")
    if t2:
        lines += [
            "## T2 Basis B (TCREmbedding k-means mean)",
            "",
            f"ARI={_fmt(t2.get('ARI'))}, NMI={_fmt(t2.get('NMI'))}, Purity={_fmt(t2.get('Purity'))}",
            "",
        ]

    t3 = _task_metrics(summary, "t3")
    if t3:
        lines += [
            "## T3 (SCEPTR deep 6-pMHC few-shot AUROC)",
            "",
            " | ".join(f"k{k}={_fmt(t3.get(f'deep_k{k}'))}" for k in ("5", "20", "100", "200")),
            "",
        ]

    t4 = _task_metrics(summary, "t4")
    if t4:
        lines += ["## T4 Setting A", "", f"JSD={_fmt(t4.get('settingA_jsd'), 4)}", ""]

    mint = _task_metrics(summary, "mint")
    if mint:
        lines += ["## MINT", ""]
        for task in ("HumanPPI", "Bernett", "YeastPPI", "MutationalPPI", "SKEMPI"):
            values = []
            for metric in ("AUROC", "AUPRC", "Accuracy", "F1", "Pearson", "Spearman", "RMSE"):
                value = mint.get(f"{task}:{metric}")
                if value is not None:
                    values.append(f"{metric}={_fmt(value)}")
            if values:
                lines.append(f"- **{task}**: {', '.join(values)}")
        lines.append("")

    flab = _task_metrics(summary, "flab")
    if flab:
        flab_heading = (
            "## FLAb (nested 10×5-fold outer mean R²)"
            if any(str(k).endswith(":R2") for k in flab)
            else "## FLAb (legacy aggregate; pooled Spearman ρ)"
        )
        lines += [flab_heading, ""]
        for k, v in flab.items():
            lines.append(f"- {k}: {_fmt(v)}")
        lines.append("")

    ab = _task_metrics(summary, "ab")
    if ab:
        lines += ["## Antibody", ""]
        for k, v in ab.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    nb = _task_metrics(summary, "nbbench")
    if nb:
        lines += ["## NbBench", "", str(nb), ""]

    missing = [k for k in ("t1", "t2", "t3", "t4", "mint", "ab", "flab", "nbbench")
               if not _task_metrics(summary, k)]
    if missing:
        lines += ["## Missing / incomplete families", "", ", ".join(missing), ""]

    lines += [
        "## Next steps",
        "",
        "1. Update `downstream/benchmark/RESULTS.md` Ours-integrated row from this report.",
        "2. Run `--apply-paper` (after reviewing `--dry-run` diff).",
        "3. `tectonic -X compile paper/main.tex` in env `tex`.",
        "",
    ]
    return "\n".join(lines)


def _replace_ours_row(text: str, label: str, values: list[str]) -> tuple[str, bool]:
    """Replace ``\\OursCkpt{} & \\tbd & ...`` row nearest after ``\\label{label}``."""
    pat = re.compile(
        rf"(\\label{{{re.escape(label)}}}[\s\S]*?\\midrule\s*\n)"
        rf"(\\OursCkpt\{{\}}\s*&\s*)"
        rf"(\\tbd(?:\s*&\s*\\tbd)*)"
        rf"(\s*\\\\)",
        re.MULTILINE,
    )
    repl_vals = " & ".join(values)
    new_text, n = pat.subn(rf"\1\2{repl_vals}\4", text, count=1)
    return new_text, n > 0


def apply_paper(summary: dict[str, Any], dry_run: bool) -> list[str]:
    exp = PAPER / "4_experiments.tex"
    text = exp.read_text()
    changes: list[str] = []

    t1 = _task_metrics(summary, "t1")
    if t1:
        vals = [_fmt(t1.get(k)) for k in ("seen_AUPRC", "unseen_AUPRC", "seen_AUROC", "unseen_AUROC")]
        if all(v != "—" for v in vals):
            text, ok = _replace_ours_row(text, "tab:t1", vals)
            changes.append(f"tab:t1 -> {vals}" if ok else "tab:t1 SKIPPED (pattern not found)")

    t2 = _task_metrics(summary, "t2")
    if t2:
        vals = [_fmt(t2.get(k)) for k in ("ARI", "NMI", "Purity")]
        if all(v != "—" for v in vals):
            text, ok = _replace_ours_row(text, "tab:t2", vals)
            changes.append(f"tab:t2 -> {vals}" if ok else "tab:t2 SKIPPED")

    t3 = _task_metrics(summary, "t3")
    if t3:
        vals = [_fmt(t3.get(f"deep_k{k}")) for k in ("5", "20", "100", "200")]
        if all(v != "—" for v in vals):
            text, ok = _replace_ours_row(text, "tab:t3", vals)
            changes.append(f"tab:t3 -> {vals}" if ok else "tab:t3 SKIPPED")

    if changes and not dry_run:
        exp.write_text(text)
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="re-collect from outputs if --summary omitted")
    ap.add_argument("--summary", default=None, help="existing summary JSON")
    ap.add_argument("--report", action="store_true", help="write C1_FILL_REPORT.md")
    ap.add_argument("--apply-paper", action="store_true", help="replace T1/T2/T3 Ours rows in 4_experiments.tex")
    ap.add_argument("--dry-run", action="store_true", help="with --apply-paper, print changes only")
    args = ap.parse_args()

    if args.summary:
        summary_path = Path(args.summary)
        summary = _load_summary(summary_path)
    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        run = ckpt.parent.name
        summary_path = ROOT / "output" / "downstream_generation" / f"summary_{run}.json"
        summary = _collect(ckpt.resolve(), summary_path)
    else:
        ap.error("provide --summary or --checkpoint")

    report = build_report(summary)
    print(report)

    if args.report:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(report)
        print(f"wrote {REPORT}")

    if args.apply_paper:
        changes = apply_paper(summary, dry_run=args.dry_run)
        for c in changes:
            print(c)
        if args.dry_run:
            print("(dry-run: paper not modified)")


if __name__ == "__main__":
    main()
