#!/usr/bin/env python
"""Report one BioSeq checkpoint's 1,000-candidate RVR Track-A trial.

Run::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/report_bioseq_rvr_checkpoint.py \
      --trial-dir /absolute/path/to/trial \
      --checkpoint /absolute/path/to/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench.track_a import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    RVR_TARGET_ID,
    is_valid_cdr3b,
    load_public_targets,
    read_generations,
    sequence_recovery,
    validity_reason,
)


DEFAULT_TRIAL_DIR = (
    BENCH
    / "outputs"
    / "tcr_beta_public_benchmark"
    / "checkpoint_comparison"
    / "bioseq_step189000_rvr_epitope_only"
)
DEFAULT_CHECKPOINT = (
    BENCH.parent.parent
    / "output"
    / "grammar_v2_esmc300m_integrated_llada_7l_step189000"
    / "best.pt"
)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _nearest_reference(sequence: str, references: tuple[str, ...]) -> tuple[str, float]:
    reference = max(references, key=lambda value: sequence_recovery(sequence, value))
    return reference, sequence_recovery(sequence, reference)


def _score_candidates(frame: pd.DataFrame, references: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for row in frame.sort_values("rank").itertuples(index=False):
        if not is_valid_cdr3b(row.cdr3b):
            continue
        reference, recovery = _nearest_reference(row.cdr3b, references)
        rows.append({
            "rank": int(row.rank),
            "generated_cdr3b": row.cdr3b,
            "nearest_reference_cdr3b": reference,
            "recovery": recovery,
        })
    return pd.DataFrame(rows).sort_values(
        ["recovery", "rank"], ascending=[False, True], kind="stable"
    )


def build_report(trial_dir: Path, canonical_results: Path, checkpoint: Path) -> None:
    generations = read_generations(trial_dir / "generations.csv")
    if len(generations) != 1_000:
        raise RuntimeError(f"expected 1,000 trial rows, found {len(generations)}")
    model_labels = generations["model"].unique().tolist()
    target_ids = generations["target_id"].unique().tolist()
    if len(model_labels) != 1 or target_ids != [RVR_TARGET_ID]:
        raise RuntimeError(f"unexpected trial identity: models={model_labels}, targets={target_ids}")
    model = str(model_labels[0])

    trial_metrics = pd.read_csv(trial_dir / "per_target_metrics.csv")
    if len(trial_metrics) != 1 or str(trial_metrics.iloc[0]["Model"]) != model:
        raise RuntimeError("trial metric row does not match generation label")
    metric = trial_metrics.iloc[0]
    if str(metric["GIANAStatus"]) != "ok":
        raise RuntimeError(f"GIANA did not complete successfully: {metric['GIANAStatus']}")

    targets = load_public_targets()
    references = targets[RVR_TARGET_ID].references
    scored = _score_candidates(generations, references)
    scored.head(20).to_csv(trial_dir / "top_recoveries.csv", index=False)

    canonical_metrics = pd.read_csv(canonical_results / "per_target_metrics.csv")
    comparison = canonical_metrics[
        (canonical_metrics["target_id"] == RVR_TARGET_ID)
        & canonical_metrics["Model"].isin([
            "BioSeq-7L-step117000", "GRATCR", "TCRT5", "TcrDesign",
        ])
    ].copy()
    comparison = pd.concat([comparison, trial_metrics], ignore_index=True)
    order = {
        "TCRT5": 0,
        "TcrDesign": 1,
        "GRATCR": 2,
        "BioSeq-7L-step117000": 3,
        model: 4,
    }
    comparison["_order"] = comparison["Model"].map(order)
    comparison = comparison.sort_values("_order").drop(columns="_order")
    comparison.to_csv(trial_dir / "rvr_comparison.csv", index=False)

    canonical_generations = read_generations(canonical_results / "generations.csv")
    step117 = canonical_generations[
        (canonical_generations["model"] == "BioSeq-7L-step117000")
        & (canonical_generations["target_id"] == RVR_TARGET_ID)
        & canonical_generations["cdr3b"].map(is_valid_cdr3b)
    ]
    set117 = set(step117["cdr3b"])
    set_trial = set(generations.loc[generations["cdr3b"].map(is_valid_cdr3b), "cdr3b"])
    intersection = len(set117 & set_trial)
    union = len(set117 | set_trial)

    invalid = generations[~generations["cdr3b"].map(is_valid_cdr3b)].copy()
    invalid["reason"] = invalid["cdr3b"].map(validity_reason)
    invalid[["rank", "cdr3b", "reason"]].to_csv(trial_dir / "invalid_generations.csv", index=False)

    manifest_path = checkpoint.with_name("checkpoint_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    recovery = scored["recovery"]
    quantiles = recovery.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    lengths = generations["cdr3b"].str.len()

    comparison_rows = []
    for _, row in comparison.iterrows():
        comparison_rows.append([
            row["Model"],
            f"{row['ValidRate']:.3f}",
            f"{row['UniqueRate']:.3f}",
            int(row["ExactHit@100"]),
            int(row["ExactHit@1000"]),
            int(row["Recovery90Hit@1000"]),
            f"{row['MedianBestRecovery']:.6f}",
            int(row["GIANAHit"]),
            int(row["GIANAReferenceClusterCoverage"]),
            int(row["GIANAReferenceSequenceCoverage"]),
        ])

    top_rows = [[
        int(row.rank),
        row.generated_cdr3b,
        row.nearest_reference_cdr3b,
        f"{row.recovery:.6f}",
    ] for row in scored.head(10).itertuples(index=False)]
    first_rows = [[int(row.rank), row.cdr3b] for row in generations.sort_values("rank").head(10).itertuples()]

    lines = [
        f"# {model} RVR epitope-only Track-A trial",
        "",
        f"Checkpoint: `{checkpoint.resolve()}`  ",
        f"Step: `{manifest['step']}`  ",
        f"Validation loss: `{manifest['val_loss']}`  ",
        f"SHA256: `{manifest['sha256']}`",
        "",
        "Protocol: `RVRAYTYSK` epitope-only conditioning; the HLA-A*03:01 allele and pseudo-sequence are retained as output metadata but are not consumed. Seed 42, batch size 100, 32 iterations, temperature 1.0, `gumbel_argmax`, and 1,000 raw candidates. Rank is seeded emission order and no C/F/W anchor is fixed or repaired.",
        "",
        "## Result",
        "",
        f"Valid={int(metric['n_valid'])}/1000 ({metric['ValidRate']:.3f}); unique valid={int(metric['n_unique_valid'])}/{int(metric['n_valid'])} ({metric['UniqueRate']:.3f}); ExactHit@100/1000={int(metric['ExactHit@100'])}/{int(metric['ExactHit@1000'])}; ReferenceRecall={metric['ReferenceRecall@1000']:.6f}; Recovery90Hit={int(metric['Recovery90Hit@1000'])}; median best recovery={metric['MedianBestRecovery']:.6f}; GIANA generated/reference-cluster/reference-sequence={int(metric['GIANAHit'])}/{int(metric['GIANAReferenceClusterCoverage'])}/{int(metric['GIANAReferenceSequenceCoverage'])} (`{metric['GIANAStatus']}`).",
        "",
        f"Length min/mean/median/max={int(lengths.min())}/{lengths.mean():.3f}/{lengths.median():.0f}/{int(lengths.max())}. Invalid={len(invalid)} with reasons `{metric['invalid_reasons']}`.",
        "",
        f"Recovery quartiles: p25={quantiles[0.25]:.6f}, median={quantiles[0.5]:.6f}, p75={quantiles[0.75]:.6f}, p90={quantiles[0.9]:.6f}, p95={quantiles[0.95]:.6f}, p99={quantiles[0.99]:.6f}, max={recovery.max():.6f}.",
        "",
        "## RVR comparison",
        "",
        *_markdown_table(
            ["Model", "Valid", "Unique", "Exact@100", "Exact@1000", "Recovery90", "MedianRecovery", "GIANA hit", "Ref clusters", "Ref sequences"],
            comparison_rows,
        ),
        "",
        f"The valid candidate sets of step117000 and {model} overlap by {intersection} sequences out of a union of {union} (Jaccard={intersection / union if union else 0.0:.6f}).",
        "",
        "## First 10 raw generations",
        "",
        *_markdown_table(["rank", "CDR3β"], first_rows),
        "",
        "## Top 10 nearest-reference generations",
        "",
        *_markdown_table(["rank", "generated", "nearest reference", "recovery"], top_rows),
        "",
        "## Interpretation",
        "",
        "This checkpoint retains near-perfect syntax and complete within-target uniqueness. Relative to step117000, median and maximum nearest-reference recovery improve slightly, but exact recovery, Recovery>=90%, and GIANA reference-cluster hits remain zero. Under this epitope-only prompt, the later checkpoint therefore does not close the target-specificity gap to TCRT5 or TcrDesign.",
        "",
        "This is an epitope-only diagnostic, not the training-consistent HLA-pseudosequence + epitope prompt. RVRAYTYSK is absent from the relevant conditional TCR training splits.",
        "",
        "## Files",
        "",
        f"- `{(trial_dir / 'generations.csv').resolve()}`",
        f"- `{(trial_dir / 'per_target_metrics.csv').resolve()}`",
        f"- `{(trial_dir / 'top_recoveries.csv').resolve()}`",
        f"- `{(trial_dir / 'rvr_comparison.csv').resolve()}`",
        f"- `{(trial_dir / 'invalid_generations.csv').resolve()}`",
    ]
    (trial_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL_DIR)
    parser.add_argument("--canonical-results", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    build_report(args.trial_dir, args.canonical_results, args.checkpoint)
    print(f"report -> {args.trial_dir / 'report.md'}")


if __name__ == "__main__":
    main()
