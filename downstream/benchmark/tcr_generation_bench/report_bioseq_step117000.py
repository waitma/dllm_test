#!/usr/bin/env python
"""Materialise a focused report for the BioSeq 7L step117000 Track-A run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench.bioseq_generator import BIOSEQ_7L_STEP117000_LABEL  # noqa: E402
from tcr_generation_bench.track_a import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    RVR_TARGET_ID,
    best_sequence_recovery,
    is_valid_cdr3b,
    load_public_targets,
    read_generations,
    sequence_recovery,
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _giana_hits(results_dir: Path, target_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target_id in target_ids:
        path = (
            results_dir
            / "giana"
            / BIOSEQ_7L_STEP117000_LABEL
            / _safe_name(target_id)
            / "giana_clusters.tsv"
        )
        if not path.is_file():
            continue
        records = []
        for line in path.read_text().splitlines():
            if not line or line.startswith("##"):
                continue
            fields = line.split("\t")
            if len(fields) < 4:
                continue
            records.append({
                "cdr3b": fields[0],
                "cluster": fields[1],
                "source": fields[2],
                "sequence_id": fields[3],
            })
        clustered = pd.DataFrame(records)
        if clustered.empty:
            continue
        reference_clusters = set(
            clustered.loc[clustered["source"] == "reference", "cluster"]
        )
        for generated in clustered[
            (clustered["source"] == "generated")
            & (clustered["cluster"].isin(reference_clusters))
        ].itertuples(index=False):
            references = clustered[
                (clustered["source"] == "reference")
                & (clustered["cluster"] == generated.cluster)
            ]["cdr3b"].tolist()
            rows.append({
                "target_id": target_id,
                "cluster": generated.cluster,
                "generated_cdr3b": generated.cdr3b,
                "reference_cdr3b": ";".join(references),
            })
    return pd.DataFrame(
        rows,
        columns=["target_id", "cluster", "generated_cdr3b", "reference_cdr3b"],
    )


def build_report(results_dir: Path) -> None:
    targets = load_public_targets()
    generations = read_generations(results_dir / "generations.csv")
    bioseq = generations[generations["model"] == BIOSEQ_7L_STEP117000_LABEL].copy()
    if len(bioseq) != 14_000:
        raise RuntimeError(f"expected 14,000 BioSeq rows, found {len(bioseq)}")
    per_target = pd.read_csv(results_dir / "per_target_metrics.csv")
    per_target = per_target[per_target["Model"] == BIOSEQ_7L_STEP117000_LABEL].copy()
    summary = pd.read_csv(results_dir / "summary_metrics.csv")

    top_rows: list[dict[str, object]] = []
    for target_id, block in bioseq.groupby("target_id", sort=True):
        references = targets[target_id].references
        scored = []
        for row in block.itertuples(index=False):
            if is_valid_cdr3b(row.cdr3b):
                scored.append((
                    best_sequence_recovery(row.cdr3b, references),
                    int(row.rank),
                    row.cdr3b,
                ))
        scored.sort(reverse=True)
        for recovery, rank, sequence in scored[:5]:
            reference = max(
                references,
                key=lambda candidate: sequence_recovery(sequence, candidate),
            )
            top_rows.append({
                "target_id": target_id,
                "rank": rank,
                "generated_cdr3b": sequence,
                "nearest_reference_cdr3b": reference,
                "recovery": recovery,
            })
    top = pd.DataFrame(top_rows)
    giana_hits = _giana_hits(results_dir, sorted(targets))

    bioseq.to_csv(results_dir / "bioseq_step117000_generations.csv", index=False)
    per_target.to_csv(results_dir / "bioseq_step117000_per_target_metrics.csv", index=False)
    top.to_csv(results_dir / "bioseq_step117000_top_recoveries.csv", index=False)
    giana_hits.to_csv(results_dir / "bioseq_step117000_giana_hits.csv", index=False)

    invalid_counts: dict[str, int] = {}
    for value in per_target["invalid_reasons"]:
        for reason, count in json.loads(value).items():
            invalid_counts[reason] = invalid_counts.get(reason, 0) + int(count)
    rvr = per_target[per_target["target_id"] == RVR_TARGET_ID].iloc[0]
    rvr_top = top[top["target_id"] == RVR_TARGET_ID]

    comparison_rows = []
    for _, row in summary.iterrows():
        comparison_rows.append([
            row["Model"],
            f"{row['ValidRate']:.6f}",
            f"{row['UniqueRate']:.6f}",
            int(row["ExactHit@1000"]),
            int(row["Recovery90Hit"]),
            int(row["GIANAHit"]),
            f"{row['CrossEpitopeJaccardMean']:.6f}",
        ])
    target_rows = []
    for _, row in per_target.iterrows():
        target_rows.append([
            row["target_id"],
            f"{row['ValidRate']:.3f}",
            f"{row['UniqueRate']:.3f}",
            int(row["ExactHit@1000"]),
            int(row["Recovery90Hit@1000"]),
            f"{row['MedianBestRecovery']:.3f}",
            int(row["GIANAHit"]),
        ])

    lines = [
        "# BioSeq 7L step117000 public CDR3β Track-A result",
        "",
        "Checkpoint: `/project_c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt`  ",
        "SHA256: `e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`",
        "",
        "Protocol: epitope-only conditioning, seed 42, batch size 100, 32 masked-diffusion iterations, temperature 1.0, `gumbel_argmax`, 1,000 raw candidates per target. MHC fields are retained in the interchange table but are not consumed by this sampler. No candidate is repaired or given a synthetic C/F/W anchor. Rank is seeded emission order and `raw_score` is empty.",
        "",
        "## Main comparison",
        "",
        *_markdown_table(
            ["Model", "ValidRate", "UniqueRate", "Exact@1000", "Recovery90", "GIANAHit", "Cross-target Jaccard"],
            comparison_rows,
        ),
        "",
        "BioSeq generates highly valid and nearly unique sequences, with almost no exact cross-target reuse. However, it recovers no exact reference, has no candidate at ≥90% recovery, and has only one GIANA reference-cluster hit. This checkpoint therefore trails TCRT5 and TcrDesign on target-specific recovery despite strong raw diversity.",
        "",
        f"Invalid raw candidates: {sum(invalid_counts.values())}/14,000; reasons: `{json.dumps(invalid_counts, sort_keys=True)}`.",
        "",
        "## BioSeq per target",
        "",
        *_markdown_table(
            ["target_id", "Valid", "Unique", "Exact", "Recovery90", "MedianRecovery", "GIANA"],
            target_rows,
        ),
        "",
        "## RVR diagnostic",
        "",
        f"RVR ValidRate={rvr['ValidRate']:.3f}, UniqueRate={rvr['UniqueRate']:.3f}, ExactHit@1000={int(rvr['ExactHit@1000'])}, Recovery90Hit={int(rvr['Recovery90Hit@1000'])}, GIANAHit={int(rvr['GIANAHit'])}, median recovery={rvr['MedianBestRecovery']:.6f}.",
        "",
        *_markdown_table(
            ["rank", "generated", "nearest reference", "recovery"],
            [[
                int(row.rank),
                row.generated_cdr3b,
                row.nearest_reference_cdr3b,
                f"{row.recovery:.6f}",
            ] for row in rvr_top.itertuples(index=False)],
        ),
        "",
        "## GIANA hit",
        "",
    ]
    if giana_hits.empty:
        lines.append("No BioSeq generated sequence entered a reference-containing cluster.")
    else:
        lines.extend(_markdown_table(
            ["target_id", "cluster", "generated", "reference"],
            [[row.target_id, row.cluster, row.generated_cdr3b, row.reference_cdr3b]
             for row in giana_hits.itertuples(index=False)],
        ))
    lines.extend([
        "",
        "## Files",
        "",
        f"- `{(results_dir / 'bioseq_step117000_generations.csv').resolve()}`",
        f"- `{(results_dir / 'bioseq_step117000_per_target_metrics.csv').resolve()}`",
        f"- `{(results_dir / 'bioseq_step117000_top_recoveries.csv').resolve()}`",
        f"- `{(results_dir / 'bioseq_step117000_giana_hits.csv').resolve()}`",
    ])
    (results_dir / "bioseq_step117000_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()
    build_report(args.results_dir)
    print(f"BioSeq report -> {args.results_dir / 'bioseq_step117000_report.md'}")


if __name__ == "__main__":
    main()
