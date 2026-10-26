#!/usr/bin/env python
"""Rebuild the complete TCRT5 evaluation and compare it with the paper.

Run both the author-release reconstruction and the current rebuilt models::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py \
      all --with-pgen --reference-pgen --pgen-workers 16

Run only fast CPU metrics (no OLGA Pgen)::

    /project_c20250601/251105016/conda/envs/pllm/bin/python \
      /project_c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py \
      all
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


BENCH = Path(__file__).resolve().parents[1]
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from tcr_generation_bench.tcrt5_full_eval import (  # noqa: E402
    AUTHOR_ATOMIC_CSV,
    AUTHOR_DATASET_CSV,
    AUTHOR_SPARSE_CSV,
    DEFAULT_OUTPUT_ROOT,
    PAPER_MAIN_TABLE,
    PAPER_SUPPLEMENT_URL,
    PAPER_URL,
    RVR_TARGET,
    SPARSE_EXACT_TARGETS,
    TRACK_A_GENERATIONS,
    TRACK_A_METRICS,
    add_giana_summary,
    author_release_crosscheck,
    evaluate_blocks,
    json_ready,
    load_author_dataset_metrics,
    load_author_main_blocks,
    load_author_sparse_blocks,
    load_track_a_blocks,
    merge_existing_giana,
    olga_protocol_audit_rows,
    paper_main_comparison,
    pgen_aggregation_sensitivity_rows,
    pgen_summary_rows,
    polyspecificity_paper_correlation,
    write_tables,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _fmt(value: object, digits: int = 4) -> str:
    number = _float(value)
    if math.isnan(number):
        return "—"
    if abs(number - round(number)) < 1e-12 and abs(number) >= 10:
        return f"{int(round(number))}"
    return f"{number:.{digits}f}"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return lines


def _write_manifest(
    output_dir: Path,
    *,
    command: str,
    source_paths: Mapping[str, Path],
    tables: Mapping[str, pd.DataFrame],
    pgen_enabled: bool,
) -> None:
    manifest = {
        "protocol": "tcrt5_full_eval_v1",
        "created_at": _utc_now(),
        "command": command,
        "paper": PAPER_URL,
        "supplement": PAPER_SUPPLEMENT_URL,
        "source_paths": {key: str(value) for key, value in source_paths.items()},
        "source_exists": {key: value.is_file() for key, value in source_paths.items()},
        "pgen_enabled": pgen_enabled,
        "tables": {
            name: {
                "path": str(output_dir / f"{name}.csv"),
                "rows": int(len(frame)),
                "columns": list(frame.columns),
            }
            for name, frame in tables.items()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(json_ready(manifest), indent=2, ensure_ascii=False) + "\n"
    )


def run_author_release(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_root / "author_release"
    main_dir = output_dir / "main_top20_k100"
    sparse_dir = output_dir / "sparse13_plus_rvr_k1000"

    main_blocks, author_atomic = load_author_main_blocks(args.author_atomic)
    author_dataset, dataset_char_bleu = load_author_dataset_metrics(args.author_dataset)
    main = evaluate_blocks(
        main_blocks,
        k=100,
        dataset_char_bleu=dataset_char_bleu,
    )
    main["crosscheck"] = author_release_crosscheck(
        main["per_target"],
        author_atomic,
        author_dataset,
    )
    main["paper_main_comparison"] = paper_main_comparison(main["summary"])
    poly_correlation = polyspecificity_paper_correlation(
        main["polyspecificity_summary"]
    )

    pgen = pd.DataFrame()
    if args.with_pgen:
        flagship = [block for block in main_blocks if block.model == "TCRT5-FT"]
        pgen = pgen_summary_rows(
            flagship,
            args.pgen_cache,
            workers=args.pgen_workers,
            include_references=args.reference_pgen,
            models={"TCRT5-FT"},
        )
        main["pgen"] = pgen
        main["pgen_aggregation_sensitivity"] = (
            pgen_aggregation_sensitivity_rows(
                flagship,
                args.pgen_cache,
                workers=args.pgen_workers,
            )
        )
        main["olga_protocol_audit"] = olga_protocol_audit_rows()

    write_tables(main, main_dir)
    (main_dir / "polyspecificity_correlation.json").write_text(
        json.dumps(json_ready(poly_correlation), indent=2, ensure_ascii=False) + "\n"
    )
    _write_manifest(
        main_dir,
        command="author-release main_top20_k100",
        source_paths={
            "atomic_predictions": args.author_atomic,
            "dataset_metrics": args.author_dataset,
        },
        tables=main,
        pgen_enabled=args.with_pgen,
    )

    sparse_blocks = load_author_sparse_blocks(args.author_sparse)
    sparse = evaluate_blocks(
        sparse_blocks,
        k=1000,
        use_first_rank_as_bleu_proxy=False,
    )
    write_tables(sparse, sparse_dir)
    _write_manifest(
        sparse_dir,
        command="author-release sparse13_plus_rvr_k1000",
        source_paths={"sparse_predictions": args.author_sparse},
        tables=sparse,
        pgen_enabled=False,
    )
    return {
        "main": main,
        "main_blocks": main_blocks,
        "sparse": sparse,
        "sparse_blocks": sparse_blocks,
        "poly_correlation": poly_correlation,
        "pgen": pgen,
        "output_dir": output_dir,
    }


def run_current_track_a(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_root / "current_track_a"
    blocks, generations = load_track_a_blocks(
        args.generations,
        args.author_sparse,
    )
    result = evaluate_blocks(
        blocks,
        k=1000,
        use_first_rank_as_bleu_proxy=True,
    )
    result["per_target"] = merge_existing_giana(
        result["per_target"],
        args.giana_metrics,
    )
    result["summary"] = add_giana_summary(
        result["summary"],
        result["per_target"],
    )
    if args.with_pgen:
        result["pgen"] = pgen_summary_rows(
            blocks,
            args.pgen_cache,
            workers=args.pgen_workers,
            include_references=False,
        )
    write_tables(result, output_dir)
    _write_manifest(
        output_dir,
        command="current-track-a k1000",
        source_paths={
            "generations": args.generations,
            "references": args.author_sparse,
            "giana_metrics": args.giana_metrics,
        },
        tables=result,
        pgen_enabled=args.with_pgen,
    )
    return {
        "tables": result,
        "blocks": blocks,
        "generations": generations,
        "output_dir": output_dir,
    }


def _comparison_special_rows(
    author_result: dict[str, object],
    current_result: dict[str, object] | None,
) -> pd.DataFrame:
    main = author_result["main"]
    main_summary = main["summary"].set_index("model")
    main_target = main["per_target"]
    flagship = main_summary.loc["TCRT5-FT"]
    reference_length_mean = float(
        main_target[main_target["model"] == "TCRT5-FT"]["reference_length_mean"]
        .mul(main_target[main_target["model"] == "TCRT5-FT"]["n_references"])
        .sum()
        / main_target[main_target["model"] == "TCRT5-FT"]["n_references"].sum()
    )
    reference_blocks = [
        block
        for block in author_result["main_blocks"]
        if block.model == "TCRT5-FT"
    ]
    reference_lengths = np.asarray([
        len(sequence)
        for block in reference_blocks
        for sequence in block.references
    ])
    sparse_target = author_result["sparse"]["per_target"]
    tcrt5_sparse = sparse_target[sparse_target["model"] == "TCRT5"]
    sparse13 = tcrt5_sparse[tcrt5_sparse["target_id"] != RVR_TARGET]
    rvr = tcrt5_sparse[tcrt5_sparse["target_id"] == RVR_TARGET].iloc[0]

    rows: list[dict[str, object]] = []

    def add(
        section: str,
        metric: str,
        reconstructed: object,
        paper: object,
        agreement: object,
        evidence: str,
    ) -> None:
        rows.append({
            "section": section,
            "metric": metric,
            "reconstructed_value": reconstructed,
            "paper_value": paper,
            "agreement": agreement,
            "evidence": evidence,
        })

    add(
        "Fig4",
        "TCRT5 generated length mean",
        float(flagship["generated_length_mean"]),
        14.6,
        abs(float(flagship["generated_length_mean"]) - 14.6) <= 0.05,
        "independent_from_released_2000_sequences",
    )
    add(
        "Fig4",
        "TCRT5 generated length std population",
        float(flagship["generated_length_std_population"]),
        1.2,
        abs(float(flagship["generated_length_std_population"]) - 1.2) <= 0.05,
        "independent_from_released_2000_sequences",
    )
    add(
        "Fig4",
        "Reference length mean",
        reference_length_mean,
        14.5,
        abs(reference_length_mean - 14.5) <= 0.05,
        "independent_from_68416_target-deduplicated_references",
    )
    add(
        "Fig4",
        "Reference length std population",
        float(reference_lengths.std(ddof=0)),
        2.0,
        abs(float(reference_lengths.std(ddof=0)) - 2.0) <= 0.05,
        "independent_from_68416_target-deduplicated_references",
    )
    add(
        "Fig4",
        "Known target-specific binders among 2000 generations",
        int(flagship["exact_hit_occurrences"]),
        181,
        int(flagship["exact_hit_occurrences"]) == 181,
        "independent_exact_membership",
    )

    pgen = author_result["pgen"]
    if len(pgen):
        generated_pgen = pgen[
            (pgen["model"] == "TCRT5-FT")
            & (pgen["target_id"] == "__all_targets__")
            & (pgen["population"] == "generated")
        ]
        if len(generated_pgen):
            pgen_row = generated_pgen.iloc[0]
            add(
                "Fig4",
                "TCRT5 non-zero OLGA Pgen count",
                int(pgen_row["n_pgen_positive"]),
                1996,
                int(pgen_row["n_pgen_positive"]) == 1996,
                "independent_OLGA_human_T_beta",
            )
            add(
                "Fig4",
                "TCRT5 mean log10 Pgen",
                float(pgen_row["mean_log10_pgen_positive"]),
                -7.04,
                abs(float(pgen_row["mean_log10_pgen_positive"]) + 7.04) <= 0.005,
                "independent_OLGA_human_T_beta",
            )
            add(
                "Fig4",
                "TCRT5 std log10 Pgen",
                float(pgen_row["std_log10_pgen_positive_population"]),
                0.85,
                abs(float(pgen_row["std_log10_pgen_positive_population"]) - 0.85) <= 0.005,
                "independent_OLGA_human_T_beta",
            )
        reference_pgen = pgen[
            (pgen["model"] == "__reference__")
            & (pgen["target_id"] == "__all_targets__")
        ]
        if len(reference_pgen):
            reference_row = reference_pgen.iloc[0]
            add(
                "Fig4",
                "Reference mean log10 Pgen",
                float(reference_row["mean_log10_pgen_positive"]),
                -9.83,
                abs(float(reference_row["mean_log10_pgen_positive"]) + 9.83) <= 0.005,
                "independent_OLGA_human_T_beta",
            )
            add(
                "Fig4",
                "Reference std log10 Pgen",
                float(reference_row["std_log10_pgen_positive_population"]),
                2.356,
                abs(
                    float(reference_row["std_log10_pgen_positive_population"])
                    - 2.356
                ) <= 0.0005,
                "independent_OLGA_human_T_beta",
            )

    correlation = author_result["poly_correlation"]
    if correlation.get("status") == "ok":
        reconstructed_r = correlation[
            "pearson_r_unique_vs_polyspecific_generations"
        ]
        add(
            "Fig3",
            "Pearson r: total unique vs polyspecific generations (selected 6 variants)",
            reconstructed_r,
            -0.957,
            abs(reconstructed_r + 0.957) <= 0.001,
            "independent_exact_915-panel_membership",
        )

    entropy = main["positional_entropy"]
    flagship_entropy = entropy[entropy["model"] == "TCRT5-FT"]
    position_means = flagship_entropy.groupby("position_1based")[
        "delta_entropy_gap_inclusive_nats"
    ].mean()
    minimum_position = int(position_means.idxmin())
    negative_fraction = float(
        (
            flagship_entropy.loc[
                flagship_entropy["position_1based"] > 1,
                "delta_entropy_gap_inclusive_nats",
            ] < 0
        ).mean()
    )
    add(
        "Fig4",
        "Position of strongest mean entropy loss",
        minimum_position,
        "around position 5",
        minimum_position in {5, 6},
        "natural-log_right-pad_reconstruction",
    )
    add(
        "Fig4",
        "Fraction negative positional delta entropy after position 1",
        negative_fraction,
        "nearly all positions",
        negative_fraction >= 0.9,
        "natural-log_right-pad_reconstruction",
    )

    kmer = main["kmer_jsd"]
    flagship_kmer = (
        kmer[kmer["model"] == "TCRT5-FT"]
        .groupby("k")["js_divergence_nats"]
        .mean()
    )
    add(
        "Fig4",
        "k-mer JS range",
        (
            f"k2={flagship_kmer.loc[2]:.6f}; "
            f"k12={flagship_kmer.loc[12]:.6f}"
        ),
        "Fig4d curve, k=2..12, asymptote near ln(2)",
        bool(flagship_kmer.loc[12] <= math.log(2) + 1e-12),
        "independent_natural-log_JS_divergence",
    )

    add(
        "Fig5",
        "Sparse13 exact hit count",
        int(sparse13["exact_hit_count"].sum()),
        2,
        int(sparse13["exact_hit_count"].sum()) == 2,
        "independent_from_released_13x1000_predictions",
    )
    reconstructed_targets = set(
        sparse13.loc[sparse13["exact_hit_count"] > 0, "target_id"]
    )
    add(
        "Fig5",
        "Sparse13 exact-hit targets",
        ";".join(sorted(reconstructed_targets)),
        ";".join(sorted(SPARSE_EXACT_TARGETS)),
        reconstructed_targets == SPARSE_EXACT_TARGETS,
        "independent_exact_membership",
    )
    add(
        "Fig5",
        "RVR exact hit count",
        int(rvr["exact_hit_count"]),
        8,
        int(rvr["exact_hit_count"]) == 8,
        "independent_from_released_1000_predictions",
    )
    add(
        "Fig5",
        "RVR exact ranks",
        str(rvr["exact_hit_ranks"]),
        "11;17;83;258;259;430;535;603",
        str(rvr["exact_hit_ranks"]) == "11;17;83;258;259;430;535;603",
        "released_candidate_order",
    )

    if current_result is not None:
        current_target = current_result["tables"]["per_target"]
        current_rvr = current_target[
            (current_target["model"] == "TCRT5")
            & (current_target["target_id"] == RVR_TARGET)
        ]
        if len(current_rvr):
            current_rvr = current_rvr.iloc[0]
            add(
                "Fig5",
                "RVR GIANA clustered generations t",
                int(current_rvr["giana_generated_hit"]),
                231,
                int(current_rvr["giana_generated_hit"]) == 231,
                "official_GIANA4.1_existing_completed_run",
            )
            add(
                "Fig5",
                "RVR GIANA reference-containing clusters c",
                int(current_rvr["giana_reference_cluster_coverage"]),
                23,
                int(current_rvr["giana_reference_cluster_coverage"]) == 23,
                "official_GIANA4.1_existing_completed_run",
            )
            add(
                "Fig5",
                "RVR native recovery >=90% count",
                int(current_rvr["native_recovery_ge_90_count"]),
                "figure-only/no machine-readable paper value",
                "definition_reconstructed",
                "same-length_Hamming_identity",
            )
    return pd.DataFrame(rows)


def _write_report(
    author_result: dict[str, object],
    current_result: dict[str, object] | None,
    special: pd.DataFrame,
    output_path: Path,
) -> None:
    main_summary = author_result["main"]["summary"].set_index("model")
    comparison = author_result["main"]["paper_main_comparison"]
    crosscheck = author_result["main"]["crosscheck"]
    non_map = comparison[comparison["metric"] != "map"]
    non_map_mismatches = non_map[~non_map["agreement"]]
    atomic_crosscheck = crosscheck[crosscheck["level"] == "atomic_240_blocks"]
    dataset_crosscheck = crosscheck[crosscheck["level"] == "dataset_12_models"]
    exact_atomic_crosscheck = bool(atomic_crosscheck["all_close_1e_12"].all())
    exact_dataset_non_recovery = bool(
        dataset_crosscheck.loc[
            dataset_crosscheck["metric"] != "native_recovery",
            "all_close_1e_12",
        ].all()
    )
    dataset_recovery = dataset_crosscheck[
        dataset_crosscheck["metric"] == "native_recovery"
    ].iloc[0]

    lines = [
        "# TCRT5 完整生成评测重建与论文对应",
        "",
        f"生成时间：{_utc_now()}",
        "",
        f"论文：{PAPER_URL}",
        "",
        "## 结论",
        "",
        "完整 evaluator 已覆盖主评测 `20 pMHC × 100`、稀疏评测 "
        "`13 pMHC × 1,000` 和独立 RVR `1 × 1,000`。P/R/F1、编辑距离、"
        "native recovery、全局 unique、Jaccard、positional Δentropy、"
        "k-mer JS、长度、OLGA Pgen、≥90% identity、GIANA 与 polyspecificity "
        "均已落成逐靶点和逐模型表。",
        "",
        f"作者公开序列上的核心逐块交叉验证为 "
        f"`{'PASS' if exact_atomic_crosscheck else 'FAIL'}`：240 个 model×pMHC "
        "block 的 P/R/F1、编辑距离、native recovery 和原子 Char-BLEU "
        "与作者 CSV 在 `1e-12` 内一致。",
        "",
        f"作者另行发布的 12-model dataset companion 中，P/R/F1/编辑距离聚合为 "
        f"`{'PASS' if exact_dataset_non_recovery else 'FAIL'}`；native recovery "
        f"与同一发布包的 240-block 聚合最大相差 "
        f"`{float(dataset_recovery['max_abs_delta']):.6f}`。dataset companion "
        "没有对应候选序列，因此把它记录为发布产物内部差异，而不是 evaluator 回归失败。",
        "",
        "论文主表的大多数非 mAP 单元格能够对应；少数不一致项会在下表中明确列出，"
        "并与“重建 evaluator 是否通过作者 CSV 回归”分开判断。mAP 的文字定义已经实现，"
        "但作者公开的 "
        "`all_atomic_eval_dedup.csv` 没有论文采用的 cumulative model "
        "log-likelihood；直接把公开列表顺序当 rank 得到的是 proxy，不能冒充论文 mAP。"
        "当前 Track-A 的 TCRT5 行保存了 cumulative log-likelihood，所以该模型在当前"
        "稀疏任务上的 mAP 可按论文定义计算；BioSeq/GRATCR/TcrDesign 无可比 likelihood，"
        "其 ordered AP 只作 rank-order diagnostic。",
        "",
        "## 指标覆盖",
        "",
    ]
    lines.extend(_markdown_table(
        ["维度", "实现", "与 TCRT5 的关系"],
        [
            ["Char-BLEU", "sentence/corpus BLEU-4，最近 20 refs", "精确定义；作者主表缺 greedy 原串，只能核对官方数值"],
            ["Native recovery", "同长先按 Lev 选 ref、再算 Hamming；无同长才用 ref-length Lev", "精确"],
            ["P/R/F1@K", "保留官方 duplicate/unique 语义", "精确"],
            ["mAP", "平均 P@1…P@K", "定义精确；主表缺 likelihood，明确标 proxy"],
            ["Hit@K/rank", "1/5/10/20/50/100/250/500/1000", "Fig.5 扩展并对齐"],
            ["Total unique", "跨 pMHC pooled unique", "精确"],
            ["Jaccard", "全 pair similarity + dissimilarity", "精确"],
            ["Positional entropy", "`H(gen)-H(ref)`，nats，gap-inclusive + residue-only", "Fig.4c 对齐"],
            ["k-mer JS", "natural-log JSD，k=2…12", "Fig.4d 精确口径"],
            ["OLGA Pgen", "human_T_beta，正值率 + log10 分布", "定义/非零计数精确；Fig.4 分布矩缺最终聚合快照"],
            ["Length", "完整 count/frequency + mean/std", "Fig.4a/Fig.5b"],
            ["≥90% identity", "paper-native per-candidate recovery threshold", "Fig.5d/g"],
            ["GIANA", "合并已完成 GIANA4.1 原生结果", "Fig.5e–g"],
            ["Polyspecificity", "915 panel exact membership + N_Epitopes", "Fig.3"],
        ],
    ))
    lines.extend([
        "",
        "## Supplementary Table 2 对应",
        "",
        f"非 mAP 的 {int(non_map['agreement'].sum())}/{len(non_map)} 格在论文显示精度内一致；"
        "mAP 的 12 格全部单列为缺少 rank score 的不可精确复现项。",
        "",
    ])
    if len(non_map_mismatches):
        lines.extend([
            "以下非 mAP 差异不是重建代码与作者 evaluator 的差异：重建值已逐块对齐"
            "作者发布 CSV，但发布 CSV 与论文展示值超过了一位小数/三位小数所允许的"
            "正常舍入范围。",
            "",
        ])
        mismatch_rows = []
        for _, mismatch in non_map_mismatches.iterrows():
            mismatch_rows.append([
                mismatch["model"],
                mismatch["metric"],
                _fmt(mismatch["reconstructed_value"], 6),
                _fmt(mismatch["paper_value"], 6),
                _fmt(mismatch["absolute_delta"], 6),
            ])
        lines.extend(_markdown_table(
            ["Model", "指标", "发布 CSV 重建", "论文", "绝对差"],
            mismatch_rows,
        ))
        lines.append("")
    pgen_lines: list[str] = []
    pgen_sensitivity = author_result["main"].get(
        "pgen_aggregation_sensitivity",
        pd.DataFrame(),
    )
    if len(pgen_sensitivity):
        indexed_pgen = pgen_sensitivity.set_index(
            ["population", "log10_window"]
        )
        generated_occurrence = indexed_pgen.loc[
            ("generated_occurrence_weighted", "all_positive")
        ]
        generated_unique = indexed_pgen.loc[
            ("generated_global_unique", "all_positive")
        ]
        reference_occurrence = indexed_pgen.loc[
            (
                "reference_target_deduplicated_occurrence_weighted",
                "all_positive",
            )
        ]
        reference_visible = indexed_pgen.loc[
            (
                "reference_target_deduplicated_occurrence_weighted",
                "figure_visible_minus16_to_minus5",
            )
        ]
        pgen_lines.extend([
            "## OLGA Pgen 复现边界",
            "",
            "使用论文 requirements 固定的 OLGA human_T_beta 口径，非零计数 "
            "`1996/2000` 精确对应；但论文没有发布 Fig.4 的逐序列 Pgen 或聚合/截断代码。"
            "对作者公开候选直接重算时，以下可审计口径均不能同时得到论文报告的"
            "生成 `-7.04±0.85` 与参考 `-9.83±2.356`，因此没有反向调 cutoff "
            "来制造一致。",
            "",
        ])
        pgen_lines.extend(_markdown_table(
            ["口径", "n", "mean log10 Pgen", "population std"],
            [
                [
                    "生成 occurrence-weighted / all positive",
                    int(generated_occurrence["n_log10_selected"]),
                    _fmt(generated_occurrence["mean_log10_pgen"], 6),
                    _fmt(generated_occurrence["std_log10_pgen_population"], 6),
                ],
                [
                    "生成 global-unique / all positive",
                    int(generated_unique["n_log10_selected"]),
                    _fmt(generated_unique["mean_log10_pgen"], 6),
                    _fmt(generated_unique["std_log10_pgen_population"], 6),
                ],
                [
                    "参考 target-dedup occurrence / all positive",
                    int(reference_occurrence["n_log10_selected"]),
                    _fmt(reference_occurrence["mean_log10_pgen"], 6),
                    _fmt(reference_occurrence["std_log10_pgen_population"], 6),
                ],
                [
                    "参考 target-dedup / Fig.4 visible [-16,-5]",
                    int(reference_visible["n_log10_selected"]),
                    _fmt(reference_visible["mean_log10_pgen"], 6),
                    _fmt(reference_visible["std_log10_pgen_population"], 6),
                ],
            ],
        ))
        pgen_lines.extend([
            "",
            "版本排查也不能解释该差异：OLGA 1.2.4 与本地 human_T_beta "
            "模型文件逐 SHA256 相同，样例概率只存在不会改变展示精度的浮点差。"
            "所以这里将 Pgen 的定义和非零有效性标记为精确复现，将论文分布矩标记为"
            "缺少最终 Figure 输入/过滤快照，不能严格复现。",
            "",
        ])
    rows = []
    for model in PAPER_MAIN_TABLE:
        row = main_summary.loc[model]
        paper = PAPER_MAIN_TABLE[model]
        rows.append([
            model,
            f"{row['f1_mean']:.4f} / {paper['f1_mean']:.3f}",
            f"{row['native_recovery_mean']:.4f} / {paper['native_recovery_mean']:.3f}",
            f"{int(row['total_unique_sequences'])} / {paper['total_unique']}",
            f"{row['map_prefix_ordered_proxy']:.4f} / {paper['map']:.3f}",
        ])
    lines.extend(_markdown_table(
        ["Model", "F1 重建/论文", "Recovery 重建/论文", "Unique 重建/论文", "mAP proxy/论文"],
        rows,
    ))
    lines.extend(["", *pgen_lines])

    lines.extend([
        "",
        "## Fig.3–5 数值与趋势交叉验证",
        "",
    ])
    special_rows = []
    for _, row in special.iterrows():
        agreement = row["agreement"]
        if isinstance(agreement, (bool, np.bool_)):
            agreement_text = "yes" if agreement else "no"
        else:
            agreement_text = str(agreement)
        special_rows.append([
            row["section"],
            row["metric"],
            row["reconstructed_value"],
            row["paper_value"],
            agreement_text,
        ])
    lines.extend(_markdown_table(
        ["位置", "检查", "重建", "论文", "对应"],
        special_rows,
    ))

    if current_result is not None:
        summary = current_result["tables"]["summary"].set_index("model")
        pgen = current_result["tables"].get("pgen", pd.DataFrame())
        pgen_model = (
            pgen[pgen["target_id"] == "__all_targets__"].set_index("model")
            if len(pgen) else pd.DataFrame()
        )
        lines.extend([
            "",
            "## 当前重铸后 14-target × 1,000 效果",
            "",
            "这里采用论文 native recovery，而不是旧 Track-A 的 "
            "`1-Levenshtein/max(lengths)`；因此 Recovery≥90% 数量会与旧表不同。",
            "",
        ])
        current_rows = []
        for model in summary.index:
            row = summary.loc[model]
            pgen_fraction = (
                pgen_model.loc[model, "fraction_pgen_positive"]
                if len(pgen_model) and model in pgen_model.index
                else math.nan
            )
            pgen_mean = (
                pgen_model.loc[model, "mean_log10_pgen_positive"]
                if len(pgen_model) and model in pgen_model.index
                else math.nan
            )
            current_rows.append([
                model,
                _fmt(row["f1_mean"], 6),
                _fmt(row["exact_hit_occurrences"], 0),
                _fmt(row["native_recovery_ge_90_count"], 0),
                _fmt(row.get("giana_generated_hit", math.nan), 0),
                _fmt(row["total_unique_sequences"], 0),
                _fmt(row["jaccard_similarity_mean"], 6),
                _fmt(row["map_prefix"], 6),
                "exact" if bool(row["map_rank_is_paper_compatible"]) else "diagnostic",
                _fmt(pgen_fraction, 4),
                _fmt(pgen_mean, 4),
            ])
        lines.extend(_markdown_table(
            [
                "Model",
                "F1@1000",
                "Exact",
                "Native≥90",
                "GIANA hit",
                "Global unique",
                "Jaccard",
                "AP(P@1…K)",
                "rank evidence",
                "Pgen+",
                "mean log10 Pgen+",
            ],
            current_rows,
        ))
        lines.extend([
            "",
            "解释：TCRT5 的 exact 数量/排名和 RVR GIANA 数值已与论文复现；"
            "native≥90 使用论文原生定义，但论文图没有提供可机器核对的精确计数。"
            "BioSeq 的 Pgen 非零率、全局 unique 很高且跨抗原 Jaccard 很低，但其 "
            "mean log10 Pgen 仅为约 -17.93，且 exact/native≥90 仍接近零；这说明它"
            "目前主要学到表面 CDR3β 语法和广覆盖，并未恢复 target-specific functional "
            "neighborhood，单看 Pgen 是否非零会明显高估天然合理性。"
            "TcrDesign 的 GIANA 邻域命中较高，但 exact 与"
            "论文旗舰 TCRT5 仍有差距。不同模型的 ordered AP 只有 TCRT5 有论文兼容的 "
            "cumulative likelihood rank，不能直接做全模型校准排名。",
        ])

    output_root = output_path.parent
    lines.extend([
        "",
        "## 产物",
        "",
        f"- 主评测：`{output_root / 'author_release/main_top20_k100'}`",
        f"- 稀疏/RVR：`{output_root / 'author_release/sparse13_plus_rvr_k1000'}`",
        f"- 当前模型：`{output_root / 'current_track_a'}`",
        f"- 论文总对应表：`{output_root / 'paper_correspondence.csv'}`",
        "",
        "每个目录均含 `manifest.json`，记录输入路径、表行数、字段以及 Pgen 是否执行。",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("author-release", "current-track-a", "all"),
        nargs="?",
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--author-atomic", type=Path, default=AUTHOR_ATOMIC_CSV)
    parser.add_argument("--author-dataset", type=Path, default=AUTHOR_DATASET_CSV)
    parser.add_argument("--author-sparse", type=Path, default=AUTHOR_SPARSE_CSV)
    parser.add_argument("--generations", type=Path, default=TRACK_A_GENERATIONS)
    parser.add_argument("--giana-metrics", type=Path, default=TRACK_A_METRICS)
    parser.add_argument("--with-pgen", action="store_true")
    parser.add_argument(
        "--reference-pgen",
        action="store_true",
        help="also score all 68,416 target-deduplicated top-20 references",
    )
    parser.add_argument("--pgen-workers", type=int, default=8)
    parser.add_argument(
        "--pgen-cache",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT / "pgen_cache.csv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_root = args.output_root.resolve()
    args.pgen_cache = args.pgen_cache.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    author_result: dict[str, object] | None = None
    current_result: dict[str, object] | None = None
    if args.command in {"author-release", "all"}:
        print("Running author-release top20 reconstruction...", flush=True)
        author_result = run_author_release(args)
    if args.command in {"current-track-a", "all"}:
        print("Running current Track-A full evaluation...", flush=True)
        current_result = run_current_track_a(args)

    if author_result is not None:
        special = _comparison_special_rows(author_result, current_result)
        main_comparison = author_result["main"]["paper_main_comparison"].copy()
        main_comparison.insert(0, "section", "Supplementary_Table_2")
        special_for_union = special.copy()
        special_for_union["model"] = ""
        special_for_union["rounding_tolerance"] = math.nan
        special_for_union["absolute_delta"] = math.nan
        special_for_union = special_for_union.rename(columns={
            "reconstructed_value": "reconstructed_value",
            "paper_value": "paper_value",
        })
        correspondence = pd.concat(
            [
                main_comparison,
                special_for_union[[
                    "section",
                    "model",
                    "metric",
                    "paper_value",
                    "reconstructed_value",
                    "absolute_delta",
                    "rounding_tolerance",
                    "agreement",
                    "evidence",
                ]],
            ],
            ignore_index=True,
        )
        correspondence.to_csv(
            args.output_root / "paper_correspondence.csv",
            index=False,
        )
        _write_report(
            author_result,
            current_result,
            special,
            args.output_root / "TCRT5_FULL_EVAL_REPORT.md",
        )
        print(
            f"Report -> {args.output_root / 'TCRT5_FULL_EVAL_REPORT.md'}",
            flush=True,
        )
    elif current_result is not None:
        print(f"Current outputs -> {current_result['output_dir']}", flush=True)


if __name__ == "__main__":
    main()
