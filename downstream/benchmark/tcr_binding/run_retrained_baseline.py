#!/usr/bin/env python
"""Evaluate published retrained TCR checkpoints without retraining.

Run every currently connected model on the official AS five-fold protocol::

    source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
    conda activate /vepfs-mlp2/c20250601/251105016/conda/envs/pllm
    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_retrained_baseline.py \
      --name all --fold all --eval-set all --neg-source AS

The driver reads CSV bytes directly from official ``retrain.zip``, pairs fold N
with the official fold-N checkpoint nested in ``Retraining_model.zip``, and
calls each author's inference implementation in its pinned environment.  No
parameter is fitted.  TCR-H additionally reads fold train data solely to replay
its published inference-time correlation feature selection.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.metrics import precrec_pr_auc  # noqa: E402
from tcr_binding.retrained_protocol import (  # noqa: E402
    FIGSHARE_DOI,
    MODEL_SPECS,
    PAPER_EXPECTED_AS,
    RETRAIN_DATA_ZIP,
    RETRAIN_MODELS_ZIP,
    SUPPLEMENTARY_TABLE_7,
    T1_CITATION,
    ensure_checkpoint,
    load_official_train,
    load_official_test,
    sha256_file,
    verify_official_archives,
)


OUT_ROOT = BENCH / "outputs" / "tcr_binding_nm2025_retrained"
EVAL_SETS = ("seen_test", "seen_independent", "unseen_independent")
FOLDS = (1, 2, 3, 4, 5)
THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}
TCRH_PREDICTION_JOBS = 8


def _command_for(
    spec,
    checkpoint: Path,
    input_csv: Path,
    output_csv: Path,
    *,
    train_csv: Path | None = None,
    feature_cache: Path | None = None,
) -> list[str]:
    if spec.infer_kind == "atmtcr":
        return [
            str(spec.env_python), str(spec.infer_script),
            "--atm-dir", str(spec.official_code_dir),
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
        ]
    if spec.infer_kind == "nettcr":
        return [
            str(spec.env_python), str(spec.infer_script),
            "--nettcr-dir", str(spec.official_code_dir),
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
        ]
    if spec.infer_kind == "epitcr":
        return [
            str(spec.env_python), str(spec.infer_script),
            "--modules-dir", str(spec.official_code_dir),
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
            "--chain", "ce",
            "--feature-layout", "retrained15",
        ]
    if spec.infer_kind == "teim_retrained":
        return [
            str(spec.env_python), str(spec.infer_script),
            "--teim-dir", str(spec.official_code_dir),
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
        ]
    if spec.infer_kind == "teinet":
        return [
            str(spec.env_python), str(spec.infer_script),
            "--teinet-dir", str(spec.official_code_dir),
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
        ]
    if spec.infer_kind in {"ergo_ae_retrained", "ergo_lstm_retrained"}:
        model_type = "ae" if spec.infer_kind == "ergo_ae_retrained" else "lstm"
        return [
            str(spec.env_python), str(spec.infer_script),
            "--ergo-dir", str(spec.official_code_dir),
            "--model-type", model_type,
            "--model", str(checkpoint),
            "--input", str(input_csv),
            "--output", str(output_csv),
            "--drop-remainder",
        ]
    if spec.infer_kind == "tcrh_retrained":
        if train_csv is None or feature_cache is None:
            raise ValueError("TCR-H inference requires train_csv and feature_cache")
        return [
            str(spec.env_python), str(spec.infer_script),
            "--model", str(checkpoint),
            "--train", str(train_csv),
            "--input", str(input_csv),
            "--output", str(output_csv),
            "--feature-cache", str(feature_cache),
            "--jobs", str(TCRH_PREDICTION_JOBS),
        ]
    raise ValueError(f"unsupported inference kind: {spec.infer_kind}")


def _validate_environment(spec) -> None:
    missing = [
        path for path in (
            spec.env_python,
            spec.infer_script,
            spec.official_code_dir,
            *spec.auxiliary_artifacts,
        )
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"official inference runtime is incomplete: {missing}")


def _row_multiset(frame: pd.DataFrame, *, result: bool) -> Counter:
    if result:
        columns = ("Epitope", "CDR3B", "y_true")
    else:
        columns = ("Epitope", "CDR3B", "Affinity")
    return Counter(
        zip(
            frame[columns[0]].astype(str),
            frame[columns[1]].astype(str),
            frame[columns[2]].astype(int),
        )
    )


def _validate_predictions(test: pd.DataFrame, result: pd.DataFrame, spec) -> None:
    required = {"Epitope", "CDR3B", "y_true", "y_prob"}
    missing = required - set(result.columns)
    if missing:
        raise RuntimeError(f"official prediction output lacks {sorted(missing)}")
    expected = test
    if spec.drop_remainder_batch_size is not None:
        batch_size = spec.drop_remainder_batch_size
        expected = test.iloc[: len(test) // batch_size * batch_size]
    if len(result) != len(expected):
        raise RuntimeError(
            f"official code scored {len(result)} rows; protocol expects "
            f"{len(expected)} of {len(test)}"
        )
    scores = pd.to_numeric(result["y_prob"], errors="coerce")
    if not np.isfinite(scores).all():
        raise RuntimeError("official code produced a non-finite score")
    if _row_multiset(expected, result=False) != _row_multiset(result, result=True):
        raise RuntimeError("official code changed the test sequence/label multiset")


def _paper_metrics(result: pd.DataFrame) -> dict:
    labels = result["y_true"].astype(int).to_numpy()
    scores = result["y_prob"].astype(float).to_numpy()
    predicted = (scores > 0.5).astype(int)
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc_precrec": precrec_pr_auc(labels, scores),
        "average_precision_sklearn": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def _output_dir(tag: str, neg_source: str, fold: int, eval_set: str) -> Path:
    return OUT_ROOT / tag / "cdr3b" / neg_source / f"fold_{fold}" / eval_set


def _compatibility_repairs(infer_kind: str) -> list[str]:
    if infer_kind == "epitcr":
        return [
            "official epiTCR retrained pickle expects 680 features; corrected "
            "notebook processor reshape from 220 to 300 for its documented "
            "15-aa peptide encoding"
        ]
    if infer_kind == "teim_retrained":
        return [
            "released TEIM state dict contains a 15-aa AE (32x480), while the "
            "checked-in YAML/dataset remains at 12 aa; instantiated the official "
            "TEIM classes at the checkpoint-required shape and used the notebook's "
            "published 15-aa encoding"
        ]
    if infer_kind == "tcrh_retrained":
        return [
            "released TCR-H inference reconstructs its correlation-filtered "
            "descriptor columns from the official fold train CSV and scales "
            "each test set independently; this is preprocessing only, not fitting",
            "checkpoint metadata reports scikit-learn 1.6.1 although the released "
            "environment YAML pins 1.5.2; inference uses 1.6.1 for checkpoint "
            "compatibility",
        ]
    if infer_kind in {"ergo_ae_retrained", "ergo_lstm_retrained"}:
        repairs = [
            "released retrained checkpoints are raw state_dict objects while "
            "original checkpoints wrap model_state_dict; the official model "
            "loader accepts the published retrained serialization",
        ]
        if infer_kind == "ergo_ae_retrained":
            repairs.insert(
                0,
                "released ERGO-AE retraining notebook uses ae_utils.get_batches(), "
                "which intentionally omits the final partial 50-row batch; the "
                "rerun preserves and explicitly reports that published behavior",
            )
        return repairs
    return []


def run_one(*, tag: str, neg_source: str, fold: int, eval_set: str, force: bool) -> dict:
    spec = MODEL_SPECS[tag]
    _validate_environment(spec)
    out_dir = _output_dir(tag, neg_source, fold, eval_set)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file() and not force:
        prior = json.loads(metrics_path.read_text(encoding="utf-8"))
        print(f"[reuse] {tag} fold={fold} {eval_set}: {metrics_path}")
        return prior

    checkpoint, checkpoint_meta = ensure_checkpoint(
        spec,
        neg_source=neg_source,
        fold=fold,
    )
    test, test_payload, test_meta = load_official_test(
        eval_set=eval_set,
        neg_source=neg_source,
        fold=fold,
    )
    train_payload = None
    train_meta = {}
    feature_cache = None
    if spec.infer_kind == "tcrh_retrained":
        _, train_payload, train_meta = load_official_train(
            neg_source=neg_source,
            fold=fold,
        )
        feature_cache = (
            OUT_ROOT / tag / "cdr3b" / neg_source / f"fold_{fold}"
            / "tcrh_feature_columns.json"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"nm2025_{tag}_f{fold}_") as tmp:
        tmp_dir = Path(tmp)
        input_csv = tmp_dir / "official_test.csv"
        train_csv = None
        if train_payload is not None:
            train_csv = tmp_dir / "official_train.csv"
            train_csv.write_bytes(train_payload)
        official_output = tmp_dir / (
            "probability.npy" if spec.infer_kind == "epitcr" else "probability.csv"
        )
        input_csv.write_bytes(test_payload)
        env = dict(os.environ)
        for variable, value in THREAD_ENV_DEFAULTS.items():
            env.setdefault(variable, value)
        env["LD_LIBRARY_PATH"] = str(spec.env_python.parent.parent / "lib")
        subprocess.run(
            _command_for(
                spec,
                checkpoint,
                input_csv,
                official_output,
                train_csv=train_csv,
                feature_cache=feature_cache,
            ),
            check=True,
            env=env,
        )
        if spec.infer_kind == "epitcr":
            scores = np.load(official_output).astype(float)
            result = pd.DataFrame({
                "Epitope": test["Epitope"].astype(str),
                "CDR3B": test["CDR3B"].astype(str),
                "y_true": test["Affinity"].astype(int),
                "y_prob": scores,
            })
        else:
            result = pd.read_csv(official_output)
    _validate_predictions(test, result, spec)
    elapsed = time.time() - started

    preprocessing_meta = {}
    if feature_cache is not None:
        cache_payload = json.loads(feature_cache.read_text(encoding="utf-8"))
        preprocessing_meta = {
            "feature_selection_cache_path": str(feature_cache),
            "feature_selection_cache_sha256": sha256_file(feature_cache),
            "feature_selection_protocol": cache_payload["protocol"],
            "feature_selection_runtime_versions": cache_payload["runtime_versions"],
            "n_raw_descriptor_features": cache_payload["n_raw_descriptor_features"],
            "n_dropped_correlated_features": cache_payload[
                "n_dropped_correlated_features"
            ],
            "n_checkpoint_features": cache_payload["n_kept_features"],
        }

    prediction = pd.DataFrame({
        "peptide": result["Epitope"].astype(str),
        "cdr3b": result["CDR3B"].astype(str),
        "label": result["y_true"].astype(int),
        "score": result["y_prob"].astype(float),
    })
    prediction.to_csv(out_dir / "predictions.csv", index=False)
    scored = _paper_metrics(result)
    payload = {
        "status": "official_retrained_checkpoint_rerun_complete",
        "method": spec.display_name,
        "tag": tag,
        "track": "cdr3b",
        "neg_source": neg_source,
        "fold": fold,
        "eval_set": eval_set,
        "n_input": int(len(test)),
        "n_scored": int(len(result)),
        "n_unscored_by_official_protocol": int(len(test) - len(result)),
        "elapsed_sec": round(elapsed, 3),
        "metrics": scored,
        "metric_protocol": {
            "paper_headline_auprc": "precrec::evalmod default-compatible PRC-AUC",
            "precrec_x_bins": 1000,
            "diagnostic_only": "sklearn average_precision_score",
            "paper_aggregation": "pooled all_values per fold; five-fold mean and sample SD",
        },
        "baseline_protocol": {
            "name": "nm2025_official_retrained_five_fold",
            "checkpoint": "official_retrained",
            "inference_code": "official",
            "training_performed": False,
            "train_csv_loaded": train_payload is not None,
            "train_csv_use": (
                "inference_time_feature_selection_only"
                if train_payload is not None else None
            ),
            "prediction_jobs": (
                TCRH_PREDICTION_JOBS
                if spec.infer_kind == "tcrh_retrained" else 1
            ),
            "citation": T1_CITATION,
            "artifact_source": FIGSHARE_DOI,
            "paper_table_source": SUPPLEMENTARY_TABLE_7,
            "compatibility_repairs": _compatibility_repairs(spec.infer_kind),
            "models_archive": str(RETRAIN_MODELS_ZIP),
            "data_archive": str(RETRAIN_DATA_ZIP),
            **checkpoint_meta,
            **test_meta,
            **train_meta,
            **preprocessing_meta,
        },
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[done] {tag} fold={fold} {eval_set}: "
        f"AUROC={scored['auroc']:.6f} "
        f"AUPRC(precrec)={scored['auprc_precrec']:.6f} "
        f"AP={scored['average_precision_sklearn']:.6f} "
        f"n={len(result)} sec={elapsed:.1f}"
    )
    return payload


def summarize(tag: str, neg_source: str) -> dict:
    """Aggregate every complete five-fold result and compare with Table 7."""

    summary = {
        "tag": tag,
        "method": MODEL_SPECS[tag].display_name,
        "track": "cdr3b",
        "neg_source": neg_source,
        "folds_expected": list(FOLDS),
        "paper_table_source": SUPPLEMENTARY_TABLE_7,
        "eval_sets": {},
    }
    for eval_set in EVAL_SETS:
        rows = []
        for fold in FOLDS:
            path = _output_dir(tag, neg_source, fold, eval_set) / "metrics.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
        record = {"folds_complete": [row["fold"] for row in rows]}
        if len(rows) == len(FOLDS):
            auroc = np.asarray([row["metrics"]["auroc"] for row in rows])
            auprc = np.asarray([row["metrics"]["auprc_precrec"] for row in rows])
            ap = np.asarray([
                row["metrics"]["average_precision_sklearn"] for row in rows
            ])
            local = {
                "auroc_mean": float(auroc.mean()),
                "auroc_std_sample": float(auroc.std(ddof=1)),
                "auprc_precrec_mean": float(auprc.mean()),
                "auprc_precrec_std_sample": float(auprc.std(ddof=1)),
                "average_precision_mean": float(ap.mean()),
                "average_precision_std_sample": float(ap.std(ddof=1)),
            }
            paper = PAPER_EXPECTED_AS[tag][eval_set] if neg_source == "AS" else None
            comparison = None
            if paper is not None:
                comparison = {
                    "paper": paper,
                    "delta_auroc_mean": local["auroc_mean"] - paper["auroc"],
                    "delta_auprc_mean": local["auprc_precrec_mean"] - paper["auprc"],
                    "mean_rounds_to_paper": {
                        "auroc": round(local["auroc_mean"], 4) == paper["auroc"],
                        "auprc": round(local["auprc_precrec_mean"], 4) == paper["auprc"],
                    },
                }
            record.update({"local": local, "paper_comparison": comparison})
        summary["eval_sets"][eval_set] = record

    summary_path = OUT_ROOT / tag / "cdr3b" / neg_source / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def write_comparison(neg_source: str) -> list[dict]:
    """Write a single local-vs-paper table with an explicit source decision."""

    rows = []
    for tag, spec in MODEL_SPECS.items():
        path = OUT_ROOT / tag / "cdr3b" / neg_source / "summary.json"
        if not path.is_file():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        for eval_set, record in summary["eval_sets"].items():
            if "local" not in record or record.get("paper_comparison") is None:
                continue
            local = record["local"]
            comparison = record["paper_comparison"]
            paper = comparison["paper"]
            matches = comparison["mean_rounds_to_paper"]
            exact = bool(matches["auroc"] and matches["auprc"])
            source = (
                "official_retrained_checkpoint_rerun"
                if exact else "paper_reported_retrained"
            )
            label = (
                "本地复现：官方 retrained checkpoint（均值四位小数与论文一致）"
                if exact else "论文值，发布 artifact 未精确复现"
            )
            rows.append({
                "tag": tag,
                "method": spec.display_name,
                "eval_set": eval_set,
                "neg_source": neg_source,
                "folds": len(record["folds_complete"]),
                "local_auroc_mean": local["auroc_mean"],
                "local_auroc_std": local["auroc_std_sample"],
                "local_auprc_mean": local["auprc_precrec_mean"],
                "local_auprc_std": local["auprc_precrec_std_sample"],
                "paper_auroc_mean": paper["auroc"],
                "paper_auroc_std": paper["auroc_std"],
                "paper_auprc_mean": paper["auprc"],
                "paper_auprc_std": paper["auprc_std"],
                "delta_auroc_mean": comparison["delta_auroc_mean"],
                "delta_auprc_mean": comparison["delta_auprc_mean"],
                "local_matches_paper_4dp": exact,
                "selected_source": source,
                "selected_result_label": label,
                "selected_auroc_mean": (
                    local["auroc_mean"] if exact else paper["auroc"]
                ),
                "selected_auprc_mean": (
                    local["auprc_precrec_mean"] if exact else paper["auprc"]
                ),
            })
    comparison_csv = OUT_ROOT / f"comparison_{neg_source}.csv"
    pd.DataFrame(rows).to_csv(comparison_csv, index=False)
    comparison_json = OUT_ROOT / f"comparison_{neg_source}.json"
    comparison_json.write_text(
        json.dumps({
            "protocol": "nm2025_official_retrained_five_fold",
            "neg_source": neg_source,
            "selection_rule": (
                "use local only when AUROC and precrec AUPRC means both round "
                "to the paper's four decimals; otherwise cite the marked paper value"
            ),
            "paper_table_source": SUPPLEMENTARY_TABLE_7,
            "rows": rows,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["all", *MODEL_SPECS], default="all")
    parser.add_argument("--fold", choices=["all", "1", "2", "3", "4", "5"], default="all")
    parser.add_argument("--eval-set", choices=["all", *EVAL_SETS], default="all")
    parser.add_argument("--neg-source", choices=["AS", "PS", "HS"], default="AS")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    integrity = verify_official_archives()
    print(f"[verified] {len(integrity['artifacts'])} official archives")
    if args.verify_only:
        print(json.dumps(integrity, ensure_ascii=False, indent=2))
        return

    tags = list(MODEL_SPECS) if args.name == "all" else [args.name]
    folds = list(FOLDS) if args.fold == "all" else [int(args.fold)]
    eval_sets = list(EVAL_SETS) if args.eval_set == "all" else [args.eval_set]
    for tag in tags:
        for fold in folds:
            for eval_set in eval_sets:
                run_one(
                    tag=tag,
                    neg_source=args.neg_source,
                    fold=fold,
                    eval_set=eval_set,
                    force=args.force,
                )
        aggregated = summarize(tag, args.neg_source)
        print(json.dumps(aggregated, ensure_ascii=False, indent=2))
    comparison = write_comparison(args.neg_source)
    print(
        f"[comparison] {len(comparison)} rows -> "
        f"{OUT_ROOT / f'comparison_{args.neg_source}.csv'}"
    )


if __name__ == "__main__":
    main()
