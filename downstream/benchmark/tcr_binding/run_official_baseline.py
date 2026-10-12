#!/usr/bin/env python
"""Run one published T1 baseline under the original-only protocol.

Each baseline is executed by its wrapper in ``baselines/wrappers/<name>.py``,
which subprocess-calls the model's OFFICIAL code + OFFICIAL weights in that
model's own conda env and returns a ``DataFrame[group, label, score]`` (grouped
by epitope). This driver writes predictions + official-parity metrics under
``outputs/tcr_binding_nm2025/<name>/<track>/<split>/`` so published baselines
share the exact layout / scoring as Random / kNN / epiTCR / Ours-BioSeq.

Only ``original.zip``-derived test rows are loaded.  ``train.csv`` is neither
opened nor passed to a wrapper, and no model/head is fitted.  If the official
checkpoint/code/environment is unavailable, this command exits without making
up predictions; ``scripts/summarize_nm2025.py`` then selects the paper value and
labels it ``论文值，未本地复现``.

Example::

    /vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
      /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_official_baseline.py \
      --name teinet --tag teinet_small --track cdr3b --neg-source AS
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.baseline_provenance import (  # noqa: E402
    COMPATIBLE,
    EXACT,
    OFFICIAL_CODE_RERUN,
    baseline_provenance,
)
from tcr_binding.original_protocol import (  # noqa: E402
    ORIGINAL_TEST_MANIFEST,
    ORIGINAL_ZIP,
    ORIGINAL_ZIP_SHA256,
    T1_CITATION,
    get_original_spec,
)
from tcr_binding.run_nm2025 import _load_tests, _split_dirname, _metrics_block  # noqa: E402

OUT = BENCH / "outputs" / "tcr_binding_nm2025"
DATA = BENCH / "data" / "tcr_binding_nm2025"
VARIANT_WRAPPERS = {"teinet", "ergo", "tpbte"}
THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "NUMEXPR_NUM_THREADS": "4",
}


def _write_run_status(tag: str, track: str, payload: dict) -> Path:
    status_dir = OUT / tag / track
    status_dir.mkdir(parents=True, exist_ok=True)
    status_path = status_dir / "original_run_status.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return status_path


def _configure_variant(name: str, tag: str):
    """Validate wrapper↔artifact mapping and configure it before import."""

    spec = get_original_spec(tag)
    if spec is not None and spec.wrapper != name:
        raise ValueError(
            f"tag {tag!r} belongs to wrapper {spec.wrapper!r}, not {name!r}"
        )
    if name in VARIANT_WRAPPERS:
        if spec is None or spec.variant_tag is None:
            raise ValueError(
                f"{name} requires a canonical --tag selecting its checkpoint variant"
            )
        # This must happen before import: the wrappers bind NAME at module load.
        os.environ["TCRBENCH_BASELINE_TAG"] = spec.variant_tag
    return spec


def _verify_original_zip() -> None:
    if not ORIGINAL_ZIP.is_file():
        raise FileNotFoundError(f"official original.zip is missing: {ORIGINAL_ZIP}")
    digest = _sha256_file(ORIGINAL_ZIP)
    if digest != ORIGINAL_ZIP_SHA256:
        raise RuntimeError(
            f"unexpected original.zip sha256={digest}; expected={ORIGINAL_ZIP_SHA256}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_normalized_test(path: Path) -> str:
    """Verify a normalized test file against the original.zip extraction manifest."""

    if not ORIGINAL_TEST_MANIFEST.is_file():
        raise FileNotFoundError(
            f"missing original test manifest: {ORIGINAL_TEST_MANIFEST}; run "
            "scripts/prepare_nm2025_binding.py --tests-only"
        )
    try:
        manifest = json.loads(ORIGINAL_TEST_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid original test manifest: {ORIGINAL_TEST_MANIFEST}") from exc
    if manifest.get("source_artifact") != str(ORIGINAL_ZIP):
        raise RuntimeError("original test manifest points to a different source artifact")
    if manifest.get("source_artifact_sha256") != ORIGINAL_ZIP_SHA256:
        raise RuntimeError("original test manifest has the wrong original.zip checksum")
    try:
        relative = str(path.relative_to(DATA))
    except ValueError as exc:
        raise RuntimeError(f"normalized test is outside the benchmark data root: {path}") from exc
    expected = manifest.get("files", {}).get(relative, {}).get("sha256")
    if not expected:
        raise RuntimeError(f"normalized test is absent from original test manifest: {relative}")
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"normalized test checksum mismatch for {relative}: {actual} != {expected}"
        )
    return actual


def _checkpoint_paths_for(wrapper, spec) -> tuple[Path, ...]:
    """Require wrappers to disclose the exact model artifacts they open."""

    reporter = getattr(wrapper, "official_checkpoint_paths", None)
    if not callable(reporter):
        raise ValueError("official wrapper does not report its runtime checkpoint paths")
    reported = tuple(Path(path) for path in reporter())
    if not reported or any(not path.is_absolute() for path in reported):
        raise ValueError("official wrapper reported empty or non-absolute checkpoint paths")
    if spec is not None and reported != spec.checkpoint_paths:
        raise ValueError(
            f"wrapper runtime artifacts {reported!r} do not match catalog "
            f"artifacts {spec.checkpoint_paths!r}"
        )
    if spec is not None and not spec.artifact_identity_verified:
        raise ValueError("runtime checkpoint copy does not match its bundled official alias")
    return reported


def _validate_predictions(test_df: pd.DataFrame, preds: pd.DataFrame) -> None:
    required = {"group", "label", "score"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"official wrapper output is missing columns: {sorted(missing)}")
    if len(preds) != len(test_df):
        raise ValueError(f"official wrapper scored {len(preds)} rows for {len(test_df)} inputs")
    if not pd.to_numeric(preds["score"], errors="coerce").notna().all():
        raise ValueError("official wrapper produced non-numeric/NaN scores")

    # Wrappers may preserve or stably sort rows.  Compare the label/group
    # multiset so sorting is allowed but silent dropping/relabeling is not.
    expected = (
        pd.DataFrame({
            "group": test_df["peptide"].astype(str),
            "label": test_df["label"].astype(int),
        })
        .value_counts(sort=False)
        .sort_index()
    )
    actual = (
        preds.assign(
            group=preds["group"].astype(str),
            label=preds["label"].astype(int),
        )[["group", "label"]]
        .value_counts(sort=False)
        .sort_index()
    )
    if not expected.equals(actual):
        raise ValueError("official wrapper changed the test group/label multiset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="official wrapper module name")
    ap.add_argument("--track", choices=["cdr3b", "others"], default="cdr3b")
    ap.add_argument(
        "--tag", default=None,
        help="canonical output/artifact tag; required for TEINet/ERGO/TPBTE variants",
    )
    ap.add_argument("--split", choices=["all", "seen", "unseen"], default="all")
    ap.add_argument("--neg-source", choices=["all", "AS", "PS", "HS"], default="all")
    args = ap.parse_args()

    # Old baseline stacks often default to every visible CPU core, which makes
    # concurrent official runs dramatically slower. Respect explicit caller
    # settings and otherwise use a conservative, deterministic thread cap.
    for variable, value in THREAD_ENV_DEFAULTS.items():
        os.environ.setdefault(variable, value)

    tag = args.tag or args.name
    try:
        spec = _configure_variant(args.name, tag)
    except ValueError as exc:
        raise SystemExit(f"[artifact mapping error] {exc}") from exc

    _verify_original_zip()
    wrapper = importlib.import_module(f"baselines.wrappers.{args.name}")
    try:
        checkpoint_paths = _checkpoint_paths_for(wrapper, spec)
    except ValueError as exc:
        _write_run_status(tag, args.track, {
            "status": "paper_fallback",
            "tag": tag,
            "wrapper": args.name,
            "catalog_checkpoint_paths": (
                [str(path) for path in spec.checkpoint_paths] if spec is not None else []
            ),
            "reason": f"official artifact mapping error: {exc}",
            "result_label": "论文值，未本地复现",
            "test_artifact": str(ORIGINAL_ZIP),
            "test_artifact_sha256": ORIGINAL_ZIP_SHA256,
            "train_csv_loaded": False,
            "retrained": False,
        })
        raise SystemExit(f"[artifact mapping error] {exc}") from exc
    if not wrapper.is_available():
        _write_run_status(tag, args.track, {
            "status": "paper_fallback",
            "tag": tag,
            "wrapper": args.name,
            "checkpoint_paths": [str(path) for path in checkpoint_paths],
            "reason": "official checkpoint/code/environment unavailable",
            "result_label": "论文值，未本地复现",
            "train_csv_loaded": False,
            "retrained": False,
        })
        raise SystemExit(
            f"[paper fallback] {tag}: official checkpoint/code/env unavailable; "
            "do not retrain. Run scripts/summarize_nm2025.py to select the "
            "paper original-model value as '论文值，未本地复现'."
        )

    tests = {
        (split, neg): frame
        for (split, neg), frame in _load_tests(args.track).items()
        if (args.split == "all" or split == args.split)
        and (args.neg_source == "all" or neg == args.neg_source)
    }
    if not tests:
        raise SystemExit(
            f"no original.zip-derived tests for track={args.track}, "
            f"split={args.split}, neg_source={args.neg_source}"
        )
    test_hashes = {}
    for split, neg in tests:
        normalized_test = DATA / split / args.track / neg / "test.csv"
        test_hashes[(split, neg)] = _verify_normalized_test(normalized_test)
    model_name = getattr(wrapper, "NAME", args.name)

    summary = {}
    for (split, neg), df in tests.items():
        t0 = time.time()
        try:
            preds = wrapper.predict_official(df)
            _validate_predictions(df, preds)
        except Exception as exc:
            status_path = _write_run_status(tag, args.track, {
                "status": "paper_fallback",
                "tag": tag,
                "wrapper": args.name,
                "variant_tag": spec.variant_tag if spec is not None else None,
                "checkpoint_paths": [str(path) for path in checkpoint_paths],
                "failed_split": split,
                "failed_neg_source": neg,
                "reason": f"{type(exc).__name__}: {exc}",
                "result_label": "论文值，未本地复现",
                "test_artifact": str(ORIGINAL_ZIP),
                "test_artifact_sha256": ORIGINAL_ZIP_SHA256,
                "test_manifest": str(ORIGINAL_TEST_MANIFEST),
                "normalized_test_sha256": test_hashes[(split, neg)],
                "train_csv_loaded": False,
                "retrained": False,
            })
            print(
                f"[paper fallback] {tag} failed official inference for {split}/{neg}; "
                f"status -> {status_path}",
                file=sys.stderr,
            )
            raise
        elapsed = time.time() - t0
        out_dir = OUT / tag / args.track / _split_dirname(split, neg)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred = pd.DataFrame({
            "id": [f"{split}_{neg}_{i}" for i in range(len(preds))],
            "score": preds["score"].astype(float),
            "label": preds["label"].astype(int),
            "group": preds["group"].astype(str),
            "seen": split,
        })
        pred.to_csv(out_dir / "predictions.csv", index=False)
        block = _metrics_block(pred)
        normalized_test = DATA / split / args.track / neg / "test.csv"
        alignment = EXACT if spec is not None and args.track == "cdr3b" and neg == "AS" else COMPATIBLE
        provenance = baseline_provenance(
            evidence_type=OFFICIAL_CODE_RERUN,
            protocol_alignment=alignment,
            citation=T1_CITATION,
            source_location=(
                f"{normalized_test} (extracted from {ORIGINAL_ZIP}; "
                f"sha256={ORIGINAL_ZIP_SHA256})"
            ),
            notes=(
                f"wrapper=baselines.wrappers.{args.name}; tag={tag}; "
                "official original checkpoint; no train.csv loaded; no retraining"
            ),
        )
        result = {
            "method": "official_baseline", "model": model_name, "track": args.track,
            "split": split, "neg_source": neg, "elapsed_sec": round(elapsed, 1),
            "n_scored": int(len(pred)), "n_input": int(len(df)),
            "baseline_provenance": provenance,
            "baseline_protocol": {
                "name": "nm2025_original_only",
                "checkpoint": "official_original",
                "checkpoint_paths": [str(path) for path in checkpoint_paths],
                "inference_code": "official",
                "wrapper": args.name,
                "variant_tag": spec.variant_tag if spec is not None else None,
                "test_artifact": str(ORIGINAL_ZIP),
                "test_artifact_sha256": ORIGINAL_ZIP_SHA256,
                "test_manifest": str(ORIGINAL_TEST_MANIFEST),
                "normalized_test": str(normalized_test),
                "normalized_test_sha256": test_hashes[(split, neg)],
                "train_csv_loaded": False,
                "retrained": False,
            },
            **block,
        }
        with (out_dir / "metrics.json").open("w") as fh:
            json.dump(result, fh, indent=2)
        summary[f"{split}/{neg}"] = block

    _write_run_status(tag, args.track, {
        "status": "official_code_rerun_complete",
        "tag": tag,
        "wrapper": args.name,
        "variant_tag": spec.variant_tag if spec is not None else None,
        "checkpoint_paths": [str(path) for path in checkpoint_paths],
        "completed_tests": [f"{split}/{neg}" for split, neg in tests],
        "test_artifact": str(ORIGINAL_ZIP),
        "test_artifact_sha256": ORIGINAL_ZIP_SHA256,
        "test_manifest": str(ORIGINAL_TEST_MANIFEST),
        "normalized_test_sha256": {
            f"{split}/{neg}": test_hashes[(split, neg)] for split, neg in tests
        },
        "train_csv_loaded": False,
        "retrained": False,
    })

    print(f"\n=== {model_name} | track={args.track} ===")
    for key in sorted(summary):
        b = summary[key]
        print(f"  {key:12s}: AUPRC={b['auprc_macro']:.3f}(macro)/{b['auprc_overall']:.3f}(all) "
              f"AUROC={b['auroc_macro']:.3f}/{b['auroc_overall']:.3f} n_ep={b['n_epitopes']}")
    print(f"  -> {OUT / tag / args.track}")


if __name__ == "__main__":
    main()
