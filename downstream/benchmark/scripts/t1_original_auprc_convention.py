#!/usr/bin/env python
"""T1 -- which AUPRC convention reproduces the NM2025 original-protocol paper values?

The Nature Methods 2025 benchmark computes its reported AUPRC with R
``precrec::evalmod`` + ``precrec::auc``, which uses tie-aware nonlinear PR
interpolation and trapezoidal integration. sklearn's
``average_precision_score`` is a step-wise sum and is *not* the same estimator;
they diverge most when a model produces saturated or heavily tied scores.

The retrained protocol (``run_retrained_*.py``) already reports the
precrec-compatible port. The original protocol (``run_nm2025.py`` ->
``common.metrics.official_binding_report``) reports sklearn AP instead, while
§T1 of RESULTS.md places those ``[R]`` local values in the same columns as
``[P]`` paper values. This script measures the size of that mismatch: for every
original-protocol method with local predictions, it recomputes both estimators
and compares each against the 58 paper-reported rows
(Supplementary Table 4-1 / 4-2, AS).

Methods without a paper row (Ours-* / kNN / Random) are still recomputed under
both estimators, because the RESULTS.md tables place them in the same columns as
the paper values and therefore inherit the same convention requirement.

Outputs:
    outputs/tcr_binding_nm2025/auprc_convention_audit.csv
    outputs/tcr_binding_nm2025/auprc_convention_audit.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))

from common.metrics import _safe_auprc, precrec_pr_auc  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = BENCH / "outputs" / "tcr_binding_nm2025"
PAPER = BENCH / "outputs" / "external" / "paper_reported_baselines.csv"
TRACK = "cdr3b"

# local artifact dir -> method name used in the paper table
METHOD_MAP = {
    "atmtcr": "ATM-TCR",
    "nettcr": "NetTCR",
    "teim": "TEIM",
    "epitcr_nomhc": "epiTCR",
    "panpep": "PanPep",
    "teinet_large": "TEINet-large",
    "teinet_small": "TEINet-small",
    "ergo_ae_mc": "ERGO-AE-mc",
    "ergo_ae_vdj": "ERGO-AE-vdj",
    "ergo_lstm_mc": "ERGO-lstm-mc",
    "ergo_lstm_vdj": "ERGO-lstm-vdj",
    "tpbte_mc": "TPBTE-mc",
    "tpbte_vdj": "TPBTE-vdj",
}


def main() -> None:
    paper = pd.read_csv(PAPER)
    paper = paper[paper["task"] == "original_cdr3b_as"]
    pv = paper.pivot_table(index="method", columns=["split", "metric"], values="value")

    artifacts = sorted({p.parents[2].name for p in
                        OUT.glob(f"*/{TRACK}/*/predictions.csv")})
    rows = []
    for adir in artifacts:
        name = METHOD_MAP.get(adir, adir)
        for split, sub in (("seen", "seen"), ("unseen", "unseen")):
            f = OUT / adir / TRACK / sub / "predictions.csv"
            if not f.exists():
                continue
            p = pd.read_csv(f)
            y, s = p["label"].to_numpy().astype(int), p["score"].to_numpy(dtype=float)
            ap = _safe_auprc(y, s)
            pr = precrec_pr_auc(y, s)
            roc = float(roc_auc_score(y, s))
            paper_auprc = pv.loc[name, (split, "auprc")] if name in pv.index else np.nan
            paper_auroc = pv.loc[name, (split, "auroc")] if name in pv.index else np.nan
            n_uniq = int(len(np.unique(s)))
            rows.append({
                "method": name, "artifact": adir, "split": split,
                "has_paper_row": bool(name in pv.index),
                "n": len(p), "n_unique_scores": n_uniq,
                "tie_frac": round(1 - n_uniq / len(p), 4),
                "auprc_sklearn_ap": ap,
                "auprc_precrec": pr,
                "auprc_paper": paper_auprc,
                "d_ap_vs_paper": ap - paper_auprc,
                "d_precrec_vs_paper": pr - paper_auprc,
                "auroc_local": roc,
                "auroc_paper": paper_auroc,
                "d_auroc": roc - paper_auroc,
            })

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "auprc_convention_audit.csv", index=False)

    summary = {}
    for split in ("seen", "unseen"):
        d = df[(df["split"] == split)].dropna(subset=["auprc_paper"])
        # Rows whose AUROC already agrees isolate the metric-convention effect
        # from methods whose *run* diverges from the paper's run.
        agree = d[d["d_auroc"].abs() <= 0.001]
        summary[split] = {
            "n_methods": int(len(d)),
            "auroc_agreeing_rows": {
                "n": int(len(agree)),
                "precrec_mae": float(agree["d_precrec_vs_paper"].abs().mean()),
                "precrec_max_abs": float(agree["d_precrec_vs_paper"].abs().max()),
                "sklearn_ap_mae": float(agree["d_ap_vs_paper"].abs().mean()),
                "sklearn_ap_n_positive": int((agree["d_ap_vs_paper"] > 0).sum()),
                "sklearn_ap_n_negative": int((agree["d_ap_vs_paper"] < 0).sum()),
            },
            "run_divergent_rows": sorted(
                f"{r.method}/{r.split}" for r in
                d[d["d_auroc"].abs() > 0.001].itertuples()),
            "sklearn_ap": {
                "mae": float(d["d_ap_vs_paper"].abs().mean()),
                "max_abs": float(d["d_ap_vs_paper"].abs().max()),
                "n_within_0.001": int((d["d_ap_vs_paper"].abs() <= 0.001).sum()),
                "worst": d.loc[d["d_ap_vs_paper"].abs().idxmax(), "method"],
            },
            "precrec": {
                "mae": float(d["d_precrec_vs_paper"].abs().mean()),
                "max_abs": float(d["d_precrec_vs_paper"].abs().max()),
                "n_within_0.001": int((d["d_precrec_vs_paper"].abs() <= 0.001).sum()),
                "worst": d.loc[d["d_precrec_vs_paper"].abs().idxmax(), "method"],
            },
            "auroc_control": {
                "mae": float(d["d_auroc"].abs().mean()),
                "max_abs": float(d["d_auroc"].abs().max()),
                "n_within_0.001": int((d["d_auroc"].abs() <= 0.001).sum()),
            },
        }
    (OUT / "auprc_convention_audit.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("=== methods with a paper row ===")
    print(df[df["has_paper_row"]].round(4).to_string(index=False))
    print("\n=== methods without a paper row (convention still applies) ===")
    cols = ["method", "split", "n_unique_scores", "tie_frac",
            "auprc_sklearn_ap", "auprc_precrec", "auroc_local"]
    nop = df[~df["has_paper_row"]].copy()
    nop["ap_minus_precrec"] = nop["auprc_sklearn_ap"] - nop["auprc_precrec"]
    print(nop[cols + ["ap_minus_precrec"]].round(4).to_string(index=False))
    print(f"\nAP - precrec over the {len(nop)} paperless rows: "
          f"mean {nop['ap_minus_precrec'].mean():+.4f}, "
          f"range {nop['ap_minus_precrec'].min():+.4f}..{nop['ap_minus_precrec'].max():+.4f}, "
          f"n_positive {(nop['ap_minus_precrec'] > 0).sum()}/{len(nop)}")
    print("\n=== which estimator reproduces the paper? (|delta| vs Supplementary Table 4) ===")
    print(json.dumps(summary, indent=2))
    print(f"\n-> {OUT/'auprc_convention_audit.csv'}")
    print(f"-> {OUT/'auprc_convention_audit.json'}")


if __name__ == "__main__":
    main()
