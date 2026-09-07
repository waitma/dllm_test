#!/usr/bin/env python
"""Which BERT-vs-diffusion comparisons are actually resolvable?

The earlier diagnostics plotted single point estimates, so a reader could not
tell a real effect from resampling noise. This figure adds the error bars and
separates two questions that were previously conflated:

Panel A  probe AUROC per arm per readout, with the spread over 20 paired
         stratified splits. Also shows that whitening fitted on the training
         rows only ("honest") matches the transductive version, so the gain is
         not leakage from the probe's own test rows.
Panel B  every arm-vs-arm difference expressed in standard deviations of its
         own paired null. Bars inside the shaded band are not resolvable.
Panel C  the same scale applied to the readout effect within one arm. The
         readout moves each arm by far more than the arms differ from each
         other, which is the reason single-readout arm rankings are fragile.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
SRC = ROOT / "output" / "repr_diagnostics" / "noise_floor.json"
DEST = ROOT / "debug" / "figures" / "noise_floor.png"

BERT_C, DIFF_C = "#c0392b", "#2471a3"


def paired(res: dict, family: str, mode: str) -> tuple[float, float]:
    b = np.asarray(res[family][f"bert|{mode}"])
    d = np.asarray(res[family][f"diff|{mode}"])
    delta = d - b
    return float(delta.mean()), float(delta.std(ddof=1))


def readout_effect(res: dict, family: str, arm: str, hi: str, lo: str = "raw"):
    a = np.asarray(res[family][f"{arm}|{hi}"])
    b = np.asarray(res[family][f"{arm}|{lo}"])
    delta = a - b
    return float(delta.mean()), float(delta.std(ddof=1))


def main() -> int:
    res = json.loads(SRC.read_text())
    fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(19, 5.4))

    # ---- Panel A: probe AUROC with spread ------------------------------
    modes = ["raw", "pcaw-256", "pcaw-256-honest"]
    labels = ["RAW\n(headline)", "whitened\n(transductive)", "whitened\n(train-only fit)"]
    xs = np.arange(len(modes))
    for arm, color, off, name in [
        ("bert", BERT_C, -0.18, "BERT 270m@49000"),
        ("diff", DIFF_C, 0.18, "Diffusion 270m@42000"),
    ]:
        mu = [np.mean(res["probe"][f"{arm}|{m}"]) for m in modes]
        sd = [np.std(res["probe"][f"{arm}|{m}"]) for m in modes]
        axa.bar(xs + off, mu, 0.34, yerr=sd, capsize=5, color=color,
                alpha=0.9, label=name, error_kw=dict(lw=1.4))
        for x, m_, s_ in zip(xs + off, mu, sd):
            axa.text(x, m_ + s_ + 0.006, f"{m_:.3f}", ha="center", fontsize=8.5)

    axa.set_xticks(xs)
    axa.set_xticklabels(labels, fontsize=9)
    axa.set_ylabel("25-way linear probe AUROC")
    axa.set_ylim(0.55, 0.80)
    axa.legend(fontsize=8.5, loc="upper left")
    axa.grid(alpha=0.25, axis="y")
    axa.set_title(
        "A.  Probe AUROC, mean +- sd over 20 paired splits\n"
        "gap is real under RAW, gone under whitening; train-only fit rules out leakage",
        fontsize=10,
    )
    axa.annotate("", xy=(0.18, 0.682), xytext=(-0.18, 0.649),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.5))
    axa.text(0.0, 0.578, "gap +0.033\n= 5.1 sd\nREAL", ha="center", fontsize=8.5,
             fontweight="bold",
             bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=1.5))
    axa.text(1.0, 0.578, "gap -0.004\n= 0.7 sd\nZERO", ha="center", fontsize=8.5,
             fontweight="bold", color="#444",
             bbox=dict(fc="white", ec="0.6", alpha=0.9, pad=1.5))

    # ---- Panel B: arm differences in sd units --------------------------
    arm_rows = [
        ("probe AUROC  RAW", *paired(res, "probe", "raw")),
        ("probe AUROC  whitened", *paired(res, "probe", "pcaw-256")),
        ("probe AUROC  whitened-honest", *paired(res, "probe", "pcaw-256-honest")),
        ("K-sweep ARI  RAW", *paired(res, "ari", "raw")),
        ("K-sweep ARI  whitened", *paired(res, "ari", "pcaw-256")),
        ("kNN@1  RAW", res["knn"]["delta"]["raw"]["point"],
         res["knn"]["delta"]["raw"]["std"]),
        ("kNN@1  whitened", res["knn"]["delta"]["pcaw-256"]["point"],
         res["knn"]["delta"]["pcaw-256"]["std"]),
    ]
    names = [r[0] for r in arm_rows]
    sigmas = [r[1] / r[2] if r[2] else 0.0 for r in arm_rows]
    ys = np.arange(len(names))[::-1]
    colors = ["#2471a3" if s > 0 else "#c0392b" for s in sigmas]
    axb.axvspan(-2, 2, color="0.85", zorder=0)
    axb.barh(ys, sigmas, 0.6, color=colors, alpha=0.9)
    for y, s, row in zip(ys, sigmas, arm_rows):
        axb.text(s + (0.25 if s >= 0 else -0.25), y,
                 f"{row[1]:+.4f}  ({s:+.1f} sd)", va="center",
                 ha="left" if s >= 0 else "right", fontsize=8.5)
    axb.axvline(0, color="k", lw=1)
    axb.set_yticks(ys)
    axb.set_yticklabels(names, fontsize=9)
    axb.set_xlabel("(Diffusion - BERT) in sd of its own paired null\n"
                   "positive = diffusion ahead")
    axb.set_xlim(-6.5, 10)
    axb.grid(alpha=0.25, axis="x")
    axb.set_title(
        "B.  Only 2 of 7 arm comparisons are resolvable\n"
        "grey band = |effect| < 2 sd, i.e. indistinguishable from zero",
        fontsize=10,
    )

    # ---- Panel C: readout effect within one arm ------------------------
    ro_rows = [
        ("probe AUROC  BERT", *readout_effect(res, "probe", "bert", "pcaw-256")),
        ("probe AUROC  Diffusion", *readout_effect(res, "probe", "diff", "pcaw-256")),
        ("K-sweep ARI  BERT", *readout_effect(res, "ari", "bert", "pcaw-256")),
        ("K-sweep ARI  Diffusion", *readout_effect(res, "ari", "diff", "pcaw-256")),
    ]
    names_c = [r[0] for r in ro_rows]
    sig_c = [r[1] / r[2] if r[2] else 0.0 for r in ro_rows]
    ys_c = np.arange(len(names_c))[::-1]
    col_c = [BERT_C if "BERT" in n else DIFF_C for n in names_c]
    axc.axvspan(-2, 2, color="0.85", zorder=0)
    axc.barh(ys_c, sig_c, 0.55, color=col_c, alpha=0.9)
    for y, s, row in zip(ys_c, sig_c, ro_rows):
        axc.text(s + (0.4 if s >= 0 else -0.4), y, f"{row[1]:+.4f}  ({s:+.1f} sd)",
                 va="center", ha="left" if s >= 0 else "right", fontsize=8.5)
    axc.axvline(0, color="k", lw=1)
    axc.set_yticks(ys_c)
    axc.set_yticklabels(names_c, fontsize=9)
    axc.set_xlabel("(whitened - RAW) within the SAME arm, in sd")
    axc.set_xlim(-13, 24)
    axc.grid(alpha=0.25, axis="x")
    axc.set_title(
        "C.  Why each gap closes -- and it is NOT the same reason\n"
        "probe: BERT gains 2x more than diffusion. ARI: diffusion is damaged instead.",
        fontsize=10,
    )

    fig.suptitle(
        "Adding the missing error bars, which the earlier figures lacked. Under the RAW headline readout "
        "diffusion IS genuinely ahead (probe +0.033 at 5.1 sd, ARI +0.009 at 7.6 sd): the original "
        "observation was real, and the\nclaim that BERT overtakes it was noise. Whitening erases both "
        "gaps rather than reversing them, but for different reasons (panel C): on the linear probe BERT "
        "gains +0.082 vs diffusion's +0.045, so BERT's\nfeature was information-rich yet geometrically "
        "unusable -- whereas on ARI whitening simply damages diffusion (-0.005), so diffusion's clustering "
        "advantage is NOT explained away. kNN@1 resolved nothing.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    DEST.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DEST, dpi=170)
    print(f"-> {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
