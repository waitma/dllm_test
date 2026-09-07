#!/usr/bin/env python
"""Headline figure for the BERT-representation debug: what the raw readout costs.

IMPORTANT: re-scaling is a **diagnostic probe here, not an adopted method**. The
headline readout is and stays raw (see RESULTS.md 0.0 defect (f) and
debug/BERT_REPR_DEBUG.md 6). The probe exists to quantify two things:

Panel A  official T3 deep few-shot NN AUROC vs k. The distance between the raw
         curves and the whitened curves is the *geometry loss* -- how much of
         our reported number is left on the table by the final layer's
         anisotropy. Published baselines are drawn for scale only; we do NOT
         claim to pass them, since they are themselves un-rescaled.
Panel B  official T2 basis-B K-means ARI for the same runs. T2 prefers the
         *opposite* rescaling to T3, which is why no single re-scaling is a
         viable fix -- one of the reasons it was not adopted.

Reads only official benchmark outputs (``run_paper6.py`` / ``run_embed_bench.py``),
no proxy harness numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test")
P6 = ROOT / "downstream/benchmark/outputs/tcr_representation_paper6"
T2 = ROOT / "downstream/benchmark/outputs/tcr_clustering_embed"
OUT = ROOT / "debug/figures/readout_effect_official.png"

# (label, t3 tag, t2 tag, colour, linestyle, marker)
RUNS = [
    ("BERT — RAW (headline)", "ours_fusion_270m_bert_49000", "diag_bert_none",
     "#c0392b", "-", "o"),
    ("BERT · centered (probe)", "diag_t3_bert_center", "diag_bert_center",
     "#c0392b", "--", "s"),
    ("BERT · whitened (probe)", "diag_t3_bert_pcaw256", "diag_bert_pcaw256",
     "#c0392b", ":", "D"),
    ("Diffusion — RAW (headline)", "ours_fusion_270m_diff_42000", "diag_diff_none",
     "#2471a3", "-", "o"),
    ("Diffusion · centered (probe)", "diag_t3_diff_center", "diag_diff_center",
     "#2471a3", "--", "s"),
    ("Diffusion · whitened (probe)", "diag_t3_diff_pcaw256", "diag_diff_pcaw256",
     "#2471a3", ":", "D"),
]

# published reference points, RESULTS.md 0.3 deep track (k=200 macro AUROC)
BASELINES = [
    ("SCEPTR 0.790", 0.790, "#16a085"),
    ("TCRdist 0.777", 0.777, "#8e44ad"),
    ("CDR3-Levenshtein 0.733", 0.733, "#7f8c8d"),
    ("k-mer(3) 0.728", 0.728, "#95a5a6"),
]


def t3_curve(tag: str) -> tuple[list[int], list[float]]:
    d = json.loads((P6 / tag / "fewshot.json").read_text())
    by_shot = d["auroc_by_shot"]
    ks = sorted(int(k) for k in by_shot)
    return ks, [by_shot[str(k)]["mean"] for k in ks]


def t2_ari(tag: str) -> float:
    return json.loads((T2 / tag / "metrics.json").read_text())["kmeans_ari_mean"]


def main() -> None:
    fig, (ax3, ax2) = plt.subplots(
        1, 2, figsize=(15.5, 5.6), gridspec_kw={"width_ratios": [1.55, 1.0]}
    )

    # ---- Panel A: T3 deep, AUROC vs k -----------------------------------
    for label, t3tag, _, color, ls, mk in RUNS:
        ks, ys = t3_curve(t3tag)
        ax3.plot(ks, ys, ls=ls, marker=mk, color=color, ms=5,
                 lw=2.8 if ls == "-" else 1.8,
                 alpha=1.0 if ls == "-" else 0.65, label=label)

    for name, val, color in BASELINES:
        ax3.axhline(val, color=color, lw=1.1, ls=(0, (6, 4)), alpha=0.7, zorder=0)
        ax3.text(1.6, val, f"{name}  (also un-rescaled)", fontsize=7.5, color=color,
                 va="center", bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.9))

    ax3.set_xscale("log")
    ax3.set_xlim(0.9, 340)
    ax3.set_xticks([1, 2, 5, 10, 20, 50, 100, 200])
    ax3.set_xticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
    ax3.set_xlabel("k  (labelled binders per pMHC, few-shot reference pool)")
    ax3.set_ylabel("macro NN AUROC over 6 pMHC")
    ax3.set_title(
        "A.  Official T3 deep few-shot (run_paper6.py, 100 seeds)\n"
        "solid = our headline (RAW). dashed/dotted = probes only, NOT adopted.",
        fontsize=10.5,
    )
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=8, loc="lower right", framealpha=0.95)

    # annotate the gain at k=200
    for tag_none, tag_pcaw, color, xa, xt in [
        ("ours_fusion_270m_bert_49000", "diag_t3_bert_pcaw256", "#c0392b", 200, 218),
        ("ours_fusion_270m_diff_42000", "diag_t3_diff_pcaw256", "#2471a3", 165, 108),
    ]:
        lo = t3_curve(tag_none)[1][-1]
        hi = t3_curve(tag_pcaw)[1][-1]
        ax3.annotate(
            "", xy=(xa, hi), xytext=(xa, lo),
            arrowprops=dict(arrowstyle="<->", color=color, lw=1.6),
        )
        ax3.text(xt, (lo + hi) / 2, f"{hi - lo:.3f}\ngeometry loss",
                 fontsize=8.5, color=color, fontweight="bold", va="center",
                 ha="center" if xt < xa else "left",
                 bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.8))

    # ---- Panel B: T2 ARI bars -------------------------------------------
    labels = ["BERT\nRAW", "BERT\ncentered", "BERT\nwhitened",
              "Diff\nRAW", "Diff\ncentered", "Diff\nwhitened"]
    vals = [t2_ari(r[2]) for r in RUNS]
    colors = [r[3] for r in RUNS]
    alphas = [1.0, 0.5, 0.5, 1.0, 0.5, 0.5]
    xs = np.arange(len(vals))
    for x, v, c, a in zip(xs, vals, colors, alphas):
        ax2.bar(x, v, color=c, alpha=a, edgecolor="white")
        ax2.text(x, v + 0.0006, f"{v:.4f}", ha="center", fontsize=8.5)

    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("K-means ARI (mean over 19-point K sweep)")
    ax2.set_ylim(0, 0.045)
    ax2.set_title(
        "B.  Official T2 basis-B clustering (run_embed_bench.py)\n"
        "T2 prefers centering, T3 prefers whitening — opposite, so no single fix",
        fontsize=10.5,
    )
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Cost of the RAW readout, which we keep. The final decoder layer is severely anisotropic\n"
        "(mean pairwise cosine 0.99, effective rank 17-20 of 768): re-scaling the SAME feature "
        "recovers ~0.047 T3 AUROC for both arms,\nlarger than any model-vs-model difference in the "
        "table (previous max 0.011). Re-scaling is a PROBE, NOT adopted — T2/T3 want opposite\n"
        "rescalings, and the reference baselines are un-rescaled too, so no baseline is 'passed'.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
