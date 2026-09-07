#!/usr/bin/env python
"""Plot the layer sweep from ``diag_repr_layers.py`` for both training arms."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DIAG = Path("/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/repr_diagnostics")
ARMS = {
    "BERT 270m@49000": ("bert_270m_49000", "#c0392b"),
    "Diffusion 270m@42000": ("diff_270m_42000", "#2471a3"),
}
OFFICIAL = {"BERT 270m@49000": 0.0186, "Diffusion 270m@42000": 0.0277}


def load(tag: str) -> pd.DataFrame:
    df = pd.read_csv(DIAG / tag / "layer_sweep.csv")
    df["layer"] = df["source"].str.replace("layer", "", regex=False)
    return df


def main():
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    ax_probe, ax_ari, ax_cos, ax_rank = axes

    for label, (tag, color) in ARMS.items():
        df = load(tag)
        lay = df[df["source"].str.startswith("layer")].copy()
        lay["li"] = lay["layer"].astype(int)
        esmc = df[df["source"] == "esmc_encoder"]

        raw = lay[lay["post"] == "raw"].sort_values("li")
        zs = lay[lay["post"] == "zscore"].sort_values("li")
        ctr = lay[lay["post"] == "center"].sort_values("li")

        ax_probe.plot(raw["li"], raw["probe_auroc"], "-o", color=color, ms=4,
                      label=f"{label} · raw")
        ax_probe.plot(zs["li"], zs["probe_auroc"], "--s", color=color, ms=4,
                      alpha=0.65, label=f"{label} · z-scored")
        ax_probe.axhline(float(esmc[esmc["post"] == "zscore"]["probe_auroc"].iloc[0]),
                         color=color, ls=":", lw=1.2, alpha=0.8)

        ax_ari.plot(raw["li"], raw["ari_mean"], "-o", color=color, ms=4,
                    label=f"{label} · raw")
        ax_ari.plot(ctr["li"], ctr["ari_mean"], "--s", color=color, ms=4,
                    alpha=0.65, label=f"{label} · centered")

        geo = raw
        ax_cos.plot(geo["li"], geo["mean_cos"], "-o", color=color, ms=4, label=label)
        ax_rank.plot(geo["li"], geo["eff_rank"], "-o", color=color, ms=4, label=label)

    n_layers = 8
    for ax in axes:
        ax.axvspan(n_layers - 0.35, n_layers + 0.35, color="#f39c12", alpha=0.18, zorder=0)
        ax.set_xlabel("decoder hidden-state index  (8 = headline readout)")
        ax.grid(alpha=0.25)

    ax_probe.set_title("25-way linear probe AUROC\n(dotted = ESMC encoder alone, z-scored)")
    ax_probe.set_ylabel("probe AUROC")
    ax_probe.legend(fontsize=7, loc="lower left")

    ax_ari.axhline(0.033, color="green", ls="-.", lw=1.2)
    ax_ari.text(0.15, 0.0335, "SCEPTR 0.033", fontsize=7, color="green")
    ax_ari.set_title("T2 basis-B K-means ARI (mean over K sweep)")
    ax_ari.set_ylabel("ARI")
    ax_ari.legend(fontsize=7, loc="upper left")

    ax_cos.set_title("Anisotropy: mean pairwise cosine\n(1.0 = all embeddings identical)")
    ax_cos.set_ylabel("mean pairwise cosine")
    ax_cos.legend(fontsize=8)

    ax_rank.set_title("Effective rank of the embedding set\n(entropy of PCA spectrum)")
    ax_rank.set_ylabel("effective rank")
    ax_rank.legend(fontsize=8)

    fig.suptitle(
        "Mechanism: the headline layer (index 8, post-`ln_f`) is the most anisotropic and least "
        "linearly-decodable layer in both arms.\nPanels 3-4 are the cause, panels 1-2 the effect. "
        "Note BERT's cosine/rank degrade further than diffusion's — that asymmetry is why the raw "
        "readout distorts the\nBERT-vs-diffusion comparison. These are proxy-harness probes on a "
        "frozen backbone; the dotted ESMC lines are raw-feature readings and do NOT imply the "
        "decoder is useless.",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = Path(
        "/vepfs-mlp2/c20250601/251105016/project/dllm_test/debug/figures/"
        "repr_layer_diagnosis.png"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
