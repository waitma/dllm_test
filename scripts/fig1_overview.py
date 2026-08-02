"""Figure 1 - concept / motivation.

Tells the reader, at a glance: what our model is, which biological inputs it
accepts, and which downstream tasks it addresses.

Inputs (immune + interaction families)  ->  one grammar-conditioned generative
protein language model  ->  two capability streams (representation & conditional
generation)  ->  the concrete downstream tasks.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts")
import figstyle as S  # noqa: E402


def glyph(ax, x, y, rows):
    """Small multi-chain molecule glyph made of bead rows."""
    for ri, (n, start) in enumerate(rows):
        S.bead_chain(ax, x, y - ri * 2.0, n, r=0.55, dx=1.35, start=start)


def main():
    S.use_style()
    fig, ax = S.new_ax(7.6, 3.9)   # ylim ~ 51.3

    YT = 47.5
    ax.text(3.0, YT, "A unified generative model for immune receptors and protein interactions",
            fontsize=8.6, fontweight="bold", color=S.INK, ha="left")

    # ---------------- inputs (left) ----------------
    xin = 4.0
    ax.text(xin, 41.5, "INPUTS", fontsize=6.6, fontweight="bold", color=S.SUB, ha="left")
    inputs = [
        ("TCR",           [(5, 0), (5, 3)]),
        ("Antibody",      [(6, 1), (4, 4)]),
        ("Nanobody",      [(6, 2)]),
        ("Peptide\u2013MHC", [(3, 5), (7, 0)]),
        ("Protein\u2013protein", [(6, 3), (6, 0)]),
    ]
    ys = [37.0, 30.5, 24.5, 18.5, 11.5]
    for (name, rows), yy in zip(inputs, ys):
        glyph(ax, xin + 1.0, yy, rows)
        ax.text(xin + 12.5, yy - (len(rows) - 1) * 1.0, name, fontsize=7.0,
                color=S.INK, ha="left", va="center")

    # converging guide lines into the model (single funnel point)
    fx, fy = 31.5, 24.5
    for (name, rows), yy in zip(inputs, ys):
        y_anchor = yy - (len(rows) - 1) * 1.0
        ax.plot([xin + 22.0, fx], [y_anchor, fy], color=S.HAIR, lw=0.9, zorder=1,
                solid_capstyle="round")
    S.arrow(ax, (fx, fy), (33.4, fy), c=S.SUB, lw=1.3)

    # ---------------- model (center) ----------------
    mx, my, mw, mh = 33.8, 14.0, 30.0, 23.0
    S.rbox(ax, mx, my, mw, mh, fc=S.tint(S.TEAL, 0.90), ec=S.TEAL, lw=1.6, r=1.8,
           z=5, shadow=True)
    cxm = mx + mw / 2
    ax.text(cxm, my + mh - 3.6, "Grammar-conditioned", fontsize=8.2,
            fontweight="bold", color=S.TEAL, ha="center", zorder=9)
    ax.text(cxm, my + mh - 6.3, "protein language model", fontsize=8.2,
            fontweight="bold", color=S.TEAL, ha="center", zorder=9)

    strip = [S.NAVY, S.TEAL, S.PLUM, S.PLUM, S.CORAL, S.NAVY, S.TEAL, S.PLUM, S.NAVY]
    bw = 2.3
    sx = cxm - len(strip) * bw / 2
    for i, c in enumerate(strip):
        S.rbox(ax, sx + i * bw, my + mh - 11.4, bw - 0.35, 2.2, fc=c, ec="white",
               lw=0.5, r=0.3, z=8)
    ax.text(cxm, my + 4.4, "ESMC per-chain encoder", fontsize=6.4, color=S.SUB,
            ha="center", zorder=9)
    ax.text(cxm, my + 2.1, "bidirectional LLaDA diffusion", fontsize=6.4, color=S.SUB,
            ha="center", zorder=9)

    # ---------------- capability arrows ----------------
    xtask = 71.0
    S.arrow(ax, (mx + mw, my + mh - 5.0), (xtask - 1.0, 38.0), c=S.BLUE, lw=1.6)
    ax.text((mx + mw + xtask) / 2, 41.2, "represent", fontsize=6.2, color=S.BLUE,
            ha="center", style="italic", zorder=9)
    S.arrow(ax, (mx + mw, my + 5.0), (xtask - 1.0, 14.0), c=S.CORAL, lw=1.6)
    ax.text((mx + mw + xtask) / 2, 8.8, "generate", fontsize=6.2, color=S.CORAL,
            ha="center", style="italic", zorder=9)

    # ---------------- tasks (right) ----------------
    def task_card(x, y, w, h, color, title, items):
        S.rbox(ax, x, y, w, h, fc=S.tint(color, 0.93), ec=color, lw=1.3, r=1.4, z=4)
        ax.text(x + 2.0, y + h - 3.0, title, fontsize=7.2, fontweight="bold",
                color=color, ha="left", zorder=9)
        for i, it in enumerate(items):
            ax.text(x + 2.8, y + h - 5.8 - i * 2.5, "\u2022  " + it, fontsize=6.4,
                    color=S.INK, ha="left", zorder=9)

    task_card(xtask, 29.5, 27.0, 17.0, S.BLUE, "Prediction & representation",
              ["binding / interaction", "affinity & mutation effect",
               "receptor clustering", "few-shot transfer"])
    task_card(xtask, 5.0, 27.0, 17.0, S.CORAL, "Conditional generation",
              ["CDR sequence design", "heavy\u2013light chain pairing",
               "grammar-constrained sampling"])

    S.save(fig, "fig1_overview")


if __name__ == "__main__":
    main()
