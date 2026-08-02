"""Shared figure style for the paper (Nature-like).

One place for font + palette so every figure is visually consistent.
Font: Nimbus Sans (open Helvetica/Arial equivalent, present on this box).
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.colors import to_rgb

FIGDIR = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/figures"


def use_style():
    plt.rcParams.update({
        "font.family": "Nimbus Sans",
        "font.size": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "axes.linewidth": 0.6,
    })


# ---- one coherent palette shared by all figures ----------------------
INK, SUB, HAIR = "#1E2A33", "#5B6B77", "#D3DBE1"
NAVY = "#2C5578"
BLUE = "#4E86AE"
TEAL = "#2E9B94"
GREEN = "#5AA65B"
GOLD = "#D9A63C"
CORAL = "#E1795A"
PLUM = "#7C6BA8"

# grammar token classes (used in fig2 and the mini-strip in fig1)
KIND = {
    "struct": NAVY,
    "type":   TEAL,
    "rel":    CORAL,
    "res":    PLUM,
}
# amino-acid bead colours cycled by position
AA = [BLUE, TEAL, GOLD, CORAL, PLUM, GREEN]
MASK_FC, MASK_EC = "#EEF1F3", "#B7C1C9"

SHADOW = [pe.withSimplePatchShadow(offset=(0.9, -0.9), shadow_rgbFace="#93A0A9", alpha=0.16)]


def tint(color, frac=0.86):
    """Lighten a colour toward white by `frac` (0=color, 1=white)."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * frac, g + (1 - g) * frac, b + (1 - b) * frac)


def rbox(ax, x, y, w, h, fc="#fff", ec=HAIR, lw=1.0, r=1.0, z=2, shadow=False, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z, linestyle=ls)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)
    return p


def arrow(ax, p0, p1, c=SUB, lw=1.3, mut=8, z=3):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=mut,
                 linewidth=lw, color=c, shrinkA=0, shrinkB=0, zorder=z))


def bead(ax, cx, cy, r, filled=True, ci=0, cond=False, z=6):
    if filled:
        ax.add_patch(Circle((cx, cy), r, facecolor=AA[ci % len(AA)],
                            edgecolor="white", lw=0.5, zorder=z))
    else:
        ax.add_patch(Circle((cx, cy), r, facecolor=MASK_FC, edgecolor="#B7C1C9",
                            lw=0.7, linestyle=(0, (2, 1.3)), zorder=z))
    if cond:
        ax.add_patch(Circle((cx, cy), r + 0.30, facecolor="none",
                            edgecolor=GOLD, lw=1.1, zorder=z + 1))


def bead_chain(ax, x0, y, n, r=0.6, dx=1.5, start=0, z=6):
    for i in range(n):
        bead(ax, x0 + i * dx, y, r, filled=True, ci=start + i, z=z)


def new_ax(w_in, h_in):
    """Full-bleed axes with equal aspect; data coords 0..100 in x."""
    fig = plt.figure(figsize=(w_in, h_in), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def panel_letter(ax, x, y, letter, title):
    ax.text(x, y, letter, fontsize=11, fontweight="bold", color=INK)
    ax.text(x + 2.6, y, title, fontsize=9.2, fontweight="bold", color=INK, va="baseline")


def save(fig, name):
    fig.savefig(f"{FIGDIR}/{name}.pdf")
    fig.savefig(f"{FIGDIR}/{name}.png", dpi=400)
    plt.close(fig)
    print("wrote", name)
