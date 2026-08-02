"""Figure 1 subpanels - publication style, one shared palette, separate files.

Produces FOUR independent subfigure files (not a single combined image):
  figures/fig1a_overview.pdf      - what the model does + downstream tasks
  figures/fig1b_grammar.pdf       - designed grammar token format
  figures/fig1c_architecture.pdf  - ESMC encoder + LLaDA denoiser pipeline
  figures/fig1d_generation.pdf    - iterative unmasking (conditional diffusion)

Typography follows Nature-style conventions: sans-serif, small regular-weight
labels, bold lowercase panel letters top-left, thin rules, no heavy centered
titles. One consistent, colour-blind-safe palette is shared across all panels.

Run: python scripts/gen_fig1_mpl.py
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

FIGDIR = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/figures"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# ---- ONE shared palette (semantic roles, reused everywhere) -----------
INK, SUB, HAIR = "#1F2A33", "#5C6B77", "#D5DCE2"
NAVY = "#2C5578"   # structure / model core / TCR family
TEAL = "#2E9B94"   # ESMC encoder / type / antibody family
PLUM = "#7C6BA8"   # LLaDA denoiser / PPI family
CORAL = "#E1795A"  # relation / generated / nanobody family
GOLD = "#E0B23C"   # conditioning / fixed context
FIXED_BG, GEN_BG = "#EAF1F6", "#FBEBDA"

# residue (amino-acid) beads cycle through the SAME palette hues
AA = [NAVY, TEAL, PLUM, CORAL, GOLD]
MASK_FC, MASK_EC = "#EEF1F3", "#B7C1C9"  # light grey (masked positions)

SHADOW = [pe.withSimplePatchShadow(offset=(0.9, -0.9), shadow_rgbFace="#93A0A9", alpha=0.16)]


def new_ax(w_in, h_in):
    fig = plt.figure(figsize=(w_in, h_in), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax, 100 * h_in / w_in


def rbox(ax, x, y, w, h, fc="#fff", ec=HAIR, lw=1.0, r=1.0, z=2, shadow=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.02,rounding_size={r}",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z)
    if shadow:
        p.set_path_effects(SHADOW)
    ax.add_patch(p)


def arrow(ax, p0, p1, c=SUB, lw=1.3, ms=8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                 linewidth=lw, color=c, shrinkA=0, shrinkB=0, zorder=3))


def panel_letter(ax, ymax, letter):
    ax.text(1.5, ymax - 2.2, letter, fontsize=12.5, fontweight="bold",
            color=INK, va="top", ha="left")


def bead(ax, cx, cy, r, filled=True, ci=0, cond=False, z=6):
    if filled:
        ax.add_patch(Circle((cx, cy), r, facecolor=AA[ci % len(AA)],
                            edgecolor="white", lw=0.5, zorder=z))
    else:
        ax.add_patch(Circle((cx, cy), r, facecolor=MASK_FC, edgecolor=MASK_EC,
                            lw=0.7, linestyle=(0, (2, 1.3)), zorder=z))
    if cond:
        ax.add_patch(Circle((cx, cy), r + 0.30, facecolor="none",
                            edgecolor=GOLD, lw=1.1, zorder=z + 1))


def bead_chain(ax, x0, y, n, r=0.6, dx=1.5, start=0, z=6):
    for i in range(n):
        bead(ax, x0 + i * dx, y, r, filled=True, ci=start + i, z=z)


# =======================================================================
# (a) OVERVIEW: what the model does + which tasks
# =======================================================================
def fig1a_overview():
    fig, ax, ymax = new_ax(7.6, 3.0)
    panel_letter(ax, ymax, "a")

    cy = ymax / 2 - 1.0
    hy = ymax - 3.2  # header row

    # --- inputs (immune-protein complexes) ---
    ax.text(5.0, hy, "Immune-protein complexes", fontsize=7.0,
            color=INK, ha="left", va="center", zorder=9)
    inputs = [
        ("TCR \u2013 peptide", [(3, GOLD), (7, NAVY)]),
        ("antibody \u2013 antigen", [(6, TEAL), (5, HAIR)]),
        ("protein \u2013 protein", [(6, PLUM), (6, PLUM)]),
    ]
    iy = [cy + 6.5, cy, cy - 6.5]
    for (lab, segs), yy in zip(inputs, iy):
        ax.text(5.0, yy, lab, fontsize=6.0, color=SUB, ha="left", va="center", zorder=9)
        bx = 18.0
        for (n, col) in segs:
            for i in range(n):
                ax.add_patch(Circle((bx + i * 1.1, yy), 0.48, facecolor=col,
                                    edgecolor="white", lw=0.4, zorder=6))
            bx += n * 1.1 + 1.0

    arrow(ax, (31.5, cy), (36.0, cy), c=SUB, lw=1.5, ms=10)

    # --- unified model (core) ---
    mx, mw = 37.0, 21.0
    rbox(ax, mx, cy - 8.5, mw, 17.0, fc="#FFFFFF", ec=NAVY, lw=1.6, r=1.4, z=4, shadow=True)
    ax.text(mx + mw / 2, cy + 4.6, "Unified generative", ha="center", va="center",
            fontsize=7.8, color=NAVY, fontweight="bold", zorder=9)
    ax.text(mx + mw / 2, cy + 1.9, "sequence model", ha="center", va="center",
            fontsize=7.8, color=NAVY, fontweight="bold", zorder=9)
    ax.plot([mx + 3, mx + mw - 3], [cy - 0.4, cy - 0.4], color=HAIR, lw=0.8, zorder=9)
    ax.text(mx + mw / 2, cy - 2.3, "grammar-token sequences", ha="center", va="center",
            fontsize=5.9, color=SUB, zorder=9)
    ax.text(mx + mw / 2, cy - 4.3, "ESMC encoder + LLaDA denoiser", ha="center",
            va="center", fontsize=5.9, color=SUB, zorder=9)
    ax.text(mx + mw / 2, cy - 6.6, "one frozen checkpoint", ha="center", va="center",
            fontsize=5.9, color=CORAL, style="italic", zorder=9)

    arrow(ax, (mx + mw + 0.5, cy), (mx + mw + 5.0, cy), c=SUB, lw=1.5, ms=10)

    # --- downstream tasks ---
    tx = mx + mw + 6.0
    ax.text(tx, hy, "Downstream tasks", fontsize=7.0, color=INK,
            ha="left", va="center", zorder=9)
    tasks = [
        ("TCR\u2013peptide binding\n& specificity", NAVY),
        ("Antibody CDR design\n& chain pairing", TEAL),
        ("Protein-interaction\naffinity (PPI)", PLUM),
        ("Nanobody property\nprediction", CORAL),
    ]
    cw, ch, gx, gy = 16.5, 6.8, 1.3, 1.2
    x00 = tx
    y00 = cy + ch + gy / 2
    for i, (lab, col) in enumerate(tasks):
        r, c = divmod(i, 2)
        xx = x00 + c * (cw + gx)
        yy = y00 - r * (ch + gy)
        rbox(ax, xx, yy - ch, cw, ch, fc="#FFFFFF", ec=col, lw=1.3, r=0.9, z=4)
        ax.add_patch(Circle((xx + 2.2, yy - ch / 2), 0.85, facecolor=col,
                            edgecolor="none", zorder=6))
        ax.text(xx + 4.0, yy - ch / 2, lab, ha="left", va="center",
                fontsize=5.9, color=INK, linespacing=1.2, zorder=9)

    fig.savefig(f"{FIGDIR}/fig1a_overview.pdf")
    fig.savefig(f"{FIGDIR}/fig1a_overview.png", dpi=400)
    plt.close(fig)


# =======================================================================
# (b) GRAMMAR TOKEN FORMAT
# =======================================================================
RB, DX, DOT = 0.62, 1.55, 1.9


def _capsule(ax, x, y, h, tlabel, groups, state):
    tabw = 0.85 * len(tlabel) + 2.6
    bw = 0.0
    for gi, (n, _) in enumerate(groups):
        bw += (n - 1) * DX + 2 * RB + (DOT if gi < len(groups) - 1 else 0)
    pad = 1.4
    w = tabw + 1.2 + bw + pad * 2
    rbox(ax, x, y, w, h, fc=(FIXED_BG if state == "cond" else GEN_BG),
         ec=NAVY, lw=1.2, r=1.2, z=4, shadow=True)
    ec, fc = (TEAL, "#E3F3F1")
    rbox(ax, x + 1.2, y + h / 2 - 1.6, tabw, 3.2, fc=fc, ec=ec, lw=1.0, r=0.8, z=6)
    ax.text(x + 1.2 + tabw / 2, y + h / 2, tlabel, ha="center", va="center",
            fontsize=5.9, color=INK, zorder=7)
    bx = x + 1.2 + tabw + pad + RB
    for gi, (n, st) in enumerate(groups):
        bead_chain(ax, bx, y + h / 2, n, r=RB, dx=DX, start=st)
        bx += (n - 1) * DX
        if gi < len(groups) - 1:
            ax.text(bx + DOT / 2, y + h / 2 - 0.2, "\u00b7", ha="center", va="center",
                    fontsize=10, color=SUB, zorder=7)
            bx += DOT + RB
        else:
            bx += RB
    return x + w


def _relation(ax, x, ymid, label):
    w = 0.85 * len(label) + 2.6
    ax.plot([x, x + 1.3], [ymid, ymid], color=CORAL, lw=1.2, zorder=3)
    rbox(ax, x + 1.3, ymid - 1.6, w, 3.2, fc="#FBECE4", ec=CORAL, lw=1.0, r=1.6, z=5)
    ax.text(x + 1.3 + w / 2, ymid, label, ha="center", va="center",
            fontsize=5.8, color=CORAL, zorder=6)
    ax.plot([x + 1.3 + w, x + 2.6 + w], [ymid, ymid], color=CORAL, lw=1.2, zorder=3)
    return x + 2.6 + w


def fig1b_grammar():
    fig, ax, ymax = new_ax(7.6, 3.1)
    panel_letter(ax, ymax, "b")
    x0 = 3.0

    # legend (own row, clear of the panel letter)
    ly = ymax - 3.0
    lx = 7.0
    rbox(ax, lx, ly - 1.3, 3.4, 2.6, fc="#fff", ec=NAVY, lw=1.2, r=0.8, z=4)
    ax.text(lx + 4.4, ly, "structure delimiter", fontsize=6.0, color=SUB, va="center")
    rbox(ax, lx + 27, ly - 1.3, 3.4, 2.6, fc="#E3F3F1", ec=TEAL, lw=1.0, r=0.7, z=4)
    ax.text(lx + 31.4, ly, "type tab", fontsize=6.0, color=SUB, va="center")
    rbox(ax, lx + 43, ly - 1.3, 3.4, 2.6, fc="#FBECE4", ec=CORAL, lw=1.0, r=1.4, z=4)
    ax.text(lx + 47.4, ly, "relation", fontsize=6.0, color=SUB, va="center")
    for i in range(4):
        bead(ax, lx + 62 + i * 1.5, ly, RB, filled=True, ci=i)
    ax.text(lx + 62 + 4 * 1.5 + 0.6, ly, "residues", fontsize=6.0, color=SUB, va="center")
    ax.plot([x0, 97], [ymax - 5.4, ymax - 5.4], color=HAIR, lw=0.7, zorder=1)

    def record(title, ybot, items):
        ax.text(x0, ybot + 5.6, title, ha="left", va="center", fontsize=7.0,
                fontweight="bold", color=INK)
        x = x0
        for it in items:
            if it[0] == "cap":
                _, tl, groups, state = it
                xe = _capsule(ax, x, ybot, 5.0, tl, groups, state)
                ax.text((x + xe) / 2, ybot - 1.7, "conditioning" if state == "cond" else "generated",
                        ha="center", va="center", fontsize=5.8,
                        color=(NAVY if state == "cond" else CORAL))
                x = xe + 1.6
            else:
                x = _relation(ax, x, ybot + 2.5, it[1]) + 1.6

    top = ymax - 13.5
    step = 9.4
    record("OTS (TCR)  \u00b7  unconditional", top, [
        ("cap", "<tcr>", [(6, 0), (6, 6)], "gen")])
    record("TCR + peptide  \u00b7  conditional", top - step, [
        ("cap", "<pep>", [(5, 0)], "cond"),
        ("rel", "binding"),
        ("cap", "<tcr>", [(5, 5), (5, 10)], "gen")])
    record("Protein interaction  \u00b7  conditional", top - 2 * step, [
        ("cap", "<ab>", [(8, 0)], "cond"),
        ("rel", "binding"),
        ("cap", "<ag>", [(7, 8)], "gen")])

    fig.savefig(f"{FIGDIR}/fig1b_grammar.pdf")
    fig.savefig(f"{FIGDIR}/fig1b_grammar.png", dpi=400)
    plt.close(fig)


# =======================================================================
# (c) MODEL ARCHITECTURE
# =======================================================================
def _module(ax, cx, cy, w, h, color, fill, name):
    for i in range(2, 0, -1):
        rbox(ax, cx - w / 2 + i * 0.8, cy - h / 2 + i * 0.8, w, h,
             fc=fill, ec=color, lw=0.9, r=1.0, z=3 + (3 - i))
    rbox(ax, cx - w / 2, cy - h / 2, w, h, fc=fill, ec=color, lw=1.4, r=1.0, z=7, shadow=True)
    ax.text(cx, cy, name, ha="center", va="center", fontsize=7.4, color=color, zorder=8)


def fig1c_architecture():
    fig, ax, ymax = new_ax(7.6, 2.2)
    panel_letter(ax, ymax, "c")
    cy = ymax / 2 - 0.5

    ax.text(9.0, ymax - 2.4, "protein chains", fontsize=6.2, color=SUB, ha="center")
    for yy, n, st in [(cy + 3.0, 8, 0), (cy, 6, 8), (cy - 3.0, 9, 14)]:
        bead_chain(ax, 4.0, yy, n, r=0.5, dx=1.15, start=st)
    arrow(ax, (18.0, cy), (21.5, cy))

    _module(ax, 30.0, cy, 15.5, 10.5, TEAL, "#E3F3F1", "ESMC encoder")
    ax.text(30.0, cy - 8.0, "per-chain", fontsize=5.8, color=SUB, ha="center")
    arrow(ax, (38.5, cy), (42.0, cy))

    ax.text(52.0, ymax - 2.4, "token stream", fontsize=6.2, color=SUB, ha="center")
    segc = [NAVY, TEAL, PLUM, PLUM, CORAL, NAVY, TEAL, PLUM, PLUM, NAVY]
    bw = 1.85
    sx = 52.0 - len(segc) * bw / 2
    for i, c in enumerate(segc):
        rbox(ax, sx + i * bw, cy - 1.2, bw - 0.28, 2.4, fc=c, ec="white", lw=0.6, r=0.3, z=6)
    arrow(ax, (62.5, cy), (66.0, cy))

    _module(ax, 76.0, cy, 16.0, 10.5, PLUM, "#EEEAF6", "LLaDA denoiser")
    ax.text(76.0, cy - 8.0, "bidirectional", fontsize=5.8, color=SUB, ha="center")
    arrow(ax, (84.5, cy), (88.0, cy))

    ax.text(94.5, ymax - 2.4, "sequence", fontsize=6.2, color=SUB, ha="center")
    bead_chain(ax, 90.0, cy, 7, r=0.5, dx=1.25, start=2)

    fig.savefig(f"{FIGDIR}/fig1c_architecture.pdf")
    fig.savefig(f"{FIGDIR}/fig1c_architecture.png", dpi=400)
    plt.close(fig)


# =======================================================================
# (d) ITERATIVE UNMASKING
# =======================================================================
def fig1d_generation():
    fig, ax, ymax = new_ax(7.6, 2.1)
    panel_letter(ax, ymax, "d")

    ax.text(50.0, ymax - 2.6, "denoising steps  \u2192  unmask non-fixed positions",
            ha="center", fontsize=6.4, color=SUB)

    n = 11
    cond = {0, 1, 2}
    gen = [i for i in range(n) if i not in cond]
    schedule = [0, 2, 5, len(gen)]
    labels = ["t = T", "", "", "t = 0"]
    subs = ["all masked", "denoising", "denoising", "generated"]
    r, dx = 0.68, 1.55
    chain_w = (n - 1) * dx + 2 * r
    xs = [4.0, 29.0, 54.0, 79.0]
    y = ymax / 2 - 0.5

    for si, (xstart, k, lab, sub) in enumerate(zip(xs, schedule, labels, subs)):
        revealed = set(gen[:k])
        for i in range(n):
            cxp = xstart + r + i * dx
            if i in cond:
                bead(ax, cxp, y, r, filled=True, ci=i, cond=True)
            elif i in revealed:
                bead(ax, cxp, y, r, filled=True, ci=i)
            else:
                bead(ax, cxp, y, r, filled=False)
        if lab:
            ax.text(xstart + chain_w / 2, y + 2.6, lab, ha="center", fontsize=6.6,
                    fontweight="bold", color=INK)
        ax.text(xstart + chain_w / 2, y - 2.8, sub, ha="center", fontsize=6.0, color=SUB)
        if si < len(xs) - 1:
            arrow(ax, (xstart + chain_w + 1.2, y), (xs[si + 1] - 1.2, y))

    ly = 3.0
    bead(ax, 4.5, ly, 0.68, filled=True, ci=3, cond=True)
    ax.text(6.0, ly, "conditioning (fixed)", fontsize=5.8, color=SUB, va="center")
    bead(ax, 40.0, ly, 0.68, filled=False)
    ax.text(41.5, ly, "masked", fontsize=5.8, color=SUB, va="center")
    bead(ax, 62.0, ly, 0.68, filled=True, ci=1)
    ax.text(63.5, ly, "generated residue", fontsize=5.8, color=SUB, va="center")

    fig.savefig(f"{FIGDIR}/fig1d_generation.pdf")
    fig.savefig(f"{FIGDIR}/fig1d_generation.png", dpi=400)
    plt.close(fig)


def main():
    fig1a_overview()
    fig1b_grammar()
    fig1c_architecture()
    fig1d_generation()
    print("wrote 4 subfigures to", FIGDIR)


if __name__ == "__main__":
    main()
