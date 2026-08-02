"""Figure 3 - model architecture (standalone).

Left-to-right pipeline with restrained, over-set labels (no big centered titles
inside boxes):
  protein chains -> ESMC per-chain encoder -> unified token stream ->
  bidirectional LLaDA denoiser -> iterative unmasking (masked -> generated).
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts")
import figstyle as S  # noqa: E402


def stack(ax, x, y, w, h, color):
    for i in range(2, 0, -1):
        S.rbox(ax, x + i * 0.8, y + i * 0.8, w, h, fc=S.tint(color, 0.90),
               ec=color, lw=0.9, r=0.9, z=3 + (3 - i))
    S.rbox(ax, x, y, w, h, fc=S.tint(color, 0.90), ec=color, lw=1.4, r=0.9,
           z=7, shadow=True)


def over_label(ax, cx, ytop, name, sub=None):
    ax.text(cx, ytop, name, ha="center", va="bottom", fontsize=7.0,
            fontweight="bold", color=S.INK, zorder=9)
    if sub:
        ax.text(cx, ytop - 1.9, sub, ha="center", va="bottom", fontsize=5.7,
                color=S.SUB, zorder=9, style="italic")


def main():
    S.use_style()
    fig, ax = S.new_ax(9.8, 3.6)   # ylim ~ 36.7
    YM = 36.7

    ax.text(3.0, YM - 3.0, "Model architecture", fontsize=9.0, fontweight="bold",
            color=S.INK, ha="left")

    yc = 18.0

    # 1) protein chains
    over_label(ax, 8.5, 24.5, "protein chains")
    for yy, n, st in [(21.0, 8, 0), (19.0, 6, 8), (17.0, 9, 14)]:
        S.bead_chain(ax, 4.0, yy, n, r=0.5, dx=1.1, start=st)
    S.arrow(ax, (15.0, yc), (17.6, yc))

    # 2) ESMC encoder
    ex, ew = 18.5, 14.0
    stack(ax, ex, yc - 5.0, ew, 10.0, S.GREEN)
    over_label(ax, ex + ew / 2, 27.0, "ESMC per-chain encoder", "each chain encoded independently")
    for i, yy in enumerate([yc + 1.4, yc - 1.2, yc - 3.8]):
        S.bead_chain(ax, ex + 2.2, yy, 6, r=0.4, dx=0.9, start=i * 3)
    S.arrow(ax, (ex + ew + 0.8, yc), (ex + ew + 3.4, yc))

    # 3) unified token stream
    sx = ex + ew + 4.6      # 37.1
    segc = [S.NAVY, S.TEAL, S.PLUM, S.PLUM, S.CORAL, S.NAVY, S.TEAL, S.NAVY]
    bw = 2.0
    over_label(ax, sx + len(segc) * bw / 2, 23.0, "unified token stream")
    for i, c in enumerate(segc):
        S.rbox(ax, sx + i * bw, yc - 1.3, bw - 0.3, 2.6, fc=c, ec="white", lw=0.6, r=0.35, z=6)
    xend = sx + len(segc) * bw
    S.arrow(ax, (xend + 0.6, yc), (xend + 3.2, yc))

    # 4) bidirectional denoiser
    dx0 = xend + 4.4        # ~57.5
    dw = 14.0
    stack(ax, dx0, yc - 5.0, dw, 10.0, S.PLUM)
    over_label(ax, dx0 + dw / 2, 27.0, "bidirectional LLaDA denoiser", "full self-attention")
    for r in range(3):
        for cc in range(5):
            S.bead(ax, dx0 + 2.8 + cc * 1.9, yc + 1.4 - r * 1.8, 0.32,
                   filled=True, ci=(r + cc) % len(S.AA), z=8)
    S.arrow(ax, (dx0 + dw + 0.8, yc), (dx0 + dw + 3.4, yc))

    # 5) iterative unmasking (compact 3-state)
    ox = dx0 + dw + 4.6     # ~76.1
    n, r, dxx = 7, 0.6, 1.45
    over_label(ax, ox + n * dxx / 2, 27.0, "iterative unmasking")
    states = [(), (0, 1, 2), (0, 1, 2, 3, 4, 5)]
    cond = {0}
    gen_idx = [1, 2, 3, 4, 5, 6]
    for si, rev in enumerate(states):
        yy = yc + 4.6 - si * 4.6
        revealed = set(gen_idx[:len(rev)])
        for i in range(n):
            cxp = ox + r + i * dxx
            if i in cond:
                S.bead(ax, cxp, yy, r, filled=True, ci=i, cond=True)
            elif i in revealed:
                S.bead(ax, cxp, yy, r, filled=True, ci=i)
            else:
                S.bead(ax, cxp, yy, r, filled=False)
        if si < len(states) - 1:
            S.arrow(ax, (ox + n * dxx / 2, yy - 1.3), (ox + n * dxx / 2, yy - 3.3),
                    c=S.SUB, lw=1.0)
    ax.text(ox + n * dxx / 2, yc - 8.0, "t = T  \u2192  t = 0", ha="center",
            fontsize=6.0, color=S.SUB)

    # bottom legend
    ly = 4.0
    S.bead(ax, 4.2, ly, 0.6, filled=True, ci=2, cond=True)
    ax.text(5.4, ly, "conditioning context (fixed)", ha="left", va="center", fontsize=6.0, color=S.SUB)
    S.bead(ax, 40.0, ly, 0.6, filled=False)
    ax.text(41.2, ly, "masked position", ha="left", va="center", fontsize=6.0, color=S.SUB)
    S.bead(ax, 64.0, ly, 0.6, filled=True, ci=1)
    ax.text(65.2, ly, "generated residue", ha="left", va="center", fontsize=6.0, color=S.SUB)

    S.save(fig, "fig3_architecture")


if __name__ == "__main__":
    main()
