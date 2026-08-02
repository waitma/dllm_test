"""Figure 2 - the designed grammar token format (standalone).

structure delimiters | type tab | relation node | residues (amino-acid beads).
Three record forms: unconditional and two conditional layouts, with the fixed
context vs generated segments tinted.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts")
import figstyle as S  # noqa: E402

RB, DX, GAP = 0.62, 1.55, 2.0


def capsule(ax, x, y, h, tlabel, groups, state):
    tabw = 0.85 * len(tlabel) + 2.6
    bw = 0.0
    for gi, (n, _) in enumerate(groups):
        bw += (n - 1) * DX + 2 * RB + (GAP if gi < len(groups) - 1 else 0)
    pad = 1.5
    w = tabw + 1.2 + bw + pad * 2
    fill = S.tint(S.NAVY, 0.92) if state == "cond" else S.tint(S.CORAL, 0.90)
    S.rbox(ax, x, y, w, h, fc=fill, ec=S.NAVY, lw=1.3, r=1.2, z=4, shadow=True)

    # type tab
    S.rbox(ax, x + 1.2, y + h / 2 - 1.7, tabw, 3.4, fc=S.tint(S.TEAL, 0.80),
           ec=S.TEAL, lw=1.0, r=0.9, z=6)
    ax.text(x + 1.2 + tabw / 2, y + h / 2, tlabel, ha="center", va="center",
            fontsize=6.0, color=S.INK, zorder=7)

    bx = x + 1.2 + tabw + pad + RB
    for gi, (n, start) in enumerate(groups):
        S.bead_chain(ax, bx, y + h / 2, n, r=RB, dx=DX, start=start)
        bx += (n - 1) * DX
        if gi < len(groups) - 1:
            ax.text(bx + GAP / 2, y + h / 2 - 0.2, "\u00b7", ha="center", va="center",
                    fontsize=11, color=S.SUB, zorder=7)
            bx += GAP + RB
        else:
            bx += RB
    return x + w


def relation(ax, x, ymid, label):
    w = 0.85 * len(label) + 3.0
    ax.plot([x, x + 1.4], [ymid, ymid], color=S.CORAL, lw=1.3, zorder=3)
    S.rbox(ax, x + 1.4, ymid - 1.7, w, 3.4, fc=S.tint(S.CORAL, 0.86), ec=S.CORAL,
           lw=1.0, r=1.7, z=5)
    ax.text(x + 1.4 + w / 2, ymid, label, ha="center", va="center",
            fontsize=5.9, color=S.CORAL, zorder=6)
    ax.plot([x + 1.4 + w, x + 1.4 + w + 1.4], [ymid, ymid], color=S.CORAL, lw=1.3, zorder=3)
    return x + 1.4 + w + 1.4


def main():
    S.use_style()
    fig, ax = S.new_ax(7.4, 4.2)   # ylim ~ 56.8
    YM = 56.8
    x0 = 3.0

    ax.text(x0, YM - 3.0, "Designed grammar token format", fontsize=9.0,
            fontweight="bold", color=S.INK, ha="left")

    # ---- legend (left-aligned labels) ----
    ly = YM - 8.5
    lx = x0
    S.rbox(ax, lx, ly, 4.0, 3.2, fc="#fff", ec=S.NAVY, lw=1.3, r=0.9, z=4)
    ax.text(lx + 5.0, ly + 1.6, "structure delimiter", ha="left", va="center",
            fontsize=6.6, color=S.INK)
    lx += 26
    S.rbox(ax, lx, ly, 4.0, 3.2, fc=S.tint(S.TEAL, 0.80), ec=S.TEAL, lw=1.0, r=0.8, z=4)
    ax.text(lx + 5.0, ly + 1.6, "type tab", ha="left", va="center", fontsize=6.6, color=S.INK)
    lx += 16
    S.rbox(ax, lx, ly, 4.0, 3.2, fc=S.tint(S.CORAL, 0.86), ec=S.CORAL, lw=1.0, r=1.6, z=4)
    ax.text(lx + 5.0, ly + 1.6, "relation", ha="left", va="center", fontsize=6.6, color=S.INK)
    lx += 15
    for i in range(3):
        S.bead(ax, lx + 0.6 + i * 1.5, ly + 1.6, RB, filled=True, ci=i)
    ax.text(lx + 0.6 + 3 * 1.5 + 1.0, ly + 1.6, "residues", ha="left", va="center",
            fontsize=6.6, color=S.INK)

    # ---- records ----
    def record(title, ybot, items):
        ax.text(x0, ybot + 6.0, title, ha="left", va="center", fontsize=7.2,
                fontweight="bold", color=S.INK)
        x = x0
        for it in items:
            if it[0] == "cap":
                _, tl, groups, state = it
                xe = capsule(ax, x, ybot, 5.2, tl, groups, state)
                ax.text((x + xe) / 2, ybot - 1.9, "fixed context" if state == "cond" else "generated",
                        ha="center", va="center", fontsize=5.9,
                        color=(S.NAVY if state == "cond" else S.CORAL))
                x = xe + 1.8
            else:
                x = relation(ax, x, ybot + 2.6, it[1]) + 1.8

    record("OTS (TCR)  \u00b7  unconditional", 34.5, [
        ("cap", "<tcr>", [(6, 0), (6, 6)], "gen")])
    record("TCR + peptide  \u00b7  conditional", 22.0, [
        ("cap", "<pep>", [(5, 0)], "cond"),
        ("rel", "binding"),
        ("cap", "<tcr>", [(5, 5), (5, 10)], "gen")])
    record("PPI / STRING actions  \u00b7  conditional", 9.5, [
        ("cap", "<ab>", [(8, 0)], "cond"),
        ("rel", "binding"),
        ("cap", "<ag>", [(7, 8)], "gen")])

    ax.text(x0, 2.6, "Chain identity is carried by chain / position embeddings; "
            "conditional tasks fix a context segment and generate the remainder.",
            ha="left", va="center", fontsize=6.2, color=S.SUB, style="italic")

    S.save(fig, "fig2_grammar")


if __name__ == "__main__":
    main()
