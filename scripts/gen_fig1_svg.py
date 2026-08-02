"""Figure 1 subpanels, authored from the real Ophiuchus-X implementation.

Grounding (code, not guesswork):
  * grammar.py         -> grammar-v2 token stream: <prots>..<protd> blocks,
                          type markers <ab>/<tcr>/<nb>/<pep>, relation tokens
                          <binding>..; residues in one-letter code; a fixed
                          context vs a diffusion target span.
  * modeling_bioseq.py -> per-chain ESMC encoder on the noisy proxy x_t (the
                          decoder's <mask> corruption is mirrored onto the
                          encoder input); gathered residue features REPLACE the
                          decoder token embeddings at residue sites
                          (hidden = wte*(1-m) + esmc*m), while structure /
                          relation tokens keep their token embeddings; a
                          bidirectional LLaDA Transformer denoises; masked
                          cross-entropy on target residues only.
  * sampling_bioseq.py -> confidence-based iterative unmasking; structure /
                          relation / fixed-context positions stay visible.

Style: flat vector, Wong colour-blind-safe palette, Helvetica/Arial labels,
amino acids in Courier one-letter code, hairline rules, no shadows / no cards.
Panel letters come from the LaTeX subcaptions.

Run: python scripts/gen_fig1_svg.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts")
import figsvg as F  # noqa: E402
from figsvg import (INK, SUB, MUT, HAIR, FAINT, BLUE, GREEN, PURPLE, ORANGE,  # noqa: E402
                    VERM, SKY, tint, kicker, save, char_w)
Canvas = F.Canvas

EPITOPE = "GILGFVFTL"      # influenza M1 pMHC-I epitope
CDR3B = "CASSIRSSYEQYF"    # a TRB CDR3-beta


# =======================================================================
# (a) OVERVIEW  -- PyMOL Nature-coloured structure icons → model → tasks
#     Icons: 1AO7 TCR–pMHC · 1NL0 Ab–Ag · 3K1K Nb–Ag · 1BRS PPI
#     Layout: compact single flow (no left/right split, no title overlap)
# =======================================================================
ICONDIR = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/figures/icons"

# Nature-like accents aligned with PyMOL chain palette
N_STEEL = "#3C6E8F"
N_TERR = "#C17F65"
N_SAGE = "#5B8F7A"
N_SLATE = "#8B7BA0"


def _ig_domain(c, cx, cy, w, h, color, tilt=0.0):
    """One immunoglobulin-fold domain: a two-tone rounded capsule.

    `tilt` slides the top edge sideways so a paired VH/VL can splay outward.
    """
    rx = min(w, h) * 0.46
    x0, y0 = cx - w / 2, cy - h / 2
    xr, yb = x0 + w, y0 + h
    c.path(
        f"M{x0 + tilt:.2f},{y0 + rx:.2f} "
        f"q0,{-rx:.2f} {rx:.2f},{-rx:.2f} "
        f"h{w - 2 * rx:.2f} "
        f"q{rx:.2f},0 {rx:.2f},{rx:.2f} "
        f"L{xr:.2f},{yb - rx:.2f} "
        f"q0,{rx:.2f} {-rx:.2f},{rx:.2f} "
        f"h{-(w - 2 * rx):.2f} "
        f"q{-rx:.2f},0 {-rx:.2f},{-rx:.2f} "
        f"Z",
        fill=tint(color, 0.62), stroke=color, sw=1.35)
    c.line(cx - w / 2 + 2.5, cy, cx + w / 2 - 2.5, cy,
           stroke=tint(color, 0.15), sw=0.7)


def _cdr_loops(c, x0, y_top, span, color, n=3):
    """A row of small denoising/binding loops arcing above a domain top edge."""
    step = span / n
    for i in range(n):
        bx = x0 - span / 2 + step * (i + 0.5)
        c.path(f"M{bx - step * 0.42},{y_top} q{step * 0.42},{-5.4} "
               f"{step * 0.84},0", stroke=color, sw=1.7)


def _task_icon(c, kind, cx, cy):
    """Unified biological glyphs built from one Ig-fold visual language."""
    if kind == "tcr":
        # MHC platform holds a peptide; TCR Valpha/Vbeta dock from above.
        c.rect(cx - 19, cy + 9, 38, 8.5, fill=tint(N_SLATE, 0.5),
               stroke=N_SLATE, sw=1.2, rx=3)
        c.line(cx - 10, cy + 8, cx + 10, cy + 8, stroke=N_TERR, sw=2.8)
        _ig_domain(c, cx - 6.5, cy - 6, 10.5, 22, N_STEEL, tilt=-2.4)
        _ig_domain(c, cx + 6.5, cy - 6, 10.5, 22, N_SAGE, tilt=2.4)
        for dx, col in ((-6.5, N_STEEL), (6.5, N_SAGE)):
            c.path(f"M{cx + dx - 3},{cy + 5} q3,4 6,0", stroke=col, sw=1.4)
    elif kind == "ppi":
        # Two globular domains meeting at a highlighted binding interface.
        c.circle(cx - 9.5, cy, 12, fill=tint(N_STEEL, 0.6), stroke=N_STEEL, sw=1.35)
        c.circle(cx + 9.5, cy, 10.5, fill=tint(N_TERR, 0.62), stroke=N_TERR, sw=1.35)
        for dy in (-4.5, 0, 4.5):
            c.circle(cx, cy + dy, 1.5, fill=N_SLATE)
    elif kind == "nb":
        # Single VHH domain (3 CDR loops) beside a small multi-property panel.
        _ig_domain(c, cx - 8, cy, 13, 27, N_STEEL)
        _cdr_loops(c, cx - 8, cy - 13.5, 11, N_TERR)
        for i, col in enumerate((N_STEEL, N_SAGE, N_TERR)):
            c.rect(cx + 4, cy - 10 + i * 8.4, 16, 4.6,
                   fill=tint(col, 0.5), stroke=col, sw=0.95, rx=2.3)
    elif kind == "cdr":
        # Paired VH/VL splayed outward, three CDR loops highlighted on each.
        _ig_domain(c, cx - 7.5, cy + 1, 12, 26, N_STEEL, tilt=-2.6)
        _ig_domain(c, cx + 7.5, cy + 1, 12, 26, N_SAGE, tilt=2.6)
        _cdr_loops(c, cx - 7.5, cy - 12, 10, N_TERR)
        _cdr_loops(c, cx + 7.5, cy - 12, 10, N_TERR)
    elif kind == "sample":
        # Grammar-visible context and a masked target denoised into residues.
        cell_w, cell_h = 8, 9
        x0 = cx - 27
        top = [
            (tint(N_STEEL, 0.76), N_STEEL, None),
            (tint(N_SAGE, 0.76), N_SAGE, None),
            ("#FFFFFF", MUT, "2,1.5"),
            ("#FFFFFF", MUT, "2,1.5"),
            ("#FFFFFF", MUT, "2,1.5"),
            (tint(N_STEEL, 0.76), N_STEEL, None),
        ]
        bottom = [
            (tint(N_STEEL, 0.76), N_STEEL),
            (tint(N_SAGE, 0.76), N_SAGE),
            (tint(N_TERR, 0.72), N_TERR),
            (tint(N_TERR, 0.72), N_TERR),
            (tint(N_TERR, 0.72), N_TERR),
            (tint(N_STEEL, 0.76), N_STEEL),
        ]
        for i, (fill, stroke, dash) in enumerate(top):
            c.rect(x0 + i * 9.2, cy - 17, cell_w, cell_h, fill=fill,
                   stroke=stroke, sw=1.0, rx=1, dash=dash)
        c.arrow(cx, cy - 5, cx, cy + 3, stroke=MUT, sw=1.2, head=4)
        for i, (fill, stroke) in enumerate(bottom):
            c.rect(x0 + i * 9.2, cy + 7, cell_w, cell_h, fill=fill,
                   stroke=stroke, sw=1.0, rx=1)


def _nn_cartoon(c, x, y, w, h):
    """Sparse neural-network glyph: enough structure without a line web."""
    layers = [
        (3, "#9EC0D8"),
        (4, "#8FCBB8"),
        (4, "#B0B8D0"),
        (2, "#E0A898"),
    ]
    pad_x, pad_y = 18, 10
    usable_w = w - 2 * pad_x
    usable_h = h - 2 * pad_y
    xs = [x + pad_x + usable_w * i / (len(layers) - 1)
          for i in range(len(layers))]
    centres = []
    for (n, col), lx in zip(layers, xs):
        span = usable_h * 0.8
        y0 = y + pad_y + (usable_h - span) / 2
        ys = [y0 + span * j / (n - 1) for j in range(n)]
        centres.append([(lx, yy, col) for yy in ys])

    # Each node connects only to its two nearest neighbours in the next layer.
    for li in range(len(centres) - 1):
        target = centres[li + 1]
        for x0, y0, _ in centres[li]:
            nearest = sorted(target, key=lambda p: abs(p[1] - y0))[:2]
            for x1, y1, _ in nearest:
                c.line(x0, y0, x1, y1, stroke="#7890A3", sw=0.8)

    for layer in centres:
        for cx, cy, col in layer:
            c.circle(cx, cy, 4.5, fill="#FFFFFF", stroke=col, sw=1.3)


def _simple_fork(c, x0, y0, x1, y_top, y_bot):
    """Two quiet rays from one junction; no brackets and no arrowheads."""
    joint_x = x0 + 8
    c.line(x0, y0, joint_x, y0, stroke=MUT, sw=1.25, cap="butt")
    c.line(joint_x, y0, x1, y_top, stroke=N_STEEL, sw=1.25, cap="butt")
    c.line(joint_x, y0, x1, y_bot, stroke=N_TERR, sw=1.25, cap="butt")
    c.circle(joint_x, y0, 2.0, fill=MUT)


def fig1a_overview():
    """Unified biological overview: real complexes, model, and task glyphs."""
    W, H = 930, 260
    c = Canvas(W, H)

    # ---- left: real structural inputs, with a single quiet crop boundary ----
    kicker(c, 16, 14, "Complexes", fill=SUB, size=10.5)
    icons = [
        (48, 60, "tcr_pmhc.png", "TCR\u2013pMHC"),
        (132, 60, "antibody.png", "Ab\u2013Ag"),
        (48, 164, "nanobody.png", "Nb\u2013Ag"),
        (132, 164, "ppi.png", "PPI"),
    ]
    iw = 66
    for cx, cy, fname, lab in icons:
        x, y = cx - iw / 2, cy - iw / 2
        c.rect(x, y, iw, iw, fill="#FFFFFF", stroke=HAIR, sw=0.9, rx=3)
        c.image(x + 1, y + 1, iw - 2, iw - 2, f"{ICONDIR}/{fname}",
                rx=2, frame=False)
        c.text(cx, y + iw + 13, lab, size=10.5, fill=SUB, anchor="middle")

    midy = 108
    c.arrow(176, midy, 202, midy, stroke=MUT, sw=1.5, head=5.5)

    # ---- centre: one borderless cool block with a sparse network ----
    mx, my, mw, mh = 210, 36, 162, 144
    face = "#49677F"
    c.rect(mx, my, mw, mh, fill=face, stroke="none", sw=0, rx=5)
    c.text(mx + mw / 2, my + 22, "Ophiuchus-X", size=14, fill="#FFFFFF",
           weight="bold", anchor="middle")
    _nn_cartoon(c, mx + 5, my + 33, mw - 10, mh - 43)

    # ---- right: two open lanes; task icons are vector biology cartoons ----
    row_x0 = 430
    row_w = 475
    y_und, y_gen = 70, 190
    # Rays terminate in the two lane rules, not in free space.
    _simple_fork(c, mx + mw, midy, row_x0, y_und - 25, y_gen - 25)

    def _branch_row(x0, y_icon, w, color, branch, title, tasks):
        c.text(x0, y_icon - 35, branch.upper(), size=10, fill=color,
               weight="bold", spacing=1.1)
        c.text(x0 + 92, y_icon - 35, title, size=12, fill=INK,
               weight="bold")
        c.line(x0, y_icon - 25, x0 + w, y_icon - 25,
               stroke=HAIR, sw=0.9, cap="butt")
        gap = 24
        tw = (w - gap * (len(tasks) - 1)) / len(tasks)
        for i, (kind, label, detail) in enumerate(tasks):
            cx = x0 + i * (tw + gap) + tw / 2
            _task_icon(c, kind, cx, y_icon)
            c.text(cx, y_icon + 34, label, size=11, fill=INK,
                   weight="bold", anchor="middle")
            c.text(cx, y_icon + 50, detail, size=9.8, fill=SUB,
                   anchor="middle")

    _branch_row(
        row_x0, y_und, row_w, N_STEEL,
        "Understand", "Prediction & representation",
        [
            ("tcr", "TCR\u2013peptide", "binding \u00b7 clustering"),
            ("ppi", "PPI affinity", "\u0394\u0394G \u00b7 classification"),
            ("nb", "Nanobody", "12-property panel"),
        ],
    )
    _branch_row(
        row_x0, y_gen, row_w, N_TERR,
        "Generate", "Conditional generation",
        [
            ("cdr", "Antibody CDR", "design \u00b7 H\u2013L pairing"),
            ("sample", "Grammar sampling", "constrained denoising"),
        ],
    )
    save(c, "fig1a_overview")


# =======================================================================
# (b) GRAMMAR  -- the real grammar-v2 layout, fixed context vs target span
# =======================================================================
def _run(c, x, ymid, seq, state):
    for ch in seq:
        x = c.tok(x, ymid, ch, "res", state=state)
    return x


def fig1b_grammar():
    W, H = 1250, 300
    c = Canvas(W, H)

    # legend row
    ly = 26
    x = 30
    x = c.tok(x, ly, "<prots>", "struct") + 8
    x = c.tok(x, ly, "<protd>", "struct") + 14
    c.text(x, ly + 4, "block", size=11.5, fill=SUB); x += char_w("block", 11.5) + 26
    x = c.tok(x, ly, "<tcr>", "type") + 10
    c.text(x, ly + 4, "type", size=11.5, fill=SUB); x += char_w("type", 11.5) + 26
    x = c.tok(x, ly, "<binding>", "rel") + 10
    c.text(x, ly + 4, "relation", size=11.5, fill=SUB); x += char_w("relation", 11.5) + 24
    x = c.tok(x, ly, "A", "res", state="fixed") + 3
    x = c.tok(x, ly, "M", "res", state="target") + 3
    x = c.tok(x, ly, " ", "res", state="mask") + 8
    c.text(x, ly + 4, "fixed / target / masked residue", size=11.5, fill=SUB)
    c.line(30, 46, W - 30, 46, stroke=HAIR, sw=1.0)

    def record(title, ytop, build):
        c.text(30, ytop, title, size=12.5, weight="bold", fill=INK)
        build(ytop + 30)

    # 1) Antibody pair -- unconditional (whole record is the target span)
    def rec_ab(ymid):
        x = 30
        x = c.tok(x, ymid, "<prots>", "struct") + 4
        x = c.tok(x, ymid, "<ab>", "type") + 6
        x = _run(c, x, ymid, "QVQLVQ", "target") + 3
        x = c.tok(x, ymid, ".", "sep") + 3
        x = _run(c, x, ymid, "DIQMTQ", "target") + 4
        x = c.tok(x, ymid, "<protd>", "struct")
        c.text(30, ymid + 30, "heavy \u00b7 light chains, all residues generated",
               size=11, fill=MUT)
    record("Antibody pair  \u00b7  unconditional", 74, rec_ab)

    # 2) TCR-pMHC -- conditional, the full three-entity layout
    def rec_tcr(ymid):
        x = 30
        x = c.tok(x, ymid, "<prots>", "struct") + 3
        x = _run(c, x, ymid, "MHC", "fixed") + 3
        x = c.tok(x, ymid, ".", "sep") + 3
        x = _run(c, x, ymid, "B2M", "fixed") + 3
        x = c.tok(x, ymid, "<protd>", "struct") + 6
        x = c.tok(x, ymid, "<binding>", "rel") + 6
        x = c.tok(x, ymid, "<prots>", "struct") + 3
        x = c.tok(x, ymid, "<pep>", "type") + 4
        x = _run(c, x, ymid, EPITOPE, "fixed") + 3
        x = c.tok(x, ymid, "<protd>", "struct") + 6
        x = c.tok(x, ymid, "<binding>", "rel") + 6
        x = c.tok(x, ymid, "<prots>", "struct") + 3
        x = c.tok(x, ymid, "<tcr>", "type") + 4
        x = _run(c, x, ymid, "CASSIR", "target") + 3
        x = c.tok(x, ymid, ".", "sep") + 3
        x = _run(c, x, ymid, "CASSLG", "target") + 3
        x = c.tok(x, ymid, "<protd>", "struct")
        c.text(30, ymid + 30, "MHC\u00b7B2M and peptide are fixed context; the TCR \u03b1/\u03b2 span is generated",
               size=11, fill=MUT)
    record("TCR\u2013pMHC  \u00b7  conditional", 158, rec_tcr)

    # 3) Protein-protein / STRING -- conditional
    def rec_ppi(ymid):
        x = 30
        x = c.tok(x, ymid, "<prots>", "struct") + 3
        x = _run(c, x, ymid, "KVFGRCEL", "fixed") + 3
        x = c.tok(x, ymid, "<protd>", "struct") + 6
        x = c.tok(x, ymid, "<binding>", "rel") + 6
        x = c.tok(x, ymid, "<prots>", "struct") + 3
        x = _run(c, x, ymid, "AARDGYYG", "target") + 3
        x = c.tok(x, ymid, "<protd>", "struct")
        c.text(30, ymid + 30, "partner A fixed, typed relation, partner B generated",
               size=11, fill=MUT)
    record("Protein\u2013protein / STRING  \u00b7  conditional", 242, rec_ppi)

    save(c, "fig1b_grammar")


# =======================================================================
# (c) ARCHITECTURE  -- faithful dual-path: per-chain ESMC (corruption-
#     mirrored) features REPLACE residue embeddings; bidirectional LLaDA.
# =======================================================================
def _stack(c, x, y, w, h, color, name, sub):
    for i in (2, 1):
        c.rect(x + i * 5, y - i * 5, w, h, fill="#FFFFFF", stroke=color, sw=1.1, rx=2)
    c.rect(x, y, w, h, fill=tint(color, 0.9), stroke=color, sw=1.8, rx=2)
    c.text(x + w / 2, y + h / 2 + 1, name, size=15, fill=color, weight="bold", anchor="middle")
    c.text(x + w / 2, y + h / 2 + 16, sub, size=10, fill=SUB, anchor="middle")


def _mini_stream(c, x, ymid, specs, size=10, h=20, pad=5, track=False):
    """Render a compact grammar stream.

    If track=True, also return [(group, x0, x1), ...] where group is
    'feat' (ESMC columns) or 'tok' (structure / relation / residue chips).
    """
    segs = []
    cur_kind = None
    seg_x0 = x
    for kind, label, state in specs:
        group = "feat" if kind == "feat" else "tok"
        x_before = x
        if kind == "res":
            x = c.tok(x, ymid, label, "res", state=state, size=size, h=h) + 1
        elif kind == "feat":
            fw = size + 4
            c.featcol(x, ymid, fw, h - 2, seed=int(label))
            x = x + fw + 1
        else:
            x = c.tok(x, ymid, label, kind, size=size, h=h, pad=pad) + 2
        if track:
            if group != cur_kind:
                if cur_kind is not None:
                    segs.append((cur_kind, seg_x0, x_before))
                cur_kind = group
                seg_x0 = x_before
    if track and cur_kind is not None:
        segs.append((cur_kind, seg_x0, x))
    return (x, segs) if track else x


def _brace(c, x0, x1, y, label, color):
    """Square brace under a span, with optional centred label."""
    if x1 - x0 < 8:
        return
    tick = 5
    c.line(x0, y, x0, y - tick, stroke=color, sw=1.2)
    c.line(x0, y, x1, y, stroke=color, sw=1.2)
    c.line(x1, y, x1, y - tick, stroke=color, sw=1.2)
    if label:
        c.text((x0 + x1) / 2, y + 12, label, size=9.5, fill=color, anchor="middle")


def fig1c_architecture():
    """Dual-path fusion; fused box braces mark ESMC vs token-embedding cells."""
    W, H = 1360, 345
    c = Canvas(W, H)

    xt_specs = [
        ("struct", "<prots>", "f"), ("type", "<pep>", "f"),
        ("res", "G", "fixed"), ("res", "I", "fixed"), ("res", "L", "fixed"),
        ("struct", "<protd>", "f"),
        ("rel", "<binding>", "f"),
        ("struct", "<prots>", "f"), ("type", "<tcr>", "f"),
        ("res", " ", "mask"), ("res", " ", "mask"), ("res", " ", "mask"),
        ("struct", "<protd>", "f"),
    ]
    fuse_specs = [
        ("struct", "<prots>", "f"), ("type", "<pep>", "f"),
        ("feat", "0", ""), ("feat", "1", ""), ("feat", "2", ""),
        ("struct", "<protd>", "f"),
        ("rel", "<binding>", "f"),
        ("struct", "<prots>", "f"), ("type", "<tcr>", "f"),
        ("feat", "3", ""), ("feat", "4", ""), ("feat", "5", ""),
        ("struct", "<protd>", "f"),
    ]

    band1_midy = 76
    band2_midy = 188

    # ---- band 1 ----
    xt_h = 40
    xt_x, xt_y = 30, band1_midy - xt_h / 2
    xt_end = _mini_stream(c, xt_x + 10, band1_midy, xt_specs, size=10, h=22, pad=5)
    xt_w = xt_end - xt_x + 10
    c.rect(xt_x, xt_y, xt_w, xt_h, stroke=HAIR, sw=1.1, rx=2)
    c.text_xt(xt_x + xt_w / 2, xt_y - 12, prefix="noisy token ", size=12,
              fill=SUB, anchor="middle")

    esmc_w, esmc_h = 160, 56
    esmc_x = xt_x + xt_w + 70
    esmc_y = band1_midy - esmc_h / 2
    c.arrow(xt_x + xt_w + 4, band1_midy, esmc_x - 6, band1_midy,
            stroke=MUT, sw=1.8, head=8)
    _stack(c, esmc_x, esmc_y, esmc_w, esmc_h, GREEN, "ESMC",
           "per-chain \u00b7 trainable")
    c.text(esmc_x + esmc_w + 12, band1_midy + 4,
           "mirrors the x_t <mask> corruption", size=10, fill=MUT, anchor="start")

    # ---- band 2 ----
    te_w, te_h = 140, 44
    te_x = xt_x + (xt_w - te_w) / 2
    te_y = band2_midy - te_h / 2

    drop_x = xt_x + xt_w / 2
    c.line(drop_x, xt_y + xt_h + 2, drop_x, te_y - 10,
           stroke=MUT, sw=1.4, dash="4,3")
    c.arrow(drop_x, te_y - 12, drop_x, te_y - 2, stroke=MUT, sw=1.4, head=6)

    c.rect(te_x, te_y, te_w, te_h, fill=FAINT, stroke=HAIR, sw=1.3, rx=2)
    c.text(te_x + te_w / 2, te_y + 18, "token embeddings", size=11.5, fill=INK,
           weight="bold", anchor="middle")
    c.text(te_x + te_w / 2, te_y + 34, "structure / relation", size=10, fill=MUT,
           anchor="middle")

    fuse_h = 50
    fuse_inner, _ = _mini_stream(Canvas(1, 1), 0, 0, fuse_specs,
                                 size=10, h=24, pad=5, track=True)
    fuse_w = fuse_inner + 20
    fuse_x = te_x + te_w + 55
    fuse_y = band2_midy - fuse_h / 2
    c.rect(fuse_x, fuse_y, fuse_w, fuse_h, fill="#FFFFFF", stroke=BLUE,
           sw=1.6, rx=2)
    _, segs = _mini_stream(c, fuse_x + 10, band2_midy, fuse_specs,
                           size=10, h=24, pad=5, track=True)

    feat_spans = [(a, b) for kind, a, b in segs if kind == "feat"]
    tok_spans = [(a, b) for kind, a, b in segs if kind == "tok"]

    brace_y = fuse_y + fuse_h + 10
    for i, (a, b) in enumerate(feat_spans):
        _brace(c, a, b, brace_y, "ESMC" if i == 0 else "ESMC", GREEN)
    # mark every tok span; put text on the widest one to avoid clutter
    if tok_spans:
        widest = max(tok_spans, key=lambda ab: ab[1] - ab[0])
        for a, b in tok_spans:
            lab = "token emb." if (a, b) == widest else ""
            _brace(c, a, b, brace_y, lab, MUT)

    c.text(fuse_x + fuse_w / 2, brace_y + 28,
           "blue columns = residue sites from ESMC"
           "   \u00b7   coloured chips = structure/relation from token embeddings",
           size=10, fill=MUT, anchor="middle")

    c.arrow(te_x + te_w + 4, band2_midy, fuse_x - 6, band2_midy,
            stroke=MUT, sw=1.6, head=7)

    # ESMC drops onto both feat groups (lines stay centred on the landings);
    # only the "residue features" label is shifted to the right of the bar.
    esmc_cx = esmc_x + esmc_w / 2
    elbow_y = fuse_y - 22
    if feat_spans:
        lands = [(a + b) / 2 for a, b in feat_spans]
        x0, x1 = min(lands), max(lands)
        c.line(esmc_cx, esmc_y + esmc_h + 4, esmc_cx, elbow_y, stroke=GREEN, sw=1.8)
        c.line(min(esmc_cx, x0), elbow_y, max(esmc_cx, x1), elbow_y,
               stroke=GREEN, sw=1.8)
        for lx in lands:
            c.arrow(lx, elbow_y, lx, fuse_y - 3, stroke=GREEN, sw=1.6, head=6)
        c.text(max(esmc_cx, x1) + 10, elbow_y - 6, "residue features",
               size=11, fill=GREEN, anchor="start")

    ll_w, ll_h = 150, 58
    # pull LLaDA left so the LLaDA→CE gap can hold "logits" cleanly
    ll_x = fuse_x + fuse_w + 108
    ll_y = band2_midy - ll_h / 2
    c.arrow(fuse_x + fuse_w + 6, band2_midy, ll_x - 6, band2_midy,
            stroke=MUT, sw=1.8, head=8)
    _stack(c, ll_x, ll_y, ll_w, ll_h, PURPLE, "LLaDA", "bidirectional \u00d7 N")
    c.text(ll_x + ll_w / 2, ll_y + ll_h + 18,
           "RoPE \u00b7 RMSNorm \u00b7 SwiGLU", size=10, fill=MUT, anchor="middle")

    ce_w, ce_h = 155, 50
    ce_x = ll_x + ll_w + 70
    ce_y = band2_midy - ce_h / 2
    c.arrow(ll_x + ll_w + 4, band2_midy, ce_x - 6, band2_midy,
            stroke=MUT, sw=1.8, head=8)
    # keep "logits" clearly left of the CE box (not entering its border)
    c.text(ce_x - 38, band2_midy - 14,
           "logits", size=10, fill=SUB, anchor="middle")
    c.rect(ce_x, ce_y, ce_w, ce_h, fill=tint(VERM, 0.94), stroke=VERM,
           sw=1.5, rx=2)
    c.text(ce_x + ce_w / 2, ce_y + 22, "masked cross-entropy", size=11.5,
           fill=VERM, weight="bold", anchor="middle")
    c.text(ce_x + ce_w / 2, ce_y + 38, "target residues only", size=10,
           fill=SUB, anchor="middle")

    save(c, "fig1c_architecture")


def fig1d_generation():
    # Abbreviated CDR3: prefix + "..." + suffix  (ellipsis = many residues omitted)
    left = list("CASS")
    right = list("YEQYF")
    n_left, n_right = len(left), len(right)
    # reveal order over the visible slots only (left then right)
    # indices: 0..n_left-1 | ellipsis | n_left .. n_left+n_right-1
    n_vis = n_left + n_right
    reveal_order = [0, n_left + 1, 2, n_left + 3, 1, n_left, 3, n_left + 2, n_left + 4]
    # keep only valid indices
    reveal_order = [i for i in reveal_order if i < n_vis]
    schedule = [0, 3, 6, n_vis]
    tops = ["t = T", "", "", "t = 0"]
    subs = ["target masked", "high-confidence sites", "more sites", "sequence"]
    cell = 15.0
    h = 18.0
    dots_w = 18.0  # width reserved for "..."

    def _stage_w():
        dummy = Canvas(1, 1)
        x = 0
        x = dummy.tok(x, 0, "<prots>", "struct", size=10, h=h, pad=5) + 3
        x = dummy.tok(x, 0, "<tcr>", "type", size=10, h=h, pad=5) + 4
        x += n_left * (cell + 1) + dots_w + 4 + n_right * (cell + 1)
        x = dummy.tok(x, 0, "<protd>", "struct", size=10, h=h, pad=5)
        return x

    block_w = _stage_w()
    gap = 36
    total = 4 * block_w + 3 * gap
    margin = 40
    # auto-size canvas so the last <protd> is never clipped
    W = int(total + 2 * margin)
    H = 220
    c = Canvas(W, H)
    c.text(W / 2, 28,
           "confidence-ordered iterative unmasking  \u00b7  fixed context stays visible",
           size=12, fill=SUB, anchor="middle")

    x_start = (W - total) / 2
    xs = [x_start + i * (block_w + gap) for i in range(4)]
    y = 100

    def _draw_residues(x, revealed):
        """Draw left residues, ellipsis, right residues. Returns new x."""
        for i, ch in enumerate(left):
            if i in revealed:
                c.tok(x, y, ch, "res", state="target", size=10, h=h)
            else:
                c.tok(x, y, " ", "res", state="mask", size=10, h=h)
            x += cell + 1
        # ellipsis standing in for the omitted middle residues
        c.text(x + dots_w / 2, y + 4, "\u2026", size=14, fill=MUT, anchor="middle")
        x += dots_w + 4
        for j, ch in enumerate(right):
            idx = n_left + j
            if idx in revealed:
                c.tok(x, y, ch, "res", state="target", size=10, h=h)
            else:
                c.tok(x, y, " ", "res", state="mask", size=10, h=h)
            x += cell + 1
        return x

    for si, (x0, k, top, sub) in enumerate(zip(xs, schedule, tops, subs)):
        revealed = set(reveal_order[:k])
        x = x0
        x = c.tok(x, y, "<prots>", "struct", size=10, h=h, pad=5) + 3
        x = c.tok(x, y, "<tcr>", "type", size=10, h=h, pad=5) + 4
        x = _draw_residues(x, revealed)
        c.tok(x, y, "<protd>", "struct", size=10, h=h, pad=5)
        if top:
            c.text(x0 + block_w / 2, y - 22, top, size=12, weight="bold",
                   fill=INK, anchor="middle")
        c.text(x0 + block_w / 2, y + 28, sub, size=10.5, fill=SUB, anchor="middle")
        if si < 3:
            c.arrow(x0 + block_w + 4, y, xs[si + 1] - 4, y, stroke=MUT, sw=2.0,
                    head=8)

    # legend
    ly = 186
    c.tok(40, ly, "<prots>", "struct", size=10, h=16, pad=5)
    c.tok(98, ly, "<tcr>", "type", size=10, h=16, pad=5)
    c.tok(148, ly, "<protd>", "struct", size=10, h=16, pad=5)
    c.text(210, ly + 4, "fixed context (block / type / conditioning)", size=11,
           fill=SUB)
    c.tok(500, ly, " ", "res", state="mask", size=10, h=16)
    c.text(522, ly + 4, "masked", size=11, fill=SUB)
    c.tok(590, ly, "A", "res", state="target", size=10, h=16)
    c.text(612, ly + 4, "generated residue", size=11, fill=SUB)
    save(c, "fig1d_generation")


def main():
    fig1a_overview()
    fig1b_grammar()
    fig1c_architecture()
    fig1d_generation()
    print("done -> 4 faithful subpanels in", F.FIGDIR)


if __name__ == "__main__":
    main()
