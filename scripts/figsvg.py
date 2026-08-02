"""Nature-caliber SVG design system for the paper's schematic figures.

Design contract (distilled from Nature's artwork guide + ppt-master's
swiss-minimal / editorial discipline):

* Flat vector only. NO drop shadows, NO glossy cards, NO gradient fills —
  structure, hairline rules and whitespace carry the page (Swiss discipline).
* Sharp geometry: rectangles with rx 0-3, true circles, single-weight rules.
* Colour: cool, low-saturation SCI palette (Paul Tol / Nature-style muted);
  colorblind-safe; no rainbow; one muted warm accent for generated targets.
  Avoid high-chroma Wong rainbow look that reads as "AI poster".
* Sans-serif (Helvetica/Arial-equivalent Nimbus Sans) for labels; amino-acid
  sequences in a monospace one-letter code (Courier-equivalent Nimbus Mono),
  exactly as Nature specifies for sequences.
* Text sized so that AT FINAL WIDTH (\\textwidth = 5.5in over a 1000-unit
  canvas -> 1 unit approx 0.4 pt) labels land in the 5-7 pt band and panel
  letters at 8 pt bold lowercase.
* Strokes approx 0.4-1 pt at final size (approx 1-2.5 canvas units).

Output: editable vector PDF (for \\includegraphics) + PNG preview via cairosvg.
"""
from __future__ import annotations

import html
import math
import os

import cairosvg

FIGDIR = "/vepfs-mlp2/c20250601/251105016/project/dllm_test/paper/figures"

# --------------------------------------------------------- Cool SCI palette
# Inspired by Paul Tol muted + Nature-style cool categorical schemes
# (SciFig / Nature Methods guidance): low saturation, cool-biased,
# colorblind-safe, one muted warm accent only for generated targets.
INK = "#1B2838"     # cool near-black text
SUB = "#5A6A7A"     # secondary text
MUT = "#8A97A5"     # tertiary / captions
HAIR = "#C9D2DB"    # hairline rules
FAINT = "#F2F5F8"   # faint panel wash
PAPER = "#FFFFFF"

BLUE = "#3D6B99"    # steel blue — model core / structure / TCR
GREEN = "#3D8B7A"   # muted teal — ESMC / type tabs / antibody
PURPLE = "#6B7A9C"  # cool slate — LLaDA / PPI (replaces Wong pink)
ORANGE = "#5B9AAA"  # cool cyan-steel — relation / nanobody
VERM = "#C4705A"    # muted coral — generated residues / CE (sole warm note)
SKY = "#8AB0C8"     # soft sky — ESMC feature columns
YELLOW = "#A8B8C8"  # cool grey-blue


FONT = "Nimbus Sans, Helvetica, Arial, sans-serif"
MONO = "Nimbus Mono PS, Courier New, monospace"

# residue-bead hues (used sparingly; a chain reads as a sequence, not tags)
AA = [BLUE, GREEN, ORANGE, PURPLE, SKY]
MASK_FC, MASK_EC = "#EDEFF1", "#C2C9CF"


def _rgb(h):
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def tint(color, frac=0.86):
    """Lighten toward white by `frac` (0=colour, 1=white)."""
    r, g, b = _rgb(color)
    return "#%02X%02X%02X" % (round(r + (255 - r) * frac),
                              round(g + (255 - g) * frac),
                              round(b + (255 - b) * frac))


class Canvas:
    """Minimal flat-SVG builder. Origin top-left, px units."""

    def __init__(self, w, h, bg=PAPER):
        self.w, self.h = w, h
        self._els = []
        if bg:
            self.rect(0, 0, w, h, fill=bg)

    # -- primitives -------------------------------------------------------
    def rect(self, x, y, w, h, fill="none", stroke="none", sw=1.2, rx=0.0,
             dash=None, opacity=1.0):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        op = f' opacity="{opacity:g}"' if opacity != 1.0 else ""
        self._els.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'rx="{rx:.2f}" ry="{rx:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{sw:g}"{d}{op}/>')

    def circle(self, cx, cy, r, fill="none", stroke="none", sw=1.2, dash=None):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._els.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{sw:g}"{d}/>')

    def image(self, x, y, w, h, path, rx=0.0, frame=True):
        """Embed a raster image (PNG/JPEG) as a data-URI <image>."""
        import base64
        with open(path, "rb") as f:
            raw = f.read()
        ext = os.path.splitext(path)[1].lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        b64 = base64.b64encode(raw).decode("ascii")
        href = f"data:{mime};base64,{b64}"
        # optional rounded clip
        if rx > 0:
            cid = f"clip{len(self._els)}"
            self._els.append(
                f'<defs><clipPath id="{cid}">'
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
                f'rx="{rx:.2f}" ry="{rx:.2f}"/></clipPath></defs>')
            clip = f' clip-path="url(#{cid})"'
        else:
            clip = ""
        self._els.append(
            f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'href="{href}"{clip} preserveAspectRatio="xMidYMid meet"/>')
        if frame:
            self.rect(x, y, w, h, fill="none", stroke=HAIR, sw=1.0, rx=rx)

    def line(self, x0, y0, x1, y1, stroke=SUB, sw=1.2, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self._els.append(
            f'<line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="{stroke}" stroke-width="{sw:g}" stroke-linecap="{cap}"{d}/>')

    def path(self, d, fill="none", stroke="none", sw=1.2, cap="round", join="round",
             dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self._els.append(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}" '
            f'stroke-linecap="{cap}" stroke-linejoin="{join}"{da}/>')

    def text(self, x, y, s, size=15, fill=INK, weight="normal", anchor="start",
             italic=False, mono=False, spacing=None, family=None):
        st = ' font-style="italic"' if italic else ""
        ls = f' letter-spacing="{spacing}"' if spacing is not None else ""
        fam = family or (MONO if mono else FONT)
        self._els.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{fam}" '
            f'font-size="{size:g}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}"{st}{ls}>{html.escape(s)}</text>')

    def text_xt(self, x, y, prefix="noisy token ", size=11.5, fill=SUB,
                anchor="middle"):
        """Label with italic x and subscript t (e.g. 'noisy token x_t').

        Uses dy-based subscripts so cairosvg renders them reliably.
        """
        fam = FONT
        sub = max(7.5, size * 0.72)
        # Approximate centred layout: measure prefix+x, then place as one text
        # with tspans. For middle-anchor, wrap the whole phrase in one <text>.
        self._els.append(
            f'<text x="{x:.2f}" y="{y:.2f}" font-family="{fam}" '
            f'font-size="{size:g}" fill="{fill}" text-anchor="{anchor}">'
            f'{html.escape(prefix)}'
            f'<tspan font-style="italic">x</tspan>'
            f'<tspan font-style="italic" font-size="{sub:g}" '
            f'dy="{size * 0.28:.2f}">t</tspan>'
            f'<tspan dy="{-size * 0.28:.2f}"></tspan></text>')

    def arrow(self, x0, y0, x1, y1, stroke=SUB, sw=1.8, head=6.5):
        ang = math.atan2(y1 - y0, x1 - x0)
        bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
        self.line(x0, y0, bx, by, stroke=stroke, sw=sw, cap="butt")
        ox, oy = -math.sin(ang), math.cos(ang)
        pts = " ".join(f"{px:.2f},{py:.2f}" for px, py in (
            (x1, y1),
            (bx + ox * head * 0.52, by + oy * head * 0.52),
            (bx - ox * head * 0.52, by - oy * head * 0.52)))
        self._els.append(f'<polygon points="{pts}" fill="{stroke}"/>')

    # -- residue beads / sequences ---------------------------------------
    def bead(self, cx, cy, r, ci=0, filled=True, cond=False):
        if filled:
            self.circle(cx, cy, r, fill=AA[ci % len(AA)])
        else:
            self.circle(cx, cy, r, fill=MASK_FC, stroke=MASK_EC, sw=1.0, dash="2,1.5")
        if cond:
            self.circle(cx, cy, r + 1.8, stroke=ORANGE, sw=1.8)

    def aa_row(self, x, y, seq, size=13, fill=INK, dx=None):
        """Monospace one-letter amino-acid code (Nature convention)."""
        step = dx if dx is not None else size * 0.62
        for i, ch in enumerate(seq):
            self.text(x + i * step + step / 2, y, ch, size=size, fill=fill,
                      mono=True, anchor="middle")
        return x + len(seq) * step

    # -- molecular cartoons (flat, refined, biologically legible) --------
    # Every cartoon is centred on (cx, cy) and fits within roughly +/-24*s so
    # the four sit on a shared baseline. Two-tone fills (light core + saturated
    # outline) give crafted dimensionality without any drop shadow.
    def _domain(self, x, y, w, h, color, rx=None):
        self.rect(x, y, w, h, fill=tint(color, 0.55), stroke=color, sw=1.4,
                  rx=(w * 0.34 if rx is None else rx))

    def mol_antibody(self, cx, cy, s, color):
        """Symmetric IgG: Fab arms + Fc stem as segmented Ig domains, antigen
        contacting BOTH Fab tips."""
        lw = 6.0 * s
        hinge = (cx, cy + 3 * s)
        tipL, tipR = (cx - 15 * s, cy - 17 * s), (cx + 15 * s, cy - 17 * s)
        foot = (cx, cy + 19 * s)
        # tubes (rounded)
        self.line(hinge[0], hinge[1], foot[0], foot[1], stroke=color, sw=lw)
        self.line(hinge[0], hinge[1], tipL[0], tipL[1], stroke=color, sw=lw)
        self.line(hinge[0], hinge[1], tipR[0], tipR[1], stroke=color, sw=lw)
        # domain-segmentation ticks on each arm/stem
        for a, b in ((hinge, tipL), (hinge, tipR), (hinge, foot)):
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            self.circle(mx, my, lw * 0.42, fill=tint(color, 0.7))
        # variable-domain tips
        for tp in (tipL, tipR):
            self.circle(tp[0], tp[1], 4.4 * s, fill=tint(color, 0.5), stroke=color, sw=1.3)
        # antigen at both tips
        for tp in (tipL, tipR):
            self.circle(tp[0], tp[1] - 6.5 * s, 3.0 * s, fill=tint(INK, 0.5))
        self.circle(hinge[0], hinge[1], 2.4 * s, fill=color)

    def mol_nanobody(self, cx, cy, s, color):
        """Single VHH immunoglobulin domain with three CDR loops + antigen."""
        w, h = 13 * s, 23 * s
        y = cy - h / 2 + 3 * s
        self._domain(cx - w / 2, y, w, h, color, rx=w * 0.34)
        # the two beta-sheet fold hint lines
        self.line(cx - w / 2 + 3 * s, cy + 4 * s, cx + w / 2 - 3 * s, cy + 4 * s,
                  stroke=tint(color, 0.15), sw=0.8)
        # three CDR loops along the top
        for dx in (-4.2 * s, 0, 4.2 * s):
            self.path(f"M {cx+dx-2.1*s:.1f} {y:.1f} q {2.1*s:.1f} {-4.4*s:.1f} {4.2*s:.1f} 0",
                      stroke=color, sw=1.5)
        # antigen contacting the CDR face
        self.circle(cx, y - 8.5 * s, 3.8 * s, fill=tint(INK, 0.5))

    def mol_tcr_pmhc(self, cx, cy, s, color):
        """TCR Valpha/Vbeta docking on a peptide held in the MHC groove."""
        # clean MHC platform (single slab) + the epitope resting on top
        self.rect(cx - 15 * s, cy + 11 * s, 30 * s, 7 * s, fill=tint(INK, 0.68),
                  stroke=tint(INK, 0.32), sw=1.2, rx=2.5)
        self.line(cx - 8 * s, cy + 9.5 * s, cx + 8 * s, cy + 9.5 * s,
                  stroke=ORANGE, sw=3.0 * s)  # peptide (conditioning accent)
        # TCR two variable domains
        self._domain(cx - 9.5 * s, cy - 15 * s, 8.5 * s, 21 * s, color, rx=3.6 * s)
        self._domain(cx + 1.0 * s, cy - 15 * s, 8.5 * s, 21 * s, color, rx=3.6 * s)
        # CDR contact loops reaching down to the peptide
        for dx in (-5.2 * s, 5.2 * s):
            self.path(f"M {cx+dx-2.2*s:.1f} {cy+6*s:.1f} q {2.2*s:.1f} {3.0*s:.1f} {4.4*s:.1f} 0",
                      stroke=color, sw=1.4)

    def mol_ppi(self, cx, cy, s, color):
        """Two globular protein domains touching at a complementary interface:
        left carries a convex knob, right a matching concave pocket."""
        r = 11 * s
        # left domain with a bump on its right side
        self.path(
            f"M {cx-2*s:.1f} {cy-r:.1f} "
            f"a {r:.1f} {r:.1f} 0 1 0 0 {2*r:.1f} "
            f"q {5*s:.1f} {-3*s:.1f} {5*s:.1f} {-r:.1f} "
            f"q 0 {-(r-3*s):.1f} {-5*s:.1f} {-r:.1f} Z",
            fill=tint(color, 0.5), stroke=color, sw=1.4)
        # right domain with a matching notch on its left side
        self.path(
            f"M {cx+3.5*s:.1f} {cy-r:.1f} "
            f"q {-5*s:.1f} {3*s:.1f} {-5*s:.1f} {r:.1f} "
            f"q 0 {(r-3*s):.1f} {5*s:.1f} {r:.1f} "
            f"a {r:.1f} {r:.1f} 0 1 0 0 {-2*r:.1f} Z",
            fill=tint(BLUE, 0.5), stroke=BLUE, sw=1.4)

    # -- grammar token stream vocabulary (faithful to grammar-v2) --------
    def tok(self, x, ymid, label, kind, state="fixed", size=13, h=26, pad=9):
        """One grammar-stream cell. Returns the right edge x.

        kind: 'struct' (<prots>/<protd>), 'type' (<ab>/<tcr>/<nb>/<pep>),
              'rel' (<binding> ...), 'res' (amino acid), 'sep' (. separator).
        state: 'fixed' (conditioning context) | 'target' | 'mask'.
        """
        if kind == "sep":
            self.text(x + 5, ymid + 5, ".", size=size + 3, fill=SUB, mono=True,
                      anchor="middle")
            return x + 10
        if kind == "res":
            w = size + 4
            y = ymid - h / 2
            if state == "mask":
                self.rect(x, y, w, h, fill="#F5F7F8", stroke="#C2C9CF", sw=1.0,
                          rx=2, dash="2,1.6")
            elif state == "target":
                self.rect(x, y, w, h, fill=tint(VERM, 0.9), stroke=VERM, sw=1.2, rx=2)
                self.text(x + w / 2, ymid + 4.5, label, size=size, fill=VERM,
                          mono=True, anchor="middle")
            else:  # fixed context residue
                self.rect(x, y, w, h, fill=tint(INK, 0.9), stroke=tint(INK, 0.45),
                          sw=1.0, rx=2)
                self.text(x + w / 2, ymid + 4.5, label, size=size, fill=INK,
                          mono=True, anchor="middle")
            return x + w
        w = len(label) * size * 0.6 + pad
        y = ymid - h / 2
        if kind == "struct":
            self.rect(x, y, w, h, fill="#FFFFFF", stroke=BLUE, sw=1.4, rx=2)
            tc = BLUE
        elif kind == "type":
            self.rect(x, y, w, h, fill=tint(GREEN, 0.82), stroke=GREEN, sw=1.3, rx=2)
            tc = "#2A5F54"
        elif kind == "rel":
            self.rect(x, y, w, h, fill=tint(ORANGE, 0.82), stroke=ORANGE, sw=1.3, rx=h / 2)
            tc = "#3A5F6E"
        else:
            self.rect(x, y, w, h, fill=FAINT, stroke=HAIR, sw=1.0, rx=2)
            tc = INK
        self.text(x + w / 2, ymid + 4.5, label, size=size, fill=tc, mono=True,
                  anchor="middle")
        return x + w

    def featcol(self, x, ymid, w, h, seed=0, tones=None):
        """ESMC residue-feature column: 4 stacked embedding segments."""
        tones = tones or [tint(BLUE, 0.2), tint(SKY, 0.12), tint(BLUE, 0.5),
                          tint(SKY, 0.38)]
        seg = h / 4
        y = ymid - h / 2
        for k in range(4):
            self.rect(x, y + k * seg, w, seg - 0.6, fill=tones[(seed * 7 + k * 5) % 4],
                      rx=1.0)
        self.rect(x, y, w, h, stroke=tint(BLUE, 0.4), sw=0.8, rx=1.5)
        return x + w

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">'
                + "".join(self._els) + "</svg>")


def panel_letter(c, letter, x, y):
    c.text(x, y, letter, size=20, weight="bold", fill=INK)


def kicker(c, x, y, s, fill=SUB, size=14):
    """Small-caps-ish section eyebrow (editorial kicker)."""
    c.text(x, y, s.upper(), size=size, fill=fill, weight="bold", spacing=1.3)


def save(c, name):
    os.makedirs(FIGDIR, exist_ok=True)
    svg = c.render()
    with open(f"{FIGDIR}/{name}.svg", "w") as f:
        f.write(svg)
    cairosvg.svg2pdf(bytestring=svg.encode(), write_to=f"{FIGDIR}/{name}.pdf")
    cairosvg.svg2png(bytestring=svg.encode(), write_to=f"{FIGDIR}/{name}.png",
                     output_width=int(c.w * 2.2))
    print("wrote", name)


def char_w(s, size):
    return len(s) * size * 0.54
