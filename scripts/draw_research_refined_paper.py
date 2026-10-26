"""Draw the English paper-style A4/f011 architecture (original research_refined).

Run after activating conda pllm:
    python scripts/draw_research_refined_paper.py

Sequence strips and a compact network inset follow the visual conventions of
DPLM (ICML 2024) and EvoDiff; all diagram elements are drawn from scratch.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
INK, GRAY, LIGHT = '#252B30', '#72797D', '#D7DCDE'
BLUE, GREEN, OCHRE = '#6E9CB6', '#77A998', '#D0AA6A'
PALE_BLUE, PALE_GREEN, PALE_OCHRE = '#E6EFF4', '#E8F1EC', '#F5EDD9'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
    'ps.fonttype': 42, 'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans',
})


def draw(panel='all'):
    height = 440 if panel == 'a' else 800
    fig = plt.figure(figsize=(14.4, height / 100), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1440), ylim=(height, 0), aspect='equal')
    ax.axis('off')

    def text(x, y, s, fs=9, ha='left', bold=False, color=INK, **kwargs):
        ax.text(x, y, s, fontsize=fs, ha=ha, va='center', color=color,
                weight='bold' if bold else 'normal', linespacing=1.4, zorder=6, **kwargs)

    def rect(x, y, w, h, fill='white', edge=GRAY, lw=.8, dashed=False):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                     linewidth=lw, linestyle=(0, (3, 3)) if dashed else '-', zorder=2))

    def line(points, color=GRAY, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=.85,
                linestyle=(0, (3, 3)) if dashed else '-', zorder=1)

    def arrow(a, b, color=INK, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=9,
                     lw=.9, color=color, shrinkA=1, shrinkB=1,
                     linestyle=(0, (3, 3)) if dashed else '-', zorder=4))

    def module(x, y, w, h, label, fill='white', fs=9):
        rect(x, y, w, h, fill)
        text(x+w/2, y+h/2, label, fs, ha='center')

    def sequence(x, y, s, color, cell=23):
        for i, letter in enumerate(s):
            masked, terminal = letter == '*', letter == '$'
            fill = '#F3F4F4' if masked else color
            rect(x+i*cell, y, cell, cell, fill, 'white', .75)
            glyph = '×' if masked else ('end' if terminal else letter)
            text(x+(i+.5)*cell, y+cell/2, glyph,
                 5.8 if terminal else 8, ha='center',
                 color=GRAY if masked else ('white' if terminal else INK))
            if terminal:
                rect(x+i*cell, y, cell, cell, 'none', INK, 1)
        return x+len(s)*cell

    def plus(x, y):
        ax.add_patch(Circle((x, y), 9, facecolor='white', edgecolor=GRAY, lw=.8, zorder=4))
        text(x, y-.4, '+', 11, ha='center')

    # a. Receptor-specific inputs, shared denoiser and decoded outputs.
    text(24, 28, 'a', 14, bold=True)
    text(47, 28, 'Conditional immune sequence generation', 11, bold=True)
    text(26, 83, 'Input sequences', 10, bold=True)
    text(133, 114, '<ab> / <tcr>  ·  relation token', 8, color=GRAY)
    text(26, 155, 'Context', 9)
    sequence(133, 143, 'ACDFGKLNPQ$', BLUE)
    text(133, 183, 'Antigen, peptide–MHC or a known receptor chain', 7.7, color=GRAY)
    text(26, 231, r'H / $\beta$', 9)
    sequence(133, 219, 'C*S**G**Y**', GREEN)
    text(26, 279, r'L / $\alpha$', 9)
    sequence(133, 267, '*Q*S*******', OCHRE)
    text(133, 311, 'Masked target slots', 8.5)
    text(133, 332, '256 / chain (full) · 64 / chain (fragment)', 7.5, color=GRAY)
    line([(404, 143), (416, 143), (416, 290), (404, 290)])
    line([(416, 216), (451, 216), (451, 331)])
    arrow((451, 331), (565, 331))
    text(456, 310, 'Serialize', 7.5)

    text(689, 82, 'Denoising Transformer', 10, ha='center', bold=True)
    module(570, 309, 240, 44, r'$E_{\mathrm{token}} + E_{\mathrm{role}} + E_{\mathrm{scope}}$', PALE_BLUE, 10)
    arrow((690, 308), (690, 289))
    # Stacked outlines indicate repeated local/global layer pairs.
    rect(578, 178, 239, 115, '#FAFBFB', LIGHT)
    rect(574, 182, 239, 115, '#FAFBFB', LIGHT)
    rect(570, 186, 239, 115, '#FAFBFB', GRAY)
    module(581, 251, 217, 38, 'Chain-local Transformer', PALE_GREEN, 8.5)
    arrow((690, 249), (690, 238))
    module(581, 198, 217, 38, 'Global Transformer', PALE_BLUE, 8.5)
    text(840, 243, '× 4', 10, ha='center')
    arrow((690, 185), (690, 166))
    module(570, 122, 240, 44, 'RMSNorm + token head', '#F5F5F5', 9)
    text(689, 373, '8 layers · 768 hidden · 12 heads', 8, ha='center', color=GRAY)
    text(689, 393, 'Role: chain type   Scope: observed region', 7.3, ha='center', color=GRAY)

    arrow((811, 144), (1028, 144))
    text(920, 122, r'$p_\theta(x_0\,|\,x_t,c)$', 10, ha='center')
    text(920, 167, 'Token predictions', 7.8, ha='center')
    text(1047, 83, 'Generated receptor sequences', 10, bold=True)
    text(1105, 174, r'H / $\beta$', 9)
    sequence(1105, 188, 'CASSLGQAY$', GREEN)
    text(1105, 226, r'L / $\alpha$', 9)
    sequence(1105, 240, 'QQYSTP$', OCHRE)
    text(1105, 288, 'Sequence ends at the first <protd>', 7.8, color=GRAY)
    line([(1032, 144), (1032, 263)])
    arrow((1032, 200), (1095, 200))
    arrow((1032, 252), (1095, 252))
    # A return path makes iterative decoding explicit, rather than a one-pass output.
    line([(965, 144), (965, 420), (451, 420), (451, 347)])
    arrow((451, 347), (451, 332))
    rect(583, 409, 250, 21, 'white', 'white', 0)
    text(708, 420, 'Iterative unmasking / remasking', 8.5, ha='center')
    text(1180, 354, 'Context remains visible', 8, ha='center', color=GRAY)
    rect(1054, 384, 15, 15, '#F3F4F4', LIGHT)
    text(1061.5, 391.5, '×', 8, ha='center', color=GRAY)
    text(1077, 392, '<mask>', 7.8)
    rect(1189, 384, 25, 15, GREEN, INK)
    text(1201.5, 391.5, 'end', 5.8, ha='center', color='white')
    text(1222, 392, '<protd>', 7.8)

    if panel == 'all':
        # b. One block; only mechanisms active in f011 are shown.
        text(24, 479, 'b', 14, bold=True)
        text(47, 479, 'Transformer block', 11, bold=True)
        text(32, 563, r'$h_\ell$', 11)
        arrow((61, 560), (93, 560))
        module(98, 539, 115, 42, 'RMSNorm', '#F5F5F5', 8.5)
        arrow((215, 560), (241, 560))
        module(246, 530, 219, 60, 'Q, K, V projections\nRoPE on Q and K', PALE_BLUE, 8.5)
        arrow((467, 560), (493, 560))
        module(498, 530, 214, 60, 'Bidirectional attention\nChain bias + attention mask', PALE_GREEN, 8)
        arrow((714, 560), (739, 560))
        module(744, 539, 85, 42, r'$W_O\,\times\,0.1$', PALE_OCHRE, 9)
        arrow((831, 560), (851, 560))
        plus(863, 560)
        line([(76, 560), (76, 613), (863, 613)])
        arrow((863, 613), (863, 571))
        line([(874, 560), (895, 560), (895, 638), (76, 638), (76, 686)])
        arrow((76, 686), (93, 686))
        module(98, 665, 115, 42, 'RMSNorm', '#F5F5F5', 8.5)
        arrow((215, 686), (241, 686))
        module(246, 656, 298, 60, 'SwiGLU feed-forward network\n768 → 3072 → 768', PALE_GREEN, 8.5)
        arrow((546, 686), (575, 686))
        module(580, 665, 97, 42, '× 0.1', PALE_OCHRE, 9)
        arrow((679, 686), (722, 686))
        plus(735, 686)
        line([(85, 686), (85, 740), (735, 740)])
        arrow((735, 740), (735, 697))
        arrow((746, 686), (841, 686))
        text(852, 687, r'$h_{\ell+1}$', 11)

        # c. Actual attention topology is expressed visually, not as prose cards.
        text(952, 479, 'c', 14, bold=True)
        text(975, 479, 'Attention topology', 11, bold=True)

        def matrix(x, y, local):
            n, cell = 12, 13
            for i in range(n):
                for j in range(n):
                    same = i//4 == j//4
                    allowed = (same and abs(i-j) <= 1) if local else True
                    color = (BLUE, GREEN, OCHRE)[i//4] if same else '#DCE5E8'
                    rect(x+j*cell, y+i*cell, cell, cell, color if allowed else 'white',
                         '#F1F3F3', .35)
            rect(x, y, n*cell, n*cell, 'none', GRAY, .7)
            for k, color in enumerate((BLUE, GREEN, OCHRE)):
                rect(x+k*4*cell, y-8, 4*cell, 4, color, color, 0)
                rect(x-8, y+k*4*cell, 4, 4*cell, color, color, 0)
            for k, label in enumerate(('C', 'H/β', 'L/α')):
                text(x+(k*4+2)*cell, y+169, label, 7, ha='center')

        matrix(981, 541, True)
        matrix(1230, 541, False)
        arrow((1156, 618), (1210, 618))
        text(1059, 515, 'Chain-local', 9, ha='center')
        text(1308, 515, 'Global', 9, ha='center')
        text(1059, 735, 'Odd layers · radius 32', 8, ha='center', color=GRAY)
        text(1308, 735, 'Even layers · all chains', 8, ha='center', color=GRAY)
        text(1180, 771, 'Schematic masks; C denotes context', 7.5, ha='center', color=GRAY)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / ('research_refined_architecture_a' if panel == 'a'
                  else 'research_refined_architecture_paper')
    for ext in ('svg', 'pdf'):
        fig.savefig(stem.with_suffix('.'+ext), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(stem.with_name(stem.name + '_hires.png'), dpi=400, facecolor='white')
    stem.with_suffix('.txt').write_text(
        'Conditional immune sequence generation. '
        '(a) Context sequences and masked receptor-chain slots are serialized with '
        'family and relation tokens. Token, role and scope embeddings feed an '
        'eight-layer bidirectional Transformer. Chain-local and global layers '
        'alternate. Iterative token prediction and remasking produce receptor '
        'sequences, each truncated at its first <protd> token. '
        '(b) Each pre-norm block uses rotary positions, learned same-chain versus '
        'other-chain attention biases, and a SwiGLU feed-forward network. Both '
        'residual branches are scaled by 0.1. '
        '(c) Local attention is restricted to the same chain and a position radius '
        'of 32; global attention permits interactions across all valid tokens. '
        'Matrices are schematic, with the prefix group and padding omitted. '
        'Sequence letters illustrate the interface and are not experimental samples. '
        'The figure shows the historical research_refined FactorialModel A4/f011 '
        'configuration, with head gating disabled. It does not describe the later '
        'fixed_chain configuration.\n\n'
        'Visual references (new vector artwork; no copied figure assets):\n'
        'DPLM, ICML 2024, Figure 1: https://proceedings.mlr.press/v235/wang24ct.html\n'
        'EvoDiff, Figure 1: https://www.biorxiv.org/content/10.1101/2023.09.11.556673v2\n'
    )
    if panel == 'a':
        caption_path = stem.with_suffix('.txt')
        caption = caption_path.read_text()
        caption_path.write_text(
            caption.split('(b)')[0]
            + 'Shown: original research_refined A4/f011 configuration. '
            + 'Sequence letters are schematic, not experimental samples.\n\n'
            + 'Visual references' + caption.split('Visual references', 1)[1])
    plt.close(fig)
    print(stem)
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel', choices=('all', 'a'), default='all')
    draw(parser.parse_args().panel)
