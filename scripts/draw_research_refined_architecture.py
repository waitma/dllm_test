"""Draw the original research_refined A4/f011 exploration architecture.

Run after activating conda pllm:
    python scripts/draw_research_refined_architecture.py

Exports editable SVG, vector PDF and PNG to docs/figures. This figure describes
the historical A4/f011 configuration, not the newer fixed_chain data layout.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'figures'
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')
INK, MUTED, LINE = '#24323D', '#596A75', '#D3DDE2'
BLUE, TEAL, ORANGE, PURPLE = '#3C7396', '#368B7E', '#C98243', '#8573A3'
PALE_BLUE, PALE_TEAL, PALE_ORANGE, PALE_PURPLE = '#EDF4F8', '#EDF6F2', '#FCF2E5', '#F3EFF7'
plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def draw():
    """Keep coordinates in design units so all exports share one layout."""
    fig = plt.figure(figsize=(20, 14.1), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1800), ylim=(1269, 0), aspect='equal')
    ax.axis('off')

    def text(x, y, label, size=12, bold=False, color=INK, ha='left', va='center'):
        return ax.text(x, y, label, fontproperties=BOLD if bold else FONT,
                       fontsize=size, color=color, ha=ha, va=va, linespacing=1.55, zorder=5)

    def box(x, y, w, h, fill='white', edge=LINE, dashed=False, radius=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=fill, edgecolor=edge, linewidth=1,
                     linestyle=(0, (4, 3)) if dashed else '-', zorder=1))

    def arrow(x1, y1, x2, y2, color=MUTED, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                     mutation_scale=12, linewidth=1.25, color=color,
                     linestyle=(0, (4, 3)) if dashed else '-', shrinkA=1, shrinkB=2,
                     zorder=3))

    def path(points, color=MUTED, dashed=False):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=1.1,
                linestyle=(0, (4, 3)) if dashed else '-', zorder=2)

    def panel(x, y, w, h, letter, title):
        box(x, y, w, h)
        text(x+20, y+28, letter, 20, True)
        text(x+53, y+28, title, 16, True)

    def card(x, y, w, h, title, body, fill, edge, title_size=13, body_size=11):
        box(x, y, w, h, fill, edge)
        text(x+w/2, y+25, title, title_size, True, ha='center')
        text(x+w/2, y+h/2+17, body, body_size, ha='center')

    def chip(x, y, label, kind='structure', w=None):
        fill, edge = {'structure': ('#F2F4F5', LINE), 'context': (PALE_BLUE, BLUE),
                      'target': (PALE_TEAL, TEAL), 'mask': (PALE_ORANGE, ORANGE)}[kind]
        w = w or (86 if label.startswith('<') else 36)
        box(x, y, w, 30, fill, edge, radius=4)
        text(x+w/2, y+15, label, 10.3, ha='center')
        return x+w+7

    def plus(x, y):
        ax.add_patch(Circle((x, y), 13, edgecolor=MUTED, facecolor='white', zorder=4))
        text(x, y-1, '+', 17, ha='center')

    text(36, 43, '免疫序列掩码去噪模型', 25, True)
    text(36, 82, 'research_refined / FactorialModel  ·  原探索框架  ·  A4 数据布局，f011 配置示例', 13, color=MUTED)
    box(1307, 26, 457, 60, PALE_BLUE, BLUE)
    text(1535, 47, '8 层  ·  768 维  ·  12 个注意力头', 13, True, ha='center')
    text(1535, 72, 'FFN 3072  ·  词表 50  ·  无 ESMC', 11, ha='center')

    # a. Explicit data and supervision boundaries.
    panel(36, 116, 1728, 342, 'a', '输入：条件保持可见，目标链在固定容量内加噪')
    card(58, 174, 262, 96, '条件序列（如有）', '抗原 / peptide / MHC\n也可包含已知的受体链', PALE_BLUE, BLUE)
    card(58, 286, 262, 100, '待生成的目标链', '抗体 H / L，或 TCR β / α\n训练时有参考序列 y', PALE_TEAL, TEAL)
    arrow(322, 284, 356, 284)
    card(361, 195, 250, 172, 'A4 序列化 + 随机遮蔽',
         't ~ Uniform(0.001, 1)\n目标残基与终止符按概率 t 遮蔽\n条件序列保持可见', PALE_ORANGE, ORANGE, body_size=10.8)
    text(486, 391, 't 只控制加噪，不作为时间嵌入', 10.2, color=MUTED, ha='center')
    arrow(612, 284, 655, 284)
    text(677, 181, '送入模型的 input_ids = x_t（示意；各行实际拼接为一条序列）', 12.5, True)
    text(677, 221, '前缀', 11)
    p = chip(804, 206, '<ab>')
    chip(p, 206, '<relation>', w=107)
    text(1035, 222, '<ab> / <tcr>；关系状态可为 binding / nonbinding / unknown', 10.5, color=MUTED)
    for y, label, tokens in [
        (250, '条件链', [('<prots>', 'structure'), ('A', 'context'), ('C', 'context'), ('D', 'context'), ('…', 'context'), ('<protd>', 'structure')]),
        (294, '目标 H / β', [('<prots>', 'structure'), ('C', 'target'), ('<mask>', 'mask'), ('S', 'target'), ('…', 'target'), ('<mask>', 'mask'), ('<mask>', 'mask'), ('…', 'mask')]),
        (338, '目标 L / α', [('<prots>', 'structure'), ('<mask>', 'mask'), ('Q', 'target'), ('<mask>', 'mask'), ('…', 'target'), ('<mask>', 'mask'), ('<mask>', 'mask'), ('…', 'mask')]),
    ]:
        text(677, y+15, label, 11)
        p = 804
        for token, kind in tokens:
            p = chip(p, y, token, kind)
    text(1470, 312, '目标槽位容量\n全长 256 / 片段 64\n包含终止符位置', 10.8, color=MUTED)
    text(804, 395, '终止符为 <protd>；真实末尾之后填 <mask>，这些尾部位置不作为训练标签。', 11, color=MUTED)
    text(58, 430, '监督分支：参考 token y + eligible_mask → 仅用于选取被遮蔽位置的 loss；真实目标长度不用于裁剪模型可见区域。', 11, color=MUTED)

    # b. The actual forward graph, with metadata routed by its purpose.
    panel(36, 483, 1728, 311, 'b', '主干网络：双向 Transformer 去噪，一次预测所有位置')
    text(70, 548, 'input_ids + role_ids + scope_ids', 11, True)
    text(543, 548, 'chain_ids → 链关系偏置 / 拓扑掩码      position_ids → 链内位置 RoPE', 11, True)
    text(1372, 548, 'target_slot_mask → 词表约束', 11, True)
    arrow(220, 560, 220, 588)
    arrow(850, 560, 850, 588)
    arrow(1544, 560, 1544, 588)
    card(70, 592, 300, 122, '输入嵌入相加', 'Token + Role + Scope\n隐藏状态 h_0：[B, S, 768]', PALE_BLUE, BLUE)
    arrow(372, 654, 421, 654)
    box(427, 579, 625, 148, '#FAFCFB', TEAL)
    card(446, 601, 258, 106, '链内局部层', '同链，位置距离 ≤ 32\n第 1 / 3 / 5 / 7 层', PALE_TEAL, TEAL)
    arrow(709, 654, 749, 654)
    card(755, 601, 278, 106, '全局层', '允许跨链与条件信息交互\n第 2 / 4 / 6 / 8 层', PALE_BLUE, BLUE)
    text(740, 754, '（局部层 → 全局层）× 4；每层均含注意力与 FFN', 11, color=MUTED, ha='center')
    arrow(1054, 654, 1101, 654)
    card(1107, 592, 240, 122, '输出归一化与投影', 'RMSNorm\nLinear：768 → 50', PALE_PURPLE, PURPLE)
    arrow(1349, 654, 1396, 654)
    card(1402, 592, 329, 122, '输出 logits：[B, S, 50]', '目标槽位仅保留 26 类\n25 种残基符号 + <protd>', PALE_ORANGE, ORANGE)
    text(70, 754, '输入：[B, S]；Role = 链角色，Scope = 观测范围', 10.2, color=MUTED)
    text(1110, 754, 'attention_mask 排除 batch padding；无因果遮蔽', 11, color=MUTED)

    # c. Expand one block. The optional gate is visually distinct and explicitly off.
    panel(36, 819, 1050, 374, 'c', '单层结构：Pre-norm + 双残差')
    text(58, 897, 'h', 13, True)
    arrow(80, 897, 104, 897)
    box(110, 874, 125, 47, PALE_BLUE, BLUE)
    text(172, 897, 'RMSNorm', 11, ha='center')
    arrow(236, 897, 260, 897)
    box(266, 869, 162, 57, PALE_BLUE, BLUE)
    text(347, 897, 'Q / K / V 投影\nQ、K 加 RoPE', 10.5, ha='center')
    arrow(429, 897, 453, 897)
    box(459, 869, 205, 57, PALE_TEAL, TEAL)
    text(561, 897, '双向多头注意力\n+ 链偏置 + 拓扑掩码', 10.5, ha='center')
    arrow(665, 897, 689, 897)
    box(695, 869, 147, 57, 'white', PURPLE, dashed=True)
    text(768, 897, '可选 G 头门控\nf011：关闭', 10.5, ha='center')
    arrow(843, 897, 865, 897)
    box(871, 869, 123, 57, PALE_ORANGE, ORANGE)
    text(932, 897, '输出投影 W_o\n× α', 10.5, ha='center')
    arrow(996, 897, 1021, 897)
    plus(1038, 897)
    # Residual shortcut below the attention sublayer.
    path([(91, 897), (91, 949), (1038, 949)])
    arrow(1038, 949, 1038, 911)
    text(514, 942, '残差连接', 10, color=MUTED)
    path([(1052, 897), (1067, 897), (1067, 970), (90, 970), (90, 1007)])
    arrow(90, 1007, 110, 1007)
    box(116, 983, 160, 48, PALE_BLUE, BLUE)
    text(196, 1007, 'RMSNorm', 11, ha='center')
    arrow(277, 1007, 313, 1007)
    box(319, 978, 356, 59, PALE_TEAL, TEAL)
    text(497, 1007, 'SwiGLU FFN：768 → 3072 → 768\nWdown [SiLU(Wgate h) ⊙ Wup h]', 10.5, ha='center')
    arrow(676, 1007, 712, 1007)
    box(718, 983, 116, 48, PALE_ORANGE, ORANGE)
    text(776, 1007, '× α', 12, ha='center')
    arrow(835, 1007, 911, 1007)
    plus(928, 1007)
    arrow(942, 1007, 1031, 1007)
    text(1048, 1007, 'h′', 13, True)
    path([(100, 1007), (100, 1060), (928, 1060)])
    arrow(928, 1060, 928, 1021)
    text(499, 1054, '残差连接', 10, color=MUTED)
    box(58, 1090, 1006, 80, '#F8FAFB', LINE)
    text(77, 1113, 'fGSL 探索开关', 12, True)
    text(286, 1113, 'G：逐 token / 逐 head 输出门控', 10.5)
    text(286, 1146, 'S：两条残差分支固定乘 α = 0.1', 10.5)
    text(692, 1113, 'L：链内局部 / 全局交替', 10.5)
    text(692, 1146, '本图 f011：G 关，S / L 开', 10.5, True)

    # d. Training and inference consume the same denoiser outputs differently.
    panel(1111, 819, 653, 374, 'd', '输出：训练目标与序列生成')
    box(1133, 871, 579, 104, PALE_ORANGE, ORANGE)
    text(1152, 895, '训练 → 标量 loss', 13, True)
    text(1152, 924, 'logits + 参考 y + 本轮遮蔽集合 M → 平均交叉熵', 11)
    text(1152, 952, 'M 仅含可监督的被遮蔽 token；含残基与 <protd>', 10.5)
    box(1133, 990, 579, 151, PALE_TEAL, TEAL)
    text(1152, 1016, '生成 → 一条或多条受体氨基酸序列', 13, True)
    text(1152, 1047, '目标槽位全 <mask> → 迭代预测 / 重遮蔽', 11)
    text(1152, 1077, '按置信度逐步揭示；每轮复用同一去噪网络', 11)
    text(1152, 1107, '按各链首个 <protd> 截取序列，并返回链状态', 11)
    text(1135, 1168, '可选表征：最终 hidden states [B, S, 768]（投影前）', 10.5, color=MUTED)
    # Clear output branch from logits to the two uses.
    path([(1567, 716), (1567, 776), (1749, 776), (1749, 1065)])
    arrow(1749, 923, 1714, 923)
    arrow(1749, 1065, 1714, 1065)
    # Both boxes explicitly name how they consume the same logits / denoiser.

    text(38, 1222, '图例', 10.5, True)
    for x, fill, edge, label in [(94, PALE_BLUE, BLUE, '条件 / 嵌入'), (270, PALE_TEAL, TEAL, '目标 / 链内模块'),
                                  (485, PALE_ORANGE, ORANGE, 'mask / 预测'), (680, PALE_PURPLE, PURPLE, '输出投影')]:
        box(x, 1213, 22, 18, fill, edge, radius=3)
        text(x+32, 1222, label, 10.5)
    text(1745, 1222, 'B = batch size；S = 拼接后的序列长度', 10.5, color=MUTED, ha='right')
    text(38, 1251, '注：示例省略缺失链与 padding；关系 token 在明确配置监督时也参与 loss。局部层中前缀组保持独立。容量参数来自原 A4 配置。', 9.5, color=MUTED)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / 'research_refined_architecture'
    for suffix in ('svg', 'pdf'):
        fig.savefig(stem.with_suffix('.'+suffix), facecolor='white')
    fig.savefig(stem.with_suffix('.png'), dpi=160, facecolor='white')
    fig.savefig(OUT / 'research_refined_architecture_hires.png', dpi=300, facecolor='white')
    plt.close(fig)
    print(stem)


if __name__ == '__main__':
    draw()
