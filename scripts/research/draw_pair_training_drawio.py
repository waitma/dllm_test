"""Export a simple, editable baseline/pair architecture comparison.

Activate conda pllm, then run:
  python -B /project_c20250601/251105016/project/dllm_test/scripts/research/draw_pair_training_drawio.py

Writes four native draw.io pages and matching PNG/PDF previews. This describes
the shared-pair ESMFold2 / intra-chain KD experiment, without loading models.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path("/project_c20250601/251105016/project/dllm_test")
OUT = ROOT / "docs/figures"
STEM = "pair_training_simple_20261025_v2"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
COLORS = {
    "blue": ("#EAF2FC", "#3766A1"),
    "orange": ("#FFF0DF", "#BC681C"),
    "green": ("#E8F5EF", "#26816A"),
    "purple": ("#F2EBFA", "#8061A4"),
    "gray": ("#F4F6F8", "#8B99A8"),
}
INK = "#23364B"
W, H = 1800, 1190


class Page:
    def __init__(self, name, title, subtitle):
        self.name, self.nodes, self.edges = name, {}, []
        self.node("title", 50, 22, 1700, 55, title, size=33, text=True, bold=True)
        self.node("subtitle", 50, 85, 1700, 38, subtitle, size=20, text=True)

    def node(self, key, x, y, w, h, label, color="blue", size=23,
             text=False, bold=False, panel=False):
        self.nodes[key] = dict(x=x, y=y, w=w, h=h, label=label, color=color,
                               size=size, text=text, bold=bold, panel=panel)

    def edge(self, source, target, color="blue", start=(.5, 1), end=(.5, 0),
             points=(), dashed=False):
        self.edges.append(dict(source=source, target=target, color=color,
                               start=start, end=end, points=points, dashed=dashed))

    def export(self, mxfile, pdf):
        diagram = ET.SubElement(mxfile, "diagram", id=self.name, name=self.name)
        model = ET.SubElement(diagram, "mxGraphModel", grid="1", gridSize="10",
                              guides="1", connect="1", arrows="1", page="1",
                              pageScale="1", pageWidth=str(W), pageHeight=str(H),
                              math="0", shadow="0", background="#FFFFFF")
        root = ET.SubElement(model, "root")
        ET.SubElement(root, "mxCell", id="0")
        ET.SubElement(root, "mxCell", id="1", parent="0")
        fig = plt.figure(figsize=(18, 11.9), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set(xlim=(0, W), ylim=(H, 0), aspect="equal")
        ax.axis("off")
        texts = []
        for key, n in self.nodes.items():
            fill, stroke = COLORS[n["color"]]
            style = ("rounded=1;arcSize=12;whiteSpace=wrap;html=0;"
                     f"fillColor={fill};strokeColor={stroke};strokeWidth=1.5;"
                     f"fontFamily=Noto Sans CJK SC;fontSize={n['size']};"
                     f"fontColor={INK};fontStyle={int(n['bold'])};"
                     "align=center;verticalAlign=middle;spacing=10;")
            if n["text"]:
                style += "shape=text;strokeColor=none;fillColor=none;"
            if n["panel"]:
                style += "fillColor=none;dashed=1;verticalAlign=top;spacingTop=12;"
            cell = ET.SubElement(root, "mxCell", id=key, value=n["label"],
                                 style=style, vertex="1", parent="1")
            ET.SubElement(cell, "mxGeometry", x=str(n["x"]), y=str(n["y"]),
                          width=str(n["w"]), height=str(n["h"]), attrib={"as": "geometry"})
            if not n["text"]:
                ax.add_patch(FancyBboxPatch((n["x"], n["y"]), n["w"], n["h"],
                    boxstyle="round,pad=0,rounding_size=10", linewidth=1.3,
                    facecolor="none" if n["panel"] else fill, edgecolor=stroke,
                    linestyle="--" if n["panel"] else "-", zorder=1))
            artist = ax.text(n["x"] + n["w"]/2,
                n["y"] + (27 if n["panel"] else n["h"]/2), n["label"],
                ha="center", va="center", fontproperties=FONT, fontsize=n["size"]*.72,
                fontweight="bold" if n["bold"] else "normal", color=INK,
                linespacing=1.5, zorder=4)
            texts.append((key, n, artist))
        for i, e in enumerate(self.edges):
            stroke = COLORS[e["color"]][1]
            sx, sy = e["start"]
            tx, ty = e["end"]
            style = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=0;"
                     f"strokeColor={stroke};strokeWidth=2;endArrow=block;endFill=1;"
                     f"exitX={sx};exitY={sy};exitDx=0;exitDy=0;"
                     f"entryX={tx};entryY={ty};entryDx=0;entryDy=0;"
                     f"dashed={int(e['dashed'])};")
            cell = ET.SubElement(root, "mxCell", id=f"edge_{i}", edge="1", parent="1",
                                 source=e["source"], target=e["target"], style=style)
            geo = ET.SubElement(cell, "mxGeometry", relative="1", attrib={"as": "geometry"})
            if e["points"]:
                arr = ET.SubElement(geo, "Array", attrib={"as": "points"})
                for x, y in e["points"]:
                    ET.SubElement(arr, "mxPoint", x=str(x), y=str(y))
            a, b = self.nodes[e["source"]], self.nodes[e["target"]]
            points = [(a["x"]+a["w"]*sx, a["y"]+a["h"]*sy), *e["points"],
                      (b["x"]+b["w"]*tx, b["y"]+b["h"]*ty)]
            if len(points) > 2:
                ax.plot(*zip(*points[:-1]), color=stroke, linewidth=1.5,
                        linestyle="--" if e["dashed"] else "-", zorder=2)
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                mutation_scale=14, linewidth=1.5, color=stroke,
                linestyle="--" if e["dashed"] else "-", shrinkA=1, shrinkB=2, zorder=2))
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for key, n, artist in texts:
            bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
            assert bounds.x0 >= n["x"]+3 and bounds.x1 <= n["x"]+n["w"]-3, key
            assert min(bounds.y0, bounds.y1) >= n["y"]+2, key
            assert max(bounds.y0, bounds.y1) <= n["y"]+n["h"]-2, key
        fig.savefig(OUT / f"{STEM}_{self.name}.png", dpi=160)
        pdf.savefig(fig)
        plt.close(fig)


def overview():
    p = Page("01_overview", "原模型 → 加入 Pair Feature 的训练架构",
             "ESMFold2 蒸馏版：8 层共享 Pair Encoder，结构监督使用链内距离分布")
    p.node("base_title", 60, 140, 440, 48, "原模型：双向 BERT", text=True, bold=True, size=27)
    p.node("new_title", 620, 140, 600, 48, "加入 Pair Feature 的学生", text=True, bold=True, size=27)
    p.node("teacher_title", 1320, 140, 420, 48, "训练时的结构教师", text=True, bold=True, size=27)
    for prefix, x, width in [("b", 60, 440), ("s", 620, 600)]:
        p.node(prefix+"_input", x, 210, width, 70, "序列 + MLM 随机遮盖\n链角色 / 链内位置", size=22)
        p.node(prefix+"_embed", x, 320, width, 85, "输入嵌入（768 维）\nToken + Role + Scope*", size=24)
        p.node(prefix+"_stack", x, 445, width, 225, "8 层 Transformer = 下列两层 × 4", color="gray", size=22, panel=True)
        suffix = "  + Pair Bias" if prefix == "s" else ""
        p.node(prefix+"_odd", x+20, 500, width-40, 56, "奇数层：链内 Attention"+suffix+" → FFN",
               size=21, color="orange" if prefix == "s" else "blue")
        p.node(prefix+"_even", x+20, 590, width-40, 56, "偶数层：全局 Attention"+suffix+" → FFN",
               size=21, color="orange" if prefix == "s" else "blue")
        p.node(prefix+"_norm", x, 710, width, 60, "最终 RMSNorm → 序列表征 h", size=24)
        for a, b in [("_input", "_embed"), ("_embed", "_stack"), ("_odd", "_even"), ("_even", "_norm")]:
            p.edge(prefix+a, prefix+b)
    p.node("b_head", 60, 815, 440, 70, "Token 预测头\n768 → 50 类", size=24)
    p.node("b_loss", 60, 940, 440, 70, "MLM 损失：恢复原 token", color="purple", size=24)
    p.edge("b_norm", "b_head")
    p.edge("b_head", "b_loss", "purple")
    p.node("b_note", 60, 1045, 440, 110, "主干：8 层 / 12 个头 / 768 维\nFFN：3072 维，双向注意力\nScope*：受体槽位的 scope_id 固定为 0", text=True, size=20)
    p.node("s_head", 620, 815, 240, 70, "原 Token 预测头\n768 → 50 类", size=23)
    p.node("s_pairhead", 930, 810, 290, 90, "共享 Pair Encoder\n+ 距离预测头（64 类）", color="orange", size=22)
    p.edge("s_norm", "s_head", start=(.2, 1), points=[(740, 790)])
    p.edge("s_norm", "s_pairhead", "orange", start=(.758333, 1), points=[(1075, 790)])
    p.node("s_mlm", 620, 940, 240, 70, "MLM 损失", color="purple", size=24)
    p.node("s_kd", 930, 940, 290, 70, "结构蒸馏损失\nKL（教师 ‖ 学生）", color="purple", size=23)
    p.edge("s_head", "s_mlm", "purple")
    p.edge("s_pairhead", "s_kd", "purple")
    p.node("t_input", 1320, 210, 420, 70, "同一样本的干净序列", color="green", size=24)
    p.node("t_model", 1320, 320, 420, 85, "ESMFold2-Fast\n冻结参数", color="green", size=24)
    p.node("t_dist", 1320, 500, 420, 110, "教师链内距离分布（64 类）\n按残基位置对齐学生", color="green", size=24)
    p.edge("t_input", "t_model", "green")
    p.edge("t_model", "t_dist", "green")
    p.edge("t_dist", "s_kd", "green", start=(1, .5), end=(1, .5),
           points=[(1765, 555), (1765, 975)])
    p.node("t_note", 1310, 690, 440, 100, "教师输出作为 loss 目标\n学生 Pair Feature 由自身 h 生成", text=True, size=21)
    p.node("total", 620, 1060, 600, 65, "联合训练：L = L_MLM + λ · L_KD", color="purple", size=26)
    p.edge("s_mlm", "total", "purple", end=(.2, 0))
    p.edge("s_kd", "total", "purple", end=(.758333, 0))
    p.node("train_note", 620, 1135, 600, 36, "反向传播：更新学生主干、Pair Encoder 和预测头", text=True, size=20)
    p.node("legend", 1310, 1070, 450, 70, "蓝：原有模块  橙：Pair 相关\n绿：冻结教师  紫：训练损失", text=True, size=21)
    return p


def block_detail():
    p = Page("02_block", "单层展开：原有主干与新增 Pair 分支",
             "蓝色为原模型的完整单层路径；橙色分支在每层生成 Pair Bias，并加入 Attention 分数")
    p.node("main_panel", 60, 145, 820, 955, "原模型单层（重复 8 次）", color="gray", panel=True, size=25)
    p.node("pair_panel", 1010, 145, 730, 650, "Pair 分支：Encoder 共享，Bias 投影独立", color="orange", panel=True, size=25)
    p.node("h", 210, 215, 520, 65, "当前层输入 h（768 维）", size=26)
    p.node("norm", 210, 320, 520, 55, "RMSNorm", size=25)
    p.node("qkv", 210, 415, 520, 75, "Q / K / V 投影 + 链内 RoPE\n12 个头，每头 64 维", size=23)
    p.node("attn", 170, 535, 600, 115, "Attention：分数 + Bias → softmax → 加权 V\n原有：QK^T / √64 + 链关系 Bias + Mask\n新增：+ Pair Bias", size=22)
    p.node("add1", 210, 695, 520, 55, "残差相加：h′ = h + 0.1 × 输出投影", size=22)
    p.node("ffn", 210, 800, 520, 90, "RMSNorm → SwiGLU FFN\n768 → 3072 → 768", size=25)
    p.node("add2", 210, 935, 520, 55, "残差相加：h_next = h′ + 0.1 × FFN", size=22)
    p.node("next", 210, 1030, 520, 55, "传入下一层的 h_next（768 维）", size=24)
    for a, b in [("h", "norm"), ("norm", "qkv"), ("qkv", "attn"), ("attn", "add1"),
                 ("add1", "ffn"), ("ffn", "add2"), ("add2", "next")]:
        p.edge(a, b)
    p.edge("h", "add1", start=(0, .5), end=(0, .5), points=[(110, 247.5), (110, 722.5)])
    p.edge("add1", "add2", start=(1, .5), end=(1, .5), points=[(820, 722.5), (820, 962.5)])
    p.node("proj", 1060, 215, 630, 65, "LayerNorm + 单位置投影：768 → 64", color="orange", size=25)
    p.node("features", 1060, 325, 630, 130, "组合两个位置的特征\nu_i + u_j   /   |u_i − u_j|   /   u_i ⊙ u_j\n+ 相对位置 + 链 / 角色关系", color="orange", size=25)
    p.node("pair", 1060, 500, 630, 95, "Pair MLP：320 → 128 → 64\n生成对称 Pair Feature z_ij", color="orange", size=25)
    p.node("bias", 1060, 650, 630, 90, "本层独立投影：64 → 12\n每个注意力头一个 Pair Bias（零初始化）", color="orange", size=24)
    p.edge("h", "proj", "orange", start=(1, .5), end=(0, .5))
    for a, b in [("proj", "features"), ("features", "pair"), ("pair", "bias")]:
        p.edge(a, b, "orange")
    p.edge("bias", "attn", "orange", start=(0, .5), end=(1, .5),
           points=[(940, 695), (940, 592.5)])
    p.node("notes", 1020, 840, 710, 130,
           "每层读取当前 h，重新计算 Pair Feature\n跨层传递的是序列表征 h\n奇数层链内可见，偶数层全局可见", text=True, size=25)
    p.node("readout", 1020, 1000, 710, 100,
           "第 8 层后：最终 RMSNorm → 共享 Pair Encoder\n→ 64 类距离头 → 与教师距离分布计算 KL", color="purple", size=23)
    p.node("footer", 60, 1130, 1680, 42,
           "原模型即左侧完整路径；加入 Pair 后，仍按原链内 / 全局 Mask 限制连接。教师目标只用于训练损失。",
           text=True, size=22)
    return p


def input_detail():
    p = Page("03_inputs", "Token、Role、Scope 如何形成序列特征 h",
             "每个 token 位置有一个 768 维向量；S 是位置数，图中省略 batch 维度")
    columns = [
        ("token", 60, "Token：当前位置的符号\n例如 A、G、[MASK]、结束符", "E_token：查 Token embedding 表"),
        ("role", 650, "Role：当前位置的链类型\n例如重链、轻链、TCR α / β 链", "E_role：查 Role embedding 表"),
        ("scope", 1240, "Scope：通用接口中的观测范围\nfull / fragment / unknown 等\n本版受体槽位 scope_id = 0", "E_scope：查 Scope embedding 表"),
    ]
    for key, x, label, embed in columns:
        p.node(key, x, 165, 500, 110, label, size=22 if key == "scope" else 25)
        p.node(key+"_emb", x, 345, 500, 80, embed+"\n输出 [S, 768]", size=23)
        p.edge(key, key+"_emb")
    p.node("sum", 460, 495, 880, 90,
           "逐位置相加：h0 = E_token + E_role + E_scope\n得到初始序列表征 h0：[S, 768]", size=27)
    p.edge("token_emb", "sum", start=(.5, 1), end=(.15, 0), points=[(310, 460), (592, 460)])
    p.edge("role_emb", "sum")
    p.edge("scope_emb", "sum", start=(.5, 1), end=(.85, 0), points=[(1490, 460), (1208, 460)])
    p.node("layer", 460, 635, 880, 80,
           "第 1 层读取 h0；后续层读取上一层输出\n每层的 h 都包含 S 个位置向量", size=26)
    p.edge("sum", "layer")
    p.node("h_rows", 60, 785, 500, 150,
           "当前层 h：[S, 768]\nh_1：第 1 个位置的 768 维特征\nh_2：第 2 个位置的 768 维特征\n… h_S：第 S 个位置的特征", size=23)
    p.edge("layer", "h_rows", start=(.15, 1), points=[(592, 750), (310, 750)])
    p.node("metadata", 650, 770, 500, 65, "位置 / chain_id / role_id", color="gray", size=26)
    p.node("encoder", 650, 895, 500, 100,
           "Pair Encoder\n在内部选取 h_i、h_j 并构造成对特征", color="orange", size=23)
    p.edge("metadata", "encoder", "orange")
    p.edge("h_rows", "encoder", "orange", start=(1, .75), end=(0, .5),
           points=[(605, 897.5), (605, 945)])
    p.node("pair_out", 1240, 895, 500, 100,
           "输出 Pair Feature z_ij：64 维\n所有位置对：[S, S, 64]（逻辑形状）", color="orange", size=24)
    p.edge("encoder", "pair_out", "orange", start=(1, .5), end=(0, .5))
    p.node("input_note", 60, 1055, 1680, 90,
           "h 本身就是逐位置 feature；Pair Encoder 内部把单位置特征组合成位置对特征。\nrole_id 表示链的类型；chain_id 区分具体哪一条链。计算可分块进行，不要求一次存下完整 Pair 张量。",
           text=True, size=23)
    return p


def pair_and_sharing():
    p = Page("04_pair_sharing", "Pair Feature → Pair Bias，以及参数如何共享",
             "左侧展开一个位置对的计算；右侧展示不同层重复调用同一个 Pair Encoder")
    p.node("encoder_panel", 60, 150, 970, 790,
           "Pair Encoder 内部：以下参数跨层共享", color="orange", panel=True, size=25)
    p.node("h", 80, 220, 500, 65, "当前层 h_i、h_j：各 768 维", size=25)
    p.node("meta", 640, 220, 370, 65, "position / chain / role 信息", color="gray", size=23)
    p.node("proj", 80, 335, 500, 80,
           "同一 LayerNorm + Linear（768 → 64）\n得到 u_i、u_j：各 64 维", color="orange", size=23)
    p.edge("h", "proj", "orange")
    features = [
        ("sum_feature", 80, 155, "u_i + u_j\n64 维"),
        ("diff_feature", 255, 155, "|u_i − u_j|\n64 维"),
        ("prod_feature", 430, 155, "u_i ⊙ u_j\n64 维"),
        ("pos_feature", 640, 170, "相对位置\nEmbedding\n64 维"),
        ("role_feature", 840, 170, "同 / 异链\n+ 角色组合\nEmbedding 64 维"),
    ]
    for key, x, width, label in features:
        p.node(key, x, 485, width, 95, label, color="orange", size=20)
        if x < 600:
            p.edge("proj", key, "orange", points=[(330, 450), (x+width/2, 450)])
        else:
            p.edge("meta", key, "orange", points=[(825, 450), (x+width/2, 450)])
    p.node("cat", 80, 650, 930, 65, "Concat：拼接 5 组 64 维特征 → 320 维", color="orange", size=27)
    for key, x, width, _ in features:
        center = x+width/2
        p.edge(key, "cat", "orange", end=((center-80)/930, 0))
    p.node("mlp", 80, 755, 930, 65,
           "Linear 320 → 128 → SiLU → Linear 128 → 64 → LayerNorm", color="orange", size=23)
    p.node("z", 80, 860, 930, 60, "Pair Feature z_ij：64 维（对称：z_ij = z_ji）", color="orange", size=26)
    p.edge("cat", "mlp", "orange")
    p.edge("mlp", "z", "orange")
    p.node("bias", 80, 990, 430, 95,
           "本层 W_bias：64 → 12\n每个头得到一个 bias 标量", color="orange", size=24)
    p.node("attn", 580, 990, 430, 95,
           "加入 Attention 分数\n再 softmax → 加权 V", size=25)
    p.edge("z", "bias", "orange", start=(.231183, 1))
    p.edge("bias", "attn", "orange", start=(1, .5), end=(0, .5))
    p.node("sharing_title", 1090, 150, 660, 80,
           "共享的是参数 θ\n每次输入 h 不同，输出 z 也不同", text=True, size=27)
    for key, y, h, z, output in [
        ("one", 275, "第 1 层 h0", "z1", "第 1 层 W1\n64 → 12"),
        ("two", 440, "第 2 层 h1", "z2", "第 2 层 W2\n64 → 12"),
        ("eight", 675, "第 8 层 h7", "z8", "第 8 层 W8\n64 → 12"),
        ("final", 930, "最终 h_final", "z_final", "距离预测头\n64 → 64 类"),
    ]:
        p.node(key+"_h", 1090, y, 165, 85, h, size=22)
        p.node(key+"_enc", 1300, y, 240, 85,
               "同一个 Encoder θ\n→ "+z, color="orange", size=22)
        p.node(key+"_out", 1585, y, 165, 85, output,
               color="purple" if key == "final" else "orange", size=20)
        p.edge(key+"_h", key+"_enc", "orange", start=(1, .5), end=(0, .5))
        p.edge(key+"_enc", key+"_out", "orange", start=(1, .5), end=(0, .5))
    p.node("ellipsis", 1090, 575, 660, 55, "… 第 3–7 层同样调用 …", text=True, size=25)
    p.node("readout_note", 1090, 820, 660, 65,
           "第 8 层结束，经过最终 RMSNorm 后\n结构预测读出也复用同一个 Encoder θ", text=True, size=23)
    p.node("sharing_note", 1090, 1060, 660, 100,
           "θ：跨层共享，也用于最终距离预测\nW1 … W8：各层独立，初始权重为 0\nPair Feature 每层重算，跨层传递的是 h", text=True, size=23)
    p.node("pair_note", 60, 1110, 970, 55,
           "Bias 逻辑形状：[12, S, S]；相对位置 / 链角色也参与构造 Pair Feature。", text=True, size=22)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mxfile = ET.Element("mxfile", host="app.diagrams.net", type="device", version="24.7.17")
    with PdfPages(OUT / f"{STEM}.pdf") as pdf:
        for page in [overview(), block_detail(), input_detail(), pair_and_sharing()]:
            page.export(mxfile, pdf)
    ET.indent(mxfile, space="  ")
    path = OUT / f"{STEM}.drawio"
    ET.ElementTree(mxfile).write(path, encoding="utf-8", xml_declaration=True)
    # Keep previously shared links current while giving this expanded version a distinct filename.
    (OUT / "pair_training_simple_20261025.drawio").write_bytes(path.read_bytes())
    (OUT / "pair_training_simple_20261025.pdf").write_bytes((OUT / f"{STEM}.pdf").read_bytes())
    parsed = ET.parse(path)
    for diagram in parsed.findall("diagram"):
        cells = diagram.findall(".//mxCell")
        ids = {c.get("id") for c in cells}
        assert len(ids) == len(cells)
        edges = [c for c in cells if c.get("edge") == "1"]
        assert all(e.get("source") in ids and e.get("target") in ids for e in edges)
        assert all("image=" not in c.get("style", "") for c in cells)
        print(f"{diagram.get('name')}: {len(cells)-len(edges)-2} editable nodes, {len(edges)} connected edges")
    print(f"Saved {path}; XML, connector references and text bounds validated.")


if __name__ == "__main__":
    main()
