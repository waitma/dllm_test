"""Draw the implemented Stage 1 architecture, without running either model.

Activate conda pllm, then run:
  python -B /project_c20250601/251105016/project/dllm_test/scripts/research/draw_stage1_architecture.py

Exports a large-type feature-flow diagram, three explanatory PNG/SVG panels,
and vector PDFs into docs/figures.
The rendering style follows scripts/draw_research_refined_architecture.py;
the content describes StructureMLMModel, not that historical A4 architecture.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "figures"
STEM = "stage1_structure_architecture_20261022"
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
INK, MUTED, LINE = "#213348", "#52667C", "#CBD5E1"
BLUE, ORANGE, TEAL, PURPLE = "#2D61A0", "#B35D19", "#16776E", "#7755A1"
PB, PO, PT, PP, PG = "#EAF2FC", "#FFF2E5", "#E8F6F3", "#F3EEFA", "#F3F6FA"
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})


class Diagram:
    def __init__(self, title, subtitle, number, height=1200):
        self.height = height
        self.fig = plt.figure(figsize=(18, height / 100), facecolor="white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1800), ylim=(height, 0), aspect="equal")
        self.ax.axis("off")
        self.texts = []
        self.text(40, 43, title, 24, bold=True)
        self.text(40, 85, subtitle, 11.8, color=MUTED)
        self.text(1760, 43, f"{number} / 3", 13, color=MUTED, ha="right")

    def text(self, x, y, label, size=12, *, bold=False, color=INK, ha="left", limit=None):
        artist = self.ax.text(x, y, label, fontproperties=BOLD if bold else FONT,
                              fontsize=size, color=color, ha=ha, va="center",
                              linespacing=1.45, zorder=5)
        self.texts.append((artist, limit))
        return artist

    def box(self, x, y, w, h, fill="white", edge=LINE, dashed=False, radius=10):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={radius}", facecolor=fill,
            edgecolor=edge, linewidth=1.15, linestyle=(0, (5, 3)) if dashed else "-", zorder=1))

    def card(self, x, y, w, h, title, body="", fill=PG, edge=LINE, ts=13, bs=11.5):
        self.box(x, y, w, h, fill, edge)
        bound = (x + 6, y + 3, x + w - 6, y + h - 3)
        self.text(x+w/2, y+(22 if body else h/2), title, ts, bold=True, ha="center", limit=bound)
        if body:
            self.text(x+w/2, y+(h+31)/2, body, bs, ha="center", limit=bound)

    def arrow(self, start, stop, color=MUTED, dashed=False):
        self.ax.add_patch(FancyArrowPatch(start, stop, arrowstyle="-|>",
            mutation_scale=13, linewidth=1.35, color=color,
            linestyle=(0, (5, 3)) if dashed else "-", shrinkA=2, shrinkB=3, zorder=3))

    def route(self, points, color=MUTED, dashed=False):
        if len(points) > 2:
            x, y = zip(*points[:-1])
            self.ax.plot(x, y, color=color, linewidth=1.35,
                         linestyle=(0, (5, 3)) if dashed else "-", zorder=2)
        self.arrow(points[-2], points[-1], color, dashed)

    def panel(self, x, y, w, h, title, color=INK):
        self.box(x, y, w, h)
        self.text(x+24, y+29, title, 15, bold=True, color=color)

    def footer(self, label):
        self.text(40, self.height-29, label, 10.6, color=MUTED)

    def validate(self):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        problems = []
        for artist, limit in self.texts:
            bb = artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            x0, x1 = sorted((bb.x0, bb.x1))
            y0, y1 = sorted((bb.y0, bb.y1))
            lx, ly, rx, ry = limit or (10, 10, 1790, self.height-10)
            if x0 < lx or x1 > rx or y0 < ly or y1 > ry:
                problems.append({"text": artist.get_text(), "bounds": [x0,y0,x1,y1],
                                 "allowed": [lx,ly,rx,ry]})
        return problems


def overview():
    d = Diagram("Stage 1：结构增强 BERT 的整体训练架构",
                "当前本地实现 · 全部 8 层参与 pair 反馈 · 教师离线缓存 · 学生联合 MLM + 结构蒸馏", 1)
    for x, title, fill, edge in [(40,"蓝色：继承的学生模块",PB,BLUE),
        (470,"橙色：新增可训练模块",PO,ORANGE),(900,"绿色：冻结教师 / 缓存",PT,TEAL),
        (1330,"紫色：监督与损失",PP,PURPLE)]:
        d.card(x, 115, 410, 40, title, fill=fill, edge=edge, ts=11.4)
    d.card(40, 185, 1720, 90, "同一条 prepared 记录：真实序列 + 链角色 + 观测来源",
           "先建立干净残基 → 学生 token 位置映射，再给学生加噪；补齐 / 片段链保留 MLM 用途，排除其结构监督。",
           ts=14, bs=12)
    d.arrow((435,275),(435,320),BLUE)
    d.arrow((1315,275),(1315,320),TEAL)
    d.panel(40,320,790,660,"学生：已有 BERT 权重完整继承，所有参数可训练",BLUE)
    d.panel(870,320,890,660,"教师：ESMFold2-Fast，6.534B 参数冻结",TEAL)
    d.card(70,380,730,85,"Full-token MLM → Token + Role + Scope embedding",
           "15% 独立选择，80 / 10 / 10 扰动；$h^{(0)}$ 的形状为 [B, S, 768]",PB,BLUE,bs=11.1)
    d.arrow((290,465),(290,490),BLUE)
    for layer in range(8):
        y=490+43*layer
        kind="链内注意力" if layer%2==0 else "全局注意力"
        d.card(70,y,440,34,f"第 {layer+1} 层   {kind} + Pair Bias",fill=PB,edge=BLUE,ts=12)
        if layer<7:
            d.arrow((290,y+34),(290,y+43),BLUE)
        d.arrow((560,y+17),(510,y+17),ORANGE)
    d.card(560,490,240,335,"共享 Pair Encoder",
           "每层从当前 h 重新构造\n64 维对称 pair feature\n\n每层独立 64 → 12 投影\n8 个反馈出口零初始化\n\n保留链内 / 全局拓扑\nQ/K/V、RoPE、残差不变",PO,ORANGE,ts=12.5,bs=10.3)
    d.arrow((290,825),(290,850),BLUE)
    d.card(70,850,730,40,"最终 RMSNorm → h_final  [B, S, 768]",fill=PB,edge=BLUE,ts=12)
    d.arrow((240,890),(240,920),BLUE)
    d.arrow((630,890),(630,920),ORANGE)
    d.card(70,920,340,50,"原 token 输出层 → MLM logits",fill=PB,edge=BLUE,ts=11.5)
    d.card(460,920,340,50,"采样 pair → 新增 64 类距离头",fill=PO,edge=ORANGE,ts=11.5)
    d.card(910,380,810,85,"筛选可用的干净氨基酸链",
           "真实完整链 / 可变区；跨链监督还需配对来源；无学生语法 token、EOS 或空槽",PT,TEAL,bs=10.7)
    d.arrow((1315,465),(1315,500),TEAL)
    d.card(910,500,810,85,"逐链构建原生特征，再组合为多链输入",
           "序列 / 残基 / 原子特征 + asym_id / entity_id / sym_id + 链内位置",PT,TEAL,bs=11.1)
    d.arrow((1315,585),(1315,620),TEAL)
    d.card(910,620,810,105,"冻结 ESMFold2-Fast：仅 trunk forward",
           "本次工程 pilot：num_loops = 3，固定随机种子\n输出原生 distogram logits [L, L, 64]；不运行坐标生成",PT,TEAL,bs=11.6)
    d.arrow((1315,725),(1315,760),TEAL)
    d.card(910,760,810,105,"离线稀疏缓存：教师分数 + 残基对 + 位置映射",
           "在合格链内 / 链间抽样，排除对角线；保存版本、权重哈希、seed、loops\n教师分数只作为目标，梯度在教师端停止",PT,TEAL,bs=11.3)
    d.arrow((1315,865),(1315,900),TEAL)
    d.card(910,900,810,65,"student_positions 将教师残基对映射到学生 token 对",
           "缓存读取时检查序列、链身份、端点与来源；EOS / 空槽无结构标签",PT,TEAL,ts=11.8,bs=10.7)
    d.arrow((240,970),(240,1025),PURPLE)
    d.arrow((630,970),(630,1025),PURPLE)
    d.arrow((1315,965),(1315,1025),PURPLE)
    d.card(40,1025,1720,110,"L = L_MLM + λ_intra · L_KD,intra + λ_inter · L_KD,inter",
           "KL(teacher || student)：残基对 → 链对 → 记录；链内 / 链间独立归一化。梯度更新全部学生层、新 Pair Encoder 与两个输出路径。\n教师只监督距离类别分布；学生 encoder 的前向输入不包含教师张量或真实残基监督掩码。",
           PP,PURPLE,ts=17,bs=11.2)
    d.footer("继承 75,601,176 参数；新增 131,136（约 0.173%）。工程验证完成；教师结构质量、正式预训练及后续 diffusion 生成收益仍待验证。")
    return d


def layer_detail():
    d = Diagram("单层展开：pair feature 如何进入全部 8 层注意力",
                "蓝色路径继承原参数；橙色路径新增。下图在第 1–8 层重复，Pair Encoder 参数共享，bias 投影逐层独立。",2)
    d.card(40,150,800,65,"当前层输入 $h^{(l)}$  [B, S, 768]",fill=PB,edge=BLUE,ts=15)
    d.arrow((440,215),(440,270),BLUE)
    d.card(40,270,800,100,"原 RMSNorm → Q / K / V → 链内 RoPE",
           "12 个注意力头，每头 64 维；保留原投影与位置编码",PB,BLUE,ts=14,bs=12)
    d.arrow((440,370),(440,395),BLUE)
    d.arrow((440,445),(440,475),BLUE)
    d.text(440,420,"chain_ids → 原链关系 bias 与拓扑掩码",11.5,ha="center",color=BLUE)
    d.card(40,475,800,120,"注意力：合并新增 bias 后重新施加原 hard mask",
           "softmax($QK^T$ / √64 + b_chain + b_pair + M) V\n奇数层：同链全注意力；偶数层：全局注意力",PB,BLUE,ts=13.4,bs=13)
    d.arrow((440,595),(440,650),BLUE)
    d.card(40,650,800,65,"h′ = $h^{(l)}$ + 0.1 × AttentionOutput",fill=PB,edge=BLUE,ts=14)
    d.arrow((440,715),(440,770),BLUE)
    d.card(40,770,800,100,"原 RMSNorm → SwiGLU FFN",
           "隐藏宽度 3072；$h^{(l+1)}$ = h′ + 0.1 × FFN(h′)",PB,BLUE,ts=14,bs=12.5)
    d.arrow((440,870),(440,925),BLUE)
    d.card(40,925,800,65,"下一层输入 $h^{(l+1)}$  [B, S, 768]",fill=PB,edge=BLUE,ts=14)
    d.arrow((840,182),(970,182),ORANGE)
    d.card(970,150,790,90,"共享单位置投影：$u_i$ = Linear(LayerNorm($h_i$))",
           "768 → 64；在当前层输入上计算",PO,ORANGE,ts=13.3,bs=12)
    d.arrow((1365,240),(1365,290),ORANGE)
    d.card(970,290,790,130,"拼接 5 组各 64 维的特征 → 320 维",
           "$u_i + u_j$    |    abs($u_i - u_j$)    |    $u_i \\odot u_j$\n位置 embedding：链内 abs(Δpos) 截断到 32；跨链用类别 33\n链 / 角色 embedding：是否同链 + 无序角色组合",PO,ORANGE,ts=13.2,bs=11.7)
    d.arrow((1365,420),(1365,465),ORANGE)
    d.card(970,465,790,100,"共享 Pair MLP：320 → 128 → 64",
           "Linear → SiLU → Linear → LayerNorm\n输出对称 pair feature $z_{ij} = z_{ji}$",PO,ORANGE,ts=14,bs=12)
    d.arrow((1365,565),(1365,615),ORANGE)
    d.card(970,615,790,90,"本层独立投影 $W^{(l)}_{bias}$：64 → 12，无 bias 参数",
           "投影权重零初始化；输出 [B, 12, S, S] 的 attention bias",PO,ORANGE,ts=13,bs=11.8)
    d.route([(970,660),(905,660),(905,535),(840,535)],ORANGE)
    d.card(970,790,790,200,"最终层的另一路用途：距离分布预测",
           "第 8 层结束后，先做最终 RMSNorm\n只对采样的真实残基对，复用共享 Pair Encoder\n64 维 pair → 新增 Linear(64, 64) → 距离类别 logits\n与冻结教师做 KL；梯度可以传回全部 8 层主干",
           PP,PURPLE,ts=14,bs=12)
    d.arrow((840,958),(970,958),PURPLE,dashed=True)
    d.text(905,1015,"仅最后一层",10.3,ha="center",color=PURPLE)
    d.box(40,1060,1720,75,PG,LINE)
    d.text(65,1084,"跨层状态由 BERT hidden states 传递；pair feature 每层重算。初始反馈为零，加载后保持原 BERT 输出。",12)
    d.text(65,1111,"Pair 中间特征按 query 分块并做梯度检查点；attention bias 仍是稠密 S × S，计算量随长度二次增长。",11.5,color=MUTED)
    d.footer("M 沿用原链内 / 全局与 padding 可见性；pair bias 不会打开被 hard mask 禁止的连接。教师输出与 loss 掩码均不进入该前向路径。")
    return d


def alignment():
    d = Diagram("固定槽位与教师对齐：真实 MASK、EOS、尾部空槽",
                "取已验收的第一条 OAS H/L 样本：学生 S = 310 tokens；教师 L = 238 residues。所有绝对索引从 0 开始。",3)
    d.panel(40,145,1720,305,"A  学生固定容量：H 167 槽 + L 135 槽，保留原语法 token",BLUE)
    for y,label,n,resrange,eos,tail,tailrange in [
        (220,"H · 167 槽",128,"6–133",134,38,"135–172"),
        (340,"L · 135 槽",110,"174–283",284,24,"285–308")]:
        d.text(65,y+35,label,13,bold=True,color=BLUE)
        d.card(245,y,900,70,f"{n} 个真实残基位置：学生索引 {resrange}",
               "MLM 后：可见残基 … [MASK] … 可见残基；被遮盖位置仍是原残基位置",PB,BLUE,ts=12.5,bs=10.5)
        d.card(1160,y,160,70,"EOS",f"索引 {eos}",PP,PURPLE,ts=13,bs=11.5)
        d.card(1335,y,395,70,f"{tail} 个尾部 [MASK] 空槽",f"索引 {tailrange}；没有残基标签",PO,ORANGE,ts=12.1,bs=10.8)
    d.text(65,425,"前缀索引 0–5；链分隔符 '.' 在 173；末尾 <protd> 在 309。块宽为示意，不按残基数等比例绘制。",10.5,color=MUTED)
    d.panel(40,480,1720,200,"B  教师只见干净真实序列；通过 student_positions 精确映射",TEAL)
    d.card(70,545,690,65,"教师 H：0–127  →  学生 H：6–133",fill=PT,edge=TEAL,ts=13.5)
    d.card(800,545,690,65,"教师 L：128–237  →  学生 L：174–283",fill=PT,edge=TEAL,ts=13.5)
    d.text(1525,577,"EOS / 空槽\n不进入教师",10.5,color=TEAL)
    d.text(70,647,"跨链 pair 示例：(0, 128) → (6, 174)。映射在 MLM 加噪前建立；MASK / 随机替换不会移动或删除原位置。",12)
    d.panel(40,710,1720,330,"C  参与前向 ≠ 具有直接监督：两种 MASK 使用同一个 token ID",INK)
    widths=[480,270,270,300,340]
    columns=["位置类型","普通 attention","新增 pair bias","直接 MLM 监督","直接结构 KD"]
    x0,y0=70,770
    x=x0
    for width,label in zip(widths,columns):
        d.box(x,y0,width,42,PG,LINE,radius=0)
        d.text(x+width/2,y0+21,label,11.5,bold=True,ha="center")
        x+=width
    rows=[
        ["真实残基 → MLM MASK","参与","参与","恢复原残基","有合格教师目标时参与"],
        ["EOS（本身也可被 MLM 遮盖）","参与","参与","被选中时恢复 EOS","无"],
        ["EOS 后固定 MASK 空槽","参与","参与","无，label = −100","无"],
        ["链外语法 token / 链分隔符","按原拓扑参与","不添加","按原 full-token 配方","无"],
        ["Batch padding","屏蔽其 key；输出丢弃","不添加","无","无"],
    ]
    for row,values in enumerate(rows):
        x=x0;y=y0+42+row*40
        for width,label in zip(widths,values):
            d.box(x,y,width,40,PO if row==2 else "white",LINE,radius=0)
            d.text(x+width/2,y+20,label,10.8,ha="center")
            x+=width
    d.box(40,1060,1720,75,PG,LINE)
    d.text(65,1083,"前向 pair 有效位置 = attention_mask 且 chain_id ≥ 0 且 position_id ≥ 1；尾部空槽仍可间接影响真实残基表示。",11.4)
    d.text(65,1112,"真实残基映射 / labels 仅用于监督，不传给 encoder；原生成解析器按各链第一个 EOS 截断，尾部内容不计入输出序列。",11.2,color=MUTED)
    d.footer("当前 Stage 1 保留固定槽位设计。含 synthetic 补齐或片段标注的链整体排除 KD；其 MLM 用途保留。Batch padding 与链内空槽是不同概念。")
    return d


class FeatureFlow(Diagram):
    """Large labels and explicit tensor routes; no small explanatory paragraphs."""

    def __init__(self):
        self.height, self.width = 2540, 2400
        self.fig = plt.figure(figsize=(24,25.4),facecolor="white")
        self.ax = self.fig.add_axes([0,0,1,1])
        self.ax.set(xlim=(0,self.width),ylim=(self.height,0),aspect="equal")
        self.ax.axis("off")
        self.texts=[]

    def text(self,x,y,label,size=20,**kwargs):
        if size < 18:
            raise ValueError("Feature-flow labels must be at least 18 pt")
        artist=super().text(x,y,label,size,**kwargs)
        artist.set_linespacing(1.0)
        return artist

    def arrow(self,start,stop,color=MUTED,dashed=False):
        self.ax.add_patch(FancyArrowPatch(start,stop,arrowstyle="-|>",
            mutation_scale=20,linewidth=2,color=color,
            linestyle=(0,(5,4)) if dashed else "-",shrinkA=2,shrinkB=4,zorder=3))

    def route(self,points,color=MUTED,dashed=False):
        if len(points)>2:
            x,y=zip(*points[:-1])
            self.ax.plot(x,y,color=color,linewidth=2,
                         linestyle=(0,(5,4)) if dashed else "-",zorder=2)
        self.arrow(points[-2],points[-1],color,dashed)

    def node(self,x,y,w,h,label,fill=PB,edge=BLUE,size=20):
        self.box(x,y,w,h,fill,edge,radius=12)
        self.text(x+w/2,y+h/2,label,size,ha="center",
                  limit=(x+7,y+4,x+w-7,y+h-4))

    def panel(self,x,y,w,h,title,color=INK):
        self.box(x,y,w,h,edge=LINE,radius=16)
        self.text(x+25,y+37,title,24,bold=True,color=color)

    def join(self,x,y,label="+",color=ORANGE):
        self.ax.add_patch(Circle((x,y),26,facecolor="white",edgecolor=color,
                                linewidth=2,zorder=4))
        self.text(x,y,label,25,ha="center",color=color)

    def validate(self):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        problems=[]
        for artist,limit in self.texts:
            bb=artist.get_window_extent(renderer).transformed(self.ax.transData.inverted())
            x0,x1=sorted((bb.x0,bb.x1)); y0,y1=sorted((bb.y0,bb.y1))
            lx,ly,rx,ry=limit or (10,10,self.width-10,self.height-10)
            if x0<lx or x1>rx or y0<ly or y1>ry:
                problems.append({"text":artist.get_text(),"bounds":[x0,y0,x1,y1],
                                 "allowed":[lx,ly,rx,ry]})
        return problems


class VerticalOffset:
    """Place a detail panel lower without duplicating drawing primitives."""

    def __init__(self,diagram,offset):
        self.diagram,self.offset=diagram,offset

    def node(self,x,y,*args,**kwargs):
        return self.diagram.node(x,y+self.offset,*args,**kwargs)

    def text(self,x,y,*args,**kwargs):
        return self.diagram.text(x,y+self.offset,*args,**kwargs)

    def panel(self,x,y,*args,**kwargs):
        return self.diagram.panel(x,y+self.offset,*args,**kwargs)

    def arrow(self,start,stop,*args,**kwargs):
        return self.diagram.arrow((start[0],start[1]+self.offset),
                                  (stop[0],stop[1]+self.offset),*args,**kwargs)

    def route(self,points,*args,**kwargs):
        return self.diagram.route([(x,y+self.offset) for x,y in points],*args,**kwargs)


def feature_flow():
    d=FeatureFlow()
    d.text(40,51,"Stage 1 BERT：整体架构与内部特征流",32,bold=True)
    d.text(40,108,"左：完整前向主线     右：A / B / C / D 模块展开",22,color=MUTED)
    for x,label,color in [(1370,"继承模块",BLUE),(1640,"新增 Pair",ORANGE),
                          (1920,"冻结教师",TEAL),(2190,"Loss",PURPLE)]:
        d.text(x,108,label,20,color=color)

    # Main architecture: each feature has an incoming and outgoing route.
    d.panel(40,160,640,2300,"整体架构",BLUE)
    d.node(80,245,560,115,"MLM 序列 + 固定链元信息\nToken / Role / Scope IDs  [B,S]",size=21)
    d.arrow((360,360),(360,430),BLUE)
    d.node(80,430,560,100,"A  输入嵌入\n[B,S] → [B,S,768]",size=23)
    d.arrow((360,530),(360,620),BLUE)
    d.panel(70,620,580,595,"B  Transformer × 8",BLUE)
    d.node(100,710,520,75,"$h^{(l)}$   [B,S,768]",size=23)
    d.arrow((240,785),(240,1000),BLUE)
    d.arrow((520,785),(520,850),ORANGE)
    d.node(420,850,200,100,"C  Pair\n64 → 12",PO,ORANGE,size=21)
    d.route([(520,950),(520,1045),(380,1045)],ORANGE)
    d.text(555,980,"bias",20,color=ORANGE)
    d.node(100,1000,280,90,"Attention + FFN\n768 → 768",size=20)
    d.arrow((240,1090),(240,1130),BLUE)
    d.node(100,1130,520,60,"$h^{(l+1)}$   [B,S,768]",size=22)
    d.text(360,1255,"1 / 3 / 5 / 7：链内",21,ha="center",color=BLUE)
    d.text(360,1298,"2 / 4 / 6 / 8：全局",21,ha="center",color=BLUE)
    d.route([(360,1190),(665,1190),(665,1360),(640,1360)],BLUE)
    d.node(80,1320,560,90,"最终 RMSNorm\n$h_{final}$  [B,S,768]",size=22)
    d.route([(360,1410),(360,1460),(200,1460),(200,1510)],BLUE)
    d.route([(360,1460),(520,1460),(520,1510)],ORANGE)
    d.node(80,1510,240,115,"D  MLM 头\n[B,S,50]",size=22)
    d.node(400,1510,240,115,"D  距离头\n[N,64]",PO,ORANGE,size=22)
    d.arrow((200,1625),(200,1720),PURPLE)
    d.arrow((520,1625),(520,1720),PURPLE)
    d.node(80,1720,240,90,"$L_{MLM}$",PP,PURPLE,size=26)
    d.node(400,1720,240,90,"$L_{KD}$\n教师 → 学生 KL",PP,PURPLE,size=20)
    d.route([(200,1810),(200,1970)],PURPLE)
    d.route([(520,1810),(520,1970)],PURPLE)
    d.node(80,1970,560,125,"联合更新全部学生参数\nMLM + 链内 KD + 链间 KD",PP,PURPLE,size=22)
    d.text(360,1895,"教师只提供 loss 目标",21,ha="center",color=TEAL)
    d.node(80,2230,560,150,"干净完整链 → ESMFold2-Fast\n冻结教师输出 [L,L,64]\n抽样 + 对齐 → [N,64]",PT,TEAL,size=21)
    d.route([(640,2305),(665,2305),(665,1765),(640,1765)],TEAL)

    # A: three real embedding branches, not a prose box.
    d.panel(760,160,1600,300,"A  输入嵌入：三路相加",BLUE)
    for y,label in [(235,"Token IDs"),(310,"Role IDs"),(385,"Scope IDs")]:
        d.node(800,y,240,55,label,size=20)
        d.node(1110,y,350,55,"Embedding  [B,S,768]",size=19)
        d.arrow((1040,y+27.5),(1110,y+27.5),BLUE)
    d.arrow((1460,262.5),(1542,316),BLUE)
    d.arrow((1460,337.5),(1544,337.5),BLUE)
    d.arrow((1460,412.5),(1542,359),BLUE)
    d.join(1570,337.5,"+",BLUE)
    d.arrow((1596,337.5),(1680,337.5),BLUE)
    d.node(1680,290,600,95,"$h^{(0)}$\n[B,S,768]",size=25)
    d.route([(640,480),(720,480),(720,337),(760,337)],BLUE,dashed=True)

    # B: explicit Q/K/V, head, logit, probability, projection and residual routes.
    d.panel(760,500,1600,1000,"B  单层 Transformer：注意力 → 残差 → FFN → 残差",BLUE)
    d.node(800,590,260,70,"$h^{(l)}$\n[B,S,768]",size=20)
    d.node(1120,590,260,70,"RMSNorm",size=22)
    d.node(1440,590,340,70,"Q / K / V 投影\n各 [B,S,768]",size=20)
    d.node(1840,590,440,70,"拆成 12 个头\n各 [B,12,S,64]",size=21)
    for a,b in [(1060,1120),(1380,1440),(1780,1840)]:
        d.arrow((a,625),(b,625),BLUE)
    d.arrow((2060,660),(2060,705),BLUE)
    d.node(1840,705,440,65,"Q、K → 链内 RoPE",size=21)
    d.arrow((2060,770),(2060,815),BLUE)
    d.node(1840,815,440,80,"$QK^T / \\sqrt{64}$\n[B,12,S,S]",size=21)
    d.node(1440,705,340,70,"C  Pair bias\n[B,12,S,S]",PO,ORANGE,size=20)
    d.arrow((1610,775),(1610,815),ORANGE)
    d.node(1440,815,340,80,"+ chain / pair bias\n重施 hard mask",PO,ORANGE,size=20)
    d.arrow((1840,855),(1780,855),BLUE)
    d.node(1120,815,260,80,"Softmax\n[B,12,S,S]",size=20)
    d.arrow((1440,855),(1380,855),BLUE)
    d.node(800,815,260,80,"Attention × V\n[B,12,S,64]",size=19)
    d.arrow((1120,855),(1060,855),BLUE)
    # V bypasses RoPE and attention-score computation.
    d.route([(2170,590),(2170,565),(780,565),(780,855),(800,855)],BLUE)
    d.text(1735,565,"V",20,color=BLUE).set_bbox(dict(facecolor="white",edgecolor="none",pad=2))
    d.arrow((930,895),(930,990),BLUE)
    d.node(800,990,330,80,"合头 + 输出投影\n[B,S,768]",size=21)
    d.node(1180,990,300,80,"$h + 0.1 \\times Attn$\n[B,S,768]",size=20)
    d.arrow((1130,1030),(1180,1030),BLUE)
    d.route([(930,660),(930,730),(1090,730),(1090,970),(1330,970),(1330,990)],BLUE)
    d.node(1540,990,400,80,"RMSNorm + SwiGLU\n768 → 3072 → 768",size=20)
    d.arrow((1480,1030),(1540,1030),BLUE)
    d.node(2010,990,270,80,"$h' + 0.1 \\times FFN$\n[B,S,768]",size=19)
    d.arrow((1940,1030),(2010,1030),BLUE)
    d.route([(1480,1030),(1510,1030),(1510,1120),(2145,1120),(2145,1070)],BLUE)
    d.text(2245,1145,"$h^{(l+1)}$",22,ha="center",color=BLUE)
    d.text(1005,1135,"两条回路：原 hidden state 的残差连接",18,color=MUTED)

    # Expand the inherited gated FFN using both actual projections.
    d.ax.plot([800,2320],[1190,1190],color=LINE,linewidth=1.4)
    d.text(800,1228,"FFN 内部展开：双路投影与 SiLU 门控",22,bold=True,color=BLUE)
    d.node(800,1290,240,75,"h′\n[B,S,768]",size=20)
    d.node(1100,1290,250,75,"RMSNorm\n[B,S,768]",size=20)
    d.arrow((1040,1327),(1100,1327),BLUE)
    d.node(1430,1290,300,75,"Linear 768 → 3072\n[B,S,3072]",size=19)
    d.node(1430,1410,300,75,"Linear 768 → 3072\n[B,S,3072]",size=19)
    d.arrow((1350,1327),(1430,1327),BLUE)
    d.route([(1350,1327),(1390,1327),(1390,1447),(1430,1447)],BLUE)
    d.node(1790,1290,180,75,"SiLU",size=23)
    d.arrow((1730,1327),(1790,1327),BLUE)
    d.join(2000,1405,"×",BLUE)
    d.route([(1970,1327),(2000,1327),(2000,1379)],BLUE)
    d.route([(1730,1447),(2000,1447),(2000,1431)],BLUE)
    d.node(2080,1350,240,110,"Linear\n3072 → 768\n[B,S,768]",size=19)
    d.arrow((2026,1405),(2080,1405),BLUE)

    # C: show all five feature sources and two successive feature dimensions.
    original=d
    d=VerticalOffset(original,300)
    d.panel(760,1220,1600,560,"C  共享 Pair Encoder：五路特征 → pair → 本层 bias",ORANGE)
    d.node(800,1295,300,75,"$h^{(l)}$\n[B,S,768]",size=20)
    d.node(1160,1295,400,75,"LayerNorm + Linear\n768 → 64",PO,ORANGE,size=20)
    d.node(1620,1295,300,75,"u\n[B,S,64]",PO,ORANGE,size=22)
    d.node(1980,1295,300,75,"位置 / 链 / role\n[B,S]",PG,LINE,size=20)
    d.arrow((1100,1332),(1160,1332),ORANGE)
    d.arrow((1560,1332),(1620,1332),ORANGE)
    xs=[800,1100,1400,1700,2000]
    labels=["$u_i + u_j$","$|u_i-u_j|$","$u_i \\odot u_j$","相对位置 Emb","链 / role Emb"]
    for x,label in zip(xs,labels):
        d.node(x,1420,280,65,label,PO,ORANGE,size=20)
    d.route([(1770,1370),(1770,1395),(940,1395),(940,1420)],ORANGE)
    for center in [1240,1540]:
        d.arrow((center,1395),(center,1420),ORANGE)
    d.route([(2130,1370),(2130,1395),(1840,1395),(1840,1420)],MUTED)
    d.route([(2130,1395),(2140,1395),(2140,1420)],MUTED)
    original.ax.plot([940,2140],[1810,1810],color=ORANGE,linewidth=2,zorder=2)
    for center in [940,1240,1540,1840,2140]:
        original.ax.plot([center,center],[1785,1810],color=ORANGE,linewidth=2,zorder=2)
    d.arrow((1020,1510),(1020,1540),ORANGE)
    d.node(800,1540,440,70,"Concat：5 × 64\n[B,q,S,320]",PO,ORANGE,size=20)
    d.node(1340,1540,460,70,"MLP：320 → 128 → 64\nSiLU + 最终 LayerNorm",PO,ORANGE,size=20)
    d.node(1860,1540,420,70,"对称 pair feature z\n[B,q,S,64]",PO,ORANGE,size=21)
    d.arrow((1240,1575),(1340,1575),ORANGE)
    d.arrow((1800,1575),(1860,1575),ORANGE)
    d.route([(2280,1575),(2320,1575),(2320,1630),(1020,1630),(1020,1650)],ORANGE)
    d.node(800,1650,440,70,"本层 Linear：64 → 12\n[B,q,S,12]",PO,ORANGE,size=20)
    d.node(1340,1650,460,70,"Permute + 拼接 query 块",PO,ORANGE,size=20)
    d.node(1860,1650,420,70,"回到 B 的 attention bias\n[B,12,S,S]",PO,ORANGE,size=20)
    d.arrow((1240,1685),(1340,1685),ORANGE)
    d.arrow((1800,1685),(1860,1685),ORANGE)
    d.text(800,1748,"Pair Encoder 全层共享；64 → 12 投影逐层独立，权重零初始化。",20,color=ORANGE)

    # D: two output routes plus an independent frozen teacher-target input.
    d.panel(760,1820,1600,340,"D  输出分支：序列预测与结构蒸馏",PURPLE)
    d.node(800,1905,310,85,"$h_{final}$\n[B,S,768]",size=21)
    d.node(1200,1890,320,70,"Linear：768 → 50",size=20)
    d.node(1600,1890,300,70,"MLM logits\n[B,S,50]",size=21)
    d.node(1980,1890,300,70,"选中 token 的 CE",PP,PURPLE,size=20)
    d.arrow((1110,1947),(1200,1925),BLUE)
    d.arrow((1520,1925),(1600,1925),BLUE)
    d.arrow((1900,1925),(1980,1925),PURPLE)
    d.node(1200,2025,320,70,"复用 C 的 pair z\n[N,64]",PO,ORANGE,size=20)
    d.route([(1110,1947),(1150,1947),(1150,2060),(1200,2060)],ORANGE)
    d.node(1600,2025,300,70,"Linear：64 → 64\n距离 logits [N,64]",PO,ORANGE,size=19)
    d.arrow((1520,2060),(1600,2060),ORANGE)
    d.node(1980,2025,300,70,"KL：链内 / 链间",PP,PURPLE,size=20)
    d.arrow((1900,2060),(1980,2060),PURPLE)
    d.node(800,2025,310,70,"ESMFold2-Fast\n冻结目标 [N,64]",PT,TEAL,size=20)
    d.route([(955,2095),(955,2125),(2130,2125),(2130,2095)],TEAL)
    d.text(40,2200,"S：学生 token 数    q：query 分块宽度（16）    N：采样残基对数    原主干：8 层 / 768 维 / 12 头",20,color=MUTED)
    return original


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    diagrams=[("overview",overview()),("layer",layer_detail()),("alignment",alignment())]
    problems={name:d.validate() for name,d in diagrams}
    if any(problems.values()):
        raise ValueError(json.dumps(problems,ensure_ascii=False,indent=2))
    with PdfPages(OUT / f"{STEM}.pdf") as pdf:
        for name,d in diagrams:
            d.fig.savefig(OUT / f"{STEM}_{name}.svg")
            d.fig.savefig(OUT / f"{STEM}_{name}.png",dpi=160)
            pdf.savefig(d.fig)
            plt.close(d.fig)
    flow=feature_flow()
    flow_problems=flow.validate()
    if flow_problems:
        raise ValueError(json.dumps(flow_problems,ensure_ascii=False,indent=2))
    for ext in ["png","svg","pdf"]:
        flow.fig.savefig(OUT / f"{STEM}_feature_flow.{ext}",dpi=160)
    plt.close(flow.fig)
    print(json.dumps({"output":str(OUT),"panels":[x[0] for x in diagrams],
                      "text_bounds_check":"passed","minimum_feature_flow_font_pt":18,
                      "pdf":str(OUT / f"{STEM}.pdf")},ensure_ascii=False))


if __name__ == "__main__":
    main()
