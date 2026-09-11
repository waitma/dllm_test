# TCR repertoire（TcrDesign bCDR3）处理方式

记录 2026-09-10～09-12 对无标签 CDR3β 预训练池的处理约定、试过的算法，以及 **junc80 构建**。

- **v4 / v5 prepared 指向 `data/tcr_repertoire_junc80/`**（完整 IMGT junction）。
- 旧 core 语料 `data/tcr_repertoire/` 仍在盘上；部分 v3 checkpoint / 构建报告仍引用它。
  两者 `cdr3b` 语义不同，不可互换。
- 当前训练默认不再是「随机 2M core」。prepared 入口见
  [`dllm/pipelines/immune_llada/README.md`](../../dllm/pipelines/immune_llada/README.md)。

相关脚本：

- 线上随机 2M：`build_repertoire.py` → `data/tcr_repertoire/`
- 默认 linclust（已否定）：`cluster_sample_repertoire.py` → `data/tcr_repertoire_cluster80/`
- Hamming / Hobohm：`cluster_sample_repertoire_hamming.py`
  - core 版（已跑完，对照）：`data/tcr_repertoire_hamming80/`
  - **junction 版（当前处理）**：`data/tcr_repertoire_junc80/`

---

## 1. 数据是什么

源头：TcrDesign-2026 `data/tcr_papers/raw/tcrdesign2026/pretrain/bCDR3_{train,val}.csv`。

- 每行一条裸 β CDR3，**原文就是完整 IMGT junction**，例如 `CASSLPEGAGGNTGELFF`。
- 读入约 **40,308,610** 行；精确 unique 后约 **40,308,605**。
- TcrDesign-G / TCRT5 / OLGA / GRATCR 训练和出序列都用带 **C…[FW]** 的 junction，不去掉两端。

两种字符串（同一条受体）：

| 名称 | 例子 | 长度（这条） |
|---|---|---|
| junction | `CASSLPEGAGGNTGELFF` | 15 |
| core（剥开头 C、结尾 F/W） | `ASSLPEGAGGNTGELF` | 13 |

OTS 的 `chain1_cdr3` 已是 ANARCI loop（core）。和 OTS / blocklist 做**集合运算**时必须统一成 core，否则 `CASS…F` 和 `ASS…` 对不上。  
这只是比对 key，**不是**「训练也必须写成 core」。

---

## 2. 旧 core 构建（`data/tcr_repertoire/`；v3 仍指向，v4+ 不用）

`build_repertoire.py`：

1. 读 junction，剥成 core，按 core 去重。
2. core 落在 blocklist 上则丢（精确相同）。blocklist = OTS valid/holdout ∪ T4 ∪ T2/T3 ∪ OTS benchmark ∪ binding benchmark，并集约 12.4 万 core。
3. 按 core hash 切 valid/holdout 各 0.5%，其余 train。
4. train 随机抽 **2,000,000**（seed 42）。全量约 3000 万条会占掉约 31% token 预算。
5. **写入 `cdr3b` 的是 core，不是 junction。** 完整串在内存里有过，落盘扔掉了。

结果：`data/tcr_repertoire/dataset/`，train **2,128,750**（含后来从 valid 搬回的近重复）。  
`tcr_single` 训练看到的是 `ASS…`，和 TcrDesign-G 的 `CASS…F` 不一致。

---

## 3. 试过、已否定或仅作对照的算法

### 3.1 默认 `easy-linclust` 0.80（`tcr_repertoire_cluster80`）

对 4038 万条（干净 core + blocklist）跑 MMseqs `easy-linclust --min-seq-id 0.8 -c 0.8`。

- 墙钟约 6–8 分钟。4038 万 → 3114 万簇，**87.4% 单条**，平均大小 1.30。
- 第二轮 align 平均每条只有 1.000001 hit（几乎只打到自己）。
- 仓库里已有结论：12–18 aa 上 k-mer 预过滤不可靠，Lev≤1 曾漏约 41%（`move_near_dup_eval_rows.py`）。
- 作者 issue 也要求短肽用 `-k 5`、`--mask 0`、关掉 spaced k-mer，不要默认 linclust。

**结论：这不是 0.80 家族抽取，接近漏聚后再随机抽。目录保留作对照，不当正式语料。**

### 3.2 Hamming + Hobohm，打在 core 上（`tcr_repertoire_hamming80`）

按 **core 长度** 分桶，`d = ⌊0.2 × L_core⌋`（13 mer → d=2，identity 实际是 11/13≈**84.6%**，不是正好 80%）。  
Hobohm-1 贪心留代表（见 §4）。写出 junction。

- 干净 core 40,257,598 → kept **6,832,016**（17%）→ train 抽 2,000,000。
- 已完成，`PASS=true`。主体 L=13 kept 约 138 万。
- 问题：长度按 core 算，0.80 被算严；训练串虽已补回 C/F，相似度仍不在 junction 上。

---

## 4. 当前处理（`tcr_repertoire_junc80`）

脚本：`cluster_sample_repertoire_hamming.py`（默认 `--out-root data/tcr_repertoire_junc80`）。  
tmux：`cdr3-junc80`。日志：`data/tcr_repertoire_junc80/run.log`。

**定性：这是按 0.80 邻域做的代表抽取（packing），不是「分簇 + 权重 + 全员训练」。**  
absorbed / near_block **不会**写入 CSV。

### 4.1 准备

1. 读 TcrDesign 两份文件，校验氨基酸。
2. 按 **完整 junction 字符串** 精确去重（不再先剥成 core 再 unique）。
3. 若 `cdr3_core(junction)` 落在 blocklist → 精确丢掉（`drop_benchmark`）。
4. 干净池约 **40,257,603** 条 junction。

### 4.2 相似度（和上一版的差别）

- **不剥 C/[FW]**，Hamming 打在完整 junction 上。
- **只和相同 junction 长度比**，不同长度不聚类。
- 近邻当且仅当 **identity ≥ 0.80**，即替换数 `d ≤ ⌊0.20 × L⌋`。  
  **identity < 0.80 的保持分开。**

| junction 长度 L | d | 最低 identity |
|---|---|---|
| 8–9 | 1 | 0.875 / 0.889 |
| 10 | 2 | **0.80** |
| 14 | 2 | 0.857 |
| **15（主体，对应 core 13）** | **3** | **0.80（12/15）** |
| 16 | 3 | 0.812 |
| 20 | 4 | 0.80 |

没有 indel，不用 BLOSUM。等长、共享一把「删掉 d 个位置后的钥匙」⇔ Hamming ≤ d。

### 4.3 Hobohm-1（kept / absorbed / near_block）

同一长度内按 **字母序** 扫干净 junction：

1. 和某条 **blocklist 的 junction 形式** Hamming ≤ d → **near_block，删除**（不进代表集）。  
   blocklist 本身是 core，比较前补成 `C+core+F` 和 `C+core+W`。
2. 否则和某条 **已经 kept 的代表** Hamming ≤ d → **absorbed，删除**（被那条代表盖住）。
3. 否则 **自己 kept**，之后别人不能再挨它太近。

因此 kept 彼此 identity **< 0.80**（15 mer 上至少差 4 个字母）。

不用单链接：A–B、B–C 各差 1 个时，不会把 A 和 C 并成一家只留 1 条。

**absorbed = 对输出语料直接删除**，原文 TcrDesign 文件里还在，只是这份抽样不再单独留行。

### 4.4 划分与写出

kept 全体再按 junction 做 hash（seed 42，标签 `junc80`）：

- 0.5% valid、0.5% holdout，其余 train；
- train 若超过 `--max-train`（默认 200 万）再随机抽到 200 万。

CSV 的 `cdr3b` = **完整 junction**（以 C 开头）。  
`fv_source=tcrdesign2026_cdr3_junc80`。  
加载：`sources.py` `_repertoire` 用存盘字符串当 `tcr_beta`；若形如 `C…[FW]` 则剥一次当 `cdr3b_core` / blocklist key。

### 4.5 中间数字（跑的时候以 `run.log` / `work/cluster.done.json` 为准）

prepare 已完成。主体档在跑 L=15/16/17（d=3）。  
L=15（1001 万条）曾留下约 37 万（约 3.7%），`near_block` 约 850 万——d=3 的评测球在最常见长度上盖得很满。

---

## 5. 和「聚类」的差别（已讨论、尚未改）

当前逻辑是 **抽取**：每家只留代表，近邻删除，无 `cluster_id`、无权重。

若做成真正聚类，应是：

1. 每条 junction 都归属一个簇；
2. 给出权重（例如 `weight = 1 / 簇大小`，每个 0.80 家族对 loss 贡献相同）；
3. **全部数据仍用于训练**，不因 absorbed 从语料消失。

全量约 4000 万条会重新碰到 token 预算（原先抽 2M 就是为了把 repertoire 压在约 2.4% 残基）。  
未落地：全量 + 权重，或按簇抽样但保留成员。

---

## 6. 目录与训练指向

| 路径 | 内容 | 训练 |
|---|---|---|
| `data/tcr_repertoire/` | v3 随机 2M，`cdr3b`=core | **线上仍指向这里** |
| `data/tcr_repertoire_cluster80/` | 默认 linclust，勿当 0.80 家族 | 否 |
| `data/tcr_repertoire_hamming80/` | Hobohm on **core**，写出 junction | 否（对照） |
| `data/tcr_repertoire_junc80/` | Hobohm on **junction**，identity≥0.80 | 未切过去 |

未改 v3 prepared / `immune_v4` 的 `--tcr-repertoire-dir`。要换语料需显式改配置并重跑 prepare。

---

## 7. 复现

```bash
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
# 当前 junction 版（不要盖掉 hamming80）
tmux new-session -d -s cdr3-junc80 \
  "/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python \
     scripts/data/tcr_native/cluster_sample_repertoire_hamming.py \
     --max-train 2000000 \
     --out-root /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_repertoire_junc80 \
     > /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_repertoire_junc80/run.log 2>&1"
```

core 对照版把 `--out-root` 换成 `data/tcr_repertoire_hamming80`，并使用该目录里已写死的旧脚本行为（当时 `cluster_on=anchor_free_core`）。现脚本默认已是 junction。
