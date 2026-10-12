# PROJGUIDE — Agent 下游基准执行指南

2026-10-06 新增V7 no-ESMC原生f011模型评测适配；下文ESMC fusion说明属于旧模型谱系，不适用于V7加载方式。V7冻结源码/原生checkpoint使用规则和现有CDR适配见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4；结果仍须来自模型最终decoder输出，范围与榜单纪律不变。

2026-09-30 当前授权新增 [UpTCR binding](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-binding)：作者发布划分/标签原样使用，不做清洗去重或重划分；baseline论文 [P]，不重训。此任务不套用历史清洗gate；数据差异作为比较备注，执行状态统一见 [UpTCR 当前结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#uptcr-binding-status)。范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准。

2026-09-27 AB-AG 范围更新以 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准；纯序列派生协议和本模型入口见[任务手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

> **本文件是 Agent 进入 `downstream/` 时的第一必读文档。**
> 读完本文件再动手写代码、提交作业或更新结果表。
>
> 最后更新：2026-09-30（UTC）

> **2026-09-13 T1 纠错补充**：用户确定的输入约束与后续适配实验建议见
> [TCR baseline 纠错记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本页 T1 的历史 clean/global-mean 描述不能代替新协议验收；
> 冻结探针、全参数适配与关系 token 预测的状态以该记录为准；当前 T1 运行时适配入口与验证见其 §9，直接使用通用 embed_pairs 不等于该协议。其他任务不由 T1 的输入约束自动改写。

---

## 0. 项目目的（先理解再执行）

### 0.1 我们在做什么

本仓库的**核心产物**是正在训练的免疫受体基础模型（`examples/llada` immune fusion）：

```
output/protein_esmc_llada{270m,8b}_{bert,diffusion}_immune[/ _v3]/checkpoint-<step>
```

即 **ESMC encoder + LLaDA decoder**（`LLaDAEsmcFusion` / `protein_pretrain_esmc.py`）。旧 `grammar_v2_*_llada/best.pt` 数字全部 VOID，不作 headline。

`downstream/` 里的一切都是**下游评测**，唯一目的是：

> **量化我们主模型的表征能力与生成能力，并与公开 baseline 公平对比。**

### 0.1.1 评测范围

范围及冻结规则统一引用 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md)；单任务执行约束引用对应任务手册。此处不维护另一份任务排除表。

### 0.2 角色划分

| 角色 | 代表 | Agent 对待方式 |
|------|------|----------------|
| **主模型（我们）** | immune fusion ckpt 目录（含 `model.safetensors`） | 必须按正确协议评测；数字进 headline 主表 |
| **下游参照 baseline** | TCR 侧：SCEPTR、TCRdist3、TCR-BERT、epiTCR/TEIM/ATM-TCR/NetTCR、TCRT5、TcrDesign、OLGA；AB 侧：Ophiuchus-Ab、AntiBERTy、AbLang2、IgGM、dyMEAN、p-IgGen、LiChen；通用对照 ESM2 / ProtBERT | 优先论文表/官方 artifact；本地重跑必须另标来源与协议一致性 |
| **消融/错误口径** | encoder-only（`bioseq:` / `grammar:encoder:`） | 可保留作对照行，**禁止**当作 headline 或能力主张 |

### 0.2.1 Baseline 来源分层（强制）

Baseline 名称里出现 “official” 不代表数字就与论文可比。每条 baseline
结果必须同时记录：

- `evidence_type`：`paper_reported` / `official_artifact_rescored` /
  `official_code_rerun` / `local_reimplementation` / `local_control`；
- `protocol_alignment`：`exact` / `compatible` / `mismatched` /
  `not_comparable`。

### 0.2.2 排行榜构造：论文值优先（强制，2026-09-02 起）

**2026-09-15 用户指定的 AB 汇总口径**：当前抗体结果汇总必须同时展示 Ophiuchus-Ab baseline，baseline 直接引用其论文 Table 1–5 / Figure 4 的 `[P]` 值，不以本地复跑值替代。论文没有精确数值的格（如 m396 六档 Figure 4 点值）标明“未披露精确值”；不得将本地成绩或未标注的读图估计冒充论文值。Ours 固定本轮 checkpoint；输入、长度条件、pooling、选模或 split 不同的比较须就地披露。本条不删除既有历史复跑/其他方法台账，不自动扩展任务范围。

> **本节取代旧「协议不一致时才用论文值」的条件式规约。现在无条件论文值优先。**
> 完整规则见 [`.cursor/rules/downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc) §3。

1. **每个已列 baseline 必须带上它自己的最佳已发表值，作为 `[P]` 主榜行。**
2. **`[R]` 官方代码/权重本地复跑作附行或附列，永不取代 `[P]` 主榜行。** 复跑的作用是证明我们的 harness 正确，不是充当该 baseline 的代表值。
3. **不得因为复跑失败就把 baseline 从表里删掉。** 论文值与复跑都没有时，行保留并显式标 `未获取` + 原因。
4. **论文值协议与我们不一致时，同行披露协议差**（数据版本 / split / 解码 / 负采样 / 聚合），不靠删行回避。

检索范围：只覆盖表内已列 baseline，**不做全文献 SOTA 搜寻**。

**为什么无条件优先**：只摆本地复跑会系统性抬高我们的相对身位。实测例证在 T1 —— `seen_test` 上
8 个官方 baseline 有 7 个的本地复跑**低于**论文值约 0.010（见 [`RESULTS.md`](RESULTS.md) §0.1），
主榜若只放复跑值，Ours 的名次被无偿抬高。

统一注册表：`outputs/external/paper_reported_baselines.csv`；审计入口：
`scripts/audit_baseline_provenance.py`。本地值若与论文协议不一致，除按规则 4 披露外，
仍**禁止**与论文值或 Ours 合并排序。

**T1 binding 特例（original-only）**：主榜取论文 original-model 值作 `[P]` 行（按 §0.2.2 规则 1）。
官方 original checkpoint + 官方推理代码可用时，在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip` 上本地重跑，产出 `[R]` **附行**用于展示复现偏差，**不取代 `[P]` 主榜行**；重跑时不得读取 `train.csv` 或 retrain。Canonical 决策表是 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`；本地运行明细只进同目录 `summary_local_runs.csv`。

> ⚠️ 旧口径是「官方环节可用则取 `[R]`，不可用才 fallback 到 `[P]`」（`[R]` 优先），已于 2026-09-02 反转。

### 0.3 三步工作流（不变）

```
构建下游任务 → 测 baseline → 测我们的模型
```

每步完成后**立即**更新文档（见 §3），不等 GPU 作业全部结束。

---

## 1. CRITICAL — 特征与生成口径（Agent 必须遵守）

### 1.1 核心原则

**评测我们自己的模型时，无论哪种下游任务，读出的 feature 或生成的序列都必须来自 LLaDA decoder 的最终输出。**

```
ESMC encoder（condition）
        ↓
LLaDA decoder（我们训练的核心）
        ↓
final hidden state（post-ln_f）  →  probing / clustering / binding head
        或
扩散迭代 mask-fill 采样         →  CDR infilling / pairing / TCR generation
```

**只跑到 ESMC encoder 末层 = encoder-only = 错误的主口径。**

它只测了冻结骨干，**完全没有体现 LLaDA 训练的部分**，不能代表我们模型的能力。

### 1.2 两种任务形态

| 任务类型 | 正确协议 | 错误协议（仅消融） |
|----------|----------|-------------------|
| **探针/嵌入类**（binding、clustering、representation、PPI、NbBench probing） | clean grammar record → ESMC condition → **LLaDA forward** → `hidden_states[-1]` → **global mean-pool** → 单向量 | `bioseq:` / `grammar:encoder:` 只取 ESMC encoder mean-pool |
| **生成类**（CDR infilling、light pairing、TCR generation、NbBench CDRInfilling gen） | `GrammarEmbedder` 全链路 **mask-fill 扩散解码**至 argmax 收敛 | 用 encoder 特征 + 外接生成头；或只跑 encoder 再贪心解码 |

### 1.3 禁止的分段特征（headline 强制 global pool）

除 encoder-only 外，以下做法也**不能**作为 headline 默认：

- per-chain / per-segment mean-pool 再 concat（`[β‖α‖pep]`）
- `sep_chains=True` 分链拼接后接 head（MINT 官方 baseline 可用，我们模型 headline 不用）
- 两半向量 Hadamard / 逐元素乘

**Headline 统一协议**：

```
渲染完整 grammar record（全链一起）
  → clean post-LLaDA forward（无 mask 噪声）
  → 一次全局 mean-pool（residue_mask 覆盖的所有位置）
  → 单向量 [D] / 样本
```

实现：`GrammarEmbedder(pool_mode="global")`（默认）；MINT PPI 用 `GrammarEmbedder(sep_chains=False)`。
Legacy `segment_concat` / `*_sep_*` / `*_postllada_*`（分段版）**仅作对照脚注**，不作 leaderboard 默认。

### 1.4 Embedder spec 速查表（Agent 接入主模型时用这张表）

2026-09-13 TCR v5 修正入口优先于下表历史通用 spec：T1 使用专用 masked-query adapter；T2/T3 使用 `tcr-v5:/abs/fusion_checkpoint`（仍为 decoder/global，但池化排除 synthetic）。具体 protocol/gate 分别见 [T1 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)、[T2 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)、[T3 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。不得因字符串不含 `decoder` 就误判该新 spec 为 encoder-only；旧全残基池化不适用于 v5 合成 X。

| 用途 | ✅ 正确 spec | ❌ 错误 spec（消融 only） |
|------|-------------|--------------------------|
| 通用 post-LLaDA 单序列嵌入 | `grammar:decoder:/abs/best.pt` | `bioseq:/abs/best.pt` |
| 显式 global pool | `grammar:decoder:global:/abs/best.pt` | `grammar:decoder:segment:/abs/best.pt` |
| 语义别名 | `bioseq-llada:/abs/best.pt` | `grammar:encoder:/abs/best.pt` |
| T1 binding（β+peptide 联合） | `grammar:decoder:` + `embed_pairs()` | `grammar:encoder:` + MLP |
| T3 deep（β+α 双链） | `grammar:decoder:sequence:/abs/best.pt`（`tcr_pair` 整段 pool） | `bioseq:` 伪双链（每链孤立 ESMC） |
| MINT PPI | `grammar:/abs/best.pt`（默认 decoder + global pool） | `--sep_chains`（legacy 对照） |
| NbBench probing | `grammar:decoder:/abs/best.pt` | `bioseq:` encoder-only |
| AB CDR / pairing 生成 | `GrammarEmbedder` 扩散采样（`argmax` / `gumbel_argmax`） | encoder 特征 + 外接 decoder |

代码入口：`benchmark/common/model_api.py`（`GrammarEmbedder` / `EsmcLladaBioSeqEmbedder` / `build_embedder`）。

**读出不做任何重标定（决策记录 2026-08-31）**：pooled 向量原样送进余弦 / K-means，
**不去中心化、不 z-score、不白化**。曾短暂加过一个 opt-in 的 `post=` spec token，
**已移除**，代码回到 raw-only —— 若在旧命令或文档里看到 `grammar:...:post=pcaw256:<ckpt>`，
那是已废弃写法，现在会被当成路径的一部分而报错。

> ⚠️ **但要知道这个选择的代价**（已量化，引用 §0.1–§0.3 时须带上）：
> 最后一层各向异性极重（mean pairwise cosine 0.99，effective rank 仅 17–20 / 768）。
> 用重标定作探针测出：官方 T3 deep k=200 macro AUROC 还有约 **+0.047** 的空间
> （**大于整张表里任何模型间差异**），且 **raw 读出会扭曲两臂比较** ——
> 两臂差距的成因**不统一**（2026-08-31 配对实测）：raw 下 diffusion 真的领先
> （probe +0.033 = 5.1 sd、ARI +0.0092 = 7.6 sd），其中 **probe 的差距可由几何解释**
> （方差均衡后 BERT 涨 +0.082 vs diff +0.045，差距抹平到 0.7 sd），
> **但 ARI 的差距不能**（白化只是弄坏了 diffusion）。
> 🟠 8B 上胜负与 270m 相反，且每格 n=1 次训练。
> 所以：**表内绝对值是下界；且不能用来判定「哪条臂/哪个预训练目标表征更好」。**
> 不采用重标定的理由与 durable 修法（训练侧加空间约束）见
> [`RESULTS.md`](RESULTS.md) §0.0 缺陷 (f) 与
> [`debug/BERT_REPR_DEBUG.md`](../../debug/BERT_REPR_DEBUG.md)。

### 1.5 Agent 自检清单（每次跑我们的模型前）

- [ ] embedder spec 含 `decoder` / `bioseq-llada`，或使用 §1.4 的专用 `tcr-v5:`；**不是** `bioseq:` 或 `grammar:encoder:`
- [ ] `pool_mode="global"`（或等价的 `grammar:decoder:global:`），不是 `segment_concat`
- [ ] 生成任务走扩散 mask-fill，不是 encoder-only + 贪心
- [ ] 输出 tag 含 `globalfeat` / `llada` / `decoder`，**不是**裸 `bioseq_` 或 `encoder`
- [ ] 主表行标注为 **Ours-BioSeq (post-LLaDA)**，encoder-only 行明确标 **（消融）**
- [ ] strict load 报告 **0 missing / 0 unexpected**（证明真加载了我们的 ckpt）

### 1.6 已知历史错误（Agent 读到这些 tag 时不要当 headline）

| 历史 tag / 目录 | 问题 | 正确处理 |
|----------------|------|----------|
| `ours_biseq_mlp_*` | T1 encoder-only | 仅消融对照；headline 用 `ours_globalfeat_*` |
| `bioseq_grammar_v2_*`（无 `-llada`） | T2/T3 encoder-only | 仅消融；headline 用 `bioseq_llada_*` 或 `grammar_tcrpair_seq_*` |
| `ours_postllada_*`（T1 分段版） | post-LLaDA 但 segment concat | 对照脚注；headline 用 `ours_globalfeat_*` |
| `grammar_best_sep_t3000` | MINT PPI 分链拼接 | 对照脚注；headline 用 `*_globalfeat_t3000` |
| `bioseq_esmc*_encoder` | NbBench encoder-only 误跑 | 不作 headline；用 `ours_*_llada` |

---

## 2. 文档地图（Agent 读什么、写什么）

### 2.1 阅读顺序（进入任务时）

```
1. 本文件 PROJGUIDE.md              ← 你现在在这里（原则 + 口径 + §0.1.1 范围 + §0.2.2 论文值优先）
2. downstream/tasks/<TASK>.md       ← 【主入口】做什么、怎么跑、怎么填，自包含执行手册
3. benchmark/README.md §0/§3        ← 评测范围与逐任务锚点（唯一登记表）
4. RESULTS.md §0                    ← 唯一权威结果表（填数字前核对列名）
5. PROGRESS.md                      ← 过程记录（每次推进追加 changelog）
```

在范围内的六份任务文档：
`tasks/TCR_T1_BINDING.md` · `tasks/TCR_T2_CLUSTERING.md` · `tasks/TCR_T3_REPRESENTATION.md` ·
`tasks/TCR_T4_GENERATION.md` · `tasks/AB_CDR_INFILLING.md` · `tasks/AB_LIGHT_CHAIN_PAIRING.md`

> **仅历史/复现用，不作现行口径**（均带 `⚠️ ARCHIVE` 头部）：
> `downstream/AB_TCR_EVAL_SUMMARY.md` · `downstream/downstream.md` · `TCR_BENCHMARK_DESIGN.md` ·
> `TCR_GENERATION_BENCHMARK.md` · `TCR_TASK_TAXONOMY.md` · `nbbench/*` · `mint_tasks/*`。
> 这些文档与任务文档冲突时，**以任务文档和 `RESULTS.md` 为准**。

### 2.2 写入顺序（每次推进后）

Agent 完成任意一步（提交作业、跑通脚本、发现 blocker、产出数字）后，**在同一轮对话内**按优先级更新：

| 优先级 | 文件 | 更新什么 |
|:------:|------|----------|
| P0 | `downstream/tasks/<TASK>.md` | 协议、数据根、入口、gate、缺陷、执行手册变更 |
| P0 | `benchmark/README.md` §0/§3 | 范围变更、任务锚点/出处层变更 |
| P1 | `benchmark/RESULTS.md` §0 | 排行榜数字（必须可溯源到 `outputs/`；按 §0.2.2 论文值优先排版） |
| P1 | `benchmark/PROGRESS.md` | 追加 changelog 条目（日期 + 做了什么 + 结论） |
| P2 | `PROJECT_PROCESS.md` | Volc submit/cancel 记录（规则见 `.cursor/rules/volc-train-task-log.mdc`） |

> 完整映射见 [`.cursor/rules/downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。
> **不要**再往 `AB_TCR_EVAL_SUMMARY.md` / `downstream.md` / `TCR_BENCHMARK_DESIGN.md` /
> `TCR_GENERATION_BENCHMARK.md` 写新内容——它们已归档，新协议一律写进 `tasks/<TASK>.md`。

### 2.3 实时更新规则（Live Update Rule）

> **禁止**等所有 GPU 作业 Success 后才一次性补文档。

| 时机 | 必须更新 |
|------|----------|
| 做出口径决策（如 globalfeat 取代 segment_concat） | 立即写对应 `tasks/<TASK>.md` §4 协议 + §7 缺陷 + 本文件若需修订 |
| 提交 Volc 作业 | 立即记 task_id + yaml 名 + 范围 → `PROGRESS.md` + 执行计划状态表 |
| 作业 Running / 部分落盘 | 更新状态表（`running` / `partial`），已产出的格标 `done` |
| 作业 Success / Failed | 更新状态 + 回填数字或 blocker 原因 |
| 发现口径错误（如误用 encoder-only 当 headline） | 立即标注旧结果「不作 headline」+ 排期重跑 |

**数字纪律**：`RESULTS.md` 主表中的每个数字必须能指到 `outputs/` 或 `output/downstream_generation/` 下的具体 `metrics.json` / `fewshot.json` / `*_summary.txt`。**禁止编造。**

### 2.4 结果呈现规约（Ours 永远置于对比表末尾）

> **硬性呈现规则**：任何「我们的模型 vs baseline」的对比（排行榜表、对比图、canvas 看板、论文/报告表格），**我们的模型（Ours-BioSeq，含全部 checkpoint 与消融/control 行）一律排在所有 baseline 之后** —— 放在表格最末的 Ours 行块、或图例的最后一项，**绝不与 baseline 按分数穿插排序**。

| 呈现形态 | 排法 |
|----------|------|
| 行式对比表（方法作行） | baseline 各行在上（可按主指标排序），**Ours 全部行作为一个整块置于表末** |
| 列式对比表（方法作列） | baseline 列在左，**Ours 作最右列** |
| 柱状 / 折线图 | baseline 系列在前，**Ours 系列为图例最后一项 / 最后一个 category** |
| Ours 内部次序 | 多 ckpt 按规模或语料排序；**消融 / control（encoder-only、segment-concat）紧跟 headline 之后，仍在表末 Ours 区块内** |

**目的**：让读者一眼把「我们的方法」与所有对照分离、跨任务一致定位 Ours，避免 Ours 淹没在 baseline 中间。**即使某个同协议本地结果更高也不上移**，靠加粗/高亮而非排序位置；跨协议结果则禁止作领先结论。适用于 `RESULTS.md`、`tasks/<TASK>.md`、canvas 看板与一切对外对比呈现。

---

## 3. 执行分工（本地 vs Volc）

2026-09-13 用户补充：简单任务优先本地；最新授权 others 全部完成后将剩余仍 Queue 的 TCR 任务也本地接续、持续监控，见 [审计 §9.24](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。执行地点改变不改变完整评测协议；迁移前核对 ID/名称、取消并确认 Killed，已云端 Running 的任务保留，原常驻进程不动。短任务与接续队列通过前置状态和锁互斥，失败不自动减数据或覆盖重跑。

| 类型 | 在哪里跑 | 示例 |
|------|----------|------|
| 轻量（<5 min GPU） | 本地 | smoke、汇总脚本、数据准备、官方预计算结果解析 |
| 重量（预计/实测较长或资源占用大） | **Volc 提交** | 大规模嵌入抽取、长时全量评测、生成 n>100；小规模完整任务按实际耗时判断 |
| 提交方式 | no-proxy 封装 | `/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/<yaml>` |

结果统一落 **vepfs 挂载盘**（`benchmark/outputs/`、`output/downstream_generation/`），本地与 Volc 互读。

---

## 4. 下游任务一览（服务于主模型的评测矩阵）

> 范围按 §0.1.1：**只有 §4.1 TCR 与 §4.3 AB 在范围内**；§4.2 MINT PPI、§4.4 已冻结任务仅作存档。

### 4.1 TCR（`benchmark/`，四大方向）

| 任务 | 主基础 | 我们的模型 headline 口径 | 当前状态 |
|------|--------|--------------------------|----------|
| T1 Binding | NM2025 | `grammar:decoder:global:` + MLP | post-LLaDA ✅；globalfeat ⏳ |
| T2 Clustering | NAR-GAB + TCREmbedding | `bioseq-llada:` / globalfeat | post-LLaDA ✅；globalfeat ⏳ |
| T3 Representation | SCEPTR | `grammar:decoder:sequence:`（tcr_pair） | deep+broad ✅ |
| T4 Generation | TCRT5 | 扩散采样 adapter | A/B ✅；C 部分 pending |

### 4.2 ⛔ MINT PPI（`mint_tasks/`，已冻结）

| 任务 | 我们的模型 headline | 历史状态 |
|------|---------------------|----------|
| ⛔ HumanPPI + Bernett | `grammar:…/best.pt` global pool | baseline 6/6、globalfeat 6/6 曾跑通；**已冻结，不再更新** |

### 4.3 AB 抗体（`downstream/` 根目录）

| 任务 | 我们的模型 headline | 当前状态 |
|------|---------------------|----------|
| CDR infilling（SAbDab Kong + SAb23H2） | 扩散解码 `argmax`，报 `iter=1` | 见 `RESULTS.md` §0.5 |
| Light pairing（OAS holdout500 prompt3） | 现行 Ours 长度协议和验收 gate 见 [AB pairing §4.3 / §8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)；不得沿用旧“必须长度先验”要求 | 新协议全量、预检与历史数字分列于 [RESULTS §0.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)；验收和归因边界见任务入口 |
| Ours GDPa1 / Specificity / m396 | 原生 antibody record、post-LLaDA global pool；不是默认 TCR embed_pairs | 新授权附加测评；适配/验收/待提交状态见 [原生任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md) |
| Humanization | — | 本轮仍排除 |

### 4.4 ⛔ 已冻结（不在范围内，仅存档）

| 任务 | 目录 | 冻结原因 |
|------|------|----------|
| ⛔ NbBench 纳米抗体 11/12 任务 | `benchmark/nbbench/` | 纳米抗体，与 AB + TCR 主线分开 |
| ⛔ STRING 90/90 PPI | `benchmark/ppi/` | 通用蛋白 PPI，非免疫受体 |
| ⛔ FLAb 抗体属性回归 | `downstream/flab/` | 锚点是 MINT 而非 AB 主线 |

---

## 5. Agent 常见错误与纠正

| 错误 | 后果 | 纠正 |
|------|------|------|
| 用 `bioseq:` 填主表「Ours-BioSeq」行 | 只测 ESMC，虚报/低估 LLaDA 贡献 | 改 `grammar:decoder:` 或 `bioseq-llada:` 重跑；旧结果标「消融」 |
| 把 `*_postllada_*`（分段）当 headline | 违反 global pool 协议 | 用 `*_globalfeat_*` tag 重跑 |
| 等作业全完成才更新 MD | 团队看不到在途/blocker，重复劳动 | 提交即记 task_id；部分落盘即更状态表 |
| 用自制脚本冒充官方 baseline | 对比不公平，审稿风险 | 只用 `baselines/wrappers/` + 官方权重 |
| 编造未落盘的数字 | 破坏可复现性 | 只写 `outputs/` 里真实存在的 metrics |

---

## 6. 主 checkpoint 与 PRIMARY 选取

> ⚠️ **2026-09-02**：PRIMARY 已从作废的 `grammar_v2_esmc600m_cmp500k_llada/best.pt` 改为 immune fusion。旧三条 grammar_v2 ckpt **VOID**。

Headline（锁定，不追后续 step）：

```
output/protein_esmc_llada8b_bert_immune/checkpoint-43000          ← 8B BERT 表征
output/protein_esmc_llada8b_diffusion_immune/checkpoint-45000    ← 8B diffusion 表征+生成
output/protein_esmc_llada270m_bert_immune/checkpoint-49000
output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000
```

- 生成任务（T4 / CDR / pairing）**只跑 diffusion**。
- 任何「Ours」主榜行必须注明 **post-LLaDA + global pool**（表征）或 fusion `_denoise`（生成）。
- v3 七源 ckpt 另标，≠ 上列 270m。

---

## 7. 快速命令模板

> ⚠️ **§7 示例里的 `grammar_v2_* / integrated_llada` 路径是历史模板，ckpt 已 VOID。**
> 现行 headline 用 §6 的 immune fusion 目录（8B BERT `@43000` / 8B diffusion `@45000` / 270m BERT `@49000` / 270m diffusion `@42000`）。
> 范围内任务的现行命令以 [`../tasks/`](../tasks/) §8 为准，不要照抄下面的 grammar_v2 `CKPT=`。

### 7.0 一键全下游编排器（迭代模型即用）

`scripts/downstream/run_all_downstream.py` 是"给一个 ckpt → 跑/收全下游"的单一入口，统一 post-LLaDA `grammar:decoder:global:` 口径，是 8 个 `eval_jobs/eval_integrated_*.yml`（由 `scripts/downstream/gen_eval_integrated_ymls.sh` 生成）背后的唯一命令来源。

```bash
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada/best.pt
# 打印每个任务族的确切命令（校验接线，不跑）
python scripts/downstream/run_all_downstream.py plan --checkpoint $CKPT
# 聚合已产出的 metrics → summary_<run>.json（可迭代对比）
python scripts/downstream/run_all_downstream.py collect --checkpoint $CKPT
# 本地实跑选定任务族（重活优先走 Volc YAML）
python scripts/downstream/run_all_downstream.py run --only t3,flab --checkpoint $CKPT
# Volc：8 族一键提交（各作业等 best.pt 后自动跑；task id 记入 output/downstream_generation/eval_integrated_task_ids.tsv）
bash scripts/downstream/submit_eval_integrated.sh
# 或单族：volc ml_task submit --conf eval_jobs/eval_integrated_t3.yml

# 全族跑完后聚合 + 生成回填报告
python scripts/downstream/run_all_downstream.py collect --checkpoint $CKPT
python scripts/downstream/fill_ours_from_summary.py --checkpoint $CKPT --report
# 审阅 C1_FILL_REPORT.md 后：--apply-paper --dry-run → --apply-paper → tectonic 重编译
# 监控 train + 8 eval 状态（非交互 JSON，避免 volc TUI 卡住）
bash scripts/downstream/watch_integrated_c1.sh
bash scripts/downstream/watch_integrated_c1.sh --loop 5m
```

任务键：`t1`(binding) `t2`(clustering) `t3`(representation) `t4`(generation) `ab`(CDR/pairing) 为范围内；
⛔ `mint`(PPI) / `flab` / `nbbench` 已按 §0.1.1 冻结 —— 编排器仍能接受这三个键（代码未删），但**不要用它们出数**。

并发单族作业分别写 `summary_<run>_<task>.json`，避免互相覆盖；全部作业结束后，不带 `--only` 执行 `collect`，生成规范的 `summary_<run>.json`。FLAb 汇总主指标为论文对齐的 nested 10×5-fold outer mean R²；Spearman 仅保留在单数据集 artifact 中作为诊断指标。FLAb 主编排通过 `run_flab_parallel.py` 使用 8 个 joblib worker；每个 outer fold 还用 sklearn `Pipeline(memory=...)` 复用 13 个 Ridge α 完全相同的 fold-local `PowerTransformer` 结果。二者都只改变执行调度/缓存，不改变数据、折分、预处理拟合边界、α 网格或指标。YAML 生成器默认将长且有状态的 AB/FLAb 设为 non-preemptible；其余短任务仍可抢占。MINT collector 只展示当前五任务：HumanPPI、YeastPPI、Gold-standard PPI、MutationalPPI、SKEMPI；`Pdb-bind` 不再进入默认汇总。

AB light-pairing 生成每完成一个 heavy batch，会原子写 `<final_stem>.progress.pt`，其中同时保存已生成行与 Python/NumPy/Torch/CUDA RNG 状态。相同配置重启会精确续跑；配置、数据文件元信息或 checkpoint 路径/文件元信息不一致时拒绝复用。只有全部 heavy×`num_seqs` 完成后才原子发布公开 `_n<num_seqs>.csv`；progress 在请求的 comp-chain scoring 成功后（或显式 `--skip-eval` 完成后）删除，因此评分阶段被抢占也不会触发全量重新生成。

### 7.1 单任务模板

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc600m_cmp500k_llada/best.pt

# ✅ 正确：post-LLaDA probing
$ENV/bin/python benchmark/tcr_binding/run_nm2025.py --method embed \
  --embedder grammar:decoder:global:$CKPT --head mlp --tag ours_globalfeat_esmc600m

# ❌ 错误：encoder-only（仅消融时可用）
# --embedder grammar:encoder:$CKPT  或  --embedder bioseq:$CKPT

# Volc 提交（重活）
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
/root/.codex/skills/volc-no-proxy/scripts/volc-no-proxy.sh ml_task submit \
  --conf eval_jobs/eval_tcr_binding_nm2025_globalfeat.yml
```

---

## 8. Changelog

| 日期 | 变更 |
|------|------|
| 2026-07-07 | 初版：明确主模型中心论、post-LLaDA 强制口径、global pool headline、Live Update Rule、文档地图、embedder 速查表 |
| 2026-09-02 | 新增 §0.2.2「论文值优先」并**反转**旧 `[R]` 优先规约（含 T1 original-only 特例）；§2.1 阅读顺序改以 `tasks/<TASK>.md` 为主入口、列出归档文档；§2.2 写入阶梯把任务文档提到 P0 并移除对已归档 `AB_TCR_EVAL_SUMMARY.md` / `downstream.md` 的写入要求；§6 PRIMARY 从 grammar_v2 改为 immune fusion；§7 标注 grammar_v2 命令模板 VOID；配套新规则 `.cursor/rules/downstream-doc-sync.mdc` |

2026-09-24 范围同步：用户授权的 BioMap ΔG 官方 baseline 首轮执行以 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md#0-评测范围2026-08-29-收窄强制) 为准；运行状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。

<a id="report-integrity"></a>

## 2026-09-30 报告与 CPU 验证约定

历史协议和原始成绩保留。派生报告必须写入新的目录，已有目录拒绝覆盖；报告记录读取的文件及 SHA256。预训练暴露按实际 checkpoint、训练 manifest、审计 registry/scope 和评测 cohort 的哈希链判断，缺环为 `unknown`。新清理语料的审计不能自动授予旧 checkpoint clean 状态；下游 split 中的 unseen 也不等同预训练 unseen。当前结果解释见 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，被动审计配置见 [配置清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/configs/integrity_20260930.json)。

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
EVAL_PY=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python
export CUDA_VISIBLE_DEVICES='' USE_TF=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
$EVAL_PY /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_nm2025.py
$EVAL_PY /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/summarize_tcr_representation_paper6.py
$EVAL_PY /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/audit_result_integrity.py --config /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/configs/integrity_20260930.json
$EVAL_PY /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/check_source_release.py --smoke-copy
```

前三个命令默认输出到带 UTC 时间戳的 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_reports/`。T1 审计命令是 `$EVAL_PY /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/audit_baseline_provenance.py --summary-dir <上述新T1报告的绝对路径> --strict`；它验证论文主列与不可替补规则，缺失且已披露的论文证据是 pending，不要求重跑已冻结任务。

CPU 依赖单独记录在 [requirements-cpu.txt](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/requirements-cpu.txt)：本次验证环境为 Python 3.12.12、CPU 可见的 PyTorch 2.8.0。新环境先从 PyTorch CPU wheel 索引安装 `torch==2.8.0`，再安装该依赖文件；不把训练用 pyproject 的 GPU 依赖装入该环境。本轮使用既有环境验证，没有安装或升级活动环境。官方 baseline 的外部代码、权重、数据仍需按各任务手册准备；源码副本检查不声称附带这些资产。

源码检查使用 `git ls-files --cached --others --exclude-standard`，覆盖尚未提交的新修复；复制可见源码到临时空目录，在离线 CPU 环境导入核心模块并跑合成回归。它同时检查 vendor/data/weights 仍被忽略。验收后应把白名单中的第一方源码与本次变更一起提交；仅修改 .gitignore 不等于已经提交。
