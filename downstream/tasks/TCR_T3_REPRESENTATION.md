# TCR T3 Representation

> 2026-09-30 历史作业配置整理：已归档 YAML 的位置见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)；文中的旧提交命令及原路径文字保留作历史记录，复用配置前请先按索引定位。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.3。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | T3 |
| 锚点论文 | Nagano et al., SCEPTR, Cell Systems **2025**, [10.1016/j.cels.2024.12.006](https://doi.org/10.1016/j.cels.2024.12.006)；2024 预印本的 Table SI / III.1 是既有登记来源，正式版对应 Fig.1 / STAR Methods；原文复核见 §7.2 |
| RESULTS 锚点 | `RESULTS.md` **§0.3** |
| 主指标 | few-shot per-epitope nearest-neighbour AUROC ↑（deep track k=200 macro） |
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_representation_paper6/`（deep：6 pMHC，universe 25,816） |
| 我们的入口 | `tcr_representation/run_paper6.py --method embed`（主）+ `tcr_representation/run.py`（broad） |
| 可引用性 | ⚠️ 当前本地协议与论文存在实质失配，不作论文复现或纠正后 v5 headline；详见 §7.2，优先于旧“校准兼容”表述 |
| 最后更新 | 2026-09-13（正式论文、作者数据代码与本地产物交叉核查） |

---

## 1. 任务定义与范围

冻表征、不训头。给定支持集，用 nearest-neighbour 判断 query TCR 属于哪个表位。主协议是 SCEPTR 口径的 few-shot NN AUROC；24-way linear probe 只是辅报。

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | 表征检索 |
| Ours-BERT | yes | 表征检索 |

### 1.2 不在本文档范围内的相邻任务

- T1 结合分类（训 MLP 头）：见 `TCR_T1_BINDING.md`。旧 taxonomy 曾把 few-shot 写成 T1 子协议，**已作废**。
- T2 无监督聚类。
- 24-way probe：辅，不得当 headline。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| Table SI III.1 deep few-shot | 6 pMHC，k∈{1,2,5,10,20,50,100,200} | per-pMHC AUROC 再 macro | SCEPTR / TCRdist 领先 PLM | 是（主协议；registry 只转录了 k=200 的 36 行 `[P]`） | 其它 shot 无 `[P]` 对照 |
| broad 24 表位 | 更大表位集 | 同上 | 趋势同 | 是（辅表，无完整 `[P]`） | — |
| 输入含 V/J | SCEPTR 吃 TRAV/TRBV；TCRdist 再加 J | — | 这是论文自己的协议 | 是，但**必须分层** | 我们只喂 CDR3β+α |
| 线性探针 | 非主表 | AUROC/Acc | — | 辅报 | 监督足够时人人都能分 |

### 2.2 指标定义（逐字对齐）

| 指标 | 论文定义 | 我们的实现 | 已知差异 |
|---|---|---|---|
| few-shot NN AUROC | 每表位、每 seed、每 k；macro over pMHC | `common/fewshot.py`；deep：100 seeds，`max_ref_pool=2000` | 36 行校准 mean \|Δ\| 0.006–0.057，**几乎全为负**（共享协议差，不是单方法 bug） |
| 距离 | SCEPTR native cdist；TCRdist 矩形距离；PLM 余弦 | 同 | 见输入字段 |
| pooling | 我们：整条 record 一次 global mean-pool | `pair_pooling="sequence"` | 旧 docstring 曾写错成分段 pool |

### 2.3 论文未公开的关键细节

负偏候选未隔离：universe 25,816 + `max_ref_pool=2000` 截断、seed 数、background pool 组成。不能据此保证改变协议后方法排序不变；历史「高度一致」仅描述已观察到的局部结果。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| SCEPTR 代码+权重 | yes | MIT | `.../baselines/sceptr` commit `8c1fa02ec4`；pip `sceptr==1.2.0` 自带 `state_dict.pt` | 见 manifest | 直连 / 镜像 | 已落地并跑通 |
| tcrdist3 | yes | — | `.../baselines/tcrdist3` + pip `tcrdist3==0.3` | — | pip | 已跑通 |
| TCR-BERT 权重 | yes | Apache-2.0 | 本地控制用 HF `wukevin/tcr-bert-mlm-only`；论文指定另一 checkpoint，须分臂，见 §7.2 | HF snapshot | `HF_ENDPOINT=https://hf-mirror.com` | 本地控制已跑通，不等于论文复现 |
| ProtBERT | yes | — | HF `Rostlab/prot_bert` | — | hf-mirror | 已跑通 |
| 官方评测脚本 | 部分 | — | 我们 `run_paper6.py` 复现 Table SI 协议 | — | 本地 | `[L]` 重建 |

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| 数据根 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/data/tcr_representation_paper6/` |
| split | `scripts/prepare_tcr_repertoire.py` + paper6 准备脚本 |
| 入口主 | `python tcr_representation/run_paper6.py --method embed --embedder grammar:decoder:global:<ckpt> --columns cdr3b cdr3a --tag <tag>` |
| 入口 broad | `python tcr_representation/run.py --method embed --embedder ... --columns cdr3b cdr3a --tag <tag>` |
| 对照 | `--method sceptr` / `tcrdist` / `embed --embedder {esm2_150m,protbert,tcrbert}` |
| Volc | `eval_jobs/eval_immune_*_repr.yml` t3 stage；`eval_paper6_*.yml` |
| 输出 | `outputs/tcr_representation_paper6/` |

**无表位泄露**：`grammar_pair_embed` 传 `peptide=None`，渲染 `order=("beta","alpha")`，表位不在 record 里。V/J 我们从不供给。

### 4.2 口径切割线（⚠️ 引用前必读）

以下分层描述**本地旧输入**，不是论文输入表；论文 PLM 的链范围、版本及 TCR-BERT 权重/层数有重大区别，见 §7.2。

| 分组 A | 分组 B | 为什么不可同表 | 证据 |
|---|---|---|---|
| SCEPTR（β+α+V）/ TCRdist（β+α+V+J） | 我们 / ESM2 / ProtBERT / Lev / k-mer（β+α） | 输入字段不对等 | `RESULTS.md:656`；审计 T3 §1 |
| TCR-BERT（β only） | 双链方法 | 少一个字段 | 同上 |
| few-shot NN | 24-way probe | 监督强度不同；probe 人人能分 | `RESULTS.md:704` |
| 论文 k=200 `[P]` | 其它 shot 的本地曲线 | registry 只录了 k=200 | 审计 T3 §4 |
| raw 嵌入 | 任何重标定后的探针 | 已决定不采用重标定；探针只测代价 | `PROJGUIDE.md` §1.4 |

---

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

1. `[P]` 主榜行 = Table SI k=200 各方法最佳已发表值。
2. 本地 SCEPTR/TCRdist 跑数是 `[L]`/`[R]` 附行（且系统性偏低），不取代 `[P]`。
3. 源码 clone 失败（tcr-bert dest）不得删 HF 权重已跑通的行。
4. 输入字段写在「输入」列，禁止扁平排名当能力差。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| SCEPTR | `[P]` 主；本地 `[L]` | Table SI k=200，36 行 registry | yes 权重 | yes | 本地全 6 pMHC ≤ 论文；多 V 基因 |
| TCRdist3 | `[P]` + 本地 | 同上 | yes | yes | 多 V+J；5/6 pMHC 负偏 |
| CDR3-Levenshtein-NN | `[P]` + `[C]`/`[L]` | 同上 | n/a | yes | 同输入档（β+α） |
| k-mer(3) | 本地 `[C]` | — | n/a | yes | 论文 SI 有组成对照 |
| ESM2-T6-8M（论文） / ESM2-150M（本地） | `[P]` / `[L]` 分组 | Table SI 对应 8M，不是 150M | 本地 150M HF 权重 | 150M 已跑；论文 8M 同输入复跑未验收 | 完整链 vs CDR3αβ，不能视作同模型校准 |
| ProtBERT | `[P]` + `[L]` | Table SI | HF | yes | 负偏 |
| TCR-BERT MLM-only | `[P]` + `[L]` | Table SI | HF 权重 yes；git dest `DL_FAIL` | yes | 只 CDR3β；勿用分类微调权重 |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | `RESULTS.md` §0.3 |
| 任务锚点 | `benchmark/README.md` §3.1 T3 行 |
| changelog | `benchmark/PROGRESS.md` |
| Volc | `PROJECT_PROCESS.md` |
| 本任务协议 | **本文件** |

原始产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_representation_paper6/`

---

## 7. 已知缺陷与不可比口径

2026-09-30 汇总代码与 §7.2/§7.3 对齐：论文 `esm2_t6_8m` 和本地 `esm2_150m` 是不同身份；按方法名自动标 `compatible` 的逻辑已撤回。已知数据规模或模型差异为 mismatched，其余缺少成员/输入/选模证明的比较为 unknown，均不输出差值/排名。新 [T3 报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/t3_paper6/_summary.json) 保留来源 SHA，不改旧 fewshot.json。

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | 输入字段不对等 | 已披露，表带「输入」列 | 「低于 SCEPTR 0.06」不能当纯模型差 | 审计 T3 §1 |
| b | 同输入档仍低于 Lev/k-mer | 未修 | 不能用信息不对等开脱 | 审计 T3 §1 Restated |
| c | 本地 vs 论文系统性负偏 | 未隔离 | 校准是有偏复现 | 审计 T3 §4 |
| d | `columns` 曾低估 SCEPTR/TCRdist 输入 | 已修并回填 46 个 artifact | 旧 json 的 `columns` 不可信，看 `input_fields` | 审计 T3 §3 |
| e | 各向异性 / 不重标定 | 历史诊断、口径未改 | 可影响距离几何；不是性能下界的证明 | `RESULTS.md` §0.0(f) |
| f | 表位作为模型输入 | 当前 Ours 接线未发现 | 不等于排除预训练重叠或所有数据泄露 | 本节 §7.1 |

### 7.1 2026-09-13：T3 relation、support/query 与 v5 输入审计

**结论：编码器不需要 `<binding>`，但 few-shot 需要 support 的已知表位标签来选参考 TCR。** 不能把合法的 support 标签使用与将 query 真值输入模型混为一谈。本轮审计当前本地实现，不重新宣称与论文逐项等价；没有训练或真实模型重评。

**实际计算链与标签边界**

- [包装脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_repr.sh:72) 同时运行 deep / broad，Ours 默认 `cdr3b cdr3a`。[grammar_pair_embed](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/fewshot.py:104) 只传 β、可选 α、`peptide=None`；不传 epitope、binding 真值、MHC 或 V/J。
- 共享 [record 构造](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py:676) 与 [renderer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:419) 给出无条件结构：`<prots><null><protd><unknown><prots><tcr> CDR3β . CDR3α <protd>`。`<unknown>` 是固定无上下文前缀，不是预先给定正类，也不是 T1 待预测 recognition；不应机械套用带 peptide 的 T1 查询模板。
- 每条 receptor 独立产生冻结表征。对表位 E，支持集为已知结合 E 的 k 条 TCR；query 分数为它到支持集的最大余弦相似度（默认 1-NN），然后用 query 真值计算 AUROC。**主任务不训练分类头，也不预测 relation token**；broad 默认 `--protocol both` 另跑辅助 logistic probe，不能说整个包装流程完全不训头。
- broad 用 train 标签选择 support，test 作为 query；默认评估类别由两池标签交集决定。deep 的标签用于构造 episode、采样正类 support 和评估正负，不是完全不读 query 标签的 benchmark；**给定固定 episode/support 后，query 真值不进入表征或相似度计算**。
- [deep 评分](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/fewshot.py:350) 每轮删除抽中的 support 行后才算 AUROC；不同 seed 允许交换角色。当前表位的负例包含背景和其余目标表位行，不只是 `background_pool.csv`。这是 benchmark 的负例定义，不是逐一实验验证所有交叉配对不结合。
- cosine 仅逐行 L2 归一化；辅助 [probe](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/metrics.py:481) 的 StandardScaler 与 LogisticRegression 在 train 上 fit（固定 C=1），未发现 query-fit scaler/PCA。此结论不替代历史人工选 ckpt/超参审计。

**真实数据检查与未解决问题**

| 检查项 | 本轮观察 / 后续要求 |
|---|---|
| v5 运行时输入 | 当前共享 embedder 直接把 CDR3 当链，未接入 T1 已实现的运行时补全。后续需独立的无 epitope adapter；不能把当前输入称为完整 v5 训练范式对齐。 |
| 补全后的 pooling | 当前 global pool 只用 `residue_mask`；尚无 synthetic 残基。接入补全时必须同时实现 synthetic 排除和真实 CDR3 范围验收，不能只换字符串。 |
| deep 成对身份 | target 3,941 / background 21,875 行；各自 αβ pair 重复为 0，交集为 0。运行入口不再断言这些性质；更换数据需重新 gate。 |
| broad 成对身份 | train 6,887 / test 1,722 行，αβ pair 跨集合交集为 0；集合内部重复多出 2 / 1 行，对应 2 / 1 个同 αβ、多表位组。应明确交叉反应/多标签与单标签 OVR 的处理，不能直接判为标注错误或自动删掉。 |
| β-only 的身份边界 | broad train/test 有 274 种共有 β；deep target/background 有 163 种共有 β。成对不重叠不能外推为 β-only 不重叠。deep 仅排除 support 行，不排除所有同 β 行；实际各 episode 重叠及其分数影响待统计，不能将这 163 种称为实测 support-query 泄露数。 |
| 缺链 / 输入验证 | broad train 有 2 行 α 为空，其余三份数据均无空 α；四份均无空 β，非空 CDR3 经当前 sanitizer 均不变。空 α 会退为 β-only，应披露而非填入真值。适配器仍有缺 α 列静默降级、β 必选等宽松行为，需严格 schema gate。 |
| 尚未覆盖 | 预训练数据重叠、近重复、baseline 训练标签范围、输入档逐一一致性、ckpt/缓存完整身份、真实 GPU 前向与性能；不能将本轮理解成全套通过。 |

**CPU 验证范围（无权重加载）**：通过 deep `_embed_vectors` 调用 spy embedder，将 8 条真实样本的 peptide/MHC/V/J/clonotype 改值后，实际 `embed_pairs` 参数不变；这是入口边界测试，不是真实 hidden-state 测试。另调用生产评分函数：broad 8 个合成 episode 在固定 support 下反转 query 真值，打分全部不变；deep 8 个合成 episode 重建 RNG，确认评分数组恰好排除抽中 support 行且分数匹配。真实 CSV 全量身份/缺链统计见上表。本轮未重复 T2 的原生 tokenizer/collator 测试，也未改变代码或数据。

历史性能仍保留在 [RESULTS §0.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)，不作纠正后 v5 T3 headline；本轮没有证明任何修正必然提分。关键日志与 SHA256 见 [TCR 审计 §9.10](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

---

### 7.2 2026-09-13：原论文 baseline 复核——撤回“当前实现等同论文”

**来源与效力**：已读 [正式发表 PDF](https://discovery.ucl.ac.uk/10205427/1/Nagano2025.pdf) 的主实验、STAR Methods（e2–e4），并对照 [arXiv v2 方法及补充分析](https://arxiv.org/html/2406.06397v2#A1)、[作者数据 notebook](https://github.com/yutanagano/tcrlm/blob/70ef27939051937a8a5c1d8edd52b51ea4326706/notebooks/data_preprocessing_labelled.ipynb) 和两份官方 TCR-BERT 模型卡。本节优先于本文、旧审计和代码 docstring 中的“论文就是 CDR3-only / TCR-BERT 官方 β-only / 同输入已复现”描述。未取得完整原始评测 episode/所有历史模型 snapshot，不能宣称已重现作者全部实验。

**论文的主问题仍是 TCR-only 的最近邻表征评测**：六个模型不输入待测 peptide/MHC 序列或 query binding 标签；表位标注用于选 reference 与评分。统一取 query 到同表位参考 TCR 的最近距离，计算 AUROC。主评测不为每个任务重新微调骨干，不代表所有 checkpoint 的历史训练都无监督。

| 模型 | 论文使用的受体输入与读出 | 当前本地差异 / 核查要求 |
|---|---|---|
| CDR3 Levenshtein | α、β CDR3 编辑距离之和；无需训练 | 输入级别一致，但数据过滤及 episode 不同，不能凭分数接近声明 exact |
| TCRdist | 配对 TCR 的序列比对距离，包含胚系相关区域信息 | 当前官方 tcrdist3 adapter；参数、数据与版本仍需锁定，不能仅以收到 V/J 字段证明所有字段参与距离 |
| SCEPTR | αβ 六个 CDR；联合编码，CLS 读出；无标签 MLM＋autocontrastive 预训练 | 当前 native wrapper 从 V/CDR3 恢复 CDR1/2；不是只输入两条 CDR3，也不是输入完整链的所有残基 |
| TCR-BERT | CDR3α、CDR3β 分别编码，**第 8 层**氨基酸 token 均值后拼接；指定 `wukevin/tcr-bert` | 历史 artifact 是 β-only；当前 adapter 为 `tcr-bert-mlm-only`、末层且 attention-mask pooling 包含特殊 token。链数、checkpoint、层数及池化均非原文 recipe |
| ESM2 | **T6 8M**；用 CDR3＋V/J 经 **Stitchr 重建完整 α/β 链**；逐链末层氨基酸均值、拼接 | 当前主表为 **150M＋CDR3αβ**，不是论文 ESM2；attention-mask pooling 还含特殊 token |
| ProtBert | Stitchr 重建完整 α/β 链；逐链末层氨基酸均值、拼接 | 当前主表仅 CDR3αβ；需与重建长链、special-token 处理一起对齐 |

上述是原文的模型/输入定义；具体归一化、距离及特殊 token 实现仍应通过作者版本的逐步数值校验，不从“都是 embedding”推断完全相同。**完整链是 V/J/CDR3 重建结果，不等同于测序提供的天然全长，也不等同于我们的 v5 随机框架补全。**

**训练方式不能混为一谈**：论文主 NN 评测冻结已有模型；另有线性 SVC 辅助实验（每个表位参考正例＋固定的 1,000 条背景负例），不是本地 24-way logistic probe。另一个独立 SCEPTR 微调实验用每表位 200 条、共 1,200 条训练 TCR，按研究来源隔离 test，结合 MLM / supervised contrastive，只解冻最后 self-attention 层、validation early stopping，并非全参数微调。

[TCR-BERT 官方模型卡](https://huggingface.co/wukevin/tcr-bert/blob/main/README.md) 明确主模型含 PIRD 抗原分类训练；[MLM-only 模型卡](https://huggingface.co/wukevin/tcr-bert-mlm-only/blob/main/README.md) 对应另一版本。因此不能把论文主表一律称作“所有模型从未用过抗原监督”，也不能仅凭用过监督就断言测试泄露。应并列论文指定 checkpoint 的复现臂与 MLM-only 自监督控制，分别核验训练/测试 overlap；不擅自替换成更弱输入再称原 baseline。

**已确认的数据/统计失配**

- 正式论文主数据为 **7,168 条、864 个 pMHC**；主评测排除原始 10x 白皮书数据，另做纳入处理后 10x 的扩展实验。本地 **25,816** 不是这一主数据集。作者 notebook 使用 `vdjdb_20240128.tsv`，而本地 metadata 指向 `Database_2023_10/vdjdb.txt`；不能继续引用 metadata 中未经核实的“论文 2023-06-01”作为对齐证明。
- 作者 notebook 的 `remove_10x_data` 明确按来源 URL 排除；当前 [准备脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/prepare_tcr_representation_paper6.py:80) 没有 reference 过滤，且导出时丢弃来源字段。“没有另外添加 10x 文件”不等于“从 VDJdb 中排除了 10x”。
- 本轮只读实际源文件：其 reference 中带 `10x` 的值只有作者明确排除的那个 URL。按当前配链与 CDR3 规范化规则，得到 18,658 个 αβ pair；其中 **1,275** 个出现在当前 target 文件（GILGFVFTL 1,273 / NLVPMVATV 2），**17,383** 个出现在 background。这是**来源相关 pair 的匹配数**，不是标签泄露数，也不意味着这些 pair 都只由该研究提供。不能直接把它们全部删除来冒充重建原论文数据；应先恢复来源再按作者过滤和去重规则重建。
- 论文 k=1 穷举全部 reference 选择，k=2–200 各 100 次共享随机 split；当前所有 k 都固定 100 次、候选池上限 2,000。当前数据每目标少于 2,000，cap 实际未截断，不能把它当成本次差异的已证实根因。
- 论文误差条采用同 split 的模型相对均值差异方差；当前 macro 的 std 是跨表位均值的标准差。二者不能互换作显著性证据。broad 24 表位/80–20 与论文扩展实验、线性 SVC、微调实验也须分别对齐，不能只因叫 broad/probe 就认作相同。

**处置**：本轮只修引用与审计文档，不改数据/代码/既有数值，不提交 GPU。当前产物保留为本地控制；旧 `[P]` 原值不改，但 registry 的 `exact/compatible` 及旧校准结论不能自动证明本地复现。后续先锁定“论文复现 / 同输入控制 / v5 适配”三种不同目的，再做数据、权重、链范围、层数/池化、episode 的逐项验收。关键日志见 [总审计 §9.11](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

### 7.3 2026-09-14：新旧 baseline 比较的收数保护

再次核对 [Table SI 原表](https://arxiv.org/html/2406.06397v2)：历史 registry 的六行 `ESM2-150M` 论文数字实际上属于论文 **ESM2-T6-8M**。历史 CSV 和数值保留；[新收集器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_tcr_49000_comparison.py) 在独立输出中显式修正显示名、记录旧标签映射，不继承旧 `exact` 标志。不能将该论文列与本地 150M/CDR3 运行合并；六表位均值由印刷数值计算，不伪称作者提供了更高精度宏均值。

本轮 49000 broad 已按唯一受体和多标签协议评分，但历史 baseline 未重跑同一修正。因此新结果与旧 broad 表只能并列参考；deep 的历史本地输入及运行证据也不自动升级为本轮门禁通过。正式论文 `[P]`、历史本地控制、当前 49000 分层呈现，数值唯一见 [RESULTS §0.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。没有 v5 29k 的同协议 T3 成绩，不借用 v3/allch，亦未新启动 baseline 或旧模型 GPU 重跑。

## 8. Agent 执行手册

2026-09-30 汇总使用独立新目录；命令、CPU 环境和 source-copy 验证见 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)。历史原始结果保持只读。

### 8.0 2026-09-13：v5 runtime 本地对照（不是论文同版复现）

执行地点追加：broad 本机执行完成；others 完成后 deep 也已取消云端 Queue 至 Killed、于本地接续。沿用 v2 完整数据/参数/输出身份，不截断 few-shot episodes；结果需独立复核。安全入口与状态见 [审计 §9.24](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

- 新入口：[run_tcr_native_repr.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_tcr_native_repr.py) `--task t3deep / t3broad`。通过 `tcr-v5:<checkpoint>` 使用 post-LLaDA、运行时 v5 补全和 observed-only 全局池化；αβ 一起输入，不输入 peptide/MHC/VJ 或 binding。null prefix 固定 `<unknown>`；标签只用于 support 构造与评分。
- deep 保留现有 target/background 25,816 行的本地数据版本与固定 k/100-seed 协议。§7.2 的来源、k=1 穷举、统计方差与论文差异仍存在；名称含 `paper6` 不等于论文数据复现。
- broad 不改 CSV；运行时按 `(cdr3b,cdr3a)` 分组，保留一个受体对应的全部已知表位集合。每个表位分别取唯一受体 support，query 的正例是标签集合包含该表位，不能把同一受体的另一行当已知负例。训练/测试组交集必须为零，保存所有抽样 support 索引、seed、k 与逐 episode 分数。共保留 6,885 train / 1,721 test 身份，原 6,887/1,722 行不删除。
- broad 原始训练集的两条空 α 显式变为未知 X 伙伴；不伪造观测序列，也不默默删行。β 必须有效，其他坏序列拒绝。已有 24-way probe/kNN 作为 `probe_diagnostic.json` 单独保留，是单行注释预测诊断，不冒称互斥结合特异性或论文 SVC。
- 验收：T2 评分修正会改变共同源码指纹，旧 T2/T3 四个排队提交已取消后重新验收；不是发现了新的 T3 模型输入缺陷。当前 [v2 repr gate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/tcr_v5_49000_20260913/repr_preflight_v2/passed.json) 固化数据/代码/权重身份，真实 GPU batch=1/8、倒序、最长与缺链样本重新通过。配置入口 `prepare_tcr_native_jobs.py --stage repr-v2`，输出 tag 为 `ours_tcr_v5_49000_20260913_runtime_v2`，旧 gate/产物不覆盖。
- 新增 [评分回归](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_tcr_scoring_audit.py)：deep 在所有非自身距离相同的 oracle 中，逐次核对支持样本已从 query 排除，AUROC 恰为 0.5；broad 检查多标签配对身份与唯一 support。联合回归 81 passed。完整小样本评分预检独立于正式质量结果，状态见 [审计 §9.22](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。
- 当前仅放行 local-control 工程重评；预训练 overlap/近同源、论文同版数据、各 baseline 原生输入/episode 对齐仍是可引用 gate。旧 baseline 不改成新多标签协议的复跑结果；新分数不得以旧“exact”标签冒称公平超越论文。下方 §8.1 可引用 gate 未全部完成，与本次输入验收/提交不是同一状态。

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

- [ ] 按 §7.1 完成无 epitope 的 v5 运行时输入、synthetic 排除及真实前向验收；固定 support 下改 query 标签不得改变输入/打分
- [ ] 锁定 αβ / β-only 输入档、缺链/多表位政策、对应身份去重、episode 清单与预训练 overlap；不得把成对去重等同于 β-only 去重
- [ ] embedder spec 含 `decoder` / `bioseq-llada`
- [ ] `--columns cdr3b cdr3a`；SCEPTR/TCRdist 走官方后端（会自己读 V/J）
- [ ] 主表是 few-shot NN，不是 24-way probe
- [ ] 填表时带「输入」列
- [ ] BERT 允许
- [ ] 完成 §7.2 的论文/本地差异 gate；TCR-BERT 论文 checkpoint 与 MLM-only 控制分臂，不把后者替代论文原模型
- [ ] strict load 0 missing / 0 unexpected

### 8.2 提交命令

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
REPR_STAGES=t3 bash scripts/downstream/run_immune_fusion_repr.sh \
  /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000 \
  ours_fusion_270m_diff_42000 16
```

Oracle 自检：`python -m common.fewshot`（支持集含 query 正样本 → AUROC≈1.0）。

Volc：`bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_repr.yml`

### 8.3 收数

读 `outputs/tcr_representation_paper6/<tag>/` 的 `fewshot.json` / `per_epitope_by_shot["200"]`。对照 registry k=200 `[P]`。

### 8.4 填表

1. 先按输入字段分组，再写数字进 §0.3。Ours 表末。
2. 禁止把 probe 数字当 headline。
3. changelog → `PROGRESS.md`。

### 8.5 同步哪些文档

`RESULTS.md` §0.3 + `PROGRESS.md` + 本文件 + `PROJECT_PROCESS.md`。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-02 | 初版。主协议锁定 few-shot NN；强制输入字段分层。 |
| 2026-09-13 | 新增 §7.1 专项审计与提交 gate；仅 CPU 诊断及文档同步，无新模型指标或生产代码修改。 |
| 2026-09-13 | §7.2 复核正式论文/作者 notebook/模型卡，撤回当前本地协议等同论文的表述；登记来源过滤、模型输入/版本/读出及训练协议差异。 |


### 2026-09-30 Foundation预训练暴露范围补充

预训练去重覆盖broad的train支持池与test查询池，以及paper6的target_binders/background_pool全采样宇宙。`__background__`不是epitope序列；这些行的受体仍需保护，原始表位溯源不能假定已完成。详细方法、实测证据与待办统一见 [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) 的“新增TCR下游暴露审计”。这不改变合法support标签使用方式，也不把epitope/query标签送入encoder。


2026-09-30更新：完整support/query池的foundation暴露审计已经完成，详细证据见 [完整审计记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#tcr-exposure-20260930)。已确认受体去重不能代表表位去重；background的受体重复与缺失原始表位元数据分别记录。support/query合法共享目标表位的协议不改变，新增排除要求作用于foundation预训练数据侧。
