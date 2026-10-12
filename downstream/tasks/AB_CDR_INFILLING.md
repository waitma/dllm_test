# AB CDR Infilling

## 2026-10-12 CDR 评测验收目标（用户确认）

以 Ophiuchus-Ab 官方权重能否复现论文 CDR 结果来验证评测流程，不能仅以复用 AirGen 函数或代码形式相同作为验收。覆盖 Kong 三个重链 CDR 与 SAb23H2 六个 CDR；统一数据、掩码、可见上下文、解码设置和 AAR 聚合，再替换为我们模型的 forward 适配。模型自身输入布局不强行改成另一模型的训练布局。

历史官方权重复跑见 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)，仅证明其历史入口的结果接近论文，不能替当前 AirGen 对齐入口完成端到端验收。当前入口尚需绑定官方权重完成同协议对照；论文未披露参数、重建划分差异及逐 CDR 误差须如实记录。不得按测试分数逐 CDR 选择 iteration 来贴合论文，也不预先宣称精确复现。既有实测成绩保留，未经重跑不改写数值。

## 2026-10-08 当前模型按AirGen协议对齐（优先于下方历史iter1约定）

用户明确要求全部CDR infilling严格对齐。当前best45000的Kong H1/H2/H3与SAb23H2六CDR以新独立评测为准；历史模型结果不改写，也不冒充重评结果。

- Kong读取原JSON位置字段 `[start,end]`（inclusive），禁止首次子串定位；H/L可见上下文、空轻链保留，十折各自batch64，不过滤样本。
- SAb23H2直接读取官方60个ID和native/design FASTA，以design中全部X为mask（含非连续掩码），不从目标子串推断。原目录h_cdr1等仅映射到AirGen代码cdrh1等名称，不改原始文件。
- Kong固定argmax/4步（Python入口默认）；SAb23H2固定argmax/2步（run/zero_shot_test.sh中的示例命令，Python默认另为4步）。这是事先固定的源码/命令对齐，不声称论文历史运行参数已证实，不按成绩选步数。
- temperature1、CFG0；直接从AirGen源码提取并执行generate/forward_decoder/_decoding等原函数。原始logits排序、linear cutoff、并列严格小于阈值、EOS微惩罚保持；不替换为softmax置信度或一次argmax。
- 虚拟150/128双链token网格对应AirGen评测控制和打分；模型forward仍映射到自身训练V5对象302布局。两者词表/结构不同，native没有ESM的gap '-'，此项须披露，不能称逐token完全相同。Kong使用bf16 autocast；SAb23H2沿脚本不启用autocast。
- AAR按每样本被mask位置计算；Kong输出样本均值、十折宏均值、折间std(ddof0)；SAb23H2按原代码先round4再乘100，并记录样本std(ddof0)。所有目标位置、预测和参考逐行保存。

入口与全量输入哈希：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_airgen_aligned_20261008/evaluate.py、protocol_audit.json。结果唯一维护在RESULTS.md的AirGen协议对齐区。旧iter1结果保留为历史对照，不作本次对齐headline。


> **2026-09-16 本地多步对照已完成**：完整 SAb23H2（60 条、六 CDR）、既有 92000 评测快照、iter=1/2/4/8 的 1,440 条预测及输入审计通过；本轮未运行 Kong 多步对照。采用已修复的 ESMC 逐步反馈，详情见 §4.3，数字见 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)。既有 iter=1 主结果不被测试集上的 best-iteration 替换。

> 2026-09-14完成更新：指定v5 49000的Kong/SAb23两个任务均Success，全部模式聚合日志齐全；数值和核验边界见 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。下方09-13“已提交/尚未完成”为历史状态。原入口未保存逐序列预测，尚未独立重生成验证AAR；v5训练重叠仍未核清。

> Agent 进入本任务时：先读 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md`，再读本文，数字只写 [`RESULTS.md`](../benchmark/RESULTS.md) §0.5。
> 模板：[`_TEMPLATE.md`](_TEMPLATE.md)。规则：[`downstream-doc-sync.mdc`](../../.cursor/rules/downstream-doc-sync.mdc)。

> 2026-09-13 执行更新：用户指定最新 v5，统一快照的 Kong（H1/H2/H3、10折）与 SAb23H2（六个CDR、1折）两个非闲时单卡任务已分别提交，均 argmax / iter1；提交前真实模型各模式 GPU 门禁通过。具体 checkpoint/SHA/YAML/task ID 见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 的 `Submitted Ours AB v5 49000` 记录。正式成绩尚未完成；历史污染比例不直接当作 v5 的重算结果，v5 训练重叠仍需核验，不能称干净 zero-shot。

---

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | AB-CDR |
| 锚点论文 | Zhu et al., *Ophiuchus-Ab*, bioRxiv 2026, [10.64898/2026.02.02.703197](https://doi.org/10.64898/2026.02.02.703197) Table 1（SAb23H2）+ Table 2（SAbDab Kong） |
| RESULTS 锚点 | `RESULTS.md` **§0.5** |
| 主指标 | AAR % ↑，固定 `iter=1`，聚合 `average_aar_all_folds`（逐折宏平均） |
| 数据根 | Headline：**`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab_kong/`**（3,127）。SAb23H2：`.../sab23h2_converted/`。旧快照 `.../sabdab/`（3,320）**不是 headline** |
| 我们的入口 | `python -m downstream.grammar.cdr_infill` |
| 可引用性 | v5 训练重叠仍未核清，不能称干净 zero-shot。下方历史 54.4% / 25% 报告不直接当作 v5 的污染率；推理输入防泄露与预训练去污染是两回事 |
| 最后更新 | 2026-09-16 |

---

## 1. 任务定义与范围

一次掩一个 CDR，argmax 回填，报氨基酸恢复率。SAbDab 只评 H1–H3；SAb23H2 评 L1–L3+H1–H3。**BERT 禁止。**

### 1.1 模型适用性矩阵（硬约束）

| 我们的模型臂 | 本任务是否适用 | 原因 |
|---|:--:|---|
| Ours-Diffusion | yes | mask-fill 生成 |
| Ours-BERT | **no** | 无可用迭代解码路径 |

### 1.2 不在本文档范围内的相邻任务

- IgCraft humanization：不是 CDR infilling。
- Light-chain pairing：见 `AB_LIGHT_CHAIN_PAIRING.md`。
- 结构感知方法（DiffAb / MEAN / dyMEAN / IgGM）：仅 `[P]` 转录，本轮不本地复跑 headline。

---

## 2. 参考文献调研（锚点论文做了什么）

### 2.1 论文实验全景

| 论文实验 | 数据 / split | 指标 | 论文结论 | 我们是否对照 | 不对照的原因 |
|---|---|---|---|:--:|---|
| Table 2 SAbDab 10-fold | Kong et al. 冻结快照，3,127 complexes，heavy H1–H3 | AAR % | Ophiuchus zero-shot 与 PLM 接近、强于结构方法 | 是 | — |
| Table 1 SAb23H2 | 官方 `IgGM_Test_set` n=60，六 CDR | AAR（论文印分数，registry 存百分数） | 同上 | 是 | — |
| Table 3 pairing | OAS | ImmunoMatch 等 | — | 否（另一文档） | — |
| Table 4/5、Fig 3/4 | GDPa1 / specificity / humanization / m396 | 各异 | — | 否 | 本轮按计划排除。后三项数据见 [`../../data/downstream/PROBE_DATA.md`](../../data/downstream/PROBE_DATA.md) |
| Kong split 构造 | MEAN `data/split.py`，`mmseqs --min-seq-id 0.4`，seed 2022，filter 111 | — | 3,127 与论文吻合 | 是（本地重建） | Kong **没有** release 处理好的数据 |

### 2.2 指标定义（逐字对齐）

| 指标 | 论文定义 | 我们的实现 | 已知差异 |
|---|---|---|---|
| AAR | 被掩 CDR 的逐残基准确率 | `grammar/metrics.py` masked token accuracy | — |
| 聚合 | 论文 H1 75.50 更贴宏平均 | **`average_aar_all_folds`**；`average_aar`（微平均）H1 可差 1.02 pp | 必须用宏平均 |
| `max_iter` | **论文未公布** | 报告口径 **固定 iter=1**；1/2/4/8 sweep 仅作波动带 | 两数据集指向不同最优步数，残差=协议不确定性 |
| 解码 | 官方 AirGen `zero_shot_test.sh` 对 CDR 传 `--max_iter 2` | 我们 headline 仍报 iter=1（更贴 SAbDab 论文） | SAb23H2 上 iter=2 更贴论文 |

### 2.3 论文未公开的关键细节

没有单一 `max_iter` 能同时贴合两个数据集。不得把 per-CDR 测试集 argmax 写成「已复现」。

---

## 3. 官方产物 release 与获取状态

| 产物 | 是否公开 | License | 本地绝对路径 | SHA256 | 获取方式 | 状态 |
|---|:--:|---|---|---|---|---|
| Ophiuchus-Ab 代码 | yes | — | 官方仓 `Ophiuchus-Team/Ophiuchus-Ab`；本地 eval `downstream/ophiuchus_eval/` | — | GitHub / 代理 | 已落地 eval |
| 权重 | yes | — | `/c20250601/mj/model_weights/ophiuchus_ab/` 或仓库副本；Zenodo 18478480 | 见 ophiuchus_eval README | Zenodo | 已 `[R]` 复跑 |
| Kong 数据 | 无处理好的 release | — | `data/downstream/cdr_infilling/sabdab_kong/` | 30 个 `test.json` md5 重建一致 | `bash scripts/downstream/build_sabdab_kong_split.sh` | 已重建 |
| SAb23H2 | yes（IgGM test set） | — | raw `.../sab23h2/`；eval `.../sab23h2_converted/` | — | 官方 tar | 已落地 |
| MEAN split.py | yes | — | 随 THUNLP-MT/MEAN | — | GitHub | 已用来出折 |
| AntiBERTy / AbLang2 | 论文 `[P]` | — | 本地 masked-fill 在 `output/downstream_generation/cdr_baselines/` | — | HF | **协议不同，禁止与 Table 2 混排** |

---

## 4. 我们的协议

### 4.1 数据与入口

| 项 | 值 |
|---|---|
| Headline 数据 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab_kong/{cdrh1,cdrh2,cdrh3}/fold_*/test.json` |
| SAb23H2 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sab23h2_converted/` |
| 入口 | `python -m downstream.grammar.cdr_infill --test-set <dir> --mode <cdrh*> --checkpoint-path <ckpt> --sampling-strategy argmax --max-iter 1 --num-folds 10` |
| Kong 作业 | `eval_jobs/eval_immune_{270m,8b}_diff_cdr_kong.yml`（**显式** `DATA=.../sabdab_kong`） |
| 危险作业 | `eval_v3_*_cdr.yml` / `run_immune_fusion_gen.sh` 默认 `SABDAB=.../sabdab`（旧 3,320） |
| 输出 | `output/downstream_generation/kong_split/` 、 `.../sab23h2_fixed/` 、 `.../ophiuchus_ab/kong_split/` |

### 4.2 口径切割线（⚠️ 引用前必读）

| 分组 A | 分组 B | 为什么不可同表 | 证据 |
|---|---|---|---|
| `sabdab_kong` n=3,127 | 旧 `sabdab/` n=3,320 | 不同快照；fold 一致率 9–12%；H3 更长 | `RESULTS.md:813-816` |
| v3 CDR（走 wrapper 默认路径） | Kong headline | **v3 跑在错误数据集上** | `run_immune_fusion_gen.sh:37`；`eval_v3_*_cdr.yml` 只设 `GEN_STAGES=cdr` |
| 固定 iter=1 | per-CDR best-of {1,2,4,8} | 后者是测试集调参，已撤回 | 审计 AB_cdr §2 |
| `average_aar_all_folds` | `average_aar` | 宏 vs 微 | `RESULTS.md:867` |
| Ophiuchus `[P]` 43.55 | `[R]` 43.70 | 论文值 vs 官方 ckpt 复跑 | 审计 §4 |
| 本地 AntiBERTy/AbLang2 `[L]` | Table 2 `[P]` | 协议不同 | `infill/README.md` |
| BERT CDR | diffusion CDR | BERT 不适用 | §1.1 |

---

### 4.3 2026-09-16 本地 iteration 对照（诊断，不选 headline）

- 用户授权本地运行。先覆盖完整 **SAb23H2 60 条 × 六个 CDR × iter=1/2/4/8**，共1,440个 CDR 预测；Kong 不在本轮运行范围内。冻结独立留存 `output/ab_eval_checkpoints/ab_v5_92000_llada_20260915`，权重／tokenizer 不变。
- 使用原生 fusion loader、原有 CDR mask helper 与 `run_grammar_generate`，模型只加载一次；仍为 batch=1、argmax、CFG0、temperature1、seed42，所有步数都跑相同样本，不筛选候选。目标长度是已知 CDR 长度；其他 CDR、框架及另一条链可见。
- 共享 sampler 已能将接受的生成残基回填 ESMC；目标未接受位置继续 MASK。新启动进程生效，不会更改旧任务内存。旧的多步记录不能冒称本次修复后的结果。
- 新 [runner](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_ab_cdr_iter_sweep.py) 在每模式首条样本、每个步数捕获 fusion 输入及真正 encoder forward 输入，检查已生成 token 的词表转换、pending MASK、固定上下文与已接受残基不变；在最大步数额外替换隐藏目标内容，要求输出与轨迹不变。此为真实权重输入检查，不单独构成性能结果。
- 每模式 iter=1 与对应旧92000日志的 `Average AAR all folds` 差异须≤0.0001个百分点，否则停止，不把差异归因为步数。六模式共24格都保留，不在测试集挑每个 CDR 的最好步数后填主榜。
- 独立产物根：[本轮运行](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_cdr_iter_20260916)。保存 `manifest.json`（权重／数据／源码SHA）、`status.json`（PID／状态）、`feedback_audit.json`、每模式每步数的 `predictions.json`／`metrics.json` 及 `summary.json`。逐条保留预测、目标、匹配数和AAR，未完整时 `complete=false`；旧日志和主结果不覆盖。
- 启动入口：[本地 shell](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_ab_cdr_iter_sweep_local.sh)。37项 CPU 回归已通过（新增 CDR 回填8项、宏平均1项，加原生成28项）；使用既有 protenix_abtcr 环境，本机常驻进程不动，无 Volc submit/cancel。
- 完成验收：`summary.complete=true`，24格／1,440条齐全；独立核对数据身份、CDR区间、预测词表、逐条匹配数与重算AAR，六模式iter1与历史日志差值均为0。13份源码及6份数据SHA未变；24组首样本真实encoder轨迹和6次隐藏参考扰动检查通过。例如H3首样本iter4的ESMC目标区MASK数为17→12→8→3，对应可见生成残基0→5→9→14，最后一步输出全部填完；没有永久mask住已接受位置。
- `max_iter` 是上限：目标填完会提前结束，不表示每个短CDR都恰好forward该次数。上述输入抓取仅针对每模式首样本；全量预测完整性／指标重算覆盖全部样本。完整诊断表见 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)，不把修复后步数对照说成修复前后性能对照；v5预训练重叠本轮未审计。

### 4.4 V7 no-ESMC 原生适配（2026-10-06；真实两轮九项已完成）

当前实际状态：50082/epoch2终态、九项完整评测及原两个监督实际rc0均已核实，输入扰动审计/覆盖/原生单步要求保持。数值唯一见 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json。下述带早前时间的等待文字为历史，不应重发该批次。

2026-10-06T21:22:17.810232+00:00 执行接续：旧训练/生成监督因启动锁故障退出后，独立恢复首段已真实通过；当前有效生成包为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/package_v1.json，新等待监督PID320497/start2458326416。只改等待/完成验收路径为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/COMPLETE.json，仍固定50082、原单步解码/9项/9741 region cases及同一冻结前端和评分器，未发布或推理。旧generation_pool_v1包保留历史，不再执行其已停监督。执行状态唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

2026-10-06T14:15:40.813569+00:00 历史执行更新（已由上方独立恢复包接替）：9个dataset/mode任务复用现有8卡，监督已实际部署等待两轮，不是已入队。当时封存包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_pool_v1/package_v1.json；前端 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_source_v3/scripts/downstream/run_v7_cdr_eval.py，新增--mode只拆分任务，不改采样或评分。每项summary显式coverage=assigned_mode，整批9项和9741条全部验收才称本批完成。36个文件只读格式/区间/容量检查通过，Kong缺light样本仍保留，不生成新注释。v2串行计划保留历史，不与自动9项重复运行。

用户指定累计两个epoch后的固定checkpoint50082。新的入口为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_cdr_eval.py；底层适配 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_native.py。旧fusion入口保持。V7入口从checkpoint的run_provenance定位并核验冻结模型源码、词表、权重SHA和f011/A4配置，严格载入原生FactorialModel；实际50082、epoch2与八rank完成验收齐全才接受正式评测。

沿用本文既有数据、单步argmax、按序列后按fold宏平均；Kong三种heavy CDR、SAb23H2六种CDR。这里评测的是已知CDR长度的回填，不是自主长度或抗原条件设计。模型输入保留训练时A4 receptor容量布局、框架、其他链与链末端；关系为unknown。CDR先替换为常量占位再渲染并全部MASK，只有七种原生feature张量传入denoise；labels、residue_mask、terminal_mask不传入。逐条替换隐藏参考字母验证实际特征不变，并保存模型收到的张量指纹。沿用旧入口第一次精确子串定位，歧义单独记录。

2026-10-06 对比口径补充：V7固定单步结果出齐后，同时呈现既有V5 92000的iter=1历史对照及V5 159000的iter=2主行，并逐行标注解码设置。具体数字与产物出处仅引用 [RESULTS §0.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr)。V5 92000与V7均为单步不等于受控架构消融：训练数据、模型规模、训练预算和内部输入布局仍不同；V5 159000还多了解码步数差异。历史Kong结果只有聚合日志，SAb23H2 92000另有完整逐条预测，不能将两者统一描述为已逐样本复算。不使用测试集结果选择V7 checkpoint或解码步数，不改当前50082和单步的冻结执行设置。

输出使用原生A4 residue＋protd词表；若在CDR内预测出protd，保留为错误token，不静默修复/删去。不强行转换为旧fusion词表或采样器，不把两种模型的内部布局说成相同。GPU采用FP32参数＋BF16 autocast，CPU为FP32；记录实际精度、模型源码和全部输入身份。保留现有CDRResultWriter逐行预测/参考/位置与重评分机制；缺行或max-samples截断不能标完整成绩。下游数字只写RESULTS，不能以训练验证指标代替。

实际CPU合成验收通过：使用冻结f011的tiny模型验证heavy/light/缺light三种输入、隐藏参考扰动、真实denoise参数捕获、原生argmax一致、固定上下文保持、END不修补、非法区间拒绝和临时save/reload一致；全部临时checkpoint已删除。日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/cpu_generation_adapter.log。本次没有用正式权重或下游数据运行模型，尚无V7生成成绩，不声称预训练全范围无重叠。

固定执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_plan_v2.json，原生前端冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_source_v2。v1因旧包提前导入fusion而废弃；v2隔离CLI和评分器载入通过，未导入任何模型/Torch。下面为工作区命令示例；正式执行应使用封存计划中的完整命令并先核对SHA（两轮完成后，原容器自动执行已接入等待，不手动重复启动；本地--device cpu仅为独立诊断示例；不新申请资源）：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python -B /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_cdr_eval.py \
  --checkpoint /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run/checkpoint-50082 \
  --training-acceptance /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/COMPLETE.json \
  --dataset sab23 --device cuda:0 \
  --output-root /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_sab23
```

Kong使用同一命令的--dataset kong与独立输出目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_kong。不得复制/改造提前产生的checkpoint或验收记录以通过50082检查；正式作业安排与原训练衔接状态见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

### 4.5 V7固定九项完成后的只读结果汇总（2026-10-06T21:49:54.047686+00:00）

收集入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_v7_cdr_results.py，在conda pllm执行 python -B 该绝对路径 --output /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json。入口复用当前恢复阶段50082与生成批次9项真实完成证据；只读取原生验收、metrics/manifest/summary及已有V5聚合来源，不运行模型、注释、预测或重新评分，不自动编辑RESULTS。缺任何当前完成证据立即拒绝，且不创建输出目录。固定逐序列后逐fold宏平均，分别报告九项，不给跨CDR混合总分；仍保留known-length/A4/iter1限制及先前V5对照差异。

准备核查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_result_collection_v1/root_preparation_review.json（fa332b rc0）：历史18格来源可读取，真实两轮验收未出现时拒绝已核实。真实V7批次汇总已于2026-10-06T22:28:30.597931+00:00运行通过（febf9e rc0），完整报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json；根端已更新唯一数字文档 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 与过程 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md。本节不重复榜单数字，不据测试结果选择checkpoint或采样步数。

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先阶梯（强制，全项目统一）

Table 2 已按 `[P]` 主行 + Ophiuchus `[R]` 附行排列，合规。结构方法仅 `[P]`、未本地复跑，保留。

### 5.2 Baseline 表

| 方法 | 出处层 | 最佳已发表值来源 | 官方产物是否拿到 | 是否真的本地跑通 | 协议差 |
|---|:--:|---|:--:|:--:|---|
| Ophiuchus-Ab | `[P]` + `[R]` | Table 2 / Table 1 | yes | yes | iter 未公开 |
| AntiBERTy | `[P]` | Table 2 | 权重有 | 本地 masked-fill ≠ 论文协议 | 禁止混入 `[L]` |
| AbLang2 | `[P]` | Table 2 | 同上 | 同上 | 同上 |
| RADD / ADesigner / dyMEAN / MEAN / AbBFN / DiffAb / IgGM | `[P]` | Table 1/2 | 部分 | **否** headline | 结构方法 vs 序列 infill |

---

## 6. 结果写入位置

| 内容 | 唯一权威位置 |
|---|---|
| 排行榜数字 | `RESULTS.md` §0.5 |
| 任务锚点 | `benchmark/README.md` §3.1 AB CDR 行 |
| changelog | `benchmark/PROGRESS.md` |
| Volc | `PROJECT_PROCESS.md` |
| 本任务协议 | **本文件** |

产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/`

---

## 7. 已知缺陷与不可比口径

2026-09-30 完整性约束：旧日志只有汇总 AAR，标为 `aggregate_only`，不能补造逐样本预测。新入口将完整数据集行数作为预期集合；空 fold/空掩码、缺行、重复 ID、身份/计数/hash 不一致不能当作完整结果；`max_samples` 截断标为 `partial`。此变更不改变采样与 AAR 定义。结构见 [产物格式](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#cdr-artifacts)。

| # | 缺陷 | 状态 | 后果 | 证据 |
|---|---|:--:|---|---|
| a | ASD 去污禁用 → 54.4% Kong H3 进训练 | 未修（ckpt 已污染） | Ours=上界；与 Ophiuchus zero-shot 不对称 | `RESULTS.md:21,471-495` |
| b | 54.4% 生成脚本缺失 | 报告 json 仍在；`scripts/ab_cdr_leakage.py` **不存在** | 结论可引用但不可复现；重算需重写 | `outputs/ab_cdr_leakage_report.json` |
| c | v3 CDR 默认旧 `sabdab/` | 仅登记，本轮不改 wrapper | v3 行不可与 Kong headline 同表 | `run_immune_fusion_gen.sh:37` |
| d | encoder-stream 泄露 | 已修 | 旧虚高 AAR VOID | `RESULTS.md:391-410` |
| e | 「与官方打平」措辞 | RESULTS 已撤回；`AB_TCR_EVAL_SUMMARY.md` 已 ARCHIVE 并对齐「只能作上界」 | 以 RESULTS 为准 | `AB_TCR_EVAL_SUMMARY.md:20` vs `RESULTS.md:21` |
| f | 污染率 54.4 vs yml 注释 29.8 | 8B yml 已加注释：以 54.4% 为准 | 29.8% 是另一判据 | 8B train yml L61 vs RESULTS |
| g | SAb23H2 文档路径 | 已改 | eval 用 `sab23h2_converted/` | `benchmark/README.md` §5 |
| h | 共享 grammar sampler 的 pending mask 与实际输入脱离 | 已修；SAb23H2 92000 多步重评及状态检查完成，见 §4.3 | 旧多步产物不自动升级为新结果；Kong多步未重评 | [AB pairing §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)（唯一问题记录） |
| i | ESMC 未读取逐步已接受残基 | 共享代码已修；SAb23H2 92000 多步真实输入检查与重评完成，见 §4.3 | 六模式iter1复现通过；新多步仅作独立诊断，未覆盖主结果 | [AB pairing §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md) |

---

## 8. Agent 执行手册

2026-09-30 新运行在原有 `cdr_infill` 命令上增加 `--output-dir <全新绝对路径>`。三个未来作业模板已默认按 dataset/mode 分目录保存；旧 YAML 不重新生成。离线复算使用：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/bin/python -m downstream.grammar.cdr_result_io --score-only <产物目录绝对路径>
```

该命令只读 manifest/predictions，不加载 checkpoint 或重新采样。截断/中断产物只有显式 `--allow-partial` 才能输出诊断分数，状态仍为 partial；完整报告须覆盖 manifest 预期行集合。不要覆盖旧目录或用历史日志伪造 predictions。

### 8.1 提交前 gate（逐条打勾，任一不过就不要提交）

- [ ] **diffusion only**
- [ ] **`DATA` / `--test-set` 指向 `.../cdr_infilling/sabdab_kong`，不是 `.../sabdab`**
- [ ] 显式记录步数：历史 49000/92000 主行 `max-iter=1`；140000 及本轮 159000 对照固定 `max-iter=2`，均为 `sampling-strategy argmax`，不可混称相同解码协议。
- [ ] 收数用 `Average AAR all folds`，不是 `Average AAR`
- [ ] 不要提交只设 `GEN_STAGES=cdr` 且走 `run_immune_fusion_gen.sh` 默认路径的 yml 当 Kong headline
- [ ] `test -f ${CKPT}/model.safetensors`

### 8.2 提交命令

Kong（推荐直接用 Kong yml，不要用 gen wrapper 默认）：

```bash
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
bash scripts/volc-no-proxy.sh ml_task submit --conf eval_jobs/eval_immune_270m_diff_cdr_kong.yml
```

本地：

```bash
ENV=/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr
export PYTHONPATH=/vepfs-mlp2/c20250601/251105016/project/dllm_test
export LD_LIBRARY_PATH=${ENV}/lib:${LD_LIBRARY_PATH:-}
CKPT=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/protein_esmc_llada270m_diffusion_immune/checkpoint-42000
DATA=/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab_kong
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python -u -m downstream.grammar.cdr_infill \
  --test-set "${DATA}/cdrh3" --mode cdrh3 \
  --checkpoint-path "${CKPT}" --device cuda --num-folds 10 \
  --sampling-strategy argmax --max-iter 1
```

重建 Kong split：`bash scripts/downstream/build_sabdab_kong_split.sh`

### 8.3 收数

```bash
grep -E "Average AAR all folds" /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/kong_split/*.log
```

官方 `[R]`：`output/downstream_generation/ophiuchus_ab/kong_split/cdrh*_iter1.json` 的 `average_aar_all_folds`。

### 8.4 填表

1. `[P]` 在上，`[R]` 附行，Ours 表末并标注上界。
2. v3 若数据根不是 kong，分表或拒绝填入 §0.5 headline。
3. changelog → `PROGRESS.md`。

### 8.5 同步哪些文档

`RESULTS.md` §0.5 + `PROGRESS.md` + 本文件 + `PROJECT_PROCESS.md`。

---

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-02 | 初版。Kong gate；iter=1；v3 走错数据集仅登记；54.4% 脚本缺失。 |


### 2026-10-09 CDR iteration 对照

新增iteration对照：固定269000步与AirGen任务协议，max_iter取1/2/4/8/16/32；全量Kong H1/H2/H3十折、SAb23H2六CDR。只改变迭代次数，保留argmax、temperature1、CFG0、batch64及各数据集原精度。报告各CDR AAR、折统计与整数据集耗时；测试集敏感性分析不得替换默认Kong4/SAb2主协议。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v5_cdr_iterations.py --run /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations269000_20261009 --dataset kong|sab23；--check执行所有步数的CPU上下文/答案独立性检查。


### 2026-10-09 验证最佳138000评测

iteration对照增加验证最佳138000步，沿用269000的六个步数、同一数据/掩码/解码/精度/评分方式。独立入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations138000_20261009/run_iterations.py；测试集敏感性分析不替换默认协议。
