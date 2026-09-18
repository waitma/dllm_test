# Ophiuchus-Ab baseline 测评审计与完善指南

> 更新：2026-09-15（UTC）。
> Specificity新增200轮独立学习曲线诊断的LR/固定末轮/缓存与产物约束见 [AB Native Probes §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#45-specificity-200-epoch-学习曲线诊断2026-09-15)；保留原100轮报告，不将诊断曲线选优当作CV成绩。
> AB结果展示最新要求见 [PROJGUIDE §0.2.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#022-排行榜构造论文值优先强制2026-09-02-起)；Ours Specificity分类头与AirGen的结构/宽度区别及CPU核验见 [AB_NATIVE_PROBES §4.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#44-specificity-分类头对齐核查2026-09-15)。
> GDPa1 最新用户决定：不需要 nested；当前唯一执行/收数协议及历史产物边界见 [AB_NATIVE_PROBES §4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#42-预测头与协议切割线)。不改 Ophiuchus baseline 原脚本或旧结果。
> 本轮Ours结果：指定v5 49000的11项抗体任务全部Success，CDR/六组pairing/三个原生探针完整汇总见 [RESULTS][results] §0.5/§0.6/§0.6a；探针保存预测重算与跨硬件重载差异见 [AB_NATIVE_PROBES](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md) §7。下方§2–8仍为Ophiuchus baseline独立审计，不因Ours跑完而标作全部关闭。
> 状态：§2–8 表征探针仍为审计/待完善；§9 pairing 的 reference-length v3 全量生成、现有评分及产物验收已完成。新结果按独立长度协议报告，不等同于复现原生 Ophiuchus-Ab。
> 用户确定的要求：特异性应在每折选定一个分类头状态后计算该折指标，再汇总五折；不能把多个 epoch 的测试指标平均当作五折交叉验证结果。
> 新增用户决策：Ours light-chain pairing 采用 reference 轻链长度，作为已知长度条件生成；不强制使用随机长度先验或自主终止。见 §9；当前实现与验收入口为任务 §7 h。
> 最新授权：用户明确要求在我们 ESMC+LLaDA 模型上测这些 AB 任务，并增加 pairing prompt0/prompt3 与 CFG 对照，按非闲时单卡拆作业；进一步指定使用最新 v5。已固定独立权重快照并通过 GPU gate；正式矩阵状态见 [PROJECT_PROCESS][process]。入口见 §10，不把历史 baseline 成绩当成 Ours。

## 1. 范围、文档职责与阅读顺序

§2–8 只考虑官方 Ophiuchus-Ab baseline 的三个冻结表征任务：Specificity（HD / Flu / CoV，Table 5）、GDPa1 developability（Table 4）、m396 affinity（Figure 4）。
按用户新增要求，§9 补充记录本项目 LLaDA/Ours 的 light-chain pairing 长度协议决策，以及与 Ophiuchus baseline 比较时的边界；不是将 Ours 改称 Ophiuchus baseline。
§10 仅链接新授权的 Ours 原生探针与生成矩阵；CDR 详细协议仍由其任务文档维护。不扩展到 NbBench、FLAb、MINT 通用 PPI、TCR 或 humanization。

本文是这三个探针的**纠错问题、证据边界、待实现方案与验收条件**的集中记录，不是新的排行榜或数据清单：

- headline 范围仍以 [benchmark README][scope] §0 为准；本次文档工作不将三个探针加入 headline。
- 排行榜数字由 [RESULTS][results] 管理。本文仅引用解释缺陷所需的诊断数字，完整运行数字读取对应原始 JSON，不能将诊断表升级为主榜。
- 数据版本、落地路径、官方对照与 SHA 只在 [PROBE_DATA][data] 维护；不要另建一份相互矛盾的数据台账。
- 已有作业历史见 [PROJECT_PROCESS][process] 和 [benchmark PROGRESS][progress]；历史提交/排队记录不等于当前平台状态。
- 新 agent 先读本文，再读 [probe README][probe-readme]、[数据台账][data]、相关源码及原始 metrics。
- §9 是 pairing 决策入口；单任务可执行协议、缺陷和验收条件仍只在 [AB pairing 任务文档][ab-pairing] 维护，不把三个表征探针的 CV 规则套到生成任务。

证据用语：**已确认**表示当前文件或产物直接支持；**待验证**表示仍需运行或追溯；**拟实施**表示尚未落地；**论文未披露**不能改写为“论文采用了我们的默认值”。

## 2. 论文、AirGen-Dev 与本地实现：分清三层证据

### 2.1 论文明确披露与未披露的内容

依据 [Ophiuchus-Ab 论文][paper] 2026-02-04 版本（[出版页面](https://www.biorxiv.org/content/10.64898/2026.02.02.703197v1)），§2.3.3（第 9–10 页）、Table 5（第 18 页）：

- 冻结预训练参数，训练轻量分类头；正文称为 linear classification。
- HD / Flu / CoV 三分类，类别均衡，采用五折分层交叉验证；指标为 Accuracy、F1、MCC。
- 没有明确分类头 epoch、学习率、随机种子、early stopping、最佳轮次规则，或跨 epoch 平均规则；F1 的 averaging 细节也未在该段明确。
- “without task-specific tuning” 不足以推出任何具体选模/聚合算法。
- §4.3 的 100k steps 等参数描述主模型预训练，不是分类头训练配置。

网页全文/补充材料此前访问受限；本记录对“未披露”的判断限于已核查的本地 PDF，不声称穷尽了未能读取的补充文件。

### 2.2 AirGen-Dev 是重要实现证据，但不是论文运行日志

[原启动脚本][air-spec-run] 明确使用 100 epoch、batch size 8，并遍历学习率 5e-5 / 1e-4 / 5e-4 / 1e-3，开启 use_multimer 与 sep_chains。
[原分类头][air-spec-head] 是 Linear → tanh → Linear，dropout=0；与 [本地分类头][local-spec-head] 一致。
原 Python 的默认 epoch=5 会被启动脚本覆盖，不能只读 argparse 就认定原实验只训 5 轮。

因此，“100 epoch 是本地任意加长”“两层头与 AirGen 原实现不一致”均不成立。
但是，启动脚本中的历史 checkpoint 路径及超参网格，不能单独证明 Table 5 最终使用了哪个 checkpoint、学习率或哪条 CSV 记录。

### 2.3 需要保留的实现差异

- AirGen 主循环读取 TTE 五折文件。当前本地 runner 也优先读 TTE，但缺失时会回退到 StratifiedKFold；正式复现须改为缺折即失败，不能静默换划分。
- AirGen 主流程没有显式设训练 seed；未调用的 TrainingArguments helper 中的 seed=42 不构成主流程已设 seed 的证据。本地按 42 + fold 初始化分类头。
- AirGen 每 5 epoch 评测并追加记录，最后直接对所有记录求平均；这是 [代码行为][air-spec-aggregate]，**不是已确认的论文表格口径**。
- 本地每 epoch 留历史，但 summary.mean 只汇总每折最后一轮；见 [本地聚合][local-spec-aggregate]。不要把它误报成“当前程序也在跨 epoch 平均”。

来源必须区分论文报告、AirGen 代码、公开权重与本地移植 harness。仅有官方权重，不足以把本地重实现笼统标成“官方评测脚本原样复跑”；正式来源层按 [benchmark 规约][scope] 核定。

## 3. Specificity 的纠正协议（后续实现与验收依据）

### 3.1 五折到底选什么 checkpoint？

本任务 backbone 始终是同一个冻结的 Ophiuchus-Ab checkpoint；五折分别初始化、训练**五个独立分类头**。
不能先在全数据上训练一个头，再在五个测试折上轮流打分，也不能让同一个头跨折继续训练。

每折流程必须是：

```text
固定外层 train_k / test_k
  → 只用 train_k 训练该折分类头
  → 按预先声明的规则选定 head_k（不使用 test_k 选模）
  → 用 head_k 对 test_k 计算一次最终指标
  → 收齐 k=0…4 的五组指标，逐指标求五折算术均值
```

五折分层 CV 本身**不规定**必须取哪一轮。固定末轮时五折轮数相同；若只根据各折训练集内部的 validation 早停，各折选中的 epoch 可以不同。
选定“一个状态”不一定要先保存到磁盘才能评分，但为可审计复跑，后续必须保存每折所选 head checkpoint。

### 3.2 两条合法但不能混称的实验路径

**A. 原实现配置复跑 / fixed-last-100**

- 以 AirGen 启动脚本为依据，显式固定 epochs=100、batch size=8、lr=5e-5、原 ESM head、AdamW、10% warmup 与 linear schedule。
- 每折只取第 100 轮状态。禁止看外层测试曲线后改成 ep94、200、300 或挑最佳学习率。
- 这是有代码依据的本地复跑协议，不代表论文已明确采用末轮，也不能宣称盲评：当前测试折已被用于既往多轮数/超参探索，须披露该历史。

**B. train 内选模 / inner-validation**

- 外层五折保持不变；每个外层 train 内另划分分层 validation，预先固定选择指标、方向、patience、最大轮数、并列规则及超参搜索预算。
- 只根据 inner validation 选 checkpoint，外层 test 不参与任何选择。预先声明是直接评估该 inner-train checkpoint，还是按选定超参/轮数在完整外层 train 上重训后测试，不能事后混用。
- 此路径属于新的规范化评测协议；没有额外证据时，不能称为论文原设置复现，也不能消除既往查看测试分数的历史。

A/B 必须使用不同 protocol_id、输出目录和汇总行；本轮仅登记方案，不默认启动两套实验。

### 3.3 指标与历史记录的边界

- 正式本地输出明确使用 Accuracy、macro-F1、MCC；每折各计算一次，summary 对五折等权平均。pooled OOF 指标可附报，但不能与折均值混用。
- 新协议同时保存五折原始值及 sample SD（ddof=1），注明不是置信区间或多 seed 方差。旧 JSON 的 std 使用 ddof=0，保留原值，不静默改写。
- epoch_history 仅用于诊断；禁止对全部 epoch 的指标求平均充当 CV score，禁止取每折外层 test 最佳值，禁止跨 epoch 拼接 Accuracy/F1/MCC。
- 固定并记录 Python / NumPy / Torch / CUDA seed 和每折 seed 派生规则；seed 差异的影响须实测，不能当作已有数值差距的确定解释。
- 稳健性多 seed 运行应预先定义并全部报告，不能只留下最好 seed。

### 3.4 已有结果如何解释、如何保留

[100 epoch 原始 JSON][spec100] 的 mean 是末轮五折均值，Accuracy=0.677804、macro-F1=0.676818、MCC=0.517054；接近论文 Table 5，但不是所有轮次平均。
仅为展示聚合差异，从同一 JSON 的历史按 AirGen 每 5 轮全部平均，得到 0.653464 / 0.650563 / 0.485116。后一行只作**错误聚合口径对照**，既不是正式五折结果，也不能用来认定论文复现失败。

可只读复核，无需 GPU：

```bash
jq '{last_epoch_fold_mean: .mean, diagnostic_only_epoch_average: ([.epoch_mean[] | select(.epoch % 5 == 0)] | {accuracy: (map(.accuracy) | add / length), f1_macro: (map(.f1_macro) | add / length), mcc: (map(.mcc) | add / length)})}' /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_esm_ep100/specificity_hd_flu_cov_metrics.json
```

原始 [5 epoch][spec5]、[50 epoch][spec50]、[100 epoch][spec100]、[200 epoch][spec200]、[300 epoch][spec300] 产物全部保留；后两档已有完整结果，不要因为旧日志只记“已提交”就写成未完成。
原始 5 epoch 文件未被后续覆盖；[run_probe.sh][probe-run] 默认仍走 5 epoch，并且发现 metrics 已存在就跳过，**不是修正后协议的重跑入口**。
当前 100 epoch 目录仅保存 metrics JSON（含 histories），没有逐折 head checkpoint / 逐样本预测；指标可重聚合不等于已完成独立 checkpoint 重评分。

## 4. GDPa1 与 m396：不能机械套用特异性规则

### 4.1 GDPa1（Table 4）

[代码][dev-code] 使用冻结特征、PowerTransformer、Ridge GridSearchCV 和数据指定的固定折。
[现有结果][dev-result] 中四个论文属性的 cv_spearman_transformed 在论文三位小数精度上对齐，可称为**该历史协议下的数值复现**。

待完善事项：

- cv_spearman_transformed 是 GridSearchCV.best_score_（选定 alpha 的折均值），cv_spearman_raw_vs_pred 是合并 OOF 后的相关性；不能混称为同一估计量。
- 当前标签变换在全部标签上拟合，alpha 也在同一组 CV 上选择并报告成绩；这复刻了参考流程，但不是独立外层测试的无偏估计。
- 保留 paper-reproduction 路径；后续执行以 [原生任务 §4.2 的用户决定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#42-预测头与协议切割线) 为准，不再将 nested-CV 列为待实施或验收要求。
- 每个属性独立处理缺失值，保留它实际参与的折和样本索引；输出逐折预测、分数、alpha 与变换器 provenance。
- Ridge 没有分类头 epoch；本任务问题是超参选择、变换拟合范围及折聚合，不是“选第 100 轮”。

### 4.2 m396（Figure 4）

[本地实现][aff-code] 使用 WT−mut 表征差、Ridge(alpha=0.01)，对六档固定训练比例分别评估；主指标是五个能量目标各自的 Spearman 再取平均。
[原 AirGen Ridge 分支][air-aff-ridge] 也有六档比例与五目标均值。这是 low-N train/test 评测，**不是五折 CV**，也没有分类头 epoch。
标签是计算能量/ΔΔG，而不是湿实验 Kd；不得将该成绩解释为已经复现实验亲和力预测。

论文正文 §2.3.2 给出“0.5% 数据时相对 MINT 超过 6%”的描述，Figure 4 没有标出六档逐点精确数值。
[本地雷达图源代码][radar] 写有 Binding Affinity 参照 0.864 与 Ophiuchus-Ab 0.917，相对差约 6.13%；这提示它们可能对应上述低样本档，但该脚本没有写训练比例。
这些是本地绘图常量，**不是已验证的 Figure 4 六档源数据**。本地 [m396 JSON][aff-result] 的 0.5% 档为 0.930065，不能据此宣称超过论文。

必须优先核对的实现差异：

- 当前公共提特征路径是固定 VH/VL token 上限及分链池化后拼接，见 [collator][collator]、[embedder][embedder]；输入文件全长不等于全长都进入模型，需要输出实际截断范围。
- 当前 AirGen m396 main 选择的是 DesautelsCollateFn 而非注释中的 MyCollateFn；前者使用动态 padding，wrapper 的 forward_one 使用全局池化。不可只看到同一个 Ridge/数据集就宣称特征协议一致。
- AirGen DesautelsCollateFn 构造 WT 时两条都使用 wt_heavy_chain，且 main 引用旧 vr FASTA；这是当前源码可见的问题，不证明论文运行时也如此。不要盲目恢复到有疑点的分支。
- 特征差异、WT 处理对分数的实际影响需要受控实验；不能未经验证就把所有分差归因于其中一项（Ridge 带截距时，固定 WT 向量差可能被截距吸收）。
- 当前 y_train / y_test 分别标准化；Spearman 对正比例缩放不敏感，但 MSE 的尺度不可与 train-only 变换结果直接混用。规范化改动单列协议。

验收需要：六档 train/test ID 指纹、split seed、输入/截断/pooling 规范、Ridge 配置、逐目标分数，以及论文曲线来源。没有精确源数据时，图像读数必须注明近似与误差，不能捏造四位小数论文值。

## 5. 纠错台账与关闭条件

| ID | 优先级 | 状态/问题 | 关闭条件 |
|---|---|---|---|
| AB-SPEC-01 | P0 | 已确认：AirGen 输出跨 epoch 平均；论文未规定此口径 | 本地新协议明确一折一头一选定状态；测试拒绝跨 epoch 聚合；原脚本输出仅作历史诊断 |
| AB-SPEC-02 | P0 | 已确认：旧实验查看过多轮数/超参的外层 test | 历史探索明确披露；新 protocol 固定选模规则，test 不进入选模代码路径 |
| AB-SPEC-03 | P0 | 已确认：缺 TTE 可回退；缓存仅凭文件存在复用 | 正式模式 fail-fast；校验每折 ID/标签/序列指纹与缓存协议，任何不匹配失败 |
| AB-SPEC-04 | P1 | 已确认：没有逐折持久化 head/预测，旧 std 未说明 ddof | 保存所选 head、预测、选择记录；独立重评分复现 summary；显式聚合与 std 定义 |
| AB-SPEC-05 | P1 | 未解决：论文最终轮次、汇总规则、训练 seed 不明 | 获取原 CSV/汇总脚本/作者说明，或永久保留 paper-selection-unknown 标签，不以猜测关单 |
| AB-DEV-01 | P1 | 已确认：复现分数与规范化无偏评估是两条协议 | 保留原结果；需要规范化评测时另建嵌套 CV，并保存每折预处理/alpha 证据 |
| AB-AFF-01 | P0 | 已确认：AirGen 与当前输入/pooling/WT 分支不同；影响未验证 | 用同批样本做输入与 embedding 对照，锁定待复现版本；差异实验另存，不覆盖原结果 |
| AB-AFF-02 | P1 | 未解决：六档论文精确源数据未找到 | 获取可追溯源数据；或保留“近似图对照”，不声明逐点精确复现 |
| AB-COMMON-01 | P0 | 已确认：旧入口默认 5 轮/存在即跳过，历史 probe YAML 已不在当前工作树 | 显式版本化配置、新输出目录、失败即停；必要时在现有模板基础上重建 YAML 并验收 |
| AB-COMMON-02 | P1 | 待完善：公开权重、本地 harness、数据与运行版本 provenance 不完整 | 每次运行记录实际权重 SHA、代码版本/dirty diff、数据/缓存指纹、软件环境与参数 |

## 6. 后续产物契约（拟实施，当前不存在）

新协议使用独立目录，例如 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/probe_audit_v2/`，不得覆盖旧 JSON 或旧缓存。

至少记录：

- run manifest：task、protocol_id/version、checkpoint path + SHA256、代码 revision/dirty patch、环境、设备/精度、seed、所有显式参数、数据与缓存指纹、known_deviations、是否曾用该测试集探索。
- 特异性每折：train/test ID、inner-validation ID（若有）、head 初始化 seed、selection_rule、selected_epoch、selected_head checkpoint、预测行（sample_id/fold/y_true/probabilities/y_pred）、逐折 metrics。
- 汇总：五折指标列表、mean、sample SD、n_folds、F1 averaging、aggregation=mean_of_selected_fold_metrics；epoch_history 与 summary 分离。
- 表征缓存：模型权重 hash、序列与顺序 hash、tokenizer、chain layout、截断、pooling、提取层/精度。不能只按文件名或 tensor shape 判断兼容。
- GDPa1 保存 fitted Ridge/transformer、fold 与 alpha；m396 保存每档 fitted Ridge、split ID、逐目标预测/指标及 target scale。

论文值、本地复跑值、图像估读值分别标来源；不能将数值接近、公开权重、脚本同名当作严格复现验收的替代品。

## 7. 后续 agent 的实施顺序

1. **先读，不重跑**：读取本文、[数据台账][data]、论文、AirGen 与本地代码；检查工作树已有改动并保留。确认三个探针仍不进 headline。
2. **先明确协议**：Specificity 优先补 A 路径的显式配置、checkpoint 持久化与正确聚合；B 路径作为单独方案。m396 先做输入/特征差异审计，GDPa1 保留已数值对齐的复现协议。
3. **实现轻量测试**：用合成数据检查折独立性、选模不读 test、指标聚合、保存/加载头、缓存失配和缺折失败；不要用完整 GPU 重跑代替单元测试。
4. **恢复可审计配置**：历史 probe YAML 当前缺失，不能照抄文档中的失效命令。重建时使用现行 eval_jobs 模板；逐项核对环境（旧 run_probe.sh 硬编码 protenix_abtcr）、参数、数据、权重和全新输出路径。运行脚本前遵循项目 AGENTS 的 pllm 环境要求，GPU worker 环境须在配置中显式说明并检查依赖。
5. **获得实施/运行授权后再执行**：本次用户要求的是文档，不是立即提交评测。未来 GPU 评测通过 Volc 的 eval_jobs 配置提交；submit/cancel 必须同步 [PROJECT_PROCESS][process] 的 Active 表与 dated log。
6. **先独立验收再写结论**：从保存的 checkpoint 与预测重新评分；完成下节检查。数字更新只进入其权威文档，其他入口放链接；不得自行解冻 headline 或修改无关 baseline。

## 8. 表征探针验收清单

- [ ] 五折训练/测试互斥、ID/标签/序列覆盖与既定 TTE 对齐；正式模式缺折失败，绝不静默重新划分。
- [ ] backbone 无参数更新；五个 head 独立初始化，无跨折训练状态复用。
- [ ] Specificity 的 selection_rule 在评分前固定；测试标签/测试指标不能进入分类 head 或其超参选择过程。GDPa1 按 §4.1 参考 GridSearchCV 流程执行并披露其局限，不套用本条来要求新增 nested。
- [ ] 每折只有一个 selected head；各指标来自同一状态，禁止跨 epoch/超参/seed 拼接最优项。
- [ ] 五折 macro 指标可从逐样本预测重算；summary 与五个所选状态逐项一致；history 的长度不会改变 summary 的聚合逻辑。
- [ ] 保存/加载所选 head 后的预测一致；断点恢复与缓存命中有 provenance 校验。
- [ ] GDPa1 的选 alpha、变换拟合范围和折均值/pooled OOF 明确分开；只收参考流程结果，不要求 nested-CV 或覆盖历史产物。
- [ ] m396 的六档 split、实际进入模型的序列、pooling、WT、逐目标 Spearman 均可追溯；不将 low-N 误写成五折。
- [ ] 论文未披露内容保持 unknown；雷达图常量/图像估读不伪装成 Figure 4 原始表。
- [ ] 旧产物完整保留；新旧协议分目录；文档清楚区分“已登记”“已实现”“已运行”“已验收”。

## 9. Light-chain pairing：采用 reference 长度的用户决策（2026-09-13）

**新增步数诊断（2026-09-16）**：用户要求无初始三个轻链残基时测试8/16/32/64/96/128步的IM，具体控制条件、完整候选分母、固定49000快照、124步参照和提交方案统一见 [AB pairing §4.5][ab-pairing]。这是额外无前缀诊断，不撤销reference长度决定，不自动替换有前缀主表，也不恢复CFG扫描。

**相关 CDR 授权（2026-09-16）**：共享ESMC反馈修复也适用于CDR多步调用。用户已要求本地测试多个iter，先按 [AB CDR §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md) 运行完整SAb23H2、92000、iter1/2/4/8。旧iter1主结果保留；新多步仅作诊断，不能用测试集逐CDR最优步数替代主行。执行及产物状态只维护于任务文档／PROJECT_PROCESS。

**2026-09-16 更新优先于本节旧完成状态**：用户要求 ESMC 在每轮读取模型已接受的生成残基，但不读取隐藏 ground truth。逐步回填、词表转换、防参考泄露及 CFG 的准确协议见 [任务文档 §7 k][ab-pairing]。已实施 v5 修复和 CPU 回归，同一 49000 权重的 p0/p3 全量重评及产物验收已完成；v3/v4 旧 pairing 数字仅为修复前历史，不冒称 v5 新结果。reference 长度与 p3 允许前三残基的决定不变，p0 不提供真实轻链残基；训练权重及旧产物不改。

**用户已确定：当前 Ours pairing 后续主协议直接采用 reference 轻链的长度。已知长度是允许的任务条件，不再要求必须隐藏长度、从训练先验随机抽样，或先实现 `<protd>` 自主终止。**

理由是当前模型训练时 PAD 不参与 attention，恢复任务的有效区间按真实序列长度确定；推理直接开放超长 MASK 窗口并不能视为已验证的等价适配。采用真实长度是本阶段与现有训练/接口相适应的合理评测方式，不代表已经证明模型在原理上只能使用真实长度。源码与 attention 证据见 [任务文档 §7 i/j][ab-pairing]。

需要准确命名和解释结果：

- 本任务称为 **reference-length-conditioned light-chain generation（给定真实轻链长度的条件生成）**，不是未知长度的自由生成。长度已知本身不再是该协议的违规项，也不能单凭长度匹配率高就判为序列泄露。
- 保留重链及 prompt3 允许的轻链前三残基；除长度与这三残基外，不允许真实轻链内容进入 decoder 或 ESMC 条件。允许长度不等于允许答案序列。
- 不改动官方 Ophiuchus-Ab 的原生生成协议。与其论文值或原生复跑值并列时必须显式标明长度条件不同，不能称完全同协议复现或据此直接宣称模型优劣；如另做同长度条件 baseline 对照，应单独标记为适配协议。
- 旧 prior / reference 产物原样保留，新协议独立记录。此次决定不自动恢复旧 reference 数字的可引用性，也不关闭已确认的 sampler 状态问题或真实序列泄露问题。

后续 agent 按 [任务文档 §4 与 §8][ab-pairing] 实施：先修正并验证 pending-mask 状态与双路序列遮蔽，再显式使用 `--light-length-mode reference`（脚本适配见任务文档），使用独立输出 tag 和新协议记录。`prior` 仅保留为可选诊断，不再是该主协议的前置要求；自主长度建模不作为本阶段评测的阻塞条件。

**实施进展**：用户随后授权开始测试，pairing 代码变更、CPU 回归与完整 holdout500 的生成、现有评分及产物核验均已完成。最终数字、与旧结果/原生 baseline 的比较及解释边界见 [RESULTS §0.6 全量][results]；准确实现、验证范围和产物入口见 [任务文档 §7 h][ab-pairing]，平台终态见 [PROJECT_PROCESS][process]。以全量取代 pilot 的性能判断；本次联合改变 sampler 与长度条件，单因素修复贡献尚未隔离，不能宣称同协议复现论文；`W_property` 仍未实现。表征探针 §2–8 的协议与状态不变。

## 10. 我们模型上的原生 AB 测评（2026-09-13 新授权）

前文固定150/128 token、EOS补齐、分链pool拼接描述的是 Ophiuchus baseline，不能套到 Ours。我们需要原生抗体 grammar、动态 PAD、完整 ESMC→LLaDA、有效残基 global pooling；不能直接调用默认构造 TCR 记录的通用 `embed_pairs()`。原生输入、冻结头、缓存/产物契约及执行手册唯一见 [AB_NATIVE_PROBES](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。

Specificity 采用一折一个固定末轮头；GDPa1 当前协议见 [原生任务 §4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#42-预测头与协议切割线)。m396 的行随机与序列分组协议仍分开保存：数据中完全相同序列可能跨行随机 train/test，分组对照不覆盖原始复现结果。实际输入长度、完整突变位点与既有 baseline pooling 差异仍不能混为同一问题。

pairing 的无前缀/前三残基 × CFG 原生 v4 定义与修复证据只在 [任务 §4.4][ab-pairing] 维护；CDR 使用现有 [CDR 任务](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md) 的 Kong/converted 入口，不能回退旧快照。

本轮配置方案为6个pairing、2个CDR数据集、3个原生探针，共11个非闲时非抢占单卡作业。用户已指定使用最新 v5，已选择查询时最新完整保存点并制作独立快照，矩阵内不再追逐后续训练步数。GPU 门禁通过，具体 YAML、权重 SHA、提交和正式产物状态以 [PROJECT_PROCESS][process] 为准。此授权是附加测评，不自动改变 headline 或解冻无关 baseline。

[ab-pairing]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md
[scope]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md
[results]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md
[data]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/PROBE_DATA.md
[process]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md
[progress]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md
[paper]: /root/oph_paper/oph.pdf
[probe-readme]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/README.md
[probe-run]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/run_probe.sh
[air-spec-run]: /vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/run/specificity.sh:7
[air-spec-head]: /vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/specificity/HD_Flu_Cov-paired.py:51
[air-spec-aggregate]: /vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/specificity/HD_Flu_Cov-paired.py:415
[local-spec-head]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/specificity.py:33
[local-spec-aggregate]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/specificity.py:220
[spec5]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_hd_flu_cov_metrics.json
[spec50]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_esm_ep50/specificity_hd_flu_cov_metrics.json
[spec100]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_esm_ep100/specificity_hd_flu_cov_metrics.json
[spec200]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_esm_ep200/specificity_hd_flu_cov_metrics.json
[spec300]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/specificity_esm_ep300/specificity_hd_flu_cov_metrics.json
[dev-code]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/developability.py:84
[dev-result]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/developability_gdp_a1_metrics.json
[aff-code]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/affinity_m396.py:125
[aff-result]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ophiuchus_ab/affinity_m396_metrics.json
[air-aff-ridge]: /vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/in_silico/finetune_in_silico.py:398
[radar]: /vepfs-mlp2/c20250601/251105016/project/airgen/draw/draw_rida.py:22
[collator]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py:34
[embedder]: /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/embeddings.py:56
