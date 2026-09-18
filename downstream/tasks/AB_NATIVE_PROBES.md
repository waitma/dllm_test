# AB Native Frozen Probes — 我们的 ESMC + LLaDA 模型

> 先读 [PROJGUIDE](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md)。本文件管理 Ours 的三个 AB 表征任务；Ophiuchus baseline 的原脚本、历史结果与审计保持独立。

## 0. 元信息

| 字段 | 值 |
|---|---|
| 任务 ID | AB-NATIVE-PROBES |
| 锚点论文 | Ophiuchus-Ab Table 5 / Table 4 / Figure 4；[baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md) |
| RESULTS 锚点 | §0.6a：2026-09-14 v5 49000完整结果；不自动进入headline |
| 主指标 | Specificity Accuracy/macro-F1/MCC；GDPa1 分属性 Spearman；m396 五目标平均 Spearman |
| 数据根 | [PROBE_DATA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/PROBE_DATA.md) 为唯一数据台账 |
| 我们的入口 | `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/ab_probes.py` |
| 可引用性 | 三任务完整产物已落盘，保存预测重算核对；不同CV/split分开报告，见§7限制 |
| 最后更新 | 2026-09-15 |

## 1. 任务定义与范围

用户明确要求在我们的模型上测试 Specificity、GDPa1、m396，不是继续把 Ophiuchus 的输入适配套在 Ours 上。生成任务仍分别由 [CDR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md)、[pairing](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md) 管理。

### 1.1 模型适用性矩阵

| 模型 | 本文件表征任务 | 训练范围 |
|---|---|---|
| Ours-Diffusion | yes | backbone 冻结，只拟合分类/回归头 |
| Ours-BERT | yes | 同上；不得因此启动 BERT 生成任务 |

### 1.2 不在本文范围

NbBench、FLAb、TCR、humanization 不由本轮授权解冻。本文件不修改预训练配方、权重或 Ophiuchus baseline 脚本。

## 2. 参考文献调研

### 2.1 论文实验全景

| 实验 | 数据/目标 | 本次对照 |
|---|---|---|
| Table 5 | 配对抗体 HD/Flu/CoV，五折分类 | 相同数据与 TTE，Ours 原生输入；固定末轮头 |
| Table 4 | GDPa1 四项主要属性，五折回归 | 仅运行 Ophiuchus 参考流程 CV；不再运行 nested，Tm2 仅附加 |
| Figure 4 | m396 五计算能量，六档 low-N | 行随机划分与序列分组划分分开保存；不是实测 Kd |

### 2.2 指标定义

分类每折只评估选定的一个 head，汇总五折等权均值及 sample SD(ddof=1)。GDPa1 区分折均值和 pooled OOF；m396 先分别计算五目标 Spearman，再平均。所有额外协议必须独立标记，不能跨协议选最好项拼表。

### 2.3 未公开细节

论文分类头最终选模方式及 m396 六点精确论文源数据仍未知；见 baseline 审计。采用原生输入并不意味着严格复现 Ophiuchus 特征协议。

## 3. 官方产物 release 与获取状态

数据已经落地，本轮不重新下载或覆盖。源文件、版本与 SHA 只在 [数据台账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/PROBE_DATA.md) 维护；实际 run manifest 重新记录读取文件 SHA。Ours checkpoint 必须明确选择；训练中被 top-k 清理的目录应先制作独立评测快照。

## 4. 我们的协议

### 4.1 输入与入口

原生表征实现：[ab_features.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/ab_features.py)。

- 显式构造 `task_type=antibody`，角色为 heavy/light；原生 grammar、null-context 前缀、tokenizer remap、动态 PAD。不得用通用 TCR `embed_pairs()` 冒充 AB 输入。
- 完整序列、不加噪声、不输入类别/属性标签。超长失败，不做 Ophiuchus H148/L126 截断或 EOS 填充。
- ESMC 条件 → LLaDA 最后一层 → 全部有效残基 global mean pooling，特殊 token/PAD 排除。270M 通常 768 维，实际维度读取所选权重。
- m396 分别编码 full-length WT 与突变体，再做 WT−mut；不把四条链拼在同一条记录。
- 特征缓存验证权重、tokenizer JSON、序列顺序、源码、精度与配置；缺乏 provenance 的旧 Ophiuchus 缓存不得复用。

### 4.2 预测头与协议切割线

| 任务 | 执行方式 | 边界 |
|---|---|---|
| Specificity | TTE 缺失即失败；五个独立 ESM-style head；100 epoch / batch8 / lr5e-5 / seed42+fold；只用 train 训练，保存末轮并从该状态评分 | 无 test early stopping；历史探索已看过该测试集，不称新盲评 |
| GDPa1 reference CV | 全标签 PowerTransformer + Ridge，同一组数据指定五折选 alpha 和报告；保留 OOF | 用户选定的唯一当前协议；参考流程有选择偏差，不称独立外层成绩 |
| m396 row_reference | 原行随机六档、seed2023、Ridge alpha0.01，train/test 分别 z-score | 有完全相同序列跨 train/test；保留并报告重叠数 |
| m396 sequence_grouped | 六档 GroupShuffleSplit，按完整 H/L 序列分组；train-only 标签变换，预测还原原尺度 | fraction 是独立序列组比例，实际训练行数与旧协议不同；必须报告，不混排 |

**2026-09-15 用户决定：不需要 nested。** GDPa1 入口已移除 nested 分支，只运行上表参考流程，不新增 opt-in nested 任务。主指标仍是 `GridSearchCV.best_score_`，不是 pooled OOF；全标签变换、alpha 网格、固定折与参考分数计算均不改变。新产物协议 ID 为 `native_ab_gdp_reference_only_v2`；保留 `legacy_reference` 字段及文件名用于兼容既有读取器，不表示该参考流程被停用。已有 nested JSON/OOF/回归器原样保留，仅为历史诊断，不作为当前主结果，也不要求给 baseline 补跑 nested。此决定不改变 Specificity、m396 或生成任务。

### 4.3 可审计产物

每个任务单独目录：`run_manifest.json`、`feature_manifest.json`、`features.pt`、`metrics.json`。Specificity 保存每折 head、selection、训练历史与逐样本概率/预测及 OOF CSV；GDPa1 每属性保存 `legacy_reference.joblib`（选定 alpha 后全数据 refit 的 Ridge 与全标签 transformer）及 `legacy_oof.npz`（行索引、折号、原始/变换标签与 OOF 预测），不再生成 `outer_*` 或 `nested_oof.npz`；m396 保存每档 fitted Ridge、训练变换、train/test 索引及逐目标预测 NPZ。已有 metrics 拒绝覆盖，正式运行仍须使用新目录。

### 4.4 Specificity 分类头对齐核查（2026-09-15）

- 原 AirGen [EsmClassificationHead](/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/downstream/specificity/HD_Flu_Cov-paired.py:51) 与 Ours 实际导入的 [本地实现](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/ophiuchus_eval/specificity.py:33) 都是 `Dropout(0) → Linear(D,D) → tanh → Dropout(0) → Linear(D,3)`；原文件的 `MLPClassifier(…,512,…)` 没有被主循环使用。原生训练调用见 [ab_probes.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/ab_probes.py:110)。
- “一致”指结构模板，不指输入宽度/参数量：AirGen `--sep_chains` 对1280维重/轻链各自pool再拼接，D=2560、头参数6,563,843；本轮Ours使用post-LLaDA global pool，保存的fold0头D=768、头参数592,899。不能因头模板相同就忽略读出方式和容量差异，也不能从该差异单独推出成绩差距的原因。
- 原启动脚本是100 epoch、batch8，并扫四档学习率；Ours固定lr5e-5、AdamW、交叉熵、10% warmup+linear decay，固定100轮头，seed42+fold。这些设置有原脚本依据，但论文最终学习率/选头记录未知；原主流程没有显式固定seed。
- CPU核查直接从原文件AST提取该头类，分别在D=768/2560加载同一state_dict后，原/本地输出逐位一致；并读取本轮保存头确认dense/out_proj形状和fixed_last元数据。没有重训、改头、重提特征或加载基础模型；这不等于复算整次Specificity性能。

### 4.5 Specificity 200 epoch 学习曲线诊断（2026-09-15）

- 用户要求检查LR、对照Ophiuchus本地记录，并尝试200轮。独立实验仍固定v5 49000快照、同一4398条/TTE五折、768维头、batch8、lr峰值5e-5、AdamW、seed42+fold；只训练分类头，不动基础模型或现有100轮结果。正式实验YAML：[eval_ab_v5_49000_specificity_ep200_20260915.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_49000_specificity_ep200_20260915.yml)。
- 从头初始化，不从100轮末尾续训。10% warmup+linear decay保持同一策略，但完整预算由100变200，warmup由约10轮变20轮，LR在200轮末归零；新实验第100轮不是旧100轮实验的同一优化状态，不能当作完全一致前缀的续训比较。
- 新增 `--log-epoch-metrics`，仅Specificity可用，默认关闭，不改变其他任务。每折逐轮保存训练loss、留出折test_loss/Accuracy/macro-F1/MCC与该轮首/末/下一步LR，文件为 `epoch_metrics.json`；每20轮及最后一轮保存 `diagnostic_head_epoch_*.pt`。五折全部完成后写 `epoch_mean.json`。每折只以固定第200轮的 `selected_head.pt` 生成正式OOF与metrics，曲线不用于early stopping、选学习率或选最好轮次，也不跨epoch平均正式成绩。
- 这里的“验证曲线”实际为原TTE留出折诊断，没有新增nested或内部validation split；由于查看过历史结果，不称新盲评。原100轮仅有train_history及末轮预测，不能据此声称留出折已收敛。
- 复用原特征的独立副本，现有严格cache provenance核验权重、数据顺序、源码、精度、配置；真实缓存CPU门禁通过（4398×768、五折），不重提特征、不修改源缓存。新目录：[ep200](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915)。生产入口拒绝已有run manifest，禁止覆盖或自动重启旧目录。
- 新增观测回调CPU测试确认同seed下训练loss及最终state_dict逐位不变；总10项原生探针CPU测试通过。YAML单卡/非闲时/独立输出与Bash语法门禁通过。实际任务状态只读 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，不把门禁当完整200轮结果。
- Ophiuchus本地50/100/200/300轮记录属于诊断对照，不是论文baseline或新重跑；准确数值与局限统一见RESULTS §0.6a学习曲线段。
- 生产任务 `t-20260915103954-2sw2c` 已Success（2026-09-15T02:39:55Z–02:47:04Z，429秒）。固定第200轮五折均值：Accuracy 0.627325、macro-F1 0.625782、MCC 0.441693，sample SD分别0.012818/0.012705/0.019362；train loss 0.742728、留出折loss 0.838495。平均留出折loss在第200轮为全曲线最低；各折自己的最低loss位于141/184/193/178/187轮，末轮均仍低于同一次运行第100轮，说明后半程是缓慢改善伴随波动，而非明显反弹。曲线最大Accuracy 0.630281位于191轮，只作诊断，正式结果仍固定末轮。
- 验收：五折各有连续1–200轮、10个每20轮诊断头和最终selected head；末轮`lr_next_step=0`。`epoch_mean.json`逐字段重算最大误差0，`metrics.json`折指标/均值及第200轮一致性最大误差0；4398条OOF独立复算为Accuracy 0.627331、macro-F1 0.625866、MCC 0.441598，与正式“各折等权均值”的约1e-4差异仅来自聚合口径。来源：[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915/metrics.json)、[curve](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915/epoch_mean.json)、[OOF](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_49000_specificity_ep200_20260915/oof_predictions.csv)。

### 4.6 v5 step92000 同协议对照（2026-09-15）

- 用户要求用“最新89000”测试。磁盘与训练日志核对表明step89000已被训练top-k清理，当前最新完整保存点为step92000；因此按用户“最新”意图执行step92000，并在全部产物中保留真实步数，不标成89000。
- 冻结快照：[ab_v5_92000_llada_20260915](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/ab_v5_92000_llada_20260915)，`trainer_state.global_step=92000`，权重SHA256=`2f003a5b42a15498a7d6f2fc51414cad717f96a9cb405dca2388fcef0de3039c`，与源文件一致。只复制评测需要的权重/tokenizer/JSON，不复制optimizer；训练任务继续运行不受影响。
- 比较协议与§4.5完全相同，只替换冻结基础模型checkpoint：同一4398条/TTE五折、完整H/L原生输入、post-LLaDA有效残基global mean、768维、fresh classification heads、200轮、batch8、lr5e-5、seed42+fold、fixed-last；曲线仅诊断。因checkpoint改变，必须重新提取特征，严禁复用49000缓存。
- 任务 `t-20260915132856-tlxtc` 已Success（477秒）；YAML：[eval_ab_v5_92000_specificity_ep200_20260915.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep200_20260915.yml)，输出：[step92000目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915)。固定第200轮Accuracy=`0.643468 ± 0.015785`、macro-F1=`0.641730 ± 0.015289`、MCC=`0.466119 ± 0.024016`；train/eval loss=`0.707176/0.821146`。相对step49000同协议提升`0.016143/0.015948/0.024426`。
- 五折各200条曲线、10个诊断头、最终头与4398条OOF完整，LR末轮归零；曲线/折指标/均值/第200轮复算误差0。平均eval loss在第200轮最低，各折最低轮次`102/184/173/119/139`；第189轮瞬时Accuracy最高`0.646197`只作诊断，正式仍为固定末轮。来源：[metrics](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915/metrics.json)、[curve](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915/epoch_mean.json)、[OOF](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_v5_92000_specificity_ep200_20260915/oof_predictions.csv)。

### 4.7 v5 92000：300 / 400 / 500 epoch 独立预算对照（2026-09-15）

- 2026-09-15追加用户授权的独立100轮：沿用本节全部数据/缓存/初始化/固定末轮规则，warmup为预算的10%；不是200轮的中间检查点。[100轮YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep100_20260915.yml)。监控通过`--extra-job 100:TASK_ID`接入，沿用完整验收后发布门禁。
- 后台[验收发布器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/monitor_ab_specificity_budgets.py)每30秒检查这三项任务；只有平台Success且完整产物核验通过才回填RESULTS的三条预算行，并同步PROJECT_PROCESS对应终态/Active行。4项发布器CPU测试通过；保留基线、旧结果及其他任务，不修改生产任务或配置，不自动重试失败训练。
- 用户在查看92000的200轮曲线后，要求测试300/400/500轮。三项均固定§4.6的92000快照、4398条/TTE五折、768维表征、batch8、AdamW、lr峰值5e-5、seed42+fold；只训练分类头，不改基础模型和数据，不从200轮末尾续训。
- 每个预算独立初始化；10% warmup分别约30/40/50轮，再线性衰减到各自末轮的0。因此比较的是完整训练预算（含scheduler时长），不是同一500轮训练的三个中间检查点。正式只取预先指定的第300/400/500轮，不使用留出折early stopping或best-epoch替换末轮。
- 三项复用已完成92000/200轮的features.pt独立副本；先核对源/副本字节一致，再通过现有cached_features校验权重、checkpoint JSON、序列顺序、源码、torch版本、精度、batch和最大长度。不得复用49000特征；原200轮缓存/预测不覆盖。
- YAML：[300轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep300_20260915.yml)、[400轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep400_20260915.yml)、[500轮](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_ab_v5_92000_specificity_ep500_20260915.yml)。每项独立单卡、非闲时；输出各自的output/downstream_generation/ab_v5_92000_specificity_ep{300,400,500}_20260915目录，拒绝已有run manifest，保留逐轮曲线、每20轮诊断头、最终selected head及OOF。
- 验收继续核对五折完整epoch序列、末轮LR=0、selected_epoch、OOF唯一覆盖、逐折指标/mean/SD以及曲线聚合；只在完成后写入RESULTS §0.6a。本文不提前填分数；任务状态及提交ID唯一见[PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。这三项是用户授权的诊断对照，已观察过的测试曲线不构成新盲评。

## 5. Baseline 清单与排行榜构造

### 5.1 论文值优先

当前AB汇总遵守 [PROJGUIDE §0.2.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#022-排行榜构造论文值优先强制2026-09-02-起) 的用户决定；Ours置后，协议差异就地披露。Ours本轮完整数字及核验边界见RESULTS §0.6a。

### 5.2 Baseline 表

| 方法 | 来源 | 本轮处理 |
|---|---|---|
| Ophiuchus-Ab | 论文 + 历史官方权重本地 harness | 保留，不修改输入或旧结果 |
| Ours ESMC + LLaDA | 本文件原生适配 | 已固定用户指定的最新 v5；提交/状态见 PROJECT_PROCESS |

## 6. 结果写入位置

数值仍统一由 [RESULTS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) 管理；本轮为附加测评，获得完整且通过核验的产物前不新增成绩。提交/平台状态只在 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 维护。

## 7. 已知缺陷与不可比口径

- 冻结探针不是全参数微调；当前代码没有新增 full-finetuning 协议。
- m396 原始行随机拆分的序列重叠、参考输入长度/pooling 差异不能被新 Ours 结果掩盖；按两套 split 分开解释。
- GDPa1 参考 CV 有全标签变换及同 CV 选参/报分的局限，须持续披露；当前只采用 §4.2 用户选定的参考流程，不按结果高低切换协议，历史 nested 不代表当前口径。
- 新模块已通过 CPU 原生输入、保存/重评分和 synthetic CV 测试，以及真实 v5 checkpoint GPU 预检；门禁通过不等于整任务完成。正式运行与产物状态见 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。
- `pllm` 缺 joblib/sklearn；不改共享环境。新表征测试/worker 用已有 `protenix_abtcr`，启动前仍按 AGENTS 激活 pllm，再明确切换运行环境。
- 2026-09-14：Specificity保存预测五折均值/SD、GDPa1两协议5属性OOF、m396两协议六档预测/索引/重叠均已重算核对。Specificity独立CPU重载head与原GPU logits有约0.00145的最大绝对差，4398条中1条argmax改变；生产脚本原设备保存/恢复一致性已验证，不能声称跨硬件逐位复现，也不以CPU诊断覆盖已保存GPU指标。来源与数字只在RESULTS §0.6a维护。

## 8. Agent 执行手册

### 8.1 提交前 gate

- [x] 用户已指定用最新 v5；选择与 SHA 见平台账本，不能把旧42000与v5混用。
- [x] 真实 checkpoint 完整加载、原生输入与 GPU 小样本前向通过。
- [x] CPU 数据/头/分组/产物测试通过；不等于完整模型分数完成。
- [x] 新目录、独立权重快照、所有条件固定；单卡、非抢占、非闲时。

### 8.2 配置与提交

配置生成器 [prepare_ab_native_jobs.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/prepare_ab_native_jobs.py) 必须显式传 `--checkpoint /abs/chosen_checkpoint --tag ab_native_<unique_tag>`；可加 `--snapshot-checkpoint` 防训练 top-k 清理。它只生成配置，不提交。

生成矩阵为 11 个独立单卡作业：pairing 六组（任务文档 §4.4）、CDR 两数据集、三个表征任务。资源为 `c20250601`、`ml.pni2.3xlarge`、1 replica、`Preemptible:false`，每任务上限12小时。确认配置后逐个执行生成器输出的 `scripts/volc-no-proxy.sh ml_task submit --conf /abs/eval_jobs/...yml`，每次提交立即更新平台账本。

加 `--with-preflight` 生成独立单卡门禁 YAML，先于正式矩阵运行。[preflight_ab_native.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_ab_native.py) 加载真实快照，检查三个任务的16样本原生表征（含最长样本）、六种 prompt/CFG 的2重链×8候选、两套数据所有9个 CDR 模式。预检 pairing 仅4步，用于执行/长度/前缀不变量，不作为质量测评；正式 pairing 仍124步。通过标志为预检目录 `passed.json`。

### 8.3 收数

检查 `metrics.json` 与保存预测的一致性。GDPa1 只收 `properties[属性].legacy_reference.cv_mean_spearman`，与 Ophiuchus 参考流程比较；不要改取 pooled OOF 或历史 nested，也不要把缺少 nested 字段当作未完成。m396 仍按两个 split 协议分别收数，Specificity 必须复核五折所选头的指标。不要把任务进入 Queue 或文件存在当作完成。

### 8.4 填表

仅验收后填独立协议结果；不得由这次实施自动改变论文 headline。

### 8.5 文档同步

方法更新本文件；进度更新 PROGRESS；提交更新 PROJECT_PROCESS；数值更新 RESULTS；总审计只链接，避免重复台账。

## 9. Changelog

| 日期 | 变更 |
|---|---|
| 2026-09-15 | 按用户决定移除 GDPa1 nested 执行；参考流程和兼容字段保留，历史产物不改，当前结果只展示参考 CV |
| 2026-09-13 | 用户授权原生 AB 探针及 prompt/CFG 分组单卡测评；实现与 CPU 验收完成，checkpoint 未确定，未提交 |
| 2026-09-13 | 用户指定最新 v5；独立快照及真实 GPU 门禁通过，正式矩阵执行以平台账本为准 |
