# BioSeq Model Plan

## 2026-10-07T06:11:39.860103+00:00 V5固定对象容量、自主生成分隔与终点

已完成用户要求：当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 中full_capacity=512，表示整个受体对象的容量（含`.`与`<protd>`），不再要求真实生成长度。固定目标头部；真实残基／分隔／终点监督，synthetic与尾部mask不计loss，512位均参与attention。CDR-only仍保持先前全长scaffold及mask补位，VHH继续不在本版训练范围。目标role/scope不按真实边界编码；denoise只按腐蚀后当前可见`.`分链和重置位置，保留可见链内／全局交替以及残差1.0。

训练入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py 使用 --input-layout V5 --variant cv_segment_global --object-capacity 512；统一生成及CDR入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，模型generate也接入同一解析器。支持整体生成与已知前链→生成后链；反向补链需另定条件协议，明确拒绝隐藏参考边界。无终点／非法链结构报告失败。此记录替代此前“真实长度／generation_lengths”的当前版本要求；历史A4及冻结训练不改。

83项测试通过，7来源1250条抽查通过，130513个synthetic补位保留；不是全量数据扫描。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_validation.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_regression.log，说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md。临时保存恢复测试权重已清理。未提交新训练，无本版下游成绩。

## 2026-10-07T06:01:00.599765+00:00 原消融与新工作区输入布局保持分离

因工作区已另行准备V5对象布局，本轮只读核对原消融的源码绑定。八个已启动任务的run_provenance仍为A4，其50份记录源码与原冻结目录逐项SHA一致；四份pending票据与原catalog逐项相同，绑定的原dispatcher以隔离Python入口执行冻结runner。工作区layout.py与冻结layout.py的SHA确实不同，但这些已登记消融没有采用该工作区版本。原22格报告继续对应其原配置和来源；不能把这批结果标为新V5布局的训练结果。

证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/source_isolation_20261007T055915Z.json（SHA 8665060e3208b90daa07be8ab93d26b3915fe88196aa3120477d893974dfeb42，实际e85d0f rc0）。本项核对覆盖磁盘源码、既有运行来源及待执行票据，不是新的内存模块检查，也不替代pending任务的原启动门禁。没有加载模型、读取原始序列、改票据、重启/提交任务或新增checkpoint；当前仍等待同一会话3185的3250步指标。

## 2026-10-07T05:55:11.432519+00:00 CDR-only TCR 保留全长并以 mask 补位

当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 partial_sequence_policy：只有 CDRβ／CDRα 的记录沿用先前 prepared 全长 scaffold、区域及长度，送入模型时按 synthetic_residue_mask 将缺失位置转为 <mask>；保留真实CDR及锚点，真实观测X不替换。全缺失配对链沿用已有scaffold全mask。补位参与注意力但labels=-100，不计重建损失，不重新采样长度或缩短为CDR-only。实现 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py，仅改变新V5布局；历史A4不变，VHH仍不在本版训练范围。43项测试通过；1250条真实数据抽查中620条含synthetic，共130513个补位检查通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/scaffold_mask_validation.json 与 scaffold_mask_regression.log。本次未提交训练、未生成checkpoint。

## 2026-10-07 当前新训练版本排除 VHH

用户明确：本版不需要考虑 VHH。版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 training_scope：常规抗体 H/L、TCR 在范围内；nanobody_vhh 链和 nanobody／nanobody_antigen 任务不纳入本版训练、专门评测或验收。版本说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md 已移除当前训练格式表中的 VHH 行。通用 renderer／兼容性测试保留，不代表本版训练覆盖；本次只登记范围，不修改现有语义数据或运行时过滤逻辑，后续本版训练清单构建须落实排除。

## 2026-10-07 V5对象布局的新版本

按用户要求，新版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 采用 `ablation=V5`、`cv_segment_global`。context在前，关系居中，受体对象内用`.`连接H/L或beta/alpha，无条件使用固定null前缀；恢复变长scaffold及V5结构token训练，synthetic不计损失。保留紧凑词表、无ESMC、残差1.0、整链/全局交替。全部7来源复用最终语义JSONL，通过collator转换；不改历史A4配置。训练入口显式 `--input-layout V5`，下游新适配 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，生成必须显式声明长度。配置与格式说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md；验证 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/validation.json。尚未提交新训练，无新checkpoint或下游成绩。

## 2026-10-07T05:41:32.629026+00:00 3000中途指标仍须分项判断，继续完整消融

已核实3000同曝光点；残基CE、准确率与context/结束标记并非一致变化，不据此确定统一最优改动。具体数字与来源限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新3000章节，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3000_root_review.json。维持已完成的正式f011与原解码，继续22格3750完整epoch；不按中途或下游test结果修改长度、解码或选择性重跑。两个seed、有限context、混合源码及历史退出未知的解释边界保持。本轮没有新增模型实现或训练配置。

## 2026-10-07T05:31:28.961Z 用户新版本第二项：链内全注意力与全局交替

在第一项无残差缩放基础上，移除链内radius32，复用cv_segment_global：1/3/5/7层整个同链块注意力，2/4/6/8层全局；attention/FFN残差仍为1。新配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_chain_global_no_residual_20261007/research_config.json 接替第一阶段f001配置作为本次新版本选择；旧f001/f011定义、冻结任务和数据不改。其余A4/主干/词表/容量/位置编码保持，参数75,601,176。

完整模型CPU检查通过：全部8层scale=1、同链distance99允许、仅全局层跨链、padding key排除、同链层前缀隔离。已有回归4 passed/20 deselected，覆盖梯度隔离、非重入checkpoint和临时保存重载。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_chain_global_no_residual_20261007/model_check.json 与 regression.log。未新训练、无新效果结论、无保留测试权重；新任务封版/启动尚未执行。

## 2026-10-07T05:28:51.659Z 用户新版本第一项：移除固定残差缩放

已从正式f011/50082配置派生 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/research_config.json，复用现有f001。唯一架构差异为attention/FFN残差系数0.1→1；保留局部radius32/全局交替、A4及主干/词表/role/scope/RoPE/chain bias。完整75,601,176参数CPU实例检查8层两类scale均1、mask交替正确；既有因子配置和单因子数值/梯度回归7 passed、17 deselected。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_f001_no_residual_20261007/model_check.json 和 regression.log。

版本配置已准备，未提交或启动新训练，未改旧冻结模型/任务，没有测试权重。现有正式入口支持--variant f001；新训练封版需独立源码/输出与原审核。默认从零构造，未迁移50082权重/优化器。工作区SDPA auto和where实现继承，后续受控比较应统一后端。版本说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_f001_no_residual_20261007/README.md。

## 2026-10-07T05:08:27.070594+00:00 V5/V7比较需区分模型、数据与训练量；2750消融继续

V5与当前V7的主干深度/宽度相同，同时变化的因素包括ESMC预训练来源、词表、链/位置编码、attention/残差、目标布局、损失空间及语料组成/训练预算。实际配置和manifest对照已更新 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md 顶部，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/v5_v7_model_data_comparison_20261007.json。这不是单变量去ESMC实验，不能据下游差值归因某一项。本轮未改变架构、数据或训练配置。

原八项2750同曝光验证已通过，平均AA和整体END改善，但全目标AA CE与context指标仍非全项一致；数字唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2750章节。维持已完成的正式f011及原解码，继续22格3750完整epoch，再综合比较；两seed、有限context、混合源码与历史退出未知限制保持。

## 2026-10-07T04:34:13.262347+00:00 2500中途结果仍不支持统一最优改动

整体残基指标与context全目标表现存在分化，结束标记也非全面改善。具体数字、同曝光核对和来源限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2500章节；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2500_root_review.json。维持已完成的正式f011，继续原22格3750完整epoch后综合比较；不按中途或下游test结果修改长度、解码或选择性重跑。保留两个seed、有限context与混合源码的解释边界；没有新增模型实现或训练配置。

## 2026-10-07T03:54:11.238678+00:00 2250残基重建改善仍不足以选定统一最优改动

整体平均与全目标残基CE均下降，但结束标记准确率仍有多项回退；context组的重建与终止变化又不同于整体。具体数字和来源限制唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2250章节，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2250_root_review.json。继续原22格3750完整epoch，正式f011保留为本阶段已完成版本；不按中途或下游test结果更换架构、长度或解码。两个seed、有限context与混合源码差异继续披露，没有新增模型实现或训练配置。

## 2026-10-07T03:17:48.065803+00:00 2000验证出现context重建与终止的取舍，继续原完整消融

有context样本的残基重建全部改善，结束标记准确率却全部回退；全目标残基还呈seed42与43不同方向。具体数字和来源限制唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新2000章节，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2000_root_review.json。该中途波动尚不能确定统一最优改动。正式f011仍作为本阶段已完成版本保留，继续原22格3750完整epoch；不按中途或下游test结果更换架构、长度或解码。两个seed、有限context及混合源码差异继续披露，没有新增模型实现或训练配置。

## 2026-10-07T02:42:21.260230+00:00 1750中途验证仍不支持统一最优改动

新增八项平均重建持续改善，但整链/全局交替seed42的全目标残基与结束标记表现仍有退步；context目标与整体指标也非全部一致。具体数字和来源限制唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1750章节，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1750_root_review.json。正式f011保持为本阶段已完成版本，原22格继续3750完整epoch；不按中途验证或已完成下游test调整架构、长度或解码。两个seed、有限context及混合源码差异继续披露，没有新增模型实现或训练配置。

## 2026-10-07T02:08:15.750729+00:00 1500同曝光比较仍有重建/终止取舍，保留正式f011至完整消融

已核实9种架构、18个固定1500点的同seed/同曝光对比，数字与来源限制唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 最新1500章节，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1500_root_review.json。固定残差0.1在平均重建及context目标残基上有观察收益，但context结束标记在两个seed都受损；组合改动与整链/全局交替也无全指标一致提升。正式f011保留为当前已完成阶段版本，继续原22格3750完整epoch；不按这一中途点替换正式架构或调整下游解码。

固定头门控与后期跨链融合控制尚未完成，G的收益不能直接归因于动态门控学习。各基线配对的源码一致性、仅两个seed以及context样本有限均已披露；同源码记录也不等同位级复现，不把描述性均值当统计显著性或结合功能证据。没有新增训练配置、模型运行或测试集调参。

## 2026-10-07T00:39:13.894414+00:00 本轮三类生成已完成，当前结果不支持全面优于V5

正式两轮后CDR、轻链和T4原生评测完成，具体数字/比较限制唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md。轻链存在部分指标收益，CDR和TCR仍有明显短板；TCR原生自主终止与V5训练长度先验等不同，不能把差距直接归因于一项架构改动。保留本轮固定checkpoint和协议，不按test改长度或挑选更好采样结果。正式f011作为本阶段已完成版本保留，最终架构判断仍等待原22格完整epoch和独立验证依据；750点的平均AA/END取舍已在过程记录中说明。完整TCR证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json。

## 2026-10-07T00:20:18.580374+00:00 750步显示平均重建与终止指标取舍，保留完整epoch决策

新八项750验证已核实。相对各自500点，平均AA重建均改善，但全遮蔽AA只有半数改善，平均/全遮蔽END均回落；详细数字仅见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 本日750验证记录及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first750_root_review.json。不据早期/测试输出改正式f011或解码参数；继续原120000条/3750步和两seed终态比较。来源cohort差异保留，不能将该点当架构普遍收益或功能结合证据。

## 2026-10-06T23:50:17.965727+00:00 T4结果比较保留长度与输入布局差异

既有V5 159000与本次V7使用相同基准和主指标源码，比较成员/分母绑定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/v5_comparison_binding_v1.json。报告固定benchmark14、sparse13、历史common6、held20；不把历史common6标成V7未见，也不比较V5未提供的novelty。固定C/F与训练长度先验对比全槽位/结束生成、batch/RNG、模型/语料/预算差异须明确列出。V7全量尚未完成，原f011和协议不变；不按正在生成的test输出改配置。

## 2026-10-06T23:42:54.045644+00:00 固定500中间图补齐八项，不改变正式架构

固定500对比图更新为18/22、仍PARTIAL：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/step500_architecture.png；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/root_review.json。平均遮蔽与全遮蔽AA CE呈不同取舍，500不是最终3750，来源cohort混合、两个seed不足以给统计显著性；维持正式f011与既定完整epoch计划，不依据早期指标换架构。具体对比产物仅引用以上CSV/JSON。

## 2026-10-06T23:39:56.588650+00:00 T4全量评测执行中，固定协议不变

已按 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md §8.6 执行首次GPU输入审计并开始全量生成；不据生成中结果变更K100/32步/seed/参考隔离。评分入口只读取已完成原始候选与独立argmax，不加载模型/注释或修复候选；固定源/数据/实际exit/原始行与导出逐项一致后才评分。准备证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scorer_preparation_root_review.json；原正式f011保持。

## 2026-10-06T23:35:21.048373+00:00 固定两轮下游比较与T4评分分母约定

轻链全量结果已验收，比较结论仅见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-pairing；不据test选择新checkpoint或改架构。T4沿用既有序列评分器，所有100候选和空串均保留；只有原始数量可证明无定义的指标记录null和原因，不能用零填充或隐去目标。历史BioSeq视图不表示V7去污状态。实现/人工控制 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scoring_controls_root_review.json；协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md §8.6。

## 2026-10-06T23:24:18.465554+00:00 轻链固定协议完整预测结束并进入原评分器

完整500×8预测及实际退出已通过，当前只接续既有协议评分，不按生成样本改参数或架构。正式结果仍须等待完整评分与真实退出，不能凭全部候选非空推断质量。协议/范围唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md §4.6；原f011与22格3750计划不变。

## 2026-10-06T23:18:02.210187+00:00 T4固定候选预算和随机种子分配

全量入口保持原34目标，每目标100随机＋1独立argmax；seed固定42+target_index×101+candidate_index（100为argmax），native64、32步上限、batch1。输出保留空/未终止/非法候选，不能按合法率补采或据test调参；greedy不会进入主K100。协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md §8.6，当前未真实全量生成。

## 2026-10-06T23:12:31.503718+00:00 T4原生协议已通过实际50082 CPU输入检查

固定协议的真实双类别输入隔离检查通过，模型每步输入与生成对隐藏参考数量/内容/长度均不变；只证明输入边界和执行路径。未进行完整T4或生成质量判断，协议不因预检输出而调参。GPU和全量评测待继续；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_cpu_preflight_root_review.json。

## 2026-10-06T23:07:56.130738+00:00 两轮后T4采用原生junction生成协议

V7 A4已有明确junction scope及64槽位的无参考GenerationRequest，因此直接对CDR3β junction和终止符生成；使用实际pMHC条件及desired-binding token，alpha按原生missing/null表示。不同于V5的训练先验长度/固定C-F，也不同于V6全β后提取；本版不读取参考长度、补锚点、扩增采样预算或替换原成绩。锚点由模型产生并报告合法率。完整协议/边界唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md §8.6；当前仅tiny模型CPU检查通过，正式f011及原3750消融不改。

## 2026-10-06T22:59:41.079874+00:00 固定轻链结果的评分规则

沿用现有评分器，保留逐候选IM值和逐条链型/有效性比较，核对全量顺序与分母；IM及整体匹配率按全候选计算，历史Diversity会排除空串，需如实披露，不能声称所有指标均使用相同分母。当前只有准备验收，没有质量结论；协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md §4.6，正式架构不变。

## 2026-10-06T22:54:53.864578+00:00 两轮后轻链评测执行阶段

真实CPU/GPU输入隔离验收已通过，固定全量生成单次开始；采样/长度/分母协议不再依据test调参，协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md §4.6。仅表示入口可执行，尚无生成质量结论；正式f011和原22格消融不改。

## 2026-10-06T22:37:03.056870+00:00 V7轻链评测采用独立原生适配，保留固定已知长度对照

输入/解码模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_pairing.py 已完成新的CPU人工检查，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/synthetic_root_review.json；仍未真实评测。下一初始固定运行选择p0、500重链×8候选、reference长度、iter16/gumbel_argmax/CFG0/temperature1，对齐既有V5 159000的可见条件和主要解码设置，不做test网格。模块保留p3兼容能力，CPU p3通过不表示额外p3评测已授权调参或已产生结果。

继续使用当前50082原生权重；复用冻结legacy采样/置信度提交原语，但A4词表/输入布局与V5不同，不能声称相同随机轨迹或受控架构增益。目标slot预测protd时保留全部raw token，解析按首次protd结束，不修复候选或排除失败候选。当前阶段未调用评分/注释；此前被停的VNAR/source-H3覆盖注释分支保持停止。任务协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md。

## 2026-10-06T22:28:30.597931+00:00 两轮后的固定CDR结果已获得；继续原消融，不据test更换架构

正式f011在固定50082/两轮完成九项CDR。结果与历史V5有升有降，不支持全面提升；数字及不同数据/规模/预算/解码限制只见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr。保持这次固定结果，不使用测试集挑checkpoint或逐CDR迭代数。后续仍以原120k/3750、两个seed和固定验证对照完成12项缺格，原22格图最终统一呈现；不自动开始第三轮。light-chain pairing/T4仍是未完成的独立适配范围。

## 2026-10-06T22:17:06.607485+00:00 f010单seed完整epoch结果：保留正式f011，等待剩余对照

120k/3750完整epoch与实际退出通过，固定3750图扩为10/22。f010相对同seed同来源f000平均AA改善，但全遮蔽AA CE略退且context END准确率回退，不能推出理解与生成全面收益。第二seed及其余12格仍缺，当前不更改正式f011或固定50082/单步CDR评测设置。完整数据和限制唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md，图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1_f010/step3750_architecture.png；两轮正式训练仍继续。


## 2026-10-06T21:37:34.475996+00:00 两轮后的实验顺序保持；仅修正工程衔接

正式f011架构、context CE=0和已有正例关系监督、micro32/GA1/world8/global256保持，目标累计50082/2epoch后评固定checkpoint的9项CDR，不自动增开第三轮。当前46625、46000保存/验证已通过。生成之后继续原22格消融；新的准备入口绑定当前恢复完成证据，旧失败记录保留，未改模型/数据/解码/评分。该6项工程检查不计为模型效果实验。接续与未完成事项见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md；两轮及生成协议见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。



## 2026-10-06T21:13:43.071416+00:00 故障后的接续计划保持模型和评测节点

独立恢复只改通用协调器心跳持久化及执行衔接，不改f011/where/math模型、原50082训练日程与两轮后固定生成协议。8步恢复验收后才执行余下5000＋24步；独立包尚未启动，真实GPU恢复尚未通过。完整状态与证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_STARTUP_LOCK_FAILURE_20261006.md。本地3500对照仍不足以更换正式组合，继续3750与第二seed。

## 2026-10-06T14:15:40.813569+00:00 两轮后生成并行执行

保持既定模型与argmax单步方法，按dataset/mode分9项复用原8卡；先真实两轮训练结束，再生成评测，再准入缺失消融，不在25042就抢占资源。 绝对路径、真实进程与门禁见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

## 2026-10-06T13:54:38.042567+00:00 两轮后原生CDR生成评测

先完成固定50082的原生CDR评测，保持f011/where/math训练模型和既定单步协议；模型输入与旧fusion分开适配。CPU通过不代表实际下游性能。 绝对路径及验收状态见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md；任务协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4。

## 2026-10-06T13:38:54.423749+00:00 两轮接续执行配置

冻结模型仍f011/where-only/math，第二阶段采用显式LR重启与跨轮sampler修复；已部署等待父训练自然完成。源码和执行包、真实验收边界唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。当前实际新阶段GPU验收及生成评测未完成，禁止把待命当训练启动或完成。

## 2026-10-06T13:21:35.490096+00:00 用户新增两轮后生成评测；准确节点50082

当前冻结运行仍到25042。实际25041步/epoch，两轮为50082；需要再接25040步，纠正此前50084估算。CPU发现并修复跨轮skip loader重置epoch的问题，修复后原生恢复/样本顺序/最终模型与连续两轮完全一致，临时权重已清理。独立分布式接续与生成入口核实仍进行中。计划和证据唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md；原Goal身份保持，完成须包括新目标与原消融，未提交/重启任务或降低门禁。

## 2026-10-06T13:05:08.199910+00:00 9000验证与当前运行

2026-10-06T13:05:08.199910+00:00：正式where-DDP当前9303/25042；最新完整验证/保存9000已核对，当前段7016→12016仍在训练，最近段退出验收7016。监督PID212174/start2455180199与八worker正常，无abort。best3 7000/7008/7016＋latest9000共4份完整状态，8000权重已轮换删除、JSON证据保留。

1000步8001→9001墙钟平均0.722880秒/步（含一次验证/保存），999普通步均值0.713420秒；这是实际DDP/global256的新完整窗口，不含段启动恢复，不作FSDP吞吐声明。9000主eval_loss1.890182、8000为1.889376；质量指标有升降，继续原f011与总25042计划。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint9000_root_review.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。

本地f010/42当前1600/3750，已验收1500，下一1750；22格仍9完整epoch产物、1运行、12未启动，500图10/22、3750图9/22均PARTIAL。下一正式10000；原Goal仍active，上一回合/本回合均progress。有限观察63484已退出0，持续监督及训练继续；禁止重启旧监督、重复提交，原注释停线/CUDA暂停/数据门禁保持。

## 2026-10-06T12:52:51.652079+00:00 新代码正式运行与保留策略当前状态

2026-10-06T12:52:51.652079+00:00：新where-DDP当前8278/25042，8000步验证/完整保存已核实，段目标12016；持续监督PID212174/start2455180199和8worker正常，abort=null。best3 7000/7008/7016＋latest8000共4份完整恢复状态。300步均值0.740862秒（稳态含数据/DDP，排除验证保存；旧10.2–10.7秒），原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json。新版模型仅where等价展开，仍micro32/GA1/world8/global256和math attention。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_WHERE_DDP_CONTINUATION_20261006.md。

原7000完整checkpoint接续，旧最后打印7600后的约600个已打印未保存更新已重算；不冒称原进度无损。新段7000→7008、7008→7016通过实际八rank原生恢复/保存/退出，当前长段7016→12016仍在训练。最近段退出验收7016与最新完整checkpoint8000是不同概念。旧监督PID3977587已停，不重启。下一常规验证9000；原注释停线、CUDA暂停及数据门禁保持，不重复提交或重启容器。

本地f010/42仍运行，新增1500点已核对：平均AA改善、全遮蔽和END有回退；下一1750。22格仍9完整epoch产物、1运行、12未启动；500图10/22、3750图9/22均PARTIAL。原Goal持续active，不缩小范围或宣称全部完成。

## 2026-10-06T12:50:03.929367+00:00 本地f010/42新增1500步验证元数据已核对

读取现有原件，未重跑模型。同seed42、48000/120000条曝光，170计数字段与40个mask_ratio完全对齐；固定1024条验证，其中context37条。平均AA CE 0.931826→0.899168改善；全遮蔽AA CE 1.782721→1.784553略差，准确率40.9893%→40.4453%。平均END准确率78.3659%→77.2258%，全遮蔽END 34.0111%→33.7646%，均回退。

context全遮蔽AA CE 1.927439→1.880236改善，但END准确率38.6364%→27.2727%（17→12/44），本点不能称固定残差全面更好；已有正例关系准确率均1不证明正负binding或功能质量。保持正式f011、完整3750步和第二seed判断计划；下一本地1750。此本地任务仍用原冻结实现，与新版正式where训练的吞吐分开。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1500_comparison_20261006.json；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1500_root_review_20261006.json。22格仍9完整epoch产物、1运行、12未启动；图500点10/22、3750点9/22均PARTIAL，本次未重画、未造新退出回执或测试ckpt。

## 2026-10-06T12:46:41.042969+00:00 两节点FSDP独立文件已同步best3＋latest

新交付 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml（SHA13b09c34a0bcbffae644f1e52c6cdbea8cab053f5c8db14f2f60986c66727274），未提交。仍是where-only f011、FSDP1 FULL_SHARD、2节点×4卡、micro32/GA1/global256、25042步、动态长度上限2048；目标mem_peak仍待目标机器合成验收。改动仅完整checkpoint保留策略：主eval_loss最小3份＋最新取并集，最多4份，完整optimizer/scheduler/8rank RNG保持。原生全部rank保存回执通过后才由rank0归档JSON并删本run非保留点；禁用Trainer的latest3提前删除。新增FSDP sidecar的bytes＋sha256格式适配，活动DDP冻结helper未改。

10项临时目录检查通过，包含实际新FSDP入口对原生sidecar格式的保留/删除、错误size拒绝；去除声明的保留策略差异后，新旧训练入口AST完全一致。618源码绑定、3个缺失源码拒绝、隔离CLI help和YAML/bash检查通过。没有重复模型/GPU验收；自有测试checkpoint已自动清理。沿用此前CPU/Gloo模型验收的范围，目标2×4 GPU/NCCL仍待该任务实际启动；先执行原合成4→8保存恢复与2048长度显存验收，再执行原数据审核，未降低门禁。

源码清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/source_manifest_v1.json（SHAbe1d78cba6349b4d4ef9d212efc1a1c41578d90e5fd86136b62cb63a443cb9fb）；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/root_review.json；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_BEST3_2NODE4GPU_20261006.md。原where-FSDP文件与原始DDP冻结源保持，当前容器仍按既有where-DDP持续训练；本文件为另一机器独立训练，不直接读取DDP checkpoint。

## 2026-10-06T12:42:38.746417+00:00 新代码已完成八卡接续，300步实测平均0.740862秒/步

原容器同DDP正式训练已切换到独立where-only冻结源。7000→7008与7008→7016均通过实际8rank原生optimizer/scheduler/RNG恢复、参数/完整保存和8slot退出检查；随后7016→12016于2026-10-06T12:35:30.046083+00:00单次启动，当前实际计时到7450/25042。新监督PID212174/start2455180199已通过/proc核实，8worker心跳新鲜，abort=null。平台仍为t-20261005205851-lqfjg，没有重复submit/cancel或重建容器。原监督PID3977587已按用户授权停止，不重启。

| 计时窗口 | 更新数 | 平均秒/步 |
|---|---:|---:|
| 旧版5475→5875 | 400 | 10.199121 |
| 旧版6125→6850 | 725 | 10.695606 |
| where新版7027–7326 | 300 | 0.740862 |

新版中位数0.785829秒、P95 1.057530秒，global256对应约345.54条/秒；相对上述旧窗口观察到13.77–14.44倍提速。跳过本段最初10步；计时含数据与DDP的训练循环墙钟，不含验证/保存/初始化。旧窗口使用日志首次观察端点，有轮询误差且batch不同；不能将此作为同batch严格配对或端到端epoch耗时。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json，含300条原始计时和来源哈希；未新增模型或CUDA专项测试。

保存已实际采用主eval_loss最小3份＋最新完整恢复点取并集，当前保留7000、7008、7016（最新与best重合）。原5016、6000权重已按新规则删除，删除前的JSON/指标证据保留在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run/checkpoint_retention_evidence；每份保留完整优化器/调度器/8rank RNG。当前7016的eval_loss=1.889094435879232；旧7000=1.8842671278774004，短接续验收不证明效果提高。原旧段最后打印7600，只有7000完整保存，约600个已打印未保存更新已从7000重算，不宣称无进度回退。

接续证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7008_root_review.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment7016_acceptance.json；新监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervisor_v1/supervise_v7_where_20261006.py，实时状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervision_state.json；当前输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run。后续每段5000更新，经实际恢复/保存/退出验收后接到总25042；遇错误停止继续准入，不重复提交。当前仍为f011、micro32/GA1/world8/global256、BF16 autocast、math attention；仅替换等价chain_bias展开表达式，未启用高效SDPA或切FSDP。原数据门禁、序列注释停线、独立CUDA暂停保持；原Goal仍active，完整正式训练和22格消融未完成。

## 2026-10-06T12:28:24.471769+00:00 原容器八卡where版本已单次接续：7000→7008

原8worker内执行了8份同host/boot只读进程检查，确认旧8个slot controller与8个训练child的PID及进程组全部消失。rank6原始wait回执仍缺，未伪造旧exit/cleanup或改写旧group terminal；采用新的实际进程消失证据满足新模型启动前的互斥要求。汇总 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/OLD_PROCESSES_ABSENT.json；原件在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches_v2/segment_7008/meeting/old_process_absence_*.json。

新8票于2026-10-06T12:26:54.007405+00:00单次入队，7000→7008；8rank已注册，coordinator started/abort=null。新源613文件绑定及原数据审核保持，正在初始化并恢复原native checkpoint，尚未声称GPU恢复、保存或新速度通过。新plan /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches_v2/segment_7008/plan.json；enqueue /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches_v2/segment_7008/enqueue_receipt.json；实际输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run。旧原生7000完整恢复点保留，7600日志后的旧段用户中断，约600个已打印但未保存更新从7000重算，总日程仍25042。

新driver包含best3主eval_loss＋latest完整恢复状态，并记录逐步训练墙钟；下一步核实7008真实8rank恢复/参数一致/保存/退出及实际轮换，再继续新版本长段测稳态速度。平台task保持t-20261005205851-lqfjg，未submit/cancel或重建容器；原旧监督已按用户授权停止，不重启。独立CUDA及注释停线/数据门禁保持。


## 2026-10-06T12:21:04.640757+00:00 where同DDP接续的停止前检查已通过

8项best3+latest临时目录控制通过（9466e9 rc0）；CPU旧模型checkpoint→where模型原生恢复通过（6a4a49 rc0），恢复optimizer/scheduler/RNG、最终step4与全部optimizer step4，旧checkpoint未变，较连续旧实现最大参数差7.64e-11。首次CPU测试因独立导入模块缺少包名失败于模型构造前，改为正确包内名称后通过，两个日志保留。自有临时checkpoint均已清理。

613个候选源码绑定、原只读父数据审核、允许的模型单表达式变化和8种不兼容元数据拒绝检查通过；原checkpoint7000全部17个native文件重新核验，仍可完整恢复。预停记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/pre_stop_review.json；新冻结源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/source_snapshot_v1；原冻结源不改。新8步接续plan已准备未入队，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches/segment_7008/plan.json。

接下来按用户明确授权停止旧训练子进程并在同一容器由7000→7008接续；平台容器保留。当前没有立即保存接口，7000之后未保存更新将从该完整恢复点重算，实际停止日志步数会单独记录。全部八rank退出清理证据齐全才允许新8票入队。此处CPU通过不等于新八卡恢复或吞吐通过。保存策略与新旧来源迁移保持显式记录；原安全停线和审核门禁不变。


## 2026-10-06T12:08:05.749567+00:00 用户授权原容器内where接续及best3＋latest保存策略

用户明确要求从现有checkpoint接续新代码并监察真实训练速度；允许对此次切换受控停止旧训练，但不重建容器或重复提交资源。准备独立DDP接续版本，保持world8/micro32/GA1/global256、数据/目标/精度/25042日程；保留原冻结源码和旧provenance，以显式代码迁移记录连接旧checkpoint与新版本。旧运行尚未停止，最新完整恢复点仍7000，实际切换时重新核实。旧进程没有立即保存请求接口，恢复点之后未保存更新不能声称已保留。

保存策略按已有主eval_loss最小的3份＋最新完整恢复点取并集，重合不复制；全部保留完整优化器/调度器/各rank RNG，先完成八rank保存验收再轮换，只处理明确所属目录，删除前归档JSON证据。现有代码仍是latest3；新增模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/native_best3_latest.py 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_where_transition.py 准备中，尚未测试或部署。更早已被旧策略删除的权重不可追溯恢复。

两节点FSDP文件当前仍是上一版本，best3接入需独立更新后明确交付。下一项是接续/轮换CPU检查和完整启动包，再使用已有容器接续；不恢复注释分支、不修改审核证书或绕过门禁，独立CUDA专项保持暂停。


## 2026-10-06T12:01:37.551Z V7 高效 SDPA 已接入工作区

RefinedModel 与 FactorialModel/f011 默认 auto：CUDA efficient SDPA + math 回退，CPU math；IMMUNE_SDPA_BACKEND=math 可切回，efficient 可强制 CUDA 验收。策略置于 block 内，覆盖 checkpoint 重计算并恢复外部 flags。现有 A100 / PyTorch 2.8.0 / CUDA 12.9 已验证，无需额外 flash-attn/xformers。最终98项测试通过，包含f011 FP32/BF16 GPU前反向和梯度；BF16与math存在舍入差异，r20链偏置相对差约3.27%，不承诺逐位/收敛等价。

工作区实现已更新，运行中的冻结任务及旧where-only FSDP封版包未改，实际多卡发布需重新绑定快照并验收。未提交/重启/停止训练；仅保留测试日志，无测试权重。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_EFFICIENT_SDPA_20261006.md。

## 2026-10-06T11:57:12.223404+00:00 链偏置修复版2×4 FSDP已完成CPU集成验收

已封版 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_fsdp_2node4gpu_20261006.yml，未提交。f011架构与目标保持，仅将chain_bias重复索引改为where广播选择；原attention后端不变。已有真实V7单卡均值6.6288→0.5450秒定位了主要模型计算瓶颈，不代表原八卡已经提速；原正式/消融冻结实现未替换。

新接入CPU合成loss/全部梯度、四mask、原生Trainer保存恢复与模型重载通过；2-rank CPU/Gloo FSDP实际分片、重计算、梯度更新、评估及完整模型/优化器往返通过。自有临时checkpoint已删除。CLI源码绑定与拓扑拒绝、YAML/bash检查通过；两节点GPU/NCCL及目标mem_peak仍待实际任务，未重复GPU验收。

2节点×4卡，micro32/GA1/global256，25042步；默认c20250601/每节点ml.pni2.14xlarge，合成4→8步原生恢复与2048长度验收通过后再走原数据门禁。独立训练输出，不接续/覆盖当前DDP。原FSDP YAML/source_manifest_v2亦保持原样。

交付说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_2NODE4GPU_20261006.md；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/root_review.json；新manifest SHA c4105f8708c235b0e8c211f9a891c46d74049f815dfdbe57275b299203db3aff。停止注释分支、独立CUDA暂停、原审核及恢复约束保持。此次变更是工程性能修复，不能据此宣称下游或生成效果提升。



## 2026-10-06T11:46:41.677300+00:00 独立where-only FSDP候选接入中

新增入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp_where.py 与适配器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_fsdp_where_trainer.py，仅替换已验证的bias展开表达式，保留原attention后端及参数/监督语义，复用原训练器和数据/恢复门禁。原冻结源、原八卡任务、现有FSDP入口及YAML保持。CPU新集成验证准备运行；新候选YAML尚未封版或提交。测试脚本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_fsdp_where.py，使用合成张量，临时checkpoint自动清理。

## 2026-10-06T11:41:39.650760+00:00 补齐已有GPU逐参数梯度证据：仅where＋math路径

已读取并核对既有真实V7单卡记录，未重跑GPU。75,601,176参数f011、一个B32/S516的原生输入，原版与where版复制相同权重并恢复相同CPU/CUDA RNG，固定math attention与BF16 autocast；loss均3.8769800662994385，完全相等。78个有梯度参数张量逐元素检查通过，rtol=1e-3/atol=5e-6；只有chain_bias有非零差异，最大绝对误差2.7315691113471985e-6、相对L2误差0.00017428879800718278。其余77个有梯度张量记录为完全一致。

实际执行脚本SHA与控制器记录一致，where子进程returncode0；没有保存梯度或测试权重。本证据补充上一条“标量不足以证明完整梯度”的限制，覆盖的是一个初始化状态下的原生batch及where＋math路径，不能扩展为where＋efficient、多长度/后续优化器状态或八卡FSDP恢复验收。当前正式运行与交付FSDP绑定源码未在此轮修改；后续接入仍应保持独立版本及原数据/恢复门禁。

根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/native_gradient_equivalence_root_review_20261006.json；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1/where/gpu_gradient_equivalence.json（SHAba96064c598c8db439cd97e7052bb04faa65c5d4f442c2ded91bda9f4124c765）。测速均值和原单卡范围保持，不把该梯度检查当作新吞吐测试。



## 2026-10-06T11:35:49+00:00 chain bias展开修复：真实V7 GPU验证通过

GPU根因为重复高级索引的IndexBackward：真实B32×S867整步9.5329秒，单个索引反向kernel8.8094秒。工作区research_refined/model.py仅将base_bias构造改为torch.where广播选择，保留24参数、梯度、mask、目标和架构。8个真实batch单卡更新均值6.6288→0.5450秒；仍用math后端。75.6M GPU首步loss一致，78组梯度容差检查和64项既有回归均通过。浮点归约顺序改变，不保证历史训练轨迹bitwise一致。

efficient后端是独立优化：固定S516单换后端2.1404→2.0150秒；与where组合在真实批次为0.3103秒，此次工作区只落地where修复。原正式/消融冻结源与恢复状态未改，八卡吞吐待新冻结版本验证。占卡开/关对照完成，测试后两个原进程已恢复。

报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SPEED_ROOT_CAUSE_20261006.md；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/verified_summary.json。

## 2026-10-06T11:33:52.924409+00:00 f010完成1250步验证：重建改善，END仍有分化

同seed42、同来源；40000/120000条曝光，input15391562、eligible6157115、selected3096993，LR7.747811222909726e-5。170计数字段与40个mask_ratio一致，现有metadata validator核实新点的四档分子/分母、均值及来源/context聚合。固定1024条验证，context37条。

| 指标 | f000基线 | f010固定残差0.1 |
|---|---:|---:|
| 平均AA CE | 0.961122 | 0.918567 |
| 平均AA准确率(%) | 70.930929 | 72.489807 |
| 全遮蔽AA CE | 1.787297 | 1.776067 |
| 全遮蔽AA准确率(%) | 41.235131 | 41.497486 |
| 平均END准确率(%) | 78.917998 | 79.654119 |
| 全遮蔽END CE | 1.485902 | 1.467244 |
| 全遮蔽END准确率(%) | 36.229205 | 34.627234 |

平均及全遮蔽AA在本点均改善；全遮蔽END CE略降，但准确率36.2292%→34.6272%，并非全面受益。context全遮蔽AA CE1.980387→1.987997略差、准确率42.9337%→43.6997%；context全遮蔽END准确率47.7273%→52.2727%（21→23/44）。已有正例关系准确率均1，不能推出正负binding或功能质量。

f010是R24固定残差0.1单因素；1000与1250点的END表现波动。维持正式f011及3750完整epoch/第二seed计划，下一本地1500。完整22格仍9格完整epoch产物（8格成功退出、f000/42退出码未知）、1运行、12未启动；500固定图10/22、3750图9/22，均PARTIAL，未重画。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1250_comparison_20261006.json（SHAd0fa3949be9ebb665b7291b9db1a1524bd12d2735cbcbcaf4d7837f0a30b22fc）；根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1250_root_review_20261006.json。本监督只解析新指标，未重复运行模型。

## 2026-10-06T11:33:52.924409+00:00 真实V7单卡测速已有原件：主要热点为chain bias索引反向

复核共享工作区既有独立控制器的三个case：均returncode0、passed，原始model与where/where+efficient共用75,601,176参数、B32、BF16 autocast、whole-layer checkpoint、seed42和相同输入来源/选择/顺序。使用既有1024条前缀探针中的8个固定batch，长度516–867；8次total-only计时均值已从原始记录重算。

| 实现 | 单卡整步均值(秒) | 分阶段反向均值(秒) |
|---|---:|---:|
| 原始索引展开 | 6.628798 | 6.451850 |
| where展开 | 0.545025 | 0.384400 |
| where＋efficient SDPA | 0.310286 | 0.242226 |

原始实现实际使用math SDPA；被profile的batch有一次索引梯度kernel约8.81秒，where后该热点消失。合成B32/S516对照中，原始2.140秒、仅efficient2.015秒、仅where0.328秒，支持优先处理bias索引反向。不能仅用换FSDP解释慢化。

这是单卡完整模型重放，不含CPU组batch、稳态H2D、DDP、验证与保存；8个固定batch不是全量分布，不能直接宣称八卡正式加速12或21倍。19次GPU scalar loss最大差where约0.000193、where+efficient约0.001207；标量及梯度范数不能证明完整GPU梯度向量相等。后续接入需独立版本的数值、分布式与恢复验收。原正式冻结模型、当前任务和已交付FSDP绑定源码均未由本监督改动。

原控制器11:27:21结束并写RESUMED.json，本监督实际/proc核实PID80642/start2452750909恢复为R。暂停窗口不是训练失败，也不用于消融吞吐对比。本监督只读取既有结果，未发送信号或重复GPU测试。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/native_speed_evidence_root_review_20261006.json（SHAb38e5aca907c546f2084a2e489e2fd25ca3e15132855edaf059d7a76d866eae1）；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/native_v7_attempt1。旧“尚无GPU实测”由本条更新，独立CUDA专项和既有安全停线仍保持。


## 2026-10-06T11:06:53+00:00 本地完整模型速度验证

针对用户新增的实测要求，以冻结f011进行单卡对照：原始实现、强制efficient SDPA、将可训练chain_bias索引展开替换为数学等价torch.where归约路径、二者组合。保留全部参数/链关系/mask/损失与whole-layer重计算，只在诊断子进程替换执行方法，不修改冻结源码。先验证展开前向与梯度，再测完整模型并收集实际kernel。where候选针对24参数的大量重复索引反向，是否主因以GPU证据为准。

使用有界SIGSTOP/SIGCONT临时让出本地f010算力，独立watchdog保证异常时恢复，原权重/优化器留在内存；八卡正式任务继续。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/profile_local.py。属于用户此次要求的整模型性能验证，不重启已停的独立CUDA专项，无checkpoint。

## 2026-10-06T10:41:11.204701+00:00 正式7000步验证与保存已验收

平均AA CE0.746101→0.736503、全遮蔽AA CE1.749312→1.728628改善；全遮蔽END准确率57.4277%→56.3310%、首END准确率16.7498%→16.3011%回退，长度MAE3.220339→3.438404变差（分母2006→2005）。误报END减少，但不能据总体loss宣布全面提升。17原生文件/6JSON归档/8rank一致性、112固定计数、70核心聚合、26结束指标核对通过；不重复载入tensor或运行模型。详细表 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint7000_root_observation.json。

新FSDP 2×4 YAML及CPU验收保持上一轮状态，未提交，未重复验收或改写已交付源码。本地f010/42最近验证1000保持，下一1250；固定500/3750图与22格计划保持。

速度补充来源与端点已复核：普通训练区间仍10.1991/10.6956秒/更新，慢化不只发生在验证/保存。冻结代码禁用memory-efficient、可训练bias限制其他融合后端，源码推断math SDPA为优先排查对象，尚无实际kernel trace。根核对 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_metadata_root_review_20261006.json；原安全停线和独立CUDA专项暂停保持。首次文档发布工具在JavaScript解析阶段失败，未执行或写文件；修正后发布，不重跑7000验收。

## 2026-10-06T10:29:39.027411+00:00 两节点四卡FSDP文件完成，CPU通过，GPU验收待新任务

已准备 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_fsdp_2node4gpu_20261006.yml；独立入口沿用冻结f011，FSDP1 FULL_SHARD，2×4卡、micro32/GA1/global256、25042步。默认c20250601/每节点ml.pni2.14xlarge，目标GPU mem_peak待profile。YAML按合成保存/原生恢复与全长显存验收→自有合成ckpt清理→原数据审核→正式训练执行；没有提交或启动新GPU工作。CPU数值/尾批/保存恢复和真实2-rank CPU FSDP已通过，不能称两节点GPU通过。原DDP、消融与冻结模型不变。

详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_2NODE4GPU_20261006.md；最终根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/root_review_v2.json。新增整模型分阶段计时/有界trace帮助定位慢化，当前主因仍未实测确认，独立CUDA专项继续暂停。正式恢复状态保留，自有CPU测试临时权重已清理。

## 2026-10-06T10:27:22.350189+00:00 f010完成1000步验证：平均AA改善，END指标回退

同seed42、同本地来源，与f000同1000步比较；32000/120000条曝光，170个计数字段及40个mask_ratio一致，固定验证1024条（context37条）。平均AA CE1.008167→0.969030、准确率69.2347%→70.9678%；全遮蔽AA CE1.798585→1.835593变差、准确率40.3622%→40.4934%基本持平。平均END CE0.778868→0.924024、准确率70.9231%→69.9270%；全遮蔽END CE2.082055→2.592752、准确率18.7307%→12.1380%，本点明显回退。context全遮蔽END准确率50%→43.1818%（22→19/44）。已有正例关系准确率均为1，不能推断正负判别或功能质量。

f010是R24固定残差0.1单因素，不是R26；结果进一步表明不能凭平均重建收益宣布全面提升。原正式组合f011与3750完整epoch计划保持；本地下一验证1250。固定500/3750图不替换为1000点，完整22格仍9完整epoch产物（其中1格退出码未知）、1运行、12未启动。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1000_comparison_20261006.json；根核实 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation1000_root_review_20261006.json。本轮只读取新指标、未重跑模型。发布时正式日志6975/25042，读到的完整checkpoint目录6000；最近根验收仍6000，新checkpoint若出现需单独验收。本地训练1025/3750、验证1000。

## 2026-10-06T10:17:16.991407+00:00 V7速度调试补充与工程验证次序

当前冻结模型继承关闭memory-efficient SDPA的设置；provenance记录torch2.8.0，可训练非空chain bias在该版本选择器中又阻止Flash/cuDNN，静态分析指向math路径。实际kernel及各项耗时仍未测，不能把此单项认定为全部4倍差距。local/global FP32 bias复用没有覆盖逐层BF16转换/无穷扫描，whole-layer重计算会重复；显式RoPE和每microstep日志还有GPU→host同步。

工程验证顺序：整模型分项/实际后端 → 保留bias梯度的efficient对照 → forward内bias转换复用、位置检查移出层循环、日志标量延迟取值；每项先验证数学/梯度，再计时。micro/GA或长度分组会影响顺序、RNG和token权重，独立比较；目标容量及END监督保持原语义。本次仅整理源码/已有元数据证据，未实施模型或训练参数更改。

详细依据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_windows_and_geometry.json。本次未运行GPU诊断、创建测试ckpt或恢复已暂停专项；另行进行的FSDP工程接入仍按其独立文档记录，不计为本次速度验收。

## 2026-10-06T10:16:38.158427+00:00 用户新增两节点四卡FSDP任务：工程接入中

保持现有正式f011及原DDP运行；新增独立FSDP1入口与包装/评估/完整checkpoint恢复、计时适配。CPU合成验证已启动，尚无GPU或两节点验收；用户自行提交，不重复申请资源。状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_2NODE4GPU_20261006.md。速度主因仍待受控测量，新入口加入分阶段计时和整模型trace，不恢复独立CUDA测试或已停序列注释分支。原数据审核与来源门禁保持。

## 2026-10-06T10:01:03.673427+00:00 更正“统一padding到1024/2048”的说法

先前把最大允许长度与实际padding长度混称，造成误导。V5实际运行上限1024，其记录commit的collator取本batch最长记录再补PAD；当前上限2048，实际仍取本batch最长记录。2048不是所有输入的张量长度，也不能据此解释训练慢。当前长记录需要较高上限，但尚无证据证明2048这个具体数值是唯一或最优选择。

复用既有1024条有限CPU诊断样本，不读取原始训练序列：4条渲染后记录超过1024，长度1090/1109/1148/1206，涉及32批中的3批；17批的实际长度为516。渲染长度包含context、多链、grammar和目标容量，不能说成单链有1206残基；该样本不代表全量比例。

只比较其余29个可放入1024的原批次，当前动态padding共568224个张量位置；强制统一1024会变成950272个，即1.6724倍，稠密attention矩阵元素数量为2.6607倍。这仅为几何量计算，不能当实测时间或速度倍数。详见 [padding1024_existing_geometry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/padding1024_existing_geometry.json)。

1024可作为后续短输入上限候选，但需明确长记录处理；优先评估动态padding与按长度分组，而不是把所有短批次也扩到1024。长记录可保留独立批次及较高上限；若改变目标容量/布局，需要另行验证，不能直接截断或丢弃记录。当前冻结运行、数据与证书均未改变。本轮仅更正文档与整理已有证据，无新模型测试；不恢复任何已停止分支。

## 2026-10-06T09:41:42.093903+00:00 V5/current源码对比完成；澄清2048上限与checkpoint策略

23份关键源码对照及解释已写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md，当前源码逐份匹配冻结manifest。V5 FSDP由保存参数确认；历史git版本可复原但未提交补丁缺失，报告明确此限制。新确认层内显式RoPE读取position_ids.max()及component float同步、稠密可训练bias反向、固定target容量与batch padding均值得受控计时；尚未实测瓶颈或切换并行方式。

2048是整条记录上限，并非统一padding。既有1024条小样本/micro32的32批实际S=516–1206，17批为516；固定target容量256/64仍在。当前每1000步及分段边界保存，保留最近3个完整恢复点（5000/5016/6000，合计约2.724GB），不同于V5最优3+最新/slim。新增证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/root_review.json，没有重跑模型、读原始序列、修改训练配置或清理正式恢复状态。文档f010/R24标签已纠正。

当前正式6725/25042、本地850/3750，原进程与8rank心跳正常，完整验证仍6000/750；有限观察41757已实际rc0结束，原训练继续。完整Goal仍active，本轮progress。既有安全停线、独立CUDA暂停、远端403不重试和数据/来源门禁保持。


## 2026-10-06T09:32:57.526652+00:00 纠正消融解释中的变体误标

对照正式与本地冻结factorial.py后确认：f010=R24固定残差缩放0.1，f001=R26局部/全局radius32，f011=两者组合。此前500/750点解释及下一实验说明中把f010写成R26是文档错误，现已纠正4份文档。实际运行variant、指标原件、图上的f编号与训练配置不变；f010相对f000的收益不能归因于R26。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_review/factorial_label_correction.json。


## 2026-10-06T09:29:33.300520+00:00 V5实际速度对照已核实，继续完整代码差异审阅

V5两段原始长运行平均2.734181/2.731451秒/更新；当前保存区间10.926703/11.170567，耗时约4.00–4.09倍。两边global256，V5采用FSDP/micro4/GA8/最大1024/可训练ESMC，当前DDP/micro32/GA1/最大2048/noESMC。配置不同、区间均含验证/保存，不能直接归因DDP或当同输入GPU计时。

详细对照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LLADA270_SPEED_COMPARISON_20261006.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_review/review.json（SHAcf93472fde876ecb550fb8dd111038d8054d52eec69a97e19f1580b9d4ced0b6）。用户继续要求完整对比改动代码，正在核实V5实际git版本和当前冻结入口，覆盖前向、注意力、loss、loader、精度、并行和保存恢复；尚未完成此代码审阅。没有新模型测试、提交、停止、重启或正式配置修改，现有门禁保持。首次记录工具在JavaScript解析阶段失败，未执行命令或写文件；修正引号后生成本记录。


## 2026-10-06T09:17:22.685086+00:00 f010完成750步验证：平均AA改善，END优势缩小且context回退

| 指标（750步，同seed42/本地来源） | f000基线 | f010 | 差值 |
|---|---:|---:|---:|
| 平均AA CE | 1.066193 | 1.005254 | -0.060938 |
| 平均AA准确率 | 67.6066% | 70.1068% | +2.5002个百分点 |
| 全遮蔽AA CE | 1.777032 | 1.801922 | +0.024889 |
| 全遮蔽AA准确率 | 42.0923% | 42.1760% | +0.0838个百分点 |
| 平均END CE | 0.724573 | 0.724286 | -0.000287 |
| 平均END准确率 | 70.2284% | 71.7161% | +1.4877个百分点 |
| 全遮蔽END CE | 1.622146 | 1.778963 | +0.156817 |
| 全遮蔽END准确率 | 28.4658% | 29.1436% | +0.6778个百分点 |

f010/R24在750步保持平均/部分遮蔽AA收益，但全遮蔽AA和END CE较基线更高，context END准确率回退；两组自身500→750的END准确率均下降。暂不替换正式f011，也不提前终止完整3750与第二seed。详细表、来源差异和小样本限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md。固定500/3750图不变，无训练配置或模型改动。

## 2026-10-06T08:12:10.109919+00:00 f010完成500步固定点：平均重建与END改善，全遮蔽AA回退

END的改善也不是每个遮蔽档位都成立：mask25的END CE0.155862→0.168819上升，准确率97.3747%→97.1360%下降；平均和全遮蔽END改善不能外推到全部档位。四档原始差值已记录在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation500_root_review_20261006.json 的mask_deltas。

| 指标（500步，同seed42/本地来源） | f000基线 | f010 | 差值 |
|---|---:|---:|---:|
| 平均AA CE | 1.185176 | 1.106683 | -0.078493 |
| 平均AA准确率 | 63.2778% | 66.2457% | +2.9680个百分点 |
| 全遮蔽AA CE | 1.793335 | 1.807474 | +0.014138 |
| 全遮蔽AA准确率 | 40.2420% | 39.7036% | -0.5384个百分点 |
| 平均END CE | 0.567858 | 0.520388 | -0.047470 |
| 平均END准确率 | 80.6511% | 85.2052% | +4.5541个百分点 |
| 全遮蔽END CE | 1.301628 | 1.274908 | -0.026719 |
| 全遮蔽END准确率 | 48.0591% | 58.7800% | +10.7209个百分点 |

f010/R24在500步仍主要改善部分遮蔽与END，不能据此替换正式f011。完整epoch和第二seed继续；不更改模型、训练预算或数据门禁。含context组损失/准确率分化及小分组限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md。新500图10/22，完整3750图9/22，均PARTIAL。

## 2026-10-06T07:40:52.614659+00:00 正式6000步解释与当前配置

正式训练仍为f011、DDP+NCCL、每卡32/累积1/全局256；当前入口明确禁用FSDP/DeepSpeed。历史纯LLaDA270M使用FSDP是已核实的配置差异，但尚无同条件GPU计时证明它是慢化原因，不在运行中切换并行方式。

6000较5000平均AA CE略降（0.748578→0.746101），全遮蔽AA CE上升（1.737691→1.749312），长度MAE变差（2.760219→3.220339）；END的token汇总与来源等权方向不一致。保持初步f011及既定消融预算，等待后续固定验证点，不能用平均重建改善替代完整生成/理解评价。详表及原件见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。本轮未改模型、数据、训练入口或调度；所有既有安全停线和门禁保持。

## 2026-10-06T07:04:56.525907+00:00 f010首个250步验证：部分遮蔽与END改善，全遮蔽AA持平

2026-10-06T07:04:56.525907+00:00：f010 seed42已完成250步验证，8000/120000条训练曝光，固定1024条验证（context37/无context987）。与本地f000/42同250步比较，来源记录仅variant不同，数据、初始化摘要、训练配方和源码记录一致；曝光/token/学习率、170个计数字段与40个mask_ratio逐项相同。四档平均AA CE1.538930→1.425135、准确率51.5941%→55.9645%；全遮蔽AA CE1.799719→1.799752、准确率41.2489%→41.0792%，没有同步改善。全遮蔽END准确率23.5983%→29.2052%。这是同seed早期对照，完整epoch尚未完成。

含context37条的全遮蔽AA CE2.017438→1.973137、AA准确率44.7338%→45.5381%、END准确率31.8182%→38.6364%；分来源并非普遍受益，tcr_papers平均/全遮蔽AA CE均回退。tcr_native仅2条/4个END，trait仅1条/1个END，不放大其百分比波动。已有正例关系准确率两组均1，不代表正负binding或生物功能。保持正式f011初步版本及f010的3750完整epoch计划，等待500/3750固定点和第二seed，不据单点换架构。

实际对照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation250_comparison_20261006.json（SHA1e2626c6567f05abf59d4f8914f847a2e7b440635233fe9c5b8fad0fcfd1a34b），根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_validation250_root_review_20261006.json（SHA8b7544a348083cc11f5fdbd47859dd7f6887f8b7643c670719de95d9fb1eceba）；原读取工具1ad374实际rc0、1.308秒，没有模型重跑。15分钟观察86292与随后等待验证的50137均实际rc0结束，当前无运行中的有限观察会话；原训练和监督仍在运行。正式最新5825/25042步、loss=0.4655，最近完整验证/保存5016；本地最新训练275步，最近完整验证250。原PID/start/argv与8rank新鲜心跳再次核实，无abort或训练退出码。

完整22格仍9格完整epoch产物（8格实际成功退出已核实、f000/42退出码未知）、1运行、12未启动。固定3750图仍9/22、PARTIAL，本轮不重画500/3750图。没有测试权重/tmp、额外模型运行、新平台任务或重启；正式恢复状态保留。原序列注释停线、独立CUDA暂停、远端403不重试及数据/来源门禁保持。完整Goal仍active；下一本地验证500、正式验证6000。

| 指标 | f000/42 @250 | f010/42 @250 |
|---|---:|---:|
| 四档平均AA CE | 1.538930 | 1.425135 |
| 四档平均AA准确率 | 51.5941% | 55.9645% |
| 全遮蔽AA CE | 1.799719 | 1.799752 |
| 全遮蔽AA准确率 | 41.2489% | 41.0792% |
| 四档平均END CE | 1.491647 | 1.028035 |
| 四档平均END准确率 | 46.5320% | 64.7072% |
| 全遮蔽END CE | 2.385153 | 2.119363 |
| 全遮蔽END准确率 | 23.5983% | 29.2052% |

| 遮蔽比例 | f000 AA CE | f010 AA CE | AA准确率差（百分点） |
|---|---:|---:|---:|
| mask25 | 1.343659 | 1.173423 | +6.6122 |
| mask50 | 1.431796 | 1.267541 | +6.4244 |
| mask75 | 1.580546 | 1.459824 | +4.6149 |
| mask100 | 1.799719 | 1.799752 | -0.1696 |


## 2026-10-06T05:43:41.648750+00:00 本地f000完整epoch验收与9/22固定末点对照

本地f000 seed42已完成3750/3750步、120000条唯一记录、epoch1/position0。最终输入46043658、eligible18432158、selected AA/END9205174；末次更新使用LR1.8520555061796174e-11，峰值已分配显存18.095971GiB。COMPLETE绑定summary SHA b834155d907317c8fc0b61573fbd335ece2033eda980e6a52067aad1f882d86c与原provenance df6c14a09b0f8cd9074013710deb157ffe97530f31c03cefcc947e500e9060a8，原PID已消失、无FAILURE或自有checkpoint/tmp。实际训练退出码未留存，不能写rc0。读取器在/proc消失时停止的异常与训练失败分开记录。

| 本地指标 | 3500步 | 3750步完整末点 |
|---|---:|---:|
| 四档平均AA CE | 0.826046 | 0.824812 |
| 四档平均AA准确率 | 74.9333% | 74.9618% |
| 全遮蔽AA CE | 1.696441 | 1.693748 |
| 全遮蔽AA准确率 | 43.5153% | 43.5146% |
| 四档平均END准确率 | 81.8718% | 79.5137% |
| 全遮蔽END CE | 1.505086 | 1.595907 |
| 全遮蔽END准确率 | 42.9452% | 33.2717% |

含context的37条验证记录平均AA CE1.219970→1.220773、全遮蔽AA CE1.853081→1.857462、全遮蔽END准确率50%→52.2727%；不含context的987条全遮蔽END准确率42.7486%→32.7422%。已有正例关系准确率1.0不等于正负binding判别或功能能力。新末点170计数/40mask_ratio和1024验证记录保持，只新增3750指标核对；工具5bc4ff实际rc0，无模型重跑。末点原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3750_trend_20261006.json（SHA0b0254d9141c850a387bfa4226ef4346ae9d0bbf2cafd9d26947f8eca9554288）。

| 条件 | seed | 四档平均AA CE | 全遮蔽AA CE | 全遮蔽END准确率 |
|---|---:|---:|---:|---:|
| f000 | 42 | 0.824812 | 1.693748 | 33.2717% |
| f000 | 43 | 0.821038 | 1.697685 | 42.1442% |
| f001 | 42 | 0.808769 | 1.694904 | 27.2335% |
| f001 | 43 | 0.808351 | 1.701822 | 33.3949% |
| f011 | 42 | 0.802201 | 1.693988 | 30.5607% |
| f011 | 43 | 0.803388 | 1.701324 | 33.8262% |
| f100 | 42 | 0.833389 | 1.697030 | 32.6556% |
| f100 | 43 | 0.830072 | 1.696440 | 31.3001% |
| f101 | 42 | 0.808952 | 1.695659 | 33.4566% |

f011相对f000的平均AA CE在seed42/43分别低0.022612/0.017650，平均AA准确率高0.6812/0.3860个百分点；但全遮蔽AA CE反而高0.000240/0.003639，全遮蔽END准确率低2.7110/8.3179个百分点。seed42比较跨local_metrics_only_v1与legacy_large_v1执行来源，seed43为同一旧来源；仅两seed、不做显著性或孤立因果结论。维持正式f011初步版本，优先补f010两个seed拆解R24固定残差缩放的独立作用，继续分别观察部分遮蔽重建和全遮蔽指标；不能宣布理解与生成两端都提高。

新图 [3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/step3750_architecture.png) 覆盖9/22，仍PARTIAL；9格有完整产物，8格真实成功退出已核实，当前本地1格退出码未知，13格待跑。原选择/来源/训练预算不变，未以3500较好END点替代3750。真实渲染45156d退出0、warnings空，CSV44条设计末点行/7200曲线行/49配对行；PNG已人工查看，双mask面板、missing标签、来源/seed图例可见。本地退出码未知点明确淡化。SVG/PDF/CSV/JSON及根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1/root_review.json（SHAdfb0191a887268f86297ed0e58251efb97838bfd7a122563d312b450f2b3b613）；旧图和必要证据保留。

完整验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/completion3750_root_review_20261006.json（SHA1c72315c75e9d82434c283a5b12f178381384e999b1c8a61b19b1ac62688a46c）。GPU仍有未知归属PID66091、1680MiB、利用率61%观测，不能用此次共享环境当受控速度对照，profile仍UNRUN。既定本地质量消融允许沿原共享锁/资源合同准备f010/42，不新增全卡idle条件；真实门禁通过前不启动，当前接续包准备中。原八卡继续5400/25042，最近验证/保存5016；原Goal完整范围、安全停线和审核门禁保持。

## 2026-10-06T04:39:59.360843+00:00 正式5016段完成并接续10016；段尾评估单独记录

16→5016的5000更新已完成，8个实际退出/进程清理及24项optimizer/scheduler/各rank RNG恢复摘要通过；原监督在同一8卡pool接续5016→10016，新段已有5025日志，完整目标仍25042。新段完整恢复回执尚待结束后核实，不声称与不间断轨迹等价。根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json（SHA7dcc51cf7133f82146e92b6c99b50413bcb2bb4548eb98d6ef3d70b7b84fc51c）。

5016是冻结callback强制生成的段尾评估，不只是保存；较5000只增加16步。平均AA CE0.748578→0.754328、全遮蔽AA准确率42.8318%→41.3847%回落；真实END准确率55.6830%→62.1635%上升，首END准确率16.3011%→18.7438%，但AA误报END1945→2258、token END精确率下降。分来源差异和完整表见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；新点 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5016_root_observation.json（SHAf13fe5643e7fc844b593a8a85eec9ef80758acd2659c1131889f77a923c4e43f）。不挑较好中间点，也不据单点改变当前架构。

5条外层清理扫描FileNotFoundError与原生进程清理分开记录；8slot无测试权重，正式状态完整，不热改冻结pool。状态读取修正了“5016无新评估”的假设，实际训练未受影响。本地3500已验证，继续3750；下一常规正式6000，22格及全部原门禁保持。


## 2026-10-06T04:31:36.542828+00:00 正式5000步验证与保存完成：重建CE和真实END准确率改善，首END与误报仍分化

正式checkpoint-5000已核实：17原生文件/6JSON归档齐全，8rank参数一致且已更新；固定1330条验证/112计数字段保持，70聚合与18结束位置核对通过。平均AA CE0.759145→0.748578、全遮蔽AA CE1.748737→1.737691；但全遮蔽AA准确率42.9299%→42.8318%略降。真实END位置准确率48.7537%→55.6830%上升，首END准确率16.8495%→16.3011%下降，AA误报END1561→1945，长度MAE4.073815→2.760219但分母2005→2006。逐来源仍有退步，不能仅凭总体loss宣布架构或生成能力通过。

详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_root_observation.json（SHA4e51f7c8d2006dd5221d548aa6cdd04ea1ab2ca36b1245de6ab846a5b0e99b64）、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_endpoint_diagnostics.json（SHAb94916684d8607e474bf8a98b9155cbdcc3dc2e3476c1cad90f0a3abc4dd08e2）。原八卡继续25042总目标，下一常规评估6000；5016原生接续由原监督器处理并单独核实。本地3500已验证，继续3750；22格仍8完成/1运行/13未启动，固定末点图和全部门禁保持，无新模型测试、提交或重启。


## 2026-10-06T04:27:38.275704+00:00 本地3500步验证：整体重建和结束标记改善，context重建CE略回升

本地f000 seed42完成3500步验证、112000/120000条曝光（93.33%），继续3750完整epoch。平均AA CE0.831506→0.826046，全遮蔽AA CE1.711951→1.696441、准确率42.3560%→43.5153%；整体全遮蔽terminal准确率25.3235%→42.9452%。含context37条的平均/全遮蔽AA CE略升，但全遮蔽terminal准确率45.4545%→50.0000%、CE下降；组别与指标差异均记录，不据单点或已有正例关系准确率推断功能收益。

新点170计数/40mask_ratio、1024验证记录与provenance保持，summary匹配，无终态或测试权重。详细数值见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3500_trend_20261006.json（SHA83579fc9124cc40dffa8fd769bd18b12c649d913c828ead6db799596350ae5e1）。正式八卡f011继续25042目标，评估独立记录。完整22格仍8完成/1运行/13未启动，固定500/3750图不重画；下一本地3750完整epoch，既有训练和门禁不变。


## 2026-10-06T03:22:05.045022+00:00 本地3250步验证：AA重建回升，整体结束标记继续回落

本地f000 seed42完成3250步验证、104000/120000条曝光（86.67%），继续3750完整epoch。平均AA CE0.849367→0.831506，全遮蔽AA CE1.767363→1.711951、准确率38.5985%→42.3560%；整体全遮蔽terminal准确率29.2052%→25.3235%。含context37条的全遮蔽terminal准确率43.1818%→45.4545%，但CE上升；组别和指标方向分别记录，不据单点或已有正例关系准确率推断功能收益。

新点170计数/40mask_ratio、1024验证记录与provenance保持，summary匹配，无终态或测试权重。详细数值见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3250_trend_20261006.json（SHA751d8c32272cc014655d5967536d5e0c89eee48e806cb1e82b44c2b6629bd82f）。原八卡f011最近完整保存4000且继续25042目标。完整22格仍8完成/1运行/13未启动，固定500/3750图不重画；下一本地3500、正式5000，既有训练和门禁不变。


## 2026-10-06T02:21:32.688388+00:00 本地3000步验证完成：平均AA CE持平，全遮蔽与结束标记准确率回落

本地f000 seed42完成3000步验证、96000/120000条曝光（80%），继续3750步完整epoch。平均AA CE0.849889→0.849367基本持平；全遮蔽AA CE1.744851→1.767363、准确率41.8326%→38.5985%；全遮蔽terminal准确率39.0018%→29.2052%。含context37条的全遮蔽AA CE上升、AA准确率下降，terminal准确率43.1818%持平。已有正例关系CE下降不等于正负binding能力。

仅核对新3000点，170计数/40mask_ratio、1024验证记录与provenance保持；summary匹配，无终态或测试权重。详细统计见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3000_trend_20261006.json（SHAb11c26591a09b065f7176f76abf0a42a46d1db24036f2f23b36ed7185ba1e1fa）。原八卡f011最近完整验证保存4000并继续25042目标，两配方不混用。完整22格仍8完成/1运行/13未启动；固定500/3750图保持，下一本地3250/正式5000，既有训练和所有门禁不变。


## 2026-10-06T02:04:56.990109+00:00 正常读取与内存回放的配对测速入口已实现，GPU尚未执行

私有v3入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/profile_real_batch_pair_v3.py 已完成根源码复核；SHA256=b8be85365f947b2aa352f6f6c105750188b5919aaea5ca1bb6c2c9c710553a01。新4项人工CPU控制检查实际通过（suite0.094秒、脚本3.125秒、工具退出0），覆盖固定窗口/预算、配对一致与差异拒绝、CPU随机状态重置和延后噪声哈希，以及原v2父进程生命周期保持。没有重复原输入3项或旧模型测试，没有真实V7读取、模型实例/forward/backward、CUDA或远端API调用。结果/实际退出记录及运行说明见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/README.md；根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/root_pair_runner_review_v3.json，SHA256=4a9b3cce0cc33ec199196709f30b24f2be25311ccda7dd6cc81969b84f2690fe。

实际实现限定f011、micro32/GA1、1024条唯一输入；normal和CPU回放分别重新创建seed42模型、空AdamW及全部RNG，每模式5预热+16分项+10整步+1profiler=32更新，共64，不超过原80更新/1800秒（默认900秒）。两阶段核对初始化及每步输入/标签/噪声/随机状态/曝光一致；同口径CPU→GPU复制，读取、拼批、验证clone、H2D及模型阶段分别记时。噪声哈希与D2H观察在本步GPU计时和profiler外；原forward同步保留。测试不写权重、序列或token缓存。

当前仅达到“代码准备与CPU控制检查完成”，还没有真实V7/GPU计时结果，不能确认GPU随机配对、显存容量、速度收益或DDP瓶颈。16个分项样本不证明统计稳定；整步wall包含额外且不对称的验证/clone成本，不能直接相减当loader收益。首次prefix索引已预热页缓存，单卡诊断不等价于正式8rank DataLoader/通信/验证/保存吞吐。未来须资源实际释放并确认独占后才可执行，当前八卡和本地训练继续，不抢占其GPU。

v1/v2及冻结正式源未改，人工fixture和测试临时权重均无残留，正式恢复状态保留。远端403不重试，序列注释停线、独立CUDA暂停与原审核范围保持。原正式25042和完整22格消融目标仍未完成。


## 2026-10-06T01:49:40.093225+00:00 批次回放输入接口已复核；配对GPU计时仍待整合

输入适配器已实现并由根端完成源码与原始证据复核。原人工CPU接口检查3项通过、实际退出0，覆盖单次捕获、同序回放、clone隔离、读取失败不重试及context序列CE0/已有binding正例。根端没有重复测试、读取真实V7记录或运行GPU。临时人工fixture已清理，无测试权重或bytecode残留。根复核原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/root_input_adapter_review_v1.json，SHA256=e15dd7305fdec3da1b8dd21a46e52a9816f818ea537266e60bf355f93624d39d；接口及限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation/README.md。

这不是GPU测速通过：真实输入捕获尚未执行，正式数据审核没有变更，micro64只有计划边界检查。后续只准备f011的normal与CPU缓存replay配对入口，每模式32个唯一batch，5预热+16分项+10整步+1profiler，共64更新；保留原80更新/1800秒上限与默认900秒。两个模式分别重建同seed模型、optimizer和噪声RNG，核对初始化、输入/标签/噪声选择及曝光一致，H2D同口径，校验/clone成本单列。16个分项样本只用于短诊断，不能声称稳态或端到端DDP加速；此入口尚未完成或执行。

正式训练继续单机8卡DDP+NCCL，历史纯LLaDA270M为两机各4卡FSDP；并行方式差异不是已证明的慢化原因。本轮没有submit/cancel/restart或正式配置修改。原远端进程查询API403分支不重试，序列注释停线与独立CUDA暂停保持；已有完整22格消融和25042步正式目标未完成。


## 2026-10-06T01:33:09.367200+00:00 真实批次回放适配准备；原训练继续

上一Goal回合完成正式4000/本地2750结果复核、三组工程计时CPU接口和远程API拒绝记录，分类progress。本轮再次直接核实原监督/本地PID、start_ticks/argv及八worker心跳，平台最近Running、无abort/终态；本地推进到2800步。原4000验证与保存证据不重复检查，CURRENT交接表已从旧3000更新至实际4000，历史章节保留。

用户速度定位仍缺实际V7批次回放：已授权准备私有适配器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/real_batch_replay_preparation，复用原1024记录前缀选择、原PreparedImmuneDataset getter和ImmuneCollator、三metadata/SHA限定审核门禁。预期normal首次读取并拼批时缓存CPU批次，replay仅用同一内存缓存；不持久化序列/记录/token，不执行序列注释或重建审核。本轮仅代码与必要人工CPU接口验证，未再读真实V7或运行GPU。

公平性边界已明确：原2048行读取预算=1024建立索引+1024实际getitem；额外先预拼再读取会变3072，不能沿用旧预算标签。1024条只提供32个micro32 batch；人工计时入口默认46更新不能直接照搬，更不能靠重复读或缩短稳定窗口冒称公平实测。先交付批次捕获/回放接口与显式数量边界，再确定最小GPU计时整合；实际GPU准备/执行状态必须分开。三组人工入口v2和已验CPU证据保留。

原八卡/本地任务均继续，不抢占GPU、不重启/提交/取消或更改正在运行的来源、数据、batch、loss。云平台StartExecCommand已经403拒绝，保持不重试、不换接口/agent绕过；该诊断缺口不阻断正在运行的训练。独立CUDA算子暂停与序列注释停线保持。


## 2026-10-06T01:28:20.983906+00:00 官方远程进程查询实际被API拒绝，原训练继续

已核实官方SDK存在StartExecCommand（2024-07-01、ml_platform、POST），并核实当前实例t-20261005205851-lqfjg-worker-0为Running。既有CLI没有exec，先前top握手失败未提供进程表；本轮使用官方endpoint/签名规则与现有认证，在内存中签名，一次调用有20秒上限的只读进程/GPU元数据命令。无SDK安装、权限/配置修改、原始cmdline或环境输出，签名/凭据没有落盘或输出。

实际响应HTTP403、AccessDenied，未返回ExecutionId，也没有生成远程进程快照。包装器本地退出0仅表示收到了结构化错误，不表示远程命令成功。实际Start请求1次、history0、重试0；独占intent保留，不重发、不换agent/接口绕过拒绝。本地full access不改变云平台API授权，此处无需用户再确认本地执行。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/remote_process_access_preparation/root_execution_receipt.json（SHAcd565d2cc692c5071a7dd55791ea3a41be2bd4af61dbf20c075e30c633ac5ccd），官方接口与host依据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/remote_process_access_preparation/preparation.md。现有pool源码在训练启动前stop/wait占卡进程，8份占卡日志在2026-10-05约13:28停止，仍支持已停止判断；因实时进程快照缺失，不能称直接排除了当前容器所有竞争。没有据此停止容器或正式训练。三组GPU计时仍待现有训练释放资源，原4000步正式训练/2750步本地验证已核实并继续，无新GPU模型测试/测试ckpt。


## 2026-10-06T01:27:15.290670+00:00 正式4000步验证与保存完成：AA改善，结束标记与长度误差仍波动

正式checkpoint-4000已完成：17原生文件与6JSON归档齐全，8rank参数一致且已更新；固定1330验证记录/112计数字段保持，70聚合与18结束位置检查通过。平均AA CE0.773253→0.759145，全遮蔽AA准确率42.1558%→42.9299%；全遮蔽terminal准确率57.7268%→48.7537%，长度MAE2.761715→4.073815，来源方向不一致，不能仅凭平均loss宣布架构或生成能力通过。

细项与逐来源/长度限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_root_observation.json（SHA98f9e2b5e386ef0d3cb3fabe9d9a8ddc8bcc1dad5e935a57fb7d3c5be68c91a9）、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_endpoint_diagnostics.json（SHAbc7170739dd205bbabad27213e8bd1592ab1ddf2bec5c73b431ae0101d404e11）。原八卡继续5016段/25042总目标，本地最近2750验证；下一正式5000、本地3000。旧1000权重按原三份保留规则轮替、JSON证据保留；未重启或重跑模型。完整22格及全部原边界保持，Goal active。


## 2026-10-06T01:20:17.771177+00:00 三组整模型计时入口补齐；GPU尚未执行

用户要求与原纯LLaDA比较速度，本轮新增版本化 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/profile_whole_model_v2.py（SHA3323b8ae32b824f5016b873f3da243db3b6ad927ba3e4a87b6c23539c9ac3645），支持plain_v50/f000/f011。普通control为d768/L8/H12/FFN3072、V50、原生LLaDA blocks、默认全局RoPE与padding mask，移除role/scope/chain-bias；与另两组复用冻结noise、selected CE、相同整数输入和优化器/精度/重计算配置。相同seed保留构造RNG消耗使共同权重一致，预计标准模型75587328参数；不是旧126464词表270M重放，也不是实际V7效果对照。

新增tiny CPU模型（d32/L2/H4/FFN64）检查实际exit0、14.72秒、CUDA未初始化：共同权重、噪声/selected、手算CE、padding-only调度和忽略链特征通过，三组各一个完整BF16 forward/backward/AdamW更新通过。三份CLI计划exit0；只是CPU接口和计划检查，未构造75M GPU模型、测速或读真实数据。记录中的float(loss)提示仅为报告转换建议，原样保留，没有为消除提示重复测试。

v1及证据保留，8个几何/隔离/空闲GPU检查/执行生命周期函数AST不变；父子进程仍拒绝已占用GPU，默认46更新/900秒、保留实际exit/timeout且不保存权重。根读取新源码/差异、原件SHA和CPU输出并复核范围，没有重跑模型。根审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/root_three_control_review_v2.json（SHA9ad9517d51b2f3cf39a6559d1defbb9c28f4d0b4e3b6923d1167cbb8e7779038）；执行证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/plain_control_cpu_validation_v2.json（SHAe1d3101d28d6ce2d33174b750f275ae0b2ae84cced0bf2f1f5096b3b6d8d67e7）。

后续在原授权且GPU可用的训练边界执行三组计时。plain→f000改变整套链特征/位置路径，不能只归因于chain bias；f000→f011测固定残差缩放及local/global组合增量。真实V7回放、八卡通信/loader分项、实际SDPA backend和micro64容量仍未验收，不把准备完成称为已找到瓶颈。没有发布pool任务、重启/暂停原训练、修改冻结源或恢复独立CUDA专项；无测试ckpt需清理。


## 2026-10-06T01:18:25.792578+00:00 本地2750步验证：总体重建回升，context结束标记仍有回落

本地f000 seed42完成2750步验证、88000/120000条曝光，仍继续3750步完整epoch。平均AA CE0.897508→0.849889；全遮蔽AA CE1.898242→1.744851、准确率37.9543%→41.8326%；全遮蔽terminal准确率20.1479%→39.0018%。含context37条的全遮蔽AA CE略升、terminal准确率52.2727%→43.1818%，正例关系CE也回升，不能只报告总体改善或声称结合功能收益。

新点170计数/40mask_ratio与2500相同，原provenance和1024验证记录保持；summary匹配，无测试权重或终态。学习率核对的检查脚本索引偏一已按冻结源修正，训练未受影响。详细数值与限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2750_trend_20261006.json（SHAb2deacc4257cef0180858d94ac99c3b903ccb0351cf2bef769db0420acf59a7a）。完整22格仍8完成/1运行/13未启动，固定500/3750图不按中途点重画。原正式八卡继续完整25042目标；下一本地3000/正式4000。


## 2026-10-06T01:00:12.404213+00:00 整模型计时入口已准备并完成CPU/CLI检查，GPU测速未运行

用户要求继续定位正式训练慢因。本轮在原八卡/本地任务继续运行时完成了private整模型计时入口与审阅，不改变训练。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/profile_whole_model.py（SHA581976199c87c8faa98199bec3924e44dec8123a665322ef7a9c83928ecf2db9），说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/README.md；根端验收范围原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/root_preparation_review.json（SHAe9b0c664736fccf938ccbe2949f6965dda7f063cc5189b18697dfcb36c6bb1f7）。

已实现f000/f011两组同主干、同人工形状、同seed、FP32参数/BF16 autocast、whole-layer checkpointing、原forward/noise/CE、真实backward/clip/AdamW。默认5个预热更新、30个按阶段同步计时、10个仅整步边界同步计时、1个独立完整更新profiler，共46个更新，父进程900秒上限；记录真实child exit，不自动重试，不保存模型/optimizer权重。额外检查链bias梯度存在并记录范数，profiler保留实际SDPA、index/scatter等事件，不预设瓶颈。

根端审查后已修复两点：仅查顶层模型路径不足，改为拒绝混合项目模块缓存、过滤工作树sys.path、逐个检查实际加载的dllm/scripts/examples文件与namespace路径并匹配既有冻结manifest；分阶段同步会扰动吞吐，故`instrumented_*`与`whole_step_boundary_only`分开，后者仍保留原模型/optimizer内部同步和诊断开销，不冒称正式Trainer/DDP吞吐。未重复扫描完整611源验收。

CPU/CLI证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/preparation_validation.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/preparation_validation_v2.json（最终SHAc4d46ae6b88f1507022f68bcaee8124cc523cae569f21b49419e1975027f37c2）：原三种人工布局检查实际exit0且未初始化CUDA/模型；新增源码隔离、AST、f000/f011计划与更新预算拒绝检查通过。根端读取原件与最终代码SHA并复核接口/预算，不重复已完成CPU布局测试。没有新的GPU运行、真实数据读取、独立CUDA算子测试或测试ckpt。

当前状态必须分开：**代码准备与CPU/CLI检查完成；GPU执行/吞吐/实际backend/OOM容量尚未验收；普通V50 LLaDA control、真实V7批次回放与八卡分项时间仍待完成。** 人工512个target均有监督，不能代替正式空槽/关系/context配方或实际训练数据实验，不能纳入架构效果消融排名。入口只支持明确独占的空闲单卡，父子进程均先拒绝已有compute PID；该快照不是资源预约。本轮没有发布pool执行入口、提交/取消任务、暂停/重启原训练或新增占卡负载。

原正式训练继续5016段与完整25042目标，本地f000继续3750步完整epoch。无新验证节点不重复旧验收/绘图，完整22格仍8完成、1运行、13未启动。Goal保持active；后续优先采集正式4000和本地2750新节点，在原授权及资源允许的训练边界执行整模型计时。保留正式恢复状态；本次仅代码/小型JSON/MD证据，无测试权重需删除。原序列注释停线、审核门禁和独立CUDA暂停保持。


## 2026-10-06T00:48:10.353404+00:00 整模型速度定位入口准备中，原训练继续

上一Goal回合为progress：已交付纯270M历史对照与有界CPU loader实测。本轮先核实原监督PID3977587、本地PID3970535的start_ticks/argv及8worker心跳，原任务继续，无新验证节点。已关闭的84217不再读取；本轮有限只读观察为session25737，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_bounded_watch_20261006T004708Z.jsonl。它不重启或替换原监督器，也不改训练。

原已授权性能定位继续：已有agent复用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/profile_v7_large_batch.py，准备private metrics-only完整模型计时入口，输出限制在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/。先覆盖f000/f011相同形状、FP32参数/BF16 autocast、whole-layer重计算、完整forward/backward/optimizer与profiler事件；普通小词表LLaDA control仍待明确实现，不能把两组称为三组已完成。

本阶段只准备实现和必要CPU/CLI验证，不占用现有训练GPU、不读取更多实际数据、不生成模型权重、不改变生产/冻结源、调度或门禁。人工整数输入只能用于整模型工程路径，不能替代真实V7 pipeline或效果验证。独立CUDA算子专项仍暂停；chain_bias展开反向等仅是待整模型profile检验的假设，不指定已找到瓶颈。脚本可用、CPU验收、实际GPU测量和速度改善分开记账，后续资源可用时再在原授权范围内执行。


## 2026-10-06T00:43:32.848969+00:00 新增CPU实测：loader小样本完成，下一定位对象为模型路径与padding

用户最新问题已核实：当前正式八卡是DDP+NCCL，旧270M是两机八卡FSDP，当前没有切换。完整旧纯LLaDA比较、来源与受控计时方案已完成于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LLADA270_SPEED_COMPARISON_20261006.md；它包含旧2.054/2.122秒每新增step的原始长段证据、相同主干和词表参数差异。补充原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/llada270_review/supplement.json（SHA4953b8e3f042ae75413b99ff7355c2a4b10e4a46c14baf3ecdae270678fd1bf3）确认旧逐残基token映射不压缩长度；实际旧训练长度分布仍未知。

本轮CPU-only诊断实际完成，agent工具观测退出0，1024条不同记录、总计2048次记录行读取；激活pllm后用正式固定解释器和单CPU线程，未初始化CUDA或模型。原正式绑定三份metadata/SHA与限定审核范围保持。没有重跑全量审核、全量索引构造或模型验收。

| 本地CPU测项 | micro32 | micro64 |
|---|---:|---:|
| getitem 中位/p95（毫秒） | 1.484 / 2.140 | 2.959 / 4.136（同一次32组读取配对求和） |
| 实际collate 中位/p95（毫秒） | 36.036 / 49.057 | 78.842 / 102.158 |
| 读取+collate代理 中位/p95（毫秒） | 37.459 / 50.578 | 81.787 / 105.275 |
| 稳态计时batch数（各去前2组） | 30 | 14 |
| batch渲染长度 中位/p95 | 539 / 1130.45 | 743 / 1168.30 |
| 全部batch外部padding比例 | 43.5846% | 51.2686% |
| attended位置中的target内部空槽比例 | 53.0817% | 53.0817% |

这次局部样本的CPU读取/拼批远小于正式10.6–11秒/步，尚无证据说JSON读取是主要原因；不过索引预读已预热缓存、七来源取固定分片前缀，本地主机单进程不代表正式容器八rank的存储/CPU争用，不能完全排除远端loader等待。micro64的getitem不是独立再次读取；两种batch的collate是各自实际计时。稳态分位数与全部batch的padding统计口径分开。

padding是值得优先量化的线索：同样1024条记录拼成更大batch后，最长序列增加、外部padding升高。内部target空槽与外部padding不同，属于现有布局，不能直接删除后称数学等价。最大观测长度1206，小样本不证明没有更长真实batch，不据此保证64显存安全或GPU速度翻倍。

根端判断：已完成历史速度差、当前分布式、占卡实现/日志、容量依据和本地CPU加载的核实；唯一瓶颈仍待整模型GPU时间分解。下一受控测试先固定批次/形状/全局batch，对比loader回放以及普通小词表LLaDA→f000→f011的前后向/attention路径，再定位单项优化；独立CUDA算子专项保持暂停。当前8卡和本地卡已有实际训练，没有在其上叠加profile负载，没有为测试重启或提交新任务，未改正在运行的冻结源、架构、loss、batch、数据或恢复状态。

结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/result.json（SHA98e28f54dba6dd665459f3bfc73b0c425762ff170370597d707e7d4aabb4bf75），原脚本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/probe_data_loader_cpu.py，详细方法/限制 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_DATALOADER_SPEED_REVIEW_20261006.md。只产生脚本与小型JSON/MD证据，无测试ckpt、训练数据副本或模型权重需删除；实际正式恢复文件按原规则保留。


## 2026-10-06T00:40:00.511063+00:00 纯LLaDA270M速度对照与当前DDP核实

当前正式训练为单机8卡 **DDP + NCCL**，不是FSDP；冻结入口明确拒绝FSDP/DeepSpeed。旧纯no-ESMC LLaDA270M为两机各4卡、FSDP world8。当前最新完整训练日志为3725/25042步，最近完整验证/保存3000，训练继续；本次没有切换分布式策略、杀进程、停容器或重复提交。

| 项目 | 旧纯LLaDA270M（2026-09-23/24实际长段） | 当前V7 f011 |
|---|---|---|
| 平均秒/优化器步 | 2.054348 / 2.121932 | 10.956243 / 10.596197 |
| 名义全局记录/秒 | 124.61 / 120.64 | 23.37 / 24.16 |
| 全局batch | 4 × GA8 × 8卡 = 256 | 32 × GA1 × 8卡 = 256 |
| 参数总量 | 269759232 | 75601176 |
| Transformer主干 | d768 / 8层 / 12头 / FFN3072 | 相同 |
| 最大长度配置 | 1024 | 2048（按批内最长padding） |
| 数据加载workers | 4 | 0 |
| 整层activation checkpointing | 开启 | 开启 |
| 分布式 | FSDP，2机×4卡 | DDP，1机×8卡 |

旧记录使用长段实际新增step计算；续训从36000起，178416进度对应142416新增步，避免把恢复前step计入短段吞吐。新旧速率均含各自验证/保存开销，约5.0–5.3倍差距已证实；数据、长度分布、mask、微批和分布式不同，仍不是控制变量因果测试。旧270M较大的参数数目主要来自126464词表embedding/head；不能认为新模型主干缩小3.6倍，也不能把固定residual scale当作新增大型网络。

本轮判定：旧版也开整层重计算，因此它不能单独解释新旧速度差。优先检查密集链bias/局部mask的构造与反向、实际SDPA backend、批内padding及数据等待；现有证据尚不足以指定唯一瓶颈。链bias虽然仅24个可学习参数，却展开为[B,12,L,L]；B32/L1024单张FP32 bias理论1.5GiB，L2048为6GiB，尚未计副本/梯度。这是形状算术，不能代替实测峰值或性能归因。局部mask仍为密集计算，没有获得稀疏attention的理论节省。

占卡核实：原pool源码在每卡启动真实训练前调用standby.stop并wait，8份standby日志均在2026-10-05约13:28 UTC停止；8worker当前均running。现有证据不支持占卡负载持续干扰。只读instance查询为Running，但CLI top返回websocket: bad handshake（CLI退出码虽0，不能视为取到进程表）；尚无容器当前逐进程快照，因此不将日志推断冒充实时进程证明，未据此停容器。

batch64：现有共同容量选择实际测试4/16/32，选择32；没有当前f011在真实长度分布上的64容量/吞吐结论。代表配置micro32已测reserved约49.64–61.88GiB，不能用当前瞬时显存约27GiB直接保证64。若8卡GA1改64，全局batch变512、相同数据预算约12521步，需要单独记录优化器/调度差异，不能仅步数减半就声称时间减半。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/root_readonly_followup_v1.json（SHA4c1e4a5a00875f0b03f15755d757e020eda3d6bb698282cb597da53638c14096）；历史细项 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/llada270_review/review.json（SHAb1b0e98132adfdd0a013356378d4c2e5d136b6c044441fc2a1607d9d48fe5b28）。CPU稳态loader有界probe由已有agent继续，尚未交付结果；没有新增GPU模型测试或CUDA算子专项。真实训练恢复文件保留，未生成测试权重。


## 2026-10-06T00:27:37.052546+00:00 用户速度问题：已核实步数预算和吞吐，当前训练继续

25042=ceil(6410749 / (32×8×1))，是约一轮自然数据的预算。查询时最新训练日志3675步，最近完整保存3000；这两个数字分开报告。最近两个千步区间为10.956/10.596秒每步（含验证/保存），名义吞吐23.37–24.16条/秒；若速率保持，整轮约73.7–76.2小时，从3675步约还需62.9–65.0小时。

当前8卡快照100%忙、26.59–28.51GiB显存，不能当峰值或算力效率。源码确认整层重计算、密集[B,H,L,L]链bias/局部mask、按batch最长padding及单进程loader；尚无组件耗时和实际SDPA backend证据，不能指定唯一瓶颈。保持原训练继续，后续按整模型计时与峰值检查优化重计算/attention/padding；独立CUDA专项继续暂停，不改当前冻结源码或既定batch/调度。

详细记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/SPEED_ANALYSIS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/speed_diagnosis_20261006T002559Z.json（SHA8c3899842357341cbac9e3415875752ef49a91656b2e72d9f4e11e26923fbb81）。本轮没有新GPU任务、实际profiler或模型测试；记录脚本的JSON整数键读回断言已修正解释并核实原件，实际训练未受影响。

## 2026-10-06T00:11:42.114535+00:00 本地2500步：记录全遮蔽退步，继续固定完整epoch

本轮f000的2250→2500平均AA CE0.876467→0.897508，全遮蔽AA CE1.774273→1.898242、准确率41.1548%→37.9543%；terminal准确率27.1103%→20.1479%。25%/50%/75%遮蔽AA CE改善，含context37条的平均AA CE1.265946→1.234962，但其全遮蔽AA CE略升。保留不同方向的变化，不以整体均值概括全部档位或子集。

继续原3750完整epoch与正式f011/25042协议，架构、loss、batch、500/3750比较终点和完整22格保持。170计数/40mask_ratio与2250相同，来源/data provenance保持；仅核对新2500点，旧同seed曲线止于2000。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2500_trend_20261006.json（SHAa9e3a6b66553171173c3977149f542a0d7bf7c3a97ce6848a4224aa388846e76）。下一本地2750、正式4000；无新增模型测试、GPU任务或训练源码修改。

## 2026-10-05T23:07:29.260939+00:00 本地2250步：保留全遮蔽退步和context子集差异，继续完整epoch

本轮f000从2000→2250平均AA CE0.886307→0.876467，但全遮蔽AA CE1.761686→1.774273、准确率42.2681%→41.1548%；terminal CE1.236364→1.884492、准确率55.5761%→27.1103%。含context37条验证的目标AA平均CE1.256698→1.265946，无context987条则0.879368→0.869177。不能用总体平均loss下降概括全部训练效果；分组样本少，不归因架构或binding能力。

继续固定3750步完整epoch及正式f011/25042协议，保持原数据、loss、batch与500/3750比较终点，不按单点挑最好结果。新点170计数/40mask_ratio与前一2000相同，provenance保持；旧同seed曲线止于2000，2250只能作本轮时间变化。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2250_trend_20261005.json（SHAda1eb090ec4fdda83c8ce822e2c05deb77ed4116c3c4a48485779bb1b7fcb500）。完整22格范围不变，下一本地2500、正式4000；无额外模型测试或源码改动。

## 2026-10-05T22:30:51.057390+00:00 正式3000步：继续原协议，保留来源与结束位置的混合趋势

正式2000→3000按token平均AA CE0.797064→0.773253、全遮蔽AA CE1.771696→1.760729；全遮蔽terminal准确率41.4257%→57.7268%。OTS/TCR native的AA CE回升，部分TCR来源terminal退步；来源等权四档平均terminal准确率也下降。首END准确率/长度MAE改善，但token END精确率下降、AA位置误报率上升。不能将某个聚合量改善等同于所有来源或生成能力提升，不作因果或显著性结论。

继续固定f011、V7配方、loss、batch与25042步调度；下一4000观察相同AA/terminal/首END/长度及逐来源结果。完整22格消融与固定500/3750终点保持，本地仍跑原3750步完整epoch，下一2250；不因单点中途改架构。正式1330验证与pilot1024验证独立，正例token监督不代表正负binding。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_root_observation.json（SHA88a4133af4a1f1005b84a5133ee7645d59c9ea99f812bb8aa1f99c9899aeff56）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint3000_endpoint_diagnostics.json（SHA743b9b80db5eb35acd46ab2798afa4e344fe78bbf8b6c55ea103b2127f834976）。无新增模型测试或训练源码改动。

## 2026-10-05T22:01:53.627091+00:00 本地2000步：保留实际改善，继续既定完整epoch

本轮f000从1750→2000的平均AA CE0.907134→0.886307，全遮蔽AA CE1.779121→1.761686、准确率40.4027%→42.2681%；terminal CE1.920001→1.236364、准确率32.4707%→55.5761%。结束标记在此前回落后改善，仍不足以判定完整训练或生成质量；正例关系平均CE略升也保留。继续原3750步固定协议，正式f011/25042步不变，不因中途单点调整架构、loss或batch。

与旧同架构/seed2000平均AA CE0.886866/新0.886307接近，曝光/调度、170计数字段及40mask_ratio匹配，provenance仍有两顶层差异。不是新架构、ESMC、位级复现或生物功能收益结论。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2000_comparison_20261005.json（SHA0b7214c684756ff325aeca81365bb3b870cf4dabd3d061d0ad4af90e7c88ce43）。旧失败曲线最后可用点为2000，后续2250以上只用本轮实际曲线，不补造旧点。完整22格与固定500/3750终点保持；下一本地2250、正式3000，没有新模型测试或源码变更。

## 2026-10-05T20:59:44.248817+00:00 本地1750步：保留混合趋势，继续固定完整epoch

当前f000从1500→1750的四档平均AA CE0.931826→0.907134，全遮蔽AA CE1.782721→1.779121；但全遮蔽AA准确率40.9893%→40.4027%、terminal准确率34.0111%→32.4707%，terminal CE1.610497→1.920001。保留这些不同方向的变化，完整3750步之前不改变已固定对照的架构、loss、数据或batch；正式f011/25042步继续，后续比较仍使用原固定500/3750终点。

同架构/seed旧1750与新1750的平均AA CE0.905755/0.907134、full AA CE1.773030/1.779121；曝光/调度、170计数字段和40mask_ratio匹配，provenance仍有两顶层差异。数值差异原因未确认，不能归因新架构、宣称位级复现、ESMC或生物功能收益。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1750_comparison_20261005.json（SHAb938476b03478015c2b3b7c6823365affc35857de09722f8a7f9a81919c8d530）。下一本地2000、正式3000；完整22格范围保持，未新增模型测试。

## 2026-10-05T19:52:34.799686+00:00 本地1500步：AA loss改善与准确率回落并存，原选型保持

当前f000从1250→1500的四档平均AA CE0.961122→0.931826，全遮蔽AA CE1.787297→1.782721，但全遮蔽AA准确率41.2351%→40.9893%、terminal准确率36.2292%→34.0111%，terminal CE也上升。保留这些变化，不根据单点调架构、loss或batch；继续原3750步完整epoch和正式f011/25042步计划。

同架构/seed旧1500与新1500的平均AA CE0.930847/0.931826，full AA CE1.782431/1.782721；曝光/调度、170计数字段、40mask_ratio匹配，provenance仍有两顶层差异。不是位级复现、ESMC或架构增益结论。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1500_comparison_20261005.json（SHA4e4afa7f149b2838567c7e3aa8b29eb4861c1f4eaa759f8078d284395e956968）。完整22格范围与固定500/3750对比保持，下一本地1750、正式3000；没有训练源码/数据变更。

## 2026-10-05T19:35:46.431403+00:00 正式2000步：补充结束位置诊断（只读已有结果）

复核冻结验证源码并重新汇总原1000/2000日志，新增模型运行0。当前评估是一次全遮蔽去噪前向的argmax，不是多步生成或生物功能验证。“terminal准确率”仅看真实结束标记位置；“首个结束位置准确率”在包含容量尾部的全部可见target槽位查找第一个预测END，按链计数。

| 全遮蔽指标 | 第1000步 | 第2000步 |
|---|---:|---:|
| 按链汇总首个END位置准确率 | 16.1515%（324/2006） | 13.2602%（266/2006） |
| 七来源等权首个END位置准确率 | 13.2088% | 15.8266% |
| 按链汇总长度绝对误差 | 2.463268（2001条可比较链） | 3.310070（2006条可比较链） |
| 七来源等权长度绝对误差 | 2.563148 | 2.850117 |
| 未预测出END的链数 | 5/2006 | 0/2006 |
| AA位置误预测END比例 | 1.2241%（2135/174410） | 0.7855%（1370/174410） |
| token END精确率 | 34.9284% | 37.7556% |

首个END指标在日志中归入token_micro，但实际分母为链数；不能与七来源等权混用。长度误差只在存在预测END的链中统计，两点分母不同，不能声称逐链误差对照。token精确率和AA误报率不计容量尾部，与首个END的搜索范围也不同。总体首个结束位置退步与来源等权改善并存；两种长度误差均变差，缺失END减少。原因未确认，不能据此认定训练目标、架构或解码出现特定故障，也不称生成质量改善。

继续原f011/25042步协议，在下一既定3000步验证观察同一组指标。没有重新验证旧checkpoint、改变训练、启动额外任务或重启。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint2000_endpoint_diagnostics.json（SHAc6986c542ef2e87340f2593b562061e83c015010a87f0d9331681680507e6596），36项计数/聚合复核；定义源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/architecture_prototypes/validation_metrics.py，SHA2a5382be35c364140143d65060d60a41bba7aa609b51fc29846db3d1160d23ab。正式训练与本地消融继续，完整任务尚未完成。

## 2026-10-05T19:32:51.240366+00:00 正式2000步：AA学习继续改善，保留结束标记退步观察

第1000→2000步按token汇总四档平均AA CE0.913768→0.797064，全遮蔽AA CE1.837341→1.771696；全遮蔽terminal准确率57.1286%→41.4257%，其CE1.545641→1.659838。不能将AA改善等同于生成质量全面改善，也不据单个节点临时调训练协议。原f011、数据配方、损失、batch和固定25042步保持，下一3000步观察AA与terminal是否持续分化；完整22格消融与3750步固定比较口径保持。

同一1330记录/四档mask、112计数字段相同，70聚合指标已从逐来源复算；不是架构对照、ESMC对比或生物功能结论。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint2000_root_observation.json；详细结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。本次仅解释新训练结果，没有额外模型测试/源码/数据改动。

## 2026-10-05T18:47:08.472956+00:00 本地1250步改善，初步f011与消融协议保持

同一当前实验1000→1250步：四档平均AA CE由1.008167降至0.961122，全遮蔽AA CE由1.798585降至1.787297，AA准确率由40.3622%升至41.2351%。全遮蔽terminal准确率由18.7307%升至36.2292%，仍需完整epoch观察；这不是位级复现、架构收益或ESMC比较。

新旧同格1250步四档平均AA CE为0.962144/0.961122，全遮蔽AA CE为1.787628/1.787297，完整3750步尚未完成；不因小数值差或阶段波动改架构。曝光、调度和各验证计数一致，关系指标仍只包含已有正例。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1250_comparison_20261005.json。正式八卡继续原f011，完整22格与正式25042目标保持，下一观察为本地1500步、正式2000步。


## 2026-10-05T18:33:58.003212+00:00 消融接续工程补齐，训练方法保持

未来本地消融的私有guard、共用格子/设备锁和报告器v2接入已完成本轮工程检查：子agent7项人工检查（5.984秒、rc0），追加实际临时receipt→reader接口1项（1.456秒、rc0）；根端报告器3项定向检查（1.411秒、rc0）。只有锁探针进行了真实CPU exec，训练exec为mock，无新模型/GPU测试。根端源码/原metadata字段/固定绑定和当前4个正式公共发布SHA已复核；没有重复运行既有pool或模型验收。

原f011正式训练与本地f000完整epoch继续，模型、样本、mask、batch与训练目标均未改变；不新增架构收益结论。未来新本地实验与pool共用格子锁，本地另持GPU0锁，保持既有3750步/120k/micro32/GA1/eval250/不保存测试权重协议。检查证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_guard_root_review_20261005.json。实际admission与接续尚待当前任务结束，完整22格和正式25042目标未完成。


## 2026-10-05T17:44:15.579408+00:00 1000步观察为混合趋势，初步f011选择保持

同架构/seed的1000步比较（按token汇总；不是ESMC对比或架构增益）：

| 指标 | 旧同格历史点 | 本地本轮 | 本轮−旧 |
|---|---:|---:|---:|
| 四档平均AA CE | 1.006546 | 1.008167 | +0.001621 |
| 全遮蔽AA CE | 1.798385 | 1.798585 | +0.000200 |
| 全遮蔽AA准确率 | 40.4124% | 40.3622% | −0.0501个百分点 |
| 全遮蔽terminal CE | 2.156175 | 2.082055 | −0.074121 |
| 全遮蔽terminal准确率 | 19.0388% | 18.7307% | −0.3081个百分点 |
| 四档平均terminal准确率 | 70.9275% | 70.9231% | −0.0044个百分点 |

相较本轮750步，四档平均AA CE从1.066193降至1.008167，但全遮蔽AA CE从1.777032升至1.798585，AA准确率从42.0923%降至40.3622%。不能把均值改善概括为所有条件都改善，也不据单点改架构或调训练协议。

曝光和调度一致：32000条、输入token12336197、eligible4937019、selected2488592、学习率8.57637576121707e-5。170个records/*_count字段与40个mask_ratio字段分别相同；1024条验证含37条context。根端仅用既有metadata helper核对新旧1000点的计数、分子和聚合，未额外重跑模型。两份provenance SHA与此前原件保持，差异仍仅checkpoint_saving/source_sha256；不称位级复现、受控速度比较或生物功能收益。旧点来自最终失败/未完成的历史尝试，不算旧完整epoch成功；关系准确率1.0仍仅为既有正例token结果。

实际证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation1000_comparison_20261005.json。继续当前f000完整3750步和原八卡f011正式训练；这次同架构复测不形成新方法增益结论。

## 2026-10-05T17:12:40.120217+00:00 保留当前训练，私有后续消融接入已检查

私有后续消融接入代码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py 已完成本轮实现和审阅，SHA a9baf1390cdeeba315c7984c8602599e29033a356fd7649b82dcb206899a5fe8；对应人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_metrics_only_ablation_dispatch.py，SHA 9b6d70e8844373299857ac9b160b828f3f0846d12afc3c8c90a47b48de772c0a。子agent实际执行6项人工元数据检查，3.467秒、rc0；根端检查了源码和覆盖范围，没有重复运行这些检查。根端另用实际既有来源/旧池容量/样本与父审核元数据运行check_sources_and_batch和check_data，2.153秒、rc0，未导入任何ML模块或读取序列。

已修正resume默认False、worker字符串device、formal_member别名，以及拿锁后重读认领信息和核对锁inode。exec只在人工测试中被mock；真实prepare/main、真实catalog/票据、公共发布和GPU执行均未发生。现有本地启动器尚未共用该锁，后续必须接入后再混合调度；本轮不能宣称跨环境互斥或GPU接入已验收。正式25042及八rank清理/空闲、剩余资源预算和最新缺失格子核实仍是未来发布前置。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_formal_pool_adapter_root_metadata_review_20261005.json（SHA e347b9dece500d8e59b07999a4eed04ce66803ddee0414d2697e7ad92881df2b）；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。当前四个正式公共发布文件SHA不变；自有人工fixture及本地测试权重/tmp均无遗留。正式恢复文件保持。

本次只改执行接入，不改变f011正式架构、消融22格范围、120k/3750/micro32GA1/eval4协议或审核范围。

## 2026-10-05T17:10:27.763497+00:00 私有消融接入实现中

后续消融的私有接入代码已写入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py，尚在检查，未发布/入队或运行模型。根端初审已要求并核对修正resume默认False、worker字符串device、formal_member别名和拿锁后重查。人工元数据检查尚未完成；未来本地启动也要接入共用锁，不能把单个适配器当成跨环境互斥已完成。

架构和既定实验配置保持；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。

## 2026-10-05T16:52:36.604444+00:00 正式训练后复用八卡补齐已有消融

后续八卡复用的代码只读审阅已记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md：既有worker可继续使用，但旧消融入口绑定旧pool/568文件快照，不能仅换目录直接启动新的metrics-only任务。拟用私有薄适配器保留worker分配的CUDA mask和原数据/来源/batch门禁；正式25042及全部清理完成后刷新缺失格子，再决定实际接入。适配器尚未实现/发布/入队；当前公共发布文件与原SHA一致，训练继续。

本次是执行方式判断，不新增架构或修改初步f011选择，也不把尚未执行的接入检查记为通过。

## 2026-10-05T16:39:14.429613+00:00 本地750步同格观察：继续完整epoch，不据中期差异换型

四档平均AA CE新1.066193/旧1.065402（差+0.000791，约+0.0743%）；全遮蔽AA CE新1.777032/旧1.775615（约+0.0798%），AA准确率新42.0923%/旧42.2900%。全遮蔽terminal准确率新28.4658%/旧31.0536%（差-2.5878个百分点），terminal CE新1.622146/旧1.626278；不因AA指标接近而隐去结束标记差异。四档平均terminal准确率新70.2284%/旧71.5257%。

样本/输入token/eligible/selected token/学习率一致，170个records或*_count字段及40个mask_ratio字段分别核对一致；1024条验证含37条context。逐来源和context分组的计数、分子与聚合已用既有metadata helper核对，没有重跑模型。provenance SHA与500步原件保持，差异仍为checkpoint_saving/source_sha256；不称位级复现，数值差异原因未确认，不归因架构收益、运行速度或功能改进。旧同格为未完成任务的750步历史观察，不作为旧完整epoch终态。原f011初步选择与当前训练保持。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation750_comparison_20261005.json。本次是现有f000训练的新常规观察，没有新增架构或更改f011正式训练。

## 2026-10-05T16:30:57.845872+00:00 正式1000步中期结果，f011选择保持

正式f011的1330条验证中，四档平均token汇总AA CE0.913768，全遮蔽1.837341/准确率40.5602%；来源等权AA CE分别1.435434/2.096753。汇总口径、配方与pilot不同，不作直接架构收益比较。完整训练/22格消融均未完成，不据单个正式中期点换型。原正例监督、context序列CE0及既定架构保持；模型保存和八卡一致性通过，不等于生成或功能质量验收。完整结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。

## 2026-10-05T15:55:59.001982+00:00 实际消融报告接入，初步架构保持

[500步局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step500_architecture.png)（9/22），[3750步完整epoch局部图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/step3750_architecture.png)（8/22）。两图均明确PARTIAL，分别展示mask_mean和mask100，每seed一个点；13项尚未启动，本地f000仍在跑。SVG/PDF、CSV/JSON及实际渲染凭据均在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1；根端复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1/root_review.json。

新报告逐run核对各自冻结来源，并保持同数据/batch/调度/评估及同seed初始化与曝光；来源版本仍分列，不把跨来源差值全部归为架构收益。coverage_complete、all_exit_verified、source_homogeneous均false，当前不是最终22格比较。provenance中的源码字典只证明记录来源，不能证明其中每个模块实际执行。正式f011（R26+R24）及自然V7配方保持，不根据新中期图改型；下一节点为本地750验证、正式1000验证保存。

## 2026-10-05T15:34:01.700681+00:00 本地同格500步观察：初步f011选择保持

新f000 seed42的四档平均AA CE1.185176，旧同格1.184855；全遮蔽AA CE1.793335/1.792839，准确率40.2420%/40.4412%。各曝光/验证分母和学习率相同，执行来源差异仍显式保留。全遮蔽terminal准确率在500步均48.0591%，不将250步差异延续为稳定效果结论，也不宣称已解释其来源。完整3750步与22格尚未齐全；继续原正式f011和本地完整epoch，不依据这次同模型中期观察换型。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation500_comparison_20261005.json。

## 2026-10-05T15:22:05.935892+00:00 消融结果汇总接入与后续本地退出记录

现有比较器只支持原单pool、相同执行来源及pool rc0；本地retention改动带来5处实际执行来源差异，不能假造pool成功或当成同一来源。已敲定独立混合来源报告策略，分别标注完整末点覆盖、实际退出和来源一致性；代码尚未实现，不算最终图完成。当前本地f000照常运行；其原启动者未保留未来wait退出码，不能倒填rc0。后续13项已准备复用既有run_in_tmux并记录实际PIPESTATUS退出值的命令，未启动或重试。训练源码、数据门禁、f011选型和旧账本均保持。

说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_REPORT_INTEGRATION_20261005.md；未执行命令 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/managed_local_launch_plan_v1.json（SHA7413c5fc9783ea2aa81dc37ebbdb5ba25588e09f76cc88190039e1a9de26970e）。当前真实运行状态仍以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_SUPERVISION_20261005.md 及本地progress为准。

## 2026-10-05T14:28:32.090226+00:00 消融比较口径：250步同模型复跑记录

当前正式初步f011（R26+R24）维持，不据本地f000早期结果换型。新f000 seed42完成250步验证；全遮蔽AA CE1.799719（旧同格1.798783），四档mask平均1.538930（旧1.537953）。训练日志摘要是mask_mean，token_pooled顶层是mask100，不能交叉计算收益。数据曝光/学习率/验证计数一致，但不是逐位复现；terminal准确率旧12.6309%/新23.5983%的差异仍如实记录，来源未确认。继续完整3750步，不根据早期差异宣称架构提升或环境速度收益。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation250_comparison_20261005.json；完整矩阵仍8/22完成。

## 2026-10-05T13:11:04.146916+00:00 f011 首段八卡联合训练已实际入队

新容器 t-20261005205851-lqfjg 内已一次登记8个正式成员，初次观测 pending=0、running=8。初步f011（R26+R24）、seed42、每卡micro32×GA1×8=global256；同一完整horizon25042、LR1e-4/cosine、warmup1000、每1000步评估/保存，首次实际累计终点8步。正式目标约一遍6410749条训练记录；drop_last及epoch尾部实际曝光以Trainer记录为准，不虚称完全无重复/无遗漏。通过首段完整保存和8rank退出后，恢复至16步，原生optimizer/scheduler/每rank RNG及真实续步须通过原冻结driver的检查。

正式数据为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，原v3/611来源与限定scope双证书保持并在runtime再检查。当前正式采用自然epoch随机采样（context约4.65%），不是pilot的20.54%context配方；context序列重建CE为0，已有正例关系token监督保持，无新增负例。初步选择依据为实际完成两seed f001/f011，不宣称完整22消融已完成。

计划原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/plan.json，8个具体job ID在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/enqueue_receipt.json；新容器实际parent七字段/完整argv已由唯一metadata-only job成功捕获。复用旧已冻结slot wrapper/transport与原正式Trainer，没有更改模型、数据审核或保存恢复门禁。正式输出独立为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run，保留原生恢复checkpoint，测试清理器不覆盖它。

目前仅确认实际入队/worker进程，真实优化步、参数同步、保存和恢复尚待日志/产物验证；不得记作验收通过。未新submit/cancel平台资源或重启容器/worker/监听，停止注释和独立CUDA专项暂停保持。上一Goal回合为progress（新容器实际Running且8worker就绪）；本回合推进正式任务实际启动，原Goal保持active。


## 2026-10-05T13:01:01.223411+00:00 执行顺序：先申请八卡容器

用户最新要求先申请资源。新任务 `t-20261005205851-lqfjg` 已创建且实际平台 Running，空队列，尚无正式模型训练。初步 f011（R26+R24）选择保持；后续在现行数据/来源与完整原生恢复检查下接入训练，剩余消融独立补齐。申请成功不等于联合训练/保存恢复已经验收。配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/eight_gpu_pool_20261005.yml，资源证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005/platform_after_submit.json。


## 2026-10-05 初步版本确定，V7 训练优先

按用户最新要求，初步训练采用 `f011`：R26 链内局部/全局交替加 R24 固定 0.1 残差缩放。最新完整两 seed/120k 结果：f011 AA CE0.802794、f001/R26 0.808560；前者约低 0.71%，全遮蔽准确率差约 0.09 个百分点。该选择用于初步训练，不把未完成 22 项矩阵标为最终选型；context37 正例和重建指标不证明实际生成或正负判别效果。原 A4、链编码、target256/64容量与结束标记保持。

申请独立 c20250601 非闲时八卡资源运行 V7 联合训练，后续消融在本机 A100 上依次执行并与训练并行。训练配置与实际提交/启动证据尚待完成；原八卡已在 2026-10-03T19:24:33Z 终态 Failed/worker exited，不能沿用旧 running 表述。详见 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_INITIAL_TRAINING_20261005.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_INITIAL_TRAINING_20261005.md)。全部原数据、来源和实际恢复门禁及安全停线保持。

## 2026-10-03T09:09:20Z 测试结果持久化改为 metrics-only

用户要求测试仅记录中间结果，不保留 checkpoint/tmp。公共 p01/context/large-batch 测试 core 默认关闭模型序列化，验证点仍记录完整 curve/summary，progress、来源 provenance 和 COMPLETE/SEGMENT 保持；checkpoint_saving=false 写入新 provenance。模型前向、损失、mask、优化器、scheduler、batch、数据和评估计算未改，仅改变结果持久化。显式 --save-checkpoints 才启用保存，metrics-only resume 拒绝；现有 p04 恢复文件已按用户要求删除，因此不能继续沿用旧的测试 checkpoint 接续方案。

当前原 frozen core/runner 保持 SHA；正在运行及排队的冻结测试仍使用原逻辑，新发布的模型文件由外部 10 秒清理器删除，不把后台清理写成 frozen 代码不再序列化。6 项标准库文件控制和一次两步人工 CPU 入口已实际通过，指标/完成记录存在、模型/tmp 为 0；这不是新 GPU 学习效果。原消融结果、方法选择和正式训练门禁不改。政策与证据位于 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/policy.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/policy.json)。

## 2026-10-03T04:32:08.720556+00:00 新500真实证据：汇总CE不替代完整遮蔽与context质量

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA CE | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|---:|
| R20/f000 | 1.179252 | 0.00% | 1.790015 | 41.13% | 1.999002 |
| R24/f010 | 1.105767 | 6.23% | 1.799239 | 40.62% | 1.976865 |
| R26/f001 | 1.063047 | 9.85% | 1.831919 | 40.13% | 1.962080 |
| R24+R26/f011 | 1.043684 | 11.50% | 1.837388 | 39.73% | 1.969006 |

f011两seed汇总AA CE均低于匹配R20，均值优势由250的20.58%收窄到500的11.50%；f001为9.85%、f010为6.23%。全mask AA准确率当前R20最高，f011低1.40个百分点，且全mask CE更高；context子集三候选均值略低，但仅37条正例验证。不能将部分遮蔽收益解释为生成或多链理解已改善，不凭早期汇总选正式模型。两seed无显著性结论，仍8/22 partial/final=false，旧4×8协议独立。

[500概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/v7_large_batch_ablation_overview.png)；原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/report.json（SHAd8d1ae3b9c8f6e33592f5c7478a494ad944826fe61ed4b579878054ef0b7ac94）；[12图及实际渲染归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json)（SHA9d19428788fad9579a9196c1c0053a071ab296c99ad63184013869d9b50a1af4）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json)（SHAcf0571f33cbb9358ae15e7fda91ef744e1496e0dff7c626d2842bdc5a4b6d0a4）。250独立归档保留，500真实endpoint只渲染一次。

只读审阅确认：原冻结核心已有--resume，恢复模型/Adam/scheduler/CPU-CUDA RNG、epoch/position和累计计数；现有pool启动脚本却不能传递到此入口，且outer新attempt必须有自己的COMPLETE。正在准备独立通用接续模块，保留原账本、完整来源/plan/capacity/data检查和同cell持锁；实际票据必须等原job terminal及checkpoint保存边界核实后创建。本回合未读取或散列实际latest.pt、未恢复模型，不能将存在文件当恢复通过。原保存替换latest→追加curve→summary，若硬停致末步不一致，原恢复拒绝保持，不补写证据绕过。

正式选择仍待原完整矩阵与实际证据，原scope/配方/门禁保持。

## 2026-10-03T04:02:11.291493+00:00 R20定义保持，新共同250只支持早期重建比较

用户R20问题已只读复核原五份冻结源码SHA：R20=r20_refined_control、A4布局、no-ESMC随机初始化、75,601,176参数；8层/hidden768/12头/FFN3072，token+role+scope embedding、每链重置RoPE、同链/其他位置可学习bias、八层全局双向注意力、残差系数1。Context序列重建CE0，既有监督关系token参与遮蔽CE；target full/fragment仍256/64槽位并包含<protd>。旧120k完整3750的R20均值0.825971，与新250均值1.565687按轮分别呈现。

| 条件 | 两seed平均AA CE | 相对R20降低 | 全mask AA准确率 | context子集AA CE |
|---|---:|---:|---:|---:|
| R20基线/f000 | 1.565687 | 0.00% | 40.58% | 1.997667 |
| R24/f010 | 1.449769 | 7.40% | 41.04% | 1.972678 |
| R26/f001 | 1.269924 | 18.89% | 40.59% | 2.064495 |
| R24+R26/f011 | 1.243442 | 20.58% | 40.51% | 2.118483 |

这是新组四条件×两seed的早期重建结果，8/22 partial、final=false；另14项仍待运行。组合f011的汇总AA CE最低，但全mask准确率未显示同步改善，且仅37条context验证上的AA CE更高；不能认定生成、多链理解、二分类或生物功能提升，不能据250步选正式模型。两个seed不构成显著性证据；旧micro4×GA8结果独立保留。

真实250报告及12图已逐字节独立留存 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json)；[实际概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png)；[渲染/归档记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render250_20261003.json)（SHA41f067ee719394b508be77c97080b01e06bc367a7d203cb987672e5a1ad981f3）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared250_20261003.json)（SHA514bf2eba5a2ac69603e08e31373a10421690ba740451b260fc5851441863965）。未重绘、未重复模型验证或读取二进制checkpoint。

正式选型/配方仍待完整证据；默认原22消融及正式-after-validation顺序保持。

## 2026-10-03T03:41:39.451832+00:00 正式入口接续保持训练数学，预算不改变消融范围

formal v2只替换旧正式入口未知alias的冷委派链，旧formal prepare/main/ticket/source/data/选择/resume/八slot与后续NCCL/checkpoint全部复用；原611 model driver不变，不需重复其已完成来源门禁。新三项冷控制一次通过，根端实际审阅/两源冻结完成，清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2.json（SHAe92de529db8b77f5c0e083ff115186c1419fbfc6c9a53ba61b386d5fde1bb9df）。public未发布正式入口、formal选型/配方/计划/八票据仍待；没有实际GPU验收。

2026-10-03T03:41:16.404111+00:00固定新组实际8 running/14 pending，首8为200–200/3750步、6400–6400条记录；完整共同评估点仍[0]。真实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/observation_20261003T034116Z.json（SHA102c50223a69406674451099e89a68cd681e2cc061726da4974dad957ba4f14f），没有新组终态、最终架构或完整22图结论。 新32×1实际25→175约8.42–8.57秒/步，比旧4×8完整epoch计时更慢；目前未测padding后形状或具体瓶颈，不能归因某CUDA算子或承诺加大batch加速。完整22与当前资源有约26小时对15.75小时的条件预测冲突，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review.json。默认原完整消融顺序及范围保持，250完成实际评估/保存后再预算；不能静默改正在跑的配方或把首8称完整矩阵。当前没有最终架构、正式自然/增加context配方、horizon或训练段预算决定。

## 2026-10-03T03:14:48.193324+00:00 统一batch完整消融已执行登记，R20基线定义保持

新11条件×seed42/43独立组单次登记22固定ID，2026-10-03T03:10:52.992945+00:00首8实际running、余14 pending。采用已实测且原件严格绑定的micro32×GA1/global32、eval4，120k/3750完整epoch、原模型与既有binding正例关系监督/context序列CE0保持。首批f000/R20、f010/R24、f001/R26、f011组合两seed；组合收益尚未有训练结论，不能预设可相加。实际计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json、完整账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json，登记证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/enqueue_receipt.json；源码与首次控制接续 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_V2_INTEGRATION_20261003.md。训练mask与micro内损失加权因GA改变，必须在新组内配对比较，不拼旧8项。

R20是约7560万参数、no-ESMC随机初始化的A4 grammar/全局双向Transformer；已有链role/scope、每链重置RoPE、同链/其他位置偏置、残差系数1。当前target256/64容量及<protd>仍存在，不能宣称已无固定长度或结束token训练；完整定义和旧epoch比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

正式训练仍未选最终架构或执行真实八rank联合更新/原生恢复。当前正式driver自然父集context约4.65%，筛选子集约20.54%；需明确正式配方后绑定实际来源，不将pilot路径冒充formal-parent。八卡两段固定horizon及新增步数≤5000的原约束保持，时间预算以新组真实步速评估，不能承诺完整正式训练已可在本窗口完成。当前仅将Readonly方案建议作为待决策，不提前宣布natural配方或10000horizon已选。

## 2026-10-03T02:36:47Z 下一统一消融采用已测micro32×GA1

真实v2九格/COMPLETE/pool rc0及40已加载来源根端核实，三个代表在micro32时allocated/reserved均≤85%total，最重reserved61.885/79.151GiB。后续采用统一32×1/global32、eval micro4，模型/已有正例关系目标/context序列CE0/120k完整3750步均保持；不能将容量测量当语料吞吐或学习验证。实际plan/22账本及新的v2消费者尚未落盘/发布；原私有消费者要求旧8更新/6loss，需明确新协议4更新/2loss及真实四原件，不能修改旧原件或放松其余数据/来源/成功/余量条件。根端原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_root_terminal_review.json（SHAb6ae90637728f22bef378e71d0d2c943e50ee74e4ad52165b1f0a66d95970c5b）。

## 2026-10-03T02:27:52Z 原八项完整epoch比较与容量v2单次执行

原120k八项完整3750步，R20对应基线AA CE0.825971，R26对应0.812045（两seed均低、均值低1.69%），CV整段/全局0.816965及延迟融合0.818486；仍缺14格，优先保留R24/R26/f011，不称最终模型或理解/生成效果已选定。实际原报告、派生与边界已写 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

独立36更新容量协议已单次通过原池执行，九格真实passed/rc0；三个micro32均保留15%显存余量。旧p04超时失败/568模型源/数据审核不改，未复用失败job或签造通过件。仅容量从2+6改为2+2，既有架构实验仍3750完整epoch；使用v2结果前必须明确核验新协议的4更新/2loss及原生COMPLETE/真实pool结果，不能冒充旧8更新/6loss通过。完整统一batch22格的source/plan/账本及消费者尚未发布，正式训练未启动。源码/CPU/actual GPU各层分开，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CAPACITY_TIMEOUT_REVISION_20261003.md。

## 2026-10-03T01:50:54Z 容量测量限时反馈与独立有界协议

原九格容量任务真实首格180秒超时、非OOM，完成4更新而原要求8更新；不采纳其部分显存值选batch，不重放原job。准备独立新协议p05_v7_capacity_v2，复用原568冻结及原生profilermodel（不改模型/输入2048/global32/BF16/AdamW/九格），仅改为2warmup+2measure/格、共36更新、每格240秒、总仍1800秒和硬上限100。四更新用于建立Adam状态并测显存/吞吐，不承担模型学习效果判断；架构实验仍3750完整120k epoch与两seed。只有九格实际终态、选档三模型passed/allocated及reserved均≤85%total、COMPLETE与pool成功才采用新档。当前新协议仅源码准备、未发布/入队，原失败原件与此前未部署源码冻结均保留，所有数据审核/安全门禁不改。

原p02八项3750终态已出现，最终原件/图核验中，十四撤下保持；后续统一batch22格仍用于验证R24/R26/组合及CV对照。实际父pool identity/argv原件已取得，仅metadata，不称容量或world8训练通过。正式训练未启动。当前状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，原容量说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_PROFILE_20261002.md。

## 2026-10-03T01:27:30Z 恢复观测v3冻结/来源门禁完成，候选仍待完整epoch

新观测根端源码与首次CPU证据审阅完成，原14顶层函数及训练数学/CLI/门禁/采样/checkpoint回调不改；独立611文件v3冻结，原helper新来源一次真实门禁passed、59.723秒。旧v1/v2与数据证书不变；真实八卡联合更新、optimizer/scheduler/CUDA RNG恢复仍未通过。源码/CPU/来源门禁和未来GPU证据分开，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。

共同3000真实AA CE两seed均值：基线0.841631、R26对应f001 0.826145（低1.84%），CV整段/全局和延迟融合低0.94%/0.90%。R26两个seed均优于匹配基线，但1000/2000的5.22%/2.20%收窄；保留R24/R26及f011组合优先，未选最终赢家。实际3750终态、统一batch完整11条件两seed及必要确认仍待。原报告和判断 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

## 2026-10-03T01:10:58Z 正式恢复采用原生返回点观测，保留原训练数学

只读核查：formal v2强制真实final-parent、没有synthetic入口，旧V4仅准备/CPU证据。选定候选后采用固定horizon的两段真实训练，停止于S再原生恢复至S+Δ，保留新增步数≤5000的原pool限制。为补足内存恢复证据，仅在HF原 _load_optimizer_and_scheduler / _load_rng_state 返回点只读比较完整state_dict与本rank RNG和SHA绑定保存文件；八rank真实观测及续步齐全才生成独立原件。不改变数学、RNG恢复时机、数据跳过、loss/context/已有正例配方，不另开连续训练对照，不宣称后续轨迹等价。

新公共源码已实现，一次必要CPU5/5、27.250秒通过；根端审阅及新v3冻结待完成，已有v1/v2原件不改，实际GPU恢复仍未通过。设计和证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。候选/最终配方未选，新统一batch组未启动；当前交接明确实际Goal active/final-parent现行审核/真实探索在跑，旧blocked保留历史，不修改原目标或门禁。

## 2026-10-03T00:22:42Z R20 对照定义保持，共同2000步证据已归档

本次回答用户的基线问题，按实际冻结源再核实 R20 是随机初始化 no-ESMC、A4 grammar 输入的全局双向 Transformer；链类型/观测范围 embedding、每链 RoPE 和同链/其他位置偏置均已有。它尚未加入固定残差0.1或局部/全局交替，target256/64容量与 `<protd>` 仍存在；context 序列 CE 为0，只有已有标注关系参与原遮蔽监督。没有新增架构或训练目标决策，首轮完整epoch基线 AA CE0.837899 与第二轮共同2000步0.900098分组呈现。

已保留不会被后续绘图覆盖的第二轮共同2000步原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared2000_20261003.json，SHA256 ebde7475b316aac968c9fb1f70666d251516783871d367455ed1a71e0387e26d。历史1000步派生记录仍保留，当前中间判断以2000步为准；完整3750、组合f011及最终架构选择尚未完成。详细定义/比较见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

## 2026-10-03T00:09:34.673530+00:00 共同2000判断与原容器父身份读取接入

根端已核实旧八项共同2000/64000曝光的真实新图与报告：f001/R26两seed token AA CE均值0.880305较同轮基线0.900098低2.20%，CV整段/全局和延迟融合分别低1.60%/1.22%。R26仍是优先候选，但早期1000步的5.22%幅度收窄，完整3750终态、f011组合和统一新batch矩阵未完成，正式架构未选。完整首轮R24/R26结果保持单独比较，不能跨配方混比。真实数值与边界见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md。

为取得已实现formal wrapper必须的实际父pool identity/argv，std-only读取已通过5冷控制和3最小早期entry检查，并只接入原唯一pending容量任务。它不读训练数据、查GPU、创建子进程/监听或预留资源；原profiler/568源和模型目标不改，输出只到pool/logs，避免触发profile run初始目录限制。实际capture尚未发生，容量尚未测量，不能由hook发布推导GPU/训练通过。原正式launcher修正、门禁及停止分支保持；详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md。

## 2026-10-02T23:49:12.122059+00:00 正式八卡训练分段语义与实际调用兼容修正

原pool单job仍≤5000新增更新，不修改已运行资源；正式plan.segment_steps为driver累计stop_after，用现有checkpoint SHA-bound JSON读prior，要求target-prior==八张同配置票据steps，resume与冻结driver实际参数一致。未来formal入口兼容已冻结large→capacity→p02，未知argv原样委派。这些是通用训练调度/恢复整合修正，不改变模型、学习目标或数据门禁。

根端检查到首次5项人工控制与实现共同沿用了compact摘要，而原driver.digest用默认JSON空格。已改为复用原driver摘要，保留plan/intent摘要；另根据实际pool YAML相对script argv，改现场/proc/cwd解析。新7项必要metadata控制通过0.293秒（未重复旧GPU/Trainer验证），三文件独立冻结；当前未发布正式入口/计划/票据，不把人工JSON通过称作native optimizer或world8通过。原件、SHA和冻结路径见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md。

真实pool完整身份及argv仍需原容器实际读取；只允许原唯一capacity启动时std-only捕获，在pool/logs保留独立metadata，不能先污染capacity run目录以致拒绝profile。代码准备中，原profile/driver/transport冻结源不改。真实最终模型/配方选择、联合8rank更新及保存恢复未完成，正式训练未启动。

## 2026-10-02T23:25:23.615132+00:00 新大batch试验的启动条件已实现为代码

独立调用检查已实现并经过5组人工标准库控制；根端源码/日志SHA复核及与真实pool/profiler字段对照完成，未重跑。执行前必须绑定实际唯一capacity任务的ledger/summary/COMPLETE/pool result四原件、完整九格与选档三代表通过/85%headroom、同一568模型冻结、实际统一batch plan及全部22个job ID账本。先完成账本再发布pending；未知alias原argv交回现有capacity→p02。代码和专用控制已独立冻结，清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1.json。

当前只是工程边界通过，没有选档、真实plan/账本或训练发布；不据此解释架构收益或GPU容量。原八项继续且不改micro，原十四撤下保持。模型/目标/数据门禁不改，源码/人工控制证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_ABLATION_20261002.md。

## 2026-10-02T23:19:52Z R20 共同基线的实际结构与边界

根端按首轮实际运行冻结源和 R20 seed42 provenance 核实：`r20_refined_control` 为 A4 输入布局、随机初始化、no-ESMC 的 75,601,176 参数模型；8 层、hidden768、12头、FFN3072、RMSNorm/SiLU门控FFN。输入有 token/role/scope embedding、每链重新编号的 RoPE、每head同链/其他位置两项可学习偏置；全8层全局双向 attention，attention/FFN残差系数1，未加入 R21 head gate、R24固定0.1或R26局部/全局交替。偏置的其他位置包含grammar prefix，不等同于更细分链角色配对表。

target 抽样遮蔽 CE，context 序列不遮蔽/不计序列重建 CE；已有标注关系token按eligible监督，目前配方只含已有binding正例。target full/fragment仍预留256/64内容位置并学习 `<protd>`，真实内容后的无label mask槽仍attended；context按实际长度，额外batch pad不attended。动态batch padding与完全取消固定target容量是两回事。forward没有显式 timestep embedding，不宣称已有该功能。

R24/R26首轮与R20共享同seed共同初始化、数据顺序、micro4×GA8/global32和完整3125步；后续120k新context组f000使用相同基线结构，数据及曝光不同，绝对CE不跨组比较。完整解释与冻结路径：[阶段架构记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md)。本次仅源码/已有metadata核实，无新模型、数据格式、GPU验收或checkpoint；原停线和数据门禁保持。

## 2026-10-02T18:53:05+00:00 当前最终测试范围数据已验收

使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。按用户确认范围最终审核通过、training_ready=true，真实 assert_ready 已通过；保留 6,460,277 条。规则、明确排除项、证书、低样本限制及哈希以[最终记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准。历史版本 pending 不代表此新版本状态。未启动训练。

## 2026-10-02T18:40:43+00:00 开始最终范围收尾

回填既有 alpha CDR3、复核实际 T4 reference、对齐 test-only 门禁。新版本及限定范围见[最终收尾记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)；当前构建运行中，未宣称审核完成。

## 2026-10-02T18:19:18+00:00 alpha CDR3 缺字段追溯完成

8,765 条均来自 VDJdb 衍生 fulllength；上游已有唯一 cdr3a，当前 prepared 转换未保留该标识。可追溯证据、代码原因与待回填状态见[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。未进行新注释或修改数据。

## 2026-10-02T18:15:56+00:00 用户明确排除四项审核要求

原 V7 的 264 条 H3 未确认项及 GDPa1/Specificity/m396 专门保护证据不再作为本轮必需审核或阻断项；不是已审核通过。剩余 TCR 字段、T4 覆盖和训练入口对齐保持。当前范围以[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)与[机器可读修订](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/user_scope_amendment.json)为准。

## 2026-10-02T17:40:43+00:00 r2 观测链修复与复审完成

当前新构建版本为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2；配对误判缺陷已修复并独立验收，整体审核仍未通过。数量、版本哈希、规则与剩余缺口以[r2 完成记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准；未修改训练入口或 ready。

## 2026-10-02T17:24:16+00:00 真实观测链修复 r2 开始重构

按 synthetic_residue_mask 区分真实伙伴与全合成占位链；新规则、版本路径和进度以[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准。13 项边界测试通过，当前运行中；未解除整体审核门禁。

## 2026-10-02T17:05:42+00:00 新版审核暂不通过：真实观测与合成占位链需区分

上一轮导出按角色字段判定配对不充分，导致部分单链记录绕过非配对过滤。当前版本不得作为审核通过数据；真实观测链的判定、全量影响数与其余缺口以[审核报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_AUDIT_20261002.md)为准。未更改数据/门禁/ready。

## 2026-10-02T16:46:51+00:00 测试集配对完全重复版本已完成重构

采用实际测试范围与同一测试记录 CDR3α＋CDR3β 完全匹配规则；非配对记录保留测试范围内旧规则。新增原行号删除台账，原划分及保留行字节不变。导出和独立选择/字节复核已通过，仍未认证；版本布局、实测数量与默认口径依据见[本轮记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_REBUILD_20261002.md)。


## 2026-10-02T16:25:08.444845+00:00 用户目标更新：真实数据比较架构，通过验证后正式训练

当前Goal工具目标为“1. 根据部分真实数据开始确定最终的架构. 2. 验证完毕后设定training ready为true 直接开始跑”，thread仍01a0f245-4559-7b62-9dc2-0917a8ed2267、active；本轮没有create_goal或重置用量。历史18M累计快照保留为历史，当前用量以最新get_goal为准，不把旧快照写回当前Goal。

本轮只读取当前V7清单及审核文件存在性，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/real_data_goal_prerequisite_20261002T162508Z.json。manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段存在性=False、值=None；decontamination=audit_pending，pending=['parent_current_scope_audit_pending', 'updated_manifest_not_covered_by_parent_certificate']。两份当前审核存在性为False/False。仍缺覆盖当前数据版本与完整下游test/query/few-shot范围的真实审核结果；缺审核不代表已经证实泄漏。此前停止的序列注释分支不得恢复、重试、转交或绕过。本轮没有读取真实分片/抽batch、生成或伪造审核、修改training_ready、开始架构训练。

验收含义严格区分：模型/监督工程测试通过不能将数据ready置True；人工合成学习结果不能据以确定最终真实任务架构。当前V4仅CPU与启动暂存验收完成，尚未新GPU执行；原合成计划保留，但不以新增合成验证替代用户当前要求的真实数据比较。正式目标所需的真实数据比较证据及正式训练均未完成。缺失审核是当前真实数据路径的具体外部前置，不是full access或提交权限问题。

上一可见工作回合有实际代码与CPU验证进展；本轮对新目标的只读核实未推进真实数据实验，按新目标记录首次同条件阻断观察（1），尚不将Goal标为blocked或complete。下一可执行条件是取得可核实的当前数据/下游范围审核结果，并另行确认现有工程条件；不得仅手填ready或将用户认可代替审核。所有既有安全停线及禁止容器/监听重启、重复提交要求保持。

## 2026-10-02T16:18:58.824015+00:00 V4监督修复两环境首次验证通过，八卡启动文件已暂存

V4实际生产入口已调用同一supervise_process_observations，仅对来源模块/组/contract/operation绑定的只读group_status超时有界继续观察。原2秒锁等待、30秒健康窗口、启动/总期限与持久中止/退出检查保持；原模型、Trainer和checkpoint函数未改。独立源码静态复核无必须修正项，root完成唯一新控制草稿的必要断言后首次执行。

pllm47.2566秒、worker47.3272秒，各两项通过：真实八CPU子进程在实际2秒状态锁超时后全部仍活，恢复后同一Popen正常退出、八份原协调器退出记录齐全；不可观测时健康时间未刷新。六个拒绝分支覆盖持久中止、非专用错误、无首份有效状态、启动期限、总期限与实际child rc7。真实无首份状态分别31.2912/31.2901秒拒绝，含单次在途锁等待，不能称精确30秒硬实时；启动/总期限使用明确缩短的CPU参数，不声称等过180/1800秒GPU期限。34自有子进程均wait/PGgone，临时目录已删；每环境65份原始元数据/计时文件保留，未创建checkpoint。

根端结果逐值/来源/退出核实：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_supervisor_root_review.json（SHA 9d9288e9de86e5bc2b767a4943c855db0d878093b934553868a8947d7792ac21）。完整545来源与16保护未变，冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/source_freeze_before_CPU.json（SHA2123626cf035d387173b8729fdd4c91596a0cdd29ebf202891d60863759fc7af）。没有重跑旧CPU/500/profile/FULL/Gloo实验。

三份启动文件已暂存但未发布：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/stage_v1/stage_metadata.json；固定新八成员计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/group_launch_plan.json（SHAb5a5a87ab7a64632c33f212f0ce1602580f7df397cc04a6a0f1ec8b5edcbccec），group original_r21_world8_1790957765992952124_ba1f12c5。Root确认旧public保持、dispatcher与冻结源相同；独立只读复核已完成并接受，见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_and_stage_independent_review.json。actual group/host intent/launch gate尚未创建，GPU attempts0。下一步仅在完成当前前置与暂存复核后，于原容器执行一次八卡纯合成Trainer/保存恢复验收；不重新stage/生成ID、不重放V1/V2/V3。

最新实际GPU仍是V3八注册后失败并完成八子进程清理，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_GPU_terminal_root_review.json。CPU修复通过不等于GPU/NCCL/Trainer恢复通过，真实V7小batch/训练仍0，当前审核门禁与停止注释分支/独立CUDA暂停保持。最后平台实际查询 2026-10-02T15:58:02.091075+00:00 原task Running、八worker维护待命、池0/0/97；无新平台提交取消或容器/worker/监听重启。原Goal身份目标累计保留并active，本回合有源码/实际验证进展，未达到最终训练目标；实际Goal工具快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/goal_actual_after_CPU.json。

## 2026-10-02T15:37:05.010795+00:00 V3八卡终态：只读查询超时被上层误升级为中止

V3固定group original_r21_world8_1790953672752409704_de857f5f于15:20:33 UTC实际一次入队，8结果均failed/rc1；8 checked joins、注册、startup领取、rank退出及member cleanup齐全，8子进程actual wait=-15/PGgone/cleanup=true。无Trainer目录或完成训练步/保存恢复证据，checkpoint0，无需删除。原池0/0/97，原资源不重启。V3此次预算已耗尽，源码/计划/原结果保持，不能重放g03；实际发布三文件的前后字节证据保留。

新计时直接确认：heartbeat PID595196/TID595225持锁3.835443757秒，其中commit3.830358904秒。八个超时全是group_status且abort_on_lock_timeout=False，0条mutating timeout，coordinator/abort-request.json不存在；24份coor trace共756行、每文件最多75行，未触及128行上限。首observer timeout完成24216167.42190974→member持久abort24216167.426369432→长心跳释放24216169.18076661→普通coor abort24216169.182679232。成员锁878条，maxhold0.008579805秒/maxwait0.060349118秒，本次无旧成员锁2秒级竞争。八cold日志为空，不能将commit慢唯一归因于fsync、共享存储或GIL。

根端原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_GPU_terminal_root_review.json（SHAc09f71558fa0396a7a10c96634117efdd1cf7fd267a1fb8713a74b25e6399cdd）；独立有限监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_group_bounded_readonly_watch.json（SHA1a68386f9cb2972dd30a85bf9e7abc8b44874b180494cb958d7a98cf1d244967）。545当前源保持，16保护中仅两份已审核发布目标发生计划内变化。观察已结束，无永久监听/待poll进程。最近CLI15:17:02 UTC平台Running；没有Volc提交取消或容器/worker/监听重启。

下一必要修复是通用监督层的异常分类，不改模型或加长2秒锁时限：仅来源绑定、明确来自group_status的只读观测超时可在原30秒freshness/启动与总deadline内继续观察同一child；失败不得刷新健康时间，未取得首份有效状态也必须有界。真实持久abort、mutating/成员锁超时、身份来源错误与异常退出仍失败，暂不可观测不能产生成功。应在新私有版本用生产复用的实际监督循环做一个必要CPU场景及其拒绝分支，不用替代循环/伪GPU intent，不重复已验旧控制。独立只读策略复核确认这个边界；V3失败原件和完整门禁保持。正式V7小batch/训练仍0，双审核仍缺；停止注释分支及独立CUDA暂停不变。原Goal继续active，本执行回合有实际证据进展而未完成最终目标。


## 2026-10-02T15:13:28.540259+00:00 当前状态：V3锁修复的CPU验证通过，八卡启动文件已准备

启动失败已确认的链路是持有成员锁时等待协调器锁，触发2秒超时并连锁中止；首次协调器持锁慢操作仍未知。V3拆开成员/协调器锁、把进程探测移到锁外、保留探测后最终中止检查，父进程每0.25秒检查一次组状态并保留0.05秒退出/期限检查；模型数学、原生训练/保存/恢复函数不变。

两环境首次各3/3必要CPU控制通过：pllm8.1196秒、worker7.7225秒。2.25秒显式慢探测期间，另一状态查询耗时分别0.0365/0.0455秒；协调锁等待期间成员锁获取0.00143/0.00135秒。纯观察约2.001秒超时未写abort，真实变更约2.005秒超时写入持久abort，后续变更拒绝；八CPU进程真实注册/启动/退出成功。22个自有子进程全部rc0/wait/进程组消失，临时目录已删。每环境保留15份原始timing内容和实际命令/日志，未创建checkpoint。控制3验证共享snapshot原语而非伪造GPU入口；这些CPU结果不能称真实GPU/NCCL/Trainer已通过。

根端逐值/原件复核：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/CPU_regression_root_review.json。完整545来源冻结SHA546cec975bfc6fbfac9800ea5db21d073008fecee8ccf3197b80c81dcf6f879a，16份前驱/公共文件未变。新三份启动文件仅暂存于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_v1；固定group=original_r21_world8_1790953672752409704_de857f5f、八g03 jobID位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/group_launch_plan.json，plan SHA234d6ec48d10143074564cf00baca77016c87c33116642f4cf1f4d5de4f3ac5d。根端与独立agent已核对registry仅八别名、runner仅一个路径、dispatcher与冻结源字节相同，记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_independent_review.json。未发布、未入队，实际group目录/host intent/launch gate均未创建。

准备阶段完成证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/preparation_completion.json（SHA0186617b44a5e6e5154a08493fbe8280aff63f5d07d6db98f54d59e72679b08f）。下一步在独立执行阶段复核原8 worker及原学习/profile/来源/恢复前置，发布已审阅暂存字节，仅将现有八固定ID一次入队做synthetic Trainer联调；不要重stage/生成ID、重跑已过CPU/500/FULL/Gloo或重放已耗尽V1/V2。原资源最后CLI14:35:34 UTC Running；没有提交/取消/容器或监听重启。

V7 manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready缺失非True、audit_pending、两份审核仍缺，正式小batch和训练0。缺审核不代表已证实泄漏，也不是权限问题。停止注释分支不恢复/重试/转交/绕过，独立CUDA算子暂停，真实数据门禁不变。原Goal仍active且身份/累计不重置，本回合为实际progress，未完成最终训练目标；Goal实际工具记录见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/goal_actual_after_CPU.json。


## 2026-10-02 V3启动协调修复进行中

本轮成员锁修复已落盘至 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/gpu_worker_bootstrap_v1.py，当前候选SHA `2011bd152bc9ad6f44c3e849141f717da8a0692073a1194139c47ebd0d90269c`；新coordinator的实际来源绑定尚待统一固定，状态为**未执行、未验收、未发布**。这次仅调整快照、assignment读取、abort及cleanup的锁边界：这些路径的协调器状态、退出和abort操作移到成员锁外，实际父子身份、原record_exit、无重复提交、完整来源检查与持久abort保留；完整8份cleanup之后才查询原协调器终态，`None`不等同失败。共享快照helper允许原有显式CPU元数据控制域，不授予GPU、模型或正式数据准入。预算路径为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/execution_budget.json。本次未改模型、损失、Trainer、fixture或训练日程，不代表八卡保存恢复通过、真实语料收益或最终架构选定；V1/V2失败证据和已有单卡验收保留，正式数据审核门禁及停止注释/独立CUDA暂停不变。

仅修复通用进程协调，不改模型数学：成员状态快照取出后再查询协调器，状态观测的进程探测移到协调锁外，父进程组状态查询改为每0.25秒一次并去掉重复查询；退出/总期限仍每0.05秒检查。原心跳、2秒锁超时、持久abort和来源/身份/退出门禁保持。源码位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3；此时尚未测试或发布。新控制只覆盖此次锁竞争故障，已验单卡500步不重复；正式数据审核门禁及停止注释/独立CUDA暂停不变。

## 2026-10-02T14:16:09.255833+00:00 当前状态：V2八卡已终态失败，下一步新版本最小锁修复

截至 2026-10-02T14:14:14.929183+00:00，V2固定共同八卡组 `original_r21_world8_1790948366803074817_36c1a3d5` 已真实终态失败：8个pool result均failed/rc1，组与coordinator均aborted。新版本完成8个checked join、assignment、8个实际子进程注册和8份startup token领取，推进超过V1仅3成员加入的阶段；但没有Trainer目录、已完成训练步数或保存恢复成功证据，不能称八卡DDP/模型训练通过。八份实际子进程receipt均rc=-15、wait/PGgone/cleanup=true，8份rank退出及member cleanup齐全；自有临时checkpoint=0，未删除实际训练恢复状态。

直接错误是锁等待超时。实际父进程PID577559在group_summary持join.lock时等待coordinator.group_status的第二把锁，持join锁2.011217秒，commit仅0.002797秒；随后6个成员锁超时。更早的coordinator持久abort比member abort早约10.90ms。旧coordinator没有持锁计时，最初持锁者/具体慢操作仍未知，不认定死锁或某次I/O是唯一根因，不称完整545来源扫描重回join锁。16份实际timing JSONL及原日志/失败原件保留。

当前原池pending0/running0/results89，八worker新鲜维护待命。平台任务t-20261001033327-fmnql最后实际CLI查询13:54:41 UTC Running；没有新平台提交/取消、容器/worker/监听重启。V2一次执行预算已消耗，当前所有源码、预算、freeze、gate、固定8job IDs及失败原件保持，不自动重试或重放V1/V2；有限监督已真实结束，无观察进程待poll。

原八单卡500人工学习和已完成CPU验收仍保留通过记录；R20/R21/R24/R26已实现，但真实架构选型/理解与生成端下游收益尚未证明。正式V7小batch及训练仍0：training_ready字段缺失/null非True，audit_pending，两份当前数据/下游范围审核均缺。缺审核不是已证明数据泄漏，不伪造证书或降低门禁；停止的序列注释分支不恢复/重试/转交/绕过，独立CUDA算子保持暂停。

下一步在独立的新不可变通用工程修复版本，消除成员锁内等待协调器锁及重复高频状态查询，保留来源/实际host与父子身份/2秒时限/持久abort/退出清理门禁，并针对真实协调器竞争路径做有限必要CPU验证和耗时记录。首次coordinator竞争原因须保留不确定性；消除嵌套锁不能单凭静态推断当通过。不改模型数学或真实数据审核，不增加无关方法波次，不重跑已完成500/profile/FULL/Gloo或原CPU套件。

[根端真实终态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_GPU_group_failure_terminal_root_readonly_review.json)（SHA 2bb59dda1ad9a254362d6fce67b23a3fd80a0dfcbd782156c99d18504a1686ff）；[有限监督原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_group_bounded_readonly_watch.json)；[锁因果及边界复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_failure_lock_remaining_work_independent_readonly_review.json)。Root首轮只读字段误读source_check_state已保留错误原件，依实际JSON修正为source_phase；没有修改原validator/生产源或重试GPU。原Goal 01a0f245-4559-7b62-9dc2-0917a8ed2267仍active，上一准备回合及本执行回合均有实际progress；不新建或重置，实际累计原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/goal_actual_after_V2_terminal.json。

以下带日期内容为保留的历史状态。

## 2026-10-02T13:48:54.596399+00:00 当前决策：先完成共同八卡工程验收，最终架构待选

V2仅修复通用启动事务，模型/Trainer数学及原生保存恢复逻辑未改；新增四项CPU边界每环境首次通过并经独立复核，暂存入口也已复核。八项单卡500合成学习已验，但当前共同八卡实测仍V1启动失败0步，正式语料效果和最终架构选择未完成。下一项为固定新计划的一次有限共同八卡纯合成训练/恢复一致性验收；不是新增方法波次，少量更新不用于架构优劣判定。当前准备阶段不发布/入队，不改预算和原准入门禁。数据双审核、停止注释及独立CUDA暂停保持。

证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_CPU_lock_and_stage_root_acceptance.json；后续计划：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json。以下带日期段落为历史状态。

## 2026-10-02T12:55:34.324017+00:00 当前阶段：单卡学习已验，共同八卡须修复启动协调

截至 2026-10-02T12:54:01.376745+00:00，R20基础对照、R21注意力头门控、R24残差缩放、R26链内局部/全局注意力交替的候选已实现；原八项独立单GPU人工合成实验均完成500步并通过原学习验收。新八卡入口CPU检查在worker/pllm两环境分别4/4通过，已通过的检查不重复。这些证据不证明真实下游提升，最终架构尚未选定。

共同八GPU合成组 `original_r21_world8_1790944282201927112_f554453b` 的唯一V1尝试已终态失败：8个pool result均failed/rc1，直接错误 `Member lock wait expired`，只有3成员加入，assignment/startup未产生，8份member receipt均显示cold child未启动、清理完成。模型训练步数0，原生GPU保存恢复未验证，本次checkpoint目录0。此次失败发生在模型/NCCL启动之前；不撤销已有单卡500步通过记录，也不把源码已实现或入队当共同八卡成功。

只读源码诊断确认锁获取时限为2秒，join临界区内再次核对完整545来源/gate/plan并提交状态；这是需修正的长临界区设计。没有阶段耗时trace，不能认定某次SHA、I/O或调度是唯一根因，也不能仅凭锁文件存在称死锁。下一步保留V1源码、预算、计划、日志和失败原件，在新版本做最小通用协调修复：完整来源核对移至短事务外、两阶段状态绑定防止未核对成员放行、保持启动前完整核对和不可逆abort，并记录真实等待/持锁耗时。只补新增边界所需有限CPU锁竞争/来源变化拒绝/超时清理检查，再经来源复核安排有限纯合成验收；本状态回合没有重试、运行新测试或新入队。

原平台任务`t-20261001033327-fmnql`最后实际查询为12:37 UTC Running；12:54 UTC远端八worker metadata均standby_load/maintenance_only，心跳年龄1.39–2.62秒。池pending0/running0/results81。原八卡容器/worker/监听没有重启，没有Volc submit/cancel；pool成员失败不等于平台八卡容器失败。

V7 `immune_v7_mhc_sequence_v1` manifest SHA仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失（null，非True）、audit_pending，audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明泄漏；正式V7小batch和训练尚未启动，仍须当前manifest的真实数据审核及完整下游范围证据。当前只做通用工程与人工合成验收；安全停止的序列注释分支不得恢复/重试/转交/绕过，独立CUDA算子保持暂停。

[本次根端终态核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_overall_progress_after_group_failure_readonly.json)；[有限观察原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/group_bounded_watch_v1.json)；[原七项500验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original7_terminal_root_v1/original7_terminal_root_acceptance.json)；[入口CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/world8_new_boundary_controls_v1/cpu_boundary_root_acceptance.json)。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267` 仍active，身份、目标及累计不重置；实际工具快照见[Goal原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/goal_actual_user_status_after_group_failure.json)。

以下带日期内容保留为历史记录。

## 2026-10-02T12:10:40.759435+00:00 五个原模型八GPU接入源码固定版本，实际验收尚未执行

原模型共同八GPU Trainer所需5个私有源码已全部交付，根端11:58 UTC仅做AST/接口和物理SHA静态核对，固定完整545文件来源，15旧保护文件未变。冻结SHA 1b5cfd23a1e9c3cbacb9f30b2fd9dd949fbdcf2d4063962eeac8780ee2311bba。没有导入执行新源码、运行新CPU控制、GPU训练、stage/发布/入队或生成launch gate；代码完成不代表八卡验收通过。原模型/Trainer原生训练和保存方法保持，完整Adam、scheduler、八份真实RNG及恢复后实际状态比较已实现于新源码，等待实际工程验收。

原native optimizer实际构造一次后从真实参数身份记录分组，共同metadata一致再经同一v2 factory重挂callback；train开始确认复用该optimizer。保持horizon10000/LR1e-4/warm100/cosine和原PilotStop1000；额外短工程stop不改训练方法或学习标准。完整checkpoint检查使用实际模型spec、vocab list及八份真实HF RNG，不使用单RNG占位。

截至2026-10-02T12:05:11.695172+00:00，原七项单GPU人工合成实验全部500/500、实际pool result.state=succeeded/returncode=0，inner与outer COMPLETE均存在；原池pending0/running0/results73。CE R21/profile43原500此前已通过唯一原checker及独立复核，不重复；其余七项刚取得终态，尚未执行各自原verify_diagnostic(500,require_learning=True)，不能记为验收通过。8个原worker在12:06 UTC独立只读核实均standby_load、maintenance_only=True，原资源不重启。

下一步在既有授权范围内，先对七项真实终态各执行一次原500学习验收和结果/来源/退出清理复核（CE已验不重复），完成新来源每环境最多4项必要入口边界控制及独立复核；全部原来源/profile/学习/GPU/完整恢复前置、实际八worker可用及新启动检查满足后，再通过现有pool运行一次原模型共同八GPU纯合成Trainer保存恢复工程验收。连续3步与保存2步后恢复至累计3步用于状态一致性，不代表1000/5000训练阶段或架构效果；不新增架构波次或重复已通过CPU/Gloo/numeric验收。

V7仍为immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready字段缺失（null，非True），decontamination.status=audit_pending，当前audit_certificate.json和downstream_scope_certificate.json均缺。缺审核不等于已证明数据泄漏，也不是操作权限问题；正式资源绑定仍缺权威完整下游范围/输入清单及绑定格式。未读取真实分片/抽真实小batch/重建审核selection/启动真实训练，不伪造审核、手改ready或降低门禁。

停止的序列注释分支不恢复、重试、转交agent或绕过；独立CUDA算子继续暂停。原八卡容器/作业/worker/监听2213164不重启，不重复Volc提交；有限observer90075已自然结束，不重新启动。临时验证checkpoint仅在真实退出与独立证据复核后清理自有文件，保留原失败、来源、原件和实际训练恢复状态。

[当前状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_status_after_source_freeze_readonly.json)；[根端静态核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_delivery_root_static_review.json)；[完整来源冻结](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_freeze_before_CPU.json)；[工程实现与接续说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_ORIGINAL_TRAINER_EIGHT_GPU_20261002.md)。

以下带各自时间的记录为历史状态。

## 2026-10-02T11:35:31.718386+00:00 必要原模型八GPU训练器接入源码准备

只读独立决策已完成，root重新核实9旧source SHA一致：V7正式scope/双审核资源缺口不阻止单独纯合成的实际模型world8 Trainer工程；旧CPU预算及通过记录不能直接授权GPU。报告/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_GPU_Trainer_remaining_readonly_decision.json（42de8eb7fb6e72ca96996eaded4fe23ee8f4f799a3897c6cdb6511f2e53b3f5c）。

新私有来源目录/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1，有限预算/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/execution_budget.json（4397c4fec564113d9871523760c8e4df5531426c2d715d9ba7254b30d8637d5d）：最多5个必要源码文件，复用原assembly/模型/Trainer/日程。A负责完整原模型checkpoint validator，B负责实际GPU bootstrap及冷rank/member supervisor，C负责唯一原模型Trainer driver，root负责固定pool dispatcher/stage metadata。本轮仅实现准备，禁止测试、GPU执行、生产发布和新入队；不得改旧模型或immutable包。

后续工程验收固定真实原CE R21 Transformer、source/profile43、人工学习42、world8/micro32/GA1/global256、BF16、horizon10000/LR1e-4/warm100/cosine。比较连续3步与原native checkpoint2→累计3步，每rank共6次实际optimizer updates，原PilotStop1000保留但另有明确短工程stop/save observer；不得称1000/5000阶段完成或架构收益。新完整validator须实际16类必需文件、8原生RNG、原模型权重/Adam/scheduler/config/vocab list/source/datafixture完整检查，不造rng_state.pth占位或hash-only代替。未来只允许一个8成员group attempt/1800秒/峰allocated70GiB；原七任务真实终态、原来源/profile/500学习/GPU/完整恢复条件、新源码闭合/独立评审与有限新控制通过后，才另做实际启动核验。

V7 metadata/审核缺口仍在，真实训练0，不读取真实shard/小batch或执行审核selection。停止注释分支不恢复、重试、换agent或绕过；独立CUDA算子暂停；原八卡容器/作业/worker/监听不重启、不重复平台提交。此前CE500及两环境CPU完成验收不重复。

以下为较早记录。


## 2026-10-02T11:03:50.603902+00:00 当前原CE500学习验收已闭合

CE R21/profile43原500于10:56:27 UTC succeeded/rc0，原verify_diagnostic(500,require_learning=True)唯一root只读调用及独立stdlib复核均通过，trace25..500共20点。学习seed42：四mask平均loss4.0881584711→0.1847930732，AA准确率6.38465%→92.65651%，terminal100%；完全遮蔽mask100 AA71.54613%、loss0.7071843，平均不能称全遮蔽/生成准确率。39当前source物理SHA、9实际输入及62独立引用哈希、日志/双COMPLETE/实际子退出清理均一致，checkpoint目录0。

截至2026-10-02T11:03:50.603902+00:00，原池pending=0/running=7/results=66；CE已完成验收，其余七项最新步数为375、375、375、375、375、375、375/500，仍需真实终态后各自一次原验收。GPU1回原维护状态，没有重启worker/作业/监听或新增GPU任务。

该证据只证既有当前来源人工单GPU学习，不证真实V7/生物下游效果、单模型八GPU-DDP或正式恢复；profile43不是新学习seed。不同来源、不同步数及相同learningseed42的各profile不能当独立多seed比较或最终选型。原CPU Trainer/通信验收保持完成，不重复；正式资源接入及实际共同GPU保存恢复仍未完成。

本回合重新仅读V7清单元数据：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、training_ready字段缺失、audit_pending，两份当前审核均缺。合法当前完整下游资源范围清单及绑定格式缺失，原正式gate无法正确定稿；不造清单/证书、手改ready、降低门禁。真实shard/小batch/审核selection/训练0；停止注释不恢复/重试/转交/绕过、独立CUDA算子暂停、原资源不重启或重复提交。

当前验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json；原只读调用尝试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_attempt.json；independent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_independent_review.json；最新原任务快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_after_CE500_independent_closure.json；V7 metadata /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/V7_metadata_after_original_CE500.json。

## 以下为较早记录

## 2026-10-02T10:34:05.741083+00:00 候选已实现；新增通信验收通过，最终架构仍未选定

本轮新增八进程通信/清理控制两环境首次各4/4通过：worker5.747秒、pllm7.253秒。每环境8实际parent和8实际CPU/Gloo cold rank全部rc0/wait/自有进程组已清，rank0持有FD直接交TCPStore；倒序member正确映射rank0..7、实际store/all-gather8身份一致、缺员不可逆abort和未登记失败child rc2清理均通过。26旧文件及9冻结source不变，临时目录0、checkpoint0，不重复已完成numeric Trainer。root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

此证据只覆盖新增CPU通信及词表列表元数据格式校验；原生产Trainer类组装仅静态复审完成，未实际构造/训练；新词表callback未通过实际native保存恢复。实际单模型8GPU-DDP完整保存恢复、正式外部资源参数全调用链和最终架构选型仍未完成。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool pending0/running8/results65，八项各自pool result及内外COMPLETE均缺。八项是独立单GPU，不是单模型八GPU训练；不同来源/步数和同learning seed不能直接选优。

下一步收尾原八项500实际终态，各自只执行一次原verify_diagnostic(500,require_learning=True)及原始结果/来源/退出清理核验。原八卡全部空闲后，在单独冻结当前来源与有限预算下，补必要通用单模型GPU组及原Trainer保存恢复证据；不新增架构波次，不重跑旧通过验收。真实数据训练须当前有效审核及原全部data/scope/source/selection/learning/profile/GPU/full-resume条件满足。

V7为immune_v7_mhc_sequence_v1，train6,849,243/valid52,572，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94。training_ready字段缺失、audit_pending；audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明数据泄漏；用户认可或操作权限不能替代当前manifest及新增下游范围审核。真实分片/小batch/语料训练0。

停止序列注释不恢复、重试、换agent或绕过；独立CUDA算子暂停。原容器/worker/监听2213164不重启、无重复Volc提交，本轮新增GPU/平台submit/cancel/restart为0；不造证书/手改ready/降低门禁。

[最新原任务与V7快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_and_V7_readonly_snapshot.json)；[本轮根端复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_review.json)；[新增工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md)。

## 以下为较早记录

## 2026-10-02T09:54:10.064871+00:00 候选实现完成，通用保存恢复已验，正式八GPU接入仍未完成

候选R20/R21/R24/R26代码已实现，最终架构选型未完成。生产单模型八GPU入口、原pool八rank组调度、正式外部审核资源参数在preflight/selection/build_trainer/execute/resume的贯通及新来源实际GPU完整恢复仍未完成；原controller launch拒绝实际执行，旧pilot/metrics仍限world1、生产checkpoint仍单份rng_state.pth，冻结包十资源目录缺失。新CPU模块未接入生产包，不能将通过声明移给新GPU来源。

两环境各一次实际八CPU/Gloo numeric Trainer主case及独立复核全部通过：worker19.747秒、pllm22.008秒，连续3步与原生checkpoint2→3恢复比较，参数最大误差0、Adam最大1.862645149230957e-9、scheduler0；16 rank原生RNG及下一批IDs一致。384组source/mask/rank-phase、2256指标由stdlib独立重算，误差0；独立float32损失oracle最大1.375491803790041e-8，低于原1e-6。16 child均rc0/wait/进程组清理完成，自有numeric临时checkpoint及二进制剩余0，18旧保护文件保持。

此验收仅证明纯数字CPU fixture的新增训练器工程，不证明免疫生产Trainer、真实GPU-DDP、1000→5000续训、架构生物效果或数据准入。已清理二进制未被独立review重读，证据来自执行时保留的原值/哈希及冻结source；保留全部原失败/raw/source/log与实际训练恢复状态。

原八项500步合成学习：CE350/500，其余七项225/500，池pending=0/running=8/results=65；09:50只读平台查询Running，09:51远程poll-gated状态及原trace无终态。八项均独立单GPU，不能称单模型八GPU训练。

下一步只收尾当前八项500的实际终态并按各自原verify_diagnostic(500,require_learning=True)核实，再完成必要通用生产调用链及八rank冷入口；不新增架构波次、不重复已通过CPU/旧模型验收、不启动无关验证。现八GPU仍被原任务占用，任何必要同模型GPU组验收等待全部原worker空闲并满足新来源原门禁；真实数据训练需有效当前审核和全部前置满足。

当前V7审核/ready缺失，真实训练0；本阶段不追加CV/架构波次。保持停止注释/独立CUDA暂停及原数据、来源、selection、学习与完整恢复门禁。用户条件授权已存在，无需重复申请操作权限。

[本轮根端验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/actual_CPU_conformance_root_acceptance.json)；[最新数据与原任务状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/user_training_readiness_after_strong_reviews.json)；[必要八卡接入缺口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EIGHT_GPU_TRAINING_INTEGRATION_GAPS_20261002.md)。


## 2026-10-02T09:02:06.944393+00:00 本轮只收尾现有候选与必要训练工程

候选实现与最终选型分别记录。V7本版审核尚缺，不视作已证实有问题；不执行真实小batch或语料训练。实际八rank numeric Trainer源码复审中，修正CPU parent异常清理边界后才冻结/首次验收；不新增架构波次或GPU任务，不撤销原准入。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/user_v7_readiness_current.json。


## 2026-10-02T08:44:04.665526+00:00 新Trainer分布式接入范围

新factory真正继承原compute_loss/evaluation_loop并执行world8评价、8rank callback/save/restore；不修改架构或旧source，不以纯metadata当实际DDP。numeric CPU conformance只测新增训练器边界，真实模型质量与原500/当前数据审核仍另验。

[有限来源和真实Trainer计划](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_RANK_TRAINER_INTEGRATION_20261002.md)。

## 2026-10-02T08:35:16.052458+00:00 八rank通用接口已验，下一步实际Trainer接入

两个私有接口及四个CPU控制两环境首次各4/4通过，实际八CPU rank屏障/退出和错误binding、锁超时不可继续均已验；world8/global256、唯一指标归约及8RNG/完整16文件续训元数据的新增边界通过。此结果只支持新通用接口，不支持真实DDP/模型效果或保存恢复已完成；未改旧模型/损失/immutable包或生产入口。下一必要项是实际Trainer/metrics/save/cold rank调用链与正式资源参数接入，保持原所有审核/source/selection/500/GPU/full-resume门禁。当前8项人工500未终态，CE200/其余50，不以不同步数或同learning42的两profile副本选架构。

[有限工程与实际证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_RANK_INTEGRATION_20261002.md)。


## 2026-10-02T08:13:53.151504+00:00 八卡通用工程的必要实现范围

本阶段不新增模型架构，先实现正式单模型八卡训练必需的私有stdlib协调与checkpoint/指标/批量合同；真实8子进程CPU屏障与故障控制只验新接口，不替代真实Trainer/DDP/完整恢复。新来源尚未进入immutable生产包，旧模型/损失/GPU/500证据不改也不迁移。真实审核/selection/数据读取仍不执行，所有既有门禁和安全停线保持。

[有限来源与检查预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_RANK_INTEGRATION_20261002.md)。

## 2026-10-02T08:09:25.072907+00:00 架构实现与正式训练条件的当前决策

R20基础对照、R21注意力头门控、R24固定残差缩放、R26链内局部/全局注意力交替均已实现；模型候选实现完成不代表最终选型及单模型八卡训练接入完成。当前八项500合成学习仍运行，尚无最终验收；不能用合成成绩定真实数据上的最终架构。本阶段收尾现有500、正式外部审核资源接入和必要分布式训练工程，不追加新架构探索波次、不重复已通过检查。V7 MHC当前审核/明确ready缺失，真实训练未开始；全部原数据-scope-selection-resume门禁、安全停线和独立CUDA暂停保持。用户已条件授权完成后训练，无额外权限确认，但当前条件尚未满足。

[本次就绪状态与剩余项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TRAINING_READINESS_20261002.md)。

## 2026-10-02T03:00:55.226271+00:00 唯一失败项数值观测准备

本阶段只补原失败tiny的GPU数值观察，不改变R20/R21/R24/R26设计、原梯度阈值或500学习/完整恢复/数据准入。新观测即使完整也不是GPU接受或架构选优。 [源码diff](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/staging/source_delta_review.json)；[metadata](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/user_readiness_check.json)。


## 2026-10-02T02:49:04.121405+00:00 必要单项GPU数值观测整合开始

上一Goal回合属progress：两次必要CPU观察及七项当前GPU原始核验完成。本轮只补唯一失败R21seed43 tiny的GPU数值证据，固定原source/profile/seed/BF16/micro1GA8、原120秒tiny与总180秒硬监督预算；单项新alias，原七项不重跑，原断言/门禁不改。计划新driver复用原CPU三数值helper、事后CPU叶子求导；新stdlib dispatcher监督fresh -I子进程并实际退出清理，输出只是观测完成、全部准入false。现阶段仅预算与来源冻结，源码未写/控制未跑/未部署/未入队，真实训练0。

原池实际{'pending': 0, 'running': 0, 'results': 50}、八worker维护、原监听存活；原worker动态读取registry可接新alias，不指定物理GPU2，记录实际GPU/UUID。唯一既有live改动为early入口literal和registry追加一项，新dispatcher冻结副本；旧d02/guard/worker/release_map、全部runtime/private/CPUhelper不改。注释停止/数据门禁/CUDA暂停保持。 [预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget.json)；[来源保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before.json)；[资源原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/readiness_before_engineering.json)。


## 2026-10-02T02:39:30.314663+00:00 当前七项完整原始结果核验完成

根端 2026-10-02T02:39:30.314663+00:00 已对七项当前source/seed结果逐一用原verify_completion→verify_candidate作fresh worker-Python只读核验，全部通过原source/fullmap/profile/seed、独立CPU评价重算、3formal窗及正常退出/清理绑定。此轮GPU结果7/8工程验收通过，R21seed43 tiny梯度失败未解决。全池0/0/50、原8worker维护，平台缓存Running、监听PID2213164存活。四候选实现/新synthetic接入完成，不称整体选型或真实效果已完成。

| 方法 | seed | GPU | 本批当前来源工程结果 | 三窗人工loss |
|---|---:|---:|---|---|
| R20 | 42 | 0 | 独立原始复核通过 | 3.4509 → 3.4485 → 3.3803 |
| R20 | 43 | 5 | 独立原始复核通过 | 3.5974 → 3.6034 → 3.5327 |
| R21 | 42 | 3 | 独立原始复核通过 | 3.5924 → 3.5982 → 3.4992 |
| R21 | 43 | 2 | tiny梯度对照失败，保留 | 未进入formal |
| R24 | 42 | 7 | 独立原始复核通过 | 3.5335 → 3.5421 → 3.4421 |
| R24 | 43 | 4 | 独立原始复核通过 | 3.8274 → 3.8285 → 3.7500 |
| R26 | 42 | 6 | 独立原始复核通过 | 3.6179 → 3.6042 → 3.5345 |
| R26 | 43 | 1 | 独立原始复核通过 | 3.5053 → 3.5112 → 3.4577 |

七项各3窗/24microsteps/768人工记录，LR依次0/约1e-6/约2e-6，后两窗参数确实更新；完成四mask及原清理，不当500步学习证明或真实效果比较。CPU R21seed43两环境各一次约25秒均未复现；26参数梯度误差0，8loss最大差7.15e-7、RNG一致，16组CE叶子最大误差2.98e-8。只能缩小假设，不能确定GPU根因、撤销失败或放宽标准。

指定版本仍immune_v7_mhc_sequence_v1、manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、audit_pending，缺明确ready与两份当前版本/范围报告；既有清理记录保留，不称已证实泄漏。真实小batch/初步1000→5000比较/最终训练均0。下一独立必要工程是隔离记录原失败tiny的GPU数值，再按证据决定修复；不重复七项已通过验收。原500学习、完整恢复、数据门禁保持，停止注释不恢复、独立CUDA暂停，无容器/worker/监听重启或平台重复提交。

[完整原件与来源/资源/数据状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/final_results_and_readiness.json)；[CPU数值与4710B临时清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/cpu_results_and_cleanup.json)；[修正driver原件/预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/terminal_review_v2_before_execution.json)；[实际工具返回](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/execution_tool_returns.json)。本轮无验证checkpoint，两个独占库tmp共4710B和两空HF目录已清理；14个0字节自有控制文件删除，失败日志/源码/raw/proof及真实恢复状态保留。首轮root工具AttributeError仅属核对driver，v2修正未改原验收/GPU结果。原Goal实际active、身份/累计工具计量保留，不标complete。


## 2026-10-02T02:34:39.929269+00:00 独立只读复核driver接口修正

七项GPU原作业succeeded保持，但root只读复核driver首轮臆用了不存在execution.ROOT，七个日志均在原verify_completion前AttributeError，当前root完整核验0。首轮source/log/空stdout保留；新v2仅使用原模块实际__file__路径及manifestSHA校验，不改生产/原验收或门禁。先只复核R20seed42已有原件，通过才逐项核对其余六项；无模型/GPU重跑、CPUtiny不重复。 [首轮失败](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/terminal_review_first_attempt_failure.json)；[修正及预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/terminal_review_v2_before_execution.json)。


## 2026-10-02T02:32:11.628607+00:00 八项终态与CPU未复現

本批八项GPU已终态，原池{'pending': 0, 'running': 0, 'results': 50}；七项succeeded、R21seed43 failed，七项root只读强核验尚待完成，不提前写全部通过。两环境各一次CPU隔离诊断均original_tiny_returned(25.13/25.02秒)，26参数梯度误差0、scale1/L2=0，8组loss最大差7.15e-7、RNG一致，16组CE叶子最大误差2.98e-8；CPU没有复现GPU失败，不定GPU根因、不撤销失败。atexit库tmp清理误拒原日志保留，root仅逐文件删除两独占2355B文件及空目录，总4710B，另两个自有空HF目录删除，checkpoint0；冻结源码/17保护文件保持。

[CPU数值/清理原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/cpu_results_and_cleanup.json)；[本批终态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/live_observation_20261002T023211Z.json)；[七项原始结果只读核验预算/源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/terminal_review_before_execution.json)。V7当前审核门禁及ready/双报告缺保持，真实语料0；停止注释、独立CUDA暂停、不重启/重复提交保持。


## 2026-10-02T02:25:18.839510+00:00 必要失败诊断源码冻结

独立R21seed43 tiny CPU诊断源已冻结并由root/B只读复审，SHA0f24916d193e4cbcdf563219c9ea474614fa07d99c7d55166d85fd9201c8fe45；原17项受保护文件保持。按原预算即将每环境一次/180秒执行，仅CPU数值观察，GPU/data/training接受均false，无新GPUjob，不更改原断言。 [源码冻结与静态复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/source_review_before_execution.json)，[源码](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/diagnose_r21_seed43_cpu.py)。


## 2026-10-02T02:23:33.681117+00:00 七项首次非零学习率参数更新

根端 2026-10-02T02:23:33.681117+00:00 直接读取七项formal原件，当前已完成窗口数[2, 2, 2, 2, 2, 2, 2]。第2窗实际learning_rate=1.0000000000000002e-6，finite/nonzero gradients=true、parameter_delta_l2范围0.00831123–0.00850931，实际参数更新成立；各项第2窗完成时累计16microsteps/512人工记录。仍7running/1failed/0完整验收，第三窗/评估/退出清理待完成。R21seed43 tiny失败保留，独立CPU诊断源码已落盘，最终静态复审待完成、尚未执行。

[实际窗口原件摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/live_observation_20261002T022333Z.json)。真实训练0；V7审核门禁、活跃源冻结、停止注释/CUDA暂停及禁止重启或重复提交保持。记录工具首次因对LR作浮点精确比较拒绝，未写MD/状态；仅修正记录判读，不改训练标准、不重跑模型。


## 2026-10-02T02:18:49.402651+00:00 首窗预热与当前完整验收边界

根端 2026-10-02T02:18:49.402651+00:00 直接核实原池 pending0/running7/results43；新八项仍7运行、1失败、0完整终态。七项独立 full CE/GA/RNG 核对已完成；当前 formal 已落盘窗口数为 [1, 1, 1, 1, 1, 1, 1]，首窗均 learning_rate=0、parameter_delta_l2=0，仅说明预热/梯度和 AdamW 状态正常，不能称三个已更新窗口。后续非零学习率和完整评价、退出/清理仍待观察。R21seed43 tiny 失败保留，独立CPU诊断源码准备中、尚未执行。

V7 MHC metadata仍audit_pending，manifest SHA未变，缺明确training_ready及当前版本/范围两份检查报告；既有清理记录保留，不能据此称数据已证实泄漏。四候选实现及synthetic接入已完成，当前source/seed完整验收尚未完成，真实训练入口未发布，真实语料训练0。原500学习、完整恢复、数据门禁全部保持；注释停止分支不恢复，独立CUDA算子暂停；容器/worker/监听不重启、不重复提交，全部活跃v2来源保持。

[本次实际观察](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/live_observation_20261002T021849Z.json)；[失败诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_SEED43_GRADIENT_DIAGNOSIS_20261002.md)。

## 2026-10-02T02:10:26.502679+00:00 已失败阶段的必要隔离诊断准备

上一Goal回合归类progress：实际运行版本发布/八项synthetic启动及新梯度失败改变了下一行动。本轮只针对已失败R21seed43的tiny micro1×GA8阶段准备独立CPU诊断，每环境一次、180秒上限；原d16/L2/H2/MLP32、length64和原数值阈值保持。将导出8micro loss/RNG、逐参数误差和同captured logits两种CE的dL/dlogits，不认证GPU或架构通过。v2/live/私有来源全部冻结，当前诊断未执行、新GPU0、真实语料0。

原七项继续，最新agent只读02:09:38 UTC确认七项gradient_oracle完成并进入formal窗口，但整体完成0/8；独立原件以[实际快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/live_observation_20261002T020918Z.json)及现有runs的阶段文件为准。R21seed43保留失败，不盲重试或放宽标准；V7缺有效ready与双报告，原500学习/完整恢复/数据门禁、注释停线/CUDA暂停保持。

[诊断预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/execution_budget.json)、[受保护来源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/protected_before.json)。


## 2026-10-02T02:02:27.596571+00:00 新seed实际失败及当前边界

最新实际状态为7运行、1失败、0/8完整验收通过。R21 seed43/GPU2在lifetime_preflight独立CE/GA梯度核对报wte.weight不一致；profile正常退出，release子进程rc1、runner rc2，父进程记录进程清理完成，GPU2恢复维护。错误时对象弱引用存活，不把它写作成功阶段清理，也不凭此认定持续泄漏。其余7项仍gradient_oracle，真实训练0。

根因尚未确认，原断言未保存逐参数误差统计；下一步是隔离通用数值/随机流诊断。/root/architecture_keep_or_change只读分析，/root/candidate_gpu_watch只读监督；全部v2活跃源码/私有来源/运行包保持冻结，不热改、重试、放宽阈值或重启。V7仍缺有效ready及双报告，原500学习、完整恢复、数据门禁与安全停线保持，独立CUDA算子暂停，停止注释分支不恢复。

[实际失败复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/r21_seed43_failure_review.json)、[最新逐项状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/observation_20261002T020227Z.json)。

## 2026-10-02T01:57:11.789606+00:00 当前首轮架构和验证边界

四候选R20/R21/R24/R26架构实现保持，首轮仍四方法×seed42/43；新seed运行接入完成，八项人工Trainer验证在原八卡分别运行，0/8完整终态。仅工程验证，不评价正式语料的理解/生成收益，不选最终赢家或扩大G02。当前源实际profile后检查原formal/oracle/lifetime/评价/清理，三profile更新与三formal窗口分开计，原500学习及完整恢复门禁保留。正式V7训练仍缺有效数据审核，不启动真实小batch。

[新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_cpu_results.json)、[四实际运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_software_build.json)、[容器接入变更](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_publication/publication.json)、[八项任务与目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/launch.json)、[领取状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/observation_20261002T015550Z.json)、[V7数据状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/v7_seed_runtime_readiness_20261002.json)。

## 2026-10-02T01:49:20.360499+00:00 当前实现边界

四R候选架构实现保持，首轮仍R20/R21/R24/R26×seed42/43。新执行整合定向CPU两环境各42/42完成；不构成新源GPU或正式语料效果证据，运行包/容器接入及规定学习诊断仍未完成。暂不扩大模型改动或将G02前置。总3600秒含当前源实际profile，原数据与500步学习/完整恢复门禁不降低。

[新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_cpu_results.json)、[当前数据与资源原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/v7_seed_runtime_readiness_20261002.json)。

## 2026-10-02T01:40:13.606889+00:00 当前种子版本人工Trainer整合

上一Goal回合有实际进展：八身份软件接口CPU两环境各13/13完成及MD/source记录保持。本轮继续独立必要工程，三agent分别写新seed-aware人工Trainer producer、新runtime package/entry、独立extension dispatcher/guard/stage；root负责整合、预算、实际检查与MD。当前源码写入中，未冻结/构建/执行新checks/部署/入队。新的四方法×seed42/43共8人工验收须实际定点micro32三更新profile、原完整GA/oracle/评估/退出清理及精确seed/helper/source绑定；旧39源profile不重标，原500学习门禁独立保留未完成。

只读01:32:35 UTC：原八卡pending0/running0/results42、8worker维护无job，平台缓存01:32:10 Running、监听PID2213164存活。指定V7数据仍按01:29 metadata观察缺明确ready和双报告；不读取shard/真实小batch，不恢复停止的注释。无平台提交/重启，独立CUDA算子暂停。原Goal actual active，保留身份及实际计量，上一回合归类progress、不复用旧阻断计数。

首轮四候选集合不扩张、不改目标/采样/骨干；人工profile三更新和formal三窗各自记数，不能当模型效果提升证据。

[执行前预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_execution_budget.json)、[22项受保护来源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_protected_before.json)。

## 2026-10-02T01:30:12.523373+00:00 八身份输入整合CPU完成，正式执行仍待验

八身份软件请求接口完成，pllm与worker对应Python各13/13新增定向CPU检查首轮通过；实际核对四v4包的manifest/fullmap/helper、八个显式seed请求、软件1000→5000续接一致性及缺执行准入时无副作用拒绝。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_seed_launcher_v1.py SHA867427c73dd7c2b16e8f419d5e22d1b1735aad54d832a13316e2ff2f0980ea24。独立静态复审1项收尾SHA位置问题已修正，冻结前writer变更使旧SHA记录拒绝，未将过期复审写成通过；原件保留。受保护18项源码/manifest SHA保持，旧core28/execution11/8次完整导入/旧Trainer验收未重跑。两pytest临时根未创建，临时文件删除0，checkpoint创建/删除0；必要日志/JUnit/冻结与复审记录保留。 [新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_cpu_results.json)。

本次只完成软件请求接口；完整八身份dispatcher/registry/entry接入、seed-aware runtime/producer与当前源/seed的GPU及规定profile/diagnostic证据仍待完成。未部署/注册/入队/运行新GPU/真实训练。原八卡01:29:00 UTC pending0/running0/results42、8worker维护无job；平台缓存01:28:52 Running、监听PID2213164存活。指定immune_v7_mhc_sequence_v1 manifest SHA未变，无明确training_ready字段及两份当前版本/范围报告。旧V7已有清理；当前缺绑定不能解释为已证实残留泄漏。独立CUDA暂停、停止注释分支与原数据门禁全部保持；原Goal actual active及身份/计量保留。 [资源及V7原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_final_resource.json)。

本轮没有新增模型结构或改变loss/采样；四候选集合与两seed共同对照保持。软件接口通过不决定候选胜负，不表示真实理解/生成效果已提升。

下一必要工程：私有正式入口/registry/dispatcher和新seed-aware runtime verifier/人工Trainer producer；随后只获取受影响的当前source/seed证据。旧profile/500结果按原来源保留，不替换SHA、不删原检查。有效data/selection和完整checkpoint恢复是实际训练前置；团队数据审核未闭环时不抽取真实小batch、不启动正式训练。首轮仍R20/R21/R24/R26×seed42/43共8独立单卡比较，G02为条件P2。

## 2026-10-02T01:18:38.651326+00:00 八卡训练完成条件核实

用户条件授权架构完成后使用指定V7启动原八卡；当前指定版本仍为immune_v7_mhc_sequence_v1，manifest SHA未变，无明确training_ready字段、audit_certificate.json和downstream_scope_certificate.json。旧V7已有清理记录；缺当前MHC/新增范围绑定不等于已经证明残留泄漏。本轮仅metadata/软件核实，无shard/注释/真实小batch。四个R候选与旧synthetic Trainer已验，v4软件8/8导入已验；新seed执行入口、当前来源GPU证据与数据准入仍未完成。启动最小私有八身份controller工程，writer /root/architecture_validation_strategy，root负责定向CPU、MD与清理；受保护18项SHA已冻结，预算先行，尚未执行新checks。原八卡01:17:56 UTC pending0/running0/results42、8worker维护无job，平台缓存Running，监听PID2213164存活。未submit/cancel/restart/入队；独立CUDA暂停及停止注释分支保持。原Goal工具active/身份计量保持，不把有独立工程误记为全部外部阻断。 [状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_readiness_observation.json)。

既定P1保持R20/R21/R24/R26×seed42/43共8独立单卡任务；四候选实现完成不等于当前多seed版本完整执行链完成。G02有条件且非P1前置。本轮不新增模型结构、不按synthetic选赢家。

## 2026-10-02T01:07:48.674852+00:00 多seed来源整合验收完成

v4源码隔离导入两环境共8/8通过，每次54项目模块、8seed软件身份；临时库2355B实际清理。新源GPU/容器执行接入与数据准入尚未完成，真实训练未启动。 [实际v4验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_package_v4_validation.json)。四个固定v4软件包保留全部source-only falseflags，无runtime准入manifest/数据alias/worker注册。既定P1仍是四候选×seed42/43共八独立单卡比较；G02非前置，不因人工/源码验收选优。原core/execution SHA保持，旧GPU/profile来源不移给新seed源码。缺报告时不抽取真实小batch、不训练、不恢复停止的注释分支；继续通用执行整合，独立CUDA算子暂停。

## 2026-10-02T00:26:41.616168+00:00 导入边界v2修正进行中

四个v1源码副本已构建、source-package CPU各23/23通过；实际fresh导入首轮两环境仅R20失败，原因是guard误拦第三方临时模板/无参数激活对象，R21/24/26未跑。失败原件保留；精准库预热及自有tmp修正进行中，新v2不覆盖v1，未通过完整导入/GPU/真实训练。 [原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_import_attempt1_failure.json)。只允许准确已核对的第三方无参数兼容对象及自有临时库模板，项目模型构造/训练/CUDA/data/source-write仍禁；全部GPU/data/execution准入标记false，不借此降低真实门禁。

## 2026-10-02T00:09:37.510228+00:00 seed执行层来源与证明绑定

私有执行层seed/source绑定新增CPU两环境各11/11通过，旧GPU schema与跨seed证明拒绝；原gate-first/500诊断/完整resume/phase保持。源码包与隔离导入实现中，未部署/入队/新GPU/真实训练，数据门禁及停线保持。 [验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/execution_cpu_results.json)；新schema为seed_bound.v2，不声明旧记录已覆盖新源或seed43。新GPU证明producer和正式controller仍未完成，source-only包不是准入。

## 2026-10-02T00:05:41.621649+00:00 P1种子身份设计同步

禁用P1设计新增8个明确seed42/43身份及4个拟source-only包路径，CPU核心各28项通过；不等于worker注册/新源GPU证明，production_ready、execution_enabled、eligible、enqueued全部false。 [配置原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/decision_matrix_seed_binding.json)；[具体配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。保持原目标/数据/资源安全停线。

## 2026-10-02T00:01:13.670226+00:00 多随机种子核心CPU完成、纯源码包开始

必要seed核心私有修复完成：pllm与worker Python各28/28定向CPU检查通过（AST隔离+tiny Torch构造/RNG，0训练更新）；14受保护SHA保持，22自有控制文件已清理，checkpoint=0。继续构建明确source-only新来源包和隔离导入工程，未部署/入队/运行新GPU；指定V7缺ready及两份检查报告，真实训练未启动。原八卡0/0/42与监听保持，Goal active，全部停线保持。 [本轮验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/cpu_results.json)、[多seed工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_MULTI_SEED_PILOT_ENGINEERING_20261001.md)。新增来源包只作未发布软件副本，execution_enabled=false；不复制旧运行manifest，不生成准入pointer、不修改既有GPU/profile证明或数据门禁。

## 2026-10-01T23:48:36.005298+00:00 必要多随机种子工程启动

Goal工具实际返回active，目标更新为完成架构探索并在容器启动八卡初步训练；保留返回身份/计量及旧历史，不调用create_goal、不复用旧阻断计数。独立复核发现P1四方法×seed42/43的入口实际写死42，开始必要私有seed工程；G02属于条件P2，不是P1前置。指定V7数据状态仍未通过，原八卡0/0/42；无真实训练/重启/提交，已过验收不重复，全部停线保持。 [当前Goal与必要工程原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/implementation_started.json)、[详细工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_MULTI_SEED_PILOT_ENGINEERING_20261001.md)。当前有实际工程可推进，Goal active且未完成；不把旧“无独立工程”结论用于新发现。

## 2026-10-01T23:41:25.476114+00:00 V7与八卡启动条件核实

用户倾向使用指定V7并已条件授权架构完成后启动原八卡。当前核实：目标为immune_v7_mhc_sequence_v1（train 6,849,243 / valid 52,572），缺明确ready和两份检查报告；四个R候选实现及synthetic Trainer GPU验收完成，私有训练接入未部署，G02正式联合模型未适配。原池0/0/42、八worker维护、监听存活；真实训练未启动。仅metadata/原件/源码核对与MD记录，无重复验收、注释恢复、门禁变更、任务操作或新checkpoint；原Goal blocked保持。 [当前版本、原始验收与资源核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/v7_eight_gpu_readiness_20261001T233803Z.json)。用户启动授权已记录，无需重复申请权限；本条未将数据可用意见写成审核通过，也未启动真实训练。下方带日期内容保留为历史。

本回合是当前实现状态核实，未新增架构决策。四个R候选编码完成；G02等联合方案仍为待正式适配的条件研究项，不能把所有探索方案统称为已完成。已有synthetic验收不重复；当前主要剩余条件是有效数据绑定、正式接入发布与实际小batch验收。

## 2026-10-01T12:17:53.844266+00:00 四候选接入私有CPU检查完成

四候选正式训练接入的私有源码与新增CPU拒绝/身份检查完成，pllm/worker各24/24通过；1项静态来源缺口已修正。原R20 12函数AST/字节、6live源码、4runtime pins保持。清理30个自有控制文件/1118字节和10别名，验证checkpoint=0。未部署/入队/实际stage或准入执行，不代表架构效果；MHC无ready字段且双证书缺，真实小batch仍未启动。 [本次状态、SHA与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)、[实际CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/validation.json)、[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/cleanup.json)。 维持保留R20/R21与R24单项比较/R26成本对照的既有决策；收到有效数据审核和新版本绑定后再作原八卡实验，不因拒绝检查通过改选模型。

## 2026-10-01T12:01:52.464326+00:00 四候选正式训练接入准备

R21/R24/R26已具备各自正式API，当前补齐独立外部执行身份、来源/验收/续训绑定；4项私有源码准备进行中，CPU尚未执行，未部署/入队。原R20门禁和已验runtime保持，指定MHC缺双证书及ready绑定，真实小batch未运行。[本轮状态与边界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)。

## 2026-10-01T11:47:52.931165+00:00 三方架构决策完成

三个用户授权subagent只读决策已完成：保留R20/A4主干，先单独比较R21/R24并以R26作成本对照；联合任务G02末端adapter为条件候选，G01降为读出/优化控制，硬瓶颈和EMA暂缓。8卡4方法×2seed及后续联合组设计已记录，未启动实验。原指定数据缺双证书/ready绑定，真实小batch未运行；原Goal与资源、门禁、注释停线和CUDA暂停保持。

[决策与具体测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)、[三方审查摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)、[禁用决策矩阵YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。新增发现：G01当前任务仿射可吸收到通用原型独立Linear头，不能预称主干任务特化；G04/G05按role而非任意段ID隔离且关系头存在池化旁路；正式R20输出untied但尚无G00式联合分类头。审核范围计划与旧执行scope还需一致的新版本绑定，本轮不改gate、不签证。以下带时间条目是阶段历史。

## 2026-10-01T11:29:06.399847+00:00 当前GPU工程准备与正式数据选择

通用GPU引擎CPU验收完成：两环境各44个唯一案例分三阶段覆盖，未完整重跑最终44项；修正4项工程边界，固定15文件源码包与8份禁用YAML核对通过。清理50个自有CPU状态文件，验证checkpoint=0。现有8卡执行端尚未接入G入口，新增GPU任务0。用户指定immune_v7_mhc_sequence_v1仅metadata核实，缺双证书且无training_ready=true，真实小batch未启动。

[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。原Goal保持active；保留原注释停线、真实数据审核与独立CUDA暂停条件。正式架构比较优先使用审核通过的指定训练数据小batch及实际模型输入/目标/评价；人工整数仅作工程控制，不替代正式效果。下一步完成原8卡执行接入、独立结果复核与自有续接状态清理工具，不重跑已通过检查。

## 2026-10-01T11:15:42+00:00 去重范围按用户决定更正

后续仅保护实际测试/query 记录，不自动保护 benchmark train/valid/support；混合池按实际评分角色界定。旧全支持池规则作为历史记录保留，新决定优先。OTS 的单链近似排除不能称为配对重复；当前仅完成范围核算，未重导数据。详见 [测试集保护范围与 OTS 行级核算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/DATA_BENCHMARK_DEDUP_SCOPE_20261001.md)。

## 2026-10-01 10:30 UTC 人工GPU连续500步采用明确分段协议

现有worker单job人工预算≤100，采用5个独立G段各真实100步、完整状态接续到500，profile额外更新单计；不借R/D身份，不填假data_dir，不把500藏在100预算。共同global128/seed42/LR3e-4、整程500/25warmup日程保持，明确共同clip norm1.0。短暂内部续接状态纳入产物记录，最终核验后清理；常规checkpoint_save=false不代表分段期间无状态文件。[协议与实现进度](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)。当前新入口实现中、接续CPU待验，未发布/入队；旧8原型CPU验收及live版本不变，真实语料未准入。

## 2026-10-01T09:37:59+00:00 G00–G07实现与CPU验收完成

8个独立通用整数联合原型已实现：共享理解/生成主干、任务末端norm、末端适配层、4个全局register、主干4/16个register瓶颈、自身低噪声EMA对齐及同计算零权重对照。真实d192/L4/H6、从头初始化；两环境各35个唯一语义案例完成覆盖，8配置各执行3优化窗口/9微批/24人工记录，并完成原始评价独立重算。详细实现、参数量、失败修正及边界见[原型验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CV_JOINT_PROTOTYPE_20261001.md)，[验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/validation.json)。

本次决策：保留训练FP32 CE及原数值阈值，只将评价CE汇总改为从实际FP32 logits以FP64累计，修正首轮worker汇总差异；仅复查受影响2项，不重复已通过的33项。G04/G05仅限制主干跨角色attention，关系头仍拼接两角色池化特征；不宣称所有关系信息经过register。人工随机流/记录ID隔离不作为内容严格去重证明；给定长度的原型不验证EOS/停止能力。

实际CPU执行global8/micro3/固定LR3e-4与规划global128/warmup25/500步不同，只验功能与统计。8份方法YAML保持execution_enabled=false；下一步是固定通用GPU入口、原池接入和显存档位测量，然后按计划观察联合能力曲线，不按三步结果排名。新GPU任务0，真实语料训练未启动，原数据审核与停线全部保持。下方带时间条目是当时状态。

## 2026-10-01T08:48:23.170929+00:00 通用联合能力原型实现开始

开始实现已规划G00–G07的独立通用整数对象原型：关系标签为两随机段的相对整数变化，目标段为二者的数学合成；理解端目标完全遮蔽，生成端按四档mask预测。复用已验证record-normalized CE，固定共享基础参数初始化，记录teacher和额外模块成本。当前实现进行中，CPU/GPU新方法检查尚未完成；真实数据审核与停线要求保持。[研究与配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)。

## 2026-10-01T08:38:15.022797+00:00 私有训练接入支持更新完成

当前支持源码下的R20训练接入准备已完成；两个Python环境各9项新兼容/拒绝检查通过，原launcher全部函数及候选校验函数保持。14个自有临时文件已清理，checkpoint=0。[具体更新、测试和未完成项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_SUPPORT_REBASE_20261001.md)、[原始验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/validation.json)。当前仅私有准备，未部署；R21/R24/R26外部训练接入及有效数据审核仍待完成。真实训练尚未启动，新CV方法仍待实现。以下日期记录为当时状态。

## 2026-10-01T08:20:56.541520+00:00 当前方法判断与验收边界

R20、R21、R24、R26均已完成各自完整synthetic Trainer GPU工程验收；新增三项[终态原始结果复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_terminal_review.json)通过。短工程检查不作为架构收益排名，历史500步人工结果的局限仍保留。新CV联合能力研究的8个方案处于“文献/配置已完成、原型未实现、CPU/GPU未运行”；下一步按[CV研究和8份配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)做独立通用整数对象原型。真实数据与最终训练未准入，原安全停线不变。

当前进展以[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)为准，下方为历史判断；不得用历史“待验收”状态覆盖本次实测，也不得把计划写成已完成。

## 2026-10-01T08:10:49.754641+00:00 CV联合能力探索记录点

CV联合理解/生成研究新增原文和作者代码核读；7份固定源码快照、8份实验YAML规划已写入。新方法实现尚未完成，CPU/GPU未运行，配置未启用；现有R21/R24/R26仍在原8卡合成验证。 [研究与证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)；[8份配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。

## 2026-10-01T07:48:49.897115+00:00 扩展CV联合理解与生成研究

保留现有R21/R24/R26验证并行运行。用户授权扩大架构研究，先对照既有CV/序列方法文档筛选未覆盖机制，再用人工多对象任务检验理解/生成两个指标及关系依赖，避免只看去噪loss。尚未新增架构实现或宣称收益；[本轮记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CV_UNIFIED_RELATIONAL_EXPLORATION_20261001.md)。

## 2026-10-01 06:46 UTC 三候选运行包实际构建与CPU隔离通过

R21/R24/R26同Trainer所需源实现及CPU验收已完成，三固定包实际构建并通过两环境隔离导入；父524和原R20 v3数学/容差/预算保持。下一步仅完成stdlib运输路由后跑原完整GPU工程验收，尚未新增候选GPU结果；真实1000→累计5000比较仍按原计划及数据准入执行。[根端源/CPU复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/root_source_review.json)、[三候选验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINER_GPU_PLAN_20261001.md)。

## 2026-10-01 06:22 UTC 继续候选GPU验收和训练整合

R20 v3 actual GPU/根端强复审已通过；继续原计划R21/R24/R26同Trainer验收。新固定三包528文件/父524不变，复用existing model variant的profile-synthetic/steps1以兼容旧worker，不重启GPU池、不新增算子或方法。39历史任务无该组合冲突。当前仅私有实现/CPU准备，未发布/build/入队；数据仍training_ready=false、双证书缺失、真实训练未启动。[具体版本、工程标准与剩余条件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINER_GPU_PLAN_20261001.md)。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:37 UTC T20 v3私有整合复审与正式窗口进度

T20 v3私有bridge已完成，两环境原48项软件检查通过（20.88/23.50秒），自有basetemp及测试checkpoint已清理。固定绑定runtime v3/cc13036f…08294fe和future launcher manifest的runtime_root/runtime_manifest_sha256；根端AST复审只有reviewed_admission/load_entry增加严格版本绑定，launch、validate_transport、bound_inputs及原数据/续训门禁未变。[私有验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[根端复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)。source-map另有23项/环境通过，旧28项实际map仍7878dc5c…54f77；尚未发布、实际stage/apply、pointer或入队。

v3 GPU3已完成第1个正式窗口（8micro/256人工记录），LR0/delta0符合固定warmup；后2窗口、四mask评估与parent退出/强完成核验待完成，整体尚未通过。2026-10-01 05:37 UTC只读真实数据manifest仍601db781…ebfce/training_ready=false、audit_certificate/downstream_scope_certificate均缺；未读shard/序列或恢复此前停止分支，最终训练未启动。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC v3运行与证据边界

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

## 2026-10-01 04:44 UTC v3私有工程准备

v3 canonical已确定，root固定路径/guard/typed必要CPU两环境各108项通过；初次basetemp设置失败保留，修正父目录后复测，两个自有fixture目录2626文件已清理。实际tiny CPU对象弱引用全释放；PyTorch2.8原生cuBLAS workspace是待GPU验证的具体候选，producer正补释放前后allocated观测及固定120秒tiny preflight，原full工作量/数值/显存标准不变。仍未发布/构建/入队v3、T20未部署、最终训练未启动。[v3私有定位与必要验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_LIFETIME_V3.md)。

## 2026-10-01 04:24 UTC v2工程验收失败与定位

v2实际梯度累积核对通过，但清理后GPU张量仍驻留，正式三窗口未运行；完整GPU验收未通过。v3只在新私有副本定位生命周期，尚无发布/构建/入队，不改变模型架构、工作量、目标或数值/显存标准。T20未部署，真实候选比较与最终训练未启动。 [整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

## 2026-10-01 04:07 UTC 新预算版本已实际执行

v2已通过必要CPU复审、构建固定源码包并在原八卡GPU7执行新D01；三正式窗口加独立一窗oracle的工作量与数值标准不变，仅按v1实测修正时限并补失败统计。尚无GPU通过证据或架构收益结论。T20新私有版已绑定v2，两环境各48项通过，未部署；数据双证书未齐，真实四候选1000→累计5000及最终预训练尚未启动。[实际记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。

## 2026-10-01 03:26 UTC D01阶段时限失败

实际v1在oracle701.6219秒后因预设240秒预算失败，未进入正式三窗口。保留旧runtime及失败证据；新不可变v2事前采用oracle1200/formal1800/evaluation600、总体3600秒，工作量和全部数值标准不变。修订是依据实测的工程预算调整，不是架构收益、数值验收通过或真实训练。T20桥接私有48项CPU/环境已通过，仍需新版本绑定、实际GPU通过和原数据证书。[整合与失败原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

## 2026-10-01 03:18 UTC 预训练整合与实际 GPU 验收

当前目标收敛到完成预训练整合并启动最终训练。保留原no-ESMC/scratch主干、现有attention和R20/R21/R24/R26比较；D00算子支线按用户要求保持暂停。修正后的Trainer已封装为526文件不可变runtime，在现有八卡GPU5实际执行三个完整窗口加独立梯度核对，尚无通过记录；短检查只验实现，不据此选择架构。T20实际训练桥接正在私有分支实现，第一版仅覆盖R20，不把旧人工500的架构证据追认为新Trainer的GPU验收。[当前整合和剩余条件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

真实数据仍training_ready=false、缺双证书。待原准入及各候选同版本验收通过，按四候选各1000→选定后累计5000与成本/分层曲线确定阶段架构，再启动最终训练；不能声称正式训练已经开始。验证记录进入MD；验证checkpoint清理，实际恢复依赖与最终必要模型保留。

## 2026-09-30T23:31:26+00:00 实际累积窗口验收与半程判读

250步对照R20重新领先总体loss及conditional full-AA，R26仍接近；继续完整500步，当前无普遍提升证据。新增方法库见[自条件/置信度/重新遮蔽](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DENOISING_METHOD_EXPANSION.md)，只有阅读与独立设计，未部署或归入已有实验成绩。

[每记录目标的真实框架验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_LOSS_FRAMEWORK.md)完成两环境各14项、各22次SGD更新。原生token数与实际record分母分离，预取窗口上限规范化也是适配的一部分；全部参数梯度/更新与完整窗口oracle一致。证明限于固定人工输入、CPU、Linear+SGD，不覆盖实际模型的随机mask/BF16/AdamW或checkpoint恢复，不改变现有loss。未来目标对照必须采用相同窗口规则，避免混入尾窗读取差异。

[下一版本协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)新增当前目标下的microbatch一致性排名条件；各方法仍各自测最大可用档，若档位不同则先补独立权重控制。CUDA探针两环境各59项CPU通过，实际kernel/梯度/显存证据仍待当前500完成后获取；没有凭CPU通过切换后端。

## 2026-09-30T23:06:15+00:00 评估v2、训练权重与下一版本协议

本轮继续扩展[去噪方法调研](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DENOISING_METHOD_EXPANSION.md)：Analog Bits/SED自身预测反馈、ReMDM采样器、RemeDi置信度双流，核读原文相关章节并冻结两套作者代码的4份文本。自条件不必依赖ESMC，但连续扩散迁移、额外前向成本、预训练初始化和训练代码缺失均有边界；三项保持未实现候选，不与当前骨干初筛混改。

R20–R27同主干八卡500步诊断已到200；175→200又有排序变化，继续观察完整曲线，尚未选定优胜架构。[分层证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_FINDINGS.md)。既有视觉/序列架构调研继续作为候选库，不能用少量人工步数给所有方法定性。

评估改为显式区分AA/end/relation的CE与准确率，并新增END precision、AA误报和仅full-mask的首次END/missing/长度误差。真实END位置的召回不能代替正确停止；当前无条件人工输入的首次END理论上限1/3，不可把END recall100%称为长度已学会。新评估与准入适配器90项CPU检查分别在两环境通过，尚未部署；旧曲线不混入新协议。[定义与验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)。

新候选自然采样与等来源采样的曝光差距已用整数索引核验，温度配额只是建议；独立每记录loss原型46项CPU测试证明micro分组不变性，但改变了现有token加权目标，且HF/Accelerate额外缩放尚未集成验证。这两个方法必须与架构分开消融，不能悄然一起切换。[来源曝光](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_SOURCE_EXPOSURE_PLAN.md) · [loss原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_NORMALIZED_LOSS.md)。

[下一版本协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)明确先完成当前500、冻结源码，再验证真实CUDA后端/梯度和新评估，再在数据双证书满足时执行1000→累计5000。当前YAML无执行器、候选与microbatch未定、真实数据仍不可训练；不能把软件原型或计划视为已执行的实验。

## 2026-09-30T22:15:16+00:00 监督驱动的隔离评估与下一版本候选

R20–R27维持LLaDA270相同d768/L8/H12/MLP3072主干与蛋白小词表，人工500步继续；100步排序仍变化，不提前选定架构。新[评估原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)将source macro、token micro、四mask平均及AA/end分开，修正回调/RNG与batch聚合；两环境各26CPU测试通过，未热接入运行实验。

[注意力后端审阅](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ATTENTION_BACKEND_REVIEW.md)给出math回退强推断和38项CPU数学/梯度验收；下一独立release优先实测memory-efficient的实际CUDA kernel、梯度、速度与显存，不能把CPU Flex成立视为GPU已获益。[噪声/目标数学诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_NOISE_OBJECTIVE_DIAGNOSTIC.md)证明共享t下全mask约1/(N+1)，并发现当前microbatch分组权重差异；后续按记录归一化、显式全mask曝光、END权重须各自与对照分离，不能混同backend变更或声称简单1/t即复现原ELBO。

[独立清理候选](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)已完成字节/索引验收但未认证。自动安全检查中断的序列注释分支已停止，原覆盖未决项继续pending；不得将本轮软件验收解释为语料训练许可。

## 2026-09-30 下游评测完整性修复

按已确认方案先修正来源、报告和复算能力，保留历史参考协议；不清洗预训练库、不重切下游数据、不新增严格训练轨。AB–抗原 binding 的划分统计支撑不纳入本次修复。方法边界与 CPU 验收见 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)；阶段与 endpoint 的完成状态分开，见 [独立完整性报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。

2026-09-30 V7构建口径汇总：继承七来源v6配方，经旧下游范围过滤、精确语义去重、受体家族划分和共同布局检查导出r2，再执行MHC序列优先更新；最新版本共6,901,815条，MHC阶段不删行。V7数据版本与V3模型/布局版本分开，CDR3观测片段不补造成完整受体。新增UpTCR/支持查询池审计已完成并发现重叠，拟清理尚未应用，264条benchmark H3覆盖仍pending；本次仅归纳既有实现，不改变规则或宣称正式训练已获准。来源表、算法和状态见 [V7整体构建说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#v7-construction-overview-20260930)。

## 2026-09-30 已实施MHC原序列优先规则

用户已明确决定：提供MHC序列时保留并作为训练条件；仅有HLA名称且能唯一查表时补34位pseudo。已产出`immune_v6_mhc_sequence_v1`和后续v3使用的`immune_v7_mhc_sequence_v1`，覆盖下方“尚不采纳新配方”的历史状态。恢复长MHC/B2M并保留链边界，以metadata和现有scope区分pseudo/完整链，不改骨干、词表或TCR目标。v7新增pseudo220条、恢复长链10,529条；v6分别220/41,039。73项测试及实际loader/collator、tiny模型CPU前向/反向通过；不是已执行正式训练或旧checkpoint已获得新版训练效果。

pilot生成器默认新v7，任务/输出名携带数据版本，既有数据/下游范围审核门禁保留。格式、未解析记录、parent依赖和v6既有1条alpha画布溢出见 [数据更新记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#2026-09-30-mhc序列优先数据更新已完成)。

2026-09-30 MHC表示核查：v6预训练的全部带MHC记录已使用34位pseudo，历史全长MHC+B2M来源在摄入时亦查表转换；v7 r2沿用此表示。现有grammar可装载一般MHC序列，但不能据此声称no-ESMC150000已训练全长MHC输入；已有全长原始序列可以保留，是否作为新输入配方属于独立选择。本轮未转换数据或变更架构。UpTCR的34位MHCseq可沿既有MHC条件块适配，不必反推唯一等位基因/全长蛋白；当前已运行αβp轨仍不含HLA。具体数量、来源和部分X/映射差异见 [MHC表示审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#2026-09-30-当前-foundation-预训练-mhc-表示实查)。v3默认`sequence_scope=full`仅为软件默认值，当前pseudo缺少专用scope，不能解释成完整HLA；后续若引入混合表示需显式记录其定义，不在本轮修改。

2026-09-30 UpTCR首轮执行已锁定no-ESMC150000，冻结骨干+既有MLP，输入完整FVα/FVβ+peptide无HLA；原valid选模，不再从train切验证集。单seed0，三套作者划分原样保留。参数与状态见 [任务入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-binding)。仅本模型探针训练，不重新训练论文baseline，不改骨干架构。

## 2026-09-30 UpTCR binding 原发布协议增量

用户最终决定将UpTCR作为当前foundation model另一项binding下游基准，完全沿用作者发布train/valid/test及标签，不清洗、去重或重划分；baseline直接引用论文 [P]，不重训。覆盖此前自建清洗划分建议。执行约定及比较备注见 [T1 UpTCR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-binding)。当前仅完成范围/协议登记，Ours checkpoint及适配方式执行前锁定，尚无本任务结果，不改变模型架构。

## 2026-09-25 TCR实验统一文档入口

本轮结果已汇入 [TCR_DESIGN_EVALUATION.md](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/TCR_DESIGN_EVALUATION.md)，并在flow_antibody_tcr README/rebuttal链接。保留所有逐样本结果和历史更正；论文表不再添加log Pgen均值列。AAR为用户提供原表；新增Validity/Pgen来自全部采样，binding为每目标首条50条。IMP历史raw-reference72%与对称处理0%明确分开。

## 2026-09-25 TCR IMP 汇总

50个目标全部配对，严格pred_dG<ref_dG；相同重建/pack/FastRelax的reference为0/50（0%），沿用AB历史raw reference比较为36/50（72%）。后者含不对称后处理贡献，不解释为结合成功率。新增imp_per_target.csv（100行，两口径）与imp_summary.csv（2行）；[报告](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax/IMP_REPORT.md)。

## 2026-09-25 TCR energy 更正结果解释

CDR侧链重建、PackRotamers及FastRelax2使IgGM2界面均值降至-35.97 REU，巨大raw排斥与闭环惩罚大幅消除；匹配AF3同处理reference为-56.65 REU。依旧不将优化后能量解释为原始侧链质量或实验亲和力。采用两组对称后处理作主比较；不采用生成优化/reference原始的36/50作为主成功率。审计与方法：[报告](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax/REPORT.md)。

## 2026-09-25 TCR energy 口径更正

AB主流程为 PackRotamers+CDR-local FastRelax，非仅chi minimization。TCR补做同类流程，并额外对两组相同CDR位置去除原侧链（包括CB）后重建，避免保留生成的异常侧链内部几何。原始AB主表GT为raw，TCR新主比较采用reference同样重建/pack/relax，另保留raw-reference比较供协议诊断，不混淆两口径。新流程沿用 eval_imp_all_baselines.py 的CDR+默认10Å邻域selector/task factory；FastRelax2，CDR bb和邻域chi；ref2015，AB对其余pMHC。结果目录：/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax。

## 2026-09-25 TCR Rosetta 能量评估方法

ref2015 对现有 IgGM2 生成坐标及同范围 AF3 reference crop 打分，interface=AB 对其余 pMHC 链。分别 raw 与固定骨架/刚体的 all-chi LBFGS 最小化（tol0.001,max_iter200），InterfaceAnalyzer 不做结合态/分离态 repack。不混用完整 pMHC 总分，不把 raw 与经过优化的结构直接横比，不把 REU 解释为 kcal/mol 或结合概率。结果显示严重 fa_rep/pro_close 惩罚；不以 ANARCI 序列编号成功证明结构质量。当前未做 FastRelax 或独立重新折叠。方法、限制及结果：[报告](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_energy/REPORT.md)。

## 2026-09-25 TCR validity 评估口径

仅当前 50 目标，primary 各方法每目标第一条；全部样本另表。ANARCI 默认全部链类型、human/mouse、IMGT、bitscore80，独立编号。基础检查包含标准 AA、模板长度与框架、单域可编号、编号重构、C23/C104/F或W118、非空 CDR；严格检查额外要求 A/B 类型。δ 最佳匹配单独标记，不等同生物学无效。全长覆盖、CDR 边界、重复、低复杂度及 Pgen 分开报告；不证明表达/折叠/配对/结合。全体 junction 与独立编号相符。脚本：[evaluate_validity.py](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_validity.py)；结果：[报告](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/validity/REPORT.md)。

## 2026-09-22 Equal-source training with full source coverage

- 2026-09-23 implementation correction: “source” in this objective means the **manifest shard group**, not the row's provenance label. `oas_paired`/`ots_paired` are row provenance, and `tcr_papers` rows can share `tcr_native` provenance with a separate training group. Runtime `source_group` / batch `source_groups` carry manifest membership through collation without rewriting provenance or prepared artifacts. The Trainer fails explicitly on missing/unknown group identifiers. This fixes the first 2×4 c20250601 launch failing before its first update; the seven-source objective and coverage schedule are unchanged. Validation and resubmission: [process log](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md).
- User requests equal training contribution from all sources, without truncating the larger datasets. Implemented as an explicit `--source_balanced_training True` opt-in in [the training entry](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py); both current v2 launch YAMLs enable it. The default remains false for legacy invocations. Already-submitted platform commands do not gain a new CLI flag from local YAML edits; no jobs were submitted/cancelled for this change.
- “Batch” means one **global optimizer batch** (ranks × local microbatch × gradient accumulation). The current local batch has 4 rows and cannot contain all 7 sources. Keep its memory footprint and the 256-row nominal effective batch: source counts are 36/37 before overflow drops; **loss contributions are exactly 1/7 each**, not merely approximately equal sampling probabilities.
- [FullCoverageSourceSampler](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/balanced_sampling.py) interleaves sources round-robin. Each source uses its own shuffled, without-replacement permutation and starts another permutation only after exhaustion. Cursors continue across coverage-epoch boundaries. No source cap, shortest-source truncation, or replacement-based weighted random sampler is used. Streams are deterministic from the seed, source, permutation cycle and resumed epoch/offset.
- Objective for an optimizer batch is `L = (1/S) * sum_s mean_{i in source s}(masked-token-mean CE_i)`. Row means prevent longer chains from dominating source weight. Optional focal/token/chain weights remain inside each row's reconstruction objective. The trainer gathers **retained row counts after collation** over all ranks and accumulation microbatches; each row receives `world_size * grad_accum / (S * retained_count_s)`, before Trainer accumulation normalization and DDP/FSDP averaging. A missing source fails explicitly instead of silently changing the objective. Relation auxiliary losses, length-grouped sampling, split/dispatch batches, DataParallel, and `ignore_data_skip=True` are rejected in this mode.
- A first complete coverage epoch offers every prepared row to the collator. Existing v2 canvas/grammar overflow exclusions remain: unrenderable rows are counted but cannot train at the current length limits. It is not a promise to use every rejected raw row. Small sources necessarily repeat more often. Subsequent epochs continue the same per-source passes, rather than resetting partially consumed permutations.
- v6 train source counts: OAS 2,485,433; OTS 2,094,218; ASD 225,913; TRAIT 11,687; TCR native 52,817; papers 444,621; repertoire 2,000,000. For effective batch 256, the first full coverage needs **67,962 updates / 17,398,272 sample presentations**. Smaller `max_steps` fails before training; early interruption still requires resume to finish coverage. Both 200k and 1M budgets are sufficient.
- Checkpoints record `source_sampling.json` with mode, source order/counts, prepared-manifest paths/SHA256, seed and batch topology. Resume rejects a changed policy or a legacy→balanced switch; starting a new run is required for a new objective. Validation remains the existing equal-source macro mean; this change does not exclude any source from evaluation or claim to solve overfitting.

> **Current architecture boundary (2026-09-11):** The only current immune data implementation is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; the formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. The active data flow is raw → adapter → `BioSeqRecord` → offline filter → prepared semantic JSONL → training-time grammar/padding/per-chain encoder reconstruction/masking, with no model-ready token cache. The deleted `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree, deleted `training` tree, deleted `GRAMMAR_V1.md`, and deleted legacy training/build/test/job files are historical deletion facts, not current dependencies. Retained model-layer files are `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`; `downstream/grammar` remains active for fusion CDR/light-pairing/TCR-generation public implementations.

## 2026-09-22 v2 decoder-only 对照

- 用户要求在当前 v2 上增加无 ESMC encoder、训练更久的对照。`fixed_receptor_lengths=True` 现在允许 `residue_cond_mode=token`，仍禁止 feature replacement；保留原词表、固定 167/135 decoder 槽、chain EOS、split token、generated-only diffusion 和数据处理。
- token 新训练完全不加载 encoder 权重、不创建 condition projection/norm，输入仅为 `wte(x_t)`；仍读取 ESMC tokenizer 文件以保持与融合模型同一 grammar/remap，这不使用预训练特征。decoder 仍从零初始化，未以增加层数补偿 encoder 参数量。
- checkpoint loader 支持真实 encoder-free state；旧 token checkpoint 如保存了 encoder 参数，仍按原结构完整加载供评估。resume 校验 mode 和 encoder-free metadata，拒绝把旧 encoder-bearing token 或 add 任务直接恢复到新结构。
- 新 YAML 的建议预算为 1M steps（融合版 200k 的 5 倍），LR/warmup 不变、cosine horizon 延长；这不是效果保证，也不是等步数/等算力的单因素对照。队列资源与正式提交尚未执行，状态见 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

## 2026-09-16 生成状态同步到 ESMC

用户进一步要求检验pairing采样步数：固定已完成的49000／反馈修复协议，新增无前缀8/16/32/64/96/128步全量对照，保留124步参照；只改采样预算，不改训练或评分器。方法／seed解释及验收边界见 [AB pairing §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。六档独立非闲时单卡，所有档保留，不在测试集选最优后改主表。

用户随后授权本地测试 CDR 多步效果：按 [AB CDR §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md) 执行SAb23H2完整六CDR的固定92000、iter1/2/4/8对照，其他条件保持一致。该实验是步数诊断，不更改单步主结果选取规则；保存逐条预测与真实ESMC反馈检查，Kong本轮未启动。

按用户要求修复共享多步 sampler：ESMC 每轮读取当前已接受的模型生成残基，pending 保持 MASK；必须转换 decoder／encoder 词表，不能从原始 encoder buffer 恢复隐藏参考答案。实现、CFG 边界、防泄露验收与本轮 pairing 同权重对照见 [AB pairing §7 k](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。这是 inference 状态修复，不修改训练 loss 或 checkpoint；旧多步指标不自动升级为修复后结果。

## 2026-09-15 T1 直接关系 token 测试

用户授权本地验证 frozen checkpoint 的原生 recognition 预测能力，作为现有 frozen+MLP 的独立方法行，不覆盖表征探针。默认无 CFG、单步关系 MASK 预测，不微调；完整输入及评分契约见 [TCR T1 §4.1a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)。

## 2026-09-15 Foundation eval 改为来源等权均值

用户明确先实施均值，不排除任何来源。[FusionTrainer.evaluate](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py:228) 对来源字典输出 `eval_loss = sum(source_losses) / number_of_sources`，替代按验证样本数加权；各来源自己的 loss、训练目标和数据保持不变。聚合以本次实际评估的来源字典为准，必须所有来源均返回 loss 才发布，并继续记录到 `state.log_history` 供 Top-K 选模。单一/拼接 Dataset 原生评估路径保持原状；正式多来源任务默认 `eval_per_source=True`。

CPU 回归：[test_eval_source_mean.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_eval_source_mean.py) 7 项通过（生产训练环境 `protenix_abtcr`，未加载权重/数据）；覆盖不等样本数、指标发布、缺失来源、显式来源子集/前缀、单来源和原生返回值。`pllm` 导入训练入口缺少 accelerate，未改环境。新代码需新进程加载，已运行任务不会热更新；未重启任务或重排旧 checkpoint。历史 row-weighted loss 不可直接与 macro loss 比较。

Antigen→抗体的现有测评缺口、MAGE/PALM-H3/结构设计参考和分层验证建议见 [初步调研](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ANTIGEN_CONDITIONED_AB_RESEARCH.md)。本轮只调研，尚未建立新生成基准。

## 2026-09-15 Specificity 训练预算与收敛诊断

用户要求的LR核查、Ophiuchus本地长轮数记录对照及Ours独立200轮实验，按 [AB Native Probes §4.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#45-specificity-200-epoch-学习曲线诊断2026-09-15) 执行；该段优先于旧默认100轮的描述，但不覆盖历史结果。只新增可观测性及独立头实验，不改变基础模型、pooling、其他AB任务或GDPa1参考CV规则。

用户后续要求使用最新训练点复测；所称step89000已被top-k清理，当前实际最新完整点step92000按 [AB Native Probes §4.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#46-v5-step92000-同协议对照2026-09-15) 冻结并完成同协议Specificity对照。比较只改变基础模型checkpoint，特征重新提取，头训练、数据、pooling与fixed-last口径不变；第200轮Accuracy从step49000的0.627325提高到0.643468，但仍低于论文baseline 0.6796。

## 2026-09-15 AB Specificity 分类头核查

分类头结构与实际宽度、训练参数及论文选模证据边界见 [AB Native Probes §4.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#44-specificity-分类头对齐核查2026-09-15)。仅诊断，不据头容量差异认定性能差距的唯一原因，不修改原生global pooling或增加训练实验。

## 2026-09-15 GDPa1 用户协议收窄

最新用户决定及代码/产物契约以 [AB Native Probes §4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md#42-预测头与协议切割线) 为准；其优先级高于下方09-13双协议回归的历史描述。当前展示见 [RESULTS §0.6a](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。不修改预训练、模型权重或其他下游任务。

## 2026-09-13 Ours 原生 AB 探针与 prompt/CFG 单卡矩阵

用户明确本轮要测我们的 ESMC+LLaDA，而非套用 Ophiuchus token窗口/pooling；原生 AB 表征适配、分类头及双协议回归实现/验收见 [AB_NATIVE_PROBES](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)。pairing 的 prompt0/prompt3 与 CFG 条件删除定义、非零分支修复及状态见 [pairing §4.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。用户进一步指定最新 v5，采用查询时最新完整保存点的独立评测快照，矩阵内不追逐后续保存；真实 GPU 门禁通过，11项正式单卡任务已提交，状态以 [PROJECT_PROCESS](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 为准。未改预训练或 baseline 权重；下方旧“未接 CFG/仅文档”描述仅对应此前阶段。

## 2026-09-13 AB pairing 多步解码诊断

**实施更新**：reference-length v3 完整生成、现有评分及产物验收已完成；实现、CPU 验证与单因素归因边界见 [AB pairing §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)，数值与结论见 [RESULTS §0.6 全量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。本次联合修正不支持将旧训练失配假说升级为唯一根因；下方“仅文档/未修复”是此前阶段记录，不再描述当前实现状态。未改训练 recipe，也未重评当前 v5。

**最新用户决定优先**：现行 Ours pairing 长度协议的选择见 [AB 测评审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)，实施与验收见 [任务文档 §4.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。本次仅同步文档；不把协议决定当成代码默认切换或新结果完成。

用户追问的 `<protd>`/padding、generated-only 与 all-chains 监督区别，以及未知长度生成建议、AirGen CFG 源码核对，见同一任务文档 §7 i/j；本轮仅 CPU 协议检查与解释纠正，不代表新长度协议或 CFG 已实现。

后续 attention 复核进一步收紧“最大 MASK 窗口”建议：其与现有训练的兼容性约束、AirGen 训练尾部的源码依据及不重训/适配两条路径的边界统一维护在同节第 5–7 点；不能当作现有 checkpoint 已验证的直接修复。

共享 grammar sampler 的已复现状态问题、与 AirGen 的差异及修正前后受控对照要求见
[AB pairing §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md)。先排除生成器混杂，再判断训练失配贡献；本轮仅诊断与留档，未实施修复或重训。

## 2026-09-13 Ophiuchus-Ab baseline 探针测评纠错

用户确定的五折选模/聚合要求、论文与源码证据边界、三个表征探针的修正方案及验收条件统一见
[AB baseline 测评审计与完善指南](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)。当前仅完成文档登记，未修改测评实现或执行新实验；不据此变更 headline 范围。

## 2026-09-13 TCR 下游评测纠错

最新执行决定优先：双链生成推迟，先实施/提交指定 49000 的当前 TCR 评测；关键决定和输入/工程 gate、论文可引用 gate 的区别见 [审计 §9.21](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。单任务实现由相应 task owner 维护；不修改预训练目标、既有 baseline 权重或 AB 共享 sampler。

已有生成模型使用的配对数据来源及复用建议见 [T4 §7.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.20](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。已核查数据实物与原生实验用途；不是新评测实现、训练调整或全部去污染通过。

生成任务按β-only / 配对αβ登记候选，原始方法、可用性证据、接入顺序与数据/预算边界见 [T4 §7.5](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.19](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。这是调研与计划登记，尚未实施双链评测或修改模型/训练方案。

benchmark14的目标级数据结构、参考与作者预测的区分，以及TCRT5发表版本的输出范围见 [T4 §7.4.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.18](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮仅核查解释，不改变任务输入或评分协议。

本轮指定 checkpoint 的全套 TCR 重评条件、others 输入档及只汇总本轮 Ours 的呈现要求见 [审计 §9.16](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。任务专属 gate 未通过前不提交旧 wrapper；本轮仅复核与登记，未实施新协议。

用户已确认T4条件联合αβ生成的未知区域语义，协议及评测输入/上游受体信息的边界见 [T4 §7.4](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.17](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。区域语义不再待确认；生产实现与验收尚未完成，不将上游参考受体字段擅自升级为生成条件。

TCRT5 held20 / benchmark14 对v5实际数据的精确成员核查已完成，结果与后续已见/未见分层建议见 [T4 §7.3](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.14](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。这是数据审计，不代表新模型重评或训练方案变更。

T4 baseline 原生生成机制与条件对齐问题见 [T4 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [审计 §9.13](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；本轮仅研究与记录，不改变生产实现或新增实验。

T4 当前生成条件/目标与历史成绩边界初查见 [T4 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md) / [总审计 §9.12](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。仅静态诊断与文档同步；生成期望关系、参考相似度的解释须遵循任务专属边界，未实施新适配或重评。

用户要求将 T3 关键理解直接固化在总审计中；解释及后续比较准则集中见 [TCR 审计 §9.11.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本次仅整理已核实结论的使用边界，不新增实验执行或改训练授权。

T3 原论文复核更新：旧“本地 recipe 与论文兼容”的判断撤回；正式原文、作者代码/模型卡与本地产物的差异及后续分臂要求统一见 [T3 §7.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)，关键日志见 [总审计 §9.11](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。仅研究与引用纠错，未改变训练方式或实施新的下游协议。

T3 专项初查及支持集标签的合法用途见 [T3 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md) / [TCR 审计 §9.10](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。仅审计，不改模型/训练；后续 v5 输入适配需遵守该任务 gate，不能把 T1 带表位的查询模板套入无条件表征。

T2 专项审计已启动；无条件表征与 T1 关系查询的区别、当前 v5 输入适配缺口及 CPU 检查边界见 [T2 任务 §7.1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md) / [TCR 审计 §9.9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本轮仅诊断，不修改训练范式或直接把所有 relation 改成 mask。

用户再次确认希望覆盖全部已构建 TCR 任务 T1–T4，并要求持续遵守审计准则；任务清单、任务特定关系/目标信息边界及后续验收要求统一见 [TCR 审计 §9.8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本次仅登记要求与入口盘点，不改变任务协议或声明全套重评已完成。

批准并接入同 others 行集的 LongA/LongB 版本：直接复用训练完整链路径，关系 mask；具体校验、上下文/池化同时变化的解释边界及验收见 [TCR 审计 §9.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本次不改预训练或 head recipe，不宣称复现 TCRconv-fullAB。

新增同 others 行集、同 v5 checkpoint 的 β-only 控制；数据选择与模型可见输入分离，补全 seed 的消融边界及验收以 [TCR 审计 §9.6](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 为准。未改预训练架构或训练 recipe。

用户确定的 T1 输入约束、baseline 训练范围审计及冻结/全参数/关系预测对照建议统一见
[TCR baseline 纠错记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。用户确认的运行时输入决策、已实现范围与测试边界以该文档 §2 / §9 为准；
不要将历史 T1 frozen-probe 数字当作纠正后 v5 的结果。others 第一版的字段取舍及当前训练方式同样只维护在该记录 §9。

## Goal

Build a diffusion-model-based immune-receptor foundation model in
`/vepfs-mlp2/c20250601/251105016/project/dllm_test` for paired antibody H/L,
paired TCR α/β, antibody-antigen recognition, and TCR-epitope/pMHC recognition.

## Active biological scope（2026-07-23）

- The next foundation-training recipe has four planes only: antibody pairing,
  TCR pairing, antibody-antigen, and TCR-epitope/pMHC.
- Nanobody/VHH, MINT/STRING/general-PPI, the current antigen-free
  `neutralization` shard, and specificity-free bulk TCR are excluded from the
  active recipe. Their historical checkpoints, benchmark artifacts, and older
  roadmap notes remain provenance, not active training requirements.
- Antibody context means actual protein/peptide sequence, not only a target
  name. TCR context keeps epitope separate from MHC/B2M and records whether an
  MHC sequence is full length or a pseudosequence.
- The canonical data record must preserve sequence scope, receptor/ligand
  clusters, donor/assay, continuous value/unit/censor, and source record
  provenance. Dataset rows with the same sequences but different HLA or assay
  are not interchangeable duplicates.
- The current immune LLaDA runtime exposes the semantic record and mask contract through
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; target views
  are represented by record metadata plus runtime grammar/mask construction, not by the deleted
  Arrow loader. H→L/L→H, α→β/β→α, antigen-conditioned receptor generation, pMHC-conditioned
  TCR generation, and chains→relation remain task/view goals to validate on the current line.
- Historical 7L / `immune_receptor_v2` candidate counts remain in
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/TRAINING_DATA_CATALOG.md` and
  `IMMUNE_RECEPTOR_DATA_V2.md`. Current prepared v4/v5 counts are owned by
  plan §4.1 / §4.2 — do not read the 7L catalog as the live mix.

## Canonical data execution（2026-08-04）

- Historical/candidate AB/TCR-only `bioseq.v2` artifacts were built under
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/immune_receptor_v2`;
  their historical build entry was
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/build_immune_receptor_v2.py`.
  They are not the current runtime training input while their export/training gates remain false.
- The frozen candidate unions contain 922,479 TCR records and 470,949 antibody
  records. The strict split populations are 41,489 TCR specificity records,
  308,267 antibody-antigen interactions, and 19,987 canonical antibody-property
  records; 19,981 property records satisfy the export-core rules.
- All nine deterministic component/group split manifests pass their disjointness
  audit. A source's released train/test assignment is provenance only and is
  never silently treated as the new training split.
- Exact measurement duplicates may merge provenance; conflicting labels,
  different assay values/censors, HLA contexts, or target mapping scopes remain
  separate. No majority-vote label collapse is allowed.
- Frozen benchmark exact/near quarantine, MMseqs2 sequence-cluster
  decontamination, group-disjoint resplitting, train-only negative construction,
  and immutable SHA256 manifests are complete. Core build
  `ir2exp_f7a60484c7e3a20db6a2` and strict OAS/OTS pairing build
  `ir2pair_6a1a5b62752caccd2cd5` both pass their technical audits with zero
  residual benchmark-cluster matches.
- Candidate recipe `ir2recipe_1e3eac44551e88aedc6e` selects strict OAS H/L,
  strict OTS alpha/beta, antibody-cluster-disjoint recognition,
  receptor-cluster-disjoint TCR recognition, and parent-disjoint antibody
  properties. It references 3,077,769 real train, 39,138 valid, and 38,565 test
  records, plus a separate 3,970-row train-only synthetic TCR-negative pack.
- This execution did not change the model, renderer, checkpoint, or sampling
  weights and did not start training. `technical_data_export_ready=true`, but
  `export_ready=false` and `training_ready=false` until the SAbDab2/CATNAP rights
  review, per-chain/per-region target rendering, and four-plane sampling/token
  budget policy are completed.
- Full schema, source decisions, counts, hashes, leakage audit, and export gates
  are authoritative in
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md`.

## Code Location

- Current immune data pipeline: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`
- Current formal training entry: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`
- Current immune data tests: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada`
- Retained model-layer compatibility code: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch`
- Historical Ophiuchus/BioSeq code and tests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq` are not the current immune training entry.
- Weight root: `/c20250601/mj/model_weights`

## MINT downstream boundary (2026-07-21)

- MINT 重建属于 benchmark/data-harness 修复，不改变 BioSeq 模型结构、训练数据或 checkpoint。
- 当前 MINT 面板严格为 HumanPPI、YeastPPI、Gold-standard PPI、MutationalPPI、SKEMPI
  五项；canonical 数据来自
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`。
- 普通 PPI 分类输入两条 partner 序列；MutationalPPI/SKEMPI 输入 WT 与 mutant 的两条
  partner 序列，并用 `embedding(WT PPI) - embedding(mutant PPI)` 训练下游 head。
- 只有与论文数据、fold、head、重复次数全部对齐的本地结果才可称复现。当前不修改
  BioSeq 模型，先修 baseline：HumanPPI、YeastPPI、Gold-standard PPI 的 8 模型结果
  直接引用 Source Data 并标 `[P]`；MutationalPPI、SKEMPI 在可审计本地固定 fold 上重跑
  同一组 8 模型并标 `[L]`，不跨协议排名为 paper-exact。
- 本地两项统一用 separate-chain、`embedding(WT)-embedding(mutant)`、WT/mutant 对齐
  窗口、3 次 640-hidden MLP；run cap=2048，ESM-1b 与 ProGen2-Large 按各自原生限制
  使用 1024。MutationalPPI 用 WT pair-group 10-fold；
  SKEMPI 用冻结的 notebook complex-held-out 三折。该变化只属于 benchmark harness。
- 数据构建和协议审计入口是
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`。

## Immune LLaDA data / receptor-completion decisions（2026-09-12）

- Current implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; formal entry: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- Current data flow is raw source reader → source adapter → canonical `BioSeqRecord` → offline deterministic filters → prepared semantic JSONL/manifest → training-time grammar rendering/collation → batch padding/tensor assembly → per-chain encoder input reconstruction → diffusion/MLM masking. No model-ready token cache is produced.
- Receptor-completion / relation-diffusion **design and risk register** are owned by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`. Layout fields: `DATA_FORMAT_AUDIT.md`. How to run: `dllm/pipelines/immune_llada/README.md`.
- Method decisions that must not be “fixed” later: (1) complete epitope-conditioned single-chain rows because `GrammarTokenizer` has no `<tcra>`/`<tcrb>` and positional identity made 8,885 alpha-only rows indistinguishable from beta-only; (2) branch on actual fv/CDR3 columns, not `sequence_scope`; (3) drop all-X epitopes via a named filter, not a silent adapter `None`; (4) do not infer MHC from epitope co-occurrence; (5) accept synthetic-`X` attention cost as the price of a single α/β layout.
- Prepared versions: v4 published; v5 published. Counts: plan §4.1 / §4.2. v3-era homotypic-pair acceptance remains in `docs/IMMUNE_LLADA_DATA_ACCEPTANCE.md` and is not the current default.
- The old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree and old `training` tree are deleted. Do not recommend those paths. `refactor_baseline` and `docs/archive` remain historical evidence.
- The retained qwen-named directory is model-layer code only: `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`. Active immune grammar/tokenizer/collator: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Architecture

- The antibody training target is the exact Ophiuchus-Ab path in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus`.
- `BioSeqEncoderDiffusionModel` is the ESMC/ESM feature-conditioned target.
- The no-encoder antibody model uses the migrated Ophiuchus-Ab ESM2 transformer with chain-aware multimer attention.
- The ESMC/ESM feature-conditioned model runs the biological encoder on the current diffusion state `x_t`, replaces decoder residue embeddings with gathered encoder features (default; no projection), and treats ESMC freezing only as a bootstrap or ablation setting.
- ESMC should be treated as a per-protein-sequence representation model, not as a native multi-chain complex encoder.
- Multi-chain reasoning should be implemented in the BioSeq decoder/collator with explicit `chain_ids`, or by using an ESMFold2-style structure head that builds complex-level pair representations after ESMC representations.
- ESMFold2 supports biomolecular complex inputs through `StructurePredictionInput` / `ProteinInput` / `DNAInput` / `LigandInput`; its complex handling is not simply independent ESMC encoding with no downstream cross-chain module.
- The ESMFold2 preprint describes full uncropped protein sequences for each chain being passed independently to frozen ESMC 6B, followed by projection into a 2D pair representation, language-model encoder layers, recurrent pair folding layers, and an atom-level diffusion module.
- The ESMC/ESMFold2 paper does discuss multi-chain systems, but the explicit multi-chain mechanism is in ESMFold2: ESMFold2 encodes each protein chain independently with frozen ESMC 6B, then builds complex-level pair representations and folds/designs complexes downstream. ESMC itself should not be treated as a native multi-chain interaction encoder.
- For antibody, TCR-pMHC, and PPI sequence modeling, ESMC embeddings alone are insufficient as the only cross-chain mechanism; the BioSeq no-encoder path must keep Ophiuchus-Ab-style chain-aware attention, and the encoder path must pass `chain_ids` plus ESMC-derived features into a decoder or structure-style pair module.
- Ophiuchus-Ab's MINT path uses `multimer_attn` inside the ESM2 transformer layers as the cross-chain mechanism. It is not the HuggingFace-style `crossattention` / adapter-cross-attention path present in other AirGen DPLM modules.
- Ophiuchus-Ab compatibility uses the ESM2 alphabet from `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`: vocabulary size 33, `<cls>` id 0, `<pad>` id 1, `<eos>` id 2, `<mask>` id 32.
- The released local ESMC tokenizer files under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` expose the same ids as ESM2 for standard amino-acid tokens, but id 31 differs: ESM2 uses `<null_1>`, while ESMC uses `|` as an additional special token. Use the encoder's own tokenizer when ESMC is active.
- Ophiuchus-Ab antibody collation should encode heavy and light chains independently with per-chain `<cls>/<eos>`, pad heavy to 150 tokens and light to 128 tokens, then concatenate to a fixed 278-token sequence with explicit `chain_ids`.
- The Ophiuchus-Ab architecture preset should use hidden size 1280, 33 transformer layers, 20 attention heads, FFN size 5120, ESM2 token dropout, no learned position embeddings, and chain-aware multimer attention.
- Loading `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` is implemented in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/multichain.py` via `load_ophiuchus_checkpoint`, using the Ophiuchus-exact mint ESM2 block under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/mint`.

## MIMIC-Inspired Design Notes

- MIMIC/LORE should be treated as a reference for the later multimodal BioSeq roadmap, not as code to import today.
- Borrow the split-track idea: aligned per-residue modalities such as amino acid sequence, chain type, CDR/region tags, ESMC features, predicted secondary structure, SASA, structure tokens, interface/surface features, and antigen/TCR/pMHC annotations should be fused within the same residue coordinate frame instead of flattened into separate long sequences.
- Keep unaligned context as separate token groups: task prompt, species, assay, tissue/cell context, target name, functional caption, and design constraints should be separate semantic tokens that can interact with sequence tracks.
- Add register tokens for global complex/entity representation; use register tokens plus track pooling for downstream antibody/TCR/PPI property heads.
- Use local group-reset RoPE or equivalent group-aware position handling so each chain or modality group preserves internal residue distances without treating all concatenated tokens as one absolute coordinate system.
- Use pathway-based training over partially observed samples: examples can have sequence only, sequence plus ESMC, sequence plus structure/surface, paired chains, antigen context, or downstream labels. Rare high-value pathways such as antibody-antigen and TCR-pMHC should be upsampled instead of drowned by sequence-only examples.
- Use asymmetric context/target budgeting for multimodal generation: large encoder context for conditioning, smaller decoder/diffusion target window for masked residues or design regions.
- Use length-bucketed dynamic batching rather than truncating protein complexes; for protein structure or interaction tasks, dropping over-budget samples is safer than cropping away global contacts.
- For protein design evaluation, borrow MIMIC's independent-oracle pattern: generate with BioSeq, then evaluate with ESMFold2/AlphaFold-style structure confidence, interface confidence, TM-score-like fold recovery, and surface/chemistry similarity instead of scoring only with the same model that generated the sequence.
- Release status as of 2026-06-11: MIMIC code, weights, and LORE release assets are not yet publicly available for direct integration.

## Qwen3-VL-Inspired Architecture Notes

- Qwen3-VL should be used as an architecture pattern for multimodal BioSeq, not as a direct code dependency.
- The useful pattern is `modality encoder -> merger/projector -> placeholder-token embedding replacement -> decoder -> task head/loss`.
- For immune receptor modeling, replace Qwen3-VL image/video tokens with typed biological placeholders such as antibody heavy/light, TCR alpha/beta, peptide, MHC, antigen, and PPI partner tokens.
- Replace Qwen3-VL 3D T/H/W multimodal RoPE with BioSeq-aware position metadata: text/control position, chain-local residue position, chain id, chain role, modality/track id, and optional interface or structure-coordinate track.
- Keep a connector layer between tunable ESMC or ESM2 encoders and the BioSeq decoder. This connector should handle length reduction, hidden-size projection, and modality/chain-type normalization, analogous to Qwen3-VL's visual merger.
- Borrow Qwen3-VL's deepstack idea cautiously: intermediate ESMC/ESM2 features can be injected into early BioSeq decoder layers at residue-token positions, but this should be behind an explicit config flag and validated against the simpler single-projection path.
- Training should expose separate learning rates for biological encoders, connector/projector modules, and decoder/diffusion modules, mirroring Qwen3-VL's separation of vision tower, merger, and LLM; the default should keep the biological encoder trainable when it is used.
- Dense decoder should be the first BioSeq foundation target; MoE-style routing can be revisited later for large mixed antibody/TCR/PPI training after the dense model and data schema are stable.

## Multi-Chain Tokenization Notes

- BioSeq should not represent multi-chain complexes by concatenating raw amino acid strings with a single generic separator only.
- ESMC's local tokenizer includes `|` at id 31 and marks it as an additional special token, which is compatible with using a chain separator for ESMC inputs. This is only tokenizer-level support; it is not proof that ESMC alone learned robust cross-chain interaction modeling.
- ESM2/MINT-compatible tokenization uses `<null_1>` at id 31 instead of `|`. Therefore `Esm2SequenceTokenizer` is correct for Ophiuchus-Ab/MINT and local ESM2 snapshots, but ESMC encoder runs should use `HuggingFaceEsmTokenizerAdapter` or an ESMC-specific tokenizer loaded from `/c20250601/mj/model_weights/esmc/<model>/tokenizer.json`.
- Keep the first pretraining token format minimal. Required tokens are only amino acid tokens plus `<pad>`, `<mask>`, `<unk>`, `<bos>`, `<eos>`, `<chain_sep>`, and a small set of complex-type header tokens such as `<type_ab>`, `<type_ab_ag>`, `<type_tcr>`, `<type_tcr_pmhc>`, and `<type_ppi>`.
- Do not add many role-specific or position-number tokens in the first version. For example, avoid `<ab_heavy>`, `<tcr_beta>`, or `<pos_53>` as default vocabulary items unless ablation shows they are needed.
- Use chain ordering conventions under each complex-type header instead of many chain-role tokens. Example: `<type_ab>` means chain order is heavy then light; `<type_tcr_pmhc>` means alpha, beta, peptide, MHC; `<type_ppi>` means partner A then partner B.
- Model hierarchy with embeddings, not vocabulary growth. Each residue should receive a chain-internal position embedding or RoPE that resets per chain, plus a chain-level index embedding for the outer multi-chain order.
- The initial position system should have two levels: `residue_position_in_chain` and `chain_index_in_complex`. Optional global absolute position can be retained only for packing/caching, not as the main biological position signal.
- Pretraining should be treated as masked generation over all eligible residues with different mask probabilities by chain/type/task. Fixed-context behavior is a mask policy choice, not a separate objective.
- The schema should support `diffusion_target_mask`, `fixed_context_mask`, and optional per-chain mask probabilities. More detailed biological relationship features or external representations can be added later through encoders/connectors rather than through more special tokens.

## Immune Receptor Modeling Considerations

- First-version pretraining should stay simple, but the data schema must preserve immune-receptor metadata even when it is not used as input tokens.
- Preserve receptor gene and region annotations where available: V/J genes for antibody light chains and TCR alpha chains, V/D/J genes for antibody heavy chains and TCR beta chains, CDR1/CDR2/CDR3 spans, framework spans, species, isotype, and source assay.
- For antibodies, paired heavy-light learning is central. Heavy-only or light-only samples can be useful for scale, but paired heavy-light data should be upsampled or given a dedicated mixture so the model learns chain pairing rather than only single-chain naturalness.
- For antibody-antigen tasks, antigen identity is often missing in large repertoire datasets. Do not treat repertoire-scale antibody samples as antigen-conditioned examples unless target antigen labels or binding assays are known.
- For TCR-pMHC, MHC allele/class and peptide identity are essential context, not optional metadata. TCR-only pretraining can learn receptor grammar, but TCR-pMHC learning requires paired alpha-beta TCR plus peptide plus MHC/HLA context whenever available.
- TCR-pMHC labels are noisy and depend on assay type. Store assay/readout metadata and avoid mixing tetramer binding, activation, expansion, tissue enrichment, and author-curated specificity as identical labels in supervised evaluations.
- Negative sampling for binding/specificity tasks must be explicit and benchmark-specific. Randomly shuffled negatives are useful for training but can overstate generalization; unseen peptide/HLA and unseen receptor splits should be maintained for evaluation.
- Multi-chain relation learning should be measured separately from single-chain grammar using pair recovery, conditional generation, binding/specificity ranking, and structure/interface oracle evaluations.
- Region-aware mask policies should be available as an ablation: uniform masked generation is the default foundation objective, while higher CDR3/paratope/peptide masking can test whether the model improves receptor-specific design.
- External structural or representation features should be added through encoder/connector paths after the minimal sequence diffusion baseline is stable, rather than expanding the first tokenizer.

## Additional Data Types

- Repertoire-scale AIRR-seq data: large unpaired or paired BCR/TCR repertoires from OAS, OTS, iReceptor/AIRR Data Commons, PIRD, and TCRdb. Use for broad masked-generation pretraining and repertoire distribution learning.
- Paired-chain receptor data: paired antibody heavy/light and paired TCR alpha/beta records. Use with higher mixture weight than single-chain data to learn receptor pairing.
- Antigen/specificity-labeled data: antibody-antigen and TCR-pMHC pairs from IEDB, VDJdb, McPAS-TCR, TBAdb/PIRD, curated antibody binding resources, and assay-specific datasets. Use for conditional mask generation, ranking, and supervised evaluation; keep assay metadata.
- Structural complex data: SAbDab/SAbDab-nano, STCRDab, TCR3d, SCEptRe, PDB-derived antibody-antigen and TCR-pMHC complexes. Use for structure/interface oracles, optional structure-conditioned training, and nonredundant benchmarks.
- Functional and biophysical data: affinity, neutralization, specificity, developability, expression, thermostability, aggregation, immunogenicity, polyreactivity, and pharmacokinetic labels. Use as downstream heads or filtering/oracle data, not as the first pretraining objective.
- MHC and epitope presentation data: peptide-HLA binding, ligand elution, immunogenicity, and MHC allele metadata. Use to make TCR-pMHC context realistic and to avoid treating peptide identity without MHC context as sufficient.
- Single-cell immune multi-omics: paired receptor sequence plus cell type, tissue, disease, clonotype expansion, gene expression, and antigen-tetramer labels. Use later for context-aware modeling and split construction, not as required input for first baseline.
- Germline/numbering/reference data: IMGT, OGRDB, ANARCI/ANARCII-style numbering, V/D/J references, CDR/FR annotations. Use for metadata normalization, region-aware masking ablations, and evaluation stratification.
- Synthetic and library screening data: display libraries, deep mutational scanning, saturation mutagenesis, designed binders, and experimentally validated generated receptors. Use for targeted fine-tuning and validation of design behavior.
- Negative and decoy data: shuffled nonbinders, hard negatives by shared peptide/HLA or similar receptor, and structure/interface decoys. Keep negative-generation protocol explicit because it strongly changes reported performance.

## TCR Data Expansion Notes

- Current local TCR resources are not empty: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final` contains about 2.12M cleaned paired alpha/beta TCR records, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr` already contains VDJdb, McPAS-TCR, ImmuneCODE/MIRA, IEDB export archive, and PIRD-derived assets.
- For first masked-generation pretraining, local OTS paired alpha/beta is enough to start; the immediate gap is integrating it cleanly into the unified BioSeq training mixture rather than only downloading more.
- For larger TCR grammar coverage, add bulk/unpaired TCR repertoire sources such as TCRdb2.0, iReceptor/AIRR Data Commons, and immuneACCESS/immunoSEQ public exports. These should be stored under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw` and converted into explicit single-chain or beta-chain records with sample metadata.
- TCRdb2.0 raw data is now downloaded under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`: 263 project zips, 263 metadata CSVs, and 1 healthy reference zip. Download validation against remote `Content-Length` passed with no missing or mismatched files; see `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests/tcrdb2_download_validation.tsv`.
- For TCR-pMHC relation learning, prioritize specificity/context datasets such as IEDB receptor exports, latest VDJdb, McPAS-TCR, ImmuneCODE/MIRA, PIRD/TBAdb, IMMREP/Kaggle-style benchmark data, and paired peptide-HLA datasets. Store raw files under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_specificity_raw`.
- Do not blindly mix all newly downloaded bulk TCR rows into the main pretraining pool. Bulk datasets can dwarf paired OTS and erase multi-chain learning unless mixture weights cap bulk/unpaired data.
- Before any large new download, add dataset manifests and adapters that record source, license/access terms, chain availability, paired/unpaired status, peptide/MHC availability, assay type, species, tissue/disease, and split group.

## End-to-End Pipeline

- Raw data layer: keep each source immutable under absolute raw roots such as `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi`.
- Manifest layer: every source should have a machine-readable manifest with source name, raw path, file checksum or byte validation, species, chain availability, paired/unpaired status, task labels, peptide/MHC availability, license/access terms, and split group. TCRdb2.0 already has manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests`.
- Adapter layer: current source adapters and offline preprocessing live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; they normalize raw records into canonical `BioSeqRecord` rows before prepared semantic JSONL export. The former `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` conversion path is historical.
- Canonical data layer: current training rows use the `BioSeqRecord`/prepared semantic contract owned by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. Bulk unpaired TCR rows from TCRdb2.0 should become single-chain or beta-chain records first, not multi-chain pseudo-pairs.
- Prepared-data layer: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py` reads the immutable prepared manifest and semantic JSONL shards. Source weighting, filtering, row conversion, and rejection decisions belong to offline preprocessing, not a deleted raw-CSV runtime dataset.
- Collation/masking layer: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py` and `collator.py` emit token tensors plus `diffusion_target_mask`, `fixed_context_mask`, chain ids, chain-internal positions, and per-chain encoder tensors. Diffusion/MLM corruption and loss operate only on the selected eligible positions.
- Retained compatibility model layer: the Ophiuchus-Ab/MINT multichain code under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus` remains historical/independent provenance, not the current immune training entry; the former `examples/bioseq/train_bioseq_ddp.py` entry point is retired and deleted.
- Current model layer: the ESMC/ESM2 feature-conditioned fusion path extracts features from the current diffusion state `x_t`, then performs multi-chain denoising over the concatenated token stream. Fixed context chains, such as antigen in antibody-antigen generation, remain clean and do not receive direct reconstruction loss.
- Checkpoint layer: training checkpoints should be written under absolute output roots, save `latest.pt` and `final.pt`, and preserve `backbone_state_dict`, optimizer state, step, epoch, and args for resume and downstream evaluation.
- Evaluation layer: downstream evaluation should use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark` as the first integration point. The benchmark already accepts `--embedder bioseq:/abs/path/final.pt` through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/common/model_api.py`.

## Fast Downstream Validation Plan

- Validation should run on every meaningful current immune-fusion checkpoint, not only the final checkpoint. Minimal cadence: initialization/smoke checkpoint, early checkpoint, mid-training checkpoint, and final checkpoint; record the actual absolute checkpoint paths in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Tier 0, pretraining sanity: held-out diffusion loss on small OAS/OTS/nanobody/TCRdb2.0 slices, amino-acid distribution checks, valid-token rate, duplicate/near-neighbor rate against train, and chain-length distribution drift. This is the fastest failure detector for bad adapters or masks.
- Tier 1, embedding-only IRBench: run the existing benchmark with frozen embeddings and cheap heads. Priority commands are T1 TCR binding, T3 TCR representation, P1 PPI, and NbBench scalar tasks under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark`. This checks whether the trained backbone representation is useful without generation noise.
- T1 comparison boundary: Ours may continue to use the frozen-head protocol on its designated training split, but external baseline rows must not be retrained. Their canonical source is official original checkpoint + official inference + `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip`; unavailable official runs fall back to the labelled paper original-model value. This keeps the model-development protocol separate from external-baseline provenance.
- T1 retrained checkpoints are a separate artifact-reproduction audit, never a replacement for the canonical original-only table. The audit uses the released fold checkpoint + official inference + exact `retrain.zip` member and R `precrec`-compatible AUPRC. A local retrained value is selectable only when both five-fold mean AUROC and AUPRC round to the paper's four decimals; otherwise retain the local result as a diagnostic and report the marked paper retrained value. Any inference-time use of fold train data (currently only TCR-H descriptor-column reconstruction) must be declared and must not call model fitting.
- Our-model T1 retrained comparison uses the same five official `retrain.zip` train/test folds but never updates the BioSeq checkpoint. Render peptide + CDR3β as one role-explicit joint binding grammar record, read the final post-LLaDA residue states, apply one global mean across both chains, and fit only a `960 -> 256 -> 128 -> 1` ReLU/dropout MLP per fold. Internal early stopping uses a stratified 10% subset of that fold's train rows only; no test rows select epochs. This is a frozen-backbone downstream head experiment, not continued foundation-model training and not an external-baseline retrain.
- The primary rank for that experiment is against official-checkpoint baseline reruns on the same released CSV bytes (`ranking_local_release_AS.csv`). Paper means are kept in a separate `ranking_paper_reference_AS.csv` marked mixed-source, because the released seen files do not reproduce several paper means exactly. Neither ranking replaces the canonical original-only T1 table.
- Tier 2, generation/infill: run small OTS/TCR CDR infilling and antibody heavy-to-light completion. Primary metrics should be AAR for known masked regions, novelty, nearest-neighbor distance, k-mer JSD, length validity, and invalid amino-acid rate.
- Tier 3, task-specific fine-tuning: only after Tier 1/2 pass, fine-tune small heads or lightweight task adapters for IMMREP23 binding, FLAb developability, antibody specificity, and PPI. This separates representation quality from generation quality.
- The main comparison table should always include `kmer`, `esm2_150m`, `esm2_650m` when feasible, `ophiuchus`, and `bioseq:/abs/path/checkpoint.pt`. Report both seen and unseen splits for TCR binding; unseen macro-AUC0.1/AUPRC is the primary signal.
- A checkpoint is worth keeping if it improves at least one of: held-out diffusion loss, T3 representation probe, T4 infill AAR/JSD, or antibody generation validity, without degrading unseen TCR binding below the baseline noise band.

## Latest Task Roadmap

- Priority 0: keep the current immune-fusion training path working through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`, and the retained fusion/model-layer integration. The exact Ophiuchus-Ab path under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq` is historical/compatibility provenance, not a replacement current entry.
- Priority 1 antibody binding task: add AbBiBench-style antibody-antigen affinity scoring and CDR design evaluation. Inputs should include heavy chain, light chain, antigen chain(s), optional complex structure, mutant region, and experimental binding score. Metrics should include Spearman/Pearson correlation for scoring, top-k enrichment for ranking, and external complex-quality oracle scores.
- Priority 1 antibody developability task: add FLAb-style property prediction heads for expression, thermostability, immunogenicity, aggregation, polyreactivity, binding affinity, and pharmacokinetics. These heads are useful for filtering generated antibodies before structure oracle evaluation.
- Priority 1 TCR-pMHC binding task: add IMMREP-style TCR specificity prediction with paired alpha-beta TCR chains, peptide, MHC allele/class, and unseen-pHLA splits. Metrics should prioritize AUPRC under strict negative sampling, with separate seen-epitope and unseen-epitope reports.
- Priority 1 TCR-pMHC model reference: track DecoderTCR as an ESM2-family baseline and architectural reference for TCR-pMHC sequence modeling. Its tasks map directly to binding prediction, interaction scoring, and TCR sequence analysis.
- Priority 1 TCR-pMHC structure task: add structure-oracle evaluation for TCR-pMHC complexes using DockQ, RMSD, TM-score, and CDR3 pLDDT-style reranking signals. SCEptRe should be considered as a frequently updated source of nonredundant immune-complex benchmark splits.
- Priority 2 PPI task: add paired-sequence PPI training/evaluation inspired by PPLM-PPI. The first target should be binary interaction prediction; the second target should be quantitative affinity prediction; the third target should be residue-level interface/contact prediction.
- Priority 2 mutation-fitness task: add ProteinGym-style single-chain and paired-chain mutation effect scoring to test whether BioSeq likelihood/diffusion scores correlate with DMS fitness and clinical variant labels.
- Priority 2 general protein foundation evaluation: use PFMBench and ProteinBench as broad external evaluation suites rather than training targets; they are useful for deciding whether BioSeq is overfit to antibody/TCR/PPI tasks or remains a general sequence foundation model.
- Priority 2 de novo binder design task: use ESMFold2, BindCraft-style filtering, and later Proteina-Complexa-style benchmarks as oracle/evaluation references for generated minibinders, antibody-derived formats, and PPI binders. Do not block the sequence foundation path on full structure generation.
- Immediate schema implication: extend BioSeq examples beyond `chains` and `task_type` to allow optional `chain_roles`, `target_chain_indices`, `mutations`, `labels`, `assay_type`, `antigen_chains`, `mhc_allele`, `peptide`, `structure_path`, and `oracle_scores`.
- Immediate code implication: add future task adapters under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data` before downloading any new large dataset; each adapter should normalize external datasets into the current prepared semantic-record contract.

## Data Schema

- Canonical BioSeq JSONL schema version is `bioseq.v1`.
- Required fields: `chains`, `task_type`, and `source`.
- Optional but preferred fields: `chain_roles`, `targets`, `generation_spec`, `split`, `labels`, `regions`, and `metadata`.
- Current `chain_roles` vocabulary should include `antibody_heavy`, `antibody_light`, `nanobody_vhh`, `tcr_alpha`, `tcr_beta`, `peptide`, `mhc`, `antigen`, and `other`.
- `targets` is only a coarse default list of chain indices that can be generated. It is not sufficient for conditional tasks by itself.
- Fine-grained generation must be represented by `generation_spec` or by a sampled training view. This view resolves to token-level `visible_mask`, `fixed_context_mask`, `diffusion_target_mask`, and `diffusion_loss_mask`.
- `generation_spec` should support chain-level completion, region-level infilling, span-level infilling, inverse region infilling, and conditional receptor generation. Examples: heavy-to-light generation, antigen-to-antibody/nanobody generation, heavy+antigen-to-light generation, alpha+beta+MHC-to-peptide design, pMHC-to-alpha+beta design, FR-conditioned CDR infilling, single-CDR infilling, and CDR-conditioned FR generation.
- `regions` should be keyed by string chain index and can store `FR1`, `CDR1`, `FR2`, `CDR2`, `FR3`, `CDR3`, and `FR4` for antibody/TCR CDR infilling.
- Current adapters, records, preparation, grammar, and collation live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. The former BioSeq conversion modules are historical and are not current runtime dependencies.
- Supported source types today: `oas`, `ots`, `nanobody`, and existing `processed` JSONL.
- Full conversion of OAS/OTS/nanobody should not be run until output root, shard size, and train mixture weights are decided. Use `--limit` for small conversion checks.

## Historical PPI and Interaction Data（not a current foundation-training entry）

- Raw interaction-task downloads live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- The rebuild script is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py`.
- Historical processed outputs retained for audit:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- The unified records CSV is a row-level integration table, not the final training format. It has `6271559` records and is useful for auditing, filtering, and writing task-specific `bioseq.v1` shards.
- Supported row-level sources in the current CSV: Figshare gold-standard PPI, HumanPPI LMDB, YeastPPI LMDB, SKEMPI, SWING MutInt, FLAb binding, TDC TCR-epitope, PISTE TCR-epitope-HLA, TEIM binding/interface metadata, oncoPPI spreadsheets, and CoV-AbDab neutralization.
- STRING-DB v12.0 was a separate large pretraining source in the historical row-level CSV plan. Its raw physical links and sequence dumps were too large for that pass; any future use requires an approved adapter under the current prepared semantic-record contract, not the deleted PPI/Arrow training path.
- PDBbind+ and the referenced bioRxiv SARS-CoV-2 binding supplement are blocked in this environment by login/subscription and HTTP 403, respectively. They should not be treated as available training sources until access is resolved.
- These CSV/builder notes are historical planning evidence only. The deleted PPI/STRING/MINT builders are not current training dependencies; future interaction work must first add an approved adapter and prepared semantic-record contract under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Training Logic

- The foundation-model objective should stay diffusion-only by default: corrupt eligible target residues at timestep `t`, predict the clean tokens, and compute denoising cross-entropy only on target/remasked positions.
- Every training example should carry explicit masks: `diffusion_target_mask` for residues that may be noised/remasked and receive diffusion loss, and `fixed_context_mask` for residues that remain clean and visible as conditioning context.
- Training should separate biological examples from target-mask construction. The prepared semantic record contains the full clean chains plus regions/metadata; the current immune grammar fixes type markers, relation/context objects, and conditioning chains, then creates token-level masks so eligible generated tokens participate in the diffusion objective.
- The current training-time data path lives under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`. It reads prepared semantic JSONL shards, renders the current immune grammar, and builds per-chain encoder tensors (`encoder_input_ids [batch, max_chains, chain_len]`). The deleted qwen data alias and its historical Arrow path, plus `GRAMMAR_V1.md`, are not active documentation.
- The BioSeq foundation model layer now starts at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`. It exposes `BioSeqNoEncoderDiffusionModel` for no-encoder training and `BioSeqEncoderDiffusionModel` for ESMC/ESM feature-conditioned training.
- The retained no-encoder model-layer design is a dense bidirectional diffusion transformer with ESM-family token embeddings, chain-local residue position embeddings (`position_ids_inner`), outer chain-index embeddings (`position_ids_chain`), timestep embeddings, RMSNorm, and SwiGLU blocks. The current immune grammar expresses entity roles through boundary/context tokens, so the decoder does not instantiate chain-role or task-type embeddings. It uses bidirectional self-attention for masked diffusion, not a causal/autoregressive language-model architecture. This model-layer design is separate from the deleted lightweight legacy backend.
- `BioSeqNoEncoderDiffusionModel.compute_loss` samples timestep noise from `diffusion_loss_mask`, replaces corrupted target residues with `<mask>`, predicts clean residue ids, and computes denoising cross-entropy only on corrupted target positions.
- `BioSeqEncoderDiffusionModel.compute_loss` uses the same diffusion objective and builds a per-chain ESMC/ESM `x_t` by applying the decoder corruption state to `encoder_input_ids`. ESMC/ESM returns token-level features for that diffusion state, and the downstream BioSeq denoiser performs joint multi-chain denoising over the concatenated decoder stream.
- Local ESMC encoder loading is implemented through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::load_local_esmc_encoder`. Because `transformers==4.48.1` does not recognize `model_type="esmc"`, `BioSeqEncoderDiffusionModel.from_esmc(...)` tries `AutoModel.from_pretrained(...)` first and then falls back to Biohub `esm==3.2.3`, mapping local Hugging Face-style safetensor keys into native `esm.models.esmc.ESMC` keys.
- ESM2 is also supported as the conditioning encoder via `--model-type esm2` / `BioSeqEncoderDiffusionModel.from_hf_encoder(...)`, which loads a local ESM2 snapshot under `/c20250601/mj/model_weights/esm2/*` through HF `EsmModel`. ESM2 hidden sizes: 8M=320, 35M=480, 150M=640, 650M=1280, 3B=2560.
- Current training uses the prepared-dataset loader and `GrammarBioSeqCollator` under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`; batch assembly remains dynamic because grammar rendering, padding, encoder reconstruction, and diffusion/MLM masking depend on the current batch/state.
- The formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. Deleted DDP examples, deleted Arrow sources, and deleted qwen `training` modules are not runnable alternatives.
- Foundation pretraining keeps the objective simple: the current immune grammar fixes type markers, relation tokens, and antigen / peptide / MHC-HLA context objects, and treats target structure tokens (including `<prots>`, `.`, `<protd>`) and target residues as diffusion targets. Antigen-conditioned receptor blocks include `<ab>` or `<nb>` inside `<prots>` to distinguish antibody vs nanobody design. Conditional capabilities such as chain completion, antigen-conditioned receptor generation, peptide design, and FR/CDR infilling are expressed as inference-time partial-mask prompts over the same current grammar, not as a separate deleted runtime view sampler.
- TCR grammar role resolution is role-first in the active renderer at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py`: an explicitly tagged single `tcr_alpha` or `tcr_beta` is never positionally paired with a peptide, antigen, or MHC context chain. The regression coverage is retained under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada`.
- To reduce multi-task imbalance, source mixture weights and optional future task schedulers should control the task distribution. The default physical batch itself is intentionally mixed rather than task-homogeneous.
- Fixed context chains, such as antigen in antibody-antigen generation, should not be remasked and should not receive direct diffusion loss. They still participate in attention or encoder conditioning, and gradients should flow through their encoder/connector parameters from the target-chain diffusion loss.
- In the BioSeq foundation loader, `full_denoise` means denoising all eligible target chains, not every chain unconditionally. Explicit `metadata["targets"]` is honored when present, but antigen, peptide, MHC, and HLA-like chains are fixed context by default and do not receive diffusion loss.
- Antibody-antigen and nanobody-antigen views should follow the receptor-design direction used by current antibody design work: fixed antigen can generate heavy/light antibody chains or VHH, fixed antigen plus one antibody chain can generate the paired antibody chain, and fixed antigen plus receptor FR regions can generate CDR regions. Antibody/nanobody-to-antigen inverse views are not part of the default task set.
- The `mhc_to_peptide_tcr` view covers the complementary TCR-pMHC case where MHC/HLA is clean fixed context and peptide plus available TCR chains are diffusion targets. This should be sampled explicitly during pretraining if peptide/TCR co-design or peptide-conditioned receptor learning is expected at inference.
- The `tcr_mhc_to_peptide` view covers peptide design from fixed TCR alpha/beta plus MHC/HLA. The `pmhc_to_tcr` view covers TCR alpha/beta design from fixed peptide plus MHC/HLA.
- FR/CDR views are region-driven, not antibody-only. They apply to TCR full-chain data whenever source adapters preserve TCR FR/CDR region annotations. For TCR-pMHC, fixed peptide/epitope plus MHC/HLA can be combined with fixed TCR FR regions to generate TCR CDR regions.
- ESMC/ESM feature-conditioned training is still diffusion training: ESMC/ESM2 features are extracted from `x_t` and condition target denoising, but the encoder is updated through the target diffusion loss rather than through a separate antigen reconstruction loss.
- The ESMC/ESM feature-conditioned path defaults to trainable encoder parameters. `freeze_encoder=True` is available only for ablations or bootstrap checks; the main foundation-model setting should fine-tune the ESMC/ESM encoder with the diffusion loss.
- Encoder conditioning **replaces** decoder residue embeddings with gathered encoder features (`use_condition_projection=False` by default). `decoder.hidden_size` is forced to match encoder latent dim for `--model-type encoder|esm2`; no-encoder ablations can use `--align-hidden-size-to-encoder`.
- The ESMC/ESM feature-conditioned path encodes each chain/sequence independently through ESMC on the current per-chain `x_t`. Implementation detail: `encoder_input_ids` has shape `[batch, max_chains, chain_len]`, is flattened to `[batch * max_chains, chain_len]` for a batched ESMC call, then reshaped to `[batch, max_chains, chain_len, hidden]`; residue features are gathered back to decoder token positions and replace residue embeddings before the multi-chain BioSeq denoiser runs.
- The current local environment keeps `transformers==4.48.1` for compatibility with Biohub `esm==3.2.3`; ESMC loading must use `BioSeqEncoderDiffusionModel.from_esmc(...)` or `load_local_esmc_encoder(...)` rather than relying on `AutoModel` alone.
- ESMC tokenizer loading for the active fusion path is implemented at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/esm_encoding.py`; do not refer to the deleted qwen `data/esm_encoding.py` alias.
- Remask schedules, block-diffusion masks, and generation masks must operate on `diffusion_target_mask`, not on all non-pad tokens.
- The grammar renderer guarantees at least one diffusion-eligible token per record (the non-fixed residues/structure/relation tokens), so `max_sequence_length` truncation cannot produce a zero-loss microbatch the way the old view sampler could.
- The current formal foundation-training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`. Historical `qwen3_vl_bioseq_*` YAML names and grammar-v2 runs may remain in audit records, but are not current submit instructions. Any new comparable ESMC/no-encoder experiment must be defined against the current immune-fusion line and recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` before submission.
- Batch-size tuning should be driven by logged CUDA peak memory, not by guessed limits. Keep effective batch approximately fixed while changing per-GPU batch and gradient accumulation. The old 8-GPU stage-1 settings (`8 x grad_accum 2`, `2 x grad_accum 8`, and `1 x grad_accum 16`) are historical baselines only; current settings must be profiled on the immune-fusion line before submission.

## Generation Task Survey and View Priority

- Survey date: 2026-06-14.
- Antibody generation should prioritize receptor-side design, not antigen generation. DiffAb samples antibody CDRs for antibody-antigen complexes and antigen-only settings, RFdiffusion antibody design explicitly focuses sampling on CDR loops while keeping a framework close to a specified therapeutic scaffold, IgLM supports full antibody generation and variable-span/CDR infilling, and paired antibody language models show that heavy-light pairing helps cross-chain feature learning.
- Current antibody/nanobody high-priority views are therefore: `antigen_to_antibody`, `antigen_to_nanobody`, `heavy_antigen_to_light`, `light_antigen_to_heavy`, `antigen_fr_to_cdr`, `antigen_single_cdr`, `heavy_to_light`, `light_to_heavy`, `fr_to_cdr`, and `single_cdr`. `cdr_to_fr` is useful for humanization/framework-design style ablations, but should not dominate early pretraining. `antibody_to_antigen` and `nanobody_to_antigen` are removed from the default task set.
- TCR generation should prioritize pMHC/epitope-conditioned receptor design and paired-chain completion. TCR-TRANSLATE generates antigen-specific TCR sequences from unseen pMHC inputs, TCRdesign uses antigen-conditioned generation with paired-chain coherence, and recent TCR structure work argues that both alpha and beta chain information matter for specificity and structure.
- Current TCR high-priority views are therefore: `pmhc_to_tcr`, `pmhc_fr_to_cdr`, `pmhc_single_cdr`, beta-chain context to alpha-chain completion, alpha-chain context to beta-chain completion, `fr_to_cdr`, and `single_cdr` when TCR FR/CDR annotations exist. The existing names `beta_epitope_to_alpha` and `alpha_epitope_to_beta` should be treated as context-to-chain completion views; a later cleanup should either rename them to `beta_context_to_alpha` / `alpha_context_to_beta` or add strict epitope-required variants.
- TCR peptide/epitope design views are biologically useful but should be lower-weight or task-specific until evaluation data is stronger: `tcr_mhc_to_peptide` is appropriate for epitope discovery/design from fixed TCR+MHC, while `mhc_to_peptide_tcr` is a co-denoising/co-design view rather than the first downstream validation target.
- The runtime view sampler has been removed: pretraining no longer samples conditional views per record. The conditioning directions above (antibody, antibody-antigen, nanobody-antigen, TCR, TCR-epitope, TCR-pMHC, PPI, generic) are instead realized as inference-time partial-mask prompts over the grammar token stream, where the prompt fixes the given context tokens and leaves the target spans masked.
- Sources: DiffAb (`https://github.com/luost26/diffab`, `https://openreview.net/forum?id=jSorGn2Tjg`), RFdiffusion antibody design (`https://www.nature.com/articles/s41586-025-09721-5`), IgLM (`https://doi.org/10.1016/j.cels.2023.10.001`), paired antibody language models (`https://doi.org/10.1371/journal.pcbi.1012646`), TCR-TRANSLATE (`https://www.nature.com/articles/s42256-025-01096-6`), TCRdesign (`https://doi.org/10.1093/bib/bbaf691`), and paired TCR alpha/beta structure analysis (`https://www.nature.com/articles/s42003-025-07708-6`).

- Ophiuchus-Ab training uses masked diffusion over heavy/light antibody tokens with AirGen-compatible timestep corruption.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py` emits `heavy_tokens` and `light_tokens`, each with `targets`, `regions`, and `chain_ids`, plus per-example `weights`.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/training.py` computes the heavy and light losses through `MultiChainOphiuchusAbModel.compute_loss`.
- Next migration quality step: add parity checks against `/vepfs-mlp2/c20250601/251105016/project/airgen/AirGen-Dev/src/byprot/models/lm/dplm_multichain.py` for `construct_x_t`, `compute_loss`, and generation mask transitions. The MINT module files themselves already match AirGen byte-for-byte except generated `__pycache__`.
- BioSeq training code must not import the old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/core/trainers` implementation.
- Historical Ophiuchus-Ab compatibility code, when independently audited, must remain limited to the exact Ophiuchus-Ab path; it must not be treated as the current immune-fusion entry. The deleted/lightweight `bioseq` backend, synthetic default data, and generic `transformers.Trainer` path are not current alternatives.
- Variable-length training (no fixed `(150, 128)` padding) uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/ophiuchus/collator.py::MultiChainDynamicCollator`, which pads each batch to the longest chain-1/chain-2 sequence with the real `<pad>` id. This is correct because the mint ESM2 backbone derives its padding mask from `tokens.eq(<pad>)`, and `compute_loss` splits logits by per-slot length. `OphiuchusAbTrainingCollator` (fixed length) is kept for exact-length reproductions.
- Multi-dataset prepared training uses `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/dataset.py`; source adapters and offline preprocessing use `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py`. OAS heavy is oriented to slot 0, OTS beta to slot 0, and prepared records preserve canonical chain roles.
- Historical Ophiuchus-Ab DDP validation used the former `examples/bioseq/train_bioseq_ddp.py` entry point; that entry point is now retired and deleted. New immune ESMC + LLaDA training uses `examples/llada/protein_pretrain_esmc.py` and the prepared `immune_llada` loader.
- Historical validation evidence included a single-process CPU smoke run and a 2-process `torchrun` (gloo) DDP smoke run loading all three datasets at variable length. This is not a fresh validation claim for the current documentation pass; current validation remains pending the main-agent TODO in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Initializing from generic ESM2 base weights is not part of the current antibody path. If it becomes necessary later, use the local ESM2 snapshots under `/c20250601/mj/model_weights/esm2` rather than adding a Hugging Face download dependency.

## Inference Logic

- The exact Ophiuchus-Ab inference path is retained as historical/independent compatibility provenance through `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` and aligned downstream scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream`; it is not the current immune-fusion training entry.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/bioseq/sample_ab.py` loads `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` through `MultiChainOphiuchusAbModel.from_checkpoint` when that historical path is explicitly audited.
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/common.py` is the shared inference layer for those aligned historical scripts; the current fusion evaluation boundary is the retained scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream` and public implementations under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar`.
- Heavy-to-light generation should fix the heavy chain and mask the light chain positions after any provided light-chain prompt. Empty light-chain input must still create masked light positions, not a fixed empty light chain.
- Antibody inference must not use `NoEncoderBioDiffusionModel`, `BioSeqDiffusionTrainer`, or any lightweight generic BioSeq backend.

## Weight Plan

- ESMC weights:
  - `/c20250601/mj/model_weights/esmc/ESMC-300M`
  - `/c20250601/mj/model_weights/esmc/ESMC-600M`
  - `/c20250601/mj/model_weights/esmc/ESMC-6B`
- ESM2 weights:
  - `/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`
  - `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D`
- ESM2 load checks should cover 150M, 650M, and 3B.
- ESM2 8M/35M should be downloaded but not included in the default load checks.
- ESM2 15B is optional and should not be downloaded in the current default task.
- Downloaded files should include Hugging Face/PyTorch-compatible weights, configs, tokenizer files, README files, and remote-code files; TensorFlow `.h5` duplicates are not required.
- Ophiuchus-Ab checkpoint is stored at `/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab/Ophiuchus-Ab.ckpt` and comes from `https://zenodo.org/records/18478480`.

## LLaDA Backbone Option (survey 2026-07-02)

- Goal: after the biological encoder (ESMC/ESM2), swap the decoder from the in-house `BioSeqDiffusionDecoder` (`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`) to the **LLaDA architecture** as a comparison backbone. The encoder + per-chain-encode + gather pipeline stays unchanged; only the denoiser transformer changes.
- In-repo implementation: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/llada/models/modeling_llada.py` (`LLaDAModelLM` / `LLaDAModel`, HF `PreTrainedModel`-compatible; config in `configuration_llada.py`).
- Same nature as the current decoder: LLaDA is a **bidirectional masked-diffusion transformer** (`is_causal=False`, `get_bidirectional_attention_bias`), LLaMA-style (RMSNorm / RoPE / SwiGLU). This matches the existing bidirectional masked-diffusion decoder, so it is a backbone swap, not an objective change.
- Integration point (reuses existing encoder path): `LLaDAModelLM.forward` natively accepts `inputs_embeds` (forwarded as `LLaDAModel.forward(input_embeddings=...)`, `x = wte(input_ids) if input_embeddings is None else input_embeddings`). Wiring:
  - encoder per-chain encode -> `gather_token_condition` -> `token_condition [B, S, E]` (unchanged);
  - build `inputs_embeds [B, S, d_model]`: residue positions = encoder features, non-residue (structure/relation/special) positions = `llada.wte(x_t)`;
  - `LLaDAModelLM(inputs_embeds=..., attention_mask=...)` -> logits -> `compute_masked_cross_entropy` on corrupted positions only.
  - Requires `E == d_model` for pure replacement, or add a condition projection.
- Key differences vs the in-house decoder (design decisions to lock before training):
  - **Position**: LLaDA uses **RoPE on flat sequence positions**; the in-house decoder uses **learned absolute** `position_ids_inner` (chain-local) + `position_ids_chain` (chain slot). The first LLaDA variant loses chain-aware absolute positions and relies on flat RoPE + grammar boundary tokens; chain-aware / group-reset RoPE is a later enhancement.
  - **Timestep**: LLaDA does **not** take `t` as input (RADD proves masked diffusion needs no timestep). The in-house decoder now replaces only the token-identity embedding at residue sites and **adds** inner-position / chain-slot / timestep on top; when moving to LLaDA the timestep addition can simply be dropped (RoPE supplies position).
  - **Loss**: `LLaDAModelLM.forward` does not compute loss (labels only warn); keep the external `compute_masked_cross_entropy` (masked positions only).
  - **Vocab**: LLaDA defaults to a text vocab (`vocab_size=50257`, `mask_token_id=50256`, `embedding_size=50304`). Using the architecture from scratch requires reconfiguring `vocab_size` / `mask_token_id` / `embedding_size` to the grammar vocab (ESM2=49 / ESMC=80).
  - **`input_emb_norm`**: LLaDA can scale input embeddings by `sqrt(d_model)`, which rescales the encoder condition (related to the `condition_norm` scale discussion). Keep it fixed/known across A/B runs.
- Config constraints: MDM usage asserts `rope=True`, `alibi=False`, and `use_cache=False` in `LLaDAModel.forward`.
- Weights: `/c20250601/mj/model_weights` currently has **no** LLaDA checkpoint. From-scratch training with the LLaDA architecture needs no weights. Loading pretrained LLaDA text weights is not directly reusable — the text vocab and residue embeddings do not match the biological grammar vocab.
- Open decisions: keep chain-aware positions (grouped RoPE) or accept flat RoPE; whether to re-inject timestep at residue sites; whether to add a condition projection (`use_condition_projection`) instead of pure replacement.

## 2026-07-21 Public CDR3β generation comparison architecture

- Track A is a model-output benchmark, not an architecture comparison. The common adapter contract is `generate(peptide, mhc_allele=None, mhc_pseudosequence=None, num_candidates=1000)` and the canonical row schema is defined in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/track_a.py`.
- TCRT5 conditions on peptide plus the 34-aa MHC pseudo-sequence; GRATCR and TcrDesign condition on epitope. TcrDesign Track-A ranking rows are invoked only through `tcrdesign_G.py -mode beta`; V/J prediction, TcrDesign-B, and the full binding-filter pipeline are not part of the first-stage comparison.
- After the β benchmark, a non-ranking TcrDesign addendum runs the released conditional chain `epitope -> generated β; (epitope, generated β) -> generated α`. It covers all 14×1,000 generated β conditions with 10 official α beams each. The batched adapter preserves the released architecture/checkpoint/tokenization/beam search and only avoids repeated model loads. Since the release contains no runnable alpha-reference or αβ pairing evaluator/checkpoint, the 140,000 paired rows are qualitative and no custom alpha metric is added.
- TCR-epiDiff is guarded as `reconstruction_only` because its released sampler starts from a noised real TCR. A reconstruction path must never be adapted with test references and presented as de-novo generation.
- The local BioSeq comparison is frozen to the only retained 7-layer checkpoint in the requested `11xxxx` range, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt` (validation loss `0.4430236165889635`, SHA256 `e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`). It is also the retained Top-K checkpoint with the lowest recorded validation loss.
- BioSeq Track-A generation is epitope-only and does not consume MHC allele or pseudo-sequence. The fixed reproducibility protocol is seed `42`, batch size `100`, `32` denoising iterations, temperature `1.0`, and `gumbel_argmax`; rank is deterministic seeded emission order and `raw_score` remains empty because this sampler does not expose a comparable candidate score. Raw sequences receive no anchor repair.
- Under that protocol, BioSeq produced 14,000 candidates with `ValidRate=0.998786`, `UniqueRate=0.999714`, no exact or Recovery>=90% hits, one GIANA reference-cluster hit, and mean cross-epitope Jaccard `0.000044`. This indicates strong syntax/diversity but weak reference-neighborhood recovery under the current epitope-only generation protocol; it is not evidence by itself about binding or the architecture in general. The focused audit is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/bioseq_step117000_report.md`.
- All models are evaluated after generation by one external validity/exact/recovery/GIANA/Jaccard implementation. Internal model scores may be retained as `raw_score` but cannot determine the cross-model result.
- TCRT5 is the one rank-sensitive adapter: rank and `raw_score` must use cumulative generated-token log-likelihood (`compute_transition_scores(...).sum()`), matching the paper's rank-cutoff protocol, rather than Hugging Face's length-normalized beam score. The candidate set is independent of this bookkeeping correction.
- Keep the paper-native TCRT5 recovery diagnostic separate from the common Track-A metric. The paper first chooses a same-length reference and uses position identity/Hamming (with an edit-distance fallback); the common leaderboard remains the user-specified normalized Levenshtein definition. Never compare their counts without a definition label.
- The complete protocol and completed local result are frozen in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`.

## 2026-07-22 MINT headline PPI evaluation lock

- The user-selected model for the formal HumanPPI, YeastPPI, and Gold-standard PPI evaluation is the immutable BioSeq `step121000` snapshot at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/grammar_v2_esmc300m_integrated_llada_7l_step121000/best.pt` (SHA256 `51f0eee5fa7b127a0487a2ea8fdbc322bda6f6908ff2662087bd87d4980d4613`). Later/final checkpoints are not substituted.
- Only the three requested classification tasks are in this run: HumanPPI (primary metric Accuracy), YeastPPI (Accuracy), and registry task `Bernett`, which is the MINT paper's Gold-standard PPI task (AUPRC). MutationalPPI and SKEMPI are explicitly outside this run.
- Data come from the pinned public MINT `prepare_data.ipynb` rebuild under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`, with no row cap. HumanPPI uses 26,319/234/180 train/validation/test rows and YeastPPI uses 4,945/95/394; both exactly match the paper counts. Gold-standard uses 163,192/59,260/52,048; its validation/test counts match, but the public notebook has 173 more training rows than the paper's 163,019, so its local result must be marked non-paper-exact.
- Each pair is rendered as one contextual two-chain `ppi` grammar record and pooled by one global mean over final decoder residue states; `--sep_chains` is not used. Each chain is deterministically left-truncated to the BioSeq native maximum of 1,024 residues. This differs from MINT baseline runs that can use a 2,048-token cap and must remain explicit in provenance.
- The downstream probe follows the paper Methods rather than the inconsistent public task defaults: two linear layers with hidden width 640, ReLU, dropout 0.2, AdamW at `1e-4`, 100 epochs for all three tasks, best validation metric selection, batch size 16, and three seeded repetitions. The public code's input-width hidden layer and Gold-standard 30-epoch setting are not used for the headline comparison because they conflict with the paper's stated 640/100 protocol.
- Before formal submission, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/eval_mint_ours_step121000_three_ppi_profile.yml` profiles worst-case 1,024+1,024-residue pairs on the same `ml.pni2.3xlarge` A100 flavor. Formal extraction batch size is selected from that measured profile and recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- The formal run is complete. BioSeq `step121000` `[C]` obtains HumanPPI Accuracy `0.677778±0.018144`, YeastPPI Accuracy `0.602369±0.007846`, and Gold-standard PPI AUPRC `0.592089±0.001405`. The best paper `[P]` values are MINT `0.879630±0.006929`, `0.686971±0.010634`, and `0.687157±0.003256`, giving gaps `-0.201852/-0.084602/-0.095068`. Human/Yeast use exact public fixed splits; Gold remains `paper_comparable=false` because the public notebook has +173 training rows. The complete report is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_ours_step121000_official3ppi/REPORT.md`.

## 2026-07-22 Retained-checkpoint CDR3β diagnostic policy and step189000 result

- The canonical all-14 BioSeq Track-A row remains the frozen `step117000` run. Later retained checkpoints are evaluated under an isolated `checkpoint_comparison/<run_name>` root and must not be appended to the canonical leaderboard unless a complete all-target replacement run is explicitly selected.
- The BioSeq adapter now validates the checkpoint manifest and derives `BioSeq-7L-step<manifest_step>` dynamically. Its default checkpoint and default label remain step117000, so existing reproduction commands and canonical rows are unchanged.
- A checkpoint comparison must keep the same generation controls (seed `42`, batch `100`, 32 iterations, temperature `1.0`, `gumbel_argmax`, no anchor repair) and the same external Track-A evaluator. For a one-target diagnostic, cross-epitope Jaccard is undefined; candidate-set overlap across checkpoints may be reported separately and must not be labeled cross-epitope specificity.
- Under this policy, step189000 generated 1,000 RVR candidates: ValidRate=`0.998`, UniqueRate=`1.000`, ExactHit@1000=`0`, ReferenceRecall=`0`, Recovery90Hit=`0`, median best recovery=`0.600000`, maximum recovery=`0.800000`, and GIANAHit=`0`. Relative to step117000, median recovery improved only from `0.588235`; this is not a meaningful target-specificity improvement.
- This run remains an epitope-only conditional generation test. A future MHC-aware experiment requires a training-consistent MHC conditioning representation and is a separate model/data change, not an inference-time field injection into this checkpoint.
- The complete focused artifact is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/checkpoint_comparison/bioseq_step189000_rvr_epitope_only/report.md`.

## 2026-07-23 TCRT5 full-evaluation method lock

- The paper-facing evaluator is now a separate evidence layer, implemented in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/tcrt5_full_eval.py`. It does not replace the common Track-A leaderboard: author top-20 `K=100`, sparse13/RVR `K=1000`, and current all-14 `K=1000` remain distinct protocols.
- Paper-native recovery is a two-stage operation: select the closest same-length reference by Levenshtein distance, then compute positional Hamming identity to that selected sequence. Only the no-same-length branch uses reference-length-normalized Levenshtein identity. Track-A normalized-Levenshtein recovery remains a separate metric and must keep its definition label.
- The paper's mAP is implemented literally as the pMHC macro mean of `mean(P@1,…,P@K)` after cumulative model-log-likelihood ranking. A value is paper-compatible only if the candidate block retains those scores. Current TCRT5 satisfies this condition; author main-table lists, BioSeq, GRATCR and TcrDesign do not, so their ordered AP values are diagnostics rather than calibrated cross-model mAP.
- Diversity decisions are fixed to natural-log units: positional delta entropy is `H(generated)-H(reference)`, with gap-inclusive and residue-only tables both retained; k-mer Jensen–Shannon divergence uses natural logarithms for `k=2..12`, so its upper bound is `ln(2)`.
- Evidence is three-tiered: (1) independently recomputed and checked against released sequences; (2) definition-complete but blocked from exact paper comparison by missing greedy hypotheses, likelihoods or final figure aggregation; (3) explicit extensions such as Hit@K/rank distributions. Reports must never collapse these tiers into a single “reproduced” label.
- OLGA biological plausibility requires both the positive fraction and the positive `log10 Pgen` distribution. In the current all-14 run, BioSeq-7L-step117000 has `0.979786` positive Pgen but mean positive log10 Pgen `-17.925285`, compared with TCRT5 `1.0/-6.950375`; therefore a non-zero-only validity claim is insufficient.
- The canonical comparison report is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`. Its explicit non-reproducible boundaries—main-table mAP and Fig.4 Pgen moments—are part of the method contract, not missing implementation work.

### 2026-09-13 TCR scoring re-audit

T2A now preserves exact cosine edge thresholds through clustering and artifact serialization, uses the audited per-method Fig 3A retention anchors, and records unreachable/mismatched retention points instead of calling every nearest point aligned. The predeclared local tolerance is 0.02 retention, not a paper-native criterion or proof of identical inputs. T1–T4 receive a separate reduced GPU execution/scoring gate, distinct from input-only gates, scientific decontamination, and quality evaluation. Protocol owner: [T2 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md); current decisions/status: [TCR audit §9.22](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md). No checkpoint, training corpus, shared AB sampler, or baseline performance result is changed.


## 2026-09-24 TCR v2 downstream protocol

- [Implementation](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/tcr_generation_v2.py): unconditional / peptide / pMHC-conditioned CDR3 generation uses a full 167-slot beta canvas, EOS termination, then the existing benchmark ANARCI/regex-fallback extractor. Alpha is an explicitly unknown fixed-X partner. No CDR3 length prior or reference receptor enters generation.
- Single/pair CDR3 infill is a distinct known-length window task. Single inputs reuse the training completion profile; paired inputs use supplied frameworks. Decoder-space canonical-AA allowlists prohibit internal chain EOS in all sampling phases.
- Every generation attempt is retained, including termination/extraction failures. These outputs must not be compared as the old v5 core-only length-prior protocol. The historical pinned v5 experiment runner remains pinned; new runs use the checkpoint-selectable TCR CLI.

## 2026-09-24：抗原条件 paired AB benchmark 决策

新增生成、binding、实验 KD/pKD 三条目标；范围由 [benchmark README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 定义。用户明确不测 VHH，并要求每个 benchmark 使用其作者发布的比较代码。按 [AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md) 先核验任务、数据/划分、版本和权重，排序与绝对回归分列；不把自写 head/scorer、旧 FLAb/MINT Ridge 或模型原仓库直接当作 benchmark 作者实现。原代码不可得则保留 pending，不伪造正式成绩。

## 2026-09-24T03:54:43.033669+00:00 AB-AG 范围修订

按用户要求完全排除 FLAb2，MAGE 保留并推进作者原脚本。范围以 [README §0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md) 为准，后续资源集中于显式抗原输入的候选；无需再解决 FLAb2 代码和依赖。

## 2026-09-24T04:07:06.335060+00:00 S²ALM benchmark / baseline 可用性决策

S²ALM 的 antigen 特征回归与本项目方向相关，但公开产物不足以作为可执行原代码 baseline。其固定 HER2 分类和 CDR-H3 infilling 不用于补本轮三路主任务。BioMap/A2binder 保留为可追踪官方实现的回归候选；ΔG 与 AlphaSeq 标注不能无核验合并成实验 pKD。详见 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)。

## 2026-09-24T04:48:58.155591+00:00 BioMap 回归候选深化

用户要求继续追踪 BioMap 论文与 baseline 代码；接入判断与待解决条件见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md)。保持原代码要求；未新增自写 head、scorer 或数据划分，未运行模型。

## 2026-09-24T05:06:03.673383+00:00 BioMap 首轮执行决策

用户授权先跑 BioMap，采用 A2binder 发布默认单 seed 配置及原数据，运行状态见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。保留已知 H/L 数据问题并披露；不引入自写模型或修订后的 split，不将 ΔG 转 pKD。

## 2026-09-25 TCR design VDJdb50 Method

- User selected one existing best usable LongAB binding checkpoint: v5 140000, unseen independent AUROC 0.5129; frozen backbone + mean of existing five heads, no retraining. Inputs are provided alpha/beta variable domains and peptide; HLA is ignored. GPU replay of 32 archived features/scores gates source compatibility.
- Only the expanded 50-target whitelist; IgGM2 50, each of ProteinMPNN/ESM-IF1/Rosetta 500, native 50. Chain A=alpha, B=beta verified against PDB and index; no framework substitutions. Equal-target means; no best-of-N comparison.
- Protocol/script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_vdjdb50.py`; artifacts: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925`. `designs.csv` one row per design, `panels.json` deterministic up-to-10 length-matched cross-peptides from these 50 only, separate same-HLA panel, no verified negative labels. CDR novelty uses design train/public sequence reference banks, not another evaluation cohort; OLGA uses unmodified anchored alpha/beta junctions. Public reference includes native targets.

## 2026-09-25 Correction: external released binding model only

User rejected use of our trained v5 140000 and its heads. Prior selection is superseded; task t-20260925203709-4qb26 cancelled before inference. Sequence/Pgen data stay valid. External model selection must check peptide support: local TCRconv fullAB predicts fixed epitope labels, not arbitrary novel peptides. No replacement binding score yet. Scope remains the same 50 targets.

## 2026-09-25T12:48:19.071235+00:00 External TCR-H VDJdb50 evaluation

Selected external TCR-H using pre-existing unseen-independent AUPRC (0.5267), highest among eight reproduced released external methods, before inspecting design scores. CDR3beta + peptide only; fullAB TCRconv cannot score arbitrary novel peptides. No in-house weights or fitting. Five benchmark-published checkpoint scores averaged. Official test-pool descriptor scaling retained in one shared deduplicated pool (all methods/native/decoys), never separately per method; scores are cohort-dependent and uncalibrated.

Script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_external_tcrh.py`. Outputs: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925`.

## 2026-09-25T14:44:49.171625+00:00 epiTCR original-model direct binding evaluation

User-specified epiTCR replaces TCR-H as primary binding readout for this design evaluation. Use original released CE model and official predict.py, one checkpoint, no fitting/ensemble. Inputs are target peptide plus beta CDR3 core without conserved C/F anchors, matching original model convention. Decision is predict_proba >=0.5 as implemented upstream. Report predicted binding fraction and score, not experimental hit rate or accuracy. Small native within-whitelist cross-peptide control diagnoses high positive-call rate; no threshold tuning.

Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_original/REPORT.md`; script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_external_epitcr.py`.

## 2026-09-25T14:50:26.032622+00:00 epiTCR training-overlap audit

Published original CE train: 321,365 rows; one target epitope NLVGMVATV (vdjdb2025_036) seen, zero exact native CDR3beta/pair matches. Full original source corpus: 3,255,086 rows; five target epitopes seen (ALWALPHAA, NLAPMVATV, NLVAMVATV, NLVGMVATV, NLVPMVASV), three native CDR3beta matches, zero exact pairs. Full corpus includes test data and is not solely training. All 1,550 generated beta CDR3s and their target pairs have zero exact matches in all eight audited banks. NM2025 five retraining folds: zero target-epitope or pair matches, 1–2 native beta-chain matches per fold. Original published train also contains one-edit neighbors of 32 target epitopes and 40 native beta CDR3s, considered separately. Exact checkpoint training-row linkage cannot be proven because original pickle lacks a training manifest. Current model is the original published checkpoint, not a local or NM2025 retraining.

Existing predictions regrouped without inference/retraining/threshold changes: exclude 1 train-seen epitope ->49 targets; exclude 5 corpus-seen ->45; exclude train epitope edit distance <=1 ->17. Every method remains 100% predicted binding in every subgroup, so exact target exposure alone does not resolve the broad positive-call issue. No modification to the 50-target whitelist. Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_original/overlap_audit/REPORT.md`; script `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/audit_epitcr_overlap.py`.

## 2026-09-25T14:55:37.783986+00:00 NM2025 epiTCR first-sample 50-target evaluation complete

Current primary evaluation uses NM2025 released epiTCR AS five-fold retrained checkpoints (not original CE and not local training). Preserve fold-specific scores; average then >=0.5 binary decision. Each baseline fixed design_idx=1 per target, IgGM2/Native=0; same 50 whitelist, no score selection. Full-sample same-checkpoint inference is diagnostic for composition versus batch-size effects. Retrained protocol uses anchored beta junction + peptide, published BLOSUM 19*20+15*20; original CE/core results are historical and cannot be conflated. Corresponding train-fold audit: target epitopes and pairs absent in all five folds; selected generated CDR3beta absent; 1-2 native beta chains per fold overlap. This is exact overlap evidence, not near-neighbor exclusion.

Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_nm2025_first50/REPORT.md`; script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_nm_epitcr_first50.py`.

## 2026-09-27 05:34Z Fixed-v2 downstream input adaptation

Explicit `fixed-v2-fv` probe protocol uses ANARCI-verified variable-domain boundaries; m396 additionally rejects any mutation in discarded WT flanks. The training canvas and training recipe are unchanged. Protocol and comparisons are owned by `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md`.

## 2026-09-27T05:53:11.583710+00:00 Version naming and GDPa1 interpretation

Use V6 for immune_v6_binding_only models in reported results. Existing v2 identifiers describe the fixed-canvas EOS architecture and remain in paths. GDPa1 audit does not establish a single causal defect; source balancing, training budget and EOS/canvas differ from V5. See `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/diagnostics/gdpa1_v6_20260927/REPORT.md`.


## 2026-09-27 Gradient accumulation normalization contract

Fusion forward accepts `**batch` but does not normalize using Hugging Face `num_items_in_batch`; `compute_loss` returns a microbatch loss. Set `model_accepts_loss_kwargs = False` unconditionally in FusionTrainer so Trainer applies its gradient-accumulation divisor once, independently of source sampling. With accumulation=8, each microbatch contribution is divided by 8 for both balanced and source-frequency sampling. Actual SGD-step regression tests cover both sampling modes at accumulation 1/8, checking gradients, parameter updates and reported training loss. Evaluation loss calculation is unchanged. Running processes require restart/resume to load this fix; old checkpoint optimizer history remains affected. This correction alone does not establish the cause of GDPa1 performance differences.

## 2026-09-27 AB-AG sequence v1

已实施用户确认的三任务四子集框架和 Ours zero-shot + 冻结骨干探针入口；固定无 ESMC step150000。保留历史调研，详细决策见[研究文档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ANTIGEN_CONDITIONED_AB_RESEARCH.md)；协议见[任务手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。本模型派生协议不冒充作者 baseline 原始复现。

### 2026-09-27 AB–Ag 接口实现

按用户“先写评测接口”要求，拆分模型输入、推理适配器和离线指标；Ours no-ESMC150000 zero-shot 适配沿用既有评分方法，外部 baseline 仍要求作者代码。契约见 [ab_antigen/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/README.md)。本轮未新增 GPU 作业。
# 2026-09-30 no-ESMC v3 / immune_v7 execution contract

- User authorized implementation: representation and generation together; scratch training; staged small ablations; no separate length head.
- A0 clean-data legacy control; A1 compact residue/grammar vocabulary; A2 semantic chain roles and observation scopes; A3 chain-local RoPE plus same/different-chain attention bias; A4 observed-only fragments and a single predicted `<protd>` per target chain/fragment.
- Preserve every observed residue, including junction C and terminal F/W (currently stored at FR3/FR4 boundaries). Never infer anchor presence from first/last letters. Source core-only records remain explicitly core-only when original anchors cannot be recovered; never fabricate anchors. Core-based deduplication does not alter model inputs.
- Unknown-length target capacity is independent of reference length (full chain 256 slots, fragment 64, including terminal). Tail slots remain attended MASKs, never labels. True length/regions may select labels, not inference features. Known inputs use observed lengths.
- Freeze all benchmark banks, clean across every source before resplitting, audit both exact and configured near matches, and require complete audit before training. Incomplete searches, missing H3 annotations and unknown provenance are NOT clean certificates.
- New data target: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2`. Keep existing weights/data and unrelated working-tree edits intact. New model/data/cache versions must not be loaded as legacy v2.
- Architecture pilots: common eligible semantic records, seed 42, global batch 256, LR 1e-4, warmup 500, initial 100-step numerical/sampling check, then 1,000-step pilot and 10,000 total; only validated top two proceed to 50,000 steps with seeds 42/43/44. Raw-residue/end losses and GPU-hours reported separately. No real-corpus GPU training before data and memory-profile gates; synthetic memory profiling is a separate prerequisite.

- Implementation detail: A0 is a scratch structural control in the common v3 runner, preserving the historical full vocabulary size/token IDs, 167/135 canvases, global RoPE and full-softmax CE. It is not an old checkpoint converted to v3. A1–A3 preserve full-softmax CE; A4 restricts target slots to residues and `<protd>`.
- Validation uses fixed per-source subsets (up to 2000 rows/source) and equal-source macro means; token-weighted residue/end losses and counts are separate. Training sampling remains natural fixed-seed for this architecture suite. The existing full-source balanced trainer is unchanged.
- Masked benchmark strings are not mistaken for native full chains: source-provided reference/core evidence can discharge missing-H3 coverage only when every occurrence is covered by an already-frozen protected core. Wrong-chain/truncated/unclassifiable cases remain explicit pending evidence.

- Measured initialized parameter counts: A0 269,759,232; A1 75,587,328; A2 75,601,152; A3/A4 75,601,176. The Transformer width/depth remains 768/8; compact vocabulary removes about 194.2M embedding/head parameters. This is an implementation/size result, not yet a downstream-quality result.

- The first v7 export remains diagnostic while 264 H3 cases require chain/completeness adjudication. The suspected missing native reference was a repeated CSV header, confirmed from the source row; all actual native full references are protected. Rebuilding the registry produced byte-identical banks, so the original frozen audit remains valid.

- Corrected cleaning protocol `v7.2_bidirectional_containment`: exact antigen substring matches must satisfy the same >=80% bidirectional coverage as antigen homology. Short shared strings alone do not establish an antigen family; parent-protein novelty still requires explicit provenance. The aborted first export and its over-removal statistics are excluded from all experiments.

- 2026-09-30 用户补充执行顺序：本地toy/单卡闲时短测 → 真实数据100步数值/采样诊断 → 通过后1000步pilot → 后续10000步架构比较。100步早停保持10000步cosine总周期，避免用短周期调度替代原方案。toy记忆与短测仅验证实现可训练，不能作为模型泛化或架构排名证据。

- 2026-09-30 记录要求：每个阶段的假设/单一改动、基线、公平比较条件、验收条件与实际结论同步到本方案和PROJECT_PROCESS；方案或预算改变必须标明生效顺序。保留失败/负结果和更正，不以toy记忆或未完成运行作效果结论。当前100步诊断优先于早期1000步直接试跑安排。


## 2026-09-30 新下游范围对foundation数据的约束

新增UpTCR及完整few-shot支持/查询池后，旧v7清洁结论不能外推。foundation预训练排除和下游合法适配分开：原官方UpTCR划分/标签继续保留；严格预训练暴露审计同时看受体、epitope、联合配对，并统计只有表位命中的记录。短序列增加全量编辑相似度筛查，避免只靠linclust推断无近邻。当前证据、阈值、来源规范化和待办统一见 [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) 的“新增TCR下游暴露审计”。

真实数据A0–A4 pilot必须同时通过旧v7门禁和新scope门禁。已训练v6 checkpoint的暴露不能靠后来清理数据消除；新scope未通过前不称之为receptor/epitope-unseen foundation评测。相同epitope在合法few-shot support与query之间出现是任务定义，不能当作需要删除的官方划分错误。


2026-09-30补充：新范围全量审计已结束，数据不是clean；候选规则会显著改变OTS保留量与小来源valid规模，导出最终实验数据前需要结合家族隔离复核验证设计。A0–A4必须共享最终数据与评测口径，不能把数据清理效果误归因于架构。具体证据与候选清单见 [完整审计记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#tcr-exposure-20260930)。


## 2026-09-30 架构实验执行队列更新

按用户决定，架构修改及测试验证统一在queue012执行，停止c20250601对应排队任务。先使用已完成的合成profile证据，真实数据pilot仍须通过当前数据范围的清理/审计门禁；队列变更不代表数据门禁解除。资源与任务记录见 [PROJ_GUIDE.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md) 和 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。


## 2026-09-30 queue012 小规模探索配置已准备

执行分两层，文件与命令见 [queue012 实验包说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_explore_queue012/README.md)：先用 A0–A4 的生产尺寸模型各做 20 步相同人工样本优化诊断，再在最终清洁数据上各做 10 步 pilot 检查训练、保存和验证管线。前者恒定学习率 1e-4，后者保持原正式 scheduler，二者不直接比较 loss。真实数据清理与证书仍未通过，不以合成样本替代真实效果证据。合成20 实测已全部写入该 README——A0–A4 **全部完成/Success**（A0 `14.093841 → 0.414990`，A1 `5.837228 → 0.328653`，A2 `5.164647 → 0.252067`，A3 `5.169685 → 0.185864`，A4 `4.774529 → 1.119085`）；显存 profile `t-20261001020115-zvt6f` **Success**，推荐 bs A0=16、A1/A2=256、A3/A4=32。`02_real_data_gated` **从未提交**。这不是下游效果证据。

本轮不新增架构组件：A0→A1 词表、A1→A2 角色/范围 embedding、A2→A3 链内 RoPE/跨链偏置、A3→A4 布局/结束目标的既有累加筛选。A3/A4 各包含多个变化，筛选后再决定组件级消融。不同词表、参数量和 loss 定义意味着总 loss 不可直接作为架构排名；同数据下的下游评测仍是后续判断依据。真实 C/F/W 保留，A4 继续学习每链一个 <protd>。

用户在有权限的机器手动提交；本机只生成与验证配置，不再尝试 queue012 submit，也不回退 c20250601。合成 20-step A0–A4 **已全部完成/Success**，不要再提交同一组；`02_real_data_gated` **从未提交**，门禁未解除前不要提交。


## 2026-09-30 扩大 scratch foundation model 方法探索

新增持久记录 [研究与方法记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)，覆盖18项视觉/序列/蛋白工作及7份官方实现核查。首轮落实12个有实际实现的变体，以A4为底座，分别测试多粒度遮蔽、链目标配比、优化归一化、计算token、链内局部先验、输入条件调制、表征联合学习、来源分布和有向角色交互。完整创新假设、优先级、论文差异、二阶段候选以该文档为事实源。

先做每方法显存profile及合成验证，再用同一最终清洁数据做200步pilot。新suite固定global256/seed42，warmup20/horizon2000；必须使用R00对照，不能直接与旧pilot比较总loss。共享full-mask AA/end诊断，额外目标仅训练启用；最终仍以独立valid选型、多seed和多类下游任务判断。REPA外部教师、完整JEPA、MAE非对称编码、MoE及SSM暂未实现，不将其写成可用架构开关。执行入口：[24 个 YAML / 提交说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)。


## 2026-09-30 研究验证预算升级（替代200步初筛方案）

依据用户反馈，3-step显存扫描与20-step合成诊断仅作为工程前置检查；真实方法比较升级为1000步统一初筛、R00及选中候选累计5000步。使用相同global256/seed42/lr1e-4/warmup100/horizon10000，固定每100步的验证与checkpoint。验证覆盖25/50/75/100%遮蔽及来源宏平均AA/end，不仅看总loss。可从完整checkpoint继续，不能更换学习率日程后伪装同一条学习曲线。

已实现代码/配置更新，证据、预算充分性的判断和首版显存表集中于[研究记录及验证预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)，入口为[36个YAML与续训说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)。没有固定step数能保证泛化有效；若学习曲线/跨来源排名仍不稳定，保留多seed、更长训练及正式下游验证需求。旧200步/warmup20/horizon2000记录保留为历史，不能和新版直接续训或拼表。


## 2026-09-30 八卡持久实验环境

用户将先前六卡改为单实例八卡非闲时；c20250601 / ml.pni2.28xlarge / RoleReplicas=1。R00–R11先独立profile与合成工程验证，再用通过清理门禁的数据做1000步初筛和优选5000步续训；均可复用同一实例。当前旧合成20步不能确定最终架构，真实训练数据清理仍未完成。

运行与控制说明：[八卡工作池](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/README.md)；实际结果：[持续记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)。


## 2026-09-30T20:19:38.413413+00:00 第二轮架构研究已落地

两份扩展文献/官方代码调研已完成；R12–R19包括同版本对照、head输出gate及固定0.5对照、LayerScale及固定0.1对照、前4后4链融合、radius32局部/全局交替、Value residual。115项集成测试通过，8个GPU profile通过并运行100步四档mask诊断。现阶段不宣称下游增益，真实训练还要通过清理证书。

详细研究、测试和运行证据：[架构扩展v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。


## 2026-09-30T20:24:51.052023+00:00 batch试训前补充条件

晋级检查已增强，63项通过；新bias缓存等价原型82项CPU通过，尚未GPU部署，下一版本必须新profile。当前toy短链可能放大R18局部窗口对MASK尾部的优势，真实验证按长度/来源分层后再选架构。详见 [v2记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。

原型完整[等价验证与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_BIAS_CACHE_VALIDATION.md)已完成。它属于实现成本优化，下一release各候选及R12须统一使用并重新测显存；不能作为独立生物学架构增益。20:28Z新8项尚为45/100步，候选选择继续等待完整诊断与真实分层验证。


## 2026-09-30T20:53:52.568809+00:00 以现有LLaDA270主干为基础的持续探索

参数名义270M主要为大词表，真实主干d768/L8/H12/MLP3072与当前A4一致；下一轮先保持这一基线及蛋白小词表。R12–R19的100步工程诊断全部成功，但排序变化且toy饱和，R20–R27必须扩大人工heldout长度/链组合并采用独立源码/显存测量。等价bias缓存生产候选80项CPU通过；人工诊断与真实1000/5000步分别记录，不据toy确定生物学最优。

[持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md) · [270基线审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA270_BASELINE_AUDIT.md) · [100步结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R12_R19_100STEP_RESULTS.md) · [数据准备状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_DATA_READINESS.md)


## 2026-09-30T21:17:56.103865+00:00 Refined扩展验证与真实训练边界

保持现有LLaDA主干，R20–R27通过等价bias复用减少实现开销；新GPU实测全部micro32，局部/链隔离方法因两种bias略占更多显存。扩展人工576/72、18组长度/链组合，先profile+25步流程检查再500步heldout诊断；步数按曲线与可识别性调整，最终候选依据真实隔离语料的1000/5000步和多seed。当前仅新25步运行，未做新真实训练。

[Refined配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md)


## 2026-09-30T21:25:58.529062+00:00 八项500步人工诊断已开始

R20–R27的25步广输入检查全部succeeded，累计28成功/0失败。各项约718秒，四档mean loss为1.768–1.908、AA约30.06%–36.38%；end accuracy仍0，全mask end只占132/17996≈0.73%，须独立观察，不能仅看总loss。局部/全局方法在周期8人工任务上的优势不等于生物学优势。

已实际执行经74项测试验收的入队助手 synthetic500 --apply，8项均被worker领取运行。新500步initial的18组指标与同方法25步initial完全一致，fixture、39源码SHA、global256也一致，确认为同协议重新初始化。500步是更广人工heldout诊断，不是初步真实语料训练。按25步实耗粗估约4小时，准确耗时以运行结果为准。

[启动证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/synthetic500_started.json) · [配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md) · [持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。目标保持active，真实数据清理与1000/5000步尚未完成。

数据覆盖仍在推进：替代编号工具给出了VNAR候选，但与原ANARCI对部分“源空H3”案例存在分歧；184项只能表示来源/原编号已有解释，不能自动当成184项生物学缺失已确认或认证覆盖已完成。保留pending并量化对未知loop使用训练已知H3匹配整条benchmark观测串的保守附加筛查。新expanded exporter已开始实现与CPU测试，尚未扫描/导出全量或签证。

## 2026-10-01 typed诊断、监督上线与pilot生命周期验收

当前同一LLaDA主干/原目标的八项人工350步仍属诊断。R26条件AA98.66%、R20为96.05%，R21/R26平均loss接近；R26均值loss的93.43%来自无条件full-mask，须并列分层指标，不能据此确认真实任务收益。研究覆盖与优先级集中于[IMMUNE_ARCHITECTURE_RESEARCH_INDEX](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_RESEARCH_INDEX.md)，没有新增未经验证的架构开关。

下一步工程合同已具象化：typed随机attention接口两环境各132项CPU通过；监督32+跨模块3项通过并更新本地daemon；pilot生命周期各27项通过，仍要求独立数据认证和未来实际GPU训练/评估验收。当前loss/采样、39源、模型权重和500步预算不变。每记录loss的人工CPU框架验收在规划YAML中更新为已完成，生产接入仍pending。[IMMUNE_CERTIFIED_PILOT_EXECUTION](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CERTIFIED_PILOT_EXECUTION.md)仅证明隔离生命周期，不证明GPU收益或连续续训数值等价。

00:22Z监督增量：八项375/500正常，R26条件full-AA99.69%、R20为95.50%；R24等方法仍有回退，继续固定500预算，真实语料准入条件保持未满足。

## 2026-10-01 00:47Z 归档与固定路由准备验收

同一主干八项人工诊断继续至425/500，源码一致、无alerts；R21/R26/R24本点mean loss为0.18734/0.18810/0.18822，差距小于0.0009，条件full-AA则为98.39%/99.13%/98.73%。继续保留至500并比较最近100步范围与均值，不能凭单点选优或宣称下游改善。

[归档工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_REFINED_RELEASE_ARCHIVE.md)和[固定路由补丁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DIAGNOSTIC_DISPATCH_STAGING.md)分别通过两环境各22项和38项CPU验证，均尚未部署。归档将补充历史39源之外的运行依赖，但补充SHA不是追溯GPU认证。架构、目标和样本协议仍保持当前版本。新pilot的live来源在入口/registry修改后会与旧profile失配；须独立解决来源构造并验证完整GPU训练/评估，不能覆写旧证据。

## 2026-10-01 真实框架窗口审计带来的修正

未部署的评估/训练原型在HF完整窗口中漏做GA除法：模型`**batch`签名被误认为已消费窗口计数，实际compute_loss仍是microbatch均值。两环境真实Trainer+人工Linear/SGD证明GA2/4时梯度为预期均值2/4倍；明确model_accepts_loss_kwargs=False后归一化恢复，原90回归和每环境5个修正后窗口实验均通过，历史字节/修正前实验保留。[修正证据与限界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

这是使未来框架接入符合既定micro-mean的实现修正，不是record-mean新目标或架构收益。当前475步人工训练手动loss/accumulation且39源一致，无需因此重启或丢弃曲线。新GPU producer与真实factory须使用同一修正版本，先验证实际完整窗口、BF16/AdamW、四档指标与显存；不能只修producer再调用旧的错误factory。尾窗和恢复仍单独验收。[完整GPU方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_GPU_RELEASE_PRODUCER_PLAN.md)已更新，尚未运行。

## 2026-10-01 500步终态后的候选方向

八项人工500步全部完成。最近100步R21/R26/R24总体loss均值相差0.000743，R24条件full-AA均值99.00%、R26为98.77%、R21为98.53%；R26长输入显存代价+28.51%。保留R20对照及R21/R24/R26后续比较方向，尚不认定任何方法有真实任务收益。该toy部分mask已饱和，且总体loss主要受有输入冲突的无条件full-mask控制，因此不统一延长八项来挑选赢家。[终态报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_RESULTS.md)区分末点、尾段稳定性、学习速度和成本。

524文件父快照已保存、8项release_map已绑定；新路由的来源隔离修正完成后才运行随机张量CUDA诊断。未来真实短程阶段仍须原数据认证与完整新GPU训练/评估证据，候选方向不等于准入已满足。

## 2026-10-01 用户确认后的主线顺序

独立CUDA算子验证移出当前训练前置条件，继续现有attention；规划主线仅包含新Trainer/评估的实际GPU验收、通过数据准入的四候选1000步和选择后的累计5000步。三次完整optimizer更新仅验证实现；学习收益以足够长的真实训练曲线及第二seed支持。R20/R21/R24/R26暂为比较方向，不能从人工饱和成绩推断真实收益。四项先保持目标/采样/日程和micro分组一致，loss或来源采样修改随后各自独立消融。

数据审核和工程验收并行，最终选择只使用训练内固定验证，不使用下游test/few-shot query挑架构。CUDA专项产物保存供未来独立性能研究使用，当前不执行。

## 2026-10-01 当前整体架构与原型接入边界

[整体架构说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CURRENT_ARCHITECTURE_20261001.md)集中列出保留的 LLaDA270 主干、词表参数差异、链内 RoPE/role/scope/chain bias、A4 单终止符与容量边界，以及八项已完成500步的独立机制。当前比较方向是 R20/R21/R24/R26，不把它们预先组合；R21动态门控与R22固定0.5、R23可学习缩放与R24固定0.1各自保留对照。R26使用dense mask，长输入显存代价+28.51%，没有稀疏计算收益证据。真实语料及下游效果均未验证。

新固定 Trainer 入口两环境各29项CPU通过，运输/隔离与展示各34/14项通过；这些只证明对应接口与拒绝行为。producer 原址有未归属并发内容变化，完整字节保留后在私有工作副本继续适配，发布先复审并重绑定来源，不能把不完整producer的旧CPU成绩作为当前GPU合同证明。三次正式窗口只验实现；仍需完整原数据准入才能开始1000/5000学习比较。

## 2026-10-02T03:11:16.880451+00:00 单项失败GPU观测源码冻结

新driver61ab69fc、dispatcher8f9d4f8d、12项控制60601742已冻结，root/A/B/C完成源码与实际API静态对齐；18原保护来源未改。下一步仅两Python环境各12项新增纯控制/180秒，不重跑旧model/数值验收；当前控制未执行、未部署、GPU新任务0、真实训练0。原tiny/BF16/seed43/断言和全部原准入保持。

[实际冻结原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_before_execution.json)。


## 2026-10-02T03:13:11.684065+00:00 新控制各12通过与单项数值入口发布

pllm与worker Python新增纯控制各12/12首轮通过（约2秒），无模型/CUDA/旧数值检查重跑，OWN/_cpu_tmp已由fixture清理、checkpoint0。已实际发布冻结新dispatcher及两个既定live最小diff：原37registry项保持，追加一个launcher_only数值观测项；16原受保护源保持，2项有记录改变。旧d02/guard/worker/release_map/runtime/helper不改；原8worker/监听/容器未重启、Volc未提交/取消。当前尚未入队、新GPU0、真实训练0。

[实际CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_results.json)；[实际发布及差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication.json)。


## 2026-10-02T03:19:25.984054+00:00 单项观测启动失败及最小临时目录修正计划

唯一D03/GPU2已终态failed/rc2，child10.14秒正常退出并实际进程清理。status=not_started、tiny未执行、梯度/loss/RNG捕获0，无COMPLETE，不把它算作原模型数值失败复现。原源码隔离检查正确拒绝了Torch追加的PROJECT内临时库搜索路径；仅新观测driver临时布局有误。自有tmp及库已清理，checkpoint0，池0/0/51、8worker维护/监听保持，真实训练0。

按新证据准备独立v2修正：专属库tmp放在项目树之外的固定conda/cache/tmp子树，保留Torch原sys.path行为和原strict verifier。不热改v1、运行包或门禁；仅新增v2源码及两个受影响stdlib控制/每环境180秒，旧12项和已接受模型检查不重跑。v2另设一次新GPU/总180秒预算，现阶段源码准备、尚未执行/发布/入队；原tiny120秒/seed43/BF16与所有原数据/学习/恢复门禁保持。

[失败原件复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/failed_observation_review.json)；[v2明确范围与预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v2.json)；[v1及原源保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v2.json)。


## 2026-10-02T03:25:29.401687+00:00 v2目录修正控制通过并实际启动观测

新增目录/源码绑定控制两环境各2/2首轮通过（约3秒）；原12未重跑、无模型/旧数值验收。已发布最小v2，只改runner的dispatcher literal并新增冻结dispatcher副本，21保护旧源/v1/registry/release_map/runtime/verifier保持。原池仅入队一次新job `1790911529399018156_d03_r21_seed43_tiny_gpu_observation_v1_6acb659f`，observer_version=2、D03_v1只作固定transport身份，source/report精确绑定v2。原v1启动失败原件保留，不追认模型通过；本次尚待actual数值/正常退出。池{'pending': 1, 'running': 0, 'results': 51}，不重启/Volc提交，真实训练0。

[CPU控制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v2_results.json)；[实际最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v2.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v2.json)。


## 2026-10-02T03:32:00.954350+00:00 v2首批IPC失败与最小短路径修正准备

v2/GPU2终态failed/rc2：源码隔离已通过并记录正确TA/设备，但TMPDIR157字节导致resource_sharer的AF_UNIX socket path too long，等待首批到原120秒超时。gradient_call_count0/replaynull/数值捕获0，不能算原梯度失败复现；child正常退出并进程清理，wrapper约135秒<180，库tmp已清理，checkpoint0。全池0/0/52、8worker维护，真实训练0。

仅准备新v3短tmp base `/tmp/immobs3_251105016` + `j`及actual job_id SHA前16位叶，避免socket地址上限；先两环境各2项受影响stdlib检查，其中一个实际resource_sharer.DupFd/detach描述符通信，不导入Torch或运行模型。旧12+2/已接受数值不重跑，v1/v2源与原runtime/verifier不改，v3尚未执行/发布/入队。原窗口/seed/BF16/数值阈值和数据/500学习/恢复门禁保持，不重启/Volc重复提交，不恢复注释或独立CUDA算子。

[实际IPC失败](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/failed_observation_v2_review.json)；[v3明确预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v3.json)。


## 2026-10-02T03:38:23.692209+00:00 v3实际IPC控制通过并单项GPU观测启动

两环境各2/2新增控制首轮通过（约3秒），包括实际resource_sharer.DupFd/detach socket/FD传递及本case线程/socket/文件/环境完整清理；旧12+2/数值验收不重跑。只发布新冻结dispatcher和runner literal，25保护旧源/v1/v2/registry/release_map/原verifier/IPC不改。原池一次入队job `1790912303689399341_d03_r21_seed43_tiny_gpu_observation_v1_324e2b1c`，observer_version3，tmp短映射实际/tmp/immobs3_251105016/j2035c7b732f32f84；原120秒tiny/总180秒与全部门禁保持。实际池{'pending': 1, 'running': 0, 'results': 52}，尚未记数值观察完成或模型接受，真实训练0。

[实际CPU IPC检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v3_results.json)；[最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v3.json)；[真实启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v3.json)。


## 2026-10-02T03:44:16.057634+00:00 原GPU梯度差异已复现并取得完整数值，正式训练仍未启动

v3单项观测在GPU3于03:38:44 UTC结束，root使用同一worker Python只读绑定report/COMPLETE/log SHA与实际exit/清理全部核验。COMPLETE仅observed、完整采样true；原tiny约2.01秒仍报wte.weight梯度不一致，model_test_passed/GPU/data/training接受全部false。当前source/seed原工程验收仍7/8，七项不重复、不撤销原失败。全池0/0/53、8worker维护、原监听2213164存活；原Goal身份/active/累计计量保持。

26梯度张量均有限、成员一致；17/26张量、42/7156元素超原逐元素阈值。wte4/800失败，maxabs0.000306189；全局scale0.9999226166、relative L2=0.0003664514（0.036645%），通过整体数值阈值但不能替代逐元素失败。8对RNG、mask、selected labels及前向logits完全一致，GPUloss最大差4.768e-7，观测重算满足loss标准（原loss断言在梯度报错后未执行）。16组CPU/FP32叶子比较最大误差1.49e-8、无失败；这是CPU观察，不能证明CUDA kernel根因。下一必要工作限通用synthetic原模型反向路径定位，先区分CE导数、混合精度舍入和checkpoint反传；不放宽阈值、不执行独立算子验证。

本轮每环境12+2+2=16个新增定向控制首轮通过，旧已接受数值/模型检查未重跑。v1隔离失败、v2 IPC失败均保留，v3短路径通过actual CPU Socket/FD IPC后才获取GPU原断言证据。root首次只读复审错用pllm导致sys.executable绑定拒绝，原件保留；仅用actual worker Python修正receipt核对，未改validator/生产源或重复GPU。25保护原源保持（runner既定literal变更有记录）、v1/v2源码/raw及实际训练恢复状态保留；GPUchild/库tmp清理完成，另外两个自有空tmp父目录删除，checkpoint创建/删除0。

03:44:16 UTC直接读取V7 MHC metadata，manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，audit_pending、ready字段缺失、当前audit_certificate/downstream_scope_certificate缺失。既有V7清理保留，不称已证实泄漏。数据版本是V7，条件启动授权已有，但当前审核/ready、原500学习/完整恢复/data-selection等门禁仍未闭环；真实shard/小batch/语料训练0，不能直接启动八卡正式训练。停止注释不恢复、独立CUDA暂停；没有Volc提交/取消或容器/worker/监听重启。

[本次完整数值复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v3_terminal_numerical_review.json)；[资源、metadata及清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_and_numerical_evidence.json)；[原始观测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790912303689399341_d03_r21_seed43_tiny_gpu_observation_v1_324e2b1c/numerical_evidence/gpu_diagnosis.json)；[原Goal实际工具值](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/goal_after_numerical_actual.json)；[首次只读核对错误](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v3_terminal_first_read_failure.json)。


## 2026-10-02T03:52:02.174489+00:00 原反向中间梯度观察准备

本Goal上一回合属progress：原GPU梯度差异与完整数值已核实。当前单次只读核实pool0/0/53、8worker新鲜心跳维护、监听2213164存活；指定V7 MHC ready字段及两审核仍缺，真实训练0。已制定仅v4观察预算：原tiny/native8+manual8不变，只在原observed_denoise返回的共同BF16 logits注册return None的梯度读取，最多16原回调，无额外GPU forward/backward/算子probe；实际CPU梯度原件保留。原七项、TA、模型、独立oracle、阈值及所有训练门禁不变，原源冻结。先独立只读评审与每环境最多4项新hook控制，通过后原池至多一次单GPU原tiny。此刻v4未发布/入队，无平台提交、重启或真实shard读取，停止注释与独立CUDA暂停保持。

[明确预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v4.json)；[保护旧源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v4.json)。

## 2026-10-02T04:00:16.741571+00:00 v4只读评审发现与修正

两独立评审确认原GA口径一致，原native Trainer loss/8、Accelerator GA1，manual loss/8；最小观察点为原exercise唯一observed_denoise返回共同BF16 logits，16个return None hook只复制实际原梯度到CPU。原输入/模型/TA/独立CE/阈值不改，无额外GPU计算，CPU复制可能影响同步。A指出torch.equal不是严格位相等，driver已在未执行稿改称exact_value_equal并显式BF16要求；还指出新报告job_id必须与实际child输出目录绑定，B正在仅补该元数据校验，交付旧字节留证。四个新控制含该跨job拒绝，不增加case。此刻未root冻结、控制未执行、GPU入队0；原7/8及失败保持。

[评审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_observation_design_review.json)。

## 2026-10-02T04:05:28.845861+00:00 v4原反向观察源码冻结

Root actualdiff/AST及独立评审通过；driver59961065、dispatcherbf843792、control585d705e、budgetec07314e固定。31旧路径不变，模型/TA/原断言阈值不改。报告精确绑定child job；共同BF16读取字段称数值相等。即将仅每环境4项新增hook/trace/metadata/CPU leaf控制，旧验收不重复；GPU尚未入队、真实训练0。

[源冻结原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_v4_before_execution.json)。

## 2026-10-02T04:07:14.414668+00:00 v4四项控制首轮通过并单项GPU观察启动

两环境各4/4新增控制通过（worker6.43秒、pllm6.63秒），真实CPU leaf hook return None/不改梯度/remove后无旧回调、trace/caller与跨job报告拒绝全部通过。Root首次pllm命令错指不存在解释器，child未创建/测试0；仅修正到实际activated sys.executable后首次4项通过，错误元数据保留、零字节自有log删除，源码未改。旧七项及旧验收不重跑。只发布新dispatcher和runner literal，30保护旧路径及31中其余项保持。原池一次入队`1790914034411779826_d03_r21_seed43_tiny_gpu_observation_v1_e950d2a0`，原native8+manual8/120秒、总180秒、16原hook读出；尚待实际数值/退出与清理，观察完成不代表模型准入。真实训练0，数据/500学习/恢复门禁保持；无Volc提交或重启。

[CPU证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v4_results.json)；[最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v4.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v4.json)。

## 2026-10-02T04:18:00.540716+00:00 原BF16反向入口已定位一处差异，完整原件复审完成

v4 job `1790914034411779826_d03_r21_seed43_tiny_gpu_observation_v1_e950d2a0` 于04:07:37 UTC在GPU3正常结束，root用actual worker Python只读verify_sources/report/child/COMPLETE/raw绑定与独立stdlib逐值重算通过。COMPLETE仅observed、原tiny仍同wte.weight ValueError，原source/seedGPU工程仍7/8、R21s43未准入；没有撤销原失败或改标准。原反向8native+8manual、16hook全部移除，parent19.785秒、child14.670秒、原tiny2.100秒，实际进程组/库tmp/exclusive子目录清理完成。checkpoint创建/删除0，自有零字节错误launcher log已删，源码/失败原件/原数值及实际训练恢复状态保留。

8对原共同BF16 [1,64,50]梯度中7对完全数值相同；第7步只flat2654=[0,53,4]一元素不同：native0.0003871917724609375，manual0.00038909912109375，差1.9073486328125e-6（相邻BF16值）。该边界仍满足原allclose，relativeL2=4.813081e-5；但参数17/26张量、42元素超原标准，全部参数统计与v3精确相同。8对RNG/mask/labels/forward仍同，GPUloss maxdiff4.768e-7。证据证明至少一处差异已进入模型反传边界的loss/layout/cast一侧，不证明该1元素解释全部42、不认具体CUDA kernel根因；16CPU leaf仍只CPU，未执行独立算子。

A/B只读下一决策：暂不改shared native CE，下一必要点先审阅/有限观测原已有FP32 loss输入的实际gradient，CPU统一布局、return None、无额外model/CE/backward，不改变原GA8及阈值。若以后采用数学等价row-major CE，须新的私有源码/CPU等价/原tiny与必要完整GPU证据，独立oracle保持，旧七项不迁移到新SHA。此刻没有v5源码/预算/新job。

最新04:12:17快照pool0/0/54、8worker维护且心跳<5秒、监听2213164存活，无容器/worker/监听重启或Volcsubmit/cancel。V7 MHC metadata仍e12e55cc…/audit_pending、training_ready字段缺失、两当前版本/下游报告缺失，不等于已证实泄漏。真实shard/小batch/训练0，原数据审核、500学习、完整恢复/data-selection等门禁全部保持；停止注释不恢复，独立CUDA暂停。原Goal身份/active/actual累计用量保持，本回合属progress，不计全部外部阻断；无子agent继续执行，不以新增无关测试填充Goal。

[原反向独立复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_terminal_backward_review.json)；[实时状态/metadata/清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_after_v4_backward.json)；[下一必要工程决策](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_next_engineering_decision.json)；[Goal实际工具累计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/goal_after_v4_actual.json)。

2026-10-02T04:18:40.733707+00:00：状态JSON原v1的source/控制/失败字段移到legacy_v1_observation_fields；当前字段明确指v4来源与actual observed=true/model_test_passed=false，避免历史与最新混用。仅状态/MD同步，无新模型或测试执行。

## 2026-10-02T04:21:40.077216+00:00 v5原FP32反向观察预算

上一Goal回合是progress，已取得原BF16入口1相邻值差异及raw复审。当前一次只读snapshot04:20:13，原池0/0/54，8worker新鲜心跳维护、监听2213164存活；V7 MHC ready字段/两审核仍缺。仅预算一次必要原tiny读出：保留原16 BF16 hooks，新加16原FP32 CE-input/manual-logits None hooks，原8+8 backward/GA8/120秒、总180秒不改，32handles finally移除；CPU转置统一BLV、CPU cast对照同轮真实BF16，raw保留。无额外GPU CE/forward/backward或独立算子，原模型/TA/assert/门禁不改。4项新增定向CPU控制/env，旧验收不repeat，来源评审和控制通过才允许原池一次入队。目前未发布/执行，无真实shards/训练、无重启/平台提交，停止注释与独立CUDA暂停。取得该边界证据后先做最小通用工程决策，不无止境增加无关观察。

[预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v5.json)；[保护旧源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v5.json)。


## 2026-10-02T04:29:13.960945+00:00 v5通用工程来源冻结

v5原FP32观察源码root冻结：A416e6ec1/Bcad2625c/Cdcf8bbec，37旧源保持，AST差异限新FP32观察/元数据校验；4新增CPU控制/环境即将执行，当前未发布/入队。原7/8、全部数据门禁/停止注释/独立CUDA暂停保持，真实训练0。

[复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_v5_before_execution.json)。


## 2026-10-02T04:31:34.602707+00:00 v5控制通过与原池唯一观察启动

两环境各4/4新控制通过，新冻结来源已发布；原八卡池唯一v5合成观察 1790915494600058674_d03_r21_seed43_tiny_gpu_observation_v1_5d224d8a 已入队，保留原8+8/GA8/阈值、32 None-hooks，转置/cast仅CPU。待实际终态/raw复核，原7/8不变，全部数据/安全停线保持、真实训练0；未重启/平台提交。首次root读取worker错用heartbeat字段，实际字段time，发生在发布前，未运行child/重复checks/改validator，失败原件保留。

[CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v5_results.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v5.json)。


## 2026-10-02T04:37:11.813335+00:00 v5实际FP32证据完成与训练就绪状态

v5唯一观察 1790915494600058674_d03_r21_seed43_tiny_gpu_observation_v1_5d224d8a 于04:31:58 UTC在GPU1完成，root actual worker Python仅stdlib只读receipt/COMPLETE/源SHA/raw逐值重算通过。新增控制两环境各4/4、原32 None-hooks全部移除，子进程/库tmp/独占临时目录清理完成，checkpoint创建/删除0；源/raw/失败原件及实际训练恢复状态保留。FP32共287值差（maxabs1.86265e-9），独立IEEE round-nearest-even重算16组cast全部匹配同轮实际BF16；第7步[0,53,4]的FP32值距midpoint -2.91e-11/+8.73e-11，因此产生1个相邻BF16值差。参数统计与v4完全一致，仍17/26张量、42元素超原阈值；原wte断言仍失败、当前工程7/8，COMPLETE仅observed而非模型通过。此证据不证明全部42差的完整因果或某个CUDA kernel。

下一决策：停止加hooks/观察波次，准备一个私有数学等价的native CE行优先布局候选，保持独立oracle、GA/BF16/阈值及原接口，先针对性CPU等价/原tiny，再按新来源补必要完整证据；此刻候选未实现、未验收，旧七项证据不迁移。原当前来源学习/完整checkpoint恢复/data-selection门禁不撤销。V7 MHC manifest仍e12e55cc…ba94、audit_pending、training_ready字段及两份当前版本/下游审核缺失，既有清理保留，不称数据已证实泄漏；真实shard/小batch/训练0。原八卡池0/0/55、8worker维护且新鲜心跳、监听2213164存活，无重启/Volc提交或取消。停止序列注释不恢复、独立CUDA暂停。

本轮root两次只读记录字段误读（worker实际time、环境symlink路径需resolve）已保留并修正核对，不改生产源/validator或重复模型检查；最终强复核无ML导入。

[本轮原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v5_terminal_FP32_review.json)；[资源与数据状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_after_v5_FP32.json)；[下一具体决策](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v5_next_engineering_decision.json)。


## 2026-10-02T04:39:55.699962+00:00 私有通用损失布局修正准备

上一回合progress：v5原FP32/cast实证及root raw复核完成，本轮只推进私有数学等价native CE行优先布局修正。原八卡池/来源/阈值及独立oracle保持，计划4必要合成CPU合同测试/env、tiny最多一次/原120秒8+8，不再加hooks；当前仅预算/源准备，未执行/发布/入队。原7/8、V7 audit_pending/ready与双证书缺失、真实语料0不变。停止注释不恢复、独立CUDA暂停、不得重启/重复平台提交。保护清单首轮路径误写的只读错误保留，按实际文件修正，不改原源。A写私有forward最小改动、B只读原包整合API、C写4CPU合同源；root独占执行与MD。

[范围与预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/execution_budget.json)；[原来源保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/protected_original_sources.json)。


## 2026-10-02T04:44:32.155121+00:00 CE修正必要来源绑定预算

只读原包API确认不接受model override/新root；必须新的固定namespace、原531 parent清单严格验证、534同名清单仅model与必要bootstrap新SHA。原verifier不改；producer算法/oracle不动，只新固定roots/schema。预算追加3必要软件绑定控制/env180秒（另有原4数学合同），tiny-only不伪造profile且全部GPU/训练准入false，完整release入口仍待接入/新profile。当前软件源准备，无构建/GPU/真实数据。

[绑定补充预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_binding_addendum.json)。


## 2026-10-02T04:47:13.937285+00:00 私有CE数学合同来源冻结

root实diff/AST确认除forward CE四行及运行说明外全部模型定义相同；原10源保护保持。即将仅4新增CPU合同/env，旧模型验收不重跑；新固定运行包正在写，未发布/GPU。原阈值/独立oracle/数据门禁及安全停线保持。

[源码复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_source_review_before_CPU.json)；[实际diff](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_source.diff)。


## 2026-10-02T04:48:37.127426+00:00 CE私有修正数学合同通过

pllm/worker Python各4/4首轮通过（约4.92/4.23秒），直接actual全forward AST覆盖FP32/BF16、稀疏/非连续/-inf允许类、空选择和全忽略反传零grad、balanced分组全部输出指标及梯度；原loss2e-5、grad2e-5/.005保持。辅助分支未执行，AST原样保持，不宣称其新验收。新固定runtime/bootstrap仍待交付与3必要软件控制；此刻无GPU新case、修复GPU未证明、原7/8/真实训练0。

[CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_CPU_contract_results.json)。


## 2026-10-02T04:52:50.263506+00:00 CE新固定运行包来源冻结

root实SHA/AST确认producer全部函数不变、package仅plan/verify/build和固定namespace/schema/singleton override，原531/534清单保持。3新stdlib绑定控制/env待执行；tiny driver完整原schema_version已按事实修正，parent字段按actual报告对齐，均发布前未执行的静态纠错。数学4项不重跑，GPU新case0/无真实数据或训练，原安全停线/门禁保持。

[来源复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_source_review_before_CPU.json)。


## 2026-10-02T04:58:56.129931+00:00 CE修正来源包实际构建

两环境3软件控制的不同body全部通过：前2首轮通过；第三fixture先前未执行，补自有父目录后首次body通过，不repeat已通过项。自有CPU tmp及其内pytest链接安全删除，setup与root过严清理guard错误原件保留，source/门禁未改。新source-only固定R21包实际构建534来源，531 parent完整保留，model唯一parent override、39 measured仅model SHA变化；原10保护源保持。全部准入false、fullrelease entry未接入；待actual manifest绑定父监督和原tiny一次执行，当前GPU新0/真实训练0。

[真实构建](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_build_actual.json)；[控制与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_CPU_controls_and_cleanup.json)。


## 2026-10-02T04:59:42.888071+00:00 CE修正原tiny实际启动

4数学+3软件合同/env通过，新固定source-only R21包534来源已实际构建，父监督/source/budget均冻结。原池一次入队CE修正原tiny 1790917182885479822_d03_r21_seed43_tiny_gpu_observation_v1_6c5bc4a4，原8native+8manual/GA8/seed43/120秒，parent180、无新增hooks。profile=None、原fullprofile/完整release/训练门禁保留，全部准入false；实际原梯度断言结果尚未返回。仅新增冻结dispatcher+runner literal既定改变，原其余9保护源/旧包/失败原件保持，无容器/worker/监听重启、平台重复提交或真实训练。

[启动原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/launch_actual.json)；[最后源/入口复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/tiny_entry_review_before_execution.json)。


## 2026-10-02T05:09:28.411744+00:00 CE 修正版原 tiny 通过，完整训练验收仍待闭环

CE 行优先布局修正版已于 2026-10-02 05:00:05 UTC 在原 GPU1 一次通过原 tiny：26 个参数梯度全部通过，梯度倍率 1、相对 L2 误差 0、RNG replay 一致，loss 最大差 4.76837e-7。原阈值、GA8、BF16、seed43 和独立 oracle 均保留，没有新增 hooks 或独立 CUDA 算子测试。tiny 1.780 秒、总监督 20.589 秒，真实退出码 0，子进程、对象及独占临时资源清理完成。

两套环境各 4 个新增模型数学合同及 3 个新增软件来源合同均通过；软件首轮前 2 个通过，第 3 个因自有 tmp 父目录缺失在 fixture setup 失败，补目录后仅该 body 首次执行通过。没有重复已通过项目，错误原件及来源保留。新不可变包 534 个源码文件，model SHA 7e79bfb7d2327c07aa3c39e0bed89997fa627b3fa6e20cb084630b347e886592，manifest SHA 2c2c9ee3ca2f17c29da402b937902d158d82eeedb5ba7753dcf1e0be1307c148。

这只证明新源码的原 tiny 数值检查通过；该检查仅 8 条人工记录、1 个窗口，LR=0、参数变化=0，不是学习效果证据。旧源码完整工程验收仍 7/8，旧 R21 seed43 失败原件保留，不能与新 tiny 相加成 8/8。新源码完整 release 尚未执行完成/未准入，当前完整 entry 仍固定旧目录，新源码完整 profile 尚未测量；所有完整 GPU、数据与训练准入标志仍为 false。

2026-10-02T05:06:26.708214+00:00 只读快照：原八卡池 pending/running/results=0/0/56，8 个 worker 都在维护待机，心跳均小于 5 秒，监听 PID2213164 存活。此刻没有新平台状态查询，不把缓存 Running 当作新查询；未重启容器/worker/监听，未 Volc submit/cancel。checkpoint 创建/删除均为 0，自有 CPU tmp 已删除，GPU 子进程独占 tmp 清理有原报告；保留全部必要来源、原数值/失败证据及实际训练恢复状态。

用户指定数据仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，train 6,849,243 / valid 52,572。manifest 的 training_ready 字段缺失，decontamination=audit_pending；audit_certificate.json、downstream_scope_certificate.json 仍缺失，父级当前范围审核未闭环且新增 manifest 未被父证书覆盖。既有清理保留，此状态不证明已发生泄漏。没有读取真实 shard、小 batch 或启动真实语料训练。

下一真实工程动作是新建不可变的完整 release 入口/运行包版本，保持既有来源验证、独立 oracle、阈值和数据门禁；不能热改已冻结的本轮 private/runtime 包。复用未变且已接受的模型数学证据，仅补必要的新入口/来源控制、当前源码完整 profile、完整梯度核对、3 个正式优化窗口、分 mask 评估及退出清理证据。当前源码学习、完整 checkpoint 恢复和原 data-selection 门禁仍需闭环。不再增加 hooks、重复已接受 tiny/CPU 项或迁移旧七项证据；不恢复此前停止的序列注释分支，独立 CUDA 算子验证继续暂停。正式数据审核未满足时不得启动真实小 batch 或八卡语料训练，用户授权无需重复确认。

[原 tiny 独立强复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/tiny_terminal_root_review.json)；[当前资源与数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/final_state_after_CE_tiny.json)；[下一具体工程动作](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/next_engineering_after_tiny.json)；[模型 CPU 合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_CPU_contract_results.json)；[软件合同与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_CPU_controls_and_cleanup.json)。


## 2026-10-02T05:11:09.264315+00:00 CE 修正版完整合成训练入口接入预算

上一 Goal 回合是实质 progress：CE 私有修正、两环境各 4+3 合同与原 tiny PASS 已核实。本轮只接入同一模型 SHA 的新不可变完整 release 版本，限定 R21 seed43，一个完整合成验收，不热改本轮前已冻结来源。最多每环境 4 项必要新入口/namespace/来源绑定控制，180 秒；通过后才可新建来源包、绑定实际 manifest 并在原池执行 1 次。原 profile 900 秒、总 3600 秒、完整独立梯度/3 窗口/分 mask/退出清理协议保持；不单独重复已接受 tiny/数学合同，原完整 producer 自带 lifetime 阶段保留。当前未写新生产源、未构建/执行/GPU 入队。

停止注释不恢复、独立 CUDA 算子暂停；真实 shard/小 batch/训练不准入，V7 审核/ready 缺口保持；无容器/worker/监听重启或 Volc 提交取消。初次只读 guard 路径查错已留原件，按实际外部 private guard 位置读取，不改验证器。

[本轮预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/execution_budget.json)；[冻结保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/protected_before.json)。


## 2026-10-02T05:14:19.096400+00:00 完整入口元数据构建前置

必要依赖已核实：旧完整 entry 的 package_bootstrap 与 external guard 只接受固定实际目录，因此先 root 静态复审、实际 exclusive source-only build（全部准入 false）、绑定真实 manifest/guard，再做最多 4 项新增入口合同/env。没有 fake mirror/伪造 GPU profile；GPU 入队仍须上述合同和原完整门禁。A/B 正在仅交付来源/入口/分派源码，C 编写 4 项 stdlib-only 合同，root 独占构建/执行/发布。当前未运行模型或新 GPU 任务。

[依赖补充](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_only_build_dependency_addendum.json)。


## 2026-10-02T05:16:03.114559+00:00 CE 完整入口来源包实际构建

Root AST/diff 确认 package/producer/entry 全部函数与已接受来源相同，模型 SHA/字节不变；只新 namespace/schema、R21 seed43 身份与外部 guard 固定目录。12 个保护旧源不变。新实际 source-only 运行包 534 文件/531 parent/39 measured，manifest SHA 2ef8f6777120434c0371083838d3ad03d259e807153d7798a745197e7e9cdefd，全部准入 false，无 GPU profile/模型执行；构建是入口合同前置，不是完整验收。下一步绑定实际 manifest/guard 后只做 4 项必要新入口控制/env。

[静态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_review_before_build.json)；[实际构建](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_build_actual.json)。


## 2026-10-02T05:20:53.599353+00:00 完整入口分派静态复审与最小变更暂存

新 parent 源 221 行/SHA48799767…1b3f909 已实际读取并由 A 独立只读复核；仅 R21 seed43 接新不可变运行包，其余既有身份保持字节固定旧入口，独立 CUDA 分支先拒绝。实际 manifest/guard/534 来源和冷启动 marker、loaded namespace 验证保持；没有真实训练入口。仅 runner 单个路径 literal 的 reviewable diff 已暂存，未发布、未执行或入队。12 个保护路径保持，等待 4 项新入口合同/env；模型原数学/tiny 不另跑。

[源与暂存](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/parent_source_review_and_staging.json)；[完整验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CE_FULL_RELEASE_20261002.md)。


## 2026-10-02T05:25:05.103260+00:00 完整合成入口合同通过并实际接入

pllm/实际worker Python各4/4新增stdlib入口合同首轮通过，约4.58/4.83秒，无模型/Trainer/GPU计算，无旧数学/tiny重跑。真实ENTRY bootstrap/loaded namespace、fresh guard SHA/seed拒绝、完整绑定与五选项拒绝均通过；A独立静态复审无阻断。只发布新byteidentical parent和runner一个路径literal，原其余11保护源保持；新534源码包不改。自有空CPU tmp父目录已删，checkpoint0。发布前8worker维护/心跳新鲜，监听2213164存活。此刻GPU尚未入队、没有新profile或完整GPU准入，真实语料训练0；无重启/Volc提交取消。

[四项合同原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/entry_CPU_contract_results.json)；[实际接入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/publication_actual.json)；[完整验收文档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CE_FULL_RELEASE_20261002.md)。


## 2026-10-02T05:25:53.527609+00:00 修正版完整合成验收唯一入队

原池唯一入队新的 R21 seed43 完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438。transport profile-synthetic/steps=1 只表示一个验收调用，内部保留真实micro32完整输入profile、独立完整梯度窗口及3个正式优化窗口/768人工记录、四档mask和原退出清理协议；不是1步正式语料训练。原阈值/GA/BF16/seed/source/data门禁保持，当前实际GPU结果未返回，全部完整准入仍false。模型源7e79…保持，旧七项不能迁移或相加为新8/8。只使用原非闲时八卡资源；没有重启或Volc重复提交，真实shard/小batch/训练0。

[实际入队原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/launch_actual.json)。


## 2026-10-02T05:33:06.012891+00:00 新完整profile通过，完整梯度阶段实际运行

唯一新完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438 在 GPU7/PID461451 运行。新源码实际 micro32、length2048 的 profile 已通过，峰值 allocated 42.18383693695 GiB；实际测量108.852秒、profile子进程119.698秒、rc0/真实观察退出及清理均完成。Root使用实际worker Python只读原 profile 与 observed-exit 验证器重核，来源manifest2ef8…cdefd和新模型SHA7e79…均吻合；本地CUDA未初始化，没有重复forward/backward或GPU计算。

完整producer原有lifetime阶段记录26参数oracle通过、relativeL2/maxabs=0、RNG一致，11 weakrefs全部释放且allocated/reserved最终0；该阶段原始partial观察2.369秒，1窗LR0/参数变化0，不构成学习效果。此刻完整模型 gradient_oracle 仍在运行，3个正式窗口/评估/完整raw/正常退出及全部新源码完整准入尚未完成。Root只独立强核已完成profile，partial lifetime不冒充完整raw validator验收。旧完整仍7/8，新tiny/profile不得拼成8/8，不迁移旧proof。

最新 2026-10-02T05:33:06.012891+00:00，pool pending/running/results=0/1/56，该job poll-gated worker心跳/PID持续更新，与实际request/profile/新phase产物及GPU遥测交叉核对；其他7worker维护、监听2213164存活。root主机不能直接查看容器PID namespace，不因本地ps不见就判退出。没有观察超时重启/重试/重复入队、Volc提交或取消。

V7 MHC manifest仍e12e55cc…ba94，training_ready字段缺失、audit_pending、当前audit_certificate.json/downstream_scope_certificate.json缺失；真实shard/小batch/训练0，旧清理保留且不等于已证实泄漏。当前data-selection、当前源码学习/完整checkpoint恢复及所有安全停线保留；停止的序列注释不得恢复/重试/换agent，独立CUDA算子暂停。自有空CPU tmp已删，checkpoint0；运行中GPU产物和实际训练恢复状态保留。

下一步只监督这一真实存活任务至原完整梯度、3窗口与四mask评估和退出终态，再按当前实际源码/seed/profile/全raw与原validators独立复核。观察超时不等于任务退出，不能重启或再入队。C监督agent正在只读跟踪该job并报告阶段/错误，A/B源码与独立静态审查已完成；root拥有实际执行、门禁与MD更新。真实数据审核未满足时不得正式训练。本轮是实际source-only包/4合同/入口接入/新GPU profile进展，Goal保持原身份active，未按软件验收缩小原训练目标。

[profile强复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/profile_stage_root_review.json)；[当前运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/profile_running_state_actual.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/launch_actual.json)；[四项新合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/entry_CPU_contract_results.json)；[完整验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CE_FULL_RELEASE_20261002.md)。


## 2026-10-02T05:48:27.525960+00:00 完整梯度进展与准入边界核实

阶段性架构候选已实现，最终胜出方案未选择。当前 2026-10-02T05:48:27.525960+00:00：唯一新完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438 仍在 GPU7/PID461451，原profile已通过（micro32、length2048、42.18384 GiB）。完整梯度检查原阶段已通过：86参数、8 microsteps/256人工记录，relative L2/maxabs=0、RNG一致，耗时698.856秒<原1200秒；11 weakrefs释放，最终allocated/reserved均0。仅核实原partial阶段，完整raw及独立终态validator尚待任务结束，不将局部通过当作完整准入。

正式优化已写出第1/3窗口（8 microsteps/256人工记录，loss3.660816848、原warmup LR0、参数变化0），仍待第2/3非零LR窗口、四mask评估及实际正常退出。池pending/running/results=0/1/56；其他7worker维护，未新查询平台状态。没有重复入队、Volc提交取消、容器/worker/监听重启。旧完整7/8保持历史，不和修正版局部结果合并为8/8。

V7确认为 immune_v7_mhc_sequence_v1；manifest e12e55cc…ba94、training_ready字段缺失、audit_pending、两份当前审核文件缺失。既有清理保留，这不等于已证实泄漏；真实shard/小batch/训练0。原数据门禁、停止的序列注释及独立CUDA算子暂停全部保留。

源码只读复核纠正剩余工作表述：checkpoint完整恢复核心已实现，初次1000不要求先有checkpoint；真实恢复证据须来自后续实际checkpoint，原resume校验不变。当前源码500步学习证据须位于新profile的同一evidence_root；3窗口不能替代。正式seed训练调用还未接到已有execute_admitted，source-only launcher仍拒绝。完整release通过后补必要学习证据和最小调用绑定，无需再写Trainer或重跑旧Linear恢复。最终架构选择、真实1000→5000及八卡正式训练尚未完成。

[本轮阶段原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_gradient_stage_root_read.json)；[最新状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/gradient_pass_formal_running_state_actual.json)；[剩余工作源码核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/post_release_remaining_work_readonly_review.json)。

## 2026-10-02T05:54:25.560151+00:00 第2优化窗口实际更新及后续接口核实

正式优化已完成2/3窗口：第2窗学习率1e-6、parameter_delta_l2=0.00839836089，loss=3.660629302，86参数梯度有限且非零；正在第3窗，随后四mask评估及实际退出。完整raw/独立终态验收尚未完成，不能以局部参数更新替代当前源码500步学习或真实语料效果。

已核实原run_immune_refined.train(args)/expanded_diagnostic可复用，不另写Trainer。保留validate_job对当前源码25步成功的前置、人工manifest重建校验和micro32/GA8/global256；当前source25/500的真实产物须归同一evidence_root。固定seed42人工学习协议不声称seed43学习，后者正式软件/profile/GPU/resume绑定原样保持。完整release通过前本轮不启动新25/500或发布正式调用；有效数据审核仍独立必要。

[第2窗口原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/formal_window2_root_read.json)；[最新实际状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/formal_window2_state_actual.json)；[最小学习及正式调用方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/existing_learning_and_formal_call_readonly_plan.json)。

辅助只读路径误读已按原件留证并发现真实enqueue文件，不涉及执行或源变更。[路径错误](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/readonly_verifier_preparation_path_error.json)。

## 2026-10-02T06:04:34.231753+00:00 当前来源完整合成验收独立通过

当前 2026-10-02T06:04:34.231753+00:00：唯一修正版完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438 于05:58:07 UTC succeeded/rc0，root实际worker Python冷进程已通过原ENTRY.verify_completion/producer.verify_candidate、完整raw梯度统计/人工输入/CPU评估重算及源码/seed/profile绑定；本地CUDA未初始化、未重复模型。新manifest2ef8…cdefd/model7e79…6592保持，source-only包全部准入flags仍false，独立GPU证明单列；不和旧七份结果合并为8/8。

完整86参数梯度误差0，三个完整GA8窗口/768人工记录，其中原warmup首窗LR0/参数变化0，后两窗LR1e-6/2e-6、deltaL2 0.00839836/0.01648205；loss3.66081685→3.66062930→3.55274889。正式峰值allocated42.42445GiB、1051.854秒；四mask每档24条/共96、5.033秒，完整raw/原population计数一致。profile119.698秒+release1786.258秒≤原full-proof总1915.742秒<3600，子进程实际rc0/normalexit/cleanup齐全，各阶段11 weakrefs释放/allocated reserved最终0。A独立只读复查一致；raw19,898,032字节及必要source/失败原件保留，两个自有空HF目录删除，checkpoint0/实际训练恢复状态不动。

原池pending/running/results=0/0/57，8worker维护且心跳新鲜，监听2213164存活；未新查平台、重启/Volc提交取消或重复入队。下一必要工程只补新源码25→500学习证据：已预算两个薄控制模块与最多4新增合同/环境，复用既有冻结引擎及原25前置，同一evidence_root、固定seed42人工协议与seed43正式身份分清；本刻未发布/入队新学习项。FULL/model/validator不改，不重复已通过数学/tiny/profile/full/恢复或独立算子。

V7 MHC仍training_ready字段缺失/audit_pending/两当前审核报告缺失；已有清理保留且不代表证实泄漏，真实shard/小batch/训练0。正式调用接入、最终架构选择、真实1000→5000及八卡训练尚未完成；原数据准入和resume校验保持，停止的序列注释不恢复、独立CUDA暂停，原Goalactive/身份及累计计量保持。

[完整独立复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_terminal_root_review.json)；[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_GPU_owned_cleanup.json)；[真实资源状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_terminal_resource_actual.json)；[最小后续预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/execution_budget.json)。


## 2026-10-02T06:08:00.211898+00:00 必要学习任务识别项核实

原worker每次command都动态AST读取literal VARIANTS；新学习任务需要追加一个launcher_only识别项，随后由严格冷控制器处理。最终发布只允许d04_r21_CE_learning_v1这一条，原entry/函数AST保持，FULL包/模型数学/validator不改；不需要也不允许重启worker/容器/监听。实际八卡YAML job-seconds86400容纳28800秒学习预算，不修改worker参数。本刻只source准备，registry未改、新job0；人工seed42与正式profile来源seed43分明，数据/停线要求保留。[依赖原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/literal_registration_dependency_addendum.json)。


## 2026-10-02T06:10:35.497433+00:00 必要学习入口两项literal变更已暂存

仅暂存一个launcher_only学习识别项与runner入口路径替换，全部旧识别项及registry函数AST保持；公开文件未改，新job0。两个薄source模块仍由B实现，A准备唯一4项stdlib合同，root将在实际源码审阅与合同通过后才发布。FULL包/模型/validator不动。[实际暂存复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/literal_staging_review.json)。


## 2026-10-02T06:23:52.752151+00:00 两个薄控制source审阅及固定

两个新source仅复用原人工学习引擎与进程退出清理API；root已逐行审阅，修正本次index删除归属、signal退出标记和来源字段，并一次固定child SHA。所有旧保护路径保持，公开registry/runner未改、四项合同尚未运行、新job0。A正在完成唯一4项stdlib控制及独立边界复审，root随后只运行这4项/环境，完成后才发布或实际入队。[固定source原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/source_review_before_controls.json)。

## 2026-10-02T06:35:51.179554+00:00 新学习调用检查通过并最小接入

两环境各四项新增stdlib合同首次执行通过（pllm4.673秒、worker Python4.831秒），root实际原件/SHA复核及A独立静态审阅均通过。三源固定；只发布一个新控制入口、runner路径字面替换和d04学习识别项，旧registry项/函数AST、534文件FULL包/模型/validator、worker及其他受保护路径保持。无新Trainer、无已通过模型/profile/full/checkpoint/算子重复。人工学习seed42与正式来源/profile seed43分明；原25前置、真实exit/清理和产物哈希先于结果索引，500原heldout学习标准保留。

本刻尚未新入队；下一项仅原池一次25，原验证通过再一次500。真实数据/训练0，审核、selection、resume门禁及停线保持。测试ckpt0，自有空临时目录按实际清理，必要证据/实际恢复状态保留。

[新增检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/CPU_control_results.json)；[实际接入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/publication.json)；[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/CPU_owned_cleanup.json)。

## 2026-10-02T06:48:24.130919+00:00 正式调用剩余项独立源码核实

两agent独立只读与root核实一致：当前冻结ENTRY.execute_admitted已具备原数据优先preflight、同源500学习、完整GPU/来源/seed及1000→5000完整恢复链，无须新增Trainer；缺的是正式请求/冷隔离加载的薄接入层。旧v4 seed launch永拒且只证明软件descriptor，不能当正式接入已完成。私有源码绑定可先实现，实际selection验证会重建并读真实数据，必须等待有效审核；不得通过运行该路径制造“准备完成”。当前修正版FULL仅R21/seed43，旧七份证明不迁移或并成当前8/8，learning42/selection42/formal43区分。

前一Goal回合为实际progress（4/4两环境、最小接入和唯一25已运行）。当前同一GPU1外层PID477484仍有poll-gated heartbeat：原worker只有process.poll() is None时更新heartbeat，源码SHA未变；不是只凭一个Running字样。25无终态，原观察继续，未重启/重试/新增任务。真实训练0、所有停线及数据/selection/resume门禁保留。

[独立源码与活进程证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/formal_call_remaining_readonly_review.json)。


## 2026-10-02T07:23:48.263825+00:00 八卡开训范围与剩余条件

用户已授权条件具备后在原八卡资源开始训练，无须重复请求权限。当前正式数据版本确为 immune_v7_mhc_sequence_v1，train6,849,243/valid52,572；manifest仍e12e55cc…ba94，training_ready字段缺失、audit_pending、当前audit_certificate.json与downstream_scope_certificate.json均缺。pending为父数据当前范围审核未完成及更新manifest未被父证书覆盖，父证据另有unresolved_benchmark_h3；既有清理保留，不表示已确认泄漏。两份审核分别覆盖当前数据清理/划分与下游test、few-shot support/query等保护范围，均须绑定最终当前manifest，不能由目录名或修改ready代替。

候选R20/R21/R24/R26代码实现已完成，七份旧来源完整GPU验收及修正版R21/43完整验收各自有效；不能并称同一源码8/8。Root本轮仅metadata核实八项profile与各自冻结runtime来源匹配，其余七项各自当前evidence_root中没有25/500学习记录；不迁移旧pool记录补齐。现有CE修正版25已原标准通过，唯一500 job 1790924220633707073_d04_r21_CE_learning_v1_dd5651b2在原GPU1运行，outerPID482000，最新已发布50/500（非终态）；四mask均值loss=1.350228、AA=52.34%、terminal准确率=0.00%，不得当真实理解/生成效果。正式训练薄调用尚未接完，最终架构尚未选择，当前不具备真实八卡开训条件。

下一步仅收尾既有同源synthetic学习前置和最小正式调用整合；已通过梯度/显存/profile/full/恢复验收不重复，不增加新架构波次。数据有效审核与对应selection绑定是实际语料初训的原前置，满足全部原门禁后按四候选×两seed并行初训，再依实测选择阶段架构。当前真实shard/小batch/训练0，新增入队/Volc提交取消/容器worker监听重启0，验证checkpoint0；原Goal身份/累计用量保持，停止序列注释不恢复/转交，独立CUDA算子继续暂停。当前人工500运行中，不能把整个Goal标为完成或仅剩外部审核。

[本次只读原件及八项矩阵](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/training_readiness_20261002T072348Z.json)。

## 2026-10-02T07:26:59.319320+00:00 七项缺失人工学习前置有界整合

七项缺失的当前来源人工学习前置已冻结为真实逐slot绑定和有界预算。仅读旧强复核/COMPLETE/result/profile/sidecar及SHA，不重跑已通过GPU/full。新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002 仅允许3薄source、每环境4新增stdlib控制/180秒；每slot至多一次25及原验证通过后一次500，同一自己的evidence_root，learning42/micro32/GA8/global256、formal/profile42/43分别保留。复用现有原生成器人工manifest，不新增sequence/shard/数据。当前CE500及所有旧验收不重跑，原engine/FULL包/worker/validator/data-scope-selection-resume全部保持；停止注释不恢复，独立CUDA暂停，真实shard/训练0。此刻未发布或入队，下一步实现/源码审阅/4控制通过后才执行缺失项。root首轮metadata准备误把旧review.complete_sha字段对GPU_COMPLETE比较，已实际读schema纠正为原COMPLETE；首轮仅建空自有目录，无测试/model/job，错误元数据保留。

[绑定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/slot_bindings.json)；[预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/execution_budget.json)。

## 2026-10-02T07:28:55.860695+00:00 七项学习识别项最小发布diff暂存

仅生成reviewable publication diff，未应用：runner一个入口路径literal，registry追加七个launcher_only/synthetic_learning_only识别项，旧registry项及模型函数AST保持；之后新父入口将委托所有非七新alias到固定CE父，当前CE500不受影响。两新source与4合同仍准备中，须实际审阅/冻结和首轮4控制通过才发布/入队。当前公开ROOT/worker/运行包未改，真实训练0。

[暂存diff](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/publication_draft.diff)；[literal复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/literal_publication_draft_review.json)。

## 2026-10-02T07:37:29.854839+00:00 正式训练资源合同实际阻断核实

正式入口源码/路径审计发现此前“只需薄controller”的判断不完整：五冻结runtime各缺两类下游审核资源目录，共10路径；原scope输入根绑定runtime内部，package禁止额外文件/链接。正式调用需要新版本显式外部资源合同并保留全部原审核条件，不能修改旧包或monkeypatch/改ROOT绕过。另原assert_ready未显式检查training_ready，正式外层必须按用户要求strictTrue；WORLD_SIZE=1只支持八个独立候选初筛，不等于单模型八卡DDP。当前仅读源/路径metadata，无gate/selection/真实数据/训练，原安全停线保持。

[精确缺口与最小方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FORMAL_RESOURCE_BINDING_GAP_20261002.md)；[原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/formal_resource_binding_readonly_gap.json)。

## 2026-10-02T07:39:19.214520+00:00 七项学习source实际审阅和冻结

Root已实际逐行读取child228行/parent249行及4个新stdlib控制；A独立静态复核完成，三源SHA已固定，所有保护旧source未变。仅7slot固定binding、原引擎及原退出清理API，无新Trainer/模型/审核逻辑；下一步只首次运行这4控制/pllm和worker各一次/180秒，检查软件元数据和拒绝边界，不导入ML或重跑旧GPU/full/math/profile/恢复。暂存literal diff尚未应用，新增job0，真实训练0。

[三源冻结](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/source_freeze_before_CPU.json)。

## 2026-10-02T07:40:55.741762+00:00 七项学习四控制通过和最小接入

七项学习的4个新增stdlib控制在pllm/worker首次各4/4通过，elapsed13.690/14.348秒；root核对实际logs/receipts/三源SHA通过。仅发布新父入口字节副本、runner一个路径literal和七条synthetic-only launcher识别项；旧registry项/模型函数AST、所有FULL/模型/validator/worker及当前CE父child保护来源不变。没有新增Trainer、旧验收重跑或独立CUDA算子。当前未新入队，下一步每缺失slot一次原25，原验证通过才一次500；六个准入flags仍false，真实shard/训练0，停止注释/数据-scope-selection-resume门禁全部保留。当前CE500已实际发布75/500，无终态。

[首轮4控制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/CPU_control_results.json)；[最小实际接入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/publication.json)。


## 2026-10-02T07:49:44.329851+00:00 八项人工学习的模型与seed口径

Root直接读原件确认七项25均在原GPU上实际运行，poll-gated heartbeat新鲜；GPU0 R20/42、GPU7 R20/43、GPU5 R21/42、GPU2 R24/42、GPU3 R24/43、GPU6 R26/42、GPU4 R26/43，GPU1保留CE500。七项inner已实际发布initial，source39匹配各自原runtime、config和人工fixture与CE相同；d768/8层/12头/MLP3072、parameters75,601,176或headgate75,674,904，learning42/micro32/GA8/global256。七项已发布trace为空，只表示尚无发布检查点，不推断实际优化步数；25未验收、500未新入队。CE已发布100/500，四mask均值loss0.565483、AA81.29%、terminal准确率21.94%，仍非终态；此处不称真实理解/生成质量。池0/8/58，实际Volc提交取消及容器/worker/监听重启0。

[八卡实际初始进度](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/artificial25_initial_status.json)。

人工学习协议统一seed42；同方法profile/formal seed42与43的人工项不能当两条独立随机种子学习重复。两个正式seed的真实初筛仍未开始，后续1000→5000及最终单模型多卡需另行准入和实际证明。

## 2026-10-02T07:58:40.899023+00:00 正式资源与单模型DDP必要接口的独立源码审阅

A/B实际源码审阅完成：正式资源合同须成为实际preflight/selection/build_trainer/execute/resume参数，新checker再调用旧assert_ready仍失败；不能通过短资源列表缩审核范围，完整inventory/scope版本仍依赖有效当前审核。单模型8卡还需global256的world因子（micro32→GA1）、metrics全局计数只reduce一次、8实际rank RNG/完整同拓扑恢复及rank0共同文件，原worker只单GPU需显式8rank组协调；HF/Accelerate基础可复用，生产源未改、新DDP测试/GPU/训练0。旧模型/CE/dataset可保持，不用旧full证明授权新adapter/ENTRY来源。

[精确缺口与实际最小接口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EIGHT_GPU_TRAINING_INTEGRATION_GAPS_20261002.md)；[源码SHA原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/formal_and_DDP_interface_readonly_decisions.json)。


## 2026-10-02T08:01:32.480361+00:00 同源学习诊断进入原500步骤

七项当前来源500实际各入队一次，原七25已由root独立原verify_diagnostic通过，原stage500 gate/own-source25/人工manifest语义四runtime冷进程全部通过。只复用原已生成人工manifest；固定learning42、micro32/GA8/global256、fresh scratch不resume25，formal/profile42/43分开，原own evidence_root与39源保持。每项28800秒原预算，终态必须原verify_diagnostic(500,require_learning=True)真实heldout学习/完整产物及exit/清理，当前未验收。既有CE500继续未重复；本阶段新增平台submit/cancel/restart0，使用原8卡非闲时t-20261001033327-fmnql，真实shard/小batch/语料训练0。所有data/scope/selection/strictTrue/完整resume及安全停线保持，独立CUDA算子仍暂停。

[七项唯一500启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/launch500.json)；[原500前置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/artificial500_preflight.json)；[原25终态强复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/artificial25_terminal_root_review.json)。

## 2026-10-02T09:16:09.780145+00:00 数值Trainer主case已实现

当前四私有来源已交付，验收范围仍是原3连续与2→3恢复/8原生RNG/实际Adam及scheduler/4mask×13与17尾批。保留可重算raw，不称模型效果或GPU/DDP/正式训练通过；静态独立复审完成后才首次CPU，每环境最多一次，不重复旧验收/扩大架构范围。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/static_source_review_current.json。

## 2026-10-02T09:19:16.320175+00:00 单模型八rank工程首次数值验收来源确定

四源冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/source_freeze_before_CPU.json；实际CPU/Gloo验证尚未执行/通过。以原连续3和2→3恢复检查新Trainer工程，训练器schedule horizon10000不改；当前只numeric fixture，不能用来选免疫模型架构或声称正式八GPU数据训练完成。

## 2026-10-02T09:27:18.509373+00:00 实测CPU优化器恢复兼容修正

worker world8原nativeHF map_location=cpu:0 实测失败，pllm同路径从源推断但未执行。只在numeric CPU参数getter规范化cpu，并补阶段与验证receipt留存；保持world8和native加载/安全allowlist/原阈值，真实模型与生物数据不改。新2source修复独立来源验收，不覆盖旧v1失败，不将CPU案例当实际GPU/DDP/正式准入。预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/execution_budget.json。

## 2026-10-02T09:32:44.629446+00:00 CPU工程修正版与准入边界

最小两源静态复审/冻结完成，真实效果待首次同主case验收；新SHA不复用旧v1通过声明（v1未通过）。当前仍generic numeric fixture，无GPU/DDP/真实模型选型或正式训练准入。来源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/source_freeze_before_CPU.json。


## 2026-10-02T10:03:47.608079+00:00 本轮不追加架构，补旧worker动态claim与冷rank边界

复用原模型数学/TokenValidationTrainer loss及loop/_save；新synthetic构造器不等于正式生产入口，原资源审核/selection/新来源GPU/full-resume仍缺。 新记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md；预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget.json。安全停线、独立CUDA暂停、原data-scope-source-selection-learning-profile-GPU-full-resume门禁保留。


## 2026-10-02T10:09:29.380497+00:00 原词表列表保存格式的实际兼容修复

原模型_save保存vocabulary.json为list，冻结numeric callback要求dict，无法直接接入。新显式私有回调 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py 仅_metadata保留原非空unique字符串列表、其余3dict，其他全部函数AST不变；不用FunctionType/globals或改词表包装，旧源/阈值/门禁/native/8RNG不改。新SHA 21815b3fd54ee51481f72f77cce1879d2fdf9d0e4818665c2cf72531169084b8 不继承旧GPU/CPU通过声明，当前未运行。新增1个必要source，修订预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget_v2.json SHA 43ed1ccfc4cbeb71dbaf58d82781e3d1e6c8a473dab63d4b8ccebf0598a6657e；原未执行预算保留，CPU控制仍4/env180秒、GPU/发布/真实训练0。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/checkpoint_vocabulary_source_delta.json。


## 2026-10-02T10:31:00.164239+00:00 八进程通信源码冻结与首次控制边界

5个新私有source及4个复用source已冻结，26旧保护文件未改。A/B接口复审完成；root补自有进程清理helper抛错时的有界fallback，C只修顶部CLI说明。首次每环境4项CPU transport/metadata控制待执行；不重跑已通过Trainer。此阶段无GPU作业/真实数据读取/生产准入，实际八GPU共同Trainer和原native词表保存恢复仍待验证。 来源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/source_freeze_before_CPU.json；详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md。


## 2026-10-02T10:41:27.020725+00:00 新增通信独立复核闭合与原任务新进度

root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool0/8/65，目标8项结果及COMPLETE均缺。新快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_readonly_continuation_1037.json 保留旧已绑定快照不改。不重跑通过测试；实际GPU/生产保存恢复与V7审核均未完成，真实训练0。停止注释/CUDA暂停/原资源无重启或重复提交及全部门禁保持。


## 2026-10-02T10:44:51.097287+00:00 当前收尾与剩余外部资源阻断

新增八进程通信/清理的两环境首次4/4及独立raw复核已闭合：root88与独立103原件哈希重新核实，9冻结source/26旧文件不变，temp及checkpoint均0，不再重复已通过控制。实际单模型八GPU训练/原Trainer构造/native词表保存恢复仍未验证；生产入口未接通，真实语料训练0。

正式接入的真实阻断是缺当前V7合法完整scope/input inventory/registry与报告绑定格式，不仅是两份审核尚未passed。调用链会重复原固定资源gate、请求重建丢附加字段，selection还要求旧payload完整重算相等；自行选短path/SHA清单会缩减原保护范围。不能补假清单/证书、手改ready、monkeypatch旧ROOT或加入永拒placeholder充数。证书未通过本身不禁止通用开发，但没有权威完整资源契约就无法正确定稿这部分formal gate。

外部必要材料：当前scope_version及完整逻辑输入清单；批准的外部根、逻辑ID→绝对路径映射及文件/registry/evidence哈希格式；绑定当前manifest和同一输入全集的双审核字段语义；各候选/selection/执行/续训共同使用的同一版本资源契约。未来真实执行仍要求training_ready is True和两份有效当前审核，不降低任何原门禁。

当前可继续核实同8原任务的实际500终态并各自一次原verify_diagnostic(500,require_learning=True)；它们仍Running时不得因观察超时重启。后续单模型GPU/正式接入须满足当前来源及资源条件，不能用新numeric/GPU测试填补外部资源缺口。待权威契约到位，最小后续六类source为参数化资源gate、selection外envelope、正式request/preflight/build、execution/fullresume、cold entry及immutable package；此处仅明确必要边界，没有创建或执行六类实现。

只读决策 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/formal_resource_remaining_implementation_decision.json（d2631a1f…ba5c），root物理核5原source SHA及10实际调用位置；CPU阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json；10:43实际平台Running原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/platform_task_readonly_continuation_1043.json；10:38同8原worker poll-gated/trace仍CE450、其余325，无目标终态。

此前停止序列注释分支不恢复/重试/转交/绕过；独立CUDA算子暂停。原容器/作业/worker/监听不重启，不重复平台提交；不读取真实shard/抽真实小batch/执行审核selection/开始真实训练。全部原data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。

## 2026-10-02T13:02:01.736603+00:00 V2最小锁协调修复准备

前一Goal回合实际核实V1八项终态失败、模型0步并定位长临界区，分类为progress；原Goal身份和累计保持。新有限预算/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_budget.json，SHA29eb09bf8449ba92101db33be7fe614d0a2f300799a4340e3a05e7b91989fa7c，保留V1已耗尽预算及全部失败原件。五个私有源已机械迁移到/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2；只有bootstrap将做通用锁行为修复，其余训练/保存/恢复逻辑保持，来源尚未冻结、未运行新控制。V2采用g02八个新成员名与后续全新jobIDs；准备阶段不stage/发布/入队，不重放g01失败job、重启旧资源或重复平台提交。

B只编辑V2bootstrap，C只准备一个stdlib定向CPU锁控制源码，A只读复核其余四源路径/预算重绑定；最多4项新增控制/环境/180秒，不重复原500、FULL/数学或已通过CPU/Gloo套件。真实数据门禁及停止注释、独立CUDA暂停保持。修复须在锁外完整核对来源、用两阶段确认防止未核对成员放行并记录实际锁耗时，不能简单增加时限或删除核对。当前八卡实测最新状态仍是V1启动失败、模型0步、worker回维护，V2源码准备不算八卡通过。

## 2026-10-02T13:35:29.543004+00:00 V2锁修复完成，静态问题已修正，首次CPU控制准备

五个V2源已交付并冻结完整545个物理输入；当前bootstrap SHA2e81b0b8b3cfd439693e5b95faa64ac244861a961c2ef958443b6cb572fd856c，freeze SHA22a20b96a1e7c43cc0741f98ac458593483f2c77500790a5a552cd7345f36435。独立复核发现初稿20380把原post-publication完整检查移到commit前，root在未执行任何控制/源码/任务时仅补一行assignment_after_commit锁外完整检查，保留初稿字节、初始freeze e979及原finding报告于/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/initial_review_bytes；修正复核/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/bootstrap_postcommit_correction_independent_review.json确认唯一差异和原期限/接口/35旧函数保持。未删除问题记录或改V1失败预算。

成员状态pending→checked、八成员generation/source/snapshot两阶段提交、完整来源扫描/协调器初始化在锁外、真实硬件/父子身份与持久abort保持。三个非bootstrap训练模块函数体不变，dispatcher只必要路径/新g02身份重绑定。四项新stdlib CPU控制源码SHAe301ccf21b3c2935ea6158b0d8ae9c9becc925e87f2894a49eb91585f02c5274，已独立静态复核。Root物理核对545输入、14旧保护及四份review；执行计划/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_execution_plan.json，将各环境首次最多4项/180秒，当前尚未执行控制/模型/GPU/stage/发布/入队。正式V7训练仍0，全部安全停线/审核门禁保持。

## 2026-10-02T13:41:46.940748+00:00 V2锁控制两环境首轮各4/4，八成员启动文件已准备待复核

实际父进程session73874 rc0结束，两环境并行总4.9816秒；worker4.7145秒/pllm4.6881秒，各4项首次通过，未重试。每环境8个member CPU子进程+1个holder均实际rc0/wait/PGgone，临时fixture已删，源/14旧保护/545来源不变。2.25秒锁外有限慢检查期间实际获取锁仅worker0.001245秒/pllm0.001999秒；actual2秒超时worker2.009051秒/pllm2.008957秒，持久CPUabort与后续拒绝生效。实际来源字节改变拒绝、generation/source/snapshot三拒绝及纯CPU合法assignment提交均通过。每环境保留12份原始计时JSONL及命令/日志/hash；此结果只证新增CPU锁协议，不证明共同GPU/model/NCCL/save-resume已通过。原件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_first_execution_receipt.json，独立raw复核正在执行。

一次metadata-only stage于2026-10-02T13:39:26.830527+00:00实际rc0结束/2.2419秒，文件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/stage_v1/stage_metadata.json和/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json已生成：八g02成员/八新ID/six-update合成计划，无GPU主机/intent。stage字节独立复核正在执行；本准备turn不publish/入队GPU，预算仍保持其原禁止字段。最后一次已有共同GPU运行仍是V1 failed/模型0步；V2 CPU通过不改变该历史结果，旧500/CPU/Gloo/FULL不重跑。真实V7数据/小batch/训练未启动，双审核及全部安全停线保持。

## 2026-10-02T13:53:26.861822+00:00 原Goal下一执行阶段开始：固定新八成员纯合成验收

上一Goal回合为progress：V2 CPU两环境各4/4及暂存字节实际验收完成。本回合沿原Goal身份/目标/累计，执行阶段有限预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_phase_budget.json（SHA 415ece86836d13f755619621a2814384002c1b9734af89179286932996f673f8）；原准备预算29eb和禁止本准备turn字段不改，已完成控制/500/profile/FULL/Gloo不重跑。新执行阶段至多发布三份既有独立复核暂存字节，并使用既有固定八个新g02 jobID一次入队原池；不重新stage或生成ID，不使用会随机生成ID的普通enqueue CLI。新源码修改上限0，GPU组1/8成员/1800秒/70GiB；实际训练/原生保存恢复结果尚无。

已按task ID实际执行volc ml_task get -i t-20261001033327-fmnql -o json，返回Running；此前带bioseq名称过滤的list返回[]，不据此推断资源终态。没有平台submit/cancel、容器/worker/监听重启。两个子agent仅只读检查当前执行前置与传输路径；独立审查和新鲜资源/来源/原证据核对通过后才构造真正synthetic-only启动记录并入队。全部正式数据/GPU/训练准入仍False，真实V7小batch/训练0，停止注释及独立CUDA暂停保持。

## 2026-10-02T14:03:48.274562+00:00 V2固定共同八卡纯合成组一次实际入队

14:02:17 UTC通过原共享池发布既有固定8个g02 packet，group=original_r21_world8_1790948366803074817_36c1a3d5；启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_enqueue_receipt.json（SHA bc9285806e789f2124ac10db4b03ecc5f51112041f5526ecd4add555aa9cc647）。仅三份已独立复核暂存字节发布，原生545来源不变、12旧保护未改、两个public旧文件为明确已审阅变化，另新增V2 public dispatcher，runner最后发布。未使用随机ID enqueue、重stage、重放V1、重启容器/worker/监听或Volc提交取消。独立执行审查已完成，新synthetic-only启动记录绑定115实际前置原件；全部正式准入仍False，V7小batch/真实训练0，停止注释/独立CUDA暂停保持。

当前首次只读状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_fixed_group_status_after_enqueue.json：pool={'pending': 0, 'running': 0, 'results': 89}；具体8job是否被领取/真实group intent以原件为准，实际Trainer/DDP/保存恢复尚未验收，不能把发布入队当成功。一次尝试预算已消耗，不自动retry或再发布pending。有限监督subagent只读观察同8个具体任务至实际终态/有限窗口；观察超时不表示原job退出。真实产物复核通过且各rank退出后再清理自有临时合成checkpoint，原证据及实际训练恢复状态保留。

## 2026-10-02T14:21:09.798620+00:00 V3通用协调修复准备

上一Goal回合为progress：V2实际8注册/8启动领取后锁超时，终态及清理已核实。本回合新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3；预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/execution_budget.json SHA d552c60b66d12a21785fb022fda997a366601638ddacd6d5b291a1f322fcaf71。仅修复成员/协调器嵌套锁、状态观测持锁及重复查询，保留模型数学、所有身份/来源/时限/abort/退出门禁。六份私有源码准备中，当前协调器SHA待最终绑定，未冻结、未测试、未发布或GPU入队。旧V1/V2一次尝试均耗尽且不可重放，现有资源不重启。V7真实训练0，双审核缺失；停止注释及独立CUDA暂停保持。


## 2026-10-02T14:46:31.239832+00:00 V3私有coordinator源码交付：source-only，尚未执行

仅修改 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/eight_rank_coordinator_v2.py，当前源码SHA `4e0de6787d9dcfca7aace1f0b53e370030bbfaa85f83cf3150641a52a68424e2`。`group_status` 在原协调锁内取得独立元数据快照，原 `/proc` 存活探测移至锁外；探测完成后再用原 `_locked(allow_aborted=True, abort_on_lock_timeout=False)` 做一次短元数据核对，刷新abort/phase/terminal并保留其最终commit fence。注册、心跳、启动令牌、实际父子身份、来源、2秒锁时限、持久abort及退出清理语义保持；不得把观测快照或source-only交付当作GPU/训练准入。

`_locked` 增加真实wait/hold/commit计时，只在锁已释放、文件句柄已退出后追加到自有组目录的 `coordinator_timings_<pid>_<native_tid>.jsonl`，每文件最多128条且131072字节，按实际PID/线程分文件并以追加写入避免覆盖，诊断写入异常不改变原协调结果。这些是下一阶段定位所需的源码，不是当前已产生的GPU计时。纯stdlib AST解析确认原31个函数/类保持、仅原 `_locked`/`group_status` 改变，新增 `_coordination_timing`；未导入或执行coordinator、模型或任何CPU/GPU控制。

独立静态复审指出初稿在慢probe期间可能漏报随后发布的持久abort或正常abort_group状态；尚未冻结/执行时已按root要求补上述最后短事务。初稿SHA `25784e6c3abd6aa9d35ec2647789d8960af4060b8b99d06b8d476a2515f3342a` 的原字节保留于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/initial_coordinator_draft.py.txt。V2最初coordinator持锁者与具体慢操作仍缺实测证据，不归因为已证明的死锁或I/O；V1/V2失败与已耗尽尝试保持不动。

同次只读核对候选bootstrap `2011bd152bc9ad6f44c3e849141f717da8a0692073a1194139c47ebd0d90269c` 未见成员 `_locked` 内的coordinator调用；候选cold `bdd5f93d5a6598855b7d0f2c00b89e30d9e630331d0076ca78c40b48a52f3631` 去掉重复状态查询，仍每0.05秒检查退出/期限、每0.25秒读取一次包含协调器abort的组summary。此结论仅对应所列候选字节，尚待root绑定实际coordinator SHA、冻结和后续有限CPU验证；不是整组运行或保存恢复通过。有限预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/execution_budget.json SHA `d552c60b66d12a21785fb022fda997a366601638ddacd6d5b291a1f322fcaf71` 不变。

本次source-only交付未测试、未stage/publish/enqueue或重启原资源；模型数学、Trainer、旧runtime/public源码及阈值未改。正式V7小batch/训练仍未启动，原training_ready/双审核/source/selection/full-resume门禁不降低；安全停止序列注释不得恢复/重试/转交/绕过，独立CUDA算子保持暂停。

## 2026-10-02T14:49:51.778353+00:00 V3启动锁修复已冻结，定向控制尚待首次执行

已完成六份私有源及独立静态复审；完整545输入冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/source_freeze_before_CPU.json（SHA546cec975bfc6fbfac9800ea5db21d073008fecee8ccf3197b80c81dcf6f879a），16份V2/公共保护文件保持。原Trainer driver与checkpoint validator所有函数AST不变；cold只去重状态轮询，bootstrap拆开两层锁；coordinator进程探测移至锁外并保留探测后的最终中止检查。初稿缺最终中止检查的问题及原字节均保留。实际source_metadata核对545输入已通过且没有ML导入，不把来源核对称作CPU并发/GPU验收。

唯一新CPU控制源码准备中，尚未执行，未stage/发布/入队新GPU组。14:35:34 UTC按原task ID查询平台仍Running；原池pending0/running0/results89，8 worker状态均以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/resource_readonly_during_repair.json 为准，不把历史heartbeat当实时存活。V2失败终态原件保持；没有新提交/取消/重启。上一状态答复只复述已有证据，本续接已实际改动通用工程并冻结源码，属于progress。下一步仅首次执行新锁竞争路径的有限CPU控制，不重复原500/profile/FULL/Gloo验收。真实V7训练0、双审核缺失，安全停线和独立CUDA暂停保持。

## 2026-10-02T15:18:38.993880+00:00 V3八卡synthetic执行阶段开始

上一Goal回合为progress：新锁修复在两环境首次各3项通过，22自有子进程实际退出并清理，三份暂存字节已独立复核。本阶段预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/execution_phase_budget.json（SHAc00c6f961ea0c4b47e89f07e62a33b9079e08270913ee53f89e09db5c158bbcb），原准备预算和冻结源不改：源码新增/修改0、旧验收重跑0、重stage0、最多发布三份已审阅文件并将既有八个固定g03 ID入队一次。synthetic六实际updates/rank、world8/global256/1800秒/70GiB，不代表真实语料训练。

15:17:02 UTC实际volc get确认原task t-20261001033327-fmnql Running；8 worker各standby_load/maintenance_only、心跳约3.75秒，原池0 pending/0 running/89 results。545当前源、16保护及115原前置原件物理hash一致；原CE与其余7项500验收保留接受状态，无重跑。此时尚未发布/入队，本阶段将完成实际synthetic启动与终态核实；不重启原资源，不重复平台提交，不恢复停止注释或独立CUDA。正式V7双审核及所有真实训练门禁保持。

## 2026-10-02T15:40:28.282800+00:00 V4只读观测异常分类修复准备

V3失败组已实际终态并清理。独立新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4，预算SHAb6e899550495a2743fbab12193ff7c3502741cb70644c8b144a57cee31c8240d，仅修复纯观测锁超时被上层误升级abort的问题。六份源已机械复制，下一只改coordinator明确异常类型与cold生产共用监督循环；最终source pin待绑定，未冻结/执行/发布/入队。原2秒锁时限、30秒心跳freshness、启动/总期限和持久abort/身份/来源/退出门禁保持，不依赖错误字符串判断。最多两项定向CPU场景/环境包含必要拒绝分支，不重复旧控制/500/Gloo。V3及全部前驱证据和既有资源保持，不重放旧ID，不恢复停线分支，不读真实语料。


## 2026-10-02T15:46:19.341298+00:00 V4类型化状态观测源码交付：source-only，未执行

私有模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/eight_rank_coordinator_v2.py 已交付，SHA `dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001`。新增 `StatusObservationUnavailable(CoordinationError)`，仅原 `group_status` 实际代码对象发起、`abort_on_lock_timeout is False` 的只读锁acquisition在原2秒期限内未取得锁时产生，字段为group_directory、contract_sha256、operation='group_status'、abort_on_lock_timeout=False。其他CoordinationError、mutating/member锁错误和实际持久abort仍fatal；调用方必须核对实际来源和同一模块实例的异常类型/字段，不能按异常消息或宽泛ValueError分类。

新增 `raise_if_aborted(directory, contract_sha256)->None`，不取得协调锁、不写文件，只复用原严格JSON读取、validate_contract、digest和_merge_abort_request，核对原state/contract/group/host/boot与全部metadata/准入False绑定并合并实际持久abort-request。普通state abort、已持久request或aborted终态立即raise；返回None不等于健康、成功或生产准入，仅供有界重试前检查。原group_status快照/锁外probes/最后短事务abort fence、计时、原心跳和身份/退出门禁保持。静态AST仅原_locked变化，另新增两个定义，其余33个原函数/类AST相同；未导入或执行该源码，没有CPU/GPU控制结果。

有限预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/execution_budget.json SHA `b6e899550495a2743fbab12193ff7c3502741cb70644c8b144a57cee31c8240d` 保持；后续实际shared supervisor来源组装、来源冻结和必要定向CPU控制由root统一安排，此source-only交付不称接入或八卡Trainer/save-resume通过。本次仅改此V4模块及同步本条MD，没有改前驱/public源、测试、发布、入队或重启资源，不重放V1/V2/V3已消耗尝试。V3实测3.835秒hold/3.830秒commit已记录，但commit内具体慢操作与GIL/fsync因果仍未知，不能把类型化错误修复当作已解决该慢操作。正式V7小batch/训练仍0，原data/双审核/source/selection/完整恢复门禁和停止序列注释、独立CUDA暂停保持。

## 2026-10-02T15:56:03.273624+00:00 V4生产监督循环修复进行中

上一状态答复仅核实/复述已有终态，按Goal规则属于no progress；本续接继续实际修复。V4 bootstrap已绑定交付coordinator SHA dbaf3e8536370309a42fd2c33d23b020224b62d981451b2038387a21009e4001；cold共享监督循环及唯一新CPU harness由现有A/B交付中，尚未冻结或执行。下一步是检查实际生产调用同一监督helper，再首次验证短暂观测失败后恢复与真实故障拒绝；不重跑已验旧实验，不重启资源或重复提交。当前失败不是平台提交失败：V3已实际启动八个子进程，失败由状态观测异常升级中止触发。

最新实际GPU结果仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_GPU_terminal_root_review.json；八子进程已清理，暂无成功Trainer步/恢复证据。当前准备目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4；真实数据小batch/训练0，安全停线、原审核及独立CUDA暂停保持。

## 2026-10-02 V4共享监督循环已落盘，尚未运行

Root因A未交付源码中断其编辑任务后，直接在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/gpu_worker_cold_rank_v1.py 实现supervise_process_observations并由实际supervise_rank_child调用。只容忍同一模块类型/组目录/contract SHA绑定的group_status纯观察超时，失败不刷新健康时间；原30秒freshness及启动/总期限仍检查，持久abort、非专用错误和非零实际child退出仍拒绝。退出成功也要求可用状态与八rank启动确认。观察记录放在外层member_cleanup证据，不改原严格receipt字段；coordinator SHA与bootstrap均绑定dbaf3e。AST解析通过不等于测试通过；B正在交付针对同一helper的唯一CPU harness。未冻结、执行控制、发布或入队。

## 2026-10-02T16:08:42.414954+00:00 V4来源冻结完成，等待新控制首次运行

六份私有源及完整545来源已冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/source_freeze_before_CPU.json（SHA 2123626cf035d387173b8729fdd4c91596a0cdd29ebf202891d60863759fc7af），16份前驱/公共保护文件未变。根端AST核对只改变监督/异常分类和版本路径，原模型/Trainer/保存恢复函数保持；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/root_source_static_review.json。尚未导入执行新源、CPU控制、GPU、发布或入队；下一步必须实际运行共享监督循环的必要控制，不能以冻结代替验收。

## 2026-10-02T16:12:33.722727+00:00 V4新监督控制准备完毕

唯一新控制 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/supervisor_observation_controls_v4.py（SHA 06737954763c38de2b323eecef9186f6b64d929909b0c5abfb9a5152dfc7e332），生产同一helper覆盖真实8 CPU子进程在实际锁超时后恢复，以及持久中止、非专用错误、首状态不可用原30秒界限、缩短CPU启动/总期限和真实非零子进程退出拒绝。后两期限明确是CPU参数控制，不冒充180/1800秒GPU实验。B已独立静态确认root cold源无必须修改项；其harness草稿由root接管收尾，未运行或发布。源与控制审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_pre_execution_review.json。下一步两环境首次各两项，180秒上限，不重跑旧验收。

## 2026-10-02T16:21:33.119075+00:00 V4准备完成，独立复核接受；尚未启动新的八卡组

独立复核实际结果与暂存字节通过，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_and_stage_independent_review.json。证据限制明确：持久中止测试是在group_status超时已抛出后、共享helper catch前发布abort，证明catch内中止检查生效，不声称在锁等待过程中发布。没有因此改源或重跑。准备阶段完成记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/preparation_completion.json 仅指CPU修复和启动文件准备，原Goal仍active未完成。

当前只读V7清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/V7_manifest_only_readonly_status.json：manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready=None，两份审核文件存在性分别False/False。未读真实分片或启动真实训练，门禁与安全停线保持。下一执行阶段使用已有固定g04计划，至多一次原容器纯合成八卡Trainer保存恢复验收；不重stage、不生成另一组ID、不重启或平台重复提交。

## 2026-10-02T18:55:52.751454+00:00 V7真实小样本架构比较准备

按用户最新要求，暂缓正式训练并从已物化V7抽少量真实样本。两位subagent审阅后选择R20/R21/R24/R26×训练seed42/43八项独立单卡；抽样/入口实现中，尚未执行。最新final V7已存在限定范围passed审核与ready=True（其他工作流程完成，不是本轮修改），不再套用旧版本缺审核结论。方案、来源与比较局限见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SMALL_SAMPLE_ARCHITECTURE_PILOT_20261002.md；原容器/监听保持，停止注释及CUDA暂停保持。

## 2026-10-02T19:03:26.562307+00:00 V7探索扩大至十万条并筹备完整消融图

用户明确改为100000 train，valid1024；四候选×两seed不变，计划global32、3125步、eval250。已完成父数据限定范围证据只读核验（293 evidence files），未创建或改动审核；实际样本抽取及GPU训练尚未开始。抽样脚本已落盘，准备执行必要fixture；已确定实验通过入口核验即可先排入原池，无需等后续组合方法。详细方案 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SMALL_SAMPLE_ARCHITECTURE_PILOT_20261002.md。

## 2026-10-02T19:08:18.446121+00:00 十万条V7抽样完成，完整消融图计划登记

实际抽样train100000/valid1024成功，原行/split保持；source分层、SHA与count已核实，样本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample/sample_manifest.json。尚未启动GPU训练。完成五份原始论文/官方指南查阅，消融设计为四项初筛、三因素八条件组合与逐项去除，另列固定门控/同等调参预算对照；最终六面板图及日志字段见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ARCHITECTURE_ABLATION_FIGURE_PLAN_20261002.md。已确定首批实验不等待组合实现。

## 2026-10-02T19:11:34.656162+00:00 V7入口检查与模型源码冻结

新增runner的CPU功能检查首轮失败于torch动态模块伪文件_classes.py，不涉及模型数值；修复source_hashes仅纳入实际存在文件后，第二轮2步人工入口检查通过，初末验证loss 4.234493→4.141472。只验证新入口，不当作真实效果。已清理该次自有latest.pt并保留日志、完成与失败证据。冻结563个模型/入口Python文件于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/source_snapshot_v1；新增既有pool分派器尚未接入/入队。十万条实际输入布局全扫描进行中，不截断或替换失败行。

## 2026-10-02T19:14:30.135931+00:00 首批V7实际架构实验已进入原八卡池

独立review与8别名参数检查通过；实际101024行布局全通过。已最小更新公共runner入口路径和8个launch-only alias，原70项注册不变。首批R20 seed42、R21 seed42各3125步已入现有池（不是新平台提交），job IDs：1790968470149719891_p01_v7_r20_seed42_838aa052, 1790968470161653699_p01_v7_r21_seed42_ccb7e981。此时只有入队证据，实际优化步待观察；其余6项在相同入口开始初始化后接续入队。没有重启容器/worker/监听。新MD方案及冻结源码参见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SMALL_SAMPLE_ARCHITECTURE_PILOT_20261002.md。此前未发布的V4 world8 stage属于历史方案，其公共源码前置已改变，不可再直接发布旧stage。

## 2026-10-02T19:17:16.170057+00:00 八项V7探索已全部入现有池

R20 seed42/GPU0、R21 seed42/GPU4已实际初始化并完成step0四mask验证，共同参数初值SHA均44ff26ac8144ece78c84d9660e2d40d8f0746fa4e555a9ae2afeaadbd7d4b6a2，参数量75,601,176/75,674,904，torch2.8.0。其余R24/R26 seed42和四项seed43已依次入队，总8项。每项100000训练样本/1024验证样本，3125优化步，实际进度由progress/curve证明。完整job账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/jobs.json。real_data_pilot_started=true，formal_training_started=false。未新建平台任务或重启。

## 2026-10-02T19:21:41.759165+00:00 消融解释口径补充

当前四架构×两seed真实pilot已在八GPU运行，仅比较通用遮蔽重建。最终完整三新增因素矩阵与去除组件计划见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ARCHITECTURE_ABLATION_FIGURE_PLAN_20261002.md。既有位置/链身份/长度机制若主张贡献需要额外单列消融，不混同三因素矩阵；未评测理解/生成不能填入最终图。来源少样本条目须标分母。

## 2026-10-02T19:24:28.477070+00:00 实测作图读取已通过，等待共同250步验证

作图入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_architecture_pilot.py 已实现并用当前八项真实日志执行。数据/配置/源码/同seed初始化一致性检查通过；返回awaiting_shared_evaluation、common_evaluation_step=null、missing_curves=[]、figures=[]。当前只有step0曲线，正确输出CSV/JSON证据而不生成假效果图。作图证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/figures/v7_pilot.json。

当前优化步：p01_v7_r20_seed42=200, p01_v7_r21_seed42=175, p01_v7_r24_seed42=125, p01_v7_r26_seed42=125, p01_v7_r20_seed43=125, p01_v7_r21_seed43=125, p01_v7_r24_seed43=125, p01_v7_r26_seed43=125。首轮完整3125步仍运行中，未确定胜出架构。根任务在本轮完成方案、抽样、入口整合与实际八卡启动；后续监督负责共同250/1000/2000/3125里程碑及错误告警，不提交新任务或改动运行中源码。

## 2026-10-02T19:52:22.715860+00:00 grammar/context loss核验及context数据增强计划

冻结A4的101024行loss资格核验通过：train/valid context位置697220/5852，eligible与有效label均0；关系监督资格4648/37，实际标签全部binding正例。用户明确要求保留关系监督并增加context，计划原100000追加20000既有context监督记录、原valid不变；尚在实现/未抽样/未启动新组，不修改当前八项。完整epoch索引与第1000步断点接续控制通过，3125步恰好覆盖100000索引。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。

多图/多帧新增阅读与三份固定作者源码已归档；优先研究分段全长/global交替及既有延后融合，其次target/context cross-attention，保持context无序列CE与明确关系监督。新方法未实现/执行；研究取舍见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/CV_MULTIVIEW_VIDEO_ARCHITECTURE_20261002.md。

## 2026-10-02T19:56:12.027484+00:00 nonbinding缺失的配方原因已定位

parent manifest filter_names含recipe.binding_only，recipe.drop_nonbinding_sources显式列出asd_antibody/trait/tcr_native/tcr_papers。当前subset监督标签全部binding不是抽样偶然性的充分解释，而是与该配方一致；全parent统计将在追加扫描给出。用户已被告知，并收到可选目标澄清（后续可靠负例正负判别，或本轮仅binding正例）；等待该偏好不阻塞现有八卡epoch及20k已审核context正例增补。不会把未观测配对/unknown造为nonbinding。

## 2026-10-02T19:58:16.131168+00:00 用户确认本轮只学习binding正例

异步目标澄清已明确选择“本轮只学习已有binding正例，暂不做正负判别”。据此追加候选只取原V7明确binding=True监督的context记录，保留关系token监督，不引入negative、不生成负例、不请求新负例来源。缺nonbinding不再作为本轮阻断，也不把正例学习描述为正负区分能力。当前八卡样本/源码/3125步不变；独立追加20000正例context仍在实现中。

## 2026-10-02T20:26:02.091441+00:00 context正例新增20000条完成，加载与loss布局检查通过

用户选择本轮只学习已有binding正例、暂不做正负判别。独立新子集 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k 已生成：保留原100000训练行作为逐字节前缀，追加20000现有V7 parent train记录，总120000；valid1024逐字节不变。context从4648/100000（4.648%）增至24648/120000（20.54%）。新增均labels.relation=binding且metadata.relation_supervised=True，未生成标签或负例。原74个train分片的SHA/count全部核对，候选context正例297751条，排除base中4648行后可选293103条，分source比例抽样。

| 新增来源 | 行数 |
|---|---:|
| asd_antibody | 11335 |
| tcr_native | 825 |
| tcr_papers | 7685 |
| trait | 155 |

原冻结loader实际加载120000/1024并通过parent ready/两证书当前SHA绑定检查；新增20000行逐条A4布局通过，最长1299 tokens，无丢行/截断。新增context位置2972878个，label/eligible均0；binding关系监督资格20000个，target监督资格2292707个。独立复核新增原行SHA唯一、与base train/valid原行SHA无交集。这里排重指原行字节与来源身份，不声称做了新的序列相似度清理。eligible仅指可被遮蔽监督，不表示每步都纳入loss。

数据manifest SHA 1efa27ea19d10bc94143c3977c80dba8c3a29bad921c093ca7f24447635b9478；完整计数/来源证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_context_plus20k/sample_manifest.json；加载与mask检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_preflight.json。增补入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/augment_v7_context_subset.py；独立只读源码review无必须修正项，记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_code_review.json。没有产生checkpoint或修改正式ready/审核文件、恢复注释分支或独立CUDA算子工作。

当前原8项记录步数1400–1475/3125，继续原100000条完整epoch，原样本/冻结源码/作业不变。新120000条尚未用于训练；global32下下一独立组应完整3750步。旧p01 runner/dispatcher/plot固定3125，不能直接拿它训练120000并称完整epoch；后续必须独立配置3750的训练调度、终态与作图并分组报告。没有平台submit/cancel、容器/worker/监听重启或重复入队。

## 2026-10-02T20:36:33Z 独立 G/S/L 组合架构与 CPU 整数输入合同

新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research_refined/factorial.py，提供 `FactorialModel(config, variant)`，不修改旧模块、公共 registry 或任何冻结运行包。八个标识按 **fGSL** 位序明确对应 `f000/f001/f010/f011/f100/f101/f110/f111`：G 为动态 head gate，S 为固定 attention/FFN residual scale 0.1，L 为零起算偶数层同段半径 32、奇数层 global。复用原 `ResearchModel`、`ArchitectureBlock` 和 `RefinedModel.denoise`，保持原目标及词表。

另提供 `cv_segment_global`（偶数层同段全长，奇数层 global）、`cv_late_fusion`（原 R25）及 `fixed_gate_control`（原 R22）。同段层中 prefix 保持独立分组，不能向其他段转发信息；padding 不能作为 key，丢弃的 padding query 保留有效 key 避免空 softmax。新的 `load_common_parameters` 只允许 head gate 专有权重差异，完整复制相同结构/配置的公共参数；保存采用显式 `immune_architecture.factorial.v1` schema，校验位序、options、双配置及词表。该类须显式导入，尚未接入旧训练/加载分派入口。

首轮 pllm CPU 实际验收 **24/24 passed，12.99 秒**：六个旧对应项（四个原初筛条件及 R22/R25）的 logits/loss/所有梯度一致，新增组合双活跃 forward 的激活检查点梯度一致，段隔离/global/radius/padding 路径正确，全部 11 种配置保存恢复及 schema/options/vocabulary 拒绝检查通过。全部输入为人工整数张量，无记录读取、Trainer、GPU、真实数据或功能设计。测试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_factorial_architecture.py；原始日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/factorial_architecture_cpu_20261002/pytest.log；JUnit /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/factorial_architecture_cpu_20261002/pytest.xml。本轮只验 CPU FP32 工程合同，不证明 GPU/BF16 数值、学习效果或架构优劣；旧 autocast CPU API 产生 306 条弃用 warning，无失败。所有自有测试 checkpoint 随 TemporaryDirectory 退出删除，并在测试中确认目录消失。

源码 SHA256：factorial.py=`2ee8a8812262cc8da57dee29d1a7e9ce2309c5da350998f5449aeed1974532ac`；测试=`615dfc3bf9b4257824c218bc1eb8624a36038deb42cd0d32b176ea229e5afb61`。原作业/监听未操作，停止注释与独立 CUDA 算子暂停要求保持。

## 2026-10-02T20:44:10.457859+00:00 新120k组合架构与context指标入口完成CPU验收

上一Goal回合为progress：独立追加20k context正例并完成120k数据加载及新增A4布局验收。本回合继续原Goal，八p01实时worker心跳/优化步在推进，不重启或重复提交。新增Factorial 8组合+3对照，CPU24项通过；新3750入口2步人工检查和关系指标2项通过，自有CPU checkpoint已清理。11份方法YAML/22个两seed初筛格已登记，但尚未冻结/发布/入队，第三seed确认仍待完成。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_FACTORIAL_ABLATION_20261002.md。

正式world8缺口已只读定位：现pool每worker只见一GPU，p01是8独立模型；旧V4仅synthetic/R21/短步数，不能换data_dir直接作正式训练。新driver正复用原Trainer及八rank通用checkpoint接口，仍待实际GPU联合更新/保存恢复验收，不借旧失败或CPU记录宣称成功。当前final V7数据审核有效；所有原停线与源/数据门禁保持。

## 2026-10-02T21:01:39.089377+00:00 22项120k组合/CV实验已一次入原池

p02冻结566份源，复用p01完整源再显式覆写4个已验新/共享文件；冻结清单SHA 0a88b6a1b38b85fdaa9491444286da6b1bd023c721ce4828d7f5ca0b0b0e53ab。22个launch参数检查通过；只更新公共入口指向新分派并追加22个launch-only名，原78个名及p01分派保持。发布前后字节证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/deployment_v1/publication.json。

已实际登记22个唯一job ID到原pool（pending22、原p01 running8）：账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/jobs.json。先排baseline/原局部global/整段global/延后融合各两seed，再排其余GSL组合与固定门控对照；每项120000条、3750步。只是已入队，尚无p02 GPU初始化/优化步/完成证据，不宣称新方法有效。原卡自然完成p01后领取，不重启或取消在跑任务，不新平台submit。真实平台查询返回同一task Running；按原worker 258600秒预算估计入队时余22.41小时，原始CLI输出带版本提示行，已提取真正JSON记录而未把提示当任务故障。

正式八rank driver已落盘：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py；分离冻结模型源与真实审核资源，显式支持11组合/CV模型及原4候选、global=micro×GA×8、rank0写共享状态、八rank实际参数SHA与native optimizer/scheduler/RNG完整恢复。新协议CPU9项通过（子agent执行报告；原始日志归档待根端读取），仍未联合GPU运行或接入pool组启动；不将此写成正式训练已启动。接口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md。

## 2026-10-02T21:07:58.922239+00:00 正式入口隔离审核首次真实执行通过，p02作图已登记

冻结正式模型/Trainer与真实审核资源的分离接口已首次执行：原helper在python -I -B、CUDA隐藏的子进程调用原assert_ready，真实校验final V7所有分片及现行scope/evidence，54.28秒返回passed；parent清单与两证书SHA前后不变，实际门禁模块39份。没有创建审核证书或改ready、恢复注释或启动模型。源清单1137项、SHA 2c5a2a8a5ba9841dbffc569f1318a09d2497eed39a9ccc7ba5ed8e99104172b1；事实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.stdout.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.json。此处是新隔离执行接口的首次实际核验，非重复签证。

root已读取原工具stdout归档，正式driver新增协议测试9/9、3AST通过；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/receipt.json。正式入口源已冻结，但launcher还没接通，联合NCCL更新、实参同步、保存恢复实际GPU证据仍缺，formal_training_started=false。正式长程预算需结合原资源剩余时间设置可恢复阶段；不允许因全部小实验排完便把8份独立模型视为world8联合训练成功。

p02作图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 新4项纯数学/结构控制通过；首次读真实22项账本输出JSON及4CSV，awaiting_shared_evaluation、missing22、figures=[]。没有用假效果值画图。数据列包含source/context分母、关系CE/accuracy及正交G/S/L对比，完整图页的真实渲染待共同评估出现；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_ABLATION_FIGURES_20261002.md。原p01监督完成后续看p02，至多12小时或22项全终态，无新增OS监听/平台任务。


## 2026-10-02T21:23:10+00:00 正例目标一致性核对与正式入口尾批问题

用户再次确认“本轮只学习已有 binding 正例，暂不做正负判别”。此选择与已冻结 p02 入口、关系指标及 11 份 YAML 一致，无需修改在跑实验或重新入队。Context 序列不计重建 CE，只有已有明确监督的关系 token 在被选中时计入原 CE；unknown/未观测配对不变成负例。已有新增 20000 行布局证据显示 context 的 label/eligible 均零，关系 eligible 为 20000，本次复用原证据而未重复扫描或测试。只读一致性记录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/positive_only_reconciliation_20261002T2122.json。单类关系准确率可能由恒预测正例获得，不能证明使用了 context 或具有正负判别能力；非 binding 缺失不是本轮阻断。

21:23:10 监督读取原 p01 八项 2575–2650/3125，共同 2500 验证、80000 行曝光；未出现失败或失联。p02 共 22 项已入原池、尚待空卡，不能把已排队说成已有优化步。原 resource/worker/源码/数据继续运行，无平台提交、取消或重启。

正式入口只读复核发现真实问题：原 HF Trainer 的 dataloader_drop_last=True 也传入验证 loader，Accelerate 分片时可能漏掉小来源或不完整尾批；num_samples==len(dataset) 不能验证实际覆盖。正在公共 driver 中仅修正验证 loader，随后做人工整数 ID 的覆盖检查；冻结 v1 和 p01/p02 不改，不以旧测试代替新修正的验收。正式八 rank 联合更新、保存恢复尚未执行。

正式数据配方仍待最后对齐：p02 的 context 比例为 20.54%；当前正式 driver 用完整父训练集自然抽样，既有计数 297751/6410749≈4.645%。两者不是相同采样配方，不能宣称正式入口已经继承 p02 的 context 比例。此处只记录差异；没有调整权重、重采样、改审核或启动正式训练。


## 2026-10-02T21:35:07.714476+00:00 验证尾批修复、新源码冻结与当前接续

根端已读原始证据：正式入口的验证 loader 现在单独使用 drop_last=False，并在 finally 恢复训练的 True。worker 环境（torch2.8.0/transformers4.48.1/accelerate1.14.0）新增 2/2 人工整数检查通过：真实 HF loader、8 个逻辑 BatchSamplerShard/DataLoaderShard 及原 gather_for_metrics 完整保留 13/17 行；训练 33 行实际保留 32，异常后参数恢复。没有 forward、训练、真实数据或分布式通信；临时输出自动删除。结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eval_tail_20261002/worker/result.json 与同级 worker.stdout.log。旧 9 项未重复执行，不把它们算作新版全验收。

独立 transport 的 8/8 新 CPU metadata 检查通过，根端已读原日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_transport_evidence_20261002/self_test.stdout.log。复用已冻结通用协调器、held FD、进程监督与清理，旧 synthetic 标志不变；测试禁用 socket。尚未实现实际 pool 的八 worker 一次预留/发布接线，未创建 intent/reservation/group/listener 或新 GPU/平台任务。不能把 CPU 协议通过称作八 GPU 启动通过。

新版冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v2，driver SHA a751eea2893aea463ab23798b0f685edbf1333a3da8e7a287aa4a7a4f453c94d，原 v1 字节保留。清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v2.json，SHA 566bec35f5ee0ea65047a3e69d864e729cff2de2ad1a861929401551cc368799，共 610 文件：570 冻结源码、39 份已实际执行审核源码和固定 metadata contract。没有继续绑定与审核无关的可变研究文件；实际审核执行源码仍逐项拒绝遗漏/漂移，数据门禁未修改。

因为修复使源清单改变，原 helper 对新版绑定真实执行一次，58.79 秒 passed，stderr 空；原 parent manifest 与两证书 SHA 保持相同，未签发或改写证书。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v2.stdout.json，执行记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v2.json。不可把旧 passed 直接换上新 SHA，本次是有实际源码修正依据的绑定核验。

21:34:15 原 p01 八项 2800–2875/3125，共同 2750 验证，未观察到失败/失联；p02 已入队 22 项仍待空卡，目标保留用户选择的已有 binding 正例/context 序列零重建 loss。正式模型与配方最终选择、八 worker 统一启动接线、实际联合更新/完整 checkpoint 保存恢复仍待完成。正式训练未启动，原 Goal 保持 active；不重启容器/worker/监听，不重复提交，不恢复注释停线或独立 CUDA 算子。


## 2026-10-02T22:15:25.209702+00:00 更大物理 batch 的独立受控实验方案

实测p02已观察峰值allocated约3.15–3.86GiB/A10080GB，支持优先评估micro16/32，但尚无这两档实际GPU证据。保持global32仅保证每步记录数：现实现loss为8个micro各自selected-token mean后等权平均；mask seed又含GA且随机张量依赖局部padding长度。因此4×8与16×2/32×1不是同一loss加权/mask流，不能直接改14个候选却用旧4×8 baseline宣称架构收益。

决策：已运行8项不重启，未启动14项已按上述用户变更保留原件撤下；新增独立大batch组仍覆盖11条件×2seed，全部同档、同有效batch32、同120k顺序/3750完整epoch。验证micro固定4，binding已有正例/context序列重建0不变。容量probe采用人工整数输入、实际模型配置及最长2048形状，候选4/16/32与有限optimizer更新，记录显存/吞吐/OOM；不是独立CUDA算子专项，也不恢复注释。档位待实测选择，不能伪造profile通过或修改旧resume provenance。新代码/YAML准备中，未发布/入队。三seed确认及正式八卡单模型训练仍未完成。

## 2026-10-02T22:39:58Z 大 batch 与八卡整合方案的实际实现状态

前述大 batch 独立入口及 11 份 YAML 已完成，原 p02 core 不改；四项参数控制及一次 micro16×GA2、2 步人工 CPU 入口运行通过。micro32×GA1 只经过参数约束检查，未运行该档模型。新组源码已独立冻结 568 文件，SHA 清单及绑定 CPU 证据已实际核对；尚未进行 GPU 容量测试、选择共同档位或启动新组。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_ABLATION_20261002.md 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/cpu_entry_evidence.json。

首轮 p01 已完整完成，但原 p02 的八项仍在运行，十四项已在启动前撤下；完整三因素矩阵、第三 seed 确认及最终架构选择尚未完成。旧小 batch 与新大 batch 的 loss 权重和 mask 流不同，继续分组比较。当前已有关系正例和 context 序列不计重建 loss 的方案不变，尚不能由这些重建结果推出下游理解或生成能力结论。

八卡共同训练的 pool wrapper 与调用入口源码已实现，分别新增 8/8、4/4 人工 CPU 检查通过。入口拒绝路径 symlink 逃逸和八票据跨组计划引用；票据后缀不作为 GPU rank。检查只证明 metadata/调用协议，exec 为 mock，checkpoint 为人工 fixture；公共接线未部署，实际八卡联合更新与保存恢复仍待验收。详细证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md。原 driver v2、已有数据证书和停线分支保持；本次核实没有重跑旧验收或更改架构。

## 2026-10-02T22:52:04Z 容量测量接入与独立消融图复用决策

更大 batch 的模型容量测量调用入口四组新标准库控制通过，已完成一次实际接入与登记，唯一 job 为 1790981433321562451_p04_v7_capacity_febb28e3，等待原卡自然空出。只读取人工整数输入、固定全长 2048、三代表模型 × 三档 × 八次更新，总预算 72；GPU 测量尚未执行，无 batch 推荐。接入保留原 100 个名及旧冻结模型源，只增加一个 capacity 名；登记/接入原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v1/publication.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json。新 22 项训练不在本次登记中。

独立只读代码审查确认新入口的 curve/progress/COMPLETE 布局可复用既有作图聚合、分母核对、同 seed 差值与四页 SVG/PDF/PNG。当前决定实现独立 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_large_batch_ablation.py，要求显式新 study、同组统一 micro16×GA2 或32×GA1、eval4、batch-plan 本体与实际 profile 引用 SHA、独立 job 绑定；禁止以旧 p02 填新组缺格。完整 22 项在 3750 步且 COMPLETE/summary/pool success 对齐方为 final；部分共同评估标 partial，未出现正步共同评估只导出缺格表。旧绘图默认行为保留，最多扩展一行可选标题。源码和新人工 metadata 验收正在实现，当前尚未声称通过；不复跑模型训练、容量检查或旧验收。

## 2026-10-02T23:01:46Z 当前优先候选与已实现绘图协议

已有真实记录支持优先保留R24固定残差缩放和R26交替局部/全局注意力：完整100k首轮 token AA CE两seed均值分别0.817343/0.820938，较R20的0.837899下降2.45%/2.02%；R26来源宏平均下降7.05%。新增context第二轮1000步的f001对应R26，均值0.965856较同轮f000低5.22%。R21动态头门控首轮CE较基线高1.20%，暂不作为默认。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md 与其中原始证据。

新统一batch组前八格计划优先基线f000、固定缩放f010、局部/全局f001及组合f011各两seed，再完整覆盖全部11条件。此为执行优先级决定，尚未登记新训练或确定组合胜出；保持其余门控/CV对照、完整epoch与后续第三seed确认。token重建与稀疏来源宏平均仅用于阶段筛选，最终理解/生成下游优势未证实。

独立新绘图协议已实现并通过5项人工metadata/Matplotlib检查；复用旧聚合/分母/四页图，旧源仅一行默认兼容标题扩展。只出现部分正步评估时标partial，完整22项共同3750且pool成功/完整SHA对齐方final，旧p02禁止混入新组，计划/profile SHA仅证明引用字节而非GPU通过。图源已独立冻结两文件，原模型568源清单不变；详细原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_plot_evidence_20261002/receipt.json。

## 2026-10-05T13:40:03.875317+00:00 初步正式训练实际运行与监督

初步f011已经完成真实world8前16步及8→16原生保存恢复验收；原8卡正在接续累计5016。后续保持原25042固定调度、自然epoch配方及每1000评估/保存，采用全部8rank成功后有界接下一段的监督计划。新通用监督器不更改模型、训练目标或数据准入；实际进展见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_INITIAL_TRAINING_20261005.md。完整22消融仍未完成，不能将初步版本称为最终矩阵赢家。

## 2026-10-05T13:52:43.383249+00:00 补齐消融的优先级

原22格8完整、14待补（含正在跑的f000 seed42）；f000 seed43已完整不重跑。先完成f000 seed42，再f010 seed42/43，以形成R20/R24/R26/组合完整两seed对照；其余11个缺格按原范围接续，不新增架构变量。全部fresh、相同120k一epoch与32×1/eval4，metrics-only。逐格证据与13项未执行命令：/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_COMPLETION_MATRIX_20261005.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/remaining_ablation_plan.json。当前尚不能产出完整22格最终效果图。

## 2026-10-05T14:06:47.605732+00:00 本地补消融的速度解释范围

本地补消融与原池保持质量评价协议；但本次NVML实际显示两处显存分配且PID归属未解析，计算竞争尚未证实，故不能把本地/原池wall-time差异直接归因于架构。最终速度图需标出环境差异；不为此重复已完成质量实验或干预未确认归属的进程。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_gpu_environment_observation.json。

## 2026-10-06T10:54:21.144337+00:00 速度归因答复：具体低效路径已找到，主因占比尚未实测

普通训练区间约10.20/10.70秒每更新，V5历史整体约2.73秒；协议和数据不同，约4倍只描述现象。保存不能解释全部慢化，2048是上限而非每批固定padding。优先检查attention：memory-efficient被关闭、可训练dense bias限制其他融合后端，按PyTorch2.8源码推断当前训练使用math路径；尚无实际kernel trace。局部层仍稠密计算，逐层bias转换/扫描、重计算以及GPU标量转Python的同步均是已发现的额外工作，耗时占比未知。

原1024条本地CPU热缓存样本micro32取数+collate均值38.75ms，不能据此完全排除生产共享存储等待。既有8份standby日志在训练开始时停止，代码先停并等待standby；实际远端进程清单缺失，不能称已逐进程排除，目前没有证据将占卡程序视为主因。长度混批与容量增加计算，几何比例不等于时间比例。

下一证据是整模型GPU trace和取数/训练/通信/保存计时，再对最高耗时项逐项验证。已交付2×4 FSDP入口包含计时、CPU工程验收通过，GPU尚未运行；本次输出目录仍不存在。FSDP不是已验证的提速修复，跨节点通信和精度路径也影响比较。当前无架构更换、冻结源码修改、任务提交或重启；独立CUDA专项保持暂停。

依据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_windows_and_geometry.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/data_loader_review/result.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/root_readonly_followup_v1.json。官方后端约束：https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/cuda/sdp_utils.cpp 与 https://raw.githubusercontent.com/pytorch/pytorch/v2.8.0/aten/src/ATen/native/transformers/sdp_utils_cpp.h 。本条整理现有证据，不是新增模型实验。


## 2026-10-06T14:35:22.374740+00:00 后续消融的调度接续，不改架构/配方

保持原22格120k/3750计划，执行顺序为正式累计两轮50082→固定末点9项生成评测→重新核实缺失消融。v2私有适配已完成13项调度边界检查，未发布或运行新模型，不能当作新架构结果。方法和训练目标没有改动；详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。


## 2026-10-06T14:53:38.864998+00:00 固定残差单因素的2000步判读

f010/42与f000/42在相同64000条曝光和固定计数下，平均AA CE改善，但全遮蔽AA CE/准确率及平均/全遮蔽END准确率回退。这是局部重建收益，尚不足以支持理解/生成全面收益，也不据此中断既定f011组合正式实验。继续完整3750和第二seed，以及正式两轮后的独立生成评测。具体数字和原件见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LOCAL_ABLATION_PREPARATION_20261005.md 的本时间节点；本轮未改模型。
