# Data Format Audit

## 2026-10-07T06:11:39.860103+00:00 V5固定对象容量、自主生成分隔与终点

已完成用户要求：当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 中full_capacity=512，表示整个受体对象的容量（含`.`与`<protd>`），不再要求真实生成长度。固定目标头部；真实残基／分隔／终点监督，synthetic与尾部mask不计loss，512位均参与attention。CDR-only仍保持先前全长scaffold及mask补位，VHH继续不在本版训练范围。目标role/scope不按真实边界编码；denoise只按腐蚀后当前可见`.`分链和重置位置，保留可见链内／全局交替以及残差1.0。

训练入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py 使用 --input-layout V5 --variant cv_segment_global --object-capacity 512；统一生成及CDR入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，模型generate也接入同一解析器。支持整体生成与已知前链→生成后链；反向补链需另定条件协议，明确拒绝隐藏参考边界。无终点／非法链结构报告失败。此记录替代此前“真实长度／generation_lengths”的当前版本要求；历史A4及冻结训练不改。

83项测试通过，7来源1250条抽查通过，130513个synthetic补位保留；不是全量数据扫描。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_validation.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_regression.log，说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md。临时保存恢复测试权重已清理。未提交新训练，无本版下游成绩。

## 2026-10-07T05:55:11.432519+00:00 CDR-only TCR 保留全长并以 mask 补位

当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 partial_sequence_policy：只有 CDRβ／CDRα 的记录沿用先前 prepared 全长 scaffold、区域及长度，送入模型时按 synthetic_residue_mask 将缺失位置转为 <mask>；保留真实CDR及锚点，真实观测X不替换。全缺失配对链沿用已有scaffold全mask。补位参与注意力但labels=-100，不计重建损失，不重新采样长度或缩短为CDR-only。实现 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py，仅改变新V5布局；历史A4不变，VHH仍不在本版训练范围。43项测试通过；1250条真实数据抽查中620条含synthetic，共130513个补位检查通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/scaffold_mask_validation.json 与 scaffold_mask_regression.log。本次未提交训练、未生成checkpoint。

## 2026-10-07 当前新训练版本排除 VHH

用户明确：本版不需要考虑 VHH。版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 training_scope：常规抗体 H/L、TCR 在范围内；nanobody_vhh 链和 nanobody／nanobody_antigen 任务不纳入本版训练、专门评测或验收。版本说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md 已移除当前训练格式表中的 VHH 行。通用 renderer／兼容性测试保留，不代表本版训练覆盖；本次只登记范围，不修改现有语义数据或运行时过滤逻辑，后续本版训练清单构建须落实排除。

## 2026-10-07 V5对象布局的新版本

按用户要求，新版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 采用 `ablation=V5`、`cv_segment_global`。context在前，关系居中，受体对象内用`.`连接H/L或beta/alpha，无条件使用固定null前缀；恢复变长scaffold及V5结构token训练，synthetic不计损失。保留紧凑词表、无ESMC、残差1.0、整链/全局交替。全部7来源复用最终语义JSONL，通过collator转换；不改历史A4配置。训练入口显式 `--input-layout V5`，下游新适配 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，生成必须显式声明长度。配置与格式说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md；验证 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/validation.json。尚未提交新训练，无新checkpoint或下游成绩。

## 2026-10-07T05:35:18.340Z V7两轮删除按下游任务只读归因

重算初版479684和后续441532条删除台账。后续联合键exact10508（含αβ8433）、仅组分exact148250、仅近似282774；ASD34768是抗原包含TCR表位，不是已证实抗体答案重复。17个cohort逐任务/分split/原因计数见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FILTER_TASK_ATTRIBUTION_20261007.md 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_filter_attribution_20261007/summary.json。初版为混合保护库同簇过滤，cdr3a400683不等于完整配对exact，台账缺少具体任务归属，不能伪造任务精确归因。原台账哈希已保存；没有原始序列扫描、重注释、恢复行、改证书或训练。

## 2026-10-06T14:15:40.813569+00:00 原pool生成任务证据

新增每任务EVALUATION_START.json、EVALUATION_EXIT.json与COMPLETE.json，分别记录实际单卡分配/父进程/命令、真实评测子进程退出码、固定checkpoint及结果引用；全部显式work_type=generation_evaluation、optimizer_updates=0。队列兼容字段train/steps1不是训练计数。summary新增coverage=assigned_mode和assigned_mode；整批完成由9项、9741条、逐行重评分与pool/子进程双重退出验收证明。实际产物仍未产生。协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4；执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_pool_v1/package_v1.json。

## 2026-10-06T13:54:38.042567+00:00 V7原生CDR观测产物

现有CDRResultWriter的manifest/predictions.jsonl/metrics.json schema不变。V7输出根增加model_load.json（checkpoint/step/native_epoch、模型来源与实际导入文件SHA），每模式增加input_audit.jsonl（fold/row/source SHA、CDR位置、歧义标记、实际前向特征指纹、隐藏参考扰动相等及精度）。status.json与summary.json显式running/failed/partial/complete；预测完整性继续由既有评分器重算。固定源清单与命令 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_plan_v2.json；尚未产出正式评测数据。语义与范围唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4。

## 2026-10-06T10:01:03.673427+00:00 更正“统一padding到1024/2048”的说法

先前把最大允许长度与实际padding长度混称，造成误导。V5实际运行上限1024，其记录commit的collator取本batch最长记录再补PAD；当前上限2048，实际仍取本batch最长记录。2048不是所有输入的张量长度，也不能据此解释训练慢。当前长记录需要较高上限，但尚无证据证明2048这个具体数值是唯一或最优选择。

复用既有1024条有限CPU诊断样本，不读取原始训练序列：4条渲染后记录超过1024，长度1090/1109/1148/1206，涉及32批中的3批；17批的实际长度为516。渲染长度包含context、多链、grammar和目标容量，不能说成单链有1206残基；该样本不代表全量比例。

只比较其余29个可放入1024的原批次，当前动态padding共568224个张量位置；强制统一1024会变成950272个，即1.6724倍，稠密attention矩阵元素数量为2.6607倍。这仅为几何量计算，不能当实测时间或速度倍数。详见 [padding1024_existing_geometry.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/padding1024_existing_geometry.json)。

1024可作为后续短输入上限候选，但需明确长记录处理；优先评估动态padding与按长度分组，而不是把所有短批次也扩到1024。长记录可保留独立批次及较高上限；若改变目标容量/布局，需要另行验证，不能直接截断或丢弃记录。当前冻结运行、数据与证书均未改变。本轮仅更正文档与整理已有证据，无新模型测试；不恢复任何已停止分支。

## 2026-10-06T09:41:42.093903+00:00 当前A4长度行为澄清（数据未改）

正式V7的2048是整条grammar记录上限，collator按本batch最长记录做外层padding；target完整/片段容量256/64仍在，未使用余槽参与attention但label=-100。context残基不计算CE，保留单个目标终止<protd>及已知binding正例关系监督。现有小样本长度与源码依据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md。本轮没有数据重构、证书改动、原始序列读取或恢复已停止注释分支。


## 2026-10-02T18:53:05+00:00 当前最终测试范围数据已验收

使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。按用户确认范围最终审核通过、training_ready=true，真实 assert_ready 已通过；保留 6,460,277 条。规则、明确排除项、证书、低样本限制及哈希以[最终记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准。历史版本 pending 不代表此新版本状态。未启动训练。

## 2026-10-02T18:40:43+00:00 开始最终范围收尾

回填既有 alpha CDR3、复核实际 T4 reference、对齐 test-only 门禁。新版本及限定范围见[最终收尾记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)；当前构建运行中，未宣称审核完成。

## 2026-10-02T18:19:18+00:00 alpha CDR3 缺字段追溯完成

8,765 条均来自 VDJdb 衍生 fulllength；上游已有唯一 cdr3a，当前 prepared 转换未保留该标识。可追溯证据、代码原因与待回填状态见[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。未进行新注释或修改数据。

## 2026-10-02T17:40:43+00:00 r2 观测链修复与复审完成

当前新构建版本为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2；配对误判缺陷已修复并独立验收，整体审核仍未通过。数量、版本哈希、规则与剩余缺口以[r2 完成记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准；未修改训练入口或 ready。

## 2026-10-02T17:24:16+00:00 真实观测链修复 r2 开始重构

按 synthetic_residue_mask 区分真实伙伴与全合成占位链；新规则、版本路径和进度以[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准。13 项边界测试通过，当前运行中；未解除整体审核门禁。

## 2026-10-02T17:05:42+00:00 新版审核暂不通过：真实观测与合成占位链需区分

上一轮导出按角色字段判定配对不充分，导致部分单链记录绕过非配对过滤。当前版本不得作为审核通过数据；真实观测链的判定、全量影响数与其余缺口以[审核报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_AUDIT_20261002.md)为准。未更改数据/门禁/ready。

## 2026-10-02T16:46:51+00:00 测试集配对完全重复版本已完成重构

采用实际测试范围与同一测试记录 CDR3α＋CDR3β 完全匹配规则；非配对记录保留测试范围内旧规则。新增原行号删除台账，原划分及保留行字节不变。导出和独立选择/字节复核已通过，仍未认证；版本布局、实测数量与默认口径依据见[本轮记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_REBUILD_20261002.md)。

## 2026-10-01 06:46 UTC 三候选运行包实际构建与CPU隔离通过

本轮仅人工整数fixture的软件实现/CPU导入及来源包构建，没有读取或修改真实序列/shard。[本次只读准入metadata](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/data_metadata_after_build.json)：dataset manifest SHA 601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce，training_ready=False，audit/downstream_scope两证书存在性False/False。三候选source-only包及CPU结果不构成数据准入；[三候选验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINER_GPU_PLAN_20261001.md)。

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

## 2026-10-01 04:24 UTC v2失败阶段记录边界

v2运行仅6份失败原件；producer_stage_evidence保留1个oracle窗口/8 microsteps/256人工记录和通过的GA统计，但gpu_accepted=false/raw_completed=false。失败先于正式三窗口，不能把oracle1记为formal3。没有raw/summary/PRODUCER_SUCCESS/GPU_RELEASE_COMPLETE/COMPLETE或checkpoint；v2没有记录清理后allocated数值，后续观测须显式补充。训练数据及selection、原证书门禁不变。 [整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

## 2026-10-01 04:07 UTC v2运行与阶段记录

v2 manifest3805679f…已实际创建，526文件/2别名布局不变，包acceptance/training标志仍false。新D01已生成request、producer_stage_evidence和progress，当前starting/gradient_oracle/1200秒；partial明确gpu_accepted=false/raw_completed=false，不能计完整三窗口或通过。新job为1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a，v1失败另保留。[合同与证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。

自有CPU临时验证目录已清理，小指标/源码/日志保留；人工GPU保存关闭。prepared/split/selection和双证书未改，03:41元数据仍training_ready=false、证书缺失，正式训练未启动。

## 2026-10-01 03:26 UTC v1失败产物与v2计划

实际v1仅有request.json、producer.log、FAILED.json、process_evidence.json和worker result；没有candidate/raw/summary/GPU通过标记，checkpoint=0。parent716.1639秒退出/回收证据不等于完整三窗口完成。v2将新增producer_stage_evidence.json保存小型阶段统计与budget_failed状态，不能作为completed raw或通过证书。原prepared/split/selection/双证书均未改；正式训练仍待原数据准入。[实际证据与新版本计划](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

## 2026-10-01 03:18 UTC 实际 runtime 与人工验收产物

固定runtime的526源码文件和两处资源别名已创建；manifest SHA baeac98b962650770cf17c2f83014654c2520622e2785ead17db680a1228421a。不可变包status仍source_package_only_not_gpu_or_data_acceptance，三个acceptance/training标志为false；实际验收状态从独立GPU完成记录读取，不修改包manifest来签证。[实际整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

D01运行目录已生成request.json、producer.log和无checkpoint的hf_artificial目录，尚无raw/summary/父进程完成记录。实际合同为3完整窗口/24micro/768呈现与额外1窗口oracle，四mask计数及首次END均须独立核对；steps=1仅为池运输参数，biological_quality_claim=false。正式T20将分别绑定桥接源码、固定数据/selection和同版本实际GPU证据，未生成pointer或训练输出。

本轮只核对真实数据元数据与文件存在性，prepared/split/selection不变。候选manifest仍601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce、training_ready=false、双证书缺失；资源别名与人工检查不等于覆盖认证。

## 2026-10-01 人工 Trainer 入口与结果合同

新入口的D01运输参数profile-synthetic/steps=1不是实际优化步数。正式人工检查计划为三个global256完整窗口（micro32×GA8），summary以optimizer_window_count、completed_optimizer_window_count和逐窗口计数展示；人工训练与真实语料训练分字段，biological_quality_claim=false、corpus_training_started=false。[入口与CPU证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_RUNTIME_ENTRY_V1.md)。

producer拟输出gpu_release_raw.json、gpu_release_summary.json、PRODUCER_SUCCESS.json作为候选；parent实际观察正常退出与后代清理，另写process_evidence.json并强核验，最后才写GPU_RELEASE_COMPLETE及池COMPLETE。producer尚在私有副本适配，当前没有真实GPU产物或正式runtime，不能将预定schema当作已通过结果。

本轮未改真实prepared、split、固定selection或数据证书，也未改变A4链内位置、role/scope及full256/fragment64布局。两环境入口临时测试58文件已按专属ownership删除，checkpoint=0，清单在 [cleanup](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_runtime_entry_20261001/cleanup.json)。

## 2026-10-01 临时文件清理记录

已删除旧 ASD 分析临时目录和历史测试临时子树，具体清单统一见 [清理执行记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/PROJECT_CLEANUP_AUDIT_20261001.md)。`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_cleanup_audit_20261001/deletion_manifest.json` schema v1 记录原路径、分组、SHA256、字节数、时间、权限/inode 等删除前元数据；ASD 文本额外记录行数与 FASTA 记录数，不保存原序列内容或推断实验结论。清单与事件记录仅用于追溯，不能用于恢复原文件。本轮未修改训练/评测数据、prepared 父目录链接、样本布局及模型快照。

## 2026-09-30T23:31:26+00:00 人工框架原型的记录计数元数据

新增隔离CPU测试中的 `_record_window=(window_index,micro_index,actual_records)` 只随人工模型原样传给loss callback，未加入生产prepared、模型特征或collator。原生num_items_in_batch仍统计非-100的标签token；实测7条记录的短epoch窗口为[2,2,2,1]，record分母7、token数25，两者不能互换。两环境各14项真实Trainer检查通过，所有呈现ID无丢弃/重复。[合同及逐窗口证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_LOSS_FRAMEWORK.md)。

本轮未新增真实数据布局或证书，候选manifest及固定selection保持原样。计算/文献/进程验收不改变覆盖pending状态，也未恢复此前停止的序列注释分支。

## 2026-09-30T23:06:15+00:00 评估v2的计数合同与来源曝光

新原型不修改prepared内容、split、链编码或监督布局。修正的是统计接口：现有v3/layout191可包含监督relation前缀，除AA/end外单列binding/nonbinding CE、数量与准确率，并严格限制其位于目标槽位外且chain_id<0；三组之和覆盖全部被选监督token。新增END误报计数与每链首次END/missing/长度误差；首次END搜索整个有效target_slot_mask，包括容量尾部。仅full-mask评价终点，缺席分母返回null，labels只用于评分。[完整格式定义](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)。

[固定selection](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_data_readiness_20260930/expanded_candidate_validation_selection.json)仍为1288条，native/TRAIT各4条/4family。候选manifest仍 `601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`，未生成证书；原门禁实际拒绝后没有GPU/model或输出目录副作用。90项双环境CPU验收只验证软件合同。

[source_exposure报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_source_exposure_20260930/exposure.json)仅使用各来源行数及整数索引，natural/equal-source的1000/5000步计数独立核验。行索引唯一不代表生物实体独立；温度配额未实现采样器，未对原数据、采样顺序或真实训练做改动。此前停止的序列注释分支未恢复，覆盖认证状态保持pending。

## 2026-09-30T22:15:16+00:00 Expanded独立候选与固定验证清单已物化

新目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930` 共4,781,843条（train4,735,948/valid45,895），81分片、9,136,609,607 bytes。manifest SHA=`601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`。以锁定行号并集筛除2,119,972条，其中343,422条含exact理由、1,776,550条仅near理由；这些不能统称精确重复。全部6,901,815条原索引映射与命中理由逐条复核，原数据与split未改。[完整实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)。

138绑定文件、所有候选分片SHA/行数已验收；既有family metadata下train/valid交集0。175个整family迁移示例仅分析、未执行。seed42每来源最多256的selection共1288条，TRAIT/native各4条/4family；绑定整个valid内容及每条选中行定位/hash，不更改拆分凑数。[选择器38项CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_VALIDATION_SELECTION.md)。

候选仍training_ready=false/audit_pending，双证书未生成，实际assert_ready拒绝。更深入序列注释子任务被自动安全检查中断后已停止，未决覆盖继续保留；本轮只完成行号过滤、字节与索引完整性，不代表新范围无污染或已可训练。staging/SQLite临时文件已清理，必要映射及小型审计证据保留。

## 2026-09-30 历史作业配置目录布局

作业 YAML 的归档布局为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/train_jobs/<file>.yml` 与 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/eval_jobs/<file>.yml`。清单 schema v1 的 `files` 同时记录 `original_path`、`current_path`、`action`、`sha256`、`bytes`、原修改时间/权限、跟踪状态、引用与保留理由；TSV 只列已归档文件，便于按原文件名检索。具体数量与筛选依据唯一见 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md)。YAML 字节未改，本次不改变任何训练/评测数据格式或样本布局。

<a id="cdr-artifacts"></a>

## 2026-09-30 CDR 可复算产物与独立报告

新 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/cdr_result_io.py` 产物契约：`manifest.json` 含 schema、identity/hash、checkpoint 权重与配置 SHA、词表、源码 SHA、原始 fold 文件 SHA/行数和运行参数；`predictions.jsonl` 每行保存由 source SHA/fold/row/mode 构成的稳定 ID、identity SHA、参考/预测 token ID 与文本、掩码位置、正确计数、长度、AAR；`metrics.json` 在完整性校验后写出。逐序列 float32 AAR、fold 宏平均和 fold 总体标准差（ddof=0）沿用原定义；pooled sequence mean 与 fold macro 分列。partial 与 complete 分开，已有目录拒绝覆盖。

历史日志继续标 aggregate_only，不生成虚构行。审计配置、输入指纹、endpoint 状态及未知证据理由保存在独立版本化报告，见 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity) 和 [独立完整性报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_integrity_20260930/final/results/integrity_report.json)。原数据、旧指标、正在运行的 AB–抗原评测所锁源码未重写。

<a id="v7-construction-overview-20260930"></a>
## 2026-09-30 V7整体构建思路与当前落地边界

本节汇总实际代码、manifest和审计产物，不引入新清理规则。当前版本链为：`immune_v6_binding_only`（7,381,499条）→冻结旧下游保护范围、提取观测序列比较键、重叠过滤、语义去重、家族划分与共同布局检查→`immune_v7_clean_r2`（6,901,815条）→MHC序列优先更新→`immune_v7_mhc_sequence_v1`（仍6,901,815条）。前一步减少479,684条；MHC更新不增删行、不重切split。

### 七类来源和用途

| 来源 | 内容 | 当前train | 当前valid |
|---|---|---:|---:|
| oas | 配对抗体VH/VL序列 | 2,472,502 | 24,857 |
| ots | 配对TCR α/β序列 | 1,701,193 | 3,854 |
| asd_antibody | 抗原与抗体配对 | 203,189 | 960 |
| trait | TCR与肽，部分带MHC | 10,593 | 46 |
| tcr_native | 多来源TCR–肽数据，序列范围混合，部分带MHC | 22,508 | 106 |
| tcr_papers | 论文来源TCR–肽数据，序列范围混合，部分带MHC | 437,770 | 2,978 |
| tcr_repertoire | TCR repertoire，常为单链CDR3、无抗原 | 2,001,488 | 19,771 |
| 合计 | | 6,849,243 | 52,572 |

来源和上下文可用性均保留。继承v6的binding-only配方：对有结合标签的指定来源去掉明确nonbinding记录；无结合实验标签的序列库不能据此称为已验证binder，也不是监督binding二分类训练集。当前七类来源不新增VHH任务。

### 已实施的数据处理

1. **冻结与追溯**：锁定源manifest、shard和下游保护文件SHA。按来源定义提取真实观测的重链/H3、TCR CDR3α/β、配对、肽和抗原键；人工X画布不作为真实序列。需要的受体注释缺失时记录排除原因，不补造序列。
2. **旧范围重叠过滤**：[构建脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/prepare_immune_v7.py)按角色做联合MMseqs聚类，identity阈值为heavy 0.95、H3/CDR3α/CDR3β/peptide 0.80、antigen 0.70，coverage 0.80；另查配对精确重叠、短肽全长编辑距离、抗原包含保护肽和双向覆盖至少80%的抗原包含关系。这些是冻结的操作性规则，不是穷尽所有同源关系的证明。MHC相同本身不作排除依据。
3. **去重与划分**：按观测链的角色、范围、序列及labels做精确语义去重，故不等于已调和冲突标签。对保留记录的TCR双链core家族、抗体heavy/H3家族建立连通分量，同组只进入一个split；按组hash确定约1%的验证家族，不要求每来源恰好1%的记录。抗体侧未据此证明独立轻链家族隔离。V7重切的是foundation train/valid，作者benchmark划分保持原样。
4. **共同输入资格**：保留旧固定画布与A4观测布局均能容纳的共同记录，冻结后用于模型比较。V7是数据版本，V3/A0–A4是模型/布局版本；prepared仍可含带synthetic mask的旧X画布，A4运行时根据来源掩码抽取观测片段，保留真实junction C/F/W，不把CDR3样本伪称完整受体。
5. **MHC更新**：实际提供的MHC链优先保留；仅有名称时严格查锁定表，唯一匹配才补34位pseudo；不能解析则保留记录和缺失状态。已恢复的MHC重链/B2M各自保留链边界及来源，reference-resolved不冒称实验直接测得。v7新增pseudo220条、恢复长链10,529条；详见下节。

### 尚未落地的部分

旧范围证书仍有264条benchmark H3覆盖待解决。扩展到UpTCR及全部few-shot支持/查询池的[全量审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#tcr-exposure-20260930)已经完成且发现重叠；其[cleanup_plan.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json)仍为`not_applied`。保守并集规则标记的2,119,972条是候选排除量，不是本轮已删除量或已证实泄漏比例；若全部排除，native/TRAIT验证集各只余4条，需先审视规则与验证集设计。MHC更新保留了所有非MHC序列，不能消除这些受体/表位暴露。

因此最新V7已完成导出与MHC接口验证，但尚未形成覆盖当前全部下游范围的最终通过版本。新数据正式使用前仍需完成既有审核，且不能把旧范围证书直接用于新manifest。当前no-ESMC150000对应旧v6训练，新数据导出不追溯改变该checkpoint。最新目录：[immune_v7_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1)；未变化来源使用相对链接，须保留parent目录。

## 2026-09-30 MHC序列优先数据更新（已完成）

**用户已明确采用的新规则，覆盖下文“只读核查/尚不采纳新配方”的历史状态：已提供MHC序列则保留；仅有HLA名称且能唯一查表时补34位pseudo。** 已实际生成两个新prepared版本，原始CSV、原v6/v7和checkpoint保留。只更新MHC上下文及其元数据；样本成员、顺序、train/valid、标签、TCR/肽序列、非MHC区域/掩码及其他元数据逐行不变。

| 新版本 | train / valid总记录 | 新补pseudo（train / valid） | 恢复长MHC+B2M（train / valid） |
|---|---:|---:|---:|
| [immune_v6_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v6_mhc_sequence_v1) | 7,314,689 / 66,810 | 210 / 10 | 38,055 / 2,984 |
| [immune_v7_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1) | 6,849,243 / 52,572 | 218 / 2 | 10,497 / 32 |

两版分别继承各自parent，数字不能相加当独立样本。v6新补MHC上下文共246条（220条pseudo＋26条恢复长链），更新后带MHC=378,127条，其中长链41,039、pseudo337,088。v7新补上下文243条（220＋23），带MHC=339,021条，其中长链10,529、pseudo328,492。B2M独立保留为另一条`mhc`角色链，既不拼成一条序列，也不再压缩为pseudo。

### 规则和来源

1. 直接给定有效MHC序列/多链数组优先保留；不因有HLA名称而替换，不裁剪长链，不猜测补全部分X，已有34位成品也保留。空值/全X继续表示缺失，原值另存。
2. 仅名称时使用锁定的[DTU NetMHCpan4.1表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/mhc_sequence_v1/reference/MHC_pseudo.dat)，仅规范大小写、HLA前缀和标准分隔符/H2别名；同规范名称仅有唯一序列才补入。不截断高分辨率字段、不选一字段代表、不按肽猜HLA。表外/缺失/冲突保留未解析记录；此规则不等于通用MHC-II提取器。
3. 历史全长来源用原`record_id("full", source_row)`追回源JSONL并核对肽，恢复337–366位MHC重链和99位B2M。来源仍标为**早期IMGT等位基因解析/固定B2M参考**，不是实验直接测得。`ingest_fulllength`现在也保留源链。
4. 旧prepared未存全量原记录ID，故以来源＋观测TCR＋肽＋原MHC＋标签精确回溯上游；多个候选只有MHC序列/表示相同才应用，记录候选名称和数量。两个版本实际均为**0未匹配、0序列方案歧义**，不靠模糊匹配或来源顺序猜测。

[版本化上游CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/mhc_sequence_v1/sources)包含TRAIT/native/papers各train/valid，909,801行全部保留；精确补pseudo224条、恢复长链54,093条，是配方过滤前的数量。[source_report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/mhc_sequence_v1/source_report.json)锁定原/新CSV、长链来源及参考表SHA。新prepared仍无MHC的记录：v6为139,988条（表外名称1,399、无名称138,589），v7为134,980条（1,327、133,653）；保持记录与肽–TCR布局，未任意补值。

### 数据接口

- CSV新增`mhc_seq_original`、`mhc_chains_json`、`mhc_resolution_json`；`mhc_seq`作为兼容的主链字段，多链由JSON数组完整携带。公共unified schema新增后两列，旧CSV无这些列时仍走原适配路径。
- prepared保留`immune_llada.semantic.v1`。MHC链metadata含`mhc_representation`（`provided_sequence`/`existing_pseudo34`/`allele_pseudo34`）、`mhc_origin`、`mhc_component`、`mhc_policy`和`sequence_scope`。pseudo使用现有`fragment` scope，并显式标记非连续压缩表示；已知长链/B2M用`full`，未知范围直接序列用`unknown`。不改骨干或词表；MHC仍为固定条件。
- [sources.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py)支持多条MHC并携带解析信息；JSONL增减MHC链时同步重排`regions`/`chain_metadata`索引。旧输入不在线查表。
- [构建入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/update_mhc_inputs.py)、[公共MHC规则](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/mhc.py)、[独立验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/verify_mhc_inputs.py)。本次实际构建脚本的[快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/mhc_sequence_v1/build_source_snapshot)与报告script SHA对应；随后入口增加已有显式多链列的重用支持，不改变已导出内容。
- 未改变的四来源用相对符号链接指向parent，三个更新来源是独立文件；**不能删除/移动parent后将新目录当独立归档**。manifest记录parent和source-report SHA，旧审计证书不直接复制。

### 验证和训练状态

73项测试通过，覆盖HLA精度/别名/冲突、部分X/序列优先、多链原始摄入和适配、索引及source/preprocessing/v3回归。独立复读v6的518,115条、v7的474,001条改写记录，验证行数/顺序/非MHC内容与shard SHA。全部通过A4容量检查，最长998 tokens；每版8类MHC表示/长度边界通过A1/A4 collator，train/valid实际PreparedImmuneDataset加载成功；无MHC、pseudo、长链三类实际样本均通过tiny模型CPU前向/反向，loss和梯度有限。这不是正式预训练结果。

v6保留**1条原已超过135位alpha固定画布**的记录；非MHC逐行相同，证明不是本次引入。它通过A4容量检查，但旧固定画布仍拒绝，不擅自删除或裁剪；v7没有该项问题。证据：[v6报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v6_mhc_sequence_v1/mhc_update_report.json)、[v6验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v6_mhc_sequence_v1/mhc_validation.json)、[v7报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1/mhc_update_report.json)、[v7验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1/mhc_validation.json)。

新v7可被训练接口读取；**正式v3训练仍需项目既有数据/当前下游范围审核**，parent原本pending，本次不签发整体通过证书。pilot配置生成器默认新v7，任务/输出名含数据版本；直接调用训练入口应显式传该`--data-dir`。本数据更新任务未启动正式训练，未修改运行中评测或作者benchmark划分。

## 2026-09-30 当前 foundation 预训练 MHC 表示实查

**结论：当前 no-ESMC150000 对应的 `immune_v6_binding_only` 已使用34位MHC伪序列；三个含肽TCR来源中没有长MHC链或B2M链进入prepared记录。** 这不是本轮新增的转换。新导出的 `immune_v7_clean_r2` 沿用同一MHC表示；不能将v7导出状态当作旧checkpoint已经使用v7训练。

核查依据：[no-ESMC训练配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_llada270m_noesmc_diffusion_v2_immune_v6_2node4gpu_1m.yml:41)、[checkpoint快照来源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/v2_noesmc_immune_v6_150000_20260928/eval_snapshot_manifest.json)、两版manifest和三个来源全部train/valid JSONL。可重复的只读[审计脚本](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/audit_pretraining_mhc.py)及[完整统计/文件SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/pretraining_mhc_audit.json)已保存。逐shard行数与manifest一致，v7带SHA的被扫描shard全部一致，扫描前后manifest SHA未变。其他四个来源的行数读取manifest，未逐行重扫；不宣称此次扫描了所有738万行。

### 实际用于v6训练的数据

七个来源为OAS抗体配对、OTS TCR配对、ASD抗原–抗体、TRAIT、TCR native、TCR papers及TCR repertoire；总train=7,314,689，valid=66,810。MHC统计针对TRAIT/native/papers这三个含肽TCR来源：

| v6来源 | train记录 | 其中带MHC | 其中无MHC | valid记录 | valid带MHC |
|---|---:|---:|---:|---:|---:|
| trait | 11,687 | 10,185 | 1,502 | 540 | 404 |
| tcr_native | 52,817 | 52,664 | 153 | 3,848 | 3,819 |
| tcr_papers | 444,621 | 307,558 | 137,063 | 4,602 | 3,251 |
| 合计 | 509,125 | 370,407 | 138,718 | 8,990 | 7,474 |

全部377,881条带MHC记录均只有一条34位MHC链，共139种字符串；这不是139个已确认唯一等位基因。prepared字段为`chains`、`chain_roles`、`task_type`、`regions`、`chain_metadata`、`metadata`、`labels`、`identifiers`等。带MHC通常是`[mhc, peptide, tcr_beta, tcr_alpha]`，缺MHC是`[peptide, tcr_beta, tcr_alpha]`。三个来源的relation均为binding，对应binding-only预训练配方，不是已训练二分类头。TCR侧既有Fv，也有仅CDR3观测的记录；后者的非观测区域使用X和`synthetic_residue_mask`表达，不能把固定画布误称完整观测TCR序列。

[grammar.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py:625)把MHC放在固定条件蛋白块，接固定肽块和TCR块；模型读氨基酸token，不读HLA名称，也不使用UpTCR的45维手工MHC特征。现有grammar并不强制MHC长度为34，也没有独立pseudo/full标志；支持一般长序列输入的能力不等于本checkpoint已有长MHC训练覆盖。

### 上游到底怎样得到MHC

- TRAIT：[step2_hla.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/trait/scripts/step2_hla.py:20)用PISTE表查名称→伪序列，存入`mhc_seq`。代码对一字段名称如A*02使用首个表内代表，不能当作精确分型。
- native：[common.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/common.py:81)读[PISTE映射表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/piste_tcr_epitope_hla/PISTE/data/raw_data/common_hla_sequence.csv)，有源序列时沿用；不同来源再合入统一CSV。
- papers：[ingest_papers.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/ingest_papers.py:122)使用PISTE表并补充TCRT5表；TCRT5/TCRdesign26可直接带34位字符串，TCRDiff按名称查表。此处历史反向字典对同一pseudo取字典序最小的等位基因名，仅是确定性代表，不能据此恢复唯一完整MHC。TCRDiff入口明确丢弃无法表示的class II和鼠H2；这说明现有摄入范围并非所有pMHC种类。
- **全长来源确实存在，但已在摄入时转成pseudo。** [历史JSONL](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_pmhc_fulllength/records_train.jsonl)共57,913行，含337–366位MHC重链和99位B2M。但[ingest_fulllength](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_native/ingest_repo.py:149)只取TCR、肽及metadata中的等位基因名，再查34位表；不把两条原MHC/B2M序列带进统一训练CSV。v6来源标签`tcr_pmhc_fulllength`仍保留，但带MHC的38,029条train及2,984条valid均为34位。不能从来源名推断模型使用全长MHC。
- [sources.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/sources.py:439)对空MHC和全X占位去掉MHC链，保留肽–TCR布局；部分X不属于全X判定。本轮上游CSV统计见[报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/pretraining_mhc_upstream_audit.json)。这些是既有处理，本轮未执行新过滤。

### 兼容性边界与本轮发现

DTU表精确覆盖v6的107/139种MHC字符串、375,373/377,881条带MHC记录。其余32种/2,508条全部来自`tcrdesign26_pmhc`；其中59条train含部分X，涉及2种字符串。未精确匹配不自动等于错误，需要核对原始定义/版本；本轮不修正、补全或删除。UpTCR自身51/51种均匹配DTU的结论不受此影响，但不能外推为所有预训练MHC已经通过同等核验。

v7 r2当前总train/valid=6,849,243/52,572；三个相关来源的带MHCtrain/valid=336,496/2,282，仍全部34位，共138种。其manifest仍标`audit_pending`，并且历史冻结范围不能代表新增下游范围全通过；本次只验证MHC表示与文件一致性，不签发去污染结论。v3 `observe()`对无`sequence_scope`元数据的链默认标`full`；当前MHC无链元数据，因此该软件scope不应被解释为生物学上完整HLA序列。这是待澄清的元数据语义，不在本轮改布局或训练。

### 对“是否都转pseudo”的回答口径

MHC-I且具备可明确匹配的等位基因或对应全序列，才可按已锁定定义提供兼容的34位表示；不能无条件推广到class II、分型不足或缺失MHC。已有真实全序列的记录可以保留原序列，无需为存储强制丢失信息；只有名字仍需映射，已有34位则无需再次转换。完整MHC作为本模型输入在架构上可行，但现有v6 checkpoint的MHC训练表示是34位，不能直接宣称切换到全长后等价。未来若另建全长/混合输入版本，应记录来源、物种/class、等位基因分辨率、pseudo/full与定义版本，分别验证；本轮不采纳新训练配方。

对UpTCR评测，已有MHCseq与现有MHC条件块在长度/表示类型上可接轨，不要求先反推完整HLA或另造模型架构。此前“需明确伪序列布局”的意思应限定为校验输入适配与披露条件，不能解释为本模型从未见过pseudo。现有αβp首轮仍是无HLA轨；本轮没有修改运行中任务、源数据、划分或checkpoint。

## 2026-09-30 HLA/MHC 伪序列参考表已获取

[DTU参考表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/hla_reference/dtu_MHC_pseudo.dat)按名称与34位序列存储，13010记录含别名/重复。UpTCR51种字符串100%精确覆盖；以别名规范后的候选集合输出，不将多对一映射强行变为唯一等位基因。45种多候选、6种当前表单候选，其中5种匹配鼠H2。官方位点文件+官方B0702全序列的提取一致性已验证，任意新FASTA仍需处理成熟链编号/对齐，不硬编码统一信号肽长度。参见 [完整来源/方法](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md#2026-09-30-公开-hlamhc-伪序列获取路径已验证) 和 [候选映射CSV](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/hla_reference/uptcr_reference_matches.csv)。参考资源外置，不覆盖原数据。

## 2026-09-30 HLA 伪序列获取方法的文献依据

正文Representation of HLA：说明来自已知肽–HLA结构的近肽残基定义；引用NetMHCpan文献34/35。补充Note1仅明确IPD-IMGT/HLA用于规范化等位基因名称，未据此确认完整序列下载来源/版本。当前未定位该批数据的完整等位基因→34位序列映射表、原位点编号或完整提取脚本；官方loader读取已发布MHCseq。应区分构建原则有说明、成品数据已发布、逐条原始生成链路尚未完整定位。未改输入数据或运行任务。

## 2026-09-30 UpTCR MHCseq 实际多样性统计

原9份分类CSV全部314102行共51种不同MHCseq字符串，均为34个标准氨基酸；只读统计，未去重训练记录或修改划分。导出的 [51种序列与逐split频数](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/hla_pseudosequences.csv) 为描述性目录，不是替代评测数据；[统计JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/hla_pseudosequence_summary.json) 保存原文件SHA。全文件按行累计第一/二种占36.81%/25.15%，前三种71.70%；跨场景重复计数保留，不能解读为独立实验或人群HLA频率。

不同伪序列数train/valid/test：majority12/12/12，few-shot20/20/20，zero-shot48/45/29。51种字符串的1275个无序配对按34位直接比较，差异最小1、中位16、最大24；每种字符串等权，不按样本出现次数加权。5个伪序列位置在全部51种中一致（1、5、18、24、31），其余29位在全体中有变化。位置编号是伪序列索引，不是原蛋白残基编号。序列多数有共同框架但并非全相同；某些组合仅差1位，不能据此推断功能等价。未映射等位基因/物种，不把51种字符串等同于51种已确认HLA等位基因。

## 2026-09-30 UpTCR 发布 embedding 文件抽查

在线核验Zenodo中央目录：pretrained_emb/pepesm=41599、tcr_a=27066、tcr_b=27946个.pt；9表所需唯一序列1300/26972/27847全部有对应文件名。每类1个真实文件通过ZIP CRC、CPU weights_only加载及finite检查，float32，样本shape为[15,1280]、[1,145,1280]、[1,140,1280]；实际序列长度9/116/118，说明样本已做长度填充。只验证3个内容样本，不能外推全量文件完整性或确切原提取细节。证据 [embedding_release_recheck.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/research/uptcr_20260930/embedding_release_recheck.json)。已有embedding公开，不是当前原发布benchmark的缺失项。

## 2026-09-30 UpTCR HLA 输入核验

逐行读取原9份分类CSV，314102条记录的MHCseq全部长34：是氨基酸伪序列，不是HLA名称、全长蛋白或结构坐标。论文称位点依据已知肽–HLA结构中的近肽残基定义；当前loader直接读取预制MHCseq，不在线查结构或映射等位基因。官方mhc_encoding每位置拼接one-hot20、BLOSUM50 20、Atchley5为45维；配置34×45，经MHCEncoder卷积/残差块变为34×256，再与肽表示融合。HLA分支不读取预计算ESM特征；缺失ESM-pep/ESM-ppi-TCR流程是另一个问题。当前本模型αβp首轮没有输入MHCseq；加入该字段需独立明确伪序列布局，不能当成完整HLA链。只读核查，未更改正在运行的评测。

## 2026-09-30 UpTCR 原样输入与输出格式

[新入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_binding/run_uptcr_ours.py)读取原9份CSV，兼容Label/label，不修改源文件；314102行通过检查，无去重、删除、重标或split改变。首轮使用FV_alpha/FV_beta/Epitope；HLA及CDR3保留源列但不独立输入。每split特征npy按原行存储，json记录源SHA、模型/代码身份和next_row；predictions.csv保留源列与行序，新增source_row_index和prediction。train归一化、原valid选模、test评分；指标明确AUROC和sklearn average precision。产物在 [运行目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/uptcr_noesmc150000_original_20260930)。

## 2026-09-25 TCR实验统一文档入口

本轮结果已汇入 [TCR_DESIGN_EVALUATION.md](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/TCR_DESIGN_EVALUATION.md)，并在flow_antibody_tcr README/rebuttal链接。保留所有逐样本结果和历史更正；论文表不再添加log Pgen均值列。AAR为用户提供原表；新增Validity/Pgen来自全部采样，binding为每目标首条50条。IMP历史raw-reference72%与对称处理0%明确分开。

## 2026-09-25 TCR IMP 汇总

50个目标全部配对，严格pred_dG<ref_dG；相同重建/pack/FastRelax的reference为0/50（0%），沿用AB历史raw reference比较为36/50（72%）。后者含不对称后处理贡献，不解释为结合成功率。新增imp_per_target.csv（100行，两口径）与imp_summary.csv（2行）；[报告](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax/IMP_REPORT.md)。

## 2026-09-25 TCR pack/relax 输出验证完成

/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax/per_structure.csv共有300行，三stage各100；paired_deltas.csv是IgGM2-reference；verification_per_structure.csv含100行，记录序列不变、非虚拟重原子完整、直接CA坐标RMSD、跨界面短接触及最终PDB重算分项。虚拟NV等不属于应导出的重原子，不计缺失；PDB重算分项因坐标3位小数与内存主分数可能稍异。records/100JSON及relaxed/100PDB完整。REPORT.md为当前正式更正结果。

## 2026-09-25 TCR rebuilt/pack/FastRelax 格式

/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_rebuilt_pack_relax：inputs.csv 100输入；records/逐结构JSON用于完成记录；stripped/仅CDR去侧链的PDB；relaxed/完整处理后的PDB；per_structure.csv的stage为rebuilt、packed、pack_fastrelax，并记录removed_cdr_sidechain_atoms、n_cdr、n_packable、elapsed_s和能量。protocol.json锁定执行脚本与复用AB脚本SHA；ab_existing_results_audit.json存AB已有结果路径及重新统计。旧rosetta_energy目录为raw/chi-only诊断，不能冒充完整处理后的结果。

## 2026-09-25 TCR Rosetta energy 产物

[能量目录](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/rosetta_energy)：inputs.csv 共100输入（method,prot_id,pdb,sha256）；per_structure.csv 共200行，唯一键(method,prot_id,stage)，stage=raw/chi_minimized；summary.csv 含均值/中位数/SD；paired_deltas.csv 差值定义 IgGM2-reference；raw_energy_geometry_audit.csv 含加权能量分项及距离检查；reference_provenance.csv 逐目标原始 AF3 来源及所有裁剪原子坐标匹配数；minimized/ 保存100结构副本；protocol.json、verification.json 锁定方法及检查；paired_energy.png/pdf 为配对图。已有 generation ground_truth.pdb 实际是 AF3 预测参考裁剪，非实验真值，原始与生成的抗原链范围一致。

## 2026-09-25 TCR validity 输出

[输出目录](/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/validity) 包含 per_design_validity.csv（1600 行，design_id 主键）、first50_integrated_metrics.csv（250 行，与 sequence/Pgen/novelty 及 NM2025 epiTCR 结果一对一合并）、first50/all_samples_summary.csv、invalid_designs.csv（严格类型失败候选，并非生物学无效结论）、boundary_mismatches.csv、uniqueness.csv、pgen_validity_crosstab.csv。paired_basic_sequence_valid 为基础检查；paired_valid 为另加严格 A/B HMM 分类的检查；assigned_chain_type 保留原始最佳类型。anarci_cache.json 与 protocol.json 记录输入/代码/HMM/hmmscan 哈希；编号缓存指纹对应编号协议，独立汇总定义见脚本及报告。

Date: 2026-09-23

## 2026-09-23 Manifest source groups versus row provenance

Source-balanced training uses the manifest shard's `source` as its group identifier. Prepared row `source` is provenance and may differ: current v6 samples show `oas` → `oas_paired`, `ots` → `ots_paired`, and `tcr_papers` → `tcr_native`; the last must remain distinct from the `tcr_native` manifest group. The prepared loader now carries runtime-only `source_group` on a `BioSeqRecord` subclass. The collator emits retained-row-aligned `source_groups`, and the Trainer uses these groups for global counts and `source_loss_weights`. Missing/unknown groups fail explicitly.

Original `record.source`, metadata, deterministic rendering seeds, serialized prepared records, manifests, and provenance-based drop counters are preserved. No prepared format migration or data rewrite is needed. CPU integration regressions cover loader → grammar/remap → Trainer, worker/spawn transport, overflow alignment, and independent groups sharing provenance. Submission/failure/retry evidence is in [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md), 2026-09-23.

## 2026-09-22 Training batch metadata and full-coverage balancing

Prepared JSONL rows, shard membership, and offline filters are unchanged. The opt-in source-balanced training stream visits all prepared rows without truncating large sources and cycles small sources only after a full shuffled pass. Equal loss weight is enforced over the global optimizer batch, not each 4-row GPU microbatch. Runtime overflow exclusions still apply and are not silently declared “trained”; the precise coverage/weighting contract is in [BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md).

The [grammar collator](/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py) now returns `_canvas_drop_counts` and `_canvas_drop_counts_by_source` when dropping is enabled. These are **per-batch deltas**, not worker-cumulative snapshots. The remap wrapper preserves them. [FusionTrainer](/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py) consumes them once in the training process, before forward; evaluation deltas are removed and do not contaminate training totals. `canvas_drop/*` and `canvas_drop_by_source/*` are rank-local, consumed-training totals (rank-zero logs), not an all-rank corpus audit or prefetched-row count. Existing all-dropped batches still fail explicitly.

Balanced training additionally injects `source_loss_weights: float32[B]` after gathering actual retained source counts across the optimizer batch. This is runtime loss metadata, never a stored semantic record or encoder condition. Evaluation has no such weights. Balanced resume metadata is `source_sampling.json` beside checkpoint weights; its exact policy must match on resume.

2026-09-14 TCR 生成结果 JSON 的 nullable novelty / availability 及恢复 provenance 契约见 [T4 §7.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。仅为评分产物边界；原始数据、训练布局与生成 MASK/X 语义不变。

2026-09-13 TCR 下游运行时适配更新：原始评测 CSV/JSON 不改写，v5 表征补全与生成目标边界分别见 [T2 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md)、[T3 §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)、[T4 执行手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。与离线训练数据准备不是同一阶段；新缓存/产物必须保留协议身份。决定索引见 [TCR 审计 §9.21](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。

Root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data`

How to run preprocess:
[`dllm/pipelines/immune_llada/README.md`](dllm/pipelines/immune_llada/README.md).
Design rationale / risk register:
[`docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md`](docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md).
Raw-corpus decontam accidents (PASS is not enough; `语料 ∩ benchmark`; key spaces):
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).
Long-lived rules: [`PROJ_GUIDE.md`](PROJ_GUIDE.md).

## v2 fixed-canvas / variable-length decoder contract (2026-09-21)

For the second model version, grammar rendering remains semantic and runtime, but
receptor chains use fixed ESMC canvases: heavy/beta `168` including `<cls>` and
light/alpha `136` including `<cls>`, uniformly for antibody and TCR. The decoder
has respectively `167` and `135` slots. Real amino acids are followed by repeated
EOS tokens; one EOS slot is kept available for termination (practical residue
maxima `166` and `134`). EOS is visible to attention and loss, while PAD remains
an attention/padding concern. See
[`examples/llada/VARIABLE_LENGTH_GENERATION_V2.md`](examples/llada/VARIABLE_LENGTH_GENERATION_V2.md).

The batch now distinguishes `residue_mask`, `chain_eos_mask`, and
`chain_slot_mask`; ESMC `encoder_residue_mask` still contains only real amino
acids. This prevents EOS padding from shifting decoder-to-ESMC amino-acid
alignment. The decoder condition path uses additive fusion and the sampler keeps
pending positions masked, so reference suffixes are not fed back to ESMC. Output
extraction truncates each chain at its first EOS.


TCR 下游评测的输入转换与训练格式的衔接约定（含 others 真实 CDR3αβ 第一版）见 [TCR baseline 审计 §2 / §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)；该约定不改变本节的预训练数据产物与离线流程。

同 others 数据新增 LongA/LongB 运行时输入，复用完整链路径而非 CDR3 补全；字段、校验、池化和不做预处理副本的边界见 [TCR 审计 §9.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。没有改变官方原始 CSV 或预训练 prepared 格式。

The only current immune data implementation is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada` and the formal
training entry is
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
The runtime contract is:

```text
raw source files
  -> source adapter
  -> canonical BioSeqRecord
  -> offline deterministic filters
  -> prepared semantic JSONL shards + manifest
  -> training-time grammar rendering
  -> padding/mask/tensor assembly
  -> per-chain encoder input reconstruction
  -> diffusion/MLM corruption and loss
```

Raw CSV/JSONL parsing, source conversion, receptor completion (when configured),
blacklist/decontamination, deterministic validity and length rejection, and bad-row
replacement are offline operations. The prepared loader builds byte-offset indexes and
decodes semantic rows; it does not create or consume a model-ready token cache. Grammar
encoding, padding, encoder reconstruction, and stochastic masking remain runtime
operations.

Current prepared roots: **v4 published**, **v5 published**. v3 /
`immune_v3_heterotypic` still exist on disk and some checkpoints/eval jobs still
point at them; they are not the current training default. See the pipeline README
version table. Counts: plan §4.1 (v4) / §4.2 (v5).

### Shared layout (v4 and v5)

Seven-source mix: `oas+ots+asd_antibody+trait+tcr_native+tcr_papers+tcr_repertoire`.
Before completing listed sources, the offline pipeline scans complete TCR rows from
the selected `ots` / `tcr_native` / `tcr_papers` sources, applies the same
adapter/filter policy, and stream-counts FR/CDR lengths. The frozen profile, seed,
source policy, observation counts, fallback regions, digest, and (once written)
`completion_sources` are recorded in `dataset_manifest.json`.

Receptor completion (when a source is in `PreprocessConfig.completion_sources`):

- Chains are assembled **beta-first**. A real full-length Fv is kept verbatim with
  no invented region annotation; only missing/fragmentary chains are synthesized.
- Missing regions are `X` at profile-sampled lengths. Synthetic `X` is visible
  input but excluded from diffusion eligibility and loss (`synthetic_residue_mask`).
  The original CDR3β core remains the decontamination/benchmark identifier.
- Junction vs core is decided from provenance (`fv_source` / `provenance` /
  `schema_version`), **never** from the first/last character: an anchor-free core
  can legitimately begin with `C`. `trait` is IMGT junction form (`C...F/W`,
  anchors peeled into FR3/FR4); `tcr_native` / `tcr_papers` are anchor-free core.
- Branching is on the actual `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` column
  contents, **not** the `sequence_scope` label (351-row `tcr_native` counterexample
  and the grammar-ambiguity rationale: plan §2.5). Full-length Fv exists only in
  `tcr_native`; `tcr_papers` is CDR3-scope.
- Default `completion_sources` is `("tcr_repertoire",)`. Withholding a source
  from the list is what disables completion for it.

all-X placeholders (upstream Zenodo 14545852 / TcrDesign: `'X'` means missing):

- all-X **epitope** rows are dropped by the named offline filter
  `quality.blank_epitope` (applies to `trait` / `tcr_native` / `tcr_papers`) so
  the drop count lands in `filter_report.json`. Not a silent adapter `None`.
- all-X **MHC** chains are stripped; the record degrades to the `tcr_peptide`
  layout. `task_type` is derived from whether the MHC chain survived.

Circular-profile guard: `add_tcr_region_lengths` skips regions listed in a
chain's `synthetic_regions` metadata. The profile's learning sources are
`['ots','tcr_native','tcr_papers']`; once those sources can themselves be
completed, an unguarded profile would learn its own synthesized lengths.

`profiles.py::sample_region_lengths` sorts support lengths **numerically**.
`atomic_json_dump` writes the manifest with `sort_keys=True`, so a reloaded
profile iterates lexicographically (`'10'` before `'2'`). Weights were always
looked up by length (the distribution was correct); a given seed drew different
lengths, so a published dataset could not be regenerated from its own manifest.
Only regions whose support spans single and double digits are affected (CDR3);
framework regions are all two-digit and coincidentally escaped.

Relation / grammar (unchanged vs v4): only explicit `binding` / `nonbinding`
labels are relation targets. For pMHC, MHC→peptide presentation `<binding>` is
fixed; peptide→TCR recognition is the supervised target. Generated-block
skeleton tokens (`<prots>`, type marker, `<protd>`) are eligible with the
relation target. Unconditional layouts emit a fixed null-context prefix
`<prots> <null> <protd> <unknown>` at render time (not stored in JSONL).
Audited 2026-09-12: that prefix already applies to exactly the unconditional
layouts (`antibody_pair` / `tcr_pair` / `tcr_single` / `nanobody`) and not to
conditioned ones; `grammar.py` unchanged. Open item: unused `single_entity`
fallback has no prefix.

### v4 published counts

Root: `data/prepared/immune_v4_beta_relation` from
`configs/data/immune_v4_beta_relation.yaml` (completion = `tcr_repertoire` only).
Authoritative per-source table: plan §4.1. Headline: train **7,665,574** /
valid **102,719**; raw 8,441,615 → kept 7,768,293. Formal v4 job:
`train_jobs/protein_esmc_llada270m_diffusion_immune_v4_4gpu.yml`.

v4 `tcr_repertoire` is byte-reproducible from its own manifest after the
numeric-sort fix: 3,000 raw-CSV rows vs the on-disk shard, **0 mismatches**.

### v5 published

Root: `data/prepared/immune_v5_receptor_completion` from
`configs/data/immune_v5_receptor_completion.yaml`
(`completion_sources: [trait, tcr_native, tcr_papers, tcr_repertoire]`).
Authoritative per-source table, v4 Δ, supervised-token shares, max rendered
tokens, profile digest, and invariants: plan §4.2. Headline: train
**7,626,885** / valid **102,308** (13G; valid is the sum of the seven audited
source rows). Do not copy that table here.

### Manifest `filter_names` and `filter_report` counters (2026-09-12)

Code is fixed (`95c04a96`). **Published v5 artifacts were not rewritten** — a
future preprocess writes the new fields. Do not read "code fixed" as "the
13G v5 manifest / `filter_report` already carry them".

**`filter_names` was actively wrong, not merely incomplete.** The field was
hardcoded to `BLOCKLIST_NAMES` plus an optional `quality.homotypic_pair` —
i.e. the list of blocklist **config keys**, not the filters that ran. It
therefore omitted `quality.blank_epitope`, both budget filters
(`budget.max_protein_length`, `budget.max_length`), and
`decontam.repertoire_core_projection`, and it **listed
`asd_nanobody_benchmark`, a source that was never processed**. Manifest
filter provenance was wrong, not just missing one name.

**New semantics:** dataset-level `filter_names` is the **union**, in
first-seen order, of the `RecordFilter` names that `build_filters` actually
constructed for the processed sources. Filters remain per-source
(`quality.blank_epitope` exists only for `trait` / `tcr_native` /
`tcr_papers`), so the union must **not** be read as "every source ran every
name"; each `filter_report.json` source entry also carries its own
`filter_names`. Disabled/empty blocklists never become filters and are
omitted; a filter that ran and dropped zero rows is still listed.

The union v5 *would* have had (published v5 manifest was not rewritten):
`decontam.oas_benchmark`, `budget.max_protein_length`, `budget.max_length`,
`quality.homotypic_pair`, `decontam.ots_benchmark`,
`decontam.asd_antibody_benchmark`, `dedup.replaces_trait`,
`decontam.trait_benchmark`, `decontam.t4_refbinder`, `decontam.t2t3_eval`,
`quality.blank_epitope`, `decontam.repertoire_core_projection`.

**Plan §3.1 item 13 counters are implemented.** `filter_report.json` now
carries, per source and in `totals`, three kept-record transformation
counters — `downgraded_all_x_mhc`, `beta_only_completed`,
`alpha_only_completed` — distinct from the six first-failure drop counters
(`count_immune_drops.py` still compares the original six keys).
`dropped_all_x_epitope` is not duplicated; it stays the
`quality.blank_epitope` filter attribution. **Authoritative post-hoc v5
numbers, including the naive-vs-correct `downgraded_all_x_mhc` caveat
(do not recount MHC-strip with a peptide-without-MHC heuristic): plan §4.2.**
The published v5 `filter_report.json` does not contain these keys.

### Still open (do not mark done)

- `tcr_papers` records still carry `source: "tcr_native"` in record identity
  (pre-existing). `metadata.dataset_source` and the shard name are correctly
  `tcr_papers`.
- The published v5 `dataset_manifest.json` / `filter_report.json` still have
  the old `filter_names` and no item-13 counters.

The ingest root cause (`ingest_papers.py` `_drop_placeholder` not applied to
`epitope` / `mhc_pseudo`) is owned by
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md)
§6.0.1 and remains **the single most important unfixed item**. Rebuilding
`tcr_papers_v2/dataset/` would re-admit all-X rows.

The former `qwen3_vl_arch/data` alias tree, old `training` tree, `GRAMMAR_V1.md`, old dataset
implementation, and retired builders/tests/jobs are deleted. `refactor_baseline` and
`docs/archive` remain historical evidence and are not current runtime inputs.

## Historical pre-refactor notes（不可作为当前入口）

The old direct-CSV seven-source mix (2026-08-29, after near-dup moves) kept
**7,794,375** train rows. Those counts are **not** v4/v5 prepared counts.
v4 published counts: plan §4.1. Decontam accidents, key spaces, and the
prefix-sampling / `语料 ∩ benchmark` / `PASS: true` rules:
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).

Still-true format pitfalls (do not drop):

- **Do not use** `data/tcr_repertoire/dataset/build_report.json::split_counts`
  for row counts. Near-dup relocation rewrote CSVs, not that report. Counts
  come from the CSV + `data/tcr_native/dataset/near_dup_eval_move_report.json`.
  `build_report.json` remains the **decontam** authority (`PASS` /
  `decontam_mode` / `blocklist_provenance`).
- **`TCR_PAPERS_DEFAULT_DIR` points at `data/tcr_papers_v2/dataset`**
  (fixed 2026-08-29). Pre-fix stats may silently have measured v1 (−274k rows).
- **ASD valid loss is not a model-selection signal** for v5 and earlier: same
  blocklist, train drop 67.5% vs valid drop 5.0%. Root cause found 2026-09-19 --
  the blocklist is generated *from the split's own train.csv*
  (`decontam_extra.py:52`), so its `H:` keys are exact train heavy chains and a
  cluster-disjoint split structurally cannot hit them on valid. Fixed by
  decontaminating before splitting (`downstream/asd/scripts/step6_symmetric.py`,
  corpus `downstream/asd/step6_symmetric/`, config
  `configs/data/immune_v6_binding_only.yaml`); post-rebuild hit rate is 0.00% on
  all three splits. Detail: DATA_PIPELINE_README §6.4 and
  `downstream/asd/README.md`.
- **Layout shares must be reservoir-sampled**, never a prefix.
  `tcr_papers_v2` is seven corpora concatenated; the first 30k rows are 100%
  `tcr_peptide`, the full set is 80.3% `tcr_pmhc`.
- Record share ≠ residue share. `ImmuneSourceSpec.weight` was historically
  discarded at load; mixture was disk-row-count. Prepared `BioSeqRecord`
  now stores source weight; whether loss reads it is a separate modelling
  decision (plan / DATA_PIPELINE_README §6.3).

Decontam accidents (`PASS` ≠ freshness; exact blocklists must be benchmark-derived,
never `语料 ∩ benchmark`; dual-layer load-time filter) are owned by
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md) §5.

### valid/train 近重复：`split_disjoint_PASS` 也不是充分证据

同一类问题的第二个面：各源报告里的 `split_disjoint_PASS=true` 只保证三 split 的 CDR3β
**精确**互斥。编辑距离 1 的变体不算 overlap，于是 valid 里可以塞满 train 序列的单点突变
体，报告照样全绿。ckpt 是按 valid 的 `eval_loss` 选的，这直接影响选模。

**先说度量单位，这是这件事最容易做错的地方**：近重复必须按**该源的生成目标**来量，不是
一律按 CDR3。`ImmuneSourceSpec` 的 `roles` 决定哪些链是固定上下文、哪些是去噪目标：

- `tcr_native` / `tcr_papers` / `trait` / `tcr_repertoire`：epitope 和 MHC 是固定上下文，
  **CDR3 才是生成目标** → 按 CDR3β core 量。
- `oas` / `ots`：两条链 100% 都是生成目标（实测 `gen_res == res`）→ 按**全长配对 Fv** 量。
- `asd_antibody`：抗原是固定上下文（生成残基仅占 36%）→ 按 **heavy_fv + light_fv** 量。

按错单位会得出完全相反的结论。`ots` 若按 CDR3 量是 exact 27.25% / Lev≤1 69.90%，看着像
重大泄漏；但它按全长配对量是 exact 0.00% / Lev≤1 1.85%。差异不是切分失败，而是天然
repertoire 里同一条 CDR3β 本来就会与不同 α 链配对——而模型要生成的是整条链，CDR3 复现
不构成答案键。

2026-08-28 晚七源全测（valid 抽样 2,000–全量，train 全量为参考）：

| source | 度量单位 | exact | Lev≤1 | 处置 |
|---|---|---:|---:|---|
| `tcr_papers`(v2，v3 在用) | CDR3β core | — | **49.2%** | 已搬迁 |
| `tcr_papers`(v1) | CDR3β core | — | 46.8% | 已搬迁（当日早先） |
| `tcr_native` | CDR3β core | — | 40.9% | 已搬迁（当日早先） |
| `tcr_repertoire` | CDR3β core | 0.00% | **38.9%** | 已搬迁 |
| `trait` | CDR3β 全 junction | — | **19.1%** | 已搬迁 |
| `asd_antibody` | heavy_fv+light_fv | 0.00% | 4.20% | **无需处理** |
| `ots` | 全长配对 Fv | 0.00% | 1.85% | **无需处理** |
| `oas` | 全长配对 Fv | 0.00% | 0.45% | **无需处理** |

即：占训练**残基** 94% 的三个全长源（`oas`+`ots`+`asd_antibody`）在自己的生成目标上
几乎没有 valid/train 近重复，问题**集中在 CDR3 生成类的四个源**，且都已处理。

`tcr_papers_v2` 漏做的原因值得记：v1 当天早些时候做过这一步，但 v2 是
`finalize_papers.py --out-root data/tcr_papers_v2` 从零重建的，**不继承 v1 已清洗的
split**。「某个源已经处理过」不能推广到它的重建版本。

处理方式是**搬进 train 而非丢弃**（`scripts/data/tcr_native/move_near_dup_eval_rows.py`），
迭代到收敛——搬入 train 的行会成为新参考，单轮不够（v2 用 3 轮，trait 用 5 轮）。

**这不是答案键泄漏。** 所有已处理源中，被搬走的 eval 行里 `(CDR3β, epitope)` 组合在
train 中精确出现的都是 **0 行**。修的是选模信号的可信度，不是补泄漏窟窿。

连带后果：新旧 `eval_loss` 不可比（valid 变小、变难、构成变了），不要与上一轮的
0.2578 / 0.5638 横向对比。

**顺带纠正一个常被引用的过期数字**：「valid 里 73% 是 `tcr_pmhc_fulllength`，train 只
8%」描述的是「单一混合 eval + 前缀截断」时代。现在 `subsample_seed=0` 走 reservoir 抽样、
eval 分源各截 `max_eval_rows_per_source=2000` 行，实测 eval 构成是**七源近等权**（各
14.93%，`trait` 因只有 1,393 行占 10.40%），而非任何单源占 73%。逐源实测表与「等权 vs
按 train 比例加权哪个才是对的选模口径」的论证见
`downstream/benchmark/audit_2026_08_27/RETRAIN_PLAN.md` §3c。

**record 与 residue 口径严重脱节，且当前无法调节**：`ImmuneSourceSpec.weight` 在
`ImmuneBioSeqDataset.__getitem__` 中被丢弃，混合比例纯由磁盘行数决定。
`tcr_repertoire` 占 25.85% 记录但仅 2.03% 残基（裸 CDR3β，均长 13）；
`asd_antibody` 反向，3.62% 记录吃掉 11.37% 残基（长抗原，均长 523）。

### 长度上限口径（2026-08-28 修正）

`DataArguments.max_length` / `max_protein_length` 的默认值已由 **512 改为 1024**，
与全部 `train_jobs/protein_esmc_*immune*.yml` 显式传参一致。基类
`dllm/utils/configs.py::DataArguments.max_length` 本来就是 1024，是两个 protein 入口
把它往下覆盖了。

过滤语义为 `_record_chain_lengths_ok`：每条链 ≤ `max_protein_length`，**且**
`sum(len(chain)) + 3*n_chains + 8 ≤ max_length`。第二个（总长）约束才是实际生效的那个。

**只有 `asd_antibody` 对该参数敏感**，因为它是 antigen+heavy+light 三链，抗原长度分位数
为 q0.25=395、q0.5=q0.75=q0.9=q0.95=**607**（超半数行共用同一条 607 aa 抗原，
来自 `buzz`/trastuzumab 突变库）：

| `max_length` | 抗原预算中位数 | `asd_antibody` 保留 |
|---:|---:|---:|
| 512 | ~268 aa | 159,331（24.3%） |
| 768 | ~524 aa | 295,583（34.8%） |
| **1024** | ~780 aa | **276,412**（32.5%，叠加去污染后） |

607 正好夹在 512 与 1024 的预算之间，故该参数是悬崖式的。其余六源在 512 与 1024 下
逐行相同。历史报告若未显式传 1024，其 `asd_antibody` 行数不可用——
`RETRAIN_PLAN.md` §7.2b/§7.8 初版即因此误记为 159,331，已修正为 276,412。

### 新增数据源（2026-08-28）

- **`data/tcr_repertoire/dataset`** — TcrDesign-2026 `pretrain/bCDR3_train.csv` 的无标签
  单链 CDR3β。读 40,308,610 行，池 40,257,598，train 上限截到 2,000,000、簇级去污染
  再剔 28,206 后为 **1,971,794**（valid 201,504 / holdout 201,170）。
  `build_report.json` 记录 `decontam_mode=exact+cluster_0.80_0.80`，blocklist 并集
  124,176 核心，`residual_blocked_in_train=0`，三 split 互斥。当时是唯一渲染为
  `tcr_single` 的源（v4 起该源已补成双链）。详见 `data/tcr_repertoire/README.md`。
- **`data/tcr_papers_v2/dataset`** — v1 四源（`tcrt5` 295,316 / `tcrdiff` 100,590 /
  `gratcr_tep` 17,768 / `epidiff` 2,484）加 TcrDesign-2026 三层（`tcrdesign26_beta`
  98,536 / `tcrdesign26_paired` 18,383 / `tcrdesign26_pmhc` 163,843）；以上为
  `finalize_report.json::by_source`，即去重后跨 split 的 final rows，合计 696,920。
  split 为 train 668,331 / valid 14,449 / holdout 14,140，唯一 epitope 3,218，
  相对现役语料净新 epitope **+1,763**（v1 为 +881）。`decontam_mode
  =exact+cluster_0.80_0.80`，`residual_binding_exact_hits=0`、`residual_t4_hits=0`、
  `split_disjoint_PASS=true`。v3 通过把 `TCR_PAPERS_DIR` 指向该目录接入，
  **未新增 dataset token**，因此 `--dataset_args` 里仍写 `tcr_papers`。

### 新落盘但未接线的原始数据（2026-08-28）

由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/download_missing_tcr_sources.sh`
下载，该脚本可重跑（已校验文件跳过）。

| 路径 | bytes | 校验 |
|---|---:|---|
| `data/tcr/vdjdb-2026-06-03.zip` | 41,017,409 | 与 GitHub `content-length` 一致 |
| `data/ots_tcrlang/raw/TCRLang_Datasets.tar.gz` | 126,348,093 | MD5 `f2d5cbdcbc518c7f4b73f27515ecb37d` ✓ 官方 |
| `data/ots_tcrlang/raw/tcrlang-weights.tar.gz` | 166,045,148 | MD5 `8bbade5ba653096a490467cbc68b4034` ✓ 官方 |
| `data/ots_tcrlang/raw/OTS_CoherenceCode.tar.gz` | 86,914 | MD5 `53900bf6864a5f4a2cd07876698eba8c` ✓ 官方 |

**VDJdb 2026-06-03**：`vdjdb_full.txt` 由 139,745 行（2025-12-29）增至 192,754 行
（+53,009，+37.9%），已解包至 `data/tcr/vdjdb_2026_06_03/vdjdb-2026-06-03/`。发布包
结构变了：2026 版不再含 `*_scored.txt` / `*_broken.txt` / `_filtered`，只有 10 个核心
文件，因此 zip 从 72.8 MB 缩到 41 MB——**不代表数据变少**。
`scripts/data/tcr_native/ingest_papers.py` 中「+3 net-new epitopes for UniPMT /
VDJdb / McPAS / GLIPH combined」是针对旧版的判断，**已过期**，需用
`assess_candidate.py` 重评。

**Zenodo 11208211（OTS/TCRLang，CC-BY-4.0）**：解包至
`data/ots_tcrlang/TCRLang_Data/`，格式为每行 `<BETA_FV>|<ALPHA_FV>` 全长可变域。
train paired 1,361,284 / test paired 100,000 / eval paired 100,000；
train heavy 4,624,002 / light 4,585,975（其 "heavy" 指 β、"light" 指 α）。

> ⚠️ **该记录的官方 test/eval 不能作为本项目的 held-out benchmark。** 按全长 β+α 精确
> 配对比对 `data/ots_paired_clean/final/train.csv`（即 `OTS_DEFAULT_DIR`，2,102,715 行）：
> test 命中 98,221/100,000 = **98.2%**，eval 命中 98,304/100,000 = **98.3%**。两边同源于
> OTS 而本项目自行重切 split，把对方测试集切进了训练集。反向亦然：不能拿本项目模型
> 与 TCRLang 论文在该 test 上报告的数字比较。其 train（136 万）小于本项目 OTS
> （210 万）且同源，预计不增序列覆盖。可用价值在 `tcrlang-weights.tar.gz`
> （配对模型权重，可作 baseline）与 `OTS_CoherenceCode`（α/β V-gene / V-allele
> coherence 校验函数 + 两个测试 pkl）。

## AB/TCR canonical v2（2026-08-04；不是现役训练输入）

Authoritative record: [`IMMUNE_RECEPTOR_DATA_V2.md`](IMMUNE_RECEPTOR_DATA_V2.md).
`export_ready=false` / `training_ready=false`. Do not treat `data/immune_receptor_v2`
as the current LLaDA prepared mix.

## 当前训练数据单一入口（2026-07-23）— 已被取代

Historical 7L / `bioseq_grammar_v1` Arrow lineage is owned by
[`TRAINING_DATA_CATALOG.md`](TRAINING_DATA_CATALOG.md). It is not the current
prepared semantic JSONL mix. Do not restore a raw-CSV training loader.

The 2026-07-08 downstream decontam bank predates the 2026-07-21 MINT official /
NM2025 / Track-A artifacts and is not proof that the current prepared mix is
clean. Current decontam rules live in
[`examples/llada/DATA_PIPELINE_README.md`](examples/llada/DATA_PIPELINE_README.md).

## MINT 五任务官方 notebook 数据（2026-07-21）

Downstream-owned. Canonical root and protocol:
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`
and `data/downstream/mint_official`. Do not restate task schemas or row counts here.

## High-Level Inventory

Top-level data size:

| Path | Size | Current role |
|---|---:|---|
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed` | 27G | cleaned nanobody/VHH pretraining CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean` | 22G | cleaned paired antibody heavy/light CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_raw` | 19G | mixed nanobody raw sources |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw` | 18G | TCRdb2.0 bulk repertoire raw zips/metadata |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean` | 12G | cleaned paired TCR beta/alpha CSV |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_raw` | 7.5G | OTS raw paired TCR CSV.gz |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_paired_raw` | 3.4G | OAS raw paired antibody CSV.gz |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr` | 903M | VDJdb, McPAS, MIRA, IEDB/PIRD-related TCR specificity resources |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2` | 664M | existing JSONL mix: PPI + TCR-epitope |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi` | 653M | STRING-style PPI Hugging Face Arrow dataset |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed` | 500M | existing JSONL mix capped to max chain length 512 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream` | 43M | downstream benchmark data and symlinks |

Main file extensions under the data root are `.gz`, `.fasta`, `.json`, `.csv`, `.zip`, `.pdb`, `.py`, `.txt`, `.npy`, `.tsv`, `.parquet`, `.jsonl`, and Hugging Face `.arrow`.

## Closest Existing Unified JSONL

Current `processed` and `processed_v2` are the closest existing multi-chain JSONL format.

Record shape:

```json
{
  "chains": ["SEQUENCE_A", "SEQUENCE_B"],
  "types": ["other", "other"],
  "targets": [0, 1],
  "source": "ppi"
}
```

Stats:

| Dataset | Train rows | Val rows | Sources | Notes |
|---|---:|---:|---|---|
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed` | 805,095 | 14,405 | ppi, vdjdb, mira, mcpas | safer for current model; max chain length 512 |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/processed_v2` | 803,591 | 14,377 | ppi, vdjdb, mira, mcpas | preserves PPI chains up to 32,000 aa; needs cropping/bucketing |

`processed_v2/train.jsonl` distribution:

| Source | Rows |
|---|---:|
| ppi | 639,866 |
| vdjdb | 91,122 |
| mira | 46,824 |
| mcpas | 25,779 |

Chain-count distribution:

| Number of chains | Rows |
|---:|---:|
| 1 | 13,162 |
| 2 | 719,311 |
| 3 | 71,118 |

Top type combinations:

| Types | Rows |
|---|---:|
| `["other", "other"]` | 639,866 |
| `["beta", "antigen"]` | 73,532 |
| `["alpha", "beta", "antigen"]` | 71,118 |
| `["beta"]` | 13,162 |
| `["alpha", "beta"]` | 5,913 |

Important limitation: this JSONL does not include the large cleaned OAS, OTS, nanobody, or TCRdb2.0 pools yet. It is not the full foundation-model pretraining corpus.

## Cleaned Paired Antibody: OAS

Final/current path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/oas_previous_clean/splits/cleaned_merged_data_step_clustered_{train,valid,holdout}_oas_label.csv`

Rows:

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 2,486,443 | 2,486,442 |
| valid | 12,554 | 12,553 |
| holdout | 12,654 | 12,653 |

CSV schema:

```text
h_sequence, l_sequence, species, l_locus,
h_v_call, h_d_call, h_j_call, l_v_call, l_j_call,
source,
h_fwr1, h_cdr1, h_fwr2, h_cdr2, h_fwr3, h_cdr3, h_fwr4,
l_fwr1, l_cdr1, l_fwr2, l_cdr2, l_fwr3, l_cdr3, l_fwr4,
cleaned_h_sequence, cleaned_l_sequence,
H_cluster_id, L_cluster_id, ab_cluster_key, ab_cluster_id,
ab_cluster_id_counts, split, h_region_labels, l_region_labels
```

Semantics:

- `source=OAS`
- `cleaned_h_sequence` is the heavy chain sequence
- `cleaned_l_sequence` is the paired light-chain-side sequence; `l_locus` is usually K or L
- FR/CDR fields preserve region-level segmentation
- cluster fields support leakage-aware split/grouping

For BioSeq foundation, this should map to:

- `task_type="antibody"`
- `complex_type="<type_ab>"`
- `chains=[heavy, light]` after role-oriented ordering
- `chain_roles=["antibody_heavy", "antibody_light"]`
- `targets=[0,1]` by default
- `regions` and V/J metadata preserved but not necessarily tokenized in v1

## Cleaned Paired TCR: OTS

Final/current path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ots_paired_clean/final`

Rows:

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 2,102,716 | 2,102,715 |
| valid | 10,620 | 10,619 |
| holdout | 10,622 | 10,621 |

CSV schema is aligned with OAS:

```text
cleaned_chain1_seq, cleaned_chain2_seq,
chain1_cdr3, chain2_cdr3,
chain1_anarci_type, chain2_anarci_type,
chain1_FR1, chain2_FR1, chain1_CDR1, chain2_CDR1,
chain1_FR2, chain2_FR2, chain1_CDR2, chain2_CDR2,
chain1_FR3, chain2_FR3, chain1_CDR3, chain2_CDR3,
chain1_FR4, chain2_FR4,
species, data_type,
chain1_type, chain2_type,
chain1_v, chain1_j, chain2_v, chain2_j,
source_file,
chain1_cluster, chain2_cluster, pair_cluster, cluster_id, split
```

Semantics:

- `data_type=tcr`
- `chain*_type` is beta/alpha
- `chain*_anarci_type` is B/A
- V/J metadata and FR/CDR segmentation are available

For BioSeq foundation, this should map to:

- `task_type="tcr"`
- `complex_type="<type_tcr>"`
- `chains=[beta, alpha]` after role-oriented ordering
- `chain_roles=["tcr_beta", "tcr_alpha"]`
- `targets=[0,1]` by default

## Cleaned Nanobody/VHH

**训练用最终路径（去泄漏后，2026-07-08 起用这个）**：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean`

由 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/clean_nanobody_training.py` 从 `step6_final` 产出。两处清洗：(1) 删除所有 `nbbench_*` 源行——它们是 NbBench 下游基准的样本，被 step0 解析器混进了训练集，留着即对下游 nanobody 评测直接泄漏；(2) 对 `cleaned_seq` 施加 [90,160] aa 长度窗（真实 VHH 约 110-130 aa，p1=97/p99=132，此窗保留 99.9% 真样本、只切离群）。**不做**序列相似度去重（由独立的跨测试集去重工具负责）。

| Split | step6_final rows | step7_clean rows | drop nbbench | drop len |
|---|---:|---:|---:|---:|
| train | 11,649,792 | 11,525,884 | 109,644 | 14,264 |
| valid | 58,862 | 58,311 | 479 | 72 |
| holdout | 58,981 | 58,336 | 577 | 68 |

报告：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step7_clean/clean_report.json`。

上游/原始路径（含 nbbench，勿直接训练）：

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/nanobody_processed/step6_final`

| Split | Lines including header | Examples |
|---|---:|---:|
| train | 11,649,793 | 11,649,792 |
| valid | 58,863 | 58,862 |
| holdout | 58,982 | 58,981 |

CSV schema:

```text
vhh_seq, source, cleaned_seq, anarci_chain_type,
FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4,
cluster_id, split
```

For BioSeq foundation, this should map to:

- `task_type="antibody"`
- `complex_type="<type_nb>"` or `<type_ab>` with `chain_roles=["nanobody_vhh"]`
- `chains=[cleaned_seq]`
- `targets=[0]`
- FR/CDR regions preserved

## TCR Specificity Resources

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr`

Representative formats:

- `vdjdb_full.txt`: TSV with `cdr3.alpha`, `v.alpha`, `j.alpha`, `cdr3.beta`, `v.beta`, `d.beta`, `j.beta`, `species`, `mhc.a`, `mhc.b`, `mhc.class`, `antigen.epitope`, antigen metadata, method metadata, tissue/donor metadata, and score.
- `McPAS-TCR.csv`: CSV with `CDR3.alpha.aa`, `CDR3.beta.aa`, species/category/pathology, antigen protein, `Epitope.peptide`, `MHC`, tissue/T cell type, TRAV/TRAJ/TRBV/TRBD/TRBJ, PubMed ID, and remarks.
- `MIRA/ImmuneCODE-MIRA-Release002.1/peptide-detail-ci.csv`: CSV with TCR beta bioidentity/nucleotide sequence, experiment, ORF coverage, peptide amino acids, and genome coordinates.
- PIRD-related code/reference files are present under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr/PIRD_alt`.

These are not in a single unified schema yet. Existing `processed_v2` already contains a subset converted from VDJdb/McPAS/MIRA into `chains/types/targets/source`.

For BioSeq foundation, these should map to:

- `task_type="tcr_pmhc"`
- `complex_type="<type_tcr_pmhc>"`
- `chains=[beta]`, `[alpha,beta]`, `[beta,peptide]`, or `[alpha,beta,peptide]` depending on availability
- future extension: add MHC chain or MHC allele as metadata/conditioning, not necessarily as sequence in v1
- `targets` should usually include receptor chains, while peptide/MHC may be fixed context depending on task

## TCRdb2.0 Bulk Repertoire Raw Data

Root:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0`

Downloaded structure:

- 263 project zips under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/project_zips`
- 263 project metadata CSVs under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/metadata`
- 1 healthy reference zip under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/raw/healthy`
- validation manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/tcr_bulk_raw/tcrdb2_0/manifests`

Metadata CSV schema example:

```text
CellSource, CellType, Condition, Chain, SampleId, ExperimentId, ProjectId,
RunId, Species, Comment, Gender, Instrument, LibraryLayout,
LibrarySelection, LibraryStrategy, Length, Spots
```

Project zip CSV schema example:

```text
AASeq, cloneCount, cloneFraction, Vregion, Dregion, Jregion,
NNSeq, Length, RunId, Chain
```

Healthy reference CSV schema:

```text
AASeq, Vregion, Dregion, Jregion, cloneFraction, cloneCount
```

For BioSeq foundation, this should map to single-chain or beta/alpha repertoire records first:

- `task_type="tcr_repertoire"` or `task_type="tcr"`
- `complex_type="<type_tcr>"`
- `chains=[AASeq]`
- `chain_roles=["tcr_beta"]`, `["tcr_alpha"]`, or chain-specific role from `Chain`
- `targets=[0]`
- clone count/fraction and disease/source metadata preserved

This source should be capped or downsampled during mixture training so it does not drown paired-chain learning.

## PPI

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi/string_model_org_90_90_split`

Hugging Face Arrow schema:

```text
IDs: string
score: float64
OrgA: string
OrgB: string
SeqA: string
SeqB: string
```

Splits:

| Split | Examples |
|---|---:|
| train | 645,692 |
| valid | 5,854 |
| test | 1,322 |

Current `processed` converts this to:

- `chains=[SeqA, SeqB]`
- `types=["other", "other"]`
- `targets=[0,1]`
- `source="ppi"`

For BioSeq foundation:

- `task_type="ppi"`
- `complex_type="<type_ppi>"`
- `chain_roles=["protein_a", "protein_b"]`
- keep `score`, organism IDs, and pair IDs as labels/metadata if needed

## Downstream Benchmark Data

Path:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream`

Main formats:

- CDR infilling: JSON/JSONL-style records with full chain sequences plus `{cdr_mode}_seq` and `{cdr_mode}_pos`.
- TCR binding: VDJdb/McPAS/ATLAS/IEDB/DeepInsight/TCRDesign/Nature Methods style task files.
- FLAb/in-silico/comp-chain: CSV/FASTA-style task files, some paths are symlinks to older AirGen locations.
- Humanization: documented as incomplete.

These should be treated as evaluation/fine-tuning data, not first-pass pretraining mixture data.

## Current Code-Level Schemas

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/bioseq/adapters.py` defines the intended `bioseq.v1` JSONL:

Required:

```text
chains, task_type, source
```

Preferred optional fields:

```text
chain_roles, targets, split, labels, regions, metadata, schema_version
```

The old raw-CSV dataset and generic collator described in this historical section were deleted.
The current semantic record contract is implemented by
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/records.py`,
with prepared loading in `dataset.py`, grammar/tokenization in `grammar.py` and `esm_encoding.py`,
and batch assembly in `collator.py`. The current training model receives semantic records only
after offline preparation; no raw source dataset is imported at runtime.

## Implication for Adapting Qwen

The model-side input should not be designed around raw CSV columns. The stable boundary should be a unified BioSeq JSONL/example object. Chain-level `targets` is only a coarse default; complex conditional generation must be resolved through a task view or `generation_spec` into token-level masks.

```json
{
  "schema_version": "bioseq.v1",
  "task_type": "tcr_pmhc",
  "complex_type": "type_tcr_pmhc",
  "chains": ["TRA_SEQUENCE", "TRB_SEQUENCE", "PEPTIDE_OR_ANTIGEN"],
  "chain_roles": ["tcr_alpha", "tcr_beta", "peptide"],
  "targets": [0],
  "generation_spec": {
    "name": "beta_epitope_to_alpha",
    "fixed": [
      {"chain": 1, "scope": "full_chain"},
      {"chain": 2, "scope": "full_chain"}
    ],
    "generate": [
      {"chain": 0, "scope": "full_chain"}
    ]
  },
  "regions": {
    "0": {"CDR3": "..."}
  },
  "metadata": {
    "species": "HomoSapiens",
    "v_gene": "...",
    "j_gene": "...",
    "mhc_allele": "..."
  }
}
```

For the first Qwen-derived diffusion model, the canonical tensor batch should be:

```text
input_ids             [B, L]
labels                [B, L]
attention_mask        [B, L]
visible_mask          [B, L]
diffusion_loss_mask   [B, L]
fixed_context_mask    [B, L]
diffusion_target_mask [B, L]
chain_ids             [B, L]
chain_role_ids        [B, L]
task_type_ids         [B]
position_ids_inner    [B, L]
position_ids_chain    [B, L]
```

The view sampler should support at least these target constructions:

- Chain completion: fixed heavy generates light, fixed light generates heavy, fixed beta+epitope generates alpha.
- Antibody-antigen receptor design: fixed antigen generates antibody heavy/light or nanobody VHH; fixed antigen plus one antibody chain generates the paired antibody chain.
- Antigen-conditioned CDR design: fixed antigen plus antibody/nanobody FR residues generates all CDR regions or one selected CDR.
- MHC-conditioned TCR-pMHC denoising: fixed MHC/HLA generates or denoises peptide plus available TCR alpha/beta chains.
- Peptide design: fixed TCR alpha/beta plus MHC/HLA generates peptide or epitope.
- TCR design: fixed peptide or epitope plus MHC/HLA generates TCR alpha/beta.
- pMHC-conditioned TCR CDR design: fixed peptide/epitope plus MHC/HLA and TCR FR residues generates all TCR CDR regions or one selected CDR.
- Region infilling: fixed antibody/TCR FR regions generate all CDR regions.
- Single-region infilling: fixed all other residues generate one selected CDR.
- Inverse region infilling: fixed six CDR regions generate FR regions.
- Conditional receptor generation: fixed antigen/peptide/MHC/PPI partner generates selected receptor chains.

`full_denoise` in the BioSeq foundation loader should be read as full denoising over eligible target chains, not all biological chains. Antigen, peptide, MHC, and HLA-like chains are fixed context by default. They are visible conditioning residues but should not be remasked or included in `diffusion_loss_mask`.

## Encoder Tokenizer Boundary

The active immune loader under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data` keeps canonical biological records independent from the ESMC/decoder tokenizer. Tokenization remains a collator/encoder concern. The retained model-layer utilities under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch` do not imply that its deleted `data` tree still exists.

Local tokenizer verification:

- ESM2 snapshots under `/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D`, `/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D`, and `/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D` use id 31 for `<null_1>`.
- ESMC snapshots under `/c20250601/mj/model_weights/esmc/ESMC-300M`, `/c20250601/mj/model_weights/esmc/ESMC-600M`, and `/c20250601/mj/model_weights/esmc/ESMC-6B` use id 31 for `|`, and mark `|` as an additional special token.
- Standard amino-acid token ids and `<mask>` id 32 match between the verified ESM2 and ESMC local tokenizers.

Implementation rule:

- Use the ESM2/MINT-compatible tokenizer for Ophiuchus-Ab and no-encoder MINT paths.
- Use the encoder's own local Hugging Face tokenizer when an ESMC encoder is active.
- Do not infer multi-chain interaction capability from the `|` token alone. The ESMC/ESMFold2 paper places explicit multi-chain complex modeling in ESMFold2, where each chain is encoded independently by frozen ESMC 6B and then fused through downstream pair/folding/diffusion modules.

The current normalization boundary is already the prepared `BioSeqRecord`/semantic JSONL contract. Future source additions must implement an offline adapter and manifest under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; they must not reintroduce a raw-CSV runtime loader or the deleted qwen data alias.

---

## 训练↔下游测试去重（integrated data 版，2026-07-08）

**原因**：整合全部数据训一个统一版前，必须保证训练语料不泄漏进任一下游 test，否则 headline 全线虚高。通用蛋白 50% 阈值不适用于抗体/TCR 设计（CDR3 是克隆型主键、框架区高度保守），按类定键与阈值。

**工具**（`scripts/data/dedup/`，env `protenix_abtcr`；mmseqs 用 `flow` env 的 18 版）：
1. `build_downstream_banks.py` → `data/dedup/banks/`：把所有下游 **test/eval** 读成按生物类型的 bank（跳过 benchmark 自带 train/background split，避免过删）。规模：`ab_cdrh3=9,619`、`ab_heavy=106,806`、`ab_light=6,329`、`tcr_cdr3b=32,887`、`tcr_cdr3a=16,969`、`antigen=221`、`ppi_proteins=15,051`。覆盖 OAS-pairing / SAbDab-CDR-infill(H1/H2/H3) / FLAb / OTS-holdout / T4-gen / NbBench(12) / IRBench-PPI / MINT(6) / NM2025 / TCR-clustering·representation。归一化与 shard 侧一致（`records.normalize_sequence`：去空格+大写+J→L）。
2. `dedup_train_vs_downstream.py --source <s>` → `data/dedup/reports/<s>.json` + `blocklists/<s>.jsonl`：
   - **tcr**（ots/tcr/tcr_piste）：CDR3β 精确单连接（set 交）。
   - **antibody**（oas/nanobody/neutralization）：CDRH3 精确 + CDRH3 70% linclust 聚类 + 全长 heavy/VHH 95% linclust；grammar shard 无 CDR 区段的源（neutralization）仅全长。
   - **ppi**（ppi/mint_ppi/mint_actions）：全局 40% linclust。
   - 相似度统一用 `mmseqs easy-linclust`（test bank + train 并集聚类；train 落到含 test 成员的簇即判泄漏）——线性时间，可扩到千万级；早期用 `search -s 5.7` 在 250 万 OAS 上 >12min 未完，已弃。
3. `apply_blocklists.py --source <s> [--promote]`：按 blocklist row index 过滤 Arrow shard（迭代序与 extractor 一致，且**硬断言** `n_rows` 与 report 一致防错位）。默认产 `<s>/train_dedup`；`--promote` 时 `train→train_prededup`、`train_dedup→train`（原件留底、可回滚）。
4. `inventory_integrated.py [--write-manifest]`：盘点 8 源整合混合的 raw/deduped 行数、weight、采样占比（=weight/Σweight，与行数无关）。

**逐源泄漏结果**（train 行数 / 泄漏行 / 比例）：

| source | 域 | 键/阈值 | rows | leaked | frac |
|---|---|---|---|---|---|
| oas | ab | CDRH3 精确+70% / 全长95% | 2,486,442 | 1,684 | 0.07% |
| ots | tcr | CDR3β 单连接 | 2,102,715 | 17,301 | 0.82% |
| nanobody | ab | CDRH3 精确+70% / 全长95% | 11,525,884 | 603,398 | 5.24% |
| tcr (processed_v2) | tcr | CDR3β 单连接 | 163,725 | 18,215 | **11.13%** |
| tcr_piste | tcr | CDR3β 单连接 | 284,144 | 63,117 | **22.21%** |
| ppi (STRING 90/90) | ppi | 全局40% | 319,429 | 57,528 | **18.01%** |
| mint_actions | ppi | 全局40% | 9,237,455 | 133,494 | 1.45% |
| neutralization | ab | 全长95%（无CDR区段）| 12,346 | 872 | 7.06% |
| mint_ppi | ppi | 全局40% | (重建中) | — | pending |

结论：TCR 源（tcr/tcr_piste）与 PPI(STRING 90/90) 与基准重叠最重（同源公库），必须去重后再入整合版；抗体/nanobody 泄漏比例低但绝对量不小（nanobody 60 万行）。

**mint_ppi 说明**：`rebuild_mint_training_shards.sh`（v12 binding 重建）完成后再跑其去重（40% 全局），并入整合 manifest。2026-07-08 已完成 promote（81,717,793 行）。

**train↔valid 去重（2026-07-09）**：整合训练提交前，用 `scripts/data/dedup/check_valid_in_train.py` 检查各源 `valid` 记录是否出现在 `train`（whole-record sorted-chain key）。8 源中仅 `tcr_piste` 命中 7 行（0.010% valid keys），经 `apply_validleak.py --promote` 从 train 删除；其余 7 源零重叠。`neutralization` 无 valid shard。工具与 blocklist 落 `data/dedup/reports/valid_in_train_<src>.json`、`data/dedup/blocklists/validleak_<src>.jsonl`。

**SAbDab2 说明（2026-08-03 校正）**：早先“归档无 `abag_split.csv`”的判断来自
一个不完整/损坏的本地下载，已作废。经官方 MD5
`0dbb4cc499e9eb77f14008b232f2c38c` 验证的 Zenodo 20083995 完整
`splits.tar.gz`（876,381,859 bytes）同时包含 `ab_split.csv`、
`ab_split_sd.csv`、`abag_split.csv` 与 `abag_split_sd.csv`。v2 adapter 只读取
paired VH/VL 的 `abag_split.csv`，排除 VHH/VNAR，并只接受 resolved
protein/peptide antigen；得到 6,412 个 antigen-component records，其中 3,363 个
single-polymer core、3,049 个 multi-component aux。官方 `ab_ag_split` 与 cluster
只保留为 provenance；本轮另建 antibody/antigen/joint-hard disjoint split。

**FLAb/AbRank 说明**：本地
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/raw/flab/FLAb/data/binding/AbRank_dataset.csv.zip`
有 342,356 rows。严格 H/L/Ag sequence QC 后 149,782 rows 合法；76,515 rows 同时
满足当前单链 ≤1024，75,483 rows 另有 `fitness`。192,559 个 RBD-escape row 的
`Ag_seq` 是 mutation expression 而不是序列；AlphaSeq/AbCoV 的 full-spike context
大多超过 1024。不能把这些字段当氨基酸或静默截断。可用子集还需规范
affinity/IC50/censor，并按 antibody cluster 与 antigen cluster 的连通分量划分。

**TCR specificity merge 说明**：PISTE、TDC、TEIM 和 legacy
VDJdb/MIRA/McPAS 的 positive `(CDR3β,epitope)` 简单相加为 224,488，exact union
只有 142,456。TEIM 与 legacy exact overlap 40,912。PISTE random-train 若保留
`HLA_type`，284,144 个 triplet 无标签冲突；忽略 HLA 后只有 209,632 个 pair，
其中 13,901 个出现正负冲突。canonical schema 必须保留 HLA sequence scope、
assay、donor 和 source provenance，不能按无 HLA pair 多数投票。

---

## 全长 TCR-pMHC 五实体重建（2026-07-09，ARCH_AUDIT gap2 / FUTURE D1）

**原因**：旧 TCR 源是 CDR3 片段、无全长可变域、无 MHC+B2M，grammar 设计的五实体布局 `<prots> MHC . B2M <protd> <binding> <prots> <pep> PEP <protd> <binding> <prots> <tcr> α . β <protd>` 从未落地。

**产物**：`data/tcr_pmhc_fulllength/`（`scripts/data/build_fulllength_tcr_pmhc.py`，env `protenix_abtcr` + `PATH` 含 Stitchr/thimble）：
- 输入：VDJdb `data/tcr/vdjdb_full.txt` + McPAS `data/tcr/McPAS-TCR.csv`（human、MHC class I）。
- 全长 α/β：Stitchr/thimble（HUMAN IMGT ref，`stitchrdl -s human`）从 V/J 基因 + CDR3 拼全长可变+恒定域。
- 全长 MHC-I 重链：IMGT/HLA `imgt_hla/hla_prot.fasta`（45,762 等位基因，`HLAResolver` 精确→2-field→gene 级回退）；B2M = 成熟人 B2M（UniProt P61769 去信号肽，99 aa）。
- schema：`bioseq.v1`，`chains=[mhc, b2m, peptide, tcr_alpha, tcr_beta]`，`chain_roles=[mhc, mhc, peptide, tcr_alpha, tcr_beta]`（MHC+B2M 同 role `mhc` → 渲染成一个 `<prots> MHC . B2M <protd>` 块），`targets=[3,4]`，`labels.relation="binding"`（curated binder）。

**统计**（`build_stats.json`）：human MHCI 解析 134,526 → 配对可拼 79,040 → thimble OK 78,740 → **下游 CDR3β 去重去掉 15,324** → 唯一 59,093 → train **57,913** / valid 590 / holdout 590。

**历史 shard**：`data/bioseq_grammar_v1/tcr_pmhc_fulllength/{train,valid}`（旧 `build_bioseq_grammar_v1.py --sources tcr_pmhc_fulllength`，新增 `iter_tcr_pmhc_fulllength` reader 保留 roles + relation）。其渲染验证数字仅为历史证据，不属于当前 prepared semantic JSONL 训练线。

**待办（B4）**：并入整合 manifest + 权重后重训。数据布局/方法同步见 `ARCH_AUDIT.md`、`FUTURE_EXPERIMENTS.md` D1。

---

## Downstream format schemas (not owned here)

T1 / Track-A / TCRT5 output schemas and paper-correspondence rules are owned by
`downstream/benchmark/` (`RESULTS.md`, `README.md`, task docs, and
`outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`). Do not copy headline
numbers or protocol tables into this file.


## 2026-09-24 TCR v2 generation outputs

- [TCR sampler](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/tcr_generation.py): v2 conditional JSONL retains `sequences` and adds `protocol` plus one `candidates` row per requested attempt. Unconditional text gains a `.candidates.jsonl` sidecar; use the sidecar to preserve failed attempts.
- Candidates carry `index`, `sequence` (CDR3 or empty on failure), `beta_sequence`, raw `beta_token_ids`, `terminated_by_eos`, `valid_beta_tokens`, `valid`, `failure_reason`, `extraction_method`, and `protocol`. Failures are not silently filtered. `valid` is syntactic benchmark validity, not biological validation.
- Infill rows add `length_condition=reference_window`; CDR3-only coordinates remain junction-relative after framework completion, while paired coordinates remain full-beta-relative. Training record/shard format is unchanged.

## 2026-09-24：AB-AG 官方数据字段预审

已获取的 CoV-UniBind 原始 binding/SPR 产物保持原表不变，路径与哈希见 [获取清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/baselines/_download_manifest.tsv)，字段检查结果见 [AB-AG 执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-antigen)。保留 VH、VL、多个 antigen 表示、target_type、source_doi、assay_name 与 PDB 来源。尚未构建训练 split、未转换 pKD、未将重复上界当作精确测量；具体输入/单位/截断限制见 [AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md) §4/7。

## 2026-09-24T04:07:06.335060+00:00 S²ALM 上游 benchmark 字段核查

实际读取 PALM 作者原 BioMap/14H/14L 表。BioMap 有两链、antigen 和 delta_g，需进一步确定 VH/VL 边界、抗原片段与标签来源；14H/14L 发布入口只读取变化链和 CDR。字段统计和来源见 [S²ALM 文献参照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#s2alm-audit)，协议见 [S²ALM 复现审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/S2ALM_REPRODUCIBILITY_AUDIT.md)。无新 split、无 pKD 转换、无 VHH 评测。

## 2026-09-24T04:48:58.155591+00:00 BioMap 原 CSV 链角色核查

字段语义与 1DQJ 链顺序不一致的官方/PDB 证据见 [BioMap 官方 baseline 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/BIOMAP_BASELINE_REPRODUCIBILITY_AUDIT.md) §4；计数唯一权威为 [RESULTS BioMap 审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-audit)。原始 CSV 未校正、未裁剪、未转换 ΔG 标签。

## 2026-09-24T05:06:03.673383+00:00 BioMap 原 loader split 证据

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_antigen_official_20260924/biomap_seed3/split_membership.csv` 由作者数据加载器在独立 CPU 预检进程中的 `pair_df` 和原 sampler indices 导出；新增 `split`、`shuffled_dataset_index` 两列。该索引属于 loader 打乱后的数据，不是原 CSV 行号。无数据过滤、链交换、标签转换或自定义 split；统计见 [BioMap 官方运行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#biomap-official-run)。

## 2026-09-25 TCR design VDJdb50 Data

- User selected one existing best usable LongAB binding checkpoint: v5 140000, unseen independent AUROC 0.5129; frozen backbone + mean of existing five heads, no retraining. Inputs are provided alpha/beta variable domains and peptide; HLA is ignored. GPU replay of 32 archived features/scores gates source compatibility.
- Only the expanded 50-target whitelist; IgGM2 50, each of ProteinMPNN/ESM-IF1/Rosetta 500, native 50. Chain A=alpha, B=beta verified against PDB and index; no framework substitutions. Equal-target means; no best-of-N comparison.
- Protocol/script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_vdjdb50.py`; artifacts: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925`. `designs.csv` one row per design, `panels.json` deterministic up-to-10 length-matched cross-peptides from these 50 only, separate same-HLA panel, no verified negative labels. CDR novelty uses design train/public sequence reference banks, not another evaluation cohort; OLGA uses unmodified anchored alpha/beta junctions. Public reference includes native targets.

## 2026-09-25 Correction: external released binding model only

User rejected use of our trained v5 140000 and its heads. Prior selection is superseded; task t-20260925203709-4qb26 cancelled before inference. Sequence/Pgen data stay valid. External model selection must check peptide support: local TCRconv fullAB predicts fixed epitope labels, not arbitrary novel peptides. No replacement binding score yet. Scope remains the same 50 targets.

## 2026-09-25T12:48:19.071235+00:00 External TCR-H VDJdb50 evaluation

External TCR-H inputs: `external_tcrh/pairs.csv` has 15,201 distinct Epitope/CDR3B pairs drawn only from the 50-target whitelist/panels. Affinity=0 is an adapter placeholder and not a known label or model feature. Full alpha/beta variable domains stay in designs.csv for identity and sequence metrics but are not consumed by TCR-H. Official train CSVs are used only to validate cached published inference feature selection; external train data are not another evaluated test set.

Script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_external_tcrh.py`. Outputs: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925`.

## 2026-09-25T14:44:49.171625+00:00 epiTCR original-model direct binding evaluation

New isolated external_epitcr_original directory: input.csv and official_predictions.csv contain 1600 rows across same 50 targets; per_design_predictions.csv has design_id, method, prot_id, design_idx, peptide, beta_junction, beta_cdr3, binding_score, predicted_binder, prediction. beta_junction must start C/end F and its [1:-1] exactly matches indexed beta_cdr3; core lengths 10-16 within model 8-19, peptide lengths 8/9/11 within model 8-11. No truncation or missing rows. binder=0 is an unused upstream API placeholder, never true-label evaluation. Primary score files remain isolated from historical TCR-H artifacts. Native cross-peptide control has 472 rows / 49 eligible targets, drawn only from the same 50 peptides.

Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_original/REPORT.md`; script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_external_epitcr.py`.

## 2026-09-25T14:50:26.032622+00:00 epiTCR training-overlap audit

Published original CE train: 321,365 rows; one target epitope NLVGMVATV (vdjdb2025_036) seen, zero exact native CDR3beta/pair matches. Full original source corpus: 3,255,086 rows; five target epitopes seen (ALWALPHAA, NLAPMVATV, NLVAMVATV, NLVGMVATV, NLVPMVASV), three native CDR3beta matches, zero exact pairs. Full corpus includes test data and is not solely training. All 1,550 generated beta CDR3s and their target pairs have zero exact matches in all eight audited banks. NM2025 five retraining folds: zero target-epitope or pair matches, 1–2 native beta-chain matches per fold. Original published train also contains one-edit neighbors of 32 target epitopes and 40 native beta CDR3s, considered separately. Exact checkpoint training-row linkage cannot be proven because original pickle lacks a training manifest. Current model is the original published checkpoint, not a local or NM2025 retraining.

Existing predictions regrouped without inference/retraining/threshold changes: exclude 1 train-seen epitope ->49 targets; exclude 5 corpus-seen ->45; exclude train epitope edit distance <=1 ->17. Every method remains 100% predicted binding in every subgroup, so exact target exposure alone does not resolve the broad positive-call issue. No modification to the 50-target whitelist. Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_original/overlap_audit/REPORT.md`; script `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/audit_epitcr_overlap.py`.

## 2026-09-25T14:55:37.783986+00:00 NM2025 epiTCR first-sample 50-target evaluation complete

New external_epitcr_nm2025_first50 contains selected_first50_designs.csv (250 rows, 50/method), first50_input.csv, first50_predictions.csv with 5 fold scores/ensemble score/predicted_binder, full-cohort companion inputs/predictions, checkpoint/train provenance manifest, training_overlap_summary.csv and per_design_training_overlap.csv. Baseline indices all 1, native and IgGM2 0. Affinity=0 is upstream API placeholder only. first50_sequence_summary.csv recomputes CDR AAR, novelty and zero Pgen fractions on exactly the selected rows. Prior score files untouched; root report points to current model/sample protocol.

Report: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/outputs/tcr_design_quality_vdjdb50_20260925/external_epitcr_nm2025_first50/REPORT.md`; script: `/vepfs-mlp2/c20250601/251105016/project/flow_antibody_tcr/scripts/tcr_quality/evaluate_nm_epitcr_first50.py`.

## 2026-09-27 05:34Z AB fixed-v2 Fv boundary manifest

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/ab_fixed_v2_fv_manifest.json` contains raw data SHA256, original chain sequences, ANARCI chain types and zero-based half-open `[start,end)` boundaries. Rebuild with `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/prepare_ab_fv_input.py` in the flow environment. Original CSV/FASTA files are unchanged. Boundary extraction and full m396 flank validation must pass before publication. At runtime unknown overflow, over-capacity Fv and outside-Fv mutations are errors, never silent truncation.

## 2026-09-27 AB-AG sequence v1

新数据根 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/ab_antigen_sequence_v1`；JSONL 使用稳定 id、antigen/heavy/light、来源及标签；CHIMERA 参考编号/接触映射仅用于独立评分。数量与逐行排除权威：`build_audit.json`；官方原始下载及哈希：`raw/acquisition.json`。训练精确/近同源审计输出 `audit/training`；SPR pKD 单位与检测限仍待核。

2026-09-27 final AB–AG data audit: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/ab_antigen_sequence_v1/build_audit.json`; exact overlap refreshed for all219995 accepted rows (four subsets combined), with zero unnumbered benchmark H3. Upstream full-source overlap remains unknown. See task manual for MMseqs direction and heuristic limits.

### 2026-09-27 AB–Ag 接口数据契约

四子集 JSONL 通过稳定 ID 关联预测；interface_manifest.json 锁定数据指纹、代码、模型身份及采样配置。生成请求不含参考抗体，失败输出保留并说明原因；缺失预测默认报错。SPR pKD 未核验不进入回归。完整字段及 CLI 见 [ab_antigen/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/README.md)。

### 2026-09-30 AB–Ag 分批执行记录

本轮batch2改64续跑，以每项输出目录`batch_execution_change.json`记录旧/新批量、已有完整行数与前缀SHA256、等价性验证路径；原provenance不覆写。证据位于`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000_local_20260928`。
# 2026-09-30 v3 observation/anchor contract (implementation underway)

The v3 input must preserve observed junction C and F/W even when legacy segmentation assigns them to FR3/FR4. Remove only positions explicitly marked synthetic, never all framework-labelled positions. Source junction/core conventions are provenance, not a string-prefix heuristic. Irrecoverable core-only inputs remain core fragments, never synthetic C/F labels. Each target fragment/chain has one `<protd>` label; attended trailing MASK capacity is reference-length independent and excluded from loss. Existing v2 data and vocabulary remain unchanged.

- v3 `GenerationRequest` contains observed condition chains plus `(role, scope)` targets; it carries no reference sequence/length and inserts no dummy amino acid. It is accepted only by A4 generation mode. Training still reads versioned semantic records.
- v7 derived coverage evidence is bound to registry, ANARCI annotation and effective-bank SHA256. It additionally binds the legacy direct-read TCR holdout text. Export clustering maps must match both inventory and effective-bank hashes; emitted data are independently rescanned before the certificate is written.

- Legacy nanobody CDR-infilling has both masked `seq` and native `label`. Freezing new banks must protect native full references as well as source CDR3 fields; stripping gap characters from a masked input does not recover its native heavy chain. Existing immutable banks are not silently rewritten. An apparent extra native reference was the repeated CSV header (`sequence`, `sequence`, `CDR1`, `CDR2`, `CDR3`), not a receptor. Verification counts that exact non-data artifact separately; all 2,846 actual native full references are already protected.

- Antigen exact-containment filtering now requires >=80% coverage of both the candidate and protected reference. Short antigen fragments must not block all longer proteins sharing a few residues. The first unbounded-containment export is aborted; the current version is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2` with protocol revision `v7.2_bidirectional_containment`.

- 2026-09-30T15:46:19Z 状态核对：r2导出仍在运行，尚无最终数量或通过证书；264条benchmark H3覆盖问题仍未解决。审计阶段状态、失败/更正和最终保留/剔除数量需同步到 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，并链接对应manifest、清理报告和证书；不得用“clean”目录名称表示审核通过。

- 2026-09-30T17:23:11Z r2导出数量已确认：train6,849,243 / valid52,572，总计6,901,815，剔除479,684；依据 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/dataset_manifest.json` 和同目录cleaning_report.json。这是导出数量，不是审计通过声明。独立复核从头重启，保护集合并集改为每次复核构建一次，membership语义/哈希检查不变；264条benchmark H3覆盖问题仍待处理。


<a id="tcr-exposure-20260930"></a>
## 2026-09-30 新增TCR下游暴露审计

**状态：全量CPU审计完成，明确存在重叠；新数据清理尚未执行，未发新scope清洁证书。** 旧 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2/audit_certificate.json` 只证明旧冻结操作性规则下的扫描结果，仍有H3覆盖pending；不包含新增UpTCR。

实现：[scripts/data/audit_tcr_exposure.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/audit_tcr_exposure.py)；版本化产物 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2`。registry.json逐文件记录路径/SHA、用途、行数、字段异常及来源语义；banks.json保留每任务/子划分独立集合。官方下游表不改写。UpTCR原样协议见 [downstream/tasks/TCR_T1_BINDING.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md)；T3见 [downstream/tasks/TCR_T3_REPRESENTATION.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md)。

保护范围：UpTCR majority/few-shot/zero-shot全部train/valid/test（foundation侧排除，合法下游适配另算）；T3 broad train支持池/test查询池；paper6 target_binders与background_pool整个采样宇宙；T2两任务；既有IMMREP binding test及NM2025 test。其他旧AB/TCR保护范围继续由v7管理，本次扩展不替换它们。不能用文件名含train/background就跳过；background的`__background__`是类别标记，不是肽序列，其原始表位来源仍需上游追溯。

审计键：CDR3α/β core、可观察完整FVα/FVβ exact、αβ、α/β–epitope、αβ–epitope和有可比较MHC序列的β–epitope–MHC联合键。联合匹配必须来自同一预训练记录，不能把两个独立序列命中拼成配对命中。HLA allele字符串不猜测映射为伪序列；共享HLA本身不列为违规。

相似度：CDR3α/β和短peptide全量RapidFuzz normalized Levenshtein ≥0.80，即`1-edit_distance/max_length`，含exact且蕴含双边长度比≥0.80；不是MMseqs局部identity或相同功能证明。使用v6所有唯一序列的hash绑定inventory，并独立扫描v6/v7实际train/valid，按任务、foundation来源/split统计，输出逐行reason位图及零基行号。表位还单独检查是否完整出现在AB–Ag训练记录的antigen链中；命中仅证明片段暴露，不声称整个parent antigen相同。FV同源、源蛋白ID/同源性未作已完成声明。

CDR3规范化仅作用比较键：IMMREP broad/binding已经是core，不能再次剥离真实端点C/F；UpTCR/VDJdb/NM2025按junction来源规则处理并登记非典型端点。训练模型实际观察到的C/F/W和原CSV不改动。首个探索目录已标ABORTED，不能引用其混合字段口径结果。

新训练gate：[scripts/data/tcr_scope_gate.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/tcr_scope_gate.py) 要求数据目录额外提供`downstream_scope_certificate.json`，scope_version=`tcr_all_support_query_uptcr_v1`、passed/无pending/overlap=0、dataset manifest SHA、当前每个support/query输入SHA及证据文件SHA全部匹配。现阶段不签发该证书；不能改一个status绕过真实清理与独立复核。

复现：激活pllm，在仓库根执行`python scripts/data/audit_tcr_exposure.py --stage freeze`，随后`--stage near --workers 4`；完成后对明确的数据目录分别执行`--stage scan --data-dir /绝对路径`。不加载checkpoint，不训练、不使用GPU。

### 独立表位精确核对（先行快照；完整同split口径见下方最终结果）

证据：[v6唯一序列交集](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/inventory_exact.json)、[v7真实train/valid peptide链独立扫描](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/epitope_direct_scan_v7.json)。下表单位为**独立epitope种类**；v6列是train+valid并集，v7列是train，分母是对应test/pool独立表位数，不能据此声称模型优化器实际访问过每一行。

| 下游集合 | v6 train+valid精确命中 | v7 r2 train精确命中 |
|---|---:|---:|
| UpTCR majority test | 31 / 32 | 31 / 32 |
| UpTCR few-shot test | 205 / 216 | 198 / 216 |
| UpTCR zero-shot test | 129 / 142 | 125 / 142 |
| T3 broad support train | 24 / 24 | 23 / 24 |
| T3 paper6目标表位 | 6 / 6 | 6 / 6 |

因此v7在旧银行中overlap=0并不能支持新范围下的epitope-unseen结论。后续完整扫描按相同foundation split分别呈现受体/表位/联合键，避免混用口径。源CSV中的序列对命中不自动等于正负标签一致；联合键不包含label。UpTCR原9表的CDR3字符串与对应FV包含关系另见 [字段交叉检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/field_provenance_review.json)，该检查不是IMGT生物学注释证明，非典型端点保留为review。

测试：pllm，RapidFuzz3.14.5，CPU4线程，新增审计+scope门禁+旧v7+v3共40项通过（已有PyTorch autocast弃用warning，无失败）。目录r2的near_run.json记录库版本、完整inventorySQLite SHA、输入registry SHA及确定性阈值。没有训练/随机抽样，seed不适用。

### 2026-09-30T18:28:30+00:00 全量审计完成：明确存在重叠，未执行新版本清理

产物：[汇总](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/summary.json)、[v6逐来源/逐任务报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/immune_v6_binding_only/report.json)、[v7逐来源/逐任务报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/immune_v7_clean_r2/report.json)、[拟清理清单契约](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/cleanup_plan.json)。两套实际prepared数据全量扫描完成，各分片行数/SHA一致；结束后再次校验24个冻结cohort的374,373行下游输入，SHA未变化。两个row_hits.jsonl均重新校验SHA；独立raw peptide扫描与语义记录扫描的train/valid结果全部一致。

下表统一对比**foundation train**，单位为对应下游集合中不同的比较键，分母为该集合全部不同键。β为CDR3β core，表位列只计算直接peptide链精确命中；antigen中片段命中另计。联合β–epitope命中来自同一训练记录，不是分别命中后拼接，且未按正负标签区分。

| 下游集合 | v6 train β精确 | v7 train β精确 | v6 train epitope精确 | v7 train epitope精确 | v7 train β–epitope联合精确 |
|---|---:|---:|---:|---:|---:|
| majority test | 7,016 / 15,315 | 4,871 / 15,315 | 31 / 32 | 31 / 32 | 933 / 23,968 |
| few-shot test | 1,835 / 3,479 | 1,224 / 3,479 | 205 / 216 | 198 / 216 | 329 / 3,995 |
| zero-shot test | 3,319 / 7,024 | 2,169 / 7,024 | 129 / 142 | 125 / 142 | 214 / 8,482 |

T3补充：按本轮规范化比较键，paper6 target_binders在v7 train的α/β精确命中均为0，但6/6目标表位仍出现；其background_pool有7,945个α core、4,653个β core、2,215个αβ组合精确命中。broad support train有1,386个α core、895个β core精确命中；broad query test仍有36个β core、2个β–epitope联合键命中。不能以单一receptor去重结论推断epitope或所有支持/查询池都未暴露。

跨任务：v7 train ASD antibody中35,798行的antigen包含保护表位片段；其中33,024行对UpTCR zero-shot test表位命中。它是抗原片段暴露，不等于相同TCR-pMHC训练样本。对zero-shot test，v7 train另有50,031行仅命中表位/片段而没有本轮阈值下的受体命中；因此单查TCR必然漏掉这部分。

**近似规则影响（不是“真实泄漏比例”）**：保护全部24cohort，以任一单链core/epitope全长编辑相似度≥80%、完整FV/联合键exact或epitope-in-antigen为并集，v6 train+valid标记2,573,820 / 7,381,499行；v7标记2,119,972 / 6,901,815行。跨cohort和原因均有交集，不可相加。下表是v7按该保守规则的候选排除量，尚未导出/删除。

| foundation来源/split | 当前行数 | 候选标记行数 | 若全部排除后的行数 |
|---|---:|---:|---:|
| train/asd_antibody | 203,189 | 35,798 | 167,391 |
| train/oas | 2,472,502 | 0 | 2,472,502 |
| train/ots | 1,701,193 | 1,623,941 | 77,252 |
| train/tcr_native | 22,508 | 21,862 | 646 |
| train/tcr_papers | 437,770 | 356,477 | 81,293 |
| train/tcr_repertoire | 2,001,488 | 65,475 | 1,936,013 |
| train/trait | 10,593 | 9,742 | 851 |
| valid/asd_antibody | 960 | 365 | 595 |
| valid/oas | 24,857 | 0 | 24,857 |
| valid/ots | 3,854 | 3,324 | 530 |
| valid/tcr_native | 106 | 102 | 4 |
| valid/tcr_papers | 2,978 | 2,246 | 732 |
| valid/tcr_repertoire | 19,771 | 598 | 19,173 |
| valid/trait | 46 | 42 | 4 |

**不能直接把候选清单当最终数据方案**：任一链80%近似会标记OTS train的大多数记录；这与同一样本精确重复不同。若全部剔除，tcr_native和trait的valid各仅余4行，需重新审视验证集规模/家族隔离，避免用这样的小集合评估架构优劣。新数据导出后还要独立复核并签发scope证书，本轮不签发、不修改旧certificate、不追溯声称历史checkpoint变干净。

字段/覆盖边界：v6有15处α缺失/非标准及3处β比较键缺失/非标准，v7剩3处β比较键缺失/非标准；MHC非标准不可比较字段分别59/54处，不影响其他键的独立匹配。benchmark非典型junction端点、paper6 background原始表位、上游parent antigen身份/同源性、完整FV同源性仍不能视作已解决；54处MHC不等同于54条坏TCR。标准HLA共享不是排除规则。`near_*.jsonl`中的witness是全局最佳参考例，不是每cohort的专属witness；cohort位图来自全部达到阈值的参考比较。

资源/复现：near PID1818190，4 CPU线程；CDR3α990,345个、β3,918,908个、peptide3,344个唯一候选与冻结库全量比较，命中候选分别777,294 / 1,182,453 / 1,617。v6/v7扫描PID1834691/1834690，各单进程，耗时580.40s/534.17s，无GPU/权重加载。冻结registry SHA=`2465629c9a7dd233377dc7e3e3361dd24af8ec6ce5eb50def39a5e48b5ca41ff`；主脚本SHA=`55b5a102d7164824b176c89b737a3bfd4c28e7625b9cbfd0c326d5ebe75b72cd`；源代码与环境快照见 [source_snapshot](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/source_snapshot/manifest.json)，最终独立unit fixture记录见 [final_tests](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/source_snapshot/final_tests/manifest.json)。

结论边界：当前已证明训练**数据池**存在序列、表位及部分联合样本暴露；未读取逐optimizer-step采样日志，不声称每个命中都一定被某checkpoint访问。作者的UpTCR zero-shot指其发布划分，不能直接外推为本foundation预训练也未见。原样benchmark仍按既定协议执行并披露暴露，排行榜数值本轮不改。


## 2026-09-30 手动实验包的数据边界

[queue012 实验包说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_explore_queue012/README.md) 的合成 20-step 组（A0–A4）**已全部完成/Success**，只使用代码内人工构造的双链 `BioSeqRecord`，source=synthetic_optimization_only，不读取预训练语料或下游 test/few-shot/epitope。C/F/W 原始残基通过既有 collator 保留，未改变真实数据格式或执行数据删除。

真实数据组仍指向待审计的 immune_v7_clean_r2 候选目录；YAML 的存在不表示它已 clean。当前 assert_ready 仍报 audit incomplete；最终导出需通过已有与新增下游范围的全部证书，并为 A0–A4 提供同一份数据。此前 cleanup_plan 的 not_applied 状态不因本轮配置准备而改变。


## 2026-09-30 扩大架构研究的数据/输入边界

[研究与方法记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md) 的新suite使用既有A4语法与真实残基，未改变prepared JSONL或原始benchmark。R05/R11的register是模型内部向量，不属于氨基酸词表、不添加标签、不改变输出序列长度。R06卷积严格屏蔽跨链与padding；R07条件只来自可见MASK和容量槽；R08辅助池的观测残基mask仅用于训练loss，不传给denoise。

第一组人工TCR/AB/nanobody记录只作合成诊断，没有接入真实epitope/test/few-shot。第二组默认使用最新MHC输入更新候选immune_v7_mhc_sequence_v1，但实际检查发现其audit_certificate.json缺失，正确拒绝。MHC输入更新不等于完成泄漏清理；最终数据必须以新manifest和全部范围证书重新认证。没有更改旧去重清单/cleanup_plan的状态。


## 2026-09-30 研究训练预算更新的数据口径

[研究记录及验证预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)的新1000/5000步方案不改变prepared行、C/F/W残基、来源或去重门禁；只是增加训练和验证预算。256,000/1,280,000为global256下累计样本呈现次数，来源均衡/重复采样时不等于独立记录数。验证仍仅取训练数据版本自带valid，每来源固定seed42至多256条，按记录内容生成固定25/50/75/100%遮蔽，并不引入下游test或few-shot query。用于选择方法的valid范围必须继续满足既有审计与隔离规则。


## 2026-09-30 八卡实验数据门禁说明

资源从六卡改为八卡不会改变数据格式或解除泄漏检查。当前immune_v7_mhc_sequence_v1仍缺audit_certificate.json及downstream_scope_certificate.json；初始12个任务全部合成profile/工程检查。真实1000/5000步仍须assert_ready通过，不能将测试/few-shot或epitope重叠审计未清理的数据直接训练。

运行与控制说明：[八卡工作池](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/README.md)；实际结果：[持续记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)。


## 2026-09-30T20:19:38.413413+00:00 架构v2诊断边界

未修改真实语料格式、C/F/W边界或数据划分。新诊断评估固定25/50/75/100% mask，仅selected位置计分；不将labels/residue_mask/terminal_mask输入denoise。全mask下四条toy TCR不可区分，AA上限56.10%仅指当前人工toy。真实训练仍要求两份去重/下游范围证书，不以合成passed替代数据清理。

详细研究、测试和运行证据：[架构扩展v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。


## 2026-09-30T20:53:52.568809+00:00 初步训练前只读审计的新发现

两候选目前均不能通过assert_ready。92范围条目SHA全部匹配；264 H3待定项已定位，133为作者标签明确的轻链、35为VNAR，另有短loop与不完整/未知观测，不能统一作为普通VHH缺H3处理。角色纠正必须补相应light/VNAR全链及已知loop保护。expanded清理export与双证书签发尚待实现；没有改原prepared数据、benchmark或过滤阈值。精确/近邻影响量与来源验证集不足问题已单列。

[持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md) · [270基线审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA270_BASELINE_AUDIT.md) · [100步结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R12_R19_100STEP_RESULTS.md) · [数据准备状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_DATA_READINESS.md)


## 2026-09-30T21:17:56.103865+00:00 观测角色核验与人工诊断格式

264项H3缺口中184有明确证据：133 light、48源空H3完整观测、1止于IMGT81片段、2部分H3。80项仍pending：35 VNAR未知loop、26未编号H尾巴、17无domain、2源构造冲突。角色保护方案保留旧规则并补light/VNAR全链、L3和片段覆盖，尚未应用到prepared或发证。

新增人工manifest仅程序重建576/72样本、记录source/generator/content SHA和18分层，模型features不接收分层metadata或labels。它是工程诊断数据，不能用来替代真实清理证书或证明下游泛化。

[Refined配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md) · [角色观测审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_V7_OBSERVATION_ROLES.md)


## 2026-09-30T21:25:58.529062+00:00 八项500步人工诊断已开始

R20–R27的25步广输入检查全部succeeded，累计28成功/0失败。各项约718秒，四档mean loss为1.768–1.908、AA约30.06%–36.38%；end accuracy仍0，全mask end只占132/17996≈0.73%，须独立观察，不能仅看总loss。局部/全局方法在周期8人工任务上的优势不等于生物学优势。

已实际执行经74项测试验收的入队助手 synthetic500 --apply，8项均被worker领取运行。新500步initial的18组指标与同方法25步initial完全一致，fixture、39源码SHA、global256也一致，确认为同协议重新初始化。500步是更广人工heldout诊断，不是初步真实语料训练。按25步实耗粗估约4小时，准确耗时以运行结果为准。

[启动证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/synthetic500_started.json) · [配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md) · [持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。目标保持active，真实数据清理与1000/5000步尚未完成。

数据覆盖仍在推进：替代编号工具给出了VNAR候选，但与原ANARCI对部分“源空H3”案例存在分歧；184项只能表示来源/原编号已有解释，不能自动当成184项生物学缺失已确认或认证覆盖已完成。保留pending并量化对未知loop使用训练已知H3匹配整条benchmark观测串的保守附加筛查。新expanded exporter已开始实现与CPU测试，尚未扫描/导出全量或签证。

## 2026-10-01 typed诊断、监督上线与pilot生命周期验收

本轮没有修改prepared、训练/验证划分、序列注释或benchmark。候选仍training_ready=false；隔离执行器对实际候选调用原assert_ready，因缺audit_certificate在第一门禁拒绝，未创建输出、未进入GPU/模型。自动检查中断的注释分支未恢复。

新增的是未来运行产物合同：execution_request.json、execution_provenance.json与checkpoint/execution_provenance.json。checkpoint侧车绑定run_provenance、research_config、immune_config、vocabulary、trainer_state、model.safetensors、optimizer、scheduler、rng_state九文件字节；累计续训绑定固定验证与数据/源证据，停止步以外不得变化。operator诊断则为独立kind和逐case状态，不含训练global_step。[IMMUNE_CERTIFIED_PILOT_EXECUTION](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CERTIFIED_PILOT_EXECUTION.md)与[IMMUNE_TYPED_DIAGNOSTIC_ADAPTER](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TYPED_DIAGNOSTIC_ADAPTER.md)记录格式；这些软件合同不会签发数据覆盖证书。

## 2026-10-01 00:47Z 运行源码归档和待部署补丁格式

本轮仅新增归档计划与入口部署证据，prepared、划分、真实序列注释、benchmark和双证书状态不变。未来R20 release_manifest保留schema 1的source_root/frozen_root、measured_source_sha256和完整copied_files_sha256，另记录原任务/summary/profile证据SHA与依赖差异；现阶段只有dry-run，快照目录未创建。9个profile清单外实际导入文件与R12同字节，补充指纹不等于历史GPU认证。[归档合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_REFINED_RELEASE_ARCHIVE.md)。

待部署routing.patch和deployment_manifest保存旧3文件前后SHA、dispatcher SHA及固定28模型+1诊断路由。临时副本38项CPU测试与真实历史快照只读校验没有改写旧结果：R00只有32项历史measured哈希，不能将其465个报告复制文件全称为哈希认证；R12的474项copied哈希可完整核验。[路由证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DIAGNOSTIC_DISPATCH_STAGING.md)。

## 2026-10-01 梯度审计产物与候选元数据复查

新增HF micro-mean审计只使用8条人工数值记录，保存每次实际microbatch的record ID、梯度/参数误差、框架配置及来源SHA；不是语料训练输出。旧7份源码和两环境修正前报告有独立before_fix_manifest，修正后另存pllm_after_fix.json和worker_after_fix.json，不覆写旧数值。每环境90项回归及前后各5次单步实验已完成，自有1358个临时文件已清理。[产物与限界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

只读复查真实候选dataset_manifest.json，SHA仍为601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce，training_ready=false；audit_certificate.json和downstream_scope_certificate.json仍不存在。没有读取序列、修改prepared/split或恢复停止的注释分支。派生runtime发现的两处scope资源路径将单独显式绑定，原证书路径和内容校验必须保留；该运行包工作不构成数据认证。

## 2026-10-01 500终态及来源归档产物

[500步终态汇总](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r20_r27_stage500_summary.json)绑定8项原始summary/COMPLETE/worker/profile及绘图来源，保留四mask、18分层、训练probe、成本和末100步描述统计。END仍指真端点召回，不改名为首次终止/长度准确率。

[release_manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/release_manifest.json)包含524 copied源/配置、39 measured源、8 terminal_job_ids及archive_evidence_sha256。额外支持源是归档时字节证明，不是补签GPU profile。此次只修改8项来源映射，未修改候选数据、split、旧实验指标或证书。

## 2026-10-01T01:49:19+00:00 真实阶段准入复核

本轮只读candidate dataset_manifest与两证书存在性，training_ready=false，两证书仍缺。未改split/selection/原数据。后续真实1000与累计5000继续使用固定seed42训练内验证索引并呈现每source计数；首次END/长度与真端点召回分别记录。工程GPU验收不签发或替代数据审核证明。

## 2026-10-02T06:55:41.057373+00:00 人工学习manifest复用记录

固定目录 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/artificial_fixture_seed42](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/artificial_fixture_seed42) 仅包含 [synthetic_fixture_manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/artificial_fixture_seed42/synthetic_fixture_manifest.json)；沿用原immune.synthetic_fixture.v1协议，seed42/train576/valid72/full256/fragment64/max_length2048。原生成器重建人工记录且逐字段/hash核对，序列只在内存，不产生jsonl/shards。此manifest仅用于synthetic500，不是正式数据manifest/审核证书，不授权真实训练。V7 prepared格式和数据布局未改，真实数据未读取。

## 2026-10-02T19:03:26.562307+00:00 探索子集格式准备

抽样入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/sample_v7_architecture_pilot.py 保留原行字节/原split，输出train.jsonl、valid.jsonl、provenance.jsonl及sample_manifest.json；按source分层，保存parent元数据SHA与实际读取shard的SHA/count，子集formal_training_ready=false（探索描述，不冒充正式证书）。预算100000/1024，实际抽样尚未执行。

## 2026-10-02T19:08:18.446121+00:00 十万条V7抽样完成，完整消融图计划登记

实际抽样train100000/valid1024成功，原行/split保持；source分层、SHA与count已核实，样本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample/sample_manifest.json。尚未启动GPU训练。完成五份原始论文/官方指南查阅，消融设计为四项初筛、三因素八条件组合与逐项去除，另列固定门控/同等调参预算对照；最终六面板图及日志字段见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/ARCHITECTURE_ABLATION_FIGURE_PLAN_20261002.md。已确定首批实验不等待组合实现。

## 2026-10-02T19:21:41.759165+00:00 V7十万条实际子集布局通过

已真实抽取train100000、valid1024；原split/原行字节保持。sample manifest SHA a437dcf83069cd773058c936edae132043ca16c9c5cb5e6d44d2d84dffa89bad；实际读取的源shard均完成原清单SHA及count核对。逐条调用现有A4布局共101024行，错误0，最长train1281/valid1200 tokens，未截断或丢样。manifest /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample/sample_manifest.json；布局证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/sample_layout_preflight.json。

|来源|train|valid|
|---|---:|---:|
|asd_antibody|2633|14|
|oas|38566|512|
|ots|26532|80|
|tcr_native|192|2|
|tcr_papers|1786|20|
|tcr_repertoire|30254|395|
|trait|37|1|

来源宏平均必须同时报告以上分母；trait仅1条、tcr_native仅2条验证，不足以单独得出稳定来源收益结论。该子集未生成新的正式审核证书；训练入口绑定已通过限定范围审核的parent，formal_training_ready=false仅表示本子集用途为探索。

## 2026-10-02T19:52:22.715860+00:00 grammar/context loss核验及context数据增强计划

冻结A4的101024行loss资格核验通过：train/valid context位置697220/5852，eligible与有效label均0；关系监督资格4648/37，实际标签全部binding正例。用户明确要求保留关系监督并增加context，计划原100000追加20000既有context监督记录、原valid不变；尚在实现/未抽样/未启动新组，不修改当前八项。完整epoch索引与第1000步断点接续控制通过，3125步恰好覆盖100000索引。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_GRAMMAR_CONTEXT_LOSS_20261002.md。

多图/多帧新增阅读与三份固定作者源码已归档；优先研究分段全长/global交替及既有延后融合，其次target/context cross-attention，保持context无序列CE与明确关系监督。新方法未实现/执行；研究取舍见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/CV_MULTIVIEW_VIDEO_ARCHITECTURE_20261002.md。

## 2026-10-02T19:56:12.027484+00:00 nonbinding缺失的配方原因已定位

parent manifest filter_names含recipe.binding_only，recipe.drop_nonbinding_sources显式列出asd_antibody/trait/tcr_native/tcr_papers。当前subset监督标签全部binding不是抽样偶然性的充分解释，而是与该配方一致；全parent统计将在追加扫描给出。用户已被告知，并收到可选目标澄清（后续可靠负例正负判别，或本轮仅binding正例）；等待该偏好不阻塞现有八卡epoch及20k已审核context正例增补。不会把未观测配对/unknown造为nonbinding。

## 2026-10-02T19:58:16.131168+00:00 用户确认本轮只学习binding正例

异步目标澄清已明确选择“本轮只学习已有binding正例，暂不做正负判别”。据此追加候选只取原V7明确binding=True监督的context记录，保留关系token监督，不引入negative、不生成负例、不请求新负例来源。缺nonbinding不再作为本轮阻断，也不把正例学习描述为正负区分能力。当前八卡样本/源码/3125步不变；独立追加20000正例context仍在实现中。

## 2026-10-02T20:17:42.129532+00:00 context正例增补入口验收通过，准备一次实际扫描

新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/augment_v7_context_subset.py 复用原抽样器的SHA、stat和来源配额逻辑；人工fixture通过确定性、原train前缀/valid字节保留、正例与原train/valid逐行来源及原字节排重、父分片变动拒绝。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/context_augmentation_self_test.json。仅从现有父train追加20000条明确binding且relation_supervised=True的context记录，不生成标签/负例，不恢复注释分支。实际增补尚未完成；输出独立sample_context_plus20k，原八项与冻结源码保持。

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

## 2026-10-02T20:44:10.457859+00:00 新120k组合架构与context指标入口完成CPU验收

上一Goal回合为progress：独立追加20k context正例并完成120k数据加载及新增A4布局验收。本回合继续原Goal，八p01实时worker心跳/优化步在推进，不重启或重复提交。新增Factorial 8组合+3对照，CPU24项通过；新3750入口2步人工检查和关系指标2项通过，自有CPU checkpoint已清理。11份方法YAML/22个两seed初筛格已登记，但尚未冻结/发布/入队，第三seed确认仍待完成。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_FACTORIAL_ABLATION_20261002.md。

正式world8缺口已只读定位：现pool每worker只见一GPU，p01是8独立模型；旧V4仅synthetic/R21/短步数，不能换data_dir直接作正式训练。新driver正复用原Trainer及八rank通用checkpoint接口，仍待实际GPU联合更新/保存恢复验收，不借旧失败或CPU记录宣称成功。当前final V7数据审核有效；所有原停线与源/数据门禁保持。

## 2026-10-02T21:07:58.922239+00:00 正式入口隔离审核首次真实执行通过，p02作图已登记

冻结正式模型/Trainer与真实审核资源的分离接口已首次执行：原helper在python -I -B、CUDA隐藏的子进程调用原assert_ready，真实校验final V7所有分片及现行scope/evidence，54.28秒返回passed；parent清单与两证书SHA前后不变，实际门禁模块39份。没有创建审核证书或改ready、恢复注释或启动模型。源清单1137项、SHA 2c5a2a8a5ba9841dbffc569f1318a09d2497eed39a9ccc7ba5ed8e99104172b1；事实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.stdout.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.json。此处是新隔离执行接口的首次实际核验，非重复签证。

root已读取原工具stdout归档，正式driver新增协议测试9/9、3AST通过；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/receipt.json。正式入口源已冻结，但launcher还没接通，联合NCCL更新、实参同步、保存恢复实际GPU证据仍缺，formal_training_started=false。正式长程预算需结合原资源剩余时间设置可恢复阶段；不允许因全部小实验排完便把8份独立模型视为world8联合训练成功。

p02作图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 新4项纯数学/结构控制通过；首次读真实22项账本输出JSON及4CSV，awaiting_shared_evaluation、missing22、figures=[]。没有用假效果值画图。数据列包含source/context分母、关系CE/accuracy及正交G/S/L对比，完整图页的真实渲染待共同评估出现；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_ABLATION_FIGURES_20261002.md。原p01监督完成后续看p02，至多12小时或22项全终态，无新增OS监听/平台任务。


## 2026-10-06T14:35:22.374740+00:00 后续消融调度catalog v2（不改变序列格式）

私有catalog schema由v7.metrics_only.pool_catalog.v1独立升级为v7.metrics_only.pool_catalog.v2，原job、study、数据/源码和22格claims schema不变。formal_completion/formal_stage绑定epoch2的50082完成和实际末段；新增generation_completion、generation_release、generation_package三项绝对路径＋SHA引用。generation_package必须为原冻结包，逐项检查原输入/退出/评分文件；不制造缺失完成回执。ledger的父目录固定为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_dispatch_v1，以复用本地已存在锁。当前无真实v2 catalog/ledger/票据发布，检查用人工临时文件已清理；实现 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_v2_dispatch.py、/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_post_generation_acceptance.py。


## 2026-10-06T14:49:24.133710+00:00 后续消融发布产物与旧未启动票据归档

未来实际发布目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1：publication_started.json记录原件与时限；inventory_report.json为当时原22格实测；catalog.json为v2任务来源；claims位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_dispatch_v1/claims_post_generation_pool_v1.json；mixed_report_selection.json只替换原pending格；release.json记录实际公共入口；enqueue_receipt.json先publishing逐项写票据，释放共享锁后才成为enqueued。worker会检查此状态和catalog的path/SHA后进入原冻结训练。当前只有准备包/源码/检查证据，上述实际发布产物均未创建。

旧终止pool的3张从未启动票据原字节归档于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_adapter_v2/old_unstarted_tickets_archive：intent.json保留原位置/SHA、平台终态和worker状态，receipt.json记实际移动。原pending清空，未添加results/COMPLETE或声称这些任务执行过；原catalog作为历史不改。序列格式及审核证书未改变。
