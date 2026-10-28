> 2026-10-28 探索 index `arch70m.data.v1`：原 prepared JSONL 不变，新建 10% 分层 byte offsets；完整原 valid/probe valid 受体与 AB specificity 完整/Fv 组合从预训练排除。screen/confirm 受体 hash 分区，family 词表与 V8 输入布局不变。保护仅精确组合，不冒称同源性隔离。数据路径、SHA 和过滤审计见 [架构探索实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md)。

# 当前训练数据与格式

2026-10-28 用户确认架构探索的数据格式冻结：保持当前 V8 prepared JSONL 记录及原 reader/collator 输入约定。核对探索索引的 109 个训练分片，路径与内容 SHA 均与父索引一致；10% 子集仅另存原记录 offsets。`arch70m.data.v1` 标识实验采样清单，未替换原训练记录 schema。原字段、链序列/顺序、角色、元数据、relation/family 标签、词表/特殊 token 与固定链布局保持。评估的来源/受体分组信息仅为外围记账，不写回原记录。本次仅记录约束，未重建数据、修改训练代码或重启作业。

2026-10-27 1B测试新增物化迁移绑定 `immune.bert_mlm.materialized_v8.v1`：独立index/binding/manifest位于 `/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/data`，全部运行路径使用新存储。原plan/index/binding/manifest/config/reader验收文件只作为归档证据，不读取旧路径。门禁比较原证据哈希链、121分片集合/顺序/offset、开发集、reader/objective可执行AST、family词表和布局，允许四个模型尺寸字段变化，再执行原训练器全量分片SHA。6,604,459条记录及划分不变；未改原数据或历史绑定。CPU另抽读全部分片首尾242条。历史父目录未完整迁移，故不冒称原符号链接递归门禁在新机器可运行。[详情](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

> 2026-10-27 ownkaifa02 已同步 V8 four-relations r2、Diffusion V7 及现役下游输入；文件字节保持源 SHA-256，跨目录数据链接物化为普通文件。未改划分、标签或评分口径；index/binding 路径迁移尚待验收。详见 [OWNKAIFA02_TRANSFER_20261027.md](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/OWNKAIFA02_TRANSFER_20261027.md)。


2026-10-27 family_labels规则v3：所有关系状态下的实际观测受体链均参与family分类；relation不再决定是否置-100。仅缺失/无观测残基的槽忽略，缺失/冲突注释或词表外类别归unknown。全121分片重扫，train目标11055744（unknown2809682），valid106343（unknown64545），183类、验证有效类均被训练词表覆盖。原序列/split不变，nonbinding序列MLM仍为0。[全部关系family审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_all_relations_20261027/audit.json)。

2026-10-27 V-family标签规则更新为v2：family_labels[B,4]的-100仅用于nonbinding、缺失链或没有观测残基的槽；其他缺失/被过滤/词表外标签统一映射unknown。VHH归第1槽且当前正例标签unknown。全121分片重扫，183类词表仅拟合train；train有效目标10581439（已知8246062、unknown2335377），valid102012（已知41798、unknown60214），无未见有效类别。原始数据、split与链序列均不变；旧audit中的过滤理由在此版表示不采用其known标注，不代表整条链没有辅助loss。[unknown标签审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_unknown_20261027/audit.json)。

2026-10-27 V8 R2记录与split不改；collator新增仅用于监督的family_labels[B,4]，顺序H/L/β/α，缺失/未知类/nonbinding为-100，不进入encoder feature_args。182类词表只拟合train OAS/OTS的oas_paired/ots_paired源注释，按物种命名、去allele及gene编号；共享TRAV/DV名称作为单个gene解析。训练排除260465条α链ANARCI链型歧义、11条locus/role冲突和2条不支持call；这些记录仍保留原MLM用途。valid排除1290条α链型歧义，无未见有效类别。首轮审计误把shard source当record source，零类输出未使用；已修正并全量重扫，当前仅使用output/v8_v_family_labels_20261027_r2/family_spec.json。[标签审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_20261027_r2/audit.json)。

<!-- four-relations-20261026 -->
2026-10-26 当前有效四类负例child为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v8_four_relations_20261026_r2`：train6533037/valid71422，共121分片；继承96分片、新增177528行（176822负/706正）。negx分片保存evidence_kind、raw行号或双train端点；精确交互键保留MHC/轻链，pair_key/epitope随新组合重建。VHH保留nanobody_vhh/<nb>与源裁剪表示，索引显式含VHH，required_objective拒绝旧目标；R1已标记SUPERSEDED。全新行collator及独立核验通过。[四类负例探索、证据与R2入口](/project_c20250601/251105016/project/dllm_test/docs/V8_FOUR_RELATION_NEGATIVES_20261026.md)
<!-- /four-relations-20261026 -->

2026-10-26 V8 MHC补全版新增独立抗体子版本：原94分片逐字节继承，新增2个igdesign_antibody分片、28条配对VH/VL负例（train23/valid5），总train6355547/valid71384。metadata保留原始CSV行号、重复次数、固定框架恢复证据及参考抗原来源；IL17A用Q16552，不能误用6PPG的IL17F。全部新行通过collator与独立来源重查，抗体负例共2984；VHH清单不计入训练。[数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V8_ANTIBODY_NEGATIVE_EXPANSION_20261026.md)。

2026-10-26 Diffusion新增仅用于监督/统计的cdr_ids（H1/H2/H3/L1/L2/L3编码1..6）及mask_scope/cdr_full_mask/cdr_eligible/edit_task注释；不进入encoder特征。完整七段拼接和无synthetic残基才接受CDR映射，按原序列位置映射固定槽位。ASD/OAS全量2263036行覆盖审计，OAS训练36条不合格回退、其他OAS均可用；ASD缺少完整区域标注保留旧任务。语料/index offset未改，继续独立885条验证。[CDR任务与当前入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

2026-10-26 V8缺失MHC的6752条全部属tcrdesign26_pmhc；作者原始TSV与本项目CSV中的MHC均为34个X，reader按占位符去除。6752个当前记录均找到原始文件行，原文件仅9列且无数据库/实验accession，源头可确证到TcrDesign-B发布，具体原始数据库及缺失/增强遮盖的区分仍不可辨。[来源逐行审计](/project_c20250601/251105016/project/dllm_test/output/mhc_completion_v8_20261026/missing_mhc_origin/summary.json)。

2026-10-26 V8新增负样本发布MHC补全子版本：56399条改变（train55253/valid1146），其余6370504条逐字节不变；105006条TCR中完整参考上下文56399/pseudo41855/无MHC6752。独立全量验证及每个改变行的真实unbinding collator通过，最长803tokens；总train6355524/valid71379、94分片及标签/split/受体/肽/mask均保持。参考填充来源保留在metadata和audit。[V8 MHC补全与当前训练入口](/project_c20250601/251105016/project/dllm_test/docs/V8_MHC_COMPLETION_20261026.md)。

2026-10-26 Diffusion全量V7复核：84个shard与全部6,249,956 train＋68,985 valid通过byte/index SHA及mask/slot检查。发现监控token_valid889条中4条观测受体与train精确重叠；新Diffusion冻结index仅将token_valid替换为独立885条，并记录原引用/过滤原因/audit SHA，原分片、offset、split、标签与旧验证均保留。训练新source比例写入training.source_sampling，不重写数据。分层896条train＋885条valid实际collator核对通过；不宣称同源隔离。[数据审计与新提交入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_BERT_RESTART_DATA_REVIEW_20261026.md)

2026-10-26 发布[immune_v8_unbinding_slots167_135_20261026](/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v8_unbinding_slots167_135_20261026)：84个V7 shard原样继承，新增10个非空nonbinding shard；train6355524/valid71379，新增105568/2394。V5显式负样本347700条重新过保护集合/近邻/聚类/长度/probe-valid检查，239738条测试重叠被排除；恢复行全量真实collator与独立二次选择/字节验证通过。slots167/135、max2048、原split/重复/标签/synthetic mask保留。原V7 MHC补全保留，新负样本保留V5上下文，未额外补全MHC。[完整格式、计数与证据](/project_c20250601/251105016/project/dllm_test/docs/BERT_V8_UNBINDING_20261026.md)。

2026-10-25 Stage1冻结评估新增独立评测产物，不改变训练数据或教师缓存：天然伙伴records/candidates JSON固定身份，scores JSONL保存两伙伴条件×五区域的log-probability和计数；native_recovery_context_per_query/summary CSV保存条件差值与分组均值，由已有预测派生；实验结构JSON仅保留通过核对的H/L序列、IMGT keys及CA坐标，geometry CSV记录query/gallery和实验RMSD。schema/SHA、输入边界及复现分别归[伙伴协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PARTNER_RETRIEVAL.md)和[结构协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_EXPERIMENTAL_STRUCTURE_RETRIEVAL.md)；执行状态见[记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25 在线实验新增只读基线引用[baseline_reference.json](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch/baseline_reference.json)，schema=immune.stage1.baseline_reference.v1：绑定旧scratch MLM终点9770的plan、checkpoint、验证统计路径/SHA及参数一致性审核，明确不同运行环境和只新增Pair训练。该文件仅供实验比较，不是学生初始化或教师标签；旧checkpoint/统计不复制，训练plan及源码SHA不变。jobs.json的selected_for_submission仅对train_pair_kd为true。

2026-10-25 新增immune.stage1.protenix_online.v1计划，teacher_target_storage=none，cache_root/cache_manifest=null。实时目标为每micro FP32[P,64] logits、FP32[P] confidence及int64[P,3]学生batch/i/j映射，P覆盖所有真实残基非对角上三角（链内＋跨链）；confidence为native PDE≤2Å双向均值，detach且按原pair数归约。所有教师张量仅在内存中，checkpoint不含教师权重或特征。随机种子绑定有序H/L序列及base seed，重启重算；新schema及source/teacher/data SHA约束身份。

PairConfig新增independent_layers（默认false）、distogram_chunk_size（默认0，保留旧非分块读出）；新在线计划设true/4096并开启Pair重计算。独立模式新增layer_pair_encoders.参数前缀，最后一套也负责终层读出；共享模式保留旧参数键。验证records.npz只含12列损失/计数汇总和row_id；不含teacher target。[在线方案与状态](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PROTENIX_AB500K_CONFIDENCE_PLAN_20261025.md)。


<!-- stage1-scratch-fast-20261024 -->
2026-10-24T12:36:34.657206+00:00 PairConfig/research_config增加bias_implementation、autocast_pair_embeddings、symmetric_pair_chunk，可选字段旧默认dense/false/4096；原checkpoint_pair_chunks在优化包设false。配置写入plan.pair及训练provenance，源码SHA与plan共同绑定恢复身份；旧配置通过新loader使用原dense行为，权重键/形状、语料、64bin教师缓存均不变。新独立输出为stage1_ab500k_scratch_fast_20261024，不接续旧实验身份。[优化版启动与核验](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_FAST_20261024.md)。
<!-- /stage1-scratch-fast-20261024 -->

<!-- stage1-contact-residues-20261024 -->
2026-10-24T11:33:55.652866+00:00 新增独立contact_residues诊断产物：CSV保留1基序列位置、AA、PDB链/残基编号及接触距离；NPZ索引用0基，保存全部真实接触Z_HL与Z_HL+Z_LH，以及所选对照双向向量、标准化展示、64bin概率和界面矩阵。三例162对，复用原坐标与教师SHA，不改变正式训练缓存。[接触残基与256维教师特征](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CONTACT_RESIDUE_FEATURES_20261024.md)。
<!-- /stage1-contact-residues-20261024 -->

<!-- stage1-cross-chain-diagnostic-20261024 -->
2026-10-24T08:13:09.252819+00:00 新增独立诊断格式：完整教师distogram_logits为FP32[N,N,64]、pair_states为FP16[N,N,256]；学生clean_logits/masked_logits及最终KD pair_states均FP32[N,N,64]。9例按request.student_positions仅映射真实残基，保留真实MASK位置，排除特殊token和尾部填充；完整张量位于容器/tmp，未并入正式SQLite缓存。持久化visualization_arrays.npz包含57份float32二维图，键为样例ID__图名，不能代替64-bin分布；原始结构、映射、SHA和存储位置详见[完整矩阵、实验接触与Pair特征诊断](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CROSS_CHAIN_DISTOGRAM_DIAGNOSTIC_20261024.md)。
<!-- /stage1-cross-chain-diagnostic-20261024 -->

2026-10-24 AB500K新增可选初始化契约 immune.stage1.random_init.v1：mode=scratch、seed、config_sha256、shared_state_sha256；scratch模式的parent_checkpoint/step/model_sha256/receipt_sha256均null，训练provenance绑定初始化身份，混入旧权重或指纹不符时拒绝。Pair research_config的parent记录scratch及inherited_parameters=0；数据布局、词表、教师缓存和张量格式不变。两组比较仍绑定同plan/初始化/数据/mask，报告按plan步数验证完整暴露及终点。[详细约定](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_PLAN_20261024.md)。

2026-10-23 AB500K训练修复包的metrics将累计token计数保留为selected_tokens/eligible_tokens，新增batch_selected_tokens/batch_eligible_tokens保存单步计数；缓存格式、子集及教师目标不变。新代码使用独立训练输出与provenance，不改写旧checkpoint的身份。[说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_TRAINING_REVIEW_20261023.md)。

2026-10-23 AB500K独立MLM包：A组run provenance的cache_manifest_sha256为null，训练/验证Dataset不打开缓存；D组保留完整缓存校验和实际SHA绑定。两组仍严格比较同plan、子集、父权重、训练配方、mask及验证seed。[入口说明](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/launch_mlm_independent/README.md)。

<!-- stage1-ab500k-offline-pilot-20261022 -->
2026-10-22 AB500K独立子集已生成并全量核验：500000原train行、4096唯一原valid配对、504096请求和16互斥教师分片，辅助文件约3.762GB；完整H/L配对精确重叠0，单独重链/轻链重叠训练记录分别114/67441，不宣称链或家族隔离。绑定当前V7索引，主语料未改。紧凑缓存为FP16 logits＋uint16索引＋字节重排/Zstd3独立记录帧、不可变分片及只读SQLite索引，读取还原FP32/long，64bins与每链1024对保留。504096条缓存已全部生成，16个完成回执通过；实占104.186GB，辅助数据3.762GB。2026-10-23 19:00:10 UTC完整payload验收通过，最终SQLite索引/manifest已发布，覆盖500000 train＋4096 valid、1032388608对；结构质量仍unvalidated，旧在线格式独立保留。[数据与缓存合同](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md)。
<!-- /stage1-ab500k-offline-pilot-20261022 -->

<!-- stage1-online-20261022 -->
2026-10-22 在线教师复用既有只读pilot SQLite，并增加按contract SHA/请求SHA分桶的动态缓存 `/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online_teacher_cache`。仍使用原request与sparse_cache.v1，原记录/配置/完整链/学生位置决定请求，随机MASK不进入缓存键；文件锁、临时写入、校验、fsync及原子rename处理并发与中断。读取逐文件核验，损坏/错契约/错映射失败。checkpoint冻结在线策略及命名空间，允许缓存追加；训练split和自然采样不变，valid指标继续用固定pilot。没有可用完整链的记录保留MLM-only；在线生成失败不静默降级。[格式与恢复说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md)。
<!-- /stage1-online-20261022 -->

<!-- stage1-cache-audit-20261022 -->
2026-10-22 新增工程审计报告 `immune.stage1.cache_audit.v1`，绑定cache manifest SHA，记录来源/split覆盖、排除链原因、最大pair数、对称性与同seed重跑测量覆盖、各worker显存。`status=passed`仅表示工程完整性，实验结构质量单独标为unvalidated，正式就绪为false并列出缺项；不替代 `immune.stage1.teacher_quality.v1`。原request/cache/SQLite格式与训练数据不变。[说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-cache-audit-20261022 -->

<!-- stage1-training-build-20261022 -->
2026-10-22 Stage 1新增 `immune.stage1.cache_index.v1`：SQLite以(split,row_index)绑定冻结V7的规范记录SHA、request与缓存metadata；缺项保留MLM，错行/缓存变化失败。教师JSONL按 `{split,row_index,request}` 流式处理，不改原语料。旧smoke索引保留训练OAS2条；新pilot已完成1785条请求，缓存train768/valid761共1529条、2,257,088对，256条TCR repertoire无可用完整链，保留MLM-only；新pilot仅链内KD。train覆盖0.012288%、valid覆盖1.103138%，工程审计passed，结构质量仍unvalidated。[Stage 1训练构建与提交](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-training-build-20261022 -->

<!-- stage1-cache-contract -->
2026-10-22 Stage 1新增独立教师request/cache契约，不改原prepared语料：请求绑定原记录、学生config、完整真实链、残基位置和显式跨链来源；synthetic／片段链保留MLM而排除该链KD。缓存存稀疏teacher pair索引、原生distogram logits、学生位置、权重revision／SHA、seed／loops；读取前检查内容哈希、序列及链映射。教师质量默认unvalidated，在显式engineering smoke或本轮在线探索模式中使用；配对来源不等于结构质量通过。规范见[实现记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md)。
<!-- /stage1-cache-contract -->


2026-10-22 当前V7存储元数据迁移到 `/project_c20250601/251105016`：manifest、验收引用、binding、index及开发记录的来源路径已同步更新，关联SHA重新绑定。84个训练/验证分片行字节、numpy偏移、记录数和split保持；标准完整数据门禁通过，84分片实际reader/collator抽读通过。新的release manifest SHA为 `3be10776edd599de4a6d6031ff4c4828ce516cf47bd93c4ccbbb7b86e307b2a5`，迁移前元数据及SHA见备份。[迁移验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/storage_rename_20261022/verification.json)。

2026-10-17 旧V5/fixed112000→LoRA3000测试暴露审计完成，Specificity/GDPa1完整配对仍进入父训练语料；现行CDR精确重叠及valid使用范围也已分开核对。未改数据格式或原始行，不能将后续V7清理追溯到旧权重。[唯一审计记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ab-exposure-and-layers-20261017)。

2026-10-16 当前BERT绑定最新V7 MHC-completed：84分片，train6,249,956 / valid68,985。全量字节哈希、无显式nonbinding、167/135槽位（含EOS）和观测残基检查通过，0条额外排除。加载数据目录内manifest绑定的immune_config.json（full_capacity304/max_length2048）；未改行字节，原父目录符号链接也绑定实际目标。MLM只监督观测受体AA，关系token统一MASK。V7固定MLM valid889条；外部二分类probe再按V7测试保护规则及候选覆盖筛选，最终TCR papers/TRAIT两来源：头训练383条、valid211条；ASD因负例不足不参与选点。它们绝不进入预训练加载器，valid受体在全量V7 train中0精确重叠。初始3193条开发记录通过实际新MLM collator，最终使用的1483条是其子集，V7 token-dev最长1041、外部probe最长1059，均低于2048；原MHC报告845仅指改变记录。见[新索引](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016/data/index_test_clean.json)。


2026-10-16 新原始来源 TCRvdb 只读核查：`/project_c20250601/251105016/project/dllm_test/data/tcrvdb/ttd_gil_ylq_nlv_glc_01-10-2026_TCRvdb.csv` 为 24 列、2,034 行，成对 αβ/CDR3/VJ、peptide、HLA 名称及 pos/neg/ambiguous 均有值；没有 MHC 序列或 split。α 103–121 / β 109–122 aa 满足 135/167 槽位长度；全文件命中当前冻结测试保护 CDR3 配对键，未接入训练或修改已发布版本。[核查报告](/project_c20250601/251105016/project/dllm_test/docs/TCRVDB_INSPECTION_20261016.md)。

2026-10-15 **新 V7 当前数据入口**更新为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_mhc_completed_20261015`。67,648 条上下文已补齐：MHC-I heavy+B2M 67,614，MHC-II α+β 34；train 更新 64,369、valid 更新 3,279。总量与原 split 保持 6,318,941（train 6,249,956 / valid 68,985）。HLA 标 `reference_longest`、B2M 标 99-aa `mature`，均明确 reference-derived；未通过映射者原样保留。仍加载目录内 167/135 配置；全量比较和真实 reader 验收通过。[完整数据契约与证据](/project_c20250601/251105016/project/dllm_test/docs/V7_MHC_COMPLETION_20261015.md)。以下 MHC 探索和槽位更正条目为补全前的历史阶段。

2026-10-15 DecoderTCR 外部格式核查：论文使用 MHC-I heavy+B2M / MHC-II α+β；官方参考实有 43 个 Class I 与 45 个 Class II 键，链范围并非统一全前体，且 Class II allele wrapper 查询失败已复现。可借鉴参考序列恢复流程，但未导入这些参考或修改当前 V7；无 TCR 的 pMHC 数据需要新增监督任务。[完整核查](/project_c20250601/251105016/project/dllm_test/docs/DECODERTCR_RESEARCH_20261015.md)。

2026-10-15 MHC探索解释更正：112条仅是“参考字符串完全一致”的粗筛样本数（train79/valid33，全部E01:03，共享一种358 aa重链）。A02:01的10种参考字符串已确认均来自同一条365 aa序列的不同覆盖范围；不能将字符串不一致一概认定为蛋白序列歧义，也不能把112当作可接入量上限。全量候选的覆盖感知重分类尚未实施，正式训练数据未变。[更正证据](/project_c20250601/251105016/project/dllm_test/docs/DUAL_MHC_DATA_EXPLORATION_20261015.md)。

2026-10-15 双链MHC探索：匹配当前新V7的39,048条MHC-I上下文恢复候选全部通过167/135真实collator检查（最长789 token）；严格全长参考唯一映射仅112条（train79/valid33），候选尚未写入训练数据。外部TCR3d有12个人源MHC-II结构未命中当前精确配对/T4测试键，尚待组装/可变域/融合构建体及最终入库核查。不能把历史完整MHC序列当作逐样本观测序列，也不能把候选数量当作已新增量。[探索与证据](/project_c20250601/251105016/project/dllm_test/docs/DUAL_MHC_DATA_EXPLORATION_20261015.md)。

2026-10-15 槽位更正：**新 V7 当前入口**为 `/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v7_from_v5_no_negative_design_slots167_135_20261015`。H/β 167、L/α 135（含EOS），残基上限166/134，`full_capacity=304`。使用目录内manifest绑定的 `immune_config.json`，不得套用旧187/154默认配置。原no-negative V7中4条超长训练记录整条隔离，train **6,249,956** / valid **68,985**，总计 **6,318,941**；标签、split、保留行字节及synthetic scaffold不变。原187/154验收是父版本的历史证据。构建及新验收见[重建说明](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。下列OAS-only及后训练记录描述各自已冻结的独立实验。

2026-10-15 OAS-only scratch 使用原 V5 manifest 的 `source=oas` 分片：train 25 分片 / 2,485,433 条，valid 1 分片 / 12,551 条，正式逐次验证固定抽样 256 条。原 JSONL、187/154 槽位及语义格式保持；不以行内 `oas_paired` 等 provenance 别名筛选。父数据完整性审计仍覆盖原 V5，训练/验证加载器只打开 OAS 分片。[选择清单](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_75m_oas_20261015/data_selection.json)；[实际加载与抽查](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_75m_oas_20261015/data_smoke.json)。

2026-10-15 V5→V7 更正版本通过全量复核：train 6,249,960 / valid 68,985；明确 binding 195,646 条、未标注／unknown 6,123,299 条，明确负样本为 0。保留 JSONL 与 V5 字节一致，187/154 design 布局和 CDR3β synthetic mask 保持；336 例张量对照、168 例实际 reader 检查通过。[数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。

2026-10-14 全量后训练以行号索引继承pilot的exact receptor-tuple排除规则：原训练7,626,885条，排除137条，保留7,626,748条；85分片哈希全部复核。按原manifest来源索引，tcr_papers不与raw tcr_native别名混合。原始JSONL、链槽位、token/role/scope均未改；不是同源去污染，也不撤销预训练既有暴露。[唯一索引清单](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/full_data_index/manifest.json)。

2026-10-14 偏好分支仅从冻结训练子集构造2909对样本；按条件、链角色、补齐类型、观测长度分桶匹配，chosen/rejected均设相同binding条件。全量生成特征逐张量相等；训练时全目标槽位遮蔽，只以真实残基及全观测链EOS计分。原始数据和确认集未改。证据：[/preference_data/manifest.json](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/preference_data/manifest.json)。

2026-10-14 后训练r5新增训练/评分专用cdr_mask、observed_terminal_mask、binding_labels及来源分组；均被denoise特征白名单排除。observed_eos仅把有synthetic链的模板EOS改为mask并去掉其loss，槽位/角色/位置不变；旧prepared文件与旧冻结训练保持。新增验证明确区分残基与可评估终点，不将缺失终点记为完美预测。[方案](/project_c20250601/251105016/project/dllm_test/docs/FIXED_CHAIN_EXPLORATION_20261014.md)。

2026-10-14 后训练探索核查：当前fixed-chain collator在存在任一真实残基时，仍于补齐序列末尾监督EOS。部分TRAIT/TCR记录末位为synthetic，因此该EOS是渲染模板终点，不能一概解释成真实观测链长。数据和历史分数未改；将比较仅观测终点监督，并区分真实全链/模板终点验证。[审计](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/synthetic_endpoint_audit.json)。

## 2026-10-13 历史固定长度统计溯源

已找到[原始长度统计](/project_c20250601/251105016/project/dllm_test/scripts/chain_length_stats.json)：heavy P99=165、light P99=131，`recommendations.cover_99` 分别为168、136。[统计脚本](/project_c20250601/251105016/project/dllm_test/scripts/stats_chain_lengths.py:193)按残基分位数向上对齐到8的倍数，建议值本身不含特殊token；历史v2最终将168/136定义为含CLS的encoder画布，decoder为167/135、最大残基数166/134。原始统计混合CSV、prepared、canonical及不同sequence scope，不能将其P99当作当前V5独立样本分布；但这份文件补充了此前未定位到的历史统计依据。

当前V5的[初始容量全量审计](/project_c20250601/251105016/project/dllm_test/output/v5_fixed_chain_f011_20261012/initial_capacity_audit.json)表明：167/135槽位在7,729,193条train+valid记录中仅5条超限，分别为抗体重链1、TCRβ链1、TCRα链3，抗体轻链0；记录超限率约0.00006469%。187/154源于容纳全部极端长度的选择，不能表述为绝大多数样本必需的容量。本次仅补充证据，未修改运行中的训练、槽位配置或数据过滤。

2026-10-13 CHIMERA新增独立H3数据版本，按复合物保留官方原始split，参考信息与模型特征分离；数据schema、路径、输入异常记录及验收见[CHIMERA现行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-13 AB–Ag评测数据采用独立v2原始行版本，取消样本去重和按重复/标签冲突删除；原v1保留。数据根、输入兼容性限制及split语义见 [AB–Ag 当前原始行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

2026-10-12 新版本已正式运行：75,601,176参数，V5训练集7,626,885条；残差0.1、双链独立RoPE、链内全注意力/全局交替，固定槽位187/154。300000步/global512=8×micro64×GA1，LR1e-4/warmup2000/cosine，随机从头初始化。2026-10-12T18:23:05.641727+00:00核实已到65步，最近记录训练loss=2.9047，梯度有限；13项CPU测试、8/8整模GPU检查、8/8 NCCL、85分片全哈希门禁通过。每1000步验证/保存，首次验证尚未到达。证据：[启动验收](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup_acceptance.json)。旧270M已按用户要求停止（最后约72279步，完整保存72000保留），原V7 f011此前已完成。

以下object302/270M说明属于此前已启动版本；新方案以以上配置为准。

更新：2026-10-07T06:41:12.247809+00:00

## 数据选择

**本次新训练使用V5**：/project_c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion。

| split | manifest记录数 | 分片数 |
|---|---:|---:|
| train | 7,626,885 | 78 |
| valid | 102,308 | 7 |

已逐条检查全部7,729,193条，85分片行数与manifest一致，并计算全部分片SHA256。来源为OAS、OTS、asd_antibody、TRAIT、tcr_native、tcr_papers、tcr_repertoire。沿用V5已有数据及过滤结果，不叠加V7后续清理作为默认输入。

V7历史数据仍保留于 /project_c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，只用于其已有训练／评测追溯。本次不修改或删除任何数据分片。

## 输入规则

| 类型 | 干净输入结构（受体内容随后补mask至302容量） |
|---|---|
| 无条件抗体 | `<prots><null><protd><unknown><prots><ab>H.L<protd>` |
| 抗原→抗体 | `<prots>antigen<protd><关系><prots><ab>H.L<protd>` |
| 无条件TCR | `<prots><null><protd><unknown><prots><tcr>beta[.alpha]<protd>` |
| 肽→TCR | `<prots><pep>peptide<protd><关系><prots><tcr>beta[.alpha]<protd>` |
| pMHC→TCR | `<prots>MHC<protd><binding><prots><pep>peptide<protd><关系><prots><tcr>beta[.alpha]<protd>` |

- 语义JSONL保存chains、chain_roles、regions、chain_metadata等；送入模型时统一转换，不复制一套文本数据。
- 对象容量302（166+134残基＋split＋protd），总输入上限2048。生成不需要真实目标长度。
- attention按`<prots>…<protd>`对象分段，头部和内容同段；内部`.`不分段、不重置位置。目标对象的全部预留槽位同段，不按真实／预测终点动态缩短。
- 保留CDR-only全长scaffold及真实锚点；synthetic_residue_mask标记的位置转mask，labels=-100。真实观测X保持原样。
- synthetic和尾部mask参与attention但不计损失；真实残基、`.`和`<protd>`参与监督。固定对象头部作为条件。
- VHH不纳入本版范围；本次全量扫描确认VHH为0，无需额外排除。

## 容量依据

现有固定canvas代码 /project_c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/grammar.py 定义H/β encoder 168、decoder 167、残基最多166；L/α分别136、135、134。encoder包含cls，decoder至少留一个EOS。

新内容预算为 **166+134+split(1)+protd(1)=302**；固定头部prots与ab/tcr在预算外，整个受体块为304。只采用总预算，不限制每条链分别166/134；单链也使用同一对象预算。split位置由模型学习，attention仍按对象分段。全量检查确认V5所有对象均能放入302，无需裁剪或丢弃。

## 全量检查结果

| 对象类型 | 条数 | P99 | 最大长度 |
|---|---:|---:|---:|
| 完整双链 | 4,839,491 | 244 | 299 |
| 补全scaffold双链 | 2,764,364 | 237 | 271 |
| 单链 | 125,338 | 142 | 146 |

对象长度包含split和protd；总输入最大1218，低于2048。585,124,843个synthetic位置的长度／取值检查通过；非法记录、超容量记录、VHH均为0。

全量逐条执行schema、残基、角色、synthetic及等价布局长度检查；另以每分片3条、共255条真实collator输出核对公式。启动门禁再次校验全部分片SHA256。完整报告 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json；启动回执 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/native_parent_preflight.json；97项代码回归测试通过。

这次是V5布局与来源绑定审核，不重新运行benchmark去重，不把V7隔离结论套用到V5。原V7抽样报告仅作历史记录。旧完整数据审计见[归档](/project_c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/DATA_FORMAT_AUDIT.md)。


2026-10-12历史草案已被75.6M固定双链实现替代；不再待用户选择链边界。

## 2026-10-15 CDR概率聚合探索数据派生

源：[较新SAbDab快照](/project_c20250601/251105016/project/dllm_test/data/downstream/cdr_infilling/sabdab)。没有覆盖原数据或Kong主榜。逐条验证`heavy[start:end+1] == cdr_seq`，三类H CDR各3,320条均为闭区间；排除与上一轮相同PDB的1,818条、同完整重链的26条、超出原150/128虚拟网格的2条，每类保留1,474条。

用“同PDB、同完整重链、同H3序列”的传递连通分量整组划分，共1,180组；哈希排序前256组探索，其余确认。每类CDR探索318条、确认1,156条，三个CDR共享同一划分。新派生row使用fold=0表示该单一划分，保留source_fold；主指标为序列平均AAR，不冒充原10折分数。索引与SHA：[rows](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/rows.json)、[plan](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/plan.json)。排除的是上一轮基准暴露，不宣称预训练去污染或所有历史评测均未见过。

2026-10-15 更新：CDR新探索派生预留组已用于冻结软上下文确认，不再是后续方案的未见确认集。全量原始参考与分组隔离验收通过，历史Kong/SAb仅作迁移检查，不混称新独立测试。[来源验收](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015/confirmation_source_audit.json)。

<a id="cdr-block-data-20261015"></a>
## 2026-10-15 分组软上下文的 OAS 确认数据

从原 V5 OAS valid 的12,551条配对抗体中，按两份checkpoint记录的选模indices及已有实验heavy/H3键排除277条；准备阶段完整扫描当前绑定训练集，按相同完整重链、H3或已有H-cluster标签再排除1,154条。剩余11,120对按三类键的传递连通分量分成10,870组，组哈希排序前2,048组用于确认，包含2,094对；其余8,822组、9,026对在2026-10-15阶段预留。H1/H2/H3共享配对及划分，每模式分别为confirm 2,094条、reserve 9,026条，合计33,360条任务记录。[冻结数据清单](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015/data_plan.json)绑定来源、训练分片及派生rows的SHA256。

独立审计逐条恢复原始序列，核对regions拼接与已有IMGT位置标签，并重建全部分组；确认与预留的heavy/H3/H-cluster交集均为0。准备扫描的78个train分片哈希与两checkpoint绑定的原始训练审计一致；本次独立审计复用该哈希证据，未再次逐条重算训练排除。`mask_h`使用0-based残基坐标，`mask_l=[]`；完整参考保留于数据文件，必须经既有adapter遮蔽目标后送入模型。该集合是OAS补充任务，排除范围不等同普遍同源或预训练去污染。2026-10-15阶段结束时，confirm已完成冻结候选评分，reserve全部未评分；2026-10-16已从该reserve固定划出下述新消融测试，不能继续把全部9,026对称为未分配预留。[独立审计](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_context_20261015/data_independent_audit.json)；[评测协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-context-20261015)。

<a id="cdr-block-ablation-data-20261016"></a>
## 2026-10-16 分组软上下文消融数据派生

仅从上述原`reserve`取样，禁止使用旧`confirm`。按既有group哈希字典序，前4,096个组件的全部4,202对用于H3主评估；其中前512个组件的526对同时用于H1/H2附加评估。新数据共5,254条任务记录，H3为4,202条，H1/H2各526条；所有选择在看评分前固定，无数据驱动选参。新row仅将`split`改为`ablation`并新增`parent_split="reserve"`，原`id/group/source/row`及序列、坐标保持，旧文件未改。[冻结派生清单](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016/data_plan.json)。

两checkpoint均已完成本轮全部5,254条任务记录的评分及全量审计；4,096个组件、4,202对已用于本轮评估，不能再称未见测试。其余4,726个组件、4,824对仍完全未分配使用；已评估集合中另有3,584个组件、3,676对仅完成H3评分，其H1/H2未评分，但整对不能计入完全未触碰的预留。独立审计重建选取顺序、逐条恢复原始来源与CDR坐标，核实新集合与旧confirm、剩余完全未分配集合之间的group/heavy/H3/H-cluster交集均为0。训练与选模排除继承原哈希绑定及审计边界，本次未重扫训练记录，也不新增同源或更广预训练去污染结论。[独立审计](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_block_ablation_20261016/data_independent_audit.json)；[消融协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-ablation-20261016)；[最终结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-ablation-results-20261016)。

<a id="cdr-block-budget-data-20261016"></a>
## 2026-10-16 4/8前向补充的数据复用

新实验完整复用已评分的消融rows字节和顺序：H3 4202条/4096组件，H1/H2各526条/512组件，总5254条任务记录。没有新增样本或改变来源/坐标/分组，没有消耗剩余4824对/4726组件；这是已用集合上的固定后续比较，不能称未见确认。七组时H3有16条长度<7（6条长5、10条长6），H2有1条长6，合计每checkpoint 23次空组样本查询，均保留成本计数；按原batch24没有整batch为空的组查询。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-budget-20261016)。

2026-10-16 4/8前向复用评估及恢复后的完整审计完成；没有增加样本或消耗reserve，原来源/分组/坐标及使用边界保持。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-budget-results-20261016)。

<a id="cdr-left-to-right-data-20261016"></a>
## 2026-10-16 从左到右CDR验证数据复用

完全复用budget/ablation同一rows（SHA256 `12e081c035b180bf027eaf5ca749d0fce5e9b360f582a80da400e97cfb15d7e8`）与batch24顺序：H3 4202行/4096组件，H1、H2各526行/512组件。此前已评分，不另消耗未分配reserve，沿用原exact/cluster排除范围，非普遍预训练去污染保证。[冻结数据记录](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016/data_plan.json)。

每checkpoint有效逻辑前向：H1 4402（平均8.368821，长度8–16），H2 4120（7.832700，6–10），H3 65565（15.603284，5–36）；总74087。固定219批、batch24末批22，不裁减已完成行：5083次批前向、121948次含padding的样本前向，47861为短行继续占位成本；均已包含初始prior，门禁另计。行为与统计协议见[任务说明](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-left-to-right-20261016)。

2026-10-16 左到右验证完成，所有数据/批次/来源及实际成本与冻结方案一致，未消耗剩余reserve；[交付](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_left_to_right_20261016/left_to_right_delivery.json)。


2026-10-18T16:32:27.274268+00:00：新增OTS衍生TCR无标签回放池，沿用prepared BioSeqRecord和现有runtime输入格式；原始样本不改，行来源独立记录。[无标签TCR回放数据准备](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#tcr-preservation-replay-20261018)。

- 2026-10-18T17:02:18.805844+00:00：新CDR缓存使用native坐标和26类有限支持；padding与lengths规则、全部样本保留说明见[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-preservation-targets-20261018)。


2026-10-19T01:33:21.258505+00:00：新增目标诊断目录/project_c20250601/251105016/project/dllm_test/output/diagnostics/token_head_norm_targets_20261019及/project_c20250601/251105016/project/dllm_test/output/diagnostics/token_mean_head_norm_targets_20261019；features.npz保存246×768目标池化及原层缓存，calibration.npz保存head/gamma/加权行空间，token_checks.json保存逐batch约束误差与token数（不保存全token特征），各性质legacy_oof.npz复用原协议。均无训练参数文件。[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#token-head-norm-targets-20261019)。


2026-10-19T01:43:54.432221+00:00：/project_c20250601/251105016/project/dllm_test/output/diagnostics/feasible_final_targets_20261019新增train/screen ragged token_targets.npy(float32,Ntokens×768)，coordinates.npz含int64 offsets、int16固定坐标positions/token_ids；pooled_features.npz含原模型、第二层、desired、float64实现均值、float32存储均值，checks.json为batch约束/迭代统计。verification.json绑定全部数据SHA；训练读取每条样本前对照collator实际token坐标/ID。不是模型参数或GDP标签集。[格式生成入口](/project_c20250601/251105016/project/dllm_test/scripts/research/prepare_feasible_final_targets.py)。
# 通用教师工程验证数据补充（2026-10-24）

本轮8个固定案例、PDB原坐标来源与SHA、链/残基映射见[首轮验证协议](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_TEACHER_VALIDATION_20261024.md)和[examples.json](/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024/examples.json)。输出为native前18/19/20-bin二维分数及JSON计时/接触诊断；不是正式训练缓存，不修改既有distogram合同或数据切分。标准蛋白序列和缺坐标掩码明确绑定，未见结构泛化未验证。

2026-10-25全部24个分数NPZ通过形状、有限性、范围和累积关系检查，共19,700,330字节。[comparison.csv](/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024/comparison.csv)逐例记录计时、显存、原始链内/跨链AP及分数SHA；[equivalent_chain_evaluation.json](/project_c20250601/251105016/project/dllm_test/output/general_teacher_validation_20261024/equivalent_chain_evaluation.json)保留仅相同序列链交换的全部评价映射和AP19最优映射。映射是参考结构辅助的评价结果，不修改输入顺序、NPZ分数或训练标签。

2026-10-25新增[置信度8例回执](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_confidence_20261025/probe_r2/summary.json)。每例NPZ采用无对角上三角顺序`i<j`，P=L(L−1)/2：`logits`为FP16[P,64]，`confidence_cdf`为FP16[P,3]，三列为原生PDE≤1/2/4Å概率；`confidence`为最后一列的兼容副本，另有`length`和`chain_lengths`。该NPZ仅为小样本诊断格式，尚不是50万分片缓存合同；未保存原生Pair latent或坐标。每例SHA、原生坐标误差/分布NLL分组诊断保存于JSON；PDE未校准为distogram正确率。
