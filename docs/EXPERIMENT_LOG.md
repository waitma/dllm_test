> 2026-10-28T02:32:25.344359+00:00：A0/A1 本地筛选队列已启动；micro56、global512；8项CPU检查及真实GPU评估通过。实现commit e3c6692，远程push因凭据缺失未成功；终态及完整曲线以架构探索记录为准。

> 2026-10-28 架构探索实施里程碑：独立分支及冻结 V8 基线完成；实现八变体、单 GPU 累积归一化与恢复、四卡/多机分片、生成和表征评估、原始曲线报告、结束自动 commit/push。CPU/GPU 流程检查完成情况及本地 A0/A1 进展以 [架构探索实验记录](/mnt/project-storage/fset-rN4kfh/251105016/dllm_arch_explore/docs/ARCH70M_EXPLORATION.md) 为准；未改写现役 1B 训练。

# 实验记录与更正

2026-10-28 03:00:11 +08:00：按用户“不要测试，直接开始训练”，准备1.057B V8正式随机初始化300000步包，micro16×GA4×8/global512，LR1e-4、warmup15000、cosine；沿用正式方案每1000步验证/保存、每25步日志，保留最近两份＋最佳。原micro16命令已转接正式入口，跳过独立NCCL资格/profile/smoke/200步测试；保留源码、数据、设备、认证及checkpoint完整性检查。语法、冻结输入与121分片检查通过；未新增GPU测试。当前开发机仅1卡且无平台控制CLI，未提交或开始八卡训练，等待用户平台执行。[正式入口与配置](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_DIRECT_TRAIN_20261028.md)。

2026-10-28 02:49:05 +08:00：micro8的1B用户任务job-wkwas11lxghg-master-0已完成200/200步；checkpoint-200文件哈希及八rank参数一致性核验通过。按用户要求准备独立micro16×GA4×8/global512配置，保留NCCL2.31.2、确定性算法和严格恢复门禁。RPBZZZ6单卡最大长度2048三步实测通过，峰值allocated51.25/reserved55.38 GiB（总94.97 GiB）；冻结入口与121分片检查通过，八卡DDP和smoke待执行。新包仍为独立随机初始化200步测试，原结果保留，助手未提交/取消云任务。[配置与入口](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_MICRO16_20261028.md)。

2026-10-28 02:28:01 +08:00：用户任务job-wkwas11lxghg-master-0完整八卡smoke已通过，资格回执匹配当前plan且各文件SHA核验通过；BF16及独立FP32/math恢复重放的八rank参数最大差均为0。正式训练已运行至step 135/200，最新检查点step 100，尚未完成200步；当前数值以train/status.json为准。助手未修改运行中的冻结包或提交/取消任务。[状态快照](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/observations/job-wkwas11lxghg-master-0/20261027T182801Z/status.json)。

2026-10-28 02:04:37 +08:00观察：用户任务job-wkwas11lxghg-master-0使用新plan 2e51a87693445ade7b43e57e369def42b5bdb255caa6c4c70ff118e8add7bfef；八卡NCCL2.31.2和1B/DDP profile通过，W&B新ID成功初始化，checkpoint-1已提交。八个rank的BF16恢复重放参数、Adam一阶/二阶矩差均为0，所有记录指标完全一致；此前失败的BF16重放门禁在本轮通过。独立FP32/math恢复检查与完整smoke资格仍在运行，正式200步尚未确认开始。profile和恢复阶段缺少控制台进度，短暂静默不等于卡死；本轮只读观察训练产物，没有修改正在执行的冻结源码、取消或重启任务。[本轮实时证据](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/observations/job-wkwas11lxghg-master-0/20261027T180437Z/status.json)。

2026-10-28：用户任务job-3mz0r8fapsvx-master-0八卡NCCL2.31.2及确定性1B/DDP profile通过，01:25:54 +08:00因重复W&B ID air1btest20261027smoke与resume="never"冲突退出；这是重试入口归档本地产物却遗漏云端身份管理的实现错误。新增每次新建独立ID、checkpoint哈希绑定实际ID、严格恢复复用；更新实际冻结trainer/plan并在同账号在线验证后重绑定登录回执。27项回归、真实W&B两次新建加一次恢复、冻结入口与121分片检查通过；两条云端验证run明确标记validation-only，无预训练。前次恢复重放参数最大2.8769e-6差异在最新任务尚未运行到，仍待八卡验证，严格阈值保留。旧包与证据备份，助手未提交/取消任务。[修订回执](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/wandb_retry_revision.json)。

2026-10-27：按用户要求精简1B包W&B标量日志。valid只保留loss、mask_accuracy、relation_accuracy、family_loss、family_known_macro_f1；训练保留六项，probe保留macro_auroc。直接白名单检查通过，完整本地指标保持；同步冻结trainer/plan哈希，核对现有W&B身份并更新登录回执绑定。模型、训练预算、数据与验证计算不变，无训练或云任务提交。[验收](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/output/bert_v8_1b_test_20261027/logging_revision.json)。

2026-10-27：用户授权1B测试包W&B重新登录，独立API确认waitma，pressure_jian/dllm-immune真实认证检查run上传并结束成功。SDK0.23.1可选viewer.flags=null异常通过支持的x_disable_viewer设置处理，登录verify、独立身份和真实写入验证保留；未升级SDK。只修改认证启动模块及冻结哈希，模型/数据/训练配置不变；共享凭据0600且不写日志。未启动训练或提交云任务。[认证说明](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)。

2026-10-27 ownkaifa02：构建独立1,057,514,727参数BERT V8测试包（28层/1536/24头/FFN6144），200步、global512，八rank最长形状DDP及BF16/FP32恢复验证为强制前置。新路径环境与物化数据绑定通过CPU检查，原33项目标/恢复回归及8项新迁移/认证门禁测试通过；真实1B CPU前向通过。W&B要求重新登录并验证目标项目写权限，尚未取得凭据；新平台存储资源字段待确认。无GPU训练、无submit/cancel，不新增Active云任务。以[测试说明](/mnt/project-storage/fset-rN4kfh/251105016/dllm_test/docs/BERT_1B_TEST_20261027.md)和同目录验收回执为准，不作为1B性能结论。

2026-10-27T04:36:28.400835+00:00 修复V8 family启动data_gate的隐式旧路径依赖。旧包在audit hook拒绝/vepfs访问时复现FileNotFoundError，调用链bert_mlm_v8_data→prepare_immune_v8→v7_rebuild_scope→prepare_immune_v7→build_downstream_banks._load_normalizer。修复normalizer为当前包导入并更新相关旧路径，新增bundle_pathfix；plan SHA cf58934611dfaaca05757bc64bd884567d16e601d4f58a804195bf6f9f47681f。数据、模型、family目标和8卡配置不变。禁旧挂载完整预检、W&B导入及10项启动测试通过；证据 output/bert_v8_vfamily_all_relations_scratch_300k_20261027/pathfix_verification.json。云端后续GPU资格未验证。

2026-10-27T04:08:38.107064+00:00 修复V8 family网页提交EnvironmentNameNotFound：原脚本按名称激活pllm依赖开发机Conda搜索配置，现对齐成功diffusion任务，以绝对路径 /project_c20250601/251105016/conda/envs/pllm 激活。新增 /project_c20250601/251105016/project/dllm_test/output/bert_v8_vfamily_all_relations_scratch_300k_20261027/launch_envfix.sh；旧launch与冻结bundle保持。三个网页入口及当前两份train_jobs YAML重定向修复脚本并同步入口SHA，生成器同步修复。训练解释器仍为原已验证protenix_abtcr绝对路径；模型、数据、8卡及loss配置不变。未提交任务。 验证：隔离Conda环境搜索目录的绝对路径激活、bash语法及原网页入口--verify-only均通过；121分片/6,604,459条数据和冻结plan SHA核验通过。GPU启动待用户重提。

2026-10-27T03:31:17.906861+00:00 更正先前过宽的负例family排除：用户要求nonbinding也预测V-family，现token关系监督与family辅助监督分别定义。43项测试通过，含known/unknown负例family梯度非零且序列完全不mask/不做MLM，checkpoint往返与2-rank空标签rank累计仍通过。新v3冻结包与原网页Shell转接完成；数据划分未改，无提交或训练提升结论。[全部关系family审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_all_relations_20261027/audit.json)；[原网页Shell（已更新）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:26:57.121447+00:00 用户要求显式unknown分类，已实现183类family头及schema v2；41项测试通过，新增missing annotation/VHH positive→unknown的非零梯度、缺链/合成占位忽略测试，保留negative零辅助梯度及2-rank累计归一化测试。全数据重审10581439训练链目标/102012验证链目标；未改变序列/划分。新包与网页Shell准备，未提交。[unknown标签审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_unknown_20261027/audit.json)；[原网页Shell入口（已转接unknown版）](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:22:19.421970+00:00 V8 V-family训练新增用户自行网页提交的Shell包装，参照先前nonspot网页入口，增加挂载检查、证书环境、双路日志及launch哈希验证，训练包不改。语法检查通过，未提交作业。[网页提交Shell](/project_c20250601/251105016/project/dllm_test/output/bert_unbinding_v8_vfamily_scratch_300k_20261027/web_submission/entrypoint.sh)。

2026-10-27T03:20:43.874477+00:00 V-family辅助头落地：新增182类species-qualified train-only词表，8246062训练链标签/41798验证链标签，λ=0.1；未添加J预测。39项测试通过，其中2-rank Gloo包含空标签rank并与单批次参考更新比较，另覆盖负例辅助梯度为零、checkpoint往返和验证macro-F1。冻结包CPU数据门禁通过；GPU/multi-rank正式资格待作业内执行，无提交/无性能提升结论。原始记录、split、旧冻结实验不改；前轮零类审计未使用，当前r2重新扫描结果见[标签审计](/project_c20250601/251105016/project/dllm_test/output/v8_v_family_labels_20261027_r2/audit.json)。[当前family训练YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_vfamily_scratch_300k_8gpu_20261027.yml)。

2026-10-27T02:46:58.303100+00:00 V8 R2从零训练脚本准备：复用R2冻结源码/data/index并仅更新启动器，显式随机初始化、无parent_checkpoint，暂定300000总步、LR1e-4/warmup15000。8项初始化边界测试＋5项VHH监督/兼容测试通过；准备阶段数据门禁通过。新八卡profile/replay待作业内执行，无submit/cancel。family预测头未加入。[从零训练配置](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_unbinding_v8_scratch_300k_8gpu_20261027.yml)。

2026-10-27 汇报Pair完整矩阵诊断：复用同revision/4cycles/noMSA的既有Protenix教师缓存，在CPU上读取当前学生初始与step1250检查点，提取完整Pair/logit张量，按真实残基索引比较固定通道与原生短距离分数；另对既有1JHL/1MLC/1VFB完整H–L块重算实验接触AP。三案例平均教师0.97448，学生0.00675→0.08037，仅作既有诊断案例证据，不作独立泛化或下游收益。完整MLM与Pair/KL轨迹冻结到9770/1400步；78项来源SHA、完整矩阵分数与3例接触AP重算通过。无新训练、教师推理/GPU任务或headline成绩变更。[结果与参数](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/protenix_matrix/summary.json)、[验收](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/data/protenix_matrix/verification.json)。

2026-10-27 汇报Pair诊断：读取既有Protenix在线日志并冻结快照，整理共同step1250的MLM对照和按记录配对bootstrap；另在8条现有验证记录上只读CPU提取真实64维Pair latent及反馈开/关CE，初始与训练后使用共同PCA基底，原始特征/投影/参数均保存。证据支持教师目标拟合及反馈通道被使用，不扩展为真实结构精度或下游收益结论；没有新训练、GPU/云任务或headline成绩更新。[诊断与数据](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/说明.txt)、[来源SHA](/project_c20250601/251105016/project/dllm_test/PPT/20261027_项目进度汇报/references/protenix_evidence.json)。

2026-10-26T16:35:04.061786+00:00 Diffusion四卡任务t-20261027001920-m9ct7为Failed，四rank smoke通过、正式0步；根因是Python3.10执行时继承cp312 W&B vendor。已准备兼容依赖、GPU前真实SDK预检及独立修复包，原Shell/YAML转接；80项CPU＋冻结16项、离线init/log/finish及A100 max2048复测通过，原失败现场保留。未重新提交或启动正式训练。[故障与当前四卡入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

<!-- four-relations-20261026 -->
2026-10-26T16:49:11.514915+00:00 四类负例R2完成：解除10x历史4000导入上限并保留测试保护，负例107990→284812；新增176822中51668筛选阴性、125154推断。全部177528新行全量读取/独立来源与ID核验通过，24项CPU、44条真权重批次检查通过。收尾修复R1的TCR辅助配对标识及smoke简化索引兼容，R1停用；R2仅63319行identifiers变化，序列/标签/split/mask全部不变。未正式训练，八rank资格待运行。[四类负例探索、证据与R2入口](/project_c20250601/251105016/project/dllm_test/docs/V8_FOUR_RELATION_NEGATIVES_20261026.md)
<!-- /four-relations-20261026 -->

2026-10-26T15:58:04.024810+00:00 用户要求改为四卡：新增独立CDR Diffusion四卡冻结包，保持global512/原LR/配方，每rank256视图按48×5＋16累积。71项CPU＋冻结7项测试、1781条collator及四卡每rank负载A100短测通过，reserved69.246GiB；原Shell转接，旧八卡YAML禁用。四rank NCCL smoke待运行，无云提交或正式训练。[CDR任务与四卡入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

2026-10-26T13:51:19.790378+00:00 官方Ophiuchus-Ab三个冻结表征评测完整执行并通过最终核验，Specificity/m396按用户授权在本机完成。当前结果已写回原表，源码适配及GDP参考CV边界保留；Ridge警告经固定参数float64控制检查。[执行与诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ophiuchus-probe-audit-20261026)。

2026-10-26T13:28:21Z 按用户要求参照最近BERT/Stage1的Shell形式，CDR Diffusion新增独立launch.sh，包含pllm环境、8rank smoke→正式训练及CPU --verify-only；原train.sh/YAML统一调用。实际CPU预检exit0，冻结plan/源码不变，无GPU训练或云提交。[当前启动说明](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)。

2026-10-26 V8抗体实测负例扩充：1243条IgDesign原始实验行经严格标签/序列与既有测试保护筛选，新增28条配对VH/VL阴性；原V8抗体负例2956→2984。15项CPU测试、全部新增行collator/独立序列审计及BERT287000真实28条混合批次单A100更新通过；新增来源最终全阴性，未声明模型提升。VHH557236条负标签仅清点，现有train中15507条仍未接入。本轮无submit/cancel、无正式训练。[来源与新版本](/project_c20250601/251105016/project/dllm_test/docs/V8_ANTIBODY_NEGATIVE_EXPANSION_20261026.md)。

2026-10-26T12:51:41.096524+00:00 按用户指定把固定框架单/多CDR生成接入现有75%mask训练分支；有标注AB内部默认50/25/25，保持edit配方。准备独立BERT287000起步的CDR训练包，65＋36测试和A100短测通过，未提交正式训练。[CDR任务与当前入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_FIXED_FRAME_CDR_20261026.md)

2026-10-26T09:39:38.359633+00:00 V8负样本MHC仅在来源和参考一致时补全56399条；其余48607条TCR保持原表示。完整数据独立重查、43项MHC测试+19项目标回归及真实父权重单卡更新通过。补全数据新建索引和冻结包，未运行正式训练，尚无效果结论。[V8 MHC补全与当前训练入口](/project_c20250601/251105016/project/dllm_test/docs/V8_MHC_COMPLETION_20261026.md)。

2026-10-26T09:24:58.837459+00:00 Diffusion76500恢复建议撤销，按用户指定从BERT287000开启新Diffusion。完成全量数据复核、带上限source sampler和4条暴露验证样本过滤；50＋12测试及单卡更新通过，8rank/长期效果未运行。无新云任务。[数据审计与新提交入口](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_BERT_RESTART_DATA_REVIEW_20261026.md)

2026-10-26T02:57:07Z Diffusion梯度审计：grad_norm历史定义为裁剪前；后期已记录梯度未超阈值，验证退化主要由TRAIT/tcr_papers贡献。旧status=training不能代表正在运行，最后监控为Killed。独立代码修复和保守LR控制完成，原等权采样过拟合风险另行记录。CPU30项、单卡真实更新等价及降LR数值检查通过，没有长训收益结论或新云提交。[依据](/project_c20250601/251105016/project/dllm_test/docs/DIFFUSION_GRADIENT_REVIEW_20261026.md)。

2026-10-26T02:56:11Z V8 unbinding工程版本完成：恢复107962条通过原保护集合检查的负样本；负样本仅关系CE、正样本保留MLM。24项CPU回归、真实287000权重/混合数据单GPU更新通过，父张量逐值一致；micro32×GA2×8/global512准备就绪。默认新增13000步；未提交或开始正式训练，八rank资格待启动，不构成模型收益证据。[实验说明与启动](/project_c20250601/251105016/project/dllm_test/docs/BERT_V8_UNBINDING_20261026.md)。

2026-10-25T22:47:39Z 周报按用户要求精简：结果表范围与上周一致；ESMFold2探索以架构、50万条/10epoch训练曲线及GDPa1结果呈现，删除pipeline与复杂候选说明。只重绘已完成9770步日志，不新增模型结果。[当前发布验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/simplification_verification.json)。

2026-10-25T22:29:49Z 本周周报已通过Lark CLI用户身份发布到飞书，访问限制已解除。沿用原文周报布局，在20261019前新增20261025节；18张表、正文与pipeline图片读回通过，历史内容完整保留。仅整理既有diffusion和Pair证据，未新增训练或评测。[本周节](https://rcntyhvf331j.feishu.cn/docx/QLALdzCrjoV5y1xLNB7cu4jxn0e#doxcnaCXbEjMDdx7AuVBJebbjff)、[发布验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/feishu_sync/verification.json)。

2026-10-25T22:04:09Z 本周周报以diffusion/edit76500已有核验产物更新，并整理Pair结构蒸馏pipeline及阶段性结果；Word/PDF、源图和替换行已交付，飞书原文和上周模板仍受登录访问限制。整理中观察到新版在线Pair已有八卡资格回执及首批正式训练日志，最新状态只维护于[项目进度](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。本轮未产生新的模型评测成绩、训练提交或平台操作。[周报草稿](/project_c20250601/251105016/project/dllm_test/docs/WEEKLY_REPORT_20261025.md)、[数字与版式验收](/project_c20250601/251105016/project/dllm_test/output/weekly_report_20261025/verification.json)。

2026-10-25T15:56:06.533546+00:00 在线Pair旧任务`t-20261025182020-v6lz2`已于2026-10-25 13:27:53 UTC结束，平台Killed；正式训练仍为step0/seen0、只有checkpoint0。最新loader修复版输出目录尚不存在，未见profile、正式metrics或新checkpoint，当前未观察到该组训练启动。修复包与8项CPU/真实四worker输入验收保持就绪；复用已完成MLM，教师目标不落盘。本轮仅查询，无submit/cancel。[状态快照](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/training_status_latest.json)；[修复版启动入口](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_loader_20261025/README.md)。

2026-10-25T12:50:56.972685+00:00 在线Pair任务`t-20261025182020-v6lz2`平台Running，但自10:36 UTC八rank均停在creating_loader/step0，未产出正式metrics。spawn子进程因本项目scripts命名空间与Protenix同名常规包冲突而导入失败；已将本项目scripts设为显式包，并同步冻结器。独立loader修复包8项CPU回归、原生路径旧失败/新成功对照、4worker读取128条真实输入和完整SHA校验均通过。原plan/source/checkpoint0保留，原公共Pair sh转接新包；新输出train_fix_loader_20261025，仍同seed42从零初始化，MLM复用，教师目标不落盘。新包未启动，GPU资格待重跑；当前任务未取消。本轮未submit/cancel。历史最大形状profile约83.12秒/512条（2.22万条/小时），正式速度尚未测得。[现场与速度](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/loader_failure_20261025/status.json)；[修复与启动](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_loader_20261025/README.md)。

2026-10-25T07:59:05Z 根据用户对BERT任务适配的质疑复核真实前向：无生成/预测回填。更正将表征与MLM诊断并列作整体结论的解释范围；未改成绩、协议或模型。[依据与边界](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md#stage1-bert-evaluation-boundary-20261025)。

2026-10-25T02:35:07Z 现有Stage1效果验证完成验收；三条主线结果与图表已发布，既有权重不变，结论保留混合/无明确收益。[逐项审计与结论](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md#7-完成审计与当前结论)。

2026-10-25T02:31:26Z 现有Stage1伙伴检索、区域恢复与事后敏感性全部完成，原推理/评分session真实exit0。三条评估主线均有完整结果；原始伙伴评分没有一致优势，局部增益读数与其他任务共同解释。冻结权重不变，无新预训练。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#stage1-native-partner-results-20261025)、[终态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25T01:54:37Z Stage1结构同源敏感性及全部输入独立核验完成，原始评估和权重不变；伙伴Pair仍运行。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#stage1-structure-homology-20261025)、[执行证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25T01:53:13Z 在线Pair任务`t-20261025092740-q7qjs`于01:43 UTC因采样器漏导入退出，平台Failed；原profile和8rank恢复通过、初始4096条验证通过，正式训练0步。修复为增加已有CompleteEpochBatchSampler导入；独立冻结包7项CPU回归及完整输入验收通过，正式run的首步更新、日志、checkpoint、停止/续跑及epoch尾批均覆盖。旧plan/源码/checkpoint0逐项SHA保留；原Pair sh转接新包，新输出从同随机初始化重跑资格，未提交。[修复与校验](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch_fix_sampler_20261025/repair.json)；[原始失败记录](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/failure_sampler_20261025/status.json)。

2026-10-25T01:40:27Z 现有Stage1实验结构检索完成：冻结模型、实际实验坐标、query/gallery与全部几何/指标验证通过，图与表已发布；解释限定为表示—几何秩相关改善，未超过强序列基线，Top1结构质量改善不明确。两项测试的近同源暴露审计亦完成。伙伴MLM全部候选已验收，Pair仍运行；整体效果验证尚未完成。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#stage1-experimental-structure-20261025)、[执行与审计](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25T01:33:39Z 首次确认用户提交的Protenix在线Pair任务`t-20261025092740-q7qjs` Running，八rank正在micro64联合profile，在线教师进度137–139条/卡。共享盘仅run.lock，无正式loss/profile完成/ckpt；近期无异常，教师目标不落盘。已将外部提交任务绑定至jobs.json并更新Active状态；本轮仅查状态，未submit/cancel。[状态快照](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/diagnostics/log_review_20261025/status.json)。

2026-10-25T01:28:00Z 已训练Stage1天然伙伴检索进入正式本机GPU推理；候选、mask和权重身份冻结，两个真实模型CPU/GPU及计数检查通过，串行运行。新增实验结构资料/几何与特征提取入口，实验CIF直接复用本地包，资料筛选不代表效果完成。所有主模型冻结，无教师调用、云端submit/cancel或新预训练。[协议、句柄、状态](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-native-partner-20261025)。

2026-10-25T01:24:24Z 按用户要求，在线实验复用已完成的纯MLM，只新增Pair KD。核对原scratch MLM已完成9770步/10epoch，终点16个checkpoint文件及验证统计SHA通过；与新计划的数据、主干初始化、MLM训练参数、核心mask/采样代码和验证seed一致。终点AA CE0.3026622425；基线引用及jobs选择已落盘，README/方案同步。旧MLM仅供比较，Pair仍从零开始；pllm与新联合环境不同，最终统一评测需复核。未改冻结训练plan/源码、旧结果或缓存，未提交/启动训练。[基线证据](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch/baseline_reference.json)；[本轮唯一训练入口](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch/README.md)。

2026-10-25T01:08:28Z 已训练Stage1两组最终9770步GDP冻结评估完成；完整Pair前向、同规格线性读出器、少样本曲线、逐预测复核、完全重复/近同源训练暴露检查均已落盘。收益混合，未宣称显著或跨链成功；配对/实验结构检索未执行。只用CPU，未改预训练权重、缓存或历史benchmark协议，无云端submit/cancel。[执行与来源](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#stage1-frozen-gdp-20261025)、[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#stage1-frozen-gdp-20261025)。

2026-10-25T01:01:10.792078+00:00 按用户“不存储教师feature”纠正接入方向：冻结AB500K在线A/C启动包，新增目标常驻内存、微批流式消费、独立层Pair及跨链置信度KL；学生参数75,601,176/76,578,136，从相同随机主干开始，global512/9770步。冻结包28项CPU测试通过（含2rank Gloo、完整学生恢复、teacher重算、三mask验证），原生Protenix CPU预处理禁止写文件下两次一致，完整输入/资产验收通过；没有新增教师缓存。GPU联合profile/八卡NCCL与科学收益未验证，本地GPU另有采样任务，无新submit/cancel/训练。[在线方案与状态](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PROTENIX_AB500K_CONFIDENCE_PLAN_20261025.md)；[启动包](/project_c20250601/251105016/project/dllm_test/output/stage1_protenix_online_ab500k_20261025/launch/README.md)。


2026-10-25T00:55:23.335610+00:00 按用户澄清，将本次任务限定为已有checkpoint的下游/自建任务评估。核对AntiBERTa2-CSSP、IgBert/IgT5、BALM、AbLang-2、ImmunoMatch、PROPHET-Ab、SCEPTR等一手来源；区分完整冻结评分/检索与仅拟合读出器，提出优先任务并标明自建适配，未宣称已验证Pair收益。现有fusion特征loader不能直接视为支持Stage1 Pair，需保留完整反馈前向；本轮无训练或评测提交。[已训练模型下游评估调研](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINED_MODEL_EVALUATION_REVIEW_20261025.md)。

2026-10-25 Protenix置信度蒸馏基础模块完成：23项CPU检查通过；原生PDE需要坐标，8例完整路径与trunk对照通过，保留上三角64-bin FP16及1/2/4Å置信度。四配对例带置信度第二次前向均值约0.75秒，非吞吐基准。1AKE链内长程PDE≤4Å平均预测0.948、真实比例0.659，表明不能将教师置信度直接等同标签正确性。损失按原pair数归约，低置信度可真正降低贡献；生产50万接入尚未完成。[候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PROTENIX_AB500K_CONFIDENCE_PLAN_20261025.md)。

2026-10-25 通用教师真实权重工程验证完成：同一本地A100上三模型各8例通过，24个二维分数文件验收通过；Protenix/Boltz完整与截断路径对照误差均0。ESMFold设备迁移、Boltz重复实体链重排、Protenix确定性设置均已在冻结启动包中修复并保留失败记录。四配对例前向均值1.430/0.547/3.633秒（ESMFold2/Protenix/Boltz）；仅小样本接入和标签合理性证据。评价更正：2HHB重复链交换可使Protenix跨链AP19由0.1228变为0.9934，须保留原始与等价映射两种结果。云提交未创建任务，本地进程均正常结束。[结果、限制与启动脚本](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_TEACHER_VALIDATION_20261024.md)。

2026-10-24 通用教师验证进入真实资产准备：冻结8例、取得官方Mini-ESM/ESM2-3B/Boltz2权重及21个原生CCD组分，独立venv避免训练依赖冲突。ESMFold CPU八例通过；初次云提交被旧队列零容量拒绝，无任务创建。后续状态见[验证记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_TEACHER_VALIDATION_20261024.md)，不把源码/CPU通过当GPU结果。

2026-10-24T23:04:07+00:00 通用结构教师可行性审计完成：15个仓库固定源码，补齐AF2/3、RFAA/RF3；确认Protenix/Boltz/AF3/RF3的距离头可从独立trunk结果读取，而AF2式坐标recycling不能整体删除。现有ESMFold2已trunk-only，历史504096条记录约167.44 worker-GPU小时为计时下界，不能用作新模型纯forward比较。核算完整distogram缓存与在线多epoch成本，保留三条优先联合路线及RF3/AF3备选。仅源码、文档和权重HEAD核查，无新教师GPU测量、模型加载、缓存生成或作业提交。[详细调研](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_GENERAL_STRUCTURE_TEACHER_FEASIBILITY_20261024.md)。

2026-10-24T21:16:05.107257+00:00 AB100K仅distogram候选已设计：10万训练＋2000dev＋2000test按既有长度外推FP16全方阵713.72GB，对称验收后上三角358.40GB，保留链内/跨链完整位置。候选A/B/C/D区分MLM、KD和独立层反馈；同随机主干、20epoch3920步筛查及50epoch9800步延长候选。主证据为冻结hidden的新实验结构接触探针和功能探针，KD仅诊断；原Intra/Inter及Mask-aware验证范围保留，严格划分/教师质量/三seed须实际执行。尚未抽样、生成缓存、实现新模型或提交作业。[AB100K候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB100K_REPRESENTATION_PLAN_20261024.md)。

2026-10-24T20:50:34.352551+00:00 原生Pair容量已核算：教师256维pair＋64维distogram，扫描504096条请求真实长度，500k训练平均H+L231.503。全有向L²、最终一份教师状态、16位未压缩本体为训练17.157TB、加验证17.297TB；FP32两者均原样存储则翻倍。合格OAS训练池实际2160824条，外推约74.145TB；400万同长度场景约137.253TB，非已核准全量AB计数。4096有向位置抽样为另选项，不静默替代全位置目标；学生8层共用教师最终目标无需8份缓存。未生成数据或提交GPU。[容量依据与范围](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_FEATURE_STORAGE_20261024.md)。

2026-10-24T19:41:59.284628+00:00 从零训练曲线已生成：MLM完成9770步，Pair日志至9275、验证与完整checkpoint至9250，尚无COMPLETE；本地最后心跳19:24 UTC，不据此断言仍在运行。共同9250步AA CE为MLM0.302923586、Pair0.302463270，相对改善0.151958%；链内KL从3.218140降至0.155180（下降95.18%）。核验随机初始化/数据/MASK/训练条件一致，恢复重放剔除17行，保留最新轨迹；PNG/PDF含完整曲线与后期放大，CSV及快照保存。单seed且Pair未完成，不作稳定收益或跨链结论；未改训练或submit/cancel。[整体曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/scratch_overview_zh.png)、[链内KD](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/scratch_kd_zh.png)、[数值快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/curves_20261024_1945/summary.json)。

2026-10-24T19:08:10.020221+00:00 scratch进度核对：纯MLM（原scratch目录）已成功9770/9770、10epoch，终点AA CE0.302662；优化Pair最后记录9200/9770（94.17%），但云任务t-20261024231915-9vjtx于18:24:56Z已Killed，最新完整checkpoint9000，尚无COMPLETE。确认8rank profile及恢复短测通过，micro64/global512、reserved峰值50.541GiB，9050–9200区间约1.512秒/步。共同9000步AA CE：MLM0.303221、Pair0.302593，Pair相对低0.20697%，仅暂时结果。Pair计时因恢复为下界；终止原因尾日志未明，不推断人为或平台原因；本轮只读，无submit/cancel/resume。[只读进度快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_scratch_fast_20261024/diagnostics/status_20261024/snapshot.json)。

<!-- stage1-scratch-fast-20261024 -->
2026-10-24T12:36:34.657206+00:00 Pair速度优先候选已从RAM原型接入正式结构实现，独立从零训练包完成。50项CPU检查覆盖旧行为、候选逐值一致、非零全模型反馈、硬mask、完整状态恢复和两rank尾批；冻结包10项复测通过。首次冻结测试用stdin无法spawn子进程，改为文件入口后通过，失败日志保留；非模型失败。两组CPU入场及shell/YAML/SHA核验通过，共享78state指纹与旧scratch一致。此前A10080局部约1.20秒/更新、50.11GiB仍是候选测量，八卡性能/恢复待作业内验证；未训练或云端submit。[优化版启动与核验](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_FAST_20261024.md)。
<!-- /stage1-scratch-fast-20261024 -->

<!-- stage1-contact-residues-20261024 -->
2026-10-24T11:33:55.652866+00:00 按用户要求将教师热图落实到具体残基对：选择规则完全依据实验几何，1JHL H104A对应L34A/L46L/L49Y/L50S/L89Q/L91H等6对，距离3.687–7.703Å，教师短距离分数0.7516–0.9970；同H的6个≥12Å对照均<0.0004。另检查1MLC H102N及1VFB H45L，全部162对接触特征落盘；对称256通道和64类分布逐行对齐，大图保留准确编号，修复字体后无警告。局部案例、控制组未匹配L的AA身份，不作新训练或泛化结论。[接触残基与256维教师特征](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CONTACT_RESIDUE_FEATURES_20261024.md)。
<!-- /stage1-contact-residues-20261024 -->

<!-- stage1-cross-chain-diagnostic-20261024 -->
2026-10-24T08:13:09.252819+00:00 完整跨链诊断完成：seed20261024先固定六条valid及1JHL/1MLC/1VFB，补跑同revision教师全矩阵，与继承BERT287000的Pair step0/3000比较；教师AP 0.9692、学生0.00855，链内maskedKL改善而跨链全例退化。新增实验结构真值、clean/masked视图、18/19/20bins敏感性、Pair PCA及211个预测AA的伙伴上下文干预。结论限定当前链内KD模型与这些案例，保持全4096×3MASK无MLM收益的已有判断；不将PCA纹理当结构证据。5组图与指标持久化，完整张量因quota留/tmp，27份检查通过；无训练/云提交。[完整矩阵、实验接触与Pair特征诊断](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_CROSS_CHAIN_DISTOGRAM_DIAGNOSTIC_20261024.md)。
<!-- /stage1-cross-chain-diagnostic-20261024 -->

2026-10-24T08:04:15.397210+00:00 按用户“每层一个Pair网络”补充官方源码复核：ESMFold ModuleList、AF2/AF3 layer_stack均支持层级独立参数，并持续传递pair状态；Uni-Mol提供较轻量的逐层关系更新参照。建议独立PairUpdate＋残差z作为下一版候选，保留仅解除共享的U对照；CPU实例化核对8个encoder均120832、合计966656参数且存储独立，指定U总模型约76.58M。已将旧文档的父BERT初始化建议改为遵循用户要求的同一随机主干。独立权重不自动提速；有向/三角更新不能照搬对称加速。仅研究与CPU参数核算，未改训练或提交任务。[参考及适配方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md#8-针对每层有一个pair网络的官方实现复核)。

2026-10-24T07:47:46.897119+00:00 AB500K Pair单卡速度探索完成：A10080、micro64/长度310，两批真实输入确认原版约2.46秒/更新；对称计算＋提前dtype转换约1.43–1.45秒、allocated5.44GiB，再关闭Pair分块重计算约1.20秒、allocated50.11GiB。8项CPU测试及两批非零反馈完整模型输出/梯度检查通过；编译候选未过预设logits门槛并保留失败结果。单卡更新速度约1.7/2.05倍，不代表八卡端到端性能。已保存脚本、原始测量、证据SHA和PNG/PDF；未改scratch冻结包或提交云端任务，八卡资格待正式接入。[性能诊断与候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_SPEED_EXPLORATION_20261024.md)。

2026-10-24T07:24:30Z Stage 1目标偏差更正：原稿第10节A–F为最小验证范围，已完成AB500K只验证A和D的Intra-only版本；B/C、Intra+Inter增量及Mask-aware均未被该实验覆盖。已在原始方案与AB500K文档明确范围和缺口，保留历史冻结参数，不将链内试验未获MLM收益外推为跨链蒸馏无效。现有FP16/UINT16/Zstd缓存全索引504096条，每条2048个链内位置；train/valid负载各抽查一条，HL均0且无raw Pair State，正式缓存加辅助文件108.199GB。完整真实链序列和位置映射保留；跨链/raw latent需新生成目标。未重做全负载验收，未改训练包/缓存，未提交或取消GPU任务。[核查证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/supervision_scope_audit_20261024.json)。

2026-10-24T07:24:38.997184+00:00 按用户要求准备AB500K从零训练：两组旧学生权重继承0，直接随机初始化seed42，共享78个state张量逐值一致，2条真实样本初始logits一致；MLM/Pair参数75.60M/75.73M，保留当前共享Pair架构及已验收缓存。新独立包默认9770步=10epoch、主干/Pair LR均1e-4、warmup500；23项CPU测试、两组84分片/缓存入场、冻结源码和shell/YAML核验通过。GPU profile/8rank短测待用户提交，本轮无GPU或云端submit。[新计划与两个启动命令](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_SCRATCH_PLAN_20261024.md)。

2026-10-24T07:09:47.671073+00:00 Pair架构探索完成：对照Graphormer、Uni-Mol、ESMFold v1、AF2/AF3原论文及官方代码，区分跨层权重共享、Pair状态传递、recycle复用。当前共享动态bias缺少显式跨层z状态；共享本身未证实是退化原因。优先候选为低成本持久Pair状态与独立层级更新，并保留仅解除共享的对照；后者按指定构造估计总76.58M，不天然加速。候选未实现或提交训练。[架构对照与实验设计](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_PAIR_ARCHITECTURE_EXPLORATION_20261024.md)。

2026-10-24T06:48:07.307840+00:00 AB500K两组3000步均COMPLETE=success（各约3.07epoch）；固定4096条×3mask终点AA CE：MLM 0.288281123、Pair 0.289064492，Pair相对高0.271738%。KD从3.414345降至0.584299（−82.887%），但两组最佳AA CE仍是step0；本轮没有MLM收益。训练会话耗时MLM20.74分钟、Pair150.52分钟（7.26倍，不含教师生成及启动资格）。已刷新[完整中文曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/loss_explained_zh.png)及[共同终点数据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_completed_20261024/summary.json)。仅本地读取与绘图，未改训练或提交作业。

2026-10-23T21:57:29.079264+00:00 AB500K中文曲线及机制复核完成：21:51:44 UTC快照MLM3000、Pair800；共同验证750时Pair AA CE比MLM高0.062796%，教师KD由3.414345降至1.304292（−61.80%），尚无MLM收益。checkpoint750的4条CPU探针确认8层反馈已更新且梯度连通，加权KD/MLM主干梯度范数约2.81%，仅为小样本线索。提出KD-only/feedback-only对照及单因素Pair容量、跨层状态候选；未改训练或提交GPU。中文PNG/PDF、158行训练CSV、17行验证CSV及数值核验通过。[Pair特征与中文曲线复核](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_PAIR_FEATURE_REVIEW_20261023.md)。

2026-10-23T21:40:24.124992+00:00 AB500K长训预算讨论已核算：500000/global512保留尾批时每epoch977步，5epoch=4885、10epoch=9770、20倍总60000=61.412672epoch；按8卡现速总时长约MLM6.9小时/Pair50小时。建议当前Pair完成3000后，两组匹配续接到总9770先看趋势，主干/Pair峰值1e-5/1e-4、500新增步再warmup为候选；尚未改运行计划或生成/提交长训作业。[候选方案](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md#长训预算与学习率候选2026-10-23讨论方案)。

2026-10-23T21:34:37.469844+00:00 AB500K本地loss曲线已生成：纯MLM已完成3000/3000（COMPLETE=success），Pair/KD快照475/3000；两组共同step的MLM loss几乎重合。MLM固定验证AA CE初始0.288198632、最终0.288281123，best仍step0；Pair仅到step250验证，尚无训练收益结论。新增只读绘图脚本，输出完整/共同step训练曲线、验证/蒸馏分项图、PNG/PDF、原始CSV及冻结快照；粗线为5个已记录批次的移动平均，验证不平滑。数据行数、目标函数分解及图像检查通过，无训练代码/配置修改或GPU提交。[训练曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/training_loss_curves.png)、[验证及KD曲线](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/validation_and_kd_curves.png)、[数据快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/loss_curves_20261023/snapshot.json)。

2026-10-23T21:19:56Z 按用户要求回填diffusion/edit76500的全部15任务成绩，两份RESULTS文档共各21个表项；指标/预测SHA、模型身份、表格数值及原内容保留核验通过，PROGRESS同步。[发布验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step76500-local-20261023)。

2026-10-23T21:18:52.244398+00:00 修复版AB500K两组均已进入正式训练：MLM step1375/3000、Pair/KD step150/3000，最新保存点分别为1250/0。两组实际micro64×8卡/global512；checkpoint张量实数为75,601,176/75,732,312参数，Pair新增131,136。保持同一父BERT287000、50万训练子集、seed42；原首步故障已越过，训练收益仍待固定验证对比。本次只读核对，无submit/cancel或训练配置修改。[配置与进度快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/training_configuration_20261023_211852.json)。

2026-10-23T21:16:54.293844+00:00 修复版AB500K纯MLM八卡实跑已推进到step1050/3000，累计537376条；最新完整保存点step1000。日志步数递增，loss/梯度均有限；保存回执8rank参数哈希一致且相对step0已变化。profile与恢复smoke均passed/passed。固定4096条×3组mask验证AA CE从0.288198632到step1000的0.288470148，best_step仍0，尚未显示验证收益。原首步日志故障已在该MLM正式循环中越过；本次未核对Pair/KD，未submit/cancel或修改训练代码。[实跑快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/repair_mlm_progress_20261023_211654.json)。

2026-10-23T21:15Z 核验diffusion/edit76500单卡15项评测全部成功，15:32:50 UTC结束；采样审计、18份指标及最终汇总一致，已同步完成状态，尚未回填RESULTS表。本轮无GPU启动或云端submit/cancel。[终态验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step76500-local-20261023)。

2026-10-23T20:51:04.744462+00:00 旧AB500K纯MLM首步失败原因已由真实rank0堆栈确认：冻结run_stage1_ab500k.py:605的metrics=dict(arm=arm, **state, **scores)报selected_tokens重复关键字TypeError。结合源代码顺序，rank0在更新后写首步日志时失败，其他rank进入下一步ALLREDUCE并在3小时后超时；只有checkpoint0，无metrics。此证据支持日志异常导致本次分布式停滞，fork仅作为独立风险保留。修复入口launch_training_fix_20261023已修正该语句及跨rank异常传播，17项CPU测试/真实读取/CPU入口通过，新八卡实跑待执行。未submit/cancel。[真实rank0错误](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/code_review_20261023/mlm_rank0_failure.log)。

2026-10-23T20:50:05.632947+00:00 旧AB500K纯MLM真实故障日志已出现：20:47:24 UTC ranks1–7报告NCCL ALLREDUCE(seq82，3元素)等待10800000ms后超时，20:48:24报watchdog异常和ChildFailedError。最新平台仍Running，但训练子进程已失败；最后保存仍17:47:22 UTC的checkpoint0，无metrics。8rank恢复smoke与step0参数一致性通过，不代表正式循环通过。最初卡住/异常rank的Python栈未取得，不能将timeout单独归因为网络故障；已确认日志重复关键字缺陷及fork风险已在独立修复包处理。修复输出目录尚不存在。本次无submit/cancel。[最新故障证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/code_review_20261023/mlm_checkpoint_followup.json)。

2026-10-23T20:46:48.470829+00:00 AB500K训练代码复核发现并复现首步日志dict重复关键字TypeError；原NCCL后4worker默认fork也存在死锁风险。已修复为区分累计/批计数、spawn＋120秒读批超时、阶段日志及异常广播，17项CPU测试和两组各128条真实数据多worker读取通过。新冻结入口launch_training_fix_20261023使用相同父BERT/seed/配方，独立train_fix_20261023输出；旧两组仅checkpoint0，新包首次从相同初始化重开，后续本包可续训。旧MLM平台仍Running、Pair/KD此前Killed，故障实时栈及Killed原因未确证；未submit/cancel。[复核与修复](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_TRAINING_REVIEW_20261023.md)。

2026-10-23T20:24:46.276626+00:00 AB500K两组并未正常推进：MLM任务t-20261024014536-ccpc2平台Running，但自17:47:22 UTC仅checkpoint0，无metrics/首步日志，疑似卡住；Pair/KD任务t-20261024034159-7zsjv平台Killed（19:46:13 UTC），保留19:45:09的checkpoint0。两组8卡profile、恢复smoke和step0验证已通过；无正式优化step1完成证据，不能认为正在正常训练。停止/卡住原因尚未确认，top查询bad handshake；本次未cancel/submit或改冻结代码。[核查证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/training_status_20261023_2025/snapshot.json)。

2026-10-23T19:27:53.683274+00:00 核实tmux完整缓存验收于2026-10-23 19:00:10 UTC通过（exit0）：500000 train＋4096 valid，共504096条、1032388608对，最终SQLite索引和manifest已发布。结构质量仍unvalidated；本轮仅完成工程完整性验收。后续Pair/KD使用launch_mlm_independent/train_pair_kd.sh，与独立MLM共用冻结plan/source，单节点8卡，3000步/global512，启动时仍需GPU profile及8rank短测。本轮未提交GPU训练。[完整验收回执](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/local_cache_validation.json)。

2026-10-23T17:33:39.750799+00:00 已准备独立MLM启动包，纯MLM不再检查教师权重/缓存，保留84分片、50万/4096子集及父BERT287000完整校验；真实CPU入口在cache_ready=false时passed。Pair/KD仍要求完整缓存。两组训练及报告统一使用[新入口目录](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/launch_mlm_independent/README.md)，共用新冻结plan/source以保持对比约束；MLM缓存哈希记null，D组报告仍绑定真实cache SHA。原冻结包及tmux缓存验收不改，未启动GPU训练。

2026-10-23T17:04:18.478319+00:00 按用户要求已在本地tmux会话ab500k_cache_check重新启动完整CPU缓存验收（启动2026-10-23T17:03:59.649956+00:00）。教师缓存504096条沿用；旧临时索引保留，新索引从头构建。验证日志与真实退出码按尝试独立保存，进度每30秒更新；尚未通过验收，未启动GPU或学生训练。[实时状态与最终回执](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/local_cache_validation.json)。

2026-10-23T16:59:17.905653+00:00 本地CPU缓存验收已中断：验证及观察进程均不在，最终cache_index/manifest.json不存在。最后进度为487078/504096（96.62%，15:55:37 UTC），临时SQLite最后修改15:56:01 UTC；日志无终态，原因未知，不能视为通过。教师缓存仍有504096条，学生训练未启动。现有索引构建无部分续接能力，重跑需要重新全量验收，不需重新生成教师特征。[状态证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/local_cache_validation.json)。

<!-- stage1-ab500k-local-cache-check-20261023 -->
2026-10-23T14:15:31.728671+00:00 已按用户要求启动本地CPU完整check_cache（PID430996，14:07 UTC启动），公共输入预检通过，正在逐条校验504096条缓存并构建SQLite索引。后台观察器每30秒更新[实时进度与最终回执](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/local_cache_validation.json)，进程结束后核对最终成功输出、manifest计数和SQLite quick_check；目前尚未验收完成。缓存104.186GB，辅助数据3.762GB；两组学生训练尚未启动。
<!-- /stage1-ab500k-local-cache-check-20261023 -->

<!-- stage1-ab500k-teacher-complete-storage-20261023 -->
2026-10-23T14:03:10.564638+00:00 AB500K教师特征全部生成：504096/504096条、16/16完整回执绑定通过，最后worker于14:00:01 UTC结束。teacher_cache实占104.186GB（97.031GiB），子集/request辅助文件3.762GB，合计107.948GB（100.534GiB），不含后续索引及学生checkpoint。cache_index尚不存在，A/D训练目录尚不存在；下一步先check_cache汇总索引并全量验收，再分别从BERT287000续训纯MLM和全8层Pair反馈＋链内KD各3000步/global512/8卡。本次只读统计与完成回执检查，不等于已做全负载缓存验收，没有启动训练。[完成与存储证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/completion_storage_20261023_140311/snapshot.json)。
<!-- /stage1-ab500k-teacher-complete-storage-20261023 -->

2026-10-23T13:59:28Z 按用户要求启动最新diffusion/edit76500的本地单卡15项评测；独立快照、CPU/CUDA和首任务身份核验通过，CDR已进入运行。尚未产生整套终态或发布新成绩。[运行与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step76500-local-20261023)。

<!-- stage1-ab500k-three-complete-20261023 -->
2026-10-23T13:43:55.102463+00:00 AB500K已封存497380/504096条（98.67%）。任务1/2/3均完成126024条，12个worker完整回执与冻结请求/teacher/pilot绑定通过；2/3平台已Success。任务0已于13:07 UTC进入第6次会话，4个worker心跳均在60秒内，已封存119308条，剩余6716条按可恢复计数，正在续跑，无需重复提交。未封存尾片未计入，cache总索引尚未生成，学生A/D训练目录尚不存在。[最新快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/progress_20261023_134355/snapshot.json)。本次只读核查，无GPU启动或云端submit/cancel。
<!-- /stage1-ab500k-three-complete-20261023 -->

<!-- stage1-ab500k-progress-interruption-20261023 -->
2026-10-23T12:28:24.898228+00:00 AB500K已封存490034/504096条（97.21%），cache总索引尚未生成。任务1已完整完成；任务2平台Running且3/4worker完成；任务3平台Running持续生成。任务0的新续跑`t-20261023180449-52m9k`于11:44:27 UTC Killed，保留115212/126024条，需用户继续用原teacher_cache_0.sh续跑。部分timer残留running不代表存活；以平台终态与停止更新结合判断。剩余各组数量与未封存尾片口径见[缓存快照](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/progress_20261023_122412/snapshot.json)；[当前平台任务](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/progress_20261023_122412/platform_status.json)。本次只读核查，无提交、取消、重启或缓存修改。
<!-- /stage1-ab500k-progress-interruption-20261023 -->

<!-- stage1-ab500k-resume-20261023 -->
2026-10-23T08:41:44.923064+00:00 AB500K中断恢复点核查：已封存392807/504096条，teacher_cache_1平台Success且完整126024条，0/2/3分别90636/89584/86563条。只需用户按原4卡shell续跑0/2/3；1无需重提。原四任务均终态，已从Active表移除；0/2/3实际缓存含后来恢复会话，不把原Killed时间当作最新停止时间。代码/入口/索引/pilot与完成回执只读检查通过，正式恢复仍需payload校验。本次未submit/cancel或运行GPU。[恢复证据](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/resume_20261023_0744/cache_resume_snapshot.json)。
<!-- /stage1-ab500k-resume-20261023 -->

<!-- stage1-ab500k-live-eta-20261022 -->
2026-10-22T22:11:58.107432+00:00 AB500K首次稳定窗口ETA：16worker最近约10分钟合计14.126条/秒，最晚预计2026-10-23 07:55 UTC完成教师缓存，当前约剩9.7小时。估算不含cache索引验收或学生训练；核实实际4个网页作业Preemptible=true，与准备YAML false不同，故估计以任务不中断为条件。[实测及假设](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/eta_20261022_2211/estimate.json)。
<!-- /stage1-ab500k-live-eta-20261022 -->

<!-- stage1-ab500k-four-running-20261022 -->
2026-10-22T22:00:21.938793+00:00 AB500K四个用户任务已实际Running，worker0–15均有日志与缓存增长，2048条pilot全部通过；最高reserved13.055GiB，teacher缓存投影104.106GB。21:59已发布6144条，未发布尾片另计。确认缺挂载后的重提已开始计算；两组学生训练未启动，无loss收益结论。本次只读核查，无submit/cancel。[四任务](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/status_20261022_2158/platform_status.json)；[缓存](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/status_20261022_2158/cache_snapshot.json)。
<!-- /stage1-ab500k-four-running-20261022 -->

<!-- stage1-ab500k-missing-mount-20261022 -->
2026-10-22T21:28:05.210793+00:00 用户首个AB500K四卡任务`t-20261023052313-45khp`因缺少项目盘挂载而在bash阶段失败；脚本在21:15已存在，失败日志21:23:57，未运行教师。实际云配置仅personal/public目录，现有六份YAML的VepfsId/SubPath/MountPath均正确。此前四条启动命令未说明挂载前提，现已补齐README、生成器文档模板与操作指南；冻结源及plan不改，无自动提交或取消。[诊断](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/diagnostics/mount_failure_20261022/failed_task_mount.json)。
<!-- /stage1-ab500k-missing-mount-20261022 -->

<!-- stage1-ab500k-offline-pilot-20261022 -->
2026-10-22T21:17:17.138042+00:00 AB500K离线小规模验证准备完成：500000/4096子集全量对齐核验通过；最大教师长度261、学生固定310，16worker互斥请求，总pilot2048，辅助文件约3.762GB。保留现有8层Pair架构和BERT287000，A/D各3000步/global512，缓存150GB上限。已冻结8个sh与6份GPU YAML，33项CPU测试通过，包含不等长DDP累积、恢复及初始best0保留；teacher实际CPU入口通过，训练缺缓存拒绝符合预期。GPU缓存/训练尚未执行，无loss收益结论，用户自行提交。[计划与交付](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_AB500K_OFFLINE_PILOT_PLAN_20261022.md)。
<!-- /stage1-ab500k-offline-pilot-20261022 -->

<!-- stage1-timing-diagnosis-20261022 -->
2026-10-22T19:27:24Z Stage1只读速度诊断：20个完整批、7275条metadata，短输入约1.1秒，近千残基输入73–77秒；第58批卡间目标完成相差80.22秒。前三步教师准备区间重建占87–89%，初始准备约18.1分钟，前79批平均约111.3秒/批。计时含CPU/IO且非算子profile，运行包及训练未改。[完整证据](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md#训练速度诊断2026-10-22)。
<!-- /stage1-timing-diagnosis-20261022 -->

2026-10-22T18:53:17Z 用户改选最新diffusion checkpoint，已固定准备时latest step43000并完成单卡15任务评测包及CPU门禁；先前best2500包保留，未执行GPU或提交作业。[当前入口、选点与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-latest-single-gpu-20261022)。

<!-- stage1-lmdb-feasibility-20261022 -->
2026-10-22T18:22:17.466028+00:00 本地临时LMDB＋FP16/字节重排/Zstd64条往返全部通过；35.73MB原负载→13.00MB库，推算含页开销734GB。仅本地单进程写入/重开随机读取，非8卡或GPFS资格检查；训练未改变。[本地LMDB往返验证](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/lmdb_compressed_cache_poc_20261022.json)。
<!-- /stage1-lmdb-feasibility-20261022 -->

<!-- stage1-cache-compression-20261022 -->
2026-10-22T18:16:48.111459+00:00 缓存只读CPU可行性试验：64条，全部Zstd/字节重排往返通过；FP16目标KL均值1.30e-6、最大9.54e-5，压缩负载外推715GB。仅编码样本分析，不代表训练收益或迁移验收；训练/缓存未修改。[64条缓存压缩核验](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/cache_compression_feasibility_20261022.json)。
<!-- /stage1-cache-compression-20261022 -->

2026-10-22T18:06:12.868Z diffusion/edit best2500下游单卡包已准备，CPU真实权重及ANARCI/ImmunoMatch依赖检查通过，15任务仍pending；用户要求自行提交，未启动GPU或提交资源。[入口、范围和验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-best2500-single-gpu-20261022)。

<!-- stage1-online-started-20261022 -->
2026-10-22 用户启动的在线Stage 1通过8卡同卡profile（micro16/global512，联合峰值估算64.21 GiB）及8卡在线缓存/恢复短测（Adam最大差0）；完成step0基线验证（889条MLM loss0.215225、MASK准确率94.12%；761条链内KL2.689849），已进入正式优化。截至2026-10-22T17:06:05.895795+00:00，已记录第2个优化步；最新MLM loss 0.212827、链内KL 3.367364、梯度范数0.108929均有限，结构监督359/512（70.12%）。已记录的2步生成721条教师缓存；单步墙钟耗时为72.8 / 148.3秒（包含在线教师；teacher_seconds是八rank求和，不能当墙钟时间）。当前为KD warmup，inter=0表示无跨链目标；尚不足以判断收敛。 本轮未修改冻结运行代码，未submit/cancel。[训练指标](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/train/metrics.jsonl)、[最新日志核验](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online/log_review/latest.json)。
<!-- /stage1-online-started-20261022 -->

<!-- stage1-online-20261022 -->
2026-10-22T16:27:34Z Stage 1按需生成/原子缓存/复用已实现；39项冻结CPU测试通过，包含竞争写入、缓存损坏、重启命中、学生RNG、故障退出和分布式更新恢复。CPU入口验证84分片、BERT及教师6权重分片和真实教师运行时。单卡A10080真实权重联调：首次生成1条（2048个pair），再次命中0次推理，目标与pilot完全一致；BERT完成一次MLM+KD更新，8层Pair Bias与主干均获梯度，教师进程退出。初版联调发现Python软链接resolve误选学生环境，已修复并增加运行时预检；仅使用launch_online_v2。新的同卡最大长度profile与8卡在线恢复短测将在同一提交作业内执行，尚未产生该两项结果；无Volc submit/cancel，长训未启动。[本地联调](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online_local_verification_v2/verification.json)、[测试](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/online_cpu_tests_v2.xml)、[操作说明](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_ONLINE_DISTILLATION_20261022.md)。
<!-- /stage1-online-20261022 -->

<!-- stage1-validation-complete-20261022 -->
2026-10-22T15:30:29Z 用户运行的Stage 1合并作业工程通过：8rank恢复噪声精确、Adam矩差0；teacher8worker/1529条缓存/2,257,088对通过，固定seed重跑及对称差0，教师最大allocated23.22 GiB。训练缓存仅768/6,249,956，正式结构质量与覆盖仍待完成；本轮核对回执哈希，未启动长训。日志改为常规60秒汇总、新训练默认100步指标，14项日志及流程测试通过；保留既有冻结教师脚本与回执。[核验](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/validation_review_20261022.json)、[日志测试](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/logging_tests.xml)。
<!-- /stage1-validation-complete-20261022 -->

<!-- stage1-combined-checks-20261022 -->
2026-10-22 将Stage 1八卡恢复、教师pilot、缓存索引和工程审计合并为一次提交，复用现有冻结入口与教师权重，无新的模型/训练配方变化。7项CPU测试通过，覆盖损坏/缺失缓存、恢复结果复用与拒绝、阶段失败停止及工程/结构质量状态区分。未提交GPU或生成新教师推理结果，实验结构质量仍未验证。[统一入口](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_checks/entrypoint.sh)、[测试](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/combined_checks_tests.xml)。
<!-- /stage1-combined-checks-20261022 -->

<!-- stage1-profile-smoke-20261022 -->
2026-10-22 核验用户运行的Stage 1八卡最大长度profile：micro64/48 OOM，micro32 reserved77.58 GiB余量不足，micro24八卡均通过3次Adam更新，allocated56.26/reserved65.05 GiB。按实际回执生成独立 `launch_smoke` 包，84分片/父权重/资格CPU预检通过，尚未执行GPU恢复验证；本次未提交或取消平台作业。该profile只验证工程分配，不产生结构或预训练效果结论。[回执](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/profile/COMPLETE.json)、[新smoke入口](/project_c20250601/251105016/project/dllm_test/output/stage1_training_20261022/launch_smoke/entrypoint.sh)。
<!-- /stage1-profile-smoke-20261022 -->

<!-- stage1-training-build-20261022 -->
2026-10-22 Stage 1从单设备pilot扩展到完整训练管线；两进程CPU/Gloo对照验证不均匀microbatch、零KD rank及模型/Adam/RNG下一步精确恢复，学生13项+独立教师3项通过。84分片预检与6分片教师权重校验通过；四类YAML已构建但本轮无Volc submit/cancel。新GPU profile、8卡恢复、教师质量和正式收益仍未验证。[Stage 1训练构建与提交](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_TRAINING_PIPELINE_20261022.md)。
<!-- /stage1-training-build-20261022 -->

<!-- stage1-local-engineering-20261022 -->
2026-10-22 Stage 1本地工程验收通过：40项测试、真实ESMFold2-Fast全部权重、真实BERT＋KD三步更新和精确断点恢复；2048-token前后向单样本BF16峰值3.15 GiB（不含Adam），正式收益实验未开展。加载时发现并按官方release语义修正Fast禁用MSA分支的兼容问题；固定seed并开启确定性计算后教师重放差为0。[完整记录](/project_c20250601/251105016/project/dllm_test/docs/STAGE1_STRUCTURE_IMPLEMENTATION_20261022.md)。
<!-- /stage1-local-engineering-20261022 -->

2026-10-22T10:22:08.761565+00:00 用户网页新任务 `t-20261022181600-wcfcx`（`cq9rf89httddx5ka`）平台Running；八rank已于10:18:49 UTC从完整checkpoint28000恢复模型、AdamW、scheduler与各rank RNG，恢复回执8/8通过。当前观察28075/200000，world_size=8；新checkpoint28025的16文件SHA与八rank参数一致已验证。沿用W&B `diffedit287k200k1020`；服务端旧日志游标28201，高于恢复点28000，故暂时忽略较小step的重复日志；本地metrics继续保存且训练更新正常，超过该游标后继续显示新点。每60秒只读监控已绑定当前任务，依赖开发机进程存活。[监控](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/monitor_status.json)、[新保存点验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/restart_checkpoint_verification.json)。

2026-10-22T10:08:17.985973+00:00 按用户要求完成当前存储路径的实际迁移：入口、Conda、训练源码、数据manifest/binding/index、checkpoint provenance/receipt均使用 `/project_c20250601/251105016`；删除入口的旧挂载链接逻辑和路径适配模块，使用原标准数据门禁。禁止旧挂载访问的同一sh完整预检通过，旧路径访问0次；84分片全量哈希、父BERT287000、两份完整checkpoint28000校验通过，全部84分片reader/collator抽读及17项CPU回归测试通过。权重、AdamW/scheduler/8-rank RNG文件哈希与训练配方保持，累计目标200000。上次网页任务 `t-20261022172952-vshml` 已Failed，本轮未提交或启动GPU训练，需在网页用同一入口重新提交。[迁移验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/storage_rename_20261022/verification.json)。

2026-10-20 用户要求评测当前diffusion/edit任务的下游生成与表征。已固定请求时最新完整step16500，沿用本地偏好启动15入口；完整模型绑定、CPU/CUDA门禁通过。任务协议/状态统一维护于[下游执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step16500-20261020)。

2026-10-20用户明确要求diffusion/edit至少200000步；已从latest10000恢复完整状态，04:06:09 UTC核实10050/200000和完整checkpoint10025。13项冻结CPU测试、八rank AdamW／RNG恢复及更新重放通过，起点固定验证完全一致；旧best2500保留，W&B新run在线。原八卡容器接续，无新资源提交。[扩展配置与证据](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-19 CoV-UniBind作者原划分核查完成：原fold未公开于已核查资产，官方分数重算发现Binding聚合仍不精确一致。新实验草稿撤回，未训练。[证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-author-split-audit-20261019)。

2026-10-20核查BERT287000→条件diffusion/edit正式10000步已成功完成，终点为10月19日23:37:20 UTC，exit0和W&B finished一致。内部验证best2500／latest10000的macro CE为1.104110／1.124632；六来源继续改善、TRAIT回升拉高均值，编辑正确token误改仍需关注。原八卡容器standby；本次无新训练或下游评测。[配置、结果与终态证据](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-19 CoV-UniBind整体五折评估完成，逐行覆盖与独立计算验收通过；缓存缺失组合未补造分数。[执行与产物](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-pooled-5cv-20261019)。

2026-10-19 CoV-UniBind Ag+H+L Ridge五折重评完成：8组完整缓存均已训练和验收，结果写入两份任务结果表；本次为CPU缓存评估，无平台作业变更。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#covunibind-aghl-ridge5cv-20261019)。

2026-10-18T18:09:27.498029+00:00 已在 [launch_oas_resume.py](/project_c20250601/251105016/project/dllm_test/scripts/research/launch_oas_resume.py) 为实际导入的冻结 `FixedChainConfig` 安装 `to_dict=dataclasses.asdict` 运行时适配；安装前校验源码路径，启动回执记录源码SHA，`--check-only`覆盖适配和已导入模块的源码清单。原冻结源码文件和模型字段未改。两项真实W&B离线回归分别针对工作区和冻结源码，均先复现原异常，再通过on_train_begin/on_log/on_train_end，包含错误源码拒绝、重复安装、CPU隔离检查。checkpoint-11000全部17份文件SHA匹配，r3新runtime/run名及launcher/spec哈希、YAML与bash语法通过。r1/r2旧spec绑定旧launcher哈希，不再使用；r3沿用300k/global512、原数据和模型。本轮修复及准备完成，未提交/取消GPU任务，分布式续训尚未验证。[检查证据](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261018/verification.json)；[新配置](/project_c20250601/251105016/project/dllm_test/train_jobs/v5_fixed_chain_75m_oas_resume11000_8gpu_queue012_20261018.yml)。

2026-10-18T18:05:38.009014+00:00 查询确认 r2 `t-20261017122656-nzn5w` 于2026-10-18T04:26:08Z Failed，已移出Active表。r1/r2完整可取平台日志均显示 rank0 在 `WandbCallback.setup` 调用 `model.config.to_dict()` 时抛出 `AttributeError: 'FixedChainConfig' object has no attribute 'to_dict'`；其他七rank的Gloo RNG集合通信1800s超时为继发错误。原scratch关闭W&B，续训launcher开启后触发；更换entity/runtime没有修复接口。两次续训新增0步，11845为原scratch历史记录，完整恢复点仍11000。本轮只诊断并同步证据，未改训练源码或提交任务。[原始日志](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261018/r2_platform_failure.log:633)；[诊断](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261018/failure_review.json)。

2026-10-18 BERT122070/166000部分下游成绩已完成保存预测重计分及聚合复核；两组继续本地评测。[执行与证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step166000-representation-20261018)。

2026-10-18 用户指定BERT122070的下游评测已在本机启动，与166000同协议；八卡训练未变更。[执行与证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step122070-representation-20261018)。

2026-10-18 按用户要求开始BERT166000下游表征评测，沿用前两checkpoint协议、本地执行、跳过AbBiBench。快照和CPU/CUDA检查通过；未改八卡训练。[执行证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-step166000-representation-20261018)。

2026-10-18 BERT内部开发最佳选点已由48828更新为第10轮122070；仅按训练期间固定开发探针选点，非下游测试择优。后续第11—13轮未刷新最佳，训练按300k计划继续。本轮只读核查及状态同步，无作业变更。[进度与原始开发分数](/project_c20250601/251105016/project/dllm_test/output/bert_full_token_v7_300k_20261017/review_20261018_1443.json)。

2026-10-18 BERT48828/107000榜单同步遗漏已更正：已验证的结果回填各任务原有表格，T2A列错位修复，历史模型成绩保留；本轮没有新实验或重新计分。[更正与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#bert-task-table-sync-20261018)。

2026-10-17 旧V5/fixed112000→LoRA3000完整测试暴露审计及GDPa1层级CPU诊断完成；发现Specificity/GDPa1预训练配对重叠，结果页已明确不可作严格隔离headline；层级读出提供新诊断证据，正式默认与历史分数未改。无GPU提交。[执行、复核与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ab-exposure-and-layers-20261017)。

2026-10-17 第5轮BERT开发评估更新：61035步macro AUROC0.8806，相比第4轮最佳0.8914回落；同轮旧AA-only为0.8737。64000步AA恢复旧92.14%／新92.00%，近期MLM验证loss约0.29，未出现单调表征提升。训练继续，最佳checkpoint保持48828；只核查既有指标，没有改动训练或提交评测。[证据](/project_c20250601/251105016/project/dllm_test/output/bert_full_token_v7_300k_20261017/review_20261017_1621.json)。

2026-10-17 用户授权的Ophiuchus-Ab GDPa1重新复现完成，CPU全量新提特征和原AirGen回归运行成功，生产回归入口复核一致，结果已回填；首个包装器的前缀断言失败记录独立保留。[执行与验收](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#ophiuchus-gdpa1-reproduction-20261017)。

<!-- bert-aa-only-comparison-20261017 -->
2026-10-17T14:32:32.575789+00:00 复核全token与旧AA-only BERT的同预算效果：相同模型、预算、数据与开发探针协议，第2/3/4轮macro AUROC分别领先0.0140/0.0147/0.0321；同56000步受体AA准确率差−0.0297个百分点，基本持平。新增上下文AA和relation可见性使本次不是仅特殊token的单因素消融；单seed/211条开发验证不足以断言完整下游收益。只比较已有文件，未改训练或提交作业。[结果与证据](/project_c20250601/251105016/project/dllm_test/docs/BERT_FULL_TOKEN_PRETRAINING_20261017.md)。
<!-- /bert-aa-only-comparison-20261017 -->

2026-10-17 GDPa1与AirGen的方法核查及LoRA分链CPU诊断完成；补正结果表中缺少的读出差异说明，历史分数及默认保留。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#gdpa1-airgen-audit-20261017)。

2026-10-17T04:27:00.427638+00:00 OAS-only 从 checkpoint-11000 续训已重提：`t-20261017122656-nzn5w` Queue。旧 `t-20261016162026-4rq8w` 在恢复 RNG 集合通信处 Failed；新作业换独立 runtime 目录，W&B 写到已验证的 pressure_jian/dllm-immune。[提交](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261017/submission.json)。

2026-10-17T04:19:22.845204+00:00 全有效 token BERT 仍在训练：9300/300000，容器 `t-20261007153431-vhx2f` Running。OAS-only 续训 `t-20261016162026-4rq8w` 已于 2026-10-16T21:09:25Z Failed，未重提。[实时状态](/project_c20250601/251105016/project/dllm_test/output/bert_full_token_v7_300k_20261017/preparation/live_status_20261017T041922Z.json)。

2026-10-17 按用户授权停止旧AA-only BERT（最后记录68050，完整68000已核验），在原八卡容器启动全有效token MLM：V7自然比例、300k、从头初始化。新训练已核实75步，checkpoint50完整；CPU16项、八卡显存／原生恢复／FP32复算／尾批484、84分片哈希及W&B服务端读取通过。新增上下文、EOS、类型、relation等预测目标，表征探针隐藏关系真值。[配方](/project_c20250601/251105016/project/dllm_test/docs/BERT_FULL_TOKEN_PRETRAINING_20261017.md)；[启动证据](/project_c20250601/251105016/project/dllm_test/output/bert_full_token_v7_300k_20261017/preparation/startup_verification.json)。

2026-10-16 按用户要求复核历史与当前来源策略。当前冻结BERT代码／索引哈希通过，CPU实际采样器8rank完整遍历6,249,956条，所有索引恰好一次、源计数匹配、尾批484及恢复前缀通过。历史LoRA三个冻结文件哈希通过，实际等源采样器前70000条每源10000，coverage epoch33981步与既有完成回执一致。明确旧后训练曾均衡来源、当前BERT使用自然来源比例；只写复核证据，没有修改训练。[报告](/project_c20250601/251105016/project/dllm_test/output/bert_source_balance_research_20261016/sampler_reverification.json)。

2026-10-16 完成V7 BERT来源平衡只读调研及静态曝光计算。冻结索引train6,249,956、observed receptor AA1,022,400,774；OAS+OTS占95.27%真实AA，TCR repertoire占2.86%。温度采样、长度归一化与簇平衡分别处理曝光、loss系数、多样性；提出共同父checkpoint的2×2候选对照。七来源等量在30万步×512预算下使TRAIT每行期望曝光约11133次，因此不建议直接使用。未改动当前训练或提交作业。[方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_SOURCE_BALANCING_RESEARCH_20261016.md)；[计算证据](/project_c20250601/251105016/project/dllm_test/output/bert_source_balance_research_20261016/exposure_analysis.json)。

2026-10-16T10:45:07.158708+00:00 BERT正式训练核实运行：150/148970步，checkpoint50完整保存；八卡参数一致，W&B新账号waitma/pressure_jian已从服务端读回step25以上日志。r3原生状态恢复与133尾批通过后新建正式run，未使用smoke权重。step0探针macro AUROC0.90287仅为未训练基线。[最终启动证据](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_10epoch_20261016/preparation/startup_verification.json)。


2026-10-16 BERT训练启动实现：用户指定至少10完整epoch，已冻结V5 7,626,885条索引，原85分片逐个哈希通过；总148,970步，尾批133条不丢弃、不重复。八卡A10080长度2048检查：micro64 reserved78.49GiB，改micro32×GA2×8/global512，最大reserved38.47GiB。7项CPU语义/恢复测试通过。W&B旧SDK0.21.1遇到新账号viewer.flags=null异常，参考[官方问题记录](https://github.com/wandb/wandb/issues/11761)，隔离安装0.30.0后新账号waitma已成功在线写入pressure_jian/dllm-immune。首次八卡逐位恢复检查未通过，已保留失败证据并修正AdamW计数器设备映射；r2检查在执行，正式训练待通过。[准备和票据](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_10epoch_20261016)。


2026-10-16 按“先训练 BERT 表征模型”的问题补齐首轮建议：A1 三分支 MLM、优化器与完整 epoch 预算、冻结表征选点和实现差距。已读取 V5／新 V7 的 train manifest 分别估算步数，明确旧 270M BERT YAML 不适用于新 75.6M；没有切换数据或启动训练。[方案](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md#first-bert-run)。

2026-10-16 完成 GIDD 原论文 §3–5、附录 B/C 与官方 v1.1.0 源码静态核查，记录混合日程、KL＋IS 损失、反向采样和逐 token 自纠错；强调生成收益不能直接外推为表征提升，并保留动态权重公式与源码差异。没有运行模型或改变训练。[专项补充](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md#gidd-original-method)。

2026-10-16 按用户最新方向，优先研究含氨基酸替换修复的 BERT 式表征预训练。已核对 BERT 15%＋80/10/10 与 LLaDA2.1 §3.1 的 CPT/SFT 混合 M2T/T2T，明确替换训练从阶段 A 开始；候选配比、源码差距及冻结表征／修复／误改验证见[专项记录](/project_c20250601/251105016/project/dllm_test/docs/BERT_RESIDUE_EDIT_PRETRAINING_20261016.md)。主实验多来源范围与 OAS-only 对照继续区分。本次仅更新调研和方案，未实现或提交新训练。

2026-10-16 按用户澄清更正调研中的数据范围误读：OAS-only 属于对照实验，原“主实验只使用 OAS”的表述错误。已修正报告主线、输入图、阶段 A 与实验对照说明，并同步模型方案和操作约定；现有作业、数据及训练配置未变。[更正说明](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md#data-scope-correction-20261016)。

2026-10-16 OAS-only 从 checkpoint-11000 续训已提交：task_id=`t-20261016162026-4rq8w`，queue012，`Queue`。W&B 使用 waitma / `dllm-immune` / `models` 新 run，不接旧 scratch。冻结源码与 300k 调度不变。[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_resume11000_20261016/platform_status.json)。

2026-10-16 已按用户授权删除当前汇总表可核对的特征缓存及旧 8B 评测结果。保留现行指标、训练 checkpoint 和新 CDR 作业依赖，完成逐文件核验；没有新模型实验或成绩变更。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#feature-cache-cleanup-20261016)。

<!-- sampling-cleanup-log -->
2026-10-16 已记录采样产物清理完成，期间误删的两个并发父预测已按原SHA256恢复，两项原冻结CPU审计通过；指标与结果表保留。[执行与恢复记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#sampling-cleanup-20261016)。
<!-- /sampling-cleanup-log -->

2026-10-16 TCRvdb 新原始文件检查完成：2,034 条中阳性 910、阴性 872、不确定 252；全部通过基础序列/长度检查，全部命中既有冻结测试保护 CDR3 配对键。逐行命中、当前有标签来源重叠检查及参考映射状态已落盘，未修改 raw/prepared 或执行模型。[检查结果](/project_c20250601/251105016/project/dllm_test/docs/TCRVDB_INSPECTION_20261016.md)。

2026-10-16T06:37:55Z OAS-only scratch 失败根因更正：`t-20261015124455-xrns4` 的完整平台日志确认rank0在 `on_step_end` 写 `step_timing.jsonl` 时触发 `OSError: [Errno 122] Disk quota exceeded`。该异常离开训练并进入进程组清理，其余七rank等待BROADCAST 1800s后超时退出；先前仅取末尾100行，漏掉rank0写盘栈，将继发NCCL异常记作根因。最后记录11845步（2026-10-15T15:34:16.570026Z），最新完整保存11000，最佳6000。适用配额的具体层级及容量/inode限制尚未确定，当前全文件系统 `df` 余量不能代表该配额。已补存1200行平台尾日志并逐项核对冻结callback与finally路径；未改训练源码、清理数据或重提任务。[失败核对](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/failure_review.json)；[rank0原始异常](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261016/platform_failure_full_tail.log:961)。

2026-10-15 表征到编辑生成研究完成文献与源码复核：补齐两阶段训练、直接替换、插删和功能证据边界；评审后增加同采样器对照及跨全部训练阶段的数据隔离要求。本文没有新模型成绩。[报告](/project_c20250601/251105016/project/dllm_test/docs/REPRESENTATION_TO_EDIT_GENERATION_REVIEW_20261015.md)。

2026-10-15 双链 MHC 补全已发布：从全部 104,384 条相关当前记录追溯原始来源，67,648 条通过覆盖一致性解析并写入独立 V7，含 34 条明确 MHC-II；保留所有受体、肽、标签与 split。39 项测试、全量 before/after collator、独立来源/参考/逐行比较及 reader 验收通过；未训练或切换已有作业。旧 112 条粗筛由本次完整解析结果取代。[结果与复现](/project_c20250601/251105016/project/dllm_test/docs/V7_MHC_COMPLETION_20261015.md)。

2026-10-15 DecoderTCR 论文与源码调研完成：固定官方 commit，核查 88 个参考键及两类 MHC 查询行为，记录公开数据获取边界、论文/代码版本差别和当前模型适配建议。旧 112 条仅为字符串一致性粗筛，不再作为双链 MHC 恢复上限或优先恢复范围。本轮只完成资料与 CPU 参考核查，没有模型实验。[报告](/project_c20250601/251105016/project/dllm_test/docs/DECODERTCR_RESEARCH_20261015.md)。

2026-10-15 OAS-only fixed-chain 75.6M scratch 已提交：task_id=`t-20261015124455-xrns4`，queue012，平台状态 `Queue`，1 × `ml.pni2.28xlarge`，`Preemptible: false`。训练尚未启动。[配置](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_75m_oas_20261015/README.md)；[平台查询](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_oas_queue012_20261015/platform_status.json)。

2026-10-15 V7 标签过滤纠错完成：之前误将“不要负样本”收窄为显式 binding-only，误排 6,624,920 条 repertoire。更正后仅按标签去除 347,700 条 nonbinding，隔离 3 条无可解析 core，并去除 1,062,545 条测试命中；最终 6,318,945 条（train 6,249,960 / valid 68,985）。12 项测试、全量逐字节与选择复核、336 例 collator 对照和 168 例实际 reader 验收通过。扩大联合聚类池使 1,409 条 AB 正样本选择变化，已逐项记录原因；未改输入布局、原 V5 或既有训练。[数据与验收](/project_c20250601/251105016/project/dllm_test/docs/V7_FROM_V5_REBUILD_20261015.md)。

2026-10-15T03:26:31.243385+00:00 完成18篇Flow/diffusion推理效率与质量调研，核对一手来源及当前绑定源码，产出候选优先级、兼容限制和同成本验证设计；未新增模型实验成绩。[报告](/project_c20250601/251105016/project/dllm_test/docs/INFERENCE_FLOW_DIFFUSION_REVIEW_20261015.md)。

2026-10-15T03:12:37.071641+00:00 CDR步数退化首轮诊断完成，480条真实CPU输出和逐步提交独立复算通过；历史Kong配对及SAb原预测等价通过。归一化排序无已验证的稳定提升，未部署质量修复。[记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-steps-debug-20261015)。

2026-10-15T02:58:11.625722+00:00 fixed reference采样器生产入口对齐；定向回归3项、两个真实checkpoint各500条输入及各2例冻结实现CPU逐token比对均通过；终审已落盘，历史指标及冻结作业保持。[协议与验证记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-alignment-20261015)。

2026-10-14 UpTCR低分首轮debug与输入输出复核完成；证据、候选原因和检查范围统一见[诊断记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#uptcr-debug-20261014)。

2026-10-14 UpTCR binding已补齐论文全部7个比较模型，统一提取Fig.2官方Source Data的5次运行均值；纠正原正文单次数值与均值混用，Ours已有成绩未改。见[UpTCR baseline核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T1_BINDING.md#uptcr-baseline-audit-20261014)。

2026-10-14 T4 sparse13 baseline来源核查完成，已修正候选预算/F1标签及后续评分入口的严格对齐误标。未改原始预测、历史metrics或结果数值；差异及独立重算见[任务核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#sparse13-baseline-source-audit)。

2026-10-14 T4 sparse13差异核查完成：保存序列的距离/恢复率重算通过，新增长度统计和ANARCI整链抽样证据；已修正详细结果页单链有效性的解释。成绩与简洁版数值未变，无模型重跑。见[任务核查](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#t4-length-audit-20261014)。

2026-10-14 按用户要求，从详细版与简洁版结果汇总删除 S²ALM / BioMap 补充论文表：该表没有同协议 Ours 对照，不属于当前任务成绩比较。历史转录与审计继续保存在[过程文档](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#s2alm-audit)，已更新相关来源链接。

2026-10-14 按用户要求，从 RESULTS.md 和 RESULTS_SUMMARY.md 删除 CHIMERA 历史整链生成区块及四个 checkpoint 结果行；当前官方 H3 任务结果保留，原始实验产物未改动。

2026-10-14 270M / checkpoint32000 的 AbBiBench 16项已于01:50 UTC完成；修正详细版/简洁版此前滞留的11/16状态。已核验各项完成状态、metrics SHA256、范围、样本数及等权均值；[发布核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/270m32000_abbibench_publish_20261014/verification.json)，成绩见[RESULTS](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md)。本轮未重跑模型；下方2026-10-13状态表为历史快照。

2026-10-14 AbBiBench 论文/仓库榜单来源核查完成；差异及验证证据见[任务协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#abbibench-paper-scope)。已补充详细结果页与出处登记，保持用户指定的16项范围及已有分数；简洁版继续只列结果。未重跑模型。

2026-10-14 T3 representation 当前协议核查完成：broad/deep仍为本地任务，论文数据、sampling和baseline recipe未对齐；汇总聚合及αβ身份分离检查通过，但不等于完整复现。未重跑或修改分数，修正旧入口说明。权威结论见 [T3 §7.4](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T3_REPRESENTATION.md#74-2026-10-14当前-fixed-chain50000-协议复核)。

2026-10-14 按用户要求移除 T2B KMeans 评测：两版结果表删除 MiniBatchKMeans 与完整KMeans区块；任务生成器/批量入口跳过T2B，直接运行和重建发布入口禁用。T2A保留。历史结果区块及移除记录见 [存档](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_retired_20261014)，原始特征/预测/评分不删除。范围以 [README](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/README.md#0-评测范围2026-08-29-收窄强制) 为准。旧冻结作业的源文件哈希不改写，不将旧计划当作新评测计划复用。

2026-10-14 fixed-chain75.6M checkpoint50000 Kong iter1补测完成；同权重、数据/掩码、batch64、bf16、argmax/temperature1/CFG0，唯一变化为4步改1步。3个CDR各3127条、共9381条预测独立重算通过，已回填详细与简洁结果表。证据：[独立核验](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_cdr_iter1_20261014/verification.json)、[原始指标](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/fixed_chain_75m_step50000_cdr_iter1_20261014/kong/metrics.json)。

2026-10-13 T2B 作者发布包核查与旧入口协议说明修正完成；3个公开编码器全量特征一致、历史算法回归通过，结果数字未变。证据见[代码核查](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_paper_alignment_20261013/code_release_audit/AUDIT.md)。

2026-10-13 T2B独立重建已完成并保存逐K标签与特征；[汇总与论文差异](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/t2b_paper_alignment_20261013/summary.json)。

2026-10-13 T2B 按用户要求启动论文协议对齐，采用独立入口和独立输出；作者完整评分流程缺失及未披露参数均显式记录，不预先宣称复现成功。见 [T2 任务 §7](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md#7-已知缺陷与不可比口径)。

2026-10-13 T2B 名称与可比性更正：已将详细版、简洁版和评测过程页的任务名标为实际使用的 MiniBatchKMeans，并登记论文协议差异。证据见 [T2 任务 §7](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md#7-已知缺陷与不可比口径)。仅修正文档，原指标与评测代码保留，尚未按论文重跑。

2026-10-13 T2A 引用纠错：任务文档作者误写已修正；原论文与本地 retention 对齐的协议边界已补充，并同步详细结果页及评测过程索引。权威说明见 [T2 任务 §2.1](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T2_CLUSTERING.md#21-论文实验全景)。结果数值、评测代码及简洁版结果表未改动。

2026-10-13 评测数据处理更正：用户要求不去重，移除AB–Ag构建中的重复行、冲突测量和CHIMERA跨split重复抗原删除逻辑。旧数据和结果保留；新入口见 [AB–Ag 当前原始行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

更新：2026-10-07T11:55:56.167617+00:00。仅保留影响当前判断的里程碑；当前运行状态见[PROJECT_PROCESS.md](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。

## 关键记录

| 时间／阶段 | 结论 | 证据 |
|---|---|---|
| V5历史基线 | ESMC＋LLaDA版本；V5数据train 7,626,885 / valid 102,308 | [docs/V7_V5_CODE_COMPARISON_20261006.md](/project_c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md) |
| V7 f011正式训练 | 两轮完成至50082；保留正式恢复点 | /project_c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run |
| V7下游评测 | CDR、轻链、TCR三类生成已完成；不支持“全面超过V5”的结论 | [downstream/benchmark/RESULTS.md](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) |
| V7原架构消融 | 3500步、各112000/120000条曝光；相对各自3250步，平均残基loss/准确率改善8/8、3/8；全目标残基准确率改善3/8，context全目标END准确率改善3/8。 完整epoch仍10/22；非最终架构排名 | [详细比较](/project_c20250601/251105016/project/dllm_test/docs/V7_ABLATION_REPORT_INTEGRATION_20261005.md)；[原件](/project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3500_root_review.json) |
| 2026-10-07 新模型改动 | 残差1.0、蛋白对象内／全局交替、V5对象布局、自主终止（总容量302，沿用旧decoder合计预算）；排除VHH | [BIOSEQ_MODEL_PLAN.md](/project_c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md) |
| 2026-10-07 实现验证 | 83项CPU测试；V7样本1250条通过；测试权重已清理 | /project_c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_regression.log；canvas_validation.json |
| 2026-10-07 数据选择 | 用户决定新版本改用V5数据；不恢复旧ESMC架构，不叠加V7后续过滤 | /project_c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json |
| 2026-10-07 attention修正 | prots/protd对象分段；split仅分输出链，位置不重置；84项CPU测试通过 | /project_c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/object_attention_regression.log |
| 2026-10-07 容量确定 | 采用已有H/β 167＋L/α 135 decoder槽位合计302；不固定split位置；86项测试通过 | /project_c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/capacity302_regression.log |
| 2026-10-07 V5全量检查与启动绑定 | 7,729,193条／85分片全部通过；最大对象299、最大总输入1218；无VHH；97项测试及原生父数据检查通过，未提交训练 | /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/full_audit.json；/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/native_parent_preflight.json |

## 2026-10-07 新训练预算

按用户要求设300,000更新步（历史下游150,000步×2）、global batch512（8×8×8）。LR1e-4、warmup2000、cosine；总样本处理量153,600,000，约20.14遍V5训练集。冻结入口参数及源码校验通过；未做GPU显存实测、未提交训练。证据：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v1/training_config_validation.json。

## 已更正的说法

| 旧说法 | 当前结论 |
|---|---|
| 可见split要切换attention段／重置位置 | 同一个受体对象始终同段；context也按prots/protd对象划分 |
| 512是V5规定或已验证的最佳容量 | 已改为原固定decoder总预算302；现已全量确认V5最大对象299 |
| 当前新训练仍用V7清理版 | 改为V5 receptor_completion；原V7只保留历史用途及原作业 |
| V5形式必须给定真实生成长度 | 当前新模型只给对象容量302，`.`和`<protd>`均由模型预测 |
| 只有CDR时使用短链／把合成X直接送入模型 | 保留原全长scaffold；synthetic转mask且不计损失 |
| VHH属于本次训练任务 | 本版不纳入VHH；通用代码支持不代表训练范围 |
| f010是局部注意力改动 | f010是固定残差0.1；f001才是radius32局部／全局；f011两者组合 |
| max_length=2048代表所有旧输入都补到2048 | 是总长度上限；旧动态batch padding与新302对象容量须分别描述 |
| 新模型已有V5验证／训练成绩 | V5全量布局与启动绑定现已通过；仍无新模型训练或下游成绩 |

## 2026-10-07 3250步记录同步修复

3250步指标读取实际452525退出0，八项各104000/120000条曝光的核对已通过。旧写回器7c77aa退出1：主文档精简后不再含旧实时状态标记；已写出的指标与验收记录完整保留。新[同步器](/project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/sync_ablation_observation_docs_v2.py)实际4ad5b1退出0，按精简结构补齐5份文档并回读通过，没有重复验证或训练。后续普通观察只更新状态行、输出证据；关键结果集中写本记录和专题报告，保持新版本V5决定。

修复证据：[first3250_doc_sync_recovery.json](/project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3250_doc_sync_recovery.json)。原观察会话3185已正常退出0；下一步为3500验证，原训练与四项排队任务继续。

## 历史与维护

完整旧日志及更正原文：[归档说明](/project_c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/README.md)。专题报告、原始指标和冻结配置保留原位，不重写历史数值。今后每项实验只维护一行状态，只有关键改变才追加里程碑；普通心跳不写入主文档。

## 2026-10-07 用户暂停原消融

用户明确要求暂停消融、提交新版本训练。已请求停止原八卡消融，保留3500步已验收指标及后续部分曲线；四项未启动任务保持未完成。消融仅保存指标，未保存可精确续训的模型状态，不能宣称之后可无损接续。实际退出证据写入 /project_c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json。

停止核实：八项3600–3650步，八worker均退出；平台07:08:26Z Failed为主动中断。停止记录保留真实TimeoutError及缺失returncode，不伪造成功退出。10/22完整产物不变；其余12项暂停未完成。

## 2026-10-07 八卡独立训练入口

完成launch_v2及YAML；补齐分隔符验证、best3＋latest和时间/显存记录。原回归首次46通过、2个旧固定16列fixture失败；已按新统计列扩展fixture，复验进行中。保留原失败日志，不宣称原97测试已覆盖这次整合。

启动整合复验：48/48相关CPU测试通过，checkpoint保留10/10通过，源码差异仅验证器/入口/驱动及2测试文件；模型主体未变。提交记录：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/submission.json；任务ID ['t-20261007151807-cq4fb']，当前只确认提交结果，尚无正式训练成绩。

## 2026-10-07 新八卡任务启动故障

任务t-20261007151807-cq4fb已Failed，rank4/6 SIGSEGV，正式0步、synthetic验收0/8。已完成的48项CPU测试和10项保留测试仍只证明其原范围。根因尚未证明；原日志 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/torchrun_attempt1.log，不盲目重提相同配置。

配置核对及接续：用户指定training_config与已提交v2／未提交v3参数完全匹配。v3候选采用旧成功入口的冷导入/单GPU隔离/LOCAL_WORLD_SIZE=1、显式设备及启动阶段记录，按NVIDIA运行时文档准备cuMem-host关闭的单机兼容配置；这只是待验证修正，根因未定，未提交第二任务。原失败包、停止标记及原始日志不改。源码说明：/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/startup_change.json；NVIDIA参考：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/runtime_and_mpi_issues.html。

## 2026-10-07 资源方式纠正

用户要求资源容器常驻、进入其中调试训练。新增容器监督入口与分离的PAUSE_TRAINING/STOP，复用已有占卡和进程组回收函数；本次只申请持续八卡资源，训练ticket在资源就绪后发布。初始队列为空；旧失败任务与其日志保留。

持续容器已提交，ID=['t-20261007153431-vhx2f']；CPU进程控制验收通过（17自有子进程回收、失败保留容器、无重复执行），这不是GPU验收。实际训练仍0步，待资源就绪。

容器t-20261007153431-vhx2f就绪后已发布唯一v3训练票据。修改仅在启动层；未变训练参数/模型主体/数据。实际GPU通过、训练开始及速度仍待原件，不把票据发布当完成。证据：/project_c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/training_enqueue.json。

实际八卡启动通过：task=t-20261007153431-vhx2f，8/8 synthetic profile和8/8 NCCL ready，micro8／1218长度峰值最大4.48 GiB。3步测试模型已在各进程丢弃，不保存权重；正式driver重新seed42初始化。原始证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2/startup_review.json。

正式训练启动实证：容器t-20261007153431-vhx2f，121步，最近25步中位0.798秒；V5七源7,626,885训练记录、global512、scratch参数与配置一致，原生数据门禁通过。下一验收点1000步验证及完整保存。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/formal_start_review.json。

## 2026-10-07 首个checkpoint验收准备

上轮Goal为progress：按用户要求恢复持续容器方式并已产生正式更新步。本轮实际轮询同一会话62851确认训练继续，没有重启或重提。新增只读产物核对入口 /project_c20250601/251105016/project/dllm_test/scripts/research/review_v5_checkpoint.py，复用原JSON指标读取/数值比较，检查17份完整状态文件哈希、八rank保存确认、四噪声等级含split的90项聚合以及best3＋latest实际目录。首次1000步产物尚未出现，当前仅完成核对代码，不宣称保存或恢复验收通过。

观察记录维护：只读watch原每次覆盖active_bounded_observation会丢失工具session编号；下次观察窗口使用更新后的局部字段更新，并动态标记下次1000步保存点。当前会话62851按原版本继续直到自然结束；未重启监听、训练或容器。

## 2026-10-07 1000步保存实际验收

07:54:20Z /project_c20250601/251105016/project/dllm_test/scripts/research/review_v5_checkpoint.py 首次实际执行通过，报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint1000_root_review.json，SHA256=54ddb42e9f1a1ea4662badb6d7e367a240e4ea0f846360e5f64c4a0aa89fa875。独立重哈希17个完整状态文件，核对八rank保存/参数、分隔符计入总loss、四mask等级90项核心聚合及best3＋latest保留目录；未加载模型张量、重跑训练/评测或恢复测试。

验证共七源、每源256条、每条四个mask等级。主eval_loss（mask100/source-macro）2.1790857；mask_mean/token-micro AA准确率0.629156，mask100/token-micro AA准确率0.312690；终止符单位为完整受体对象，不能解释为各链独立长度成绩。此为首个训练中验证点，不能据此宣称下游效果改善。当前只保存checkpoint-1000，既是best也是latest；无新临时测试checkpoint。容器/训练持续运行，旧62851观察窗口于07:50:13自然结束，07:54开始44566只读有界观察，不涉及训练或容器重启。下一常规检查2000步，原消融继续按用户要求暂停。

## 2026-10-07 完整窗口计时与监督状态区分

实际step1→1001共1000更新，含一次验证/完整保存，803.212秒，平均0.803212秒/步、约637.44条/秒（global512完整累积窗口）。训练callback之和793.750秒，其外9.462秒包括验证/保存/保留检查及其间主机工作，不作进一步归因。rank0峰值allocated4.783 GiB、reserved5.344 GiB。报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/speed_first1000_updates_root_review.json 保留原始计时前1001行摘要及callback源码哈希；未新增训练或性能扫档。

只读观察入口 /project_c20250601/251105016/project/dllm_test/scripts/research/watch_v5_direct.py 原先只展示平台状态，而持续容器中的训练退出不一定使平台退出。现补充指定训练票据的result/returncode及容器训练状态；训练失败或退出时结束观察并交回人工核实，不重试任务。另将日志nan/inf识别为异常，避免只提取较早的有限loss。语法检查及当前真实容器状态读取通过；本次没有人为制造GPU失败。旧44566于08:09:19按900秒边界自然结束；新53939于08:09:29实际首读2287步，容器Running且训练running，新观察字段已生效（工具5be746）。没有重启训练/容器，不影响训练源绑定。

## 2026-10-07 V5阶段验证与完整状态（1000–72000步）

2000步时实际只读核对通过，原件 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint2000_root_review.json（SHA256=d51b72a70973744277c6f100d46a5fc8e1b21e29475f839e15bed08fc56fade8）。17个状态文件哈希、八rank一致性、90项核心指标、实际保留目录均符合约定，参数摘要相对1000步改变；恢复重放未执行。best3＋latest当时并集为1000/2000两份完整状态，2000为当时最优与最新。各后续保存及清理结果见本节对应步数记录，实时状态见PROJECT_PROCESS。

主loss采用mask100/source-macro；AA均值采用mask_mean/token-micro；其余准确率采用mask100/token-micro。终止符单位为完整受体对象。

| 步数 | 主loss | AA均值准确率 | AA全遮蔽准确率 | 终止符准确率 | 关系标签准确率 | 分隔符准确率 |
|---|---:|---:|---:|---:|---:|---:|
| 1000 | 2.179086 | 62.92% | 31.27% | 72.71% | 80.76% | 73.49% |
| 2000 | 2.132296 | 72.81% | 33.54% | 83.37% | 69.34% | 74.67% |
| 3000 | 2.072132 | 74.61% | 34.35% | 86.83% | 71.48% | 78.03% |
| 4000 | 2.108269 | 74.82% | 33.70% | 81.25% | 78.81% | 75.42% |
| 5000 | 2.091514 | 75.09% | 34.27% | 79.35% | 75.78% | 75.48% |
| 6000 | 2.050985 | 75.52% | 35.15% | 82.65% | 79.00% | 75.23% |
| 7000 | 2.102113 | 75.32% | 34.17% | 76.90% | 71.88% | 75.67% |
| 8000 | 2.070347 | 75.50% | 34.52% | 74.89% | 77.25% | 74.49% |
| 9000 | 2.033069 | 75.98% | 36.05% | 81.19% | 74.12% | 76.97% |
| 10000 | 2.028133 | 76.12% | 36.53% | 75.61% | 71.29% | 78.03% |
| 11000 | 2.025401 | 75.97% | 35.61% | 75.95% | 74.71% | 75.42% |
| 12000 | 2.025920 | 75.82% | 34.97% | 84.49% | 73.83% | 76.41% |
| 13000 | 2.122830 | 75.80% | 34.68% | 87.28% | 74.02% | 75.48% |
| 14000 | 2.018878 | 76.38% | 36.93% | 82.76% | 71.97% | 78.09% |
| 15000 | 2.044152 | 75.93% | 34.92% | 85.83% | 71.48% | 75.48% |
| 16000 | 2.052560 | 75.82% | 34.47% | 84.60% | 76.46% | 75.48% |
| 17000 | 2.046143 | 75.91% | 34.79% | 82.92% | 75.29% | 78.03% |
| 18000 | 2.073688 | 75.95% | 34.81% | 81.75% | 70.61% | 75.36% |
| 19000 | 2.064763 | 75.94% | 34.82% | 82.92% | 75.10% | 75.48% |
| 20000 | 2.040837 | 75.98% | 34.90% | 81.47% | 75.78% | 75.42% |
| 21000 | 2.027940 | 76.07% | 35.05% | 84.54% | 75.68% | 77.96% |
| 22000 | 2.052559 | 76.03% | 34.92% | 80.41% | 79.49% | 75.42% |
| 23000 | 2.025071 | 76.22% | 35.61% | 80.25% | 72.85% | 75.54% |
| 24000 | 2.024323 | 76.20% | 35.53% | 81.47% | 75.68% | 78.27% |
| 25000 | 2.089885 | 75.97% | 34.63% | 81.53% | 74.51% | 74.92% |
| 26000 | 2.061376 | 76.09% | 34.87% | 79.07% | 74.61% | 75.48% |
| 27000 | 2.011242 | 76.29% | 35.64% | 70.93% | 84.96% | 75.48% |
| 28000 | 2.012737 | 76.28% | 35.59% | 79.74% | 74.80% | 78.09% |
| 29000 | 2.019129 | 76.56% | 36.69% | 80.69% | 71.68% | 75.48% |
| 30000 | 2.039798 | 76.09% | 34.90% | 75.00% | 71.48% | 74.18% |
| 31000 | 2.042209 | 76.17% | 35.03% | 80.30% | 71.00% | 75.42% |
| 32000 | 2.061845 | 76.14% | 35.02% | 82.92% | 71.19% | 75.48% |
| 33000 | 2.026425 | 76.22% | 35.14% | 83.71% | 66.89% | 78.09% |
| 34000 | 2.066092 | 76.23% | 35.20% | 79.69% | 64.16% | 80.01% |
| 35000 | 2.032845 | 76.26% | 35.18% | 75.17% | 75.88% | 78.09% |
| 36000 | 2.022820 | 76.20% | 34.98% | 80.64% | 73.63% | 75.42% |
| 37000 | 2.056276 | 76.06% | 34.39% | 76.23% | 73.34% | 75.54% |
| 38000 | 2.057750 | 76.23% | 35.16% | 83.09% | 71.88% | 75.42% |
| 39000 | 2.040339 | 76.24% | 35.05% | 79.24% | 74.12% | 74.49% |
| 40000 | 2.000729 | 76.50% | 36.15% | 76.90% | 71.00% | 75.36% |
| 41000 | 2.011555 | 76.42% | 35.73% | 80.58% | 81.74% | 74.49% |
| 42000 | 1.995668 | 76.38% | 35.56% | 78.40% | 74.80% | 78.03% |
| 43000 | 2.007580 | 76.29% | 35.25% | 79.07% | 74.41% | 74.43% |
| 44000 | 2.027899 | 76.25% | 35.06% | 85.44% | 72.07% | 75.36% |
| 45000 | 1.992721 | 76.84% | 37.35% | 80.69% | 71.29% | 77.90% |
| 46000 | 2.041335 | 76.28% | 35.22% | 77.29% | 71.97% | 73.87% |
| 47000 | 2.007618 | 76.25% | 35.03% | 80.13% | 77.05% | 75.42% |
| 48000 | 2.015223 | 76.26% | 35.14% | 82.59% | 71.97% | 75.48% |
| 49000 | 2.025853 | 76.27% | 35.09% | 77.12% | 75.49% | 75.48% |
| 50000 | 2.021366 | 76.33% | 35.27% | 77.68% | 70.80% | 75.42% |
| 51000 | 2.050266 | 76.22% | 34.93% | 85.99% | 76.07% | 75.48% |
| 52000 | 2.032902 | 76.29% | 35.01% | 76.95% | 71.48% | 75.42% |
| 53000 | 2.050998 | 76.20% | 34.82% | 79.74% | 72.75% | 74.43% |
| 54000 | 2.009030 | 76.33% | 35.06% | 84.93% | 72.17% | 75.48% |
| 55000 | 2.051494 | 76.30% | 35.14% | 81.25% | 76.37% | 75.42% |
| 56000 | 2.010377 | 76.34% | 35.19% | 81.42% | 74.22% | 75.42% |
| 57000 | 2.017770 | 76.42% | 35.54% | 80.64% | 72.95% | 74.49% |
| 58000 | 2.014520 | 76.33% | 35.14% | 84.26% | 75.10% | 75.48% |
| 59000 | 2.047211 | 76.27% | 34.92% | 83.04% | 75.98% | 75.48% |
| 60000 | 2.012125 | 76.38% | 35.25% | 82.37% | 74.71% | 75.48% |
| 61000 | 1.998545 | 76.60% | 36.18% | 83.31% | 73.93% | 78.09% |
| 62000 | 2.032696 | 76.26% | 34.94% | 82.31% | 76.76% | 75.48% |
| 63000 | 2.060971 | 76.28% | 35.00% | 79.63% | 75.98% | 75.42% |
| 64000 | 2.030086 | 76.31% | 35.00% | 80.52% | 73.73% | 75.42% |
| 65000 | 2.029600 | 76.31% | 35.16% | 83.15% | 74.71% | 75.42% |
| 66000 | 2.030482 | 76.28% | 35.04% | 80.19% | 75.59% | 75.48% |
| 67000 | 2.056695 | 76.35% | 35.12% | 79.46% | 73.14% | 75.42% |
| 68000 | 1.998187 | 76.73% | 36.96% | 84.49% | 71.78% | 77.96% |
| 69000 | 2.019536 | 76.32% | 35.07% | 79.35% | 75.88% | 75.42% |
| 70000 | 2.050812 | 76.36% | 35.09% | 81.36% | 72.56% | 75.42% |
| 71000 | 2.011193 | 76.36% | 35.12% | 81.53% | 71.58% | 75.42% |
| 72000 | 1.999634 | 76.41% | 35.24% | 85.21% | 72.95% | 75.42% |

前两个早期点的损失、AA和边界指标改善，但关系标签重建回退；当时不足以判断长期趋势。关系标签重建亦不等于下游正负binding判别。保持用户指定配置和原300000步训练，不因此自动恢复消融、调参或重跑下游。

进一步核对既有验证日志：mask100关系标签共1024个，四个有标签来源各256个，两次来源/标签数量一致。下降集中在tcr_native，不能用验证来源数量变化解释；未据此归因优化器或数据错误。

各数据源关系标签准确率，mask100、每源256个标签：

| 步数 | asd_antibody | tcr_native | tcr_papers | trait |
|---|---:|---:|---:|---:|
| 1000 | 87.89% | 96.48% | 78.91% | 59.77% |
| 2000 | 87.89% | 37.50% | 83.59% | 68.36% |
| 3000 | 87.89% | 36.72% | 91.02% | 70.31% |
| 4000 | 87.89% | 56.25% | 89.84% | 81.25% |
| 5000 | 87.89% | 44.92% | 92.19% | 78.12% |
| 6000 | 87.89% | 58.59% | 87.50% | 82.03% |
| 7000 | 87.89% | 35.55% | 92.19% | 71.88% |
| 8000 | 87.89% | 52.34% | 91.41% | 77.34% |
| 9000 | 87.89% | 39.45% | 94.14% | 75.00% |
| 10000 | 88.28% | 32.81% | 94.14% | 69.92% |
| 11000 | 87.89% | 42.19% | 94.14% | 74.61% |
| 12000 | 87.50% | 38.28% | 95.31% | 74.22% |
| 13000 | 88.28% | 43.36% | 93.75% | 70.70% |
| 14000 | 87.89% | 34.38% | 93.75% | 71.88% |
| 15000 | 87.50% | 34.77% | 95.31% | 68.36% |
| 16000 | 87.89% | 50.78% | 92.19% | 75.00% |
| 17000 | 89.06% | 47.66% | 90.23% | 74.22% |
| 18000 | 87.89% | 31.25% | 93.75% | 69.53% |
| 19000 | 90.63% | 42.97% | 94.53% | 72.27% |
| 20000 | 89.45% | 45.31% | 91.41% | 76.95% |
| 21000 | 87.50% | 46.48% | 91.02% | 77.73% |
| 22000 | 90.63% | 52.73% | 91.80% | 82.81% |
| 23000 | 87.50% | 35.55% | 92.19% | 76.17% |
| 24000 | 88.28% | 44.92% | 93.75% | 75.78% |
| 25000 | 90.63% | 39.45% | 94.92% | 73.05% |
| 26000 | 90.63% | 41.02% | 93.75% | 73.05% |
| 27000 | 87.50% | 87.50% | 86.72% | 78.13% |
| 28000 | 90.23% | 38.28% | 94.92% | 75.78% |
| 29000 | 88.28% | 34.77% | 94.53% | 69.14% |
| 30000 | 87.50% | 32.42% | 96.48% | 69.53% |
| 31000 | 87.50% | 31.64% | 96.09% | 68.75% |
| 32000 | 86.72% | 32.81% | 94.53% | 70.70% |
| 33000 | 87.89% | 12.11% | 96.09% | 71.48% |
| 34000 | 87.89% | 3.91% | 95.70% | 69.14% |
| 35000 | 91.02% | 43.36% | 93.75% | 75.39% |
| 36000 | 88.67% | 39.45% | 93.75% | 72.66% |
| 37000 | 87.50% | 40.23% | 93.36% | 72.27% |
| 38000 | 87.89% | 32.03% | 95.70% | 71.88% |
| 39000 | 89.45% | 39.06% | 93.36% | 74.61% |
| 40000 | 87.89% | 31.25% | 94.92% | 69.92% |
| 41000 | 87.50% | 73.83% | 91.02% | 74.61% |
| 42000 | 87.89% | 41.41% | 93.75% | 76.17% |
| 43000 | 89.84% | 41.80% | 93.75% | 72.27% |
| 44000 | 88.28% | 34.38% | 94.53% | 71.09% |
| 45000 | 87.50% | 32.42% | 95.31% | 69.92% |
| 46000 | 88.67% | 31.64% | 95.31% | 72.27% |
| 47000 | 90.63% | 45.31% | 94.92% | 77.34% |
| 48000 | 88.28% | 33.20% | 95.31% | 71.09% |
| 49000 | 87.50% | 46.88% | 93.75% | 73.83% |
| 50000 | 88.28% | 30.47% | 95.31% | 69.14% |
| 51000 | 87.50% | 45.31% | 94.53% | 76.95% |
| 52000 | 88.28% | 32.03% | 96.09% | 69.53% |
| 53000 | 91.41% | 32.42% | 95.31% | 71.88% |
| 54000 | 87.50% | 33.59% | 95.70% | 71.88% |
| 55000 | 89.45% | 45.70% | 94.14% | 76.17% |
| 56000 | 88.67% | 35.55% | 96.88% | 75.78% |
| 57000 | 87.89% | 35.16% | 95.70% | 73.05% |
| 58000 | 89.06% | 41.02% | 94.92% | 75.39% |
| 59000 | 87.89% | 43.75% | 93.75% | 78.52% |
| 60000 | 91.41% | 37.11% | 96.48% | 73.83% |
| 61000 | 90.63% | 35.94% | 94.14% | 75.00% |
| 62000 | 89.84% | 46.09% | 93.36% | 77.73% |
| 63000 | 88.28% | 47.66% | 94.53% | 73.44% |
| 64000 | 87.50% | 38.28% | 96.88% | 72.27% |
| 65000 | 89.06% | 39.84% | 94.92% | 75.00% |
| 66000 | 91.80% | 38.67% | 96.09% | 75.78% |
| 67000 | 90.23% | 35.55% | 94.14% | 72.66% |
| 68000 | 88.67% | 32.03% | 96.48% | 69.92% |
| 69000 | 92.19% | 42.58% | 94.53% | 74.22% |
| 70000 | 88.67% | 33.98% | 96.48% | 71.09% |
| 71000 | 89.84% | 30.08% | 96.09% | 70.31% |
| 72000 | 87.50% | 35.94% | 95.70% | 72.66% |

前两个点原件已归档于 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-1000_ce6f98b98cda/trainer_state.json 与 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-2000_ce6f98b98cda/trainer_state.json。监督只读取现有指标，未重新评测或读写序列数据；持续观察该来源的变化。

3000步实际核对通过（08:20:20Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint3000_root_review.json，SHA256=1ae44347bf536dafb5cb922e9e194328e30e91ab4e545a87e6c10280c1fcef13。新增完整文件/八rank/90项聚合/参数更新/保留目录检查均通过；保留1000/2000/3000三份完整状态，3000为最优与最新。关系来源/标签数量仍与前两次相同；总体关系准确率稍回升，但tcr_native没有恢复，不归因为验证数量变化或宣称下游binding改善。该点原始分源指标归档 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-3000_ce6f98b98cda/trainer_state.json。继续原配置至下一4000步，未新增模型测试、调参或checkpoint恢复重放。

4000步实际核对通过（08:33:07Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint4000_root_review.json，SHA256=6fb721a5674a8843c6906a0968424c42158cadaed453ad5abf95658401bb60ab。主loss与全遮蔽AA较3000步回退，关系标签指标回升，未据此宣称全面提升或调整训练参数。最优3000、最新4000，保留2000/3000/4000三份完整状态；1000步完整权重由原生保留策略删除。

首次清理后新增只读核对：1000步的六份JSON档案哈希与原生保存回执一致，且其原始回执哈希仍与先前1000步核对报告一致。实际报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/retention_after4000_archive_review.json，SHA256=a2a904796725a769e3ca64a58ca980d4b4b309ae4ffd49b99a0f2e1d8db45b1b。只读入口 /project_c20250601/251105016/project/dllm_test/scripts/research/review_v5_checkpoint.py 已增加删除后档案检查，当前真实清理产物验收通过，后续新checkpoint验收自动包含该检查；未重复执行已通过的4000步完整检查，未改运行中的训练代码。下一5000步。

5000步实际核对通过（08:46:26Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint5000_root_review.json，SHA256=0fe100da1788fc2e356091b1991cd2e43b62085b6ffc3c5b665986f6f844eed8。完整状态/八rank/90项聚合/参数更新及保留检查通过；新增删除后档案核对也已随本次主验收实际执行，2000步权重已清理、其六份JSON与原始回执哈希一致。当前保留3000/4000/5000，最优3000、最新5000。loss和全遮蔽AA较4000改善，但终止符/关系标签准确率回退，tcr_native关系标签仍在波动；样本/标签数量保持相同，不能将此称为全面改善或下游成绩。保持原配置，下一6000步；未新增模型测试、重启训练或执行恢复重放。

6000步实际核对通过（09:00:02Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint6000_root_review.json，SHA256=a3dbcab18d2d000939bc970a25839b72d57240640d8c3a0fa4392ea1e06195e3。17状态文件、八rank、90项聚合、参数更新、保留目录及新清理4000步的六份JSON档案哈希均通过。主loss2.050985低于原最佳3000步2.072132，6000成为最优与最新；保留3000/5000/6000。全遮蔽AA、终止符和总体关系标签指标较5000步改善，分隔符准确率略退，tcr_papers关系标签92.19%→87.50%；不能只报整体提升忽略分项回退。tcr_native关系标签58.59%，仍低于1000步96.48%，无原因定论。样本/标签数量保持原范围；保持训练配置至下一7000步，没有额外模型测试、训练重启或恢复重放。

7000步实际核对通过（09:13:47Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint7000_root_review.json，SHA256=804ece0924ae7804a087d0ef6fd69c385b872543a92d25be1a6400c583827975。17文件、八rank、90项聚合、参数更新及实际保留目录通过。7000主loss2.102113未进入当前前三，最佳仍6000；最佳3000/5000/6000加最新7000，实际四份完整状态保留，本次remove为空，未清理旧权重。首次真实出现latest不在best3的情况，已按并集核对，不能误删最新完整恢复点。

7000主loss、全遮蔽AA、终止符与总体关系标签指标较6000回退；分隔符和tcr_papers关系标签改善。tcr_native关系标签58.59%→35.55%，仍有明显波动；验证数量/范围不变，不据此归因具体训练故障、宣称长期趋势或下游binding表现。继续同一配置至8000步，没有额外模型测试、训练重启或checkpoint恢复重放。

8000步实际核对通过（09:26:30Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint8000_root_review.json，SHA256=ea6a9f3c6fb335a393263674c5bb4e2e9f8a488d4ff9d93bd70265314dace3a1。17文件、八rank、90项聚合、参数更新及实际保留目录通过。best3=6000/8000/3000，latest8000已在其中，实际保留3000/6000/8000三份完整状态；此次同时移除5000与原latest7000，两点共12份JSON档案哈希均通过核对。未删除其他正式恢复点或重复旧checkpoint完整验收。

8000主loss、全遮蔽AA和关系标签准确率较7000改善，但终止符76.90%→74.89%、分隔符75.67%→74.49%下降；tcr_native关系标签52.34%，仍波动且低于初期。验证数量/范围保持一致，不解释为全面提升或下游binding结果；当前最优仍6000的2.050985。保持原配置，下一9000步，无额外模型测试、训练重启或恢复重放。

9000步实际核对通过（09:40:04Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint9000_root_review.json，SHA256=12c5eacbd4d634f6720554847d58a47f2f8b076793d808c688e15a990814aac0。17文件、八rank、90项聚合、参数更新、实际保留目录与新清理3000步的六份JSON档案哈希均通过。主loss2.033069低于原最佳6000步2.050985，9000为新最优与最新；实际完整保留6000/8000/9000，3000权重由原生规则清理，指标/来源档案保留。

9000全遮蔽AA、终止符和分隔符较8000改善；总体关系标签77.25%→74.12%、tcr_native52.34%→39.45%、trait77.34%→75.00%回退，tcr_papers91.41%→94.14%提高。保持原验证范围/数量，不以主loss改善掩盖分项回退，也不据此归因训练故障或宣称下游binding效果。继续同一配置至10000步，未新增模型测试、重启训练或执行恢复重放。

10000步实际核对通过（09:52:56Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint10000_root_review.json，SHA256=f9e53729454d414ad28d7f0502d8327c9afc578727b0faa8c303cd914420707c。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；新清理8000步的六份JSON档案哈希一致。best3=10000/9000/6000，latest10000与之重合，实际保留6000/9000/10000三份完整状态。主loss2.028133低于9000步2.033069。

10000全遮蔽AA36.05%→36.53%、分隔符76.97%→78.03%；终止符81.19%→75.61%、关系标签74.12%→71.29%下降。tcr_native39.45%→32.81%、trait75.00%→69.92%，asd_antibody87.89%→88.28%、tcr_papers94.14%持平；四个有关系标签来源仍各256。分项回退如实记录，不推断原因或下游binding成绩。下一11000步，现有训练源/配置与容器保持，不重跑已完成验收、模型测试或恢复重放。

11000步实际核对通过（10:06:44Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint11000_root_review.json，SHA256=429d12bf08892d048dfb6966aabf781a1f30a11d81946bc788b78389084c7dc3。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；新清理6000步的六份JSON档案哈希一致。best3=11000/10000/9000，latest11000与之重合，实际保留9000/10000/11000三份完整状态。主loss2.025401低于10000步2.028133。

11000终止符75.61%→75.95%、关系标签71.29%→74.71%回升；全遮蔽AA36.53%→35.61%、AA均值76.12%→75.97%、分隔符78.03%→75.42%回退。tcr_native32.81%→42.19%、trait69.92%→74.61%，asd_antibody88.28%→87.89%，tcr_papers94.14%持平。来源/标签数量不变，不将主loss新低视为全面改善或下游binding提升，不据波动归因具体故障。下一12000步，保持同一训练配置及容器，无额外模型测试、训练重启或恢复重放。

12000步实际核对通过（10:20:02Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint12000_root_review.json，SHA256=f45c53753051ebfb1957d0165bcd934b4f81d6460c503eac227339ff4c73b5ed。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；新清理9000步的六份JSON档案哈希一致。best3=11000/12000/10000，latest12000已在其中，实际保留10000/11000/12000三份完整状态；最佳仍11000，主loss2.025401→2.025920小幅回升。

12000终止符75.95%→84.49%、分隔符75.42%→76.41%回升；全遮蔽AA35.61%→34.97%、AA均值75.97%→75.82%、关系标签74.71%→73.83%回退。asd_antibody87.89%→87.50%、tcr_native42.19%→38.28%、trait74.61%→74.22%，tcr_papers94.14%→95.31%。验证来源/标签数量保持一致，不推断具体故障或下游binding效果；完整受体对象终止符不等于独立每条链的结束位置。下一13000步，同一配置继续，无额外模型测试、训练重启或恢复重放。

13000步实际核对通过（10:33:23Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_root_review.json，SHA256=a733f9063ea6a5e418b633d92b2aeaa351049a9139258b81ca1db7a9ec8facda。17文件、八rank、90项核心聚合、参数更新及保留目录通过。best3仍11000/12000/10000，latest13000不在前三，实际保留10000/11000/12000/13000四份完整状态；remove为空，不产生新的清理档案。本次完整保存通过不代表指标改善，未执行恢复重放。

13000主loss2.025920→2.122830回退；全遮蔽AA34.97%→34.68%、AA均值75.82%→75.80%、分隔符76.41%→75.48%下降，终止符84.49%→87.28%、关系标签73.83%→74.02%提高。asd_antibody关系标签87.50%→88.28%、tcr_native38.28%→43.36%，tcr_papers95.31%→93.75%、trait74.22%→70.70%。不将部分准确率提高解释为整体改善或下游binding成绩。

此次loss回升新增只读分源核对，原件 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_source_delta_review.json（SHA256=8512286e4d5d3499279ecda7ce3194358c426afe5ed2c0585b5a42b8ada3662c）。读取12000/13000已归档trainer_state，未加载模型或数据。七来源的全遮蔽selected/AA/terminal/relation/split计数逐项相同；source-macro总增量0.096910466，其中tcr_native贡献0.090232705，约93.11%。该来源loss1.900705→2.532334、AA loss1.922339→2.558714、AA准确率34.39%→33.23%；terminal准确率99.22%→82.81%、split87.50%→71.48%，总体终止符提升不能覆盖此来源回退。这是等权指标的算术分解，不能据此确定根因；下一14000步重点观察该来源。现有训练配置和容器保持，没有额外模型测试、重启或恢复重放。

14000步实际核对通过（10:46:34Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_root_review.json，SHA256=fc0545e2ed56e48658ed162eebd8d6ded7ca4e83baf5d0d6487d4192331f7d25。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；新清理10000/13000的12份JSON档案哈希通过。best3=14000/11000/12000，latest14000重合，实际完整保留11000/12000/14000三份；主loss2.018878低于原最佳11000的2.025401。未执行恢复重放。

14000全遮蔽AA34.68%→36.93%、AA均值75.80%→76.38%、分隔符75.48%→78.09%提高，终止符87.28%→82.76%、关系标签74.02%→71.97%下降。关系标签分源asd_antibody88.28%→87.89%、tcr_native43.36%→34.38%、tcr_papers93.75%持平、trait70.70%→71.88%。不能以主loss新低掩盖分项回退，亦不等同下游binding表现。

13k回退的后续只读核对 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_source_followup_review.json（SHA256=2d5943f3d03c2968c552a95cd91fa0c9a2e2bbbf4aa92896f96b785fcfd20b52），复用13k诊断原件并读取新14k归档指标，七来源五类全遮蔽计数保持一致。tcr_native loss2.532334→1.873917，低于12k的1.900705；AA loss2.558714→1.894525、AA准确率33.23%→43.07%、terminal82.81%→99.22%、split71.48%→87.50%，本点恢复。但该来源关系标签loss1.166453→1.296756、准确率43.36%→34.38%回退，不能称为全面恢复。source-macro总变化-0.103952745，其中tcr_native贡献-0.094059529；仅为算术核对，根因未确定。继续相同配置至15000步，没有加载模型或序列进行额外测试、训练重启或恢复重放。

15000步实际核对通过（11:00:15Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint15000_root_review.json，SHA256=bd0db8c3ead1b765b0c0e48b039ac867c597d4c6d5f2208d205baebbbc46f2a1。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。主loss2.044152未进入前三；best3=14000/11000/12000，加latest15000，实际保留11000/12000/14000/15000四份完整状态；remove为空，无新清理档案。原生checkpoint的trainer_state.json记录global_step=15000、epoch=1.0069144919023245，已越过本轮训练的第一个epoch；不据此声称原300000步目标完成或所有样本均无重复曝光。

15000主loss2.018878→2.044152；全遮蔽AA36.93%→34.92%、AA均值76.38%→75.93%、关系标签71.97%→71.48%、分隔符78.09%→75.48%回退，终止符82.76%→85.83%提高。关系标签分源asd_antibody87.89%→87.50%、tcr_native34.38%→34.77%、tcr_papers93.75%→95.31%、trait71.88%→68.36%；四个有标签来源仍各256。

按既有原始指标继续观察tcr_native：loss1.873917→2.022110、AA loss1.894525→2.042375、AA准确率43.07%→33.91%、split87.50%→71.48%，terminal99.22%持平。关系标签准确率略升但loss1.296756→1.450427，指标含义不同，不应仅凭准确率判断恢复。该来源仍有波动，原因尚未确定；未因此调参、读取序列或开展额外模型测试。保持现有配置及容器，下一16000步；恢复重放未执行。

16000步实际核对通过（11:13:22Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint16000_root_review.json，SHA256=707891937fa63032aa04dddbe91723120e7dd71bebfde47816e6446e1cccfe5e。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过，15000清理后六份JSON档案哈希一致。best3仍14000/11000/12000，加最新16000，保留11000/12000/14000/16000四份完整状态；原生epoch=1.0740454812452798。15000的首轮记录保存在原JSON档案，未执行恢复重放。

16000主loss2.044152→2.052560、全遮蔽AA34.92%→34.47%、AA均值75.93%→75.82%、终止符85.83%→84.60%回退，分隔符75.48%持平；关系标签71.48%→76.46%提高。分源关系标签asd_antibody87.50%→87.89%、tcr_native34.77%→50.78%、trait68.36%→75.00%提高，tcr_papers95.31%→92.19%回退；各有标签来源仍256。tcr_native整体loss2.022110→2.072725、AA loss2.042375→2.095836、AA准确率33.91%→33.00%回退，而关系标签loss1.450427→0.925839改善；不能用单项提升替代全面效果判断。最佳仍14000的2.018878，保持原配置至17000步，无新增模型测试、重启或因果归因。

17000步实际核对通过（11:27:19Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint17000_root_review.json，SHA256=eea2c53de92f164dc77ca36f4e71ef66cb3fc62ecf5f9102a94af5a53b480726。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；16000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest17000，实际保留11000/12000/14000/17000四份完整状态；原生epoch=1.1411764705882352。只读观察41678实际exit0后接续99500，训练进程与容器未重启；未执行恢复重放。

17000主loss2.052560→2.046143改善；全遮蔽AA34.47%→34.79%、AA均值75.82%→75.91%、分隔符75.48%→78.03%提高，终止符84.60%→82.92%、关系标签76.46%→75.29%回退。关系标签分源asd_antibody87.89%→89.06%，tcr_native50.78%→47.66%、tcr_papers92.19%→90.23%、trait75.00%→74.22%；四个有标签来源仍各256。tcr_native整体loss2.072725→2.043230、AA loss2.095836→2.067374、AA准确率33.00%→33.70%、split71.48%→87.50%改善，terminal99.22%持平；关系标签loss0.925839→0.843874下降但准确率回退，分别记录。不据此宣称全面改善、根因或下游binding效果，最佳仍14000。保持原配置至18000步，无额外模型测试。

18000步实际核对通过（11:40:09Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint18000_root_review.json，SHA256=3c3cacc9e959583d1d5d480b9618195f174a9257ca7c5a069f90e9e4ace55ff4。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；17000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest18000，实际保留11000/12000/14000/18000四份完整状态；原生epoch=1.2083074599311907。未执行恢复重放。

18000主loss2.046143→2.073688回升，连续四个后续验证点仍未优于14k最佳2.018878；不据此宣称长期收敛、过拟合或特定故障。全遮蔽AA34.79%→34.81%、AA均值75.91%→75.95%略升；终止符82.92%→81.75%、关系标签75.29%→70.61%、分隔符78.03%→75.36%下降。关系标签分源asd_antibody89.06%→87.89%、tcr_native47.66%→31.25%、trait74.22%→69.53%，tcr_papers90.23%→93.75%；四个有标签来源仍各256。

tcr_native整体loss2.043230→2.229023、AA loss2.067374→2.249873、AA准确率33.70%→33.65%、terminal99.22%→82.81%、split87.50%→71.09%，关系标签loss0.843874→1.517046；各项回退按原始日志记录，不能用总体AA小幅提高覆盖。现有训练继续，原因未确定，不改配置或开展额外模型测试，下一19000步。

19000步实际核对通过（11:53:58Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint19000_root_review.json，SHA256=27fc6861eed6dd66461868765050d010ee0e80ffc72f85b502a737539f27790d。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；18000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest19000，实际保留11000/12000/14000/19000四份完整状态；原生epoch=1.2754384492741462。未执行恢复重放。

19000主loss2.073688→2.064763改善，14k之后五个验证点尚未优于原最佳2.018878；不据此作收敛或过拟合归因。全遮蔽AA34.81%→34.82%、终止符81.75%→82.92%、关系标签70.61%→75.10%、分隔符75.36%→75.48%提高，AA均值75.95%→75.94%略退。关系标签四来源分别为asd_antibody87.89%→90.63%、tcr_native31.25%→42.97%、tcr_papers93.75%→94.53%、trait69.53%→72.27%，各来源仍256个标签。

tcr_native整体loss2.229023→2.076314、AA loss2.249873→2.099318、关系标签loss1.517046→0.681047改善；AA准确率33.65%→33.52%回退，terminal82.81%持平、split71.09%→71.48%略升。asd_antibody、tcr_repertoire与trait的整体loss较18k提高，不能以总体改善替代分源结果。未新增模型测试、调整训练或推断下游binding表现；保持原配置至20000步。

20000步实际核对通过（12:06:45Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint20000_root_review.json，SHA256=25fbbdbb9e9b244adcd43c218cb81aad21a2f5cd9b62fb70a3c8b854511a53ac。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；19000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest20000，实际保留11000/12000/14000/20000四份完整状态；原生epoch=1.3425694386171017。未执行恢复重放。

20000主loss2.064763→2.040837改善，14k之后六个验证点尚未优于原最佳2.018878；不据此作长期收敛或过拟合归因。全遮蔽AA34.82%→34.90%、AA均值75.94%→75.98%、关系标签75.10%→75.78%提高，终止符82.92%→81.47%、分隔符75.48%→75.42%回退。关系标签分源asd_antibody90.63%→89.45%、tcr_papers94.53%→91.41%下降，tcr_native42.97%→45.31%、trait72.27%→76.95%上升，各来源仍256个标签。

tcr_native整体loss2.076314→2.024939、AA loss2.099318→2.049700、AA准确率33.52%→34.01%、terminal82.81%→99.22%改善，split71.48%持平；关系标签loss0.681047→1.054381回升，不能以关系准确率小幅提高掩盖。asd_antibody与oas的terminal分别21.88%→13.28%、79.30%→61.33%下降，tcr_papers整体loss2.034677→2.046681回升。仅记录既有验证指标，原因未定，不推断下游binding效果，不改配置或新增模型测试，下一21000步。

21000步实际核对通过（12:20:23Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint21000_root_review.json，SHA256=cb250c8ce09e385192d7f877d8d2e9a2a2b6b7e9b5890e04d6f30a35c482e270。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；20000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest21000，实际保留11000/12000/14000/21000四份完整状态；原生epoch=1.4097004279600571。未执行恢复重放。

21000主loss2.040837→2.027940改善，仍未优于14k的2.018878。全遮蔽AA34.90%→35.05%、AA均值75.98%→76.07%、终止符81.47%→84.54%、分隔符75.42%→77.96%提高；关系标签75.78%→75.68%略退。关系标签分源asd_antibody89.45%→87.50%、tcr_papers91.41%→91.02%下降，tcr_native45.31%→46.48%、trait76.95%→77.73%上升，各有标签来源仍256条。

tcr_native整体loss2.024939→1.943660、AA loss2.049700→1.968401、AA准确率34.01%→34.38%、关系标签loss1.054381→0.850785、split71.48%→87.50%改善，terminal99.22%持平。asd_antibody的terminal13.28%→33.59%回升，oas61.33%持平；ots整体loss2.173933→2.214548、AA准确率27.93%→27.76%回退，tcr_papers的AA与terminal准确率亦回退。全遮蔽各源目标计数与20k一致，不能宣称全面改善或下游binding提升；保持既定配置，下一22000步。

22000步实际核对通过（12:33:45Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint22000_root_review.json，SHA256=ed85fed5eb3d88bef6a16910d4fdeb5e002ce072b79f13739b874091f3f4c8f8。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过；21000清理后六份JSON档案哈希一致。best3=14000/11000/12000，加latest22000，实际保留11000/12000/14000/22000四份完整状态；原生epoch=1.4768314173030124。未执行恢复重放。

22000主loss2.027940→2.052559回升，最佳仍14k的2.018878。全遮蔽AA35.05%→34.92%、AA均值76.07%→76.03%、终止符84.54%→80.41%、分隔符77.96%→75.42%回退，关系标签75.68%→79.49%提高。关系标签四来源分别为asd_antibody87.50%→90.63%、tcr_native46.48%→52.73%、tcr_papers91.02%→91.80%、trait77.73%→82.81%，各有标签来源仍256条。

tcr_native整体loss1.943660→2.142883、AA loss1.968401→2.171316、AA准确率34.38%→33.53%、split87.50%→71.48%回退，terminal99.22%持平；关系标签loss0.850785→0.645049改善。asd_antibody与oas的terminal分别33.59%→14.06%、61.33%→53.91%下降，ots整体loss与AA改善，tcr_papers及tcr_repertoire整体loss回升。全遮蔽各源目标计数与21k一致；不以关系标签提升覆盖生成目标回退，不据此确定原因或下游表现。保持既定配置，下一23000步。

23000步实际核对通过（12:47:52Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint23000_root_review.json，SHA256=d99e0001b3204dcb08ebce5df9412853330abcdef4c67c8e55dd43c724a7718f。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=14000/23000/11000，latest23000已在best3内，实际保留11000/14000/23000三份完整状态；12000和22000旧权重自动清理，两点共12份JSON档案哈希一致。原生epoch=1.543962406645968；未执行恢复重放。

23000主loss2.052559→2.025071改善，按既定主loss排序为当前第二，仍未优于14k的2.018878。全遮蔽AA34.92%→35.61%、AA均值76.03%→76.22%、分隔符75.42%→75.54%提高，终止符80.41%→80.25%、关系标签79.49%→72.85%回退。关系标签分源asd_antibody90.63%→87.50%、tcr_native52.73%→35.55%、trait82.81%→76.17%下降，tcr_papers91.80%→92.19%提高，各有标签来源仍256条。

tcr_native整体loss2.142883→1.912657、AA loss2.171316→1.933684、AA准确率33.53%→36.87%改善，terminal99.22%、split71.48%持平；关系标签loss0.645049→1.184413回升。asd_antibody与trait整体loss回升、终止符准确率分别14.06%→13.28%、99.22%→98.83%下降；tcr_repertoire整体loss亦略升。全遮蔽各源目标计数与22k一致。主loss排名提升不代表所有指标或下游binding改善；保持既定配置，下一24000步。

24000步实际核对通过（13:00:45Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint24000_root_review.json，SHA256=6dcb36ab6bd4ea75bdd31c9b203947d12eaec89673bd125367b5ace1612b4f32。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=14000/24000/23000，latest24000已在best3内，实际保留14000/23000/24000三份完整状态；11000旧权重自动清理，六份JSON档案哈希一致。原生epoch=1.6110933959889233；未执行恢复重放。

24000主loss2.025071→2.024323小幅改善，当前第二，最佳仍14k的2.018878。终止符80.25%→81.47%、关系标签72.85%→75.68%、分隔符75.54%→78.27%提高；全遮蔽AA35.61%→35.53%、AA均值76.22%→76.20%略退。关系标签分源asd_antibody87.50%→88.28%、tcr_native35.55%→44.92%、tcr_papers92.19%→93.75%提高，trait76.17%→75.78%下降，各有标签来源仍256条。

tcr_native整体loss1.912657→1.891072、AA loss1.933684→1.914587、AA准确率36.87%→37.44%、关系标签loss1.184413→0.925801、split71.48%→87.89%改善，terminal99.22%持平。asd_antibody终止符13.28%→21.09%提高但AA准确率下降；ots整体loss2.188934→2.211604、AA28.00%→27.62%，trait整体loss2.212857→2.238395、AA23.41%→21.95%回退，tcr_papers整体loss亦略升。全遮蔽各源目标计数与23k一致；不据此宣称全面改善、确定原因或下游binding表现。保持既定配置，下一25000步。

25000步实际核对通过（13:14:01Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint25000_root_review.json，SHA256=64fae6a7bfc9ab9908400e1517ca721580712d99f74a7dc18ca513bf231df314。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍14000/24000/23000，latest25000不在best3内，实际保留14000/23000/24000/25000四份完整状态；本次没有旧权重删除。原生epoch=1.6782243853318788；未执行恢复重放。

25000主loss2.024323→2.089885回升，最佳仍14k的2.018878。全遮蔽AA35.53%→34.63%、AA均值76.20%→75.97%、关系标签75.68%→74.51%、分隔符78.27%→74.92%回退，终止符81.47%→81.53%略升。关系标签分源asd_antibody88.28%→90.63%、tcr_papers93.75%→94.92%提高，tcr_native44.92%→39.45%、trait75.78%→73.05%下降，各有标签来源仍256条。

tcr_native整体loss1.891072→2.339390、AA loss1.914587→2.368018、AA准确率37.44%→33.19%、terminal99.22%→82.81%、split87.89%→71.48%回退；关系标签loss0.925801→0.902532略降但准确率下降。oas终止符53.91%→69.14%、ots及tcr_repertoire终止符升至100%，使总体terminal略升，不能覆盖tcr_native的边界回退。tcr_repertoire分隔符99.61%→93.75%亦下降；ots/trait整体loss改善。

对24k→25k七来源既有主loss作算术分解：净增量0.06556192469738864，tcr_native贡献0.06404538216381964（占净增量97.68685477040394%）；其余来源合计为剩余部分，含正负抵消。全遮蔽各源selected/AA/terminal/relation/split目标计数均未变；这只定位指标变化来源，不是优化器或数据故障的因果诊断。原始指标来自 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-24000_ce6f98b98cda/trainer_state.json 与 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-25000_ce6f98b98cda/trainer_state.json；未新增模型测试或改配置，下一26000步继续观察。

26000步实际核对通过（13:28:57Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint26000_root_review.json，SHA256=efd9583e40c5156e38eb9fcd74957d1df68b9472bd615207147352ec2badb50b。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍14000/24000/23000，加latest26000，实际保留14000/23000/24000/26000四份完整状态；25000旧权重自动清理、六份JSON档案哈希一致。原生epoch=1.7453553746748343；未执行恢复重放。

26000主loss2.089885→2.061376改善，仍高于24k的2.024323，最佳仍14k的2.018878。全遮蔽AA34.63%→34.87%、AA均值75.97%→76.09%、关系标签74.51%→74.61%、分隔符74.92%→75.48%提高；终止符81.53%→79.07%下降。关系标签分源asd_antibody90.63%、trait73.05%持平，tcr_native39.45%→41.02%提高，tcr_papers94.92%→93.75%下降，各有标签来源仍256条。

tcr_native整体loss2.339390→2.172965、AA loss2.368018→2.200295、AA准确率33.19%→33.35%仅部分恢复，仍未回到24k的loss1.891072及AA37.44%；terminal82.81%→99.22%恢复到24k水平，split71.48%持平，仍低于24k的87.89%。关系标签loss0.902532→0.960802回升，尽管准确率略升。asd_antibody与oas终止符20.31%→13.67%、69.14%→44.53%下降；tcr_repertoire分隔符93.75%→98.83%回升。ots/tcr_papers/trait整体loss回升。全遮蔽各源目标计数与25k一致；分项恢复程度不同，原因未确定，不据此改配置或推断下游binding效果，下一27000步。

27000步实际核对通过（13:41:13Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint27000_root_review.json，SHA256=d9d102f666b26d1486eda6964ab2a81e6257e2746eb96f4b5f1d243f42e76845。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=27000/14000/24000，latest27000已在best3内，实际保留14000/24000/27000三份完整状态；23000与26000旧权重自动清理，两点共12份JSON档案哈希一致。原生epoch=1.8124863640177897；未执行恢复重放。

27000主loss2.061376→2.011242，优于原14k的2.018878，成为当前主loss最佳。全遮蔽AA34.87%→35.64%、AA均值76.09%→76.29%、关系标签74.61%→84.96%提高，分隔符75.48%持平；终止符79.07%→70.93%下降。关系标签分源asd_antibody90.63%→87.50%、tcr_papers93.75%→86.72%下降，tcr_native41.02%→87.50%、trait73.05%→78.13%提高，各有标签来源仍256条。

tcr_native整体loss2.172965→1.849630、AA loss2.200295→1.872593、AA准确率33.35%→36.32%、关系标签loss0.960802→0.315292改善；其主loss已低于24k的1.891072，但AA仍低于24k的37.44%。terminal99.22%→82.42%回退，split71.48%持平。oas终止符44.53%→9.77%、asd_antibody13.67%→12.11%下降；其余四来源终止符也下降，七来源全遮蔽AA准确率则均提高。全遮蔽各源目标计数与26k一致；主loss最佳不代表边界重建或所有下游指标最佳，原因未定，保持既定配置，下一28000步重点观察终止符回退。

28000步实际核对通过（13:54:59Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint28000_root_review.json，SHA256=94d9c27661b57775f519ae6bf07e7b794c94f7aa8fa9b7892085a4bb86deb1d5。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=27000/28000/14000，latest28000已在其中，实际保留14000/27000/28000三份完整状态；24000旧权重自动清理，六份JSON档案哈希一致。原生epoch=1.8796173533607452；未执行恢复重放。

28000主loss2.011242→2.012737略升，当前第二，最佳仍27k。终止符70.93%→79.74%、分隔符75.48%→78.09%回升；全遮蔽AA35.64%→35.59%、AA均值76.29%→76.28%、关系标签84.96%→74.80%回退。关系标签分源asd_antibody87.50%→90.23%、tcr_papers86.72%→94.92%提高，tcr_native87.50%→38.28%、trait78.13%→75.78%下降，各有标签来源仍256条。

七来源终止符均较27k回升：oas9.77%→61.33%、asd_antibody12.11%→17.97%，tcr_native82.42%→82.81%；tcr_native仍低于26k的99.22%，其他六来源已达到或超过26k。tcr_native分隔符71.48%→87.89%、AA36.32%→37.02%提高，但关系标签loss0.315292→1.109715回升，整体loss1.849630→1.854652略升。oas/ots/tcr_repertoire的AA准确率下降，ots整体loss2.152263→2.202137回升。全遮蔽各源目标计数与27k一致；不能以边界回升掩盖关系标签回退，不作因果或下游binding结论，配置保持，下一29000步。

29000步实际核对通过（14:08:01Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint29000_root_review.json，SHA256=5bb6a9a093ff921e2690f94107f2bcbd099ebfe4ac03467ed7e078aa0bfe8241。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest29000不在其中，实际保留14000/27000/28000/29000四份完整状态；本次无新清理档案，先前清理证据保留。原生epoch=1.9467483427037005；未执行恢复重放。

29000主loss2.012737→2.019129回升，最佳仍27k。全遮蔽AA35.59%→36.69%、AA均值76.28%→76.56%、终止符79.74%→80.69%提高；关系标签74.80%→71.68%、分隔符78.09%→75.48%下降。四个有关系标签来源准确率均下降：asd_antibody90.23%→88.28%、tcr_native38.28%→34.77%、tcr_papers94.92%→94.53%、trait75.78%→69.14%，各来源仍256条。

tcr_native终止符82.81%→99.22%恢复到26k水平，AA37.02%→41.41%提高，但整体loss1.854652→1.880929、AA loss1.875086→1.902987回升，分隔符87.89%→71.48%下降。oas终止符61.33%→44.53%回退，asd_antibody17.97%→23.44%提高；ots终止符98.83%→100.00%。七来源全遮蔽目标计数与28k一致；指标升降并存，不由单点推断原因或下游生成质量，保持既定配置，下一30000步。

30000步实际核对通过（14:20:55Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint30000_root_review.json，SHA256=c9df2a3b1a48136b49ff11eafdddb2cb1d4f8eed1aa21bc7135cb08633dbfbf0。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest30000不在其中，实际保留14000/27000/28000/30000四份完整状态；29000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.013828983804649，是本运行首个已验收且epoch超过2的常规保存点；仅按Trainer口径记录，不据此声称每条独立记录均恰好访问两次。原300000步目标仍未完成，未执行恢复重放或新增生成测试。

30000主loss2.019129→2.039798回升，最佳仍27k。全遮蔽AA36.69%→34.90%、AA均值76.56%→76.09%、终止符80.69%→75.00%、关系标签71.68%→71.48%、分隔符75.48%→74.18%均下降。关系标签分源asd_antibody88.28%→87.50%、tcr_native34.77%→32.42%下降，tcr_papers94.53%→96.48%、trait69.14%→69.53%提高，各有标签来源仍256条。

tcr_native整体loss1.880929→2.055636、AA loss1.902987→2.079467回升，AA41.41%→33.80%、分隔符71.48%→71.09%下降，终止符99.22%持平。asd_antibody终止符23.44%→11.33%、分隔符26.67%→1.33%下降（分隔符计数仍75）；oas终止符44.53%→19.53%继续下降。全遮蔽七来源计数与29k一致，回退原因未定，不由跨epoch时间重合推断原因或下游生成质量；保持既定配置，下一31000步跟踪。

31000步实际核对通过（14:35:00Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint31000_root_review.json，SHA256=7ff4290150f71864be545d6991a0276e658306920306129754fd00562584d626。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest31000不在其中，实际保留14000/27000/28000/31000四份完整状态；30000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.0809599731476043；未执行恢复重放。首次越过两轮的30k Trainer证据已由 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512/checkpoint_retention_evidence/checkpoint-30000_ce6f98b98cda/trainer_state.json 保留，哈希与原完整状态一致。

31000主loss2.039798→2.042209略升，最佳仍27k。全遮蔽AA34.90%→35.03%、AA均值76.09%→76.17%、终止符75.00%→80.30%、分隔符74.18%→75.42%提高，关系标签71.48%→71.00%继续下降。关系标签分源asd_antibody87.50%持平；tcr_native32.42%→31.64%、tcr_papers96.48%→96.09%、trait69.53%→68.75%下降，各有标签来源仍256条。

asd_antibody终止符11.33%→13.28%、分隔符1.33%→26.67%提高，分隔符回到29k水平，终止符仍低于29k的23.44%；oas终止符19.53%→53.91%提高并超过29k的44.53%。tcr_native整体loss2.055636→2.070189、AA loss2.079467→2.094536、关系loss1.130686→1.193347继续回升，AA33.80%→33.35%继续下降；终止符99.22%、分隔符71.09%持平。七来源全遮蔽目标计数与30k一致；边界指标部分恢复，不等于全面改善，原因未定，保持既定配置，下一32000步。

32000步实际核对通过（14:47:50Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint32000_root_review.json，SHA256=4109332c8e730c29df30c06e8790d23d3ccaa5c1338d044c479bc5853454b54f。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest32000不在其中，实际保留14000/27000/28000/32000四份完整状态；31000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.1480909624905595；未执行恢复重放，历史一轮/两轮及分源分析证据保留。

32000主loss2.042209→2.061845上升，最佳仍27k；从27k到32k的六份已保存Trainer元数据核实，27k后的五次相邻变化均为上升（2.011242→2.012737→2.019129→2.039798→2.042209→2.061845）。根交接v5_loss_trend_at32000保存六份原件路径与哈希。这是固定验证集上的记录趋势，尚不能判定过拟合、数据问题或其他原因，不改变原300000步计划。

终止符80.30%→82.92%、关系标签71.00%→71.19%、分隔符75.42%→75.48%提高；全遮蔽AA35.03%→35.02%、AA均值76.17%→76.14%略降。关系标签分源asd_antibody87.50%→86.72%、tcr_papers96.09%→94.53%下降，tcr_native31.64%→32.81%、trait68.75%→70.70%提高，各有标签来源仍256条。

tcr_native整体loss2.070189→2.177002、AA loss2.094536→2.201422继续回升，但AA33.35%→33.52%、关系准确率31.64%→32.81%、分隔符71.09%→71.48%提高，关系loss1.193347→1.054144下降，终止符99.22%持平。asd_antibody终止符13.28%→29.69%提高，分隔符26.67%持平；oas终止符53.91%持平。七来源终止符均未下降，全遮蔽目标计数与31k一致；不将主loss趋势或边界改善外推为下游生成质量，下一33000步继续核对。

33000步实际核对通过（15:01:44Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint33000_root_review.json，SHA256=120e726fe9c70ad60116349f3311a4351db96cd6fcdb420d208fb2f62deca8c1。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest33000不在其中，实际保留14000/27000/28000/33000四份完整状态；32000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.2152219518335152；未执行恢复重放，历史证据保留。

33000主loss2.061845→2.026425下降，结束27k后连续五次上升；仍高于27k最佳2.011242。根交接保留原v5_loss_trend_at32000历史快照，另以v5_loss_trend_followup_at33000记录本次回落，避免旧趋势被误当成当前结论。全遮蔽AA35.02%→35.14%、AA均值76.14%→76.22%、终止符82.92%→83.71%、分隔符75.48%→78.09%提高；关系标签71.19%→66.89%下降。

关系标签分源asd_antibody86.72%→87.89%、tcr_papers94.53%→96.09%、trait70.70%→71.48%提高，tcr_native32.81%→12.11%下降，各有标签来源仍256条。native整体loss2.177002→1.956067、AA loss2.201422→1.977388改善，AA33.52%→33.68%、分隔符71.48%→87.89%提高，终止符99.22%持平；但关系loss1.054144→1.424468上升。主loss改善不能替代关系指标检查。oas终止符53.91%→61.33%提高，asd_antibody29.69%→29.30%、ots100.00%→98.83%下降。七来源全遮蔽目标计数与32k一致；原因未定，不作下游效果推断，保持既定配置，下一34000步重点跟踪native关系回退。

34000步实际核对通过（15:14:38Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint34000_root_review.json，SHA256=274f19149f5103e6bed0e9d60a04543843adae153626bf5183f87164a027bfe8。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest34000不在其中，实际保留14000/27000/28000/34000四份完整状态；33000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.2823529411764705；未执行恢复重放，历史证据保留。

34000主loss2.026425→2.066092回升，最佳仍27k。全遮蔽AA35.14%→35.20%、AA均值76.22%→76.23%、分隔符78.09%→80.01%提高；终止符83.71%→79.69%、关系标签66.89%→64.16%下降。关系分源asd_antibody87.89%持平，tcr_native12.11%→3.91%、tcr_papers96.09%→95.70%、trait71.48%→69.14%下降，各有标签来源仍256条。

tcr_native整体loss1.956067→2.159799、AA loss1.977388→2.186577、关系loss1.424468→1.498463回升；AA33.68%→35.57%、终止符99.22%→99.61%、分隔符87.89%→100.00%提高，但全遮蔽关系准确率仅10/256。asd_antibody终止符29.30%→15.23%、oas61.33%→44.53%下降。七来源全遮蔽目标计数与33k一致；不作原因或下游效果推断，保持既定配置，下一35000步重点跟踪关系回退。

针对连续两次native关系明显回退，只读对照27k、32k、33k、34k已保存的四遮蔽等级指标。表中为关系标签准确率，括号内为该等级固定的被评分关系token数量；各等级使用的被评分子集大小不同，不将跨等级差异当作同一子集对照。原件路径/哈希、loss及数量保存在根交接v5_native_relation_mask_watch_at34000；未加载模型、序列或重跑评估。

| step | mask25（70） | mask50（140） | mask75（189） | mask100（256） |
|---|---|---|---|---|
| 27000 | 87.14% | 85.71% | 86.77% | 87.50% |
| 32000 | 77.14% | 76.43% | 79.89% | 32.81% |
| 33000 | 78.57% | 77.86% | 80.95% | 12.11% |
| 34000 | 75.71% | 74.29% | 78.84% | 3.91% |

全遮蔽下降更剧烈，但其余三个等级也较33k下降且低于27k；不能表述为只有全遮蔽退化，也不能凭这些聚合指标确定原因。下一35000步继续按同口径记录，保留本次历史快照。

35000步实际核对通过（15:28:18Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint35000_root_review.json，SHA256=bdbbdc56350d5f94e06342b4dbaefc3dfa1cdf7a364c590fd33a84e1a9cbb178。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest35000不在其中，实际保留14000/27000/28000/35000四份完整状态；34000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.349483930519426；未执行恢复重放，历史证据保留。

35000主loss2.066092→2.032845下降，最佳仍27k。关系标签64.16%→75.88%、AA均值76.23%→76.26%提高；全遮蔽AA35.20%→35.18%、终止符79.69%→75.17%、分隔符80.01%→78.09%下降。关系分源asd_antibody87.89%→91.02%、tcr_native3.91%→43.36%、trait69.14%→75.39%提高，tcr_papers95.70%→93.75%下降，各有标签来源仍256条。

native关系分遮蔽跟踪：mask25/50/75/100准确率分别80.00%/77.86%/80.95%/43.36%，均较34k提高，各等级数量仍70/140/189/256；但四等级均未回到27k的87.14%/85.71%/86.77%/87.50%。根交接v5_native_relation_mask_followup_at35000保留本次原件哈希和统计，34k历史快照不覆盖。仅为固定验证集上的标签重建，不作下游binding效果或原因推断。

tcr_native整体loss2.159799→2.017229、AA loss2.186577→2.042138、关系loss1.498463→1.113264改善，但AA35.57%→33.90%、终止符99.61%→99.22%、分隔符100.00%→87.89%下降。asd_antibody终止符15.23%→11.72%、oas44.53%→19.53%下降；七来源中六来源终止符下降，trait持平。全遮蔽各源目标计数与34k一致，原配置保持，下一36000步继续观察关系和边界波动。

36000步实际核对通过（15:41:43Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint36000_root_review.json，SHA256=7c1e0d2700233fce8a3c613484636713563363f45e993c33b51febcbcb18fbf2。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest36000不在其中，实际保留14000/27000/28000/36000四份完整状态；35000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.4166149198623814；未执行恢复重放，历史证据保留。

36000主loss2.032845→2.022820继续下降，最佳仍27k。终止符75.17%→80.64%提高；全遮蔽AA35.18%→34.98%、AA均值76.26%→76.20%、关系标签75.88%→73.63%、分隔符78.09%→75.42%下降。关系分源asd_antibody91.02%→88.67%、tcr_native43.36%→39.45%、trait75.39%→72.66%下降，tcr_papers93.75%持平，各有标签来源仍256条。

native mask25/50/75/100关系准确率75.71%/75.00%/78.84%/39.45%，均较35k下降，各等级数量仍70/140/189/256；全遮蔽关系loss1.113264→1.019317下降，其余三级关系loss回升。根交接v5_native_relation_mask_followup_at36000保存原件哈希与统计，34k/35k历史快照保留。标签重建指标不外推为下游binding效果或原因判断。

tcr_native整体loss2.017229→1.982058、AA loss2.042138→2.003400改善，但AA33.90%→33.50%、分隔符87.89%→71.09%下降，终止符99.22%持平。asd_antibody终止符11.72%→13.67%、oas19.53%→53.91%提高；五来源终止符提高、native持平、trait99.22%→98.83%下降。全遮蔽各源目标计数与35k一致，原配置保持，下一37000步继续观察。

37000步实际核对通过（15:54:47Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint37000_root_review.json，SHA256=190b559804ae28f39041f733b2b2ba3e7350f31246c14c1e73efcb976c0e34bf。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest37000不在其中，实际保留14000/27000/28000/37000四份完整状态；36000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.4837459092053367；未执行恢复重放，历史证据保留。

37000主loss2.022820→2.056276回升，最佳仍27k。全遮蔽AA34.98%→34.39%、AA均值76.20%→76.06%、终止符80.64%→76.23%、关系标签73.63%→73.34%下降，分隔符75.42%→75.54%略升。关系分源asd_antibody88.67%→87.50%、tcr_papers93.75%→93.36%、trait72.66%→72.27%下降，tcr_native39.45%→40.23%提高，各有标签来源仍256条。

native mask25/50/75/100关系准确率82.86%/80.00%/80.95%/40.23%，均较36k提高，四等级关系loss均下降，各等级数量仍70/140/189/256。根交接v5_native_relation_mask_followup_at37000保存原件哈希与统计，历史快照保留。仅为固定验证集上的标签重建，不作下游binding效果或原因判断。

tcr_native整体loss1.982058→2.118852、AA loss2.003400→2.143043回升，尽管关系loss1.019317→0.908726下降；AA33.50%→33.53%略升，终止符99.22%、分隔符71.09%持平。asd_antibody AA48.69%→45.03%、终止符13.67%→12.89%下降，分隔符26.67%→29.33%提高；oas终止符53.91%→25.39%下降。全遮蔽各源目标计数与36k一致，变化原因未定，保持配置，下一38000步继续观察。

38000步实际核对通过（16:11:39Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint38000_root_review.json，SHA256=e8c7e2ce7a5c9af4f5b0e47a6eb9f3a814741231951f78c73300ed4c5ddf7409。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest38000不在其中，实际保留14000/27000/28000/38000四份完整状态；37000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.5508768985482924；未执行恢复重放。

38000主loss2.056276→2.057750略升，最佳仍27k。全遮蔽AA34.39%→35.16%、AA均值76.06%→76.23%、终止符76.23%→83.09%提高；关系标签73.34%→71.88%、分隔符75.54%→75.42%下降。关系分源asd_antibody87.50%→87.89%、tcr_papers93.36%→95.70%提高，tcr_native40.23%→32.03%、trait72.27%→71.88%下降，各有标签来源仍256条。

native mask25/50/75/100关系准确率75.71%/74.29%/78.31%/32.03%，均较37k下降，四等级关系loss均回升，各等级数量仍70/140/189/256。根交接v5_native_relation_mask_followup_at38000保留原件哈希与统计。仅为既有验证集上的标签重建指标，不作下游binding效果或原因判断。

tcr_native整体loss2.118852→2.197570、AA loss2.143043→2.220937、关系loss0.908726→1.137300回升，但AA33.53%→33.70%、分隔符71.09%→71.48%提高，终止符99.22%持平。asd_antibody AA45.03%→48.86%、终止符12.89%→25.00%提高，分隔符29.33%→26.67%下降；oas终止符25.39%→61.33%提高。全遮蔽各源目标计数与37k一致。现有容器和训练继续、占卡进程为空，未新增模型测试或修改配置，下一39000步。

39000步实际核对通过（16:22:10Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint39000_root_review.json，SHA256=221ed57435d6989f366cf91f37e4c68ffecb6fedcd2b337d20961de5b11ef7cd。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3仍27000/28000/14000，latest39000不在其中，实际保留14000/27000/28000/39000四份完整状态；38000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.618007887891248；未执行恢复重放。

39000主loss2.057750→2.040339下降，最佳仍27k。AA均值76.23%→76.24%、全遮蔽关系标签71.88%→74.12%提高；全遮蔽AA35.16%→35.05%、终止符83.09%→79.24%、分隔符75.42%→74.49%下降。关系分源asd_antibody87.89%→89.45%、tcr_native32.03%→39.06%、trait71.88%→74.61%提高，tcr_papers95.70%→93.36%下降，各有标签来源仍256条。

native mask25/50/75/100关系准确率75.71%/76.43%/79.37%/39.06%，mask25较38k持平，其余三级提高；四等级关系loss均下降，数量仍70/140/189/256。根交接v5_native_relation_mask_followup_at39000保留原件哈希和统计。native整体loss2.197570→2.035567、AA loss2.220937→2.058346、关系loss1.137300→0.826621改善，但分隔符71.48%→71.09%下降；AA33.70%→33.78%提高，终止符99.22%持平。

asd_antibody终止符25.00%→14.84%、oas61.33%→44.53%、tcr_repertoire分隔符98.83%→93.36%下降。全遮蔽各源目标计数与38k一致；这些是既有验证集上的重建指标，不作下游效果或原因判断。原容器和训练继续，配置保持，下一40000步。

40000步实际核对通过（16:34:59Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint40000_root_review.json，SHA256=ea1a1facbcc389cdf73c23176320ee6efdcfabfbe3f50d068f2782be180b47ff。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=40000/27000/28000，latest40000已在其中，实际保留27000/28000/40000三份完整状态；14000及39000旧权重自动清理，两点共12份JSON档案哈希一致。原生epoch=2.6851388772342033；未执行恢复重放。

40000主loss2.040339→2.000729下降，优于原27k的2.011242，成为当前主loss最佳。AA均值76.24%→76.50%、全遮蔽AA35.05%→36.15%、分隔符74.49%→75.36%提高；终止符79.24%→76.90%、关系标签74.12%→71.00%下降。关系分源asd_antibody89.45%→87.89%、tcr_native39.06%→31.25%、trait74.61%→69.92%下降，tcr_papers93.36%→94.92%提高，各有标签来源仍256条。

native mask25/50/75/100关系准确率75.71%/74.29%/78.31%/31.25%，mask25较39k持平、其余三级下降；四等级关系loss均回升，数量仍70/140/189/256。根交接v5_native_relation_mask_followup_at40000保留原件哈希和统计。native整体loss2.035567→1.786770、AA loss2.058346→1.805653改善，AA33.78%→38.62%、分隔符71.09%→71.48%提高；但终止符99.22%→82.81%下降，关系loss0.826621→1.054800回升。

asd_antibody终止符14.84%→13.67%、trait99.22%→98.83%、tcr_papers99.61%→99.22%下降；oas44.53%持平，ots98.83%→100.00%、tcr_repertoire98.44%→99.22%提高。全遮蔽各源目标计数与39k一致。主loss最佳不代表边界、关系或下游效果全面改善；保持配置，下一41000步。只读窗口66245实际exit0后接续5765，原容器和训练未重启。

41000步实际核对通过（16:48:53Z），报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint41000_root_review.json，SHA256=70eaa261f833c5a3ee5584ec4531e2fe3fa00e35c848b86581b99183f221f55d。17文件、八rank、90项核心聚合、参数更新及实际保留目录通过。best3=40000/27000/41000，latest41000已在其中，实际保留27000/40000/41000三份完整状态；28000旧权重自动清理，六份JSON档案哈希一致。原生epoch=2.7522698665771586；未执行恢复重放。

41000主loss2.000729→2.011555回升，略高于27k的2.011242、低于28k的2.012737，按主loss排名第三，最佳仍40k。全遮蔽关系标签71.00%→81.74%、终止符76.90%→80.58%提高；全遮蔽AA36.15%→35.73%、AA均值76.50%→76.42%、分隔符75.36%→74.49%下降。关系分源tcr_native31.25%→73.83%、trait69.92%→74.61%提高，asd_antibody87.89%→87.50%、tcr_papers94.92%→91.02%下降，各有标签来源仍256条。

native mask25/50/75/100关系准确率75.71%/76.43%/79.89%/73.83%，mask25较40k持平、其余三级提高；四等级关系loss均下降，数量仍70/140/189/256。根交接v5_native_relation_mask_followup_at41000保留原件哈希和统计。native终止符82.81%→99.22%恢复到39k水平，关系loss1.054800→0.613305改善；但整体loss1.786770→1.819363、AA loss1.805653→1.841445回升，AA38.62%→36.75%、分隔符71.48%→71.09%下降。

oas终止符44.53%→53.91%提高，asd_antibody13.67%→13.28%、tcr_repertoire分隔符98.83%→93.36%下降。全遮蔽各源目标计数与40k一致；分项升降并存，不作原因或下游效果判断。原容器、训练及只读观察5765继续，下一42000步，配置保持。

## 2026-10-07 23:33 UTC 接续观察及42000–71000步记录补齐

上一人工完整核对为41000步；只读观察5765在17:29:45自然结束，最后观察44140步，实际exit0于本次用户询问时确认。期间容器训练持续推进，但不能宣称人工监督持续在线。23:33:32重新只读查询确认平台Running、正式71395步、无可见错误；未重启容器或训练，新的3600秒观察句柄26248。

42k至70k的29点已核对保存JSON哈希、原生八rank回执、2610项核心指标聚合及历史保留决定；仍保留的42k/45k/68k另核对各17个完整状态文件。已被原生策略清理的其他旧点不能追溯核对其二进制权重，也没有重跑训练、验证或恢复。证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint42000_70000_metadata_catchup.json，SHA256=d94c0761b4c6464361ef7576dfa57df3d9fd139388fd4d4ffa5a1c399d73c631。上方两张表新增42k–71k原生结果，历史数字保持。

71k完整核对通过：17文件、八rank、90项核心聚合、参数更新及保留目录；70k清理后六份JSON档案哈希通过。证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint71000_root_review.json，SHA256=90b5074e3d3efe92149e49232aa2a9830a7324f2cc881cf558eb37fe651709af。主eval_loss=2.011193，epoch=4.766098850381807；当前best3=45000/42000/68000，加latest71000，实际保留42k/45k/68k/71k四份完整状态。

最佳45k主loss=1.992721，AA全遮蔽37.35%、终止符80.69%、关系标签71.29%、分隔符77.90%；最新71k分别为2.011193、35.12%、81.53%、71.58%、75.42%。45k之后至71k未刷新最佳；相较70k，71k主loss下降、AA与终止符略升、关系标签下降、分隔符持平。仅为同一运行固定验证集的重建结果，不代表与旧ESMC V5或V7的同口径下游比较。当前新版本尚无下游成绩，继续原300000步计划，下一72000步。

## 2026-10-08 当前新模型全下游评测

按用户要求固定best45000（val1.992721、epoch3.02074），硬链接独立评测快照，原训练继续。37项现役Ours任务含UpTCR、AB–AG；原划分和评分器保留，新模型输入/生成协议独立标记。CPU表征、批量一致性、CDR隐藏答案隔离及无目标终止符检查通过；4组独立单卡作业已提交c20250601，当前均Queue；模型训练保持运行。计划：/project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/plan.json。结果统一更新RESULTS.md，不覆盖历史模型表。

## 2026-10-07 72000步完整验收及历史loss对照

72k核对通过17完整状态文件、八rank与90项核心聚合；epoch=4.833229839724763，主eval_loss=1.9996342094258956。best3仍45k/42k/68k，加latest72k；71k原生清理后六份JSON哈希通过。未执行恢复重放。报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint72000_root_review.json，SHA256=ad1f8c8f1b80d8e6787c5a75f9e5e816956cf63ed48263edadd9b2c4e5597da4。

训练及验证历史曲线已整理到 [V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md](/project_c20250601/251105016/project/dllm_test/docs/V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md)；仅使用已保存日志、配置和源码，不重跑模型。旧no-ESMC基线实际V6 binding-only；当前V5全遮蔽主验证与旧随机遮蔽验证并非同协议。旧日志178k→179k训练loss尺度突变，历史未提交代码/进程信息不完整，不擅自除8修正。两版原始数字及独立曲线不能证明架构优劣。当前配置保持。

2026-10-08：用户授权非闲时资源；原4个Queue作业cancel后均Killed，已提交4个非闲时1GPU替代作业。checkpoint、输入和评分协议不变；新ID及状态见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/submissions.json。

## 2026-10-08 17:40 UTC 训练状态恢复核实及153000步验收

只读平台查询：2026-10-08T17:42:25.442699+00:00，训练153097/300000，平台与训练均running；supervisor737、training1283，standby为空。旧观察26248于00:33:32结束（实际工具回收exit0）；至本次恢复期间没有持续工具监督，不把旧75,903步快照当实时。

153k完整核对17文件哈希、八rank与90项核心聚合通过，未执行恢复重放；报告 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint153000_root_review.json，SHA256=ba22c1fd00db72995e75eed41ca24c1fbdb356409bf0fb9a1e1fd6416d626956。152k核对恰逢原生153k轮换，因目录集合已变化而exit1；这次核对没有通过，也不是训练故障。随后核对稳定的153k通过；未重试旧权重或更改训练。

| 步数 | 主验证loss | 四档AA均值/token-micro | 全遮蔽AA/token-micro | 全遮蔽关系标签重建 |
|---|---:|---:|---:|---:|
| 45000 | 1.992721 | 76.84% | 37.35% | 71.29% |
| 72000 | 1.999634 | 76.41% | 35.24% | 72.95% |
| 100000 | 2.016764 | 76.44% | 35.22% | 74.12% |
| 125000 | 2.007419 | 76.40% | 34.99% | 76.37% |
| 138000 | 1.991442 | 76.48% | 35.28% | 71.97% |
| 150000 | 2.020098 | 76.44% | 35.11% | 75.10% |
| 152000 | 2.026603 | 76.43% | 35.11% | 76.37% |
| 153000 | 2.008281 | 76.41% | 35.11% | 73.05% |

当前best3为138k/45k/42k，加latest153k共4份完整状态；最新epoch10.270538。历史73k–152k表格数字来自原生Trainer历史，本轮没有对每个历史checkpoint做完整二进制验收。主验证best较旧45k仅约0.064%改善，仍属于平台附近波动；标签重建不等于下游binding能力。最近10,000步墙钟均速0.7969秒/更新，含该时间窗口内验证/保存；按此速度剩余约32.5小时，仅为不中断情况下估计。训练配置与原数据门禁不变。证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/status_20261008T1740_root_review.json。

## 2026-10-08 训练退出至占卡监督

现有container.json绑定的supervisor/helper/standby三份源码哈希均一致；源码已实现训练退出后先保存结果，再在外循环恢复八卡standby。当前训练运行、standby为空、无STOP/PAUSE。未改变容器、源码绑定、训练或资源票据。

新增只读观察器复用watch_v5_direct.observe，50小时有界，本地PID1100460、exec会话90153；使用独占文件锁，记录当前心跳与错误，等待真实训练退出，再检查八个不同PID及各卡属于新PID的GPU初始化日志。CPU五项检查通过（当前训练非standby、八卡正例、training状态拒绝、重复PID拒绝、初始化报错拒绝），临时fixture已删除，无GPU测试或checkpoint。首次实际观察2026-10-08T17:53:16.896818+00:00训练153929/300000、running；占卡切换尚未发生。准备记录 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/setup_review.json；后续状态 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/latest.json，终态 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。本观察器只确认训练进程退出与占卡，不自动宣告最终模型验收或Goal完成。

2026-10-08 Kong：空轻链兼容修复经CPU检查通过；按用户授权单项重提，原失败证据保留，其他任务不重跑。见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/kong_retry1.json。

2026-10-08 全CDR协议对齐：9个CDR目标、空轻链路径的实际CPU推理通过；复用AirGen原生成逻辑，参考答案输入隔离通过；Kong/SAb23H2两项非闲时作业已提交。协议边界及结果见RESULTS.md新独立区，提交证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_airgen_aligned_20261008/submissions.json。


### 2026-10-09 评测修复

根因：T1基线比较文件错目录；CHIMERA批量ANARCI被单序列TypeError中断；TCR beta-only采样允许dot而解析要求单链。已有预测保留，评分恢复与单链新协议复测本地运行。22项回归验证通过。未将协议运行成功视为泛化或生成质量改善。

追加核查：60项输入协议回归通过（累计82项）；另独立重算4轨×5折×3集合的60组AUROC，全部与原指标吻合。未发现评分汇总错误，未见表位泛化弱仍保留为模型评测结果。


2026-10-09：固定latest269000进行37项评测，选点依据为最新完整保存而非验证loss最优；继续使用修复后的CDR/TCR协议。选点证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009/selection.json，结果独立写入RESULTS.md。


### 2026-10-09 CDR iteration 对照

按用户要求增加infilling迭代次数对照，预先固定1/2/4/8/16/32，覆盖9类CDR。只改变max_iter，报告全矩阵，不以测试集最佳步数代替默认协议。


### 2026-10-09 验证最佳138000评测

按用户要求增加验证最佳138000步完整评测及CDR iteration矩阵，选点依据是训练验证loss，不以任何下游测试成绩选点。与45000和269000的相同协议分开对照。


2026-10-09 修复复测已终态：{"t1-cdr3b": "success", "ag-chimera": "success", "t4-benchmark14": "success", "t4-held20": "success"}。结果及CDR3语法有效率见 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/repair_final_20261009.json；原始候选保留，MMseqs临时目录已清理。82项回归及60组T1 AUROC复核通过；未见表位泛化弱仍未由流程修复解决。


## 2026-10-10T02:07:51.760263+00:00 训练至占卡只读监督结束

监督结果=training_exited_standby_observed；训练票据状态=succeeded；八卡占卡恢复观察=True。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。

## 2026-10-10 270M新尺寸CPU验收与GPU检查启动

CPU实例化参数268,590,112；16层/d1024/head16/FFN4096。独立270M门禁的有效绑定、错误尺寸、training_ready=false、失败审核、等长字节篡改五项CPU检查通过；临时fixture已删除。旧冻结包和原门禁未变。元数据check-only通过。synthetic票据v5_270m_profile_bs64_20261010.json已进入当前持续容器；检查代码来源差异只含尺寸入口和独立门禁，见 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_profile_bs64_20261010/preparation.json。GPU结果待实际回执，尚无新训练checkpoint。

## 2026-10-10 270M八卡synthetic通过并发布正式训练

8/8 rank整模3步通过，参数均268590112，micro64/长度1218；峰值allocated42.7879705GiB，reserved48.87890625GiB；rank0 loss2.213626→0.145421→0.161268。NCCL barrier通过，profile票据成功退出0，全部8个旧standby已wait回收；无checkpoint。profile原件 /project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/profile_bs64。

正式包 /project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010 与profile源码/尺寸/batch契约完全一致，复用验收、重新从随机种子初始化；独立训练票据v5_270m_scratch_bs64_20261010.json已发布，输出独立，未重启容器或修改旧训练产物。真正开训仍须85分片全哈希门禁及原生更新步。

## 2026-10-10 270M正式训练已启动

2026-10-10T08:14:43.885827+00:00：真实更新67/300000；首个25步日志loss3.2884有限；当前日志loss2.8957。V5训练7,626,885条、85分片全哈希核对通过；config d1024/L16/head16/FFN4096与268590112参数检查一致，micro64/GA1/world8/global512。真实训练峰值allocated40.79GiB/reserved46.73GiB，早期普通步中位2.481秒，不作完整窗口/最终吞吐结论。监督PID1737482、exec26867最多15天；旧75.6M观察90153实际退出0，旧模型最终17文件/八rank/COMPLETE与成功退出封存通过，证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/formal_completion_root_review_20261010.json。270M启动证据 /project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/formal_start_review.json。没有新测试checkpoint或独立CUDA算子工作；旧序列注释停线和数据审核门禁保持。

## 2026-10-11 270M训练进度与28000步保存核对

2026-10-11本次核实已到28082/300000步（约9.36%）；最新完整保存28000步/epoch1.8797，累计28次原生验证。主eval_loss（全遮蔽/source-macro）最佳17000步1.993521，最新28000步2.092312，近期波动，尚无持续下降结论。最近10000更新墙钟平均2.8864秒/步，包含窗口内验证/保存；不中断约余9.1天。 训练和容器均running，占卡为空，原只读监督仍运行。保留best3（17000/16000/14000）＋latest28000共4份完整状态，其余旧权重按规则轮换。28000步完整核对通过，未执行恢复重放；本次无提交、重启或训练配置变动。证据：[状态与计时](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261011_checkpoint28000_root_review.json)、[28000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint28000_root_review.json)。

| 步数 | 主验证loss | 四档AA均值/token-micro | 全遮蔽AA/token-micro |
|---|---:|---:|---:|
| 1000 | 2.133271 | 70.13% | 33.13% |
| 5000 | 2.054368 | 75.94% | 36.05% |
| 10000 | 2.075569 | 75.97% | 34.95% |
| 14000 | 2.013956 | 76.28% | 35.75% |
| 16000 | 2.001041 | 76.50% | 36.34% |
| 17000 | 1.993521 | 76.59% | 36.98% |
| 20000 | 2.049416 | 76.15% | 34.80% |
| 25000 | 2.053179 | 76.09% | 34.27% |
| 27000 | 2.020156 | 76.29% | 35.13% |
| 28000 | 2.092312 | 76.28% | 35.01% |

28000步训练日志loss=0.4048；训练与主验证遮蔽/聚合口径不同，不直接比较数值高低。28次验证来自原生Trainer历史，本次独立完整文件核对仅28000步，17文件/8rank/90字段通过；未执行checkpoint恢复重放，未检验270M下游效果。计时18001→28001实际10000更新，墙钟约28864.421秒；预测完成UTC=2026-10-20T08:46:10.538989+00:00，只是在吞吐不变且不中断条件下的估算。

## 2026-10-12 270M训练进度与60000步保存核对

2026-10-12本次核实已到60305/300000步（约20.10%）；最新完整保存60000步/epoch4.0279，累计60次原生验证。主eval_loss（全遮蔽/source-macro）最佳32000步1.993467，最新60000步2.007204；相对17000步旧最佳仅改善0.0027%，整体仍波动。最近10000更新墙钟平均2.8940秒/步，包含窗口内验证/保存；不中断约余8.0天。 训练和容器均running，standby为空，原只读监督PID1737482/exec26867仍运行。保留best3（32000/17000/43000）＋latest60000共4份完整状态，其余旧权重按规则轮换；正式恢复状态保留。本次无提交、重启或训练配置变动，未执行恢复重放。证据：[状态与计时](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261012_checkpoint60000_root_review.json)、[60000步核对](/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint60000_root_review.json)。

| 步数 | 主验证loss | 四档AA均值/token-micro | 全遮蔽AA/token-micro |
|---|---:|---:|---:|
| 17000 | 1.993521 | 76.59% | 36.98% |
| 28000 | 2.092312 | 76.28% | 35.01% |
| 32000 | 1.993467 | 76.83% | 37.15% |
| 40000 | 2.021273 | 76.31% | 35.11% |
| 43000 | 1.999736 | 76.94% | 37.41% |
| 50000 | 2.018846 | 76.43% | 35.21% |
| 55000 | 2.032598 | 76.44% | 35.12% |
| 57000 | 2.057908 | 76.46% | 35.23% |
| 58000 | 2.006649 | 76.52% | 35.29% |
| 59000 | 2.001417 | 76.50% | 35.41% |
| 60000 | 2.007204 | 76.39% | 35.18% |

60000步训练日志loss=0.3876（与主验证的遮蔽及聚合口径不同）。相较昨日28000步验证2.092312，本次2.007204回落；最佳32000与旧最佳17000的绝对差仅0.00005391，不能据此宣称质量显著改善。60次验证来自原生历史，本次完整二进制核对仅60000步，17文件/8rank/90字段通过；未执行恢复重放。本轮没有进行270M下游评测。计时50001→60001实际10000更新、28939.890秒；预测完成UTC=2026-10-20T09:20:31.817642+00:00，仅为吞吐不变且不中断条件下的估算。


2026-10-12：三checkpoint全下游与两组iteration已全部完成，完整同协议对照置于RESULTS.md顶部。平台终态与汇总证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/completed_comparison_20261012.json。没有统一最优checkpoint，未见表位泛化及抗原条件生成仍是短板。


2026-10-12：启动270M验证最佳32000步的37项下游评测，四个非闲时单卡组；沿用75.6M修复后协议，CDR Kong iter4、SAb iter2，不按测试分数选点。真实权重CPU门禁通过。结果唯一维护于 /project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md；证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012 。


2026-10-12：按用户要求先提交当前代码快照e8ad5fb（未push），再准备相同270M checkpoint32000的旧CDR入口对照。旧run_v5_cdr_eval及六个依赖SHA与45000历史manifest一致；原V7评测共用argmax和评分器，输入适配不同。协议差异见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。不更改运行中AirGen源码与成绩。

2026-10-12：旧入口对照已提交两个非闲时单卡任务，九类CDR完整评测；Kong H3十折小样本、SAb六模式CPU预检通过。任务与状态见PROJECT_PROCESS.md，结果自动回填RESULTS.md独立“旧版CDR入口对照”表。


2026-10-12：Ophiuchus-Ab本地采样复现核查完成，11组数据集设置/51个CDR指标/49,065条区域预测独立重算通过。官方权重严格加载，11份对应主干源码与AirGen一致；本次范围内Kong argmax1/bf16、SAb argmax2/fp32最接近论文，仍非逐项精确复现。数字见RESULTS.md，方案和证据见 /project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。本地运行退出码0，无新checkpoint、无云提交。


2026-10-12：270M固定32000新坐标/X掩码的单步CDR本地评测完成；9741条区域预测独立重算通过，75.6M两组既有单步也重算通过。结果及原多步差值已回填RESULTS.md。模型硬链接自动释放，原训练权重未动。证据 /project_c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_single_step270m_20261012 。


2026-10-12：用户要求回归f011残差及双链/每链RoPE机制，保留当前输入语法。默认保留270M尺寸和V5数据；已核实f011开关可复用，但V5 collator需同步修改，不能只改variant。链边界采用固定槽位或自由split动态编号待用户选择；本次只形成设计记录，未完成实现、未启动训练。记录 /project_c20250601/251105016/project/dllm_test/configs/models/v5_270m_f011_chain_return_20261012/README.md 。


2026-10-12 更新：用户确认采用更小的 **75.6M固定双链f011**，保留V5外层输入语法。实现见 [v5_fixed_chain_f011_75m_20261012](/project_c20250601/251105016/project/dllm_test/configs/models/v5_fixed_chain_f011_75m_20261012/README.md)。恢复残差0.1、同链半径32/全局交替、每链独立RoPE；固定链槽位187/154，各含一个可学习EOS，dot/protd固定。原270M object302作业保持独立，本次未启动新训练。
12项CPU测试通过，完整75,601,176参数模型前后向loss及梯度有限；旧object collator误用被拒绝。全量扫描发现原167/135含EOS预算溢出5条，按全量最大值扩容至187/154，无删样本。验证报告位于 /project_c20250601/251105016/project/dllm_test/output/v5_fixed_chain_f011_20261012。

全量新布局审核已通过：85分片、7,729,193条，零溢出/零排除，最大输入1136＜2048；170条实际collator输出与长度公式一致。分片SHA256与完整报告见 [full_audit.json](/project_c20250601/251105016/project/dllm_test/output/v5_fixed_chain_f011_20261012/full_audit.json)。12项CPU测试及整模前后向检查通过，未保存测试权重。正式启动绑定与多卡入口验收尚未执行，不能沿用object302的旧启动票据。


2026-10-12 更正：上一轮将“回归f011”误解为恢复半径32，超出了用户要求。新75.6M已修正为偶数层链内全注意力、奇数层全局，保留残差0.1和每链RoPE；variant明确标为fixed_chain_full_global，避免误称原版f011。仅修改新fixed_chain实现及配置，旧f011和运行中模型不变；数据布局与全量审核不受影响。


## 2026-10-12T18:17:24.162084+00:00 训练至占卡只读监督结束

监督结果=training_exited_standby_observed；训练票据状态=failed；八卡占卡恢复观察=True。证据：/project_c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。


2026-10-12 当前启动：75,601,176参数、V5数据、残差0.1、独立双链RoPE、偶数层链内全注意力/奇数层全局。固定槽位187/154，EOS分别学习，dot/protd固定。训练预算300000步/global512（8×micro64×GA1），LR1e-4/warmup2000/cosine，seed42，从头初始化。新启动包 [launch_fixed_chain_75m_20261012](/project_c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_fixed_chain_75m_20261012/README.md)。13项CPU检查通过，8/8整模显存检查和NCCL通过；当前进入85分片全哈希门禁，正式步数待日志确认。原270M已按用户要求停止于约72279步，最新完整checkpoint72000保留。
新验证入口对全synthetic链不计算未知长度；固定dot不计split loss。冻结入口与旧训练分开，不改既有运行源码或数据文件。


2026-10-12 新版本已正式运行：75,601,176参数，V5训练集7,626,885条；残差0.1、双链独立RoPE、链内全注意力/全局交替，固定槽位187/154。300000步/global512=8×micro64×GA1，LR1e-4/warmup2000/cosine，随机从头初始化。2026-10-12T18:23:05.641727+00:00核实已到65步，最近记录训练loss=2.9047，梯度有限；13项CPU测试、8/8整模GPU检查、8/8 NCCL、85分片全哈希门禁通过。每1000步验证/保存，首次验证尚未到达。证据：[启动验收](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/startup_acceptance.json)。旧270M已按用户要求停止（最后约72279步，完整保存72000保留），原V7 f011此前已完成。


2026-10-13 2026-10-13T04:32:53.707950+00:00核实：新75.6M固定双链版本继续训练，33281/300000步（约2.23 epoch）。已完成33次验证；最佳checkpoint-29000 eval_loss=1.8604366，最新完整checkpoint-33000 eval_loss=1.8642278；最近训练loss=0.3918。固定验证子集为七来源各256条，共1792条；主指标为全mask/source-macro。最佳AA准确率38.31%，EOS单点80.93%，首EOS位置完全正确10.41%，长度MAE4.39；后两项为全mask单次预测，不能当多轮生成成绩。新版本尚无已回填的下游benchmark成绩。最近10000步含验证/保存平均1.101秒/步，按此速度剩余约81.6小时（未计后续中断）。证据：[本次核实](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_300k_b512_supervision/training_review_20261013.json)。


## 2026-10-13T08:06:40.767574+00:00 CHIMERA任务定义修正

用户要求保持baseline任务，已完成官方H3适配、全量测试ID/条件隔离/计分核验及入口切换。合成模型用于接口验收，未产生真实checkpoint分数；历史整链产物保留。协议细节与证据仅维护在[CHIMERA现行协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。pllm缺少既有binding测试的sklearn，相关联合回归改用已有protenix_abtcr环境后通过，未安装或修改环境。

## 2026-10-14T11:10:05.165048+00:00 固定双链架构与后训练探索启动

2026-10-14T11:10:05.165048+00:00 架构/后训练探索进行中：原预训练已主动停止，131000完整恢复点和112000共同初始化已独立保留。首轮12候选当前{'success': 7, 'failed': 1, 'running': 4}；repair仅门禁失败，修正后24项CPU测试通过、独立r4已排队。真实八卡容器仍Running。单步EOS与16步生成排名不同；补齐链EOS为模板终点，尚不能据此确定最终方案。[状态](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/current.json)、[实验方案](/project_c20250601/251105016/project/dllm_test/docs/FIXED_CHAIN_EXPLORATION_20261014.md)。

启动包缺失examples依赖已修正；repair的no_grad/autocast缓存与checkpoint重计算冲突已在CPU复现并修正，GPU重验排队。健康候选及原下游任务不重启，历史结果不覆盖；当前属于探索，最终选型与部署未完成。

## 2026-10-14T11:39:46.624878+00:00 后训练探索检查

2026-10-14T11:39:46.624878+00:00 已核实首轮12组完成1000步（repair采用r4修正版），12组导出重载一致性通过；第二轮r5仍在完成。训练集2909对弱标签偏好输入一致性全部通过；四项偏好/SFT对照及三项基线的r6冻结准备中，52项CPU测试通过，尚未入队。确认集未打开，未最终选型。

## 2026-10-14T12:01:37.901295+00:00 三轮筛选完成与长程验证

2026-10-14T12:01:37.901295+00:00 三轮短程实验共27次有效运行、23种方案已完成；r7十五组长程验证已实际启动，八模型表征诊断已排队。短程弱偏好β1有效链52/61，β10未改善重建；验证偏好21对仅4条件组，不据此选型。后训练下游CPU接口已通过，确认集仍未打开，最终训练尚未部署。 [状态](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/current.json)，[短程指标](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/screening_complete_so_far.json)。


2026-10-14T12:34:47.491657+00:00 全量后训练准备完成：索引/loader、任务混合和LoRA原生断点恢复测试通过；确认后36任务下游入口完成，r7 LoRA真实权重CPU桥接通过。最终选择与部署仍待独立确认。[当前实验](/project_c20250601/251105016/project/dllm_test/docs/FIXED_CHAIN_EXPLORATION_20261014.md)。


2026-10-14T12:43:50.280649+00:00 三种子长程15/15与冻结表征8/8完成；16模型独立确认已入队，协议未按screen结果修改。[候选冻结](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/candidate_freeze_r1.json)，[执行](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_confirmation_r1.json)。


2026-10-14T12:50:48.718337+00:00 独立确认选定LoRA；正式全量后训练首50步入队，真实八卡恢复验证待完成。[决策](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/confirmation_r1/selection.json)，[执行](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/enqueue_final_r1_stage1.json)。


2026-10-14T13:02:15.800552+00:00 最终LoRA方案部署验收完成；八卡原生50→100恢复通过，主段已超过100步持续训练。36项下游提交完成、等待结果。[验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_exploration_20261014/deployment_complete.json)。


## 2026-10-15T03:32:47.066512+00:00 LoRA覆盖完整epoch续训

原10000步已完成且退出0；用户要求跑完整epoch，目标累计33981步。CPU连续/中断恢复及数据游标验证通过；现有八卡容器首段10000→10050已入队，待实际验收后接主段。证据：[准备](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/prepared.json)、[入队](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/enqueue_stage1.json)。

2026-10-15T03:39:56.498595+00:00 续训实际部署验收：10000→10050八rank恢复均匹配；112个LoRA张量更新、78个基座张量逐值不变，完整checkpoint保留。主段实际已到10117/33981，继续运行；原10000步产物未修改。[验收](/project_c20250601/251105016/project/dllm_test/output/fixed_chain_lora_epoch1_20261015/startup_acceptance.json)。

## 2026-10-15 CDR 免训练评测启动

隔离实现多轮对照，进入真实 checkpoint 门禁。状态以[执行记录](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)与[cdr_inference_rounds_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015)为准；尚无新成绩。

2026-10-15：CDR免训练R1已完成，第二轮确认已提交；[提交记录](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)、[指标](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)。

## 2026-10-15 CDR免训练三轮评测完成

三个Volc作业均Success；255,771条预测完成独立核对，冻结候选跨权重确认结束，未找到稳定优于单步的H3方案，正式默认保持。[三轮诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-inference-rounds-results-20261015)；[完成证据](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_inference_rounds_20261015/completion_audit.json)。

## 2026-10-15 排除旧策略后继续探索

新概率聚合方向完成CPU门禁并提交GPU探索。保留确认集分数，尚未证明改善；[执行记录](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)、[cdr_marginal_exploration_20261015](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/cdr_marginal_exploration_20261015)。

2026-10-15 新概率聚合探索完成并独立验收，未得稳定改善；两个既有checkpoint概率融合探索接续提交，确认分数未开启。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。

2026-10-15 四组新方向探索及独立输出验收完成，冻结软embedding候选进入保留组确认及跨checkpoint检查；未宣称稳定改善。[结果与证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。

2026-10-15 新推理探索与确认验收完成，五作业Success；发现受限微弱信号，尚非实用稳定改善，正式默认不变。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-marginal-results-20261015)。

## 2026-10-15 分组软上下文启动

环境恢复后继续免训练探索；修正toy测试，完成数据来源与分组独立审计，冻结开发选择和确认流程。 [协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-context-20261015)。

2026-10-15 分组soft开发作业Success，全量独立审计通过并冻结选择；双checkpoint新OAS确认已提交。[结果与冻结选择](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-results-20261015)。

2026-10-15 分组soft本轮收尾：两作业Success，数据/门禁/全量输出/冻结选择验收通过。跨checkpoint收益不一致，未认定修复。 [权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-results-20261015)。

## 2026-10-16 八卡训练终态与验证效果核对

06:33:56Z平台与产物核对：OAS-only queue012作业`t-20261015124455-xrns4`已Failed，最后更新11845步（约2.44个自然epoch）、保存11000；直接退出错误为NCCL BROADCAST等待1800秒超时，底层根因未确定。固定256条OAS验证，1000→6000→11000步loss为1.686219→1.620522→1.626349，AA准确率42.35%→45.07%→45.14%，首EOS正确7.62%→13.48%→13.48%，长度MAE2.97→6.19→6.19 aa；结束位置并未随loss全面改善。

LoRA续训在2026-10-15T10:49:07Z成功退出，八rank success、COMPLETE与票据exit0一致；33981步为1个来源均衡采样epoch。固定7来源各128条验证，10000→22000→33981步source-macro loss为1.809004→1.807105→1.807499，首EOS正确26.51%→26.77%→26.77%，长度MAE3.13→3.24→3.25 aa，续训增益有限。保存20000/21000/22000及33981；原八卡资源容器`t-20261007153431-vhx2f`仍Running、supervisor为standby。单步全mask指标不等于多轮生成或binding效果，两条训练验证数据不同不能直接排名；现有下游汇总的LoRA seed42/3000是另一权重。本次未启动、重启或提交任务。[完整状态和指标快照](/project_c20250601/251105016/project/dllm_test/output/immune_v5_fixed_chain_75m_oas_300k_b512_20261015_supervision/review_20261016/training_status_and_metrics.json)。

## 2026-10-16 组件消融启动

用户授权后开始免训练机制消融，保持先验模型和旧实验产物，新增独立数据划分、真实上下文审计及多重比较校正。 [协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-ablation-20261016)。

2026-10-16 组件消融收尾：两作业Success，冻结协议/数据/源码、真实门禁和全量产物验收通过；未选择winner。soft组件获支持，完整方法优于原多步但未超过单步，保留负结果。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-ablation-results-20261016)。

## 2026-10-16 旧checkpoint清理完成

按用户授权，仅清理历史pre-V7系列中无测试/结果依据的权重；最近native/fixed-chain/OAS/LoRA及V7保留。核对评测产物/配置/脚本/对照文档、软链接及活跃任务后，冻结逐文件清单并执行：早期270M/8B 6个、V3 32个、V5 23个、V6 8个，共69个checkpoint、252份权重及训练状态文件。逻辑大小326.29 GiB，按最后硬链接删除计释放255.07 GiB文件块；共享存储实际回收可能异步。保留37个旧系列checkpoint及全部独立评测快照，配置、日志、指标、来源不删除。独立复核1844个原有保留文件（含172个权重），两份结果表SHA256保持一致；每个清理目录有WEIGHTS_REMOVED标记。本次无训练/评测提交、取消或重跑。[精确路径、保留依据、删除日志与复核](/project_c20250601/251105016/project/dllm_test/output/checkpoint_cleanup_20261016/README.md)。

2026-10-16 用户授权验证新方法8次前向，冻结1+3与1+7两种分组预算，其余配置保持，重用旧评分集合而不消耗reserve。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-block-budget-20261016)。

2026-10-16 4/8前向两作业推理完成后审计因外部清理父预测文件失败；已记录Failed、未重提GPU。新增分数查看前提出缩减审计范围，尚未执行即发现清理侧已独立提交原SHA精确恢复；等待恢复后执行原冻结审计，全部比较保持。[恢复记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-budget-recovery-20261016)。

2026-10-16 4/8前向收尾：父预测精确原SHA恢复，原CPU审计全部通过，原统计及对照完整保留；GPU任务Failed与CPU恢复验收分别记录。新增8前向无明确H3收益，不改变正式默认。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-block-budget-results-20261016)。

## 2026-10-16 CDR从左到右解码准备

同样本/两checkpoint，首forward复用、每位置hard argmax、保留旧词表；先冻结并验证再提交。[协议](/project_c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md#cdr-left-to-right-20261016)。

## 2026-10-16 CDR左到右验证完成

两checkpoint作业Success，完整预测/轨迹/参考来源/旧控制/成本审计通过。H3均低于单步，正式默认保持；复用已评分样本、无新reserve消耗。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#cdr-left-to-right-results-20261016)。


## 2026-10-16 用户更正 BERT 数据与步数

V5 `bertv5mlm1016` 是已中止的启动尝试，不作为新实验继续训练。10:50 UTC 写入本人 PAUSE_TRAINING 后，supervisor 已回收训练进程组并恢复八个 standby；平台容器未取消。新目标为最新清理并补全 MHC 的 V7 / 300000 optimizer steps；后续检查与启动记录将写入 `/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016`。


### 2026-10-16 V7启动检查与开发集隔离更正

首次V7 smoke已通过全量数据门禁、新账号在线连接、保存状态和扰动精确恢复；下一步loss一致，参数最大差7.45e-9，梯度范数相对差1.2747e-6，超过原1e-6阈值后主动失败。修订检查将归约范数容差明示为1e-5，并新增与微小warmup学习率无关的Adam一阶／二阶矩更新相对L2比较（1e-5），保存状态仍必须精确。11项CPU测试通过。

额外核查发现旧V5外部开发探针虽然与V7 train无精确受体重叠，但会命中新版测试保护规则；不得沿用“无benchmark test”断言。正式训练尚未启动，首次smoke也在任何开发评估前失败。按V7冻结matching、candidate coverage及upstream key证据重筛；不在匹配候选覆盖中的负样本保守排除，不将查无记录当作无重叠。原数据索引保留，新建index_test_clean.json。准备过的smoke_r2未入队；后续使用独立smoke_r3票据。


V7 smoke_r3的加强检查测得BF16 efficient更新矩相对L2误差：exp_avg=1.0684e-4、exp_avg_sq=4.7368e-5；参数最大差仍7.45e-9，保存状态／输入和loss一致，任务按阈值失败。r4明确将高精度恢复核对与生产后端数值检查同时保留：FP32/math独立重放要求moment误差≤1e-5；BF16 efficient重放≤1e-3（0.1%），另保持参数、loss、LR和梯度范数检查。不会只因小warmup LR导致参数变化小就验收。正式训练仍用profile通过的BF16 efficient；历史失败结果不覆盖。PyTorch官方说明不同SDPA实现可能有数值差异，math对half/BF16使用float中间值：[PyTorch2.8 SDPA](https://docs.pytorch.org/docs/2.8/generated/torch.nn.functional.scaled_dot_product_attention.html)、[数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)。独立math复算结果尚待r4，当前不将失败归因视作已证明。


### 2026-10-16T11:11:55.347946+00:00 V7正式启动

r4实际exit0；8rank FP32/math恢复重放的参数和Adam矩误差均为0，BF16 efficient误差在明示界内，484条真实尾批与固定MLM/清理后两来源开发探针全部通过。用户授权的V7 / 300000步正式票据已提交现有八卡容器；数据、warmup15000、配置、同版评估与新W&B run均已绑定。[正式提交回执](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016/preparation/formal_enqueue.json)。等待首步及线上历史核实，不将入队等同于已经训练。


2026-10-16T11:14:43.384403+00:00 正式启动核实完成：实际训练125/300000、8rank/global512，累计64000条；checkpoint50全文件哈希、八rank RNG和参数一致性检查通过。W&B服务器状态running，已读回训练step1/2/3/25；新账号新run可持续记录。[启动核实](/project_c20250601/251105016/project/dllm_test/output/bert_mlm_v7_300k_20261016/preparation/startup_verification.json)。


### 2026-10-17 LoRA 第二层跨任务表征

按用户指定固定block2，已实现早停读出并通过完整前向精确等价检查；[执行与来源](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#lora-layer2-representation-20261017)。独立输出保留原末层对照，当前阶段为GPU评测提交，分数待验收。


### 2026-10-17 第二层评测切换本地

用户明确改为本地测试；原7个作业全部Killed，已启动本地A100单卡顺序执行器，保留同一模型和任务计划。[执行详情](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#lora-layer2-representation-20261017)。


2026-10-17 第二层本地评测9/13项完成，均值/划分检查通过，结果显示明显任务依赖性；[阶段结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#lora-layer2-partial-results-20261017)。T1大数据、UpTCR及CoV-UniBind仍待完整结果。


### 2026-10-18 第二层本地评测全部收数

13入口成功、7组exit0，新增预测/聚合及划分核查通过；SPR探针按原协议跳过，不记为有效第二层比较。结果显示任务依赖性，保留原默认。[全部结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#lora-layer2-complete-results-20261018)。


### 2026-10-18 fixed50000第二层测试

用户授权本地第二层对照，独立绑定该checkpoint与原末层基线；旧block返回tuple已据冻结源码适配，初次预检记录保留。12项有效表征任务，SPR既有探针跳过不重复计入。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#fixed50000-layer2-20261018)。


### 2026-10-18 GDPa1六组池化诊断

用户建议的分链及非MASK读出完成，global分数/alpha复现与30组OOF通过；数值控制不改变主要判断，默认未改。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#fixed50000-gdpa1-pooling-20261018)。


## 2026-10-18 层间机制与末层改进

完成逐层／归一化及attention拓扑×MASK-key实测、子模块追踪和文献核查，目标active；下一阶段固定教师辅助训练。[研究记录](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)、[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#fixed50000-depth-diagnosis-20261018)。


## 2026-10-18 末层保持实际训练

完成四种300步后训练、真实派生权重重新加载与20组GDP预测复算；已完成T3-broad，其他保持性任务继续。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-preservation-pilot-20261018)、[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-preservation-pilot-20261018)。


## 2026-10-18 有界谱加权与Kong核验修复

完成teacher谱消融，启动同起点700步追加两组；Kong原采样已完成，核验依赖不存在的旧缓存导致exit1，改为原始测试数据核对后CPU-only验收成功，旧源与失败保留。[记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-spectral-pilot-20261018)。


2026-10-18T03:00:38.310252+00:00：末层普通／谱加权等预算续训完成并复核，弱方向和HIC部分恢复但存在属性权衡；继续MASK-key交互与跨任务验证，未完成目标。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-spectral-results-20261018)、[研究解释](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)。


2026-10-18T03:02:47.624857+00:00：MASK-key与末层训练交互验证完成，HIC已接近第二层但其他性质仍未达到；关系300的m396保持性完成。尚缺谱加权组合的跨任务确认和整体性质恢复，不能结束目标。[唯一数值页](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-mask-interaction-20261018)。下一步需针对Titer等尚未恢复的方向检查教师目标与实际末层约束，而非继续只优化HIC或只看CKA。


2026-10-18T03:04:11.935021+00:00：关系300保持性全部完成并复核；Specificity、m396和T3-broad改善，Kong H3略降。所有本轮自有进程已终态。谱加权＋no-mask候选尚需跨任务和固定方案确认，Titer等属性差距尚在；目标active。[结果与模型归属](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-mask-interaction-20261018)。


2026-10-18T03:11:48.865836+00:00 梯度审计完成，按证据启动同起点optimizer恢复的attention对齐实验和OAS-only线性诊断。[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-attention-alignment-20261018)。


2026-10-18T03:16:11.322722+00:00：OAS-only线性恢复没有全面恢复GDP属性；同拟合映射在GDP的误差更大，而教师距离分布仍重叠，需区分方向误差与粗粒度分布偏离。条件OOF配对区间不支持把近似均值当成非劣证据；Titer五fold均下降。继续已启动的attention训练对照，不将诊断投影作为目标完成。[权威证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-linear-recoverability-20261018)。


2026-10-18T03:20:37.465233+00:00：attention训练／推理对齐对Titer有同预算收益，但没有全面达到第二层；单纯延长当前池化目标也不是充分解。下一阶段应针对保留信息的粒度和泛化检查，例如分链／token级教师约束与更多预训练样本的受控比较，而不是继续仅延长同一个全局均值损失。本轮所有自有任务已结束，目标保持active；尚缺实际末层整体恢复与固定候选独立确认。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-attention-alignment-results-20261018)。


2026-10-18T03:27:54.818307+00:00 分链保持对照启动，训练数据校准和必要RMSNorm界检查完成，尚无最终评分。[方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-chain-preservation-20261018)。


2026-10-18T03:39:07.622508+00:00：冻结教师目标的九组诊断完成；输出特征白化伤害部分属性，不能与误差谱加权混为一谈。保留自然方向尺度的混合教师仍有属性权衡，没有全面优于第二层；不将教师投影结果计为实际末层改进。[唯一数值页](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#multilayer-teacher-target-diagnostics-20261018)。


2026-10-18T03:43:02.038536+00:00：分链约束降低了目标重建误差，却未改善整体GDP；不再沿此单一方向增加损失。启动严格嵌套4096→8192 OAS覆盖量对照，保留原全局损失及校准，检验此前OAS到GDP恢复差距是否受样本覆盖影响，尚无新分数。[完成证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-chain-preservation-results-20261018)、[新固定方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-diversity-pilot-20261018)。


2026-10-18T03:54:53.883492+00:00：8192训练及后训练归一化／attention审计完成，60项新增OOF核查通过。[末层最新实验结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-latest-audits-20261018)。


2026-10-18T03:56:36.970382+00:00：新增谱加权训练CE反向消融，直接检验生成约束与表征恢复的取舍。[CE消融固定方案](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-ce-ablation-20261018)。


2026-10-18T14:40:49.748853+00:00：CE消融观察恢复：原PID已不存在，状态文件running已过期，终态原因未知，无最终评分；当前可执行普通命令。[恢复检查](/project_c20250601/251105016/project/dllm_test/output/diagnostics/final_layer_ce_ablation_20261018/environment_recovery_observation.json)。


2026-10-18T14:44:09.795452+00:00：原CE进程已确认终止且无中途checkpoint，新目录重放同一固定对照；补做折内性质预测方向误差诊断。[CE消融恢复及方向审计](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-ce-ablation-recovery-20261018)。


2026-10-18T14:47:52.675674+00:00：CE恢复实验终态完成，重载／10项OOF通过；学习率单因素实验启动。[CE消融实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-ce-ablation-results-20261018)、[学习率对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-optimization-pilot-20261018)。


2026-10-18T14:55:49.307895+00:00：CE0和十倍LR两组训练、性质方向审计、遮蔽输入诊断全部终态，实际重载和保存OOF核查通过。[学习率对照实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-optimization-results-20261018)。

2026-10-18T15:03:28.749188+00:00：启动token方向约束的32位置／全残基覆盖对照，检查池化监督下的token信息保持是否改善未恢复性质；并补查高LR候选的Specificity。结论以固定终点下游实测为准。[token方向对齐协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-token-direction-20261018)。

2026-10-18T15:06:54.737443+00:00：逐token约束的稀疏组仍未恢复Titer；增加固定预算谱权重下限对照，检验弱方向损失分配，保留原Gram目标以隔离干预。[谱权重下限对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-spectral-floor-20261018)。

2026-10-18T15:13:45.829044+00:00：token方向32位置与全残基两组训练终态，session46035 exit0，20项OOF／重载通过；整体恢复未达成。[token方向对齐实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-token-direction-results-20261018)。

2026-10-18T15:16:06.343634+00:00：本轮三组训练、30项OOF／重载及分带几何审计终态通过；Specificity仍进行。更强谱加权的弱方向改善伴随主要方向损失，整体恢复未达成。[谱下限实测与分方向核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-spectral-floor-results-20261018)。

2026-10-18T15:19:31.952372+00:00：已有高LR1700终点screen误差仍下降，启动同损失／学习率／数据的固定5000预算对照，恢复权重和优化器，不按中途GDP选模型；Specificity旧进程继续。[固定5000步收敛对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-convergence-20261018)。

2026-10-18T15:24:14.085161+00:00：研究报告当前结论已合并CE直接消融、方向恢复取舍与尚未充分收敛的边界；续训起点复现及首个恢复包读取通过。[当前结论](/project_c20250601/251105016/project/dllm_test/docs/LAYER_REPRESENTATION_DIAGNOSIS_20261018.md)、[执行核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-convergence-20261018)。

2026-10-18T15:31:24.191026+00:00：高LR Gram1700的Specificity完整完成，五折保存预测及对照协议核查通过；5000预算实验仍进行。[高LR末层Specificity实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-optimization-specificity-20261018)。

2026-10-18T15:34:39.490181+00:00：5000续训继续原进程；新增无标签的train／screen／GDP方向误差核查，先建立1700基线，待5000固定终点后比较，区分拟合与泛化。[泛化分方向核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-generalization-20261018)。

2026-10-18T15:37:29.563164+00:00：1700 train／screen／GDP无标签方向审计完成；增加固定权重FP32推理对照，核查残余弱信号误差是否受BF16推理舍入影响。5000续训沿原进程推进。[FP32推理对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-inference-precision-20261018)。

2026-10-18T15:41:06.365537+00:00：FP32推理未实质补回dense-token1700的性质差距，不把BF16推理舍入当作主要解释；5000固定终点结果仍待完成。[末层推理精度实测](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-inference-precision-results-20261018)。

2026-10-18T15:48:49.145256+00:00：CE0固定5000、两端泛化与FP32推理审计均终态通过；Titer部分改善但其他性质及去噪有代价。启动同起点／同预算CE1反向对照，首批匹配通过。[5000步实测与泛化核查](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-convergence-results-20261018)、[同预算CE1对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-convergence-20261018)。

2026-10-18T15:56:37.943675+00:00：1700／5000性质方向审计完成；Titer存在能较好迁移的teacher线性方向，AC退化也不能解释为所测方向误差增大。统一加密Ridge网格诊断已启动，原报告不改；CE1仍运行。[长训性质方向诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-long-training-directions-20261018)、[Ridge网格诊断协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-alpha-resolution-20261018)。

2026-10-18T16:01:04.361293+00:00：统一Ridge网格加密只解释部分分数变化，未消除主要差距。原第二层＋全global末层的互补教师目标诊断启动，仅评目标潜力；CE1原进程继续。[Ridge网格诊断结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-alpha-resolution-results-20261018)、[互补教师目标协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#global-teacher-targets-20261018)。

2026-10-18T16:04:28.838715+00:00：互补教师目标完成诊断，四个正相关性质改善而Tm2更差；这是目标设计证据，非真实学生成绩。当前先完成CE1配对实验；若仍不能整体恢复，可用同维数early512控制与混合教师进行实际蒸馏对照，不能直接将拼接目标升级为结果。[互补教师目标诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#global-teacher-target-results-20261018)。


2026-10-18T16:16:39.460186+00:00：CE1固定5000完成并通过配对／方向／泛化核查；去噪保持改善，末层整体目标尚未完成。新同维数教师蒸馏已本地运行。[CE1固定终点与对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-convergence-results-20261018)；[互补教师实际蒸馏协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-complementary-teacher-20261018)。


2026-10-18T16:22:11.386155+00:00：CE1/no-mask-keys的Specificity、m396、T3-broad及native Kong生成保持性已本地启动；三表征任务GDP读出预核查通过。互补教师蒸馏原session44393继续。[CE1跨任务保持性协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-cross-task-20261018)。


2026-10-18T16:26:54.676371+00:00：CE1/no-mask-keys T3-broad完成，七档点估计均略低于原native末层；启动两个缺失attention控制单元分离因素，其余既定任务继续。[CE1 T3-broad保持性结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-tcr-results-20261018)；[TCR attention配对控制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-joint-tcr-controls-20261018)。


2026-10-18T16:32:27.274268+00:00：TCR四格对照及两组距离层核查完成，固定attention后训练也均下降，不能归因于MASK排除单因素。CPU完成OTS4096／126无标签回放池，为后续保持约束做准备；原四个GPU任务继续。[TCR权重与attention完整对照](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-tcr-factorial-results-20261018)；[无标签TCR回放数据准备](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#tcr-preservation-replay-20261018)。


2026-10-18T16:41:34.841449+00:00：CE1真实Kong三个CDR均下降，原始预测和协议重算通过；TCR原末层教师缓存及漂移诊断完成。连续CDR单步诊断KL数值问题已定位并用独立v2修复重跑，其余任务继续。[CE1真实CDR保持性](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-cdr-results-20261018)；[TCR教师目标与漂移诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#tcr-preservation-targets-20261018)。


2026-10-18T16:54:48.651248+00:00：m39612设置保持／改善，early512固定终点完成但Titer未达第二层；连续CDR和置信度反事实诊断完成，单改排序不足以恢复OAS四步生成。Specificity和互补教师另一arm继续。[CE1 m396结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-m396-results-20261018)；[末层保持性与置信度诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-span-confidence-results-20261018)；[early512固定终点](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-early512-results-20261018)。

- 2026-10-18T17:02:18.805844+00:00：CE1 Specificity五折已核验；[数字与界限](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-joint-specificity-results-20261018)。启动[整段CDR保持缓存](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-preservation-targets-20261018)，互补教师组继续原进程。

- 2026-10-18T17:07:30.034772+00:00：CDR完整条件预测缓存完成且核验通过；互补教师原进程继续，终点审计依赖已启动。[执行与证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#cdr-preservation-targets-20261018)。

- 2026-10-18T17:11:57.906581+00:00：对已确认TCR/CDR下降的CE1候选，准备同起点、同原目标与前向、仅切换回放反传的保持性对照；先运行无更新preflight，不预判回放会恢复任务。[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-pilot-20261018)。

- 2026-10-18T17:14:32.175761+00:00：互补教师实际末层固定终点完成，补回Titer点估计而HIC/AC接近第二层；待确认保持性和稳定性，不能宣称目标完成。启动候选native跨任务核验与独立CE1回放对照。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-results-20261018)、[保持性执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-complementary-cross-task-20261018)。

- 2026-10-18T17:20:32.160531+00:00：新候选全层与冻结控制完成，后层性质确有改善，但Titer均值收益集中于一个fold；条件区间不能证明稳定达标。TCR同native七档点估计均下降，保持训练和其余任务继续。[逐层证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-depth-results-20261018)、[TCR保持性](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-tcr-results-20261018)。

- 2026-10-18T17:26:36.923921+00:00：词表头nullspace目标可行性与等距控制完成：提供将语义目标放入输出不敏感方向并保留原均值head分量的候选设计；只是目标级证据，非token或生成保持，未训练且不改现有live实验。[结果与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#head-nullspace-target-results-20261018)、[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#head-nullspace-target-protocol-20261018)。

- 2026-10-18T17:31:22.500601+00:00：互补候选完整Kong已核验，GDP改善伴随真实生成损失；继续原保持性对照，预备固定终点审计，不将新目标可行性或monitor当作恢复证据。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-cdr-results-20261018)。

- 2026-10-18T17:38:32.837671+00:00：互补候选m396评估及保存预测复核完成；保持性两组固定终点实际任务队列等待原审计。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-m396-results-20261018)、[执行](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-cross-task-20261018)。

- 2026-10-18T17:42:15.040036+00:00：Specificity预测核验绑定原进程完成后自动运行；保持性双组仍等待固定终点。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-cross-task-20261018)。

- 2026-10-18T17:46:41.120328+00:00：互补候选Specificity与四任务身份核验完成；整体保持目标未达到。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-complementary-specificity-results-20261018)。

- 2026-10-19T01:02:52.113456+00:00：保持性原进程已缺失，恢复5500模型+AdamW并核对固定schedule；对照复用，后续评测待成功审计。[执行与恢复证据](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-recovery-20261019)。

- 2026-10-19T01:07:07.354144+00:00：恢复V1严格复现检查失败；相同5500状态两次20步重复诊断完成，V2带显式数值健康检查续训。[证据与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-recovery-repeatability-20261019)。

- 2026-10-19T01:11:42.494423+00:00：十项任务完成后的同attention配对汇总已安排，尚待完整结果。[协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#final-layer-retention-suite-20261019)。

- 2026-10-19T01:15:39.243968+00:00：保持性配对fixed6000与40项OOF审计完成；真实任务开始，GDP取舍未解决。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#final-layer-retention-endpoint-results-20261019)。

- 2026-10-19T01:20:20.563515+00:00：保持性native TCR部分恢复但未全面追平原模型；GDP缓存头子空间诊断完成。[TCR](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-native-tcr-results-20261019)、[诊断](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-head-subspaces-20261019)。

- 2026-10-19T01:24:14.531527+00:00：保持性no-mask TCR配对完成，MSE监测改善未换来实际分类全面恢复；开始真实CDR。[结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-no-mask-tcr-results-20261019)。


2026-10-19T01:33:21.258505+00:00：逐token head/norm目标诊断完成；发现直接归一化损失Titer，继续固定序列均值的可行性检查。AB对照6000实际CDR完成且各区域均退化，保持组仍在运行。目标active。[结果与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#token-head-norm-targets-20261019)。


2026-10-19T01:35:01.243994+00:00：序列均值约束目标诊断完成并通过10项OOF，原head/norm与较好HIC/Titer目标可在clean AB输入上共存；未训练student、未整体达标。对照Specificity完成；保持组CDR/Specificity运行，m396排队。[结果与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#token-mean-head-norm-targets-20261019)。


2026-10-19T01:43:54.432221+00:00：旧保持性CDR两组全部完成，实际生成部分恢复但三个区域仍低于原模型。OAS可行目标构造及真实反向preflight通过，新原模型同起点两组1000步已启动；目标active。[CDR权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-paired-cdr-results-20261019)、[新训练协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#feasible-final-pilot-20261019)。


2026-10-19T01:48:24.720933+00:00：新配对实验补齐独立终点汇总与任务权重/attention核验，已启动等待进程2601657；主诊断报告的当前结论已合并更新，历史记录保留。[汇总协议](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#feasible-final-suite-20261019)。


2026-10-19T01:55:15.728843+00:00：新pool-only100实际参数/AdamW与冻结RMSNorm核验通过；留出集均值目标与token目标误差出现不同方向变化，仅作为中间监测，不改固定1000预算或选checkpoint。[权威记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#feasible-final-first-screen-20261019)。


2026-10-19T02:09:29.647350+00:00：6000保持性两组Specificity五折及保存预测全部核验，点估计接近且逐折方向混合；m396继续排队执行，新可行目标固定1000训练继续，目标未达。[权威结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#retention-paired-specificity-results-20261019)。


2026-10-19T02:22:59.351202+00:00：新pool-only固定1000及10项OOF/重载/冻结norm独立核查完成，GDP仍有HIC与Titer缺口；feasible-token组已按同一起点启动，首批共享前向指标完全一致。跨任务尚待两组完成，目标active。[完整结果](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#feasible-final-pool-endpoint-20261019)。


2026-10-19T02:26:15.829111+00:00：新两组同100步恢复包与留出监测配对核验完成，token约束确实降低其留出目标误差，但不替代最终GDP/保持性验证。实验组和旧m396继续原进程，目标active。[数字与限制](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#feasible-final-paired-screen100-20261019)。

2026-10-20T08:31:08.710592+00:00 diffusion/edit16500下游按用户授权切6组非闲时云端并行；完成项保留，本地m396已收尾。[执行记录](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step16500-20261020)。

2026-10-20T09:52:06.698971+00:00 diffusion/edit16500下游15项完成并验收，6组非闲时作业均成功退出。[执行与结果入口](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#diffusion-edit-step16500-20261020)。

2026-10-20T11:00:33.241582+00:00 按用户要求重新申请八卡非闲时常驻容器，已核实standby并发布diffusion/edit20500完整状态续训票据，目标200000。[恢复记录](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md)。

2026-10-20T11:04:02.313874+00:00 diffusion/edit20500在新八卡常驻容器中恢复验收通过，已推进20575并完整保存20525；原W&B续接。[完整验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/restart_20261020/startup_verification.json)。


## 2026-10-22 diffusion/edit 闲时续训准备

2026-10-22T07:25:48.353897+00:00 最新完整28000、最后日志28225；原容器本地已退出。用户授权闲时续训当前分支至原200000目标。16份恢复状态、874源码、1567依赖哈希通过；8项CPU断点选择／旧心跳／平台状态测试通过，YAML及shell语法通过。Volc缺AK/SK，未提交、未启动新GPU训练。[准备](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/preparation.json)、[过程](/project_c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)。


## 2026-10-22 diffusion/edit 闲时任务已提交

2026-10-22T08:04:21.223840+00:00 已提交闲时续训任务`t-20261022160418-jwww4`；1×ml.pni2.28xlarge（8×A10080）、Preemptible=true、c20250601。从完整28000恢复模型／AdamW／scheduler／8-rank RNG，目标累计200000；原冻结源码、数据、batch及W&B run沿用。平台状态正在回读，尚未宣称实际开始训练。每60秒只读监控已运行。[YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_spot_20261022.yml)、[回执](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/submission.json)、[实时监督](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/monitor_status.json)。


## 2026-10-22 闲时排队原因核查

2026-10-22T08:21:00.739831+00:00 闲时续训`t-20261022160418-jwww4`仍Queueing。API实查本队列c20250601 GPU配额0；同资源组可出借且Running的队列为xts001_temp（24配额／19已分配／SharedQuotaAllocated=5）和devinstance-queue3（32／32／0）。仅按配额减已分配计算的余量上限为5张，小于单节点8卡需求；尚未取得节点级物理空闲卡数。平台任务提示“当前队列中有配额，可能存在碎片问题”是调度诊断，不能替代上述配额统计。[查询证据](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/spot_20261022/queue_diagnostic.json)。


## 2026-10-22 非闲时 ZENITH 续训提交被拒绝

2026-10-22T08:29:50.240397+00:00 当前未运行：原闲时任务 `t-20261022160418-jwww4` 已取消，平台确认 `Killed`。按用户明确选择，非闲时提交到 `ZENITH_MLP2_TASK`（`q-20261009101456-4m9np`）被平台拒绝：当前Key缺少该队列的 `ml_platform:CreateCustomTask` 权限；提交后按唯一任务名查无任务，未获得新task_id。完整28000恢复配置已就绪，目标累计200000；旧日志28225不代表新训练。原只读监控已停止，新任务尚不存在。[提交失败证据](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/submission_blocked.json)、[非闲时YAML](/project_c20250601/251105016/project/dllm_test/train_jobs/bert_diffusion_edit_resume28000_nonspot_zenith_20261022.yml)。 原模型计算及全状态恢复方案保持；本轮仅资源类型和队列切换，无新训练步。


## 2026-10-22 续训网页入口挂载路径更正

2026-10-22T09:01:17.865507+00:00 已核实当前项目实际路径为 `/project_c20250601/251105016/project/dllm_test`，共享存储挂载根为 `/project_c20250601/251105016`。网页入口修正为 `bash /project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/entrypoint.sh`；脚本使用当前挂载并在必要时建立旧前缀兼容链接供冻结checkpoint与conda使用。恢复器对源码根路径先resolve，再检验加载模块归属，修复符号链接导致的错误拒绝；冻结训练计划、数据、优化器与原200000目标保持。新入口 --verify-only 通过16份checkpoint／874源码／1567依赖哈希检查；未启动训练、未新提交任务。[路径验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/web_submission/mount_fix.json)。


## 2026-10-22 当前挂载路径与历史日志索引更新

2026-10-22T09:08:28.512124+00:00 文件访问前缀统一为 `/project_c20250601/251105016`；当前恢复控制配置和文档链接同步。原日志、训练数据、权重、优化器与冻结配置保持；本次无新训练步。[代码与日志路径表](/project_c20250601/251105016/project/dllm_test/docs/BERT_TO_DIFFUSION_EDIT_20261019.md#当前存储代码与日志路径)、[迁移前清单](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/path_migration_20261022/before.json)。

本次路径更新验收通过：8项CPU测试；真实入口 --verify-only 核对16份checkpoint、874源码与1567依赖；启动边界CPU模拟确认仅规范化输出目录，训练计划其他字段保持。历史日志／冻结计划哈希检查通过；此处没有执行GPU训练。[验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/path_migration_20261022/verification.json)。


## 2026-10-22 网页续训数据门禁失败与更正

2026-10-22T09:44:07.786097+00:00 网页任务 `t-20261022172952-vshml`（`mr3w2d46j1idw66u`，ZENITH_MLP2_TASK，Preemptible=false）平台确认 `Failed`，结束于09:31:23 UTC；09:30:37的verified仅代表恢复包哈希通过。实际在数据门禁报 `Shard resolves outside the release and its immutable parent`：新挂载解析路径与旧V7不可变绑定的路径比较不一致，未进入模型恢复／首个训练更新。已加入独立哈希绑定的路径适配，保留允许目录、目标一致性和内容哈希检查；原84分片及索引、开发集、父BERT287000和完整checkpoint28000实际CPU检查通过，17项回归测试通过。原任务已退出，需在网页用同一入口重新提交。[修复验收](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/data_path_fix_20261022/verification.json)、[实际数据检查](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/data_preflight.json)、[失败日志](/project_c20250601/251105016/project/dllm_test/output/bert_diffusion_edit_v7_from287000_200k_20261020/nonspot_20261022/attempts/20261022T093034_13d9a827/torchrun.log)。 此前仅校验恢复文件和CPU模拟启动边界，未执行实际数据门禁，不能视为完整GPU启动验收。当前仍未恢复GPU训练，无新增训练步。

2026-10-22T19:18:42.204231+00:00 step43000单卡评测按用户要求在zhuyihen启动；临时路径兼容及执行记录见 [EVALUATION_PROCESS](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md)，没有修改固定权重或提交云端任务。

2026-10-23 step43000本地15项评测全部成功完成；终态证据见 [EVALUATION_PROCESS](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md)。

独立MLM入口验证：真实无缓存CPU预检通过，Pair/KD缺缓存返回not_ready（exit2），14项CPU训练/恢复/报告测试通过；[验证回执](/project_c20250601/251105016/project/dllm_test/output/stage1_ab500k_pilot_20261022/launch_mlm_independent/verification.json)。

2026-10-23 Pair/KD新入口真实CPU预检通过：cache_ready=true、student_admission=passed_cpu_only，84分片及500000/4096子集通过；GPU profile/8rank短测仍在用户提交后执行。

2026-10-24T07:21:29.282673+00:00 OAS-only r4已重新提交，task_id=t-20261024152107-fm8ws，Queue，resume-11000；修复r3在线W&B登录失败并采用离线模式，训练尚未开始。

2026-10-24T07:28:45.337086+00:00 OAS-only r5提交 t-20261024152826-krr7n，状态Staging；修复完整父数据绑定，85分片内容SHA及metadata预检通过。见 /project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024。

2026-10-24T07:31:09.825549+00:00 OAS-only r5 t-20261024152826-krr7n确认实际续训：11000→11166，0.338秒/步；当前完整checkpoint11000。见 /project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/training_status.json。

2026-10-25T11:43:58.565616+00:00 OAS-only r5 t-20261024152826-krr7n已Failed：训练到15979、完整保存15000；rank0日志追加命中磁盘配额，尚未重提。证据 /project_c20250601/251105016/project/dllm_test/output/oas_resume11000_r5_20261024/failure_20261025.log。

2026-10-25T11:46:52.260322+00:00 OAS-only r6 t-20261025194633-xrh4w已提交，状态Staging，完整状态从15000恢复；17文件SHA、85分片gate及写入检查通过。见 /project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025。

2026-10-25T11:48:55.297846+00:00 OAS-only r6 t-20261025194633-xrh4w确认从15000恢复到15049，约0.353秒/步，平台Running。证据 /project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025/training_status.json。

2026-10-25T13:29:47.387664+00:00 OAS-only r6完整32000：训练稳定，全mask验证loss约1.62、AA约45.4%，15000以来基本平台。见 /project_c20250601/251105016/project/dllm_test/output/oas_resume15000_r6_20261025/log_review_latest.json。

2026-10-26T23:52:54+00:00 Diffusion四卡用户任务t-20261027004403-495cm确认Running，修复后smoke及正式更新均成功，训练7425/200000步。300条已记录更新中无NaN/Inf；warmup后梯度均值0.2783、P95 0.3288、最大0.4111，峰值随后回落；日志每25步采样，不能代表逐步全覆盖。验证loss从1.908869降至7000步1.102351，最佳6500步1.101507；6000步LR guard首次减半。CDR验证仅128条内部已知长度抗体，masked target-only CE/accuracy暂未明显改善，H3准确率42.483%保持；没有把整体loss改善当作CDR生成收益。本次未改训练设置或submit/cancel。[冻结日志统计与曲线](/project_c20250601/251105016/project/dllm_test/output/diffusion_bert287000_cdr_4gpu_wandbfix_20261026/diagnostics/training_review_20261026/review.json)。

2026-10-27T03:35:02.511904+00:00 完成Diffusion/edit 76500 held20保存产物debug：全量token解析及ANARCI重提取复核通过；空评分串来源、EOS与覆盖口径更正统一见 [T4 §7.8](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#78-2026-10-27diffusionedit-76500-held20-空评分串定位)。新增可复用离线审计入口及JSON证据，结果表增加子集均值限制；未改冻结生成/历史指标，未训练或提交任务。


2026-10-27T04:24:11.595306+00:00 held20提前EOS根因追踪完成：原权重CPU逐轮、冻结训练index扫描、40项输入因子对照及20目标全MASK α槽位干预均已复核；结论与边界统一见 [T4 §7.9](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#79-2026-10-27提前eos的案例输入布局与提交机制诊断)。新增诊断脚本及证据回执；未改正式模型/采样器/评测入口或历史分数，无GPU训练/云任务。


2026-10-27T04:39:23.876548+00:00 核对76500 sparse13采样链路与缺失α占位的影响范围，原始记录与视图分母核验通过；结论统一见 [T4 §7.9](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#79-2026-10-27提前eos的案例输入布局与提交机制诊断)。结果表补正式引用限制，历史指标保持；未新生成或提交任务。


2026-10-27T04:44:41.520502+00:00 用户授权76500 held20/benchmark14在本地修正重跑；新增α全MASK上下文适配与仅目标β输出校验，4项CPU测试通过，CUDA门禁执行中；生成尚未启动。状态见 [T4 §7.10](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#710-2026-10-27用户授权本地α占位修正重跑)，[新输出](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_alpha_context_local_20261027)。未提交云任务，旧结果保留。

2026-10-27T05:13:20.085933+00:00 用户授权的76500本地α占位修正重跑完成：held20与benchmark14共3434条原始候选全部生成、独立重放/重评分通过，旧产物哈希不变；结果总表/简表四个视图已新增修正行，旧行保留并标明输入问题。held20覆盖19/20，仍有一目标CDR3格式全部失败；benchmark14/sparse13/common6分别14/14、13/13、6/6。详见 [T4 §7.10](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#710-2026-10-27用户授权本地α占位修正重跑)；[验收回执](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_alpha_context_local_20261027/verification.json)。本地GPU作业已正常结束，未提交云任务。

2026-10-27T05:21:51.297681+00:00 用户授权76500 αβ联合生成探索性对照，本地tmux启动；CPU五项和真实CUDA门禁通过，自动生成→独立复核→报告。任务协议与状态见 [T4 §7.11](/project_c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md#711-2026-10-27αβ联合生成对照)，[输出](/project_c20250601/251105016/project/dllm_test/output/downstream_generation/diffusion_edit_v7_76500_joint_ab_local_20261027)；旧结果保留，无云任务。

2026-10-27T05:26:50.702506+00:00 76500 pairing本地tmux探索启动；CPU/CUDA与评分器冒烟通过。协议和执行证据见[评测过程](/project_c20250601/251105016/project/dllm_test/downstream/benchmark/EVALUATION_PROCESS.md#pairing-76500-explore-20261027)。
