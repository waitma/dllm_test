# 预训练整合与实际 GPU 验收

## 2026-10-01T16:34:42.531238+00:00 恢复周期数据审核阻断确认

恢复周期连续3次核实同一数据审核阻断，独立复审未发现可独立推进的必要工程；原Goal已实际设为blocked，原身份、目标和累计用量保留，未完成。MHC无明确ready及双证书，expanded仍false且缺双证书，manifest SHA未变。原八卡0/0/42、8worker维护无job、监听PID2213164存活；真实小batch和最终训练未启动。本轮无模型/测试/shard读取、提交/取消/重启或新增checkpoint；不恢复注释、不改门禁、不重复验收。 [本周期第三次审计及实际状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_external_review_1790871604_3.json)、[Goal工具实际返回](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/resumed_blocked_goal_update_1790871604.json)。当前状态覆盖下方带日期历史。

## 2026-10-01T12:28:08.781013+00:00 原Goal blocked，最终训练未启动

原Goal实际设为blocked（外部数据审核）：连续3次同一阻断、独立复审确认无可独立推进的必要工程；目标和累计用量保留，未完成。MHC缺明确ready及双证书，expanded仍false/双证书缺，真实小batch/最终训练均未启动。原八卡0/0/42与监听保留；无新测试/训练/提交/取消/重启，checkpoint未新增。 [第三次审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/external_review_recheck_3.json)。有效审核与对应新版本绑定、发布及fresh-child/真实小batch验收仍待完成；CPU检查不是准入或效果证明。以下带日期内容为历史阶段。

## 2026-10-01T12:19:30.270649+00:00 四候选私有接入CPU完成、真实数据审核阻断

私有通用接入已补齐独立身份/来源/GPU证明/续训绑定，两环境各24/24新增检查通过；修正1项静态来源缺口。原R20核心、6live源码及4runtime pins保持。自有30控制文件/1118字节和10别名已删，checkpoint=0。未部署/准入执行/入队，模型效果未经真实验证。[本轮工程原件与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)、[实际CPU](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/validation.json)。

指定MHC无ready字段且双证书缺；审核范围与gate/input/selection/manifest须有新的有效共同绑定。当前直接相关私有整合收尾，真实小batch及最终训练推进依赖外部有效审核，不恢复停止的注释、不降低门禁、不做无关人工实验。原Goal与原八卡/监听保持，独立CUDA暂停。以下日期为阶段历史。

## 2026-10-01T11:47:52.931165+00:00 三方架构决策完成

三个用户授权subagent只读决策已完成：保留R20/A4主干，先单独比较R21/R24并以R26作成本对照；联合任务G02末端adapter为条件候选，G01降为读出/优化控制，硬瓶颈和EMA暂缓。8卡4方法×2seed及后续联合组设计已记录，未启动实验。原指定数据缺双证书/ready绑定，真实小batch未运行；原Goal与资源、门禁、注释停线和CUDA暂停保持。

[决策与具体测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)、[三方审查摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)、[禁用决策矩阵YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。新增发现：G01当前任务仿射可吸收到通用原型独立Linear头，不能预称主干任务特化；G04/G05按role而非任意段ID隔离且关系头存在池化旁路；正式R20输出untied但尚无G00式联合分类头。审核范围计划与旧执行scope还需一致的新版本绑定，本轮不改gate、不签证。以下带时间条目是阶段历史。

## 2026-10-01T11:29:06.399847+00:00 当前GPU工程准备与正式数据选择

通用GPU引擎CPU验收完成：两环境各44个唯一案例分三阶段覆盖，未完整重跑最终44项；修正4项工程边界，固定15文件源码包与8份禁用YAML核对通过。清理50个自有CPU状态文件，验证checkpoint=0。现有8卡执行端尚未接入G入口，新增GPU任务0。用户指定immune_v7_mhc_sequence_v1仅metadata核实，缺双证书且无training_ready=true，真实小batch未启动。

[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。原Goal保持active；保留原注释停线、真实数据审核与独立CUDA暂停条件。正式架构比较优先使用审核通过的指定训练数据小batch及实际模型输入/目标/评价；人工整数仅作工程控制，不替代正式效果。下一步完成原8卡执行接入、独立结果复核与自有续接状态清理工具，不重跑已通过检查。

## 2026-10-01T09:37:59+00:00 当前接续

R20/R21/R24/R26 synthetic Trainer GPU工程验收均已通过，不重复；私有训练接入支持准备已通过但未部署，真实数据仍training_ready=false且双证书缺失，真实语料未启动。G00–G07八个新通用整数原型另已实现并完成CPU功能验收，**GPU入口/显存测量/500步曲线仍待完成**；[新原型的实际证据及接续](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CV_JOINT_PROTOTYPE_20261001.md)。三步验功能与长期训练效果分开记录；以下日期为当时状态，当前状态以[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)为准。

## 2026-10-01T08:38:15.022797+00:00 私有训练接入支持更新完成

当前支持源码下的R20训练接入准备已完成；两个Python环境各9项新兼容/拒绝检查通过，原launcher全部函数及候选校验函数保持。14个自有临时文件已清理，checkpoint=0。[具体更新、测试和未完成项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_SUPPORT_REBASE_20261001.md)、[原始验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/validation.json)。当前仅私有准备，未部署；R21/R24/R26外部训练接入及有效数据审核仍待完成。真实训练尚未启动，新CV方法仍待实现。以下日期记录为当时状态。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC 当前实际v3状态

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

以下带时间记录属于当时状态。



## 2026-10-01 04:24 UTC 终态与下一步

04:12:41 UTC，新D01 `1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a` 已终态failed，runner rc2、child rc1。独立梯度核对耗时695.0528秒，低于1200秒预算；8个microstep/256条人工记录、78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194。随后报 `Oracle left GPU tensors resident before formal measurement`，错误先于正式三窗口；未生成raw/summary/通过标记、checkpoint=0。parent709.2715秒观察正常退出并完成后代清理；这与进程内GPU张量驻留检查是两个不同结果。[终态原件核验](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/terminal_review.json)、[实际失败日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer.log)、[阶段统计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer_stage_evidence.json)。

v2未记录清理后具体allocated字节，不能凭峰值判断驻留对象。新私有v3分支正在核查对象生命周期与AMP包装；真实tiny CPU弱引用检查已释放对象，尚未复现GPU驻留，不据此确定根因。尚未发布、构建或入队v3，不放宽显存/数值标准、不重跑独立CUDA算子。父快照、v1/v2不可变runtime和失败证据保留。

原八卡任务继续Running，最新独立监督为pending/running/results=0/0/38、8worker维护；监听PID2172045继续watching。T20 v2私有两环境各48项通过但保持未部署；同版本GPU验收及原数据准入未齐，真实四候选1000→累计5000和最终训练未启动。验证checkpoint不保存；小型失败日志、统计与哈希保留。

以下带时间记录属于当时状态。

## 当前 v2 状态

03:53根端复审发布9源码/测试，03:54创建526文件v2，manifest SHA3805679f1d7370042cd39741d4839ad1c6acc3805be899bed3b1f95872dace12；524父文件不变。两环境producer29项、打包/入口/运输89项、typed18项CPU和实际八profile/54模块导入检查通过。03:56本地监听重载PID2172045，原GPU任务/worker不重启。

04:00:41只入队新D01 `1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a`，GPU7实际领取；新预算1200/1800/600、总体3600秒，工作量和数值标准不变。尚无GPU通过。T20在新私有目录绑定v2，两环境各48项通过，未发布/pointer/入队；数据双证书仍缺。[v2源码、检查、启动与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。下方带日期的v1描述属于当时历史状态。

## v1 失败与必要修正

03:20:13 UTC 唯一 D01 已终态 failed，池 runner returncode=2，producer returncode=1；[原日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/producer.log)记录 `train_runtime=701.6219`，随后明确报错 `Gradient oracle exceeded fixed budget`。240秒阶段预算低于实际耗时；[父进程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/process_evidence.json)为716.1639秒、cleanup_completed=true，未达到1200秒总体期限。错误发生在正式三窗口调用前，不能称其已完成。没有raw/summary/PRODUCER_SUCCESS/GPU_RELEASE_COMPLETE/COMPLETE，也没有checkpoint；八个worker已恢复维护，池37份结果=36历史成功+1本次失败。

v1 runtime及失败证据保持不可变。v2已独立构建/入队；D01仍是v1运输合同，真实运行包以目录、manifest SHA/job_id区分。依据实测，事前预算1200/1800/600、总体3600秒，保持2048输入、micro32×GA8、三个正式窗口和全部数值/显存标准。新增小阶段记录保存实际耗时和统计，不能代替完整通过结果。

## 已部署的入口

保留原 LLaDA270 Transformer 主干、50-token 词表、scratch/no-ESMC 路线和现有 attention。R20–R27 八项人工 500 步结果已经归档；后续架构比较仍保留 R20 对照、R21、R24、R26，不能把当前仅覆盖 R20 的工程验收当作最终架构选择。用户指定不需要独立 CUDA 算子验证，该支线保持暂停。

producer 在私有工作副本完成两环境各 24 项 CPU 检查后，根线程核对原址 SHA 并发布。当前 producer SHA 为 `5843eb2a65e9637455c31731fa9709c4fdf0c62e84bba02cdc0df77ac041550b`；[实现与 CPU 范围](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_GPU_RELEASE_PRODUCER_IMPLEMENTATION.md)、[发布证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001/root_reviewed_merge.json)。原来源冲突证据保留，不推断未知写入者。

实际不可变 runtime 已创建于 [r20_r27_pilot_v1_20261001](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001/pilot_runtime_manifest.json)。manifest SHA 为 `baeac98b962650770cf17c2f83014654c2520622e2785ead17db680a1228421a`；526 个源码/配置文件中，524 个父文件全部保留原字节，新增 producer 与固定入口。两环境实际检查八份 profile 和 54 个项目模块导入通过；两处固定资源别名仅保证路径兼容，不认证数据。包 manifest 的 acceptance/training=false 是不可变源码包合同，实际 GPU 结果由另一个完成记录表示。

03:04:10 UTC 根线程实际应用经复审的 D01 路由补丁。活动 runner、registry、pool 的 SHA 分别为 `89e154f682e7ced2680f38bd0895eea2d18fd3447eb2e8aa0cd4184afc34fd30`、`2b0b4de27f6c1dac950af510c50eb49158ae577e31e0d323d2e1be9ff370bd84`、`f3522eca58f7fedf145e8d61986d5124e0cdd9e2718c6062c53c94e563ba832f`。[应用证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/application.json)绑定补丁、前后文件和固定 runtime；历史 28 项研究路由及父快照保持原样。

03:05:19 UTC 本地监听重载到 PID **2143672**，原 PID2033525 已退出。[重载记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/monitor_refresh.json)记录身份和新 heartbeat；原八卡容器与 GPU worker 未重启，没有新平台申请。

## v1 人工验收的启动记录

03:08:08 UTC 在既有任务 `t-20261001033327-fmnql` 的共享工作池入队唯一 D01：`1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156`，由 GPU5 领取。[启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_trainer_v1/d01_launch.json)与[运行目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790824088674261408_d01_pilot_runtime_gpu_r20_v1_9af03156/request.json)保留实际绑定。

运输参数 profile-synthetic/steps=1 表示一次验收调用。实际检查三个完整优化窗口，micro32×GA8、global256，共24个 microstep/768次人工记录呈现；另有独立一窗未裁剪梯度核对。检查真实 HF/Accelerate、BF16、AdamW、四档 mask、独立计数、RNG、尾批和显存；三次更新只验证实现，不能判断学习收益。

截至03:15 UTC，GPU5实际遥测为100%利用率、51401/81920 MiB，worker heartbeat 正常；目录只有 request/producer.log/hf_artificial，尚无终态、窗口计数或通过记录。日志中的旧 kernel 建议不是异常终态。以真实退出、清理、原始记录与强核验决定通过；整体期限1200秒，失败保留日志，不能凭 running 状态生成 GPU_RELEASE_COMPLETE。监督分支只读监控、记录，未修改或重试任务。[实时监督](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_GPU_SUPERVISION.md)。

## 正式训练仍需完成的条件

D01 不会启动真实数据训练，原研究 R20 的 mode=train 也不会自动切换到新入口。专属私有分支正在实现 `t20_r20_certified_pilot_v1` 桥接，调用固定 runtime 的 execute_admitted，保留数据 preflight 优先顺序，并单独绑定桥接源码来源。它尚未发布、部署或入队；第一版只覆盖 R20，其他候选须另有同版本验收。

候选数据 manifest SHA 仍为 `601db781ebf81ee857dde98ddcfb2d1da8e121a4c57ff2d80c864d420a4ebfce`，training_ready=false，audit_certificate.json/downstream_scope_certificate.json 缺失。固定1288条 selection SHA 为 `659ef88421dba606fb327e8221091b0feb5f939cb78c19a643da6aeac35b94f2`。等待团队已有审核或已通过相同准入的数据目录；没有重写门禁、签发证书或恢复此前停止的序列注释分支。软件验收不替代下游 test/few-shot/epitope/antigen 覆盖审核。

既定真实比较为四候选各1000步、同数据/主干/global256/目标/日程/固定验证，按曲线和成本选择后恢复完整状态到累计5000步，再敲定阶段架构并启动最终训练。当前既无正式语料训练，也无最终架构收益结论。

## 结果与文件保留

每次实际验收的结果、失败与修正同步到对应 MD。人工验收关闭 checkpoint 保存；已完成验证的临时 checkpoint 和无继续使用需求的测试输出按专属 ownership 删除，保留小型日志、指标、哈希与源码证据。正式训练正在使用的完整恢复状态和最终必要模型产物保留，后续验证结束再清理不再依赖的 checkpoint。
