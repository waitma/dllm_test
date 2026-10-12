# V7 两轮训练与生成评测计划

## 2026-10-07T00:39:13.894414+00:00 本轮两轮与三类计划生成评测完成

正式50082两轮及9CDR、轻链、TCR生成/评分全部有实际完成证据；TCR最终 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json，全部数字/可比性 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md。本阶段不自动开始第三轮或重做评测。原Goal未完成，原22格消融/最终图及真实终态核实继续；原停线和数据门禁保持。

## 2026-10-06T23:35:21.048373+00:00 两轮后CDR与轻链均完成

当前成绩唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md。轻链最终证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json；T4实际CPU输入通过、GPU和全量尚待接续，沿用既定协议，不据test改选模型。实时状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。

## 当前执行更新 2026-10-06T23:07:56.130738+00:00

累计两轮50082及九项CDR实际完成；best3＋latest完整恢复状态保留，未开始第三轮。轻链全量生成进行中、评分待候选完成；T4仅原生适配合成准备通过。原八卡继续22格消融中的8项，4项排队，完整图仍10/22。当前状态唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json；下方旧日期的waiting/未完成描述保留为历史，不得据其重启任何原作业或监督。

更新：2026-10-06T22:28:30.597931+00:00。正式50082/epoch2及九项CDR已真实完成、两个监督rc0退出。最终证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50082_root_review.json；评测 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json；唯一数字 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr。原八卡已转接12项未完成消融，不启动第三轮。

实际每轮为floor(6410749/256)=25041步，已由原生checkpoint9000的global_step/epoch=25041核实。两轮为50082步，纠正此前50084的预算翻倍估算。父阶段已实际完成25042步、进入第二轮一步；第二阶段已原生接续并通过25050验收，按已冻结实现跳过已消费的第二轮第一batch。50082为固定生成评测点，第三轮待本次评测后安排。

保留原25042步warmup1000＋cosine曲线，从零LR终点恢复模型/optimizer/scheduler/8rank RNG，显式第二阶段warmup1000到1e-4，再cosine降至总50082。模型仍f011/where-only/math，DDP8卡micro32/GA1/global256，原数据/目标保持。新日程必须独立版本和显式迁移，旧provenance不改写；best3主eval_loss＋latest全部完整状态保留。

CPU整epoch边界7→14测试通过后，追加实际偏移场景（7步/epoch，旧预算8，恢复到14）暴露skip_first_batches重建loader时iteration归零，第二轮采样错误。优化器/调度器/RNG原生恢复虽匹配，参数仍较连续训练差0.0007587373；失败及诊断保留。

修复 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_epoch_schedule.py：在单次Trainer.train作用域内保留skip loader原iteration/set_epoch，退出还原函数。实际重测通过（8089；976d3f rc0）：模型、optimizer、scheduler与连续14步完全一致，参数最大差0；第二轮顺序正确、已消费batch正确跳过、CPU RNG与原生恢复匹配。原25042步LR曲线全点和第二阶段端点验证通过，全部自有临时checkpoint已清理；无GPU/语料测试。测试入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_epoch_schedule.py；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/cpu_schedule_resume_offset1_fixed.log。

独立冻结driver/迁移/launcher/监督于2026-10-06T13:38:54.423749+00:00部署等待。父25042实际末点/退出与原数据门禁通过后，第二轮已单次入队并通过25050真实八rank恢复/保存/退出验收；当前目标50082。原数据审核/范围门禁不降，序列注释停线不恢复，独立CUDA暂停。

生成任务先核实V7 f011/A4与现有入口兼容、固定划分、输入防泄露和指标。9项CDR原生适配已准备；此前部署的等待监督已因上游故障退出，没有生成输出或新成绩；轻链配对/T4适配仍未完成，不把CDR覆盖称作其他任务已通过。任务协议唯一引用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md；附加AB-AG范围仍由 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/README.md §0定义。数字只发布到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md；过程记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROGRESS.md。不在test上选checkpoint/调参，不恢复被停注释分支。

机器请求与完整原件引用：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/request_v1.json。两轮训练、生成评测和原22格消融均未完成。


## 2026-10-06T13:38:54.423749+00:00 两轮接续已部署，等待现有阶段自然完成

原八卡实际12016步，当前段目标17016，abort=None；现有冻结训练上限25042。累计两轮目标50082（实际25041步/epoch）。独立接续源码615项和启动包已冻结；CPU跨轮原生恢复/采样顺序完全等价测试、元数据来源与5种不兼容恢复拒绝检查通过。新接续监督PID226559/start2455552329已由/proc核实，tmux v7_epoch2_supervision_20261006，当前仅等待父阶段，不抢占或重启原训练。

父阶段25042完整checkpoint、八rank成功退出、监督自然退出齐全后，才重新执行原只读数据门禁并封存真实迁移，先25042→25050八步实际接续验收，再每段最多5000步至50082。此时未产生25042/50082权重，未声称新阶段GPU恢复通过。遇失败保留证据并停止后续准入，不自动重放。总目标到达后保留50082完整latest并写评测就绪指针，生成适配仍准备中，不把就绪指针当生成结果，不自动开始第三轮。

启动包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/package_v1.json（SHA71a31cea950e975e58497eeaa137c4fbc256ca80d36abaa8837e72b274fc46ed）；元数据检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/package_root_review.json；实时 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/supervision_state.json；计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。数据审核、序列注释停线、独立CUDA暂停不变；未submit/cancel或新建GPU容器。原Goal保持active及累计用量，原22格消融仍未完成。


## 2026-10-06T13:54:38.042567+00:00 生成适配和当前衔接状态

原生CDR适配和固定两个数据集的执行计划已准备，CPU forward/防隐藏参考泄露/save-reload与v2隔离启动通过。正式协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4；执行计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_plan_v2.json，状态prepared_not_enqueued；实际源/评分器检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_v2_cold_review.json。v1启动错误证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_v1_startup_failure.json，v1不得执行。

目前8卡继续训练，two-epoch监督仍waiting_parent；到50082后会写EVALUATION_READY，该文件只表示训练节点就绪。原pool训练组入口不接受独立生成任务，后续仍需衔接实际评测执行，或使用已授权本地CPU方式；没有生成任务入队、没有生成成绩。轻链配对和T4原生适配仍未完成；不恢复注释分支。下一步是真实执行衔接和未来原生两轮完成验收，不重跑已经通过的CPU检查。


## 2026-10-06T14:15:40.813569+00:00 后续九项CDR评测已部署到等待监督

2026-10-06T14:15:40.813569+00:00：正式where-DDP当前14943、目标17016、最近段退出验收12016；14000常规保存/验证元数据已核对。父阶段25042由PID212174/start2455180199监督；累计两轮50082由PID226559/start2455552329等待接续。两轮后生成监督PID234066/start2455753425已部署，等待真实两轮完成；三者/proc身份均核实，当前无abort。

生成衔接已实现为9个CDR任务复用现有8卡，固定50082。只有训练自然终态、真实退出/清理、原数据与空闲检查通过后才发布；当前公开入口/别名注册仍保持原SHA，未入队或推理。执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_pool_v1/package_v1.json；状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_pool_v1/state.json。旧v2串行计划不是当前自动发布源；当前为generation_source_v3。完整状态见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

本地f010/42最新1750/3750，1750点已比较，下一2000；22格9完整产物、1运行、12未启动，末点图仍PARTIAL。后续缺失消融须等待两轮及本批生成任务完成后重新准入，不使用旧25042适配器直接执行。原Goal active，本回合progress；未submit/cancel/重启，原安全停线与数据门禁保持。

阶段执行与验证详情见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 本时间戳节点；评测协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4。此前“执行衔接待实现”属于历史状态，现已实现并部署等待，尚未触发实际发布。


## 2026-10-06T14:26:11.771262+00:00 最新运行核实

2026-10-06T14:26:11.771262+00:00：正式八卡where-DDP实测15800/50082步，当前段目标17016、最近段退出验收12016，abort=null。15000完整保存/验证元数据通过；两轮接续waiting_parent，生成监督waiting_two_epochs，尚未入队或产生生成结果。三个监督的PID/start/源码身份与8worker新鲜心跳已核实。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/generation_pool_v1/root_status_20261006T142611Z.json。检查点15000的只读元数据检查通过；后续仍按真实50082完成后执行，未提前用test挑选checkpoint或启动生成。本轮未更改训练/评测实现，未重新运行既有验收。


## 2026-10-06T14:49:24.133710+00:00 正式段接续与后续消融准备

最新17523/50082，17016段8rank恢复/保存/退出清理已核实，当前17016→22016。两轮接续和生成监督仍等待。后续消融的发布包已准备并通过6项检查，要求两轮及本批生成完成后才能执行；没有提前发布或争用GPU。当前真实状态与证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/root_review_v1.json。


## 2026-10-06T16:33:29.100254+00:00 第二轮25050真实恢复验收通过，已继续25050→30050

原八卡累计25465/50082，当前25050→30050段有新训练更新、abort=null；现有第二轮监督PID226559/start2455552329和生成监督PID234066/start2455753425身份、8worker新鲜心跳核实。父阶段25042已真实完成且原PID212174退出，勿重启；其原生epoch=1.0000399345074078，完整终态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment25042_root_review.json。

新25042→25050首段的8pool实际rc0、8份进程/进程组清理、24项optimizer/scheduler/各rank RNG原生恢复摘要均通过；旧绑定与新绑定通过MIGRATION_RESTORE和封存transition明确连接，615项冻结源哈希保持，模型仍f011/A4、micro32/GA1/world8/global256。25043–25050实际八个更新均有计时，保存参数与25042不同；17项原生文件及八rank摘要一致，实际epoch=1.0003594105666707、max_steps=50082。固定1330×4验证的112计数与70核心聚合核查通过；本首段未单独重算26项结束指标，不声称额外范围验证。

根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/initial25050_root_review.json（SHA69f1a7c2cbcea87eebeed2327982ed30dc5cd4d9a0e30bbc1b47f652e511c8cf，0f8f4e rc0）。首段eval_loss=1.8663481936605006；这8步用于实际接续验收，不据此评价方法收益或声称八卡连续轨迹位级等价。现保留best3＋latest并集：父输出24000/25000及新输出25050，共3份完整状态；父25042权重已正常轮替，原生归档与恢复证据保留。

原数据检查真实passed，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/parent_gate.stdout.json；封存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/seal_receipt.json，未生成审核证书或降低门禁。只读根核查最初12565d rc1，原因是将归档引用路径与原保存路径作整体比较；2ef2a0确认内容SHA一致，修正仅比较归档内容哈希后通过。未改模型/训练源码、未重跑模型或重放任务，初次失败原因保留在根复核JSON。

当前正式输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run，冻结源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/source_snapshot_v1（615项），状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/supervision_state.json；旧where目录是已完成的父阶段。原where专用核查脚本只接受旧目录/25042日程，后续新点须针对第二轮原生元数据核查，不能误用旧脚本或重跑已通过的25050验收。下一常规保存26000，累计两轮终点50082及之后9项生成仍待完成。

本地f010/42训练2425/3750、2250对照已核查，下一2500；22格9完整产物、1运行、12未启动，末点图仍PARTIAL。后续消融发布包继续等待真实两轮/生成终态。原Goal active，原身份/累计用量/完整范围保持；上轮与本轮progress。全部安全停线、数据门禁和独立CUDA暂停保持；无新资源提交、容器/worker重启或测试checkpoint。交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。


## 2026-10-06T20:59:34.081590+00:00 当前执行停止，原两轮后评测目标保持

当前故障/候选证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_STARTUP_LOCK_FAILURE_20261006.md；完整checkpoint45050保留，固定50082尚未产生。生成等待监督已因上游失败退出，不能继续称作自动等待发布。没有新评测运行、数字或协议变更；恢复必须保留原失败记录与所有门禁。

## 2026-10-06T21:13:43.071416+00:00 独立恢复入口/监督冻结完成，真实元数据预检通过，尚未启动

独立执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/package_v1.json（SHA4181c9e15cd210895a63420522b4fd0dd0165f42ab68d8032981573f861497f9）绑定46项源码/原失败证据；新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/source_v1/launch_v7_recovery_segment.py，新监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/source_v1/supervise_v7_recovery.py。只允许45050→45058→50058→50082，总新增5032步；先8步实际原生恢复验收，后5000＋24步。原失败50050段、两个已停监督、abort与STOP记录保留，不在原目录重放，不恢复被停止的序列注释分支。

新人工元数据检查5组/14项拒绝通过（841227 rc0）：不完整清理、未知退出、失败段已经训练、重复rank、非法预算边及未写完计时尾行均拒绝/隔离。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/cpu_metadata_controls_v1.json（SHA61d09ba8cb3239e250718977edc3a746e1d277e9bfd2a6e7d6ae3a2c9015024a）。真实只读预检973d6d rc0核实46项绑定、原615项冻结源码/迁移、8失败rank清理、旧监督缺席及45050原生JSON；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/root_preflight_v1.json（SHA0fe4deea89806819cc04f8698494008576f1e571290f95c20168b17fbbc29b30）。training_ready当前true、两证书存在且原SHA绑定通过；没有新证书或降低门禁。

此时没有新supervisor_launch、没有新stage/job入队、没有GPU恢复结果；正式仍45050后停止。模型f011/where/math、DDP8卡micro32/GA1/global256、50082日程和best3＋latest全状态保持。新监督首次真实成功后才接下一段，失败即停且不自动重放；生成衔接仍需适配独立恢复的终态，本轮未启动下游。原Goal active，上轮/本轮progress，完整22格/两轮/生成范围保持，独立CUDA专项暂停。


## 2026-10-06T21:15:37.822412+00:00 独立恢复首段已在原八卡单次入队，等待实际恢复验收

在既有“继续训练累计两轮”的授权下，独立版本一次启动（1c9897 rc0），没有重新申请/重启原GPU容器，也未重放原50050失败stage或重启旧监督。新监督PID319218/start2458293575已由/proc核实；tmux v7_epoch2_recovery_20261006，日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_epoch2_recovery_20261006/run.log。新段45050→45058的8个唯一票据已入队，原worker实际running；当前最新计时45050，phase=started、abort=None。这仅证明已入队与进程存活，8步原生恢复/保存/退出尚未验收。

新状态唯一源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/supervision_state.json；绑定执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/package_v1.json（SHA4181c9e15cd210895a63420522b4fd0dd0165f42ab68d8032981573f861497f9），首段票据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/launches_v1/segment_45058/enqueue_receipt.json。旧 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/supervision_state.json 继续保留真实aborted状态，不能覆盖或把它当当前新段。旧训练/生成监督仍缺席；下游9项CDR未启动，需适配新恢复终态。首段真实通过才继续50058、50082，不自动开始第三轮。模型/批量/日程及best3＋latest保持。

本地f010/42当前3525/3750、3500新对照已核查，下一3750。完整22格9产物/1运行/12未启动，固定图PARTIAL。原Goal active，本轮progress，完整目标与全部安全停线、原数据门禁、独立CUDA暂停保持。无volc提交或取消；新恢复版本的启动不作为原失败段成功证据。


## 2026-10-06T21:22:17.810232+00:00 新八卡恢复验收通过并继续训练，生成等待监督已接入新终态

45050→45058实际8更新、八个pool成功退出/child清理、24项optimizer/scheduler/各rank RNG恢复均已根核查（e43403 rc0）。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment45058_root_review.json（SHAa5cb727997d22ca0a807c0ccc8eec78439f7acfc29984bffaa75615132e2cbc6）；45058的17项原生文件、615源码绑定、参数摘要与固定验证112计数/70聚合/26结束指标均通过833ade rc0：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint45058_root_review.json（SHAf2ef99d3eccc2276169b1d3b5fa9056a9c689e8b38793a6762ba499fe279e337）。真实恢复通过不等于完整轨迹位级等价或下游能力通过；原50050失败仍保留，具体慢系统调用仍未独立定位。

原八卡已单次接续45058→50058，当前45349/50082、abort=null，恢复监督PID319218/start2458293575存活。best3为45000/44000/45058、latest重合，3份完整状态；45050权重已由原保留策略正常轮替，JSON/恢复证据保留，不另复制测试权重。首段含初始化/恢复/验证/保存/退出共140.671275秒，是8步恢复验收，不拿17.5839秒/更新当稳态速度；真实新长段吞吐另从已产生的训练计时观察。

独立生成等待监督PID320497/start2458326416已由/proc核实，tmux v7_recovery_generation_20261006（13657a rc0）。新包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/package_v1.json（SHAfb377e6af4b72c4b5ed23888e1f161b0eeeb07c399a6d6720fe70e2ee084d9bd）；原生成监督234066仍退出。新生命周期冷核查5项通过（0a677c rc0），原解码/评分函数未改、公开入口和registry未发布；实际在训练监督仍活着时提前发布被明确拒绝，未创建队列或改公开源码。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/root_cold_review_v1.json（SHA499cf8f9c8037d24df30b2f5d24c9ecfc7bbc03f999434bc2a42caccd83f11fe）。固定50082真实完成/退出后才准入9项CDR，当前没有生成结果；原轻链/T4适配缺项与22格消融未完成。

本地f010/42当前3550/3750、3500同曝光验证已归档，下一3750；完整22格9产物/1运行/12未启动。原Goal active，本轮progress；原数据门禁、序列注释停线和独立CUDA暂停保持，无新资源提交/容器重启或旧stage重放。当前源：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/supervision_state.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/state.json。

