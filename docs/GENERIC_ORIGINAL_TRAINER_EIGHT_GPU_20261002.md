# 原模型八 GPU 训练器与保存恢复工程

## 2026-10-02T14:16:09.255833+00:00 当前状态：V2八卡已终态失败，下一步新版本最小锁修复

截至 2026-10-02T14:14:14.929183+00:00，V2固定共同八卡组 `original_r21_world8_1790948366803074817_36c1a3d5` 已真实终态失败：8个pool result均failed/rc1，组与coordinator均aborted。新版本完成8个checked join、assignment、8个实际子进程注册和8份startup token领取，推进超过V1仅3成员加入的阶段；但没有Trainer目录、已完成训练步数或保存恢复成功证据，不能称八卡DDP/模型训练通过。八份实际子进程receipt均rc=-15、wait/PGgone/cleanup=true，8份rank退出及member cleanup齐全；自有临时checkpoint=0，未删除实际训练恢复状态。

直接错误是锁等待超时。实际父进程PID577559在group_summary持join.lock时等待coordinator.group_status的第二把锁，持join锁2.011217秒，commit仅0.002797秒；随后6个成员锁超时。更早的coordinator持久abort比member abort早约10.90ms。旧coordinator没有持锁计时，最初持锁者/具体慢操作仍未知，不认定死锁或某次I/O是唯一根因，不称完整545来源扫描重回join锁。16份实际timing JSONL及原日志/失败原件保留。

当前原池pending0/running0/results89，八worker新鲜维护待命。平台任务t-20261001033327-fmnql最后实际CLI查询13:54:41 UTC Running；没有新平台提交/取消、容器/worker/监听重启。V2一次执行预算已消耗，当前所有源码、预算、freeze、gate、固定8job IDs及失败原件保持，不自动重试或重放V1/V2；有限监督已真实结束，无观察进程待poll。

原八单卡500人工学习和已完成CPU验收仍保留通过记录；R20/R21/R24/R26已实现，但真实架构选型/理解与生成端下游收益尚未证明。正式V7小batch及训练仍0：training_ready字段缺失/null非True，audit_pending，两份当前数据/下游范围审核均缺。缺审核不是已证明数据泄漏，不伪造证书或降低门禁；停止的序列注释分支不恢复/重试/转交/绕过，独立CUDA算子保持暂停。

下一步在独立的新不可变通用工程修复版本，消除成员锁内等待协调器锁及重复高频状态查询，保留来源/实际host与父子身份/2秒时限/持久abort/退出清理门禁，并针对真实协调器竞争路径做有限必要CPU验证和耗时记录。首次coordinator竞争原因须保留不确定性；消除嵌套锁不能单凭静态推断当通过。不改模型数学或真实数据审核，不增加无关方法波次，不重跑已完成500/profile/FULL/Gloo或原CPU套件。

[根端真实终态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_GPU_group_failure_terminal_root_readonly_review.json)（SHA 2bb59dda1ad9a254362d6fce67b23a3fd80a0dfcbd782156c99d18504a1686ff）；[有限监督原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_group_bounded_readonly_watch.json)；[锁因果及边界复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_failure_lock_remaining_work_independent_readonly_review.json)。Root首轮只读字段误读source_check_state已保留错误原件，依实际JSON修正为source_phase；没有修改原validator/生产源或重试GPU。原Goal 01a0f245-4559-7b62-9dc2-0917a8ed2267仍active，上一准备回合及本执行回合均有实际progress；不新建或重置，实际累计原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/goal_actual_after_V2_terminal.json。

以下带日期内容为保留的历史状态。

## 2026-10-02T13:48:54.596399+00:00 最新状态

V2锁修复的新CPU控制两环境各4/4首次通过，原始及暂存字节已独立/根端复核；未发布或入队GPU。共同八卡最后实际结果仍V1启动锁超时、模型0步；真实训练0，最终架构待选。最新证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_CPU_lock_and_stage_root_acceptance.json；接续 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_GPU_LOCK_REPAIR_20261002.md。以下带日期内容属于历史状态。

## 2026-10-02T12:55:34.324017+00:00 当前实际终态：八成员启动失败，模型0步

截至 2026-10-02T12:54:01.376745+00:00，R20基础对照、R21注意力头门控、R24残差缩放、R26链内局部/全局注意力交替的候选已实现；原八项独立单GPU人工合成实验均完成500步并通过原学习验收。新八卡入口CPU检查在worker/pllm两环境分别4/4通过，已通过的检查不重复。这些证据不证明真实下游提升，最终架构尚未选定。

共同八GPU合成组 `original_r21_world8_1790944282201927112_f554453b` 的唯一V1尝试已终态失败：8个pool result均failed/rc1，直接错误 `Member lock wait expired`，只有3成员加入，assignment/startup未产生，8份member receipt均显示cold child未启动、清理完成。模型训练步数0，原生GPU保存恢复未验证，本次checkpoint目录0。此次失败发生在模型/NCCL启动之前；不撤销已有单卡500步通过记录，也不把源码已实现或入队当共同八卡成功。

只读源码诊断确认锁获取时限为2秒，join临界区内再次核对完整545来源/gate/plan并提交状态；这是需修正的长临界区设计。没有阶段耗时trace，不能认定某次SHA、I/O或调度是唯一根因，也不能仅凭锁文件存在称死锁。下一步保留V1源码、预算、计划、日志和失败原件，在新版本做最小通用协调修复：完整来源核对移至短事务外、两阶段状态绑定防止未核对成员放行、保持启动前完整核对和不可逆abort，并记录真实等待/持锁耗时。只补新增边界所需有限CPU锁竞争/来源变化拒绝/超时清理检查，再经来源复核安排有限纯合成验收；本状态回合没有重试、运行新测试或新入队。

原平台任务`t-20261001033327-fmnql`最后实际查询为12:37 UTC Running；12:54 UTC远端八worker metadata均standby_load/maintenance_only，心跳年龄1.39–2.62秒。池pending0/running0/results81。原八卡容器/worker/监听没有重启，没有Volc submit/cancel；pool成员失败不等于平台八卡容器失败。

V7 `immune_v7_mhc_sequence_v1` manifest SHA仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失（null，非True）、audit_pending，audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明泄漏；正式V7小batch和训练尚未启动，仍须当前manifest的真实数据审核及完整下游范围证据。当前只做通用工程与人工合成验收；安全停止的序列注释分支不得恢复/重试/转交/绕过，独立CUDA算子保持暂停。

[本次根端终态核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_overall_progress_after_group_failure_readonly.json)；[有限观察原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/group_bounded_watch_v1.json)；[原七项500验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original7_terminal_root_v1/original7_terminal_root_acceptance.json)；[入口CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/world8_new_boundary_controls_v1/cpu_boundary_root_acceptance.json)。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267` 仍active，身份、目标及累计不重置；实际工具快照见[Goal原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/goal_actual_user_status_after_group_failure.json)。

以下带日期内容保留为历史记录。

## 2026-10-02T12:10:40.759435+00:00 最新状态

截至2026-10-02T12:05:11.695172+00:00，原七项单GPU人工合成实验全部500/500、实际pool result.state=succeeded/returncode=0，inner与outer COMPLETE均存在；原池pending0/running0/results73。CE R21/profile43原500此前已通过唯一原checker及独立复核，不重复；其余七项刚取得终态，尚未执行各自原verify_diagnostic(500,require_learning=True)，不能记为验收通过。8个原worker在12:06 UTC独立只读核实均standby_load、maintenance_only=True，原资源不重启。

原模型共同八GPU Trainer所需5个私有源码已全部交付，根端11:58 UTC仅做AST/接口和物理SHA静态核对，固定完整545文件来源，15旧保护文件未变。冻结SHA 1b5cfd23a1e9c3cbacb9f30b2fd9dd949fbdcf2d4063962eeac8780ee2311bba。没有导入执行新源码、运行新CPU控制、GPU训练、stage/发布/入队或生成launch gate；代码完成不代表八卡验收通过。原模型/Trainer原生训练和保存方法保持，完整Adam、scheduler、八份真实RNG及恢复后实际状态比较已实现于新源码，等待实际工程验收。

下一步在既有授权范围内，先对七项真实终态各执行一次原500学习验收和结果/来源/退出清理复核（CE已验不重复），完成新来源每环境最多4项必要入口边界控制及独立复核；全部原来源/profile/学习/GPU/完整恢复前置、实际八worker可用及新启动检查满足后，再通过现有pool运行一次原模型共同八GPU纯合成Trainer保存恢复工程验收。连续3步与保存2步后恢复至累计3步用于状态一致性，不代表1000/5000训练阶段或架构效果；不新增架构波次或重复已通过CPU/Gloo/numeric验收。

[12:05真实终态快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_status_after_source_freeze_readonly.json)。以下11:59来源交付段落保留为历史记录。

截至 2026-10-02T11:59:01.221467+00:00，五个必要私有源码已交付并完成根端静态核对，完整来源共 545 个实际文件。**尚未执行新 CPU 控制、GPU 训练、发布或入队，不能称为八卡验收通过。**

原 Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267` 与完整目标保持；当前回合为实际源码进展。原七项单 GPU 合成学习均 475/500、心跳新鲜、没有终态；首项 CE500 已完成原验收与独立复核，不重复。

所有新源码与预算：
/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1

## 已完成的工程准备

| 文件 | 已实现内容 | 实际 SHA256 |
|---|---|---|
| gpu_worker_bootstrap_v1.py | 固定八成员与真实 GPU/主机身份、完整来源和启动计划绑定、不可逆失败处理 | 096038496f8dba2984cc85b0ceb30148ffa05a85a59088b477fd300cf46623c5 |
| gpu_worker_cold_rank_v1.py | 原 worker 单卡掩码、全局 rank 与逻辑设备0、冷 NCCL、持有 FD、父子身份和退出清理 | 8eadd70577cf94cddc5f74bf392a914baee9e738be83baec3e41661cb77a3080 |
| gpu_original_Trainer_driver_v1.py | 原模型和 Trainer 连续3步与保存2步后恢复至3步、真实 native load 观察、状态/指标/尾批比较 | 27d3e4a79638c2c659f9213014b827d9f8aa8555d2cae73062c80c0929fa2971 |
| gpu_complete_checkpoint_validators_v1.py | 原生模型、完整 Adam、scheduler、八份 RNG 和来源检查；恢复再检查真实 sidecar 与八 rank 确认 | 2fc88de01e8d822a0cee468476a4a4b4dfe6aeb6e3a7d67a7090e7e9294a6484 |
| gpu_pool_dispatcher_v1.py | 八个固定成员的严格 worker 参数、原入口委托、实际全组退出后终态、复核后清理自有临时 checkpoint | 4b3fe3ece5a86fb1f3b993bdfa071ec7e6f2750ce0f2f19f30c276f829f6e821 |

根端仅解析 AST、核对接口及实际物理 SHA：五个新文件、五个复用工程模块、完整 immutable runtime 的534文件及其 manifest。15个受保护旧文件未变；没有导入或运行这些新模块的函数。源码冻结文件：
/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_freeze_before_CPU.json

根端静态原件：
/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_delivery_root_static_review.json

## 固定验证范围

只用原不可变来源生成的人工 fixture。模型为原 CE R21 Transformer，保留 d768、8层、12头、MLP3072、完整容量和原损失；source/profile43 与人工学习 seed42 分别记录，不当作独立学习种子。

八卡共同执行 world8 / micro32 / GA1 / global256，BF16；原日程 horizon10000、LR1e-4、warmup100、cosine、eval/save100不变。原 PilotStop1000保留，另有明确短工程 observer 控制连续3、独立保存2、恢复至累计3，每 rank 实际共6次 optimizer updates。**这不代表1000或5000训练阶段完成，也不判断模型生物效果或架构赢家。**

真实参数分组必须来自该 Trainer 原生 create_optimizer；记录后在训练前关闭初始 callback，并使用同一个原v2 factory与完整元数据重挂。各 rank 同序，原训练方法不修改；训练复用同一个 native optimizer，setup失败与正常退出均关闭自有 callback/hook。

保存检查不能直接沿用要求单份 rng_state.pth 的旧正式 validator，也不能造占位文件。新的完整检查保留实际16类必需文件、完整权重与 Adam、scheduler、八份原生 RNG；save先检查真实原生文件，resume额外要求后置 sidecar 和八 rank 确认。原正式门禁保持不变。

最多一个八成员 GPU group、1800秒完整预算、峰 allocated70GiB；原生 load 返回后的实际内存状态、恢复后第三批 IDs/RNG、全部权重/Adam/scheduler、四mask评价与7/17条尾批去重必须真实测量。参数/Adam最大绝对误差1e-6、scheduler1e-12、RNG及下一批精确一致。短步数只验证这些工程性质。

## 接续条件与未完成项

1. 收尾原七项的实际500终态；每项仅一次原 verify_diagnostic(500, require_learning=True)，保留原退出、来源和原始产物核验。原 CE500 与已过 CPU/FULL 不重复。
2. 冻结来源下仅执行新入口边界所需的有限 CPU 控制与独立复核；旧 CPU 通信和 NumericMLP 不重跑。
3. 原七项终态且原 profile/学习/GPU/full-resume 所需证据满足后，生成实际检查所得的单独 launch gate，核验八 worker 可用、固定成员映射与预算。此 gate 仅为合成工程，不是数据审核证书。
4. 只在原八卡资源上执行一次实际 GPU 组验收；不能因观察超时重启容器、worker、监听或重复提交。
5. 保存必要原始证据并独立复核后，删除本目录自有 synthetic checkpoint。清理函数不在 worker启动时自动调用；须全部真实 rank退出，根端复核逐文件绑定后再保留 native JSON及删除清单。正式训练恢复状态不动。

当前真实训练仍为0。V7 manifest为 e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失、audit_pending、当前两审核文件缺；完整下游范围和审核资源绑定缺口仍独立存在。工程验收不能替代审核或放宽任何原data/scope/source/selection/learning/profile/GPU/full-resume条件。

停止的序列注释分支不恢复、重试、换agent或绕过；独立CUDA算子继续暂停。本轮未读取真实shard/小batch、执行审核selection或启动真实语料训练。

## 2026-10-02T14:03:48.274562+00:00 V2固定共同八卡纯合成组一次实际入队

14:02:17 UTC通过原共享池发布既有固定8个g02 packet，group=original_r21_world8_1790948366803074817_36c1a3d5；启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_enqueue_receipt.json（SHA bc9285806e789f2124ac10db4b03ecc5f51112041f5526ecd4add555aa9cc647）。仅三份已独立复核暂存字节发布，原生545来源不变、12旧保护未改、两个public旧文件为明确已审阅变化，另新增V2 public dispatcher，runner最后发布。未使用随机ID enqueue、重stage、重放V1、重启容器/worker/监听或Volc提交取消。独立执行审查已完成，新synthetic-only启动记录绑定115实际前置原件；全部正式准入仍False，V7小batch/真实训练0，停止注释/独立CUDA暂停保持。

当前首次只读状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_fixed_group_status_after_enqueue.json：pool={'pending': 0, 'running': 0, 'results': 89}；具体8job是否被领取/真实group intent以原件为准，实际Trainer/DDP/保存恢复尚未验收，不能把发布入队当成功。一次尝试预算已消耗，不自动retry或再发布pending。有限监督subagent只读观察同8个具体任务至实际终态/有限窗口；观察超时不表示原job退出。真实产物复核通过且各rank退出后再清理自有临时合成checkpoint，原证据及实际训练恢复状态保留。

## 2026-10-02T15:13:28.540259+00:00 当前状态：V3锁修复的CPU验证通过，八卡启动文件已准备

启动失败已确认的链路是持有成员锁时等待协调器锁，触发2秒超时并连锁中止；首次协调器持锁慢操作仍未知。V3拆开成员/协调器锁、把进程探测移到锁外、保留探测后最终中止检查，父进程每0.25秒检查一次组状态并保留0.05秒退出/期限检查；模型数学、原生训练/保存/恢复函数不变。

两环境首次各3/3必要CPU控制通过：pllm8.1196秒、worker7.7225秒。2.25秒显式慢探测期间，另一状态查询耗时分别0.0365/0.0455秒；协调锁等待期间成员锁获取0.00143/0.00135秒。纯观察约2.001秒超时未写abort，真实变更约2.005秒超时写入持久abort，后续变更拒绝；八CPU进程真实注册/启动/退出成功。22个自有子进程全部rc0/wait/进程组消失，临时目录已删。每环境保留15份原始timing内容和实际命令/日志，未创建checkpoint。控制3验证共享snapshot原语而非伪造GPU入口；这些CPU结果不能称真实GPU/NCCL/Trainer已通过。

根端逐值/原件复核：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/CPU_regression_root_review.json。完整545来源冻结SHA546cec975bfc6fbfac9800ea5db21d073008fecee8ccf3197b80c81dcf6f879a，16份前驱/公共文件未变。新三份启动文件仅暂存于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_v1；固定group=original_r21_world8_1790953672752409704_de857f5f、八g03 jobID位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/group_launch_plan.json，plan SHA234d6ec48d10143074564cf00baca77016c87c33116642f4cf1f4d5de4f3ac5d。根端与独立agent已核对registry仅八别名、runner仅一个路径、dispatcher与冻结源字节相同，记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_independent_review.json。未发布、未入队，实际group目录/host intent/launch gate均未创建。

准备阶段完成证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/preparation_completion.json（SHA0186617b44a5e6e5154a08493fbe8280aff63f5d07d6db98f54d59e72679b08f）。下一步在独立执行阶段复核原8 worker及原学习/profile/来源/恢复前置，发布已审阅暂存字节，仅将现有八固定ID一次入队做synthetic Trainer联调；不要重stage/生成ID、重跑已过CPU/500/FULL/Gloo或重放已耗尽V1/V2。原资源最后CLI14:35:34 UTC Running；没有提交/取消/容器或监听重启。

V7 manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready缺失非True、audit_pending、两份审核仍缺，正式小batch和训练0。缺审核不代表已证实泄漏，也不是权限问题。停止注释分支不恢复/重试/转交/绕过，独立CUDA算子暂停，真实数据门禁不变。原Goal仍active且身份/累计不重置，本回合为实际progress，未完成最终训练目标；Goal实际工具记录见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/goal_actual_after_CPU.json。

## 2026-10-02T15:37:05.010795+00:00 V3八卡终态：只读查询超时被上层误升级为中止

V3固定group original_r21_world8_1790953672752409704_de857f5f于15:20:33 UTC实际一次入队，8结果均failed/rc1；8 checked joins、注册、startup领取、rank退出及member cleanup齐全，8子进程actual wait=-15/PGgone/cleanup=true。无Trainer目录或完成训练步/保存恢复证据，checkpoint0，无需删除。原池0/0/97，原资源不重启。V3此次预算已耗尽，源码/计划/原结果保持，不能重放g03；实际发布三文件的前后字节证据保留。

新计时直接确认：heartbeat PID595196/TID595225持锁3.835443757秒，其中commit3.830358904秒。八个超时全是group_status且abort_on_lock_timeout=False，0条mutating timeout，coordinator/abort-request.json不存在；24份coor trace共756行、每文件最多75行，未触及128行上限。首observer timeout完成24216167.42190974→member持久abort24216167.426369432→长心跳释放24216169.18076661→普通coor abort24216169.182679232。成员锁878条，maxhold0.008579805秒/maxwait0.060349118秒，本次无旧成员锁2秒级竞争。八cold日志为空，不能将commit慢唯一归因于fsync、共享存储或GIL。

根端原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_GPU_terminal_root_review.json（SHAc09f71558fa0396a7a10c96634117efdd1cf7fd267a1fb8713a74b25e6399cdd）；独立有限监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_group_bounded_readonly_watch.json（SHA1a68386f9cb2972dd30a85bf9e7abc8b44874b180494cb958d7a98cf1d244967）。545当前源保持，16保护中仅两份已审核发布目标发生计划内变化。观察已结束，无永久监听/待poll进程。最近CLI15:17:02 UTC平台Running；没有Volc提交取消或容器/worker/监听重启。

下一必要修复是通用监督层的异常分类，不改模型或加长2秒锁时限：仅来源绑定、明确来自group_status的只读观测超时可在原30秒freshness/启动与总deadline内继续观察同一child；失败不得刷新健康时间，未取得首份有效状态也必须有界。真实持久abort、mutating/成员锁超时、身份来源错误与异常退出仍失败，暂不可观测不能产生成功。应在新私有版本用生产复用的实际监督循环做一个必要CPU场景及其拒绝分支，不用替代循环/伪GPU intent，不重复已验旧控制。独立只读策略复核确认这个边界；V3失败原件和完整门禁保持。正式V7小batch/训练仍0，双审核仍缺；停止注释分支及独立CUDA暂停不变。原Goal继续active，本执行回合有实际证据进展而未完成最终目标。
