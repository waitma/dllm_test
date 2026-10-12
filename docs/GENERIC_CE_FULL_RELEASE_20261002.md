# CE 修正版完整合成训练验收

## 2026-10-02T11:03:50.603902+00:00 当前原CE500学习验收已闭合

CE R21/profile43原500于10:56:27 UTC succeeded/rc0，原verify_diagnostic(500,require_learning=True)唯一root只读调用及独立stdlib复核均通过，trace25..500共20点。学习seed42：四mask平均loss4.0881584711→0.1847930732，AA准确率6.38465%→92.65651%，terminal100%；完全遮蔽mask100 AA71.54613%、loss0.7071843，平均不能称全遮蔽/生成准确率。39当前source物理SHA、9实际输入及62独立引用哈希、日志/双COMPLETE/实际子退出清理均一致，checkpoint目录0。

截至2026-10-02T11:03:50.603902+00:00，原池pending=0/running=7/results=66；CE已完成验收，其余七项最新步数为375、375、375、375、375、375、375/500，仍需真实终态后各自一次原验收。GPU1回原维护状态，没有重启worker/作业/监听或新增GPU任务。

该证据只证既有当前来源人工单GPU学习，不证真实V7/生物下游效果、单模型八GPU-DDP或正式恢复；profile43不是新学习seed。不同来源、不同步数及相同learningseed42的各profile不能当独立多seed比较或最终选型。原CPU Trainer/通信验收保持完成，不重复；正式资源接入及实际共同GPU保存恢复仍未完成。

本回合重新仅读V7清单元数据：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、training_ready字段缺失、audit_pending，两份当前审核均缺。合法当前完整下游资源范围清单及绑定格式缺失，原正式gate无法正确定稿；不造清单/证书、手改ready、降低门禁。真实shard/小batch/审核selection/训练0；停止注释不恢复/重试/转交/绕过、独立CUDA算子暂停、原资源不重启或重复提交。

当前验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json；原只读调用尝试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_attempt.json；independent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_independent_review.json；最新原任务快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_after_CE500_independent_closure.json；V7 metadata /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/V7_metadata_after_original_CE500.json。

## 以下为较早记录

当前来源准备状态见 [实际来源构建](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_build_actual.json)；后续实际终态以 [项目状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 和 [执行状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json) 为准。

本轮接入 R21 seed43 的完整合成验收。上一轮已接受的 CE 模型源 SHA 为 `7e79bfb7d2327c07aa3c39e0bed89997fa627b3fa6e20cb084630b347e886592`，本轮模型字节保持；旧七份完整结果属于旧源码，不能作为修正版的完整通过证据。

新运行目录为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_packages/r21_seed_runtime_v2`。实际 manifest SHA 为 `2ef8f6777120434c0371083838d3ad03d259e807153d7798a745197e7e9cdefd`，source-only、534 文件、531 父来源、39 个 measured source，全部来源包准入标志为 false。

Root 实际 AST/diff 确认 package、producer、entry 的全部函数保持旧协议，只更新固定路径、来源 schema、单一 R21 seed43 身份和外部 guard。新 guard 位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/private_sources/scripts/architecture_prototypes/seed_release_startup/sitecustomize.py`，SHA 为 `2f8a2700542bd03fca7198a4efe91c6cc03c1bc84d35945b6febbb10d39e44e0`。[来源复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_review_before_build.json)。

固定入口要求实际新目录存在，所以先完成 exclusive source-only build，再对实际入口、guard 和进程分派执行最多四项新增 CPU 绑定合同，每套环境 180 秒。没有复制假验收目录或生成假 profile。[预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/execution_budget.json)，[构建前置依赖](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_only_build_dependency_addendum.json)。

通过新增绑定检查和 root 复核后，最多在既有池执行一次完整合成验收：原完整输入/micro32 profile 900 秒，原生命周期阶段、独立完整梯度核对、三个原生 Trainer 优化窗口和四档遮蔽评估保留，总预算 3600 秒；通过必须有实际正常退出、进程清理和当前源码绑定的证据。原阈值、oracle、GA/BF16、源验证和 checkpoint 门禁保持；不另加独立 tiny 波次或 CUDA 算子测试。

完整合成验收尚未取得通过证据，也不构成正式数据训练或生物学效果证据。当前源码学习、完整 checkpoint 恢复、原 data-selection 要求及 V7 当前范围数据审核仍需闭环。V7 manifest 没有 `training_ready=true`，两份当前绑定的审核文件缺失，禁止读取真实 shard/小 batch 或启动真实训练；此前停止的序列注释分支不恢复、不重试或换 agent 绕过。

不重启既有八卡容器、worker 或监听，不重复申请或提交平台任务。旧来源/失败证据与实际训练恢复状态保留，只清理自有临时验证产物。此处记录软件验收状态，不预先声明架构优胜或完成原 Goal。


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


## 2026-10-02T05:48:27.525960+00:00 原完整梯度阶段通过、正式窗口运行

当前 2026-10-02T05:48:27.525960+00:00：唯一新完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438 仍在 GPU7/PID461451，原profile已通过（micro32、length2048、42.18384 GiB）。完整梯度检查原阶段已通过：86参数、8 microsteps/256人工记录，relative L2/maxabs=0、RNG一致，耗时698.856秒<原1200秒；11 weakrefs释放，最终allocated/reserved均0。仅核实原partial阶段，完整raw及独立终态validator尚待任务结束，不将局部通过当作完整准入。

正式优化已写出第1/3窗口（8 microsteps/256人工记录，loss3.660816848、原warmup LR0、参数变化0），仍待第2/3非零LR窗口、四mask评估及实际正常退出。池pending/running/results=0/1/56；其他7worker维护，未新查询平台状态。没有重复入队、Volc提交取消、容器/worker/监听重启。旧完整7/8保持历史，不和修正版局部结果合并为8/8。

V7确认为 immune_v7_mhc_sequence_v1；manifest e12e55cc…ba94、training_ready字段缺失、audit_pending、两份当前审核文件缺失。既有清理保留，这不等于已证实泄漏；真实shard/小batch/训练0。原数据门禁、停止的序列注释及独立CUDA算子暂停全部保留。

源码只读复核纠正剩余工作表述：checkpoint完整恢复核心已实现，初次1000不要求先有checkpoint；真实恢复证据须来自后续实际checkpoint，原resume校验不变。当前源码500步学习证据须位于新profile的同一evidence_root；3窗口不能替代。正式seed训练调用还未接到已有execute_admitted，source-only launcher仍拒绝。完整release通过后补必要学习证据和最小调用绑定，无需再写Trainer或重跑旧Linear恢复。最终架构选择、真实1000→5000及八卡正式训练尚未完成。

[本轮阶段原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_gradient_stage_root_read.json)；[最新状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/gradient_pass_formal_running_state_actual.json)；[剩余工作源码核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/post_release_remaining_work_readonly_review.json)。


## 2026-10-02T05:50:52.859823+00:00 只读终态核验准备

已读实际不可变ENTRY.verify_completion及producer.verify_candidate，准备待本次真实终态后仅核对原件，不重跑GPU。一次辅助文件名猜测不存在，shell只读退出2；未创建模型/测试子进程、未改源码或validator、未影响当前任务。后续使用实际已核实API，不需要该猜测路径。[只读路径错误原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/readonly_verifier_preparation_path_error.json)。

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
