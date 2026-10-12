# 当前生效接续（2026-10-12T08:39:18.895940+00:00）

用户最新要求训练约270M模型。原Goal身份及累计不变，未新建/重置/修改服务状态；旧75.6M已完成，当前任务为270M独立scratch运行。旧最终验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/formal_completion_root_review_20261010.json，300000步、17完整状态文件、八rank成功、COMPLETE和训练退出码0一致；八卡占卡实际恢复后，本次新训练才发布。

当前run /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512；实际60305/300000步（本次读取），参数268590112（16层/d1024/head16/FFN4096）、V5数据、50词表、对象容量302、DDP8/global512=64×1×8。固定运行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010，正式票据v5_270m_scratch_bs64_20261010。现有容器t-20261007153431-vhx2f未重启，supervisor737、训练PID964799；standby为空。八卡synthetic峰值allocated42.79GiB/reserved48.88GiB、3步有限loss、NCCL通过；真实语料85分片全哈希门禁通过，已产生更新步，首个日志loss3.2884。启动证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/formal_start_review.json。

当前只读监督exec26867、PID1737482，上限1296000秒（15天）；实际状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010/latest.json，终态exit.json。先poll同句柄/核实进程，勿重复开观察器。旧90153已actual exit0（工具5bb7fc），不再poll。训练退出后原容器监督器自动恢复八卡占卡，本观察器只确认真实退出和新PID初始化日志，不自动重试训练，也不自动宣告最终产物全部验收。

2026-10-12本次核实已到60305/300000步（约20.10%）；最新完整保存60000步/epoch4.0279，累计60次原生验证。主eval_loss（全遮蔽/source-macro）最佳32000步1.993467，最新60000步2.007204；相对17000步旧最佳仅改善0.0027%，整体仍波动。最近10000更新墙钟平均2.8940秒/步，包含窗口内验证/保存；不中断约余8.0天。 60000步完整核对已通过，报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint60000_root_review.json；best3为32000/17000/43000，加latest60000。下一61000步；禁止无依据重复已完成验收。状态与计时 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261012_checkpoint60000_root_review.json。本轮未执行270M下游评测。所有历史已暂停消融、序列注释安全停线、独立CUDA算子暂停和数据门禁保持；training_ready=false不得启动，不伪造审核，不改旧模型/数据绑定，不重启旧终态任务。没有测试checkpoint；新旧正式best3＋latest及恢复状态保留。最新表格见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。

# 最新运行状态（2026-10-07T07:28:27.608241+00:00）

原消融池已按用户停止请求终态，不能直接复用已释放容器。新任务t-20261007151807-cq4fb已提交并启动失败（正式0步），原Goal继续active且身份累计不变。用户指定training_config超参数一致；launch_v3是未提交候选，不自动复活原消融或失败作业。当前证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/failure_review.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/training_config_agreement_20261007.json。

# 当前执行覆盖说明（2026-10-07T07:08:05.743243+00:00）

用户现要求暂停原22格消融并提交当前新版本训练。原Goal保持active，身份及累计用量不变；下文旧“继续至3750/补齐22格”指令由本次用户暂停覆盖，不得自动续跑。停止证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json。

# 持续目标：根据部分真实数据确定架构，验证通过后正式训练

当前状态请先读取 [PROJECT_PROCESS.md 当前状态与Active表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，再看本文件的 `CURRENT：在跑任务、实际验收和接续责任` 表与 `CURRENT_VERIFIED_WAIT` 标记后的最新记录。机器交接以 [root_observation_latest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json) 的 `current_snapshot` 和时间戳为准；其他字段可能保留历史证据。下方带日期的旧章节不代表当前执行状态，不可据其重发已结束的任务。

## 2026-10-06T13:21:35.490096+00:00 用户新增两轮后生成评测；准确节点50082

当前冻结运行仍到25042。实际25041步/epoch，两轮为50082；需要再接25040步，纠正此前50084估算。CPU发现并修复跨轮skip loader重置epoch的问题，修复后原生恢复/样本顺序/最终模型与连续两轮完全一致，临时权重已清理。独立分布式接续与生成入口核实仍进行中。计划和证据唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md；原Goal身份保持，完成须包括新目标与原消融，未提交/重启任务或降低门禁。

## 2026-10-06T13:05:08.199910+00:00 9000验证与当前运行

2026-10-06T13:05:08.199910+00:00：正式where-DDP当前9303/25042；最新完整验证/保存9000已核对，当前段7016→12016仍在训练，最近段退出验收7016。监督PID212174/start2455180199与八worker正常，无abort。best3 7000/7008/7016＋latest9000共4份完整状态，8000权重已轮换删除、JSON证据保留。

1000步8001→9001墙钟平均0.722880秒/步（含一次验证/保存），999普通步均值0.713420秒；这是实际DDP/global256的新完整窗口，不含段启动恢复，不作FSDP吞吐声明。9000主eval_loss1.890182、8000为1.889376；质量指标有升降，继续原f011与总25042计划。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint9000_root_review.json；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。

本地f010/42当前1600/3750，已验收1500，下一1750；22格仍9完整epoch产物、1运行、12未启动，500图10/22、3750图9/22均PARTIAL。下一正式10000；原Goal仍active，上一回合/本回合均progress。有限观察63484已退出0，持续监督及训练继续；禁止重启旧监督、重复提交，原注释停线/CUDA暂停/数据门禁保持。

## 2026-10-06T12:52:51.652079+00:00 当前八卡接续状态

2026-10-06T12:52:51.652079+00:00：新where-DDP当前8278/25042，8000步验证/完整保存已核实，段目标12016；持续监督PID212174/start2455180199和8worker正常，abort=null。best3 7000/7008/7016＋latest8000共4份完整恢复状态。300步均值0.740862秒（稳态含数据/DDP，排除验证保存；旧10.2–10.7秒），原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json。新版模型仅where等价展开，仍micro32/GA1/world8/global256和math attention。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_WHERE_DDP_CONTINUATION_20261006.md。

原7000完整checkpoint接续，旧最后打印7600后的约600个已打印未保存更新已重算；不冒称原进度无损。新段7000→7008、7008→7016通过实际八rank原生恢复/保存/退出，当前长段7016→12016仍在训练。最近段退出验收7016与最新完整checkpoint8000是不同概念。旧监督PID3977587已停，不重启。下一常规验证9000；原注释停线、CUDA暂停及数据门禁保持，不重复提交或重启容器。

本地f010/42仍运行，新增1500点已核对：平均AA改善、全遮蔽和END有回退；下一1750。22格仍9完整epoch产物、1运行、12未启动；500图10/22、3750图9/22均PARTIAL。原Goal持续active，不缩小范围或宣称全部完成。

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


## 2026-10-06T12:24:40.222288+00:00 用户授权停止已执行；rank6清理回执缺失待真实进程复核

原监督PID3977587已经由STOP退出；旧10016段收到一次显式代码切换停止消息，最后训练日志7600，恢复点7000保留。8个slot controller真实退出失败（用户中断造成），7份rank清理回执确认进程与进程组消失；rank6的reap等待1秒超时，cleanup=null、旧group terminal尚未完成。不能将此写成8rank完整退出验收通过。原容器和8worker仍存在并已待命，未提交/重启平台任务。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/OLD_CONTROLLERS_TERMINAL.json。停止观察器因未处理cleanup=null在最终汇总抛TypeError；停止消息已实际执行一次，不重发。下一步使用现有pool的独立新启动包装器，在容器同host/boot上只读检查旧PID/PGID全部消失，形成单独补充证据后才允许模型启动；旧回执不改造、不伪造退出码或清理结果。新模型尚未启动，源/data门禁保持。


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


最新实际状态以本文件 CURRENT_VERIFIED_WAIT 区的带时间戳记录为准；其余带日期章节保留历史证据。

<!-- CURRENT_EXECUTION_HANDOFF_START -->

## 2026-10-06T05:52:50.913570+00:00 f010 seed42已通过真实准入并单次启动

2026-10-06T05:52:50.913570+00:00：正式八卡f011继续运行，最新核实训练日志为5425/25042步、loss=0.4741，最近完整验证/保存5016，当前段目标10016、下次常规验证6000。原监督PID3977587/start2446951610与八rank新鲜心跳已核实，无abort；DDP+NCCL、micro32/GA1/global256保持。

f010 seed42于2026-10-06T05:50:34Z真实单次启动，持共用格子/设备锁的guard.main二次检查通过，PID80642/start2452750909已exec原冻结训练CLI。当前处于启动初始化，run_provenance已出现、首个训练loss尚未观察到；wrapper rc0只证明tmux启动，训练退出码仍未知。原120000条、3750步完整epoch、micro32/GA1/eval250/eval4、metrics-only不变。此项使用已有授权本地共享GPU，未申请、抢占或重启八卡正式资源，不作受控测速。

完整22格：9格完整epoch产物（8格成功退出已核实、f000/42退出码未知），f010/42运行中，另12格未启动。原3750步图仍9/22、PARTIAL，未重画或声称新末点完成。实际启动记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_launch_package_preparation_20261006/actual_launch_receipt_v2.json；根观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_launch_package_preparation_20261006/actual_launch_root_observation.json；新显式选择文件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_selection_v2.json（SHA9a60ebd56551c71c16faf86dd5efd4c44448e71fe3a44de31c86221a089dc0fc），只把f010/42从pending改为该真实local attempt，其他21格保持，旧选择/图保留。

后续直接观察同一tmux任务v7_local_f010_seed42_attempt1及原正式监督，禁止重放本启动包。原训练PID/start/argv与真实guard/header/started绑定已经核实；结束后读取原runner保存的exit_code/done/failed再形成真实exit v2，不把PID消失或COMPLETE等同rc0。元数据发布工具6254d9在三个原件已写入后，仅末尾打印误用native_checkpoint_step键报错；0be65d只读确认正确字段latest_completed_checkpoint_step=5016，未重发启动或改造结果。

本轮progress（实际接续一项原计划消融）；上一问答仅状态答复属no_progress。完整Goal仍active，正式25042及22格最终图尚未完成。序列注释停线、独立CUDA专项暂停、403不重试、原审核及来源门禁保持；未知GPU进程未干预，受控profile仍UNRUN。无测试checkpoint生成，正式恢复状态保留。

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


## 2026-10-06T03:44:45.778430+00:00 修正跨训练段的步数观察逻辑（只读元数据）

旧根窗口状态读取固定segment_5016，并要求最近完整checkpoint步数恰有同一步训练日志。因日志每25步打印、接续保存点为5016/10016等，此假设会在自然接段时失效；这是进度记录逻辑的问题，不是训练失败。新增 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/observe_v7_logged_steps.py，按原监督state定位当前段，核对enqueue绑定的plan、既定25步日志周期与25042/1000/cosine调度，并匹配原生trainer_state历史。保存步数与最近已打印步数分开；未完整写入的尾行不计入，缺行/重复/原生记录不一致时拒绝给出推测步数。

新5项人工元数据检查通过（工具dc86d8实际rc0、测试0.005秒）：非整25保存点、下一段5025首日志、尚无新日志、未写完尾行，以及缺失/重复/原生不一致拒绝。现有真实元数据只读核对通过（工具36596b实际rc0）：当前190条训练日志映射到4750步，160条与checkpoint-4000原生训练记录匹配，学习率全部相符；尚未观察真实5016接段，不把模拟检查声称为实际接续验收。

证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/logged_step_reader_review_20261006.json（SHAee02baa4fb65512ad31e1ac349b376b2c64bec4f3528d2b9ae4ed590c4337af8）。根窗口状态更新改用此读取器；冻结训练、原监督器、配置、数据门禁均未改动，没有GPU/模型测试、新任务、重启、权重或临时测试目录。完整25042及22格目标保持active；当前观察会话51035继续，既有安全停线和CUDA专项暂停不变。




## 2026-10-06T03:22:05.045022+00:00 本地3250步验证：AA重建回升，整体结束标记继续回落

本地f000 seed42完成3250步验证、104000/120000条曝光（86.67%），继续3750完整epoch。平均AA CE0.849367→0.831506，全遮蔽AA CE1.767363→1.711951、准确率38.5985%→42.3560%；整体全遮蔽terminal准确率29.2052%→25.3235%。含context37条的全遮蔽terminal准确率43.1818%→45.4545%，但CE上升；组别和指标方向分别记录，不据单点或已有正例关系准确率推断功能收益。

新点170计数/40mask_ratio、1024验证记录与provenance保持，summary匹配，无终态或测试权重。详细数值见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3250_trend_20261006.json（SHA751d8c32272cc014655d5967536d5e0c89eee48e806cb1e82b44c2b6629bd82f）。原八卡f011最近完整保存4000且继续25042目标。完整22格仍8完成/1运行/13未启动，固定500/3750图不重画；下一本地3500、正式5000，既有训练和门禁不变。


## 2026-10-06T02:21:32.688388+00:00 本地3000步验证完成：平均AA CE持平，全遮蔽与结束标记准确率回落

本地f000 seed42完成3000步验证、96000/120000条曝光（80%），继续3750步完整epoch。平均AA CE0.849889→0.849367基本持平；全遮蔽AA CE1.744851→1.767363、准确率41.8326%→38.5985%；全遮蔽terminal准确率39.0018%→29.2052%。含context37条的全遮蔽AA CE上升、AA准确率下降，terminal准确率43.1818%持平。已有正例关系CE下降不等于正负binding能力。

仅核对新3000点，170计数/40mask_ratio、1024验证记录与provenance保持；summary匹配，无终态或测试权重。详细统计见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation3000_trend_20261006.json（SHAb11c26591a09b065f7176f76abf0a42a46d1db24036f2f23b36ed7185ba1e1fa）。原八卡f011最近完整验证保存4000并继续25042目标，两配方不混用。完整22格仍8完成/1运行/13未启动；固定500/3750图保持，下一本地3250/正式5000，既有训练和所有门禁不变。


## 2026-10-06T01:27:15.290670+00:00 正式4000步验证与保存完成：AA改善，结束标记与长度误差仍波动

正式checkpoint-4000已完成：17原生文件与6JSON归档齐全，8rank参数一致且已更新；固定1330验证记录/112计数字段保持，70聚合与18结束位置检查通过。平均AA CE0.773253→0.759145，全遮蔽AA准确率42.1558%→42.9299%；全遮蔽terminal准确率57.7268%→48.7537%，长度MAE2.761715→4.073815，来源方向不一致，不能仅凭平均loss宣布架构或生成能力通过。

细项与逐来源/长度限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_root_observation.json（SHA98f9e2b5e386ef0d3cb3fabe9d9a8ddc8bcc1dad5e935a57fb7d3c5be68c91a9）、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint4000_endpoint_diagnostics.json（SHAbc7170739dd205bbabad27213e8bd1592ab1ddf2bec5c73b431ae0101d404e11）。原八卡继续5016段/25042总目标，本地最近2750验证；下一正式5000、本地3000。旧1000权重按原三份保留规则轮替、JSON证据保留；未重启或重跑模型。完整22格及全部原边界保持，Goal active。


## 2026-10-06T01:18:25.792578+00:00 本地2750步验证：总体重建回升，context结束标记仍有回落

本地f000 seed42完成2750步验证、88000/120000条曝光，仍继续3750步完整epoch。平均AA CE0.897508→0.849889；全遮蔽AA CE1.898242→1.744851、准确率37.9543%→41.8326%；全遮蔽terminal准确率20.1479%→39.0018%。含context37条的全遮蔽AA CE略升、terminal准确率52.2727%→43.1818%，正例关系CE也回升，不能只报告总体改善或声称结合功能收益。

新点170计数/40mask_ratio与2500相同，原provenance和1024验证记录保持；summary匹配，无测试权重或终态。学习率核对的检查脚本索引偏一已按冻结源修正，训练未受影响。详细数值与限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/validation2750_trend_20261006.json（SHAb2deacc4257cef0180858d94ac99c3b903ccb0351cf2bef769db0420acf59a7a）。完整22格仍8完成/1运行/13未启动，固定500/3750图不按中途点重画。原正式八卡继续完整25042目标；下一本地3000/正式4000。


## CURRENT：在跑任务、实际验收和接续责任

更新：2026-10-07T06:59:48.011192+00:00。同一Goal/threadId 01a0f245-4559-7b62-9dc2-0917a8ed2267，objective「开始监督跑正式版本的实验」，active；不缩减、重置或标完成。下方旧日期记录属于历史。

| 工作 | 当前已核实状态 | 证据 |
|---|---|---|
| 八卡正式训练 | 50082/epoch2完成，最终8rank成功退出，监督rc0；不启动第三轮 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50082_root_review.json |
| 正式保存 | best3 45000/46000/47000＋latest50082，共4份完整恢复状态 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/checkpoint50082_root_review.json |
| 下游CDR | 9项实际完成，9741个CDR区域案例；监督rc0 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json；数字唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr |
| 完整22格消融 | 10完整产物（9成功退出、f000/42未知）；8运行/4排队，3550–3575步，3500验证已核实 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T065937Z_watch_open.json |
| 现有消融图 | 500固定点18/22 PARTIAL；3750末点仍10/22 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/root_review.json |
| 后续报告选择 | 已接入本地f010/42真实rc0；冻结旧选择不改 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/mixed_report_selection_with_local_exit_v1.json |
| 轻链生成/评分 | 4000预测与评分完成、实际exit0，逐项分母复核通过 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json |
| T4生成/评分 | 全部34条件生成/评分完成，实际exit0，主视图覆盖完整 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json |
| 尚未完成 | 原22格完整epoch、实际终态核实及完整消融图 | 仍10完整产物、8运行、4排队；f000/42退出未知保留 |

已完成V7正式对照的配置为f011=R26链内局部/全局交替＋R24固定残差0.1、no-ESMC A4 scratch，75,601,176参数，micro32/GA1/world8/global256。该已完成V7训练的数据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002 的training_ready=true，双证书仅immune_test_only_observed_pair_exact_v3；不声称全部query/few-shot或相似度去重。context序列CE=0、仅已有正例关系监督，256/64槽位和protd仍保留。

原训练/生成各监督均退出，原STOP和50050失败记录保留。当前执行者是原资源t-20261005205851-lqfjg中的8worker；无容器/worker重启、平台提交或失败stage重放。新消融尚无另一个常驻监督，由root观察实际pool结果，不自动重试；不可再次运行publisher。正式/本地/source cohort差异应在最终图保留，不能宣称全程位级复现或受控吞吐。

<!-- EPOCH_COMPLETION_HANDOFF_BEGIN -->

### 原八项即将进入3750步完成检查

已接受3500；当前只读会话6737观察3750。原worker可能在第一项完成后就领取原四份pending之一，早于其余七项完成。观察器出现state_changed_inspect或读取短暂空缺时，先查原job ID的COMPLETE.json、results、实际worker领取和pending票据；不能把正常接续当失败，也不能仅凭观察器关闭就认定训练完成。不得因此重启/重发任务。

首次完成后要从原始证据重新计算完成/运行/排队数，不再沿用10/8/4断言。3750不能调用中途read_saved_fixed_point_v1.py或accept_saved_fixed_point_state_v2；完整报告使用冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/source_v1/plot_v7_mixed_ablation.py 的endpoint_step=3750及选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/mixed_report_selection_with_local_exit_v1.json。待新500/3750覆盖时才更新固定图；产物完整与实际退出分别核实，保留f000/42历史退出未知和混合源码限制。原数据/注释停线、CUDA暂停及正式恢复状态保留要求不变。

<!-- EPOCH_COMPLETION_HANDOFF_END -->

<!-- CURRENT_VERIFIED_WAIT -->

<!-- LATEST_ABLATION_OBSERVATION_BEGIN -->

最新核实：2026-10-07T06:59:37.533785+00:00。原22格消融10完整产物（9成功退出、f000/42退出未知），8项运行至3550–3575/3750步、4排队；3500已验收，下一3750。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T065937Z_watch_open.json。正式两轮与三类生成已完成；原Goal保持active，原范围和全部停线/数据门禁不变。新版本V5数据选择以PROJECT_PROCESS为准，不自动迁移原消融。 当前只读会话6737实际仍在运行。

<!-- LATEST_ABLATION_OBSERVATION_END -->

最新 2026-10-07T06:08:49.794322+00:00：原八卡平台实际Running，八项推进到3200–3225/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T060849Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T06:05:57.756313+00:00：原八卡平台实际Running，八项推进到3175–3200/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T060557Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T06:03:22.984730+00:00：原八卡平台实际Running，八项推进到3150–3175/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T060322Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T06:01:00.599765+00:00：原八卡平台实际Running，八项推进到3125–3175/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T060100Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:56:34.967542+00:00：原八卡平台实际Running，八项推进到3100–3125/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T055634Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:53:57.733880+00:00：原八卡平台实际Running，八项推进到3100–3125/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T055357Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:51:29.041095+00:00：原八卡平台实际Running，八项推进到3075–3100/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T055129Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:49:46.650117+00:00：原八卡平台实际Running，八项推进到3075–3075/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T054946Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:47:30.103556+00:00：原八卡平台实际Running，八项推进到3050–3075/3750步、loss有限、4排队，无新失败或退出。3000已验收，下一3250，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话3185仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T054730Z_watch_open.json。上一轮progress、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:41:32.629026+00:00：八项3000已核实、18个同曝光点；原训练3000–3025/3750，8运行/4排队，完整epoch10/22，下一3250。只读会话17579实际exit0已结束，不再poll或重启；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first3000_acceptance_receipt.json。正式两轮与三类生成保持完成，原Goal active、本轮progress，完整范围及全部停线/门禁保持。

最新 2026-10-07T05:29:33.219783+00:00：原八卡平台实际Running，八项推进到2925–2950/3750步、loss有限、4排队，无新失败或退出。2750已验收，下一3000，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话17579仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T052933Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:25:13.449317+00:00：原八卡平台实际Running，八项推进到2900–2900/3750步、loss有限、4排队，无新失败或退出。2750已验收，下一3000，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话17579仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T052513Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:21:06.548876+00:00：原八卡平台实际Running，八项推进到2875–2875/3750步、loss有限、4排队，无新失败或退出。2750已验收，下一3000，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话17579仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T052106Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:17:07.532222+00:00：原八卡平台实际Running，八项推进到2850–2850/3750步、loss有限、4排队，无新失败或退出。2750已验收，下一3000，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话17579仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T051707Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:12:48.577896+00:00：原八卡平台实际Running，八项推进到2800–2825/3750步、loss有限、4排队，无新失败或退出。2750已验收，下一3000，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话17579仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T051248Z_watch_open.json。上一轮progress、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T05:08:27.070594+00:00：八项2750已核实、18个同曝光点；原训练2775–2800/3750，8运行/4排队，完整epoch10/22，下一3000。只读会话14812实际exit0已结束，不再poll或重启。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2750_acceptance_receipt.json；V5/V7模型与数据比较 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/v5_v7_model_data_comparison_20261007.json。正式两轮及三类生成保持完成，原Goal active、本轮progress，完整范围与全部停线/门禁保持。

最新 2026-10-07T04:44:23.018740+00:00：原八卡平台实际Running，八项推进到2600–2625/3750步、loss有限、4排队，无新失败或退出。2500已验收，下一2750，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话14812仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T044423Z_watch_open.json。上一轮verified_wait、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T04:39:55.385560+00:00：原八卡平台实际Running，八项推进到2575–2600/3750步、loss有限、4排队，无新失败或退出。2500已验收，下一2750，完整epoch10/22；正式两轮及三类生成保持完成。同一只读会话14812仍实际运行，继续poll，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T043955Z_watch_open.json。上一轮progress、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T04:34:13.262347+00:00：正式两轮及三类生成保持完成；原八卡消融8运行/4排队，2525–2550/3750步。2500的18个同曝光点已核实，完整epoch仍10/22；下一2750/3750。只读会话71161实际exit0已结束，不再poll或重启；原worker继续。记录脚本曝光常量误替换已更正，训练未失败；详情及指标见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2500_acceptance_receipt.json。原Goal active、本轮progress、上一轮verified wait，完整范围与全部停线/门禁保持。

最新 2026-10-07T04:15:25.251409+00:00：上一/本轮verified wait；原平台实际Running，同一观察会话71161持续运行，八项推进到2400–2425、loss有限、4排队，无新失败或退出。2250已验收，下一2500/3750，完整epoch10/22；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T041525Z_watch_open.json。继续poll原句柄，不重复启动，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T04:05:06.838563+00:00：上一轮progress、本轮verified wait；原平台实际Running，八项推进到2325–2350、loss有限、4排队，无新失败或退出。2250已验收，下一2500/3750，完整epoch10/22。同一只读会话71161仍实际运行，继续poll该句柄；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T040506Z_watch_open.json。原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T03:57:38.723075+00:00：原平台实际Running，8项2275–2300步、4排队，2250已验收，下一2500/3750。只读会话71161已接续且实际仍运行，请继续poll同一句柄；实时记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/bounded_watch_20261007T035544Z_fixed2500.jsonl，状态入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。上一轮progress、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T03:54:11.238678+00:00：八项2250产物/预算核对通过，18个同曝光点；残基CE改善但终止仍有回退，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。完整epoch10/22，8运行/4排队，下一2500/3750。原只读会话73314实际exit0已结束，原worker继续；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2250_acceptance_receipt.json。原Goal active、本轮progress、上一轮verified wait，完整范围及全部停线/门禁保持。

最新 2026-10-07T03:39:36.561111+00:00：上一/本轮verified wait；原平台实际Running，同一观察会话73314持续运行，八项推进到2150–2175、loss有限、4排队，无新失败或退出。2000已验收，下一2250/3750，完整epoch10/22；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T033936Z_watch_open.json。继续poll原句柄，不重复启动，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T03:28:21.579818+00:00：上一轮progress、本轮verified wait；原平台实际Running，八项推进到2075、loss有限、4排队，无新失败或退出。2000已验收，下一2250/3750，完整epoch10/22。同一只读会话73314仍实际运行，继续poll该句柄；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T032821Z_watch_open.json。原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T03:20:58.210910+00:00：原平台实际Running，8项2025–2025步、4排队，2000已验收，下一2250/3750。只读会话73314已接续且实际仍运行，请继续poll同一句柄；实时记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/bounded_watch_20261007T031905Z_fixed2250.jsonl，状态入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。上一轮progress、本轮verified wait，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T03:17:48.065803+00:00：八项2000产物/预算核对通过，18个同曝光点；context残基改善而END回退，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。完整epoch10/22，8运行/4排队，下一2250/3750。原只读会话31864实际exit0已结束，原worker继续；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first2000_acceptance_receipt.json。原Goal active、本轮progress、上一轮verified wait，完整范围及全部停线/门禁保持。

最新 2026-10-07T02:56:51.244512+00:00：上一/本轮verified wait；原平台实际Running，同一观察会话31864持续运行，八项推进至1850–1850步、loss有限、4排队，无新失败或退出。1750已验收，下一2000/3750，完整epoch10/22。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T025651Z_watch_open.json；继续poll原句柄，不新建重复观察或重启训练。原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T02:45:55.201332+00:00：上一轮progress，本轮verified wait；原平台实际Running，八项1775–1800步、loss有限、4排队，无新失败或退出。1750已验收，下一2000/3750，完整epoch10/22；原只读会话31864仍运行，继续poll同一句柄，不重复启动。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T024555Z_watch_open.json。原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T02:42:21.260230+00:00：八项1750产物/预算验收通过，18个同曝光点，完整epoch10/22；8运行/4排队，下一2000/3750。原只读会话33333实际exit0已结束，原worker继续；不得据此重启训练。根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1750_acceptance_receipt.json，详细指标与限制见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。原Goal active、本轮progress、上一轮verified wait，完整范围及全部停线/门禁保持。

最新 2026-10-07T02:26:44.254795+00:00：上一/本轮verified wait；原平台实际Running，原观察会话33333持续运行，八项已到1650、loss有限、4排队，无新失败或退出。最新验收1500，下一1750/3750，完整epoch10/22；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T022644Z_watch_open.json。继续poll原会话，不重复启动，Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T02:20:57.096786+00:00：上一/本轮verified wait；原平台实际Running，沿用会话33333核实原八项推进到1600、loss有限、4排队，无失败或新退出。1500已验收，下一1750/3750，完整epoch10/22；观察仍运行，请继续poll原句柄，无新进程启动。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T022057Z_watch_open.json；Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T02:15:34.713397+00:00：本轮verified wait、上一轮progress；原平台实际Running，相同八job/worker记录PID、心跳新鲜且loss有限，八项均推进到1575，4排队、无新失败或退出。1500已验收，下一1750/3750；完整epoch10/22。观察会话33333仍实际运行，继续poll同一句柄，不重复启动；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T021534Z_watch_open.json 为追加日志的不可变已观察前缀。原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T02:08:15.750729+00:00：八项1500产物/预算核对及9种架构18/22同曝光对比完成；重建、context和END仍有取舍，正式f011保持。原八卡Running，8项1525、4排队，完整epoch10/22，下一1750/3750。根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1500_acceptance_receipt.json，CSV /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/fixed1500_architecture_comparison.csv；只读观察已结束，原任务继续。原Goal active、本轮progress、上一轮verified wait，完整范围及全部停线/门禁保持。

最新 2026-10-07T01:41:27.101184+00:00：本轮verified wait、上一轮progress；原平台实际Running，十分钟21次观察相同八job/worker记录PID、心跳新鲜且保存loss有限，八项均推进到1350，4排队、无新失败或退出。1250已验收，下一1500/3750；完整epoch10/22和三类生成完成状态保持。只读观察正常结束，原任务继续。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T014127Z.json；原Goal active、完整范围及全部停线/门禁不变。

最新 2026-10-07T01:29:12.607715+00:00：原八项1250验证通过，40000/120000曝光；平均AA改善而context全遮蔽/END仍分化。原平台Running，8项1250–1275、4排队；完整epoch10/22，下一1500/3750。只读观察正常结束，原任务继续。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1250_acceptance_receipt.json；上一轮verified wait、本轮progress，原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T01:11:20.093764+00:00：上一/本轮均verified wait；原平台实际Running，十分钟21次观察相同八job/worker记录PID、心跳新鲜，八项均推进到1125，4排队、无失败或新退出。1000已验收，下一1250/3750；完整epoch10/22和三类生成完成状态保持。有限观察自然结束，原任务继续。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T011120Z.json；Goal active、完整范围及全部停线/门禁不变。

最新 2026-10-07T00:59:29.637370+00:00：本轮verified wait，上一轮progress；原平台实际Running，180秒七次观察相同八job/PID、心跳新鲜且实际推进到1025–1050，4排队、无新失败或退出。1000已验收，下一1250/最终3750；完整epoch仍10/22，三类生成保持完成。有限观察正常结束不代表训练退出。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/verified_wait_20261007T005929Z.json；原Goal active、完整范围及全部停线/门禁保持。

最新 2026-10-07T00:53:58.799641+00:00：正式两轮及计划三类生成已完成；原八项1000固定验证通过，8运行/4排队，完整22格仍10产物。原平台Running，有限观察已正常结束、训练继续；下一1250/最终3750。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first1000_acceptance_receipt.json；原Goal active、本轮progress，身份/累计用量/完整范围及全部停线/门禁保持。

最新 2026-10-07T00:39:13.894414+00:00：TCR完整生成和评分均实际exit0，三类计划生成已完成，报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json。原八卡Running，8项消融875–900、4排队；750验证已核实，完整22格仍10产物，下一1000/3750。原Goal active、本轮progress，完整范围和全部停线/门禁保持。

最新 2026-10-07T00:20:18.580374+00:00：八项750验证完成，8/8平均AA改善而END回落，正式f011不改；完整epoch仍10/22、8运行/4排队。TCR同一PID397587为2767/3434、未退出/评分。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first750_acceptance_receipt.json。上一turn verified wait，本turn progress，原Goal active且完整范围保持。

最新 2026-10-07T00:03:09.851527+00:00：本轮/上轮verified wait；原TCR PID/start/argv11次存活，1626/3434、未退出/评分；原8项消融625–650、4排队，无失败或新终态。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/verified_wait_20261007T0002.json。Goal active，原停线/门禁及完整范围保持。

最新 2026-10-06T23:55:58.543024+00:00：本轮verified wait，实际TCR原PID/start/argv九次存活、1151/3434、未退出/评分；原8消融575–600、4排队，无新终态。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/verified_wait_20261006T2355.json。Goal active，不因等待计算而标blocked，不重做已完成验收。

最新 2026-10-06T23:50:17.965727+00:00：原TCR进程实际PID/start/argv七次核对存活，757/3434、未退出/评分；原八卡平台按任务名确认Running，消融550、8运行/4排队。V5四视图比较依据已核实，根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/comparison_and_verified_wait_root_review.json。上一turn progress；本turn progress＋verified wait，不是阻断，原Goal active且完整范围不变。

最新 2026-10-06T23:42:54.045644+00:00：八项500验证通过，新固定500图18/22，完整3750仍10/22；8运行/4排队。T4同一PID397587生成353/3434，未退出/评分；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/live_status.json。原Goal active，下一T4完整评分和原消融后续验证。

最新 2026-10-06T23:39:56.588650+00:00：T4 GPU验收通过、完整生成PID397587/start2459151386运行，80/3434原始候选；纯CPU评分入口准备完成但尚未评分。消融475–500步、8运行/4排队，完整22格仍10。当前证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_launch_live_snapshot.json；旧记录为历史，原Goal active，禁止重复启动。

最新 2026-10-06T23:35:21.048373+00:00：轻链全量评分已完成并写入RESULTS，原评分PID退出，本地GPU释放。T4纯序列评分人工控制通过，真实GPU/全量仍待接续。具体证据与下一步见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 顶部；下方旧日期为历史，禁止重复启动已完成任务。

最新 2026-10-06T23:24:18.465554+00:00：轻链4000预测已真实exit0、原生成PID消失，不能重放；评分现由PID346252/start2459060767持同一local_gpu0锁执行，尚无分数终态，根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/score_local_launch_root_observation.json。八卡消融375–400步、4排队、10/22完整；T4 GPU/全量待本地资源接续。下方旧“生成中”是历史，原Goal保持active，完整范围和全部停线/审核门禁保持。


最新 2026-10-06T23:18:02.210187+00:00：T4全量入口及3434原始候选预算已准备并通过artifact控制，尚待实际GPU preflight/全量运行；/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_generation_preparation_root_review.json。轻链3496/4000且原PID/start存活，评分未执行。原Goal完整范围保持active，上一/本轮progress；禁止任何重复启动或停线分支恢复。


最新 2026-10-06T23:12:31.503718+00:00：T4实际50082 CPU输入检查已通过（2例×32轮×扰动复算），GPU与完整评测尚未执行，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_cpu_preflight_root_review.json。轻链2768/4000且原PID/start存活；消融300–300步，原8项250验收完成、4排队、最终图10/22。下一轻链评分/T4 GPU准入与全量入口/消融500；原Goal active，所有安全停线/数据门禁/独立CUDA暂停保持。


最新 2026-10-06T23:07:56.130738+00:00：轻链2184/4000、原PID/start已核实，评分仍未开始；消融275–275步、8首250验证已通过、4排队、10/22完整。T4原生adapter已有tiny CPU验收与基准元数据，尚需真实50082输入检查和完整评测。当前证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/current_preparation_and_live_status.json；下一固定500消融、轻链完整退出后单次评分、T4实际适配。原Goal保持active，上轮/本轮progress，安全停线/审核/CUDA暂停不变。


最新 2026-10-06T22:59:41.079874+00:00：轻链1152/4000且原PID/start仍活跃；评分入口已准备，无评分产物。八卡消融225–225步、8运行/4排队，下一固定250验证。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/scorer_preparation_root_review.json。原Goal active，全部安全停线/审核门禁和独立CUDA暂停保持；下方旧时间戳是历史，不能据其重复启动。


最新 2026-10-06T22:54:53.864578+00:00：两轮/9CDR已完成；本地轻链全量预测正在运行PID339584/start2458869800，当前560/4000，实际CPU/GPU输入检查通过，评分未运行。原8卡消融175–175步、4排队、10/22完整。当前证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/actual_input_and_full_launch_root_review.json；下方早期状态按日期保留，不得重放任何启动。T4未适配、Goal仍active。


下一步核实12个消融的真实首批更新、失败和3750终态，完成原22格对比；不得以入队或进程存在代替完成。九项CDR结果已汇总并写入唯一RESULTS，训练阶段EVALUATION_READY中的pending是历史快照。本地f010/42已退出、无本地消融在跑。当前交接 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。

原Goal保持active，本轮progress；已停序列注释不得恢复/重试/转交/绕过，独立CUDA专项暂停，原数据审核/来源/恢复门禁保持。无测试权重复制，正式best3＋latest及必要证据保留。

2026-10-06T22:30:09.108912+00:00 新证据：八项新运行均完成step0初始验证，未将其计作训练更新；固定报告器接入通过，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/initial_actual_report.json。下一首25更新/固定250验证。当前仍8运行/4排队、10完整产物，原两轮/九CDR已完成。

2026-10-06T22:37:03.056870+00:00：八项消融已核实50步/1600条，4项排队，根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first_updates_root_observation.json。轻链独立输入/解码CPU准备通过，真实评测未执行，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/synthetic_root_review.json；后续仍需benchmark/来源门禁、原生50082输入检查与全量评分。T4未适配，完整Goal保持active。

<!-- CURRENT_EXECUTION_HANDOFF_END -->

## 2026-10-06T00:27:37.052546+00:00 用户速度问题：已核实步数预算和吞吐，当前训练继续

25042=ceil(6410749 / (32×8×1))，是约一轮自然数据的预算。查询时最新训练日志3675步，最近完整保存3000；这两个数字分开报告。最近两个千步区间为10.956/10.596秒每步（含验证/保存），名义吞吐23.37–24.16条/秒；若速率保持，整轮约73.7–76.2小时，从3675步约还需62.9–65.0小时。

当前8卡快照100%忙、26.59–28.51GiB显存，不能当峰值或算力效率。源码确认整层重计算、密集[B,H,L,L]链bias/局部mask、按batch最长padding及单进程loader；尚无组件耗时和实际SDPA backend证据，不能指定唯一瓶颈。保持原训练继续，后续按整模型计时与峰值检查优化重计算/attention/padding；独立CUDA专项继续暂停，不改当前冻结源码或既定batch/调度。

详细记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/SPEED_ANALYSIS.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/speed_diagnosis_20261006T002559Z.json（SHA8c3899842357341cbac9e3415875752ef49a91656b2e72d9f4e11e26923fbb81）。本轮没有新GPU任务、实际profiler或模型测试；记录脚本的JSON整数键读回断言已修正解释并核实原件，实际训练未受影响。


## 2026-10-06T00:20:55.802080+00:00 用户查询：正式训练最新日志为3650步，最近保存仍为3000步

本次只读查询核实八卡正式训练最新日志为3650/25042步（14.58%），原任务继续运行；日志每25步输出，两个日志点之间的瞬时步数没有直接观测。3000步是最近完整验证/保存点，不等于当前训练停在3000；下一常规验证/保存为4000步。

依据原生checkpoint-3000/trainer_state.json中的global_step=3000、logging_steps=25，在当前rank0日志中定位完全匹配的训练记录，并顺序核对之后26条训练日志。各条学习率都与该步的原固定cosine调度一致；未从epoch小数估算、未改训练日志/配置或新增模型运行。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/logged_step_observation_20261006T002055Z.json（SHAb5bec393e552ac449f3f65e97baaa41fb681a20165c7cc3041d20bfc57c8b044）。后续向用户报告时区分“最新训练日志步数”和“最近完整保存步数”，本段是带时刻的查询结果，不能冒充之后的即时步数。



最新状态以本页 CURRENT_EXECUTION_HANDOFF 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 顶部为准。旧 V4 blocked 审计保留为历史原件，不能代表现有 final V7、原 Goal 或当前在跑的真实数据实验。

共同500有界监督已于04:24:25实际节点自然结束；原任务继续，未改队列、重启资源或重复250验收。

<!-- HISTORICAL_EXECUTION_HANDOFF_20261003T043208_START -->
## 2026-10-03T04:32:08.720556+00:00 CURRENT：原Goal active，共同500真实证据已归档，独立接续模块准备中

原Goal threadId=01a0f245-4559-7b62-9dc2-0917a8ed2267，objective「继续explore，完成最后的验证后，提交我们的正式训练，因为我们目前的数据是已经处理好的完全可以启动8卡的训练」保持active；身份和累计用量未新建/重置/暂停。上一回合为progress：实际接入并一次登记完整22项，新首8已运行；本回合为progress：必要formal v2接续冻结、首250实际图及两seed派生已归档，并复核实际耗时/预算。不能把准备或部分实验当Goal完成。

| 工作 | 当前已核实事实 | 下一节点 |
|---|---|---|
| 原八卡资源 | t-20261001033327-fmnql最新直接get Running；原pool/八worker保持，当前新8具体ID实际running、心跳新鲜 | 按真实剩余预算，复用而不重启/重复提交 |
| 数据 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002；ready=true、限定范围双证书passed/pending=[]，三份SHA与实际门禁保持 | 原assert_ready及immune_test_only_observed_pair_exact_v3不改，不扩大证书结论 |
| 旧结果 | p01八项完整100k；原p02八项完整120k/3750、十四withdrawn；R20 AA CE0.825971，R26 0.812045低1.69% | 旧8/22图保持partial，不与新32×1拼表 |
| 新完整22消融 | 04:24:25首8共同500完整评估，每项16000条唯一曝光；8running/14pending、無失败或3750终态 | 真实8/22 partial，继续完整epoch/原22范围，不据早期CE选正式模型 |
| 新容量与启动 | 容量v2九格/40加载源及32档15%余量已核；新消费者4项和公共接入3项各一次passed，568模型保持 | 不重跑已验收项或原失败job；runtime原数据/来源/选择/恢复检查保持 |
| 正式八卡启动准备 | v3/611来源及parent_gate_v3 passed保持；新增formal v2薄接续及3项冷控制/root审阅/两源冻结完成 | 最终选型/配方/预算/全八卡空闲后创建正式计划与八票据；目前未发布/入队 |
| 正式联合与恢复 | 尚无真实新formal世界8联合更新/保存/原生恢复验收 | 八rank参数同步、完整native ckpt、optimizer/scheduler/本rank RNG返回点及实际续步/退出/清理全部必须实证 |

最新共同500原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/readonly_watch_v1/first_common500_observation_20261003T042425Z.json（SHA8ce69d9467587459d2bde40e53a54f9eac31f797e0524fbc88f9f3c43420d4f1）；原运行任务继续，500watch自然结束。500真实报告及12图独立留存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003，渲染receipt /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json；均值 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json。f000=1.179252、f010=1.105767、f001=1.063047、f011=1.043684；组合汇总收益11.50%较250收窄，全mask准确率低于基线，context仅37，不作生成或理解质量结论。

固定catalog /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json SHA987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d，plan /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json SHA42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04。当前public entry SHA c6540fc5663177c807fa7df3282a531279dc599a0c9f24632685fd18c5c8683c、registry SHA00cfc989f54645f0b1720eef435560dc64deff90e0c597fef6d9f38f20ba74d5；实际发布与一次登记 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/publication.json、enqueue_receipt.json。历史catalog queued_at_creation=false只表述创建时，实际状态读pool固定ID，不补回旧withdrawn或重放任何新job。

正式新启动准备清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2.json SHAe92de529db8b77f5c0e083ff115186c1419fbfc6c9a53ba61b386d5fde1bb9df，审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_v2_root_review.json SHAae9a7f7a2cad1a4c3ffd92834b1947453dfc31182f76f62c615ac48486d327ff，专属说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_V2_ENTRY_INTEGRATION_20261003.md。它仅接续large-v2→capacity-v2→旧链，原formal prepare/main/ticket与后续全部门禁不改；公共入口仍large-v2，formal catalog/选型/计划/八票据尚不存在。v3 source_manifest_v3.json SHA992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b，parent_gate_v3.json SHA3bb405020788470b823fe15ee49016143582ab42ee56ecbd3102716c96835122，不重复来源门禁。当前formal自然context约4.65%而筛选20.54%，正式配方待决，不能用pilot目录冒充parent。固定horizon两段票据steps为新增Δ≤5000、累计segment_steps为S或S+Δ。

预算原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common500.json，250→500实测8.806–9.000秒/步，代表未启动14项的条件三波预测26.29–26.87小时，04:24:25平台proxy估计余14.98小时；最近直接get真实Running，不计验证/保存/正式。默认原22→正式顺序保持，优先偏好未有新答复，不撤/缩短/改batch或重放。必要通用恢复适配器源码准备中：核心--resume已有、pool入口传递和outerCOMPLETE缺口已核实；无真实票据/发布/恢复，原保存事务不一致就拒绝，不补写绕过。

安全与工程边界：停止的序列注释不得恢复、重试、转交或绕过；独立CUDA专项暂停；所有原严格ready/审核/选择/来源/完整恢复门禁保持。现有容器、worker、监听和作业不重启、平台任务不重复提交。保留sole latest.pt/实际恢复状态，自有CPU临时fixture与ckpt为0。关键节点同回合同步PROJECT_PROCESS/BIOSEQ_MODEL_PLAN/PROJ_GUIDE及CURRENT，数据布局无变化时不虚报DATA_FORMAT改动。先读本CURRENT和PROJECT顶部，再读具体固定ID原件；下方旧blocked和旧未审核版本都是历史。
<!-- HISTORICAL_EXECUTION_HANDOFF_20261003T043208_END -->

以下带各自日期的章节是历史记录。

## 2026-10-02T11:26:57.045193+00:00 用户整体进度只读核实

截至2026-10-02T11:25:43.248933+00:00，首项CE R21/profile43的原500合成学习已完成并通过root原验收及独立复核；其余七项均425/500，原job身份一致、poll-gated heartbeat 0.06–1.16秒，无各自终态。原池pending0/running7/results66。八个单GPU验证与单模型八GPU共同训练须分开：后者尚未完成。

R20基础对照、R21注意力头门控、R24残差缩放、R26链内局部/全局注意力交替的候选代码已实现，最终选型未完成。两环境numeric八CPU Trainer保存恢复、八进程CPU通信/清理已通过；真实模型Trainer组装、原生词表保存恢复及实际world8 GPU-DDP尚未验收。当前子agent仅做有限只读依赖判断，没有实施或运行新GPU验证。

V7 manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失、audit_pending，当前数据审核和下游范围审核文件均缺；这不是操作权限问题，也不等于已证明数据泄漏。真实数据shard/小batch/语料训练0。

有限只读observer session90075已于11:21:44 UTC按900秒窗口自然结束，仅结束观察窗口，不能推断原七job退出；本轮只读快照确认原任务仍运行，没有新observer/监听、容器或worker重启、重复提交。原CE500及已完成CPU验收不重复，独立CUDA算子继续暂停；停止的序列注释分支不恢复、重试、转交或绕过，全部真实数据/来源/selection/学习/profile/GPU/完整恢复门禁保留。

当前原件：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/user_overall_progress_20261002_1125.json；已完成CE500验收：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json。后续先收尾原七项真实500终态及各自一次原验收，再补必要单模型八GPU工程；正式数据训练仍须当前有效审核与原完整准入条件。

以下带各自时间的记录为历史状态；本节及上述最新原件优先。


<!-- HISTORICAL_EXECUTION_HANDOFF_20261002T162850_START -->
## 2026-10-02T16:28:50.446971+00:00 Goal已标记blocked：当前V7审核证据持续缺失

连续三个Goal回合实际核实同一前置：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready未设True，audit_pending，两份当前数据与下游范围审核文件均缺。已请求团队提供已有审核结果，当前没有新结果或可核实替代输入。上一回合及本回合均no progress，非运行中任务的verified wait。Goal工具已返回blocked，保持当前目标与thread，未标complete、未新建目标。工具原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/real_data_goal_blocked_tool_result.json；三回合阻断链 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/real_data_goal_blocked_audit.json。

恢复条件：提供并核实覆盖当前V7清单及完整下游test/query/few-shot范围的真实审核结果，且不得恢复、重试、转交或绕过已停止的序列注释分支。真实数据架构比较、最终架构选择、ready置True与正式训练均尚未完成；工程与合成测试不能替代这些证据。现有V4私有修复/CPU验收/暂存计划保留，未新发布或运行。没有读真实分片、手改ready、伪造审核、重启资源或重复提交，也没有新增无关验证。取得新证据后需按实际内容重新确认可继续范围。

<!-- HISTORICAL_EXECUTION_HANDOFF_20261002T162850_END -->

## 2026-10-02T14:16:09.255833+00:00 上一交接块存档（历史）

## 2026-10-02T14:03:48.274562+00:00 V2固定共同八卡纯合成组一次实际入队

14:02:17 UTC通过原共享池发布既有固定8个g02 packet，group=original_r21_world8_1790948366803074817_36c1a3d5；启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_enqueue_receipt.json（SHA bc9285806e789f2124ac10db4b03ecc5f51112041f5526ecd4add555aa9cc647）。仅三份已独立复核暂存字节发布，原生545来源不变、12旧保护未改、两个public旧文件为明确已审阅变化，另新增V2 public dispatcher，runner最后发布。未使用随机ID enqueue、重stage、重放V1、重启容器/worker/监听或Volc提交取消。独立执行审查已完成，新synthetic-only启动记录绑定115实际前置原件；全部正式准入仍False，V7小batch/真实训练0，停止注释/独立CUDA暂停保持。

当前首次只读状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_fixed_group_status_after_enqueue.json：pool={'pending': 0, 'running': 0, 'results': 89}；具体8job是否被领取/真实group intent以原件为准，实际Trainer/DDP/保存恢复尚未验收，不能把发布入队当成功。一次尝试预算已消耗，不自动retry或再发布pending。有限监督subagent只读观察同8个具体任务至实际终态/有限窗口；观察超时不表示原job退出。真实产物复核通过且各rank退出后再清理自有临时合成checkpoint，原证据及实际训练恢复状态保留。

## 2026-10-02T13:48:54.596399+00:00 当前执行交接：V2 CPU/暂存已验，GPU尚未执行

截至 2026-10-02T13:47:01.415122+00:00，原八项单卡人工合成实验均完成500步并通过学习验收；R20/R21/R24/R26候选已实现，最终架构尚未选定，也没有真实下游提升结论。共同八卡最后实际尝试仍为V1启动锁超时失败、模型0步。

V2启动锁修复已完成；worker/pllm各4项CPU控制首次通过，原始结果与暂存的三个启动文件均经独立复核。Root只读重核641个实际输入（含完整545来源及14旧保护），接受范围仅为新增CPU事务边界及暂存字节；没有重跑控制、GPU发布或入队，不能据此称八卡训练/原生保存恢复通过。当前源码检查在锁外，成员先核对再提交，2秒锁等待及持久abort保持，初稿发现及V1失败原件保留。

13:47 UTC远端八worker均维护待命，新鲜心跳0.95–1.32秒；池pending0/running0/results81。平台任务t-20261001033327-fmnql最后实际CLI查询12:37 UTC为Running，此次未重新查询CLI，没有容器/worker/监听重启或Volc提交取消。自有CPU临时fixture及空父目录已清理，18个测试子进程已记录wait/退出，本次没有checkpoint。

V7仍为immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready字段缺失（null、非True），audit_pending，当前audit_certificate.json及downstream_scope_certificate.json均缺。缺审核不等于已证明泄漏；正式真实小batch及训练均未启动。

下一步在单独记录的执行阶段，先核对原资源和不可变前置，再使用固定的新八成员计划做至多一次纯合成共同八卡Trainer/原生完整保存恢复验收（连续3更新对比保存2更新后恢复至累计3更新，每rank合计6更新）；这些少量更新只验证工程一致性，不评估架构效果。本准备阶段不发布/入队，预算原禁止字段不改，V1已耗尽尝试不重放，已完成500/CPU/Gloo/FULL验收不重复。正式语料训练仍须真实当前数据审核和完整下游范围证据、原学习/profile/GPU/full-resume等全部门禁。

[最新根端证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_CPU_lock_and_stage_root_acceptance.json)（SHA 40c35a62c3389542162b75c225fa3f34ab676d461b592d4194def7bdb7502715）；[锁修复记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_GPU_LOCK_REPAIR_20261002.md)。停止的序列注释分支不得恢复、重试、转交或绕过；独立CUDA算子仍暂停。原Goal 01a0f245-4559-7b62-9dc2-0917a8ed2267继续active，不新建或重置。

读取顺序：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_CPU_lock_and_stage_root_acceptance.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/source_freeze_before_CPU.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_budget.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_GPU_LOCK_REPAIR_20261002.md。CPU/暂存复核已完成，不重跑；new group=original_r21_world8_1790948366803074817_36c1a3d5，八个固定g02 ID已存在于计划，但实际组目录/host intent不存在。准备阶段不publish/enqueue；后续新执行阶段先读取真实资源与来源/原500/profile/学习/full-resume前置，再构造真实generic启动检查，正式准入标志仍False。不要把代码已实现、CPU通过、暂存或enqueue当GPU/真实语料训练通过。原Goal累计来自 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/goal_actual_current_status.json；不新建Goal、不修改历史预算、失败或初稿问题原件。



## 2026-10-02T13:48:54.596399+00:00 先前交接块存档（历史状态）

## 2026-10-02T13:41:46.940748+00:00 V2锁控制两环境首轮各4/4，八成员启动文件已准备待复核

实际父进程session73874 rc0结束，两环境并行总4.9816秒；worker4.7145秒/pllm4.6881秒，各4项首次通过，未重试。每环境8个member CPU子进程+1个holder均实际rc0/wait/PGgone，临时fixture已删，源/14旧保护/545来源不变。2.25秒锁外有限慢检查期间实际获取锁仅worker0.001245秒/pllm0.001999秒；actual2秒超时worker2.009051秒/pllm2.008957秒，持久CPUabort与后续拒绝生效。实际来源字节改变拒绝、generation/source/snapshot三拒绝及纯CPU合法assignment提交均通过。每环境保留12份原始计时JSONL及命令/日志/hash；此结果只证新增CPU锁协议，不证明共同GPU/model/NCCL/save-resume已通过。原件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_first_execution_receipt.json，独立raw复核正在执行。

一次metadata-only stage于2026-10-02T13:39:26.830527+00:00实际rc0结束/2.2419秒，文件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/stage_v1/stage_metadata.json和/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json已生成：八g02成员/八新ID/six-update合成计划，无GPU主机/intent。stage字节独立复核正在执行；本准备turn不publish/入队GPU，预算仍保持其原禁止字段。最后一次已有共同GPU运行仍是V1 failed/模型0步；V2 CPU通过不改变该历史结果，旧500/CPU/Gloo/FULL不重跑。真实V7数据/小batch/训练未启动，双审核及全部安全停线保持。


## 2026-10-02T13:35:29.543004+00:00 V2锁修复完成，静态问题已修正，首次CPU控制准备

五个V2源已交付并冻结完整545个物理输入；当前bootstrap SHA2e81b0b8b3cfd439693e5b95faa64ac244861a961c2ef958443b6cb572fd856c，freeze SHA22a20b96a1e7c43cc0741f98ac458593483f2c77500790a5a552cd7345f36435。独立复核发现初稿20380把原post-publication完整检查移到commit前，root在未执行任何控制/源码/任务时仅补一行assignment_after_commit锁外完整检查，保留初稿字节、初始freeze e979及原finding报告于/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/initial_review_bytes；修正复核/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/bootstrap_postcommit_correction_independent_review.json确认唯一差异和原期限/接口/35旧函数保持。未删除问题记录或改V1失败预算。

成员状态pending→checked、八成员generation/source/snapshot两阶段提交、完整来源扫描/协调器初始化在锁外、真实硬件/父子身份与持久abort保持。三个非bootstrap训练模块函数体不变，dispatcher只必要路径/新g02身份重绑定。四项新stdlib CPU控制源码SHAe301ccf21b3c2935ea6158b0d8ae9c9becc925e87f2894a49eb91585f02c5274，已独立静态复核。Root物理核对545输入、14旧保护及四份review；执行计划/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_execution_plan.json，将各环境首次最多4项/180秒，当前尚未执行控制/模型/GPU/stage/发布/入队。正式V7训练仍0，全部安全停线/审核门禁保持。


## 2026-10-02T13:02:01.736603+00:00 V2最小锁协调修复准备

前一Goal回合实际核实V1八项终态失败、模型0步并定位长临界区，分类为progress；原Goal身份和累计保持。新有限预算/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_budget.json，SHA29eb09bf8449ba92101db33be7fe614d0a2f300799a4340e3a05e7b91989fa7c，保留V1已耗尽预算及全部失败原件。五个私有源已机械迁移到/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2；只有bootstrap将做通用锁行为修复，其余训练/保存/恢复逻辑保持，来源尚未冻结、未运行新控制。V2采用g02八个新成员名与后续全新jobIDs；准备阶段不stage/发布/入队，不重放g01失败job、重启旧资源或重复平台提交。

B只编辑V2bootstrap，C只准备一个stdlib定向CPU锁控制源码，A只读复核其余四源路径/预算重绑定；最多4项新增控制/环境/180秒，不重复原500、FULL/数学或已通过CPU/Gloo套件。真实数据门禁及停止注释、独立CUDA暂停保持。修复须在锁外完整核对来源、用两阶段确认防止未核对成员放行并记录实际锁耗时，不能简单增加时限或删除核对。当前八卡实测最新状态仍是V1启动失败、模型0步、worker回维护，V2源码准备不算八卡通过。


## 2026-10-02T12:55:34.324017+00:00 当前执行交接

截至 2026-10-02T12:54:01.376745+00:00，R20基础对照、R21注意力头门控、R24残差缩放、R26链内局部/全局注意力交替的候选已实现；原八项独立单GPU人工合成实验均完成500步并通过原学习验收。新八卡入口CPU检查在worker/pllm两环境分别4/4通过，已通过的检查不重复。这些证据不证明真实下游提升，最终架构尚未选定。

共同八GPU合成组 `original_r21_world8_1790944282201927112_f554453b` 的唯一V1尝试已终态失败：8个pool result均failed/rc1，直接错误 `Member lock wait expired`，只有3成员加入，assignment/startup未产生，8份member receipt均显示cold child未启动、清理完成。模型训练步数0，原生GPU保存恢复未验证，本次checkpoint目录0。此次失败发生在模型/NCCL启动之前；不撤销已有单卡500步通过记录，也不把源码已实现或入队当共同八卡成功。

只读源码诊断确认锁获取时限为2秒，join临界区内再次核对完整545来源/gate/plan并提交状态；这是需修正的长临界区设计。没有阶段耗时trace，不能认定某次SHA、I/O或调度是唯一根因，也不能仅凭锁文件存在称死锁。下一步保留V1源码、预算、计划、日志和失败原件，在新版本做最小通用协调修复：完整来源核对移至短事务外、两阶段状态绑定防止未核对成员放行、保持启动前完整核对和不可逆abort，并记录真实等待/持锁耗时。只补新增边界所需有限CPU锁竞争/来源变化拒绝/超时清理检查，再经来源复核安排有限纯合成验收；本状态回合没有重试、运行新测试或新入队。

原平台任务`t-20261001033327-fmnql`最后实际查询为12:37 UTC Running；12:54 UTC远端八worker metadata均standby_load/maintenance_only，心跳年龄1.39–2.62秒。池pending0/running0/results81。原八卡容器/worker/监听没有重启，没有Volc submit/cancel；pool成员失败不等于平台八卡容器失败。

V7 `immune_v7_mhc_sequence_v1` manifest SHA仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失（null，非True）、audit_pending，audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明泄漏；正式V7小batch和训练尚未启动，仍须当前manifest的真实数据审核及完整下游范围证据。当前只做通用工程与人工合成验收；安全停止的序列注释分支不得恢复/重试/转交/绕过，独立CUDA算子保持暂停。

[本次根端终态核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_overall_progress_after_group_failure_readonly.json)；[有限观察原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/group_bounded_watch_v1.json)；[原七项500验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original7_terminal_root_v1/original7_terminal_root_acceptance.json)；[入口CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/world8_new_boundary_controls_v1/cpu_boundary_root_acceptance.json)。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267` 仍active，身份、目标及累计不重置；实际工具快照见[Goal原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/goal_actual_user_status_after_group_failure.json)。

下一执行者先读/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_overall_progress_after_group_failure_readonly.json、锁事务诊断和原失败日志，再作新版本的最小修复。V1一次GPU尝试已耗尽，禁止原job重放及源冻结补改；已完成原500/CPU/Gloo验收不无据重复。




## 以下为历史阶段记录

更新：2026-10-01 05:52 UTC。v3 R20完整GPU工程验收及根端强复审通过，3/24/768；T20私有准备完成，数据仍未准入，正式训练尚未启动。

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



[最新整合证据与未完成项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。下一步完成D01真实终态核验和T20实际训练桥接，数据原准入及候选比较满足后再启动最终版本。两份数据证书仍缺，等待团队已有审核；此前停止的注释分支不恢复。以下带日期章节为历史阶段记录。

用户要求：持续依据监督subagent结果修正；必要时增加验证步数与覆盖范围；以现有LLaDA270为基础；最终进入初步阶段训练。继续使用已有c20250601单实例八卡非闲时资源，不重复申请单卡。原始数据、历史结果及其他会话任务不覆盖。

## 已核实的基线

旧no-ESMC真实checkpoint总参269,759,232，其中Transformer blocks为75,509,760，词表embedding/head共194,248,704。主干为d768、L8、H12、MLP3072。当前A4保留相同主干，50-token蛋白词表及角色/范围/链编码后总参75,601,176。“270M”不能直接解释为更深或更宽的主干。

已向用户询问是否要求保留大词表总参数或扩充有效主干。等待期间保持现有主干与蛋白词表，不擅自增加深度、宽度或加载旧checkpoint权重。原始从头训练路线保持。

## 验证与晋级

1. R12–R19的100步小型合成诊断完成并审阅。75步已经多次改变排序，多项接近此toy的可识别上限；它用于排错，不能直接决定架构。
2. 隔离发布R20–R27：各自保留上述机制及对照，采用已经CPU验证的bias复用实现。新source、新profile，不复用旧测量。
3. 扩展人工训练/heldout集合，覆盖长度、链组合、条件及无条件上下文。人工诊断可做500步，按分组和四档遮蔽观察稳定性；若曲线仍不稳定，再按证据延长或增加seed。人工语料训练始终明确标记synthetic，不计为本目标的初步生物语料训练。
4. 完成真实数据准备：复核旧H3缺口、新TCR/epitope/few-shot/AB-antigen范围与输入SHA，生成独立清理版本并独立复审。不能伪造证书、沿用未覆盖新范围的证书或按80%单链阈值直接删除原始数据。
5. 清理及工程条件齐备后，使用同global256、固定验证与学习率日程做1000步架构初筛；根据分层曲线及对照选择候选，记录依据，再累计到5000步初步训练。多seed和最终下游泛化仍单独报告。

## 持续监督与完成证据

[实时监督](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION.md)与[判读记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_SUPERVISION_FINDINGS.md)持续更新。异常处理为保留证据、隔离定位修复、CPU回归、干净GPU复测、新版本发布，不热改活跃实验。短期无需的checkpoint清理，保留有效续训依赖与小型审计记录。

当前目标完成须同时有：已整合且来源固定的预训练入口、必要候选学习曲线与明确阶段架构选择、有效数据原准入证据、最终版本训练实际启动与可核验进度、MD同步及验证产物清理。既定真实1000→累计5000是阶段选择计划；软件测试不能替代该阶段。当前未完成，数据认证和训练接入仍是前置条件。


## 2026-09-30T21:25:58.529062+00:00 八项500步人工诊断已开始

R20–R27的25步广输入检查全部succeeded，累计28成功/0失败。各项约718秒，四档mean loss为1.768–1.908、AA约30.06%–36.38%；end accuracy仍0，全mask end只占132/17996≈0.73%，须独立观察，不能仅看总loss。局部/全局方法在周期8人工任务上的优势不等于生物学优势。

已实际执行经74项测试验收的入队助手 synthetic500 --apply，8项均被worker领取运行。新500步initial的18组指标与同方法25步initial完全一致，fixture、39源码SHA、global256也一致，确认为同协议重新初始化。500步是更广人工heldout诊断，不是初步真实语料训练。按25步实耗粗估约4小时，准确耗时以运行结果为准。

[启动证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/synthetic500_started.json) · [配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md) · [持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。目标保持active，真实数据清理与1000/5000步尚未完成。

数据覆盖仍在推进：替代编号工具给出了VNAR候选，但与原ANARCI对部分“源空H3”案例存在分歧；184项只能表示来源/原编号已有解释，不能自动当成184项生物学缺失已确认或认证覆盖已完成。当时拟进一步核查；该序列注释分支随后被自动安全检查中断并已停止，未继续该筛查，覆盖保持pending。新expanded exporter已开始实现与CPU测试，尚未扫描/导出全量或签证。

## 2026-09-30T22:15:16+00:00 监督、评估及候选导出更新

同一非闲时八卡池继续运行，无新增平台 submit/cancel。R20–R27 已全部到100/500人工诊断检查点，39源码SHA匹配、无执行异常；50→75→100步排序多次变化。100步部分mask AA约96.25%–98.12%，conditional full-mask AA仍约31.97%–38.58%；R25/R26 end已明显改善，不能据此宣称真实生成质量或选定最终架构。[持续分层判读](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R20_R27_500STEP_FINDINGS.md)。

独立候选已物化：4,781,843条（train4,735,948/valid45,895）、81分片；138输入绑定、全部输出SHA/行数及6,901,815条新旧映射核验通过。原prepared和split保留，候选training_ready=false。固定验证子集1288条，TRAIT/native各4条/4family；实际assert_ready因缺audit_certificate拒绝。[实测记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)。

新增隔离软件验收：exporter58项、validation selection38项、attention随机数学/梯度38项；token评估原型26项分别在pllm及worker对应Python环境通过，未执行该原型GPU验证。评估补AA/end准确率、明确macro/micro/四mask均值，纠正汇总回调、随机流、批大小依赖；训练梯度保持父类一致。[评估协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)。人工采样概率与loss归一化诊断另完成6组验证，发现microbatch分组会改变当前训练loss权重；本轮8项均micro32，比较口径一致。[数学诊断](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_NOISE_OBJECTIVE_DIAGNOSTIC.md)。

自动安全检查中断了进一步序列注释子任务，该分支已停止，未重试或转交其他agent继续；尚未解决的覆盖项保留pending，等待团队已有人工审核信息。文件/索引验收不替代覆盖审核与独立重扫，未签证或启动真实1000/5000步训练。监督daemon和pool_monitor继续运行，goal保持active。测试临时目录已清理，没有新增测试checkpoint。

## 2026-09-30T23:06:15+00:00 200步监督与下一版本验收

本轮为实质进展：8项人工诊断到200/500、仍Running无错误；继续观察，不按175或200单点选优。新增END首次停止与relation兼容评估，联合适配器在两环境各90项CPU测试通过；来源曝光12项测试及两个现有采样器1.28M整数索引复核；每记录归一化loss的46项CPU梯度等价测试通过，均为隔离原型。当前39源码保持冻结。

CUDA后端探针首版47项CPU测试通过但没有实际GPU成绩，退出清理问题正在单独修复。当前Volc无exec接口；后续在500结束后，通过冻结旧release和受限诊断分派才可能在同一worker扩展执行，未热改或入队。[下一版本参数和未完成项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/README.md)。

监督daemon和subagent持续工作；目标仍active。实际候选仍被双证书门禁拒绝，自动检查中断的注释分支保持停止，未进入真实1000/5000步。软件验收与人工曲线均不满足本目标的最终完成条件。

23:20Z：八项已过225步。方法库补充[自条件/置信度/重新遮蔽](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DENOISING_METHOD_EXPANSION.md)的原文与固定作者代码阅读；三条候选未实现，避免与当前骨干对照混改。独立GPU探针的退出清理及loss原型的真实框架缩放继续在CPU验证，完成后记录结果再考虑新release。当前活跃训练源码不变。

## 2026-09-30T23:31:26+00:00 本轮收尾与后续依赖

本轮是实质进展：八项250/500仍正常，R20对照半程领先但没有最终选优；新增4项原文/作者代码阅读；评估v2与pilot工厂各环境90项、probe及真实进程清理各59项、record-mean数学46项、真实HF/Accelerate窗口各14项验收完成，相关临时文件清理。后两套环境各验证22次实际SGD更新，独立复审无阻断。上述新代码仍隔离，当前39源码未改。

下一步继续监督至500终态；将当前依赖冻结成旧release后，才能实现有界诊断分派，在已分配GPU验证kernel/数值及新评估路径。当前规划YAML没有执行器，真实数据未认证，不能自动跳过门禁进入1000步。goal仍active，真实1000→累计5000尚未完成；如果后续工程事项完成而数据审核仍无法推进，须按实际阻断条件记录，不能靠继续添加无关测试冒充目标达成。

## 2026-10-01 typed诊断、监督上线与pilot生命周期验收

本轮取得实质进展：八项人工350/500继续正常；typed入口各132项、监督32+联通3项、pilot生命周期各27项CPU验收完成，独立复审无阻断。仅本地监听更新到PID2033525，训练39源、容器和任务均未重启。测试checkpoint已清理。[IMMUNE_ARCHITECTURE_RESEARCH_INDEX](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_RESEARCH_INDEX.md)给出当前方法状态。

目标保持active。下一步完成500终态并冻结release，才能部署受限路由与实际GPU核验；真实1000→累计5000仍缺数据认证及新GPU证据。当前tiny累计续训属于软件测试，不计作目标达成。自动检查中断的注释分支继续停止。

00:22Z监督增量：八项375/500正常，R26条件full-AA99.69%、R20为95.50%；R24等方法仍有回退，继续固定500预算，真实语料准入条件保持未满足。

## 2026-10-01 00:47Z 425步监督与部署准备验收

本轮是实质推进：八项已至425/500且39源matched、无alerts；R21/R26/R24平均loss相近，继续完整尾段并列分组指标。监督daemon与pool_monitor继续，仍只有同一八卡平台任务。

[归档helper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_REFINED_RELEASE_ARCHIVE.md)两环境各22项、[固定路由补丁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DIAGNOSTIC_DISPATCH_STAGING.md)各38项CPU通过，临时目录3186文件已清理。归档实际dry-run被8running正确阻止，尚未写快照/map或部署路由，当前模型训练未改。下一步等待500原证据完整终态后执行既定部署顺序，不追加无关测试。

真实1000→累计5000仍未开始，数据双证书、新评估/pilot的来源绑定及独立GPU release尚未齐全。来源绑定缺口已写入规划YAML；旧profile不能被覆写来绕过检查，先前停止的序列注释分支保持停止。当前目标保持active，软件准备和人工曲线不计作最终训练目标达成。

## 2026-10-01 475步监督与实际GA错误修正

本轮取得实质进展：8项475/500继续正常；真实框架窗口审计发现并修正尚未部署Trainer跳过GA除法的问题，两环境各90回归及修正前后各5次人工更新通过。旧7份精确源码和原报告保留，1358个自有临时文件清理。当前GPU500已经手动除以accumulation，不受影响，39源未改。[复现与修正](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

派生runtime正在准备，后续必须从完整父快照、显式新增脚本和原scope资源路径建立可核验来源；GPU验收需真实训练/评估，不接受仅填cuda_execution布尔。候选元数据再次确认training_ready=false及缺双证书，仍未进入真实1000/5000。goal保持active；下一行动仍是当前500终态核验、稳定源码归档和已准备的受限CUDA诊断，不以CPU修复或人工曲线冒充最终完成。

## 2026-10-01 500步完成及实际归档

八项500步终态完整，根端原验证器和监督独立核验；原36份历史记录继续matched。524文件快照及8项map切换已落盘。当前新的部署隔离缺口正在修正，随后执行既有容器内D00；没有提交新平台任务。目标保持active，本轮有终态分析和真实归档进展；人工500与CPU软件验收仍不满足真实1000→累计5000的最终条件。

## 2026-10-02T14:49:51.778353+00:00 上一交接块存档

## 2026-10-02T14:16:09.255833+00:00 当前执行交接：V2已失败退出，旧尝试不可重放

截至 2026-10-02T14:14:14.929183+00:00，V2固定共同八卡组 `original_r21_world8_1790948366803074817_36c1a3d5` 已真实终态失败：8个pool result均failed/rc1，组与coordinator均aborted。新版本完成8个checked join、assignment、8个实际子进程注册和8份startup token领取，推进超过V1仅3成员加入的阶段；但没有Trainer目录、已完成训练步数或保存恢复成功证据，不能称八卡DDP/模型训练通过。八份实际子进程receipt均rc=-15、wait/PGgone/cleanup=true，8份rank退出及member cleanup齐全；自有临时checkpoint=0，未删除实际训练恢复状态。

直接错误是锁等待超时。实际父进程PID577559在group_summary持join.lock时等待coordinator.group_status的第二把锁，持join锁2.011217秒，commit仅0.002797秒；随后6个成员锁超时。更早的coordinator持久abort比member abort早约10.90ms。旧coordinator没有持锁计时，最初持锁者/具体慢操作仍未知，不认定死锁或某次I/O是唯一根因，不称完整545来源扫描重回join锁。16份实际timing JSONL及原日志/失败原件保留。

当前原池pending0/running0/results89，八worker新鲜维护待命。平台任务t-20261001033327-fmnql最后实际CLI查询13:54:41 UTC Running；没有新平台提交/取消、容器/worker/监听重启。V2一次执行预算已消耗，当前所有源码、预算、freeze、gate、固定8job IDs及失败原件保持，不自动重试或重放V1/V2；有限监督已真实结束，无观察进程待poll。

原八单卡500人工学习和已完成CPU验收仍保留通过记录；R20/R21/R24/R26已实现，但真实架构选型/理解与生成端下游收益尚未证明。正式V7小batch及训练仍0：training_ready字段缺失/null非True，audit_pending，两份当前数据/下游范围审核均缺。缺审核不是已证明数据泄漏，不伪造证书或降低门禁；停止的序列注释分支不恢复/重试/转交/绕过，独立CUDA算子保持暂停。

下一步在独立的新不可变通用工程修复版本，消除成员锁内等待协调器锁及重复高频状态查询，保留来源/实际host与父子身份/2秒时限/持久abort/退出清理门禁，并针对真实协调器竞争路径做有限必要CPU验证和耗时记录。首次coordinator竞争原因须保留不确定性；消除嵌套锁不能单凭静态推断当通过。不改模型数学或真实数据审核，不增加无关方法波次，不重跑已完成500/profile/FULL/Gloo或原CPU套件。

[根端真实终态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_GPU_group_failure_terminal_root_readonly_review.json)（SHA 2bb59dda1ad9a254362d6fce67b23a3fd80a0dfcbd782156c99d18504a1686ff）；[有限监督原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_group_bounded_readonly_watch.json)；[锁因果及边界复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_failure_lock_remaining_work_independent_readonly_review.json)。Root首轮只读字段误读source_check_state已保留错误原件，依实际JSON修正为source_phase；没有修改原validator/生产源或重试GPU。原Goal 01a0f245-4559-7b62-9dc2-0917a8ed2267仍active，上一准备回合及本执行回合均有实际progress；不新建或重置，实际累计原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/goal_actual_after_V2_terminal.json。

接手读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_GPU_group_failure_terminal_root_readonly_review.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_failure_lock_remaining_work_independent_readonly_review.json → /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_GPU_LOCK_REPAIR_20261002.md，先保留V1/V2消费预算及全部实际失败，再在新目录开始有审阅依据的必要通用锁修复。V2公开三目标仍为已审阅发布字节，两个旧before字节已存档，不得把旧public before SHA称当前不变。当前无live模型/observer/test session；旧入队session64337已经rc0结束，不可再poll或重启。实际平台任务与worker状态仍由工具/远端metadata核实，不能用root本地/proc推断远端进程。V7当前manifest SHA仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，真实shard/小batch/audit selection不执行。

## 2026-10-02T15:13:28.540259+00:00 上一交接块存档

## 2026-10-02T14:49:51.778353+00:00 V3启动锁修复已冻结，定向控制尚待首次执行

已完成六份私有源及独立静态复审；完整545输入冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/source_freeze_before_CPU.json（SHA546cec975bfc6fbfac9800ea5db21d073008fecee8ccf3197b80c81dcf6f879a），16份V2/公共保护文件保持。原Trainer driver与checkpoint validator所有函数AST不变；cold只去重状态轮询，bootstrap拆开两层锁；coordinator进程探测移至锁外并保留探测后的最终中止检查。初稿缺最终中止检查的问题及原字节均保留。实际source_metadata核对545输入已通过且没有ML导入，不把来源核对称作CPU并发/GPU验收。

唯一新CPU控制源码准备中，尚未执行，未stage/发布/入队新GPU组。14:35:34 UTC按原task ID查询平台仍Running；原池pending0/running0/results89，8 worker状态均以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/resource_readonly_during_repair.json 为准，不把历史heartbeat当实时存活。V2失败终态原件保持；没有新提交/取消/重启。上一状态答复只复述已有证据，本续接已实际改动通用工程并冻结源码，属于progress。下一步仅首次执行新锁竞争路径的有限CPU控制，不重复原500/profile/FULL/Gloo验收。真实V7训练0、双审核缺失，安全停线和独立CUDA暂停保持。


## 2026-10-02T15:37:05.010795+00:00 上一交接块存档

## 2026-10-02T15:13:28.540259+00:00 当前状态：V3锁修复的CPU验证通过，八卡启动文件已准备

启动失败已确认的链路是持有成员锁时等待协调器锁，触发2秒超时并连锁中止；首次协调器持锁慢操作仍未知。V3拆开成员/协调器锁、把进程探测移到锁外、保留探测后最终中止检查，父进程每0.25秒检查一次组状态并保留0.05秒退出/期限检查；模型数学、原生训练/保存/恢复函数不变。

两环境首次各3/3必要CPU控制通过：pllm8.1196秒、worker7.7225秒。2.25秒显式慢探测期间，另一状态查询耗时分别0.0365/0.0455秒；协调锁等待期间成员锁获取0.00143/0.00135秒。纯观察约2.001秒超时未写abort，真实变更约2.005秒超时写入持久abort，后续变更拒绝；八CPU进程真实注册/启动/退出成功。22个自有子进程全部rc0/wait/进程组消失，临时目录已删。每环境保留15份原始timing内容和实际命令/日志，未创建checkpoint。控制3验证共享snapshot原语而非伪造GPU入口；这些CPU结果不能称真实GPU/NCCL/Trainer已通过。

根端逐值/原件复核：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/CPU_regression_root_review.json。完整545来源冻结SHA546cec975bfc6fbfac9800ea5db21d073008fecee8ccf3197b80c81dcf6f879a，16份前驱/公共文件未变。新三份启动文件仅暂存于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_v1；固定group=original_r21_world8_1790953672752409704_de857f5f、八g03 jobID位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/group_launch_plan.json，plan SHA234d6ec48d10143074564cf00baca77016c87c33116642f4cf1f4d5de4f3ac5d。根端与独立agent已核对registry仅八别名、runner仅一个路径、dispatcher与冻结源字节相同，记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/stage_independent_review.json。未发布、未入队，实际group目录/host intent/launch gate均未创建。

准备阶段完成证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/preparation_completion.json（SHA0186617b44a5e6e5154a08493fbe8280aff63f5d07d6db98f54d59e72679b08f）。下一步在独立执行阶段复核原8 worker及原学习/profile/来源/恢复前置，发布已审阅暂存字节，仅将现有八固定ID一次入队做synthetic Trainer联调；不要重stage/生成ID、重跑已过CPU/500/FULL/Gloo或重放已耗尽V1/V2。原资源最后CLI14:35:34 UTC Running；没有提交/取消/容器或监听重启。

V7 manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready缺失非True、audit_pending、两份审核仍缺，正式小batch和训练0。缺审核不代表已证实泄漏，也不是权限问题。停止注释分支不恢复/重试/转交/绕过，独立CUDA算子暂停，真实数据门禁不变。原Goal仍active且身份/累计不重置，本回合为实际progress，未完成最终训练目标；Goal实际工具记录见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/goal_actual_after_CPU.json。


## 已归档的V4工程准备交接块（非当前目标进展）

## 2026-10-02T16:18:58.824015+00:00 V4监督修复两环境首次验证通过，八卡启动文件已暂存

V4实际生产入口已调用同一supervise_process_observations，仅对来源模块/组/contract/operation绑定的只读group_status超时有界继续观察。原2秒锁等待、30秒健康窗口、启动/总期限与持久中止/退出检查保持；原模型、Trainer和checkpoint函数未改。独立源码静态复核无必须修正项，root完成唯一新控制草稿的必要断言后首次执行。

pllm47.2566秒、worker47.3272秒，各两项通过：真实八CPU子进程在实际2秒状态锁超时后全部仍活，恢复后同一Popen正常退出、八份原协调器退出记录齐全；不可观测时健康时间未刷新。六个拒绝分支覆盖持久中止、非专用错误、无首份有效状态、启动期限、总期限与实际child rc7。真实无首份状态分别31.2912/31.2901秒拒绝，含单次在途锁等待，不能称精确30秒硬实时；启动/总期限使用明确缩短的CPU参数，不声称等过180/1800秒GPU期限。34自有子进程均wait/PGgone，临时目录已删；每环境65份原始元数据/计时文件保留，未创建checkpoint。

根端结果逐值/来源/退出核实：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_supervisor_root_review.json（SHA 9d9288e9de86e5bc2b767a4943c855db0d878093b934553868a8947d7792ac21）。完整545来源与16保护未变，冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/source_freeze_before_CPU.json（SHA2123626cf035d387173b8729fdd4c91596a0cdd29ebf202891d60863759fc7af）。没有重跑旧CPU/500/profile/FULL/Gloo实验。

三份启动文件已暂存但未发布：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/stage_v1/stage_metadata.json；固定新八成员计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/group_launch_plan.json（SHAb5a5a87ab7a64632c33f212f0ce1602580f7df397cc04a6a0f1ec8b5edcbccec），group original_r21_world8_1790957765992952124_ba1f12c5。Root确认旧public保持、dispatcher与冻结源相同；独立只读复核已完成并接受，见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_and_stage_independent_review.json。actual group/host intent/launch gate尚未创建，GPU attempts0。下一步仅在完成当前前置与暂存复核后，于原容器执行一次八卡纯合成Trainer/保存恢复验收；不重新stage/生成ID、不重放V1/V2/V3。

最新实际GPU仍是V3八注册后失败并完成八子进程清理，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/actual_GPU_terminal_root_review.json。CPU修复通过不等于GPU/NCCL/Trainer恢复通过，真实V7小batch/训练仍0，当前审核门禁与停止注释分支/独立CUDA暂停保持。最后平台实际查询 2026-10-02T15:58:02.091075+00:00 原task Running、八worker维护待命、池0/0/97；无新平台提交取消或容器/worker/监听重启。原Goal身份目标累计保留并active，本回合有源码/实际验证进展，未达到最终训练目标；实际Goal工具快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/goal_actual_after_CPU.json。


## 2026-10-02T16:21:33.119075+00:00 V4准备完成，独立复核接受；尚未启动新的八卡组

独立复核实际结果与暂存字节通过，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_and_stage_independent_review.json。证据限制明确：持久中止测试是在group_status超时已抛出后、共享helper catch前发布abort，证明catch内中止检查生效，不声称在锁等待过程中发布。没有因此改源或重跑。准备阶段完成记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/preparation_completion.json 仅指CPU修复和启动文件准备，原Goal仍active未完成。

当前只读V7清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/V7_manifest_only_readonly_status.json：manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready=None，两份审核文件存在性分别False/False。未读真实分片或启动真实训练，门禁与安全停线保持。下一执行阶段使用已有固定g04计划，至多一次原容器纯合成八卡Trainer保存恢复验收；不重stage、不生成另一组ID、不重启或平台重复提交。
