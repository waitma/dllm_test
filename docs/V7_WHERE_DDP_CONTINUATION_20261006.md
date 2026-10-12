# V7 原八卡DDP代码切换与速度验收

## 2026-10-06T13:05:08.199910+00:00 正式9000步保存与验证已核实，持续到9303

where-DDP实际9303/25042，当前段目标12016；监督PID212174/start2455180199和本地PID80642/start2452750909仍在，八卡25次有界观察均有新鲜正在执行心跳、无abort。9000原生八rank保存、17文件/JSON归档、源码绑定、112固定计数、70核心聚合和26结束指标核对通过，实际检查会话70352退出0（c33975）。模型和验证集合未重跑。新工具 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_where_checkpoint.py 已对真实9000点执行通过；后续仅核查新产生的点，不重复旧验收。段内保存验收不等于12016段退出验收。

| 指标 | 8000步 | 9000步 |
|---|---:|---:|
| 主eval_loss | 1.889376 | 1.890182 |
| 平均AA CE | 0.732485 | 0.734610 |
| 全遮蔽AA CE | 1.723888 | 1.740789 |
| 全遮蔽END准确率(%) | 55.682951 | 53.688933 |
| 首END准确率(%) | 17.497507 | 16.400798 |
| 未预测出END比例(%) | 1.994018 | 1.246261 |
| 长度MAE | 2.820448 | 3.249369 |
| 长度MAE分母 | 1966.000000 | 1981.000000 |

主eval_loss和AA CE略回升；END CE下降、漏检减少，但END准确率、首END、长度MAE及误报有回退，不能宣布整体质量提升。MAE分母1966→1981随漏检改变，不能脱离分母解读。继续原f011/25042日程与后续验证，不因一个点改架构。

best3 7000/7008/7016＋latest9000实际共4份完整恢复状态；8000权重已由训练策略删除，保留前JSON证据仍可读取，9000核查成功复用了8000归档，未因删除权重丢失比较依据。旧5016/6000及自有临时验证checkpoint保持已清理；不额外删除保留恢复点。根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint9000_root_review.json（SHA85df3e257935ccb5506df67b38ad5d34be246ddb4a879ef949a3eec02dadc2d1）。

新增已完成1000更新墙钟窗口8001→9001为722.879864秒，平均0.722880秒/步，包含一次完整常规验证/保存/源码检查/保留轮换；8002–9000的999个普通更新均值0.713420秒，中位数0.763121秒。9000非训练边界约9.44秒；此窗口没有段启动或恢复，不能当完整epoch ETA。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。此前300步0.740862秒的证据保留，不覆盖原统计。

本地f010/42进度1600/3750、51200/120000条；最近已验收1500，下一1750，无真实退出回执。22格仍9完整epoch产物、1运行、12未启动，500图10/22、3750图9/22均PARTIAL。只读nvidia-smi显示本地GPU利用率100%，其中PID为设备/宿主侧视图，未据此映射或干预未知进程。

有界观察63484已于13:01:31 UTC正常退出0（220e9c），25次从8505观察到9000，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/root_watch_to9000_20261006T125530Z.jsonl（SHA88f955363dec278284d4902449df5379112021c737fea57dc625bfae72f0afbe）；结束的是观察窗口，训练/持续监督未停止。上一Goal回合和本回合均progress；原Goal身份/累计用量及25042＋22格完整范围保持，未完成。下一正式10000、本地1750；独立CUDA暂停、序列注释停线、原数据审核门禁保持，未提交/重启容器或任务。

## 2026-10-06T12:52:51.652079+00:00 新版常规8000步验证/保存通过，继续训练到8278

同一八卡where-DDP仍运行，当前8278/25042、当前段目标12016，总目标不变；新监督PID212174/start2455180199和8个正在执行的worker心跳再次核实，无abort。8000完整17文件、八rank参数与保存模型一致、六JSON归档与SHA、112固定验证计数、70核心聚合和26结束指标核对通过；固定1330条×4遮蔽档保持。此为段内checkpoint验收，整段7016→12016尚未结束；最近通过退出验收的段仍7016。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/checkpoint8000_root_review.json。

| 指标 | 旧代码7000 | 新代码8000 |
|---|---:|---:|
| 主eval_loss | 1.884267 | 1.889376 |
| 平均AA CE | 0.736503 | 0.732485 |
| 全遮蔽AA CE | 1.728628 | 1.723888 |
| 全遮蔽END准确率(%) | 56.331007 | 55.682951 |
| 首END准确率(%) | 16.301097 | 17.497507 |
| 未预测出END比例(%) | 0.049850 | 1.994018 |
| 长度MAE（有END的链） | 3.438404 | 2.820448 |

AA CE略改善，END准确率/漏检有回退，首END和有END链的长度MAE改善。长度MAE的比较分母随漏检变化，不能只据该数值判断生成长度变好；整体主eval_loss略高，不据单点宣布效果提高或更改架构。此表跨显式where代码迁移边界、相隔1000训练更新，并非同step效果对照。

best3＋latest实际保留7000、7008、7016、8000四份完整状态：前三份为主eval_loss最低，8000是最新；本次未删除这四份之一。原5016/6000已在先前轮换删除，JSON证据保留。保存完成后已观察到8001及更后训练计时，运行继续。下一常规验证/保存9000。300步稳态均值0.740862秒（旧10.199–10.696秒，约13.77–14.44倍）的统计范围不变，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json；不把验证保存耗时混为训练步时。

正式当前输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/run；持续监督 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/supervision_state.json。新独立2×4 FSDP的best3版本已完成CPU元数据/源码/CLI检查，未提交，文件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml。旧容器未重建、未重复提交；独立CUDA暂停、序列注释停线与原数据审核门禁保持。原Goal身份和累计用量保持，25042正式训练与22格消融尚未全部完成。

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


## 2026-10-06T12:21:04.640757+00:00 where同DDP接续的停止前检查已通过

8项best3+latest临时目录控制通过（9466e9 rc0）；CPU旧模型checkpoint→where模型原生恢复通过（6a4a49 rc0），恢复optimizer/scheduler/RNG、最终step4与全部optimizer step4，旧checkpoint未变，较连续旧实现最大参数差7.64e-11。首次CPU测试因独立导入模块缺少包名失败于模型构造前，改为正确包内名称后通过，两个日志保留。自有临时checkpoint均已清理。

613个候选源码绑定、原只读父数据审核、允许的模型单表达式变化和8种不兼容元数据拒绝检查通过；原checkpoint7000全部17个native文件重新核验，仍可完整恢复。预停记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/pre_stop_review.json；新冻结源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/source_snapshot_v1；原冻结源不改。新8步接续plan已准备未入队，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches/segment_7008/plan.json。

接下来按用户明确授权停止旧训练子进程并在同一容器由7000→7008接续；平台容器保留。当前没有立即保存接口，7000之后未保存更新将从该完整恢复点重算，实际停止日志步数会单独记录。全部八rank退出清理证据齐全才允许新8票入队。此处CPU通过不等于新八卡恢复或吞吐通过。保存策略与新旧来源迁移保持显式记录；原安全停线和审核门禁不变。

本文件后续记录实际停止、首次八rank恢复、best3+latest保留结果和独立稳态速度；未完成的步骤不写为通过。
