# Training Speed Analysis（current immune LLaDA line）

## 2026-10-06T22:21:08.237756+00:00 恢复后完整5000步：包含恢复、验证、保存与退出的平均0.750700秒/步

45058→50058实际入队至验收3753.498933秒，5000次连续真实更新，平均0.750699787秒/步；global256名义341.015条曝光/秒。八rank恢复/保存/退出通过，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50058_root_review.json（SHA14301194f8e75fddc96cfb7e6dee613b44864e0ab932e8d9a5ce410033682f05）。这是实际生产长段的整段口径，不是重复benchmark；与此前300个训练循环的0.721418秒范围不同，不能混当同口径加速。模型/批量/math attention及原8GPU未改；最后24更新的冷恢复摊销也不得解释成稳态吞吐。




## 2026-10-06T21:24:31.178216+00:00 恢复后真实300步速度核查：0.721418秒/步，八卡继续

新长段45069–45368共300个实际更新（省去最初10步）平均0.721418251秒、中位0.772892173秒、p95 nearest-rank为1.012516931秒；global256名义355条曝光/秒。scope是rank0训练循环墙钟，包含数据与DDP，不含验证/保存/冷恢复/退出；所有first_or_after_save=false，完整行和来源前缀哈希保留。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/speed_first300_updates_root_review.json（SHA8b4fffa7527253fb7ce59802ec840f55877128e834931495d062a57720120687，0879c0 rc0）。此前同口径窗口均值0.735977504秒；新旧样本/LR不同，只能说明当前观察未见速度回退，不能据此声明受控1.98%加速。

实际当前45549/50082，45058→50058正常、abort=null；两新监督身份存活、8worker新鲜running。最新完整保存/验证/恢复45058已核查，下一46000。新生成监督继续等待固定50082和真实退出，未发布或推理。本地f010/42为3550/3750、验证3500，下一3750。原22格/完整图、9项真实生成及两轮终态仍未完成，Goal active；未重复既有验收或启动额外测速/CUDA专项。原失败段/STOP/安全门禁保持。


## 2026-10-06T21:22:17.810232+00:00 原生恢复已通过，首8步不作稳态吞吐比较

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment45058_root_review.json 的whole_segment_timing为新45050→45058入队至8rank验收140.671275秒，含冷初始化、原生恢复、评估、保存和退出。8步的分摊值17.583909秒/更新不能与此前5000步长段0.754369秒/更新比较。当前长段45058→50058已产生新更新，继续用实际step_timing观察，不新增测速模型或CUDA专项。

## 2026-10-06T20:45:09.009724+00:00 45050之前整段速度稳定；下一段启动锁超时已停

40050→45050整段3771.846267秒/5000更新，平均0.7543692534秒/更新；上一段0.7541745738秒/更新，均含恢复/验证/保存/八rank退出。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment45050_root_review.json 的whole_segment_timing。无额外测速；不能把随后启动失败计入正常训练步耗时，也不能据健康段吞吐声称新段仍运行。

下一50050段启动时的协调锁超时已导致8个训练child清理、训练/生成监督rc1退出，当前最后完整45050。实际heartbeat持锁2.364763/2.350461秒，最长一次提交阶段2.362406秒，超过2秒锁等待阈值；具体文件操作/调度延迟尚未定位。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment50050_lock_diagnosis.json，工程记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_STARTUP_LOCK_FAILURE_20261006.md。原失败标记保留，未重启或重试。

## 2026-10-06T19:36:40.075718+00:00 第二轮第三个5000步段：含恢复/验证/保存/退出平均0.754175秒/步

35050→40050整段3770.872869秒（62分50.87秒），平均0.7541745738秒/更新；上一段0.7540041532秒/更新。原f011、DDP/global256、同一八卡和冻结日程保持；实际8rank退出/清理和24项恢复核查通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment40050_root_review.json 的 whole_segment_timing（SHAfa75e09146c9f05710bf242588272b4f8f2734923f77f8c1973c2c33f6e59395）。未新增测速实验，不作为跨数据顺序/架构的受控比较或生成质量结论。原监督已单次接续45050，累计两轮目标50082。

## 2026-10-06T18:34:57.137689+00:00 第二轮第二个5000步段：含恢复/验证/保存/退出平均0.754004秒/步

30050→35050整段3770.020766秒（62分50.02秒），平均0.7540041532秒/更新；前一5000段0.7519159536秒/更新。原f011、DDP/global256、原八卡与冻结日程保持，实际8rank恢复/保存/退出核查通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment35050_root_review.json 的 whole_segment_timing；未新建测速任务，不把跨epoch/数据顺序的观测当作受控架构比较。原监督已接续40050，累计两轮目标50082。

## 2026-10-06T17:32:33.406977+00:00 第二轮完整5000更新：含恢复/验证/保存/退出平均0.751916秒/步

原八卡f011、DDP/global256，25050→30050；开始enqueue=2026-10-06T16:26:37.018819+00:00，实际8rank退出验收=2026-10-06T17:29:16.598587+00:00，耗时3759.579768秒（62分39.58秒），均值0.7519159536秒/更新。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/segment30050_root_review.json 的 whole_segment_timing；8个实际退出与24项原生恢复核查通过，未新建测速任务。与先前父阶段完整5000段约0.752/0.754秒/步相近，支持提速在更长实际运行中保持；这不是不同架构/epoch的受控性能比较，也不表示下游生成质量提升。下一段已由现有监督接续35050，累计目标50082。


## 2026-10-06T16:38:02.720689+00:00 第二轮300步实际速度与新阶段只读核查入口

新阶段25061–25360共300个实际训练间隔，跳过30050段最初10个更新、无保存间隔；平均0.735977504秒/更新，中位数0.782279909、P95最近秩1.037570808，global256对应名义347.84条曝光/秒。来自现有step_timing日志，包含数据和DDP训练循环，排除初始化/恢复/验证/保存/退出；不是额外测速任务或不同epoch之间的受控对照。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/speed_first300_updates_root_review.json（SHA1acfea10525de7949581c59828c063aff6bacf035706ddd68fa569730220d791，72c674 rc0），保留完整读取前缀长度/哈希及选中行哈希，无重复大日志。当前正式25835/50082、25050→30050继续。




## 2026-10-06T15:48:10.321315+00:00 新5000更新完整段速度保持

17016→22016实际入队2026-10-06T14:41:17.746781Z至8rank验收15:44:08.778170Z，共3771.031389秒/5000更新，平均0.7542062778秒/更新（62分51秒）。含初始化、原生恢复、验证、保存、8rank退出与监督验收；上一12016→17016窗口0.752044206秒/更新。两段同where-DDP、micro32/GA1/world8/global256，速度相近；不作CUDA算子归因或整轮耗时保证，无新测速实验。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment22016_root_review.json 的whole_segment_timing字段，SHA9e12b444fb0b964fe13f6be93d55be3f1863db5af4171088dcca6840da2bc103。原任务已接续22016→25042，当前22185步，之后仍按累计两轮50082安排。



## 2026-10-06T13:05:08.199910+00:00 新增1000更新真实八卡墙钟窗口

8001→9001实际722.879864秒，0.722880秒/更新，包含一次常规验证/保存/源码检查/轮换；8002–9000的999个普通更新均值0.713420秒，中位数0.763121秒。9000非训练边界约9.44秒，不含段启动恢复；不可将该窗口当完整epoch或FSDP性能。未运行新的模型测试，仅读取现有逐步计时。原300步0.740862秒窗口保持；新原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_after_checkpoint9000_root_review.json。正式9000保存与验证已核实并继续训练，质量指标有升降，详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。

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

## 2026-10-06T11:35:49+00:00 根因已用本地完整模型GPU验证，工作区修复通过

主瓶颈为chain_bias重复索引反向。真实V7 B32×S867的profile整步9.5329秒，indexing_backward_kernel_small_stride单个kernel8.8094秒，占92.41%。仅where广播替代索引展开，8个真实batch整步均值6.6288→0.5450秒，12.16倍；参数、mask、目标及math SDPA保持。固定S516单换efficient只改善1.062倍，修正此前将后端作为最高嫌疑的判断。

本地同卡occupy.py开/关对照为3.8562/2.1404秒；测试后两个原进程已恢复。工作区RefinedModel已修复，75.6M GPU首步loss一致、78组梯度通过，64项CPU回归通过，无测试ckpt。原冻结任务未切换，八卡最终吞吐未验收。

报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SPEED_ROOT_CAUSE_20261006.md；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/local_verified_profile/verified_summary.json。下方早期“GPU未执行”保留为历史阶段，当前以实测报告为准。

## 2026-10-06T10:17:16.991407+00:00 V7/V5调试：普通训练区间仍为10.2–10.7秒/更新

同global256，实际V5约2.73秒/更新，V7保存区间约10.93–11.17秒/更新。新增已有监督日志重算：避开验证/保存/接段边界后，5475→5875约10.20秒/更新、6125→6850约10.70秒/更新；记录/轮询有端点误差，但慢化持续存在，不能主要归因保存。

已核实当前torch2.8.0及7份冻结源码SHA。旧V5和当前均关闭memory-efficient SDPA；当前可训练dense bias又被2.8 Flash/cuDNN选择器拒绝，故源码条件指向math回退，尚无GPU kernel实测。local/global虽复用FP32 mask，各层仍重复转BF16并扫描负无穷，whole-layer重计算再次执行；链内RoPE和component日志存在GPU→host同步。局部attention仍为稠密计算，75.6M参数减少不代表主干8层/d768缩小。

既有1024条长度重算：原micro32比同顺序micro4多23.59% padding后位置、58.80%稠密attention元素，仅为同一V7样本的几何量，不是V5真实长度或加速实测。2048是上限，不能按统一padding解释。

完整依据及修复验证次序见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_V5_CODE_COMPARISON_20261006.md；原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/v5_code_review_v1/steady_windows_and_geometry.json。本轮仅标准库元数据重算和源码核对，无新增模型/GPU测试、权重或训练修改；GPU计时仍未完成。




## 2026-10-06T04:31:36.542828+00:00 正式5000步验证与保存完成：重建CE和真实END准确率改善，首END与误报仍分化

新增保存事件区间4000→5000为10926.702977秒，即10.9267秒/步，包含验证/保存及最多一次观察间隔，不能当GPU纯计算耗时或归因DDP/loader/占卡。正式单机8卡DDP配置保持；新节点的重建与结束位置指标方向分化，详细结果见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint5000_root_observation.json（SHA4e51f7c8d2006dd5221d548aa6cdd04ea1ab2ca36b1245de6ab846a5b0e99b64）。整模型GPU计时仍未执行，独立CUDA专项继续暂停；未重启训练或重试已拒绝的远端进程查询。


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


## 2026-10-06T01:00:12.404213+00:00 整模型计时入口已准备并完成CPU/CLI检查，GPU测速未运行

用户要求继续定位正式训练慢因。本轮在原八卡/本地任务继续运行时完成了private整模型计时入口与审阅，不改变训练。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/profile_whole_model.py（SHA581976199c87c8faa98199bec3924e44dec8123a665322ef7a9c83928ecf2db9），说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/README.md；根端验收范围原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/root_preparation_review.json（SHAe9b0c664736fccf938ccbe2949f6965dda7f063cc5189b18697dfcb36c6bb1f7）。

已实现f000/f011两组同主干、同人工形状、同seed、FP32参数/BF16 autocast、whole-layer checkpointing、原forward/noise/CE、真实backward/clip/AdamW。默认5个预热更新、30个按阶段同步计时、10个仅整步边界同步计时、1个独立完整更新profiler，共46个更新，父进程900秒上限；记录真实child exit，不自动重试，不保存模型/optimizer权重。额外检查链bias梯度存在并记录范数，profiler保留实际SDPA、index/scatter等事件，不预设瓶颈。

根端审查后已修复两点：仅查顶层模型路径不足，改为拒绝混合项目模块缓存、过滤工作树sys.path、逐个检查实际加载的dllm/scripts/examples文件与namespace路径并匹配既有冻结manifest；分阶段同步会扰动吞吐，故`instrumented_*`与`whole_step_boundary_only`分开，后者仍保留原模型/optimizer内部同步和诊断开销，不冒称正式Trainer/DDP吞吐。未重复扫描完整611源验收。

CPU/CLI证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/preparation_validation.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_speed_diagnosis_20261006/whole_model_profile_preparation/preparation_validation_v2.json（最终SHAc4d46ae6b88f1507022f68bcaee8124cc523cae569f21b49419e1975027f37c2）：原三种人工布局检查实际exit0且未初始化CUDA/模型；新增源码隔离、AST、f000/f011计划与更新预算拒绝检查通过。根端读取原件与最终代码SHA并复核接口/预算，不重复已完成CPU布局测试。没有新的GPU运行、真实数据读取、独立CUDA算子测试或测试ckpt。

当前状态必须分开：**代码准备与CPU/CLI检查完成；GPU执行/吞吐/实际backend/OOM容量尚未验收；普通V50 LLaDA control、真实V7批次回放与八卡分项时间仍待完成。** 人工512个target均有监督，不能代替正式空槽/关系/context配方或实际训练数据实验，不能纳入架构效果消融排名。入口只支持明确独占的空闲单卡，父子进程均先拒绝已有compute PID；该快照不是资源预约。本轮没有发布pool执行入口、提交/取消任务、暂停/重启原训练或新增占卡负载。

原正式训练继续5016段与完整25042目标，本地f000继续3750步完整epoch。无新验证节点不重复旧验收/绘图，完整22格仍8完成、1运行、13未启动。Goal保持active；后续优先采集正式4000和本地2750新节点，在原授权及资源允许的训练边界执行整模型计时。保留正式恢复状态；本次仅代码/小型JSON/MD证据，无测试权重需删除。原序列注释停线、审核门禁和独立CUDA暂停保持。


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


## 2026-10-06T00:27:37.052546+00:00 V7 正式训练：25042步预算与当前速度

当前实际训练是no-ESMC f011，8张A100、每卡micro32、GA1，全局batch256；约一轮的数据预算为ceil(6410749 / 256)=25042步。这是既定预算，不是架构要求的固定步数；drop_last与分布式epoch尾部仍须按实际曝光解释，不能声称每条记录恰好一次。此查询时最新训练日志3675/25042（14.68%），最近完整验证/保存仍为3000，下一4000；日志每25步更新，不把保存点当作实时训练步数。

| 实测区间 | 完整保存间隔 | 平均秒/步（含验证、保存与观察延迟） | 名义全局记录/秒 |
|---|---:|---:|---:|
| 1000→2000 | 10956.2427秒 | 10.956243 | 23.365674 |
| 2000→3000 | 10596.196886秒 | 10.596197 | 24.159611 |

按上述速率保持不变估算，完整25042步约73.7–76.2小时；从本次日志3675步起还需约62.9–65.0小时。它是基于两个实际区间的条件估算，包含常规开销，不含未来异常、暂停或优化收益；不是纯GPU kernel计时，也不是与旧ESMC配置的受控速度比较。

当前GPU快照8卡均100%利用率，显存26.59–28.51GiB/80GiB。利用率说明采样时GPU忙，不等于算力利用率100%；快照显存也不能代替训练峰值。已读取实际f011调用链，确认以下实现事实：

- driver启用whole_layer activation checkpointing，反向会重算前向，存在额外计算成本。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/scripts/research/run_v7_formal.py。
- 链关系bias为[B,H,L,L]密集张量；R26局部层也通过密集mask限制可见位置，没有改成稀疏/窗口计算。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_refined/model.py 及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/research_next/model.py。
- ArchitectureBlock传入非空attention_bias，模型中显式flash_attn分支要求attn_mask=None，因此走PyTorch SDPA调用。尚未采集其实际底层backend，不能断言必为math fallback或已使用FlashAttention。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/llada/models/modeling_llada.py。
- collator按当前batch最长序列padding，上限2048，并非每批一律padding到2048；full/fragment target槽位256/64仍保留。跨长度随机混批可能产生padding开销，实际比例尚未测量。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3/dllm/pipelines/immune_llada/v3/layout.py。
- num_workers=0，prepared JSONL seek/decode及在线collation仍在训练进程，现有文件句柄缓存保留；它是待计时项，不能仅凭配置认定CPU加载为主瓶颈。主训练要求NCCL，额外Gloo组用于控制/保存检查。
- 池源码在领取真实任务后先调用standby.stop并等待其进程退出，再启动训练；本次没有重启或补发占卡任务。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py。GPU快照本身不是逐进程性能剖析。

下一性能检查按整模型执行，保留独立CUDA算子专项暂停：优先量化数据等待、前后向、attention backend、all-reduce、保存和峰值显存；再在适当训练边界比较选择性/关闭重计算，保持相同模型、数据、loss和global batch。密集bias/padding与更大microbatch需实测端到端吞吐和峰值；不能只减少step或改变batch后直接声称速度提升。任何实际优化仍须保留当前来源/审核/恢复约束及可比性，不能热改正在运行的冻结代码。本次只完成只读诊断与计划，没有执行profiler、GPU测试、模型改动、停训、重启或新提交。

原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/speed_diagnosis_20261006T002559Z.json（SHA8c3899842357341cbac9e3415875752ef49a91656b2e72d9f4e11e26923fbb81），包含实际GPU快照、千步事件、训练预算、日志步数核对和9份已读取源码SHA。首次metadata写后比较因整数里程碑键在JSON中转为字符串触发断言，已按JSON语义复核原文件，文件未改；这是记录脚本读回比较问题，不是训练失败，未重跑模型。


## 以下为2026-09-12及更早的历史ESMC速度记录

以下原文中的“current”仅指当时版本，不代表上述V7实际入口或当前速度。

> This document separates the current prepared-data speed boundary from historical
> `grammar_v2` measurements. Historical numbers are retained as evidence only and
> must not be read as a current validation result.
>
> Last updated: 2026-09-12. v5 `immune_v5_receptor_completion` is published
> (counts: plan §4.2). The CPU loader table below is still **not** a v5
> measurement. The queued v5 8-GPU job (`t-20260912021346-5xq4s`) has not
> launched, so there is no live v5 DataLoader / GPU profile. The section
> 「v5 8-GPU proxy measurements」below is from matching historical v3 runs
> (same global batch 256, or the same per-device/ga load), not from that job.

## Current speed boundary

The current training entry is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`

The active data implementation is:

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`

The measured path to profile is:

```text
prepared semantic JSONL + manifest
  -> prepared loader / byte-offset decode
  -> current immune grammar rendering
  -> batch padding and tensor assembly
  -> per-chain ESMC input reconstruction
  -> diffusion/MLM masking
  -> GPU forward/backward
```

This line does not consume the deleted qwen `data` alias, deleted `training` modules, raw-CSV
runtime loader, infinite weighted Arrow stream, or model-ready token cache. The retained
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`
and related model-layer files may still provide encoder/model primitives, but they do not define
the current data-loader speed boundary.

## Existing CPU prepared-loader baseline

The following values are prior prepared-loader measurements and are included for planning only:

| `num_workers` | throughput |
|---:|---:|
| 0 | **1367.76 samples/s** |
| 1 | **721.62 samples/s** |
| 2 | **1337.44 samples/s** |

These values are not a fresh run in this documentation pass. They do not establish the best
setting for GPU training, DDP, FSDP, or the current target hardware. In particular, do not infer
that `num_workers=1` or `num_workers=2` is universally correct from this table.

## v5 8-GPU proxy measurements (2026-09-12)

Worksheet (outside the repo):
`/vepfs-mlp2/c20250601/251105016/project/_jobmon/DECISION_NUMBERS.md`.
Decision recorded in `PROJECT_PROCESS.md` (2026-09-12 v5 entry) and
`examples/llada/PROTEIN_PRETRAIN_PROGRESS.md` §6.3 / §6.7:
`--save_top_k 3`, `--eval_steps 1000`, `ActiveDeadlineSeconds 950400` stay.

**Lesson:** do not copy one run's s/it onto another `per_device` / `ga` setup.
Read `training_args.bin` for `per_device × ga × world_size` first.

| Config | Evidence | Actual factors | Measured / inferred s/it |
|---|---|---|---|
| Historical v3 8-GPU | `t-20260901033935-h55f4`, `t-20260904014842-qgjbg` | `per_device=2`, `ga=16`, `world_size=8`, `max_length=1024`, global **256** | **2.94–2.95 s/it** (tqdm) |
| v3 4-GPU, same per-GPU load as v5 | `..._v3_4gpu` `checkpoint-42000/training_args.bin` | `per_device=4`, `ga=8`, `world_size=4` | **1.51–1.91 s/it** (tqdm) |
| Queued v5 8-GPU | `t-20260912021346-5xq4s` (not launched) | `per_device=4`, `ga=8`, 8 GPU, global **256** | **1.6–2.5 s/it** (center ~2.2) → train **4.0–5.8 days** |

Microbatch scaling is **sublinear**: `t(batch4)/t(batch2) ≈ 1.29`, so v5 step wall-clock
is about **0.65×** the v3 8-GPU 2.95 s/it (`0.5 × 1.29`: half as many accumulation
steps, each microstep only 1.29× slower).

**Eval lesson:** never extrapolate a single-source rate to the full valid split.

- `eval_oas_runtime=4.4792` at 2000 rows/source → OAS **446.5 samples/sec**.
  Source: `..._v3_8gpu_2m/checkpoint-105000`. That run **was already 8-GPU**;
  do not normalize by GPU count again.
- Same 8-GPU eval: `asd_antibody` is only **~114 samples/sec** (~3.9× slower)
  and is 44% of v5 valid.
- Per-source extrapolation on v5 valid **102,308** rows: one full eval
  **~8.7 min**; `--eval_steps 1000` × 200 evals **~29 h ≈ 1.21 days**.

## Historical grammar-v2 evidence

Earlier integrated ESMC + LLaDA logs under historical `output/grammar_v2_*` runs reported roughly
34–40 samples/s on 8 GPUs, with `wait_s` around 0.43–0.66 seconds per microstep after the old
worker/prefetch changes and a recorded peak near 46.0 GB on a 79.2 GB GPU. Those logs belong to
the deleted grammar-v2/Arrow training line. They remain useful only as historical context for
where CPU loading and encoder forward cost appeared, not as a reproducible current baseline.

The historical line also recorded two old optimizations: gradient all-reduce suppression on
non-boundary accumulation steps and worker prefetch overlap. These should be re-measured against
the current prepared loader before being described as deployed current behavior.

## Ranked current profiling plan

### 1. Profile the prepared loader and GPU overlap first

Run a short, representative profile using the current formal entry and current prepared-data
configuration. Record, with absolute artifact paths:

- prepared-loader startup/index-build time and steady-state samples/s;
- JSON decode, grammar rendering, padding, encoder reconstruction, and masking time;
- host-to-device wait time and GPU forward/backward time;
- peak CUDA memory, effective tokens/s, and batch shape/length distribution;
- DDP/FSDP rank synchronization and first-batch identity where applicable.

The profile must distinguish index construction from steady-state iteration. The prepared loader
still decodes semantic rows and performs runtime construction; it is not a zero-cost cache.

### 2. Gate worker/prefetch choices on the profile

Compare `num_workers=0`, `1`, and only then higher values on the target environment. Also compare
persistent workers and prefetch settings only when they are supported by the active loader. Do not
hard-code the historical `nw=1` rule: the old Arrow worker-sharding rationale does not establish
an equivalent invariant for the current prepared semantic loader.

For every candidate, verify that all ranks receive compatible batch shapes/tasks and that no
worker-specific ordering or shard identity causes DDP divergence. Select the setting by measured
GPU overlap and end-to-end samples/tokens per second, not CPU-only throughput alone.

### 3. Profile length-aware batching before increasing model batch size

The current grammar and per-chain encoder reconstruction make padding and sequence length a major
cost. Prefer length buckets or a token-budget batch policy if the active collator can preserve
record/task semantics and deterministic distributed sampling. First measure padding waste and
maximum encoder length; then compare fixed sample count with token-budget batches.

Every change must be short-run profiled on the target GPU. Keep effective global batch comparable
when comparing encoder variants, and record peak memory before committing a new batch size.

### 4. Measure gradient accumulation and checkpointing separately

Retain gradient accumulation or non-boundary synchronization changes only after checking their
actual communication and wall-clock effect on the current Trainer/DDP/FSDP path. Likewise, compare
whole-model and selective activation checkpointing with the current fusion architecture. Report
peak memory and tokens/s together; a faster microstep that reduces feasible batch size may be a
regression at the global-batch level.

### 5. Treat flash attention as environment-gated

The retained model layer exposes the ESMC flash-attention option, but the option is not evidence
that flash attention is installed or enabled in the current environment. If the package is
available, run numerical/parity checks and a short throughput comparison against the non-flash
path. If it is unavailable, the fallback must remain the documented behavior; do not report a
speedup or claim that the current baseline uses flash attention.

## Recommendation

The next speed action is a profile of the current prepared loader plus GPU overlap on the formal
`protein_pretrain_esmc.py` line. Use the result to choose workers, prefetch, length bucketing,
token budget, batch size, and checkpointing. Do not reuse deleted qwen data/training modules or
fix a worker count from historical Arrow measurements. The main agent must append the actual
command, exit code, elapsed time, and absolute profile artifact to
`/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`; this documentation update
does not claim that the latest speed or full-model validation passed.


## 2026-10-06T04:39:59.360843+00:00 分段边界的额外工作已确认

冻结driver在5016段尾额外执行评估与保存，然后由原监督接续并读取原生恢复状态。5000是常规评估点，5016段尾结果另行记录；不能把16步间隔或接段加载时间当作稳态GPU吞吐。4000→5000的既有区间计时保持，当前仍没有GPU组件profile。8rank退出/恢复完整；5条外层清理扫描提示及无测试权重残留的证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json。未据该提示归因训练慢化或改变运行资源。


## 2026-10-06T14:59:56.446079+00:00 where版本新5000步实际完整段

2026-10-06T13:38:33.712836+00:00至2026-10-06T14:41:13.933865+00:00，5000次更新总3760.221029秒，平均0.752044秒/更新（含恢复、评估/保存、8rank退出验收）；普通训练间隔均值0.716325秒。端点训练窗口包含常规保存但排除段初始化/退出，为0.725637秒/更新。该段长时实测支持此前短窗口约0.72秒的速度，不将端点开销混入普通步或当作全epoch ETA。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/segment17016_speed_root_review.json；没有FSDP/attention backend新切换或额外模型测试。
