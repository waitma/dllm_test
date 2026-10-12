# 项目进度

Last updated: 2026-10-12T09:40:31.716357+00:00

## 当前决定

**新版本使用 V5 数据训练**，目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion。V7清理版不再作为本次新训练的数据来源。保留学习`.`／`<protd>`、残差1.0和蛋白对象内／全局交替；对象由prots/protd划分，内部split不分attention。容量改为302，采用原H/β与L/α的decoder总槽位167+135；不纳入VHH。

| 工作 | 当前状态 | 下一步 |
|---|---|---|
| 75.6M正式训练 | 300000/300000完成；最终完整保存核对通过，退出码0；八卡standby已恢复 | 原产物保留 |
| 270M新训练 | 容器Running；synthetic 8/8；正式61581/300000步；训练running | 核实已保存产物，并监督62000步验证/保存 |
| 新模型全下游评测 | best45000本地修复已终态；详见repair_final_20261009.json | [新状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/dedicated_status_check.json) |
| 新训练数据 | V5全量7,729,193条通过；302容量覆盖；启动绑定通过 | [启动参数与回执](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/README.md) |
| V7 f011正式训练 | 已完成两轮，checkpoint-50082 | 保留历史结果，不重跑 |
| V7三类生成评测 | CDR、轻链、TCR已完成 | 数字见[downstream/benchmark/RESULTS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) |
| 原22格消融 | 已按用户要求停止：8项停于3600–3650步、4项未启动；原曲线保留 | 不自动续跑；[停止记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json) |

## Active Volc Training Tasks

本表记录当前作业最近的实际平台查询与训练观察；时间及原始证据见各行。

| 任务 | ID | 最后记录 | 证据 |
|---|---|---|---|
| 270M legacy CDR32000 g1 | t-20261012174026-thrps | 2026-10-12T09:40:53.223713+00:00 Queue；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_legacycdr32000_20261012/submissions.json) |
| 270M legacy CDR32000 g0 | t-20261012174021-jrcbb | 2026-10-12T09:40:53.223713+00:00 Staging；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_legacycdr32000_20261012/submissions.json) |
| 270M best32000 下游 g3 | t-20261012172749-tx78f | 2026-10-12T09:28:12.757870+00:00 Staging；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/submissions.json) |
| 270M best32000 下游 g2 | t-20261012172745-kxt5s | 2026-10-12T09:28:12.757870+00:00 Staging；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/submissions.json) |
| 270M best32000 下游 g1 | t-20261012172742-pbvpt | 2026-10-12T09:28:12.757870+00:00 Staging；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/submissions.json) |
| 270M best32000 下游 g0 | t-20261012172738-sfpmb | 2026-10-12T09:28:12.757870+00:00 Staging；非闲时 | [提交记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012/submissions.json) |
| immune-v5-devpool-8gpu-20261007 | t-20261007153431-vhx2f | 2026-10-12T09:40:31.716357+00:00 容器Running；synthetic 8/8；正式61581/300000步；训练running | [状态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/latest.json) |


原消融仍使用它原先冻结的数据／源码，不因新版本改用V5而切换。完成状态以实际退出及产物为准；不将单次进度观察当终态。

## 下一步

当前270M正式训练使用已有八卡容器t-20261007153431-vhx2f，唯一票据v5_270m_scratch_bs64_20261010；从头初始化，实际268,590,112参数。八卡整模synthetic和通信通过，85分片全哈希数据门禁通过，已产生真实训练更新步。训练前占卡已全部停止并回收；训练退出后由原容器监督器自动恢复八卡standby，不自动重试失败训练。

实际参数：300000步、8×micro64×GA1/global512、LR1e-4、warmup2000、cosine，每1000步验证/保存，best3＋latest完整状态。配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json；已固定运行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010。2026-10-12本次核实已到60305/300000步（约20.10%）；最新完整保存60000步/epoch4.0279，累计60次原生验证。主eval_loss（全遮蔽/source-macro）最佳32000步1.993467，最新60000步2.007204；相对17000步旧最佳仅改善0.0027%，整体仍波动。最近10000更新墙钟平均2.8940秒/步，包含窗口内验证/保存；不中断约余8.0天。 60000步17文件哈希、八rank保存确认及90项核心指标聚合核对通过；本轮仅独立核对该份完整保存，历史指标来自Trainer记录。下一61000步验证/保存。

只读监督已启动：exec会话26867、PID1737482，最多15天；当前状态与训练退出后的八卡占卡证据位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010。旧50小时观察90153已真实exit0，旧75.6M已完成300000步并通过最终完整产物/八rank/进程退出核对；历史产物保留，不重跑。

## 文档入口

- [BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md)：当前模型方案。
- [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md)：V5数据及输入规则。
- [PROJ_GUIDE.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md)：日常操作约定。
- [实验与更正](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)：关键里程碑、证据、已纠正结论。
- [完整历史](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/PROJECT_PROCESS.md)：整理前逐次日志。

## 2026-10-07 数据选择与文档整理

按用户要求切换新版本数据选择到V5；归档重复过程记录并精简当前文档。保留历史实验数据、指标、checkpoint来源与原作业状态。数据绑定现已完成；实际训练参数现已配置；运行仍需显存检查与资源调度。

## 2026-10-07 按用户要求暂停消融并切换新训练

已向原八卡池写入STOP_NOW，请求终止在跑消融并阻止四项排队消融启动。指标、原始日志、排队票据及正式训练恢复checkpoint保留。此刻只确认停止请求，实际退出与平台终态另核实。新训练采用当前V5冻结包；未重复提交。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/user_pause_20261007.json。

原池已于2026-10-07T07:08:26Z终止，八worker均exited、running目录清空；平台标为Failed，原因为用户主动中断，不是新发现模型故障。task_id=t-20261005205851-lqfjg，Preemptible=false，YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/eight_gpu_pool_20261005.yml。未调用cancel或重启；终态任务已移出Active表。

## 2026-10-07 新训练启动整合

YAML已写 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml；使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2。权重从头随机初始化，源码固定副本不代表冻结参数。提交前补齐V5分隔符验证统计，否则首次eval会将`.`误归入binding校验而失败；训练目标及模型架构不变。保留best3＋latest完整状态、速度与显存记录。GPU实测待任务启动。

## 2026-10-07 提交V5新模型八卡训练

操作：submit；task_id=['t-20261007151807-cq4fb']；任务名=immune-v5-object302-scratch-8gpu-20261007；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml；初始状态=submitted_pending_platform_query；Preemptible=false；命令returncode=0。global512=8×micro8×GA8，300000步，DDP；全部权重随机初始化。显存待所分配GPU上synthetic实测，不将提交成功当作已训练。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/submission.json。

## 2026-10-07 新训练首次启动失败

任务t-20261007151807-cq4fb于07:20:18Z Failed；rank4/6 SIGSEGV(-11)，发生在NCCL初始化后、synthetic报告前。正式训练0步，无新增checkpoint。平台终态已移出Active表；失败任务不重启、原始日志保留，不将CPU测试通过当八卡启动通过。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v2/failure_review.json。

## 2026-10-07 配置确认与原容器状态

用户指定training_config.json与实际launch_v2／候选launch_v3的全部训练超参数逐项一致，见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/training_config_agreement_20261007.json。状态字段已更正：曾提交、Failed、正式0步；超参数未改。用户希望复用原训练容器，但原池STOP_NOW使八worker及容器任务终止，无法直接进入已释放资源。用户随后明确持续容器方式，已新申请t-20261007153431-vhx2f；候选修正v3将通过容器内票据执行，不提交v3直接训练YAML，也不重启失败任务。

## 2026-10-07 按用户纠正恢复持续容器方式

按最新要求，改为先申请持续八卡容器，训练在容器内执行；训练退出/失败不退出容器，占卡进程必须在训练前全部停止并wait。资源YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml；控制目录=/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007。当前仅完成准备、空训练队列；未提交候选v3直接训练YAML。

## 2026-10-07 提交持续八卡容器

操作submit；task_id=['t-20261007153431-vhx2f']；任务名=immune-v5-devpool-8gpu-20261007；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml；Preemptible=false；returncode=0；初始状态待平台查询。只提交资源容器，训练队列仍空；不重复提交已失败直接训练任务。17个CPU fixture子进程全部wait回收，已验证训练失败后容器继续、无自动重试、standby训练前停止。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/submission.json。

## 2026-10-07 容器内发布正式训练票据

持续资源t-20261007153431-vhx2f Running，host=t-20261007153431-vhx2f-worker-0，supervisor PID737，八个standby PID770–777。已发布一次v5_scratch_launch_v3_20261007；训练初始化scratch，参数与用户指定training_config一致。此为容器内执行，不是第二次直接训练平台submit。占卡回收、GPU启动验收、正式更新步需看实际结果。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/training_enqueue.json。

## 2026-10-07 持续容器内启动检查通过

八卡均完成3步synthetic前后向，micro8／长度1218，峰值allocated最大4.48 GiB；八个rank_nccl_ready全部生成，已越过首次失败位置。无synthetic checkpoint。容器state为training、standby_pids为空。随后已通过原生V5数据门禁并产生正式更新步；实时步数以step_timing.jsonl为准。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2/startup_review.json。首次SIGSEGV确切根因仍未单独归因。

## 2026-10-07 正式V5训练已开始

实际至121/300000步；单机八卡DDP，micro8×GA8/global512，完全随机初始化；数据train 7,626,885条，使用launch_v3代码。最近25个普通更新步中位耗时0.798秒。八卡显存及通信已通过，正式门禁回执随run_provenance保存。尚未到首次1000步完整验证/保存；不提前宣称长程、checkpoint或下游成绩通过。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/formal_start_review.json。

## 2026-10-07 首次1000步验证与保存通过

07:54:20Z只读验收通过：17个完整状态文件重新核对SHA256，八rank参数/保存确认一致；四个mask等级、七数据源的90项核心指标聚合一致。主eval_loss=2.1790857（mask100/source-macro）；当前best与latest均为checkpoint-1000，符合best3＋latest并集规则。训练随后继续超过1100步，没有重启或额外提交；已保存可恢复所需状态，本次未执行恢复重放。证据：[1000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint1000_root_review.json)。

首个完整1000更新窗口（已完成step1→1001）实际803.212秒，含一次验证/保存，平均0.803212秒/步；训练callback计时平均0.793750秒/步，其外合计9.462秒，不能进一步拆成单独验证或保存耗时。此为当前DDP/global512配置的实际吞吐，不是与旧版本的同配置对照。证据：[完整窗口计时](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/speed_first1000_updates_root_review.json)。

## 2026-10-07 2000步验证与保存通过

第二份checkpoint的17个状态文件、八rank参数/保存确认及90项核心聚合通过，参数相对1000步已更新；训练已继续超过2000步。主eval_loss（mask100/source-macro）2.179086→2.132296，mask100/token-micro AA准确率31.27%→33.54%；同口径关系标签重建准确率80.76%→69.34%，需继续观察，不能宣称全面改善或下游binding判别提升。保留完整checkpoint-1000与checkpoint-2000，best/latest当前为2000；未执行恢复重放。证据：[2000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint2000_root_review.json)。

关系标签回退主要来自tcr_native（96.48%→37.50%）；四个有关系标签来源的数量均保持各256，尚无原因定论。分源原件与数值见[实验记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)，3000步继续观察，训练配置不变。

## 2026-10-07 3000步验证与保存通过

第三份完整保存通过17文件哈希、八rank一致性、90项核心聚合和保留目录核对；主eval_loss=2.072132，mask100/token-micro AA准确率34.35%、关系标签准确率71.48%。tcr_native关系标签准确率36.72%，较2000步37.50%未恢复，不能以总体指标回升掩盖分源回退。最优/最新为3000，完整保留1000/2000/3000，无超额权重待清理；恢复重放未执行。证据：[3000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint3000_root_review.json)。继续同一配置，下一4000步。

## 2026-10-07 4000步验证与首次checkpoint轮换

4000步完整保存/验证检查通过。主eval_loss=2.108269，较3000步回升；mask100/token-micro AA准确率33.70%、关系标签准确率78.81%，其中tcr_native回升至56.25%。当前最优仍为3000、最新为4000，按best3＋latest保留2000/3000/4000三份完整状态。1000步权重已由训练原生保留逻辑删除；其六份JSON档案与删除前记录哈希一致，指标及来源证据保留。未重启训练或执行恢复重放。证据：[4000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint4000_root_review.json)、[清理后档案核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/retention_after4000_archive_review.json)。下一5000步，配置不变。

## 2026-10-07 5000步验证与保存通过

5000步17文件、八rank、90项核心聚合、参数更新及实际保留目录均通过。主eval_loss=2.091514，较4000步改善但最佳仍为3000；mask100/token-micro AA准确率34.27%、关系标签准确率75.78%，tcr_native关系标签44.92%，继续观察。现保留3000/4000/5000三份完整状态；2000步权重自动清理，其六份JSON档案哈希检查已随本次完整验收通过。未重启训练、恢复checkpoint或修改配置。证据：[5000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint5000_root_review.json)。下一6000步。

## 2026-10-07 6000步验证与保存通过

6000步完整状态与验证检查通过，主eval_loss=2.050985，为当前最低；mask100/token-micro AA准确率35.15%、关系标签准确率79.00%，tcr_native关系标签58.59%。分隔符准确率较5000步略退，不宣称全面改善。最优与最新均为6000，保留3000/5000/6000三份完整状态；4000步权重自动清理，六份JSON档案哈希检查通过。原容器与训练继续，配置不变，无恢复重放。证据：[6000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint6000_root_review.json)。下一7000步。

## 2026-10-07 7000步验证与保存通过

7000步完整状态与验证检查通过，主eval_loss=2.102113，较6000步回退；mask100/token-micro AA准确率34.17%、关系标签准确率71.88%，tcr_native关系标签35.55%。当前最优仍为6000；最佳三份3000/5000/6000与最新7000取并集，实际保留四份完整状态，本次没有旧权重被清理。该数量符合best3＋latest规则。配置保持不变，无训练重启或恢复重放。证据：[7000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint7000_root_review.json)。下一8000步。

## 2026-10-07 8000步验证与保存通过

8000步完整状态/验证检查通过，主eval_loss=2.070347，较7000改善，最佳仍6000；mask100/token-micro AA准确率34.52%、关系标签准确率77.25%，tcr_native关系标签52.34%。终止符与分隔符准确率较7000回退，继续如实记录。best3为6000/8000/3000，latest8000与best3重合，实际保留3000/6000/8000三份完整状态；5000/7000权重自动清理，两点共12份JSON档案哈希检查通过。配置及训练进程保持，未执行恢复重放。证据：[8000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint8000_root_review.json)。下一9000步。

## 2026-10-07 9000步验证与保存通过

9000步完整状态/验证检查通过，主eval_loss=2.033069，为当前最低；mask100/token-micro AA准确率36.05%、关系标签准确率74.12%，tcr_native关系标签39.45%。主loss、AA及边界指标较8000改善，关系标签指标回退，不宣称全面提升。最优与最新均为9000，保留6000/8000/9000三份完整状态；3000步权重自动清理，其六份JSON档案哈希检查通过。保持原配置，无额外训练启动或恢复重放。证据：[9000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint9000_root_review.json)。下一10000步。

## 2026-10-07 10000步验证与保存通过

10000步完整状态/验证检查通过，主eval_loss=2.028133，为当前最低；mask100/token-micro AA准确率36.53%、终止符75.61%、关系标签71.29%、分隔符78.03%。AA与分隔符提高，终止符及关系标签回退；tcr_native关系标签32.81%，不宣称全面改善或下游效果。最优与最新均为10000，实际保留6000/9000/10000三份完整状态；8000步权重自动清理，其六份JSON档案哈希核对通过。原容器内训练继续，未改配置或执行恢复重放。证据：[10000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint10000_root_review.json)。下一11000步。

## 2026-10-07 11000步验证与保存通过

11000步完整状态/验证检查通过，主eval_loss=2.025401，为当前最低；mask100/token-micro AA准确率35.61%、终止符75.95%、关系标签74.71%、分隔符75.42%。终止符与关系标签回升，AA和分隔符回退；tcr_native关系标签42.19%，仍有波动。best与latest均为11000，保留9000/10000/11000三份完整状态；6000步权重由原生策略清理，其六份JSON档案哈希核对通过。原容器内训练继续，配置保持，未执行恢复重放。证据：[11000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint11000_root_review.json)。下一12000步。

## 2026-10-07 12000步验证与保存通过

12000步完整状态/验证检查通过，主eval_loss=2.025920，略高于11000步；mask100/token-micro AA准确率34.97%、终止符84.49%、关系标签73.83%、分隔符76.41%。终止符与分隔符回升，AA及关系标签回退；tcr_native关系标签38.28%。最优仍为11000，最新12000；保留10000/11000/12000三份完整状态，9000步权重自动清理，其六份JSON档案哈希核对通过。继续原配置，未执行恢复重放。证据：[12000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint12000_root_review.json)。下一13000步。

## 2026-10-07 13000步验证回退与保存通过

13000步完整保存及指标核对通过，主eval_loss=2.122830，高于12000步2.025920；最优仍11000。终止符/关系标签准确率上升，AA和分隔符回退。按source-macro等权分解，tcr_native loss1.900705→2.532334贡献本次总增量约93.11%，各来源全遮蔽计数未变，原因尚未确定。best3为11000/12000/10000，加最新13000，实际保留四份完整状态，本次无清理。下一14000步继续观察，原配置与容器保持。证据：[13000步核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_root_review.json)、[分来源增量](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint13000_source_delta_review.json)。

## 2026-10-07 14000步验证恢复与保存通过

14000步完整保存及指标核对通过，主eval_loss=2.018878，为新最佳。tcr_native loss2.532334→1.873917、AA准确率33.23%→43.07%，本点恢复；该来源关系标签43.36%→34.38%回退。整体全遮蔽AA36.93%、终止符82.76%、关系标签71.97%、分隔符78.09%，存在分项回退。实际保留11000/12000/14000三份完整状态，10000/13000权重由原生策略清理，两点共12份JSON档案哈希通过。继续原配置至15000步。证据：[14000步核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_root_review.json)、[分来源后续核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint14000_source_followup_review.json)。

## 2026-10-07 15000步验证与首轮训练记录

15000步完整保存及指标核对通过，原生trainer_state记录epoch=1.006914。主eval_loss=2.044152，较14000步回升；全遮蔽AA34.92%、终止符85.83%、关系标签71.48%、分隔符75.48%，最佳仍14000。tcr_native loss1.873917→2.022110、AA43.07%→33.91%，仍有波动。best3=14000/11000/12000加最新15000，实际保留四份完整状态，本次无清理。首轮完成不代表300000步目标完成；保持配置，下一16000步。证据：[15000步核对报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint15000_root_review.json)。

## 2026-10-07 16000步验证与保存通过

16000步完整保存及指标核对通过，主eval_loss=2.052560，较15000步小幅回升，最佳仍14000。全遮蔽AA34.47%、终止符84.60%下降，关系标签76.46%提高，分隔符75.48%持平。保留11000/12000/14000/16000四份完整状态；15000旧权重自动清理，六份JSON档案哈希通过，首轮训练记录保留。继续原配置至17000步，未执行恢复重放。证据：[16000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint16000_root_review.json)。

## 2026-10-07 17000步验证与保存通过

17000步完整保存及指标核对通过，主eval_loss=2.046143，较16000步改善，最佳仍14000。全遮蔽AA34.79%、分隔符78.03%提高，终止符82.92%、关系标签75.29%回退。保留11000/12000/14000/17000四份完整状态；16000旧权重自动清理，六份JSON档案哈希通过。现有训练及配置保持，下一18000步。证据：[17000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint17000_root_review.json)。

## 2026-10-07 18000步验证与保存通过

18000步完整保存及指标核对通过，主eval_loss=2.073688，较17000步回升；最佳仍14000。全遮蔽AA34.81%略升，终止符81.75%、关系标签70.61%、分隔符75.36%回退；tcr_native loss2.043230→2.229023。保留11000/12000/14000/18000四份完整状态；17000旧权重自动清理，六份JSON档案哈希通过。保持原配置至19000步。证据：[18000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint18000_root_review.json)。

## 2026-10-07 19000步验证与保存通过

19000步完整保存及指标核对通过，主eval_loss=2.064763，较18000步改善；最佳仍14000。全遮蔽AA34.82%、终止符82.92%、关系标签75.10%、分隔符75.48%提高，AA均值75.94%略退；分源仍有回退。保留11000/12000/14000/19000四份完整状态，18000旧权重自动清理，六份JSON档案哈希通过。保持原配置至20000步。证据：[19000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint19000_root_review.json)。

## 2026-10-07 20000步验证与保存通过

20000步完整保存及指标核对通过，主eval_loss=2.040837，较19000步改善，最佳仍14000。全遮蔽AA34.90%、关系标签75.78%提高，终止符81.47%、分隔符75.42%回退；分源变化见实验记录。现保留11000/12000/14000/20000四份完整状态，19000旧权重自动清理，其六份JSON档案哈希一致。训练继续超过20000步，保持原配置，下一21000步。证据：[20000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint20000_root_review.json)。

## 2026-10-07 21000步验证与保存通过

21000步完整保存及指标核对通过，主eval_loss=2.027940，较20000步改善，最佳仍14000。全遮蔽AA35.05%、终止符84.54%、分隔符77.96%提高，关系标签75.68%略退。现保留11000/12000/14000/21000四份完整状态，20000旧权重自动清理，六份JSON档案哈希一致。原训练继续，配置保持，下一22000步。证据：[21000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint21000_root_review.json)。

## 2026-10-07 22000步验证与保存通过

22000步完整保存及指标核对通过，主eval_loss=2.052559，较21000步回升；最佳仍14000。全遮蔽AA34.92%、终止符80.41%、分隔符75.42%回退，关系标签79.49%提高。现保留11000/12000/14000/22000四份完整状态，21000旧权重自动清理、六份JSON档案哈希一致。原训练继续，配置保持，下一23000步。证据：[22000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint22000_root_review.json)。

## 2026-10-07 23000步验证与保存通过

23000步完整保存及指标核对通过，主eval_loss=2.025071，改善并进入最佳三份，最佳仍14000。全遮蔽AA35.61%、分隔符75.54%提高，终止符80.25%、关系标签72.85%回退。best3为14000/23000/11000，最新23000与之重合，实际保留11000/14000/23000三份完整状态；12000和22000旧权重自动清理，两点共12份JSON档案哈希一致。原训练继续，配置保持，下一24000步。证据：[23000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint23000_root_review.json)。

## 2026-10-07 24000步验证与保存通过

24000步完整保存及指标核对通过，主eval_loss=2.024323，较23000步小幅改善，当前第二、最佳仍14000。终止符81.47%、关系标签75.68%、分隔符78.27%提高，全遮蔽AA35.53%、AA均值76.20%略退。best3为14000/24000/23000，最新24000已在best3内，实际保留14000/23000/24000三份完整状态；11000旧权重自动清理，六份JSON档案哈希一致。原训练继续，配置保持，下一25000步。证据：[24000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint24000_root_review.json)。

## 2026-10-07 25000步验证与保存通过

25000步完整保存及指标核对通过，主eval_loss=2.089885，较24000步回升；按分源loss差值计算，tcr_native贡献主loss净增量约97.69%，仅为算术分解、原因未定。全遮蔽AA34.63%、关系标签74.51%、分隔符74.92%回退，终止符81.53%略升。最佳三份仍14000/24000/23000，加latest25000，实际保留14000/23000/24000/25000四份完整状态；本次无旧权重删除。原训练继续，配置保持，下一26000步。证据：[25000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint25000_root_review.json)。

## 2026-10-07 26000步验证与保存通过

26000步完整保存及指标核对通过，主eval_loss=2.061376，较25000步改善；tcr_native的loss/AA仅部分恢复，终止符恢复到99.22%，分隔符仍71.48%。全遮蔽AA34.87%、关系标签74.61%、分隔符75.48%提高，总体终止符79.07%回退。最佳仍14000；实际保留14000/23000/24000/26000四份完整状态，25000旧权重自动清理、六份JSON档案哈希一致。原训练继续，配置保持，下一27000步。证据：[26000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint26000_root_review.json)。

## 2026-10-07 27000步刷新主loss最佳值

27000步完整保存及指标核对通过，主eval_loss=2.011242，优于原14k的2.018878，为当前主loss最佳。全遮蔽AA35.64%、关系标签84.96%提高，分隔符75.48%持平；终止符降至70.93%，七来源均回退，不称为全面改善。best3=27000/14000/24000，latest已在其中，实际保留14000/24000/27000三份完整状态；23000与26000旧权重自动清理，共12份JSON档案哈希一致。原训练及配置保持，下一28000步。证据：[27000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint27000_root_review.json)。

## 2026-10-07 28000步验证与保存通过

28000步完整保存及指标核对通过，主eval_loss=2.012737，当前第二，最佳仍27000。七来源终止符均回升，总体79.74%；分隔符78.09%提高，全遮蔽AA35.59%、关系标签74.80%回退。tcr_native终止符仍低于26k水平，关系标签87.50%→38.28%下降，不能称为全面恢复。实际保留14000/27000/28000三份完整状态；24000旧权重自动清理，六份JSON档案哈希一致。原训练及配置保持，下一29000步。证据：[28000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint28000_root_review.json)。

## 2026-10-07 29000步验证与保存通过

29000步完整保存及指标核对通过，主eval_loss=2.019129，最佳仍27000。全遮蔽AA35.59%→36.69%、终止符79.74%→80.69%提高，关系标签74.80%→71.68%、分隔符78.09%→75.48%回退。tcr_native终止符恢复到99.22%，但oas降至44.53%；分源表现仍有波动。best3保持27000/28000/14000，最新29000不在best3，实际保留14000/27000/28000/29000四份完整状态，本次无旧状态待删。原训练及配置保持，下一30000步。证据：[29000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint29000_root_review.json)。

## 2026-10-07 30000步保存通过并越过两轮

30000步完整保存及指标核对通过，Trainer epoch=2.013829，完成计划步数的10%；仍按原300000步计划训练。主eval_loss=2.039798回升，最佳仍27000；全遮蔽AA34.90%、终止符75.00%、关系标签71.48%、分隔符74.18%均较29k下降。tcr_native AA41.41%→33.80%，asd_antibody分隔符26.67%→1.33%，需继续观察。best3保持27000/28000/14000，实际保留14000/27000/28000/30000四份完整状态；29000旧权重自动清理，六份JSON档案核对通过。原训练及配置保持，下一31000步。证据：[30000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint30000_root_review.json)。

## 2026-10-07 31000步验证与保存通过

31000步完整保存及指标核对通过，主eval_loss=2.042209略升，最佳仍27000。全遮蔽AA34.90%→35.03%、终止符75.00%→80.30%、分隔符74.18%→75.42%提高，关系标签71.48%→71.00%下降。asd_antibody分隔符1.33%→26.67%、oas终止符19.53%→53.91%恢复；tcr_native的AA及关系标签准确率继续下降。实际保留14000/27000/28000/31000四份完整状态；30000旧权重自动清理，六份JSON档案核对通过，首次越过两轮的证据保留。原训练及配置保持，下一32000步。证据：[31000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint31000_root_review.json)。

## 2026-10-07 32000步验证与保存通过

32000步完整保存及指标核对通过，主eval_loss=2.061845；27k后连续五个保存点上升，最佳仍27000。终止符80.30%→82.92%、关系标签71.00%→71.19%、分隔符75.42%→75.48%提高，全遮蔽AA35.03%→35.02%略降。tcr_native AA及关系准确率略升，整体loss继续回升；暂不作原因判断或改动配置。实际保留14000/27000/28000/32000四份完整状态；31000旧权重自动清理，六份JSON档案核对通过。原训练继续，下一33000步。证据：[32000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint32000_root_review.json)。

## 2026-10-07 33000步验证与保存通过

33000步完整保存及指标核对通过，主eval_loss=2.026425下降，结束27k后的连续五次上升，最佳仍27000。全遮蔽AA35.14%、终止符83.71%、分隔符78.09%提高；关系标签71.19%→66.89%下降，tcr_native32.81%→12.11%明显回退。各源目标计数保持一致，原因未定，继续跟踪。实际保留14000/27000/28000/33000四份完整状态；32000旧权重自动清理，六份JSON档案核对通过。原训练及配置保持，下一34000步。证据：[33000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint33000_root_review.json)。

## 2026-10-07 34000步验证与保存通过

34000步完整保存及指标核对通过，主eval_loss=2.066092回升，最佳仍27000。全遮蔽AA35.20%、分隔符80.01%提高，终止符79.69%、关系标签64.16%下降。tcr_native关系标签12.11%→3.91%继续回退；其25%/50%/75%遮蔽准确率为75.71%/74.29%/78.84%，各等级数量与此前一致，全遮蔽回退更剧烈。原因未定，继续跟踪。实际保留14000/27000/28000/34000四份完整状态；33000旧权重自动清理，六份JSON档案通过。原配置保持，下一35000步。证据：[34000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint34000_root_review.json)。

## 2026-10-07 35000步验证与保存通过

35000步完整保存及指标核对通过，主eval_loss=2.032845下降，最佳仍27000。关系标签64.16%→75.88%提高；tcr_native全遮蔽关系3.91%→43.36%、其他三遮蔽等级也回升，但四等级仍低于27k。终止符79.69%→75.17%、分隔符80.01%→78.09%、全遮蔽AA35.20%→35.18%下降。实际保留14000/27000/28000/35000四份完整状态；34000旧权重自动清理、六份JSON档案通过。只读观察20150实际exit0后接续66245，训练未重启。下一36000步，原配置保持。证据：[35000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint35000_root_review.json)。

## 2026-10-07 36000步验证与保存通过

36000步完整保存及指标核对通过，主eval_loss=2.022820继续下降，最佳仍27000。终止符75.17%→80.64%提高；关系标签75.88%→73.63%、全遮蔽AA35.18%→34.98%、分隔符78.09%→75.42%下降。native四遮蔽关系准确率均较35k下降，全遮蔽43.36%→39.45%，各等级计数不变。实际保留14000/27000/28000/36000四份完整状态；35000旧权重自动清理、六份JSON档案通过。原训练与只读观察66245继续，下一37000步，配置保持。证据：[36000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint36000_root_review.json)。

## 2026-10-07 37000步验证与保存通过

37000步完整保存及指标核对通过，主eval_loss=2.056276回升，最佳仍27000。全遮蔽AA34.98%→34.39%、终止符80.64%→76.23%、关系标签73.63%→73.34%下降，分隔符75.42%→75.54%略升。native四遮蔽关系准确率均回升，全遮蔽39.45%→40.23%，各等级计数保持不变。实际保留14000/27000/28000/37000四份完整状态；36000旧权重自动清理、六份JSON档案通过。原训练与观察66245继续，下一38000步，配置保持。证据：[37000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint37000_root_review.json)。

## 2026-10-07 38000步验证与保存通过

38000步完整保存及指标核对通过，主eval_loss=2.057750略升，最佳仍27000。全遮蔽AA34.39%→35.16%、终止符76.23%→83.09%提高；关系标签73.34%→71.88%、分隔符75.54%→75.42%下降。native四遮蔽关系准确率均下降，全遮蔽40.23%→32.03%，各等级计数不变。实际保留14000/27000/28000/38000四份完整状态；37000旧权重自动清理、六份JSON档案通过。当前仍在已申请的八卡容器内训练，standby_pids为空；未重新申请容器或提交训练。下一39000步，配置保持。证据：[38000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint38000_root_review.json)。

## 2026-10-07 39000步验证与保存通过

39000步完整保存及指标核对通过，主eval_loss=2.040339下降，最佳仍27000。全遮蔽关系标签71.88%→74.12%提高；AA35.16%→35.05%、终止符83.09%→79.24%、分隔符75.42%→74.49%下降。native关系准确率在mask25持平，其他三等级提高，全遮蔽32.03%→39.06%；各等级计数不变。实际保留14000/27000/28000/39000四份完整状态，38000旧权重自动清理、六份JSON档案核对通过。原容器及训练继续，下一40000步，配置保持。证据：[39000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint39000_root_review.json)。

## 2026-10-07 40000步验证成为当前主loss最佳

40000步完整保存及指标核对通过，主eval_loss=2.000729，优于原27000步的2.011242。全遮蔽AA35.05%→36.15%、分隔符74.49%→75.36%提高，终止符79.24%→76.90%、关系标签74.12%→71.00%下降；native终止符99.22%→82.81%、关系39.06%→31.25%回退。best3=40000/27000/28000，latest与best重合，实际保留27000/28000/40000三份完整状态；14000及39000旧权重自动清理，共12份JSON档案核对通过。只读窗口66245自然结束后接续5765，原容器与训练未重启。下一41000步，配置保持。证据：[40000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint40000_root_review.json)。

## 2026-10-07 41000步验证与保存通过

41000步完整保存及指标核对通过，主eval_loss=2.011555，较40k回升，按主loss排名第三。全遮蔽关系标签71.00%→81.74%、终止符76.90%→80.58%提高，AA36.15%→35.73%、分隔符75.36%→74.49%下降；native关系31.25%→73.83%、终止符82.81%→99.22%回升。best3=40000/27000/41000，latest已在其中，实际保留27000/40000/41000三份完整状态；28000旧权重自动清理、六份JSON档案核对通过。原容器、训练及只读观察5765继续，下一42000步，配置保持。证据：[41000步报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint41000_root_review.json)。

## 2026-10-07 23:33 UTC 当前状态与观察间隔补核对

只读观察5765在17:29:45自然结束；23:33:32重新查询已到71395步、平台和训练均Running。原容器、训练PID及配置保持，占卡进程为空；未重新提交。42k–70k的29点保存元数据、2610项聚合与保留记录已补核对，旧已删除二进制不宣称复核；当前42k/45k/68k及最新71k完整文件哈希通过。最佳45k主loss1.992721，最新71k为2.011193；保留42k/45k/68k/71k四份完整状态。新只读观察26248，下一72000步。证据和指标见 [实验记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md)。

## 2026-10-07 72000步验收与旧no-ESMC曲线核对

72k完整保存通过17文件哈希、八rank和90项核心聚合核对；主eval_loss=1.999634，当前最佳仍45k。按best3＋latest保留42k/45k/68k/72k；71k权重由原生逻辑清理，六份JSON证据已核对。报告：[checkpoint72000_root_review.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint72000_root_review.json)。

已从原始Trainer记录整理两版训练和验证曲线。旧no-ESMC实际为V6 binding-only，当前为V5；验证mask、EOS目标和日志尺度不同，不能以原始loss排名。当前10k→72k主验证下降约1.41%，旧版自身验证同期约9.71%，仅为各自协议下的趋势。图表与限定见 [V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V5_CURRENT_VS_LEGACY_NOESMC_LOSS_20261007.md)。未改变正在运行的配置、未新增GPU实验。

## 2026-10-08 新模型下游评测提交

固定best45000，37任务/4组。首次queue012提交被平台拒绝：CreateCustomTask无权限，未产生作业；改用当前已授权c20250601队列的独立单卡评测资源。原八卡训练不变。

- submit：`t-20261008075358-bkn9q`，v5-native-best45000-all-g0-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075414-ldm2d`，v5-native-best45000-all-g1-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075418-pbx7q`，v5-native-best45000-all-g2-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- submit：`t-20261008075421-nsd4d`，v5-native-best45000-all-g3-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008.yml；初始Submitted，c20250601，Preemptible=true。

- 2026-10-08T17:40:22.173688+00:00 cancel：t-20261008075358-bkn9q（闲时g0，原YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:22.659598+00:00 cancel：t-20261008075414-ldm2d（闲时g1，原YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:23.443411+00:00 cancel：t-20261008075418-pbx7q（闲时g2，原YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

- 2026-10-08T17:40:23.817528+00:00 cancel：t-20261008075421-nsd4d（闲时g3，原YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008.yml），按用户授权切换非闲时；请求成功，等待终态，不重复提交。

## 2026-10-08 授权切换非闲时下游评测

4个旧闲时作业均已终止，未执行评测；终态证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/spot_terminal_states.json。使用原best45000、相同37项任务和划分，改为4个独立单卡非闲时作业，不修改训练。

- submit：t-20261009014055-jwbqq，v5-native-best45000-all-g0-dedicated-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g0_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014058-nwkcf，v5-native-best45000-all-g1-dedicated-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g1_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014102-gd7sw，v5-native-best45000-all-g2-dedicated-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g2_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009014105-lgdcg，v5-native-best45000-all-g3-dedicated-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_all_g3_20261008_dedicated.yml；初始Submitted，c20250601，Preemptible=false。

## 2026-10-08 17:40 UTC 训练过半与153000步核对

实际查询至153097/300000步（51.03%），容器及训练均Running；无可见报错，占卡进程为空。最新153k验证loss=2.008281，最佳138k=1.991442，比此前45k最佳仅改善约0.064%。验证仍在平台区间，不能宣称显著效果提升。最新完整保存核对通过；进度、实际耗时和验收范围见 [本次状态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/status_20261008T1740_root_review.json)。未改变训练配置、提交或重启任务。

## 2026-10-08 训练结束后恢复八卡占卡

按用户要求核实已有自动切换逻辑；无需重启容器、改训练或新增GPU负载。当前训练继续，standby为空。已启动本地只读监督PID1100460、会话90153，上限50小时；跟踪训练状态，并在训练退出后确认八卡占卡新进程和初始化日志，记录实际结果。监督准备及CPU边界检查通过，证据 [占卡恢复监督](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/setup_review.json)。此刻训练未完成，不提前称占卡已恢复。

## 2026-10-08 Kong失败修复并重提

用户授权重提失败项；只有Kong失败。独立重试入口将空轻链表示为缺失链，不删样本、不补造序列；CPU真实样本前向、隐藏答案隔离、双链路径一致性通过。原4组源码哈希不变。submit t-20261009015628-wqmrg；任务名v5-native-best45000-kong-retry1-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best45000_kong_retry1_20261008.yml；初始Submitted，Preemptible=false，c20250601。旧失败产物归档于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/attempt_history/cdr-kong_attempt1。

## 2026-10-08 全部CDR协议对齐

按用户要求，新增Kong三CDR/十折与SAb23H2六CDR官方design掩码评测，复用AirGen原解码逻辑；原iter1成绩保留历史。全量输入9381+360条核查通过；独立best45000评测快照，其他37项原作业代码哈希保持。规范见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md。

- submit：t-20261009020953-x4xxj；v5-airgen-aligned-cdr-kong-45000-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_airgen_aligned_cdr_kong_20261008.yml；初始Submitted，c20250601，Preemptible=false。

- submit：t-20261009020956-sj4dd；v5-airgen-aligned-cdr-sab23-45000-20261008；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_airgen_aligned_cdr_sab23_20261008.yml；初始Submitted，c20250601，Preemptible=false。

## 2026-10-09 下游失败修复

原四组评测平台状态为Failed/Failed/Success/Failed，两个AirGen对齐CDR作业均Success，均移出Active表。T1修复基线表路径，只恢复已有五折结果的比较表；CHIMERA隔离ANARCI单序列异常，保留候选及失败原因。本地运行评分恢复与TCR beta-only禁止dot的新协议复测；原始失败输出归档，302容量、learned protd、checkpoint及种子不变。22项回归测试通过。结果以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 为准。

本地收尾器已启动（scripts/downstream/finalize_v5_repairs.py），等待两个修复进程实际退出后再同步终态/RESULTS并删除CHIMERA novelty_tmp目录；未提交新云端资源，未修改训练进程。TCR当前已完成目标的部分进度见repair_audit_20261009.json，不能当作全量成绩。

## 2026-10-09 最新269000步评测准备

按用户要求固定最新完整checkpoint-269000（选择时刻2026-10-09T19:24:02Z），8 rank及六文件SHA通过，独立硬链接仅保留推理权重/元数据。验证最佳仍为138000步。37项任务沿用修复协议，CDR为AirGen对齐，输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_latest269000_20261009；本轮与45000分开记录，完整后释放评测权重链接，不保留探针ckpt。

2026-10-09 submit：latest269000 group0，task_id=['t-20261010032615-hpdkc']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g0_20261009.yml。

2026-10-09 submit：latest269000 group1，task_id=['t-20261010032618-5nptj']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g1_20261009.yml。

2026-10-09 submit：latest269000 group2，task_id=['t-20261010032622-nx9nd']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g2_20261009.yml。

2026-10-09 submit：latest269000 group3，task_id=['t-20261010032626-sdw56']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_latest269000_g3_20261009.yml。


### 2026-10-09 CDR iteration 对照

固定269000步开展CDR infilling iteration敏感性实验：1/2/4/8/16/32步，Kong三CDR十折和SAb六CDR全量，共12个数据集×步数设置、54个CDR对照格。独立权重硬链接防训练轮转，默认主评测继续；结果与逐样本预测保存于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations269000_20261009。

2026-10-09 submit：CDR iterations kong，id=['t-20261010033011-gjsw2']，returncode=0，非闲时1GPU，YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations269000_kong_20261009.yml；六种步数CPU检查全部通过。

2026-10-09 submit：CDR iterations sab23，id=['t-20261010033015-85bgb']，returncode=0，非闲时1GPU，YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations269000_sab23_20261009.yml；六种步数CPU检查全部通过。


### 2026-10-09 验证最佳138000评测

固定验证最佳checkpoint-138000，eval_loss=1.991442，8 rank与推理权重/元数据SHA核对通过。对齐269000运行37项全下游及CDR 1/2/4/8/16/32步敏感性分析。独立源码副本保持已有作业代码哈希不变；独立硬链接评测快照仅包含推理权重和元数据。目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best138000_20261009 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_cdr_iterations138000_20261009。

2026-10-09 submit：v5_native_best138000_g0_20261009，id=['t-20261010033209-q99gz']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g0_20261009.yml。

2026-10-09 submit：v5_native_best138000_g1_20261009，id=['t-20261010033213-6fkx8']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g1_20261009.yml。

2026-10-09 submit：v5_native_best138000_g2_20261009，id=['t-20261010033216-8j2s8']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g2_20261009.yml。

2026-10-09 submit：v5_native_best138000_g3_20261009，id=['t-20261010033220-q4fgp']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_best138000_g3_20261009.yml。

2026-10-09 submit：v5_cdr_iterations138000_kong_20261009，id=['t-20261010033224-djh66']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations138000_kong_20261009.yml。

2026-10-09 submit：v5_cdr_iterations138000_sab23_20261009，id=['t-20261010033228-m7rng']，returncode=0，Preemptible=false；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_cdr_iterations138000_sab23_20261009.yml。


2026-10-09 修复复测已终态：{"t1-cdr3b": "success", "ag-chimera": "success", "t4-benchmark14": "success", "t4-held20": "success"}。结果及CDR3语法有效率见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_best_20261008/repair_final_20261009.json；原始候选保留，MMseqs临时目录已清理。82项回归及60组T1 AUROC复核通过；未见表位泛化弱仍未由流程修复解决。


## 2026-10-10T02:07:51.760263+00:00 训练至占卡只读监督结束

监督结果=training_exited_standby_observed；训练票据状态=succeeded；八卡占卡恢复观察=True。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。未变更GPU进程或票据；完整训练产物最终验收另行核对。

### 2026-10-10 评测进度核对

45000修复后37/37成功；269000全下游31/37成功、2项运行、4项等待；138000全下游7/37成功、1项运行、29项等待，另三组平台Queue。269000 iteration 12/12设置完成，138000 iteration两个作业仍Queue。已完成的四个评测作业移出Active表。训练300000步已成功结束，验证最优仍138000，当前新评测固定269000/138000，不自动切换300000。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/progress_20261010.json。

## 2026-10-10 75.6M完成及270M准备

旧训练真实成功退出、COMPLETE为300000；最终17文件/八rank/90项指标核对通过，最终主loss2.003477，best138k=1.991442，保留138k/45k/190k及latest300k。容器仍运行且八个占卡已恢复。最终保存报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/checkpoint300000_root_review.json。

用户要求新训270M，已完成参数核实与独立配置；复用当前容器，不重复提交资源。配置 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/training_config.json；候选整模profile包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_profile_bs64_20261010。当前尚未发布新票据，不将准备完成称为开训。

## 2026-10-10 发布270M整模synthetic检查票据

向现有容器t-20261007153431-vhx2f发布一次v5_270m_profile_bs64_20261010.json；仅整模synthetic micro64×3步/每GPU及通信检查，不访问语料训练。五项CPU门禁测试与启动元数据核对通过。容器先回收八个standby再执行；本次无平台submit、无容器重启。正式训练尚未启动。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_profile_bs64_20261010/ticket_published.json。

## 2026-10-10 发布270M正式训练票据

v5_270m_scratch_bs64_20261010.json已发布到原持续容器，未平台submit/重启。八卡synthetic及NCCL通过，allocated峰值42.78797GiB、reserved48.87891GiB；8卡参数均268,590,112，3步有限loss，profile实际退出码0，无测试checkpoint。正式训练复用这份匹配源码/尺寸/batch的检查证据，不重复synthetic；原生数据全哈希门禁保留。每卡64、GA1、global512为同预算可用最大micro。参数和参考YAML见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010/README.md；发布证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010/ticket_published.json。

## 2026-10-10 270M正式更新步已确认

2026-10-10T08:14:43.885827+00:00实读67/300000步；训练loss=2.8957有限，8rank通信及85分片全内容哈希通过，真实训练峰值allocated=40.79GiB。监督会话26867已实际输出并持续运行；训练结束后的占卡恢复仍由原容器负责。启动证据 [270M正式训练](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/formal_start_review.json)。本阶段尚无1000步验证/保存，不宣称质量收益。

## 2026-10-11 270M训练进度与28000步保存核对

2026-10-11本次核实已到28082/300000步（约9.36%）；最新完整保存28000步/epoch1.8797，累计28次原生验证。主eval_loss（全遮蔽/source-macro）最佳17000步1.993521，最新28000步2.092312，近期波动，尚无持续下降结论。最近10000更新墙钟平均2.8864秒/步，包含窗口内验证/保存；不中断约余9.1天。 训练和容器均running，占卡为空，原只读监督仍运行。保留best3（17000/16000/14000）＋latest28000共4份完整状态，其余旧权重按规则轮换。28000步完整核对通过，未执行恢复重放；本次无提交、重启或训练配置变动。证据：[状态与计时](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261011_checkpoint28000_root_review.json)、[28000步核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint28000_root_review.json)。

## 2026-10-12 270M训练进度与60000步保存核对

2026-10-12本次核实已到60305/300000步（约20.10%）；最新完整保存60000步/epoch4.0279，累计60次原生验证。主eval_loss（全遮蔽/source-macro）最佳32000步1.993467，最新60000步2.007204；相对17000步旧最佳仅改善0.0027%，整体仍波动。最近10000更新墙钟平均2.8940秒/步，包含窗口内验证/保存；不中断约余8.0天。 训练和容器均running，standby为空，原只读监督PID1737482/exec26867仍运行。保留best3（32000/17000/43000）＋latest60000共4份完整状态，其余旧权重按规则轮换；正式恢复状态保留。本次无提交、重启或训练配置变动，未执行恢复重放。证据：[状态与计时](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/status_20261012_checkpoint60000_root_review.json)、[60000步核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/checkpoint60000_root_review.json)。

### 2026-10-12 下游评测全部终态

三组各37/37成功；两组iteration各12/12设置完成。核实12个新云端作业均Success并移出Active表；本轮没有待跑/失败的评测任务。所有质量结论与对照数字统一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md；不能将任务执行成功当作模型质量达标。


2026-10-12：澄清 CDR 主表版本：f011 50082 的 74.5438 / 68.6565 / 42.4700 为历史原生 A4、iter=1 成绩，非当前 V5 数据 native object302 权重；“两轮”指训练轮数。保留历史数值，在 Kong 主表补入当前 75.6M 三个 checkpoint 的 AirGen 对齐 iter=4 原始 metrics，并明确跨版本不构成受控消融。仅修正文档展示，未重新评测。


2026-10-12：用户确认 CDR 评测以 Ophiuchus-Ab 官方权重复现论文为验收目标，不以代码形式对齐代替结果验证。历史复跑与当前入口验收分开；当前端到端官方权重对照尚待完成。协议见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。本次仅明确验收规则，未新增评测或修改成绩。


## 2026-10-12T09:27:37.255160+00:00 270M 下游提交 g0

操作 submit；task_id=['t-20261012172738-sfpmb']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g0_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:40.815762+00:00 270M 下游提交 g1

操作 submit；task_id=['t-20261012172742-pbvpt']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g1_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:44.305878+00:00 270M 下游提交 g2

操作 submit；task_id=['t-20261012172745-kxt5s']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g2_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:27:48.146769+00:00 270M 下游提交 g3

操作 submit；task_id=['t-20261012172749-tx78f']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_best32000_g3_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，37项分四组，独立评测快照为硬链接；训练不受影响。


2026-10-12：270M 首次下游固定验证最佳32000步（epoch 2.14823，eval_loss 1.993467），四组37任务已提交，CPU真实权重CDR门禁通过；独立结果见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 。选点与平台状态见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v5_native_270m_best32000_20261012 。


2026-10-12：按用户要求先提交当前代码快照e8ad5fb（未push），再准备相同270M checkpoint32000的旧CDR入口对照。旧run_v5_cdr_eval及六个依赖SHA与45000历史manifest一致；原V7评测共用argmax和评分器，输入适配不同。协议差异见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md 。不更改运行中AirGen源码与成绩。


## 2026-10-12T09:40:19.756460+00:00 270M 下游提交 g0

操作 submit；task_id=['t-20261012174021-jrcbb']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_legacycdr32000_g0_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，9类CDR分两组，独立评测快照为硬链接；训练不受影响。


## 2026-10-12T09:40:23.536049+00:00 270M 下游提交 g1

操作 submit；task_id=['t-20261012174026-thrps']；YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/v5_native_270m_legacycdr32000_g1_20261012.yml；returncode=0；初始状态 submitted_pending_platform_query；Preemptible=false。固定验证最佳32000步，9类CDR分两组，独立评测快照为硬链接；训练不受影响。
