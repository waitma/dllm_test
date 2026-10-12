# dllm_test Project Guide

## 2026-10-07T06:11:39.860103+00:00 V5固定对象容量、自主生成分隔与终点

已完成用户要求：当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 中full_capacity=512，表示整个受体对象的容量（含`.`与`<protd>`），不再要求真实生成长度。固定目标头部；真实残基／分隔／终点监督，synthetic与尾部mask不计loss，512位均参与attention。CDR-only仍保持先前全长scaffold及mask补位，VHH继续不在本版训练范围。目标role/scope不按真实边界编码；denoise只按腐蚀后当前可见`.`分链和重置位置，保留可见链内／全局交替以及残差1.0。

训练入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py 使用 --input-layout V5 --variant cv_segment_global --object-capacity 512；统一生成及CDR入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，模型generate也接入同一解析器。支持整体生成与已知前链→生成后链；反向补链需另定条件协议，明确拒绝隐藏参考边界。无终点／非法链结构报告失败。此记录替代此前“真实长度／generation_lengths”的当前版本要求；历史A4及冻结训练不改。

83项测试通过，7来源1250条抽查通过，130513个synthetic补位保留；不是全量数据扫描。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_validation.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/canvas_regression.log，说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md。临时保存恢复测试权重已清理。未提交新训练，无本版下游成绩。

## 2026-10-07T05:55:11.432519+00:00 CDR-only TCR 保留全长并以 mask 补位

当前版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 partial_sequence_policy：只有 CDRβ／CDRα 的记录沿用先前 prepared 全长 scaffold、区域及长度，送入模型时按 synthetic_residue_mask 将缺失位置转为 <mask>；保留真实CDR及锚点，真实观测X不替换。全缺失配对链沿用已有scaffold全mask。补位参与注意力但labels=-100，不计重建损失，不重新采样长度或缩短为CDR-only。实现 /vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/layout.py，仅改变新V5布局；历史A4不变，VHH仍不在本版训练范围。43项测试通过；1250条真实数据抽查中620条含synthetic，共130513个补位检查通过。证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/scaffold_mask_validation.json 与 scaffold_mask_regression.log。本次未提交训练、未生成checkpoint。

## 2026-10-07 当前新训练版本排除 VHH

用户明确：本版不需要考虑 VHH。版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/version.json 已登记 training_scope：常规抗体 H/L、TCR 在范围内；nanobody_vhh 链和 nanobody／nanobody_antigen 任务不纳入本版训练、专门评测或验收。版本说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md 已移除当前训练格式表中的 VHH 行。通用 renderer／兼容性测试保留，不代表本版训练覆盖；本次只登记范围，不修改现有语义数据或运行时过滤逻辑，后续本版训练清单构建须落实排除。

## 2026-10-07 V5对象布局的新版本

按用户要求，新版本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/research_config.json 采用 `ablation=V5`、`cv_segment_global`。context在前，关系居中，受体对象内用`.`连接H/L或beta/alpha，无条件使用固定null前缀；恢复变长scaffold及V5结构token训练，synthetic不计损失。保留紧凑词表、无ESMC、残差1.0、整链/全局交替。全部7来源复用最终语义JSONL，通过collator转换；不改历史A4配置。训练入口显式 `--input-layout V5`，下游新适配 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v5_layout.py，生成必须显式声明长度。配置与格式说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v7_v5_layout_chain_global_20261007/README.md；验证 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/validation.json。尚未提交新训练，无新checkpoint或下游成绩。

## 2026-10-07T00:39:13.894414+00:00 两轮生成评测最终证据与禁止重放的已完成任务

本轮三类生成报告：/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_tcr/report.json。TCR原始输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_all_local_gpu_20261006；评分 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_scores_cpu_20261006；评分真实日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_tcr50082_score_cpu_20261007。GPU生成和CPU评分均已实际exit0，不重放；保留原始候选/控制/评分和来源，未产生新checkpoint。原GPU资源继续原12消融票据，现场 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/pool_after_TCR_scores_20261007.json。

## 2026-10-06T23:50:17.965727+00:00 当前原八卡平台查询与T4比较证据

原平台任务名immune-v7-pool-8gpu-20261005、ID t-20261005205851-lqfjg。查询该任务用 volc ml_task list -n immune-v7-pool-8gpu-20261005 --status Initialized,Queue,Staging,Running,Killing -o json --limit 100；CLI会先打印版本更新提示，应保存输出并解析JSON正文。旧bioseq名称过滤及跨用户最近100项不构成当前任务不存在的证据。最新实际原任务查询 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/existing_pool_platform_status_by_name_20261006T2348.json；元数据临时listing清理 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/metadata_lookup_cleanup_20261006T2349.json。T4既有V5比较依据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/v5_comparison_binding_v1.json，不得重评或复用旧训练seen/unseen标记冒充V7。

## 2026-10-06T23:42:54.045644+00:00 当前固定500消融图路径

新500点图/CSV/JSON在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new，根验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_step500_eight_new/root_review.json；18/22 PARTIAL。旧3750图仍 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1_f010/step3750_architecture.png（10/22）。后续继续原12票据，不重发或改正在使用的dispatcher；等新增3750真实产物与退出再扩完整图。

## 2026-10-06T23:39:56.588650+00:00 T4单次运行和后续纯CPU评分路径

现有任务v7_tcr50082_all_local_gpu_20261006，PID397587/start2459151386，日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_tcr50082_all_local_gpu_20261006，输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_all_local_gpu_20261006。实际GPU核查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_gpu_preflight_root_review.json；禁止重跑CPU/GPU验收及生成。后续评分入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_tcr_generation.py、计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scoring_plan_v1.json、独立输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_scores_cpu_20261006。activate pllm后使用计划固定环境及worker-compatible Python，须先取得真实预测exit0；当前评分未运行。

## 2026-10-06T23:35:21.048373+00:00 轻链完成产物与T4纯序列评分辅助入口

轻链最终报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_pairing/report.json、根验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/score_actual_completion_root_review.json；该attempt已完成，不重复启动。评分自有cache清理回执 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006/owned_cache_cleanup.json。T4评分模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_tcr_scoring.py，人工验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/scoring_controls_root_review.json；实际T4尚未评分，GPU preflight仍须独立完成。

## 2026-10-06T23:24:18.465554+00:00 当前轻链评分实际进程

原完整预测已正常结束，不能重放。评分PID346252/start2459060767，tmux v7_pairing50082_score_local_gpu_20261006，实际日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_pairing50082_score_local_gpu_20261006，输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006，根观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/score_local_launch_root_observation.json。当前评分源码/固定计划/复制的评分源运行中禁止修改，不能重复启动；用日志exit_code/done/failed及输出COMPLETE/metrics判断真实结束。仅成功退出后清理自有datasets_cache及无需留存的中间CSV，不动共享模型缓存或正式恢复状态。

## 2026-10-06T23:18:02.210187+00:00 TCR全量预测计划路径

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_tcr_generation.py；固定计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_generation_plan_v1.json；预算/输出人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_artifacts.py 与根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/full_generation_preparation_root_review.json。当前仅prepared_not_launched，GPU preflight尚不存在，不得伪造、复用CPU设备证据或重复启动。完整输出PREDICTIONS_COMPLETE仍仅表示3434原始行已封存，评分和真实退出需另核实。

## 2026-10-06T23:12:31.503718+00:00 T4真实原生输入验收入口

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/preflight_v7_tcr_generation.py；CPU已完成的输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_tcr_input_audit_cpu_20261006，根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/actual_cpu_preflight_root_review.json。该CPU attempt已通过，不重复；GPU必须独立新输出与真实设备准入，CPU证据不能替代。当前完整T4生成入口/评分仍待接续。此入口不训练、注释、提交、重试或写checkpoint。

## 2026-10-06T23:07:56.130738+00:00 V7 T4准备入口和证据

原生输入/解码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_tcr_generation.py；人工检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_tcr_generation_adapter.py；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/synthetic_root_review.json 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_tcr_adapter_20261006/benchmark_inventory_v1.json。真实模型运行入口仍待整合，不得把合成检查说成50082输入已验或T4成绩。复用的pairing_decode文件仍受运行中轻链绑定保护，不能热改。原8项首250验证根证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/first250_root_review.json。

## 2026-10-06T22:59:41.079874+00:00 V7轻链单独评分入口

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/score_v7_pairing.py；固定依赖和环境 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/scoring_plan_v1.json；准备检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/scorer_preparation_root_review.json。当前尚未执行评分，未来输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_scores_local_gpu_20261006。不能绕过原生成runner真实退出和4000候选完整性检查，也不能将预测COMPLETE当评分完成；同一attempt不覆盖/重放。评分成功后可清理该评分输出下自有datasets_cache，保留逐候选评分和源CSV，不动共享模型缓存或正式checkpoint。

## 2026-10-06T22:54:53.864578+00:00 轻链完整预测入口和当前证据

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_v7_pairing_eval.py，artifact检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_pairing_artifacts.py；实际输入/单次启动证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/actual_input_and_full_launch_root_review.json。全量输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/v7_f011_step50082_pairing_local_gpu_20261006；PREDICTIONS_COMPLETE仅代表候选完整，不能替代评分或runner真实exit_code。当前PID339584/start2458869800，日志 /vepfs-mlp2/c20250601/251105016/project/dllm_test/logs/v7_pairing50082_local_gpu_20261006；禁止重复启动或运行中修改绑定代码。canonical基准清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/benchmark_inventory_v2.json。

## 2026-10-06T22:37:03.056870+00:00 V7轻链适配准备路径

新模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/v7_pairing.py；新人工CPU检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_pairing_adapter.py；实测证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_pairing_adapter_20261006/synthetic_root_review.json。真实运行尚未接通，任务协议唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md §4.6。运行时必须先用原生loader隔离到实际checkpoint绑定的冻结模型源；禁止直接把新模块导入现有GPU训练进程。新CPU检查未写任何checkpoint，旧正式恢复文件和必要生成证据继续保留。

## 2026-10-06T22:28:30.597931+00:00 两轮完成后结果与消融交接路径

最终正式恢复目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run，当前best3 45000/46000/47000＋latest50082完整保留；两轮终态根核查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/segment50082_root_review.json。九项CDR已完成，汇总 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr/report.json，排行榜唯一 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md#ab-cdr；执行时EVALUATION_READY中的pending是当时快照，不代表当前评测未运行。

后续消融唯一实际批次 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/catalog.json，入队 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/enqueue_receipt.json；不能重放publisher。根进度以原pool的running/results及worker心跳核实，现阶段无新常驻监督。最终报告用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/mixed_report_selection_with_local_exit_v1.json，已接入本地f010真实rc0，旧f000退出码仍未知；不修改任何冻结选择或运行源码。

## 2026-10-06T22:17:06.607485+00:00 本地f010当前已完成，真实退出与图的路径

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/terminal_review.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/actual_exit_receipt_v2.json 记录实际rc0/3750完整epoch；旧80642已退出，不得重放启动器。当前已含真实exit的选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/mixed_report_selection_v3.json，最新固定3750报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_20261006_epoch1_f010/step3750.json，图同前缀_architecture.png/SVG/PDF。10/22产物、9成功退出、f000/42未知、12未启动；未来pool组合选择须另建新文件合并本次exit ref，旧冻结选择/publisher保持。完整核查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006/root_full_epoch_review.json。


## 2026-10-06T22:01:12.787358+00:00 当前本地f010终态记录与不可变选择文件

conda pllm下的单次终态记录入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/record_v7_f010_local_completion.py，准备证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_terminal_recorder_preparation_20261006/root_preparation_review.json。只能在已有runner实际完成后读取并保存真实rc0/done、同boot原PID身份和3750原生产物，不能启动训练/替代缺失凭据。预期新目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/f010_seed42_terminal_evidence_20261006 当前不存在。旧mixed_report_selection_v2和已冻结后续publisher保持；最终混合报告显式从新selection接入真实exit引用，不改冻结输入。f000/42没有实际退出证据这一限制保持。当前位置/下一节点看 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。


## 2026-10-06T21:49:54.047686+00:00 V7两轮生成收数入口与唯一结果文档

收数入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/collect_v7_cdr_results.py，准备证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_result_collection_v1/root_preparation_review.json。输出只允许新建在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/v7_f011_step50082_cdr 下，必须先有当前恢复50082与9项实际生成完成验收；当前未创建成绩。脚本不改RESULTS，不运行模型或重评分。根端核对实际JSON后仅在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md 发布数字，协议/命令见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.5；进展另写PROGRESS/PROJECT_PROCESS。V5历史log不冒充逐样本重算，iter1和iter2差异必须显示。


## 2026-10-06T21:44:32.310631+00:00 worker心跳字段按实际状态读取

当前八张卡running记录使用heartbeat；闲置/standby_load记录使用time，依据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py 的worker函数。只读汇总应依次读取实际存在的heartbeat/time；两者都缺失时表示启动过渡、尚无可核实心跳，不填0推断失败或重启。此前说明“只用time”不适用于running。

本轮MD同步的首次52c9c5在写入前因直接读取time而KeyError，训练未受影响。读取八份实际running原件后改为heartbeat/time兼容，再次同步b8ee70 rc0；47000原生验收97030c没有重跑，模型/公开入口/worker源码均未改。


## 2026-10-06T21:37:34.475996+00:00 生成后消融当前准备包v2已核查、未部署

当前入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/source_v1/publish_v7_post_generation_ablation.py；只读清点命令为 conda pllm 中 python -B 上述绝对路径 --action preview。冻结包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/package_v1.json（SHAff3ee6d8617a1621522b4cadf1f9d81259e4bf138ed7e905979bc4f28e2a7865）；检查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/root_cold_review_v1.json。当前只有准备与检查，无新监督/发布/claims/队列。旧post_generation_pool_v1保持原字节和未发布状态，不能执行其旧终态绑定。

新文件门禁仅接受 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/COMPLETE.json 的50082/前序50058和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/COMPLETE.json 的9项真实生成完成。原失败段的46项冻结包绑定及8失败退出/清理仍只读核对；原SUPERVISION_STOPPED保留，旧路径新增STOP/STOP_NOW继续拒绝。worker不探测前端PID；根发布器检查当前与旧监督实际退出、空闲/资源截止、共享cell_locks及最新22格。所有adapter函数与旧冻结版本AST相同；模型/source/data/执行主体不变。禁止绕过门禁、重新提交已完成/正在跑或失败的格子。


## 2026-10-06T21:32:18.621472+00:00 后续消融接续路径更新中

当前私有准备包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v2/package_v1.json，旧 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/package_v1.json 冻结不变。新入口只接受 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/COMPLETE.json 的50082终态（前序50058）及 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/COMPLETE.json 的9项真实生成；同时核对原失败记录不可变。worker只读文件核实，不在GPU进程命名空间探测前端PID；根发布器单独核实监督退出。当前检查未完、未发布/入队，不能执行旧包。



## 2026-10-06T21:24:31.178216+00:00 启动锁事故后的当前执行状态来源

原失败50050段及旧训练/生成监督全部保留，不能编辑其abort或STOP来继续。此次独立版本的新状态唯一读取 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/supervision_state.json；实际run/模型/日程来源仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/run 和原615项冻结源，8步恢复已通过。新监督源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/source_v1/supervise_v7_recovery.py，单次启动记录存在后不得再启动。后续新失败依旧停止，不自动复制新版本重试。

两轮原生完成指针属于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/COMPLETE.json 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/EVALUATION_READY.json，不是旧已停EPOCH监督目录。新生成状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/state.json、执行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/recovery_v1/generation_pool_v1/package_v1.json；旧生成PID234066退出并保持原件。9项原生解码协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4。未运行的后续消融发布器仍需绑定此新完成链，不能使用旧prepared包越过实际训练/生成终态。

读取checkpoint保留情况以原callback的checkpoint_retention及实际目录为准；45050权重已在恢复后正常轮替，JSON/24项原生恢复证据另存，不为了预检强留旧权重。实时根汇总 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_observation_latest.json。原安全停线、序列注释禁启、数据门禁、独立CUDA暂停不变。


## 2026-10-06T14:15:40.813569+00:00 后续评测入口发布约束

当前自动评测包为epoch2_continuation_20261006/generation_pool_v1/package_v1.json，前端为generation_source_v3。监督先等待50082真实终态及进程退出，再归档并扩展公共入口/注册；这两个文件经查不在冻结模型source_manifest内。原pool生命周期/时限不变，兼容队列外壳train/steps1实际为generation_evaluation、optimizer_updates0。 绝对路径、真实进程与门禁见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。

## 2026-10-06T13:54:38.042567+00:00 V7原生评测入口和包隔离

原生no-ESMC适配位于downstream/v7_native.py，避免grammar/__init__.py提前引入fusion；CDR纯产物评分器按绑定文件载入。不得使用generation_source_v1；当前有效计划是epoch2_continuation_20261006/generation_plan_v2.json。 绝对路径及验收状态见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md；任务协议唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_CDR_INFILLING.md §4.4。

## 2026-10-06T13:38:54.423749+00:00 两轮接续执行配置

冻结模型仍f011/where-only/math，第二阶段采用显式LR重启与跨轮sampler修复；已部署等待父训练自然完成。源码和执行包、真实验收边界唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md。当前实际新阶段GPU验收及生成评测未完成，禁止把待命当训练启动或完成。

## 2026-10-06T13:21:35.490096+00:00 用户新增两轮后生成评测；准确节点50082

当前冻结运行仍到25042。实际25041步/epoch，两轮为50082；需要再接25040步，纠正此前50084估算。CPU发现并修复跨轮skip loader重置epoch的问题，修复后原生恢复/样本顺序/最终模型与连续两轮完全一致，临时权重已清理。独立分布式接续与生成入口核实仍进行中。计划和证据唯一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_TWO_EPOCH_GENERATION_PLAN_20261006.md；原Goal身份保持，完成须包括新目标与原消融，未提交/重启任务或降低门禁。

## 2026-10-06T12:57:45.724638+00:00 新版DDP元数据检查入口

新点使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_where_checkpoint.py --step NEW --previous PREVIOUS，起点为已验收8000，后续只接受相同where运行来源；不会覆盖旧验收。读取该run的保留前JSON归档，旧7000迁移证据保持独立。无需GPU或加载权重，不改变训练或门禁。

## 2026-10-06T12:52:51.652079+00:00 新代码正式运行与保留策略当前状态

2026-10-06T12:52:51.652079+00:00：新where-DDP当前8278/25042，8000步验证/完整保存已核实，段目标12016；持续监督PID212174/start2455180199和8worker正常，abort=null。best3 7000/7008/7016＋latest8000共4份完整恢复状态。300步均值0.740862秒（稳态含数据/DDP，排除验证保存；旧10.2–10.7秒），原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/speed_300_updates_root_review.json。新版模型仅where等价展开，仍micro32/GA1/world8/global256和math attention。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_WHERE_DDP_CONTINUATION_20261006.md。

原7000完整checkpoint接续，旧最后打印7600后的约600个已打印未保存更新已重算；不冒称原进度无损。新段7000→7008、7008→7016通过实际八rank原生恢复/保存/退出，当前长段7016→12016仍在训练。最近段退出验收7016与最新完整checkpoint8000是不同概念。旧监督PID3977587已停，不重启。下一常规验证9000；原注释停线、CUDA暂停及数据门禁保持，不重复提交或重启容器。

本地f010/42仍运行，新增1500点已核对：平均AA改善、全遮蔽和END有回退；下一1750。22格仍9完整epoch产物、1运行、12未启动；500图10/22、3750图9/22均PARTIAL。原Goal持续active，不缩小范围或宣称全部完成。

## 2026-10-06T12:46:41.042969+00:00 两节点FSDP独立文件已同步best3＋latest

新交付 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml（SHA13b09c34a0bcbffae644f1e52c6cdbea8cab053f5c8db14f2f60986c66727274），未提交。仍是where-only f011、FSDP1 FULL_SHARD、2节点×4卡、micro32/GA1/global256、25042步、动态长度上限2048；目标mem_peak仍待目标机器合成验收。改动仅完整checkpoint保留策略：主eval_loss最小3份＋最新取并集，最多4份，完整optimizer/scheduler/8rank RNG保持。原生全部rank保存回执通过后才由rank0归档JSON并删本run非保留点；禁用Trainer的latest3提前删除。新增FSDP sidecar的bytes＋sha256格式适配，活动DDP冻结helper未改。

10项临时目录检查通过，包含实际新FSDP入口对原生sidecar格式的保留/删除、错误size拒绝；去除声明的保留策略差异后，新旧训练入口AST完全一致。618源码绑定、3个缺失源码拒绝、隔离CLI help和YAML/bash检查通过。没有重复模型/GPU验收；自有测试checkpoint已自动清理。沿用此前CPU/Gloo模型验收的范围，目标2×4 GPU/NCCL仍待该任务实际启动；先执行原合成4→8保存恢复与2048长度显存验收，再执行原数据审核，未降低门禁。

源码清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/source_manifest_v1.json（SHAbe1d78cba6349b4d4ef9d212efc1a1c41578d90e5fd86136b62cb63a443cb9fb）；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/root_review.json；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_BEST3_2NODE4GPU_20261006.md。原where-FSDP文件与原始DDP冻结源保持，当前容器仍按既有where-DDP持续训练；本文件为另一机器独立训练，不直接读取DDP checkpoint。

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


## 2026-10-06T12:21:04.640757+00:00 where同DDP接续的停止前检查已通过

8项best3+latest临时目录控制通过（9466e9 rc0）；CPU旧模型checkpoint→where模型原生恢复通过（6a4a49 rc0），恢复optimizer/scheduler/RNG、最终step4与全部optimizer step4，旧checkpoint未变，较连续旧实现最大参数差7.64e-11。首次CPU测试因独立导入模块缺少包名失败于模型构造前，改为正确包内名称后通过，两个日志保留。自有临时checkpoint均已清理。

613个候选源码绑定、原只读父数据审核、允许的模型单表达式变化和8种不兼容元数据拒绝检查通过；原checkpoint7000全部17个native文件重新核验，仍可完整恢复。预停记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/pre_stop_review.json；新冻结源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/source_snapshot_v1；原冻结源不改。新8步接续plan已准备未入队，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/where_continuation_20261006/launches/segment_7008/plan.json。

接下来按用户明确授权停止旧训练子进程并在同一容器由7000→7008接续；平台容器保留。当前没有立即保存接口，7000之后未保存更新将从该完整恢复点重算，实际停止日志步数会单独记录。全部八rank退出清理证据齐全才允许新8票入队。此处CPU通过不等于新八卡恢复或吞吐通过。保存策略与新旧来源迁移保持显式记录；原安全停线和审核门禁不变。


## 2026-10-06T12:08:05.749567+00:00 用户授权原容器内where接续及best3＋latest保存策略

用户明确要求从现有checkpoint接续新代码并监察真实训练速度；允许对此次切换受控停止旧训练，但不重建容器或重复提交资源。准备独立DDP接续版本，保持world8/micro32/GA1/global256、数据/目标/精度/25042日程；保留原冻结源码和旧provenance，以显式代码迁移记录连接旧checkpoint与新版本。旧运行尚未停止，最新完整恢复点仍7000，实际切换时重新核实。旧进程没有立即保存请求接口，恢复点之后未保存更新不能声称已保留。

保存策略按已有主eval_loss最小的3份＋最新完整恢复点取并集，重合不复制；全部保留完整优化器/调度器/各rank RNG，先完成八rank保存验收再轮换，只处理明确所属目录，删除前归档JSON证据。现有代码仍是latest3；新增模块 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/native_best3_latest.py 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_where_transition.py 准备中，尚未测试或部署。更早已被旧策略删除的权重不可追溯恢复。

两节点FSDP文件当前仍是上一版本，best3接入需独立更新后明确交付。下一项是接续/轮换CPU检查和完整启动包，再使用已有容器接续；不恢复注释分支、不修改审核证书或绕过门禁，独立CUDA专项保持暂停。


## 2026-10-06T11:57:12.223404+00:00 链偏置修复版2×4 FSDP已完成CPU集成验收

已封版 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_fsdp_2node4gpu_20261006.yml，未提交。f011架构与目标保持，仅将chain_bias重复索引改为where广播选择；原attention后端不变。已有真实V7单卡均值6.6288→0.5450秒定位了主要模型计算瓶颈，不代表原八卡已经提速；原正式/消融冻结实现未替换。

新接入CPU合成loss/全部梯度、四mask、原生Trainer保存恢复与模型重载通过；2-rank CPU/Gloo FSDP实际分片、重计算、梯度更新、评估及完整模型/优化器往返通过。自有临时checkpoint已删除。CLI源码绑定与拓扑拒绝、YAML/bash检查通过；两节点GPU/NCCL及目标mem_peak仍待实际任务，未重复GPU验收。

2节点×4卡，micro32/GA1/global256，25042步；默认c20250601/每节点ml.pni2.14xlarge，合成4→8步原生恢复与2048长度验收通过后再走原数据门禁。独立训练输出，不接续/覆盖当前DDP。原FSDP YAML/source_manifest_v2亦保持原样。

交付说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_2NODE4GPU_20261006.md；根记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/root_review.json；新manifest SHA c4105f8708c235b0e8c211f9a891c46d74049f815dfdbe57275b299203db3aff。停止注释分支、独立CUDA暂停、原审核及恢复约束保持。此次变更是工程性能修复，不能据此宣称下游或生成效果提升。


## 2026-10-06T10:41:11.204701+00:00 原DDP新checkpoint的只读复核入口

/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_formal_checkpoint.py 接受 --step 新点 --previous 已验收点，只读原正式目录JSON和原生文件大小，不加载模型、tensor或原始训练数据；若该新点结果已存在则拒绝重跑。仅适用于原native DDP的17文件/8rank schema，独立FSDP不可混用。每次新点复核后同步PROJECT_PROCESS和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md，不据保存完成推断训练终态。7000首次真实执行f4b88e rc0。

## 2026-10-06T10:29:39.027411+00:00 两节点四卡FSDP文件完成，CPU通过，GPU验收待新任务

已准备 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_fsdp_2node4gpu_20261006.yml；独立入口沿用冻结f011，FSDP1 FULL_SHARD，2×4卡、micro32/GA1/global256、25042步。默认c20250601/每节点ml.pni2.14xlarge，目标GPU mem_peak待profile。YAML按合成保存/原生恢复与全长显存验收→自有合成ckpt清理→原数据审核→正式训练执行；没有提交或启动新GPU工作。CPU数值/尾批/保存恢复和真实2-rank CPU FSDP已通过，不能称两节点GPU通过。原DDP、消融与冻结模型不变。

详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_2NODE4GPU_20261006.md；最终根复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_2node4gpu_20261006/root_review_v2.json。新增整模型分阶段计时/有界trace帮助定位慢化，当前主因仍未实测确认，独立CUDA专项继续暂停。正式恢复状态保留，自有CPU测试临时权重已清理。

## 2026-10-05T18:33:58.003212+00:00 未来本地guard与统一认领路径

私有本地入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/local_metrics_only_ablation_dispatch.py 的prepare只读，main会在持锁时exec原冻结CLI；已通过人工检查，尚未用于真实接续。基础managed plan不变，新tmux命令只将训练尾部替换为显式admission绑定的guard，底层training_argv原样保留。真实启动后使用v7.local.tmux_launch.v2/exit.v2和guard_execution；旧f000不迁移、不重启，也不伪造退出码。

未来各版本的不可变认领账本/admission统一放在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/shared_claims_v1（本轮未创建），所有local/pool版本共用其中cell_locks/<variant>_seed<seed>.lock；本地还共用local_gpu0.lock。不得按任务另建互不关联的锁目录。发布前核实新task/log/output身份、最新已完成/运行格子和原数据/来源条件；guard本身不建账本或重试任务。八卡复用另须原正式25042终态和清理完成。完整说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md；检查原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/local_guard_root_review_20261005.json。


## 2026-10-05T17:12:40.120217+00:00 后续消融私有接入路径

/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py 为未部署的显式catalog/ledger适配器；其prepare只读，main会exec冻结训练CLI，当前不得作为正式训练启动器调用。完整状态与剩余接入条件见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。后续混合local/pool使用须先让本地启动采用同一格子锁；当前本地进程不重启、不迁移，公共入口与全部冻结训练源保持。实际当前监督仍由 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py 唯一负责。

## 2026-10-05T16:30:57.845872+00:00 正式结果与周期checkpoint记录

正式周期验证汇总记录在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_TRAINING_RESULTS_20261005.md；第1000步根端观察 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint1000_root_observation.json。原生checkpoint的JSON由唯一监督器归档到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence，权重仅保留正式run的原生恢复状态，不另复制。周期保存通过不能替代对应恢复或整段退出证据。正式默认eval_loss是全遮蔽来源等权总loss；结果表需区分source_macro/token_micro及mask_mean/mask100，不能直接拼pilot消融。

## 2026-10-05T15:55:59.001982+00:00 独立混合来源消融报告入口

入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_mixed_ablation.py；使用绝对路径 --selection、--output-prefix，并显式选择 --endpoint-step 500 或3750。当前真实选择 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_selection_v1.json，产物 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/mixed_report_v1。旧单pool严格比较器保持；新报告分开记录产物完整、真实退出与来源一致性，不造pool结果/退出码，不降档或挑最佳步，不提供跨环境速度图。provenance源码字典不等于逐模块实际执行证明。

后续tmux启动凭据schema为v7.local.tmux_launch.v1，记录training_argv、完整command、runner引用、task_name、log_dir、started引用、source_manifest及SHA、checkpoint_saving=false；退出凭据v7.local.tmux_exit.v1绑定同一launch/runner/task/command/started及实际exit_code、done/failed。字段格式见上述脚本docstring。当前f000不重启或补造退出；新任务仍需全新log/session/output并检查真实前项结束。

## 2026-10-05T15:22:05.935892+00:00 后续本地消融的真实退出记录

后续新本地任务使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_in_tmux.sh 等实际等待子进程的既有托管方式，保存真实exit_code与done/failed并绑定启动argv、环境及冻结源码。全新task/log/output身份避免覆盖旧哨兵；不迁移或重启正在运行的f000。原生COMPLETE和PID消失不代替真实退出码，缺项如实标记。已准备但未执行的13项命令 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/managed_local_launch_plan_v1.json；汇总来源与完成状态分开的约定见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_REPORT_INTEGRATION_20261005.md。测试仍metrics-only，正式恢复checkpoint另存。

## 2026-10-05T13:15:45.137987+00:00 真实八卡首段8步完成，单卡补消融已并行启动

8个正式pool result均 succeeded/rc0，8rank实际参数digest相同，checkpoint-8包含模型/Adam/scheduler和8份rank RNG，SEGMENT_8为真实累计8步。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/root_segment_acceptance.json；保存恢复只完成“保存”，下一段必须实际恢复至16步后才能记原生恢复通过。rank0 train_loss=3.7763006687164307（本rank训练均值，非全局重算均值），训练调用实际耗时100.266秒。HF因固定horizon提前停止而打印的全量吞吐不能使用；8步/实际耗时=0.079788 updates/s，名义2048记录/100.266秒=20.426 records/s（含该调用内评估/保存）。

本地A100独立新 f000 seed42 已启动，PID3970535实际存活、argv匹配；相同原worker解释器 Torch2.8.0/Transformers4.48.1/NumPy2.1.3，源另冻结为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/source_snapshot_v1.json。120k/3750步、micro32/GA1/global32/eval4，metrics-only，不resume旧失败、不覆盖原账本；实际初始化SHA当前为 44ff26ac8144ece78c84d9660e2d40d8f0746fa4e555a9ae2afeaadbd7d4b6a2，完整epoch尚未完成。启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/launch_receipt.json。新八卡继续正式训练，本地消融并行，不再新申请GPU。


## 2026-10-05T13:11:04.146916+00:00 f011 首段八卡联合训练已实际入队

新容器 t-20261005205851-lqfjg 内已一次登记8个正式成员，初次观测 pending=0、running=8。初步f011（R26+R24）、seed42、每卡micro32×GA1×8=global256；同一完整horizon25042、LR1e-4/cosine、warmup1000、每1000步评估/保存，首次实际累计终点8步。正式目标约一遍6410749条训练记录；drop_last及epoch尾部实际曝光以Trainer记录为准，不虚称完全无重复/无遗漏。通过首段完整保存和8rank退出后，恢复至16步，原生optimizer/scheduler/每rank RNG及真实续步须通过原冻结driver的检查。

正式数据为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002，原v3/611来源与限定scope双证书保持并在runtime再检查。当前正式采用自然epoch随机采样（context约4.65%），不是pilot的20.54%context配方；context序列重建CE为0，已有正例关系token监督保持，无新增负例。初步选择依据为实际完成两seed f001/f011，不宣称完整22消融已完成。

计划原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/plan.json，8个具体job ID在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_8/enqueue_receipt.json；新容器实际parent七字段/完整argv已由唯一metadata-only job成功捕获。复用旧已冻结slot wrapper/transport与原正式Trainer，没有更改模型、数据审核或保存恢复门禁。正式输出独立为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_initial_training_20261005/run，保留原生恢复checkpoint，测试清理器不覆盖它。

目前仅确认实际入队/worker进程，真实优化步、参数同步、保存和恢复尚待日志/产物验证；不得记作验收通过。未新submit/cancel平台资源或重启容器/worker/监听，停止注释和独立CUDA专项暂停保持。上一Goal回合为progress（新容器实际Running且8worker就绪）；本回合推进正式任务实际启动，原Goal保持active。


## 2026-10-05T13:06:30.253153+00:00 新容器正式入口接入，真实父进程身份采集已入队

新池8个worker均实际待命且心跳<15秒。已冻结新薄入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/pool_formal_20261005_dispatch.py，仅重绑定新pool/catalog/8别名，复用原冻结正式 prepare/main、wrapper、transport及v3/611训练来源；旧未知任务分发保持。三项必要冷检查通过，无模型/GPU重复验收。

只入队一个实际容器元数据采集 job `1791205590250592385_p06_v7_pool_identity_bed2f0ab`，在新容器读取真实parent PID/start_ticks/boot/argv，结果用于绑定八卡正式计划；它不执行Torch、模型、CUDA专项或读取训练分片，不能记为GPU训练通过。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/identity_enqueue.json。未新增Volc提交或重启资源；正式模型仍未启动。


## 2026-10-05T13:01:01.223411+00:00 新八卡容器入口（空队列）

任务 `t-20261005205851-lqfjg` 已 submit 并实际 get 为 Running；c20250601 非闲时、1×ml.pni2.28xlarge（8×A100-80GB），7天上限、无自动重试。资源配置 [eight_gpu_pool_20261005.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/eight_gpu_pool_20261005.yml)，新池 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20261005；正式输出后续单独绑定，不复用旧池路径。worker 沿用原 protenix_abtcr 的已验证解释器，入口先激活 pllm；只启用原有 maintenance_only 待命负载，实验队列为空，不能把占卡称为训练。新任务按用户“先申请容器”执行，未重启原失败容器。训练 batch/峰值待各实际训练配置，资源占卡不记为模型 profile。


## 2026-10-05 当前执行与产物保留规则

初步 V7 版本为 `f011`（R26 加 R24 固定残差缩放），正式训练优先，完整消融仍保留待办。当前决定和真实状态见 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_INITIAL_TRAINING_20261005.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_INITIAL_TRAINING_20261005.md)。原八卡容器已经终态退出，不重启或重复提交；新资源和输出必须独立。

测试只记录指标/曲线/必要日志与来源，不保留 checkpoint/tmp。2026-10-03T09:09 的明确用户要求覆盖之前保留 p04 latest.pt 的历史策略；这些测试权重已清理，不可声称可以恢复。正式训练须保留必要原生恢复状态，输出不可被测试清理器覆盖。停止的序列注释、独立 CUDA 专项暂停和所有原审核/ready/来源/实际恢复门禁不改。

## 2026-10-03T09:09:20Z 当前架构测试的结果保留规则

政策原件：[仅保留测试结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/policy.json)。

用户已明确要求测试只保留中间结果，不保留 ckpt/tmp。当前 p01/p02/p04 架构/消融测试保留 curve.jsonl、summary.json、progress.json、run_provenance.json、COMPLETE/SEGMENT、日志及图；模型/optimizer/RNG 文件不保留，旧恢复点保留规则被本条覆盖。失败测试记录保持真实，不因没有模型文件补写恢复或完成证明。新公共测试入口默认 checkpoint_saving=false，只有用户以后明确需要时才使用 --save-checkpoints；metrics-only 测试不能断点恢复。

在跑的冻结代码未改，现有输出由 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/cleanup_test_artifacts.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/cleanup_test_artifacts.py) 的单实例守护每 10 秒清理，最长 24 小时，pool 停止后自动退出。仅 p01/p02/p04 测试 run 输出在范围内，保留近期或活跃 writer 的临时文件直到写入结束；陈旧 monitor 临时文件超过 15 分钟清理，不删除模型输入、数据、来源快照或结果文件。动态状态 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/state.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/state.json)；全部删除记录 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/deletions.jsonl](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_checkpoint_metrics_only_20261003/deletions.jsonl)。正式训练入口的 checkpoint 策略仍独立管理。

## 2026-10-03T04:32:08.720556+00:00 500实际证据与未完成实验接续边界

[500概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/v7_large_batch_ablation_overview.png)；原报告 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared500_20261003/report.json（SHAd8d1ae3b9c8f6e33592f5c7478a494ad944826fe61ed4b579878054ef0b7ac94）；[12图及实际渲染归档](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render500_20261003.json)（SHA9d19428788fad9579a9196c1c0053a071ab296c99ad63184013869d9b50a1af4）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared500_20261003.json)（SHAcf0571f33cbb9358ae15e7fda91ef744e1496e0dff7c626d2842bdc5a4b6d0a4）。250独立归档保留，500真实endpoint只渲染一次。

只读审阅确认：原冻结核心已有--resume，恢复模型/Adam/scheduler/CPU-CUDA RNG、epoch/position和累计计数；现有pool启动脚本却不能传递到此入口，且outer新attempt必须有自己的COMPLETE。正在准备独立通用接续模块，保留原账本、完整来源/plan/capacity/data检查和同cell持锁；实际票据必须等原job terminal及checkpoint保存边界核实后创建。本回合未读取或散列实际latest.pt、未恢复模型，不能将存在文件当恢复通过。原保存替换latest→追加curve→summary，若硬停致末步不一致，原恢复拒绝保持，不补写证据绕过。

外层attempt/原inner run必须分开，禁止覆盖原jobs/result或补写curve绕过；当前没有真实恢复票据、source publication或恢复验证。实际sole latest.pt保留。

## 2026-10-03T04:02:11.291493+00:00 实际250里程碑的独立证据路径

真实250报告及12图已逐字节独立留存 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/report.json)；[实际概览图](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/shared250_20261003/v7_large_batch_ablation_overview.png)；[渲染/归档记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/watch_receipts/render250_20261003.json)（SHA41f067ee719394b508be77c97080b01e06bc367a7d203cb987672e5a1ad981f3）；[两seed派生](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/current_method_comparison_shared250_20261003.json)（SHA514bf2eba5a2ac69603e08e31373a10421690ba740451b260fc5851441863965）。未重绘、未重复模型验证或读取二进制checkpoint。

后续figures更新时保留shared250归档，未运行14项保持缺失标记，不能把早期或不同micro/GA的结果拼作完整终态。真实恢复latest.pt保留；预算原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review_common250.json。原安全/数据/选择/来源/恢复门禁不改。

## 2026-10-03T03:41:39.451832+00:00 正式启动衔接v2与实际预算原件

已冻结准备件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2.json（SHAe92de529db8b77f5c0e083ff115186c1419fbfc6c9a53ba61b386d5fde1bb9df），两份源在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2/scripts/research；根端新审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_v2_root_review.json（SHAae9a7f7a2cad1a4c3ffd92834b1947453dfc31182f76f62c615ac48486d327ff），首次三项CPU冷原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_v2_dispatch_controls_20261003/receipt.json。它只接续已冻结large-batch v2，旧formal wrapper/transport/v3及全部原門禁不改；当前formal未发布/入队，不能把准备件当GPU通过或模型选择完成。

实际预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/resource_budget_v1/budget_review.json（SHA9e31330742a3e6ef1cdf35c553e89bb1c970c1c41e6b6d6e49166be4c5c0bbac），绑定first25/175、旧3750终态与实际平台原件；预算不是完成证书。首8仍3750完整epoch、另14仍pending，默认原完整范围顺序。后续每次里程碑读固定ID真实curve/progress/summary/result，不减重复初始eval时间、不把未完成评估当曲线点、不自动缩短或重入队。最新接续 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md CURRENT。

## 2026-10-03T03:14:48.193324+00:00 新完整22项的实际启动与证据路径

现有pool /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930保持原容器及8worker。新组原件优先读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/enqueue_receipt.json和/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/initial_observation.json；当前实测8 running/14 pending，不能将登记或初始summary当epoch完成。历史prepared账本不回写：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json固定SHA987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d。实际状态看各固定ID在pool的pending/running/results及run原件，不自动重入队。

共同档位计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json固定SHA42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04，明确容量v2协议、实际九格/四原件/根端审阅。micro32×GA1/global32、eval4；三代表capacity measured，其余8条件只能标代表参考，不能填作直接实测峰值。原568模型冻结不改。新的启动源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2.json SHA4cd5f793e74f56730a0ca4468cebb2783fe2de22b33c7325f664f54634db43aa，公共entry/registry最新SHA见publication。11份清单YAML已记录实际ID/status/queued，/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/yaml_after_enrollment.json保留变动前后SHA；不使用它们重复申请GPU资源。现行限定范围数据审核、停止注释、独立CUDA暂停及实际sole latest.pt保留规则不变。

2026-10-03T02:36:47Z：容量v2终态根端实际核实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_root_terminal_review.json（SHAb6ae90637728f22bef378e71d0d2c943e50ee74e4ad52165b1f0a66d95970c5b）；四原件及40加载源已核对，后续统一micro32×GA1。此为容量选档依据，真实plan/jobs/新22项仍未发布，不代表正式训练准入或旧8更新协议通过。当前接续 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md CURRENT块。

## 2026-10-03T02:27:52Z 当前容量v2与完整epoch原件路径

独立容量intent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job_v2.json（SHA3429ab434375501e56860e52dd83ebe6702f436a6b18b54d26752b7ff2efc152）永久记录prepared状态；实际执行状态读取同一job1790992416352344471_p05_v7_capacity_v2_a4854542的pool results及run原件，不改历史intent。新启动源冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_launcher_source_v2.json（SHA20750dd2113dcb861c63ce02116059fdedfcbb8eb94d1b44ca4b468345d45bdc）；public runner/registry当前SHA分别c94ecec00de0f07671aeca2cecae1377075acf983511260e7e9b3c8e9f54b3f1、9eda402d0dfcf5528aa7a8cf7724e3566ca155a6932d4dfcc49faceed65a7cff。原p04失败/原568模型/profiler及旧入口链保持，不能重放已耗尽单次attempt。

v2是人工九格、2warmup+2measure、36更新、每格240秒/总1800秒/hard100；其实际succeeded/COMPLETE不等于旧8更新协议通过，更不等于真实语料或formal联合训练通过。未来统一batch必须明确绑定新协议和真实三代表15%余量。当前新22项plan/jobs不存在，旧私有large消费者仍为旧协议，需新版本明确接纳之后才发布训练。专属说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CAPACITY_TIMEOUT_REVISION_20261003.md。

原p02最终120k/3750报告独立留存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/watch_receipts/report_shared3750_20261003.json；两seed均值 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/current_method_comparison_shared3750_20261003.json。保留真实sole latest.pt，不用旧2000/3000覆盖当前终点。真实父身份metadata位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json，仅identity/argv，不作为GPU验收。

## 2026-10-03T01:27:30Z 正式训练新v3来源与恢复原件路径

正式新来源根 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3；611文件清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v3.json（SHA256 992bc3c5718b71c379c7a2ee4f39d8e5866f974266500d2c2ec69d5ab4db700b）。旧v1/v2保持历史不可变，不跨版本恢复。新driver/helper分别SHA40746122485cbaad7acc8eee6eab9a4b72b51ef7b9a3207e89e1d93cddf0391a、0a0ca3baa74ae2567953db9fdebab01f031fe0a3ebb1f4ccd28405bf78fa4c97。

新版本原helper实际门禁原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v3.json（SHA3bb405020788470b823fe15ee49016143582ab42ee56ecbd3102716c96835122），一次passed；不签证或改ready。未来真实resume需保存八rank原生恢复返回观测与真实续步的RESUME原件，仍要求原checkpoint sidecar、八rank参数同步/退出与段终态；CPU或来源门禁不等同于实际GPU恢复。专属说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md。

## 2026-10-02T18:53:05+00:00 当前最终测试范围数据已验收

使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_scope_final_20261002。按用户确认范围最终审核通过、training_ready=true，真实 assert_ready 已通过；保留 6,460,277 条。规则、明确排除项、证书、低样本限制及哈希以[最终记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)为准。历史版本 pending 不代表此新版本状态。未启动训练。

## 2026-10-02T18:40:43+00:00 开始最终范围收尾

回填既有 alpha CDR3、复核实际 T4 reference、对齐 test-only 门禁。新版本及限定范围见[最终收尾记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_SCOPE_FINAL_20261002.md)；当前构建运行中，未宣称审核完成。

## 2026-10-02T18:19:18+00:00 alpha CDR3 缺字段追溯完成

8,765 条均来自 VDJdb 衍生 fulllength；上游已有唯一 cdr3a，当前 prepared 转换未保留该标识。可追溯证据、代码原因与待回填状态见[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)。未进行新注释或修改数据。

## 2026-10-02T18:15:56+00:00 用户明确排除四项审核要求

原 V7 的 264 条 H3 未确认项及 GDPa1/Specificity/m396 专门保护证据不再作为本轮必需审核或阻断项；不是已审核通过。剩余 TCR 字段、T4 覆盖和训练入口对齐保持。当前范围以[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)与[机器可读修订](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/test_pair_release_rebuild_r2_20261002/user_scope_amendment.json)为准。

## 2026-10-02T17:40:43+00:00 r2 观测链修复与复审完成

当前新构建版本为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_test_pair_exact_20261002_r2；配对误判缺陷已修复并独立验收，整体审核仍未通过。数量、版本哈希、规则与剩余缺口以[r2 完成记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准；未修改训练入口或 ready。

## 2026-10-02T17:24:16+00:00 真实观测链修复 r2 开始重构

按 synthetic_residue_mask 区分真实伙伴与全合成占位链；新规则、版本路径和进度以[r2 记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_R2_20261002.md)为准。13 项边界测试通过，当前运行中；未解除整体审核门禁。

## 2026-10-02T17:05:42+00:00 新版审核暂不通过：真实观测与合成占位链需区分

上一轮导出按角色字段判定配对不充分，导致部分单链记录绕过非配对过滤。当前版本不得作为审核通过数据；真实观测链的判定、全量影响数与其余缺口以[审核报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_AUDIT_20261002.md)为准。未更改数据/门禁/ready。

## 2026-10-02T16:46:51+00:00 测试集配对完全重复版本已完成重构

采用实际测试范围与同一测试记录 CDR3α＋CDR3β 完全匹配规则；非配对记录保留测试范围内旧规则。新增原行号删除台账，原划分及保留行字节不变。导出和独立选择/字节复核已通过，仍未认证；版本布局、实测数量与默认口径依据见[本轮记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TEST_PAIR_REBUILD_20261002.md)。

## 2026-10-02T14:16:09.255833+00:00 当前状态：V2八卡已终态失败，下一步新版本最小锁修复

截至 2026-10-02T14:14:14.929183+00:00，V2固定共同八卡组 `original_r21_world8_1790948366803074817_36c1a3d5` 已真实终态失败：8个pool result均failed/rc1，组与coordinator均aborted。新版本完成8个checked join、assignment、8个实际子进程注册和8份startup token领取，推进超过V1仅3成员加入的阶段；但没有Trainer目录、已完成训练步数或保存恢复成功证据，不能称八卡DDP/模型训练通过。八份实际子进程receipt均rc=-15、wait/PGgone/cleanup=true，8份rank退出及member cleanup齐全；自有临时checkpoint=0，未删除实际训练恢复状态。

直接错误是锁等待超时。实际父进程PID577559在group_summary持join.lock时等待coordinator.group_status的第二把锁，持join锁2.011217秒，commit仅0.002797秒；随后6个成员锁超时。更早的coordinator持久abort比member abort早约10.90ms。旧coordinator没有持锁计时，最初持锁者/具体慢操作仍未知，不认定死锁或某次I/O是唯一根因，不称完整545来源扫描重回join锁。16份实际timing JSONL及原日志/失败原件保留。

当前原池pending0/running0/results89，八worker新鲜维护待命。平台任务t-20261001033327-fmnql最后实际CLI查询13:54:41 UTC Running；没有新平台提交/取消、容器/worker/监听重启。V2一次执行预算已消耗，当前所有源码、预算、freeze、gate、固定8job IDs及失败原件保持，不自动重试或重放V1/V2；有限监督已真实结束，无观察进程待poll。

原八单卡500人工学习和已完成CPU验收仍保留通过记录；R20/R21/R24/R26已实现，但真实架构选型/理解与生成端下游收益尚未证明。正式V7小batch及训练仍0：training_ready字段缺失/null非True，audit_pending，两份当前数据/下游范围审核均缺。缺审核不是已证明数据泄漏，不伪造证书或降低门禁；停止的序列注释分支不恢复/重试/转交/绕过，独立CUDA算子保持暂停。

下一步在独立的新不可变通用工程修复版本，消除成员锁内等待协调器锁及重复高频状态查询，保留来源/实际host与父子身份/2秒时限/持久abort/退出清理门禁，并针对真实协调器竞争路径做有限必要CPU验证和耗时记录。首次coordinator竞争原因须保留不确定性；消除嵌套锁不能单凭静态推断当通过。不改模型数学或真实数据审核，不增加无关方法波次，不重跑已完成500/profile/FULL/Gloo或原CPU套件。

[根端真实终态证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_GPU_group_failure_terminal_root_readonly_review.json)（SHA 2bb59dda1ad9a254362d6fce67b23a3fd80a0dfcbd782156c99d18504a1686ff）；[有限监督原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_group_bounded_readonly_watch.json)；[锁因果及边界复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_failure_lock_remaining_work_independent_readonly_review.json)。Root首轮只读字段误读source_check_state已保留错误原件，依实际JSON修正为source_phase；没有修改原validator/生产源或重试GPU。原Goal 01a0f245-4559-7b62-9dc2-0917a8ed2267仍active，上一准备回合及本执行回合均有实际progress；不新建或重置，实际累计原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/goal_actual_after_V2_terminal.json。

以下带日期内容为保留的历史状态。

## 2026-10-02T13:48:54.596399+00:00 当前接手入口：V2 CPU及暂存已验，GPU尚未执行

先读 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_CPU_lock_and_stage_root_acceptance.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md、/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_GPU_LOCK_REPAIR_20261002.md。完整545来源冻结SHA22a20b96a1e7c43cc0741f98ac458593483f2c77500790a5a552cd7345f36435；固定新计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json，SHA b8a0bbe61df61c31def88aaf2295194b5d24a5cb5bc2cac39ae3a017b1af0a5f。公开入口仍V1，V2三个文件仅暂存、没有host intent/发布/入队。后续执行阶段须先只读核对原资源、当前来源和全部真实前置，不重建计划或重放V1 job，不重启原容器/worker/监听，不重复平台提交或已完成验收。CPU证据不是GPU恢复或真实数据准入；真实训练仍须strict True及两份当前真实审核等原门禁。停止注释与独立CUDA暂停不变。以下带日期段落为历史状态。

## 2026-10-02T12:55:34.324017+00:00 当前交接路径与已耗尽V1预算

截至 2026-10-02T12:54:01.376745+00:00，R20基础对照、R21注意力头门控、R24残差缩放、R26链内局部/全局注意力交替的候选已实现；原八项独立单GPU人工合成实验均完成500步并通过原学习验收。新八卡入口CPU检查在worker/pllm两环境分别4/4通过，已通过的检查不重复。这些证据不证明真实下游提升，最终架构尚未选定。

共同八GPU合成组 `original_r21_world8_1790944282201927112_f554453b` 的唯一V1尝试已终态失败：8个pool result均failed/rc1，直接错误 `Member lock wait expired`，只有3成员加入，assignment/startup未产生，8份member receipt均显示cold child未启动、清理完成。模型训练步数0，原生GPU保存恢复未验证，本次checkpoint目录0。此次失败发生在模型/NCCL启动之前；不撤销已有单卡500步通过记录，也不把源码已实现或入队当共同八卡成功。

只读源码诊断确认锁获取时限为2秒，join临界区内再次核对完整545来源/gate/plan并提交状态；这是需修正的长临界区设计。没有阶段耗时trace，不能认定某次SHA、I/O或调度是唯一根因，也不能仅凭锁文件存在称死锁。下一步保留V1源码、预算、计划、日志和失败原件，在新版本做最小通用协调修复：完整来源核对移至短事务外、两阶段状态绑定防止未核对成员放行、保持启动前完整核对和不可逆abort，并记录真实等待/持锁耗时。只补新增边界所需有限CPU锁竞争/来源变化拒绝/超时清理检查，再经来源复核安排有限纯合成验收；本状态回合没有重试、运行新测试或新入队。

原平台任务`t-20261001033327-fmnql`最后实际查询为12:37 UTC Running；12:54 UTC远端八worker metadata均standby_load/maintenance_only，心跳年龄1.39–2.62秒。池pending0/running0/results81。原八卡容器/worker/监听没有重启，没有Volc submit/cancel；pool成员失败不等于平台八卡容器失败。

V7 `immune_v7_mhc_sequence_v1` manifest SHA仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，training_ready字段缺失（null，非True）、audit_pending，audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明泄漏；正式V7小batch和训练尚未启动，仍须当前manifest的真实数据审核及完整下游范围证据。当前只做通用工程与人工合成验收；安全停止的序列注释分支不得恢复/重试/转交/绕过，独立CUDA算子保持暂停。

[本次根端终态核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_overall_progress_after_group_failure_readonly.json)；[有限观察原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/group_bounded_watch_v1.json)；[原七项500验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original7_terminal_root_v1/original7_terminal_root_acceptance.json)；[入口CPU验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/world8_new_boundary_controls_v1/cpu_boundary_root_acceptance.json)。原Goal `01a0f245-4559-7b62-9dc2-0917a8ed2267` 仍active，身份、目标及累计不重置；实际工具快照见[Goal原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/goal_actual_user_status_after_group_failure.json)。

V1唯一GPU尝试预算已消耗，不重放原8job ID、不修改旧freeze/budget/gate伪装新尝试。当前通用入口为/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/gpu_pool_dispatcher_v1.py；之前已评审发布仅改model.py的8个入口别名和run_immune_research.py的分发路径，原模型类/函数不改。原15保护文件中13不变、2为有据的入口改动；不得继续声称发布后15均不变。任何修复使用新私有版本，复用原模型和已完成证据，正式数据准入保持独立。

以下带日期内容保留为历史记录。

## 2026-10-02T12:10:40.759435+00:00 共同八GPU接入当前来源与交接路径

原模型共同八GPU Trainer所需5个私有源码已全部交付，根端11:58 UTC仅做AST/接口和物理SHA静态核对，固定完整545文件来源，15旧保护文件未变。冻结SHA 1b5cfd23a1e9c3cbacb9f30b2fd9dd949fbdcf2d4063962eeac8780ee2311bba。没有导入执行新源码、运行新CPU控制、GPU训练、stage/发布/入队或生成launch gate；代码完成不代表八卡验收通过。原模型/Trainer原生训练和保存方法保持，完整Adam、scheduler、八份真实RNG及恢复后实际状态比较已实现于新源码，等待实际工程验收。

新来源目录固定为/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1；预算/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/execution_budget.json的SHA4397c4fec564113d9871523760c8e4df5531426c2d715d9ba7254b30d8637d5d及545文件freeze不可在执行后补改。source/profile43和人工learning seed42分开记录；原七项profile不是独立学习seed复现。group plan/launch gate尚未生成，不得凭代码完成或标记字段伪造启动证明。GPU host/boot/UUID只能由实际GPU worker观察。

截至2026-10-02T12:05:11.695172+00:00，原七项单GPU人工合成实验全部500/500、实际pool result.state=succeeded/returncode=0，inner与outer COMPLETE均存在；原池pending0/running0/results73。CE R21/profile43原500此前已通过唯一原checker及独立复核，不重复；其余七项刚取得终态，尚未执行各自原verify_diagnostic(500,require_learning=True)，不能记为验收通过。8个原worker在12:06 UTC独立只读核实均standby_load、maintenance_only=True，原资源不重启。

下一步在既有授权范围内，先对七项真实终态各执行一次原500学习验收和结果/来源/退出清理复核（CE已验不重复），完成新来源每环境最多4项必要入口边界控制及独立复核；全部原来源/profile/学习/GPU/完整恢复前置、实际八worker可用及新启动检查满足后，再通过现有pool运行一次原模型共同八GPU纯合成Trainer保存恢复工程验收。连续3步与保存2步后恢复至累计3步用于状态一致性，不代表1000/5000训练阶段或架构效果；不新增架构波次或重复已通过CPU/Gloo/numeric验收。

V7仍为immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94；training_ready字段缺失（null，非True），decontamination.status=audit_pending，当前audit_certificate.json和downstream_scope_certificate.json均缺。缺审核不等于已证明数据泄漏，也不是操作权限问题；正式资源绑定仍缺权威完整下游范围/输入清单及绑定格式。未读取真实分片/抽真实小batch/重建审核selection/启动真实训练，不伪造审核、手改ready或降低门禁。

停止的序列注释分支不恢复、重试、转交agent或绕过；独立CUDA算子继续暂停。原八卡容器/作业/worker/监听2213164不重启，不重复Volc提交；有限observer90075已自然结束，不重新启动。临时验证checkpoint仅在真实退出与独立证据复核后清理自有文件，保留原失败、来源、原件和实际训练恢复状态。

[当前状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/user_status_after_source_freeze_readonly.json)；[根端静态核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_delivery_root_static_review.json)；[完整来源冻结](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1/source_freeze_before_CPU.json)；[工程实现与接续说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_ORIGINAL_TRAINER_EIGHT_GPU_20261002.md)。

以下带各自时间的记录为历史状态。

## 2026-10-02T11:35:31.718386+00:00 原模型八GPU通用验收来源与边界

新私有源码/预算统一在/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v1，旧CPU来源与预算不可修改或当GPU准入。新source freeze未齐不能创建intent；root生成的group metadata不能代填GPU host/boot/UUID，必须实际mask0 worker现场创建intent，每member实际观察不同UUID/PCI并保持自己的物理mask，LOCAL_RANK0/全局RANK0..7。正式数据审核仍独立；工程验收不是1000/5000训练阶段。本轮只源码准备，旧已完成验收不重复。实际状态优先读/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md及/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json。


## 2026-10-02T11:03:50.603902+00:00 当前原CE500学习验收已闭合

CE R21/profile43原500于10:56:27 UTC succeeded/rc0，原verify_diagnostic(500,require_learning=True)唯一root只读调用及独立stdlib复核均通过，trace25..500共20点。学习seed42：四mask平均loss4.0881584711→0.1847930732，AA准确率6.38465%→92.65651%，terminal100%；完全遮蔽mask100 AA71.54613%、loss0.7071843，平均不能称全遮蔽/生成准确率。39当前source物理SHA、9实际输入及62独立引用哈希、日志/双COMPLETE/实际子退出清理均一致，checkpoint目录0。

截至2026-10-02T11:03:50.603902+00:00，原池pending=0/running=7/results=66；CE已完成验收，其余七项最新步数为375、375、375、375、375、375、375/500，仍需真实终态后各自一次原验收。GPU1回原维护状态，没有重启worker/作业/监听或新增GPU任务。

该证据只证既有当前来源人工单GPU学习，不证真实V7/生物下游效果、单模型八GPU-DDP或正式恢复；profile43不是新学习seed。不同来源、不同步数及相同learningseed42的各profile不能当独立多seed比较或最终选型。原CPU Trainer/通信验收保持完成，不重复；正式资源接入及实际共同GPU保存恢复仍未完成。

本回合重新仅读V7清单元数据：manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94、training_ready字段缺失、audit_pending，两份当前审核均缺。合法当前完整下游资源范围清单及绑定格式缺失，原正式gate无法正确定稿；不造清单/证书、手改ready、降低门禁。真实shard/小batch/审核selection/训练0；停止注释不恢复/重试/转交/绕过、独立CUDA算子暂停、原资源不重启或重复提交。

当前验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_acceptance.json；原只读调用尝试 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_root_attempt.json；independent /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_CE500_terminal_independent_review.json；最新原任务快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_after_CE500_independent_closure.json；V7 metadata /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/V7_metadata_after_original_CE500.json。

## 以下为较早记录

## 2026-10-02T10:34:05.741083+00:00 新增通信控制预算已用及证据边界

本轮新增八进程通信/清理控制两环境首次各4/4通过：worker5.747秒、pllm7.253秒。每环境8实际parent和8实际CPU/Gloo cold rank全部rc0/wait/自有进程组已清，rank0持有FD直接交TCPStore；倒序member正确映射rank0..7、实际store/all-gather8身份一致、缺员不可逆abort和未登记失败child rc2清理均通过。26旧文件及9冻结source不变，临时目录0、checkpoint0，不重复已完成numeric Trainer。root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

此证据只覆盖新增CPU通信及词表列表元数据格式校验；原生产Trainer类组装仅静态复审完成，未实际构造/训练；新词表callback未通过实际native保存恢复。实际单模型8GPU-DDP完整保存恢复、正式外部资源参数全调用链和最终架构选型仍未完成。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool pending0/running8/results65，八项各自pool result及内外COMPLETE均缺。八项是独立单GPU，不是单模型八GPU训练；不同来源/步数和同learning seed不能直接选优。

下一步收尾原八项500实际终态，各自只执行一次原verify_diagnostic(500,require_learning=True)及原始结果/来源/退出清理核验。原八卡全部空闲后，在单独冻结当前来源与有限预算下，补必要通用单模型GPU组及原Trainer保存恢复证据；不新增架构波次，不重跑旧通过验收。真实数据训练须当前有效审核及原全部data/scope/source/selection/learning/profile/GPU/full-resume条件满足。

V7为immune_v7_mhc_sequence_v1，train6,849,243/valid52,572，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94。training_ready字段缺失、audit_pending；audit_certificate.json与downstream_scope_certificate.json均缺。缺审核不等于已证明数据泄漏；用户认可或操作权限不能替代当前manifest及新增下游范围审核。真实分片/小batch/语料训练0。

停止序列注释不恢复、重试、换agent或绕过；独立CUDA算子暂停。原容器/worker/监听2213164不重启、无重复Volc提交，本轮新增GPU/平台submit/cancel/restart为0；不造证书/手改ready/降低门禁。

[最新原任务与V7快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_and_V7_readonly_snapshot.json)；[本轮根端复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_review.json)；[新增工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md)。

两个环境本来源各4控制的一次预算已使用并通过，不得重跑；禁止pytest启动此CLI。原已过numeric Trainer保持独立证据，不向新callback/生产来源迁移声明。固定源码清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/source_freeze_before_CPU.json。

## 以下为较早记录

## 2026-10-02T09:54:10.064871+00:00 已完成的私有八CPU Trainer工程证据与生产边界

[当前根端验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/actual_CPU_conformance_root_acceptance.json) 为numeric CPU-only已通过证据，冻结五source、两个新source加三个原source。入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/run_cpu_eight_rank_trainer_v2.py 每环境一次实际case预算均已使用并通过，**不得重复执行**；原V1失败和未执行pllm预算不被修正版覆盖。原件、source SHA、170/38个复核输入及清理记录见验收JSON。

此验收仅证明纯数字CPU fixture的新增训练器工程，不证明免疫生产Trainer、真实GPU-DDP、1000→5000续训、架构生物效果或数据准入。已清理二进制未被独立review重读，证据来自执行时保留的原值/哈希及冻结source；保留全部原失败/raw/source/log与实际训练恢复状态。

正式单模型八GPU接入仍须真正连到生产Trainer和原数据/完整恢复门禁；不能把CPU fixture、新来源或八独立任务合并成实际GPU-DDP已通过。原8GPU任务PID只在同namespace核实，root本地/proc缺远程PID不能当退出或重启依据。必要新GPU组验收须等原八worker空闲，不中断当前任务、不重复提交。

V7版本确为immune_v7_mhc_sequence_v1：train6,849,243/valid52,572，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94。training_ready字段缺失（null，非True）、decontamination.status=audit_pending，audit_certificate.json与downstream_scope_certificate.json均缺；父版当前范围审核待完成，更新manifest未被父证书覆盖。不能据此认定数据泄漏，也不能以用户认可/权限或手改ready代替原审核。

真实shard/小batch/语料训练0；原strict training_ready is True、当前双审核及全部data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。停止序列注释不恢复、重试、转交或绕过；独立CUDA算子暂停；原容器/worker/监听2213164不重启、不重复Volc提交，新GPU/平台submit/cancel/restart0。

最新当前状态路径 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/user_training_readiness_after_strong_reviews.json；机器状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json。


## 2026-10-02T03:00:55.226271+00:00 唯一失败项数值观测准备

单项诊断源码与控制归档于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002；runtime/profile/seed及旧CPU数值helper SHA固定。仅一个launcher_only观测项接原worker，不改release_map、不重启容器/worker/监听或提交Volc。真实V7无ready及双报告时不读取真实小batch/启动训练；停止的注释分支和独立CUDA暂停保持。 [源码diff](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/staging/source_delta_review.json)；[metadata](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/user_readiness_check.json)。


## 2026-10-02T02:49:04.121405+00:00 单项原失败GPU数值观测目录

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002只用于必要R21seed43 tiny原失败观察，全部准入false；保护原runtime/private/helper/release_map/worker，原7项接受不重跑。原physicalGPU2不可指定，新任务实际卡/UUID须记录。 [固定预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget.json)。


## 2026-10-02T02:39:30.314663+00:00 当前v2七项工程证据与失败边界

外部[当前七项完整核验/一项失败](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/final_results_and_readiness.json)绑定原runtime/source/seed；四源码包仍source-only/接受标志false，不热改包追认，真实训练门禁不变化。CPU未复现GPU拒绝不算GPU修复；不为清理日志重跑tiny。必要raw/proof/source/日志及恢复状态保留，14个0字节自有控制文件和4710B临时库已清理、ckpt0。


## 2026-10-02T02:34:39.929269+00:00 修正只读终态driver路径

[review_seed_gpu_terminal_v2.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/review_seed_gpu_terminal_v2.py)替代本次有接口错误的只读driver用于未完成的核验；原driver和首轮日志保留。v2仅读实际executor __file__/SHA，预算先单项后六项，原GPU/CPUtiny不重跑。


## 2026-10-02T02:32:11.628607+00:00 当前新GPU结果只读复核路径

[review_seed_gpu_terminal.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/review_seed_gpu_terminal.py)仅复用原verify_completion/verify_candidate检查新七项结果，使用worker原Python、无模型或训练；输出父端日志/复核摘要，绝不生成GPU证明。CPU数值与4710B自有临时库清理见[原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/cpu_results_and_cleanup.json)；既有失败与日志不覆盖，不为修日志重复诊断。


## 2026-10-02T02:25:18.839510+00:00 独立CPU失败诊断源码路径

[diagnose_r21_seed43_cpu.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/diagnose_r21_seed43_cpu.py) 为固定R21seed43一次tiny数值观察，输出仅专属新子目录pllm_once/worker_once；原生产/live/private/runtime来源不改。其结果不能写为GPU/正式训练接受，原data/500学习/恢复标准保持。冻结：[source_review_before_execution.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/source_review_before_execution.json)。


## 2026-10-02T02:18:49.402651+00:00 当前交接段与历史保存

当前入口 [IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md) 已集中当前事实，原交接段完整归档至 [handoff_before_20261002T021849Z.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002/handoff_before_20261002T021849Z.md)；不能将旧版本通过或零学习率预热写成当前版本完整通过。失败诊断仅新目录，全部活跃v2源冻结；停止注释/数据门禁/CUDA暂停与禁止重启或重复提交保持。


## 2026-10-02T02:10:26.502679+00:00 独立失败诊断路径

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed_failure_diagnosis_20261002只保存本次失败tiny阶段的CPU源码/数值统计。原v2/live/private_sources/runtime_packages不修改；当前未运行、不产生GPU/数据接受marker或checkpoint。见[诊断记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_SEED43_GRADIENT_DIAGNOSIS_20261002.md)。


## 2026-10-02T02:02:27.596571+00:00 新seed实际失败及当前边界

最新实际状态为7运行、1失败、0/8完整验收通过。R21 seed43/GPU2在lifetime_preflight独立CE/GA梯度核对报wte.weight不一致；profile正常退出，release子进程rc1、runner rc2，父进程记录进程清理完成，GPU2恢复维护。错误时对象弱引用存活，不把它写作成功阶段清理，也不凭此认定持续泄漏。其余7项仍gradient_oracle，真实训练0。

根因尚未确认，原断言未保存逐参数误差统计；下一步是隔离通用数值/随机流诊断。/root/architecture_keep_or_change只读分析，/root/candidate_gpu_watch只读监督；全部v2活跃源码/私有来源/运行包保持冻结，不热改、重试、放宽阈值或重启。V7仍缺有效ready及双报告，原500学习、完整恢复、数据门禁与安全停线保持，独立CUDA算子暂停，停止注释分支不恢复。

[实际失败复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/r21_seed43_failure_review.json)、[最新逐项状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/observation_20261002T020227Z.json)。

## 2026-10-02T01:57:11.789606+00:00 当前seed运行版本与状态路径

实际四包位于/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/runtime_packages/{r20,r21,r24,r26}_seed_runtime_v2，各有seed_runtime_manifest_v2.json；新增入口/producer只属于软件来源，GPU/data接受仍false。新独立接入已部署至/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_seed_release_dispatcher_v2.py与/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/seed_release_startup/sitecustomize.py；旧dispatcher/guard/worker不变，无重启。仅八个d02_pilot_seed_gpu_rXX_seedYY_v2的profile-synthetic/1人工验收执行，不代表t20/t21/t24/t26真实训练入口已注册。数据缺报告时禁止真实小batch。

总预算3600秒含profile；正常退出和子进程清理后才可产生GPU证明，当前0/8完整通过。日志/源码/备份及真实恢复状态保留；仅清理自有临时控制与无用验证ckpt。

[新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_cpu_results.json)、[四实际运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_software_build.json)、[容器接入变更](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_publication/publication.json)、[八项任务与目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/launch.json)、[领取状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_launch/observation_20261002T015550Z.json)、[V7数据状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/v7_seed_runtime_readiness_20261002.json)。

## 2026-10-02T01:49:20.360499+00:00 seed运行实现状态

新seed producer/package/entry及独立dispatcher/guard/stage位于/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources；各组件新增CPU通过，四运行包尚未构建。仅私有源码，不等于容器可执行或数据准入。stage仅生成审阅文件；旧dispatcher/guard及worker不修改、不重启。总预算3600秒包括校准。

[新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_cpu_results.json)、[当前数据与资源原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/v7_seed_runtime_readiness_20261002.json)。

## 2026-10-02T01:40:13.606889+00:00 新seed runtime命名与执行边界

上一Goal回合有实际进展：八身份软件接口CPU两环境各13/13完成及MD/source记录保持。本轮继续独立必要工程，三agent分别写新seed-aware人工Trainer producer、新runtime package/entry、独立extension dispatcher/guard/stage；root负责整合、预算、实际检查与MD。当前源码写入中，未冻结/构建/执行新checks/部署/入队。新的四方法×seed42/43共8人工验收须实际定点micro32三更新profile、原完整GA/oracle/评估/退出清理及精确seed/helper/source绑定；旧39源profile不重标，原500学习门禁独立保留未完成。

只读01:32:35 UTC：原八卡pending0/running0/results42、8worker维护无job，平台缓存01:32:10 Running、监听PID2213164存活。指定V7数据仍按01:29 metadata观察缺明确ready和双报告；不读取shard/真实小batch，不恢复停止的注释。无平台提交/重启，独立CUDA算子暂停。原Goal actual active，保留身份及实际计量，上一回合归类progress、不复用旧阻断计数。

新源仅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes；派生软件目录计划为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/runtime_packages/r20_seed_runtime_v2、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/runtime_packages/r21_seed_runtime_v2、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/runtime_packages/r24_seed_runtime_v2、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/runtime_packages/r26_seed_runtime_v2，manifest seed_runtime_manifest_v2.json仍仅source-only falseflags。8人工身份d02_pilot_seed_gpu_r{20,21,24,26}_seed{42,43}_v2固定，未注册。新guard仅这8项，旧dispatcher/guard计划保持SHA并委托；stage仅review产物，不apply。

[执行前预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_execution_budget.json)、[22项受保护来源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_runtime_v2_protected_before.json)。

## 2026-10-02T01:30:12.523373+00:00 八身份私有软件接口边界及证据路径

八身份软件请求接口完成，pllm与worker对应Python各13/13新增定向CPU检查首轮通过；实际核对四v4包的manifest/fullmap/helper、八个显式seed请求、软件1000→5000续接一致性及缺执行准入时无副作用拒绝。源码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_seed_launcher_v1.py SHA867427c73dd7c2b16e8f419d5e22d1b1735aad54d832a13316e2ff2f0980ea24。独立静态复审1项收尾SHA位置问题已修正，冻结前writer变更使旧SHA记录拒绝，未将过期复审写成通过；原件保留。受保护18项源码/manifest SHA保持，旧core28/execution11/8次完整导入/旧Trainer验收未重跑。两pytest临时根未创建，临时文件删除0，checkpoint创建/删除0；必要日志/JUnit/冻结与复审记录保留。 [新增CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_cpu_results.json)。

本次只完成软件请求接口；完整八身份dispatcher/registry/entry接入、seed-aware runtime/producer与当前源/seed的GPU及规定profile/diagnostic证据仍待完成。未部署/注册/入队/运行新GPU/真实训练。原八卡01:29:00 UTC pending0/running0/results42、8worker维护无job；平台缓存01:28:52 Running、监听PID2213164存活。指定immune_v7_mhc_sequence_v1 manifest SHA未变，无明确training_ready字段及两份当前版本/范围报告。旧V7已有清理；当前缺绑定不能解释为已证实残留泄漏。独立CUDA暂停、停止注释分支与原数据门禁全部保持；原Goal actual active及身份/计量保留。 [资源及V7原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_final_resource.json)。

软件接口仅在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_seed_launcher_v1.py，inspect API只返回来源检查的descriptor，launch会拒绝；当前不是生产训练入口。冻结输入 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_source_freeze.json；两套环境实际JUnit/log索引 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_cpu_results.json。禁止把此结果标成完整controller部署。

下一必要工程：私有正式入口/registry/dispatcher和新seed-aware runtime verifier/人工Trainer producer；随后只获取受影响的当前source/seed证据。旧profile/500结果按原来源保留，不替换SHA、不删原检查。有效data/selection和完整checkpoint恢复是实际训练前置；团队数据审核未闭环时不抽取真实小batch、不启动正式训练。首轮仍R20/R21/R24/R26×seed42/43共8独立单卡比较，G02为条件P2。

## 2026-10-02T01:18:38.651326+00:00 八seed私有训练入口路径

用户条件授权架构完成后使用指定V7启动原八卡；当前指定版本仍为immune_v7_mhc_sequence_v1，manifest SHA未变，无明确training_ready字段、audit_certificate.json和downstream_scope_certificate.json。旧V7已有清理记录；缺当前MHC/新增范围绑定不等于已经证明残留泄漏。本轮仅metadata/软件核实，无shard/注释/真实小batch。四个R候选与旧synthetic Trainer已验，v4软件8/8导入已验；新seed执行入口、当前来源GPU证据与数据准入仍未完成。启动最小私有八身份controller工程，writer /root/architecture_validation_strategy，root负责定向CPU、MD与清理；受保护18项SHA已冻结，预算先行，尚未执行新checks。原八卡01:17:56 UTC pending0/running0/results42、8worker维护无job，平台缓存Running，监听PID2213164存活。未submit/cancel/restart/入队；独立CUDA暂停及停止注释分支保持。原Goal工具active/身份计量保持，不把有独立工程误记为全部外部阻断。 [状态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/seed_controller_readiness_observation.json)。

新增软件接口只在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/private_sources/scripts/architecture_prototypes/pilot_pool_seed_launcher_v1.py；绑定实际四个v4源包，未部署/注册，不能替代新runtime/GPU/data准入。

## 2026-10-02T01:07:48.674852+00:00 固定v4源码包与环境路径

v4源码隔离导入两环境共8/8通过，每次54项目模块、8seed软件身份；临时库2355B实际清理。新源GPU/容器执行接入与数据准入尚未完成，真实训练未启动。 [实际v4验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_package_v4_validation.json)。当前来源软件根为`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_packages`，四个目录为`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_packages/r20_seed_sources_v4`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_packages/r21_seed_sources_v4`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_packages/r24_seed_sources_v4`、`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_packages/r26_seed_sources_v4`，仍不可执行准入。

worker既有declared site `/vepfs-mlp2/c20250601/251105016/conda/envs/protenix_abtcr/lib/python3.12/site-packages`明确绑定唯一canonical site `/vepfs-mlp2/c20250601/251105016/conda/envs/flow/lib/python3.12/site-packages`；只接受对应准确module文件、无叶symlink/再重定向及同code来源，未改变环境。库预热只允许已核实8无参数兼容对象、固定Linux空设备最多9类型探测及自有tmp最大256KiB；项目模型/data/CUDA/源码写入限制持续，真实库tmp每次2355B全部清理。不得把源码导入通过当GPU/data/training通过，不热改原不可变包，不重启原容器/worker/监听。

## 2026-10-02T00:26:41.616168+00:00 导入边界v2修正进行中

四个v1源码副本已构建、source-package CPU各23/23通过；实际fresh导入首轮两环境仅R20失败，原因是guard误拦第三方临时模板/无参数激活对象，R21/24/26未跑。失败原件保留；精准库预热及自有tmp修正进行中，新v2不覆盖v1，未通过完整导入/GPU/真实训练。 [原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/source_import_attempt1_failure.json)。只允许准确已核对的第三方无参数兼容对象及自有临时库模板，项目模型构造/训练/CUDA/data/source-write仍禁；全部GPU/data/execution准入标记false，不借此降低真实门禁。

## 2026-10-02T00:09:37.510228+00:00 seed执行层来源与证明绑定

私有执行层seed/source绑定新增CPU两环境各11/11通过，旧GPU schema与跨seed证明拒绝；原gate-first/500诊断/完整resume/phase保持。源码包与隔离导入实现中，未部署/入队/新GPU/真实训练，数据门禁及停线保持。 [验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/execution_cpu_results.json)；新schema为seed_bound.v2，不声明旧记录已覆盖新源或seed43。新GPU证明producer和正式controller仍未完成，source-only包不是准入。

## 2026-10-02T00:05:41.621649+00:00 P1种子身份设计同步

禁用P1设计新增8个明确seed42/43身份及4个拟source-only包路径，CPU核心各28项通过；不等于worker注册/新源GPU证明，production_ready、execution_enabled、eligible、enqueued全部false。 [配置原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/decision_matrix_seed_binding.json)；[具体配置](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。保持原目标/数据/资源安全停线。

## 2026-10-02T00:01:13.670226+00:00 多随机种子核心CPU完成、纯源码包开始

必要seed核心私有修复完成：pllm与worker Python各28/28定向CPU检查通过（AST隔离+tiny Torch构造/RNG，0训练更新）；14受保护SHA保持，22自有控制文件已清理，checkpoint=0。继续构建明确source-only新来源包和隔离导入工程，未部署/入队/运行新GPU；指定V7缺ready及两份检查报告，真实训练未启动。原八卡0/0/42与监听保持，Goal active，全部停线保持。 [本轮验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/cpu_results.json)、[多seed工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_MULTI_SEED_PILOT_ENGINEERING_20261001.md)。新增来源包只作未发布软件副本，execution_enabled=false；不复制旧运行manifest，不生成准入pointer、不修改既有GPU/profile证明或数据门禁。

## 2026-10-01T23:48:36.005298+00:00 必要多随机种子工程启动

Goal工具实际返回active，目标更新为完成架构探索并在容器启动八卡初步训练；保留返回身份/计量及旧历史，不调用create_goal、不复用旧阻断计数。独立复核发现P1四方法×seed42/43的入口实际写死42，开始必要私有seed工程；G02属于条件P2，不是P1前置。指定V7数据状态仍未通过，原八卡0/0/42；无真实训练/重启/提交，已过验收不重复，全部停线保持。 [当前Goal与必要工程原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_multiseed_pilot_engineering_20261001/implementation_started.json)、[详细工程记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_MULTI_SEED_PILOT_ENGINEERING_20261001.md)。当前有实际工程可推进，Goal active且未完成；不把旧“无独立工程”结论用于新发现。

## 2026-10-01T23:31:00.205418+00:00 审核文件术语

项目所称“双证书”是 audit_certificate.json（指定版本清理检查）与 downstream_scope_certificate.json（代码定义的下游范围覆盖检查）两份机器可读 JSON 报告。它们不是外部签章、两人审批或额外权限要求；通过须有真实核查证据及一致版本绑定，不能手填 passed 代替核查。定义与代码依据见[两份审核文件说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/DATA_BENCHMARK_DEDUP_SCOPE_20261001.md)。本轮未改变数据格式、审核范围、门禁或 Goal 状态，未生成证书或执行训练；此前注释停线与独立 CUDA 暂停保持。

## 2026-10-01T12:17:53.844266+00:00 四候选私有训练接入边界

四候选正式训练接入的私有源码与新增CPU拒绝/身份检查完成，pllm/worker各24/24通过；1项静态来源缺口已修正。原R20 12函数AST/字节、6live源码、4runtime pins保持。清理30个自有控制文件/1118字节和10别名，验证checkpoint=0。未部署/入队/实际stage或准入执行，不代表架构效果；MHC无ready字段且双证书缺，真实小batch仍未启动。 [本次状态、SHA与限制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)、[实际CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/validation.json)、[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_training_bridges_cpu_20261001/cleanup.json)。 四身份仅固定各自的runtime/profile/GPU证据及完整恢复；两个stage helper仅写自有review artifacts，无apply/CAS。stage须在任何output前核对实际dispatcher before SHA。旧expanded/selection保持冻结拒绝基准，MHC不得沿用其selection。原Goal、八卡/监听、注释停线及CUDA暂停保持。

## 2026-10-01T12:01:52.464326+00:00 四候选正式训练接入准备

R21/R24/R26已具备各自正式API，当前补齐独立外部执行身份、来源/验收/续训绑定；4项私有源码准备进行中，CPU尚未执行，未部署/入队。原R20门禁和已验runtime保持，指定MHC缺双证书及ready绑定，真实小batch未运行。[本轮状态与边界](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINING_BRIDGES_20261001.md)。

## 2026-10-01T11:47:52.931165+00:00 三方架构决策完成

三个用户授权subagent只读决策已完成：保留R20/A4主干，先单独比较R21/R24并以R26作成本对照；联合任务G02末端adapter为条件候选，G01降为读出/优化控制，硬瓶颈和EMA暂缓。8卡4方法×2seed及后续联合组设计已记录，未启动实验。原指定数据缺双证书/ready绑定，真实小batch未运行；原Goal与资源、门禁、注释停线和CUDA暂停保持。

[决策与具体测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_CHANGE_DECISION_20261001.md)、[三方审查摘要](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_architecture_decision_20261001/agent_review_summaries.json)、[禁用决策矩阵YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/decision_matrix_20261001.yml)。新增发现：G01当前任务仿射可吸收到通用原型独立Linear头，不能预称主干任务特化；G04/G05按role而非任意段ID隔离且关系头存在池化旁路；正式R20输出untied但尚无G00式联合分类头。审核范围计划与旧执行scope还需一致的新版本绑定，本轮不改gate、不签证。以下带时间条目是阶段历史。

## 2026-10-01T11:29:06.399847+00:00 当前GPU工程准备与正式数据选择

通用GPU引擎CPU验收完成：两环境各44个唯一案例分三阶段覆盖，未完整重跑最终44项；修正4项工程边界，固定15文件源码包与8份禁用YAML核对通过。清理50个自有CPU状态文件，验证checkpoint=0。现有8卡执行端尚未接入G入口，新增GPU任务0。用户指定immune_v7_mhc_sequence_v1仅metadata核实，缺双证书且无training_ready=true，真实小batch未启动。

[当前GPU工程协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)、[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/validation_profile_count_fix.json)、[固定源码包及核对](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/source_package_review.json)、[指定数据门禁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/requested_training_data_gate.json)、[当前配置SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_joint_gpu_engine_cpu_20261001/updated_plan_manifest.json)。原Goal保持active；保留原注释停线、真实数据审核与独立CUDA暂停条件。正式架构比较优先使用审核通过的指定训练数据小batch及实际模型输入/目标/评价；人工整数仅作工程控制，不替代正式效果。下一步完成原8卡执行接入、独立结果复核与自有续接状态清理工具，不重跑已通过检查。

## 2026-10-01T11:15:42+00:00 去重范围按用户决定更正

后续仅保护实际测试/query 记录，不自动保护 benchmark train/valid/support；混合池按实际评分角色界定。旧全支持池规则作为历史记录保留，新决定优先。OTS 的单链近似排除不能称为配对重复；当前仅完成范围核算，未重导数据。详见 [测试集保护范围与 OTS 行级核算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/DATA_BENCHMARK_DEDUP_SCOPE_20261001.md)。

## 2026-10-01 10:55 UTC 通用GPU入口与临时续接约定

新GPU源码为[固定协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_protocol_v1.py)、[连续训练engine](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_engine_v1.py)、[包入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_gpu_entry_v1.py)；[状态与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_JOINT_GPU_SEGMENT_PROTOCOL_20261001.md)。源码存在不等于已安装/可入队：当前live入口和G注册表未改，独立源码包、发布、结果强复核及enqueue仍待完成。当前CPU两环境各26项首轮通过，复审修正定向检查正在执行。

G profile单独身份最多4次独立更新；G的5段各真实100次更新及累计步数分别登记，不能借R/D身份或伪造data_dir。常规checkpoint_save=false与内部临时续接状态必须分开描述；后者绑定明确前段/源码/配置/seed/完整optimizer/EMA/scheduler/RNG，完成最终验证后删除自有状态，恢复需要期间保留。执行COMPLETE不是独立ROOT_REVIEW，更不是数据证书。CPU控制状态只放本轮自有目录，检查结束清理，不碰真实恢复文件。原Goal、资源、数据门禁、注释停线和CUDA专项暂停保持。

## 2026-10-01T09:37:59+00:00 通用联合原型路径与记录约定

G00–G07当前已实现并完成CPU功能验收，GPU入口/显存测量/500步曲线待完成。接续先读[原型记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CV_JOINT_PROTOTYPE_20261001.md)和[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)，再核对原件；以下日期条目描述当时状态。

通用原型分为[整数数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_data.py)、[共同模型及变体](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_model.py)、[联合目标与CPU入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/generic_joint_experiment.py)。只接受人工整数对象；当前CPU CLI不提供GPU或真实文件入口。所有Python执行先激活pllm；worker版本调用也在该激活之后进行。源码5项哈希与实际报告索引在[CPU验收原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/validation.json)，不要改源码后沿用旧通过结论。

8份[方法设计YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)仍未启用，不能直接volc submit；当前版本SHA见[配置清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_cv_joint_prototype_cpu_20261001/updated_plan_manifest.json)。原8卡和已通过R系列版本保持，G系列单独记录模型规模、任务、来源与结果，不能复用R身份登记新方法。独立随机流与不同ID不等同内容去重，synthetic记录不成为真实数据审核证书。

每个过程节点采用同一接手口径：**UTC与维护agent → 变更/实际执行范围 → 配置/源码版本 → 原始证据及核验 → 失败和修正 → 结论限制 → 清理 → 当前未完成与下一动作**。明确区分计划、已实现、执行中、执行完成和核验通过。当前状态页更新覆盖旧状态，历史失败不覆盖；同回合同步PROJECT_PROCESS/BIOSEQ_MODEL_PLAN及对应实验MD，路径均绝对。追加事件和最新机器快照保留原Goal身份，事件不是验收证书。

本轮验证checkpoint=0，自有临时目录已清理；首次清理包装中断及未知文件计数如实保留。原始输出、失败日志、版本证据和真实训练恢复状态保留。停止的注释分支不得恢复，training_ready=false或缺证书时不启动真实语料，独立CUDA算子测试保持暂停。

## 2026-10-01T08:38:15.022797+00:00 私有训练接入支持更新完成

当前支持源码下的R20训练接入准备已完成；两个Python环境各9项新兼容/拒绝检查通过，原launcher全部函数及候选校验函数保持。14个自有临时文件已清理，checkpoint=0。[具体更新、测试和未完成项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_SUPPORT_REBASE_20261001.md)、[原始验收](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_candidate_support_rebase_cpu_20261001/validation.json)。当前仅私有准备，未部署；R21/R24/R26外部训练接入及有效数据审核仍待完成。真实训练尚未启动，新CV方法仍待实现。以下日期记录为当时状态。

## 过程记录与跨agent接手约定（2026-10-01T08:20:56.541520+00:00）

先读[统一接手入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)，再读[过程总账](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)及对应实验MD。顶部当前状态优先于下方带时间的历史记录；遇到冲突，核对明确job原始文件，不能猜测或重跑确认。

每个实质节点都在同一回合落盘：方案/配置定稿、必要检查、实际入队与领取、阶段完成、失败与修正、终态核验、清理以及阻断变化。记录至少包含UTC、维护agent、原Goal身份、状态、实际job/资源、配置或源码版本、绝对证据路径、结论适用范围、未完成项和下一步。失败原件保留，修正另记，计划、执行中、执行完成、核验通过分开表达。

状态快照与追加式事件分别为[EXECUTION_STATE.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json)、[execution_checkpoints.jsonl](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/execution_checkpoints.jsonl)。[节点记录工具](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/update_execution_checkpoint.py)由执行agent在节点发生时调用，不是新增常驻监听；它只能登记观察，不能自行认证验收。最终验收必须关联已完成的原始核验及SHA。本次三任务已核验，[终态原始结果复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_terminal_review.json)与[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_gpu_validation_cleanup.json)保留；不要重复执行。

当前继续原8卡平台任务与Goal，不重启现有监听/worker、不重复提交；只读监督agent已完成本轮职责，不声称仍活跃。仅清理自有临时验证产物；真实续训状态、必要指标/失败记录/版本证据保留。原数据审核、停止的注释分支、独立CUDA算子暂停约束继续生效。

下方日期记录描述当时状态；资源队列、实现/验收进度等以当前接手入口为准。

## 2026-10-01 07:09 UTC 终态只读复核工具就绪

三候选终态只读工具已准备并根端复审，固定3个runtime/manifest和worker Python，强检查仍调用原entry.verify_completion/producer.verify_candidate；两环境各8项CLI拒绝及help/stdlib binding通过。尚无新GPU job，未调用实际终态强复审，不制造通过样例，不重跑旧测试。工具检查及使用见 [根端复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/terminal_tool_root_review.json)，[工具说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_candidate_trainers_cpu_20261001/terminal_review_tools/TERMINAL_REVIEW_TOOL_V1.md)。

## 2026-10-01 06:46 UTC 三候选运行包实际构建与CPU隔离通过

三候选NEW8来源已固定，实际canonical runtime为r21_pilot_v1_20261001、r24_pilot_v1_20261001、r26_pilot_v1_20261001；各524父文件+对应4个NEW文件=528，manifest始终source-only。来源和绝对路径见[实际构建记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/runtime_build.json)，两环境实际导入见[worker环境实际导入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/inspection_worker.json)。旧live路由/8GPU worker未改；GPU验收待路由完成。自有CPU fixture共29046文件/39993940字节已清理，checkpoint=0，小型QA证据保留。

## 2026-10-01 06:22 UTC 继续候选GPU验收和训练整合

R20 v3 actual GPU/根端强复审已通过；继续原计划R21/R24/R26同Trainer验收。新固定三包528文件/父524不变，复用existing model variant的profile-synthetic/steps1以兼容旧worker，不重启GPU池、不新增算子或方法。39历史任务无该组合冲突。当前仅私有实现/CPU准备，未发布/build/入队；数据仍training_ready=false、双证书缺失、真实训练未启动。[具体版本、工程标准与剩余条件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CANDIDATE_TRAINER_GPU_PLAN_20261001.md)。

## 2026-10-01 05:52 UTC v3完整GPU工程验收通过与剩余数据准入

05:44:51 UTC，v3 D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3终态succeeded/rc0。根端05:48以worker同一Python的全新-I进程执行原不可变verify_completion/verify_candidate，通过全部raw/summary/COMPLETES/installed source/独立CPU计数重算；禁止本地CUDA初始化，未重跑GPU。runtime manifest仍cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe，524父文件不变。

实际正式3窗/24 microsteps/768人工记录，formal1054.3747秒<1800，训练loss3.450910→3.448508→3.380331；首窗LR0符合固定warmup，后两窗LR1e-6/2e-6且参数确有更新。独立full oracle706.9962秒<1200、GA/CE/RNG比较通过；四mask评估5.1172秒<600、每mask24条/共96次，eval_batch16含实际尾批及完整2048输入，指标由raw packets独立重算。正式峰值allocated45,350,397,440字节（42.24 GiB），reserved56,555,995,136（52.67 GiB），原85%标准通过；tiny/full/formal清理弱引用释放、最终allocated0。parent1794.0503秒<3600，child rc0、正常退出观察/后代清理true。[强复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/terminal_review.json)、[GPU通过原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959/GPU_RELEASE_COMPLETE.json)。

两个自有空hf目录已删除，checkpoint=0；raw20,049,785字节、summary/日志/完成哈希是后续strong verifier及训练入口依赖，按必要证据保留，v1/v2失败原件不覆盖。[清理清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/gpu_validation_cleanup.json)。首次root提取曾把字段读到不同COMPLETE schema，读取错误已保存在review中，修正后通过，生产代码/标准未改。原8卡任务Running，0/0/39=36历史成功+1新D01成功+2D01失败，8worker维护；本地监听PID2213164继续watching，无新平台申请。

T20 v3私有两环境各48项通过且根端AST复审，source-map私有各23项通过，未publish/实际stage/apply/pointer/入队。[T20私有检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[桥接复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)、[源映射复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/source_map_root_review.json)。GPU验收仅覆盖R20工程入口，不认证R21/R24/R26同版本训练，不证明架构收益。2026-10-01 05:52 UTC数据metadata仍training_ready=false、双证书缺失；审核后manifest改变须另做明确版本绑定，不能仅补文件绕过False。真实四候选1000→累计5000与最终版本训练未启动，goal仍active；独立CUDA算子保持暂停。

## 2026-10-01 05:37 UTC T20 v3私有整合复审与正式窗口进度

T20 v3私有bridge已完成，两环境原48项软件检查通过（20.88/23.50秒），自有basetemp及测试checkpoint已清理。固定绑定runtime v3/cc13036f…08294fe和future launcher manifest的runtime_root/runtime_manifest_sha256；根端AST复审只有reviewed_admission/load_entry增加严格版本绑定，launch、validate_transport、bound_inputs及原数据/续训门禁未变。[私有验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v3_20261001/validation.json)、[根端复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/t20_v3_root_review.json)。source-map另有23项/环境通过，旧28项实际map仍7878dc5c…54f77；尚未发布、实际stage/apply、pointer或入队。

v3 GPU3已完成第1个正式窗口（8micro/256人工记录），LR0/delta0符合固定warmup；后2窗口、四mask评估与parent退出/强完成核验待完成，整体尚未通过。2026-10-01 05:37 UTC只读真实数据manifest仍601db781…ebfce/training_ready=false、audit_certificate/downstream_scope_certificate均缺；未读shard/序列或恢复此前停止分支，最终训练未启动。

## 2026-10-01 05:31 UTC full梯度与清理实测通过，正式窗口运行中

v3新D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959 在原GPU3已完成独立full gradient_oracle：706.9962秒<1200，1窗/8 microsteps/256人工记录，78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194，CE/RNG核对通过。11个对象弱引用全部失效；gc/empty-cache后allocated17,039,360字节，清cuBLAS workspace后allocated/reserved均0；新运行tiny与full阶段均实测支持工作区驻留候选，历史v2仍缺残留数值，不追认其根因。[完整oracle局部实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/full_oracle_observation.json)。

已进入train_optimizer_step/1800秒预算，正式三窗口和四mask评估仍待完成，整体gpu_accepted/raw_completed=false。524父字节、原架构、数值标准和工作量不变；独立CUDA算子暂停。监听PID2213164/原八卡任务保持；T20 v3私有rebase中、源映射未部署，数据双证书仍缺/候选training_ready=false，真实比较与最终训练未启动。验证checkpoint=0。

## 2026-10-01 05:20 UTC v3运行与证据边界

v3于05:04:13 UTC实际发布8个支持/producer文件，05:09:59构建526文件不可变包，manifest SHA cc13036fc492109afbf4c9cf9fd11a4c87684fc9702b83ce86c961b2a08294fe；524父文件保持。producer必要CPU两环境各30项，support各108项通过，实际隔离导入各8 profiles/54项目模块通过。05:12:16本地监听实际重载PID2213164并核验watching，GPU任务/worker未重启。05:14:40只在原八卡池入队D01 1790831680607223314_d01_pilot_runtime_gpu_r20_v1_48f59959，GPU3实际领取。固定tiny preflight耗时1.8074秒、8 microsteps/8人工记录；11个对象弱引用均释放，gc/empty-cache后allocated17,039,360字节，清原生cuBLAS workspace后allocated/reserved均0。当前完整gradient_oracle进行中，formal三窗口及四mask评估尚未通过；仍gpu_accepted=false，不能把局部tiny通过记作完整验收。[发布原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/publication.json)、[固定运行包](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v3_20261001/pilot_runtime_manifest.json)、[实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/d01_v3_launch.json)、[tiny阶段实测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_lifetime_v3_cpu_20261001/lifetime_preflight_observation.json)。

T20 source-map私有两环境各23项通过，原28项映射保持；[软件源绑定记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_source_map_cpu_20261001/validation.json)。T20新私有v3正在显式rebase固定root/manifest，未发布/apply/pointer/入队；原数据准入全部保留。候选training_ready仍false、双证书未齐；真实四候选1000→累计5000与最终训练未启动，goal active。独立CUDA算子仍暂停。自有CPU临时目录已清理，人工GPU关闭checkpoint保存，小型指标/失败原件/源码哈希保留。

## 2026-10-01 04:44 UTC v3私有工程准备

v3 canonical已确定，root固定路径/guard/typed必要CPU两环境各108项通过；初次basetemp设置失败保留，修正父目录后复测，两个自有fixture目录2626文件已清理。实际tiny CPU对象弱引用全释放；PyTorch2.8原生cuBLAS workspace是待GPU验证的具体候选，producer正补释放前后allocated观测及固定120秒tiny preflight，原full工作量/数值/显存标准不变。仍未发布/构建/入队v3、T20未部署、最终训练未启动。[v3私有定位与必要验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_LIFETIME_V3.md)。

## 2026-10-01 04:24 UTC 失败版本保留与生命周期修订

v1/v2固定runtime及两次失败证据保持不可变。v2子进程正常退出/后代清理为true，进程内GPU张量驻留检查仍失败；不能互相替代。修订须先证明对象释放、保留实际allocated观测，再另建版本及复审来源，不热改旧包/追认通过。当前本地监听PID2172045，原八卡worker保持。 [整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。

## 2026-10-01 04:07 UTC 当前运行包与监听

当前D01选择固定v2包，manifest3805679f…，父524文件保留。D01 variant仍为transport v1，真实版本必须以request.runtime_root/manifest SHA/job_id区分。typed监听只接受固定canonical v1/v2目录，本地PID2172045实际watching。旧v1失败目录不覆盖、不热改；GPU未终态时不切live支持源码。T20只在新私有目录rebase和验收，最终应用等待D01终态及来源复审，数据原准入保留。[实际版本与路径](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。

## 2026-10-01 03:26 UTC 运行包预算修订

D01 v1失败证据和immutable runtime保留；新版本目录固定r20_r27_pilot_v2_20261001，时限调整必须写入新源码/plan/manifest后再运行，不热改旧快照或在失败后扩大同一通过标准。D01是固定transport v1合同，真实运行包以目录和manifest SHA区分；typed监听须分别读取v1/v2的固定来源。T20仍未部署；新版本source变化需要明确重绑定，不能复用旧GPU proof。

## 2026-10-01 03:18 UTC 当前训练整合入口

[预训练整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)为当前状态入口。固定runtime已实际创建、D01路由已部署；新本地监听PID2143672每30秒监督原八卡池，GPU任务/worker不重启。D01仅人工Trainer验收，T20语料桥接另行实现和来源绑定；原R20 mode=train不能当作已经接通新Trainer。

数据原preflight优先于GPU证明、模型和训练输出；没有数据双证书不能启动语料训练。测试/验证checkpoint关闭保存或在验证结束后按ownership删除，保留小型结果和源码证据；清理前保护实际恢复依赖，不能广泛删除其他会话目录。独立CUDA算子测试仍非当前前置条件。

## 2026-10-01 当前架构与固定 Trainer 入口

整体架构入口为 [IMMUNE_CURRENT_ARCHITECTURE_20261001](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CURRENT_ARCHITECTURE_20261001.md)。原主干、各版本机制、当前候选和实际验证范围集中在此；新工程检查与人工loss不当作下游收益。

[固定runtime入口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_entry_v1.py)的池参数只运输D01人工验收，不允许任意data/resume/命令。producer仅写candidate，正常退出与后代清理、强核验全部通过后由parent最后写完成记录。通用process helper的复用不代表执行CUDA算子专项。第一版新GPU证据计划只覆盖R20，不自动准入其他架构。实际数据API先执行原preflight；审核不完整时不构造模型或训练输出。[验证范围](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINER_RUNTIME_ENTRY_V1.md)。

新源码存在并发漂移时保留完整字节、停止竞争写入，在专属私有副本继续实现；最终源发布前独立review并核对原址SHA，不覆盖别人后续修改、不热改冻结release。不凭进程列表认定或停止未知writer。新磁盘typed展示不会自动刷新已运行daemon，待实际部署时单独替换本地监听；不重启八卡worker。

## 2026-10-01 已删除的历史入口

根目录旧任务导出、旧 grammar_v2 看护 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/watch_llada_train_jobs.sh` 及其状态配置、两份依赖缺失的 debug 脚本已按用户要求删除，详情见 [清理执行记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/PROJECT_CLEANUP_AUDIT_20261001.md)。任务账本里的原调用命令仅为历史记录，不能作为当前入口。Python/pytest 缓存沿用忽略规则；论文 `.aux`、`.fls`、`.fdb_latexmk`、`.blg`、`.log` 已加入忽略，保留 `.bbl`、源文件和交付物。现有训练/评测看护继续使用各自当前入口。

## 2026-09-30T23:31:26+00:00 独立验证的生命周期和累积窗口

[随机CUDA探针](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/CUDA_ATTENTION_PROBE_README.md)已完成两环境各59项CPU验收。Linux主线程内使用同session的case进程组和临时subreaper，只回收本case；退出/超时的最多2秒清理在原预算内，失败停止后续case。父SIGKILL或主动脱离组的后代不作普遍保证；实际CUDA尚未执行，原47项证据为历史版本。将来接pool仍需独立typed分派、完成状态与监督适配，不能直接入队任意shell。

[RecordWindowTrainer原型](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RECORD_LOSS_FRAMEWORK.md)仅支持已核验版本的CPU人工输入；原生num_items_in_batch始终是token数，以独立元数据记录实际累积窗口记录数。原生training_step/backward数值验收不等于生产Trainer改造，真实GPU精度/优化器/恢复仍需单独证明。改变目标或窗口规则须独立版本并对对照组使用一致规则。源文件、框架方法SHA和逐窗口小型JSON保留，临时pytest目录清理。

## 2026-09-30T23:06:15+00:00 下一版本的执行与指标边界

[protocol.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)是planned_not_deployed的研究协议，尚无读取该schema的执行器，不能传给Volc或现有pool。继续用已分配八卡；本机Volc CLI没有task exec入口，现有worker动态读取registry并每job启动入口的行为只能在当前任务结束、旧release冻结后作受控扩展。不得把任意shell塞进旧job。诊断case必须具备父退出/超时的有界回收，实际支持程度以[执行审阅](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXISTING_POOL_DIAGNOSTIC_EXECUTION.md)与CPU进程测试为准。

[评估v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)将token计数与链终点计数分开；首次END搜索包括所有有效目标容量槽位，不能用真值裁剪。partial-mask首次END指标无定义；missing不记零长度误差。relation监督单列，未知监督位置拒绝。九项token指标可做四mask均值，三项full-mask终点指标不可混入。所有指标协议、源码SHA与固定selection必须随run绑定，历史召回日志不得事后改名。

新[pilot适配器](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_CERTIFIED_PILOT_ADAPTER.md)仍为隔离工厂，不执行train/evaluate/save。两环境各90项CPU测试不能代替新GPU发布、数值续训或数据认证。每记录loss及温度采样也都未部署；变更目标、采样、后端或评估需独立版本，不能复用旧成绩。只删除自有测试临时文件，保留真实resume依赖和小型审计证据。

## 2026-09-30T22:15:16+00:00 后续研究发布与评估约定

当前8项500步实验冻结39文件，所有新评估/后端/采样研究保留在隔离原型，直到当前stage结束后再建立独立release、重新profile并验证。训练/验证指标的归一化协议必须随源码SHA记录；最大microbatch因架构/后端变化时，不能假定相同global256自动带来完全相同loss权重。[数学依据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_NOISE_OBJECTIVE_DIAGNOSTIC.md)。

未来真实pilot绑定[固定selection清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_VALIDATION_SELECTION.md)的manifest、全部valid分片、精确source-local索引和selection SHA，使用[完整指标协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TOKEN_VALIDATION_PROTOCOL.md)记录AA/end、macro/micro与四mask；小来源/低family不能静默隐藏。原型只验证了单进程CPU，CUDA/分布式不能未经实测宣称支持。

新候选目录为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_expanded_candidate_20260930`；training_ready=false，缺双证书，现有assert_ready拒绝。保留原数据及必要新旧映射，测试临时目录/无用checkpoint清理；不把候选物化与完整性验收当作覆盖认证。[状态与产物](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXPANDED_CANDIDATE_EXPORT_20260930.md)。

## Historical job YAML layout

当前配置与被代码依赖的模板继续放在 [训练目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/README.md) 和 [评测目录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/README.md)；旧配置的路径映射与恢复方式由 [归档索引](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/job_yamls_20260930/README.md) 维护。整理时须检查源码、动态文件名、看护配置及非终态任务；不能仅按日期移动仍被读取的 YAML。保留文件原内容、SHA256 与原路径映射，文档链接随移动更新。Git 对本地生成配置的忽略规则保持，归档备份范围见索引。

## 2026-09-30 下游报告与源码规则

派生报告写新目录，未知训练语料绑定不得标 clean；论文主列与本地复现分开，缺证比较禁止排名。具体执行规则统一维护在 [报告与 CPU 验证约定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#report-integrity)。第一方评测源码白名单和源码副本检查须与任务入口一起维护，weights/vendor/data 不入源码发布集合。

## Long-Term Rules

- 2026-09-30：MHC采用原序列优先；直接提供的长链/多链保持边界，仅HLA名称且唯一可映射时补34位pseudo。构建入口 [update_mhc_inputs.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/update_mhc_inputs.py)；新v3数据 [immune_v7_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1)，旧v6对应更新 [immune_v6_mhc_sequence_v1](/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v6_mhc_sequence_v1)。pilot配置生成器默认新v7；直接调用训练入口显式传新版`--data-dir`。未改写来源以相对符号链接依赖parent，不能删除/移动parent后当独立归档。MHC验证不替代既有数据/下游范围证书，不可给旧checkpoint改名冒充新版训练。见 [数据更新记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#2026-09-30-mhc序列优先数据更新已完成)。

- 2026-09-23: source-balanced training must use manifest shard groups carried as runtime `source_group` / batch `source_groups`. Keep row `source` as provenance; it can differ from or overlap across groups (`tcr_papers` rows may say `tcr_native`). Prepared data and rendering seeds are not rewritten to normalize these names. See `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`.
- 2026-09-22: current v2 launch configurations opt into `--source_balanced_training True`. Balance is over the global optimizer batch, with full per-source permutation coverage and oversampling of smaller sources, not truncation of larger sources. Current v6 / batch 256 needs 67,962 updates for the first full coverage. Existing overflow exclusions are counted, not trained. `source_sampling.json` fixes data/source/seed/batch topology for resume; do not silently switch old checkpoints to this new objective. Full contract: [model plan](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md).
- 2026-09-22: top-K retention restores the run's `topk_val_manifest.json` before pruning, recovers missing entries from exact-step Trainer history, protects unranked checkpoints, and atomically publishes the manifest. Corrupt/path-mismatched manifests fail closed. Independent evaluation snapshots remain outside this retention system and must never be pruned. Worker drop metrics now travel as per-batch deltas; training counters are rank-local consumed-row counts, not full-corpus/global counts.

- 2026-09-15 用户明确：任何用于 eval 的 checkpoint 必须在评测前保留独立、按版本/step 命名的权重快照及加载所需配置/tokenizer，记录源路径和 SHA256；评测指向该固定快照。快照不参与训练 top-k/最新 checkpoint 轮换，不得被新 checkpoint 覆盖或自动清理；清理已评测快照须用户明确授权。AB 已保存的 step49000、step92000 位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/`。

- 当前 AB 结果汇总必须携带论文 baseline，来源及缺失值处理统一遵守 [downstream PROJGUIDE §0.2.2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/PROJGUIDE.md#022-排行榜构造论文值优先强制2026-09-02-起) 的2026-09-15用户决定。

- TCR 正式生成的严格 JSON 缺失指标契约、不可覆写的 CPU 评分恢复及新 gate 规则见 [T4 §7.7 / §8.0](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/TCR_T4_GENERATION.md)。已评分失败的完整生成不得自动重采或以成功标记覆盖；后续关键修复继续登记总审计。

- 后续测试任何已构建 TCR 任务（T1–T4），必须先读 [TCR baseline 审计 §9.8](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 与对应任务手册，遵守任务专属输入/验收约束；所有关键改动同回合回写该审计。该规则不把待执行实验视作已完成，也不替代现有结果/范围/任务账本。

- The project root is `/vepfs-mlp2/c20250601/251105016/project/dllm_test`.
- The model weight root is `/c20250601/mj/model_weights`.
- All project docs, configs, examples, and scripts must use absolute paths.
- Plan changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md`.
- Process changes must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`.
- Long-term project rules must be recorded in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJ_GUIDE.md`.
- Ophiuchus-Ab baseline 表征探针的纠错与后续验收先读 [AB baseline 测评指南](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/AB_BASELINE_EVALUATION_AUDIT.md)；该入口不替代现有数据台账、排行榜或范围定义。
- Ours 的原生 AB 表征测评走 [AB_NATIVE_PROBES](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_NATIVE_PROBES.md)，不用 Ophiuchus 固定窗口或默认 TCR record。`.gitignore` 只放行本次新增的 `ab_features.py`、`ab_probes.py` 和该任务文档，其余未跟踪 downstream 资源继续忽略；配置生成器必须显式传 checkpoint，不能自动提交或猜选权重。
- 训练中 checkpoint 的 AB 矩阵统一固定到 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/ab_eval_checkpoints/<run_tag>` 独立评测快照；复制前后 SHA 一致，仅保留评测所需权重/tokenizer等，不复制优化器。生成器 `--with-preflight` 输出独立 GPU gate，先通过再提交正式矩阵；执行细则见原生 AB 任务 §8，最新权重选择不等于验证集最优选模。

## Active foundation-training data boundary

TCR 下游后续关键决策、代码改动与验收必须在 [TCR baseline 审计 §9](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md) 同回合登记；其输入转换约束见该文档 §2 / §4，others 与 β-only 的入口/缓存边界见 §9，不得把下述离线训练数据管线要求直接套到评测数据上。

- The only current immune training implementation is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; the formal entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- The active data contract is raw source → adapter → `BioSeqRecord` → offline deterministic filters → prepared semantic JSONL/manifest → training-time grammar, padding, per-chain encoder reconstruction, and diffusion/MLM masking. No model-ready token cache is produced.
- Current prepared roots: **v4** `data/prepared/immune_v4_beta_relation` (published) and **v5** `data/prepared/immune_v5_receptor_completion` (published). v3 / `immune_v3_heterotypic` remain on disk as live artifacts some checkpoints still point at; they are not the current training default. v4 headline counts live in `docs/PLAN_TCR_BETA_ONLY_RELATION_DIFFUSION.md` §4.1; v5 counts / v4 Δ live in §4.2. Operational how-to: `dllm/pipelines/immune_llada/README.md`. Layout: `DATA_FORMAT_AUDIT.md`.
- Receptor completion is gated by `tcr_region_profile.completion_sources` (default `("tcr_repertoire",)`). Withholding a source from that list is what disables completion. Do not branch completion on the `sequence_scope` label — use actual `alpha_fv` / `beta_fv` / `cdr3a` / `cdr3b` contents. Do not infer junction vs core from the first/last character; use provenance. `add_tcr_region_lengths` must skip `synthetic_regions` (circular-profile guard). `sample_region_lengths` must sort support lengths numerically (`atomic_json_dump` uses `sort_keys=True`). Drops that need audit attribution must be named filters (e.g. `quality.blank_epitope`), not silent adapter `None`. Manifest `filter_names` contract (union of constructed filters, not `BLOCKLIST_NAMES` keys): `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md`. Item-13 counter numbers (`downgraded_all_x_mhc` / partner completion): plan §4.2.
- The active recipe remains immune-receptor-focused: antibody H/L, TCR α/β, antibody-antigen, and TCR-epitope/pMHC. Do not make deleted PPI/STRING/MINT training builders, nanobody/VHH, antigen-free neutralization, or specificity-free bulk TCR into current foundation-training inputs.
- The old `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/data` alias tree, old `training` tree, old raw-CSV dataset implementation, and retired training entry points are deleted. Do not restore them or document them as current dependencies. The remaining `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py` are model-layer compatibility code used by the fusion model and selected diagnostics.
- The former AB/TCR v2 canonical work under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/immune_receptor_v2` and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/IMMUNE_RECEPTOR_DATA_V2.md` remains a historical/candidate data artifact. It is not the current runtime training input while its export/training gates are false.
- Historical audit snapshots under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/refactor_baseline` and historical evidence under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive` must remain readable and must be labeled historical, not silently deleted or presented as executable current pipelines.
- Data layout and schema facts must stay synchronized with `/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md` and the independent pipeline description at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/README.md`.

## MINT benchmark rules

- MINT GeneralPPI 的 canonical 范围只有五项：`HumanPPI`、`YeastPPI`、
  `Bernett`（显示名 Gold-standard PPI）、`MutationalPPI`、`SKEMPI`。
  `Pdb-bind` 和 `MutationalPPI_cs` 不得加入默认注册表或全量下游编排。
- Canonical 数据根必须是
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official`；
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint` 仅为历史诊断数据。
- embedding cache 名必须包含 `mint_official_06694b7` 数据协议标识；数据版本切换后不得
  复用无该标识的历史 `.pt`。
- 每项数据必须由固定 MINT commit 中该目录的 `prepare_data.ipynb` 原样逐 code cell
  生成，并通过 notebook/input/output SHA256、schema、行数和 split 校验；不得用本地
  “等价实现”覆盖 canonical 输出。
- MutationalPPI 和 SKEMPI 的模型输入必须同时保留 WT 与 mutant 序列，并使用两者的
  PPI embedding 差；不能沿用公开 `MutationalPPI` 路由中只读 WT pair 的行为。
- 论文、notebook、公开评测代码不一致时不得猜造未发布 split。当前混合主表固定为：
  HumanPPI、YeastPPI、Gold-standard PPI 使用论文 Source Data `[P]`；MutationalPPI、
  SKEMPI 使用提交的本地固定 split 重跑全部 8 个 Figure-2 模型并标 `[L]`。后二者不得
  用论文值静默补空，也不得称为 paper-exact。
- `[L]` cache 必须同时带 `mint_official_06694b7_localfixed-v1-l2048`；MutationalPPI
  protocol 固定为 `mutppi_pair_group10_sgkf_seed0_v1`，SKEMPI 固定为
  `skempi_notebook_complex3_mt19937_v1`。选表入口只认这两个 protocol、完整数据、
  separate-chain、WT/mutant 对齐窗口和 3-repetition 640-hidden MLP；run cap 为 2048，
  但 ESM-1b 与 ProGen2-Large 必须按各自原生限制使用 1024，且逐模型记录实际长度。
- MINT 混合结果唯一机器可读入口为
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/mint_tasks/_selected_baseline_results.csv`；
  `_baseline_audit.json` 中存在 `missing_local_fixed_models` 时不得宣称五任务 baseline 已完成。
- Canonical 说明与已知差异位于
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/mint_tasks/OFFICIAL_REBUILD_AUDIT.md`，
  machine-readable provenance 位于
  `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/downstream/mint_official/manifest.json`。

## Volc job layout

- **Training YAMLs** live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/`; current training/resume jobs, research configurations and templates still read by source code belong here. Old `grammar_v2` YAMLs are historical and must not be used to infer a runnable pipeline.
- **Eval YAMLs** live under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/`; downstream, pairing, and metrics jobs belong here. The old grammar-v2 generators/wrappers/sweeps/retry jobs are retired and must not be regenerated.
- TCR 简单任务优先本地；用户现已授权 others 完成后将剩余仍排队任务也接续到本机，见 [TCR 审计 §9.24](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/TCR_BASELINE_EVALUATION_AUDIT.md)。本地顺序执行并检查 GPU 占用，不干预其他进程；已实际云端 Running 的任务保留。迁移前核对同范围云端 ID/名称、取消 Queue 并确认 Killed，避免双跑；已完成 tag 不覆盖。执行位置不改变数据、epoch、fold 或候选预算；§9.3–§9.5 及 §9.23 的资源选择是历史，不自动重提已取消任务。
- Current local evaluation examples are the retained immune-fusion scripts under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream`: `run_immune_fusion_gen.sh`, `run_immune_fusion_pairing.sh`, `run_immune_fusion_repr.sh`, and `run_pairing_pll.sh`. They use the active fusion checkpoint line and the retained public implementations under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar`.
- Pairing's current input/default/provenance contract and separate scoring entry are maintained in [AB pairing §4.3 / §7 h](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_LIGHT_CHAIN_PAIRING.md). Use that owner instead of historical prior-only comments; do not silently resume an old generation protocol.
- Submit commands, when a current job is explicitly approved, use absolute YAML paths:
  - Training: `volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/<job>.yml`
  - Eval: `volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/eval_jobs/<job>.yml`
- Historical grammar-v2 checkpoints, wrappers, sweeps, and retry jobs remain only as provenance where archived; do not run them, regenerate them, or append their numbers to new headline tables. Do not alter external baseline protocols.

## Training batch sizing (Volc)

- 2026-09-23: additional c20250601 v2 fusion configuration `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/protein_esmc_llada270m_diffusion_v2_immune_v6_2node4gpu_c20250601.yml` retains bs4 × accumulation8 × 8 GPUs = global 256 and enables equal-source training. Submitted with `mem_peak pending profile`: the local A100 was occupied, and CPU preflight does not establish GPU/FSDP capacity. Full submission evidence and limits are in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md` (2026-09-23).
- **Same experiment, different encoders** (e.g. LLaDA ESMC-300M vs 600M): use the **same global effective batch** (`batch_size × grad_accum × world_size`) for fair comparison.
- **Maximize GPU memory**: on the target Volc flavor, increase per-GPU `--batch-size` until near OOM (~75–85% `mem_peak`), then set `--grad-accum` to hit the agreed global batch. Do not keep a legacy micro-batch just because an old run used it.
- **Before every submit**: profile short run on matching GPU; record `mem_peak` in YAML `Description` and/or `PROJECT_PROCESS.md`. Rule detail: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/.cursor/rules/volc-batch-sizing.mdc`.

## BioSeq Pipeline Boundary

- The active immune data pipeline is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada`; its formal training entry is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/protein_pretrain_esmc.py`.
- The active prepared-data path is `raw → adapter → BioSeqRecord → offline filter → prepared semantic JSONL → training-time grammar/padding/per-chain encoder reconstruction/masking`. The prepared loader still builds offsets and decodes semantic rows; it is not a zero-cost token cache.
- The current model-layer compatibility code retained under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch` is limited to `modeling_bioseq.py`, `sampling_bioseq.py`, and `relation_aux.py`. The deleted `data` alias tree, deleted `training` tree, deleted grammar document, and deleted legacy entry points must not be documented as active dependencies.
- `examples/llada/protein_fusion_model.py` reuses the retained model-layer functions for diffusion/MLM corruption, encoder-state mirroring, and relation auxiliary loss. The current immune data grammar/collator lives under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.
- Public downstream grammar remains active where the fusion scripts use it: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar` retains the shared CDR, light-chain pairing, and TCR-generation implementations. Do not describe that directory as entirely deleted.
- Retained typical downstream entry points are `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/downstream/run_immune_fusion_gen.sh`, `run_immune_fusion_pairing.sh`, `run_immune_fusion_repr.sh`, and `run_pairing_pll.sh`; external baseline protocols and artifacts remain unchanged.
- Historical identifiers such as `qwen3_vl_arch`, `qwen3_vl_bioseq_*`, and `bioseq-qwen3-vl` may remain in old paths, task IDs, output directories, or archived records, but they are not current training instructions.

## Downstream Feature Extraction (post-LLaDA, mandatory)

- **All downstream tasks must extract features by passing sequences through the final LLaDA decoding architecture (post-LLaDA), not only the ESMC encoder's last layer.** This applies uniformly across every task family and both task modes:
  - Task families: MINT PPI (protein–protein interaction), antibody / nanobody (AB), and TCR.
  - Task modes: generative (e.g. CDR infilling, light-chain pairing, TCR generation) and predictive / embedding (e.g. binding classification, clustering, few-shot representation, PPI heads).
- Rationale: ESMC-encoder-only features capture only the frozen backbone and do not reflect the LLaDA decoder training that defines our model. Encoder-only extraction understates our contribution.
- Practical guidance:
  - Generative tasks are post-LLaDA by construction (decoding runs the full fusion model = ESMC encoder + LLaDA decoder), e.g. `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/grammar/cdr_infill.py`, `light_chain_pairing.py`, and `tcr_generation.py`.
  - Predictive / embedding tasks must use the post-LLaDA decoder path. Current fusion runners use the explicit `grammar:decoder:global:/absolute/path/to/checkpoint` specification; do **not** use encoder-only controls for headline numbers.
  - Encoder-only runs may be kept only as explicitly labeled controls (distinct tag), never as the primary reported result.

## External Baseline Protocol (original-only)

- For Nature Methods 2025 T1 binding, an external model baseline must use the author's original checkpoint, the author's inference code, and test rows derived from `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ntmethod_binding/original.zip`. It must not load `train.csv`, fit a head, or retrain the baseline.
- If the original checkpoint/code/environment is unavailable or the official inference fails, use the paper's original-model value and label it exactly **“论文值，未本地复现”**. Never replace it with a locally retrained approximation, random score, or another checkpoint variant.
- The checkpoint/output tag mapping is part of the protocol. Multi-checkpoint wrappers (TEINet, ERGO, TPBTE) must bind the canonical variant tag before importing the wrapper; an output directory name alone is not evidence of which checkpoint ran.
- Canonical T1 source decisions live in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025/summary_original_baselines.csv`. Local-only diagnostics live in `summary_local_runs.csv`; paper transcriptions live in `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/external/paper_reported_baselines.csv`.
- Nature Methods 2025 retrained artifacts are audited separately under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_binding_nm2025_retrained`. Never merge them into the canonical original-model row. Verify both official archive MD5 values before inference, pair fold N only with fold-N checkpoint/data, use `precrec`-compatible AUPRC, and retain local plus paper values side by side. If both five-fold mean AUROC/AUPRC do not match the paper at four decimals, the selected retrained reference must be the paper value labelled “论文值，发布 artifact 未精确复现”. TCR-H may read fold train only to rebuild its released descriptor feature mask; `training_performed` must remain false. Officially dropped rows (currently ERGO-AE tail batches) must be counted explicitly rather than hidden.
- When placing Ours into the retrained-fold comparison, freeze every BioSeq backbone parameter and cache only `GrammarEmbedder(feature_source="decoder", pool_mode="global")` features from a joint peptide–CDR3β record. Train one independent MLP on each official fold train; validation must be carved only from that train. Do not reuse the old normalized `downstream/benchmark/data/tcr_binding_nm2025/train.csv` for this comparison.
- Rank Ours first against the local official-checkpoint reruns on the same `retrain.zip` members. Keep paper-reported retrained means in a second, explicitly mixed-source reference ranking. Preserve both AUROC and `precrec` AUPRC ranks and never silently substitute the paper value into the same-release primary table.

## Weight Layout

```text
/c20250601/mj/model_weights/esmc/ESMC-300M
/c20250601/mj/model_weights/esmc/ESMC-600M
/c20250601/mj/model_weights/esmc/ESMC-6B
/c20250601/mj/model_weights/esm2/esm2_t6_8M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t12_35M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t30_150M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t33_650M_UR50D
/c20250601/mj/model_weights/esm2/esm2_t36_3B_UR50D
/c20250601/mj/model_weights/ophiuchus_ab/Ophiuchus-Ab
```

The environment variable `BIOSEQ_MODEL_WEIGHTS_ROOT` may override the root, but its default must remain `/c20250601/mj/model_weights`.

`/c20250601/mj/model_weights/esm2/esm2_t48_15B_UR50D` is optional and is not part of the current default download set.

Current ESMC environment note: local ESMC checkpoints declare `model_type="esmc"`, but `transformers==4.48.1` does not recognize that model type. The retained model-layer loader is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/qwen3_vl_arch/modeling_bioseq.py::BioSeqEncoderDiffusionModel.from_esmc` / `load_local_esmc_encoder`; those paths fall back to Biohub `esm==3.2.3` and load local safetensors without relying on `AutoModel` alone. Active ESMC tokenization is under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data/esm_encoding.py`.

Historical BioSeq foundation stage-1 runs used offline wandb and output roots such as `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc300m_stage1`, `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_esmc600m_stage1`, and `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/qwen3_vl_bioseq_no_encoder_stage1`. Preserve those paths as provenance only; current training must use the approved immune-fusion entry and output configuration. Do not use `/tmp` for ESMC training outputs or checkpoint tests because the root filesystem can be full and ESMC checkpoints are multi-GB.

## Historical PPI Data Layout（audit only; not current foundation training）

- PPI and interaction task raw data root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw`.
- PPI and interaction task processed outputs:
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_sources_manifest.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_summary.csv`
  - `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/ppi_task_raw/processed/interaction_records_unified.csv`
- Historical rebuild command, retained for audit provenance only:

```bash
python /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/build_ppi_interaction_csv.py
```

- `interaction_records_unified.csv` is an audit/consolidation table. It is not a current training input; current foundation training uses prepared semantic records under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/data`.

## Public epitope-conditioned CDR3β Track-A rules

- The authoritative protocol is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/TCR_BETA_PUBLIC_TRACK_A.md`; do not mix its table with the older T4 three-setting metrics.
- Generate and retain exactly 1,000 raw candidates per model-target before filtering. A valid CDR3β contains only 20 standard amino acids, is 7–24 aa, starts with C, and ends with F or W. Never add anchors or otherwise repair a raw model output.
- References are target-specific values from `benchmark_data_w_preds.csv::reference_translations`; prediction columns are never references. Exact and recovery hits must remain within the same pMHC target.
- Run TcrDesign only in beta mode for the Track-A ranking. A separate qualitative artifact may run the official generated-beta-conditioned alpha module, but its beta inputs must come from the canonical generated `generations.csv`, never from reference/oracle β. TCR-epiDiff remains `reconstruction_only` unless the authors release a genuine pure-random de-novo sampler; never seed it with a test reference for a de-novo comparison.
- All comparable models use the same external exact/recovery/GIANA/Jaccard evaluator. GIANA is run per model-target with generated and reference sequences in the same pool and without V genes.
- TCRT5 `rank` and `raw_score` must be based on cumulative transition log-likelihood, not the length-normalized Hugging Face beam score. Every TCRT5 rerun must refresh `tcrt5_paper_consistency.{csv,md}` and verify candidate-set, exact-hit, and RVR GIANA agreement with the author artifact/paper.
- Do not compare the canonical all-14 TCRT5 exact-hit total directly with the paper's sparse-13 aggregate: the paper treats `RVRAYTYSK_HLA-A*03:01` as a separate deep-simulation experiment. Likewise, label the paper-native same-length identity recovery separately from Track A normalized-Levenshtein recovery.
- **RETIRED (2026-09-08): the `grammar_v2` BioSeq row is no longer reproducible.** All `grammar_v2_*` checkpoints were deleted during storage cleanup, so the 7-layer `step117000` pin (`output/grammar_v2_esmc300m_integrated_llada_7l_step117000/best.pt`, SHA256 `e3d4c98ea84b4aa2280fe76ec5bf8a806de376699610e9a96960e66ceddb5c01`) and the `step189000` RVR diagnostic now reference absent files. Its training corpus `data/bioseq_grammar_v1` was already gone, so the run cannot be retrained. Treat the recorded numbers as frozen historical artifacts: existing reports under `outputs/tcr_beta_public_benchmark/` and `checkpoint_comparison/bioseq_step*/` stay readable, but no `grammar_v2` row may be regenerated, re-scored, or added to a new table. Any future local BioSeq row must come from the `examples/llada` immune-fusion line. Retired provenance (top-k manifests, wandb summaries, deleted-file inventory) is archived at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/grammar_v2_retired_20260908`.
- Checkpoint diagnostics for the active line must use the same fixed protocol, derive the model label from the validated checkpoint manifest, and write under `outputs/tcr_beta_public_benchmark/checkpoint_comparison/<run_name>`. Never append a single-target or exploratory checkpoint run to the canonical all-14 tables.
- Canonical outputs live at `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results`; every rerun must refresh `reproduction_status.md` and retain runtime provenance.
- TcrDesign qualitative α output is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcr_beta_public_benchmark/results/tcrdesign_alpha_generations.csv`; the official-release audit is `tcrdesign_alpha_evaluation_audit.json` in the same directory. Until an official runnable evaluator is released, keep `Official alpha evaluation: unavailable` and `Included in quantitative benchmark: no`; do not invent alpha exact/recovery/pairing metrics or use TcrDesign-B as a common αβ evaluator.

## TCRT5 full-eval canonical path and rules（2026-07-23）

- Full runner: `/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/tcr_generation_bench/run_tcrt5_full_eval.py all --with-pgen --reference-pgen --pgen-workers 16`.
- Canonical root: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval`; the human-readable authority is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/TCRT5_FULL_EVAL_REPORT.md`, and the machine-readable paper comparison is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval/paper_correspondence.csv`.
- Never merge the paper's top-20 `K=100`, sparse13 `K=1000`, and separate RVR `K=1000` into one paper aggregate. The current all-14 comparison is an engineering view and must retain its own protocol label.
- For native recovery, use Levenshtein only to choose the closest same-length reference, then score positional Hamming identity; only use reference-length-normalized Levenshtein when no same-length reference exists. Do not substitute the common Track-A `1-Levenshtein/max(lengths)` formula.
- Do not report a paper-compatible mAP unless cumulative model-log-likelihood scores determine rank. Main-release ordered lists and models with scoreless emission order may expose AP diagnostics, but their evidence field must say proxy/diagnostic.
- Char-BLEU requires the greedy hypothesis and 20 nearest references. The main release contains stored numeric Char-BLEU but not the greedy strings, so it is a numeric companion check, not an independent recomputation from released hypotheses.
- OLGA reports must retain positive fraction plus positive `log10 Pgen` moments. The Fig.4 moments are not strictly reconstructable from the published candidate lists and stated protocol; keep the all-positive and `[-16,-5]` sensitivity rows, and never tune an undocumented cutoff to force agreement.
- After evaluator changes, run `/vepfs-mlp2/c20250601/251105016/conda/envs/pllm/bin/python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/scripts/test_tcrt5_full_eval.py` and refresh all manifests under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/outputs/tcrt5_full_eval`.

## 2026-09-24：AB-AG baseline 原代码约束

新任务按 [AB-AG 官方协议手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md) 执行。只测 conventional paired VH/VL；不启动含 VHH 的全量默认配置。官方代码锁和逐方法入口位于 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen`；第三方 checkout、数据、环境、权重保持在忽略目录。原入口的路径适配须记录；改变模型、head、split、打分或聚合须明确为新协议，不能冒充作者 baseline。CLI help 和静态校验不构成模型复现。

## 2026-09-24T03:54:43.033669+00:00 FLAb2 排除决定

FLAb2 不再属于 AB-AG 候选或试跑对象，默认 source lock 已移除。历史 checkout/审计保留为档案。MAGE 保留，继续遵守原代码/原权重及无 VHH 约束；见 [AB-AG 手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。

## 2026-09-24T05:06:03.673383+00:00 BioMap 原代码运行路径

恢复路径和运行边界见 [AB-AG 执行手册](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md#86-biomap-a2binder-原代码运行)。依赖隔离在 `palm_env`，权重路径由 acquisition manifest 管理；不更改 pllm 包，也不在本地 GPU 执行训练/评测。

### 2026-09-27 v2 eval monitor runtime

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_v2_eval_spot_start.sh` runs the monitor with the pllm Python interpreter (PyYAML required). Entrypoint metadata is parsed with YAML; do not manually unescape multiline YAML. Completed pairing CSVs are kept for validated score-only retry.

### 2026-09-27 Model display names

Reported model versions follow the training data: immune_v6_binding_only → V6. Preserve v2 in historical paths, task IDs and fixed-canvas architecture identifiers. Do not rename files/checkpoints just to change a results label.

## 2026-09-27 AB-AG sequence v1

AB-AG 纯序列数据与运行边界见[任务手册 §8.7](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/tasks/AB_ANTIGEN_BENCHMARKS.md)。Ours 实现目录 `downstream/benchmark/ab_antigen/sequence_*.py`；GPU 仅 Volc。首轮 output `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/downstream_generation/ab_ag_sequence_v1_noesmc150000`，不覆盖历史 BioMap/MAGE 任务。

### 2026-09-27 AB–Ag 统一评测入口

四子集接口与运行说明：[ab_antigen/README.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/ab_antigen/README.md)。interfaces 提供 CPU validate/export/evaluate；native_adapter 接入已固定 no-ESMC150000 zero-shot；冻结探针继续使用 sequence_eval。模型推理仍通过 Volc 执行，接口验证不代表模型结果。
# 2026-09-30 no-ESMC v3 execution paths and gates

- Model/layout: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/v3/`; training entrypoint: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/examples/llada/immune_pretrain_v3.py`.
- Dataset audit: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/data/prepare_immune_v7.py` (`freeze`, `inventory`, `cluster`, `coverage`, `export`, `verify`); immutable registry under `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/immune_v7_20260930`.
- Training must pass `assert_ready`: hash-bound audit certificate, zero forbidden overlaps, no pending evidence. A directory named `immune_v7_clean` alone is not authorization to claim clean data.
- Global effective batch remains 256. Architecture profile/pilot now use queue012; template `/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_profile_20261001_queue012.yml`. Use the completed profile evidence and existing compatibility gates to select micro-batch/accumulation.
- Local `pllm` lacked accelerate; installed accelerate 1.14.0 without dependency upgrades to match the existing production environment. Legacy tests that require peft run in the existing `protenix_abtcr` environment after activating pllm; new CPU tests run directly in pllm.

- Queue policy updated by user on 2026-09-30: architecture validation uses queue012; the c20250601 profile was canceled. Current task outcomes are recorded in PROJECT_PROCESS.md. A0–A4 share the v3 runner, with A0 retaining the legacy large vocabulary/layout. No legacy checkpoint is loaded as v3.
- Resuming v3 requires the exact dataset/certificate hashes, seed, architecture, sampling policy, and cosine horizon in checkpoint `run_provenance.json`. Source-macro validation and residue/end losses are reported separately.
- Pilot YAML generator: `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/prepare_immune_v3_pilots.py`; requires a complete unchanged GPU profile and passing data certificate. `--pilot-stop-step 100 --max-steps 10000` is the current first diagnostic stage and preserves the intended schedule. Continue with the same horizon and `--pilot-stop-step 0`.

- Formal worker entry must receive `--profile-dir`; a passing unchanged profile is rechecked on the worker, including batch size, precision and exact model configuration. Source fingerprints are retained in run/checkpoint provenance. Never bypass the data/profile gates by calling the training entry directly.

- Current candidate training directory is `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean_r2`, still awaiting export/rescan and H3 coverage adjudication. The sibling `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_clean` is an aborted diagnostic export with over-broad containment; never train from it or cite its deletion statistics.

- 2026-09-30 用户资源偏好（本轮no-ESMC架构验证）：先闲时单卡或本地CPU小规模验证，再依据结果扩大。排队不是切换非闲时资源的授权。pilot生成器默认100步、闲时、最长7200秒；仍保留global batch256和10000步scheduler horizon。CPU toy入口 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/smoke_immune_v3_cpu.py` 限制最多500步且拒绝覆盖已有输出目录。

## 2026-09-30 用户明确：完整实验记录同回合同步Markdown

- 所有实验的启动、配置变化、进度复核、完成、失败、取消、复跑和结果更正，都在操作/获得证据的同一回合写入对应文档；原始日志/JSON作为证据链接，不能代替Markdown中的结论和状态。
- [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)：实验目的/变体、数据与checkpoint版本或哈希、seed、batch/累积、lr/步数/采样参数、设备/闲时属性/资源上限、命令或配置路径、task ID或PID、UTC核对时间、日志/结果路径、指标、失败原因、限制和下一步。未查询的状态不得写成最新；待执行与已完成分开。
- [BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md)：假设、架构/目标函数/对照/验收标准及决策变化；[DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md)：数据格式、锚点、去重/划分规则、覆盖缺口、清理数量与证书状态。规则、资源偏好、路径约定同步本指南。
- 下游实验再同步对应任务手册和benchmark结果账本；诊断、论文引用、部分运行和正式复现分开。后续新证据推翻旧结论时，明确标记旧结论失效，保留原因与来源，不能悄悄覆盖。

- 当前r2独立复核日志为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/immune_v7_20260930/verify_r2_optimized.log`；每个完成分片记录累计复核行数和禁入命中行数。manifest/清理报告已存在仍不等于通过；只以最终hash绑定证书判断能否进入真实数据pilot。


## 2026-09-30 下游范围变化时重新验证预训练暴露

新任务加入必须登记test/valid及所有few-shot支持/查询候选池，含background；不要按train文件名自动排除。每个CDR3字段声明core/junction来源语义，保持模型输入真实锚点。预训练侧审计不改官方下游划分。当前入口/产物/证书契约统一见 [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) 的“新增TCR下游暴露审计”；训练和pilot生成均经`assert_ready`要求新scope证书，resume记录其SHA。历史checkpoint另做暴露披露，不能以未来clean数据为其背书。


2026-09-30审计完成产物：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/data/dedup/tcr_exposure_20260930_r2/summary.json`、每数据版本`report.json`/`row_hits.jsonl`及`cleanup_plan.json`。命中清单的行号为shard内零基非空记录号，cohort位顺序在报告中，必须连同源manifest/SHA使用；跨任务计数不能相加。清单是候选清理输入，不能当通过证书。证据与剩余覆盖限制见 [完整审计记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md#tcr-exposure-20260930)。


## 2026-09-30 架构验证队列策略（当前用户决定）

后续no-ESMC架构改动、显存profile及小规模pilot统一使用queue012（q-20260524172355-rnqtf），不再往c20250601提交排队副本。pilot生成器已切换queue012模板；原c20250601 YAML保留作历史记录。单卡小规模验证和数据/profile检查继续执行，实际运行记录以 [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) 为准。


## queue012 手动提交工作流（2026-09-30）

用户确认本机没有 queue012 排队权限。架构任务由本机准备多个 YAML 和本地验证，用户在有权限机器统一提交；禁止因无权限回退 c20250601 或重复尝试提交。具体包见 [queue012 实验包说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_explore_queue012/README.md)。

生成器：[prepare_immune_v3_exploration.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/prepare_immune_v3_exploration.py)；诊断入口：[explore_immune_v3_synthetic.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/explore_immune_v3_synthetic.py)。第一组仅人工数据，**已全部完成/Success**，不要再提交同一组；第二组是真实数据草稿，**从未提交**，不能混入统一提交列表。生成草稿不表示数据已通过审计，运行时旧/新数据门禁和 profile 指纹仍强制执行。最终数据目录变化后用 --data-dir 重新生成配置。生产代码变化后重新测显存，再用 --profile-dir 指定新证据。

结果根目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v3_explore_queue012`；各 phase/ablation/UTC 时间独立目录。提交日志保存 task ID 后，更新 PROJECT_PROCESS 的 Active 表及 dated log。合成20 数字以 [explore README 实测表](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_v3_explore_queue012/README.md) 为准：A0–A4 **全部完成/Success**（A0 `14.093841 → 0.414990`，A1 `5.837228 → 0.328653`，A2 `5.164647 → 0.252067`，A3 `5.169685 → 0.185864`，A4 `4.774529 → 1.119085`）。显存 profile `t-20261001020115-zvt6f` **Success**。`02_real_data_gated` **从未提交**。


## no-ESMC research suite（2026-09-30）

长期研究记录：[研究与方法记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FOUNDATION_ARCHITECTURE_RESEARCH.md)。独立模型模块：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/dllm/pipelines/immune_llada/research`；运行脚本：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/run_immune_research.py`；生成器：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/prepare_immune_research_jobs.py`。新suite入口：[24 个 YAML / 提交说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)，本机只准备/验证，用户有权限机器提交queue012。

每个新方法必须重新profile，先测2048 token长输入再选microbatch与global256对应accum；register方法额外4个内部token。profile源SHA覆盖模型、layout、data/sampler、trainer、scope gate等，代码变动后用新的--profile-root，不覆盖旧成功证据。同方法profile有文件锁，OOM候选用独立进程。合成组上限7200秒；真实200步探索pilot上限21600秒，仅对本research suite适用。

真实数据默认最新immune_v7_mhc_sequence_v1候选，但必须最终clean旧/新证书通过，门禁在模型构建前执行。不同variant统一最终数据版本。研究checkpoint额外携带research_config.json，并用ResearchModel.from_pretrained加载；不能用普通v3加载器忽略新增参数。结果根目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_queue012`，按phase/method/UTC隔离。没有实际task ID时不填Active。


## research suite当前预算与恢复规则（2026-09-30更新）

当前配置以[36个YAML与续训说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_queue012/README.md)为准，共36个文件；此前本节24文件、真实200步与21600秒上限为历史方案。01组保持显存profile后合成20步/7200秒；02组真实1000步/86400秒；03组可选累计5000步/259200秒，建议仅R00和2–3个候选。新组统一warmup100/horizon10000/global256，每100步保存和固定验证。

`run_immune_research.py --resume-from-checkpoint`只接受模型、optimizer、scheduler、RNG、trainer_state、research_config与provenance齐全且匹配的checkpoint。`prepare_immune_research_jobs.py --resume-map <JSON>`可给选中的03组方法注入绝对checkpoint路径；未提供映射的03组仍是scratch，不自动寻找最新目录。重启目标是累计step，不是额外step。自然采样按epoch固定种子，验证mask按记录固定且不扰动训练RNG。源代码变动会让研究profile指纹失效；保留旧证据并在新profile-root重测。

已完成首版A0–A4显存结果仍是资源参考，不改成失败、训练成绩或新方法实测结果。所有新任务仍由用户有权限机器提交，不回退c20250601。


## 2026-09-30 八卡资源与测试产物规则

八卡工作池已提交一次，禁止重复提交同一共享队列。每卡实验与维护负载互斥，维护负载标记maintenance_only；本地监听只读更新MD，不保证聊天界面在线。最大72h，STOP排空、STOP加STOP_NOW中断、DISABLE_KEEPALIVE仅停维护负载。测试无权重；真实完成后保留最后2份、best和已登记续训checkpoint，成功结果与数据证据保留。

运行与控制说明：[八卡工作池](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_research_pool_c20250601/README.md)；实际结果：[持续记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_POOL_LIVE.md)。


## 2026-09-30T19:40:21.115588+00:00 R00–R07当前显存测量

同一8×A100-SXM4-80GB实例中，每方法独立单卡、2048-token profile已通过。R00–R07推荐micro-bs均32、accum8、每实验global256；峰值分别42.00/42.00/42.00/42.00/42.75/42.15/43.31/42.18GiB，下一档64均OOM。该结论只对profile绑定的源码、精度、设备有效；R08–R11尚未测完，不提前沿用。证据见 [profiles](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/profiles)。


## 2026-09-30T20:19:38.413413+00:00 持久池版本切换与第二轮显存

原R00–R11代码已保存不可变快照，经旧入口路由保持原profile有效；R12–R19用新runner与独立profile，每实验global256，当前micro16×accum16。源码只在无运行job时发布，release_map让监督对应版本。每job_id记录多阶段结果；synthetic不留checkpoint，测试临时ckpt清理，真实训练保留续训依赖。

详细研究、测试和运行证据：[架构扩展v2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_EXPLORATION_V2.md)。


## 2026-09-30T20:24:51.052023+00:00 v2入队与代码边界

`enqueue_immune_architecture.py`对真实阶段先验证数据证书，再验证完整100步、source/profile SHA与四档学习改善；持锁去重。初始pool仅入队旧12项，v2配方独立。bias_cache原型位于非运行指纹目录，不要热绑定正在运行GPU实验；下次发布需新profile并保留旧release/source对应关系。63项晋级、11项pool/monitor回归通过。

20:29Z监督规则补充：按release_map对应目录同时核验文件SHA和runner收集的源码文件集合，新增被收集文件也属于版本变化；对应9项监督测试通过。后台daemon持续写MD，监督agent在本轮负责解释与隔离排错；后台监听不会自动改模型、重提平台任务或解除数据门禁。原型使用与下一版条件见[bias缓存验证](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_BIAS_CACHE_VALIDATION.md)。


## 2026-09-30T20:53:52.568809+00:00 持续研究版本与参数口径

LLaDA270名称不等于270M Transformer；报告拆开blocks、词表embedding/head与附加参数。R00–R11、R12–R19分别使用冻结源码目录；release_map先于公共入口变更。新R20–R27独立profile，不能沿用旧测量。人工500步将使用明确的synthetic fixture manifest和kind标签，训练模式不自动代表真实语料训练，真实数据仍须双证书。

[持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md) · [270基线审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_LLADA270_BASELINE_AUDIT.md) · [100步结果](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_R12_R19_100STEP_RESULTS.md) · [数据准备状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TRAINING_DATA_READINESS.md)


## 2026-09-30T21:17:56.103865+00:00 Refined实测batch与阶段含义

R20–R27最新39文件profile均passed，单实验micro32×accum8=256，推荐档峰值41.995–53.968GiB、64档OOM。谱系不同不可沿用旧profile；旧R12–R19仍对应micro16及冻结源码。500步人工诊断使用mode=train配合可复算synthetic_fixture_manifest，kind及MD明确标人工；真实prepared仍须双证书。入队助手先验证整批、SHA/完整trace/heldout分层和来源，锁内去重；dry-run不创建锁。

[Refined配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md)


## 2026-09-30T21:25:58.529062+00:00 八项500步人工诊断已开始

R20–R27的25步广输入检查全部succeeded，累计28成功/0失败。各项约718秒，四档mean loss为1.768–1.908、AA约30.06%–36.38%；end accuracy仍0，全mask end只占132/17996≈0.73%，须独立观察，不能仅看总loss。局部/全局方法在周期8人工任务上的优势不等于生物学优势。

已实际执行经74项测试验收的入队助手 synthetic500 --apply，8项均被worker领取运行。新500步initial的18组指标与同方法25步initial完全一致，fixture、39源码SHA、global256也一致，确认为同协议重新初始化。500步是更广人工heldout诊断，不是初步真实语料训练。按25步实耗粗估约4小时，准确耗时以运行结果为准。

[启动证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/synthetic500_started.json) · [配置与协议](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_refined/README.md) · [持续目标](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)。目标保持active，真实数据清理与1000/5000步尚未完成。

数据覆盖仍在推进：替代编号工具给出了VNAR候选，但与原ANARCI对部分“源空H3”案例存在分歧；184项只能表示来源/原编号已有解释，不能自动当成184项生物学缺失已确认或认证覆盖已完成。保留pending并量化对未知loop使用训练已知H3匹配整条benchmark观测串的保守附加筛查。新expanded exporter已开始实现与CPU测试，尚未扫描/导出全量或签证。

## 2026-10-01 typed诊断、监督上线与pilot生命周期验收

后续固定operator诊断使用kind=cuda_attention_backend_probe，旧运输steps=1仅是一次调用；监督显示cases，绝不伪造global_step或推荐模型microbatch。执行完成与全部数值通过独立报告；unsupported/OOM/numerical mismatch为可完整记录的负结果，backend mismatch/超时/损坏/异常退出不能晋级。实际GPU仍需当前500终态、冻结release、动态registry/早期入口路由和禁止init自动入队后才执行。[IMMUNE_TYPED_DIAGNOSTIC_ADAPTER](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TYPED_DIAGNOSTIC_ADAPTER.md)定义预算及证据合同。

本地只读monitor已于2026-10-01T00:06:15Z替换为PID2033525，平台/容器/训练均未重启；不能误将监听维护当重新申请八卡。operator子进程退出需正常returncode与清理完成证据，报告字段之外还绑定request/plan/result/log字节。[IMMUNE_TYPED_DIAGNOSTIC_MONITOR](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TYPED_DIAGNOSTIC_MONITOR.md)记录32+3项双环境验证和部署。

未来真实pilot保留原assert_ready第一门禁，另需500诊断和新GPU release证据；无GPU producer/通过证据时不能以旧profile代替。保存完整optimizer/scheduler/RNG，checkpoint侧车绑定九文件SHA，累计1000→5000只允许停止步改变，不重置10000步日程。tiny恢复完成不等于连续运行数值等价。测试checkpoint已清理，保留小型日志和SHA证据。

## 2026-10-01 00:47Z 归档和入口切换顺序

后续变更顺序固定为当前八项500步终态及原证据核验、完整依赖归档、release_map重绑定、早期路由补丁、实际GPU诊断。[归档helper](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_REFINED_RELEASE_ARCHIVE.md)默认只读，任何pending/running或缺终态都会拒绝apply；复制包含必要Python/JSON支持树，不能仅复制39份profile清单。当前计划520文件只是预检结果，正式归档须重新规划。

[待部署补丁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_DIAGNOSTIC_DISPATCH_STAGING.md)将28个模型variant定向到对应冻结快照；在任何ML导入前exec新进程，禁止混用live和冻结模块。唯一D00只接收固定4-case诊断，不接受data/resume/任意执行目标，也不参与pool init自动入队。当前内存worker不会因磁盘补丁自动更新函数，入口和adapter必须继续独立限制参数。两环境各22/38项CPU通过不等于部署或GPU通过，当前八卡训练未重启。

新pilot的源码绑定与GPU release仍是独立未完成项；不能用改写旧profile、关闭门禁或仅修改规划YAML的execution_enabled来代替。当前规划与待部署产物见[下一版本README](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/README.md)。

## 2026-10-01 完整窗口归一化与历史源码保留

新TokenValidationTrainer明确model_accepts_loss_kwargs=False，因其训练compute_loss返回microbatch均值、不使用num_items_in_batch。当前factory不传compute_loss_func；独立实际Trainer/SGD窗口验证已通过，不能只用compute_loss.backward测试来认证GA语义。两版HF尾窗除数不同，完整窗口CPU通过不扩张为尾窗或GPU/AdamW保证。[详细证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

改变未部署原型前先保存旧精确字节和报告SHA：本次见[旧版本保存清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_hf_micro_mean_audit_20261001/before_fix_manifest.json)。新版本报告保留修正后实际梯度/参数差、父类与Accelerator来源。当前39冻结源不编辑，GPU500不重启；后续完整归档会收录最终稳定的原型版本，派生runtime以父manifest为准，不硬编码已经过时的原型SHA。测试临时文件及时清理，保留小型证据和必要源码历史。

## 2026-10-01 已归档的R20–R27来源

[固定父快照](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/release_manifest.json)已于01:29:56Z发布，524文件全量copied SHA、39份历史测量SHA、8份500步终态分别记录。原子map切换的前后字节及SHA见[证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/deployment/archive_and_map.json)。旧快照和旧profile不改写。

仅设置PYTHONPATH/cwd不足以保证worker内项目来源隔离：.pth可加入live子目录，同名namespace还可能合并site-packages路径。后续入口必须验证自身及实际Python子进程的来源；这是已发现的部署条件缺口，目前没有证据表明历史模型模块来自错误路径。修复前不宣称路由已部署或完整隔离。

## 2026-10-01 训练主线与可选性能测试

[规划YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_architecture_next_release/protocol.yml)仍execution_enabled=false且没有消费该schema的执行器，不可直接提交。stages只列Trainer GPU验收、真实1000与累计5000；optional_attention_optimization存放用户本轮取消的CUDA专项，不是训练前置条件。下一次实际入队需完成并核验新入口，不能只改execution_enabled启动。

本地读写已恢复；monitor实际进程与fresh heartbeat已重新查验，不从旧状态假定已中断或重启。当前无新增平台操作。实验checkpoint保留继续按latest/实际best/恢复依赖规则，测试checkpoint及时清理。


## 2026-10-01T07:43:39.278049+00:00 三候选合成验收的执行和观察

R21、R24、R26已各新增一次合成Trainer验证，交给现有8卡容器执行。每项3个正式优化窗口（24个microsteps/768条人工记录），另含完整梯度核对与四档mask评估；等待worker领取，尚未报告通过。没有新增平台任务或重启容器、worker、监听。真实语料training_ready=false且双证书缺失，最终训练未启动。 [三项实际启动记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/candidate_gpu_launch.json)、[CPU检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_candidate_transport_cpu_20261001/resume_cpu_review.json)、[执行文件更新记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_candidate_transport_cpu_20261001/support_publication.json)。现有监听PID2213164不重启，已加载的旧展示代码不支持新三候选；本轮直接读取三个明确job目录与新的只读报告模块核对状态，不能把旧展示的未知候选提示当作GPU失败。旧T20私有实现绑定先前执行文件哈希，后续必须显式复核再接入，不直接套用旧绑定。


## 2026-10-01T07:50:48.006294+00:00 agent接续记录规则

接手持续实验先读[当前目标CURRENT交接块](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md)和[总账顶部及Active/Live Progress](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md)，再使用交接块引用的当时实际原件并读取明确job最新进度/结果。旧 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/EXECUTION_STATE.json 只记录10月1日三候选，不能作为当前final-parent、p01/p02或Goal状态。每个关键节点同回合同步匹配项目md及CURRENT引用；保留失败/修正，区分计划、已启动、已完成、已核验。不凭旧段落/mtime/COMPLETE存在性宣布成功，不重复提交，不新建原Goal。


## 2026-10-01T08:10:49.754641+00:00 阶段记录更新工具

[update_execution_checkpoint.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_candidate_runtime_integration_20261001/update_execution_checkpoint.py)仅观察三个已存在的job，更新结构化快照、当前目标、总账Active/Live Progress和候选MD；不会启动作业，也不代替终态强核验。激活pllm后以--note填写已核实阶段事件。新CV配置是规划协议，不可作为平台资源YAML提交，状态见[配置说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_cv_joint_exploration_20261001/README.md)。

2026-10-03T01:10:58Z 接续限定：上述工具属于旧三候选观察流程，当前不调用它覆盖final-parent/p01/p02交接。当前状态原件按UTC日期留存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/project_status_sync_20261003；平台name过滤空不当作终态，直接get同一task核实，不因此重启或重交。原生恢复观测只由未来实际resume产生八rank原件；专属路径 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_RESUME_OBSERVATION_20261003.md，CPU通过、新源准备、真实恢复通过分别记录。

## 2026-10-02T03:11:16.880451+00:00 单项失败GPU观测源码冻结

新driver61ab69fc、dispatcher8f9d4f8d、12项控制60601742已冻结，root/A/B/C完成源码与实际API静态对齐；18原保护来源未改。下一步仅两Python环境各12项新增纯控制/180秒，不重跑旧model/数值验收；当前控制未执行、未部署、GPU新任务0、真实训练0。原tiny/BF16/seed43/断言和全部原准入保持。

[实际冻结原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_before_execution.json)。


## 2026-10-02T03:13:11.684065+00:00 新控制各12通过与单项数值入口发布

pllm与worker Python新增纯控制各12/12首轮通过（约2秒），无模型/CUDA/旧数值检查重跑，OWN/_cpu_tmp已由fixture清理、checkpoint0。已实际发布冻结新dispatcher及两个既定live最小diff：原37registry项保持，追加一个launcher_only数值观测项；16原受保护源保持，2项有记录改变。旧d02/guard/worker/release_map/runtime/helper不改；原8worker/监听/容器未重启、Volc未提交/取消。当前尚未入队、新GPU0、真实训练0。

[实际CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_results.json)；[实际发布及差异](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication.json)。


## 2026-10-02T03:19:25.984054+00:00 单项观测启动失败及最小临时目录修正计划

唯一D03/GPU2已终态failed/rc2，child10.14秒正常退出并实际进程清理。status=not_started、tiny未执行、梯度/loss/RNG捕获0，无COMPLETE，不把它算作原模型数值失败复现。原源码隔离检查正确拒绝了Torch追加的PROJECT内临时库搜索路径；仅新观测driver临时布局有误。自有tmp及库已清理，checkpoint0，池0/0/51、8worker维护/监听保持，真实训练0。

按新证据准备独立v2修正：专属库tmp放在项目树之外的固定conda/cache/tmp子树，保留Torch原sys.path行为和原strict verifier。不热改v1、运行包或门禁；仅新增v2源码及两个受影响stdlib控制/每环境180秒，旧12项和已接受模型检查不重跑。v2另设一次新GPU/总180秒预算，现阶段源码准备、尚未执行/发布/入队；原tiny120秒/seed43/BF16与所有原数据/学习/恢复门禁保持。

[失败原件复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/failed_observation_review.json)；[v2明确范围与预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v2.json)；[v1及原源保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v2.json)。


## 2026-10-02T03:25:29.401687+00:00 v2目录修正控制通过并实际启动观测

新增目录/源码绑定控制两环境各2/2首轮通过（约3秒）；原12未重跑、无模型/旧数值验收。已发布最小v2，只改runner的dispatcher literal并新增冻结dispatcher副本，21保护旧源/v1/registry/release_map/runtime/verifier保持。原池仅入队一次新job `1790911529399018156_d03_r21_seed43_tiny_gpu_observation_v1_6acb659f`，observer_version=2、D03_v1只作固定transport身份，source/report精确绑定v2。原v1启动失败原件保留，不追认模型通过；本次尚待actual数值/正常退出。池{'pending': 1, 'running': 0, 'results': 51}，不重启/Volc提交，真实训练0。

[CPU控制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v2_results.json)；[实际最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v2.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v2.json)。


## 2026-10-02T03:32:00.954350+00:00 v2首批IPC失败与最小短路径修正准备

v2/GPU2终态failed/rc2：源码隔离已通过并记录正确TA/设备，但TMPDIR157字节导致resource_sharer的AF_UNIX socket path too long，等待首批到原120秒超时。gradient_call_count0/replaynull/数值捕获0，不能算原梯度失败复现；child正常退出并进程清理，wrapper约135秒<180，库tmp已清理，checkpoint0。全池0/0/52、8worker维护，真实训练0。

仅准备新v3短tmp base `/tmp/immobs3_251105016` + `j`及actual job_id SHA前16位叶，避免socket地址上限；先两环境各2项受影响stdlib检查，其中一个实际resource_sharer.DupFd/detach描述符通信，不导入Torch或运行模型。旧12+2/已接受数值不重跑，v1/v2源与原runtime/verifier不改，v3尚未执行/发布/入队。原窗口/seed/BF16/数值阈值和数据/500学习/恢复门禁保持，不重启/Volc重复提交，不恢复注释或独立CUDA算子。

[实际IPC失败](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/failed_observation_v2_review.json)；[v3明确预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v3.json)。


## 2026-10-02T03:38:23.692209+00:00 v3实际IPC控制通过并单项GPU观测启动

两环境各2/2新增控制首轮通过（约3秒），包括实际resource_sharer.DupFd/detach socket/FD传递及本case线程/socket/文件/环境完整清理；旧12+2/数值验收不重跑。只发布新冻结dispatcher和runner literal，25保护旧源/v1/v2/registry/release_map/原verifier/IPC不改。原池一次入队job `1790912303689399341_d03_r21_seed43_tiny_gpu_observation_v1_324e2b1c`，observer_version3，tmp短映射实际/tmp/immobs3_251105016/j2035c7b732f32f84；原120秒tiny/总180秒与全部门禁保持。实际池{'pending': 1, 'running': 0, 'results': 52}，尚未记数值观察完成或模型接受，真实训练0。

[实际CPU IPC检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v3_results.json)；[最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v3.json)；[真实启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v3.json)。


## 2026-10-02T03:44:16.057634+00:00 原GPU梯度差异已复现并取得完整数值，正式训练仍未启动

v3单项观测在GPU3于03:38:44 UTC结束，root使用同一worker Python只读绑定report/COMPLETE/log SHA与实际exit/清理全部核验。COMPLETE仅observed、完整采样true；原tiny约2.01秒仍报wte.weight梯度不一致，model_test_passed/GPU/data/training接受全部false。当前source/seed原工程验收仍7/8，七项不重复、不撤销原失败。全池0/0/53、8worker维护、原监听2213164存活；原Goal身份/active/累计计量保持。

26梯度张量均有限、成员一致；17/26张量、42/7156元素超原逐元素阈值。wte4/800失败，maxabs0.000306189；全局scale0.9999226166、relative L2=0.0003664514（0.036645%），通过整体数值阈值但不能替代逐元素失败。8对RNG、mask、selected labels及前向logits完全一致，GPUloss最大差4.768e-7，观测重算满足loss标准（原loss断言在梯度报错后未执行）。16组CPU/FP32叶子比较最大误差1.49e-8、无失败；这是CPU观察，不能证明CUDA kernel根因。下一必要工作限通用synthetic原模型反向路径定位，先区分CE导数、混合精度舍入和checkpoint反传；不放宽阈值、不执行独立算子验证。

本轮每环境12+2+2=16个新增定向控制首轮通过，旧已接受数值/模型检查未重跑。v1隔离失败、v2 IPC失败均保留，v3短路径通过actual CPU Socket/FD IPC后才获取GPU原断言证据。root首次只读复审错用pllm导致sys.executable绑定拒绝，原件保留；仅用actual worker Python修正receipt核对，未改validator/生产源或重复GPU。25保护原源保持（runner既定literal变更有记录）、v1/v2源码/raw及实际训练恢复状态保留；GPUchild/库tmp清理完成，另外两个自有空tmp父目录删除，checkpoint创建/删除0。

03:44:16 UTC直接读取V7 MHC metadata，manifest仍e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，audit_pending、ready字段缺失、当前audit_certificate/downstream_scope_certificate缺失。既有V7清理保留，不称已证实泄漏。数据版本是V7，条件启动授权已有，但当前审核/ready、原500学习/完整恢复/data-selection等门禁仍未闭环；真实shard/小batch/语料训练0，不能直接启动八卡正式训练。停止注释不恢复、独立CUDA暂停；没有Volc提交/取消或容器/worker/监听重启。

[本次完整数值复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v3_terminal_numerical_review.json)；[资源、metadata及清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_and_numerical_evidence.json)；[原始观测](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790912303689399341_d03_r21_seed43_tiny_gpu_observation_v1_324e2b1c/numerical_evidence/gpu_diagnosis.json)；[原Goal实际工具值](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/goal_after_numerical_actual.json)；[首次只读核对错误](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v3_terminal_first_read_failure.json)。


## 2026-10-02T03:52:02.174489+00:00 原反向中间梯度观察准备

本Goal上一回合属progress：原GPU梯度差异与完整数值已核实。当前单次只读核实pool0/0/53、8worker新鲜心跳维护、监听2213164存活；指定V7 MHC ready字段及两审核仍缺，真实训练0。已制定仅v4观察预算：原tiny/native8+manual8不变，只在原observed_denoise返回的共同BF16 logits注册return None的梯度读取，最多16原回调，无额外GPU forward/backward/算子probe；实际CPU梯度原件保留。原七项、TA、模型、独立oracle、阈值及所有训练门禁不变，原源冻结。先独立只读评审与每环境最多4项新hook控制，通过后原池至多一次单GPU原tiny。此刻v4未发布/入队，无平台提交、重启或真实shard读取，停止注释与独立CUDA暂停保持。

[明确预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v4.json)；[保护旧源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v4.json)。

## 2026-10-02T04:00:16.741571+00:00 v4只读评审发现与修正

两独立评审确认原GA口径一致，原native Trainer loss/8、Accelerator GA1，manual loss/8；最小观察点为原exercise唯一observed_denoise返回共同BF16 logits，16个return None hook只复制实际原梯度到CPU。原输入/模型/TA/独立CE/阈值不改，无额外GPU计算，CPU复制可能影响同步。A指出torch.equal不是严格位相等，driver已在未执行稿改称exact_value_equal并显式BF16要求；还指出新报告job_id必须与实际child输出目录绑定，B正在仅补该元数据校验，交付旧字节留证。四个新控制含该跨job拒绝，不增加case。此刻未root冻结、控制未执行、GPU入队0；原7/8及失败保持。

[评审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_observation_design_review.json)。

## 2026-10-02T04:05:28.845861+00:00 v4原反向观察源码冻结

Root actualdiff/AST及独立评审通过；driver59961065、dispatcherbf843792、control585d705e、budgetec07314e固定。31旧路径不变，模型/TA/原断言阈值不改。报告精确绑定child job；共同BF16读取字段称数值相等。即将仅每环境4项新增hook/trace/metadata/CPU leaf控制，旧验收不重复；GPU尚未入队、真实训练0。

[源冻结原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_v4_before_execution.json)。

## 2026-10-02T04:07:14.414668+00:00 v4四项控制首轮通过并单项GPU观察启动

两环境各4/4新增控制通过（worker6.43秒、pllm6.63秒），真实CPU leaf hook return None/不改梯度/remove后无旧回调、trace/caller与跨job报告拒绝全部通过。Root首次pllm命令错指不存在解释器，child未创建/测试0；仅修正到实际activated sys.executable后首次4项通过，错误元数据保留、零字节自有log删除，源码未改。旧七项及旧验收不重跑。只发布新dispatcher和runner literal，30保护旧路径及31中其余项保持。原池一次入队`1790914034411779826_d03_r21_seed43_tiny_gpu_observation_v1_e950d2a0`，原native8+manual8/120秒、总180秒、16原hook读出；尚待实际数值/退出与清理，观察完成不代表模型准入。真实训练0，数据/500学习/恢复门禁保持；无Volc提交或重启。

[CPU证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v4_results.json)；[最小发布](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/publication_v4.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v4.json)。

## 2026-10-02T04:18:00.540716+00:00 原BF16反向入口已定位一处差异，完整原件复审完成

v4 job `1790914034411779826_d03_r21_seed43_tiny_gpu_observation_v1_e950d2a0` 于04:07:37 UTC在GPU3正常结束，root用actual worker Python只读verify_sources/report/child/COMPLETE/raw绑定与独立stdlib逐值重算通过。COMPLETE仅observed、原tiny仍同wte.weight ValueError，原source/seedGPU工程仍7/8、R21s43未准入；没有撤销原失败或改标准。原反向8native+8manual、16hook全部移除，parent19.785秒、child14.670秒、原tiny2.100秒，实际进程组/库tmp/exclusive子目录清理完成。checkpoint创建/删除0，自有零字节错误launcher log已删，源码/失败原件/原数值及实际训练恢复状态保留。

8对原共同BF16 [1,64,50]梯度中7对完全数值相同；第7步只flat2654=[0,53,4]一元素不同：native0.0003871917724609375，manual0.00038909912109375，差1.9073486328125e-6（相邻BF16值）。该边界仍满足原allclose，relativeL2=4.813081e-5；但参数17/26张量、42元素超原标准，全部参数统计与v3精确相同。8对RNG/mask/labels/forward仍同，GPUloss maxdiff4.768e-7。证据证明至少一处差异已进入模型反传边界的loss/layout/cast一侧，不证明该1元素解释全部42、不认具体CUDA kernel根因；16CPU leaf仍只CPU，未执行独立算子。

A/B只读下一决策：暂不改shared native CE，下一必要点先审阅/有限观测原已有FP32 loss输入的实际gradient，CPU统一布局、return None、无额外model/CE/backward，不改变原GA8及阈值。若以后采用数学等价row-major CE，须新的私有源码/CPU等价/原tiny与必要完整GPU证据，独立oracle保持，旧七项不迁移到新SHA。此刻没有v5源码/预算/新job。

最新04:12:17快照pool0/0/54、8worker维护且心跳<5秒、监听2213164存活，无容器/worker/监听重启或Volcsubmit/cancel。V7 MHC metadata仍e12e55cc…/audit_pending、training_ready字段缺失、两当前版本/下游报告缺失，不等于已证实泄漏。真实shard/小batch/训练0，原数据审核、500学习、完整恢复/data-selection等门禁全部保持；停止注释不恢复，独立CUDA暂停。原Goal身份/active/actual累计用量保持，本回合属progress，不计全部外部阻断；无子agent继续执行，不以新增无关测试填充Goal。

[原反向独立复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_terminal_backward_review.json)；[实时状态/metadata/清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_after_v4_backward.json)；[下一必要工程决策](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v4_next_engineering_decision.json)；[Goal实际工具累计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/goal_after_v4_actual.json)。

2026-10-02T04:18:40.733707+00:00：状态JSON原v1的source/控制/失败字段移到legacy_v1_observation_fields；当前字段明确指v4来源与actual observed=true/model_test_passed=false，避免历史与最新混用。仅状态/MD同步，无新模型或测试执行。

## 2026-10-02T04:21:40.077216+00:00 v5原FP32反向观察预算

上一Goal回合是progress，已取得原BF16入口1相邻值差异及raw复审。当前一次只读snapshot04:20:13，原池0/0/54，8worker新鲜心跳维护、监听2213164存活；V7 MHC ready字段/两审核仍缺。仅预算一次必要原tiny读出：保留原16 BF16 hooks，新加16原FP32 CE-input/manual-logits None hooks，原8+8 backward/GA8/120秒、总180秒不改，32handles finally移除；CPU转置统一BLV、CPU cast对照同轮真实BF16，raw保留。无额外GPU CE/forward/backward或独立算子，原模型/TA/assert/门禁不改。4项新增定向CPU控制/env，旧验收不repeat，来源评审和控制通过才允许原池一次入队。目前未发布/执行，无真实shards/训练、无重启/平台提交，停止注释与独立CUDA暂停。取得该边界证据后先做最小通用工程决策，不无止境增加无关观察。

[预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/execution_budget_v5.json)；[保护旧源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/protected_before_v5.json)。


## 2026-10-02T04:29:13.960945+00:00 v5通用工程来源冻结

v5原FP32观察源码root冻结：A416e6ec1/Bcad2625c/Cdcf8bbec，37旧源保持，AST差异限新FP32观察/元数据校验；4新增CPU控制/环境即将执行，当前未发布/入队。原7/8、全部数据门禁/停止注释/独立CUDA暂停保持，真实训练0。

[复审原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/source_review_v5_before_execution.json)。


## 2026-10-02T04:31:34.602707+00:00 v5控制通过与原池唯一观察启动

两环境各4/4新控制通过，新冻结来源已发布；原八卡池唯一v5合成观察 1790915494600058674_d03_r21_seed43_tiny_gpu_observation_v1_5d224d8a 已入队，保留原8+8/GA8/阈值、32 None-hooks，转置/cast仅CPU。待实际终态/raw复核，原7/8不变，全部数据/安全停线保持、真实训练0；未重启/平台提交。首次root读取worker错用heartbeat字段，实际字段time，发生在发布前，未运行child/重复checks/改validator，失败原件保留。

[CPU原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/cpu_control_v5_results.json)；[实际启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/launch_v5.json)。


## 2026-10-02T04:37:11.813335+00:00 v5实际FP32证据完成与训练就绪状态

v5唯一观察 1790915494600058674_d03_r21_seed43_tiny_gpu_observation_v1_5d224d8a 于04:31:58 UTC在GPU1完成，root actual worker Python仅stdlib只读receipt/COMPLETE/源SHA/raw逐值重算通过。新增控制两环境各4/4、原32 None-hooks全部移除，子进程/库tmp/独占临时目录清理完成，checkpoint创建/删除0；源/raw/失败原件及实际训练恢复状态保留。FP32共287值差（maxabs1.86265e-9），独立IEEE round-nearest-even重算16组cast全部匹配同轮实际BF16；第7步[0,53,4]的FP32值距midpoint -2.91e-11/+8.73e-11，因此产生1个相邻BF16值差。参数统计与v4完全一致，仍17/26张量、42元素超原阈值；原wte断言仍失败、当前工程7/8，COMPLETE仅observed而非模型通过。此证据不证明全部42差的完整因果或某个CUDA kernel。

下一决策：停止加hooks/观察波次，准备一个私有数学等价的native CE行优先布局候选，保持独立oracle、GA/BF16/阈值及原接口，先针对性CPU等价/原tiny，再按新来源补必要完整证据；此刻候选未实现、未验收，旧七项证据不迁移。原当前来源学习/完整checkpoint恢复/data-selection门禁不撤销。V7 MHC manifest仍e12e55cc…ba94、audit_pending、training_ready字段及两份当前版本/下游审核缺失，既有清理保留，不称数据已证实泄漏；真实shard/小batch/训练0。原八卡池0/0/55、8worker维护且新鲜心跳、监听2213164存活，无重启/Volc提交或取消。停止序列注释不恢复、独立CUDA暂停。

本轮root两次只读记录字段误读（worker实际time、环境symlink路径需resolve）已保留并修正核对，不改生产源/validator或重复模型检查；最终强复核无ML导入。

[本轮原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v5_terminal_FP32_review.json)；[资源与数据状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/final_state_after_v5_FP32.json)；[下一具体决策](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_seed43_gpu_observation_20261002/v5_next_engineering_decision.json)。


## 2026-10-02T04:39:55.699962+00:00 私有通用损失布局修正准备

上一回合progress：v5原FP32/cast实证及root raw复核完成，本轮只推进私有数学等价native CE行优先布局修正。原八卡池/来源/阈值及独立oracle保持，计划4必要合成CPU合同测试/env、tiny最多一次/原120秒8+8，不再加hooks；当前仅预算/源准备，未执行/发布/入队。原7/8、V7 audit_pending/ready与双证书缺失、真实语料0不变。停止注释不恢复、独立CUDA暂停、不得重启/重复平台提交。保护清单首轮路径误写的只读错误保留，按实际文件修正，不改原源。A写私有forward最小改动、B只读原包整合API、C写4CPU合同源；root独占执行与MD。

[范围与预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/execution_budget.json)；[原来源保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/protected_original_sources.json)。


## 2026-10-02T04:44:32.155121+00:00 CE修正必要来源绑定预算

只读原包API确认不接受model override/新root；必须新的固定namespace、原531 parent清单严格验证、534同名清单仅model与必要bootstrap新SHA。原verifier不改；producer算法/oracle不动，只新固定roots/schema。预算追加3必要软件绑定控制/env180秒（另有原4数学合同），tiny-only不伪造profile且全部GPU/训练准入false，完整release入口仍待接入/新profile。当前软件源准备，无构建/GPU/真实数据。

[绑定补充预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_binding_addendum.json)。


## 2026-10-02T04:52:50.263506+00:00 CE新固定运行包来源冻结

root实SHA/AST确认producer全部函数不变、package仅plan/verify/build和固定namespace/schema/singleton override，原531/534清单保持。3新stdlib绑定控制/env待执行；tiny driver完整原schema_version已按事实修正，parent字段按actual报告对齐，均发布前未执行的静态纠错。数学4项不重跑，GPU新case0/无真实数据或训练，原安全停线/门禁保持。

[来源复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_source_review_before_CPU.json)。


## 2026-10-02T04:58:56.129931+00:00 CE修正来源包实际构建

两环境3软件控制的不同body全部通过：前2首轮通过；第三fixture先前未执行，补自有父目录后首次body通过，不repeat已通过项。自有CPU tmp及其内pytest链接安全删除，setup与root过严清理guard错误原件保留，source/门禁未改。新source-only固定R21包实际构建534来源，531 parent完整保留，model唯一parent override、39 measured仅model SHA变化；原10保护源保持。全部准入false、fullrelease entry未接入；待actual manifest绑定父监督和原tiny一次执行，当前GPU新0/真实训练0。

[真实构建](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_build_actual.json)；[控制与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_CPU_controls_and_cleanup.json)。


## 2026-10-02T04:59:42.888071+00:00 CE修正原tiny实际启动

4数学+3软件合同/env通过，新固定source-only R21包534来源已实际构建，父监督/source/budget均冻结。原池一次入队CE修正原tiny 1790917182885479822_d03_r21_seed43_tiny_gpu_observation_v1_6c5bc4a4，原8native+8manual/GA8/seed43/120秒，parent180、无新增hooks。profile=None、原fullprofile/完整release/训练门禁保留，全部准入false；实际原梯度断言结果尚未返回。仅新增冻结dispatcher+runner literal既定改变，原其余9保护源/旧包/失败原件保持，无容器/worker/监听重启、平台重复提交或真实训练。

[启动原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/launch_actual.json)；[最后源/入口复审](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/tiny_entry_review_before_execution.json)。


## 2026-10-02T05:09:28.411744+00:00 CE 修正版原 tiny 通过，完整训练验收仍待闭环

CE 行优先布局修正版已于 2026-10-02 05:00:05 UTC 在原 GPU1 一次通过原 tiny：26 个参数梯度全部通过，梯度倍率 1、相对 L2 误差 0、RNG replay 一致，loss 最大差 4.76837e-7。原阈值、GA8、BF16、seed43 和独立 oracle 均保留，没有新增 hooks 或独立 CUDA 算子测试。tiny 1.780 秒、总监督 20.589 秒，真实退出码 0，子进程、对象及独占临时资源清理完成。

两套环境各 4 个新增模型数学合同及 3 个新增软件来源合同均通过；软件首轮前 2 个通过，第 3 个因自有 tmp 父目录缺失在 fixture setup 失败，补目录后仅该 body 首次执行通过。没有重复已通过项目，错误原件及来源保留。新不可变包 534 个源码文件，model SHA 7e79bfb7d2327c07aa3c39e0bed89997fa627b3fa6e20cb084630b347e886592，manifest SHA 2c2c9ee3ca2f17c29da402b937902d158d82eeedb5ba7753dcf1e0be1307c148。

这只证明新源码的原 tiny 数值检查通过；该检查仅 8 条人工记录、1 个窗口，LR=0、参数变化=0，不是学习效果证据。旧源码完整工程验收仍 7/8，旧 R21 seed43 失败原件保留，不能与新 tiny 相加成 8/8。新源码完整 release 尚未执行完成/未准入，当前完整 entry 仍固定旧目录，新源码完整 profile 尚未测量；所有完整 GPU、数据与训练准入标志仍为 false。

2026-10-02T05:06:26.708214+00:00 只读快照：原八卡池 pending/running/results=0/0/56，8 个 worker 都在维护待机，心跳均小于 5 秒，监听 PID2213164 存活。此刻没有新平台状态查询，不把缓存 Running 当作新查询；未重启容器/worker/监听，未 Volc submit/cancel。checkpoint 创建/删除均为 0，自有 CPU tmp 已删除，GPU 子进程独占 tmp 清理有原报告；保留全部必要来源、原数值/失败证据及实际训练恢复状态。

用户指定数据仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v7_mhc_sequence_v1，manifest SHA e12e55ccb7a7a68c80d1aa1d0fe9da925b91fd74c6d8922abf3b536862d1ba94，train 6,849,243 / valid 52,572。manifest 的 training_ready 字段缺失，decontamination=audit_pending；audit_certificate.json、downstream_scope_certificate.json 仍缺失，父级当前范围审核未闭环且新增 manifest 未被父证书覆盖。既有清理保留，此状态不证明已发生泄漏。没有读取真实 shard、小 batch 或启动真实语料训练。

下一真实工程动作是新建不可变的完整 release 入口/运行包版本，保持既有来源验证、独立 oracle、阈值和数据门禁；不能热改已冻结的本轮 private/runtime 包。复用未变且已接受的模型数学证据，仅补必要的新入口/来源控制、当前源码完整 profile、完整梯度核对、3 个正式优化窗口、分 mask 评估及退出清理证据。当前源码学习、完整 checkpoint 恢复和原 data-selection 门禁仍需闭环。不再增加 hooks、重复已接受 tiny/CPU 项或迁移旧七项证据；不恢复此前停止的序列注释分支，独立 CUDA 算子验证继续暂停。正式数据审核未满足时不得启动真实小 batch 或八卡语料训练，用户授权无需重复确认。

[原 tiny 独立强复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/tiny_terminal_root_review.json)；[当前资源与数据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/final_state_after_CE_tiny.json)；[下一具体工程动作](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/next_engineering_after_tiny.json)；[模型 CPU 合同](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/model_CPU_contract_results.json)；[软件合同与清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_ce_layout_repair_20261002/runtime_CPU_controls_and_cleanup.json)。


## 2026-10-02T05:11:09.264315+00:00 CE 修正版完整合成训练入口接入预算

上一 Goal 回合是实质 progress：CE 私有修正、两环境各 4+3 合同与原 tiny PASS 已核实。本轮只接入同一模型 SHA 的新不可变完整 release 版本，限定 R21 seed43，一个完整合成验收，不热改本轮前已冻结来源。最多每环境 4 项必要新入口/namespace/来源绑定控制，180 秒；通过后才可新建来源包、绑定实际 manifest 并在原池执行 1 次。原 profile 900 秒、总 3600 秒、完整独立梯度/3 窗口/分 mask/退出清理协议保持；不单独重复已接受 tiny/数学合同，原完整 producer 自带 lifetime 阶段保留。当前未写新生产源、未构建/执行/GPU 入队。

停止注释不恢复、独立 CUDA 算子暂停；真实 shard/小 batch/训练不准入，V7 审核/ready 缺口保持；无容器/worker/监听重启或 Volc 提交取消。初次只读 guard 路径查错已留原件，按实际外部 private guard 位置读取，不改验证器。

[本轮预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/execution_budget.json)；[冻结保护](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/protected_before.json)。


## 2026-10-02T05:14:19.096400+00:00 完整入口元数据构建前置

必要依赖已核实：旧完整 entry 的 package_bootstrap 与 external guard 只接受固定实际目录，因此先 root 静态复审、实际 exclusive source-only build（全部准入 false）、绑定真实 manifest/guard，再做最多 4 项新增入口合同/env。没有 fake mirror/伪造 GPU profile；GPU 入队仍须上述合同和原完整门禁。A/B 正在仅交付来源/入口/分派源码，C 编写 4 项 stdlib-only 合同，root 独占构建/执行/发布。当前未运行模型或新 GPU 任务。

[依赖补充](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_only_build_dependency_addendum.json)。


## 2026-10-02T05:16:03.114559+00:00 CE 完整入口来源包实际构建

Root AST/diff 确认 package/producer/entry 全部函数与已接受来源相同，模型 SHA/字节不变；只新 namespace/schema、R21 seed43 身份与外部 guard 固定目录。12 个保护旧源不变。新实际 source-only 运行包 534 文件/531 parent/39 measured，manifest SHA 2ef8f6777120434c0371083838d3ad03d259e807153d7798a745197e7e9cdefd，全部准入 false，无 GPU profile/模型执行；构建是入口合同前置，不是完整验收。下一步绑定实际 manifest/guard 后只做 4 项必要新入口控制/env。

[静态原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/source_review_before_build.json)；[实际构建](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/runtime_build_actual.json)。


## 2026-10-02T05:20:53.599353+00:00 完整入口分派静态复审与最小变更暂存

新 parent 源 221 行/SHA48799767…1b3f909 已实际读取并由 A 独立只读复核；仅 R21 seed43 接新不可变运行包，其余既有身份保持字节固定旧入口，独立 CUDA 分支先拒绝。实际 manifest/guard/534 来源和冷启动 marker、loaded namespace 验证保持；没有真实训练入口。仅 runner 单个路径 literal 的 reviewable diff 已暂存，未发布、未执行或入队。12 个保护路径保持，等待 4 项新入口合同/env；模型原数学/tiny 不另跑。

[源与暂存](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/parent_source_review_and_staging.json)；[完整验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_CE_FULL_RELEASE_20261002.md)。


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


## 2026-10-02T05:48:27.525960+00:00 完整梯度进展与准入边界核实

当前权威进度看下列状态原件。profile/500学习必须绑定实际新源码及同一evidence_root；禁止重标旧来源证据。现有checkpoint与resume校验保持，不添加“初次训练前必须有真实checkpoint”的额外要求。当前 2026-10-02T05:48:27.525960+00:00：唯一新完整合成验收 1790918753525278168_d02_pilot_seed_gpu_r21_seed43_v2_82aa7438 仍在 GPU7/PID461451，原profile已通过（micro32、length2048、42.18384 GiB）。完整梯度检查原阶段已通过：86参数、8 microsteps/256人工记录，relative L2/maxabs=0、RNG一致，耗时698.856秒<原1200秒；11 weakrefs释放，最终allocated/reserved均0。仅核实原partial阶段，完整raw及独立终态validator尚待任务结束，不将局部通过当作完整准入。

正式优化已写出第1/3窗口（8 microsteps/256人工记录，loss3.660816848、原warmup LR0、参数变化0），仍待第2/3非零LR窗口、四mask评估及实际正常退出。池pending/running/results=0/1/56；其他7worker维护，未新查询平台状态。没有重复入队、Volc提交取消、容器/worker/监听重启。旧完整7/8保持历史，不和修正版局部结果合并为8/8。

V7确认为 immune_v7_mhc_sequence_v1；manifest e12e55cc…ba94、training_ready字段缺失、audit_pending、两份当前审核文件缺失。既有清理保留，这不等于已证实泄漏；真实shard/小batch/训练0。原数据门禁、停止的序列注释及独立CUDA算子暂停全部保留。

源码只读复核纠正剩余工作表述：checkpoint完整恢复核心已实现，初次1000不要求先有checkpoint；真实恢复证据须来自后续实际checkpoint，原resume校验不变。当前源码500步学习证据须位于新profile的同一evidence_root；3窗口不能替代。正式seed训练调用还未接到已有execute_admitted，source-only launcher仍拒绝。完整release通过后补必要学习证据和最小调用绑定，无需再写Trainer或重跑旧Linear恢复。最终架构选择、真实1000→5000及八卡正式训练尚未完成。

[本轮阶段原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/full_gradient_stage_root_read.json)；[最新状态](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/gradient_pass_formal_running_state_actual.json)；[剩余工作源码核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_full_release_20261002/post_release_remaining_work_readonly_review.json)。

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


## 2026-10-02T06:08:00.211898+00:00 必要学习任务识别项核实

原worker每次command都动态AST读取literal VARIANTS；新学习任务需要追加一个launcher_only识别项，随后由严格冷控制器处理。最终发布只允许d04_r21_CE_learning_v1这一条，原entry/函数AST保持，FULL包/模型数学/validator不改；不需要也不允许重启worker/容器/监听。实际八卡YAML job-seconds86400容纳28800秒学习预算，不修改worker参数。本刻只source准备，registry未改、新job0；人工seed42与正式profile来源seed43分明，数据/停线要求保留。[依赖原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/literal_registration_dependency_addendum.json)。


## 2026-10-02T06:10:35.497433+00:00 必要学习入口两项literal变更已暂存

仅暂存一个launcher_only学习识别项与runner入口路径替换，全部旧识别项及registry函数AST保持；公开文件未改，新job0。两个薄source模块仍由B实现，A准备唯一4项stdlib合同，root将在实际源码审阅与合同通过后才发布。FULL包/模型/validator不动。[实际暂存复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/literal_staging_review.json)。


## 2026-10-02T06:23:52.752151+00:00 两个薄控制source审阅及固定

两个新source仅复用原人工学习引擎与进程退出清理API；root已逐行审阅，修正本次index删除归属、signal退出标记和来源字段，并一次固定child SHA。所有旧保护路径保持，公开registry/runner未改、四项合同尚未运行、新job0。A正在完成唯一4项stdlib控制及独立边界复审，root随后只运行这4项/环境，完成后才发布或实际入队。[固定source原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/source_review_before_controls.json)。

## 2026-10-02T06:35:51.179554+00:00 新学习调用检查通过并最小接入

两环境各四项新增stdlib合同首次执行通过（pllm4.673秒、worker Python4.831秒），root实际原件/SHA复核及A独立静态审阅均通过。三源固定；只发布一个新控制入口、runner路径字面替换和d04学习识别项，旧registry项/函数AST、534文件FULL包/模型/validator、worker及其他受保护路径保持。无新Trainer、无已通过模型/profile/full/checkpoint/算子重复。人工学习seed42与正式来源/profile seed43分明；原25前置、真实exit/清理和产物哈希先于结果索引，500原heldout学习标准保留。

本刻尚未新入队；下一项仅原池一次25，原验证通过再一次500。真实数据/训练0，审核、selection、resume门禁及停线保持。测试ckpt0，自有空临时目录按实际清理，必要证据/实际恢复状态保留。

[新增检查](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/CPU_control_results.json)；[实际接入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/publication.json)；[清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/CPU_owned_cleanup.json)。

## 2026-10-02T06:48:24.130919+00:00 正式调用剩余项独立源码核实

两agent独立只读与root核实一致：当前冻结ENTRY.execute_admitted已具备原数据优先preflight、同源500学习、完整GPU/来源/seed及1000→5000完整恢复链，无须新增Trainer；缺的是正式请求/冷隔离加载的薄接入层。旧v4 seed launch永拒且只证明软件descriptor，不能当正式接入已完成。私有源码绑定可先实现，实际selection验证会重建并读真实数据，必须等待有效审核；不得通过运行该路径制造“准备完成”。当前修正版FULL仅R21/seed43，旧七份证明不迁移或并成当前8/8，learning42/selection42/formal43区分。

前一Goal回合为实际progress（4/4两环境、最小接入和唯一25已运行）。当前同一GPU1外层PID477484仍有poll-gated heartbeat：原worker只有process.poll() is None时更新heartbeat，源码SHA未变；不是只凭一个Running字样。25无终态，原观察继续，未重启/重试/新增任务。真实训练0、所有停线及数据/selection/resume门禁保留。

[独立源码与活进程证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/formal_call_remaining_readonly_review.json)。


## 2026-10-02T07:23:48.263825+00:00 八卡开训范围与剩余条件

用户已授权条件具备后在原八卡资源开始训练，无须重复请求权限。当前正式数据版本确为 immune_v7_mhc_sequence_v1，train6,849,243/valid52,572；manifest仍e12e55cc…ba94，training_ready字段缺失、audit_pending、当前audit_certificate.json与downstream_scope_certificate.json均缺。pending为父数据当前范围审核未完成及更新manifest未被父证书覆盖，父证据另有unresolved_benchmark_h3；既有清理保留，不表示已确认泄漏。两份审核分别覆盖当前数据清理/划分与下游test、few-shot support/query等保护范围，均须绑定最终当前manifest，不能由目录名或修改ready代替。

候选R20/R21/R24/R26代码实现已完成，七份旧来源完整GPU验收及修正版R21/43完整验收各自有效；不能并称同一源码8/8。Root本轮仅metadata核实八项profile与各自冻结runtime来源匹配，其余七项各自当前evidence_root中没有25/500学习记录；不迁移旧pool记录补齐。现有CE修正版25已原标准通过，唯一500 job 1790924220633707073_d04_r21_CE_learning_v1_dd5651b2在原GPU1运行，outerPID482000，最新已发布50/500（非终态）；四mask均值loss=1.350228、AA=52.34%、terminal准确率=0.00%，不得当真实理解/生成效果。正式训练薄调用尚未接完，最终架构尚未选择，当前不具备真实八卡开训条件。

下一步仅收尾既有同源synthetic学习前置和最小正式调用整合；已通过梯度/显存/profile/full/恢复验收不重复，不增加新架构波次。数据有效审核与对应selection绑定是实际语料初训的原前置，满足全部原门禁后按四候选×两seed并行初训，再依实测选择阶段架构。当前真实shard/小batch/训练0，新增入队/Volc提交取消/容器worker监听重启0，验证checkpoint0；原Goal身份/累计用量保持，停止序列注释不恢复/转交，独立CUDA算子继续暂停。当前人工500运行中，不能把整个Goal标为完成或仅剩外部审核。

[本次只读原件及八项矩阵](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_CE_learning_integration_20261002/training_readiness_20261002T072348Z.json)。

## 2026-10-02T07:26:59.319320+00:00 七项缺失人工学习前置有界整合

七项缺失的当前来源人工学习前置已冻结为真实逐slot绑定和有界预算。仅读旧强复核/COMPLETE/result/profile/sidecar及SHA，不重跑已通过GPU/full。新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002 仅允许3薄source、每环境4新增stdlib控制/180秒；每slot至多一次25及原验证通过后一次500，同一自己的evidence_root，learning42/micro32/GA8/global256、formal/profile42/43分别保留。复用现有原生成器人工manifest，不新增sequence/shard/数据。当前CE500及所有旧验收不重跑，原engine/FULL包/worker/validator/data-scope-selection-resume全部保持；停止注释不恢复，独立CUDA暂停，真实shard/训练0。此刻未发布或入队，下一步实现/源码审阅/4控制通过后才执行缺失项。root首轮metadata准备误把旧review.complete_sha字段对GPU_COMPLETE比较，已实际读schema纠正为原COMPLETE；首轮仅建空自有目录，无测试/model/job，错误元数据保留。

[绑定](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/slot_bindings.json)；[预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/execution_budget.json)。

## 2026-10-02T07:37:29.854839+00:00 正式训练资源合同实际阻断核实

正式入口源码/路径审计发现此前“只需薄controller”的判断不完整：五冻结runtime各缺两类下游审核资源目录，共10路径；原scope输入根绑定runtime内部，package禁止额外文件/链接。正式调用需要新版本显式外部资源合同并保留全部原审核条件，不能修改旧包或monkeypatch/改ROOT绕过。另原assert_ready未显式检查training_ready，正式外层必须按用户要求strictTrue；WORLD_SIZE=1只支持八个独立候选初筛，不等于单模型八卡DDP。当前仅读源/路径metadata，无gate/selection/真实数据/训练，原安全停线保持。

[精确缺口与最小方案](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_FORMAL_RESOURCE_BINDING_GAP_20261002.md)；[原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/formal_resource_binding_readonly_gap.json)。

## 2026-10-02T07:40:55.741762+00:00 七项学习四控制通过和最小接入

七项学习的4个新增stdlib控制在pllm/worker首次各4/4通过，elapsed13.690/14.348秒；root核对实际logs/receipts/三源SHA通过。仅发布新父入口字节副本、runner一个路径literal和七条synthetic-only launcher识别项；旧registry项/模型函数AST、所有FULL/模型/validator/worker及当前CE父child保护来源不变。没有新增Trainer、旧验收重跑或独立CUDA算子。当前未新入队，下一步每缺失slot一次原25，原验证通过才一次500；六个准入flags仍false，真实shard/训练0，停止注释/数据-scope-selection-resume门禁全部保留。当前CE500已实际发布75/500，无终态。

[首轮4控制](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/CPU_control_results.json)；[最小实际接入](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/publication.json)。

## 2026-10-02T07:58:40.899023+00:00 正式资源与单模型DDP必要接口的独立源码审阅

A/B实际源码审阅完成：正式资源合同须成为实际preflight/selection/build_trainer/execute/resume参数，新checker再调用旧assert_ready仍失败；不能通过短资源列表缩审核范围，完整inventory/scope版本仍依赖有效当前审核。单模型8卡还需global256的world因子（micro32→GA1）、metrics全局计数只reduce一次、8实际rank RNG/完整同拓扑恢复及rank0共同文件，原worker只单GPU需显式8rank组协调；HF/Accelerate基础可复用，生产源未改、新DDP测试/GPU/训练0。旧模型/CE/dataset可保持，不用旧full证明授权新adapter/ENTRY来源。

[精确缺口与实际最小接口](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EIGHT_GPU_TRAINING_INTEGRATION_GAPS_20261002.md)；[源码SHA原件](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_remaining_slot_learning_integration_20261002/formal_and_DDP_interface_readonly_decisions.json)。


## 2026-10-02T08:13:53.151504+00:00 单模型八卡私有工程路径

新私有接口与四项新增CPU控制位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002；预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/execution_budget.json，记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_RANK_INTEGRATION_20261002.md。仅stdlib通用协调/元数据，尚未发布或接入正式Trainer，不算DDP/GPU/数据准入，旧source包与门禁保持。


## 2026-10-02T08:35:16.052458+00:00 八rank私有接口验收状态

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002 的三个声明源码与四CPU控制已冻结/两环境首次各4/4；实际证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/CPU_control_results.json。此目录仍私有、不在旧immutable生产包，不授GPU/data/train准入；未来正式接口必须真正调用新协调/完整resume并保留原门禁。CPU控制临时材料已清理，source/log/receipt保留。


## 2026-10-02T08:44:04.665526+00:00 实际八rank Trainer私有路径

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002：4源码以内的numeric-only CPU验收，每环境一次600秒，旧模块保持。记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_RANK_TRAINER_INTEGRATION_20261002.md；尚未生产发布、GPU/真实训练准入。fixture checkpoint只在自有temp树，完成或失败收尾后保留JSON/raw日志、删除无需保留的测试ckpt，原真实恢复状态不动。

## 2026-10-02T09:02:06.944393+00:00 实际八CPU Trainer验收的进程与清理边界

私有来源目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002；启动前保持sourcefreeze/budget/source绑定，CPU effective CUDA为空；parent捕获setup及所有child收尾异常，leader已退也清理自身session残余组，再保存真实exit/log/JSON证据，仅删除自己的exclusive numeric临时checkpoint树。任何异常不自动重复该环境单次预算。当前未执行/未发布，不提供真实训练准入。

## 2026-10-02T09:16:09.780145+00:00 GPU任务PID必须在相同namespace核实

本地host di-20260121220916-9rvmj 的/proc不能判定remote t-20261001033327-fmnql容器child是否存在；PIDs缺失不可作为终态/重启依据。实际平台结果、remote poll-gated worker元数据和绑定原trace分别记录，强终态仍需原COMPLETE/result/exit及cleanup复核。本次纠正与证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/platform_task_readonly_current.json；错误probe保留 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/local_proc_namespace_probe_inconclusive.json。

## 2026-10-02T09:19:16.320175+00:00 当前私有四source冻结与执行入口

来源冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/source_freeze_before_CPU.json；入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/run_cpu_eight_rank_trainer_v1.py --environment worker|pllm。各环境只能独占一次ATTEMPT_STARTED，600秒parent总界；进程初始CPU group timeout不等同TA ddp_timeout60。执行前激活pllm，有效GPU visibility为空、旧保护源不改；失败不得自动复跑同环境，保留actual logs/receipts后仅清自有numeric checkpoint临时树。

## 2026-10-02T09:27:18.509373+00:00 失败数值阶段的必要原件留存

每个numeric Trainer成功阶段立即保存逐rank参数/Adam/scheduler/RNG raw与绑定，实际resume验证返回后native train前保存原receipt，后续失败仍保留，不只在整轮成功返回时留存。新修复目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002；原失败 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_integration_20261002/actual_CPU_failure_root_review.json。仅fixture CPU device规范化，不改安装源/native加载或生产门禁。

## 2026-10-02T09:32:44.629446+00:00 复用冻结CPU parent的修正版入口

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_trainer_CPU_device_repair_20261002/run_cpu_eight_rank_trainer_v2.py --environment worker|pllm；仅配置冻结v1parent函数共享globals的OWN/CHILD到新目录，前后核旧parent7e9c64f1及freeze/budget/protected gates，仍各环境独占一次600秒、空CUDA、实际PG/exit清理。新子类固定模块qualname以走原nativeHF args pickle，不修改/放宽加载。阶段及resume验证raw逐rank即时保存，失败原件保留。


## 2026-10-02T10:03:47.608079+00:00 八worker共享工作与真实端口FD边界

成员身份先于rank映射，实际remotehost创建group；端口持续FD租约到rank0 store，登记前失败不得伪造原coordinator完成。 新记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md；预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget.json。安全停线、独立CUDA暂停、原data-scope-source-selection-learning-profile-GPU-full-resume门禁保留。


## 2026-10-02T10:09:29.380497+00:00 原词表列表保存格式的实际兼容修复

原模型_save保存vocabulary.json为list，冻结numeric callback要求dict，无法直接接入。新显式私有回调 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/eight_rank_checkpoint_adapter_v2.py 仅_metadata保留原非空unique字符串列表、其余3dict，其他全部函数AST不变；不用FunctionType/globals或改词表包装，旧源/阈值/门禁/native/8RNG不改。新SHA 21815b3fd54ee51481f72f77cce1879d2fdf9d0e4818665c2cf72531169084b8 不继承旧GPU/CPU通过声明，当前未运行。新增1个必要source，修订预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/execution_budget_v2.json SHA 43ed1ccfc4cbeb71dbaf58d82781e3d1e6c8a473dab63d4b8ccebf0598a6657e；原未执行预算保留，CPU控制仍4/env180秒、GPU/发布/真实训练0。原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/checkpoint_vocabulary_source_delta.json。


## 2026-10-02T10:31:00.164239+00:00 八进程通信源码冻结与首次控制边界

5个新私有source及4个复用source已冻结，26旧保护文件未改。A/B接口复审完成；root补自有进程清理helper抛错时的有界fallback，C只修顶部CLI说明。首次每环境4项CPU transport/metadata控制待执行；不重跑已通过Trainer。此阶段无GPU作业/真实数据读取/生产准入，实际八GPU共同Trainer和原native词表保存恢复仍待验证。 来源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/source_freeze_before_CPU.json；详情 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/GENERIC_EIGHT_WORKER_GROUP_INTEGRATION_20261002.md。


## 2026-10-02T10:41:27.020725+00:00 新增通信独立复核闭合与原任务新进度

root原始证据与独立只读复核均通过；root复核88原件、独立报告103原件物理SHA，9source和26旧保护文件不变。完整本阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json。

截至2026-10-02 10:38:29 UTC，原同八项500步合成学习仍reported running，poll-gated心跳0.47–1.62秒：CE R21/profile43为450/500，其余七项325/500；pool0/8/65，目标8项结果及COMPLETE均缺。新快照 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/original_jobs_readonly_continuation_1037.json 保留旧已绑定快照不改。不重跑通过测试；实际GPU/生产保存恢复与V7审核均未完成，真实训练0。停止注释/CUDA暂停/原资源无重启或重复提交及全部门禁保持。


## 2026-10-02T10:44:51.097287+00:00 当前收尾与剩余外部资源阻断

新增八进程通信/清理的两环境首次4/4及独立raw复核已闭合：root88与独立103原件哈希重新核实，9冻结source/26旧文件不变，temp及checkpoint均0，不再重复已通过控制。实际单模型八GPU训练/原Trainer构造/native词表保存恢复仍未验证；生产入口未接通，真实语料训练0。

正式接入的真实阻断是缺当前V7合法完整scope/input inventory/registry与报告绑定格式，不仅是两份审核尚未passed。调用链会重复原固定资源gate、请求重建丢附加字段，selection还要求旧payload完整重算相等；自行选短path/SHA清单会缩减原保护范围。不能补假清单/证书、手改ready、monkeypatch旧ROOT或加入永拒placeholder充数。证书未通过本身不禁止通用开发，但没有权威完整资源契约就无法正确定稿这部分formal gate。

外部必要材料：当前scope_version及完整逻辑输入清单；批准的外部根、逻辑ID→绝对路径映射及文件/registry/evidence哈希格式；绑定当前manifest和同一输入全集的双审核字段语义；各候选/selection/执行/续训共同使用的同一版本资源契约。未来真实执行仍要求training_ready is True和两份有效当前审核，不降低任何原门禁。

当前可继续核实同8原任务的实际500终态并各自一次原verify_diagnostic(500,require_learning=True)；它们仍Running时不得因观察超时重启。后续单模型GPU/正式接入须满足当前来源及资源条件，不能用新numeric/GPU测试填补外部资源缺口。待权威契约到位，最小后续六类source为参数化资源gate、selection外envelope、正式request/preflight/build、execution/fullresume、cold entry及immutable package；此处仅明确必要边界，没有创建或执行六类实现。

只读决策 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/formal_resource_remaining_implementation_decision.json（d2631a1f…ba5c），root物理核5原source SHA及10实际调用位置；CPU阶段验收 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_CPU_transport_root_acceptance.json；10:43实际平台Running原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/platform_task_readonly_continuation_1043.json；10:38同8原worker poll-gated/trace仍CE450、其余325，无目标终态。

此前停止序列注释分支不恢复/重试/转交/绕过；独立CUDA算子暂停。原容器/作业/worker/监听不重启，不重复平台提交；不读取真实shard/抽真实小batch/执行审核selection/开始真实训练。全部原data/scope/source/selection/learning/profile/GPU/full-resume门禁保留。

## 2026-10-02T13:02:01.736603+00:00 V2最小锁协调修复准备

前一Goal回合实际核实V1八项终态失败、模型0步并定位长临界区，分类为progress；原Goal身份和累计保持。新有限预算/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_budget.json，SHA29eb09bf8449ba92101db33be7fe614d0a2f300799a4340e3a05e7b91989fa7c，保留V1已耗尽预算及全部失败原件。五个私有源已机械迁移到/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2；只有bootstrap将做通用锁行为修复，其余训练/保存/恢复逻辑保持，来源尚未冻结、未运行新控制。V2采用g02八个新成员名与后续全新jobIDs；准备阶段不stage/发布/入队，不重放g01失败job、重启旧资源或重复平台提交。

B只编辑V2bootstrap，C只准备一个stdlib定向CPU锁控制源码，A只读复核其余四源路径/预算重绑定；最多4项新增控制/环境/180秒，不重复原500、FULL/数学或已通过CPU/Gloo套件。真实数据门禁及停止注释、独立CUDA暂停保持。修复须在锁外完整核对来源、用两阶段确认防止未核对成员放行并记录实际锁耗时，不能简单增加时限或删除核对。当前八卡实测最新状态仍是V1启动失败、模型0步、worker回维护，V2源码准备不算八卡通过。

## 2026-10-02T13:35:29.543004+00:00 V2锁修复完成，静态问题已修正，首次CPU控制准备

五个V2源已交付并冻结完整545个物理输入；当前bootstrap SHA2e81b0b8b3cfd439693e5b95faa64ac244861a961c2ef958443b6cb572fd856c，freeze SHA22a20b96a1e7c43cc0741f98ac458593483f2c77500790a5a552cd7345f36435。独立复核发现初稿20380把原post-publication完整检查移到commit前，root在未执行任何控制/源码/任务时仅补一行assignment_after_commit锁外完整检查，保留初稿字节、初始freeze e979及原finding报告于/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/initial_review_bytes；修正复核/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/bootstrap_postcommit_correction_independent_review.json确认唯一差异和原期限/接口/35旧函数保持。未删除问题记录或改V1失败预算。

成员状态pending→checked、八成员generation/source/snapshot两阶段提交、完整来源扫描/协调器初始化在锁外、真实硬件/父子身份与持久abort保持。三个非bootstrap训练模块函数体不变，dispatcher只必要路径/新g02身份重绑定。四项新stdlib CPU控制源码SHAe301ccf21b3c2935ea6158b0d8ae9c9becc925e87f2894a49eb91585f02c5274，已独立静态复核。Root物理核对545输入、14旧保护及四份review；执行计划/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_execution_plan.json，将各环境首次最多4项/180秒，当前尚未执行控制/模型/GPU/stage/发布/入队。正式V7训练仍0，全部安全停线/审核门禁保持。

## 2026-10-02T13:41:46.940748+00:00 V2锁控制两环境首轮各4/4，八成员启动文件已准备待复核

实际父进程session73874 rc0结束，两环境并行总4.9816秒；worker4.7145秒/pllm4.6881秒，各4项首次通过，未重试。每环境8个member CPU子进程+1个holder均实际rc0/wait/PGgone，临时fixture已删，源/14旧保护/545来源不变。2.25秒锁外有限慢检查期间实际获取锁仅worker0.001245秒/pllm0.001999秒；actual2秒超时worker2.009051秒/pllm2.008957秒，持久CPUabort与后续拒绝生效。实际来源字节改变拒绝、generation/source/snapshot三拒绝及纯CPU合法assignment提交均通过。每环境保留12份原始计时JSONL及命令/日志/hash；此结果只证新增CPU锁协议，不证明共同GPU/model/NCCL/save-resume已通过。原件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/CPU_first_execution_receipt.json，独立raw复核正在执行。

一次metadata-only stage于2026-10-02T13:39:26.830527+00:00实际rc0结束/2.2419秒，文件/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/stage_v1/stage_metadata.json和/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_launch_plan.json已生成：八g02成员/八新ID/six-update合成计划，无GPU主机/intent。stage字节独立复核正在执行；本准备turn不publish/入队GPU，预算仍保持其原禁止字段。最后一次已有共同GPU运行仍是V1 failed/模型0步；V2 CPU通过不改变该历史结果，旧500/CPU/Gloo/FULL不重跑。真实V7数据/小batch/训练未启动，双审核及全部安全停线保持。

## 2026-10-02T13:53:26.861822+00:00 原Goal下一执行阶段开始：固定新八成员纯合成验收

上一Goal回合为progress：V2 CPU两环境各4/4及暂存字节实际验收完成。本回合沿原Goal身份/目标/累计，执行阶段有限预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/execution_phase_budget.json（SHA 415ece86836d13f755619621a2814384002c1b9734af89179286932996f673f8）；原准备预算29eb和禁止本准备turn字段不改，已完成控制/500/profile/FULL/Gloo不重跑。新执行阶段至多发布三份既有独立复核暂存字节，并使用既有固定八个新g02 jobID一次入队原池；不重新stage或生成ID，不使用会随机生成ID的普通enqueue CLI。新源码修改上限0，GPU组1/8成员/1800秒/70GiB；实际训练/原生保存恢复结果尚无。

已按task ID实际执行volc ml_task get -i t-20261001033327-fmnql -o json，返回Running；此前带bioseq名称过滤的list返回[]，不据此推断资源终态。没有平台submit/cancel、容器/worker/监听重启。两个子agent仅只读检查当前执行前置与传输路径；独立审查和新鲜资源/来源/原证据核对通过后才构造真正synthetic-only启动记录并入队。全部正式数据/GPU/训练准入仍False，真实V7小batch/训练0，停止注释及独立CUDA暂停保持。

## 2026-10-02T14:03:48.274562+00:00 V2固定共同八卡纯合成组一次实际入队

14:02:17 UTC通过原共享池发布既有固定8个g02 packet，group=original_r21_world8_1790948366803074817_36c1a3d5；启动原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/group_enqueue_receipt.json（SHA bc9285806e789f2124ac10db4b03ecc5f51112041f5526ecd4add555aa9cc647）。仅三份已独立复核暂存字节发布，原生545来源不变、12旧保护未改、两个public旧文件为明确已审阅变化，另新增V2 public dispatcher，runner最后发布。未使用随机ID enqueue、重stage、重放V1、重启容器/worker/监听或Volc提交取消。独立执行审查已完成，新synthetic-only启动记录绑定115实际前置原件；全部正式准入仍False，V7小batch/真实训练0，停止注释/独立CUDA暂停保持。

当前首次只读状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v2/actual_fixed_group_status_after_enqueue.json：pool={'pending': 0, 'running': 0, 'results': 89}；具体8job是否被领取/真实group intent以原件为准，实际Trainer/DDP/保存恢复尚未验收，不能把发布入队当成功。一次尝试预算已消耗，不自动retry或再发布pending。有限监督subagent只读观察同8个具体任务至实际终态/有限窗口；观察超时不表示原job退出。真实产物复核通过且各rank退出后再清理自有临时合成checkpoint，原证据及实际训练恢复状态保留。

## 2026-10-02T14:21:09.798620+00:00 V3通用协调修复准备

上一Goal回合为progress：V2实际8注册/8启动领取后锁超时，终态及清理已核实。本回合新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3；预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v3/execution_budget.json SHA d552c60b66d12a21785fb022fda997a366601638ddacd6d5b291a1f322fcaf71。仅修复成员/协调器嵌套锁、状态观测持锁及重复查询，保留模型数学、所有身份/来源/时限/abort/退出门禁。六份私有源码准备中，当前协调器SHA待最终绑定，未冻结、未测试、未发布或GPU入队。旧V1/V2一次尝试均耗尽且不可重放，现有资源不重启。V7真实训练0，双审核缺失；停止注释及独立CUDA暂停保持。

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

## 2026-10-02T15:40:28.282800+00:00 V4只读观测异常分类修复准备

V3失败组已实际终态并清理。独立新私有目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4，预算SHAb6e899550495a2743fbab12193ff7c3502741cb70644c8b144a57cee31c8240d，仅修复纯观测锁超时被上层误升级abort的问题。六份源已机械复制，下一只改coordinator明确异常类型与cold生产共用监督循环；最终source pin待绑定，未冻结/执行/发布/入队。原2秒锁时限、30秒心跳freshness、启动/总期限和持久abort/身份/来源/退出门禁保持，不依赖错误字符串判断。最多两项定向CPU场景/环境包含必要拒绝分支，不重复旧控制/500/Gloo。V3及全部前驱证据和既有资源保持，不重放旧ID，不恢复停线分支，不读真实语料。

## 2026-10-02T16:18:58.824015+00:00 当前V4监督修复接续入口

V4首次CPU已验结果 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/CPU_supervisor_root_review.json；暂存启动计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_worker_group_integration_20261002/actual_gpu_original_Trainer_v4/group_launch_plan.json 已生成，不重复stage或重建ID。真实GPU尚未执行，不能将CPU证明当作正式训练准入。当前进展以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_ACTIVE_GOAL.md 当前块为准；V3失败原件保留，已有安全/审核要求不变。

## 2026-10-02T19:14:30.135931+00:00 V7小样本实验入口

原池现在经 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_v7_pilot_dispatch.py 将p01_v7四候选×两seed导向冻结源码及固定十万条子集，旧别名委托旧入口；无容器重启。正式训练门禁不改。运行ID与恢复状态查 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_architecture_pilot_20261002/jobs.json；未完成任务不要删除latest.pt，不重复入队。

## 2026-10-02T20:44:10.457859+00:00 新120k组合架构与context指标入口完成CPU验收

上一Goal回合为progress：独立追加20k context正例并完成120k数据加载及新增A4布局验收。本回合继续原Goal，八p01实时worker心跳/优化步在推进，不重启或重复提交。新增Factorial 8组合+3对照，CPU24项通过；新3750入口2步人工检查和关系指标2项通过，自有CPU checkpoint已清理。11份方法YAML/22个两seed初筛格已登记，但尚未冻结/发布/入队，第三seed确认仍待完成。详见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_FACTORIAL_ABLATION_20261002.md。

正式world8缺口已只读定位：现pool每worker只见一GPU，p01是8独立模型；旧V4仅synthetic/R21/短步数，不能换data_dir直接作正式训练。新driver正复用原Trainer及八rank通用checkpoint接口，仍待实际GPU联合更新/保存恢复验收，不借旧失败或CPU记录宣称成功。当前final V7数据审核有效；所有原停线与源/数据门禁保持。

## 2026-10-02T21:01:39.089377+00:00 22项120k组合/CV实验已一次入原池

p02冻结566份源，复用p01完整源再显式覆写4个已验新/共享文件；冻结清单SHA 0a88b6a1b38b85fdaa9491444286da6b1bd023c721ce4828d7f5ca0b0b0e53ab。22个launch参数检查通过；只更新公共入口指向新分派并追加22个launch-only名，原78个名及p01分派保持。发布前后字节证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/deployment_v1/publication.json。

已实际登记22个唯一job ID到原pool（pending22、原p01 running8）：账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/jobs.json。先排baseline/原局部global/整段global/延后融合各两seed，再排其余GSL组合与固定门控对照；每项120000条、3750步。只是已入队，尚无p02 GPU初始化/优化步/完成证据，不宣称新方法有效。原卡自然完成p01后领取，不重启或取消在跑任务，不新平台submit。真实平台查询返回同一task Running；按原worker 258600秒预算估计入队时余22.41小时，原始CLI输出带版本提示行，已提取真正JSON记录而未把提示当任务故障。

正式八rank driver已落盘：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_formal.py；分离冻结模型源与真实审核资源，显式支持11组合/CV模型及原4候选、global=micro×GA×8、rank0写共享状态、八rank实际参数SHA与native optimizer/scheduler/RNG完整恢复。新协议CPU9项通过（子agent执行报告；原始日志归档待根端读取），仍未联合GPU运行或接入pool组启动；不将此写成正式训练已启动。接口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md。

## 2026-10-02T21:07:58.922239+00:00 正式入口隔离审核首次真实执行通过，p02作图已登记

冻结正式模型/Trainer与真实审核资源的分离接口已首次执行：原helper在python -I -B、CUDA隐藏的子进程调用原assert_ready，真实校验final V7所有分片及现行scope/evidence，54.28秒返回passed；parent清单与两证书SHA前后不变，实际门禁模块39份。没有创建审核证书或改ready、恢复注释或启动模型。源清单1137项、SHA 2c5a2a8a5ba9841dbffc569f1318a09d2497eed39a9ccc7ba5ed8e99104172b1；事实原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.stdout.json 与 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_first_execution.json。此处是新隔离执行接口的首次实际核验，非重复签证。

root已读取原工具stdout归档，正式driver新增协议测试9/9、3AST通过；证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_driver_evidence_20261002/receipt.json。正式入口源已冻结，但launcher还没接通，联合NCCL更新、实参同步、保存恢复实际GPU证据仍缺，formal_training_started=false。正式长程预算需结合原资源剩余时间设置可恢复阶段；不允许因全部小实验排完便把8份独立模型视为world8联合训练成功。

p02作图 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/plot_v7_context_ablation.py 新4项纯数学/结构控制通过；首次读真实22项账本输出JSON及4CSV，awaiting_shared_evaluation、missing22、figures=[]。没有用假效果值画图。数据列包含source/context分母、关系CE/accuracy及正交G/S/L对比，完整图页的真实渲染待共同评估出现；说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_CONTEXT_ABLATION_FIGURES_20261002.md。原p01监督完成后续看p02，至多12小时或22项全终态，无新增OS监听/平台任务。


## 2026-10-02T21:35:07.714476+00:00 正式入口源码 v2 与验证 loader 规则

正式源码候选应读取 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_manifest_v2.json（SHA 566bec35f5ee0ea65047a3e69d864e729cff2de2ad1a861929401551cc368799）；修复仅使 eval loader 保留尾批，训练 drop_last=True 不变。HF 的 num_samples==len(dataset) 不能证明实际行覆盖，须使用真实汇总分母。新 helper 绑定证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/parent_gate_v2.stdout.json 已真实 passed。v1 仅保留历史，不直接发布；v2 也未获联合 GPU 运行验收。transport 内存 intent 校验不等于实际占有全部 worker，必须在原队列任务结束后统一协调八个真实空闲 worker，禁止部分 rank 先启动等其他实验腾卡。完整状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_EIGHT_GPU_ENTRY_20261002.md。


## 2026-10-02T22:15:25.209702+00:00 大batch调整的当前参数与证据

运行中的旧p02仍micro4×GA8/global32，验证micro4；已观察显存峰值约3.15–3.86GiB，不能代替micro16/32容量证明。下一协议候选16×2或32×1，统一global32、完整3750步、验证micro4，最终档位须来自同资源实际profile。由于现有micro-mean loss与GA绑定mask种子，跨batch协议分开作图，旧checkpoint不能靠改provenance恢复。14个未启动旧参数job原件位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/batch_revision_20261002/withdrawn_pending，撤下证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_context_ablation_20261002/batch_revision_20261002/withdrawal_receipt.json；不再把这14项当pending/failed或自行补回。新大batch实验清单尚在准备，不是已提交的平台资源YAML。

## 2026-10-02T22:39:58Z 接续记录与新大 batch 文件约定

每个实际里程碑应在同一回合写入对应项目 `.md`：进度和原始证据到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md，方法决策到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md，入口索引到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ARCHITECTURE_RESEARCH_INDEX.md。记录核实时间、已执行范围、结果/失败、证据绝对路径与下一步；明确区分源码完成、CPU 验收、GPU 验收、入队和训练完成。历史记录保留，但最新入口必须反映当前实际状态。

新大 batch 的 11 份 YAML 已位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_large_batch_ablation_20261002，每份对应一个条件和两个 seed；它们是既有 pool 的实验清单，不是 Volc 资源申请 conf。当前 batch/峰值/profile 证据仍待实测，不得填入推测值。新冻结清单位于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/source_snapshot_v1.json，SHA256 `2f416e927d0b2e95aaa2e5932490330464622a3490bea2eac15c1f8b4bb2d783`，共 568 文件。源码冻结不等于 GPU 容量通过或实验已入队。

八卡共同训练新增接线的当前源码/人工检查记录统一在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_POOL_INTEGRATION_20261002.md；真实 pool 是一个进程中的八个 worker 线程，八张卡独立实验成功不能替代同一个模型的八 rank 联合训练验收。自有 CPU 临时 checkpoint 已清理；p01/p02 唯一实际恢复 checkpoint 及证据继续保留。数据格式本轮未变，因此没有为本次状态核实改写数据格式文档或审核文件。

## 2026-10-02T22:52:04Z 既有 pool 的唯一容量任务与公共入口版本

容量测量已登记一次，唯一 job1790981433321562451_p04_v7_capacity_febb28e3，账本 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_job.json。它是原池的 profile-synthetic/72、无 data/resume；登记后 pending1、旧训练 running8，尚无 GPU 测量。不得因等待或观察超时再次登记、重启或将原十四项补回。新大 batch 的 22 项训练仍未登记。

公共早期调用脚本现在为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_dispatch.py，SHA256 `f259fb0f0946fb7434c1f78b06f64e5d186ece7b781fd29dac7f3f0fe0e5d866`；原 p02 调用脚本仍 SHA `0946d193e7f6e5bb080cc79447a72e09095c543df64fa44b36b9f1e98eb923d6`。原 100 个 registry 项完整保留，新增 capacity 项后为 101。公共训练脚本 SHA `b56125b96db1561833d417fd71f9065ea191cb6769ec2db0a304e76629db12bc`，registry SHA `0a6460f49a8c6aef93d70375a14698640ef508f185dcb91169f24164e7ff4ea6`。前后字节和恢复依据存于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v1。

模型源冻结清单的 status 保留冻结时刻事实；当前运行状态应读 capacity_job、pool 和监督原件，不为更新状态而改写冻结清单 SHA。该 capacity 接入不表示正式八卡启动代码已部署。未来新训练接线应基于实际公共入口版本审查，保留当前验收来源与全部数据/安全门禁；不能仅把旧 SHA 换成新 SHA便称通过。

## 2026-10-02T23:01:46Z 大batch绘图源及阶段判断入口

已验新的metadata绘图两文件独立冻结在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/analysis_source_v1；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/analysis_source_v1.json（SHA256 `0e9634f9398a46c08fb6f4a30500fa5b608ec00dd401cdb8c21bdc8176368b0d`）。未来从该目录的plot_v7_large_batch_ablation.py执行，显式指定独立jobs、pool、batch-plan及其SHA、输出prefix。这不是模型训练源或审核证书，不能改动原568模型源来更新图。

图/CSV/JSON标注实际study、训练micro/GA、eval4、plan SHA、partial范围和缺格；旧组结果不作为新组对照。人工metadata验收图已清理，实际新组图尚无；真实训练唯一恢复checkpoint保留。当前优先候选及限制统一见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ARCHITECTURE_INTERIM_DECISION_20261002.md，候选排序不能作为正式架构已选定的证据。
## 2026-10-02T23:25:23.615132+00:00 新大batch调用检查的独立冻结边界

已审阅调用检查与专用人工控制的原件冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1.json，SHA256 a7cad07e42638134bfb19c6b94d280a24ad36b22de823871bb20c18204fc673f。与568份模型源、两份绘图源分开保留，不改写既有模型清单或替换正在运行的源。实际能力仅限5组人工metadata协议控制及源码字段核实，非GPU验收。

未来新组固定22个 `p04_v7_{method}_seed42/43` 名称；必须先保存完整 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json 后发布任何pending，统一计划 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json 必须引用唯一capacity任务的真实四原件和实际选档通过证据。当前上述真实文件均未创建，alias及公共调用尚未发布。capacity_job.json是历史入队快照，最新成功取同job实际pool result，不重写该快照以声称完成。现有data/audit门禁仍须真实执行，冻结或人工控制不构成豁免。
## 2026-10-02T23:49:12.122059+00:00 正式pool票据与checkpoint恢复约定

pool票据 `job.steps` 表示本段新增optimizer updates、范围1..5000；正式plan.segment_steps表示累计driver停止步。resume起点必须读既有formal_checkpoint.json和其中SHA绑定的trainer_state.json/run_provenance.json，禁止从目录名猜步数；八票据resume/data/steps必须一致且等于冻结driver实际参数，target-prior严格等job.steps。恢复binding摘要复用冻结driver['digest']，不得使用compact plan/intent摘要替代，两种JSON序列化不同。实际完整二进制/optimizer/RNG恢复继续由原driver核验。

正式wrapper、dispatcher和新segment控制三文件独立冻结于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v1；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v1.json，SHA256 828a0010868da8bef2172b3b2940b18ad1c8b66a89a94db364789cd2aad9f0bb。软件检查原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_segment_digest_fix_20261002/receipt.json；当前未发布或实际world8验收。

原容器pool实际argv使用相对 `scripts/immune_research_pool.py`；以实际父/proc/cwd解析source，原argv保持，禁止重写成猜测的绝对调用。真实pool身份必须七字段和完整cmdline，由原容器读取，旧controller父PID线索不够。拟capture原件固定pool/logs而非capacity run目录，因冻结profiler会拒绝初始额外文件；当前尚未capture。数据审核、停止注释及独立CUDA暂停不受此通用工程约定影响。
## 2026-10-03T00:09:34.673530+00:00 原唯一容量任务的父pool读取已接入

metadata observer独立冻结 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_source_v1；清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_source_v1.json，SHA256 aa8dee4b53240758310d8dd814e6476c5712310b8ffe6d8feca8ad8863958253。公共早期入口只接入它，并按8d13c32c6c8e0b6ee65df4573a14e784947e98602e50c7ef10cf2459393e79c8核对source；当前公共脚本SHA d8dda1c32226bdc4ac070d5c605c28a55ba460d481f48708e801b5392e554f8c，capacity dispatcher/registry不改。真实publication及前后字节保留 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/pool_parent_metadata_deployment_v1。

仅原job1790981433321562451_p04_v7_capacity_febb28e3实际启动时才capture；唯一目的地 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/logs/1790981433321562451_p04_v7_capacity_febb28e3.pool_parent_runtime_observation.json，不写profile run目录或重写原件。保留原pool_argv及实际cwd，完整身份读取前后相同；source_sha仅证明观察到的磁盘字节，不冒充pool已加载源码证明。其他job立即无IO返回；没有新任务或监听。当前仍pending、实际原件缺失，未来构造formal plan须读该真实原件并由wrapper再次现场验证，不用开发机self/parent测试身份替代、不把它当审核/容量/训练准入件。

## 2026-10-05T13:24:26.771800+00:00 新容器续训attempt与证据保留

修复后的launcher只使用新stage/group/job IDs；--previous-stage必须指向同次训练实际成功的前段，八rank结果和清理均核实后才能恢复。失败stage与abort持久记录保留，不覆盖、不重放。冷启动修复源与检查固定保存在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/transport_preload_v1，模型/训练冻结源仍为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/source_snapshot_v3。原launcher按字节归档在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_20261005/launch_v7_formal_segment_original.py。

## 2026-10-05T13:30:36.519597+00:00 正式恢复验收与日志指标口径

实际8→16原生恢复已经通过，原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/segment_16_attempt2/root_segment_acceptance.json。后续仍使用固定25042调度，单段新增≤5000，必须显式指定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/transport_preload_v1/formal_pool_transport.py；每段八rank实际成功后接下一段。HF4.48.1提前stop的train吞吐使用全horizon，resume train_loss使用累计global_step；报告段内吞吐需用实际新增步/实际调用秒数，报告本段rank0 loss需校正分母，不能称为全rank均值。正式保留最近3个native checkpoint，早期验收JSON另存 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/launch_20261005/first_resume_evidence，不额外复制weights/optimizer/RNG。

## 2026-10-05T13:40:03.875317+00:00 正式训练监督接续约定

新监督器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/supervise_v7_formal_20261005.py 采用独立单实例锁、一次launch记录，启动后不能静默重启；STOP路径 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/STOP 只停止监督与后续接段，不终止既有训练。相同stage存在即拒绝重放，冻结launcher只调用一次，部分入队失败不自动重试。运行状态只读观察，非平台重启/新申请；轮替的旧checkpoint仅保留JSON/恢复/指标证据，不额外复制权重或optimizer。实时文档为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FORMAL_SUPERVISION_20261005.md，关键事件同步PROJECT_PROCESS。

## 2026-10-05T13:45:46.163722+00:00 监督器已接管后续训练段

PID3977587及启动identity见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/root_process_identity.json，固定运行源 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/source_v1/supervise_v7_formal_20261005.py。它存活时不得手工并发接段或起第二个监督器。后续helper已固定为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/supervisor_launcher_v1/launch_v7_formal_segment_20261005.py：STOP在admission_commit前阻止所有票据；commit后保证整组8票据发布，之后停止下一段。正常checkpoint轮替与文件观察失败不等于训练失败；绑定/来源/ack错误不降级。每段JSON/指标证据归档在 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/checkpoint_evidence，权重仍仅正式run最近3份。监督文件/进程不重启、GPU平台资源不重新申请。

## 2026-10-05T13:52:43.383249+00:00 剩余消融执行清单

/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/remaining_ablation_plan.json 保存原22格固定job及结果、8完整/1本地在跑/13尚未启动的事实，以及后续绝对路径命令和实际观察的白名单环境。planned_not_enqueued不是已提交；运行前仍检查实际上一进程与COMPLETE、原数据/来源/batch条件并使用独立fresh输出，禁止重复当前本地任务或抢占正式world8组。矩阵说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_ABLATION_COMPLETION_MATRIX_20261005.md。


## 2026-10-06T04:39:59.360843+00:00 正式分段训练的观察口径

当前冻结driver在每1000步常规eval/save之外，还在stop_after段尾强制eval/save/stop；5016已实际产生独立评估。训练日志仍每25步输出，已保存步数、已评估步数、最新已打印训练步数必须独立读取，不从epoch小数或目标步数推算。SEGMENT标记的paused是原程序段尾状态，不能当成用户暂停Goal；接续由唯一原监督器在8rank成功/清理/原生恢复门禁通过后完成。当前原件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/segment5016_root_review_20261006.json；后续agent以 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md CURRENT和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_supervision_20261005/state.json 为准。跨段日志只读工具 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/observe_v7_logged_steps.py；根状态文字另外从native log_history读取最近eval步数。不要修改冻结训练/监督/pool源码来刷新这些状态。


## 2026-10-06T14:35:22.374740+00:00 后续八卡消融入口版本

当前已准备的私有入口为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_v2_dispatch.py，它要求 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_post_generation_acceptance.py 的两轮及9项生成完成证据。原 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_metrics_only_ablation_dispatch.py 仍由现有本地guard按SHA绑定，不改写、不按旧终点启动八卡消融。v2尚未发布，未来publisher需先核实空闲/监督退出/剩余时限和最新22格账本；所有新local/pool尝试共享 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/guarded_dispatch_v1/cell_locks，不另起一套同名锁。检查与剩余工作见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。


## 2026-10-06T14:49:24.133710+00:00 原八卡后续消融发布包

固定准备包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/package_v1.json；入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_local_ablation_20261005/post_generation_pool_v1/source_v1/publish_v7_post_generation_ablation.py。conda pllm下 --action preview 只读查看真实22格，--action publish 必须在50082及9项生成完成、三个监督实际退出和八worker空闲后才允许发布。当前prepared_not_published，原运行不变。资源按实际原pool argv的604200秒总时限和86400秒单任务上限核实，不沿用旧README的72小时说法，也不延长deadline。发布记录存在时不得重放。共享锁释放后才写enqueued，worker仅等本次批次完成发布后执行。工程验收与未来实际执行分开记录于 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_POST_FORMAL_ABLATION_REUSE_20261005.md。


## 2026-10-06T16:38:02.720689+00:00 第二轮原生checkpoint核查入口

/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_epoch2_checkpoint.py 用于已实际进入50082日程的第二轮，激活conda pllm后运行 --step 新保存步数 --previous 上一已验收点。首个上一点25050绑定 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/epoch2_continuation_20261006/initial25050_root_review.json；后续使用同目录checkpointN_root_review.json。沿用原指标helper，只读元数据/源码，不创建模型验证。旧 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v7_where_checkpoint.py 仅用于已结束的父阶段，勿用于第二轮。已完成验收不重复，不改冻结训练源；实际检查结果写PROJECT_PROCESS。
