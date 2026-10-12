# 项目操作约定

项目根目录：/vepfs-mlp2/c20250601/251105016/project/dllm_test

## 当前入口

| 内容 | 唯一维护位置 |
|---|---|
| 当前状态和下一步 | [PROJECT_PROCESS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md) |
| 模型方案 | [BIOSEQ_MODEL_PLAN.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/BIOSEQ_MODEL_PLAN.md) |
| 数据与格式 | [DATA_FORMAT_AUDIT.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/DATA_FORMAT_AUDIT.md) |
| 实验里程碑与更正 | [实验记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/EXPERIMENT_LOG.md) |
| 下游分数和比较口径 | [downstream/benchmark/RESULTS.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/downstream/benchmark/RESULTS.md) |

当前新训练数据是 /vepfs-mlp2/c20250601/251105016/project/dllm_test/data/prepared/immune_v5_receptor_completion。配置身份仍为v7_v5_layout_chain_global_20261007，名称不代表数据必须来自V7。

当前attention分段单位是prots/protd蛋白对象；split仅用于输出链解析。实现证据见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_v5_layout_20261007/capacity302_regression.log。

## 操作规则

- 运行Python前激活pllm；测试仅保存报告／指标，不保留测试checkpoint和临时文件。正式恢复点不能按测试文件删除。
- 正式checkpoint按best3＋latest自动轮换后，核对被删除点的JSON档案与原生回执哈希一致；历史指标及来源证据保留。只读核对入口为 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v5_checkpoint.py，不重新运行已通过的整模检查来验证档案。
- 使用绝对路径。提交训练走volc ml_task submit --conf及训练YAML；评测走评测YAML。提交／取消后更新进度表和简短事件记录。
- 当前已有作业使用冻结源码和数据；不修改运行中的绑定、不重复提交已完成或正在运行的任务，不将工作区新版本注入旧训练进程。
- 保留既有STOP／abort／监督停线；禁止自动移除它们或重复启动失败作业。先前暂停的独立CUDA诊断、序列注释／VNAR分支不因文档整理自动恢复。
- 原平台池按任务名immune-v7-pool-8gpu-20261005查询；状态查询需非交互JSON。2026-10-07用户要求暂停后已停止，不自动恢复原消融或占卡负载。
- 当前V5容器内训练使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_v3/launch_data_args.json 及固定源码副本；data-binding和SHA256为必填。原生门禁重新哈希全部85分片，不能只替换路径绕过。旧A4继续原V7门禁。
- 下游协议及指标保留各自版本、数据、checkpoint和生成条件，不能因切换新数据覆盖历史成绩。

- 2026-10-08用户授权下游评测优先使用闲时资源（Preemptible=true），也可本地GPU执行。迁移到本地前核实资源占用，并避免与已提交作业重复执行；不抢占当前训练或其他正在运行的任务。

## 文档保持简单

当前文档只写结论、状态、下一步和证据链接。同一实验更新原行；逐次轮询、心跳、重复进度保留在output原始记录中，不每次新增大段章节。只有数据／方案决定、启动／终态、结果和更正进入实验记录。同一事实不复制到四份md。

详细老约定和历史特殊情况：[完整历史](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/archive/project_docs_20261007T061603Z/PROJ_GUIDE.md)。归档不是当前方案；有新的用户决定时以当前文档为准。

历史直接训练YAML /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_object302_scratch_8gpu_20261007.yml 对应已失败的首次启动，不能作为当前运行入口重提。当前global512=8×8×8；八卡3步synthetic启动检查已通过，证据 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2。启动前如发现明确training_ready=false则拒绝，不创建或伪造证书。训练权重不从synthetic或历史checkpoint初始化。

2026-10-07用户明确采用持续容器：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007，资源YAML=/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/immune_training_container_8gpu_20261007.yml。训练由容器内supervisor执行，每个票据一次；PAUSE_TRAINING只停训练，STOP才释放容器。子任务失败保留资源，正式训练前停止全部standby进程。后续不再将暂停实验解释为结束整个容器。

持续容器平台Running与训练完成/失败必须分开核实：同时检查实际进度和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_training_container_c20250601_20261007/results 下对应训练票据的退出结果。观察窗口到时退出不代表训练停止；完整训练需最终300000步产物与真实成功退出共同证明。

2026-10-07本版本显存实测：8×A10080、micro8、synthetic总长1218，峰值allocated最大4.48 GiB；实际用户指定global512配置保持。尚未对更大micro做显存搜索，不称micro8为吞吐最优。证据：/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/attempt2/startup_review.json。

2026-10-08用户进一步授权本轮37项下游评测使用非闲时资源；4组替代作业已设Preemptible=false，原闲时作业确认终止后才提交，不重复执行。其他评测仍优先闲时/本地。

2026-10-08用户明确：持续监督当前训练，训练退出后继续八卡占卡、保留当前容器。现有容器监督器已具备此切换：训练期间占卡全部停止；训练子进程wait回收、结果归档后，恢复每卡一个standby。失败也保留资源且不自动重试训练；STOP/资源到期仍按原规则退出，不能移除停线。已核对启动绑定的supervisor/helper/standby哈希一致。

当前只读训练至占卡观察入口：/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/watch_v5_training_to_standby.py；本次有界50小时（180000秒），状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/latest.json，退出记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_object302_noesmc_300k_b512_supervision/training_standby_watch_20261008/exit.json。观察器不控制GPU、不重复发训练票据；以训练真实退出、state新心跳、八个不同PID和各卡新初始化日志共同确认占卡恢复。最终完整训练产物仍须另行核对。禁止重复启动同一个观察器，文件锁防重入。

2026-10-10新增270M实验配置目录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/configs/models/v5_noesmc_270m_20261010。实际268,590,112参数；候选batch64×GA1×8/global512，先完成八卡A10080整模profile，显存数据尚待实测。旧75.6M冻结包及原门禁文件保持；新270M门禁使用 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/check_v5_270m_formal_parent.py 和 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v5_270m_data_binding.py，仍强制原审核、85分片字节哈希、架构精确匹配及training_ready停线。

270M最终batch采用64×GA1×8/global512：八卡A10080实测长度1218，allocated42.79GiB/reserved48.88GiB；micro64已达保持global512时的上限，不额外增大全局batch。实际从容器内票据v5_270m_scratch_bs64_20261010.json执行。参考YAML /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v5_noesmc_270m_8gpu_20261010.yml不重复提交；运行包 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v5_full_binding_20261007/launch_270m_train_bs64_20261010。

270M当前监督入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/watch_v5_270m_training_to_standby.py，状态 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_v5_270m_noesmc_300k_b512_supervision/training_standby_watch_20261010/latest.json，退出结果同目录exit.json；本轮会话26867最多15天，不重复启动。保存核对入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/review_v5_270m_checkpoint.py --step STEP；与75.6M核对入口分开，实际参数数268590112、micro64/GA1固定检查。
