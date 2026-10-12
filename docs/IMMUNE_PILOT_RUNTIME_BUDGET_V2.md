# Trainer 运行包 v2：预算修正与实际启动

更新：2026-10-01 04:24 UTC。v2已终态失败：梯度核对通过后GPU张量驻留；正式三窗口未运行。新私有v3定位中，最终训练未启动。

## 2026-10-01 04:24 UTC 终态与下一步

04:12:41 UTC，新D01 `1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a` 已终态failed，runner rc2、child rc1。独立梯度核对耗时695.0528秒，低于1200秒预算；8个microstep/256条人工记录、78参数比较通过，梯度倍率1.00000077、相对L2误差0.00015194。随后报 `Oracle left GPU tensors resident before formal measurement`，错误先于正式三窗口；未生成raw/summary/通过标记、checkpoint=0。parent709.2715秒观察正常退出并完成后代清理；这与进程内GPU张量驻留检查是两个不同结果。[终态原件核验](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/terminal_review.json)、[实际失败日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer.log)、[阶段统计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/producer_stage_evidence.json)。

v2未记录清理后具体allocated字节，不能凭峰值判断驻留对象。新私有v3分支正在核查对象生命周期与AMP包装；真实tiny CPU弱引用检查已释放对象，尚未复现GPU驻留，不据此确定根因。尚未发布、构建或入队v3，不放宽显存/数值标准、不重跑独立CUDA算子。父快照、v1/v2不可变runtime和失败证据保留。

原八卡任务继续Running，最新独立监督为pending/running/results=0/0/38、8worker维护；监听PID2172045继续watching。T20 v2私有两环境各48项通过但保持未部署；同版本GPU验收及原数据准入未齐，真实四候选1000→累计5000和最终训练未启动。验证checkpoint不保存；小型失败日志、统计与哈希保留。

以下带时间记录属于当时状态。

v1 的独立梯度核对耗时701.6219秒，超过事前240秒阶段上限；parent716.1639秒正常回收，失败先于正式三窗口。旧 runtime 和失败原件保留。v2 事前预算为oracle1200/formal1800/evaluation600、整体3600秒；2048输入、micro32×GA8/global256、三个正式窗口及全部梯度/CE/计数/RNG/85%显存标准保持原样。根线程独立AST核对32个原函数和固定数值常量未变。[producer复核](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001/budget_v2_revision/root_source_review.json)。

新增producer_stage_evidence.json在开始、实际窗口和阶段返回后保存小型统计，始终标记gpu_accepted=false/raw_completed=false；不能作为完整raw或通过证书。失败保留具体phase、耗时与已有统计，没有缩小batch或放宽容差的fallback。

两环境必要CPU检查为producer各29项、package/entry/transport各89项、typed结果各18项，正常退出且CUDA未执行。[producer证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001/budget_v2_revision/final_review.json)、[入口证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/validation.json)、[结果呈现证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/validation.json)。typed只接受固定canonical v1/v2目录，summary优先，缺summary时按request核对来源；source matched不表示GPU通过或窗口完成。根端另外以新版只读代码核对真实v1失败记录的526文件来源。

03:53:00 UTC根线程按原址SHA核对后发布9源码/测试，[发布记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/publication.json)。03:54:07实际创建526文件v2，全部524父文件保留原字节：

| 对象 | SHA256 |
| --- | --- |
| v2 manifest | `3805679f1d7370042cd39741d4839ad1c6acc3805be899bed3b1f95872dace12` |
| producer | `ae1a60cbf0f155482f93dd2b1a612883b3a9de06b8be4b07458fc39360da175d` |
| 固定入口 | `370d34a5a68dff3a8ba411bf658f5cda8e32157e729d35b0133ffaf0602f71e9` |

[实际构建证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/build_v2_evidence.json)。两环境在实际v2独立检查八份profile与54项目模块导入通过，未构造模型/执行GPU/数据门禁：[pllm](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/runtime_cpu_inspection_v2_pllm.json)、[worker](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/runtime_cpu_inspection_v2_worker.json)。

03:56:42仅本地监听重载为PID2172045，旧2143672退出；第一次检查早于首轮heartbeat，后续实际watching核验通过，没有再次重启。八卡任务及GPU worker不变。[监听记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/monitor_refresh_v2.json)。

04:00:41在原池入队唯一新验收 `1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a`，GPU7于04:00:42领取。[启动证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/d01_v2_launch.json)。variant的v1表示运输合同；实际运行包以job_id、request.runtime_root和manifest SHA区分。当前progress为gradient_oracle/starting，budget1200；04:02实际GPU7 100%/51401MiB，尚无正式三窗口/通过记录。监督持续只读检查。[实际request](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/runs/1790827241091679008_d01_pilot_runtime_gpu_r20_v1_ea0bba4a/request.json)。

root两个自有pytest目录共1924文件已删除、checkpoint=0；producer五处显式自有目录清理前保留小CPU记录，typed36个标记fixture已删除。[root清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_budget_v2_cpu_20261001/cleanup.json)、[producer清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_gpu_producer_cpu_20261001/budget_v2_revision/ownership_cleanup.json)、[typed清理](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/cleanup.json)。人工GPU关闭checkpoint保存；小型指标/哈希/失败来源保留。

T20已经在新的私有目录绑定runtime v2，两环境各48项通过，旧v1源码和证据保留；未发布/apply/生成实际pointer/入队，等待D01终态后复审。[T20新版本证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_v2_20261001/validation.json)。03:41真实数据元数据仍training_ready=false、双证书缺失，manifest601db7…efce和selection659ef8…4f2未变。数据原准入、四候选比较与最终训练条件见[整合记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PRETRAIN_INTEGRATION_20261001.md)。
