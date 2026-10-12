# CUDA operator诊断的监督类型

2026-10-01。新监督分支通过两套环境各32项CPU测试及3项跨模块测试；读取实际八个325步任务的状态、kind、源码匹配、告警、目标步数和microbatch，与旧daemon相同。随后已更新本地daemon，未接入GPU诊断或改动39份训练源码。

[轻量报告模块](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_diagnostic_reporting.py)只使用标准库，识别固定variant `d00_cuda_attention_probe_v1` 与 `kind=cuda_attention_backend_probe`。旧worker运输参数 `profile-synthetic/steps=1` 表示一次专用调用，监督显示 `n/4 cases`，训练step与推荐microbatch保持未定义，不读取或沿用该诊断的模型profile。

报告分开保留已严格验收的case数、逐case状态、整体执行完成、全部数值通过。unsupported、OOM及numerical mismatch可以构成完整的负结果；backend mismatch表示未执行请求的后端，和timeout、CUDA unavailable、不足预算、损坏证据或整体源码变动一样，不能当作成功。四个结果字符串均为passed，也不能覆盖整体校验失败。监督只核对报告的一致性、源码与产物SHA、summary/COMPLETE绑定及case身份；实际kernel、梯度容差和子进程退出合同由执行adapter负责，监督不替代它的严格验收。

现有[持续监听](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/monitor_immune_research_pool.py)和[监督判读](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_supervision.py)加入这一分支，原训练判断保持原样。负后端结果以原状态展示，不伪造 `global_step=1` 消除警报。监督拒绝混入训练进度/模型profile宣称、缺失summary、重复case、校验不一致、损坏哈希及worker错误的success状态。只读监听不执行任务、不入队、不签发证书。

[最终验收JSON](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_diagnostic_monitor_20260930/validation_final.json)绑定实现、测试、两套日志/JUnit和实际八项source检查。最终回归为17项新增人工JSON合同、10项原supervision及5项原monitor，pllm为32 passed/1.07s，worker为32 passed/0.51s。再用实际adapter输出，经磁盘读取进入supervisor/monitor，验证全部通过、numerical mismatch与backend mismatch三种情况；各环境3项通过（2.58s/0.26s），子进程结果仍明确为CPU TEST DOUBLE，未使用CUDA。测试显式禁止GPU操作，并核对三模块状态集合相同。已有monitor测试包含真实监听子进程的锁、心跳与退出。

初版31项及中间32项成功日志保留；最终版本按adapter合同明确了整体失败可以保留已完成case，以及backend mismatch不属于完整负结果。没有修改旧结果或把历史SHA当成新版验收。

本轮没有测试checkpoint。2026-10-01T00:06:15Z仅替换本地监听进程，PID1951375→2033525，保留共享锁和既有state；新heartbeat为watching、8项running/28份结果，后续持续落盘。八份running job字节保持相同，平台任务、容器worker和八项500步训练未重启。[部署证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_diagnostic_monitor_20260930/deployment.json)。
