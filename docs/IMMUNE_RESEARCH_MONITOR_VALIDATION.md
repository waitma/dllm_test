# 工作池监听验证与边界

2026-09-30 UTC 更新：Refined runner、条件分组曲线及独立可识别性审计范围回归全部通过，15项测试用时0.63秒。源文件、测试文件SHA与命令见 [local_validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/local_validation.json)。监听仅只读查询平台与共享目录，不执行 pending 中的命令，不取消、不重提或自动加入任务。

新增覆盖 `run_immune_refined.py` 实际 JSON 格式：`mode=train` 且 `kind=synthetic_generalization_diagnostic` 仍标明人工诊断；按 job_id 独立保留25/500等不同预算，读取25步 trace、四档 mask 均值、group/length 分组及计数、train_probe、人工 manifest。全局 level 指标按 runner 的 token 计数汇总，不将各分组直接等权替代全局 loss。后台文档保留每次全局评测，分组初始/最新/训练探针由监督文档呈现，逐次全部分组指标仍在原 summary.json。

除原来的 profile 证书、旧源码快照外，新增检查 refined 源文件集合中的新增模块、真实训练 provenance 源文件、COMPLETE.summary_sha256、人工诊断的 completed 状态和 biological_quality_claim=false。遇到源码/结果SHA错位或NaN时只记录告警，不更改实验。

旧 R00–R11 与 R12–R19 已分别映射各自冻结 source_root，实际20条终态记录均 matched，20成功/0失败。旧20步 end=0仍作为历史学习信号保留，不能混同执行失败。

本地 daemon 每次升级均先确认上一进程正常退出，再用setsid方式启动，主机di-20260121220916-9rvmj。共享flock防止同池重复监听。当前进程和心跳以动态 [pid.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/pid.json) 与 [heartbeat.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/heartbeat.json) 为准；后台进程不代表对话窗口常在线，也不承诺模型会在主对话结束后持续自主排错。

复算命令（先激活 pllm）：

```bash
python -m pytest -q /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_pool_monitor.py /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_research_supervision.py
```

测试覆盖坏JSON/中文/Markdown转义、旧证据保留、训练step读取、遥测过期、文件锁/STOP、禁止执行队列内容、只读平台查询/90秒频率/20秒超时、预期OOM分类、失败与完成一致性、分版本指纹、不同预算独立记录、嵌套NaN和新文件检测。

另已只读验收实际 Refined CPU smoke 输出：[control2](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/control2)。39个源码SHA匹配，COMPLETE与summary摘要匹配；18个group/length分组、四档mask、train_probe、kind与step均正确读取，零告警。证据：[monitor_format_validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_runner_cpu_20260930/monitor_format_validation.json)。该项只有36条人工train/18条valid、2步/global2，只验证报告格式，不能冒充生产GPU profile或25步结果；临时验收目录已删除。

500步阶段另验证8个新job与各自25步job的initial全部18组指标、fixture manifest和39个source SHA完全相同，global batch均256，kind均为synthetic_generalization_diagnostic。证据：[r20_r27_500_boot_validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r20_r27_500_boot_validation.json)。这是同协议重新初始化的人工诊断，不能称为checkpoint续训或真实语料训练。

监督文档新增每25步的conditional/unconditional曲线，按每档实际AA/end/selected计数汇总，再平均四档；不把各组未经加权的百分比直接平均。新回归覆盖不均匀计数、无end样本、未知组排除，原18个分组仍全部保留。

500步任务首次25步评测已在8项全部落盘。每项完整step25记录（全部18 strata、四档mask和gradient norm）与之前同方法独立25步任务逐字段完全一致；conditional/unconditional end此时仍均为0，继续观察50步及以后，不认定为执行失败。复算对照证据：[r20_r27_500_step25_verification.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930/monitor/r20_r27_500_step25_verification.json)。

自动监督文档把原六条toy审计与新72条扩展heldout审计分开，后者显式展示对应fixture SHA、conditional/unconditional上限和联合最优时end取舍。旧toy上限不能套用新实验；新审计只解释输入碰撞，并非泛化保证。独立数学审计还有2项回归通过，见 [扩展审计](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_RESEARCH_EXPANDED_IDENTIFIABILITY.md)。
