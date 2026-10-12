# 单模型八卡工程接入记录

最新 2026-10-02T08:35:16.052458+00:00：两个私有通用接口与四项CPU控制完成，两环境首次各4/4通过；尚未接实际Trainer或验证DDP。

2026-10-02T08:13:53.151504+00:00：进入必要通用工程实现。原 Goal 身份与完整目标保持；上一轮属于 verified wait：本轮再次核实原八个具体 job 仍由新鲜 poll-gated heartbeat 的 running worker 执行，没有将观察到期当作退出。

当前只实现私有纯标准库接口：八个 rank 的共同启动和失败退出，以及全局批量、分布式指标、八份 RNG/完整 checkpoint 的元数据合同。旧 worker、模型、Trainer、完整来源包、validator 与已通过验收不改。此阶段不会执行真实审核、selection、真实数据分片读取或训练，也不执行 CUDA 算子、GPU/DDP 任务、平台提交/取消或重启。

[独立有限预算](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/execution_budget.json)：最多三个新源码模块，每环境四个新增定向 CPU 控制、180 秒，新增 GPU/池任务为零。哑文件仅为临时 CPU 控制材料，不能写成实际训练恢复证据。四个控制只针对新边界，不重跑旧模型、数学、profile、完整 GPU 或学习检查。

| 新接口 | 负责 agent | 当前状态 | 要解决的实际问题 |
|---|---|---|---|
| eight_rank_coordinator_v1.py | architecture_keep_or_change | 实现并通过首次CPU控制 | 原 worker 每次只领取单卡任务；八进程必须同组、八个不同 GPU、完整到齐后才能共同启动；缺 rank 或退出须全组停止 |
| eight_rank_contracts_v1.py | architecture_validation_strategy | 实现并通过首次CPU控制 | world8 时 micro32 要用 GA1 才是 global256；全局指标不得重复归约；实际保存恢复必须包含八份 rank RNG 与公共状态 |
| test_eight_rank_contracts_v1.py | root | 四case首次两环境均4/4 | 真实 CPU 并发登记/启动屏障、失败传播与不完整恢复拒绝等必要新控制 |

上述接口尚未接入正式 Trainer。CPU 控制通过也不能宣布实际八卡 DDP、完整恢复或真实训练已通过；后续冷入口、Trainer/metrics/save 接入与实际合成验收仍需完成。正式资源合同、当前 V7 双审核、严格 training_ready=True、原 selection/source/GPU/完整 resume 门禁仍保留；停止的序列注释分支不得恢复、重试、转交或绕过。

原八项 500 步合成学习继续运行，pool0/8/65。CE 已发布150步，其余七项现在也发布25步；八项均无终态，不称500验收完成。

[本轮具体原任务核实](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/existing_eight500_readonly_start.json)；[受保护旧源](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/protected_before.json)；[必要八卡接口依据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EIGHT_GPU_TRAINING_INTEGRATION_GAPS_20261002.md)。

## 2026-10-02T08:35:16.052458+00:00 首次验收与未完成边界

三源SHA见 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/source_freeze_before_CPU.json，CPU实收日志/receipt与root复核 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/generic_eight_rank_integration_20261002/CPU_control_results.json。pllm4.223秒、worker4.331秒，各4case首轮通过；八CPU rank真实登记/同workSHA/8claims屏障/实际回收、wrongwork/dead/missing/stale/lock持久abort、world8/指标顺序与8RNG文件/ACK/TOCTOU/同源续训元数据均覆盖。所有返回execution_ready/data/train准入false；CPU文件没有tensor反序列化或实际Trainer恢复。源码静态修正前coordinator948073ea…、修正后2d5a5e1f…，修正的是锁超时持久abort与共同work identity等，不放宽旧门禁。

原八卡8项500合成学习继续，池0/8/65；CE已发布200/500，四mask均值loss0.389031、AA86.00%、terminal94.42%。七新500均已发布50/500，R20/R21/R24/R26平均loss分别1.461254/1.347973/1.207699/0.989363，terminal准确率当前均0；所有任务无终态或最终验收。学习种子统一42，profile42/43不是独立学习重复；不同已发布步数不可横向选优。真实语料训练0。

下一步需要真正调用这两个接口的冷rank/Trainer/metrics/save adapters、共同工作和原审核资源参数的贯通、实际分布式Trainer保存恢复验收；当前没有这些执行证据，不能把CPU接口通过称成单模型八卡训练完成。原旧GPU/full/model检查不重复；所有新验收必须针对实际新增边界，有独立有限预算，不能把旧证明直接授权新adapter/package。审核依赖外部有效当前范围结果，不能自行补签或恢复被停止分支。
