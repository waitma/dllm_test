# V7 同协议接续准备，2026-10-03

状态：独立适配器已准备，未发布、未创建真实恢复票据、未入队或执行恢复。完整 22 格目标保持；本模块只处理原任务已经终态失败、保留一致 checkpoint 的同协议接续。原资源到期后可使用显式的新 attempt pool，不重启旧 pool。停止的序列注释、独立 CUDA 专项与数据审核门禁保持。

源码：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_v7_ablation_resume.py`，SHA256 `1d095118992ee9977b616da1845106506ac2ca11768808d28727de6abf7991c2`。标准库入口 `prepare(argv, ticket_catalog_sha256=...)` 返回原冻结 runner 命令；`main(argv, ticket_catalog_sha256=...)` 在完整准入、单 GPU mask 与长持锁后监督 child。调用方必须显式绑定恢复账本 SHA，不能从未绑定的默认参数自动授权。实际账本缺失时拒绝。

## 票据、来源与两个输出目录

未来账本固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/ablation_resume_tickets.json`，schema 为 `v7.ablation.resume_tickets.v1`，`automatic_retry=false`。账本绑定原 study、原 22 jobs、batch plan、568 源清单与 frozen large-v2 dispatcher 的绝对路径和 SHA；`tickets` 以新 attempt ID 为键，同一账本不允许两个 attempt 指向同一原 cell。

每票据包含 `original_job_id`、`original_run`、`attempt_pool`、`attempt_run`、`prior_step`、`timeout_seconds`、完整 `job` 及 `evidence`。evidence 按真实原件绑定原 result、finished、provenance、summary、curve、checkpoint 的路径/SHA。job 的 steps 必须为 `3750-prior_step`，其 `resume_from_checkpoint` 固定原 inner 的 latest.pt；prior 必须是小于 3750 的已保存 250 步边界。timeout 显式且不超过 86400 秒，外层 pool 资源期限仍可更早终止任务。

origin pool 固定 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930`。原 inner、终态证据、原 catalog 与 base.prepare 的路径约束全部保持。新 outer 和 profile 必须来自票据明确声明、绝对且无软链接的 attempt_pool。适配器分别检查 origin 与 attempt pool 的 pending/running/workers；两者相同时去重，仅当前 attempt pool 的精确本任务可被豁免，并验证 worker PID。两池出现其他相同 cell 的持有者均拒绝。

完整复用 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py`（SHA `b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3`）的原 3750 job 冷 prepare，维持真实 catalog、registry、capacity、source、模型与数据路径检查。生成命令仅追加 `--resume`；inner、variant、seed、micro32/GA1/global32、eval4、3750 horizon 均不变。旧 p02 协议拒绝迁移。

## 长持锁、执行与完成原件

先解析绑定票据确定 cell，再获取按 original job ID 命名的共享 flock；完整 prepare 在锁内重新检查终态、双 pool 所有权、所有 SHA 与 checkpoint 事务一致性。未来运行只在这些检查后读取 checkpoint 字节计算 SHA，外层不加载 checkpoint 模型内容。原冻结 core 继续验证 binary provenance、curve 和模型/Adam/scheduler/CPU 与 CUDA RNG，并执行真实续步。

child 使用原冻结命令的 `-I -B -u`，保留外层进程组。`pass_fds` 传入同一 flock FD，使 parent 清理不确定或退出时，尚未退出的 child 仍持有锁；直到最后一个 FD 引用关闭，第二 writer 不能获得该 cell。监督器有明确 timeout 和有界 terminate/wait/kill/wait。receipt 只表示直接 child 的实际 wait/reap，不宣称证明整个进程组清理。冻结 core 无 DataLoader 子进程；原 pool 的进程组退出处理仍保留。

outer 只创建本 attempt 的 ATTEMPT、CHILD_EXIT、FAILURE 或 COMPLETE 原件，不覆盖原 result/jobs。只有 child 正常 wait、rc0、已 reap、无监督错误，且 inner 的真实 COMPLETE、summary、provenance、curve 前缀与最终记录/epoch/source 全部吻合后，才写带完整 inner 文件引用的 outer COMPLETE。失败或清理不确定均不能生成完成声明。

实际字段已按冻结 core 静态核对：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/source_snapshot_v1/scripts/research/run_v7_architecture_pilot.py` 第 114–115 行使用 `provenance.data.path`；第 225–241 行执行 binary 恢复检查；第 261–269 行按 latest→curve→summary 保存，非完整事务必须拒绝；第 319–325 行使用 `COMPLETE.global_step=3750`，最终 records/record_exposures=120000、epoch=1、position=0。没有用 fixture 自创字段替代这些原协议。

## 本次有限 CPU 控制与归档

均在 pllm、空 CUDA mask 下执行；未运行真实 Popen/exec、模型或 GPU，未读取真实序列或真实 latest.pt，未创建真实票据、registry、队列或审核证书。人工 checkpoint 仅为自有 TemporaryDirectory 内普通字节；全部人工目录已清理。实际 frozen prepare 只冷读来源、元数据及既有容量证据。

1. 初版五组控制首次执行通过，2026-10-03 04:36:54 UTC，1.0644 秒，exit 0。覆盖缺票据拒绝、原 frozen prepare 参数翻译、live/steps/curve/p02/checkpoint/source/catalog 拒绝、锁与 mock child 失败、人工完整 inner 的 outer 证据。原始源码 SHA 为 `57958d65114f310238186592314f79c8da6e26f11e417e466d22ed062cd0e43f`；该结果不冒充覆盖后续改动。
2. 双 pool 与 inherited FD 变动的定向控制仅执行一次，通过，04:49:36 UTC，1.0293 秒，exit 0。实际冷 prepare 验证新 outer/旧 inner，拒绝旧池 live、旧池冒用当前 ID、新池错误 PID、错 outer 与已终态 attempt。人工 dup FD 验证 parent close 后锁仍排他，直到最后引用关闭；mock unreaped child 不能完成。未重跑初版五组。

初版原件目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_ablation_resume_controls_20261003`。receipt SHA `4cbbaabeee9023c2a5b455845a9922b968d1a79d22604d0c0cb6463de8041748`；stdout SHA `b3bf9c13e5a65269de0531914e262e7cd5d0d22fcaccb3ddca4c94756a5bbbd1`。

当前定向原件目录：`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_ablation_resume_dual_pool_controls_20261003`。receipt SHA `2b3cbb2257a9e767cb7a87c23a12a8c0bad2e86a7524e00e7381f30f8139f7ae`；stdout SHA `8e32a0d75b66b5251fac9be21ae887745e8600313fad2a690448c787b5e877cc`。两目录各自保留 receipt.json、stdout.log、source 原字节副本及 execution_archive.json；初版归档不覆盖。

## 尚未完成的实际接线

尚需根端审阅/冻结、未来任务真实终态后的票据创建与 SHA 绑定、明确新 pool 资源准入及公共冷分派接线，随后才可能取得真实 GPU 恢复和最终完成证据。当前没有端到端恢复通过，也没有后续轨迹数值等价声明。原矩阵中 never-started pending 格的迁移、作图消费恢复 attempt lineage 是两个独立未完成事项，不能把本模块称为完整 22 格矩阵已经接通。主会话负责同步 PROJECT_PROCESS、BIOSEQ_MODEL_PLAN、PROJ_GUIDE 与 CURRENT。
