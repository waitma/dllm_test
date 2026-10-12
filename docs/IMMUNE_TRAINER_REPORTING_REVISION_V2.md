# Trainer runtime v1/v2 只读呈现修订

2026-10-01。本修订只在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/isolated_worktree` 实现，未改 live reporting/daemon、runtime source 或 T20 冻结候选。父代理维护 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md`，本文件记录独立实现与 CPU 证据。

## 固定版本选择与边界

`/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/isolated_worktree/scripts/immune_diagnostic_reporting.py` 新增固定 v1/v2 白名单。相对于调用方提供的 canonical 原始 source root，只接受：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v2_20261001`

有 summary 时，以 `summary.runtime_root` 为唯一声明；缺 summary 时，才读取同 run 的 `request.json.runtime_root`。未知版本、非 canonical 字符串、relative path 和 symlink runtime root 均呈现 `source_status=rejected` 与 runtime identity mismatch，不回退另一版本或 request。pending 尚无产物时不猜 runtime，保留未知状态。记录额外提供 `runtime_version` 和 `runtime_identity_source`，原 D01 transport variant/route、script 名称与 proof schema v1 不变。

有 summary 的分支保留原 runtime manifest SHA、完整 source map/逐文件 SHA、五个 GPU artifact SHA、formal/raw windows 与 proof extension 一致性、严格 boolean/int、三 windows、父进程观察和 outer COMPLETE/summary hash 全部标准。summary 缺 source map 不允许回退 manifest 来掩盖缺失证据。

缺 summary 的 failed run 可从 pinned request manifest 核验对应版本 source bytes。`source_status=matched` 只表示源码相符；`execution_completed`、`functional_gpu_passed`、`training_started` 仍为 unknown，实际 optimizer windows 为 empty、进度 `?/3 optimizer windows`，不会把 oracle 耗时当 formal windows 或 GPU 通过。无 GPU artifact 或 COMPLETE 的失败仍明确呈现 execution failed。

原 v1 失败冻结保留。父代理计划的新 runtime v2 预算为整体 3600 秒、oracle 1200/formal 1800/evaluation 600 秒；本 reporting 修订不设置预算、不执行 producer、不签任何 GPU/data 证书。T20 v1 仍冻结在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_t20_pool_launcher_cpu_20261001`，本轮没有 rebase T20。

## 验证与来源记录

只运行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/isolated_worktree/scripts/tests/immune_llada/test_typed_trainer_runtime_reporting.py`：原 14 项 typed Trainer 检查加四个必要版本测试，共 18 项；pllm Python 3.10.19 通过（2.15 秒），worker Python 3.12.12 通过（0.42 秒）。没有重跑 D01 routing、T20、attention 或其它旧测试组。

四个新测试覆盖 legacy v1 在 v2 存在时仍绑定旧来源、当前 v2 保留来源/artifact/windows 标准、unknown/noncanonical/symlink 拒绝且不 request fallback、failed request 分别准确选择两版并保持 GPU/windows unknown。fixtures 全部位于 pytest 临时 artificial_repo，并以 `FABRICATED_REPORTING_ONLY.json` 标记；源文件注释和 JSON metadata 明确 fabricated reporting-only。固定 job envelope 保持原始精确三个字段，不往生产 job schema 填入测试标签。所有 fake proof 只供呈现测试，不是实际 acceptance。

源文件修改前的副本在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/source_before`，原 SHA 记录在 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/source_before.json`。supervision/monitor 仅作为测试依赖复制，字节未改。

最终证据路径：

- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/pllm_reporting_pytest.log`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/pllm_reporting_junit.xml`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/worker_reporting_pytest.log`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/worker_reporting_junit.xml`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/reporting_revision_v2.patch`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/cleanup.json`
- `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_trainer_reporting_revision_v2_cpu_20261001/validation.json`

cleanup 仅删除带此轮明确 fabricated reporting-only marker 的 artificial_repo 和经内容核对的对应呈现文档，不删除 pytest 的共享父目录或其它 agent 文件。validation 绑定 source_before、最终 source/test/doc/diff、两环境日志/JUnit 和 cleanup SHA。该记录仅为未发布的只读 CPU 修订；根端可在 CAS review 后发布此 reporting 字节，并独立决定 LOCAL monitor 是否重载。
