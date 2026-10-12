# 固定 CUDA attention 诊断 adapter（未部署）

2026-10-01。实现只使用随机张量和整数 mask，不读取序列/数据集，不构造模型，不训练，不保存 checkpoint。本轮只执行 CPU 合同测试和真实 CPU 子进程生命周期回归；**没有执行 CUDA，也没有修改 worker、registry、稳定入口或冻结的 39 个训练文件，没有入队。** 根 MD 由 root 同步。

入口：[pool_attention_diagnostic.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_attention_diagnostic.py)；[测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pool_attention_diagnostic.py)。底层复用 [cuda_attention_probe.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/cuda_attention_probe.py) 和 [attention_backends.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/attention_backends.py)。现有容器的后续路由条件见 [执行入口审阅](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXISTING_POOL_DIAGNOSTIC_EXECUTION.md)。

## 固定入口和预算

公开接口：`VARIANT`、`validate_envelope(job)`、`parser()`、`run_diagnostic(job, *, output_dir, profile_dir)`、`main(argv=None)`。仅接受这个精确 envelope：

```json
{"variant":"d00_cuda_attention_probe_v1","mode":"profile-synthetic","steps":1}
```

`steps` 必须是整数 1，布尔/浮点不接受；任何多余键，包括 data、resume、shell 均拒绝。CLI 只接受上面三个字段及 `--output-dir`、`--profile-dir`，拒绝缩写、隐藏 worker 参数和预算覆盖。旧 mode/steps 仅为现有池的运输格式；这里没有一个训练步，也不产生模型显存 profile。

输出必须是尚不存在的绝对 `<pool>/runs/<job_id>`，父 `runs` 已存在；profile 参数必须精确指向同一个池的 `<pool>/profiles/d00_cuda_attention_probe_v1`。拒绝符号链接、非规范路径、覆盖或恢复；**只校验 profile 参数，从不读写该目录**。GPU 设备索引固定 0，后续池路由仍须确保 worker 给子进程隔离单张 `CUDA_VISIBLE_DEVICES`；adapter 不申请资源。

四项按 MATH FP32、MATH BF16、EFFICIENT FP32、EFFICIENT BF16 的顺序执行，固定 B1/H12/L256/D64、global mask、table bias、有 padding、local_radius=32、seed=8421、5 次 warmup/20 次计时。单项 120 秒、整体 600 秒。调用 `probe.execute_plan`，每 case 新进程、整体串行，保留原数值容忍度、数学对照、实际 kernel 和 bias 梯度要求；不自动尝试其他后端，也不放宽失败条件。

## 子进程退出证据的必要补充

原 59 项验收版本没有将实际子进程退出码保存到报告：worker 写出一个负结果后异常退出，父级可能仍保留该状态。本轮仅对隔离 probe 作了最小修正：

- `run_case_process` 正常返回后记录 `child_process.returncode` 和 `cleanup_completed=true`。该函数的 `finally` 已完成本 case 进程组的终止和 wait；异常或清理失败不能凭 JSON 推断成功。
- 记录 `reported_status`、`expected_returncode`、`protocol_valid`。`passed` 必须退出 0，其余 worker 状态必须退出 2；0/1/负信号/143 等不匹配一律转 `failed`、`phase=parent_execution`，保留真实退出证据。
- 协议错误使用 `ValueError`，避免报错文字中的 `unsupported` 被后端关键字分类器再次误归为正常负结果。

原生命周期方案保持不变：同 session 独立进程组、`posix_spawn` 无 `preexec_fn`、临时 subreaper、只回收本 case PGID、清理预留在原 timeout 内且最多 2 秒，小于池 10 秒 TERM→KILL 宽限。父 SIGKILL、后代主动脱离 case 组仍无法普遍保证；adapter 写报告阶段被默认 SIGTERM 终止时也不保证写出 FAILED，但不会凭不完整文件补写 COMPLETE。

## 结果完整性与退出码

adapter 同时核验 returned summary 与磁盘严格 JSON 完全一致，拒绝重复 key、NaN/Infinity/溢出浮点。固定 plan、四个唯一且有序的 case ID/配置、每项 spec/result 与 summary 一致；重新计算 status_counts 和后端配对摘要。所有完成项必须有正常子进程退出/清理证据、CUDA 实际尝试阶段、固定 CPU 私有 RNG 输入 SHA、同一运行环境、有效错误或通过证据。

`numerical_mismatch` 必须有完整五项比较、原容忍度、至少一项失败、实际请求后端的 kernel 证据、完整误差数值；非有限比较只接受明确失败且指标为 null。不能仅写一个状态字符串就完成诊断。SHA 绑定入口 request、plan、summary、spec/result、完成项 worker.log；发布标记前再次检查这些文件，并检查 adapter/probe/oracle 三个项目依赖的前后 SHA。Torch/CUDA/Python/GPU 版本由每个 case 的实际环境字段记录，未把环境版本号当作全部第三方包的文件指纹。

| 逐项结果或整体状况 | `execution_completed` | `all_cases_passed` | adapter 退出码/标记 |
|---|---|---|---|
| 四项均有完整证据并 `passed` | true | true | 0 / COMPLETE |
| 四项完整，其中有 `unsupported`、`oom`、`numerical_mismatch` | true | false | 0 / COMPLETE，保留具体负结果 |
| `backend_mismatch`、`unavailable`、`timeout`、`failed`、`not_run_budget`、`dry_run` | false | false | 2 / FAILED |
| 结果损坏、缺项/重复、异常 worker 退出、源码/证据变化、清理失败 | false | false | 2 / FAILED（能写报告时） |
| 父 TERM/INT | false | false | 143/130；不写 COMPLETE |

这与原 probe CLI 的退出语义不同：原 CLI 存在未通过项即退出 2；adapter 直接调用 `execute_plan` API，将**完整、可核验的负诊断**视作执行完成。没有伪造一次原 probe CLI 的退出码。实际每个 worker 的退出码始终保留。

`completed_case_count` 只计严格验证完成的 case；它与最终整体成功分离。四项通过后源码改变，计数仍可为 4，`execution_completed=false`、`all_cases_passed=false`、error 指明 SourceChanged，绝不写 COMPLETE。输出损坏时计数可能只包含损坏位置前已核实的项；不得凭状态字符串补齐。

## 报告合同

新目录保存 `execution_request.json`、`probe/` 原始证据、顶层 `summary.json` 及 COMPLETE 或 FAILED。schema 固定 `immune.pool_attention_diagnostic.v1`。summary/marker 都包含：

```text
schema_version, kind="cuda_attention_backend_probe", variant
training_started=false, biological_quality_claim=false
case_count=4, completed_case_count, case_statuses={case_id: status}
execution_completed, all_cases_passed
source_sha256={repo-relative path: SHA256}
source_sha256_before, source_sha256_after
probe_summary_sha256, evidence_sha256={run-relative path: SHA256}
transport={mode: "profile-synthetic", steps: 1}, error, created_at
```

marker 另有 `summary_sha256` 和 `status=completed/failed`。COMPLETE 仅在 execution_completed 为真时出现。没有 global_step、训练 trace、推荐 microbatch 或晋级训练证书。监控应先按 kind 展示 case 完成数和 backend 状态；仍须后续独立审核早期路由、禁止 init 自动入队及旧 release 快照/来源绑定，本 adapter 不实现这些接入动作。

## CPU 验收与历史

激活 pllm 后运行以下命令；worker 环境使用对应 Python 3.12 可重复：

```bash
python -m pytest -q scripts/tests/immune_llada/test_pool_attention_diagnostic.py \
  scripts/tests/immune_llada/test_cuda_attention_probe_cpu.py \
  scripts/tests/immune_llada/test_cuda_attention_probe_lifecycle_cpu.py
```

最终日志与精确 SHA 见 [验收记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/validation.json)、[pllm 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/pllm_pytest.log)、[worker 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/worker_pytest.log)。adapter 的 CUDA 环境/kernel/计时 JSON 全部是明确标注的 CPU test double，仅存在 pytest 临时目录中，不能当成 GPU 通过。真实进程回收仍由既有 12 项子/孙进程测试承担。

历史保留：[初始 47 项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_validation_20260930/validation_manifest.json)；[生命周期版 59 项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_lifecycle_validation_20260930/validation_manifest.json)。它们绑定各自历史 probe SHA，不适用于本轮新增退出证据后的文件。初次联合测试 123 项中 10 项揭示上述 RuntimeError 关键字误分类，113 项通过；修复后两环境各 125 项通过，再补 7 项误差字段和 request SHA 检查，最终以本轮验收记录为准。
