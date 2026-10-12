# Isolated CUDA attention probe

状态：2026-10-01 **退出码证据修订后，probe 64 项 + typed adapter 68 项，两环境各 132 项 CPU 测试通过；实际 CUDA 验证待执行**。当前 500 步任务结束后，由 root 在确认空闲的 GPU 上运行。工具不提交任务、不接入 worker/registry、不修改当前 39 个冻结文件，也不导入项目模型、数据集或 benchmark。

入口：[cuda_attention_probe.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/cuda_attention_probe.py)。数学 helper 复用 [attention_backends.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/attention_backends.py) 的整数 group、padding query fallback、local mask 与可学习 bias。背景审阅：[IMMUNE_ATTENTION_BACKEND_REVIEW.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_ATTENTION_BACKEND_REVIEW.md)。

## 执行协议

- 默认四个 case：MATH/EFFICIENT × FP32/BF16；B=1、H=12、L=256、D=64、table bias、global mask、有 padding。
- 每个 case 使用全新 Python 子进程，整体逐个执行；MATH 是该子进程内的数学对照。精度/backend 变化不改变原始 FP32 随机输入；保存完整输入 SHA。
- 执行生命周期仅支持 Linux 主线程。`os.posix_spawn(setpgroup=0)` 为 case 创建同一 session 内的进程组，无 `preexec_fn`，兼容 Python 3.10/3.12；执行期间暂时启用 child subreaper，结束恢复原状态，只回收本 case PGID。导入和 dry run 不改变这些进程设置。
- 父 SIGTERM/SIGINT、超时、异常以及 leader 正常退出都会清理整个 case 组：先 TERM，必要时 KILL，再 wait 回收子/孙进程。清理期间的新终止信号延后处理，不会被丢弃。最多 2 秒清理时间从原 case timeout 中预留，小于现有池的 10 秒 TERM→KILL 宽限；清理失败直接中止计划，不继续下一 case。不能保证拦截父进程 SIGKILL，也不支持后代主动脱离本组。
- 始终只启用请求的一个 SDPA backend，不会捕获错误后重试 math。选择器诊断与实际 profiler 证据都保存；EFFICIENT 必须捕获其 forward/backward ATen 操作，且 profiler 必须记录 CUDA kernel 名。仅 enabled flag 或 can_use=True 不构成通过。
- 对比输出、Q/K/V 及原始可学习 bias 参数的全部梯度。浮点误差同时满足逐元素和 L2 条件才通过。FP32 `atol=2e-4, rtol=3e-3`；BF16 `atol=3e-3, rtol=8e-2`；L2 条件为 `norm(diff) <= atol + rtol*norm(reference)`。容忍值是本工具预设诊断标准，尚未经过本项目 CUDA 实测校准；失败后应审阅误差，不自动放宽。
- attention 的 Q/K/V 与 bias 显式转为指定 dtype，叶张量/参数保持 FP32，标量 loss 使用 FP32，TF32 关闭。它验证 FP32 参数与 BF16 attention 的数值路径，未模拟整个 HF/Accelerate forward 或 optimizer。
- correctness/profile 结束并丢弃相关张量后，默认 5 次 warmup、20 次独立计时。CUDA events 分别记录 forward/backward/总耗时；另记 wall time、allocated/reserved 峰值及相对输入基线的增量。
- forward 计时包含 dtype 转换、bias/mask 构造、attention 和标量 loss；没有模型投影、优化器或训练调度。当前模型逐 forward 复用多层 bias 的收益也没有被模拟。这里的结果不能替代全模型 profile 或直接预测训练加速。

[PyTorch profiler](https://docs.pytorch.org/docs/2.8/profiler.html)、[CUDA Event](https://docs.pytorch.org/docs/2.8/generated/torch.cuda.Event.html) 和 [backend API](https://docs.pytorch.org/docs/2.8/backends.html) 是所用官方接口依据。

## CPU 规划与检查

以下命令不执行 GPU、不查询 CUDA 可用性、不启动 worker；目标目录必须不存在：

```bash
source /vepfs-mlp2/c20250601/251105016/miniforge3/etc/profile.d/conda.sh
conda activate pllm
cd /vepfs-mlp2/c20250601/251105016/project/dllm_test
python scripts/architecture_prototypes/cuda_attention_probe.py \
  --dry-run \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/cuda_attention_plan_next
```

已执行的 dry run：[summary.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_validation_20260930/default_dry_run/summary.json)，状态明确为 `dry_run_not_executed`，四项均 `dry_run`。

CPU 测试：

```bash
python -m pytest -q scripts/tests/immune_llada/test_cuda_attention_probe_cpu.py \
  scripts/tests/immune_llada/test_cuda_attention_probe_lifecycle_cpu.py
```

历史生命周期版 **pllm/Python 3.10.19：59 passed（12.84 秒）；worker/Python 3.12.12：59 passed（11.34 秒）**；[pllm 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_lifecycle_validation_20260930/pllm_pytest.log)、[worker 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_lifecycle_validation_20260930/worker_pytest.log)、[历史验收清单与 SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_lifecycle_validation_20260930/validation_manifest.json)。

其中原 47 项涵盖配置边界、独立随机流、CPU FP32/BF16 math 梯度、mask 语义、单一 backend、不回退、证据结构、失败分类及计时协议；新增 [12 项生命周期测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_cuda_attention_probe_lifecycle_cpu.py) 包括真实 CPU 子/孙进程的正常/非零退出、超时、忽略 TERM、父 TERM/INT、Python 异常和清理期信号，验证 PID/PGID 消失、wait 回收及状态恢复。CUDA event/结果的 fixture 仍明确是 test double，不是 CUDA 证据。

初版 [47 passed 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_validation_20260930/pytest.log) 及旧 SHA 清单保留为历史。新增测试首轮收集因一个参数签名漏写而中止，修正后两环境均通过；初始收集错误日志保留在新验收目录，未覆盖旧证据。

2026-10-01 新增 5 项“worker 已写负结果但异常退出”的检查，原 probe 合计 64 项。每项现在保存真实 `child_process.returncode`、预期退出码、协议匹配与进程组清理证据；passed 必须退出 0，其他 worker 结果必须退出 2，不匹配转 `failed`。与 [typed adapter 68 项](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pool_attention_diagnostic.py) 联合，两环境各 132 项通过；当前 SHA 和日志以 [新验收清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/validation.json) 为准。adapter 只在独立新目录内实现固定 envelope/预算与结果完整性合同，尚未接入 pool；其“完整负诊断可退出 0”语义见 [adapter 说明](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TYPED_DIAGNOSTIC_ADAPTER.md)，不改变本工具 CLI 的退出语义。

## GPU 命令：仅留待当前任务结束后执行

由 root 先设置 `CUDA_VISIBLE_DEVICES` 指向已获分配且空闲的 GPU。下方 `--device-index 0` 指该可见设备列表中的第一个设备。工具自身不会申请或抢占 GPU。

最小四项对照，默认 5 warmup + 20 measured，单项上限 120 秒、总上限 600 秒：

```bash
python scripts/architecture_prototypes/cuda_attention_probe.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/cuda_attention_small_next \
  --device-index 0 --length 256 --batch-size 1 \
  --backends math,efficient --dtypes float32,bfloat16 \
  --masks global --bias-kinds table --padding \
  --case-timeout 120 --total-timeout 600
```

单 case 复核。它仍执行同精度的 MATH 数学对照，但只有 EFFICIENT 的独立计时，不能据此给出两者速度比：

```bash
python scripts/architecture_prototypes/cuda_attention_probe.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/cuda_attention_single_next \
  --device-index 0 --length 256 --batch-size 1 \
  --backends efficient --dtypes bfloat16 \
  --masks local --local-radius 32 --bias-kinds table --padding \
  --case-timeout 120 --total-timeout 150
```

基础对照通过后，可在新目录运行完整 24 项小矩阵：

```bash
python scripts/architecture_prototypes/cuda_attention_probe.py \
  --output-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/cuda_attention_masks_next \
  --device-index 0 --length 512 --batch-size 1 \
  --backends math,efficient --dtypes float32,bfloat16 \
  --masks global,chain,local --bias-kinds table,dense --padding \
  --case-timeout 120 --total-timeout 1800
```

可用 `--no-padding` 单独验证无 padding，或用 `--length 257` 检查非整齐长度。不同布局不支持 efficient 时应保留该失败记录，不能替换布局后仍沿用原 case 名称。

## 产物和失败语义

根目录保存 `plan.json`、`summary.json`；每项子目录保存 `spec.json`、`result.json`、实际执行时的 `worker.log`。不保存模型、checkpoint、输入张量、完整 profiler trace 或生成样本。

每项记录脚本与 oracle SHA、配置/seed、实际 torch/Python/平台/CUDA build/cuDNN/GPU/UUID、构造前 backend flags、bias shape/stride/dtype/requires_grad、kernel 名、梯度差异、计时和峰值。

| status | 含义 |
|---|---|
| `passed` | 实际 CUDA kernel、数值/梯度和完整计时协议均通过 |
| `unsupported` | 请求后端不能执行该配置，保留报错和失败阶段 |
| `oom` | CUDA 显存不足；`phase` 区分 math reference、候选或计时阶段 |
| `unavailable` | CUDA 不可用，没有转用 CPU 测试冒充通过 |
| `backend_mismatch` | kernel 不符合请求，或没有获得 CUDA kernel 证据（包括 profiler 环境问题） |
| `numerical_mismatch` | 输出或梯度未达到预设标准，不继续做速度结论 |
| `timeout` | 父进程超时终止该子进程 |
| `not_run_budget` | 总预算已耗尽，未启动该项 |
| `failed` | 其他错误、源码变化、结果身份/结构不符等 |
| `dry_run` | 仅规划，未执行 |

只有两个后端都 `passed`，输入 SHA 和环境一致时，`matched_comparisons` 才计算 `math median / efficient median` 和峰值差；其他情况为 `not_comparable`。整体退出码 0 仅用于 dry run 或全部通过，存在未通过项时为 2；应结合 JSON 判断具体原因。

默认单项最多 180 秒、总体最多 1200 秒；可设置单项 1–3600 秒、总体 1–7200 秒。最多 24 项，至少 3 warmup、20 measured，最多 20/100 次。单 attention 的 B×H×L×L 上限 `2**29`，即一份 FP32 dense tensor 2 GiB；实际峰值包括梯度和中间量，可能更高。这个上限限制诊断规模，不是可用显存保证。

下一阶段先阅读真实 CUDA 结果再决定后端候选；[现有容器执行入口审阅](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_EXISTING_POOL_DIAGNOSTIC_EXECUTION.md) 区分了现有 CLI、动态注册表扩展及 WebShell 条件。当前工具的 CPU 通过不授权修改或部署任何正在运行的 release。共享根 MD 由 root 同步。
