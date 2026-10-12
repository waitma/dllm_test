# 已分配八卡容器执行独立 attention probe：只读审阅与后续方案

2026-09-30；对象为 `t-20261001033327-fmnql`。本轮没有入队、进入容器执行命令、改 worker/39 个冻结文件、中断任务或申请资源。

**结论：本机 CLI 没有已确认的 task exec 入口；但现有 worker 可以通过“动态注册表 + 每任务新进程入口”的受控后续扩展执行一种专用诊断，并不必然需要重启 worker。该扩展尚未实施，必须等八个 500 步任务全部结束并冻结旧 release。**

2026-10-01 补充：独立 [typed adapter](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_TYPED_DIAGNOSTIC_ADAPTER.md) 已实现固定 envelope、四 case/120 秒/600 秒预算、完整证据绑定及明确退出映射；**未接入 worker/registry/稳定入口、未入队、未运行 GPU**。probe 同时补录子进程退出码，防止负结果 JSON 掩盖异常退出；probe 64 项 + adapter 68 项两环境联合验收见 [132 项记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/validation.json)。下文仍是后续接入条件，不代表当前容器已执行诊断；监督类型分支由 root 独立处理。

## 实际观察的命令能力

| 只读命令/检查 | 本机结果 |
|---|---|
| `volc version` | `/root/.volc/bin/volc` 1.2.52；没有升级 |
| `volc ml_task --help` | submit、sbatch、cancel、get、logs、list、top、upload、export、template、instance；无 exec |
| `volc ml_task instance --help` | 只有 list |
| `volc ml_task exec --help` / `volc ml_task instance exec --help` | 均为 `No help topic for 'exec'` |
| `/root/.volc/bin/mlp --help` / `job instance --help` | 委派二进制 commit `4d1dffc`；v2 job instance 只有 get/list |
| `volc ml_task instance list --id t-20261001033327-fmnql -o json` | 返回 `t-20261001033327-fmnql-worker-0`，Running，启动于 `2026-09-30T19:34:08Z` |
| ELF/Go 符号及 `sbatch --help` | 安装的是二进制而非可读源码；存在 Slurm `startShell/configShell` 符号，但公开 sbatch 接口是创建任务/集群，不能据此推断可 attach 此现有 Custom 任务 |

这证明的是**当前已安装 CLI 的公开能力**，不证明平台没有其他接口。官方文档确认平台存在 WebShell 容器排障入口；此任务的实际 WebShell 权限/入口尚未验证。不能把开发机 SSH 或新建 Slurm 集群当作当前训练容器的 exec。[官方自定义训练文档](https://www.volcengine.com/docs/6459/72350?lang=zh)

## 最小后续集成：沿用运行中的 worker

[现有 worker](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py:30) 的 `variants()` 每次 AST 读取磁盘上的注册表，`command()` 每次重新验证并启动新的稳定入口进程；并非所有分发逻辑都被常驻进程固定。它只接受 `variant/mode/steps/data_dir/resume_from_checkpoint`，不接受任意命令。新增原型或 JSON 本身不足以执行；以下是**待审阅实现方案**：

1. 八个 500 步任务全部完成后，保存完整运行依赖快照并核对旧 39 指纹；将旧 R20–R27 的 `release_map.json` 和后续入口明确绑定该快照。当前 R20 映射仍指向活动仓库，必须先处理，否则改注册表/稳定入口会污染旧证据。旧入口用新进程进入快照，避免已导入模块混用。
2. 只新增一个显式 `diagnostic_only` 注册项，例如 `d00_cuda_attention_probe_v1`，在稳定入口最早路由到独立 adapter。保留旧 worker 的运输格式，例如固定 `mode=profile-synthetic, steps=1`；这里的 1 是一次受限 probe 调用，**不是训练步数或 profile 通过**。专用 enqueue 校验和 adapter 都拒绝其它 mode/steps、data/resume、任意路径或命令扩展。
3. Adapter 固定随机张量小矩阵：1 GPU，MATH/EFFICIENT × FP32/BF16，B1/H12/L256/D64，global/table/padding，5 warmup + 20 measured，120 秒/case、600 秒总体。使用 worker 已隔离的 `CUDA_VISIBLE_DEVICES`，设备索引为 0。独立新目录保存原始 probe 结果、实际环境/源码 SHA 和 kernel/梯度证据；不写模型 profile、不训练、不保存 checkpoint。
4. 新 summary/COMPLETE 明确 `kind=cuda_attention_backend_probe`、`training_started=false`、`biological_quality_claim=false`，把 `execution_completed` 与 `all_cases_passed` 分开。已完整记录 unsupported/OOM/数值不符可视为诊断执行完成，但绝不能称 backend 通过；不完整、超时、源码变动或无有效结果必须失败。wrapper 的退出映射须有独立测试，保留原 probe 退出码和逐项状态。
5. 监控需先识别该 kind，显示 case 完成数和 backend 状态，跳过训练步数/学习信号/profile 晋级规则。现 [supervisor](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_supervision.py:195) 会把 `profile-synthetic` 当训练式 synthetic 并检查 global_step，未经类型扩展会误报；禁止伪造 global_step 消除报警。

还有一处必须处理：[现 init](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/immune_research_pool.py:273) 只排除 architecture_v2/refined，新增 diagnostic 项会被自动按 20 步入队。后续仓库版本应显式排除 diagnostic_only，或改用明确的 legacy 初始化清单；不得伪装成 architecture_refined 来绕过。此处修改是未来初始化路径的安全修正，**不要求替换当前内存中的 worker 循环**。若要求磁盘 worker 也完全不动，则这项初始化缺口不能声称已解决。

## 退出清理与其他路径

初版 probe 的每 case 另建 session，旧 pool 只终止外层 job 进程组，外层退出时存在 case 遗留风险。**现已仅在隔离原型修复，并在 Python 3.10/3.12 两环境各通过 59 项 CPU 测试**：`posix_spawn(setpgroup=0)` 同 session 分组，无 Torch 导入后的 `preexec_fn`；父 TERM/INT、超时和异常均清理组并 wait；临时 subreaper 只回收本 case PGID，结束恢复。真实 CPU 子/孙进程验证了 PID/PGID 消失、忽略 TERM 后 KILL 回收及父退出前清理。最多 2 秒清理从原 case 预算内预留，小于 pool 的 10 秒宽限；失败即中止后续 case。父 SIGKILL 无法普遍拦截，后代主动脱离进程组也不在保证范围内。[日志与源码 SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_cuda_probe_lifecycle_validation_20260930/validation_manifest.json)。**仍未接入 pool，尚无 CUDA 通过证据。**

直接 WebShell 是另一条可能路径：需要实际可访问的授权终端；在无待执行训练且 GPU 已空闲时运行 [既有有界命令](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/CUDA_ATTENTION_PROBE_README.md)。必须协调 worker 的后续 claim 和维护负载，不能只看到 GPU 瞬时空闲就另起进程竞争。当前 YAML 的 `DelayExitTimeSeconds=0`，也不能先停整个池再假定容器会保留。

若改成全新的 worker `kind=diagnostic` 协议，接口会更整洁，但当前常驻 worker 不会自动加载新的 Python 函数；仍需要已获授权的容器执行入口/受控交接。对于当前分配，优先评估上述单一注册项方案，不能把“必须重申请资源”当作既定前提。CPU 验收至少覆盖路由白名单、预算上限、旧 release SHA、禁止 init 自动入队、类型化完成语义及退出回收；这些通过后才安排真实 GPU 测试。
