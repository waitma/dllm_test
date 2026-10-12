# 诊断入口与冻结 release 分发：待部署补丁

2026-10-01。本轮仅新增标准库 dispatcher、补丁生成器与 CPU 测试。**活跃入口、registry、worker 和训练 39 个冻结文件未修改；未提交、入队、执行 GPU 或读取真实序列。** 根 MD、研究索引和计划 YAML 由 root 同步。

实现：[pool_release_dispatcher.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_release_dispatcher.py)；生成器：[stage_pool_diagnostic_routing.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/stage_pool_diagnostic_routing.py)；[测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pool_release_dispatcher.py)。

## 固定路由

dispatcher 只导入 Python 标准库，接受固定的 28 个模型 variant 和唯一诊断 `d00_cuda_attention_probe_v1`。不接受目标脚本、release 目录、shell 或可执行程序覆盖。重复 `--variant`、未知 variant 和缩写拒绝。模型参数原样交给既有冻结 runner 解析。

| 家族 | 固定快照目录名 | 新进程入口 |
|---|---|---|
| R00–R11，12 项 | `r00_r11_20260930` | `scripts/run_immune_research.py` |
| R12–R19，8 项 | `r12_r19_20260930` | `scripts/run_immune_architecture.py` |
| R20–R27，8 项 | `r20_r27_20261001`，仍须独立归档 | `scripts/run_immune_refined.py` |
| d00，1 项 | 当前项目中已验收的三个隔离源码 SHA | `scripts/architecture_prototypes/pool_attention_diagnostic.py` |

模型快照均位于项目的固定 `output/immune_research_releases/` 下。分发前校验 manifest、入口和已记录文件哈希；`os.execve` 保持池拥有的 PID/进程组，但更换整个 Python 进程映像，cwd/PYTHONPATH 设为选中快照，避免先导入 live 模型再混用冻结源码。

d00 在任何 Torch/adapter 导入前，仅允许五个唯一参数：variant、mode、steps、profile-dir、output-dir。mode/steps 固定 `profile-synthetic/1`，路径必须符合新建 pool/runs 与对应 profiles 的合同；拒绝 data、resume、预算和隐藏 worker 参数。再次核对 [132 项 CPU 验收版 adapter/probe/oracle](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_typed_attention_adapter_20260930/validation.json) 的固定 SHA，然后进入既有 adapter。它仍只运行固定四项、5 warmup/20 measured、120 秒/case、600 秒总体；本轮没有修改其实现或数值条件。

## 可审阅补丁

[routing.patch](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage/routing.patch) 尚未应用；[deployment_manifest.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage/deployment_manifest.json) 记录三个旧文件前后 SHA、patch SHA、新 dispatcher SHA、前置条件和 28+1 注册项数量。拟修改内容保存在同目录 `proposed/`，仅供审阅。

1. 稳定入口在 `import torch` 之前、仅 `__main__` 路径调用 dispatcher，移除原末尾先导入 live 模型后路由的分支。作为模块导入的旧函数仍保留。
2. registry 原 28 项及其 options 完全保留，只增加 `diagnostic_only=true` 的 d00。`ResearchModel.__new__` 在正常模型构造前拒绝诊断 variant。
3. 未来磁盘 worker 校验重复 registry key，并严格限制 d00 envelope；`init` 显式排除 diagnostic_only，保持原 12 个 legacy 初始化任务。

当前内存 worker 的 Python 函数不会被该补丁热替换。它会动态读取新 registry，旧 validate 可能允许 d00 的错误 mode/steps，但新早期 dispatcher 与既有 adapter 再次拒绝这些参数。本轮使用实际旧 validate 函数、临时新 registry 和真实子进程验证了这一点。

生成器只输出补丁，不提供 apply 行为；若旧文件 SHA 已变，它会拒绝生成。实际部署必须由 root 在八项 500 步任务全部终态、旧证据核验、R20 独立归档完成、release_map 重新绑定快照之后进行。还必须保留已独立处理的 typed 监控分支。当前 R20 快照缺失时 dispatcher 会明确拒绝该路由，不回退 live 源码。

## 历史快照的实际只读验证

独立于 sentinel 测试，对真实历史目录调用 `resolve_route`，未 exec 入口，也未导入 Torch。结果：[historical_snapshot_resolution.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/historical_snapshot_resolution.json)。

| 真实历史快照 | measured | manifest 报告复制数量 | 有记录并实际核验的 copied SHA |
|---|---:|---:|---:|
| R00–R11 | 32 | 465 | 旧格式没有完整 copied map；仅核验 32 个 measured |
| R12–R19 | 35 | 474 | 474；35 个 measured 与 copied map 一致 |

R00 旧 manifest 为 `immutable_legacy_source`，没有 `frozen_root` 或完整复制文件 SHA 表。dispatcher 对此有明确的旧格式分支，不能宣称 465 个文件全部已获历史哈希认证。R12 和未来 R20 使用 schema 1，核验完整 copied map 与 measured 子集；支持文件哈希也不构成补做 GPU 认证。

## 验收与未解决边界

两环境各 **38 passed**：pllm/Python 3.10 为 1.84 秒，worker/Python 3.12 为 1.96 秒。测试在临时仓库副本实际执行 `git apply --check` 和 `git apply`，验证三个家族通过冻结目录中的无害标准库 sentinel 新进程运行、28 项 options 不变、诊断不能构造模型、不自动初始化入队、重复/未知 variant、坏 manifest/哈希/路径、旧 worker 非法 envelope 及固定四项诊断来源。sentinel 不代表真实模型或 CUDA 执行。

[验收清单及源码 SHA](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/validation.json)；[pllm 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/pllm_pytest.log)；[worker 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/worker_pytest.log)。只清理本轮自有临时 repo，保留小型补丁、拟源码和证据；没有生成 checkpoint。

2026-10-01 补充清理：确认默认临时目录 `/vepfs-mlp2/c20250601/251105016/conda/cache/tmp/pytest-of-root/pytest-440` 属于本轮首批分发器测试后，删除其中 337 个文件（4,367,949 字节）。29 个真实子目录名称均匹配本测试模块；每份 stage manifest 的旧/新源码 SHA、patch SHA 和初版 dispatcher SHA 均核验一致，且无 pytest lock。归属证据及删除后不存在的状态写入上述验收清单。未操作其它默认 pytest 目录，未重跑测试或修改 staged patch。

**未来 certified pilot 的来源绑定仍未解决。** 它当前导入 live 39 源码；应用入口/registry 补丁后，其 live fingerprints 会与旧 profile 不一致，原门禁应当拒绝。此 dispatcher 只处理既有模型 CLI 和 d00，不新增 pilot 训练路由、不覆写 profile、不绕过数据或 GPU release 门禁。后续须独立设计从已冻结 release 构造 pilot，或取得对应新 release 的 profile/GPU 验收；当前不实施。
