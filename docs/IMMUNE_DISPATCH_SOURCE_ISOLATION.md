# 冻结入口与子进程的来源隔离（未部署）

2026-10-01。仅完成必要的软件隔离修正和 CPU 验收；不部署、不入队 D00、不运行 GPU。attention backend 探针保留为可选支线，本修正可供后续主线 Trainer 验收入口复用，不能视为该验收已经通过。

旧 dispatcher 的 `PYTHONPATH=快照根` 和 cwd 不能阻止 Python site 配置。只读查询发现 worker 的 `.pth` 加入两个 live-root 的 downstream 源码路径，`scripts` namespace 还合并了 `site-packages/scripts`。查询到的旧 runner/dllm 仍解析到冻结快照，未证明历史模型运行混入其它源码，不能据此宣布旧结果失效。原 38 项 sentinel 测试没有覆盖这个来源边界。

## 最小修正

[dispatcher](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_release_dispatcher.py)核验固定 [sitecustomize guard](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/release_startup/sitecustomize.py) SHA，仅在该 job 环境中设置固定 route、guard SHA 和 `PYTHONPATH`。exec 仍保留 PID/进程组，但先进入全新标准库 bootstrap，再执行已核验入口。bootstrap 检查本进程 guard 标记，并再次清理 CPython 在 site 之后插入的 live dispatcher 目录。

guard 在首次项目 import 前拒绝已预加载的 `scripts/examples/dllm`；移除未选中的 live-root 搜索路径，将所选 root 放到首位；固定 `scripts/examples` namespace 为所选根的目录，`dllm` 保持正常包导入。关闭 user site 和字节码写入，不改安装环境。标准库检查中的任何异常使用 `os._exit(78)` 结束，防止 Python 吞掉普通 sitecustomize 异常后继续执行目标；顶层未加载 guard 或来源变动另由 bootstrap 拒绝。三个稳定入口拟改动仍只在 [stage_v2 补丁](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_v2/routing.patch)中：最早入口改为标准库 `runpy` 从固定文件加载 dispatcher，避免它自己先被外部同名包解析。28 个模型 variant 与一个 d00 注册项保持原计划。

冻结 R00/R12/R20 profile 使用普通 `sys.executable` 子进程且不覆盖环境；d00 case 的 `posix_spawn` 复制 `os.environ`。这些已审计的启动链路都会再次运行 guard；无需修改冻结 runner、39 个测量源码或 d00 的三份测量源码。**此合同不覆盖 `-I/-E/-S`、子进程替换环境、后续任意代码改写 import 状态，或启动文件被并发删除/破坏；它不是通用沙箱，也不阻止在 guard 之前执行的任意恶意 `.pth` 代码。** 发布时须冻结启动源码并保留来源核验，不能把一次 marker 视为通用隔离证明。

## 实际验收

两环境各 **48 passed**：pllm/Python 3.10 为 3.53 秒；worker/Python 3.12 为 3.85 秒。[测试](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pool_release_dispatcher.py)保留原 38 项并新增 10 项：临时 venv 中真实 `.pth`、外部 regular 同名包、三家族及 diagnostic CPU transport sentinel，实际执行父进程与 `subprocess`/`posix_spawn` 子进程，核对不同 PID、实际模块文件、namespace 和搜索路径；验证三种预加载项目包、错误 guard SHA/route 均在目标前失败。d00 的 sentinel 只证明相同运输方式的继承隔离，没有执行 adapter、Torch 或 CUDA。

另对真实历史目录进行两环境纯标准库查询及 manifest 校验，未执行模型入口：R00 仅有 32 个 measured SHA（历史报告复制 465 文件，不能补称全部已认证）；R12 核验 35 measured/474 copied SHA；新 R20 核验 39 measured/524 copied SHA。三个根的实际 guard namespace 与 dllm 解析位置均在所选快照。[只读证据](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/guarded_historical_resolution.json)、[pllm 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/pllm_guard_pytest.log)、[worker 日志](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/worker_guard_pytest.log)。

旧 `stage/`、旧验收和历史 SHA 保留；新 [stage_v2 manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/stage_v2/deployment_manifest.json)额外绑定 guard，三个 live 旧源码未应用补丁。新 [验收清单](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/guard_validation_v2.json)记录最终 SHA、边界及仅本轮两个显式临时测试目录的清理；未生成 checkpoint。主线 Trainer 的完整窗口、四档评估与显存仍须独立验收。
