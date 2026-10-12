# 独立 pilot 源码运行包准备

更新：2026-10-01 04:07 UTC。v1已创建并因阶段时限失败保留。实际v2位于r20_r27_pilot_v2_20261001，manifest3805679f1d7370042cd39741d4839ad1c6acc3805be899bed3b1f95872dace12；526文件/2别名、524父字节及两环境profile/54模块导入检查通过，GPU7验收运行中。包标志仍source-only false，不表示数据/GPU通过。[v2构建与启动](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_PILOT_RUNTIME_BUDGET_V2.md)。下文26项/rehearsal/v1目录为准备历史；当前live builder固定v2，继承父verifier字节不变。

实现：[pilot_runtime_package.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_package.py)、[pilot_runtime_inspect.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pilot_runtime_inspect.py)。测试：[test_pilot_runtime_package.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_pilot_runtime_package.py)。本轮包工具没有修改模型39源、旧快照、v1适配器、registry、路由或队列。

## 需要解决的问题

活动入口和 registry 变化后，直接从活动仓库导入 pilot，会让现有 `fingerprints()` 与旧 profile 不一致。单独把 profile 路径指回旧目录无法恢复旧源码；在已导入活动模块的 Python 进程中临时改 `sys.path` 也不足以保证隔离。

采用完整、独立源码根：从已验收 R20 archive 复制所有 `copied_files_sha256` 文件，模型39项、支持依赖、适配器及指标代码全部保留父快照字节。旧 profile/500 诊断 map 使用相对路径，其字节不变；runner 和 adapter 的 `ROOT` 由 `__file__` 自然指向新包。profile、pool、人工 fixture、selection、prepared data 仍使用原绝对路径。

这里不复制完整 executor，不 patch 其来源函数。根线程另行修正未部署指标 Trainer 的 GA 标志并保存修正前精确源码与报告；此运行包只继承最终 parent manifest，不硬编码修正前的指标 SHA。相关语义验收见 [IMMUNE_HF_MICRO_MEAN_FIX.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/IMMUNE_HF_MICRO_MEAN_FIX.md)。

## 窄接口和完整性

`plan_runtime(base_snapshot, destination, additions={})`、`build_runtime(...)`、`verify_runtime(directory)`；没有训练 CLI 或提交入口。

父目录固定为 `r20_r27_20261001`，先验证完整 archive、八项 terminal job IDs 和原证据 SHA。实际父快照尚不存在时只返回 blocked，不创建 runtime。目标固定为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtimes/r20_r27_pilot_v1_20261001`；已存在则拒绝覆盖。

完整复制父快照的全部支持文件，包括原 adapter。additions 只允许四个固定新脚本名：`pilot_runtime_package.py`、`pilot_runtime_inspect.py`、以后明确实现的 `pilot_runtime_entry_v1.py`、`pilot_gpu_release_producer_v1.py`，都位于 `scripts/architecture_prototypes/`。任何父快照已有路径都禁止覆盖，即使该路径本身在白名单；模型39项自然也不能覆盖。新文件必须来自原 `source_root` 下同名路径。

`pilot_runtime_manifest.json` 绑定父 manifest SHA、完整父文件 map、39项 map、新增文件 map、最终全部源码 map和资源别名；状态严格为 `source_package_only_not_gpu_or_data_acceptance`。三个 acceptance/training 标志必须为 false。未登记源码、额外/替换链接、父证据变化、复制时漂移和文件损坏均拒绝。临时目录校验后发布，失败清理本工具 staging。

## 数据门禁路径缺口及拟处理

进一步只读检查发现：原 `tcr_scope_gate` 延迟调用 `audit_tcr_exposure.input_specs()`，后者按自身 `ROOT` 查找审计输入。纯源码副本缺少这些资源；未来即使 prepared data 仍用原绝对路径，scope gate 也会因路径缺失拒绝。这是兼容性缺口，不能宣称仅复制源码已通过真实数据准入。

合同只允许两处固定外部资源目录别名，绝对目标从父快照原 `source_root` 推导：

| runtime 内路径 | 绝对目标 |
| --- | --- |
| `downstream/benchmark/data` | 原 source_root 下的同名目录 |
| `output/research/uptcr_20260930/released_tables/data/finetune/abpm` | 原 source_root 下的同名目录 |

禁止额外、替换或与源码重叠的别名，缺目标目录即拒绝。symlink **不天然只读**；本合同只是保留原资源位置，不授予数据认证。原 gate 仍对每个输入的 `.resolve()` 路径及内容 SHA 校验，继续使用原证书；不会生成或替代证书。

本轮只用临时人工空目录/空文件验证路径解析和枚举，不创建上述实际资源链接，不读真实序列，不运行 data/scope gate。prepared candidate 的认证状态不由源码包改变。

## fresh Python 隔离实测与修复

`inspect_fresh()` 只能选择两个已安装解释器或当前解释器，调用固定 inspection 脚本；不接受任意命令、模块或训练模式。子进程使用 `-I`、runtime cwd、CUDA 隐藏及禁止初始化，模型构造也被检查入口禁止。

真实 worker 环境揭示仅 `-I`、cwd 和 PYTHONPATH 不够。`.pth` 会引入原项目中的 p-IgGen/LICHEN 子目录；移除这些路径后，`site-packages/scripts` 仍会与项目的 `scripts` namespace 合并。初次检查因此拒绝，没有将 namespace 外泄结果判成通过。

最终 inspection bootstrap 在第一次项目 import 前拒绝预加载项目模块、清除原仓库搜索路径，并通过标准 `ModuleSpec` 将 `scripts`、`examples` namespace 的搜索目录固定为 runtime 内目录，保留外部依赖正常加载。随后逐项验证所有已导入项目模块文件 SHA、模块 `ROOT` 和 namespace 路径，并在检查结束再次核验源码。

这只证明本次 fixed inspection 新进程；它不自动保证其他旧 runner/probe/dataloader 子进程的启动隔离。未来生产 wrapper/producer 与旧 runner 子进程须分别采用并验证相应启动机制，不能从本次检查推断全子进程已覆盖。

## 验收范围

人工文件系统测试 **26 passed，0.30 秒**：缺父快照、不可覆盖、完整父模型/adapter字节、固定新增脚本、live树变化不影响父副本、别名解析/额外别名拒绝、源码及证据漂移、复制中断清理、不能冒充数据认证，以及 `.pth`/namespace 隔离。

另把真实源码支持树复制到明确标记为 rehearsal 的临时目录，**没有生产 runtime manifest，没有实际资源别名**。pllm（Transformers 4.57.6）和 worker 对应环境（4.48.1）都在新进程内运行原 `verify_profile_evidence`：各 **8/8 profile 通过**；39项 source map一致，54个已导入项目模块和所有项目 namespace 均位于临时源码副本。adapter/executor 的来源 map 也全部属于同一根目录。本轮不执行 `verify_diagnostic`、模型、GPU、数据门禁或训练。

准备开始时实际父快照缺失，blocked 行为已有证据。随后根线程在全部500步任务终态后另行完成 R20 archive：524文件、39 measured；parent manifest SHA `46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7`。此事不等于本轮已创建 pilot runtime。三个包工具源码已冻结并被该父快照收入。

归档后只运行一次实际 `plan_runtime`：`ready=true`、`blockers=[]`，524项父源码与此前两环境 rehearsal 的完整 map 全部相同，39项 measured map 也相同，目标目录不存在。只核验源码和证据字节以及两个资源目录是否存在，没有创建目录或链接、没有读取数据内容。[实际只读规划](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/actual_parent_plan.json) 中的 `ready` 仅表示源码包可构建；新固定 entry/producer 尚未实现，GPU和真实数据准入仍未满足。

证据：[validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_pilot_runtime_package_20261001/validation.json)，包含测试、最终两环境 profile/import 报告、初次 namespace 拒绝和修复记录、完整 rehearsal 源码 SHA、清理记录。人工临时仓库、资源链接和源码副本验收后全部删除。

后续仍需正式 runtime 构建后的独立检查、完整新 GPU release、原双数据证书及精确 selection 验收。未来 GPU 报告可额外绑定 runtime manifest 和 producer 来源；独立 wrapper需严格核验这些扩展字段，再调用原 executor。原 executor 保存的整份 GPU summary/COMPLETE SHA会继续绑定到 checkpoint 侧车；这些后续工作本轮均未执行。
