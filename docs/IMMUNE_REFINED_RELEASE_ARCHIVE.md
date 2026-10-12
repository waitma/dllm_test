# R20–R27 旧 release 归档准备

日期：2026-10-01 UTC。状态：隔离 helper、人工 CPU 测试与真实只读 dry-run 完成；**尚未归档、切换映射或改动活动源码**。固定未来目录为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001`，当前不存在。

实现：[archive_refined_release.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/archive_refined_release.py)。测试：[test_archive_refined_release.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/tests/immune_llada/test_archive_refined_release.py)。只操作运行所需源码、任务元数据和原有合成诊断证据，不运行 GPU、模型、训练或读取生物语料。

## 实际依赖检查

两套已安装环境分别使用干净子进程，仅导入 `scripts.run_immune_refined`，禁用 CUDA 初始化，不构造模型或运行前向。pllm 使用 Transformers 4.57.6，worker 对应环境使用 4.48.1。两环境观测到相同的 45 个仓库 Python 模块，文件 SHA 映射完全一致；原 profile 绑定的 39 项映射也一致。

其中 9 个导入模块不在历史 39 项内，但均与既有 R12 快照字节一致：

- `dllm/__init__.py`、`dllm/pipelines/__init__.py`、`dllm/pipelines/immune_llada/__init__.py`。
- `dllm/pipelines/llada/__init__.py`、`dllm/pipelines/llada/models/__init__.py`。
- `configuration_llada.py`、`configuration_lladamoe.py`、`modeling_lladamoe.py`，位于 `dllm/pipelines/llada/models/`。
- `dllm/pipelines/qwen3_vl_arch/__init__.py`。

仅复制 39 项会遗漏上述依赖。helper 保留 R12 快照完整支持文件集合，补充当前 `dllm`、`examples`、`scripts` 下全部 Python 文件，以及 `dllm` 下 JSON 配置，因此覆盖这些目录中的延迟导入代码；启动导入观测用于补充核对，不冒充全部动态执行路径追踪。

本次 dry-run 计划共 520 个支持文件、4,887,028 字节。39 项外的 481 个文件相对 R12 快照有 431 个同字节、6 个变化、44 个未记录。变化的 6 项为 `immune_research_pool.py`、`immune_research_supervision.py`、`monitor_immune_research_pool.py`，以及对应目录下 `test_research_pool_monitor.py`、`test_research_refined.py`、`test_research_supervision.py`。这些变化均不在本次观测到的 R20 runner 导入闭包内；完整逐项 SHA 和新增文件清单保留在计划的 `dependency_audit.all_unmeasured_support`。

**本次补充哈希不等于追溯补做 GPU 认证。** 与 R12 一致只能证明两个保存时点字节相同，不额外保证历史进程使用了未经记录的依赖。外部包记录当前解释器实际版本，不进行 vendoring，也不声称冻结了完整容器环境。若未被 profile 覆盖的实际导入模块相对 R12 发生变化或没有旧基线，helper 会阻止归档，要求先明确处理这个证据缺口。

## 写入门禁与复制合同

默认 `archive_release(request, apply=False)` 仅返回报告。`--apply` 会重新执行检查，不能把旧 dry-run 当作当前准入证明：

1. 任意 pending/running 任务都会阻止快照写入。当前 family 每个变体必须恰有一个原始 500 步终态结果；缺项、重复、失败或不完整均拒绝。
2. 八份原 profile 经已有验证器验收，完整 source map、生产配置、显存档位和 summary SHA 必须一致。八项诊断必须 succeeded/returncode 0，原 `COMPLETE`、summary SHA、变体、500 步曲线及现有完整诊断验证器均通过。
3. 归档使用原诊断验证器的 `require_learning=False`：保存旧实验不等于批准后续训练，不以某个方法学习效果好坏决定是否保存历史；原始证据完整性仍必须通过。
4. 先验证 R12 的完整 copied-file SHA，再规划当前支持树。必须同时覆盖所有 measured 文件、实际导入文件及明确列出的 R20 runner、父 runner、expanded fixtures、配置和词表依赖。
5. 正式复制前取得已有 architecture/refined 入队助手锁，再次检查全部计划。复制到独立临时目录，逐文件校验；复制后重新检查队列、源码和原证据，验证临时快照，再发布到不存在的固定目录。失败清除本工具创建的 staging，不覆盖旧快照。

锁与已有入队助手协作，不能阻止绕过它们的外部手动队列写入；发布前再次检查活动任务。helper 不修改 `release_map.json`、公共入口、注册表、模型、旧 39 源或已有快照。后续映射/入口切换仍由独立部署步骤在同样终态前提下执行。

`release_manifest.json` 保持 R12 的 schema 1 和 `source_root`、`frozen_root`、`measured_source_sha256`、`copied_files_sha256`，另保存原证据 SHA、八项任务 ID 和依赖差异。固定入口 `scripts/run_immune_refined.py` 必须在两张 SHA map 中。已与 [pool_release_dispatcher.py](/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/architecture_prototypes/pool_release_dispatcher.py) 双向只读复核，字段和路径兼容；该 dispatcher 目前也只是隔离待部署原型。

## 本轮验收

人工临时文件/任务测试：pllm 22 项通过（0.49 秒），worker 对应 Python 环境 22 项通过（0.59 秒）。生产 profile/诊断接口和 import inventory 在人工测试里明确 mock；队列状态、原标志文件 SHA、复制前后漂移检查、复制及快照 SHA 校验实际执行。覆盖活动任务、不完整结果、错误摘要、来源变化、旧快照损坏、重复结果、复制期间新任务/文件漂移、不可覆盖和快照缺文件。

两环境真实 dry-run 均 `ready=false`：八项 500 步任务仍在 running，终态结果为零；八份 profile 验收通过。两个解释器的实际本地导入 SHA 相同，外部包版本差异单独记录。固定目标目录未创建，未执行 `--apply`。

人工测试产生的临时仓库和快照在验收后删除。小型证据保留于 [validation.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_archive_20261001/validation.json)、[pllm dry-run](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_archive_20261001/dry_run_plan.json) 和 [worker dry-run](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_refined_archive_20261001/worker_dry_run_plan.json)。后续源码继续新增或编辑时，文件数量与 SHA 可能变化，实际归档前必须重新规划。

只读复查命令（激活 pllm 后）：

```bash
USE_TF=0 PYTHONDONTWRITEBYTECODE=1 python scripts/architecture_prototypes/archive_refined_release.py \
  --pool-dir /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_pool_c20250601_20260930
```

## 实际归档：2026-10-01 01:29:56 UTC

当前八项500步全部终态且pending/running=0后，原helper的apply完成；正式重新规划为524文件，取代先前520文件dry-run。根端独立verify_snapshot核验全部copied SHA及39项measured子集；8份终态来源完整。[正式manifest](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_research_releases/r20_r27_20261001/release_manifest.json) SHA=46399204c7018d00726e3ae08142bd4eb28475c8a9a2991b3a5c90725c8824f7。

01:31:05Z原子切换8项R20–R27 release_map，保留其它20项；[前后字节和SHA记录](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/immune_diagnostic_dispatch_20261001/deployment/archive_and_map.json)。监督随后独立重新inspect，36/36历史记录matched、无alerts。此次归档包含修正后的TokenValidationTrainer以及已冻结runtime package/inspector源码，额外支持文件不被追认成历史GPU profile。

live路由仍未部署：独立只读审查确认PYTHONPATH/cwd不能单独阻止worker的.pth和同名namespace合并，正在修正入口及实际继承子进程路径。没有证据表明已解析的历史模型模块逃逸；不因此重写旧实验或重新计数。
