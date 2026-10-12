# Formal 冷入口接续容量 v2 委派链

## 2026-10-03T03:41:39.451832+00:00 根端正式审阅与独立冻结完成，仅准备状态

根端已读取实际源码、三项控制完整stdout/receipt和归档SHA，前后原件一致，未重复执行。审阅 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/entry_v2_root_review.json（SHAae9a7f7a2cad1a4c3ffd92834b1947453dfc31182f76f62c615ac48486d327ff）；两源冻结清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v2.json（SHAe92de529db8b77f5c0e083ff115186c1419fbfc6c9a53ba61b386d5fde1bb9df）。旧formal wrapper、运输、v3/611 source与parent_gate_v3不改，当前public仍large-batch v2。正式selection/plan/catalog与八票据未创建/入队，真实联合更新和恢复未验收。首次冷控制证明变动接线，不能替代实际world8。新组真实进度 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_RUNTIME_WATCH_20261003.md，预算 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_RUNTIME_RESOURCE_BUDGET_20261003.md。

2026-10-03T03:25:08Z：新增薄入口和唯一三项冷控制完成，仅源码准备。未发布 public/registry、未创建正式 selection/plan/catalog、未入队或启动任何模型/GPU。本项不修改冻结文件、数据、证书或原门禁；全局 PROJECT/BIO/GUIDE/CURRENT 文档由根端同步。

## 最小实现

新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_formal_v2_dispatch.py`，接口沿用 `prepare(original_argv)` 和 `main(original_argv)`。

它通过 SHA-bound runpy 冷加载 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_launcher_source_v1/pool_formal_dispatch.py`，实际 SHA256 为 `21f2ffe71d23950207eb4c067b0cb26cf5dbb875a13e2ba15bdec3a7d3f5ce59`。在独立内存 namespace 内仅替换 `PREVIOUS/PREVIOUS_SHA`；原正式 prepare/main 函数直接复用，未复制或改写 ticket、source/data、模型选择、resume、八slot、NCCL、checkpoint 的既有接续检查。这里的“复用”不代表本轮重验这些旧流程。

未知 alias 的新 fallback 为 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2/scripts/research/pool_large_batch_ablation_v2_dispatch.py`，SHA256 `b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3`。同时只读核验对应 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2.json`，SHA256 `4cd5f793e74f56730a0ca4468cebb2783fe2de22b33c7325f664f54634db43aa`，要求 schema、snapshot root 和入口文件SHA一致。

因此未来正式入口接入后，未知 alias 保持原 argv 进入 large-batch v2 → capacity v2 → 原 capacity/p02 链，避免切回要求旧容量协议的 large-batch v1。此处不执行各下游 main，也不为正式训练创建准入证据。

## 首次三项冷控制

激活pllm后执行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_formal_v2_dispatch.py`，唯一一轮三项通过，控制耗时0.0184秒，exit0：

1. 实际冻结 base/fallback/manifest SHA匹配，未知 alias 的原始 argv 原样委派；下游 main mocked。
2. 自有 TemporaryDirectory 内修改源码/manifest字节，SHA漂移在委派前拒绝。
3. 原 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_eight_gpu_20261002/pool_job_catalog.json` 仍不存在；原 formal prepare 对人工票据直接拒绝，未创建计划或输出目录。

测试禁止ML导入、Popen、socket和exec，不读取序列或数据分片；临时fixture已清理，真实原件SHA前后相同。没有重复旧7项、CPU/Gloo/GPU或611来源门禁测试，也不构造checkpoint或数据证书。

原件目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_v2_dispatch_controls_20261003`：`receipt.json` 记录首次执行与完整来源SHA，`stdout.log` 保存实际stdout/stderr，`source` 归档本轮两个Python文件。

| 文件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_formal_v2_dispatch.py` | `eeef5b8656e66c636e6f5873dcb6c89a7ed0f30e0edb15c18fe03c8b0d529a78` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_formal_v2_dispatch.py` | `2b8b411cd928fc4cbf814b2953e394f8de50c72647b394971be19b5ca16b498b` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_formal_v2_dispatch_controls_20261003/stdout.log` | `d38a1c298aa2fdd234b594baa9a092a33131865144f7bf4a991a3ff6f9731a04` |

待根端独立审阅/冻结；正式训练仍须真实选型及全部八卡空闲准入，本轮冷检查不替代真实联合训练/恢复验收。停止的序列注释分支、独立CUDA专项暂停、全部原审核/ready/选择/来源/恢复门禁及禁止重启、重复提交约束保持。
