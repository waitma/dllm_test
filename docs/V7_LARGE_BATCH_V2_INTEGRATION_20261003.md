# 容量 v2 与完整消融训练入口整合

## 2026-10-03T03:14:48.193324+00:00 最新真实状态：完整22项已登记，首八项已在原八卡运行

2026-10-03T03:10:52.992945+00:00为8 running/14 pending，首批R20/R24/R26及组合各两seed；统一micro32×GA1/global32、eval4、原120k/3750完整epoch。实际计划与固定账本已存在，尚无新组训练结论或完整图。发布/登记/初始原件依次 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/publication.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/enqueue_receipt.json、/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/deployment_v2/initial_observation.json；新启动源清单 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v2.json，原568模型不改。有限只读监督的后续记录 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_LARGE_BATCH_RUNTIME_WATCH_20261003.md。正式八卡联合训练/恢复未启动，原Goal active、数据/来源/恢复门禁与全部停线保留。以下各带日期段是历史状态，当前以本节、PROJECT顶部和CURRENT为准。

2026-10-03T02:50:14Z：新增入口的源码和首次必要冷控制完成，未发布 public/registry、未入队或运行训练/GPU。本项只读取根端已完成容量 v2 的真实原件以及新建计划/完整22项账本。旧私有 dispatcher、568份模型源码及原 runner 保持原字节。

## 复用和变更边界

新增 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_ablation_v2_dispatch.py` 通过 SHA-bound runpy 冷读取私有 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/launcher_source_v1/pool_large_batch_ablation_dispatch.py`（SHA256 `dc3408ff7a8e74b7944b5ceb17ca18bdaf0baf9a226d270c9cb12fe99058aa0f`）。在独立内存 namespace 中只替换 `check_capacity`、`PREVIOUS/PREVIOUS_SHA` 与 `CAPACITY_JOB`。旧 registry/catalog、568源及两项CPU证据核验、data路径/SHA、原runner与CLI/预算解析均复用原函数，未改写旧文件。

新容量证据绑定 job `1790992416352344471_p05_v7_capacity_v2_a4854542`：36次总更新、九格、每格两次warmup和两次测量、单格240秒/总1800秒/hard100。所有实际 passed 格必须有4次更新、2个有限loss、optimizer状态和真实吞吐；selected micro需在三个代表架构全部passed，allocated和reserved各不超过总显存85%。原 A10080、全长2048、原模型配置/来源、AdamW/BF16/whole-layer、返回码/COMPLETE flags 等检查保持。

容量的实际 ledger/summary/COMPLETE/pool result 路径及 SHA 必须出现在计划 profile_evidence。每格原 `measurement.json` 也必须有其SHA引用且内容与summary中该格完全一致。根端原终态审核 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_v2_root_terminal_review.json` 固定 SHA256 `b6ae90637728f22bef378e71d0d2c943e50ee74e4ad52165b1f0a66d95970c5b`；核对其v2协议、job、40份已加载源检查范围以及四原件/九measurement引用与计划一致。

新固定 intent SHA及旧失败四refs的语义核验，直接复用私有 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/capacity_launcher_source_v2/scripts/research/pool_large_batch_capacity_v2_dispatch.py` 的只读 helper；不会调用其已终态容量任务的 prepare/main。未知 alias 保留原 argv，委派同一私有 capacity-v2 dispatcher（SHA256 `998dc1c8f6debbaf1b811a0bb30bebbdd2b450749653f6c92ac5b7f6b02a44d6`），其原 capacity→p02 链不变。

## 实际计划与接口

`prepare(original_pool_argv)` 返回原 frozen runner 的命令，不创建输出或进程；`main` 仅供以后根端发布后由原pool调用，要求单mask0..7并execv保持PID。alias仍为原22个p04_v7架构/seed42或43，不增加新设计条件。

根端真实计划 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/batch_selection_v1.json` SHA256 `42be6e8e7c7ebbe578de16646cd6339c963390d2ffae5ed1226980dea46fbc04`，包含16项真实引用。完整账本 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_ablation_20261002/jobs.json` SHA256 `987394f0e0a539a0ad5e0057aa0952474eecb50f374e8ea1410f9f3168ea746d`，已预分配22个唯一job ID。统一micro32/GA1/global32、eval micro4、3750步、原120k子集；原568源码/样本SHA不变。registry未来必须一次绑定全部22条相同catalog/plan/source SHA；当前缺少未发布registry时仍拒绝。

容量实测只有三个代表架构，不能宣称其他八个方法已直接做过容量测量；synthetic结果不证明实际语料学习质量或生物效果。本入口不修改任何数据、审核或安全门禁。

## 唯一新增冷控制

激活pllm后执行 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_ablation_v2_dispatch.py`，四组控制通过，1.317秒，exit0。实际计划/账本/容量原件不使用假件替代；只把尚未发布的registry替换为自有TemporaryDirectory中的literal fixture。

首次prepare真实核验568源码和新check_capacity；剩余21项参数控制复用已经核验的源码/容量结果，只检查各alias/seed/output/统一32×1日程命令及原runner解析，不重复旧source suite。八种负向边界只改内存副本：旧72/180协议、旧8更新/6loss、选中档OOM、15%余量不足、缺少原measurement、原measurement与summary不一致、错误intent SHA、错误旧失败引用。另验证未发布registry拒绝、unknown argv向SHA-bound私有capacity-v2原样委派（下游main mocked）。

全部冷控制禁止ML导入、Popen、socket与exec；没有模型/真实序列读取/GPU/队列操作。TemporaryDirectory已清理，22个真实run目录仍不存在，公共registry和计划/账本/冻结源SHA前后相同。此次不是新的GPU验收，引用的是此前根端已完成的容量v2结果。

原件目录 `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_v2_dispatch_controls_20261003`：`receipt.json`记录首次执行和完整22命令；`stdout.log`保留实际stdout/stderr；`source`归档本轮两个Python原件。

| 原件 | SHA256 |
| --- | --- |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/pool_large_batch_ablation_v2_dispatch.py` | `b088ba3c19f58060a05393411cdf57a3abe8ecd7ba91508add62142896476cc3` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_pool_large_batch_ablation_v2_dispatch.py` | `314bb3a46249ac3bf1c5ca85f8c35ffab4304db9b2ad84ff26d1aca470c63c5c` |
| `/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_large_batch_v2_dispatch_controls_20261003/stdout.log` | `24cf15d0d9065060c2f6f86bf583a9b3978a59662b2daeb0370862ce129a6847` |

下一步由根端独立审阅并决定发布/入队；本项不执行这些动作。停止的注释分支和独立CUDA专项继续暂停。
