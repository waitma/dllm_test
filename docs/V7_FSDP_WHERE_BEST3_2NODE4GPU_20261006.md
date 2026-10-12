# V7 where-only FSDP：best3＋最新完整checkpoint

更新：2026-10-06T12:46:41.042969+00:00。新配置已完成保存策略及源码集成检查，未提交；两节点GPU/NCCL和目标显存验收待实际任务。

提交文件：[v7_f011_where_best3_fsdp_2node4gpu_20261006.yml](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml)。此文件取代本项目后续独立FSDP训练的latest3配方；原文件保持历史版本。

```bash
volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml
```

默认c20250601、非闲时、2×ml.pni2.14xlarge（每节点4卡），global256=32×1×8，总25042步。换队列或规格时保留真实2×4拓扑和共享挂载。目标mem_peak尚未测量，不能把本机DDP实测当作FSDP显存/速度验收。

主eval_loss越小越好；保留best3与最新checkpoint的并集，重合不复制，最多4份。每份都是完整训练状态。先验证八rank原生保存，再归档全部JSON证据并删除本run非保留点；不能追溯恢复旧版本已删除的权重。最终25042步始终作为latest保留。合成任务验收结束后仍会清理本次合成权重，保留小证据。

模型/目标、where表达式、attention后端、FSDP分片、优化器和数据准入不变。正式输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/job_attempt1/train；合成输出 /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/job_attempt1/synthetic。仅允许此run相同源码/数据/配置绑定的FSDP恢复，不接续当前DDP。原合成4→8步恢复、2048长度、真实8rank/19条尾批检查与父数据审核均保留。

本次新增实际检查：10项元数据保留测试（包括FSDP bytes/sha256库存格式）、AST限制差异、618文件绑定、缺失源码拒绝、CLI与YAML/bash语法。未重复训练模型；临时checkpoint全部删除。既有CPU/Gloo模型验证见 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_2NODE4GPU_20261006.md](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_2NODE4GPU_20261006.md)，新集成根证据 [/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/root_review.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_best3_2node4gpu_20261006/root_review.json)。

新manifest SHA be1d78cba6349b4d4ef9d212efc1a1c41578d90e5fd86136b62cb63a443cb9fb；YAML SHA 13b09c34a0bcbffae644f1e52c6cdbea8cab053f5c8db14f2f60986c66727274。原审核范围和安全停线继续生效。真实submit/cancel后仍须更新 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md 的Active表与日期记录。
