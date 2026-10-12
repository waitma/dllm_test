# V7 f011：链偏置性能修复与两节点四卡 FSDP 配置

后续版本（2026-10-06T12:46:41.042969+00:00）：用户要求的best3＋latest已在独立新文件 /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_best3_fsdp_2node4gpu_20261006.yml 实现。新说明 /vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_FSDP_WHERE_BEST3_2NODE4GPU_20261006.md。下方最近3份策略和旧YAML保留为本历史版本，不代表最新配方。

更新：2026-10-06T11:57:12.223404+00:00。CPU数值、保存恢复、两进程Gloo FSDP和CLI检查已通过；新YAML未提交，两节点GPU/NCCL、BF16和显存验收待实际任务执行。

## 本次改动与选择

继续采用f011：R24固定残差缩放0.1＋R26链内局部/全局交替radius32，no-ESMC、8层/d768/H12/FF3072。此次是实现性能修复，不新增架构或监督目标。

原chain_bias通过同/异链索引展开为B×H×S×S，反向把大量重复索引梯度累加回24个参数。只把这一个表达式改成torch.where广播选择；AST检查确认其余denoise逻辑一致。参数名、chain_bias可训练性、mask、context序列CE=0、已有关系正例监督与原目标均保持；未默认启用efficient SDPA。

已有单卡A100、8个相同真实V7批次的完整模型计时为6.6288→0.5450秒/更新，仅where约12.16倍；计时不含CPU组batch、稳态H2D、DDP、评估及保存。一个B32/S867的profile中索引反向占8.8094/9.5329秒。不能将此作为两节点八卡端到端速度承诺。原始证据及本地占卡对照见 [速度根因报告](/vepfs-mlp2/c20250601/251105016/project/dllm_test/docs/V7_SPEED_ROOT_CAUSE_20261006.md)。

## 提交文件

使用 [修复版YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_fsdp_2node4gpu_20261006.yml)，与 [原FSDP YAML](/vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_fsdp_2node4gpu_20261006.yml)、原DDP训练分开。原文件及冻结源码没有改写。

```bash
volc ml_task submit --conf /vepfs-mlp2/c20250601/251105016/project/dllm_test/train_jobs/v7_f011_where_fsdp_2node4gpu_20261006.yml
```

默认ResourceQueueName=c20250601，2×ml.pni2.14xlarge，每节点4GPU，非闲时、不自动重试。若另一机器使用不同队列/规格，修改这两个资源字段，并保证同一绝对共享挂载和Python环境可见。当前目标GPU型号与显存峰值未实测，不填造mem_peak。

| 参数 | 配置 |
|---|---|
| 并行 | FSDP1 FULL_SHARD/use_orig_params，ArchitectureBlock包装 |
| Batch | 每卡32 × GA1 × 8卡 = global256 |
| 日程 | 25042步，LR1e-4，cosine，warmup1000 |
| 精度与重计算 | 原生BF16，FSDP内部非重入activation checkpointing |
| 长度 | A4上限2048，按batch动态padding；目标容量256/64 |
| 评估和保存 | 每1000步及末步，最近3份完整恢复点 |
| 正式输出 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/job_attempt1/train |
| 合成验收输出 | /vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/job_attempt1/synthetic |

micro32是维持global256、world8与整数GA时的最大每卡batch。不能直接改成64而仍称global256。2048是上限，并非所有真实batch填充到2048；启动合成验收会覆盖该上限显存。

## 已完成与待完成

- CPU合成完整模型：有padding/无padding的loss及全部梯度通过，最大梯度差2.2118911147e-9；四mask指标精确一致、尾批3条计数正确。
- 原生CPU Trainer：第2步保存、恢复至4步与连续4步逐参数完全一致；optimizer/scheduler/RNG恢复、模型重新加载通过。
- 两进程CPU/Gloo FSDP：实际FULL_SHARD、内部重计算、平均梯度更新、forward内四mask评估及完整模型/优化器状态往返通过。
- CLI/source guards：615个源码绑定、缺少wrapper或原adapter绑定的拒绝、错误拓扑在创建输出前拒绝、help、YAML解析与bash语法通过。
- 复用已有75.6M模型单卡CUDA梯度证据：一个B32/S516、math后端、BF16 autocast，78个梯度张量检查通过；不将其冒充FSDP或跨节点验收。
- 未执行：两节点GPU/NCCL、目标显存及端到端训练速度。所有自有CPU测试临时checkpoint已删除，只留小日志与JSON。

根验收 [root_review.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/root_review.json)；训练源绑定 [source_manifest_v1.json](/vepfs-mlp2/c20250601/251105016/project/dllm_test/output/v7_fsdp_where_2node4gpu_20261006/source_manifest_v1.json)（SHA c4105f8708c235b0e8c211f9a891c46d74049f815dfdbe57275b299203db3aff）。新入口 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/run_v7_fsdp_where.py；适配器 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/v7_fsdp_where_trainer.py；验证代码 /vepfs-mlp2/c20250601/251105016/project/dllm_test/scripts/research/test_v7_fsdp_where.py。

## 任务启动与恢复边界

同一真实2×4拓扑先执行8步日程的合成训练：第4步保存退出、再从4恢复至8；检查实际rank、原生optimizer/scheduler/RNG、四mask/19条尾批和全长有限loss/显存。通过后保留ADMISSION.json并清理自有合成checkpoint，再执行原正式父数据审核。合成成功不代替数据审核；training_ready、审核证书或范围证书未通过即停止。

本任务独立从头训练。仅支持本run源码/数据/参数/日程绑定的完整FSDP checkpoint恢复，不接续旧DDP或旧FSDP run，不迁移旧optimizer。已有正式权重与恢复状态保留。

新FSDP采用原生BF16/内部重计算，与当前DDP的FP32参数＋autocast/外层重计算存在差异，因此端到端对比不能全部归因于where。入口保留取batch、训练、评估、保存计时及有界整模型trace；嵌套host时间不能相加，trace窗口不作为稳态吞吐。

本轮没有提交、取消、重启任务或恢复任何已停分支。独立CUDA专项继续暂停。原Goal持续；后续提交时须将真实task ID和状态同步到 /vepfs-mlp2/c20250601/251105016/project/dllm_test/PROJECT_PROCESS.md。
